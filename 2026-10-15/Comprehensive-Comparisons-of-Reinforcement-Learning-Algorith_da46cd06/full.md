# Comprehensive Comparisons of Reinforcement Learning Algorithms in Autonomous Driving

Dechen Xie and Mushuang Liu

Abstract—This paper presents a comprehensive comparison of four representative reinforcement learning (RL) methods in autonomous vehicle (AV) decision-making, including: Fitted Value Iteration (FVI), Rollout Classification Policy Iteration (RCPI), Deep Q-Network (DQN), and a discrete-action adaptation of Deep Deterministic Policy Gradient (D-DDPG). The four methods are applied to the same AV decision-making problem at an unsignalized intersection and are compared in terms of optimality and safety, training stability, and training and decision cost. RCPI is the closest to a reference solution and has the most stable training process. However, it is the least data-efficient. D-DDPG reaches the second-smallest gap to a reference solution with the highest data efficiency, but has the least training stability. DQN is the most stable model-free method but has the highest collision rate. FVI achieves a lower collision rate than DQN, but is less data-efficient and less stable. Based on these results, we provide suggestions for selecting a method according to whether a model is available, the budget for training, and whether stable training is required.

Keywords—Autonomous driving, reinforcement learning, decision-making, unsignalized intersection, model-based reinforcement learning, model-free reinforcement learning, comparative study.

## I. INTRODUCTION

UTONOMOUS Driving (AD) can bring benefits to society by improving road safety and enhancing transportation efficiency [1]. However, realizing these benefits requires Autonomous Vehicles (AVs) to make safe and efficient decisions in complex and dynamic traffic environments. Such decision-making remains a critical challenge [2].

Reinforcement Learning (RL) provides a mathematical framework for sequential decision-making, in which a policy is learned to maximize the cumulative reward over time [3]. Since each driving decision affects the future traffic situation, RL has been widely explored for AV decision-making [4]. Most RL methods can be distinguished along two dimensions: whether a method uses a model of the environment and whether a method represents the policy explicitly.

A model of the environment predicts the resulting next state and reward from taking an action at a given state [3]. Modelbased methods can query the model at any state repeatedly to predict the results of taking an action without executing it. However, in AD, an exact model is not always available. In practice, the exact model is replaced by an approximate one which is either designed from prior knowledge or learned from collected data [5]. For example, Approximate Dynamic Programming (ADP) can be applied to solve path-tracking control problems with a designed environment model [6]. Think2Drive learns a model of the driving environment from collected data and uses it to train a driving policy [7]. In contrast, model-free methods learn without a model. Instead, they execute an action in the environment and observe the resulting next state and reward. This procedure is closer to the way vehicles act in the real world. Deep Deterministic Policy Gradient (DDPG) has been applied to learn lane following on a real vehicle [8]. During training, the vehicle drove along the road under the current policy until the safety driver intervened. Every time step was recorded and used to update the learned policy. Deep Q-Network (DQN) has been used to learn how to cross unsignalized intersections in a traffic simulator [9].

These methods can also be distinguished by whether they represent the policy explicitly. Value-based methods learn the value of states or actions and derive the policy from it. For example, DQN learns the value of each action and selects the action with the largest value [9]. Policy-based methods instead represent the policy explicitly. The ADP used in [6] follows a policy iteration framework and learns a policy network. Actorcritic methods use an actor to learn a policy and a critic to learn the value of states or actions. Some studies treat actorcritic methods as a third category that combines value-based and policy-based methods [10]. Since the critic is only used to help the actor learn and actions are selected by the actor alone, we classify actor-critic methods as policy-based in this study. The DDPG agent for lane following described above is such an actor-critic method [8].

Although many RL methods have been developed and implemented, they are rarely compared under identical conditions, and their respective strengths and limitations for AV decision-making remain unclear. A deeper understanding of these methods would benefit the selection and design of RL methods. This paper presents a comprehensive comparison of four RL methods, one for each combination of the two dimensions: Fitted Value Iteration (FVI), Rollout Classification Policy Iteration (RCPI), DQN, and a discrete-action adaptation of DDPG, denoted D-DDPG. FVI and RCPI are model-based, whereas DQN and D-DDPG are model-free; FVI and DQN are value-based, whereas RCPI and D-DDPG are policy-based, as summarized in Table I. The four methods are applied to the same decision-making problem at an unsignalized intersection and compared in terms of optimality and safety, training stability, and training and decision cost. Based on the results and features of each method, we provide suggestions for method selection.

The remainder of this paper is organized as follows. Section II formulates the problem as a Deterministic Markov

TABLE I  
CLASSIFICATION OF THE FOUR RL METHODS COMPARED IN THIS PAPER ALONG THE TWO DIMENSIONS.
<table><tr><td></td><td>Value-based</td><td>Policy-based</td></tr><tr><td>Model-based</td><td>FVI [11]</td><td>RCPI [12]</td></tr><tr><td>Model-free</td><td>DQN [13]</td><td>D-DDPG [14], [15]</td></tr></table>

Decision Process (DMDP). Section III introduces the preliminaries of RL. Section IV presents the four methods and how each of them computes its training target. Section V presents the numerical results and suggestions for method selection. Section VI concludes the paper.

## II. PROBLEM FORMULATION

In this section, we formulate the AV decision-making problem as a DMDP.

Consider a general traffic scenario with N vehicles, where each vehicle is indexed by $i \in \{ 1 , 2 , \ldots , N \}$ . Let vehicle 1 be the controlled vehicle and vehicles $2 , \ldots , N$ be the surrounding vehicles. The state of vehicle i is denoted by $\xi _ { i } \in \mathbb { R } ^ { n }$ , where n is the dimension of a single-vehicle state. In autonomous driving, the state of a vehicle often contains its position, velocity, and heading [16]. The joint state is defined as $\begin{array} { r } { s = ( \xi _ { 1 } ^ { \top } , \ldots , \xi _ { N } ^ { \top } ) ^ { \top } \in \mathcal { S } . } \end{array}$ where ${ \mathcal { S } } \subseteq \mathbb { R } ^ { N n }$ is the joint state space. The controlled vehicle follows a deterministic policy $\pi : { \mathcal { S } }  A$ , where $\mathcal { A }$ is the action space and $| { \cal A } |$ denotes the number of actions in ${ \mathcal { A } } .$ . In autonomous driving, the action often contains acceleration and steering angle [16]. In this work, the state space $s$ is continuous and the action space $\mathcal { A }$ is finite. Given a state s, the controlled vehicle selects an action $a = \pi ( s ) \in { \mathcal { A } } .$ The next state is determined by $s ^ { \prime } = f ( s , a )$ where $f : \mathcal { S } \times \mathcal { A } \to \mathcal { S }$ is the deterministic state-transition function. The controlled vehicle then receives an immediate reward $r ( s , a ) \in \mathbb { R }$ , where $r : \mathcal { S } \times \mathcal { A } \to \mathbb { R }$ is the reward function.

The decision-making process proceeds over discrete time steps $t ~ \in ~ \{ 0 , 1 , \ldots \}$ . At each time step t, the controlled vehicle observes the current state $s _ { t } ,$ selects an action $a _ { t } ~ = ~ \pi ( s _ { t } )$ , transitions to the next state $s _ { t + 1 } = f ( s _ { t } , a _ { t } )$ and receives the reward $r ( s _ { t } , a _ { t } )$ . The observed sequence $\{ s _ { 0 } , a _ { 0 } , r ( s _ { 0 } , a _ { 0 } ) , s _ { 1 } , a _ { 1 } , . . . \}$ is called a trajectory. The statevalue function $V ^ { \pi } ( s )$ is the cumulative discounted reward collected along the trajectory that starts at s and follows π,

$$
V ^ { \pi } ( s ) = \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r ( s _ { t } , a _ { t } ) , \quad s _ { 0 } = s ,\tag{1}
$$

where $\gamma \in [ 0 , 1 )$ is the discount factor. The objective is to find an optimal policy $\pi ^ { * }$ that maximizes $V ^ { \pi } ( s )$ at every state,

$$
\pi ^ { * } = \arg \operatorname* { m a x } _ { \pi } V ^ { \pi } ( s ) , \qquad \forall s \in { \mathcal { S } } .\tag{2}
$$

Since $s _ { t + 1 }$ is determined by $s _ { t }$ and $a _ { t }$ alone, the Markov property holds. The problem can be modeled as a DMDP represented by the tuple

$$
\begin{array} { r } { \ M = ( S , A , f , r , \gamma ) . } \end{array}\tag{3}
$$

## III. PRELIMINARIES OF REINFORCEMENT LEARNING

Solving (2) requires evaluating the cumulative reward of candidate policies, which depends on the transition function $f$ and the reward function r. An exact model is not always available in AD, since the driving policies of the surrounding vehicles are unknown. Instead of solving (2) with $f$ and r, RL starts from an estimate of the value function or the policy and corrects it with the reward and next state that result from taking an action in a state. This information can be obtained by querying either an exact or approximate model or by executing the action in the environment and observing the outcome. This section presents a brief introduction to RL.

To improve the estimate of the value function or the policy, an RL method needs to know how good it is to take an action at a state. RL measures this by the action-value function. For a policy π and a state-action pair $( s , a )$ , the action-value function $Q ^ { \pi } : S \times \mathcal { A }  \mathbb { R }$ is defined as

$$
\begin{array} { l } { { \displaystyle Q ^ { \pi } ( s , a ) = \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r ( s _ { t } , a _ { t } ) } \ ~ } \\ { { \displaystyle ~ = r ( s , a ) + \gamma \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r ( s _ { t + 1 } , a _ { t + 1 } ) } \ ~ } \\ { { \displaystyle ~ = r ( s , a ) + \gamma Q ^ { \pi } \big ( f ( s , a ) , \pi ( f ( s , a ) ) \big ) , } } \end{array}\tag{4}
$$

(5)

where $s _ { t + 1 } = f ( s _ { t } , a _ { t } ) , ( s _ { 0 } , a _ { 0 } ) = ( s , a )$ and $\boldsymbol { a } _ { t } = \pi ( \boldsymbol { s } _ { t } )$ for $t \geq 1$ . Similarly, $V ^ { \pi } ( s )$ satisfies

$$
V ^ { \pi } ( s ) = r ( s , \pi ( s ) ) + \gamma V ^ { \pi } \bigl ( f ( s , \pi ( s ) ) \bigr ) .\tag{6}
$$

Most policy-based methods follow two steps: policy evaluation and policy improvement. In policy evaluation, the current policy is evaluated by $V ^ { \pi }$ or $Q ^ { \pi }$ . Policy improvement is conducted based on this evaluation by updating the policy toward actions with higher $V ^ { \pi }$ or $Q ^ { \pi }$

The optimal value functions $V ^ { * } ( s ) \ = \ \operatorname* { m a x } _ { \pi } V ^ { \pi } ( s )$ and $Q ^ { * } ( s , a ) \ = \ \operatorname * { m a x } _ { \pi } Q ^ { \pi } ( s , a )$ satisfy the Bellman optimality equations [3]

$$
V ^ { * } ( s ) = \operatorname* { m a x } _ { a \in \mathcal { A } } \bigl [ r ( s , a ) + \gamma V ^ { * } \bigl ( f ( s , a ) \bigr ) \bigr ] ,\tag{7}
$$

$$
Q ^ { * } ( s , a ) = r ( s , a ) + \gamma \operatorname* { m a x } _ { a ^ { \prime } \in \mathcal { A } } Q ^ { * } \big ( f ( s , a ) , a ^ { \prime } \big ) .\tag{8}
$$

An optimal policy can be obtained from either function,

$$
\begin{array} { c l c r } { { \pi ^ { * } ( s ) = \displaystyle \arg \operatorname* { m a x } _ { a \in \mathcal { A } } \bigl [ r ( s , a ) + \gamma V ^ { * } \bigl ( f ( s , a ) \bigr ) \bigr ] } } \\ { { \mathrm { } } } \\ { { \mathrm { } } } \end{array}\tag{9}
$$

Value-based methods learn an estimate of the optimal value function, $V ^ { * }$ or $Q ^ { * }$ , and derive the policy from it by (9).

Both value-based and policy-based methods start from an initial estimate of the value function or the policy and improve it iteratively. Each update corrects the estimates with the actual reward and next state that result from taking an action at a state, which enter the update through the right-hand side of (5), (6), (7), or (8). We call this information an experience and represent it by the tuple $( s , a , r , s ^ { \prime } )$ , where $r = r ( s , a )$ and $s ^ { \prime } = f ( s , a )$

An experience can be obtained in two ways. Given a model, the experience of any action at any state can be obtained by querying the model. Without a model, an experience is collected by executing an action in the environment, and only the outcome of the executed action is observed. Methods that rely on the first way are model-based, and methods that only need the second way are model-free [3]. Since f is deterministic, querying $f$ and r at $( s , a )$ yields exactly the experience that executing a at s would produce. The data used by both kinds of methods can therefore be compared by the number of experiences.

## IV. REINFORCEMENT LEARNING METHODS

In this section, we introduce the four RL methods compared in this paper in the order of FVI, DQN, RCPI, and D-DDPG. All four methods use neural networks as function approximators, so we first state the notation for them and then introduce each method in detail.

Throughout this paper, a function approximated by a neural network is written with its parameter as a subscript, e.g., $V _ { \omega }$ denotes a network with parameter ω that represents a statevalue function V, and $V _ { \omega } ( s )$ denotes its output at state s. The same convention is used for every network introduced below. Each network has its own parameter: ω for FVI, ψ for DQN, η for RCPI, and θ and ϕ for the actor and the critic of D-DDPG.

## A. FVI

FVI [11] is a value-based and model-based method that learns an estimate of $V ^ { * }$ . It extends Value Iteration [3] by using a Neural Network (NN) $V _ { \omega }$ to represent the state-value function. It maintains two networks: the online network $V _ { \omega } ,$ which changes at every iteration, and a target network $V _ { \omega ^ { - } }$ which is used to compute the targets. The two networks share the same architecture. $\omega ^ { - }$ is a delayed copy of ω which is updated by $\omega ^ { - }  \omega$ every M iterations. At iteration k (for $k = 1 , \ldots , K )$ , FVI samples a set of states $\mathcal { D } \subset \mathcal { S }$ from a replay buffer R. For each state $s \in \mathcal { D }$ , it collects the experiences $( s , a , r , s ^ { \prime } )$ of all actions $a \in { \mathcal { A } }$ by querying the model. The value network is updated by minimizing the loss

$$
\mathcal { L } _ { V } ^ { ( k ) } ( \omega ) = \frac { 1 } { | \mathcal { D } | } \sum _ { s \in \mathcal { D } } \bigl [ V _ { \omega } ( s ) - V _ { \mathrm { t a r g e t } } ^ { k } ( s ) \bigr ] ^ { 2 } ,\tag{10}
$$

yielding $\omega _ { k } ,$ , where $V _ { \mathrm { t a r g e t } } ^ { k } ( s )$ is the target at iteration k defined as

$$
V _ { \mathrm { t a r g e t } } ^ { k } ( s ) = \operatorname* { m a x } _ { a \in \cal { A } } \bigl [ r ( s , a ) + \gamma V _ { \omega ^ { - } } \bigl ( f ( s , a ) \bigr ) \bigr ] .\tag{11}
$$

After K iterations, the learned policy is derived as

$$
\pi ( s ) = \arg \operatorname* { m a x } _ { a \in \mathcal { A } } \ [ r ( s , a ) + \gamma V _ { \omega _ { K } } \left( f ( s , a ) \right) ] .\tag{12}
$$

Computing the target (11) requires solving an optimization problem over a at the state s. The objective $r ( s , a ) +$ $\gamma V _ { \omega ^ { - } } ( f ( s , a ) )$ has to be evaluated over the action space, which requires knowing how $r ( s , a )$ and $f ( s , a )$ vary with a. FVI is therefore a model-based method. In this study, A is finite, so the maximum can be obtained by enumerating every $a \in A .$ . Each evaluation requires one experience, so FVI requires |A| experiences to compute each target. The policy (12) also takes the maximum over all actions at the current state, so FVI requires the model not only in training but also at every decision.

```perl
Algorithm 1: Fitted Value Iteration (FVI)
Require: access to the model of the environment,
discount factor $\gamma ,$ target update period M
1 Initialize the online network $V _ { \omega _ { 0 } }$ and the target
network $V _ { \omega }$ − with $\omega ^ { - }  \omega _ { 0 } ;$
2 Initialize the replay buffer R;
3 for iteration $k = 1 , \ldots , K$ do
4 Sample a set of states D from $\mathcal { R } ;$
5 Collect the experiences $( s , a , r , s ^ { \prime } )$ of all actions
a $\in { \mathcal { A } }$ for each $s \in \mathcal { D }$ by querying the model;
6 Compute the target $V _ { \mathrm { t a r g e t } } ^ { k } ( s )$ for each $s \in \mathcal { D }$
according to (11);
7 Update the online network by minimizing the
loss (10), yielding ω<sub>k</sub>;
8 Every M iterations, set $\omega ^ { - }  \omega _ { k } ;$
9 return the value network $V _ { \omega _ { K } } ;$
```

The FVI training procedure is summarized in Algorithm 1.

## B. DQN

DQN [13] is a value-based and model-free method. It extends Q-learning [3] with function approximation. Instead of learning the value of a state, DQN uses an NN $Q _ { \psi }$ to represent the action-value function, which outputs $Q _ { \psi } ( s , a )$ for every $a \in { \mathcal { A } }$ at a given state s. DQN also maintains an online network $Q _ { \psi }$ and a target network $Q _ { \psi ^ { - } }$ , where $\psi ^ { - }$ is updated by $\psi ^ { - }  \psi$ every M iterations. Experiences are collected by interacting with the environment under an ϵ-greedy policy and stored in a replay buffer R. At state s, a random action is selected with a probability of ϵ, and otherwise, the action arg ma $\mathfrak { c } _ { a \in \mathcal { A } } Q _ { \psi } ( s , a )$ is selected. At iteration k (for $k = 1 , \ldots , K )$ , DQN executes one action in the environment, stores the resulting experience in ${ \mathcal { R } } ,$ and samples a batch of experiences B from R. The online network is updated by minimizing the loss

$$
\mathcal { L } _ { Q } ( \psi ) = \frac { 1 } { | \mathcal { B } | } \sum _ { ( s , a , r , s ^ { \prime } ) \in \mathcal { B } } \big [ Q _ { \psi } ( s , a ) - Q _ { \mathrm { t a r g e t } } ( s , a ) \big ] ^ { 2 } ,\tag{13}
$$

yielding $\psi _ { k }$ , where $Q _ { \mathrm { t a r g e t } } ( s , a )$ is the target defined as

$$
Q _ { \mathrm { t a r g e t } } ( s , a ) = r + \gamma \operatorname* { m a x } _ { a ^ { \prime } \in \mathcal { A } } Q _ { \psi ^ { - } } ( s ^ { \prime } , a ^ { \prime } ) .\tag{14}
$$

After K iterations, the learned policy is derived as

$$
\pi ( s ) = \arg \operatorname* { m a x } _ { a \in \mathcal { A } } Q _ { \psi _ { K } } ( s , a ) .\tag{15}
$$

Unlike (11), the maximum in (14) is taken over the actions at the next state $s ^ { \prime } .$ The action-values $Q _ { \psi ^ { - } } \left( s ^ { \prime } , a ^ { \prime } \right)$ for every $a ^ { \prime } \in { \mathcal { A } }$ can be provided by the target network itself. Computing the target therefore only requires one experience $( s , a , r , s ^ { \prime } )$ , which can be collected by executing a at s in the environment. DQN is thus a model-free method.

The DQN training procedure is summarized in Algorithm 2.

Algorithm 2: Deep Q-Network (DQN) Algorithm 3: Rollout Classification Policy Iteration   
Require: access to the environment for interaction, (RCPI)   
discount factor $\gamma ,$ exploration rate $\epsilon ,$ target Require: access to the model of the environment,   
update period M discount factor $\gamma$   
1 Initialize the online network $Q _ { \psi _ { 0 } }$ and the target 1 Initialize the policy network $\pi _ { \eta _ { 0 } }$ and the replay buffer   
network $Q _ { \psi ^ { - } }$ with $\psi ^ { - }  \psi _ { 0 } ;$ $\mathcal { R } ;$   
2 Initialize the replay buffer $\mathcal { R } ;$ 2 for iteration $k = 1 , \ldots , K$ do   
3 for iteration $k = 1 , \ldots , K$ do 3 Sample a set of states D from $\mathcal { R } ;$   
4 Observe $s ,$ select a by the ϵ-greedy policy with 4 Fix the current policy $\pi _ { k } = \pi _ { \eta _ { k - 1 } } ;$   
$Q _ { \psi _ { k - 1 } }$ , execute it, and store $( s , a , r , s ^ { \prime } )$ in $\mathcal { R } ;$ 5 For each $s \in \mathcal { D }$ and $a \in A ,$ collect the experiences   
5 Sample a batch of experiences B from $\mathcal { R } ;$ along the trajectory that starts from s with action   
6 Compute the target $Q _ { \mathrm { t a r g e t } } ( s , a )$ for each a and then follows $\pi _ { k }$ by querying the model;   
experience in B according to (14); 6 Compute $Q ^ { \pi _ { k } } ( s , a )$ according to (4);   
7 Update the online network by minimizing the 7 Compute the target action $a _ { \mathrm { t a r g e t } } ( s )$ for each   
loss (13), yielding ψ<sub>k</sub>; $s \in \mathcal { D }$ according to (17);   
8 Every M iterations, set $\psi ^ { - }  \psi _ { k } ;$ 8 Update the policy network by minimizing the   
9 return the action-value network $Q _ { \psi _ { K } } ;$ loss (16), yielding $\eta _ { k }$   
9 return the policy network $\pi _ { \eta _ { K } } ;$

## C. RCPI

RCPI [12] is a policy-based and model-based method that learns an estimate of $\pi ^ { * }$ . It extends Policy Iteration [3] by using an NN $\pi _ { \eta }$ to represent the policy. Given a state s, the policy network outputs a vector $\pi _ { \eta } ( s ) \in \mathbb { R } ^ { | A | }$ , whose entry $\pi _ { \eta } ( s , a )$ is the score of action $a \in A .$ . The policy selects the action with the highest score, π $\mathbf { \partial } \cdot ( s ) = \operatorname { a r g }$ $\mathrm { m a x } _ { a \in A } \pi _ { \eta } ( s , a )$ . At iteration k (for $k = 1 , \ldots , K )$ , RCPI samples a set of states $\mathcal { D } \subset \mathcal { S }$ from a replay buffer R and fixes the current policy as $\pi _ { k } = \pi _ { \eta _ { k - 1 } }$ . The policy network is updated to predict the target action $a _ { \mathrm { t a r g e t } } ( s )$ by minimizing the loss

$$
\mathcal { L } _ { \pi } ^ { ( k ) } ( \eta ) = - \frac { 1 } { | \mathcal { D } | } \sum _ { s \in \mathcal { D } } \log \frac { \exp \bigl ( \pi _ { \eta } ( s , a _ { \mathrm { t a r g e t } } ( s ) ) \bigr ) } { \sum _ { a \in \mathcal { A } } \exp \bigl ( \pi _ { \eta } ( s , a ) \bigr ) } ,\tag{16}
$$

yielding $\eta _ { k }$ , where the target action is defined as

$$
a _ { \mathrm { t a r g e t } } ( s ) = \arg \operatorname* { m a x } _ { a \in \mathcal { A } } Q ^ { \pi _ { k } } ( s , a ) .\tag{17}
$$

RCPI computes $Q ^ { \pi _ { k } } ( s , a )$ according to (4). For each action a at state $s ,$ it collects the experiences along the trajectory that starts from s with action a and then follows $\pi _ { k }$ by querying the model. RCPI is therefore a model-based method. In this study, A is finite, so RCPI simulates one trajectory for every $a \in A .$ . Let H be the number of time steps of a trajectory. Each step requires one experience, so RCPI requires $| { \mathcal { A } } | H$ experiences to compute each target. After K iterations, the learned policy is $\pi ( s ) = \arg \operatorname* { m a x } _ { a \in \mathcal { A } } \pi _ { \eta _ { K } } ( s , a )$ . The learned policy is obtained from the network alone, so RCPI does not need the model at deployment.

The RCPI training procedure is summarized in Algorithm 3.

## D. D-DDPG

D-DDPG is a policy-based and model-free method. It is derived from DDPG [14] by adapting it to a discrete action space. It follows the actor-critic structure. Like RCPI, the actor uses a policy network $\pi _ { \theta }$ to represent the policy, which outputs a score vector $\pi _ { \boldsymbol { \theta } } ( s )$ whose entry $\pi _ { \boldsymbol { \theta } } ( s , a )$ is the score of $a \in$ A. Instead of calculating $Q ^ { \pi } ( s , a )$ from trajectories, D-DDPG introduces a critic network $Q _ { \phi }$ to learn $Q ^ { \pi } ( s , a )$

Standard DDPG is designed for continuous action spaces. The actor is updated through the gradient of the critic, which requires the selected action to be differentiable with respect to the actor parameters. This differentiability does not hold in a discrete action space. To address this, we adopt the straightthrough Gumbel-Softmax (STGS) estimator [15]. STGS passes the gradient to the actor through a softmax vector over the action scores, so the critic has to take an action input that lies in the same space as this vector. We therefore encode each action $a \in { \mathcal { A } }$ as a one-hot vector $\mathbf { e } _ { a } \in \{ 0 , 1 \} ^ { | \mathcal { A } | }$ , whose entry corresponding to a is 1 and all other entries are 0. Every action is passed to the critic in this form. During training, independent Gumbel noise $g _ { a } \sim \mathrm { G u m b e l } ( 0 , 1 )$ is added to each score. The action with the largest perturbed score,

$$
a ^ { \star } = \arg \operatorname* { m a x } _ { a \in \mathcal { A } } \bigl [ \pi _ { \theta } ( s , a ) + g _ { a } \bigr ] ,\tag{18}
$$

is selected and passed to the critic as $\mathbf { e } _ { a ^ { \star } }$ . In the backward pass, the gradient of ${ \bf e } _ { a ^ { \star } }$ ⋆ is replaced by that of the tempered softmax softmax $( ( \pi _ { \theta } ( s ) + g ) / \tau _ { \mathrm { G S } } )$ , where $g$ is the vector of the noise $g _ { a }$ and $\tau _ { \mathrm { G S } } > 0$ is the temperature. Therefore the gradient of the critic output with respect to θ can be propagated back to the actor.

D-DDPG maintains an online actor $\pi _ { \theta }$ and an online critic $Q _ { \phi } ,$ , together with a target actor $\pi _ { \theta } .$ − and a target critic $Q _ { \phi } .$ −. At each iteration, the target parameters are moved toward the online parameters by a soft update with coefficient τ , which stabilizes training [14]. At iteration k (for $k = 1 , \ldots , K )$ , D-DDPG selects one action by (18), executes it in the environment, stores the resulting experience in a replay buffer R, and samples a batch of experiences B from R. The online critic is updated by minimizing the loss

$$
\mathcal { L } _ { \mathrm { c r i t i c } } ( \phi ) = \frac { 1 } { | \mathcal { B } | } \sum _ { ( s , a , r , s ^ { \prime } ) \in \mathcal { B } } \bigl [ Q _ { \phi } ( s , a ) - Q _ { \mathrm { t a r g e t } } ( s , a ) \bigr ] ^ { 2 } ,\tag{19}
$$

Algorithm 4: Discrete DDPG (D-DDPG)   
Require: access to the environment for interaction,   
discount factor $\gamma ,$ temperature τ<sub>GS</sub>,   
soft-update coefficient $\tau _ { \mathrm { t a r g e t } }$   
1 Initialize the online actor $\pi _ { \theta _ { 0 } }$ and the online critic $Q _ { \phi _ { 0 } } ;$   
2 Initialize the target actor $\pi _ { \theta } .$ − and the target critic $Q _ { \phi }$   
with $\theta ^ { - }  \theta _ { 0 }$ and $\phi ^ { - }  \phi _ { 0 } ;$   
3 Initialize the replay buffer $\mathcal { R } ;$   
4 for iteration $k = 1 , \ldots , K$ do   
5 Observe s, select $a ^ { \star }$ by (18) with $\pi _ { \theta _ { k } }$ , execute it,   
and store $( s , a ^ { \star } , r , s ^ { \prime } )$ in R;   
6 Sample a batch of experiences B from $\mathcal { R } \mathrm { : }$   
7 Compute the target $Q _ { \mathrm { t a r g e t } } ( s , a )$ for each   
experience in $\boldsymbol { B }$ according to (20);   
8 Update the online critic by minimizing the   
loss (19), yielding $\phi _ { k } ;$   
9 Update the online actor by minimizing the   
loss (21), yielding $\theta _ { k } ;$   
10 $\theta ^ { - }  ( 1 - \tau _ { \mathrm { t a r g e t } } ) \theta ^ { - } + \tau _ { \mathrm { t a r g e t } } \theta _ { k } ;$   
11 $\phi ^ { - }  ( 1 - \tau _ { \mathrm { t a r g e t } } ) \phi ^ { - } + \tau _ { \mathrm { t a r g e t } } \phi _ { k } ;$   
12 return the policy network $\pi _ { \boldsymbol { \theta } _ { K } } ;$

yielding $\phi _ { k }$ , where $Q _ { \mathrm { t a r g e t } } ( s , a )$ is the target defined as

$$
Q _ { \mathrm { t a r g e t } } ( s , a ) = r + \gamma Q _ { \phi ^ { - } } ( s ^ { \prime } , a ^ { \pi } ) ,\tag{20}
$$

and $a ^ { \pi }$ is the action given by (18) at $s ^ { \prime }$ with the target actor $\pi _ { \theta ^ { - } }$

The online actor is then updated by minimizing the loss

$$
\mathcal { L } _ { \mathrm { a c t o r } } ( \theta ) = - \frac { 1 } { | \mathcal { B } | } \sum _ { s \in \mathcal { B } } Q _ { \phi } \big ( s , a ^ { \star } \big ) ,\tag{21}
$$

yielding $\theta _ { k } ,$ , where $a ^ { \star }$ is given by the online actor $\pi _ { \theta }$ . After K iterations, the learned policy is derived as

$$
\pi ( s ) = \arg \operatorname* { m a x } _ { a \in \mathcal { A } } \pi _ { \theta _ { K } } ( s , a ) .\tag{22}
$$

In (20), the action at $s ^ { \prime }$ is given by the target actor and its value by the target critic. Computing the target therefore only requires one experience $( s , a , r , s ^ { \prime } )$ , which can be collected by executing a at s in the environment. D-DDPG is thus a model-free method and requires one experience to compute each target. The policy (22) is obtained from the actor alone, so D-DDPG does not need the model at deployment.

The D-DDPG training procedure is summarized in Algorithm 4.

## V. NUMERICAL RESULTS

To compare the four RL methods, we test them on a simulated unsignalized intersection scenario. Each method is evaluated in three aspects: optimality and safety, training stability, and training and decision cost. Optimality is measured by the gap in discounted reward between the learned policy and a reference solution. Safety is measured by the collision rate of the learned policy. Training stability is presented by the learning curve and is measured by the drop from the peak discounted reward to the discounted reward at the end of training. Training cost is measured by the number of experiences required to reach given levels of discounted reward. Decision cost is measured by the time the learned policy takes to make a decision at one state.

![](images/70a017eabfa7bde5d3a981b26ea96250a8859b1ac47eec1ef45c3ed93f285ef5.jpg)  
Fig. 1. Illustration of the unsignalized intersection.

In the following subsections, we first introduce the simulation setup in Section V-A, then compare the four methods in the three aspects in Section V-B, and finally provide suggestions for method selection in Section V-C.

## A. Simulation Setup

1) Environment Setup: We consider an unsignalized intersection environment with $N \ = \ 5$ vehicles as illustrated in Fig. 1. Vehicle 1 is the controlled vehicle and vehicles $2 , \ldots , 5$ are the surrounding vehicles that are moving at a constant speed. All vehicles travel straight through the intersection. The goal is to find the optimal policy (2) of the controlled vehicle, which maximizes the discounted reward (1) with discount factor $\gamma = 0 . 9 9$

The intersection is described in a fixed Cartesian frame with its origin at the intersection center. The x-axis points east and the y-axis points north. The state of vehicle i is

$$
\xi _ { i } = [ x _ { i } , y _ { i } , v _ { x , i } , v _ { y , i } ] ^ { \top } ,\tag{23}
$$

where $( x _ { i } , y _ { i } )$ is its position and $( v _ { x , i } , v _ { y , i } )$ is its velocity. The joint state defined in Section II is therefore

$$
\begin{array} { r } { s = ( \xi _ { 1 } ^ { \top } , \xi _ { 2 } ^ { \top } , \ldots , \xi _ { 5 } ^ { \top } ) ^ { \top } \in \mathbb { R } ^ { 2 0 } . } \end{array}\tag{24}
$$

The environment proceeds with a time step $\Delta t = 0 . 2 5 { \mathrm { ~ s . ~ A ~ } }$ simulation terminates when the controlled vehicle passes the intersection and reaches $y _ { 1 } > 3 0 \mathrm { ~ m ~ }$ , when a collision occurs, or after 80 time steps (20 s), whichever occurs first. A collision is detected when $d _ { i } < 0 . 5$ m for any $i \in \{ 2 , \ldots , N \}$ , where $d _ { i }$ is the Euclidean distance between the controlled vehicle and surrounding vehicle i,

$$
d _ { i } = { \sqrt { ( x _ { 1 } - x _ { i } ) ^ { 2 } + ( y _ { 1 } - y _ { i } ) ^ { 2 } } } .\tag{25}
$$

Each vehicle is modeled as a point mass. The controlled vehicle travels along the y-axis in the positive direction. Its action a is its longitudinal acceleration. The finite action space of the controlled vehicle is

$$
\mathcal { A } = \{ - 4 , - 3 , . . . , 3 , 4 \} \ \mathrm { m / s ^ { 2 } } .\tag{26}
$$

There are a total of nine actions in $\mathcal { A } \thinspace \mathrm { s o } \ \lvert \mathcal { A } \rvert = 9$ in this work. The transition function introduced in Section II is therefore

$$
\boldsymbol { f } ( \boldsymbol { s } , \boldsymbol { a } ) = \big ( ( A \xi _ { 1 } + B \boldsymbol { a } ) ^ { \top } , ( A \xi _ { 2 } ) ^ { \top } , \ldots , ( A \xi _ { N } ) ^ { \top } \big ) ^ { \top } ,\tag{27}
$$

with

$$
A = \left[ { \begin{array} { c c c c } { 1 } & { 0 } & { \Delta t } & { 0 } \\ { 0 } & { 1 } & { 0 } & { \Delta t } \\ { 0 } & { 0 } & { 1 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 1 } \end{array} } \right] , \qquad B = \left[ { \begin{array} { c } { 0 } \\ { 0 } \\ { 0 } \\ { \Delta t } \end{array} } \right] .\tag{28}
$$

Following the reward design in [17], the reward function $r$ is defined as

$$
r ( s , a ) = - \alpha \left( \frac { v _ { y , 1 } ^ { \prime } - v _ { \mathrm { r e f } } } { v _ { \mathrm { r e f } } } \right) ^ { 2 } - \beta \sum _ { i = 2 } ^ { N } \frac { 1 } { ( d _ { i } ^ { \prime } ) ^ { 2 } + \delta } ,\tag{29}
$$

where $v _ { y , 1 } ^ { \prime }$ and $d _ { i } ^ { \prime }$ are the longitudinal velocity of the controlled vehicle and its distance to surrounding vehicle i at the next state $s ^ { \prime } = f ( s , a )$ . Here $v _ { \mathrm { r e f } } ~ = ~ 5$ m/s is the desired speed, $\delta \ = \ 0 . 0 1$ prevents division by zero, and $\alpha ~ = ~ 6$ and $\beta = 1 0 0$ are weighting coefficients. The first term of r penalizes deviation from the desired speed, while the second term penalizes proximity to surrounding vehicles.

2) Training Setup: In practice, different layouts of surrounding vehicles may occur within the same scenario. A method therefore needs to learn a general policy that works across multiple layouts. In this study, we design four layouts of the unsignalized intersection scenario, denoted C1–C4, as illustrated in Fig. 2. Each method learns a single policy that is trained on all four layouts. We characterize the complexity of a layout by the number of conflicting vehicles, i.e., surrounding vehicles whose future paths cross the path of the controlled vehicle. The positions and travel directions of the surrounding vehicles are assigned so that C1, C2, C3, and C4 contain 1, 2, 3, and 4 conflicting vehicles, respectively. Every additional conflicting vehicle decreases the number of feasible safe decisions of the controlled vehicle. The complexity of the layout thus increases from C1 to C4.

All four methods sample their training data from a replay buffer R. The replay buffer of model-based methods FVI and RCPI stores states only. To cover the states that the controlled vehicle can reach from the initial states, these states are taken from trajectories that start from an initial state sampled from the four layouts and select the action at every step uniformly from A until termination. Every state on these trajectories is stored in R. Model-free methods DQN and D-DDPG cannot query the model, so R stores experiences $( s , a , r , s ^ { \prime } )$ collected by executing actions in the environment. The experiences are collected by an ϵ-greedy policy for DQN and by (18) for D-DDPG. In all four methods, R is first-in-first-out, and $| \mathcal { D } | =$ 64 states or |B| = 64 experiences are sampled uniformly from it at each iteration.

All four methods use NNs as function approximators. As shown in Fig. 3, all networks share the same backbone of two fully connected hidden layers with 256 units each and ReLU activations with same initialization. The networks differ only in their input and output layers, which depend on the method. For FVI, DQN, RCPI, and the actor of D-DDPG, the input is the state s. The value network of FVI outputs the scalar $V _ { \omega } ( s )$

the Q-network of DQN outputs $Q _ { \psi } ( s , a )$ for every $a \in { \mathcal { A } } ,$ and the policy network of RCPI and the actor of D-DDPG output the scores $\pi _ { \eta } ( s , a )$ and $\pi _ { \boldsymbol { \theta } } ( s , a )$ , respectively, for every $a \in A .$ The critic of D-DDPG takes s and an action a, encoded as a one-hot vector, as input and outputs the scalar $Q _ { \phi } ( s , a )$

We measure training progress by the number of network updates. For FVI, RCPI, and DQN, one network update is one update of the parameters of $V _ { \omega } , \pi _ { \eta } ,$ and $Q _ { \psi }$ , respectively. For D-DDPG, one network update includes one update of the critic $Q _ { \phi }$ and one update of the actor $\pi _ { \boldsymbol { \theta } } .$ . The number of network updates measures how much each method has learned. Each method is trained for $2 . 5 \times 1 0 ^ { 6 }$ network updates with five random seeds. This budget is chosen to make sure that every method is trained sufficiently.

3) Evaluation Setup: To obtain the training outcome of each method, the policies at all recorded checkpoints are first evaluated on a validation set of 500 initial states that are sampled equally from each of the four layouts. For each method and each seed, the checkpoint with the highest mean discounted reward over the validation set is considered as the peak and selected as the learned policy. The learned policies are then evaluated on a separate test set of 4000 initial states, sampled in the same way as the validation set.

To better understand the optimality of each method, we obtain a reference solution at each initial state by searching over action sequences with Differential Evolution (DE). DE maximizes the discounted reward of an 80-step action sequence over the same action space. It uses a population of 8,000 and 500 generations. Since DE is a heuristic optimizer, the reference is near-optimal rather than exact, so a policy can occasionally outperform it in individual states. We evaluate the optimality of each method by the gap to the reference and the safety by the collision rate. The gap is defined as

$$
\begin{array} { r } { \mathrm { G a p } = \bar { V } ^ { \mathrm { D E } } - \bar { V } ^ { \pi } , } \end{array}\tag{30}
$$

where ${ \bar { V } } ^ { \pi }$ and $\bar { V } ^ { \mathrm { D E } }$ are the mean discounted reward of the learned policy and of the DE reference, respectively. A smaller gap indicates a policy closer to the reference. The collision rate is the percentage of the trajectories of the learned policy that end in a collision. Collisions among the surrounding vehicles are not considered. For all 4500 sampled states, the DE solution is collision-free.

We measure training stability by the learning curve and by the drop from the peak discounted reward to the discounted reward at the end of training. The learning curve shows the learning process through the evaluation of all recorded checkpoints on the validation set. The drop is calculated by

$$
\mathrm { D r o p } = \frac { \bar { V } _ { \mathrm { b e s t } } - \bar { V } _ { \mathrm { f i n a l } } } { \left| \bar { V } ^ { \mathrm { D E } } \right| } ,\tag{31}
$$

where $\bar { V } _ { \mathrm { b e s t } }$ is the mean discounted reward of the selected checkpoint, $\bar { V } _ { \mathrm { f i n a l } }$ is the mean discounted reward averaged over the last ten checkpoints, and $\bar { V } ^ { \mathrm { D E } }$ is the mean discounted reward of the DE reference. All three quantities are computed on the validation set. A large drop indicates that the peak reached during training is not retained.

We measure the training cost and decision cost of each method by the number of experiences it uses and by its decision time. The number of experiences is recorded when the mean discounted reward first exceeds $2 \hat { V } ^ { \mathrm { D E } }$ and $1 . 5 \bar { V } ^ { \mathrm { D E } }$ respectively, and when the peak is reached. Since the reward (29) is non-positive, $\bar { V } ^ { \mathrm { D E } } < 0$ . Therefore, $2 \hat { V } ^ { \mathrm { D E } }$ is the lower threshold. An experience is counted each time it is used to compute a target defined in Section IV. The decision time is measured on a set of 100 states sampled equally from the four layouts. For each method, the selected checkpoint of every seed is first warmed up on 100 states. The decision time is then recorded for making the decision at each state, one state at a time. This is repeated five times. For DE, the decision time is the time required to optimize the action sequence from a given state. It is measured at each of the 100 states and repeated five times with different random seeds, using the same parameters as for the reference solution. We report the mean and the standard deviation of the time per decision.

![](images/f4e9b8725ef8c0cdc4ee61791db0b8f8b55a23600de428494c1a205db2af7237.jpg)  
Fig. 2. Intersection layouts C1–C4. Vehicle 1 (red) is the controlled vehicle. A surrounding vehicle is conflicting (dark blue) or non-conflicting (light blue).

![](images/ed4c83eb6073344f4df11bad39cb75a63519a15750933a0fb38c075ccce3f727.jpg)  
Fig. 3. Shared network architecture.

## B. Comparison Results

1) Optimality and Safety: Table II reports the gap to the DE reference and the collision rate under the four layouts.

Overall, RCPI achieves the smallest gap and the lowest collision rate among the four methods. D-DDPG ranks second with a smaller gap than both FVI and DQN. FVI and DQN have similar overall gaps, but DQN has the highest collision rate among the four methods. As the complexity increases from C1 to C4, both the gap and the collision rate generally increase. RCPI and D-DDPG maintain relatively stable gaps from C1 to C3 and their gaps increase in C4. The gap of FVI increases approximately linearly with the number of conflicting vehicles, whereas the gap of DQN increases at an accelerating rate. Although DQN has a smaller gap than FVI in C1–C3, it exceeds FVI in C4. Measured by the increase in the gap from C1 to C4, RCPI is the least sensitive to layout complexity (1.71), followed by D-DDPG (11.06), FVI (20.04), and DQN (28.02).

The ordering by collision rate does not coincide with the ordering by gap. D-DDPG has a smaller gap than FVI but a slightly higher collision rate. The reason is that collisions are rare in the test set. Over the five seeds, the learned policies are evaluated on 20,000 trajectories in total, of which RCPI, FVI, D-DDPG, and DQN end in a collision in about 1, 20, 26, and 79 trajectories, respectively. The influence of these collisions is diluted when the discounted reward is averaged over all trajectories. Therefore, a smaller gap does not imply a lower collision rate.

RCPI differs from the other three methods in how it computes its target. RCPI obtains $Q ^ { \pi _ { k } } ( s , a )$ by summing the discounted rewards along a trajectory. Since the environment is deterministic, a single trajectory yields the exact value, and the target action contains no value-estimation error. Instead, FVI, DQN, and D-DDPG replace the true value of the next state with the prediction of a network. Any error of these networks therefore enters the target. Since the network is then updated toward this target and used to compute the next one, the error introduced at each update is carried forward and accumulates over training. This accumulation has been analyzed for FVI [11], DQN [18], and actor-critic methods such as DDPG [19].

2) Training Stability: Fig. 4 and Table III present the training stability of each method.

From Fig. 4, we can see that RCPI improves and approaches the DE reference within about $0 . 5 \times 1 0 ^ { 6 }$ updates. It then maintains this level until the end of training. From Table III, we can see that RCPI has the smallest drop among the four methods, which supports our observation. D-DDPG reaches the secondhighest discounted reward at about $0 . 5 \times 1 0 ^ { 6 }$ updates and starts to decline after about $1 . 3 \times 1 0 ^ { 6 }$ updates. In the second half of training, D-DDPG shows large oscillations. Its drop of 0.946 is the largest among the four methods. D-DDPG therefore has the least stable training process. DQN starts from the lowest discounted reward and oscillates strongly in the early updates. It then improves slowly but continuously. DQN has the secondsmallest drop. FVI improves quickly in the early stage and then plateaus at a level clearly below the reference. Although the mean curve of FVI appears flat, its drop shows that each seed declines noticeably after reaching its peak.

TABLE II  
OPTIMALITY GAP AND COLLISION RATE OVER FOUR LAYOUTS. THE SMALLEST GAP IN EACH COLUMN IS IN BOLD. COL.% IS THE COLLISION RATE IN PERCENT.
<table><tr><td></td><td colspan="2">C1</td><td colspan="2">C2</td><td colspan="2">C3</td><td colspan="2">C4</td><td colspan="2">All</td></tr><tr><td>Method</td><td>Gap</td><td>Col.%</td><td>Gap</td><td>Col.%</td><td>Gap</td><td>Col.%</td><td>Gap</td><td>Col.%</td><td>Gap</td><td>Col.%</td></tr><tr><td>RCPI</td><td>-0.15</td><td>0.00</td><td>0.16</td><td>0.00</td><td>0.63</td><td>0.00</td><td>1.56</td><td>0.02</td><td>0.55</td><td>0.01</td></tr><tr><td>FVI</td><td>11.25</td><td>0.00</td><td>18.06</td><td>0.00</td><td>23.30</td><td>0.02</td><td>31.29</td><td>0.38</td><td>20.98</td><td>0.10</td></tr><tr><td>DQN</td><td>7.24</td><td>0.00</td><td>11.11</td><td>0.16</td><td>19.98</td><td>0.36</td><td>35.26</td><td>1.06</td><td>18.40</td><td>0.40</td></tr><tr><td>D-DDPG</td><td>6.03</td><td>0.00</td><td>6.60</td><td>0.00</td><td>9.96</td><td>0.12</td><td>17.09</td><td>0.40</td><td>9.92</td><td>0.13</td></tr></table>

![](images/db2f316d3599db48800d68fdfc957ebf82f63c6a51a6b1e56b7e8e81124c7ec6.jpg)  
Fig. 4. Discounted reward of the four methods versus the number of network updates.

TABLE III  
DROP FROM THE PEAK DISCOUNTED REWARD TO THE DISCOUNTED REWARD AT THE END OF TRAINING. MEAN ± STANDARD DEVIATION OVER FIVE SEEDS.
<table><tr><td>Method</td><td>Drop</td></tr><tr><td>FVI</td><td> $0 . 3 5 6 \pm 0 . 1 6 9$ </td></tr><tr><td>RCPI</td><td> $0 . 0 0 4 \pm 0 . 0 0 1$ </td></tr><tr><td>DQN</td><td> $0 . 1 0 5 \pm 0 . 0 1 1$ </td></tr><tr><td>D-DDPG</td><td> $0 . 9 4 6 \pm 0 . 4 9 6$ </td></tr></table>

As discussed above, RCPI is updated toward a target action that contains no value-estimation error. Its updates are correct from the start of training and do not rely on accurate value estimates of the next states. This allows RCPI to reach the reference level in fewer updates than the other methods. In contrast, the other three methods compute their targets from the estimates of their own networks, which are inaccurate in the early stage of training. They therefore need more updates for these estimates to become accurate before their targets become reliable. In D-DDPG, the critic is trained on the actions selected by the actor, and the actor is trained to maximize the output of the critic. If the critic estimates the value of an action inaccurately, the actor moves toward that action, and the critic is then trained on it. The errors of the two networks can therefore reinforce each other instead of being corrected. This may explain the large oscillations and the decline of D-DDPG in the later stage of training.

![](images/4f3ca775007fb3136dca4985d4dc69042dade38a9a739f3509f35e30154397e6.jpg)  
Fig. 5. Discounted reward of the four methods versus the number of experiences used (log scale). The dashed lines mark $2 \bar { V } ^ { \mathrm { D E } }$ and $1 . 5 \bar { V } ^ { \mathrm { D E } }$

TABLE IV  
EXPERIENCE USAGE AND DECISION TIME OF EACH METHOD. MEAN ± SD OVER FIVE SEEDS.
<table><tr><td rowspan="2">Method</td><td colspan="2">Experiences</td><td rowspan="2">Decision time (ms)</td></tr><tr><td>Peak</td><td>Total</td></tr><tr><td>RCPI</td><td> $( 3 . 6 2 \pm 0 . 1 1 ) \times 1 0 ^ { 1 0 }$ </td><td> $3 . 7 9 \times 1 0 ^ { 1 0 }$ </td><td> $0 . 0 5 1 \pm 0 . 0 0 3$ </td></tr><tr><td>FVI</td><td> $( 4 . 4 6 \pm 1 . 5 3 ) \times 1 0 ^ { 8 }$ </td><td> $1 . 4 4 \times 1 0 ^ { 9 }$ </td><td> $0 . 2 2 7 \pm 0 . 0 0 7$ </td></tr><tr><td>DQN</td><td> $( 1 . 5 4 \pm 0 . 0 4 ) \times 1 0 ^ { 8 }$ </td><td> $1 . 6 0 \times 1 0 ^ { 8 }$ </td><td> $0 . 0 4 8 \pm 0 . 0 0 3$ </td></tr><tr><td>D-DDPG</td><td> $( 5 . 4 7 \pm 1 . 6 8 ) \times 1 0 ^ { 7 }$ </td><td> $1 . 6 0 \times 1 0 ^ { 8 }$ </td><td> $0 . 0 5 0 \pm 0 . 0 0 3$ </td></tr><tr><td>DE</td><td></td><td></td><td> $3 7 0 , 4 1 6 \pm 9 , 3 2 6$ </td></tr></table>

3) Training and Decision Cost: Fig. 5 shows the discounted reward of each method versus the number of experiences used. The dashed lines mark the thresholds $2 \bar { V } ^ { \mathrm { D E } }$ and $1 . 5 \bar { V } ^ { \mathrm { D E } }$ Table IV reports the number of experiences used to reach the peak and the total number used in training.

Fig. 5 and Table IV show that RCPI, which has the smallest gap, is the least data-efficient method. D-DDPG is the most data-efficient method, reaching each threshold with the fewest experiences, followed by DQN and FVI. RCPI requires the most experiences at every threshold. To reach its peak, RCPI uses $3 . 6 \bar { 2 } \times 1 0 ^ { 1 0 }$ experiences. This is about 660 times as many as D-DDPG.

This order is consistent with the number of experiences required to compute one target. Although DQN and D-DDPG both require one experience per target, D-DDPG reaches each threshold with fewer experiences than DQN. A possible reason is how an experience affects the policy. In DQN, an experience $( s , a , r , s ^ { \prime } )$ updates only the estimate $Q _ { \psi } ( s , a )$ of the action a it contains. Since the policy (15) selects the action with the largest Q-value, the policy at s changes only when this update is large enough to change which action has the largest value. In D-DDPG, the same experience updates both the critic and the actor. Each experience directly improves the policy. Thus D-DDPG is more data-efficient than DQN. FVI is less dataefficient than DQN and D-DDPG because, to compute each target (11) in our simulation, it needs to enumerate all actions in the action space, which requires |A| experiences. On the other hand, RCPI not only enumerates all actions but also simulates a trajectory for each action until termination, which requires |A|H experiences.

Table IV also shows the decision time, measured according to the procedure in Section V-A3. The time is measured on a single core of an AMD EPYC 9454 CPU in the Owl cluster of Virginia Tech Advanced Research Computing (ARC). All four methods require much less decision time than DE. Among the four methods, RCPI, DQN, and D-DDPG have similar decision times. The training outcomes of RCPI and D-DDPG are policy networks that output the action directly. Although the training outcome of DQN is an action-value network, its action is obtained by taking the action with the largest value according to (15). In contrast, the training outcome of FVI is a state-value network. In our simulation, to select an action according to (12), FVI needs to enumerate all actions, obtain the reward and next state of each action, and evaluate the value network at each next state. FVI therefore requires a longer decision time than the other three methods.

## C. Method Selection

The results in Section V-B show that no single method is the best in all aspects. We therefore suggest selecting a method according to its features as follows.

RCPI achieves the smallest gap, the lowest collision rate, the most stable training, and the least sensitivity to layout complexity. Its limitation is that it requires a model of the environment during training and the largest number of experiences. RCPI is suggested when a model is available and the budget for training is not a constraint.

D-DDPG ranks second in optimality and is the most dataefficient method. It is model-free, so it can be trained by interacting with a simulator that cannot be queried at an arbitrary state. Its limitation is the least stable training. D-DDPG is suggested when no reliable model is available or the budget for training is limited. It is not suitable when the policy is updated with newly collected data during deployment, since the policy in use can degrade unpredictably [20].

DQN is also model-free and is the most stable among the model-free methods. However, it has the highest collision rate and is the most sensitive to layout complexity. DQN is suggested when no reliable model is available and stable training is required.

FVI does not achieve the best result in any single aspect. It requires the model not only during training but also at every decision according to (12), and its decision time therefore grows with the cost of querying the model. On the other hand, it requires far fewer experiences than RCPI, its training is more stable than that of D-DDPG, and its collision rate is lower than that of DQN. FVI is suggested as a compromise when a model is available at both training and deployment but the budget for training does not allow RCPI.

Since none of the four methods is collision-free, a safety mechanism is recommended at deployment regardless of the method selected, such as a short-horizon collision check [21].

## VI. CONCLUSIONS

RL methods for AV decision-making are rarely compared under identical conditions. This paper compares FVI, RCPI, DQN, and D-DDPG on the same decision-making problem at an unsignalized intersection in terms of optimality and safety, training stability, and training and decision cost. RCPI is the closest to the reference and the most stable method because it computes its target from exact trajectories, but it also requires the most experiences. D-DDPG ranks second in optimality and is the most data-efficient method, but its training is the least stable because the errors of its actor and critic can reinforce each other. DQN is the most stable model-free method but has the highest collision rate. FVI does not achieve the best result in any single aspect, but balances training cost, stability, and safety at the price of requiring the model at every decision. These features suggest selecting a method according to whether a model is available, the budget for training, and whether stable training is required, as discussed in Section V-C. The comparison is conducted in a deterministic environment with the exact model. Extending it to stochastic traffic and approximate models is left for future work.

## REFERENCES

[1] D. J. Fagnant and K. Kockelman, “Preparing a nation for autonomous vehicles: Opportunities, barriers and policy recommendations,” Transportation Research Part A: Policy and Practice, vol. 77, pp. 167–181, 2015.

[2] W. Schwarting, J. Alonso-Mora, and D. Rus, “Planning and decisionmaking for autonomous vehicles,” Annual Review of Control, Robotics, and Autonomous Systems, vol. 1, pp. 187–210, 2018.

[3] R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction, 2nd ed. Cambridge, MA, USA: MIT Press, 2018.

[4] Z. Zhu and H. Zhao, “A survey of deep RL and IL for autonomous driving policy learning,” IEEE Transactions on Intelligent Transportation Systems, vol. 23, no. 9, pp. 14 043–14 065, 2022.

[5] T. M. Moerland, J. Broekens, A. Plaat, and C. M. Jonker, “Model-based reinforcement learning: A survey,” Foundations and Trends in Machine Learning, vol. 16, no. 1, pp. 1–118, 2023.

[6] Z. Lin, J. Ma, J. Duan, S. E. Li, H. Ma, B. Cheng, and T. H. Lee, “Policy iteration based approximate dynamic programming toward autonomous driving in constrained dynamic environment,” IEEE Transactions on Intelligent Transportation Systems, vol. 24, no. 5, pp. 5003–5013, 2023.

[7] Q. Li, X. Jia, S. Wang, and J. Yan, “Think2Drive: Efficient reinforcement learning by thinking with latent world model for autonomous driving (in CARLA-V2),” in Proceedings ofthe European Conference on Computer Vision (ECCV), 2024, pp. 142–158.

[8] A. Kendall, J. Hawke, D. Janz, P. Mazur, D. Reda, J.-M. Allen, V.- D. Lam, A. Bewley, and A. Shah, “Learning to drive in a day,” in Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2019, pp. 8248–8254.

[9] D. Isele, R. Rahimi, A. Cosgun, K. Subramanian, and K. Fujimura, “Navigating occluded intersections with autonomous vehicles using deep reinforcement learning,” in Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2018, pp. 2034–2039.

[10] B. R. Kiran, I. Sobh, V. Talpaert, P. Mannion, A. A. Al Sallab, S. Yogamani, and P. Perez, “Deep reinforcement learning for autonomous´ driving: A survey,” IEEE Transactions on Intelligent Transportation Systems, vol. 23, no. 6, pp. 4909–4926, 2022.

[11] R. Munos and C. Szepesvari, “Finite-time bounds for fitted value´ iteration,” Journal of Machine Learning Research, vol. 9, no. 27, pp. 815–857, 2008.

[12] M. G. Lagoudakis and R. Parr, “Reinforcement learning as classification: Leveraging modern classifiers,” in Proceedings of the International Conference on Machine Learning (ICML), 2003, pp. 424–431.

[13] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis, “Human-level control through deep reinforcement learning,” Nature, vol. 518, no. 7540, pp. 529–533, 2015.

[14] T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, and D. Wierstra, “Continuous control with deep reinforcement learning,” in Proceedings of the International Conference on Learning Representations (ICLR), 2016.

[15] C. R. Tilbury, F. Christianos, and S. V. Albrecht, “Revisiting the Gumbel-Softmax in MADDPG,” arXiv preprint arXiv:2302.11793, 2023.

[16] J. Kong, M. Pfeiffer, G. Schildbach, and F. Borrelli, “Kinematic and dynamic vehicle models for autonomous driving control design,” in Proceedings of the IEEE Intelligent Vehicles Symposium (IV), 2015, pp. 1094–1099.

[17] S. Liu and M. Liu, “Real-time solution-seeking for game-theoretic autonomous driving via time-distributed iterations,” arXiv preprint arXiv:2604.16184, 2026.

[18] S. Thrun and A. Schwartz, “Issues in using function approximation for reinforcement learning,” in Proceedings of the 1993 Connectionist Models Summer School, 1994, pp. 255–263.

[19] S. Fujimoto, H. van Hoof, and D. Meger, “Addressing function approximation error in actor-critic methods,” in Proceedings of the International Conference on Machine Learning (ICML), 2018, pp. 1587–1596.

[20] S. C. Y. Chan, S. Fishman, J. Canny, A. Korattikara, and S. Guadarrama, “Measuring the reliability of reinforcement learning algorithms,” in Proceedings of the International Conference on Learning Representations (ICLR), 2020.

[21] K. Mokhtari and A. R. Wagner, “Safe deep Q-network for autonomous vehicles at unsignalized intersection,” arXiv preprint arXiv:2106.04561, 2021.