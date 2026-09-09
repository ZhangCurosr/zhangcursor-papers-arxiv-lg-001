# Learning to build covering structures with continuous adjustments

Gabriel Vallat, Maryam Kamgarpour and Stefana Parascho

Abstract— Robotic construction offers the potential to use materials more efficiently and create complex geometries, but current methods rely on rigid, high-precision plans that cannot accommodate the tolerances, inaccuracies, and unexpected changes inherent in physical fabrication. In this work, we introduce a reinforcement learning approach that forgoes predefined plans entirely, instead generating construction sequences adaptively as the structure is built. Our method operates on graphstructured state representations and a mixed (parameterized) action space, requiring both discrete block selection and continuous placement parameters. Because the stability simulation of a structure is computationally heavy, we develop an efficient exploration strategy by incorporating unilateral edges into graph neural networks, extending soft actor-critic (SAC) to this hybrid setting. We evaluate our algorithm, HSAC, against the prior method hybrid-PPO (HPPO), demonstrating significantly higher asymptotic performance and good sample efficiency. We also demonstrate HSAC’s robustness to hyperparameter choices and its exploration capability, handling up to 10 discrete actions without performance degradation. Finally, we validate our approach on a physical two-robot setup, successfully building a spanning arch with 3D-printed blocks in closed-loop execution, confirming that policies trained in simulation transfer to real hardware.

## I. INTRODUCTION

Robotic construction is becoming increasingly prominent, allowing designers and architects to use materials more efficiently and enabling the creation of complex geometries that would be difficult to achieve with traditional methods [1]. However, this approach still has significant limitations. One specific challenge is the need for materials to conform to a high-precision plan defined before construction begins. Tolerances, inaccuracies, and unexpected changes create mismatches with this idealized case, as illustrated in Fig. 1. In the worst case, such errors can lead to catastrophic failure; at the very least, they necessitate costly and timeconsuming plan revisions during construction. To address this, we propose to forego the use of an initial plan entirely and instead generate one incrementally, at each time step of the construction process.

To develop our method, we build upon an assembly of rigid blocks with dry joints. In our experiments, we use plastic blocks, though stone blocks could be used in practice. This fabrication process allows components to be reused but requires the structure to remain purely under compression. After evaluating different construction methods, such as single-robot block stacking, we choose to build an arch using two robots working in tandem [2], [3]: one placing blocks while the other provides temporary support. This method requires the negotiation of multiple construction parameters, which is not easily done intuitively, while highlighting the impact of minor placement errors, keeping the structure perpetually on the verge of instability.

![](images/e462429c4f39b4efce052702df2cd824f8c37cc6358d4f085872d94fa4d66aec.jpg)  
Fig. 1. Initial state (left): it contains an initial block and a yellow rectangle that the structure needs to cover. Example of an arch built with small noise added at each step (right): The arch slowly deviates from the desired trajectory, and the yellow area is not completely covered.

To address the challenge above, we model the construction process as a controlled dynamical system and generate a policy: a mapping from states to actions. This formulation ensures fabrication feasibility by considering structural stability at each step of construction. Furthermore, it accommodates continuous adjustments in actions, whereas a plan-based approach typically requires restarting once certain thresholds are exceeded. Despite its promise, research on plan-free methods remains limited; most methods still rely on plans and attempt to increase their adaptability. For instance, some approaches allow the sequence of block placements to change in response to part delivery delays [4].

Previous plan-free strategies also generally rely on strong simplifications, such as discretizing the space [5], [6] or only considering 2D structures [7], [3]. Given the complexity of our dynamical system, we face the same constraints as these approaches and cannot rely on model-based control frameworks. We therefore turn to model-free reinforcement learning (RL).

In robotic construction, learning a policy requires reasoning over a mixed continuous–discrete action space: the agent must choose both which block to place (a discrete selection from available types) and where to place it (a continuous parameter). These decisions are coupled—the optimal placement depends on the chosen block, and vice versa—and existing algorithms such as HPPO [8] and parametrized-DDPG [9] struggle in complex environments with costly physicsbased stability checks. Mixed action spaces of this kind arise broadly in robotics, from manipulation to locomotion, yet remain poorly addressed by current methods.

We address this challenge with hybrid soft actor-critic (HSAC), a novel extension of SAC to mixed action spaces that, like most RL algorithms, makes no assumptions about state representation. For our construction task, we represent the structure as a graph—a problem-specific choice that allows us to leverage geometric invariants (translation, permutation, and unlike prior work [4], vertical rotation) to improve generalization. This combination of generality and domain-specific inductive bias makes HSAC broadly applicable to robotics problems that couple discrete and continuous decisions.

The resulting algorithm is, to our knowledge, the first to achieve sample-efficient learning in mixed action spaces with costly simulation. We validate HSAC against HPPO [8] and demonstrate its use on a physical two-robot setup that builds an arch from 3D-printed blocks, confirming that policies trained in simulation transfer to real hardware.

## II. DYNAMICAL SYSTEM DEFINITION

As an example of a robotic construction task, we use a spanning structure as shown in Fig. 1. Starting from an initial fixed block, one robot places a new element while another holds the free end to maintain stability. To enable an agent to design a structure autonomously, we require two key elements: a training environment, described in this section as a discrete-time dynamical system, and a learning algorithm, covered in section III, to find an optimal policy for the process.

As the construction of a structure is inherently sequential, we model it as a discrete-time controlled dynamic process, described by $( S , A , T , R , s _ { 0 } , \gamma )$ , where S is the state space, A the action space, $T : S \times A \to S$ the transition function, $R : S \times A \to \mathbb { R }$ the reward function, $s _ { 0 } \in S$ the initial state and $\gamma \in [ 0 , 1 ]$ is the discount factor. We describe below each of its components. Note that while we use a deterministic transition function here, our method readily extends to stochastic dynamics.

## A. State

To train our agent, we need a state $s \in S$ to represent the current structure. Moreover, this state should be invariant to factors that do not modify the optimal action (a change in coordinate system, for example), as our agent would otherwise need to learn to ignore these factors. To fulfill this task, we use a labeled heterogeneous graph $G _ { s } = \{ \mathcal { N } _ { s } , \mathcal { O } _ { s } \}$ where $\mathcal { N } _ { s } = \mathcal { N } _ { b } \cup \mathcal { N } _ { a } \cup \mathcal { N } _ { o }$ are the nodes representing, respectively, the present blocks $\mathcal { N } _ { b }$ , possible type of block to be added $\mathcal { N } _ { a }$ and objective ${ \mathcal { N } } _ { o } ,$ , and $\mathcal { E } _ { s } = \mathcal { E } _ { b b } \cup \mathcal { E } _ { b a } \cup \mathcal { E } _ { b o }$ are the edges linking these different nodes. The complete process through which the graph is labeled to represent a structure is described in the appendix ${ \mathrm { V } } { \mathrm { - } } { \mathrm { A } } ,$ and an example is shown in Fig. 2. The key novelty of our method is to attach a reference frame to each of our blocks, and to label the edges connecting them with their relative positions, rather than using a global coordinate system. This creates a graph that is invariant to rotation, translation and permutation. Our design follows a simple principle: the state should change only when the optimal action changes. Encoding block positions relative to one another rather than in a global frame achieves this, allowing the agent to learn more efficiently.

![](images/a39d8b8e389de90cf935d313af03921fe284f40adc47c65df68043c6425a90cc.jpg)

Fig. 2. Left: a simple three-block structure shown from two perspectives. The black block is the ground, the white block is free-standing, and the red block is held by a robot. Top right: the corresponding graph representation, showing block nodes ${ \mathcal { N } } _ { b } ,$ , action nodes $\mathcal { N } _ { a }$ (red), and objective $\mathcal { N } _ { o }$ (yellow). Bottom right: detailed view of the objective subgraph, with corner nodes $\mathcal { N } _ { o }$ and the edges $\mathcal { E } _ { o o }$ connecting them in yellow.  
![](images/3f0b359bb4b6b02c201a3a48016ca491fd83f7d78b67ffcbc8f5255d752788bd.jpg)  
Fig. 3. Our blocks are always trapezoidal extrusions, using a fixed aspect ratio, and the angle, or slope, γ to parametrize different shapes. On the left, we show how we parametrize the action with the translation $( x , y )$ and the angle α. The dashed rectangle is the B face of the next block.

Note also that our state space does not require us to know the exact shape of the blocks, as long as we can simulate its physical behavior: The algorithm is only given a block index that could represent any shape. This framework, with minor adjustments for scanning tolerances, could then be used with any shape, including natural materials with irregular geometries.

## B. Actions

The action selection proceeds in two steps. First, the agent chooses the type of block $a \in \{ 1 , \ldots , N \}$ to add to the structure. Once this decision is made, we describe a target position for the new block. We always orient it so that the face B of the new block touches the face A of the old one. The final position could then be parametrized using the 3 components $u = ( x , y , \alpha )$ shown on the right of Fig. 3. These represent a 2D translation from the center of face A of the currently supported block to the center of face B of the new block, along with an angle α between the two reference frames. Note that in practice, we use 4 parameters, and not 3, as we represent α by its cosine and sine [10], creating a continuous representation of the angle.

This makes our continuous action u a vector in $\mathbb { R } ^ { 4 }$ and our complete action set can then be represented as its Cartesian product with the discrete action set, $A =$ $\{ 1 , \ldots , N \} \times \cup _ { i = 1 } ^ { N } \mathbb { R } ^ { m _ { i } }$ , with $N = 3$ and $m _ { i } = m = 4$ for all i in our experiments (except if explicitly mentioned). We describe in more detail how we handled this mix of discrete and continuous actions in the section III, dedicated to the algorithm.

## C. Initial state

The initial state $s _ { 0 } \in G _ { s }$ simply consists of a permanently fixed base block, and a narrow rectangular area to cover, shown in yellow on Fig. 1 and Fig. 2. These two elements allow us to explore different classes of structures. By choosing a small angle γ for the initial block, the resulting construction tends to be tall arches, whereas if we increase the length of our rectangular area, we reward the agent for building wider arches.

## D. Reward and transition function

The transition function $T ( s , a )$ is separated in two phases. First, the new held block and its contact interface are added to a physics model based on Rigid Block Analysis (RBA) [4], and the previously held block is released. Our model then simulates the structure’s stability. If the structure is unstable, the episode ends in failure with a reward of −1. Otherwise, the agent receives a reward proportional to the area covered by the new block. This dense reward allows us to train the agent to keep the arch in a chosen plane. Finally, if one of the corners of the newly placed block is below the ground level, we consider the spanning structure as complete and end the episode, marking it as a success.

Together, this dynamical system is particularly hard to optimize: the graph-based state requires specific models (GNNs) to be processed into an action; this action comprises a discrete and a continuous part; and a suboptimal choice can lead to irrecoverable states: either immediate collapse or, worse, guaranteed collapse several steps later. To address these challenges, we propose hybrid soft actor-critic (HSAC), which we describe in detail in the next section.

## III. HYBRID SAC ALGORITHM

Classical reinforcement learning algorithms such as proximal policy gradient (PPO) [11] and SAC are designed for purely continuous or purely discrete actions and cannot be applied directly. While specialized methods for mixed action spaces exist—HPPO [8] being a notable example—they suffer from a critical limitation: their exploration strategy is the same for all states. In contrast, SAC, not yet extended to mixed action sets, uses a state-dependent exploration, empirically leading to both a faster exploration and a higher asymptotic performance.

In the remainder of this section, we first describe how we adapt the learning algorithm of SAC to accommodate hybrid actions, and then show how we can use GNNs not only to handle the graph-structured states, but also to remove the common over-parametrization of the Q-function caused by hybrid actions.

1) HSAC: Our training algorithm alternates between two phases. The first one consists of sampling the environment, using a stochastic policy $\pi _ { \phi } ( ( a , u ) | s )$ , where the discrete action is denoted as a and the continuous one as u, and storing the transitions $\left( s _ { t } , a _ { t } , u _ { t } , r _ { t } , s _ { t + 1 } \right)$ in a large replay buffer D. The second phase consists in optimizing the parameters φ of this policy and three other functions: a V-function $V _ { \psi } : S \to \mathbb { R }$ , and two Q-functions $Q _ { \theta _ { i } } : S \times A  \mathbb { R }$ , using the double Q-learning method [12]. We handle the mixed action set by using the chain rule: $\pi _ { \phi } ( ( a , u ) | s ) = \pi _ { \phi } ^ { c } ( u | a , s ) \pi _ { \phi } ^ { d } ( a | s )$ and optimize separately the discrete and continuous parts. These functions are optimized using objectives similar to those in the original SAC algorithm, with the loss functions given by:

$$
J _ { \boldsymbol { Q } } ( \theta _ { i } ) = \mathbb { E } _ { s _ { t } , a _ { t } , u _ { t } \sim \mathcal { D } } \left[ \frac { 1 } { 2 } ( Q _ { \theta _ { i } } ( s _ { t } , a _ { t } , u _ { t } ) - \hat { \boldsymbol { Q } } ( s _ { t } , a _ { t } , u _ { t } ) ) ^ { 2 } \right] ,\tag{1}
$$

$$
J _ { V } ( \psi ) = \mathbb { E } _ { s _ { t } \sim \mathcal { D } , a , u \sim \pi _ { \phi } ( s _ { t } ) } [ \frac { 1 } { 2 } ( V _ { \psi } ( s _ { t } ) - [ \operatorname* { m i n } _ { i } \mathcal { Q } _ { \theta _ { i } } ( s _ { t } , a , u ) 
$$

$$
- \alpha _ { c } \log \pi _ { \phi } ^ { c } ( u | a , s _ { t } ) - \alpha _ { d } \log ( \pi _ { \phi } ^ { d } ( a | s _ { t } ) ] ) ) ^ { 2 } \bigg ] ,\tag{2}
$$

$$
\begin{array} { r } { \mathbb { D } _ { K L } ( \pi _ { \phi } ^ { d } ( \cdot | s _ { t } ) | | \frac { \exp ( \alpha _ { d } ^ { - 1 } \operatorname* { m i n } _ { i } { Q } _ { \theta _ { i } } ( s _ { t } , \cdot , u ) ) } { Z _ { \theta } ^ { d } ( s , u ) } ) | , } \end{array}\tag{3}
$$

$$
\begin{array} { r l } & { J _ { \pi ^ { c } } ( \phi ) = \mathbb { E } _ { s _ { t } \sim \mathcal { D } , a \sim \pi ^ { d } ( \cdot | s _ { t } ) } \Bigg [ } \\ & { \quad \quad \quad \mathrm { D } _ { K L } \left( \pi _ { \phi } ^ { c } ( \cdot | a , s _ { t } ) \Bigg | \Bigg | \frac { \exp ( \alpha _ { c } ^ { - 1 } \operatorname* { m i n } _ { i } \mathcal { Q } _ { \theta _ { i } } ( s _ { t } , a , \cdot ) ) } { Z _ { \theta } ^ { c } ( s , a ) } \right) \Bigg ] , } \end{array}\tag{4}
$$

(5)

where $\hat { \cal Q } ( s _ { t } , a _ { t } , u _ { t } ) = r _ { t } + \gamma V _ { \bar { \psi } } ( s _ { t + 1 } )$ $\mathrm { D } _ { K L }$ denotes the $\mathrm { K L } -$ divergence, ψ¯ is an exponential moving average of the parameters ψ and $Z _ { \theta } ^ { \cdot }$ are normalizing partition functions that do not contribute to the gradient. In our implementation, the coefficients $\alpha _ { c }$ and $\alpha _ { d }$ are dynamic, and the entropies of both policy components converge to target values $H _ { T } ^ { c }$ and $H _ { T } ^ { d }$ respectively [13].

To optimize our policy in practice, we use the same reparameterization as in the original SAC paper [14]. This allows us to write the continuous stochastic policy as a deterministic function $u = f _ { u } ( s , a , \varepsilon )$ , that takes noise $\varepsilon \sim$ $N ( 0 , I _ { m } )$ , a spherical Gaussian, as input, implicitly defining $\pi _ { \phi } ^ { c } ( u | a , s )$ as the probability density of $f _ { u }$ under this noise. This allows us to rewrite the total policy cost as

$$
\begin{array} { r l } { J _ { \pi } ( \phi ) = C \big ( J _ { \pi ^ { c } } ( \phi ) + J _ { \pi ^ { d } } ( \phi ) \big ) } & { } \\ { = \mathbb { E } _ { s _ { t } \sim \mathcal { D } , \varepsilon \sim N ( 0 , I _ { m } ) } \Bigg [ \underset { a = 1 } { \overset { N } { \sum } } \pi _ { \phi } ^ { d } ( a | s _ { t } ) } & { } \\ { \cdot \big ( \alpha _ { c } \log ( \pi _ { \phi } ^ { c } ( f _ { u } ( s , a , \varepsilon ) | a , s _ { t } ) ) + \alpha _ { d } \log ( \pi _ { \phi } ^ { d } ( a | s _ { t } ) ) } & { } \\ { \quad - Q ( s _ { t } , a , f _ { u } ( s , a , \varepsilon ) ) \big ) \Bigg ] , } & { } \end{array}\tag{6}
$$

where C is constant with respect to $\phi$ and can therefore be ignored and $\begin{array} { r } { Q ( s , a , u ) = \operatorname* { m i n } _ { i } Q _ { \theta _ { i } } ( s , a , u ) } \end{array}$ . This yields the update procedure summarized in Algorithm 1.

Algorithm 1 HSAC   
Require: Some initial parameters $\phi , \theta _ { 1 } , \theta _ { 2 }$ and $\psi = { \bar { \psi } } ,$ , and   
an empty replay buffer ${ \mathcal { D } } .$   
$t  0$   
while $t < t _ { m a x }$ do   
$t _ { e p } \gets 0$   
$s _ { t } \gets s _ { 0 }$   
while Episode not terminated do   
$a _ { t } \sim \pi _ { \phi } ^ { d } ( \cdot | s _ { t } )$   
$u _ { t } \sim \pi _ { \phi } ^ { \dot { c } } ( \cdot | a _ { t } , s _ { t } )$   
$\mathcal { D } \gets \dot { \mathcal { D } } \cup ( s _ { t } , a _ { t } , u _ { t } , R ( s _ { t } , a _ { t } , u _ { t } ) , T ( s _ { t } , a _ { t } , u _ { t } ) )$   
$s _ { t + 1 } \gets T ( s _ { t } , a _ { t } , u _ { t } )$   
$t \gets t + 1$   
$t _ { e p } \gets t _ { e p } + 1$   
end while   
$t _ { o p t } \gets 0$   
while $t _ { o p t } < k t _ { e q }$ do   
$\theta _ { i } \gets \theta _ { i } - \delta \hat { \nabla } _ { \theta _ { i } } J _ { Q } ( \theta _ { i } )$ ▷ Using eq. 1   
$\psi  \psi - \delta \hat { \nabla } _ { \psi } J _ { V } ( \psi )$ ▷ Using eq. 2   
$\bar { \psi }  \tau \psi + ( 1 - \tau ) \bar { \psi }$   
$\phi  \phi - \delta \nabla J _ { \pi } ( \phi )$ ▷ Using eq. 6   
▷ Update of $\alpha _ { c }$ and $\alpha _ { d }$   
Sample $s \sim \mathcal { D } , \varepsilon _ { a } \sim N ( 0 , I _ { m } )$   
$H ^ { d } \gets \sum _ { a = 1 } ^ { N } \pi _ { \phi } ^ { d } ( a | s ) \log ( \pi _ { \phi } ^ { d } ( a | s ) ) )$   
$\begin{array} { r } { H ^ { c } \gets \sum _ { a } \pi ^ { d } ( a | s ) \log ( \pi ^ { c } ( \dot { f } ( s , a , \pmb { \varepsilon } _ { a } ) | a , s ) ) ) } \end{array}$   
log $\alpha _ { c } \gets \log \alpha _ { c } - \delta ( H _ { T } ^ { c } - H ^ { c } )$   
log $\alpha _ { d } \gets \log { \alpha _ { d } } - \delta ( \bar { H _ { T } ^ { d } } - H ^ { d } )$   
$t _ { o p t } \gets t _ { o p t } + 1$   
end while   
end while

The key novelty of our algorithm is to split the policy cost into Equations 4 and 5, instead of directly computing a single cost $J _ { \pi } ( \phi )$ for both the continuous and discrete policies. Note that doing so would result in the exact same loss functions as in the original SAC algorithm. Splitting the cost allows us to control the importance of the entropy of each part independently, using different $\alpha _ { c }$ and $\alpha _ { d }$ . This method is necessary in practice, as we observed that when using a single coefficient for both parts of the policy, our agents consistently set the discrete entropy close to 0 (i.e., chose the discrete action nearly deterministically), and used a more stochastic continuous action to reach the target total entropy. This behavior causes the learning to be stuck in local minima, while our method ensures that our discrete policy is explorative enough.

2) Graph neural networks: Due to the structure of our state and action spaces, we need to parametrize a map from graphs to a mix of continuous and discrete variables. We choose to do so using graph neural networks (GNN) and fully connected networks. The first step of each of these maps is to process heterogeneous graphs to extract features at the discrete action level: $f _ { \mathrm { v } } : G _ { s } \to \mathbb { R } ^ { N \times n _ { \mathrm { y } } }$ , where $n _ { y }$ is the size of the action-level feature. These features are then used to compute the policy and values. We use the transformer convolutional networks [15] to build our GNN. We then represent $f _ { y }$ by keeping the N features $y _ { a } \in \mathbb { R } ^ { n _ { \phi } }$ with $a \in \{ 1 , \ldots , N \}$ , attached to the action nodes as shown in Fig. 4. We use this architecture as a basis for both $V _ { \psi }$ and $\pi _ { \phi }$

![](images/3f3d5f1bf3ce37b1ed40973af05656f76a628e5fdcb02a2b1eb3684ebc68693c.jpg)  
Fig. 4. GNN architecture. (1) Input state (the objective nodes are not drawn for simplicity) (2) Output of the GNN. The features attached to the block nodes ζ can be seen as hidden states of a recurrent neural network and are discarded. (3) Output of the function $f _ { y } ( s )$ , in this case, a matrix of size $2 \times n _ { \mathrm { y } }$

To compute $V _ { \psi } ,$ , these features are aggregated using the minimum, maximum, and mean value across the nodes, resulting in a vector $\nu _ { \psi } \in \mathbb { R } ^ { 3 n _ { \phi } }$ . This vector is then used as input of a fully connected neural network with only one output.

To compute the policy, we process each of the features $y _ { a }$ as a separate input of three fully connected neural networks $f _ { d } , f _ { \mu }$ and $f _ { \Sigma }$ . This allows us to parametrize

$$
\pi _ { \phi } ^ { d } ( a | s ) = \frac { \exp ( f _ { d } ( y _ { a } ) ) } { \sum _ { j = 1 } ^ { N } \exp ( f _ { d } ( y _ { j } ) ) } ,\tag{7}
$$

$$
\pi _ { \phi } ^ { c } ( u | a , s ) = N ( f _ { \mu } ( y _ { a } ) , f _ { \Sigma } ( y _ { a } ) ) ,\tag{8}
$$

where $N ( \mu , \Sigma )$ is a multivariate Gaussian distribution of mean $\mu$ and covariance Σ. Note that the deterministic function $f _ { u } ( s , a , \varepsilon )$ can then be easily written as $f _ { u } ( s , a , \varepsilon ) =$ $f _ { \mu } ( y _ { a } ) + f _ { \Sigma } ( y _ { a } ) \varepsilon$

Finally, to parametrize an estimation of $Q : S \times A  \mathbb { R }$ we take inspiration from deep Q-networks (DQN) [16] with a discrete action space of size N. In that setting, a neural network $f _ { Q d } : S \to \mathbb { R } ^ { N }$ is used to do so, with $Q ( s , a ) = f _ { Q } ( s ) _ { a }$ . However, simply extending this method to mixed action sets has led to little success. The authors of [8] note that naively expanding DQN to mixed action sets causes overparametrization problems. As an example, let us define a deep hybrid Q-function $Q _ { \theta } : S \times \mathbb { R } ^ { N m }  \mathcal { \bar { \mathbb { R } } } ^ { N }$ , with $N = 2$ . We can then write

$$
Q _ { \theta } ( s , \left[ { u } _ { 1 } \right] ) = \left[ { q } _ { 1 } \right] ,\tag{9}
$$

$$
Q _ { \theta } ( s , \left[ { u } _ { 1 } \right] ) = \left[ { q } _ { 1 } ^ { \prime } \right] ,\tag{10}
$$

where $u _ { 2 }$ and $u _ { 2 } ^ { \prime }$ are two different actions parameters for discrete action $a = 2$ . If we parametrize $Q _ { \theta }$ using a fully connected network, we cannot ensure that $q _ { 1 } = q _ { 1 } ^ { \prime }$

![](images/8bd7ff15962718a07c96d82080701193be8d5a1682941b7da53d1d5bd053a20c.jpg)  
Fig. 5. The actions are attached to the state with unidirectional edges, ensuring that $q _ { 1 } = q _ { 1 } ^ { \prime }$ in Equation 10. In this example, the state is composed of 3 blocks, and the two possible block types that can be added. The objective nodes are omitted for simplicity.

The key novelty of our method—and what we believe is the main reason for its effectiveness—is leveraging unidirectional edges in the input graph, as shown in Fig. 5, to enforce this equality. As $u _ { 1 }$ is not connected to $u _ { 2 } ,$ we structurally ensure that the two variables do not interact. While we use this architecture to adapt SAC to our setup, we believe that this method can be used to extend any DQN-based method to mixed action sets effectively.

## IV. IMPLEMENTATION

## A. Simulation

The code for our environment implementation and algorithm is available on GitHub<sup>1</sup>. We used Warp [17] to parallelize contact detection across multiple simulation threads, significantly accelerating training. Gurobi [18] served as our optimization solver for the Rigid Block Analysis (RBA) stability checks. We stored each state-action pair using PyTorch Geometric objects [19], which enabled efficient batching and GPU-based graph neural network training.

## B. Simulation Results

To test the limits of our algorithm, we performed three experiments. First, we compared the learning curve of our algorithm with HPPO. Then, we showed that our method works with a wide range of hyperparameters. Finally, we illustrated the exploration capabilities and generality of our method by training it in different environments, varying both the friction coefficient between the blocks and the number of discrete actions.

1) HSAC vs HPPO: To benchmark our algorithm, we implemented HPPO using a similar GNN architecture. The hyperparameters were tuned manually, and their values are available in appendix V-B. As shown in Fig. 6, HSAC achieves better policies when converged. What the graph does not show, however, is that HSAC requires more time to optimize the network: a single HSAC step takes on average five times longer than an HPPO step. This overhead can be mitigated by adjusting the ratio of training updates to simulation steps, effectively making this ratio a tunable hyperparameter that can be optimized based on the available simulation budget.

![](images/05531a44baca1430aee06bb020c97d505f11df75488efacb3aae213f17e2aac8.jpg)  
Fig. 6. Total reward accumulated during an episode. Results are averaged over three runs. As one can see, HPPO gets stuck in local minima and never reaches the maximum return.

Different hyperparameters  
![](images/bc8137954a2f6f817277aed6c58362f54e0ea5cd2ffc0ee42fa476e9bc8eab63.jpg)  
Fig. 7. Total reward accumulated during an episode. We used time rather than samples, as the duration of a single step could vary widely. Results are averaged over three runs.

2) Hyperparameters: To show the stability of HSAC with respect to hyperparameters, we ran four different versions.

• A base setup, using around 24 million parameters and performing 64 update steps for each environment step taken

• A larger model, using around 143 million parameters, still performing 64 update steps per environment step

• A less trained model, using 24 million parameters and performing 8 update steps per environment step

• A deterministic rollout, using the same training setup as the base one, but using a deterministic policy when interacting with the environment

As shown in Fig. 7, using a larger model did not significantly improve the results, and a smaller training ratio slowed convergence. Moreover, as the deterministic version of the algorithm cannot take advantage of the parallelization of the environments, its performance is slightly worse than that of the stochastic policy.

3) Change in environment: To show that our algorithm is able to handle different kinds of environments, we changed some parameters in the simulations. We focused on two parameters: the number of discrete actions, and the friction coefficient between our blocks. An increase in the number of discrete actions N, allowing our agent to select more types of blocks, increases the size both of the search space and of the solution space. Increasing the friction coefficient has roughly the same effect, as more states are stable. This setup increases the variety of possible spanning structures, but also allows the agent to search in a bad direction for longer. As shown in Fig. 8, HSAC can still handle a discrete action set of size N = 10 effectively, and is able to obtain better results when the friction coefficient is higher. This latter result implies that our algorithm is exploring the environment efficiently, and that narrowing the search space is not required.

![](images/f9a4a017852bb17020bac2f1df9707eb630be96c8c803c143ef34e78c7ecb55c.jpg)  
Fig. 8. Violin plot of the return over the last 1024 episodes of training, aggregated across 3 runs.

4) Advantage of a policy: To illustrate how our policy can adapt to unforeseen modifications, we forced the actions taken at the second and fourth steps during construction of the arch shown in Fig. 9. The top arch is purely produced by a trained policy, and the lower one is the resulting structure after these modifications. As one can see, the agent has been able to compensate for the change over time, leading to nearly identical structures.

## C. Prototype

As a proof of concept for our algorithm, we used a policy trained with a friction coefficient of 0.15 to build an arch. We used trapezoid blocks with angles γ = 5<sup>◦</sup>,10<sup>◦</sup>, and 20<sup>◦</sup>, and a ground block with γ= 30<sup>◦</sup>. Interestingly, our agent did not use any of the 20<sup>◦</sup> blocks in its structure. Our setup, building on the one used in [6], consisted of two ABB Gofa robots, a Zivid 3D camera and a set of 3D-printed blocks. At each time-step, the position of each block was captured using the camera and Aruco codes. The position of each block relative to the ground was then transferred to our simulator, and we selected the action based on our trained policy. To minimize the influence of calibration errors, we placed all blocks using the same robot, using the other only as a temporary support, leading to the construction process shown in Fig. 11.

Finally, we compare the closed-loop system results with those of an open-loop implementation. In this baseline setup, we computed the whole structure in simulation, and used the result as a plan. Our results show that our method was able to keep the arch in the target plane, as shown in the left of Fig. 10, while the open-loop version deviated more, as shown on the right photo.

![](images/25ee56df930798abfeac2f10c5daebf47fc00d8943be46ece8abe944f560de1f.jpg)

Fig. 9. The top arch is the result of our policy. For the lower arch, we forced a different action on steps 2 and 4. Such adaptations are often needed in robotic construction, as the simulation generally does not include all the constraints like temporary obstacles or human intervention.  
![](images/d1feed5ff55a3d11593fa8eca7a0e4249d0c15bc83edff28d58b09018d8933e7.jpg)  
Fig. 10. Left: Structure built in closed-loop. Right: Structure built in openloop. As one can see, the closed-loop is better aligned with the ground block.

## V. CONCLUSION

In this work, we introduced a reinforcement learning framework for plan-free robotic construction that operates directly on graph-structured representations of 3D structures. By modeling the construction process as a dynamical system, we formulated the problem as learning a policy that maps states to mixed continuous–discrete actions, selecting both the type of block to place and its precise position.

The key technical novelty of our approach lies in leveraging graph neural networks with unidirectional edges to enforce independence between Q-values for different discrete actions, a structural inductive bias that proved essential for stable learning. This led to hybrid soft actor-critic (HSAC), an algorithm that extends SAC to mixed action spaces while maintaining adaptive exploration through separate entropy coefficients for the discrete and continuous components of

![](images/603e3c17e883daf9d25e4e83875de1f14a2cc7b2709b0dbdf51d44918cac1d52.jpg)  
Fig. 11. Construction process using two robots. First row (left to right): the first robot places a block, the second robot supports it, and the first robot withdraws. After each placement, the camera captures the block positions, and our policy computes the next block’s target location. Second row: subsequent stages of the construction process.

the policy.

Through extensive experiments, we demonstrate that HSAC consistently outperforms prior methods such as HPPO, achieving higher asymptotic performance. We also show its robustness to hyperparameter choices, and that it scales effectively to larger action spaces—handling up to ten discrete block types without degradation—and adapts to varied environment dynamics such as changes in friction coefficients. As a proof of concept, we validated our approach on a physical robot setup, where two ABB GoFa arms successfully built an arch using 3D-printed blocks in closed-loop execution, demonstrating that policies trained in simulation transfer to real hardware.

## APPENDIX

## A. State Construction

• The nodes in $\mathcal { N } _ { b }$ are labeled with all features necessary to describe a block, except their position. This includes the type of block used, its $I _ { z z }$ inertia component (which encode its vertical orientation), and two binary flags indicating whether the block is held by a robot or part of the fixed setup. Formally, we define a labeling function $f _ { b } : \mathcal { N } _ { b }  \{ 1 , \ldots , N \} \times \mathbb { R } \times \{ 0 , 1 \} ^ { 2 }$ , where N denotes the number of different block types.

• The nodes in $\mathcal { N } _ { a } = \{ 1 , . . . , N \}$ contain the potential blocks that can be added to the structure. While it is not classical to include the list of all possible actions in the state of a dynamical system, these nodes serve as placeholders for the output of the policy model. They are simply labeled with their index, allowing our policy to later differentiate between them. In different setups, where some blocks are only available in finite quantity, we could also remove some nodes from $\mathcal { N } _ { a }$ to restrict the agent to only use available parts.

• The nodes in $\mathcal { N } _ { o }$ are used to define the area that the agent has to cover. Each of them represents the corner of a polygon. They do not require a label, as the edges $\mathcal { E } _ { o o }$ are sufficient to define any polygon.

• The edges in $\mathcal { O } _ { b b } = \{ ( i , j ) : i , j \in \mathcal { N } _ { b } \}$ are connecting all nodes in ${ \mathcal { N } } _ { b } .$ . Their labels are used to encode the transformation between the reference frames of blocks i and j. The resulting labeling function can be defined as $f _ { b b } : \mathcal { N } _ { b } \times \mathcal { N } _ { b }  \mathbb { R } ^ { 1 2 }$ , as we vectorize only the nontrivial components of the $4 \times 4$ transformation matrix. Note that we keep all 9 components of the rotation matrix, resulting in an overparameterized representation. The reason behind this choice is to increase the relative number of parameters related to the block position, adding a small inductive bias toward these features in our models.

• The edges in $\mathcal { E } _ { b a } = \{ ( i , j ) : i = \operatorname* { m a x } \mathcal { N } _ { b } , j \in \mathcal { N } _ { a } \}$ attach the most recent block to each action. This instructs the model that the following block will be placed against it. As our models do not modify the labels attached to the edges, we also label these edges with the action they are attached to: $f _ { b a } ( i , j ) = f _ { a } ( j ) = j .$ . In practice, this acts similarly to a residual connection in a convolutional neural network, reducing the gradient vanishing problem.

• To determine the existence of an edge in $\mathcal { E } _ { o o } \subseteq \{ ( i , j )$ $i , j \in \mathcal { N } _ { o } \}$ , we performe a constrained Delaunay triangulation of the polygon we want to cover. The corners of each triangle are then connected. These edges are then labeled by the distance between the two corners.

• Finally, each objective node is linked to each block by the bidirectional edges $\mathcal { E } _ { b o } ^ { o }$ . Each of the edges is labeled with the position of the corner in the reference frame of the block.

## B. Hyperparameters

Table I lists the parameters of our base training environment, Table II those of our base HSAC algorithm, and Table III the additional hyperparameters for HPPO. For the

ENVIRONMENT BASE SETUP (ALL EXPERIMENTS)

Parallel environments 16 Discount factor 0.9 angles $5 ^ { \circ } , 1 0 ^ { \circ } \ \& \ 2 0 ^ { \circ }$ ground angle 5<sup>◦</sup> friction coef. 0.3

TABLE I

hidden channels 64 GNN layers 8 Fully connected layers Attention heads $_ { 2 } ^ { 2 }$ optimizer Adam learning rate $5 \cdot 1 0 ^ { - 4 } $ Exponential rate of ψ¯ $1 0 ^ { - 5 }$ Replay buffer size 100’000   
Update steps / env steps 4 Update batch size 256   
Continuous target entropy -3 Discrete target entropy 0.5

TABLE II

HSAC BASE SETUP (ALL EXPERIMENTS)

larger model, we increased the GNN layers to 12 and hidden channels to 128. To reduce training time, we decreased the batch size to 128 and performed only two update steps. For experiments with fewer discrete actions, we retained only blocks with $\gamma = 5 ^ { \circ }$ and $1 0 ^ { \circ }$ ; for more actions, we used homogeneous spacing between $\gamma = 5 ^ { \circ }$ and 20<sup>◦</sup>.

## ACKNOWLEDGMENT

We would like to deeply thank Jingwen Wang for her code to control the robots in close-loop, in particular for her modules regarding path and motion planning and force control. This work was supported as a part of NCCR Automation, a National Centre of Competence in Research, funded by the Swiss National Science Foundation (grant number 51NF40 225155), by the EPFL AI center, and by the SNFS (grant number:

## REFERENCES

[1] L. Salamanca, A. A. Apolinarska, F. Perez-Cruz, and M. Kohler,´ “Augmented intelligence for architectural design with conditional autoencoders: Semiramis case study,” in Towards Radical Regeneration, C. Gengnagel, O. Baverel, G. Betti, M. Popescu, M. R. Thomsen, and J. Wurm, Eds. Cham: Springer International Publishing, 2023, pp. 108–121.

[2] S. Parascho, I. Han, S. Walker, A. Beghini, E. Bruun, and S. Adriaenssens, “Robotic vault: a cooperative robotic assembly method for brick vault construction,” Construction Robotics, vol. 4, 12 2020.

[3] J. Wang, W. Liu, G. T.-M. Kao, I. Mitropoulou, F. Ranaudo, P. Block, and B. Dillenburger, “Multi-robotic assembly of discrete shell structures,” in Advances in Architectural Geometry 2023, 2023.

ε 0.2 GAE λ 0.95   
Continuous entropy bonus 10<sup>−3</sup>   
Discrete entropy bonus 10<sup>−2</sup> TABLE III HPPO SPECIFIC PARAMETERS

[4] Z. Wang, W. Liu, J. Wang, G. Vallat, F. Shi, S. Parascho, and M. Kamgarpour, “Learning to assemble with alternative plans,” p. 1–16, jul 2025. [Online]. Available: https://infoscience.epfl.ch/handle/20.500.14299/252688

[5] G. Vallat, J. Wang, A. Maddux, M. Kamgarpour, and S. Parascho, “Reinforcement learning for scaffold-free construction of spanning structures,” in Proceedings of the 8th ACM Symposium on Computational Fabrication, ser. SCF ’23. New York, NY, USA: Association for Computing Machinery, 2023. [Online]. Available: https://doi.org/10.1145/3623263.3623359

[6] J. Wang, J. Kirschner, P. Rolland, L. Salamanca, and S. Parascho, “Learning to build: Autonomous robotic assembly of stable structures without predefined plans,” 2026. [Online]. Available: https://arxiv.org/abs/2602.23934

[7] V. Bapst, A. Sanchez-Gonzalez, C. Doersch, K. L. Stachenfeld, P. Kohli, P. W. Battaglia, and J. B. Hamrick, “Structured agents for physical construction,” in International Conference on Machine Learning, 2019. [Online]. Available: https://api.semanticscholar.org/CorpusID:102352078

[8] Z. Fan, R. Su, W. Zhang, and Y. Yu, “Hybrid actor-critic reinforcement learning in parameterized action space,” in Proceedings of the 28th International Joint Conference on Artificial Intelligence, ser. IJCAI’19. AAAI Press, 2019, p. 2279–2285.

[9] M. Hausknecht and P. Stone, “Deep reinforcement learning in parameterized action space,” 2024. [Online]. Available: https://arxiv.org/abs/1511.04143

[10] Y. Zhou, C. Barnes, J. Lu, J. Yang, and H. Li, “ On the Continuity of Rotation Representations in Neural Networks ,” in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). Los Alamitos, CA, USA: IEEE Computer Society, Jun. 2019, pp. 5738–5746. [Online]. Available: https://doi.ieeecomputersociety.org/10.1109/CVPR.2019.00589

[11] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” 2017. [Online]. Available: https://arxiv.org/abs/1707.06347

[12] S. Fujimoto, H. van Hoof, and D. Meger, “Addressing function approximation error in actor-critic methods,” in International Conference on Machine Learning, 2018. [Online]. Available: https://api.semanticscholar.org/CorpusID:3544558

[13] T. Haarnoja, A. Zhou, K. Hartikainen, G. Tucker, S. Ha, J. Tan, V. Kumar, H. Zhu, A. Gupta, P. Abbeel, and S. Levine, “Soft actor-critic algorithms and applications,” 2019. [Online]. Available: https://arxiv.org/abs/1812.05905

[14] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “Soft actorcritic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor,” in Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmassan,¨ Stockholm, Sweden, July 10-15, 2018, ser. Proceedings of Machine Learning Research, J. G. Dy and A. Krause, Eds., vol. 80. PMLR, 2018, pp. 1856–1865. [Online]. Available: http://proceedings.mlr.press/v80/haarnoja18b.html

[15] Y. Shi, Z. Huang, W. Wang, H. Zhong, S. Feng, and Y. Sun, “Masked label prediction: Unified massage passing model for semi-supervised classification,” ArXiv, vol. abs/2009.03509, 2020. [Online]. Available: https://api.semanticscholar.org/CorpusID:221534325

[16] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis, “Human-level control through deep reinforcement learning,” Nature, vol. 518, no. 7540, pp. 529–533, Feb 2015. [Online]. Available: https://doi.org/10.1038/nature14236

[17] M. Macklin, “Warp: A high-performance python framework for gpu simulation and graphics,” https://github.com/nvidia/warp, March 2022, nVIDIA GPU Technology Conference (GTC).

[18] Gurobi Optimization, LLC, “Gurobi Optimizer Reference Manual,” 2026. [Online]. Available: https://www.gurobi.com

[19] M. Fey, J. Sunil, A. Nitta, R. Puri, M. Shah, B. Stojanovic,ˇ R. Bendias, A. Barghi, V. Kocijan, Z. Zhang, X. He, J. E. Lenssen, and J. Leskovec, “Pyg 2.0: Scalable learning on real world graphs,” 2025. [Online]. Available: https://arxiv.org/abs/2507.16991