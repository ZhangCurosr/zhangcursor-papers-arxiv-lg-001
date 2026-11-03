# Bridging Neural Tangent Kernel Analysis and Reinforcement Learning for Deep Function Approximation

William St-Arnaud Mila, Université de Montréal

Margarida Carvalho Mila, Université de Montréal

Golnoosh Farnadi Mila, McGill University

william.st-arnaud@umontreal.ca

carvalho@iro.umontreal.ca

farnadig@mila.quebec

## Abstract

Despite their widespread success across machine learning, the learning dynamics and generalization properties of neural networks remain poorly understood. The neural tangent kernel (NTK) provides an analytical characterization of the learning dynamics of infinitely wide neural networks and serves as a useful proxy for finite-width networks in supervised learning. In this work, we extend the NTK framework to reinforcement learning (RL) by introducing a temporal-diference (TD) flow formulation that accommodates time-varying mini-batches and TD targets, yielding a closed-form solution for value function evolution in an overparametrized setting. We also devise an optimization-based state-space exploration method that leverages the closed-form solution, coupled with constraints on the environment, to model the set of feasible states. On a deterministic grid environment, we validate that the proposed RL-based NTK formulation is indeed learning eficiently and adapting to the distribution shift in the training data distribution, and we demonstrate that the resulting exploration strategy improves sample eficiency. To our knowledge, this is the first work to extend NTK theory to characterize the learning dynamics of RL training with time-varying mini-batches and TD updates.

## 1 Introduction

In the context of reinforcement learning (RL), deep Q-learning marks an important milestone by demonstrating the ability of deep neural networks to learn complex policies (Mnih et al., 2015). Prior to its introduction, combining tabular methods like Q-learning with nonlinear function approximators was notoriously unstable and prone to divergence. Deep Q-Networks (DQN) resolve these instabilities by introducing experience replay to break temporal correlations and a periodically updated target network to stabilize parameter updates. Its major breakthrough on Atari 2600 games demonstrated that RL could scale directly to high-dimensional visual domains from raw pixels (Mnih et al., 2013). Nevertheless, DQN remains susceptible to certain pitfalls in various contexts, such as overestimation bias and sample ineficiency. To address the former, double DQN (DDQN) decouples action selection from value evaluation, using the online DQN to select the greedy action and the target DQN to evaluate its value during the online network’s parameter update.

In order to better understand how these algorithmic ideas stabilize optimization during training, one can study their training dynamics directly. Observing black-box network updates empirically is often uninformative, making analytical approaches like the neural tangent kernel (NTK) essential. The NTK is a valuable tool to study deep neural networks in simple settings, yet it normally rests on assumptions that are inconsistent with the common RL training framework. Specifically, it is very natural to update parameters using batches of data instead of computing the gradient over the full dataset. In addition, many learning algorithms feature distribution shifts when new data is introduced (e.g., of-policy RL algorithms).

In this work, our first theoretical contribution is to bridge the gap between classical NTK formulations and RL training introducing a temporal-diference (TD) flow formulation that accommodates time-varying minibatches and TD targets. For overparameterized ReLU networks (i.e., in the infinite-width limit), we derive a closed-form solution governing value function evolution. On finite state-action spaces, this solution is both strongly convex and smooth. Second, on the practical side, we leverage these properties to formulate a mathematical program, enabling of-the-shelf optimization solvers to search the state-action space globally for high-value exploration targets in a single optimization step. Third, through computational experiments, we validate the proposed NTK formulation by showing that it produces convergent, stable learning behavior, and we demonstrate that our solver-augmented exploration framework substantially improves sample eficiency over standard deep Q-learning baselines.

## 2 Literature Review

Experience replay and mini-batch dynamics Replay bufers are used in deep Q-learning to help decorrelate temporal data. Existing methods for memory management rely on empirical tuning, heuristic transformations and statistical approximations over historical samples. We review a few key contributions on the use of replay bufers.

Fedus et al. (2020) provide a detailed ablation study of experience replay (ER) in order to identify key properties that make it eficient. They show that increasing capacity of the replay bufer greatly improves performance in some tasks while not afecting others. Finally, they also characterize the importance of the replay ratio across various tasks. They formulate hypotheses and validate them to obtain general practical recommendations: 1) increasing capacity and decreasing the age of the oldest policy, and 2) using n-step returns in an of-policy setting. Liu & Zou (2018) analyze the properties of the replay bufer on solutions to simple ODEs derived from linear neural networks. They show that the amount of memory can substantially alter the performance of the algorithm and they also derive a regime where prioritized experience replay (PER) is harmful. Our proposed extension of the NTK to multiple mini-batches allows for a continuoustime equivalent of multi-batch training. It can be used to test the same hypotheses in overparameterized networks at a reduced computational cost. Importantly, we devise a solver-augmented exploration strategy that can select state-action pairs with high TD loss under the current model predictions. Contrary to PER, we do not replay tuples with high TD loss more often, and we treat the added data simply as a new addition to a dataset.

Fujita (2025) extend random reshufling (RR) of the dataset to experience replay and show that its theoretical advantages over with-replacement strategies translates to the RL case. Di-Castro et al. (2022) analyze the theoretical properties of a stochastic process that is pushed in a replay bufer and sampled. Their analysis provides theoretical tools to prove the convergence of multiple replay-based algorithms. The random reshufling idea can be analyzed using our closed-form solution with mini-batches. While the shufling ensures that all instances in the bufer are used in training, regardless of the age of the policy used to retrieve them, significantly more weight is attributed to instances that appear multiple times. In a deterministic environment, this can be prevented by only storing unique instances in the bufer; we use this idea in our experiments, and we sample with replacement to satisfy typical SGD assumptions.

Isele & Cosgun (2018) explore diferent selection strategies for experiences that are stored in the bufer. They compare methods that favour surprise, reward, training distribution matching, and state space coverage. They show that training distribution matching is consistently the best approach. However, they derive a case where state space coverage is the best approach. Zhao et al. (2024) propose a diversity-based experience replay (DBER) algorithm to achieve state-of-the-art results in sparse result environments. Using the closedform solution derived via the NTK, we instead use solver-augmented exploration of the state space to select states whose predicted TD loss is high. We note, however, that we can enforce diversity by introducing no-good cuts or hamming cuts to the optimization model that is fed to the solver.

Andrychowicz et al. (2017) present a novel algorithm, hindsight experience replay (HER), to improve the sample eficiency of the experience-based algorithms for sparse rewards. The goal of a replayed example is modified post-hoc to obtain a stronger signal during learning. Crowder et al. (2024) apply HER to PPO and demonstrate that it dramatically accelerates the learning. The same can be said of common on-policy learning algorithms on a tested predator-prey model. Kim et al. (2024) introduce symmetric replay training (SRT) in the context of combinatorial optimization problems. Their methodology modifies common RL approaches such as PPO, and even GFlowNets, that seek to obtain high-reward samples. Given a highreward sample, a symmetric trajectory is replayed uniformly at random. This corresponds to maximizing the entropy of the distribution of trajectories ending in a given state x. On all tested approaches, they show improvement in the sample eficiency and overall obtained solutions in the travelling salesman problem (TSP). Finding symmetric trajectories can be computationally hard, and we instead rely on a solver-based approach that leverages the closed-form solution of the NTK in order to select state-action pairs that have high TD loss.

Neural Tangent Kernel Analysis in RL Prior RL applications use neural network approximators purely as analytical tools to derive asymptotic global bounds. Next, we revise these results and their relation to the NTK theory.

Xu & Gu (2020) prove that deep Q-learning finds the optimal policy with a $O \left( { \scriptstyle { \frac { 1 } { \sqrt { T } } } } \right)$ convergence rate if the network approximator is suficiently overparameterized (explicitly, wide). Their proof does not rely on the i.i.d. assumption, but requires that the data is generated from an MDP. Qiu et al. (2021) devise a provably sample and computationally eficient reward-free RL algorithm, making use of nonlinear kernel and neural function approximators for single-agent MDPs. They achieve polynomial sample complexity in $\textstyle { \frac { 1 } { \epsilon ^ { 2 } } }$ for a ϵ-suboptimal policy. Chen et al. (2020) show that neural networks trained with noisy gradient descent and weight regularization can exhibit kernel training behaviour. They also obtain generalization bounds using mean-field analysis. They show that kernel approaches, when trained with noisy gradient, can eficiently learn functions if the initial weight distribution is not too far from the optimal weight distribution leading to interpolation on the data.

Che et al. (2024) provide a theoretical analysis of overparameterized linear networks in the context of TDlearning. The authors compare the learning dynamics of various approaches to TD-learning. The first uses the same set of parameters on both sides of the TD equation; the second freezes the right-hand side (i.e. the TD target) and updates the parameters on the left-hand side, periodically updating the target parameters to the latest network parameters. Under mild assumptions, TD-learning with periodic target updates is shown to converge to an optimal solution on the training dataset (ofline learning). This work covers linear networks and does not extend to deep neural networks. The results that are provided for Q-learning are in the continuous control case, and while convergence in the parameter space is established, no bounds on the error are given. As a point of comparison in section 7, we demonstrate the stable learning dynamics of overparameterized deep neural networks under varying mini-batches of data.

Lyle et al. (2022) apply the theory of kernel regression in the context of TD learning. Central to their results is a closed-form solution to the gradient flow of a kernel predictor under TD error. The authors argue that the value function evaluated on novel observations depends strongly on the training data, and that the minimization of large TD errors can have a significant interference efect. This observation runs contrary to the intuition that methods that generalize well exhibit weak interference. The gradient flow in the closed-form solution updates both the left-hand and right-hand sides of the TD equation simultaneously; in other words, the TD target does not rely on a frozen instance of the predictor.

St-Arnaud et al. (2025) analyze the properties of the NTK in the lazy regime. They show that, contrary to prior beliefs, the NTK leads to non-trivial predictions. By studying the NTK of infinitely-wide deep ReLU networks, it can be inferred that its eigenvalues are bounded and strictly greater than zero. In the context of stochastic gradient descent analysis, this implies that infinitely-wide deep ReLU networks exhibit both smoothness and strong convexity during training. These two properties combine to yield desirable convergence properties in mini-batch training, when the full dataset is held constant. While it is in theory intractable to train truly overparameterized networks, the existence of a closed-form solution for infinitelywide deep ReLU networks provides us with a proxy to these models. Moreover, we show in the experiments section how to accommodate the distribution shift in the RL setting when training an NTK model. This approach can also be applied to finite-width neural networks.

Active Exploration and State Space Optimization Eysenbach et al. (2019) look at the intersection of RL and planning. They devise a planning algorithm based on the current value estimates. With the obtained goal in mind, they are able to use the RL agent’s policy to guide towards that goal. The planning algorithm is run on a graph-based abstraction of the state space obtained via exploration with a RL agent; nodes are derived from observations stored in the replay bufer. However, the current value estimates are given by a black-box and no additional information can be gained. By leveraging the NTK for infinitely-wide deep ReLU networks in our work, we gain access to a closed-form solution, facilitating novel approaches in planning and exploration.

In addition to training a Q-network to maximize cumulative rewards, Simmons-Edler et al. (2020) train a second neural network whose value function is induce by an auxiliary reward function that is equal to the absolute temporal diference error of the first model. Both models select training data from a shared replay bufer and together, they form a an adversarial training framework where exploration is driven by state-action pairs that lead to high cumulative temporal diference error. Their approach performs comparably or outperforms most baselines on a variety of continuous control tasks. We apply the NTK to this adversarial setup, enabling us to study the learning dynamics in the context of overparameterized networks. The existence of a closed-form solution further allows us to enhance the adversarial exploration by way of constrained optimization problem solvers.

In summary, existing literature addresses mini-batch replay through empirical sample selection, uses NTK theory strictly for asymptotic proofs, and restricts active exploration to local graph rollouts. We bridge these domains by extending NTK gradient flow to sequential mini-batches and temporal diference targets. By leveraging the smoothness and strong convexity of the closed-form NTK proxy (St-Arnaud et al., 2025), we guarantee training stability by choose time increments appropriately, and we replace local exploration heuristics with a mathematical program that optimizes across the feasible state space C in a single step.

## 3 Notation

In this section, we present the notation and definitions that will be useful for the rest of this article.

Consider an L-layer fully connected neural network parameterized by $\theta = \{ W ^ { ( l ) } \} _ { l = 0 } ^ { L - 1 }$ , where $W ^ { ( l ) } \in \mathbb { R } ^ { n _ { l + 1 } \times n _ { l } }$ The input dimension is $n _ { 0 } .$ , and $S ^ { n _ { 0 } - 1 }$ denotes the $n _ { 0 } .$ -dimensional sphere. Let $\alpha ^ { ( 0 ) } = x$ be the input. For a scalar activation function $\sigma : \mathbb { R }  \mathbb { R }$ applied component-wise, the forward pass is defined recursively for layers $l \in \{ 1 , \ldots , L \}$ as:

$$
\tilde { \alpha } ^ { ( l ) } = \frac { 1 } { \sqrt { n _ { l - 1 } } } W ^ { ( l - 1 ) } \alpha ^ { ( l - 1 ) } ,
$$

$$
\alpha ^ { ( l ) } = \sigma \left( \tilde { \alpha } ^ { ( l ) } \right)
$$

where $\tilde { \alpha } ( l )$ are the pre-activations, $\alpha ( l )$ are the post-activations, and $n _ { l }$ is the width of layer l.

Definition 1. The function $h : [ - 1 , 1 ] \to \mathbb { R }$ is defined as

$$
h ( x ) = { \frac { x \arcsin ( x ) + { \sqrt { 1 - x ^ { 2 } } } } { \pi } } + { \frac { x } { 2 } } .
$$

Definition 2. Adopting the notation of Jacot et al. $( 2 0 1 8 )$ , the functions $\Theta _ { \infty } ^ { ( l ) } : [ - 1 , 1 ] \to \mathbb { R }$ and $\rho ^ { ( L ) }$ $[ - 1 , 1 ] \to \mathbb { R }$ are defined recursively with base cases $\rho ^ { ( 1 ) } ( x ) = x$ and $\Theta _ { \infty } ^ { ( 1 ) } ( x ) = x$ as

$$
\begin{array} { l } { { \displaystyle { \Theta ^ { ( l ) } ( x ) = h ( \rho ^ { ( l - 1 ) } ( x ) ) } } } \\ { { \displaystyle { \Theta _ { \infty } ^ { ( l ) } ( x ) = \frac { l - 1 } { l } h ^ { \prime } ( \rho ^ { ( l - 1 ) } ( x ) ) \Theta _ { \infty } ^ { ( l - 1 ) } ( x ) + \frac { 1 } { l } \rho ^ { ( l ) } ( x ) . } } } \end{array}
$$

Definition 2 is a normalized version of the NTK limit kernel from Jacot et al. (2018). By abuse of notation, for any $g : [ - 1 , 1 ] \to \mathbb { R }$ and $m \times n$ matrix $A , g ( A )$ refers to the component-wise application of the function $g$ to the matrix A. Moreover, given a dataset $\{ x _ { i } \in S ^ { n _ { 0 } - 1 } \} _ { i = 1 } ^ { n }$ , the data matrix $X \in \mathbb { R } ^ { n \times n _ { 0 } }$ is formed by ${ \mathrm { s t a c k i n g ~ } } x _ { i } ^ { \top }$ along its rows. This yields the kernel matrices $\rho ^ { ( l ) } ( X X ^ { \top } )$ and $\Theta _ { \infty } ^ { ( l ) } ( X X ^ { \top } )$ ). For parameters $\theta = \{ \breve { W } ^ { ( l ) } \} _ { l = 0 } ^ { L - 1 }$ , the notation $\theta _ { t }$ refers to the parameters at time $t \geq 0$ . When clear from the context, any time-dependent mathematical object with subscript t refers to that object evaluated at time t.

## 4 Mini-batch training with the NTK

The typical setting for training with the NTK, namely NTK regression, uses the same dataset over the full training horizon. By exploiting the closed-form solution to the gradient flow dynamics (Jacot et al., 2018; Arora et al., 2019), one can obtain the output formula for arbitrary time t. Crucially, this solution implicitly assumes that the gradient is computed over the entire dataset simultaneously. However, practical neural network training updates parameters sequentially using batches of the dataset. Thus, when adapting this sequential updating procedure to its gradient flow equivalent, we can derive the output of an infinitely-wide neural network trained on successive data batches. Mathematically, given indices $b \in \{ 1 , \ldots , B \}$ of dataset batches, where $B \in \mathbb N$ , we formalize the evolution of the model output $f _ { t }$ via gradient flow and changing data batches in Definition $3 . { } ^ { 1 }$

Definition 3. Given a time horizon $T \in \mathbb { R } _ { > 0 }$ and a batch of data $\{ X ^ { ( b ) } \} _ { b \in [ B ] }$ where $[ B ] = \{ 1 , \dots , B \}$ , we call a sequence of stopping times any sequence of real numbers $0 = t _ { 1 } < \cdot \cdot \cdot < \dot { t } _ { B } = T$ such that the NTK at time $t \in [ t _ { b } , t _ { b + 1 } [ , b \in [ B - 1 ]$ evolves with respect to $X ^ { ( b ) }$ . Specifically, the dynamics of the gradient flow are given $b y$

$$
\begin{array} { r l } & { \partial _ { t } f _ { t } ( x ) = - \Theta _ { \infty } ^ { ( L ) } \left( x ^ { \top } x ^ { ( b ) \top } \right) \left( f _ { t } \left( X ^ { ( b ) } \right) - y ^ { * } \left( X ^ { ( b ) } \right) \right) , } \\ & { \qquad t \in [ t _ { b } , t _ { b + 1 } [ , b \in [ B - 1 ] } \end{array}
$$

where $y ^ { * } \left( X ^ { ( b ) } \right)$ are the target values over the batch $X ^ { ( b ) }$

In Section 5, we provide Proposition 1, which shows that in the infinite-width limit, the NTK $( \mathrm { i . e . ~ } \ \Theta _ { \infty } ^ { ( L ) } )$ does not change when replacing data batches at fixed intervals. As in Definition 3, the dynamics evolve with respect to the latest batch $X ^ { ( b ) }$ , however the kernel function $\Theta _ { t } ^ { ( L ) }$ does not vary $( \mathrm { i . e . , }$ is independent of t). Therefore, even with a constant kernel, the specific ordering and selection of data batches significantly influences the final output of the model at the end of the horizon (time T). We underline that while standard supervised mini-batch flow assumes fixed targets $y ^ { * }$ , RL introduces non-stationary TD targets by default. However, in alignment with common learning approaches, we fix the TD targets during training. We describe the resulting dynamics, which we refer to as the Temporal Diference Flow formulation, in Section 5 to extend the NTK framework to RL.

## 5 Extension to RL

A stated goal of our analysis of the NTK is to ofer a more realistic proxy to neural network training in order to prototype, benchmark, or even understand various RL algorithms. Prior work focuses on a fixed dataset throughout training and the diferential equation underlying the learning dynamics is taken to the infinite time limit. In practice, these assumptions are not met and to this end, we endeavour to adapt the NTK analysis to a setting with varying batches of data and a finite time training schedule. Meanwhile, a realistic proxy in the RL setting must incorporate the method by which TD targets are updated over time. In light of these requirements, we provide a diferential equation and its corresponding solution that satisfies the sought-after framework.

This section details key aspects of the NTK in the context of RL. First, in the infinite-width regime, the kernel stays constant throughout training, as highlighted in Proposition 1. Second, the training dynamics follow a specific diferential equation given by Definition 7, from which we obtain a closed-form solution in Theorem 1. Third, the kernel is shown to exhibit invertibility and induce certain smoothness properties in the training objective. Finally, we provide an iterative procedure to compute the closed-form solution of Theorem 1 for arbitrary inputs, and we analyze the computational cost derived from training with varying mini-batches and performing early stopping.

Preliminaries In the RL setting, we have to adapt the aforementioned sequential NTK framework to account for the diferent nature of the temporal diference objective that we optimize. In this work, we suppose that we have a deterministic environment. In other words, given a state s and an action a, we can compute the next state $s ^ { \prime }$ with probability 1. In contrast to the typical learning framework introduced in the previous section, the data points come in the form of tuples $( s , a , s ^ { \prime } , r )$ , where s is the state prior to the action selection, a is the action taken, $s ^ { \prime }$ is the state obtained by transitioning from s after taking action a, and r is the corresponding reward depending on $( s , a )$ . We can relate this to the typical learning notation of data $( x , y )$ by setting $x = ( s , a )$ and $y = r$ . Here, the vector x corresponds to the features of the data point, and y acts as the label. The value $\gamma \in ] 0 , 1 ]$ is the discount factor, while the parameterized action-value network is $Q _ { \theta _ { t } }$ . Given a target network frozen at time $\hat { t } ,$ the objective is thus to minimize the Euclidean 2-norm error for $\begin{array} { r } { t \ge \hat { t } : \left\| Q _ { \theta _ { t } } ( s , a ) - r ( s , a ) - \gamma \operatorname* { m a x } _ { a ^ { \prime } } Q _ { \theta _ { t } } ( s ^ { \prime } , a ^ { \prime } ) \right\| _ { 2 } . } \end{array}$

Constant NTK for RL Just as in the supervised setting, taking the infinite-width limit of these overparameterized networks allows us to recover a kernel formulation. Specifically, the NTK converges to a limiting kernel as the hidden layer widths go to infinity as shown next in Proposition 1.

Proposition 1. Let $d _ { t } : \mathbb { R } ^ { n _ { 0 } }  \mathbb { R } ^ { n _ { L } }$ be such that $\textstyle \int _ { 0 } ^ { T } \| d _ { t } \| _ { X } < \infty$ with probability 1 for some $T > 0$ and let σ be c-Lipschitz. The kernel $\Theta _ { t } ^ { ( L ) } ~ d e f i n e d$ by

$$
\Theta _ { t } ^ { ( L ) } ( x , x ^ { \prime } ) = \nabla _ { \theta _ { t } } Q _ { \theta _ { t } } ( x ) ^ { \top } \nabla _ { \theta _ { t } } Q _ { \theta _ { t } } ( x ^ { \prime } )
$$

is bounded on [0, T] in the limit $n _ { 1 } , \dotsc , n _ { L - 1 } \dotsc \infty$ . Moreover, it remains constant on the interval [0, T].

The proof of Proposition 1 follows a similar treatment as Theorem 2 in Jacot et al. (2018) and is provided in Appendix A. Building upon Proposition 1, we show that given a sequence of mini-batches, it is possible to obtain a closed-form solution by following the gradient flow equation on each time interval.

Flow of the Q-value function Before formalizing the main statement of this section as Theorem 1, we introduce some useful definitions.

Definition 4 (Inner product space induced by X). Let $X _ { i , \cdot } = x _ { i } ^ { \top } ~ f o r ~ i = 1 , \ldots , n$ . The inner product space induced by X is the space of functions $\mathbb { R } ^ { n _ { 0 } } \to \mathbb { R } ^ { n _ { L } }$ such that

$$
\langle f , g \rangle = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } f ( x _ { i } ) ^ { \top } g ( x _ { i } ) .
$$

The notation $\| f \| _ { X }$ refers to $\sqrt { \langle f , f \rangle }$

Definition 5. The function $\Phi _ { \kappa }$ given a kernel $\kappa : \mathbb { R } ^ { n _ { 0 } \times n _ { 0 } }  \mathbb { R } ^ { n _ { L } }$ on a dataset X of size n is defined by

$$
\begin{array} { r } { \Phi _ { \kappa } ( \delta ) : = \delta ( \kappa ( x , \cdot ) ) , } \end{array}
$$

where δ is any dual function with respect to the inner product space induced by the dataset X.

Definition 6. Let $X _ { i , . } = x _ { i } ^ { \top } ~ f o r ~ i = 1 , . . . , n$ . Let $\mathcal { N } ( x )$ be the neighbourhood of $x , i . e .$ , the admissible states that can be reached in one step from x, and define the function $\mu _ { t } ( x ) \in \mathcal { N } ( x )$ via

$$
\mu _ { t } ( x ) = \underset { x ^ { \prime } \in \mathcal { N } ( x ) } { \arg \operatorname* { m a x } } Q _ { t } ( x ^ { \prime } )
$$

and define

$$
\mu _ { t } \left( \boldsymbol { X } \right) _ { i } = \mu _ { t } ( \boldsymbol { x } _ { i } )
$$

over the dataset X by abuse of notation.

With the function space and neighborhood mechanics in place, we can now define the exact dynamics of the network under our batch-trained NTK framework.

Definition 7 (Temporal diference flow equation (TDF)). Given a sequence of stopping times $\{ t _ { 1 } , \dots , t _ { B } \}$ and $\hat { t } _ { b } \in \{ t _ { 1 } , \ldots , t _ { b } \}$ for all $b \in [ B - 1 ]$ , the solution $Q _ { t }$ to the flow equation for $t \in [ t _ { b } , t _ { b + 1 } [$ is given by

$$
\begin{array} { r } { \partial _ { t } Q _ { t } ( x ) = - \left( \Phi _ { \Theta _ { t } ^ { ( L ) } } ( \delta _ { t } ) \right) , } \end{array}
$$

where $\delta _ { t } ~ = ~ \langle d _ { t } , \cdot \rangle$ and $d _ { t } = Q _ { t } ( X ^ { ( b ) } ) - \gamma Q _ { \hat { t } _ { b } } \circ \mu _ { \hat { t } _ { b } } ( X ^ { ( b ) } ) - R ( X ^ { ( b ) } )$ , is called a solution to the temporal diference flow equation (TDF).

By solving the equation in Defintion 7, we obtain a closed-form solution that gives a proxy to the Q-value approximator of an overparameterized neural network in Theorem 1. In order to simplify notation in the Theorem, we first introduce the following definition:

Definition 8. We define the function $\Lambda _ { s , t } ^ { ( L ) } : \mathbb { R } ^ { n \times n _ { 0 } }  \mathbb { R } ^ { n \times n }$ as

$$
\Lambda _ { s , t } ^ { ( L ) } ( X ) = \left( \exp \left( - ( t - s ) \Theta _ { s } ^ { ( L ) } \left( X X ^ { \top } \right) \right) - I _ { n } \right) .
$$

Definition 8 is used to compute the closed-form solution of $Q _ { t }$ on a sub-interval $[ t _ { b } , t _ { b + 1 } [$ via Theorem 1. Theorem 1. Given a sequence of stopping times $\{ t _ { 1 } , \dots , t _ { B } \}$ and $Q _ { 0 }$ , the function $Q _ { t }$ has a closed-form solution $\mathbb { R } ^ { n _ { 0 } } \to \mathbb { R }$ given by the solution to the TDF (Definition 7) on each interval $[ t _ { i } , t _ { i + 1 } [ \ f o r \ i \in [ B - 1 ]$ Given $\hat { t } _ { b } \in \{ t _ { 1 } , \ldots , t _ { b } \}$ and an invertible $\Theta _ { \infty } ^ { ( L ) } \left( X X ^ { \top } \right)$ , the closed-form solution on $t \in [ t _ { b } , t _ { b + 1 } [$ is given by

$$
\begin{array} { r l } & { Q _ { t } ( x ) = \Theta _ { t } ^ { ( L ) } ( x ^ { \top } X ^ { ( b ) \top } ) \left( \Theta _ { t } ^ { ( L ) } \left( X ^ { ( b ) } X ^ { ( b ) \top } \right) \right) ^ { - 1 } } \\ & { \qquad \Lambda _ { t _ { b } , t } ^ { ( L ) } ( X ^ { ( b ) } ) \left( Q _ { t _ { b } } ( X ^ { ( b ) } ) - R ( X ^ { ( b ) } ) - \gamma Q _ { t _ { b } } \circ \mu _ { \hat { t } _ { b } } ( X ^ { ( b ) } ) \right) } \\ & { \qquad + Q _ { t _ { b } } ( x ) , } \end{array}
$$

where $\Theta _ { t } ^ { ( L ) } = \Theta _ { \infty } ^ { ( L ) }$ for all $t \geq 0$

The time t<sup>ˆ</sup> refers to a prior batch where the target network was frozen $( \mathrm { e . g . }$ , see Mnih et al. (2015)). The closed-form solution of Theorem 1 evolves with kernel $\Theta _ { \infty } ^ { ( L ) } \left( X ^ { ( b ) } X ^ { ( b ) \top } \right)$ on the interval $[ t _ { b } , t _ { b + 1 } [$ . By comparison with the supervised learning setting, Theorem 1 follows similar training dynamics within each intervals $[ t _ { b } , t _ { b + 1 } [ ;$ in contrast, the evolution of the solution sews together various pieces in a non-smooth continuous path. In that sense, the solution to to TDF is not a true gradient flow solution, but rather a right-derivative counterpart since the right-hand derivative of the solution exists at each discontinuity point t, while it is in fact diferentiable everywhere else. Below, we detail properties of the NTK that are useful in theoretical analysis.

Properties of $\Theta _ { \infty } ^ { ( L ) }$ We can show that the continuous-time TD error is ${ \boldsymbol n } ^ { ( b ) }$ -smooth with respect to t along any interval $[ t _ { b } , t _ { b + 1 } [ ,$ with $n ^ { ( b ) } = | \boldsymbol { X } ^ { ( b ) } |$ since the diagonal of $\Theta _ { \infty } ^ { ( L ) } \left( X ^ { ( b ) } X ^ { ( b ) \top } \right)$ consists of ones. It is also χ-strongly convex with respect to t on that interval if the smallest eigenvalue of $\Theta _ { \infty } ^ { ( L ) } \left( X ^ { ( b ) } X ^ { ( b ) \top } \right)$ is $\chi > 0$ . The following proposition (Proposition 2) states that for a depth $L \geq 2$ , the NTK is a positive definite kernel. In particular, ReLU activation and fully-connected networks yield a closed-form solution that is strongly convex and smooth on each interval.

Proposition 2 (from Jacot et al. (2018)). The NTK $\Theta _ { t } ^ { ( L ) }$ is positive definite with probability 1 for $L \geq 2$

Computation of closed-form solution When computing the closed-form solution in Theorem 1, by inspection of the equation we can compute $Q _ { t } ( x )$ by recursively compute $Q _ { t _ { b } }$ are prior stopping times. This yields the following decomposition:

$$
\begin{array} { r l } & { u ( x ) _ { b } ^ { \top } : = \Theta _ { \infty } ^ { ( L ) } \left( x ^ { \top } X ^ { ( b ) \top } \right) } \\ & { \qquad v _ { b } : = \left( \Theta _ { \infty } ^ { ( L ) } \left( X ^ { ( b ) } X ^ { ( b ) \top } \right) \right) ^ { - 1 } \Lambda _ { t _ { b } , t } ^ { ( L ) } \left( X ^ { ( b ) } \right) } \\ & { \qquad \left( Q _ { t _ { b } } \left( X ^ { ( b ) } \right) - R \left( X ^ { ( b ) } \right) - \gamma Q _ { \hat { t } _ { b } } \circ \mu _ { \hat { t } _ { b } } \left( X ^ { ( b ) } \right) \right) , } \end{array}
$$

from which $\begin{array} { r } { Q _ { t } ( x ) = \sum _ { 1 , \dots , b \mid t _ { b } \leq t } u ( x ) _ { b } ^ { \top } v _ { b } } \end{array}$ . The values $v _ { b }$ are kept in memory as they are computed, and for a new batch $X ^ { ( b ) }$ of data, the values $u ( x _ { i } ) _ { b }$ are computed for $x _ { i } \in X ^ { ( b ) }$ . Each new $u ( x _ { i } ) _ { b }$ is used to compute the new term $v _ { b }$ that is stored in memory. The time spent to compute $Q _ { t } ( x )$ is $\dot { O } \left( g ^ { 2 } ( t ) \right)$ , where

$g ( t ) = | \{ t _ { i } \ | \ t _ { i } \leq t \} | ^ { 2 }$ . This is much more eficient than training a deep neural network with extremely wide hidden layers, given the condition that their width is much larger than the number of data points in the replay bufer.

## 6 Solver-augmented exploration

The analytical proxy for neural network training dynamics that is described in section 5 provides a concise mathematical tool that can be queried for insights into its predictions. Deterministic environments that are suitable to description via constraints pave the way to new avenues for exploration in RL; we exploit the solution from Theorem 1 and formulate a search problem to gather new states that are stored in a bufer for subsequent learning.

Our algorithmic approach searches for a solution $x \in { \mathcal { C } }$ , where C is the set induced by constraints on the state-action space $S \times A$ and the variable x corresponds to the description of a state-action pair $x = ( s , a )$ In a grid world example, the variable x can be decomposed into variables sˆ and aˆ: $\hat { s } \in \mathbb R ^ { h \times w }$ (for a grid of size $h \times w )$ models each possible cell content, with diferent cases equivalent to diferent integers, while ${ \hat { a } } \in \mathbb { R }$ indexes the diferent action scenarios. Constraints are introduced to ensure that sˆ describes a a valid state. Given $t _ { b } ,$ , the objective is to find $x ^ { \ast } \in \mathcal { C }$ such that the estimate $Q _ { \infty } ( x ^ { * } )$ is maximized or minimized. This objective contributes to surprise whenever an optimal $x ^ { \ast } \in \mathcal { C }$ is not part of the bufer and the TD target of $x ^ { * }$ is not close to the estimate. In this case, the error is corrected during the next training step. We formulate the search problem $P _ { \omega } ^ { \pm } ( \lambda )$ , which is passed to a MINLP solver to find arg max $\cdot _ { x \in { \mathcal { C } } } Q _ { \infty } ( x ) ^ { 3 } ;$

$$
\begin{array} { c } { { \displaystyle \operatorname* { m i n } _ { x , y , u } \mp c _ { \omega } ^ { \top } u \quad \mathrm { s u b j e c t ~ t o } \mathrm { } } } \\ { { y = X x } } \\ { { \Theta _ { \infty } ^ { ( L ) } \left( X X ^ { \top } \right) u = \Theta _ { \infty } ^ { ( L ) } ( y ) } } \\ { { c _ { \omega } ^ { \top } u \in \mathbb { R } ^ { \pm } } } \\ { { x \in \mathcal { C } , } } \end{array}\tag{P<sup>±</sup><sub>ω</sub> (λ)}
$$

where the cost $c _ { w }$ corresponds to the TD target under the sample ω with initial value $R \left( X \right) + \gamma Q _ { 0 } \circ \mu _ { 0 } \left( X \right) \left( \omega \right)$ With $( P _ { \omega } ^ { \pm } ( \lambda ) )$ ), it is possible to recover a solution $x \in { \mathcal { C } }$ such that the Q-value is either maximized or minimized (±). The random variable $R \left( X \right) + \gamma Q _ { 0 } \circ \mu _ { 0 } \left( X \right)$ can also be replaced by its mean $R \left( X ^ { ( 1 ) } \right)$ as is the case in section 7.

The closed-form estimate $Q _ { \infty }$ is a proxy for a single batch update as we take $t \to \infty$ while holding the mini batch constant for the interval $[ 0 , \infty [$ , achieving perfect interpolation on the new mini-batch. Alternatively, we could model $Q _ { t }$ directly via the equation of Theorem 1 by introducing additional constraints for $t _ { 1 } , \dots , t _ { B }$ and the corresponding mini-batches $\mathbf { \tilde { \cal X } } ^ { ( 1 ) } , \ldots , { \cal X } ^ { ( B ) }$ . For simplicity and to make $P _ { \omega } ^ { \pm } ( \lambda )$ more compact, we focus on the first use case. We observe that $\Theta _ { \infty } ^ { ( L ) }$ is a convex function component-wise; we handle the nonconvex constraints using a mixed-integer nonlinear programming (MINLP) solver running a specialized branch-and-bound algorithm (Vigerske & Gleixner, 2018). With the mathematical program $P _ { \omega } ^ { \pm } ( \lambda )$ now defined, we introduce Algorithm 1.

Using the output of Algorithm 1, we can augment the initial dataset X to define a new generalizing function. Since at each iteration in the loop we augment the dataset with $x \in { \mathcal { C } }$ by selecting the point with the highest Q-value, we obtain a sequence of high predicted value candidates across the full set C. When solving $P _ { \omega } ^ { \pm }$ and once an optimal $x ^ { * }$ is found, the dataset X can be extended with $x ^ { * }$ . If Algorithm 1 is run with $m = | { \mathcal { C } } \setminus X |$ ， the final model would indeed interpolate perfectly the whole state space as $t \to \infty$ . While conceivable in theory, this is practically not achievable. However, the model gains a certain amount of surprise from the procedure at each step, as it finds the x believed to have highest value.

Algorithm 1 Solver-Augmented State-Space Exploration   
Input: Dataset X, small $\overline { { \lambda > 0 , } }$ m ∈ N   
Draw ω at random   
Draw $+ \mathrm { ~ o r ~ - ~ }$ case at random   
for m iterations do   
Solve $P _ { \omega } ^ { \pm } ( \lambda )$ with X   
Recover $x ^ { \ast } \in \mathcal { C }$ and observe $R ( x ^ { * } )$   
if TD error is high then   
$X  X \cup \{ x ^ { * } \}$   
end if   
end for   
return X, ω, ± case

Since the condition number of $\Theta _ { \infty } ^ { ( L ) } \left( X X ^ { \top } \right)$ can be high for large $n ,$ it can be useful to make a change of variable with $\delta _ { \omega }$ and rewrite the objective $c _ { \omega } ^ { \top }$ u as $d _ { \omega } ^ { \top } \Theta _ { L }$ after solving

$$
\Theta _ { \infty } ^ { ( L ) } \left( X X ^ { \top } \right) d _ { \omega } = c _ { \omega }\tag{1}
$$

and removing the constraint $\Theta _ { \infty } ^ { ( L ) } \left( X X ^ { \top } \right) u = \Theta _ { \infty } ^ { ( L ) } ( y )$ . Expression (1) only involves the variable $d _ { \omega }$ and if the equation is solved beforehand, the changed objective is dependent only on $\Theta _ { \infty } ^ { ( L ) } ( y ) \ ^ { 4 } .$ , for which both lower and upper bounds exist. Therefore, spatial branch-and-bound techniques will be able to exploit the restricted domains in their search. This simplification comes at the cost of solving (1) for $d _ { \omega } .$ , which is $O ( n ^ { 3 } )$ via Cholesky decomposition for a dense matrix. There also exist alternatives like, GMRES-based iterative refinement (Carson & Higham, 2018) or using higher precision floating-point numbers, that return a solution to (1) with high precision when the condition number is large.

## 7 Experiments

In this section, we perform experiments to demonstrate the usefulness of our NTK-based approach to model common RL algorithms. We focus on deep Q-learning, the simplest bufer-based algorithm whose update is modelled directly by the TD flow of Definition 7, but our method extends to other bufer-based algorithms. Prior work has already compared NTK-based approaches with neural networks (St-Arnaud et al., 2025; Lee et al., 2020; Yang et al., 2020) and highlighted the similitudes and diferences between the two. In order to bridge the divide, we endeavour to adjust the typical NTK framework to align more closely with empirical practice and study its behaviour directly. We divide the set of experiments along two axes. In the first axis, we study the efect of training on a sequence of mini-batches, rather than on a fixed dataset, where the mini-batches are drawn from a replay bufer whose contents drift as new data is collected. In the second axis, we study the sample eficiency of introducing a MINLP solver (Gurobi 13) that optimizes the closed-form NTK estimate of the Q-value in the exploration of the state-action space (i.e., Algorithm 1). All our experiments involve the deterministic environment DoorKey from the Minigrid library (Chevalier-Boisvert et al., 2023). This choice of environment satisfies the assumptions of our closed-form solution, namely deterministic transitions and a finite state-action space; its combinatorial structure, where the goal is reachable only after picking up a key and unlocking a door, makes exploration nontrivial. In order to alleviate the issue of the reward sparsity, we implemented a customized version of the environment, DenseDoorKey, which introduces rewards for subgoals. Details are provided in Appendix B.

## 7.1 Training with minibatch sampling

To evaluate the learning capability of the NTK approach, we study the training loss (i.e. Euclidean loss) as a function of the number of training steps for a deep Q-learning-type algorithm on the DenseDoorKey environment, where the closed-form solution of Theorem 1 replaces the network as the action-value estimator.

![](images/2e7f685d79c11a816a7b3ed24e983580c020d8f43462fb2c1c7d63493ef4f3d7.jpg)  
Figure 1: Training loss of the gradient flow equivalent to deep Q-learning (i.e. the NTK). Each iteration corresponds to a training step, i.e. the closed-form solution is updated via K mini-batches selected from the fixed replay bufer.

In order to stabilize training, we collect new data and then hold the replay bufer fixed while performing K mini-batch training steps, before resuming data collection; we repeat these two steps throughout training. The time increments for each batch are decreased using a typical stochastic gradient descent schedule with $\begin{array} { r } { O \left( \frac { 1 } { \sqrt { b } } \right) } \end{array}$ . When introducing fresh data, the schedule is restarted to prevent training collapse. Moreover, the target Q-values are kept frozen and updated periodically as in regular DQN. We plot the loss in Figure 1, where we observe that the training loss peaks after introducing new data, but that it quickly falls back to near-zero.

The two-step process and the scheduling idea are inspired by regular SGD training, since on a fixed dataset (or batch), the Euclidean loss of the output induced by the NTK of dimension $n \times n$ is both n-smooth and χ-strongly convex for some $\chi > 0$ . On a finite, yet large state-action space, each batch corresponds to a χ<sub>b</sub> parameter and χ can be taken to be $\begin{array} { r } { \chi = \operatorname* { m i n } _ { b \in [ B ] } \chi _ { b } > 0 } \end{array}$ . Figure 1 shows that the Q-values are adapting to the TD targets.

## 7.2 Solver-augmented exploration

In this section, we take advantage of the closed-form solution of infinitely-wide deep ReLU networks to the exploration of the deterministic grid environment. Given that the DoorKey environment can be described by constraints and integer variables, we can use the solver-augmented exploration approach of section 6 to select state-action pairs with high value. We provide additional details in Appendix C. During the exploration of the state-action space, we call Algorithm 1 as a callback to obtain state-action pairs that are added to the replay bufer. We used Gurobi 13, leveraging its nonlinear mixed-integer solving capabilities. In order to evaluate the sample eficiency of this approach, we compare of-policy RL algorithms with and without solver callback. In addition to DQN, we use a softmax version of the Bellman operator when rolling out the policy. We refer to this approach as SQN. All algorithms use the standard hyperparameters from the stable\_baselines3 codebase (Rafin et al., 2021). The solver is queried as a callback after each run of the exploratory policy. If a new state-action pair is discovered, it is added to the replay bufer and used in subsequent calls to the solver to build the internal mathematical model. We perform a total of 1000 learning steps and report the rollout policy score in Table 1. The rollout score is computed by evaluating total trajectory rewards using the learned policy.

<table><tr><td>Algorithm</td><td>Rollout score</td></tr><tr><td>DQN</td><td>0.0 (0.0)</td></tr><tr><td>DQN + solver</td><td>0.10 (0.07)</td></tr><tr><td>SQN</td><td>0.74 (0.24)</td></tr><tr><td>SQN + solver</td><td>1.02 (0.22)</td></tr></table>

Table 1: Average (std) rollout scores on various RL algorithms on 100 episodes in a $8 \times 8$ grid.

![](images/9c6352cf256d81b839ccbb1385b90fda4145379b4afc88654fbb93e30a78d252.jpg)  
Figure 2: Average rollout scores for SQN and SQN + solver after 1000 learning steps in a $8 \times 8$ grid. The standard deviations are computed from 10 training runs for each method.

In both cases, introducing the solver improves the rollout score. To complement Table 1, we compare policy evaluation curves of the best performing approach, SQN, with its solver-augmented version during rollouts in Figure 2. We used diferent training instances than in Table 1. We observe a steady increase in the performance as we perform more learning steps in both approaches, yet the addition of the solver improves the sample eficiency. While these results cannot be expected in all environments, the ability to search for high valued inputs can guide the exploration to new areas of the state-action space that are profitable. In addition, it becomes much easier to favour exploration by adding constraints that forbid the solver from returning state-action pairs it has already encountered (no-good cuts), or that force the next solution to difer from previous ones in a prescribed number of components (Hamming cuts). These constraints let us explore regions of C far from the solutions previously found. In problem instances where diversity is key, this capability is especially useful.

## 8 Conclusion

In this paper, we show how to apply the NTK to a reinforcement learning setting, allowing us to bridge the rich theory of overparameterized deep neural networks with of-policy learning algorithms like deep Qlearning. We demonstrate the ability to extend the NTK to varying batches of data that better align with the common practice of neural network training. Through the closed-form solution of infinitely-wide deep ReLU networks, we can also stabilize the training by satisfying typical convergence criteria in stochastic gradient descent, in combination with delayed data collection. Our approach serves as an introduction to the exploring the use of mini-batches and finite time evaluation in overparameterized neural networks. These ideas can be applied to other RL algorithms. As a future research direction, we highlight the possibility of understanding various bufer replay mechanisms and state exploration methodologies by studying the closed-form solution of the NTK. In addition, our solver-augmented exploration approach leaves a rich set of possibilities to be explored; it would be of particular interest to understand how to balance diversity with high value candidates during exploration. Moreover, a central idea is to understand how surprise afects the model predictions going forward.

## References

R Andreani, JM Martínez, and ML Schuverdt. On the relation between constant positive linear dependence condition and quasinormality constraint qualification. Journal of optimization theory and applications, 125(2):473–483, 2005.

Roberto Andreani, Gabriel Haeser, María Laura Schuverdt, and Paulo JS Silva. A relaxed constant positive linear dependence constraint qualification and applications. Mathematical Programming, 135(1):255–273, 2012.

Marcin Andrychowicz, Dwight Crow, Alex Ray, Jonas Schneider, Rachel Fong, Peter Welinder, Bob McGrew, Joshua Tobin, P. Abbeel, and Wojciech Zaremba. Hindsight experience replay. In Neural Information Processing Systems, 2017. URL https://api.semanticscholar.org/CorpusID:3532908.

Sanjeev Arora, Simon S Du, Wei Hu, Zhiyuan Li, Russ R Salakhutdinov, and Ruosong Wang. On exact computation with an infinitely wide neural net. Advances in neural information processing systems, 32, 2019.

Erin Carson and Nicholas J Higham. Accelerating the solution of linear systems by iterative refinement in three precisions. SIAM Journal on Scientific Computing, 40(2):A817–A847, 2018.

Fengdi Che, Chenjun Xiao, Jincheng Mei, Bo Dai, Ramki Gummadi, Oscar A Ramirez, Christopher K Harris, A. Rupam Mahmood, and Dale Schuurmans. Target networks and over-parameterization stabilize of-policy bootstrapping with function approximation. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 6372–6396. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/v235/che24a.html.

Zixiang Chen, Yuan Cao, Quanquan Gu, and Tong Zhang. A generalized neural tangent kernel analysis for two-layer neural networks. Advances in Neural Information Processing Systems, 33:13363–13373, 2020.

Maxime Chevalier-Boisvert, Bolun Dai, Mark Towers, Rodrigo Perez-Vicente, Lucas Willems, Salem Lahlou, Suman Pal, Pablo Samuel Castro, and Jordan Terry. Minigrid & miniworld: Modular & customizable reinforcement learning environments for goal-oriented tasks. In Advances in Neural Information Processing Systems 36, New Orleans, LA, USA, December 2023.

Douglas C Crowder, Darrien M McKenzie, Matthew L Trappett, and Frances S Chance. Hindsight experience replay accelerates proximal policy optimization. arXiv preprint arXiv:2410.22524, 2024.

Shirli Di-Castro, Shie Mannor, and Dotan Di Castro. Analysis of stochastic processes through replay bufers. In International Conference on Machine Learning, pp. 5039–5060. PMLR, 2022.

Ben Eysenbach, Russ R Salakhutdinov, and Sergey Levine. Search on the replay bufer: Bridging planning and reinforcement learning. Advances in neural information processing systems, 32, 2019.

William Fedus, Prajit Ramachandran, Rishabh Agarwal, Yoshua Bengio, Hugo Larochelle, Mark Rowland, and Will Dabney. Revisiting fundamentals of experience replay. In International conference on machine learning, pp. 3061–3071. PMLR, 2020.

Yasuhiro Fujita. Experience replay with random reshufling. arXiv preprint arXiv:2503.02269, 2025.

Guillaume Garrigos and Robert M Gower. Handbook of convergence theorems for (stochastic) gradient methods. arXiv preprint arXiv:2301.11235, 2023.

David Isele and Akansel Cosgun. Selective experience replay for lifelong learning. Proceedings of the AAAI Conference on Artificial Intelligence, 32(1), Apr. 2018. doi: 10.1609/aaai.v32i1.11595. URL https://ojs. aaai.org/index.php/AAAI/article/view/11595.

Arthur Jacot, Franck Gabriel, and Clément Hongler. Neural tangent kernel: Convergence and generalization in neural networks. Advances in neural information processing systems, 31, 2018.

Hyeonah Kim, Minsu Kim, Sungsoo Ahn, and Jinkyoo Park. Symmetric replay training: Enhancing sample eficiency in deep reinforcement learning for combinatorial optimization. International Conference on Machine Learning, pp. 24110–24136, 2024.

Jaehoon Lee, Sam S. Schoenholz, Jefrey Pennington, Ben Adlam, Lechao Xiao, Roman Novak, and Jascha Sohl-dickstein. Finite versus infinite neural networks:an empirical study. In NeurIPS 2020, 2020. URL https://arxiv.org/abs/2007.15801.

Ruishan Liu and James Zou. The efects of memory replay in reinforcement learning. In 2018 56th Annual Allerton Conference on Communication, Control, and Computing (Allerton), pp. 478–485, 2018. doi: 10.1109/ALLERTON.2018.8636075.

Clare Lyle, Mark Rowland, Will Dabney, Marta Kwiatkowska, and Yarin Gal. Learning dynamics and generalization in deep reinforcement learning. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (eds.), Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pp. 14560–14581. PMLR, 17–23 Jul 2022. URL https://proceedings.mlr.press/v162/lyle22a.html.

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602, 2013.

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. nature, 518(7540):529–533, 2015.

Takato Okudo and Seiji Yamada. Reward shaping with dynamic trajectory aggregation. In 2021 International Joint Conference on Neural Networks (IJCNN), pp. 1–9, 2021. doi: 10.1109/ijcnn52387.2021.9533401.

Shuang Qiu, Jieping Ye, Zhaoran Wang, and Zhuoran Yang. On reward-free RL with kernel and neural function approximations: Single-agent MDP and Markov game. In International Conference on Machine Learning, pp. 8737–8747. PMLR, 2021.

Antonin Rafin, Ashley Hill, Adam Gleave, Anssi Kanervisto, Maximilian Ernestus, and Noah Dormann. Stable-baselines3: Reliable reinforcement learning implementations. Journal of Machine Learning Research, 22(268):1–8, 2021. URL https://jmlr.org/papers/v22/20-1364.html.

Riley Simmons-Edler, Ben Eisner, Daniel Yang, Anthony Bisulco, Eric Mitchell, Sebastian Seung, and Daniel Lee. Reward prediction error as an exploration objective in deep rl. In Christian Bessiere (ed.), Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, IJCAI-20, pp. 2816–2823. International Joint Conferences on Artificial Intelligence Organization, 7 2020. doi: 10.24963/ijcai.2020/ 390. URL https://doi.org/10.24963/ijcai.2020/390. Main track.

William St-Arnaud, Margarida Carvalho, and Golnoosh Farnadi. Revisiting the neural tangent kernel: the role of large width and depth. arXiv preprint arXiv:2511.07272, 2025.

Stefan Vigerske and Ambros Gleixner. SCIP: Global optimization of mixed-integer nonlinear programs in a building block architecture. Optimization Methods and Software, 33(3):563–593, 2018. doi: 10.1080/ 10556788.2017.1335312.

Pan Xu and Quanquan Gu. A finite-time analysis of q-learning with neural network function approximation. In International conference on machine learning, pp. 10555–10565. PMLR, 2020.

Zhuoran Yang, Chi Jin, Zhaoran Wang, Mengdi Wang, and Michael I Jordan. On function approximation in reinforcement learning: optimism in the face of large state spaces. In Proceedings of the 34th International Conference on Neural Information Processing Systems, pp. 13903–13916, 2020.

Mingqi Yuan, Bo Li, Xin Jin, and Wenjun Zeng. Automatic intrinsic reward shaping for exploration in deep reinforcement learning. In The Fortieth International Conference on Machine Learning (ICML), 2023. doi: 10.48550/arxiv.2301.10886. URL https://arxiv.org/abs/2301.10886.

Kaiyan Zhao, Yiming Wang, Yuyang Chen, Yan Li, Xiaoguang Niu, et al. Eficient diversity-based experience replay for deep reinforcement learning. arXiv preprint arXiv:2410.20487, 2024.

## A Proof of Proposition 1

To prove Proposition 1, we require a generalized form of Grönwall’s lemma.

Lemma 1 (Generalized Grönwall’s lemma). Let $u , v \in C ^ { 1 } ( \left] a , b \right[ ) \cap C ( \left[ a , b \right] )$ and $F \in C ^ { 1 } ( \mathbb { R } )$ with $a < b$ Suppose that

$$
\begin{array} { r l } & { u ^ { \prime } ( t ) \leq F ( u ( t ) ) \quad t \in ] a , b [ } \\ & { v ^ { \prime } ( t ) = F ( v ( t ) ) \quad t \in ] a , b [ } \\ & { v ( 0 ) = u ( 0 ) . } \end{array}
$$

Then, u(t) satisfies

$$
u ( t ) \leq v ( t ) \quad t \in ] a , b [ .
$$

We can now prove Proposition 1. The proof strategy closely follows the proof of Theorem 2 from Jacot et al.   
(2018) and we make use of implicit index summation to simplify notation.

Proof of Proposition 1. We prove the statement for $L = 2$ and the extension to arbitrary depth $L > 2$ follows by induction over layers. First, we have

$$
\begin{array} { r l } & { \partial _ { t } \tilde { \alpha } ^ { ( l ) } = \Phi _ { \Theta _ { t } ^ { ( l ) } } \left( \langle d _ { t } ^ { ( l ) } , \cdot \rangle \right) } \\ & { \partial _ { t } W ^ { ( l ) } = \displaystyle \frac { 1 } { \sqrt { n _ { l } } } \langle \alpha ^ { ( l ) } , d _ { t } ^ { ( l + 1 ) } \rangle , } \end{array}
$$

where

$$
\begin{array} { r } { d _ { t } ^ { ( l ) } = \left\{ \begin{array} { l l } { d _ { t } } & { \mathrm { i f ~ } l = L } \\ { \left( \sigma ^ { \prime } \left( \tilde { \alpha } ^ { ( l ) } \right) \right) \odot \left( \left( \frac { 1 } { \sqrt { n _ { l } } } W ^ { ( l ) } \right) ^ { \top } d _ { t } ^ { ( l + 1 ) } \right) } & { \mathrm { i f ~ } l < L . } \end{array} \right. } \end{array}
$$

The subnetwork NTKs are

$$
\begin{array} { l } { { \Theta _ { t } ^ { ( 1 ) } = \displaystyle \frac { 1 } { n _ { 0 } } \left( \alpha ^ { ( 0 ) } \right) ^ { \top } \left( \alpha ^ { ( 0 ) } \right) \otimes { \cal I } _ { n _ { 1 } } } } \\ { { \Theta _ { t } ^ { ( 2 ) } = \displaystyle \frac { 1 } { n _ { 1 } } W ^ { ( 1 ) } \sigma ^ { \prime } \left( \tilde { \alpha } ^ { ( 1 ) } \right) \Theta _ { t } ^ { ( 1 ) } \sigma ^ { \prime } \left( \tilde { \alpha } ^ { ( 1 ) } \right) W ^ { ( 1 ) } } } \\ { { \phantom { \Theta _ { t } ^ { ( 2 ) } = \displaystyle \frac { 1 } { n _ { 1 } } \left( \alpha ^ { ( 1 ) } \right) ^ { \top } \left( \alpha ^ { ( 1 ) } \right) . } } } \end{array}
$$

We set

$$
\begin{array} { r } { \boldsymbol { w } ^ { ( l ) } ( t ) = \left. \frac { 1 } { \sqrt { n _ { l } } } \boldsymbol { W } ^ { ( l ) } ( t ) \right. _ { 2 } } \\ { \boldsymbol { a } ^ { ( l ) } ( t ) = \left. \frac { 1 } { \sqrt { n _ { l } } } \boldsymbol { \alpha } ^ { ( l ) } ( t ) \right. _ { \boldsymbol { X } } } \end{array}
$$

and we obtain the bounds

$$
\begin{array} { r l } & { \left\| d _ { t } ^ { ( 1 ) } \right\| _ { X } \leq c w ^ { ( 1 ) } ( t ) \left\| d _ { t } ^ { ( 2 ) } \right\| _ { X } } \\ & { \qquad = c w ^ { ( 1 ) } ( t ) \left\| d _ { t } \right\| _ { X } } \\ & { \left\| \Theta _ { t } ^ { ( 1 ) } \right\| _ { 2 } \leq ( a ^ { ( 0 ) } ( t ) ) ^ { 2 } } \\ & { \left\| \Theta _ { t } ^ { ( 2 ) } \right\| _ { 2 } \leq c ^ { 2 } ( w ^ { ( 1 ) } ( t ) ) ^ { 2 } \left\| \Theta _ { t } ^ { ( 1 ) } \right\| _ { 2 } + ( a ^ { ( 1 ) } ( t ) ) ^ { 2 } . } \end{array}
$$

We set

$$
\begin{array} { l } { \displaystyle \Delta _ { a } ^ { ( l ) } ( t ) = \left\| \frac { 1 } { \sqrt { n _ { l } } } \left( \tilde { \alpha } ^ { ( l ) } ( t ) - \tilde { \alpha } ^ { ( l ) } ( 0 ) \right) \right\| _ { X } } \\ { \displaystyle \Delta _ { w } ^ { ( l ) } ( t ) = \left\| \frac { 1 } { \sqrt { n _ { l } } } \left( W ^ { ( l ) } ( t ) - W ^ { ( l ) } ( 0 ) \right) \right\| _ { 2 } } \\ { \displaystyle A ( t ) = a ^ { ( 1 ) } ( 0 ) + \Delta _ { a } ^ { ( 1 ) } ( t ) + w ^ { ( 0 ) } ( 0 ) + \Delta _ { w } ^ { ( 0 ) } ( t ) } \\ { \displaystyle ~ + a ^ { ( 2 ) } ( 0 ) + \Delta _ { a } ^ { ( 2 ) } ( t ) + w ^ { ( 1 ) } ( 0 ) + \Delta _ { w } ^ { ( 1 ) } ( t ) . } \end{array}
$$

From these last definitions, we obtain the bounds

$$
\partial _ { t } \Delta _ { a } ^ { ( l ) } ( t ) \leq \frac { 1 } { \sqrt { n _ { l } } } \left. \Theta _ { t } ^ { ( l ) } \right. _ { 2 } \left. d _ { t } ^ { ( l ) } \right. _ { X }
$$

$$
\partial _ { t } \Delta _ { w } ^ { ( l ) } ( t ) \leq \frac { 1 } { \sqrt { n _ { l } } } a ^ { ( l ) } ( t ) \left. d _ { t } ^ { ( l + 1 ) } \right. _ { X } .
$$

We use these bounds to bound the derivative $\partial _ { t } A ( t )$ with

$$
\partial _ { t } A ( t ) \leq \frac { 1 } { \sqrt { n _ { 1 } } } q \left( w ^ { ( 0 ) } ( t ) , w ^ { ( 1 ) } ( t ) , a ^ { ( 1 ) } ( t ) , a ^ { ( 2 ) } ( t ) \right) \| d _ { t } \| _ { X }
$$

for some polynomial $q$ with positive coeficients. Observe that $A ( 0 )$ is stochastically bounded as $n _ { 1 }  \infty ,$ Using Lemma 1 and using the fact that q is bounded and continuous over $[ 0 , T ]$ , we obtain $A ( t )$ uniformly bounded over [0, T]. This implies that $\partial _ { t } \boldsymbol { A } ( t ) \ \to \ \boldsymbol { 0 }$ as $n _ { 1 } ~  ~ \infty$ , which in turn implies $A ( t ) \ \to \ A ( 0 )$ The boundedness condition on $A ( t )$ implies that $\Theta _ { \mu } ^ { ( 2 ) }$ remains bounded on [0, T]. To show that it remains constant, we study the derivatives $\partial _ { t } \Theta _ { t } ^ { ( 2 ) }$

$$
\partial _ { t } \Theta _ { t } ^ { ( 2 ) } ( x , x ^ { \prime } ) = \nabla _ { \theta _ { t } } \Theta _ { t } ^ { ( 2 ) } ( x , x ^ { \prime } ) ^ { \top } \partial _ { t } \theta _ { t }
$$

and we derive bounds for the norms of $\nabla _ { \theta _ { t } } \Theta _ { t } ^ { ( 2 ) } ( x , x ^ { \prime } )$ and $\partial _ { t } \theta _ { t }$ :

$$
\left\| \partial _ { W ^ { ( 1 ) } } \Theta _ { t } ^ { ( 2 ) } ( x , x ^ { \prime } ) \right\| _ { 2 } \leq \frac { 2 } { \sqrt { n _ { 1 } } } c ^ { 2 } \left\| \Theta _ { t } ^ { ( 2 ) } \right\| _ { 2 } w ^ { ( 1 ) } ( t )\tag{2}
$$

$$
\left\| \partial _ { W ^ { ( 0 ) } } \Theta _ { t } ^ { ( 2 ) } ( x , x ^ { \prime } ) \right\| _ { 2 } \leq \frac { 2 } { \sqrt { n _ { 1 } } } c \operatorname* { s u p } _ { z \in \hat { \cal N } } \left\| z \right\| _ { 2 } a ^ { ( 1 ) } ( t ) ,\tag{3}
$$

where $\textstyle { \hat { N } } = \bigcup _ { i } \{ x _ { i } \} \cup { \mathcal { N } } ( x _ { i } )$ . As $n _ { 1 } \to \infty$ , we conclude that both (2) and (3) converge to 0 because of the assumption $\left| \hat { \mathcal { N } } \right| < \infty$ . Finally, since $A ( t )$ is bounded, the same can be said of $\partial _ { t } \theta$ as $n _ { 1 } \to \infty$ □

## B DenseDoorKey Environment

In the DoorKey gridworld environment, an agent must reach a target state located behind a locked door, which requires first finding and picking up a key. A non-zero reward $r _ { t } ~ > ~ 0$ is obtained strictly upon reaching the final target state at terminal step $T ,$ whereas all intermediate transitions $( t < T )$ yield $r _ { t } = 0$ To overcome this reward sparsity, we introduce a new reward function that targets subgoals. This is a common trick in RL and it has been demonstrated to help with sample eficiency (Okudo & Yamada, 2021). In other cases, intrinsic rewards can also alleviate the sparse reward problem (Yuan et al., 2023). The subgoals the we introduce are: 1) picking up the key, and 2) unlocking the door, to which we attribute a reward of 0.2 and 0.4 respectively. In addition, we include intermediate rewards based on the Manhattan distance $( l _ { 1 } ) \colon$ : consider an agent in position $p _ { t }$ who, after an action, transitions to position $p _ { t + 1 } ;$ then, they obtain a reward equal to $r _ { t + 1 } = 0 . 0 1 \cdot \Delta l _ { 1 } ( p _ { t } , p _ { t + 1 } , g )$ where $g$ is the position of the next immediate subgoal and $\Delta l _ { 1 } ( p _ { t } , p _ { t + 1 } , g ) = ( l _ { 1 } ( p _ { t } , g ) - l _ { 1 } ( p _ { t + 1 } , g ) )$ . Note that these intermediary rewards are positive when the agent moves closer to g and negative otherwise. The reward for reaching the terminal goal (green cell) is 1. We refer to the reader to Figure 3 for an illustration of the reward mechanism on a $5 \times 5$ grid.

![](images/52eb93ec7a005c1d3e6a0e2fd169e5c009a57d0653b6d5924998420e6cba9a88.jpg)  
(a) Initial grid potential −0.01 × l<sub>1</sub>(p<sub>0</sub>, g) = −0.02

![](images/836ec918a00b6e966d19b77cadc3fc2be191bd4e2fb3ddb693a67e51fb1c0c4d.jpg)  
(b) Step toward $g = \mathrm { K e y } \ r _ { 1 } = 0 . 0 1 \times \Delta l _ { 1 } ( p _ { 0 } , p _ { 1 } , g )$ = +0.01

![](images/c11ef20324da8add29232b7fd35780b61e02ff17d2efb19eb05fcc9b03a5f3ef.jpg)  
(c) Subgoal 1: Pickup Key $r _ { \mathrm { s u b g o a l } } = + 0 . 2$

![](images/23f3d3f1ea7dfd1e927243353f57f8580e06e343e35918120cd3044911b98497.jpg)  
(d) Subgoals 2: Unlock door $r _ { \mathrm { s u b g o a l } } = + 0 . 4$  
Figure 3: Sequence of rewards in the DenseDoorKey environment. The agent is represented in red. The agent receives incremental potential-based distance rewards towards active targets, along with sparse intermediate subgoal rewards upon picking up the key (+0.2), and unlocking the door (+0.4).

## C Mathematical model used by the solver

In reinforcement learning, active exploration traditionally relies on local, step-by-step environment rollouts or random trajectory sampling during training. In contrast, our mathematical optimization model $P _ { \omega } ^ { \pm } ( \lambda )$ reformulates active exploration as a global search problem over the environment’s state-action space. Our objective is to retrieve a high-value state $x = ( s , a )$ by finding the corresponding grid and action. In this section, we detail the implementation of $P _ { \omega } ^ { \pm } ( \lambda )$ within the RL training procedure. In this section, we describe in detail how $P _ { \omega } ^ { \pm } ( \lambda )$ is implemented. Specifically, we detail for the Minigrid environment the constraints defining the set ${ \mathcal { C } } ,$ and we also specify the explicit formulation of the constraints involving the kernel $\Theta _ { \infty } ^ { ( L ) }$ . To simplify notation, we let $[ m ] = \{ 1 , \dots , m \}$ , where $m \in \mathbb { N }$ , and we let $[ m ] _ { k } = \{ k , \dots , m \}$ , where $k \leq m , k \in \mathbb { N }$ In our Minigrid environment, the grid dimensions are $h \times w$ , where h denotes the height and w denotes the width.

Every Minigrid environment follows the convention that grids are tensors of shape $( h , w , 3 )$ . For any cell, the three values along the final dimension encode its object type, object colour, and state, respectively. For DoorKey grids, these values are mapped as follows:

$$
\begin{array} { r l } & { \mathrm { o b j e c t }  \{ \mathrm { u n s e e n } : 0 , \mathrm { e m p t y } : 1 , \mathrm { w a l l } : 2 , \mathrm { d o o r } : 4 , } \\ & { \mathrm { ~ k e y } : 5 , \mathrm { g o a l } : 8 , \mathrm { a g e n t } : 1 0 \} } \\ & { \mathrm { c o l o u r }  \{ \mathrm { r e d } : 0 , \mathrm { g r e e n } : 1 , \mathrm { y e l l o w } : 4 , \mathrm { g r e y } : 5 \} } \\ & { \mathrm { s t a t e }  \{ \mathrm { o p e n } : 0 , \mathrm { c l o s e d } : 1 , \mathrm { l o c k e d } : 2 \} . } \end{array}
$$

Available actions follow the convention

$$
\begin{array} { r l r } & { } & { \mathrm { t u r n ~ l e f t }  0 } \\ & { } & { \mathrm { t u r n ~ r i g h t }  1 } \\ & { } & { \mathrm { m o v e ~ f o r w a r d }  2 } \\ & { } & { \mathrm { p i c k u p ~ k e y }  3 } \\ & { } & { \mathrm { u n l o c k ~ d o o r }  5 . } \end{array}
$$

## C.1 Grid constraints

For each grid cell $( i , j ) \in [ h ] \times [ w ]$ and $k \in [ 3 ]$ , we introduce a variable $s _ { i , j , k } \in \mathbb { N }$ within

$$
s _ { i , j , k } \in \left\{ { \begin{array} { l l } { \left[ 0 , 1 0 \right] } & { { \mathrm { i f ~ } } k = 1 } \\ { \left[ 0 , 5 \right] } & { { \mathrm { i f ~ } } k = 2 } \\ { \left[ 0 , 2 \right] } & { { \mathrm { i f ~ } } k = 3 . } \end{array} } \right.
$$

In simple words, these variables serve to describe the feature tuple $\left( { { s } _ { i , j , 1 } } , { { s } _ { i , j , 2 } } , { { s } _ { i , j , 3 } } \right)$ of cell $( i , j )$ , where component $s _ { i , j , 1 }$ denotes the object type, $s _ { i , j , 2 }$ the object color, and $s _ { i , j , 3 }$ the state..

We also introduce a variable $a \in \{ 0 , 1 , 2 , 3 , 5 \}$ for the available actions. We add big-M constraints to determine the door and key positions. We also introduce binary variables $d _ { i , j } , k _ { i , j }$ , for $( i , j ) \in [ h ] \times [ w ]$ , indicating the presence of a door or key in cell $( i , j )$ , respectively. With these variables, we add constraints to ensure that there is one door and at most one key on the grid .

$$
\begin{array} { l } { { \displaystyle s _ { i , j , 1 } \leq 4 + 1 0 \times ( 1 - d _ { i , j } ) } } \\ { { \displaystyle s _ { i , j , 1 } \geq 4 - 1 0 \times ( 1 - d _ { i , j } ) } } \\ { { \displaystyle \sum _ { ( i , j ) } d _ { i , j } = 1 } } \\ { { \displaystyle s _ { i , j , 1 } \leq 5 + 1 0 \times ( 1 - k _ { i , j } ) } } \\ { { \displaystyle s _ { i , j , 1 } \geq 5 - 1 0 \times ( 1 - k _ { i , j } ) } } \\ { { \displaystyle \sum _ { ( i , j ) } k _ { i , j } \leq 1 . } } \end{array}\tag{door in cell}
$$

(exactly one door)

(key in cell)

(at most one key)

We fix the cell at position $( h - 1 , w - 1 )$ to be a green goal square. In other words,

$$
s _ { h - 1 , w - 1 , 1 } = 8 , \quad s _ { h , w , 2 } = 1 , \quad s _ { h , w , 3 } = 0 .
$$

We introduce the binary variables agen $\mathsf { t } _ { i , j } \in \{ 0 , 1 \}$ for $( i , j ) \in [ h - 1 ] _ { 2 } \times [ w - 1 ] _ { 2 }$ to determine the agent’s position. In order to have only one agent on the grid, we add the constraint

$$
\sum _ { i = 2 } ^ { h - 1 } \sum _ { j = 2 } ^ { w - 1 } \mathrm { a g e n t } _ { i , j } = 1 .
$$

To determine if each cell is empty, we introduce the variables empty $\mathbf { \chi } _ { i , j } \in \{ 0 , 1 \}$ for $( i , j ) \in [ h - 1 ] _ { 2 } \times [ w - 1 ] _ { 2 }$ We introduce the variables wall $_ { i , j } \in \{ 0 , 1 \}$ for $( i , j ) \in [ h - 1 ] _ { 2 } \times [ \stackrel { \sim } { w } - 2 ] _ { 3 }$ to determine the presence of a wall. We note that DoorKey environments generate a separating wall containing a door at random between columns $j = 3$ and $j = w - 2$ . We introduce the constraints

$$
\mathrm { a g e n t } _ { i , j } \leq 1 - \mathrm { w a l l } _ { i , j }
$$

(agent not on wall)

to ensure the agent is not on a wall. The variables $\mathrm { c o l } _ { j }$ for $j \in [ w - 2 ] _ { 3 }$ correspond to the presence of a separating wall in column j. The following constraints ensure 1) that there is one separating wall and 2) the separating wall consists only of wall cells or a door cell.

$$
\sum _ { j \in [ w ] } \mathrm { c o l } _ { j } = 1\tag{1}
$$

$$
\mathrm { w a l l } _ { i , j } + d _ { i , j } = \mathrm { c o l } _ { j }\tag{2}
$$

The agent orientation at cell $( i , j ) \in [ h - 1 ] _ { 2 } \times [ w - 1 ] _ { 2 }$ is determined through variables east $_ { i , j } \in \{ 0 , 1 \}$ , sout $\mathbf { \sigma } _ { 1 i , j } \in \{ 0 , 1 \}$ , wes $\hphantom { } _ { i , j } \in \{ 0 , 1 \}$ , and north $_ { \cdot i , j } \in \{ 0 , 1 \}$ . We further require constraints for $( i , j ) \in [ h - 1 ] _ { 2 } \times$ $[ w - 2 ] _ { 3 }$ to ensure that the agent is not on the key:

$$
\begin{array} { r } { \mathrm { a g e n t } _ { i , j } \le 1 - k _ { i , j } . } \end{array}
$$

The following constraint ensure the agent is not on the goal:

$$
\mathrm { a g e n t } _ { h - 1 , w - 1 } = 1 - \mathrm { g o a l } _ { h - 1 , w - 1 } .
$$

The agent orientation variables can only be active if there is an agent on the same cell:

$$
\begin{array} { r } { \mathrm { a g e n t } _ { i , j } = \mathrm { e a s t } _ { i , j } + \mathrm { s o u t h } _ { i , j } + \mathrm { w e s t } _ { i , j } + \mathrm { n o r t h } _ { i , j } . } \end{array}
$$

We also fix the boundary cells of the grid with $s _ { i , j , 1 } = 2 , s _ { i , j , 2 } = 5 , s _ { i , j , 3 } = 0$ . For each grid cell not on the boundary, we add two sets of constraints to determine its colour. The first set involves indices $( i , j ) \in [ h - 1 ] _ { 2 } \times [ w - 2 ] _ { 3 }$ , corresponding to cells where the separating wall can lie:

$$
\begin{array} { r } { s _ { i , j , 1 } = 5 \times k _ { i , j } + 2 \times \mathrm { w a l l } _ { i , j } \quad } \\ { + 4 \times d _ { i , j } + 1 0 \times \mathrm { a g e n t } _ { i , j } + \mathrm { e m p t y } _ { i , j } . } \end{array}
$$

The second set of constraints involves indices $( i , j ) \in [ h - 1 ] _ { 2 } \times \{ 2 , w - 1 \}$ , where the separating wall cannnot lie:

$$
\begin{array} { r } { s _ { i , j , 1 } = 5 \times k _ { i , j } + 1 0 \times \mathrm { a g e n t } _ { i , j } + \mathrm { e m p t y } _ { i , j } . } \end{array}
$$

Finally, we introduce binary variables to model the door being open, closed, locked, and the key being carried. These variables are $\mathrm { o p e n } _ { i , j } \in \{ 0 , 1 \}$ , close $\mathrm { l } _ { i , j } \in \{ 0 , 1 \}$ , $\mathrm { o c k e d } _ { i , j } \in \{ 0 , 1 \}$ . For $( i , j ) \in [ h - 1 ] _ { 2 } , \times [ w - 2 ] _ { 3 }$ , we enforce the door to be in one of its three possible states if present in cell $( i , j )$

$$
d _ { i , j } = \mathrm { o p e n } _ { i , j } + \mathrm { c l o s e d } _ { i , j } + \mathrm { l o c k e d } _ { i , j } .
$$

We introduce a constraint that enforces the door to be locked if the agent does not carry the key, and a second constraint that forces to agent to be on the left-hand side of the door if it is not open; if the door is open, the agent can be on the right-hand side of the door.

$$
\begin{array} { l } { { \displaystyle \sum _ { i = 2 } ^ { h - 1 } \sum _ { j = 3 } ^ { w - 2 } \mathrm { l o c k e d } _ { i , j } \geq \sum _ { ( i , j ) } k _ { i , j } } } \\ { { \displaystyle \sum _ { ( k , l ) \mid l \geq j } \mathrm { a g e n t } _ { k , l } \leq 1 - \mathrm { c o l } _ { j } + \sum _ { ( k , l ) } \mathrm { o p e n } _ { k , l } } } \end{array} \qquad j \in [ w - 2 ] _ { 3 } .
$$

The second equation ensures that if the door is locked, the agent can only be found on the left-hand side of the separating wall. Otherwise, it is free to be on either side. The remaining grid constraints fix the state of the grid cells $( i , j ) \in [ h - 1 ] _ { 2 } \times [ w - 2 ] _ { 3 } \ :$ , i.e. a separating wall can be found:

$$
\begin{array} { l } { { s _ { i , j , 3 } = 1 \times \mathrm { c l o s e d } _ { i , j } + 2 \times \mathrm { l o c k e d } _ { i , j } } } \\ { { \qquad + 3 \times \mathrm { n o r t h } _ { i , j } + 2 \times \mathrm { w e s t } _ { i , j } + \mathrm { s o u t h } _ { i , j } } } \end{array}
$$

and of the cells $( i , j ) \in [ h - 1 ] _ { 2 } \times \{ 2 , w - 1 \}$ , i.e. outside the separating wall range of possible columns:

$$
s _ { i , j , 3 } = 3 \times \mathrm { n o r t h } _ { i , j } + 2 \times \mathrm { w e s t } _ { i , j } + \mathrm { s o u t h } _ { i , j }
$$

All the variables and constraints defined in this section form the variable vector x and the constraint set C appearing in $P _ { \omega } ^ { \pm } ( \lambda )$

## C.2 Kernel constraints

In this section, we introduce the remaining variables and constraints derived from $P _ { \omega } ^ { \pm } ( \lambda )$ . The additional variables are

$$
\begin{array} { r l } & { \qquad y _ { i } \in \mathbb { R } , } \\ & { \qquad \bar { e } _ { i } \in \mathbb { R } , } \\ & { \mathrm { n o r m } \in \mathbb { R } _ { \geq 0 } } \\ & { \mathrm { a s i n } _ { l , i } \in \left[ - \frac { \pi } { 2 } , \frac { \pi } { 2 } \right] , } \\ & { \qquad \rho _ { l , i } \in \left[ L \right] , i \in \left[ n \right] } \\ & { \qquad \rho _ { l , i } ^ { \prime } \in \left[ - 1 , 1 \right] } \\ & { \qquad \theta _ { l , i } \in \left[ - 1 , 1 \right] , } \\ & { \qquad \quad \theta _ { l , i } \in \left[ L \right] , i \in \left[ n \right] , } \end{array}
$$

(embedding)

(normalization)

where variable y corresponds to the variable $y = X x$ in $P _ { \omega } ^ { \pm } ( \lambda )$ . We include constraint to model the data in $S ^ { n _ { 0 } - 1 }$ :

$$
\begin{array} { c } { x = \mathrm { c o n c a t } \left( \operatorname { v e c } ( s ) , a \right) } \\ { \mathrm { n o r m } ^ { 2 } = \| x \| _ { 2 } ^ { 2 } } \\ { y _ { i } = x _ { i } ^ { \top } x } \\ { \bar { e } _ { i } \times \mathrm { n o r m } = \frac { y _ { i } } { \| x _ { i } \| _ { 2 } } } \end{array}\tag{dot product}
$$

(normalization)

and constraints to model the kernel:

$$
\begin{array} { c c c } { { \rho _ { 1 , i } = \overline { { { \epsilon } } } _ { i , j } } } & { { \mathrm { ~ } i \in [ n ] } } \\ { { \rho _ { l , i } = \mathrm { a i n } _ { l , i } \times \displaystyle \frac { \sin ( \mathrm { a s i n } _ { l , i } ) } { \pi } } } & { { } } & { { } } \\ { { \displaystyle ~ + \frac { \cos ( \mathrm { a s i n } _ { l , i } ) } { \pi } + \frac { \sin ( \mathrm { a s i n } _ { l , i } ) } { 2 } , ~ } } & { { } } & { { ~ l \in [ L ] _ { 2 , ~ } i \in [ n ] } } \\ { { \rho _ { l , i } ^ { \prime } = \displaystyle \frac { \sin ( \mathrm { a s i n } _ { l , i } ) } { \pi } + \frac { 1 } { 2 } , ~ } } & { { } } & { { ~ l \in [ L ] _ { 2 , ~ } i \in [ n ] } } \\ { { \mathrm { s i n } ( \mathrm { a s i n } _ { l , i } ) = \rho _ { l - 1 , i } , ~ } } & { { } } & { { ~ l \in [ L ] _ { 2 , i } \in [ n ] } } \\ { { \Theta _ { 1 , i } = \rho _ { 1 , i } , ~ } } & { { } } & { { ~ i \in [ n ] } } \\ { { \Theta _ { l , i } = \displaystyle \frac { l - 1 } { l } \rho _ { l , i } ^ { \prime } \Theta _ { l - 1 , i } + \frac { 1 } { l } \rho _ { l , i } , ~ } } & { { } } & { { ~ l \in [ L ] _ { 2 , i } \in [ n ] } . } \end{array}
$$

The variables $\Theta _ { L , i }$ model the term $\Theta _ { \infty } ^ { ( L ) } ( y )$ in $P _ { \omega } ^ { \pm } ( \lambda )$ through the equations

$$
\Theta _ { L , i } = \Theta _ { \infty } ^ { ( L ) } \left( \frac { x _ { i } ^ { \top } x } { \| x _ { i } \| _ { 2 } \| x \| _ { 2 } } \right) ,
$$

where $i \in [ n ]$ . We remark that the matrix $\Theta _ { \infty } ^ { ( L ) } \left( X X ^ { \top } \right)$ is known beforehand and is therefore a matrix of coeficients. In addition, the variables asi $_ { 1 l , i }$ model the arcsin terms in the definition of $\rho ^ { ( l ) }$ . This is achieve through the constraint type sin $. ( \alpha ) = \beta$ for variables α and $\beta ,$ where the relationship arcsin $( \beta ) = \alpha$ is enforced. Moreover, functions $\bar { \rho } ^ { ( l ) } ( x )$ contain terms $\sqrt { 1 - x ^ { 2 } }$ , which can lead to numerical issues when computing their derivative with respect to x near −1 or 1. Instead, we model these terms with constraints $\mathrm { c o s } \big ( \mathrm { a s i n } _ { l . i } \big )$ , since ${ \sqrt { 1 - \sin ^ { 2 } ( \arcsin _ { l , i } ) } } = \cos ( \arcsin _ { l , i } )$ . The derivative of the cos function with respect to its input is bounded on its domain, thus numerically stable.

## D Additional Experimental Details

In our experiments, we train on a sequence of mini-batches $\{ X ^ { ( b ) } \} _ { b = 1 } ^ { B }$ . The default approach in RL is to train on data selected from the replay bufer uniformly at random, or through some prioritized replay mechanism. In these instances, the replay bufer data goes through a distribution shift. In our experiments, we noticed that by freezing exploration following the introduction of new data, we obtained a more stable training when we introduced decreasing time increments. This idea was motivated by the literature on stochastic gradient descent (SGD), where, given a fixed dataset, a schedule of decreasing learning rates guarantees convergence to a global minimum<sup>5</sup> in the n-smooth and χ-strongly convex setting (Garrigos & Gower, 2023). Because our overparameterized networks satisfy these convergence assumptions, we could expect this behaviour on a fixed dataset if batches are selected uniformly at random. However, distribution shift breaks the uniformity assumption by giving more weight to tuples obtained via an older policy or a prioritized sampling. For this reason, we periodically collect new samples and between each data collection phase, we perform batch training through the closed-form solution to the TDF (Theorem 1). The training procedure is illustrated in Figure 4.

In addition to the optimization model described in Appendix C, we summarize the callback loop used during the solver-augmented exploration in Figure 5. While we made minimal use of diversity constraints (i.e. no-good cuts and hamming cuts), the solver-enhanced approach allows for further exploration of diversityrelated ideas that are not available without the closed-form solution of the NTK. We compute the average running time of the solver callback procedure, i.e. the time to solve the optimization problem. The average time was 0.097 seconds with a standard deviation of 0.028 seconds.

![](images/7bf77c2f2785e0229b0a322cc8ea16ca7fc20eb3d61e2919d06cfca89fb9e435.jpg)  
Figure 4: Piecewise continuous NTK gradient flow across mini-batch transitions. At each stopping time $t _ { b } .$ mini-batch replacement causes a loss spike due to distribution shift, followed by exponential decay governed by the local strong convexity constant $\chi _ { b } = \lambda _ { \mathrm { m i n } } \left( \Theta _ { \infty } ^ { ( L ) } \left( X ^ { ( b ) } X ^ { ( b ) \top } \right) \right) > 0 .$

![](images/91b07ef12509a6142e58f08adbaba7020cd20a8dc664ec6ef4fffca6769353de.jpg)  
Solver-Augmented Exploration Callback Loop  
Figure 5: Architecture of the solver-augmented exploration callback loop. The analytical NTK proxy constructs a smooth, strongly convex value landscape from bufer mini-batches. The MINLP solver optimizes over this landscape subject to No-Good cuts, identifying high-error state-action targets $( s ^ { * } , a ^ { * } )$ for environment rollouts.

## E Necessity of KKT conditions

In this section, we show that the KKT conditions for $P _ { \omega } ^ { \pm } ( \lambda )$ are necessary for a local optimum with respect to the primal variables (e.g., x and y). We begin by providing the constraint qualification required in the proof of necessity (Andreani et al., 2012).

Definition 9 (RCPLD). Denote the equality constraints with $h _ { i } ( x ) = 0 , i \in \mathcal { T }$ and the inequality constraints $g _ { j } ( x ) \leq 0 , j \in \mathcal { I }$ , and let $\mathcal { I } ( x ^ { \star } ) = \{ j \in \mathcal { I } : g _ { j } ( x ^ { \star } ) = 0 \}$ be the set of constraints active at $x ^ { \star }$ . The relaxed constant positive linear dependence (RCPLD) condition holds if there exists a neighbourhood $B ( x ^ { \star } , \delta ) = \{ x :$ $\| x - x ^ { \star } \| _ { 2 } < \delta \}$ for some $\delta > 0$ such that 1) the set $\{ \nabla h _ { i } ( x ) | i \in \mathcal { T } \}$ has constant rank $f o r \ x \in B ( x ^ { \star } , \delta )$ and 2) for every basis of $\{ \nabla h _ { i } ( x ^ { \star } ) \mid \mathcal { T } \}$ with index set $I \subseteq \mathbb { Z }$ and for any subset $J \subseteq { \mathcal { I } } ( x ^ { \star } )$ , the following

implication is satisfied:

$$
\begin{array} { r l } & { \exists ( \lambda , \mu ) \neq 0 , \ \mu \geq 0 \ : \ \displaystyle \sum _ { i \in I } \lambda _ { i } \nabla h _ { i } ( x ^ { \star } ) + \displaystyle \sum _ { j \in J } \mu _ { j } \nabla g _ { j } ( x ^ { \star } ) = 0 } \\ { \implies } & { \forall x \in B ( x ^ { \star } , \delta ) \exists ( \alpha , \beta ) \neq 0 \ : \ \displaystyle \sum _ { i \in I } \alpha _ { i } \nabla h _ { i } ( x ) + \displaystyle \sum _ { j \in J } \beta _ { j } \nabla g _ { j } ( x ) = 0 . } \end{array}
$$

Definition 9 (Andreani et al., 2005) is a relaxed version of the CPLD constraint qualification introduced by Andreani et al. (2005). With Definition 9, we state the circumstances under which the KKT conditions are necessary for a local optimum in Theorem 2.

Theorem 2. For a primal variable v of $P _ { \omega } ^ { \pm } ( \lambda )$ , let $v ^ { \star }$ denote the corresponding variable value in a (feasible) local optimum. Suppose the following:

$x ^ { \star }$ is such that $\| { \boldsymbol x } \| _ { 2 } > 0$ and $x ^ { \star } \neq \pm x _ { i }$ for all $i \in [ n ] ,$

2. $x _ { i } ^ { \top } x _ { j } \in [ - 1 , 1 ]$ for $i , j \in [ n ]$ and that $x _ { i } ^ { \top } x _ { j } = 1$ only $i f i = j ,$

3. $L \geq 2$ so that $\Theta _ { \infty } ^ { ( L ) } \left( X X ^ { \top } \right)$ is invertible.

Then, the local optimum satisfies the KKT conditions for some dual coeficients multiplying the equality and inequality constraints.

Proof. From the stationarity condition, we get that a linear combination of the gradients of the equality constraints and the active inequality constraints is equal to 0 only if the dual coeficients are all equal to $0 ,$ except (possibly) the dual coeficients of $\mathcal { C } ( s , a )$ . However, $\mathcal { C } ( s , a )$ corresponds to linear equalities and inequalities. A linear combination of an arbitrary subset of these gradients that is equal to 0 at $x ^ { \star }$ leads to a linear combination that is equal to 0 at x. Because of this fact, the rank of $\{ \nabla h _ { i } ( x ) \mid i \in \mathcal { T } \}$ is constant for $x \in B ( x ^ { \star } , \delta )$ . We conclude that the constraints of $( P _ { \omega } ^ { \pm } ( \lambda ) )$ satisfy RCPLD and that the KKT conditions are necessary for the local optimal solution. □

In the statement of Theorem 2, the vectors $x _ { i } , \ i \in [ n ]$ , are normalized with unit norm, but since the unnormalized version $\Theta _ { \infty } ^ { ( L ) }$ is homogeneous component-wise (Jacot et al., 2018), the statement holds in the general sense by assuming the $x _ { i } \mathrm { { ' s } }$ and x are not pairwise colinear. The proof of Theorem 2 also holds in the more general case where the constraints of $\mathcal { C } ( s , a )$ are not linear; in this instance, one needs some regularity condition like quasi-normality constraint qualification. Theorem 2 guarantees that in a MIP approach to solving $P _ { \omega } ^ { \pm } ( \lambda )$ (e.g. a spatial branch-and-bound), the upper bound<sup>6</sup> search in the root node is well-behaved when using first-order approximations. For example, primal-dual interior point solvers like IPOPT will converge to stationary points (local minimum or maximum, saddle point) that are feasible, and any local optimum will satisfy the KKT conditions. As shown in Corollary 1, the first-order necessary condition extends to any local node in a MIP approach whenever the depth L and the size of the dataset n are large enough compared to the number of variables required to model the state space.

Corollary 1. Under the same assumptions as Theorem 2, and if we propagate the bounds of variable asin ${ \bf \Phi } _ { ; , i } , \rho _ { L , i } , \rho _ { l , i } ^ { \prime } , \Theta _ { l , i } .$ , a local node will also satisfy RCPLD whenever

$$
( 4 L - 2 ) n + 1 > | x | - m ,
$$

where $| x | = | s | + 1$ is the length of variable x and m is the number of equality constraints in $\mathcal { C } ( s , a )$

Proof. The gradients $\nabla g _ { j } ( x ^ { \star } )$ for $j \in J$ induce a cone $C _ { J }$ , while the gradients $\nabla h _ { i } ( x ^ { \star } )$ for $i \in \mathcal { T }$ induce a subspace $H _ { \mathcal { I } } ( x ^ { \star } )$ . We suppose $C _ { J } \cap H _ { \mathcal { T } } ( x ^ { \star } ) \neq \{ 0 \}$ and we observe that

$$
\operatorname { s p a n } ( \{ \nabla g _ { j } ( y ) \mid j \in J \} ) \cap H _ { \mathbb { Z } } ( y ) = \operatorname { s p a n } ( \{ \nabla g _ { j } ( x ^ { \star } ) \mid j \in J \} ) \cap H _ { \mathbb { Z } } ( y ) .\tag{4}
$$

For vector subspaces V and $W$

$$
\dim ( V \cap W ) = \dim ( V ) + \dim ( W ) - \dim ( V + W ) .
$$

Therefore, subspace (4) has dimension strictly greater than zero if

$$
\begin{array} { r l } & { \dim ( H _ { \mathcal { Z } } ( y ) ) + \dim ( \mathrm { s p a n } ( \{ \nabla g _ { j } ( y ) ~ | ~ j \in J \} ) ) > | y | } \\ & { ~ 4 L n + | y _ { x } | + 1 + m + ( 4 L - 2 ) n > 4 L n + 1 + 2 | y _ { x } | } \\ & { ~ ( 8 L - 2 ) n + | y _ { x } | + 1 + m > 4 L n + 1 + 2 | y _ { x } | } \\ & { ~ ( 4 L - 2 ) n + 1 > | y _ { x } | - m , } \end{array}
$$

where $y _ { x }$ is the section of $y$ indexed by variable x. Given our initial assumption, RCPLD is satisfied in a neighbourhood of x<sup>⋆</sup> $x ^ { \star }$ □