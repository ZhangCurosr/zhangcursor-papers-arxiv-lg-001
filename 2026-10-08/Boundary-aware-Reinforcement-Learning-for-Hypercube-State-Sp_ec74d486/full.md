# Boundary-aware Reinforcement Learning for Hypercube State Spaces via Deterministic Policy Gradient

Lijun Bo ∗ Yijie Huang <sup>†</sup> Chenhao Lu <sup>‡</sup>

## Abstract

We develop a continuous-time deterministic policy gradient framework for reinforcement learning with reflected state dynamics, where the state process is governed by a controlled reflected stochastic diferential equation on a hypercube. Under suitable regularity assumptions, we establish the connection between the value function and the Neumann Bellman equation, introduce an advantage-rate function that yields a deterministic policy gradient formula, and prove the martingale characterization theorem. Motivated by these theoretical results, we propose a continuous-time deep deterministic policy gradient algorithm for reflected stochastic systems, in which the Neumann boundary condition is imposed via either soft penalization or hard architectural constraint. We further quantify the discrepancy between the ideal continuous-time dynamics and the discretely sampled exploratory dynamics executed in practice, showing that the error decays as the time grid is refined and exploration noise vanishes. Our experiments on reservoir control problems illustrate the efectiveness of the RL framework, highlighting that boundary-aware methods substantially reduce Neumann boundary residuals and enhance learning stability.

Keywords: reflected difusion process, deterministic policy gradient, reinforcement learning, Neumann boundary condition, reservoir regulation.

## 1 Introduction

Reinforcement learning (RL) provides a general framework for sequential decision making in which an agent improves its policy through repeated interaction with an unknown environment. Classical RL has been developed predominantly in discrete time, where the environment is modeled as a Markov decision process and the agent updates value functions or policies from sampled transitions. Among the most influential value-based methods, Q-learning (see Watkins (1989) and Watkins and Dayan (1992)) learns the optimal state-action value function in a model-free manner and has become a cornerstone of discrete-time RL. Temporal-diference learning (Sutton (1988)), SARSA (Rummery and Niranjan (1994)), and actor-critic methods (Konda and Tsitsiklis (2000), Sutton and Barto (2018)) further established the algorithmic foundation for modern RL. The combination of valuebased RL with deep neural networks led to the deep Q-network (DQN) (Mnih et al. (2015)), which demonstrated that high-dimensional sensory inputs can be handled by deep function approximation. Policy-gradient methods (Williams (1992) and Sutton et al. (2000)) and their actor-critic variants have also become central in continuous-action control, where direct optimization of a parameterized policy is often more natural than maximization over a discrete action set.

Despite these successes, many real-world decision problems are not intrinsically discrete in time. Robotic manipulation, autonomous driving, high-frequency trading, queueing networks, energy storage, and water reservoir regulation all involve systems whose states evolve continuously and whose decisions may be updated at high frequency. Early attempts to bring RL into continuous time include advantage updating (Baird (1994)), the continuous-time value-function and policyimprovement framework of Doya (2000), and the stochastic control perspective of Munos and Bourgine (1997). These works already emphasized that the continuous-time formulation should not be viewed as a mere limiting case of a fixed discrete-time algorithm. A central dificulty is that objects that are well behaved in discrete time may degenerate when the time step becomes small. In particular, Tallec et al. (2019) showed that standard Q-learning-type methods can be highly sensitive to time discretization in near continuous-time environments, and that the classical discretetime Q-function is not the right object in the continuous-time limit. This observation motivates the development of RL algorithms directly at the continuous-time level, with time discretization introduced only at the implementation stage.

A systematic stochastic-control formulation of continuous-time RL was introduced by Wang et al. (2020), where exploration is modeled through entropy-regularized relaxed controls for difusion processes. Building on this formulation, Jia and Zhou (2022a) developed a martingale approach to policy evaluation in continuous time and space, showing that the value function of a fixed policy can be characterized by the martingality of a suitable reward-adjusted process. The same martingale viewpoint was then used in Jia and Zhou (2022b) to derive continuous-time policy gradient and actor-critic algorithms. A major step toward value-based learning in continuous time was made by Jia and Zhou (2023), who introduced the continuous-time counterpart of Q-learning. Since the usual ’big’ Q-function collapses to the value function in continuous time, Jia and Zhou (2023) defined a first-order state-action object, the so-called ’little’ q-function, which is closely related to the instantaneous advantage rate and the Hamiltonian. This q-function can be characterized by martingale conditions and used to design on-policy and of-policy actor-critic algorithms without discretizing the control problem in advance. The continuous-time RL literature has since expanded in several directions. Continuous-time q-learning has been extended to mean field control problems, where the state distribution enters the dynamics and reward, and the integrated and essential qfunctions arise naturally (see Wei et al. (2024) and Wei and Yu (2025)). More recently, Ren et al. (2026a) investigated mean field control with controlled common noise and the accompanying work Ren et al. (2026b) developed martingale-based q-learning and actor-critic algorithms for this formulation. Related martingale and q-learning ideas have also been developed for jump-difusion models, both under Shannon entropy Gao et al. (2025) and under Tsallis entropy Bo et al. (2024b). Another line of work studies the accuracy of discretely sampled stochastic policies, addressing the gap between continuous-time exploratory formulations and implementable piecewise-constant action processes (see Jia et al. (2025)). These developments show that continuous-time RL is not only a mathematical limit of discrete-time RL, but a distinct framework in which stochastic calculus, martingale methods, Hamilton-Jacobi-Bellman equations, and policy optimization interact

in a fundamental way.

Most of the above continuous-time RL works are formulated with stochastic or relaxed poli cies. This is natural for exploration, but it also creates computational and conceptual dificulties. First, the relaxed policy is a measure-valued control, and its implementation therefore requires high-frequency sampling from the prescribed action distribution. Second, stochastic policies introduce an additional action-integration layer into policy evaluation and policy improvement. In continuous action spaces, expectations with respect to the policy distribution must generally be computed analytically or approximated by Monte Carlo sampling. The latter may increase the computational cost of the policy update, especially for high-dimensional action spaces. These considerations motivate deterministic policy gradient methods, which optimize a parameterized deterministic feedback policy. In discrete time, deterministic policy gradient methods (Silver et al. (2014)) provide an eficient alternative to stochastic policy gradients in continuous action spaces, since the policy gradient only integrates over the state distribution. The deep deterministic policy gradient algorithm (DDPG) by Lillicrap et al. (2015) combines this idea with replay bufers and target networks, two stabilizing devices inherited from deep Q-learning. Recently, deterministic policy gradients have also been developed directly in continuous time. Cheng et al. (2025) derived a continuous-time deterministic policy gradient formula through the advantage-rate function and proposed a CT-DDPG algorithm based on martingale characterization. This framework has subsequently been extended to time-inconsistent stochastic control, where Guo et al. (2026) combined deterministic policy-gradient updates, martingale characterizations and inner fixed-point iterations to learn deterministic equilibrium policies. In the mean-field direction, Cheng et al. (2026) developed a model-free deterministic actor-critic framework for extended mean-field control, in which both the dynamics and rewards may depend on the joint distribution of states and controls. This line of work is particularly relevant to applications in which deterministic feedback controls are ultimately intended for implementation.

The present paper studies a complementary issue that has received much less attention in continuous-time RL: state boundary. In many applications, the state process is constrained by some limits. For example, a reservoir has lower and upper storage bounds; an inventory system has capacity constraints; a queue length or storage process may be constrained to remain nonnegative; an engineering system may be required to stay inside a prescribed safe region. A natural mathematical model for such constrained dynamics is a reflected stochastic diferential equation. When the state reaches the boundary, a finite-variation regulator pushes it back into the admissible domain. On a hypercube, this reflection acts componentwise on the lower and upper faces. The corresponding Bellman equation is therefore not an ordinary interior PDE, but a parabolic equation with Neumann boundary condition. The boundary condition determines how the critic should behave on reflecting faces, how the boundary regulator enters Itˆo’s formula, and how one should design stable value-function approximations near the boundary.

Motivated by these observations, this paper develops a boundary-aware actor-critic RL framework for reflected difusion control on a hypercube. For a fixed policy, the value function satisfies a Neumann Bellman equation. We then use this structure to construct a continuous-time deep deterministic policy gradient method for reflected dynamics, which we call CT-DDPG-R. The main contributions of this paper are summarized as follows.

(i) For a fixed parameterized deterministic Markov policy, we characterize the corresponding value function through a parabolic Bellman equation on the interior of the hypercube together with Neumann conditions on its reflecting faces. We derive a continuous-time deterministic policy gradient formula. A key feature of the analysis is that the regulator terms disappear from the final gradient expression because the value function satisfies the Neumann boundary condition. The result extends continuous-time deterministic policy gradient theory from unconstrained difusions to state-constrained reflected systems. Then, we introduce the continuous-time advantage-rate function associated with a deterministic policy and establish a martingale characterization of both the value function and the advantage-rate function.

(ii) Guided by the theoretical results, we propose CT-DDPG-R, a boundary-aware continuoustime deterministic policy algorithm. The method combines martingale-based critic learning, deterministic actor updates, experience replay, target networks, and two treatments of the Neumann condition: a soft Neumann penalty in the critic loss and a hard Neumann architecture based on cosine Fourier features. We evaluate the proposed method on one-, two-, and five-dimensional reservoir regulation problems. The experiments demonstrate that incorporating the Neumann condition substantially reduces boundary residuals and improves training stability, with the advantages becoming more pronounced as the state dimension increases.

(iii) The theoretical target policy is a continuous-time deterministic feedback, whereas the actions executed during training are sampled on a discrete time grid and perturbed by exploration noise. We quantify the discrepancy between the reflected state process under the ideal target policy and the process generated by the implementable piecewise-constant exploratory actions. The resulting estimate provides a theoretical justification for the discrete-time datacollection procedure used in CT-DDPG-R.

The rest of the paper is organized as follows. Section 2 introduces the controlled reflected SDE on a hypercube and the objective functional. Section 3 derives the Neumann Bellman equation for a fixed deterministic policy and connects it with the value function. Section 4 establishes the deterministic policy gradient formula and develops the martingale characterizations used for modelfree learning. Section 5 presents the CT-DDPG-R algorithm with the soft and hard constraints, and estimates the error of the state process in real environment. Section 6 applies our algorithm to the reservoir control problems and reports its performance.

## 2 Problem Formulation

In this section, we formulate the controlled reflected difusion problem on a hypercube. After specifying the reflected state dynamics and admissible policies, we define the value function associated with a deterministic Markov policy. This setup provides the stochastic control framework on which the subsequent Neumann Bellman equation, martingale characterization, and deterministic policy gradient are built.

Let $n \geq 1$ and $- \infty < \underline { { x } } _ { i } < \overline { { x } } _ { i } < \infty$ for $i = 1 , \ldots , n$ . Consider a hypercube given by $Q : =$ $\textstyle \prod _ { i = 1 } ^ { n } [ \underline { { x } } _ { i } , \overline { { x } } _ { i } ] \subset \mathbb { R } ^ { n }$ whose inner is $\begin{array} { r } { Q ^ { \circ } : = \prod _ { i = 1 } ^ { n } ( \underline { { x } } _ { i } , \overline { { x } } _ { i } ) } \end{array}$ . We then introduce the following reflected dynamics on the hypercube given by $Q ,$ which is given by, for $t \in ( 0 , T ]$ with $T > 0$

$$
X _ { t } = \xi _ { 0 } + \int _ { 0 } ^ { t } b ( s , X _ { s } , a _ { s } ) d s + \int _ { 0 } ^ { t } \sigma ( s , X _ { s } , a _ { s } ) d W _ { s } + K _ { t } ^ { - } - K _ { t } ^ { + } \in Q ,\tag{2.1}
$$

where $W = ( W _ { t } ) _ { t \in [ 0 , T ] }$ is an m-dimensional Brownian motion defined on the filtered probability space $( \Omega , \mathcal { F } , \mathbb { P } , \mathbb { F } )$ with the filtration $\mathbb { F } = ( \mathcal { F } _ { t } ) _ { t \in [ 0 , T ] }$ satisfying the usual conditions, and $\xi _ { 0 } \in Q$ is the square-integrable initial data independent of $W$ . Here, $a = ( a _ { t } ) _ { t \in [ 0 , T ] }$ is the F-adapted control strategy which takes values on a convex and compact action set $A \subset \mathbb { R } ^ { d } , b : [ 0 , T ] \times \mathbb { R } ^ { n } \times A  \mathbb { R } ^ { n }$ and $\sigma : [ 0 , T ] \times \mathbb { R } ^ { n } \times A  \mathbb { R } ^ { n \times m }$ are coeficients of reflected dynamics. In particular, $K ^ { \pm } =$ $( K _ { t } ^ { \pm , 1 } , \ldots , K _ { t } ^ { \pm , n } ) _ { t \in [ 0 , T ] }$ are continuous and componentwisely non-decreasing processes satisfying $K _ { 0 } ^ { \pm } = 0$ and for any $i = 1 , \ldots , n$

$$
\int _ { 0 } ^ { T } \mathbf { 1 } _ { \{ X _ { t } ^ { i } > \underline { { x } } _ { i } \} } d K _ { t } ^ { - , i } = 0 , \quad \int _ { 0 } ^ { T } \mathbf { 1 } _ { \{ X _ { t } ^ { i } < \overline { { x } } _ { i } \} } d K _ { t } ^ { + , i } = 0 .
$$

The i-th element of $K ^ { \pm }$ is pushed only when it hits the corresponding lower or upper face. Equation (2.1) can be understood in the standard Skorokhod sense on $Q$ (see Pilipenko (2014)).

In contrast to the stochastic policies commonly adopted in prior continuous-time RL works $( \mathrm { e . g . } )$ , Jia and Zhou (2023)), which require sampling from action distributions and integrating over continuous action spaces, we adopt a deterministic Markov policy framework. Under this setting, the control action at time t and state x is prescribed by a deterministic feedback law of the form $a _ { t } = \mu ( t , X _ { t } )$ , with $\mu : [ 0 , T ] \times Q  A$ . To enable flexible function approximation and integration with deep learning, we introduce a parameterized family of deterministic policies $\mu _ { \phi } : [ 0 , T ] \times \mathbb { R } ^ { n } $ $A ,$ indexed by the parameter vector $\phi \in \mathbb { R } ^ { k }$

Let $r : [ 0 , T ] \times Q \times A \to \mathbb { R }$ be the running reward and $g : Q  \mathbb { R }$ be the terminal reward. Now, we fix a discount factor $\beta > 0$ . For any policy parameter $\phi \in \mathbb { R } ^ { k }$ , let $X ^ { \phi } = ( X _ { t } ^ { \phi } ) _ { t \in [ 0 , T ] }$ be the reflected state process satisfying (2.1) with the Markovian feedback control $a _ { t } = \mu _ { \phi } ( t , X _ { t } ^ { \phi } )$ for $t \in [ 0 , T ]$ . The performance of this policy is measured by the following dynamical expected cumulative return given by, for $( t , x ) \in [ 0 , T ] \times Q$ and $\phi \in \mathbb { R } ^ { k }$

$$
V ^ { \phi } ( t , x ) : = \mathbb { E } \left[ \int _ { t } ^ { T } e ^ { - \beta ( s - t ) } r \left( s , X _ { s } ^ { \phi } , \mu _ { \phi } ( s , X _ { s } ^ { \phi } ) \right) d s + e ^ { - \beta ( T - t ) } g ( X _ { T } ^ { \phi } ) \big | X _ { t } ^ { \phi } = x \right] ,\tag{2.2}
$$

and we aim to maximize $V ^ { \phi } ( t , x )$ over $\phi \in \mathbb { R } ^ { k }$

## 3 Neumann Bellman Equation on Hypercube

In this section, we characterize the value function (2.2) through a Neumann Bellman equation on the hypercube. Before that, several assumptions are imposed to ensure that our problem is well posed and that the associated value function is well defined.

Assumption 3.1 (Regularity of Coeficients). The functions $b ( t , x , a ) , \sigma ( t , x , a )$ and $r ( t , x , a )$ are continuous in $( t , x , a )$ . Moreover, there exists a constant $C > 0$ such that, for any $( t , x , a ) , ( t , x ^ { \prime } , a ^ { \prime } ) \in$ $[ 0 , T ] \times Q \times A$

$$
\left| b ( t , x , a ) - b ( t , x ^ { \prime } , a ^ { \prime } ) \right| + \left| \sigma ( t , x , a ) - \sigma ( t , x ^ { \prime } , a ^ { \prime } ) \right| \leq C \left( \left| x - x ^ { \prime } \right| + \left| a - a ^ { \prime } \right| \right) ,
$$

$$
\begin{array} { r l r } & { } & { \left| b ( t , x , a ) \right| + \left| \sigma ( t , x , a ) \right| \le C \left( 1 + | x | + | a | \right) , } \\ & { } & { \left| r ( t , x , a ) - r ( t , x ^ { \prime } , a ^ { \prime } ) \right| + \left| g ( x ) - g ( x ^ { \prime } ) \right| \le C \left( | x - x ^ { \prime } | + | a - a ^ { \prime } | \right) , } \\ & { } & { \left| r ( t , x , a ) \right| + | g ( x ) | \le C \left( 1 + | x | ^ { 2 } + | a | ^ { 2 } \right) . } \end{array}
$$

Assumption 3.2 (Regularity of Feedback Policy). The map $( \phi , t , x ) \mapsto \mu _ { \phi } ( t , x )$ is continuous on $\mathbb { R } ^ { k } \times [ 0 , T ] \times Q$ . Moreover, for any compact parameter set $\Phi \subset \mathbb { R } ^ { k }$ , there exists $L _ { \Phi } > 0$ such that $| \mu _ { \phi } ( t , x ) - \mu _ { \phi } ( t ^ { \prime } , x ^ { \prime } ) | \leq L _ { \Phi } ( | t - t ^ { \prime } | + | x - x ^ { \prime } | )$ for any $( \phi , t , t ^ { \prime } , x , x ^ { \prime } ) \in \Phi \times [ 0 , T ] ^ { 2 } \times Q ^ { 2 }$

Remark 3.1. Assumptions 3.1, 3.2 yield that the closed-loop coeficients $b _ { \phi } ( t , x ) : = b ( t , x , \mu _ { \phi } ( t , x ) )$ and $\sigma _ { \phi } ( t , x ) : = \sigma ( t , x , \mu _ { \phi } ( t , x ) )$ are Lipschitz in $x \in Q$ . Since Q satisfies the uniform exterior sphere condition and the uniform cone condition, it follows from Theorem 2.5.4 in Pilipenko $( 2 0 1 \llcorner )$ that reflected stochastic diferential equation (RSDE) (2.1) has a unique strong solution for each fixed $\phi .$

Assumption 3.3 (Uniform Ellipticity). The difusion coeficient $\sigma ( t , x , a )$ satisfies the uniform ellipticity condition, i.e., there exists a constant $\kappa > 0$ such that $\xi ^ { \top } \Sigma ( t , x , a ) \xi \geq \kappa | \xi | ^ { 2 } f o r \left( t , x , a \right) \in$ $[ 0 , T ] \times Q \times A$ and $\xi \in \mathbb { R } ^ { n }$ . Here, $\Sigma ( t , x , a ) : = \sigma ( t , x , a ) \sigma ( t , x , a ) ^ { \top }$

To support the subsequent theoretical derivations of the deterministic policy gradient and the martingale characterization, we impose the following regularity assumption on the value function (2.2) :

Assumption 3.4. For each fixed policy parameter $\phi ,$ the value function $V ^ { \phi } \in C ^ { 1 , 2 } ( [ 0 , T ) \times Q ^ { \circ } ) \cap$ $C ( [ 0 , T ] \times Q )$ with $\nabla _ { x } V ^ { \phi } \in C ( [ 0 , T ) \times Q )$ . Furthermore, its 1st and 2nd spatial derivatives are absolutely integrable over [0, T) for any $x \in Q , i . e .$ , there exists a function $\Psi \in L ^ { 1 } ( [ 0 , T ) )$ such that

$$
\operatorname* { s u p } _ { x \in Q } \left( | \nabla _ { x } V ^ { \phi } ( t , x ) | + | \nabla _ { x } ^ { 2 } V ^ { \phi } ( t , x ) | \right) \leq \Psi ( t ) , \quad \forall t \in [ 0 , T ] .
$$

Remark 3.2. For reflected difusion control problems defined on hypercubes, the value function is conventionally characterized via standard dynamic programming and viscosity solution theory as the unique continuous viscosity solution to the corresponding Neumann boundary value problem (3.1). Nevertheless, to the best of our knowledge, a general classical regularity theorem for such reflected hypercube control problems remains absent in the existing literature. In this work, we impose the regularity condition $V ^ { \phi } \in C ^ { 1 , 2 } ( [ 0 , T ) \times Q ^ { \circ } ) \cap C ( [ 0 , T ] \times Q )$ with $\nabla _ { x } V ^ { \phi } \in C ( [ 0 , T ) \times Q )$ instead of the stronger regularity requirement $V ^ { \phi } \in C ^ { 1 , 2 } ( [ 0 , T ] \times Q )$ . This distinction is critical for problem formulations on hypercube domains. Specifically, compatibility conditions, which connect the terminal condition and the Neumann boundary condition, may be violated. As a result, full classical regularity over the closed cylindrical domain $[ 0 , T ] \times Q$ cannot be generally guaranteed. The regularity assumption adopted herein aligns with the classical regularity results established in Bo et al. (2024a) for Robin boundary value problems on the nonnegative orthant.

Before deriving the Neumann Bellman equation, we record a basic property of reflected difusions that will be used repeatedly below. Although the finite-variation regulators $K ^ { - }$ and $K ^ { + }$ accumulate when the process hits the boundary, the state process itself does not spend positive Lebesgue time on the boundary. In other words, the reflection is carried by boundary local time, which is singular with respect to the time variable (c.f. Proposition 6.1 in Kang and Ramanan (2014)).

Lemma 3.1. Under Assumptions 3.1, 3.2 and 3.3, let $X ^ { \phi } = ( X _ { t } ^ { \phi } ) _ { t \in [ 0 , T ] }$ be a strong solution to the RSDE (2.1) under $\mu _ { \phi }$ . Then $\begin{array} { r } { \int _ { 0 } ^ { T } \mathbf { 1 } _ { \partial Q } ( X _ { s } ^ { \phi } ) d s = 0 } \end{array}$ , a.s..

We define the controlled generator associated with a fixed action $a \in A$ by, for $\varphi \in C ^ { 2 } ( [ 0 , T ] \times Q )$

$$
\mathscr { L } ^ { a } [ \varphi ] ( t , x ) : = \partial _ { t } \varphi ( t , x ) + b ( t , x , a ) ^ { \top } \nabla _ { x } \varphi ( t , x ) + \frac { 1 } { 2 } \mathrm { T r } [ \Sigma ( t , x , a ) \nabla _ { x } ^ { 2 } \varphi ( t , x ) ] - \beta \varphi ( t , x ) .
$$

Then, we have the following result on the HJB equation for reflected difusion processes under deterministic feedback policies.

Lemma 3.2. Let Assumptions 3.1, 3.2, 3.3 and 3.4 hold. Then, the value function $V ^ { \phi }$ defined by (2.2) satisfies the following Neumann boundary value problem given by

$$
\left\{ \begin{array} { l l } { \mathcal { L } ^ { \mu _ { \phi } ( t , x ) } [ V ^ { \phi } ] ( t , x ) + r ( t , x , \mu _ { \phi } ( t , x ) ) = 0 , } & { \forall ( t , x ) \in [ 0 , T ) \times Q ^ { \circ } , } \\ { \partial _ { x _ { i } } V ^ { \phi } ( t , x ) = 0 , } & { \forall ( t , x ) \in [ 0 , T ) \times \partial Q , \ i \in I _ { - } ( x ) \cup I _ { + } ( x ) , } \\ { V ^ { \phi } ( T , x ) = g ( x ) , } & { \forall x \in Q , } \end{array} \right.\tag{3.1}
$$

where the index sets $I _ { - } ( x )$ and $I _ { + } ( x )$ are respectively defined by $I _ { - } ( x ) : = \{ i \in \{ 1 , \dots , n \} : \ x _ { i } = \underline { { x } } _ { i } \}$ and $I _ { + } ( x ) : = \{ i \in \{ 1 , \dots , n \} : \ x _ { i } = \overline { { x } } _ { i } \}$ . Conversely, if a function $V \in C ^ { 1 , 2 } ( [ 0 , T ) \times Q ^ { \circ } ) \cap C ( [ 0 , T ] \times$ $Q )$ satisfying $\nabla _ { x } V \in C ( [ 0 , T ) \times Q )$ is a classical solution of (3.1), then it admits the probabilistic representation (2.2).

Proof. Let $( t , x ) \in [ 0 , T ] \times Q$ and write $X _ { s } = X _ { s } ^ { t , x , \phi }$ for $s \in [ t , T ]$ . We first prove the interior equation in (3.1). Fix $( t , x ) \in [ 0 , T ) \times Q ^ { \circ }$ . We choose $\rho > 0$ such that $\overline { { B ( x , \rho ) } } \subset Q ^ { \circ }$ , where $B ( x , \rho ) : = \{ y : | y - x | < \rho \}$ . Given $h > 0$ , define the stopping time $\tau _ { h } : = \operatorname* { i n f } \{ s \geq t : \ X _ { s } \ \notin$ $B ( x , \rho ) \} \land ( t + h ) \land T$ . For $s \in [ t , \tau _ { h } ]$ , no reflection occurs for $X _ { s }$ , and hence d $K _ { s } ^ { - , i } = \mathrm { d } K _ { s } ^ { + , i } = 0$ for $i = 1 , \ldots , n$ . Applying Itˆo’s formula to $e ^ { - \beta ( s - t ) } V ^ { \phi } ( s , X _ { s } )$ from t to $\tau _ { h }$ , and using the dynamic programming principle (DPP), we obtain

$$
0 = \mathbb { E } \left[ \int _ { t } ^ { \tau _ { h } } e ^ { - \beta ( s - t ) } \left( \mathcal { L } ^ { \mu _ { \phi } ( s , X _ { s } ) } [ V ^ { \phi } ] ( s , X _ { s } ) + r \left( s , X _ { s } , \mu _ { \phi } ( s , X _ { s } ) \right) \right) \mathrm { d } s \right] .\tag{3.2}
$$

Dividing both sides of of (3.2) by h and letting $h \downarrow 0$ , the dominated convergence theorem (DCT), together with the continuity of the coeficients and of the derivatives of $V ^ { \phi }$ in $Q ^ { \circ }$ , yields that

$$
\begin{array} { r } { \mathcal L ^ { \mu _ { \phi } ( t , x ) } [ V ^ { \phi } ] ( t , x ) + r ( t , x , \mu _ { \phi } ( t , x ) ) = 0 , \quad ( t , x ) \in [ 0 , T ) \times Q ^ { \circ } . } \end{array}
$$

Substituting $t = T$ into (2.2) gives the terminal condition $V ^ { \phi } ( T , x ) = g ( x )$ for all $x \in Q$

Next, we prove the Neumann boundary condition in (3.1). $\operatorname { F i x } i \in \{ 1 , \ldots , n \}$ and $( t , x ) \in [ 0 , T ) \times$ $\mathrm { r i } ( \Gamma _ { i } ^ { - } )$ , where $\mathrm { r i } ( \Gamma _ { i } ^ { - } )$ denotes the relative interior of the face $\Gamma _ { i } ^ { - } = \{ x \in Q : ~ x _ { i } = \underline { { x } } _ { i } \}$ . We choose a neighborhood U of x such that $U \cap \partial Q \subset \Gamma _ { i } ^ { - }$ . Given $h > 0$ , let $\tau _ { h } : = \operatorname* { i n f } \{ s \geq t : X _ { s } \notin U \} \wedge ( t + h ) \wedge T$ On $[ t , \tau _ { h } ]$ , the only possible regulator term is $K ^ { - , i }$ . By DPP, the interior equation and Lemma 3.1 yield that

$$
\mathbb { E } \left[ \int _ { t } ^ { \tau _ { h } } e ^ { - \beta ( s - t ) } \partial _ { x _ { i } } V ^ { \phi } ( s , X _ { s } ) \mathrm { d } K _ { s } ^ { - , i } \right] = 0 .\tag{3.3}
$$

We now show that $\partial _ { x _ { i } } V ^ { \phi } ( t , x ) = 0$ by contradiction. Suppose first that $\partial _ { x _ { i } } V ^ { \phi } ( t , x ) > 0$ . By continuity of $\partial _ { x _ { i } } V ^ { \phi }$ , we may assume that $\partial _ { x _ { i } } V ^ { \phi } ( \tilde { t } , \tilde { x } ) \geq \varepsilon > 0$ , for all $( \tilde { t } , \tilde { x } ) \in [ t , t + h ] \times ( U \cap \Gamma _ { i } ^ { - } )$ .

By Assumption 3.3, the difusion is uniformly nondegenerate. Since the reflected state process starts from the boundary face, the regulator $K ^ { - , i }$ increases with positive probability before $\tau _ { h }$ that is $\mathbb { P } ( K _ { \tau _ { h } } ^ { - , i } - K _ { t } ^ { - , i } > 0 ) > 0$ . Hence, we have

$$
\mathbb { E } \left[ \int _ { t } ^ { \tau _ { h } } e ^ { - \beta ( s - t ) } \partial _ { x _ { i } } V ^ { \phi } ( s , X _ { s } ) \mathrm { d } K _ { s } ^ { - , i } \right] \geq \varepsilon \mathbb { E } \left[ \int _ { t } ^ { \tau _ { h } } e ^ { - \beta ( s - t ) } \mathrm { d } K _ { s } ^ { - , i } \right] > 0 ,
$$

which contradicts (3.3). The case $\partial _ { x _ { i } } V ^ { \phi } ( t , x ) < 0$ leads to an analogous contradiction. Hence, we have $\partial _ { x _ { i } } V ^ { \phi } ( t , x ) = 0$ for all $( t , x ) \in [ 0 , T ) \times \mathrm { r i } ( \Gamma _ { i } ^ { - } )$ . The proof for the upper face $\Gamma _ { i } ^ { + } : = \{ x \in Q : x _ { i } =$ $\overline { { x } } _ { i } \}$ is analogous, and applying the same contradiction argument yields that $\partial _ { x _ { i } } V ^ { \phi } ( t , x ) = 0$ for all $( t , x ) \in [ 0 , T ) \times \mathrm { r i } ( \Gamma _ { i } ^ { + } )$ . Thus we can extend the result to the whole $[ 0 , T ) \times \partial Q$ by the continuity of $\nabla _ { x } V ^ { \phi } \in C ( [ 0 , T ) \times Q )$

Conversely, suppose that $V \in C ^ { 1 , 2 } ( [ 0 , T ) \times Q ^ { \circ } ) \cap C ( [ 0 , T ] \times Q )$ with $\nabla _ { x } V \in C ( [ 0 , T ) \times Q )$ be a classical solution of (3.1). Fix $( t , x ) \in [ 0 , T ) \times Q$ and set $\varepsilon \in ( 0 , T - t )$ . For $s \in [ t , T - \varepsilon ]$ , we define

$$
Y _ { s } : = e ^ { - \beta ( s - t ) } V ( s , X _ { s } ^ { \phi } ) + \int _ { t } ^ { s } e ^ { - \beta ( u - t ) } r \left( u , X _ { u } ^ { \phi } , \mu _ { \phi } ( u , X _ { u } ^ { \phi } ) \right) d u .\tag{3.4}
$$

It follows from Itˆo’s rule that

$$
\begin{array} { r l } & { \mathrm { d } Y _ { s } = e ^ { - \beta ( s - t ) } \left( \mathcal { L } ^ { \mu _ { \phi } ( s , X _ { s } ^ { \phi } ) } [ V ] ( s , X _ { s } ^ { \phi } ) + r \left( s , X _ { s } ^ { \phi } , \mu _ { \phi } ( s , X _ { s } ^ { \phi } ) \right) \right) \mathrm { d } s } \\ & { \qquad + e ^ { - \beta ( s - t ) } \nabla _ { x } V ( s , X _ { s } ^ { \phi } ) ^ { \top } \sigma \left( s , X _ { s } ^ { \phi } , \mu _ { \phi } ( s , X _ { s } ^ { \phi } ) \right) \mathrm { d } W _ { s } } \\ & { \qquad + e ^ { - \beta ( s - t ) } \displaystyle \sum _ { i = 1 } ^ { n } \partial _ { x _ { i } } V ( s , X _ { s } ^ { \phi } ) \mathrm { d } K _ { s } ^ { - , i } - e ^ { - \beta ( s - t ) } \displaystyle \sum _ { i = 1 } ^ { n } \partial _ { x _ { i } } V ( s , X _ { s } ^ { \phi } ) \mathrm { d } K _ { s } ^ { + , i } . } \end{array}
$$

The $\mathrm { d } K ^ { \pm , i }$ terms vanish by the Neumann boundary condition, and the drift term vanishes by the interior equation in (3.1) and Lemma 3.1. Consequently, $Y = ( Y _ { s } ) _ { s \in [ t , T - \varepsilon ] }$ is a martingale. Applying the martingality on the right hand of (3.4), it follows that

$$
V ( t , x ) = \mathbb { E } _ { t , x } \left[ \int _ { t } ^ { T - \varepsilon } e ^ { - \beta ( s - t ) } r \left( s , X _ { s } ^ { \phi } , \mu _ { \phi } ( s , X _ { s } ^ { \phi } ) \right) \mathrm { d } s + e ^ { - \beta ( T - \varepsilon - t ) } V ( T - \varepsilon , X _ { T - \varepsilon } ^ { \phi } ) \right] .
$$

Let $\varepsilon \downarrow 0$ and then the continuity of $V \in C ( [ 0 , T ] \times Q )$ , the terminal condition $V ( T , \cdot ) = g ( \cdot )$ and the boundedness of the compact set $[ 0 , T ] \times Q$ jointly imply that

$$
V ( t , x ) = \mathbb { E } _ { t , x } \left[ \int _ { t } ^ { T } e ^ { - \beta ( s - t ) } r \left( s , X _ { s } ^ { \phi } , \mu _ { \phi } ( s , X _ { s } ^ { \phi } ) \right) \mathrm { d } s + e ^ { - \beta ( T - t ) } g ( X _ { T } ^ { \phi } ) \right] ,
$$

which is precisely (2.2). Thus, the proof of the lemma is complete.

## 4 Policy Gradient and Martingales on Hypercube

In this section, we establish two theoretical foundations for reflected difusion control on the hypercube: a deterministic policy gradient proposition and a martingale characterization. The first result follows the line of deterministic policy gradient methods developed in Silver et al. (2014) and Cheng et al. (2025), while the second extends the martingale approach of Jia and Zhou (2023) for controlled difusions to the reflected state setting. These two results correspond, respectively, to the policy-improvement and policy-evaluation components of the actor-critic algorithm developed in the next section. In particular, the deterministic policy gradient formula provides the direction for updating the actor, whereas the martingale characterization yields model-free learning conditions for the value function and the advantage-rate function.

## 4.1 Deterministic policy gradient on the hypercube

For each $\phi \in \mathbb { R } ^ { k }$ , we define the continuous-time advantage rate function by

$$
\begin{array} { r } { q _ { \phi } ( t , x , a ) : = \mathcal L ^ { a } [ V ^ { \phi } ] ( t , x ) + r ( t , x , a ) , \quad \forall ( t , x , a ) \in [ 0 , T ) \times Q \times A . } \end{array}\tag{4.1}
$$

Then, we have from Lemma 3.2 that

$$
\begin{array} { r } { q _ { \phi } \left( t , x , \mu _ { \phi } ( t , x ) \right) = 0 , \quad \forall ( t , x ) \in [ 0 , T ) \times Q ^ { \circ } . } \end{array}\tag{4.2}
$$

To establish the deterministic policy gradient, we need additional diferentiability of the coeficients and feedback policy with respect to the action variable and the policy parameter.

Assumption 4.1. The functions $\nabla _ { a } b , \nabla _ { a } \Sigma$ , and $\nabla _ { a } r$ exist and are continuous on $[ 0 , T ] \times Q \times A$ Moreover, the partial derivative $( \phi , t , x ) \mapsto \nabla _ { \phi } \mu _ { \phi } ( t , x )$ is continuous on $\mathbb { R } ^ { k } \times [ 0 , T ] \times Q$

Fix $\phi \in \mathbb { R } ^ { k }$ . By using the diferentiability from Assumption 4.1, we can take the derivative with respect to $\phi$ on the both side of (4.1), and hence, for $( t , x , a ) \in [ 0 , T ) \times Q \times A$

$$
\nabla _ { a } q _ { \phi } ( t , x , a ) = ( \nabla _ { a } b ( t , x , a ) ) ^ { \top } \nabla _ { x } V ^ { \phi } ( t , x ) + \frac { 1 } { 2 } \mathrm { T r } ( ( \nabla _ { a } \Sigma ( t , x , a ) ) \nabla _ { x } ^ { 2 } V ^ { \phi } ( t , x ) ) + \nabla _ { a } r ( t , x , a ) .\tag{4.3}
$$

We also need to derive a stability property of the closed-loop system under parameter perturbations. The following lemma asserts that small changes in the policy parameter $\phi$ result in corresponding small changes in the resulting state trajectory in the mean-square sense, uniformly over the entire time horizon.

Lemma 4.1. Let Assumptions 3.1 and 3.2 hold. Then, for any $( t , x ) \in [ 0 , T ] \times Q$ and $\boldsymbol { h } \in \mathbb { R } ^ { k }$ , it holds that

$$
\operatorname* { l i m } _ { \varepsilon \to 0 } \mathbb { E } \left[ \operatorname* { s u p } _ { s \in \left[ t , T \right] } \left| X _ { s } ^ { t , x , \phi + \varepsilon h } - X _ { s } ^ { t , x , \phi } \right| ^ { 2 } \right] = 0 .
$$

Proof. We write $X _ { u } ^ { \varepsilon } : = X _ { u } ^ { t , x , \phi + \varepsilon h }$ and $X _ { u } : = X _ { u } ^ { t , x , \phi }$ , and define the corresponding closed-loop co-6 ${ \mathrm { f f i c i e n t s ~ } } b _ { \varepsilon } ( u , y ) : = b ( u , y , \mu _ { \phi + \varepsilon h } ( u , y ) ) , \sigma _ { \varepsilon } ( u , y ) : = \sigma ( u , y , \mu _ { \phi + \varepsilon h } ( u , y ) ) , b _ { 0 } ( u , y ) : = b ( u , y , \mu _ { \phi } ( u , y ) )$ and $\sigma _ { 0 } ( u , y ) : = \sigma ( u , y , \mu _ { \phi } ( u , y ) )$ . Let $Y ^ { \varepsilon } = ( Y _ { u } ^ { \varepsilon } ) _ { u \in [ t , T ] }$ and $Y = ( Y _ { u } ) _ { u \in [ t , T ] }$ satisfy the following dynamics without reflection:

$$
Y _ { u } ^ { \varepsilon } = x + \int _ { t } ^ { u } b _ { \varepsilon } ( r , X _ { r } ^ { \varepsilon } ) \mathrm { d } r + \int _ { t } ^ { u } \sigma _ { \varepsilon } ( r , X _ { r } ^ { \varepsilon } ) \mathrm { d } W _ { r } , \quad Y _ { u } = x + \int _ { t } ^ { u } b _ { 0 } ( r , X _ { r } ) \mathrm { d } r + \int _ { t } ^ { u } \sigma _ { 0 } ( r , X _ { r } ) \mathrm { d } W _ { r } .
$$

Then, the reflected processes $X ^ { \varepsilon } = ( X _ { t } ^ { \varepsilon } ) _ { t \in [ 0 , T ] }$ and $X = ( X _ { t } ) _ { t \in [ 0 , T ] }$ can be represented via a so-called Skorokhod map $\Gamma _ { Q } : C ( [ 0 , T ] ; \mathbb { R } ^ { n } ) \to C ( [ 0 , T ] ; Q )$ on the hypercube $Q$ by $X ^ { \varepsilon } = \Gamma _ { Q } ( Y ^ { \varepsilon } )$ and $X =$ $\Gamma _ { Q } ( Y )$ , which means that $X ^ { \varepsilon } , X$ are solutions respectively to the Skorokhod problems $( Q , Y ^ { \varepsilon } )$ and $( Q , Y )$ . Moreover, the Skorokhod map $\Gamma _ { Q }$ is Lipschitz in the sup norm (c.f. Dupuis et al. (1991)).

Consider $F _ { \varepsilon } ( u ) : = \mathbb E [ \operatorname* { s u p } _ { s \in [ t , u ] } | X _ { s } ^ { \varepsilon } - X _ { s } | ^ { 2 } ]$ for $u \geq t$ and we have $\begin{array} { r } { F _ { \varepsilon } ( u ) \leq C \mathbb { E } [ \operatorname* { s u p } _ { s \in [ t , u ] } | Y _ { s } ^ { \varepsilon } - Y _ { s } | ^ { 2 } ] } \end{array}$ for some constant $C = C ( T )$ . From Jensen’s inequality and Burkholder-Davis-Gundy Inequality (BDG Inequality), it follows that

$$
\mathbb { E } \left[ \operatorname* { s u p } _ { s \in [ t , u ] } | Y _ { s } ^ { \varepsilon } - Y _ { s } | ^ { 2 } \right] \leq C \int _ { t } ^ { u } \mathbb { E } \left[ | b _ { \varepsilon } ( r , X _ { r } ^ { \varepsilon } ) - b _ { 0 } ( r , X _ { r } ) | ^ { 2 } \right] \mathrm { d } r + C \int _ { t } ^ { u } \mathbb { E } \left[ | \sigma _ { \varepsilon } ( r , X _ { r } ^ { \varepsilon } ) - \sigma _ { 0 } ( r , X _ { r } ) | ^ { 2 } \right] \mathrm { d } r .
$$

Note that $\vert b _ { \varepsilon } ( r , X _ { r } ^ { \varepsilon } ) - b _ { 0 } ( r , X _ { r } ) \vert \leq \vert b _ { \varepsilon } ( r , X _ { r } ^ { \varepsilon } ) - b _ { \varepsilon } ( r , X _ { r } ) \vert + \vert b _ { \varepsilon } ( r , X _ { r } ) - b _ { 0 } ( r , X _ { r } ) \vert$ . Since b is Lipschitz in $( x , a )$ , the first term $| b _ { \varepsilon } ( r , X _ { r } ^ { \varepsilon } ) - b _ { \varepsilon } ( r , X _ { r } ) | \leq C ( | X _ { r } ^ { \varepsilon } - X _ { r } | + | \mu _ { \phi + \varepsilon h } ( r , X _ { r } ^ { \varepsilon } ) - \mu _ { \phi + \varepsilon h } ( r , X _ { r } ) | )$ . We can choose $| h | \leq 1 , \varepsilon \leq 1$ so that, by Assumption 3.2, there exists $L > 0$ such that $| \mu _ { \tilde { \phi } } ( r , x ) - \mu _ { \tilde { \phi } } ( r , x ^ { \prime } ) | \leq$ $L | x { - } x ^ { \prime } |$ for $( \tilde { \phi } , r , x , x ^ { \prime } ) \in \{ \tilde { \phi } : | \tilde { \phi } - \phi | \leq 1 \} \times [ 0 , T ] \times Q ^ { 2 }$ , which yields $| b _ { \varepsilon } ( r , X _ { r } ^ { \varepsilon } ) - b _ { \varepsilon } ( r , X _ { r } ) | \leq C ( | X _ { r } ^ { \varepsilon } -$ $X _ { r } | )$ . Here, the constant C depends on the compact set of parameters containing both ϕ and $\phi + \varepsilon h$ For the second term, we consider $\delta _ { \varepsilon } : = \operatorname* { s u p } _ { ( u , y ) \in [ t , T ] \times Q } | b _ { \varepsilon } ( u , y ) - b _ { 0 } ( u , y ) | + \operatorname* { s u p } _ { ( u , y ) \in [ t , T ] \times Q } | \sigma _ { \varepsilon } ( u , y ) -$ $\sigma _ { 0 } ( u , y ) |$ which tends to zero since Q is compact and the mapping $( \phi , u , y ) \mapsto \mu _ { \phi } ( u , y )$ is continuous. Then $| b _ { \varepsilon } ( r , X _ { r } ) - b _ { 0 } ( r , X _ { r } ) | \leq \operatorname* { s u p } _ { ( u , y ) \in [ t , T ] \times Q } | b _ { \varepsilon } ( u , y ) - b _ { 0 } ( u , y ) | \leq \delta _ { \varepsilon }$ . Combining these estimates, we obtain that $| b _ { \varepsilon } ( r , X _ { r } ^ { \varepsilon } ) - b _ { 0 } ( r , X _ { r } ) | \leq C | X _ { r } ^ { \varepsilon } - X _ { r } | + \delta _ { \varepsilon }$ . A similar argument for the difusion coeficient gives that $| \sigma _ { \varepsilon } ( r , X _ { r } ^ { \varepsilon } ) - \sigma _ { 0 } ( r , X _ { r } ) | \leq C | X _ { r } ^ { \varepsilon } - X _ { r } | + \delta _ { \varepsilon }$ . Consequently, we obtain

$$
\mathbb { E } \left[ \operatorname* { s u p } _ { s \in \left[ t , u \right] } \left| Y _ { s } ^ { \varepsilon } - Y _ { s } \right| ^ { 2 } \right] \leq C \int _ { t } ^ { u } \mathbb { E } \left[ \left| X _ { r } ^ { \varepsilon } - X _ { r } \right| ^ { 2 } \right] \mathrm { d } r + C \delta _ { \varepsilon } ^ { 2 } .
$$

This implies that $\begin{array} { r } { F _ { \varepsilon } ( u ) \leq C \int _ { t } ^ { u } F _ { \varepsilon } ( r ) \mathrm { d } r + C \delta _ { \varepsilon } ^ { 2 } } \end{array}$ . It follows from Gronwall’s inequality that $F _ { \varepsilon } ( T ) \leq$ $C \delta _ { \varepsilon } ^ { 2 } e ^ { C ( T - t ) } \to 0 \mathrm { ~ a s ~ } \varepsilon \to 0$ . Thus, the desired $L ^ { 2 } .$ -uniform stability is established. □

We now present the main result of this section, which gives the deterministic policy gradient for reflected difusions on a hypercube.

Proposition 4.1. Let assumptions of Lemma 3.2 and Assumption $\it 4 . 1$ hold. Then, for any $( t , x ) \in$ $[ 0 , T ] \times Q$ , the mapping $\phi \mapsto V ^ { \phi } ( t , x )$ is Gˆateaux diferentiable. More precisely, for any $\boldsymbol { h } \in \mathbb { R } ^ { k }$

$$
\operatorname* { l i m } _ { \varepsilon \to 0 } \frac { V ^ { \phi + \varepsilon h } ( t , x ) - V ^ { \phi } ( t , x ) } { \varepsilon } = \mathbb E \left[ \int _ { t } ^ { T } e ^ { - \beta ( s - t ) } ( \nabla _ { \phi } \mu _ { \phi } ( s , X _ { s } ^ { \phi } ) h ) ^ { \top } \nabla _ { a } q _ { \phi } ( s , X _ { s } ^ { \phi } , \mu _ { \phi } ( s , X _ { s } ^ { \phi } ) ) \mathrm { d } s \Bigg | X _ { t } ^ { \phi } = x \right] .
$$

This equivalently yields that

$$
\nabla _ { \phi } V ^ { \phi } ( t , x ) = \mathbb { E } \left[ \int _ { t } ^ { T } e ^ { - \beta ( s - t ) } ( \nabla _ { \phi } \mu _ { \phi } ( s , X _ { s } ^ { \phi } ) ) ^ { \top } \nabla _ { a } q _ { \phi } ( s , X _ { s } ^ { \phi } , \mu _ { \phi } ( s , X _ { s } ^ { \phi } ) ) d s \middle | X _ { t } ^ { \phi } = x \right] .\tag{4.4}
$$

Proof. The case $t = T$ is immediate, since the terminal condition $V ^ { \phi } ( T , x ) = g ( x )$ does not depend on $\phi .$ . We therefore fix $( t , x ) \in [ 0 , T ) \times Q$ and $\boldsymbol { h } \in \mathbb { R } ^ { k }$ , and consider a perturbed policy parameter $\phi + \varepsilon h$ for $\varepsilon \in \mathbb { R } / \{ 0 \}$ . For any $\eta \in ( 0 , T - t )$ , by applying the Itˆo’s rule to $e ^ { - \beta ( s - t ) } V ^ { \phi } ( s , X _ { s } ^ { \phi + \varepsilon h } )$ on $\left[ t , T - \eta \right]$ , we have

$$
\begin{array} { l } { { \displaystyle { \mathrm { d } \left( e ^ { - \beta ( s - t ) } V ^ { \phi } ( s , X _ { s } ^ { \phi + \varepsilon h } ) \right) = e ^ { - \beta ( s - t ) } \mathcal { L } ^ { \mu _ { \phi + \varepsilon h } ( s , X _ { s } ^ { \phi + \varepsilon h } ) } [ V ^ { \phi } ] ( s , X _ { s } ^ { \phi + \varepsilon h } ) \mathrm { d } s } } } \\ { { \displaystyle { \qquad + e ^ { - \beta ( s - t ) } \nabla _ { x } V ^ { \phi } ( s , X _ { s } ^ { \phi + \varepsilon h } ) ^ { \top } \sigma ( s , X _ { s } ^ { \phi + \varepsilon h } , \mu _ { \phi + \varepsilon h } ( s , X _ { s } ^ { \phi + \varepsilon h } ) ) \mathrm { d } W _ { s } } } } \\ { { \displaystyle { \qquad + e ^ { - \beta ( s - t ) } \sum _ { i = 1 } ^ { n } \partial _ { x _ { i } } V ^ { \phi } ( s , X _ { s } ^ { \phi + \varepsilon h } ) \mathrm { d } K _ { s } ^ { - , i , \phi + \varepsilon h } - e ^ { - \beta ( s - t ) } \sum _ { i = 1 } ^ { n } \partial _ { x _ { i } } V ^ { \phi } ( s , X _ { s } ^ { \phi + \varepsilon h } ) \mathrm { d } K _ { s } ^ { + , i , \phi + \varepsilon h } . } } } \end{array}
$$

Using the Neumann boundary condition in (3.1), the reflection terms vanish. Integrating over $\left[ t , T - \eta \right]$ and taking expectations given $X _ { t } ^ { \phi + \varepsilon h } = x$ , we obtain

$$
\begin{array} { r l } & { V ^ { \phi } ( t , x ) = \mathbb { E } \biggl [ e ^ { - \beta ( T - \eta - t ) } V ^ { \phi } ( T - \eta , X _ { T - \eta } ^ { \phi + \varepsilon h } ) } \\ & { \qquad - \displaystyle \int _ { t } ^ { T - \eta } e ^ { - \beta ( s - t ) } \mathcal { L } ^ { \mu _ { \phi + \varepsilon h } ( s , X _ { s } ^ { \phi + \varepsilon h } ) } [ V ^ { \phi } ] ( s , X _ { s } ^ { \phi + \varepsilon h } ) \mathrm { d } s \biggl | X _ { t } ^ { \phi + \varepsilon h } = x \biggr ] . } \end{array}
$$

Since $V ^ { \phi } \in C ( [ 0 , T ] \times Q )$ and $Q$ is compact, letting $\eta \downarrow 0$ , and the DCT yields that

$$
\begin{array} { l } { { \displaystyle V ^ { \phi } ( t , x ) = \mathbb { E } \biggl [ e ^ { - \beta ( T - t ) } g ( X _ { T } ^ { \phi + \varepsilon h } ) } } \\ { { \displaystyle \qquad - \int _ { t } ^ { T } e ^ { - \beta ( s - t ) } { \mathcal { L } } ^ { \mu _ { \phi + \varepsilon h } ( s , X _ { s } ^ { \phi + \varepsilon h } ) } [ V ^ { \phi } ] ( s , X _ { s } ^ { \phi + \varepsilon h } ) \mathrm { d } s \biggl \vert X _ { t } ^ { \phi + \varepsilon h } = x \biggr ] . } } \end{array}\tag{4.5}
$$

On the other hand, by the definition of (2.2), we have

$$
\begin{array} { l } { { V ^ { \phi + \varepsilon h } ( t , x ) = \mathbb E \biggl [ e ^ { - \beta ( T - t ) } g ( X _ { T } ^ { \phi + \varepsilon h } ) } } \\ { { \qquad + \left. \int _ { t } ^ { T } e ^ { - \beta ( s - t ) } r \left( s , X _ { s } ^ { \phi + \varepsilon h } , \mu _ { \phi + \varepsilon h } ( s , X _ { s } ^ { \phi + \varepsilon h } ) \right) \mathrm { d } s \right| X _ { t } ^ { \phi + \varepsilon h } = x \biggr ] . } } \end{array}\tag{4.6}
$$

Subtracting (4.5) from (4.6) gives that

$$
\begin{array} { r l } & { V ^ { \phi + \varepsilon h } ( t , x ) - V ^ { \phi } ( t , x ) = \mathbb { E } \Bigg [ \int _ { t } ^ { T } e ^ { - \beta ( s - t ) } \big ( r ( s , X _ { s } ^ { \phi + \varepsilon h } , \mu _ { \phi + \varepsilon h } ( s , X _ { s } ^ { \phi + \varepsilon h } ) ) } \\ & { \qquad + \mathcal { L } ^ { \mu _ { \phi + \varepsilon h } ( s , X _ { s } ^ { \phi + \varepsilon h } ) } [ V ^ { \phi } ] ( s , X _ { s } ^ { \phi + \varepsilon h } ) \big ) \mathrm { d } s \Big \vert X _ { t } ^ { \phi + \varepsilon h } = x \Bigg ] } \\ & { \qquad = \mathbb { E } \left[ \int _ { t } ^ { T } e ^ { - \beta ( s - t ) } q _ { \phi } ( s , X _ { s } ^ { \phi + \varepsilon h } , \mu _ { \phi + \varepsilon h } ( s , X _ { s } ^ { \phi + \varepsilon h } ) ) \mathrm { d } s \Big \vert X _ { t } ^ { \phi + \varepsilon h } = x \right] . } \end{array}
$$

By using Lemma 3.1 and (4.2), we obtain

$$
\begin{array} { r l r }   { V ^ { \phi + \varepsilon h } ( t , x ) - V ^ { \phi } ( t , x ) =  { \mathbb { E } } \bigg [ \int _ { t } ^ { T } e ^ { - \beta ( s - t ) } \Big ( q _ { \phi } ( s , X _ { s } ^ { \phi + \varepsilon h } , \mu _ { \phi + \varepsilon h } ( s , X _ { s } ^ { \phi + \varepsilon h } ) ) } \\ & { } & { \qquad - \ q _ { \phi } ( s , X _ { s } ^ { \phi + \varepsilon h } , \mu _ { \phi } ( s , X _ { s } ^ { \phi + \varepsilon h } ) ) \Big ) \mathrm { d } s \Big | X _ { t } ^ { \phi + \varepsilon h } = x \bigg ] , } \end{array}
$$

which yields that

$$
\begin{array} { r l } & { \frac { V ^ { \phi + s h } ( t , x ) - V ^ { \phi } ( t , x ) } { \varepsilon } } \\ & { \ = \mathbb { E } \left[ \int _ { t } ^ { T } e ^ { - \beta ( s - t ) } \frac { q _ { \phi } \left( s , X _ { s } ^ { \phi + \varepsilon h } , \mu _ { \phi + \varepsilon h } ( s , X _ { s } ^ { \phi + \varepsilon h } ) \right) - q _ { \phi } \left( s , X _ { s } ^ { \phi + \varepsilon h } , \mu _ { \phi } \left( s , X _ { s } ^ { \phi + \varepsilon h } \right) \right) } { \varepsilon } \mathrm { d } s \bigg | X _ { t } ^ { \phi + \varepsilon h } = x \right] . } \end{array}
$$

This yields that

$$
\frac { q _ { \phi } \big ( s , X _ { s } ^ { \phi + \varepsilon h } , \mu _ { \phi + \varepsilon h } ( s , X _ { s } ^ { \phi + \varepsilon h } ) \big ) - q _ { \phi } \big ( s , X _ { s } ^ { \phi + \varepsilon h } , \mu _ { \phi } ( s , X _ { s } ^ { \phi + \varepsilon h } ) \big ) } { \varepsilon }
$$

$$
= \int _ { 0 } ^ { 1 } \nabla _ { a } q _ { \phi } ( s , X _ { s } ^ { \phi + \varepsilon h } , \mu _ { \phi + \theta \varepsilon h } ( s , X _ { s } ^ { \phi + \varepsilon h } ) ) ^ { \top } \nabla _ { \phi } \mu _ { \phi + \theta \varepsilon h } ( s , X _ { s } ^ { \phi + \varepsilon h } ) h \mathrm { d } \theta .\tag{4.7}
$$

By Lemma 4.1, $X _ { s } ^ { \phi + \varepsilon h } \  \ X _ { s } ^ { \phi }$ in probability for every $s \in [ t , T ]$ . Moreover, the continuity of $( \phi , s , x ) \ \mapsto \ \mu _ { \phi } ( s , x )$ and $( \phi , s , x ) \ \mapsto \ \nabla _ { \phi } \mu _ { \phi } ( s , x )$ , together with the continuity of $\nabla _ { a } q _ { \phi }$ in the state and action variables, implies that the integrand on the right side of (4.7) converges to $\nabla _ { a } q _ { \phi } \left( s , X _ { s } ^ { \phi } , \mu _ { \phi } ( s , X _ { s } ^ { \phi } ) \right) ^ { \mid } \nabla _ { \phi } \mu _ { \phi } ( s , X _ { s } ^ { \phi } ) h$ in probability, dsdθ-a.e..By Assumption 3.4 and (4.3), there exists a deterministic function $\Psi \in L ^ { 1 } ( [ t , T ] )$ independent of ε such that $| \nabla _ { a } q _ { \phi } ( s , x , a ) | \leq \Psi ( s )$ for all $( s , x , a ) \in [ t , T ) \times Q \times A$ . Furthermore, for suficiently small $\varepsilon , \ \vert \nabla _ { \phi } \mu _ { \phi + \theta \varepsilon h } \vert$ is uniformly bounded by a deterministic constant because $X _ { s } ^ { \phi + \varepsilon h } \in Q$ and $( \phi , s , x ) \mapsto \nabla _ { \phi } \mu _ { \phi } ( s , x )$ is continuous on a compact set. Hence, we have from DCT that

$$
\operatorname* { l i m } _ { \varepsilon \to 0 } \frac { V ^ { \phi + \varepsilon h } ( t , x ) - V ^ { \phi } ( t , x ) } { \varepsilon } = \mathbb E \left[ \int _ { t } ^ { T } e ^ { - \beta ( s - t ) } \nabla _ { a } q _ { \phi } \big ( s , X _ { s } ^ { \phi } , \mu _ { \phi } ( s , X _ { s } ^ { \phi } ) \big ) ^ { \top } \nabla _ { \phi } \mu _ { \phi } ( s , X _ { s } ^ { \phi } ) h \mathrm { d } s \Bigg | X _ { t } ^ { \phi } = x \right] ,
$$

which completes the proof and yields (4.4).

Proposition 4.1 extends the classical deterministic policy gradient of Cheng et al. (2025) to the continuous-time setting with reflected state dynamics. Notably, the reflection terms are absent from the final gradient formula: the Neumann boundary condition exactly cancels the boundary regulator contributions, ensuring that the actor update depends only on the interior dynamics.

## 4.2 Martingale characterization on the hypercube

We now give the martingale characterization used for learning the value function and the advantagerate function. The conditions imposed on the candidate pair $( \widehat { V } , \widehat { q } )$ are consistent with the Neumann Bellman equation and the properties of the advantage-rate function. A key feature of the following characterization is that the martingale condition is not imposed only along the closed-loop trajectory generated by $\mu _ { \phi }$ . Instead, it is required to hold under the locally perturbed of $\mu _ { \phi } ( t , \boldsymbol { x } )$ . This leads to an of-policy learning scheme with exploration.

Theorem 4.2. Let assumptions of Lemma 3.2 hold. Fix $\phi \in \mathbb { R } ^ { k }$ . Let the candidate value function $\widehat { V } \in C ^ { 1 , 2 } ( [ 0 , T ) \times Q ^ { \circ } ) \cap C ( [ 0 , T ] \times Q )$ with $\nabla _ { x } \widehat { V } \in C ( [ 0 , T ) \times Q )$ , and the candidate advantage function $\widehat { q } \in C ( [ 0 , T ) \times Q \times A )$ satisfy

$$
\widehat { V } ( T , x ) = g ( x ) , \ \forall x \in Q , \quad \partial _ { x _ { i } } \widehat { V } ( t , x ) = 0 , \ \forall ( t , x ) \in [ 0 , T ) \times \partial Q \ a n d \ i \in I _ { - } ( x ) \cup I _ { + } ( x ) ,\tag{4.8}
$$

$$
\widehat { q } ( t , x , \mu _ { \phi } ( t , x ) ) = 0 , \quad \forall ( t , x ) \in [ 0 , T ) \times Q ^ { \circ } .\tag{4.9}
$$

Assume moreover that for every $( t , x ) \in [ 0 , T ) \times Q ^ { \circ }$ , there exists a neighborhood $\mathcal { O } _ { t , x } \subset A$ of $\mu _ { \phi } ( t , \boldsymbol { x } )$ such that, for every $a \in \mathcal { O } _ { t , x } .$ , there exists a square-integrable A-valued adapted control process $\alpha = ( \alpha _ { s } ) _ { s \in [ t , T ] }$ with lim ${ \it s \downarrow t } \alpha _ { s } = a , a . s .$ , under which the process

$$
\left( e ^ { - \beta ( s - t ) } \widehat { V } ( s , X _ { s } ^ { t , x , \alpha } ) + \int _ { t } ^ { s } e ^ { - \beta ( u - t ) } \left( r ( u , X _ { u } ^ { t , x , \alpha } , \alpha _ { u } ) - \widehat { q } ( u , X _ { u } ^ { t , x , \alpha } , \alpha _ { u } ) \right) \mathrm { d } u \right) _ { s \in [ t , T ] }
$$

is an F-martingale, where the reflected state process $X ^ { t , x , \alpha }$ solves the dynamics:

$$
\mathrm { d } X _ { s } ^ { t , x , \alpha } = b ( s , X _ { s } ^ { t , x , \alpha } , \alpha _ { s } ) \mathrm { d } s + \sigma ( s , X _ { s } ^ { t , x , \alpha } , \alpha _ { s } ) \mathrm { d } W _ { s } + \mathrm { d } K _ { s } ^ { - } - \mathrm { d } K _ { s } ^ { + } , \quad X _ { t } ^ { t , x , \alpha } = x .
$$

$T h e n , \widehat { V } ( t , x ) = V ^ { \phi } ( t , x ) f o r ( t , x ) \in [ 0 , T ] \times Q$ and $\widehat { q } ( t , x , a ) = q _ { \phi } ( t , x , a ) \ f o r \ ( t , x , a ) \in [ 0 , T ) \times Q ^ { \circ } \times$ $\mathcal { O } _ { t , x }$

Proof. Define the function $F ( t , x , a ) : = \mathcal { L } ^ { a } [ \widehat { V } ] ( t , x ) + r ( t , x , a ) - \widehat { q } ( t , x , a )$ for $( t , x , a ) \in [ 0 , T ) \times Q ^ { \circ } \times A$ We prove that $F ( t , x , a ) = 0$ for any $( t , x , a ) \in [ 0 , T ) \times Q ^ { \circ } \times \mathcal { O } _ { t , x }$ . Suppose that there exist $( t , x ) \in$ $[ 0 , T ) \times Q ^ { \circ }$ and $a \in \mathcal { O } _ { t , x }$ such that $F ( t , x , a ) > 0$ . By using the continuity of $( t , x , a ) \mapsto F ( t , x , a )$ there exist $\eta , \delta > 0$ such that $F ( \bar { t } , \bar { x } , \bar { a } ) \geq \eta$ for $( \bar { t } , \bar { x } , \bar { a } )$ satisfying max $\{ | \bar { t } - t | , \ | \bar { x } - x | , \ | \bar { a } - a | \} < \delta$ Without loss of generality, we may take $\delta < T - t .$ Then there exists an A-valued adapted control process $\alpha = ( \alpha _ { s } ) _ { s \in [ t , T ] }$ with lim $\boldsymbol { \mathrm { 1 } } _ { s } \boldsymbol { \downarrow } t ^ { \alpha _ { s } } = \boldsymbol { a }$ , a.s. such that the process defined by

$$
M _ { s } : = e ^ { - \beta ( s - t ) } \widehat { V } ( s , X _ { s } ^ { t , x , \alpha } ) + \int _ { t } ^ { s } e ^ { - \beta ( u - t ) } \left( r ( u , X _ { u } ^ { t , x , \alpha } , \alpha _ { u } ) - \widehat { q } ( u , X _ { u } ^ { t , x , \alpha } , \alpha _ { u } ) \right) \mathrm { d } u , \quad \forall s \in [ t , T ] .
$$

is an F-martingale. Let the stopping time $\tau : = \operatorname* { i n f } \{ s \geq t : \operatorname* { m a x } \{ | s - t | , | X _ { s } ^ { t , x , \alpha } - x | , | \alpha _ { s } - a | \} \geq$ $\delta \} \wedge ( t + \delta )$ . By the continuity of $s  X _ { s } ^ { t , x , \alpha }$ and the property $\alpha _ { s }  a$ as $s \downarrow t .$ , we have $\tau > t { \mathrm { ~ a . s . } }$ For $\varepsilon \in ( 0 , T - t )$ , applying the Itˆo rule to $e ^ { - \beta ( s - t ) } \widehat { V } ( s , X _ { s } ^ { t , x , \alpha } )$ on $[ t , T - \varepsilon ]$ , we obtain

$$
\begin{array} { r l } & { \mathrm { d } M _ { s } = e ^ { - \beta ( s - t ) } ( \mathcal { L } ^ { \alpha _ { s } } [ \widehat { V } ] ( s , X _ { s } ^ { t , x , \alpha } ) + r ( s , X _ { s } ^ { t , x , \alpha } , \alpha _ { s } ) - \widehat { q } ( s , X _ { s } ^ { t , x , \alpha } , \alpha _ { s } ) ) \mathrm { d } s } \\ & { \qquad + e ^ { - \beta ( s - t ) } \nabla _ { x } \widehat { V } ( s , X _ { s } ^ { t , x , \alpha } ) ^ { \top } \sigma ( s , X _ { s } ^ { t , x , \alpha } , \alpha _ { s } ) \mathrm { d } W _ { s } } \\ & { \qquad + e ^ { - \beta ( s - t ) } \displaystyle \sum _ { i = 1 } ^ { n } \partial _ { x _ { i } } \widehat { V } ( s , X _ { s } ^ { t , x , \alpha } ) \mathrm { d } K _ { s } ^ { - , i } - e ^ { - \beta ( s - t ) } \displaystyle \sum _ { i = 1 } ^ { n } \partial _ { x _ { i } } \widehat { V } ( s , X _ { s } ^ { t , x , \alpha } ) \mathrm { d } K _ { s } ^ { + , i } . } \end{array}
$$

By the boundary condition (4.8), the reflecting terms vanish. Since $( M _ { s } ) _ { s \in [ t , T - \varepsilon ] }$ is still a martingale, its finite-variation part must vanish. Therefore, for all $s \in [ t , T - \varepsilon ]$

$$
\int _ { t } ^ { s } e ^ { - \beta ( u - t ) } \left( \mathcal { L } ^ { \alpha _ { u } } [ \widehat { V } ] ( u , X _ { u } ^ { t , x , \alpha } ) + r ( u , X _ { u } ^ { t , x , \alpha } , \alpha _ { u } ) - \widehat { q } ( u , X _ { u } ^ { t , x , \alpha } , \alpha _ { u } ) \right) \mathrm { d } u = 0 , \quad \mathrm { a . s . }\tag{4.10}
$$

Since $\tau \leq t + \delta < T$ , we may apply (4.10) up to time τ , and obtain that

$$
0 = \int _ { t } ^ { \tau } e ^ { - \beta ( u - t ) } F ( u , X _ { u } ^ { t , x , \alpha } , \alpha _ { u } ) \mathrm { d } u \geq \eta \int _ { t } ^ { \tau } e ^ { - \beta ( u - t ) } \mathrm { d } u > 0 , \quad \mathrm { a . s . } ,
$$

which is a contradiction. Hence, $\widehat { q } ( t , x , a ) = \mathcal { L } ^ { a } [ \widehat { V } ] ( t , x ) + r ( t , x , a )$ for all $( t , x , a ) \in [ 0 , T ) \times Q ^ { \circ } \times \mathcal { O } _ { t , x } .$ Taking $a ~ = ~ \mu _ { \phi } ( t , x ) ~ \in ~ \mathcal { O } _ { t , x }$ and using (4.9), we get $\mathcal { L } ^ { \mu _ { \phi } ( t , x ) } [ \widehat { V } ] ( t , x ) + r ( t , x , \mu _ { \phi } ( t , x ) ) = 0 $ for all $( t , x ) \in [ 0 , T ) \times Q ^ { \circ }$ . Together with the boundary condition (4.8), this shows that V“ is a classical solution of the Neumann boundary problem (3.1). By Lemma 3.2, it follows that ${ \widehat { V } } = V ^ { \phi }$ on $[ 0 , T ] \times Q$ Then, we conclude that $\widehat { q } ( t , x , a ) = \mathcal { L } ^ { a } [ V ^ { \phi } ] ( t , x ) + r ( t , x , a ) = q _ { \phi } ( t , x , a )$ for all $( t , x ) \in [ 0 , T ) \times Q ^ { \circ }$ and $a \in \mathcal { O } _ { t , x }$ □

## 5 Model-free CT-DDPG-R Algorithm

In this section, we design the model-free continuous-time actor-critic algorithm for reflected diffusion control based on the theoretical results in Section 4. To ensure that the parameterized neural networks are consistent with the conditions (4.8) required by the theoretical framework, we introduce two boundary-aware approximation methods: a soft penalty formulation and a hard architectural constraint. We also analyze the discrepancy between the ideal continuous-time state process under the target policy and the state process generated in practical implementation by discretely sampled, piecewise-constant exploratory controls, and derive an upper bound for this approximation error.

## 5.1 Algorithm design

Unlike model-based control approaches, our continuous-time RL framework does not assume prior knowledge of the coeficients b and σ governing the state dynamics and the reward functions r and $g .$ Instead, we assume only that the agent can interact with the environment and collect trajectory data in the form of state-action-reward tuples. Proposition 4.1 and Theorem 4.2 provide the theoretical underpinnings for a model-free actor-critic implementation: the martingale characterization enables critic learning from temporal-diference (TD) residuals, while the policy gradient theorem guides the actor update toward improved performance.

We introduce parametric function approximators $V _ { \theta } : [ 0 , T ] \times Q \to \mathbb { R } , \ \bar { q } _ { \psi } : [ 0 , T ] \times Q \times A \to$ R, $\mu _ { \phi } : [ 0 , T ] \times Q  A$ for the value function, the advantage-rate function, and the deterministic $\mathrm { p o l i c y } .$ , where $\theta , \psi ,$ , and $\phi$ denote the respective parameter vectors. To avoid that the parameters $\theta , \phi$ for value network and policy network change too fast, we use target networks $\bar { \theta } , \bar { \phi }$ updated by

$$
\bar { \theta }  \tau \theta + ( 1 - \tau ) \bar { \theta } , \qquad \bar { \phi }  \tau \phi + ( 1 - \tau ) \bar { \phi } .
$$

with $\tau \in ( 0 , 1 )$ controlling the update rate. Following Cheng et al. (2025), we enforce the normalization condition (4.9) via the reparameterization

$$
\begin{array} { r } { q _ { \psi } ( t , x , a ) : = \bar { q } _ { \psi } ( t , x , a ) - \bar { q } _ { \psi } \left( t , x , \mu _ { \bar { \phi } } ( t , x ) \right) , } \end{array}
$$

for the current policy $\mu _ { \phi }$ when training the critic.

In our paper, two methods are adopted to address the Neumann boundary condition:

• Soft constraint. The soft constraint approach uses a standard neural network $V _ { \theta } ( t , x )$ without architectural modifications, but adds the critic loss with a boundary penalty term. Specifically, we sample boundary points $\{ ( t ^ { ( j ) } , x ^ { ( j ) } ) \} _ { j = 1 } ^ { B _ { N } }$ from the faces of the hypercube and penalize the squared normal derivatives of the value function:

$$
\mathcal { L } _ { N } ( \theta ) = \frac { 1 } { B _ { N } } \sum _ { j = 1 } ^ { B _ { N } } \sum _ { \substack { i \in I _ { - } ( x ^ { ( j ) } ) \cup I _ { + } ( x ^ { ( j ) } ) } } \left| \partial _ { x _ { i } } V _ { \theta } ( t ^ { ( j ) } , x ^ { ( j ) } ) \right| ^ { 2 } .
$$

• Hard constraint. For the hypercube domain, the Neumann boundary condition can be imposed exactly by constructing the value network using cosine Fourier features. Inspired by Straub et al. (2025), we embed the state coordinates through cosine functions of multiple frequencies, which naturally satisfy zero normal derivatives at the boundaries. The value network takes the form

$$
\begin{array} { r l } & { V _ { \theta } ( t , x ) = \bar { V } _ { \theta } \bigg ( t , \cos \bigg ( \pi \frac { x _ { 1 } - \underline { { x } } _ { 1 } } { \overline { { x } } _ { 1 } - \underline { { x } } _ { 1 } } \bigg ) , \cos \bigg ( \pi \frac { x _ { 2 } - \underline { { x } } _ { 2 } } { \overline { { x } } _ { 2 } - \underline { { x } } _ { 2 } } \bigg ) , \cdots , \cos \bigg ( \pi \frac { x _ { d } - \underline { { x } } _ { d } } { \overline { { x } } _ { d } - \underline { { x } } _ { d } } \bigg ) , } \\ & { \qquad \cos \bigg ( f _ { 1 } \pi \frac { x _ { 1 } - \underline { { x } } _ { 1 } } { \overline { { x } } _ { 1 } - \underline { { x } } _ { 1 } } \bigg ) , \cos \bigg ( f _ { 1 } \pi \frac { x _ { 2 } - \underline { { x } } _ { 2 } } { \overline { { x } } _ { 2 } - \underline { { x } } _ { 2 } } \bigg ) , \cdots , \cos \bigg ( f _ { 1 } \pi \frac { x _ { d } - \underline { { x } } _ { d } } { \overline { { x } } _ { d } - \underline { { x } } _ { d } } \bigg ) , } \\ & { \qquad \cos \bigg ( f _ { M } \pi \frac { x _ { 1 } - \underline { { x } } _ { 1 } } { \overline { { x } } _ { 1 } - \underline { { x } } _ { 1 } } \bigg ) , \cos \bigg ( f _ { M } \pi \frac { x _ { 2 } - \underline { { x } } _ { 2 } } { \overline { { x } } _ { 2 } - \underline { { x } } _ { 2 } } \bigg ) , \cdots , \cos \bigg ( f _ { M } \pi \frac { x _ { d } - \underline { { x } } _ { d } } { \overline { { x } } _ { d } - \underline { { x } } _ { d } } \bigg ) \bigg ) , } \end{array}\tag{5.1}
$$

where the integer frequencies $f _ { 1 } , f _ { 2 } , \cdot \cdot \cdot , f _ { M } \in \mathbb { Z } _ { + }$ are chosen in order to approximate the high frequency or multiscale data. It is obvious that the neural network (5.1) incorporates the Neumann condition by the property of cosine function.

Fix a time step $\Delta t$ and we get the time grid $0 = t _ { 0 } < t _ { 1 } < \cdots < t _ { N } = T$ with $\Delta t _ { k } : = t _ { k + 1 } - t _ { k } =$ $\Delta t .$ In step $k ,$ the agent observes the state $x _ { k } \in Q$ , selects an exploration action $a _ { k } \in A$ , and receives the next state $x _ { k + 1 } \in Q$ together with the running reward $r _ { k }$ . We store transitions of the form $\left( { { t } _ { k } } , { { x } _ { k } } , { { a } _ { k } } , { { r } _ { k } } \right)$ in a replay bufer $\mathcal { R }$ . To encourage suficient exploration of the state-action space, we adopt a perturbed version of the current deterministic policy:

$$
a _ { k } = \Pi _ { A } \left( \mu _ { \phi } ( t _ { k } , x _ { k } ) + \eta _ { k } \right) , \quad \eta _ { k } \sim \mathcal { N } ( 0 , \sigma _ { \exp } ^ { 2 } I _ { d } ) ,
$$

where $\sigma _ { \mathrm { e x p } } ^ { 2 }$ is a small variance to explore the behavior around $\mu _ { \phi } ( t _ { k } , x _ { k } ) , I _ { d }$ is the d-dimensional identity matrix, and $\Pi _ { A }$ denotes projection onto the action set A when needed. This exploration scheme injects Gaussian noise into the action, allowing the agent to explore the environment around the current policy and gather informative samples for learning the advantage-rate function.

We adopt the multi-step TD method when training the value network and the advantage-rate network. For any L-step sampled subsequence $\{ ( t _ { k + \ell } , x _ { k + \ell } , a _ { k + \ell } , r _ { k + \ell } ) \} _ { \ell = 0 } ^ { L }$ which is sampled from the replay bufer $\mathcal { R }$ , we define the following martingale TD given by

$$
\delta _ { k , L } ^ { \theta , \psi , \bar { \theta } } : = V _ { \theta } ( t _ { k } , x _ { k } ) - \sum _ { \ell = 0 } ^ { L - 1 } e ^ { - \beta ( t _ { k + \ell } - t _ { k } ) } \left( r _ { k + \ell } - q _ { \psi } ( t _ { k + \ell } , x _ { k + \ell } , a _ { k + \ell } ) \right) \Delta t - e ^ { - \beta ( t _ { k + L } - t _ { k } ) } V _ { \bar { \theta } } ( t _ { k + L } , x _ { k + L } ) ,
$$

where $V _ { \bar { \theta } }$ is the target value network. This residual is precisely the discrete-time approximation of the martingale condition from Theorem 4.2. After sampling a batch of B such subsequences, we compute the martingale loss

$$
\mathcal { L } _ { M } ( \theta , \psi ) : = \frac { 1 } { B } \sum _ { j = 1 } ^ { B } \left| \delta _ { k _ { j } , L } ^ { \theta , \psi , \bar { \theta } } \right| ^ { 2 } .
$$

Minimizing this loss drives the critic toward satisfying the martingale characterization, thereby aligning the value and advantage-rate estimates with the underlying dynamics. To enforce the terminal condition $V ^ { \phi } ( T , x ) = g ( x )$ , we also add a terminal penalty

$$
\mathcal { L } _ { T } ( \theta ) : = \frac { 1 } { B _ { T } } \sum _ { j = 1 } ^ { B _ { T } } \left| V _ { \theta } ( T , x ^ { ( j ) } ) - g ( x ^ { ( j ) } ) \right| ^ { 2 } ,
$$

where the terminal states $\{ x ^ { ( j ) } \} _ { j = 1 } ^ { B _ { T } }$ are sampled randomly from $Q .$

The overall critic loss is therefore a weighted combination of the martingale loss, the terminal penalty and, if using the soft constraint, the boundary penalty:

$$
\mathcal L _ { \mathrm { c r i t i c } } ( \theta , \psi ) = \mathcal L _ { M } ( \theta , \psi ) + \lambda _ { T } \mathcal L _ { T } ( \theta ) + \lambda _ { N } \mathcal L _ { N } ( \theta ) ,
$$

with weights $\lambda _ { T } > 0$ and $\lambda _ { N } > 0$ (where $\lambda _ { N } = 0$ for the hard constraint). The critic parameters are then updated via semi-gradient descent:

$$
\theta  \theta - \alpha _ { \theta } \partial _ { \theta } \mathcal { L } _ { \mathrm { c r i t i c } } , \qquad \psi  \psi - \alpha _ { \psi } \partial _ { \psi } \mathcal { L } _ { \mathrm { c r i t i c } } .
$$

Motivated by the deterministic policy gradient in Proposition 4.1, we update the actor parameters by ascending the following objective:

$$
\mathcal { T } _ { \mathrm { a c t o r } } ( \phi ) : = \frac { 1 } { B _ { A } } \sum _ { j = 1 } ^ { B _ { A } } \bar { q } _ { \psi } \left( t ^ { ( j ) } , x ^ { ( j ) } , \mu _ { \phi } ( t ^ { ( j ) } , x ^ { ( j ) } ) \right) \Delta t ,
$$

where $\{ ( t ^ { ( j ) } , x ^ { ( j ) } ) \} _ { j = 1 } ^ { B _ { A } }$ are sampled from replay bufer $\mathcal { R }$ and the actor parameters are updated as

$$
\phi  \phi + \alpha _ { \phi } \partial _ { \phi } \mathcal { T } _ { \mathrm { a c t o r } } .
$$

The full procedure of continuous-time deep deterministic policy gradient with reflection (CT-DDPG-R) on hypercube is described in Algorithm 1.

Algorithm 1 CT-DDPG-R on the hypercube   
Require: Time step $\Delta t ,$ time grid $0 = t _ { 0 } < t _ { 1 } < \cdots < t _ { K } = T ,$ discount rate $\beta ,$ number of   
episodes $N _ { ; }$ , policy net $\mu _ { \phi }$ , advantage-rate net ${ \bar { q } } _ { \psi }$ , value net $V _ { \theta } .$ , critic update frequency f<sub>critic</sub>,   
actor update frequency $f _ { \mathrm { a c t o r } } .$ , trajectory length $L ,$ exploration noise $\sigma _ { \mathrm { e x p } } ^ { 2 } ,$ soft update parameter   
$\tau ,$ learning rates $\alpha _ { \phi } , \alpha _ { \psi } , \alpha _ { \theta }$ , batch sizes $B , B _ { A } , B _ { T } , B _ { N }$ , loss weight $\lambda _ { T } , \lambda _ { N }$   
1: Initialize $\phi , \psi , \theta ,$ target $\bar { \theta } = \theta , \bar { \phi } = \phi$ and bufer $\mathcal { R }$   
2: for $n = 1 , 2 , \ldots , N$ do   
3: Observe the initial state $x _ { 0 } ,$ collect the transitions from a trajectory by exploratory action   
$a _ { t _ { k } } = \Pi _ { A } \left( \mu _ { \phi } ( t _ { k } , x _ { t _ { k } } ) + \eta _ { t _ { k } } \right) , \ \eta _ { k } \sim \mathcal { N } ( 0 , \sigma _ { \exp } ^ { 2 } I _ { d } ) .$   
4: for $m = 1 , \ldots , f _ { \mathrm { c r i t i c } }$ do   
5: Sample a batch of L-step trajectories and compute the martingale loss $\mathcal { L } _ { M } ( \theta , \psi )$   
6: Sample a batch of terminal points and compute the terminal penalty ${ \mathcal { L } } _ { T } ( \theta )$   
7: Sample a batch of boundary points and compute the boundary penalty ${ \mathcal { L } } _ { N } ( \theta )$   
8: Compute the critic loss $\mathcal { L } _ { \mathrm { c r i t i c } } ( \theta , \psi ) = \mathcal { L } _ { M } ( \theta , \psi ) + \lambda _ { T } \mathcal { L } _ { T } ( \theta ) + \lambda _ { N } \mathcal { L } _ { N } ( \theta )$   
9: Update $( \theta , \psi )$ by $\theta  \theta - \alpha _ { \theta } \partial _ { \theta } \mathcal { L } _ { \mathrm { c r i t i c } }$ and $\psi  \psi - \alpha _ { \psi } \partial _ { \psi } \mathcal { L } _ { \mathrm { c r i t i c } } .$   
10: end for   
11: Sample a batch of states from R and compute the policy loss ${ \mathcal { I } } _ { \mathrm { a c t o r } } ( \phi )$   
12: Update $\phi$ by $\phi  \phi + \alpha _ { \phi } \partial _ { \phi } \mathcal { T } _ { \mathrm { a c t o r } }$   
13: Update target networks: $\bar { \theta }  \tau \theta + ( 1 - \tau ) \bar { \theta } , \ \bar { \phi }  \tau \phi + ( 1 - \tau ) \bar { \phi } .$   
14: end for

## 5.2 Accuracy of discretely sampled exploratory dynamics

We now turn to quantify the discrepancy between the ideal continuous-time reflected dynamics governed by the deterministic policy $\mu _ { \phi }$ and the dynamics actually executed during training. This analysis is crucial for understanding the approximation error introduced by the discrete-time implementation of our otherwise continuous-time algorithm. Let

$$
\mathcal { G } : = \{ ( t _ { 0 } , t _ { 1 } , . . . , t _ { K } ) : ~ 0 = t _ { 0 } < t _ { 1 } < \cdot \cdot \cdot < t _ { K } = T \} , \qquad | \mathcal { G } | : = \operatorname* { m a x } _ { 0 \leq k \leq K - 1 } ( t _ { k + 1 } - t _ { k } )
$$

denote a uniform time grid and its mesh size, respectively. For $t \in [ t _ { k } , t _ { k + 1 } )$ , define the left-endpoint projection $\iota ( t ) : = t _ { k }$ . To model the exploration noise injected by the algorithm, let $\{ \zeta _ { k } \} _ { k \ge 0 }$ be i.i.d. random variables in $\mathbb { R } ^ { d }$ satisfying $\mathbb { E } [ \zeta _ { k } ] = 0 , \mathbb { E } | \zeta _ { k } | ^ { 2 } = M _ { \zeta } < \infty$ , and let $\sigma _ { \mathrm { e x p } } > 0$ be the exploration scale.

## 5.2.1 Extension of the probability space

To accommodate the independent random exploration noise $\{ \zeta _ { k } \} _ { k \ge 0 }$ generated by the algorithm, we extend the underlying probability space accordingly. Let $( \Omega ^ { \zeta } , { \mathcal { F } } ^ { \zeta } , \mathbb { P } ^ { \zeta } )$ be a probability space supporting $( \zeta _ { k } ) _ { k \geq 0 }$ . Define the product space $\bar { \Omega } : = \Omega \times \Omega ^ { \zeta } , \bar { \mathcal { F } } : = \mathcal { F } \otimes \mathcal { F } ^ { \zeta } , \bar { \mathbb { P } } : = \mathbb { P } \otimes \mathbb { P } ^ { \zeta }$ . On the extended space $( \bar { \Omega } , \bar { \mathcal { F } } , \bar { \mathbb { P } } )$ , we set $\bar { W } _ { t } ( \omega , \omega ^ { \zeta } ) : = W _ { t } ( \omega ) , ~ \bar { \xi } _ { 0 } ( \omega , \omega ^ { \zeta } ) : = \xi _ { 0 } ( \omega ) , ~ \bar { \zeta } _ { k } ( \omega , \omega ^ { \zeta } ) : = \zeta _ { k } ( \omega ^ { \zeta } )$ . For notational simplicity, we retain the original symbols $( W , \xi _ { 0 } , \zeta _ { k } )$ in place of $( \bar { W } , \bar { \xi } _ { 0 } , \bar { \zeta } _ { k } )$ . We equip the extended space with the enlarged filtration $\bar { \mathcal { F } } _ { t } ^ { \mathcal { G } } : = \sigma ( \xi _ { 0 } , W _ { s } : 0 \leq s \leq t ) \vee \sigma ( \zeta _ { k } : t _ { k } \leq t ) \vee \mathcal { N }$ for $t \in [ 0 , T ]$ , where $\mathcal { N }$ denotes the family of P<sup>¯</sup>-null sets. By independence, W remains a Brownian motion with respect to $\{ \bar { \mathcal { F } } _ { t } ^ { \mathcal { G } } \} _ { t \in [ 0 , T ] }$

## 5.2.2 Discretely executed exploratory reflected dynamics

In the actual implementation, the agent updates its action only at the discrete grid points $t _ { k }$ . At each intervention time $t _ { k }$ , the algorithm samples the exploratory action

$$
a _ { k } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } : = \Pi _ { A } \left( \mu _ { \phi } \left( t _ { k } , \boldsymbol { X } _ { t _ { k } } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } \right) + \sigma _ { \mathrm { e x p } } \zeta _ { k } \right) ,
$$

and holds it constant on $[ t _ { k } , t _ { k + 1 } )$ , that is, $a _ { t } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } = a _ { k } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } }$ for $t \in [ t _ { k } , t _ { k + 1 } )$ . This piecewiseconstant action construction mirrors the actual execution of the algorithm, where the policy is evaluated and the action is updated at regular intervals of length $\Delta t$

Consequently, the reflected state process under this discretely sampled exploratory policy satisfies

$$
\left\{ \begin{array} { l l } { \mathrm { d } X _ { t } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } = b \left( t , X _ { t } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } , a _ { t } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } \right) \mathrm { d } t + \sigma \left( t , X _ { t } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } , a _ { t } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } \right) \mathrm { d } W _ { t } } \\ { \qquad + \mathrm { d } K _ { t } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } , - } - \mathrm { d } K _ { t } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } , + } , } \\ { X _ { 0 } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } = \xi _ { 0 } . } \end{array} \right.\tag{5.2}
$$

where the regulators $K ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } , \pm }$ enforce the reflection on the boundary of $Q .$ . In contrast, the ideal deterministic reflected dynamics, which would be obtained if the action could be updated continuously in time, is governed by

$$
\left\{ \begin{array} { l } { \mathrm { d } X _ { t } ^ { \phi } = b \left( t , X _ { t } ^ { \phi } , \mu _ { \phi } ( t , X _ { t } ^ { \phi } ) \right) \mathrm { d } t + \sigma \left( t , X _ { t } ^ { \phi } , \mu _ { \phi } ( t , X _ { t } ^ { \phi } ) \right) \mathrm { d } W _ { t } + \mathrm { d } K _ { t } ^ { \phi , - } - \mathrm { d } K _ { t } ^ { \phi , + } , } \\ { X _ { 0 } ^ { \phi } = \xi _ { 0 } . } \end{array} \right.
$$

Both the sampled and the ideal processes are defined on the same product space $( \bar { \Omega } , \bar { \mathcal { F } } , \bar { \mathbb { P } } )$ . The following lemma ensures that the exploratory algorithm is well-defined at every step of the simulation, justifying the use of the piecewise-constant action approximation.

Lemma 5.1 (Well-posedness of the exploratory dynamics). Under the assumptions of Lemma ${ \it 3 . 2 , }$ for every grid G and every $\sigma _ { \mathrm { e x p } } > 0$ , the reflected SDE (5.2) admits a unique strong solution $( X ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } , K ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } , - } , K ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } , + } )$ adapted to $\{ \bar { \mathcal { F } } _ { t } ^ { \mathcal { G } } \} _ { t \in [ 0 , T ] }$

Proof. We proceed by induction over the grid intervals. The case at $t = 0$ is immediate from the initial condition $X _ { 0 } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } = \xi _ { 0 } \in \bar { \mathcal { F } } _ { 0 } ^ { \mathcal { G } }$ . Suppose that for some $k \in \mathbb N .$ the reflected SDE (5.2) admits a unique strong solution on $[ 0 , t _ { k } ]$ . Then we have $X _ { t _ { k } } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } \in \bar { \mathcal { F } } _ { t _ { k } } ^ { \mathcal { G } }$ , and

$$
a _ { k } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } = \Pi _ { A } \left( \mu _ { \phi } ( t _ { k } , X _ { t _ { k } } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } ) + \sigma _ { \mathrm { e x p } } \zeta _ { k } \right) \in \bar { \mathcal { F } } _ { t _ { k } } ^ { \mathcal { G } } .
$$

On $[ t _ { k } , t _ { k + 1 } )$ , we consider the following reflected SDE

$$
\begin{array} { r } { \left\{ \begin{array} { l l } { \mathrm { d } X _ { t } = b ( t , X _ { t } , a _ { k } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } ) \mathrm { d } t + \sigma ( t , X _ { t } , a _ { k } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } ) \mathrm { d } W _ { t } + \mathrm { d } K _ { t } ^ { - } - \mathrm { d } K _ { t } ^ { + } , } \\ { X _ { t _ { k } } = X _ { t _ { k } } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } . } \end{array} \right. } \end{array}
$$

For fixed $a _ { k } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } }$ , it follows from Assumptions 3.1 and 3.2 that

$$
| b ( t , x , a _ { k } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } ) - b ( t , y , a _ { k } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } ) | + | \sigma ( t , x , a _ { k } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } ) - \sigma ( t , y , a _ { k } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } ) | \le C | x - y | ,
$$

for some constant $C > 0$ . Therefore, the reflected SDE admits a unique strong solution on $[ t _ { k } , t _ { k + 1 } )$ This extends the solution to $[ 0 , t _ { k + 1 } ]$ , completing the induction. Hence the unique strong solution exists on the entire horizon $[ 0 , T ]$ □

## 5.2.3 State process error

We now analyze the discrepancy between the ideal continuous-time trajectory and the discretely sampled exploratory trajectory. This analysis is essential for quantifying the approximation error introduced by the piecewise-constant action implementation and the exploration noise.

For notational convenience, we introduce the shorthand $a _ { t } ^ { \phi } : = \mu _ { \phi } ( t , X _ { t } ^ { \phi } ) , K _ { t } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } : = K _ { t } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } , - } -$ $K _ { t } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } , + } , K _ { t } ^ { \phi } : = K _ { t } ^ { \phi , - } - K _ { t } ^ { \phi , + }$ and define the state deviation process $\Delta _ { t } : = X _ { t } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } - X _ { t } ^ { \phi }$

Theorem 5.1. Under the assumptions of Lemma $5 . 1 ,$ , there exists a constant $C > 0$ , independent $o f { \mathcal { G } }$ and $\sigma _ { \mathrm { e x p } }$ , such that

$$
\mathbb { E } \left[ \operatorname* { s u p } _ { 0 \leq t \leq T } \left| X _ { t } ^ { \mathcal { G } , \sigma _ { \exp } } - X _ { t } ^ { \phi } \right| ^ { 2 } \right] \leq C \left( | \mathcal { G } | + \sigma _ { \exp } ^ { 2 } M _ { \zeta } \right) .\tag{5.3}
$$

Proof. Define the diferences in drift and difusion coeficients as

$$
\Delta b _ { t } : = b ( t , X _ { t } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } , a _ { t } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } ) - b ( t , X _ { t } ^ { \phi } , a _ { t } ^ { \phi } ) , \quad \Delta \sigma _ { t } : = \sigma ( t , X _ { t } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } , a _ { t } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } ) - \sigma ( t , X _ { t } ^ { \phi } , a _ { t } ^ { \phi } ) .
$$

Then the deviation process satisfies the SDE

$$
\mathrm { d } \Delta _ { t } = \Delta b _ { t } \mathrm { d } t + \Delta \sigma _ { t } \mathrm { d } W _ { t } + \mathrm { d } K _ { t } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } - \mathrm { d } K _ { t } ^ { \phi } .
$$

Applying Itˆo’s formula to $| \Delta _ { t } | ^ { 2 }$ gives that

$$
| \Delta _ { t } | ^ { 2 } = 2 \int _ { 0 } ^ { t } \langle \Delta _ { s } , \Delta b _ { s } \rangle \mathrm { d } s + \int _ { 0 } ^ { t } | \Delta \sigma _ { s } | ^ { 2 } \mathrm { d } s + 2 \int _ { 0 } ^ { t } \langle \Delta _ { s } , \Delta \sigma _ { s } \mathrm { d } W _ { s } \rangle + 2 \int _ { 0 } ^ { t } \langle \Delta _ { s } , \mathrm { d } K _ { s } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } - \mathrm { d } K _ { s } ^ { \phi } \rangle .
$$

We claim that $\begin{array} { r } { \int _ { 0 } ^ { t } \langle \Delta _ { s } , \mathrm { d } K _ { s } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } - \mathrm { d } K _ { s } ^ { \phi } \rangle \leq 0 } \end{array}$ . For the lower face, if $\mathrm { d } K _ { s } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } , - , i } > 0$ , we have $X _ { s } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } , i } = \underline { { x } } _ { i }$ and then, $X _ { s } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } , i } - X _ { s } ^ { \phi , i } = \underline { { x } } _ { i } - X _ { s } ^ { \phi , i } \leq 0$ . For the upper face, if d $1 K _ { s } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } , + , i } > 0$ we have $X _ { s } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } , i } = \overline { { x } } _ { i } ,$ , and then $- ( X _ { s } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } , i } - X _ { s } ^ { \phi , i } ) = X _ { s } ^ { \phi , i } - \overline { { x } } _ { i } \leq 0$ . Therefore, we obtain that $\begin{array} { r } { \int _ { 0 } ^ { t } \langle X _ { s } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } - X _ { s } ^ { \phi } , \mathrm { d } K _ { s } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } \rangle \leq 0 } \end{array}$ . An analogous argument applied to the ideal regulator $K ^ { \phi }$ gives $\begin{array} { r } { \cdot \int _ { 0 } ^ { t } \langle X _ { s } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } - X _ { s } ^ { \phi } , \mathrm { d } K _ { s } ^ { \phi } \rangle \leq 0 } \end{array}$ . Adding the two inequalities gives $\begin{array} { r } { \int _ { 0 } ^ { t } \langle \Delta _ { s } , \mathrm { d } K _ { s } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } - \mathrm { d } K _ { s } ^ { \phi } \rangle \leq 0 } \end{array}$ , which yields

$$
| \Delta _ { t } | ^ { 2 } \leq 2 \int _ { 0 } ^ { t } \langle \Delta _ { s } , \Delta b _ { s } \rangle \mathrm { d } s + \int _ { 0 } ^ { t } | \Delta \sigma _ { s } | ^ { 2 } \mathrm { d } s + 2 \int _ { 0 } ^ { t } \langle \Delta _ { s } , \Delta \sigma _ { s } \mathrm { d } W _ { s } \rangle .\tag{5.4}
$$

By the Lipschitz condition in Assumption 3.1, $| \Delta b _ { s } | + | \Delta \sigma _ { s } | \le C ( | \Delta _ { s } | + | a _ { s } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } - a _ { s } ^ { \phi } | )$ for some constant $C \ > \ 0$ . Therefore, $2 \langle \Delta _ { s } , \Delta b _ { s } \rangle + | \Delta \sigma _ { s } | ^ { 2 } \leq C | \Delta _ { s } | ^ { 2 } + C | a _ { s } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } - a _ { s } ^ { \phi } | ^ { 2 }$ . Let $F ( t ) : =$ $\mathbb { E } [ \operatorname { s u p } _ { 0 \leq u \leq t } | \Delta _ { u } | ^ { 2 } ]$ . Taking expectations and supremum on both sides of $( 5 . 4 )$ , and applying the BDG inequality and Young’s inequality, we obtain

$$
\begin{array} { r l } & { F ( t ) \leq C \displaystyle \int _ { 0 } ^ { t } \mathbb { E } | \Delta _ { s } | ^ { 2 } \mathrm { d } s + C \displaystyle \int _ { 0 } ^ { t } \mathbb { E } | a _ { s } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } - a _ { s } ^ { \phi } | ^ { 2 } \mathrm { d } s + C \mathbb { E } \left[ \left( \displaystyle \int _ { 0 } ^ { t } | \Delta _ { s } | ^ { 2 } | \Delta \sigma _ { s } | ^ { 2 } \mathrm { d } s \right) ^ { 1 / 2 } \right] } \\ & { \qquad \leq \displaystyle \frac { 1 } { 2 } F ( t ) + C \displaystyle \int _ { 0 } ^ { t } F ( s ) \mathrm { d } s + C \displaystyle \int _ { 0 } ^ { t } \mathbb { E } | a _ { s } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } - a _ { s } ^ { \phi } | ^ { 2 } \mathrm { d } s . } \end{array}
$$

Thus, it follows that

$$
F ( t ) \leq C \int _ { 0 } ^ { t } F ( s ) \mathrm { d } s + C \int _ { 0 } ^ { t } \mathbb { E } | a _ { s } ^ { \mathcal { G } , \sigma _ { \exp } } - a _ { s } ^ { \phi } | ^ { 2 } \mathrm { d } s .\tag{5.5}
$$

It remains to bound the action discrepancy. For $s \in [ t _ { k } , t _ { k + 1 } )$ , since $\Pi _ { A }$ is nonexpansive and $\Pi _ { A } ( a _ { s } ^ { \phi } ) = a _ { s } ^ { \phi }$ , we have

$$
\begin{array} { r l } & { | a _ { s } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } - a _ { s } ^ { \phi } | = \left| \Pi _ { A } \left( \mu _ { \phi } ( t _ { k } , X _ { t _ { k } } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } ) + \sigma _ { \mathrm { e x p } } \zeta _ { k } \right) - \Pi _ { A } \left( \mu _ { \phi } ( s , X _ { s } ^ { \phi } ) \right) \right| } \\ & { \qquad \leq \left| \mu _ { \phi } ( t _ { k } , X _ { t _ { k } } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } ) - \mu _ { \phi } ( s , X _ { s } ^ { \phi } ) + \sigma _ { \mathrm { e x p } } \zeta _ { k } \right| } \\ & { \qquad \leq C \left( | X _ { t _ { k } } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } - X _ { t _ { k } } ^ { \phi } | + | X _ { t _ { k } } ^ { \phi } - X _ { s } ^ { \phi } | + | s - t _ { k } | + \sigma _ { \mathrm { e x p } } | \zeta _ { k } | \right) } \\ & { \qquad \leq C \left( \displaystyle \operatorname* { s u p } _ { 0 \leq u \leq s } | \Delta _ { u } | + | X _ { \iota ( s ) } ^ { \phi } - X _ { s } ^ { \phi } | + | \mathcal { G } | + \sigma _ { \mathrm { e x p } } | \zeta _ { k } | \right) . } \end{array}
$$

For the second term on the right side, we apply Itˆo’s formula and obtain

$$
\begin{array} { l } { { \displaystyle | X _ { s } ^ { \phi } - X _ { \iota ( s ) } ^ { \phi } | ^ { 2 } = 2 \int _ { \iota ( s ) } ^ { s } \langle X _ { u } ^ { \phi } - X _ { \iota ( s ) } ^ { \phi } , b ( u , X _ { u } ^ { \phi } , a _ { u } ^ { \phi } ) \rangle \mathrm { d } u + \int _ { \iota ( s ) } ^ { s } \mathrm { T r } \left( \sigma \sigma ^ { \top } ( u , X _ { u } ^ { \phi } , a _ { u } ^ { \phi } ) \right) \mathrm { d } u } \ ~ } \\ { { \displaystyle ~ + ~ 2 \int _ { \iota ( s ) } ^ { s } \langle X _ { u } ^ { \phi } - X _ { \iota ( s ) } ^ { \phi } , \sigma ( u , X _ { u } ^ { \phi } , a _ { u } ^ { \phi } ) \mathrm { d } W _ { u } \rangle + 2 \int _ { \iota ( s ) } ^ { s } \langle X _ { u } ^ { \phi } - X _ { \iota ( s ) } ^ { \phi } , \mathrm { d } K _ { u } ^ { \phi , - } - \mathrm { d } K _ { u } ^ { \phi , + } \rangle } . } \end{array}
$$

By the same argument as before, we know that the reflection term is non-positive, that is

$$
\int _ { \iota ( s ) } ^ { s } \langle X _ { u } ^ { \phi } - X _ { \iota ( s ) } ^ { \phi } , \mathrm { d } K _ { u } ^ { \phi , - } - \mathrm { d } K _ { u } ^ { \phi , + } \rangle \leq 0 .
$$

Then we obtain

$$
\begin{array} { r l } & { \mathbb { E } | X _ { s } ^ { \phi } - X _ { \iota ( s ) } ^ { \phi } | ^ { 2 } \leq \mathbb { E } \int _ { \iota ( s ) } ^ { s } \left( | X _ { u } ^ { \phi } - X _ { \iota ( s ) } ^ { \phi } | ^ { 2 } + | b ( u , X _ { u } ^ { \phi } , a _ { u } ^ { \phi } ) | ^ { 2 } + \operatorname { T r } \left( \sigma \sigma ^ { \top } ( u , X _ { u } ^ { \phi } , a _ { u } ^ { \phi } ) \right) \right) \mathrm { d } u } \\ & { \qquad \leq \displaystyle \int _ { \iota ( s ) } ^ { s } \mathbb { E } | X _ { u } ^ { \phi } - X _ { \iota ( s ) } ^ { \phi } | ^ { 2 } \mathrm { d } u + M ( s - \iota ( s ) ) , } \end{array}
$$

where $\begin{array} { r } { M : = \operatorname* { s u p } _ { t \in [ 0 , T ] , x \in Q , a \in A } \bigl ( | b ( t , x , a ) | ^ { 2 } + \operatorname { T r } ( \sigma \sigma ^ { \top } ( t , x , a ) ) \bigr ) } \end{array}$ . By Gronwall’s inequality, we conclude that E $| X _ { \iota ( s ) } ^ { \phi } - X _ { s } ^ { \phi } | ^ { 2 } \leq C | \iota ( s ) - s | \leq C | \mathcal { G } |$ for some constant $C > 0$ independent of $\phi$ and $\mathcal { G } .$

Thus, we have

$$
\begin{array} { r } { \mathbb { E } | a _ { s } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } - a _ { s } ^ { \phi } | ^ { 2 } \leq C F ( s ) + C \mathbb { E } | X _ { \iota ( s ) } ^ { \phi } - X _ { s } ^ { \phi } | ^ { 2 } + C | \mathcal { G } | ^ { 2 } + C \sigma _ { \mathrm { e x p } } ^ { 2 } \mathbb { E } | \zeta _ { k } | ^ { 2 } } \end{array}
$$

$$
\leq C F ( s ) + C | \mathcal G | + C \sigma _ { \exp } ^ { 2 } M _ { \zeta } .
$$

Substituting this bound into (5.5) yields

$$
F ( t ) \leq C \int _ { 0 } ^ { t } F ( s ) \mathrm { d } s + C \int _ { 0 } ^ { t } \left( | \mathcal { G } | + \sigma _ { \exp } ^ { 2 } M _ { \zeta } \right) \mathrm { d } s \leq C \int _ { 0 } ^ { t } F ( s ) \mathrm { d } s + C \left( | \mathcal { G } | + \sigma _ { \exp } ^ { 2 } M _ { \zeta } \right) .
$$

By Gronwall’s inequality, we have $F ( T ) \leq C \left( \left| \mathcal { G } \right| + \sigma _ { \exp } ^ { 2 } M _ { \zeta } \right)$ , which is precisely the desired error bound (5.3). □

## 5.2.4 Objective error

We now translate the state error bound into a bound on the diference between the ideal and the actually executed objective values. Recall that the ideal objective under the deterministic policy is

$$
J ( \phi ) : = \mathbb { E } \left[ \int _ { 0 } ^ { T } e ^ { - \beta t } r ( t , X _ { t } ^ { \phi } , a _ { t } ^ { \phi } ) \mathrm { d } t + e ^ { - \beta T } g ( X _ { T } ^ { \phi } ) \right] ,
$$

while the objective achieved by the discretely sampled exploratory policy is

$$
J _ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } ( \phi ) : = \mathbb { E } \left[ \int _ { 0 } ^ { T } e ^ { - \beta t } r ( t , X _ { t } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } , a _ { t } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } ) \mathrm { d } t + e ^ { - \beta T } g ( X _ { T } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } ) \right] .
$$

Corollary 5.2. Under the assumptions of Lemma 5.1, there exists a constant $C > 0$ , independent of $\mathcal { G }$ and $\sigma _ { \mathrm { e x p } } ,$ such that

$$
| J _ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } ( \phi ) - J ( \phi ) | \leq C \left( | \mathcal { G } | ^ { 1 / 2 } + \sigma _ { \mathrm { e x p } } M _ { \zeta } ^ { 1 / 2 } \right) .
$$

Proof. By Lipschitz continuity of $r$ in $( x , a )$ and $g$ in $x ,$ it follows that

$$
| J _ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } ( \phi ) - J ( \phi ) | \leq C \int _ { 0 } ^ { T } \mathbb { E } \left[ | X _ { t } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } - X _ { t } ^ { \phi } | + | a _ { t } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } - a _ { t } ^ { \phi } | \right] \mathrm { d } t + C \mathbb { E } | X _ { T } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } - X _ { T } ^ { \phi } | ,
$$

for some constant $C > 0$ . Applying Theorem 5.1 gives

$$
\begin{array} { r } { \mathbb { E } \vert X _ { t } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } - X _ { t } ^ { \phi } \vert \leq \left( \mathbb { E } \vert \Delta _ { t } \vert ^ { 2 } \right) ^ { 1 / 2 } \leq C \left( \vert \mathcal { G } \vert ^ { 1 / 2 } + \sigma _ { \mathrm { e x p } } M _ { \zeta } ^ { 1 / 2 } \right) . } \end{array}
$$

Similarly, from the proof of Theorem 5.1, we have

$$
\begin{array} { r } { \mathbb { E } | a _ { t } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } - a _ { t } ^ { \phi } | ^ { 2 } \leq C F ( t ) + C | \mathcal { G } | + C \sigma _ { \mathrm { e x p } } ^ { 2 } M _ { \zeta } \leq C \left( | \mathcal { G } | + \sigma _ { \mathrm { e x p } } ^ { 2 } M _ { \zeta } \right) , } \end{array}
$$

and hence $\mathbb { E } | a _ { t } ^ { \mathcal { G } , \sigma _ { \mathrm { e x p } } } - a _ { t } ^ { \phi } | \leq C ( | \mathcal { G } | ^ { 1 / 2 } + \sigma _ { \mathrm { e x p } } M _ { \zeta } ^ { 1 / 2 } )$ . Combining these estimates yields the desired objective error bound. □

Corollary 5.2 provides a quantitative guarantee that the performance of the discretely executed exploratory policy converges to that of the ideal deterministic policy as the time grid becomes finer $( | \mathcal { G } |  0 )$ and the exploration noise vanishes $( \sigma _ { \mathrm { e x p } }  0 )$ . The error bound reveals a trade-of inherent in the algorithm design: while exploration noise is necessary for learning the advantagerate function in an of-policy manner, it introduces a bias in the objective relative to the idea deterministic policy. Furthermore, this error analysis justifies the use of piecewise-constant action approximations in the implementation of the continuous-time algorithm. The bound assures us that the discrepancy between the theoretical ideal and the practical implementation is controlled by quantities that can be made arbitrarily small at the agent’s discretion, which provides a rigorous foundation for the efectiveness of the CT-DDPG-R algorithm.

## 6 Numerical Experiments

In this section, we execute the proposed CT-DDPG-R algorithm on a family of reflected reservoir control problems (Pliska (1975), Lai et al. (2022), Luo et al. (2025)). Reservoir systems provide a natural class of constrained continuous-time control tasks: the state variables represent waterlevel deviations from desired levels, and the admissible state region is a bounded box imposed by physical storage capacities and safety limits. We compare the soft constraint, hard constraint and the additional unconstrained case of the Neumann boundary condition in the value approximation.

## 6.1 One-dimensional reservoir with an explicit optimal solution

We first consider a one-dimensional reservoir control problem for which the optimal value function and optimal policy admit an explicit analytical solution. This experiment serves as a valuable test for validating the proposed algorithm, as it allows us to compare the learned approximations against the ground truth.

Let $X _ { t } \in [ - H , H ]$ denote the deviation of a reservoir level from its target level with $H > 0$ , and let $a _ { t } \in [ - A , A ]$ be the regulation rate $( \mathrm { i . e . }$ , the rate at which water is released or retained) with $A > 0$ . The reflected state dynamic is governed by

$$
\mathrm { d } X _ { t } = a _ { t } \mathrm { d } t + \sigma \mathrm { d } W _ { t } + \mathrm { d } K _ { t } ^ { - } - \mathrm { d } K _ { t } ^ { + } , \quad X _ { t } \in [ - H , H ] ,
$$

where the local time processes $K ^ { - }$ and $K ^ { + }$ increase only when the process X hits the lower boundary −H and the upper boundary H respectively. Our goal is to maximize the discounted reward

$$
V ^ { * } ( t , x ) = \operatorname* { s u p } _ { a \in \mathcal { A } } \mathbb { E } _ { t , x } \left[ \int _ { t } ^ { T } e ^ { - \beta ( s - t ) } r ( X _ { s } , a _ { s } ) \mathrm { d } s + e ^ { - \beta ( T - t ) } g ( X _ { T } ) \right] .
$$

Here the running and terminal rewards are given by $r ( x ) = - \eta x ^ { 2 } , g ( x ) = - \kappa x ^ { 2 }$ with $\eta , \kappa > 0$ which penalize deviations from the target water level during operation and at the terminal time, and A is the admissible set of all F-adapted processes $a = \{ a _ { s } \} _ { s \in [ 0 , T ] }$ taking values in $[ - A , A ]$

For this problem, the associated HJB equation takes the form

$$
\left\{ \begin{array} { l l } { \partial _ { t } V ^ { * } - \beta V ^ { * } + \displaystyle \frac { \sigma ^ { 2 } } { 2 } \partial _ { x x } V ^ { * } - \eta x ^ { 2 } + \operatorname* { s u p } _ { a \in [ - A , A ] } ( a \partial _ { x } V ^ { * } ) = 0 , } & { ( t , x ) \in [ 0 , T ) \times ( - H , H ) , } \\ { \partial _ { x } V ^ { * } ( t , - H ) = \partial _ { x } V ^ { * } ( t , H ) = 0 , } & { t \in [ 0 , T ) , } \\ { V ^ { * } ( T , x ) = - \kappa x ^ { 2 } , } & { x \in [ - H , H ] . } \end{array} \right.
$$

By symmetry, $V ^ { * } ( t , x )$ is even in x. Since the value decreases as |x| increases, we have $\partial _ { x } V ^ { \ast } ( t , x ) \leq 0$ for $x > 0$ , and $\partial _ { x } V ^ { \ast } ( t , x ) \geq 0$ for $x < 0$ . Therefore, the optimal policy has a bang-bang form:

$$
\mu ^ { * } ( t , x ) = \underset { a \in [ - A , A ] } { \arg \operatorname* { m a x } } \left( a \partial _ { x } V ^ { * } ( t , x ) \right) = - A \mathrm { s g n } ( x ) .\tag{6.1}
$$

Thus the optimal controller pushes the reservoir level toward the target level with maximal admissible intensity. Let $y = | x | , \tau = T - t .$ and write $V ^ { * } ( t , x ) = - C ( \tau , y )$ . Under the bang-bang

control (6.1), the transformed function C satisfies the following linear parabolic equation on the half-interval [0, H]

$$
\left\{ \begin{array} { l l } { \displaystyle \partial _ { \tau } C = \frac { \sigma ^ { 2 } } { 2 } \partial _ { y y } C - A \partial _ { y } C - \beta C + \eta y ^ { 2 } , } & { ( \tau , y ) \in ( 0 , T ] \times ( 0 , H ) , } \\ { \displaystyle \partial _ { y } C ( \tau , 0 ) = \partial _ { y } C ( \tau , H ) = 0 , } & { \tau \in ( 0 , T ] , } \\ { C ( 0 , y ) = \kappa y ^ { 2 } , } & { y \in [ 0 , H ] . } \end{array} \right.
$$

This equation is linear and can be solved explicitly via eigenfunction expansion. Define the constants $\textstyle \alpha = { \frac { A } { \sigma ^ { 2 } } } , k _ { n } = { \frac { n \pi } { H } }$ for $n \geq 1$ . The eigenfunctions of the operator $\begin{array} { r } { \mathcal { L } f = \frac { \sigma ^ { 2 } } { 2 } f ^ { \prime \prime } - A f ^ { \prime } } \end{array}$ with Neumann boundary conditions $f ^ { \prime } ( 0 ) = f ^ { \prime } ( H ) = 0$ are $\phi _ { 0 } ( y ) = 1$ , and $\begin{array} { r } { \phi _ { n } ( y ) = e ^ { \alpha y } [ \cos ( k _ { n } y ) - \frac { \alpha } { k _ { n } } \sin ( k _ { n } y ) ] } \end{array}$ for $n \geq 1$ , with corresponding eigenvalues $\begin{array} { r } { \lambda _ { 0 } = 0 , \lambda _ { n } = \frac { A ^ { 2 } } { 2 \sigma ^ { 2 } } + \frac { \sigma ^ { 2 } k _ { n } ^ { 2 } } { 2 } } \end{array}$ for $n \geq 1$ . Let

$$
f _ { n } = \frac { \displaystyle \int _ { 0 } ^ { H } e ^ { - 2 \alpha y } y ^ { 2 } \phi _ { n } ( y ) \mathrm { d } y } { \displaystyle \int _ { 0 } ^ { H } e ^ { - 2 \alpha y } \phi _ { n } ^ { 2 } ( y ) \mathrm { d } y } , \quad n \ge 0 .
$$

Then the expansion $\begin{array} { r } { y ^ { 2 } = \sum _ { n \geq 0 } f _ { n } \phi _ { n } ( y ) } \end{array}$ holds, and the solution of the transformed PDE is given by

$$
C ( \tau , y ) = \sum _ { n = 0 } ^ { \infty } \left[ e ^ { - ( \beta + \lambda _ { n } ) \tau } \kappa f _ { n } + \eta \frac { 1 - e ^ { - ( \beta + \lambda _ { n } ) \tau } } { \beta + \lambda _ { n } } f _ { n } \right] \phi _ { n } ( y ) .
$$

Thus, by a standard verification argument, the optimal value function admits the explicit series representation

$$
V ^ { * } ( t , x ) = - \sum _ { n = 0 } ^ { \infty } \left[ e ^ { - ( \beta + \lambda _ { n } ) ( T - t ) } \kappa f _ { n } + \eta \frac { 1 - e ^ { - ( \beta + \lambda _ { n } ) ( T - t ) } } { \beta + \lambda _ { n } } f _ { n } \right] \phi _ { n } ( | x | ) .\tag{6.2}
$$

In the numerical experiment, the infinite series in (6.2) is approximated by the first 150 eigenfunctions, and the coeficients $f _ { n }$ are computed by numerical quadrature on 4000 grid points, ensuring suficient accuracy for the ground truth. The parameters are $T = 1 , \beta = 0 . 8 , \sigma = 0 . 3 , H =$ $1 , A = 1 , \kappa = 1 , \eta = 5$ . The value network, advantage-rate network, and policy network are fully connected neural networks with three hidden layers, 64 hidden units per layer and hyperbolic tangent activations. The actor output is bounded by A tanh(·). The hard value network uses the cosine frequencies {1, 2, 3, 4, 6, 10, 14} for the rescaled state variable. We train for 3000 episodes with $\Delta t = 0 . 0 1$ with Gaussian exploration noise decaying from 0.5 to 0.2. The deterministic return is evaluated every 50 episodes using 100 rollout paths.

Figure 1 reports the returns without exploration and the boundary residual during the updating process. We can see a clear separation among the three boundary treatments in terms of the Neumann boundary residual which confirms that both boundary-aware treatments efectively encode the structure of the value network. It is worth noting that the unconstrained neural networks, due to strong expression power, are still able to recover the value function and the optimal bangbang policy with reasonable accuracy; see Figures 2. Nevertheless, we find that the unconstrained training process exhibits more pronounced oscillations and occasional degradation. In contrast, the soft and hard boundary treatments provide additional structural information and therefore lead to more stable learning dynamics in many runs.

At the same time, this experiment also shows a limitation of imposing the boundary condition that the learning process may quickly drive the policy toward the boundary and remain trapped there for a long period of training, especially under the hard constraint. This issue is no longer observed in the subsequent reservoir experiments, where the running reward contains a logarithmic boundary penalty. In favor of the boundary penalty, as the dimension of the state space increases and the value function and policy become more complex, the benefit of incorporating the Neumann constraints becomes more visible and they lead to faster and more stable convergence than the unconstrained baseline.

![](images/14a96cf67b237a49868a9a5e1ca5c15cd89dc3726bbbecb704c5935f5f2562e6.jpg)

![](images/21014ddd9f5e8a762e99ba9c71898aa920ccde1b6660b265984849d0d9cba43f.jpg)  
Figure 1: Returns and boundary residual

![](images/4bf0ea2bfd86688e7d431aa8abe67be78f543934cde5f16c65ad455264ba7a1a.jpg)  
Figure 2: Value functions and policies under diferent constraints

## 6.2 Two-dimensional reservoir cascade regulation

We next consider a two-dimensional reservoir cascade. Let $X _ { t } ^ { i } \in [ - H _ { i } , H _ { i } ]$ be the water-level deviation of reservoir $i ,$ and let $a _ { t } ^ { i } \in [ - A _ { i } , A _ { i } ]$ be the relative release rate of reservoir i. A larger release from reservoir 1 reduces the upstream level but increases the downstream inflow into reservoir 2. The reflected dynamics are governed by the following system of RSDEs:

$$
\begin{array} { r l } & { \mathrm { d } X _ { t } ^ { 1 } = \left( m _ { 1 } ( t ) - a _ { t } ^ { 1 } \right) \mathrm { d } t + \sigma _ { 1 } \mathrm { d } W _ { t } ^ { 1 } + \mathrm { d } K _ { t } ^ { - , 1 } - \mathrm { d } K _ { t } ^ { + , 1 } , } \\ & { \mathrm { d } X _ { t } ^ { 2 } = \left( m _ { 2 } ( t ) + \rho a _ { t } ^ { 1 } - a _ { t } ^ { 2 } \right) \mathrm { d } t + \sigma _ { 2 } \mathrm { d } W _ { t } ^ { 2 } + \mathrm { d } K _ { t } ^ { - , 2 } - \mathrm { d } K _ { t } ^ { + , 2 } . } \end{array}
$$

Here $m _ { i } ( t )$ is the mean net inflow deviation of reservoir $i ,$ which may vary over time due to seasonal or stochastic factors, and $\rho \in [ 0 , 1 ]$ is the downstream transfer coeficient. The term $\rho { a } _ { t } ^ { 1 }$ models the fraction of the upstream release that enters the downstream reservoir. The reflection terms $K ^ { \pm , i }$ ensure that the state process remains within the admissible hypercube $[ - H _ { 1 } , H _ { 1 } ] \times [ - H _ { 2 } , H _ { 2 } ]$ by enforcing normal reflection at the boundaries.

The running reward is

$$
\begin{array} { l } { \displaystyle r ( t , x , a ) = - \Bigg [ q _ { 1 } x _ { 1 } ^ { 2 } + q _ { 2 } x _ { 2 } ^ { 2 } + q _ { 1 2 } \left( \frac { x _ { 1 } } { H _ { 1 } } - \frac { x _ { 2 } } { H _ { 2 } } \right) ^ { 2 } + \lambda _ { 1 } a _ { 1 } ^ { 2 } + \lambda _ { 2 } a _ { 2 } ^ { 2 } } \\ { \displaystyle \qquad + \left. \gamma _ { b } \left[ - \log \left( 1 - \left( \frac { x _ { 1 } } { H _ { 1 } } \right) ^ { 2 } + \varepsilon _ { b } \right) - \log \left( 1 - \left( \frac { x _ { 2 } } { H _ { 2 } } \right) ^ { 2 } + \varepsilon _ { b } \right) \right] \right] . } \end{array}
$$

The first two terms penalize deviations of each reservoir from its target level, encouraging the system to maintain desired storage volumes. The third term penalizes imbalance between the relative storage levels of the two reservoirs, promoting an equitable distribution of water across the cascade. The fourth and fifth terms penalize aggressive control which causes operational costs. The logarithmic terms serve as barrier functions that increasingly penalize states approaching the boundary of the admissible domain, thereby discouraging the system from operating near the physical limits. The terminal reward is analogously defined as

$$
g ( x ) = - \left[ \kappa _ { 1 } x _ { 1 } ^ { 2 } + \kappa _ { 2 } x _ { 2 } ^ { 2 } + \kappa _ { 1 2 } \left( \frac { x _ { 1 } } { H _ { 1 } } - \frac { x _ { 2 } } { H _ { 2 } } \right) ^ { 2 } \right] .
$$

In this experiment, the parameters are set as $T = 1 , \beta = 0 . 5 , \Delta t = 0 . 0 1 , H _ { 1 } = H _ { 2 } = A _ { 1 } =$ $A _ { 2 } = 1 , \sigma _ { 1 } = \sigma _ { 2 } = 0 . 4 5 , \rho = 0 . 6 5 , m _ { 1 } ( t ) = 0 . 2 5 \sin ( 2 \pi t / T ) , m _ { 2 } ( t ) = 0 . 2 0 \cos ( 2 \pi t / T ) , q _ { 1 } = q _ { 2 } = 1$ $q _ { 1 2 } = 0 . 8 , \lambda _ { 1 } = \lambda _ { 2 } = 0 . 0 3 , \kappa _ { 1 } = \kappa _ { 2 } = 1 , \kappa _ { 1 2 } = 0 . 8 , \gamma _ { b } = 0 . 0 3$ , and $\varepsilon _ { b } = 1 0 ^ { - 3 }$ . The value, advantagerate, and actor networks are fully connected networks with three hidden layers, 128 hidden units per layer, and tanh activations. For the hard Neumann method, the value network replaces the raw state input by cosine features with frequencies {1, 2, 3, 4, 6, 8} in each coordinate. The actor is bounded componentwise by $( A _ { 1 } , A _ { 2 } )$ · tanh(·). We train for 4000 episodes and the returns during training can be seen in Figure 3 with five times the standard deviation for visual clarity. Since no closed-form solution is available for this two-dimensional problem, we evaluate the learned policies qualitatively by visualizing their surfaces at selected time slice. Figures 4, 5, and 6 display the learned policy functions for the unconstrained, soft-constrained, and hard-constrained approaches, respectively.

![](images/2de289325fa266eed2ef004051cac63329b880fd820617fcdaf832b9d10293dd.jpg)  
Figure 3: Returns during training

To provide a quantitative reference, we also compute a finite-diference approximation of the optimal policy by numerically solving the associated HJB equation using the method of Kushner and Dupuis (2001); the resulting reference policy is shown in Figure 7. While the finite-diference reference exhibits some numerical oscillations owing to the limitations of grid-based discretization in two dimensions, the overall qualitative structure is broadly consistent across all methods. Notably, the agreement with the reference policy becomes more apparent when the Neumann boundary condition is incorporated through either the soft or hard constraint, suggesting that boundary awareness helps the neural network capture the correct boundary behavior. These results further support the conclusion that the proposed boundary-aware treatments efectively encode the structural information which is helpful for stable and accurate learning in multidimensional settings.

![](images/b5592ea64354a7f04f726adedc011bf662ade4dea3ba7e55ff40bce0559686b0.jpg)  
Figure 4: Policy surfaces without boundary constraint at t = 0.5.

## 6.3 Five-dimensional reservoir cascade regulation

Finally, to assess the scalability of the proposed method to higher-dimensional settings, we consider a five-dimensional reservoir cascade. The state vector is given by

$$
X _ { t } = ( X _ { t } ^ { 1 } , \ldots , X _ { t } ^ { 5 } ) \in Q : = \prod _ { i = 1 } ^ { 5 } [ - H _ { i } , H _ { i } ] ,
$$

![](images/04f8c206a0c6587c3abde6302e015c5deb1e80ff07fd8e509bf045b5e904794c.jpg)  
Figure 5: Policy surfaces under soft constraint at $t = 0 . 5 .$

![](images/c7681796a0806344e00acf97814d86bf560ceec6446a01ab3dfd609ee64b8ab9.jpg)  
Figure 6: Policy surfaces under hard constraint at $t = 0 . 5$

where $X _ { t } ^ { i }$ is the water-level deviation of reservoir i from its target level. The control $\begin{array} { r l } { a _ { t } } & { { } = } \end{array}$ $( a _ { t } ^ { 1 } , \ldots , a _ { t } ^ { 5 } ) \in \prod _ { i = 1 } ^ { 5 } [ - A _ { i } , A _ { i } ]$ represents the release rates. The reflected difusion dynamics are governed by

$$
\mathrm { d } X _ { t } = b ( t , X _ { t } , a _ { t } ) \mathrm { d } t + \mathrm { d i a g } ( \sigma _ { 1 } , \dots , \sigma _ { 5 } ) \mathrm { d } W _ { t } + \mathrm { d } K _ { t } ^ { - } - \mathrm { d } K _ { t } ^ { + } .
$$

The drift structure remains the same as in the two-dimensional cascade, with each reservoir receiving inflow from its upstream neighbor: $b _ { 1 } ( t , x , a ) = m _ { 1 } ( t ) - a _ { 1 }$ and $b _ { i } ( t , x , a ) = m _ { i } ( t ) + \rho _ { i - 1 } a _ { i - 1 } - a _ { i }$ for $i = 2 , \ldots , 5$ . We set the net inflow deviations to $m _ { i } ( t ) = \bar { m } _ { i } \sin ( 2 \pi t / T + \varphi _ { i } ) , \mathrm { f o r } i = 1 , \ldots , 5$

The running reward is designed to capture the same operational objectives as in the lowerdimensional cases, now extended to five reservoirs:

$$
r ( t , x , a ) = - \sum _ { i = 1 } ^ { 5 } \left[ ( q _ { i } x _ { i } ^ { 2 } + \lambda _ { i } a _ { i } ^ { 2 } ) - \gamma _ { b } \log \left( 1 - \left( \frac { x _ { i } } { H _ { i } } \right) ^ { 2 } + \varepsilon _ { b } \right) \right] - q _ { a d j } \sum _ { i = 1 } ^ { 4 } \left( \frac { x _ { i } } { H _ { i } } - \frac { x _ { i + 1 } } { H _ { i + 1 } } \right) ^ { 2 } ,
$$

and the terminal reward is analogously defined as

$$
g ( x ) = - \left[ \sum _ { i = 1 } ^ { 5 } \kappa _ { i } x _ { i } ^ { 2 } + \kappa _ { a d j } \sum _ { i = 1 } ^ { 4 } \left( \frac { x _ { i } } { H _ { i } } - \frac { x _ { i + 1 } } { H _ { i + 1 } } \right) ^ { 2 } \right] .
$$

![](images/18e302c57733c65a414abcc01f2014fe56bdc9187d88284defe8f5dd00770cec.jpg)  
Figure 7: Finite-diference reference policy at t = 0.5.

In the experiment, we choose the parameters as $T = 1 , \beta = 0 . 5 , \Delta t = 0 . 0 1 , H _ { i } = A _ { i } = q _ { i } =$ $\kappa _ { i } = 1 , \lambda _ { i } = 0 . 0 3$ , for $i = 1 , \dots , 5 , ( \sigma _ { 1 } , \dots , \sigma _ { 5 } ) = ( 0 . 4 5 , 0 . 4 3 , 0 . 4 0 , 0 . 3 8 , 0 . 3 5 ) , ( \rho _ { 1 } , \rho _ { 2 } , \rho _ { 3 } , \rho _ { 4 } ) =$ (0.65, 0.55, 0.45, 0.35), $( { \bar { m } } _ { 1 } , \ldots , { \bar { m } } _ { 5 } ) = ( 0 . 2 5 , 0 . 2 2 , 0 . 2 0 , 0 . 1 8 , 0 . 1 6 ) , ( \varphi _ { 1 } , \ldots , \varphi _ { 5 } ) = \left( 0 , { \frac { \pi } { 2 } } , \pi , { \frac { 3 \pi } { 2 } } , { \frac { \pi } { 4 } } \right)$ and $q _ { a d j } = \kappa _ { a d j } = 0 . 8 , \gamma _ { b } = 0 . 0 3 , \varepsilon _ { b } = 1 0 ^ { - 3 }$ . The value, advantage-rate, and actor networks use five hidden layers with 128 hidden units per layer and tanh activations. The hard Neumann value network uses cosine frequencies {1, 2, 3, 4, 6, 8, 10, 12} in each state coordinate. We train for 4000 episodes and display the returns in Figure 8 with five times the standard deviation for visual clarity. The same as in the two-dimensional experiment, the results here demonstrate that both the soft and hard boundary treatments lead to more stable and faster convergence than the unconstrained baseline. The boundary-aware methods achieve slightly better final performance, as showed in the return curves, and the learned policy surfaces exhibit smoother and more physically plausible behavior. This observation reinforces the central message of this paper: incorporating structural information about the Neumann boundary condition into the value network, whether through a soft penalty or a hard architectural constraint, provides significant benefits in terms of learning stability and final performance, particularly as the state dimension increases and the control problem becomes more complex.

![](images/6dbd14717c1f8c3bab184e8ef21dfe5a6a5aff1f926a025aed4786d1f6c8fdf5.jpg)  
Figure 8: Returns during training.

## References

L. C. Baird. Reinforcement learning in continuous time: Advantage updating. In Proceedings of the IEEE International Conference on Neural Networks, 4:2448–2453, 1994.

L. Bo, Y. Huang and X. Yu. A decomposition-homogenization method for Robin boundary problems on the nonnegative orthant. Electronic Journal of Probability, 29:1-25, 2024.

L. Bo, Y. Huang, X. Yu and T. Zhang. Continuous-time q-learning in jump-difusion models under Tsallis entropy. Mathematics and Financial Economics, online first, https://doi.org/10.1007/s11579-026-00419-4, 2026.

L. Bo, Y. Huang and X. Yu. On optimal tracking portfolio in incomplete markets: The reinforcement learning approach. SIAM Journal on Control and Optimization, 63(1):321–348, 2025.

Z. Cheng, X. Guo and Y. Zhang. Deterministic Policy Gradient for Reinforcement Learning with Continuous Time and State. arXiv preprint arXiv:2509.23711, 2025.

Z. Cheng, X. Guo, H. Pham and Y. Zhang. Actor-critic learning for extended mean field control with deterministic policies. arXiv preprint arXiv:2607.11005, 2026.

K. Doya. Reinforcement learning in continuous time and space. Neural Computation, 12(1):219–245, 2000.

P. Dupuis and H. Ishii. On lipschitz continuity of the solution mapping to the skorokhod problem, with applications. Stochastics and Stochastic Reports, 35(1):31-62, 1991.

X. Gao, L. Li and X. Y. Zhou. Reinforcement learning for jump-difusions, with financial applications. Mathematical Finance, 1-32, 2025.

X. Guo, Y. Huang and X. Yu. Deterministic policy gradient for learning equilibrium in time-inconsistent control problems. arXiv preprint arXiv:2606.11798, 2026.

Y. Jia and X. Y. Zhou. Policy evaluation and temporal-diference learning in continuous time and space: A martingale approach. Journal of Machine Learning Research, 23(154):1–55, 2022.

Y. Jia and X. Y. Zhou. Policy gradient and actor-critic learning in continuous time and space: Theory and algorithms. Journal of Machine Learning Research, 23(275):1–50, 2022.

Y. Jia and X. Y. Zhou. q-Learning in continuous time. Journal of Machine Learning Research, 24(161):1-61, 2023.

Y. Jia, D. Ouyang and Y. Zhang. Accuracy of discretely sampled stochastic policies in continuous-time reinforcement learning. SIAM Journal on Control and Optimization, 64(3):1889-1929, 2026.

W. Kang and K. Ramanan. Characterization of stationary distributions of reflected difusions. The Annals of Applied Probability, 24(4):1329-1374, 2014.

V. R. Konda and J. N. Tsitsiklis. Actor-critic algorithms. In Advances in Neural Information Processing Systems, 12, 2000.

H. J. Kushner and P. Dupuis. Numerical Methods for Stochastic Control Problems in Continuous Time. Springer, second edition, 2001.

V. Lai, Y. F. Huang, C. H. Koo, A. N. Ahmed and A. El-Shafie. A Review of Reservoir Operation Optimisations: from Traditional Models to Metaheuristic Algorithms Arch Computat Methods Eng, 29:3435–3457, 2022.

T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver and D. Wierstra. Continuous control with deep reinforcement learning. In International Conference on Learning Representations, 2015.

W. Luo, C. Wang, Y. Zhang, J. Zhao, Z. Huang, J. Wang and C. Zhang. A deep reinforcement learning approach for joint scheduling of cascade reservoir system. Journal of Hydrology, 2025.

V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg and D. Hassabis. Human-level control through deep reinforcement learning. Nature, 518:529–533, 2015.

R. Munos and P. Bourgine. Reinforcement learning for continuous stochastic control problems. In Advances in Neural Information Processing Systems, 10, 1997.

A. Pilipenko: An introduction to stochastic diferential equations with reflection. Potsdam University Press, 2014.

S. R. Pliska: A difusion process model for the optimal operation of a reservoir system. Journal of Applied Probability, 12(4):859-863, 1975.

Z. Ren, X. Wei, X. Yu and X. Y. Zhou. Continuous-time q-learning for mean-field control with common noise, Part I: Theoretical foundations. arXiv preprint arXiv:2604.27372, 2026.

Z. Ren, X. Wei, X. Yu and X. Y. Zhou. Continuous-time q-learning for mean-field control with common noise, Part II: q-learning algorithms. arXiv preprint arXiv:2604.27378, 2026.

G. A. Rummery and M. Niranjan. On-line Q-learning using connectionist systems. Technical Report, 1994.

D. Silver, G. Lever, N. Heess, T. Degris, D. Wierstra and M. Riedmiller. Deterministic policy gradient algorithms. In Proceedings of the 31st International Conference on Machine Learning, PMLR 32(1):387– 395, 2014.

C. Straub, P. Brendel, V. Medvedev and A. Rosskopf: Hard-constraining Neumann boundary conditions in physics-informed neural networks via Fourier feature embeddings. ICLR 2025 Workshop on Machine Learning Multiscale Processes, 2025.

R. S. Sutton. Learning to predict by the methods of temporal diferences. Machine Learning, 3:9–44, 1988.

R. S. Sutton, D. McAllester, S. Singh and Y. Mansour. Policy gradient methods for reinforcement learning with function approximation. In Advances in Neural Information Processing Systems, 12, 2000.

R. S. Sutton and A. G. Barto. Reinforcement Learning: An Introduction. MIT Press, second edition, 2018.

C. Tallec, L. Blier and Y. Ollivier. Making deep Q-learning methods robust to time discretization. In Proceedings of the 36th International Conference on Machine Learning, PMLR 97:6096–6104, 2019.

H. Wang, T. Zariphopoulou and X. Y. Zhou: Reinforcement Learning in Continuous Time and Space: A Stochastic Control Approach. Journal of Machine Learning Research, 21(198):1–34, 2020.

C. J. C. H. Watkins. Learning from delayed rewards. Robotics & Autonomous Systems, 1989

C. J. C. H. Watkins and P. Dayan. Technical note: Q-learning. Machine Learning, 8:279–292, 1992.

X. Wei and X. Yu. Continuous-time q-learning for mean-field control problems. Applied Mathematics and Optimization, 91(10), 2025.

X. Wei, X. Yu and F. Yuan. Unified continuous-time q-learning for mean-field game and mean-field control problems. arXiv preprint arXiv:2407.04521 2024.

R. J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8:229–256, 1992.

A. Z˘alinescu: Second order Hamilton–Jacobi–Bellman equations with an unbounded operator. Nonlinear Anal., 75(13):4784-4797, 2012.

H. Zhao, W. Tang and D. D. Yao. Policy optimization for continuous reinforcement learning. In Advances in Neural Information Processing Systems, 36, 2023.