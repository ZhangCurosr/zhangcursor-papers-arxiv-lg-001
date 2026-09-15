# Bridging Control, Inference, Transport, and Thermodynamics From Theory to Applications in Learning

Emmy Blumenthal,<sup>1,</sup> <sup>∗</sup> Nikolas Claussen,<sup>1,</sup> <sup>2,</sup> <sup>†</sup> Benjamin Eysenbach,<sup>3,</sup> <sup>‡</sup> Catherine Ji,<sup>1,</sup> <sup>§</sup> Gautam Reddy,<sup>1,</sup> <sup>¶</sup> Colin Scheibner,<sup>1,</sup> <sup>2,</sup> <sup>∗∗</sup> and Benjamin Sorkin<sup>1,</sup> <sup>2,</sup> <sup>††</sup>

<sup>1</sup>Joseph Henry Laboratories of Physics, Princeton University, Princeton, New Jersey 08544, USA <sup>2</sup>Princeton Center for Theoretical Science, Princeton University, Princeton, New Jersey 08544, USA <sup>3</sup>Department of Computer Science, Princeton University, Princeton, New Jersey 08544, USA (Dated: September 15, 2026)

The last decade has seen the development of powerful methods for learning complex structure from highdimensional data. These advances have brought to the foreground fundamental connections between subdisciplines of physics, applied mathematics, and machine learning. In this review, we bring together some of these ideas, often expressed in different languages, to highlight a conceptual thread that links five distinct fields: control theory, optimal transport, probabilistic inference, non-equilibrium thermodynamics, and machine learning. A common theme is the optimization of free-energy-like functionals under dynamical or statistical constraints. We offer a guided tour through this thread and present selected applications in reinforcement learning, variational inference, and generative modeling. The review does not assume prior familiarity with these topics, and begins with principles originating from physics.

## CONTENTS

Introduction 2   
I. Variational structure of mechanics, control and   
inference 4   
A. Finding optimal paths in classical mechanics and   
control theory 4   
B. Variational basis for probabilistic inference and   
control 6   
1. Sanov’s theorem and the inference method 6   
2. Maximum-entropy inference 7   
C. Probabilistic inference over paths 8   
1. From inference to distributions over paths 9   
2. Girsanov’s theorem 10   
3. Probabilistic inference as a control problem 10   
D. Mathematical structure of path integral stochastic   
control 10   
1. The Cole–Hopf transformation of the HJB   
equation 11   
2. Feynman–Kac formula and Doob’s   
h-transform 11   
3. Example: Brownian bridge 12   
II. Control and transport of densities 13   
A. The Schrodinger bridge problem¨ 13   
1. From trajectories to densities 13   
2. The coupled forward–backward construction 13   
B. Entropic optimal transport and the Wasserstein   
distance 15   
C. Wasserstein geometry and Wasserstein gradient   
flows 17   
III. Thermodynamics 19   
A. Equilibrium thermodynamics as   
maximum-entropy inference 19   
B. Nonequilibrium thermodynamics as a control   
problem 20   
1. Optimal transport bounds efficiency via   
Onsager’s least-dissipation principle 21   
2. Dynamics far from equilibrium 22   
3. Geometric interpretation of entropy   
production 23   
4. Thermodynamic fluctuation theorems 25   
IV. Sampling and control 26   
A. What is sampling and why is it hard? 26   
1. The Monte Carlo method 26   
2. Probability distributions in high dimension 27   
3. Basic sampling algorithms: importance and   
Langevin sampling 27   
B. Annealed importance sampling 28   
1. Sampling with time-varying energies 28   
2. Proof of the annealed importance sampling   
identity 29   
C. Sampling as a control problem 30   
1. Sampler design as a control problem 30   
2. Non-equilibrium systems and transition path   
sampling 31   
3. Control as a sampling problem 31   
V. Applications 32   
A. Reinforcement learning 32   
1. The Markov decision process 33   
2. An overview of online RL algorithms 34   
3. MaxEnt RL & Soft Q-learning 36   
4. Soft actor-critic 39   
B. Wasserstein gradient flows 40   
1. The Jordan-Kinderlehrer-Otto (JKO) scheme 40   
2. The mean-field limit of interacting particles 41   
3. Variational inference 43   
C. Generative modeling with flows and diffusion 46   
1. Variational transform sampling 46   
2. Flow matching 47   
3. The score function 48   
4. From flow matching to diffusion 49   
5. One-step generation and normalizing flows 50   
6. Generalizations and ongoing work 51   
Contributions and acknowledgments 52   
References 52   
Tables of notation 57

## INTRODUCTION

Recent decades have seen the development of remarkably powerful tools for learning complex structure from data. These methods have brought to the foreground certain key connections between subdisciplines of physics and applied mathematics. The conceptual ideas have appeared in different fields at different times, sometimes rediscovered and often expressed in different languages. In this review, we bring together some of these connections to highlight a fascinating conceptual thread that links five distinct fields: control, transport, inference, thermodynamics and learning. The visual table of contents, Fig. 1, lays out the topics we cover and the links between them. These connections have proved fruitful for the development of powerful practical algorithms for generative modeling and control. We offer a guided tour through this thread and present selected applications in reinforcement learning, variational inference, and generative modeling. The review is written in the language of physicists and does not assume prior familiarity with these topics. Ideas from mechanics and thermodynamics are used as familiar anchors to introduce concepts in other areas.

The conceptual thread begins at the level of paths (left column in Fig. 1) in Variational structure of mechanics, control, and inference. This chapter develops a common variational language that is reused throughout the review. We introduce optimal control, which seeks the best way to steer a dynamical system—for example, guiding a rocket to a target while minimizing fuel consumption. We illustrate two complementary structural approaches for solving control problems. Introducing Lagrange multipliers for imposing a constraint on the dynamics leads to Hamiltonian equations. Another technique, known as dynamic programming, reformulates the optimization as a recursive equation. This reformulation leads to the Hamilton–Jacobi equation and, for stochastic control, its more general Hamilton–Jacobi–Bellman (HJB) counterpart. Controls and rewards play roles analogous to velocities and the negative Lagrangian in classical mechanics. We then pass from optimizing individual paths to distributions (right column in Fig. 1). The Maximum Entropy Principle (Max-Ent) selects a probability distribution that remains as close as possible to a reference distribution, as measured by the Kullback–Leibler (KL) divergence, while remaining consistent with data. We then generalize these tools for probabilistic inference to distributions over paths. For controlled driftdiffusive processes, Girsanov’s theorem identifies this pathspace KL divergence with a quadratic control cost, thereby connecting inference and control.

The next chapter, Control and transport of densities, illustrates how the control of paths can be used to obtain a geometric description for the evolution of probability densities. The topic of transport brings together ideas from the study of (stochastic) paths and of distributions (center column in Fig. 1). The first step is to impose constraints on both the initial and final densities of a stochastic control problem. These constraints lead to the Schrodinger bridge problem¨ , which finds the most likely ensemble of stochastic paths that connects the two densities. If one retains the endpoint constraints but describes only how initial and final positions are paired, the problem becomes entropic optimal transport. In general, optimal transport asks how to move one density into another at minimum cost, much like reshaping a pile of earth while minimizing the cost of moving it. The entropic version adds a term that favors spreading probability among different pairings. For a quadratic cost, the zero-noise limit defines the 2-Wasserstein distance, which has special mathematical properties. This distance equips the space of probability densities with a geometry and leads to Wasserstein gradient flows: steepest-descent dynamics in which the evolving object is a probability distribution.

The chapter on Thermodynamics grounds this geometry in physics. At equilibrium, statistical mechanics assigns probabilities to microscopic states while retaining only a small number of macroscopic constraints. Maximum-entropy inference reproduces the familiar Gibbs distributions from constraints on mean energy, making a heuristic connection between inference and equilibrium thermodynamics. Away from equilibrium, control and transport describe finite-time transformations between distributions. Onsager’s least-dissipation principle generates phenomenological field theories whose dissipation rate is bounded by the Wasserstein distance. For stochastic dynamics farther from equilibrium, the Wasserstein transport cost provides thermodynamic speed limits on the entropy produced during a finite-time transformation. Nonequilibrium work fluctuation relations provide a complementary link between fluctuations, free-energy differences, and the energetic cost of driving. Thermodynamics therefore supplies physical meaning for the variational structures introduced earlier, which in turn furnish quantitative limits on thermodynamic processes.

The fourth conceptual chapter, Sampling and control, presents sampling as a general approach to probabilistic inference, optimization, and control. Monte Carlo methods estimate averages from samples, while simulated annealing searches for low-energy states. Exploring possible trajectories and weighting them according to their rewards also links sampling to control and reinforcement learning; the Feynman–

![](images/ab3fd054888a8caa2de51a281c673acccf827562c32305f326cd23eb2c4d7fb1.jpg)  
FIG. 1. Graphical table of contents: topics of the review and the connections between them.

Kac formula makes this connection precise for a class of stochastic control problems. Sampling faces several distinct challenges, including high dimensionality, separated modes that impede efficient exploration, and observables whose averages are dominated by rare trajectories. When an unnormalized target density is available, Annealed Importance Sampling (AIS) replaces direct sampling by a sequence of intermediate distributions and corrects the resulting nonequilibrium process through trajectory-dependent weights. The control perspective shows how guiding the dynamics toward important trajectories can reduce fluctuations. Although this ideal control is generally unknown, it provides a target for practical algorithms that learn progressively better sampling dynamics. This chapter thus closes the conceptual thread by turning efficient sampling into a control problem.

These connections provide a framework and common language for Applications (bottom row in Fig. 1). The first application is Reinforcement learning. Stochastic optimal control assumes that the dynamics of the environment are known. Reinforcement learning instead seeks good controls—called actions—by interacting with an environment and observing the resulting states and rewards. A policy gives the probability of taking each action in a given state and therefore defines an ensemble of trajectories. The connection between inference and control leads to Maximum Entropy Reinforcement Learning (MaxEnt RL), in which high-reward trajectories are favored while the policy retains some randomness. Applying dynamic programming to this objective leads to soft Qlearning and Soft Actor-Critic (SAC), practical algorithms for learning stochastically controlled paths when the dynamics are not known in advance.

The second application concerns Wasserstein gradient flows. The geometry of probability distributions provides a variational method for integrating Fokker–Planck equations. The same viewpoint helps describe large systems of interacting particles. In a suitable mean-field limit, the individual particles are replaced by an evolving density, which can obey a Wasserstein gradient flow. This description has also been used for wide neural networks, where many parameters are treated as interacting particles. Finally, in variational inference, a difficult target distribution is approximated within a simpler family of distributions. The KL objective has the form of a free energy, and projecting its Wasserstein gradient flow onto that family yields a practical approximation scheme.

The final application, Generative modeling with flows and diffusion, treats targets represented by data rather than by a known density. Normalizing flows and diffusion models learn deterministic maps or stochastic dynamics that transform samples from a simple base distribution into samples from the target distribution. In this sense, generative modeling is a learned transport problem. Flow matching learns a velocity field that carries the base density toward the data, while diffusion models learn to reverse a gradual noising process. These methods need not find a minimum-cost transport; their goal is to learn a transformation that can be estimated from finite data and used efficiently for sampling. The earlier connections explain why ideas from transport, inference, control, and nonequilibrium physics repeatedly appear in their construction, and why additional guidance can be understood as a control that favors selected outcomes.

A brief note on notation. We have sought to unify notation across the chapters to the extent possible. The notation tables at the end of this review are organized by the chapter in which

a symbol is first introduced.

## I. VARIATIONAL STRUCTURE OF MECHANICS, CONTROL AND INFERENCE

In this chapter, we develop a common variational language for mechanics, control, and inference. We introduce two complementary approaches for solving control problems. We then turn to inferring probability distributions from observations, and show how inference over stochastic paths leads back to a particular class of control problems.

## A. Finding optimal paths in classical mechanics and control theory

We begin with a gentle introduction to control theory. Consider an illustrative example — suppose we would like to guide a rocket towards a target in a certain amount of time by controlling its thrust. A large thrust accelerates the rocket but consumes more fuel, whereas insufficient thrust may not be enough to reach the target on time. How does one choose a thrust protocol so that the rocket reaches the target while consuming the least amount of fuel? Different control strategies generate different paths. In optimal control, we assume that there is a reward function that assigns a value or utility to each path. This reward may favor reaching a target, penalize control costs, or balance competing goals. Control theory offers mathematical tools for finding a control strategy that maximizes the total reward along the path. Although its variational roots are much older, modern optimal control took shape in the 1950s. Bellman’s dynamic programming and the maximum principle of Pontryagin offer two powerful and complementary approaches for solving control problems [1–3]. We now describe these two solution principles, which will reoccur throughout the rest of the review. The principles are best illustrated by demonstrating their relationship with standard formulations of classical mechanics.

The optimal control problem. In optimal control, a system is described by a (possibly multi-dimensional) state $x _ { t }$ whose evolution is modulated by a control $u _ { t }$ . For the rocket, the state includes its position, velocity, and remaining fuel, while the control is the thrust. For convenience, we rescale time to the interval $0 \leq t \leq 1$

In stochastic optimal control, the state obeys <sup>1</sup>

$$
\dot { x } _ { t } = g _ { t } ( x _ { t } , u _ { t } ) + \sqrt { \varepsilon } \eta _ { t } ,\tag{1}
$$

where ε sets the noise scale and $\eta _ { t }$ is standard Gaussian white noise, with $\langle \eta _ { t , i } \eta _ { s , j } \rangle = \delta _ { i j } \delta ( t - s )$ . A reward $r _ { t } ( x _ { t } , u _ { t } )$ assigns a value per unit time to each state-control pair, while a

terminal reward $r _ { 1 } ( x _ { 1 } )$ assigns value to the final state. Starting from a fixed $x _ { 0 } ,$ , the optimized objective is

$$
V _ { 0 } ( x _ { 0 } ) = \operatorname* { m a x } _ { u _ { 0 } \leq s < 1 } \left. r _ { 1 } ( x _ { 1 } ) + \int _ { 0 } ^ { 1 } r _ { s } ( x _ { s } , u _ { s } ) d s \biggm | x _ { 0 } \right. ,\tag{2}
$$

where the expectation is taken over the noise histories $\{ \eta _ { t } \}$ conditioned on the initial state $x _ { 0 }$ [4–6]. The controls may depend on the observed state, but not on future noise. A fuel cost, for example, enters with a minus sign in $r _ { t } ,$ , while a terminal reward favors arrival near the target.

Perhaps intuitively, classical mechanics can be described as a special case of this optimization problem. Consider a particle with coordinates $x _ { t }$ and velocity ${ \dot { x } } _ { t }$ whose motion is described by a Lagrangian $L _ { t } ( x , \dot { x } )$ . Its action is

$$
S [ x ( \cdot ) ] = \int _ { 0 } ^ { 1 } L _ { t } ( x _ { t } , \dot { x } _ { t } ) d t ,\tag{3}
$$

where $x ( \cdot )$ denotes the full path. The principle of stationary action states that the physical path $\{ x _ { t } ^ { * } \}$ makes $S$ stationary under variations that leave the endpoints $x _ { 0 }$ and $x _ { 1 }$ fixed. Imposing stationarity leads to the Euler–Lagrange equations. We now map action minimization onto a control problem by choosing

$$
g _ { t } ( x , u ) = u , \qquad \varepsilon = 0 ,\tag{4}
$$

$$
r _ { t } ( x , u ) = - L _ { t } ( x , u ) ,\tag{5}
$$

and imposing that the terminal endpoint is $x _ { 1 }$ . Here the control plays the role of velocity, $u _ { t } ~ = { \dot { x } } _ { t }$ . The endpoint constraint can equivalently be imposed through a terminal reward that is zero at $x _ { 1 }$ and $- \infty$ elsewhere. Maximizing the reward then minimizes the action. Note that classical mechanics only requires stationarity, that is, physical trajectories need not be minima and can instead be saddles. The optimization therefore selects action-minimizing paths, while the variational equations also describe other stationary paths.

There are two structural approaches for solving the control problem. The first treats the dynamics in Eq. (1) as a constraint and introduces Lagrange multipliers, leading to a Hamiltonian formulation. The second recursively expresses the optimal solution in terms of optimal solutions to shorter remaining problems, leading to the Hamilton–Jacobi–Bellman equation. Relating the two will identify the Lagrange multiplier as the gradient of a value function.

Lagrange multipliers and Hamiltonian dynamics. A Lagrange multiplier converts a constrained optimization problem into a stationary problem in an enlarged set of variables. Variation with respect to the multiplier enforces the constraint, while variation with respect to the original variables supplies the optimality equations. To see this, let’s first take the deterministic limit $\varepsilon = 0$ . Introducing a time-dependent multiplier $p _ { t }$ for the constraint $\dot { \boldsymbol { x } } _ { t } = g _ { t } ( \boldsymbol { x } _ { t } , \boldsymbol { u } _ { t } )$ gives the augmented objective

$$
\begin{array} { l } { \displaystyle \widetilde { V } ( x , u , p ) = r _ { 1 } ( x _ { 1 } ) + \int _ { 0 } ^ { 1 } \left[ r _ { t } ( x _ { t } , u _ { t } ) \right. } \\ { \displaystyle \left. + p _ { t } \cdot \left( g _ { t } ( x _ { t } , u _ { t } ) - \dot { x } _ { t } \right) \right] d t , } \end{array}\tag{6}
$$

with the initial endpoint $x _ { 0 }$ held fixed. The sign of $p _ { t }$ is conventional. Define the control Hamiltonian and its optimized form by

$$
\mathcal { H } _ { t } ( x , p , u ) = p \cdot g _ { t } ( x , u ) + r _ { t } ( x , u ) ,\tag{7}
$$

$$
H _ { t } ( x , p ) = \operatorname* { m a x } _ { u } \mathcal { H } _ { t } ( x , p , u ) .\tag{8}
$$

Variation with respect to $p _ { t }$ and $x _ { t }$ (after integrating by parts the term involving ${ \dot { x } } _ { t }$ in Eq. (6)), together with maximization over $u ,$ gives the necessary optimality conditions [2, 4]. When $H _ { t }$ is differentiable, these take the form

$$
\begin{array} { r } { \dot { x } _ { t } = \nabla _ { p } H _ { t } ( x _ { t } , p _ { t } ) , } \end{array}\tag{9}
$$

$$
\begin{array} { r } { \dot { p } _ { t } = - \nabla _ { x } H _ { t } ( x _ { t } , p _ { t } ) , } \end{array}\tag{10}
$$

$$
\begin{array} { r } { u _ { t } ^ { * } = \arg \operatorname* { m a x } _ { u } \mathcal { H } _ { t } ( x _ { t } , p _ { t } , u ) . } \end{array}\tag{11}
$$

If the terminal state is free, the endpoint variation gives $p _ { 1 } =$ $\nabla r _ { 1 } ( x _ { 1 } )$ , whereas if it is fixed, the terminal variation vanishes. For a free terminal state, the multiplier equation is thus evolved backward from the terminal condition along the trajectory. We will see below that $p _ { t }$ measures how the optimal future reward changes under a displacement of the state.

For the specialization $g _ { t } ( x , u ) \ = \ u$ and $r _ { t } ( x , u ) \ =$ $- L _ { t } ( x , u )$ , the optimized Hamiltonian becomes

$$
H _ { t } ( x , p ) = \operatorname* { m a x } _ { u } \left\{ p \cdot u - L _ { t } ( x , u ) \right\} ,\tag{12}
$$

which is the Legendre transform of the Lagrangian. The stationarity condition with respect to u gives

$$
p _ { t } = \nabla _ { u } L _ { t } \big ( x _ { t } , u _ { t } \big ) = \nabla _ { \dot { x } } L _ { t } \big ( x _ { t } , \dot { x } _ { t } \big ) ,\tag{13}
$$

so the multiplier is the canonical momentum, and the remaining conditions are Hamilton’s equations.

Dynamic programming and the Hamilton–Jacobi equation. The second approach involves keeping track of the maximum reward that remains available from every possible intermediate state (‘reward-to-go’). For the rocket, suppose we already knew the maximum total future reward attainable from every state $x _ { t + d t }$ at time $t + d t$ . The value of choosing a particular thrust at time t is its immediate reward, including the fuel cost, plus the expected best future reward from the state it produces. The optimal thrust is the one that maximizes this sum. In other words, if the expected best future reward is known, solving for the optimal thrust involves optimizing over a single step rather than over a full trajectory.

This method for solving a problem by breaking it into shorter subproblems and reusing their solutions is called $d y .$ namic programming [1]. In this setting, the key observation is that the control at the next state for an optimal strategy must itself be optimal. This gives a recursive relation between the best reward now and the best reward at the next time step, which we can solve backward from the terminal reward. The reuse of partial results is analogous to the transfer-matrix method in statistical mechanics, where a partial sum over configurations is updated one site at a time [7].

The central object in dynamic programming is the value

function

$$
V _ { t } ( x ) = \operatorname* { m a x } _ { u _ { t \leq s < 1 } } \left. \int _ { t } ^ { 1 } r _ { s } ( x _ { s } , u _ { s } ) d s + r _ { 1 } ( x _ { 1 } ) \Bigm \lvert x _ { t } = x \right. ,\tag{14}
$$

with terminal condition $V _ { 1 } ( x ) = r _ { 1 } ( x )$ . Thus, $V _ { t } ( x )$ is the largest expected total reward that can still be collected from time t onward, starting at $x ,$ including the terminal reward. It does not include rewards already received. This is the optimal reward-to-go, and $V _ { 0 } ( x _ { 0 } )$ is the objective in Eq. (2).

The controls after time $t + d t$ have already been optimized inside $V _ { t + d t }$ , so only the control applied during the first short interval remains to be chosen. Splitting the total reward into the immediate and future contributions gives Bellman’s recursion, to first order in $d t$

$$
\begin{array} { l } { { V _ { t } } ( x ) = \displaystyle \operatorname* { m a x } _ { u } \left\{ { r _ { t } } ( x , u ) d t \right. } \\ { \left. + \left. V _ { t + d t } \big ( x + { g _ { t } } ( x , u ) d t + \sqrt { \varepsilon } d W _ { t } \big ) \right. \right\} } \\ { \left. + o ( d t ) . \right. } \end{array}\tag{15}
$$

Here $d W _ { t }$ is a Gaussian noise increment with zero mean and $\langle d W _ { t , i } d W _ { t , j } \rangle = \delta _ { i j } d t$ , and the expectation is over this increment. Assuming the value function is smooth, expanding $V _ { t + d t }$ to first order in time and second order in the state increment, averaging over the noise, and retaining terms of order dt gives the Hamilton–Jacobi–Bellman (HJB) equation [5, 6]

$$
\begin{array} { l } { \displaystyle \partial _ { t } V _ { t } ( x ) + \frac { \varepsilon } { 2 } \nabla ^ { 2 } V _ { t } ( x ) } \\ { \displaystyle \quad + \operatorname* { m a x } _ { u } \left\{ g _ { t } ( x , u ) \cdot \nabla V _ { t } ( x ) + r _ { t } ( x , u ) \right\} = 0 . } \end{array}\tag{16}
$$

The control chosen at state x is the maximizer of the local expression,

$$
u _ { t } ^ { * } ( x ) = \arg \operatorname* { m a x } _ { u } \left\{ g _ { t } ( x , u ) \cdot \nabla V _ { t } ( x ) + r _ { t } ( x , u ) \right\} .\tag{17}
$$

Thus, when $g _ { t }$ and $r _ { t }$ are known, the difficult optimization over an entire control history can be replaced by a backward equation for $V _ { t }$ followed by a local maximization at each state and time. Finding this function over a high-dimensional state space can nevertheless be difficult.

In the deterministic limit, the Hamiltonian $H _ { t } ( x , p )$ introduced above allows us to write the HJB equation as

$$
\begin{array} { r } { \partial _ { t } V _ { t } ( x ) + H _ { t } \big ( x , \nabla V _ { t } ( x ) \big ) = 0 . } \end{array}\tag{18}
$$

This equation also makes the connection between dynamic programming and the Lagrange-multiplier formulation explicit. Where $V _ { t }$ and $H _ { t }$ are smooth, set $p _ { t } = \nabla V _ { t } ( x _ { t } )$ along an optimal trajectory. The state evolves as

$$
\dot { x } _ { t } = g _ { t } ( x _ { t } , u _ { t } ^ { * } ) = \nabla _ { p } H _ { t } ( x _ { t } , p _ { t } ) .\tag{19}
$$

The total time derivative of $p _ { t }$ along this trajectory is

$$
\dot { p } _ { t } = \partial _ { t } \nabla V _ { t } ( x _ { t } ) + ( \dot { x } _ { t } \cdot \nabla ) \nabla V _ { t } ( x _ { t } ) .\tag{20}
$$

Taking the spatial gradient of Eq. (18) gives

$$
\partial _ { t } \nabla V _ { t } + \nabla _ { x } H _ { t } + ( \nabla _ { p } H _ { t } \cdot \nabla ) \nabla V _ { t } = 0 ,\tag{21}
$$

where $\nabla _ { x } H _ { t }$ denotes differentiation with respect to the explicit state argument of $H _ { t } ( x , p )$ while holding p fixed, and the Hamiltonian derivatives are evaluated at $p = \nabla V _ { t } ( x )$ . Using $\dot { x } _ { t } = \nabla _ { p } H _ { t }$ and combining the above two equations, we get

$$
\begin{array} { r } { \dot { p } _ { t } = - \nabla _ { x } H _ { t } ( x _ { t } , p _ { t } ) . } \end{array}\tag{22}
$$

Thus, the trajectories along which the Hamilton–Jacobi equation is solved, its characteristics, obey precisely the Hamilton equations obtained from the Lagrange-multiplier approach. The multiplier $p _ { t }$ is the gradient of the value function because it measures the sensitivity of the optimal future reward to a displacement of the current state.

For action-minimizing trajectories in classical mechanics, the value function is the negative of the minimum action remaining to the fixed endpoint $x _ { 1 }$

$$
V _ { t } ( x ) = - \operatorname* { m i n } _ { \{ x _ { s } \} : x _ { t } = x , \int _ { t } ^ { 1 } } L _ { s } ( x _ { s } , \dot { x } _ { s } ) d s .\tag{23}
$$

On a minimizing trajectory, variation of the initial endpoint gives $\delta V _ { t } = p _ { t } \cdot \delta x$ , while variation with respect to the velocity gives $p _ { t } = \nabla _ { \dot { x } } L _ { t }$ . Hence

$$
p _ { t } = \nabla V _ { t } ( x _ { t } ) = \nabla _ { \dot { x } } L _ { t } ,\tag{24}
$$

so the multiplier of optimal control becomes the canonical momentum of mechanics.

The two approaches thus give complementary descriptions: the Lagrangian viewpoint follows individual trajectories, while the Eulerian viewpoint of dynamic programming finds a field $V _ { t } ( x )$ over all states. Note that we have assumed that the reward $r _ { t }$ and dynamics $g _ { t }$ are known. Reinforcement learning (RL) addresses the challenging problem of finding the optimal control when this information is not available in advance. In RL, one interacts with the environment to learn which controls, or actions, produce large cumulative reward from observed state transitions and rewards. The value function remains central because it provides an estimate of the long-term consequence of acting from each state. We return to the RL framework in Sec. V A of the Applications chapter.

## B. Variational basis for probabilistic inference and control

We now turn to a different but related problem, wherein measurements constrain probabilities of random degrees of freedom, instead of control functions that steer individual paths. Since measurements rarely provide every microscopic detail of a system, we describe the system by a probability distribution that we seek to infer from the available observations. Bayesian inference is an example of such a capability: what is the distribution of $x ,$ given observations $A$ and a prior $p ^ { 0 } ( x )$ (which encodes existing knowledge about the system [8])? By Bayes’ rule, the conditional distribution is $\dot { p ( x | A ) } = p ( A | \dot { x } ) p ^ { 0 } ( x ) / p ^ { 0 } ( A )$ , thereby recasting the problem as specifying a model for A given x. More generally, the central task of probabilistic inference is to determine conditional probabilities. In this chapter, we show that this task can be formulated as constrained optimization — a variational problem under constraints arising from observations. This formulation highlights a conceptual connection to the optimization problems of mechanics and control (Sec. I A), and provides the theoretical basis for path-integral stochastic control in Sec. I D. Later in the review, variational probabilistic inference will serve as a foundation for practical inference algorithms: Chapter IV uses control theory to address inference problems, while Sec. V B 3 considers variational inference, where the system distribution is approximated using a parametrized ansatz.

Here, we will introduce maximum-entropy inference, a method for finding a probability distribution for the system without prior knowledge, constrained to be consistent with observations. The observations take the form of measurements of moments like the mean or variance, which, however, do not fully determine the distribution. There is thus a manyto-one correspondence between microscopic models and the measurements. How to choose one model out of the many? One well-motivated approach is to construct a model of behavior $p ^ { A } ( x )$ that is closest to a reference distribution $p ^ { 0 } ( x )$ but remains consistent with the observations A. As we will show below, a natural candidate for measuring closeness is the Kullback-Leibler divergence,

$$
\mathrm { D } _ { \mathrm { K L } } ( p \Vert q ) = \int \mathrm { d } x p ( x ) \log \frac { p ( x ) } { q ( x ) } .\tag{25}
$$

measuring how different (“the excess surprise”) the distribution $p$ is from q [9]. We will furthermore see that using Lagrange multipliers that enforce consistency with observed paths has a natural interpretation as rewards. This choice will establish a conceptually and practically relevant connection between probabilistic inference and control.

## 1. Sanov’s theorem and the inference method

We first make a plausibility argument for the inference method through Sanov’s theorem. It equips $\mathrm { D } _ { \mathrm { K I } }$ with a probabilistic interpretation as a measure of how a distribution estimated from samples deviates from the underlying true distribution. Namely, suppose that N independent and identically distributed (IID) samples $\{ x _ { n } \} _ { 1 , \ldots , N }$ are drawn from a base measure, $p ^ { 0 } ( x )$ . Having drawn these samples, to what extent does the empirical measure,

$$
\hat { p } ( x ) = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \delta ( x - x _ { n } ) ,\tag{26}
$$

depart from $p ^ { 0 } ( x ) \ i$ Sanov’s theorem [10, 11] states that the probability of the empirical measure, $\mathbb { P } [ \hat { p } ]$ , deviating from the ground truth $p ^ { 0 } ( x )$ is exponentially suppressed with an increasing number of samples N:

$$
\mathbb { P } [ \boldsymbol { \hat { p } } ] \sim e ^ { - N \mathrm { D } _ { \mathrm { K L } } ( \boldsymbol { \hat { p } } \| p ^ { 0 } ) } ,\tag{27}
$$

Eq. (27) is achieved using large-deviations theory, which concerns finding the asymptotic decline in probability of a rare event, wherein the probability typically takes the form $\sim e ^ { - N I }$ with I called the rate function. The rate function here is the $\mathrm { D } _ { \mathrm { K I } }$ , which possesses two appealing properties: $\mathrm { D } _ { \mathrm { K L } } = 0$ and minimal if and only if ${ \hat { p } } = { \boldsymbol { p } } ^ { 0 } .$ , and the spread itself gets smaller as $N \to \infty ; \hat { p } { - } \dot { p } ^ { 0 } \stackrel { \cdot } { \sim } { \hat { N ^ { - 1 / 2 } } }$ . The rarer the event, the larger the KL divergence, and the more the probability is suppressed.

We will prove Eq. (27) via a path-integral calculation, exemplifying the path-integral machinery we will encounter later in the review. We compute the distribution of empirical measures P[ˆp] by definition, from the expectation of the functional delta function over the drawn samples,

$$
\begin{array} { l } { \displaystyle \mathbb { P } [ \hat { p } ] = \prod _ { m = 1 } ^ { N } \left[ \int \mathrm { d } x _ { m } p ^ { 0 } ( x _ { m } ) \right] } \\ { \displaystyle \qquad \times \delta \left[ \hat { p } ( x ) - \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \delta ( x - x _ { n } ) \right] . } \end{array}\tag{28}
$$

Expressing the delta function via its Fourier transform, $\begin{array} { r } { \delta [ \dot { f } ( x ) ] \propto \int \mathrm { D } [ \mathrm { i } \mu ( x ) ] e ^ { \int \mathrm { d } x \mu ( x ) f ( x ) } } \end{array}$ , and noting the identical and independent integrals over samples, we rewrite

$$
\begin{array} { l } { { \displaystyle \mathbb { P } [ \hat { p } ] \propto \int \mathrm { D } [ \mathrm { i } \mu ( { \boldsymbol { x } } ) ] \exp \left\{ N \int \mathrm { d } { \boldsymbol { x } } \frac { \mu ( { \boldsymbol { x } } ) } { N } \hat { p } ( { \boldsymbol { x } } ) \right. } \ ~ ( 2 9 ) }  \\ { { \displaystyle \left. + N \log \left[ \int \mathrm { d } { \boldsymbol { x } } ^ { \prime } p ^ { 0 } ( { \boldsymbol { x } } ^ { \prime } ) e ^ { - \mu ( { \boldsymbol { x } } ^ { \prime } ) / N } \right] \right\} } } \end{array}
$$

Since $N  \infty$ , we note that only $\mu ( x ) \sim N$ contributes to the Fourier integral, overall implying that the argument of the exponential is $\sim N$ . This justifies a saddle-point evaluation of the integral — the saddle point $\mu ^ { * } ( x )$ satisfies

$$
\begin{array} { r } { \frac { p ^ { 0 } ( x ) e ^ { - \mu ^ { * } ( x ) / N } } { \int \mathrm { d } x ^ { \prime } p ^ { 0 } ( x ^ { \prime } ) e ^ { - \mu ^ { * } ( x ^ { \prime } ) / N } } = \hat { p } ( x ) , } \end{array}\tag{30}
$$

Inserting Eq. (30) in Eq. (29), we obtain $\operatorname { E q . }$ (27).

Sanov’s theorem ensures that the deviations of the estimated measure from the ground truth are minimized with more samples. However, during probabilistic inference, the choice of a reference distribution is often that which can be sampled easily, meaning that there is no inherent reason why the underlying model is that of the base $p ^ { 0 }$ . This suggests that if we observe data inconsistent with $p ^ { 0 }$ and seek the most likely alternative explanation, we should find the distribution $p ^ { A }$ that is closest to $\bar { p } ^ { 0 }$ in KL divergence while remaining consistent with the observations A. By observations we mean, $e . g .$ , measurable moments or state occupancies of the system, dictated by the unknown underlying distribution. For instance, later we will be interested in inferring and controlling the distributions over paths. Via a large-deviation principle, we may motivate the fact that the distribution $p ^ { A }$ closest to $p ^ { 0 }$ , consistent with observations $A ,$ satisfies

$$
\begin{array} { c } { { p ^ { A } = \displaystyle \arg \operatorname* { m i n } _ { p } \mathrm { D } _ { \mathrm { K L } } ( p \| p ^ { 0 } ) , } } \\ { { \mathrm { s . t . } \quad \displaystyle \int \mathrm { d } x p ( x ) = 1 \quad \mathrm { a n d } \quad \displaystyle \int \mathrm { d } x p ( x ) a ( x ) = A . } } \end{array}\tag{31}
$$

This will serve as the basis for probabilistic inference throughout this review.

To motivate this inference, we recall that not all ${ \hat { p } } ( x ) \mathbf { s }$ are possible if the value of a certain observable, A, is known and thus imposed as a constraint. The space of empirical distributions is thus reduced to those which satisfy $A \ =$ $\textstyle N ^ { - 1 } \sum _ { n = 1 } ^ { N } a ( x _ { n } )$ . Therefore, Eq. (28) should be rewritten instead as

$$
\begin{array} { l } { \displaystyle \mathbb { P } [ \hat { p } | A ] = \prod _ { m = 1 } ^ { N } \left[ \int \mathrm { d } x _ { m } p ^ { 0 } ( x _ { m } ) \right] } \\ { \displaystyle \qquad \times \delta \left[ \hat { p } ( x ) - \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \delta ( x - x _ { n } ) \right] } \\ { \displaystyle \qquad \times \delta \left[ A - \frac { 1 } { N } \sum _ { n = 1 } ^ { N } a ( x _ { n } ) \right] . } \end{array}\tag{32}
$$

Even more so, note that Eq. (30) implies additionally that ${ \hat { p } } ( x )$ must be normalized $\textstyle ( \int \mathrm { d } x { \hat { p } } ( x ) = 1 )$ . The rest of the procedure to obtain Eq. (27) remains unchanged other than the additionally imposed observables. Following the above saddlepoint calculation, Eq. (27) can be rewritten as

$$
\begin{array} { l } { \displaystyle \mathbb { P } [ \hat { p } | A ] \sim \exp \Biggl \{ - N \mathrm { D } _ { \mathrm { K L } } ( \hat { p } | | p ^ { 0 } ) } \\ { \displaystyle \qquad + \eta ^ { * } \left[ \int \mathrm { d } x \hat { p } ( x ) - 1 \right] } \\ { \displaystyle \qquad + \lambda ^ { * } \left[ \int \mathrm { d } x \hat { p } ( x ) a ( x ) - A \right] \Biggr \} , } \end{array}\tag{33}
$$

where $\eta ^ { * }$ and $\lambda ^ { * }$ are such that the overall rate function (the argument of the exponential) is minimized. (This calculation should be regarded as qualitative, as the saddle point in η and λ may often not be justified.) Therefore, we conclude that Sanov’s theorem motivates turning the inference problem of $p ^ { A } ( x )$ into a constrained optimization problem, Eq. (31).

## 2. Maximum-entropy inference

Sanov’s theorem is closely related to maximum-entropy inference. Suppose, for simplicity, that the prior is the uniform distribution, $\dot { \ p } ^ { 0 } ( x ) \ = \ 1 / \ \dot { \int } \mathrm { d } x ^ { \dot { \prime } }$ . In this case, the KL divergence simplifies to the negative of the Shannon entropy of a distribution p(x) [9],

$$
H [ p ] = - \int \mathrm { d } \boldsymbol { x } p ( \boldsymbol { x } ) \log p ( \boldsymbol { x } ) .\tag{34}
$$

Shannon argued that Eq. (34) is a functional that can quantify information [12]. When the only available information is a set of measured expectation values, the “least-committal” distribution consistent with them is the one that maximizes entropy [13, 14].

Mathematically put, suppose a set of observables was measured, $\begin{array} { r } { A = \int \mathrm { d } x p ( x ) a ( x ) } \end{array}$ . Combined with normalization, according to Eq. (31) (with entropy replacing the KL divergence) the inferred distribution is that which satisfies

$$
\begin{array} { l } { { \displaystyle p ^ { A } = \arg \operatorname* { m a x } _ { p } H [ p ] , \quad \mathrm { s . t . } \quad \int \mathrm { d } x p ( x ) = 1 \quad \mathrm { a n d } } } \\ { { \displaystyle \left\{ \int \mathrm { d } x p ( x ) a _ { m } ( x ) = A _ { m } \right\} _ { m = 1 , \dots , M } . } } \end{array}\tag{35}
$$

Using the method of Lagrange multipliers, one finds the solution

$$
p ^ { A } ( x ) = \frac { 1 } { Z } \exp \left[ - \sum _ { m = 1 } ^ { M } \lambda _ { m } a _ { m } ( x ) \right] ,\tag{36}
$$

where

$$
Z = \int \mathrm { d } x \exp \left[ - \sum _ { m = 1 } ^ { M } \lambda _ { m } a _ { m } ( x ) \right]\tag{37}
$$

arose from the Lagrange multiplier enforcing normalization. $\lambda _ { m }$ are the remaining Lagrange multipliers, determined from the implicit relation

$$
A _ { m } = - \frac { \partial \log Z } { \partial \lambda _ { m } } .\tag{38}
$$

The maximized entropy reads

$$
H [ p ^ { A } ] = \log Z - \sum _ { m = 1 } ^ { M } \lambda _ { m } A _ { m } .\tag{39}
$$

The above is the general statement for an inference scheme based on the maximum-entropy principle. It can be readily generalized to the case where the prior $p ^ { 0 } ( x )$ is not uniform, $e . g .$ , the inferred distribution will become the “exponential $\mathrm { t i l t } ^ { \prime }$ of the base distribution

$$
p ^ { A } ( x ) \propto p ^ { 0 } ( x ) \exp \left[ - \sum _ { m = 1 } ^ { M } \lambda _ { m } a _ { m } ( x ) \right] .\tag{40}
$$

In the above expressions, $a _ { m } ( x )$ can be polynomials, in which case A gives the moments of the underlying distribution. However, a particularly relevant observable for later is that which checks if x is in a region A, $a ( x ) : = \ \mathbf { 1 } _ { \mathcal { A } } ( x )$ , so $\begin{array} { r } { A = \int a ( x ) p ( x ) d x \ = \ 1 } \end{array}$ The constraint is then $p ^ { A } ( x \in$ $\mathcal { A } ) \mathrel { \mathop : } = \check { p } ^ { A } ( \mathcal { A } ) = 1$ . The maximum-entropy solution Eq. (35) is exactly the conditional probability distribution. At the same time, Bayes’ rule means $p ( x | \mathcal { A } ) \overset { \cdot } { = } \mathbf { 1 } _ { \mathcal { A } } ( x ) p ^ { 0 } ( x ) / p ^ { 0 } ( \mathcal { A } )$ Thus, the constraint $p ^ { A } ( \mathcal { A } ) = \hat { 1 }$ implies:

$$
\begin{array} { r l } & { \mathrm { D } _ { \mathrm { K L } } ( p ^ { A } \| p ^ { 0 } ) \big | _ { p ^ { A } ( \mathcal { A } ) = 1 } = \mathrm { D } _ { \mathrm { K L } } ( \mathbf { 1 } _ { \mathcal { A } } \cdot p ^ { A } \| \mathbf { 1 } _ { \mathcal { A } } \cdot p ^ { 0 } ) } \\ & { = \mathrm { D } _ { \mathrm { K L } } ( p ^ { A } \| p ( \cdot | \mathcal { A } ) ) - \log p ^ { 0 } ( \mathcal { A } ) } \end{array}\tag{41}
$$

whose minimum is $\begin{array} { r c l } { p ^ { A } ( x ) } & { = } & { p ( x | \mathcal { A } ) } \end{array}$ . In this setting, maximum-entropy inference exactly reproduces Bayesian inference.

The minimum-KL-divergence formulation is closely related to Bayesian inference more generally. The “update” ratio $p ^ { A } ( x ) / \dot { p } ^ { 0 } ( x ) = e ^ { \cdots }$ can be interpreted as the likelihood ratio, $p ( A | x )$ by Bayes’ rule — the exponential factor reweighing the prior according to the information encoded by the imposed constraints. In this sense, the minimum-KL solution is thus a Bayesian-style update of the reference distribution, with the exponential tilt playing the role of an (unnormalized) likelihood. These general functional forms will appear repeatedly in the upcoming sections for specialized observables. We will furthermore see that the Lagrange multipliers will take the form of rewards, thereby generalizing ideas from Sec. I A to path-integral stochastic control. Furthermore, the D<sub>KL</sub> objective arises naturally in the context of Variational Inference (VI), the problem of approximating a complex distribution by a simpler — typically parametric — one, which will be discussed in Sec. V B 3.

Another noteworthy connection of maximum-entropy models is to exponential-family models [15, 16]. Maximumentropy distributions subject to linear expectation constraints take the exponential-family form of Eq. (36), in which the Lagrange multipliers $\lambda _ { m }$ enter linearly in the exponent and determine the partition function Z. Beyond their conceptual connection to maximum entropy, exponential-family models are useful because their parameters can be estimated through convex optimization, making them particularly attractive for statistical inference and applications such as image processing.

In addition to Sanov’s theorem and the Bayesian-inference connections, there is a subtle connection of maximum-entropy inference to equilibrium statistical mechanics. Sanov’s theorem provides concrete grounds for the maximum-entropy approach, as large deviations are exponentially suppressed as the number of samples increases. In the thermodynamic limit the dimensionality of x (the number of particles), reaching as much as ∼ $1 0 ^ { 2 3 }$ , far exceeds any feasible number of samples that one may obtain. Instead, the second law of thermodynamics and statistical mechanics provide the theoretical basis for finding the microscopic statistical model of materials. Nevertheless, both mechanisms — equilibrium statistical mechanics and maximum-entropy modeling — provide the same mathematical structures [11]. We shall discuss this correspondence in Sec. III A.

## C. Probabilistic inference over paths

In Sec. I A, we considered how to control a system with known dynamics to maximize a prescribed reward. In Sec. I B, we saw how inference problems can be formulated as optimization problems in which we seek a distribution that is consistent with observed data while remaining as close as possible to a prior distribution. We now turn to the application of the large-deviation principle to the inference of distributions over paths. We will exploit the tools of stochastic optimal control to solve these path-distribution inference problems.

In a stochastic system, repeated realizations of the noise generate different trajectories, so in the context of stochastic optimal control, a given control determines a probability distribution over paths. When investigating the properties of a distribution over paths, we will exploit that, in generic settings, there is a correspondence between stochastic differential equations (SDEs) and distributions over paths. In particular, an SDE determines a distribution over paths in which each realization of the noise generates a different trajectory. Under certain regularity conditions, there is a converse relationship where a distribution over paths can be represented by an SDE whose drift and diffusion coefficients are determined by the properties of the distribution; we will restrict the inference problems we consider to those for which this correspondence holds. We use uppercase P for distributions over complete paths, reserving lowercase p for state densities. If we have a prior distribution over paths, $P ^ { 0 } [ x ( \cdot ) ]$ , and we observe some data about the system, the large-deviation principle tells us how to find the distribution over paths, $P ^ { u } [ x ( \cdot ) ]$ that is most similar to the prior while remaining consistent with the observations. Thus, because of the path distribution-SDE correspondence, the result of inference is a new SDE that describes the dynamics of the system under the inferred distribution over paths. Importantly, the inferred SDE will contain additional forces that will be determined by an optimization problem. To see this, consider a simple example of such a path-conditioning problem, namely, the Brownian bridge. In the Brownian bridge, we consider a particle known to start at a point $x _ { 0 }$ at time $t = 0$ and undergo Brownian motion, but we impose the condition that the particle is at a prescribed target y at time $t = 1$

$$
\begin{array} { r } { \dot { x } _ { t } = \sqrt { \varepsilon } \eta _ { t } , \qquad } \\ { \mathrm { c o n d i t i o n e d ~ o n } \quad x _ { t = 0 } = x _ { 0 } , \quad x _ { t = 1 } = y . } \end{array}\tag{42}
$$

Without this conditioning, the particle would hit y at time $t = 1$ formally with probability zero. Within the ensemble of such conditioned paths, there is apparently an additional force that drives the particle to y at time $t = 1$ , which is not present in the unconditioned dynamics. Here, we will consider similar problems in which we condition on more general observations, and we will see that the large-deviation principle provides a natural way to infer the additional forces that emerge from this conditioning.

## 1. From inference to distributions over paths

Consider the prior distribution over paths, $P ^ { 0 } [ x ( \cdot ) ]$ , induced by the passive dynamics (i.e., the dynamics without any additional forces),

$$
\begin{array} { r } { \dot { x } _ { t } = f _ { t } ( x _ { t } ) + \sqrt { \varepsilon } \eta _ { t } . } \end{array}\tag{43}
$$

We seek to infer how this ensemble changes when observations constrain its properties. We will represent the inferred ensemble through an additional control,

$$
\begin{array} { r } { \dot { x } _ { t } = f _ { t } ( x _ { t } ) + u _ { t } ( x _ { t } ) + \sqrt { \varepsilon } \eta _ { t } , } \end{array}\tag{44}
$$

and denote its path distribution by $P ^ { u } [ x ( \cdot ) ]$ . The reference and controlled processes start at the same fixed $x _ { 0 } .$ , as in Sec. I A. The marginal density at a single time is

$$
p _ { t } ^ { u } ( x ) = \langle \delta ( x - x _ { t } ) \rangle _ { P ^ { u } } .
$$

We omit the superscript on a marginal when only one ensemble is under discussion. The large-deviation argument applies here with each sample being an entire trajectory. Among empirical ensembles compatible with the observations, it identifies those most likely to arise from repeated sampling of the passive dynamics. In the limit of many trajectories, the empirical path distribution concentrates on the corresponding inferred distribution.

Suppose the observations specify averages $A _ { m }$ of trajectory observables $a _ { m } [ x ( \cdot ) ]$ ], so that

$$
\int { \mathcal { D } } x P ^ { u } [ x ( \cdot ) ] a _ { m } [ x ( \cdot ) ] = A _ { m } .\tag{45}
$$

Eq. (31) then states that the most likely path measure is found by minimizing over $P ^ { u }$ the KL divergence from the prior $\dot { P ^ { 0 } } [ 1 7 ]$

$$
\mathrm { D } _ { \mathrm { K L } } \bigl ( P ^ { u } \| P ^ { 0 } \bigr ) = \int \mathcal { D } x P ^ { u } [ x ( \cdot ) ] \log \frac { P ^ { u } [ x ( \cdot ) ] } { P ^ { 0 } [ x ( \cdot ) ] } .\tag{46}
$$

while satisfying these constraints. These constraints can be enforced with Lagrange multipliers $r _ { m }$ . We use the reward sign convention $r _ { m } = - \lambda _ { m } .$ , where $\lambda _ { m }$ is the multiplier in the convention used in Sec. I B 2, so a positive reward appears with a positive sign in the exponential tilt. This gives the constrained objective:

$$
\begin{array} { l } { \displaystyle \mathcal { L } = \operatorname { D } _ { \mathrm { K L } } \bigl ( P ^ { u } \| P ^ { 0 } \bigr ) } \\ { \displaystyle \qquad - \sum _ { m } r _ { m } \biggl [ \int \mathcal { D } \boldsymbol { x } P ^ { u } [ \boldsymbol { x } ( \cdot ) ] a _ { m } [ \boldsymbol { x } ( \cdot ) ] - A _ { m } \biggr ] . } \end{array}\tag{47}
$$

Taking the variation with respect to the path distribution and setting it to zero gives the minimizing distribution. When represented by the optimal control $u ^ { * }$ , it reads

$$
{ P ^ { u } } ^ { * } [ x ( \cdot ) ] \propto P ^ { 0 } [ x ( \cdot ) ] \exp { \left( \sum _ { m } r _ { m } a _ { m } [ x ( \cdot ) ] \right) } ,\tag{48}
$$

meaning the inferred path measure is an exponential tilt of $P ^ { 0 }$ where the Lagrange multipliers $r _ { m }$ enforce consistency with the observations. These Lagrange multipliers can be roughly interpreted as inferred rewards that the system, conditioned on the observations, is receiving. Because of the correspondence between path distributions and SDEs, the ‘tilts’ $r _ { m }$ to the prior path distribution translate into apparent forces in the SDE describing paths drawn from $P ^ { u ^ { * } }$ . Our goal now is to solve the constrained optimization problem introduced to get an expression for $\boldsymbol { u } _ { t } ^ { * }$ which is the apparent force that emerges when the system is conditioned on the observations.

## 2. Girsanov’s theorem

To do this, we exploit Girsanov’s theorem, which allows us to express the change of measure from $P ^ { 0 }$ to $P ^ { u }$ in terms of the control $u _ { t }$ [18]. Girsanov’s theorem is a fundamental result in stochastic calculus because it provides a way to relate probability measures of stochastic processes with different drift terms. In the context of inference over paths, Girsanov’s theorem allows us to express the KL divergence between the prior and inferred path measures in terms of the added control $u _ { t }$ . This will allow us to formulate the inference problem as an optimization problem over the control, which can then be solved using techniques from stochastic optimal control.

Girsanov’s theorem states that the ratio of the path measures of two SDEs which are driven by (potentially inhomogeneous) Gaussian noise and have differing drift terms can be expressed in terms of the difference in the drift terms. To see this, note that the controlled dynamics can be made to look like the passive dynamics by relabeling the noise:

$$
\begin{array} { l } { \displaystyle \dot { x } _ { t } = f _ { t } ( x _ { t } ) + u _ { t } ( x _ { t } ) + \sqrt { \varepsilon } \eta _ { t } ^ { u } } \\ { \displaystyle = f _ { t } ( x _ { t } ) + \sqrt { \varepsilon } \bigg [ \eta _ { t } ^ { u } + \frac { 1 } { \sqrt { \varepsilon } } u _ { t } ( x _ { t } ) \bigg ] . } \end{array}\tag{49}
$$

Here $\eta _ { t } ^ { u }$ denotes the zero-mean noise under the controlled path measure. Thus, the same realized trajectory that requires the noise $\eta _ { t } ^ { u }$ under the controlled dynamics would require the shifted noise $\begin{array} { r } { \eta _ { t } ^ { 0 } = \eta _ { t } ^ { u } + \frac { 1 } { \sqrt { \varepsilon } } u _ { t } ( x _ { t } ) } \end{array}$ under the passive dynamics. The ratio between the two measures can then be written as: <sup>2</sup>

$$
\begin{array} { l } { \displaystyle \frac { P ^ { u } [ { \boldsymbol { x } } ( \cdot ) ] } { P ^ { 0 } [ { \boldsymbol { x } } ( \cdot ) ] } = \frac { \exp \left[ - \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \mathrm { d } t \ \left. \eta _ { t } ^ { u } \right. ^ { 2 } \right] } { \exp \left[ - \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \mathrm { d } t \ \left. \eta _ { t } ^ { 0 } \right. ^ { 2 } \right] } } \\ { \displaystyle = \exp \left[ \frac { 1 } { 2 \varepsilon } \int _ { 0 } ^ { 1 } \mathrm { d } t \ \left. u _ { t } ( { \boldsymbol { x } } _ { t } ) \right. ^ { 2 } + \frac { 1 } { \sqrt { \varepsilon } } \int _ { 0 } ^ { 1 } \mathrm { d } t \ u _ { t } ( { \boldsymbol { x } } _ { t } ) \cdot \eta _ { t } ^ { u } \right] . } \end{array}
$$

The KL divergence is the expectation of the logarithm of this ratio under the controlled path measure:

$$
\begin{array} { l } { { \displaystyle \mathrm { D } _ { \mathrm { K L } } \big ( P ^ { u } \| P ^ { 0 } \big ) = \frac { 1 } { 2 \varepsilon } \left. \int _ { 0 } ^ { 1 } \mathrm { d } t \ \| u _ { t } ( x _ { t } ) \| ^ { 2 } \right. _ { P ^ { u } } } } \\ { { \displaystyle ~ + \frac { 1 } { \sqrt { \varepsilon } } \left. \int _ { 0 } ^ { 1 } \mathrm { d } t \ u _ { t } ( x _ { t } ) \cdot \eta _ { t } ^ { u } \right. _ { P ^ { u } } } } \\ { { \displaystyle ~ = \frac { 1 } { 2 \varepsilon } \left. \int _ { 0 } ^ { 1 } \mathrm { d } t \ \| u _ { t } ( x _ { t } ) \| ^ { 2 } \right. _ { P ^ { u } } . } } \end{array}\tag{51}
$$

In the final equality, the term linear in the controlled noise has vanished because under the controlled path measure, the noise $\eta _ { t } ^ { u }$ has mean zero. Eq. (50) is the standard statement of Girsanov’s theorem [5].

## 3. Probabilistic inference as a control problem

With Girsanov’s theorem, this means that we can write Eq. (31) more formally as an optimization problem:

$$
\begin{array} { c } { \displaystyle { u ^ { * } = \arg \operatorname* { m a x } _ { u } \left. - \frac { 1 } { 2 \varepsilon } \int _ { 0 } ^ { 1 } \mathrm { d } t \left\| u _ { t } ( \boldsymbol { x } _ { t } ) \right\| ^ { 2 } \right. } } \\ { \displaystyle { \left. + \int _ { 0 } ^ { 1 } \mathrm { d } t r _ { t } ( \boldsymbol { x } _ { t } ) + r _ { 1 } ( \boldsymbol { x } _ { 1 } ) \right. _ { P ^ { u } } } } \end{array}\tag{52}
$$

where $r _ { t }$ and $r _ { 1 }$ are again Lagrange multipliers chosen to enforce the constraint of consistency with observations. Here the trajectory reward is additive in time, with a separate terminal term. In this special case, the running reward of Sec. I A is

$$
\underbrace { r _ { t } ( x , u ) } _ { \mathrm { t o t a l ~ r u n n i n g ~ r e w a r d } } = \underbrace { r _ { t } ( x ) } _ { \mathrm { s t a t e ~ r e w a r d } } - \frac { \Vert u \Vert ^ { 2 } } { 2 \varepsilon } .
$$

This now has the form of a control problem in which we are solving to find the optimal control $\boldsymbol { u } _ { t } ^ { * } ( \boldsymbol { x } _ { t } )$ which maximizes the reward but has a cost from applying too strong a control [19–21]. We will see that the problem in Eq. (52) can be solved by converting it into a linear PDE for which Monte Carlo methods can be used to sample the optimal control.

In this context, the quadratic control cost has a probabilistic interpretation in that it measures how strongly the controlled ensemble departs from the passive dynamics. The same relation between noise and control cost that makes this inference interpretation possible also gives the associated HJB equation a particularly simple mathematical structure.

## D. Mathematical structure of path integral stochastic control

Problems with objectives such as Eq. (52) are called path integral stochastic control (PISC) problems [22–26]. PISC problems are a special class of stochastic control problems in which the cost of control is quadratic in the control and the dynamics are linear in the control $( \mathrm { i . e . }$ , in Eq. (1), $g _ { t } ( x , u ) =$ $f _ { t } ( x ) + u )$ . Fig. 2 illustrates a solution to a PISC problem in which an ensemble of particles is controlled to achieve a distant terminal reward. These problems can be solved using a path integral perspective like in Eq. (48), but are most naturally solved using dynamic programming and the Hamilton– Jacobi–Bellman (HJB) equation as introduced in Sec. I A. PISC is a special class of stochastic control problems because the HJB equation takes a particularly simple form that can be transformed into a linear PDE.

![](images/5d8fad5a5e21befc9a645a016a01018a42d9cfe9877c911bcb6c70ccca1f06a4.jpg)

![](images/9f566bc86040cda93220620b3f90c6351b421461738c082b274231a5188a7630.jpg)

![](images/0898c9bb94bacc0335239601be6e7f2c940bda7fa72f7ca666fe2210606f9b4c.jpg)

![](images/fb41cc622fc60e55cddaec870b621f91b0a3c71bc4f6dd14fbc1c76fc5b6a96b.jpg)  
FIG. 2. Path integral stochastic control (PISC) for an ensemble of freely diffusing particles. Left: an ensemble of particles begins at a common initial condition and freely diffuses, failing to achieve some distant terminal reward. Middle: the value function which solves the PISC problem of the ensemble of particles achieving the maximum terminal reward subject to some control cost. Right: the resulting PISC policy leading a significant fraction of the particles to efficiently achieve the terminal reward.

## 1. The Cole–Hopf transformation of the HJB equation

To solve the inference problem, expressed as Eq. (52), we first define the value function to be the expected reward received if the optimal policy is taken starting at point x at time t:

$$
\begin{array} { r l r } & { } & { V _ { t } ( x ) = \underset { t \leq s \leq 1 } { \operatorname* { m a x } } \left. - \frac { 1 } { 2 \varepsilon } \int _ { t } ^ { 1 } \mathrm { d } s \left\| u _ { s } ( x _ { s } ) \right\| ^ { 2 } \right. } \\ & { } & { \left. + \int _ { t } ^ { 1 } r _ { s } ( x _ { s } ) \mathrm { d } s + r _ { 1 } ( x _ { 1 } ) \bigg | x _ { t } = x \right. _ { P ^ { u } } } \end{array}\tag{53}
$$

which using dynamic programming (Eq. (16)) must satisfy

$$
\begin{array} { c } { 0 = \displaystyle \operatorname* { m a x } _ { u } \left\{ { r _ { t } } ( x ) + \partial _ { t } V _ { t } + \nabla V _ { t } \cdot \left[ f _ { t } ( x ) + u \right] \right. } \\ { \displaystyle \left. + \frac { \varepsilon } { 2 } \nabla ^ { 2 } V _ { t } - \frac { 1 } { 2 \varepsilon } \| u \| ^ { 2 } \right\} } \end{array}\tag{54}
$$

with the optimal control given by

$$
\boldsymbol { u } _ { t } ^ { * } ( \boldsymbol { x } ) = \varepsilon \nabla V _ { t } ( \boldsymbol { x } ) .\tag{55}
$$

For PISC problems, the HJB equation is then the nonlinear PDE [27]

$$
\begin{array} { l } { { r _ { t } ( x ) + \partial _ { t } V _ { t } + \nabla V _ { t } \cdot f _ { t } ( x ) } } \\ { { \displaystyle ~ + \frac { \varepsilon } { 2 } \nabla ^ { 2 } V _ { t } + \frac { \varepsilon } { 2 } \| \nabla V _ { t } \| ^ { 2 } = 0 } } \end{array}\tag{56}
$$

with the boundary condition $V _ { 1 } ( x ) = r _ { 1 } ( x )$ . Remarkably, this nonlinear PDE can be converted into a linear equation by the Cole–Hopf transformation [22, 23],

$$
V _ { t } ( x ) = \log \psi _ { t } ( x ) ,\tag{57}
$$

which upon substitution gives:

$$
\Big [ r _ { t } ( x ) + \partial _ { t } + f _ { t } ( x ) \cdot \nabla + \frac { \varepsilon } { 2 } \nabla ^ { 2 } \Big ] \psi _ { t } ( x ) = 0\tag{58}
$$

which is a backward Kolmogorov equation with a potential term. Notably, this is also the Schrodinger equation in imag-¨ inary time (i.e., with t → it) when $f _ { t } = 0$ with $\psi$ being the wavefunction, ε playing the role of $\hbar / m$ , and $- r _ { t } ( x )$ playing the role of a potential energy term. In the small-ε limit, this parallels the passage from wave optics to geometrical optics, where the Hamilton–Jacobi equation becomes the eikonal equation. With this change of variables, the optimal control can be expressed as:

$$
\boldsymbol { u } _ { t } ^ { * } ( \boldsymbol { x } ) = \varepsilon \nabla \log \psi _ { t } ( \boldsymbol { x } ) .\tag{59}
$$

## 2. Feynman–Kacformula and Doob’s h-transform

The transformed equation is linear, but solving it over the full state space can still be difficult in high dimensions. An alternative is to express its solution as an average over trajectories of the passive dynamics, weighted by their accumulated rewards. This is the stochastic counterpart of the imaginarytime path-integral representation from quantum mechanics and provides a starting point for the sampling methods discussed in Chapter IV. From the same reasoning as in Eq. (48), the optimal path measure conditional on starting at x at time t is given below. In these conditional path laws, $x ( \cdot )$ denotes the remaining trajectory on $t \leq s \leq 1 \colon$

$$
P ^ { u ^ { * } } [ x ( \cdot ) | x _ { t } = x ] = { \frac { 1 } { Z _ { t } ( x ) } } P ^ { 0 } [ x ( \cdot ) | x _ { t } = x ] e ^ { R _ { t } [ x ( \cdot ) ] }\tag{60}
$$

where $\begin{array} { r } { R _ { t } [ x ( \cdot ) ] = \int _ { t } ^ { 1 } r _ { s } ( x _ { s } ) \mathrm { d } s + r _ { 1 } ( x _ { 1 } ) } \end{array}$ is the cumulative state and terminal reward from time t to the final time 1, excluding control cost. We temporarily denote the normalization factor by $Z _ { t } ( x )$ , which is a conditional partition function. Observe from Eq. (53) and Girsanov’s theorem that we can write the value function as

$$
\begin{array} { r l } & { V _ { t } ( x ) = \bigg \langle R _ { t } [ x ( \cdot ) ] - \displaystyle \frac { 1 } { 2 \varepsilon } \int _ { t } ^ { 1 } \mathrm { d } s \left\| u _ { s } ^ { * } ( x _ { s } ) \right\| ^ { 2 } \bigg | \ x _ { t } = x \bigg \rangle _ { P ^ { u ^ { * } } } } \\ & { \quad \quad \quad = \bigg \langle R _ { t } [ x ( \cdot ) ] - \log \frac { P ^ { u ^ { * } } [ x ( \cdot ) | x _ { t } = x ] } { P ^ { 0 } [ x ( \cdot ) | x _ { t } = x ] } \biggm | x _ { t } = x \bigg \rangle _ { P ^ { u ^ { * } } } } \\ & { \quad \quad = \langle R _ { t } [ x ( \cdot ) ] - ( R _ { t } [ x ( \cdot ) ] - \log Z _ { t } ( x ) ) \mid x _ { t } = x \rangle _ { P ^ { u ^ { * } } } } \\ & { \quad \quad = \log Z _ { t } ( x ) , } \end{array}\tag{61}
$$

where the expectations above are all taken with respect to the optimal path measure $P ^ { u ^ { * } } [ x ( \cdot ) | x _ { t } = x ]$ . From Eq. (57), we then have $\psi _ { t } ( x ) = Z _ { t } ( x )$ . We henceforth use $\psi _ { t }$ for this conditional partition function. It is a positive function of the current state, not a normalized state density, and can be expressed as a path integral over the passive dynamics:

$$
\begin{array} { l } { \displaystyle \psi _ { t } ( \boldsymbol { x } ) = \int \mathcal { D } \boldsymbol { x } P ^ { 0 } [ \boldsymbol { x } ( \cdot ) | \boldsymbol { x } _ { t } = \boldsymbol { x } ] } \\ { \displaystyle \times \exp \left( \int _ { t } ^ { 1 } r _ { s } ( x _ { s } ) \mathrm { d } s + r _ { 1 } ( x _ { 1 } ) \right) . } \end{array}\tag{62}
$$

This is generalized to a result that a function $\psi _ { t } ( x )$ satisfying Eq. (58) can be expressed as a conditional expectation over the passive dynamics, which is known as the Feynman–Kac formula [5, 23]. Expressing $\psi _ { t }$ in terms of a path integral is familiar from quantum mechanics, where the solution to the Schrodinger equation can be expressed as a path integral over¨ all possible paths weighted by the exponential of the action, and the Feynman–Kac formula is the stochastic process analog of this result.

We can apply this result to the case where we are interested in inferring the dynamics of a system that is observed to be in a particular set $\mathcal { A }$ at time $t = 1$ . For a path measure $P ,$ we write $P ( \mathcal { A } ) \equiv P ( x _ { 1 } \in \mathcal { A } )$ for the probability of this terminal event. In this conditioning example the running state reward is zero, $r _ { t } ( x ) = 0$ We are therefore interested in studying the case $P ^ { u } ( \mathcal { A } ) = 1$ , which requires $r _ { 1 } ( x _ { 1 } )$ to be zero for $x _ { 1 } \in { \mathcal { A } }$ and −∞ for $x _ { 1 } \not \in A ( \mathrm { i . e . , } \psi _ { 1 } ( x _ { 1 } ) = \mathbf { 1 } _ { A } ( x _ { 1 } )$ where $\mathbf { 1 } _ { A }$ is the indicator function for the set A). From Eq. (62), we then have $\psi _ { t } ( x ) = P ^ { 0 } ( A | x _ { t } = x )$ , which is the probability that the system is in the set A at time $t = 1$ given that it is in state x at time t under the passive dynamics; this conditioning function is the function conventionally called h in Doob’ $: h -$ transform [28]. We retain the symbol $\psi _ { t } ( x )$ here. Eq. (60) is then directly Bayes’ rule:

$$
P ^ { u ^ { * } } [ x ( \cdot ) | x _ { t } = x ] = P ^ { 0 } [ x ( \cdot ) | x _ { t } = x ] \frac { \mathbf { 1 } _ { A } ( x _ { 1 } ) } { P ^ { 0 } ( A | x _ { t } = x ) }\tag{63}
$$

From Eq. (59), the inferred dynamics (known as Doob’s $h -$ transform) that lead to the system being in the set A at time $t = 1$ are then given by

$$
\dot { x } _ { t } = f _ { t } ( x _ { t } ) + \varepsilon \nabla \log \psi _ { t } ( x _ { t } ) + \sqrt { \varepsilon } \eta _ { t } .\tag{64}
$$

The logarithmic gradient ∇ log $\psi _ { t } ( x )$ is the conditioning or guidance term. It is closely related to the score $\nabla \log p _ { t } ( x )$ used in generative modeling and diffusion models, but the two are generally distinct: $\psi _ { t } ( x )$ is a conditioning probability, whereas $p _ { t } ( x )$ is a marginal density. We return to scores and guidance in Sec. V C.

## 3. Example: Brownian bridge

The result in Eq. (64) allows us to now study the Brownian bridge (Eq. (42)) introduced in Sec. I C. In particular, we consider one-dimensional Brownian motion as the passive dynamics $( { \mathrm { i . e . , ~ } } f _ { t } ( x ) = 0 )$ and the system is observed to be in a particular state $x _ { 1 } = y$ at time $t = 1$ while starting at $x _ { 0 } = 0$ at time $t = 0$ . To use Eq. (64) to solve this problem, we treat the target set A as a small ball of radius $\delta  0 ^ { + }$ around the target state y, so that $\mathcal { A } = \{ x : \| x - y \| < \delta \}$ . Let $k _ { t | s } ( z | x )$ denote the transition density of the passive process from state x at time s to state z at time $t > s .$ . Then:

$$
\begin{array} { l } { { \displaystyle P ^ { 0 } ( A | x _ { t } = x ) = \int _ { A } \mathrm { d } x _ { 1 } ~ k _ { 1 | t } ( x _ { 1 } | x ) } } \\ { { \displaystyle ~ = \int _ { A } \mathrm { d } x _ { 1 } ~ \frac { 1 } { \sqrt { 2 \pi \varepsilon ( 1 - t ) } } \exp \left( - \frac { \| x _ { 1 } - x \| ^ { 2 } } { 2 \varepsilon ( 1 - t ) } \right) } } \\ { { \displaystyle ~ \approx \frac { 1 } { \sqrt { 2 \pi \varepsilon ( 1 - t ) } } \exp \left( - \frac { \| y - x \| ^ { 2 } } { 2 \varepsilon ( 1 - t ) } \right) \int _ { A } \mathrm { d } x _ { 1 } } } \end{array}\tag{65}
$$

The dynamics of a Brownian bridge are given by Eq. (64):

$$
\dot { x } _ { t } = - \frac { x _ { t } - y } { 1 - t } + \sqrt { \varepsilon } \eta _ { t } .\tag{66}
$$

This means that in the conditioned dynamics, there is a drift term that pushes the system to move in a straight line from the starting point to the target point, with a strength that diverges as $t  1$

Conditioning a stochastic system can therefore be understood dynamically. The information supplied by an observation is expressed as a control that makes the observed behavior typical. The Brownian bridge makes this equivalence concrete, with an endpoint constraint appearing as a drift toward the target.

## II. CONTROL AND TRANSPORT OF DENSITIES

The variational viewpoint of Chapter I compares trajectories through their action or control cost. We now consider the problem of moving an entire distribution between prescribed initial and final distributions, rather than a particle between prescribed locations. An intuitive picture is moving a pile of earth into another shape: the total amount of earth moved is unchanged, but there are many ways to rearrange it. Choosing a rearrangement that minimizes a specified cost is an optimal transport problem. When the motion of mass is stochastic, one considers the most likely evolution consistent with the two distributions at the endpoints. We begin with a version of transport, called the Schrodinger bridge problem, formulated¨ in terms of stochastic control. We then consider more general formulations of transport and the geometry it defines on probability distributions. This geometry allows us to describe the evolution of densities as steepest descent, connecting the present chapter to the discussions of nonequilibrium thermodynamics, sampling, and learning further below.

## A. The Schrodinger bridge problem¨

The Schrodinger bridge problem (SBP) asks for the distri-¨ bution over stochastic trajectories that matches two prescribed endpoint densities while differing as little as possible from a reference process [29–32]. Unlike the fixed-endpoint bridge considered in Sec. I D 3, the initial and final positions are now drawn from distributions $\rho _ { 0 }$ and $\rho _ { 1 }$ . The control generates a curve of densities $t \mapsto \rho _ { t }$ connecting these endpoints while accounting for all possible particle trajectories between them. Fig. 3 illustrates this passage from controlling trajectories to transporting densities.

## 1. From trajectories to densities

As in Sec. I C, we consider non-interacting particles with controlled Langevin dynamics,

$$
\dot { x } _ { t } = f _ { t } ( x _ { t } ) + u _ { t } ( x _ { t } ) + \sqrt { \varepsilon } \eta _ { t } , \qquad 0 \leq t \leq 1 ,\tag{67}
$$

which is Eq. (1) with $g _ { t } ( x , u ) ~ = ~ f _ { t } ( x ) + u$ and feedback $u \ : = \ : u _ { t } ( x )$ Here $\eta _ { t }$ is standard Gaussian white noise and $\varepsilon > 0$ The reference dynamics have $u _ { t } = 0$ . Both the reference and controlled processes are initialized with $x _ { 0 } \sim \rho _ { 0 }$ We write $P ^ { 0 } [ x ( \cdot ) ]$ and $P ^ { u } [ x ( \cdot ) ]$ for the reference and controlled distributions over complete paths, respectively; these are the path distributions introduced in Sec. I C. In contrast, $\rho _ { t } ( x )$ is the density of the position at a single time under $P ^ { u }$ i.e., it is the marginal $p _ { t } ^ { u } ( x )$ of Chapter I. We use $\rho _ { t }$ when emphasizing the transport of this density, and add a superscript when distinguishing different ensembles. For the independent particles considered here, this is both the single-particle marginal density and the density described by a large ensemble. The density generated by Eq. (67) obeys the Fokker– Planck equation, which expresses local conservation of probability:

$$
\begin{array} { c } { \displaystyle \partial _ { t } \rho _ { t } = - \nabla \cdot \left[ ( f _ { t } + u _ { t } ) \rho _ { t } \right] + \frac { \varepsilon } { 2 } \nabla ^ { 2 } \rho _ { t } , } \\ { \displaystyle \rho _ { t = 0 } = \rho _ { 0 } , \quad \rho _ { t = 1 } = \rho _ { 1 } . } \end{array}\tag{68}
$$

It is useful to distinguish the added control $u _ { t }$ from the velocity $v _ { t }$ that transports the density. Eq. (68) can equivalently be written as

$$
\begin{array} { r } { \partial _ { t } \rho _ { t } + \nabla \cdot \left( \rho _ { t } v _ { t } \right) = 0 , \quad } \\ { v _ { t } = f _ { t } + u _ { t } - \frac { \varepsilon } { 2 } \nabla \log \rho _ { t } . } \end{array}
$$

Here $\rho _ { t } v _ { t }$ is the probability current. In the deterministic, driftfree case, $v _ { t } = u _ { t }$ . Note that even when intermediate densities $\rho _ { t }$ are specified, we cannot identify a unique velocity field; adding a field $w _ { t }$ with $\nabla \boldsymbol { \cdot } \left( \rho _ { t } \boldsymbol { w } _ { t } \right) = 0$ leaves the continuity equation unchanged. Specifying the two endpoint densities leaves still more freedom, since the intermediate densities are also unspecified. Thus, as in maximum-entropy inference, the constraints must be supplemented by a criterion for choosing among the compatible path distributions.

Suppose a large collection of independent particles evolving under $P ^ { 0 }$ is observed to begin with density $\rho _ { 0 }$ and end with density $\rho _ { 1 }$ . The large-deviation argument of Eq. (31), now applied to paths, selects the distribution over paths that satisfies these observations and is closest to $P ^ { 0 }$ in KL divergence. This gives the SBP,

$$
\begin{array} { r l r } { \underset { P ^ { u } } { \operatorname* { m i n } } } & { \mathrm { D } _ { \mathrm { K L } } ( P ^ { u } \| P ^ { 0 } ) , } \\ { \mathrm { s . t . } } & { \rho _ { t = 0 } = \rho _ { 0 } , } & { \rho _ { t = 1 } = \rho _ { 1 } , } \end{array}\tag{69}
$$

where $\rho _ { t }$ is the marginal density under $P ^ { u }$ . To relate this problem to control, we write $P ^ { u } [ { \dot { x ( \cdot ) } } ] = \rho _ { 0 } ( x _ { 0 } ) P ^ { u } [ x ( \cdot ) | x _ { 0 } ]$ , and similarly for $P ^ { 0 }$ . The chain rule for KL divergence gives

$$
\begin{array} { r l } & { \mathrm { D } _ { \mathrm { K L } } ( P ^ { u } \| P ^ { 0 } ) } \\ & { \quad = \displaystyle \int \mathrm { d } x _ { 0 } \rho _ { 0 } ( x _ { 0 } ) \mathrm { D } _ { \mathrm { K L } } \big ( P ^ { u } [ \cdot | x _ { 0 } ] \| P ^ { 0 } [ \cdot | x _ { 0 } ] \big ) } \\ & { \quad = \displaystyle \frac { 1 } { 2 \varepsilon } \int _ { 0 } ^ { 1 } \mathrm { d } t \int \mathrm { d } x \rho _ { t } ( x ) \| u _ { t } ( x ) \| ^ { 2 } , } \end{array}\tag{70}
$$

where the second equality follows from Girsanov’s theorem, Eq. (51), applied to paths with the same initial position. Consequently, minimizing Eq. (69) is equivalent to minimizing the expected quadratic control effort subject to Eq. (68) and its endpoint constraints. The SBP thus chooses the least surprising controlled ensemble relative to the reference process, or equivalently the one requiring the least expected control effort. This introduces the link between control, inference and transport, i.e., the control changes the path distribution and, through its single-time marginals, transports the density from $\rho _ { 0 } \ t \mathbf { o \ } \rho _ { 1 }$

## 2. The coupledforward–backward construction

We can impose the initial constraint directly by drawing x<sub>0</sub> from $\rho _ { 0 }$ , and impose the terminal constraint using a Lagrange

![](images/ebadb008695bf25d6404e7e94ce6a2c69cd79ffeccb2606bc8f5035d63e9421b.jpg)

![](images/37f20361e669d652deb2af776749a3a9b2774974908fa66f1c9d3d507f826ece.jpg)

![](images/10dca1381505177d2db4820f3bc7ce3a05ffb9b5de65c1773de5455cfb0e390b.jpg)

![](images/080e7dbb5798dd17348f76758d4d1fae3b50d98fd66e57a17170aa628c918091.jpg)  
FIG. 3. The Schrodinger bridge problem (SBP) for an ensemble of freely diffusing particles. Left: an ensemble of particles begins distributed¨ according to an initial density $\rho _ { 0 }$ and freely diffuses to $\rho _ { 1 } ^ { 0 } .$ , failing to achieve the target terminal density $\rho _ { 1 }$ . Middle: the value function $V _ { t } ( x ) = \log \psi _ { t } ( x )$ which solves the SBP of transporting the ensemble between the prescribed initial and terminal densities subject to a control cost. Right: the resulting control policy transports the particles so that the terminal density of the controlled ensemble matches the target density.

multiplier function $r _ { 1 } ( x _ { 1 } )$ . As in Sec. I D, this multiplier acts as a terminal reward. For a fixed $r _ { 1 }$ , and after dropping terms independent of the control, the problem is

$$
\begin{array} { l } { \displaystyle \operatorname* { m i n } _ { u } \left. \frac { 1 } { 2 \varepsilon } \int _ { 0 } ^ { 1 } \mathrm { d } t \ \| u _ { t } ( x _ { t } ) \| ^ { 2 } - r _ { 1 } ( x _ { 1 } ) \right. _ { P ^ { u } } , } \\ { \displaystyle x _ { 0 } \sim \rho _ { 0 } . } \end{array}\tag{71}
$$

An initial reward $r _ { 0 } ( x _ { 0 } )$ would have the fixed expectation R dx $\rho _ { 0 } ( x ) r _ { 0 } ( x )$ and would not change this minimization. Maximizing the negative of the objective in Eq. (71) is analogous to solving the PISC problem considered in Sec. I D, with no additional running reward. Its value function therefore admits the Cole–Hopf transformation $V _ { t } = \log \psi _ { t }$ <sub>t</sub>, and Eq. (59) gives

$$
\begin{array} { c } { { u _ { t } ^ { * } ( x ) = \varepsilon \nabla \log \psi _ { t } ( x ) , \qquad \psi _ { 1 } ( x ) = e ^ { r _ { 1 } ( x ) } , } } \\ { { \partial _ { t } \psi _ { t } + f _ { t } \cdot \nabla \psi _ { t } + \displaystyle \frac { \varepsilon } { 2 } \nabla ^ { 2 } \psi _ { t } = 0 . } } \end{array}\tag{72}
$$

Unlike a prescribed terminal reward in PISC, $r _ { 1 }$ must now be chosen so that the terminal density is $\rho _ { 1 }$ . Since $\psi _ { t }$ satisfies the backward Kolmogorov equation, it is interpreted as a field that propagates information backwards from $t = 1$

To find the density under this control, define a second field $\psi _ { t } ^ { \dagger } ( x ) = \rho _ { t } ( x ) / \psi _ { t } ( x )$ , so that

$$
\rho _ { t } ( x ) = \psi _ { t } ( x ) \psi _ { t } ^ { \dagger } ( x ) .\tag{73}
$$

The two factors need not be normalized densities. Schrodinger’s key insight was that¨ $\psi _ { t } ^ { \dagger }$ obeys the forward Kolmogorov equation of the reference process,

$$
\partial _ { t } \psi _ { t } ^ { \dagger } + \nabla \cdot ( f _ { t } \psi _ { t } ^ { \dagger } ) - \frac { \varepsilon } { 2 } \nabla ^ { 2 } \psi _ { t } ^ { \dagger } = 0 .\tag{74}
$$

To see this, substitute $\rho _ { t } = \psi _ { t } \psi _ { t } ^ { \dagger }$ and $u _ { t } ^ { * } = \varepsilon \nabla \log \psi _ { t }$ into Eq. (68). Expanding the derivatives gives

$$
\begin{array} { r l } & { \partial _ { t } ( \psi _ { t } \psi _ { t } ^ { \dagger } ) = - \nabla \cdot ( f _ { t } \psi _ { t } \psi _ { t } ^ { \dagger } ) } \\ & { \qquad - \varepsilon \nabla \cdot ( \psi _ { t } ^ { \dagger } \nabla \psi _ { t } ) + \frac { \varepsilon } { 2 } \nabla ^ { 2 } ( \psi _ { t } \psi _ { t } ^ { \dagger } ) } \\ & { \qquad = \psi _ { t } ^ { \dagger } \left[ - f _ { t } \cdot \nabla \psi _ { t } - \frac { \varepsilon } { 2 } \nabla ^ { 2 } \psi _ { t } \right] } \\ & { \qquad + \psi _ { t } \left[ - \nabla \cdot ( f _ { t } \psi _ { t } ^ { \dagger } ) + \frac { \varepsilon } { 2 } \nabla ^ { 2 } \psi _ { t } ^ { \dagger } \right] . } \end{array}\tag{75}
$$

The first bracket is $\partial _ { t } \psi _ { t }$ by Eq. (72). Cancelling ψ $\partial _ { t } \psi _ { t }$ from both sides and dividing by $\psi _ { t }$ yields Eq. (74). Thus, ψ<sub>t</sub> propagates information backward from the final constraint, while $\hat { \psi _ { t } ^ { \dagger } }$ propagates information forward from the initial constraint.

Let $k _ { t \mid s } ( x | y )$ denote the transition density of the reference process, namely, the conditional density of being at x at time t given position y at time $s < t ,$ with $u = 0$ in Eq. (67). It is the propagator of the reference Fokker–Planck equation. The solutions of the backward and forward equations can therefore be written as

$$
\begin{array} { l } { { \displaystyle \psi _ { t } ( x ) = \int \mathrm { d } y ~ k _ { 1 \mid t } ( y \vert x ) \psi _ { 1 } ( y ) , } } \\ { { \displaystyle \psi _ { t } ^ { \dagger } ( x ) = \int \mathrm { d } y ~ k _ { t \mid 0 } ( x \vert y ) \psi _ { 0 } ^ { \dagger } ( y ) . } } \end{array}\tag{76}
$$

The endpoint constraints couple these otherwise linear equations,

$$
\begin{array} { r } { \psi _ { 0 } ( x ) \psi _ { 0 } ^ { \dagger } ( x ) = \rho _ { 0 } ( x ) , } \\ { \psi _ { 1 } ( x ) \psi _ { 1 } ^ { \dagger } ( x ) = \rho _ { 1 } ( x ) . } \end{array}\tag{77}
$$

Note that neither endpoint factor is known in advance; specifying $\psi _ { 1 }$ determines $\psi _ { 0 }$ by backward propagation, while specifying $\psi _ { 0 } ^ { \dagger }$ determines $\psi _ { 1 } ^ { \dagger }$ by forward propagation. One may solve this Schrodinger system by alternating between the two¨ constraints. Starting from a positive guess for $\psi _ { 1 }$ , propagate it backward using the first line of Eq. (76) to obtain $\psi _ { 0 }$ . Set $\psi _ { 0 } ^ { \dagger } = \rho _ { 0 } / \psi _ { 0 }$ , propagate this field forward using the second line to obtain $\psi _ { 1 } ^ { \dagger }$ , and update $\psi _ { 1 } = \rho _ { 1 } / \psi _ { 1 } ^ { \dagger }$ . Repeating these steps enforces the initial and final constraints in turn. This procedure is the so-called Sinkhorn algorithm [32, 33].

With the two fields determined, the optimal distribution over paths is an endpoint reweighting, $\mathrm { ~ o r ~ } ^ { \bullet } \mathrm { t i l t } ^ { \bullet }$ , of the reference distribution [31, 32],

$$
P ^ { u ^ { * } } [ x ( \cdot ) ] = P ^ { 0 } [ x ( \cdot ) ] \frac { \psi _ { 1 } ( x _ { 1 } ) } { \psi _ { 0 } ( x _ { 0 } ) } .\tag{78}
$$

Eq. (78) has the same exponential-tilt structure encountered in maximum-entropy inference and extends the Doob $h -$ transform to prescribed endpoint distributions. Instead of conditioning an endpoint to lie in a set, the endpoint factors are chosen together to reproduce two full densities. The central result is therefore not a new form of the optimal control, which remains $u _ { t } ^ { * } = \varepsilon \nabla \log \psi _ { t }$ , but the coupled forward–backward construction that determines this control from the two distributional constraints.

## B. Entropic optimal transport and the Wasserstein distance

The Schrodinger bridge specifies a complete distribution¨ over paths, and thus solves a dynamic transport problem. More generally, optimal transport considers the least expensive rearrangement of mass from one particular configuration to another, given a cost $c ( x _ { 0 } , x _ { 1 } )$ for moving a unit mass from $x _ { 0 }$ to $x _ { 1 }$ . In Monge’s original formulation, one views optimal transport as the most efficient strategy for moving a pile of earth from one location to another. Each point $x _ { 0 }$ is assigned a destination $T ( x _ { 0 } )$ , and one minimizes R dx<sub>0</sub> $\rho _ { 0 } ( x _ { 0 } ) c ( x _ { 0 } , T ( x _ { 0 } ) )$ while requiring $T ( x _ { 0 } ) \ \sim \ \rho _ { 1 }$ for $x _ { 0 } \sim \rho _ { 0 }$ . However, a deterministic map cannot split mass, and therefore need not exist; for example, a point mass cannot be mapped deterministically into two separated point masses. Kantorovich’s formulation instead optimizes over a joint endpoint distribution $\pi ( x _ { 0 } , x _ { 1 } )$ satisfying

$$
\begin{array} { l } { \displaystyle \int \mathrm { d } x _ { 1 } \pi ( x _ { 0 } , x _ { 1 } ) = \rho _ { 0 } ( x _ { 0 } ) , } \\ { \displaystyle \int \mathrm { d } x _ { 0 } \pi ( x _ { 0 } , x _ { 1 } ) = \rho _ { 1 } ( x _ { 1 } ) . } \end{array}\tag{79}
$$

Any such joint distribution is called a coupling of $\rho _ { 0 }$ and $\rho _ { 1 }$ Minimizing the mean cost over couplings allows the probability at one $x _ { 0 }$ to be distributed among several destinations [34].

To connect this static problem to the Schrodinger bridge, let¨ $k ( x _ { 1 } | x _ { 0 } )$ denote the transition density of the reference process from $t = 0$ to $t = 1$ (we have dropped the 1|0 subscript from our notation in Sec. II A). Consider the following optimization over couplings [31]:

$$
\begin{array} { r } { \pi ^ { * } = \arg \operatorname* { m i n } _ { \pi } \left\{ \varepsilon \mathrm { D } _ { \mathrm { K L } } \big ( \pi \| \rho _ { 0 } k \big ) \right\} , } \end{array}\tag{80}
$$

subject to Eq. (79), where $( \rho _ { 0 } k ) ( x _ { 0 } , x _ { 1 } ) = \rho _ { 0 } ( x _ { 0 } ) k ( x _ { 1 } | x _ { 0 } )$ $\operatorname { N o w } ,$ let’s say $k ( x _ { 1 } | x _ { 0 } ) = \exp \left( - c ( x _ { 0 } , x _ { 1 } ) / \varepsilon \right) / Z ( x _ { 0 } )$ . For example, if the reference process is Brownian motion, we have $c ( x _ { 0 } , x _ { 1 } ) = \left\| x _ { 0 } - x _ { 1 } \right\| ^ { 2 } / 2$ . Here the same $\varepsilon > 0$ that sets the noise covariance in Eq. (67) sets the strength of the regularization (and is analogous to a temperature), and $Z ( x _ { 0 } )$ normalizes the kernel over $x _ { 1 } .$ Using the fixed marginal constraints, Eq. (80) is equivalent, up to constants, to

$$
\begin{array} { r l r } {  { \pi ^ { * } = \arg \operatorname* { m i n } _ { \pi } \Bigg \{ \int \mathrm { d } x _ { 0 } \mathrm { d } x _ { 1 } \pi ( x _ { 0 } , x _ { 1 } ) c ( x _ { 0 } , x _ { 1 } ) } } \\ & { } & { + \varepsilon \int \mathrm { d } x _ { 0 } \mathrm { d } x _ { 1 } \pi ( x _ { 0 } , x _ { 1 } ) \log \pi ( x _ { 0 } , x _ { 1 } ) \Bigg \} . } \end{array}\tag{81}
$$

This objective corresponds to entropic optimal transport [35]. The first term is the mean transport cost, while the second is minus the Shannon entropy of the coupling introduced in Sec. I B 2. Eq. (81) therefore has the form of a constrained free-energy minimization: the cost favors efficient transport, whereas the entropy spreads probability among several possible pairings. $\mathbf { A s } \ \varepsilon  \ 0$ , the entropic term disappears and one obtains the Kantorovich formulation of optimal transport [34, 36].

Introducing Lagrange multipliers $\varphi _ { 0 } ( x _ { 0 } )$ and $\varphi _ { 1 } ( x _ { 1 } )$ for imposing the constraints in Eq. (79), and absorbing the overall normalization into them, gives

$$
\begin{array} { l } { \displaystyle \pi ^ { * } ( x _ { 0 } , x _ { 1 } ) = \exp \left( \frac { \varphi _ { 0 } ( x _ { 0 } ) + \varphi _ { 1 } ( x _ { 1 } ) - c ( x _ { 0 } , x _ { 1 } ) } { \varepsilon } \right) } \\ { \displaystyle \quad = Z ( x _ { 0 } ) \exp \left( \frac { \varphi _ { 0 } ( x _ { 0 } ) } { \varepsilon } \right) k _ { 1 \mid 0 } ( x _ { 1 } | x _ { 0 } ) } \\ { \displaystyle \quad \times \exp \left( \frac { \varphi _ { 1 } ( x _ { 1 } ) } { \varepsilon } \right) . } \end{array}\tag{82}
$$

These multipliers depend on $\varepsilon$ and their zero-noise limits are known as the Kantorovich potentials. A connection to the Schrodinger potentials can be made by setting¨ $\psi _ { 0 } ^ { \dagger } ~ =$ $Z \exp ( \varphi _ { 0 } / \varepsilon )$ and $\psi _ { 1 } = \exp ( \varphi _ { 1 } / \varepsilon )$ , which gives

$$
\begin{array} { r l r } {  { \pi ^ { * } ( x _ { 0 } , x _ { 1 } ) = \psi _ { 0 } ^ { \dagger } ( x _ { 0 } ) k _ { 1 \mid 0 } ( x _ { 1 } | x _ { 0 } ) \psi _ { 1 } ( x _ { 1 } ) , } } \\ & { } & { \psi _ { 0 } ( x _ { 0 } ) = \int \mathrm { d } x _ { 1 } ~ k _ { 1 \mid 0 } ( x _ { 1 } | x _ { 0 } ) \psi _ { 1 } ( x _ { 1 } ) , } \\ & { } & { \psi _ { 1 } ^ { \dagger } ( x _ { 1 } ) = \int \mathrm { d } x _ { 0 } ~ k _ { 1 \mid 0 } ( x _ { 1 } | x _ { 0 } ) \psi _ { 0 } ^ { \dagger } ( x _ { 0 } ) , ~ } \end{array}\tag{83}
$$

together with $\rho _ { 0 } = \psi _ { 0 } ^ { \dagger } \psi _ { 0 }$ and $\rho _ { 1 } = \psi _ { 1 } ^ { \dagger } \psi _ { 1 }$ . These are precisely the endpoint versions of Eqs. (76) and (77). Sinkhorn iteration, analogous to the Sinkhorn algorithm introduced previously, is the alternating rescaling of $\psi _ { 0 } ^ { \dagger }$ and $\psi _ { 1 }$ needed to solve for these quantities, and thus the optimal coupling $\pi ^ { * } \left[ 3 5 \right]$

The quadratic cost $c ( x _ { 0 } , x _ { 1 } ) = \left\| x _ { 0 } - x _ { 1 } \right\| ^ { 2 } / 2$ has special geometric structure. For the zero-noise argument, write $\pi _ { \varepsilon } ^ { * }$ and $\varphi _ { 1 } ^ { \varepsilon }$ to display their dependence on ε. From Eq. (82), the conditional density of $x _ { 1 }$ given $x _ { 0 }$ is

$$
\frac { \pi _ { \varepsilon } ^ { * } ( x _ { 0 } , x _ { 1 } ) } { \rho _ { 0 } ( x _ { 0 } ) } \propto \exp \left[ \frac { \varphi _ { 1 } ^ { \varepsilon } ( x _ { 1 } ) - \frac { 1 } { 2 } \big \| x _ { 0 } - x _ { 1 } \big \| ^ { 2 } } { \varepsilon } \right] ,\tag{84}
$$

where factors depending only on $x _ { 0 }$ are absorbed into the normalization. Rewriting the exponent as

$$
- \frac { \left\| x _ { 0 } \right\| ^ { 2 } } { 2 \varepsilon } + \frac 1 { \varepsilon } \left[ x _ { 0 } \cdot x _ { 1 } - \left( \frac { \left\| x _ { 1 } \right\| ^ { 2 } } 2 - \varphi _ { 1 } ^ { \varepsilon } ( x _ { 1 } ) \right) \right] ,\tag{85}
$$

shows by the Laplace principle that, as $\varepsilon \to 0$ , the conditional density concentrates at the maximizing endpoints. Assuming the potentials converge with a fixed choice of additive constants, denote the limiting potential by $\varphi _ { 1 } = \operatorname* { l i m } _ { \varepsilon \to 0 } \varphi _ { 1 } ^ { \varepsilon }$ and define the supremum

$$
\begin{array} { l } { { \displaystyle \zeta ( x _ { 0 } ) = \operatorname* { s u p } _ { x _ { 1 } } \left\{ x _ { 0 } \cdot x _ { 1 } - \bar { \varphi } _ { 1 } ( x _ { 1 } ) \right\} , } } \\ { { \displaystyle \mathrm { w h e r e } ~ \bar { \varphi } _ { 1 } ( x _ { 1 } ) = \frac { 1 } { 2 } \| x _ { 1 } \| ^ { 2 } - \varphi _ { 1 } ( x _ { 1 } ) . } } \end{array}\tag{86}
$$

If the maximizer $x _ { 1 } ^ { * } ( x _ { 0 } )$ is unique, differentiating the supremum with respect to x<sub>0</sub> gives $\nabla \zeta ( x _ { 0 } ) = x _ { 1 } ^ { * } ( x _ { 0 } )$ , so the limiting optimal coupling is supported on the map

$$
x _ { 1 } = T ( x _ { 0 } ) = \nabla \zeta ( x _ { 0 } ) .\tag{87}
$$

Because $\zeta$ is a convex conjugate, it is convex. This result is known as Brenier’s theorem, which states that for a quadratic cost and absolutely continuous $\rho _ { 0 }$ , the optimal transport map is the gradient of such a convex function [37]. In one dimension, this reduces to the familiar method for mapping a uniform distribution to an arbitrary distribution using the inverse of its cumulative distribution. Specifically, defining the cumulative distribution functions $\begin{array} { r } { C _ { i } \dot { ( } x ) = \int _ { - \infty } ^ { \bar { x } } \mathrm { d } y \ \rho _ { i } \bar { ( } y ) , C _ { 0 } ( x _ { 0 } ) } \end{array}$ is uniform on (0, 1) when $x _ { 0 } \sim \rho _ { 0 }$ , and $x _ { 1 } = C _ { 1 } ^ { - 1 } ( C _ { 0 } ( x _ { 0 } ) )$ has density $\rho _ { 1 }$ , with $C _ { i } ^ { - 1 }$ denoting the quantile function. Hence,

$$
T = C _ { 1 } ^ { - 1 } \circ C _ { 0 }\tag{88}
$$

is the one-dimensional Brenier map.

These special properties motivate the 2-Wasserstein distance,

$$
\begin{array} { l } { { \displaystyle { \cal W } _ { 2 } ^ { 2 } ( \rho _ { 0 } , \rho _ { 1 } ) } \ ~ } \\ { { \displaystyle ~ = \operatorname* { m i n } _ { \pi } \left\{ \int \mathrm { d } x _ { 0 } \mathrm { d } x _ { 1 } ~ \left\| x _ { 0 } - x _ { 1 } \right\| ^ { 2 } \pi ( x _ { 0 } , x _ { 1 } ) \right\} , } } \end{array}\tag{89}
$$

where the minimization is over the couplings in Eq. (79). With the convention $c = \left\| x _ { 0 } - x _ { 1 } \right\| ^ { 2 } / 2$ used above, the unregularized minimum mean cost is $\ddot { W } _ { 2 } ^ { 2 } / 2$ . In one dimension, the relationship to the cumulative distribution gives

$$
W _ { 2 } ^ { 2 } ( \rho _ { 0 } , \rho _ { 1 } ) = \int _ { 0 } ^ { 1 } \mathrm { d } z \left| C _ { 0 } ^ { - 1 } ( z ) - C _ { 1 } ^ { - 1 } ( z ) \right| ^ { 2 } .\tag{90}
$$

The distinction between KL divergence and the 2-Wasserstein distance is illustrated in Fig. 4. In the chosen family of bimodal densities, changes in the relative weights and separation of the modes produce comparable KL divergences, whereas Wasserstein distance distinguishes them strongly because changing the relative weights requires transporting probability between the modes.

Note that the static formulation specifies which initial and final points are paired, but not how mass moves between them. In this deterministic, drift-free setting, the transport velocity equals the control, $\ v _ { t } \ \ = \ u _ { t }$ For a particle following ${ \dot { x } } _ { t } ~ = ~ v _ { t } ( x _ { t } )$ , the Cauchy–Schwarz inequality gives $\begin{array} { r } { \left\| x _ { 1 } - x _ { 0 } \right\| ^ { 2 } \leq \int _ { 0 } ^ { 1 } } \end{array}$ dt $\| v _ { t } ( x _ { t } ) \| ^ { 2 }$ , with equality for straight motion at constant speed. For an entire density, the velocity becomes a field $v _ { t } ( x )$ and valid paths must conserve probability locally. The remarkable content of the Benamou–Brenier theorem is that minimizing this kinetic action over all density paths and velocity fields gives exactly the same quantity as minimizing the static endpoint cost over couplings [36, 39]:

$$
\begin{array} { r l r } {  { W _ { 2 } ^ { 2 } ( \rho _ { 0 } , \rho _ { 1 } ) } } \\ & { } & { = \operatorname* { m i n } _ { \rho _ { t } , v _ { t } } \{ \int _ { 0 } ^ { 1 } \mathrm { d } t \int \mathrm { d } x ~ \rho _ { t } ( x ) \| v _ { t } ( x ) \| ^ { 2 } \} , } \\ & { } & { \partial _ { t } { \rho _ { t } } + \nabla \cdot ( \rho _ { t } v _ { t } ) = 0 , } \end{array}\tag{91}
$$

with endpoint constraints on $\rho _ { 0 }$ and $\rho _ { 1 }$ . For a Brownian reference with $f _ { t } = 0$ , multiplying the objective in Eq. (69) by $2 \varepsilon$ does not change its minimizer. In the limit $\varepsilon \  \ 0$ , the diffusion term in Eq. (68) vanishes, so Eq. (91) can be interpreted as the zero-noise limit of the Schrodinger bridge,¨ namely, the rescaled minimum 2ε min $P ^ { u } \mathrm { D } _ { \mathrm { K L } } ( \bar { P ^ { u } } \Vert P ^ { 0 } )$ approaches $W _ { 2 } ^ { 2 } \ [ 4 0 ]$

Once the optimal map $T$ is known, the minimizing density path is generated by linear interpolation of each transported particle,

$$
x _ { t } = ( 1 - t ) x _ { 0 } + t T ( x _ { 0 } ) , \qquad x _ { 0 } \sim \rho _ { 0 } .\tag{92}
$$

The resulting $\rho _ { t }$ is called displacement interpolation, which is analogous to a straight line in the geometry of probability distributions [41]. A useful feature of the 2-Wasserstein distance is that the static map $T$ also gives us the transport map for all intermediate times $t ,$ which we will return to later when we discuss flow matching in Sec. V C 2. As expected, this picture is consistent with the Brownian bridge discussed previously (Eq. (42)). For a fixed endpoint pair, the mean at an intermediate time t follows $( 1 - t ) x _ { 0 } + t x _ { 1 }$ , with covariance $\varepsilon t ( 1 - t ) I$ $\mathbf { A s } \varepsilon  0 ,$ , the fluctuations disappear and the bridge collapses onto that straight path.

Although we focus on the quadratic cost here, other choices can produce different optimal assignments between the same endpoints. Increasing the exponent in $c ( x _ { 0 } , x _ { 1 } ) =$ $\| x _ { 0 } - x _ { 1 } \| ^ { p }$ penalizes long displacements more strongly and changes how the source is partitioned among those targets. The optimal pairing also depends on the choice of transport cost. Fig. 5 shows an example with three target locations and compares linear, quadratic, and quartic distance costs for the same initial and final densities, showing how this choice changes the regions of the initial density assigned to each target mode.

![](images/398b317301d4fe2d6841102d780eea46505a9af2e04b77903fd3b73e80095660.jpg)

![](images/7f654921eb7b4723011b53192fcbec98b6ae9e5d8a5ff8f53b494a9b441cdd13.jpg)  
FIG. 4. A comparison of the geometries induced by the KL divergence and the 2-Wasserstein distance on a two-parameter family of probability densities $p _ { \theta }$ relative to the reference density $q .$ The inset curves show representative densities at different points in parameter space. The $\theta _ { 1 }$ parameter changes the balance of modes of a bi-modal distribution; the $\theta _ { 2 }$ parameter shifts the distance between the modes. Left: the KL divergence assigns similar costs to changes along the two parameter directions, producing approximately isotropic contours. Right: the 2- Wasserstein distance assigns a greater cost to changes that transport probability between the separated modes, producing strongly anisotropic contours. The figure is adapted from Ref. [38], and the distribution’s parameterization is described in the Supplementary Information of that work.

## C. Wasserstein geometry and Wasserstein gradient flows

As we’ve now seen, the 2-Wasserstein distance $W _ { 2 }$ in Eq. (91) provides a notion of distance between probability densities in terms of the optimal flow field to transport one to another. This notion of distance provides the starting point for Wasserstein geometry, in which one treats the space of probability distributions like a Riemannian manifold in which the square root of the minimum in Eq. (91) is the distance between points [42]. This will allow us to generalize the notion of gradient flow for particles in a potential ${ \dot { x } } = - \nabla F ( x )$ to that of densities flowing down gradients of a functional. This geometric picture will provide the basis for elegant algorithms presented in the Applications Sec. V B.

In Wasserstein geometry, a probability density $\rho$ is regarded as a point, a time-dependent density $\rho _ { t }$ as a curve, and its instantaneous rate of change $s _ { t } ( x ) = \partial _ { t } \rho _ { t } ( x )$ as a tangent vector. Crucially, Wasserstein geometry only permits trajectories $\rho _ { t } ( x )$ that obey the continuity constraint, namely that they can be realized via a flow field $v _ { t }$

$$
\partial _ { t } \rho _ { t } = - \nabla \cdot \left( \rho _ { t } v _ { t } \right)\tag{93}
$$

Some trajectories are not allowed because they cannot be realized by a smooth flow field, for instance $\rho _ { t } = \rho _ { 1 } t + ( 1 - t ) \rho _ { 0 }$ when $\rho _ { 0 }$ and $\rho _ { 1 }$ have disjoint support. This space of probability measures with finite second moment, equipped with $W _ { 2 }$ is denoted W<sub>2</sub> [43, 44].

Due to the continuity constraint in Eq. (93), it is possible to set up a duality between each tangent vector $s ( x ) = \partial _ { t } \rho _ { t }$ and the velocity field v that generates it. The subtlety is that the velocity is not unique: adding a field whose probability current has zero divergence leaves s unchanged. The canonical way to remove this redundancy is motivated by the metric $W _ { 2 }$ . Namely, given the tangent vector s in the tangent space of the distribution $\rho ,$ choose the velocity field v with minimum kinetic energy that generates s

$$
\begin{array} { r } { v = \arg \underset { v ^ { \prime } } { \operatorname* { m i n } } \Big \{ \displaystyle \int \mathrm { d } x \rho \big \| v ^ { \prime } \big \| ^ { 2 } \qquad } \\ { \mathrm { s . t . } \quad s + \nabla \cdot ( \rho v ^ { \prime } ) = 0 \Big \} } \end{array}\tag{94}
$$

This minimization problem can be solved with a Lagrange multiplier for the continuity constraint. Let’s say we choose its sign and normalization by adding $\begin{array}{c} \begin{array} { r l r } { \mathrm { ~ } } & { { } } & { \mathrm { ~ } } \\ { \mathrm { ~  ~ \omega ~ } } & { { } } & { \mathrm { ~ } } \\ { \mathrm { ~  ~ \omega ~ } } & { { } } & { \mathrm { ~ } } \end{array} - 2 \int \mathrm { d } x \chi ( x ) [ s ( x ) + \nabla  \end{array}$ $( \rho v ) ]$ to the kinetic energy. Variation with respect to v then gives $\boldsymbol { v } ( \boldsymbol { x } ) = - \nabla \chi ( \boldsymbol { x } )$ . Thus, the minimum energy velocity is a potential gradient.

Having fixed v in this way, the minimum $\begin{array} { r l } { \| s \| _ { \rho } ^ { 2 } } & { { } = } \end{array}$ $\textstyle \int \mathrm { d } x \rho \| v \| ^ { 2 }$ in Eq. (94) defines the squared norm of the tangent at $\rho .$ Along a regular curve, this is the squared speed measured by $W _ { 2 } { : }$

$$
\| s _ { t } \| _ { \rho _ { t } } = \operatorname* { l i m } _ { \Delta t \to 0 } \frac { W _ { 2 } ( \rho _ { t } , \rho _ { t + \Delta t } ) } { | \Delta t | } , \qquad s _ { t } = \partial _ { t } \rho _ { t } .\tag{95}
$$

Equivalently, $W _ { 2 } ^ { 2 } ( \rho _ { t } , \rho _ { t + \Delta t } ) = ( \Delta t ) ^ { 2 } \| s _ { t } \| _ { \rho _ { t } } ^ { 2 } + o ( ( \Delta t ) ^ { 2 } )$ . The inner product of two tangents $s , s ^ { \prime }$ follows from the polarization identity $\begin{array} { r } { \langle s , s ^ { \prime } \rangle _ { \rho } = \frac { 1 } { 2 } [ \| s + s ^ { \prime } \| _ { \rho } ^ { 2 } - \| s \| _ { \rho } ^ { 2 } - \| s ^ { \prime } \| _ { \rho } ^ { 2 } ] } \end{array}$ , giving

$$
\langle s , s ^ { \prime } \rangle _ { \rho } = \int \mathrm { d } x \rho ( x ) v ( x ) \cdot v ^ { \prime } ( x ) .\tag{96}
$$

![](images/e0bb76439de358edbb73e718d0998115e113a1b6f72ca43f1a4309c3fd5bdc93.jpg)  
FIG. 5. Optimal transport maps between the same source density $\rho _ { 0 }$ and three-mode target density $\rho _ { 1 }$ under different choices of transport cost, c. In each panel, the three colored regions partition the initial density according to the correspondingly colored target mode, while one arrow shows a representative assignment from each region. Increasing the exponent penalizes long displacements more strongly and consequently changes how the initial density is partitioned among the target modes. The axes $x ^ { ( 1 ) } , x ^ { ( 2 ) }$ denote spatial components, whereas the subscripts in x<sub>0</sub>, x<sub>1</sub> denote initial and final endpoints.

where $v ( x )$ and $v ^ { \prime } ( x )$ are the minimum-energy velocity fields associated with s and $s ^ { \prime } ,$ respectively.

With the ingredients of Wasserstein geometry in hand, we can now define gradient descent for a functional $\mathcal { F } [ \rho ]$ . Recall that for a function $F ( x )$ in Euclidean space, its gradient is defined in such a way as to enforce the chain rule: $\nabla F$ is the vector with the property ${ \mathrm { d } } F ( x _ { t } ) / { \mathrm { d } } t = { \dot { x } } _ { t } \cdot \nabla F ( x _ { t } )$ for all trajectories $x _ { t }$ . In complete analogy, the Wasserstein gradient is defined to be the velocity field $\nabla _ { \mathrm { W } } \mathcal { F } [ \rho ]$ , with associated tangent vector $s = - \nabla \cdot ( \rho \nabla \mathrm { w } \mathcal { F } [ \rho ] )$ , such that $\begin{array} { r } { \frac { \mathrm { d } \mathcal { F } [ \rho _ { t } ] } { \mathrm { d } t } = \langle s , \partial _ { t } \rho _ { t } \rangle _ { \rho _ { t } } } \end{array}$ for all trajectories $\rho _ { t } .$ To derive an explicit expression for $\nabla _ { \mathrm { W } } \mathcal { F } [ \rho ]$ , consider dF/dt over a flow generated by a potential $\boldsymbol { v } = - \nabla \chi$ . Writing $\delta \mathcal { F } / \delta \rho$ for the functional derivative, evaluated at the current density $\rho _ { t }$ , one obtains

$$
\begin{array} { r l } & { \frac { \mathrm { d } \mathcal { F } [ \rho _ { t } ] } { \mathrm { d } t } = \int \mathrm { d } x \ \frac { \delta \mathcal { F } } { \delta \rho } \partial _ { t } \rho _ { t } = \int \mathrm { d } x \ \frac { \delta \mathcal { F } } { \delta \rho } \nabla \cdot ( \rho \nabla \chi ) } \\ & { \quad \quad \quad = - \int \mathrm { d } x \ \rho \nabla \frac { \delta \mathcal { F } } { \delta \rho } \cdot \nabla \chi . } \end{array}\tag{97}
$$

Comparing Eq. (97) with the inner product in Eq. (96), the Wasserstein gradient velocity field is given by $\nabla _ { \mathrm { W } } \mathcal { F } [ \rho ] ~ =$ $\nabla ( \delta \mathcal { F } / \delta \rho )$ . This is the gradient in its velocity representation, whereas the associated density tangent is $- \dot { \nabla } \cdot \bar { ( } \rho \nabla _ { \mathrm { W } } \mathcal { F } [ \rho ] )$ A Wasserstein gradientflow (WGF) follows the negative gradient, with descent velocity $v _ { t } = - \nabla _ { \mathrm { W } } \mathcal { F } [ \rho _ { t } ]$ . The resulting density evolution is

$$
\begin{array} { r l } & { \quad \partial _ { t } \rho _ { t } = \nabla \cdot \left( \rho _ { t } \nabla _ { \mathrm { W } } \mathcal { F } [ \rho _ { t } ] \right) , } \\ & { \quad \displaystyle \frac { \mathrm { d } \mathcal { F } [ \rho _ { t } ] } { \mathrm { d } t } = - \int \mathrm { d } x ~ \rho _ { t } \| \nabla _ { \mathrm { W } } \mathcal { F } [ \rho _ { t } ] \| ^ { 2 } \leq 0 . } \end{array}\tag{98}
$$

The WGF is a distributional analog of gradient descent ${ \dot { x } } = $ $- \nabla F ( x )$ in Euclidean space. For example, consider the free energy

$$
\mathcal F [ \rho ] = \int \mathrm { d } x \rho ( x ) \Phi ( x ) + \int \mathrm { d } x \rho ( x ) \log \rho ( x ) .\tag{99}
$$

Eq. (98) becomes $\partial _ { t } \rho _ { t } = \nabla \cdot ( \rho _ { t } \nabla \Phi ) + \nabla ^ { 2 } \rho _ { t }$ , the familiar drift-diffusion Fokker–Planck equation [45]. The potential energy term drives probability down Φ, while the entropy term produces diffusion. Thus, the relaxation of a diffusive system can be read as steepest descent of a free energy in Wasserstein geometry, a connection that will later be discussed in the context of non-equilibrium thermodynamics (Sec. III B) and applications (Sec. V B).

Once steepest descent has been defined, one can ask familiar questions from optimization, now in the space of densities. For example, it is useful to define a notion of convexity that rules out competing local minima and controls the convergence rate—these ideas will come up again in Sec. V B on applications of WGF, particularly in the setting of variational inference. Recall that in Euclidean space, a function $F$ is $\alpha \cdot$ convex if

$$
\begin{array} { r l r } {  { F ( ( 1 - t ) x _ { 0 } + t x _ { 1 } ) } } \\ & { } & { \leq ( 1 - t ) F ( x _ { 0 } ) + t F ( x _ { 1 } ) } \\ & { } & { \quad - \displaystyle \frac { \alpha } { 2 } t ( 1 - t ) \| x _ { 0 } - x _ { 1 } \| ^ { 2 } . } \end{array}\tag{100}
$$

Ordinary convexity corresponds to $\alpha = 0$ . Analogously, a functional $\mathcal { F } : \mathcal { W } _ { 2 } \to \mathbb { R }$ is α-geodesically convex if, along the constant-speed Wasserstein geodesic $\rho _ { t } \ [ 4 1 ]$

$$
\begin{array} { r } { \mathcal { F } [ \rho _ { t } ] \leq ( 1 - t ) \mathcal { F } [ \rho _ { 0 } ] + t \mathcal { F } [ \rho _ { 1 } ] } \\ { - \displaystyle \frac { \alpha } { 2 } t ( 1 - t ) W _ { 2 } ^ { 2 } ( \rho _ { 0 } , \rho _ { 1 } ) . } \end{array}\tag{101}
$$

We emphasize that the comparison in Eq. (101) is made along geodesic interpolation, rather than along the point-wise mixture of two densities. For $\alpha > 0$ , under standard regularity assumptions, if $\rho _ { t }$ follows the Wasserstein gradient flow of an α-geodesically convex ${ \mathcal F } ,$ then

$$
\mathcal { F } [ \rho _ { t } ] - \operatorname* { i n f } \mathcal { F } \leq e ^ { - 2 \alpha t } \left[ \mathcal { F } [ \rho _ { 0 } ] - \operatorname* { i n f } \mathcal { F } \right] .\tag{102}
$$

assuming $\mathcal { F }$ is bounded from below [43]. Eq. (102) states that the functional gap decays exponentially. If a minimizer $\rho _ { * }$ exists, positive geodesic convexity makes it unique and gives $\mathcal { F } [ \rho ] \stackrel { - } { - } \mathcal { F } [ \rho _ { * } ] \stackrel { - } { \geq } \alpha W _ { 2 } ^ { 2 } ( \rho , \rho _ { * } ) / 2 ,$ , so the same estimate also implies exponential convergence in $W _ { 2 }$ . In addition to defining convex functionals, we can define convex sets. We say that a set of probability distributions is geodesically convex if it contains the Wasserstein geodesic between any two of its densities. Functionals that are α-geodesically convex over $\mathcal { W } _ { 2 }$ retain their convexity properties and convergence guarantees when restricted to geodesically convex subsets. In later sections, we will use this framework to describe irreversible thermodynamic relaxation, numerical integration of Fokker– Planck equations, variational inference, and the mean-field dynamics of interacting particles.

## III. THERMODYNAMICS

As we elucidate in this section, the previously explored ideas in control and inference have a deep connection to the time-honored subject of thermodynamics. Upon specifying a state-space geometry, statistical inference aims to select probability distributions consistent with partial information, whereas control theory seeks protocols that transport systems between prescribed configurations while minimizing a cost. Thermodynamics simultaneously determines admissible equilibrium macroscopic states and identifies the least costly transformations among them, compatible with the imposed constraints such as fixed temperature, fixed volume, or adiabaticity. Here, we aim to make these connections concrete. In Sec. III A, we discuss the maximum-entropy interpretation of equilibrium statistical mechanics. Subsequently, in Sec. III B we demonstrate that the Onsager least-dissipation principle and its extension to stochastic processes amount to an optimal transport problem.

## A. Equilibrium thermodynamics as maximum-entropy inference

Here, we discuss the extent to which statistical mechanics of materials in equilibrium can be interpreted as a maximumentropy inference. As materials are composed of $\sim ~ 1 0 ^ { 2 3 }$ degrees of freedom, it is infeasible to keep track of their individual dynamics. Instead, equilibrium statistical mechanics provides a means of describing microscopic particle configurations probabilistically. Using thermodynamic axioms (such as the thermodynamic entropy S reaching a maximum in an isolated system at equilibrium, and the ergodic hypothesis wherein all states are equiprobable within a fixed energy shell), Boltzmann, Gibbs, and Maxwell obtained the expressions for the microstate distribution at equilibrium [46].

Boltzmann argued that by the ergodic hypothesis, all microstates x with the same energy are equally likely in an isolated equilibrium system. He postulated that the microstate multiplicity $W$ compatible with the conserved quantities is maximized at equilibrium (by a reasoning akin to modern saddle-point approaches), and thus sets the thermodynamic entropy via $S \ : = \ : k _ { \mathrm { B } }$ log W (where the logarithm guarantees that entropy is additive). This is called the microcanonical ensemble and can be refined to accommodate additional conserved quantities, like particle number.

Boltzmann then considered a system whose energy fluctuates due to energy exchange with a large thermal bath, constituting the canonical ensemble. Suppose that the system has energy $E ( x )$ and the bath $E _ { \mathrm { t o t } } - E _ { \mathrm { : } }$ , so that $E _ { \mathrm { t o t } }$ is the conserved energy of the joint system (system and bath). The equilibrium probability distribution of the joint configuration $x _ { \mathrm { t o t } } ~ = ~ ( x , x _ { \mathrm { b a t h } } )$ is uniform, $p ^ { \mathrm { e q } } ( x _ { \mathrm { t o t } } ) ~ = ~ 1 / W _ { \mathrm { t o t } } ( E _ { \mathrm { t o t } } )$ However, integrating the bath degrees of freedom for a fixed system configuration x yields

$$
\begin{array} { c } { { p ^ { \mathrm { e q } } ( x ) = \displaystyle \frac { p ^ { \mathrm { e q } } ( x , x _ { \mathrm { b a t h } } ) } { p ^ { \mathrm { e q } } \bigl ( x _ { \mathrm { b a t h } } | x \bigr ) } = \displaystyle \frac { W _ { \mathrm { b a t h } } \bigl ( E _ { \mathrm { t o t } } - E ( x ) \bigr ) } { W _ { \mathrm { t o t } } \bigl ( E _ { \mathrm { t o t } } \bigr ) } } } \\ { { \propto e ^ { - k _ { \mathrm { B } } ^ { - 1 } S _ { \mathrm { b a t h } } \left( E _ { \mathrm { t o t } } - E ( x ) \right) } } } \end{array}\tag{103}
$$

(since $W _ { \mathrm { t o t } } ( E _ { \mathrm { t o t } } ) ~ = ~ \mathrm { c o n s t } )$ Therefore, physics naturally equipped us with a probability following a large-deviation principle, as log $p ^ { \mathrm { { e q } } }$ is an extensive function. Since the bath is much larger than the subsystem $E ( x ) \ \ll \ E _ { \mathrm { t o t } }$ , we Taylor expand $\dot { S } _ { \mathrm { b a t h } } ( E _ { \mathrm { t o t } } - E ( \dot { x } ) ) \simeq S _ { \mathrm { b a t h } } ( E _ { \mathrm { t o t } } ) - E ( x ) / \dot { T }$ where $1 / T = \mathrm { d } \dot { S } _ { \mathrm { b a t h } } ( E ) / \mathrm { d } \dot { E }$ is the bath temperature. We write $\beta = ( k _ { \mathrm { B } } T ) ^ { - 1 }$ for the inverse thermal energy. Thus, the equilibrium distribution of the subsystem microstates x reads:

$$
p ^ { \mathrm { e q } } ( x ) = \frac { 1 } { Z } e ^ { - \beta E ( x ) } ,\tag{104}
$$

We thus arrive at the Boltzmann distribution from physical principles: energy conservation, the ergodic hypothesis, and a large-deviation principle arising from the thermodynamic limit. In the above, $Z$ is the partition function, and it is related to the free energy via $\mathcal { F } ^ { \mathrm { e q } } = - k _ { \mathrm { B } } T$ log $Z .$ Similar fluctuation principles upon coupling to different reservoirs give rise to all other statistical-mechanical ensembles. Notably, the entropy across all ensembles can be written as

$$
S = - k _ { \mathrm { B } } \int \mathrm { d } \boldsymbol { x } p ^ { \mathrm { e q } } ( \boldsymbol { x } ) \log p ^ { \mathrm { e q } } ( \boldsymbol { x } ) .\tag{105}
$$

Note that the Boltzmann factor (Eq. (104)) belongs to the exponential family (Eq. (36)), and the thermodynamic entropy (Eq. (105)) takes the form of a Shannon entropy (Eq. (34)). This suggests a link between equilibrium statistical mechanics and maximum-entropy modelling (discussed in Sec. I B 2). Indeed, Jaynes proposed interpreting thermodynamics as enforcing a maximum-entropy inference [13] wherein, for example, the average energy of a system, $\begin{array} { r } { \mathcal { E } = \int d x p ^ { \mathrm { e q } } ( x ) E ( x ) } \end{array}$ when coupled to a thermal bath, acts as a constraint. Upon maximization, the thermodynamic entropy is given by the Shannon entropy of the entropy-maximizing distribution,

$$
S = k _ { \mathrm { B } } H [ p ^ { \mathrm { e q } } ] .\tag{106}
$$

It is important to stress, however, that the conceptual justification behind maximum-entropy modelling and Boltzmann statistics is fundamentally different. The former is motivated by the principle that, given limited information, the least-committal distribution is the one that maximizes entropy, while the latter follows from physical laws governing equilibrium statistical mechanics. The connection between maximum-entropy modelling and statistical mechanics thus arises from the coincidence between the functional forms obtained in the two frameworks. In that regard, while Eq. (106) is conceptually fascinating, there is no indication that either this relation or the information-theoretic interpretation of statistical mechanics should hold away from equilibrium.

While thermodynamic entropy (like other thermodynamic state functions such as temperature [47] and pressure [48]) is an equilibrium property of matter, is it possible to extend thermodynamic notions far from equilibrium? Since Shannon entropy is well-defined for a system out of equilibrium given its instantaneous distribution $p ( x )$ , can it serve as an extension of entropy outside equilibrium, $S \to k _ { \mathrm { B } } H [ p ]$ , even though this quantity is not generally maximized? The answer lies in whether the emergent relations obtained by presuming a particular entropy functional are consistent with thermodynamic theorems and whether they provide useful predictions, e.g., for the extractable work during thermodynamic transformations.

For example, in certain systems (most often those uncoupled from an external bath), Shannon entropy is explicitly an inadequate measure of thermodynamic entropy [49, 50]. For instance, it is a Lyapunov function of the Liouville equation describing isolated systems, indicating that it does not increase as the system approaches equilibrium (despite the system potentially being ergodic). Therefore, not only may there be no physical reason to invoke maximum-entropy inference in such systems, but Shannon entropy may not be an adequate functional in the first place.

On the other hand, in other cases (most often stochastic systems away from the thermodynamic limit, where the number of particles is small), Shannon entropy provides a useful generalization of thermodynamic entropy out of equilibrium. For example, as we show in Sec. III B 4, it recovers the second law of thermodynamics as well as additional useful fluctuation theorems concerning the extractable work from stochastic systems [51, 52]. Furthermore, returning to Shannon’s construction, Shannon entropy is a functional that adequately quantifies uncertainty [12]. Therefore, in the absence of any knowledge about a problem’s structure beyond a few known expectation values, maximizing entropy under constraints is not an unreasonable inference methodology [14]. Indeed, maximum-entropy inference, both static and dynamical (where $P$ denotes a distribution over trajectories), has proved useful across scales, including predicting protein structure [53], modeling neural communication [54], identifying bacterial swarming transitions [55], and characterizing alignment interactions during flocking [56]. Since thermodynamic variables are often uninformative in these systems, the imposed constraints are no longer temperature, pressure, and related quantities, but rather static and dynamic correlation functions [57, 58], compressed snapshot data [59, 60], transport coefficients [61, 62], and more, which nonetheless carry substantial information about the microstructure of the system. The success of such inference schemes depends not only on the adequacy of the maximum-entropy principle itself, but also on the judicious choice of the moments used to constrain the distribution [55]: moments that capture relevant aspects of the underlying dynamics can yield accurate predictions, whereas constraints on physically irrelevant observables may provide little useful information.

To conclude, there is a deep but subtle connection between equilibrium thermodynamics and information theory. Upon accepting the axioms of thermodynamics, Gibbsian statistical mechanics can be formulated as an inference problem on rigorous grounds. Since entropy is constructed to provide a consistent quantification of statistical uncertainty, exponentialfamily models can also be useful for inferring the steady-state microstructure of athermal systems, albeit with important limitations. The thermodynamic meaning of these models, however, remains subject to debate.

## B. Nonequilibrium thermodynamics as a control problem

In Sec. III A, we recalled the statistical-mechanical characterization of systems at thermodynamic equilibrium, and discussed its information-theoretic interpretation. In this section, we will frame thermodynamic processes close to and far from equilibrium in the language of optimal transport. Thermodynamics provides a concrete notion of efficiency grounded in the second law,

$$
T \Sigma = T \Delta S - Q = W - \Delta \mathcal { F } \geq 0 ,\tag{107}
$$

where W is the mechanical work along a process and $\Delta \mathcal { F }$ is the resulting change in free energy ${ \mathcal F } .$ . Here Σ is the entropy production, so that TΣ is the dissipated heat — the energy irreversibly lost during the transformation. (Note that, confusingly, Σ is often called entropy production; more often than not, entropy production Σ and change in entropy $\Delta S$ are different objects.) In particular, an efficient process between an initial and final state is one which minimizes the entropy production Σ, in turn allowing the extractable work −W to reach the thermodynamic upper bound $- \Delta \mathcal { F }$

In this section, we will discuss useful approaches for estimating the extractable work $W$ and different computational strategies to minimize entropy production Σ using optimal transport and path-integral methods. We begin by studying thermodynamic relaxation for small departures from equilibrium, described by the Onsager least-dissipation theory. We then extend these ideas to stochastic processes far from the thermodynamic limit (few degrees of freedom instead of $N _ { \mathrm { p } } ~ \sim ~ \mathrm { i { \bar { 0 } } ^ { 2 3 } }$ particles), for which we will derive useful thermodynamics-inspired bounds, a geometric interpretation for minimizing the entropy production, and nontrivial equalities for the extractable-work fluctuations.

## 1. Optimal transport bounds efficiency via Onsager’s least-dissipation principle

Let’s start with a modest question — what are the dynamics when we depart from equilibrium but remain close to it? In this setting, Onsager formulated an approach that is phenomenological in nature [63, 64], which has since permitted the calculation of bounds on dissipation in terms of the Wasserstein metric. The connection to Wasserstein geometry ultimately occurs in the context of field theories, but Onsager’s governing principles are perhaps easiest to state in a discrete setting. In brief, consider an isolated system described by the generalized variables $\{ X _ { m } \} _ { m = 1 , \ldots , M } -$ for instance, concentrations of various compounds in a chemical mixture — that determine the system free energy, $\mathcal { F } ( X _ { 1 } , \ldots , X _ { M } )$ . Onsager postulated that, around equilibrium, the system responds linearly to gradients in the free energy:

$$
\frac { \mathrm { d } X _ { m } } { \mathrm { d } t } = - \sum _ { n = 1 } ^ { M } L _ { m n } \frac { \partial \mathcal { F } } { \partial X _ { n } } .\tag{108}
$$

where $L _ { m n }$ is the Onsager matrix of phenomenological transport coefficients. Physically, the coefficients $L _ { m n }$ , which can be measured experimentally as well as predicted theoretically in some cases [65], describe how changing one variable affects all other variables in response. For instance, in the thermoelectric effect, the corresponding $L _ { m n }$ is known as the thermoelectric coefficient and describes how changes in temperature drive changes in charge density. Eq. (108) serves as the basis for modern linear-irreversible thermodynamics [66], wherein a system relaxes to equilibrium linearly in the thermodynamic forces $( \partial \mathcal { F } / \partial X _ { m } )$ , but the forces may be arbitrarily nonlinear in the $X \ ' s$

It is worth noting that Onsager proved various nontrivial properties of $L _ { m n }$ [64]. First, to satisfy the second law of thermodynamics (the nonnegativity of heat dissipation, Eq. (107)), $L _ { m n }$ must be positive-definite. Namely, close to equilibrium, very generally [63] the dissipation functional is given by the thermodynamic dissipative force $\begin{array} { r } { - \sum _ { m } ( L ^ { - 1 } ) _ { n m } \dot { X } _ { m } } \end{array}$ (friction coefficient times velocity) along a displacement $\dot { X } _ { n }$

$$
T \dot { \Sigma } = \sum _ { n , m } ( L ^ { - 1 } ) _ { n m } \dot { X } _ { n } \dot { X } _ { m } ,\tag{109}
$$

which therefore requires a positive-definite $L _ { m n }$ to satisfy $\dot { \Sigma } > 0$ . Moreover, the so-called Onsager reciprocity theorem states that $L _ { m n }$ must be symmetric, $L _ { m n } = L _ { n m } .$ , when the microscopic equations of motion are time-reversal symmetric.

Useful for our purposes is that, for systems obeying Onsager reciprocity [64], Eq. (108) can be obtained from a variational principle. Namely, near equilibrium, the velocities $\{ \dot { X } _ { m } \}$ extremize the following function:

$$
\begin{array} { r } { \mathcal { R } ( \dot { X } _ { 1 } , \dots , \dot { X } _ { M } ) = \displaystyle \frac 1 2 T \dot { \Sigma } + \frac { \mathrm { d } \mathcal { F } } { \mathrm { d } t } \qquad } \\ { \qquad = \frac 1 2 T \dot { \Sigma } + \displaystyle \sum _ { m } \frac { \partial \mathcal { F } } { \partial X _ { m } } \dot { X } _ { m } } \end{array}\tag{110}
$$

where $\mathcal { R }$ is known as the Rayleighian, and in the second equality we used the chain rule for the free energy. Solving $\partial \mathbf { \bar { \mathcal { R } } } / \partial \dot { \bar { X } } _ { m } = 0$ is equivalent to Eq. (108) with the heat dissipation of Eq. (109). In this formulation, it is natural to think of gradients of $\mathcal { F }$ as driving forces, and $L ^ { - 1 }$ as yielding a metric that defines steepest descent.

Onsager’s theory can be connected with optimal transport when Eq. (110) is extended to field theories. We follow the lines of Ref. [67]. In a continuum (hydrodynamic) approach, one models a fluid via a collection of conserved density fields (e.g., mass, momentum, or energy density fields [68]). For simplicity, we will concentrate on a single scalar field $n _ { t } ( x )$ representing the density of a conserved quantity, such as particle density. We denote the conserved particle number by $\begin{array} { r } { N _ { \mathrm { p } } = \int n _ { t } ( x ) } \end{array}$ dx; unlike a probability density, $n _ { t }$ need not integrate to one. Here $n _ { t } ( x )$ plays the role of $X _ { m } ( t )$ in Eqs. (108)–(110), and the spatial coordinate x plays the role of the index m. The field-theoretical generalization of Eq. (110) can be used as a prescription to derive the equations of motion for $n _ { t } ( x )$ . First, because the field is conserved, it evolves according to the continuum equation

$$
\frac { \partial n _ { t } ( x ) } { \partial t } + \nabla \cdot [ n _ { t } ( x ) { v } _ { t } ( x ) ] = 0 ,\tag{111}
$$

where v is the velocity field. Next, we assume that the free energy may be expressed as a local functional of $n ,$ namely $\mathcal { F } [ n ]$ . Finally, we consider phenomenologically that the dissipation Σ<sup>˙</sup> is quadratic in $v _ { t } ( x )$ as in Eq. (109),

$$
T \dot { \Sigma } = \int \mathrm { d } x n _ { t } ( x ) \Gamma ( n _ { t } ( x ) ) | v _ { t } ( x ) | ^ { 2 } .\tag{112}
$$

where the phenomenological dissipation coefficient $\Gamma ( n )$ may be an arbitrary function of density n.

As a concrete example, one can have in mind a system of very dilute colloidal particles in a viscous fluid. In this case $n _ { t } ( x )$ represents the number density of the particles; neglecting hydrodynamic interactions, the drag is given by the Stokes drag Γ = 6πηr (where r is the radius of the suspended particles, and η is the viscosity); and, assuming the particles are non-interacting, the free energy is the negative of the entropy $\begin{array} { r } { \mathcal { F } = - T S = k _ { \mathrm { B } } T \int n _ { t } ( x ) \log n _ { t } ( x ) \mathrm { d } x } \end{array}$

Upon extending the Onsager Rayleighian Eq. (110) to fields, we obtain

$$
\begin{array} { r l } & { \mathcal { R } = \displaystyle \frac { 1 } { 2 } T \dot { \Sigma } + \frac { \mathrm { d } \mathcal { F } } { \mathrm { d } t } } \\ & { ~ = \displaystyle \frac { 1 } { 2 } T \dot { \Sigma } + \int \mathrm { d } x n _ { t } ( \boldsymbol { x } ) v _ { t } ( \boldsymbol { x } ) \cdot \nabla \frac { \delta \mathcal { F } } { \delta n _ { t } ( \boldsymbol { x } ) } . } \end{array}\tag{113}
$$

with the entropy production $\dot { \Sigma }$ of Eq. (112), and we used the functional chain rule and Eq. (111) for evaluating the change in free energy. According to the Onsager variational principle, the equation of motion for $n _ { t } ( x )$ is determined by choosing the flow $v _ { t } ( x )$ that minimizes R. Taking $\delta \mathcal { R } / \delta v = 0$ yields

$$
\Gamma ( n _ { t } ( x ) ) v _ { t } ( x ) = - \nabla \frac { \delta \mathcal { F } } { \delta n _ { t } ( x ) } .\tag{114}
$$

We write $\mu _ { t } ( x ) = \delta \mathcal { F } / \delta n _ { t } ( x )$ for the local chemical potential. Eq. (114), when combined with Eq. (111), yields a closed dynamical equation for n:

$$
\partial _ { t } n = \nabla \cdot \left( { \frac { n } { \Gamma ( n ) } } \nabla { \frac { \delta { \mathcal { F } } } { \delta n } } \right)\tag{115}
$$

Returning to our example of a dilute colloidal suspension, one obtains

$$
\frac { \partial n _ { t } ( x ) } { \partial t } = D \nabla ^ { 2 } n _ { t } ( x ) ,\tag{116}
$$

where $D = k _ { \mathrm { B } } T / ( 6 \pi \eta r )$ is the Stokes-Einstein diffusion constant. Indeed, many of the known dynamical theories in soft matter, ranging from diffusion to phase separation and viscoelasticity, can be derived from Onsager’s variational principle [67].

Onsager’s least-dissipation theorem is closely related to optimal transport — as we show, the dissipation associated with the WGF serves as a lower bound on the true dissipated heat. Note the close resemblance of Eq. (115) to Eq. (98). The Onsager Rayleighian, Eq. (113), can be compared per particle with probability transport. Defining $\rho _ { t } ( x ) = n _ { t } ( x ) / N _ { \mathrm { p } }$ gives

$$
\begin{array} { r l r } {  { \frac { 2 } { \Gamma N _ { \mathrm { p } } } \int _ { 0 } ^ { 1 } \mathrm { d } t \mathcal { R } _ { t } = \int _ { 0 } ^ { 1 } \mathrm { d } t \int \mathrm { d } x \rho _ { t } ( x ) \| v _ { t } ( x ) \| ^ { 2 } } } & { { } } & { { ( 1 1 7 } } \\ { \quad } & { { } } & { { + \displaystyle \int _ { 0 } ^ { 1 } \mathrm { d } t \int \mathrm { d } x \rho _ { t } ( x ) [ \frac { 2 } { \Gamma } \nabla \mu _ { t } ( x ) ] \cdot v _ { t } ( x ) . } } \end{array}
$$

where, for simplicity, we have taken Γ = const. At the same time, return to Eq. (91). Using the method of Lagrange multipliers, the minimum of the 2-Wasserstein distance subject to the conservation law and initial and final distributions is found from the saddle point of the following Lagrangian,

$$
\begin{array} { r l } & { \displaystyle \int _ { 0 } ^ { 1 } \mathrm { d } t \int \mathrm { d } x \rho _ { t } ( x ) \| v _ { t } ( x ) \| ^ { 2 } } \\ & { \displaystyle + \int _ { 0 } ^ { 1 } \mathrm { d } t \int \mathrm { d } x \rho _ { t } ( x ) [ 2 \nabla \chi _ { t } ( x ) ] \cdot v _ { t } ( x ) } \\ & { \displaystyle + \mathrm { ( t e r m s ~ i n d e p e n d e n t ~ o f ~ } v ) . } \end{array}\tag{118}
$$

Here $\chi _ { t } ( x )$ is the transport potential of Chapter II, introduced by adding $- 2 \chi _ { t } [ \partial _ { t } \rho _ { t } + \nabla \cdot \left( \rho _ { t } v _ { t } \right) ]$ to the integrand before integrating by parts. Variation with respect to $v _ { t }$ gives $v _ { t } =$ $- \nabla \chi _ { t }$ , where terms independent of $v _ { t }$ have been ignored.

Eqs. (117) and (118) have the same quadratic velocity term. The former describes the particle density $n _ { t } ~ = ~ N _ { \mathrm { p } } \rho _ { t }$ , with dissipation expressed per particle, whereas the latter is the probability transport variational problem. Thermodynamics fixes the potential to $\chi _ { t } ~ = ~ \mu _ { t } / \Gamma$ , where $\mu _ { t } ~ = ~ \delta \mathcal { F } / \delta n _ { t }$ is the chemical potential, instead of optimizing over $\chi .$ Thus, the Benamou-Brenier solution is a more efficient transportation mechanism;<sup>3</sup> linear-irreversible thermodynamics is a Benamou-Brenier-type “partially” optimized transport, as the chemical potential is specified. We will return to the farreaching consequence of this realization — a bound, tighter than Eq. (107), for finite-time transformations — shortly. But until then — may a similar conclusion apply far from equilibrium?

## 2. Dynamics far from equilibrium

In constructing Onsager’s least-dissipation principle, we considered a coarse-grained hydrodynamic theory close to equilibrium. Yet, the same mathematical structures arise in many other types of dynamics, notably in stochastic processes far from equilibrium. In the following, we will be working within the framework of stochastic thermodynamics [51, 52], aiming to extend thermodynamic notions and theorems to fluctuation-dominated microscopic systems, such as driven colloidal suspensions and individual motor proteins. Since we are no longer in the thermodynamic limit, it is not obvious whether state functions such as temperature or pressure carry over to the microscale or whether the (fluctuating) mechanical work satisfies any relations akin to the second law (Eq. (107)). The immense success of stochastic thermodynamics, summarized in part here and in the coming sections, is the fact that many functions, including mechanical work, can indeed be used still; $e . g .$ , the fluctuating work is just any externally applied forces along a stochastic displacement. Upon selecting the surrounding bath properties (temperature, pressure, etc.) and adopting the Shannon entropy as the microscopic analogue of a nonequilibrium thermodynamic entropy, the second law of thermodynamics and many additional useful new fluctuation relations can be derived. The aim of this section is to explore the correspondence between the thermodynamics of stochastic systems and optimal control.

To demonstrate this point, we will consider the paradigmatic example of an overdamped colloid submerged in a thermal bath, which an agent drives via an external force $F _ { t } ( x )$

$$
\dot { x } _ { t } = D \beta F _ { t } ( x _ { t } ) + \sqrt { 2 D } \eta _ { t } ,\tag{119}
$$

where $D$ is the particle diffusivity and $\eta _ { t }$ is standard white noise. Thus, $\varepsilon ~ = ~ 2 D$ , the stochastic drift is $f _ { t } ( x ) \ =$

$D \beta F _ { t } ( x )$ , and $F _ { t }$ denotes a physical force. The timedependent probability distribution to obtain a microstate, $p _ { t } ( x )$ , satisfies the Fokker-Planck equation [69] — the continuity equation with the current velocity

$$
v _ { t } ( x ) = D \beta F _ { t } ( x ) - D \nabla \log p _ { t } ( x ) .\tag{120}
$$

As preparation for later, we distinguish between two types of steady states: One is an equilibrium state (as in Sec. III $\mathrm { A } )$ where the external driving force is constant in time and conservative, $F = - \nabla E$ , yielding the Boltzmann factor as the equilibrium distribution, Eq. (104), and a zero current, $v  0$ The other is a nonequilibrium steady state, where the current need not be zero $( v \neq 0 )$ , but the flux is divergence-free $\nabla \cdot ( p v )  0 \mathrm { s o } \partial p / \partial t = 0$

Since we may not be close to equilibrium, we may not rely on Onsager’s dissipation functional, and instead refer to the machinery of stochastic thermodynamics [52, 70]. Given our precaution in Sec. III $\mathrm { A } ,$ we note that the Shannon entropy (Eq. (34)) is indeed an adequate choice for the thermodynamic entropy out of equilibrium in stochastic processes, $S [ p _ { t } ] =$ $\begin{array} { r } { - k _ { \mathrm { B } } \int \mathrm { d } x p _ { t } ( x ) \log p _ { t } ( x ) } \end{array}$ , since it will provide insightful and thermodynamically consistent results shortly. The change in entropy over time is

$$
\frac { \mathrm { d } S } { \mathrm { d } t } = - k _ { \mathrm { B } } \int \mathrm { d } x v _ { t } ( x ) \cdot \nabla p _ { t } ( x ) .\tag{121}
$$

According to the first law of thermodynamics, in this setting, the heat exchanged with the bath is minus the external work — the average of the externally applied forces along a displacement,

$$
\dot { Q } = - \dot { W } = - \int \mathrm { d } x p _ { t } ( x ) v _ { t } ( x ) \cdot F _ { t } ( x ) .\tag{122}
$$

Thus, the entropy production is given by

$$
\dot { \Sigma } _ { t } = \frac { \mathrm { d } S } { \mathrm { d } t } - \frac { \dot { Q } } { T } = \frac { k _ { \mathrm { B } } } { D } \int \mathrm { d } x p _ { t } ( x ) | v _ { t } ( x ) | ^ { 2 } .\tag{123}
$$

Indeed, it is zero at equilibrium $( v = 0 )$ , otherwise positive, and functionally consistent with Onsager’s Eq. (112).

Upon further comparison with the Benamou-Brenier approach, Eq. (91), we obtain an optimal-transport-inspired bound on the entropy production [71]. The mere fact that one started and terminated the process at particular distributions during a finite time τ means that one must have produced entropy which is bounded by the Wasserstein metric between the distributions

$$
\int _ { 0 } ^ { \tau } \mathrm { d } t \dot { \Sigma } _ { t } \geq \frac { k _ { \mathrm { B } } } { D \tau } W _ { 2 } ^ { 2 } ( p _ { 0 } , p _ { \tau } ) .\tag{124}
$$

Since $W _ { 2 } ^ { 2 }$ is itself nonnegative, Eq. (124) is a nontrivial bound on the total dissipated heat TΣ that is tighter than the second law of thermodynamics, Eq. (107).

The minimization of Eq. (118) necessitates a gradient current velocity, $\boldsymbol { v } _ { t } = - \nabla \chi _ { t }$ . Therefore, unless $F _ { t }$ is conservative $( F _ { t } = k _ { \mathrm { B } } T \nabla [ \log p _ { t } - \chi _ { t } / D ] )$ , there is no chance to saturate the inequality in Eq. (124), let alone approach $T { \dot { \Sigma } } = 0 .$ Indeed, unless $F _ { t }$ is given by the optimal value of $\chi _ { t } .$ , even the Onsager least-dissipation principle will dissipate more heat than that prescribed by the Wasserstein-metric bound. To illustrate a related consequence of the above thermodynamic interpretations of Wasserstein geometry and its connection to the kinetics of stochastic processes, we outline in Sec. V B 1 the Jordan-Kinderlehrer-Otto scheme [45] in the language of optimal transport.

Interestingly, since $W _ { 2 } ^ { 2 }$ is a property of the two distributions (but not time; cf. Eq. (89)), the bound of Eq. (124) is inversely proportional to the elapsed time. Indeed, the second law only saturates for quasistatic transformations; otherwise, Eq. (124) provides a lower bound on the minimal heat that must have been dissipated to enable a finite-time process. Conversely, Eq. (124) can be interpreted as a thermodynamic speed limit on how quickly one may transform from $p _ { 0 } ( x )$ to $p _ { \tau } ( x )$ , giving rise to the lower bound $\tau \geq k _ { \mathrm { B } } W _ { 2 } ^ { 2 } / ( D \Sigma )$

To conclude, the Benamou-Brenier optimal transport problem sets a bound on the thermodynamic cost of very general, finite-time, nonequilibrium transformations [71]. There are additional, fascinating extensions, stemming from combining optimal transport with stochastic mesoscopic systems. For instance, the above statements were shown to apply to discrete-state Markov systems as well [72]. Upon introducing feedback via measurements, the dissipated heat bound can be made tighter in terms of the acquired information [73]. It is important to stress that although the above are nontrivial connections for far-from-equilibrium systems, these connections only apply in cases where the Einstein relation is valid [71, 72]. Otherwise, the Wasserstein metric only yields a bound on the entropy production in athermal and active systems (which is still given by $\operatorname { E q . }$ (123) [51, 74]), but not on the dissipated heat, as the connection between the latter two breaks in the absence of a scalar temperature [75].

Is it possible to saturate the inequalities provided by the second law of thermodynamics? We present here two strategies — one involving the expression of entropy production as a thermodynamic metric in protocol space [76, 77], and the other involving fluctuation properties of stochastic processes [78, 79].

## 3. Geometric interpretation of entropy production

In the previous sections we found nontrivial bounds on the entropy production using optimal transport. Here, our aim is more practical — is it possible to engineer efficient thermodynamic transformations? By efficient we mean that the entropy production Σ is minimal so, according to Eq. (107), the extractable work is closest to the upper bound dictated by free energy. More concretely, while we did not specify $F _ { t } ( x )$ in Sec. III B 2, here we consider specifically a conservative force, $F _ { t } ( x ) = - \nabla E _ { t } ( x )$ . We furthermore note that the time dependence of the potential, in practice, arises from a “control knob” that the experimentalist varies over time $( e . g .$ , increasing the electric power of an optical trap, thereby increasing the radiation-pressure force on a colloid). These knobs are often called protocols, which we denote by $u _ { t } .$ Finding the most efficient thermodynamic transformation, therefore, entails optimizing the entropy production over the protocols $u _ { t } .$ . For the purpose of this section, the protocol $u _ { t }$ can be thought of as the controls in Sec. I A.

In trying to find the optimization framework for the entropy production, we mention several noteworthy properties of entropy production, Eq. (123): First, clearly the quasi-static transformations are most efficient — since, close to equilibrium $v ( x ) $ 0 everywhere in $x ,$ the dissipation $\Sigma  0$ , necessarily from above due to the quadratic form. Even more so, we see that work extraction in the quasi-static limit is indeed possible. Qualitatively, the slower the time scale over which the process occurs, $\tau  \infty$ , then $v _ { t } ( x ) \sim \tau ^ { - 1 }$ . Since changes in free energy, entropy (Eq. (121)), work, and heat (Eq. (122)) are proportional to v but involve the integration over time, order- $\textstyle \int _ { 0 } ^ { \bar { \tau } }$ dtv ∼ 1 energy extraction is possible while the dissipation is $\sim \int _ { 0 } ^ { \tau } \mathrm { d } t v ^ { 2 } \sim 1 / \tau  0$ . Are all quasi-static transformations the same, or are some more efficient than others? For instance, due to critical slowing down, it may be counterproductive to move throughout Ising-model phase space through the critical point [77].

We restrict our attention to exploring the space of quasistatic transformations, as their entropy production is smallest. Since $v _ { t } ( x )$ is an involved function, requiring the knowledge of the instantaneous distribution in addition to the force, we seek to make the connection of quasistaticity to the protocol $u _ { t }$ more concrete, as follows. For the process to be quasistatic, $p _ { t } ( x )$ must equilibrate on a faster timescale than the protocol evolves; namely all eigenmodes of the Fokker-Planck operator converge to equilibrium faster than the protocol varies. We first define the deviation $\delta p _ { t } ( x )$ of the instantaneous distribution $p _ { t } ( x )$ from the equilibrium Boltzmann factor corresponding to the instantaneous protocol value $p _ { u _ { t } } ^ { \mathrm { e q } } ( x )$

$$
\begin{array} { c } { { p _ { t } ( x ) : = p _ { u _ { t } } ^ { \mathrm { e q } } ( x ) + \delta p _ { t } ( x ) , } } \\ { { \displaystyle p _ { u } ^ { \mathrm { e q } } ( x ) = \frac { 1 } { Z _ { u } } \exp \left[ - \frac { E _ { u } ( x ) } { k _ { \mathrm { B } } T } \right] , } } \\ { { Z _ { u } = \displaystyle \int \mathrm { d } x \exp \left[ - \frac { E _ { u } ( x ) } { k _ { \mathrm { B } } T } \right] . } } \end{array}\tag{125}
$$

By inserting this solution into the Fokker-Planck equation,

$$
\begin{array} { r l r } {  { \frac { \partial p _ { t } ( x ) } { \partial t } = - \mathcal { L } _ { u _ { t } } p _ { t } ( x ) , } } \\ & { } & { \quad \mathcal { L } _ { u } = - D \nabla \cdot [ \frac { \nabla E _ { u } ( x ) } { k _ { \mathrm { B } } T } + \nabla ] , } \end{array}\tag{126}
$$

where $\mathcal { L } _ { u } p _ { u } ^ { \mathrm { e q } } = 0$ , we find the exact expression

$$
\delta p _ { t } ( \boldsymbol { x } ) = - \int _ { - \infty } ^ { t } \mathrm { d } s e _ { \mathrm { T } } ^ { - \int _ { s } ^ { t } \mathrm { d } s ^ { \prime } \mathcal { L } _ { u _ { s ^ { \prime } } } } \dot { u } _ { s } \cdot \frac { \partial p _ { u _ { s } } ^ { \mathrm { e q } } ( \boldsymbol { x } ) } { \partial u } ,\tag{127}
$$

where $e _ { \mathrm { T } }$ is a time-ordered exponential. We will now use the assumption of a quasistatic protocol: Since all modes of $\mathcal { L } _ { u }$ converge prior to $u _ { t }$ changing considerably, $e _ { \mathrm { T } } ^ { - \int _ { s } ^ { t } \mathrm { d } s ^ { \prime } \mathcal { L } _ { u _ { s ^ { \prime } } } }$ suppresses integrand contributions in Eq. (127) where $u _ { s }$ departed much from $u _ { t }$ . Thus, we approximately replace all protocol values $u _ { s }$ with $u _ { t } .$

$$
\delta p _ { t } ( x ) \simeq - \int _ { - \infty } ^ { t } \mathrm { d } s e ^ { - ( t - s ) \mathcal { L } _ { u _ { t } } } \dot { u } _ { t } \cdot \frac { \partial p _ { u _ { t } } ^ { \mathrm { e q } } ( x ) } { \partial u } .\tag{128}
$$

In preparation for later, we define the protocol score (a vector for a multicomponent protocol):

$$
\begin{array} { l } { \displaystyle \Theta _ { u } ( x ) : = \frac { 1 } { p _ { u } ^ { \mathrm { e q } } ( x ) } \frac { \partial p _ { u } ^ { \mathrm { e q } } ( x ) } { \partial u } } \\ { \displaystyle = - \frac { 1 } { k _ { \mathrm { B } } T } \left( \frac { \partial E _ { u } ( x ) } { \partial u } - \left. \frac { \partial E _ { u } ( x ) } { \partial u } \right. _ { u } ^ { \mathrm { e q } } \right) } \end{array}\tag{129}
$$

where $\begin{array} { r } { \langle a ( x ) \rangle _ { u } ^ { \mathrm { e q } } = \int \mathrm { d } x p _ { u } ^ { \mathrm { e q } } ( x ) a ( x ) . } \end{array}$

Recall that the work is, by definition, the energy involved in shifting the potential in time [78],

$$
W = \int _ { 0 } ^ { 1 } \mathrm { d } t \int \mathrm { d } x p _ { t } ( x ) \dot { u } _ { t } \cdot \frac { \partial E _ { u _ { t } } ( x ) } { \partial u } .\tag{130}
$$

and the change in free energy is $\Delta \mathcal { F } ^ { \mathrm { e q } } = - k _ { \mathrm { B } } T$ [log $Z _ { u _ { 1 } } -$ log $Z _ { u _ { 0 } } ]$ . Using Eq. (107), we find the quasi-static expression for the entropy production:

$$
\begin{array} { l } { \displaystyle \Sigma = \frac { 1 } { T } \int _ { 0 } ^ { 1 } \mathrm { d } t \int \mathrm { d } x \delta p _ { t } ( x ) \dot { u } _ { t } \cdot \frac { \partial E _ { u _ { t } } ( x ) } { \partial { u } } } \\ { \displaystyle = k _ { \mathrm { B } } \int _ { 0 } ^ { 1 } \mathrm { d } t \dot { u } _ { t } \cdot \zeta _ { u _ { t } } \cdot \dot { u } _ { t } , } \end{array}\tag{131}
$$

where the friction coefficient can be expressed as the integrated autocorrelation function of the thermodynamic force

$$
\begin{array} { l } { \displaystyle \zeta _ { u } = \int _ { 0 } ^ { \infty } \mathrm { d } s \int \mathrm { d } x p _ { u } ^ { \mathrm { e q } } ( x ) \Theta _ { u } ( x ) e ^ { - s \mathcal { L } _ { u } ^ { \dagger } } \Theta _ { u } ^ { \mathsf { T } } ( x ) } \\ { \displaystyle = \int _ { 0 } ^ { \infty } \mathrm { d } s \langle \Theta _ { u } ( x _ { s } ) \Theta _ { u } ^ { \mathsf { T } } ( x _ { 0 } ) \rangle _ { u } ^ { \mathrm { e q } } , } \end{array}\tag{132}
$$

where the average is in the equilibrium problem, Eq. (119), where the protocol is fixed (with $x _ { 0 } \sim p _ { u } ^ { \mathrm { e q } } )$

Thus, Eq. (131) is an exact expression for the entropy production in terms of a Riemannian geometry in the protocol space [76]. (This contrasts with Eq. (124), which provides an optimal-transport-inspired bound on the entropy production.) The thermodynamic metric, $\zeta _ { u } ,$ describes the “friction” one would experience in a given position in phase space u. We now concretely see how entropy production decreases quadratically with slower driving u˙ . However, we may now further ask questions such as how to optimally transition between states quasistatically, $e . g .$ , how to vary temperature and magnetic field to flip the magnetization in the zerotemperature Ising model [77]. These trajectories would correspond to finding the geodesics on the curved surface described by $\zeta _ { u } , e . g .$ ., using geometric minimum-action methods [80] or by backpropagation [81]. Eq. (131) then allows us to investigate the efficiency and geometry of near-equilibrium thermodynamic transformations.

## 4. Thermodynamic fluctuation theorems

While the above has been a detailed, geometric treatise on close-to-equilibrium thermodynamics, one can find additional, surprising statistical characterizations of the work distributions far from equilibrium. Eq. (130) provides the mean work, and in the previous sections we derived second-lawlike bounds $T \Sigma = \bar { W } - \Delta \mathcal { F } ^ { \mathrm { e q } } \geq 0$ . Here we seek additional statistical properties of the work along a stochastic trajectory $\{ x _ { t } \}$ ,

$$
w _ { t } [ \boldsymbol { x } ( \cdot ) ] = \int _ { 0 } ^ { t } \mathrm { d } s \dot { u } _ { s } \cdot \frac { \partial E _ { u _ { s } } ( \boldsymbol { x } _ { s } ) } { \partial u } , \qquad W = \langle w _ { 1 } \rangle _ { P } ,\tag{133}
$$

where $P$ is the forward path distribution and $w _ { 1 }$ is the work over the complete unit-time protocol. The trajectories start and end at equilibria $( x _ { 0 } \sim p _ { u _ { 0 } } ^ { \mathrm { e q } }$ and $x _ { 1 } \ \sim \ p _ { u _ { 1 } } ^ { \mathrm { e q } } )$ . For that purpose, we will be computing its moment-generating function $( \mathbf { M } \mathbf { G } \mathbf { F } ) , \langle e ^ { \lambda w _ { 1 } } \rangle$ ⟩. We will show that it satisfies two useful equalities: the Jarzynski equality [78, 82],

$$
\langle e ^ { - w _ { 1 } / ( k _ { \mathrm { B } } T ) } \rangle = e ^ { - \Delta \mathcal { F } ^ { \mathrm { e q } } / ( k _ { \mathrm { B } } T ) } ,\tag{134}
$$

and the Crooks fluctuation theorem [79].

$$
\frac { p ( w _ { 1 } ) } { \bar { p } ( \not { p } ( \not { p } ( \not { p } _ { 1 } ) } = e ^ { ( w _ { 1 } - \Delta \mathcal { F } ^ { \mathrm { e q } } ) / ( k _ { \mathrm { B } } T ) } ,\tag{135}
$$

where $\bar { P }$ is the reverse path distribution and $\bar { p }$ is its distribution of work along a trajectory that is being orchestrated by the reversed protocol, $\bar { u } _ { t } : = u _ { 1 - t }$ , which starts from the corresponding (reversed) equilibrium initial condition, $p _ { \bar { u } _ { 0 } } ^ { \mathrm { e q } } = p _ { u _ { 1 } } ^ { \mathrm { e q } }$

The Jarzynski equality has been proven using the Feynman-Kac formula in Ref. [83]. Inspired by their approach, we present a new method for deriving the Crooks fluctuation theorem, involving a Feynman-Kac calculation of the forwardand reverse-process MGFs. While Eq. (134) vividly involves the work MGF at $ \lambda = - 1 / ( k _ { \mathrm { B } } T )$ , we convert Eq. (135) to involve the MGFs by multiplying it by $\bar { p } ( - w _ { 1 } ) e ^ { \lambda w _ { 1 } }$ and integrating over $w _ { 1 }$ , to find:

$$
\langle e ^ { \lambda w _ { 1 } } \rangle _ { P } = e ^ { - \Delta \mathcal { F } ^ { \mathrm { e q } } / ( k _ { \mathrm { B } } T ) } \langle e ^ { - [ \lambda + 1 / ( k _ { \mathrm { B } } T ) ] w _ { 1 } } \rangle _ { \bar { P } } .\tag{136}
$$

If the Crooks fluctuation theorem holds, Eq. (136) readily reproduces the Jarzynski relation, Eq. (134), by inserting $ \lambda = - 1 / ( k _ { \mathrm { B } } T )$ ). Therefore, we aim to derive the Crooks fluctuation theorem, involving both the forward- and backwardrunning controls, $u _ { t }$ and ${ \bar { u } } _ { t } = u _ { 1 - t } .$

Define the work-weighted endpoint density (similar to a moment-generating function) to appear at x at time t along the $u _ { t }$ -controlled trajectories:

$$
M _ { t } ( x , \lambda ) = \left. \exp \left[ \lambda \int _ { 0 } ^ { t } \mathrm { d } s \dot { u } _ { s } \cdot \frac { \partial E _ { u _ { s } } ( x _ { s } ) } { \partial u } \right] \delta ( x - x _ { t } ) \right. _ { P } .\tag{137}
$$

Using Ito’s lemma [ˆ 69], one can show that it follows the initial-value partial-differential equation:

$$
\begin{array} { l } { \displaystyle { \frac { \partial M _ { t } ( x , \lambda ) } { \partial t } = - \mathcal { L } _ { u _ { t } } M _ { t } ( x , \lambda ) + \lambda \dot { u } _ { t } \cdot \frac { \partial E _ { u _ { t } } ( x ) } { \partial u } M _ { t } ( x , \lambda ) , } } \\ { \displaystyle M _ { 0 } ( x , \lambda ) = p _ { u _ { 0 } } ^ { \mathrm { e q } } ( x ) . } \end{array}\tag{138}
$$

We define the initially conditioned MGF along the reversedprotocol, $\bar { u } _ { t } \equiv u _ { 1 - }$ <sub>t</sub>-controlled trajectories:

$$
\begin{array} { r l } { \bar { M } _ { t } ( x , \lambda ) } & { { } = \left. \exp \left[ \lambda \displaystyle \int _ { t } ^ { 1 } \mathrm { d } s \dot { \bar { u } } _ { s } \cdot \frac { \partial E _ { \bar { u } _ { s } } ( x _ { s } ) } { \partial \bar { u } } \right] \Bigg | x _ { t } = x \right. _ { \bar { P } } . } \end{array}\tag{139}
$$

Using the Feynman-Kac formula, it satisfies the terminalvalue partial-differential equation:

$$
\begin{array} { l } { \displaystyle { \frac { \partial \bar { M } _ { t } ( x , \lambda ) } { \partial t } = \mathcal { L } _ { \bar { u } _ { t } } ^ { \dagger } \bar { M } _ { t } ( x , \lambda ) - \lambda \dot { \bar { u } } _ { t } \cdot \frac { \partial E _ { \bar { u } _ { t } } ( x ) } { \partial \bar { u } } \bar { M } _ { t } ( x , \lambda ) , } } \\ { \displaystyle { \bar { M } _ { 1 } ( x , \lambda ) = 1 } . } \end{array}\tag{140}
$$

In order to prove the Jarzynski and Crooks fluctuation theorems, we should find the connection between M and $\bar { M }$ which will involve a time-reversal of the latter.

A useful, intermediate MGF we define is $\widetilde { M } _ { t } ( x , \lambda ) \ : =$ $\bar { M } _ { 1 - t } ( x , - \lambda - \beta )$ . Based on Eq. (140), one can show that $\widetilde { M }$ satisfies the following equation:

$$
\begin{array} { r l r } {  { \frac { \partial \widetilde { M } _ { t } ( \boldsymbol { x } , \lambda ) } { \partial t } = - \mathcal { L } _ { \boldsymbol { u } _ { t } } ^ { \dagger } \widetilde { M } _ { t } ( \boldsymbol { x } , \lambda ) } } \\ & { } & { + ( \lambda + \frac { 1 } { k _ { \mathrm { B } } T } ) \boldsymbol { \dot { u } } _ { t } \cdot \frac { \partial E _ { \boldsymbol { u } _ { t } } ( \boldsymbol { x } ) } { \partial \boldsymbol { u } } \widetilde { M } _ { t } ( \boldsymbol { x } , \lambda ) , } \end{array}\tag{141}
$$

where the forward-time protocol has been restored, $\bar { u } _ { 1 - t } ~ =$ $u _ { t } ,$ and $\begin{array} { r l r } { \mathrm { d } / \mathrm { d } ( 1 - t ) } & { { } = } & { - \mathrm { d } / \mathrm { d } t } \end{array}$ Note the property $e ^ { - E _ { u } ( x ) / ( k _ { \mathrm { B } } ^ { \prime } T ) } \dot { \mathcal { L } } _ { u } ^ { \dagger } e ^ { E _ { u } ^ { \prime } ( x ) / ( k _ { \mathrm { B } } T ) } \stackrel { \prime } { = } \mathcal { L } _ { u }$ . With this in mind, we find that

$$
{ \cal M } _ { t } ( x , \lambda ) = { \frac { 1 } { Z _ { u _ { 0 } } } } e ^ { - E _ { u _ { t } } ( x ) / ( k _ { \mathrm { B } } T ) } \widetilde { \cal M } _ { t } ( x , \lambda ) ,\tag{142}
$$

as by direct substitution, we see that the right-hand side of Eq. (142) indeed satisfies Eq. (138) along with its initial condition. Upon integrating Eq. (142) over $x ,$ evaluated at $t = 1$ and recalling the definitions of the forward and backward MGFs, Eqs. (137) and (139), we recover the Crooks fluctuation theorem, Eq. (136),

$$
\begin{array} { l } { { \displaystyle \langle e ^ { \lambda w _ { 1 } } \rangle _ { P } = \int \mathrm { d } x M _ { 1 } ( x , \lambda ) } } \\ { { \displaystyle \qquad = \frac { Z _ { u _ { 1 } } } { Z _ { u _ { 0 } } } \int \mathrm { d } x p _ { u _ { 1 } } ^ { \mathrm { e q } } ( x ) \bar { M } _ { 0 } ( x , - \lambda - \beta ) } } \\ { { \displaystyle \qquad = \frac { Z _ { u _ { 1 } } } { Z _ { u _ { 0 } } } \langle e ^ { - [ \lambda + 1 / ( k _ { \mathrm { B } } T ) ] w _ { 1 } } \rangle _ { \bar { P } } } . } \end{array}\tag{143}
$$

Upon replacing the partition functions with the free energy, ${ Z _ { u } ^ { \mathrm { { ' } } } = e ^ { - \mathcal { F } _ { u } ^ { \mathrm { { e q } } } / ( \overline { { k } } _ { \mathrm { B } } T ) } }$ , and inverting the Laplace transform, we find the original Crooks fluctuation theorem, Eq. (135).

It is remarkable that for any equilibrium-to-equilibrium transformation, independently of how fast or sub-optimal the controls were, how strong the stochasticity is in the microscopic system, and how far the system has been out of equilibrium during the transformation, equalities such as Eqs. (134) and (135) hold without any approximation. As a corollary, by the Jensen inequality, they reproduce the second law of thermodynamics, $\mathcal { F } _ { u _ { 1 } } ^ { \mathrm { e q } } - \mathcal { F } _ { u _ { 0 } } ^ { \mathrm { e q } } \leq \left. w _ { 1 } \right.$ , which is an inequality for the average work involved in a protocol-driven transformation of the stochastic system. Previous sections involved nearequilibrium expansions or optimal-transport-inspired bounds. Fluctuation relations such as these are routinely used to estimate free energy differences in microscopic transformations [83, 84]. In Sec. IV B, we show that the Jarzynski relation can also have utility for importance sampling.

## IV. SAMPLING AND CONTROL

Sampling from complex, high-dimensional probability distributions is a fundamental task in scientific computing, statistics, and machine learning. The sampling problem asks for the production of independent samples distributed according to a target probability distribution $p _ { 1 }$ . In statistical mechanics, for instance, samples from the Boltzmann distribution allow estimation of physical observables, while in Bayesian inference, samples from the posterior distribution allow predictions based on observed data. This chapter describes the close connection between sampling, inference, control theory, and non-equilibrium thermodynamics, which forms the basis for modern machine-learning methods in reinforcement learning (Sec. V A) and generative modeling (Sec. V C). These links arise from two directions. On the one hand, sampling provides a general-purpose method for solving inference and control problems. On the other hand, control theory and non-equilibrium thermodynamics give rise to practical sampling algorithms. Indeed, such algorithms typically use a controlled, dynamic process that transforms samples from a simple base distribution $p _ { 0 }$ into the more complex target $p _ { 1 }$ . Sampling thus becomes a control and transport problem of “guiding” base samples into regions of high target probability.

Sec. IV A begins by briefly reviewing the use cases and challenges of sampling in high dimension. Many classic algorithms generate samples using equilibrium physics. Sec. IV B introduces Annealed Importance Sampling (AIS), which allows harnessing non-equilibrium processes to generate samples, and can greatly outperform classical methods.

AIS is closely connected to non-equilibrium thermodynamics (Sec. III B). Finally, Sec. IV C casts sampling as a control problem by optimizing the statistical efficiency of the sample-generation process. Vice versa, control problems can be solved by sampling trajectories and keeping count of the resulting reward (Sec. I D 2).

## A. What is sampling and why is it hard?

We begin by fixing the setting. Our goal is to draw N independent samples $x _ { n } \sim p _ { 1 } , n = 1 , . . . , N$ from a complex target distribution $p _ { 1 }$ . We assume that we can evaluate an unnormalized density $\widetilde { p } _ { 1 } ( x )$ for the target, with $p _ { 1 } ( x ) \ : = \ : \widetilde { p } _ { 1 } ( x ) / Z _ { 1 }$ Throughout this chapter, $p$ denotes a normalized probability density and a tilde denotes its unnormalized form. Below, we use dimensionless “energies” and write $E _ { i } ( x ) = - \log \widetilde { p } _ { i } ( x )$ so $p _ { i } = e ^ { - E _ { i } } / Z _ { i }$ and − log p<sub>i</sub> = E<sub>i</sub> + log $Z _ { i } .$ . As we explain next, the sampling task naturally arises in statistical mechanics and Bayesian inference.

## 1. The Monte Carlo method

A key problem in equilibrium statistical mechanics is to calculate expectation values of physical observables, denoted $a ( x )$ for “average”, for example, the magnetization of a piece of metal [7]. This requires summing (or integrating) over all d-dimensional configurations x, weighted by the Boltzmann measure:

$$
\begin{array} { c } { \displaystyle { \langle a \rangle = \int a ( \boldsymbol { x } ) p _ { 1 } ( \boldsymbol { x } ) \mathrm { d } ^ { d } \boldsymbol { x } , } } \\ { \displaystyle { p _ { 1 } ( \boldsymbol { x } ) = \exp \Big ( - \beta E ( \boldsymbol { x } ) \Big ) / Z } } \end{array}\tag{144}
$$

where $\beta = ( k _ { \mathrm { B } } T ) ^ { - 1 }$ , E is the energy, and $Z$ is the normalization constant. Computing this integral in large dimensions requires summing a number of configurations exponential in $d$ (the number of system degrees of freedom). This is generally intractable. A similar problem occurs in Bayesian inference [8], already encountered in Sec. I B. Here, the target distribution $p _ { 1 }$ is the posterior of a statistical model. By Bayes rule, the posterior for parameters or hidden variables x, conditioned on observations $A ,$ reads:

$$
\begin{array} { r l } & { p ( x | A ) = p ( A | x ) p ( x ) / p ( A ) } \\ & { \qquad = \exp \Big ( \log p ( A | x ) + \log p ^ { 0 } ( x ) \Big ) / Z } \end{array}\tag{145}
$$

The target is now $p _ { 1 } ( x ) = p ( x | A )$ . The final form emphasizes that the prior $p ( x )$ , the statistical model $p ( A | x )$ , and the observations A define an effective “energy landscape” for the parameters x. The x-independent $p ( A )$ acts like a normalization constant. Like in statistical mechanics, in complex, highdimensional models, directly computing expectation values, like the parameter means or normalized posterior probabilities, is intractable.

The Monte Carlo method relies on a set of N (independent) samples from the target distribution $p _ { 1 }$ to estimate expectation values:

$$
\begin{array} { c c c } { { \displaystyle { \langle a \rangle _ { 1 } \approx \frac { 1 } { N } \sum _ { n = 1 } ^ { N } a ( x _ { n } ) = : \hat { a } _ { N } , } } } \\ { { \displaystyle { x _ { n } \sim p _ { 1 } , \quad n = 1 , . . . , N . } } } \end{array}\tag{146}
$$

We use ˆ· for estimators, and simplify subscripts $\langle a \rangle _ { 1 } : = \langle a \rangle _ { p _ { 1 } }$ where unambiguous. Since it uses a finite number of samples, the estimator $\operatorname { E q } .$ . (146) is not perfect (throughout, we use ˆ· for estimators). It is correct on average, but has a nonzero mean squared error

$$
\langle ( { \hat { a } } _ { n } - \langle a \rangle ) \rangle = 0 , \quad \left. ( { \hat { a } } _ { n } - \langle a \rangle ) ^ { 2 } \right. = { \frac { \operatorname { V a r } ( a ) } { N } } .\tag{147}
$$

Estimating strongly fluctuating quantities requires many samples.

Computing statistical-mechanical observables and Bayesian inference thus become sampling tasks: Given an energy or an unnormalized probability density, draw independent samples to estimate observables or normalized probabilities.

## 2. Probability distributions in high dimension

Sampling in high dimension $d \gg 1$ is challenging. First, complicated probability distributions often feature multiple, disconnected modes. An example from physics is the Ising model in the ordered phase (Fig. 6, left). In the highdimensional (thermodynamic) limit, the free energy barrier between the two magnetization states scales like d and thus becomes very large [7]. In Bayesian statistics, different modes in the posterior are distinct “explanations” of the observed data. Capturing all modes is therefore essential.

Second, the high-probability regions that sampling must locate and explore become needles in a high-dimensional haystack [85]. This issue is often called the curse of dimensionality. More precisely, high-dimensional probability measures tend to concentrate into very narrow sets (Fig. 7, right), which sampling algorithms must explore. This phenomenon, a generalization of the central limit theorem, is called concentration of measure [86]. In statistical mechanics, it underlies the equivalence of micro- and macrocanonical ensembles in the thermodynamic limit $d \to \infty :$ large energy fluctuations become unlikely.

How fluctuations grow with the number of dimensions is also of great importance in sampling. As a rule of thumb, estimating an extensive quantity like the energy is feasible. Let us consider a scalar observable a that is approximately Gaussian and for which the mean and standard deviation are of the same order. If the magnitude is extensive, $| a | \sim d ,$ the estimator Eq. (146) requires $N \gtrsim d$ samples for an accurate estimate. However, estimating the exponential of an extensive quantity $e ^ { a }$ is very hard. Since $\mathrm { V a r } [ e ^ { a } ] \sim e ^ { 2 \mathrm { V a r } [ a ] } \sim e ^ { 2 d }$ , it requires $N \gtrsim e ^ { d }$ samples and is thus infeasible. Controlling this variance is a key question in sampling algorithm design.

![](images/fd6d02d8dc1a4d4aad7da8db36b354b9b01a67a72dd0518deff1ebef0a3fd28c.jpg)

![](images/8bde4a20870904bd3dbd2e882f1004147aa2ae70811e8e85eb66dd4fc3c11311.jpg)

FIG. 6. (Left) Boltzmann distribution of the magnetization at different temperatures in the fully connected Ising model with energy $\begin{array} { r } { E = d ^ { - 1 } \mathrm { \hat { Z } } _ { i \neq j } \sigma _ { i } \sigma _ { j } } \end{array}$ where the $\sigma _ { i } \in \{ - 1 , 1 \}$ are binary “spins”. The magnetization is the average over its spins, $m ( { \boldsymbol { x } } ) = d ^ { - 1 } \sum _ { i } \sigma _ { i }$ At high temperature, the distribution is simple and unimodal, but at low temperature, it has two disconnected modes. (Right) Magnetization of MCMC samples with $d = 2 0$ spins in the low-temperature ferromagnetic phase $\beta = 2$ . The MCMC sampler carries out a random walk over spin configurations; jumps are barrier-crossing events between positive and negative magnetization. Convergence is very slow: it takes $\sim 1 0 ^ { 6 }$ steps to escape a local free energy minimum.  
![](images/990676f07483239f05853a2d738d49598e75a2e0dfee4ee6e639cbfc7a943a85.jpg)  
FIG. 7. Samples from a Gaussian with zero mean and variance $\sigma ^ { 2 } = 1 / d$ in dimension $d = 2$ and $d = 1 0 0 0$ , plotted in (projected) polar coordinates $r = \sqrt { \textstyle \sum _ { i } x _ { i } ^ { 2 } } , \theta = \arctan ( x _ { 1 } / x _ { 2 } )$ . For large d, samples concentrate on a sphere of radius 1 around the mean.

## 3. Basic sampling algorithms: importance and Langevin sampling

Most sampling algorithms begin with a base distribution $p _ { 0 }$ from which one can sample easily (often, the standard ddimensional Gaussian $\mathcal { N } ( 0 , \mathbb { I } ) ) [ 8 7 ]$ . The game is then to convert samples from $p _ { 0 }$ into samples of the target distribution $p _ { 1 }$ . The most straightforward implementation of this idea, transform sampling, applies a map to base samples. This idea underlies modern generative models and will be discussed in Application Sec. $\bar { \mathsf { V } } \bar { \mathsf { C } }$

Importance sampling is based on reweighting. Assuming that the unnormalized density $\widetilde { p } _ { 1 }$ is known, one uses the base distribution $p _ { 0 }$ to produce a set of proposals $x _ { n } \sim p _ { 0 }$ (in this context, $p _ { 0 }$ is often called the proposal distribution). Weights $\omega _ { n }$ compensate for the difference between $p _ { 0 }$ and $p _ { 1 }$ :

$$
\begin{array} { r l r } & { \langle a \rangle _ { 1 } = \langle a \omega \rangle _ { 0 } = \displaystyle \frac { \langle a \widetilde \omega \rangle _ { 0 } } { \langle \widetilde \omega \rangle _ { 0 } } \approx \displaystyle \frac { \sum _ { n = 1 } ^ { N } \widetilde \omega _ { n } a ( x _ { n } ) } { \sum _ { n = 1 } ^ { N } \widetilde \omega _ { n } } , } & \\ & { \omega ( x ) = \displaystyle \frac { p _ { 1 } ( x ) } { p _ { 0 } ( x ) } , \qquad \widetilde \omega _ { n } = \displaystyle \frac { \widetilde p _ { 1 } ( x _ { n } ) } { \widetilde p _ { 0 } ( x _ { n } ) } , \qquad x _ { n } \sim p _ { 0 } . } & \end{array}\tag{148}
$$

Crucially, the sample estimate only requires evaluating the unnormalized probability densities, since the normalization constants $Z _ { 0 } , Z _ { 1 }$ drop out.

Reweighting converts proposals into target samples. The closer $p _ { 0 }$ is to $p _ { 1 }$ , the more efficiently the method works; if $p _ { 0 }$ and $p _ { 1 }$ are very different, most proposals land in regions of vanishing $p _ { 1 }$ -probability. This means that the finite-sample estimate will be dominated by a few samples with the largest weights. This phenomenon can be quantified by the effective sample size, the number $N _ { \mathrm { e f f } }$ of samples from $p _ { 1 }$ that yield the same error as $N$ importance samples from $p _ { 0 }$ , often approximated as [8]

$$
N _ { \mathrm { e f f } } \approx \left( \sum _ { n = 1 } ^ { N } \widetilde { \omega } _ { n } \right) ^ { 2 } / \left( \sum _ { n = 1 } ^ { N } \widetilde { \omega } _ { n } ^ { 2 } \right) .\tag{149}
$$

The effective sample size is thus set by the $2 ^ { \mathrm { n d } }$ moment of the weights. Let us see how this arises. By Eq. $( 1 4 7 ) , N _ { \mathrm { e f f } }$ samples from $p _ { 1 }$ yield an error $\mathrm { V a r } [ a ] / N _ { \mathrm { e f f } }$ . Instead of the “bare” observable $^ { a , }$ importance sampling estimates the product $a \cdot \omega .$ where $\omega = p _ { 1 } / p _ { 0 } = \exp ( \log p _ { 1 } - \log p _ { 0 } )$ is the weight. The error of the importance sampling estimator Eq. (148) is therefore $\mathrm { V a r } [ a \cdot \omega \bar { ] } / N \sim \mathrm { ( V a r } [ a ] \cdot \mathrm { V a r } [ \omega ] ) / N$ Comparing the two yields $\dot { N } _ { \mathrm { e f f } } ~ = ~ \dot { N } / \dot { \mathrm { V a r } } [ e ^ { \log p _ { 1 } - \mathrm { l o g } p _ { 0 } } ]$ In high dimensions, importance sampling thus becomes ineffective. The ratio $p _ { 1 } / p _ { 0 }$ generally fluctuates wildly since probability measures concentrate in narrow regions [88]. The $\mathrm { \bar { \ e n e r g i e s } } ^ { \mathrm { - } }$ $E _ { 0 , 1 } = - \log \widetilde { p } _ { 0 , 1 }$ are usually extensive, $E _ { 0 , 1 } \ \sim \ d ,$ and so $\omega = p _ { 1 } / p _ { 0 } \sim e ^ { d }$ . Indeed, constraining $p _ { 1 } / p _ { 0 }$ is a key part of importance-sampling-based machine learning algorithms like PPO [89], which we will discuss in Sec. ${ \textrm { V A } } .$

Markov chain Monte Carlo (MCMC) uses an easy-tosimulate stochastic process that converges to the target distribution $p _ { 1 }$ over time t. This generates samples incrementally and allows simulating complex, high-dimensional distributions. Its simplest variant, Langevin sampling, uses Brownian motion in a potential $E _ { 1 } = - \log \widetilde { p _ { 1 } }$

$$
\begin{array} { r } { \dot { x } _ { t } = - \nabla E _ { 1 } + \sqrt { \varepsilon } \eta _ { t } , \quad x _ { 0 } \sim p _ { 0 } } \end{array}\tag{150}
$$

Here $\eta _ { t }$ is the standard Gaussian white noise. We denote the marginal distribution of $x _ { t } ,$ the sampler density, by $q _ { t } ( x )$ The stationary distribution of this process is precisely $p _ { 1 } =$ $\exp ( - E _ { 1 } ) / Z _ { 1 }$ , so after waiting long enough, $x _ { t }$ can generate target samples. MCMC succeeds or fails based on how rapidly $q _ { t }$ equilibrates to the target distribution. However, for complex, high-dimensional distributions, convergence can be painfully slow. For multi-modal distributions, Langevin dynamics can get stuck in a single energy minimum and thus perform poorly (Fig. 6, right). For instance, for a mixture of Gaussians with well-separated peaks $\mu _ { 1 } , \mu _ { 2 }$ , trajectories must cross an energy barrier $\Delta E \sim \ \rvert \mu _ { 1 } - \mu _ { 2 } \rvert ^ { 2 } / \sigma ^ { 2 }$ , which takes a time $\sim \exp ( \Delta E )$ .

While Langevin sampling is conceptually based on equilibrium physics, the algorithms we discuss next use nonequilibrium physics. They transform the base into the target distribution over a fixed, finite time frame, akin to the thermodynamic protocols encountered in Sec. III B. The target distribution is no longer a stationary state. Intuitively, this allows proceeding at “full speed” where equilibrium approaches must “slow down” as the target is approached.

## B. Annealed importance sampling

The method of simulated annealing, originally introduced in statistical physics [7, 90], overcomes this issue and can be seen as an antecedent of modern diffusion methods. Simulated annealing was originally an algorithm to optimize an energy $E ( x )$ . Because greedy optimization methods like gradient descent, $\dot { x } _ { t } ~ = ~ - \nabla E .$ , can get stuck in local minima, one introduces noise (a finite temperature) to allow escape: $\dot { x } _ { t } = - \nabla E + \sqrt { \varepsilon } \eta _ { t }$ . However, if ε is too large, the state $x _ { t }$ spends much time away from the energy minimum due to “thermal” fluctuations. Simulated annealing splits the difference by starting with a high temperature and gradually lowering it over time. The initial high-temperature distribution is supposed to act as a “bridge” connecting different energy minima.

## 1. Sampling with time-varying energies

By reparametrization, a changing noise level can be absorbed into a changing energy. In this section, we rescale so that $\varepsilon = 2$ to simplify notation. One thus generates samples by:

$$
\begin{array} { r } { \dot { x } _ { t } = - \nabla E _ { t } + \sqrt { 2 } \eta _ { t } , \quad x _ { 0 } \sim p _ { 0 } } \end{array}\tag{151}
$$

The process generates proposal samples $x _ { t }$ , with actual marginal density $q _ { t } ( x )$ . For independent runs, $x _ { t } ^ { ( n ) }$ denotes the state in run $n .$ By convention, time runs from $t = 0$ to $t = 1$ . Thus, simulated annealing departs from conventional Langevin dynamics: samples are generated over finite time and using a time-dependent energy $E _ { t }$ . One simple strategy is linear interpolation between the energies:

$$
\begin{array} { l } { E _ { t } = ( 1 - t ) E _ { 0 } + t E _ { 1 } , } \\ { \mathrm { ~ w h e r e ~ } \ p _ { 0 } = e ^ { - E _ { 0 } } / Z _ { 0 } , \ p _ { 1 } = e ^ { - E _ { 1 } } / Z _ { 1 } } \end{array}\tag{152}
$$

Eq. (151) can be thought of as a thermodynamic protocol (Sec. III B). The “instantaneous equilibrium” distribution corresponding to Eq. (152) is $p _ { t } ^ { \mathrm { e q } } = \dot { e } ^ { - E _ { t } } / Z _ { t }$ , with $p _ { 0 } ^ { \mathrm { e q } } = p _ { 0 }$ and $p _ { 1 } ^ { \mathrm { e q } } = p _ { 1 }$

However, this strategy is not yet perfect. The distribution $q _ { 1 }$ of the trajectory endpoints $x _ { 1 }$ from Eq. (151) is not the same as the equilibrium distribution $p _ { 1 }$ , our target. Physically, the transformation induced by the time-dependent energy is non-adiabatic. Mathematically, the Fokker-Planck equation for $q _ { t }$ reads $\begin{array} { r } { \partial _ { t } q _ { t } = \nabla \cdot ( q _ { t } \nabla \dot { E } _ { t } ) + \nabla ^ { 2 } q _ { t } } \end{array}$ . The $p _ { t } ^ { \mathrm { { e q } } }$ cannot be a solution since they are instantaneous equilibria, $\nabla \cdot ( p _ { t } ^ { \mathrm { e q } } \nabla E _ { t } ) + \nabla ^ { 2 } p _ { t } ^ { \mathrm { e q } } = \dot { 0 } \dot { \neq } \partial _ { t } p _ { t } ^ { \mathrm { e q } }$ [91]. Therefore, $x _ { 1 }$ cannot be used directly as a sample for $p _ { 1 }$

Annealed importance sampling [92] (AIS) corrects the annealing process Eq. (151) to produce samples from $p _ { 1 }$ . Similarly to Girsanov’s theorem (Sec. I C 2), AIS computes how the marginal distribution of $x _ { t }$ changes due to the additional force $\nabla \bar { E } _ { t } - \nabla E _ { 0 } = t \nabla ( E _ { 1 } - E _ { 0 } )$ from the time-dependent energy. AIS computes a weight $\omega _ { t }$ for each realization $x _ { t }$ that compensates for the difference between q and $p _ { t } ^ { \mathrm { { e q } } }$ via importance sampling Eq. (148), i.e. $\omega _ { t } = p _ { t } ^ { \mathrm { e q } } ( x _ { t } ) / \bar { q } _ { t } ( x _ { t } )$ . As we will derive shortly, this weight can be calculated “step-bystep” by integrating along the trajectory.

$$
\begin{array} { r l r } & { \displaystyle \frac { \mathrm { d } } { \mathrm { d } t } \log \omega _ { t } = ( \partial _ { t } \log p _ { t } ^ { \mathrm { e q } } ) ( x _ { t } ) , } & \\ & { \displaystyle \omega _ { t } = \frac { Z _ { 0 } } { Z _ { t } } e ^ { - w _ { t } } , \quad } & { w _ { t } = \displaystyle \int _ { 0 } ^ { t } ( \partial _ { s } E _ { s } ) ( x _ { s } ) \mathrm { d } s . } \end{array}\tag{153}
$$

where $w _ { t }$ is the work performed on the trajectory particle due to the changing energy. In practice, the common factor $Z _ { 0 } / Z _ { t }$ need not be computed: when estimating averages, one simply normalizes by the total weight of all AIS samples. A special case of AIS is thermodynamic integration [93], in which the energy $E _ { t }$ is changed quasi-statically.

In non-equilibrium thermodynamics, Eq. (153) is known as Jarzynski’s equality [78] (Sec. III B 4 deduced it from Crooks fluctuation theorem, another non-equilibrium thermodynamics result). Jarzynski’s equality relates the free energy difference $\mathcal { F } _ { 1 } ^ { \mathrm { e q } } - \mathcal { F } _ { 0 } ^ { \mathrm { e \bar { q } } } = - \log \bar { Z } _ { 1 } + \log Z _ { 0 }$ to the work $w _ { t }$ on individual trajectories. It notably allows calculating the free energy difference between two states. This result from thermodynamics was, in fact, an important inspiration for AIS [92].

## 2. Proof of the annealed importance sampling identity

Jarzynski’s equality was proven in Sec. III B 4 using the Feynman-Kac formula. Here, we provide an alternative derivation of Eq. (153), effectively using a path-integral approach [78, 92]. The biggest challenge lies in notation: we already distinguished the instantaneous equilibria $p _ { t } ^ { \mathrm { { e q } } }$ from the distribution $q _ { t }$ of the non-equilibrium process Eq. (151).

In addition, we now introduce the time-reversed version of $\operatorname { E q }$ . (151), in which the energy is changed from $E _ { 1 }$ back to $E _ { 0 }$

$$
\begin{array} { r } { \dot { \bar { x } } _ { t } = - \nabla E _ { 1 - t } + \sqrt { 2 } \eta _ { t } , \quad \bar { x } _ { 0 } \sim p _ { 1 } } \end{array}\tag{154}
$$

Note that the reverse process is initialized at $p _ { 1 }$ , our target: $\bar { q } _ { 0 } = p _ { 1 }$ . We denote its marginal density by $\bar { q } _ { t }$ and its path law by $\bar { P } .$ . We now show how $p _ { t } ^ { \mathrm { { e q } } }$ (equilibrium), $q _ { t }$ (forward process), and $\bar { q } _ { t }$ (reverse process) are related.

We discretize time into small intervals, $t = 0 , d t , 2 d t , . . . , 1 .$ Formally, the probability of a trajectory equals the product of the (infinitesimal) transition probabilities:

$$
P [ \boldsymbol { x } ( \cdot ) ] = p _ { 0 } ( x _ { 0 } ) \prod _ { t } k _ { t } ( x _ { t + d t } | x _ { t } )\tag{155}
$$

In the discretization, we alternate between taking trajectory steps $x _ { t } \mapsto x _ { t } + \bigl ( - \nabla E _ { t } + \sqrt { 2 } \eta _ { t } \bigr ) d t$ and updating the energy, $E _ { t } \mapsto E _ { t + d t }$ . The $k _ { t } ( y | x )$ are one-step transition matrices. They are the equilibrium transition probabilities for an energy “frozen” at $E _ { t } .$ , and, in particular, obey detailed balance. The transition probabilities of the reverse process are therefore<sup>4</sup>:

$$
k _ { t } ( x _ { t } | \boldsymbol { x } _ { t + d t } ) = k _ { t } ( \boldsymbol { x } _ { t + d t } | \boldsymbol { x } _ { t } ) \frac { p _ { t } ^ { \mathrm { e q } } ( \boldsymbol { x } _ { t } ) } { p _ { t } ^ { \mathrm { e q } } ( \boldsymbol { x } _ { t + d t } ) }\tag{156}
$$

Hence, the difference in a trajectory $\boldsymbol { x } _ { t } ^ { } \boldsymbol { \mathbf { \rho } } _ { \mathrm { s } } ^ { }$ log-probability under the forward and reverse processes reads:

$$
\begin{array} { l } { { \log \frac { \bar { P } [ { \boldsymbol { x } } ( 1 - \cdot ) ] } { P [ { \boldsymbol { x } } ( \cdot ) ] } = \sum _ { t } \mathrm { d } t \frac { \log p _ { t + d t } ^ { \mathrm { e q } } ( x _ { t } ) - \log p _ { t } ^ { \mathrm { e q } } ( x _ { t } ) } { \mathrm { d } t } } } \\ { { \mathrm { ~ } = \displaystyle \int \mathrm { d } t ( \partial _ { t } \log p _ { t } ^ { \mathrm { e q } } ) ( x _ { t } ) = \log \omega _ { 1 } . \qquad ( 1 } } \end{array}\tag{57}
$$

The reverse process starts at $\bar { q } _ { 0 } = p _ { 1 }$ , so the weight $\omega _ { 1 }$ corrects for the difference between $q _ { 1 }$ and $p _ { 1 }$ . After reweighting with $\omega ,$ the trajectory endpoints $x _ { 1 } ^ { ( n ) }$ can be used as importance samples.

In summary, AIS is an importance sampling scheme that uses time-dependent energy landscapes to generate samples from a target distribution. Because AIS leverages a nonequilibrium process, it can greatly outperform, say, Langevin sampling, when convergence to equilibrium is slow, for example due to energy barriers. The AIS importance weights are easy to calculate numerically<sup>5</sup>. An example is shown in Fig. 8 for the Ising model in the spontaneously magnetized phase. While Langevin sampling took $1 0 ^ { 6 }$ steps to escape a local free energy minimum, AIS can generate independent samples in 100 steps by cooling down from the unmagnetized phase. While AIS is guaranteed to deliver unbiased importance samples, its practical performance greatly depends on the annealing schedule, i.e., on the choice of the time-dependent energy $E _ { t }$ . We discuss this question next.

![](images/2ac3759195dc2b562eda4439474cf7cbb3b9f1a3642a272d8054c21ce2f57789.jpg)

![](images/f3e4911615beb29f0666a3ecfadafa9e89277ab184b06ca1d909f068b5c53fce.jpg)  
FIG. 8. (Left) Simulated annealing trajectories for the fully connected Ising model with $d = 2 0$ spins, going from $\beta = 0$ to $\beta = 2$ in 100 steps. (Right) Distribution of magnetization at $\beta = 2$ based on annealing runs. Without reweighting, the samples are not distributed correctly.

## C. Sampling as a control problem

We now understand how to generate target samples using the AIS method. Does that mean that the sampling problem is “solved”? Further, the AIS scheme works for any timedependent energy landscape. In which sense is one choice better than another? Just like thermodynamic protocols and the transport plans of Sec. II A can be optimized to minimize energy dissipation or transport costs, we will now discuss how sampling protocols can be optimized for statistical efficiency. As we noted above, importance sampling works best when the proposal and the target distributions are close. Otherwise, many samples land in regions of vanishing probability and are thus “lost”. By Eq. (149), the performance of importance sampling is quantified by the variance of the weights $\omega = \exp ( \log p _ { 1 } - \log p _ { 0 } )$ . Our goal is thus to minimize weight variance to maximize statistical efficiency.

In the thermodynamic language of the Jarzynski equality (Sec. III B 4), minimal weight variance means that the work for moving the system between the two equilibria with free energies $\mathcal { F } _ { 0 , 1 } ^ { \mathrm { e q } } ~ = ~ - \log Z _ { 0 , 1 }$ is minimal. To see this, recall Eq. (153): the unnormalized AIS weights are $\widetilde { \omega } _ { t } = \exp ( - w _ { t } )$ where $w _ { t }$ is the work performed on a trajectory $x _ { t }$ . The Jarzynski equality Eq. (134) states that the free energy difference is $\mathcal { F } _ { 1 } ^ { \mathrm { e q } } - \mathcal { F } _ { 0 } ^ { \mathrm { e \bar { q } } } = - \log \left. e ^ { - w _ { 1 } } \right. \leq \left. w _ { 1 } \right.$ . This bound is saturated when the work $w _ { 1 }$ on all trajectories is equal, and thus the weight variance is zero (we caution that minimal work coincides with minimal variance, but away from the minimum, the two objectives differ). Physically, an efficient protocol is close to adiabatic: it ensures that samples at time t rapidly thermalize to the next energy landscape $E _ { t + d t }$ , and avoids, for instance, regions of phase transitions [77].

In an ideal importance sampler, $\mathrm { V a r } [ \omega ] ~ = ~ 0$ and thus $\omega ( x ) = \mathrm { { c o n s t } }$ . That is, the ideal proposals x are already distributed according to the target $p _ { 1 }$ . To get around this somewhat circular conclusion, we will formulate the design of a good proposal distribution as an optimization problem. Indeed, importance sampling can be seen as an optimal control problem by considering the time-dependent energy gradient $- \nabla E _ { t }$ in Eq. (151) as a control force $u _ { t }$

## 1. Sampler design as a control problem

To formalize sampler design as a Path Integral Stochastic Control (PISC) problem (Sec. I D), we consider a generalized setting. We aim to evaluate expectation values of the form $\boldsymbol { a } \cdot \boldsymbol { e } ^ { R _ { 0 } }$ over trajectories of a stochastic process:

$$
\Big \langle a [ x ( \cdot ) ] e ^ { R [ x ( \cdot ) ] } \Big \rangle , \quad \dot { x } _ { t } = f _ { t } ( x _ { t } ) + \sqrt { \varepsilon } \eta _ { t } ,\tag{158}
$$

The brackets [·] indicate that a and $R$ may now be functionals of the entire trajectory $x ( \cdot )$

To map Eq. (158) to the importance sampling setting, choose:

$$
\begin{array} { l l l } { \displaystyle R \big [ \boldsymbol { x } ( \cdot ) \big ] = \log \frac { p _ { 1 } ( \boldsymbol { x } _ { 1 } ) } { p _ { 0 } ( \boldsymbol { x } _ { 1 } ) } = E _ { 0 } ( \boldsymbol { x } _ { 1 } ) - E _ { 1 } ( \boldsymbol { x } _ { 1 } ) + \log \frac { Z _ { 0 } } { Z _ { 1 } } , } \\ { \displaystyle f _ { t } = - \nabla E _ { 0 } , \varepsilon = 2 , a [ \boldsymbol { x } ( \cdot ) ] = a ( \boldsymbol { x } _ { 1 } ) , x _ { 0 } \sim p _ { 0 } . } \end{array}\tag{59}
$$

The stochastic process produces Langevin samples from the base distribution; R is the log importance weight, so $e ^ { R }$ reweights them to the target; the expectation on the left of Eq. (158) is then $\langle a \rangle _ { p _ { 1 } }$ . As discussed below Eq. (149), in high dimensions, the importance weight $e ^ { R } \sim e ^ { d }$ fluctuates wildly. This makes a finite-sample estimate of Eq. (158) unreliable.

To remedy this, we add a control force $u _ { t }$ to drive trajectories toward higher rewards, and then reweigh the trajectories to account for the change in distribution [26, 94],

$$
\begin{array} { r } { \dot { x } _ { t } = f _ { t } ( x _ { t } ) + u _ { t } ( x _ { t } ) + \sqrt { \varepsilon } \eta _ { t } . } \end{array}\tag{160}
$$

We denote by $P ^ { 0 } [ x ( \cdot ) ]$ the probability of a path without control, and by $P ^ { u } [ x ( \cdot ) ]$ the probability with control. To compensate for the modified dynamics, we reweigh:

$$
\Big \langle a [ x ( \cdot ) ] e ^ { R [ x ( \cdot ) ] } \Big \rangle _ { 0 } = \Big \langle a [ x ( \cdot ) ] e ^ { R [ x ( \cdot ) ] } \frac { P ^ { 0 } [ x ( \cdot ) ] } { P ^ { u } [ x ( \cdot ) ] } \Big \rangle _ { u }\tag{161}
$$

We now need a suitable reward to choose the control u. We will aim to maximize $R ,$ balanced against a quadratic control cost:

$$
u ^ { * } = \arg \operatorname* { m i n } _ { u } \left. \int _ { 0 } ^ { 1 } \frac { \| u _ { t } ( x _ { t } ) \| ^ { 2 } } { 2 \varepsilon } \mathrm { d } t - R [ x ( \cdot ) ] \right. _ { u } .\tag{162}
$$

We thus arrive at a PISC problem. Recall from Eq. (60) that the optimal path measure is a tilted version of the uncontrolled measure:

$$
P ^ { * } [ x ( \cdot ) | x _ { 0 } = x ] = P ^ { 0 } [ x ( \cdot ) | x _ { 0 } = x ] \cdot e ^ { R [ x ( \cdot ) ] } / Z\tag{163}
$$

Indeed, Girsanov’s theorem from Sec. I C 2 shows that $\begin{array} { r l } {  {  \int _ { 0 } ^ { 1 } \| u _ { t } ( x _ { t } ) \| ^ { 2 } / ( 2 \varepsilon ) \mathrm { d } t  _ { \scriptscriptstyle \mathstrut } } } & { { } = \mathrm { D } _ { \mathrm { K L } } ( P ^ { u } \| P ^ { 0 } ) } \end{array}$ . The control cost is thus an entropy, and as we saw in Sec. I B 2, entropy maximization generally leads to solutions that are exponential tilts. Plugging the optimal control into Eq. (161):

$$
\begin{array} { r l } & { \Big \langle a [ x ( \cdot ) ] e ^ { R [ x ( \cdot ) ] } \mid x _ { 0 } = x \Big \rangle _ { 0 } } \\ & { \quad = \Big \langle a [ x ( \cdot ) ] e ^ { R [ x ( \cdot ) ] } \frac { P ^ { 0 } [ x ( \cdot ) ] } { P ^ { * } [ x ( \cdot ) ] } \mid x _ { 0 } = x \Big \rangle _ { u ^ { * } } } \\ & { \quad = Z \left. a [ x ( \cdot ) ] \mid x _ { 0 } = x \right. _ { u ^ { * } } . } \end{array}\tag{164}
$$

The weights $e ^ { R }$ in the expectation are now constant (conditional on the initial condition $x _ { 0 } -$ unconditional expectation values must additionally average over the easy-to-sample $x _ { 0 } \sim p _ { 0 } )$ . In the case of importance sampling, we can compute $\langle a \rangle _ { p _ { 1 } }$ purely from samples $x _ { t } , x _ { 0 } \sim p _ { 0 }$ of the Langevin equation Eq. (160), with the control force found by optimizing Eq. (162). Since no reweighting is necessary, the effective sample size is maximal, $N _ { \mathrm { e f f } } = N ;$ every proposal sample is worth one true sample.

## 2. Non-equilibrium systems and transition path sampling

The formulation Eq. (158) also applies to non-equilibrium systems, like energetically driven chemical reactions [95, 96]. Instead of a Boltzmann distribution, the state $x _ { t }$ of the system at time t follows a non-equilibrium Langevin equation, combining deterministic forces and random fluctuations. The dynamics in $\operatorname { E q . }$ . (158) thus become physically meaningful, rather than a sampling device. Evaluating expectation values over physical trajectories is of great importance, for example, to estimate chemical reaction rates. In this setting, so-called rare events can have outsize importance: the crossing of a potential barrier determines whether a reaction occurs or not. Mathematically, such events can be encoded in the reward R in Eq. (158). One faces the same challenge as above: strong fluctuations in $e ^ { R }$ lead to poor finite-sample estimates (the target event may simply not occur in a finite sample). Variational Path Sampling [95] addresses it precisely as discussed in Sec. IV C 1. One introduces a control $u _ { t }$ to encourage the target rare events, and reweighs trajectories to compensate for the fictitious control when computing physical observables (Eq. (161)).

## 3. Control as a sampling problem

In summary, the design of an optimal importance sampler is a control problem, where the reward is the “target” probability of the trajectories. This result enables systematic optimization of annealing samplers. But, in some sense, we simply kicked the can down the road: how do we solve the resulting control problem? The key to resolving this conundrum is that, vice versa, control problems can be solved by sampling. Indeed, we saw in Sec. I C 3 that control problems can be seen as probabilistic inference problems. Since sampling provides a general-purpose method to solve the latter, it should also be applicable to the former. This suggests an iterative approach in which the sampler is used to solve a control problem, whose solution in turn improves the sampler. This duality between sampling and control will play an important role in Sec. V A on reinforcement learning.

Let us make this precise. The notations $^ { \ast } \psi ^ { \ast }$ and $" R "$ in Eq. (158) are no coincidence. Conditioned on the trajectory’s initial condition, the weighted expectation in Eq. (158) is the Cole–Hopf transform $\bar { \psi _ { s } } ~ = ~ e ^ { V _ { s } ^ { \bf { a } } }$ of the all-important value function of a control problem with accumulated reward $R [ x ( \cdot ) ]$ (Sec. I D 2). We write $R _ { s } [ x ( \cdot ) ]$ for the accumulated reward from time s onward, including the terminal reward and excluding control cost. Recall that the value function obeys the HJB equation (the recursive formulation of optimality), which can be Cole-Hopf transformed into a Schrodinger equa-¨ tion (in imaginary time). The Feynman-Kac formula Eq. (62) shows that the solution to this Schrodinger equation can be¨ obtained by a Feynman path integral over stochastic trajectories:

$$
\begin{array} { c } { { \psi _ { s } ( x ) = \Big \langle e ^ { R _ { s } [ x ( \cdot ) ] } | x _ { s } = x \Big \rangle _ { P ^ { 0 } } , } } \\ { { \dot { x } _ { t } = f _ { t } ( x _ { t } ) + \sqrt { \varepsilon } \eta _ { t } . } } \end{array}\tag{165}
$$

The stochastic trajectories are now initialized at $x _ { s } = x$ instead of at random. Up to this detail, the Feynman-Kac formula is precisely Eq. (158) with $a = 1 , f _ { t }$ as the dynamics in the absence of control, and $R _ { s }$ as the reward of the control problem. Intuitively, Eq. (165) computes the value function by “undirected exploration” (no control applied) and taking inventory of the rewards encountered. In the language of this chapter, the Feynman-Kac formula is a sampling-based approach to control problems: applying the Monte Carlo method to Eq. (165) yields an estimate of the value function from a finite number of stochastic trajectories. However, Monte Carlo estimation faces the same challenge as in importance sampling. In high dimensions, $e ^ { R _ { s } } \sim \overline { { e ^ { d } } }$ fluctuates wildly. This makes a finite-sample estimate of Eq. (165) unreliable. Uncontrolled trajectories are very unlikely to gather high reward $R _ { s } ,$ , while it is precisely high-reward trajectories that dominate Eq. (165).

Sec. IV C 1 explains the solution: add a control force to bias trajectories towards higher rewards. This suggests an iterative procedure, starting with a guess for the optimal control $u _ { t } ^ { ( 0 ) }$

1. Generate importance samples from the $u _ { t } ^ { ( k ) }$ -controlled process (Eq. (161))

2. Estimate log $\widehat { \psi } ^ { ( k ) }$ by Monte Carlo, using the reweighted trajectories (Eq. (165))

3. Recompute the optimal control from the value function by optimizing the expected reward at each timestep $( u _ { t } ^ { ( k + \bar { 1 } ) } ( \bar { \boldsymbol { x } } ) = \varepsilon \nabla \log \widehat { \psi } _ { t } ^ { ( \hat { k } ) } ( \boldsymbol { x } ) , \mathrm { E q . } ( 5 5 ) )$

and repeat until convergence. This procedure is closely related to the policy iteration algorithm for control problems [97], which iterates evaluation of the current control’s value function (steps 1-2) with improvement of the control using the value function (step 3). Here, evaluation happens through Monte Carlo estimation. In Sec. V A on reinforcement learning (RL), we will see that policy iteration forms the basis for state-of-the-art methods like the soft actor-critic algorithm [98], and that controlling the variance of Monte Carlo estimates is a crucial question in RL algorithm design [97]. RL generalizes control theory to the case where the dynamics $f _ { t }$ and reward $r _ { t }$ are not known in advance but must be learned through exploration. RL is often formulated in a somewhat different language, but makes use of the same principles that connect control, inference, and sampling problems.

## V. APPLICATIONS

The Applications chapter will cover three topics. Sec. V A on reinforcement learning builds on control theory (Sec. I A), maximum-entropy methods (Sec. I C), and sampling (Chapter IV); Sec. V B on Wasserstein gradient flows draws on transport (Chapter II) and non-equilibrium thermodynamics (Sec. III B); and Sec. V C on generative models combines ideas from transport (Chapter II) and sampling (Chapter IV). A general theme is moving from settings where the dynamics, rewards, or distributional constraints are known, to settings where they have to be learned from data. The Applications chapter is thus centered on ideas from machine learning.

## A. Reinforcement learning

Before introducing the reinforcement learning (RL) problem, we begin with a simple example to build intuition and illustrate what makes the problem hard. Take a simple problem setting of a robot in a new warehouse that it has never seen before (Fig. 9), where its goal is to reach a target as soon as possible. Importantly, the robot has no prior knowledge about the warehouse, and can only make decisions based on its collected experience and observed rewards. Let’s say that the robot executes random actions until it reaches the goal. This initial path is likely suboptimal and circuitous. Should the robot continue taking this path or explore to find a potentially shorter route? Thus, if the robot wishes to find the optimal route, it must balance between collecting information about its environment and refining the solutions found so far. This balance, known as the exploitation-exploration problem, is the core difficulty of the RL problem.

The RL formalism and methods build on the stochastic optimal control (SOC) framework. As discussed in Sec. I A, SOC aims to find the optimal control $u _ { t } ( x )$ that maximizes the expected cumulative reward in a stochastic environment. Like the SOC problem, the RL problem also aims to solve for optimal controls that maximize a cumulative reward. However, general RL aims to solve for optimal actions from the perspective of a decision-making entity in a new environment where both environmental statistics and rewards are unknown [97], like the robot in a new warehouse. This entity is known as an agent. Recall that, given the dynamics of the environment, a reward function, and boundary conditions, the Hamilton–Jacobi–Bellman (HJB) equation determines the value function and hence the optimal control. Its underlying recursion decomposes the value into the immediate reward and the expected value at the next state. This recursion forms the theoretical backbone of modern RL methods. In RL, finding the optimal control not only involves solving HJB but also requires the agent to accumulate experience such that finding the optimal controls is tractable. For instance, while completely random controls are very suboptimal if an agent already has access to environment dynamics and reward functions (SOC setting), such a strategy may actually be somewhat sensible in a new environment (RL setting), where random controls can help the agent explore and build a better model of the world; more generally, the agent must carefully balance exploitation (e.g. greedily navigating to states which have already been shown to lead to high total rewards) and exploration. The desiderata to maximize reward and gather data to improve controls in the future often compete, in what is known as the exploitation-exploration problem. This balancing act is a function of the restricted, harder RL problem setting and makes the SOC and the RL problem fundamentally different, despite the shared overall objective of finding optimal controls.

The central tension between exploration and exploitation is illustrated by the so-called two-armed bandit problem. An agent repeatedly chooses between two slot machines—the “two arms of a bandit”— each offering one unit of reward with a fixed but unknown probability and zero otherwise. Only the outcome of the chosen arm is observed, and the aim is to maximize the accumulated reward. Playing the arm with the larger observed average reward exploits existing information, whereas exploring the other arm ensures that the arm that has been deemed optimal with finite observations is indeed optimal. The bandit problem admits a solution to the exploration-exploitation tradeoff in a minimal setting, known as the Lai–Robbins bound [99] (in the asymptotic limit of a large number of plays). This bound shows that for strategies that learn efficiently across possible winning probabilities, the expected reward lost relative to knowing the better arm from the outset grows at least logarithmically with the number of plays. The intuition comes from the fact that large deviations in the observed reward can make a sub-optimal arm appear optimal. Sanov’s theorem, Eq. (27), tells us that the probability of the better arm producing observations that make it seem sub-optimal decreases exponentially with the number of times it is sampled, at a rate determined by the KL divergence. Such misleading data could cause the agent to neglect the better arm, with a loss that grows with the duration of play. Suppressing these rare but costly errors therefore requires continued exploration. The competition between exponentially decreasing probabilities and accumulating losses explains the logarithmic scale in the bound. Thus, successful strategies for exploration and exploitation are ones that increasingly favor the apparently better arms (or, more generally, actions) while continuing to test alternatives that the data have not yet ruled out.

## 1. The Markov decision process

Now that we have some intuition about the structure and fundamental challenges in the RL problem, we introduce necessary terminology to discuss key algorithms. States x are the RL vocabulary for coordinates in a particular state space $\mathcal { X } .$ For example, in the state space of the Cartesian plane, one possible state an agent can occupy is the origin. An environment is a space of states that an agent can navigate subject to transition dynamics $k ( \cdot | x , u )$ , where the transitions are a (generally stochastic) function of the current state and current control. A reward function $r ( x , u )$ is a scalar function that takes as input a state and control and outputs a reward. Reinforcement learning is defined on discrete time steps $n ;$ we will use the subscripts $x _ { n }$ and $u _ { n }$ to denote the states and controls at decision time $t _ { n }$ . For finite-horizon problems, we include the timestep n in the state x throughout this section; policies and value functions therefore need no separate time subscript. An agent is a decision-making entity that executes controls in the world based on its observed transitions and rewards [97]. The controls an agent executes are known as actions u which lie within an action space U. Reactive controls are also referred to as a policy in RL, where the policy $\pi ( u | x )$ is a distribution over controls conditioned on the current state x.

If the transition dynamics are Markovian – that is, if the statistics of the next state depend only on the current state and action and not any prior states and actions – the transition dynamics, reward function, state space, and action space define what is called a Markov Decision Process (MDP). If the transition dynamics are non-Markovian and rely on additional history, this problem is referred to as a Partially Observed Markov Decision Process, or a POMDP [100]. We do not consider POMDPs here, but note that effective approaches include augmenting model-free MDP algorithms with recurrent neural networks to incorporate memory [101].

For the finite-horizon formulas, let $n = 0 , \ldots , N$ label decision times $t _ { n } = n \Delta t$ . The agent takes an action and receives a reward at the final step N. A policy induces a joint stateaction path law $P ^ { \pi } [ x _ { 0 : N } , u _ { 0 : N } ]$ , where $x _ { 0 : N } = ( x _ { 0 } , \ldots , x _ { N } )$ and similarly for u<sub>0:N</sub>.

a. The Objective. Let π be an arbitrary policy. Denote the path measure of this policy as $P ^ { \pi }$ . Then, for some reward function $r ( x , u )$ , the goal of RL is to maximize the objective $\mathcal { I } [ \pi ]$

$$
\mathcal { I } [ \pi ] \triangleq \mathbb { E } _ { P ^ { \pi } } \left[ \sum _ { n = 0 } ^ { N } r ( x _ { n } , u _ { n } ) \right] , \qquad \operatorname* { m a x } _ { \pi } \mathcal { I } [ \pi ] .
$$

In RL, it is standard to consider a cumulative reward modified by a discount factor $0 ~ < ~ \gamma ~ < ~ 1$ that is taken to be close to 1. The discount factor weighs the reward and incentivizes accumulating reward sooner, rather than later:

$$
\mathcal { I } [ \pi ] \triangleq \mathbb { E } _ { P ^ { \pi } } \left[ \sum _ { n = 0 } ^ { N } \gamma ^ { n } r ( x _ { n } , u _ { n } ) \right] , \qquad \operatorname* { m a x } _ { \pi } \mathcal { I } [ \pi ] .
$$

Another way to understand the discount factor is that it imposes an independent $1 - \gamma$ probability at every timestep that an agent permanently transitions to a “death” state with no reward (i.e. an absorbing state with no reward).

b. Value Functions For the value definitions and Bellman equations in this subsection, we use the undiscounted convention $\gamma = 1$ . To solve the MDP, RL algorithms invoke theoretical tools from SOC. These methods rely on value estimation, where value denotes the expected cumulative future reward conditioned on a starting state x at step n, for a particular policy π:

$$
V ^ { \pi } ( \boldsymbol { x } ) = \mathbb { E } _ { P ^ { \pi } } \left[ \sum _ { j = n } ^ { N } r ( x _ { j } , u _ { j } ) \Bigg | x _ { n } = \boldsymbol { x } \right] .\tag{166}
$$

This value function is also known as a critic that critiques the “goodness” of a particular policy π at state x. For n $< N$ , the value function satisfies the Bellman equation for policy π:

$$
V ^ { \pi } ( \boldsymbol { x } ) = \mathbb { E } _ { \boldsymbol { u } \sim \pi ( \cdot | \boldsymbol { x } ) } \bigl [ r ( \boldsymbol { x } , \boldsymbol { u } ) + \mathbb { E } _ { \boldsymbol { x } ^ { \prime } \sim \boldsymbol { k } ( \cdot | \boldsymbol { x } , \boldsymbol { u } ) } [ V ^ { \pi } ( \boldsymbol { x } ^ { \prime } ) ] \bigr ] .
$$

Here $x ^ { \prime }$ denotes the next state. The optimal valuefunction is

(167)

$$
\begin{array} { l } { \displaystyle V ^ { * } ( \boldsymbol { x } ) = \underset { \pi } { \mathrm { m a x } } V ^ { \pi } ( \boldsymbol { x } ) } \\ { \displaystyle \phantom { \mathrm { m a x } } = \underset { \pi } { \mathrm { m a x } } \mathbb { E } _ { P ^ { \pi } } \left[ \sum _ { j = n } ^ { N } r ( x _ { j } , u _ { j } ) \Bigg | x _ { n } = \boldsymbol { x } \right] , } \end{array}\tag{168}
$$

and measures the maximal cumulative expected reward achievable by a policy from a particular starting point. Like the value function in the continuous stochastic control setting, this discretized optimal value function similarly follows a recursive relationship:

![](images/58a4a6117434385f1d8cb9b66f4704a003c1f2063ee802ef7da6d697ced2f1e1.jpg)

(b) Discounted reward, ° = 0:9  
![](images/4caf1c794d588804ad1b262a0b77f9d20e8bd4604f650c7a94082660d36c1f1f.jpg)  
FIG. 9. The explore-exploit problem in reinforcement learning. The robot wishes to reach a goal, and is given a reward of 1 at the goal (star), a reward of 0 in empty cells, and a slightly negative reward −0.01 in hazard cells (electricity). Furthermore, at any point in time, the robot might break down with a large independent probability 10%. Thus, if the robot wants to maximize its expected reward, it should heavily prioritize shorter routes to the goal. Assume that the robot’s initial state is fixed to the top left of the warehouse. At initialization, the robot’s control is random and chooses these two actions with equal probability. From this point, the agent can choose one of two controls: go right or go down. These two controls lead to two paths towards the goal, a suboptimal path that is long and circuitous with ample opportunity to break down but with no penalties, and an optimal path that routes directly to the goal with a small incurred penalty (e.g. an activation energy barrier).

$$
V ^ { * } ( x ) = \operatorname* { m a x } _ { u } \left[ r ( x , u ) + \mathbb { E } _ { x ^ { \prime } \sim k ( \cdot \vert x , u ) } [ V ^ { * } ( x ^ { \prime } ) ] \right] .\tag{169}
$$

This equation is known as the Bellman optimality equation, the discrete-time analogue of the Hamilton-Jacobi-Bellman equation. The Q function critiques the “goodness” of a particular policy π for some state and control:

$$
Q ^ { \pi } ( x , u ) = \mathbb { E } _ { P ^ { \pi } } \left[ \sum _ { j = n } ^ { N } r ( x _ { j } , u _ { j } ) \Bigg | x _ { n } = x , u _ { n } = u \right] .\tag{170}
$$

In other words, the Q function measures the cumulative expected reward of a policy when fixing the control at step n to $u .$ For $n < N$ , the Q function satisfies the Bellman equation for policy π:

$$
\begin{array} { r l } & { Q ^ { \pi } ( x , u ) = r ( x , u ) } \\ & { \qquad + \operatorname { \mathbb { E } } _ { x ^ { \prime } \sim k ( \cdot \vert x , u ) } \left[ \operatorname { \mathbb { E } } _ { u ^ { \prime } \sim \pi ( \cdot \vert x ^ { \prime } ) } [ Q ^ { \pi } ( x ^ { \prime } , u ^ { \prime } ) ] \right] . } \end{array}\tag{171}
$$

Here $x ^ { \prime }$ and $u ^ { \prime }$ denote the next state and action, respectively. The Q function also shows how to improve the policy π by scoring the values of all actions u. The simplest scheme for policy improvement is to choose, at each state $x ,$ an action that maximizes $Q ^ { \pi } ( x , u )$ . With exact Q values, the resulting policy has expected cumulative reward at least as high as that of the original policy. However, clearly, this greedy scheme can lead to failed exploration if policy improvement is performed using an estimate of the true Q function from prior experience. Later in this section, we discuss alternative policy formulations that naturally incentivize exploration.

The optimal Qfunction definition directly follows:

$$
\begin{array} { l } { { \displaystyle Q ^ { * } ( x , u ) = \operatorname* { m a x } _ { \pi } Q ^ { \pi } ( x , u ) } } \\ { { \displaystyle = \operatorname* { m a x } _ { \pi } \mathbb { E } _ { P ^ { \pi } } \left[ \sum _ { j = n } ^ { N } r ( x _ { j } , u _ { j } ) \Bigg | x _ { n } = x , u _ { n } = u \right] . } } \end{array}\tag{172}
$$

Just like the previously introduced objects, the optimal Q function also satisfies a Bellman optimality equation:

$$
Q ^ { * } ( x , u ) = r ( x , u ) + \mathbb { E } _ { x ^ { \prime } \sim k ( \cdot \vert x , u ) } \Big [ \operatorname* { m a x } _ { u ^ { \prime } } Q ^ { * } ( x ^ { \prime } , u ^ { \prime } ) \Big ] .\tag{173}
$$

The starting point of RL methods is based on these Bellman equations, which are the discretized analogues of the Hamilton-Jacobi-Bellman equation.

## 2. An overview ofonline RL algorithms

We explicitly consider the RL setting where an agent has access to the environment, and can actively choose transitions to gather new data. This problem setting is referred to as online RL. Key methods in this category include Soft Actor Critic (SAC) [98] and Twin-Delayed Deep Deterministic Policy Gradient (TD3) [102]. In this section, we focus on building up to the SAC method, which, in addition to being an established method for continuous control on which newer algorithms build [103], is also a Maximum Entropy RL (Max-Ent RL) method with close connections to maximum entropy inference principles discussed throughout this review. To get there, we start from basic, early precursors to SAC that do not require deep learning at all, then go through the practical considerations that ultimately culminate in SAC.

a. Tabular Q Learning. A starting point to solve the RL optimization problem is to utilize the theoretical tools from SOC. Recall from Sec. I A that recursion provides a PDE called the Hamilton-Jacobi-Bellman equation. This PDE is a functional of the environment dynamics and the true reward function, where the solution is the optimal value function. Then, one route to find optimal controls is to first find the optimal value function, now with the Bellman Equation instead of the HJB equation, and extract the corresponding optimal control. However, the RL problem setting makes this route difficult: without access to the environment dynamics or the reward, we cannot directly solve the Bellman equation for the true optimal value function. One possible solution is to, instead, utilize some of the intuition from sampling. In RL, the agent actively collects experience that it must use to find optimal controls. When the agent executes a control at timestep $n ,$ the agent observes a transition with reward $r _ { n }$ from state $x _ { n }$ to the next state $x _ { n + 1 }$ , which is a sample from the true environment dynamics and reward.

Recall the iterative connection between sampling and control developed in Sec. $\operatorname { I V C } { \mathrm { : } }$ sampled trajectories update the value estimates, which guide changes to the controls, which change the next set of sampled trajectories. One route to update the value estimates and policy is to directly average the realized cumulative rewards of the current policy, improve the policy using these estimates, and then evaluate the updated policy. Evaluation and improvement thus iterate and build on each other, because changing the policy changes the expected future rewards, which is referred to as policy iteration. In the most naive approach, we can roll out M independent trajectories each starting from state x with initial control u at time $t _ { n }$ , then follow policy π for all subsequent controls. We now switch to a discounted reward formulation, with $0 < \gamma < 1$ so that rewards received later count less. Averaging the realized cumulative rewards of these rollouts gives:

$$
\widehat { Q } ^ { \pi } ( x , u ) = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \sum _ { j = n } ^ { N } \gamma ^ { j - n } r ( x _ { j } ^ { ( i ) } , u _ { j } ^ { ( i ) } ) .\tag{174}
$$

Then, a simple way to find a good policy based on these value estimates is to take the argmax over these estimated Q values:

$$
u \gets \arg \operatorname* { m a x } _ { u } \widehat { Q } ^ { \pi } ( x , u ) .\tag{175}
$$

However, evaluating different actions from complete rollouts can have high variance. Another approach uses a running estimate of value to replace the remainder of a trajectory [104]. Instead of sampling and summing all realized rewards after a particular state, we can directly use a running value estimate for the future returns from the next state. This substitution can reduce sampling variance at the cost of potentially introducing error from utilizing an incorrect running value estimate. The observed immediate reward r and the remaining estimated value from the next state $x ^ { \prime }$ provide a new prediction for the current Q value, referred to as the target y:

$$
\begin{array} { c } { { y = r + \gamma \operatorname* { m a x } _ { u ^ { \prime } } \widehat { Q } ^ { * } ( x ^ { \prime } , u ^ { \prime } ) , } } \\ { { \mathrm { f o r ~ o b s e r v a t i o n ~ } ( x , u , r , x ^ { \prime } ) . } } \end{array}\tag{176}
$$

We can use these targets to update the current Q estimate, using a learning rate of $\xi \colon$

$$
\widehat { Q } ^ { * } ( x , u ) \gets \widehat { Q } ^ { * } ( x , u ) + \xi \left[ y - \widehat { Q } ^ { * } ( x , u ) \right] .\tag{177}
$$

Repeatedly updating the current Q estimate toward these target predictions converges to the optimal Q function under the assumption that every state and action is sampled infinitely often [97, 105]. Conceptually, this update can be thought of as a “backup” of value from the future value estimate to the present estimate. Because the future value is “easier” to estimate (e.g. requires aggregating rewards over fewer timesteps) and is trivial at the endpoint of a trajectory, where it becomes exactly the expected reward, this backup moves value backwards in time from the easier subproblem to harder subproblems. This intuition is also why we can refer to Hamilton-Jacobi-Bellman as a continuous form of dynamic programming and the Bellman Equation as discretized dynamic programming. Practically, a table holds these Q values, and the Bellman Backup indexes into this table and updates the value entries. As such, this algorithm is known as tabular Q-learning.

b. Deep Q-Networks. The practical limitations of tabular Q Learning motivate the sequence of methods (Deep Q-Networks, Soft Q-Learning, and Soft Actor-Critic) we discuss below. First and foremost, storing and updating a separate Q value for every state and action in a table becomes infeasible in a large state space. Indeed, the tabular setup does not transfer to the continuous setting. Deep Q-networks (DQN) address this restriction by approximating the Q function with a neural network, allowing experience at one state to inform predictions at others [106]. We denote learned estimates of $Q ^ { \pi }$ and $Q ^ { * }$ by $Q _ { \theta } ^ { \pi }$ and $Q _ { \theta } ^ { * } ,$ , where θ denotes the network weights.

DQN fits a neural-network estimate $Q _ { \theta } ^ { * }$ of the optimal Q function using all of an agent’s prior observations – including transitions and observed rewards – aggregated within a replay buffer D. Rather than do tabular updates of the value functions following the Bellman equation, DQN directly minimizes the “error” between the value function and the recursed form of the value function over this replay buffer of all past experiences, which is amenable to neural network parameterization:

$$
\begin{array} { r l } & { L ( \theta ) = \cfrac { 1 } { 2 } \mathbb { E } _ { ( x , u , r , x ^ { \prime } ) \sim \mathcal { D } } \bigg [ } \\ & { \qquad \Big ( Q _ { \theta } ^ { * } ( x , u ) \quad - ( r + \gamma \underset { u ^ { \prime } } { \operatorname* { m a x } } Q _ { \bar { \theta } } ^ { * } ( x ^ { \prime } , u ^ { \prime } ) ) \Big ) ^ { 2 } \Big ] , } \\ & { \theta  \theta - \xi \nabla _ { \theta } L ( \theta ) . } \end{array}
$$

The notation $\bar { \theta }$ means that gradients cannot flow through the parameters in the target. The reason for this “stop gradient” is to force value to flow backwards in time: the value at time t should not be updated with the rewards that occurred before time t. While tabular Q learning provably converges, DQN does not provably converge without additional assumptions due to function approximation error [106, 107].

## 3. MaxEnt RL & Soft Q-learning

We now return to the undiscounted convention $\gamma ~ = ~ 1$ for the MaxEnt derivation and the two algorithms below. Throughout the remainder of the RL section, $Q ^ { \pi }$ and $V ^ { \pi }$ denote entropy-regularized (“soft”) values. The entropy coefficient α has the same units as the per-step reward; it is distinct from the convexity parameter used in Wasserstein geometry.

Learning values alone does not ensure that the agent explores, as is shown in Fig. 9. Indeed, the tabular Q-learning algorithm can only guarantee convergence if the policies visit every state infinitely often. In large state spaces, this condition can be impossible to satisfy even with random actions. Furthermore, always choosing the action with the largest estimated Q value can ignore routes with activation energy. In RL, a common heuristic for exploration is to take a random action a small, epsilon proportion of the time. This strategy is called epsilon-greedy exploration where, typically, the probability epsilon goes to 0 over the course of training [97, 106]. This exploration mechanism can be thought of as a heuristic version of simulated annealing. Later works employ Maximum-entropy RL (MaxEnt RL) to control exploration in a more principled way, incorporating the policy Shannon entropy $H [ \pi ( \cdot | x _ { n } ) ]$ in the objective itself:

$$
\mathcal { I } [ \pi ] \triangleq \mathbb { E } _ { P ^ { \pi } } \left[ \sum _ { n = 0 } ^ { N } \left[ r ( x _ { n } , u _ { n } ) + \alpha H [ \pi ( \cdot | x _ { n } ) ] \right] \right]\tag{178}
$$

where α is a temperature that controls the strength of the entropy term. This approach encourages the policy to spread out actions while exponentially weighting those with high expected reward. Here, we discuss Soft Q-learning, one of the early examples of MaxEnt RL that utilizes entropy as a principled mechanism to encourage exploration [108].

The crux of MaxEnt RL is the idea that the policy should induce the maximum-entropy path measure, where the energy of the measure is the expected cumulative reward. Crucially, MaxEnt RL has close connections to control as inference principles (Sec. I C). In fact, one can derive the soft Q-learning objective to learn optimal policies directly from a maximumentropy inference problem over path measures [20]. We lay out this derivation here, connecting MaxEnt RL to the approaches and perspectives discussed previously in the review.

The inference-as-control problem begins with a definition of a path measure induced by a particular control policy. If this path measure exhibits mean observables (for example, the Schrodinger Bridge problem), and these moments are all¨ that is known about the realized trajectories, Sanov’s theorem states that the most likely path measure is the maximum entropy distribution that satisfies these moments. Each of these observables corresponds to a Lagrange Multiplier r that enforces this constraint, and the resulting max-ent path measure is:

$$
P ^ { u ^ { * } } [ x ( \cdot ) ] \propto P ^ { 0 } [ x ( \cdot ) ] \exp { \left( \sum _ { m } r _ { m } a _ { m } [ x ( \cdot ) ] \right) } ,\tag{179}
$$

Here $a _ { m } [ x ( \cdot ) ]$ is the m-th observable measured along a trajectory. The Lagrange multiplier $r _ { m }$ enforces a contraint on the averaged observable, as in Sec. I C. Thus, these approaches give controls u that give a max-ent realization of these observables.

In the control setting, we instead wish for the path measure to satisfy particular observations – for example, that the path measure realizes a high expected cumulative r for some fixed reward function. The rewards r no longer directly result from enforcing specific observables (e.g. termination within a particular set). Therefore, despite MaxEnt RL optimizing path measures that look identical to the control-as-inference approach, maximum entropy is a postulate of the MaxEnt RL approach – there is not a fundamental physical principle like Sanov’s theorem for fixed observables that underlies the Max-Ent RL formulation.

To derive the MaxEnt RL objective, we now express the continuous-time path measures above using the discrete-time MDP notation introduced in Sec. V A 1. As an example, the controlled stochastic dynamics of Sec. I A

$$
\dot { x } _ { t } = f _ { t } ( x _ { t } ) + u _ { t } ( x _ { t } ) + \sqrt { \varepsilon } \eta _ { t }
$$

discretized by an Euler–Maruyama step of duration $\Delta t$ result in the Gaussian transition kernel:

$$
\begin{array} { r l } & { k ( x ^ { \prime } | x , u ) } \\ & { = \mathcal { N } ( x ^ { \prime } ; \ z + [ f _ { t _ { n } } ( x ) + u ] \Delta t , \ \varepsilon \Delta t I ) . } \end{array}\tag{180}
$$

where $\mathcal { N }$ denotes the Gaussian density for the physical coordinates, where I is the identity matrix. Then, an agent executes controls and transitions according to this discretized transition kernel. When an agent visits the pair (x, u), the agent receives a reward $r ( x , u )$ Together, the kernel (or transition dynamics), the reward, and the state and action spaces define the finite-horizon Markov decision process (MDP), thus showing how the continuous control problem becomes the MDP. <sup>6</sup> The policy $\pi ( u _ { n } | x _ { n } )$ denotes the agent’s probability of choosing action $u _ { n }$ in state $x _ { n }$ at step n and plays the role of the control law $u _ { t } ( x )$ . Iteratively combining the transition kernel and the policy through time gives the discrete-time analog of the controlled path measure $\dot { P } ^ { u } [ x ( \cdot ) ]$ , which now induces a measure over discretized trajectories $\{ x _ { n } , u _ { n } \}$

$$
\begin{array} { l } { P ^ { \pi } [ x _ { 0 : N } , u _ { 0 : N } ] \ ~ } \\ { \displaystyle = p _ { 0 } ( x _ { 0 } ) \prod _ { n = 0 } ^ { N } \pi ( u _ { n } | x _ { n } ) \prod _ { n = 0 } ^ { N - 1 } k ( x _ { n + 1 } | x _ { n } , u _ { n } ) , } \end{array}\tag{181}
$$

and the analog of the reference measure $P ^ { 0 } [ x ( \cdot ) ]$ is the measure of the passive dynamics with uniform action distribution $\pi ^ { 0 } ( u _ { n } | x _ { n } ) { \overset { \textstyle - } { = } } c _ { 0 } { \mathrm { : } }$

$$
\begin{array} { l } { { \displaystyle P ^ { 0 } [ x _ { 0 : N } , u _ { 0 : N } ] } \ ~ } \\ { { \displaystyle = p _ { 0 } ( x _ { 0 } ) \prod _ { n = 0 } ^ { N } \pi ^ { 0 } ( u _ { n } | x _ { n } ) \prod _ { n = 0 } ^ { N - 1 } k ( x _ { n + 1 } | x _ { n } , u _ { n } ) . } } \end{array}\tag{182}
$$

Say that we wish to find controls that lead to large rewards. One way to do so is to construct a target measure that upweights high-reward trajectories, then regress the controlinduced path measure towards this target measure for the training objective. Following the maximum entropy postulate in the MaxEnt RL approach, one possible choice for this target measure is (a possibly unrealizable) maximum-entropy, reward-maximizing path measure with respect to a base measure $P ^ { 0 } [ x _ { 0 : N } , u _ { 0 : N } ] \mathrm { . }$

$$
\begin{array} { l } { { \displaystyle P ^ { * } [ x _ { 0 : N } , u _ { 0 : N } ] \stackrel { \Delta } { = } } } \\ { { \displaystyle \arg \operatorname* { m a x } _ { P } \left[ \mathbb { E } _ { P } \left[ \frac { 1 } { \alpha } \sum _ { n = 0 } ^ { N } r ( x _ { n } , u _ { n } ) \right] - \mathrm { D } _ { \mathrm { K L } } ( P \| P ^ { 0 } ) \right] } } \\ { { \displaystyle ~ = \frac { 1 } { Z } P ^ { 0 } [ x _ { 0 : N } , u _ { 0 : N } ] \exp \left( \frac { 1 } { \alpha } \sum _ { n = 0 } ^ { N } r ( x _ { n } , u _ { n } ) \right) . } } \end{array}\tag{183}
$$

Here the maximization over $P$ is over normalized path measures. $P ^ { * }$ is the reward-tilted target measure and α is a temperature that controls the strength of the tilt. Importantly, this ${ \bar { P } } ^ { * }$ is the theoretically optimal max-ent target distribution and need not be achievable by a particular policy. In practice, the rewards are only observed, as the agent does not have a priori access to the true reward function following standard RL assumptions. We seek a policy whose path measure approximates this target, where the limited policies can only determine the executed controls, not the dynamics of the system. The optimal policy for this inference objective therefore produces the closest realizable path measure, which is the minimizer of $\mathrm { D } _ { \mathrm { K L } } ( P ^ { \pi } \| P ^ { * } ) ^ { \gamma }$

$$
\pi ^ { * } \triangleq \arg \operatorname* { m i n } _ { \pi } \mathrm { D } _ { \mathrm { K L } } ( P ^ { \pi } \| P ^ { * } )\tag{184}
$$

Note that the path distribution $P ^ { \pi ^ { * } }$ is the closest realizable distribution and need not equal $P ^ { * }$ . Thus, we have derived an inference objective that should provide high-reward,

$$
\begin{array} { r l } & { \mathrm { D } _ { \mathrm { K L } } ( P ^ { \pi } \| P ^ { * } ) = \displaystyle \int _ { n = 0 } ^ { N } \mathrm { d } x _ { n } \mathrm { d } u _ { n } ~ P ^ { \pi } [ x _ { 0 : N } , u _ { 0 : N } ] \log \frac { p _ { 0 } ( x _ { 0 } ) \prod _ { n = 0 } ^ { N } \pi ( u _ { n } | x _ { n } ) \prod _ { n = 0 } ^ { N - 1 } k ( x _ { n + 1 } | x _ { n } , u _ { n } ) } { \frac { 1 } { Z } p _ { 0 } ( x _ { 0 } ) \prod _ { n = 0 } ^ { N } \left[ \pi ^ { 0 } ( u _ { n } | x _ { n } ) e ^ { r ( x _ { n } , u _ { n } ) / \alpha } \right] \prod _ { n = 0 } ^ { N - 1 } k ( x _ { n + 1 } | x _ { n } , u _ { n } ) } } \\ & { = \displaystyle \int _ { n = 0 } ^ { N } \mathrm { d } x _ { n } \mathrm { d } u _ { n } ~ P ^ { \pi } [ x _ { 0 : N } , u _ { 0 : N } ] \log \frac { 1 } { \frac { 1 } { Z } \prod _ { n = 0 } ^ { N } \left[ \pi ^ { 0 } ( u _ { n } | x _ { n } ) e ^ { r ( x _ { n } , u _ { n } ) / \alpha } \right] } } \\ & { = \mathbb { E } _ { P ^ { \pi } } \left[ \displaystyle \sum _ { n = 0 } ^ { N } \left( \log \frac { \pi ( u _ { n } | x _ { n } ) } { \pi ^ { 0 } ( u _ { n } | x _ { n } ) } - \frac { r ( x _ { n } , u _ { n } ) } { \alpha } \right) \right] + \log Z } \\ & { = - \frac { 1 } { \alpha } \mathbb { E } _ { P ^ { \pi } } \left[ \displaystyle \sum _ { n = 0 } ^ { N } \left[ r ( x _ { n } , u _ { n } ) + \alpha H [ \pi ( \cdot | x _ { n } ) ] \right] \right] + \log Z - ( N + 1 ) \log c _ { 0 } . } \end{array}\tag{185}
$$

FIG. 10. Eq. (185). As the reader can see, RL-notation is no joke.

$$
Q ^ { \pi } ( x , u ) = r ( x , u ) + \mathbb { E } _ { P ^ { \pi } } [ \sum _ { j = n + 1 } ^ { N } ( r ( x _ { j } , u _ { j } ) - \alpha \log \pi ( u _ { j } | x _ { j } ) ) | x _ { n } = x , u _ { n } = u ] .\tag{187}
$$

The soft Q and soft value functions are related by

$$
\begin{array} { r } { V ^ { \pi } ( \boldsymbol { x } ) = \mathbb { E } _ { u \sim \pi ( \cdot | \boldsymbol { x } ) } \Big [ Q ^ { \pi } ( \boldsymbol { x } , u ) - \alpha \log \pi ( u | \boldsymbol { x } ) \Big ] . } \end{array}\tag{188}
$$

Here $Q ^ { \pi }$ excludes the entropy contribution of the fixed current action, while $V ^ { \pi }$ includes the current entropy contribution. By construction, $Q ^ { \pi }$ satisfies the “soft” Bellman Eq. (189),

$$
\begin{array} { r } { Q ^ { \pi } ( x , u ) = r ( x , u ) + \mathbb { E } _ { x ^ { \prime } \sim k ( \cdot \vert x , u ) } \Bigg [ \mathbb { E } _ { u ^ { \prime } \sim \pi ( \cdot \vert x ^ { \prime } ) } \Big [ Q ^ { \pi } ( x ^ { \prime } , u ^ { \prime } ) - \alpha \log \pi ( u ^ { \prime } \vert x ^ { \prime } ) \Big ] \Bigg ] . } \end{array}\tag{189}
$$

with the boundary condition $Q ^ { \pi } ( x , u ) = r ( x , u )$ at the final step N. Then, the optimal policy maximizes the conditional expectation on the right-hand side of Eq. (189), the expected Q value plus the policy entropy, independently at the next states $x ^ { \prime } .$ Setting the variation with respect to $\pi ( \boldsymbol { \dot { u } } ^ { \prime } | \boldsymbol { x } ^ { \prime } )$ to zero gives a Boltzmann distribution over actions,<sup>8</sup>

$$
\pi ^ { * } ( u | x ) = \frac { e ^ { Q ^ { * } ( x , u ) / \alpha } } { \sum _ { v } e ^ { Q ^ { * } ( x , v ) / \alpha } } .\tag{190}
$$

Substituting $\pi ^ { * }$ into Eq. (189) then reveals an alternative form of the soft Bellman equation, assuming that the policy π is the true Boltzmann distribution:

$$
\begin{array} { l } { { \displaystyle Q ^ { * } ( x , u ) = r ( x , u ) } \ ~ } \\ { { \displaystyle \qquad + \mathbb { E } _ { x ^ { \prime } \sim k ( \cdot \vert x , u ) } \Bigg [ \alpha \log \sum _ { u ^ { \prime } } e ^ { Q ^ { * } ( x ^ { \prime } , u ^ { \prime } ) / \alpha } \Bigg ] . } } \end{array}\tag{191}
$$

Thus, if the policy $\pi ^ { * }$ is known exactly, fixing $( x , u )$ leaves the environment transition $k ( x ^ { \prime } | x , u )$ as the only remaining source of randomness in this backup.

Eq. (191) is the discrete counterpart of the max-ent HJB equation after the Cole–Hopf transformation, providing a consistency equation whose right-hand side is a conditional expectation under the path measure. In continuous time, we solved the consistency equation with explicit knowledge of the dynamics and the reward. In RL, neither is known, but every observed transition $( x , u , r , x ^ { \prime } )$ provides a one-sample Monte Carlo estimate of the right-hand side, giving an update rule:

$$
\begin{array} { r } { y = r + \displaystyle \alpha \log \sum _ { u ^ { \prime } } e ^ { \widehat { Q } ^ { * } ( x ^ { \prime } , u ^ { \prime } ) / \alpha } , } \\ { \widehat { Q } ^ { * } ( x , u ) \gets \widehat { Q } ^ { * } ( x , u ) + \xi \big [ y - \widehat { Q } ^ { * } ( x , u ) \big ] . } \end{array}\tag{192}
$$

Here, ${ \widehat { Q } } ^ { * }$ is the learned estimate of $Q ^ { * }$ and ξ is the learning rate. This update rule is the soft Q-learning algorithm (Algorithm 1), where the log-sum-exp is a softened maximum. As the temperature $\alpha  0$ , the softened maximum in the logsum-exp becomes a hard maximum, recovering standard tabular Q-learning [105].

Soft Q-learning [108] has formed the basis for many popular RL algorithms, in which a neural network $Q _ { \theta } ^ { * }$ approximates the soft Q function using the update rule in Eq. (192). Implementation details, such as the use of replay buffers and target networks, are important for getting these algorithms to

work in practice.

Algorithm 1 Soft Q-learning   
Initialize $Q _ { \theta } ^ { * }$ and replay buffer $\mathcal { D } ;$ fix temperature $\alpha > 0$ and   
learning rate $\xi$   
for each iteration do   
Collect experience: Run $\pi _ { \boldsymbol { \theta } } ( u | \boldsymbol { x } )$ ∝ $e ^ { Q _ { \theta } ^ { * } ( x , u ) / \alpha }$ , and store   
$( x , u , r , x ^ { \prime } )$ in D   
Sample past experience: $( x , u , r , x ^ { \prime } ) \sim \mathcal { D }$   
Q target: $y = r +$ α log $\begin{array} { r } { \sum _ { u ^ { \prime } } e ^ { Q _ { \theta } ^ { * } ( x ^ { \prime } , u ^ { \prime } ) / \alpha } } \end{array}$   
Update $\mathbf { Q } \colon \theta \gets \theta - \xi \nabla _ { \theta } \big ( \bar { Q } _ { \theta } ^ { * } ( x , u ) - y \big ) ^ { 2 }$ , holding y fixed   
end for

## 4. Soft actor-critic

Each update must still evaluate the log-sum-exp over every action, which is cheap for a few discrete actions. With continuous actions, the sum becomes an integral, and computing the Q-conditioned partition function $\begin{array} { r } { Z _ { Q } \overline { { ( x ) } } ~ = ~ \int \mathrm { d } u ^ { \mathbf { \lambda } } e ^ { Q ( x , u ) / \alpha } } \end{array}$ can become intractable and high-variance as the action dimension grows due to the curse of dimensionality. This curse makes it difficult to evaluate the normalized Boltzmann policy $\pi _ { Q } ( u | x ) = e ^ { Q ( x , u ) / \alpha } / Z _ { Q } ( x )$ defined by the current critic $Q ,$ or the soft value α log $Z _ { Q } ( x ^ { \prime } )$ on the right-hand side of the backup in Eq. (191).

This leads to Soft Actor-Critic (SAC), which learns a policy together with the value estimate [98]. The iteration in Algorithm 2 alternates between updating the critic’s estimate of value under the current policy and improving the policy using that estimate, with entropy included in both steps. The parameterized actor $\pi _ { \theta }$ aims to fit the true Boltzmann target $\pi _ { Q }$ defined by the current critic $Q .$ Simultaneously, the learned critic $Q _ { \theta } ^ { \pi }$ aims to fit the value induced by the current policy $\pi _ { \theta }$

The unknown reward function and transition dynamics are only available through samples. Each observed transition supplies $r$ and $x ^ { \prime } .$ . Thus, the integral in $Z _ { Q } ( x ^ { \prime } )$ can nevertheless remain intractable even when the current critic $Q$ is directly evaluable. To address this integration problem, consider the Boltzmann policy $\pi _ { Q }$ defined above. Suppose we could draw the next action $u ^ { \prime }$ from $\pi _ { Q } ( u ^ { \prime } | x ^ { \prime } )$ and evaluate its log density. $\mathbf { A }$ single draw would then replace the log-sum-exp, because substituting log $\pi _ { Q } ( u ^ { \prime } | x ^ { \prime } ) = \stackrel { \textstyle - } { Q } ( x ^ { \prime } , u ^ { \prime } ) / \stackrel { \textstyle - } { \alpha } - \log Z _ { Q } ( x ^ { \prime } )$ gives

$$
\begin{array} { r l } & { Q ( x ^ { \prime } , u ^ { \prime } ) - \alpha \log { \pi _ { Q } ( u ^ { \prime } | x ^ { \prime } ) } } \\ & { \ = Q ( x ^ { \prime } , u ^ { \prime } ) } \\ & { \quad - \alpha \biggl [ \cfrac { Q ( x ^ { \prime } , u ^ { \prime } ) } { \alpha } - \log { Z _ { Q } ( x ^ { \prime } ) } \biggr ] } \\ & { \ = \alpha \log { Z _ { Q } ( x ^ { \prime } ) } } \end{array}\tag{193}
$$

for every sampled $u ^ { \prime } .$ Thus, with an exact $\pi _ { Q } ( u ^ { \prime } | x ^ { \prime } )$ , this difference is independent of the sampled action $u ^ { \prime }$ and provides a zero-variance estimate of the soft value function. In practice, we do not have access to the true Boltzmann $\pi _ { Q } .$ , so the estimator is not variance-free. Soft actor-critic [98, 110] keeps the combination $Q ( x ^ { \prime } , u ^ { \prime } ) - \alpha \log \pi _ { \phi } ( u ^ { \prime } | x ^ { \prime } )$ as the estimate of the soft value. Evaluating the normalized density $\pi _ { Q }$ itself requires the same intractable partition function $Z _ { Q } ( x )$ . Soft actor-critic therefore draws $u ^ { \prime }$ from a second network, a policy $\pi _ { \phi } ( u | x )$ with a tractable form, trained to stand in for $\pi _ { Q }$ This substitution turns Eq. (191) into the practical target used in the critic update in Algorithm 2

$$
\begin{array} { r } { y = r + Q _ { \theta } ^ { \pi } ( x ^ { \prime } , u ^ { \prime } ) - \alpha \log \pi _ { \phi } ( u ^ { \prime } | x ^ { \prime } ) , \ } \\ { \mathrm { w h e r e ~ } u ^ { \prime } \sim \pi _ { \phi } ( \cdot | x ^ { \prime } ) . \ } \end{array}\tag{194}
$$

where, like Soft Q-learning, the learned Q function $Q _ { \theta } ^ { \pi }$ is directly regressed towards the target. Then, training $\pi _ { \phi }$ to maximize

$$
\operatorname* { m a x } _ { \phi } \mathbb { E } _ { u \sim \pi _ { \phi } ( \cdot | x ) } \left[ Q _ { \theta } ^ { \pi } ( x , u ) - \alpha \log \pi _ { \phi } ( u | x ) \right] .\tag{195}
$$

is equivalent to minimizing $\operatorname { D } _ { \mathrm { K L } } ( \pi _ { \phi } | | \pi _ { Q } )$ , since the actor objective equals $- \alpha \mathrm { D } _ { \mathrm { K L } } ( \pi _ { \phi } \| \pi _ { Q } ) + \alpha \log Z _ { Q } ( x _ { k } )$ at fixed $x _ { k } .$ Thus, for a fixed critic $Q .$ , the policy objective corresponds to projecting the learned policy $\pi _ { \phi }$ onto the Boltzmann policy $\pi _ { Q }$ . However, in practice, the critic estimate $Q _ { \theta } ^ { \pi }$ can be noisy. Then, targets built from a maximum, or a soft maximum, of the estimates inherit an upward maximization bias. Practical implementations reduce overestimation by training two Q networks and using the smaller of the two targets, adding back in some pessimism [98].

Algorithm 2 Soft Actor-Critic   
Initialize critic $Q _ { \theta } ^ { \pi }$ , actor $\pi _ { \phi } ,$ and replay buffer $\mathcal { D } ;$ fix temperature   
$\alpha > 0$   
for each iteration do   
Collect experience: Run $\pi _ { \phi } ,$ observe $( x , u , r , x ^ { \prime } )$ , and store   
in $\mathcal { D }$   
Sample past experience: $( x , u , r , x ^ { \prime } ) \sim \mathcal { D }$   
Sample $\bar { u ^ { \prime } } \sim \pi _ { \phi } ( \cdot | x ^ { \prime } )$   
Critic target: $y = r + Q _ { \theta } ^ { \pi } ( x ^ { \prime } , u ^ { \prime } ) - \alpha$ log $\pi _ { \phi } ( u ^ { \prime } | x ^ { \prime } )$   
Update critic: $\theta \gets \theta - \xi \nabla _ { \theta } \left( Q _ { \theta } ^ { \pi } ( x , u ) - y \right) ^ { 2 }$ , holding y fixed   
Update actor:   
$\phi  \phi + \xi \nabla _ { \phi } \mathbb { E } _ { v \sim \pi _ { \phi } ( \cdot | x ) } [$   
$Q _ { \theta } ^ { \pi } ( x , v ) - \alpha \log \pi _ { \phi } ( v | x ) ]$   
end for

Solving a fixed-point equation by stochastic approximation raises the question of convergence. When the algorithm stores one Q value per state-action pair, the update converges to the optimal Q function [105, 111], and similar guarantees exist for simple function classes such as linear MDPs [111, 112]. In continuous or high-dimensional settings, however, the algorithm cannot enumerate every state-action pair and must instead approximate $Q ^ { * }$ with a parametric function such as a neural network. Unfortunately, the convergence guarantee does not generally survive function approximation [107], and the failure is visible in practice. A standard benchmark is mountain car [113], an underpowered car on a sinusoidal hill that must accelerate back and forth to build enough energy to reach the top. Even in mountain car, learned RL solutions show a large gap from the known optimal control solution [114]. Thus, modern RL methods learn controls from samples alone and approximate intractable objects, at the cost of the exactness guarantees of the earlier control methods.

## B. Wasserstein gradient flows

In this section, we examine three classes of applied problems for which Wasserstein geometry and Wasserstein gradient flows provide elegant solutions. The first problem is the numerical integration of Fokker-Planck equations, in which the Wasserstein geometry yields a powerful integration algorithm known as the Jordan-Kinderlehrer-Otto (JKO) scheme. Second, we discuss modeling the dynamics of many interacting particles, particularly the regime known as the mean-field limit in which each particle effectively interacts with a continuous density. The collective evolution of the particle density often takes the form of Wasserstein gradient flow, and this perspective has yielded insights into the training of wide neural networks, in which the parameters can be thought of as particles. Finally, we discuss the general problem of variational inference (VI), in which a complex distribution is approximated by a simpler one that can be used for sampling. The problem of finding the optimal parameters can be viewed as an instance of Wasserstein gradient flow, and this perspective yields physical intuition and stable algorithms. As we will see, variational inference as discussed in this section forms an “equilibrium” methodology, whose “non-equilibrium” counterparts are powerful, contemporary algorithms discussed in Sec. V C on flows and diffusion.

## 1. The Jordan-Kinderlehrer-Otto (JKO) scheme

The JKO scheme is a canonical example of using Wasserstein geometry to tackle numerical challenges, resulting in algorithms that are often more efficient, stable, and interpretable. Consider the general problem of numerically solving Fokker-Planck equations (FPE) driven by a conservative flow and constant diffusivity, which take the form:

$$
\frac { \partial \rho _ { t } ( x ) } { \partial t } = \nabla \cdot [ \rho _ { t } ( x ) \nabla \Phi ( x ) ] + \nabla ^ { 2 } \rho _ { t } ( x )\tag{196}
$$

The unit diffusion coefficient corresponds to $\varepsilon = 2$ in the convention of Chapter I. It may be tempting to numerically solve Eq. (196) by an explicit finite step update:

$$
\begin{array} { r l } & { \rho ^ { ( k + 1 ) } ( x ) = \rho ^ { ( k ) } ( x ) + \Delta t \Big \{ \nabla \cdot \left[ \rho ^ { ( k ) } ( x ) \nabla \Phi ( x ) \right] } \\ & { \qquad + \nabla ^ { 2 } \rho ^ { ( k ) } ( x ) \Big \} } \end{array}\tag{197}
$$

where $\rho ^ { ( k ) } ( x ) = \rho _ { k \Delta t } ( x )$ is the numerical discretization and $\Delta t$ is the step size. While conceptually straightforward, this direct approach and those like it encounter certain challenges. First, under Eq. (197), $\rho ^ { ( k ) }$ is not guaranteed to be either positive or normalized, which is particularly problematic if $\bar { \rho } ^ { ( k ) }$ represents a probability distribution. Second, one must explicitly compute derivatives of $\rho ^ { ( k ) } ( x )$ , which is costly in high dimensions and can introduce instabilities if $\rho ^ { ( k ) } ( \bar { x } )$ naturally concentrates onto singular manifolds such as points or lines. Third, the algorithm only converges for sufficiently small $\Delta t .$ Fourth, the continuous-time FPE takes the form of a Wasserstein gradient flow $\boldsymbol { v } = - \nabla _ { \mathrm { W } } \mathcal { F } _ { \mathrm { ~ \scriptsize ~ 1 ~ } }$ , yet under Eq. (197), F need not monotonically decrease at finite $\Delta t .$

Jordan, Kinderlehrer, and Otto proposed an integration scheme that overcomes these shortcomings [45]. It has two ingredients. First, the algorithm is variational: it breaks the numerical integration into a sequence of optimization problems. For intuition, consider integrating an ODE ${ \dot { x } } = - \nabla \Phi ( x )$ . A variational approach takes the form

$$
x ^ { ( k + 1 ) } = \arg \operatorname* { m i n } _ { x } \left\{ \frac { | x - x ^ { ( k ) } | ^ { 2 } } { 2 \Delta t } + \Phi ( x ) \right\}\tag{198}
$$

$$
\approx \arg \operatorname* { m i n } _ { x } \Big \{ \frac { | x - x ^ { ( k ) } | ^ { 2 } } { 2 \Delta t } + \Phi ( x ^ { ( k ) } ) \Big . \Big .\tag{199}
$$

$$
= x ^ { ( k ) } - \Delta t \nabla \Phi ( x ^ { ( k ) } )\tag{200}
$$

which recovers the expected Euler update as $\Delta t  0$ . As another example, the diffusion equation $\begin{array} { r } { \partial _ { t } \rho ~ = ~ \nabla ^ { 2 } \rho . } \end{array}$ , i.e. Eq. (196) with $\Phi = 0 .$ , can readily be put in variational form:

$$
\rho ^ { ( k + 1 ) } = \arg \operatorname* { m i n } _ { \rho } \left( \frac { \| \rho - \rho ^ { ( k ) } \| _ { L ^ { 2 } } ^ { 2 } } { 2 \Delta t } + \int | \nabla \rho ( x ) | ^ { 2 } \mathrm { d } x \right)\tag{201}
$$

where $\textstyle \| f \| _ { L ^ { 2 } } ^ { 2 } = \int \mathrm { d } x | f ( x ) | ^ { 2 }$ is the squared $L ^ { 2 }$ norm of $f .$ While Eq. (201) is certainly variational, it still suffers some of the practical and conceptual limitations of Eq. (197). Practically, one still needs to compute derivatives of $\rho ,$ and Eq. (201) can only capture Eq. (196) with $\nabla \Phi = 0 , \mathbf { i . e . }$ ., there is no way to amend the action of Eq. (201) to account for the drift term $\nabla \cdot ( \rho \nabla \Phi )$ . Conceptually, there is no reason to think of $\rho$ as a probability distribution in Eq. (201): both the $L _ { 2 }$ norm and the functional in the second term make sense even when $\rho$ is non-normalized or negative. In this sense, Eq. (201) emphasizes a particular interpretation of the diffusion equation — it punishes gradients. But we also know that diffusion has a probabilistic interpretation: it promotes randomness, thereby increasing entropy. Is there a version of Eq. (201) that emphasizes this interpretation?

These shortcomings are addressed by the second ingredi ent of the JKO scheme. Namely, one replaces the $L _ { 2 }$ norm in Eq. (201) with one that is adapted to probability: the Wasserstein distance. In doing so, one also replaces the gradient penalty with a thermodynamically meaningful quantity: the free energy. The resulting update scheme takes the form

$$
\rho ^ { ( k + 1 ) } = \arg \operatorname* { m i n } _ { \rho } \left( \frac { W _ { 2 } ^ { 2 } ( \rho ^ { ( k ) } , \rho ) } { 2 \Delta t } + \mathcal { F } [ \rho ] \right)\tag{202}
$$

Eq. (202) has several advantages. First, it accommodates both drift and diffusion within a single variational principle, and it has a clear probabilistic interpretation: descend $\mathcal { F }$ as fast as possible while not changing $\rho$ too much in the Wasserstein sense. This change in conceptual approach also resolves the practical issues arising in Eq. (197): $\rho ^ { ( k ) }$ is guaranteed to remain a valid probability distribution for all $k ,$ the integration is stable, and ${ \mathcal { F } } [ \rho ^ { ( k ) } ]$ is guaranteed to decrease monotonically for all $\Delta t$ . Moreover, Eq. (202) has the advantage of being derivative-free — the gradients and divergences of Eqs. (197)–(201) are replaced by an optimal transport problem [115]. In particular, the JKO scheme can be brought to the Benamou-Brenier form [116] (see Sec. II B), which allows one to rigorously take the continuum $( \Delta t  0 )$ limit, as well as to introduce efficient optimization methods [115] such as the Sinkhorn algorithm or point-mass approximations. The clear geometric motivation behind Eq. (202) also results in better interpretability; for instance, one can write down explicit formulas for the implicit bias (i.e., effective modifications of F) introduced at finite ∆t [117].

As a practical matter, the JKO scheme tends to shine in the following contexts. In very high dimensions, explicit spatial derivatives or spectral methods become computationally expensive, whereas solving the optimal transport problem at each time step remains comparatively efficient using particlebased or other methods. Another context where JKO is helpful is when $\rho ( x )$ has finite or irregular support, which is difficult to accommodate with spectral or other grid-based methods but is naturally handled by optimal transport optimization. Finally, JKO handles pattern-forming PDEs well, particularly when mass tends to aggregate onto singular points, lines, or manifolds. In short, JKO offers a practical advantage in high dimensions, with irregular boundaries, or in massconcentrating situations.

## 2. The mean-field limit of interacting particles

Now we study how WGF helps us understand the dynamics of the interaction of many identical particles, particularly the so-called mean-field limit, in which the dynamics of the individual particles can be replaced by the dynamics of a coarsegrained density. This limit is of fundamental interest in mathematical physics $( \mathrm { e . g . }$ , how do continuum theories like fluid mechanics rigorously emerge from interacting particles?), and in practical applications, such as the training of wide neural networks, in which the neurons are viewed as interacting particles. In certain situations, the coarse-grained density obeys WGF, which is typically simpler to analyze than a full manybody system and offers geometric intuition that underlies convergence theorems. For example, this approach allows one to prove under what conditions training a wide two-layer NN converges to global minima of its loss.

The mathematical setting of interest is a collection of $N _ { \mathrm { p } }$ interacting particles with coordinates $x _ { t } ^ { ( 1 ) } , \ldots , x _ { t } ^ { ( N _ { \mathrm { p } } ) }$ , obeying the following Langevin dynamics:

$$
\begin{array} { r l } & { \mathrm { d } x _ { t } ^ { ( i ) } = - \nabla _ { i } \Phi ^ { ( N _ { \mathrm { p } } ) } ( x _ { t } ^ { ( 1 ) } , \dots , x _ { t } ^ { ( N _ { \mathrm { p } } ) } ) \mathrm { d } t } \\ & { \qquad + \sqrt { \varepsilon } \mathrm { d } W _ { t } ^ { ( i ) } } \end{array}\tag{203}
$$

where $\mathrm { d } W _ { t } ^ { ( i ) }$ are independent Wiener processes and $\nabla _ { i }$ means that the gradient is evaluated with respect to argument i. To permit a mean-field limit, we assume that the particles are “identical” in the sense that their potential is only a function of their unlabeled density. That is, we can write $\Phi ^ { ( N _ { \mathrm { p } } ) } ( x _ { t } ^ { ( 1 ) } , \dots , x _ { t } ^ { ( N _ { \mathrm { p } } ) } ) = \mathcal { V } [ \widehat { \rho } _ { t } ]$ where

$$
\widehat { \rho } _ { t } ( x ) = \frac { 1 } { N _ { \mathrm { p } } } \sum _ { i = 1 } ^ { N _ { \mathrm { p } } } \delta ( x - x _ { t } ^ { ( i ) } )\tag{204}
$$

is the empirical probability measure, normalized to one. We reserve $\rho _ { t }$ for the continuous mean-field density and $N _ { \mathfrak { p } }$ is the particle count. A common class of such interactions consists of single- and two-particle potentials:

$$
\begin{array} { l } { \displaystyle \Phi ^ { ( N _ { \mathrm { p } } ) } ( x ^ { ( 1 ) } , \dots x ^ { ( N _ { \mathrm { p } } ) } ) = \sum _ { i } \Phi ( x ^ { ( i ) } ) + \frac { 1 } { 2 N _ { \mathrm { p } } } \sum _ { i , j } U ( x ^ { ( i ) } , x ^ { ( j ) } ) } \\ { = \displaystyle N _ { \mathrm { p } } \int \Phi ( x ) \widehat { \rho } ( x ) \mathrm { d } x \frac { N _ { \mathrm { p } } } { 2 } \int U ( x , y ) \widehat { \rho } ( x ) \widehat { \rho } ( y ) \mathrm { d } x \mathrm { d } y } \\ { \displaystyle \equiv \mathcal { V } [ \widehat { \rho } ] \qquad ( 2 0 5 ) } \end{array}
$$

where we assume $U ( x , x ) = 0$ to exclude self interactions. Now let’s assume that the initial positions of the particles are drawn independently from a continuous density $\rho _ { 0 }$ , namely $x _ { 0 } ^ { ( i ) } \overset { \mathrm { i i d } } { \sim } \rho _ { 0 }$ . The mean-field limit boils down to the following approximation: instead of tracking all the individual particles, one tracks the evolution of the continuous density. More precisely, one assumes that at a later time $t ,$ the particles are once again independently and identically distributed, but now according to an updated density $\rho _ { t } .$ . A representative particle $x _ { t } ,$ for which $\rho _ { t }$ is the distribution, evolves according to

$$
\mathrm { d } x _ { t } = - \nabla _ { \mathrm { W } } \mathcal { V } [ \rho _ { t } ] ( x _ { t } ) \mathrm { d } t + \sqrt { \varepsilon } \mathrm { d } W _ { t }\tag{206}
$$

where $\nabla _ { \mathrm { W } } \mathcal { V } = \nabla ( \delta \mathcal { V } / \delta \rho )$ is the Wasserstein gradient introduced in Sec. II C. Eq. (206) is often referred to as a McKean-Vlasov process [118], and is simply a rewriting of Eq. (203) in which the particle label has been dropped, and the potential is expressed in terms of the continuous density $\rho _ { t }$ instead of the empirical density $\widehat { \rho } _ { t }$ . Qualitatively, Eq. (206) mathematizes the intuition that as $N _ { \mathrm { p } }  \infty ,$ the typical particle $x _ { t }$ effectively interacts with an average continuous field $\rho _ { t }$ of particles. The corresponding Fokker-Planck equation for $\rho _ { t }$ is given by:

$$
\begin{array} { r l } & { \partial _ { t } \rho _ { t } = \nabla \cdot \left( \rho _ { t } \nabla _ { \mathrm { W } } \mathcal { F } [ \rho _ { t } ] \right) } \\ & { \qquad = \nabla \cdot \left( \rho _ { t } \nabla _ { \mathrm { W } } \mathcal { V } [ \rho _ { t } ] \right) + \frac { \varepsilon } { 2 } \nabla ^ { 2 } \rho _ { t } . } \end{array}\tag{207}
$$

where $\mathcal { F } = \mathcal { V } - \textstyle { \frac { \varepsilon } { 2 } } H$ and $\begin{array} { r } { H [ \rho ] = - \int \rho \log \rho } \end{array}$ dx is the entropy. Eq. (207) is sometimes referred to as a McKean-Vlasov equation<sup>9</sup>, and it precisely takes the form of WGF. Whether and for what values of $N _ { \mathrm { p } }$ and t the mean-field approximation is a good approximation is a field of mathematical research known as propagation ofchaos [119].

As a practical example, the WGF structure of Eq. (207) has been useful in analyzing the training dynamics of large machine learning (ML) models such as neural networks (NNs) [120–123]. Large ML models are often trained by defining a loss function and performing versions of gradient descent, e.g., Stochastic Gradient Descent (SGD) or Adaptive Moment Estimation (Adam), in their parameters. For a large but finite number of parameters, the loss landscape typically features many saddles, large manifolds of degenerate minima, and different minima-manifolds separated by shallow barriers. These features make the convergence of gradient descent difficult to analyze. However, when many of the parameters enter the loss identically, such as in a very wide NN, an $N _ { \mathrm { p } }  \infty$ limit can be taken, allowing one to analyze the evolution of the density $\rho _ { t }$ of the parameters over training time. The density $\rho _ { t }$ generally obeys WGF and the functional $\mathcal { F }$ that it minimizes is generally much simpler than the loss landscape governing the evolution of a finite number $N _ { \mathrm { p } }$ of parameters. This simplification allows convergence results to be obtained. It should be noted that there is an older collection of results, sometimes referred to as Universal Approximation Theorems [124–126], that ask which classes of functions can be approximated by a given ML architecture. The question at stake here is perhaps even more relevant: under what conditions are these functions actually reached via training?

To illustrate typical ingredients of such an approach, consider a two-layer neural network

$$
f ^ { N _ { \mathrm { p } } } ( x ; \Theta _ { t } ) = \frac { 1 } { N _ { \mathrm { p } } } \sum _ { i = 1 } ^ { N _ { \mathrm { p } } } w _ { t } ^ { ( i ) } \varphi ( x ; z _ { t } ^ { ( i ) } )\tag{208}
$$

In Eq. (208), $f ^ { { N _ { \mathrm { p } } } } ( x ; \Theta _ { t } )$ is the model to be trained. Here $\varphi ( x ; z _ { t } ^ { ( i ) } )$ is a nonlinear function, referred to as a “feature”, with weight $w _ { t } ^ { ( i ) }$ and internal parameters $\boldsymbol { z } _ { t } ^ { ( i ) }$ . All the parameters of $f ^ { N _ { \mathrm { p } } }$ may be collected in a single vector $\Theta _ { t } \ =$ $( \theta _ { t } ^ { ( 1 ) } , \dots , \theta _ { t } ^ { ( \bar { N } _ { \mathrm { p } } ) } )$ where $\theta _ { t } ^ { ( i ) } = ( w _ { t } ^ { ( i ) } , z _ { t } ^ { ( i ) } )$ . The parameters are given a time index t because they will evolve over training. The function $f ^ { { N _ { \mathrm { p } } } } ( x ; \Theta _ { t } )$ is trained to approximate a target function y(x) with x distributed according to a density $\nu ( x )$ . This is done by minimizing a loss function $L ( \Theta _ { t } )$ , a common choice for which is the mean-squared error (MSE):

$$
\begin{array} { l } { \displaystyle { L ( \Theta _ { t } ) = \frac { 1 } { 2 } \int \big | y ( x ) - f ^ { N _ { \mathrm { p } } } ( x ; \Theta _ { t } ) \big | ^ { 2 } \nu ( x ) \ \mathrm { d } x } } \\ { \displaystyle { \qquad = \frac { 1 } { 2 } \mathbb { E } _ { \nu } \bigg [ \Big | y - f _ { t } ^ { N _ { \mathrm { p } } } \Big | ^ { 2 } \bigg ] } } \end{array}\tag{209}
$$

where $f _ { t } ^ { N _ { \mathrm { p } } } ~ = ~ f ^ { N _ { \mathrm { p } } } ( \cdot ; \Theta _ { t } )$ for brevity. The loss is minimized by using some form of gradient descent, typically with stochasticity. For instance

$$
\mathrm { d } \theta _ { t } ^ { ( i ) } = - N _ { \mathrm { p } } \nabla _ { i } L \mathrm { d } t + \sqrt { \varepsilon } \mathrm { d } W _ { t } ^ { ( i ) }\tag{210}
$$

The factor $N _ { \mathrm { p } }$ compensates for the $1 / N _ { \mathrm { p } }$ in the network output: differentiating L with respect to one neuron’s parameters produces a factor $1 / N _ { \mathrm { p } }$ . Multiplying the learning rate by $N _ { \mathrm { p } }$ therefore keeps each neuron’s drift of order one as the network grows. In practice, more practical forms of gradient descent are used, for instance stochastic gradient descent in which stochasticity does not come from a Wiener process $\mathrm { d } W _ { t } ^ { ( i ) }$ , but from using finite sample-size Monte Carlo estimates of the loss. Regardless of the precise method, the loss $L$ is a complicated, nonconvex function of the parameters $\Theta _ { t } ,$ so proving (or understanding) anything about training may at first seem to be a daunting task.

To gain intuition, we can think of each neuron i as being a particle with coordinate $\theta _ { t } ^ { ( i ) }$ , and (noisy) gradient descent, e.g. Eq. (210), as introducing interactions between the particles. For the two-layer neural network, the key insight is that Eq. (208) can be written as a function of neuron density only:

$$
f ^ { { \cal N } _ { \mathrm { p } } } ( x ; \Theta _ { t } ) = \int a ( x ; \theta ) \widehat { \rho } _ { t } ( \theta ) \mathrm { d } \theta\tag{211}
$$

where $\theta ~ = ~ ( w , z ) , ~ a ( x ; \theta ) ~ = ~ w \varphi ( x ; z )$ , and $\begin{array} { r l } { \widehat { \rho } _ { t } ( \theta ) } & { { } = } \end{array}$ $\begin{array} { r } { \frac { 1 } { N _ { \mathrm { p } } } \sum _ { i = 1 } ^ { N _ { \mathrm { p } } } \delta ( \theta - \theta _ { t } ^ { ( i ) } ) } \end{array}$ is the empirical distribution of the parameters. Because $f ^ { N _ { \mathrm { p } } }$ only depends on the density of parameters, the loss $L ( \Theta _ { t } )$ can also be expressed in terms of the density alone. For example, the MSE becomes:

$$
\begin{array} { l } { { \displaystyle { \cal L } ( \Theta _ { t } ) = \mathcal { L } [ \widehat { \rho } _ { t } ] = \int \Phi ( \theta ) \widehat { \rho } _ { t } ( \theta ) \mathrm { d } \theta } } \\ { { \displaystyle ~ + \frac { 1 } { 2 } \int U ( \theta , \theta ^ { \prime } ) \widehat { \rho } _ { t } ( \theta ) \widehat { \rho } _ { t } ( \theta ^ { \prime } ) \mathrm { d } \theta \mathrm { d } \theta ^ { \prime } + C _ { y } } } \end{array}\tag{212}
$$

where $\begin{array} { r l r } { \Phi ( \theta ) } & { { } = } & { - \mathbb { E } _ { \nu } [ y ( x ) a ( x ; \theta ) ] } \end{array}$ and $\begin{array} { r l } { U ( \theta , \theta ^ { \prime } ) } & { { } = } \end{array}$ $\mathbb { E } _ { \nu } [ a ( x ; \theta ) \dot { a } ( x ; \theta ^ { \prime } ) \dot { ] }$ ] are single- and two-particle potentials, and $C _ { y } = \textstyle \frac { 1 } { 2 } \mathbb { E } _ { \nu } [ y ( x ) ^ { 2 } ]$ is independent of $\theta .$ The fact that the interactions only depend on the density ${ \widehat { \rho } } ( \theta )$ makes the convergence of training dynamics amenable to mean-field-based proof techniques.

The exact convergence statements proven in the literature [120–123] require some mathematical formalism to state precisely, and depend on the exact technical hypotheses placed on the features $\varphi ,$ the loss $L ,$ and the training algorithm. There are, however, some shared ingredients: one typically assumes that the network is “initialized” by drawing initial parameters, $\theta _ { 0 } ^ { ( i ) }$ , independently from a well-behaved distribution $\rho _ { 0 } .$ , typically obeying some precise regularity conditions. One is then interested in the evolution of the coarsegrained distribution $\rho _ { t }$ , which obeys a McKean-Vlasov equation, whose precise form depends on the exact architecture, loss, or version of gradient descent under consideration. Typically, the McKean-Vlasov equation can be interpreted in terms of WGF of a functional $\mathcal { F } , \bar { \partial _ { t } } \rho _ { t } = \nabla \cdot ( \rho _ { t } \nabla _ { \mathrm { W } } \bar { \mathcal { F } } [ \rho _ { t } ] )$ , where $\mathcal { F }$ consists of the loss $\mathcal { L }$ plus entropic terms arising from stochasticity. The dynamical problem of training the neural network is then mathematically transformed into studying the comparatively simple geometric and topological properties of $\mathcal { F }$ While the functional $\mathcal { F }$ is not necessarily geodesically convex in the Wasserstein sense, it often permits enough structure to make mathematical statements about convergence to global minima for $\rho _ { 0 }$ obeying suitable conditions. The conditions placed on $\rho _ { 0 }$ for convergence are of some practical use, because they can be interpreted as instructions on reasonable ways to initialize the NN for training. Intuitively, WGF is the right mathematical framework for analyzing these problems because of the underlying physical picture: the parameters $\theta _ { t } ^ { ( i ) }$ are particles, but these particles cannot jump, or be created or destroyed. Therefore, continuous motion of a density, i.e. WGF, naturally emerges.

To give a sense of the results proven, Ref. [122] considers a quadratic loss and training via either stochastic gradient descent or noisy stochastic gradient descent; for the latter case, they prove that any absolutely continuous $\rho _ { t }$ converges to a unique global minimum of $\mathcal { F }$ that controllably approximates the MSE. They also prove that $f _ { t } ^ { N _ { \mathrm { p } } }$ converges to the $N _ { \mathrm { p } }  \infty$ limit with bounds on the rate of convergence as a function of $N _ { \mathrm { p } }$ and training time t. Ref. [121] considers models of the form of Eq. (208) with technical hypotheses on the forms of the nonlinearity and data distribution, and identifies convergence to a global minimum, and studies the size of fluctuations for finite $N _ { \mathrm { p } }$ and t. Ref. [120] provides convergence proofs for exact gradient descent for a class of architectures and loss functions defined by the property that the functional $\mathcal { F }$ they give rise to obeys certain homogeneity conditions.

A couple of additional general remarks are in order. First, the mean-field approach based on a single density of neurons works for a two-layer neural network, but deeper networks require additional information about the connections between layers. Mean-field methods have been extended to these settings, with different assumptions and conclusions [127, 128]. Second, there is a common source of confusion: one observes that $\mathcal { L }$ in Eq. (212) is manifestly convex in the parameter density $\rho ,$ i.e. $\mathcal { L } [ \lambda \rho ^ { \prime } + ( 1 - \lambda ) \rho ] \leq \lambda \mathcal { L } [ \rho ^ { \prime } ] + ( 1 - \lambda ) \mathcal { L } [ \rho ]$ for any two densities $\rho , \rho ^ { \prime }$ and $\lambda \in [ 0 , 1 ]$ . While true, this mixture-interpolation notion of convexity is not the useful one for training based on gradient descent. Under gradient descent, the parameters flow along Wasserstein gradients, not weighted mixtures, and they certainly do not teleport (as is sometimes required for interpolation). Therefore, convexity with respect to Wasserstein geodesics is the useful mathematical framework to use, and one cannot directly conclude that $\rho _ { t }$ converges to a unique global minimum under training due to interpolation convexity. Finally, one might wonder: why not freeze the internal parameters $\boldsymbol { z } _ { t } ^ { ( i ) }$ and only optimize the feature weights $w _ { t } ^ { ( i ) } \mathbf { \bar { s } } \mathbf { ? }$ After all, the MSE loss $L ( \Theta _ { t } )$ is explicitly convex in $w _ { t } ^ { ( i ) }$ . This approach, often referred to as kernel regression, is certainly feasible, but it suffers from two drawbacks. First, a much larger $N _ { \mathrm { p } }$ is required for kernel regression to approximate the $N _ { \mathrm { p } } = \mathrm { \dot { \infty } }$ global minimum [120]. Second, kernel regression exhibits a different implicit bias: when both the $w \mathbf { \bar { s } }$ and the $z ' s$ are dynamic, it is observed that the network learns comparatively sparse distributions for the $w _ { t } ^ { \left( i \right) } \mathbf { \bar { s } } ,$ , and the selected features $\varphi ( x ; z _ { t } ^ { ( i ) } )$ have been updated via $ { \boldsymbol { z } } _ { t } ^ { ( i ) }$ , and thus tend to be informative about the learned properties of the data. When only the $z _ { t } ^ { ( i ) } \mathrm { \bar { s } }$ are frozen, the learned distribution over the $w ^ { \prime } \mathbf { s }$ tends to be broader and, by construction, no individual feature $\varphi ( x ; z _ { t } ^ { ( i ) } )$ reveals properties about the data. Whether ingredients from the mean-field approach can be extended to other architectures remains an open question.

## 3. Variational inference

Our final example of applications of WGF is to variational inference (VI) [129]. VI addresses the following challenge: one is confronted with a probability density $p _ { 1 } = \widetilde { p } _ { 1 } / Z _ { 1 }$ that is difficult to sample from and whose unnormalized form $\widetilde { p } _ { 1 }$ may be all that is available; however, we assume that we have access to a family of distributions $p _ { \theta }$ defined by a finite number of parameters $\theta ,$ and that the $p _ { \theta }$ are easy to normalize and sample from, e.g. the family of Gaussians. Our goal is to find the optimal parameters $\theta ^ { * }$ such that $p _ { \theta ^ { \ast } }$ best approximates $p _ { 1 }$ . This arises, for example, in Bayesian settings, in which one has a prior $p ^ { 0 } ( x )$ and observations $A ,$ and one would like to sample from the posterior $p _ { 1 } ( x ) \ =$ $p ( x | A ) = p ( A | x ) p ^ { 0 } ( x ) / p ^ { 0 } ( A )$ , where the normalization constant $\begin{array} { r } { p ^ { 0 } ( A ) = \int p ( A | x ) p ^ { 0 } ( x ) } \end{array}$ dx is often intractable. It is also thematically related to variational methods in quantum mechanics or statistical physics, in which the ground state of a model may be approximated by an ansatz featuring only a few parameters.

For the connection to WGF, we are interested in the setting where “best approximation” is defined as minimizing the Kullback-Leibler divergence

$$
\theta ^ { * } = \arg \operatorname* { m i n } _ { \theta } { \mathrm { D } _ { \mathrm { K L } } ( p _ { \theta } \| p _ { 1 } ) }\tag{213}
$$

As noted earlier in the review, the gradient of $\mathrm { D } _ { \mathrm { K L } } ( p _ { \boldsymbol { \theta } } \Vert p _ { 1 } )$ with respect to $\theta$ is independent of $Z _ { 1 } = \int \widetilde { p } _ { 1 }$ dx, which is often infeasible to compute. One tempting option to minimize Eq. (213) is to simply perform gradient descent in the parameters, i.e.

$$
\dot { \theta } = - \nabla _ { \theta } L ( \theta )\tag{214}
$$

with $L ( \theta ) = { \mathrm { D } } _ { \mathrm { K L } } ( p _ { \theta } \| p _ { 1 } )$ However, in practice, this naive parameter-centric approach is often problematic. The loss $\overset { \mathrm { ~ \tiny ~ - ~ } } { \boldsymbol { L } } ( \boldsymbol { \theta } )$ may be poorly conditioned and nonconvex. Moreover, certain families of probability distributions are only well defined on a subset of parameters; for instance, a Gaussian distribution is only well defined if its covariance matrix is positive-definite. Unfortunately, Eq. (214) may lead to violations of these constraints. Finally, if one were to reparameterize the same family of probability distributions, Eq. (214)

would yield a different trajectory through probability space— so Eq. (214) lacks parameterization invariance.

Wasserstein geometry is useful in addressing these issues. For convenience, write $\dot { p } _ { 1 } ( x ) = e ^ { - \Phi ( x ) } / Z _ { 1 }$ , with dimensionless potential $\Phi = - \log \widetilde { p } _ { 1 }$ , and write

$$
\begin{array} { r }   { \mathcal { L } [ p _ { \theta } ] = \mathrm { D } _ { \mathrm { K L } } ( p _ { \theta } \| p _ { 1 } ) = \int \Phi ( x ) p _ { \theta } ( x ) \mathrm { d } x } \\ { \qquad + \displaystyle \int p _ { \theta } ( x ) \log p _ { \theta } ( x ) \mathrm { d } x + \log Z _ { 1 } . } \end{array}\tag{215}
$$

It seems that one approach to minimizing $\mathrm { D } _ { \mathrm { K L } } ( p _ { \boldsymbol { \theta } } \Vert p _ { 1 } )$ would be to perform Wasserstein gradient flow down the functional ${ \mathcal { L } } [ p _ { \theta } ]$ , namely update $p _ { \theta }$ according to $\nabla \cdot p _ { \boldsymbol { \theta } } \nabla _ { \mathrm { W } } \mathcal { L } [ p _ { \boldsymbol { \theta } } ]$ . Yet, there is a fundamental problem: there is no guarantee that one can find a trajectory of parameters $\theta _ { t }$ such that $\dot { \theta } \cdot \partial _ { \theta } p _ { \theta } \ =$ $\nabla \cdot \left( p _ { \theta } { \nabla _ { \mathrm { { W } } } } { \mathcal { L } } [ p _ { \theta } ] \right)$ . In other words, the naive WGF may induce a flow field that flows out of the subspace of probability distributions defined by the parameters $\theta .$

The resolution is to project the Wasserstein descent velocity onto the tangent space of the parametric family. Recall from Sec. II C that a density tangent is represented by a minimumenergy velocity $\boldsymbol { v } = - \nabla \boldsymbol { \chi } , \mathbf { s o } \partial _ { t } p _ { t } = - \nabla \cdot \left( p _ { t } \boldsymbol { v } _ { t } \right)$ . The inner product of two such velocities at density $p$ is

$$
\langle v , v ^ { \prime } \rangle _ { p } = \int p ( x ) v ( x ) \cdot v ^ { \prime } ( x ) \mathrm { d } x .\tag{216}
$$

Consider a parametric family $\mathcal { M } = \{ p _ { \theta } : \theta \in \Omega \}$ , where $\Omega$ is a parameter domain of dimension $d _ { \theta }$ . In local coordinates $\theta = ( \theta _ { 1 } , \ldots , \theta _ { d _ { \theta } } )$ , define velocities $\boldsymbol { v } _ { i } = - \nabla \chi _ { i }$ by $\partial _ { \theta _ { i } } p _ { \theta } =$ $- \nabla \cdot \left( p _ { \theta } \boldsymbol { v } _ { i } \right)$ . They induce the metric $g _ { i j } = \langle v _ { i } , v _ { j } \rangle _ { p _ { \theta } }$ . The projection of a velocity field v is

$$
\mathrm { p r o j } _ { \mathcal { M } } v = \sum _ { i , j } v _ { i } ( g ^ { - 1 } ) _ { i j } \langle v _ { j } , v \rangle _ { p _ { \theta } } .\tag{217}
$$

The projected descent velocity $\mathrm { i s } - \mathrm { p r o j } _ { \mathcal { M } } \nabla _ { \mathrm { W } } \mathcal { L } [ p _ { \theta } ]$ , and its density evolution is

$$
\begin{array} { r } { \partial _ { t } p _ { \theta _ { t } } = \nabla \cdot \left( p _ { \theta _ { t } } \mathrm { p r o j } _ { \mathcal { M } } \nabla \mathrm { w } \mathcal { L } [ p _ { \theta _ { t } } ] \right) . } \end{array}\tag{218}
$$

Writing $L ( \theta ) = { \mathcal { L } } [ p _ { \theta } ]$ , the same evolution in parameter coordinates is

$$
\dot { \theta } _ { i } = - \sum _ { j = 1 } ^ { d _ { \theta } } ( g ^ { - 1 } ) _ { i j } \partial _ { \theta _ { j } } L ( \theta ) .\tag{219}
$$

Comparing Eq. (219) to $\operatorname { E q } .$ (214), the metric induced by Wasserstein geometry essentially acts to redirect the gradients with respect to θ. Moreover, one can show that Eq. (219) is equivalent to a parametric version of the JKO style update introduced in Sec. V B 1

$$
\theta _ { t + \Delta t } = \arg \operatorname* { m i n } _ { \theta } \left\{ \frac { W _ { 2 } ^ { 2 } ( p _ { \theta _ { t } } , p _ { \theta } ) } { 2 \Delta t } + \mathcal { L } [ p _ { \theta } ] \right\}\tag{220}
$$

in the limit $\Delta t \to 0$

So what does all this formalism buy us? The minimization algorithm in Eq. (218), or equivalently Eq. (219), comes with elegant convergence guarantees. Recall from Sec. II C that the functional $\mathcal { L }$ can have a notion of convexity (namely α-geodesic convexity) completely analogous to convexity in finite-dimensional optimization, and that this notion of convexity ensures that WGF converges to its global minimum. Recall also from Sec. II C that M is a geodesically convex subset of probability distributions if it contains the entire Wasserstein geodesic between any two of its elements. One can show that whenever M is geodesically convex and $\mathcal { L }$ is α-geodesically convex, the projected WGF in Eq. (218) is guaranteed to reach its unique minimum on $\mathcal { M } ,$ and do so exponentially fast. Especially important in the context of VI is that the loss $\mathcal { L }$ in Eq. (215) is in fact α-geodesically convex whenever $p _ { 1 }$ is strongly log-concave, which means that it is written as $\hat { p _ { 1 } } \propto e ^ { - \Phi }$ for some α-convex Φ (see Sec. II C for the definition of α-convex). So in short, one can use the structure of Wasserstein geometry to turn the problematic optimization in Eq. (214) into a well-behaved one.

Let’s explore two pedagogical examples of parametric subspaces. First, take $\mathcal { M }$ to be the set of all non-degenerate Gaussians on $\mathbb { R } ^ { d } .$ , which goes by the name Bures-Wasserstein space BW(R<sup>d</sup>) [130]. Each distribution in this family can be represented by the mean $m \in \mathbb { R } ^ { d }$ and covariance matrix $\Sigma \in \bar { \mathbb { S } _ { + + } ^ { d } } ,$ where $\mathbb { S } _ { + + } ^ { d }$ is the set of all symmetric, positive-definite $d \times d$ matrices. To see that the optimization problem is nontrivial, one can check that $L ( m , \Sigma )$ defined by Eq. (215) is not necessarily convex in the parameters $( m , \Sigma )$ even when $\Phi ( x )$ is α-convex in x—for example, consider $\dot { \Phi } ( x ) = x ^ { 4 }$ in $d = 1$ dimension. Now let’s apply the machinery of WGF and see what the dynamical optimization equations for m and Σ look like. First, let’s check that $\mathrm { B W } ( \mathbb { R } ^ { \hat { d } } )$ is geodesically convex. One can show that the Wasserstein geodesic between any two Gaussians $p _ { 0 } = { \mathcal { N } } ( m _ { 0 } , \Sigma _ { 0 } )$ and $p _ { 1 } = { \mathcal { N } } ( m _ { 1 } , \Sigma _ { 1 } )$ is given by the transport map

$$
\begin{array} { r } { x _ { t } = ( 1 - t ) x _ { 0 } + t \left[ \Sigma _ { 0 } ^ { - 1 / 2 } \left( \Sigma _ { 0 } ^ { 1 / 2 } \Sigma _ { 1 } \Sigma _ { 0 } ^ { 1 / 2 } \right) ^ { 1 / 2 } \right. } \\ { \left. \Sigma _ { 0 } ^ { - 1 / 2 } ( x _ { 0 } - m _ { 0 } ) + m _ { 1 } \right] \qquad } \end{array}\tag{221}
$$

Notably, Eq. (221) is an affine transformation for all t. Recall that any affine transformation of a normal distribution is a normal distribution, so the $p _ { t } ~ \in ~ \mathrm { B W } ( \mathbb { R } ^ { d } )$ for all $t ,$ confirming that BW $( \mathbb { R } ^ { d } )$ is geodesically convex. Next, we would like to find an explicit expression for the projected Wasserstein gradient $\nabla _ { \mathrm { B W } } \mathcal { F } = \mathrm { p r o j } _ { \mathrm { B W } ( \mathbb { R } ^ { d } ) } \nabla _ { \mathrm { W } } \mathcal { F }$ on a functional ${ \mathcal F } .$ . Because all Gaussians are connected by affine maps, and all affine maps take Gaussians to Gaussians, it follows that the tangent space to BW(R<sup>d</sup>) corresponds to all affine maps. Using this fact and a bit of algebra, one finds that:

$$
( \nabla _ { \mathrm { B W } } \mathcal { F } [ p ] ) ( x ) = \left. \nabla ^ { 2 } \frac { \delta \mathcal { F } } { \delta p } \right. _ { p } ( x - m ) + \left. \nabla \frac { \delta \mathcal { F } } { \delta p } \right. _ { p } .\tag{222}
$$

Here $\nabla ^ { 2 }$ in a matrix expression denotes the Hessian; the Laplacian is its trace. The parameter dynamics are

$$
\dot { m } _ { t } = - \left. \nabla \frac { \delta \mathcal { F } } { \delta p } \right. _ { p _ { t } } ,\tag{223}
$$

$$
\dot { \Sigma } _ { t } = - \left. \nabla ^ { 2 } \frac { \delta \mathcal { F } } { \delta \boldsymbol { p } } \right. _ { \boldsymbol { p } _ { t } } \Sigma _ { t } - \Sigma _ { t } \left. \nabla ^ { 2 } \frac { \delta \mathcal { F } } { \delta \boldsymbol { p } } \right. _ { \boldsymbol { p } _ { t } } .\tag{224}
$$

For the special case of VI, we take $\mathcal { F } = \mathrm { D } _ { \mathrm { K L } } ( p \Vert p _ { 1 } )$ and the flow equations become

$$
\begin{array} { r } { \dot { m } _ { t } = - \langle \nabla \Phi \rangle _ { p _ { t } } , } \end{array}\tag{225}
$$

$$
\dot { \Sigma } _ { t } = - \langle \nabla ^ { 2 } \Phi \rangle _ { p _ { t } } \Sigma _ { t } - \Sigma _ { t } \langle \nabla ^ { 2 } \Phi \rangle _ { p _ { t } } + 2 I .\tag{226}
$$

Eqs. (225)–(226) can be implemented straightforwardly because the expectations are taken with respect to the Gaussian measure $p _ { t } = { \mathcal N } ( m _ { t } , \Sigma _ { t } )$ . In the naive approach, Eq. (214), of directly optimizing with respect to m and $\Sigma ,$ the equations for the mean $\dot { m } _ { t }$ are unchanged, while Eq. (226) for $\dot { \Sigma } _ { t }$ differs in its dependence on $\Sigma _ { t }$ . With the standard Frobenius inner product for covariance matrices, ordinary gradient descent gives:

$$
\dot { \Sigma } _ { t } = - \frac { 1 } { 2 } \langle \nabla ^ { 2 } \Phi \rangle _ { p _ { t } } + \frac { 1 } { 2 } \Sigma _ { t } ^ { - 1 } ,\tag{227}
$$

which can create situations in which Σ becomes singular (loses its positive definiteness) during optimization. The geometric perspective of WGF avoids this pathology, makes the convex nature of the problem more transparent, and leads to other generalizations and principled convergence guarantees [131, 132].

As a second, theoretically rich, example, suppose $\mathcal { M }$ is the set of all discrete measures comprising $N _ { \mathrm { p } }$ point masses: $\begin{array} { r } { \widehat { p _ { t } } ( x ) = \frac { 1 } { N _ { \mathrm { p } } } \sum _ { i = 1 } ^ { N _ { \mathrm { p } } } \delta ( x - x _ { t } ^ { ( i ) } ) } \end{array}$ [133, 134]. Here, the parameters $\boldsymbol { x } _ { t } ^ { ( i ) }$ can be thought of as point particles, whose positions need to be optimally situated to approximate the target distribution $p _ { 1 }$ . Upon inspection, this is exactly the setting of interacting particles considered in Sec. V B 2. Therefore, using the loss in Eq. (215), the particles update according to

$$
\mathrm { d } x _ { t } ^ { ( i ) } = - \nabla \Phi ( x _ { t } ^ { ( i ) } ) \mathrm { d } t + \sqrt { 2 } \mathrm { d } W _ { t } ^ { ( i ) }\tag{228}
$$

where the noise convention is $\varepsilon \ = \ 2 .$ as in Chapter IV. This is simply traditional MCMC sampling (see Chapter IV). However, there is a conceptually interesting point: Unlike Eqs. (225)–(226), or more generally, Eq. (219), the evolution in Eq. (228) is stochastic, not deterministic. This is ultimately connected to the fact that the discrete measure $\widehat { p } _ { t }$ is singular, so gradients of the entropic contribution to Eq. (215) are ambiguous. While stochastic updates are not inherently problematic, this has led to a conceptual question: are there purely deterministic equations for the $\boldsymbol { x } _ { t } ^ { ( i ) }$ that would nevertheless give rise to an equivalent approximation in the $N _ { \mathrm { p } }  \infty$ limit?

In the language of Sec. V B 2, is there a deterministic version of Eq. (228) that gives rise to the same mean-field limit in Eq. (207)? An intuitive attempt would be to replace the delta functions $\delta ( x - x _ { t } ^ { ( i ) } )$ with a smoother function $k _ { h } ( x - x _ { t } ^ { ( i ) } )$ of width $h ,$ which gives the evolution equations

$$
\begin{array} { l } { \displaystyle \dot { x } _ { t } ^ { ( i ) } = - \nabla \Phi ( { x } _ { t } ^ { ( i ) } ) } \\ { \displaystyle - \left. \nabla _ { x } \log \left[ \frac { 1 } { N _ { \mathrm { p } } } \sum _ { j } k _ { h } ( x - { x } _ { t } ^ { ( j ) } ) \right] \right. _ { x = x _ { t } ^ { ( i ) } } } \end{array}\tag{229}
$$

where the density gradient differentiates the evaluation argument x with particle centers held fixed, before setting $x =$ $\boldsymbol { x } _ { t } ^ { ( i ) }$ . The continuum limit $N _ { \mathrm { p } }  \infty$ yields

$$
\begin{array} { r } { \partial _ { t } p _ { t } = \nabla \cdot \left[ p _ { t } \nabla \big ( \Phi + \log ( k _ { h } \star p _ { t } ) \big ) \right] } \end{array}\tag{230}
$$

where $\begin{array} { r } { ( k _ { h } \star p _ { t } ) ( x ) \ = \ \int k _ { h } ( x \ - \ - \ - y ) p _ { t } ( y ) \mathrm { d } y } \end{array}$ denotes convolution. Notably, this approach is not entirely straightforward because $p _ { t }$ in Eq. (230) does not converge to $p _ { 1 }$ unless $h \ = \ 0 ,$ , which raises a subtle question of limits for $N _ { \mathrm { p } }  \infty$ and $h  0$ . This subtlety motivates an alternative, more recent, approach known as Stein variational gradient descent [133, 134]. The basic idea is to put the regularization in a different location:

$$
\partial _ { t } p _ { t } = \nabla \cdot \left[ p _ { t } K _ { h } ^ { p _ { t } } \left( \nabla \log \frac { p _ { t } } { p _ { 1 } } \right) \right]\tag{231}
$$

In Eq. (231), $\mathcal { K } _ { h } ^ { p }$ implements convolution weighted by $p .$ Namely, given $f ( \bar { x } )$ , the smoothed version is

$$
\begin{array} { l } { \displaystyle \mathcal { K } _ { h } ^ { p } ( f ) ( x ) = ( k _ { h } \star ( p f ) ) ( x ) } \\ { \displaystyle = \int f ( y ) p ( y ) k _ { h } ( x - y ) \mathrm { d } y } \end{array}\tag{232}
$$

Interestingly, the smoothing operator $\mathcal { K } _ { h } ^ { p }$ can also be interpreted as a projection of the Wasserstein gradient $\nabla _ { \mathrm { W } } { \mathcal F } =$ $\mathbf { \bar { V } } \log ( p _ { t } / p _ { 1 } )$ [134]; however, this projection is distinct from $\mathrm { p r o j } _ { \mathcal { M } }$ in Eq. (218); it instead projects the velocity onto the reproducing kernel Hilbert space associated with the kernel $k _ { h }$ , see e.g. [134] for more detail. The discretized version of Eq. (231) takes the form:

$$
\begin{array} { c } { \displaystyle \dot { x } _ { t } ^ { ( i ) } = - \frac { 1 } { N _ { \mathrm { p } } } \sum _ { j = 1 } ^ { N _ { \mathrm { p } } } \Bigl [ k _ { h } ( x _ { t } ^ { ( i ) } - x _ { t } ^ { ( j ) } ) \nabla \Phi ( x _ { t } ^ { ( j ) } ) } \\ { \displaystyle + \nabla k _ { h } ( x _ { t } ^ { ( i ) } - x _ { t } ^ { ( j ) } ) \Bigr ] } \end{array}\tag{233}
$$

where $\nabla \Phi ( x _ { t } ^ { ( j ) } )$ differentiates the potential at particle j, while $\nabla k _ { h } ( z )$ differentiates the kernel argument $z = x _ { t } ^ { ( i ) } - x _ { t } ^ { ( j ) }$ Notice that Eq. (233) is straightforward to implement, and Eq. (231) genuinely permits $p _ { 1 }$ as a stationary solution. Yet, questions surrounding its convergence remain a topic of investigation [135–139]. Nonetheless, Eq. (233) has a clear physical interpretation: the first term captures gradient descent, while the second is a particle repulsion encouraging exploration of the full potential.

In addition to these two examples, several other parametric spaces for VI have received attention. For example, the treatment of Bures-Wasserstein space above can be extended to Gaussian mixtures by considering measures over $\mathrm { B W } ( \mathbb { R } ^ { d } )$ itself [131, $1 4 0 , 1 4 1 ] ^ { 1 0 }$ . Another powerful approach is to consider a mean-field approximation (not to be confused with the mean-field limit of interacting particles) in which $\mathcal { M } =$ $\otimes _ { i = 1 } ^ { d } \mathcal { W } _ { 2 } ( \mathbb { R } ) ~ \subset ~ \mathcal { W } _ { 2 } ( \mathbb { R } ^ { d } )$ is the space of products of onedimensional distributions [43, 142, 143]. Across these examples, VI benefits from theoretical guarantees when $\Phi ( x )$ is α-convex. Much modern interest is in situations in which $\Phi ( x )$ is not convex, but may be multi-modal and feature lowdimensional structure. Building models that are effective for sampling in this more advanced context motivates the next section on flow and diffusion (Sec. V C).

## C. Generative modeling with flows and diffusion

In Chapter IV on Sampling, we explored the following problem: given a (potentially unnormalized) probability distribution $p ( x )$ , how to draw a set of N independent samples $\lbrace x _ { n } \rbrace _ { n = 1 } ^ { N } \rbrace$ In this section, we ask the inverse problem: given a set of samples $\{ x _ { n } \} _ { n = 1 } ^ { N }$ , how to infer the probability distribution $p ( x )$ and generate a new independent sample $x _ { N + 1 }$ from the estimated distribution? The former task is known as density estimation, and the latter task is generative modeling. Like many inverse problems, generative modeling is a somewhat ill-defined task: one aims to generate samples similar, but not identical to those from the training set (an issue referred to as “generalization versus memorization”). Generative modeling is an important problem: In many contexts, like protein structures [144] or natural images [145], training samples are abundant, but unlike in physics or chemistry, an explicit model for the data is not available. Over the last ten years, deep-learning-based generative models have proven astonishingly successful at producing samples from such complex, high-dimensional distributions. This section provides an overview of these new methods, their utility for scientific simulations, and their connection to the ideas around sampling, transport, and non-equilibrium physics discussed in Chapters I–IV. Note that we do not attempt a systematic literature review.

Like for sampling, the key difficulty in generative modeling is the high dimension d of the sample space. A typical example is image generation, where d ≈ 200, 000 (for

![](images/d5f6a411f12bb0577ec92337d515c6842181e79184705dafc6bcca7058eb25a8.jpg)  
FIG. 11. Not a cat.

256 × 256-pixel color images). Learning high-dimensional probability distributions is difficult, since the training samples only cover a vanishing fraction of high-dimensional space (the “curse of dimensionality”). Vice versa, a generative model needs to target this vanishing fraction to succeed. Put flippantly, randomly choosing pixel values has vanishing probability of producing an image of a cat (Fig. 11). In addition, complex probability distributions are typically multi-modal. As for sampling, modern designed models generate samples from a high-dimensional target distribution $p _ { 1 }$ (e.g., the space of images) by dynamically transforming a simple base distribution p (e.g., the standard Gaussian). We already encountered such transport problems in Chapter II on optimal transport. Like optimal transport, generative models are recipes for transforming probability distributions, now designed to make it easy to learn the transformation from training data. $D i f { \mathrm { - } }$ fusion models [145, 146] are perhaps the most well-known machine learning methods of this type. They learn a stochastic process that generates samples from a target distribution defined by a set of training samples. Diffusion models were originally inspired by the physical ideas of non-equilibrium thermodynamics and simulated annealing discussed in Chapter IV [146]. This section approaches generative modeling using the (effectively equivalent) frameworks of stochastic interpolation [147] and flow matching [148]. We believe that this choice will make the core idea behind diffusion models easier to understand and generalize.

## 1. Variational transform sampling

Fundamentally, this section is based on the method of transform sampling, which applies a map T to transform a sample $x ~ \sim ~ p _ { 0 }$ into $y \ = \ T ( x )$ . Throughout this section, we use $" y "$ to denote samples generated by a transform. The map $T$ can also be thought of as a “decoder”, and x as a ${ } ^ { 6 6 } \mathrm { h i d - }$ den variable” [149]. The probability distribution of y is called the pushforward $T _ { \ast } p _ { 0 }$ . By changing variables, the probability density of $T _ { \ast } p _ { 0 }$ is:

$$
( T _ { * } p _ { 0 } ) ( y ) = p _ { 0 } ( T ^ { - 1 } ( y ) ) \cdot | \operatorname* { d e t } \nabla _ { y } T ^ { - 1 } |\tag{234}
$$

For example, the linear map $y = M \cdot x$ transforms a standard Gaussian into one with covariance matrix $\Sigma = M M ^ { \mathsf { T } }$

A second example is provided by one-dimensional distributions, where the map $T$ is defined by the cumulative distribution functions of $p _ { 0 }$ , p<sub>1</sub> (Eq. (88)).

The algorithms in this section combine transform sampling with variational inference (VI, Sec. V B 3). VI approximates a target distribution $p _ { 1 }$ by a member q of a parametrized family. Here, the trial family is defined by a set of transformations $\widehat { T }$ via Eq. (234). VI is an intuitive match for modern ML techniques like expressive neural networks and largescale, gradient-based optimization. To bring these techniques to bear, one needs a loss function $\mathcal { L }$ to measure the difference between the estimated $q$ and the target distribution $p _ { 1 }$ Here, our goal is to build loss functions that can be efficiently approximated using training samples, without explicit knowledge of $p _ { 1 }$ . Then, the transformation $\widehat { T }$ can be learned directly from data by minimizing ${ \mathcal { L } } .$

## 2. Flow matching

The key idea of flow matching is to construct the transformation $\widehat { T }$ step-by-step from a velocity field vb, parametrized by a neural network [150]. Flow matching generates target samples $y _ { 1 }$ by integrating the ODE $\dot { y } _ { t } = \widehat { v } _ { t } ( y _ { t } )$ starting from a random initial condition $y _ { 0 } = x _ { 0 } \sim p _ { 0 }$ (say, an image of white noise). Flow matching defines a suitable loss function that allows learning the velocity v from training samples. In contrast to diffusion models, the sample generation process is deterministic; Sec. V C 4 will introduce stochasticity.

Both flow matching and diffusion models are based on mass transport. Consider a time-dependent<sup>11</sup> vector field $\widehat { v } _ { t } ( x )$ on $\mathbb { R } ^ { d }$ . Here and below, hats denote learned maps or fields. The marginal of the generated trajectories $y _ { t }$ is $q _ { t }$ . By convention, time t ranges from 0 to 1, so that $q _ { t = 0 } = p _ { 0 }$ is the base, and we want to adjust the generative process so that $q _ { 1 }$ matches the target distribution $p _ { 1 }$ (reader beware: the reverse convention is also often used in the literature). The flow map $\widehat { T } _ { t } ( x )$ transports particles along the integral curves of $\widehat { v } _ { t }$ , and is defined by solutions to the ODE

$$
\begin{array} { r } { \dot { y } _ { t } = \widehat { v } _ { t } ( y _ { t } ) , \quad y _ { t = 0 } = x _ { 0 } \sim p _ { 0 } , \quad \widehat { T } _ { t } ( x _ { 0 } ) : = y _ { t } . } \end{array}\tag{235}
$$

If the initial conditions are sampled $y _ { t = 0 } ~ \sim ~ p _ { 0 }$ , the timedependent distribution $q _ { t } = ( \widehat { T } _ { t } ) _ { * } p _ { 0 }$ is governed by conservation of mass [152]:

$$
\partial _ { t } q _ { t } ( x ) + \nabla \cdot ( q _ { t } ( x ) \cdot \widehat { v } _ { t } ( x ) ) = 0\tag{236}
$$

If a $q _ { t }$ fulfills Eq. (236), it is said to be generated by the flow vb. In fluid dynamics terms, Eq. (235) is the Lagrangian, and Eq. (236) the Eulerian formulation.

Note that many different velocity fields can generate the same $q _ { t }$ . Indeed, adding a divergence-free vector field, $\widehat { v } _ { t } \mapsto$ $\widehat { v } _ { t } + v _ { t } ^ { \prime } ,$ , with $\nabla \cdot ( q _ { t } v _ { t } ^ { \prime } ) = 0$ , leaves Eq. (236) invariant. In Sec. II B, we saw how optimal transport singles out one of these vector fields by minimizing a transport cost. On the other hand, not all time-dependent distributions can be generated by a flow. For example, direct interpolation between two probability densities, $p _ { t } ( x ) = ( 1 - t ) p _ { 0 } ( x ) + t p _ { 1 } ( x )$ with disjoint support requires “mass teleportation” and cannot be generated by any flow.

Flow-generated distributions are very convenient: given ${ \widehat { v } } ,$ one can generate samples of $q _ { 1 }$ by integrating Eq. (235). How can one identify a velocity field vb so that $q _ { 1 } ~ = ~ ( \widehat { T } _ { 1 } ) _ { * } p _ { 0 }$ matches the target distribution $p _ { 1 } \mathrm { 2 }$ A first idea is to directly optimize the transport map $\widehat { T } _ { 1 }$ , obtained by integrating Eq. (235). This approach ${ \mathrm { i s } } ,$ however, very inefficient: evaluating the objective requires simulating the entire trajectory. Instead, one uses a stochastic interpolant [147, 148], a stochastic process $x _ { t }$ with $x _ { 0 } \sim p _ { 0 }$ and $x _ { 1 } \ \sim \ p _ { 1 }$ . Recall that in generative modeling, the target distribution $p _ { 1 }$ is unknown. Instead, one has access to a set of training samples (for example, an image database). One constructs a stochastic interpolant from these training samples, for example:

$$
x _ { t } = ( 1 - t ) x _ { 0 } + t x _ { 1 } , \quad x _ { 0 } \sim p _ { 0 } , x _ { 1 } \sim p _ { 1 } .\tag{237}
$$

The endpoints $x _ { 0 } , x _ { 1 }$ are sampled independently<sup>12</sup>. For instance, $x _ { 1 }$ could be an image sampled from the database, and x<sub>0</sub> could be Gaussian noise, so that $x _ { t }$ is a sequence of increasingly noisy images (Fig. 13). In this sense, Eq. (237) can be thought of as a version of the Brownian bridge from Sec. I D 3, conditioned to terminate in the target sample $x _ { 1 }$ The interpolant Eq. (237) defines a time-dependent probability path $x _ { t } \sim p _ { t }$ from p<sub>0</sub> to $p _ { 1 }$ . The $p _ { t }$ play a role similar to the instantaneous equilibria in annealed importance sampling (Sec. IV B). It is important to distinguish the interpolant $x _ { t }$ from the generating process $y _ { t } ;$ even if initialized at the same starting point $x _ { 0 } ~ \sim ~ p _ { 0 }$ , the two processes lead to different trajectories.

Crucially, the probability path $p _ { t }$ defined by Eq. (237) can be generated by a flow field (in the sense of mass conservation, Eq. (236)). As we show shortly, the velocity v of the interpolant $p _ { t }$ is the average of the trajectories’ speed:

$$
v _ { t } ( x ) = \langle \dot { x } _ { t } | x _ { t } = x \rangle\tag{238}
$$

Here and below, $\langle \cdot | \cdot \rangle$ denotes the conditional expectation. The average is over the randomly sampled endpoints $x _ { 0 } , x _ { 1 }$ which determine $x _ { t } , \ \dot { x } _ { t }$ . To derive Eq. (238), integrate $\partial _ { t } p _ { t }$ against a test function $\varphi ( x )$ , and use the law of total expectation:

$$
\begin{array} { l } { \displaystyle \int \varphi ( \boldsymbol { x } ) \partial _ { t } p _ { t } ( \boldsymbol { x } ) d ^ { d } \boldsymbol { x } = \frac { d } { d t } \left. \varphi ( \boldsymbol { x } _ { t } ) \right. } \\ { \displaystyle = \langle \nabla \varphi \cdot \dot { \boldsymbol { x } } _ { t } \rangle = \langle \nabla \varphi \cdot \langle \dot { \boldsymbol { x } } _ { t } | \boldsymbol { x } _ { t } \rangle \rangle } \\ { \displaystyle = - \int \varphi ( \boldsymbol { x } ) \nabla \cdot ( p _ { t } \left. \dot { \boldsymbol { x } } _ { t } | \boldsymbol { x } _ { t } = \boldsymbol { x } \right. ) d ^ { d } \boldsymbol { x } } \end{array}\tag{239}
$$

Fig. 12 shows a toy example $( p _ { 1 }$ is a mixture of Gaussians), where the true flow field v can be computed analytically.

The stochastic interpolant defines the objective: match the estimated flow field vb to the true one $v .$ The resulting flow matching loss reads, using the law of total expectation:

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { \mathrm { F M } } [ \widehat { v } ] = \int _ { 0 } ^ { 1 } \Big \langle \| \widehat { v } _ { t } ( x _ { t } ) - v _ { t } ( x _ { t } ) \| ^ { 2 } \Big \rangle d t } \\ { \displaystyle \quad = \int _ { 0 } ^ { 1 } \Big \langle \| \widehat { v } _ { t } ( x _ { t } ) - \langle \dot { x } _ { t } | x _ { t } \rangle \| ^ { 2 } \Big \rangle d t } \\ { \displaystyle \quad = \int _ { 0 } ^ { 1 } d t \~ \Big \langle \| \widehat { v } _ { t } ( x _ { t } ) \| ^ { 2 } - 2 \widehat { v } _ { t } ( x _ { t } ) \cdot \dot { x } _ { t } \Big \rangle } \\ { \displaystyle \quad \quad + \ \mathrm { c o n s t . } = \langle \| \widehat { v } _ { t } ( x _ { t } ) - \dot { x } _ { t } \| ^ { 2 } \rangle + \mathrm { c o n s t . } } \end{array}\tag{240}
$$

The additive constant in Eq. (240) is independent of ${ \widehat { v } } ,$ and thus does not affect training. Eq. (240) has three important properties. First, the unique minimizer is the true flow field. Thus, at the minimum, $q _ { t }$ matches the interpolant distribution $p _ { t }$ , and $y _ { t = 1 }$ is a sample from $p _ { 1 }$ (note that $y _ { t }$ and $x _ { t }$ have the same marginals, but the trajectory laws can differ). Second, since $x _ { t } = ( 1 - t ) x _ { 0 } + t x _ { 1 }$ <sub>1</sub>, the expectation value is easily estimated using independent samples $x _ { 0 } \sim p _ { 0 } , x _ { 1 } \sim p _ { 1 }$ from the training set. Third, the loss is simulation-free: it does not require simulating the entire trajectory $y _ { t }$ . Thus, one can efficiently train the neural network $\widehat { v }$ to generate target samples by stochastic gradient descent on $\mathcal { L } _ { \mathrm { F M } }$ . In summary, the overall flow-matching workflow reads:

1. Design a stochastic interpolant $x _ { t }$ whose distribution $p _ { t }$ bridges the simple base distribution $p _ { 0 }$ and the target $p _ { 1 }$

2. Using interpolant samples generated from a training set, train a velocity field $\widehat { v }$ by minimizing $\lVert \widehat { v } _ { t } ( x _ { t } ) - \dot { x _ { t } } \rVert ^ { 2 }$

3. Generate samples by integrating the ODE $\dot { y } _ { t } = \widehat { v } _ { t } ( y _ { t } )$ , with $y _ { t = 0 } = x _ { 0 } \sim p _ { 0 }$

One can see that flow matching bears a close resemblance to dynamic optimal transport (OT). Both aim to transform a base into a target distribution via “local”, mass-conserving dynamics, Eq. (236). Like in optimal transport, in Eq. (237), samples move along straight lines between endpoints. However, the choice of the endpoints is precisely the opposite: OT couples the endpoints deterministically via the transport map T $, x _ { 1 } = T ( x _ { 0 } )$ , while in Eq. (237), the endpoints are independent. Therefore, OT and stochastic interpolation are not identical. Sampling from the OT process is computationally hard, since it requires finding a very specific flow field that additionally minimizes a transport cost. In flow matching, the stochastic interpolant is instead designed to be easy to sample. These samples serve as a device to find a flow field vb that takes $p _ { 0 }$ to $p _ { 1 }$

![](images/7872f7bc514e64f7a309a94043ca0c9c266b6d641a740b36afe6a0d4fd44bd3d.jpg)  
FIG. 12. (Left) Stochastic interpolation using Eq. (237) between $p _ { 0 } = \mathcal { N } ( 0 , I )$ , a standard Gaussian, and $\begin{array} { r } { p _ { 1 } = \sum _ { i } c _ { j } \mathcal { N } ( \mu _ { j } , \Sigma _ { j } ) } \end{array}$ a Gaussian mixture with weights $c _ { j }$ (in the 2d example shown, $c _ { 1 } = 0 . 4 , c _ { 2 } = 0 . 6 )$ . Since $x _ { t } = ( 1 - t ) x _ { 0 } + t x _ { 1 }$ is a sum of Gaussian mixture variables, the intermediate distributions are also Gaussian mixtures, $\begin{array} { r } { p _ { t } = \sum _ { i } c _ { j } \mathcal { N } \left( t \mu _ { j } , ( 1 - t ) ^ { 2 } I + t ^ { 2 } \Sigma _ { j } \right) } \end{array}$ . The plot shows contour lines of $p _ { t }$ for $t = 0 , 1 / 2 ,$ 1. (Right) The flow map can be calculated from the intermediate $\vec { p _ { t } \mathrm { : } }$ using Eq. (242). It transports samples from $p _ { 0 }$ to samples from $p _ { 1 }$ in unit time (black). Note that the integral curves are bent while the trajectories of the interpolant are straight lines.

Flow matching can be viewed not only as data-driven transport, but also as an iterative denoising process. This perspective sheds light on how flow matching is able to learn multi-modal distributions. Multi-modality was already a challenge in the context of sampling from a known distribution (Chapter IV). Since $x _ { 0 }$ is typically Gaussian white noise, the stochastic interpolant is a sequence of increasingly noisy images (Fig. 13). One can think of the velocity field as a denoiser, which estimates $x _ { t + d t }$ from the (infinitesimally) more noisy version $x _ { t }$ as $\langle x _ { t + d t } | x _ { t } \rangle \approx x _ { t } + v _ { t } ( x _ { t } ) d t$ . The flowmatching loss Eq. (240) measures the squared error of the denoiser. Intuitively, flow matching thus breaks down a difficult sampling task (generating an image) into a sequence of simpler tasks (incrementally denoising an image).

This step-wise approach is essential to handle multimodality. Indeed, suppose one were to add a large, finite amount of noise to a target sample and attempt to denoise it in one step. The denoiser $D _ { t }$ at interpolation time t has mean squared error $\langle \| D _ { t } ( x _ { t } ) - x _ { 1 } \| ^ { 2 } \rangle$ (smaller t corresponds to more noise). Because the noise removes details, several different reconstructions are compatible with the noisy sample. The optimal denoiser (with minimum least-squares error) will return the average of all reconstructions, $D _ { t } ^ { * } ( x ) = \langle x _ { 1 } | x _ { t } = x \rangle$ . If the distribution is multi-modal, this average is a blurred mix of different modes that does not resemble any sample of the target distribution. However, for an infinitesimal amount of noise, the reconstruction problem becomes uni-modal. This is the principle that flow matching exploits to learn multi-modal distributions.

## 3. The scorefunction

Let us take a closer look at the distributions $p _ { t }$ for a linear interpolant between a standard Gaussian and an arbitrary target. If $x _ { 0 }$ is standard Gaussian and independent of $x _ { 1 }$ , the conditional density is also Gaussian, $p _ { t } ( \cdot | x _ { 1 } ) = \mathcal { N } ( t x _ { 1 } , ( 1 -$ $t ) ^ { 2 } I )$ . Taking the gradient, one obtains $\nabla$ log $p _ { t } ( x _ { t } | x _ { 1 } ) \ =$ $- x _ { 0 } / ( 1 - t )$ . Averaging over endpoints using the law of total expectation gives [153]:

![](images/e0f79271d64a136ae5d4729b4d2ba76165cebc3c32480f0d3d18c5e5466ed78b.jpg)  
FIG. 13. A stochastic interpolant adds progressively larger amounts of uncorrelated Gaussian noise to an initial image.

$$
\begin{array} { r l } { \nabla \log p _ { t } ( x ) = \langle \nabla \log p _ { t } ( x | x _ { 1 } ) | x _ { t } = x \rangle } & { { } } \\ { = - \frac { \langle x _ { 0 } | x _ { t } = x \rangle } { 1 - t } . } \end{array}\tag{241}
$$

The quantity $\nabla$ log $p _ { t }$ is called the score. We already encountered the score in Sec. I D 2 in the context of optimal control, where it served as a control force that drives a particle towards a target. Here, the score plays a similar role. Eq. (241) shows that the score can separate the noise $x _ { 0 }$ from the current $x _ { t } ,$ and thus allows reconstructing the target $x _ { 1 }$ . After a little algebra:

$$
\begin{array} { r l } & { v _ { t } ( x ) = \langle \dot { x } _ { t } | x _ { t } = x \rangle = \frac { x - \langle x _ { 0 } | x _ { t } = x \rangle } { t } } \\ & { \qquad = \frac { x + ( 1 - t ) \nabla \log p _ { t } ( x ) } { t } , \qquad 0 < t < 1 . } \end{array}\tag{242}
$$

Flow matching with an initially Gaussian distribution is therefore closely related to score matching [154], a method that uses the score to fit statistical models. Flow and diffusion models can be viewed as a combination of simulated annealing (Sec. IV B) and score matching [155]. Eq. (242) demonstrates that the velocity field is a potential gradient: log $p _ { t }$ acts as a time-dependent energy landscape that guides a sample from the initial to the target distribution (see the example in Fig. 12). Note that the relation between score and flow only holds for linear interpolation where one endpoint is Gaussian; in general, there is no simple relation between $p _ { t }$ and v.

## 4. From flow matching to diffusion

In flow matching, sampling is a deterministic process, defined by the ODE Eq. (235). Diffusion models instead use a stochastic sampling process, which, however, has the same marginals. To see how, add ∇ log $p _ { t }$ to and subtract it from the mass conservation equation for the interpolant (target) density p<sub>t</sub>:

$$
\begin{array} { r l } & { \partial _ { t } p _ { t } = - \nabla \cdot \Big ( \big ( \boldsymbol { v } _ { t } + \frac { \varepsilon } { 2 } \nabla \log q _ { t } - \frac { \varepsilon } { 2 } \nabla \log p _ { t } \big ) q _ { t } \Big ) } \\ & { \qquad = - \nabla \cdot \big ( \big ( \widehat { \boldsymbol { v } } _ { t } + \frac { \varepsilon } { 2 } \nabla \log p _ { t } \big ) p _ { t } \big ) + \frac { \varepsilon } { 2 } \nabla ^ { 2 } p _ { t } } \end{array}\tag{243}
$$

Eq. (236) is thus reformulated as a Fokker-Planck equation with diffusion, compensated by an additional drift. This seemingly tautological manipulation allows defining a new sampling process. The Fokker-Planck Eq. (243) gives a Langevin equation for trajectories:

$$
\begin{array} { l } { \dot { y } _ { t } = v _ { t } ( y _ { t } ) + \frac { \varepsilon } { 2 } \nabla \log p _ { t } ( y _ { t } ) + \sqrt { \varepsilon } \eta _ { t } } \\ { \quad = \frac { 1 } { t } \Big ( y _ { t } + \Big ( 1 + \frac { \varepsilon - 2 } { 2 } t \Big ) \nabla \log p _ { t } ( y _ { t } ) \Big ) + \sqrt { \varepsilon } \eta _ { t } . } \end{array}\tag{244}
$$

In the second step, we used the relation between score and velocity. Because the density obeys Eq. (243), the stochastic process Eq. (244) still generates samples from $p _ { 1 }$ . Effectively, diffusion models add an additional stochastic term to the interpolant Eq. (237), such that there is stochasticity at intermediate t even after fixing the endpoints. Stochastic sampling has practical advantages: it can improve stability so that even an imperfectly learned score produces reasonable samples [147].

With the above results, flow matching can be compared to the original formulation of denoising diffusion models [156]. Here, a forward diffusion process gradually adds noise to a target sample $x _ { 1 }$ . This forward process is the pendant of the stochastic interpolant and generates samples from which the score function is learned. Given the score function, target samples are generated by Eq. (244). In this context, Eq. (244) is called the reverse process, since it is the time-reversal of the forward diffusion process.

To see this, reparametrize time as s = − log t, so $s = 0$ is the target distribution and $s = \infty$ is standard Gaussian noise. The forward process is initialized at the target distribution:

$$
\dot { x } _ { s } = - x _ { s } + \sqrt { 2 } \eta _ { s } , \quad x _ { s = 0 } \sim p _ { 1 }\tag{245}
$$

Since we set $\varepsilon = 2 , \mathsf { a s } s \to \infty .$ , the distribution $p _ { s } ( x )$ converges to a standard Gaussian (the base distribution). (Note that the intermediate distributions $p _ { s }$ are not equivalent to those of the stochastic interpolant Eq. (237).) The forward process obeys the Fokker-Planck equation:

$$
\begin{array} { c } { { \partial _ { s } p _ { s } ( x ) = \nabla \cdot ( x p _ { s } ) + \frac { 1 } { 2 } \nabla ^ { 2 } p _ { s } } } \\ { { = \nabla \cdot ( p _ { s } ( x + \frac { 1 } { 2 } \nabla \log p _ { s } ) ) } } \end{array}\tag{246}
$$

To reverse the probability flow, one flips the sign of the effective flow field $\begin{array} { r } { ( x + \frac { \varepsilon } { 2 } \dot { \nabla } \log p _ { s } ) \mapsto - \bar { ( } x + \frac { \varepsilon } { 2 } \bar { \nabla } \log p _ { s } ) } \end{array}$ . One thus reverses time, starting from a large final time $S \gg 1$ . The reverse process, which generates target samples starting from noise, reads [156, 157]

$$
\begin{array} { r } { \dot { y } _ { s } = y _ { s } + \frac { \varepsilon } { 2 } \nabla \log p _ { S - s } ( y _ { s } ) + \sqrt { \varepsilon } \eta _ { s } } \end{array}\tag{247}
$$

This idea of time-reversing a process that starts at the target distribution $p _ { 1 }$ is precisely what underlies the proof of annealed importance sampling and the Jarzynski identity in Sec. IV B 2.

Finally, Eq. (244) also allows revisiting the distinction between flow matching and Langevin sampling with energy $E = - \log p _ { 1 }$ . Flow matching uses a non-equilibrium process to transform probability distributions in finite time, and the target $p _ { 1 }$ is not a stationary state. The potential is timedependent, and the intermediate-time distributions $p _ { t }$ are explicitly distinct from the target and instead act as a bridge between $p _ { 0 }$ and $p _ { 1 }$ . By contrast, Langevin dynamics only converge to its equilibrium distribution $p _ { 1 }$ as $t \to \infty$

## 5. One-step generation and normalizing flows

Once a flow-matching model is trained, generating samples requires integrating the flow field v (or an SDE for a diffusion model). This can be slow and makes the generation process difficult to fine-tune for downstream applications. Consistency models [158] and rectifying flows [159] distill the flow field vb into a neural-network model for the flow map $\widehat { T } _ { 1 }$ which allows generating samples in one step. Such one-step models parallel the transition from Benamou-Brenier dynamic optimal transport (Sec. II B, Eq. (91)) to the static Monge-Kantorovich problem (Sec. II B).

In this section, we will discuss another one-step generative method, normalizing flows [160], one of the inspirations for flow matching models. Despite their name, normalizing flows are not flows, but generate samples using a single-step transform. While flow matching excels at learning generative models from training data, normalizing flows can also learn from an energy function or unnormalized probability density. This makes them very useful for improving sampling algorithms in statistical mechanics simulations and Bayesian statistics (Chapter IV). In this setting, no training samples are available: producing them is the whole point.

As before, the goal is to transform a base $p _ { 0 }$ into a target $p _ { 1 }$ distribution via Eq. (234). Samples are generated as $y \ = \ { \widehat { T } } ( x _ { 0 } ) , x _ { 0 } \ \sim \ p _ { 0 }$ where the base distribution $p _ { 0 }$ is often Gaussian (hence, “normalizing” flow). In contrast to flow matching, which constructs $\widehat { T }$ iteratively, normalizing flows directly parametrize $\widehat { T }$ by a neural network. However, not any neural network will do: Eq. (234) shows that the inverse $\widehat { T } ^ { - 1 }$ and the Jacobian determinant det $\nabla _ { \boldsymbol { x } } \widehat { T } ^ { - 1 }$ are needed to compute the transformed density $q = \dot { T } _ { * } p _ { 0 }$ . One thus uses neural network architectures designed specifically to make these quantities easy to evaluate. A minimal example is the following neural network layer for a 2-component vector $x = \left( x ^ { ( 1 ) } \ \breve { x ^ { ( 2 ) } } \right)$ [161] :

![](images/b660e831585038c84e7f8883071777e833bd45de7398720a7e5ee0c69026c6ee.jpg)  
FIG. 14. Memorization-generalization transition for a 1D mixture of Gaussians: interpolating distributions $p _ { t }$ between a Gaussian base and the empirical distribution of a Gaussian mixture $( N = 4$ samples). At the ferromagnetic transition, $p _ { t }$ splits into the distinct modes of the target distribution. At the glass transition, trajectories become trapped near a training sample from the empirical distribution.

$$
\begin{array} { c } { { \left( x ^ { ( 1 ) } x ^ { ( 2 ) } \right) \mapsto } } \\ { { \left( \widehat { a } ( x ^ { ( 2 ) } ) \cdot x ^ { ( 1 ) } + \widehat { b } ( x ^ { ( 2 ) } ) x ^ { ( 2 ) } \right) } } \end{array}\tag{248}
$$

where $\widehat { a }$ and $\widehat { b }$ are neural networks for scale and shift. Since Eq. (248) is an affine function in $x ^ { ( 1 ) }$ and copies $x ^ { ( 2 ) }$ , it is easy to invert. More complex networks can be built by stacking layers.

Next, we need a procedure to adjust $q = \widehat { T } _ { * } p _ { 0 }$ . to match the target distribution $p _ { 1 }$ , paralleling the minimization of the flow matching loss in Eq. (240). One possibility is to use a training dataset $\left\{ x _ { n } \right\}$ of samples from $p _ { 1 }$ . In this case, one maximizes the log-likelihood $\textstyle \sum _ { n = 1 } ^ { N } \log q ( x _ { n } )$ of the training data. In the setting of Bayesian inference or statistical mechanics simulations, one has access to the unnormalized density $\widetilde { p } _ { 1 }$ instead of training samples. In Sec. V B 3, we saw that variational inference uses $p _ { 1 }$ to define a data-free objective:

$$
\begin{array} { l } { \displaystyle \mathrm { D } _ { \mathrm { K L } } ( q \| p _ { 1 } ) = \langle \log q - \log p _ { 1 } \rangle _ { q } } \\ { \displaystyle \approx \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \Big [ \log q ( x ^ { ( n ) } ) - \log p _ { 1 } ( x ^ { ( n ) } ) \Big ] , } \\ { \displaystyle x ^ { ( n ) } = \widehat { T } ( x _ { 0 } ^ { ( n ) } ) , \qquad x _ { 0 } ^ { ( n ) } \sim p _ { 0 } } \end{array}\tag{249}
$$

Here, $D _ { \mathrm { K L } }$ is the Kullback-Leibler (KL) divergence, a measure of the distance between two probability distributions encountered several times in previous chapters. Because of the logarithm, the normalization constant $Z _ { 1 }$ drops out when optimizing Eq. (249). The expectation value in Eq. (249) is approximated using samples from q, so no target distribution samples are required (hence, “data-free”). Now, $\widehat { T }$ can be trained by minimizing $\mathrm { D } _ { \mathrm { K L } } ( q \| p _ { 1 } )$ via gradient descent [162]. The data-free objective has certain drawbacks: it can encourage mode collapse, where the map $\widehat { T }$ always returns samples from a single mode and fails to explore the full distribution $p _ { 1 } \ [ 1 6 3 ]$ . This is a well-known effect of the KL divergence. It strongly penalizes any probability density outside of the support of $p _ { 1 }$ due to the term $\log ( q / p _ { 1 } )$ , but failing to sample from a part of $p _ { 1 }$ receives a much milder penalty.

Once trained, one can generate samples $\widehat { T } ( x _ { 0 } )$ However, if the variational approximation is not perfect, these samples will be biased (not exactly distributed $\sim p _ { 1 } )$ . This is a serious issue in scientific contexts. Fortunately, one can combine a normalizing flow with another sampling technique like importance sampling from Chapter IV to correct the bias [163]. As long as $q$ covers the support of $p _ { 1 }$ , the result is guaranteed to be unbiased. This achieves the “best of both worlds”: the conventional sampling algorithm provides mathematical guarantees, and the power of machine learning allows rapid identification of high-probability regions in d-dimensional space. Normalizing flows have been successfully used to accelerate statistical physics simulations, for instance in lattice QCD [164]. More generally, gradient-based optimization can systematically tune parameters of sampling algorithms that were traditionally tuned by hand, like the proposal distribution for an importance sampler.

## 6. Generalizations and ongoing work

In summary, machine learning has made striking progress on the classic problem of sampling from complex, highdimensional distributions. These techniques transform a simple base distribution to the target via a transform that is fit variationally to data. The greatest success has come from methods based on mass transport, like diffusion and flow matching, which construct the transform map step by step. In flow matching, one first designs a dynamical process that interpolates between the base (often, uniform noise) and the target, and then learns the corresponding probability flow. In the domain of image and video generation, flows and diffusions are the state of the art and have eclipsed previous approaches, like Variational Autoencoders [149] and Generative Adversarial Networks [165]. On the other hand, for language and other “sequential datasets”, autoregressive models, which iteratively predict the next sequence element or “token”, dominate.

Flow matching can be generalized in several directions. First, the design of the stochastic interpolant offers significant flexibility. Eq. (237), linear interpolation, is only the simplest possibility. Second, there is an equal amount of freedom in choosing a sampling algorithm (stochastic or deterministic sampling), which can lead to important numerical efficiency gains. Lastly, different neural network architectures can be used to approximate the velocity field or score. This leads to a large zoo of methods [147, 148].

Next, in generative modeling, one is rarely interested in generating “random” samples from the target distribution $p _ { 1 }$ (in contrast to statistical mechanics). After all, a random sample could simply be drawn from the training set. Instead, one is interested in a sample that fulfills some particular task: an image that matches a given caption, or a protein with a particular functionality. Guided diffusion uses additional “guidance” terms to nudge the sampling process $y _ { t }$ towards a user-defined aim [166]. The population distribution $p _ { 1 }$ of images/proteins/... serves as a conduit, a convenient fiction, towards the ultimate goal of a task-directed model. Guided diffusion is a form of sampling with a control $u _ { t }$ which we discussed above. A running reward can be supplied by an ML model (like a classifier) to score whether the trajectory $\mathbf { x } _ { t }$ progresses towards a desired sample like a particular type of image, and a terminal reward can score the final sample.

Finally, flow matching can also be generalized from continuous distributions on $\bar { \mathbb { R } } ^ { d }$ , such as images, to discrete ones, such as text data. In this case, a sample is a sequence $x =$ $\{ x _ { 1 } , . . . , x _ { d } \} , x _ { i } \in \mathcal { A }$ from a finite alphabet ${ \mathcal { A } } .$ The stochastic interpolant becomes a Markov chain that converts a sequence into a sample from a simple base distribution $p _ { 0 }$ . For example, masked diffusion [167] continuously masks out sequence elements (so $p _ { 0 }$ has only one element, the fully masked sequence). The goal is now to learn a Markov process that reverses the masking. The transition rates of this process play the role of the velocity field in the continuous setting, and one minimizes a loss similar to Eq. (240). Alternatively, the discrete data can be embedded into a continuous space. For language modeling, discrete diffusion has recently become competitive with the next-token-prediction paradigm that underlies most large language models [167]. In contrast to the latter, diffusion models can generate many tokens at once, and thus leverage more computing power.

Statistical mechanics of memorization and generalization. Flow matching and diffusion models are optimized by using a set of training samples, but so far, we have not paid any attention to the limited size of this training set. The models are fit not to the true target $p _ { 1 }$ , but to the empirical distribution $\begin{array} { r } { \widehat { p } _ { \mathrm { d a t a } } ( x ) = N ^ { - 1 } \sum _ { n = 1 } ^ { N } \delta ( x - x _ { n } ) } \end{array}$ of the finite training set. Therefore, the “optimal” flow map under Eq. (240) reproduces the empirical distribution, and generation always returns a sample from the training set. This evidently undesirable phenomenon is called memorization; the model fails to generalize to new samples. In practice, memorization is avoided by stopping the training process early, and systematic approaches to avoid memorization are a topic of current research. The phenomenon of benign overfitting, where a model perfectly interpolates the training data without impeding performance, is thus absent in flow models. A model with zero training flow matching loss always memorizes.

The dynamics of diffusion models starting from pure noise bear a close resemblance to a statistical mechanics system that is slowly cooled from an initially high temperature. Flow-matching time is thus analogous to temperature. Indeed, flow matching and diffusion models can be fruitfully analyzed using statistical mechanics, which has shed light on the generalization-memorization transition. Refs. [168, 169] argue that flow matching dynamics undergo a series of symmetry-breaking phase transitions as a function of time t.

At a first “ferromagnetic” transition, samples $x _ { t }$ commit to one of the modes of the target distribution, generating a novel sample. However, as the dynamics continue, a second “glass” transition traps $x _ { t }$ near a sample from the training dataset (Fig. 14).

In Chapter $^ \mathrm { I V , }$ we saw how an analogy with physical annealing gave rise to powerful algorithms for sampling from a known distribution. An interesting avenue for future research is whether the analogy between annealing and flow matching could be used to design “better”, more statistically efficient generative models, now learned from training data.

## CONTRIBUTIONS AND ACKNOWLEDGMENTS

## Contributions

E.B.: Chapters I and II

N.C.: Chapters I and IV, and Sec. V C

B.E.: Sec. V A

C.J.: Chapter I and Sec. V A

G.R.: Chapters I and II

C.S.: Chapter II and Sec. V B

B.S.: Chapters I and III

Overlapping sections were written collaboratively by the authors listed above. All authors reviewed and revised the manuscript.

## Acknowledgments

E.B. acknowledges support from the Fannie and John Hertz Foundation Fellowship. E.B. and C.J. acknowledge this material is based upon work supported by the National Science Foundation Graduate Research Fellowship Program under Grant No. DGE-2444107. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation. N.C. was supported in part by Princeton University through the Princeton Center for Theoretical Science and the Center for the Physics of Biological Function, as well as by the National Institute of General Medical Sciences (NIGMS) of the National Institutes of Health (NIH) under award numbers R01GM082938 and R35GM156427. The content is solely the responsibility of the authors and does not necessarily represent the official views of the National Institutes of Health. Part of this work was performed at the Kavli Institute for Theoretical Physics (KITP), supported by NSF grant PHY-2309135 and Gordon and Betty Moore Foundation Grant No. 2919.02. N.C. and B.S. acknowledge useful discussions at the Beg Rohu Summer School of Physics 2026 and thank all summer school lecturers. C.S. was supported in part by Princeton University through the Center for the Physics of Biological Function. This work was performed in part at the Aspen Center for Physics, which is supported by National Science Foundation grant PHY-2210452. B.S. was supported by the Princeton Center for Theoretical Science and in part by the Center for the Physics of Biological Function at Princeton University. G.R. was partially supported by a joint research agreement between NTT Research Inc and Princeton University. We acknowledge the use of generative AI for editorial purposes.

[1] R. Bellman, The theory of dynamic programming, Bulletin of the American Mathematical Society 60, 503 (1954).

[2] L. S. Pontryagin, V. G. Boltyanskii, R. V. Gamkrelidze, and E. F. Mishchenko, The Mathematical Theory of Optimal Processes (Interscience Publishers, New York, 1962) translated from the Russian by K. N. Trirogoff; English edition edited by L. W. Neustadt.

[3] S. Dreyfus, Richard Bellman on the birth of dynamic programming, Operations Research 50, 48 (2002).

[4] L. C. Evans, An introduction to mathematical optimal control theory, Lecture notes, Department of Mathematics, University of California, Berkeley (2024), spring 2024 version.

[5] B. Øksendal, Stochastic differential equations, in Stochastic differential equations: an introduction with applications (Springer, 2003) pp. 38–50.

[6] W. H. Fleming and R. W. Rishel, Deterministic and Stochastic Optimal Control, Applications of Mathematics, Vol. 1 (Springer-Verlag, New York, 1975).

[7] M. Mezard and A. Montanari, ´ Information, Physics, and Computation, 1st ed. (Oxford University PressOxford, 2009).

[8] A. Gelman, J. B. Carlin, H. S. Stern, D. B. Dunson, A. Vehtari, and D. B. Rubin, Bayesian Data Analysis, 0th ed. (Chapman and Hall/CRC, 2013).

[9] T. M. Cover and J. Thomas, Elements of information theory (Wiley, 1999).

[10] I. N. Sanov, On the probability of large deviations of random magnitudes, Matematicheskii Sbornik. Novaya Seriya 42(84), 11 (1957), in Russian.

[11] H. Touchette, The large deviation approach to statistical mechanics, Phys. Rep. 478, 1 (2009).

[12] C. E. Shannon, A mathematical theory of communication, Bell Syst. Tech. J. 27, 379 (1948).

[13] E. T. Jaynes, Information theory and statistical mechanics, Phys. Rev. 106, 620 (1957).

[14] E. Jaynes, On the rationale of maximum-entropy methods, Proc. IEEE 70, 939 (1982).

[15] S. C. Zhu, Y. N. Wu, and D. Mumford, Minimax entropy principle and its application to texture modeling, Neural Comput. 9, 1627 (1997).

[16] M. J. Wainwright, M. I. Jordan, et al., Graphical models, exponentialfamilies, and variational inference (Now Publishers, 2008).

[17] S. Presse, K. Ghosh, J. Lee, and K. A. Dill, Principles of max-´ imum entropy and maximum caliber in statistical physics, Reviews of Modern Physics 85, 1115 (2013).

[18] I. V. Girsanov, On transforming a certain class of stochastic processes by absolutely continuous substitution of measures, Theory of Probability & Its Applications 5, 285 (1960).

[19] E. Theodorou and E. Todorov, Relative entropy and free energy dualities: Connections to path integral and KL control, in IEEE 51st Annual Conference on Decision and Control (CDC) (2012) pp. 1466–1473.

[20] S. Levine, Reinforcement learning and control as probabilistic inference: Tutorial and review, arXiv preprint arXiv:1805.00909 (2018).

[21] M. Boue and P. Dupuis, A variational representation for cer-´ tain functionals of Brownian motion, The Annals of Probability 26, 1641 (1998).

[22] H. J. Kappen, Linear theory for control of nonlinear stochastic systems, Physical Review Letters 95, 200201 (2005).

[23] H. J. Kappen, Path integrals and symmetry breaking for optimal control theory, Journal of Statistical Mechanics: Theory and Experiment 2005, P11011 (2005).

[24] E. Todorov, Linearly-solvable markov decision problems, Advances in neural information processing systems 19 (2006).

[25] E. Todorov, Efficient computation of optimal actions, Proceedings of the National Academy of Sciences 106, 11478 (2009).

[26] E. Theodorou, J. Buchli, and S. Schaal, A generalized path integral control approach to reinforcement learning, Journal of Machine Learning Research 11, 3137 (2010).

[27] W. H. Fleming and H. M. Soner, Controlled Markov Processes and Viscosity Solutions, 2nd ed., Stochastic Modelling and Applied Probability, Vol. 25 (Springer, New York, 2006).

[28] R. Chetrite and H. Touchette, Nonequilibrium Markov processes conditioned on large deviations, Annales Henri Poincare´ 16, 2005 (2015).

[29] E. Schrodinger,¨ Uber die Umkehrung der Naturgesetze,<sup>¨</sup> Sitzungsberichte der Preussischen Akademie der Wissenschaften, Physikalisch-Mathematische Klasse 8–9, 144 (1931).

[30] E. Schrodinger, Sur la th¨ eorie relativiste de l’´ Electron<sup>´</sup> et l’interpretation de la m ´ ecanique quantique, Annales de´ l’Institut Henri Poincare´ 2, 269 (1932).

[31] C. Leonard, A survey of the Schr´ odinger problem and some of¨ its connections with optimal transport, Discrete and Continuous Dynamical Systems 34, 1533 (2014).

[32] Y. Chen, T. T. Georgiou, and M. Pavon, Stochastic control liaisons: Richard Sinkhorn meets Gaspard Monge on a Schrodinger bridge,¨ SIAM Review 63, 249 (2021).

[33] R. Sinkhorn, Diagonal equivalence to matrices with prescribed row and column sums, The American Mathematical Monthly 74, 402 (1967).

[34] C. Villani and A. M. Society, Topics in Optimal Transportation, Graduate studies in mathematics (American Mathematical Society, 2003).

[35] M. Cuturi, Sinkhorn distances: Lightspeed computation of optimal transport, in Advances in Neural Information Processing Systems 26 (2013) pp. 2292–2300.

[36] J.-D. Benamou and Y. Brenier, A computational fluid mechanics solution to the Monge-Kantorovich mass transfer problem, Numer. Math. 84, 375 (2000).

[37] Y. Brenier, Polar factorization and monotone rearrangement of vector-valued functions, Communications on Pure and Applied Mathematics 44, 375 (1991).

[38] E. Blumenthal and G. Reddy, Self-organized robustness in mean-field interacting systems (2026), arXiv:2606.27626 [physics.bio-ph].

[39] J.-D. Benamou, Y. Brenier, and K. Guittet, The monge–kantorovitch mass transfer and its computa-

tional fluid mechanics formulation, International Journal for Numerical Methods in Fluids 40, 21 (2002), https://onlinelibrary.wiley.com/doi/pdf/10.1002/fld.264.

[40] Y. Chen, T. T. Georgiou, and M. Pavon, On the relation between optimal transport and Schrodinger bridges: A stochastic¨ control viewpoint, Journal of Optimization Theory and Applications 169, 671 (2016).

[41] R. J. McCann, A convexity principle for interacting gases, Advances in Mathematics 128, 153 (1997).

[42] F. Otto, The geometry of dissipative evolution equations: The porous medium equation, Communications in Partial Differential Equations 26, 101 (2001).

[43] S. Chewi, J. Niles-Weed, and P. Rigollet, Statistical Optimal Transport (Springer, 2025).

[44] L. Ambrosio, N. Gigli, and G. Savare,´ Gradient Flows in Metric Spaces and in the Space of Probability Measures, 2nd ed., Lectures in Mathematics ETH Zurich (Birkh¨ auser) oCLC:¨ 254181287.

[45] R. Jordan, D. Kinderlehrer, and F. Otto, The variational formulation of the Fokker–Planck equation, SIAM J. Math. Analys. 29, 1 (1998).

[46] M. Kardar, Statistical Physics of Particles (Cambridge University Press, 2007).

[47] J. Casas-Vazquez and D. Jou, Temperature in non-equilibrium states: A review of open problems and current proposals, Rep. Prog. Phys. 66, 1937 (2003).

[48] A. P. Solon, Y. Fily, A. Baskaran, M. E. Cates, Y. Kafri, M. Kardar, and J. Tailleur, Pressure is not a state function for generic active fluids, Nat. Phys. 11, 673 (2015).

[49] J. L. Lebowitz, Microscopic origins of irreversible macroscopic behavior, Phys. A: Stat. Mech. Appl. 263, 516 (1999).

[50] Y. Levin, R. Pakter, F. B. Rizzato, T. N. Teles, and F. P. Benetti, Nonequilibrium statistical mechanics of systems with longrange interactions, Phys. Rep. 535, 1 (2014).

[51] U. Seifert, Stochastic thermodynamics, fluctuation theorems and molecular machines, Rep. Prog. Phys. 75, 126001 (2012).

[52] L. Peliti and S. Pigolotti, Stochastic Thermodynamics (Princeton University Press, 2021).

[53] M. Weigt, R. A. White, H. Szurmant, J. A. Hoch, and T. Hwa, Identification of direct residue contacts in protein–protein interaction by message passing, Proc. Natl. Acad. Sci. U.S.A. 106, 67 (2009).

[54] L. Meshulam and W. Bialek, Statistical mechanics for networks of real neurons, Rev. Mod. Phys. 97, 045002 (2025).

[55] B. Sorkin, A. Be’er, H. Diamant, and G. Ariel, Detecting and characterizing phase transitions in active matter using entropy, Soft Matter 19, 5118 (2023).

[56] A. Cavagna and I. Giardina, Bird flocks as condensed matter, Annu. Rev. Condens. Matter Phys. 5, 183 (2014).

[57] A. Cavagna, I. Giardina, F. Ginelli, T. Mora, D. Piovani, R. Tavarone, and A. M. Walczak, Dynamical maximum entropy approach to flocking, Phys. Rev. E 89, 042707 (2014).

[58] B. Sorkin, J. Ricouvier, H. Diamant, and G. Ariel, Resolving entropy contributions in nonequilibrium transitions, Phys. Rev. E 107, 014138 (2023).

[59] R. Avinery, M. Kornreich, and R. Beck, Universal and accessible entropy estimation using a compression algorithm, Phys. Rev. Lett. 123, 178102 (2019).

[60] S. Martiniani, P. M. Chaikin, and D. Levine, Quantifying hidden order out of equilibrium, Phys. Rev. X 9, 011031 (2019).

[61] J. C. Dyre, Perspective: Excess-entropy scaling, J. Chem. Phys. 149, 210901 (2018).

[62] B. Sorkin, H. Diamant, and G. Ariel, Universal relation between entropy and kinetics, Phys. Rev. Lett. 131, 147101

(2023).

[63] L. Onsager, Reciprocal relations in irreversible processes. I., Phys. Rev. 37, 405 (1931).

[64] L. Onsager, Reciprocal relations in irreversible processes. II., Phys. Rev. 38, 2265 (1931).

[65] R. Zwanzig, Nonequilibrium Statistical Mechanics (Oxford University Press, 2001).

[66] S. R. De Groot and P. Mazur, Non-Equilibrium Thermodynamics (Dover Publications, 1984).

[67] M. Doi, Onsager’s variational principle in soft matter, J. Phys.: Condens. Matter 23, 284118 (2011).

[68] J. P. Hansen and I. McDonald, Theory of Simple Liquids, 3rd ed. (Academic Press, 2003).

[69] Z. Schuss, Theory and Applications of Stochastic Processes (Springer, New York, 2010).

[70] K. Sekimoto, Langevin equation and thermodynamics, Prog. Theor. Phys. 130, 17 (1998).

[71] E. Aurell, K. Gawedzki, C. Mejia-Monasterio, R. Mohayaee, and P. Muratore-Ginanneschi, Refined second law of thermodynamics for fast random processes, J. Stat. Phys. 147, 487 (2012).

[72] T. Van Vu and K. Saito, Thermodynamic unification of optimal transport: Thermodynamic uncertainty relation, minimum dissipation, and thermodynamic speed limits, Phys. Rev. X 13, 011013 (2023).

[73] A. Taghvaei, O. M. Miangolarra, R. Fu, Y. Chen, and T. T. Georgiou, On the relation between information and power in stochastic thermodynamic engines, IEEE Control Sys. Lett. 6, 434 (2022).

[74] B. Sorkin, G. Ariel, and T. Markovich, Consistent expansion of the Langevin propagator with application to entropy production, J. Stat. Mech.: Theor. Exp. , 013208 (2025).

[75] B. Sorkin, H. Diamant, G. Ariel, and T. Markovich, Second law of thermodynamics without einstein relation, Phys. Rev. Lett. 133, 267101 (2024).

[76] D. A. Sivak and G. E. Crooks, Thermodynamic metrics and optimal paths, Phys. Rev. Lett. 108, 190602 (2012).

[77] G. M. Rotskoff and G. E. Crooks, Optimal control in nonequilibrium systems: Dynamic Riemannian geometry of the Ising model, Physical Review E 92, 060102 (2015).

[78] C. Jarzynski, Equilibrium free-energy differences from nonequilibrium measurements: A master-equation approach, Physical Review E 56, 5018 (1997).

[79] G. E. Crooks, Entropy production fluctuation theorem and the nonequilibrium work relation for free energy differences, Phys. Rev. E 60, 2721 (1999).

[80] G. M. Rotskoff, G. E. Crooks, and E. Vanden-Eijnden, Geometric approach to optimal nonequilibrium control: Minimizing dissipation in nanomagnetic spin systems, Phys. Rev. E 95, 012148 (2017).

[81] M. C. Engel, J. A. Smith, and M. P. Brenner, Optimal control of nonequilibrium systems through automatic differentiation, Phys. Rev. X 13, 041032 (2023).

[82] C. Jarzynski, Nonequilibrium equality for free energy differences, Physical Review Letters 78, 2690 (1997).

[83] G. Hummer and A. Szabo, Free energy reconstruction from nonequilibrium single-molecule pulling experiments, Proc. Natl. Acad. Sci. U.S.A. 98, 3658 (2001).

[84] D. Collin, F. Ritort, C. Jarzynski, S. B. Smith, I. Tinoco, and C. Bustamante, Verification of the Crooks fluctuation theorem and recovery of RNA folding free energies, Nature 437, 231 (2005).

[85] M. Betancourt, A Conceptual Introduction to Hamiltonian Monte Carlo (2018), arXiv:1701.02434 [stat].

[86] R. Vershynin, High-Dimensional Probability: An Introduction with Applications in Data Science, 2nd ed. (Cambridge University Press, 2026).

[87] W. Krauth, Statistical Mechanics: Algorithms and Computations (Oxford University PressOxford, 2006).

[88] S. Chatterjee and P. Diaconis, The sample size required in importance sampling, The Annals of Applied Probability 28, 1099 (2018).

[89] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, Proximal policy optimization algorithms, arXiv preprint arXiv:1707.06347 (2017).

[90] S. Kirkpatrick, C. D. Gelatt, and M. P. Vecchi, Optimization by Simulated Annealing, Science 220, 671 (1983).

[91] S. Coste, Importance sampling: The Jarzynski connection (2026).

[92] R. M. Neal, Annealed Importance Sampling (1998), arXiv:physics/9803008.

[93] D. Frenkel and B. Smit, Chapter 7 - free energy calculations, in Understanding Molecular Simulation (Second Edition) (Academic Press, San Diego, 2002) second edition ed., pp. 167– 200.

[94] H. J. Kappen and H. C. Ruiz, Adaptive importance sampling for control and inference, Journal of Statistical Physics 162, 1244 (2016).

[95] A. N. Singh, A. Das, and D. T. Limmer, Variational Path Sampling of Rare Dynamical Events, Annual Review of Physical Chemistry 76, 639 (2025).

[96] A. Das and D. T. Limmer, Variational control forces for enhanced sampling of nonequilibrium molecular dynamics simulations, The Journal of Chemical Physics 151, 244123 (2019).

[97] R. S. Sutton, A. G. Barto, et al., Reinforcement learning: An introduction, Vol. 1 (MIT press Cambridge, 1998).

[98] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, Soft actorcritic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor, in International conference on machine learning (Pmlr, 2018) pp. 1861–1870.

[99] T. L. Lai and H. Robbins, Asymptotically efficient adaptive allocation rules, Advances in Applied Mathematics 6, 4 (1985).

[100] L. P. Kaelbling, M. L. Littman, and A. R. Cassandra, Planning and acting in partially observable stochastic domains, Artificial Intelligence 101, 99 (1998).

[101] T. Ni, B. Eysenbach, and R. Salakhutdinov, Recurrent modelfree RL can be a strong baseline for many POMDPs, in Proceedings of the 39th International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 162 (PMLR, 2022) pp. 16691–16723.

[102] S. Fujimoto, H. van Hoof, and D. Meger, Addressing function approximation error in actor-critic methods, in Proceedings of the 35th International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 80 (PMLR, 2018) pp. 1587–1596.

[103] A. Bhatt, D. Palenicek, B. Belousov, M. Argus, A. Amiranashvili, T. Brox, and J. Peters, CrossQ: Batch normalization in deep reinforcement learning for greater sample efficiency and simplicity, in International Conference on Learning Representations (2024).

[104] R. S. Sutton, Learning to predict by the methods of temporal differences, Machine Learning 3, 9 (1988).

[105] C. J. Watkins and P. Dayan, Q-learning, Machine learning 8, 279 (1992).

[106] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou,

H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis, Human-level control through deep reinforcement learning, Nature 518, 529 (2015).

[107] L. Baird, Residual algorithms: Reinforcement learning with function approximation, in Proceedings ofthe Twelfth International Conference on Machine Learning (Morgan Kaufmann, 1995) pp. 30–37.

[108] T. Haarnoja, H. Tang, P. Abbeel, and S. Levine, Reinforcement learning with deep energy-based policies, in International conference on machine learning (PMLR, 2017) pp. 1352–1361.

[109] B. Eysenbach and S. Levine, Maximum entropy RL (provably) solves some robust RL problems, in 10th International Conference on Learning Representations, ICLR 2022 (2022).

[110] T. Haarnoja, A. Zhou, K. Hartikainen, G. Tucker, S. Ha, J. Tan, V. Kumar, H. Zhu, A. Gupta, P. Abbeel, and S. Levine, Soft actor-critic algorithms and applications (2018), arXiv:1812.05905 [cs.LG].

[111] A. Agarwal, N. Jiang, S. M. Kakade, and W. Sun, Reinforcement learning: Theory and algorithms, CS Dept., UW Seattle, Seattle, WA, USA, Tech. Rep 32, 96 (2019).

[112] C. Jin, Z. Yang, Z. Wang, and M. I. Jordan, Provably efficient reinforcement learning with linear function approximation, Mathematics of Operations Research 48, 1496 (2023).

[113] A. W. Moore, Efficient Memory-based Learning for Robot Control, Tech. Rep. (University of Cambridge, 1990).

[114] S. Huber, H. Unger, G. Schafer, and J. Rehrl, Chebyshev poli-¨ cies and the mountain car problem: Reinforcement learning for low-dimensional control tasks, in Forty-third International Conference on Machine Learning (2026).

[115] J. Carrillo, K. Craig, L. Wang, and C. Wei, Primal dual methods for Wasserstein gradient flows, F. Comput. Math. 22, 389 (2022).

[116] M. Burger, J. A. Carrillo, and M.-T. Wolfram, A mixed finite element method for nonlinear diffusion equations (2010).

[117] P. Halmos and B. Hanin, Implicit bias of the jko scheme (2026), arXiv:2511.14827 [stat.ML].

[118] H. P. McKean, A class of markov processes associated with nonlinear parabolic equations, Proceedings of the National Academy of Sciences 56, 1907 (1966).

[119] A.-S. Sznitman, Topics in propagation of chaos, in Ecole d’Ete´ de Probabilites de Saint-Flour XIX — 1989´ , edited by P.-L. Hennequin (Springer Berlin Heidelberg, Berlin, Heidelberg, 1991) pp. 165–251.

[120] L. Chizat and F. Bach, On the global convergence of gradient descent for over-parameterized models using optimal transport, in Proceedings of the 32nd International Conference on Neural Information Processing Systems, NIPS’18 (Curran Associates Inc., Red Hook, NY, USA, 2018) p. 3040–3050.

[121] G. Rotskoff and E. Vanden-Eijnden, Trainability and accuracy of artificial neural networks: An interacting particle system approach, Communications on Pure and Applied Mathematics 75, 1889 (2022), https://onlinelibrary.wiley.com/doi/pdf/10.1002/cpa.22074.

[122] S. Mei, A. Montanari, and P.-M. Nguyen, A mean field view of the landscape of two-layer neural networks, Proceedings of the National Academy of Sciences 115, E7665 (2018), https://www.pnas.org/doi/pdf/10.1073/pnas.1806579115.

[123] J. Sirignano and K. Spiliopoulos, Mean field analysis of neural networks: A law of large numbers, SIAM Journal on Applied Mathematics 80, 725 (2020).

[124] G. Cybenko, Approximation by superpositions of a sigmoidal function, Mathematics of Control, Signals and Systems 2, 303 (1989).

[125] A. R. Barron, Universal approximation bounds for superpositions of a sigmoidal function, IEEE Trans. Inf. Theor. 39, 930–945 (1993).

[126] J. Park and I. W. Sandberg, Universal approximation using radial-basis-function networks, Neural Computation 3, 246 (1991).

[127] P.-M. Nguyen and H. T. Pham, A rigorous framework for the mean field limit of multilayer neural networks, Mathematical Statistics and Learning 6, 201 (2023).

[128] J. Sirignano and K. Spiliopoulos, Mean field analysis of deep neural networks, Mathematics of Operations Research 47, 120 (2022).

[129] D. M. Blei, A. Kucukelbir, and J. D. McAuliffe, Variational inference: A review for statisticians, Journal of the American Statistical Association 112, 859 (2017).

[130] A. Takatsu, Wasserstein geometry of Gaussian measures, Osaka Journal of Mathematics 48, 1005 (2011).

[131] M. Lambert, S. Chewi, F. Bach, S. Bonnabel, and P. Rigollet, Variational inference via wasserstein gradient flows, in Advances in Neural Information Processing Systems, edited by A. H. Oh, A. Agarwal, D. Belgrave, and K. Cho (2022).

[132] M. Diao, K. Balasubramanian, S. Chewi, and A. Salim, Forward-backward gaussian variational inference via jko in the bures–wasserstein space, in Proceedings of the 40th International Conference on Machine Learning, ICML’23 (JMLR.org, 2023).

[133] Q. Liu and D. Wang, Stein variational gradient descent: a general purpose bayesian inference algorithm, in Proceedings of the 30th International Conference on Neural Information Processing Systems, NIPS’16 (Curran Associates Inc., Red Hook, NY, USA, 2016) p. 2378–2386.

[134] Q. Liu, Stein variational gradient descent as gradient flow, in Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS’17 (Curran Associates Inc., Red Hook, NY, USA, 2017) p. 3118–3126.

[135] J. Lu, Y. Lu, and J. Nolen, Scaling limit of the stein variational gradient descent: The mean field regime, SIAM Journal on Mathematical Analysis 51, 648 (2019).

[136] A. Korba, A. Salim, M. Arbel, G. Luise, and A. Gretton, A non-asymptotic analysis for stein variational gradient descent, in Advances in Neural Information Processing Systems, Vol. 33 (2020) pp. 4672–4682.

[137] A. Salim, L. Sun, and P. Richtarik, A convergence theory for SVGD in the population limit under talagrand’s inequality t1, in Proceedings of the 39th International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 162, edited by K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvari, G. Niu, and S. Sabato (PMLR, 2022) pp. 19139–19152.

[138] A. Das and D. Nagaraj, Provably fast finite particle variants of svgd via virtual particle stochastic approximation, in Advances in Neural Information Processing Systems, Vol. 36, edited by A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (Curran Associates, Inc., 2023) pp. 49748–49760.

[139] J. Shi and L. Mackey, A finite-particle convergence rate for stein variational gradient descent, in Proceedings of the 37th International Conference on Neural Information Processing Systems, NIPS ’23 (Curran Associates Inc., Red Hook, NY, USA, 2023).

[140] Y. Chen, T. T. Georgiou, and A. Tannenbaum, Optimal transport for gaussian mixture models, IEEE Access 7, 6269 (2019).

[141] J. Delon and A. Desolneux, A wasserstein-type distance in the space of gaussian mixture models, SIAM Journal on Imaging

Sciences 13, 936 (2020).

[142] Y. Jiang, S. Chewi, and A.-A. Pooladian, Algorithms for mean-field variational inference via polyhedral optimization in the Wasserstein space, in Proceedings of Thirty Seventh Conference on Learning Theory, Proceedings of Machine Learning Research, Vol. 247, edited by S. Agrawal and A. Roth (PMLR, 2024) pp. 2720–2721.

[143] S. Ghosh, Y. Lu, T. Nowicki, and E. Zhang, On representations of mean-field variational inference, Journal of Applied Analysis doi:10.1515/jaa-2024-0110.

[144] J. L. Watson, D. Juergens, N. R. Bennett, B. L. Trippe, J. Yim, H. E. Eisenach, W. Ahern, A. J. Borst, R. J. Ragotte, L. F. Milles, B. I. M. Wicky, N. Hanikel, S. J. Pellock, A. Courbet, W. Sheffler, J. Wang, P. Venkatesh, I. Sappington, S. V. Torres, A. Lauko, V. De Bortoli, E. Mathieu, S. Ovchinnikov, R. Barzilay, T. S. Jaakkola, F. DiMaio, M. Baek, and D. Baker, De novo design of protein structure and function with RFdiffusion, Nature 620, 1089 (2023).

[145] J. Ho, A. Jain, and P. Abbeel, Denoising Diffusion Probabilistic Models (2020), arXiv:2006.11239 [cs].

[146] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli, Deep unsupervised learning using nonequilibrium thermodynamics, in Proceedings of the 32nd International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 37, edited by F. Bach and D. Blei (PMLR, Lille, France, 2015) pp. 2256–2265.

[147] M. S. Albergo, N. M. Boffi, and E. Vanden-Eijnden, Stochastic Interpolants: A Unifying Framework for Flows and Diffusions (2025), arXiv:2303.08797 [cs].

[148] Y. Lipman, M. Havasi, P. Holderrieth, N. Shaul, M. Le, B. Karrer, R. T. Q. Chen, D. Lopez-Paz, H. Ben-Hamu, and I. Gat, Flow Matching Guide and Code (2024), arXiv:2412.06264 [cs].

[149] D. P. Kingma and M. Welling, An Introduction to Variational Autoencoders 10.48550/ARXIV.1906.02691 (2019).

[150] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le, Flow matching for generative modeling, in International Conference on Learning Representations (2023) arXiv:2210.02747.

[151] Z. Kadkhodaie, A.-A. Pooladian, S. Chewi, and E. Simoncelli, Blind denoising diffusion models and the blessings of dimensionality (2026), arXiv:2602.09639 [cs].

[152] C. Villani, Optimal Transport, edited by M. Berger, B. Eckmann, P. De La Harpe, F. Hirzebruch, N. Hitchin, L. Hormander, A. Kupiainen, G. Lebeau, M. Ratner, D. Serre,¨ Ya. G. Sinai, N. J. A. Sloane, A. M. Vershik, and M. Waldschmidt, Grundlehren Der Mathematischen Wissenschaften, Vol. 338 (Springer Berlin Heidelberg, Berlin, Heidelberg, 2009).

[153] P. Vincent, A connection between score matching and denoising autoencoders, Neural Computation 23, 1661 (2011).

[154] A. Hyvarinen, Estimation of non-normalized statistical mod-¨ els by score matching, Journal of Machine Learning Research 6, 695 (2005).

[155] Y. Song and S. Ermon, Generative modeling by estimating gradients of the data distribution, in Advances in Neural Information Processing Systems, Vol. 32 (2019).

[156] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, Score-Based Generative Modeling through Stochastic Differential Equations (2021), arXiv:2011.13456 [cs].

[157] B. D. Anderson, Reverse-time diffusion equation models, Stochastic Processes and their Applications 12, 313 (1982).

[158] Y. Song, P. Dhariwal, M. Chen, and I. Sutskever, Consistency Models (2023).

[159] X. Liu, C. Gong, and Q. Liu, Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow (2022).

[160] D. Rezende and S. Mohamed, Variational inference with normalizing flows, in Proceedings ofthe 32nd International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 37, edited by F. Bach and D. Blei (PMLR, Lille, France, 2015) pp. 1530–1538.

[161] L. Dinh, J. Sohl-Dickstein, and S. Bengio, Density estimation using Real NVP, in International Conference on Learning Representations (2017) arXiv:1605.08803.

[162] F. Noe, S. Olsson, J. K´ ohler, and H. Wu, Boltzmann genera-¨ tors: Sampling equilibrium states of many-body systems with deep learning, Science 365, eaaw1147 (2019).

[163] M. Gabrie, G. M. Rotskoff, and E. Vanden-Eijnden, Adap-´ tive Monte Carlo augmented with normalizing flows, Proceedings of the National Academy of Sciences 119, e2109420119 (2022).

[164] G. Kanwar, M. S. Albergo, D. Boyda, K. Cranmer, D. C. Hackett, S. Racaniere, D. J. Rezende, and P. E. Shanahan,\` Equivariant Flow-Based Sampling for Lattice Gauge Theory, Physical Review Letters 125, 121601 (2020).

[165] I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, Generative Adversarial Networks (2014).

[166] P. Dhariwal and A. Nichol, Diffusion models beat GANs on image synthesis, in Advances in Neural Information Processing Systems, Vol. 34 (2021).

[167] S. S. Sahoo, M. Arriola, Y. Schiff, A. Gokaslan, E. Marroquin, J. T. Chiu, A. Rush, and V. Kuleshov, Simple and Effective Masked Diffusion Language Models (2024), arXiv:2406.07524 [cs].

[168] G. Raya and L. Ambrogioni, Spontaneous symmetry breaking in generative diffusion models, in Advances in Neural Information Processing Systems 36 (Neural Information Processing Systems Foundation, Inc. (NeurIPS), New Orleans, Louisiana, USA, 2023) pp. 66377–66389.

[169] G. Biroli, T. Bonnaire, V. De Bortoli, and M. Mezard, Dynam-´ ical regimes of diffusion models, Nature Communications 15, 9957 (2024).

## TABLES OF NOTATION

The final column links to the section where each symbol first appears. Each chapter has a separate table, and we list notation in the order in which each symbol was first introduced. Familiar symbols may have different meanings in clearly identified contexts, for example, mechanical action $S [ x ( \cdot ) ]$ and thermodynamic entropy $S [ p ]$

TABLE I: Chapter I: Mechanics, Control, and Inference.
<table><tr><td>Symbol</td><td>Meaning or convention</td><td>Section</td></tr><tr><td> $x , x _ { t } , x ( \cdot )$ </td><td>State argument, state at time  $t ,$  and complete trajectory. In Chapter  $\mathrm { V } , x _ { n }$  denotes the state at decision step n and yt distinguishes a generated path from an</td><td>IA</td></tr><tr><td> $u , u _ { t } ( x )$ </td><td>interpolant. Control value and feedback control; the applied control is  $u _ { t } ( x _ { t } )$  . In Chapter</td><td>IA</td></tr><tr><td> $t , s$ </td><td>also denotes a prescribed experimental protocol. Time; the control horizon is  $0 \leq t \leq 1$ </td><td>IA</td></tr><tr><td> $g _ { t } ( x , u )$ </td><td>General controlled drift.</td><td>IA</td></tr><tr><td> $\varepsilon$ </td><td>Noise covariance per unit time; amplitude  ${ \sqrt { \varepsilon } } ,$  diffusion coefficient  $\varepsilon / 2 .$  Also the entropic transport regularization in Sec. II B. In Chapter II,  $\varepsilon = 2 D$ </td><td>IA</td></tr><tr><td> $\eta _ { t } , \ W _ { t }$ </td><td>Standard Gaussian white noise and standard Wiener process.</td><td>IA</td></tr><tr><td> $r _ { t } ( x , u )$ </td><td>Running reward, including control costs. In Sec.  ${ \mathrm { V A } } , r ( x , u )$  is a per-step reward; its running-reward limit includes a factor  $\Delta t .$ </td><td>IA</td></tr><tr><td> $r _ { 1 } ( x )$ </td><td>Terminal reward, separate from running reward. In Sec. II A it enforces the final</td><td>IA</td></tr><tr><td> $V _ { t } ( x )$ </td><td>density constraint. Optimal expected reward-to-go, including control costs. In Sec.  $\mathrm { \Delta V } \mathrm { A } , V ^ { \pi }$  evaluates</td><td>IA</td></tr><tr><td> $L _ { t } ( x , \dot { x } ) , S [ x ( \cdot ) ]$ </td><td>a policy and  $V ^ { * }$  is optimal. Mechanical Lagrangian and action.</td><td>IA</td></tr><tr><td> $p _ { t }$ </td><td>Costate; canonical momentum in the mechanics specialization.</td><td>IA</td></tr><tr><td> $\mathcal { H } _ { t } ( x , p , u )$ </td><td>Control Hamiltonian before maximization over u</td><td>IA</td></tr><tr><td> $H _ { t } ( x , p )$ </td><td>Optimized Hamiltonian,  $H _ { t } = \operatorname* { m a x } _ { u } \mathcal { H } _ { t }$ </td><td>IA</td></tr><tr><td> $p ( x ) , p _ { t } ( x )$ </td><td>Probability density and time-dependent marginal density. In Chapter  $\mathrm { I V } , p _ { 0 } , p _ { 1 }$  denote the base and target densities.</td><td>IB-IC</td></tr><tr><td> $p ^ { 0 } , p ^ { A }$ </td><td>Reference density and density inferred from observations A. Superscripts</td><td>IB</td></tr><tr><td> $\scriptstyle \mathrm { { D } _ { K L } } ( p | | q )$ </td><td>distinguish these from time subscripts. KL divergence, with averaging under its first argument; also used for path</td><td>IB</td></tr><tr><td> $\langle \cdot \rangle _ { p }$ </td><td>distributions. Expectation under the indicated density; conditioning is written inside the brackets. I B</td><td></td></tr><tr><td> $N$ </td><td>Also written  $\mathbb { E } _ { p } [ \cdot ]$  in Chapter V. Number of independent samples; in Sec.  $^ { \mathrm { ~ V ~ A , ~ } }$  N instead denotes the final decision index and M counts rollouts.</td><td>IB</td></tr><tr><td> $\hat { p }$ </td><td>Empirical density estimated from samples;  $\widehat { p } _ { \mathrm { d a t a } }$  denotes the training-set density in I B Sec. V C. Hats indicate estimates, not samples.</td><td></td></tr><tr><td>I</td><td>Large-deviation rate function.</td><td>IB</td></tr><tr><td> $\quad \boldsymbol { a } _ { m } ( \boldsymbol { x } ) , \boldsymbol { a } _ { m } [ \boldsymbol { x } ( \cdot ) ]$ </td><td>State observable and, in Sec. I C, trajectory observable.</td><td>IB-IC</td></tr><tr><td> $A _ { m }$ </td><td>Prescribed mean of observable  $a _ { m } .$ </td><td>IB</td></tr><tr><td> $H [ p ]$ </td><td>Shannon entropy with natural logarithms. In Sec.  ${ \mathrm { V } } \mathrm { A } , H [ \pi ( \cdot | x ) ]$ </td><td>IB</td></tr><tr><td> $M$ </td><td>entropy at state x. Number of imposed observable constraints; in Sec. V A, the number of independent I B</td><td></td></tr><tr><td> $\lambda _ { m }$ </td><td>rollouts. Multiplier enforcing the mean of observable  $a _ { m } .$  with the negative sign in the</td><td>IB</td></tr><tr><td> $Z$ </td><td>exponential tilt. Normalization constant or partition function. In Sec.  $ { \mathrm { I I B } } , Z ( x _ { 0 } )$  normalizes a transition kernel; in Sec. V A,  $\begin{array} { r } { Z _ { Q } ( x ) = \int e ^ { Q ( x , u ) / \alpha } } \end{array}$  du normalizes the Boltzmann</td><td>IB</td></tr><tr><td> $\mathcal { A } , \mathbf { 1 } _ { \mathcal { A } }$ </td><td>policy. Conditioning set and its indicator.</td><td>IB</td></tr><tr><td> $P ^ { 0 } , P ^ { u } , P ^ { u ^ { * } }$ </td><td>Reference, controlled, and optimally controlled path distributions. In Sec. is the joint state-action path law induced by a policy, and  $P ^ { \pi ^ { * } }$  is its optimal</td><td>IC</td></tr><tr><td>y</td><td>Prescribed terminal point in the Brownian-bridge examples.</td><td>IC</td></tr><tr><td> $f _ { t } ( x )$ </td><td>Passive drift; additive control gives  $g _ { t } ( x , u ) = f _ { t } ( x ) + u .$ </td><td>IC</td></tr><tr><td> $p _ { t } ^ { u } ( x )$ </td><td>Marginal density under  $P ^ { u } \colon$  denoted  $\rho _ { t } ( x )$  in Chapter II.</td><td>IC</td></tr><tr><td> $\langle \cdot \rangle _ { P ^ { u } }$ </td><td>Expectation over the indicated path distribution.</td><td>IC</td></tr><tr><td> $r _ { m }$ </td><td>Reward coefficient enforcing a trajectory constraint, with  $r _ { m } = - \lambda _ { m }$ </td><td>IC</td></tr><tr><td> $r _ { t } ( x )$ </td><td>State reward, excluding the separately displayed quadratic control cost.</td><td>IC</td></tr><tr><td> $\psi _ { t } ( x )$ </td><td>Exponentiated PISC value,  $\psi _ { t } = e ^ { V _ { t } }$  . Also denoted  $Z _ { t } ( x )$  during its derivation; the backward factor in Sec. II A.</td><td>ID</td></tr><tr><td> $R _ { t } [ x ( \cdot ) ]$ </td><td>Accumulated state and terminal rewards from t onward, excluding control cost.</td><td>ID</td></tr><tr><td> $\nabla \log p _ { t } ( x )$ </td><td>Spatial score of a marginal density; distinct from the conditioning field ∇ log  $\psi _ { t }$  and the protocol score ]  $\Theta _ { u }$  of Chapter III.</td><td>ID</td></tr><tr><td> $k _ { t \mid s } ( z | x )$ </td><td>Passive transition density from x at s to z at  $t > s ; k = k _ { 1 | 0 }$  in Sec. II B. In Chapter  $\operatorname { I V } , k _ { t }$  freezes the energy for one step; in Sec.  $\mathrm { V } \mathrm { A } , \dot { k } ( x \prime | x , u )$  is the environment transition kernel.</td><td>ID</td></tr></table>

TABLE II: Chapter II: Transport and Wasserstein Geometry.
<table><tr><td>Symbol</td><td>Meaning or convention</td><td>Section</td></tr><tr><td> $\rho _ { t } , \ \rho _ { 0 } , \ \rho _ { 1 }$ </td><td>Evolving density and prescribed endpoint densities. In Sec. V B,  $\begin{array} { r } { \widehat { \rho } _ { t } = N _ { \mathrm { p } } ^ { - 1 } \sum _ { i } \delta ( x - x _ { t } ^ { ( i ) } ) } \end{array}$  is the empirical particle density.</td><td> $\overline { { \Pi \mathrm { A } } }$ </td></tr><tr><td> $v _ { t } ( x )$ </td><td>Density-transport velocity:  $\begin{array} { r } { \partial _ { t } \rho _ { t } + \nabla \cdot \left( \rho _ { t } v _ { t } \right) = 0 . } \end{array}$  Generally distinct from  $u _ { t } ; \widehat { v } _ { t }$  a learned estimate in Sec.  $\mathrm { ~ V ~ C ~ } .$ </td><td>is II A</td></tr><tr><td> $\psi _ { t } ^ { \dagger } ( x )$ </td><td>Forward Schrödinger factor, with  $\rho _ { t } = \psi _ { t } \psi _ { t } ^ { \dagger }$  . The dagger does not denote complex II A conjugation.</td><td></td></tr><tr><td> $c ( x _ { 0 } , x _ { 1 } )$ </td><td>Cost of transporting unit mass between two endpoints.</td><td>ⅡIB</td></tr><tr><td> $T ( x _ { 0 } )$ </td><td>Transport map assigning a final position to an initial position. In Sec.  $\mathrm { V C } , T _ { t }$  is a  $\widehat { T } _ { t }$ </td><td>ⅡIB</td></tr><tr><td> $\pi ( x _ { 0 } , x _ { 1 } ) , \pi ^ { * }$ </td><td>flow map and a learned map, not necessarily optimal. Endpoint coupling and optimal coupling, with marginals  $\rho _ { 0 } , \rho _ { 1 }$  . Noise dependence is displayed as  $\pi _ { \varepsilon } ^ { * }$  when taking a limit.</td><td>ⅡB</td></tr><tr><td> $\varphi _ { 0 } , \varphi _ { 1 }$ </td><td>Endpoint dual potentials for entropic transport;  $\varphi _ { 1 } = \varepsilon r _ { 1 }$  with compatible</td><td>IB</td></tr><tr><td> $\zeta , \bar { \varphi } _ { 1 }$ </td><td>normalization. Their limits are Kantorovich potentials. Convex conjugate and shifted limiting endpoint potential:</td><td>IB</td></tr><tr><td> $C _ { 0 } , C _ { 1 }$ </td><td> $\bar { \varphi } _ { 1 } ( x ) = \| x \| ^ { 2 } / 2 - \varphi _ { 1 } ( x ) , T = \nabla \zeta .$  Cumulative distribution functions of the endpoint densities;  $C _ { i } ^ { - 1 }$  denotes the</td><td>IB</td></tr><tr><td> $\theta , p _ { \theta }$ </td><td>quantile function. Parameters and corresponding density in the geometric illustration; reused for the</td><td>ⅡIB</td></tr><tr><td> $W _ { 2 } ( \rho _ { 0 } , \rho _ { 1 } )$ </td><td>parametric VI family in Sec. V B. 2-Wasserstein distance; its square is the minimum mean squared displacement. The</td><td>ⅡB</td></tr><tr><td> $F ( x )$ </td><td>cost  $c = \| x _ { 0 } - x _ { 1 } \| ^ { 2 } / 2$  has minimum  $W _ { 2 } ^ { 2 } / 2$  Ordinary scalar objective in the Euclidean gradient-descent analogy.</td><td>IIC</td></tr><tr><td> $s ( x ) , s _ { t } ( x )$ </td><td>Density tangent; along a curve,  $s _ { t } = \partial _ { t } \rho _ { t } .$  with  $\textstyle \int s _ { t } d x = 0 .$ </td><td>ⅡIC</td></tr><tr><td> $\mathcal { W } _ { 2 }$ </td><td>Space of probability measures with finite second moment, equipped with</td><td>IIC</td></tr><tr><td> $\chi ( x )$ </td><td> $W _ { 2 } .$  Potential for the minimum-energy transport velocity, with  $\boldsymbol { v } = - \nabla \chi .$  In Sec. V B,</td><td>IIC</td></tr><tr><td> $\| s \| _ { \rho } , \ \langle s , s ^ { \prime } \rangle _ { \rho }$ </td><td> $\boldsymbol { v } _ { i } = - \nabla \chi _ { i }$  represents the parameter tangent  $\partial _ { \theta _ { i } } p _ { \theta } .$ </td><td></td></tr><tr><td> $\mathcal { F } [ \rho ]$ </td><td>Wasserstein tangent norm and inner product at density  $\rho .$  Functional of a density, such as free energy.</td><td>IIC ⅡIC</td></tr><tr><td> $\nabla _ { \mathrm { W } } \mathcal { F } [ \rho ]$ </td><td>Wasserstein gradient in its velocity representation; the descent velocity is its</td><td>IIC</td></tr><tr><td> $\delta \mathcal { F } / \delta \rho$ </td><td>negative. Functional derivative;  $\nabla _ { \mathrm { W } } \mathcal { F } = \nabla ( \delta \mathcal { F } / \delta \rho )$ </td><td></td></tr><tr><td> $\Phi ( x )$ </td><td>Dimensionless potential in the illustrative free energy with unit diffusion</td><td>IIC IIC</td></tr><tr><td>α</td><td>coefficient; reused in Sec. V B, with  $p _ { 1 } = e ^ { - \Phi } / Z _ { 1 }$  for VI. Convexity or geodesic-convexity parameter; positive values indicate strong</td><td></td></tr></table>

TABLE III: Chapter III: Thermodynamics.
<table><tr><td>Symbol</td><td>Meaning or convention</td><td>Section</td></tr><tr><td> $\overline { { S [ p ] } }$ </td><td>Thermodynamic entropy,  $\overline { { S [ p ] = k _ { \mathrm { B } } H [ p ] } }$  in the statistical description used here; distinct from mechanical action  $S [ x ( \cdot ) ]$ </td><td>ⅢA</td></tr><tr><td> $T , k _ { \mathrm { B } } , \beta$ </td><td>Bath temperature, Boltzmann constant, and inverse thermal energy  $\beta = ( k _ { \mathrm { B } } T ) ^ { - 1 }$ </td><td>ⅢIIA</td></tr><tr><td> $p ^ { \mathrm { e q } } , p _ { u } ^ { \mathrm { e q } }$ </td><td>Equilibrium density, including dependence on a fixed protocol u. In Chapter IV,  $p _ { t } ^ { \mathrm { e q } } = e ^ { - E _ { t } } / Z _ { t }$  is the prescribed equilibrium bridge.</td><td>ⅢA</td></tr><tr><td> $E ( x ) , E _ { u } ( x ) , E _ { t } ( x )$ </td><td>Microstate energy;  $E _ { t } = E _ { u _ { t } }$  for a protocol. General sampling algorithms in Chapter IV absorb β into  $E _ { t }$  and use dimensionless energy.</td><td>ⅢA</td></tr><tr><td> $\mathcal { F } ^ { \mathrm { e q } } , \mathcal { F } _ { u } ^ { \mathrm { e q } }$ </td><td>Equilibrium free energy,  $- k _ { \mathrm { B } } T$  log  $Z _ { u } ; - \log Z _ { u }$  in Chapter IV&#x27;s thermal units.</td><td>A</td></tr><tr><td> $\mathcal { E }$ </td><td>Mean internal energy,  $\langle E \rangle _ { p } .$ </td><td>ⅢIIA</td></tr><tr><td> $\Sigma , \dot { \Sigma } _ { t }$ </td><td>Total entropy production and its rate; distinct from the change  $\Delta S$  in system</td><td>ⅢB</td></tr><tr><td> $Q , W$ </td><td>entropy. Mean heat received by the system and mean work done on it.</td><td>ⅢB</td></tr><tr><td> $N _ { \mathrm { p } }$ </td><td>Particle number; distinct from N statistical samples. In Sec. V B, also the number</td><td>ⅢB</td></tr><tr><td> $X _ { m } , L _ { m n }$ </td><td>of neurons viewed as parameter particles. Macroscopic variables  $( m = 1 , \ldots , M )$  and Onsager transport coefficients.</td><td>ⅢB1</td></tr><tr><td>R</td><td>Rayleighian,  $\begin{array} { r } { \mathcal { R } = \frac { 1 } { 2 } T \dot { \Sigma } + \dot { \mathcal { F } } ; } \end{array}$  distinct from accumulated reward  $R _ { t } .$ </td><td>ⅢB1</td></tr><tr><td> $n _ { t } ( x )$ </td><td>Conserved particle number density, with  $\textstyle \int n _ { t } d x = N _ { \mathrm { p } }$  Its normalized form is  $\rho _ { t } = n _ { t } / N _ { \mathrm { p } }$ </td><td>Ⅲ B 1</td></tr><tr><td> $\Gamma ( n )$ </td><td>Dissipation or drag coefficient in the continuum Onsager description.</td><td>Ⅲ B 1</td></tr><tr><td> $\eta , \ r$ </td><td>Fluid viscosity and colloid radius in the Stokes-drag example; η here is distinct from white noise  $\eta _ { t }$ </td><td>Ⅲ B 1</td></tr><tr><td> $\mu _ { t } ( x )$ </td><td>Chemical potential,  $\delta \mathcal { F } / \delta n _ { t } ( x )$  ; for constant drag, relaxation has  $\chi _ { t } = \mu _ { t } / \Gamma$ </td><td>ⅢB1</td></tr><tr><td> $D$ </td><td>Physical diffusion coefficient,  $D = \varepsilon / 2$ </td><td>Ⅲ B 1</td></tr><tr><td> $F _ { t } ( x )$ </td><td>Physical force; the overdamped stochastic drift is  $f _ { t } = D \beta F _ { t }$ </td><td>Ⅲ B 2</td></tr><tr><td>T</td><td>Process duration when physical elapsed time is displayed explicitly</td><td>ⅢB 2</td></tr><tr><td> $\delta p _ { t } ( x )$ </td><td>Deviation from instantaneous equilibrium:  $p _ { t } = p _ { u _ { t } } ^ { \mathrm { e q } } + \delta p _ { t }$ </td><td>ⅢB3</td></tr><tr><td> $\mathcal { L } _ { u } , ~ \mathcal { L } _ { u } ^ { \dag }$ </td><td>Operators with convention  $\partial _ { t } p = - \mathcal { L } _ { u } p ; - \mathcal { L } _ { u } ^ { \dagger }$  is the backward generator. The dagger denotes the operator adjoint.</td><td>ⅢB3</td></tr><tr><td> $\Theta _ { u } ( x )$ </td><td>Protocol score,  $\partial _ { u } \log p _ { u } ^ { \mathrm { e q } } ( x ) ;$  a vector for a multicomponent protocol.</td><td>ⅢB3</td></tr><tr><td> $\zeta _ { u }$ </td><td>Integrated equilibrium score-correlation tensor defining the protocol-space metric. Its energy-unit friction normalization is  $k _ { \mathrm { B } } T \zeta _ { u } .$ </td><td>ⅢB3</td></tr><tr><td> $w _ { t } [ x ( \cdot ) ]$ </td><td>Work accumulated along one trajectory up to time  $t ; W = \langle w _ { 1 } \rangle _ { P }$  for the unit-time protocol. Dimensionless in Chapter  $\mathrm { I V } { \boldsymbol \mathbf { s } }$  sampling formulas</td><td>ⅢB4</td></tr><tr><td> $P , \ \bar { u } _ { t } , \ \bar { P }$ </td><td>Forward path law, reversed protocol  $\bar { u } _ { t } = u _ { 1 - t }$  , and reverse path law. In</td><td>ⅢB4</td></tr><tr><td>λ</td><td>Chapter  $\bar { \mathrm { I V } } , \bar { E } _ { t } = E _ { 1 - t }$  and  $x _ { t } ^ { \mathrm { r e v } } = x _ { 1 - t }$  denotes a reversed realization. Work-MGF transform variable in  $e ^ { \lambda w }$  ; inverse-energy units when w is physical</td><td>ⅢB4</td></tr><tr><td> $M _ { t } , \ \bar { M } _ { t } , \ \widetilde { M } _ { t }$ </td><td>work. Work-weighted endpoint density, reverse conditional MGF, and transformed reverse MGF used in the fluctuation-theorem proof.</td><td>ⅢB4</td></tr></table>

TABLE IV: Chapter IV: Sampling and Control.
<table><tr><td>Symbol</td><td>Meaning or convention</td><td>Section</td></tr><tr><td> $\overline { { p } } , p = \widetilde { p } / Z$ </td><td>Unnormalized density and the corresponding normalized probability density.</td><td>IVA</td></tr><tr><td> $d$ </td><td>Dimension of configuration space or number of model degrees of freedom.</td><td>IVA1</td></tr><tr><td> $\hat { a } _ { N }$ </td><td>Monte Carlo estimate of  $\langle a \rangle$  from N independent samples  $x _ { n } .$  For an ensemble of paths,  $x _ { t } ^ { ( n ) }$  labels time and run separately.</td><td>IVA1</td></tr><tr><td> $\sigma _ { i } , m ( x )$ </td><td>Ising spin and configuration magnetization, m  $\begin{array} { r } { { \boldsymbol \mathrm { \Sigma } } ( { \boldsymbol { x } } ) = d ^ { - 1 } \sum _ { i } { \boldsymbol \sigma } _ { i } . } \end{array}$ </td><td>IVA2</td></tr><tr><td> $\omega , \widetilde { \omega }$ </td><td>Normalized importance ratio and unnormalized weight. In  $\mathbf { A I S } , \widetilde { \omega } _ { t } = e ^ { - w _ { t } }$  and  $\omega _ { t } = ( Z _ { 0 } / Z _ { t } ) \bar { \widetilde { \omega } } _ { t }$ </td><td>IVA3</td></tr><tr><td> $N _ { \mathrm { e f f } }$ </td><td>Effective sample size estimated from the weights,  $\begin{array} { r } { ( \sum _ { n } \widetilde { \omega } _ { n } ) ^ { 2 } / \sum _ { n } \widetilde { \omega } _ { n } ^ { 2 } . } \end{array}$ </td><td>IVA3</td></tr><tr><td> $q _ { t } , \bar { q } _ { t }$ </td><td>Actual forward and reverse sampler densities, distinct from prescribed densities. In  $\mathrm { A I S } , q _ { 0 } = p _ { 0 } , \bar { q } _ { 0 } = p _ { 1 } ; \mathrm { i n S e c . } \mathrm { V C } , q _ { t } = ( \widehat { T } _ { t } ) _ { * } p _ { 0 }$  is the learned model&#x27;s density.</td><td>IVA3</td></tr></table>

TABLE V: Chapter V: Applications in Learning.
<table><tr><td>Symbol</td><td>Meaning or convention</td><td>Section</td></tr><tr><td> $\overline { { { \mathcal { X } } , { \mathcal { U } } } }$ </td><td>RL state and action spaces.</td><td>VA1</td></tr><tr><td> $\pi ( u | x )$ </td><td>Policy: conditional action distribution; time is included in the state for finite-horizon problems. Later,  $\pi _ { Q }$  denotes the Boltzmann policy defined by a </td><td>VA1</td></tr><tr><td> $n , N$ </td><td>critic. Decision step and final decision index;  $n = 0 , \ldots , N$  and  $t _ { n } = n \Delta t .$ </td><td>VA1</td></tr><tr><td> $Q ^ { \pi } ( x , u ) , Q ^ { * } ( x , u )$ </td><td>Exact policy-specific and optimal action values; hats denote learned tabular estimates. In MaxEnt RL, these denote entropy-regularized  $\mathrm { ( ^ { 6 6 } s o f t ^ { , 9 } ) }$  values.</td><td>VA1</td></tr><tr><td> $\theta , \ Q _ { \theta } ^ { \pi } , \ Q _ { \theta } ^ { * }$ </td><td>Neural-network estimates of policy-specific and optimal values; θ denotes the network weights.</td><td>VA2</td></tr><tr><td> $i , M$ </td><td>Rollout sample index and number of independent rollouts,  $i = 1 , \ldots , M .$ </td><td>VA2</td></tr><tr><td> ${ \mathcal { I } } [ \pi ]$ </td><td>Expected RL return; the text specifies discounting and any entropy bonus.</td><td>VA2</td></tr><tr><td> $\gamma$ </td><td>Reward discount factor.</td><td>VA2</td></tr><tr><td> $_ y$ </td><td>Sampled Bellman target used to update a critic.</td><td>VA2</td></tr><tr><td> $\xi$ </td><td>Learning rate.</td><td>VA2</td></tr><tr><td> $\mathcal { D }$ </td><td>Replay buffer of observed transitions and rewards.</td><td>VA2</td></tr><tr><td> $L ( \theta ) , \mathcal { L } [ \rho ]$ </td><td>Loss as a function of parameters or a functional of a density, respectively.</td><td>VA2</td></tr><tr><td> $\bar { \theta }$ </td><td>Parameters held fixed when evaluating a critic target; target-network updates, if used, are specified separately.</td><td>VA2</td></tr><tr><td>α</td><td>RL entropy temperature in reward units; distinct from the convexity parameter in Wasserstein geometry.</td><td>VA3</td></tr><tr><td> $\pi ^ { 0 } ( u \vert x ) , \ c _ { 0 }$ </td><td>Reference action distribution and its constant value in the uniform case.</td><td>VA3</td></tr><tr><td> $P ^ { * }$ </td><td>Unrestricted reward-tilted target path law; generally distinct from  $P ^ { \pi ^ { * } }$ </td><td>VA3</td></tr><tr><td> $\phi , \pi _ { \phi }$   $\Phi ^ { ( N _ { \mathrm { p } } ) } , \mathcal { V } [ \rho ]$ </td><td>Actor parameters and corresponding parameterized policy.</td><td>VA4</td></tr><tr><td></td><td>Many-particle potential and its density-functional representation, with the normalization displayed in Sec. V B 2.</td><td>VB2</td></tr><tr><td> $U ( x , y )$ </td><td>Pair-interaction potential, also used for interactions between parameter particles.</td><td>VB2</td></tr><tr><td> $f ^ { N _ { \mathrm { p } } } ( x ; \Theta _ { t } )$   $\Theta _ { t } , \ \theta _ { t } ^ { ( i ) } , \ w _ { t } ^ { ( i ) } , \ z _ { t } ^ { ( i ) }$ </td><td>Finite-width neural-network predictor; distinct from the passive drift  $f _ { t } .$ </td><td>VB2</td></tr><tr><td></td><td>Full parameter collection, neuron parameters, feature weights and internal feature parameters;  $\theta = ( w , z )$ </td><td>VB2</td></tr><tr><td> $\varphi ( x ; z )$ </td><td>Nonlinear feature; its weighted form is the observable  $a ( x ; \theta ) = w \varphi ( x ; z )$ </td><td>VB2</td></tr><tr><td> $y ( x ) , \nu ( x )$ </td><td>Target regression function and input-data density; distinct from the Bellman target  $y .$ </td><td>VB2</td></tr><tr><td> ${ \mathcal { M } } , \Omega , d _ { \theta }$ </td><td>Parametric density family, parameter domain and parameter dimension.</td><td>VB3</td></tr><tr><td> $g _ { i j }$ </td><td>Parameter-space metric induced by Wasserstein geometry.</td><td>VB3</td></tr><tr><td> $\mathrm { p r o j } _ { \mathcal { M } }$ </td><td>Orthogonal projection of a velocity onto the tangent space of the parametric density family.</td><td>VB3</td></tr><tr><td> $\operatorname { B W } ( \mathbb { R } ^ { d } )$  ∇BW</td><td>Gaussian Bures-Wasserstein space and its projected gradient.</td><td>VB3</td></tr><tr><td> $m , \Sigma , \mathbb { S } _ { + + } ^ { d }$ </td><td>Gaussian mean, covariance and space of symmetric positive-definite covariance matrices.</td><td>VB3</td></tr><tr><td> $h , k _ { h } , \mathcal { K } _ { h } ^ { p }$ </td><td>Smoothing bandwidth, spatial kernel and density-weighted smoothing operator.</td><td>VB3</td></tr><tr><td> $T _ { \ast } p$ </td><td>Pushforward of a distribution through a map  $T .$ </td><td>VC1</td></tr><tr><td> $\hat { v }$ </td><td>Estimated probability flow field.</td><td>VC2</td></tr><tr><td> $y _ { t }$ </td><td>Sample-generating process in flow and diffusion models.</td><td>VC2</td></tr><tr><td> $x _ { t }$ </td><td>Stochastic interpolant defined from training samples.</td><td>VC2</td></tr><tr><td> $\mathcal { L } _ { \mathrm { F M } } [ \widehat { v } ]$ </td><td>Flow-matching loss.</td><td>VC2</td></tr><tr><td> $c _ { j }$ </td><td>Gaussian-mixture weights, with  $c _ { j } \geq 0$  and  $\textstyle \sum _ { j } c _ { j } = 1 .$ </td><td>VC2</td></tr><tr><td> $\widehat { a } , \widehat { b }$ </td><td>Learned scale and shift functions in an affine normalizing-flow layer.</td><td>VC5</td></tr></table>