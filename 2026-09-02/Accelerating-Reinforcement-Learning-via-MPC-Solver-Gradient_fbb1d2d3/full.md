# Accelerating Reinforcement Learning via MPC Solver-Gradient Guidance for Weights-varying MPC

Baha Zarrouki, Arslan Thobani, Jasper Hoffmann, Mattia Piccinini, Rudolf Reiter, Felix Jahncke, Sebastien Gros, Davide Scaramuzza, and Johannes Betz´

Abstract—In Model Predictive Control (MPC), cost-function weights shape closed-loop behavior, yet changing conditions often make fixed parametrizations suboptimal and motivate contextdependent online adaptation. Learning such policies is difficult because behavior depends implicitly on numerical MPC solutions, producing nonlinear, potentially nonsmooth, long-horizon dependencies on policy parameters. This creates a bias–variance tradeoff: Reinforcement Learning (RL) optimizes realized closedloop return from environment samples but is sample-inefficient, whereas Gradient-Based Policy Learning (GB-PL) uses lowvariance solver gradients from differentiable MPC to optimize surrogate losses on predicted trajectories but can be biased under model mismatch. We propose Solver-Gradient Guided Reinforcement Learning (SG-RL), a solver-sensitivity augmentation for RL-based online MPC cost-weight adaptation. SG-RL keeps sampled closed-loop return as the objective and uses bounded solver-derived gradients as auxiliary guidance to improve stability and sample efficiency. We instantiate SG-RL in Proximal Policy Optimization (PPO) with four modular algorithms that inject solver-gradient guidance into actor-update scaling, policy loss, advantage estimation, and value-function learning. On two full-scale autonomous racing platforms with intentional model mismatch, SG-RL reaches PPO’s best closed-loop return with up to 70.6% fewer samples, outperforms GB-PL baselines by at least 54% in closed-loop return, and generalizes zero-shot to unseen environments.

Index Terms—Learning and Adaptive Systems; Robust/ Adaptive Control of Robotic Systems; Optimization and Optimal Control; Reinforcement Learning for MPC

## I. INTRODUCTION

Model Predictive Control (MPC) is a powerful framework for constrained optimal control of complex dynamical systems [1]. Its closed-loop performance depends heavily on the formulation of the optimization problem, including the system model, constraints, and cost function. In many applications, the model and constraints are fixed to preserve safety guarantees. The cost function remains a primary mechanism for shaping controller closed-loop behavior and trading off competing objectives [2]. However, a single fixed set of cost-function weights is often insufficient when operating conditions change. Different scenarios may require different trade-offs between objectives, making the relationship between cost weights and closed-loop performance highly context dependent. This observation has motivated the development of Weights-varying MPC approaches [3], [4], [5], which adapt the cost-function weights online while keeping the underlying model and constraints unchanged. Such adaptation is particularly relevant in safety-critical and highly dynamic applications. A prominent example is the combined nonlinear longitudinal-lateral control problem in autonomous vehicle racing [6], where operating near the handling limits requires different performance tradeoffs across racetrack segments.

![](images/0ac2e99aabf89bffeadb45310f3e8a88da0b5e8ba71749c39d980f8dc46faffd.jpg)  
Fig. 1: Solver-Gradient Guided Reinforcement Learning (SG-RL) learns an online MPC weight-adaptation policy $\pi _ { \phi }$ by combining environment-based policy-gradient learning $\nabla _ { \phi } \mathcal { J } _ { \mathrm { R I } }$ with model-based solver-gradient guidance $\nabla _ { \phi } \mathcal { J } _ { \mathrm { S G } }$ The update preserves closed-loop return optimization while using solver sensitivities as bounded, model-informed guidance for improved sample efficiency and performance.

Learning such policies is challenging because it induces a bilevel optimization structure [7]. Specifically, the return depends on the policy parameters only indirectly, through the numerical solution of a constrained MPC; decisions at one time step affect future states and therefore future MPC solutions, resulting in long-horizon credit-assignment problems; the MPC solver introduces implicit, nonlinear, and potentially non-smooth mappings that complicate gradient computation; and the policy must be learned in closed-loop interaction with the controlled system.

Existing approaches for MPC cost-weight optimization can broadly be divided into two categories. The first computes a single fixed parameter vector offline, whereas the second learns a state-dependent weight-adaptation policy online. Classical offline tuning methods, such as genetic algorithms and Bayesian optimization [8], [9], [10], [11], can identify useful fixed weights but do not address online adaptation to changing conditions. Since this work addresses online adaptation, we focus on the latter category. In the taxonomy of [12], this corresponds to a hierarchical Reinforcement Learning (RL)– MPC architecture in which the MPC is part of the actor and a neural policy supplies the MPC parameters. Within this setting, recent work distinguishes two complementary sources of learning signal: methods that build environmentbased gradients from sampled closed-loop interactions (blackbox learning), and methods that compute model-based solver gradients by differentiating through the MPC solver (whitebox learning).

The first family learns online cost-weight adaptation policies using environment-based gradients. The environment-based baselines considered here construct policy updates from sampled closed-loop returns, treating the MPC controller as part of the environment [3], [4], [13], [14]. RL has also been applied more broadly to learning fixed MPC parameters, such as terminal costs or prediction horizons [15], [14], while the concept of Weights-varying MPC introduced learning state-dependent cost-weight adaptation policies [3], [4]. More generally, MPC has been incorporated as a structured policy or value-function approximator within actor-critic architectures, deterministic and stochastic policy-gradient methods, economic MPC, learning-based MPC formulations, and joint system-identification and control frameworks [16], [17], [18], [19], [20], [21], [22], [23], [24]. These baselines benefit from long-horizon credit assignment and can use reward signals defined by broad classes of closed-loop objectives, including non-smooth or sparse terms. However, their updates still rely on sampled closed-loop interactions and inherit the usual approximation, finite-horizon, and estimator-bias issues of practical policy-gradient methods.

The second family learns MPC parameters using modelbased solver gradients obtained from differentiable optimization. Building on task-based learning through optimization problems [25], differentiable optimization layers [26], [27], and differentiable MPC [28], recent advances enable efficient backpropagation through increasingly general MPC formulations, including constrained nonlinear MPC [29], [30]. Until recently, such approaches focused almost exclusively on optimizing fixed MPC parameters through solver sensitivities, including DiffTune-MPC [31], differentiable actor learning [32], cost-matching approaches [33], and ZipMPC [34]. Only very recently, gradient-based policy learning (Gradient-Based Policy Learning (GB-PL)) for nonlinear MPC [5] extended the differentiable nonlinear MPC to learn online cost-weight adaptation policies. Our GB-PL baseline is built upon [5] and uses a recent backpropagation software feature in acados [29] to efficiently compute solver gradients for nonlinear constrained MPC. In this view, GB-PL plays a cost-matching role: it aligns a differentiable MPC surrogate loss with task-level performance terms on trajectories generated by the predictive model. Unlike environment-based RL updates, these methods exploit local first-order gradients of the differentiable surrogate under solver-regularity assumptions, resulting in substantially higher sample efficiency when the surrogate is informative.

Despite their sample efficiency, differentiable-solver methods optimize model-based surrogate objectives whose gradients are only as accurate as the predictive model and therefore may become misleading under model mismatch. Furthermore, they generally lack the exploration mechanisms and longhorizon value estimation that make RL effective for optimizing the performance of the realized closed-loop. Conversely, our environment-based PPO baseline does not use the rich firstorder information available from differentiable MPC solvers. The methods compared in this work therefore emphasize one of two complementary information sources: either sampled closed-loop returns or solver sensitivities. Environment-based gradients more directly reflect the realized closed-loop objective but can suffer from high variance, whereas solver gradients provide lower-variance local guidance but inherit bias from the predictive model. Their complementary properties motivate the combination of both within a single learning framework. This gap motivates the central question of this paper: how can environment-based and model-based solver gradients be combined effectively for learning online MPC cost-weight adaptation policies?

This question is related to a broader family of guided, structured, and hybrid reinforcement-learning methods that combine model-free policy updates with model-based gradients, planning modules, auxiliary losses, or teacher signals [35], [36], [37], [38], [39], [40], [41], [42], [43]. These methods demonstrate that combining complementary gradient estimators can substantially improve learning efficiency. However, these works do not use differentiable constrained nonlinear MPC solver sensitivities as structured auxiliary gradients for policy optimization over online MPC cost weights.

Within the differentiable MPC literature, solver sensitivities have also been used beyond direct parameter optimization. Adhau et al. [44] employ nonlinear-program sensitivities to approximate action-value quantities in a Q-learning framework, thereby reducing the number of MPC evaluations required during learning. This approach, however, neither learns an online cost-weight adaptation policy nor uses solver sensitivities to guide policy optimization itself. Romero et al. [32] propagate solver gradients through a differentiable linear MPC layer embedded in a PPO actor for quadrotor control, while related hybrid schemes study differentiable simulation gradients [45] or use RL to provide NMPC reference guidance [46]. In con trast, our differentiable nonlinear MPC-based work does not replace environment-based policy gradients with model-based solver gradients, nor does it guide the controller only through references. Instead of viewing solver gradients as an alternative optimization objective, we use them as auxiliary signals inside PPO. Consequently, the optimization objective remains the sampled closed-loop return, while solver sensitivities provide local model-informed search directions for online cost-weight adaptation with constrained nonlinear MPC.

The contributions of this work are threefold:

1. Solver-gradient Guide for online MPC weight adaptation. We introduce Solver-Gradient Guided Reinforcement Learning (SG-RL), a framework for online cost-weight adaptation in constrained nonlinear MPC. Compared to RL, which learns from environment-based policy gradients estimated from sampled closed-loop returns, and differentiable-solver approaches that optimize model-based surrogate objectives using solver sensitivities, SG-RL uses solver-gradient guidance as an auxiliary signal for RL while still optimizing the realized closed-loop return (Sec. VI).

2. Four methods for guiding Proximal Policy Optimization (PPO) through solver sensitivities, We instantiate this mechanism through four modular SG-RL algorithms that integrate the same solver-gradient guidance into distinct PPO components: actor-update scaling, performance-gated auxiliary policy learning, solver-guided advantage estimation, and solver-aware critic learning. These methods provide four complementary instantiations of the SG-RL framework and are evaluated separately to understand which forms of solver guidance are most effective in practice (Sec. VI).

3. Full-scale validation under model mismatch. We demonstrate our SG-RL on a closed-loop high-dimensional, highly nonlinear control task: nonlinear MPC for combined longitudinal and lateral vehicle control on two full-scale, highfidelity autonomous racing simulation platforms, following racelines near the vehicle handling limits under intentional model mismatch. In this setting, all SG-RL variants outperform the PPO baseline in peak closed-loop performance, reach PPO’s best return with up to 70.6 % fewer training samples, and outperform the GB-PL baselines by at least 54 % in return, while also demonstrating zero-shot generalization and interpretable weight-adaptation strategies (Sec. VIII-IX-B).

Notation and Background. We denote by $\pi _ { \phi } ( \cdot \mathrm { ~ \bf ~ \vert ~ } \sigma _ { t } )$ a stochastic actor that maps the observation $\mathbf { } _ { o _ { t } }$ to a distribution over MPC weights, with mean $\mu _ { \phi } ( o _ { t } )$ , while $V _ { \psi } ( o _ { t } )$ denotes the critic and provides the value function. We use the generalized-advantage estimate $\hat { A } _ { t }$ [47] to quantify whether the sampled weight vector performed better or worse than expected. With TD residual $\delta _ { t } : = r _ { t } + \gamma V _ { \psi } ( \pmb { o } _ { t + 1 } ) - V _ { \psi } ( \pmb { o } _ { t } )$ we write

$$
\hat { A } _ { t } = \delta _ { t } + \gamma \lambda ( 1 - d _ { t + 1 } ) \hat { A } _ { t + 1 } ,\tag{1}
$$

where $\gamma$ is the discount factor and λ is the GAE parameter [47]. PPO updates the actor through the clipped surrogate objective

$$
\begin{array} { r l } & { \mathcal { I } _ { \mathrm { P P O } } ( \phi ) = \mathbb { \hat { E } } _ { t } \Big [ \operatorname* { m i n } \big ( \rho _ { t } ( \phi ) \hat { A } _ { t } , } \\ & { \qquad \mathrm { c l i p } \big ( \rho _ { t } ( \phi ) , 1 - \epsilon , 1 + \epsilon \big ) \hat { A } _ { t } \big ) \Big ] , } \end{array}\tag{2}
$$

with sampled action $\begin{array} { r l r } { \mathbf { a } _ { t } } & { { } \equiv } & { \pmb { \theta } _ { t } . } \end{array}$ , ratio $\rho _ { t } ( \phi ) ~ = ~ \pi _ { \phi } ( a _ { t } ~ |$ ${ \mathbf { \Gamma } } _ { { \mathbf { \Gamma } } _ { \mathbf { \Gamma } } } \mathbf { \Sigma } _ { \mathbf { \Gamma } } \mathbf { \Sigma } _ { \mathbf { \mathcal { ( } \mathbf { \mathbf { \Gamma } } _ { \mathbf { \Phi } _ { t } } \mathbf { \Sigma } ) } } / \pi _ { \phi _ { \mathrm { o l d } } } ( \mathbf { \Delta } \mathbf { a } _ { t } \mathbf { \Sigma } \mid \mathbf { \Delta } _ { \mathbf { \Gamma } } \mathbf { \Sigma } _ { \mathbf { \mathbf { ( } \mathbf { \mathbf { \Gamma } } _ { t } ) } }$ , and clipping range ϵ. For the critic, we use the standard value-regression term based on the fixed target $\hat { R } _ { t } ^ { \lambda } : = \hat { A } _ { t } + V _ { \psi _ { \mathrm { o l d } } } ( \pmb { o } _ { t } )$

Actor and critic are then optimized through the standard combined PPO loss

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { P P O } } ( \phi , \psi ) = - \mathcal { I } _ { \mathrm { P P O } } ( \phi ) + c _ { V } \mathcal { L } _ { V } ( \psi ) } \\ { - c _ { \mathrm { e n t } } \hat { \mathbb { E } } _ { t } \big [ \mathcal { H } ( \pi _ { \phi } ( \cdot | \pmb { o } _ { t } ) ) \big ] , } \end{array}\tag{3}
$$

where $c _ { V }$ and $c _ { \mathrm { e n t } }$ weight the value and entropy terms, respectively. The standard RL update used as baseline later with learning rate $\eta$ is

$$
\mathcal { U } _ { \mathrm { R L } } : ( \phi _ { k + 1 } , \psi _ { k + 1 } ) = ( \phi _ { k } , \psi _ { k } ) - \eta \nabla _ { \phi , \psi } \mathcal { L } _ { \mathrm { P P O } } ( \phi , \psi )\tag{4}
$$

## II. PROBLEM FORMULATION

We formalize weights-varying Nonlinear Model Predictive Control (NMPC) as a hierarchical decision-making problem (Fig. 1). During deployment, a low-level NMPC controller solves a constrained finite-horizon optimal control problem at each sampling instant, while a high-level learning policy adapts the NMPC cost weights online along the task domain, thereby providing a mechanism to shape long-horizon behavior. During learning, the same closed-loop architecture is rolled out to collect trajectories, and an outer optimization loop updates the policy parameters across training iterations.

## A. Weights-varying Nonlinear MPC

1) Motivation: Why Adapt Cost Weights?: Standard NMPC typically relies on fixed cost weights tuned offline to balance competing objectives such as tracking accuracy and control effort. In practice, the best trade-off is context-dependent: operating regime, disturbance level, uncertainty, and anticipated task difficulty change over time, so a single static tuning is inevitably conservative in some conditions and too aggressive in others. We therefore adapt the cost weights online, which reshapes the optimization landscape and the relative emphasis placed on these objectives without modifying the underlying model or constraints.

2) Weights-varying NMPC OCP Formulation: The OCP solved at time t with a receding prediction horizon N is:

$$
\operatorname* { m i n } _ { { \bf x } , { \bf u } } ~ J ( { \bf x } , { \bf u } , \pmb { \theta } _ { t } ) = \sum _ { k = 0 } ^ { N - 1 } l ( { \bf x } _ { k } , { \bf u } _ { k } , \pmb { \theta } _ { t } ) + l _ { N } ( { \bf x } _ { N } , \pmb { \theta } _ { t } )\tag{5a}
$$

$$
\begin{array} { r } { \mathrm { s . t . } \qquad \mathbf { x } _ { 0 } = \mathbf { s } _ { t } , } \end{array}\tag{5b}
$$

$$
\begin{array} { r } { { \mathbf { x } } _ { k + 1 } = { \pmb f } ( { \mathbf { x } } _ { k } , { \mathbf { u } } _ { k } ) , \qquad k = 0 , \ldots , N - 1 , } \end{array}\tag{5c}
$$

$$
0 \leq h ( \mathbf { x } _ { k } , \mathbf { u } _ { k } ) , \qquad k = 0 , \ldots , N - 1 ,\tag{5d}
$$

$$
0 \leq h _ { N } ( \mathbf { x } _ { N } )\tag{5e}
$$

where $\mathbf { x } = [ \mathbf { x } _ { 0 } , \ldots , \mathbf { x } _ { N } ]$ and $\mathbf { u } = [ \mathbf { u } _ { 0 } , \ldots , \mathbf { u } _ { N - 1 } ]$ are the state and control trajectories, while $\mathbf { s } _ { t }$ denotes the current state estimate available to the controller at time t. The stage cost $l ( \cdot )$ and the terminal cost $l _ { N } ( \cdot )$ are parameterized by a weight vector $\pmb { \theta } _ { t } \in \mathbb { R } ^ { n _ { \theta } }$ , which is fixed on the prediction horizon at time t but may change between sampling instants. The function $f ( \cdot )$ in (5c) is the internal prediction model of the controller. In this work, we keep $f ( \cdot )$ fixed and adapt only the cost parameters. The functions (5d)-(5e) provide inequality constraints on the states and controls.

3) Learning-based Cost-Weight Adaptation Policies: We introduce a high-level policy $\pi _ { \phi }$ (parameterized by ϕ) that maps an observation $\mathbf { } _  \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { }$ to a distribution over NMPC weight vectors (Fig. 1). The applied weight vector is denoted $\theta _ { t }$ and is equivalent to the RL action $\mathbf { \delta } _ { \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } } \mathbf { \delta } _  \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathrm { \langle } \mathbf { \delta } _ { \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha \alpha } }$ , while $\mu _ { \phi } ( o _ { t } )$ denotes the policy mean used for deterministic evaluation and solverguided policy updates. The policy is evaluated online both during training rollouts and during deployment; the difference between the two phases is only whether $\phi$ is updated.

At each time t, the lower-level NMPC solves (5) with the current state $\mathbf { s } _ { t }$ and the set of parameters $\pmb { \theta } _ { t } ,$ , then applies the first input $\mathbf { u } _ { t } : = \mathbf { u } _ { 0 } ^ { * } ( \mathbf { s } _ { t } , \pmb \theta _ { t } )$ to the system in a receding-horizon fashion.

B. Problem Statement: Learning Policies for Weights-varying NMPC

Our objective is to learn the parameters $\phi$ of the highlevel weight-selection policy to optimize long-horizon closedloop performance. Let $\mathcal { L } _ { \mathrm { p e r f } } ( \mathbf { s } _ { t } , \mathbf { u } _ { t } )$ denote a user-defined instantaneous performance loss evaluated on the realized closed-loop state and the control actually applied at time t (Fig. 3). This high-level loss can differ from the NMPC internal cost J and may include objectives that are difficult to encode directly in the NMPC, including non-smooth, sparse, or history-dependent terms such as collisions, failure-to-finish, or penalties on rapid weight variation.

The objective is to find the optimal policy parameters $\phi ^ { * }$ that minimize the expected cumulative loss over a closed-loop episode of length $T ,$ , where the expectation is taken over the trajectories induced by the policy, the NMPC controller, the true rollout environment, and exogenous stochasticity:

$$
\phi ^ { * } = \underset { \phi } { \arg \operatorname* { m i n } } \ \mathbb { E } _ { \tau \sim p _ { \phi } ^ { \mathrm { c l } } } \left[ \sum _ { t = 0 } ^ { T } \mathcal { L } _ { \mathrm { p e r f } } ( \mathbf { s } _ { t } , \mathbf { u } _ { t } ) \right] ,\tag{6}
$$

Learning the high-level weight-selection policy is challenging because the return depends on $\phi$ only indirectly through the closed-loop interaction of the policy $\pi _ { \phi } ,$ , the numerical solution of a constrained NMPC problem at each time step, and the true plant or rollout-environment dynamics. Decisions at time t therefore affect future states, observations, and solver calls, which induces long-horizon credit assignment during training.

The following sections develop three approaches for solving the upper-level optimization problem in (6). We first introduce the neural policy architecture used throughout the paper (Sec. III), then present two baseline optimization paradigms based on differentiable optimization (Sec. IV) and reinforcement learning (Sec. V), before introducing our proposed Solver-Gradient Guided Reinforcement Learning framework (Sec. VI), which combines the strengths of both.

## III. DESIGNING A NEURAL POLICY FOR WEIGHTS-VARYING NMPC

Weights-varying NMPC depends on a policy $\pi _ { \phi }$ that maps high-dimensional observations to effective cost configurations. This mapping is implicit, nonlinear, and shaped by NMPC constraints, so fixed offline weights are insufficient when task requirements and disturbances change quickly.

We design a neural meta-controller in the NMPC hyperparameter space. At each step, it maps $\mathbf { } _ { o _ { t } }$ to weights $\theta _ { t }$ (Figs. 1, 2) for the next planning horizon. This hierarchical decomposition allows the NMPC to handle system constraints and low-level dynamics, while the neural policy focuses on shaping the broader behavioral objectives.

## A. Policy Architecture: History-Aware MLP

The main design choices are the policy input $\mathbf { } _ { o _ { t } }$ and output $\pmb \theta _ { t } \colon$ the input must support reaction, anticipation, and mismatch adaptation, while the output is restricted to valid NMPC weight ranges. This reduces the learning search space and improves robustness. We parameterize our policy $\pi _ { \phi }$ as a feed-forward neural network. The policy and NMPC share the online sensing/estimation pipeline: NMPC initializes the OCP with $\mathbf { s } _ { t } ,$ and the policy receives features $\mathbf { } _ { o _ { t } }$ from measured/estimated quantities, reference previews, and recent performance. To capture temporal context without the complexity of Recurrent

![](images/8ce2bdacc4f1922fa680aaafa797095023f05eb784e5ad0ab08536e74b4e2ccb.jpg)  
Fig. 2: Architecture of the online cost-weight policy.

Neural Networks (RNNs), we explicitly structure $\mathbf { } _ { o _ { t } }$ into three functional groups (Fig. 2):

1. Current measurable context (Reactive): task-relevant local state information in a task-relative frame, enabling reactive decisions.

2. Reference preview / task context (Proactive): a shorthorizon lookahead $\mathbf { r } _ { t : t + H _ { \mathrm { a n } } }$ of the reference quantities used by the NMPC, such as target speed and path geometry, so weights change before braking, curvature, or acceleration events.

3. Performance history (Adaptive): recent closed-loop tracking terms, e.g., squared lateral and velocity errors, summarizing whether the current schedule works and helping infer persistent disturbances, model mismatch, or actuator transients.

## B. Action Space: Constrained Cost Parameterization

Our policy outputs cost weights $\pmb { \theta } _ { t } ,$ , which must remain positive and well scaled. Arbitrary weights can destroy objective conditioning or strict convexity for diagonal quadratic terms, yield unbounded solutions, or drive training into numerically unstable magnitudes.

Instead of a matrix-level projection onto the positivedefinite cone, we use a Constrained Cost Parameterization: each weight is bounded within strictly positive limits. These conservative box constraints are simple, robust, and enforce positive-definiteness for diagonal costs. Domain knowledge sets $[ \theta _ { \mathrm { m i n } } , \theta _ { \mathrm { m a x } } ]$ , and unconstrained logits $\mathbf { q } _ { t } ~ \in ~ \mathbb { R } ^ { n _ { \theta } }$ are mapped to this range by a scaled tanh:

$$
\theta _ { t } = \Gamma ( { \bf q } _ { t } ) = \frac { \theta _ { \operatorname* { m a x } } - \theta _ { \operatorname* { m i n } } } { 2 } \operatorname { t a n h } ( { \bf q } _ { t } ) + \frac { \theta _ { \operatorname* { m a x } } + \theta _ { \operatorname* { m i n } } } { 2 } .\tag{7}
$$

This mechanism serves two purposes:

1. Search-space regularization: confines optimization to a bounded hyper-rectangle, preventing divergence and removing irrelevant parameter regions.

2. Smooth bounded mapping: keeps the parameterization differentiable and avoids hard clipping, at the cost of attenuated gradients near boundaries.

## IV. GB-PL FOR WEIGHTS-VARYING DIFFERENTIABLE NMPC

This section presents the GB-PL baseline for weightsvarying differentiable NMPC. Here, GB-PL refers to the setting in which a differentiable surrogate loss is backpropagated through the NMPC layer and the neural policy. We use both the

TABLE I: Comparison of Policy Architectures for Weightsvarying NMPC. Our proposed policy uniquely combines reactivity, adaptability, and proactive anticipation within a constrained action space.
<table><tr><td>Method</td><td></td><td></td><td></td><td>Reactive Adaptive Proactive Action Space</td></tr><tr><td>[3]</td><td>√</td><td>√</td><td>X</td><td>Bounded (Sigmoid)</td></tr><tr><td>[4]</td><td>√</td><td>√</td><td>√</td><td>Bounded, discrete</td></tr><tr><td>[5]</td><td>X</td><td>X</td><td>√</td><td>Unbounded (Softplus)</td></tr><tr><td>Ours</td><td>√</td><td>√</td><td>√</td><td>Bounded (Tanh)</td></tr></table>

prior reduced GB-PL formulation [5] and our new extended implementation as baselines for evaluating the SG-RL methods introduced later in Section VI.

## A. Differentiating the NMPC Solution

To propagate gradients through the NMPC solver, we follow the sensitivity analysis of nonlinear programs in [29]. Let $\mathbf { w } ^ { * } ( \pmb { \theta } _ { t } ) = [ \mathbf { z } ^ { * } , \pmb { \lambda } ^ { * } , \pmb { \mu } ^ { * } ]$ denote the optimal primal-dual solution vector of the OCP (5) given weights $\theta _ { t }$ , where $\mathbf { z } ^ { \ast }$ contains the optimal states and controls, and $\lambda ^ { * } , \mu ^ { * }$ are the Lagrange multipliers associated with the equality and inequality constraints. The optimality of $\mathbf { w } ^ { * }$ is defined by the Karush-Kuhn-Tucker (KKT) conditions, which can be expressed as a system of nonlinear equations $R ( \mathbf { w } ^ { * } , \pmb { \theta } _ { t } ) = 0 .$ Assuming the standard regularity conditions (Linear Independence Constraint Qualification, Second-Order Sufficiency Condition, and Strict Complementarity) hold locally, the Implicit Function Theorem (IFT) yields a locally unique solution map whose derivative exists piecewise, i.e., within regions where the active set remains unchanged. The sensitivity matrix $\mathbf { S } _ { t } .$ , representing the Jacobian of the solution with respect to the parameters, then follows from differentiating the KKT residual equations with respect to $\theta _ { t }$

$$
\mathbf { S } _ { t } = \frac { \partial \mathbf { w } ^ { * } } { \partial \pmb { \theta } _ { t } } = - \left[ \frac { \partial \mathbf { R } } { \partial \mathbf { w } ^ { * } } \right] ^ { - 1 } \frac { \partial \pmb { R } } { \partial \pmb { \theta } _ { t } } .\tag{8}
$$

In this work, we employ the recent differentiable-solver sensitivity functionality implemented in acados [48], [29]. Concretely, the solver returns forward sensitivities of the NLP solution with respect to the cost weights by solving the linearized KKT system associated with the current SQP iterate. This yields the Jacobian $\nabla _ { \pmb { \theta } _ { t } } \mathbf { z } ^ { * }$ , quantifying how the optimal trajectory changes with infinitesimal perturbations in the cost weights.

## B. Gradient Propagation and Policy Learning

The goal of GB-PL is to update the policy-parameter collection ϕ, comprising all trainable weights and biases of the neural policy, by differentiating through the NMPC solver. Since this requires end-to-end differentiability, GB-PL optimizes a differentiable surrogate loss $\tilde { \mathcal { L } } _ { \mathrm { p e r f } } ( \mathbf { z } ^ { * } )$ defined on the NMPC solution $\mathbf { z } ^ { \ast }$ (predicted state/control trajectory) returned by the solver. This surrogate is a differentiable proxy for the task-level loss $\mathcal { L } _ { \mathrm { p e r f } } ( \cdot )$ introduced in Section II-B. In our setting, it is built by reusing the same smooth tracking, velocity, and control-regularization terms as the task-level loss, but evaluating them on the NMPC-predicted open-loop trajectory $\mathbf { z } ^ { \ast }$ rather than the realized closed-loop trajectory. Consequently, GB-PL is restricted to objectives for which such a smooth surrogate can be written; genuinely non-smooth or sparse task terms must either be approximated by smooth surrogates or omitted from the GB-PL update, which is itself a limitation of the approach rather than something resolved here.

We compute the corresponding model-based end-to-end gradient of $\tilde { \mathcal { L } } _ { \mathrm { { p e r f } } }$ with respect to the policy parameters $\phi$ at time t using the chain rule:

$$
\nabla _ { \phi } \tilde { \mathcal { L } } _ { \mathrm { p e r f } } = \underbrace { \left( \underbrace { \frac { \partial \tilde { \mathcal { L } } _ { \mathrm { p e r f } } } { \partial \mathbf { z } ^ { * } } } _ { \mathrm { L o s s ~ G r a d i e n t } } \cdot \underbrace { \frac { \partial \mathbf { z } ^ { * } } { \partial \theta _ { t } } } _ { \mathrm { S o l v e r ~ S e n s i t i v i t y } } \right) } _ { : = \mathbf { g } _ { \mathrm { S G } , t } ^ { \theta } } \cdot \underbrace { \frac { \partial \pmb { \theta } _ { t } } { \partial \phi } } _ { \mathrm { P o l i c y ~ J a c o b i a n } }\tag{9}
$$

1. Loss Gradient: $\frac { \partial \tilde { \mathcal { L } } _ { \mathrm { p e r f } } } { \partial \mathbf { z } ^ { * } }$ is computed using automatic differentiation (e.g., PyTorch). It represents the sensitivity of the differentiable surrogate loss with respect to the optimal trajectory produced by the NMPC.

2. Solver Sensitivity: $\frac { \partial \mathbf { z } ^ { * } } { \partial \pmb { \theta } _ { t } }$ is extracted from the sensitivity matrix $\mathbf { S } _ { t }$ in (8), computed by acados. This step bridges the optimization landscape with the control parameters.

3. Weight-Space Solver Gradient: The product $\mathbf { g } _ { \mathrm { S G } , t } ^ { \theta }$ is the local gradient of the surrogate loss in NMPC-weight space. The corresponding solver-informed descent step $\mathrm { i s } \ - \mathbf { g } _ { \mathrm { S G } , t } ^ { \theta } .$ We reuse this notation in Section VI, where the same solverderived loss-gradient direction is normalized and injected into PPO.

4. Policy Jacobian: $\frac { \partial \pmb { \theta } _ { t } } { \partial \phi }$ maps the weight-space solver loss gradient back to the trainable parameters of the neural policy. Thus, GB-PL computes an exact model-based gradient of the surrogate loss $\tilde { \mathcal { L } } _ { \mathrm { p e r f } } ( \mathbf { z } ^ { * } )$ by differentiating through the solver and the policy.

Formally, the update rule of GB-PL at iteration $k ,$ over a data batch with learning rate η, is:

$$
\mathcal { U } _ { \mathrm { G B } } : \phi _ { k + 1 } = \phi _ { k } - \eta \frac { 1 } { | \mathcal { B } | } \sum _ { i \in \mathcal { B } } \nabla _ { \phi } \tilde { \mathcal { L } } _ { \mathrm { p e r f } } ^ { ( i ) } .\tag{10}
$$

Key Novelties: Although pure GB-PL is not our main focus, we introduce three structural improvements to the existing reduced GB-PL formulation in [5]. First, instead of updating the policy after each rollout in the order it is generated, we collect a batch of rollouts, shuffle them into mini-batches, and perform multiple optimization passes (epochs) over this batch before collecting new data. This improves performance and sample efficiency. Second, we constrain the policy output using a differentiable tanh projection (Section III-B), ensuring all weights remain within a safe, bounded region and avoiding the poorly scaled objectives that can arise with unbounded parameterizations. Third, our observation space is history-aware, combining temporal stacks of current states and tracking errors with future references, which enables the policy to adapt reactively to disturbances and persistent errors, rather than relying solely on feedforward prediction. These changes collectively yield a more stable, adaptive, and scalable GB-PL framework for weights-varying differentiable NMPC.

## C. GB-PL Algorithm Limitations

While GB-PL provides a direct path for optimizing a differentiable, model-based surrogate loss $\tilde { \mathcal { L } } _ { \mathrm { p e r f } } ( \cdot )$ , it suffers from specific structural limitations when the goal is closedloop performance optimization:

1. Local descent without exploration: Like other local gradient methods, GB-PL follows the immediate descent direction of its surrogate objective and does not include an explicit exploration mechanism. Its training outcome is therefore sensitive to initialization and to the local geometry of the surrogate landscape.

2. Need for differentiable loss: GB-PL requires the surrogate loss $\tilde { \mathcal { L } } _ { \mathrm { p e r f } } ( \mathbf { z } ^ { * } )$ to be differentiable with respect to the NMPC trajectory $\mathbf { z } ^ { \ast }$ . This precludes optimizing sparse event-based objectives (e.g., ”Lap completed”, ”Collision”) or using nonsmooth penalties.

3. Open-loop, model-dependent tuning: The sensitivities in (9) are derived from the NMPC’s internal open-loop predictions rather than from realized closed-loop trajectories. As a result, GB-PL primarily tunes the MPC cost to match the task loss under the predictive model used by the solver, rather than directly optimizing the true closed-loop behavior of the plant. This cost-matching view is closely related to recent model-based MPC cost-matching formulations [33]. When the predictive model is inaccurate, the resulting gradient can systematically deviate from the true closed-loop improvement direction.

## V. REINFORCEMENT LEARNING BASELINE FOR WEIGHTS-VARYING NMPC

To address the limitations of GB-PL—specifically the myopic nature of local gradient descent and the requirement of differentiable surrogate losses—we formulate NMPC weight adaptation within a RL framework and use it as the second benchmark for solver-guided RL (Sec. VI). Our baseline is PPO, an on-policy actor–critic method that updates the stochastic weight policy $\pi _ { \phi }$ from sampled closed-loop returns without solver differentiation. Building on RL-WMPC [3], [4], the plant plus NMPC controller defines the environment, and the agent adapts NMPC weights through interaction. We choose PPO over SAC because its on-policy structure uses MPC real-time iterations more efficiently than random offpolicy sampling.

We model the problem as a Partially Observable Markov Decision Process (POMDP) $\mathcal { M } = ( \mathcal { S } , \mathcal { A } , \mathcal { P } , \mathfrak { r } , \gamma )$ with latent state space $s ,$ action space , transition kernel ${ \mathcal P } _ { \mathrm { { s } } }$ , reward r, and discount $\gamma .$ . The action is the NMPC weight vector, $\mathbf { a } _ { t } \ \equiv \ \pmb { \theta } _ { t } ,$ , and $\mathcal { P }$ captures plant–controller closed-loop evolution under those weights. The policy $\pi _ { \phi } ( \cdot ~ \mid ~ \sigma _ { t } )$ conditions on observations built from measurements and references (Sec. III-A).

To ensure a rigorous comparison against GB-PL, we define the reward $r _ { t }$ as the negative of the same instantaneous highlevel performance loss introduced in Section II-B. Thus, the reward is evaluated on the realized closed-loop state and applied input at time t, rather than on the internal NMPC cost J or on the NMPC-predicted trajectory used inside the differentiable surrogate of GB-PL:

$$
r _ { t } = - \mathcal { L } _ { \mathrm { p e r f } } ( \mathbf { s } _ { t } , \mathbf { u } _ { 0 } ^ { * } ) .\tag{11}
$$

Here $\mathbf { u } _ { 0 } ^ { * }$ denotes the first control input of the NMPC solution applied to the system at time t. We denote the corresponding expected closed-loop return by

$$
\mathcal { R } ( \pi _ { \phi } ) : = \mathbb { E } _ { \pi _ { \phi } } \left[ \sum _ { t = 0 } ^ { T } \gamma ^ { t } r _ { t } \right] .\tag{12}
$$

This reward alignment ensures that any performance difference between RL and GB-PL is attributable to the learning signal and update mechanism rather than to a different task objective. In particular, PPO estimates its policy update from closedloop returns collected through environment interactions, so the resulting policy gradient is an environment-based signal that implicitly reflects exploration, stochasticity, and model mismatch. By contrast, GB-PL’s update is a model-based gradient built from the same underlying performance terms.

The preceding sections reveal a trade-off between differentiable solver-based learning and RL for Weights-varying NMPC. GB-PL incorporates solver sensitivities, enabling sample-efficient gradient-based policy updates through the NMPC layer. However, these updates optimize a differentiable surrogate defined on predicted trajectories and can therefore deviate from the true closed-loop objective under model mismatch or non-smooth task objectives. Conversely, PPO directly optimizes the expected closed-loop return through environment interaction and can capture model mismatch and long-horizon effects, but can require more interaction data and does not exploit the sensitivity information available from the NMPC solver. This motivates the central question addressed in the remainder of this paper: how can solver-gradient information from differentiable NMPC be incorporated into RL to accelerate policy optimization while preserving the closedloop return as the learning objective?

## VI. SOLVER-GRADIENT GUIDED REINFORCEMENT LEARNING

In this section, we introduce our Solver-Gradient Guided Reinforcement Learning (SG-RL), whose core idea is to inject the model-based NMPC solver loss gradient $\mathbf { g } _ { \mathrm { S G } , t }$ into selected components of the PPO algorithm.

a) Solver gradients as normalized loss gradients: Let $\pmb { \theta } _ { t } \in \mathbb { R } ^ { n _ { \theta } }$ denote the NMPC cost-weight vector used by the solver at time t. During stochastic rollouts this is the sampled RL action, i.e., $\mathbf { } a _ { t } \equiv \mathbf { \ } \theta _ { t } .$ , whereas deterministic evaluation and the differentiable solver-guidance terms use the policy mean $\mu _ { \phi } ( o _ { t } )$ unless stated otherwise. On regular regions of the piecewise-differentiable NMPC solution map, we can compute the local sensitivity of the instantaneous performance loss with respect to these weights: Following Section IV-B, we define the corresponding weight-space solver gradient as the gradient of the differentiable surrogate loss:

$$
\begin{array} { r } { \mathbf { g } _ { \mathrm { S G } , t } ^ { \theta } : = \nabla _ { \theta } \tilde { \mathcal { L } } _ { \mathrm { p e r f } } \bigl ( \mathbf { z } ^ { * } ( \pmb { \theta } ) \bigr ) \Big \rvert _ { \theta = \theta _ { t } } . } \end{array}\tag{13}
$$

![](images/9a98352576a5f3a4cdcd832cf9e342d77f77b4a0f24f500fcf6f636a9a215043.jpg)  
Fig. 3: Bi-level Weights-varying NMPC architecture. The policy selects cost weights for the NMPC, while the upper-level learner updates the policy from closed-loop rollouts using sampled performance feedback and solver-gradient guidance.

Throughout this section, we use a direction-only solver lossgradient signal by normalizing the sensitivity vector,

$$
\mathbf { g } _ { \mathrm { S G } , t } : = \frac { \mathbf { g } _ { \mathrm { S G } , t } ^ { \theta } } { \| \mathbf { g } _ { \mathrm { S G } , t } ^ { \theta } \| _ { 2 } + \epsilon } ,\tag{14}
$$

so that $\| \mathbf { g } _ { \mathrm { S G } , t } \| _ { 2 } \leq 1$ . This removes scale variability across operating contexts and makes the solver-derived signal comparable across time, while leaving update magnitude to the method-specific coefficients that determine how strongly the solver loss-gradient direction is used. A local descent step in weight space is therefore taken along $- \mathbf { g } _ { \mathrm { S G } , t } .$ . When sensitivities are unavailable (e.g., infeasible solves), we set $\mathbf { g } _ { \mathrm { S G } , t } = \mathbf { 0 }$ recovering standard PPO behavior.

b) RL gradient convention: We train PPO by minimizing the compound loss introduced in Eq. (3), which combines the clipped policy objective with auxiliary terms such as value regression and entropy regularization. Throughout this section, we denote the objective-level PPO policy gradient by

$$
\begin{array} { r } { \mathbf { g } _ { \mathrm { R L } } : = \nabla _ { \phi } \mathcal { L } _ { \mathrm { P P O } } ( \phi , \psi ) . } \end{array}\tag{15}
$$

In implementation, PPO evaluates this gradient through stochastic minibatch estimates, which we denote by g<sub>RL,</sub> $\mathbf { \nabla } _ { \cdot } B : =$ $\nabla _ { \phi } \mathcal { L } _ { \mathrm { P P O } , B } ( \phi , \psi )$ for a minibatch . At training iteration $k ,$ we denote by $\mathbf { g } _ { \mathrm { R L } , k }$ the realized stochastic PPO gradient used in that update, e.g., a minibatch estimate of the form g<sub>RL,Bk</sub> .

c) Overview: We define the ”Standard GB-PL Update” ( , Eq. 10) and ”Standard RL Update” $( \mathcal { U } _ { \mathrm { R L } }$ , Eq. 4) as the baselines. We propose four novel hybrid methods that inject the solver gradient $\mathbf { g } _ { \mathrm { S G } }$ into different components of the PPO architecture (Fig. 3). We use one SG-RL variant per injection point so that the comparison isolates where solver guidance is most useful:

1. Adaptive PPO Update Scaling (SG-SCA): Modulates the magnitude of the PPO policy update based on its alignment with a solver-consistent loss-gradient direction, acting as a dynamic learning rate.

2. Performance-Gated Loss (SG-LOS): Adds an auxiliary supervised loss that pulls the policy toward a solver-suggested target only when the RL agent performs poorly.

3. Advantage Shaping (SG-ADV): Modifies the advantage function to reward action displacements that align with the negative solver loss gradient, biasing exploration.

4. Augmented Critic (SG-CRT): Conditions the critic on the solver sensitivity vector, providing an additional landscape descriptor for value estimation.

As shown in Fig. 3 and in Algorithms 1–4, each method represents a different strategy for fusing model-based NMPC solver gradients with PPO.

## A. Method 1: Adaptive PPO Update Scaling (SG-SCA)

Standard PPO already provides a loss-gradient update direction, but it still relies on a heuristic step size. We want the solver signal to tell us when PPO should trust that direction more strongly and when it should act more conservatively. We therefore use the NMPC solver loss gradient to scale the magnitude of the PPO update while preserving the PPO update direction.

Because $\mathbf { g } _ { \mathrm { S G } , t }$ lives in the NMPC weight space, we first define for each transition a solver-consistent scalar whose gradient we can backpropagate through the policy:

$$
\ell _ { \mathrm { S G } , t } ( \phi ) : = \pmb { \mu } _ { \phi } ( \pmb { o } _ { t } ) ^ { \top } \mathbf { g } _ { \mathrm { S G } , t } ,\tag{16}
$$

During the PPO update, we treat $\mathbf { g } _ { \mathrm { S G } , t }$ as a fixed stored solver signal and differentiate only through the policy output $\mu _ { \phi } ( o _ { t } )$ This induces the lifted parameter-space loss-gradient signal

$$
\begin{array} { r } { \widetilde { \mathbf { g } } _ { \mathrm { S G } , t } : = \nabla _ { \phi } \ell _ { \mathrm { S G } , t } ( \phi ) . } \end{array}\tag{17}
$$

At the PPO-objective level, we aggregate these per-transition quantities as

$$
\ell _ { \mathrm { S G } } ( \phi ) : = \hat { \mathbb { E } } _ { t } \left[ \ell _ { \mathrm { S G } , t } ( \phi ) \right] = \hat { \mathbb { E } } _ { t } \left[ \pmb { \mu } _ { \phi } ( \pmb { o } _ { t } ) ^ { \top } \mathbf { g } _ { \mathrm { S G } , t } \right] ,
$$

which induces the lifted loss-gradient signal

(18)

$$
\tilde { \mathbf { g } } _ { \mathrm { S G } } : = \nabla _ { \phi } \ell _ { \mathrm { S G } } ( \phi ) .\tag{19}
$$

```latex
Algorithm 1 SG-SCA
1: for each PPO minibatch do
2: compute PPO minibatch gradient g<sub>RL</sub> $\mathrm { \Delta } _ { , B } = \nabla _ { \phi } { \mathcal { L } }$ P P O,B
3: form ℓ<sub>SG</sub> ${ \mathbf { \xi } } , { \mathbf { \xi } } = \sum _ { \mathbf { \beta } , \mathbf { \alpha } , \mathbf { \beta } \in B } { \pmb \mu } _ { \phi } \big ( \partial _ { i } \big ) ^ { \top }$ g<sub>SG,i</sub> and $\widetilde { \mathbf { g } } _ { \mathrm { S G } , B } = \nabla _ { \phi } \ell _ { \mathrm { S G } , B }$
4: compute $\rho = ( \widetilde { \mathbf { g } } _ { \mathrm { S G } , B } ^ { \top } \ \mathbf { g } _ { \mathrm { R L } , B } ) / ( \lVert \mathbf { g } _ { \mathrm { R L } , B } \rVert ^ { 2 } + \epsilon )$
5: compute $\alpha = \mathrm { c l i p } ( 1 + \lambda \rho , 0 , \alpha _ { \mathrm { m a x } } )$
6: update policy params $\phi$ with scaled PPO gradient α g<sub>RL,B</sub>
7: end for
```

We then compute the alignment $\rho$ between the lifted solverconsistent loss-gradient signal and the PPO policy gradient g defined above:

$$
\rho = \frac { \widetilde { \mathbf { g } } _ { \mathrm { S G } } ^ { \intercal } \mathbf { g } _ { \mathrm { R L } } } { \vert \vert \mathbf { g } _ { \mathrm { R L } } \vert \vert ^ { 2 } + \epsilon } .\tag{20}
$$

This coefficient measures how strongly the solver-derived loss gradient projects onto the PPO loss gradient. We then construct an adaptive scaling factor α:

$$
\alpha = \mathrm { c l i p } \left( 1 + \lambda \cdot \rho , 0 , \alpha _ { \mathrm { m a x } } \right) ,\tag{21}
$$

where $\lambda \geq 0$ is a strength hyperparameter and $\alpha _ { \mathrm { m a x } }$ is a saturation limit preventing excessive update steps that could destabilize the policy. The lower bound of 0 acts as a safeguard by preventing the adaptive mechanism from reversing the PPO update direction. Finally, we scale the effective PPO update step and formally define the SG-SCA operator <sub>SG−SCA</sub> as

$$
\mathcal { U } _ { \mathrm { S G - S C A } } : \phi _ { k + 1 } = \phi _ { k } - \eta \cdot \left( \mathbf { \Omega } \alpha \mathbf { \cdot } \mathbf { g } _ { \mathrm { R L } } \right) ,\tag{22}
$$

By scaling the magnitude of $\mathbf { g } _ { \mathrm { R L } }$ in this way, we take larger descent steps $- \mathbf { g } _ { \mathrm { R L } }$ when the solver loss gradient confirms the PPO loss gradient and smaller steps when it conflicts. In implementation, PPO evaluates both gradients through stochastic minibatch estimates on each minibatch $B ,$ as summarized in Algorithm 1 via $\ell _ { \mathrm { S G } , B } , \widetilde { \mathbf { g } } _ { \mathrm { S G } , B } ,$ and g<sub>RL,B</sub>.

## B. Method 2: Performance-Gated Loss (SG-LOS)

SG-LOS addresses the case in which PPO explores a poor weight configuration while the solver gradient suggests modelconsistent local correction. At the same time, we do not want the solver signal to override trajectories that already achieve high return. We therefore add a performance-gated target-following loss to the minimized PPO training loss. Unlike SG-SCA, which rescales the PPO descent step, SG-LOS introduces an explicit target-following loss $\mathcal { L } _ { \mathrm { g u i d e } }$ added to the PPO training loss:

$$
{ \mathcal { L } } _ { \mathrm { S G - L O S } } ( \phi , \psi ) = { \mathcal { L } } _ { \mathrm { P P O } } ( \phi , \psi ) + \lambda _ { \mathrm { g u i d e } } { \mathcal { L } } _ { \mathrm { g u i d e } } ( \phi ) ~ .\tag{23}
$$

The surrogate loss $\tilde { \mathcal { L } } _ { \mathrm { p e r f } }$ directly is not added to PPO as a globally active regularizer. Because that surrogate is local, open-loop, and model-based, applying it uniformly would treat the solver signal as equally reliable on all samples, including trajectories that already achieve high return. Instead, SG-LOS uses the solver gradient only to construct a local correction target and activates that correction selectively on underperforming transitions. Let $\bar { \pmb { \theta } } _ { t }$ denote the behavior-policy weight vector associated with transition t. We form a local target weight configuration by taking a fixed step $\eta _ { \mathrm { g u i d e } }$ along the surrogate descent step $- \mathbf { g } _ { \mathrm { S G } , t } ;$

Algorithm 2 SG-LOS   
1: for each collected transition do   
2: form $\pmb { \theta } _ { t } ^ { \mathrm { t a r g e t } } = \bar { \pmb { \theta } } _ { t } - \eta _ { \mathrm { g u i d e } } \mathbf { g } _ { \mathrm { S G } , t }$ with the normalized solver   
gradient   
3: compute the gate $w _ { t } = \mathrm { c l i p } ( - \hat { A } _ { t } , 0 , w _ { \mathrm { m a x } } )$   
4: ignore transitions with negligible solver-gradient norm   
5: end for   
6: for each PPO minibatch do   
7: minibatch estimate of $\mathcal { L } _ { \mathrm { g u i d e } }$ using the valid transitions   
8: add $\lambda _ { \mathrm { g u i d e } } \mathcal { L } _ { \mathrm { g u i d e } }$ to the PPO training loss   
9: minimize the combined loss   
10: end for

$$
\begin{array} { r } { \pmb { \theta } _ { t } ^ { \mathrm { t a r g e t } } = \bar { \pmb { \theta } } _ { t } - \eta _ { \mathrm { g u i d e } } \mathbf { g } _ { \mathrm { S G } , t } . } \end{array}\tag{24}
$$

We penalize the distance between the current policy mean $\mu _ { \phi } ( o _ { t } )$ and this target with the gated loss

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { g u i d e } } ( \phi ) = \hat { \mathbb { E } } _ { t } \left[ w _ { t } \lVert \pmb { \mu } _ { \phi } ( \pmb { o } _ { t } ) - \pmb { \theta } _ { t } ^ { \mathrm { t a r g e t } } \rVert _ { 2 } ^ { 2 } \right] . } \end{array}\tag{25}
$$

To implement this selective rectification, we derive the weight $w _ { t }$ from the advantage function ${ \hat { A } } _ { t } \mathbf { : }$

$$
w _ { t } = \mathrm { c l i p } ( - \hat { A } _ { t } , 0 , w _ { \mathrm { m a x } } ) ,\tag{26}
$$

where $w _ { \mathrm { m a x } }$ is the upper bound on the gating factor. In implementation, we normalize the stored raw solver gradient, use the saved rollout mean as $\bar { \pmb { \theta } } _ { t }$ when it is available, and mask out transitions with negligible solver-gradient norm before forming the minibatch estimate of $\mathcal { L } _ { \mathrm { g u i d e } }$ . With this gate, negativeadvantage samples receive a corrective pull toward the solversuggested target, whereas positive-advantage samples reduce to standard PPO. Early in training, when underperforming transitions are more common, the gate naturally activates more often; the overall aggressiveness is then controlled by $\lambda _ { \mathrm { g u i d e } } ,$ $\eta _ { \mathrm { g u i d e } } ,$ , and the cap $w _ { \mathrm { m a x } }$

## C. Method 3: Advantage Shaping (SG-ADV)

SG-ADV targets exploration rather than the optimizer step. PPO learns through the advantage estimate $\hat { A } _ { t } .$ , but $\hat { A } _ { t }$ alone does not tell us whether the proposed policy-mean displacement points along a locally favorable solver descent step. We therefore inject the solver signal directly into the advantage estimate while keeping PPO on-policy.

As reflected in the PPO surrogate objective in Eq. (2), PPO optimizes the policy based on an estimate $\hat { A } _ { t }$ of the advantage function. We augment this estimate with a gradientbased correction term derived from the alignment between the policy’s mean weight displacement and the local loss landscape in weight space. We define the shaped advantage $\hat { A } _ { t } ^ { \mathrm { S G } }$ as:

$$
\hat { A } _ { t } ^ { \mathrm { S G } } = \hat { A } _ { t } + \beta \Delta _ { t } ^ { \mathrm { S G } } ,\tag{27}
$$

where $\beta \geq 0$ is a scaling coefficient. The correction term $\Delta _ { t } ^ { \mathrm { S G } }$ encodes the consistency between the policy’s deterministic displacement from the rollout reference $\bar { \pmb { \theta } } _ { t }$ and the descent step g<sub>SG,t</sub>:

$$
\Delta _ { t } ^ { \mathrm { S G } } = - \left( \pmb { \mu } _ { \phi } ( \pmb { o } _ { t } ) - \bar { \pmb { \theta } } _ { t } \right) ^ { \top } \mathbf { g } _ { \mathrm { S G } , t } .\tag{28}
$$

Algorithm 3 SG-ADV   
1: for each collected transition do   
2: compute $\Delta _ { t } ^ { \mathrm { S G } } = - \left( \pmb { \mu } _ { \phi } ( \pmb { o } _ { t } ) - \bar { \pmb { \theta } } _ { t } \right) ^ { \top }$ g<sub>SG,t</sub>   
3: set $\hat { A } _ { t } ^ { \mathrm { S G } } = \hat { A } _ { t } + \beta \dot { \Delta } _ { t } ^ { \mathrm { S G } }$   
4: end for   
5: for each PPO minibatch do   
6: replace $\hat { A } _ { t }$ by $\hat { A } _ { t } ^ { \mathrm { S G } }$ in the clipped PPO surrogate   
7: run the standard PPO actor–critic update with $\hat { A } _ { t } ^ { \mathrm { S G } }$   
8: end for   
Observation + X g<sub>SG,t</sub> MPC Solver Gradient   
V{ψ,base

![](images/99989a3088057d559322ffd65eb61cca3d6626ab50a608c1183e8d12b5a454a8.jpg)  
Fig. 4: Residual SG-CRT critic: an observation-based value estimate plus a scalar correction from a solver-gradient encoder.

Here, $\bar { \pmb { \theta } } _ { t }$ is the behavior-policy mean for the rollout transition. The term $\Delta _ { t } ^ { \mathrm { S G } }$ represents the projection of the proposed displacement onto the descent step $- \mathbf { g } _ { \mathrm { S G } , t }$ . Thus, $\bar { \Delta } _ { t } ^ { \mathrm { S G } } > 0$ implies that the policy mean moves locally toward lower surrogate performance loss, providing a solver-informed heuristic for action quality. We then substitute the shaped advantage into the standard PPO surrogate optimization problem (2):

$$
\begin{array} { r } { \mathcal { I } _ { \mathrm { S G } } ^ { \mathrm { C L I P } } ( \phi ) = \hat { \mathbb { E } } _ { t } \Big [ \operatorname* { m i n } \big ( \rho _ { t } ( \phi ) \hat { A } _ { t } ^ { \mathrm { S G } } , ~ } \\ { \mathrm { c l i p } ( \rho _ { t } ( \phi ) , 1 - \epsilon , 1 + \epsilon ) \hat { A } _ { t } ^ { \mathrm { S G } } \big ) \Big ] . } \end{array}\tag{29}
$$

With this shaping, we reward solver-consistent action displacements and penalize solver-inconsistent ones through the advantage signal itself.

## D. Method 4: Augmented Critic (SG-CRT)

SG-CRT targets the critic rather than the actor. The same observation can correspond to very different local controllability and tuning difficulty depending on how sensitive the NMPC solution is to the cost weights, so an observation-only critic misses useful information about the local landscape. We therefore augment the critic with a solver-derived landscape descriptor based on the weight-space loss gradient g<sub>SG</sub> and inject it through a residual correction branch:

$$
V _ { \psi } \big ( { \pmb { o } } _ { t } , \mathbf { g } _ { \mathrm { S G } , t ^ { - } } \big ) = V _ { \psi , \mathrm { b a s e } } ( \pmb { o } _ { t } ) + c _ { \psi } \big ( \mathbf { g } _ { \mathrm { S G } , t ^ { - } } \big ) \ ,\tag{30}
$$

where $V _ { \psi , \mathrm { b a s e } }$ is the conventional critic (MLP neural network), and $c _ { \psi }$ is a lightweight gradient encoder, e.g., a small fully connected network with bounded scalar output, that maps the normalized solver gradient to a residual correction. We use the index $t ^ { - }$ to emphasize that, in an online loop, the critic uses the most recently available solver sensitivity (e.g., from the previous NMPC solve) when evaluating the current observation. For the policy-gradient update, $\mathbf { g } _ { \mathrm { S G } , t ^ { - } }$ is treated as a detached, action-independent critic feature, and no actor gradient is propagated through the value target or residual branch. With this residual branch, we calibrate value predictions in high-sensitivity regions and can reduce the advantage variance without modifying PPO’s actor update rule.

Algorithm 4 SG-CRT   
1: for each environment step do   
2: cache the latest available solver gradient $\mathbf { g } _ { \mathrm { S G } , t ^ { - } }$   
3: evaluate $V _ { \psi } ( \pmb { o } _ { t } , \mathbf { g } _ { \mathrm { S G } , t ^ { - } } ) = V _ { \psi , \mathrm { b a s e } } ( \pmb { o } _ { t } ) + c _ { \psi } ( \mathbf { g } _ { \mathrm { S G } , t ^ { - } } )$   
4: end for   
5: for each PPO minibatch do   
compute returns and advantages with the augmented critic   
7: actor update with standard PPO and augmented critic fitting   
8: end for

Weight-space vs. parameter-space usage: The solver signal g<sub>SG,t</sub> is an NMPC weight-space loss gradient. SG-LOS, SG-ADV, and SG-CRT use it directly in weight space (to shape advantages, construct targets, or condition the critic). SG-SCA, in contrast, needs a policy parameter-space quantity to modulate the PPO update step: it therefore lifts the weightspace signal through the policy using the construction already introduced in Eq. (16).

## VII. OPTIMIZATION INTERPRETATION OF SG-RL

We formalize the SG-RL update in the same descentdirection convention used above and analyze its relationship to the expected closed-loop return $\mathcal { R } ( \pi _ { \phi } )$ . Since the reward in Section V is defined as $r _ { t } = - \mathcal { L } _ { \mathrm { p e r f } } ( \mathbf { s } _ { t } , \mathbf { u } _ { 0 } ^ { \star } )$ , maximizing $\mathcal { R } ( \pi _ { \phi } )$ is equivalent to minimizing the expected cumulative loss in Eq. (6). We interpret g<sub>RL,k</sub> as the iteration-k stochastic realization of the PPO loss gradient in Eq. (15) and treat it as a stochastic descent proxy $\mathrm { f o r } - \nabla _ { \phi } \mathcal { R } ( \pi _ { \phi _ { k } } )$ .

This section provides a first-order optimization interpretation of SG-RL rather than formal convergence guarantees. For analytical tractability, we discuss SG-RL through descent directions, while the implementation optimizes PPO’s clipped surrogate as a trust-region approximation. The generic update model below provides a common interpretive lens for the SG-RL family; the individual methods differ only in how solvergradient information is incorporated into the PPO optimization.

## A. Solver Guidance Signal and SG-RL Update

Using the same lifting construction as in Eqs. (16) and (17), we denote the model-based NMPC solver gradient in weight space at a generic observation o by

$$
\begin{array} { r } { \mathbf { g } _ { \mathrm { S G } } ^ { \theta } ( o ) : = \nabla _ { \pmb { \theta } } \tilde { \mathcal { L } } _ { \mathrm { p e r f } } \big ( \mathbf { z } ^ { * } ( \pmb { \mu } _ { \phi } ( o ) ) \big ) . } \end{array}\tag{31}
$$

We denote by $\mathbf { g } _ { \mathrm { S G } } ^ { \phi } ( o )$ the corresponding lifted normalized loss-gradient in policy-parameter space. The descent direction is therefore $- \mathbf { g } _ { \mathrm { S G } } ^ { \mathrm { { \bar { \phi } } } }$ , which should be interpreted as a local model-based guidance signal induced by the surrogate $\tilde { \mathcal { L } } _ { \mathrm { p e r f } }$ rather than as the true policy gradient of $\mathcal { R } ( \pi _ { \phi } )$

We define the SG-RL update as $\phi _ { k + 1 } = \phi _ { k } - \alpha _ { k } \mathbf { d } _ { k }$ , where the composite loss-gradient update vector $\mathbf { d } _ { k }$ is given by:

$$
\begin{array} { r } { \mathbf { d } _ { k } = \mathcal { F } _ { k } ( \mathbf { g } _ { \mathrm { R L } , k } , \mathbf { g } _ { \mathrm { S G } , k } ^ { \phi } ) . } \end{array}\tag{32}
$$

The operator $\mathcal { F } _ { k }$ combines the PPO and solver-guidance signals into a single update direction (Fig. 1). The iteration index permits dynamic blending strategies, such as annealing solver influence through $\mathcal { F } _ { k } ( \mathbf { x } , \mathbf { y } ) = \mathbf { x } + \lambda w _ { k } \mathbf { y } , \quad w _ { k } \in [ 0 , 1 ]$

![](images/e549f3158fb7c53c9c391c701a8b6dca633cc804122f307c0f00e376ba1ba62c.jpg)  
Fig. 5: Idealized policy-parameter dynamics. RL gradients optimize closed-loop return but may get trapped in poor basins; solver gradients provide biased local guidance; SG-RL combines both to guide early learning before RL refinements dominate.

## B. Early-Training Guidance

The solver-guidance signal $\mathbf { g } _ { \mathrm { S G } , k } ^ { \phi }$ is obtained by differentiating the surrogate $\tilde { \mathcal { L } } _ { \mathrm { p e r f } }$ rather than the expected return, and therefore introduces model bias. Conversely, g<sub>RL,k</sub> is estimated from sampled closed-loop trajectories and typically exhibits much higher trajectory-level variance early in training. SG-RL exploits this complementary bias–variance trade-off by using solver gradients only as guidance toward promising regions of the policy landscape (Fig. 5).

To make this intuition explicit, let

$$
\begin{array} { r } { \phi _ { \mathrm { S G } } ^ { \star } \in \arg \operatorname* { m i n } _ { \phi } \hat { \mathbb { E } } _ { o \sim \mathcal { D } } \left[ \tilde { \mathcal { L } } _ { \mathrm { p e r f } } \left( \mathbf { z } ^ { \ast } ( \pmb { \mu } _ { \phi } ( o ) ) \right) \right] } \end{array}\tag{33}
$$

be surrogate-optimal policy parameters under a generic dataset of observations collected during training. The surrogate is useful whenever optimizing it yields policies that outperform random initialization under the true return, i.e.,

$$
\mathcal { R } ( \pi _ { \phi _ { \mathrm { S G } } ^ { \star } } ) > \mathbb { E } _ { \phi _ { 0 } \sim \mathcal { P } _ { 0 } } \left[ \mathcal { R } ( \pi _ { \phi _ { 0 } } ) \right] .\tag{34}
$$

Here, $\phi _ { 0 } ~ \sim ~ \mathcal { P } _ { 0 }$ denotes a randomly initialized policy parameter set. The surrogate need not share the true optimum. It only needs to provide informative guidance during early learning. Our SG-SCA, SG-LOS, and SG-ADV methods use this surrogate bias in actor-side training signals. In contrast, SG-CRT uses solver sensitivities only as additional critic features. When detached from actor-gradient paths, the additional critic features leave the expected policy-gradient estimator unchanged while potentially reducing variance through a more accurate value baseline.

Because the actor-side methods intentionally bias optimization toward the surrogate objective, their influence is gradually annealed by scheduling $( \lambda , \lambda _ { \mathrm { g u i d e } } , \beta )$ in Eq. (21), Eq. (23), and Eq. (27). This heuristic progressively transfers control back to PPO as learning progresses.

## C. Non-aligned Updates

During this transition, the composite update ${ \bf d } _ { k }$ generally differs from the pure PPO direction. It therefore need not maximize immediate return improvement, but it can accelerate optimization by steering learning toward more promising regions before PPO dominates $( \mathrm { F i g . } 5 )$ . To make it concrete, consider SG-SCA in a near-plateau regime where $\mathbb { E } [ \mathbf { g } _ { \mathrm { R L } , k } ] \approx \mathbf { 0 }$ Because the scaling factor $\alpha _ { k }$ in (21) depends directly on the minibatch gradient $\mathbf { g } _ { \mathrm { R L } , k }$ , the expectation of their product does not factorize:

$$
\begin{array} { r } { \mathbb { E } [ \alpha _ { k } \mathbf { g } _ { \mathrm { R L } , k } ] = \mathbb { E } [ \alpha _ { k } ] \mathbb { E } [ \mathbf { g } _ { \mathrm { R L } , k } ] + \mathrm { C o v } ( \alpha _ { k } , \mathbf { g } _ { \mathrm { R L } , k } ) . } \end{array}\tag{35}
$$

Near such a plateau, the covariance term can induce a nonzero expected drift by amplifying solver-aligned minibatch directions and suppressing conflicting ones. Still, the closedloop return objective $\mathcal { R } ( \pi _ { \phi } )$ remains the primary optimization signal.

## VIII. APPLICATION EXAMPLE: HIGH-SPEED AUTONOMOUS RACING

We validate SG-RL in high-speed autonomous racing on two full-scale, high-fidelity simulation platforms representative of contemporary competitions: the Dallara AV-24 used in the Indy Autonomous Challenge (IAC) and the Super Formula EAV-24 from the Abu Dhabi Autonomous Racing League (A2RL). Both platforms are based on simulation and NMPC models identified from real-world vehicle data, while their underlying NMPC controllers have already been validated on the physical race cars.

## A. Why Autonomous Racing Stresses Weights Tuning

Autonomous racing is a demanding benchmark for weightsvarying NMPC because it combines tight feasibility constraints at the actuation and tire limits, pronounced modeling uncertainty and nonlinearity, and strong regime shifts within a single lap. Indeed, racing combines qualitatively different regimes: straights demand accurate velocity tracking, braking zones require deceleration stability, corners require lateral accuracy under combined-acceleration limits, and corner exits prioritize traction-limited acceleration. A single static weight tuning must compromise across these regimes, motivating online adaptation of the weight vector $\theta _ { t }$

![](images/d942e654a27f013998bea2c128f4da447ebb3e578bb7a4a2acc485f25e1a6d32.jpg)  
Fig. 6: Autonomous-racing validation platforms: Dallara AV-24 on Monza (left) and Super Formula EAV-24 on Yas Marina (right), both reaching peak velocities above $8 0 \mathrm { m } \mathrm { s } ^ { - 1 }$

## B. Vehicle Dynamic Model for NMPC

In our NMPC problem (5), we employ a nonlinear singletrack vehicle model $\dot { \mathbf { x } } = \pmb { f } ( \mathbf { x } , \mathbf { u } )$ based on [49], [50]. The states $\mathbf { x } = [ x , y , \psi , v _ { x } , v _ { y } , \dot { \psi } , \delta , a _ { x } ] ^ { \top }$ model the vehicle pose, velocities, steering angle, and longitudinal acceleration, while the inputs $\mathbf { u } = [ j _ { x } , \dot { \delta } ] ^ { \top }$ are the longitudinal jerk and steering rate. Lateral tire forces are modeled by the nonlinear Pacejka Magic Formula [51], and aerodynamic drag and downforce are included explicitly, affecting both longitudinal acceleration limits and available tire grip.

## C. NMPC Formulation

The NMPC tracks a time-optimal trajectory (path and velocity profile) generated by an optimal control planner [52]. At each control step, the planner provides an N-step reference segment containing desired position, heading, velocity, longitudinal acceleration, and lateral acceleration. The NMPC problem follows the structure in Section II. The cost function consists of a stage cost l( ) and a terminal cost $l _ { N } ( \cdot )$ formulated as quadratic forms:

$$
l ( \mathbf { x } _ { k } , \mathbf { u } _ { k } , \pmb { \theta } _ { t } ) = \frac { 1 } { 2 } \mathbf { r } ( \mathbf { x } _ { k } , \mathbf { u } _ { k } ) ^ { \top } \mathbf { W } ( \pmb { \theta } _ { t } ) \mathbf { r } ( \mathbf { x } _ { k } , \mathbf { u } _ { k } ) ,\tag{36}
$$

$$
l _ { N } ( \mathbf { x } _ { N } , \pmb { \theta } _ { t } ) = \frac { 1 } { 2 } \mathbf { r } _ { N } ( \mathbf { x } _ { N } ) ^ { \top } \mathbf { W } _ { N } ( \pmb { \theta } _ { t } ) \mathbf { r } _ { N } ( \mathbf { x } _ { N } ) .\tag{37}
$$

Here, $\mathbf { r } ( \cdot )$ denotes the vector of tracking errors and regularization terms. The weighting matrix $\mathbf { W } ( \pmb { \theta } _ { t } ) \mathbf { \Psi } = \mathrm { d i a g } ( \pmb { \theta } _ { t } )$ is directly parameterized by the policy action vector $\pmb { \theta } _ { t } ,$ , allowing our cost-weight policy to reshape the quadratic cost landscape online. The weights are fixed along the prediction horizon but updated between NMPC solves.

The feature vector for the stage cost is defined as:

$$
\mathbf { r } ( \mathbf { x } , \mathbf { u } ) = \left[ e _ { \mathrm { l a t } } , e _ { \psi } , e _ { v } , e _ { a } , e _ { a _ { \mathrm { l a t } } } , u _ { j } , u _ { \bar { \delta } } \right] ^ { \top } .\tag{38}
$$

Here, $e _ { \mathrm { l a t } } , e _ { \psi } , e _ { v } , e _ { a }$ , and $e _ { a _ { \mathrm { l a t } } }$ denote the tracking errors for lateral position, heading, velocity, longitudinal acceleration, and lateral acceleration, while $u _ { j }$ and $u _ { \dot { \delta } }$ penalize highfrequency actuation. The terminal feature vector ${ \bf r } _ { N } ( { \bf x } _ { N } )$ contains the same state-dependent terms but excludes the control inputs.

The learnable weight vector (policy action) is

$$
\begin{array} { r } { \pmb { \theta } _ { t } : = \left[ q _ { \ell , t } \quad q _ { \psi , t } \quad q _ { v , t } \quad q _ { a , t } \quad q _ { a _ { y } , t } \quad r _ { j , t } \quad r _ { \hat { \delta } , t } \right] ^ { \top } \in \mathbb { R } _ { > 0 } ^ { 7 } , } \end{array}\tag{39}
$$

representing the weights on the corresponding error terms in the cost function. We enforce strict feasibility bounds on actuators and physical envelopes on the states:

a) Actuation Limits: The control inputs are bounded to respect the physical steering rate and jerk limits:

$$
\mathcal { U } = \{ \mathbf { u } ~ | ~ \dot { \delta } _ { \operatorname* { m i n } } \leq \dot { \delta } \leq \dot { \delta } _ { \operatorname* { m a x } } , ~ \dot { j } _ { \operatorname* { m i n } } \leq j _ { x } \leq j _ { \operatorname* { m a x } } \} .\tag{40}
$$

b) State Constraints: The steering angle δ is limited by the mechanical steering range. The longitudinal acceleration $a _ { x }$ is restricted by the velocity-dependent powertrain envelope, capturing the engine power curve and braking capacity:

$$
\delta _ { \operatorname* { m i n } } \leq \delta _ { k } \leq \delta _ { \operatorname* { m a x } } \mathrm { ~ a n d ~ } a _ { x , \operatorname* { m i n } } ( v _ { x , k } ) \leq a _ { x , k } \leq a _ { x , \operatorname* { m a x } } ( v _ { x , k } ) .
$$

c) Combined Friction Ellipse (GG-Diagram): We impose a coupling constraint between longitudinal acceleration $a _ { x }$ and lateral acceleration $\begin{array} { l } { a _ { y } } \end{array} \approx \begin{array} { l } { v _ { x } \dot { \psi } } \end{array}$ . We approximate the friction utilization using a generalized ellipsoid [53]: $\left( | a _ { x , k } | / a _ { x , \operatorname* { m a x } } ^ { \mathrm { c o m b } } ( v _ { x , k } ) \right) ^ { \eta } + \left( | a _ { y , k } | / a _ { y , \operatorname* { m a x } } ^ { \mathrm { \tilde { c o m b } } } ( v _ { x , k } ) \right) ^ { \eta } \overset { < } { \leq } 1$ . Here, η shapes the envelope and $a _ { x / y , \mathrm { m a x } } ^ { \mathrm { c o m b } } ( \bar { v _ { x , k } } )$ are velocity-dependent acceleration capacities.

## D. Observations for Racing

The policy input $o _ { t }$ must support reliable weight selection while staying low-dimensional and transferable. We use three information sources:

a) Current motion state (stacked measurements): We include a stack of the most recent longitudinal velocities $\left\{ v _ { x } ( t - i ) \right\} _ { i = 0 } ^ { n _ { h } - 1 }$ and yaw rates $\left\{ r ( t - i ) \right\} _ { i = 0 } ^ { n _ { h } - 1 }$ . Here, r denotes yaw rate and $n _ { h }$ is the stack length. This captures the current dynamic regime and short transients.

b) Performance history (tracking deviations): We include a stack of recent magnitudes of path-tracking deviations $\left\{ | e _ { \mathrm { l a t } } ( t - i ) | \right\} _ { i = 0 } ^ { n _ { h } - 1 }$ and velocity-tracking deviations $\left\{ \left| e _ { v } ( t - i ) \right| \right\} _ { i = 0 } ^ { n _ { h } - 1 }$

c) Task anticipation (look-ahead): We include samples of the reference velocity $\dot { \left\{ v _ { \mathrm { r e f } } ( t + \tau _ { j } ) \right\} } _ { j = 1 } ^ { K }$ and path curvature $\left\{ \kappa _ { \mathrm { r e f } } ( t + \tau _ { j } ) \right\} _ { j = 1 } ^ { K } ,$ , with K anticipation points over the NMPC horizon, where $\tau _ { j }$ are uniformly spaced look-ahead times in $[ 0 , T _ { p } ]$

## E. Learning Cost-Weight Policies

We treat the NMPC closed loop as the environment and bounded weights $\theta _ { t }$ as actions. At each decision time, the actor outputs $\theta _ { t } \in [ \theta _ { \operatorname* { m i n } } , \theta _ { \operatorname* { m a x } } ]$ , parameterizing $\mathbf { W } ( \pmb { \theta } _ { t } )$ and $\mathbf { W } _ { N } ( \pmb { \theta } _ { t } )$ . The differentiable NMPC solver provides primalsolution sensitivities with respect to the weights, enabling the closed-loop solver-derived gradient ${ \bf g } _ { S G , t }$ used by the SG-RL mechanisms. We evaluate each closed-loop transition using a fixed per-step performance loss $\mathcal { L } _ { \mathrm { p e r f } } ( \mathbf { s } _ { t } , \mathbf { u } _ { 0 } ^ { * } )$ :

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { p e r f } } ( \mathbf { s } _ { t } , \mathbf { u } _ { 0 } ^ { * } ) = w _ { v } e _ { v , t } ^ { 2 } + w _ { \mathrm { l a t } } e _ { \mathrm { l a t } , t } ^ { 2 } \qquad } \\ { + w _ { j } \mathcal { P } _ { \mathrm { e x p } } ( j _ { t } , \bar { j } ) + w _ { \dot { \delta } } \mathcal { P } _ { \mathrm { e x p } } ( \dot { \delta } _ { t } , \bar { \dot { \delta } } ) . } \end{array}\tag{41}
$$

Here, $e _ { v , t }$ and $e _ { \mathrm { l a t } , t }$ are realized velocity/lateral errors, and $j _ { t } , \dot { \delta } _ { t }$ are realized jerk and steering rate. The function $\mathcal { P } _ { \mathrm { e x p } } ( u , \bar { u } ) = \exp ( \lambda _ { \mathrm { e x p } } ( | u | - \bar { u } ) ) - 1$ if $| u | \ > \ \bar { u }$ and $u ^ { 2 }$ otherwise, with $\lambda _ { \mathrm { { e x p } } } > 0 .$ , strongly penalizes violations of $\bar { j }$ and $\bar { \dot { \delta } } .$ The weights $\mathbf { G } _ { \mathrm { p e r f } } = \{ w _ { v } , w _ { \mathrm { l a t } } , w _ { j } , w _ { \dot { \delta } } \}$ are fixed, and ${ r } _ { t } = - { \mathcal { L } } _ { \mathrm { p e r f } } ( { \bf s } _ { t } , { \bf u } _ { 0 } ^ { * } )$ aligns the RL reward with the solvergradient signal.

## IX. RESULTS

## A. Experimental Setup

We train and evaluate cost-weight policies in closed-loop simulation on the AV-24 Indy Dallara and EAV-24 Super Formula platforms. Both high-fidelity simulation and NMPC prediction models were identified from real vehicle-dynamics data and validated against racing-event measurements [54], [55]. The exact NMPC implementations used in this work were transferred to the real cars and validated during full-scale racing at Yas Marina and Modena in April/June 2025 with the same Weights-varying architecture and a rule-based policy<sup>1</sup>. Since only the weight-selection policy changes, learned policies are directly transferable to the validated controller stack. Learning remains in simulation because near-limit exploration on full-scale race cars is unsafe and costly.

A higher-level planner provides the local raceline, and the differentiable NMPC computes inputs for the high-fidelity plant simulator. Plant and prediction models are nonlinear single-track models with nonlinear tires but intentionally differ in fidelity (Table II), creating a deployment-relevant modelmismatch setting. Platform-specific parameters and actuator bounds are listed in Table III. The platforms differ in aerodynamics, load transfer, actuation envelopes, tires, and mass/inertia, making the weight-to-behavior map platformdependent and requiring distinct bounds and policies. We obtained the bounds for the 7-dimensional weight vector $\theta _ { t }$ in (39) empirically from platform-specific fixed-weight NMPC tuning.

TABLE II: Key differences between the nonlinear single-track simulation and NMPC prediction models.
<table><tr><td colspan="2">Sim. Model</td><td>NMPC Pred. Model</td></tr><tr><td>Inputs</td><td> $a _ { x } , \delta$ </td><td> $j _ { x } , \dot { \delta }$ </td></tr><tr><td>Actuator dynamics</td><td>Explicit with rate saturation</td><td>simplified</td></tr><tr><td>Tire model</td><td>Wheel-level, Pacejka MF96 [51] combined slip</td><td>Axle-level, reduced Pacejka MF [51] no combined-slip</td></tr><tr><td>Load transfer</td><td>Full, with roll, lateral and longitudinal effects</td><td>Simplified, mainly longitudinal</td></tr><tr><td>Aerodyn.</td><td>Full drag, front/rear downforce, velocity-dependent</td><td>Simplified drag, lumped downforce, no vel. dependency</td></tr></table>

TABLE III: Single-track vehicle parameters and actuation bounds for the Dallara AV-24 and EAV-24 Super Formula.
<table><tr><td>Parameter</td><td>AV-24</td><td>EAV-24</td></tr><tr><td>Wheelbase l [m]</td><td>2.971</td><td>3.115</td></tr><tr><td>CG–front axle  $l _ { f }$  [m]</td><td>1.724</td><td>1.802</td></tr><tr><td>Mass m [kg]</td><td>800</td><td>782</td></tr><tr><td>Yaw inertia  $I _ { z } ~ [ \mathrm { k g } \mathrm { m } ^ { 2 } ]$ </td><td>1000</td><td>700</td></tr><tr><td>Front track  $b _ { f }$  [m]</td><td>1.639</td><td>1.606</td></tr><tr><td>Rear track  $b _ { r }$  [m]</td><td>1.524</td><td>1.510</td></tr><tr><td>Steering δ [rad]</td><td>[-0.276, 0.276]</td><td>[−0.342, 0.342]</td></tr><tr><td>Steering rate  $\dot { \delta }$  [rad/s]</td><td>[−0.429, 0.429]</td><td>[−1.050, 1.050]</td></tr></table>

a) Benchmarks: We benchmark our four proposed SG-RL methods against four reference baselines: the RL PPO baseline, a fixed-weight human-expert NMPC controller, the reduced GB-PL baseline from [5], and our GB-PL baseline.

b) NMPC solver configuration: We solve the NMPC problem with acados [48] using a real-time SQP-RTI scheme and full-condensing HPIPM as the QP backend. QP iterations were limited to 25 and equality and stationarity tolerances were set to $1 0 ^ { - 3 }$

c) Timing, horizons, and rollout length: The plant simulator runs at $T _ { \mathrm { s , s i m } } ~ = ~ 0 . 0 2 \mathrm { s } .$ . The NMPC optimal control problem is discretized with $T _ { \mathrm { s } } ~ = ~ 0 . 0 7 5 \mathrm { s }$ over a prediction horizon of $T _ { \mathrm { p } } = 2 . 6 \mathrm { s } .$ , yielding $N = \lfloor T _ { \mathrm { p } } / T _ { \mathrm { s } } \rfloor = 3 4$ shooting intervals. At each simulator step $( T _ { \mathrm { s , s i m } } = 0 . 0 2 \mathrm { s } )$ , the policy outputs weights, NMPC solves, and the first input is applied. Each $T _ { \mathrm { r o l l } } ~ = ~ 1 3 5 \mathrm { s }$ rollout therefore has 6750 simulator steps/policy decisions/control updates/solves, while prediction discretization remains $T _ { s } = 0 . 0 7 5 \mathrm { { s } }$ s. Episodes terminate after repeated solver failures or track departures, with a trainingonly truncation penalty.

d) Learning algorithm and hyperparameters: For fairness, PPO and all SG-RL methods share the same RL setup. Only GB-PL differs. RL-based methods use Stable-Baselines3 PPO [56] with a Gaussian MLP actor–critic, bounded normalized outputs de-normalized to platform-specific weight bounds, generalized state-dependent exploration, 12 parallel environments, and deterministic evaluation every rollout. PPO trains for 2M steps, whereas SG-RL trains for 1M to test whether solver guidance halves the sample budget. Because GB-PL becomes unstable with large rollouts/batches, we follow [5] and use 10-transition fragments/minibatches with one training environment. The RL reward is $\mathrm { - } \mathcal { L } _ { \mathrm { p e r f } }$ (Eq. (41)).

![](images/6c8a923b1cb22e5cf1474dca50124fb2ab2bc21bb0f701786ea2cafd7a3aa114.jpg)

![](images/75d95d7100abbd39de7d9fa7397ec3b9d0446b0593518a4c62f9fc70618e39eb.jpg)  
Fig. 7: Deterministic evaluation return during training over 10 seeds. Lines show seed means, shaded bands show seed ranges, stars mark best mean returns, and circles mark when SG-RL first surpasses PPO’s best mean return.

![](images/0d5a429735ac0acd56506d1f4e2caeb3eb67f944a4cc3205049d73265595f455.jpg)  
Fig. 8: Best deterministic evaluation return versus training samples for the AV-24/Monza budget comparison.

e) Policy initialization: We initialize policy linear layers with orthogonal weights and zero biases, a standard PPO choice that balances initial signal scales. Unlike [5], which uses a pre-optimized output bias, we set this bias to zero so the initial normalized mean action lies near the admissible hyper-rectangle center.

f) Hardware: Experiments ran on a 2022 MacBook Air (Apple M2, 16 GB RAM): PPO took 45 min for 2M training samples and SG-RL 20 min for 1M; on Apple M4, the times dropped to 28 min and 14 min.

## B. Key Highlights: Performance and Sample Efficiency

Figs. 7 and 8 summarize closed-loop performance versus sample efficiency. On AV-24/Monza, PPO reaches its highest mean return ( 62.29) after 1.215  10<sup>6</sup> samples, whereas all four SG-RL methods surpass it earlier: SG-SCA after 4.86 10<sup>5</sup> samples (60.0% fewer), SG-LOS/SG-ADV after 5.67 10<sup>5</sup> (53.3% fewer), and SG-CRT after 6.48 10<sup>5</sup> (46.7% fewer). SG-SCA also improves PPO’s peak mean reward by 33.2%, while SG-LOS, SG-CRT, and SG-ADV gain 20.4%, 18.5%, and 14.1%.

The same picture holds on EAV-24/Yas Marina, with even larger relative sample-efficiency gains for the strongest methods. SG-SCA and SG-LOS both surpass PPO’s highest mean return after only 4.05 10<sup>5</sup> samples, i.e., 70.6% earlier, whereas SG-ADV and SG-CRT still retain 47.1% and 35.3% sample savings. The min–max bands also suggest improved training stability under solver-gradient guidance: PPO shows the widest persistent scatter in reward evolution, whereas SG-SCA and SG-LOS concentrate earlier around high-return regimes. Taken together, the mean training curves show that solver-gradient guidance consistently improves PPO sample efficiency, with SG-SCA and SG-LOS providing the strongest cross-platform trade-off.

Fig. 8 complements this cross-seed view with a singleseed AV-24/Monza budget comparison. All SG-RL methods remain far above our GB-PL baseline, with best-return gains of 54.3%–59.5% over our GB-PL and 68.5%–72.0% over the reduced GB-PL baseline. At the same time, they require only 8.10 10<sup>5</sup> to 9.72 10<sup>5</sup> samples, i.e., 50.0%–58.3% fewer than PPO’s 1.94  10<sup>6</sup> budget. SG-SCA and SG-LOS provide the clearest trade-off, combining the lowest SG-RL sample counts with the strongest returns relative to GB-PL. These results support the grey-box thesis: solver sensitivities improve exploration efficiency without sacrificing PPO stability.

Within the GB-PL family, Table IV shows that our GB-PL baseline also substantially outperforms the reduced GB-PL baseline from [5]: total return improves from 118.62 to 81.87, i.e., by 31.0%, while the required training data drop from 94K to 6K samples, i.e., by 92.9%. These gains are consistent with the architectural and training-loop changes introduced in Tab. I and Sec. IV.

## C. Closed-Loop Performance Metrics

Table IV isolates total return and sample budget for the single-seed AV-24/Monza comparison. All SG-RL methods outperform our GB-PL baseline, with SG-SCA best overall; SG-SCA and SG-LOS need only 810K samples, 58.3% fewer than PPO’s 1.94M budget. Lap time is not the primary metric;

TABLE IV: Closed-loop performance and training-sample budget on Monza over 120 s. Best, second-best, and third-best entries are highlighted in green, bold, and underline.
<table><tr><td>Method</td><td>Total return ↑</td><td>Training samples ↓</td></tr><tr><td>Human Expert</td><td>-993.53</td><td></td></tr><tr><td>Reduced GB-PL [5]</td><td>-118.62 -81.87</td><td>94K 6K</td></tr><tr><td>GB-PL (ours)</td><td>(+31.0% vs red. GB-PL (SOTA))</td><td>(-92.9% vs red. GB-PL (SOTA))</td></tr><tr><td>RL PPO (SOTA)</td><td>-38.31 (+53.2% vs GB-PL)</td><td>1.94M</td></tr><tr><td>SG-SCA (ours)</td><td>-33.17 (+59.5% vs GB-PL)</td><td>810K (-58.3% vs PPO)</td></tr><tr><td>SG-LOS (ours)</td><td>-34.69 (+57.6% vs GB-PL)</td><td>810K (-58.3% vs PPO)</td></tr><tr><td>SG-ADV (ours)</td><td>-37.41 (+54.3% vs GB-PL)</td><td>972K (-50.0% vs PPO)</td></tr><tr><td>SG-CRT (ours)</td><td>-35.55 (+56.6% vs GB-PL)</td><td>972K (-50.0% vs PPO)</td></tr></table>

TABLE V: Closed-loop metrics on Monza over 120 s: lateral-/velocity deviation $e _ { \mathrm { l a t } } / e _ { v }$ , jerk j, and steering rate <sup>˙</sup>δ.
<table><tr><td>Method</td><td>|elat| [m]</td><td>max  $| e _ { \mathrm { l a t } } |$  [m]</td><td> $\overline { { | e _ { v } | } }$  [m/s]</td><td>max|j| max |δ|  $[ \mathrm { m } / \mathrm { s } ^ { 3 } ]$ </td><td>[rad/s]</td></tr><tr><td>Human Expert</td><td>0.167</td><td>1.50</td><td>0.40</td><td>16.16</td><td>0.41</td></tr><tr><td>Reduced GB-PL [5]</td><td>0.049</td><td>0.83</td><td>0.25</td><td>127.62</td><td>0.41</td></tr><tr><td>GB-PL (ours)</td><td>0.044</td><td>0.66</td><td>0.22</td><td>83.44</td><td>0.39</td></tr><tr><td>RL PPO (SOTA)</td><td>0.035</td><td>0.38</td><td>0.14</td><td>26.60</td><td>0.35</td></tr><tr><td>SG-SCA (ours)</td><td>0.034</td><td>0.33</td><td>0.15</td><td>42.55</td><td>0.34</td></tr></table>

![](images/d0a43aa2958c575df112c3933c7c6daee1d51cbf464343171c4f297b0fb918b9.jpg)  
Fig. 9: Zero-shot evaluation on race tracks unseen during training.

the optimized objective is Eq. (41), which balances path tracking, velocity tracking, and control smoothness.

Table V reports physical closed-loop metrics. We show SG-SCA as representative because the SG-RL variants have similar Monza statistics. The fixed-weight expert is smooth in maximum jerk but tracks poorly, while both GB-PL methods are more aggressive, especially in peak jerk. PPO and SG-SCA achieve the tightest tracking and lowest maximum steering rates, so SG-RL’s sample-efficiency gains do not sacrifice closed-loop quality.

## D. Generalization in Unseen Environments

Although training uses one circuit per platform, the policies are designed to be track-agnostic: they do not receive track coordinates and instead condition weight selection on local state and short-horizon reference information (Sec. III). We therefore evaluate each learned policy zero-shot on unseen racelines (Fig. 9). Table VI shows robust transfer and large gains over our white-box GB-PL baseline on Laguna Seca and Suzuka. On Laguna Seca, the PPO baseline already improves on our GB-PL baseline by 63.7%, and all four SG-RL methods improve further to 65.6%–68.6%; the strongest transfer result is obtained by SG-CRT at 68.6%, followed closely by SG-LOS at 67.0% and SG-SCA at 66.4%. On Suzuka, PPO improves on our GB-PL baseline by 55.8%, whereas SG-SCA reaches 62.1%, SG-CRT 59.6%, SG-LOS 58.8%, and SG-ADV 56.1%. Notably, these transfer gains use the single-seed AV-24/Monza policies trained with only $8 . 1 0 \times 1 0 ^ { 5 }$ to $9 . 7 2 \times 1 0 ^ { 5 }$ interaction samples, i.e., about 50%–58% fewer than $\mathrm { P P O ^ { \prime } s 1 . 9 4 \times 1 0 ^ { 6 } }$

TABLE VI: Zero-shot evaluation of trained policies on tracks unseen during training. Each cell reports total return and the relative improvement compared to the GB-PL baseline.
<table><tr><td>Method</td><td>Laguna Seca Zero-shot</td><td>Suzuka Zero-shot</td></tr><tr><td>GB-PL baseline</td><td>-120.59</td><td>-194.15 -85.81</td></tr><tr><td>RL PPO baseline</td><td>-43.73 (+63.7% vs GB-PL) (+55.8% vs GB-PL)</td><td></td></tr><tr><td>SG-SCA (ours)</td><td>-40.51 (+66.4% vs GB-PL)(+62.1% vs GB-PL)</td><td>-73.52</td></tr><tr><td>SG-LOS (ours)</td><td>-39.77 (+67.0% vs GB-PL) (+58.8% vs GB-PL)</td><td>-79.92</td></tr><tr><td>SG-ADV (ours)</td><td>-41.48 (+65.6% vs GB-PL) (+56.1% vs GB-PL)</td><td>-85.25</td></tr><tr><td>SG-CRT (ours)</td><td>-37.89 (+68.6% vs GB-PL) (+59.6% vs GB-PL)</td><td>-78.37</td></tr></table>

![](images/e5692d160afdd4c38a75a2dbf6b17baf96e56ff14fb4d03098b6d5504fb58939.jpg)  
Fig. 10: Correlations between learned NMPC cost weights and grouped observation features on Monza. Bars compare which signals each policy uses to adapt each weight.

These zero-shot results indicate that the policy learns a transferable map from local driving context to NMPC weight rebalancing rather than overfitting to a raceline. SG-RL maintains, and often improves on, PPO’s unseen-track performance.

## E. Explaining Cost-Weight Policies via Correlation Structure

Because policies output cost weights, not controls, we interpret adaptation via correlations between observation groups and weights, and via inter-weight correlations during rollout. Fig. 10 reports signed Pearson correlations with six grouped features, while Fig. 11 shows weight co-variation. These statistics indicate associations, not causality. Across methods, the dominant dependencies differ (Figs. 10–11). Overall, all policies condition strongly on velocity signals, but solvergradient guidance changes which weights are most strongly tied to anticipation versus instantaneous state. The GB-PL baseline exhibits a near-direct velocity-aware structure: $q _ { v }$ is almost perfectly aligned with reference velocity $( \rho = 0 . 9 8 )$ while $q _ { a }$ anti-correlates with it $( \rho ~ = ~ - 0 . 8 6 )$ , yielding an optimization-like trade-off also visible as $\rho ( q _ { a } , q _ { v } ) = - 0 . 8 7 .$ The RL baseline similarly shows strong velocity conditioning (e.g., $\rho ( q _ { a }$ , ref. velocity) = 0.79 and $\rho ( q _ { a _ { y } }$ , ref. velocity) $=$ $- 0 . 8 0 )$ , but its interdependencies reveal an almost rank-one “see-saw” between aggressiveness/smoothness and lateral correction $( \mathbf { e } . \mathbf { g } . , \rho ( r _ { j } , q _ { v } ) = 0 . 9 8$ and $\rho ( q _ { v } , q _ { \ell } ) = - 0 . 9 7 )$ .

![](images/f529bb139029c555afb2e616b295771ef85daa58eb2acf1c33e9e2470d2c9f40.jpg)  
Fig. 11: Pairwise correlations between learned NMPC cost weights along Monza rollouts. Each sub-cell compares one policy, with color indicating correlation strength.

Solver-gradient guided methods exhibit distinct, methodspecific structure. SG-SCA assigns curvature-driven modulation primarily to steering smoothness: $r _ { \dot { \delta } }$ correlates more with reference curvature $( \rho = - 0 . 7 2 )$ than with reference velocity $( \rho ~ = ~ - 0 . 4 0 )$ , while lateral tracking weights anti-correlate $( \rho ( q _ { \psi } , q _ { \ell } ) = - 0 . 9 5 )$ , indicating a systematic lateral tradeoff. SG-LOS concentrates its strongest velocity dependence on steering smoothness $( \rho ( r _ { \dot { \delta } }$ , ref. velocity) $= ~ - 0 . 8 1 )$ and is the only method where longitudinal aggressiveness is primarily curvature-associated $( \rho ( q _ { a }$ , ref. curvature) $= ~ - 0 . 4 7 )$ suggesting geometry-aware tightening/relaxation of acceleration shaping. In both guided methods, positive couplings such as $\rho ( r _ { j } , q _ { v } ) \approx 0 . 9 8 – 0 . 9 9$ indicate coordinated velocity/smoothness scaling, while negative pairs (e.g., $\rho ( q _ { \psi } , q _ { \ell } ) =$ 0.99 for SG-LOS) reveal objective rebalancing rather than uniform tightening.

## F. Closed-Loop Weight Adaptation Dynamics Along the Track

Correlations identify which signals associate with each weight; Fig. 12 adds the temporal view of how weights are scheduled along the lap, separating methods more clearly than static summaries.

![](images/96a1f55de1f3f8522bc893e6f22b2b641f35f753f83f3eb5046cc789f8c6b48e.jpg)  
Fig. 12: Time evolution of learned cost weights during a Monza lap rollout. Reference velocity and curvature are shown above the seven NMPC weights; GB-PL uses separate axes where its scale differs strongly.

RL PPO baseline: coarse but behavior-linked scheduling. PPO keeps $q _ { \ell }$ and $q _ { a }$ near upper ranges, varies $q _ { \psi }$ moderately, and mainly activates $r _ { j } , \ r _ { \dot { \delta } }$ , and short $q _ { v }$ bursts around braking/turn-in. Thus, PPO adapts event-wise rather than tracing the reference-velocity template, but through a lowdimensional aggressiveness–smoothness mode.

SG-SCA: sparse, targeted reweighting. SG-SCA preserves PPO’s sparse character but shifts interventions: it adds sharper localized $q _ { \psi } , \ q _ { v }$ , and $q _ { a _ { y } }$ activations near dominant curvature/deceleration zones while keeping jerk and steeringrate regularization restrained. The schedule is selective, reinforcing task-relevant penalties only in specific segments.

SG-LOS: coordinated smoothness gating. SG-LOS most clearly co-schedules smoothness-related objectives: $q _ { a } , r _ { j }$ , and $r _ { \dot { \delta } }$ rise and fall together, while $q _ { \psi }$ and $q _ { v }$ stay small except for short corrections. This matches training diagnostics: the guidance loss synchronizes updates in performance-critical regions rather than simply increasing PPO aggressiveness.

GB-PL baseline: velocity-template overfitting. GB-PL shows the strongest continuous variation in $q _ { v }$ and $\boldsymbol { r } _ { \dot { \delta } } .$ , plus the largest $q _ { a _ { y } }$ peaks at high-curvature events. Its $q _ { v }$ trajectory nearly reproduces the reference-velocity profile, suggesting velocity-weight overfitting. By contrast, RL and SG-RL keep q<sub>v</sub> smaller and activate it jointly with lateral/smoothness weights where closed-loop behavior demands it, matching their stronger returns.

![](images/a1a92ab61662a0f2a073673b6ee3c328e5478156f044a9d7a21483bd39be561b.jpg)  
Fig. 13: Observation-ablation study for PPO and SG-RL methods. Curves compare full observations against variants without task anticipation or performance history.

TABLE VII: Effect of removing task-anticipation and performance-history features. Each cell reports ablated reward and degradation relative to the full-observation policy.
<table><tr><td>Method</td><td>No anticipation</td><td>No history</td></tr><tr><td>GB-PL baseline</td><td>-123.79 (-51.2% vs full)</td><td>-94.08 (-14.9% vs full)</td></tr><tr><td>RL PPO baseline</td><td>-74.07 (-93.4% vs full)</td><td>-81.58 (-113.0% vs full)</td></tr><tr><td>SG-SCA (ours)</td><td>-65.95 (-98.8% vs full)</td><td>-73.61 (-121.9% vs full)</td></tr><tr><td>SG-LOS (ours)</td><td>-93.96 (-170.9% vs full)</td><td>-82.36 (-137.4% vs full)</td></tr><tr><td>SG-ADV (ours)</td><td>-96.49 (-157.9% vs full)</td><td>-94.08 (-151.5% vs full)</td></tr><tr><td>SG-CRT (ours)</td><td>-72.20 (-103.1% vs full)</td><td>-63.98 (-80.0% vs full)</td></tr></table>

## G. Ablations: Removing Future and Past Context

Sec. III assumes that effective adaptation needs two channels: task anticipation from reference lookahead and performance history for compensating persistent errors or unmodeled dynamics. We test this with two ablations that keep the NMPC formulation, constraints, and reference fixed while restricting policy information: no anticipation removes horizon-sampled reference velocity/curvature, and no history removes recent lateral/velocity deviations. Fig. 13 shows learning curves; Table VII reports peak-reward degradation.

1) Ablation 1 (No anticipation): The Necessity of Preview: Removing preview information (dashed curves in Fig. 13) systematically degrades performance: the policy becomes feedback-only and cannot anticipate curvature or deceleration. It therefore modulates weights only after entering challenging segments, confirming the anticipation-aware structure in Figs. 10–11. Among guided methods, SG-SCA is most robust to losing anticipation, while SG-CRT still outperforms RL,

TABLE VIII: Core PPO signals in performance-relevant windows. Entries are window means for the winning SG-RL method and PPO baseline, reported as SG/PPO. KL and clip fraction indicate trust-region activity; value, policygradient, and entropy losses summarize critic fit, actor update magnitude, and exploration; weight/bias norms refer to the actor action head.

<table><tr><td rowspan="2">Signal</td><td colspan="2">AV-24 Monza</td><td colspan="2">EAV-24 Yas Marina</td></tr><tr><td>Acceleration window winner: SG-LOS</td><td>Peak window winner: SG-SCA</td><td>Acceleration window winner: SG-SCA</td><td>Peak window winner: SG-LOS</td></tr><tr><td>Eval return</td><td>-67.3/ - 81.6</td><td>-44.0/ - 68.4</td><td>-93.6/ - 111</td><td>-86.4/ - 103</td></tr><tr><td>KL</td><td>0.0935/0.161</td><td>0.0337/0.0721</td><td>0.107/0.164</td><td>0.0567/0.0657</td></tr><tr><td>Clip frac.</td><td>0.309/0.366</td><td>0.225/0.314</td><td>0.337/0.377</td><td>0.254/0.284</td></tr><tr><td>Value loss</td><td>296/135</td><td>1.51e3/502</td><td>0.750/0.774</td><td>2.55/6.33</td></tr><tr><td>|PG loss|</td><td>0.0328/0.0629</td><td>9.48e-3/0.0307</td><td>0.0466/0.0641</td><td>0.0165/0.0247</td></tr><tr><td>|Entropy loss|</td><td>8.31/9.26</td><td>6.99/8.37</td><td>8.79/9.16</td><td>6.63/7.57</td></tr><tr><td>Weight norm</td><td>2.67/2.65</td><td>2.69/2.66</td><td>2.67/2.66</td><td>2.73/2.69</td></tr><tr><td>Bias norm</td><td>7.23/1.99e-2</td><td>7.23/2.44e-2</td><td>1.63/1.51</td><td>1.47/1.51</td></tr></table>

suggesting gradient-informed value estimation remains useful when feedforward context is reduced.

2) Ablation 2 (No history): The Role of Temporal Context: Eliminating performance-history features (solid curves) also degrades reward, but more heterogeneously than removing anticipation. PPO and SG-SCA suffer larger losses without history than without preview, while SG-LOS, SG-ADV, and SG-CRT retain more of their full-observation performance than in the no-anticipation case. With only a memoryless state snapshot, the policy cannot distinguish transient execution errors from persistent performance deviations. As a result, it cannot reliably modulate the persistence or magnitude of weight adjustments, which impairs closed-loop consistency over extended operation. This aligns with Sec. IX-E, where deviationhistory terms strongly correlate with regulatory weights.

Our GB-PL baseline degrades only modestly, consistent with its myopic update: it descends a local sensitivity gradient <sub>θ</sub> on the current trajectory and depends less on history. This robustness costs optimality, since GB-PL remains below the best SG-guided methods in absolute reward (Table VII).

Advantage of solver gradients under information scarcity. As Table VII shows, the strongest SG-guided methods continue to outperform the pure PPO baseline in peak reward even after we remove temporal context.

## H. Training-Dynamics Analysis

a) Core PPO training dynamics in the windows where SG-RL leads.: To explain why certain SG-RL methods accelerate PPO, we inspect PPO signals in an early acceleration window, where a method first pulls ahead, and a later peak window, where it reaches best performance. Table VIII compares the winning SG method with PPO in these windows. Table VIII shows that winning SG methods beat PPO through more structured, not indiscriminately larger, updates: they regularize PPO toward better-timed and lower-variance updates.

In AV-24/Monza, SG-LOS wins the early phase, improving the windowed return from 81.6 to 67.3 compared to PPO, while reducing KL from 0.161 to 0.0935 and clip fraction from 0.366 to 0.309. The peak window is then won by SG-SCA, which lifts the mean return from 68.4 to 44.0 with roughly half the PPO KL (0.0337 vs. 0.0721), a lower clip fraction, and a much smaller PG loss . On this platform, the action-head weight norm remains close to PPO, while the bias norm separates more strongly for the winning SG methods.

TABLE IX: Method-specific SG-RL signals in performance-relevant windows. Entries are window means in the same windows as Table VIII. For SG-SCA, $\bar { c } / \bar { s } / \bar { f } _ { \mathrm { c l a m p } }$ denote alignment, projected scale, and clamp-hit fraction; for SG-LOS, $\bar { \ell } _ { \mathrm { g u i d e } }$ is the guide loss; for SG-ADV, $\bar { \rho } _ { \mathrm { A S } } / \rho _ { \mathrm { A S } } ^ { 9 0 }$ quantify advantage correction; for SG-CRT, ρ¯ $\bar { \rho } _ { V }$ is the critic correction ratio.
<table><tr><td rowspan="2">Method / signal</td><td colspan="2">AV-24 Monza</td><td colspan="2">EAV-24 Yas Marina</td></tr><tr><td>Accel.</td><td>Peak</td><td>Accel.</td><td>Peak</td></tr><tr><td>SG-SCA: ē</td><td>-1.42e-3</td><td>-0.0140</td><td>8.02e-3</td><td>0.0215</td></tr><tr><td>SG-SCA: §</td><td>0.986</td><td>0.944</td><td>1.05</td><td>1.05</td></tr><tr><td>SG-SCA:  $\underline { { \bar { f } } } _ { \mathrm { c l a m p } }$ </td><td>0.527</td><td>0.168</td><td>0.541</td><td>0.452</td></tr><tr><td>SG-LOS:  $\ell _ { \mathrm { g u i d e } }$ </td><td>2.69e-3</td><td>1.98e-3</td><td> $3 . 6 1 \mathrm { e } { - 3 }$ </td><td>2.89e-3</td></tr><tr><td>SG-ADV:  $\bar { \rho } _ { \mathrm { A S } }$ </td><td>0.336</td><td>0.447</td><td>0.0116</td><td>0.0155</td></tr><tr><td>SG-ADV:  $\rho _ { \mathrm { A S } } ^ { 9 0 }$ </td><td>0.356</td><td>0.499</td><td>0.0118</td><td>0.0159</td></tr><tr><td>SG-CRT:  $\bar { \rho } _ { V }$ </td><td>1.99</td><td>1.34</td><td>0.0451</td><td>0.0478</td></tr></table>

In EAV-24/Yas Marina, the lead changes by phase: SG-SCA wins the acceleration window and SG-LOS wins the peak window. During training, SG-SCA improves mean return from 111 to 93.6 while lowering KL, clip fraction, value loss, PG loss , and entropy loss . Near peak performance, SG-LOS improves from 103 to 86.4 and again lowers KL, clip fraction, value loss, PG loss , and entropy loss .

b) Method-specific guidance in performance-relevant windows.: Table IX resolves the window winners into their active guidance channels. On AV-24/Monza, SG-LOS keeps nonzero acceleration-window guidance $( \bar { \ell } _ { \mathrm { g u i d e } } = 2 . 6 9 \times 1 0 ^ { - 3 } )$ so early gains reflect solver-biased shaping while PPO is noisy. For SG-SCA, peak-window c¯ becomes more negative, $\bar { s } < 1$ and $f _ { \mathrm { c l a m p } }$ drops from 0.527 to 0.168, indicating selective braking of PPO steps that drift from solver-informed directions rather than persistent acceleration or repeated clamp saturation.

On EAV-24/Yas Marina, SG-SCA’s sign pattern flips: c¯ stays positive and grows from $8 . 0 2 \times 1 0 ^ { - 3 }$ to 0.0215, with $\bar { s } \approx 1 . 0 5$ so it nudges PPO along solver-aligned directions and wins the acceleration window. SG-LOS remains strong at peak because its guidance term stays large $( 3 . 6 1 \times 1 0 ^ { - 3 }$ early, $2 . 8 9 \times 1 0 ^ { - 3 }$ at peak), continuing to shape the actor objective. SG-ADV and SG-CRT mainly support platform-specific gains: large $\bar { \rho } _ { \mathrm { A S } }$ or $\bar { \rho } _ { V }$ indicate strong advantage/critic reshaping, which is pronounced on Monza but weak on Yas Marina.

## X. CONCLUSIONS

We introduced Solver-Gradient Guided Reinforcement Learning (SG-RL), a framework for online MPC cost-weight adaptation that combines environment-based policy-gradient learning with model-based solver-gradient guidance while preserving the realized closed-loop return as the optimization objective. Our SG-RL uses solver sensitivities as auxiliary guidance signals for RL to improve policy learning.

We instantiated this framework through four methods— SG-SCA, SG-LOS, SG-ADV, and SG-CRT—that incorporate solver guidance into different components of PPO. Across two full-scale autonomous racing platforms under intentional model mismatch, all four methods consistently improved sample efficiency and closed-loop performance over PPO, matched PPO’s best return with up to 70.6 % fewer training samples, outperformed GB-PL baselines by at least 54 % in return, and generalized zero-shot to unseen race tracks. Among the proposed methods, SG-SCA and SG-LOS achieved the strongest overall performance.

Overall, our results indicate that environment-based policy gradients and model-based solver guidance are complementary optimization signals for learning online MPC weightadaptation policies. More broadly, they demonstrate the potential of combining environment interaction with differentiable optimal-control structure for learning in constrained optimalcontrol problems.

Limitations: The effectiveness of SG-RL depends on the fidelity of the predictive MPC model. Under severe model mismatch, solver guidance may become less informative or even misleading for optimizing the true closed-loop objective. Although our experiments include intentional model mismatch and demonstrate robust improvements, they do not systematically study different mismatch types or magnitudes. In addition, each SG-RL variant introduces method-specific hyperparameters, and this work considers only on-policy PPO.

Future work: Future work includes validating the learned weight-adaptation policies on a physical vehicle, extending solver-gradient guidance to off-policy algorithms such as SAC and TD3, and systematically studying robustness under controlled model-mismatch regimes. Another promising direction is to generalize SG-RL beyond NMPC cost-weight adaptation to other differentiable parametric controllers. Finally, developing a stronger theoretical understanding of how solver guidance interacts with stochastic policy-gradient optimization remains an important open research direction.

## REFERENCES

[1] J. B. Rawlings, D. Q. Mayne, and M. Diehl, Model predictive control: theory, computation, and design, 2nd ed. Madison, Wisconsin: Nob Hill Publishing, 2017.

[2] A. Romero, Y. Song, and D. Scaramuzza, “Actor-Critic Model Predictive Control,” in 2024 IEEE International Conference on Robotics and Automation (ICRA). Yokohama, Japan: IEEE, May 2024.

[3] B. Zarrouki, V. Klos, N. Heppner, S. Schwan, R. Ritschel, and¨ R. Voßwinkel, “Weights-varying mpc for autonomous vehicle guidance: a deep reinforcement learning approach,” in 2021 European Control Conference (ECC), 2021.

[4] B. Zarrouki, M. Spanakakis, and J. Betz, “A safe reinforcement learning driven weights-varying model predictive control for autonomous vehicle motion control,” in 2024 IEEE Intelligent Vehicles Symposium (IV), 2024, pp. 1401–1408.

[5] F. Jahncke, B. Zarrouki, M. Piccinini, J. D’sa, D. Isele et al., “Differentiable weights-varying nonlinear mpc via gradient-based policy learning: An autonomous vehicle guidance example,” IEEE Robotics and Automation Letters, vol. 11, no. 3, pp. 3724–3731, 2026.

[6] J. Betz, H. Zheng, A. Liniger, U. Rosolia, P. Karle et al., “Autonomous vehicles on the edge: A survey on autonomous vehicle racing,” IEEE Open Journal of Intelligent Transportation Systems, vol. 3, pp. 458–488, 2022.

[7] L. Franceschi, M. Donini, P. Frasconi, and M. Pontil, “Bilevel programming for hyperparameter optimization and meta-learning,” in Proceedings of the 35th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 80. PMLR, 2018, pp. 1568–1577.

[8] V. Ramasamy, R. K. Sidharthan, R. Kannan, and G. Muralidharan, “Optimal tuning of model predictive controller weights using genetic algorithm with interactive decision tree for industrial cement kiln process,” Processes, vol. 7, no. 12, p. 938, 2019.

[9] L. L. Rodrigues, A. S. Potts, O. A. Vilcanqui, and A. J. Sguarezi Filho, “Tuning a model predictive controller for doubly fed induction generator employing a constrained genetic algorithm,” IET Electric Power Applications, vol. 13, no. 6, pp. 812–819, 2019.

[10] F. Sorourifar, G. Makrygirgos, A. Mesbah, and J. A. Paulson, “A data-driven automatic tuning method for mpc under uncertainty using constrained bayesian optimization,” IFAC-PapersOnLine, vol. 54, no. 3, pp. 243–250, 2021.

[11] A. Rupenyan, M. Khosravi, and J. Lygeros, “Performance-based trajectory optimization for path following control using bayesian optimization,” in 2021 60th IEEE Conference on Decision and Control (CDC). IEEE, 2021, pp. 2116–2121.

[12] R. Reiter, J. Hoffmann, D. Reinhardt, F. Messerer, K. Baumgartner¨ et al., “Synthesis of model predictive control and reinforcement learning: Survey and classification,” Annual Reviews in Control, vol. 61, p. 101045, 2026.

[13] R. Reiter, J. Hoffmann, J. Boedecker, and M. Diehl, “A hierarchical approach for strategic motion planning in autonomous racing,” in 2023 European Control Conference (ECC), 2023, pp. 1–8.

[14] E. Bøhn, S. Gros, S. Moe, and T. A. Johansen, “Optimization of the model predictive control meta-parameters through reinforcement learning,” Engineering Applications of Artificial Intelligence, vol. 123, p. 106211, 2023.

[15] E. Bøhn, S. Moe, S. Gros, and T. A. Johansen, “Reinforcement learning of the prediction horizon in model predictive control,” IFAC-PapersOnLine, vol. 54, no. 6, pp. 314–320, 2021.

[16] M. Zanon, S. Gros, and A. Bemporad, “Practical reinforcement learning of stabilizing economic mpc,” in 2019 European Control Conference (ECC), 2019.

[17] S. Gros and M. Zanon, “Data-driven economic nmpc using reinforcement learning,” IEEE Transactions on Automatic Control, 2020.

[18] ——, “Reinforcement learning for mixed-integer problems based on mpc,” in 21st IFAC World Congress, 2020.

[19] ——, “Reinforcement learning based on mpc and the stochastic policy gradient method,” in 2021 American Control Conference (ACC), 2021.

[20] A. B. Kordabad, W. Cai, and S. Gros, “Mpc-based reinforcement learning for economic problems with application to battery storage,” 2021.

[21] K. Seel, S. Gros, and J. T. Gravdahl, “Combining q-learning and deterministic policy gradient for learning-based mpc,” in 2023 IEEE Conference on Decision and Control (CDC), 2023.

[22] A. Martinsen, A. Lekkas, and S. Gros, “Combining system identification with reinforcement learning-based mpc,” 2020.

[23] B. Zarrouki, C. Wang, and J. Betz, “Adaptive Stochastic Nonlinear Model Predictive Control with Look-ahead Deep Reinforcement Learning for Autonomous Vehicle Motion Control,” in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). Abu Dhabi, United Arab Emirates: IEEE, Oct. 2024.

[24] S. Sawant, A. Anand, D. Reinhardt, and S. Gros, “Learning-based mpc from big data using reinforcement learning,” 2023.

[25] P. L. Donti, B. Amos, and J. Z. Kolter, “Task-based end-to-end model learning in stochastic optimization,” in Advances in Neural Information Processing Systems, vol. 30. Curran Associates, Inc., 2017.

[26] B. Amos and J. Z. Kolter, “Optnet: Differentiable optimization as a layer in neural networks,” in International conference on machine learning. PMLR, 2017, pp. 136–145.

[27] A. Agrawal, B. Amos, S. Barratt, S. Boyd, S. Diamond, and J. Z. Kolter, “Differentiable convex optimization layers,” in Advances in Neural Information Processing Systems, vol. 32. Curran Associates, Inc., 2019.

[28] B. Amos, I. Jimenez, J. Sacks, B. Boots, and J. Z. Kolter, “Differentiable mpc for end-to-end planning and control,” Advances in neural information processing systems, vol. 31, 2018.

[29] J. Frey, K. Baumgartner, G. Frison, D. Reinhardt, J. Hoffmann ¨ et al., “Differentiable Nonlinear Model Predictive Control,” 2025.

[30] L. Fichtner, D. Reinhardt, J. Hoffmann, F. Airaldi, J. Frey et al., “leapc/leap-c: v0.1.0-alpha,” 2025.

[31] R. Tao, S. Cheng, X. Wang, S. Wang, and N. Hovakimyan, “DiffTune-MPC: Closed-Loop Learning for Model Predictive Control,” IEEE Robotics and Automation Letters, vol. 9, no. 8, Aug. 2024.

[32] A. Romero, E. Aljalbout, Y. Song, and D. Scaramuzza, “Actor–critic model predictive control: Differentiable optimization meets reinforcement learning for agile flight,” IEEE Transactions on Robotics, 2025.

[33] W. Cai, K. G. Vamvoudakis, S. Gros, and A. Tzes, “Cost-matching model predictive control for efficient reinforcement learning in humanoid locomotion,” arXiv preprint arXiv:2603.28243, 2026.

[34] R. Rickenbach, A. A. Lahoud, E. Schaffernicht, M. N. Zeilinger, and J. A. Stork, “ZipMPC: Compressed Context-Dependent MPC Cost via Imitation Learning,” 2025, version Number: 1.

[35] S. Levine and P. Abbeel, “Learning neural network policies with guided policy search under unknown dynamics,” in Proceedings of the 31st International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 32. PMLR, 2014, pp. 1292–1300.

[36] N. Heess, G. Wayne, D. Silver, T. Lillicrap, T. Erez, and Y. Tassa, “Learning continuous control policies by stochastic value gradients,” in Advances in Neural Information Processing Systems, vol. 28. Curran Associates, Inc., 2015.

[37] A. Tamar, Y. Wu, G. Thomas, S. Levine, and P. Abbeel, “Value iteration networks,” in Advances in Neural Information Processing Systems, vol. 29. Curran Associates, Inc., 2016.

[38] P. Parmas, C. E. Rasmussen, J. Peters, and K. Doya, “PIPPS: Flexible model-based policy search robust to the curse of chaos,” in Proceedings of the 35th International Conference on Machine Learning, ser. ICML. PMLR, 2018, pp. 4065–4074.

[39] I. Clavera, Y. Fu, and P. Abbeel, “Model-augmented actor-critic: Backpropagation through paths,” in International Conference on Learning Representations, 2020.

[40] J. Xu, V. Makoviychuk, Y. Narang, F. Rber, W. Matusik et al., “Accelerated policy learning with parallel differentiable simulation,” in International Conference on Learning Representations, 2022.

[41] N. Hansen, H. Su, and X. Wang, “Temporal difference learning for model predictive control,” in Proceedings of the 39th International Conference on Machine Learning. PMLR, 2022, pp. 8387–8406.

[42] ——, “TD-MPC2: Scalable, robust world models for continuous control,” arXiv preprint arXiv:2310.16828, 2024.

[43] S. Schmitt, J. J. Hudson, A. Z´ıdek, S. Osindero, C. Doersch et al., “Kickstarting deep reinforcement learning,” CoRR, 2018.

[44] S. Adhau, D. Reinhardt, S. Skogestad, and S. Gros, “Fast reinforcement learning based mpc based on nlp sensitivities,” IFAC-PapersOnLine, vol. 56, no. 2, pp. 11 841–11 846, 2023, 22nd IFAC World Congress.

[45] H. J. T. Suh, M. Simchowitz, K. Zhang, and R. Tedrake, “Do differentiable simulators give better policy gradients?” in Proceedings ofthe 39th International Conference on Machine Learning, ser. ICML. PMLR, 2022, pp. 20 668–20 696.

[46] R. Reiter, A. Ghezzi, K. Baumgartner, J. Hoffmann, R. D. McAllister,¨ and M. Diehl, “Ac4mpc: Actor-critic reinforcement learning for guiding model predictive control,” IEEE Transactions on Control Systems Technology, vol. 34, no. 1, pp. 395–410, Jan 2026.

[47] J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel, “Highdimensional continuous control using generalized advantage estimation,” arXiv preprint arXiv:1506.02438, 2015.

[48] R. Verschueren, G. Frison, D. Kouzoupis, J. Frey, N. V. Duijkeren et al., “acados—a modular open-source framework for fast embedded optimal control,” Mathematical Programming Computation, no. 1, Mar. 2022.

[49] B. Zarrouki, C. Wang, and J. Betz, “A stochastic nonlinear model predictive control with an uncertainty propagation horizon for autonomous vehicle motion control,” in American Control Conference (ACC), 2024.

[50] B. Zarrouki, J. Nunes, and J. Betz, “R²NMPC: A Real-Time Reduced Robustified Nonlinear Model Predictive Control with Ellipsoidal Uncertainty Sets for Autonomous Vehicle Motion Control,” IFAC-PapersOnLine, vol. 58, no. 18, 2024.

[51] H. Pacejka and I. Besselink, “Magic formula tyre model with transient properties,” Vehicle system dynamics, vol. 27, pp. 234–249, 1997.

[52] M. Rowold, L. Ogretmen, U. Kasolowsky, and B. Lohmann, “Online<sup>¨</sup> time-optimal trajectory planning on three-dimensional race tracks,” in 2023 IEEE Intelligent Vehicles Symposium (IV), 2023, pp. 1–8.

[53] M. Piccinini, S. Taddei, M. Piazza, and F. Biral, “Impacts of g-gv constraints formulations on online minimum-time vehicle trajectory planning,” IFAC-PapersOnLine, vol. 58, no. 10, pp. 87–93, 2024, 17th IFAC Symposium on Control of Transportation Systems CTS 2024.

[54] S. Sagmeister, P. Kounatidis, S. Goblirsch, and M. Lienkamp, “Analyzing the impact of simulation fidelity on the evaluation of autonomous driving motion control,” in 2024 IEEE Intelligent Vehicles Symposium (IV), 2024, pp. 230–237.

[55] N. Musiu, E. Mascaro, A. Raji, A. D. Felice, S. Sorrentino, and M. Bertogna, “A comprehensive benchmark of vehicle dynamics models for autonomous racing: a deep dive into mpc,” Vehicle System Dynamics, pp. 1–33, 2026.

[56] A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann, “Stable-baselines3: Reliable reinforcement learning implementations,” Journal ofMachine Learning Research, vol. 22, no. 268, pp. 1–8, 2021.