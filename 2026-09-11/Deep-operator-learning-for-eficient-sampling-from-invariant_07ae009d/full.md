# Deep operator learning for eficient sampling from invariant measures of stochastic diferential equations

Ling Guo <sup>∗</sup> <sup>a</sup>, Lei Li <sup>†</sup> <sup>b,c</sup>, and Jingtong Zhang <sup>‡</sup> <sup>b</sup>

<sup>a</sup>Department of Mathematics, Shanghai Normal University, Shanghai, China <sup>b</sup>School of Mathematical Sciences, Shanghai Jiao Tong University, Shanghai, China

<sup>c</sup>Institute of Natural Sciences, MOE-LSC, Shanghai Jiao Tong University, Shanghai, China

## Abstract

We introduce an amortized neural sampler that combines operator learning with flow methods for sampling. It maps SDE coeficient functions to pushforwards from a reference measure to the invariant measures, enabling eficient sampling across families of stochastic diferential equations. Our framework shifts traditional sampling cost to an initial training phase, after which new SDE instances require only one encoder pass and a few ODE solver steps, independent of mixing time. To handle problems in high dimensions, we use Lagrangian trajectory sensors for the coeficient functions and cross attention in the architecture. We also theoretically establish the expressivity and resolution invariance of our framework. Experiments on 1D and 2D SDE families show competitive accuracy with substantial speedups over MCMC in regimes with slow mixing, transfer across sensor counts, and demonstration results on a 64D interacting particle SDE where traditional grid approaches are infeasible.

Keywords. operator learning, invariant measures, stochastic diferential equations, amor  
tized sampling, flow matching, uncertainty quantification   
MSC 2020. 60H10, 65C05, 68T07

## 1 Introduction

Sampling from the invariant measures of Stochastic Diferential Equations (SDEs) is a fundamental task in applied mathematics, statistical physics, statistics and computer science, which enables the computation of stationary expectations, free energies and rare event probabilities for uncertainty quantification and stochastic control in related applications [23, 26, 17]. Traditional methods, such as Markov Chain Monte Carlo (MCMC) [24, 11, 32], rely on sequential simulations of the dynamics. These methods are intrinsically sequential and generally produce correlated samples. They require an initial equilibration period and a suficiently long simulation to approach equilibrium, while time discretization may introduce an additional bias. More importantly, for multimodal or metastable systems, transitions between separated regions can result in a prohibitively large mixing time and a small efective sample size. When the drift or difusion coeficients vary across many problem instances, a new long simulation is usually required for every instance [32], and this repeated mixing cost can become the dominant computational bottleneck.

Operator learning [22, 18, 15], which directly learns the map between functions, is gaining interest in scientific machine learning, and provides a natural framework for amortizing this cost over a family of stochastic systems. Consider the coeficient-indexed Stochastic Diferential Equation (SDE) taking values in $\mathbb { R } ^ { d }$

$$
d X = b ( X ) d t + \sigma ( X ) d W ,\tag{1.1}
$$

with $X _ { 0 } ~ \sim ~ \mu _ { 0 }$ , drift $b : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ , difusion coeficient $\sigma : \mathbb { R } ^ { d }  \mathbb { R } ^ { d \times d _ { W } }$ , and standard Brownian motion W in $\mathbb { R } ^ { d _ { W } }$ . Both dimensions are fixed within each coeficient family. Write $a : = ( b , \sigma )$ for the pair of SDE coeficients throughout this paper, and denote its invariant measure by $\pi _ { a }$ . Rather than constructing a separate sampler for each $^ { a , }$ one may seek a generative sampler operator $s$ satisfying $S ( a , \cdot ) _ { \# } \gamma \approx \pi _ { a } ,$ where $\gamma$ is a simple reference distribution and $\#$ denotes the pushforward of measures. Thus, $s$ takes the drift and difusion functions, together with a latent random variable, as inputs and returns a sample from the corresponding invariant law. This provides a natural operator learning problem because the inputs a are functions. After an ofline training stage over a suitable family of coeficients, such an operator could be evaluated on previously unseen SDEs and generate samples in parallel without repeating a long mixing simulation for every new query.

The generative sampler has been extensively studied in recent years, including normalizing flows and the continuous versions [31, 7]; Boltzmann generators [25]; score-based diffusion models [35]. The flow matching approach [19, 20] provides a regression framework for training continuous normalizing flows by learning a target velocity field, while avoiding adversarial objectives. FlowKac uses temporal normalizing flows and the Feynman–Kac formula for Fokker–Planck equations [8]; the weak adversarial networks enforce the weak equation through a minimax problem involving trial and test networks [39], whereas the weak generative sampler uses randomized test functions to train a transport map directly and avoids the adversarial minimax optimization [5]. Although these methods can substantially reduce the cost of sampling a prescribed target, they are typically trained for a fixed distribution or a single stochastic system. The objective here is instead to combine generative sampling with operator learning. Let us have also brief review of the neural operators. Rep resentative neural operators include DeepONet [22], which encodes input functions through branch networks and evaluates outputs through a trunk network; the Fourier Neural Operator [18, 15], which parameterizes integral operator layers in the Fourier domain; multi-input operators (MIONet) [14], which accommodates several functional inputs through multiple branch networks; and Variable-Input Deep Operator Networks (VIDON) [29], which allows the number and locations of input sensors to vary between samples. Recently, generative neural operators have also been extended to learning probability laws on function spaces, first through adversarial neural operators [30] and more recently through operator flow matching for stochastic process priors [34, 12]. Huang and Lai use an empirical measure limit for sample-based attention [12]. Continuum attention also interprets discrete attention as an approximation of a population operator [6].

We propose an amortized neural sampler that uses neural operators and the flow matching [19] to construct a reusable map from previously unseen drift and difusion functions of the SDE (1.1) to samples from their corresponding invariant measures. Let A denote the space of admissible coeficient pairs. We assume a is drawn from a distribution π over A, and we have access to existing samples $\{ a _ { i } , x _ { i } \}$ generated via MCMC or other experimental methods. We want to learn a neural operator $v _ { \theta }$ that functions as a conditional velocity field:

$$
\frac { d x } { d t } = v _ { \theta } [ a ] ( x , t ) ,\tag{1.2}
$$

and apply it to unseen $a = \left( b , \sigma \right)$ to generate samples from the steady state distribution of the SDE defined by a. Since the inference procedure requires evaluating $v _ { \theta }$ multiple times to solve the ODE (1.2), we adopt an encoder-decoder design: for a given pair $a = ( b , \sigma )$ , the input functions are encoded once into a latent representation, which is then reused across all ODE integration steps, thereby reducing the computational cost at inference time. We use the Perceiver architecture [13] to aggregate the embeddings of random Lagrangian trajectory sensors into a context vector of fixed dimension.

The main contributions of this work are as follows:

• We formulate an operator-conditioned amortized sampler for parametric families of SDEs, which combines neural operator encoders with continuous normalizing flows. The high cost of training is incurred once, and the trained model then generates approximate invariant measure samples for unseen coeficient instances at an inference cost decoupled from the mixing time of the system.

• An architecture with attention uses random Lagrangian trajectory sensors for problems in high dimensions. The architecture accepts variable sensor counts during training and inference. The random trajectory sensors do not require a mesh and support flexible domains.

• We prove theoretically the expressivity of the operator samplers for compact sets of admissible coeficient functions, and also the resolution invariance properties for the proposed random Lagrangian attention sampler. We first establish that the continuity of the map from the coeficient functions to the flow that transforms a reference measure to the target measure. The resolution invariance of random Lagrangian sensors with attention aggregation is established by uniform law of large numbers for compact attention classes.

The remainder of this paper is organized as follows. Section 2 introduces the mathematical setting and section 3 presents the general framework. Section 4 describes the Lagrangian attention sampler in detail. Section 5 provides the theoretical analysis for the expressivity and section 6 establishes the resolution invariance. Lastly, Section 7 reports the experimental evaluation.

## 2 Problem Setting

We consider the problem of sampling from the invariant measures of a parametric family of SDEs of the form (1.1), where the parameter is a set of functions (e.g., a coeficient field, a boundary condition, or a potential). In this work, the parameter is taken to be the coeficient functions $a = ( b , \sigma )$ . The law $\mu _ { t }$ of $X _ { t }$ satisfies the following Fokker–Planck equation

$$
\partial _ { t } \mu _ { t } = - \nabla \cdot ( b ( x ) \mu _ { t } ) + \frac { 1 } { 2 } \nabla ^ { 2 } : ( \sigma ( x ) \sigma ^ { \top } ( x ) \mu _ { t } ) .\tag{2.1}
$$

An invariant measure of the SDE is a stationary solution of the Fokker–Planck equation (2.1). So the problem can be formulated as learning the map from the parameter $a = \left( b , \sigma \right)$ to the stationary solution of the Fokker-Planck equation (2.1). Under standard dissipativity and regularity conditions on $( b , \sigma ) -$ made precise in Section 5 — the SDE admits a unique invariant probability measure $\mu _ { a } \in \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ , and the map $a \mapsto \mu _ { a }$ is continuous [3].

Since the object to be approximated is a probability distribution, we need a metric on distributions, and we use the quadratic Wasserstein distance throughout. On the space $\mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ of probability measures with finite second moment, it is defined by

$$
W _ { 2 } ( \mu , \nu ) = \binom { \operatorname* { i n f } } { \pi \in \Pi ( \mu , \nu ) } \int _ { \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } } | x - y | ^ { 2 } d \pi ( x , y ) \biggr ) ^ { 1 / 2 } ,
$$

where $\Pi ( \mu , \nu )$ is the set of couplings of $\mu$ and $\nu ,$ i.e. joint distributions on $\mathbb { R } ^ { d } \times \mathbb { R } ^ { d }$ with marginals $\mu$ and ν. The pair $( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) , W _ { 2 } )$ is a complete metric space, and convergence in $W _ { 2 }$ is equivalent to weak convergence together with convergence of second moments [36]. The metric plays a double role in what follows: the approximation guarantees of Section 5 and Section 6 are stated in $W _ { 2 } .$ , and the experiments of Section 7 use it as the quantitative quality criterion for the trained sampler.

It remains to state precisely what “learning a sampler” means for a family of SDEs. Fix a probability distribution π on A representing the family of interest, e.g. a class of drift fields drawn from a prescribed prior. In its classical, one instance form, the sampling problem takes one fixed coeficient pair $a = ( b , \sigma )$ and asks for i.i.d. samples $X ^ { ( 1 ) } , \ldots , X ^ { ( N ) }$ whose empirical law approximates $\mu _ { a }$ . MCMC and sequential Monte Carlo solve the problem in this form, one a at a time, so each new coeficient pair triggers an independent — and, in slow mixing regimes, long — simulation of the dynamics. We consider instead the amortized form of the problem: construct a single map

$$
\mathcal S : \mathcal A \times \mathbb R ^ { d } \to \mathbb R ^ { d } , \qquad ( a , z ) \mapsto \mathcal S ( a , z ) ,
$$

where $z$ is a latent variable drawn from a fixed reference distribution $\gamma$ on $\mathbb { R } ^ { d }$ (typically $\gamma = \mathcal { N } ( 0 , I _ { d } ) )$ , such that the pushforward ${ \hat { \mu } } _ { a } : = S ( a , \cdot ) _ { \# } \gamma$ approximates $\mu _ { a }$ for every a in the support of π. The quality of S is measured by the population error

$$
\mathcal { E } ( S ) ~ = ~ \mathbb { E } _ { a \sim \pi } \big [ W _ { 2 } \big ( \hat { \mu } _ { a } , { \mu } _ { a } \big ) \big ] .\tag{2.2}
$$

The amortized formulation pays an upfront construction cost in exchange for cheap eval uation: once S is built, sampling from $\mu _ { a }$ for a previously unseen a requires only forward evaluations of $s ,$ with no per-instance simulation of the underlying SDE.

## 3 The general framework for neural operator sampler

We now explain our parametric sampler framework promised by Section 2. We formalize the neural operator sampler as follows.

Definition 3.1 (Neural Operator Sampler). Let $\mathbb { R } ^ { d }$ be a state space, and A be a Banach space of input functions. A Neural Operator Sampler is a parameterized mapping

$$
S _ { \theta } : \mathcal { A } \times \mathbb { R } ^ { d }  \mathbb { R } ^ { d } ,\tag{3.1}
$$

where $\theta \in \Theta$ denotes the learnable parameters $( w e i g h t s )$ . For a given input $a = ( b , \sigma ) \in { \mathcal { A } } ,$ the map $S _ { \theta } ( a , \cdot )$ pushes forward the reference measure γ (e.g., standard Gaussian) to a target distribution $\hat { \mu } _ { a } \in \mathcal { P } ( \mathbb { R } ^ { d } )$

To construct the map $\scriptstyle { S _ { \theta } }$ , we adopt the Continuous Normalizing Flow (CNF) paradigm and learn the transport dynamics from reference samples to target samples. We introduce a Neural Operator $\mathcal { N } _ { \theta }$ that approximates the velocity field governing the transport from the reference measure $\gamma$ to $\mu _ { a }$

$$
\mathcal { N } _ { \theta } : \mathcal { A }  C ( [ 0 , 1 ] \times \mathbb { R } ^ { d } ; \mathbb { R } ^ { d } ) .\tag{3.2}
$$

For a specific input $^ { a , }$ the operator predicts a velocity field

$$
v _ { a } ( x , t ) = \mathcal { N } _ { \theta } ( a ) ( x , t ) .\tag{3.3}
$$

The sampler $S _ { \theta } ( a , z )$ is then defined as the solution ϕ(1) of the ODE

$$
\frac { d \phi } { d t } = v _ { a } ( \phi ( t ) , t ) , \quad \phi ( 0 ) = z \sim \gamma .\tag{3.4}
$$

We use the alternative notation $v [ b , \sigma ] ( \cdot , \cdot )$ to indicate the functional dependence on b and σ.

A natural question is whether such a CNF exists for each coeficient pair a, and whether some generating velocity field can serve as a regression target during training. We recall two classical existence mechanisms for such flows: the Benamou–Brenier optimal transport flow, which is exact but depends on the optimal coupling, unavailable from data; and the marginal flow matching construction, which builds a valid transport flow from any coupling of source and target samples and underlies our training objective.

(1) Benamou–Brenier optimal transport flow.

The dynamic (Benamou–Brenier) formulation of optimal transport [2] looks for a density–velocity pair $( \rho _ { t } , v _ { t } )$ that depends on time and solves the constrained optimization problem

$$
W _ { 2 } ^ { 2 } ( \gamma , \mu _ { a } ) = \operatorname* { i n f } _ { ( \rho , v ) } \left\{ \int _ { 0 } ^ { 1 } \int _ { \mathbb { R } ^ { d } } \| v ( x , t ) \| ^ { 2 } \rho ( x , t ) d x d t \ ; \right.\tag{3.5}
$$

The optimal velocity field $v ^ { \mathrm { B B } }$ transports $\gamma$ to $\mu _ { a }$ along the $W _ { 2 } .$ -geodesic with minimal kinetic energy. By Brenier’s theorem $[ 4 ]$ , when $\gamma$ is absolutely continuous there exists a unique W -optimal transport map $T = \nabla \psi$ (with ψ convex) satisfying $T _ { \# } \gamma = \mu _ { a }$ . The Benamou–Brenier geodesic is then realized by the linear interpolation

$$
X _ { t } = ( 1 - t ) X _ { 0 } + t T ( X _ { 0 } ) , \qquad X _ { 0 } \sim \gamma ,
$$

with marginal velocity field

$$
v ^ { \mathrm { B B } } ( x , t ) = \mathbb { E } [ T ( X _ { 0 } ) - X _ { 0 } \mid X _ { t } = x ] .
$$

The optimal transport construction thus provides an exact, geometrically natural flow from $\gamma \ \mathrm { t o } \ \mu _ { a }$ . Its drawback is practical: it requires access to the optimal map or coupling for each target instance, which is generally not available during training — our data consist of unpaired samples from $\gamma$ and $\mu _ { a }$

## (2) Marginal flow matching construction.

A second existence mechanism is the one underlying flow matching, and it removes this requirement: it builds a valid transport flow from any coupling $\eta _ { a } \in \Pi ( \gamma , \mu _ { a } )$ of source and target samples — in particular the independent coupling $\gamma \otimes \mu _ { a }$ , which requires only unpaired samples. The idea is to prescribe simple dynamics conditionally on a pair of endpoints and then average. For each pair $( x _ { 0 } , x _ { 1 } )$ , choose a conditional interpolation path $\rho _ { t } ( \cdot \mathrm { ~ \vert ~ } x _ { 0 } , x _ { 1 } )$ and a conditional velocity $\boldsymbol { u } _ { t } ( \cdot \mathbf { \nu } | \mathbf { \nu } _ { x _ { 0 } , x _ { 1 } } )$ satisfying the conditional continuity equation

$$
\partial _ { t } \rho _ { t } ( x \mid x _ { 0 } , x _ { 1 } ) + \nabla \cdot ( \rho _ { t } ( x \mid x _ { 0 } , x _ { 1 } ) u _ { t } ( x \mid x _ { 0 } , x _ { 1 } ) ) = 0 ,
$$

with endpoints concentrated at $x _ { 0 }$ and $x _ { 1 }$ . Averaging over the coupling gives a marginal path and a marginal velocity:

$$
\rho _ { t } ^ { a } ( x ) = \int \rho _ { t } ( x \mid x _ { 0 } , x _ { 1 } ) d \eta _ { a } ( x _ { 0 } , x _ { 1 } ) ,
$$

$$
\rho _ { t } ^ { a } ( x ) u _ { t } ^ { a } ( x ) = \int \rho _ { t } ( x \mid x _ { 0 } , x _ { 1 } ) u _ { t } ( x \mid x _ { 0 } , x _ { 1 } ) d \eta _ { a } ( x _ { 0 } , x _ { 1 } ) ,
$$

where $u _ { t } ^ { a }$ is defined arbitrarily on the set where $\rho _ { t } ^ { a } = 0$ . Integrating the conditional continuity equations yields

$$
\partial _ { t } \rho _ { t } ^ { a } + \nabla \cdot ( \rho _ { t } ^ { a } u _ { t } ^ { a } ) = 0 , \qquad \rho _ { 0 } ^ { a } = \gamma , \quad \rho _ { 1 } ^ { a } = \mu _ { a } .
$$

Equivalently, for $X _ { t } \sim \rho _ { t } ^ { a }$

$$
u _ { t } ^ { a } ( x ) = \mathbb { E } [ u _ { t } ( X _ { t } \mid X _ { 0 } , X _ { 1 } ) \mid X _ { t } = x ] ,
$$

which is the pointwise $L ^ { 2 }$ projection of the conditional velocity onto functions of the current state. This field has a dynamical role: it generates the marginal path and defines a valid transport flow from $\gamma$ to $\mu _ { a }$ . The conditional expectation form is also what makes the construction trainable: it characterizes $\boldsymbol { u } _ { t } ^ { a }$ as the minimizer of a regression problem whose samples require only draws from the coupling and the conditional path, which turns into the training objective (see the details below).

Flow matching trains the velocity field by pointwise regression and avoids ODE solves inside each gradient step, giving it a lower online training cost than maximum likelihood CNF training. The price is that it requires training samples from the target invariant measures; in this work we focus on this supervised setting. Stochastic interpolants [1] provide a closely related formulation, and the expressivity theory of Section 5 uses a smoothed variant of the flow matching construction.

For a fixed coeficient pair $^ { a , }$ the ideal marginal flow matching loss is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { F M } } ^ { a } ( \theta ) = \mathbb { E } _ { t \sim \mathcal { U } [ 0 , 1 ] } \mathbb { E } _ { x \sim \rho _ { t } ^ { a } } \big \| v _ { \theta } [ a ] ( x , t ) - u _ { t } ^ { a } ( x ) \big \| ^ { 2 } . } \end{array}\tag{3.6}
$$

The marginal path $\rho _ { t } ^ { a }$ and velocity u<sup>a</sup> are generally intractable. Conditional flow matching [20] replaces $\left( 3 . 6 \right)$ with a tractable conditional regression problem. The common $x _ { 1 ^ { - } }$ conditioned Gaussian path corresponds to the independent source–target coupling: for each target sample $x _ { 1 } \sim \mu _ { a } ,$ , one defines a conditional probability path $\rho _ { t } ( \cdot \mid x _ { 1 } )$ and a conditional velocity field $u _ { t } ( \cdot \mid x _ { 1 } )$ satisfying

$$
\partial _ { t } \rho _ { t } ( x \mid x _ { 1 } ) + \nabla \cdot \left( \rho _ { t } ( x \mid x _ { 1 } ) u _ { t } ( x \mid x _ { 1 } ) \right) = 0 , \qquad \rho _ { 0 } ( \cdot \mid x _ { 1 } ) = \gamma , \quad \rho _ { 1 } ( \cdot \mid x _ { 1 } ) \approx \delta _ { x _ { 1 } } .
$$

The resulting conditional flow matching objective

$$
{ \mathcal { L } } _ { \mathrm { C F M } } ( \theta ) = \mathbb { E } _ { a \sim \pi } \mathbb { E } _ { t \sim \mathcal { U } [ 0 , 1 ] } \mathbb { E } _ { x _ { 1 } \sim \mu _ { a } } \mathbb { E } _ { x \sim \rho _ { t } ( \cdot \vert x _ { 1 } ) } { \big \| } v _ { \theta } [ a ] ( x , t ) - u _ { t } ( x \mid x _ { 1 } ) { \big \| } ^ { 2 }\tag{3.7}
$$

has the same gradients as the corresponding marginal loss for this independent coupling path (3.6) [20] and requires only samples from the simple conditional path.

For the independent coupling, this paper uses the standard Gaussian linear conditional path

$$
\rho _ { t } ( x \mid x _ { 1 } ) = { \mathcal { N } } { \big ( } x \mid t x _ { 1 } , ( 1 - t ) ^ { 2 } I { \big ) } , \qquad u _ { t } ( x \mid x _ { 1 } ) = { \frac { x _ { 1 } - x } { 1 - t } } .\tag{3.8}
$$

Drawing $X _ { 0 } \sim \gamma = { \mathcal { N } } ( 0 , I )$ independently of $X _ { 1 }$ and setting $X _ { t } = ( 1 - t ) X _ { 0 } + t X _ { 1 }$ gives the equivalent velocity target $u _ { t } ( X _ { t } \mid X _ { 1 } ) = X _ { 1 } - X _ { 0 }$

More generally, if a dependent source–target coupling $\eta _ { a } \in \Pi ( \gamma , \mu _ { a } )$ is used to form training pairs, the conditional path should be viewed as conditioned on the pair $( X _ { 0 } , X _ { 1 } )$

$$
X _ { t } = ( 1 - t ) X _ { 0 } + t X _ { 1 } , \qquad u _ { t } ( X _ { t } \mid X _ { 0 } , X _ { 1 } ) = X _ { 1 } - X _ { 0 } .
$$

The corresponding supervised training loss is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { C F M } } ^ { \mathrm { l i n } } ( \theta ) = \mathbb { E } _ { a \sim \pi } \mathbb { E } _ { ( X _ { 0 } , X _ { 1 } ) \sim \eta _ { a } } \mathbb { E } _ { t \sim \mathcal { U } [ 0 , 1 ] } \left[ \left\| v _ { \theta } [ a ] ( X _ { t } , t ) - ( X _ { 1 } - X _ { 0 } ) \right\| ^ { 2 } \right] . } \end{array}\tag{3.9}
$$

The default independent pairing corresponds to $\eta _ { a } = \gamma \otimes \mu _ { a } ;$ one may use an empirical or minibatch optimal transport coupling when one wants a closer approximation to the Benamou–Brenier geometry. With an optimal source–target coupling, the linear construction conditioned on pairs realizes the Benamou–Brenier geodesic (3.5); with independent source and target samples, it is a tractable flow matching interpolant. In either case, diferent couplings and interpolation paths lead to diferent valid training targets.

In order to mitigate the cost of inference the trained model multiple times to solve the flow ODE (3.4), we propose an encoder–decoder architecture to construct the neural operator $v _ { \theta }$ . The encoder maps the input functions $( b , \sigma )$ to a latent representation, and the decoder takes the latent code along with spatiotemporal coordinates $( x , t )$ to produce the velocity field. Diferent encoders and decoders can be used to fit diferent problems. Specifically, the calculation of $v _ { \theta } [ b , \sigma ] ( x , t )$ may be decomposed into two steps:

$$
\begin{array} { r } { z = \mathrm { E n c o d e r } ( b , \sigma ; \theta _ { e } ) , } \end{array}\tag{3.10}
$$

$$
v _ { \theta } [ b , \sigma ] ( x , t ) = \operatorname { D e c o d e r } ( z , x , t ; \theta _ { d } ) ,\tag{3.11}
$$

where $\theta = \left( \theta _ { e } , \theta _ { d } \right)$ are the learnable parameters of encoder and decoder respectively. In the subsequent sections, we will discuss several possible representations of the input functions $( b , \sigma )$ and the corresponding encoder and decoders.

Training uses the conditional flow matching loss in (3.9). At inference, we evaluate the model multiple times to solve the ODE (1.2). The latent representation (3.10) is computed once per coeficient pair $a = ( b , \sigma )$ , which reduces time and memory cost. The complete training and inference procedures are summarized in Algorithm 1.

Algorithm 1 Amortized Neural Sampler using flow matching loss: Training and Inference   
1: Input: Training set $\mathcal { D } = \{ ( a _ { i } , \{ x _ { i } ^ { ( j ) } \} _ { j = 1 } ^ { N _ { s } } ) \} _ { i = 1 } ^ { N _ { f } }$ with $a _ { i } = ( b _ { i } , \sigma _ { i } )$ ; reference measure γ;   
ODE solver with K steps.   
2: Output: Trained parameters $\theta = \left( \theta _ { e } , \theta _ { d } \right)$   
3: % Training   
4: for each minibatch do   
5: Sample coeficients $a \sim \pi ,$ target $X _ { 1 } \sim \mu _ { a } ,$ noise $X _ { 0 } \sim \gamma .$ , time $t \sim U [ 0 , 1 ] .$   
6: Compute interpolant $X _ { t } = ( 1 - t ) X _ { 0 } + t X _ { 1 }$   
7: Encode: $z \gets$ Encoder(a; $\theta _ { e } )$   
8: Predict velocity: $\hat { v } \gets .$ Decoder $\left( z , X _ { t } , t ; \theta _ { d } \right)$   
9: Update θ by $\begin{array} { r } { \dot { \nabla } _ { \theta } \| \hat { v } - ( X _ { 1 } - X _ { 0 } ) \| ^ { 2 } . } \end{array}$   
10: end for   
11: % Inference (for a new, unseen $a ^ { * } )$   
12: Encode once: $z ^ { * } \gets$ Encode $\cdot ( a ^ { * } ; \theta _ { e } )$   
13: Draw $X _ { 0 } \sim \gamma .$   
14: for $k = 1 , \ldots , K$ do   
15: ODE integration (e.g. Euler, RK45): $X _ { t _ { k } } \gets X _ { t _ { k - 1 } } { + \Delta t _ { k } }$ Decode $\cdot ( z ^ { \ast } , X _ { t _ { k - 1 } } , t _ { k - 1 } ; \theta _ { d } )$   
16: end for   
17: Return $X _ { 1 }$ as a sample from the learned invariant measure $\hat { \mu } _ { a ^ { * } }$

## 4 A Lagrangian attention operator sampler

The classical grid approach implements the encoder–decoder framework with a Multi-input DeepONet [14]. At m grid sensors, we record $b _ { j } = b ( x _ { j } ) \in \mathbb R ^ { d }$ and $\sigma _ { j } = \sigma ( x _ { j } ) \in \mathbb { R } ^ { d \times d _ { W } }$ We retain $p \leq d$ rows of $\sigma ,$ omitting only rows that vanish for every coeficient instance and position. Stacking each retained row across sensors gives the input tensors:

$$
\hat { b } \in \mathbb R ^ { m \times d } , \quad \hat { \sigma } _ { k } \in \mathbb R ^ { m \times d _ { W } } , \quad k = 1 , \dots , p .
$$

The velocity field is assembled via the tensor product structure of the Multi-input Deep-ONet [14]:

$$
v _ { \theta } [ b , \sigma ] ( x , t ) = \mathcal { G } ( \hat { b } , \hat { \sigma } _ { k } , x , t ; \theta ) = \sum _ { l = 1 } ^ { q } f _ { l } ^ { 0 } ( \hat { b } ) \odot f _ { l } ^ { 1 } ( \hat { \sigma } _ { 1 } ) \odot \cdots \odot f _ { l } ^ { p } ( \hat { \sigma } _ { p } ) \odot u _ { l } ( \hat { x } , \hat { t } ) ,\tag{4.1}
$$

where $\odot$ denotes the elementwise (Hadamard) product, and $\hat { x } , \hat { t }$ are $x , t$ embedded via a fixed sinusoidal map. In the notation of Section 3, the Branch nets collectively form the encoder

$$
z = ( f ^ { 0 } ( \hat { b } ) , f ^ { 1 } ( \hat { \sigma } _ { 1 } ) , \dots , f ^ { p } ( \hat { \sigma } _ { p } ) ) ,
$$

and the Trunk net together with the product-sum formula form the decoder.

For problems in high dimensions a grid of sensors is infeasible: the number of grid points grows exponentially in the dimension $d ,$ so the dense Eulerian sampling of $b ( x ) , \sigma ( x )$ used by classical neural operators [15, 18] exceeds a realistic budget. Motivated by VIDON for random point sensors [29], we aggregate information from random Lagrangian sensors— mobile probe trajectories that follow a cheap proxy of the dynamics—and encode them with attention. See also [9] for related ideas. Because the encoder reads the sensors through their empirical measure, it ingests a variable number of sensors m and runs at a sensor count $m ^ { \prime } \neq m$ without retraining. As m grows, the output law of the model with fixed weights converges—in expectation over the random probes—to its own population-probe limit (Section 6); increasing m reduces the Monte Carlo error from the finite probe set relative to that limit.

The full architecture is shown in Figure 4.1. The model embeds m sensor trajectories into the token list E; E then interacts repeatedly with a learnable latent token list $Z _ { 0 }$ of fixed size to obtain the latent context $Z _ { L } . \ Z _ { L }$ , together with $( X , t )$ after passing through a Gaussian Fourier Projection, goes through an AdaLN Decoder and produces the final velocity field $v _ { \theta } [ a ] ( X , t )$ . The following subsections describe the exact operations.

![](images/e5cf24ba37ffe6c11685e3e13322b67981e07969c92d96ebbf21d2abe4eb76b4.jpg)  
Figure 4.1: Architecture of the amortized neural sampler with an attention encoder.

## 4.1 Lagrangian trajectory sensors and the per-sensor embedding

We draw m random initial positions $\{ x _ { i } \} _ { i = 1 } ^ { m } .$ i.i.d. from a fixed probability distribution on $\mathbb { R } ^ { d }$ (the probe initial law); the count m may difer across data samples. From each $x _ { i }$ we simulate a short probe trajectory $x _ { i , 0 } , \ldots , x _ { i , L }$ and record at every step the local feature

$$
y _ { i , j } = \left( x _ { i , j } , \ b ( x _ { i , j } ) , \mathrm { \ v e c } \sigma ( x _ { i , j } ) \right) \in \mathbb { R } ^ { 2 d + d d _ { W } } , \qquad j = 0 , \ldots , L ,
$$

which pairs each visited state with the drift and difusion sampled there. The i-th probe observation is the ordered collection

$$
Y _ { i } = \left( y _ { i , 0 } , \ldots , y _ { i , L } \right) \in \mathbb { R } ^ { \left( L + 1 \right) \times c _ { 0 } } , \qquad c _ { 0 } = 2 d + d d _ { W } .
$$

Known structure in σ reduces the feature width. For $d _ { W } = d$ and isotropic difusion $\sigma =$ $\sigma _ { 0 } I _ { d }$ , we use $y _ { i , j } = ( x _ { i , j } , b ( x _ { i , j } ) , \sigma _ { 0 } ) \in \mathbb { R } ^ { 2 d + 1 }$

Sensing with trajectories enriches pointwise evaluation by recording local changes of the drift and difusion along a short path; even short trajectories provide finite diference information about the local vector field, and their ordered structure can be used by a convolutional encoder.

The probe dynamics need not coincide with the target dynamics: any process that visits the domain and permits pointwise evaluation of b and σ can drive the probes. In practice we drive them either by a cheap exploratory process—Brownian motion with a fixed step size, or a constant difusion proxy $d X = b ( X ) d t + \bar { \sigma } d W \bar { \quad } \mathrm { o r }$ by the original SDE (1.1) itself, a natural choice when the drift steers probes toward dynamically relevant regions. In either case the sensing cost is decoupled from the dificulty of the target SDE, and the probes can be extremely short. In several of our experiments (Section 7) trajectories of only $L = 2 0$ steps—and in some settings as few as $L \ = \ 1$ (two points)—already sufice for accurate reconstruction of the invariant measure. Because each probe involves only L evaluations of b and $\sigma ,$ the total sensing cost is negligible compared to the ${ \mathcal { O } } ( { \tau _ { \operatorname* { m i x } } } / { \Delta t } )$ steps required by MCMC.

After the Lagrangian probes obtained, we further make use of a small 1D convolutional neural network acting along the step axis to map each observation into a vector of fixed width. Writing $C ^ { ( 0 ) } = Y _ { i }$ , the network applies N convolutional blocks

$$
C ^ { ( \ell ) } = \mathrm { G E L U } \Big ( \mathrm { L a y e r N o r m } \big ( \mathrm { C o n v l d } _ { k = 3 } ^ { ( \ell ) } ( C ^ { ( \ell - 1 ) } ) \big ) \Big ) \in \mathbb { R } ^ { ( L + 1 ) \times \varepsilon _ { \ell } } , \qquad \ell = 1 , \ldots , N ,
$$

followed by global average pooling over the $L + 1$ steps,

$$
e _ { i } = \frac { 1 } { L + 1 } \sum _ { j = 0 } ^ { L } C _ { j , : } ^ { ( N ) } \in \mathbb { R } ^ { d _ { e } } , \qquad d _ { e } : = c _ { N } .
$$

Here GELU donates the Gaussian Error Linear Unit activation

$$
\mathrm { G E L U } ( x ) = x \Phi ( x ) , \quad \Phi ( x ) = { \frac { 1 } { \sqrt { 2 \pi } } } \int _ { - \infty } ^ { x } \exp \left( - { \frac { | y | ^ { 2 } } { 2 } } \right) d y .
$$

The convolutions with kernel size 3 use “same” padding, which preserves the step length $L + 1$ through all N blocks and keeps the embedding defined down to $L = 1 ;$ ; they capture local temporal patterns such as the curvature and magnitude of the drift, and the pooling produces an output $e _ { i }$ of fixed size for any trajectory length $L ,$ so the embedding dimension $d _ { e }$ is independent of L and $m .$ . (A small MLP on a pooled trajectory is an alternative; the 1D CNN consistently outperformed it in our experiments by exploiting the sequential structure.)

The encoder downstream sees the sensors only through their empirical measure

$$
\hat { \nu } _ { m } = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \delta _ { e _ { i } } , \qquad e _ { i } \in \mathbb { R } ^ { d _ { e } } .
$$

This is the object the resolution invariance theory acts on: increasing m sharpens $\hat { \nu } _ { m }$ toward an underlying sensor law $\nu ,$ and the encoder applies the same map to $\hat { \nu } _ { m }$ and to ν.

## 4.2 Attention Perceiver encoder

The encoder aggregates the sensor embeddings into a context vector of fixed dimension.   
This aggregation accepts variable sensor counts and is invariant to sensor permutations.

To perform this aggregation, the encoder carries a fixed set of k latent tokens (typically $k \ll m )$ , stacked as $\boldsymbol { Z } \in \mathbb { R } ^ { k \times d _ { z } }$ of latent width $d _ { z }$ . Attention updates these tokens with sensor information, and the final readout converts them into the context vector. The encoder represents the m sensors by projecting each embedding $e _ { i } \in \mathbb { R } ^ { d _ { e } }$ to width $d _ { z }$ with a shared linear map $P _ { e } \in \mathbb { R } ^ { d _ { e } \times d _ { z } }$ , stacking the rows $E _ { i } = e _ { i } P _ { e }$ into $\boldsymbol { E } \in \mathbb { R } ^ { m \times d _ { z } }$

Two attention operations act on these arrays: cross attention, in which the latents read the sensors, and self attention, in which the latents exchange information among themselves. Each operation uses H heads of width $d _ { h } = d _ { z } / H$

Both operations are built from the same attention primitive. A single head maps a query array $Q ,$ key array $K ,$ and value array $V$ (rows of width $d _ { h } )$ to

$$
\mathrm { A t t e n } ( Q , K , V ) = \mathrm { s o f t m a x } \Big ( \frac { Q K ^ { \top } } { \sqrt { d _ { h } } } \Big ) V , \qquad \Big [ \mathrm { s o f t m a x } \Big ( \frac { Q K ^ { \top } } { \sqrt { d _ { h } } } \Big ) \Big ] _ { r _ { j } } = \frac { \exp ( \langle q _ { r } , k _ { j } \rangle / \sqrt { d _ { h } } ) } { \sum _ { i } \exp ( \langle q _ { r } , k _ { i } \rangle / \sqrt { d _ { h } } ) } ,
$$

where $q _ { r } , k _ { j }$ are the rows of Q, K and the sum runs over the rows of $K$ . The softmax normalizes each row, so row r of Atten $( Q , K , V )$ is the convex combination of the rows of V weighted by query r, and the factor $\sqrt { d _ { h } }$ holds the scores at a stable scale as $d _ { h }$ grows.

Each of the H heads carries learnable projections $W _ { h } ^ { Q } , W _ { h } ^ { K } , W _ { h } ^ { V } \in \mathbb { R } ^ { d _ { z } \times d _ { h } }$ , and the H head outputs are concatenated and mixed by $W ^ { O } \in \mathbb { R } ^ { d _ { z } \times d _ { z } }$ . Cross attention draws queries from the latents, the keys and values from the sensors; self attention draws all three from the latents:

$$
\begin{array} { r l r } & { } & { \mathrm { C A } ( Z , E ) = \left[ \mathrm { A t t e n } ( Z W _ { h } ^ { Q } , E W _ { h } ^ { K } , E W _ { h } ^ { V } ) \right] _ { h = 1 } ^ { H } W ^ { O } , } \\ & { } & { \mathrm { S A } ( Z ) = \left[ \mathrm { A t t e n } ( Z W _ { h } ^ { Q } , Z W _ { h } ^ { K } , Z W _ { h } ^ { V } ) \right] _ { h = 1 } ^ { H } W ^ { O } , } \end{array}
$$

both in $\mathbb { R } ^ { k \times d _ { z } }$ , where $[ \cdot ] _ { h = 1 } ^ { H }$ concatenates the head outputs along the feature axis and crossand self attention carry separate projections. Cross attention carries the information of the m sensors into the k latents; self attention lets the latents share information.

Starting from the learned initial latents $Z ^ { ( 0 ) } \in \mathbb { R } ^ { k \times d _ { z } }$ , each of the $L _ { \mathrm { e n c } }$ blocks reads the sensors into the latents through cross attention, lets the latents exchange information through self attention, and refines them tokenwise through an MLP with two layers of width $4 d _ { z }$ and GELU activation, each step carried by a residual connection and a post-LayerNorm:

$$
\begin{array} { r l r } & { Z ^ { \prime } = \mathrm { L N } \big ( Z ^ { ( \ell ) } + \mathrm { C A } ( Z ^ { ( \ell ) } , E ) \big ) , } & \\ & { Z ^ { \prime \prime } = \mathrm { L N } \big ( Z ^ { \prime } + \mathrm { S A } ( Z ^ { \prime } ) \big ) , } & { \quad \ell = 0 , \dots , L _ { \mathrm { e n c } } - 1 . } \\ & { Z ^ { ( \ell + 1 ) } = \mathrm { L N } \big ( Z ^ { \prime \prime } + \mathrm { M L P } ( Z ^ { \prime \prime } ) \big ) , } \end{array}
$$

Every block reuses the same sensor array $E$ as keys and values and updates the latent tokens. Each cross attention head returns a ratio of two sums over the sensors—the value average in the numerator and the softmax normalization in the denominator—so its value is a function of the empirical measure $\hat { \nu } _ { m }$ alone. Section 6 and Appendix $\mathrm { A . 4 }$ build on this empirical measure structure, replacing $\hat { \nu } _ { m }$ by a general sensor law ν to establish resolution invariance as m grows.

After the $L _ { \mathrm { e n c } }$ blocks, the encoder reads out the k latent tokens by flattening them and projecting linearly to the context vector

$$
\begin{array} { r } { z = \mathrm { E n c o d e r } ( \hat { \nu } _ { m } ) : = W ^ { \mathrm { o u t } } \mathrm { v e c } \big ( Z ^ { ( L _ { \mathrm { e n c } } ) } \big ) \in \mathbb { R } ^ { d _ { z } } , \qquad W ^ { \mathrm { o u t } } \in \mathbb { R } ^ { d _ { z } \times k d _ { z } } } \end{array}
$$

(for $k = 1$ this is just the single token). The whole construction is permutation invariant in the ordering of the sensor embeddings [38]. In our experiments we use $k = 4$ latent tokens, $H = 4$ heads of width $d _ { h } = 3 2$ (so $d _ { z } = 1 2 8 )$ , and $L _ { \mathrm { e n c } } = 2$ blocks, yielding $z \in \mathbb { R } ^ { 1 2 8 }$

## 4.3 AdaLN-conditioned flow decoder

The decoder receives the context $z \in \mathbb { R } ^ { d _ { z } }$ , a state $x \in \mathbb { R } ^ { d }$ , and the flow time $t \in [ 0 , 1 ]$ , and outputs the velocity $v _ { \theta } [ b , \sigma ] ( x , t ) \in \mathbb { R } ^ { d }$

The flow time t enters through a fixed sinusoidal embedding $\tau = \mathrm { S i n E m b } ( t ) \in \mathbb { R } ^ { d _ { t } }$ , whose j-th entry is

$$
\tau _ { 2 j - 1 } = \sin ( 2 \pi f _ { j } t ) , \qquad \tau _ { 2 j } = \cos ( 2 \pi f _ { j } t ) , \qquad f _ { j } = f _ { \mathrm { m i n } } \left( \frac { f _ { \mathrm { m a x } } } { f _ { \mathrm { m i n } } } \right) ^ { ( j - 1 ) / ( d _ { k } / 2 - 1 ) } ,
$$

with geometrically spaced frequencies $f _ { 1 } = f _ { \mathrm { m i n } } \mathrm { ~ t o ~ } f _ { d _ { t } / 2 } = f _ { \mathrm { m a x } }$ and $j = 1 , \ldots , d _ { t } / 2$ . The context that modulates every block fuses the operator latent z with this time embedding through a learned linear projection,

$$
c = W _ { c } \left[ z ; \tau \right] + b _ { c } \in \mathbb { R } ^ { d _ { z } } ,
$$

and the state and time embedding are concatenated and lifted to the hidden width $D _ { : }$

$$
h ^ { ( 0 ) } = W _ { \mathrm { i n } } \left[ x ; \tau \right] + b _ { \mathrm { i n } } \in \mathbb { R } ^ { D } .
$$

We condition the decoder through adaptive layer normalization (AdaLN) [27], letting the context c set the scale and shift of every block’s normalization so that the conditioning signal reaches all depths. Each of the $N _ { d }$ blocks modulates its two normalization layers with scale and shift predicted from $c ,$

$$
{ \mathrm { A d a L N } } ( h ; c ) = \left( 1 + \gamma ( c ) \right) \odot { \mathrm { L N } } ( h ) + \beta ( c ) ,
$$

and applies two such modulations with a residual connection,

$$
\begin{array} { r l } & { h ^ { \prime } = \mathrm { a c t } \big ( \mathrm { A d a L N } ( h ^ { ( \ell ) } ; c ) \big ) , } \\ & { h ^ { \prime \prime } = \mathrm { a c t } \big ( \mathrm { A d a L N } ( W _ { 1 } h ^ { \prime } ; c ) \big ) , \qquad \ell = 0 , \dots , N _ { d } - 1 , } \\ & { h ^ { ( \ell + 1 ) } = h ^ { ( \ell ) } + \alpha ( c ) \odot W _ { 2 } h ^ { \prime \prime } , } \end{array}
$$

where $\gamma , \beta ,$ α are linear functions of c and act is Mish.

Finally, after $N _ { d }$ blocks the hidden state is normalized and projected to the velocity,

$$
v _ { \theta } [ b , \sigma ] ( x , t ) = W _ { v } \operatorname { L N } ( h ^ { ( N _ { d } ) } ) \in \mathbb { R } ^ { d } .
$$

The output weight $W _ { v }$ is initialized to zero so that the velocity field starts near zero at the beginning of training. For interacting particle systems with state $X = ( x _ { 1 } , \dots , x _ { N } ) \in$ $( \mathbb { R } ^ { d _ { p } } ) ^ { \grave { N } }$ , we use the particle variant described in Section 7.3: the decoder reshapes X into N particle tokens, embeds each token with shared weights, applies AdaLN-conditioned self attention and feedforward blocks across the particle tokens, and projects each token to its velocity $v _ { i }$ . Because the token maps are shared and self attention is permutation equivariant, relabeling the particles relabels the output velocities in the same way.

## 5 Expressivity of the Lagrangian attention sampler

This section is devoted to the expressivity result. The sampler is intended to learn a map from SDE coeficients to invariant measures. The theoretical question is whether our architecture has suficient expressive power to approximate this map uniformly over a compact family of coeficient functions. The reduction theorem establishes this result under the stated assumptions via neural operator approximation of a deterministic coeficient-dependent velocity field on compact domains. Appendix A contains the proofs and construction details. The main text records the assumptions, the key lemmas and propositions, the theorem statements, and the proof ideas needed to interpret them.

For $a = \left( b , \sigma \right)$ with $\sigma : \mathbb { R } ^ { d }  \mathbb { R } ^ { d \times d _ { W } }$ , write

$$
A _ { a } ( \boldsymbol { x } ) = \sigma ( \boldsymbol { x } ) \sigma ( \boldsymbol { x } ) ^ { \top } \in \mathbb { R } ^ { d \times d }
$$

for the difusion matrix. The generator of the SDE acts on smooth test functions by

$$
{ \mathcal L } _ { a } \varphi ( x ) = b ( x ) \cdot \nabla \varphi ( x ) + \frac { 1 } { 2 } \operatorname { T r } \bigl ( A _ { a } ( x ) D ^ { 2 } \varphi ( x ) \bigr ) .\tag{5.1}
$$

The generator depends on σ only through $A _ { a }$ . As a result, the invariant measure $\mu _ { a }$ also depends on $\sigma$ only through $A _ { a }$ . We keep the pair $a = ( b , \sigma )$ as the variable because the sampler receives b and σ as its input.

The coeficient space is

$$
\mathcal { X } _ { a } = C ( \mathbb { R } ^ { d } ; \mathbb { R } ^ { d } ) \times C ( \mathbb { R } ^ { d } ; \mathbb { R } ^ { d \times d _ { W } } ) .\tag{5.2}
$$

We equip this space with the compact-open topology, that is, uniform convergence on compact subsets. A sequence $a _ { n } = ( b _ { n } , \sigma _ { n } )$ converges to $a = ( b , \sigma )$ if, on every compact set $B \subset  { \mathbb { R } } ^ { d }$

$$
\operatorname* { s u p } _ { x \in B } | b _ { n } ( x ) - b ( x ) | + \operatorname* { s u p } _ { x \in B } \| \sigma _ { n } ( x ) - \sigma ( x ) \| \longrightarrow 0 .
$$

Choose any compact exhaustion $( B _ { j } ) _ { j \geq 1 }$ of $\mathbb { R } ^ { d }$ , and define

$$
p _ { j } ( a , a ^ { \prime } ) = \operatorname* { s u p } _ { x \in B _ { j } } | b ( x ) - b ^ { \prime } ( x ) | + \operatorname* { s u p } _ { x \in B _ { j } } \| \sigma ( x ) - \sigma ^ { \prime } ( x ) \| .
$$

The metric

$$
d _ { \mathcal { X } _ { a } } ( a , a ^ { \prime } ) = \sum _ { j = 1 } ^ { \infty } 2 ^ { - j } \operatorname* { m i n } \{ 1 , p _ { j } ( a , a ^ { \prime } ) \}
$$

metrizes the compact-open topology on $\mathcal { X } _ { a }$ . If $a _ { n } \to a$ in this topology, then $A _ { a _ { n } }  A _ { a }$ uniformly on every compact set, because $\sigma _ { n }$ is uniformly bounded on that set.

In this topology, compactness of a set $E \subset \mathcal X _ { a }$ means that every sequence in E has a subsequence converging uniformly on every compact subset of $\mathbb { R } ^ { d }$ . The Arzela–Ascoli criterion gives the following standard compactness condition.

Proposition 5.1 (Compactness criterion). A closed set $E \subset \mathcal X _ { a }$ is compact if, for each ball $B _ { m } = \{ x : | x | \leq m \}$ , the restrictions of its elements to $B _ { m }$ are uniformly bounded and equicontinuous.

## 5.1 Invariant measure and the properties

We will focus on the following set of coeficients with parameters $\nu , p _ { 0 } , \lambda , M$ . Let $\mathcal { A } ( \nu , p _ { 0 } , \lambda , M )$ be the class of coeficients $a = ( b , \sigma ) \in \mathcal { X } _ { a }$ such that:

1. Uniform ellipticity (which requires $d _ { W } \geq d )$

$$
A _ { a } ( x ) \subseteq \nu I , \qquad x \in \mathbb { R } ^ { d } .
$$

2. Khasminskii condition for polynomial Lyapunov functions. There exists $p _ { 0 } >$ 2 such that

$$
\langle b ( x ) , x \rangle + { \frac { 1 } { 2 } } ( p _ { 0 } - 1 ) \operatorname { T r } A _ { a } ( x ) \leq - \lambda | x | ^ { 2 } + M , \qquad x \in \mathbb { R } ^ { d } .
$$

In particular, for $V ( x ) = | x | ^ { p } , p \leq p _ { 0 }$ , this condition yields

$$
\begin{array} { r } { \mathcal { L } _ { a } V ( x ) \le - \lambda p V ( x ) + M p } \end{array}
$$

up to lower order constants absorbed into M.

The following is a standard result of the preceding conditions.

Proposition 5.2 (Stationary measure and moments). If $a \in \mathcal { A } ( \nu , p _ { 0 } , \lambda , M )$ , then the stationary Fokker–Planck equation $\mathcal { L } _ { a } ^ { * } \mu = 0$ has a unique probability solution $\mu _ { a }$ . Moreover, for every $p \leq p _ { 0 }$ , there is a constant $M _ { p } < \infty$ , depending only on $( \nu , p _ { 0 } , \lambda , M , p )$ , such that

$$
\operatorname* { s u p } _ { a \in A ( \nu , p _ { 0 } , \lambda , M ) } \int | x | ^ { p } \mu _ { a } ( d x ) \leq M _ { p } , \qquad 2 \leq p \leq p _ { 0 } .
$$

Classical suficient conditions for existence and uniqueness of stationary solutions to the Fokker–Planck equation are given in [3, Theorems 2.4.1 and 4.1.6]. The moment estimate follows from the standard Lyapunov cutof argument; the details are given in Appendix A.

Remark 5.1. Local Lipschitz conditions on $a = ( b , \sigma )$ are often imposed for strong wellposedness of the SDE. The present theory is stated at the level of laws and stationary Fokker– Planck equations, so one may instead work with weak solutions and the associated martingale problem. In this setting, it does not require the conditions on the derivatives of a. See also the Stroock-Varadhan uniqueness theorem [33, Theorem 24.1, Chapter V].

Lemma 5.1 (Closedness and stability of admissible coeficients). The set $\mathcal { A } ( \nu , p _ { 0 } , \lambda , M )$ is closed under the compact-open topology. That is, if $a _ { n } \in \mathcal { A } ( \nu , p _ { 0 } , \lambda , M )$ and $a _ { n } \to a$ locally uniformly, then $a \in \mathcal { A } ( \nu , p _ { 0 } , \lambda , M )$ . Moreover,

$$
W _ { 2 } ( \mu _ { a _ { n } } , \mu _ { a } )  0 .
$$

The proof of Lemma 5.1 will also be presented in Appendix A.

## 5.2 An explicit construction of approximating velocity field

We will basically use the smoothed flow matching velocity field to construct an approximating velocity field. First let $P _ { S } : \mathbb { R } ^ { d }  \overline { { B } } _ { S }$ be the radial projection and define

$$
\mu _ { a } ^ { S } = ( P _ { S } ) _ { \# } \mu _ { a } .
$$

The polynomial moment bound gives the projection error

$$
\operatorname* { s u p } _ { a \in A ( \nu , p _ { 0 } , \lambda , M ) } W _ { 2 } ( \mu _ { a } ^ { S } , \mu _ { a } ) \leq M _ { p _ { 0 } } ^ { 1 / 2 } S ^ { - ( p _ { 0 } - 2 ) / 2 } .
$$

Moreover, $a \mapsto \mu _ { a } ^ { S }$ is continuous.

Next, take $\alpha \in ( 0 , 1 )$ and the narrower Gaussian

$$
\gamma _ { \alpha } = { \mathcal { N } } ( 0 , ( 1 - \alpha ) I _ { d } ) .
$$

Fix the target measure $\mu _ { a } ^ { S }$ with $a \in \mathcal { A } ( \nu , p _ { 0 } , \lambda , M )$ . Draw

$$
X \sim \gamma _ { \alpha } , \qquad Y \sim \mu _ { a } ^ { S } , \qquad \xi \sim \mathcal { N } ( 0 , I _ { d } ) ,
$$

independently. Define

$$
M _ { t } ^ { a } = ( 1 - t ) X + t Y , \qquad Z _ { t } ^ { a } = M _ { t } ^ { a } + \sqrt { \alpha } \xi .
$$

Clearly, letting $\rho _ { t } ^ { a } = \operatorname { L a w } ( Z _ { t } ^ { a } )$ , then

$$
\rho _ { 0 } ^ { a } = \gamma , \qquad \rho _ { 1 } ^ { a } = \mu _ { a } ^ { S } \ast \mathcal { N } ( 0 , \alpha I _ { d } ) .
$$

Coupling Y with $Y + { \sqrt { \alpha } } \xi$ gives

$$
\begin{array} { r } { W _ { 2 } ^ { 2 } ( \mu _ { a } ^ { S } * \mathcal { N } ( 0 , \alpha I _ { d } ) , \mu _ { a } ^ { S } ) \leq \mathbb { E } | \sqrt { \alpha } \xi | ^ { 2 } = \alpha d \quad \Longrightarrow \quad W _ { 2 } ( \rho _ { 1 } ^ { a } , \mu _ { a } ^ { S } ) \leq \sqrt { \alpha d } . } \end{array}
$$

Let

$$
\phi _ { \alpha } ( u ) = ( 2 \pi \alpha ) ^ { - d / 2 } \exp ( - | u | ^ { 2 } / ( 2 \alpha ) ) .
$$

Then, the smoothed density is

$$
\rho _ { t } ^ { a } ( z ) = \int \phi _ { \alpha } ( z - ( ( 1 - t ) x + t y ) ) \gamma _ { \alpha } ( d x ) \mu _ { a } ^ { S } ( d y ) > 0 , \quad \forall z , t \in [ 0 , 1 ] .
$$

The current is

$$
J _ { t } ^ { a } ( z ) = \int ( y - x ) \phi _ { \alpha } ( z - ( ( 1 - t ) x + t y ) ) \gamma _ { \alpha } ( d x ) \mu _ { a } ^ { S } ( d y ) .
$$

The velocity is

$$
v ^ { a } ( z , t ) = \frac { J _ { t } ^ { a } ( z ) } { \rho _ { t } ^ { a } ( z ) } = \mathbb { E } [ Y - X \mid Z _ { t } ^ { a } = z ] .
$$

Lemma 5.2 (Smoothed flow matching velocity). For fixed $\alpha \in ( 0 , 1 )$ and $S \ < \ \infty$ , the velocity $v ^ { a }$ has uniform path moment bounds: for every $p \geq 2$

$$
\operatorname* { s u p } _ { a \in A ( \nu , p _ { 0 } , \lambda , M ) } \operatorname* { s u p } _ { t \in [ 0 , 1 ] } \int | v ^ { a } ( z , t ) | ^ { p } d \rho _ { t } ^ { a } ( z ) \leq C _ { \alpha , S , p } .
$$

It also has the uniform linear growth bound

$$
| v ^ { a } ( z , t ) | \leq C _ { \alpha , S } ( 1 + | z | ) .
$$

Moreover, $\rho _ { t } ^ { a }$ satisfies the continuity equation with this velocity field,

$$
\begin{array} { r } { \partial _ { t } \rho _ { t } ^ { a } + \nabla \cdot \left( \rho _ { t } ^ { a } v ^ { a } \right) = 0 . } \end{array}\tag{5.3}
$$

Finally, we take a smooth cutof $\chi _ { R } \in C _ { c } ^ { \infty } (  { \mathbb { R } } ^ { d } )$ such that

$$
\chi _ { R } ( x ) = 1 { \mathrm { ~ f o r ~ } } | x | \leq R , \qquad \chi _ { R } ( x ) = 0 { \mathrm { ~ f o r ~ } } | x | \geq 2 R , \qquad 0 \leq \chi _ { R } \leq 1 .
$$

For a fixed $a \in \mathcal { A } ( \nu , p _ { 0 } , \lambda , M )$ , define

$$
v ^ { a , R } ( x , t ) = \chi _ { R } ( x ) v ^ { a } ( x , t ) .
$$

Lemma 5.3 (Lipschitz continuity of the target velocity). For every fixed α $\mathbf { \chi } _ { \left[ \in \right. } ( 0 , 1 ) , S < \infty$ and $R > 0 , a \mapsto v ^ { a , R }$ is continuous as a map from $\mathcal { A } ( \nu , p _ { 0 } , \lambda , M )$ to $C ( \overline { { B } } _ { 2 R } \times [ 0 , 1 ] ; \mathbb { R } ^ { d } )$ Moreover,

$$
L _ { \alpha , S , R } = \operatorname* { s u p } _ { \substack { a \in \mathcal { A } ( \nu , p _ { 0 } , \lambda , M ) } } \operatorname* { s u p } _ { t \in [ 0 , 1 ] } \operatorname* { L i p } _ { x } v ^ { a , R } ( \cdot , t ) < \infty .
$$

Let $\mu _ { a } ^ { S , R }$ be the terminal law generated by

$$
\dot { X } _ { t } = v ^ { a , R } ( X _ { t } , t ) , \qquad X _ { 0 } \sim \gamma .
$$

Lemma 5.4 (Truncation consistency). For every fixed $\alpha \in ( 0 , 1 )$ and $S < \infty$

$$
\varepsilon _ { \mathrm { t r u n } } ( \alpha , S , R ) : = \operatorname* { s u p } _ { \substack { a \in \mathcal { A } ( \nu , p _ { 0 } , \lambda , M ) } } W _ { 2 } ( \rho _ { 1 } ^ { a } , \mu _ { a } ^ { S , R } ) \leq C _ { \alpha , S , p } R ^ { - ( p - 2 ) / 2 } .
$$

The proofs of Lemmas 5.2–5.4 are given in Appendix A.1.

## 5.3 Expressivity of the neural operator approximation

Now, we will consider a compact subset K of $\mathcal { A } ( \nu , p _ { 0 } , \lambda , M )$ for some parameters $\nu , p _ { 0 } , \lambda , M$ By Proposition 5.1, a closed set $K \subset \mathcal { A } ( \nu , p _ { 0 } , \lambda , M )$ is compact if the following hold: (i) for each ball $B _ { m } = \{ x : | x | \leq m \}$ , the restrictions of $a \in K$ on $B _ { m }$ are uniformly bounded; (ii) for each $B _ { m } .$ , the restrictions of $a \in K$ on $B _ { m }$ are equicontinuous. Lemma 5.1 shows that $\mathcal { A } ( \nu , p _ { 0 } , \lambda , M )$ is closed in $\mathcal { X } _ { a }$ . Hence the closure in $\mathcal { X } _ { a }$ of any family in $\mathcal { A } ( \nu , p _ { 0 } , \lambda , M )$ that satisfies (i) and (ii) is such a set $K$

Below, we put the following assumption on the neural operator architecture. We will then discuss how the assumption is satisfied for the architecture we use in the next section.

Assumption 5.1 (Compact domain neural operator approximation). For every compact set $K$ , every fixed envelope constant $C _ { \mathrm { e n v } } < \infty ,$ every $R > 0$ , every continuous operator

$$
\Psi : K  C ( \overline { { { B } } } _ { 2 R } \times [ 0 , 1 ] ; \mathbb { R } ^ { d } )
$$

with $| \Psi ( a ) ( x , t ) | \leq C _ { \mathrm { e n v } } ( 1 + | x | )$ , and every $\delta > 0 _ { : }$ , the following hold.

(a) The model class contains a parameter θ such that

$$
\operatorname* { s u p } _ { a \in K } \operatorname* { s u p } _ { ( x , t ) \in \overline { { B } } _ { 2 R } \times [ 0 , 1 ] } | \mathcal { N } _ { \theta } ( a ) ( x , t ) - \Psi ( a ) ( x , t ) | < \delta .
$$

(b) The selected realizations are locally Lipschitz in the spatial variable $( i . e .$ , Lipschitz on compact sets) and satisfy the global linear growth envelope for constant C depending on the chosen parameters such that

$$
| \mathcal { N } _ { \theta } ( a ) ( x , t ) | \leq C ( 1 + | x | ) , \qquad a \in K , \ x \in \mathbb { R } ^ { d } , \ t \in [ 0 , 1 ] .
$$

Assumption 5.1 is the ordinary compact domain universal approximation property (UAP) for the neural operator. The reduction theorem uses the Lipschitz constant of the target velocity $v ^ { a , R }$ , not the Lipschitz constant of the approximating network. The global linear growth clause is only used to make the exact learned ODE globally well posed and to control the probability that the learned trajectory leaves the compact approximation domain.

Next, one needs the following auxiliary results, whose proofs are given in Appendix A.3.

Proposition 5.3 (ODE stability for a Lipschitz target). Let $v _ { t }$ and $\hat { v } _ { t }$ generate flows $\Phi _ { t }$ and $\hat { \Phi } _ { t }$ from the same initial law $\rho _ { 0 }$ . Assume $v _ { t }$ is globally Lipschitz in x, uniformly in $t ,$ with constant L, and

$$
\operatorname* { s u p } _ { x , t } | v _ { t } ( x ) - \hat { v } _ { t } ( x ) | \leq \delta .
$$

Then

$$
W _ { 2 } ( ( \Phi _ { 1 } ) _ { \# } \rho _ { 0 } , ( \hat { \Phi } _ { 1 } ) _ { \# } \rho _ { 0 } ) \leq e ^ { L } \delta .
$$

Proposition 5.4 (Reduction from compact operator approximation to sampling error). Let $K \subset \mathcal { A } ( \nu , p _ { 0 } , \lambda , M )$ be a compact set of coeficients $a = ( b , \sigma )$ for some positive parameters $\nu > 0 , p _ { 0 } > 2 , \lambda > 0 , M > 0$ . Fix $\alpha \in ( 0 , 1 )$ , S < ∞, R > 0, and $p > 2$ . The implemented velocity field is the plain neural operator

$$
\hat { v } _ { \theta } [ a ] ( x , t ) : = \mathcal { N } _ { \theta } ( a ) ( x , t ) .
$$

Assume $\mathcal { N } _ { \theta } ( a )$ is locally Lipschitz in $x ,$ and satisfies

$$
| \mathcal { N } _ { \theta } ( a ) ( x , t ) | \leq C _ { \alpha , S } ( 1 + | x | ) , \qquad a \in K , \ x \in \mathbb { R } ^ { d } , \ t \in [ 0 , 1 ] ,
$$

and define

$$
\delta _ { \theta , \alpha , S , R } : = \operatorname* { s u p } _ { a \in K } \operatorname* { s u p } _ { ( x , t ) \in \overline { { B } } _ { 2 R } \times [ 0 , 1 ] } | \mathcal { N } _ { \theta } ( a ) ( x , t ) - v ^ { a , R } ( x , t ) | .
$$

If $\delta _ { \theta , \alpha , S , R } < R e ^ { - L _ { \alpha , S , R } }$ and $\hat { \mu } _ { a , \theta }$ is the terminal law generated by vˆ<sub>θ</sub>[a] from $X _ { 0 } \sim \gamma$ , then sup $W _ { 2 } ( \hat { \mu } _ { a , \theta } , \mu _ { a } ) \leq e ^ { L _ { \infty } , s , \mu } \delta _ { \theta , \alpha , S , R } + C _ { \alpha , S , p } R ^ { - ( p - 2 ) / 2 } + \varepsilon _ { \mathrm { t r u n } } ( \alpha , S , R ) + \sqrt { \alpha d } + M _ { p _ { 0 } } ^ { 1 / 2 } S ^ { - ( p _ { 0 } - 2 ) / 2 } .$ a∈K

The five terms have distinct meanings. The first is the compact domain neural operator approximation error. The second is the escape tail error introduced by using the exact, unprojected learned ODE while only controlling the approximation on $B _ { 2 R }$ . The third is the ODE error caused by truncating the target velocity outside $B _ { R }$ . The fourth is the Gaussian mollification bias, and the fifth is the error from projecting the invariant measure to $B _ { S }$

The following then gives the main claim about expressivity of the model, and the proof is provided in Appendix A.3.

Theorem 5.1 (Ordinary UAP implies sampler expressivity). Let $K \subset \mathcal { A } ( \nu , p _ { 0 } , \lambda , M )$ be a compact set for some positive parameters $\nu > 0 , p _ { 0 } > 2 , \lambda > 0 , M > 0$ . Suppose Assumption 5.1 holds. Then for every $\eta > 0$ there exist $S < \infty , \alpha \in ( 0 , 1 ) , R > 0$ , and a parameter θ such that

$$
\operatorname* { s u p } _ { a \in K } W _ { 2 } ( \hat { \mu } _ { a , \theta } , \mu _ { a } ) < \eta .
$$

## 5.4 Universal approximation property of the Lagrangian attention neural operator

In this subsection, we establish suficient conditions for the Lagrangian attention neural operator to satisfy the approximation assumption used in the preceding expressivity theorem. The key is that the finite probes must resolve the coeficient family. We show that this holds with high probability under the stated assumptions on the probe distribution.

For the Lagrangian sensors, we do not need the cutof J as for the grid samplers. We recall that a Lagrangian input is the finite unordered set of probe observations of Section 4,

$$
{ \mathcal { S } } _ { m , L } ( a ) = \{ Y _ { 1 } ^ { a } , \ldots , Y _ { m } ^ { a } \} , \qquad Y _ { i } ^ { a } = ( y _ { i , j } ^ { a } ) _ { j = 0 } ^ { L } \in \mathbb { R } ^ { ( L + 1 ) \times ( 2 d + d d _ { W } ) } ,
$$

with the initial probe locations, probe noise, step size, and trajectory length L fixed before evaluation. The realized operator ${ \mathcal { N } } _ { \theta } ( a ) = v _ { \theta } [ a ]$ is the trajectory CNN encoder, the finite depth Perceiver aggregator Encoder with readout $z ~ \in ~ \mathbb { R } ^ { q }$ , and the AdaLN-conditioned residual decoder Decoder $( z , x , t )$

Proposition 5.5 (Finite sensor Lagrangian Perceiver–AdaLN approximation). Let $Y =$ $\overline { { B } } _ { 2 R } \times [ 0 , 1 ]$ , and fix an envelope constant $C _ { \mathrm { e n v } } < \infty$ . Assume:

(a) (Finite Sensor Resolution, FS) Finite probes resolve K: for every $\varepsilon > 0$ there are $m , L ,$ fixed probe seeds, and $r > 0$ such that

$$
d _ { \mathrm { s e n s } } ( S _ { m , L } ( a ) ,  { S } _ { m , L } ( a ^ { \prime } ) ) < r \implies d _ {  { \mathcal { X } } _ { a } } ( a , a ^ { \prime } ) < \varepsilon , \qquad a , a ^ { \prime } \in K ,
$$

where $d _ { \mathrm { s e n s } }$ is any distance on finite sensor sets.

(b) (Encoder-decoder Universality, EU) On the compact image $S _ { m , L } ( K )$ , the encoder-decoder architecture class uniformly approximates any continuous map invariant under sensor permutations $F : \mathcal { S } _ { m , L } ( K ) \times Y  \mathbb { R } ^ { d }$ whose realizations are locally Lipschitz in x.

Then the finite sensor encoder-decoder class satisfies Assumption 5.1 (a). If the activation function is globally Lipschitz, then Assumption 5.1 (b) is also satisfied.

The proof is given in Appendix A.3.

We remark that a natural choice is

$$
d _ { \mathrm { s e n s } } ( \{ y _ { i } \} , \{ y _ { i } ^ { \prime } \} ) = \operatorname* { m a x } _ { i j } \| a ( x _ { i j } ) - a ^ { \prime } ( x _ { i j } ) \| .
$$

Any distance that makes $S ( K )$ compact and metrizes the topology of the sensor set sufices. Next, we discuss the validity of the conditions used. We first verify that the condition (FS) holds with large probability as summarized in Theorem 5.2. The proof is given in Appendix A.3.

Theorem 5.2. Suppose that the initial points $\{ x _ { i 0 } \} _ { i = 1 } ^ { m }$ are sampled from some distribution $\nu _ { 0 } ( d x ) = \rho _ { 0 } ( x )$ dx with a continuous density $\rho _ { 0 } ( x ) > 0$ that is positive everywhere. For any $\delta \in ( 0 , 1 )$ , there is $m _ { 0 } = m ( \delta , r , \varepsilon ) > 0$ such that when $m > m _ { 0 }$ , the Lagrangian attention neural operator satisfies the condition (FS) with probability $1 - \delta .$ . Consequently, provided the architecture satisfies the ordinary finite dimensional UAP on the compact set with permutation invariance (i.e., the (EU) property), the Lagrangian attention neural operator has the expressivity with probability at least 1 − δ.

Remark 5.2. We remark that the estimate of m in the proof of Theorem 5.2 in Appendix A.3 has curse of dimensionality. The proof in Appendix A.3 is based on the h-net and does not take advantage of the Monte Carlo feature of the method, so it should have overestimated m. The improvement needs a better topology for the coeficient functions that could make use of the Monte Carlo rate. This is left for future study.

Next, let us discuss about universality condition and provide some evidence. Since m and L are fixed, the sensor set $\{ e _ { 1 } , \ldots , e _ { m } \} \subset \mathbb { R } ^ { d _ { e } }$ lives in the compact finite dimensional space $\mathbb { R } ^ { m \times d _ { e } } .$ , so (EU) is an ordinary $\mathrm { U A P }$ for continuous functions on a compact set—the only nonstandard requirement is permutation invariance in the sensor index. Three existing results together make (EU) reasonable for the present architecture:

• Cross attention is a universal approximator. Liu et al. [21] prove that a single-head cross attention layer, preceded by a linear projection, universally approximates any continuous function on a compact domain in $L ^ { \infty }$ . The cross attention block $\operatorname { C A } ( Z , E )$ of our encoder fits this setting directly: the learned initial latent $Z ^ { ( 0 ) }$ plays the role of the query and the sensor array E provides the keys and values.

• Self attention + FFN is a universal approximator of equivariant maps. Yun et al. [37] prove that a transformer (multihead self attention followed by a positionwise FFN) universally approximates any continuous permutation-equivariant sequence-to-sequence function on a compact domain. The latent self attention and FFN blocks in each encoder layer are exactly this subarchitecture, applied to the fixed length latent array of k tokens.

• Permutation invariance is preserved by construction. Cross attention is invariant to permutation of the keys and values (the sensor embeddings) because each attention weight is computed by an inner product between a fixed query and each key independently before softmax normalization; permuting the sensors permutes the weights but leaves the weighted sum unchanged. The subsequent self attention operates on the latent array, which has no sensor index, and the linear readout $W ^ { \mathrm { o u t } } \mathrm { v e c } ( Z ^ { ( L _ { \mathrm { e n c } } ) } )$ is fixed dimension and sensor-order-independent. Our encoder is structurally identical to the Set Transformer of [16], which was introduced precisely for set aggregation that is invariant under permutations.

A formal proof combining all three stages in a single theorem statement does not yet exist in the literature; we treat (EU) as an assumption, supported by the above components. The decoder, given a fixed z and compact Y, satisfies the standard MLP UAP.

## 6 Resolution Invariance

This section is about the resolution of the input function. It explains when changing the sensor resolution leaves the realized sampler stable, in expectation over the probe trajectories, for the fixed trained attention architecture. A finite sensor Lagrangian Perceiver–AdaLN approximation statement records suficient conditions for the compact domain approximation assumption once the probes resolve the coeficient family. The resolution invariance thus allows the model to be trained on data samples with variable resolution, and be applied for diferent resolution for new unseen cases. For random probes, the probes are sampled in the whole state space according to a prescribed probe law, so the resolution parameter is the number m of probes.

In this subsection, $K \subset \mathcal X _ { a }$ denotes a compact coeficient family. Throughout this subsection the weights θ are fixed and suppressed from the notation.

For a fixed full space probe recipe, one probe produces a random observation $Y$ as introduced in Section 4—the visited states together with the drift b and difusion σ recorded along a short probe path—with CNN embedding $e = \mathrm { C N N } ( Y ) \in \mathbb { R } ^ { d _ { e } }$ and law of the sensor embedding

$$
\nu _ { a } = \operatorname { L a w } ( e ) .
$$

With m independent probes $\mathbf { e } = ( e _ { 1 } , \ldots , e _ { m } ) , e _ { i } \stackrel { \mathrm { i . i . d . } } { \sim } \nu _ { a }$ , the empirical law of the sensor embeddings is

$$
\hat { \nu } _ { a , m } = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \delta _ { e _ { i } } .
$$

Because the sensor order is irrelevant, we identify the encoded sensor set with its empirical measure.

The encoder reads the sensors only through cross attention, and only through the projected sensor $\tilde { e } = e P _ { e } \in \mathbb { R } ^ { d _ { z } }$ , with $P _ { e }$ the shared projection of Section 4.2. Replacing the sensor rows of Atten by a measure ν on the embeddings, a head with query q is

$$
A _ { \ell } ^ { h } ( q , \nu ) = \frac { \int F _ { \ell } ^ { h } ( q , e ) d \nu ( e ) } { \int \omega _ { \ell } ^ { h } ( q , e ) d \nu ( e ) } , \quad \omega _ { \ell } ^ { h } ( q , e ) = \mathrm { e x p } \Big ( \frac { \langle q , \tilde { e } W _ { \ell , h } ^ { K } \rangle } { \sqrt { d _ { h } } } \Big ) , \quad F _ { \ell } ^ { h } ( q , e ) = \omega _ { \ell } ^ { h } ( q , e ) \tilde { e } W _ { \ell , h } ^ { V } \in \mathbb { R } ^ { d _ { h } } ,
$$

which at $\begin{array} { r } { \nu = \hat { \nu } _ { a , m } = \frac { 1 } { m } \sum _ { i } \delta _ { e } } \end{array}$ reduces to the Atten head over the m sensors.

We now use this measure formulation to write the full encoder recursion. We use $H _ { \ell }$ to denote the cross attention message and $G _ { \ell }$ the self attention–MLP update. With token queries $q = z _ { r } W _ { \ell , h } ^ { Q }$ , the per-head outputs assembled by $W _ { \ell } ^ { O }$ give $H _ { \ell } ( Z , \nu ) \in \mathbb { R } ^ { k \times d _ { z } }$ , with rows

$$
\big [ H _ { \ell } ( Z , \nu ) \big ] _ { r } = \big [ A _ { \ell } ^ { h } ( z _ { r } W _ { \ell , h } ^ { Q } , \nu ) \big ] _ { h = 1 } ^ { H } W _ { \ell } ^ { O } , \quad r = 1 , \dots , k .\tag{6.1}
$$

Here, $G _ { \ell }$ represents the transform made by the self attention and the MLP

$$
G _ { \ell } ( Y ) = \mathrm { r } [ \mathrm { M L P } ] \big ( \mathrm { r } [ \mathrm { S A } ] ( Y ) \big ) ,
$$

where $\operatorname { r } [ f ] ( Y ) = \operatorname { L N } \left( Y + f ( Y ) \right)$ is a residual sublayer followed by LayerNorm. The latents are then transformed by the self attention and the MLP through the depth- $L _ { \mathrm { e n c } }$ recursion:

$$
\begin{array} { r } { Z ^ { ( \ell + 1 ) } = G _ { \ell } \bigl ( \mathrm { L N } ( Z ^ { ( \ell ) } + H _ { \ell } ( Z ^ { ( \ell ) } , \nu ) ) \bigr ) , \quad \ell = 0 , \ldots , L _ { \mathrm { e n c } } - 1 , } \end{array}\tag{6.2}
$$

from the learned $Z ^ { ( 0 ) }$

Hence, the encoder is then a functional of the measure

$$
{ \mathrm { E n c o d e r } } ( \nu ) : = W ^ { \mathrm { o u t } } \mathrm { v e c } \left( Z ^ { ( L _ { \mathrm { e n c } } ) } \right) .
$$

Denote

$$
z _ { a , m } = \mathrm { E n c o d e r } ( \hat { \nu } _ { a , m } ) , \qquad z _ { a } = \mathrm { E n c o d e r } ( \nu _ { a } )
$$

the encodings for the empirical embedded sensor law, and the population embedded sensor law respectively.

The finite and infinite sensor velocity fields are the decoder pushforwards

$$
v _ { a , m } ( x , t ) = \operatorname { D e c o d e r } ( z _ { a , m } , x , t ) , \qquad v _ { a } ( x , t ) = \operatorname { D e c o d e r } ( z _ { a } , x , t ) .
$$

The corresponding terminal laws are the random measure $\hat { \mu } _ { a , m }$ (it depends on the specific random realized probes e) and the limit measure $\hat { \mu } _ { a }$ corresponding to the continuum probe measure:

$$
\hat { \mu } _ { a , m } = \mathrm { L a w } ( X _ { 1 } ^ { a , m } ) , \qquad \dot { X } _ { t } ^ { a , m } = v _ { a , m } ( X _ { t } ^ { a , m } , t ) , \qquad X _ { 0 } ^ { a , m } \sim \gamma ,
$$

and

$$
\hat { \mu } _ { a } = \mathrm { L a w } ( X _ { 1 } ^ { a } ) , \qquad \dot { X } _ { t } ^ { a } = v _ { a } ( X _ { t } ^ { a } , t ) , \qquad X _ { 0 } ^ { a } \sim \gamma .
$$

For each probe realization, these definitions give the following finite sensor map.

$$
a = ( b , \sigma ) \longmapsto S _ { m , L } ( a ) \longmapsto { \hat { \nu } } _ { a , m } \longmapsto z _ { a , m } \longmapsto v _ { a , m } \longmapsto { \hat { \mu } } _ { a , m } .
$$

Definition 6.1 (Random sensor resolution invariance). The Lagrangian Perceiver sampler is random sensor resolution invariant on K if the population-probe terminal law $\hat { \mu } _ { a }$ is defined for each $a \in K$ and

$$
\operatorname* { l i m } _ { m \to \infty } \operatorname* { s u p } _ { a \in K } \mathbb { E _ { e } } W _ { 2 } ( \hat { \mu } _ { a , m } , \hat { \mu } _ { a } ) = 0 ,
$$

where the expectation is over the random probe trajectories.

Remark 6.1. The law $\hat { \mu } _ { a }$ is the continuum limit only in the sense that the empirical probe measure has been replaced by the underlying probe law in the same finite depth Perceiver architecture; the probe recipe itself is fixed.

We now state the two structural assumptions. Both are mild and realized by the architecture; the quantitative bounds the proof uses—boundedness and Lipschitz continuity of the attention weights, a strictly positive denominator, compact reachable latents, and a Lipschitz decoder—are then consequences (Lemma 6.1), not separate hypotheses.

Assumption 6.1 (Compact sensor embeddings and a regular decoder). The following hold uniformly over $a \in K$

(A1) Compact embedding support. There is a fixed compact set $K _ { e } \subset \mathbb { R } ^ { d _ { e } }$ with supp $\nu _ { a } \subseteq$ K<sub>e</sub> for every $a \in K$

(A2) Decoder state regularity. There is $C _ { \theta } < \infty$ , and for every $R < \infty$ an $L _ { \theta , R } ^ { x } < \infty$ such that for all $a \in K$ , every reachable latent z, all $t \in [ 0 , 1 ]$ and $x , y \in { \overline { { B } } } _ { R }$

$$
| \operatorname { D e c o d e r } ( z , x , t ) | \leq C _ { \theta } ( 1 + | x | ) , \qquad | \operatorname { D e c o d e r } ( z , x , t ) - \operatorname { D e c o d e r } ( z , y , t ) | \leq L _ { \theta , R } ^ { x } | x - y | .
$$

Let us make some discussions on the assumptions above. Regarding (A1), the persensor embedding of Section 4 pools the final convolutional block, $\begin{array} { r } { e _ { i } \ = \ \frac { 1 } { L + 1 } \sum _ { j = 0 } ^ { L } C _ { j , : } ^ { ( N ) } } \end{array}$ with $C ^ { ( N ) } = \mathrm { G E L U } ( \mathrm { L a y e r N o r m } ( \cdot \cdot \cdot ) )$ . The LayerNorm normalizes each row to Euclidean norm $\leq \sqrt { d _ { e } }$ and applies its fixed afine map, so its output rows are bounded by some $\rho _ { e } < \infty ;$ since $| \mathrm { G E L U } ( t ) | \leq | t |$ , every $\| C _ { j , : } ^ { ( N ) } \| \le \rho _ { e }$ , and averaging gives $\| e _ { i } \| \le \rho _ { e }$ . Hence supp $\nu _ { a } \subseteq K _ { e } : = \overline { { B } } _ { \rho _ { e } }$ , a fixed compact set independent of the probe.

Regarding (A2), in the decoder of Section 4, x enters only through $h ^ { ( 0 ) } = { W _ { \mathrm { i n } } [ x ; \tau ] + }$ $b _ { \mathrm { i n } } ;$ the AdaLN blocks and the output map $v = W _ { v } \mathrm { L N } ( h ^ { ( N _ { d } ) } )$ compose the afine maps $W _ { \mathrm { i n } } , W _ { 1 } , W _ { 2 } , W _ { v }$ , the activation Mish $( L _ { \mathrm { a c t } } : = \mathrm { L i p } ( \mathrm { M i s h } ) )$ , and ε-LayerNorms $( L _ { \mathrm { L N } } : =$ Lip(LN)), with the scales $1 + \gamma ( c ) , \alpha ( c )$ applied multiplicatively. Each block is thus Lipschitz in its input with constant

$$
1 + \| \alpha ( c ) \| _ { \infty } \| W _ { 1 } \| \| W _ { 2 } \| L _ { \mathrm { a c t } } ^ { 2 } L _ { \mathrm { L N } } ^ { 2 } \| 1 + \gamma ( c ) \| _ { \infty } ^ { 2 } ,
$$

so $x \mapsto$ Decoder $( z , x , t )$ has Lipschitz constant at most the product of $\lVert W _ { \mathrm { i n } } \rVert$ , these $N _ { d }$ block constants, and $\| W _ { v } \| L _ { \mathrm { L N } }$ . As γ, α are linear and $c = W _ { c } [ z ; \tau ] + b _ { c }$ ranges over the compact image of $K _ { z } \times [ 0 , 1 ]$ , the scales $\| 1 + \gamma ( c ) \| _ { \infty } , \| \alpha ( c ) \| _ { \infty }$ are bounded there, so that product is a finite Λ uniform over $K _ { z } \times [ 0 , 1 ]$ , giving (A2) with $L _ { \theta , R } ^ { x } = \Lambda$ and $C _ { \theta } = \operatorname* { s u p } _ { z \in K _ { z } , t }$ | Decoder $( z , 0 , t ) | + \Lambda$

Lemma 6.1 (Consequences of compactness). Under (A1) and the fixed smooth encoder of Section 4.2, whose blocks end in ε-regularized LayerNorm. Then, the following hold.

(i) there are constants $0 < \omega _ { \mathrm { m i n } } \le \omega _ { \mathrm { m a x } } < \infty$ and $M _ { F } > 0$ such that

$$
\omega _ { \mathrm { m i n } } \le \omega _ { \ell } ^ { h } ( q , e ) \le \omega _ { \mathrm { m a x } }
$$

and $\| F _ { \ell } ^ { h } ( q , e ) \| \leq M _ { F } ;$ consequently $\textstyle \int \omega _ { \ell } ^ { h } ( q , \cdot ) d \nu \geq \omega _ { \mathrm { m i n } }$ for every probability measure ν on $K _ { e }$ , in particular for both $\hat { \nu } _ { a , m }$ and $\nu _ { a ; }$

(ii) there is $L _ { \omega } > 0$ such that

$$
\vert \omega _ { \ell } ^ { h } ( q , e ) - \omega _ { \ell } ^ { h } ( q ^ { \prime } , e ) \vert + \Vert F _ { \ell } ^ { h } ( q , e ) - F _ { \ell } ^ { h } ( q ^ { \prime } , e ) \Vert \leq L _ { \omega } \Vert q - q ^ { \prime } \Vert ;
$$

(iii) There are two compact sets $K _ { z } , K _ { q }$ such that every reachable latent array lies in $K _ { z }$ and every query in $K _ { q } ;$ consequently, the self attention–MLP update $G _ { \ell }$ is Lipschitz on the reachable compact set, with constants $L _ { \ell } ^ { G }$ ;

(iv) for every $R < \infty$ the decoder is Lipschitz in z on $K _ { z } \times \overline { { B } } _ { R } \times [ 0 , 1 ]$ : there is $L _ { \theta , R } ^ { \mathrm { d e c } } < \infty$ with

$$
| \operatorname { D e c o d e r } ( z , x , t ) - \operatorname { D e c o d e r } ( z ^ { \prime } , x , t ) | \leq L _ { \theta , R } ^ { \operatorname { d e c } } \| z - z ^ { \prime } \| .
$$

These bounds yield the following uniform law of large numbers (ULLN). Both proofs are in Appendix $\mathrm { A . 4 }$

Lemma 6.2 (ULLN for the attention class). Over the finitely many layers, heads, and value coordinates, set

$$
\mathfrak { F } _ { \theta } = \bigcup _ { \ell , h } \Bigl ( \{ \omega _ { \ell } ^ { h } ( q , \cdot ) : q \in K _ { q } \} \cup \{ [ F _ { \ell } ^ { h } ( q , \cdot ) ] _ { r } : q \in K _ { q } , \ r = 1 , \ldots , d _ { h } \} \Bigr ) ,
$$

a uniformly bounded family of functions on $K _ { e }$ , and

$$
\delta _ { m } ^ { \mathrm { a t t } } : = \operatorname* { s u p } _ { a \in K } \mathbb { E } _ { \mathbf { e } } \operatorname* { s u p } _ { f \in \mathfrak { F } _ { \theta } } \Big | \int f d \hat { \nu } _ { a , m } - \int f d \nu _ { a } \Big | .
$$

Under Assumption 6.1, with $\mathbf { e } = ( e _ { 1 } , \hdots , e _ { m } )$ i.i.d. from $\nu _ { a }$ and supp $\nu _ { a } \subseteq K _ { e } , \delta _ { m } ^ { \mathrm { a t t } } \to 0$

Remark 6.2. The limit $\delta _ { m } ^ { \mathrm { a t t } } \to 0$ needs only total boundedness of $\mathfrak { F } _ { \theta }$ . A rate is available but is not used below: by Lemma $6 . 1 ( i i )$ the map $q \mapsto f _ { q }$ is Lipschitz from the compact $K _ { q }$ into $( C ( K _ { e } ) , \| \cdot \| _ { \infty } )$ , so $\mathfrak { F } _ { \theta }$ has finite metric entropy log $\quad \ : N ( \varepsilon , \mathfrak { F } _ { \theta } , \| \cdot \| _ { \infty } ) \lesssim \quad$ dim(q) log(1/ε), and the standard Dudley/symmetrization bound then gives $\delta _ { m } ^ { \mathrm { a t t } } \leq C _ { \theta } m ^ { - 1 / 2 }$ with $C _ { \theta }$ independent of a (the class and its envelope $M _ { \theta } = \operatorname* { s u p } _ { f \in { \mathfrak { F } } _ { \theta } } \| f \| _ { \infty }$ are distribution independent).

Finally, we obtain the following resolution invariance result for the random sensor attention sampler. The proof is given in Appendix A.4.

Theorem 6.1. Under Assumption 6.1, the population-probe terminal law $\hat { \mu } _ { a }$ is defined for each $a \in K$ , and

$$
\operatorname* { l i m } _ { m \to \infty } \operatorname* { s u p } _ { a \in K } \mathbb { E _ { e } } W _ { 2 } ( \hat { \mu } _ { a , m } , \hat { \mu } _ { a } ) = 0 .
$$

so the Lagrangian Perceiver sampler is random sensor resolution invariant on K in expectation over the probes.

We would like to point out that the limit $\hat { \mu } _ { a }$ is the population-probe law of the same Perceiver with fixed weights, not the target invariant measure $\mu _ { a }$

## 7 Numerical Experiments

We perform numerical tests to validate our framework. The necessary setup and results needed to support our conclusions are given here while leave other details (data generation, architecture, training, software etc) are in Appendix B.

To compare generated and reference samples, we use the Sinkhorn divergence, a biascorrected, entropy-regularized approximation to quadratic optimal transport [10]. For their empirical probability measures $\mu$ and $\nu ,$ define

$$
\begin{array} { l } { { \mathrm { O T } _ { \varepsilon } ( \mu , \nu ) = \displaystyle \operatorname* { i n f } _ { \pi \in \Pi ( \mu , \nu ) } \left\{ \frac { 1 } { 2 } \int | x - y | ^ { 2 } d \pi ( x , y ) + \varepsilon \mathrm { K L } ( \pi \| \mu \otimes \nu ) \right\} , \hfill } } \\ { { \quad S _ { \varepsilon } ( \mu , \nu ) = \mathrm { O T } _ { \varepsilon } ( \mu , \nu ) - \frac { 1 } { 2 } \mathrm { O T } _ { \varepsilon } ( \mu , \mu ) - \frac { 1 } { 2 } \mathrm { O T } _ { \varepsilon } ( \nu , \nu ) . \hfill } } \end{array}\tag{7.1}
$$

Here KL denotes relative entropy, and $\varepsilon > 0$ controls the amount of smoothing. $S _ { \epsilon }$ is the Sinkhorn divergence. The self-comparison terms remove the entropic bias and give $S _ { \varepsilon } ( \mu , \mu ) = 0 . \ S _ { \varepsilon } ( \mu , \nu )$ converges to $\scriptstyle { \frac { 1 } { 2 } } W _ { 2 } ^ { 2 } ( \mu , \nu )$ as $\varepsilon \downarrow 0$ . We set $\varepsilon = 0 . 0 0 2 5$ in the experiments, and report $S _ { \varepsilon }$ without a square root, so the Sinkhorn and $W _ { 2 }$ columns have diferent scales. Lower values indicate closer agreement. Sinkhorn divergence is easier to compute and more stable then $W _ { 2 }$ in high-dimensional spaces.

## 7.1 1D Variable Noise and 1D Rare Event

We first test the amortized neural sampler in 1D Variable Noise and 1D Rare Event, using the simple grid Multi-input DeepONet described at the beginning of Section 4. These experiments show that operator learning of invariant measures is feasible, and locate the regime— slow mixing—where it gains an advantage over MCMC. Both experiments are trained on 65,536 function instances and evaluated on 1,024 test functions, reporting the Sinkhorn divergence and the 2-Wasserstein distance $\left( W _ { 2 } \right)$ against ground truth; the evaluation protocol and the shared architecture and training configuration are given in Appendix B.1 (Table B.1).

## 7.1.1 1D Variable Noise

This 1D Variable Noise example serves as a sanity check in a benign, fast mixing regime. MCMC performs well here. The experiment verifies that the amortized operator sampler produces reasonable samples across a large family of SDEs and establishes a baseline for comparison with the harder rare event setting that follows. We set $\boldsymbol { b } = - \nabla \boldsymbol { w } + \boldsymbol { s }$ and $\sigma = \operatorname* { m a x } \{ 0 . 2 5 , 1 + s ^ { \prime } \}$ , where $w ( x ) = ( \operatorname* { m a x } \{ | \bar { x } | - 2 , 0 \} ) ^ { 2 }$ is a fixed confining well and the lower bound on σ ensures uniform ellipticity. The perturbations $s , s ^ { \prime }$ (functions of $x )$ are independent draws from a Gaussian random field with squared exponential kernel (length scale 1, variance 1).

Table 7.1: 1D Variable Noise. Distribution metrics on 1,024 test functions (mean / median).
<table><tr><td></td><td>Sinkhorn</td><td> $W _ { 2 }$ </td><td>Time (s)</td></tr><tr><td>Ours</td><td>0.018 / 0.002</td><td>0.204 / 0.125</td><td>19.0</td></tr><tr><td>MCMC</td><td>0.001 / 0.0002</td><td>0.083 / 0.040</td><td>8.6</td></tr></table>

Table 7.1 reports the results, together with an MCMC baseline run on the same test functions under the same per-function sample budget. In this easy mixing regime, MCMC produces accurate samples eficiently, and our model is less accurate though comparable in speed. The representative samples in Figure 7.1 show that it nevertheless reproduces the target laws across the family, although it systematically underestimates the variance of the invariant measure and the Kolmogorov–Smirnov test rejects for most test functions. This gap is expected: an amortized model that generalizes across the entire function family in a single forward pass trades per-distribution accuracy for breadth.

## 7.1.2 1D Rare Event

1D Rare Event targets the regime where amortized operator sampling provides its primary advantage: SDEs with small difusion coeficients whose invariant measures are multimodal. MCMC methods must traverse high free energy barriers between modes, leading to mixing times that scale exponentially in $1 \bar { / } \sigma ^ { 2 }$ . Our neural sampler bypasses this bottleneck entirely. Its inference cost is independent of the mixing time.

We consider 1D SDEs whose invariant measure is a prescribed mixture of three Gaussians:

$$
p ( x ) = \sum _ { i = 1 } ^ { 3 } w _ { i } \mathcal { N } ( x \mid \mu _ { i } , \sigma _ { i } ^ { 2 } ) ,\tag{7.2}
$$

where $w _ { i } > 0 , \sum _ { i } w _ { i } = 1 , \mu _ { i } \sim \mathrm { U n i f } ( - 3 , 3 )$ , and $\sigma _ { i } \sim \mathrm { U n i f } ( 0 . 1 , 0 . 5 )$ . The difusion coeficient is a small random perturbation of the constant 0.1, and the drift is constructed from the zero flux condition of the stationary Fokker–Planck equation, so that p is by design the invariant density of the resulting SDE; the explicit construction is given in Appendix B.1. Because the target density is known, training samples are drawn directly from the GMM, making the experiment a controlled test of the model’s representational capacity and generalization in a rare event regime. The model reuses the architecture and hyperparameters of 1D Variable Noise.

![](images/9e2ae1ac2ef0ac987167bad72e1dd5a2a3f980008618e082a9ad8453e1bfbff1.jpg)

![](images/05eca99c39a6462ab6cb46e99601705000bea2eebb5f24fddb6f9896188660f0.jpg)

![](images/69a8ccb11a3d2a9ae2e0d8f1f440fac5f3a6a5ce6c11ed1a0c7525bfa930f1ec.jpg)

![](images/ea5d873fc4a1f12eb6aadeb927b3e61764d3e57368d47760260136cd04c9f244.jpg)

![](images/f20a56a5610eee12fe56b5709fc3241a11c6ac0f3131005be5f08ed542038963.jpg)

![](images/60505fc67db196d2b59091012e82373c6fd1d1b456c51224d585d09c1d75a07d.jpg)

![](images/6795c6af7f4af78f7147de9e9311954d99972d3c7fda8907fec45a1332d5f05e.jpg)

![](images/6b487fd4570e938d6a5a8543cd7a589c181b795dda735d6b2a1efe6d9d124d24.jpg)

![](images/6c47e8e50757720350d771faaa95aaa0c2d2b842df69b94d9ef2c2fa0e3fcabc.jpg)

![](images/714d435c5b2ff2cce86ad8c0976b64a0c034620eb0848a1712b627619094fd26.jpg)

![](images/f09a68b268b7ae49e80439616eb19ff4e0d274841b76152c0a1cb41e38b73fc4.jpg)

![](images/9e90713bb7ea532d224308b8f204f5b5b66f23071995c9acaf1aeba8c879c77b.jpg)  
Figure 7.1: 1D Variable Noise. Representative samples on test functions at the 25th, 50th, and 75th percentile Sinkhorn divergence. Each row shows one test function: reference histogram, MCMC histogram, model histogram, and CDF overlay.

Table 7.2 compares the model with MCMC baselines at varying time budgets, including a converged baseline and three baselines that limit the per-function MCMC budget. The model attains a lower mean Sinkhorn divergence than even the converged MCMC baseline while using $7 \times$ less elapsed time, and its median is comparable to the converged MCMC median with far fewer large failures: the error distributions are heavily skewed to the right (Figure B.2 in Appendix B.1). The gap in means is driven by MCMC’s tendency to become trapped in a single mode when noise is small, occasionally producing accurate samples (when all modes happen to be found) but often yielding very poor ones; representative failure cases are shown in Figure 7.2. Figure 7.3 shows the Sinkhorn divergence as a function of elapsed time; our method (star) dominates the MCMC Pareto frontier.  
Table 7.2: 1D Rare Event. Sinkhorn divergence and $W _ { 2 }$ on 1,024 test functions (mean / median).
<table><tr><td>Method</td><td>Sinkhorn</td><td> $W _ { 2 }$ </td><td>Time (s)</td></tr><tr><td>Ours</td><td>0.131 / 0.011</td><td>0.447 / 0.239</td><td>20.6</td></tr><tr><td>MCMC (6 s)</td><td>0.348 / 0.039</td><td>0.708 / 0.474</td><td>6.3</td></tr><tr><td>MCMC (12s)</td><td>0.326 / 0.019</td><td>0.659 / 0.381</td><td>12.4</td></tr><tr><td>MCMC (25 s)</td><td>0.299 / 0.009</td><td>0.614 / 0.281</td><td>24.6</td></tr><tr><td>MCMC (conv.)</td><td>0.254 / 0.003</td><td>0.535 / 0.161</td><td>136.9</td></tr></table>

![](images/a534cd9bbadc6a59df7b5b10884d325d0ab363a23e145de944aa36621a63c66f.jpg)

![](images/9b5b635489ee2d6d4d3ee5ec41db9378a518beefb18003f84c98490c6025a4f4.jpg)

![](images/390d55787db308b310ddb2d84b15652d3bc5f4af4708235dce12df1a18bcb385.jpg)

![](images/db55f97215943941a07d731e4c2a2b3095864b3b602cc111850cd64f22b80721.jpg)

![](images/5a361081f71020ed927367b650143750463dda30b3799698410aea6d563970be.jpg)

![](images/ef498b205c3903dcde9c603a1185d2b111837935f5409d03c784d61dd47be572.jpg)

![](images/946b9fcce03316f3a9e8c2959c3bbca457e4d5d9e6dcf15a2ce8df23117a6257.jpg)

![](images/f36b19f78dba133450099348121e8637909398f4efa8a9f9fd5209daa71fc5e7.jpg)

![](images/1a7ba8eb3ec69b9859e55f73f2d270c801682b14f7b402c50dcc77d367179f1c.jpg)

![](images/0e381d1ed8d60888c3033831079492776514123109a4df55e672cdfbe3ad7d9d.jpg)

![](images/151695c59817c89ea724ccc3046ed08142be50adc464f7fad00fd622a6262b42.jpg)

![](images/abfa4ba01aeb7dd822ae40f707a0ac0fec46585ddca00a39503b5528e490db7f.jpg)  
Figure 7.2: 1D Rare Event. Representative samples on test functions chosen to illustrate MCMC failures caused by mode trapping. Each row shows one test function: reference histogram, MCMC histogram, model histogram, and CDF overlay.

![](images/51435efcc62842740cbfe73af17154cc721e09adb3b1c50c19a73237608a7e56.jpg)  
Figure 7.3: 1D Rare Event. Mean Sinkhorn divergence vs. elapsed time. Star: operator model. Circles: MCMC at increasing time budgets.

## 7.2 2D Constant Noise

This experiment compares grid DeepONet and Lagrangian sensor models for sampling invariant measures in 2D.

We consider amortized sampling from the invariant measures of the 2D SDE

$$
d X = b ( X ) d t + \sqrt { 2 } d W , \qquad X \in \mathbb { R } ^ { 2 } ,\tag{7.3}
$$

with constant isotropic difusion. The drift is generated as $b ( x ) = b _ { \mathrm { G R F } } ( x ) + b _ { \mathrm { w e l l } } ( x )$ , where b<sub>GRF</sub> is a Gaussian random field sampled on a $3 2 \times 3 2 ~ \mathrm { g r i d }$ over $[ - 5 , 5 ] ^ { 2 }$ (RBF kernel with length scale sampled from Unif(0.3, 2.0)), and $b _ { \mathrm { w e l l } }$ is a radial confining well that pushes particles back toward the origin outside radius 2 with strength 10.

Table 7.3 compares three parts of the sampler. The representation determines which drift observations the encoder receives. Grid DeepONet [14] reads the full drift grid. Point probes retain the first observation of each trajectory. Endpoint probes retain the first and last observations and their diference. These two encoders use MLPs, while the trajectory encoder applies a CNN to all observations.

The aggregator combines sensor embeddings into the context z. MeanPool averages the embeddings before an MLP. DeepSets [38] applies a shared MLP to each embedding, sums the outputs, and applies another MLP. Perceiver [13] uses cross attention between learned latent tokens and sensor embeddings, as described in Section 4.2.

Conditioning determines how the decoder uses the context. FiLM (feature-wise linear modulation) [28] uses the context to predict a scale and shift for each hidden feature. Our FiLM and AdaLN (adaptive layer normalization) [27] variants both apply these changes after layer normalization. Our AdaLN variant also uses the context to scale each residual update. Additional variants and training details are in Appendix B.2.

It turns out that the trajectory + Perceiver + AdaLN architecture is a practical choice with competitive accuracy. In the meanwhile, this architecure supports fixed or mixed counts in training and diferent counts at inference without retraining.

## 7.3 64D Interacting Particle

64D Interacting Particle is a proof of concept for sampling that depends on input in an SDE in high dimensions that does not use a grid. The central question is whether the learned sampler uses observations of an unknown pair interaction law to generate physically structured samples in a 64-dimensional particle system, beyond matching the one body scale imposed by the confining potential.

Table 7.3: 2D Constant Noise. Mean Sinkhorn divergence on 100 test drift fields, using 4,096 samples per distribution per field, after training on 65,536 fields. Probe models use 256 sensors, with trajectory + Perceiver + AdaLN as the shared baseline. Time: total training time under shared optimization settings.
<table><tr><td>Component</td><td>Variant</td><td>Mean Sinkhorn</td><td>Train (s)</td></tr><tr><td rowspan="4">Representation</td><td>Grid DeepONet</td><td>0.0209</td><td>40.3</td></tr><tr><td>Point + Perceiver + AdaLN</td><td>0.0415</td><td>87.8</td></tr><tr><td>Endpoint + Perceiver + AdaLN</td><td>0.0324</td><td>82.8</td></tr><tr><td> $\mathrm { T r a j e c t o r y + P e r c e i v e r + A d a L N }$ </td><td>0.0180</td><td>252.0</td></tr><tr><td rowspan="2">Aggregator</td><td> $\mathrm { T r a j e c t o r y + M e a n P o o l + A d a L N }$ </td><td>0.0370</td><td>226.2</td></tr><tr><td> $\mathrm { T r a j e c t o r y + D e e p S e t s + A d a L N }$ </td><td>0.0246</td><td>238.4</td></tr><tr><td>Conditioning</td><td> $\mathrm { T r a j e c t o r y + P e r c e i v e r + F i L M }$ </td><td>0.0169</td><td>258.9</td></tr></table>

Table 7.4: 2D Constant Noise. Sensor resolution transfer for the trajectory + Perceiver + AdaLN backbone after training on 65,536 functions. Rows denote the training sensor regime and columns denote the evaluation sensor count. The mixed row alternates homogeneous batches from the 512-, 1024-, and 2048-sensor trajectory views. Each probe executes a 20- step random walk with step size 0.05. Lower Sinkhorn is better.
<table><tr><td>Train regime</td><td>Eval 512</td><td>Eval 1024</td><td>Eval 2048</td></tr><tr><td>Fixed 512</td><td>0.0284</td><td>0.0172</td><td>0.0131</td></tr><tr><td>Fixed 1024</td><td>0.0287</td><td>0.0175</td><td>0.0125</td></tr><tr><td>Fixed 2048</td><td>0.0312</td><td>0.0184</td><td>0.0131</td></tr><tr><td>Mixed 512/1024/2048</td><td>0.0271</td><td>0.0177</td><td>0.0125</td></tr></table>

We evaluate the framework on amortized sampling for $N = 3 2$ interacting particles in 2D $( X \in \mathbb { R } ^ { 6 4 } )$ under overdamped Langevin dynamics

$$
d X = - \nabla V ( X ) d t + \sqrt { 2 \beta ^ { - 1 } } d W ,\tag{7.4}
$$

targeting the Boltzmann distribution $\pi ( X ) \propto \exp ( - \beta V ( X ) )$ . The potential is

$$
V ( X ) = \frac { \kappa } { 2 } \sum _ { i = 1 } ^ { N } | x _ { i } | ^ { 2 } + \sum _ { i < j } U _ { \mathrm { i n t } } ( | x _ { i } - x _ { j } | ) ,\tag{7.5}
$$

with a fixed harmonic trap $( \kappa = 0 . 5 )$ and inverse temperature $\beta = 2 . 0 ~ ( T = 0 . 5 )$ . Across problem instances, the interaction kernel $U _ { \mathrm { i n t } }$ varies; the one body term remains fixed.

We draw $U _ { \mathrm { i n t } }$ from four qualitatively distinct kernel families—Morse (bondlike attraction with a repulsive core), Gaussian (soft repulsion or attraction), double Gaussian (short range repulsion with long range attraction, creating a preferred interparticle spacing), and WCA (hard sphere exclusion). All kernels are nonsingular; their functional forms and the energy capping convention are given in Table B.3 in Appendix B.3.

Because $U _ { \mathrm { i n t } }$ acts on pairwise distances, full system probes mix many interactions at once. The sensor design therefore uses pair probe sensors: isolated trajectories of particle pairs that record the relative displacement and interaction force along short rollouts. The resulting observation tensor is a fingerprint of the interaction kernel that does not require an Eulerian grid in $\mathbb { R } ^ { 6 4 }$ . The model combines a 1D CNN sensor encoder, a Perceiver aggregator with four latent tokens, and an equivariant particle flow network conditioned by AdaLN; it has 1.78M parameters and is trained by conditional flow matching on 2,048 kernels, with metrics reported on 100 test kernels (sensor, data generation, and training details in Appendix B.3).

The state $X = ( x _ { 1 } , \dots , x _ { 3 2 } ) \in \mathbb { R } ^ { 6 4 }$ is invariant under permutations of the indistinguishable particles. We therefore evaluate the generated distributions using physical summaries invariant under particle permutations: pair correlation functions, energy distributions, center of mass and radius of gyration statistics, pair distance summaries, and nearest neighbor summaries. Raw coordinate space Sinkhorn divergence is reported only as a secondary diagnostic. In 64 dimensions it can be misleading because the fixed harmonic trap makes a pooled Gaussian baseline competitive in coordinate cost even though it contains no interaction structure.

Table 7.5 presents the quantitative evidence that our model has real prediction power over null. When the input is the correct sensors, the generated samples’ distribution is far more closer to the ground truth then shufled sensors input. And the generated sample also outperforms the pooled Gaussian (fitted to the target coordinate mean and variance), and standard Gaussian samples.

Table 7.5: 64D Interacting Particle. Conditioning diagnostic on 100 test interaction kernels. The score is sliced 2-Wasserstein on standardized physical features invariant under permutations. Lower is better.
<table><tr><td>Method</td><td>Mean</td><td>Median</td><td>Comparison</td></tr><tr><td>Correct sensors</td><td>1.7860</td><td>1.0159</td><td>Model conditioned on the true kernel</td></tr><tr><td>Shuffled sensors</td><td>30.1422</td><td>15.9140</td><td>Correct wins 99/100</td></tr><tr><td>Pooled Gaussian</td><td>23.1144</td><td>13.9604</td><td>Correct wins 99/100</td></tr><tr><td>Random normal</td><td>20.0154</td><td>12.3747</td><td>Correct wins 98/100</td></tr></table>

We can also observe that the generated samples reproduce the pair distance patterns within each family. The energy distribution, pooled across all test kernels, also has a similar shape to the target distribution. More quantitative results are collected in Table B.4 in Appendix B.3. Figure 7.4 plots the empirical pair correlation function g(r) broken down by kernel family, and Figure 7.5 compares energy histograms.

![](images/2466cc038d0a8334fac35205d42fd09993c98aed4fb42ac74a862726b4ebdee7.jpg)

![](images/16286726c023659a5725398c7e8e08e51f220cc1f00c61238d689fb60e4fa6c8.jpg)

![](images/254d3d4a57b059ae57f39ec6177b6624ecc1a1142462628ba35dda7acae09ddf.jpg)

![](images/7f9fa65e9bab0582eb23b470e54cea2860247a0d13cea51ed0d9020966d6e342.jpg)  
Figure 7.4: 64D Interacting Particle. Pair correlation function $g ( r )$ by kernel family: target (blue) vs. generated (dashed orange). Each subplot pools test kernels of one family. The curves are empirical histograms without additional smoothing.

![](images/8ce8380a7a53e41f8bc35df60b59b1745b61abd2271fb0a7140849eb21f66eed.jpg)  
Figure 7.5: 64D Interacting Particle. Energy histogram comparison: target (blue) vs. generated (orange).

Taken together, the random Lagrangian sensor architecture shows the potential to func-

tion as an amortized sampler of parametric families of high-dimensional SDEs.

## 8 Conclusion

We present an amortized neural sampler that combines operator learning with continuous normalizing flows to sample from the invariant measures of parametric families of SDEs. The framework encodes SDE coeficients into a latent representation, then conditions a continuous normalizing flow that transports a Gaussian distribution to a target distribution. In practice, we adopt flow matching to train the model. Unlike the established DeepONet, our architecture ingests random Lagrangian trajectory sensors and uses an attention mechanism, which avoids grid discretization in spaces of high dimension. On the theory side, we establish a reduction theorem showing that the UAP of a neural operator implies the expressivity of a neural operator sampler. Thus, given a suficiently powerful neural operator, our framework can fit the map from SDE coeficients to velocity fields that generate the invariant measures. We also prove that our framework is resolution invariant: the sampler remains consistent as the sensor count tends to infinity.

The 1D Variable Noise and 1D Rare Event experiments show the expected regime dependence: MCMC remains stronger in settings with easy, fast mixing; the amortized sampler becomes useful once slow mixing dominates the compute budget. In 2D Constant Noise, the model with random Lagrangian sensors materially outperforms DeepONet. In 64D Interacting Particle, the sampler demonstrates its capability in high dimensions.

Several directions remain open. First, the supervised results rely on access to training samples from target invariant measures, which may require substantial ofline simulation efort, and the amortized approach is most useful when many related SDE instances must be solved and mixing is slow. Data preparation, unsupervised training, and training without samples need further investigation. Second, the current framework targets the invariant measure, and a natural extension is to learn the full law $\mu _ { t }$ that depends on time, or the weak solution of the SDE. Lastly, our experiments in higher dimensions are only a proof of concept. The network structure and training procedure have not been extensively optimized. Scaling to larger particle counts and configuration spaces of higher dimension remains an important challenge.

## A Proofs of Main Results

## A.1 Stationary Fokker-Planck measures and stability

Proof of Proposition 5.2. Fix $a = ( b , \sigma ) \in \mathcal { A } ( \nu , p _ { 0 } , \lambda , M )$ , write $A = A _ { a } .$ , and let $\mu _ { a }$ be the corresponding stationary measure. Existence and uniqueness follow by the standard result as commented in the text below the statement. Here, we only show how the moment bound follows by the Lyapunov condition.

For the moment bound, let $V _ { p } ( x ) = 1 + | x | ^ { p } , 2 \leq p \leq p _ { 0 }$ . For $x \neq 0$ , the standard calculation gives

$$
\begin{array} { r l r } {  { \mathcal L _ { a } | x | ^ { p } \le p | x | ^ { p - 2 } ( \langle b ( x ) , x \rangle + \frac 1 2 ( p - 1 ) \operatorname { T r } A ( x ) ) } } \\ & { } & \\ & { } & { \le p | x | ^ { p - 2 } ( \langle b ( x ) , x \rangle + \frac 1 2 ( p _ { 0 } - 1 ) \operatorname { T r } A ( x ) ) } \\ & { } & \\ & { } & { \le p | x | ^ { p - 2 } ( - \lambda | x | ^ { 2 } + M ) \le - c _ { p } | x | ^ { p } + C _ { p } , } \end{array}
$$

where $c _ { p } , C _ { p } > 0$ are independent of $a .$ The nonsmooth point at $x = 0$ is handled either by smoothing $V _ { p }$ near the origin or by applying the following cutof argument to $1 + ( | x | ^ { 2 } + \varepsilon ) ^ { p / 2 }$ and then letting $\varepsilon \downarrow 0$

Let $\theta _ { n }$ be increasing, concave, and bounded, with

$$
0 \leq \theta _ { n } ^ { \prime } \leq 1 , \qquad \theta _ { n } ^ { \prime \prime } \leq 0 ,
$$

and with $\theta _ { n } ^ { \prime } ( r ) \uparrow 1 , \theta _ { n } ( r )$ ↑ r on compact intervals. Then

$$
\begin{array} { r l } & { \mathcal { L } _ { a } \theta _ { n } ( V _ { p } ) = \theta _ { n } ^ { \prime } ( V _ { p } ) \mathcal { L } _ { a } V _ { p } + \displaystyle \frac { 1 } { 2 } \theta _ { n } ^ { \prime \prime } ( V _ { p } ) \langle A \nabla V _ { p } , \nabla V _ { p } \rangle _ { \mathbb { R } ^ { d } } } \\ & { \quad \quad \quad \leq \theta _ { n } ^ { \prime } ( V _ { p } ) ( - c _ { p } V _ { p } + C _ { p } ) . } \end{array}
$$

Integrating against $\mu _ { a }$ , using stationarity, and then applying monotone convergence gives

$$
c _ { p } \int V _ { p } d \mu _ { a } \leq C _ { p } ,
$$

uniformly over $a \in \mathcal { A } ( \nu , p _ { 0 } , \lambda , M )$ , which proves the claim.

Proof of Lemma 5.1. Let $a _ { n } \in \mathcal { A } ( \nu , p _ { 0 } , \lambda , M )$ and $a _ { n } \to a$ locally uniformly. Write $A _ { n } =$ $A _ { a _ { n } }$ and $A = A _ { a }$ . On every compact set, $b _ { n } \to b$ and $A _ { n } \ \to \ A$ uniformly, as noted in Section 5. The two conditions that define $\mathcal { A } ( \nu , p _ { 0 } , \lambda , M )$ are closed conditions on the values $( b ( x ) , A _ { a } ( x ) )$ at each fixed x. Hence they pass to the limit, and $a \in \mathcal { A } ( \nu , p _ { 0 } , \lambda , M )$ . This proves closedness.

Write $\mu _ { n } = \mu _ { a _ { n } }$ and $\mu = \mu _ { a }$ . By the Lyapunov condition, $\int | x | ^ { p } d \mu _ { n }$ is uniformly bounded. This gives tightness. Take an arbitrary subsequence of $\{ \mu _ { n } \}$ . By Prokhorov’s theorem it has a further subsequence, not relabeled, such that $\mu _ { n } \implies \nu$ weakly for some probability measure ν.

Let $\varphi \in C _ { c } ^ { \infty } (  { \mathbb { R } } ^ { d } )$ . Since $\mu _ { n }$ is invariant,

$$
\int \mathcal { L } _ { a _ { n } } \varphi \mathop { } { d \mu _ { n } } = 0 .
$$

On the compact support of $\varphi , b _ { n } \to b$ and $A _ { n } \to A$ uniformly. Therefore

$$
{ \mathcal { L } } _ { a _ { n } } \varphi = b _ { n } \cdot \nabla \varphi + { \frac { 1 } { 2 } } \operatorname { T r } ( A _ { n } D ^ { 2 } \varphi ) \longrightarrow { \mathcal { L } } _ { a } \varphi
$$

uniformly. Taking the limit $n  \infty ,$ , one then has

$$
\int \mathcal { L } _ { a \varphi d \nu } = 0 \qquad \forall \varphi \in C _ { c } ^ { \infty } ( \mathbb { R } ^ { d } ) .
$$

By the density argument, this can be extended to $C _ { b } ^ { 2 }$ test functions. Hence, ν is a invariant probability measure for the limiting SDE. The invariant measure is unique, so $\nu = \mu$ . Since every subsequence has a further subsequence converging weakly to $\mu ,$ the full sequence satisfies $\mu _ { n } \Rightarrow \mu$

By the uniform p-moments, one has the uniform integrability of second moments, or

$$
\operatorname * { l i m } _ { R  \infty } \operatorname* { s u p } _ { n } \int _ { | x | \geq R } | x | ^ { 2 } \mu _ { n } ( d x ) \leq \operatorname * { l i m } _ { R  \infty } R ^ { - ( p _ { 0 } - 2 ) } M _ { p _ { 0 } } = 0 .
$$

Then, this together with the weak convergence implies that $\begin{array} { r } { \int | x | ^ { 2 } d \mu _ { n } \to \int | x | ^ { 2 } d \mu } \end{array}$ . Hence $W _ { 2 } ( \mu _ { n } , \mu )  0 .$ , proving continuity of $a \mapsto \mu _ { a }$ □

## A.2 Proofs for the canonical target velocity

We use the notation $\mu _ { a } ^ { S } , \rho _ { t } ^ { a } , J _ { t } ^ { a } , v ^ { a } , \chi _ { R } , v ^ { a , R }$ , and $\mu _ { a } ^ { S , R }$ introduced in Section 5.

Proof of Lemma 5.2. First record the uniform moment input. Since $X \sim \gamma _ { \alpha } = { \mathcal { N } } ( 0 , ( 1 -$ $\alpha ) I _ { d } )$ . Let $D = Y - X$ . One has

$$
| D | ^ { p } \leq 2 ^ { p - 1 } ( | X | ^ { p } + | Y | ^ { p } ) .
$$

Hence, one has sup $\mathbb { E } | D | ^ { p } < \infty$ by the uniform control of $\mathbb { E } | Y | ^ { p }$ . By the conditional expectation formula and Jensen’s inequality:

$$
\int | v ^ { a } ( z , t ) | ^ { p } d \rho _ { t } ^ { a } ( z ) = \mathbb { E } | \mathbb { E } [ D \mid Z _ { t } ^ { a } ] | ^ { p } \leq \mathbb { E } | D | ^ { p } \leq C _ { \alpha , S , p } ,
$$

where the last bound follows from the exponential moment bound for $D$

It remains to justify the pointwise linear growth estimate. Since $X \mid ( Z _ { t } = z , Y = y )$ is Gaussian with mean

$$
m _ { t , y , z } = \frac { ( 1 - t ) ( 1 - \alpha ) } { ( 1 - t ) ^ { 2 } ( 1 - \alpha ) + \alpha } ( z - t y ) ,
$$

one finds by the boundedness $| Y | \le S$ that

$$
\begin{array} { r } { | v ^ { a } ( z , t ) | \leq \mathbb { E } [ | Y | + | X | \mid Z _ { t } ^ { a } = z ] \leq \mathbb { E } _ { Y | Z _ { t } ^ { a } = z } \mathbb { E } [ | X | + | Y | \mid Z _ { t } ^ { a } = z , Y ] \leq C _ { \alpha , S } ( 1 + | z | ) . } \end{array}
$$

Lastly, diferentiating the Gaussian kernel in $t ,$ one has

$$
\begin{array} { r } { \partial _ { t } \phi _ { \alpha } \big ( z - \big ( ( 1 - t ) x + t y \big ) \big ) + \nabla _ { z } \cdot \big ( ( y - x ) \phi _ { \alpha } \big ( z - \big ( ( 1 - t ) x + t y \big ) \big ) \big ) = 0 . } \end{array}
$$

Integrating this identity against $\gamma _ { \alpha } ( d x ) \mu _ { a } ^ { S } ( d y )$ , justified by dominated convergence using the Gaussian kernel and the moment bounds above, gives the continuity equation. □

Proof of Lemma 5.3. We have shown that if $a _ { n } \to a$ , then $\mu _ { a _ { n } } ^ { S }  \mu _ { a } ^ { S }$ in $W _ { 2 }$ . Write $\nu _ { n } = \mu _ { a _ { r } } ^ { S }$ and $\nu = \mu _ { a } ^ { S }$ . These measures are all supported in $\overline { { B } } _ { S }$ . Note that $\rho , J , \nabla _ { z } \rho , \nabla _ { z } J$ ar obtained by integrating $\phi _ { \alpha } , ( \boldsymbol { y } - \boldsymbol { x } ) \phi _ { \alpha }$ and their z-derivatives with respect to $\gamma _ { \alpha } ( d x ) \mu _ { a } ^ { S } ( d y )$ These kernels are continuous in $( x , y , z , t )$ , while $\gamma _ { \alpha }$ is a Gaussian and $\mu _ { a } ^ { S }$ is finitely supported. Hence they are dominated by an integrable function of $x ,$ , uniformly over $y \in \overline { { B } } _ { S }$ and $( z , t ) \in \overline { { B } } _ { 2 R } \times [ 0 , 1 ]$ . Weak convergence of $\nu _ { n }$ gives pointwise convergence of the corresponding integrals. Uniform continuity of the kernels on

$$
\{ | x | \leq M , \ | y | \leq S , \ | z | \leq 2 R , \ t \in [ 0 , 1 ] \}
$$

plus the uniformly small Gaussian x-tail outside $\{ | x | \leq M \}$ upgrades this to uniform convergence on $\overline { { B } } _ { 2 R } \times [ 0 , 1 ]$

There is also a uniform positive lower bound for $\rho ^ { a }$ . Indeed, for $| z | \le 2 R , y \in \overline { { B } } _ { S }$ $t \in [ 0 , 1 ]$ , and $| x | \le 1$

$$
| z - ( ( 1 - t ) x + t y ) | \leq 2 R + 1 + S .
$$

Therefore

$$
\rho _ { t } ^ { a } ( z ) \geq \gamma _ { \alpha } ( B _ { 1 } ) ( 2 \pi \alpha ) ^ { - d / 2 } \exp \left( - \frac { ( 2 R + 1 + S ) ^ { 2 } } { 2 \alpha } \right) > 0 ,
$$

uniformly in $\nu , z ,$ and t. The quotient formula

$$
\nabla _ { z } v ^ { a } = \frac { \rho ^ { a } \nabla _ { z } J ^ { a } - J ^ { a } \otimes \nabla _ { z } \rho ^ { a } } { ( \rho ^ { a } ) ^ { 2 } }
$$

therefore gives uniform convergence of the velocities and their first z-derivatives on the compact cylinder.

Recall that $v ^ { a , R } ( x , t ) = \chi _ { R } ( x ) v ^ { a } ( x , t )$ for $a \in \mathcal { A } ( \nu , p _ { 0 } , \lambda , M )$ . Multiplication by the fixed smooth cutof preserves continuity in a. It also yields a global Lipschitz bound: on $\overline { { B } } _ { 2 R }$ the derivative of $\chi _ { R } v ^ { a }$ is uniformly bounded, while outside $\overline { { B } } _ { 2 R }$ the field is identically zero. Hence $L _ { \alpha , S , R } < \infty$ □

Proof of Lemma 5.4. Let $X _ { t } ^ { a }$ solve the untruncated ODE

$$
\dot { X } _ { t } ^ { a } = v ^ { a } ( X _ { t } ^ { a } , t ) , \qquad X _ { 0 } ^ { a } \sim \gamma ,
$$

and let $X _ { t } ^ { a , R }$ solve the truncated ODE

$$
\dot { X } _ { t } ^ { a , R } = v ^ { a , R } ( X _ { t } ^ { a , R } , t ) , \qquad X _ { 0 } ^ { a , R } = X _ { 0 } ^ { a } .
$$

The untruncated terminal law is $\rho _ { 1 } ^ { a }$ , and the truncated terminal law is $\mu _ { a } ^ { S , R }$

Define

$$
\tau _ { R } ^ { a } = \operatorname* { i n f } \{ t \in [ 0 , 1 ] : | X _ { t } ^ { a } | > R \} .
$$

On $\{ \tau _ { R } ^ { a } > 1 \} , X _ { 1 } ^ { a , R } = X _ { 1 } ^ { a } . \mathrm { ~ I f ~ } \tau _ { R } ^ { a } \leq 1$ , then either $\vert X _ { 0 } ^ { a } \vert > R / 2$ or

$$
\int _ { 0 } ^ { 1 } | v ^ { a } ( X _ { t } ^ { a } , t ) | d t > R / 2 .
$$

Since $X _ { t } ^ { a }$ has law $\rho _ { t } ^ { a }$ , the velocity moment bound gives

$$
\operatorname* { s u p } _ { a \in \mathcal { A } ( \nu , p _ { 0 } , \lambda , M ) } \mathbb { E } \int _ { 0 } ^ { 1 } \vert v ^ { a } ( X _ { t } ^ { a } , t ) \vert ^ { p } d t \leq C _ { \alpha , S , p } .
$$

Markov’s inequality and the moments of $X _ { 0 } ^ { a }$ therefore give

$$
\operatorname* { s u p } _ { a \in \mathcal { A } ( \nu , p _ { 0 } , \lambda , M ) } \mathbb { P } ( \tau _ { R } ^ { a } \leq 1 ) \leq C _ { \alpha , S , p } R ^ { - p }
$$

for every $p > 2 .$

Moreover, the linear growth bound and Gronwall’s inequality give uniform p-moment bounds for $\dot { X _ { 1 } ^ { a , R } }$ and $X _ { 1 } ^ { a }$ . Using the coupling with a common initial point,

$$
W _ { 2 } ^ { 2 } ( \rho _ { 1 } ^ { a } , \mu _ { a } ^ { S , R } ) \leq \mathbb { E } | X _ { 1 } ^ { a , R } - X _ { 1 } ^ { a } | ^ { 2 } \leq C \mathbb { P } ( \tau _ { R } ^ { a } \leq 1 ) ^ { 1 - 2 / p } \leq C R ^ { - ( p - 2 ) } .
$$

This gives the claim.

## A.3 Reduction and neural approximation

Proof of Proposition 5.3. Couple the two flows by the same initial point $X _ { 0 } \sim \rho _ { 0 } \mathrm { : }$

$$
X _ { t } = \Phi _ { t } ( X _ { 0 } ) , \qquad \hat { X } _ { t } = \hat { \Phi } _ { t } ( X _ { 0 } ) .
$$

Let $\Delta _ { t } = X _ { t } - { \hat { X } } _ { t }$ . Then, one has

$$
\frac { d } { d t } | \Delta _ { t } | ^ { 2 } \leq 2 L | \Delta _ { t } | ^ { 2 } + 2 | \Delta _ { t } | \delta ,
$$

implying that $| X _ { 1 } - \hat { X } _ { 1 } | \leq e ^ { L } \delta$ almost surely. The joint law of $( X _ { 1 } , { \hat { X } } _ { 1 } )$ is a coupling of the two terminal distributions, so

$$
W _ { 2 } ( ( \Phi _ { 1 } ) _ { \# } \rho _ { 0 } , ( \hat { \Phi } _ { 1 } ) _ { \# } \rho _ { 0 } ) \leq \left( \mathbb { E } | X _ { 1 } - \hat { X } _ { 1 } | ^ { 2 } \right) ^ { 1 / 2 } \leq e ^ { L } \delta .
$$

Proof of Proposition 5.4. Fix $a \in K$ . Let $X _ { t }$ solve the target truncated ODE

$$
\dot { X } _ { t } = v ^ { a , R } ( X _ { t } , t ) , \qquad X _ { 0 } \sim \gamma ,
$$

and let $\widehat { X } _ { t }$ solve the exact learned ODE

$$
\dot { \widehat { X _ { t } } } = \widehat { v } _ { \theta } [ a ] ( \widehat { X } _ { t } , t ) , \qquad \widehat { X } _ { 0 } = X _ { 0 } .
$$

Put $\beta = e ^ { L _ { \alpha , S , R } } \delta _ { \theta , \alpha , S , R }$ . By assumption, $\beta < R .$ . Define

$$
\tau = \operatorname* { i n f } \{ t \in [ 0 , 1 ] : | { \widehat { X } } _ { t } | \geq 2 R \} .
$$

On $[ 0 , \tau )$ , the learned path remains in $B _ { 2 R }$ , where the compact domain approximation bound applies. The same Gr¨onwall estimate as in Proposition 5.3, stopped at τ , gives

$$
| X _ { t } - { \widehat { X } } _ { t } | \leq \beta , \qquad t \leq \tau .
$$

If $\tau \leq 1$ , then $| \widehat { X } _ { \tau } | = 2 R$ , and hence $| X _ { \tau } | \geq 2 R - \beta > R$ . Therefore

$$
\{ \tau \leq 1 \} \subset \left\{ \operatorname* { s u p } _ { t \leq 1 } | X _ { t } | \geq R \right\} .
$$

The target field satisfies $| v ^ { a , R } ( x , t ) | \leq C _ { \alpha , S } ( 1 + | x | )$ , and the learned field satisfies the assumed global envelope. Gr¨onwall’s inequality and Gaussian moments of $X _ { 0 }$ give the uniform path moment bounds

$$
\operatorname* { s u p } _ { a \in K } \mathbb { E } \operatorname* { s u p } _ { t \leq 1 } | X _ { t } | ^ { p } + \operatorname* { s u p } _ { a \in K } \mathbb { E } \operatorname* { s u p } _ { t \leq 1 } | { \widehat { X } } _ { t } | ^ { p } \leq C _ { \alpha , S , p } .
$$

Hence $\mathbb { P } ( \tau \le 1 ) \le C _ { \alpha , S , p } R ^ { - p }$ , and H¨older’s inequality gives

$$
\begin{array} { r l } & { \mathbb { E } \left[ | X _ { 1 } - \widehat { X } _ { 1 } | ^ { 2 } \mathbf { 1 } _ { \left\{ \tau \leq 1 \right\} } \right] \leq \left( \mathbb { E } ( | X _ { 1 } | + | \widehat { X } _ { 1 } | ) ^ { p } \right) ^ { 2 / p } \mathbb { P } ( \tau \leq 1 ) ^ { 1 - 2 / p } } \\ & { \qquad \leq C _ { \alpha , S , p } R ^ { - ( p - 2 ) } . } \end{array}
$$

On $\{ \tau > 1 \}$ , the stopped estimate gives $| X _ { 1 } - \widehat { X } _ { 1 } | \leq \beta$ . Thus

$$
W _ { 2 } ( \hat { \mu } _ { a , \theta } , \mu _ { a } ^ { S , R } ) \leq e ^ { L _ { \alpha , S , R } } \delta _ { \theta , \alpha , S , R } + C _ { \alpha , S , p } R ^ { - ( p - 2 ) / 2 } .
$$

The triangle inequality, the truncation definition, the smoothing coupling, and the projection estimate give

$$
\begin{array} { l } { { W _ { 2 } ( \hat { \mu } _ { a , \theta } , \mu _ { a } ) \leq e ^ { L _ { \alpha , S , R } } \delta _ { \theta , \alpha , S , R } + C _ { \alpha , S , p } R ^ { - ( p - 2 ) / 2 } + W _ { 2 } ( \mu _ { a } ^ { S , R } , \rho _ { 1 } ^ { a } ) } } \\ { { \qquad + W _ { 2 } ( \rho _ { 1 } ^ { a } , \mu _ { a } ^ { S } ) + W _ { 2 } ( \mu _ { a } ^ { S } , \mu _ { a } ) } } \\ { { \leq e ^ { L _ { \alpha , S , R } } \delta _ { \theta , \alpha , S , R } + C _ { \alpha , S , p } R ^ { - ( p - 2 ) / 2 } + \varepsilon _ { \mathrm { t r u n } } ( \alpha , S , R ) + \sqrt { \alpha d } + M _ { p _ { 0 } } ^ { 1 / 2 } S ^ { - ( p _ { 0 } - 2 ) / 2 } . } } \end{array}
$$

Taking the supremum over $a \in K$ proves the theorem.

Proof of Theorem 5.1. Fix $\eta > 0$ and $p > 2 .$ . Choose $S < \infty$ so large that

$$
M _ { p _ { 0 } } ^ { 1 / 2 } S ^ { - ( p _ { 0 } - 2 ) / 2 } < \eta / 5 .
$$

Choose $\alpha \in ( 0 , 1 )$ so that $\sqrt { \alpha d } < \eta / 5$ . For this $( \alpha , S )$ , choose $R > 0$ so large that

$$
\varepsilon _ { \mathrm { t r u n } } ( \alpha , S , R ) < \eta / 5 , \qquad C _ { \alpha , S , p } R ^ { - ( p - 2 ) / 2 } < \eta / 5 .
$$

For the fixed triple $( \alpha , S , R )$ , Lemma 5.3 shows that $a \mapsto v ^ { a , R }$ is continuous on $K .$ . The operator $\Psi ( a ) = v ^ { a , \bar { R } }$ is continuous from K to $C ( \overline { { B } } _ { 2 R } \times [ 0 , 1 ] ; \mathbb { R } ^ { d } )$ . It also satisfies $| \Psi ( a ) ( x , t ) | \leq$ $C _ { \alpha , S } ( 1 + | x | )$ . Apply Assumption 5.1 with $C _ { \mathrm { e n v } } = C _ { \alpha , S }$ , and choose θ such that

$$
\delta _ { \theta , \alpha , S , R } < \operatorname* { m i n } \left\{ R e ^ { - L _ { \alpha , S , R } } , \frac { \eta } { 5 e ^ { L _ { \alpha , S , R } } } \right\} .
$$

Proposition 5.4 gives

$$
\operatorname* { s u p } _ { a \in K } W _ { 2 } ( \hat { \mu } _ { a , \theta } , \mu _ { a } ) < \eta .
$$

Proof of Proposition 5.5. Let $\Psi : K  C ( Y ; \mathbb { R } ^ { d } )$ be a continuous target operator as in Assumption 5.1, with $| \Psi ( a ) ( x , t ) | \leq C _ { \mathrm { e n v } } ( 1 + | x | )$ . Fix $\delta > 0$

By compactness of K, choose $\varepsilon _ { \Psi } ~ > ~ 0$ such that $d _ { \mathcal { X } _ { a } } ( a , a ^ { \prime } ) ~ < ~ \varepsilon _ { \Psi }$ implies $\parallel \Psi ( a ) -$ $\Psi ( a ^ { \prime } ) \| _ { C ( Y ) } \ < \ \delta / 3$ . Apply (FS) with this $\varepsilon _ { \Psi }$ and write $\begin{array} { r } { S \ = \ S _ { m , L } ; } \end{array}$ the image ${ \mathcal { S } } ( K )$ is compact. Cover it by finitely many balls $B ( S ( a _ { i } ) , r / 2 ) , i = 1 , \ldots , N .$ , and take a continuous partition of unity $\{ \chi _ { i } \} _ { i = 1 } ^ { N }$ subordinate to the enlarged balls $B ( S ( a _ { i } ) , r )$ . Define

$$
\bar { G } ( s ) ( y ) = \sum _ { i = 1 } ^ { N } \chi _ { i } ( s ) \Psi ( a _ { i } ) ( y ) , \qquad s \in S ( K ) , \ y = ( x , t ) \in Y .
$$

$\mathrm { I f } \ \chi _ { i } ( S ( a ) ) > 0$ , then $d _ { \mathrm { s e n s } } ( S ( a ) , S ( a _ { i } ) ) < r ,$ so (FS) gives $d _ { \mathcal { X } _ { a } } ( a , a _ { i } ) < \varepsilon _ { \Psi } ;$ hence

$$
\operatorname* { s u p } _ { a \in K } \| \bar { G } ( S ( a ) ) - \Psi ( a ) \| _ { C ( Y ) } < \delta / 3 .
$$

The map $( s , y ) \mapsto { \bar { G } } ( s ) ( y )$ is continuous on $S ( K ) \times Y$ and invariant under permutations of the sensor argument. By (EU), the finite sensor Perceiver–AdaLN network approximates it within $\delta / 3$ , uniformly over $a \in K$ and $y \in Y$ . Combining the two estimates gives

$$
\operatorname* { s u p } _ { a \in K ( x , t ) \in Y } | \mathcal { N } _ { \theta } ( a ) ( x , t ) - \Psi ( a ) ( x , t ) | < \delta ,
$$

the compact domain approximation part of Assumption 5.1. The final statement follows by adding the global linear growth clause; local Lipschitzness is already part of (EU). □

Proof of Theorem 5.2. First we note that for general $L \geq 1$ , one clearly has

$$
d _ { \mathrm { s e n s } } ( S _ { m , L } ( \boldsymbol { a } ) , S _ { m , L } ( \boldsymbol { a } ^ { \prime } ) ) \geq d _ { \mathrm { s e n s } } ( S _ { m , 1 } ( \boldsymbol { a } ) , S _ { m , 1 } ( \boldsymbol { a } ^ { \prime } ) ) .
$$

where $S _ { m , 1 } ( a )$ means that we only keep the first point in each trajectory and the corresponding values of a. Then,

$$
\mathcal { S } _ { m , 1 } ( a ) = \{ ( x _ { i } , b ( x _ { i } ) , \mathrm { v e c } ~ \sigma ( \mathrm { x _ { i } } ) ) \} _ { \mathrm { i = 1 } } ^ { \mathrm { m } }
$$

where $( x _ { i } ) \ '$ s are sampled i.i.d from $\rho _ { 0 } ( x )$ dx. Hence,

$$
d _ { \mathrm { s e n s } } ( S _ { m , 1 } ( a ) , S _ { m , 1 } ( a ^ { \prime } ) ) = \operatorname* { m a x } _ { 1 \leq i \leq m } \| a ( x _ { i } ) - a ^ { \prime } ( x _ { i } ) \| .
$$

Hence, it sufices to show with probability $1 - \delta$ such that

$$
\operatorname* { m a x } _ { 1 \leq i \leq m } \| a ( x _ { i } ) - a ^ { \prime } ( x _ { i } ) \| \leq r \quad \Longrightarrow \quad d _ { \mathcal { X } _ { a } } < \varepsilon .
$$

We first take a radius J so large that the tail of the metric $d _ { X _ { a } }$ is smaller than the prescribed tolerance. On the compact ball $B _ { J }$ , the family $K$ is uniformly equicontinuous. Therefore a suficiently fine spacing h gives that, if every point in $B _ { J }$ has some sample point $x _ { i }$ with distance at most $h ,$ then

$$
d _ { \mathscr { X } _ { a } } ( a , a ^ { \prime } ) < \varepsilon .
$$

Hence, it sufices to show with probability $1 - \delta$ that $\{ x _ { i } \} _ { i = 1 } ^ { m }$ forms a h-net of $B _ { J }$

We take N balls with radius $h / 2$ that cover $B _ { J }$ . The number of such balls can be chosen as $N \leq ( C J / h ) ^ { d }$ . If each ball contains at least one $x _ { i } ,$ then $\{ x _ { i } \} _ { i = 1 } ^ { m }$ forms a h-net of $B _ { J }$ For each ball $B _ { \ell } .$ , the probability that it is empty is bounded by

$$
( 1 - q _ { \ell } ) ^ { m } \leq \exp ( - m q _ { \ell } ) ,
$$

where

$$
q _ { \ell } = \int _ { B _ { \ell } } \rho _ { 0 } d x .
$$

Taking union bound, the probability that there is one empty ball is controlled by

$$
p \leq N \operatorname* { s u p } _ { \ell } \exp ( - N q _ { \ell } ) \leq ( C J / h ) ^ { d } \exp ( - m \operatorname* { m i n } _ { \ell } q _ { \ell } ) .
$$

For $p \leq \delta ,$ , one needs

$$
m \gtrsim d ( \operatorname* { m i n } _ { \ell } q _ { \ell } ) ^ { - 1 } .
$$

This shows the first claim.

The remaining claims are straightforward.

## A.4 Resolution invariance proofs

We use the notation of Section 6 for the attention maps $A _ { \ell } ^ { h } , H _ { \ell }$ , the self attention–MLP update $G _ { \ell } .$ , and the encoder Encoder. The finite and infinite sensor encodings are $z _ { a , m }$ and $z _ { a }$

Proof of Lemma 6.1. Write $\begin{array} { r } { \mathcal { K } = K _ { q } \times K _ { e } , } \end{array}$ , compact. Each $\omega _ { \ell } ^ { h } , F _ { \ell } ^ { h }$ is $C ^ { \infty }$ on $\kappa ,$ so

$$
\omega _ { \operatorname* { m i n } } = \operatorname* { m i n } _ { \ell , h , \ K } \omega _ { \ell } ^ { h } , \quad \omega _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { \ell , h , \ K } \omega _ { \ell } ^ { h } , \quad M _ { F } = \operatorname* { m a x } _ { \ell , h , \ K } \| F _ { \ell } ^ { h } \| , \quad L _ { \omega } = \operatorname* { m a x } _ { \ell , h , \ K } \big ( \| \nabla _ { q } \omega _ { \ell } ^ { h } \| + \| \nabla _ { q } F _ { \ell } ^ { h } \| \big )
$$

are finite, with $\omega _ { \mathrm { m i n } } > 0$ since $\omega _ { \ell } ^ { h } ( \boldsymbol { q } , \boldsymbol { e } ) = \exp ( \langle \boldsymbol { q } , \tilde { e } W _ { \ell , h } ^ { K } \rangle / \sqrt { d _ { h } } ) > 0 ;$ this gives (i) and (ii), and integrating the lower bound gives $\textstyle \int \omega _ { \ell } ^ { h } ( q , \cdot ) d \nu \geq \omega _ { \mathrm { m i n } }$ for every probability ν on $K _ { e }$

For (iii), the ε-regularized LayerNorm normalizes each row to Euclidean norm $\leq \sqrt { d _ { z } }$ before its fixed afine map, so its output rows are bounded, $\| \mathrm { L N } ( u ) _ { r } \| \leq \rho _ { \mathrm { L N } } < \infty .$ , and it is Lipschitz, $\mathrm { L i p } ( \mathrm { L N } ) \leq L _ { \mathrm { L N } } < \infty$ (the variance is $\geq \varepsilon )$ . Each $G _ { \ell }$ and each message addition step ends in a LayerNorm, so with $Z ^ { ( 0 ) }$ fixed the recursion (6.2) keeps every latent row at norm $\leq \rho : = \operatorname* { m a x } \{ \rho _ { \mathrm { L N } }$ , max<sub>r</sub> $\| Z _ { r } ^ { ( 0 ) } \| \}$ ; hence

$$
K _ { z } = \{ Z \in \mathbb { R } ^ { k \times d _ { z } } : \ \operatorname* { m a x } _ { r } \| Z _ { r } \| \leq \rho \} , \qquad K _ { q } = \{ z W _ { \ell , h } ^ { Q } : z \in K _ { z } , \ \ell , h \}
$$

are compact and contain all reachable latents and queries. The query map is linear, $L _ { \ell } ^ { Q } =$ max $\mathfrak { _ { l } } \| W _ { \ell , h } ^ { Q } \|$ , and $G _ { \ell } = \mathrm { r } [ \mathrm { M L P } ] \circ \mathrm { r } [ \mathrm { S A } ]$ composes ε-LayerNorms, self attention, and the MLP, Lipschitz on $K _ { z }$ with $L _ { \ell } ^ { G } = \mathrm { L i p } ( G _ { \ell } | _ { K _ { z } } )$ . For $( \mathrm { i v } )$ , Decoder is $C ^ { 1 }$ , so

$$
L _ { \theta , R } ^ { \mathrm { d e c } } = \operatorname* { s u p } _ { K _ { z } \times \overline { { B } } _ { R } \times [ 0 , 1 ] } \| \nabla _ { z } \mathrm { D e c o d e r } \| < \infty .
$$

Proof of Lemma 6.2. By Lemma $6 . 1 ( \mathrm { i i } ) , q \mapsto \omega _ { \ell } ^ { h } ( q , \cdot )$ and $q \mapsto [ F _ { \ell } ^ { h } ( q , \cdot ) ] .$ are Lipschitz from $K _ { q }$ into $( C ( K _ { e } ) , \| \cdot \| _ { \infty } )$ ; as $K _ { q }$ is compact and there are finitely many $( \ell , h , r )$ , F is a finite union of Lipschitz images of compact sets, hence totally bounded in $\| \cdot \| _ { \infty }$ , with envelope $M _ { \theta } < \infty$

Fix $\varepsilon > 0$ and a finite ε-net $\{ f _ { 1 } , \dotsc , f _ { N } \} \subset { \mathfrak { F } } _ { \theta }$ . For every $a \in K$

$$
\operatorname* { s u p } _ { f \in \mathfrak { F } _ { \theta } } \left| \int f d \hat { \nu } _ { a , m } - \int f d \nu _ { a } \right| \leq 2 \varepsilon + \operatorname* { m a x } _ { 1 \leq j \leq N } \left| \int f _ { j } d \hat { \nu } _ { a , m } - \int f _ { j } d \nu _ { a } \right| .
$$

Since $\hat { \nu } _ { a , m }$ is the empirical law of m independent samples from $\nu _ { a }$ and $\| f _ { j } \| _ { \infty } \leq M _ { \theta }$ , each term satisfies

$$
\operatorname* { s u p } _ { a } \mathbb { E } _ { \mathbf { e } } | \int f _ { j } d \hat { \nu } _ { a , m } - \int f _ { j } d \nu _ { a } | \leq M _ { \theta } m ^ { - 1 / 2 } ,
$$

so the maximum over the finite net is bounded by $N M _ { \theta } m ^ { - 1 / 2 }$

Hence lim su $\begin{array} { r } { \operatorname { l p } _ { m  \infty } \delta _ { m } ^ { \mathrm { a t t } } \leq 2 \varepsilon } \end{array}$ , and as $\varepsilon > 0$ is arbitrary, $\delta _ { m } ^ { \mathrm { a t t } } \to 0$

Proof of Theorem 6.1. Note that for the resolution invariance, the parameters and architecture are fixed. We are increasing the number of probes $( \mathrm { i . e . , } m \ )$ so that only the measure ν involved are varying.

Step 1: estimate for the encoder Let

$$
\Delta _ { \ell } : = \operatorname* { s u p } _ { a \in K } \mathbb { E _ { e } } \| Z _ { \ell } ^ { a , m } - Z _ { \ell } ^ { a } \| ,
$$

with $\Delta _ { 0 } = 0$ . We now derive the induction formula for $\Delta _ { \ell }$

For any ℓ, h, we recall that

$$
A _ { \ell } ^ { h } ( q , \nu ) = { \frac { \int F _ { \ell } ^ { h } ( q , \cdot ) d \nu } { \int \omega _ { \ell } ^ { h } ( q , \cdot ) d \nu } } .
$$

By Lemma 6.1,

$$
\int \omega _ { \ell } ^ { h } ( q , \cdot ) d \nu \geq \omega _ { \mathrm { m i n } } , \quad \| \int F _ { \ell } ^ { h } ( q , \cdot ) d \nu \| \leq M _ { F } ,
$$

where the bounds are uniform in $\ell , h .$ . Introducing $\mathfrak { F } _ { \ell , h } = \{ \omega _ { \ell } ^ { h } ( q , \cdot ) : q \in K _ { q } \} \cup \{ [ F _ { \ell } ^ { h } ( q , \cdot ) ] _ { r }$ $q \in K _ { q } , \ r = 1 , \ldots , d _ { h } \}$ , one then has

$$
\| N _ { q } ^ { \nu } - N _ { q } ^ { \bar { \nu } } \| \leq \sqrt { d _ { h } } \operatorname* { s u p } _ { f \in \tilde { \mathfrak { F } } _ { \varepsilon , h } } | \int f d \nu - \int f d \tilde { \nu } | , \quad | D _ { q } ^ { \nu } - D _ { q } ^ { \bar { \nu } } | \leq \operatorname* { s u p } _ { f \in \tilde { \mathfrak { F } } _ { \varepsilon , h } } | \int f d \nu - \int f d \tilde { \nu } | .
$$

Hence, the change of measure ν will bring in error as follows

$$
\| A _ { \ell } ^ { h } ( q , \nu ) - A _ { \ell } ^ { h } ( q , \tilde { \nu } ) \| \leq \Big ( \frac { \sqrt { d _ { h } } } { \omega _ { \operatorname* { m i n } } } + \frac { M _ { F } } { \omega _ { \operatorname* { m i n } } ^ { 2 } } \Big ) \operatorname* { s u p } _ { f \in \tilde { \mathfrak { F } } _ { \ell } , h } | \int f d \nu - \int f d \tilde { \nu } | .
$$

Together with Lemma 6.1 (ii), one has the total error brought by the change of q and change of measure:

$$
\| A _ { \ell } ^ { h } ( q , \nu ) - A _ { \ell } ^ { h } ( q ^ { \prime } , \tilde { \nu } ) \| \leq L _ { A } \| q - q ^ { \prime } \| + C _ { A } \operatorname* { s u p } _ { f \in \tilde { \mathfrak { F } } _ { \ell } { , h } } | \int f d \nu - \int f d \tilde { \nu } | ,\tag{A.1}
$$

with $C _ { A } = \sqrt { d _ { h } } / \omega _ { \operatorname* { m i n } } + M _ { F } / \omega _ { \operatorname* { m i n } } ^ { 2 }$ and $L _ { A } = L _ { \omega } / \omega _ { \mathrm { { m i n } } } + M _ { F } L _ { \omega } / \omega _ { \mathrm { { m i n } } } ^ { 2 }$

Since

$$
\begin{array} { r } { H _ { \ell } ( Z , \nu ) = \left[ A _ { \ell } ^ { h } ( z W _ { \ell , h } ^ { Q } , \nu ) \right] _ { h = 1 } ^ { H } W _ { \ell } ^ { O } , } \end{array}
$$

one has by (A.1) and the mean uniform law of large numbers in Lemma 6.2 that

$$
\operatorname* { s u p } _ { a \in K } \mathbb { E _ { \mathsf { e } } } \| H _ { \ell } ( Z _ { \ell } ^ { a , m } , \hat { \nu } _ { a , m } ) - H _ { \ell } ( Z _ { \ell } ^ { a } , \nu _ { a } ) \| \leq C _ { \ell } \Delta _ { \ell } + C _ { \ell } \delta _ { m } ^ { \mathrm { a t t } } .
$$

Recall (6.2), using the Lipschitz bounds for $G _ { \ell }$ one has,

$$
\begin{array} { r } { \Delta _ { \ell + 1 } \leq C _ { \ell } \Delta _ { \ell } + C _ { \ell } \delta _ { m } ^ { \mathrm { a t t } } . } \end{array}
$$

The depth $L _ { \mathrm { e n c } }$ is fixed, so one easily derive by induction that

$$
\operatorname* { s u p } _ { a \in K } \mathbb { E _ { e } } \| z _ { a , m } - z _ { a } \| \leq C _ { \theta } \delta _ { m } ^ { \mathrm { a t t } } .
$$

Step 2: estimate for the encoder and the terminal law Fix a localization radius R to be determined. The decoder Lipschitz condition (Lemma 6.1) on compact x-cylinders gives

$$
\begin{array} { r l } & { \eta _ { m } ( R ) = \underset { a \in K } { \operatorname* { s u p } } \mathbb { E } _ { \mathbf { e } } \quad \underset { ( x , t ) \in \overline { { B } } _ { R } \times [ 0 , 1 ] } { \operatorname* { s u p } } | v _ { a , m } ( x , t ) - v _ { a } ( x , t ) | } \\ & { \qquad \le L _ { \theta , R } ^ { { \mathrm { d e c } } } \underset { a \in K } { \operatorname* { s u p } } \mathbb { E } _ { \mathbf { e } } \| z _ { a , m } - z _ { a } \| \le C _ { \theta , R } \delta _ { m } ^ { \mathrm { a t t } } . } \end{array}
$$

Since the finite and infinite sensor flows start from the same $X _ { 0 } \sim \gamma$ , one has for fixed a and e that

$$
W _ { 2 } ^ { 2 } ( \hat { \mu } _ { a , m } , \hat { \mu } _ { a } ) \leq \mathbb { E } _ { X _ { 0 } } | X _ { 1 } ^ { a , m } - X _ { 1 } ^ { a } | ^ { 2 } .
$$

Let $\tau _ { R }$ be the first time either path exits ${ \overline { { B } } } _ { R }$ . On $\{ \tau _ { R } \ \geq \ 1 \}$ both paths stay in ${ \overline { { B } } } _ { R } ,$ so Gr¨onwall and the local x-Lipschitz bound in (A2) give the pathwise estimate

$$
| X _ { 1 } ^ { a , m } - X _ { 1 } ^ { a } | \mathbf 1 _ { \{ \tau _ { R } \geq 1 \} } \leq e ^ { L _ { \theta , R } ^ { x } } \operatorname* { s u p } _ { ( x , t ) \in \overline { { B } } _ { R } \times [ 0 , 1 ] } | v _ { a , m } ( x , t ) - v _ { a } ( x , t ) | .
$$

The linear growth in (A2) easily gives, for any fixed $p > 2$

$$
\mathbb { E } \operatorname* { s u p } _ { t \leq 1 } | X _ { t } | ^ { p } \leq C _ { \theta , p }
$$

uniformly in a and e and for both $\hat { \mu } _ { a , m }$ and $\hat { \mu } _ { a }$ . Hence, for any a and $\mathbf { e } , \mathbb { P } ( \tau _ { R } < 1 ) \le C _ { \theta , p } R ^ { - p }$ 2 and thus

$$
\begin{array} { r } { \mathbb { E } \big [ | X _ { 1 } ^ { a , m } - X _ { 1 } ^ { a } | ^ { 2 } ; \tau _ { R } < 1 \big ] \leq \big ( \mathbb { E } | X _ { 1 } ^ { a , m } - X _ { 1 } ^ { a } | ^ { p } \big ) ^ { 2 / p } \mathbb { P } ( \tau _ { R } < 1 ) ^ { 1 - 2 / p } \leq C _ { \theta , p } R ^ { - ( p - 2 ) } . } \end{array}
$$

Combining the results above, one has

$$
\operatorname* { s u p } _ { a \in K } \mathbb { E _ { e } } W _ { 2 } ( \hat { \mu } _ { a , m } , \hat { \mu } _ { a } ) \leq e ^ { L _ { \theta , R } ^ { x } } C _ { \theta , R } \delta _ { m } ^ { \mathrm { a t t } } + C _ { \theta , p } R ^ { - ( p - 2 ) / 2 } .
$$

Fixing R and letting $m  \infty ,$ , Lemma 6.2 gives $\delta _ { m } ^ { \mathrm { a t t } } \to 0 ;$ letting then $R \to \infty$ proves random sensor resolution invariance in expectation. □

## B Experimental Details

We compute Sinkhorn divergence with GeomLoss: p = 2 selects quadratic cost, and blur = 0.05 sets $\varepsilon = \mathtt { b l u r } ^ { 2 }$

All experiments are run on a single NVIDIA RTX 5090 GPU (32 GB) with an AMD EPYC 9575F CPU (128 cores), using Python 3.14, PyTorch 2.10, and CUDA 12.8. Elapsed times include all overhead (data loading, compilation, I/O). The RTX 5090 and the EPYC 9575F have comparable list prices (\$2,000 and \$2,200 respectively). The runtime comparisons between GPU neural sampling and CPU MCMC use hardware of equivalent cost.

## B.1 1D experiments

The two 1D experiments share the Multi-input DeepONet architecture of Section 4: latent dimension q = 128, base channels 32 in the branch CNN, a trunk MLP of depth 4 and width 256 with GELU activations, and a sinusoidal time embedding of dimension 32. Training uses conditional flow matching with minibatch optimal transport coupling of the source–target pairs [20]. Table B.1 summarizes the data generation and optimization set tings of the two experiments side by side.

Table B.1: Configuration of the two 1D experiments.
<table><tr><td>Function instances</td><td>1D Variable Noise</td><td>1D Rare Event</td></tr><tr><td>Train / test split Grid resolution over [—5, 5] Reference samples per function Reference generation time step ∆t equilibration time  $T _ { \mathrm { b u r n } }$ </td><td>65,536 64,512 /1,024 64 4,096 Euler-Maruyama 0.01</td><td>65,536 64,512 /1,024 256 4,096 direct GMM draws</td></tr><tr><td>collection interval Epochs (gradient steps)</td><td>50 0.1 1 (32,256)</td><td>1 (32,256)</td></tr><tr><td>Optimizer (lr, weight decay) Batch size / samples per function</td><td>AdamW (10−3, 10−4), OneCycleLR 256/32</td><td></td></tr><tr><td>Gradient clipping / precision Training time</td><td>1.0 / bfloat16 mixed</td><td></td></tr><tr><td></td><td>263.1 s</td><td>296.5 s</td></tr></table>

In 1D Variable Noise, the GRF perturbations $s , s ^ { \prime }$ are sampled on the uniform grid via the FFT, and the reference Euler–Maruyama simulation runs on the linearly interpolated drift and difusion. The MCMC baseline in Table 7.1 runs Euler–Maruyama with the same $T _ { \mathrm { b u r n } } = 5 0$ on the 1,024 test functions, collecting 4,096 samples each. Figure B.1 shows the training loss curve.

In 1D Rare Event, the difusion coeficient is a small random perturbation of a constant,

$$
\sigma ( x ) = 0 . 1 + 0 . 1 \sum _ { k = 1 } ^ { 2 } a _ { k } \cos ( b _ { k } x + c _ { k } ) ,\tag{B.1}
$$

and the drift follows from the zero flux condition of the stationary Fokker–Planck equation $\begin{array} { r } { \partial _ { x } ( b p - \frac { 1 } { 2 } \partial _ { x } ( \sigma ^ { 2 } p ) ) = 0 } \end{array}$ , which yields

$$
\begin{array} { r } { b ( x ) = \frac { 1 } { 2 } \sigma ( x ) ^ { 2 } \nabla \log p ( x ) + \sigma ( x ) \nabla \sigma ( x ) . } \end{array}\tag{B.2}
$$

Because the target density $p ( x )$ is a known Gaussian mixture, training samples are drawn directly from the GMM; this makes the experiment a controlled test of the model’s representational capacity and generalization in a rare event regime, and complete sample generation from the SDE is left for the other experiments. The MCMC baselines in Table 7.2 run Euler–Maruyama with $\Delta t = 0 . 0 0 1$ on the linearly interpolated drift and difusion for each of the 1,024 test functions, collecting 4,096 samples: the converged baseline uses $5 \times 1 0 ^ { 5 }$ steps per function (elapsed time 136.9 s), and three baselines with controlled runtimes limit the MCMC budget per function.

For inference we compared RK4 with 4 and 16 integration steps against the adaptive dopri5 solver. RK4 with only 4 steps yields Sinkhorn = 0.131 in 1.4 s (1,024 functions), matching dopri5 (Sinkhorn = 0.130, 7.4 s) at 5× lower cost; RK4-4 is therefore the default solver in all 1D experiments. Figure B.2 shows the distribution of Sinkhorn divergence across test functions for 1D Rare Event: the model’s error distribution is concentrated near zero, while MCMC exhibits a heavy right tail caused by failures from mode collapse.

![](images/d182e61f5b971a81174746924acbdff93ca37842e79fad6ffffdc197bb9567d3.jpg)  
Figure B.1: 1D Variable Noise. Training loss.

![](images/b1f296c6de151d77aaa3e92f7311d63e9026c4ad8e2f65839ff2daf88bc11ab3.jpg)

![](images/188d6a93c9582d22349cf291a3b95b1a5dbb661c5b079fa51b03c32f73dd29fc.jpg)  
Figure B.2: 1D Rare Event. Distribution of Sinkhorn divergence across 1,024 test functions: operator model vs. MCMC (converged). Left: histogram clipped at the 95th percentile; right: box plot on log scale.

## B.2 2D Constant Noise

We generate 65,536 drift fields for training and 4,096 additional drift fields for evaluation. For each drift field, reference stationary samples are obtained by Euler–Maruyama with $d t = 0 . 0 1$ and an equilibration period of 20 time units (2,000 steps), after which 4,096 samples are collected per function. All reported metrics are Sinkhorn divergences between 4,096 generated and 4,096 reference samples, computed on 100 test drift fields drawn from the evaluation split.

Probe data contain 20 steps with 8 features per step. Point and endpoint models use MLP encoders, and trajectory models use a 1D CNN. All use latent width 128 and a residual flow decoder trained with conditional flow matching and minibatch OT coupling. Each variant is trained for one epoch over the 65,536-function training split (8,192 optimizer steps at batch size 512) using OneCycleLR with peak learning rate $\bar { 1 0 } ^ { - 3 }$ , weight decay $1 0 ^ { - 4 }$ , gradient clipping 1.0, torch.compile, and bfloat16 mixed precision.

In Table B.2, Concat appends the context to the hidden features at each decoder block. Dot uses context-dependent weights to multiply the hidden features, following the branch– trunk product in DeepONet [22]. The sensor budget variants change the training count and retain 256 evaluation sensors.

Table B.2: 2D Constant Noise. Additional variants trained on 65,536 fields and evaluated on 100 test fields with 4,096 generated and reference samples each. All use trajectory probes, Perceiver, and 256 evaluation sensors.
<table><tr><td>Conditioning</td><td>Train sensors</td><td>Mean Sinkhorn</td><td>Train (s)</td></tr><tr><td>Dot</td><td>256</td><td>0.0203</td><td>248.6</td></tr><tr><td>Concat</td><td>256</td><td>0.0189</td><td>253.9</td></tr><tr><td>AdaLN</td><td>64</td><td>0.0180</td><td>180.1</td></tr><tr><td>AdaLN</td><td>128</td><td>0.0187</td><td>303.4</td></tr></table>

In the sensor resolution transfer study (Table 7.4), each probe executes a 20-step random walk with step size 0.05. For the 2048-sensor and mixed count rows we keep the efective optimizer batch fixed at 512 by using microbatch 256 with gradient accumulation over two steps; the rows with high resolution fit on a single GPU without changing the optimization budget. Figure B.3 shows qualitative comparisons of target and generated samples for representative test drift fields.

![](images/52eac33ccfc7f23e81701b8140b18c3388fd9c675923453c8bfb6b33b021225f.jpg)

![](images/b89244814f5e5f638b30a59b305c232bf5b70485875ce52adaec02d6db09eca3.jpg)

![](images/1214316295bdc1a3e0bc5727ec8ba4e9b2cf3b72cc6ea775703fa334c8860ab7.jpg)

![](images/17ec753d57f7d55afe8b318d86cb747d600b2babdec59f424fb33d582fad9130.jpg)

![](images/804ccf9afefd61f4d1b3d7eac00ba1d33b958a2ad27c8bd962e80852c751c54e.jpg)

![](images/b1f0e25fdd8dab2d5a1cd2501803c4e1e9956e6406b24c99cb019fde1fe1fc5d.jpg)

![](images/fae519921906ef10f0e125f40759c764b6e444e57a99d384dd723a5fe39950ed.jpg)

![](images/ff82dd917bf19a7c9e6a8504914767960534138e0aedf2508da01f8e7438d9d0.jpg)  
<sup>4 4 4 4</sup>Figure B.3: 2D Constant Noise. Qualitative comparison of target (ground truth) and gen-2 2 2erated samples for representative test drift fields.

## B.3 64D Interacting Particle

Table B.3 lists the four interaction kernel families. All kernels are nonsingular. For Morse and WCA, we cap the energy at $U _ { \mathrm { M A X } } = 1 0$ to avoid numerical instabilities during MCMC <sup>4</sup>equilibration; at $\beta = 2$ <sup>4 4 4</sup>, the corresponding Boltzmann weight is already exp $\left( - \beta U _ { \mathrm { M A X } } \right)$ ≈ $2 \times 1 0 ^ { - 9 }$

<sup>0 0 0</sup>Table B.3: Interaction kernel families for 64D Interacting Particle.
<table><tr><td>Family</td><td>Form</td><td>Character</td></tr><tr><td>Morse</td><td> $U ( r ) = D _ { e } ( 1 - e ^ { - a ( r - r _ { e } ) } ) ^ { 2 } - D _ { e }$ </td><td>bondlike attraction with repulsive core</td></tr><tr><td>Gaussian</td><td> $U ( r ) = \alpha \exp ( - r ^ { 2 } / 2 \sigma ^ { 2 } )$ </td><td>soft repulsion  $( \alpha > 0 )$  or attraction  $( \alpha < 0 )$ </td></tr><tr><td>Double Gaussian</td><td>short range repulsion + long range attraction</td><td>preferred interparticle spacing</td></tr><tr><td>WCA</td><td>truncated and shifted Lennard-Jones with softening</td><td>hard sphere exclusion</td></tr></table>

The sensor set for pair probes consists of 512 isolated trajectories of particle pairs per kernel. Each probe starts at a random separation $r _ { 0 } ~ \in ~ [ 0 . 2 , 4 . 0 ]$ , is evolved for 20 short steps $( d t = 0 . 0 0 1 )$ , and records the relative displacement and interaction force, giving a sensor tensor of shape (512, 20, 4) per kernel. Reference samples are generated by Euler– Maruyama with $d t = 0 . 0 0 1$ , 16 chains per system, an equilibration period of $1 0 ^ { 5 }$ steps, and $^ \mathrm { 4 , 0 9 6 }$ retained samples per kernel. The training set contains 2,048 kernels and the evaluation set contains 512 test kernels; we report metrics on 100 test kernels.

The architecture is the ParticleAmortizedSampler of Section 4: a 1D CNN sensor encoder, a Perceiver aggregator with four latent tokens, and an equivariant particle flow network conditioned by AdaLN. The model has 1.78M parameters and is trained for one epoch (2,048 optimizer steps) by conditional flow matching with learning rate $1 0 ^ { - 3 }$ , so the product of learning rate and step count is order one. On a local RTX 5090 system, the full data generation, training, evaluation, and plotting pipeline completes in roughly 40 minutes; dataset generation dominates the cost.

As a scale reference for the conditioning diagnostic in Table 7.5, comparing two independent halves of the MCMC target sample under the same sliced 2-Wasserstein score gives a noise floor from the target split of 0.2038 (mean) and 0.1071 (median); the correctly conditioned sampler (1.7860 / 1.0159) thus remains above the finite sample floor of the metric. We attribute this gap primarily to the behavior of sliced 2-Wasserstein on finite samples in 64 dimensions rather than to a correspondingly large sampling error; the pair distance distributions in Figure 7.4 give the more direct evidence that the sampler conditions on the interaction kernel.

Table B.4 reports the observable checks invariant under permutations that are summarized in Section 7.3.

Table B.4: 64D Interacting Particle. Observable checks invariant under permutations, evaluated on 100 test kernels with 4,096 samples per kernel.
<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Pair correlation  $g ( r ) ~ L ^ { 2 }$  error</td><td>0.6865</td></tr><tr><td>Mean marginal KL</td><td> $1 . 2 1 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Energy KS statistic</td><td>0.1585</td></tr><tr><td>Energy mean: model / target  $( \Delta , \mathrm { r e l . } )$ </td><td>614.40 / 638.76 (−24.36, -3.8%)</td></tr><tr><td>Energy std: model / target  $( \Delta , \mathrm { r e l . } )$ </td><td>1472.60/  $1 5 0 4 . 7 3 \ : \left( - 3 2 . 1 3 , \ : - 2 . 1 \% \right)$ </td></tr></table>

## References

[1] Michael S. Albergo, Nicholas M. Bofi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and difusions. Journal of Machine Learning Research, 26(209):1–80, 2025.

[2] Jean-David Benamou and Yann Brenier. A computational fluid mechanics solution to the monge-kantorovich mass transfer problem. Numerische Mathematik, 84(3):375–393, 2000.

[3] Vladimir I Bogachev, Nicolai V Krylov, Michael R¨ockner, and Stanislav V Shaposhnikov. Fokker–Planck–Kolmogorov Equations, volume 207. American Mathematical Society, 2022.

[4] Yann Brenier. Polar factorization and monotone rearrangement of vector-valued functions. Communications on Pure and Applied Mathematics, 44(4):375–417, 1991.

[5] Zhiqiang Cai, Yu Cao, Yuanfei Huang, and Xiang Zhou. Weak generative sampler to eficiently sample invariant distribution of stochastic diferential equation. SIAM Journal on Scientific Computing, 48(4):C708–C735, 2026. First posted as arXiv:2405.19256 in 2024.

[6] Edoardo Calvello, Nikola B. Kovachki, Matthew E. Levine, and Andrew M. Stuart. Continuum attention for neural operators. arXiv preprint arXiv:2406.06486, 2024.

[7] Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. Neural ordinary diferential equations. In Advances in Neural Information Processing Systems, volume 31, 2018.

[8] Naoufal El Bekri, Lucas Drumetz, and Franck Vermet. Flowkac: An eficient neural fokker-planck solver using temporal normalizing flows and the feynman-kac formula. Transactions on Machine Learning Research, 2025.

[9] Xue Feng, Li Wang, Deanna Needell, and Rongjie Lai. Learn to evolve: Self-supervised neural JKO operator for wasserstein gradient flow. arXiv preprint arXiv:2601.05583, 2026.

[10] Jean Feydy, Thibault S´ejourn´e, Fran¸cois-Xavier Vialard, Shun-ichi Amari, Alain Trouv´e, and Gabriel Peyr´e. Interpolating between optimal transport and MMD using Sinkhorn divergences. In Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics, volume 89 of Proceedings of Machine Learning Research, pages 2681–2690. PMLR, 2019.

[11] W. Keith Hastings. Monte carlo sampling methods using markov chains and their applications. Biometrika, 57(1):97–109, 1970.

[12] Han Huang and Rongjie Lai. Unsupervised solution operator learning for mean-field games. Journal of Computational Physics, 537:114057, 2025.

[13] Andrew Jaegle, Felix Gimeno, Andrew Brock, Andrew Zisserman, Oriol Vinyals, and Joao Carreira. Perceiver: General perception with iterative attention. In Proceedings of the 38th International Conference on Machine Learning, volume 139, pages 4651–4664. PMLR, 2021.

[14] Pengzhan Jin, Shuai Meng, and Lu Lu. MIONet: Learning multiple-input operators via tensor product. SIAM Journal on Scientific Computing, 44(6):A3490–A3514, 2022. Publisher: Society for Industrial and Applied Mathematics.

[15] Nikola Kovachki, Zongyi Li, Burigede Liu, Kamyar Azizzadenesheli, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Neural operator: Learning maps between function spaces with applications to PDEs. Journal of Machine Learning Research, 24(89):1–97, 2023.

[16] Juho Lee, Yoonho Lee, Jungtaek Kim, Adam R. Kosiorek, Seungjin Choi, and Yee Whye Teh. Set transformer: A framework for attention-based permutation-invariant neural networks. In Proceedings of the 36th International Conference on Machine Learning, pages 3744–3753. PMLR, 2019.

[17] Ben Leimkuhler and Charles Matthews. Molecular Dynamics: With Deterministic and Stochastic Numerical Methods, volume 39 of Interdisciplinary Applied Mathematics. Springer, Cham, 2015.

[18] Zongyi Li, Nikola Borislavov Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Fourier neural operator for parametric partial diferential equations. In International Conference on Learning Representations, 2021.

[19] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023.

[20] Yaron Lipman, Marton Havasi, Peter Holderrieth, Neta Shaul, Matt Le, Brian Karrer, Ricky T. Q. Chen, David Lopez-Paz, Heli Ben-Hamu, and Itai Gat. Flow matching guide and code, 2024.

[21] Hude Liu, Jerry Yao-Chieh Hu, Zhao Song, and Han Liu. Attention mechanism, maxafine partition, and universal approximation. arXiv preprint arXiv:2504.19901, 2025.

[22] Lu Lu, Pengzhan Jin, and George Em Karniadakis. DeepONet: Learning nonlinear operators for identifying diferential equations based on the universal approximation theorem of operators. Nature Machine Intelligence, 3(3):218–229, 2021.

[23] Jonathan C. Mattingly, Andrew M. Stuart, and Desmond J. Higham. Ergodicity for SDEs and approximations: Locally lipschitz vector fields and degenerate noise. Stochastic Processes and their Applications, 101(2):185–232, 2002.

[24] Nicholas Metropolis, Arianna W. Rosenbluth, Marshall N. Rosenbluth, Augusta H. Teller, and Edward Teller. Equation of state calculations by fast computing machines. The Journal of Chemical Physics, 21(6):1087–1092, 1953.

[25] Frank No´e, Simon Olsson, Jonas K¨ohler, and Hao Wu. Boltzmann generators: Sampling equilibrium states of many-body systems with deep learning. Science, 365(6457):eaaw1147, 2019.

[26] Grigorios A. Pavliotis. Stochastic Processes and Applications: Difusion Processes, the Fokker–Planck and Langevin Equations, volume 60 of Texts in Applied Mathematics. Springer, New York, NY, 2014.

[27] William Peebles and Saining Xie. Scalable difusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 4195–4205, 2023.

[28] Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, and Aaron Courville. FiLM: Visual reasoning with a general conditioning layer. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32, 2018.

[29] Michael Prasthofer, Tim De Ryck, and Siddhartha Mishra. Variable-input deep operator networks. arXiv preprint arXiv:2205.11404, 2022.

[30] Md Ashiqur Rahman, Manuel A. Florez, Anima Anandkumar, Zachary E. Ross, and Kamyar Azizzadenesheli. Generative adversarial neural operators. Transactions on Machine Learning Research, 2022.

[31] Danilo Jimenez Rezende and Shakir Mohamed. Variational inference with normalizing flows. In International Conference on Machine Learning, pages 1530–1538, 2015.

[32] Christian P. Robert and George Casella. Monte Carlo Statistical Methods. Springer, 2 edition, 2004.

[33] L Chris G Rogers and David Williams. Difusions, Markov processes, and martingales, volume 2. Cambridge university press, 2000.

[34] Yaozhong Shi, Zachary E. Ross, Domniki Asimaki, and Kamyar Azizzadenesheli. Stochastic process learning via operator flow matching. In Advances in Neural Information Processing Systems, 2025. Spotlight.

[35] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic diferential equations. In International Conference on Learning Representations, 2021.

[36] C´edric Villani et al. Optimal transport: old and new, volume 338. Springer, 2008.

[37] Chulhee Yun, Srinadh Bhojanapalli, Ankit Singh Rawat, Sashank J. Reddi, and Sanjiv Kumar. Are transformers universal approximators of sequence-to-sequence functions? In International Conference on Learning Representations (ICLR), 2020.

[38] Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos, Ruslan Salakhutdinov, and Alexander Smola. Deep sets. In Advances in Neural Information Processing Systems, volume 30, 2017.

[39] Yaohua Zang, Gang Bao, Xiaojing Ye, and Haomin Zhou. Weak adversarial networks for high-dimensional partial diferential equations. Journal of Computational Physics, 411:109409, 2020.