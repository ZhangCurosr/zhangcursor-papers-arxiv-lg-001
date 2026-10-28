# LOG-DERIVATIVE TRICK FOR PARAMETRIC MARKOV CHAINS

Arthur Gall<sup>1,2</sup>

Ichiro Hasuo<sup>1,2,3</sup>

Thomas Seiller<sup>4</sup>

<sup>1</sup>National Institute of Informatics, Tokyo, Japan <sup>2</sup>SOKENDAI (The Graduate University for Advanced Studies), Kanagawa, Japan <sup>3</sup>Imiron Co., Ltd., Tokyo, Japan <sup>4</sup>CNRS, France

Abstract. Optimizing expected rewards and reachability probabilities in parametric Markov chains (pMCs) is central to the analysis and synthesis of probabilistic systems with loops. While traditional symbolic synthesis approaches—such as parameter lifting and rational function generation—suffer from scalability bottlenecks, gradient-based optimization emerges as an effective alternative. However, existing gradient-based techniques still rely on symbolic gradient computation—such as solving large linear systems—over the global state space, inheriting scalability limits. Finite-horizon approximation is possible but introduces uncontrollable bias.

In this paper, we introduce a scalable Monte Carlo gradient estimator based on a measuretheoretic generalization of the log-derivative trick. We prove that our estimator provides unbiased gradient estimates under almost-sure termination and standard regularity conditions. Crucially, our method avoids global state-space materialization and linear system solving entirely: it operates purely along sampled trajectories, requiring only black-box evaluations of local transition probabilities (or densities) and their derivatives. Unlike existing pMC techniques that are fundamentally restricted to finite discrete systems, our measure-theoretic framework naturally extends to continuous and general σ-finite state spaces. Empirical evaluations on synthetic discrete pMCs and continuous-state case studies demonstrate that our method scales to state spaces far beyond the reach of existing solvers, significantly reducing execution time without introducing bias.

## 1 INTRODUCTION

Solving verification and optimization problems over probabilistic programs is part of a growing research trend (Barthe et al., 2020; Baier & Katoen, 2008) with a wide range of applications, including security (Barthe et al., 2012; Roy et al., 2021), reliability (Carbin et al., 2013; Wang et al., 2021) and machine learning (Bingham et al., 2019; van de Meent et al., 2018; Lew et al., 2023). Parametric Markov chains (pMCs) provide a foundational model for probabilistic systems whose transition probabilities depend on a set of global parameters, capturing parameter dependencies across states. Since they naturally support loops, they represent a wide subclass of unconditioned probabilistic programs.

![](images/09a1669baee817900fc9226b6bd1544b15d469ada786796541b6eb048a3f2499.jpg)  
Figure 1: A finite MC $T _ { 1 }$

![](images/d1617c662ba181b0b23f080c2a4f9a347cd41fe4acc236fcac5cb958171420eb.jpg)  
Figure 2: A finite pMC $T _ { 2 }$

![](images/8b741caf5c7c405634f4311809752e637913039d493b438ab2d00815daab3dc8.jpg)  
Figure 3: A continuous pMC $T _ { 3 }$

Figure 1 presents a finite Markov chain (MC); Figure $^ 2$ presents a finite pMC with a parameter p; Figure 3 presents an infinite pMC with a continuum of states $x \in \mathbb { R } _ { + }$ , with parameter $p .$ Here, −1 and +1 denote output states of the system, carrying designated terminal rewards.

Verification and optimization over pMCs have been extensively studied (Daws, 2005; Junges et al., 2024), with efficient algorithms and tools (Dehnert et al., 2015; Cubuktepe et al., 2022; Badings et al., 2023; Spel et al., 2021; Heck et al., 2025) applied to reliability analysis (Volk et al., 2016; Bartocci et al., 2011), planning under uncertainty (Junges et al., 2018; Polgreen et al., 2016), and systems biology (Ce<sup>ˇ</sup> ska et al., 2017).ˇ

Optimization over pMCs Given a real reward function defined over the output states of a parametric Markov chain ({−1, +1} in Figures 2–3), our goal is to synthesize a parameter valuation that maximizes the expected reward. Existing techniques broadly fall into two classes.

• Symbolic synthesis: Approaches based on parameter lifting, rational function generation, or feasibility region decomposition (Spel et al., 2021; Dehnert et al., 2015; Heck et al., 2025) offer rigorous global optimality guarantees by symbolically exploring the parameter space. However, their computational complexity scales poorly with the number of parameters; state-of-the-art tools rarely scale beyond dozens of parameters (Heck et al., 2025).

• Gradient descent: When systems involve higher-dimensional parameter spaces, gradient descent, while lacking global optimality guarantees, provides a practical and scalable optimization paradigm. Here, the central computational challenge lies in efficiently computing the gradient of the expected reward with respect to the parameters.

Linear Solver Method for Gradients, Exact and Approximate Recent work (Heck et al., 2022) presents what we call the linear solver method; it computes exact gradients in finite pMCs by solving a coupled linear system. Specifically, for an n-state pMC with transition matrix A and reward vector b, the expected reward vector x satisfies $x = A x + b$ . Differentiating this relation with respect to a parameter yields $\partial x = ( \partial A ) x + A \partial x + \partial b$ , which gives:

$$
\partial x = ( I - A ) ^ { - 1 } \left( ( \partial A ) x + \partial b \right) \quad { \mathrm { w h e r e } } \quad x = ( I - A ) ^ { - 1 } b .\tag{1}
$$

Various numerical solvers enable the solution of such linear systems, ranging from direct sparse factorization methods (Davis, 2006) to iterative Krylov subspace algorithms (Saad, 2003), which are particularly well-suited for large sparse representations. These solvers provide exact gradient evaluations (up to numerical precision), yet this exactness incurs a substantial computational cost on large instances.

One can imagine a natural approximation of the linear solver method, namely the Jacobi method applied to (1). It approximates the gradient by unrolling the linear system for a fixed budget of K steps. However, this finite-horizon truncation introduces uncontrollable bias and fails to capture long-horizon behaviors.

The linear solver method, both exact and approximate, comes with two fundamental limitations:

• Global state-space materialization: They require explicitly constructing and storing the full transition matrix A (and its derivative ∂A), which quickly exhausts memory on large models.

• Restriction to finite discrete systems: They fundamentally operate on discrete adjacency matrices, rendering them inapplicable to continuous or general state spaces such as Figure 3.

Our Method: Sampling Gradients via the Log-Derivative Trick In this paper, we present a new algorithm that samples the gradient via the log-derivative trick (Williams, 1992), operating entirely along sampled trajectories without state-space materialization and naturally extending to infinite and continuous state spaces. Note first that the expected reward $\mathbb { E } [ r ]$ over paths π is given by the formula

$$
\mathbb { E } [ r ] = \sum _ { \pi \in \mathrm { P a t h } } \mathsf { p } _ { \pi } r _ { \pi } = \sum _ { \pi \in \mathrm { P a t h } } \mathsf { p } _ { \pi _ { 1 } } \ldots \mathsf { p } _ { \pi _ { | \pi | } r _ { \pi } , }
$$

where ${ \mathsf p } _ { \pi _ { i } }$ is the probability of taking the branch $\pi _ { i } .$ , and $\mathsf { p } _ { \pi }$ is that for the path $\pi .$ . We assume that the reward function does not depend on the parameters, i.e. $\partial r _ { \pi } = 0$ (for simplicity; lifting this is easy). Then differentiation ∂ (with respect to parameters) yields

$$
\partial \mathbb { E } [ r ] = \sum _ { \pi \in \mathrm { P a t h } } \partial \left( \mathsf { p } _ { \pi _ { 1 } } \ldots \mathsf { p } _ { \pi _ { | \pi | } } \right) r _ { \pi } = \sum _ { \pi \in \mathrm { P a t h } } \mathsf { p } _ { \pi _ { 1 } } \ldots \mathsf { p } _ { \pi _ { | \pi | } } \partial \log \left( \mathsf { p } _ { \pi _ { 1 } } \ldots \mathsf { p } _ { \pi _ { | \pi | } } \right) r _ { \pi } = \mathbb { E } [ \partial \log \mathsf { p } _ { \pi } \cdot r ] ,
$$

where the second equality used the log-derivative trick (∂ log $f = \partial f / f )$ . The last expression $\mathbb { E } [ \partial \log { \mathfrak { p } } _ { \pi } \cdot r ]$ implies that the gradient can be estimated by sampling paths π directly from the parametric Markov chain and, by the product rule of logarithm, accumulating the sum of the logderivatives of the transition probabilities scaled by the reward of the path. Indeed,

$$
\partial \log \left( \mathsf { p } _ { \pi _ { 1 } } \ldots \mathsf { p } _ { \pi _ { \left| \pi \right| } } \right) = \partial \log \mathsf { p } _ { \pi _ { 1 } } + \cdot \cdot \cdot + \partial \log \mathsf { p } _ { \pi _ { \left| \pi \right| } } .
$$

Our algorithm operates as a Monte Carlo gradient estimator (Mohamed et al., 2020), reducing computational cost at the price of increased variance. As a theoretical result, we prove that our sampling algorithm provides unbiased gradient estimates under almost-sure termination and standard regularity conditions. We also demonstrate experimentally that, despite the potentially unbounded path lengths arising from cycles, our algorithm remains scalable and effective in practice for computing an approximation of the gradient. We show that our algorithm is suitable for gradient descent in pMCs and outperforms other methods such as the linear solver method of Heck et al. (2022).

Beyond finite systems, our framework naturally generalizes to infinite and continuous state spaces (such as the model in Figure 3), which lie entirely outside the scope of methods that require explicit state spaces (such as symbolic synthesis and the linear solver methods). As another theoretical contribution, we generalize the log-derivative trick via the Radon–Nikodym derivative, and show that our estimator remains unbiased for any σ-finite transition structure under almost-sure termination and standard regularity conditions.

Summarizing, the main contribution of this paper is a trajectory-based gradient estimator algorithm for parametric Markov chains. It features the following theoretical and practical characteristics:

• Unbiasedness: Under almost-sure termination and standard regularity conditions, our estimator yields unbiased gradient estimates. This is unlike the approximate linear solver (see above).

• No state-space materialization needed: The estimator operates purely along sampled paths. It requires only local evaluations of transition probabilities (or densities) and their derivatives, completely bypassing the need to store entire transition matrices or solve global linear systems.

• General state spaces: Formulated in general measure-theoretic terms, our framework handles discrete, continuous, and hybrid σ-finite state spaces within a unified mathematical foundation.

• Empirical scalability and effective synthesis: We show that our estimator pairs naturally with constrained gradient-descent optimizers. In end-to-end parameter synthesis under fixed time budgets, it quickly boosts rewards by trading exact gradient computations for faster, approximate steps. It scales to models orders of magnitude larger than linear solvers (and thus than symbolic synthesis). It also optimizes continuous-state pMCs where other methods cannot directly apply.

## RELATED WORK

While the techniques surveyed throughout this paper originate primarily in formal verification and synthesis (Baier & Katoen, 2008), gradient-based optimization on parametric Markov models was already explored in early foundational work from the reinforcement learning and control communities (Marbach & Tsitsiklis, 2001; Baxter & Bartlett, 2001), relying on the classical log-derivative trick (Williams, 1992). However, their algorithmic objectives and structural assumptions diverge fundamentally from ours. Specifically, those works address infinite-horizon, non-terminating systems, seeking to optimize the long-run average reward assuming ergodicity. To estimate gradients along infinite trajectories, they either assume regenerative renewal cycles (optimizing reward per time step rather than total return) or introduce artificial discount factors that render the resulting gradient estimates inherently biased (Baxter & Bartlett, 2001). In contrast, the setting studied in thi work targets terminating parametric Markov chains, where executions almost surely hit an absorbing target state, and aims to optimize terminal rewards or reachability probabilities along these finite trajectories without introducing bias or relying on ergodicity.

## PRELIMINARIES AND NOTATIONS

We adopt the following measure-theoretic, algebraic, and differential conventions.

• Sequences and Paths: For a set $S , S ^ { n }$ denotes the n-fold Cartesian product (with $S ^ { 0 } = \left\{ \varepsilon \right\}$ containing the empty sequence $\varepsilon )$ , and $\begin{array} { r } { S ^ { * } = \operatorname { I I } _ { n = 0 } ^ { \infty } S ^ { n } } \end{array}$ denotes the set of all finite sequences (paths) over S. For a sequence $s = ( s _ { 1 } , \ldots , s _ { n } ) \in S ^ { n }$ , its length is denoted by $| s | = n$

• Measurable Spaces and Coproducts: A measurable space is denoted by $( X , { \mathcal { F } } _ { X } )$ Given two measurable spaces $( X , { \bar { \mathcal { F } } } _ { X } )$ and $( Y , { \mathcal { F } } _ { Y } )$ , their coproduct (disjoint union) is denoted by $\left( X + Y , { \mathcal { F } } _ { X + Y } \right)$ , where $X + Y$ is the disjoint union and $\bar { \mathcal { F } } _ { X + Y } = \{ A _ { X } ^ { \cdot } + A _ { Y } ~ | ~ A _ { X } \in \mathcal { F } _ { X } , A _ { Y } \stackrel { \cdot } { \in }$ $\mathcal { F } _ { Y } \}$ . For a measurable space $( S , \mathcal { F } _ { S } )$ , the sequence space $\begin{array} { r } { { \dot { S } } ^ { * } = { \dot { \coprod } } _ { n = 0 } ^ { \infty } S ^ { n } } \end{array}$ is equipped with the coproduct σ-algebra ${ \mathcal { F } } _ { S ^ { \ast } }$ , where $A \in { \mathcal { F } } _ { S ^ { * } }$ if and only if $A \cap S ^ { n } \in { \mathcal { F } } _ { S ^ { n } }$ for all $n \geq 0$

• Measures and Dirac Notation: A finite signed measure $\mu$ on $( X , { \mathcal { F } } _ { X } )$ is a σ-additive function $\mu : { \mathcal { F } } _ { X } \to \mathbb { R }$ with $\mu ( \varnothing ) = 0$ . Its total variation measure is denoted by $| \mu |$ . We let ⟨x| denote the Dirac delta measure concentrated at state $x ,$ defined by $\langle x | ( A ) = 1 { \mathrm { i } } { \dot { \mathrm { f } } } x \in A$ and 0 otherwise. A discrete distribution is written as a formal sum $\textstyle \sum _ { x } c _ { x } ^ { \phantom { } } \langle { \dot { x } } |$

• Differentiation and Integration: For functions or parameterized families over an open domain $P \subseteq \mathbb { R } ^ { k } , \partial _ { v }$ denotes the directional derivative along a direction vector $v \in \mathbb { R } ^ { k } , \bar { \partial } _ { i }$ denotes the partial derivative with respect to the i-th coordinate (i.e., along the standard basis vector $e _ { i } \in \mathbb { R } ^ { k } )$ , and $\partial _ { p }$ denotes the scalar derivative with respect to a single parameter p (when $k = 1 )$ . We write dx (or $\mathbb { i } s = ( \mathrm { d } s _ { 1 } , \dots , \mathrm { d } s _ { n } ) )$ for infinitesimal volume elements when integrating over a state space X (or product space S<sup>n</sup>).

## 2 OPTIMIZATION OF PARAMETRIC SIGNED MARKOV CHAINS

## 2.1 PARAMETRIC MARKOV KERNELS AND SIGNED KERNELS

We develop our theory in general measure theoretic terms; it accommodates continuous state spaces.

To formalize parametric Markov chains over arbitrary state spaces (including continuous ones), we rely on the framework of Markov kernels. Let $( X , { \mathcal { F } } _ { X } )$ and $( \overbar { Y } , \mathcal { F } _ { Y } )$ be two measurable spaces, and let $\ b { \mathscr { P } } \subseteq \mathbb { R } ^ { k }$ denote an open parameter space. A parametric Markov kernel $\kappa : X  Y$ is a function $\kappa : P \times X \times \mathcal { F } _ { Y } \to [ 0 , 1 ]$ ] such that for every parameter $p \in P$ and state $x \in X , \kappa ( p , x , - )$ is a probability measure on $( Y , { \mathcal { F } } _ { Y } )$ , and for every $p \in P$ and measurable set $A \in { \mathcal { F } } _ { Y } , \kappa ( p , - , A )$ is ${ \mathcal { F } } _ { X }$ -measurable. Given two parametric kernels $\kappa : X  Y$ and $\kappa ^ { \prime } : Y  { \sim } Z$ over the same parameter space $P$ , their composition $\kappa \kappa ^ { \prime } : X  Z$ is defined by

$$
\kappa \kappa ^ { \prime } ( p , x , \mathrm { d } z ) : = \int _ { y \in Y } \kappa ( p , x , \mathrm { d } y ) \kappa ^ { \prime } ( p , y , \mathrm { d } z ) .
$$

In the discrete and finite case, this formulation reduces to standard stochastic transition matrices, with kernel composition corresponding to matrix multiplication. For instance, a Bernoulli distribution Ber $\mathbf { n } ( p ) = ( 1 - p ) \left. 0 \right| + p \left. 1 \right|$ parameterized by $p \in [ 0 , 1 ]$ can be represented as the kernel $\kappa ( p , * , - ) = ( 1 - p ) \left. 0 \right| + p \left. 1 \right|$ , where ∗ denotes a single input state.

Signed Extension To compute and propagate gradients through such systems, we must differentiate kernels with respect to their parameters. Differentiating a probability measure yields quantities with both positive and negative values (for instance, $\partial _ { p } \mathbf { B e r n } ( \bar { p } ) = - \mathbf { \tau } \langle 0 | + \langle 1 | )$ . We therefore generalize to parametric signed kernels $\kappa : P \times X \times \dot { \mathcal { F } } _ { Y } \to \mathbb { R }$ , where $\kappa ( p , x , - )$ is a finite-valued signed measure for all $p \in P$ and $x \in X$ . Kernel composition extends naturally to signed kernels whenever the integral converges. Under this generalized framework, the derivative of the Bernoulli distribution is represented directly by the signed kernel $\partial _ { p } \kappa ( p , * , - ) = - \left. 0 \right| + \left. 1 \right|$ . Finally, we say that a parametric kernel κ is differentiable if, for every $x \in X$ and $A \in { \mathcal { F } } _ { Y }$ , the scalar map $p \mapsto \kappa ( p , x , A )$ is differentiable on $P ,$ , and all its directional derivatives define signed kernels.

## 2.2 ABSOLUTE DIFFERENTIABILITY

In this section, we introduce a mild regularity condition for parametric kernels, termed absolute differentiability. This condition relies on strong standard measure-theoretic tools, namely absolute continuity and the Radon–Nikodym derivative, which arise naturally when differentiating parametric kernels. Moreover, this property is satisfied by most standard continuous and discrete distributions $( \mathrm { e . g . }$ ., Gaussian, exponential, Bernoulli, geometric, etc.).

Definition 1 (Absolute Continuity). Let $\mu$ and $\nu$ be two signed measures over the same measurable space $( X , { \mathcal { F } } _ { X } )$ . The signed measure ν is said to be absolutely continuous with respect to $\mu ,$ , written $\nu \ll \mu , \operatorname { i f } | \mu | ( A ) = 0$ implies $| \nu | ( A ) = 0$ for all $A \in { \mathcal { F } } _ { X }$

Theorem 2 (Radon–Nikodym Theorem). Let µ and ν be two σ-finite signed measures over the same measurable space $( X , { \mathcal { F } } _ { X } )$ . Then ν is absolutely continuous with respect to µ if and only if there exists a measurable function $f : X \to$ R such that dν $= f \mathrm { d } \mu$ . Such a function $f$ is unique up to µ-null sets and is called the Radon–Nikodym derivative ofν with respect to µ, denoted $b y ~ { \frac { \mathrm { d } \nu } { \mathrm { d } \mu } }$

Definition 3 (Absolute Differentiability). Let $\kappa : X \sim Y$ be a differentiable parametric signed kernel with parameter space $P \subseteq \mathbb { R } ^ { k }$ . The kernel κ is said to be absolutely differentiable if, for all directional vectors $v \in \mathbb { R } ^ { k }$ , parameters $p \in P$ and states $x \in X$

$$
\partial _ { v } \kappa ( p , x , - ) \ll \kappa ( p , x , - )
$$

Remark 4. Intuitively, the absolute differentiability condition states that the derivative must vanish on any set where the kernel itself assigns zero probability (i.e., the derivative cannot be non-zero if the base probability is zero). Most standard discrete and continuous parametric kernels satisfy this property. Their associated Radon–Nikodym derivative corresponds to the derivative of the logarithm of the transition probabilities (in the discrete case) or transition densities (in the continuous case).

Example 5. Let $\kappa : \{ * \}  \{ 0 , 1 \}$ be the kernel parameterized by $p \in ( 0 , 1 )$ defined as $\kappa ( p , * , - ) =$ $\left( 1 - \sqrt { p } \right) \left. 0 \right| + \sqrt { p } \left. 1 \right|$ . Its derivative is the signed kernel $\partial _ { p } \kappa : \{ * \} \{  \{ \ r { 0 } , 1 \}$ given by $\partial _ { p } \kappa ( p , * , - ) =$ $\begin{array} { r } { - \frac { 1 } { 2 \sqrt { p } } \left. 0 \right| + \frac { \dot { 1 } } { 2 \sqrt { p } } \left. 1 \right| } \end{array}$ . The Radon–Nikodym derivative is the measurable function

$$
\frac { \mathrm { d } \partial _ { p } \kappa ( p , * , - ) } { \mathrm { d } \kappa ( p , * , - ) } : 0 \longmapsto \frac { \partial _ { p } \kappa ( p , * , \{ 0 \} ) } { \kappa ( p , * , \{ 0 \} ) } = - \frac { p + \sqrt { p } } { 2 p ( 1 - p ) } \quad ; \qquad 1 \longmapsto \frac { \partial _ { p } \kappa ( p , * , \{ 1 \} ) } { \kappa ( p , * , \{ 1 \} ) } = \frac { 1 } { 2 p } .
$$

Example 6. Let $\kappa : \{ * \}  \mathbb { R } _ { + }$ define an Exponential distribution $\kappa ( p , * , - ) = \mathbf { E x p } ( p )$ for $p > 0$ with probability density function $f _ { p } : x \mapsto p e ^ { - p x }$ . The derivative of this density with respect to p is $\partial _ { p } f _ { p } : x \mapsto ( 1 - p x ) e ^ { \bar { - } p x }$ . The Radon–Nikodym derivative is the measurable function

$$
{ \frac { \mathrm { d } \partial _ { p } \kappa ( p , * , - ) } { \mathrm { d } \kappa ( p , * , - ) } } : x \longmapsto { \frac { \partial _ { p } f _ { p } ( x ) } { f _ { p } ( x ) } } = { \frac { 1 - p x } { p } } = { \frac { 1 } { p } } - x .
$$

## 2.3 PARAMETRIC (SIGNED) MARKOV CHAINS

We formally introduce parametric Markov chains (pMCs), as a special case of the signed pMCs.

Remark 7 (Scope and Comparison with Literature). In the formal verification literature, parametric Markov chains are commonly defined as finite-state, discrete-time systems whose transition probabilities are rational or polynomial functions of parameters. In this work, our definition is more general: we consider discrete-time parametric Markov chains over σ-finite state spaces equipped with explicit input and output state boundaries.

Definition 8 (Parametric (Signed) Markov Chain). Let $( X , { \mathcal { F } } _ { X } ) , ( S , { \mathcal { F } } _ { S } )$ , and $( Y , { \mathcal { F } } _ { Y } )$ be measurable spaces representing the input states, internal states, and output (halting) states, respectively. A parametric signed Markov chain is a parametric signed kernel of the form

$$
T : X + S  S + Y
$$

parameterized by $p \in P \subseteq \mathbb { R } ^ { k }$ , where + denotes the disjoint union of measurable spaces. The transition kernel T canonically induces a path kernel $\mu _ { T } ^ { \mathrm { p a t h } } : X  S ^ { * } \times Y$ defined for any parameter $p \in P$ , initial state $x \in X$ , internal path $s \in S ^ { n }$ and output state $y \in Y$ as

$$
\mu _ { T } ^ { \mathrm { p a t h } } ( p , x , \mathrm { d } s , \mathrm { d } y ) : = { \left\{ \begin{array} { l l } { T ( p , x , \mathrm { d } y ) } & { { \mathrm { i f ~ } } n = 0 , } \\ { T ( p , x , \mathrm { d } s _ { 1 } ) \displaystyle \prod _ { i = 1 } ^ { n - 1 } T ( p , s _ { i } , \mathrm { d } s _ { i + 1 } ) T ( p , s _ { n } , \mathrm { d } y ) } & { { \mathrm { i f ~ } } n > 0 . } \end{array} \right. }
$$

Moreover, it induces an output (execution) kernel $\mu _ { T } ^ { \mathrm { o u t } } : X  Y$ , obtained by marginalizing over all finite internal paths

$$
\mu _ { T } ^ { \mathrm { o u t } } ( p , x , \mathrm { d } y ) : = \mu _ { T } ^ { \mathrm { p a t h } } ( p , x , S ^ { \ast } , \mathrm { d } y ) = \sum _ { n = 0 } ^ { \infty } \int _ { s \in S ^ { n } } \mu _ { T } ^ { \mathrm { p a t h } } ( p , x , \mathrm { d } s , \mathrm { d } y ) .
$$

Parametric Markov chains (pMC) are a special case of parametric signed Markov chains, such that the associated kernel is a parametric Markov kernel (in the non-extended sense; see Section 2.1).

Example 9. Consider the finite parametric Markov chain shown in Figure 2, with input space $X =$ $\{ * \}$ , internal states $S = \{ i , j \}$ , and output states $Y = \{ - 1 , + 1 \}$ . The transition kernel $T : \dot { X } + S \cdot$ ⇝ $S + Y$ is represented by the matrix

$$
T ( p , - , - ) = \left( \begin{array} { c c c c } { { 0 } } & { { 1 } } & { { 0 } } & { { 0 } } \\ { { 1 - p } } & { { { \frac { p } { 2 } } } } & { { { \frac { p } { 2 } } } } & { { 0 } } \\ { { 0 } } & { { 0 } } & { { p } } & { { 1 - p } } \end{array} \right) .
$$

Evaluating the sum of paths yields the output kernel $\mu _ { T } ^ { \mathrm { o u t } } : \{ * \}  \{ - 1 , + 1 \}$ given by

$$
\mu _ { T } ^ { \mathrm { o u t } } ( p , * , - ) = \frac { 1 + p } { 2 } \langle - 1 | + \frac { 1 - p } { 2 } \langle + 1 | .
$$

## 2.4 THE LOG-DERIVATIVE TRICK

Having formalized parametric Markov chains and their differentiation via signed kernels, we now address the computation of the directional derivative $\partial _ { v } \mu _ { T } ^ { \mathrm { o u t } } ( p , x , - )$ , which enables gradient-based optimization of expected rewards. In this section, we establish a measure-theoretic generalization of the log-derivative trick (Williams, 1992), expressing the Radon–Nikodym derivative of the path kernel $\mu _ { T } ^ { \mathrm { { \scriptsize { s a t h } } } }$ as an additive accumulation along paths.

Theorem 10 (Log-Derivative Trick). Let $T : X + S  S + Y$ be an absolutely differentiable parametric signed Markov chain over an open parameter space $P \subseteq \mathbb { R } ^ { k }$ . The path kernel $\mu _ { T } ^ { p a t h }$ is absolutely differentiable. Moreover, for all directional vectors $v \in \mathbb { R } ^ { k }$ , parameters $p \in P _ { : }$ , initial states $x \in X$ ,finite internal paths $s \in S ^ { n }$ , output states $y \in Y$ , and $n > 0$

$$
\frac { \mathrm { d } \partial _ { v } \mu _ { T } ^ { p a n h } ( p , x , - , - ) } { \mathrm { d } \mu _ { T } ^ { p a h } ( p , x , - , - ) } ( s , y ) = \frac { \mathrm { d } \partial _ { v } T ( p , x , - ) } { \mathrm { d } T ( p , x , - ) } ( s _ { 1 } ) + \sum _ { i = 1 } ^ { n - 1 } \frac { \mathrm { d } \partial _ { v } T ( p , s _ { i } , - ) } { \mathrm { d } T ( p , s _ { i } , - ) } ( s _ { i + 1 } ) + \frac { \mathrm { d } \partial _ { v } T ( p , s _ { n } , - ) } { \mathrm { d } T ( p , s _ { n } , - ) } ( y ) .
$$

For $n = 0$ (direct transition to an output state without visiting S), the expression reduces to

$$
\frac { \mathrm { d } \partial _ { v } \mu _ { T } ^ { p a t h } ( p , x , - , - ) } { \mathrm { d } \mu _ { T } ^ { p a t h } ( p , x , - , - ) } ( \varepsilon , y ) = \frac { \mathrm { d } \partial _ { v } T ( p , x , - ) } { \mathrm { d } T ( p , x , - ) } ( y ) .
$$

Proof. Let $n \geq 1$ . For clarity, we introduce the shorthand notations

$$
V _ { 0 } , V _ { 1 } , \ldots , V _ { n } : = T ( p , x , \mathrm { d } s _ { 1 } ) , T ( p , s _ { 1 } , \mathrm { d } s _ { 2 } ) , \ldots , T ( p , s _ { n } , \mathrm { d } y )
$$

$$
{ \cal D } _ { 0 } , { \cal D } _ { 1 } , \dots , { \cal D } _ { n } : = \frac { \mathrm { d } \partial _ { v } T ( p , x , - ) } { \mathrm { d } T ( p , x , - ) } ( s _ { 1 } ) , \frac { \mathrm { d } \partial _ { v } T ( p , s _ { 1 } , - ) } { \mathrm { d } T ( p , s _ { 1 } , - ) } ( s _ { 2 } ) , \dots , \frac { \mathrm { d } \partial _ { v } T ( p , s _ { n } , - ) } { \mathrm { d } T ( p , s _ { n } , - ) } ( y )
$$

By the definition of the Radon–Nikodym derivative, we have $\partial _ { v } V _ { i } = D _ { i } V _ { i }$ for all $i \in \{ 0 , \ldots , n \}$ Applying the Leibniz product rule, we obtain

$$
\begin{array} { r l } & { \partial _ { v } \mu _ { T } ^ { \mathrm { p a t h } } ( p , x , \mathrm { d } s , \mathrm { d } y ) = \partial _ { v } \left( V _ { 0 } \cdot \cdot \cdot V _ { n } \right) } \\ & { \qquad = \partial _ { v } V _ { 0 } V _ { 1 } \cdot \cdot \cdot V _ { n } + V _ { 0 } \partial _ { v } V _ { 1 } V _ { 2 } \cdot \cdot \cdot V _ { n } + \cdot \cdot \cdot + V _ { 0 } \cdot \cdot \cdot V _ { n - 1 } \partial _ { v } V _ { n } } \\ & { \qquad = D _ { 0 } V _ { 0 } V _ { 1 } \cdot \cdot \cdot V _ { n } + V _ { 0 } D _ { 1 } V _ { 1 } V _ { 2 } \cdot \cdot \cdot V _ { n } + \cdot \cdot \cdot + V _ { 0 } \cdot \cdot \cdot V _ { n - 1 } D _ { n } V _ { n } } \\ & { \qquad = ( D _ { 0 } + \cdot \cdot \cdot + D _ { n } ) \cdot V _ { 0 } \cdot \cdot \cdot V _ { n } } \\ & { \qquad = ( D _ { 0 } + \cdot \cdot \cdot + D _ { n } ) \cdot \mu _ { T } ^ { \mathrm { p a t h } } ( p , x , \mathrm { d } s , \mathrm { d } y ) . } \end{array}
$$

Since the sum $D _ { 0 } + \cdots + D _ { n }$ is measurable, it uniquely defines the Radon–Nikodym derivative

$$
\frac { \mathrm { d } \partial _ { v } \mu _ { T } ^ { \mathrm { p a t h } } ( p , x , - , - ) } { \mathrm { d } \mu _ { T } ^ { \mathrm { p a t h } } ( p , x , - , - ) } ( s , y ) = D _ { 0 } + \cdot \cdot \cdot + D _ { n } .
$$

The base case $n = 0$ follows immediately.

Corollary 11. Let T be an absolutely differentiable parametric signed Markov chain over $P \subseteq \mathbb { R } ^ { k }$ Under standard regularity conditions allowing the interchange of differentiation and summation (satisfied when paths terminate almost surely with a bounded expected length), we have for all $v \in \mathbb { R } ^ { k } , p \in P , x \in X$ and $y \in Y$

$$
\partial _ { v } \mu _ { T } ^ { o u t } ( p , x , \mathrm { d } y ) = \int _ { s \in S ^ { * } } \frac { \mathrm { d } \partial _ { v } \mu _ { T } ^ { p a t h } ( p , x , - , - ) } { \mathrm { d } \mu _ { T } ^ { p a t h } ( p , x , - , - ) } ( s , y ) \cdot \mu _ { T } ^ { p a t h } ( p , x , \mathrm { d } s , \mathrm { d } y ) .
$$

```latex
Algorithm 1 Log-Derivative Gradient Sampler for pMCs
Input: pMC $T : X + S  S + Y$ , parameter $p \in P \subseteq \mathbb { R } ^ { k }$ , initial state $x \in X$
Output: Single weighted sample $( w _ { \mathrm { t o t a l } } , y )$ , with $w _ { \mathrm { t o t a l } } \in \mathbb { R } ^ { k }$ and $y \in Y ,$ , whose empirical mea
sure w<sub>total</sub> ⟨y| satisfies, for each $A \in \mathcal { F } _ { Y } , \mathbb { E } \big [ w _ { \mathrm { t o t a l } } \cdot { \mathbf { 1 } } _ { A } ( y ) \big ] = \nabla _ { p } \mu _ { T } ^ { \mathrm { o u t } } ( p , x , A )$
1: $s \gets x$
2: $w _ { \mathrm { t o t a l } }  \mathbf { 0 } _ { \mathbb { R } ^ { k } }$
3: while $s \not \in Y$ do
4: Sample next state $s ^ { \prime } \sim T ( p , s , - )$
5: $w _ { \mathrm { t o t a l } }  w _ { \mathrm { t o t a l } } + w _ { T } ( p , s , s ^ { \prime } )$ ▷ w is from Definition 12
6: $s \gets s ^ { \prime }$
7: end while
8: $y  s$ $\triangleright y \in Y$ is the output state
9: return $( w _ { \mathrm { t o t a l } } , y )$
```

Proof. Differentiating $\mu _ { T } ^ { \mathrm { o u t } }$ with respect to p along direction v and exchanging differentiation and integration via the dominated convergence theorem yields

$$
\begin{array} { r l } & { \partial _ { v } \mu _ { T } ^ { \mathrm { o u t } } ( p , x , \mathrm { d } y ) = \partial _ { v } \int _ { s \in S ^ { \ast } } \mu _ { T } ^ { \mathrm { p a t h } } ( p , x , \mathrm { d } s , \mathrm { d } y ) } \\ & { \qquad = \displaystyle \int _ { s \in S ^ { \ast } } \partial _ { v } \mu _ { T } ^ { \mathrm { p a t h } } ( p , x , \mathrm { d } s , \mathrm { d } y ) } \\ & { \qquad = \displaystyle \int _ { s \in S ^ { \ast } } \frac { \mathrm { d } \partial _ { v } \mu _ { T } ^ { \mathrm { p a t h } } ( p , x , - , - ) } { \mathrm { d } \mu _ { T } ^ { \mathrm { p a t h } } ( p , x , - , - ) } ( s , y ) \cdot \mu _ { T } ^ { \mathrm { p a t h } } ( p , x , \mathrm { d } s , \mathrm { d } y ) . } \end{array}
$$

## 2.5 GRADIENT SAMPLING AND OPTIMIZATION

Throughout this section, let $T : X + S  S + Y$ be an absolutely differentiable parametric Markov chain—in the restricted sense at the end of Definition 8—over an open parameter space $P \subseteq \mathbb { R } ^ { k }$ Here, each $T ( p , a , - )$ is a probability measure that can be sampled from, while its partial derivatives $\partial _ { i } T$ form parametric signed Markov chains. Leveraging the log-derivative identity from Corollary 11, we construct a Monte Carlo algorithm to estimate the gradient measure

$$
\nabla _ { p } \mu _ { T } ^ { \mathrm { o u t } } ( p , x , - ) : = \left( \partial _ { 1 } \mu _ { T } ^ { \mathrm { o u t } } ( p , x , - ) , \dots , \partial _ { k } \mu _ { T } ^ { \mathrm { o u t } } ( p , x , - ) \right)
$$

at any initial state $x \in X$ and parameter $p \in P \subseteq \mathbb { R } ^ { k }$ without solving linear systems or constructing the full state-space transition matrix.

Definition 12 (Transition Gradient Vector). Let $T : X + S  S + Y$ be an absolutely differentiable parametric Markov chain over an open parameter space $P \subseteq \mathbb { R } ^ { k }$ . The transition gradient vector w<sub>T</sub> $: P \times ( X + S ) \times ( S + Y ) \to \bar { \mathbb { R } } ^ { k }$ is defined by

$$
w _ { T } ( p , a , b ) : = \Big ( \frac { \mathrm { d } \partial _ { 1 } T ( p , a , - ) } { \mathrm { d } T ( p , a , - ) } ( b ) , . . . , \frac { \mathrm { d } \partial _ { k } T ( p , a , - ) } { \mathrm { d } T ( p , a , - ) } ( b ) \Big ) .
$$

Our algorithm is in Algorithm 1. At each step, it incrementally accumulates in $w _ { \mathrm { t o t a l } }$ the local transition gradient vectors $w _ { T } ( p , s , s ^ { \prime } )$ , reproducing the path sum in Theorem 10. The returned pair $( w _ { \mathrm { t o t a l } } , y )$ defines a random signed measure $w _ { \mathrm { t o t a l } } \langle y | ;$ ; taking its expectation over multiple sampled trajectories yields an unbiased estimate of the true gradient measure $\nabla _ { p } \mu _ { T } ^ { \mathrm { o u t } } ( p , x , - )$ , with correctness and unbiasedness following directly from Corollary 11.

In particular, given a terminal reward function $R : Y  \mathbb { R } .$ , an unbiased estimate of the expected reward gradient $\nabla _ { p } \mathbb { E } _ { y \sim \mu _ { T } ^ { \mathrm { o u t } } ( p , x , - ) } [ R ( y ) ]$ is directly obtained as $w _ { \mathrm { t o t a l } } \cdot R ( y )$ . While this framework extends directly to cumulative and parameter-dependent transition rewards, we evaluate our algorithm under this terminal reward setting in the subsequent sections, presenting our experimental methodology and empirical performance against other approaches.

## 3 EMPIRICAL EVALUATION

Given a pMC and a reward function defined over its output states, our goal is to synthesize a parameter valuation that maximizes the expected reward.

The two main approaches to tackle this optimization problem are symbolic synthesis such as parameter lifting, rational function generation and feasibility region decomposition (Spel et al., 2021; Dehnert et al., 2015; Heck et al., 2025), with global optimality guarantees but that do not scale beyond dozens of parameters, and gradient descent (based on gradient computation algorithms), with no global optimality guarantees but that scales to higher-dimensional parameter space.

Gradient computation algorithms themselves are subdivided into exact methods such as the linear solver method (Heck et al., 2022) and approximate methods such as our Monte Carlo approach (Algorithm 1) and the Jacobi method.

Since our synthetic benchmark involved high-dimensional parameter spaces, to evaluate the empirical efficiency and precision of our proposed sampling-based framework, we compare it with other gradient computation techniques for gradient computation and gradient descent parameter synthesis.

• SparseLU and BiCGSTAB: Exact linear solver methods for gradient computation, following Heck et al. (2022). For solving the linear system (1), we evaluate both a direct sparse LU factorization (SparseLU) and an iterative Krylov subspace method (BiCGSTAB).

• Jacobi: Approximates the gradient by unrolling the linear system (1) for a fixed budget of K matrix-vector multiplication steps, effectively truncating the computation to a finite horizon.

• LogDerivative (Ours): Our proposed sampling-based estimator, which computes an unbiased gradient estimate by simulating a budget of N paths using the path-space log-derivative trick.

• UniformPath: An importance-sampling baseline where successor states are drawn uniformly at random regardless of transition probabilities, parameterized by the number of sampled paths $N .$ Specifically, given d outgoing transitions with probabilities $p _ { 1 } , \ldots , p _ { d } .$ branch i is traversed with uniform probability $\textstyle { \frac { 1 } { d } }$ and reweighted by $d \cdot p _ { i }$ to maintain unbiasedness.

Benchmarks We evaluate our approach on both synthetic discrete pMCs and continuous-state benchmarks. For the discrete setting, we generate synthetic pMCs using a procedural generator in Python. Each model fixes an underlying sparse graph topology and assigns to each transition a degree-1 polynomial in a single parameter $\bar { X _ { i } } \bar { \in \{ X _ { 1 } , \ldots , X _ { k } \} }$ . The transitions are constructed such that outgoing probabilities algebraically sum to 1 and evaluate to values in (0, 1) around an initial parameter value $p _ { 0 }$ . Sequential and cyclic paths naturally compose these local degree-1 transitions into non-trivial multivariate rational functions. The generated models are exported as PRISM files; full details of the generation pipeline are provided in Appendix A. Continuous-state case studies (such as the model in Figure 3) are described in Figure 9.

Implementation All code, benchmarks, and experimental scripts are open-source and publicly available at https://github.com/Gallinou/pMC-optimization-artifact. All gradient computation methods along with the optimizers are implemented in C++ using sparse data structures. To ensure a fair runtime comparison under identical data layouts, we implemented the exact baseline solvers (SparseLU, BiCGSTAB) directly within this unified C++ framework.

Symbolic differentiation and evaluation of individual parametric transition expressions is handled using SymEngine<sup>1</sup>, whereas sparse linear systems are solved using Eigen::Sparse<sup>2</sup> (utilizing both SparseLU and BiCGSTAB). All experiments were executed on an AWS EC2 m6i.large instance (x86 64 architecture, 2 vCPUs powered by an Intel Xeon Platinum 8375C @ 3.5 GHz, 8 GiB RAM) constrained to a single core to ensure strictly deterministic, single-threaded execution.

RQ1: What is the Speed and Precision of Our Algorithm in Gradient Estimation? The primary motivation for our Monte Carlo sampler is to trade absolute precision for computational effi ciency. We therefore evaluate: (1) whether it is faster than exact linear solver methods (SparseLU, BiCGSTAB), and (2) whether it remains sufficiently accurate, yielding gradient estimates close to the exact solutions. Moreover, because computing the gradient scales poorly with the parameter dimension $k ,$ we assess runtime behavior as k increases.

![](images/c492f4645f583551ced4259d42faa085cef115a2fe1617be0c9925eb26d39add.jpg)  
Figure 4: Experiments for RQ1: impact of the number of parameters k on gradient estimation runtime, and average precision. Values are averaged over 25 synthetic pMC benchmarks, each generated with 10 times more states than parameters.

Figure 4 summarizes these results. Here, execution time denotes the duration required to compute the gradient at the initial parameter value $p _ { 0 }$ . As shown in the left plot, runtime increases for all methods as the number of parameters grows. The center plot displays runtime normalized against SparseLU as a baseline, showing a clear performance advantage for the sampling-based methods (LogDerivative and UniformPath) over the linear solver methods (SparseLU, BiCGSTAB, Jacobi), an advantage that widens considerably for larger parameter space dimensions.

The right-hand table reports the accuracy of each method. Because SparseLU produces the exact gradient, we use its output as the ground truth $g$ and compare it against the estimated gradient $g ^ { \prime }$ produced by each method using two different metrics:

• The Normalized Distance (d) measures the relative error in both magnitude and direction:

$$
d ( g , g ^ { \prime } ) : = \frac { \| g - g ^ { \prime } \| _ { 2 } } { \| g \| _ { 2 } } .
$$

It penalizes scale mismatch and angular divergence (lower is better).

• The Cosine Similarity (cos) evaluates the directional alignment irrespective of scale:

$$
\cos ( g , g ^ { \prime } ) : = { \frac { \langle g , g ^ { \prime } \rangle } { \| g \| _ { 2 } \| g ^ { \prime } \| _ { 2 } } } \in [ - 1 , 1 ] .
$$

Here, 1 denotes identical direction (higher is better).

Between the two sampling-based methods, LogDerivative consistently outperforms the precision of UniformPath. In particular, with $1 0 ^ { 5 }$ sample paths, LogDerivative achieves a normalized distance of 0.050 and a cosine similarity of 0.998, demonstrating near-perfect directional alignment and high precision relative to the exact gradient.

RQ2: What is the Overall Performance of Our Algorithm in Parameter Synthesis? Since our benchmark generates pMCs with high-dimensional parameter spaces, for which symbolic methods do not scale well, we compare the gradient computation methods—SparseLU, BiCGSTAB, LogDerivative, UniformPath and Jacobi— at gradient-based parameter synthesis. Explicitly, we compare the reward improvement over execution time of full optimization runs across 100 benchmark pMCs with 5,000 states and 100 parameters under a strict 100-second wall-clock timeout.

![](images/c5ff4ca1a1c7e9f36c28423161e094a309e87728842176e5ef66346631b38e54.jpg)  
Figure 5: Experiments for RQ2: gradient descent with a strict 100-second timeout, averaged over 100 pMCs with 5,000 states and 100 parameters, comparing optimizers.

Since the result is averaged over 100 pMCs, we introduce the notion of normalized relative reward $\rho .$ Given an initial expected reward $r _ { 0 } ,$ , final expected rewards $\{ r _ { m } \} _ { m \in M }$ achieved across all evaluated methods $m \in M$ , and the evaluated expected reward $r , \rho ( r )$ is defined as

$$
\rho ( r ) : = \frac { r - r _ { 0 } } { \operatorname* { m a x } _ { m \in M } ( r _ { m } - r _ { 0 } ) }
$$

Here, execution time accounts strictly for gradient estimation and optimizer steps. It excludes preprocessing (such as parsing PRISM files and building internal sparse representations) and postprocessing (computing the exact reachability reward after each update via $\overset { \cdot } { x } = ( I - A ) ^ { - 1 } b$ from (1)).

Because parameter values must remain within valid probability intervals (such as $[ 0 , 1 ] )$ , standard unconstrained optimizers fail: as shown in Figure 5, AdaGrad, Adam, and AdamW quickly drive parameters outside valid bounds and terminate within seconds. Standard SGD and step-clamped SGD (SmallSGD) remain inside the valid domain but make very slow progress. In contrast, constrained variants—C-AdaGrad and C-AdamW (with default learning rates $1 \mathrm { { 0 } ^ { - 2 } }$ and 10<sup>−3</sup>)—consistently achieve fast reward improvement for both exact (SparseLU) and sampling-based (LogDerivative) gradients, despite projecting values and no longer following the exact gradient direction. We therefore use C-AdaGrad and C-AdamW as the underlying optimizers for our comparison.

Figure 6 depicts the cumulative normalized relative reward improvement over time across the different exact and approximate gradient estimators. LogDerivative substantially outperforms exact linear solver methods (SparseLU, BiCGSTAB) and the sampling baseline UniformPath. While exact solvers compute true gradients, each step is slow: within 100 seconds, they execute very few steps and achieve minimal cumulative reward gain. Conversely, while UniformPath executes quickly, its precision ceiling (observed in RQ3) limits its progress.

Crucially, low-sample configurations of our estimator (LogDerivative with $N \in \{ 1 0 , 1 0 ^ { 2 } \} )$ achieve the fastest initial reward gains. This illustrates a classic trade-off: “the drunk guy will get home before the wise, though he might get stuck in front of his door.” With small N, the noisy updates are cheap enough to take hundreds of steps and quickly get close to the best reward, even if the noise prevents them from reaching the exact optimum. Larger sample sizes take fewer steps, but their lower noise allows them to get closer to the optimum.

RQ3: How Does the Number of Sampled Paths N Affect the Performance of Gradient Estimation? The two sampling-based methods (our proposed LogDerivative and the baseline Uniform-Path) rely on the number of sampled paths N as their primary hyperparameter. We evaluate the sensitivity of gradient estimation runtime and precision for different values of N.

As shown in Figure 7, the execution time of both sampling methods exhibits strongly sublinear growth with respect to the sample size N. Because transition derivatives are computed once and cached, increasing N by five orders of magnitude (from 10 to 10<sup>6</sup>) results in less than a two-ordersof-magnitude increase in runtime across all benchmark sizes.

![](images/0e47c186300b377078cc118358fc4c30499aab38e6b0aa556cb53eb8117e9700.jpg)

Figure 6: Experiments for RQ2: gradient descent with a strict 100-second timeout, averaged over 100 pMCs with 5,000 states and 100 parameters, comparing gradient computation methods.  
![](images/822eb87ddd30b2ff4c53acd59891a1371fa8d18b8b50559655dd713a8f6d610e.jpg)  
Figure 7: Experiments for RQ3: impact of the sample size N on gradient estimation runtime, cosine similarity, and normalized distance. Results are averaged over 100 pMC benchmarks.

Regarding estimation quality evaluated against the exact SparseLU solution via normalized distance and cosine similarity, UniformPath can be slightly more precise for very small N, presumably because uniform sampling may hit rare yet high-impact paths early on. However, despite its unbiasedness, its precision quickly plateaus whereas LogDerivative exhibits no such ceiling and steadily converges to the exact gradient as N increases.

RQ4: How Does the Number of States Affect Performance? Figure 8 illustrates scalability as the state space grows from 1 to over $2 . 6 \times 1 0 ^ { 5 }$ states. While exact numerical solvers such as SparseLU suffer sharply from state-space explosion—exhibiting steep, polynomial runtime growth exceeding 200 seconds at $2 . 6 \times 1 0 ^ { 5 }$ states—LogDerivative scales far more favorably, since its computational cost depends only on the average time required to simulate a path rather than on the total number of states or transitions. Furthermore, as shown in the accompanying table, LogDerivative maintains high estimation quality across these benchmarks, achieving a cosine similarity of 1.000 (normalized distance 0.013) at $\dot { N } = 1 0 ^ { 6 }$

![](images/078d113b65f9617f8fe0d224806c2309ed5eb9135632a5ffced0deb3ab2cfc80.jpg)  
Figure 8: Experiments for RQ4: Execution time and gradient approximation precision across varying state-space sizes. Values are averaged over 25 synthetic pMC benchmarks, each with $1 0 ^ { 2 }$ parameters.

RQ5: Can Our Algorithm Handle pMCs with Continuous State Spaces? Our theory, as well as our Algorithm 1, supports continuous pMCs. Here we experimentally confirm the algorithm’s capability. We evaluate our algorithm on two continuous pMC benchmarks: $T _ { 3 }$ (the mixed discrete– continuous pMC from Figure 3) with a single parameter p, and a two-parameter continuous pMC $T _ { 4 }$ with parameters $( p , q )$ . To monitor optimization progress and verify optimality, we derived closed form analytical expressions for the exact reachability rewards in both models.

The graphs in Figure 9 show that, indeed, our algorithm can optimize in continuous pMCs effectively and moreover converges to the theoretical maximum reward, respectively $2 { \sqrt { 2 } } { \stackrel { - } { - } } 3 { \mathrm { ~ a t ~ } } p = { \textstyle \frac { 1 } { 2 } }$ and $\begin{array} { r } { \frac { 1 } { e } - 1 \mathrm { a t } ( p , q ) = \bigl ( \frac { 1 } { e } , \overline { { 1 } } \bigr ) } \end{array}$ , using a constrained-AdamW optimizer. Note that existing methods such as symbolic synthesis (Spel et al., 2021; Dehnert et al., 2015; Heck et al., 2025) and exact linear solver methods (Heck et al., 2022) require finite matrix representations and are not directly applicable to infinite or continuous state spaces.

## 4 CONCLUSIONS AND FUTURE WORK

To the best of our knowledge, this establishes the first fully trajectory-sampling framework specifically tailored for verifying and optimizing terminating pMCs over general measurable state spaces. By leveraging the log-derivative trick and generalizing it via Radon–Nikodym derivatives, our approach naturally accommodates both discrete and continuous state spaces with formal unbiasedness guarantees. Empirically, our results distinguish two key benefits: first, for gradient computation, our estimator yields accurate gradient approximations orders of magnitude faster than exact linear solvers; second, for end-to-end optimization, this speed–precision trade-off accelerates gradient descent, enabling rapid parameter synthesis on models far beyond the reach of exact techniques.

Future work Because our estimator operates entirely on sampled execution traces without ever materializing this global state space, it provides a natural and scalable foundation for optimizing probabilistic programs and their compositions. On the foundational side, we also aim to develop a high-level, structural understanding of this framework, characterizing path-based gradient estimation through abstract categorical and compositional semantics. On the practical side, we plan to explore applications in machine learning, such as discrete latent-variable architectures (e.g., discrete VAEs) and recurrent stochastic models. Furthermore, because sampled paths are independent, the algorithm is naturally amenable to massive GPU parallelization, which could drastically accelerate computation.

![](images/167532472dae605d0d9c365af4e998dc62b238993cf6015696880fef3afffdd0.jpg)

![](images/06f1072a9cc4bc56f08bd082338bdbafd4fd253e68d49c90812b31835ca44811.jpg)

![](images/5c9b31511c28e283593acb43dc3eda9772d49e7037fba83d10a1109b5ae21fb0.jpg)

![](images/ab954aeb63c406d4f9f0b863782ce17d402b0e103012e89d6106e745e7913f3c.jpg)  
Figure 9: Evaluation for RQ5: Gradient-based optimization on continuous pMCs $( 1 0 ^ { 4 }$ path samples per step over $1 0 ^ { 3 }$ iterations). (a) Benchmark $T _ { 3 }$ initialized at $p = 1 ;$ (b) Benchmark $T _ { 4 }$ initialized at $( p , \bar { q } ) \ : = \ : ( 0 . 5 , 0 . 5 )$ . The analytically computed optimal rewards are indicated by closed-form expressions in bold.

Unlike prevailing automatic differentiation techniques that depend on continuous relaxations or reparameterization tricks to handle discrete randomness, our score-function approach requires no reparameterization and natively accommodates stochastic branching (if-statements) and loops (while-statements).

## AI USE STATEMENT

We used generative AI tools solely to provide minor support for code optimization and manuscript polishing. We did not use AI tools to develop theoretical models, formulate mathematical claims, write proofs, or design experimental methodology. All generated suggestions were verified by the authors, and we take full responsibility for the entirety of the text, code, and theoretical results presented in this work.

## ACKNOWLEDGMENT

This work was supported by JST SPRING, Japan Grant Number JPMJSP2104.

## REFERENCES

Thom Badings, Sebastian Junges, Ahmadreza Marandi, Ufuk Topcu, and Nils Jansen. Efficient sensitivity analysis for parametric robust Markov chains. In Computer Aided Verification (CAV), volume 13966 of LNCS, pp. 62–85. Springer, 2023. doi: 10.1007/978-3-031-37709-9 4.

Christel Baier and Joost-Pieter Katoen. Principles ofModel Checking. The MIT Press, 2008. ISBN 026202649X, 9780262026499.

Gilles Barthe, Boris Kopf, Federico Olmedo, and Santiago Zanella-B¨ eguelin. Probabilistic relational´ reasoning for differential privacy. In Principles of Programming Languages (POPL), pp. 97–110. ACM, 2012. doi: 10.1145/2103656.2103670.

Gilles Barthe, Joost-Pieter Katoen, and Alexandra Silva (eds.). Foundations of Probabilistic Programming. Cambridge University Press, 2020. doi: 10.1017/9781108770750.

Ezio Bartocci, Radu Grosu, Panagiotis Katsaros, C. R. Ramakrishnan, and Scott A. Smolka. Model repair for probabilistic systems. In Tools and Algorithms for the Construction and Analysis of Systems (TACAS), volume 6605 of LNCS, pp. 326–340. Springer, 2011. doi: 10.1007/978-3-642 -19835-9 30.

J. Baxter and P. L. Bartlett. Infinite-horizon policy-gradient estimation. Journal ofArtificial Intelligence Research, 15:319–350, November 2001. ISSN 1076-9757. doi: 10.1613/jair.806. URL http://dx.doi.org/10.1613/jair.806.

Eli Bingham, Jonathan P. Chen, Martin Jankowiak, Fritz Obermeyer, Neeraj Pradhan, Theofanis Karaletsos, Rohit Singh, Paul Szerlip, Paul Horsfall, and Noah D. Goodman. Pyro: Deep universal probabilistic programming. Journal of Machine Learning Research, 20(28):1–6, 2019.

Michael Carbin, Sasa Misailovic, and Martin C. Rinard. Verifying quantitative reliability for programs that execute on unreliable hardware. In Object-Oriented Programming, Systems, Languages & Applications (OOPSLA), pp. 33–52. ACM, 2013. doi: 10.1145/2509136.2509546.

Milan Ce<sup>ˇ</sup> ska, Frits Dannenberg, Nicola Paoletti, Marta Kwiatkowska, and Lubo ˇ s Brim. Precise ˇ parameter synthesis for stochastic biochemical systems. Acta Informatica, 54(6):589–623, 2017. doi: 10.1007/s00236-016-0265-2.

Murat Cubuktepe, Nils Jansen, Sebastian Junges, Joost-Pieter Katoen, and Ufuk Topcu. Convex optimization for parameter synthesis in MDPs. IEEE Transactions on Automatic Control, 67(12): 6333–6348, 2022. doi: 10.1109/TAC.2021.3133265.

Timothy A. Davis. Direct Methodsfor Sparse Linear Systems. Fundamentals of Algorithms. SIAM, 2006. doi: 10.1137/1.9780898718881.

Conrado Daws. Symbolic and parametric model checking of discrete-time Markov chains. In Theoretical Aspects ofComputing (ICTAC 2004), volume 3407 of LNCS, pp. 280–294. Springer, 2005. doi: 10.1007/978-3-540-31862-0 21.

Christian Dehnert, Sebastian Junges, Nils Jansen, Florian Corzilius, Matthias Volk, Harold Bruintjes, Joost-Pieter Katoen, and Erika Abrah<sup>´</sup> am. PROPhESY: A PRObabilistic ParamEter SYnthesis´ tool. In Computer Aided Verification (CAV), volume 9206 of LNCS, pp. 214–231. Springer, 2015. doi: 10.1007/978-3-319-21690-4 13.

Linus Heck, Jip Spel, Sebastian Junges, Joshua Moerman, and Joost-Pieter Katoen. Gradientdescent for randomized controllers under partial observability. In Bernd Finkbeiner and Thomas Wies (eds.), Verification, Model Checking, and Abstract Interpretation, pp. 127–150, Cham, 2022. Springer International Publishing. ISBN 978-3-030-94583-1.

Linus Heck, Tim Quatmann, Jip Spel, Joost-Pieter Katoen, and Sebastian Junges. Generalized parameter lifting: Finer abstractions for parametric Markov chains. In Automated Technologyfor Verification and Analysis (ATVA), LNCS, pp. 207–230. Springer, 2025. doi: 10.1007/978-3-032 -08707-2 10.

Sebastian Junges, Nils Jansen, Ralf Wimmer, Tim Quatmann, Leonore Winterer, Joost-Pieter Katoen, and Bernd Becker. Finite-state controllers of POMDPs using parameter synthesis. In Uncertainty in Artificial Intelligence (UAI), pp. 519–529. AUAI Press, 2018.

Sebastian Junges, Erika Abrah <sup>´</sup> am, Christian Hensel, Nils Jansen, Joost-Pieter Katoen, Tim Quat-´ mann, and Matthias Volk. Parameter synthesis for Markov models: Covering the parameter space. Formal Methods in System Design, 62(1):181–259, 2024. doi: 10.1007/s10703-023-00442-x.

Alexander K. Lew, Mathieu Huot, Sam Staton, and Vikash K. Mansinghka. ADEV: Sound automatic differentiation of expected values of probabilistic programs. Proceedings of the ACM on Programming Languages, 7(POPL):121–153, 2023. doi: 10.1145/3571198.

P. Marbach and J.N. Tsitsiklis. Simulation-based optimization of markov reward processes. IEEE Transactions on Automatic Control, 46(2):191–209, 2001. doi: 10.1109/9.905687.

Shakir Mohamed, Mihaela Rosca, Michael Figurnov, and Andriy Mnih. Monte carlo gradient estimation in machine learning, 2020. URL https://arxiv.org/abs/1906.10652.

Elizabeth Polgreen, Viraj B. Wijesuriya, Sofie Haesaert, and Alessandro Abate. Data-efficient Bayesian verification of parametric Markov chains. In Quantitative Evaluation of Systems (QEST), volume 9826 of LNCS, pp. 35–51. Springer, 2016. doi: 10.1007/978-3-319-43425-4 3.

Subhajit Roy, Justin Hsu, and Aws Albarghouthi. Learning differentially private mechanisms. In IEEE Symposium on Security and Privacy (S&P), pp. 852–865. IEEE, 2021. doi: 10.1109/SP40 001.2021.00060.

Yousef Saad. Iterative Methods for Sparse Linear Systems. SIAM, 2nd edition, 2003. doi: 10.113 7/1.9780898718003.

Jip Spel, Sebastian Junges, and Joost-Pieter Katoen. Finding provably optimal Markov chains. In Tools and Algorithms for the Construction and Analysis of Systems (TACAS), volume 12651 of LNCS, pp. 173–190. Springer, 2021. doi: 10.1007/978-3-030-72016-2 10.

Jan-Willem van de Meent, Brooks Paige, Hongseok Yang, and Frank Wood. An introduction to probabilistic programming. arXiv preprint arXiv:1809.10756, 2018.

Matthias Volk, Sebastian Junges, and Joost-Pieter Katoen. Advancing dynamic fault tree analysis – get succinct state spaces fast and synthesise failure rates. In Computer Safety, Reliability, and Security (SAFECOMP), volume 9922 of LNCS, pp. 253–265. Springer, 2016. doi: 10.1007/97 8-3-319-45477-1 20.

Di Wang, Jan Hoffmann, and Thomas Reps. Central moment analysis for cost accumulators in probabilistic programs. In Programming Language Design and Implementation (PLDI), pp. 559– 573. ACM, 2021. doi: 10.1145/3453483.3454062.

Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3):229–256, 1992.

## A BENCHMARK GENERATION

We developed a scalable benchmark generator for pMCs. The generation proceeds in two steps:

1. Topology Generation: We construct a finite, sparse directed graph ensuring that every state is accessible from the initial state and co-accessible to at least one output state. To scale up to large state spaces $( > 1 0 ^ { 5 }$ states), with non-trivial cyclic behavior, we generate modular sparse sub graphs with cycles, duplicate them, and link them with directed bridging transitions to guarantee global connectivity.

2. Parametric Transitions: For every state s, outgoing transition probabilities $\{ { P _ { s , s ^ { \prime } } } \} _ { s ^ { \prime } }$ are parameterized by affine (degree-1) univariate polynomials over a parameter $X _ { i } \in \{ X _ { 1 } , \ldots , X _ { k } \}$ These polynomials are constructed such that their formal sum satisfies $\begin{array} { r } { \sum _ { s ^ { \prime } } P _ { s , s ^ { \prime } } \ ' ( X _ { i } ) = 1 } \end{array}$ . For a designated initial parameter point $p _ { 0 }$ , we ensure $P _ { s , s ^ { \prime } } ( p _ { 0 } ) \in ( 0 , 1 )$ for all transitions. By continuity, there exists an open neighborhood $B ( p _ { 0 } , \varepsilon )$ where all transition probabilities remain strictly in (0, 1), ensuring the pMC remains well-defined and differentiable under local gradient updates.

Although individual transitions use degree-1 polynomials over a single parameter, paths through cycles and branches combine these transitions into high-degree multivariate rational functions for reachability probabilities and expected rewards. Our generator is implemented in Python and scales to models with $1 0 ^ { 5 }$ states and millions of transitions in seconds. All generated models are exported as PRISM files, ensuring that all tools evaluate the exact same benchmarks.