# Learning Rate Decay Can Exponentially Accelerate SGD for Global Optimization of Nonconvex Functions

Guneykan Ozgul<sup>∗</sup> Dylan Herman Junhyung Lyle Kim Shree Hari Sureshbabu Shouvanik Chakrabarti<sup>†</sup> Global Technology Applied Research, JPMorganChase New York, NY 10001, USA

## Abstract

Stochastic Gradient Descent (SGD) is a workhorse algorithm for continuous optimization. It is a folk belief among deep learning practitioners that learning rate decay in SGD substantially improves performance. However, it has remained unclear whether learning rate decay offers an asymptotic advantage for global nonconvex optimization. We answer this question in the affirmative by demonstrating a natural class of nonconvex functions for which, we prove that SGD with learning rate decay requires exponentially fewer gradient queries for global optimization than SGD with any fixed learning rate in the small-step size regime. To the best of our knowledge, this is the first exponential advantage that has been demonstrated for learning rate decay. The class of functions we consider includes many popular benchmark nonconvex functions. Our results provide a robust explanatory theory for the empirically observed benefits of learning rate decay for global optimization. We show that annealing performs the non-logconcave sampling task associated to the optimization problem in polynomial time, via a novel application of weak Poincaré inequalities.

## 1 Introduction

Minimizing nonconvex objective functions to global optimality is a central challenge in modern machine learning and scientific computing. From training deep neural networks [20] to solving inverse problems in the physical sciences [53], practitioners routinely face optimization landscapes with spurious local minima, saddle points, and flat regions that impede first-order methods such as gradient descent and its variants. Given a continuous nonconvex objective function $f : \mathbb { R } ^ { d }  \mathbb { R }$ , we consider the problem of finding an approximate global optimum $\widetilde { x } \in \mathbb { R } ^ { d }$ satisfying

$$
f ( \widetilde { x } ) - f ^ { \star } \leqslant \epsilon ,
$$

where $\begin{array} { r } { f ^ { \star } = \operatorname* { m i n } _ { x \in \mathbb { R } ^ { d } } f ( x ) } \end{array}$ . This problem is fundamentally harder than finding approximate stationary points [40, 54], for which gradient-based methods perform well. Despite the pessimistic worst-case complexity of first-order methods for nonconvex optimization [4, 36], stochastic algorithms equipped with appropriate sources of randomness have demonstrated empirical success, motivating a rich line of theoretical inquiry into when and why noise-driven methods can efficiently navigate complex landscapes [7, 27, 28]. The workhorse algorithm in this setting is stochastic gradient descent (SGD) [1, 9, 46], which updates iterates as

$$
\boldsymbol { x } _ { t + 1 } = \boldsymbol { x } _ { t } - \eta _ { t } \nabla f ( \boldsymbol { x } _ { t } ) + \eta _ { t } \boldsymbol { \xi } _ { t } ,
$$

where $\eta _ { t } > 0$ is the learning rate and $\xi _ { t }$ is gradient noise arising either from minibatch approximation or intentional injection. The noise is precisely what gives SGD the ability to escape local minima and explore the landscape, making it a natural candidate for global optimization despite being a first-order method.

Unfortunately, for functions with non-trivial energy barriers between local and global minima, constant-step-size SGD requires exponential time. Azizian et al. [4] recently established tight bounds showing that the expected hitting time of fixed-learning-rate SGD to the global minimum scales as $\exp ( \mathsf { \bar { \Theta } } ( E / \eta ) )$ , where E is the depth of the deepest spurious local minimum along the worstcase path and $\eta$ is the step size. For learning rates η of order $\mathcal { O } ( \epsilon / d )$ chosen so that the iterates concentrate near the global minimum [24, 50], this yields a runtime exponential in both dimension and precision. Similar conclusions follow from the analysis of Shi et al. [50]. In this step-size regime, sub-exponential time is possible when all spurious minima are extremely shallow, scaling as $\mathcal { O } ( \epsilon / d )$ or under other restrictive conditions such as the PL-inequality that are common proxies for convexity in optimization theory [7, 41]. For interesting nonconvex landscapes with barriers of constant depth, constant-step-size SGD is provably intractable.

It is a folklore belief among practitioners that decaying the learning rate over the course of training substantially improves optimization: large learning rates explore broadly while small learning rates concentrate near optima. This schedule is universally deployed in deep learning, yet it has remained unclear whether learning rate decay offers a provable asymptotic advantage for global nonconvex optimization. Crucially, the lower bound of Azizian et al. [4] applies only to fixed-step-size SGD. An annealing schedule that varies the learning rate over time could, in principle, circumvent this barrier entirely. This raises the central question we address:

Can a simple, structure-agnostic learning rate schedule provably reduce the asymptotic runtime of SGD compared to afixed learning rate?

We answer this question in the affirmative. We prove that for a natural class of block-separable nonconvex functions, including multi-well potentials with locally strongly convex basins separated by constant-height barriers, and satisfying Theorem C.2, SGD with an arithmetically decaying learning rate finds an ϵ-optimal solution with high probability in poly $( d , 1 / \epsilon )$ time, whereas any fixed learning rate $\eta = \mathcal { O } ( \epsilon / \bar { d } )$ provably requires $\exp ( \Omega ( d ) )$ time on the same instances. To our knowledge, this is the first polynomial-versus-exponential separation demonstrated for learning rate decay in nonconvex optimization for SGD.

## 1.1 Main Result

Our theoretical results are about the following class of functions.

Definition 1.1 (Block-Separable Function). A function $f : \mathcal { X } \to \mathbb { R } , \mathcal { X } \subseteq \mathbb { R } ^ { d } ,$ is block-separable if there exist an orthogonal matrix $R \in { \mathcal { O } } ( d )$ , a function $g : \mathcal { X }  \mathbb { R }$ , an integer $k \leqslant d ,$ and a partition of coordinates into blocks $\hat { x } _ { 1 } , \ldots , \hat { x } _ { k }$ of sizes $d _ { 1 } , \ldots , d _ { k }$ such that $f ( x ) \ = \ g ( R x )$ $\begin{array} { r } { g ( x ) = \sum _ { i = 1 } ^ { k } g _ { i } ( \hat { x } _ { i } ) } \end{array}$ , where $g _ { i } : \mathcal { X } _ { i }  \mathbb { R }$ and $\textstyle \sum _ { i = 1 } ^ { k } d _ { i } = d ,$ , and the number ofcoordinates in each block is bounded above by a universal constant. $\dot { I } f \bar { k } = d ,$ we say g is completely separable.

The class of (block-)separable functions encompasses many standard benchmarks used in nonconvex optimization. In the widely used BBOB test suite [23], 5 of 24 functions are explicitly separable and an additional 8 become separable after removing the orthogonal rotations that BBOB applies to test coordinate-dependence, yielding roughly 54% of the suite that is intrinsically (block-)separable. Beyond BBOB, canonical multimodal benchmarks including Rastrigin, Schwefel, Lévy, Styblinski– Tang, and Michalewicz are all separable [26].

We compare the following variants of SGD.

Definition 1.2 (Fixed-step-size SGD). Given a fixed learning rate $\eta > 0 ;$ , fixed-step-size SGD iterates

$$
x _ { t + 1 } = x _ { t } - \eta \nabla f ( x _ { t } ) + \eta \xi _ { t } , \qquad \xi _ { t } \sim \mathcal { N } ( 0 , I _ { d } ) .
$$

For ϵ-optimality, we consider the step size regime $\eta = \mathcal { O } ( \epsilon / d )$ ofTheorem D.2. The algorithm runs until the iterate first enters an ϵ-optimal region.

Definition 1.3 (Annealed SGD). Annealed SGD uses a learning rate schedule $\theta : [ 0 , T ]  \mathbb { R } _ { > 0 }$ and iterates as

$$
\boldsymbol { x } _ { t + 1 } = \boldsymbol { x } _ { t } - \eta ( \theta _ { t } ) \nabla f ( \boldsymbol { x } _ { t } ) + \eta ( \theta _ { t } ) \boldsymbol { \xi } _ { t } , \qquad \boldsymbol { \xi } _ { t } \sim \mathcal { N } ( 0 , I _ { d } ) ,
$$

where $t \mapsto \eta ( \theta _ { t } )$ is positive, non-increasing, and non-constant.

Theorem 1.4 (Informal). Let f belong to the block-separable class with bounded barriers (Theorem 1.1) satisfying the conditions of Theorem C.10. Then:

1. Lower bound for fixed step size (Theorem D.2). For any fixed learning rate $\eta = \mathcal { O } ( \epsilon / d )$ , the expected number of iterations for fixed-step-size SGD (Theorem 1.2) to reach an ϵ-optimal point i $\cdot \exp ( \Omega ( d ) )$ .

2. Upper bound for decaying step size (Theorem C.10). There exists an annealing schedule such that annealed SGD (Theorem 1.3) outputs an ϵ-optimal point with probability at least 2/3 using at most poly(d, 1/ϵ) gradient queries.

In particular, under the hypotheses of Theorems C.10 and D.2, learning rate decay provides an exponential acceleration for global optimization of this function class when both algroithms are startedfrom the same pointfrom the same initial point.

An intuitive illustration of the beneficial effects of learning rate annealing is given in Figure 1. Prior theoretical work on learning rate decay has focused on implicit bias [2, 16, 56, 57] and generalization [33, 45, 58], showing that decaying schedules select flatter minima or improve test error, but has not established any asymptotic runtime improvement for the underlying optimization problem. Our result demonstrates that the benefit of learning rate decay is not merely statistical but computational: it provably transforms an exponentially hard optimization task into a polynomially easy one. The exponential lower bound for fixed step size is not an artifact of a pathological construction; it follows from a direct bound on the exit time of SGD with Gaussian noise from a spurious well (Appendix D) and applies to the same concrete, well-studied functions on which practitioners routinely observe the advantages of annealing.

A striking feature of this result is that annealed SGD does not directly exploit the block-separable structure of the objective. The algorithm requires no knowledge of the coordinate decomposition, the barrier heights, the basin widths, or the number of local minima. A standard hyperparameter search over decay schedules suffices to achieve polynomial runtime, in contrast to methods such as coordinate descent that require knowing the decomposition. That a completely structure-agnostic modification to SGD achieves an exponential speedup suggests the benefit of learning rate decay is intrinsic to the dynamics of the algorithm rather than to any structural exploitation. Finally, while SGD is rotationally invariant, the rotation included in the function class makes them formally nonseparable and challenging for algorithms like coordinate descent that can trivially optimize separable functions.

In general, no learning rate schedule can succeed in polynomial time for all nonconvex functions due to the NP-hardness of global optimization. Nevertheless, the magnitude of the separation we establish, combined with the structure-agnostic nature of the algorithm, suggests that qualitatively similar improvements should persist for broader function classes with weak inter-variable coupling or approximate separability. Our result provides a rigorous explanation for the empirically observed benefits of learning rate decay from a purely optimization-theoretic perspective: the early largelearning-rate phase performs implicit global exploration that no fixed learning rate can replicate, and this exploration compounds through the subsequent convergence phase.

## 1.2 Theoretical Contributions

We now discuss the main theoretical ingredients that lead to Theorem 1.4. While the intuition for learning rate decay is natural, translating it into a rigorous guarantee is far from straightforward. SGD is a local algorithm: a large learning rate that enables broad exploration can just as easily cause the iterates to overshoot and leave a favorable basin as it can help them arrive there. Unlike a hypothetical structure-aware version of SGD that explicitly detects when a good region has been found and then switches strategy, SGD with a decaying schedule commits to reducing the learning rate on a predetermined timetable, regardless of the current state of the iterates. There is no a priori reason to expect that the iterates will be in the correct basin at the moment the learning rate becomes small enough for the iterates to stabilize. Establishing that this in fact occurs with high probability, and does so in polynomial time, requires a new analysis that goes beyond existing fixed-learning-rate convergence analyses. Our proof has three main ingredients which may each be of independent interest.

![](images/18503d19c5f171a6d415710c99bd3033e22f4b164392944cde77c457355e5a84.jpg)

![](images/a2cef99d89ef5d89b54eb6099c73f95338294b53cdafaa321791901b8df68480.jpg)  
global minlocal min start end

![](images/1dda6fa12402b81c53dfeb3b3c75494642bfd4efa2906e17f4c1ef23135ae7df.jpg)

![](images/361a9414b2f7da6692acfc4a29aa55872970c0c2d9d73e65702c6c2c7a23692c.jpg)

![](images/b528e6dbe1b062fde9180c2c8254a9436b9ced5d0bc479b0bf7c09939ae98166.jpg)

![](images/22e1f57da08930472b3a561bcb185c04d4e64c74e046fe36765d7535ec607b6a.jpg)  
Figure 1: The trajectories of the SGD algorithm with fixed large and small step sizes (Columns 1 and 2, respectively) and a decaying learning rate on a toy problem. While a large step size allows the algorithm to explore the landscape efficiently, the small step size causes the algorithm to be trapped in a local minimum, leading to exponential runtime. The decaying step size allows the algorithm to visit the global minimum with high probability in polynomial time (Column 3).

• Analysis of Annealed Langevin Diffusion. To demonstrate the effectiveness of annealing for SGD, we proceed via a correspondence with the annealed version of a continuous-time sampling algorithm based on Langevin diffusion, which forms the basis of discretized algorithms such as Langevin Monte Carlo (LMC) [31, 36]. We analyze the convergence of Langevin diffusion under a temperature annealing schedule, where the learning rate plays the role of temperature. The central difficulty is that at any single low temperature, the standard Poincaré inequality yields an exponentially small spectral gap, reflecting the exponential time required for the dynamics to cross energy barriers from a worst-case initialization. Our key insight is that annealing avoids this worst case: when the temperature decreases slowly, the distribution inherited from the previous high-temperature phase serves as a favorable starting point for the next phase. To formalize this, we employ the weak Poincaré inequality [10, 25], a relaxation of the standard Poincaré inequality that allows a defect term depending on the quality of the initialization. We show that for blockseparable functions, the weak Poincaré constant remains polynomially bounded across all phases of the annealing schedule. Combining these per-phase bounds via an annealing theorem adapted from [25], we establish that the continuous annealed Langevin diffusion reaches the Gibbs measure concentrated on the global minimum in poly(d, 1/ϵ) time.

• Discretization Analysis for Annealed ULA. We combine our diffusion analysis with a Rényi discretization bound to prove an end-to-end polynomial-time optimization guarantee for the annealed unadjusted Langevin algorithm (ULA). The temperatures and step sizes are chosen independently, controlling discretization error and the number of gradient queries across all phases. For blockseparable functions satisfying our smoothness and quadratic growth conditions, the resulting schedule finds an ϵ-optimal point with probability at least 2/3 using $\mathrm { p o l y } ( d , 1 / \epsilon )$ gradient queries (Theorem B.12).

• Spectral Transfer to Annealed SGD. The next problem is to translate our result on annealing the temperature of Langevin diffusion into one about annealing the learning rate of SGD. Standard pathwise discretization estimates can be too restrictive when the early learning rates are large. We derive a discrete Rényi contraction bound, combine it across phases, and prove vanishing stationary bias at the final learning rate under spectral transfer assumption (Theorem C.2). Each phase supplies the starting distribution for the next; this distribution is a conclusion of the induction over phases using Poincaré inequality.

We finally comment on the lower bound in Theorem 1.4. This result shows concretely that existing fixed-rate analyses of SGD still yield exponential runtimes for block-separable functions as for general nonconvex functions. The main idea is the following: we first explain why a step size $\eta = \mathcal { O } ( \epsilon / d )$ is the regime we consider for global optimization. We then bound the exit time from a spurious well directly to prove that there exist completely separable functions for which SGD requires exp(Ω(d)) iterations at any sufficiently small step size. Details of this calculation are given in Appendix D.

Organization. The rest of the paper is organized as follows. Our complete proofs are given in the appendix. Section 2 gives more detail and intuition on the mixing analysis of annealed Langevin diffusions. Section 3 gives more detail on the translation of this analysis to results on the convergence of SGD with decaying learning rate, and how this results in a proof of the upper bound in Theorem 1.4. Section 4 describes numerical simulations that confirm our theoretical results and indicate the applicability of the identified phenomena beyond the scope of the analysis.

## 1.3 Related Work

SGD has been extensively analyzed in the convex setting [5, 22, 38, 39, 44, 47]. For general nonconvex functions, SGD has been typically analyzed for convergence to a stationary point [3, 19, 30, 34, 42, 49]. When the objective has favorable structure, such as the Polyak–Łojasiewicz condition [29, 30], quasar convexity [21], or a star-convex path [60], fixed-step-size SGD can converge to the global minimum efficiently. SGD has also been analyzed for stylized nonconvex problems like matrix completion or phase retrieval [14, 35], where stationary points are globally optimal [17, 18].

The Langevin Monte Carlo (LMC) algorithm, a discretization of overdamped Langevin diffusion, has well-established convergence guarantees in the log-concave setting [13, 15, 55]. For nonconvex objectives, Raginsky et al. [43] and Zhang et al. [59] gave non-asymptotic bounds that depend exponentially on barrier height, and Azizian et al. [4] provided tight matching bounds confirming this dependence.

Simulated annealing is a classical strategy for escaping local minima [7]. Recently, Huang et al. [25] introduced weak Poincaré inequalities for analyzing simulated annealing on discrete spin-glass models, and Chen and Sridharan [10] developed connections between isoperimetric inequalities and optimization. Our work adapts these ideas to continuous SGD with learning rate decay and adds a discretization analysis bridging the Langevin diffusion and the discrete iterates. The annealing of noise scales also appears in score-based generative models [8, 11, 51, 52], which share technical ingredients with our analysis despite targeting a different goal.

Learning rate decay of SGD itself has been studied through the lens of implicit bias [2, 16, 56, 57] and generalization [33, 45, 58]. Large learning rates have been connected to a catapult mechanism [32], while Sclocchi and Wyart [48] studied how step-size-to-batch-size ratios determine different SGD regimes. Mohtashami et al. [37] also considered a large learning rate for nonconvex optimization, but is restricted to gradient descent with a large but constant step size.

To summarize, to the best of our knowledge, this is the first work that studies the provable performance of SGD with decaying step size for global optimization, establishing a provable separation in global optimization runtime between annealed and non-annealed SGD.

## 2 Analysis of Annealed Langevin Diffusion

We now present our proof strategy and main results. Our analysis proceeds by relating annealed SGD to a learning-rate-dependent Langevin SDE,

$$
\mathrm { d } X _ { t } = - \nabla f ( X _ { t } ) \mathrm { d } t + \sqrt { \eta ( \theta ( t ) ) } \mathrm { d } B _ { t }\tag{1}
$$

where $B _ { t }$ is a standard Brownian motion and $\theta : [ 0 , T ]  \mathbb { R } _ { > 0 }$ is the learning rate schedule. Let $\mu _ { t }$ denote the distribution of $X _ { t }$ . For a constant learning rate η, as $t \to \infty$ , we have $\mu _ { t } \to \pi _ { f , \beta } ( x )$ ∝ $e ^ { - \beta f ( x ) }$ , known as the Gibbs distribution, where $\beta = 2 / \eta$ . The Euler–Maruyama discretization of (1) yields the SGD update with Gaussian noise:

$$
x _ { t + 1 } = x _ { t } - \eta ( \theta _ { t } ) \nabla f ( x _ { t } ) + \eta ( \theta _ { t } ) \xi _ { t }
$$

where $\xi _ { t } \sim \mathcal { N } ( 0 , I _ { d } )$ are independent standard Gaussian random vectors. While the general form of SGD does not have Gaussian normal noise in practice, we make this simplification for the theoretical analysis as done in [50] and expect the results to hold as long as the noise distributions have decaying tails. We then implement the annealing through a decreasing sequence of learning rates $\eta _ { 1 } > \eta _ { 2 } >$ $\cdots > \eta _ { M } > 0$ . In phase $k = 1 , \dots , M$ , the algorithm runs with constant learning rate η for a duration $\tau _ { k }$ chosen so that the constant-temperature Langevin SDE, $d X _ { t } = - \nabla f { ( X _ { t } ) } { \overset { } { \mathrm { d } } } t + { \sqrt { \eta _ { k } } } d B _ { t }$ approximately reaches its stationary distribution. In this sense, our analysis uses the sampling problem as a proxy for the optimization, since when the step size $\eta _ { k }$ is small, the stationary distribution $\pi _ { \eta _ { k } }$ has large probability around the global minimum. These techniques are used in the context of the Langevin Monte Carlo (LMC) algorithm for nonconvex optimization [31, 36], although the runtime is usually given by an exponentially small geometric quantity such as the log Sobolev or Cheeger constant characterizing the conductance of this SDE.

We now define the key functional inequality underlying our analysis. A reversible Markov semigroup P with stationary distribution π and Dirichlet form $\mathcal { E } _ { \pi }$ satisfies a $\delta _ { \mathfrak { p } } .$ -Poincaré inequality if $\mathcal { E } _ { \pi } ( h , h )$ ⩾ $\delta _ { \mathsf { p } } \operatorname { V a r } _ { \pi } [ h ]$ for all functions h. For the Langevin diffusion (1) with a constant learning rate η, the Dirichlet form is $\begin{array} { r } { \mathcal { E } _ { \pi } ( h , h ) = \frac { \eta } { 2 } \int \| \nabla h \| ^ { 2 } d \pi } \end{array}$ . For multimodal distributions, the Poincaré constant $\delta _ { p }$ is typically exponentially small in the barrier height and inverse learning rate and characterizes the optimization hardness from the worst initialization. The weak Poincaré inequality relaxes the standard Poincaré inequality by allowing a defect term and characterizes the hardness from most initializations [10]. While we provide the self-contained definitions in our proofs, for a more detailed background on Markov semigroups and functional inequalities, we refer the reader to [6].

Definition 2.1 (Weak Poincaré inequality). We say $P$ with state space Ω satisfies $a ( \delta _ { \mathrm { w p } } ( s ) , s )$ -weak Poincaré inequality with respect to its stationary distribution π iffor all $h : \dot { \Omega }  \mathbb { R }$

$$
\begin{array} { r } { \mathcal { E } _ { \pi } ( h , h ) \geqslant \delta _ { \mathrm { w p } } ( s ) \mathrm { V a r } _ { \pi } [ h ] - s \| h \| _ { \infty } ^ { 2 } . } \end{array}
$$

When $s \ = \ 0 ,$ , we obtain the standard Poincaré inequality. The defect term $s \| h \| _ { \infty } ^ { 2 }$ allows the constant $\delta _ { \mathrm { w p } } ( s )$ to be much larger than the standard Poincaré constant, which is critical for obtaining polynomial runtime bounds. Crucially, under a $( \delta _ { \mathrm { w p } } ( s ) , s )$ -weak Poincaré inequality, the convergence of Langevin dynamics starting from an initial distribution $\nu _ { 0 }$ satisfies

$$
\chi ^ { 2 } ( \nu _ { t } \| \pi ) \leqslant e ^ { - \delta _ { \mathrm { w p } } ( s ) t } \chi ^ { 2 } ( \nu _ { 0 } \| \pi ) + \frac { s } { \delta _ { \mathrm { w p } } ( s ) } \left\| \frac { d \nu _ { 0 } } { d \pi } - 1 \right\| _ { \infty } ^ { 2 } ,
$$

where $\begin{array} { r } { \chi ^ { 2 } ( \nu | | \pi ) = \int \left( \frac { d \nu } { d \pi } \right) ^ { 2 } d \pi - 1 } \end{array}$ is the chi-square divergence. The key consequence is that if the initial distribution $\nu _ { 0 }$ is a warm start with bounded density ratio $( \mathrm { i . e . , \ } \| d \nu _ { 0 } / d \pi - 1 \| _ { \infty } = \mathcal { O } ( 1 ) )$ then the convergence rate is governed by $\delta _ { \mathrm { w p } }$ rather than by the standard Poincaré constant. In the annealing setting, the output of each high learning rate phase provides precisely such a warm start for the low learning rate phase, enabling the algorithm to bypass the exponential mixing bottleneck that slows down the fixed learning rate dynamics.

Proof Sketch. The proof has three main steps. First, we show that the continuous annealed dynamics (1) converge to a distribution $\mu _ { T }$ that assigns probability $\Omega ( 1 )$ to a small neighborhood of $x ^ { \star }$ , where $x ^ { \star }$ is the global minimum, with total runtime governed by the minimum weak Poincaré constant across phases. Next, we derive a lower bound on the weak Poincaré constant via perturbation analysis, establishing that this constant remains large even at small learning rates. Third, we show that the discretized algorithm (annealed SGD) converges to the same distribution with only polynomial overhead, using a new discretization argument that decouples the early exploratory phase from the late convergent phase.

## 2.1 Bounding the Weak Poincaré Constant

While the weak Poincaré inequality relaxes the standard Poincaré inequality significantly, its constant is still a global geometric quantity that depends heavily on the function geometry. To quantify this constant, we derive a novel comparison method in Theorem A.1. Informally, given a measure m that satisfies a Poincaré inequality, Theorem A.1 allows us to quantify the weak Poincaré constant of a measure π. This theorem generalizes the standard Holley–Stroock type of perturbation results used previously in [36] to the weak Poincaré setting. However, as opposed to the Holley–Stroock perturbation theorem, our result is also applicable to distributions with an unbounded density ratio. As an application of this theorem, we compare the weak Poincaré constant of the SGD dynamics to the standard Poincaré constant. As opposed to the standard Poincaré constant, Figure 2 shows that for a 1-D toy problem, the weak Poincaré constant initially decreases and then starts increasing again around $\eta \sim 1$ , where the stationary distribution concentrates around the basin of the global minimum. This indicates that the dynamics stabilize early on, at a learning rate much larger than what standard analyses require. As a further corollary of this theorem, by choosing m to be a Gaussian around $x ^ { \star }$ and $\pi = \pi _ { f , \beta }$ , we show the following theorem:

![](images/5be514837aa499df6fcdc453cf7c76304ab031c74cac9dd3d6e9d610a470c845.jpg)

![](images/2222d2a91fcc1624adcfd6534e99ce2b38c2f05ffd2fd06348b4974482f83b32.jpg)

![](images/4395fd9494fa1c8572ed664ce489d84dc188f9a180b35783aa4e4d9f38ba0ef7.jpg)  
Figure 2: Comparison of the weak Poincaré constant (normalized by $1 / \eta )$ to the standard Poincaré constant (normalized by $1 / \eta )$ for a 1-D toy function<sup>4</sup>. While the standard Poincaré constant decays exponentially, the weak Poincaré constant decays initially but then starts increasing, indicating that the effect of local minima can be benign with a proper annealing scheme.

Theorem 2.2 (Informal). Let $f : \mathbb { R } ^ { d }  \mathbb { R }$ be a $C ^ { 3 }$ function with a unique global minimizer $x ^ { \star }$ positive definite Hessian $H : = \nabla ^ { 2 } f ( x ^ { \star } )$ with smallest eigenvalue $\omega ,$ and quadratic growth away from $x ^ { \star }$ . Then for all β sufficiently large, the Gibbs measure $\pi _ { \beta } ( x ) \propto \exp ( - \beta f ( x ) )$ satisfies the weak Poincaré inequality

$$
\mathrm { V a r } _ { \pi _ { \beta } } [ h ] \leqslant { \frac { 1 + O ( \beta ^ { - 1 / 4 } ) } { \omega \beta } } \int \| \nabla h \| ^ { 2 } d \pi _ { \beta } + C \exp \bigl ( - c \beta ^ { 1 / 6 } \bigr ) \| h \| _ { \infty } ^ { 2 } ,
$$

where $c , C > 0$ are constants independent ofβ. Theformal statement is Theorem A.4.

The importance of Theorem $2 . 2$ is that for $\beta = 2 / \eta ,$ the weak Poincaré constant does not exponentially decay, but converges to $\begin{array} { r } { \beta \omega = \frac { 2 \omega } { \eta } } \end{array}$ asymptotically (equivalently, to ω when the inequality is stated with the Dirichlet form $\begin{array} { r } { \frac { \eta } { 2 } \int \| \nabla h \| ^ { 2 } \dot { d } \pi _ { \beta } } \end{array}$ of the generator of Langevin dynamics). The asymptotic result alone does not imply polynomial runtimes for general nonconvex functions since it is only valid in the small $\eta$ limit and $\delta _ { \mathrm { w p } }$ can be very small for larger learning rates. However, the result is also important from a mathematical point of view, since for functions with unique global minima, similar asymptotic behavior is known for the spectral gap of Schrödinger operators [24] that are connected to Langevin diffusion. While the spectrum of the generator of Langevin dynamics contains an eigenstate that could make the spectral gap very small, this eigenstate is not significant in the limit where $\beta \to \infty$

The fact that the limit of the weak Poincaré constant is $\frac { 2 \omega } { \eta }$ is not coincidental. This matches the Poincaré constant $\beta \omega$ of the Gaussian $\mathcal { N } ( \boldsymbol { x } ^ { \star } , ( \beta H ) ^ { - 1 } )$ , the Gibbs measure of a strongly convex quadratic function with its minimum at $x ^ { \star }$ . From an algorithmic point of view, this result implies that existence of a learning rate schedule that can escape from local minima early on will localize around the global minima and effectively the algorithm will only be affected by the strongly convex well around the global minima.

For the setting of block separable functions with constant size blocks, we prove that each block satisfies a weak Poincaré inequality with

$$
( \delta _ { \mathrm { w p } } ( s ) , s ) = \left( \frac { \xi } { 4 } , \frac { \xi } { 4 } C e ^ { - 2 c / \eta } \right) , \qquad \eta < \eta _ { * } ,\tag{2}
$$

where $\eta _ { * } ~ = ~ \Omega _ { d } ( 1 )$ and the constants $\xi , c , C \ > \ 0$ do not depend on d (Theorem $_ { \mathrm { A } . 6 ) }$ . The exponentially small defect in (2) provides the stronger inequality used at low temperature. The uniform weak Poincaré family of Theorem A.5 handles the earlier temperatures.

## 3 Discretization and Learning Rate Schedule

When the goal is to simulate Langevin dynamics, it is common in the literature to select a timediscretization step size that is different from the corresponding learning rate [61]. However, this algorithm is slightly different from the form of SGD in (Theorem 1.3).

We first state the optimization result for this intermediate algorithm. The unadjusted Langevin algorithm (ULA) uses the update

$$
\begin{array} { r } { x _ { t + 1 } = x _ { t } - h _ { t } \nabla f ( x _ { t } ) + \sqrt { h _ { t } \eta _ { t } } \xi _ { t } , \xi _ { t } \sim \mathcal { N } ( 0 , I _ { d } ) . } \end{array}\tag{3}
$$

Here $\eta _ { t }$ is the noise level and $h _ { t }$ is the discretization step. The choice $h _ { t } = \eta _ { t }$ gives annealed SGD (Theorem 1.3). We begin with the optimization guarantee obtained when the steps are free.

Theorem 3.1 (Informal). Under the hypotheses ofTheorem B.12, the annealed ULA algorithm (3) uses independently chosen noise levels $\eta _ { r }$ and discretization steps $h _ { r }$ . For every $\epsilon \in ( 0 , 1 \bar { ] }$ , it outputs an ϵ-optimal point with probability at least $2 / 3$ using $\mathrm { p o l y } ( d / \bar { \epsilon } )$ gradient queries.

Unlike ULA, the discrete iterates of SGD can deviate from the continuous Langevin diffusion despite its convergence. Similar phenomena have been observed for the gradient descent algorithm, that is, despite the success of gradient descent, it can follow a significantly different path from continuous gradient flow [14].

To analyze SGD, we instead transfer the weak Poincaré inequalities of Langevin diffusion to the discrete SGD kernel. This gives an annealing proof analogous to the continuous-time argument, with phase lengths determined by the discrete contraction bound. Additionally, the fact that the learning rate is changing over time also introduces challenges into the discretization analysis. Particularly, convergence results on continuous-time Langevin dynamics are all in terms of quantities that fail to satisfy the triangle inequality. Thus we have to carefully divide the learning rate sequence in such a way that we can compose all sources of error properly.

The next theorem states the resulting SGD optimization guarantee.

Theorem 3.2 (Informal). Under the hypotheses of Theorem C.10 and Theorem C.2, the annealed SGD algorithm uses $h _ { r } = \eta _ { r }$ in every phase. For every $\epsilon \in ( 0 , 1 ] ,$ , it outputs an ϵ-optimal point with probability at least $2 / 3$ using $\mathrm { p o l y } ( \dot { d } / \epsilon )$ gradient queries.

The learning rate schedule is an arithmetically decaying sequence from $\eta _ { 1 }$ to $\eta _ { M }$ with sufficiently small constant decrements. We select a constant $\eta _ { 1 }$ . To obtain an ϵ approximate global minimizer with constant probability, it suffices to take the final learning rate to be $\eta _ { M } = \Theta ( \epsilon ^ { 2 } / d )$ (proven in Section C.3).

The spectral-transfer condition in Theorem C.2 compares one-step contraction of each fixed-rate SGD kernel with Langevin diffusion, up to a constant or polylogarithmic factor $\kappa _ { d }$ . Constant comparisons between invariant and Gibbs laws control initialization and transfers between phases. These are mild requirements on each constant-size block, one-dimensional for completely separable functions, with bounds independent of the ambient dimension and target accuracy.

At low temperature, the the SGD update yield an order-η stationary bias (Theorem C.4), so the sum of the block biases at $\eta _ { M } = \Theta ( \epsilon ^ { 2 } / d )$ is small enough for optimization. On the other hand, for general nonconvex functions, an initial learning rate $\eta _ { 1 } = \mathcal { O } ( 1 / d )$ can lead to exponential runtime whereas Theorem 3.2 allows a constant initial learning rate and proves convergence using the inherited weak Poincaré inequalities when the learning rate is decayed slowly instead of initialized at such small values.

## 4 Numerical Results

We now empirically examine the polynomial-versus-exponential separation predicted by our theory through controlled experiments on the following three functions:

$$
\begin{array} { r } { 1 . ~ f ( x ) = \sum _ { i = 1 } ^ { d } [ ( x _ { i } ^ { 2 } - 1 ) ^ { 2 } + 0 . 3 x _ { i } ] ~ \mathrm { ( S e p a r a b l e ~ D o u b l e - W e l l ) } } \end{array}
$$

$$
\begin{array} { r } { 2 . \ f ( x ) = A d + \sum _ { i = 1 } ^ { d } [ x _ { i } ^ { 2 } - A \cos ( 2 \pi x _ { i } ) ] \quad \mathrm { ( R a s t r i g i n ~ w i t h ~ } A = 1 ) } \end{array}
$$

$$
\begin{array} { r } { 3 . \ f ( x ) = \sum _ { i = 1 } ^ { d } [ ( x _ { i } ^ { 2 } - 1 ) ^ { 2 } + 0 . 3 x _ { i } ] + 0 . 5 \sum _ { i = 1 } ^ { d - 1 } ( x _ { i } - x _ { i + 1 } ) ^ { 2 } \quad \mathrm { ( C o u p l e d ~ D o u b l e - W e l l ) } } \end{array}
$$

The first two functions belong to the function class satisfying Theorem 1.1. We test the third function, which is non-separable, to assess how annealed SGD performs in the setting beyond Theorem 1.4.

Separable Double-Well

![](images/c4f4ea4e48a7faf77961bc932c2fc6b56300e43fb46ec143540f63c10d9bd3a6.jpg)  
Soft Rastrigin

![](images/043c13c836d7f19ae99cd60e6f5acdcb6b32e29214a1f066318c93857149b436.jpg)  
Coupled Double-Well

![](images/1effbcec249789788986d01f868a46a994cc0675fe734730058caaa3f5012946.jpg)

![](images/26eaff9554e145c697acde0f341501018ef152e355dc102a6668206c421f8fd6.jpg)

![](images/3591898fa60c920cdc73c1a18f6644452d3a1e1ff1b427171ec0ac8093591209.jpg)

![](images/5f8479848d7b012bf9b07d242b5be6e5f52db9c9ed14378b889c207c64828ef9.jpg)  
Figure 3: Median number of iterations to convergence as a function of dimension d for fixed-rate SGD (blue) and annealed SGD (orange), both with $h _ { k } = \eta _ { k }$ . Error bars show the interquartile range over 10 runs. Dashed curves are descriptive fits; the coupled annealed power-law fit is weak.

Setup. We implement fixed-step-size and annealed SGD (Theorems 1.2 and 1.3) with Gaussian gradient noise, using the same learning rate $\eta _ { t }$ to scale the gradient and noise terms. Both algorithms are initialized near the spurious local minimum for all three functions. Fixed-step-size SGD keeps its learning rate constant throughout each run. Annealed SGD uses function-specific arithmetic learning-rate decay across constant-rate phases. Convergence is declared when $\| x - x ^ { \star } \| _ { \infty }$ falls below a tolerance. Specific parameters for each function are detailed in Appendix E.

Results. Figure 3 shows a clear separation between the fixed-step-size SGD algorithm and the annealed SGD schedules over the tested dimensions. Fixed-step-size SGD becomes rapidly more expensive as dimension increases, with approximately exponential growth across all three functions. In contrast, Annealed SGD exhibits polynomial growth in median iteration counts, $\tau \propto d ^ { c ^ { \prime } }$ , for the separable double-well and Rastrigin functions, while the coupled double-well median grows only mildly with dimension. This polynomial-versus-exponential pattern is consistent across the three functions tested. Indeed, Annealed SGD performs well in higher dimensions using the same function-specific schedules, as shown in Table 1. In particular, even the largest median recorded run time (separable double well, d = 400) is about 42 seconds, which is less than one minute.

Table 1: Annealed SGD with arithmetic learning-rate decay and $h _ { t } = \eta _ { t }$ in high dimensions. Median recorded update counts $( \times 1 0 ^ { 5 } )$ and median run times (seconds), ± one empirical standard deviation over 10 trials per function and dimension. The stopping criterion is $\| x - \overline { { x ^ { \star } } } \| _ { \infty } < \delta$ , with $\delta = 0 . 5$ for both double-well functions and $\delta = 0 . 3$ for Rastrigin.
<table><tr><td></td><td></td><td>d = 100</td><td> $d = 2 0 0$ </td><td> $d = 3 0 0$ </td><td>d = 400</td></tr><tr><td rowspan="3">Iterations  $( \times 1 0 ^ { 5 } )$ </td><td>Separable Double-Well</td><td> $3 2 . 3 1 \pm 6 7 . 2 6$ </td><td> $3 6 . 9 5 \pm 4 0 . 5 3$ </td><td> $7 6 . 6 7 \pm 2 0 . 2 0$ </td><td> $7 9 . 6 8 \pm 4 2 . 1 7$ </td></tr><tr><td>Rastrigin</td><td> $0 . 0 3 7 5 \pm 0 . 0 0 3 6$ </td><td> $0 . 1 8 4 4 \pm 0 . 2 2 1 0$ </td><td> $3 . 0 9 3 6 \pm 0 . 7 0 0 4$ </td><td> $4 . 1 2 0 0 \pm 0 . 0 0 1 3$ </td></tr><tr><td>Coupled Double-Well</td><td> $0 . 1 8 7 5 \pm 0 . 2 9 4 2$ </td><td> $0 . 2 8 2 2 \pm 0 . 2 1 3 1$ </td><td> $0 . 2 2 6 5 \pm 0 . 1 1 8 9$ </td><td> $0 . 5 2 6 0 \pm 0 . 7 5 7 7$ </td></tr><tr><td rowspan="3">Run time (seconds)</td><td>Separable Double-Well</td><td> $4 . 2 0 \pm 8 . 7 5$ </td><td> $9 . 4 2 \pm 1 0 . 3 4$ </td><td> $2 9 . 6 0 \pm 8 . 0 3$ </td><td> $4 1 . 5 7 \pm 2 1 . 8 7$ </td></tr><tr><td>Rastrigin</td><td> $0 . 0 0 7 6 \pm 0 . 0 0 0 6$ </td><td> $0 . 0 6 7 1 \pm 0 . 0 8 5 1$ </td><td> $1 . 6 7 \pm 0 . 3 7$ </td><td> $2 . 9 6 \pm 0 . 0 5$ </td></tr><tr><td>Coupled Double-Well</td><td> $0 . 0 2 5 1 \pm 0 . 0 4 0 1$ </td><td> $0 . 0 7 4 2 \pm 0 . 0 5 7 5$ </td><td> $0 . 0 9 0 5 \pm 0 . 0 4 6 8$ </td><td> $0 . 2 8 4 \pm 0 . 4 1 5$ </td></tr></table>

## 5 Conclusion

We have shown that for a special but broad class of nonconvex functions satisfying Theorem C.2, learning rate decay converts the exponential runtime of fixed-step-size SGD in the regime η = $\mathcal { O } ( \epsilon / d )$ into a polynomial one, establishing the first polynomial-versus-exponential separation for this ubiquitous heuristic. The primary limitation is the current restriction to block-separable functions, and the primary open question is whether our techniques can be used to prove (potentially smaller) separations for more general functions, such as those with weak inter-variable couplings.

## Disclaimer

This paper was prepared for informational purposes by the Global Technology Applied Research center of JPMorgan Chase & Co. This paper is not a product of the Research Department of JPMorgan Chase & Co. or its affiliates. Neither JPMorgan Chase & Co. nor any of its affiliates makes any explicit or implied representation or warranty and none of them accept any liability in connection with this paper, including, without limitation, with respect to the completeness, accuracy, or reliability of the information contained herein and the potential legal, compliance, tax, or accounting effects thereof. This document is not intended as investment research or investment advice, or as a recommendation, offer, or solicitation for the purchase or sale of any security, financial instrument, financial product or service, or to be used in any way for evaluating the merits of participating in any transaction.

## AI Usage

The initial version of this paper was written without the use of AI tools. The authors subsequently used GPT 6.1 Sol and Claude 5.5 Opus to correct the errors identified in the proofs during the review process, expand the details of the proofs and polish the writing of the appendix. The authors verified the details of the these proofs and take responsibility for the correctness and presentation of the paper.

## References

[1] Shun-ichi Amari. Backpropagation and stochastic gradient descent method. Neurocomputing, 5 (4-5):185–196, 1993.

[2] Maksym Andriushchenko, Aditya Vardhan Varre, Loucas Pillaud-Vivien, and Nicolas Flammarion. SGD with large step sizes learns sparse features. In International Conference on Machine Learning, pages 903–925. PMLR, 2023.

[3] Yossi Arjevani, Yair Carmon, John C Duchi, Dylan J Foster, Nathan Srebro, and Blake Woodworth. Lower bounds for non-convex stochastic optimization. Mathematical Programming, 199 (1):165–214, 2023.

[4] Waïss Azizian, Franck Iutzeler, Jerome Malick, and Panayotis Mertikopoulos. The global convergence time of stochastic gradient descent in non-convex landscapes: Sharp estimates via large deviations. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pages 1982–2044. PMLR, 2025. URL https://proceedings.mlr.press/v267/azizian25a.html.

[5] Francis Bach and Eric Moulines. Non-asymptotic analysis of stochastic approximation algorithms for machine learning. In Advances in Neural Information Processing Systems, volume 24, 2011.

[6] Dominique Bakry, Ivan Gentil, and Michel Ledoux. Analysis and Geometry ofMarkov Diffusion Operators, volume 348 of Grundlehren der mathematischen Wissenschaften. Springer International Publishing, Cham, 2014. doi: 10.1007/978-3-319-00227-9. URL https: //doi.org/10.1007/978-3-319-00227-9.

[7] Alexandre Belloni, Tengyuan Liang, Hariharan Narayanan, and Alexander Rakhlin. Escaping the local minima via simulated annealing: Optimization of approximately convex functions. In Conference on Learning Theory, pages 240–265. PMLR, 2015.

[8] Joe Benton, Valentin De Bortoli, Arnaud Doucet, and George Deligiannidis. Nearly d-linear convergence bounds for diffusion models via stochastic localization. In International Conference on Learning Representations, 2024. URL https://arxiv.org/abs/2308.03686.

[9] Léon Bottou. Stochastic gradient learning in neural networks. In Proceedings ofNeuro-Nîmes 1991, 1991. URL https://leon.bottou.org/publications/pdf/nimes-1991.pdf.

[10] August Y. Chen and Karthik Sridharan. Optimization, isoperimetric inequalities, and sampling via lyapunov potentials. In Proceedings of the Thirty-Eighth Conference on Learning Theory, volume 291 of Proceedings of Machine Learning Research, pages 1094–1153. PMLR, 2025. URL https://proceedings.mlr.press/v291/chen25g.html.

[11] Hongrui Chen, Holden Lee, and Jianfeng Lu. Improved analysis of score-based generative modeling: User-friendly bounds under minimal smoothness assumptions. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 4735–4763. PMLR, 2023. URL https://proceedings. mlr.press/v202/chen23q.html.

[12] Sinho Chewi. Log-Concave Sampling. Forthcoming, 2026. URL https://chewisinho. github.io/main.pdf. Available online at https://chewisinho.github.io/main.pdf.

[13] Arnak S. Dalalyan. Theoretical guarantees for approximate sampling from smooth and logconcave densities. Journal ofthe Royal Statistical Society Series B: Statistical Methodology, 79(3):651–676, 2017. doi: 10.1111/rssb.12183. URL https://doi.org/10.1111/rssb. 12183.

[14] Christopher De Sa, Christopher Re, and Kunle Olukotun. Global convergence of stochastic gradient descent for some non-convex matrix problems. In International Conference on Machine Learning, pages 2332–2341. PMLR, 2015.

[15] Alain Durmus, Szymon Majewski, and Błazej Miasojedow. Analysis of Langevin Monte Carlo˙ via convex optimization. Journal ofMachine Learning Research, 20(73):1–46, 2019. URL http://jmlr.org/papers/v20/18-173.html.

[16] Mathieu Even, Scott Pesme, Suriya Gunasekar, and Nicolas Flammarion. (S)GD over diagonal linear networks: Implicit bias, large stepsizes and edge of stability. In Advances in Neural Information Processing Systems, volume 36, pages 29406–29448, 2023.

[17] Rong Ge, Furong Huang, Chi Jin, and Yang Yuan. Escaping from saddle points—online stochastic gradient for tensor decomposition. In Conference on Learning Theory, pages 797– 842. PMLR, 2015.

[18] Rong Ge, Jason D Lee, and Tengyu Ma. Matrix completion has no spurious local minimum. In Advances in Neural Information Processing Systems, volume 29, 2016.

[19] Saeed Ghadimi and Guanghui Lan. Stochastic first-and zeroth-order methods for nonconvex stochastic programming. SIAM Journal on Optimization, 23(4):2341–2368, 2013.

[20] Ian Goodfellow, Yoshua Bengio, and Aaron Courville. Deep Learning. MIT Press, 2016. URL https://www.deeplearningbook.org/.

[21] Robert Gower, Othmane Sebbouh, and Nicolas Loizou. SGD for structured nonconvex functions: Learning rates, minibatching and interpolation. In International Conference on Artificial Intelligence and Statistics, pages 1315–1323. PMLR, 2021.

[22] Robert Mansel Gower, Nicolas Loizou, Xun Qian, Alibek Sailanbayev, Egor Shulgin, and Peter Richtárik. SGD: General analysis and improved rates. In International Conference on Machine Learning, pages 5200–5209. PMLR, 2019.

[23] Nikolaus Hansen, Steffen Finck, Raymond Ros, and Anne Auger. Real-parameter black-box optimization benchmarking 2009: Noiseless functions definitions. Technical Report RR-6829, INRIA, 2009. URL https://inria.hal.science/inria-00362633.

[24] Dylan Herman, Guneykan Ozgul, Anuj Apte, Junhyung Lyle Kim, Anupam Prakash, Jiayu Shen, and Shouvanik Chakrabarti. Mechanisms for quantum advantage in global optimization of nonconvex functions. arXiv preprint, 2025. URL https://arxiv.org/abs/2510.03385. arXiv:2510.03385.

[25] Brice Huang, Sidhanth Mohanty, Amit Rajaraman, and David X. Wu. Weak Poincaré inequalities, simulated annealing, and sampling from spherical spin glasses. In Proceedings of the 57th Annual ACM Symposium on Theory of Computing, STOC ’25, pages 915–923, New York, NY, USA, 2025. Association for Computing Machinery. ISBN 9798400715105. doi: 10.1145/3717823.3718236. URL https://arxiv.org/abs/2411.09075v2. Full version: arXiv:2411.09075v2.

[26] Momin Jamil and Xin-She Yang. A literature survey of benchmark functions for global optimisation problems. International Journal of Mathematical Modelling and Numerical Optimisation, 4(2):150–194, 2013.

[27] Chi Jin, Rong Ge, Praneeth Netrapalli, Sham M Kakade, and Michael I Jordan. How to escape saddle points efficiently. In International Conference on Machine Learning, pages 1724–1732. PMLR, 2017.

[28] Chi Jin, Praneeth Netrapalli, Rong Ge, Sham M Kakade, and Michael I Jordan. On nonconvex optimization for machine learning: Gradients, stochasticity, and saddle points. Journal of the ACM (JACM), 68(2):1–29, 2021.

[29] Hamed Karimi, Julie Nutini, and Mark Schmidt. Linear convergence of gradient and proximalgradient methods under the Polyak–Łojasiewicz condition. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pages 795–811. Springer, 2016.

[30] Ahmed Khaled and Peter Richtárik. Better theory for SGD in the nonconvex world. Transactions on Machine Learning Research, 2023. URL https://arxiv.org/abs/2002.03329.

[31] Yuri Kinoshita and Taiji Suzuki. Improved convergence rate of stochastic gradient Langevin dynamics with variance reduction and its application to optimization. In Advances in Neural Information Processing Systems, volume 35, pages 19022–19034, 2022. doi: 10.52202/ 068431-1382. URL https://doi.org/10.52202/068431-1382.

[32] Aitor Lewkowycz, Yasaman Bahri, Ethan Dyer, Jascha Sohl-Dickstein, and Guy Gur-Ari. The large learning rate phase of deep learning: the catapult mechanism. arXiv preprint, 2020. URL https://arxiv.org/abs/2003.02218. arXiv:2003.02218.

[33] Yuanzhi Li, Colin Wei, and Tengyu Ma. Towards explaining the regularization effect of initial large learning rate in training neural networks. In Advances in Neural Information Processing Systems, volume 32, 2019.

[34] Jun Liu and Ye Yuan. On almost sure convergence rates of stochastic gradient methods. In Conference on Learning Theory, pages 2963–2983. PMLR, 2022.

[35] Jianhao Ma and Salar Fattahi. Global convergence of sub-gradient method for robust matrix recovery: Small initialization, noisy measurements, and over-parameterization. Journal of Machine Learning Research, 24(96):1–84, 2023.

[36] Yi-An Ma, Yuansi Chen, Chi Jin, Nicolas Flammarion, and Michael I. Jordan. Sampling can be faster than optimization. Proceedings of the National Academy of Sciences, 116(42): 20881–20885, 2019. ISSN 1091-6490. doi: 10.1073/pnas.1820003116. URL http://dx.doi. org/10.1073/pnas.1820003116.

[37] Amirkeivan Mohtashami, Martin Jaggi, and Sebastian U Stich. Large learning rate matters for non-convex optimization. OpenReview, 2023. URL https://openreview.net/forum?id= JsrvkgM8gO2.

[38] Deanna Needell, Nathan Srebro, and Rachel Ward. Stochastic gradient descent, weighted sampling, and the randomized Kaczmarz algorithm. In Advances in Neural Information Processing Systems, volume 27, 2014.

[39] Arkadi Nemirovski, Anatoli Juditsky, Guanghui Lan, and Alexander Shapiro. Robust stochastic approximation approach to stochastic programming. SIAM Journal on Optimization, 19(4): 1574–1609, 2009.

[40] Arkadi Semenovic Nemirovsky and David Borisovich Yudin.ˇ Problem complexity and method efficiency in optimization. Wiley-Interscience, 1983.

[41] Guneykan Ozgul, Xiantao Li, Mehrdad Mahdavi, and Chunhao Wang. Quantum speedups for sampling and non-convex optimization with stochastic oracles. In 21st Conference on the Theory of Quantum Computation, Communication and Cryptography (TQC 2026), volume 389 of Leibniz International Proceedings in Informatics (LIPIcs), pages 8:1–8:25. Schloss Dagstuhl – Leibniz-Zentrum für Informatik, 2026. doi: 10.4230/LIPIcs.TQC.2026.8. URL https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.TQC.2026.8.

[42] Vivak Patel and Shushu Zhang. Stochastic gradient descent on nonconvex functions with general noise models. arXiv preprint, 2021. URL https://arxiv.org/abs/2104.00423. arXiv:2104.00423.

[43] Maxim Raginsky, Alexander Rakhlin, and Matus Telgarsky. Non-convex learning via stochastic gradient Langevin dynamics: a nonasymptotic analysis. In Conference on Learning Theory, pages 1674–1703. PMLR, 2017.

[44] Alexander Rakhlin, Ohad Shamir, and Karthik Sridharan. Making gradient descent optimal for strongly convex stochastic optimization. In Proceedings ofthe 29th International Conference on Machine Learning, 2012. URL https://arxiv.org/abs/1109.5647.

[45] Yinuo Ren, Chao Ma, and Lexing Ying. Understanding the generalization benefits of late learning rate decay. In International Conference on Artificial Intelligence and Statistics, pages 4465–4473. PMLR, 2024.

[46] Herbert Robbins and Sutton Monro. A stochastic approximation method. The Annals of Mathematical Statistics, pages 400–407, 1951.

[47] Mark Schmidt and Nicolas Le Roux. Fast convergence of stochastic gradient descent under a strong growth condition. arXiv preprint, 2013. URL https://arxiv.org/abs/1308.6370. arXiv:1308.6370.

[48] Antonio Sclocchi and Matthieu Wyart. On the different regimes of stochastic gradient descent. Proceedings ofthe National Academy ofSciences, 121(9):e2316301121, 2024.

[49] Othmane Sebbouh, Robert M Gower, and Aaron Defazio. Almost sure convergence rates for stochastic gradient descent and stochastic heavy ball. In Conference on Learning Theory, pages 3935–3971. PMLR, 2021.

[50] Bin Shi, Weijie Su, and Michael I. Jordan. On learning rates and Schrödinger operators. Journal of Machine Learning Research, 24(379):1–53, 2023. URL http://jmlr.org/papers/v24/ 20-364.html.

[51] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In Advances in Neural Information Processing Systems, volume 32, 2019.

[52] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021.

[53] Albert Tarantola. Inverse Problem Theory and Methods for Model Parameter Estimation. Society for Industrial and Applied Mathematics, 2005. doi: 10.1137/1.9780898717921. URL https://epubs.siam.org/doi/abs/10.1137/1.9780898717921.

[54] Stephen A Vavasis. Complexity issues in global optimization: a survey. In Handbook ofglobal optimization, pages 27–41. Springer, 1995.

[55] Santosh S Vempala and Andre Wibisono. Rapid convergence of the unadjusted Langevin algorithm: Isoperimetry suffices. In Advances in Neural Information Processing Systems, volume 32, 2019.

[56] Jingfeng Wu, Difan Zou, Vladimir Braverman, and Quanquan Gu. Direction matters: On the implicit bias of stochastic gradient descent with moderate learning rate. In International Conference on Learning Representations, 2021. URL https://arxiv.org/abs/2011.02538.

[57] Lei Wu, Chao Ma, and Weinan E. How SGD selects the global minima in over-parameterized learning: A dynamical stability perspective. In Advances in Neural Information Processing Systems, volume 31, 2018.

[58] Kaichao You, Mingsheng Long, Jianmin Wang, and Michael I Jordan. How does learning rate decay help modern neural networks? arXiv preprint, 2019. URL https://arxiv.org/abs/ 1908.01878. arXiv:1908.01878.

[59] Yuchen Zhang, Percy Liang, and Moses Charikar. A hitting time analysis of stochastic gradient Langevin dynamics. In Proceedings ofthe 2017 Conference on Learning Theory, volume 65 of Proceedings ofMachine Learning Research, pages 1980–2022. PMLR, 2017. URL https: //proceedings.mlr.press/v65/zhang17b.html.

[60] Yi Zhou, Junjie Yang, Huishuai Zhang, Yingbin Liang, and Vahid Tarokh. SGD converges to global minimum in deep learning via star-convex path. In International Conference on Learning Representations, 2019. URL https://arxiv.org/abs/1901.00451.

[61] Difan Zou, Pan Xu, and Quanquan Gu. Faster convergence of stochastic gradient Langevin dynamics for non-log-concave sampling. In Uncertainty in Artificial Intelligence, pages 1152– 1162. PMLR, 2021.

Technical Appendices and Supplementary Material for “Learning Rate Decay Can Exponentially Accelerate SGD for Global Optimization of Nonconvex Functions”

## Contents

1 Introduction 1   
1.1 Main Result 2   
1.2 Theoretical Contributions 3   
1.3 Related Work 5   
2 Analysis of Annealed Langevin Diffusion 5   
2.1 Bounding the Weak Poincaré Constant . 6   
3 Discretization and Learning Rate Schedule 8   
Numerical Results 8   
5 Conclusion 10   
A Weak Poincaré Inequalities 16   
B Analysis of the Unadjusted Langevin Algorithm 22   
B.1 Convergence of Annealed Langevin Diffusion 24   
B.2 Discretization of Langevin Diffusion 25   
B.3 Convergence of Annealed ULA . 28   
B.4 Sampling for Block-Separable Functions . 29   
B.5 Global Optimization 32   
C Analysis of Stochastic Gradient Descent 32   
C.1 Sampling with Annealed Stochastic Gradient Descent . 33   
C.2 Sampling for Block-Separable Functions . 37   
C.3 Global Optimization 39   
D Lower Bound for SGD with Fixed Learning Rate 40   
E Additional Details for Numerical Results 43

## A Weak Poincaré Inequalities

In this section, we transfer a Poincaré inequality for a reference measure to a weak Poincaré inequality for a measure whose density ratio with respect to the reference measure is bounded above and below in a ball, and we apply this comparison to the Gibbs measures used in the ULA and SGD analyses.

For Langevin diffusion with potential $f$ at a constant noise level $\eta > 0$ , the stationary law is $\pi \propto \exp ( - 2 f / \eta )$ and its Dirichlet form is $\begin{array} { r } { \mathcal { E } _ { \pi } ( h , h ) = \frac { \eta } { 2 } \int \| \nabla h \| ^ { 2 } d \pi } \end{array}$

We say that π satisfies a $( \delta , \epsilon ) \mathrm { - } \mathsf { W P I }$ , where $\delta > 0$ and $\epsilon \geqslant 0$ , if every bounded smooth function h satisfies

$$
\begin{array} { r } { \mathcal { E } _ { \pi } ( h , h ) \geqslant \delta \operatorname { V a r } _ { \pi } ( h ) - \epsilon \| h \| _ { \infty } ^ { 2 } . } \end{array}\tag{4}
$$

By the formula for ${ \mathcal { E } } _ { \pi }$ , if every bounded smooth function h satisfies

$$
\operatorname { V a r } _ { \pi } ( h ) \leqslant C _ { P } \int \| \nabla h \| ^ { 2 } d \pi + s \| h \| _ { \infty } ^ { 2 }\tag{5}
$$

for constants $C _ { P } > 0$ and $s \geqslant 0 ,$ , then π satisfies a $( \delta , \epsilon )$ -WPI with $\delta = \eta / ( 2 C _ { P } )$ and $\epsilon = \delta s$

Theorem A.1 (Weak Poincaré Inequality via Density Ratio Perturbation). Let m be a probability measure satisfying the Poincaré inequality

$$
\gamma \operatorname { V a r } _ { m } [ h ] \leqslant \int \| \nabla h \| ^ { 2 } d m
$$

for all smooth $h ,$ where $\gamma > 0$ is the spectral gap. Let π be a probability measure absolutely continuous with respect to m, and let $\rho = d \pi / d m$ denote the Radon–Nikodym derivative. For a given $\Delta > 0$ and a point $x ^ { \star } \in \mathbb { R } ^ { d }$ , define the local density ratio bounds

$$
K _ { \operatorname* { m a x } } : = \operatorname* { s u p } _ { \| x - x ^ { \star } \| \leqslant \Delta } \rho ( x ) , \qquad K _ { \operatorname* { m i n } } : = \operatorname* { i n f } _ { \| x - x ^ { \star } \| \leqslant \Delta } \rho ( x ) ,
$$

and let $R _ { \Delta } = B ( x ^ { \star } , \Delta )$ . Then for every $\epsilon > 0 ,$ π satisfies the weak Poincaré inequality

$$
\begin{array} { r l } & { \mathrm { V a r } _ { \pi } [ h ] \leqslant \displaystyle \frac { ( 1 + \epsilon ) ^ { 2 } K _ { \operatorname* { m a x } } } { \gamma K _ { \operatorname* { m i n } } } \int \| \nabla h \| ^ { 2 } d \pi } \\ & { \quad \quad \quad \quad + \left( 4 \mathrm { P r } ( x \notin R _ { \Delta } ) + \left( 1 + \displaystyle \frac { 1 } { \epsilon } \right) \left( 1 + \displaystyle \frac { 1 6 ( 1 + \epsilon ) } { \gamma \Delta ^ { 2 } } \right) K _ { \operatorname* { m a x } } \mathrm { P r } ( x \notin R _ { \Delta / 2 } ) \right) \| h \| _ { \infty } ^ { 2 } . } \end{array}
$$

Proof. For every bounded real-valued h and constant $c ,$ expanding the square gives

$$
\mathbb { E } _ { \pi } [ ( h - c ) ^ { 2 } ] = \operatorname { V a r } _ { \pi } [ h ] + ( \mathbb { E } _ { \pi } h - c ) ^ { 2 } .
$$

Taking $c = \mathbb { E } _ { m } h$ yields

$$
\mathbb { E } _ { \pi } [ ( h - \mathbb { E } _ { m } h ) ^ { 2 } ] = \operatorname { V a r } _ { \pi } [ h ] + ( \mathbb { E } _ { \pi } h - \mathbb { E } _ { m } h ) ^ { 2 } \geqslant \operatorname { V a r } _ { \pi } [ h ] .
$$

We split this upper bound over the ball $R _ { \Delta }$ and its complement:

$$
\begin{array} { r l r } {  { \operatorname { V a r } _ { \pi } [ h ] \leqslant \mathbb { E } _ { \pi } \| h - \mathbb { E } _ { m } h \| ^ { 2 } } } \\ & { } & { = \int _ { \| x - x ^ { \star } \| \leqslant \Delta } \| h ( x ) - \mathbb { E } _ { m } h \| ^ { 2 } d \pi ( x ) + \int _ { \| x - x ^ { \star } \| > \Delta } \| h ( x ) - \mathbb { E } _ { m } h \| ^ { 2 } d \pi ( x ) } \\ & { } & { \leqslant K _ { \operatorname* { m a x } } \mathbb { E } _ { m } \| h - \mathbb { E } _ { m } h \| ^ { 2 } + 4 \operatorname* { P r } ( \| x - x ^ { \star } \| > \Delta ) \| h \| _ { \infty } ^ { 2 } . } \end{array}
$$

It remains to bound $\mathrm { V a r } _ { m } [ h ]$ in terms of the Dirichlet form under π. Let $\phi : [ 0 , \infty )  [ 0 , 1 ]$ be a smooth non-increasing function satisfying $\phi ( t ) = 1$ for $\xi \leqslant 1 , \phi ( t ) = 0$ for $t \geqslant 2 ,$ and $| \bar { \phi } ^ { \prime } ( t ) \bar { | } \leqslant 2$ for all t. Define the cutoff $\chi ( x ) = \phi ( \bar { 2 } \| x ^ { - } x ^ { \star } \| / \Delta )$ ), so that $\chi = 1$ on $R _ { \Delta / 2 }$ , supp $( \chi ) \subseteq R _ { \Delta }$ , and $\| \nabla \chi ( x ) \| \leqslant 4 / \Delta$ . Fix $\epsilon > 0$ , and recall that $\| u + v \| ^ { 2 } \leqslant ( 1 + \epsilon ) \| u \| ^ { 2 } + ( 1 + 1 / \epsilon ) \| v \| ^ { 2 }$ for all vectors $u , v$ . The same expansion of the square, applied under m with $c = \mathbb { E } _ { m } ( h \chi )$ , justifies the first inequality next.

$$
\begin{array} { r l } & { \mathbb { E } _ { m } \| h - \mathbb { E } _ { m } h \| ^ { 2 } \leqslant \mathbb { E } _ { m } \| h - \mathbb { E } _ { m } ( h \chi ) \| ^ { 2 } } \\ & { \qquad \leqslant \Big ( 1 + \frac { 1 } { \epsilon } \Big ) \mathbb { E } _ { m } \| h ( 1 - \chi ) \| ^ { 2 } + ( 1 + \epsilon ) \mathbb { E } _ { m } \| h \chi - \mathbb { E } _ { m } ( h \chi ) \| ^ { 2 } } \\ & { \qquad \leqslant \Big ( 1 + \frac { 1 } { \epsilon } \Big ) \operatorname* { P r } \big ( \| x - x ^ { \star } \| > \Delta / 2 \big ) \| h \| _ { \infty } ^ { 2 } + ( 1 + \epsilon ) \mathbb { E } _ { m } \| h \chi - \mathbb { E } _ { m } ( h \chi ) \| ^ { 2 } . } \end{array}
$$

For the second term, we apply the Poincaré inequality for m to $h \chi$ and expand:

$$
\begin{array} { l } { \displaystyle \mathbb { E } _ { m } \| h \chi - \mathbb { E } _ { m } ( h \chi ) \| ^ { 2 } \leqslant \displaystyle \frac { 1 } { \gamma } \int \| \nabla ( h \chi ) \| ^ { 2 } d m = \displaystyle \frac { 1 } { \gamma } \int \| h \nabla \chi + \chi \nabla h \| ^ { 2 } d m } \\ { \displaystyle \leqslant \frac { 1 + \epsilon } { \gamma } \int \| \nabla h \| ^ { 2 } \chi ^ { 2 } d m + \frac { 1 + 1 / \epsilon } { \gamma } \int \| \nabla \chi \| ^ { 2 } h ^ { 2 } d m } \\ { \displaystyle \leqslant \frac { 1 + \epsilon } { \gamma } \int _ { \| x - x ^ { \star } \| \leqslant \Delta } \| \nabla h \| ^ { 2 } d m + \frac { ( 1 + 1 / \epsilon ) \| h \| _ { \infty } ^ { 2 } } { \gamma } \int \| \nabla \chi \| ^ { 2 } d m . } \end{array}
$$

Since $\chi$ is supported on $B ( x ^ { \star } , \Delta )$ and $\rho \geqslant K _ { \operatorname* { m i n } }$ therein, the first integral satisfies

$$
\int _ { \| x - x ^ { \star } \| \leqslant \Delta } \| \nabla h \| ^ { 2 } d m \leqslant \frac { 1 } { K _ { \operatorname* { m i n } } } \int \| \nabla h \| ^ { 2 } d \pi .
$$

For the gradient of $\chi ,$ we have $\| \nabla \chi ( x ) \| ^ { 2 } \leqslant 1 6 / \Delta ^ { 2 }$ whenever $\Delta / 2 < \| x - x ^ { \star } \| < \Delta$ , and $\nabla \chi = 0$ otherwise. Therefore,

$$
\mathbb { E } _ { m } \| h \chi - \mathbb { E } _ { m } ( h \chi ) \| ^ { 2 } \leqslant \frac { 1 + \epsilon } { \gamma K _ { \operatorname* { m i n } } } \int \| \nabla h \| ^ { 2 } d \pi + \frac { 1 6 ( 1 + 1 / \epsilon ) \| h \| _ { \infty } ^ { 2 } } { \gamma \Delta ^ { 2 } } \operatorname* { P r } _ { m } \bigl ( \| x - x ^ { \star } \| > \Delta / 2 \bigr ) .
$$

Combining, we arrive at

$$
\begin{array} { r l } & { \mathrm { V a r } _ { \pi } [ h ] \leqslant \displaystyle \frac { ( 1 + \epsilon ) ^ { 2 } K _ { \operatorname* { m a x } } } { \gamma K _ { \operatorname* { m i n } } } \int \| \nabla h \| ^ { 2 } d \pi + 4 \displaystyle \operatorname* { P r } _ { \pi } ( \| x - x ^ { \star } \| > \Delta ) \| h \| _ { \infty } ^ { 2 } } \\ & { \quad \quad \quad + \left( 1 + \displaystyle \frac { 1 } { \epsilon } \right) \left( 1 + \displaystyle \frac { 1 6 ( 1 + \epsilon ) } { \gamma \Delta ^ { 2 } } \right) K _ { \operatorname* { m a x } } } \\ & { \quad \quad \quad \quad \times \displaystyle \operatorname* { P r } ( \| x - x ^ { \star } \| > \Delta / 2 ) \| h \| _ { \infty } ^ { 2 } , } \end{array}
$$

which concludes the proof.

The Gaussian tail estimate below is used in the density-ratio and weak Poincaré bounds that follow. Lemma ${ \bf A } . 2$ (Gaussian tail integrals). Let $\ell \geqslant 1 , a \in \mathbb { R } ^ { \ell }$ and $\lambda > 0 ,$ , and let $\gamma _ { \lambda } = \mathcal { N } ( a , \lambda ^ { - 1 } I _ { \ell } )$ . For $r > 0$ with $\lambda r ^ { 2 } \geqslant 4 \ell$

$$
\begin{array} { l } { \displaystyle \gamma _ { \lambda } ( \| x - a \| > r ) = \left( \frac { \lambda } { 2 \pi } \right) ^ { \ell / 2 } \int _ { \| x - a \| > r } e ^ { - \lambda \| x - a \| ^ { 2 } / 2 } d x , } \\ { \displaystyle \leqslant e ^ { - \lambda r ^ { 2 } / 8 } . } \end{array}\tag{6}
$$

Proof. For $Z \sim { \mathcal { N } } ( 0 , I _ { \ell } )$ , Gaussian integration gives $\mathbb { E } e ^ { \| Z \| ^ { 2 } / 4 } = 2 ^ { \ell / 2 }$ . Markov’s inequality with $t = \dot { \lambda } r ^ { 2 } \geqslant 4 \ell$ yields

$$
\mathbb { P } ( \| Z \| ^ { 2 } > t ) \leqslant 2 ^ { \ell / 2 } e ^ { - t / 4 } \leqslant e ^ { - t / 8 } .
$$

The substitution $x = a + \lambda ^ { - 1 / 2 } Z$ gives (6), including the Gaussian normalizing factor.

Lemma A.3 (Density Ratio Bounds Near the Global Minimum). Let $f : \mathbb { R } ^ { d }  \mathbb { R }$ be $a ~ C ^ { 3 }$ function with a unique global minimizer $x ^ { \star } \ s a t i s f y$ ing $f ( x ^ { \star } ) = 0$ and $H : = \nabla ^ { 2 } f ( x ^ { \star } ) \succ 0$ , and let $\omega : = \lambda _ { \operatorname* { m i n } } ( H ) > 0$ . Suppose that $\begin{array} { r } { f ( x ) \geqslant \frac { a } { 2 } \| x - x ^ { \star } \| ^ { 2 } . } \end{array}$ for some $a > 0$ and all $x \in \mathbb { R } ^ { d }$ . Define the Gibbs measure $\tau ( x ) \propto \exp ( - \beta f ( x ) )$ ) and the comparison Gaussian m(x) ∝ exp(−βq(x)), where $\begin{array} { r } { q ( x ) : = \frac { 1 } { 2 } \langle x - x ^ { \star } , H ( x - x ^ { \star } ) \rangle } \end{array}$ , so that $m = \mathcal { N } ( \boldsymbol { x } ^ { \star } , ( \beta H ) ^ { - 1 } )$ . Let $\Delta = \beta ^ { - 5 / 1 2 }$ and $R _ { \Delta } = B ( x ^ { \star } , \Delta )$ Then the density ratio $\rho _ { \beta } = d \pi / d m$ satisfies

$$
1 - \mathcal { O } ( \beta ^ { - 1 / 4 } ) \leqslant \rho _ { \beta } ( x ) \leqslant 1 + \mathcal { O } ( \beta ^ { - 1 / 4 } ) , \qquad \forall x \in R _ { \Delta } ,
$$

for all $\beta$ sufficiently large.

Proof. The density ratio factorizes as

$$
\rho _ { \beta } ( x ) = \frac { d \pi } { d m } ( x ) = \frac { Z _ { m } } { Z _ { \pi } } \exp \big ( - \beta ( f ( x ) - q ( x ) ) \big ) .
$$

Since m is a Gaussian with covariance $( \beta H ) ^ { - 1 }$ , its normalizing constant is $\begin{array} { r l } { Z _ { m } } & { { } = } \end{array}$ $( 2 \pi / \beta ) ^ { d / 2 } ( \operatorname* { d e t } H ) ^ { - 1 / 2 }$ . For $\begin{array} { r } { Z _ { \pi } = \int \exp ( - \beta f ( x ) ) d x , } \end{array}$ we split the integral over $R _ { \Delta }$ and its complement. On $R _ { \Delta }$ , the estimate (8) below for $\exp ( - \dot { \beta } ( f ( x ) - \dot { q ( x ) } ) )$ gives

$$
\int _ { R _ { \Delta } } \exp ( - \beta f ( x ) ) d x = \left( 1 + \mathcal { O } ( \beta ^ { - 1 / 4 } ) \right) Z _ { m } \bigl ( 1 - \operatorname* { P r } _ { m } ( x \not \in R _ { \Delta } ) \bigr ) .
$$

Writing $x = x ^ { \star } + ( \beta H ) ^ { - 1 / 2 } Z$ with $Z \sim { \mathcal { N } } ( 0 , I _ { d } )$ under $m ,$ we have $\| x - x ^ { \star } \| ^ { 2 } \leqslant \| Z \| ^ { 2 } / ( \beta \omega )$ , so Theorem $\mathrm { A } . 2$ gives $\operatorname* { P r } _ { m } ( x \notin { R _ { \Delta } } ) \leqslant \operatorname* { P r } ( \| Z \| ^ { 2 } > \omega \beta ^ { 1 / 6 } ) \leqslant \exp ( - \omega \beta ^ { 1 / 6 } / 8 )$ once $\omega \beta ^ { 1 / 6 } \geqslant 4 d .$ . On the complement of $R _ { \Delta }$ , the bound $f ( x ) \geqslant { \frac { a } { 2 } } \| x - x ^ { \star } \| ^ { 2 }$ and Theorem $\mathrm { A } . 2$ applied to $\mathcal { N } ( x ^ { \star } , ( \beta a ) ^ { - 1 } I _ { d } )$ give

$$
\int _ { \| x - x ^ { \star } \| > \Delta } \exp ( - \beta f ( x ) ) d x \leqslant Z _ { m } { \Big ( } { \frac { \operatorname* { d e t } H } { a ^ { d } } } { \Big ) } ^ { 1 / 2 } \exp { \big ( } - a \beta ^ { 1 / 6 } / 8 { \big ) }\tag{7}
$$

once $a \beta ^ { 1 / 6 } \geqslant 4 d .$ . Since $\exp ( - \omega \beta ^ { 1 / 6 } / 8 )$ and $\exp ( - a \beta ^ { 1 / 6 } / 8 )$ are $\mathcal { O } ( \beta ^ { - 1 / 4 } )$ , adding the two parts of $Z _ { \pi }$ gives $Z _ { \pi } = Z _ { m } ( 1 + \mathcal { O } ( \beta ^ { - 1 / 4 } ) )$ . Therefore,

$$
\frac { Z _ { m } } { Z _ { \pi } } = 1 + \mathcal { O } ( \beta ^ { - 1 / 4 } ) .
$$

For the exponential term, by Taylor’s theorem with remainder, since $f ( x ^ { \star } ) = 0 , \nabla f ( x ^ { \star } ) = 0$ , and $\nabla ^ { 2 } f ( x ^ { \star } ) \stackrel { } { = } H$ , we have

$$
f ( \boldsymbol { x } ) - q ( \boldsymbol { x } ) = \mathcal { O } \big ( \| \boldsymbol { x } - \boldsymbol { x } ^ { \star } \| ^ { 3 } \big ) ,
$$

with $\begin{array} { r } { | f ( x ) - q ( x ) | \leqslant \frac { M _ { 3 } } { 6 } \| x - x ^ { \star } \| ^ { 3 } } \end{array}$ for all $x \in B ( x ^ { \star } , 1 )$ , where $M _ { 3 } = \operatorname* { s u p } _ { B ( x ^ { \star } , 1 ) } \| \nabla ^ { 3 } f \| _ { \mathrm { o p } }$ is finite since $f$ is $C ^ { 3 }$ . For $x \in R _ { \Delta }$ with $\Delta = \beta ^ { - 5 / 1 2 } \leqslant 1 \colon$

$$
\beta \left| f ( x ) - q ( x ) \right| \leqslant \frac { \beta M _ { 3 } } { 6 } \Delta ^ { 3 } = \frac { M _ { 3 } } { 6 } \beta ^ { - 1 / 4 } .
$$

Since $| e ^ { - u } - 1 | \leqslant 2 | u |$ for $| u | \leqslant 1$ , we conclude that for $\beta$ large enough,

$$
\exp \big ( - \beta ( f ( x ) - q ( x ) ) \big ) = 1 + \mathcal { O } ( \beta ^ { - 1 / 4 } ) .\tag{8}
$$

Multiplying the two estimates gives $\rho _ { \beta } ( x ) = ( 1 + \mathcal { O } ( \beta ^ { - 1 / 4 } ) ) ( 1 + \mathcal { O } ( \beta ^ { - 1 / 4 } ) ) = 1 + \mathcal { O } ( \beta ^ { - 1 / 4 } )$ for all $x \in R _ { \Delta }$ , completing the proof. □

Theorem A.4 (Weak Poincaré Bound for Low Temperature Gibbs Distributions). Let $f : \mathbb { R } ^ { d } $ R be a $C ^ { 3 }$ function with a unique global minimizer $\cdot \boldsymbol { x } ^ { \star }$ and $H : = \nabla ^ { 2 } f ( x ^ { \star } ) \succ 0 ,$ , and let $\omega : = \lambda _ { \operatorname* { m i n } } ( H ) > 0$ Suppose additionally that f has quadratic growth away from $x ^ { \star }$ , i.e., $\begin{array} { r } { f ( x ) - f ( x ^ { \star } ) \geqslant \frac { a } { 2 } \| x - x ^ { \star } \| ^ { 2 } } \end{array}$ for some $a > 0$ and all $x \in \mathbb { R } ^ { d }$ . Then for all β sufficiently large, the Gibbs measure $\pi _ { \beta } ( x )$ ∝ $\exp ( - \beta f ( x ) )$ satisfies the weak Poincaré inequality

$$
\mathrm { V a r } _ { \pi _ { \beta } } [ h ] \leqslant { \frac { 1 + O ( \beta ^ { - 1 / 4 } ) } { \omega \beta } } \int \| \nabla h \| ^ { 2 } d \pi _ { \beta } + C \exp \bigl ( - c \beta ^ { 1 / 6 } \bigr ) \| h \| _ { \infty } ^ { 2 } ,
$$

where $c , C > 0$ are constants depending only on $H , a$ and d.

Proof. We apply Theorem A.1 with the comparison Gaussian $\begin{array} { r } { m ( x ) \propto \exp \bigl ( - \frac { \beta } { 2 } \langle x - x ^ { \star } , H ( x - x ^ { \star } ) \rangle \bigr ) } \end{array}$ $\Delta = \beta ^ { - 5 / 1 2 }$ and $\epsilon = \beta ^ { - 1 / 4 }$ . Since adding a constant to $f$ does not change $\pi _ { \beta } .$ , we may assume $f ( x ^ { \star } ) = 0$ , so that f satisfies the assumptions of Theorem ${ \mathrm { A } } . 3$

The measure m is $\mathcal { N } ( x ^ { \star } , ( \beta H ) ^ { - 1 } )$ , which satisfies the Poincaré inequality with spectral $\displaystyle \operatorname { g a p } \gamma = \beta \omega$ By Theorem A.3, for all $x \in R _ { \Delta } = B ( x ^ { \star } , \Delta )$ (with $\Delta = \beta ^ { - 5 / 1 2 }$ , which is still $o ( 1 )$ as $\beta \to \infty )$ , we have

$$
K _ { \operatorname* { m i n } } = 1 - \mathcal { O } ( \beta ^ { - 1 / 4 } ) , \qquad K _ { \operatorname* { m a x } } = 1 + \mathcal { O } ( \beta ^ { - 1 / 4 } ) .
$$

Substituting these bounds into the Dirichlet coefficient from Theorem A.1 gives

$$
\frac { ( 1 + \epsilon ) ^ { 2 } K _ { \mathrm { m a x } } } { \gamma K _ { \mathrm { m i n } } } = \frac { ( 1 + \beta ^ { - 1 / 4 } ) ^ { 2 } ( 1 + \mathcal { O } ( \beta ^ { - 1 / 4 } ) ) } { \beta \omega ( 1 - \mathcal { O } ( \beta ^ { - 1 / 4 } ) ) } = \frac { 1 + \mathcal { O } ( \beta ^ { - 1 / 4 } ) } { \omega \beta } .
$$

Since $\beta \omega \| x - x ^ { \star } \| ^ { 2 } \leqslant \| Z \| ^ { 2 }$ under $m _ { : }$ where $x = x ^ { \star } + ( \beta H ) ^ { - 1 / 2 } Z$ with $Z \sim { \mathcal { N } } ( 0 , I _ { d } )$ , we have

$$
\operatorname* { P r } _ { m } ( \| x - x ^ { \star } \| > \Delta / 2 ) \leqslant \operatorname* { P r } \bigl ( \| Z \| ^ { 2 } > \beta \omega \Delta ^ { 2 } / 4 \bigr ) = \operatorname* { P r } \bigl ( \| Z \| ^ { 2 } > \omega \beta ^ { 1 / 6 } / 4 \bigr ) .
$$

By Theorem $\mathbf { A . } 2 ,$ for $\beta$ large enough that $\omega \beta ^ { 1 / 6 } ~ \geqslant ~ 1 6 d .$ , this gives $\mathrm { P r } _ { m } ( x \notin \ R _ { \Delta / 2 } ) \ \leqslant$ $\exp ( - \omega \beta ^ { 1 / 6 } / 3 2 )$

Since $\textstyle f ( x ) \geq { \frac { a } { 2 } } \| x - x ^ { \star } \| ^ { 2 }$ globally, the Gibbs measure $\pi _ { \beta }$ is dominated by a Gaussian in the tails. In particular, the normalizing constants of $\pi _ { \beta }$ and m satisfy $Z _ { \pi } = Z _ { m } ( 1 + \mathcal { O } ( \beta ^ { - 1 / 4 } ) ) \geqslant Z _ { m } / 2$ for $\beta$ sufficiently large by the proof of Theorem $\mathbf { A . } 3 .$ , and dividing (7) by $Z _ { \pi }$ gives $\mathrm { P r } _ { \pi _ { \beta } } ( \| x - x ^ { \star } \| >$ $\Delta ) \leqslant 2 ( \operatorname* { d e t } H / a ^ { d } ) ^ { 1 / 2 } \exp ( - a \beta ^ { 1 / 6 } / 8 )$ once $a \beta ^ { 1 / 6 } \geqslant 4 d .$

The coefficient of $\| h \| _ { \infty } ^ { 2 }$ in Theorem A.1 becomes

$$
4 \operatorname* { P r } _ { \pi _ { \beta } } ( x \notin R _ { \Delta } ) + \Big ( 1 + \frac { 1 } { \epsilon } \Big ) \Big ( 1 + \frac { 1 6 ( 1 + \epsilon ) } { \gamma \Delta ^ { 2 } } \Big ) K _ { \operatorname* { m a x } } \operatorname* { P r } _ { m } ( x \notin R _ { \Delta / 2 } ) .
$$

Since $\gamma \Delta ^ { 2 } = \beta \omega \cdot \beta ^ { - 5 / 6 } = \omega \beta ^ { 1 / 6 }$ and $\epsilon = \beta ^ { - 1 / 4 }$ , the prefactor satisfies

$$
\bigl ( 1 + \beta ^ { 1 / 4 } \bigr ) \Bigl ( 1 + \frac { 1 6 \bigl ( 1 + \beta ^ { - 1 / 4 } \bigr ) } { \omega \beta ^ { 1 / 6 } } \Bigr ) K _ { \mathrm { m a x } } \leqslant 3 \beta ^ { 1 / 4 }
$$

for $\beta$ sufficiently large. Therefore, the full $\| h \| _ { \infty } ^ { 2 }$ coefficient is at most 8(det $H / a ^ { d } ) ^ { 1 / 2 } \exp ( - a \beta ^ { 1 / 6 } / 8 ) + 3 \beta ^ { 1 / 4 } \exp ( - \omega \beta ^ { 1 / 6 } / 3 2 )$ Absorbing the factor $\beta ^ { 1 / 4 }$ into the exponential shows that this coefficient is bounded by $\dot { C } \exp ( - c \beta ^ { 1 / 6 } )$ with $c = \operatorname* { m i n } \{ a / 8 , \omega / 6 4 \}$ , concluding the proof. 口

We bound the initial density ratio and the Gibbs tail probabilities, and apply Theorem A.1 to obtain the weak Poincaré inequality required by Theorem B.8.

Lemma A.5 (Density ratios, tail probabilities, and a weak Poincaré inequality). Let $g \in C ^ { 2 } ( \mathbb { R } ^ { \ell } ) ,$ where $1 \leqslant \ell \leqslant b ,$ , have an L-Lipschitz gradient and a minimizer a with $g ( a ) = 0 , \| a \| \leqslant B _ { 0 } ,$ , and $g ( y ) \geqslant m \| y - a \| ^ { 2 } / 2 .$ For $\beta > 0 ,$ , define thefollowing distributions and constants:

$$
\begin{array} { c c } { { Z _ { \beta } : = \displaystyle \int _ { \mathbb { R } ^ { \ell } } e ^ { - \beta g ( y ) } d y , } } \\ { { \displaystyle \pi _ { \beta } ( y ) : = Z _ { \beta } ^ { - 1 } e ^ { - \beta g ( y ) } , } } & { { \qquad \gamma _ { 0 } : = N ( 0 , ( 2 L ) ^ { - 1 } I _ { \ell } ) . } } \\ { { \kappa : = ( L / m ) ^ { b / 2 } , } } & { { \qquad A : = ( 2 L / m ) ^ { b / 2 } , } } \\ { { K _ { 0 } : = A e ^ { L B _ { 0 } ^ { 2 } } , } } & { { \qquad U : = 1 + \log K _ { 0 } . } } \end{array}\tag{9}
$$

These choices give the bounds

$$
\frac { m } { 2 } \lVert y - a \rVert ^ { 2 } \leqslant g ( y ) \leqslant \frac { L } { 2 } \lVert y - a \rVert ^ { 2 } ,\tag{10}
$$

(11)

$$
{ \frac { Z _ { \beta / 2 } } { Z _ { \beta } } } \leqslant A , \qquad \pi _ { \beta } ( g > u ) \leqslant A e ^ { - \beta u / 2 } \quad ( \beta , u > 0 ) .\tag{12}
$$

For $s \in ( 0 , 1 ]$ , define

$$
S : = \operatorname* { m a x } \{ 3 2 , 1 6 b , 3 2 \log ( 8 \kappa / s ) \} ,
$$

$$
\delta : = \frac { m } { 4 \kappa } e ^ { - ( L / m - 1 ) S / 2 } .\tag{13}
$$

The generator Dirichletform $\begin{array} { r } { \mathcal { E } _ { \pi _ { \beta } } ( v , v ) = \beta ^ { - 1 } \int \| \nabla v \| ^ { 2 } d \pi _ { \beta } } \end{array}$ satisfies

$$
\begin{array} { r } { \mathcal { E } _ { \pi _ { \beta } } ( v , v ) \geqslant \delta \operatorname { V a r } _ { \pi _ { \beta } } ( v ) - \delta s \| v \| _ { \infty } ^ { 2 } . } \end{array}\tag{14}
$$

For fixed $b , m , L ,$ the inverse of δ is polynomial in $1 / s$

Proof. Taylor’s theorem gives (10). Gaussian integration therefore bounds $Z _ { \beta }$ between $( 2 \pi / ( \beta L ) ) ^ { \ell / 2 }$ and $( 2 \pi / ( \beta m ) ) ^ { \ell / 2 }$ . Completing the square gives

$$
g ( y ) - L \| y \| ^ { 2 } \leqslant L \| a \| ^ { 2 } - { \frac { L } { 2 } } \| y + a \| ^ { 2 } ,
$$

which proves (11). The ratio of the Gaussian normalizer bounds is at most A. Applying Markov’s inequality to $e ^ { \beta g / 2 }$ under $\pi _ { \beta }$ proves (12).

For (14), apply Theorem A.1 with comparison measure $m _ { \beta } = \mathcal { N } ( a , ( \beta m ) ^ { - 1 } I _ { \ell } )$ , comparison param eter $\epsilon = 1$ , and radius $r _ { \beta } = \sqrt { S / ( \beta m ) }$ . Equation (10) gives the required density-ratio bounds:

$$
\begin{array} { r l r } {  { \frac { d \pi _ { \beta } } { d m _ { \beta } } \leqslant \kappa , } } & { { } } & { K _ { \mathrm { m i n } } \geqslant e ^ { - ( L / m - 1 ) S / 2 } , } \\ & { { } } & { \gamma r _ { \beta } ^ { 2 } = S , } \end{array}\tag{15}
$$

where $\gamma = \beta m$ . Applying Theorem ${ \tt A . 2 }$ to the comparison Gaussian at radii $r _ { \beta }$ and $r _ { \beta } / 2$ bounds the defect in that comparison by

$$
4 \kappa e ^ { - S / 8 } + 2 ( 1 + 3 2 / S ) \kappa e ^ { - S / 3 2 } \leqslant 8 \kappa e ^ { - S / 3 2 } \leqslant s .\tag{16}
$$

Substituting (15) and (16) into Theorem ${ \mathrm { A . 1 } }$ , and using the normalization (5), proves (14). The dependence on s follows directly from (13). □

At low temperature, quadratic growth of a block potential gives a weak Poincaré inequality with an exponentially small defect.

Corollary A.6 (Weak Poincaré Bound with Exponentially Small Defect). Let $\mathcal { X } ^ { \prime } \subseteq \mathbb { R } ^ { d ^ { \prime } }$ be a convex domain, and let $g \in C ^ { 2 } ( \mathcal { X } ^ { \prime } )$ have a unique global minimizer $x ^ { \star }$ , which lies in the interior of $\mathcal { X } ^ { \prime }$ and satisfy $g ( x ) { \overset {  } { - } } g ( x ^ { \star } ) \ \overset {  } { \geqslant } \ { \frac { m } { 2 } } \| x - x ^ { \star } \| ^ { 2 } { \overset {  } { f o r } }$ some $m > 0$ and all $x \in \mathcal { X } ^ { \prime }$ . Then there exist constants $\beta _ { 0 } , \xi , c , C > 0 ,$ , depending only on g and $\mathcal { X } ^ { \prime }$ , such that for all $\beta \geqslant \beta _ { 0 }$ , the Gibbs measure $\pi _ { \beta } \propto \exp ( - \beta g )$ on $\mathcal { X } ^ { \prime }$ satisfies

$$
\mathrm { V a r } _ { \pi _ { \beta } } [ h ] \leqslant \frac { 4 } { \xi \beta } \int \| \nabla h \| ^ { 2 } d \pi _ { \beta } + C e ^ { - c \beta } \| h \| _ { \infty } ^ { 2 }
$$

for all bounded smooth h.

Proof. Adding a constant to $g$ does not change $\pi _ { \beta } .$ , so we may assume $g ( x ^ { \star } ) = 0$ . Choose $r _ { 0 } > 0$ and $\xi > 0$ such that the closed ball $B : = \{ x : \| x - x ^ { \star } \| \leqslant r _ { 0 } \}$ lies in $\mathcal { X } ^ { \prime }$ and $\nabla ^ { 2 } g \succeq \xi I$ on $B { \mathrm { , } }$ , and let $L _ { 0 } : = \operatorname* { m a x } _ { x \in B } \| \nabla ^ { 2 } g ( x ) \|$ . We compare $\pi _ { \beta }$ with the Gibbs measure of the function

$$
\tilde { g } ( x ) : = \operatorname* { s u p } _ { y \in B } \Big \{ g ( y ) + \langle \nabla g ( y ) , x - y \rangle + \frac { \xi } { 2 } \| x - y \| ^ { 2 } \Big \} , \qquad x \in \mathbb { R } ^ { d ^ { \prime } } .
$$

As a supremum of ξ-strongly convex functions, $\tilde { g }$ is ξ-strongly convex. Since $\nabla ^ { 2 } g \succeq \xi I$ on the convex set $B _ { ; }$ , we have $\tilde { g } = g$ on $B ,$ and the choice $y = x ^ { \star }$ , at which $g$ and $\nabla g$ vanish, gives $\tilde { g } ( \boldsymbol { x } ) \geqslant \frac { \xi } { 2 } \| \boldsymbol { x } - \boldsymbol { x } ^ { \star } \| ^ { 2 }$

Let $m _ { \beta } \propto \exp ( - \beta \tilde { g } )$ on $\mathcal { X } ^ { \prime }$ , and let $Z _ { m }$ and $Z _ { \pi }$ be the normalizing constants of $m _ { \beta }$ and $\pi _ { \beta }$ . Since $\beta \tilde { g }$ is βξ-strongly convex and $\mathcal { X } ^ { \prime }$ is convex, $m _ { \beta }$ satisfies the Poincaré inequality with spectral gap $\gamma = \beta \xi$ by the Bakry–Émery criterion [6], applied to smooth approximations of $\tilde { g } \mathrm { i f } \tilde { g }$ is not smooth. The density ratio $\begin{array} { r } { d \bar { \pi _ { \beta } } / d m _ { \beta } \bar { = } \frac { Z _ { m } } { Z _ { \tau } } e ^ { - \beta \bar { ( } g - \tilde { g } ) } } \end{array}$ equals the constant $Z _ { m } / Z _ { \pi }$ on $B .$

We apply Theorem ${ \mathrm { A . 1 } }$ to $\pi _ { \beta }$ with the reference measure $m _ { \beta }$ , with $\epsilon = 1$ and with $\Delta = r _ { 0 }$ , so that $R _ { \Delta } = B$ and $K _ { \mathrm { m a x } } = K _ { \mathrm { m i n } } = Z _ { m } / Z _ { \pi }$ . The coefficient of $\int \| \nabla h \| ^ { 2 } d \pi _ { \beta }$ is then $4 / ( \beta \xi )$ , and the coefficient of $\| h \| _ { \infty } ^ { 2 }$ is

$$
+ 2 \frac { Z _ { m } } { Z _ { \pi } } \Big ( 1 + \frac { 3 2 } { \beta \xi r _ { 0 } ^ { 2 } } \Big ) \operatorname* { P r } _ { m _ { \beta } } ( \| x - x ^ { \star } \| > r _ { 0 } / 2 ) .
$$

It remains to bound the two probabilities and the ratio $Z _ { m } / Z _ { \pi }$ . Let $\phi = g$ and $a = m , \mathbf { o r } \phi = \tilde { g }$ and $a = \xi$ , and let $P _ { \phi } \propto e ^ { - \beta \phi }$ on $\mathcal { X } ^ { \prime }$ , so that $P _ { g } = \pi _ { \beta }$ and $P _ { \tilde { g } } = m _ { \beta }$ . Then $\begin{array} { r } { \phi ( x ) \geqslant \frac { a } { 2 } \| x - x ^ { \star } \| ^ { 2 } } \end{array}$ on $\mathcal { X } ^ { \prime } .$ , and $\begin{array} { r } { \phi ( x ) \leqslant \frac { L _ { 0 } } { 2 } \| x - x ^ { \star } \| ^ { 2 } } \end{array}$ on B by Taylor’s theorem, since $\phi = g$ on $B .$ . For $s \in \{ r _ { 0 } , r _ { 0 } / 2 \}$ Theorem $\mathrm { A } . 2$ bounds the numerator of $P _ { \phi } ( \| x - x ^ { \star } \| > s )$ by the Gaussian integral of $e ^ { - \beta a \| x - x ^ { \star } \| ^ { 2 } / 2 }$ over $\{ \| x - x ^ { \star } \| > s \}$ , and its denominator is bounded below by the integral of $e ^ { - \beta L _ { 0 } \| x - x ^ { \star } \| ^ { 2 } / 2 }$ over $B ,$ which is at least half of the corresponding Gaussian integral over $\mathbb { R } ^ { d ^ { \prime } }$ for $\beta$ large. This gives, for $\beta$ large enough,

$$
P _ { \phi } ( \| x - x ^ { \star } \| > s ) \leqslant 2 \Bigl ( \frac { L _ { 0 } } { a } \Bigr ) ^ { d ^ { \prime } / 2 } \exp \Big ( - \frac { \beta a s ^ { 2 } } { 8 } \Big ) .
$$

For $\phi = \widetilde g$ and $s = r _ { 0 }$ , the right-hand side is at most $1 / 2$ for $\beta$ large, so $m _ { \beta } ( B ) \geqslant 1 / 2$ . Since $\tilde { g } = g$ on $B .$ , it follows that $\begin{array} { r } { Z _ { \pi } \geqslant \int _ { B } e ^ { - \beta g } = Z _ { m } m _ { \beta } ( B ) \geqslant Z _ { m } / 2 } \end{array}$ , that is, $Z _ { m } / Z _ { \pi } \leqslant 2$

Inserting these bounds, with $\left( \phi , s \right) = \left( g , r _ { 0 } \right)$ in the first term and $( \phi , s ) = ( \tilde { g } , r _ { 0 } / 2 )$ in the second, bounds the coefficient of $\| h \| _ { \infty } ^ { 2 }  { \mathrm { b y } } C e ^ { - c \beta }$ for all $\beta \geqslant \beta _ { 0 }$ , where $c =$ min $\{ m r _ { 0 } ^ { 2 } / 8 , \xi r _ { 0 } ^ { 2 } / 3 2 \} , \beta _ { 0 }$ is large enough for the bounds above, and C depends only on $d ^ { \prime } , m , \xi , L _ { 0 } , r _ { 0 }$ and $\beta _ { 0 }$ □

## B Analysis of the Unadjusted Langevin Algorithm

We consider an annealed version of Langevin dynamics of the form

$$
d X _ { t } = - \nabla f ( X _ { t } ) d t + \sqrt { \eta ( \theta _ { t } ) } d B _ { t } ,\tag{17}
$$

where $\theta : [ 0 , T ]  \mathbb { R } _ { > 0 }$ is a learning rate schedule. We denote the law of this process at time $T$ by $\mu _ { T }$ . Fix a decreasing sequence of learning rates $\eta _ { 1 } > \eta _ { 2 } > \cdots > \eta _ { M } > 0$ . In phase $k = 1 , \dots , \dot { M }$ we run with constant learning rate $\eta _ { k }$ for a continuous-time duration $\tau _ { k } ~ ( \mathrm { i . e . }$ . a piecewise-constant schedule). The duration $\tau _ { k }$ of each phase is chosen so that the Langevin SDE at constant learning rate $\eta _ { k }$

$$
d X _ { t } = - \nabla f ( X _ { t } ) { \mathrm { d } } t + { \sqrt { \eta _ { k } } } d B _ { t } ,\tag{18}
$$

has approximately reached its stationary distribution $\pi _ { \eta _ { k } } \propto \exp ( - 2 f / \eta _ { k } )$ . Formally, the learning rate schedule is piecewise constant: if $\begin{array} { r } { T _ { k } = \sum _ { m = 1 } ^ { k } \tau _ { m } . } \end{array}$ , then $\forall t \in [ T _ { k - 1 } , T _ { k } ] , \eta ( \theta _ { t } ) = \eta _ { k }$

We will also require the following weak condition on $f$

Definition B.1 (Confining Function). A function ϕ is said to be confining when li $\mathrm { n } _ { \| \boldsymbol { x } \|  \infty } \phi ( \boldsymbol { x } ) = \infty$ and for all $s > 0$

$$
\int _ { \mathbb { R } ^ { d } } e ^ { - \frac { 2 \phi ( x ) } { s } } \mathrm { d } x < + \infty .
$$

It is known that for confining functions that are Morse, the spectral gap of Langevin dynamics can be estimated as follows.

Theorem B.2 (Theorem 2 [50]). Suppose f is a confiningfunction (see Definition B.1),

$$
\forall \beta > 0 , \operatorname* { l i m } _ { \| x \|  \infty } ( \beta \| \nabla f ( x ) \| ^ { 2 } - \Delta f ( x ) ) = \infty ,
$$

f has at least two local minima, and f is a Morsefunction. Then there exists a $\beta _ { 0 } ,$ depending on $f ,$ such that $\forall \beta > \beta _ { 0 }$ , the reciprocal ofthe relaxation time $\delta ^ { ( C ) } ( \beta )$ of Langevin dynamics with potential f satisfies

$$
\delta ^ { ( C ) } ( \beta ) = ( \alpha + o ( 1 ) ) e ^ { - \beta H _ { f } } ,
$$

where $H _ { f }$ is a quantity known as the Morse saddle barrier that depends only on the geometry of the function ${ \bf { \bar { \rho } } } _ { f , \quad }$ , and α $> 0$ is a constant depending on $f .$

We start with the following result bounding the continuous-time convergence of Langevin dynamics in Rényi divergence under WPI.

We use the generator-normalized WPI convention (4) from Appendix A. When X is a bounded convex set, as in Theorem B.5, the diffusion (18) is reflected at the boundary of X.

We state Theorem B.3 and the convergence bounds that follow it in terms of the q-th Rényi divergence of a probability measure ν with respect to a probability measure π, defined as $\mathcal { R } _ { q } ( \nu | | \pi ) =$ $\begin{array} { r } { \frac { 1 } { q - 1 } \log \left[ \int \left( \frac { d \nu } { d \pi } \right) ^ { q } d \pi \right] } \end{array}$

Lemma B.3. Let $f ( x ) : \mathcal { X }  \mathbb { R }$ . Consider running Langevin dynamics generated by f for time T producing distribution $\mu _ { T }$ . Let π be the stationary distribution of the Langevin dynamics. Suppose that $\begin{array} { r } { K : = \operatorname* { s u p } _ { x \in \mathcal { X } } \frac { d \mu _ { 0 } } { d \pi } ( x ) < } \end{array}$ ∞ and let $q \geqslant 2 .$ . If π satisfies a $( \delta , \epsilon )$ -WPI with $\varepsilon K ^ { q } \leqslant \delta / 4$ , then $u ( t ) : = \mathcal { R } _ { q } ( \mu _ { t } | | \bar { \boldsymbol { \pi } } )$ satisfies

$$
u ( T ) \leqslant \operatorname* { m a x } \left\{ u ( 0 ) - \frac { \delta T } { q } , 1 \right\} ,\tag{19}
$$

and, for $\begin{array} { r } { T \geqslant T _ { 0 } : = q ( u ( 0 ) - 1 ) _ { + } / \delta , } \end{array}$

$$
u ( T ) \leqslant \exp \left( - \frac { 2 \delta ( T - T _ { 0 } ) } { q } \right) + \frac { 2 \epsilon K ^ { q } } { \delta } .\tag{20}
$$

Proof. The following is a modified version of the proof of Theorem 2.2.25 in Chewi [12], originally from [55], to account for a weak Poincaré inequality.

Let $\textstyle \rho _ { 0 } : = { \frac { d \mu _ { 0 } } { d \pi } }$ and $P _ { t }$ be the transition operator of Langevin dynamics. Then from reversibility of Langevin dynamics $\rho _ { t } = P _ { t } \rho _ { 0 }$ . Taking the derivative gives

$$
\frac { d } { d t } \mathcal { R } _ { q } ( \mu _ { t } | | \pi ) = - \frac { 4 } { q } \frac { \mathcal { E } ( \rho _ { t } ^ { q / 2 } , \rho _ { t } ^ { q / 2 } ) } { \int \rho _ { t } ^ { q } d \pi } .\tag{21}
$$

The $( \delta , \epsilon )$ -weak Poincaré inequality implies that

$$
\begin{array} { r l } & { \mathcal { E } _ { \boldsymbol \pi } ( \rho _ { t } ^ { q / 2 } , \rho _ { t } ^ { q / 2 } ) \geqslant \delta \mathrm { V a r } _ { \boldsymbol \pi } \left( \rho _ { t } ^ { q / 2 } \right) - \mathrm { E r r o r } _ { t } } \\ & { \qquad \mathrm { E r r o r } _ { t } = \epsilon \| \rho _ { t } \| _ { \infty } ^ { q } . } \end{array}
$$

Note that

$$
\| \rho _ { t } \| _ { \infty } \leqslant K ,
$$

where the inequality holds because $\rho _ { t } = P _ { t } \rho _ { 0 }$ and the Markov operator $P _ { t }$ is a contraction in the supremum norm. Hence Error<sub>t</sub> $\leqslant \epsilon K ^ { q }$

Combining these bounds gives

$$
\begin{array} { r l } & { \delta ^ { - 1 } \mathcal { E } _ { \pi } ( \rho _ { t } ^ { q / 2 } , \rho _ { t } ^ { q / 2 } ) \geqslant \mathrm { V a r } _ { \pi } ( \rho _ { t } ^ { q / 2 } ) - \delta ^ { - 1 } \mathrm { E r r o r } _ { t } } \\ & { \qquad = \displaystyle \int ( \rho _ { t } ) ^ { q } d \pi - ( \int ( \rho _ { t } ) ^ { q / 2 } d \pi ) ^ { 2 } - \delta ^ { - 1 } \mathrm { E r r o r } _ { t } } \\ & { \qquad = ( \int ( \rho _ { t } ) ^ { q } d \pi ) ( 1 - \frac { \exp ( ( q - 2 ) \mathcal { R } _ { q / 2 } ( \rho _ { t } \pi | | \pi ) ) } { \exp ( ( q - 1 ) \mathcal { R } _ { q } ( \rho _ { t } \pi | | \pi ) ) } ) - \delta ^ { - 1 } \mathrm { E r r o r } _ { t } } \\ & { \qquad \geqslant ( \displaystyle \int \rho _ { t } ^ { q } d \pi ) ( 1 - \exp ( - \mathcal { R } _ { q } ( \rho _ { t } \pi | | \pi ) ) ) - \delta ^ { - 1 } \mathrm { E r r o r } _ { t } , } \end{array}
$$

where the last inequality follows from the monotonicity of Rényi divergence in $q .$ Plugging this into the derivative of Rényi divergence in (21), we get

$$
\begin{array} { l } { \displaystyle - \frac { d } { d t } \mathcal { R } _ { q } ( \mu _ { t } | | \pi ) \geqslant \frac { \delta 4 } { q } ( 1 - \exp ( - \mathcal { R } _ { q } ( \rho _ { t } \pi | | \pi ) ) ) - \frac { 4 } { q \int \rho _ { t } ^ { q } d \pi } \mathrm { E r r o r } _ { t } } \\ { \displaystyle = \frac { \delta 4 } { q } ( 1 - \exp ( - \mathcal { R } _ { q } ( \mu _ { t } | | \pi ) ) ) - \frac { 4 } { q } \mathrm { E r r o r } _ { t } \exp ( - ( q - 1 ) \mathcal { R } _ { q } ( \mu _ { t } | | \pi ) ) . } \end{array}
$$

Since $\begin{array} { r } { F ( t ) : = \int \rho _ { t } ^ { q } d \pi = \exp ( ( q - 1 ) u ( t ) ) \geqslant 1 } \end{array}$ and Err $\mathbf { \Phi } ) \mathbf { r } _ { t } \leqslant \epsilon K ^ { q }$ , it follows that

$$
u ^ { \prime } ( t ) \leqslant - \frac { 4 \delta } { q } \left( 1 - \exp ( - u ( t ) ) \right) + \frac { 4 \epsilon K ^ { q } } { q } .
$$

If $u ( t ) \geqslant 1$ , then $1 - \exp ( - u ( t ) ) \geqslant 1 / 2$ , and the condition $\epsilon K ^ { q } \leqslant \delta / 4$ gives $u ^ { \prime } ( t ) \leqslant - \delta / q$ . Hence u decreases at rate at least $\delta / q$ while u $\geqslant 1$ , and once u $\leqslant$ 1 it stays at most 1. This proves (19) and shows that $u ( t ) \leqslant 1$ for all $\dot { t } \geqslant T _ { 0 } . \mathrm { I f } \ 0 \leqslant u ( t ) \leqslant 1$ , then $1 - \mathrm { e x p ( - } u ( t ) ) \geqslant u ( t ) / 2$ , so

$$
u ^ { \prime } ( t ) \leqslant - \frac { 2 \delta } { q } u ( t ) + \frac { 4 \epsilon K ^ { q } } { q } .
$$

Applying Grönwall’s inequality on $[ T _ { 0 } , T ]$ with $u ( T _ { 0 } ) \leqslant 1$ gives (20).

Theorem B.3 requires a bounded initial density ratio $K .$ , while the proof of Theorem B.5 controls only a Rényi divergence $\mathcal { R } _ { q }$ of finite order q, which gives no bound on the density ratio. The following elementary lemma restricts a measure to a set of large mass on which its density ratio is bounded, and bounds the Rényi divergence of the normalized restriction.

Lemma B.4. Let ν and π be probability measures on X, let $s > 1 , U \geqslant 0$ and $\varrho \in ( 0 , 1 / 2 ]$ , and suppose that $\mathcal { R } _ { s } ( \nu | | \pi ) \leqslant U$ . Let $S : = \{ x \in \mathcal { X } : \frac { d \nu } { d \pi } ( x ) \leqslant \varrho ^ { - 1 / ( s - 1 ) } e ^ { U } \}$ and $\tilde { \nu } : = \nu ( \cdot \cap S ) / \nu ( S )$ Then

$$
\begin{array} { r } { \nu ( S ^ { c } ) \leqslant \varrho , \displaystyle \operatorname* { s u p } _ { x \in \mathcal { X } } \frac { d \tilde { \nu } } { d \pi } ( x ) \leqslant \frac { \varrho ^ { - 1 / ( s - 1 ) } e ^ { U } } { 1 - \varrho } , } \\ { \mathcal { R } _ { s } ( \tilde { \nu } | | \pi ) \leqslant U + \frac { s } { s - 1 } \log \frac { 1 } { 1 - \varrho } . } \end{array}
$$

Proof. Write $g : = { \begin{array} { l } { d \nu } \\ { d \pi } \end{array} }$ . By the definition of $\mathcal { R } _ { s }$ , the expectation of $g ^ { s - 1 }$ under ν is $\textstyle \int g ^ { s } d \pi =$ $e ^ { ( s - 1 ) \mathcal { R } _ { s } ( \nu | | \pi ) } \leqslant e ^ { ( s - 1 ) U }$ . Markov’s inequality under ν applied to $g ^ { s - 1 }$ therefore gives $\nu ( S ^ { c } ) =$ $\nu ( g ^ { s - 1 } > \varrho ^ { - 1 } e ^ { ( s - 1 ) U } ) \leqslant \varrho .$ , and hence $\nu ( S ) \geqslant 1 - \varrho .$

By the normalization of ν˜ and this lower bound, its density with respect to π is $\begin{array} { r } { \frac { d \tilde { \nu } } { d \pi } = g \mathbf { 1 } _ { S } / \nu ( S ) \leqslant } \end{array}$ $\varrho ^ { - 1 / ( s - 1 ) } e ^ { U } / ( 1 - \varrho )$ , and $\begin{array} { r } { \int \left( \frac { d \tilde { \nu } } { d \pi } \right) ^ { s } d \pi \leqslant \nu ( S ) ^ { - s } \int g ^ { s } d \pi \leqslant ( 1 - \varrho ) ^ { - s } e ^ { ( s - 1 ) U } } \end{array}$ . Taking the logarithm and dividing by $s - 1$ gives the bound on $\mathcal { R } _ { s } ( { \tilde { \nu } } | | \pi )$ □

## B.1 Convergence of Annealed Langevin Diffusion

We use Theorems B.3 and B.4 to bound the continuous-time convergence of annealed Langevin dynamics.

Theorem B.5 (Convergence of annealed Langevin diffusion). Suppose the sequence of learning rates $\{ \eta _ { r } \} _ { r = 1 } ^ { M }$ is such that the corresponding Langevin dynamics starting from $\mu _ { 0 }$ with constant rate η<sub>r</sub> has stationary distribution $\pi ^ { ( r ) } \propto \exp ( - 2 f / \eta _ { r } )$ and satisfies $a \left( \delta _ { r } , \epsilon _ { r } \right)$ WPI. Let $q \geqslant 2$ and $p \in ( 0 , 1 / 4 ] ,$ , suppose that $\bar { R } : = \operatorname* { m a x } \{ 1 , \mathcal { R } _ { q } ( \mu _ { 0 } | | \pi ^ { ( 1 ) } ) \} < \infty ,$ , and let $\begin{array} { r } { \tau _ { r } \ = \ \frac { 2 q } { \delta _ { r } } } \end{array}$ . Then if $\Delta _ { r } \sec ( f ) \leqslant 1$ and $\begin{array} { r } { \epsilon _ { r } \leqslant \frac { p ^ { 2 } \delta _ { r } } { 4 M ^ { 2 } } e ^ { - q ( \bar { R } + 2 ) } } \end{array}$ for each $r = 2 , \ldots , M$ , where $\Delta _ { r } : = 2 / \eta _ { r } - 2 / \eta _ { r - 1 }$ and os $ ( f ) : = \operatorname* { s u p } _ { x \in \mathcal { X } } f ( x ) - \operatorname* { i n f } _ { x \in \mathcal { X } } f ( x )$ , then

$$
\mu _ { T } = \alpha \nu + ( 1 - \alpha ) \bar { \nu } , \qquad \alpha \geqslant 1 - p , \qquad \mathcal { R } _ { q } ( \nu | | \pi ^ { ( M ) } ) \leqslant \bar { R } ,
$$

for some $\alpha \in [ 0 , 1 ]$ and probability measures ν and ν¯ on $x ,$ where $\begin{array} { r } { T = \sum _ { r = 2 } ^ { M } \tau _ { r } } \end{array}$ and $\mu _ { T }$ is the output ofLangevin dynamics after runningfor time T.

Proof. The proof of this result is inspired by the proof of Theorem 4.13 [25].

Let $\pi ^ { ( r ) } \propto \exp ( - 2 f / \eta _ { r } )$ , corresponding to the stationary distribution for the r-th chain. As mentioned earlier $\eta ( t )$ is such that for all $t \in [ T _ { r - 1 } , T _ { r } ) , \dot { \eta } ( t ) = \eta _ { r }$ . Also let $T _ { 1 } = 0$ and $T _ { r } =$ $T _ { r - 1 } + \tau _ { r }$ for $r = 2 , \ldots , M ,$ , so that $T _ { M } = T$ . The distribution $\mu _ { t }$ corresponds to the output of the annealed Langevin chain at time $t \in [ 0 , T ]$ . Recall that $\Delta _ { r } = 2 / \eta _ { r } - 2 / \eta _ { r - 1 }$

We start by bounding the following quantities for consecutive Gibbs measures. Note that for two consecutive Gibbs distributions,

$$
\operatorname* { s u p } _ { x \in \mathcal { X } } \frac { d \pi ^ { ( r - 1 ) } } { d \pi ^ { ( r ) } } ( x ) = \operatorname* { s u p } _ { x \in \mathcal { X } } \frac { e ^ { \Delta _ { r } f ( x ) } } { \mathbb { E } _ { \pi ^ { ( r ) } } [ e ^ { \Delta _ { r } f } ] } \leqslant e ^ { \Delta _ { r } \operatorname { o s c } ( f ) } .
$$

Hence $D _ { r } : = \mathcal { R } _ { \infty } \big ( \pi ^ { ( r - 1 ) } | | \pi ^ { ( r ) } \big ) \leqslant \Delta _ { r } \operatorname { o s c } ( f ) \leqslant 1$ , where $\begin{array} { r } { \mathcal { R } _ { \infty } ( \pi _ { 1 } | | \pi _ { 2 } ) : = \log \operatorname* { s u p } _ { x \in \mathcal { X } } \frac { d \pi _ { 1 } } { d \pi _ { 2 } } ( x ) } \end{array}$ We also have the following inequalities for every probability measure $\mu$ on $\mathcal { X }$ and every $s > 1 { : }$

$$
\begin{array} { r l r } & { } & { \mathcal { R } _ { s } ( \mu | | \pi ^ { ( r ) } ) \leqslant \mathcal { R } _ { s } ( \mu | | \pi ^ { ( r - 1 ) } ) + D _ { r } , } \\ & { } & { \displaystyle \operatorname* { s u p } _ { x \in \mathcal { X } } \frac { d \mu } { d \pi ^ { ( r ) } } ( x ) \leqslant e ^ { D _ { r } } \displaystyle \operatorname* { s u p } _ { x \in \mathcal { X } } \frac { d \mu } { d \pi ^ { ( r - 1 ) } } ( x ) . } \end{array}\tag{22}
$$

We now show by induction on $r = 1 , \ldots , M$ that each chain is acting as a warm start for the next, in the sense that the law at time $T _ { r }$ decomposes as

$$
\begin{array} { r l r } { \mu _ { T _ { r } } = \alpha _ { r } \nu _ { r } + ( 1 - \alpha _ { r } ) \bar { \nu } _ { r } , } & { } & { \alpha _ { r } \geqslant 1 - ( r - 1 ) \rho , } \\ & { } & { \mathcal { R } _ { q } ( \nu _ { r } | | \pi ^ { ( r ) } ) \leqslant \bar { R } , } \end{array}\tag{23}
$$

for some $\alpha _ { r } \in [ 0 , 1 ]$ and probability measures $\nu _ { r }$ and $\bar { \nu } _ { r }$ on $x ,$ , where $\rho : = p / M$ . For $r = 1$ , this holds with $\nu _ { 1 } = \mu _ { 0 }$ and $\alpha _ { 1 } = 1$ by the definition of $\bar { R }$

Now let $2 \leqslant r \leqslant M$ and suppose that (23) holds for $r - 1$ . Applying Theorem B.4 to $\nu _ { r - 1 }$ and $\pi ^ { ( r - 1 ) }$ with $s ~ = ~ q , U ~ = ~ { \bar { R } }$ and $\varrho ~ = ~ \rho \leqslant 1 / 4$ gives a set $S$ with $\nu _ { r - 1 } ( S ^ { c } ) ~ \leqslant ~ \rho$ such that $\tilde { \nu } : = \nu _ { r - 1 } ( \cdot \cap S ) / \nu _ { r - 1 } ( S )$ satisfies $\begin{array} { r } { \operatorname* { s u p } _ { x \in \mathcal { X } } \frac { d \tilde { \nu } } { d \pi ^ { ( r - 1 ) } } ( x ) \leqslant K : = \rho ^ { - 1 / ( q - 1 ) } e ^ { \bar { R } } / ( 1 - \rho ) } \end{array}$ and $\mathcal { R } _ { q } ( \tilde { \nu } | | \pi ^ { ( r - 1 ) } ) \leqslant \bar { R } + 1$ , where the last bound uses $q / ( q - 1 ) \leqslant 2$ and $\rho \leqslant 1 / 4$

Changing the reference measure to the stationary distribution $\pi ^ { ( r ) }$ of the next chain, using (22) and $\begin{array} { r } { D _ { r } \leqslant 1 , \mathrm { g i v e s ~ s u p } _ { x \in \mathcal { X } } \frac { d \tilde { \nu } } { d \pi ^ { ( r ) } } ( x ) \leqslant e K } \end{array}$ and $\mathcal { R } _ { q } ( \tilde { \nu } | | \pi ^ { ( r ) } ) \leqslant \bar { R } + 2$

We apply Theorem B.3 to the Langevin dynamics with constant learning rate $\eta _ { r }$ started from ${ \tilde { \nu } } .$ Its hypothesis holds, because the density ratio of $\tilde { \nu }$ with respect to $\pi ^ { ( r ) }$ is at most $e K$ , and $\rho ^ { - q / ( q - 1 ) } \leqslant \rho ^ { - 2 } , \rho \leqslant 1 / 4$ and the assumption on $\epsilon _ { r }$ give $\begin{array} { r } { \epsilon _ { r } ( e K ) ^ { q } \leqslant \frac { \delta _ { r } } { 4 } \left( \frac { 4 } { 3 e } \right) ^ { q } \leqslant \frac { \delta _ { r } } { 4 } } \end{array}$ . Write $\mu P _ { t } ^ { ( r ) }$ for the law at time t of this dynamics started from a probability measure $\mu ,$ and let $\nu _ { r } : = \tilde { \nu } P _ { \tau _ { r } } ^ { ( r ) }$ Since $\mathcal { R } _ { q } ( \tilde { \nu } | | \pi ^ { ( r ) } ) \leqslant \bar { R } + 2 , \delta _ { r } \tau _ { r } / q = 2$ and $\bar { R } \geqslant 1 , ( 1 9 )$ gives

$$
\mathcal { R } _ { q } ( \nu _ { r } | | \pi ^ { ( r ) } ) \leqslant \operatorname* { m a x } \left\{ \bar { R } + 2 - \frac { \delta _ { r } \tau _ { r } } { q } , 1 \right\} = \bar { R } .
$$

The law $\mu P _ { \tau _ { r } } ^ { ( r ) }$ depends linearly on $\mu , \mathrm { { s o } } \mu _ { T _ { r } } = \mu _ { T _ { r - 1 } } P _ { \tau _ { r } } ^ { ( r ) }$ is the sum of $\alpha _ { r - 1 } \nu _ { r - 1 } ( S ) \nu _ { r }$ and the image under ${ P } _ { \tau _ { r } } ^ { ( r ) }$ of the remaining mass $\alpha _ { r - 1 } \nu _ { r - 1 } ( \cdot \cap S ^ { c } ) + ( 1 - \alpha _ { r - 1 } ) \bar { \nu } _ { r - 1 }$ . Hence $( 2 3 )$ holds for r with $\alpha _ { r } : = \alpha _ { r - 1 } \nu _ { r - 1 } ( S )$ , which is at least $1 - ( r - 1 ) \rho$ because $\alpha _ { r - 1 } \geqslant 1 - ( r - 2 ) \rho$ and $\nu _ { r - 1 } ( S ) \geqslant 1 - \rho ,$ and with $\bar { \nu } _ { r }$ the normalized image of the remaining mass.

With the induction complete, taking $r = M$ and using $T _ { M } = T$ gives the claim with $\nu = \nu _ { M }$ $\bar { \nu } = \bar { \nu } _ { M } \mathrm { a n d } \alpha = \alpha _ { M } \geqslant 1 - ( M - 1 ) \rho \geqslant 1 - p .$

## B.2 Discretization of Langevin Diffusion

To discretize the annealed Langevin dynamics in time, consider a sequence of step sizes $h _ { 1 } , \ldots , h _ { M }$ one for each phase of the piecewise-constant learning rate $\eta ( \theta _ { t } )$ . We consider a schedule of step sizes $h ( t )$ such that $\forall t \in [ T _ { k - 1 } , T _ { k } ] , h ( t ) = h _ { k }$ . Then, the first-order discretization of (17), evaluated at the discrete times $\begin{array} { r } { t _ { n } = \sum _ { m = 0 } ^ { n - 1 } h ( t _ { m } ) } \end{array}$ , takes the form

$$
X _ { t _ { n + 1 } } = X _ { t _ { n } } - h ( t _ { n } ) \nabla f ( X _ { t _ { n } } ) + \sqrt { h ( t _ { n } ) \eta ( \theta _ { t _ { n } } ) } Z _ { n } , \qquad Z _ { n } \sim \mathcal { N } ( 0 , I _ { d } ) ,\tag{24}
$$

which is referred to as the unadjusted Langevin algorithm (ULA). Since the total time is $T =$ $\sum _ { k = 1 } ^ { M } \tau _ { k }$ , for $N _ { k } = \tau _ { k } / h _ { k }$ , the total number of steps of the discrete process is $\begin{array} { r } { N = \sum _ { k = 1 } ^ { M } N _ { k } } \end{array}$ . If we set $h _ { k } = \eta _ { k }$ , then ULA corresponds to annealed SGD (Definition 1.3).

We write $\hat { \mu } _ { t }$ for the law of the discretized process at time t. To bound the discretization error, we compare the law $\mathrm { W } _ { T }$ of the path, up to time $T _ { \ast }$ , of the continuous SDE

$$
d X _ { t } = - \nabla f ( X _ { t } ) d t + \sqrt { \eta ( \theta _ { t } ) } d B _ { t } ,
$$

and the law $P _ { T }$ of the path of the SDE that corresponds to the discretized dynamics (ULA)

$$
d X _ { t } = - \nabla f ( X _ { t ^ { - } } ) d t + \sqrt { \eta ( \theta _ { t } ) } d \tilde { B } _ { t } ,
$$

where $t ^ { - } = t _ { n }$ for $t \in [ t _ { n } , t _ { n + 1 } )$ . Note that $\theta$ does not need to be evaluated at $t ^ { - }$ since the learning rate schedule is piecewise constant.

To keep things general, for the result below, we will let $h _ { k }$ and $\eta _ { k }$ be independent. Later on we will discuss when we can set $h _ { k } = \eta _ { k }$ . Again, $\pi ^ { ( k ) } \propto \exp ( - 2 f / \eta _ { k } )$

Theorem B.6 (Annealed Unadjusted Langevin Rényi discretization bound). Let $f :  { \mathbb { R } ^ { d } } \to  { \mathbb { R } }$ be such that $\exp ( - 2 f / \eta _ { k } )$ is integrable for $k = 1 , \dots , M$ . Assume that ∇f is L-Lipschitz. Let $\hat { \mu } _ { T }$ be the law ofthe discretized process at time $T$ and $\mu _ { T }$ the law ofthe corresponding Langevin diffusion at time $\dot { T }$ , where both processes startfrom the same distribution $\mu _ { 0 } .$ . Let $\begin{array} { r } { T _ { m } = \sum _ { r = 1 } ^ { m ^ { \smile } } \tau _ { r } } \end{array}$ , and assume that each $N _ { k } = \tau _ { k } / h _ { k }$ is an integer. There are universal constants $B , C > 0$ such that for all $q \geqslant 2 ,$ $i f h _ { k } \leqslant$ min((CT q<sup>2</sup>L<sup>2</sup>)<sup>−1</sup>, L<sup>−1</sup>) for k = 1, . . . , M, then

$$
\begin{array} { r l } & { \mathcal { R } _ { q } ( \hat { \mu } _ { T } | | \mu _ { T } ) \leqslant \displaystyle \frac { 1 } { q - 1 } \operatorname* { m a x } _ { k = 1 , \ldots , M } \Biggl [ B T q ^ { 2 } L ^ { 3 } h _ { k } ^ { 2 } \Bigl ( d + \mathcal { R } _ { q } ( \mu _ { T _ { k - 1 } } | | \pi ^ { ( k ) } ) + 1 \Bigr ) } \\ & { \qquad + B T q ^ { 2 } L ^ { 2 } h _ { k } d \Biggr ] } \end{array}\tag{25}
$$

Proof. We follow the method given by [12]. We start with Girsanov’s theorem to control the Rényi divergence in a single step.

$$
\begin{array} { r l } & { \exp \bigl ( ( q - 1 ) R _ { q } ( \mathrm { P } _ { T } \| \mathrm { W } _ { T } ) \bigr ) = \mathbb { E } _ { \mathrm { W } _ { T } } \biggl [ \biggl ( \frac { d \mathrm { P } _ { T } } { d \mathrm { W } _ { T } } \biggr ) ^ { q } \biggr ] } \\ & { = \mathbb { E } _ { \mathrm { W } _ { T } } \exp \left( q \int _ { 0 } ^ { T } \frac { \langle \nabla f ( X _ { t } ) - \nabla f ( X _ { t - } ) , d B _ { t } \rangle } { \sqrt { \eta ( \theta _ { t } ) } } - \frac { q } { 2 } \int _ { 0 } ^ { T } \frac { \| \nabla f ( X _ { t } ) - \nabla f ( X _ { t - } ) \| ^ { 2 } } { \eta ( \theta _ { t } ) } d t \right) . } \end{array}
$$

Following the standard argument, we apply Cauchy–Schwarz and the supermartingale property of the exponential martingale. For any $\lambda > 0 .$

$$
\begin{array} { r l } & { \left( \mathbb { E } _ { \mathbf { W } _ { T } } \left[ \left( \frac { d \mathbf { P } _ { T } } { d \mathbf { W } _ { T } } \right) ^ { q } \right] \right) ^ { 2 } } \\ & { \leqslant \mathbb { E } _ { \mathbf { W } _ { T } } \exp \left( 2 q \int _ { 0 } ^ { T } \frac { \langle \nabla f ( X _ { t } ) - \nabla f ( X _ { t ^ { - } } ) , d B _ { t } \rangle } { \sqrt { \eta ( \theta _ { t } ) } } - \lambda \int _ { 0 } ^ { T } \frac { \| \nabla f ( X _ { t } ) - \nabla f ( X _ { t ^ { - } } ) \| ^ { 2 } } { \eta ( \theta _ { t } ) } \mathrm { d } t \right) } \\ & { \quad \times \mathbb { E } _ { \mathbf { W } _ { T } } \exp \left( \left( \lambda - q \right) \int _ { 0 } ^ { T } \frac { \| \nabla f ( X _ { t } ) - \nabla f ( X _ { t ^ { - } } ) \| ^ { 2 } } { \eta ( \theta _ { t } ) } \mathrm { d } t \right) . } \end{array}
$$

Setting $\lambda = 2 q ^ { 2 }$ makes the first factor on the right-hand side of the Cauchy–Schwarz inequality equal to the expectation of a non-negative stochastic exponential, and since this stochastic exponential is a supermartingale, its expectation is at most 1. Noting that the left-hand side is at least 1 (by Jensen), we may drop the square and, since $\lambda - q \leqslant 2 q ^ { 2 }$ , obtain

$$
\begin{array} { r l r } {  { \mathbb { E } _ { \mathrm { W } _ { T } } \bigg [ \bigg ( \frac { d \mathrm { P } _ { T } } { d \mathrm { W } _ { T } } \bigg ) ^ { q } \bigg ] \leqslant \mathbb { E } _ { \mathrm { W } _ { T } } \exp \Bigg ( 2 q ^ { 2 } \int _ { 0 } ^ { T } \frac { \| \nabla f ( X _ { t } ) - \nabla f ( X _ { t ^ { - } } ) \| ^ { 2 } } { \eta ( \theta _ { t } ) } d t \Bigg ) } } \\ & { } & { \leqslant \mathbb { E } _ { \mathrm { W } _ { T } } \exp \Bigg ( 2 q ^ { 2 } L ^ { 2 } \int _ { 0 } ^ { T } \frac { \| X _ { t } - X _ { t ^ { - } } \| ^ { 2 } } { \eta ( \theta _ { t } ) } d t \Bigg ) } \\ & { } & { \leqslant \frac { 1 } { T } \displaystyle \sum _ { n = 0 } ^ { N - 1 } \int _ { t _ { n } } ^ { t _ { n + 1 } } \mathbb { E } _ { \mathrm { W } _ { T } } \exp \bigg ( 2 q ^ { 2 } L ^ { 2 } T \frac { \| X _ { t } - X _ { t ^ { - } } \| ^ { 2 } } { \eta ( \theta _ { t } - ) } \bigg ) d t , } \end{array}
$$

where the last step uses Jensen’s inequality.

It remains to control $\| X _ { t } - X _ { t _ { n } } \|$ on a single discretization interval. Fix the n-th interva $[ t _ { n } , t _ { n + 1 } )$ with $t _ { n + 1 } - t _ { n } = h _ { n }$ , and write $x = X _ { t _ { n } }$ . Under $\mathrm { W } _ { T } .$ , the process satisfies

$$
X _ { t } - x = - \int _ { t _ { n } } ^ { t } \nabla f ( X _ { s } ) \mathrm { d } s + \sqrt { \eta ( \theta _ { t _ { n } } ) } ( B _ { t } - B _ { t _ { n } } ) .
$$

By the triangle inequality and the L-Lipschitz property of $\nabla f .$

$$
\| X _ { t } - x \| \leqslant \int _ { t _ { n } } ^ { t } \bigl ( \| \nabla f ( x ) \| + L \| X _ { s } - x \| \bigr ) \mathrm { d } s + \sqrt { \eta ( \theta _ { t _ { n } } ) } \| B _ { t } - B _ { t _ { n } } \| .
$$

Grönwall’s inequality gives, for all $t \in [ t _ { n } , t _ { n + 1 } ]$

$$
\| X _ { t } - x \| \leqslant e ^ { L h ( t _ { n } ) } \Bigg ( h ( t _ { n } ) \| \nabla f ( x ) \| + \sqrt { \eta ( \theta _ { t _ { n } } ) } \operatorname* { s u p } _ { s \in [ t _ { n } , t _ { n + 1 } ] } \| B _ { s } - B _ { t _ { n } } \| \Bigg ) .
$$

We now set $\lambda : = 3 2 q ^ { 2 } T L ^ { 2 }$ and $\begin{array} { r } { S _ { n } : = \operatorname* { s u p } _ { s \in [ t _ { n } , t _ { n + 1 } ] } \| B _ { s } - B _ { t _ { n } } \| } \end{array}$ . Thus, since $L h ( t _ { n } ) \leqslant 1$

$$
\frac { 2 T q ^ { 2 } L ^ { 2 } } { \eta ( \theta _ { t _ { n } } ) } \| X _ { t } - X _ { t _ { n } } \| ^ { 2 } \leqslant \lambda \frac { h ( t _ { n } ) ^ { 2 } } { \eta ( \theta _ { t _ { n } } ) } \| \nabla f ( X _ { t _ { n } } ) \| ^ { 2 } + \lambda S _ { n } ^ { 2 } .\tag{26}
$$

By the reflection principle, a standard one-dimensional Brownian motion w satisfies, for $\lambda h ( t _ { n } ) \leqslant$ ${ \scriptstyle \frac { 1 } { 4 } } , \mathbb { E } \exp ( \lambda \operatorname* { s u p } _ { s \in [ 0 , h ( t _ { n } ) ] } w _ { s } ^ { 2 } ) \leqslant 2 ( 1 - 2 \lambda h ( t _ { n } ) ) ^ { - 1 / 2 } - 1 \leqslant \exp ( 4 \lambda h ( t _ { n } ) )$ . Applying the onedimensional bound to each of the d independent coordinates of $B _ { s } - B _ { t _ { n } }$ , and bounding the supremum of a sum by the sum of the suprema, gives

$$
\log \mathbb { E } \exp \bigl ( \lambda S _ { n } ^ { 2 } \bigr ) \leqslant 4 \lambda d h ( t _ { n } ) .\tag{27}
$$

Since $S _ { n }$ is independent of $X _ { t _ { n } }$ under $\mathrm { W } _ { T }$ , and since $\lambda h ( t _ { n } ) \leqslant 3 2 / C \leqslant \frac { 1 } { 4 }$ for $C \geqslant 1 2 8$ by $h ( t _ { n } ) \leqslant ( C T q ^ { 2 } L ^ { 2 } ) ^ { - 1 }$ , combining (26) and (27) gives

$$
\begin{array} { r l } & { \mathbb { E } _ { \mathbf { W } _ { T } } \left[ \mathbb { E } _ { \mathbf { W } _ { T } } \left[ \exp \left( \frac { 2 T q ^ { 2 } L ^ { 2 } } { \eta ( \theta _ { t _ { n } } ) } \| X _ { t } - X _ { t _ { n } } \| ^ { 2 } \right) \mid X _ { t _ { n } } = x \right] \right] } \\ & { \leqslant \mathbb { E } _ { \mathbf { W } _ { T } } \left[ \exp \left( \lambda \frac { h ( t _ { n } ) ^ { 2 } } { \eta ( \theta _ { t _ { n } } ) } \| \nabla f ( X _ { t _ { n } } ) \| ^ { 2 } \right) \right] \cdot \mathbb { E } _ { \mathbf { W } _ { T } } \left[ \exp \left( \lambda S _ { n } ^ { 2 } \right) \right] } \\ & { \leqslant \mathbb { E } _ { \mathbf { W } _ { T } } \left[ \exp \left( \lambda \frac { h ( t _ { n } ) ^ { 2 } } { \eta ( \theta _ { t _ { n } } ) } \| \nabla f ( X _ { t _ { n } } ) \| ^ { 2 } \right) \right] \cdot \exp \left( 4 \lambda d h ( t _ { n } ) \right) . } \end{array}
$$

So, with $Y _ { n } : = \exp ( \lambda h ( t _ { n } ) ^ { 2 } \| \nabla f ( X _ { t _ { n } } ) \| ^ { 2 } / \eta ( \theta _ { t _ { n } } ) )$ ,

$$
\begin{array} { r l } { \mathcal { R } _ { q } ( \mathrm { P } _ { T } \| \mathrm { W } _ { T } ) \leqslant \displaystyle \frac { 1 } { q - 1 } \operatorname* { m a x } _ { n = 0 , \ldots , N - 1 } \Bigg [ \log \mathbb { E } _ { \mathrm { W } _ { T } } \left[ Y _ { n } \right] } & { } \\ { \quad \quad \quad + \log \mathbb { E } _ { \mathrm { W } _ { T } } \left[ \exp \left( \lambda S _ { n } ^ { 2 } \right) \right] \Bigg ] } \end{array}\tag{28}
$$

We now proceed to bound the first term inside the maximum in (28). Let $\pi _ { t _ { n } }$ denote the stationary distribution for the n-th segment, i.e. $t \in [ t _ { n } , t _ { n + 1 } )$ , and let k be the phase that contains $[ t _ { n } , t _ { n + 1 } )$ so that $\pi _ { t _ { n } } = \pi ^ { ( k ) }$ and $\eta ( \theta _ { t _ { n } } ) = \eta _ { k }$ . From [12, Lemma 6.2.7], applied to the potential $2 f / \eta _ { k }$ , for any u $\geqslant 0 ;$

$$
\pi ^ { ( k ) } \{ \| \nabla f ( X ) \| \geqslant C ^ { \prime } { \sqrt { \eta _ { k } L } } ( { \sqrt { d } } + u ) \} \leqslant \exp ( - u ^ { 2 } ) .
$$

Then by Hölder’s inequality (see [12, Lemma 6.2.8]), with $R _ { n } : = \mathcal { R } _ { q } ( \mu _ { t _ { n } } | | \pi ^ { ( k ) } )$ , every measurable set $A \subseteq \mathbb { R } ^ { d }$ satisfies

$$
\mu _ { t _ { n } } ( A ) \leqslant \exp \left( \left( 1 - { \frac { 1 } { q } } \right) R _ { n } \right) \pi ^ { ( k ) } ( A ) ^ { 1 - 1 / q } .
$$

Thus, for any $0 < \delta < \frac { 1 } { 2 }$ , taking A to be the set where $\| \nabla f \| \geqslant C ^ { \prime } \sqrt { \eta _ { k } L } ( \sqrt { d } + u )$ with $u ^ { 2 } = { }$ $R _ { n } + 2 \log ( \bar { 1 ^ { \prime } } \delta )$ shows that with at least $1 - \delta$ probability under $\mu _ { t _ { n } }$

$$
\begin{array} { r } { \| \nabla f ( X _ { t _ { n } } ) \| ^ { 2 } \leqslant 2 C ^ { \prime 2 } \eta _ { k } L \left( d + R _ { n } + 2 \log ( 1 / \delta ) \right) . } \end{array}\tag{29}
$$

Thus we can apply the following lemma.

Lemma B.7. Let $Y \geqslant 0$ be a random variable, $v > 0 ,$ and $0 \leqslant \xi \leqslant { \frac { 1 } { ? } } . \ I f Y ^ { 2 } \leqslant v \delta ^ { - \xi }$ with probability at least $1 - \delta$ for every $\textstyle 0 < \delta < { \frac { 1 } { 2 } }$ , then $\mathbb { E } [ Y ] \leqslant e ^ { \xi } \sqrt { v } .$

Proof of Lemma B.7. The assumption gives $\mathbb { P } ( Y > \sqrt { v } \delta ^ { - \xi / 2 } ) \leqslant \delta$ for every $\begin{array} { r } { 0 < \delta < \frac { 1 } { 2 } . \operatorname { I f } \xi = 0 } \end{array}$ then $Y \leqslant { \sqrt { v } }$ almost surely, so $\mathbb { E } [ Y ] \leqslant \sqrt { v }$ . If $\xi > 0$ , let $t _ { 0 } = 2 ^ { \xi / 2 } \sqrt { v } .$ . For $t > t _ { 0 }$ , the choice $\delta = ( \sqrt { v } / t ) ^ { \dot { 2 } / \xi } < \frac { 1 } { 2 }$ yields $\mathbb { P } ( Y > \dot { t } ) \stackrel { \cdot } { \leqslant } ( \sqrt { v } / t ) ^ { 2 / \xi }$ , and integrating the tail gives

$$
\mathbb { E } [ Y ] \leqslant t _ { 0 } + \int _ { t _ { 0 } } ^ { \infty } \left( \frac { \sqrt { v } } { t } \right) ^ { 2 / \xi } \mathrm { d } t = 2 ^ { \xi / 2 } \left( 1 + \frac { \xi } { 2 ( 2 - \xi ) } \right) \sqrt { v } \leqslant e ^ { \xi } \sqrt { v } .
$$

By (29), for every $\textstyle 0 < \delta < { \frac { 1 } { 2 } }$ , with probability at least $1 - \delta .$

$$
Y _ { n } ^ { 2 } \leqslant v \delta ^ { - \xi } , \qquad v : = \exp \left( 4 C ^ { \prime 2 } \lambda L h ( t _ { n } ) ^ { 2 } ( d + R _ { n } ) \right) , \qquad \xi : = 8 C ^ { \prime 2 } \lambda L h ( t _ { n } ) ^ { 2 } .
$$

If $h ( t _ { n } ) \leqslant \operatorname* { m i n } ( ( C T q ^ { 2 } L ^ { 2 } ) ^ { - 1 } , L ^ { - 1 } )$ with $C \geqslant 5 1 2 C ^ { \prime 2 }$ , then $\xi \leqslant \frac 1 2$ , so the conditions of Lemma B.7 are satisfied and

$$
\log \mathbb { E } _ { \mathbb { W } _ { T } } \left[ Y _ { n } \right] \leqslant \xi + \frac { 1 } { 2 } \log v \leqslant 8 C ^ { \prime 2 } \lambda L h ( t _ { n } ) ^ { 2 } \left( d + R _ { n } + 1 \right) .
$$

From data processing we have that

$$
\mathcal { R } _ { q } ( \hat { \mu } _ { T } | | \mu _ { T } ) \leqslant \mathcal { R } _ { q } ( P _ { T } | | W _ { T } ) .
$$

Lastly, note that for $\begin{array} { r } { \sum _ { r = 1 } ^ { k - 1 } N _ { r } \leqslant n < \sum _ { r = 1 } ^ { k } N _ { r } } \end{array}$ , all $\pi _ { t _ { n } }$ are the same, i.e. the chain remains time homogeneous. Thus we can take the max over $k = 1 , \dots , M$ and, by data processing, upper bound $R _ { n }$ by $\mathscr { R } _ { q } ( \mu _ { T _ { k - 1 } } | | \pi ^ { ( k ) } )$ , where $\begin{array} { r } { T _ { k - 1 } = \sum _ { r = 1 } ^ { k - 1 } \tau _ { r } } \end{array}$ . Note that $\pi ^ { ( k ) }$ denotes the stationary distribution for Langevin dynamics with learning rate $\eta _ { k }$ . Thus the result follows by inserting the bound on log $\mathbb { E } _ { \mathrm { W } _ { T } } [ Y _ { n } ]$ and (27) into the bound (28) on $\mathcal { R } _ { q } ( \mathrm { P } _ { T } \| \mathrm { W } _ { T } )$ , with $\lambda = 3 2 \check { q } ^ { 2 } T L ^ { 2 }$ and $h ( t _ { n } ) = h _ { k }$

## B.3 Convergence of Annealed ULA

The following theorem combines diffusion convergence and discretization over an annealing schedule with independently chosen noise levels and step sizes. Setting the step size equal to the noise level in a phase gives the SGD update in that phase.

Theorem B.8 (Convergence of annealed ULA). Let $f : \mathcal { X }  \mathbb { R }$ , where $\mathcal { X } \subseteq \mathbb { R } ^ { d }$ , have an $L _ { - }$ Lipschitz gradient. Consider M ULA phases (24) with positive noise levels $\eta _ { r }$ and step sizes $h _ { r } ,$ chosen independently. Suppose that $\pi ^ { ( \boldsymbol { r } ) } \propto e ^ { - 2 \dot { \boldsymbol { f } } / \dot { \eta } _ { r } }$ is integrable and satisfies a $( \delta _ { r } , \epsilon _ { r } ) { - } \bar { W P I }$ in the normalization (4). Assume that Theorems B.3 and B.6 apply to the diffusion and ULA kernels on X in every phase.

Let $q \geqslant 2 , \vartheta _ { r } \in ( 0 , 1 ] , K _ { r } \geqslant 1$ and $\rho _ { r } \in [ 0 , 1 ) f o r 1 \leqslant r \leqslant M$ . Suppose that

$$
\epsilon _ { r } K _ { r } ^ { 2 q - 1 } \leqslant \frac { \delta _ { r } \vartheta _ { r } } { 8 } .\tag{30}
$$

With $B , C$ the constants ofTheorem $B . 6 ,$ choose

$$
\tau _ { r } : = \frac { 2 q - 1 } { \delta _ { r } } \left( \log K _ { r } + 1 + \log \frac { 8 } { \vartheta _ { r } } \right) ,
$$

$$
h _ { r } \leqslant \operatorname* { m i n } \left\{ \frac { 1 } { L } , \frac { 1 } { 4 C \tau _ { r } q ^ { 2 } L ^ { 2 } } , \frac { \vartheta _ { r } } { 4 B \tau _ { r } q ^ { 2 } L ^ { 2 } ( 2 d + \log K _ { r } + 1 ) } \right\} , \qquad \tau _ { r } / h _ { r } \in \mathbb { N } .\tag{31}
$$

Write $Q _ { r }$ for the ULA transition kernel with duration $\tau _ { r } ,$ , noise level $\eta _ { r }$ and step size $h _ { r }$

Suppose that the initial law has a decomposition $\mu _ { 0 } = \alpha _ { 0 } \nu _ { 0 } + ( 1 - \alpha _ { 0 } ) \bar { \nu } _ { 0 }$ , where $\alpha _ { 0 } \geqslant 1 - p _ { 0 }$ and $p _ { 0 } \in [ 0 , 1 )$ . Startingfrom $\nu _ { 0 } ,$ , define recursively

$$
\sigma _ { r } : = \frac { \nu _ { r - 1 } ( \cdot \cap G _ { r } ) } { \nu _ { r - 1 } ( G _ { r } ) } , \qquad \nu _ { r } : = \sigma _ { r } Q _ { r } ,
$$

where the measurable sets $G _ { r }$ satisfy

$$
\nu _ { r - 1 } ( G _ { r } ) \geqslant 1 - \rho _ { r } , \qquad \left\| \frac { d \sigma _ { r } } { d \pi ^ { ( r ) } } \right\| _ { \infty } \leqslant K _ { r } .\tag{32}
$$

Then the law $\mu _ { r } : = \mu _ { 0 } Q _ { 1 } \cdot \cdot \cdot Q _ { r }$ of the actual ULA iterates satisfies, for every $1 \leqslant r \leqslant M ,$

$$
\mu _ { r } = \alpha _ { r } \nu _ { r } + ( 1 - \alpha _ { r } ) \bar { \nu } _ { r } ,
$$

$$
\alpha _ { r } \geqslant 1 - p _ { 0 } - \sum _ { j = 1 } ^ { r } \rho _ { j } , \qquad \mathcal { R } _ { q } ( \nu _ { r } \| \pi ^ { ( r ) } ) \leqslant \vartheta _ { r } ,\tag{33}
$$

for probability measures $\bar { \nu } _ { r }$ . In particular,

$$
\mathrm { T V } ( \mu _ { M } , \pi ^ { ( M ) } ) \leqslant p _ { 0 } + \sum _ { r = 1 } ^ { M } \rho _ { r } + \sqrt { \vartheta _ { M } / 2 } .\tag{34}
$$

The sets $G _ { r }$ restrict measures only in the analysis; the algorithm performs the original ULA updates and uses $\scriptstyle \sum _ { r = 1 } ^ { M } \tau _ { r } / h _ { r }$ gradient queries.

Proof. Fix a phase $r ,$ and write $P _ { \tau _ { r } } ^ { ( r ) }$ for the corresponding Langevin transition kernel. The density bound (32) implies $\mathcal { R } _ { s } ( \sigma _ { r } \Vert \pi ^ { ( r ) } ) \leqslant$ log $K _ { r }$ for every $s > 1$ . At order $2 q - 1$ , the transient time in Theorem B.3 is at most $( 2 q - 1 ) ( \log K _ { r } - 1 ) _ { + } / \delta _ { r }$

Equations (20), (30) and (31) therefore give

$$
\mathcal { R } _ { 2 q - 1 } ( \sigma _ { r } P _ { \tau _ { r } } ^ { ( r ) } \| \pi ^ { ( r ) } ) \leqslant ( \vartheta _ { r } / 8 ) ^ { 2 } + \frac { 2 \epsilon _ { r } K _ { r } ^ { 2 q - 1 } } { \delta _ { r } } \leqslant \frac { \vartheta _ { r } } { 2 } .\tag{35}
$$

Apply Theorem B.6 to this phase at order $2 q .$ Since $h _ { r } \leqslant 1 / L$ , (25) and (31) yield

$$
\begin{array} { r l r } & { } & { \mathcal { R } _ { 2 q } ( \nu _ { r } \| \sigma _ { r } P _ { \tau _ { r } } ^ { ( r ) } ) \leqslant \displaystyle \frac { 4 B \tau _ { r } q ^ { 2 } } { 2 q - 1 } \left[ L ^ { 3 } h _ { r } ^ { 2 } ( d + \log K _ { r } + 1 ) + L ^ { 2 } h _ { r } d \right] } \\ & { } & { \leqslant \displaystyle \frac { 4 B \tau _ { r } q ^ { 2 } L ^ { 2 } h _ { r } } { 2 q - 1 } ( 2 d + \log K _ { r } + 1 ) \leqslant \displaystyle \frac { \vartheta _ { r } } { 2 q - 1 } . } \end{array}\tag{36}
$$

The weak triangle inequality for Rényi divergence combines (35) and (36):

$$
\begin{array} { l } { \displaystyle \mathcal { R } _ { q } \big ( \nu _ { r } \| \pi ^ { ( r ) } \big ) \leqslant \frac { q - 1 / 2 } { q - 1 } \mathcal { R } _ { 2 q } \big ( \nu _ { r } \| \sigma _ { r } P _ { \tau _ { r } } ^ { ( r ) } \big ) + \mathcal { R } _ { 2 q - 1 } \big ( \sigma _ { r } P _ { \tau _ { r } } ^ { ( r ) } \| \pi ^ { ( r ) } \big ) } \\ { \displaystyle \leqslant \frac { \vartheta _ { r } } { 2 ( q - 1 ) } + \frac { \vartheta _ { r } } { 2 } \leqslant \vartheta _ { r } . } \end{array}
$$

We now propagate the initial decomposition. If $\mu _ { r - 1 } \geqslant \alpha _ { r - 1 } \nu _ { r - 1 }$ , then restriction to $G _ { r }$ and application of the Markov kernel $Q _ { r }$ give

$$
\mu _ { r } \geqslant \alpha _ { r - 1 } \nu _ { r - 1 } ( G _ { r } ) \sigma _ { r } Q _ { r } = \alpha _ { r } \nu _ { r } , \qquad \alpha _ { r } : = \alpha _ { r - 1 } \nu _ { r - 1 } ( G _ { r } ) .\tag{37}
$$

The remaining measure is nonnegative, so normalization gives $\bar { \nu } _ { r }$ . Induction yields $\alpha _ { r } \geqslant$ α<sub>0</sub> $\begin{array} { r } { \prod _ { j = 1 } ^ { r } ( 1 - \bar { \rho } _ { j } ) \geqslant 1 - p _ { 0 } - \sum _ { j = 1 } ^ { r } \bar { \rho } _ { j } } \end{array}$ , proving (33).

Finally, Pinsker’s inequality and monotonicity of Rényi divergence give $\mathrm { T V } ( \nu _ { M } , \pi ^ { ( M ) } ) \leqslant \sqrt { \vartheta _ { M } / 2 }$ Convexity of total variation in its first argument gives (34). Each ULA step uses one gradient query, which gives the stated query count. □

## B.4 Sampling for Block-Separable Functions

For the block analysis, let $g : \mathbb { R } ^ { \ell } $ R have an L-Lipschitz gradient, a minimizer a with $g ( a ) = 0$ and $\| a \| \leqslant B _ { 0 }$ , and $g ( y ) \geqslant m \| y - a \| ^ { 2 } / 2$ , where $\ell \leqslant b$ . Write $\pi _ { \beta } \propto e ^ { - \beta g }$ and use $A = ( 2 L / m ) ^ { b / 2 }$ $K _ { 0 } = A e ^ { L B _ { 0 } ^ { 2 } }$ and $U = 1 + \log K _ { 0 }$ , as in Theorem A.5.

The restriction in the following lemma is used only in the analysis. ULA itself does not test the objective against a cutoff or discard iterates.

Lemma B.9 (Density bounds relative to the Gibbs distribution of the next phase). With these block constants, suppose $\dot { \beta ^ { \prime } } \geqslant \beta > 0 , \mathcal { R } _ { 2 } ( \nu \| \pi _ { \beta } ) \leqslant U$ , and $\rho \in \left( 0 , 1 / 3 2 \right]$ . Define

$$
H : = \frac { 4 } { \beta } \left( \frac { U } { 2 } + \frac { \log A } { 2 } + \log \frac { 2 } { \rho } \right) ,
$$

$$
G : = \left\{ y : \frac { d \nu } { d \pi _ { \beta } } ( y ) \leqslant \frac { 2 e ^ { U } } { \rho } , \quad g ( y ) \leqslant H \right\} .\tag{38}
$$

$I f ( \beta ^ { \prime } - \beta ) H \leqslant 1$ , the normalized restriction $\sigma = \nu ( \cdot \cap G ) / \nu ( G )$ satisfies

$$
\nu ( G ^ { c } ) \leqslant \rho ,\tag{39}
$$

$$
\left. \frac { d \sigma } { d \pi _ { \beta ^ { \prime } } } \right. _ { \infty } \leqslant K : = \frac { 4 e ^ { U + 1 } } { \rho } , \qquad \mathcal { R } _ { 2 } ( \sigma \| \pi _ { \beta ^ { \prime } } ) \leqslant U + 2 .\tag{40}
$$

Proof. Apply Theorem B.4 with order 2 and discarded mass $\rho / 2$ . It bounds the mass above the density cutoff in (38) by $\rho / 2 .$ . Cauchy–Schwarz and (12) give the other half:

$$
\nu ( g > H ) \leqslant e ^ { U / 2 } \pi _ { \beta } ( g > H ) ^ { 1 / 2 } \leqslant e ^ { U / 2 } A ^ { 1 / 2 } e ^ { - \beta H / 4 } = \rho / 2 .
$$

Thus (39) holds. Normalization bounds the density of σ relative to $\pi _ { \beta }$ by $4 e ^ { U } / \rho$ and its order-two divergence by $U + 2 \log ( 1 / ( 1 - \rho ) ) \leqslant U + 1$

Since $g \geqslant 0$ , we have $Z _ { \beta ^ { \prime } } \leqslant Z _ { \beta }$ . On G,

$$
\frac { d \pi _ { \beta } } { d \pi _ { \beta ^ { \prime } } } = \frac { Z _ { \beta ^ { \prime } } } { Z _ { \beta } } e ^ { ( \beta ^ { \prime } - \beta ) g } \leqslant e .\tag{41}
$$

Changing reference measures using (41) increases the logarithmic density bound and the order-two divergence by at most 1, proving (40). □

The next lemma shows that, for an arithmetic schedule of learning rates, consecutive inverse temperatures satisfy the condition of Theorem B.9 in every phase.

Lemma B.10 (Number of phases for arithmetically decreasing learning rates). For $\beta _ { * } \geqslant 2$ and $\zeta \in ( 0 , 1 / 4 ]$ , define

$$
\begin{array} { l } { \displaystyle M : = \left\lceil 1 0 2 4 ( \beta _ { * } + 1 ) ^ { 2 } ( d / \zeta ) ^ { 2 } ( U + \log A + 1 ) ^ { 2 } \right\rceil , } \\ { \displaystyle \rho : = \frac \zeta { 8 d M } , \qquad \eta _ { r } : = 2 - ( r - 1 ) \frac { 2 - 2 / \beta _ { * } } { M - 1 } , \qquad \beta _ { r } : = 2 / \eta _ { r } . } \end{array}\tag{42}
$$

Set $\beta _ { 0 } = \beta _ { 1 }$ . With H<sub>r</sub> given by (38) at $\beta = \beta _ { r - 1 } ,$ every phase satisfies $( \beta _ { r } - \beta _ { r - 1 } ) H _ { r } \leqslant 1$

Proof. The assertion is immediate for $r = 1$ . Put $D = \beta _ { * } + 1 , a = d / \zeta$ , and $J = U + \log A + 1$ Then

$$
\begin{array} { r l } & { 5 1 2 D ^ { 2 } a ^ { 2 } J ^ { 2 } \leqslant M - 1 , \qquad M \leqslant 2 0 4 8 D ^ { 2 } a ^ { 2 } J ^ { 2 } , } \\ & { \log ( 1 6 a M ) \leqslant 1 1 + 2 D + 3 a + 2 J \leqslant 8 D a J . } \end{array}\tag{43}
$$

Here $D \geqslant 3 , a \geqslant 4 ,$ and $J \geqslant 1$ . For $r \geqslant 2 ,$ , the relative inverse-temperature increment is at most $( \beta _ { * } - 1 ) / ( M - 1 )$ . Consequently, (43) gives

$$
\begin{array} { c } { \displaystyle ( \beta _ { r } - \beta _ { r - 1 } ) H _ { r } \leqslant \frac { 4 D } { M - 1 } \left( \frac { J } { 2 } + \log ( 1 6 a M ) \right) } \\ { \displaystyle \leqslant \frac { 3 4 } { 5 1 2 a J } < 1 . } \end{array}
$$

We can now apply these estimates to ULA without repeating the diffusion convergence or discretization arguments.

Theorem B.11 (Annealed Unadjusted Langevin Algorithm for Block-Separable Functions). Let $f : \mathbb { R } ^ { d } $ R be block-separable (Theorem 1.1), $f ( \bar { x ) } = g ( R x )$ with an orthogonal matrix R and $\begin{array} { r } { g ( y ) = \sum _ { i = 1 } ^ { k } g _ { i } ( \hat { y } _ { i } ) } \end{array}$ , where $k \leqslant d$ and the blocks $\hat { y } _ { i } \in \mathbb { R } ^ { d _ { i } }$ have dimensions $d _ { i } \leqslant b .$

Suppose that each $g _ { i } \in C ^ { 2 } (  { \mathbb { R } } ^ { d _ { i } } )$ has an L-Lipschitz gradient and a unique global minimizer $a _ { i }$ with $\left\| \bar { a _ { i } } \right\| \leqslant B _ { 0 } ,$ , and that

$$
g _ { i } ( y ) - g _ { i } ( a _ { i } ) \geqslant \frac { m } { 2 } \Vert y - a _ { i } \Vert ^ { 2 }
$$

for all $\boldsymbol { y } \in \mathbb { R } ^ { d _ { i } }$ , where $b , m ,$ L and $B _ { 0 }$ do not depend on d.

Let $\beta _ { * } \geqslant 2 , \zeta \in ( 0 , 1 / 4 ]$ and $\pi _ { \beta _ { * } } ( x ) : = Z ^ { - 1 } \exp ( - \beta _ { * } f ( x ) )$ , where $\begin{array} { r } { Z : = \int _ { \mathbb { R } ^ { d } } \exp \bigl ( - \beta _ { * } f ( x ) \bigr ) d x } \end{array}$

Then there exists an unadjusted Langevin algorithm (24) startedfrom $\mathcal { N } ( 0 , ( 2 L ) ^ { - 1 } I _ { d } )$ , with learning rates $\{ \eta _ { r } \} _ { r = 1 } ^ { M }$ that decrease arithmetically from $\eta _ { 1 } = 2$ to $\eta _ { M } = 2 / \beta _ { * }$ <sub>∗</sub> and sufficiently small step sizes $\{ h _ { r } \} _ { r = 1 } ^ { M }$ chosen independently ofthe learning rates.

The algorithm uses a number of gradient queries polynomial in $d , \beta _ { * }$ <sub>∗</sub> and $1 / \zeta .$ . Its output distribution µ satisfies

$$
\mathrm { T V } ( \mu , \pi _ { \beta _ { \ast } } ) \leqslant \zeta .
$$

Proof. Normalize each block so that $g _ { i } ( a _ { i } ) = 0$ , and take the constants in $( 9 )$ . The orthogonal invariance of the Gaussian initialization and noise makes the rotated ULA iterates independent across blocks. Write $\lambda _ { i , \eta }$ for the law of block i after phase r and $\pi _ { i , \beta }$ for its Gibbs distribution.

Choose the arithmetic schedule and $\rho$ from (42), and set

$$
\vartheta : = \frac { \zeta ^ { 2 } } { 6 4 d ^ { 2 } } , \qquad K : = \frac { 4 e ^ { U + 1 } } { \rho } , \qquad s : = \frac { \vartheta } { 8 K ^ { 3 } } .\tag{44}
$$

Use the WPI constant δ in (13) for this s. For every phase, choose its duration and discretization step by

$$
\begin{array} { l } { \displaystyle \tau : = \frac { 3 } { \delta } \left( \log { K } + 1 + \log { \frac { 8 } { \vartheta } } \right) , } \\ { \displaystyle \bar { h } : = \operatorname* { m i n } \left\{ \frac { 1 } { L } , \frac { 1 } { 1 6 C \tau L ^ { 2 } } , \frac { \vartheta } { 1 6 B \tau L ^ { 2 } ( 2 b + \log { K } + 1 ) } \right\} , } \\ { \displaystyle h _ { r } : = \frac { \tau } { \left\lceil \tau / \bar { h } \right\rceil } . } \end{array}\tag{45}
$$

(46)

These choices use only the dimension, accuracy, final temperature and the stated regularity bounds, not the block decomposition or the minimizers.

Apply Theorem B.8 to each block with $q = 2 , p _ { 0 } = 0 , \vartheta _ { r } = \vartheta , K _ { r } = K , \rho _ { r } = \rho , \delta _ { r } = \delta$ and $\epsilon _ { r } = \delta s$ . The choices (45)–(46) satisfy (31), and (44) gives $\epsilon _ { r } K _ { r } ^ { 3 } = \delta \vartheta / 8$

The density conditions (32) are verified inductively. Initially, (11) gives $\mathcal { R } _ { 2 } ( \nu _ { i , 0 } \rVert \pi _ { i , \beta _ { 0 } } ) \leqslant$ log $K _ { 0 } \leqslant$ U. Given the same bound with U after a phase, Theorems B.9 and B.10 provide the set for the next phase, with discarded mass at most $\rho$ and density at most K. The Rényi divergence in the phase conclusion (33) is at most $\vartheta \leqslant U$ , so this construction continues through all phases.

The convergence theorem for annealed ULA therefore gives

$$
\begin{array} { r l } & { \lambda _ { i , r } = \alpha _ { i , r } \nu _ { i , r } + ( 1 - \alpha _ { i , r } ) \bar { \nu } _ { i , r } , } \\ & { \alpha _ { i , r } \geqslant 1 - r \rho , \qquad \mathcal { R } _ { 2 } ( \nu _ { i , r } \Vert \pi _ { i , \beta _ { r } } ) \leqslant \vartheta \leqslant U . } \end{array}\tag{47}
$$

At the last phase, Pinsker’s inequality and (47) yield

$$
\begin{array} { r } { \mathrm { T V } ( \lambda _ { i , M } , \pi _ { i , \beta _ { * } } ) \leqslant M \rho + \sqrt { \vartheta / 2 } < \displaystyle \frac { \zeta } { 2 d } , \qquad } \\ { \mathrm { T V } ( \mu , \pi _ { \beta _ { * } } ) \leqslant \displaystyle \sum _ { i = 1 } ^ { k } \mathrm { T V } ( \lambda _ { i , M } , \pi _ { i , \beta _ { * } } ) \leqslant \displaystyle \frac { \zeta } { 2 } \leqslant \zeta . } \end{array}
$$

Here we used independence across blocks and invariance of total variation under rotation.

Finally, $M , \rho ^ { - 1 } , K$ and $s ^ { - 1 }$ are polynomial in d, $\beta _ { * }$ and $\zeta ^ { - 1 }$ . The same holds for $\delta ^ { - 1 }$ by Theorem $\mathbf { A . } 5 .$ , hence for $\tau$ and $\bar { h } ^ { - 1 }  { \mathrm { ~ b y ~ } } ( 4 \bar { 5 } ) - ( 4 6 )$ . Each iteration uses one gradient query, and the total number is $M [ \tau / \bar { h } ]$ , which is polynomial in those parameters. □

## B.5 Global Optimization

Theorem B.11 implies a global optimization guarantee for the same algorithm on $\mathbb { R } ^ { d }$ , because for $\beta _ { * }$ of order $d / \epsilon ^ { 2 }$ the Gibbs distribution $\pi _ { \beta _ { * } }$ assigns most of its mass to the points x with $f ( x ) - f ^ { \star } < \epsilon .$

Theorem B.12 (Global Optimization of Block-Separable Functions via the Unadjusted Langevin Algorithm). Let f satisfy the hypotheses of Theorem $B . I I ,$ let $f ^ { \star } : = \operatorname* { m i n } _ { x \in \mathbb { R } ^ { d } } f ( x )$ , and let $\epsilon \in$ $( 0 , 1 ]$ . Then there exist $\dot { \beta } _ { * } = \dot { \Theta ( d / \epsilon ^ { 2 } ) }$ and an unadjusted Langevin algorithm (24) started from $\mathsf { \bar { N } } ( 0 , ( 2 L ) ^ { - 1 } I _ { d } )$ , with learning rates that decrease arithmetically from $\eta _ { 1 } = 2 t o \eta _ { M } = 2 / \beta _ { * }$ and sufficiently small step sizes chosen independently of the learning rates, that uses poly $( d / \epsilon )$ gradient queries and outputs a point x with $f ( x ) - f ^ { \star } < \epsilon$ with probability at least $2 / 3 .$

Proof. Adding constants to the functions $g _ { i }$ changes neither the algorithm nor the event $f ( x ) { - } f ^ { \star } < \epsilon .$ $\mathbf { s o } ,$ as in the proof of Theorem B.11, we assume $g _ { i } ( a _ { i } ) = 0$ for every $i \in [ k ]$ ]. Then $g _ { i } \geqslant 0$ and $f ^ { \star } = 0$

We first bound the mean of $f$ under the Gibbs distribution $\pi _ { \beta }$ ∝ $e ^ { - \beta f }$ on $\mathbb { R } ^ { d }$ . For $\beta > 0$ and $u \geqslant 0 ,$ the function $u \mapsto u e ^ { - \beta u / 2 }$ is at most $2 / ( e \beta )$ , so $\begin{array} { r } { u e ^ { - \beta u } \leqslant \frac { \dot { 2 } } { e \beta } e ^ { - \beta u / 2 } } \end{array}$ . We apply this inequality with $u = g _ { i } ( y )$ to the block Gibbs measures $\pi _ { i , \beta } ( y ) = Z _ { i , \beta } ^ { - 1 } e ^ { - \beta g _ { i } ( y ) }$ , where $\begin{array} { r } { Z _ { i , \beta } = \int _ { \mathbb { R } ^ { d _ { i } } } e ^ { - \beta g _ { i } ( y ) } d y } \end{array}$ , and use the bound $Z _ { i , \beta / 2 } / Z _ { i , \beta } \leqslant A$ of (12), where $A = ( 2 L / m ) ^ { b / 2 }$ . Since the rotation $x \mapsto R x$ sends $\pi _ { \beta }$ to the product of the measures $\pi _ { i , \beta }$ and $k \leqslant d ,$ we obtain, for every $\beta > 0$

$$
\begin{array} { r l } & { \mathbb { E } _ { \pi _ { i , \beta } } [ g _ { i } ] = \cfrac { 1 } { Z _ { i , \beta } } \displaystyle \int _ { \mathbb { R } ^ { d _ { i } } } g _ { i } ( y ) e ^ { - \beta g _ { i } ( y ) } d y \leqslant \cfrac { 2 } { e \beta } \cfrac { Z _ { i , \beta / 2 } } { Z _ { i , \beta } } \leqslant \cfrac { 2 A } { e \beta } , } \\ & { \mathbb { E } _ { \pi _ { \beta } } [ f - f ^ { \star } ] = \displaystyle \sum _ { i = 1 } ^ { k } \mathbb { E } _ { \pi _ { i , \beta } } [ g _ { i } ] \leqslant \cfrac { 2 A k } { e \beta } \leqslant \cfrac { 2 A d } { e \beta } . } \end{array}\tag{48}
$$

We choose $\beta _ { * } ~ : = ~ \operatorname* { m a x } \{ 2 , 2 4 A d / ( e \epsilon ^ { 2 } ) \}$ , which is $\Theta ( d / \epsilon ^ { 2 } )$ because A does not depend on $d .$ Markov’s inequality, (48) with $\beta = \beta _ { * }$ <sub>∗</sub> and $\epsilon \leqslant 1$ give

$$
\begin{array} { r l } & { \pi _ { \beta _ { \ast } } ( f - f ^ { \star } \geqslant \epsilon ) \leqslant \frac { \mathbb { E } _ { \pi _ { \beta _ { \ast } } } [ f - f ^ { \star } ] } { \epsilon } \leqslant \frac { 2 A d } { e \beta _ { \ast } \epsilon } \leqslant \frac { \epsilon } { 1 2 } \leqslant \frac { 1 } { 1 2 } , } \\ & { \pi _ { \beta _ { \ast } } ( f - f ^ { \star } < \epsilon ) \geqslant \frac { 1 1 } { 1 2 } . } \end{array}\tag{49}
$$

We now apply Theorem B.11 with this $\beta _ { * } \geqslant 2$ and $\zeta : = 1 / 1 2$ . The output distribution $\mu$ of the resulting algorithm satisfies $\mathrm { T V } ( \mu , \pi _ { \beta _ { * } } ) \leqslant 1 / 1 2$ . By the definition of the total variation distance and (49), the output $X \sim \mu$ therefore satisfies

$$
\begin{array} { r l r } & { } & { { \mathbb { P } } _ { \mu } [ f ( X ) - f ^ { \star } < \epsilon ] \geqslant \pi _ { \beta _ { \ast } } ( f - f ^ { \star } < \epsilon ) - \mathrm { T V } ( \mu , \pi _ { \beta _ { \ast } } ) } \\ & { } & { \geqslant \displaystyle \frac { 1 1 } { 1 2 } - \frac { 1 } { 1 2 } = \frac { 5 } { 6 } > \frac { 2 } { 3 } . \qquad } \end{array}
$$

Theorem B.11 also bounds the number of gradient queries by a polynomial in $d , \beta _ { * }$ and $1 / \zeta = 1 2$ Since $d \leqslant d / \epsilon$ and $\beta _ { * } \leqslant ( 2 + 2 4 A / e ) ( d / \epsilon ) ^ { 2 }$ , this number is polynomial in $d / \epsilon .$ □

## C Analysis of Stochastic Gradient Descent

We analyze stochastic gradient descent with decaying and fixed learning rates.

The stationary law of discrete SGD need not equal the Gibbs law of Langevin diffusion. For a quadratic block $g ( y ) = \lambda \| y \| ^ { 2 } / 2 \operatorname { o n } \mathbb { R } ^ { \ell }$ , with $\lambda > 0$ and $0 < \eta \lambda < 2$ , SGD has stationary covariance $\bar { \eta } [ 2 \lambda ( 1 - \eta \lambda / 2 ) ] ^ { - 1 } I _ { \ell } .$ , whereas the Gibbs covariance is $\eta ( 2 \dot { \lambda } ) ^ { - 1 } I _ { \ell } .$ A spectral-transfer condition must therefore concern the actual SGD kernel and account for its stationary bias.

## C.1 Sampling with Annealed Stochastic Gradient Descent

The following discrete analogue of the diffusion convergence bound uses the stationary reverse kernel.   
It does not require the SGD kernel to be reversible.

Lemma C.1 (Rényi convergence from a discrete weak Poincaré inequality). Let P be a Markov kernel with invariant probability measure $\rho ,$ and let $P ^ { * }$ be its adjoint in $L ^ { 2 } ( \rho )$ . Suppose that, for every boundedfunction v,

$$
\frac { \lVert v \rVert _ { L ^ { 2 } ( \rho ) } ^ { 2 } - \lVert P ^ { * } v \rVert _ { L ^ { 2 } ( \rho ) } ^ { 2 } } { \eta } \geqslant \frac { \delta } { \kappa } \operatorname { V a r } _ { \rho } ( v ) - \kappa \delta s \lVert v \rVert _ { \infty } ^ { 2 } ,\tag{50}
$$

where $\eta , \delta > 0 , \kappa \geqslant 1 , s \geqslant 0$ and $a : = \eta \delta / \kappa \leqslant 1$ . Let $r \geqslant 2$ and let the initial density satisfy $\| d \mu _ { 0 } / d \rho \| _ { \infty } \leqslant K < \infty$ . Write $u _ { n } : = \mathcal { R } _ { r } ( \mu _ { 0 } P ^ { n } \| \rho )$ and $b : = \kappa ^ { 2 } s K ^ { r }$ $I f b \leqslant 1 / 4$ , then

$$
u _ { n } \leqslant \exp \left[ - \frac { a ( n - n _ { 0 } ) } { 2 ( r - 1 ) } \right] + 2 b \quad ( n \geqslant n _ { 0 } ) , \qquad n _ { 0 } : = \left\lceil \frac { 4 ( r - 1 ) ( u _ { 0 } - 1 ) _ { + } } { a } \right\rceil .\tag{51}
$$

Proof. Stationarity makes $P ^ { * }$ a Markov kernel. The densities $f _ { n } : = d ( \mu _ { 0 } P ^ { n } ) / d \rho$ therefore satisfy $f _ { n + 1 } = P ^ { * } f _ { n }$ and $0 \leqslant f _ { n } \leqslant K$ . Let $\begin{array} { r } { F _ { n } : = \int f _ { n } ^ { r } d \rho = e ^ { ( r - 1 ) u _ { n } } } \end{array}$ . Jensen’s inequality and (50) applied to $f _ { n } ^ { r / 2 }$ give

$$
\begin{array} { r l } & { F _ { n } - F _ { n + 1 } \geqslant \| f _ { n } ^ { r / 2 } \| _ { L ^ { 2 } ( \rho ) } ^ { 2 } - \| P ^ { * } ( f _ { n } ^ { r / 2 } ) \| _ { L ^ { 2 } ( \rho ) } ^ { 2 } } \\ & { \qquad \geqslant a \left\lceil \mathrm { V a r } _ { \rho } ( f _ { n } ^ { r / 2 } ) - b \right\rceil . } \end{array}
$$

Monotonicity of Rényi divergence gives $\mathrm { V a r } _ { \rho } ( f _ { n } ^ { r / 2 } ) \geqslant F _ { n } ( 1 - e ^ { - u _ { n } } )$ . Since $F _ { n } \geqslant 1$ and $\log ( 1 + x ) \leqslant$ x, it follows that

$$
u _ { n + 1 } \leqslant u _ { n } - \frac { a } { r - 1 } ( 1 - e ^ { - u _ { n } } ) + \frac { a b } { r - 1 } .
$$

When $u _ { n } \ > \ 1$ , the decrement is at least $a / [ 4 ( r - 1 ) ]$ , so $u _ { n _ { 0 } } \leqslant 1$ . Data processing makes $u _ { n }$ non-increasing. For $0 \leqslant u _ { n } \leqslant 1$ , the bound $1 - e ^ { - u _ { n } } \geqslant u _ { n } / 2$ yields

$$
u _ { n + 1 } \leqslant \left( 1 - { \frac { a } { 2 ( r - 1 ) } } \right) u _ { n } + { \frac { a b } { r - 1 } } .
$$

Iterating this scalar recursion proves (51).

For a block potential $g _ { i } : \mathbb { R } ^ { d _ { i } }  \mathbb { R } .$ , let $P _ { i , \eta }$ be the actual SGD kernel

$$
Y ^ { \prime } = Y - \eta \nabla g _ { i } ( Y ) + \eta Z , \qquad Z \sim { \cal N } ( 0 , I _ { d _ { i } } ) ,
$$

and let $\pi _ { i } ^ { \eta } \propto e ^ { - 2 g _ { i } / \eta }$ be the corresponding Langevin Gibbs law. Its Dirichlet form is $\mathcal { E } _ { \pi _ { i } ^ { \eta } } ( v , v ) =$ $\begin{array} { r } { \frac { \eta } { 2 } \int \| \nabla v \| ^ { 2 } d \pi _ { i } ^ { \eta } } \end{array}$ . We call $( \delta , \epsilon )$ a continuous WPI pair if $\mathcal { E } _ { \pi _ { i } ^ { \eta } } ( v , v ) \geqslant \delta \mathrm { V a r } _ { \pi _ { i } ^ { \eta } } ( v ) - \epsilon \| \bar { v } \| _ { \infty } ^ { 2 }$ . The continuous pairs used below are supplied by Theorems A.5 and $_ { \mathrm { A } . 6 . }$

Assumption C.2 (Spectral transfer from Langevin diffusion to SGD). Fix $q \geqslant 2 .$ . There are constants $\eta _ { \mathrm { t r } } > 0 , C _ { \kappa } \geqslant $ 1 and $\nu _ { \kappa } \geqslant 0 ,$ , independent $o f d ,$ and $\begin{array} { r } { \mathsf { l } \leqslant \kappa _ { d } \leqslant C _ { \kappa } [ \log ( e d ) ] ^ { \nu _ { \kappa } } } \end{array}$ such that, for every block $g _ { i }$ and $0 < \eta \leqslant \eta _ { \mathrm { t r } } ,$ its SGD kernel $\dot { P } _ { i , \eta }$ has an invariant law $\rho _ { i } ^ { \eta } .$ . Let $P _ { i , \eta } ^ { * }$ be its adjoint in $L ^ { 2 } ( \rho _ { i } ^ { \eta } )$ and $\pi _ { i } ^ { \eta } \propto e ^ { - 2 g _ { i } / \eta }$ its Gibbs law. For every bounded v and every continuous WPI pair $( \delta , \epsilon )$ used in the analysis, we require

$$
\begin{array} { r l } & { \quad \quad \frac { \| v \| _ { L ^ { 2 } ( \rho _ { i } ^ { \eta } ) } ^ { 2 } - \| P _ { i , \eta } ^ { * } v \| _ { L ^ { 2 } ( \rho _ { i } ^ { \eta } ) } ^ { 2 } } { \eta } \geqslant \frac { \delta } { \kappa _ { d } } \operatorname { V a r } _ { \rho _ { i } ^ { \eta } } ( v ) - \kappa _ { d } \epsilon \| v \| _ { \infty } ^ { 2 } , } \\ & { \operatorname* { m a x } \{ \mathcal { R } _ { t } ( \rho _ { i } ^ { \eta } \| \pi _ { i } ^ { \eta } ) , \mathcal { R } _ { t } ( \pi _ { i } ^ { \eta } \| \rho _ { i } ^ { \eta } ) \} \leqslant 1 , \qquad 2 \leqslant t \leqslant 8 q . } \end{array}
$$

Here $\mathcal { R } _ { t }$ is the order-t Rényi divergence. The constants may depend on q and the uniform block regularity bounds.

The spectral inequality concerns one-step dissipation of the discrete kernel, not the distribution produced by an annealing run. Constant-factor transfer corresponds to bounded $\kappa _ { d } ;$ polylogarithmic distortion changes the phase lengths but not the polynomial-versus-exponential runtime conclusion. The constant comparisons control initialization, changes between consecutive invariant laws, and the density moments needed to prove the final stationary bias.

Lemma C.3 (One-step Gibbs discrepancy for SGD). Let $g _ { i }$ have an L-Lipschitz gradient and an integrable Gibbs law $\bar { \pi _ { i } ^ { \eta } } \propto e ^ { - 2 g _ { i } / \eta }$ on $\mathbb { R } ^ { d _ { i } }$ . Fix $t \geqslant 2 ,$ , and let $B , C$ be the constants ofTheorem B.6. If

$$
0 < \eta \leqslant \operatorname* { m i n } \{ L ^ { - 1 } , ( \sqrt C { { t } L } ) ^ { - 1 } \} ,
$$

then one actual SGD update satisfies

$$
\mathcal { R } _ { t } ( \pi _ { i } ^ { \eta } P _ { i , \eta } \| \pi _ { i } ^ { \eta } ) \leqslant \frac { 3 B t ^ { 2 } } { t - 1 } d _ { i } L ^ { 2 } \eta ^ { 2 } .\tag{52}
$$

Proof. Start the Langevin diffusion from $\pi _ { i } ^ { \eta } .$ , so its law remains $\pi _ { i } ^ { \eta }$ . Apply Theorem B.6 with one phase, duration $T = \eta ,$ , step size $h = \eta$ , and order t. Its step-size condition follows from $C \eta ^ { 2 } \dot { t } ^ { 2 } L ^ { 2 } \leqslant 1$ . Equation (25) gives

$$
\mathcal { R } _ { t } ( \pi _ { i } ^ { \eta } P _ { i , \eta } \| \pi _ { i } ^ { \eta } ) \leqslant \frac { B \eta t ^ { 2 } } { t - 1 } \left[ L ^ { 3 } \eta ^ { 2 } ( d _ { i } + 1 ) + L ^ { 2 } \eta d _ { i } \right] .
$$

Using $L \eta \leqslant 1$ and $d _ { i } + 1 \leqslant 2 d _ { i }$ proves (52).

The next result derives the stationary bias from the SGD update and the Gibbs weak Poincaré inequality. Constant comparisons with the Gibbs law control the density moments used in the proof. Lemma C.4 (Stationary bias from constant Gibbs comparisons). Let $g _ { i } \in C ^ { 2 } (  { \mathbb { R } } ^ { d _ { i } } )$ , where $1 \leqslant d _ { i } \leqslant$ b, have an L-Lipschitz gradient and a minimizer $a _ { i }$ with $g _ { i } ( a _ { i } ) = 0$ and $g _ { i } ( y ) \geq m \Vert y - a _ { i } \Vert ^ { 2 } / 2 .$ . Fix $q \geqslant 2 .$ . Suppose that, for $0 < \eta \leqslant \eta _ { * }$ , the SGD kernel $P _ { i , \eta }$ has an invariant law $\rho _ { i } ^ { \eta }$ satisfying

$$
\begin{array} { r } { \mathcal { R } _ { s } ( \rho _ { i } ^ { \eta } \| \pi _ { i } ^ { \eta } ) \leqslant 1 , \qquad 2 \leqslant s \leqslant 8 q , } \end{array}
$$

and that the Gibbs law satisfies

$$
\begin{array} { r } { \mathcal { E } _ { \pi _ { i } ^ { \eta } } ( w , w ) \geqslant \delta _ { * } \operatorname { V a r } _ { \pi _ { i } ^ { \eta } } ( w ) - \delta _ { * } C _ { * } e ^ { - c _ { * } / \eta } \| w \| _ { \infty } ^ { 2 } } \end{array}\tag{53}
$$

for bounded smooth w, withfixed positive $\delta _ { * } , C _ { * } , c _ { * }$ . There are constants $\eta _ { \mathrm { s t a t } } > 0$ and $B _ { \mathrm { s t a t } }$ ⩾ 1, depending only on $q , b , m , L , \eta _ { * } , \delta _ { * } , C _ { * } , c _ { * }$ , such that

$$
\begin{array} { r } { \mathcal { R } _ { t } \big ( \rho _ { i } ^ { \eta } \| \pi _ { i } ^ { \eta } \big ) \leqslant B _ { \mathrm { s t a t } } \eta , \qquad 2 \leqslant t \leqslant 2 q - 1 , \quad 0 < \eta \leqslant \eta _ { \mathrm { s t a t } } . } \end{array}\tag{54}
$$

Proof. Write $\pi = \pi _ { i } ^ { \eta }$ and $\rho = \rho _ { i } ^ { \eta }$ , and interpolate one Euler step by

$$
X _ { h } = X _ { 0 } - h \nabla g _ { i } ( X _ { 0 } ) + \sqrt { \eta } W _ { h } , \qquad X _ { 0 } \sim \rho , \quad 0 \leqslant h \leqslant \eta ,
$$

where $W _ { h }$ is standard Brownian motion independent of $X _ { 0 }$ . Let $Q _ { h }$ be this transition kernel, $\mu _ { h } = \rho Q _ { h }$ , and $f _ { h } = d \mu _ { h } / d \pi$ . Stationarity gives $\mu _ { 0 } = \mu _ { \eta } = \rho .$

The proof of Theorem C.3, with duration and step size $h \leqslant \eta ,$ gives $\mathcal { R } _ { s } ( \pi Q _ { h } \vert \vert \pi ) \leqslant C _ { s } h ^ { 2 }$ for $2 \leqslant s \leqslant 8 q$ . Choose $\eta _ { \mathrm { s t a t } } \leqslant$ min $\{ \eta _ { * } , 1 \}$ small enough that these bounds are at most $1 / 2$ and all their step-size conditions hold. The weak triangle inequality and data processing then give

$$
\begin{array} { l } { \displaystyle \mathcal { R } _ { s } ( \mu _ { h } | | \pi ) \leqslant \frac { s - 1 / 2 } { s - 1 } \mathcal { R } _ { 2 s } ( \rho | | \pi ) + \mathcal { R } _ { 2 s - 1 } ( \pi Q _ { h } | | \pi ) \leqslant 2 , \qquad 2 \leqslant s \leqslant 4 q , } \\ { \displaystyle \int f _ { h } ^ { s } d \pi \leqslant e ^ { 2 ( s - 1 ) } , \qquad 2 \leqslant s \leqslant 4 q . } \end{array}\tag{55}
$$

The quadratic bounds and Gaussian integration give $\mathbb { E } _ { \pi } \Vert \nabla g _ { i } \Vert ^ { 8 } \leqslant C _ { g } \eta ^ { 4 }$ . Hölder’s inequality and $\mathcal { R } _ { 2 } ( \rho \| \boldsymbol { \pi } ) \leqslant 1$ therefore imply $\mathbb { E } _ { \rho } \Vert \nabla g _ { i } \Vert ^ { 4 } \leqslant C _ { q } ^ { \prime } \eta ^ { 2 }$ . Consequently,

$$
\begin{array} { r } { \mathbb { E } \| X _ { h } - X _ { 0 } \| ^ { 4 } \leqslant 8 h ^ { 4 } C _ { g } ^ { \prime } \eta ^ { 2 } + 8 \eta ^ { 2 } h ^ { 2 } d _ { i } ( d _ { i } + 2 ) \leqslant C _ { \Delta } \eta ^ { 2 } h ^ { 2 } . } \end{array}\tag{56}
$$

All constants here are uniform over $d _ { i } \leqslant b .$

Fix $r \in [ 2 , 2 q - 1 ]$ , and put $\begin{array} { r } { F ( h ) \ = \ \int f _ { h } ^ { r } } \end{array}$ dπ and $u ( h ) \ : = \ : \mathcal { R } _ { r } ( \mu _ { h } \| \pi )$ . The drift error of the interpolated law is

$$
e _ { h } ( x ) : = \nabla g _ { i } ( x ) - \mathbb { E } [ \nabla g _ { i } ( X _ { 0 } ) \mid X _ { h } = x ] .
$$

Conditional Jensen’s inequality, Cauchy–Schwarz, (55), and (56) give

$$
\begin{array} { l } { { \displaystyle { D ( h ) : = \frac { 1 } { F ( h ) } \int f _ { h } ^ { r } \| e _ { h } \| ^ { 2 } d \pi } } } \\ { { \displaystyle { \leqslant \frac { L ^ { 2 } } { F ( h ) } \left( \int f _ { h } ^ { 2 r - 1 } d \pi \right) ^ { 1 / 2 } \left( \mathbb { E } \| X _ { h } - X _ { 0 } \| ^ { 4 } \right) ^ { 1 / 2 } \leqslant C _ { e } \eta h , } } } \end{array}\tag{57}
$$

where $F ( h ) \geqslant 1$ . The Fokker–Planck equation, integration by parts, and Young’s inequality yield, with $v _ { h } = f _ { h } ^ { r / 2 }$

$$
\begin{array} { l l l } { \displaystyle { u ^ { \prime } ( h ) = - \frac { \eta r } { 2 F ( h ) } \int f _ { h } ^ { r - 2 } \| \nabla f _ { h } \| ^ { 2 } d \pi + \frac { r } { F ( h ) } \int f _ { h } ^ { r - 1 } \langle \nabla f _ { h } , e _ { h } \rangle d \pi } } \\ { \displaystyle { \leqslant - \frac { 2 } { r F ( h ) } \mathcal { E } _ { \pi } ( v _ { h } , v _ { h } ) + \frac { r } { \eta } D ( h ) . } } \end{array}\tag{58}
$$

Gaussian transition densities and the moment bounds justify these operations by smooth truncation. To apply the WPI to the unbounded $v _ { h } .$ , write $s _ { \eta } = C _ { * } e ^ { - c _ { * } / \eta }$ . For a nonnegative $v \in L ^ { 4 } ( \pi )$ , apply (53) to smooth approximations of $v \wedge K$ . The bounds

$$
\begin{array} { r l r } & { } & { \mathrm { V a r } _ { \pi } ( v ) \leqslant 2 \mathrm { V a r } _ { \pi } ( v \wedge K ) + 2 \| v - v \wedge K \| _ { L ^ { 2 } ( \pi ) } ^ { 2 } } \\ & { } & { \leqslant \displaystyle \frac { 2 } { \delta _ { * } } { \mathcal E } _ { \pi } ( v , v ) + 2 s _ { \eta } K ^ { 2 } + \frac { 2 } { K ^ { 2 } } \int v ^ { 4 } d \pi } \end{array}
$$

and the choice $\begin{array} { r } { K ^ { 4 } = ( \int v ^ { 4 } d \pi ) / s _ { \eta } } \end{array}$ give

$$
\mathrm { V a r } _ { \pi } ( v ) \leqslant \frac { 2 } { \delta _ { * } } \mathcal { E } _ { \pi } ( v , v ) + 4 \sqrt { s _ { \eta } } \left( \int v ^ { 4 } d \pi \right) ^ { 1 / 2 } .\tag{59}
$$

For $v = v _ { h }$ , Rényi monotonicity gives $\operatorname { V a r } _ { \pi } ( v _ { h } ) / F ( h ) \geqslant 1 - e ^ { - u ( h ) }$ . Equation (55) bounds $\int v _ { h } ^ { 4 }$ dπ uniformly and gives $u ( h ) \leqslant 2$ , so $1 - e ^ { - u ( h ) } \geqslant u ( h ) / 4$ . Combining (57), (58), and (59) yields uniform constants $A _ { 0 } , \dot { A } _ { 1 } ^ { \cdot } > 0$ for which

$$
u ^ { \prime } ( h ) \leqslant - \lambda u ( h ) + A _ { 0 } e ^ { - c _ { * } / ( 2 \eta ) } + A _ { 1 } h , \qquad \lambda : = \frac { \delta _ { * } } { 8 q } .\tag{60}
$$

Integrating (60) over $[ 0 , \eta ]$ and using $u ( \eta ) = u ( 0 )$ gives

$$
u ( 0 ) \leqslant \frac { A _ { 0 } } { \lambda } e ^ { - c _ { * } / ( 2 \eta ) } + \frac { A _ { 1 } } { \lambda } \eta .
$$

Since $e ^ { - c _ { * } / ( 2 \eta ) } \leqslant 2 \eta / ( e c _ { * } )$ , taking B<sub>stat</sub> = max{1, λ<sup>−1</sup>(A<sub>1</sub> + 2A<sub>0</sub>/(ec<sub>∗</sub>))} proves (54).

We first state the annealing argument for one block. The neighboring Gibbs comparison and the continuous WPI pairs will be verified explicitly in the application to block-separable functions below.

Theorem C.5 (Convergence of annealed SGD under spectral transfer). Let $P _ { r } = P _ { \eta _ { r } }$ be the SGD kernels at noise levels $\eta _ { 1 } > \cdots > \eta _ { M } > 0$ , with Gibbs laws $\pi _ { r } \propto e ^ { - 2 g / \eta _ { r } }$ and invariant laws $\rho _ { r }$ Suppose that Theorem C.2 holds, that $\eta _ { 1 } \leqslant \eta _ { \mathrm { t r } }$ , and that

$$
\begin{array} { r l } & { \mathcal { R } _ { 8 q - 6 } \big ( \pi _ { r - 1 } \| \pi _ { r } \big ) \leqslant 1 \quad ( 2 \leqslant r \leqslant M ) , } \\ & { \quad \quad \quad \left\| \frac { d \mu _ { 0 } } { d \pi _ { 1 } } \right\| _ { \infty } \leqslant K _ { \mathrm { i n i t } } < \infty . } \end{array}\tag{61}
$$

Let $p \in ( 0 , 1 / 4 8 ]$ and $\theta \in ( 0 , 1 ]$ . Suppose that g and its low-temperature Gibbs family satisfy the smoothness, quadratic-growth, and WPI hypotheses of Theorem C.4 on $0 < \eta \leqslant \eta _ { * } \leqslant \eta _ { \mathrm { t r } }$ . Let

$\eta _ { \mathrm { s t a t } } , B _ { \mathrm { s t a t } }$ be the constants proved there, and choose $\eta _ { M } \leqslant$ min $\{ \eta _ { \mathrm { s t a t } } , \theta / ( 2 B _ { \mathrm { s t a t } } ) \}$ }. Define

$$
c _ { t } : = \frac { t - 1 / 2 } { t - 1 } ,
$$

$$
H : = \operatorname* { m a x } \{ 6 , c _ { q } \log K _ { \mathrm { i n i t } } + 1 \} ,
$$

$$
\varrho : = p / M ,
$$

$$
K : = \frac { e ^ { H } \varrho ^ { - 1 / ( q - 1 ) } } { 1 - \varrho } ,
$$

$$
v : = { \frac { \theta } { 2 c _ { q } } } .\tag{62}
$$

For each phase, choose a continuous WPI pair $\left( \delta _ { r } , \epsilon _ { r } \right)$ satisfying

$$
\kappa _ { d } ^ { 2 } \frac { \epsilon _ { r } } { \delta _ { r } } K ^ { 2 q } \leqslant \frac { v } { 4 } , \qquad a _ { r } : = \frac { \eta _ { r } \delta _ { r } } { \kappa _ { d } } \leqslant 1 ,\tag{63}
$$

and run

$$
N _ { r } : = \left\lceil \frac { 4 ( 2 q - 1 ) ( \log K - 1 ) _ { + } } { a _ { r } } \right\rceil + \left\lceil \frac { 2 ( 2 q - 1 ) } { a _ { r } } \log \frac { 2 } { v } \right\rceil\tag{64}
$$

SGD updates. Then the actual output law satisfies

$$
\begin{array} { r l } & { \mu = \alpha \nu + ( 1 - \alpha ) \bar { \nu } , } \\ & { \alpha \geqslant 1 - p \geqslant 1 - 3 p , \qquad \mathcal { R } _ { q } ( \nu \| \pi _ { M } ) \leqslant \theta . } \end{array}\tag{65}
$$

The algorithm performs $\textstyle \sum _ { r } N _ { r }$ gradient queries.

Proof. The weak triangle inequality gives

$$
\begin{array} { r } { \mathcal { R } _ { q } ( \mu _ { 0 } \| \rho _ { 1 } ) \leqslant c _ { q } \mathcal { R } _ { 2 q } ( \mu _ { 0 } \| \pi _ { 1 } ) + \mathcal { R } _ { 2 q - 1 } ( \pi _ { 1 } \| \rho _ { 1 } ) \leqslant c _ { q } \log K _ { \mathrm { i n i t } } + 1 \leqslant H . } \end{array}
$$

For consecutive invariant laws, apply the same inequality twice, with $t = 2 q - 1$

$$
\begin{array} { r l } & { \mathcal { R } _ { t } ( \rho _ { r - 1 } \| \rho _ { r } ) \leqslant c _ { t } \mathcal { R } _ { 2 t } \big ( \rho _ { r - 1 } \| \pi _ { r - 1 } \big ) + c _ { 2 t - 1 } \mathcal { R } _ { 4 t - 2 } \big ( \pi _ { r - 1 } \| \pi _ { r } \big ) + \mathcal { R } _ { 4 t - 3 } \big ( \pi _ { r } \| \rho _ { r } \big ) } \\ & { \qquad \leqslant c _ { t } + c _ { 2 t - 1 } + 1 < 4 . } \end{array}
$$

All stationary comparison orders are at most $8 q .$ , and the order of the neighboring Gibbs comparison is $4 t - 2 = \dot { 8 } q - 6$

We now construct the retained component phase by phase, starting with $\nu _ { 0 } : = \mu _ { 0 }$ Given $\mathcal { R } _ { 2 q } ( \nu _ { r - 1 } \| \rho _ { r - 1 } ) \leqslant v \leqslant 1$ , the weak triangle inequality and the bound for consecutive invariant laws give $\mathcal { R } _ { q } ( \nu _ { r - 1 } \| \rho _ { r } ) \leqslant c _ { q } v + 4 < 6 \leqslant H$ . In the first phase use the initial bound just proved. Applying Theorem B.4 at order $q$ with discarded mass $\varrho$ gives a normalized restriction $\sigma _ { r }$ with density at most K relative to $\rho _ { r }$

The spectral-transfer condition gives (50) with $s = \epsilon _ { r } / \delta _ { r }$ . Apply Theorem C.1 at order $2 q$ , and let $n _ { 0 , r }$ be its transient step count. Its defect parameter is at most $v / 4$ by (63), and the phase length (64) gives

$$
\mathcal { R } _ { 2 q } ( \nu _ { r } \Vert \rho _ { r } ) \leqslant e ^ { - a _ { r } ( N _ { r } - n _ { 0 , r } ) / [ 2 ( 2 q - 1 ) ] } + v / 2 \leqslant v , \qquad \nu _ { r } : = \sigma _ { r } P _ { r } ^ { N _ { r } } .
$$

Thus the same bound holds after every phase. The restrictions are used only in the proof. The mass-propagation argument (37), applied to the Markov kernels $P _ { r } ^ { N _ { r } }$ , gives α $\dot { \geqslant } ( 1 - \varrho ) ^ { \dot { M } } \geqslant 1 - p$ for the retained component of the actual SGD law.

At the last phase, the weak triangle inequality and the proved stationary-bias estimate (54) give

$$
\begin{array} { r } { \mathcal { R } _ { q } ( \nu _ { M } \| \pi _ { M } ) \leqslant c _ { q } \mathcal { R } _ { 2 q } ( \nu _ { M } \| \rho _ { M } ) + \mathcal { R } _ { 2 q - 1 } ( \rho _ { M } \| \pi _ { M } ) \leqslant c _ { q } v + B _ { \mathrm { s t a t } } \eta _ { M } \leqslant \theta . } \end{array}
$$

This proves (65); each SGD step uses one gradient query.

Theorem C.6 (Nonconvex Sampling in Total Variation). Under the hypotheses and phase choices of Theorem C.5, the actual annealed SGD output satisfies

$$
\mathrm { T V } ( \mu , \pi _ { M } ) \leqslant p + \sqrt { \theta / 2 } \leqslant 3 p + \sqrt { \theta / 2 } .
$$

Proof. Apply the derivation of the total-variation bound (34) to the output decomposition (65), whose exceptional mass is at most $p .$ □

## C.2 Sampling for Block-Separable Functions

We apply the weak Poincaré estimates of Appendix A to the block Gibbs measures and combine the resulting block distributions.

Let $f :  { \mathbb { R } ^ { d } } \to  { \mathbb { R } }$ have the form $f ( x ) = g ( R x )$ , where R is orthogonal and $\begin{array} { r } { g ( y ) = \sum _ { i = 1 } ^ { k } g _ { i } ( \hat { y } _ { i } ) } \end{array}$ , with blocks $\hat { y } _ { i } \in \mathbb { R } ^ { d _ { i } }$ <sup>i</sup> satisfying $d _ { i } \leqslant b$ and $\textstyle \sum _ { i } d _ { i } = d .$ Each $g _ { i } \in C ^ { 2 } (  { \mathbb { R } } ^ { d _ { i } } )$ has an L-Lipschitz gradient and a unique minimizer $a _ { i } ,$ and satisfies

$$
g _ { i } ( y ) - g _ { i } ( a _ { i } ) \geqslant \frac { m } { 2 } \Vert y - a _ { i } \Vert ^ { 2 } \qquad \mathrm { f o r ~ a l l ~ } y \in \mathbb { R } ^ { d _ { i } } .
$$

Assume that there are common constants $r _ { 0 } > 0$ and $0 < \xi \leqslant L$ such that

$$
\begin{array} { r } { \nabla ^ { 2 } g _ { i } ( y ) \succeq \xi I _ { d _ { i } } \qquad \mathrm { w h e n e v e r } \| y - a _ { i } \| \leqslant r _ { 0 } , \quad 1 \leqslant i \leqslant k . } \end{array}\tag{66}
$$

A sufficient condition is the existence of a common radius on which $\| \nabla ^ { 2 } g _ { i } ( y ) - \nabla ^ { 2 } g _ { i } ( a _ { i } ) \| _ { \mathrm { o p } } \leqslant m / 2$ since quadratic growth gives $\nabla ^ { 2 } g _ { i } ( a _ { i } ) \succeq m I _ { d _ { i } }$ , and then (66) holds with $\xi = m / 2$

For $\beta > 0$ , define the block Gibbs laws by

$$
\pi _ { i , \beta } ( d y ) : = Z _ { i , \beta } ^ { - 1 } e ^ { - \beta g _ { i } ( y ) } d y , \qquad Z _ { i , \beta } : = \int _ { \mathbb { R } ^ { d _ { i } } } e ^ { - \beta g _ { i } ( y ) } d y .
$$

Apply Theorem A.6 to every block with the same radius $r _ { 0 }$ and curvature ξ from (66). In its proof, $d _ { i } \ \leqslant \ b$ and the local Hessian bound satisfies $L _ { 0 } \leqslant L$ . Thus the Gaussian tail bounds and the comparison of the normalizing constants are uniform over all blocks. For example, take

$$
\begin{array} { r } { c : = \operatorname* { m i n } \{ m r _ { 0 } ^ { 2 } / 8 , \xi r _ { 0 } ^ { 2 } / 3 2 \} , \qquad C : = 8 ( L / m ) ^ { b / 2 } + 1 6 ( L / \xi ) ^ { b / 2 } . } \end{array}
$$

Once $\beta _ { 0 }$ is sufficiently large in terms of $b , m , L , r _ { 0 } , \xi ,$ , the two tail contributions in that proof are bounded by $C e ^ { - c \beta }$ for every $\beta \geqslant \beta _ { 0 }$ . Therefore the constants $c , C , \beta _ { 0 }$ depend only on these uniform regularity constants, and every block satisfies

$$
\mathrm { V a r } _ { \pi _ { i , \beta } } ( v ) \leqslant \frac { 4 } { \xi \beta } \int \| \nabla v \| ^ { 2 } d \pi _ { i , \beta } + C e ^ { - c \beta } \| v \| _ { \infty } ^ { 2 } , \qquad \beta \geqslant \beta _ { 0 } ,
$$

valid for every block i and every bounded smooth function $v : \mathbb { R } ^ { d _ { i } }  \mathbb { R } .$

Fix $q \geqslant 2 , p \in ( 0 , 1 / 4 8 ]$ and $\theta \in ( 0 , 1 ]$ . Let $\eta _ { \mathrm { s t a t } } , B _ { \mathrm { s t a t } }$ be the constants of Theorem $\mathrm { C . 4 } .$ , applied with

$$
\begin{array} { l } { \delta _ { * } : = \xi / 4 , } \\ { c _ { * } : = 2 c , } \end{array}
$$

$$
\begin{array} { r l } & { C _ { \ast } : = C , } \\ & { \eta _ { \ast } : = \operatorname* { m i n } \{ \eta _ { \mathrm { t r } } , 2 / \beta _ { 0 } \} . } \end{array}
$$

Under Theorem C.2, choose

$$
\eta _ { 1 } : = \operatorname* { m i n } \{ 1 , \eta _ { \mathrm { t r } } , 1 / ( 8 q L ) \} , \qquad \beta _ { 1 } : = 2 / \eta _ { 1 } .
$$

Initialize at a specified deterministic point $x _ { 0 } \in \mathbb { R } ^ { d }$ . Write $y _ { i , 0 } : = ( R x _ { 0 } )$ <sub>i</sub> and assume that the initial block energies satisfy

$$
G _ { 0 } : = \operatorname* { m a x } _ { i } [ g _ { i } ( y _ { i , 0 } ) - g _ { i } ( a _ { i } ) ] < \infty .
$$

After one SGD update at the constant learning rate $\eta _ { 1 }$ , the law of each block of the iterate satisfies the density bound needed by the annealing proof.

Lemma C.7 (One-step initialization from a point). Under the block regularity conditions above, let

$$
X _ { 1 } = x _ { 0 } - \eta _ { 1 } \nabla f ( x _ { 0 } ) + \eta _ { 1 } Z , \qquad Z \sim \mathcal { N } ( 0 , I _ { d } ) .
$$

The blocks of $R X _ { 1 }$ are independent. $I f \mu _ { i , \mathrm { i n i t } }$ denotes the law of block i, then

$$
\left\| \frac { d \mu _ { i , \mathrm { i n i t } } } { d \pi _ { i , \beta _ { 1 } } } \right\| _ { \infty } \leqslant K _ { \mathrm { i n i t } } : = ( 2 m \eta _ { 1 } ) ^ { - b / 2 } \exp \left( \frac { 2 L G _ { 0 } ( 1 + L \eta _ { 1 } ) ^ { 2 } } { m \eta _ { 1 } ( 1 - 2 L \eta _ { 1 } ) } \right) .\tag{67}
$$

For fixed $G _ { 0 }$ and regularity constants, $K _ { \mathrm { i n i t } }$ is independent of d and the target accuracy.

Proof. Subtract $g _ { i } ( a _ { i } )$ from each block potential. This changes neither the update nor the Gibbs law. Orthogonal invariance of $Z$ gives independent Gaussian blocks with means

$$
\bar { y } _ { i } : = y _ { i , 0 } - \eta _ { 1 } \nabla g _ { i } ( y _ { i , 0 } )
$$

and covariances $\eta _ { 1 } ^ { 2 } I _ { d _ { i } }$ . The quadratic bounds give

$$
g _ { i } ( y ) \leqslant \frac { L } { 2 } \| y - a _ { i } \| ^ { 2 } ,
$$

$$
Z _ { i , \beta _ { 1 } } \leqslant ( \pi \eta _ { 1 } / m ) ^ { d _ { i } / 2 } ,
$$

$$
\| \bar { y } _ { i } - a _ { i } \| ^ { 2 } \leqslant \frac { 2 G _ { 0 } } { m } ( 1 + L \eta _ { 1 } ) ^ { 2 } .
$$

Consequently, for $z : = y - a _ { i }$ and $w _ { i } : = \bar { y } _ { i } - a _ { i } .$

$$
\frac { d \mu _ { i , \mathrm { i n i t } } } { d \pi _ { i , \beta _ { 1 } } } ( y ) \leqslant ( 2 m \eta _ { 1 } ) ^ { - d _ { i } / 2 } \exp \left( - \frac { \| z - w _ { i } \| ^ { 2 } } { 2 \eta _ { 1 } ^ { 2 } } + \frac { L } { \eta _ { 1 } } \| z \| ^ { 2 } \right) .
$$

The exponent is maximized at $z = w _ { i } / ( 1 - 2 L \eta _ { 1 } )$ , with maximum $L \| w _ { i } \| ^ { 2 } / [ \eta _ { 1 } ( 1 - 2 L \eta _ { 1 } ) ]$ . Since m $\leqslant \bar { L ^ { } } , 2 L \eta _ { 1 } \leqslant 1 / ( 4 q ) < 1$ and $d _ { i } \leqslant b ,$ this proves (67). □

The algorithm performs this update in the original coordinates and does not need R, the block decomposition, or the minimizers. The bound on $G _ { 0 }$ is a hypothesis on the given starting point. The annealing analysis starts from the Gaussian distributions of the blocks after the first SGD update. The first SGD update uses the same learning rate as the first annealing phase, so it adds one gradient query but no additional temperature level.

Theorem C.8 (Annealed SGD for Block-Separable Functions). Let $\begin{array} { r } { f ( x ) = \sum _ { i = 1 } ^ { k } g _ { i } ( ( R x ) _ { i } ) } \end{array}$ satisfy the block regularity conditions above, and suppose that Theorem C.2 holds. Let $\dot { \epsilon } \in ( 0 , 1 ]$ and choose

$$
\beta _ { M } \geqslant \operatorname* { m a x } \{ C _ { \mathrm { o p t } } d / \epsilon ^ { 2 } , 4 / \eta _ { 1 } , \beta _ { 0 } , 2 / \eta _ { \mathrm { s t a t } } , 4 B _ { \mathrm { s t a t } } d / \theta \} ,
$$

where $C _ { \mathrm { o p t } }$ is the constant ofTheorem C.9.

There is an arithmetic learning-rate schedulefrom $\eta _ { 1 }$ to $2 / \beta _ { M }$ that uses the SGD update in every phase and whose output satisfies

$$
\mu = \alpha \nu + ( 1 - \alpha ) \bar { \nu } , \qquad \alpha \geqslant 1 - p \geqslant 1 - 3 p , \qquad \mathcal { R } _ { q } \mathopen { } \mathclose \bgroup ( \nu \mathopen { } \mathclose \bgroup \| \pi _ { \beta _ { M } } \aftergroup \egroup ) \leqslant \theta ,\tag{68}
$$

where $\pi _ { \beta _ { M } } \propto e ^ { - \beta _ { M } f } .$ For fixed $q$ and regularity constants, the number of gradient queries, including the gradient query for the initial SGD update, is polynomial in d, $\beta _ { M } , K _ { \mathrm { i n i t } } , 1 / p$ and $1 / \theta .$ $I f \beta _ { M } = \Theta ( \breve { d } / \epsilon ^ { 2 } )$ and $G _ { 0 }$ is bounded independently of d and $\epsilon ,$ this is poly(d/ϵ). A constant or polylogarithmic distortion $\kappa _ { d }$ preserves this polynomial bound.

Proof. Rotate to $R = I$ and subtract each $g _ { i } ( a _ { i } )$ . By Theorem $\mathrm { C } . 7 ,$ , the first SGD update from $x _ { 0 }$ produces independent block laws whose densities relative to $\pi _ { i , \beta _ { 1 } }$ are at most $K _ { \mathrm { i n i t } }$ . We apply the phasewise annealing argument to these laws and count the first update separately.

Let $A : = ( 2 L / m ) ^ { b / 2 }$ and $D : = \operatorname* { m a x } \{ 1 , 1 6 q , 4 A / e \}$ . Choose

$$
\begin{array} { r l } & { M : = 3 + \lceil D ( \beta _ { M } + 1 ) ^ { 2 } \rceil , } \\ & { \eta _ { r } : = \eta _ { 1 } - ( r - 1 ) \frac { \eta _ { 1 } - 2 / \beta _ { M } } { M - 1 } , \qquad \beta _ { r } : = 2 / \eta _ { r } . } \end{array}
$$

For consecutive temperatures, put $\beta = \beta _ { r - 1 } , \Delta = \beta _ { r } - \beta$ and $t = 8 q - 6$ . The schedule gives $\Delta / \beta \leqslant ( \beta _ { M } - 1 ) / ( \dot { M } - 1 ) \leqslant \dot { 1 } / \dot { D } .$ so $( t - 1 ) \Delta / \beta \leqslant 1 / 2$

The block Gibbs mean bound (48) gives $\mathbb { E } _ { \pi _ { i , u } } g _ { i } \leqslant 2 A / ( e u )$ . With $\begin{array} { r } { Z _ { i } ( u ) : = \int e ^ { - u g } } \end{array}$ <sup>i</sup> , it follows that

$$
\begin{array} { l } { \displaystyle \mathcal { R } _ { t } \big ( \pi _ { i , \beta } \| \pi _ { i , \beta + \Delta } \big ) = \log \frac { Z _ { i } ( \beta + \Delta ) } { Z _ { i } ( \beta ) } + \frac { 1 } { t - 1 } \log \frac { Z _ { i } ( \beta - ( t - 1 ) \Delta ) } { Z _ { i } ( \beta ) } } \\ { \displaystyle \leqslant \frac { 2 A } { e ( t - 1 ) } \log \frac { \beta } { \beta - ( t - 1 ) \Delta } \leqslant \frac { 4 A \Delta } { e \beta } \leqslant 1 . } \end{array}
$$

Thus the neighboring Gibbs condition (61) is proved for every block.

Use the common H, K from (62), with $\varrho = p / ( d M )$ , and let $v : = \theta / ( 2 c _ { q } d )$ . These parameters are defined in terms of the ambient dimension, not the unknown number of blocks. Set

$$
s _ { 0 } : = \frac { v } { 4 \kappa _ { d } ^ { 2 } K ^ { 2 q } } .
$$

Let $\delta _ { 0 } ( s _ { 0 } )$ be the uniform continuous WPI rate in (13), and put

$$
\beta _ { c } : = \operatorname* { m a x } \{ \beta _ { 0 } , c ^ { - 1 } \log ( C / s _ { 0 } ) \} .
$$

Choose the proved continuous WPI pairs

$$
\begin{array} { r } { ( \delta _ { r } , \epsilon _ { r } ) : = \left\{ \begin{array} { l l } { ( \delta _ { 0 } ( s _ { 0 } ) , \delta _ { 0 } ( s _ { 0 } ) s _ { 0 } ) , } & { \beta _ { r } < \beta _ { c } , } \\ { ( \xi / 4 , ( \xi / 4 ) C e ^ { - c \beta _ { r } } ) , } & { \beta _ { r } \geqslant \beta _ { c } . } \end{array} \right. } \end{array}
$$

The first comes from Theorem $\mathbf { A . 5 , }$ the second from Theorem $_ { \mathrm { A } . 6 . }$ . Both have $\epsilon _ { r } / \delta _ { r } \leqslant s _ { 0 } .$ , giving (63). Also $\delta _ { r } \leqslant L / 4 , \ : \mathrm { s o } \ : a _ { r } = \eta _ { r } \delta _ { r } / \kappa _ { d } \leqslant 1$ . Run the common phase lengths (64) in all blocks.

Apply Theorem C.5 to each block with $p / d$ and $\theta / d .$ The constant comparisons in Theorem C.2 give the bounds involving the invariant laws that are needed at initialization and between consecutive phases. At the final temperature, Theorem C.4 gives a stationary bias of at most $B _ { \mathrm { s t a t } } \eta _ { M } \leqslant \theta / ( 2 d )$ by the choice of $\beta _ { M }$ . Each block output has a component of mass at least $1 - p / d$ and Rényi divergence at most $\theta / d$ from its final Gibbs law. Let ν be the product of the retained block laws, rotated back to the original coordinates. Independence of the blocks, Bernoulli’s inequality and the tensorization of Rényi divergence give

$$
\alpha \geqslant ( 1 - p / d ) ^ { k } \geqslant 1 - k p / d \geqslant 1 - p , \qquad \mathcal { R } _ { q } ( \nu \| \pi _ { \beta _ { M } } ) \leqslant k \theta / d \leqslant \theta ,
$$

proving (68) since $k \leqslant d .$

Finally, M is polynomial in $\beta _ { M }$ and K is polynomial in $K _ { \mathrm { i n i t } }$ and $d M / p$ for fixed $q .$ . The uniform rate satisfies $\delta _ { 0 } ( s _ { 0 } ) ^ { - 1 } \leqslant C _ { \delta } s _ { 0 } ^ { - \gamma }$ , with $\gamma : = 1 6 ( L / m - 1 )$ . Thus the phase lengths have polynomial cost, and the overhead from the distortion in the spectral-transfer assumption is at most a constant times $\kappa _ { d } ^ { 1 + 2 \gamma }$ . Since $B _ { \mathrm { s t a t } }$ and $\eta _ { \mathrm { s t a t } }$ depend only on the uniform regularity constants and not on $d ,$ the condition on the final temperature is compatible with $\beta _ { M } = \Theta ( d / \epsilon ^ { 2 } )$ for fixed θ and $\epsilon \leqslant 1$ . If $\kappa _ { d } \leqslant C _ { \kappa } [ \log ( e d ) ] ^ { \nu _ { \kappa } }$ , this overhead is polylogarithmic in d. Every schedule parameter is determined by $d ,$ the target accuracies, and the given regularity constants, without knowledge of $k , R ,$ , or the decomposition. Each vector update queries $\breve { \nabla } f$ once and updates all blocks. Including the initial SGD update, the total number of gradient queries is $\begin{array} { r } { 1 + \sum _ { r = 1 } ^ { M } N _ { r } ; } \end{array}$ ; equivalently, the first phase contains $N _ { 1 } + 1$ updates. □

## C.3 Global Optimization

In this subsection, we show that the Gibbs measure of a block-separable function concentrates on approximate global minimizers at low temperature, and we use this concentration to prove a global optimization guarantee for annealed SGD on these functions.

Lemma C.9. Let $f : \mathbb { R } ^ { d } $ R satisfy the smoothness and quadratic-growth hypotheses of Theorem C.8, and let $f ^ { \star } : = \operatorname* { m i n } _ { x } f ( x )$ . Set

$$
\begin{array} { c } { { A : = \left( \displaystyle \frac { 2 L } m \right) ^ { b / 2 } , } } \\ { { { } } } \\ { { C _ { \mathrm { o p t } } : = \operatorname* { m a x } \left\{ 2 , \displaystyle \frac { 1 2 A } e \right\} . } } \end{array}
$$

For every $\epsilon \in ( 0 , 1 ]$ and $\beta \geqslant C _ { \mathrm { o p t } } d / \epsilon ^ { 2 }$

$$
\mathbb { P } _ { \pi _ { \beta , f } } \left[ f ( X ) - f ^ { \star } < \epsilon \right] \geqslant \frac { 5 } { 6 } .\tag{69}
$$

Proof. Translating each block minimizer to the origin and applying the Gibbs mean estimate (48) gives

$$
\mathbb { E } _ { \pi _ { \beta , f } } [ f - f ^ { \star } ] \leqslant \frac { 2 A d } { e \beta } .
$$

Markov’s inequality and the choice of $C _ { \mathrm { o p t } }$ therefore yield

$$
\pi _ { \beta , f } ( f - f ^ { \star } \geqslant \epsilon ) \leqslant \frac { 2 A d } { e \beta \epsilon } \leqslant \frac { \epsilon } { 6 } \leqslant \frac { 1 } { 6 } ,
$$

which proves (69).

Theorem C.10 (Global Optimization of Block Separable Functions Via Annealing). Let $f :  { \mathbb { R } ^ { d } } \to  { \mathbb { R } }$ satisfy the block regularity hypotheses ofTheorem C.8, with minimizer $x ^ { \star }$ . Let annealed SGD start from a specified deterministic point $x _ { 0 }$ with $\begin{array} { r } { G _ { 0 } = \operatorname* { m a x } _ { i } [ g _ { i } ( ( R x _ { 0 } ) _ { i } ) - g _ { i } ( a _ { i } ) ] } \end{array}$ ] bounded independently of d and $\epsilon ,$ and let it include the initial SGD update of Theorem $C . 7 .$ Suppose that Theorem C.2 holds with $q = 2 .$ . Set $p = 1 / 4 8 , \theta = 1 / 7 2$ , and choose $\beta _ { M } = \Theta ( d / \epsilon ^ { 2 } )$ satisfying the temperature bound in Theorem C.8. Then annealed SGD uses poly $( d / \epsilon )$ gradient queries and outputs x with $f ( x ) - f ( x ^ { \star } ) < \epsilon$ with probability at least $2 / 3$

Proof. Theorem C.8, with $\textstyle q = 2 , p = { \frac { 1 } { 4 8 } }$ and $\textstyle \theta = { \frac { 1 } { 7 2 } }$ , implies that the output distribution $\mu$ of the algorithm satisfies $\mu = \alpha \nu + ( 1 - \alpha ) \bar { \nu }$ with $\alpha \geqslant \overset { \cdot } { 1 } - 3 p = \frac { 1 5 } { 1 6 }$ and $\mathcal { R } _ { 2 } ( \nu | | \pi _ { \beta _ { M } } ) \leqslant \frac { 1 } { 7 2 }$ , where ν and ν¯ are probability measures on $\mathbb { R } ^ { d }$ and $\pi _ { \beta _ { M } }$ ∝ ex $_ { ) } ( - \beta _ { M } f )$ . By Pinsker’s inequality and the monotonicity of Rényi divergences in the order, the total variation distance satisfies $\bar { \mathrm { T V } } ( \bar { \nu _ { } } , \pi _ { \beta _ { M } } )$ ⩽ $\sqrt { \mathcal { R } _ { 2 } ( \nu | | \pi _ { \beta _ { M } } ) / 2 } \leqslant \frac { 1 } { 1 2 }$

Since $\beta _ { M } \geqslant \beta ^ { \star } = C _ { \mathrm { o p t } } d / \epsilon ^ { 2 }$ , Lemma C.9 shows that sampling from $\pi _ { \beta _ { M } }$ obtains an ϵ-approximate global optimizer with probability $\geqslant \ \frac { 5 } { 6 }$ . Consequently, $\mathbb { P } _ { \mu } [ f ( X ) - f ( x ^ { \star } ) < \epsilon ] \geqslant \alpha \mathbb { P } _ { \nu } [ f ( X ) -$ $\begin{array} { r } { f ( x ^ { \star } ) < \epsilon \rfloor \geqslant \frac { 1 5 } { 1 6 } \left( \frac { 5 } { 6 } - \frac { 1 } { 1 2 } \right) = \frac { 4 5 } { 6 4 } > \frac { 7 } { 3 } } \end{array}$ . Since $\beta _ { M } = \Theta ( d / \epsilon ^ { 2 } )$ , Theorem C.8 also shows that the number of gradient queries is polynomial in $d / \epsilon .$ . Thus the result follows. □

## D Lower Bound for SGD with Fixed Learning Rate

The recent results of Azizian et al. [4] give tight bounds on the expected time for fixed-step-size SGD to reach the global minimum, under a condition that bounds the norm of the gradient noise at every point x by a fixed multiple of $1 + \| x \| [ 4 ,$ Assumption $2 ( \mathbf { b } ) ]$ . Similar bounds can be obtained from the analysis of Shi et al. [50]; however, these are specialized to mixing times and therefore do not yield complete convergence lower bounds, which are instead governed by first hitting times.

The main lower bound is Theorem D.2. The noise in fixed-step-size SGD (Theorem 1.2) is Gaussian and therefore unbounded, so we prove Theorem D.2 by bounding the first hitting time of the ϵ-optimal set directly. For SGD on a separable objective $\begin{array} { r } { f ( \boldsymbol { x } ) = \sum _ { i = 1 } ^ { d } \phi ( \boldsymbol { x } _ { i } ) } \end{array}$ started at $( a , \ldots , a )$ , where a is a spurious local minimizer of $\phi ,$ , the first hitting time is at least the time at which a fixed coordinate leaves a neighborhood of $a .$ . Theorem D.1 shows that the expected exit time from this neighborhood is at least exponential in $1 / \eta$ . For step sizes $\eta = \mathcal { O } ( \epsilon / d )$ , as in Theorem 1.2, the resulting lower bound on the expected first hitting time is exponential in d.

Lemma D.1. Let $\phi : \mathbb { R } \to \mathbb { R } b e C ^ { 2 }$ with an L-Lipschitz derivative, and let a be a local minimizer of ϕ with $\phi ^ { \prime \prime } ( a ) > 0$ . Fix $h > 0 ,$ , let D be the connected component of $\{ y \in \mathbb { R } : \phi ( y ) < \phi ( a ) + h \}$ that contains a, and assume that D is bounded and that a is the only critical point ofϕ in the closure $\bar { D } .$ Consider the SGD recursion

$$
y _ { n + 1 } = y _ { n } - \eta \phi ^ { \prime } ( y _ { n } ) + \eta \xi _ { n } , \qquad y _ { 0 } = a ,
$$

where $\xi _ { 0 } , \xi _ { 1 } , \ldots$ . are i.i.d. $\mathcal { N } ( 0 , 1 )$ , and let $\sigma _ { D } : = \operatorname* { m i n } \{ n \geqslant 0 : y _ { n } \notin D \}$ . Then there exists $\eta _ { 0 } > 0$ depending only on $\phi ,$ a and $h ,$ , such that for all $\eta \in ( 0 , \eta _ { 0 } ] ,$

$$
\begin{array} { r } { \mathbb { E } [ \sigma _ { D } ] \geqslant \exp ( h / \eta ) . } \end{array}
$$

Proof. The proof compares the growth of the Lyapunov function $W ( y ) : = \exp \left( ( \phi ( y ) - \phi ( a ) ) / \eta \right)$ before the exit time with its value at the exit time. Note that $W ( y _ { 0 } ) = W ( a ) \dot { = } 1$ , and assume $L \eta \leqslant 1 / 2$ . We first bound the expected one-step change of $W$ from a point $y \in D$ . Let $g : = \phi ^ { \prime } ( y )$ and $y ^ { \prime } : = y - \eta g + \eta \xi$ with $\xi \sim \mathcal { N } ( 0 , 1 )$ . Since $\overline { { \phi ^ { \prime } } }$ is $\breve { L } \mathrm { - } \breve { L }$ ipschitz,

$$
\phi ( y ^ { \prime } ) - \phi ( y ) \leqslant \eta \bigl ( - g ^ { 2 } + g \xi \bigr ) + \frac { L \eta ^ { 2 } } { 2 } ( \xi - g ) ^ { 2 } ,
$$

so that $\begin{array} { r } { W ( y ^ { \prime } ) / W ( y ) \leqslant \exp \big ( - g ^ { 2 } + g \xi + \frac { L \eta } { 2 } ( \xi - g ) ^ { 2 } \big ) } \end{array}$ . For $L \eta < 1$ , a Gaussian integral gives

$$
\mathbb { E } \Big [ \exp \Big ( g \xi + \frac { L \eta } { 2 } ( \xi - g ) ^ { 2 } \Big ) \Big ] = ( 1 - L \eta ) ^ { - 1 / 2 } \exp ( g ^ { 2 } / 2 ) ,
$$

and taking expectations yields

$$
\begin{array} { r l } & { \mathbb { E } [ W ( y ^ { \prime } ) ] \leqslant ( 1 - L \eta ) ^ { - 1 / 2 } \exp ( - g ^ { 2 } / 2 ) W ( y ) } \\ & { \qquad \leqslant \exp ( L \eta - g ^ { 2 } / 2 ) W ( y ) , } \end{array}\tag{70}
$$

where the second inequality uses $L \eta \leqslant 1 / 2$

We now turn (70) into a bound on the one-step change of $W$ that is uniform over $y \in D . \operatorname { I f } g ^ { 2 } \geqslant 2 L \eta$ then (70) gives $\mathbb { E } [ W ( y ^ { \prime } ) ] \leqslant W ( y )$ .

Otherwise $g ^ { 2 } < 2 L \eta ,$ and $W ( y )$ is bounded because y is close to a. Indeed, since a is the only critical point of ϕ in D<sup>¯</sup> and $\phi ^ { \prime \prime } ( a ) > 0$ , the quotient $\phi ^ { \prime } ( y ) / ( y - a )$ , extended by $\phi ^ { \prime \prime } ( a )$ at $y = a ,$ , is continuous and positive on the compact interval $\bar { D } _ { i }$ , so

$$
c _ { D } : = \operatorname* { m i n } _ { y \in \bar { D } } \frac { \phi ^ { \prime } ( y ) } { y - a } > 0 .
$$

Hence

$$
| y - a | \leqslant | g | / c _ { D } \leqslant \sqrt { 2 L \eta } / c _ { D } .
$$

Since $\phi ^ { \prime } ( a ) = 0$ and $\phi ^ { \prime }$ is L-Lipschitz, $\phi ( y ) - \phi ( a ) \ : \leqslant \ : L ( y - a ) ^ { 2 } / 2$ , and therefore $W ( y ) \ \leqslant$ $\exp ( L ^ { 2 } / \dot { c } _ { D } ^ { 2 } )$ . Together with (70) and $e ^ { L \eta } - \mathrm { i } \leqslant 2 L \eta$ , this gives

$$
\begin{array} { r } { \mathbb { E } [ W ( y ^ { \prime } ) ] - W ( y ) \leqslant 2 L \eta \exp ( L ^ { 2 } / c _ { D } ^ { 2 } ) . } \end{array}
$$

In both cases,

$$
\begin{array} { c } { { \mathbb { E } [ W ( y ^ { \prime } ) ] - W ( y ) \leqslant C _ { 1 } \eta \quad \mathrm { f o r ~ a l l } \ y \in D , } } \\ { { C _ { 1 } : = 2 L \exp ( L ^ { 2 } / c _ { D } ^ { 2 } ) . } } \end{array}
$$

For $n < \sigma _ { D }$ we have $y _ { n } \in D$ , and $\xi _ { n }$ is independent of $y _ { 0 } , \ldots , y _ { n }$ . Applying the bound $\mathbb { E } [ W ( y ^ { \prime } ) ] -$ $W ( y ) \leqslant C _ { 1 } \eta$ with $y = y _ { n }$ and $\xi = \xi _ { n }$ therefore shows that the stopped process

$$
W ( y _ { n \wedge \sigma _ { D } } ) - C _ { 1 } \eta \left( n \wedge \sigma _ { D } \right)
$$

is a supermartingale. As $W ( y _ { 0 } ) = 1$ , it follows that

$$
\mathbb { E } \big [ W ( y _ { n \wedge \sigma _ { D } } ) \big ] \leqslant 1 + C _ { 1 } \eta n \qquad \mathrm { f o r ~ a l l } \ n \geqslant 0 .\tag{71}
$$

We next show that $W ( y _ { \sigma _ { D } } ) \geqslant \exp ( h / \eta )$ unless a large noise value occurs before the exit. Write $D = ( u _ { - } , u _ { + } )$ . The endpoints satisfy $\dot { \phi ( u _ { \pm } ) } = \phi ( a ) + h ,$ and $\phi ^ { \prime } ( u _ { \pm } ) \neq 0$ because a is the only critical point of ϕ in D<sup>¯</sup> . As $\phi < \phi ( a ) + h$ on D, this gives $\phi ^ { \prime } ( u _ { + } ) > 0$ and $\phi ^ { \prime } ( u _ { - } ) < 0$ , so there is $r > 0$ such that $\phi \geqslant \phi ( a ) + h$ on $[ u _ { + } , u _ { + } + r ]$ and on $[ u _ { - } - r , u _ { - } ]$

To bound the length of a single step, let $G : = \operatorname* { m a x } _ { y \in \bar { D } } | \phi ^ { \prime } ( y ) |$ and assume in addition that $\eta \leqslant$ r/(2G). If $y _ { k } \in D$ and $| \xi _ { k } | < r / ( 2 \eta )$ , then

$$
| y _ { k + 1 } - y _ { k } | \leqslant \eta G + \eta | \xi _ { k } | < r ,
$$

so $y _ { k + 1 } \notin D$ implies $\phi ( y _ { k + 1 } ) \geqslant \phi ( a ) + h$ , that is, $W ( y _ { k + 1 } ) \geqslant \exp ( h / \eta )$

To control the noise at the last step before the exit, let $E _ { n }$ be the event that $| \xi _ { k } | < r / ( 2 \eta )$ for all $k < n$ On $E _ { n } \cap \{ \sigma _ { D } \leqslant n \}$ , the last step before the exit starts at $y _ { \sigma _ { D } - 1 } \in D$ and satisfies $| \dot { \xi } _ { \sigma _ { D } - 1 } | < r / ( 2 \eta )$ so

$$
W ( y _ { n \wedge \sigma _ { D } } ) = W ( y _ { \sigma _ { D } } ) \geqslant \exp ( h / \eta ) .
$$

Combining the bound $W ( y _ { n \wedge \sigma _ { D } } ) \geqslant \exp ( h / \eta )$ on $E _ { n } \cap \{ \sigma _ { D } \leqslant n \}$ with (71) controls the probability that $\sigma _ { D } ~ \leqslant ~ n$ . Markov’s inequality, a union bound over $k \ < \ n$ , and the Gaussian tail bound $\mathbb { P } ( | \xi _ { k } | \geqslant t ) \leqslant 2 e ^ { - t ^ { 2 } / 2 }$ give

$$
\begin{array} { r l } & { \mathbb { P } ( \sigma _ { D } \leqslant n ) \leqslant \mathbb { P } \big ( W ( y _ { n \wedge \sigma _ { D } } ) \geqslant e ^ { h / \eta } \big ) + \mathbb { P } ( E _ { n } ^ { c } ) } \\ & { \qquad \leqslant \left( 1 + C _ { 1 } \eta n \right) e ^ { - h / \eta } + 2 n e ^ { - r ^ { 2 } / ( 8 \eta ^ { 2 } ) } . } \end{array}
$$

Take $n = \lceil 2 \exp ( h / \eta ) \rceil$ . Both terms on the right-hand side then tend to 0 as $\eta  0 .$ , so there is $\eta _ { 0 } \in ( 0$ , min $\{ 1 \bar { / } ( 2 \bar { L } ) , \bar { r } / ( 2 G ) \}$ ], depending only on $\phi ,$ a and $h ,$ such that $\mathbb { P } ( \sigma _ { D } \leqslant n ) \leqslant 1 / 2$ for all $\eta \in ( 0 , \eta _ { 0 } ]$ . For these $\eta ,$

$$
\mathbb { E } [ \sigma _ { D } ] \geqslant n \mathbb { P } ( \sigma _ { D } > n ) \geqslant \frac { n } { 2 } \geqslant \exp ( h / \eta ) .
$$

We now apply Theorem D.1 to one coordinate of fixed-step-size SGD on a separable objective.

Theorem D.2 (Lower bound for fixed-step-size SGD). Let $\textstyle f ( x ) = \sum _ { i = 1 } ^ { d } \phi ( x _ { i } )$ where $\phi : \mathbb { R } $ R is a coercive $C ^ { 2 }$ Morsefunction with an L-Lipschitz derivative and exactly three critical points: a unique global minimizer $x ^ { * }$ satisfying $\phi ( x ^ { * } ) = 0$ and $\phi ^ { \prime \prime } ( x ^ { * } ) = \omega > 0 ,$ , a spurious local minimizer a with $\phi ( a ) = c > 0$ , and a local maximizer s between a and $x ^ { * }$ with barrier height $\Delta : = \phi ( s ) - \phi ( a ) > 0$ $F i x \in \mathsf { \Gamma } ( 0 , c )$ and $C > 0 .$ . Consider fixed-step-size SGD (Theorem $I . 2 )$ with any learning rate $\eta \in ( 0 , C \epsilon / d ]$ , initialized at $x _ { 0 } = ( a , \ldots , a )$ . Then for $\tau = \operatorname* { i n f } \{ t \geqslant 0 : f ( x _ { t } ) - f ^ { * } < \epsilon \}$ , there exists $d _ { 0 } ,$ , depending only on $\phi ,$ ϵ and $C ,$ , such that for all d $\geqslant d _ { 0 }$ ,

$$
\mathbb { E } [ \tau ] \geqslant \exp \left( \frac { \Delta d } { 2 C \epsilon } \right) = \exp ( \Omega ( d ) ) .
$$

Remark D.3. The step-size regime $\eta \leqslant C \epsilon / d$ of Theorem $D . 2$ is the regime $\eta ~ = ~ \mathcal { O } ( \epsilon / d )$ of Theorem $1 . 2 .$ . The regime $\eta = \mathcal { O } ( \epsilon / d )$ can be motivated by the quadratic approximation $o f \phi$ near the global minimizer. Within the basin $o f x ^ { * }$ , the per-coordinate SGD dynamics linearize as $z _ { t + 1 } = ( 1 - \eta \omega ) z _ { t } + \eta \xi _ { t , i } ,$ , where $z _ { t } = x _ { t , i } - x ^ { * }$ and $\xi _ { t , i }$ is the i-th coordinate of $\cdot \xi _ { t }$ . This recursion is exactfor the quadratic $\phi ( x ) = \omega ( x - x ^ { * } ) ^ { 2 } / 2$ . In this quadratic model, $z _ { t }$ is an autoregressive process oforder $^ { l , }$ which is stable $f o r \eta < 2 / \omega$ and has stationary variance

$$
\sigma _ { \mathrm { s t a t } } ^ { 2 } = \frac { \eta } { 2 \omega ( 1 - \eta \omega / 2 ) } .
$$

The expected suboptimality ofa point X drawnfrom the stationary measure ofthe quadratic model is therefore

$$
\mathbb { E } [ f ( X ) - f ^ { * } ] = \sum _ { i = 1 } ^ { d } \frac { \omega } { 2 } \sigma _ { \mathrm { s t a t } } ^ { 2 } = \frac { d \eta } { 4 ( 1 - \eta \omega / 2 ) } ,
$$

which is of order ϵ when $\eta$ is of order $\epsilon / d .$

ProofofTheorem D.2. By separability, the coordinates of $x _ { t }$ evolve independently, and the first coordinate $y _ { t } : = x _ { t , 1 }$ follows the recursion

$$
y _ { t + 1 } = y _ { t } - \eta \phi ^ { \prime } ( y _ { t } ) + \eta \xi _ { t , 1 } , \qquad y _ { 0 } = a ,
$$

where the first coordinates $\xi _ { 0 , 1 } , \xi _ { 1 , 1 } , \ldots$ . of the noise vectors are i.i.d. $\mathcal { N } ( 0 , 1 )$ . This is the recursion of Theorem D.1. Since $f ^ { * } = d \phi ( x ^ { * } ) = 0$ and all terms of $\begin{array} { r } { f ( \boldsymbol { x } ) - f ^ { * } = \sum _ { i = 1 } ^ { d } \left( \phi ( \boldsymbol { x } _ { i } ) - \phi ( \boldsymbol { x } ^ { * } ) \right) } \end{array}$ are nonnegative, every ϵ-optimal point x satisfies $\phi ( x _ { i } ) - \phi ( x ^ { * } ) <$ ϵ for all i. In particular, $f ( x _ { t } ) - f ^ { * } < \epsilon$ implies $\phi ( y _ { t } ) < \epsilon$

We next find a neighborhood of a on which $\phi \geqslant c .$ Let $D$ be the connected component of $\{ y \in \mathbb { R }$ $\phi ( y ) < \phi ( a ) + \Delta \bar { / } 2 \}$ that contains $^ { a , }$ and let D<sup>¯</sup> be its closure. Because $\phi$ is coercive, D is a bounded interval. On D<sup>¯</sup> we have $\phi \leqslant \phi ( a ) + \Delta / 2 < \phi ( s )$ , so D<sup>¯</sup> does not contain $s .$ . As s lies between a and $x ^ { * }$ , the interval D<sup>¯</sup> does not contain $x ^ { * }$ either. Hence a is the only critical point of $\phi$ in $\bar { D } ,$ and since $\phi = \phi ( a ) + \Delta / 2$ at the endpoints of $D ,$ the minimum of ϕ over $\bar { D }$ is $\phi ( a ) = c$ . Thus $\phi \geqslant c > \epsilon$ on $D ,$ so $x _ { t }$ is not ϵ-optimal for $t < \sigma _ { D } : = \operatorname* { m i n } \{ t \geqslant 0 : y _ { t } \notin D \}$ , and $\tau \geqslant \sigma _ { D }$

Since ϕ is a Morse function, $\phi ^ { \prime \prime } ( a ) > 0$ . Hence Theorem D.1 applies to D with $h = \Delta / 2$ and gives $\eta _ { 0 } > 0 ;$ , depending only on ϕ and not on $d ,$ such that $\begin{array} { r } { { \mathbb E } [ \sigma _ { D } ] \stackrel { \sim } { \geqslant } \exp ( \Delta / ( 2 \eta ) ) , } \end{array}$ ) for all $\eta \in ( 0 , \eta _ { 0 } ]$ Since $\eta \leqslant C \epsilon / d \to 0$ as $d \to \infty$ , the condition $\eta \leqslant \eta _ { 0 }$ holds for all sufficiently large $d ,$ namely for $d \geqslant d _ { 0 } : = \lceil C \epsilon / \eta _ { 0 } \rceil$ . For such $d ,$

$$
\mathbb { E } [ \tau ] \geqslant \mathbb { E } [ \sigma _ { D } ] \geqslant \exp \left( \frac { \Delta } { 2 \eta } \right) \geqslant \exp \left( \frac { \Delta d } { 2 C \epsilon } \right) = \exp ( \Omega ( d ) ) .
$$

Corollary D.4 (Separation from the same starting point). Let ϕ satisfy the hypotheses of Theorem D.2, and suppose additionally that

$$
\phi ( y ) \geqslant { \frac { m } { 2 } } | y - x ^ { * } | ^ { 2 } \qquad ( y \in \mathbb { R } )
$$

for a fixed $m > 0$ and that Theorem C.2 holds for its SGD kernels with $q \ : = \ : 2$ and constants independent ofd. For an arbitrary orthogonal matrix R, define

$$
f _ { d } ( \boldsymbol { x } ) : = \sum _ { i = 1 } ^ { d } \phi ( ( R \boldsymbol { x } ) _ { i } ) , \qquad \boldsymbol { x } _ { 0 } : = R ^ { \top } ( a , \ldots , a ) .
$$

Fix $0 < \epsilon \leqslant 1$ with $\epsilon < c ,$ , and fix $C _ { \mathrm { l r } } > 0 .$ . Both algorithms receive the same point $x _ { 0 }$ as input. For all sufficiently large d:

1. Fixed-step-size SGD with any $0 < \eta \leqslant C _ { \mathrm { l r } } \epsilon / d$ has an expected first hitting time of the ϵ-optimal set ofat least

$$
\exp \left( { \frac { \Delta d } { 2 C _ { \mathrm { l r } } \epsilon } } \right) .
$$

2. Annealed SGD, including its first update at the constant learning rate η , uses $\mathrm { p o l y } ( d / \epsilon )$ gradient queries and outputs an ϵ-optimal point with probability at least $2 / 3 .$

Proof. The rotation $x \mapsto$ Rx leaves SGD invariant, since RZ is standard Gaussian whenever $Z$ is standard Gaussian. It also preserves objective values and maps the starting point $x _ { 0 } \tan \left( a , \ldots , a \right)$ Thus Theorem D.2 gives the lower bound for fixed-step-size SGD.

For the upper bound, continuity of $\phi ^ { \prime \prime }$ and $\phi ^ { \prime \prime } ( x ^ { * } ) > 0$ give a radius $r _ { 0 } > 0$ such that $\phi ^ { \prime \prime } ( y ) \geqslant \xi : =$ $\phi ^ { \prime \prime } ( x ^ { * } ) / \bar { 2 }$ whenever $| y - x ^ { * } | \leqslant r _ { 0 }$ . Since every block is the same fixed potential $\phi ,$ this radius and curvature are independent of d and satisfy (66). The repeated blocks $g _ { i } = \phi$ therefore have dimension one, fixed regularity constants, and initial energies

$$
g _ { i } ( a ) - g _ { i } ( x ^ { * } ) = c , \qquad G _ { 0 } = c .
$$

The density bound after the first SGD update is therefore independent of d and ϵ by Theorem $\mathbf { C . 7 }$ Applying Theorem C.10 gives the claimed success probability and polynomial query count from the same point $x _ { 0 }$ □

## E Additional Details for Numerical Results

Common setup. The experiments in Figure 3 and Table 1 use SGD with independent standard Gaussian noise:

$$
x _ { k + 1 } = x _ { k } - \eta _ { k } \nabla f ( x _ { k } ) + \eta _ { k } Z _ { k } , \qquad Z _ { k } \sim \mathcal { N } ( 0 , I _ { d } ) .
$$

All runs start from $X _ { 0 } = c _ { f } \mathbf { 1 } _ { d } + 0 . 0 5 \xi ,$ where $\xi \sim \mathcal { N } ( 0 , I _ { d } ) , c _ { f } \simeq 0 . 9 6 0 \overset { \cdot } { _ { \textrm { \scriptsize { C } } } }$ 1496 for both double wells and $c _ { f } \simeq 0 . 9 5 1 0 5 0 4$ for Rastrigin. After each update, we check whether $\| x _ { k } - x ^ { \star } \| _ { \infty } < \delta .$ , where $\delta = 0 . 5$ for the double-well functions and $\delta = 0 . 3$ for Rastrigin. We stop the run as soon as this condition is satisfied and record the number of updates performed.

Arithmetic annealing. An annealing schedule specifies M learning-rate levels:

$$
\eta _ { j } = \eta _ { 0 } + { \frac { j } { M - 1 } } ( \eta _ { \mathrm { e n d } } - \eta _ { 0 } ) , \qquad j = 0 , \ldots , M - 1 .
$$

Normally each level is held constant for $N _ { p }$ updates before decreasing to the next.

Low-dimensional experiments (Figure 3). For the low-dimensional grid $d \in$ {1, 5, 10, 15, 20, 25, 30, 35}, we use ten trials per method and dimension. The annealed settings are:

• Separable double well: $\eta _ { \perp } = 0 . 1 8 / ( 1 + 0 . 0 8 \log d ) , \eta _ { \mathrm { e n d } } = 0 . 1 4 5 / [ 1 + ( d - 1 ) / 4 0 ] , M =$ max(3d, 15), and $N _ { p } = 1 0 ^ { 5 }$

• Rastrigin: $\eta _ { 0 } = 0 . 1 0 , \eta _ { \mathrm { e n d } } = 0 . 0 6 4 / [ 1 + ( d - 1 ) / 2 0 0 ] , M = \operatorname* { m a x } ( 3 d , 1 5 ) , \mathrm { a n d } N _ { p } = 2 0$

• Coupled double well: $\eta _ { 0 } = 0 . 1 5 5 , \eta _ { \mathrm { e n d } } = 0 . 1 5 0 / [ 1 + ( d - 1 ) / 1 4 0 ] , M = \operatorname* { m a x } ( 3 d , 2 0 )$ , and $N _ { p } = 1 0 0$

The Rastrigin schedule allows max(2000, 1000d) additional updates at the last annealing level with $\eta _ { j } \geqslant 0 . 0 6 5$ and $\mathrm { 2 0 } \lceil \operatorname* { m a x } ( 1 0 0 0 , 5 0 d ) / \mathrm { 2 0 } \rceil$ additional updates at the final rate, if the stopping criterion has not yet been met. Coupled double well allows max(50000, 5000d) additional updates at the last level with $\eta _ { j } \geqslant 0 . 1 5$ and $\dot { 2 } \times 1 0 ^ { 6 }$ additional terminal-rate updates. These allowances are used only while the stopping criterion remains unmet.

Fixed-step-size SGD keeps the corresponding terminal rate throughout the run. These terminal-rate rules were selected in pilot experiments and then locked; fixed-step-size SGD was not independently optimized for the fastest constant rate at each dimension. The cap for fixed-step-size SGD runs is $1 0 ^ { \mathbf { \bar { 9 } } }$ updates except for the separable double well at d = 30, 35 $( 2 \times \mathrm { 1 0 ^ { 1 0 } }$ and $5 \times 1 0 ^ { 1 0 } )$ ) and the coupled double well at $d = 3 5 ( 1 { \dot { 0 } } ^ { 1 0 } )$

High-dimensional experiments (Table 1). For the high-dimensional grid $d \in { \mathbf { \Gamma } }$ {100, 200, 300, 400}, we run ten annealed trials per function and dimension using the same learning-rate formulas, phase lengths, and additional holds as above, with $M = 3 d .$

Reporting and run outcomes. The table reports medians ± one empirical standard deviation computed with divisor 10 over all ten recorded update counts and run times, with update counts truncated at the limits recorded with the data. Timing starts after the initial state is generated and excludes JIT warmup and queueing. Medians and interquartile ranges use all ten planned trials when identified; groups containing nonfinite outcomes are excluded from fitting.