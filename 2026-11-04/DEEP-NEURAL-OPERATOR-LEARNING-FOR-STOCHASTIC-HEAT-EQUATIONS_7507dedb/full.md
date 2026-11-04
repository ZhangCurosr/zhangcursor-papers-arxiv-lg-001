# DEEP NEURAL OPERATOR LEARNING FOR STOCHASTIC HEAT EQUATIONS AND APPLICATIONS

HENGRONG DU, QI FENG, AND WEINAN WANG

Abstract. In this paper, we establish a universal approximation theorem for solution operators of semilinear stochastic heat equations in spatial dimension d ≥ 2, extending deep neural operator approximation results to SPDEs with distribution-valued Gaussian forcing. A smooth Fourier cutof reduces the forcing to finitely many measurements, while Besov and Schauder estimates control the resulting solution error. The finite-dimensional solution map is then approximated by ReLU branch networks together with grid-based trunk functions, yielding explicit bounds on the numbers of forcing measurements and trunk functions and approximation in every finite stochastic moment. For linear equations, the reduction coincides with truncation of the first Wiener chaos. Lastly, as an application, we prove a high-probability local-in-time approximation theorem for stochastic Navier–Stokes equations, with finite measurements of both the initial condition and forcing and an explicit sensing rate.

## 1. Introduction

Operator learning seeks a single approximation of the map from the data of an equation to its solution, enabling evaluation at new inputs without solving each instance input. The operator approximation theorems in [9], DeepONet [31], and Fourier neural operators [27, 20] have established such approximation for maps between function spaces. For stochastic partial diferential equations (SPDEs), we study solution operators that map initial data and random forcing to the corresponding solutions. The central challenge is that the forcing may be distribution-valued, requiring an operator approximation framework that accommodates inputs without pointwise values.

This paper develops a finite-measurement approximation theory for such SPDE solution operators. Our starting point is the probabilistic operator framework in [5], which combines integrability, tail estimates, and Lipschitz continuity for models driven by finite-dimensional stochastic processes. Extending this approach to distribution-valued space–time forcing requires an encoding that is well defined on the noise space and an estimate that propagates its error through the equation. We use Fourier measurements to construct the encoding and Schauder stability to control the resulting error. The resulting branch–trunk network approximates the joint dependence on the initial condition and the forcing in stochastic moments of the uniform space–time norm, with explicit measurement and reconstruction rates.

We consider the semilinear stochastic heat equation

$$
\begin{array} { r } { \left\{ \begin{array} { l l } { \partial _ { t } u - \Delta u = F ( u ) + \eta , } & { ( t , x ) \in ( 0 , T ] \times  { \mathbb { T } } ^ { d } , } \\ { u ( 0 , x ) = u _ { 0 } ( x ) , } & { x \in  { \mathbb { T } } ^ { d } , } \end{array} \right. } \end{array}\tag{1}
$$

where $d \geq 2 , F : \mathbb { R } \to \mathbb { R }$ is globally Lipschitz, $u _ { 0 } \in C (  { \mathbb { T } } ^ { d } )$ , and η is a centered Gaussian random distribution (as defined in Assumption 2.2 below). The equation is interpreted pathwisely in the mild sense. For this SPDE, our main result constructs ReLU networks of the form

$$
\Gamma _ { \theta } ( u _ { 0 } , \eta ) ( t , x ) = \sum _ { \ell = 1 } ^ { N _ { 1 } } a _ { \ell } \big ( \vec { u } _ { 0 } , c _ { J } ( \eta ) \big ) q _ { \ell } ( t , x ) ,
$$

where ${ \vec { u } } _ { 0 }$ contains $N _ { 0 }$ point values of the initial condition and $c _ { J } ( \eta )$ contains $N _ { J }$ real continuous linear measurements of the forcing. For every fixed $q \in [ 1 , \infty )$ and $\varepsilon \in ( 0 , 1 )$ , one such network satisfies

$$
\operatorname* { s u p } _ { u _ { 0 } \in \mathcal { G } _ { 0 } } \Big ( \mathbb { E } \Big [ \| S ( u _ { 0 } , \eta ) - \Gamma _ { \theta } ( u _ { 0 } , \eta ) \| _ { C ( D _ { T } ) } ^ { q } \Big ] \Big ) ^ { 1 / q } \leq \varepsilon .
$$

Thus the approximation controls full solution paths, uniformly over the initial-data class, while retaining the prescribed coupling between the noise input and the solution output. The main analytic step is a quantitative reduction of the rough/noisy input. A smooth finite-rank Fourier cutof satisfies

$$
\| ( I - \Pi _ { J } ) \eta \| _ { \mathcal { C } ^ { \beta _ { 1 } } } \lesssim 2 ^ { - J \delta } \| \eta \| _ { \mathcal { C } ^ { \beta _ { 0 } } } .
$$

The main dificulty is to turn this finite-dimensional reduction into an approximation of the SPDE solution operator. The forcing may not have pointwise values, and its truncation error must be measured in a topology in which the solution map is stable. We combine the stability estimate for (1) with partition-of-unity approximation of the finite-dimensional input dependence and parabolic interpolation of the output. This extends the function, functional, and operator approximation approach in [28, 5] to distribution-valued forcing, with explicit bounds on the numbers of input measurements and trunk functions.

Another dificulty is that the Gaussian input is not confined to a compact set. Therefore Uniform approximation on bounded sets alone does not yield the stochastic moment estimate above. We control the growth of the neural approximation outside these sets by a coordinatewise projection, and use Fernique’s theorem to estimate the remaining Gaussian tail. The construction consequently gives approximation in every fixed finite stochastic moment without imposing compact support on the noise law.

For stochastic incompressible Navier–Stokes equations, the quadratic transport term contains a spatial derivative, and the solution map is only locally Lipschitz. We establish a common deterministic existence interval for the admissible initial velocities and forcings in bounded sets, together with their Fourier truncations. On this interval, which is independent of the approximation accuracy, local stability yields uniform operator approximation on an event of high probability. The measurement bounds distinguish the regularity required of the initial velocity from that of the forcing.

1.1. Related literature. The deterministic operator-learning literature supplies the approximation architecture used below. Alongside DeepONet, the Fourier neural operator in [27] and the general neural-operator framework in [20] provide resolution-independent parameterizations of maps between function spaces. Universal approximation and error estimates for Fourier neural operators are proved in [19]; the encoding–approximation–reconstruction analysis of DeepONets in [23] is particularly close to the error decomposition used here. Random-feature operator learning [35] gives a probabilistic alternative , and its statistical error analysis extends to infinitedimensional, vector-valued regression [24]. Learning linear operators from noisy input–output pairs is studied in [12]. These general results also show why quantitative branch-network bounds are dificult without additional structure: a generic Lipschitz operator on a high-dimensional encoded input class retains the curse of dimensionality.

Operator learning has also been extended to geometry-dependent PDE solution maps. The DIMON framework [44] transports problems on a family of difeomorphic domains to a common reference domain, where the dependence on the geometry and initial or boundary data is learned.

Finite-dimensional inputs and outputs also arise only when parameters or observables of an underlying PDE solution are available. Fourier neural mappings for such parameter-to-observable maps are developed in [18]. In the inverse direction, neural inverse operators combine DeepONet and Fourier-neural-operator components to learn PDE inverse maps [34]; see [36] for a broader probabilistic account of operator learning for inverse problems. Those works concern statistical recovery from observations, whereas the present paper constructs a pathwise approximation of a forward SPDE solution map after encoding its distribution-valued forcing.

Operator learning for stochastic models has applications in mathematical finance, control, and game theory. Solution operators for structured families of forward–backward stochastic diferential equations with random terminal time are approximated in [13]. In dynamic Stackelberg games, attention-based neural operators approximate the follower’s best-response operator [1]. Neural operators for families of convex optimization problems are developed in [21], with applications to stochastic optimal control, portfolio optimization, and hedging. These operator formulations complement methods such as the Deep-BSDE method [16], which approximates the solution of a fixed equation. For optimal execution problems with unknown price impact, the authors of [32] combine in-context learning of the price-impact operator from observed trade and impact trajectories with neural-network optimization of the execution policy. Another interesting incontext operator learning on spaces of probability measures is studied in [11], using samples from source and target distributions to approximate optimal transport maps.

The authors of [22] study the recovery of a discrete-time linear transition operator from sparse observations of its powers. In [30], the authors establish consistency and learning rates for interaction kernels inferred from multiple trajectories of stochastic particle systems. These works concern inference of the governing dynamics, whereas one can fix the SDE and approximate its solution operator as a function of the initial condition and distribution-valued forcing. In [3], the authors apply separate adaptive optimizers to the individual loss terms before combining their updates. Curvature-aware corrections to first-order optimizers are developed in [2], while a regularized quasi-Newton method for high-accuracy training is proposed in [41]. These works address training algorithms; our results concern the existence and size of approximating networks.

Several architectures have been designed specifically for stochastic PDEs. Neural SPDEs [38] build the mild-solution structure into a resolution-invariant model. Feature engineering with regularity structures [10], NORS [17], and the deep latent regularity network [14] use polynomial or renormalized features of the driving noise. Such features are indispensable for singular equations in which the solution and its nonlinearities are not classically defined. SPDEBench [26] provides a numerical comparison across regular and singular stochastic equations. The present paper addresses a diferent question: for additive Gaussian forcing whose heat convolution is continuous, are linear measurements of the distributional input already suficient for a pathwise operator-approximation theorem? The answer is afirmative under Assumption 2.2.

Navier–Stokes solution operators have also been used as important test cases for neural operators. Quantitative approximation results for deterministic incompressible Navier–Stokes appear in [19], while Neural SPDEs [38] and NORS [17] include stochastic Navier–Stokes experiments. In [43], the authors introduced a conformal prediction approach from statistics to study the operator learning of deterministic Navier-Stokes equations. Our result is complementary to these works: We prove that finitely many linear measurements of the forcing and initial velocity sufice for uniform approximation on a common local existence interval, on an event of high probability. The measurement counts in Theorem 5.8 separate the forcing regularity from the initial velocity regularity.

The finite measurements also connect the construction to Wiener chaos. Classical chaos solutions of stochastic evolution equations are developed in [33, 29]; recent work learns the deterministic propagators in truncated chaos expansions [37, 39, 40]. The precise relation is recorded in Lemma 3.11: for the linear equation, the finite-rank solution used here is exactly the truncation of the first-chaos expansion to the retained coordinates. For a general globally Lipschitz nonlinearity, the present proof instead uses Schauder stability and does not require a closed propagator hierarchy. Our approximation argument follows the function, functional, and operator approximation results of [5]. Deep-BSDE methods target the solution of a fixed equation, whereas the present branch–trunk network approximates the joint dependence of the solution on the initial condition and the forcing realization.

The paper is organized as follows. Section 2 states preliminaries, the analytic assumptions, and proves well-posedness and Lipschitz stability. Section 3 constructs the finite-dimensional measurement map and the deep operator architecture. Section 4 proves the function, functional, and operator approximation results. Section 5 develops the high-probability local extension to stochastic Navier–Stokes equations.

## 2. Preliminaries

In this section, we specify the function spaces and give some preliminaries.

## 2.1. Fourier multipliers and function spaces. Fix $T > 0$ and write

$$
\mathbb { T } ^ { d } = ( \mathbb { R } / 2 \pi \mathbb { Z } ) ^ { d } , \qquad \mathbb { T } _ { T } = \mathbb { R } / T \mathbb { Z } , \qquad \Lambda _ { T } = \mathbb { T } _ { T } \times \mathbb { T } ^ { d } , \qquad D _ { T } = [ 0 , T ] \times \mathbb { T } ^ { d } .
$$

The forcing is realized on the space–time torus $\Lambda _ { T }$ and then restricted to the causal equation on $D _ { T }$ . The solution is not required to be periodic in time. On $D _ { T }$ we use the parabolic distance

$$
d _ { \mathfrak { s } } ( ( t , x ) , ( s , y ) ) : = | t - s | ^ { 1 / 2 } + d _ { \mathbb { T } ^ { d } } ( x , y ) , \qquad d _ { \mathbb { T } ^ { d } } ( x , y ) : = \operatorname* { m i n } _ { m \in \mathbb { Z } ^ { d } } | x - y - 2 \pi m | .
$$

This distance corresponds to the scaling ${ \mathfrak { s } } = ( 2 , 1 , \dots , 1 )$ , whose dimension is $| \mathfrak { s } | = d + 2$ . For $0 < \gamma < 1$ , we write

$$
\| f \| _ { C _ { s } ^ { \gamma } ( D _ { T } ) } : = \| f \| _ { \infty } + \operatorname* { s u p } _ { z \neq z ^ { \prime } } \frac { | f ( z ) - f ( z ^ { \prime } ) | } { d _ { \mathfrak { s } } ( z , z ^ { \prime } ) ^ { \gamma } } .
$$

All function-space norms below are deterministic norms. Expectations are taken separately when these norms are applied to random functions. For vector-valued functions, we take the maximum over components.

The orthonormal Fourier basis on $\Lambda _ { T }$ are

$$
\varepsilon _ { n , k } ( t , x ) : = \bigl ( T ( 2 \pi ) ^ { d } \bigr ) ^ { - 1 / 2 } e ^ { 2 \pi i n t / T + i k \cdot x } , \qquad ( n , k ) \in  { \mathbb { Z } } \times  { \mathbb { Z } ^ { d } } .
$$

For a distribution $f ,$ its Fourier coeficient ${ \widehat { f } } ( n , k )$ is its action on ${ \overline { { \varepsilon _ { n , k } } } } .$ , i.e., ${ \widehat { f } } ( n , k ) = \langle f , { \overline { { \varepsilon _ { n , k } } } } \rangle$ The corresponding spatial characters are $e _ { k } ( x ) = ( 2 \pi ) ^ { - d / 2 } e ^ { i k \cdot x }$

Now we fix a smooth dyadic decomposition compatible with parabolic scaling: choose a nonincreasing function $\varsigma \in C ^ { \infty } ( [ 0 , \infty ) ; [ 0 , 1 ] )$ that equals one on [0, 1] and vanishes on $[ 4 / 3 , \infty )$ Set

$$
\varrho ( \omega , \kappa ) : = ( \omega ^ { 2 } + \vert \kappa \vert ^ { 4 } ) ^ { 1 / 4 } , \qquad \chi ( \omega , \kappa ) : = \varsigma ( \varrho ( \omega , \kappa ) ) .
$$

Although ϱ is not smooth at the origin, χ is smooth there because it is constant in a neighborhood of the origin. Moreover, $\varrho ( \lambda ^ { 2 } \omega , \lambda \kappa ) = \lambda \varrho ( \omega , \kappa )$ and $\varrho ( \omega , \kappa ) \asymp | \omega | ^ { 1 / 2 } + | \kappa |$ . With $\delta _ { \lambda } ( \omega , \kappa ) =$ $( \lambda ^ { 2 } \omega , \lambda \kappa )$ , we define

$$
\chi _ { - 1 } : = \chi , \qquad \chi _ { j } : = \chi \circ \delta _ { 2 ^ { - j - 1 } } - \chi \circ \delta _ { 2 ^ { - j } } , \quad j \geq 0 .
$$

These symbols form a partition of unity in frequency. For $j ~ \geq ~ 0 , ~ \chi _ { j }$ is supported where $2 ^ { j } \le \varrho \le 2 ^ { j + 3 } / 3$ , so only a fixed number of neighboring blocks can overlap. Define

$$
\begin{array} { l } { { \displaystyle \Delta _ { j } ^ { \mathfrak { s } } f : = \sum _ { n , k } \chi _ { j } ( n , k ) \widehat { f } ( n , k ) \varepsilon _ { n , k } , } } \\ { { \displaystyle \Pi _ { J } : = \sum _ { j = - 1 } ^ { J } \Delta _ { j } ^ { \mathfrak { s } } , \qquad m _ { J } ( n , k ) : = \varsigma ( 2 ^ { - ( J + 1 ) } \varrho ( n , k ) ) . } } \end{array}
$$

Thus $\Pi _ { J }$ has the smooth symbol $m _ { J } , 0 \le m _ { J } \le m _ { J + 1 } \le 1$ , and $m _ { J }  1$ pointwise. The cutof retains temporal frequencies of order $2 ^ { 2 J }$ and spatial frequencies of order $2 ^ { J }$

Definition 2.1 (Parabolic H¨older–Besov spaces). For $\beta \in \mathbb { R }$ , let

$$
\mathcal { C } ^ { \beta } ( \Lambda _ { T } ) : = B _ { \infty , \infty } ^ { \beta , \mathfrak { s } } ( \Lambda _ { T } )
$$

be the inhomogeneous Besov space associated with the Littlewood–Paley decomposition $( \Delta _ { j } ^ { \mathfrak { s } } ) _ { j \geq - 1 }$ ， equipped with the norm

$$
\| f \| _ { \mathcal { C } ^ { \beta } ( \Lambda _ { T } ) } : = \operatorname* { s u p } _ { j \geq - 1 } 2 ^ { j \beta } \bigl \| \Delta _ { j } ^ { \mathfrak { s } } f \bigr \| _ { L ^ { \infty } ( \Lambda _ { T } ) } .
$$

We denote by $\mathcal { C } _ { g } ^ { \beta } ( \Lambda _ { T } )$ the closure of $C ^ { \infty } ( \Lambda _ { T } )$ in this norm. $O n \mathbb { R } \times \mathbb { T } ^ { d }$ , the same decomposition, with continuous temporal frequency, defines $\mathcal { C } ^ { \beta } ( \mathbb { R } \times \mathbb { T } ^ { d } )$ . Let $\widetilde { f } \in \mathcal { C } ^ { \beta } ( \mathbb { R } \times \mathbb { T } ^ { d } )$ denote an extension of $\dot { \boldsymbol { f } }$ to $\mathbb { R } \times \mathbb { T } ^ { d }$ . On the cylinder $D _ { T }$ , we define

$$
\| f \| _ { \mathcal { C } ^ { \beta } ( \mathbb { D } _ { T } ) } : = \operatorname* { i n f } _ { \widetilde { f } \in \mathcal { C } ^ { \beta } ( \mathbb { R } \times \mathbb { T } ^ { d } ) \atop \widetilde { f } | _ { D _ { T } } = f } \| \widetilde { f } \| _ { \mathcal { C } ^ { \beta } ( \mathbb { R } \times \mathbb { T } ^ { d } ) } .
$$

For distributions, restriction is understood on the interior of the cylinder; whenever the regularity is positive, the continuous representative supplies the boundary values.

Here, a periodic distribution f on $\Lambda _ { T }$ defines a distribution on $\mathbb { R } \times \mathbb { T } ^ { d }$ , and restriction is bounded:

$$
\| f \| _ { D _ { T } } \| _ { \mathcal { C } ^ { \beta } ( D _ { T } ) } \leq C _ { \beta , T } \| f \| _ { \mathcal { C } ^ { \beta } ( \Lambda _ { T } ) } .
$$

For $\beta < 0$ , the space $\mathcal { C } ^ { \beta }$ contains distributions that need not be functions. For $0 < \beta < 1$ the anisotropic Besov space $B _ { \infty , \infty } ^ { \beta , \mathfrak { s } }$ coincides with the anisotropic H¨older space $C _ { 5 } ^ { \beta } \colon$ ; see [7, 42]. For $\beta \geq 1$ , the space is an anisotropic Zygmund-type space and is characterized by higherorder anisotropic diferences, rather than by a single $d _ { \mathfrak { s } } – \mathrm { H } \ddot { \mathrm { o l d e r } }$ condition. We will only use the embedding

$$
\begin{array} { r } { \mathcal { C } ^ { \beta } ( D _ { T } ) \hookrightarrow C _ { \mathfrak { s } } ^ { \gamma } ( D _ { T } ) , \qquad 0 < \gamma < \mathrm { m i n } \{ 1 , \beta \} . } \end{array}
$$

Moreover, $L ^ { \infty } ( \mathbb { D } _ { T } ) \hookrightarrow { \mathcal { C } } ^ { - \kappa } ( \mathbb { D } _ { T } )$ for every $\kappa > 0 \mathrm { : }$ extending $f$ by zero outside the time interval and using the uniform $L ^ { 1 }$ bounds of the dyadic kernels, one obtains

$$
\| \Delta _ { j } ^ { \mathfrak { s } } \widetilde { f } \| _ { L ^ { \infty } } \leq C \| f \| _ { L ^ { \infty } ( D _ { T } ) } ,
$$

and hence

$$
\| \widetilde { f } \| _ { \mathcal { C } ^ { - \kappa } ( \mathbb { R } \times \mathbb { T } ^ { d } ) } \leq C \| f \| _ { L ^ { \infty } ( D _ { T } ) } .
$$

These properties and the equivalence of the resulting spaces under changes of smooth dyadic decompositions, are standard; see $[ 4 , 8 ]$

For spatial regularity, we use the same construction with $\chi ^ { x } ( k ) = \varsigma ( | k | )$ and denote its blocks by $\Delta _ { j } ^ { x }$ . Thus

$$
\mathcal { C } _ { x } ^ { s } : = B _ { \infty , \infty } ^ { s } ( \mathbb { T } ^ { d } ) , \qquad \| f \| _ { \mathcal { C } _ { x } ^ { s } } : = \operatorname* { s u p } _ { j \geq - 1 } 2 ^ { j s } \| \Delta _ { j } ^ { x } f \| _ { \infty } , \qquad \mathcal { C } _ { g , x } ^ { s } : = \overline { { C ^ { \infty } ( \mathbb { T } ^ { d } ) } } ^ { \mathcal { C } _ { x } ^ { s } } .
$$

The spatial cutof $\Pi _ { J _ { 0 } } ^ { x }$ has symbol $m _ { J _ { 0 } } ^ { x } ( k ) = \varsigma ( 2 ^ { - ( J _ { 0 } + 1 ) } | k | )$ and acts componentwise on vector fields. We also define the Leray projector used in Section 5:

$$
\widehat { P u } ( k ) : = \left( I - \frac { k \otimes k } { | k | ^ { 2 } } \right) \widehat { u } ( k ) \quad ( k \neq 0 ) , \qquad \widehat { P u } ( 0 ) : = 0 .
$$

It projects onto divergence-free, mean-zero fields.

Throughout, $C , c > 0$ may change from line to line. They depend on the fixed equation, regularity exponents, and data bounds, but not on truncation levels, approximation tolerances, or localization radii unless stated otherwise. We write $A \lesssim B$ for $A \leq C B$ and $A \asymp B$ for the corresponding two-sided comparison.

2.2. Gaussian forcing and tail estimates. All random objects are defined on a complete probability space $( \Omega , \mathcal { F } , \mathbb { P } )$ , with expectation E.

Assumption 2.2 (Gaussian forcing). There is a $\beta _ { 0 } \in ( - 2 , 0 )$ such that $\eta$ is a centered Gaussian Borel random variable with values in $X _ { 0 } : = \mathcal { C } _ { g } ^ { \beta _ { 0 } } ( \Lambda _ { T } )$

We use a weaker norm to measure perturbations of the forcing. Fix

$$
- 2 < \beta _ { 1 } < \beta _ { 0 } < 0 , \qquad \delta : = \beta _ { 0 } - \beta _ { 1 } > 0 , \qquad X _ { 1 } : = C ^ { \beta _ { 1 } } ( D _ { T } ) , \qquad Y : = C ( D _ { T } ) .\tag{2}
$$

Restriction followed by the change of regularity gives a continuous map $X _ { 0 }  X _ { 1 }$ . We suppress the restriction in the notation. The norm on Y is $\| u \| _ { Y } = \operatorname* { s u p } _ { ( t , x ) \in D _ { T } } | u ( t , x )$ |.The strict gap $\delta$ will give the Fourier truncation rate, while $\beta _ { 1 } > - 2$ ensures that the heat convolution is continuous.

Lemma 2.3 (Gaussian tails). Under Assumption 2.2, there are $c , C > 0$ such that

$$
\begin{array} { r } { \mathbb { E } \exp \bigl ( c \| \eta \| _ { X _ { 0 } } ^ { 2 } \bigr ) < \infty , \qquad \mathbb { P } ( \| \eta \| _ { X _ { 0 } } > R ) \le C e ^ { - c R ^ { 2 } } , \qquad R \ge 0 . } \end{array}
$$

In particular, $\mathbb { E } \Vert \eta \Vert _ { X _ { 0 } } ^ { p } < \infty$ for every $0 < p < \infty$ . For every $q \geq 0$ , there are $c _ { q } , C _ { q } > 0 .$ independent of R, such that

$$
\mathbb { E } \left[ ( 1 + \| \eta \| _ { X _ { 0 } } ) ^ { q } \mathbf { 1 } _ { \left\{ \| \eta \| _ { X _ { 0 } } > R \right\} } \right] \le C _ { q } e ^ { - c _ { q } R ^ { 2 } } , \qquad R \ge 0 .\tag{3}
$$

Proof. Write $Z = \| \eta \| _ { X _ { 0 } }$ . Fernique’s theorem gives an $a > 0$ for which $\mathbb { E } e ^ { a Z ^ { 2 } } < \infty ;$ see $[ 6 ,$ Chapter 2]. By Markov’s inequality applied to $e ^ { a Z ^ { 2 } }$ , for every $R > 0$

$$
\mathbb { P } ( Z > R ) = \mathbb { P } ( e ^ { a Z ^ { 2 } } > e ^ { a R ^ { 2 } } ) \le e ^ { - a R ^ { 2 } } \mathbb { E } e ^ { a Z ^ { 2 } } .
$$

Moreover, since $| z | ^ { p } \leq C _ { p , a } e ^ { a z ^ { 2 } }$ for every $p > 0$ and some $C _ { p , a } < \infty$ , the finiteness of $\mathbb { E } e ^ { a Z ^ { 2 } }$ implies

$$
\begin{array} { r } { \mathbb { E } | Z | ^ { p } \leq C _ { p , a } \mathbb { E } e ^ { a Z ^ { 2 } } < \infty . } \end{array}
$$

For the weighted tail, fix $q \geq 0$ and choose $C _ { q }$ such that $( 1 + z ) ^ { q } \leq C _ { q } e ^ { a z ^ { 2 } / 2 }$ for $z \geq 0$ . On $\{ Z > R \}$ ,

$$
( 1 + Z ) ^ { q } \leq C _ { q } e ^ { a Z ^ { 2 } } e ^ { - a R ^ { 2 } / 2 } .
$$

Taking expectations gives (3), with $c _ { q } = a / 2$ after adjusting $C _ { q }$ . In particular, these constants can be chosen before the radius $R$ □

2.3. The heat equation and the Schauder estimate. Write $S ( t ) = e ^ { t \Delta }$ . The periodic heat kernel is nonnegative and has integral one, so

$$
\| S ( t ) f \| _ { \infty } \leq \| f \| _ { \infty } , \qquad t \geq 0 .
$$

For smooth forcing $\phi ,$ we define the heat convolution by

$$
K \phi ( t ) : = \int _ { 0 } ^ { t } S ( t - r ) \phi ( r ) d r .
$$

For distributional forcing, the integral is interpreted through the zero-initial-data heat equation, as justified in the next lemma. With viscosity $\nu > 0$ , the notation $K _ { \nu }$ means the same operator with $S ( t - r )$ replaced by $e ^ { \nu ( t - r ) \Delta }$

Lemma 2.4 (Schauder estimate). $L e t - 2 < \beta < 0$ . The heat equation $( \partial _ { t } - \Delta ) v = \phi , v ( 0 ) = 0$ defines a bounded linear map (solution map) $K : { \mathcal { C } } ^ { \beta } ( D _ { T } ) \to { \mathcal { C } } ^ { \beta + 2 } ( D _ { T } )$ with $v = K \phi$ satisfying

$$
\| K \phi \| _ { \mathcal { C } ^ { \beta + 2 } ( D _ { T } ) } \leq C _ { \beta } \| \phi \| _ { \mathcal { C } ^ { \beta } ( D _ { T } ) } .
$$

In particular, $K \phi$ is continuous and has zero initial trace. For every $0 < \gamma < \operatorname* { m i n } \{ 1 , \beta + 2 \}$

$$
\| K \phi \| _ { C _ { \mathfrak { s } } ^ { \gamma } ( D _ { T } ) } \le C _ { \beta , \gamma } \| \phi \| _ { \mathcal { C } ^ { \beta } ( D _ { T } ) } .\tag{4}
$$

$$
K _ { \nu } { _ { \nu } } { _ { \nu } }
$$

$$
\nu > 0
$$

Proof. Set $\alpha = \beta + 2 \in ( 0 , 2 )$ . Choose an extension $\widetilde { \phi } \in \mathcal { C } ^ { \beta } ( \mathbb { R } \times \mathbb { T } ^ { d } )$ of $\phi$ such that

$$
\| \widetilde { \phi } \| _ { \mathcal { C } ^ { \beta } ( \mathbb { R } \times \mathbb { T } ^ { d } ) } \le 2 \| \phi \| _ { \mathcal { C } ^ { \beta } ( D _ { T } ) } .
$$

Let $\vartheta \in C _ { c } ^ { \infty } ( \mathbb { R } )$ satisfy $\vartheta \equiv 1$ on a neighbourhood of $[ 0 , T ]$ , and define

$$
g ( t , x ) : = \vartheta ( t ) e ^ { - t } \widetilde { \phi } ( t , x ) , \qquad ( t , x ) \in \mathbb { R } \times \mathbb { T } ^ { d } .
$$

Multiplication by this fixed smooth function is bounded on $\mathcal { C } ^ { \beta }$ , so

$$
\| g \| _ { \mathcal { C } ^ { \beta } ( \mathbb { R } \times \mathbb { T } ^ { d } ) } \leq C _ { T } \| \phi \| _ { \mathcal { C } ^ { \beta } ( D _ { T } ) } .
$$

On the full cylinder $\mathbb { R } \times \mathbb { T } ^ { d }$ , let $w = ( \partial _ { t } - \Delta + 1 ) ^ { - 1 } g$ , defined by the Fourier multiplier $( 1 + i \omega + | k | ^ { 2 } ) ^ { - 1 }$ . For $j \geq 0$ , the parabolic block $\Delta _ { j } ^ { \mathfrak { s } }$ localizes frequencies at scale $| \omega | \sim 2 ^ { 2 j }$ and $| k | \sim 2 ^ { j }$ . On this region,

$$
| 1 + i \omega + | k | ^ { 2 } | \gtrsim 2 ^ { 2 j } ,
$$

so the localized multiplier has convolution kernel with $L ^ { 1 }$ -norm at most $C 2 ^ { - 2 j }$ . Consequently,

$$
\| \Delta _ { j } ^ { \mathfrak { s } } w \| _ { L ^ { \infty } } \leq C 2 ^ { - 2 j } \sum _ { \ell \geq - 1 \atop | \ell - j | \leq C } \| \Delta _ { \ell } ^ { \mathfrak { s } } g \| _ { L ^ { \infty } } .
$$

For the low-frequency block $j = - 1$ , the frequencies satisfy $| \omega | \lesssim 1$ and $| k | \lesssim 1$ . The constant 1 in the resolvent ensures that

$$
| 1 + i \omega + | k | ^ { 2 } | \gtrsim 1 ,
$$

so the multiplier $( 1 + i \omega + | k | ^ { 2 } ) ^ { - 1 }$ is uniformly bounded and its localized kernel has uniformly bounded $L ^ { 1 } { \mathrm { - n o r m } }$ . Hence

$$
\| \Delta _ { - 1 } ^ { \mathfrak { s } } w \| _ { L ^ { \infty } } \leq C \sum _ { \ell \geq - 1 \atop | \ell + 1 | \leq C } \| \Delta _ { \ell } ^ { \mathfrak { s } } g \| _ { L ^ { \infty } } .
$$

By the dyadic characterization of $\mathcal { C ^ { \alpha } }$ and the identity $\alpha - 2 = \beta$ , for $j \geq 0$ we have

$$
2 ^ { j \alpha } \| \Delta _ { j } ^ { \mathfrak { s } } w \| _ { L ^ { \infty } } \lesssim \sum _ { | \ell - j | \leq N } 2 ^ { ( j - \ell ) \beta } ( 2 ^ { \ell \beta } \| \Delta _ { \ell } ^ { \mathfrak { s } } g \| _ { L ^ { \infty } } ) \lesssim \| g \| _ { C ^ { \beta } } ,
$$

since the sum contains only finitely many terms and $| j - \ell | \leq C$ . Taking the supremum over $j \geq - 1$ yields

$$
\| w \| _ { \mathcal { C } ^ { \alpha } ( \mathbb { R } \times \mathbb { T } ^ { d } ) } \leq C \| g \| _ { \mathcal { C } ^ { \beta } ( \mathbb { R } \times \mathbb { T } ^ { d } ) } .
$$

Since $\alpha > 0$ , w has a continuous trace at $t = 0$ and

$$
\| w ( 0 ) \| c _ { x } ^ { \alpha } \leq C \| w \| _ { \mathcal { C } ^ { \alpha } ( \mathbb { R } \times \mathbb { T } ^ { d } ) } .
$$

Recall the standard estimate for heat extension (see [4, 8, 15])

$$
\| S ( \cdot ) f \| _ { \mathcal { C } ^ { \alpha } ( D _ { T } ) } \leq C \| f \| _ { \mathcal { C } _ { x } ^ { \alpha } } , \qquad 0 < \alpha < 2 .
$$

Note that for $\alpha = 1$ , the spatial norm is the Zygmund norm, not the Lipschitz norm.

Now define

$$
v ( t ) : = e ^ { t } w ( t ) - S ( t ) w ( 0 ) , \qquad 0 \leq t \leq T .
$$

On $( 0 , T ) \times  { \mathbb { T } } ^ { d }$ , it satisfies $( \partial _ { t } - \Delta ) v = e ^ { t } g = \phi$ and $v ( 0 ) = 0$ . The preceding bounds give $\| v \| _ { \mathcal { C } ^ { \alpha } ( D _ { T } ) } \leq C \| \phi \| _ { \mathcal { C } ^ { \beta } ( D _ { T } ) }$ . Replacing $| k | ^ { 2 }$ by $\nu | k | ^ { 2 }$ in the argument proves the result (4). □

2.4. The semilinear solution map. Let $F : \mathbb { R } $ R be globally Lipschitz, with constant $L _ { F }$ A mild solution of (1), for deterministic forcing $\phi \in X _ { 1 } = \mathcal { C } ^ { \beta _ { 1 } } ( D _ { T } )$ , is a function $u \in Y = C ( D _ { T } )$ such that

$$
u ( t ) = S ( t ) u _ { 0 } + K \phi ( t ) + \int _ { 0 } ^ { t } S ( t - r ) F ( u ( r ) ) d r .\tag{5}
$$

Theorem 2.5 (Well-posedness and pathwise stability). For every $( u _ { 0 } , \phi ) \in C ( \mathbb { T } ^ { d } ) \times X _ { 1 }$ there is a unique mild solution $u = S ( u _ { 0 } , \phi ) \in Y$ . For two pairs of data,

$$
\begin{array} { r l } & { \| S ( u _ { 0 } ^ { 1 } , \phi ^ { 1 } ) - S ( u _ { 0 } ^ { 2 } , \phi ^ { 2 } ) \| _ { Y } } \\ & { \qquad \leq e ^ { L _ { F } T } \left( \| u _ { 0 } ^ { 1 } - u _ { 0 } ^ { 2 } \| _ { \infty } + C _ { \beta _ { 1 } } \| \phi ^ { 1 } - \phi ^ { 2 } \| _ { X _ { 1 } } \right) . } \end{array}\tag{6}
$$

Moreover,

$$
\| S ( u _ { 0 } , \phi ) \| _ { Y } \leq e ^ { L _ { F } T } \left( \| u _ { 0 } \| _ { \infty } + C _ { \beta _ { 1 } } \| \phi \| _ { X _ { 1 } } + T | F ( 0 ) | \right) .\tag{7}
$$

Proof. By lemma 2.4, we have $K \phi \in Y$ with zero initial trace. Thus the right-hand side of (5) defines a map $\boldsymbol { S } : \boldsymbol { Y }  \boldsymbol { Y }$ . Equip Y with the equivalent norm

$$
\| u \| _ { a } : = \operatorname* { s u p } _ { 0 \leq t \leq T } e ^ { - a t } \| u ( t ) \| _ { \infty } , \qquad a : = 2 L _ { F } + 1 .
$$

Direct computation gives

$$
e ^ { - a t } \| S u ( t ) - S v ( t ) \| _ { \infty } \leq L _ { F } \int _ { 0 } ^ { t } e ^ { - a ( t - r ) } d r \| u - v \| _ { a } \leq \frac { L _ { F } } { a } \| u - v \| _ { a } .
$$

Since $L _ { F } / a < 1 / 2$ , the fixed-point theorem gives a unique solution on the whole interval [0, T]. Now for $u ^ { i } = { \cal S } ( u _ { 0 } ^ { i } , \phi ^ { i } )$ , subtracting the mild equations yields

$$
\| u ^ { 1 } ( t ) - u ^ { 2 } ( t ) \| _ { \infty } \leq A + L _ { F } \int _ { 0 } ^ { t } \| u ^ { 1 } ( r ) - u ^ { 2 } ( r ) \| _ { \infty } d r ,
$$

where $A = \| u _ { 0 } ^ { 1 } - u _ { 0 } ^ { 2 } \| _ { \infty } + C _ { \beta _ { 1 } } \| \phi ^ { 1 } - \phi ^ { 2 } \| _ { X _ { 1 } }$ . Gronwall’s inequality gives $\| u ^ { 1 } ( t ) - u ^ { 2 } ( t ) \| _ { \infty } \le A e ^ { L _ { F } t }$ which proves (6) after taking the supremum over t. Finally, the estimate $| F ( r ) | \le | F ( 0 ) | + L _ { F } | r |$ gives

$$
\| u ( t ) \| _ { \infty } \leq \| u _ { 0 } \| _ { \infty } + C _ { \beta _ { 1 } } \| \phi \| _ { X _ { 1 } } + T | F ( 0 ) | + L _ { F } \int _ { 0 } ^ { t } \| u ( r ) \| _ { \infty } d r .
$$

Then (7) follows from the Gronwall inequality.

## 2.5. Uniform bounds on bounded sets of data.

Assumption 2.6 (Initial-data class). For fixed M, $L _ { 0 } > 0$ , let

$$
\mathcal { G } _ { 0 } : = \left\{ u _ { 0 } \in C ( \mathbb { T } ^ { d } ) : \| u _ { 0 } \| _ { \infty } \leq M , \quad | u _ { 0 } ( x ) - u _ { 0 } ( y ) | \leq L _ { 0 } d _ { \mathbb { T } ^ { d } } ( x , y ) \ f o r \ a l l \ x , y \in \mathbb { T } ^ { d } \right\} .
$$

For the approximation of the output functions, we also need a modulus of continuity that is uniform over bounded forcing sets.

Lemma 2.7 (Uniform solution bounds). Fix

$$
0 < \gamma < \operatorname * { m i n } \{ 1 , \beta _ { 0 } + 2 \} .\tag{8}
$$

There are increasing afine functions $B , \Lambda : [ 0 , \infty )  [ 1 , \infty )$ , depending only on d, $T , \beta _ { 0 } , \beta _ { 1 } , \gamma _ { ; }$ $M , L _ { 0 } , L _ { F }$ , and $| F ( 0 ) |$ , such that for every $R \geq 0 , u _ { 0 } \in \mathcal G _ { 0 }$ , and $\zeta \in X _ { 0 }$ with $\| \zeta \| _ { X _ { 0 } } \leq R ,$

$$
\| S ( u _ { 0 } , \zeta ) \| _ { Y } \leq B ( R ) ,
$$

$$
| S ( u _ { 0 } , \zeta ) ( z ) - S ( u _ { 0 } , \zeta ) ( z ^ { \prime } ) | \leq \Lambda ( R ) d _ { \mathfrak { s } } ( z , z ^ { \prime } ) ^ { \gamma } .
$$

In particular, $B ( R ) + \Lambda ( R ) \leq C ( 1 + R )$

Proof. Let $C _ { \mathrm { r e s } }$ be a bound for restriction from $X _ { 0 }$ to $X _ { 1 }$ . The a priori estimate (7) gives the first assertion with

$$
B ( R ) : = 1 + e ^ { L _ { F } T } \big ( M + C _ { \beta _ { 1 } } C _ { \mathrm { r e s } } R + T \vert F ( 0 ) \vert \big ) .
$$

Let $L _ { 0 } = \mathrm { L i p } ( u _ { 0 } )$ . Then $\mathrm { L i p } ( S ( t ) u _ { 0 } ) \leq L _ { 0 }$

$$
\begin{array} { r l r } {  { \| S ( h ) u _ { 0 } - u _ { 0 } \| _ { \infty } \le L _ { 0 } \int _ { \mathbb { T } ^ { d } } K _ { h } ( z ) d _ { \mathbb { T } ^ { d } } ( z , 0 ) d z } } \\ & { } & { \le C L _ { 0 } h ^ { 1 / 2 } . } \end{array}
$$

For $t \geq s \geq 0$

$$
\begin{array} { r l } & { | S ( t ) u _ { 0 } ( x ) - S ( s ) u _ { 0 } ( y ) | \leq | S ( t ) u _ { 0 } ( x ) - S ( t ) u _ { 0 } ( y ) | + \| S ( s ) ( S ( t - s ) u _ { 0 } - u _ { 0 } ) \| _ { \infty } } \\ & { \qquad \leq L _ { 0 } d _ { \mathbb { T } ^ { d } } ( x , y ) + \| S ( t - s ) u _ { 0 } - u _ { 0 } \| _ { \infty } } \\ & { \qquad \leq C L _ { 0 } \big ( d _ { \mathbb { T } ^ { d } } ( x , y ) + | t - s | ^ { 1 / 2 } \big ) . } \end{array}
$$

Writing $D = \mathrm { d i a m } _ { \mathfrak { s } } ( D _ { T } ) < \infty$ , we obtain, for every $0 < \gamma < 1$

$$
[ S ( \cdot ) u _ { 0 } ] _ { C _ { s } ^ { \gamma } ( D _ { T } ) } \leq D ^ { 1 - \gamma } [ S ( \cdot ) u _ { 0 } ] _ { C _ { s } ^ { 1 } ( D _ { T } ) } \leq C L _ { 0 } , \qquad \| S ( \cdot ) u _ { 0 } \| _ { \infty } \leq M .
$$

Hence

$$
\| S ( \cdot ) u _ { 0 } \| _ { C _ { \mathfrak { s } } ^ { \gamma } ( D _ { T } ) } \le C ( M + L _ { 0 } ) ,
$$

uniformly up to $t = 0 .$

By Lemma 2.4 with $\beta = \beta _ { 0 }$ and $\gamma < \beta _ { 0 } + 2$ , as in (8),

$$
\| K \zeta \| _ { C _ { \mathfrak { s } } ^ { \gamma } ( D _ { T } ) } \le C \| \zeta \| _ { \mathcal { C } ^ { \beta _ { 0 } } ( D _ { T } ) } \le C R .
$$

Moreover,

$$
\| F ( u ) \| _ { \infty } \leq | F ( 0 ) | + L _ { F } \| u \| _ { \infty } \leq | F ( 0 ) | + L _ { F } B ( R ) .
$$

Since

$$
L ^ { \infty } \hookrightarrow { \mathcal { C } } ^ { - 1 / 2 } , \qquad \gamma < 1 < - { \textstyle { \frac { 1 } { 2 } } } + 2 = { \textstyle { \frac { 3 } { 2 } } } ,
$$

another application of Lemma 2.4 gives

$$
\begin{array} { r l } & { \| K ( F ( u ) ) \| _ { C _ { \mathfrak { s } } ^ { \gamma } ( D _ { T } ) } \le C \| F ( u ) \| _ { C ^ { - 1 / 2 } ( D _ { T } ) } } \\ & { \qquad \le C \| F ( u ) \| _ { \infty } } \\ & { \qquad \le C \bigl ( | F ( 0 ) | + L _ { F } B ( R ) \bigr ) . } \end{array}
$$

Combining these estimates proves the second assertion with

$$
\Lambda ( R ) : = 1 + C \bigl ( M + L _ { 0 } + R + | F ( 0 ) | + L _ { F } B ( R ) \bigr ) .
$$

Both $B ( R )$ and $\Lambda ( R )$ are afine in $R ,$ with coeficients depending only on the fixed data in the statement. □

## 3. Finite-dimensional encoding and operator architecture

3.1. A real finite-rank measurement map. Let $\begin{array} { r } { \Pi _ { J } = \sum _ { j \leq J } \Delta _ { j } ^ { \mathfrak { s } } } \end{array}$ and let $m _ { J }$ denote its smooth, even Fourier multiplier. We choose the Littlewood–Paley decomposition with nested low-pass multipliers, so that

$$
0 \leq m _ { J } ( \lambda ) \leq m _ { J + 1 } ( \lambda ) \leq 1 , \qquad m _ { J } ( \lambda ) \longrightarrow 1 \quad \mathrm { f o r ~ e v e r y ~ f i x e d ~ } \lambda .\tag{9}
$$

The operator $\Pi _ { J }$ is a finite-rank smoothing cutof, not necessarily a projection. Let $( e _ { \lambda } ) _ { \lambda }$ be the real orthonormal trigonometric basis of $L ^ { 2 } ( \Lambda _ { T } )$ , consisting of the normalized constant function and the normalized sine and cosine functions, with one sine–cosine pair for each pair of nonzero frequencies $\{ \theta , - \theta \}$ . Since $m _ { J } ( \theta ) = m _ { J } ( - \theta )$ , the operator $\Pi _ { J }$ multiplies both functions in each pair by the same scalar $m _ { J } ( \theta )$ . Thus every $e _ { \lambda }$ is an eigenfunction of $\Pi _ { J }$

Let $\Lambda _ { J }$ be the set of indices of the basis functions not annihilated by $\Pi _ { J } \mathbf { : }$

$$
\Lambda _ { J } : = \{ \lambda : \Pi _ { J } e _ { \lambda } \neq 0 \} .
$$

Define

$$
c _ { J } ( \zeta ) : = ( \langle \zeta , e _ { \lambda } \rangle ) _ { \lambda \in \Lambda _ { J } } \in \mathbb { R } ^ { N _ { J } } , \qquad \operatorname { R e c } _ { J } ( c ) : = \sum _ { \lambda \in \Lambda _ { J } } m _ { J } ( \lambda ) c _ { \lambda } e _ { \lambda } .\tag{10}
$$

Then $\operatorname { R e c } _ { J } ( c _ { J } ( \zeta ) ) = \Pi _ { J } \zeta$ exactly. Parabolic frequency counting gives

$$
N _ { J } : = | \Lambda _ { J } | \asymp 2 ^ { J ( d + 2 ) } .\tag{11}
$$

The count and the uniform boundedness of the smooth cutofs will be proved in Lemma 3.1. For an abstract forcing satisfying Assumption 2.2, $c _ { J } ( \eta )$ is a centered jointly Gaussian vector with covariance $\Sigma _ { \lambda \mu } = \mathbb { E } [ \langle \eta , e _ { \lambda } \rangle \langle \eta , e _ { \mu } \rangle ]$ . No independence of these coordinates is assumed in the approximation theorems.

Lemma 3.1. For $J \geq 0$ ，

$$
\operatorname { r a n k } ( \Pi _ { J } ) \asymp 2 ^ { J ( d + 2 ) } .\tag{12}
$$

Moreover, for every $\beta \in \mathbb { R }$ , there exists $C _ { \beta } > 0$ , independent of J, such that

$$
\| \Pi _ { J } f \| _ { { \mathcal { C } } ^ { \beta } } \leq C _ { \beta } \| f \| _ { { \mathcal { C } } ^ { \beta } } .\tag{13}
$$

Proof. For $a > 0$ , set

$$
\begin{array} { r } { Q _ { J } ( a ) : = \big \{ ( n , k ) \in \mathbb { Z } \times \mathbb { Z } ^ { d } : | n | \leq a 2 ^ { 2 J } , ~ | k | \leq a 2 ^ { J } \big \} . } \end{array}
$$

The parabolic cutof construction gives fixed constants $0 < c < C$ such that

$$
Q _ { J } ( c ) \subseteq \{ ( n , k ) : m _ { J } ( n , k ) \neq 0 \} \subseteq Q _ { J } ( C ) .
$$

Since

$$
| Q _ { J } ( a ) | \asymp 2 ^ { 2 J } 2 ^ { d J } = 2 ^ { J ( d + 2 ) } ,
$$

and the rank of $\Pi _ { J }$ equals the number of frequencies on which $m _ { J }$ is nonzero, (12) follows.

For the uniform bound, finite overlap of the dyadic supports yields a fixed integer N such that

$$
\Delta _ { j } ^ { \mathfrak { s } } \Pi _ { J } f = \sum _ { | j - j ^ { \prime } | \leq N } \Delta _ { j } ^ { \mathfrak { s } } \Pi _ { J } \Delta _ { j ^ { \prime } } ^ { \mathfrak { s } } f .
$$

The operators $\Delta _ { j } ^ { \mathfrak { s } } \Pi _ { J }$ have convolution kernels with $L ^ { 1 }$ norms bounded uniformly in $j$ and $J .$ Consequently,

$$
\begin{array} { r l r } {  { 2 ^ { j \beta } \| \Delta _ { j } ^ { \mathfrak { s } } \Pi _ { J } f \| _ { \infty } \le C \sum _ { | j - j ^ { \prime } | \le N } 2 ^ { ( j - j ^ { \prime } ) \beta } 2 ^ { j ^ { \prime } \beta } \| \Delta _ { j ^ { \prime } } ^ { \mathfrak { s } } f \| _ { \infty } } } \\ & { } & { \le C _ { \beta } \| f \| _ { \mathcal { C } ^ { \beta } } . } \end{array}
$$

Taking the supremum over $j$ proves (13).

Lemma 3.2. For every $\zeta \in \mathcal { C } ^ { \beta _ { 0 } } ( \Lambda _ { T } )$ ，

$$
\| ( I - \Pi _ { J } ) \zeta \| _ { \mathcal { C } ^ { \beta _ { 1 } } } \leq C 2 ^ { - J \delta } \| \zeta \| _ { \mathcal { C } ^ { \beta _ { 0 } } } .\tag{14}
$$

Proof. Almost orthogonality gives $\Delta _ { j } ^ { \mathfrak { s } } ( I - \Pi _ { J } ) \zeta = 0$ for $j < J - N$ , then

$$
2 ^ { j \beta _ { 1 } } \| \Delta _ { j } ^ { \mathfrak { s } } ( I - \Pi _ { J } ) \zeta \| _ { \infty } \leq C 2 ^ { - j ( \beta _ { 0 } - \beta _ { 1 } ) } 2 ^ { j \beta _ { 0 } } \| \Delta _ { j } ^ { \mathfrak { s } } \zeta \| _ { \infty }
$$

for $j \ge J - N$ . Taking the supremum over $j$ proves the claim.

Lemma 3.3. There is a constant $C > 0$ , independent of J, such that

(15)

$$
\begin{array} { r l } & { \| c _ { J } ( \zeta ) \| _ { \ell ^ { \infty } } \leq C 2 ^ { - J \beta _ { 0 } } \| \zeta \| _ { \mathcal { C } ^ { \beta _ { 0 } } } , } \\ & { \| \operatorname { R e c } _ { J } ( c ) - \operatorname { R e c } _ { J } ( c ^ { \prime } ) \| _ { X _ { 1 } } \leq C N _ { J } ^ { 1 / 2 } \| c - c ^ { \prime } \| _ { \ell ^ { 2 } } . } \end{array}\tag{16}
$$

Proof. Let $e _ { \lambda }$ have parabolic frequency comparable to $2 ^ { j }$ . By finite overlap of the Littlewood– Paley supports,

$$
| \langle \zeta , e _ { \lambda } \rangle | \leq C \sum _ { | j ^ { \prime } - j | \leq N } \| \Delta _ { j ^ { \prime } } ^ { \mathfrak { s } } \zeta \| _ { \infty } \leq C 2 ^ { - j \beta _ { 0 } } \| \zeta \| _ { \mathcal { C } ^ { \beta _ { 0 } } } ,
$$

where N is fixed; the constant mode satisfies the corresponding low-frequency bound. For $\lambda \in \Lambda _ { J }$ we have $j \leq J + C _ { 0 }$ . Since $\beta _ { 0 } < 0$ 2

$$
\operatorname* { s u p } _ { \lambda \in \Lambda _ { J } } | \langle \zeta , e _ { \lambda } \rangle | \leq C 2 ^ { - J \beta _ { 0 } } \| \zeta \| _ { \mathcal { C } ^ { \beta _ { 0 } } } ,
$$

which proves (15).

Since $\beta _ { 1 } < 0$ , the normalized trigonometric basis functions satisfy

$$
\operatorname* { s u p } _ { \lambda } \| e _ { \lambda } \| _ { X _ { 1 } } < \infty .
$$

Using $0 \leq m _ { J } \leq 1$ , the triangle inequality and Cauchy–Schwarz give

$$
\begin{array} { r l } { \| \operatorname { R e c } _ { J } ( c ) - \operatorname { R e c } _ { J } ( c ^ { \prime } ) \| _ { X _ { 1 } } \leq \displaystyle \sum _ { \lambda \in \Lambda _ { J } } | m _ { J } ( \lambda ) | | c _ { \lambda } - c _ { \lambda } ^ { \prime } | \| e _ { \lambda } \| _ { X _ { 1 } } } & { } \\ { \leq C \displaystyle \sum _ { \lambda \in \Lambda _ { J } } | c _ { \lambda } - c _ { \lambda } ^ { \prime } | } & { } \\ { \leq C N _ { J } ^ { 1 / 2 } \| c - c ^ { \prime } \| _ { \ell ^ { 2 } } . } \end{array}
$$

This proves (16).

3.2. Initial-data measurements and the reduced map. Let $x _ { 1 } , \ldots , x _ { N _ { 0 } }$ be a periodic tensor grid of mesh $\delta _ { 0 }$ and let $\vec { u } _ { 0 } = ( u _ { 0 } ( x _ { i } ) ) _ { i \le N _ { 0 } }$ . Denote by $\operatorname { R e c } _ { 0 } ( \vec { v } )$ the periodic piecewise multilinear reconstruction.

Lemma 3.4. Let $( x _ { i } ) _ { i \leq N _ { 0 } }$ be a periodic tensor grid on $\mathbb { T } ^ { d }$ with mesh at most $\delta _ { 0 }$ , and let Rec<sub>0</sub> be its piecewise multilinear interpolant. If u is $L _ { 0 } { - } L i p s c h i t z$ , then

$$
\| u _ { 0 } - \mathrm { R e c } _ { 0 } ( ( u _ { 0 } ( x _ { i } ) ) _ { i } ) \| _ { \infty } \leq C _ { d } L _ { 0 } \delta _ { 0 } .\tag{17}
$$

For arbitrary nodal vectors $v , v ^ { \prime }$ , one has

$$
\begin{array} { r } { \| \operatorname { R e c } _ { 0 } ( v ) - \operatorname { R e c } _ { 0 } ( v ^ { \prime } ) \| _ { \infty } \leq | v - v ^ { \prime } | _ { \ell ^ { \infty } } . } \end{array}\tag{18}
$$

Moreover, $N _ { 0 } \times \delta _ { 0 } ^ { - d }$

Proof. On each grid cell, multilinear interpolation has the form

$$
\operatorname { R e c } _ { 0 } ( ( u _ { 0 } ( x _ { i } ) ) _ { i } ) ( x ) = \sum _ { \nu \in \{ 0 , 1 \} ^ { d } } w _ { \nu } ( x ) u _ { 0 } ( x _ { \nu } ) ,
$$

where the vertex weights are nonnegative and sum to one. Every vertex $x _ { \nu }$ of the cell satisfies $d _ { \mathbb { T } ^ { d } } ( x , x _ { \nu } ) \leq \sqrt { d } \delta _ { 0 }$ . Therefore

$$
| u _ { 0 } ( x ) - \mathrm { R e c } _ { 0 } ( ( u _ { 0 } ( x _ { i } ) ) _ { i } ) ( x ) | \leq \sum _ { \nu } w _ { \nu } ( x ) L _ { 0 } d _ { \mathbb { T } ^ { d } } ( x , x _ { \nu } ) \leq \sqrt { d } L _ { 0 } \delta _ { 0 } .
$$

The same convex-combination formula applied to $v - v ^ { \prime }$ proves (18). Finally, a periodic tensor grid has a number of nodes comparable to the reciprocal of its cell volume, namely $\delta _ { 0 } ^ { - d }$ □

For $\boldsymbol { z } = ( \vec { v } , c ) \in \mathbb { R } ^ { N _ { 0 } } \times \mathbb { R } ^ { N _ { J } }$ define

$$
\Gamma _ { J } ( z ) : = { \cal S } \big ( \mathrm { R e c } _ { 0 } ( \vec { v } ) , \mathrm { R e c } _ { J } ( c ) \big ) \in Y .\tag{19}
$$

Lemma 3.5. The map $\Gamma _ { J }$ is Lipschitz, with

$$
\| \Gamma _ { J } ( \vec { v } , c ) - \Gamma _ { J } ( \vec { v } ^ { \prime } , c ^ { \prime } ) \| _ { Y } \leq L _ { J } \big ( \| \vec { v } - \vec { v } ^ { \prime } \| _ { \ell ^ { \infty } } + \| c - c ^ { \prime } \| _ { \ell ^ { 2 } } \big ) ,
$$

where

$$
L _ { J } : = C e ^ { L _ { F } T } \operatorname* { m a x } \{ 1 , N _ { J } ^ { 1 / 2 } \} ,
$$

and $C$ is independent of J.

Proof. By Theorem 2.5, (18), and (16),

$$
\begin{array} { r l } & { \| \Gamma _ { J } ( \vec { v } , c ) - \Gamma _ { J } ( \vec { v } ^ { \prime } , c ^ { \prime } ) \| _ { Y } \leq C e ^ { L _ { F } T } \big ( \| \vec { v } - \vec { v } ^ { \prime } \| _ { \ell ^ { \infty } } + N _ { J } ^ { 1 / 2 } \| c - c ^ { \prime } \| _ { \ell ^ { 2 } } \big ) } \\ & { \qquad \leq L _ { J } \big ( \| \vec { v } - \vec { v } ^ { \prime } \| _ { \ell ^ { \infty } } + \| c - c ^ { \prime } \| _ { \ell ^ { 2 } } \big ) . } \end{array}
$$

Proposition 3.6. There is a constant C, independent of J and $\delta _ { 0 }$ , such that for every $u _ { 0 } \in \mathcal { G } _ { 0 }$ and $\zeta \in X _ { 0 }$ 2

$$
\begin{array} { r } { \big \| \mathcal S ( u _ { 0 } , \zeta ) - \Gamma _ { J } \big ( \vec { u } _ { 0 } , c _ { J } ( \zeta ) \big ) \big \| _ { Y } \leq C e ^ { L _ { F } T } \left( L _ { 0 } \delta _ { 0 } + 2 ^ { - J \delta } \| \zeta \| _ { X _ { 0 } } \right) . } \end{array}\tag{20}
$$

Equivalently, in terms of the number of forcing measurements,

$$
\begin{array} { r } { \left\| \mathcal { S } ( u _ { 0 } , \zeta ) - \Gamma _ { J } \big ( \vec { u } _ { 0 } , c _ { J } ( \zeta ) \big ) \right\| _ { Y } \leq C e ^ { L _ { F } T } \left( L _ { 0 } \delta _ { 0 } + N _ { J } ^ { - \delta / ( d + 2 ) } \| \zeta \| _ { X _ { 0 } } \right) . } \end{array}\tag{21}
$$

Proof. By (10) and (19),

$$
\Gamma _ { J } \big ( \vec { u } _ { 0 } , c _ { J } ( \zeta ) \big ) = \mathcal { S } \big ( \mathrm { R e c } _ { 0 } ( \vec { u } _ { 0 } ) , \Pi _ { J } \zeta \big ) .
$$

The stability estimate (6) therefore bounds the left-hand side of (20) by a fixed multiple of

$$
\| u _ { 0 } - \operatorname { R e c } _ { 0 } ( \vec { u } _ { 0 } ) \| _ { \infty } + \| ( I - \Pi _ { J } ) \zeta \| _ { X _ { 1 } } .
$$

Apply (17) to the first term and Lemma 3.2 to the second. This proves (20). Finally, $N _ { J } \asymp 2 ^ { J ( d + 2 ) }$ implies 2<sup>−</sup> $^ { \cdot J \delta } \asymp N _ { J } ^ { - \delta / ( d + 2 ) }$ , which yields (21). □

Remark 3.7. The exponent $\delta = \beta _ { 0 } - \beta _ { 1 }$ is the amount of regularity exchanged for compactness. A larger gap improves the decay in (21), but the stability space must still satisfy $\beta _ { 1 } > - 2$ so that the heat convolution is continuous. Hence $\delta < \beta _ { 0 } + 2$ . The limiting sensor exponent permitted by this argument is therefore $( \beta _ { 0 } + 2 ) / ( d + 2 )$ , approached by taking $\beta _ { 1 } \downarrow - 2$

3.3. Deep neural operator architecture. Let $\sigma ( r ) = \operatorname* { m a x } \{ r , 0 \}$ denote the ReLU activation, applied coordinatewise to vectors.

Definition 3.8 (ReLU network class). Let $D \subset \mathbb { R } ^ { n }$ be compact. We write

$$
\Phi \in \mathrm { F N N } ( n , m , L , p , K , \kappa , R ; D )
$$

if $\Phi : \mathbb { R } ^ { n }  \mathbb { R } ^ { m }$ has a representation

$$
\begin{array} { r l } & { x _ { 0 } = x , } \\ & { x _ { \ell } = \sigma ( W _ { \ell } x _ { \ell - 1 } + b _ { \ell } ) , \qquad 1 \le \ell < L , } \\ & { \Phi ( x ) = W _ { L } x _ { L - 1 } + b _ { L } , } \end{array}
$$

such that every hidden layer has width at most $p ,$ the matrices and biases have at most K nonzero entries in total, every parameter has absolute value at most κ, and $\| \Phi \| _ { L ^ { \infty } ( D ) } \leq R$ . Thus $L , p ,$ K, and κ record depth, width, sparsity, and parameter magnitude, respectively.

This is the network model used in [5]. We shall use the following elementary constructions.

Lemma 3.9. The following statements hold.

(1) For $a < b$ , the projection onto $[ a , b ]$ is represented exactly by

$$
P _ { [ a , b ] } ( x ) = a + \sigma ( x - a ) - \sigma ( x - b ) .
$$

Applying this formula coordinatewise gives an exact ReLU realization of the projection onto a rectangular box.

(2) For $c \in \mathbb { R }$ and $r > 0$ , the hat function

$$
H _ { c , r } ( x ) = \sigma { \big ( } 1 - r ^ { - 1 } { \big ( } \sigma ( x - c ) + \sigma ( c - x ) { \big ) } { \big ) } = \left( 1 - { \frac { | x - c | } { r } } \right) _ { + }
$$

is represented exactly by a fixed-depth ReLU network.

(3) Let $M \geq 1$ and $\tau \in ( 0 , 1 )$ . There is a ReLU network Mult $\smash { M , \tau : \mathbb { R } ^ { 2 } \to \mathbb { R } }$ such that

$$
\operatorname* { s u p } _ { | x | , | y | \leq M } | \operatorname { M u l t } _ { M , \tau } ( x , y ) - x y | \leq \tau .
$$

Its width is bounded by an absolute constant, and its depth and number of nonzero parameters are ${ \cal O } _ { M } ( \log ( \tau ^ { - 1 } ) )$

Consequently, for every integer $q \geq 2$ , a product of q factors in $[ 0 , 1 ]$ can be approximated within τ by a ReLU network with depth O(log q $\log ( q / \tau ) )$ and $O ( q \log ( q / \tau ) )$ nonzero parameters. For fixed q, both quantities are $O _ { q } ( \log ( 1 / \tau ) )$

Proof. The first two statements follow from

$$
\sigma ( x ) = \mathrm { m a x } \{ x , 0 \} , \qquad \sigma ( x - c ) + \sigma ( c - x ) = | x - c | .
$$

The third statement is [5, Lemma 3.1].

For the product approximation, let

$$
Q : = 2 ^ { \lceil \log _ { 2 } q \rceil } , \qquad q \leq Q < 2 q ,
$$

and pad the $q$ inputs with $Q - q$ factors equal to 1. Arrange the Q factors as the leaves of a balanced binary tree. At each internal node, use

$$
( u , v ) \longmapsto P _ { [ 0 , 1 ] } \big ( \mathrm { M u l t } _ { 1 , \delta } ( u , v ) \big ) , \qquad \delta : = \frac { \tau } { 2 q } .
$$

To estimate the error, let $a , b \in [ 0 , 1 ]$ be the exact products associated with the two child nodes, and let $\widetilde { a } , \widetilde { b } \in [ 0 , 1 ]$ be their approximations. Since $P _ { [ 0 , 1 ] }$ is 1-Lipschitz and fixes $a b$

$$
\begin{array} { r l } & { \left| P _ { [ 0 , 1 ] } \big ( \mathrm { M u l t } _ { 1 , \delta } ( \widetilde { a } , \widetilde { b } ) \big ) - a b \right| } \\ & { \qquad \leq \left| \mathrm { M u l t } _ { 1 , \delta } ( \widetilde { a } , \widetilde { b } ) - \widetilde { a } \widetilde { b } \right| + | \widetilde { a } \widetilde { b } - a b | } \\ & { \qquad \leq \delta + | \widetilde { a } - a | + | \widetilde { b } - b | . } \end{array}
$$

The leaf errors vanish, so induction over the tree gives a root error of at most

$$
( Q - 1 ) \delta < 2 q \delta = \tau .
$$

The tree has $\log _ { 2 } Q = O ( \log q )$ levels and $Q - 1 = O ( q )$ internal nodes. Each node requires depth and number of nonzero parameters $O ( \log ( q / \tau ) )$ , including the exact projection onto $[ 0 , 1 ]$ Implementing the nodes in parallel at each level therefore gives

$$
\mathrm { d e p t h } = O \bigl ( \log q \log ( q / \tau ) \bigr ) , \qquad \mathrm { s i z e } = O \bigl ( q \log ( q / \tau ) \bigr ) .
$$

For fixed $q ,$ both bounds are $O _ { q } ( \log ( 1 / \tau ) )$

For fixed J and $\delta _ { 0 }$ , define the measurement map

$$
\mathcal { E } _ { J , \delta _ { 0 } } ( u _ { 0 } , \zeta ) : = ( \vec { u } _ { 0 } , c _ { J } ( \zeta ) ) \in \mathbb { R } ^ { N _ { 0 } + N _ { J } } .\tag{22}
$$

The branch–trunk architecture considered below has the form

$$
\Gamma _ { \theta } \bigl ( \mathcal { E } _ { J , \delta _ { 0 } } ( u _ { 0 } , \zeta ) \bigr ) ( y ) = \sum _ { \ell = 1 } ^ { N _ { 1 } } \widetilde { a } _ { \ell } \bigl ( \vec { u } _ { 0 } , c _ { J } ( \zeta ) ; \theta _ { b } \bigr ) \widetilde { q } _ { \ell } ( y ; \theta _ { t } ) , \qquad y = ( t , x ) \in D _ { T } .
$$

The branch networks $\widetilde { a } _ { \ell }$ act on the finite measurement vector, whereas the trunk networks $\widetilde { q } _ { \ell }$ act on the output coordinate.

3.4. A finite-dimensional partition. The partition of unity is applied only after the measurement map has placed the input in a finite-dimensional Euclidean space.

Lemma 3.10. Let $K \subset \mathbb { R } ^ { N }$ be compact, let $\Gamma : K \to Y$ be L-Lipschitz with respect to the Euclidean metric, and let $z ^ { 1 } , \ldots , z ^ { m }$ be an $r / 2 – n e t$ of K. Define

$$
h _ { k } ( z ) : = \bigg ( 1 - \frac { | z - z ^ { k } | _ { 2 } } { r } \bigg ) _ { + } , \qquad \rho _ { k } ( z ) : = \frac { h _ { k } ( z ) } { \sum _ { j = 1 } ^ { m } h _ { j } ( z ) } .
$$

Then the denominator is at least $1 / 2$ on $K _ { i }$ , the functions ρ<sub>k</sub> form a continuous partition of unity, and

$$
A _ { r } ( z ) : = \sum _ { k = 1 } ^ { m } \rho _ { k } ( z ) \Gamma ( z ^ { k } ) \quad \Longrightarrow \quad \operatorname* { s u p } _ { z \in K } \| \Gamma ( z ) - A _ { r } ( z ) \| _ { Y } \leq L r .
$$

Proof. Every $z \in K$ lies within $r / 2$ of a center, so the denominator is at least $1 / 2$ . If $\rho _ { k } ( z ) > 0$ then $| z - z ^ { k } | _ { 2 } < r$ . The estimate follows from convexity:

$$
\| \Gamma ( z ) - A _ { r } ( z ) \| _ { Y } \leq \sum _ { k } \rho _ { k } ( z ) \| \Gamma ( z ) - \Gamma ( z ^ { k } ) \| _ { Y } \leq L r .
$$

3.5. Relation to Wiener chaos expansions. For the linear equation, the Fourier measurements determine a finite sum in the first Wiener chaos. The following result also gives an exact covariance formula for the mean-square truncation error.

Lemma 3.11. Assume $F \equiv 0$ . For every real Fourier mode $e _ { \lambda . }$ , set

$$
\xi _ { \lambda } : = \langle \eta , e _ { \lambda } \rangle , \qquad z _ { \lambda } : = K e _ { \lambda } ,
$$

and write $\Sigma _ { \lambda \mu } : = \mathbb { E } [ \xi _ { \lambda } \xi _ { \mu } ]$ . Then:

(1) For every J,

$$
\mathcal { S } ( u _ { 0 } , \Pi _ { J } \eta ) = e ^ { t \Delta } u _ { 0 } + \sum _ { \lambda \in \Lambda _ { J } } m _ { J } ( \lambda ) \xi _ { \lambda } z _ { \lambda } .\tag{23}
$$

(2) The solutions in (23) converge to $\boldsymbol { S } ( \boldsymbol { u } _ { 0 } , \eta )$ almost surely and in $L ^ { 2 } ( \Omega ; C ( D _ { T } ) )$ . The random part $K \eta$ belongs to the first Wiener chaos generated by $( \xi _ { \lambda } ) _ { \lambda } ;$ all chaos components of order at least two vanish.

(3) The truncation error has the exact covariance representation

$$
\begin{array} { r l r } { \left. { \mathbb { E } \| \mathcal { S } ( u _ { 0 } , \eta ) - \mathcal { S } ( u _ { 0 } , \Pi _ { J } \eta ) \| _ { L ^ { 2 } ( D _ { T } ) } ^ { 2 } } \right.} \\ & { } & { = \underset { L  \infty } { \operatorname* { l i m } } \sum _ { \lambda , \mu \in \Lambda _ { L } } ( m _ { L } ( \lambda ) - m _ { J } ( \lambda ) ) ( m _ { L } ( \mu ) - m _ { J } ( \mu ) ) \Sigma _ { \lambda \mu } \langle z _ { \lambda } , z _ { \mu } \rangle _ { L ^ { 2 } ( D _ { T } ) } , } \end{array}\tag{24}
$$

each inner sum being finite and exact for the smooth outer cutof $\Pi _ { L } , L > J$ . If the real Fourier basis diagonalizes the covariance, so that $\Sigma _ { \lambda \mu } = \sigma _ { \lambda } ^ { 2 } { \bf 1 } _ { \{ \lambda = \mu \} }$ , then

$$
\mathbb { E } \| \mathcal { S } ( u _ { 0 } , \eta ) - \mathcal { S } ( u _ { 0 } , \Pi _ { J } \eta ) \| _ { L ^ { 2 } ( D _ { T } ) } ^ { 2 } = \sum _ { \lambda } ( 1 - m _ { J } ( \lambda ) ) ^ { 2 } \sigma _ { \lambda } ^ { 2 } \| z _ { \lambda } \| _ { L ^ { 2 } ( D _ { T } ) } ^ { 2 } .\tag{25}
$$

Proof. For $F \equiv 0$ ，

$$
S ( u _ { 0 } , \phi ) = e ^ { t \Delta } u _ { 0 } + K \phi .
$$

Since

$$
\Pi _ { J \eta } = \sum _ { \lambda \in \Lambda _ { J } } m _ { J } ( \lambda ) \xi _ { \lambda } e _ { \lambda } ,
$$

linearity of K gives

$$
\mathcal { S } ( u _ { 0 } , \Pi _ { J } \eta ) = e ^ { t \Delta } u _ { 0 } + \sum _ { \lambda \in \Lambda _ { J } } m _ { J } ( \lambda ) \xi _ { \lambda } z _ { \lambda } ,
$$

proving (23).

Theorem 2.5 with $L _ { F } = 0$ and Lemma 3.2 yield

$$
\Vert S ( u _ { 0 } , \eta ) - S ( u _ { 0 } , \Pi _ { J } \eta ) \Vert _ { Y } \leq C 2 ^ { - J \delta } \Vert \eta \Vert _ { X _ { 0 } } .\tag{26}
$$

By Lemma 2.3,

$$
\| \eta \| _ { X _ { 0 } } < \infty \quad \mathrm { a l m o s t ~ s u r e l y } , \qquad \mathbb { E } \| \eta \| _ { X _ { 0 } } ^ { 2 } < \infty .
$$

Thus (26) implies convergence both almost surely and in $L ^ { 2 } ( \Omega ; C ( D _ { T } ) )$ .

Each random part

$$
K \Pi _ { J } \eta = \sum _ { \lambda \in \Lambda _ { J } } m _ { J } ( \lambda ) \xi _ { \lambda } z _ { \lambda }
$$

is a finite linear combination of the centered Gaussian coordinates $\xi _ { \lambda }$ , with deterministic coeficients $z _ { \lambda }$ . Since the first Wiener chaos is closed in $L ^ { 2 }$ , the limit $K \eta$ belongs to the first chaos generated by these coordinates. Consequently, the solution has only its deterministic component $e ^ { t \Delta } u _ { 0 }$ and its first-chaos component $K \eta$

Next fix J and let $L > J$ . By nesting, $\Lambda _ { J } \subseteq \Lambda _ { L }$ , so

$$
K ( \Pi _ { L } - \Pi _ { J } ) \eta = \sum _ { \lambda \in \Lambda _ { L } } \bigl ( m _ { L } ( \lambda ) - m _ { J } ( \lambda ) \bigr ) \xi _ { \lambda } z _ { \lambda } .
$$

Expanding the squared norm and taking expectations gives

$$
\begin{array} { r l } {  { \mathbb { E } \| K ( \Pi _ { L } - \Pi _ { J } ) \eta \| _ { L ^ { 2 } ( D _ { T } ) } ^ { 2 } } } \\ & { = \sum _ { \lambda , \mu \in \Lambda _ { L } } \bigl ( m _ { L } ( \lambda ) - m _ { J } ( \lambda ) \bigr ) \bigl ( m _ { L } ( \mu ) - m _ { J } ( \mu ) \bigr ) \Sigma _ { \lambda \mu } \langle z _ { \lambda } , z _ { \mu } \rangle _ { L ^ { 2 } ( D _ { T } ) } . } \end{array}
$$

Moreover, (26) implies

$$
{ \cal K } ( \Pi _ { L } - \Pi _ { J } ) \eta \longrightarrow { \cal K } ( I - \Pi _ { J } ) \eta \quad \mathrm { i n ~ } L ^ { 2 } ( \Omega ; L ^ { 2 } ( D _ { T } ) ) .
$$

The second moments therefore converge, proving (24).

If Σ is diagonal, the finite covariance identity reduces to

$$
\mathbb { E } \| K ( \Pi _ { L } - \Pi _ { J } ) \eta \| _ { L ^ { 2 } ( D _ { T } ) } ^ { 2 } = \sum _ { \lambda \in \Lambda _ { L } } \big ( m _ { L } ( \lambda ) - m _ { J } ( \lambda ) \big ) ^ { 2 } \sigma _ { \lambda } ^ { 2 } \| z _ { \lambda } \| _ { L ^ { 2 } ( D _ { T } ) } ^ { 2 } .
$$

For each λ, the nesting property (9) gives

$$
\left( m _ { L } ( \lambda ) - m _ { J } ( \lambda ) \right) ^ { 2 } \uparrow \left( 1 - m _ { J } ( \lambda ) \right) ^ { 2 } \qquad \mathrm { a s } \ L \to \infty .
$$

Extending each finite sum by zero outside $\Lambda _ { L }$ and applying monotone convergence proves (25). □

## 4. Approximation results

4.1. Function approximation: the trunk network. We first approximate functions $v ( t , x )$ on $D _ { T } = [ 0 , T ] \times  { \mathbb { T } } ^ { d }$ using their values at finitely many grid points. The H¨older estimate below controls their variation in terms of $| t - s | ^ { 1 / 2 } + d _ { \mathbb { T } ^ { d } } ( x , y )$ . We therefore choose a spatial mesh of size h and a temporal mesh of size $h ^ { 2 }$ , so that both contributions are of order h within each grid cell.

Theorem 4.1 (Parabolic function approximation). Let $B , \Lambda \ge 1 , 0 < \gamma \le 1$ , and let V be a family of functions $v \in C ( D _ { T } )$ satisfying

$$
\| v \| _ { \infty } \le B , \qquad | v ( z ) - v ( z ^ { \prime } ) | \le \Lambda d _ { \mathfrak { s } } ( z , z ^ { \prime } ) ^ { \gamma } .
$$

For every $\varepsilon _ { T } \in ( 0 , B ]$ with $\varepsilon _ { T } \leq 2 \Lambda$ , there are parabolic grid nodes $z _ { 1 } , \dotsc , z _ { N _ { 1 } } \in D _ { T }$ and ReLU networks $\widetilde { q } _ { 1 } , \dots , \widetilde { q } _ { N _ { 1 } } : \mathbb { R } ^ { 1 + d } \to \mathbb { R }$ such that

(27)

$$
\operatorname* { s u p } _ { v \in \mathcal { V } } \operatorname* { s u p } _ { z \in D _ { T } } \left| v ( z ) - \sum _ { \ell = 1 } ^ { N _ { 1 } } v ( z _ { \ell } ) \widetilde { q } _ { \ell } ( z ) \right| \le \varepsilon _ { T } ,\tag{28}
$$

$$
\operatorname* { s u p } _ { z \in D _ { T } } \sum _ { \ell = 1 } ^ { N _ { 1 } } | \widetilde { q } _ { \ell } ( z ) | \leq \frac { 3 } { 2 } .
$$

Moreover,

$$
N _ { 1 } \asymp \left( \frac { \Lambda } { \varepsilon _ { T } } \right) ^ { ( d + 2 ) / \gamma } .\tag{29}
$$

Each trunk network may be chosen with depth and number of nonzero parameters $O ( ( d +$ 1) $\log ( B N _ { 1 } / \varepsilon _ { T } ) )$ , with constants depending on the fixed domain and d.

Proof. We first construct an interpolant from finitely many point values, then approximate its basis functions by ReLU networks. Choose

$$
h : = \frac { 1 } { 1 + \sqrt { d } } \left( \frac { \varepsilon _ { T } } { 2 \Lambda } \right) ^ { 1 / \gamma } .
$$

Since $\varepsilon _ { T } \leq 2 \Lambda$ , we have $h < 1$ . Set

$$
n _ { t } : = \left\lceil \frac { T } { h ^ { 2 } } \right\rceil , \qquad n _ { x } : = \left\lceil \frac { 2 \pi } { h } \right\rceil , \qquad \tau : = \frac { T } { n _ { t } } , \qquad \sigma : = \frac { 2 \pi } { n _ { x } } .
$$

The temporal grid $t _ { i } = i \tau , 0 \leq i \leq n _ { t }$ , has spacing $\tau \leq h ^ { 2 }$ , and the periodic spatial grid $x _ { j } = j \sigma$ $0 \leq j < n _ { x } .$ , has spacing $\sigma \leq h$

For $0 \leq i \leq n _ { t }$ , let $\vartheta _ { i }$ be the continuous piecewise-linear interpolation function on [0, T] satisfying

$$
\vartheta _ { i } ( t _ { k } ) = \delta _ { i k } , \qquad 0 \leq k \leq n _ { t } ,
$$

and linear on each interval $[ t _ { k } , t _ { k + 1 } ]$ . Similarly, for $0 \leq j < n _ { x } .$ , let $\psi _ { j }$ be the continuous periodic piecewise-linear interpolation function satisfying

$$
\psi _ { j } ( x _ { k } ) = \delta _ { j k } , \qquad 0 \leq k < n _ { x } .
$$

Here each function equals one at its own grid point and zero at all the other grid points. The spatial functions are linear on each grid interval, including $[ x _ { n _ { x } - 1 } , 2 \pi ]$ , with 2π identified with 0. These functions satisfy

$$
\vartheta _ { i } , \psi _ { j } \geq 0 , \qquad \sum _ { i = 0 } ^ { n _ { t } } \vartheta _ { i } ( t ) = 1 , \qquad \sum _ { j = 0 } ^ { n _ { x } - 1 } \psi _ { j } ( x ) = 1 .
$$

For $\alpha = ( i , j _ { 1 } , \ldots , j _ { d } )$ , define

$$
z _ { \alpha } : = ( t _ { i } , x _ { j _ { 1 } } , \ldots , x _ { j _ { d } } ) , \qquad \varphi _ { \alpha } ( t , x ) : = \vartheta _ { i } ( t ) \prod _ { q = 1 } ^ { d } \psi _ { j _ { q } } ( x _ { q } ) .
$$

Then $\varphi _ { \alpha } \geq 0$ and $\textstyle \sum _ { \alpha } \varphi _ { \alpha } = 1$ . Moreover, on the support of $\varphi _ { \alpha } ,$ , the temporal displacement from $z _ { \alpha }$ is at most τ, and the spatial distance is at most $\sqrt { d } \sigma$ . Thus

$$
d _ { \mathfrak { s } } ( z , z _ { \alpha } ) \leq \sqrt { \tau } + \sqrt { d } \sigma \leq ( 1 + \sqrt { d } ) h .
$$

For $v \in \mathcal V$ , define

$$
I _ { h } v ( z ) : = \sum _ { \alpha } v ( z _ { \alpha } ) \varphi _ { \alpha } ( z ) .
$$

The partition-of-unity property and the assumed H¨older bound give

$$
\begin{array} { l } { | v ( z ) - I _ { h } v ( z ) | \leq \displaystyle \sum _ { \alpha : \varphi _ { \alpha } ( z ) > 0 } \varphi _ { \alpha } ( z ) | v ( z ) - v ( z _ { \alpha } ) | } \\ { \leq \Lambda ( ( 1 + \sqrt { d } ) h ) ^ { \gamma } = \displaystyle \frac { \varepsilon _ { T } } { 2 } . } \end{array}
$$

This estimate is uniform over $v \in \mathcal V$ and $z \in D _ { T }$

The number of grid nodes is

$$
N _ { 1 } = ( n _ { t } + 1 ) n _ { x } ^ { d } .
$$

Since $h < 1$ and the domain is fixed, $n _ { t } + 1 \asymp h ^ { - 2 }$ and $n _ { x } \asymp h ^ { - 1 }$ . Consequently,

$$
N _ { 1 } \asymp h ^ { - ( d + 2 ) } \asymp \left( \frac { \Lambda } { \varepsilon _ { T } } \right) ^ { ( d + 2 ) / \gamma } ,
$$

which proves (29). The comparison constants depend only on $T , d ,$ and $\gamma .$

We next approximate the interpolation basis functions by ReLU networks. The temporal interpolation functions have the explicit form

$$
\vartheta _ { i } ( t ) = \left( 1 - \frac { | t - t _ { i } | } { \tau } \right) _ { + } , \qquad t \in [ 0 , T ] .
$$

On the coordinate interval [0, 2π], the periodic spatial interpolation functions can be written as

$$
\psi _ { j } ( x ) = \sum _ { k = - 1 } ^ { 1 } \left( 1 - \frac { | x - x _ { j } - 2 \pi k | } { \sigma } \right) _ { + } .
$$

$\mathrm { B y }$ Lemma 3.9(2), each of these one-dimensional interpolation functions has an exact ReLU representation with fixed depth and a fixed number of nonzero parameters. Their values lie in [0, 1]. Each $\varphi _ { \alpha }$ is a product of d + 1 such factors. The product construction in Lemma 3.9 therefore gives, for every $\delta _ { T } \in ( 0 , 1 )$ , a ReLU network $\widetilde { q } _ { \alpha }$ satisfying

$$
\| \varphi _ { \alpha } - \widetilde { q } _ { \alpha } \| _ { C ( D _ { T } ) } \leq \delta _ { T } .
$$

Since $\psi _ { j } ( 0 ) = \psi _ { j } ( 2 \pi )$ for every $j ,$ replacing any spatial coordinate $x _ { q } = 0$ by $x _ { q } = 2 \pi$ , while keeping time and the other coordinates fixed, leaves all inputs to the product network unchanged. Thus $\widetilde { q } _ { \alpha }$ takes the same value at boundary points identified on $\mathbb { T } ^ { d }$ and defines a continuous function on $D _ { T }$ . Relabel the nodes and networks by $\ell = 1 , \ldots , N _ { 1 }$ . For every $v \in \mathcal V$ , the interpolation estimate and $| v ( z _ { \ell } ) | \le B$ imply

$$
\begin{array} { r l r } {  { \| v - \sum _ { \ell = 1 } ^ { N _ { 1 } } v ( z _ { \ell } ) \widetilde { q } _ { \ell } \| _ { C ( D _ { T } ) } \le \| v - I _ { h } v \| _ { C ( D _ { T } ) } + \displaystyle \sum _ { \ell = 1 } ^ { N _ { 1 } } | v ( z _ { \ell } ) | \| \varphi _ { \ell } - \widetilde { q } _ { \ell } \| _ { C ( D _ { T } ) } } } \\ & { } & { \le \frac { \varepsilon _ { T } } { 2 } + B N _ { 1 } \delta _ { T } . } \end{array}
$$

Choose

(30)

$$
\delta _ { T } : = \frac { \varepsilon _ { T } } { 2 B N _ { 1 } } .
$$

Since $\varepsilon _ { T } \leq B$ , this choice lies in (0, 1). The preceding bound is then at most $\varepsilon _ { T }$ , proving (27). Furthermore, for every $z \in D _ { T }$

$$
\begin{array} { c l l } { \displaystyle \sum _ { \ell = 1 } ^ { N _ { 1 } } | \widetilde { q } _ { \ell } ( z ) | \leq \sum _ { \ell = 1 } ^ { N _ { 1 } } \varphi _ { \ell } ( z ) + \sum _ { \ell = 1 } ^ { N _ { 1 } } | \widetilde { q } _ { \ell } ( z ) - \varphi _ { \ell } ( z ) | } \\ { \displaystyle \leq 1 + N _ { 1 } \delta _ { T } = 1 + \frac { \varepsilon _ { T } } { 2 B } \leq \frac { 3 } { 2 } . } \end{array}
$$

This proves (28).

Finally, Lemma 3.9 gives each product network depth and number of nonzero parameters bounded by

$$
O \bigg ( ( d + 1 ) \log { \frac { d + 1 } { \delta _ { T } } } \bigg ) .
$$

Combining this with the exact representations of the interpolation functions and substituting (30) gives $O ( ( d + 1 ) \log ( B N _ { 1 } / \varepsilon _ { T } ) )$ for fixed $d ,$ as claimed. □

## 4.2. Functional approximation: the branch network. For $y \in D _ { T }$ , define the evaluation functional

$$
\mathcal { F } _ { y } ( u _ { 0 } , \zeta ) : = S ( u _ { 0 } , \zeta ) ( y ) .
$$

The forcing $\zeta$ is represented by its Fourier measurements $c _ { J } ( \zeta )$ , which are well defined for distribution-valued inputs. Using the measurement map (22), we approximate the solution value $\mathcal { F } _ { y } ( u _ { 0 } , \zeta )$ as a function of the finite vector $( \vec { u } _ { 0 } , c _ { J } ( \zeta ) )$ by a partition of unity on a compact coeficient set. The same partition weights are used for every $y ;$ only the solution values $\Gamma _ { J } ( z ^ { k } ) ( y )$ depend on the output coordinate.

Fix $R \geq 1$ . Assumption 2.6 gives $| u _ { 0 } ( x _ { i } ) | \leq M$ for every initial-data measurement. On the event $\{ \| \eta \| _ { X _ { 0 } } \leq R \}$ , Lemma 3.3 gives

$$
\| c _ { J } ( \eta ) \| _ { \ell ^ { \infty } } \leq C 2 ^ { - J \beta _ { 0 } } R .
$$

We therefore set $R _ { J } = C 2 ^ { - J \beta _ { 0 } } R$ and define the compact box

$$
K _ { J , R } : = [ - M , M ] ^ { N _ { 0 } } \times [ - R _ { J } , R _ { J } ] ^ { N _ { J } } .
$$

Then $( \vec { u } _ { 0 } , c _ { J } ( \eta ) ) \in K _ { J , R }$ on this event. The partition of unity below is constructed on $K _ { J , R }$

We combine the reconstruction estimates with a partition of unity on $K _ { J , R }$ and ReLU approximation of its weights.

Theorem 4.2. Let $F : \mathbb { R }  \mathbb { R }$ be globally Lipschitz, and let S be the corresponding solution map from Theorem 2.5. Suppose that Assumptions 2.2 and 2.6 hold, and fix $\beta _ { 1 }$ and $\delta$ as in (2). Let $R \geq 1$ and $\varepsilon _ { B } \in ( 0 , 1 )$ be fixed. One can choose

$$
2 ^ { J } \leq C \left( \frac { R } { \varepsilon _ { B } } \right) ^ { 1 / \delta } , \qquad \delta _ { 0 } = c _ { 0 } \varepsilon _ { B } , \qquad r = \frac { \varepsilon _ { B } } { 8 L _ { J } } ,\tag{31}
$$

where $c _ { 0 } > 0$ depends only on $d , T , L _ { F }$ , and $L _ { 0 . }$ , and construct an $r / 2 .$ -net $( z ^ { k } ) _ { k \leq m }$ of $K _ { J , R }$ and $R e L U$ networks $\widetilde { \rho } _ { k } : \mathbb { R } ^ { N _ { 0 } + N _ { J } } $ R such that

$$
\operatorname* { s u p } _ { u _ { 0 } \in \mathcal { G } _ { 0 } } \operatorname* { s u p } _ { \| \zeta \| _ { \mathcal { X } _ { 0 } } \leq R } \operatorname* { s u p } _ { z _ { \bullet } \in D _ { T } } \left| S ( u _ { 0 } , \zeta ) ( z _ { \bullet } ) - \sum _ { k = 1 } ^ { m } \Gamma _ { J } ( z ^ { k } ) ( z _ { \bullet } ) \widetilde { \rho } _ { k } ( \vec { u } _ { 0 } , c _ { J } ( \zeta ) ) \right| \leq \varepsilon _ { B } .\tag{32}
$$

The networks can be chosen so that

$$
\sum _ { k = 1 } ^ { m } | \widetilde { \rho } _ { k } ( z ) | \leq 2 , \qquad z \in \mathbb { R } ^ { N _ { 0 } + N _ { J } } .\tag{33}
$$

Moreover, the number m of terms in the approximation can be chosen to satisfy

$$
m \leq \left( 1 + \frac { 2 ( M \vee R _ { J } ) \sqrt { N } } { r } \right) ^ { N } , \qquad N : = N _ { 0 } + N _ { J } .\tag{34}
$$

Choose a constant $C _ { B } \geq 1$ , depending only on $d , T , \beta _ { 0 } , \beta _ { 1 } , M , L _ { 0 } , L _ { F }$ , and $| F ( 0 ) |$ , so that $B J : =$ $C _ { B } ( 1 + N _ { J } R _ { J } )$ bounds the reference solutions $\Gamma _ { J } ( z ^ { k } )$ in Y, and set

$$
H _ { B } : = \log \left( \frac { e N m ( B _ { J } + 1 ) } { \varepsilon _ { B } } \right) .
$$

Each $\widetilde { \rho } _ { k }$ can be chosen with depth at most $C \log ( e N ) H _ { B }$ and at most $C N H _ { B }$ nonzero weights and biases. The vector-valued network $\widetilde { \pmb { \rho } } : = ( \widetilde { \rho } _ { 1 } , \dots , \widetilde { \rho } _ { m } )$ can be realized with

$$
L _ { \rho } \leq C \log ( e N ) H _ { B } , \qquad S _ { \rho } \leq C m N H _ { B } ,\tag{35}
$$

where $L _ { \rho }$ denotes its depth and $S _ { \rho }$ its total number of nonzero weights and biases. These bounds include the projection onto $K _ { J , R } ,$ the constant C is independent of J, $R , \varepsilon _ { B } , N$ , and m.

Proof. Fix $u _ { 0 } \in \mathcal { G } _ { 0 }$ and $\zeta \in X _ { 0 }$ with $\| \zeta \| _ { X _ { 0 } } \leq R .$ . Introduce

$$
\widehat { u } _ { 0 } : = \operatorname { R e c } _ { 0 } ( \vec { u } _ { 0 } ) , \qquad \zeta _ { J } : = \Pi _ { J } \zeta , \qquad z : = ( \vec { u } _ { 0 } , c _ { J } ( \zeta ) ) .
$$

By (10) and (19),

$$
\Gamma _ { J } ( z ) = { \cal S } ( \widehat { u } _ { 0 } , \zeta _ { J } ) .\tag{36}
$$

Moreover, $z \in K _ { J , R }$ by Assumption 2.6 and (15). For the Cartesian grid $( z ^ { k } ) _ { k \leq m }$ and the tensor-product partition $( \rho _ { k } ) _ { k \le m }$ constructed in the estimate of $I _ { 3 }$ below, set

$$
A _ { J , r } ( z ) : = \sum _ { k = 1 } ^ { m } \rho _ { k } ( z ) \Gamma _ { J } ( z ^ { k } ) .
$$

We decompose the approximation error into four terms. For any output point $z _ { \bullet } \in D _ { T }$ , we have

$$
\begin{array} { r l } & { S ( u _ { 0 } , \zeta ) ( z _ { \bullet } ) - \displaystyle \sum _ { k = 1 } ^ { m } \Gamma _ { J } ( z ^ { k } ) ( z _ { \bullet } ) \widetilde { \rho } _ { k } ( z ) \Bigg \vert } \\ & { \quad \le \underbrace { \| S ( u _ { 0 } , \zeta ) - S ( u _ { 0 } , \zeta _ { J } ) \| _ { Y } } _ { I _ { 1 } } + \underbrace { \| S ( u _ { 0 } , \zeta _ { J } ) - \Gamma _ { J } ( z ) \| _ { Y } } _ { I _ { 2 } } } \\ & { \qquad + \underbrace { \| \Gamma _ { J } ( z ) - A _ { J , r } ( z ) \| _ { Y } } _ { I _ { 3 } } + \underbrace { \left| \displaystyle \sum _ { k = 1 } ^ { m } \Gamma _ { J } ( z ^ { k } ) ( z _ { \bullet } ) ( \rho _ { k } ( z ) - \widetilde { \rho } _ { k } ( z ) ) \right| } _ { I _ { 4 } } . } \end{array}\tag{37}
$$

We estimate these terms separately.

Estimates of $I _ { 1 }$ . The two solutions in $I _ { 1 }$ have the same initial datum, so the stability estimate (6) and the cutof estimate (14) yield

$$
\begin{array} { r } { I _ { 1 } \leq e ^ { L _ { F } T } C _ { \beta _ { 1 } } \| ( I - \Pi _ { J } ) \zeta \| _ { X _ { 1 } } \leq C _ { 1 } 2 ^ { - J \delta } \| \zeta \| _ { X _ { 0 } } \leq C _ { 1 } 2 ^ { - J \delta } R , } \end{array}
$$

where $C _ { 1 } \geq 1$ is independent of $J , R ,$ , and ζ.

We choose J to be the smallest nonnegative integer satisfying

$$
2 ^ { J \delta } \geq \frac { 4 C _ { 1 } R } { \varepsilon _ { B } } .\tag{38}
$$

Then $I _ { 1 } \le \varepsilon _ { B } / 4$ . Since $4 C _ { 1 } R / \varepsilon _ { B } > 1$ , we have $J \geq 1$ . By the choice of $J ,$ the preceding integer $J - 1$ does not satisfy (38). Thus

$$
2 ^ { ( J - 1 ) \delta } < \frac { 4 C _ { 1 } R } { \varepsilon _ { B } } ,
$$

and consequently

$$
2 ^ { J } < 2 \left( \frac { 4 C _ { 1 } R } { \varepsilon _ { B } } \right) ^ { 1 / \delta } ,
$$

which is the first estimate in (31). Taking the smallest J is important because both $N _ { J }$ and the Lipschitz constant $L _ { J }$ increase with the cutof level.

Estimates of $I _ { 2 } .$ . By (36), the forcing in the two solutions defining $I _ { 2 }$ is the same. Hence (6) and (17) give

$$
I _ { 2 } = \| S ( u _ { 0 } , \zeta _ { J } ) - S ( \widehat { u } _ { 0 } , \zeta _ { J } ) \| _ { Y } \leq e ^ { L _ { F } T } \| u _ { 0 } - \widehat { u } _ { 0 } \| _ { \infty } \leq C _ { 2 } L _ { 0 } \delta _ { 0 } .
$$

Here $C _ { 2 } : = e ^ { L _ { F } T } C _ { d }$ , where $C _ { d }$ is the interpolation constant in (17).

Choose

$$
\delta _ { 0 } = c _ { 0 } \varepsilon _ { B } , \qquad c _ { 0 } : = ( 4 C _ { 2 } L _ { 0 } ) ^ { - 1 } .\tag{39}
$$

Then $I _ { 2 } \le \varepsilon _ { B } / 4$ , and the number of initial-data measurements satisfies

$$
N _ { 0 } \asymp \delta _ { 0 } ^ { - d } \asymp \varepsilon _ { B } ^ { - d } .
$$

Estimates of $I _ { 3 } ,$ . For $z _ { 1 } = ( \vec { v } , c )$ and $z _ { 2 } = ( \vec { v } ^ { \prime } , c ^ { \prime } )$ , one has

$$
\begin{array} { r l } & { | \vec { v } - \vec { v } ^ { \prime } | _ { \ell ^ { \infty } } + | c - c ^ { \prime } | _ { \ell ^ { 2 } } \leq | \vec { v } - \vec { v } ^ { \prime } | _ { \ell ^ { 2 } } + | c - c ^ { \prime } | _ { \ell ^ { 2 } } } \\ & { \qquad \leq 2 | z _ { 1 } - z _ { 2 } | _ { \ell ^ { 2 } } . } \end{array}
$$

Thus Lemma 3.5 shows that $\Gamma _ { J }$ is $2 L _ { J ^ { - } } \mathrm { I }$ ipschitz on the Euclidean box $K _ { J , R }$ . We construct an $r / 2 .$ -net of this box as follows. Set $N = N _ { 0 } + N _ { J }$ . For $a \in \{ M , R J \}$ , define

$$
n _ { a } : = \left\lceil \frac { 2 a \sqrt { N } } { r } \right\rceil , \qquad G _ { a } : = \left\{ - a + \frac { ( 2 j - 1 ) a } { n _ { a } } : j = 1 , \dots , n _ { a } \right\} .
$$

Let

$$
\{ z ^ { 1 } , \ldots , z ^ { m } \} : = G _ { M } ^ { N _ { 0 } } \times G _ { R _ { J } } ^ { N _ { J } } \subset K _ { J , R } .
$$

For every $z \in K _ { J , R } .$ one can choose a center $z ^ { k }$ such that

$$
| z - z ^ { k } | _ { \ell ^ { \infty } } \leq \frac { r } { 2 \sqrt { N } } .
$$

Consequently,

$$
| z - z ^ { k } | _ { \ell ^ { 2 } } \leq \sqrt { N } | z - z ^ { k } | _ { \ell ^ { \infty } } \leq \frac { r } { 2 } ,
$$

so these centers form an $r / 2 .$ -net of $K _ { J , R }$ . Moreover,

$$
m = n _ { M } ^ { N _ { 0 } } n _ { R _ { J } } ^ { N _ { J } } \leq \left( 1 + \frac { 2 ( M \vee R _ { J } ) \sqrt { N } } { r } \right) ^ { N } ,
$$

which proves $( 3 4 )$ . We now define interpolation weights on this grid. Write $G _ { a } = \{ \xi _ { a , j } : 1 \leq j \leq$ $n _ { a } \}$ in increasing order and set $s _ { a } : = 2 a / n _ { a } \le r / \sqrt { N }$ . If $n _ { a } = 1$ , set $b _ { a , 1 } \equiv 1$ . If $n _ { a } \geq 2 .$ , define

$$
b _ { a , 1 } ( s ) : = P \biggl ( \frac { \xi _ { a , 2 } - s } { s _ { a } } \biggr ) ,
$$

$$
b _ { a , n _ { a } } ( s ) : = P \biggl ( \frac { s - \xi _ { a , n _ { a } - 1 } } { s _ { a } } \biggr ) ,
$$

$$
b _ { a , j } ( s ) : = \mathopen { } \mathclose \bgroup \left( 1 - \frac { \mathopen { } \mathclose \bgroup \left| s - \xi _ { a , j } \aftergroup \egroup \right| } { s _ { a } } \aftergroup \egroup \right) _ { + } , \qquad 1 < j < n _ { a } ,
$$

where

$$
P ( v ) : = \left\{ { \begin{array} { l l } { 0 , } & { v < 0 , } \\ { v , } & { 0 \leq v \leq 1 , } \\ { 1 , } & { v > 1 . } \end{array} } \right.
$$

These functions interpolate the grid values and are extended constantly between the first or last grid point and the corresponding boundary. For $s \in [ - a , a ]$ , they satisfy

$$
b _ { a , j } ( s ) \geq 0 , \qquad \sum _ { j = 1 } ^ { n _ { a } } b _ { a , j } ( s ) = 1 , \qquad b _ { a , j } ( s ) > 0 \implies | s - \xi _ { a , j } | \leq s _ { a } .
$$

The last bound also holds when $n _ { a } = 1$ , since then $\xi _ { a , 1 } = 0$ and $| s | \leq a \leq s _ { a }$

Set $a _ { q } \ = \ M$ for $1 \leq q \leq N _ { 0 }$ and $a _ { q } \ = \ R _ { J }$ for $N _ { 0 } < q \le N$ . Index the grid points by ${ \boldsymbol { \alpha } } = ( j _ { 1 } , \dots , j _ { N } )$ and define

$$
z ^ { \alpha } : = ( \xi _ { a _ { q } , j _ { q } } ) _ { q = 1 } ^ { N } , \qquad \rho _ { \alpha } ( w ) : = \prod _ { q = 1 } ^ { N } b _ { a _ { q } , j _ { q } } ( w _ { q } ) .\tag{40}
$$

For each center $z ^ { k } = z ^ { \alpha }$ , set $\rho _ { k } : = \rho _ { \alpha }$ . On $K _ { J , R }$ , these weights are nonnegative and sum to one. Moreover, for $w \in K _ { J , R } ,$

$$
\rho _ { k } ( w ) > 0 \quad \Longrightarrow \quad | w - z ^ { k } | _ { \ell ^ { 2 } } \leq \left( \sum _ { q = 1 } ^ { N } s _ { a _ { q } } ^ { 2 } \right) ^ { 1 / 2 } \leq r .
$$

The 2L<sub>J</sub>-Lipschitz continuity of $\Gamma _ { J }$ therefore gives

$$
I _ { 3 } \leq \sum _ { k = 1 } ^ { m } \rho _ { k } ( z ) \| \Gamma _ { J } ( z ) - \Gamma _ { J } ( z ^ { k } ) \| _ { Y } \leq 2 L _ { J } r .
$$

Choosing

$$
r : = \frac { \varepsilon _ { B } } { 8 L _ { J } }\tag{41}
$$

therefore gives $I _ { 3 } \leq \varepsilon _ { B } / 4$

Estimates of $I _ { 4 } .$ . Write $z ^ { k } = ( \vec { v } ^ { k } , c ^ { k } )$ . Since $z ^ { k } \in K _ { J , R } .$ , each entry $v _ { i } ^ { k }$ belongs to $[ - M , M ]$ . At every spatial point, $\operatorname { R e c } _ { 0 } ( \vec { v } ^ { k } )$ is a weighted average of these grid values, with nonnegative weights summing to one. Hence

$$
\| \operatorname { R e c } _ { 0 } ( \vec { v } ^ { k } ) \| _ { \infty } \leq M .
$$

Similarly, $| c ^ { k } | _ { \ell ^ { 2 } } \leq \sqrt { N _ { J } } R _ { J }$ , and (16) gives

$$
\| \mathrm { R e c } _ { J } ( c ^ { k } ) \| _ { X _ { 1 } } \le C N _ { J } ^ { 1 / 2 } | c ^ { k } | _ { \ell ^ { 2 } } \le C N _ { J } R _ { J } .
$$

The a priori estimate (7) now implies, for a constant $C _ { B } \geq 1$ depending only on the parameters specified above,

$$
\operatorname* { m a x } _ { k \leq m } \| \Gamma _ { J } ( z ^ { k } ) \| _ { Y } \leq B _ { J } : = C _ { B } \big ( 1 + N _ { J } R _ { J } \big ) .\tag{42}
$$

Each one-dimensional function $b _ { a , j }$ has an exact ReLU representation of fixed depth with a fixed number of nonzero parameters: the endpoint functions use Lemma 3.9(1), the interior functions use part (2), and the case $n _ { a } = 1$ is constant. All their values lie in [0, 1]. Each weight in (40) is a product of N such factors. The product construction in Lemma 3.9 therefore gives, for every $\delta _ { B } \in ( 0 , 1 )$ , ReLU networks $\widehat { \rho } _ { k }$ with values in [0, 1] such that

$$
\operatorname* { m a x } _ { k \leq m } \| \rho _ { k } - \widehat { \rho } _ { k } \| _ { C ( K _ { J , R } ) } \leq \delta _ { B } .\tag{43}
$$

For $N = 1$ , the factor itself gives an exact representation. In all cases, each network $\widehat { \rho } _ { k }$ can be realized with

$$
\mathrm { d e p t h } ( \widehat { \rho } _ { k } ) \leq C \log ( e N ) \log \frac { e N } { \delta _ { B } } ,\tag{44}
$$

$$
\mathrm { s i z e } ( \widehat { \rho } _ { k } ) \leq C N \log \frac { e N } { \delta _ { B } } ,
$$

where size denotes the total number of nonzero weights and biases. The m networks can be realized at the same depth, using the same product construction and fixed-depth representations

of their one-dimensional factors. The projections onto [0, 1] in the product construction ensure that the outputs belong to [0, 1].

Let $P _ { K _ { J , R } }$ map each coordinate to the nearest point in its defining interval in $K _ { J , R }$ . This map leaves every point of $K _ { J , R }$ unchanged and maps every input into $K _ { J , R }$ . Define

$$
\widetilde { \rho } _ { k } : = \widehat { \rho } _ { k } \circ { \cal P } _ { K _ { J , R } } .\tag{45}
$$

Lemma 3.9(1) shows that this is again a ReLU network. Since $P _ { K _ { J , R } }$ is the identity on $K _ { J , R }$ (43) gives, for the last term in (37),

$$
\begin{array} { l } { I _ { 4 } \leq \displaystyle \sum _ { k = 1 } ^ { m } \| \Gamma _ { J } ( z ^ { k } ) \| _ { Y } | \rho _ { k } ( z ) - \widetilde { \rho } _ { k } ( z ) \| } \\ { \leq m B _ { J } \delta _ { B } . } \end{array}
$$

Set

$$
\delta _ { B } : = \frac { \varepsilon _ { B } } { 4 m ( B _ { J } + 1 ) } .\tag{46}
$$

Then $I _ { 4 } \le \varepsilon _ { B } / 4$ . The same choice gives, for every $w \in \mathbb { R } ^ { N _ { 0 } + N _ { J } }$ ，

$$
\begin{array} { l } { \displaystyle \sum _ { k = 1 } ^ { m } | \widetilde { \rho } _ { k } ( w ) | \leq \sum _ { k = 1 } ^ { m } \rho _ { k } ( P _ { K _ { J , R } } w ) + m \delta _ { B } } \\ { \leq 1 + \displaystyle \frac { \varepsilon _ { B } } { 4 ( B _ { J } + 1 ) } \leq 2 , } \end{array}
$$

which proves (33). For the choice (46),

$$
\log { \frac { e N } { \delta _ { B } } } = \log { \frac { 4 e N m ( B _ { J } + 1 ) } { \varepsilon _ { B } } } \leq C H _ { B } .
$$

The coordinate projection requires $O ( N )$ nonzero parameters and fixed depth. Thus (44) gives the stated bounds for each $\widetilde { \rho } _ { k }$ . Running these networks in parallel gives (35). Since their outputs are nonnegative, their final afine outputs may be treated as ReLU hidden activations. Appending an identity output map adds at most m parameters and one layer, which are absorbed by the same bounds.

Combining the estimates from (38), (39), (41), and (46), give $I _ { i } \le \varepsilon _ { B } / 4$ for $1 \leq i \leq 4$ Summing (37) proves (32), uniformly in u<sub>0</sub>, ζ, and $z _ { \bullet }$ □

4.3. Operator approximation. We now combine the function approximation of Theorem 4.1 with the functional approximation of Theorem 4.2. This is the SPDE counterpart of the final operator approximation step in [5].

Theorem 4.3. Let $d \geq 2$ , let Assumptions 2.2 and 2.6 hold, and let F be globally Lipschitz. Choose $\beta _ { 1 }$ as in (2) and γ as in (8). For every $\varepsilon \in ( 0 , 1 )$ there are a cutof level J, N<sub>0</sub> initial-data measurements, and ReLU branch and trunk networks such that

$$
\operatorname* { s u p } _ { u _ { 0 } \in \mathcal { G } _ { 0 } } \mathbb { E } \left[ \operatorname* { s u p } _ { ( t , x ) \in D _ { T } } \bigg | S ( u _ { 0 } , \eta ) ( t , x ) - \sum _ { \ell = 1 } ^ { N _ { 1 } } \widetilde { a } _ { \ell } ( \vec { u } _ { 0 } , c _ { J } ( \eta ) ) \widetilde { q } _ { \ell } ( t , x ) \bigg | \right] \leq \varepsilon .\tag{47}
$$

The parameters may be chosen so that, with $R = C _ { 0 } \sqrt { \log ( e / \varepsilon ) }$ 2

(48)

$$
N _ { J } \leq C \left( \frac { R } { \varepsilon } \right) ^ { ( d + 2 ) / \delta } ,\tag{49}
$$

$$
N _ { 0 } \leq C \varepsilon ^ { - d } ,
$$

$$
N _ { 1 } \leq C \left( \frac { 1 + R } { \varepsilon } \right) ^ { ( d + 2 ) / \gamma } .\tag{50}
$$

For each trunk network, its depth $L _ { T , \ell }$ and number $S _ { T , \ell }$ of nonzero weights and biases satisfy

$$
L _ { T , \ell } \leq C \log ( e / \varepsilon ) , \qquad S _ { T , \ell } \leq C \log ( e / \varepsilon ) , \qquad 1 \leq \ell \leq N _ { 1 } .\tag{51}
$$

The branch networks can be realized as the coordinates of a single network $\widetilde { \pmb { a } } : \mathbb { R } ^ { N }  \mathbb { R } ^ { N _ { 1 } }$ , where $N = N _ { 0 } + N _ { J }$ . This network uses m shared partition weights, with

$$
m \leq \left( 1 + \frac { 2 ( M \vee R _ { J } ) \sqrt { N } } { r } \right) ^ { N } , \qquad r = \frac { \varepsilon } { 4 8 L _ { J } } .\tag{52}
$$

Set $B _ { J } : = C _ { B } ( 1 + N _ { J } R _ { J } )$ , with $C _ { B }$ as in Theorem $4 . 2 ,$ and define

$$
H _ { \varepsilon } : = \log \left( \frac { e N m ( B _ { J } + 1 ) } { \varepsilon } \right) .
$$

The shared branch network has depth $L _ { B }$ and total number $S _ { B }$ of nonzero weights and biases bounded by

$$
L _ { B } \le C \log ( e N ) H _ { \varepsilon } , \qquad S _ { B } \le C \bigl ( m N H _ { \varepsilon } + m N _ { 1 } \bigr ) .\tag{53}
$$

Here the shared partition weights are counted once, and m $N _ { 1 }$ counts the coeficients in the final linear map. Consequently, the combined number of nonzero parameters in the branch and trunk networks is

$$
S _ { B } + \sum _ { \ell = 1 } ^ { N _ { 1 } } S _ { T , \ell } \le C \bigl ( m N H _ { \varepsilon } + m N _ { 1 } + N _ { 1 } \log ( e / \varepsilon ) \bigr ) .\tag{54}
$$

The constants in these size bounds are independent $o f \varepsilon$

Proof. Construction and uniform approximation. For a radius $R \geq 1$ , to be chosen below, define

$$
\Omega _ { R } : = \{ \| \eta \| _ { X _ { 0 } } \leq R \} .
$$

We first construct an approximation whose error on $\Omega _ { R }$ is uniformly bounded by $\varepsilon / 2$

Let $\varepsilon _ { T }$ denote the error in reconstructing an output function from the trunk nodes, and let $\varepsilon _ { B }$ denote the error in approximating its value at each trunk node. The latter is multiplied by the $\ell ^ { 1 }$ norm of the trunk basis. In view of (28), it is enough to require

$$
\varepsilon _ { T } + \frac 3 2 \varepsilon _ { B } \leq \frac \varepsilon 2 .\tag{55}
$$

We use the convenient choice

$$
\varepsilon _ { T } : = \frac { \varepsilon } { 4 } , \qquad \varepsilon _ { B } : = \frac { \varepsilon } { 6 } ,\tag{56}
$$

for which equality holds in (55).

Consider the class of output functions generated on the noise ball,

$$
\mathcal { V } _ { R } : = \{ S ( u _ { 0 } , \zeta ) : u _ { 0 } \in \mathcal { G } _ { 0 } , \zeta \in X _ { 0 } , \| \zeta \| _ { X _ { 0 } } \le R \} .
$$

Lemma 2.7 gives, for every $v \in { \mathcal { V } } R \cdot$

$$
\| v \| _ { \infty } \le B ( R ) , \qquad | v ( y ) - v ( y ^ { \prime } ) | \le \Lambda ( R ) d _ { 5 } ( y , y ^ { \prime } ) ^ { \gamma } .
$$

Apply Theorem 4.1 to $\gamma _ { R }$ with the accuracy $\varepsilon _ { T }$ in (56); the hypothesis $\varepsilon _ { T } \leq 2 \Lambda ( R )$ holds because $\varepsilon _ { T } < 1 \leq \Lambda ( R )$ , and the companion condition $\varepsilon _ { T } \leq B ( R )$ follows from $B ( R ) \geq 1$ . We obtain nodes $y _ { 1 } , \ldots , y _ { N _ { 1 } }$ and trunk networks $\widetilde { q } _ { 1 } , \ldots , \widetilde { q } _ { N _ { 1 } }$ satisfying

(57)

$$
\operatorname* { s u p } _ { v \in \mathcal { V } _ { R } } \left\| v - \sum _ { \ell = 1 } ^ { N _ { 1 } } v ( y _ { \ell } ) \widetilde { q } _ { \ell } \right\| _ { Y } \leq \varepsilon _ { T } ,\tag{58}
$$

$$
\operatorname* { s u p } _ { y \in D _ { T } } \sum _ { \ell = 1 } ^ { N _ { 1 } } | \widetilde { q } _ { \ell } ( y ) | \leq \frac { 3 } { 2 } .
$$

The nodes and the trunk networks depend on $R$ and ε, but they do not depend on a particular u<sub>0</sub> or realization of the forcing.

Next, apply Theorem 4.2 with the same radius R and with accuracy $\varepsilon _ { B }$ from (56). This determines $J , \delta _ { 0 }$ , the compact coeficient box $K _ { J , R } ,$ , centers $z ^ { 1 } , \ldots , z ^ { m }$ , and common partition networks $\widetilde { \rho } _ { 1 } , \ldots , \widetilde { \rho } _ { m }$ . The construction is uniform in the output point. In particular, for every $y \in D _ { T }$ ，

$$
\operatorname* { s u p } _ { u _ { 0 } \in \mathcal { G } _ { 0 } } \operatorname* { s u p } _ { \| \zeta \| _ { X _ { 0 } } \leq R } \left| \mathcal { S } ( u _ { 0 } , \zeta ) ( y ) - \sum _ { k = 1 } ^ { m } \Gamma _ { J } ( z ^ { k } ) ( y ) \widetilde { \rho } _ { k } ( \vec { u } _ { 0 } , c _ { J } ( \zeta ) ) \right| \leq \varepsilon _ { B } .\tag{59}
$$

For each trunk node $y _ { \ell } ,$ define the branch network

$$
\widetilde { \boldsymbol { a } } _ { \ell } ( \boldsymbol { w } ) : = \sum _ { k = 1 } ^ { m } \Gamma _ { J } ( \boldsymbol { z } ^ { k } ) ( \boldsymbol { y } _ { \ell } ) \widetilde { \rho } _ { k } ( \boldsymbol { w } ) , \qquad \boldsymbol { w } \in \mathbb { R } ^ { N _ { 0 } + N _ { J } } .\tag{60}
$$

These branch networks can be combined into a single vector-valued network:

$$
\widetilde { \pmb { a } } ( w ) = A \widetilde { \pmb { \rho } } ( w ) , \qquad A _ { \ell k } : = \Gamma _ { J } ( z ^ { k } ) ( y _ { \ell } ) , \quad A \in \mathbb { R } ^ { N _ { 1 } \times m } .
$$

Theorem 4.2 realizes the shared weights at a common depth with nonnegative outputs as the final hidden activations. Appending the linear map A therefore gives a ReLU network with at most $m N _ { 1 }$ additional nonzero parameters and zero output bias. Evaluating (59) at $y = y \ell$ yields

$$
\operatorname* { m a x } _ { \ell \leq N _ { 1 } } | \mathcal { S } ( u _ { 0 } , \zeta ) ( y _ { \ell } ) - \widetilde { a } _ { \ell } ( \vec { u } _ { 0 } , c _ { J } ( \zeta ) ) | \leq \varepsilon _ { B }\tag{61}
$$

whenever $u _ { 0 } \in \mathcal { G } _ { 0 }$ and $\| \zeta \| _ { X _ { 0 } } \leq R$

We now combine these approximations. Fix $u _ { 0 } \in \mathcal { G } _ { 0 }$ and a forcing realization in $\Omega _ { R }$ , and write $u : = S ( u _ { 0 } , \eta )$ . For every $y \in D _ { T }$ , add and subtract the trunk interpolant with the exact nodal values:

$$
\begin{array} { r l r } {  { u ( y ) - \sum _ { \ell = 1 } ^ { N _ { 1 } } \widetilde { u } _ { \ell } \bigl ( \vec { u } _ { 0 } , c _ { J } ( \eta ) \bigr ) \widetilde { q } _ { \ell } ( y ) \Bigg | } } \\ & { } & \\ & { } & { \leq | u ( y ) - \sum _ { \ell = 1 } ^ { N _ { 1 } } u ( y _ { \ell } ) \widetilde { q } _ { \ell } ( y ) | } \\ & { } & { + \sum _ { \ell = 1 } ^ { N _ { 1 } } | u ( y _ { \ell } ) - \widetilde { a } _ { \ell } ( \vec { u } _ { 0 } , c _ { J } ( \eta ) ) | | \widetilde { q } _ { \ell } ( y ) | . } \end{array}
$$

The first term is bounded by (57); the second is bounded using (61) and (58). Taking the supremum over y gives

$$
\left\| u - \sum _ { \ell = 1 } ^ { N _ { 1 } } \widetilde { a } _ { \ell } \big ( \vec { u } _ { 0 } , c _ { J } ( \eta ) \big ) \widetilde { q } _ { \ell } \right\| _ { Y } \leq \varepsilon _ { T } + \frac { 3 } { 2 } \varepsilon _ { B } \leq \frac { \varepsilon } { 2 } .\tag{62}
$$

This is the deterministic branch–trunk approximation on $\Omega _ { R }$

Expected approximation error. The coordinatewise projection in (45) ensures that the branch network remains bounded even when the encoded forcing lies outside $K _ { J , R }$ . Indeed, (42), (33), and (60) give, for every $w \in \mathbb { R } ^ { N _ { 0 } + N _ { J } }$ ，

$$
| \widetilde { a } _ { \ell } ( w ) | \leq B _ { J } \sum _ { k = 1 } ^ { m } | \widetilde { \rho } _ { k } ( w ) | \leq 2 B _ { J } .
$$

Combining this with (58),

$$
\operatorname* { s u p } _ { y \in D _ { T } } \left. \sum _ { \ell = 1 } ^ { N _ { 1 } } \widetilde { a } _ { \ell } ( w ) \widetilde { q } _ { \ell } ( y ) \right. \leq 3 B _ { J } .\tag{63}
$$

We next express $B _ { J }$ in terms of R and ε. From (42), the definition of $R _ { J }$ , and (11),

$$
\begin{array} { c } { { B _ { J } \leq C \big ( 1 + N _ { J } 2 ^ { - J \beta _ { 0 } } R \big ) } } \\ { { \leq C \big ( 1 + R 2 ^ { J ( d + 2 - \beta _ { 0 } ) } \big ) . } } \end{array}
$$

Since $\beta _ { 0 } > - 2$ , one has $d + 2 - \beta _ { 0 } < d + 4$ . The choice of J in Theorem 4.2, with $\varepsilon _ { B } = \varepsilon / 6$ therefore implies

$$
B J \leq C \left[ 1 + R \left( \frac { R } { \varepsilon } \right) ^ { ( d + 4 ) / \delta } \right] .\tag{64}
$$

Let

$$
\mathcal E _ { \theta } ( u _ { 0 } , \eta ) : = \left. \boldsymbol S ( u _ { 0 } , \eta ) - \sum _ { \ell = 1 } ^ { N _ { 1 } } \widetilde a _ { \ell } ( \vec { u } _ { 0 } , c _ { J } ( \eta ) ) \widetilde q _ { \ell } \right. _ { Y } .
$$

For each fixed $u _ { 0 } \in \mathcal { G } _ { 0 }$ , the map $\eta \mapsto \mathcal { E } _ { \theta } ( u _ { 0 } , \eta )$ is Borel measurable. Indeed, the measurement map is continuous on $X _ { 0 } ,$ the solution map S is Lipschitz from $C (  { \mathbb { T } } ^ { d } ) \times X _ { 1 }$ to the separable space $Y ,$ , the embedding $X _ { 0 } \hookrightarrow X _ { 1 }$ is continuous, and the norm on Y is continuous. Thus the expectations below are well defined. The a priori estimate (7) and (63) imply

$$
\mathcal { E } _ { \theta } ( u _ { 0 } , \eta ) \leq C \big ( 1 + B _ { J } + \| \eta \| _ { X _ { 0 } } \big ) ,\tag{65}
$$

uniformly in $u _ { 0 } \in \mathcal { G } _ { 0 }$ . Set $p : = ( d + 4 ) / \delta + 1$ . By (64), after increasing $C ,$

$$
1 + B _ { J } \leq C \varepsilon ^ { - p } ( 1 + R ^ { p } ) .
$$

Lemmas 2.3 and 2.3, applied to (65), now give constants $c _ { 1 } , C > 0$ such that

$$
\operatorname* { s u p } _ { u _ { 0 } \in \mathcal { G } _ { 0 } } \mathbb { E } [ \mathcal { E } _ { \theta } ( u _ { 0 } , \eta ) \mathbf { 1 } _ { \Omega _ { R } ^ { c } } ] \leq C \varepsilon ^ { - p } ( 1 + R ^ { p } ) e ^ { - c _ { 1 } R ^ { 2 } } .\tag{66}
$$

The constants $C , c _ { 1 }$ are independent of $R \geq 1$ . We now choose

$$
R : = C _ { 0 } { \sqrt { \log ( e / \varepsilon ) } } .
$$

The right-hand side of (66) is bounded by

$$
C \varepsilon ^ { - p } \big ( 1 + C _ { 0 } ^ { p } \log ( e / \varepsilon ) ^ { p / 2 } \big ) ( e / \varepsilon ) ^ { - c _ { 1 } C _ { 0 } ^ { 2 } } .
$$

Taking $C _ { 0 }$ suficiently large, depending only on the fixed data, makes $c _ { 1 } C _ { 0 } ^ { 2 } > p + 2$ and absorbs the remaining constant and logarithmic factor. The expression is then at most $\varepsilon / 2$ for $\varepsilon \in ( 0 , 1 )$ Together with (62), this yields

$$
\begin{array} { r l } & { \underset { u _ { 0 } \in \mathcal { G } _ { 0 } } { \operatorname* { s u p } } ~ \mathbb { E } [ \mathcal { E } _ { \theta } ( u _ { 0 } , \eta ) ] \leq \frac { \varepsilon } { 2 } \mathbb { P } ( \Omega _ { R } ) + \underset { u _ { 0 } \in \mathcal { G } _ { 0 } } { \operatorname* { s u p } } ~ \mathbb { E } [ \mathcal { E } _ { \theta } ( u _ { 0 } , \eta ) \mathbf { 1 } _ { \Omega _ { R } ^ { c } } ] } \\ & { \quad \quad \quad \leq \varepsilon , } \end{array}
$$

which proves (47).

Measurement counts and network sizes. With $\varepsilon _ { B } = \varepsilon / 6$ , the minimal cutof choice (38) gives

$$
2 ^ { J } \leq C \left( \frac { R } { \varepsilon } \right) ^ { 1 / \delta } .
$$

Using $N _ { J } \asymp 2 ^ { J ( d + 2 ) }$ proves (48). Likewise, (39) gives $\delta _ { 0 } \asymp \varepsilon$ , and the periodic grid count $N _ { 0 } \times \delta _ { 0 } ^ { - d }$ proves (49). Finally, Lemma 2.7 gives $\Lambda ( R ) \leq C ( 1 + R )$ , so (29) with $\varepsilon _ { T } = \varepsilon / 4$ proves (50).

For each trunk network, Theorem 4.1 bounds its depth and number of nonzero parameters by

$$
C \log \left( \frac { 4 B ( R ) N _ { 1 } } { \varepsilon } \right) .
$$

Since $B ( R ) \leq C ( 1 + R )$ , (50) implies that this is at most $C \log ( e ( 1 + R ) / \varepsilon )$ , with d and $\gamma$ fixed.   
The choice $R = C _ { 0 } \sqrt { \log ( e / \varepsilon ) }$ then gives (51).

For the shared branch network, the choice $\varepsilon _ { B } = \varepsilon / 6$ gives $r = \varepsilon / ( 4 8 L _ { J } )$ and hence (52). It also gives

$$
H _ { B } = \log \left( \frac { 6 e N m ( B _ { J } + 1 ) } { \varepsilon } \right) \leq C H _ { \varepsilon } .
$$

Thus (35) bounds the depth and number of parameters in the shared feature network by $C \log ( e N ) H _ { \varepsilon }$ and $C m N H _ { \varepsilon } .$ , respectively. The final linear map in (60) adds at most m $N _ { 1 }$ nonzero parameters and one layer. This proves (53). Summing the branch and trunk parameter counts proves (54).

Corollary 4.4. Under the assumptions of Theorem $4 . 3 ,$ let $q \in [ 1 , \infty )$ . For every $\varepsilon \in ( 0 , 1 )$ there are finite measurement maps and ReLU branch and trunk networks such that

$$
\operatorname* { s u p } _ { u _ { 0 } \in \mathcal { G } _ { 0 } } \left( \mathbb { E } \left[ \left\| \mathcal { S } ( u _ { 0 } , \eta ) - \sum _ { \ell = 1 } ^ { N _ { 1 } } \widetilde { a } _ { \ell } ( \vec { u } _ { 0 } , c _ { J } ( \eta ) ) \widetilde { q } _ { \ell } \right\| _ { Y } ^ { q } \right] \right) ^ { 1 / q } \leq \varepsilon .\tag{67}
$$

The measurement and trunk counts satisfy (48)–(50), and the network depths and parameter counts satisfy (51), (53), and (54), with

$$
R = C _ { q } ^ { \prime } \sqrt { \log ( e / \varepsilon ) } .
$$

Here $C _ { q } ^ { \prime } \geq 1$ , the quantities $N , m , B _ { J } ,$ and $H _ { \varepsilon }$ are defined as in Theorem 4.3 for this choice of $R ,$ and the constants may depend on $q .$

Proof. Use the construction in Theorem 4.3, with the deterministic error on $\Omega _ { R }$ bounded by $\varepsilon / 2$ and denote the resulting random error by ${ \mathcal { E } } _ { \theta }$ . Then

$$
\mathcal { E } _ { \theta } \mathbf { 1 } _ { \Omega _ { R } } \leq \frac { \varepsilon } { 2 } .
$$

On $\Omega _ { R } ^ { c } ,$ the coordinatewise projection onto $K _ { J , R }$ , the bound $B _ { J }$ for the reference solutions, and the a priori estimate give

$$
\mathcal { E } _ { \theta } \leq C \big ( B _ { J } + 1 + \| \eta \| _ { X _ { 0 } } \big ) .
$$

For $p _ { 0 } : = ( d + 4 ) / \delta + 1$ , equations (64) and (31) imply $B _ { J } + 1 \le C \varepsilon ^ { - p _ { 0 } } ( 1 + R ^ { p _ { 0 } } )$ . Since $B J + 1 \geq 1$ , the deterministic factorization $( B _ { J } + 1 + \| \eta \| _ { X _ { 0 } } ) ^ { q } \leq ( B _ { J } + 1 ) ^ { q } ( 1 + \| \eta \| _ { X _ { 0 } } ) ^ { q }$ , together with Lemma 2.3 at exponent $q$ and $( 1 + R ^ { p _ { 0 } } ) ^ { q } \leq 2 ^ { q } ( 1 + R ^ { q p _ { 0 } } )$ , therefore yields

$$
\mathbb { E } \left[ \mathcal { E } _ { \theta } ^ { q } \mathbf { 1 } _ { \Omega _ { R } ^ { c } } \right] \leq C _ { q } \varepsilon ^ { - q p _ { 0 } } ( 1 + R ^ { q p _ { 0 } } ) e ^ { - c _ { q } R ^ { 2 } } .
$$

Choose $R = C _ { q } ^ { \prime } \sqrt { \log ( e / \varepsilon ) }$ with $C _ { q } ^ { \prime }$ large enough that the right-hand side is at most $( \varepsilon / 2 ) ^ { q }$ . Adding the good- and bad-event contributions and taking the qth root gives the bound $2 ^ { 1 / q } \varepsilon / 2 \leq \varepsilon .$ since $q \geq 1 ;$ this proves (67). With this choice of $R ,$ the measurement and network size estimates follow by the same calculations as in Theorem 4.3, with constants that may depend on $q .$

## 5. Local approximation for stochastic Navier–Stokes equations

We consider the stochastic incompressible Navier–Stokes equations on $ { \mathbb { T } } ^ { d } , d \geq 2$ . In contrast to the globally Lipschitz nonlinearity in Section 2, the transport term is quadratic and contains a spatial derivative. We establish local existence and Lipschitz stability on a common interval for a bounded set of initial velocities and forcing convolutions. The Fourier measurement maps and partition of unity from Section 3 then give an approximation uniform on this set. Gaussian tail estimates determine the probability that the random forcing satisfies the required bound.

## 5.1. The stochastic equation and solution spaces. Fix $\nu > 0$ . Consider

$$
\begin{array} { r } { \left\{ \begin{array} { l l } { \partial _ { t } u + ( u \cdot \nabla ) u + \nabla p = \nu \Delta u + \eta , \quad ( t , x ) \in ( 0 , T ] \times \mathbb { T } ^ { d } , } \\ { \nabla \cdot u = 0 , \qquad u ( 0 ) = u _ { 0 } . } \end{array} \right. } \end{array}\tag{68}
$$

Here $\eta$ is a centered Gaussian Borel random variable with values in $( \mathcal { C } _ { g } ^ { \beta _ { 0 } } ( \Lambda _ { T } ) ) ^ { d }$ , as in Assumption 2.2, interpreted for vector-valued forcing. Its realizations may be distributions rather than functions. Equation (68) is interpreted pathwise, after restriction of the forcing to $D _ { T }$ . Both $u _ { 0 }$ and the forcing are assumed to be divergence-free and to have zero spatial mean; for $\eta ,$ these conditions hold almost surely. For a distributional forcing, these conditions are understood in the sense of distributions. Let P be the Leray projector defined in Section 2.1. For deterministic forcing $\phi ,$ , the mild formulation is

$$
\begin{array} { r l r } & { } & { \displaystyle { u ( t ) = e ^ { \nu t \Delta } u _ { 0 } + K _ { \nu } \phi ( t ) - N ( u , u ) ( t ) , } } \\ & { } & { \displaystyle { N ( u , v ) ( t ) : = \int _ { 0 } ^ { t } e ^ { \nu ( t - \sigma ) \Delta } P \nabla \cdot ( u \otimes v ) ( \sigma ) d \sigma . } } \end{array}\tag{69}
$$

We will apply this formulation both to realizations of $\eta$ and to their Fourier approximations. Fix $s \in ( 0 , 1 )$ and define

$$
\begin{array} { r } { \mathcal { C } _ { x } ^ { s } : = B _ { \infty , \infty } ^ { s } (  { \mathbb { T } } ^ { d } ;  { \mathbb { R } } ^ { d } ) , \qquad \mathcal { C } _ { g , x } ^ { s } : = \overline { { C ^ { \infty } (  { \mathbb { T } } ^ { d } ;  { \mathbb { R } } ^ { d } ) } } ^ { \mathscr { C } _ { x } ^ { s } } . } \end{array}
$$

The local-in-time solution space is

$$
E _ { \tau } : = \left\{ u \in C ( [ 0 , \tau ] ; \mathcal { C } _ { x } ^ { s } ) : \nabla \cdot u ( t ) = 0 , \int _ { \mathbb { T } ^ { d } } u ( t , x ) d x = 0 \right\} , \qquad \| u \| _ { E _ { \tau } } : = \operatorname* { s u p } _ { t \leq \tau } \| u ( t ) \| _ { \mathcal { C } _ { x } ^ { s } } .
$$

We distinguish this space from the approximation space

$$
Y _ { \tau } : = C ( [ 0 , \tau ] \times \mathbb { T } ^ { d } ; \mathbb { R } ^ { d } ) , \qquad \| u \| _ { Y _ { \tau } } : = \operatorname* { s u p } _ { ( t , x ) } | u ( t , x ) | .
$$

Here $C _ { \mathrm { e m b } }$ denotes a constant such that $\| u \| _ { Y _ { \tau } } \leq C _ { \mathrm { e m b } } \| u \| _ { E _ { \tau } }$ , uniformly for $\tau \leq T$ . All norms of vector-valued functions use the maximum over components. Taking the closure of smooth functions ensures continuity of the free solution at the initial time in the $\mathcal { C } _ { x } ^ { s }$ norm. The heat semigroup is strongly continuous on $\mathcal { C } _ { g , x } ^ { s } .$ , but not on the whole of $\mathcal { C } _ { x } ^ { s }$ . The initial-data class used in the approximation theorem lies in ${ \mathcal { C } } _ { x } ^ { \bar { s } }$ for $\bar { s } > s$ , and is therefore contained in $\mathcal { C } _ { g , x } ^ { s }$

5.2. Local existence and stability on a common interval. We present the estimates needed for the fixed-point argument. The spatial Littlewood–Paley norms are those of Section 2.1.

Lemma 5.1 (Product, projection, and semigroup estimates). Let $0 < s < 1$ . There is a constant $C _ { s . }$ , depending only on d and $s ,$ such that

$$
\begin{array} { r l } { \| f g \| _ { \mathcal { C } _ { x } ^ { s } } \leq C _ { s } \| f \| _ { \mathcal { C } _ { x } ^ { s } } \| g \| _ { \mathcal { C } _ { x } ^ { s } } , } & { } \\ { \| P f \| _ { \mathcal { C } _ { x } ^ { s } } \leq C _ { s } \| f \| _ { \mathcal { C } _ { x } ^ { s } } , } & { } \\ { \| e ^ { \nu t \Delta } f \| _ { \mathcal { C } _ { x } ^ { s } } \leq \| f \| _ { \mathcal { C } _ { x } ^ { s } } , \qquad t \geq 0 , } \\ { \| ( e ^ { \nu h \Delta } - I ) g \| _ { \mathcal { C } _ { x } ^ { s } } \leq C _ { s } ( \nu h ) ^ { 1 / 2 } \| g \| _ { \mathcal { C } _ { x } ^ { s + 1 } } , } & { \quad h > 0 . } \end{array}
$$

The product estimate applies componentwise to tensor products. Moreover, $t \mapsto e ^ { \nu t \Delta } f$ belongs to $C ( [ 0 , T ] ; \mathcal { C } _ { x } ^ { s } )$ for every $f \in \mathcal { C } _ { g , x } ^ { s }$

Proof. For $0 < s < 1$ , the Besov norm is equivalent to $\| f \| _ { \infty } + [ f ] _ { s }$ , where $[ f ] _ { i }$ <sub>s</sub> is the spatial H¨older seminorm [4]. Thus

$$
[ f g ] _ { s } \leq \| f \| _ { \infty } [ g ] _ { s } + [ f ] _ { s } \| g \| _ { \infty } ,
$$

which gives the product estimate. The symbol of $P$ is smooth and homogeneous of degree zero away from the origin. Its localization to a dyadic annulus has uniformly bounded convolutionkernel $L ^ { 1 }$ norm, so $\| \Delta _ { j } ^ { x } P f \| _ { \infty } \leq C \| \Delta _ { j } ^ { x } f \| _ { \infty }$ on the annular blocks. The low-frequency block contains only finitely many torus modes and satisfies the same bound, with $P ( 0 )$ defined to be zero.

The heat semigroup commutes with $\Delta _ { j } ^ { x }$ and contracts $L ^ { \infty } ;$ hence it contracts the stated Besov norm. For the increment estimate, Bernstein’s inequality and the semigroup contraction give

$$
\begin{array} { r } { \| ( e ^ { \nu h \Delta } - I ) \Delta _ { j } ^ { x } g \| _ { \infty } \leq C \operatorname* { m i n } \{ 1 , \nu h 2 ^ { 2 j } \} \| \Delta _ { j } ^ { x } g \| _ { \infty } . } \end{array}
$$

After multiplication by $2 ^ { j s }$ , the right-hand side is at most

$$
C \operatorname* { m i n } \lbrace 2 ^ { - j } , \nu h 2 ^ { j } \rbrace \| g \| _ { \mathcal { C } _ { x } ^ { s + 1 } } \leq C ( \nu h ) ^ { 1 / 2 } \| g \| _ { \mathcal { C } _ { x } ^ { s + 1 } } .
$$

The finitely many low modes satisfy the same estimate. Finally, if $f \in \mathcal { C } _ { g , x } ^ { s }$ , choose smooth $f _ { n }  f$ in $\mathcal { C } _ { x } ^ { s }$ . Then

$$
\| ( e ^ { \nu h \Delta } - I ) f \| _ { \mathcal { C } _ { x } ^ { s } } \leq 2 \| f - f _ { n } \| _ { \mathcal { C } _ { x } ^ { s } } + \| ( e ^ { \nu h \Delta } - I ) f _ { n } \| _ { \mathcal { C } _ { x } ^ { s } } .
$$

First let $h \downarrow 0$ and then $n  \infty$ . Continuity at other times follows from the semigroup property. □

The next embedding allows us to estimate the stochastic convolution in $C ( [ 0 , T ] ; \mathcal { C } _ { x } ^ { s } )$ by its parabolic regularity. The strict inequality $\alpha > s$ ensures continuity in time with respect to the $\mathcal { C } _ { x } ^ { s }$ norm.

Lemma 5.2 (From parabolic regularity to spatial paths). If α $> s > 0$ , then

$$
\begin{array} { r } { \begin{array} { r l r l } & { { \mathcal C } ^ { \alpha } ( D _ { T } ) \hookrightarrow C ( [ 0 , T ] ; { \mathcal C } _ { x } ^ { s } ) , } & & { \| v \| _ { C ( [ 0 , T ] ; { \mathcal C } _ { x } ^ { s } ) } \le C _ { \alpha , s } \| v \| _ { { \mathcal C } ^ { \alpha } ( D _ { T } ) } . } \end{array} } \end{array}
$$

Proof. Extend v to $\mathbb { R } \times \mathbb { T } ^ { d }$ with norm bounded by twice its restriction norm. Spatial and parabolic frequency supports imply $\Delta _ { j } ^ { x } \Delta _ { k } ^ { \mathfrak { s } } = 0$ unless $k \geq j - C$ . Since $\alpha > 0$ , summation over these parabolic blocks yields, uniformly in t,

$$
\| \Delta _ { j } ^ { x } v ( t ) \| _ { \infty } \leq C \sum _ { k \geq j - C } 2 ^ { - k \alpha } \| v \| _ { \mathcal { C } ^ { \alpha } } \leq C 2 ^ { - j \alpha } \| v \| _ { \mathcal { C } ^ { \alpha } } .
$$

This proves the norm estimate. To prove time continuity, set $\begin{array} { r } { v _ { L } : = \sum _ { k \leq L } \Delta _ { k } ^ { \mathfrak { s } } v } \end{array}$ . Each $v _ { L }$ is smooth, and the same calculation for the remaining blocks gives

$$
\operatorname* { s u p } _ { t \leq T } \| v ( t ) - v _ { L } ( t ) \| c _ { x } ^ { s } \leq C 2 ^ { - L ( \alpha - s ) } \| v \| _ { \mathcal { C } ^ { \alpha } } .
$$

Thus $v _ { L } \to v$ in $C ( [ 0 , T ] ; \mathcal { C } _ { x } ^ { s } )$ , as claimed.

Lemma 5.3 (Heat smoothing with one derivative). For $0 < s < 1$ and $t > 0$

$$
\| e ^ { \nu t \Delta } P \nabla \cdot G \| _ { \mathcal { C } _ { x } ^ { s } } \leq C _ { s } ( \nu t ) ^ { - 1 / 2 } \| G \| _ { \mathcal { C } _ { x } ^ { s } } .
$$

Proof. On a spatial annulus of frequency $2 ^ { j }$ , the localized multiplier has convolution-kernel $L ^ { 1 }$ norm bounded by $C 2 ^ { j } e ^ { - c \nu t 2 ^ { 2 j } }$ . Multiplying the block estimate by $2 ^ { j s }$ and using

$$
\operatorname* { s u p } _ { r > 0 } r e ^ { - c \nu t r ^ { 2 } } \leq C ( \nu t ) ^ { - 1 / 2 }
$$

gives the assertion on the annular blocks. The derivative annihilates the zero mode, and the finitely many remaining low modes obey the same bound. □

Lemma 5.4 (Bilinear estimate). For $u , v \in E _ { \tau }$ , the function $N ( u , v )$ belongs to $E _ { \tau }$ , vanishes at $t = 0$ , and satisfies

$$
\| N ( u , v ) \| _ { E _ { \tau } } \leq C _ { \mathrm { b } } \sqrt { \tau / \nu } \| u \| _ { E _ { \tau } } \| v \| _ { E _ { \tau } } ,
$$

where $C _ { \mathrm { b } }$ depends only on d and s.

Proof. Apply Lemma 5.3 and the product estimate to obtain

$$
\begin{array} { r l r } {  { \| N ( u , v ) ( t ) \| _ { C _ { x } ^ { s } } \leq C _ { s } ^ { 2 } \| u \| _ { E _ { \tau } } \| v \| _ { E _ { \tau } } \int _ { 0 } ^ { t } ( \nu ( t - \sigma ) ) ^ { - 1 / 2 } d \sigma } } \\ & { } & { = 2 C _ { s } ^ { 2 } \sqrt { t / \nu } \| u \| _ { E _ { \tau } } \| v \| _ { E _ { \tau } } . ~ } \end{array}
$$

We may therefore take $C _ { \mathrm { b } } = 2 C _ { s } ^ { 2 }$ . This also proves convergence to zero at the initial time. At a positive time, split the integral a distance h from its upper endpoint. The final segment has $\mathcal { C } _ { x } ^ { s }$ norm at most $C \sqrt { h / \nu } \| u \| _ { E _ { \tau } } \| v \| _ { E _ { \tau } }$ . On the remaining segment, the positive heat-evolution time gives an additional spatial derivative, and the semigroup increment estimate of Lemma 5.1 gives continuity. Letting $h \downarrow 0$ proves continuity of the full integral. The Leray projector and the divergence ensure that its values are divergence-free and have zero mean. □

Proposition 5.5 (Local well-posedness and stability). Fix $0 < s < 1$ and $\rho > 0$ . Choose $c _ { * } > 0$ ， depending only on d and $s ,$ so that $6 C _ { \mathrm { b } } \sqrt { c _ { * } } \le 1 / 2$ , and set

$$
\tau ^ { * } : = T \wedge \frac { c _ { * } \nu } { \rho ^ { 2 } } .
$$

Suppose that $u _ { 0 } \in \mathcal { C } _ { g , x } ^ { s }$ is divergence-free and mean-zero, and that $K _ { \nu } \phi \in E _ { T }$ with $( K _ { \nu } \phi ) ( 0 ) = 0$ If

$$
\| u _ { 0 } \| _ { \mathcal { C } _ { x } ^ { s } } \le \rho , \qquad \| K _ { \nu } \phi \| _ { E _ { T } } \le \rho ,\tag{70}
$$

then (69) has a unique solution in the closed ball $\{ u \in E _ { \tau ^ { * } } : \| u \| _ { E _ { \tau ^ { * } } } \leq 3 \rho \}$ . For any two data pairs satisfying these hypotheses,

$$
\| u ^ { 1 } - u ^ { 2 } \| _ { E _ { \tau ^ { * } } } \leq 2 \left( \| u _ { 0 } ^ { 1 } - u _ { 0 } ^ { 2 } \| _ { \mathcal { C } _ { x } ^ { s } } + \| K _ { \nu } ( \phi ^ { 1 } - \phi ^ { 2 } ) \| _ { E _ { T } } \right) .\tag{71}
$$

If $s - 2 < \beta _ { 1 } < 0$ , the forcing diference satisfies

$$
\| K _ { \nu } ( \phi ^ { 1 } - \phi ^ { 2 } ) \| _ { E _ { T } } \leq C _ { \nu , \beta _ { 1 } , s , T } \| \phi ^ { 1 } - \phi ^ { 2 } \| _ { \mathcal L ^ { \beta _ { 1 } } ( D _ { T } ) } .\tag{72}
$$

Proof. Step 1: the invariant ball. Set $a ( t ) : = e ^ { \nu t \Delta } u _ { 0 } + K _ { \nu } \phi ( t )$ and $\mathcal { T } ( u ) : = a - N ( u , u )$ The strong continuity in Lemma 5.1 and the assumptions on the convolution give $a \in E _ { \tau ^ { * } }$ and $\| a \| _ { E _ { \tau ^ { * } } } \leq 2 \rho$ . For later reference, the bilinear estimate reads

$$
\| N ( u , v ) \| _ { E _ { \tau } } \leq C _ { \mathrm { b } } \sqrt { \tau / \nu } \| u \| _ { E _ { \tau } } \| v \| _ { E _ { \tau } } .\tag{73}
$$

Consequently, if $\| u \| _ { E _ { \tau ^ { * } } } \leq 3 \rho$ , then

$$
\| \mathcal { T } ( u ) \| _ { E _ { \tau ^ { * } } } \leq 2 \rho + 9 C _ { \mathrm { b } } \rho ^ { 2 } \sqrt { \tau ^ { * } / \nu } \leq 2 \rho + 9 C _ { \mathrm { b } } \rho \sqrt { c _ { * } } \leq \frac { 1 1 } { 4 } \rho < 3 \rho .
$$

Thus the free term is bounded by $2 \rho ,$ , and the nonlinear term by $3 \rho / 4$ , so $\tau$ preserves the closed ball of radius $3 \rho$

Step 2: contraction and the choice of time. Bilinearity gives

$$
N ( u , u ) - N ( v , v ) = N ( u , u - v ) + N ( u - v , v ) .
$$

On the same ball, (73) therefore implies

$$
\begin{array} { r } { \| \mathcal { T } ( u ) - \mathcal { T } ( v ) \| _ { E _ { \tau ^ { * } } } \leq 6 C _ { \mathrm { b } } \rho \sqrt { \tau ^ { * } / \nu } \| u - v \| _ { E _ { \tau ^ { * } } } \leq \frac { 1 } { 2 } \| u - v \| _ { E _ { \tau ^ { * } } } . } \end{array}
$$

The condition $\tau ^ { * } \le c _ { * } \nu / \rho ^ { 2 }$ therefore gives a contraction factor at most $1 / 2$ , uniformly over all admissible data. The fixed-point theorem gives existence and uniqueness in the stated ball, and evaluation at $t = 0$ gives $u ( 0 ) = u _ { 0 }$

Step 3: stability with respect to both inputs. Subtract the two mild equations. Since both solutions lie in the $3 \rho \mathrm { - b a l l }$ , the estimate from Step 2 yields

$$
\begin{array} { r l } & { \| u ^ { 1 } - u ^ { 2 } \| _ { { \cal E } _ { \tau ^ { * } } } \leq \| u _ { 0 } ^ { 1 } - u _ { 0 } ^ { 2 } \| _ { { \cal E } _ { x } ^ { s } } + \| K _ { \nu } ( \phi ^ { 1 } - \phi ^ { 2 } ) \| _ { E _ { T } } } \\ & { \qquad + \frac { 1 } { 2 } \| u ^ { 1 } - u ^ { 2 } \| _ { E _ { \tau ^ { * } } } . } \end{array}
$$

Absorbing the last term proves (71). Finally, the viscous version of Lemma 2.4 gives $K _ { \nu }$ : $\mathcal { C } ^ { \beta _ { 1 } } ( D _ { T } ) \mathrm { ~ \bar { \to ~ } } \mathcal { C } ^ { \beta _ { 1 } + 2 } ( D _ { T } )$ . Its constant may depend on ν and T, which are fixed. Applying Lemma 5.2 with $\alpha = \beta _ { 1 } + 2 > s$ proves (72). □

For fixed $\rho ,$ write

$$
S ^ { \mathrm { l o c } } ( u _ { 0 } , \phi ) : = u \in E _ { \tau ^ { * } }
$$

for this local solution. Its domain consists of the admissible data in Proposition 5.5; all subsequent evaluations use the same $\rho$ and the same $\tau ^ { * }$

5.3. Fourier measurements and admissible coeficients. Fix $0 < s < \bar { s } < 1$ and $M > 0$ and let

$$
\mathcal { G } _ { 0 } ^ { \mathrm { N S E } } : = \left\{ u _ { 0 } \in \mathcal { C } _ { x } ^ { \bar { s } } : \nabla \cdot u _ { 0 } = 0 , \ \int _ { \mathbb { T } ^ { d } } u _ { 0 } = 0 , \ \| u _ { 0 } \| _ { \mathcal { C } _ { x } ^ { \bar { s } } } \leq M \right\} .
$$

The initial velocity is represented by its retained spatial Fourier coeficients and reconstructed using the cutof $\Pi _ { J _ { 0 } } ^ { x }$ of Section 2.1. This reconstruction preserves divergence and spatial mean; interpolation of componentwise grid values does not in general preserve either constraint.

The measurement maps act on each of the d components. Their target dimensions are

$$
D _ { 0 } : = d N _ { 0 } , \qquad N _ { 0 } : = | \Lambda _ { J _ { 0 } } ^ { x } | , \qquad D _ { J } : = d N _ { J } , \qquad N _ { J } : = | \Lambda _ { J } | .
$$

Thus $N _ { 0 }$ and $N _ { J }$ count scalar basis modes, while $D _ { 0 }$ and $D _ { J }$ count real scalar measurements of a vector field. For fixed $d , D _ { 0 } \asymp 2 ^ { J _ { 0 } d }$ and $\dot { D } _ { J } \asymp 2 ^ { J ( d + 2 ) }$

Lemma 5.6 (Spatial Fourier measurements). Let

$$
c _ { J _ { 0 } } ^ { x } ( u _ { 0 } ) : = \big ( \langle u _ { 0 , a } , e _ { \lambda } \rangle \big ) _ { 1 \leq a \leq d , \lambda \in \Lambda _ { J _ { 0 } } ^ { x } } \in \mathbb { R } ^ { D _ { 0 } } , \qquad ( \operatorname { R e c } _ { J _ { 0 } } ^ { x } ( v ) ) _ { a } : = \sum _ { \lambda \in \Lambda _ { J _ { 0 } } ^ { x } } m _ { J _ { 0 } } ^ { x } ( \lambda ) v _ { a , \lambda } e _ { \lambda } .
$$

Then Re $\mathfrak { z } _ { J _ { 0 } } ^ { x } ( c _ { J _ { 0 } } ^ { x } ( u _ { 0 } ) ) = \Pi _ { J _ { 0 } } ^ { x } u _ { 0 }$ , and

(74)

$$
\| ( I - \Pi _ { J _ { 0 } } ^ { x } ) u _ { 0 } \| _ { \mathcal { C } _ { x } ^ { s } } \le C 2 ^ { - J _ { 0 } ( \bar { s } - s ) } M , \qquad \quad u _ { 0 } \in \mathcal { G } _ { 0 } ^ { \mathrm { N S E } } ,\tag{75}
$$

$$
\| \operatorname { R e c } _ { J _ { 0 } } ^ { x } ( v ) - \operatorname { R e c } _ { J _ { 0 } } ^ { x } ( v ^ { \prime } ) \| _ { \mathcal { C } _ { x } ^ { s } } \leq C 2 ^ { J _ { 0 } s } D _ { 0 } ^ { 1 / 2 } | v - v ^ { \prime } | _ { 2 } .
$$

Moreover, uniformly over $J _ { 0 } \geq 0$

$$
| c _ { J _ { 0 } } ^ { x } ( u _ { 0 } ) | _ { \infty } \le C M , \qquad \| u _ { 0 } \| _ { \mathcal { C } _ { x } ^ { s } } + \| \Pi _ { J _ { 0 } } ^ { x } u _ { 0 } \| _ { \mathcal { C } _ { x } ^ { s } } \le C M .\tag{76}
$$

Both $u _ { 0 }$ and its truncation belong to $\mathcal { C } _ { g , x } ^ { s }$ , and the truncation is divergence-free and mean-zero.

Proof. The smooth cutof is uniformly bounded on every spatial Besov space. The multiplier $I - \Pi _ { J _ { 0 } } ^ { x }$ removes all frequencies below a fixed multiple of $2 ^ { J _ { 0 } }$ , so the remaining blocks satisfy

$$
2 ^ { j s } \| \Delta _ { j } ^ { x } ( I - \Pi _ { J _ { 0 } } ^ { x } ) u _ { 0 } \| _ { \infty } \leq C 2 ^ { - j ( \bar { s } - s ) } \| u _ { 0 } \| _ { \mathcal { C } _ { x } ^ { \bar { s } } } , \qquad j \geq J _ { 0 } - C .
$$

Taking the supremum proves (74). A retained trigonometric mode has frequency at most $C 2 ^ { J _ { 0 } }$ and $\mathcal { C } _ { x } ^ { s }$ norm at most $C 2 ^ { J _ { 0 } s }$ . The triangle inequality and Cauchy–Schwarz over the real coeficients prove (75). Each coeficient is bounded by $\| u _ { 0 } \| _ { \infty } \| e _ { \lambda } \| _ { L ^ { 1 } }$ , with uniformly bounded $\| e _ { \lambda } \| _ { L ^ { 1 } } ;$ this gives the first part of (76). The second follows from boundedness of the cutofs and ${ \mathcal { C } } _ { x } ^ { \bar { s } } \hookrightarrow { \mathcal { C } } _ { x } ^ { s }$ The cutof error also proves approximation of $u _ { 0 }$ by smooth functions in $\mathcal { C } _ { x } ^ { s }$ . Finally, the cutof acts componentwise with the same scalar symbol, so it commutes with divergence and preserves a zero mean. □

For the forcing, use $c _ { J }$ and Rec<sub>J</sub> componentwise, now with domain $\mathbb { R } ^ { D _ { J } }$ . Assume $s - 2 <$ $\beta _ { 1 } < \beta _ { 0 } < 0$ and write

$$
X _ { 0 } = ( \mathcal { C } _ { g } ^ { \beta _ { 0 } } ( \Lambda _ { T } ) ) ^ { d } , \qquad X _ { 1 } = ( \mathcal { C } ^ { \beta _ { 1 } } ( D _ { T } ) ) ^ { d } , \qquad \delta : = \beta _ { 0 } - \beta _ { 1 } > 0 .
$$

The estimates of Lemmas 3.2 and 3.3 hold componentwise. In particular,

(77)

$$
\begin{array} { r l } & { \| ( I - \Pi _ { J } ) \zeta \| _ { X _ { 1 } } \leq C 2 ^ { - J \delta } \| \zeta \| _ { X _ { 0 } } , } \\ & { \qquad | c _ { J } ( \zeta ) | _ { \infty } \leq C 2 ^ { - J \beta _ { 0 } } \| \zeta \| _ { X _ { 0 } } , } \end{array}\tag{78}
$$

$$
\| \mathrm { R e c } _ { J } ( c ) - \mathrm { R e c } _ { J } ( c ^ { \prime } ) \| _ { X _ { 1 } } \leq C D _ { J } ^ { 1 / 2 } | c - c ^ { \prime } | _ { 2 } .
$$

The finite-dimensional domain must contain the measurement vectors and must also allow the local solution to be evaluated at every center of the partition of unity. Let $\mathcal { V } _ { J _ { 0 } } \subset \mathbb { R } ^ { D _ { 0 } }$ and $\mathcal { W } _ { J } \subset \mathbb { R } ^ { D _ { J } }$ be the linear subspaces defined by

$$
\begin{array} { l c l l } { v \in \mathcal { V } _ { J _ { 0 } } } & { \iff } & { \nabla \cdot \mathrm { R e c } _ { J _ { 0 } } ^ { x } ( v ) = 0 , } & { \displaystyle \int _ { \mathbb T ^ { d } } \mathrm { R e c } _ { J _ { 0 } } ^ { x } ( v ) = 0 , } \\ { c \in \mathcal { W } _ { J } } & { \iff } & { \nabla \cdot \mathrm { R e c } _ { J } ( c ) = 0 , } & { \displaystyle \int _ { \mathbb T ^ { d } } \mathrm { R e c } _ { J } ( c ) ( t , x ) d x = 0 \quad \mathrm { f o r ~ a l l ~ } t . } \end{array}
$$

These are finite linear constraints: at each nonzero spatial frequency, the vector coeficient is orthogonal to that frequency, and the spatial zero modes vanish.

Choose fixed constants $C _ { 0 } , C _ { 1 }$ suficiently large for (76) and (78), and put $R _ { J } : = C _ { 1 } 2 ^ { - J \beta _ { 0 } } \rho .$ Define

$$
\begin{array} { r } { K _ { J , J _ { 0 } , \rho } ^ { \mathrm { N S E } } : = \left\{ ( v , c ) \in ( [ - C _ { 0 } M , C _ { 0 } M ] ^ { D _ { 0 } } \cap \mathcal { V } _ { J _ { 0 } } ) \times ( [ - R _ { J } , R _ { J } ] ^ { D _ { J } } \cap \mathcal { W } _ { J } ) : \right. } \end{array}
$$

$$
\| \operatorname { R e c } _ { J _ { 0 } } ^ { x } ( v ) \| _ { \mathcal { C } _ { x } ^ { s } } \leq \rho , \quad \| K _ { \nu } \operatorname { R e c } _ { J } ( c ) \| _ { E _ { T } } \leq \rho \big \} .
$$

This set is compact: the linear constraints and the two norm constraints are closed, and the enclosing box is compact. The reconstructed fields are smooth, so the initial trace requirements of Proposition 5.5 also hold. Hence every point of this set, not only measurement vectors arising from the random forcing, defines a solution on the common interval $[ 0 , \tau ^ { * } ]$

5.4. The approximation theorem. The forcing is the centered Gaussian random variable of Assumption 2.2, interpreted componentwise, with the additional divergence and mean constraints. The requirement $\beta _ { 0 } + 2 > s$ ensures that its convolution is a continuous path in ${ \mathcal { C } } _ { x } ^ { s }$ , which is stronger than continuity of the convolution as a function of $( t , x )$

Remark 5.7 (An admissible Gaussian forcing). One example is white-in-time forcing with a spatially smoothing covariance. Let $( e _ { k } )$ be a real spatial trigonometric basis without the constant mode and set

$$
\eta = P \sum _ { k } \sigma _ { k } e _ { k } ( x ) \dot { W } _ { k } ( t ) , \qquad \sigma _ { k } ^ { 2 } \asymp ( 1 + | k | ^ { 2 } ) ^ { - a } ,
$$

where the $\dot { W } _ { k }$ are independent $\mathbb { R } ^ { d } .$ -valued temporal white noises, realized on $\mathbb { T } _ { T }$ . Its parabolic regularity threshold is $\beta _ { * } : =$ min $\{ - 1 , a - ( d + 2 ) / 2 \}$ . More precisely, it belongs almost surely to $\mathcal { C } _ { g } ^ { \beta } ( \Lambda _ { T } )$ for every $\beta < \beta _ { * }$ and has no strictly higher parabolic regularity. To see the two restrictions, a fixed nonzero spatial mode already has the regularity of temporal white noise, namely parabolic exponent −1. For the high-frequency blocks, the pointwise variance has order

$$
2 ^ { 2 j } \sum _ { 0 < | k | \lesssim 2 ^ { j } } ( 1 + | k | ^ { 2 } ) ^ { - a } \asymp \left\{ \begin{array} { l l } { 2 ^ { j ( d + 2 - 2 a ) } , } & { a < d / 2 , } \\ { ( 1 + j ) 2 ^ { 2 j } , } & { a = d / 2 , } \\ { 2 ^ { 2 j } , } & { a > d / 2 . } \end{array} \right.
$$

Gaussian moment estimates and Bernstein’s inequality, with an arbitrarily small loss in the exponent, give the claimed regularity below $\beta _ { * }$ . The lower variance bounds exclude any exponent above $\beta _ { * }$ ; no assertion at the endpoint is needed here. Consequently, there is a choice $s - 2 <$ $\beta _ { 0 } < \beta _ { * }$ precisely when $a > s + ( d - 2 ) / 2$ . The temporal bound −1 remains even under arbitrarily strong spatial smoothing.

Theorem 5.8 (High-probability local operator approximation). Let $d \ge 2 , \nu , T , M > 0$ , and $0 ~ < ~ s ~ < ~ \bar { s } ~ < ~ 1$ . Suppose that η is a centered Gaussian Borel random variable in $X _ { 0 }$ , is divergence-free and has zero spatial mean almost surely, and satisfies

$$
s - 2 < \beta _ { 0 } < 0 .
$$

Fix $\beta _ { 1 } \in ( s - 2 , \beta _ { 0 } )$ and $\delta = \beta _ { 0 } - \beta _ { 1 }$ . There are constants $C _ { 0 } , C _ { \nu } \geq 1$ such that, for every fixed $\rho \ge C _ { 0 } M$ , the following statements hold with $\tau ^ { * } = T \wedge c _ { * } \nu / \rho ^ { 2 }$

$$
G _ { \rho } : = \{ C _ { \nu } \| \eta \| _ { X _ { 0 } } \leq \rho \} , \qquad \mathbb { P } ( G _ { \rho } ^ { c } ) \leq C e ^ { - c \rho ^ { 2 } } .
$$

For each $\varepsilon \in ( 0 , 1 )$ , there exist cutofs $J , J _ { 0 }$ , an integer $m$ , scalar ReLU branch networks $\widetilde { a } _ { k }$ , and vector-valued ReLU trunk networks $\widetilde { q } _ { k }$ , such that

$$
\Gamma _ { \theta } ^ { \mathrm { N S E } } ( u _ { 0 } , \zeta ) ( t , x ) : = \sum _ { k = 1 } ^ { m } \widetilde { a } _ { k } \big ( c _ { J _ { 0 } } ^ { x } ( u _ { 0 } ) , c _ { J } ( \zeta ) \big ) \widetilde { q } _ { k } ( t , x )\tag{79}
$$

satisfies the deterministic bound

$$
\begin{array} { r l } { \displaystyle \operatorname* { s u p } _ { u _ { 0 } \in \mathcal { G } _ { 0 } ^ { \mathrm { N S E } } } \| \mathcal { S } ^ { \mathrm { l o c } } ( u _ { 0 } , \zeta ) - \Gamma _ { \theta } ^ { \mathrm { N S E } } ( u _ { 0 } , \zeta ) \| _ { Y _ { \tau ^ { * } } } \leq \varepsilon . } & { } \\ { \zeta \in X _ { 0 } , \ C _ { \nu } \| \zeta \| _ { X _ { 0 } \leq \rho } } & { } \\ { \nabla \cdot \zeta = 0 , \ \int _ { \mathbb { T } ^ { d } } \zeta = 0 } \end{array}\tag{80}
$$

In particular,

$$
\operatorname* { s u p } _ { u _ { 0 } \in \mathcal { G } _ { 0 } ^ { \mathrm { N S E } } } \mathbb { E } \left[ \mathbf { 1 } _ { G _ { \rho } } \Vert \mathcal { S } ^ { \mathrm { l o c } } ( u _ { 0 } , \eta ) - \Gamma _ { \theta } ^ { \mathrm { N S E } } ( u _ { 0 } , \eta ) \Vert _ { Y _ { \tau ^ { * } } } \right] \leq \varepsilon ,\tag{81}
$$

where the error inside the expectation is defined to be zero of $G _ { \rho }$ . The numbers of real scalar measurements may be chosen to obey

$$
{ \cal D } _ { J } \leq C \left( 1 + \frac { \rho } { \varepsilon } \right) ^ { ( d + 2 ) / \delta } , \qquad { \cal D } _ { 0 } \leq C \left( 1 + \frac { M } { \varepsilon } \right) ^ { d / ( \bar { s } - s ) } .\tag{82}
$$

The constants are independent of $\rho$ and $\varepsilon ;$ those in the probability estimate may depend on the law of $\eta .$ . The interval $[ 0 , \tau ^ { * } ]$ is independent of ε.

Proof. We first choose an interval on which the original and truncated solutions are defined. We then bound each of four errors by $\varepsilon / 4 \colon ( 1 )$ forcing truncation, (2) initial-velocity truncation, (3) partition-of-unity approximation, and (4) neural approximation.

First of all, the Schauder estimate, Lemma 5.2, and uniform boundedness of the smooth cutofs give

$$
\| K _ { \nu } \zeta \| _ { E _ { T } } + \operatorname* { s u p } _ { J \geq 0 } \| K _ { \nu } \Pi _ { J } \zeta \| _ { E _ { T } } \leq C _ { \nu } \| \zeta \| _ { X _ { 0 } } .\tag{83}
$$

Indeed, the Schauder estimate yields parabolic regularity $\beta _ { 0 } + 2$ . Since $\beta _ { 0 } + 2 > s$ , Lemma 5.2 gives the bound in $E _ { T }$ . Then, the convolution has zero initial trace: one may approximate $\zeta \in X _ { 0 }$ by smooth forcings and pass to the limit in $E _ { T }$ . Next, we increase $C _ { \nu }$ so that $C _ { \nu } \geq 1$ . Also choose $C _ { 0 }$ so that (76) implies

$$
\| u _ { 0 } \| _ { \mathcal { C } _ { x } ^ { s } } \leq \rho , \qquad \operatorname* { s u p } _ { J _ { 0 } \geq 0 } \| \Pi _ { J _ { 0 } } ^ { x } u _ { 0 } \| _ { \mathcal { C } _ { x } ^ { s } } \leq \rho
$$

whenever $u _ { 0 } \in \mathcal { G } _ { 0 } ^ { \mathrm { N S E } }$ and $\rho \ge C _ { 0 } M$

For the remainder of the deterministic argument, we fix any $u _ { 0 }$ and $\zeta$ in the supremum in (80). Then, estimate (83) shows that $\zeta$ and all $\Pi _ { J } \zeta$ satisfy the convolution bound in (70).

Thus Proposition 5.5 gives solutions on the common interval $[ 0 , \tau ^ { * } ]$ satisfying

$$
\begin{array} { r } { \| \mathcal { S } ^ { \mathrm { l o c } } ( u _ { 0 } , \zeta ) \| _ { E _ { \tau ^ { * } } } \le 3 \rho , } \\ { \| \mathcal { S } ^ { \mathrm { l o c } } ( u _ { 0 } , \Pi _ { J } \zeta ) \| _ { E _ { \tau ^ { * } } } \le 3 \rho , } \\ { \| \mathcal { S } ^ { \mathrm { l o c } } ( \Pi _ { J _ { 0 } } ^ { x } u _ { 0 } , \Pi _ { J } \zeta ) \| _ { E _ { \tau ^ { * } } } \le 3 \rho . } \end{array}
$$

Since these bounds are uniform in J and $J _ { 0 }$ , the existence interval is independent of both cutof levels.

Next, by (71), (72), and (77), we see

$$
\begin{array} { r l } & { \| S ^ { \mathrm { l o c } } ( u _ { 0 } , \zeta ) - S ^ { \mathrm { l o c } } ( u _ { 0 } , \Pi _ { J } \zeta ) \| _ { Y _ { \tau ^ { * } } } } \\ & { \quad \le 2 C _ { \mathrm { e m b } } \| K _ { \nu } ( I - \Pi _ { J } ) \zeta \| _ { E _ { T } } \le C \| ( I - \Pi _ { J } ) \zeta \| _ { X _ { 1 } } \le C _ { \mathrm { f } } 2 ^ { - J \delta } \rho . } \end{array}
$$

Here, $\beta _ { 1 } < \beta _ { 0 }$ gives a positive truncation rate, while $\beta _ { 1 } + 2 > s$ allows the Schauder estimate to control the error in $E _ { T }$ . Since the forcing is unchanged, (71) and the embedding $E _ { \tau ^ { * } } \hookrightarrow Y _ { \tau ^ { * } }$ give the first inequality, while (74) and $\| u _ { 0 } \| _ { \mathcal { C } _ { x } ^ { \bar { s } } } \leq M$ with $\bar { s } > s$ give the second:

$$
\begin{array} { r l } & { \| \mathcal { S } ^ { \mathrm { l o c } } ( u _ { 0 } , \Pi _ { J } \zeta ) - \mathcal { S } ^ { \mathrm { l o c } } ( \Pi _ { J _ { 0 } } ^ { x } u _ { 0 } , \Pi _ { J } \zeta ) \| _ { Y _ { \tau ^ { * } } } } \\ & { \quad \leq 2 C _ { \mathrm { e m b } } \| ( I - \Pi _ { J _ { 0 } } ^ { x } ) u _ { 0 } \| _ { \mathcal { C } _ { x } ^ { s } } \leq C _ { \mathrm { i } } 2 ^ { - J _ { 0 } ( \bar { s } - s ) } M . } \end{array}
$$

Choose the smallest nonnegative integers $J , J _ { 0 }$ satisfying

$$
2 ^ { J \delta } \ge { \frac { 4 C _ { \mathrm { f } } \rho } { \varepsilon } } , \qquad 2 ^ { J _ { 0 } ( \bar { s } - s ) } \ge { \frac { 4 C _ { \mathrm { i } } M } { \varepsilon } } .
$$

Both errors are then at most $\varepsilon / 4$ . The minimal choices also give upper bounds for J and $J _ { 0 } ,$ which are needed to estimate the numbers of measurements. Since a smallest nonnegative dyadic level satisfies

$$
2 ^ { J } \leq 2 \operatorname* { m a x } \{ 1 , ( 4 C _ { \mathrm { f } } \rho / \varepsilon ) ^ { 1 / \delta } \}
$$

and similarly for $J _ { 0 } .$ , therefore we have (82).

We abbreviate $\mathcal { K } : = \mathcal { K } _ { J , J _ { 0 } , \rho } ^ { \mathrm { N S E } }$ and define

$$
\Gamma _ { J , J _ { 0 } } ^ { \mathrm { N S E } } ( v , c ) : = S ^ { \mathrm { l o c } } ( \mathrm { R e c } _ { J _ { 0 } } ^ { x } ( v ) , \mathrm { R e c } _ { J } ( c ) ) , \qquad ( v , c ) \in \mathcal { K } .
$$

Every point of $\kappa$ is admissible for the local proposition, by its definition. Moreover, $z =$ $( c _ { J _ { 0 } } ^ { x } ( u _ { 0 } ) , c _ { J } ( \zeta ) )$ belongs to $\kappa .$ . The two norm constraints follow from (83) and $( 7 6 ) ;$ ; the coeficient bounds follow from (76) and (78), using $\| \zeta \| _ { X _ { 0 } } \le \rho / C _ { \nu } \le \rho$ . At this measurement vector, we see

$$
\Gamma _ { J , J _ { 0 } } ^ { \mathrm { N S E } } ( z ) = S ^ { \mathrm { l o c } } ( \Pi _ { J _ { 0 } } ^ { x } u _ { 0 } , \Pi _ { J } \zeta ) .\tag{84}
$$

For $z = ( v , c ) , z ^ { \prime } = ( v ^ { \prime } , c ^ { \prime } ) \in \mathcal { K }$ , stability and the reconstruction estimates imply

$$
\begin{array} { r l } { \| \Gamma _ { J , J _ { 0 } } ^ { \mathrm { N S E } } ( z ) - \Gamma _ { J , J _ { 0 } } ^ { \mathrm { N S E } } ( z ^ { \prime } ) \| _ { Y _ { \tau ^ { * } } } \leq C \big ( 2 ^ { J _ { 0 } s } D _ { 0 } ^ { 1 / 2 } | v - v ^ { \prime } | _ { 2 } + D _ { J } ^ { 1 / 2 } | c - c ^ { \prime } | _ { 2 } \big ) } & { } \\ { \leq L | z - z ^ { \prime } | _ { 2 } , } \end{array}
$$

where we take

$$
L : = \operatorname* { m a x } \left\{ 1 , C \big ( 2 ^ { 2 J _ { 0 } s } D _ { 0 } + D _ { J } \big ) ^ { 1 / 2 } \right\} .
$$

Next, we put

$$
r : = \frac { \varepsilon } { 4 L } ,\tag{85}
$$

and choose an r/2-net $z ^ { 1 } , \ldots , z ^ { m }$ of $\kappa$ with centers in $\kappa .$ . Compactness ensures that the net is finite. By the partition of unity Lemma 3.10, we get

$$
h _ { k } ( z ) : = \displaystyle \left( 1 - \frac { | z - z ^ { k } | _ { 2 } } { r } \right) _ { + } , \qquad \rho _ { k } ( z ) : = \frac { h _ { k } ( z ) } { \sum _ { j = 1 } ^ { m } h _ { j } ( z ) } , \qquad z \in { \cal K } .
$$

The denominator is at least $1 / 2$ because the centers form an $r / 2 \mathrm { - n e t }$ . Thus $\rho _ { k } \geq 0$ and $\textstyle \sum _ { k } \rho _ { k } = 1$ For the solutions at the partition centers, $Q _ { k } : = \Gamma _ { J , J _ { 0 } } ^ { \mathrm { N S E } } ( z ^ { k } )$ , set

$$
A ( z ) ( t , x ) : = \sum _ { k = 1 } ^ { m } \rho _ { k } ( z ) Q _ { k } ( t , x ) .
$$

Each $Q _ { k }$ is defined on the same interval, because its center belongs to $\kappa .$ . If $\rho _ { k } ( z ) > 0$ , then $| z - z ^ { k } | _ { 2 } < r ;$ hence

$$
\begin{array} { r l } {  { \| \Gamma _ { J , J _ { 0 } } ^ { \mathrm { N S E } } ( z ) - A ( z ) \| _ { Y _ { \tau ^ { * } } } \leq \sum _ { k } \rho _ { k } ( z ) \| \Gamma _ { J , J _ { 0 } } ^ { \mathrm { N S E } } ( z ) - Q _ { k } \| _ { Y _ { \tau ^ { * } } } } \quad } & { } \\ & { \leq \displaystyle \sum _ { k } \rho _ { k } ( z ) L r = L r = \frac { \varepsilon } { 4 } . } \end{array}
$$

The choice (85) therefore gives the required partition-of-unity error.

Each reference solution satisfies

$$
\| Q _ { k } \| _ { Y _ { \tau ^ { * } } } \le 3 C _ { \mathrm { e m b } } \rho = : B .
$$

For $\delta _ { T } , \delta _ { B } > 0$ , choose vector-valued ReLU networks $\widetilde { q } _ { k }$ and scalar ReLU networks $\widetilde { a } _ { k }$ such that

$$
\| Q _ { k } - \widetilde { q } _ { k } \| _ { Y _ { \tau ^ { * } } } \leq \delta _ { T } , \qquad \operatorname* { s u p } _ { z \in K } | \rho _ { k } ( z ) - \widetilde { a } _ { k } ( z ) | \leq \delta _ { B } .\tag{86}
$$

Each partition weight $\rho _ { k }$ is continuous on the compact set $\kappa .$ The universal approximation theorem [25] therefore gives a ReLU network $\widetilde { a } _ { k }$ satisfying the second inequality in (86).

To construct $\widetilde { q } _ { k }$ , let $I _ { h } Q _ { k }$ be the periodic multilinear interpolant used in the proof of Theorem 4.1, applied to each component of $Q _ { k }$ . Since $Q _ { k }$ is uniformly continuous on $[ 0 , \tau ^ { * } ] \times \mathbb { T } ^ { d }$ , a suficiently fine grid gives

$$
\| Q _ { k } - I _ { h } Q _ { k } \| _ { Y _ { \tau ^ { * } } } \leq \frac { \delta _ { T } } { 2 } .
$$

The one-dimensional basis functions have exact ReLU representations. Approximating their products by Lemma 3.9 gives a vector-valued ReLU network $\widetilde { q } _ { k }$ . There are only finitely many interpolation terms, so the multiplication accuracies can be chosen such that

$$
\| I _ { h } Q _ { k } - \widetilde { q } _ { k } \| _ { Y _ { \tau ^ { * } } } \leq \frac { \delta _ { T } } { 2 } .
$$

The triangle inequality proves the first bound in (86). The network depends on the spatial variables through the periodic one-dimensional basis functions, so its values agree on opposite faces of $[ 0 , 2 \pi ] ^ { d }$ . Thus $\widetilde { q } _ { k }$ defines an element of $Y _ { \tau ^ { * } }$

Define $\begin{array} { r } { \widetilde A ( z ) : = \sum _ { k } \widetilde a _ { k } ( z ) \widetilde q _ { k } } \end{array}$ . Adding and subtracting $\textstyle \sum _ { k } \rho _ { k } ( z ) { \widetilde { q } } _ { k }$ gives

$$
\begin{array} { l } { \displaystyle \| A ( z ) - \widetilde A ( z ) \| _ { Y _ { \tau ^ { * } } } \leq \sum _ { k } \rho _ { k } ( z ) \| Q _ { k } - \widetilde q _ { k } \| _ { Y _ { \tau ^ { * } } } } \\ { + \sum _ { k } | \rho _ { k } ( z ) - \widetilde a _ { k } ( z ) | \| \widetilde q _ { k } \| _ { Y _ { \tau ^ { * } } } } \\ { \leq \delta _ { T } + m ( B + \delta _ { T } ) \delta _ { B } . } \end{array}
$$

The partition property bounds the first sum by $\delta _ { T }$ independently of $m$ . The second sum contains m errors, each bounded by $( B + \delta _ { T } ) \delta _ { B }$ . We therefore choose

$$
\delta _ { T } : = \frac { \varepsilon } { 8 } , \qquad \delta _ { B } : = \frac { \varepsilon } { 8 m ( B + 1 ) } .
$$

$\mathrm { A s } ~ \varepsilon < 1$ , these choices give $\delta _ { T } + m ( B + \delta _ { T } ) \delta _ { B } \leq \varepsilon / 4$ . The network in (79) is $\widetilde { A } ( c _ { J _ { 0 } } ^ { x } ( u _ { 0 } ) , c _ { J } ( \zeta ) )$ With $z = ( c _ { J _ { 0 } } ^ { x } ( u _ { 0 } ) , c _ { J } ( \zeta ) )$ , we use (84) to write

$$
\begin{array} { r l } & { \| S ^ { \mathrm { l o c } } ( u _ { 0 } , \zeta ) - \Gamma _ { \theta } ^ { \mathrm { N S E } } ( u _ { 0 } , \zeta ) \| _ { Y _ { \tau ^ { * } } } } \\ & { \quad \le \| \mathcal { S } ^ { \mathrm { l o c } } ( u _ { 0 } , \zeta ) - \mathcal { S } ^ { \mathrm { l o c } } ( u _ { 0 } , \Pi _ { J } \zeta ) \| _ { Y _ { \tau ^ { * } } } } \\ & { \quad \quad + \| \mathcal { S } ^ { \mathrm { l o c } } ( u _ { 0 } , \Pi _ { J } \zeta ) - \mathcal { S } ^ { \mathrm { l o c } } ( \Pi _ { J _ { 0 } } ^ { x } u _ { 0 } , \Pi _ { J } \zeta ) \| _ { Y _ { \tau ^ { * } } } } \\ & { \quad \quad + \| \Gamma _ { J , J _ { 0 } } ^ { \mathrm { N S E } } ( z ) - A ( z ) \| _ { Y _ { \tau ^ { * } } } + \| A ( z ) - \widetilde A ( z ) \| _ { Y _ { \tau ^ { * } } } } \\ & { \quad \le \frac \varepsilon 4 + \frac \varepsilon 4 + \frac \varepsilon 4 + \frac \varepsilon 4 = \varepsilon . } \end{array}
$$

Every bound is uniform over the initial-data class and the deterministic forcing ball in (80). This proves that estimate with one network for all these inputs.

Lastly, Fernique’s estimate in Lemma 2.3, applied to the vector-valued Gaussian variable, gives

$$
\begin{array} { r } { \mathbb { P } ( G _ { \rho } ^ { c } ) = \mathbb { P } ( \| \eta \| _ { X _ { 0 } } > \rho / C _ { \nu } ) \le C \exp ( - c \rho ^ { 2 } / C _ { \nu } ^ { 2 } ) \le C e ^ { - c ^ { \prime } \rho ^ { 2 } } . } \end{array}
$$

Fix $u _ { 0 } \in \mathcal { G } _ { 0 } ^ { \mathrm { N S E } }$ and define

$$
Z _ { u _ { 0 } } : = \left\{ \begin{array} { l l } { \Vert \mathcal { S } ^ { \mathrm { l o c } } ( u _ { 0 } , \eta ) - \Gamma _ { \theta } ^ { \mathrm { N S E } } ( u _ { 0 } , \eta ) \Vert _ { Y _ { \tau ^ { * } } } , } & { \mathrm { o n ~ } G _ { \rho } , } \\ { 0 , } & { \mathrm { o n ~ } G _ { \rho } ^ { c } . } \end{array} \right.
$$

By (71), (72), and the continuity of restriction from $X _ { 0 }$ to $X _ { 1 }$ , the solution map is continuous on the admissible forcing set. The neural approximation is also continuous in the forcing, since the Fourier measurements and ReLU networks are continuous. Thus $Z _ { u _ { 0 } }$ is measurable.

The uniform estimate (80) gives $Z _ { u _ { 0 } } \le \varepsilon { \bf 1 } _ { G _ { \rho } }$ almost surely, and hence

$$
\mathbb { E } Z _ { u _ { 0 } } \leq \varepsilon \mathbb { P } ( G _ { \rho } ) \leq \varepsilon .
$$

This bound is uniform in $u _ { 0 } \in \mathcal { G } _ { 0 } ^ { \mathrm { N S E } }$ , which proves (81).

## References

[1] Guillermo A. Alvarez, Ibrahim Ekren, Anastasis Kratsios, and Xuwei Yang. Neural operators can play dynamic Stackelberg games. Journal of Machine Learning Research, 26(303):1–61, 2025.

[2] Kang An, Chenhao Si, Shiqian Ma, and Ming Yan. Lightweight geometric adaptation for training physicsinformed neural networks, 2026. arXiv:2604.15392.

[3] Kang An, Chenhao Si, Ming Yan, and Shiqian Ma. AutoBalance: An automatic balancing framework for training physics-informed neural networks, 2025. arXiv:2510.06684.

[4] Hajer Bahouri, Jean-Yves Chemin, and Rapha¨el Danchin. Fourier Analysis and Nonlinear Partial Diferential Equations, volume 343 of Grundlehren der mathematischen Wissenschaften. Springer, 2011.

[5] Erhan Bayraktar, Qi Feng, Zecheng Zhang, and Zhaoyu Zhang. Deep neural operator learning for probabilistic models, 2025. arXiv:2511.07235.

[6] Vladimir I. Bogachev. Gaussian Measures, volume 62 of Mathematical Surveys and Monographs. American Mathematical Society, 1998.

[7] Marcin Bownik. Anisotropic Hardy Spaces and Wavelets, volume 164 of Memoirs of the American Mathematical Society. American Mathematical Society, Providence, RI, 2003.

[8] Ajay Chandra and Hendrik Weber. Stochastic PDEs, regularity structures, and interacting particle systems. Annales de la Facult´e des Sciences de Toulouse, 26(4):847–909, 2017.

[9] Tianping Chen and Hong Chen. Universal approximation to nonlinear operators by neural networks with arbitrary activation functions and its application to dynamical systems. IEEE Transactions on Neural Networks, 6(4):911–917, 1995.

[10] Ilya Chevyrev, Andris Gerasimoviˇcs, and Hendrik Weber. Feature engineering with regularity structures. Journal of Scientific Computing, 98, 2024.

[11] Frank Cole, Dixi Wang, Yineng Chen, Yulong Lu, and Rongjie Lai. In-context operator learning on the space of probability measures, 2026. arXiv:2601.09979.

[12] Maarten V. de Hoop, Nikola B. Kovachki, Nicholas H. Nelsen, and Andrew M. Stuart. Convergence rates for learning linear operators from noisy data. SIAM/ASA Journal on Uncertainty Quantification, 11(2):480–513, 2023.

[13] Takashi Furuya and Anastasis Kratsios. Simultaneously solving FBSDEs with neural operators of logarithmic depth, constant width, and sub-linear rank. arXiv:2410.14788v1, 2024. Version 1, 18 October 2024.

[14] Shiqi Gong, Peiyan Hu, Qi Meng, Yue Wang, Rongchan Zhu, Bingguang Chen, Zhi-Ming Ma, Hao Ni, and Tie-Yan Liu. Deep latent regularity network for modeling stochastic partial diferential equations. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 7740–7747, 2023.

[15] Massimiliano Gubinelli, Peter Imkeller, and Nicolas Perkowski. Paracontrolled distributions and singular PDEs. Forum of Mathematics, Pi, 3:e6, 2015.

[16] Jiequn Han, Arnulf Jentzen, and Weinan E. Solving high-dimensional partial diferential equations using deep learning. Proceedings of the National Academy of Sciences, 115(34):8505–8510, 2018.

[17] Peiyan Hu, Qi Meng, Bingguang Chen, Shiqi Gong, Yue Wang, Wei Chen, Rongchan Zhu, Zhi-Ming Ma, and Tie-Yan Liu. Neural operator with regularity structure for modeling dynamics driven by SPDEs, 2022. arXiv:2204.06255.

[18] Daniel Zhengyu Huang, Nicholas H. Nelsen, and Margaret Trautner. An operator learning perspective on parameter-to-observable maps. Foundations of Data Science, 7(1):163–225, 2025.

[19] Nikola Kovachki, Samuel Lanthaler, and Siddhartha Mishra. On universal approximation and error bounds for Fourier neural operators. Journal of Machine Learning Research, 22(290):1–76, 2021.

[20] Nikola Kovachki, Zongyi Li, Burigede Liu, Kamyar Azizzadenesheli, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Neural operator: Learning maps between function spaces with applications to PDEs. Journal of Machine Learning Research, 24(89):1–97, 2023.

[21] Anastasis Kratsios, Ariel Neufeld, and Philipp Schmocker. Neural operators which certifiably simultaneously solve infinitely many convex programs using logarithmic rank and linear complexity. arXiv:2508.14995, 2025. Version 2, revised 15 October 2026.

[22] Christian K¨ummerle, Mauro Maggioni, and Sui Tang. Learning transition operators from sparse space-time samples. IEEE Transactions on Information Theory, 70(9):6412–6446, 2024.

[23] Samuel Lanthaler, Siddhartha Mishra, and George Em Karniadakis. Error estimates for DeepONets: A deep learning framework in infinite dimensions. Transactions of Mathematics and Its Applications, 6(1):tnac001, 2022.

[24] Samuel Lanthaler and Nicholas H. Nelsen. Error bounds for learning with vector-valued random features. In Advances in Neural Information Processing Systems, volume 36, pages 71834–71861, 2023.

[25] Moshe Leshno, Vladimir Ya. Lin, Allan Pinkus, and Shimon Schocken. Multilayer feedforward networks with a nonpolynomial activation function can approximate any function. Neural Networks, 6(6):861–867, 1993.

[26] Zheyan Li, Yuantu Zhu, Hao Ni, Siran Li, Bingguang Chen, and Qi Meng. SPDEBench: An extensive benchmark for learning regular and singular stochastic partial diferential equations, 2025. arXiv:2505.18511.

[27] Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Fourier neural operator for parametric partial diferential equations. In International Conference on Learning Representations, 2021.

[28] Hao Liu, Zecheng Zhang, Wenjing Liao, and Hayden Schaefer. Neural scaling laws of deep ReLU and deep operator network: A theoretical study. 2024. Version 2, revised 26 May 2026.

[29] Sergey V. Lototsky and Boris L. Rozovskii. Wiener chaos solutions of linear stochastic evolution equations. Annals of Probability, 34(2):638–662, 2006.

[30] Fei Lu, Mauro Maggioni, and Sui Tang. Learning interaction kernels in stochastic systems of interacting particles from multiple trajectories. Foundations of Computational Mathematics, 22(4):1013–1067, 2022.

[31] Lu Lu, Pengzhan Jin, Guofei Pang, Zhongqiang Zhang, and George Em Karniadakis. Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. Nature Machine Intelligence, 3:218–229, 2021.

[32] Tingwei Meng, Moritz Voß, Nils Detering, Giulio Farolfi, Stanley Osher, and Georg Menz. Solving optimal execution problems via in-context operator networks, 2025. arXiv:2501.15106.

[33] Remigijus Mikulevicius and Boris L. Rozovskii. Linear parabolic stochastic PDEs and Wiener chaos. SIAM Journal on Mathematical Analysis, 29(2):452–480, 1998.

[34] Roberto Molinaro, Yunan Yang, Bj¨orn Engquist, and Siddhartha Mishra. Neural inverse operators for solving PDE inverse problems. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 25105–25139. PMLR, 2023.

[35] Nicholas H. Nelsen and Andrew M. Stuart. Operator learning using random features: A tool for scientific computing. SIAM Review, 66(3):535–571, 2024.

[36] Nicholas H. Nelsen and Yunan Yang. Operator learning meets inverse problems: A probabilistic perspective. In Machine Learning Solutions for Inverse Problems, Part B, volume 27 of Handbook of Numerical Analysis. Elsevier, 2026.

[37] Ariel Neufeld and Philipp Schmocker. Solving stochastic partial diferential equations using neural networks in the Wiener chaos expansion, 2024. arXiv:2411.03384.

[38] Cristopher Salvi, Maud Lemercier, and Andris Gerasimoviˇcs. Neural stochastic PDEs: Resolution-invariant learning of continuous spatiotemporal dynamics. In Advances in Neural Information Processing Systems, volume 35, 2022.

[39] Dai Shi, Lequan Lin, Andi Han, Luke Thompson, Jos´e Miguel Hern´andez-Lobato, Zhiyong Wang, and Junbin Gao. Expanding the chaos: Neural operator for stochastic (partial) diferential equations, 2026. arXiv:2601.01021.

[40] Dai Shi, Luke Thompson, Andi Han, Peiyan Hu, Junbin Gao, and Jos´e Miguel Hern´andez-Lobato. Wiener chaos expansion based neural operator for singular stochastic partial diferential equations, 2026. arXiv:2603.08219.

[41] Chenhao Si, Kang An, Shiqian Ma, and Ming Yan. From non-convex self-concordant regularization to scalable quasi-newton training of PINNs, 2026. arXiv:2608.04206.

[42] Hans Triebel. Theory of Function Spaces II, volume 84 of Monographs in Mathematics. Birkh¨auser Verlag, Basel, 1992.

[43] Weinan Wang, Bowen Gang, and Hao Deng. Operator learning for the 2D incompressible Navier–Stokes equations: a conformal prediction approach in the data-scarce regime, 2026. arXiv:2606.08654.

[44] Minglang Yin, Nicolas Charon, Ryan Brody, Lu Lu, Natalia Trayanova, and Mauro Maggioni. A scalable framework for learning the geometry-dependent solution operators of partial diferential equations. Nature Computational Science, 4(12):928–940, 2024.

Department of Mathematics and Computer Science<sub>,</sub> Fisk University<sub>,</sub> Nashville<sub>,</sub> TN 37208<sub>,</sub> USA Email address: hdu@fisk.edu

Department of Mathematics<sub>,</sub> Florida State University<sub>,</sub> Tallahassee<sub>,</sub> FL 32306<sub>,</sub> USA Email address: qfeng2@fsu.edu

Department of Mathematics<sub>,</sub> University of Oklahoma<sub>,</sub> Norman<sub>,</sub> OK 73019<sub>,</sub> USA Email address: ww@ou.edu