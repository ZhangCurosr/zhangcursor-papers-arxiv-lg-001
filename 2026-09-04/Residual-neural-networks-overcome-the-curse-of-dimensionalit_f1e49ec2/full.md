# Residual neural networks overcome the curse of dimensionality for semilinear heat equations

Ilkhom Mukhammadiev<sup>1</sup> and Diyora Salimova<sup>2</sup>

<sup>1</sup> Department of Applied Mathematics, University of Freiburg, Germany; e-mail: ilkhom.mukhammadiev@mathematik.uni-freiburg.de

<sup>2</sup> Department of Applied Mathematics, University of Freiburg, Germany; e-mail: diyora.salimova@mathematik.uni-freiburg.de

September 4, 2026

## Abstract

Rigorous results show that feedforward neural networks can overcome the curse of dimensionality in the numerical approximation of high-dimensional partial diferential equations (PDEs), but comparatively little is known about residual neural networks (ResNets) in the nonlinear PDE setting. We prove that ResNets overcome the curse of dimensionality in the numerical approximation of solutions of semilinear heat equations with globally Lipschitz continuous, gradient-independent nonlinearities: under polynomial growth and network approximability hypotheses on the PDE data, there exist $\eta \in ( 0 , \infty )$ and ResNets $\Psi _ { d , \varepsilon } , d \in \mathbb { N } _ { ; }$ $\varepsilon \in ( 0 , 1 ]$ , with at most $\eta d ^ { \eta } \varepsilon ^ { - \eta }$ parameters whose realizations approximate the solution in dimension d with an $L ^ { 2 } \mathrm { - e r r o r }$ of at most ε. The proof represents one deterministic realization of a multilevel Picard estimator by a ResNet whose shortcut connections transmit the spatial variable and a scalar accumulator, while the residual branches successively add the summands of the estimator. For ridge-sum initial conditions, admissible sigmoidal activations, and globally Lipschitz truncations of the nonlinearity, we obtain, for every $\xi > 0$ , the explicit bound $\ddot { C _ { \xi } d ^ { 4 + \hat { \xi } } } _ { \xi }  – ( 3 + \xi )$ on the number of parameters.

## 1 Introduction

## 1.1 The problem and the main result

Developing numerical methods for high-dimensional partial diferential equations (PDEs) that avoid the curse of dimensionality is among the central challenges of numerical analysis. Fix $T \in ( 0 , \infty )$ , a globally Lipschitz continuous function $f : \mathbb { R } \to \mathbb { R }$ , and, for every $d \in \mathbb { N } .$ , an initial value $g _ { d } \colon  { \mathbb { R } ^ { d } } \to  { \mathbb { R } }$ . We study residual neural network (ResNet) approximations of the solutions to the semilinear heat equations

$$
\begin{array} { r } { ( \frac { \partial } { \partial t } u _ { d } ) ( t , x ) = ( \frac { 1 } { 2 } \Delta _ { x } u _ { d } ) ( t , x ) + f \big ( u _ { d } ( t , x ) \big ) , \qquad u _ { d } ( 0 , x ) = g _ { d } ( x ) , } \end{array}\tag{1}
$$

for $( t , x ) \in ( 0 , T ) \times  { \mathbb { R } } ^ { d }$ . The curse of dimensionality refers here to the fact that, for general highdimensional approximation problems, the number of degrees of freedom employed by standard approximation methods may grow exponentially in the PDE dimension d and/or in the reciprocal of the prescribed approximation accuracy (cf., e.g., Bellman [9], Novak & Wo´zniakowski [50], and the references mentioned therein).

The family of PDEs in (1) includes, for example, equations with a globally Lipschitz truncation of the Allen–Cahn nonlinearity $f ( u ) = u - u ^ { 3 } ;$ ; the untruncated cubic is not globally Lipschitz continuous and is therefore not covered directly by our hypotheses (see Section 6 below). Additionally, a suitable globally Lipschitz truncation of the logistic reaction $r \mapsto r ( 1 - r )$ leads to a Fisher–KPP-type equation. After a time reversal, equations of the form (1) arise in nonlinear pricing, while more general semilinear Kolmogorov PDEs appear in stochastic optimal control. In such applications the PDE dimension d may be quite large.

The key contribution of this article is to prove that residual neural networks (ResNets) in the sense of He et al. [31, 32] approximate the solutions of (1) without the curse of dimensionality. More specifically, the main result of this article, Theorem 5.1 below, proves that, under suitable Lipschitz continuity, polynomial growth, and network approximation hypotheses, there exist $\eta \in ( 0 , \infty )$ and ResNets $\Psi _ { d , \varepsilon } , d \in \mathbb { N } , \varepsilon \in ( 0 , 1 ]$ , such that $\mathfrak { R } _ { a } ( \Psi _ { d , \varepsilon } ) \in C ( \mathbb { R } ^ { d } , \mathbb { R } )$ and

$$
\begin{array} { r } { \mathfrak { P } ( \Psi _ { d , { \varepsilon } } ) \le \eta d ^ { \eta } { \varepsilon } ^ { - \eta } \qquad \mathrm { a n d } \qquad \Big [ \int _ { [ 0 , 1 ] ^ { d } } | u _ { d } ( T , x ) - ( \mathfrak { R } _ { a } ( \Psi _ { d , { \varepsilon } } ) ) ( x ) | ^ { 2 } d x \Big ] ^ { 1 / 2 } \le { \varepsilon } , } \end{array}
$$

where P denotes the number of parameters of a ResNet and $\Re _ { a }$ denotes its realization with respect to the activation function $a \colon  { \mathbb { R } } \to  { \mathbb { R } }$

Since the real number η depends neither on d nor on ε, the number of parameters used to describe the ResNets $\Psi _ { d , { \varepsilon } }$ grows at most polynomially in both the PDE dimension d and the reciprocal $\varepsilon ^ { - 1 }$ of the prescribed approximation accuracy. In this sense, the ResNets $\Psi _ { d , { \varepsilon } }$ overcome the curse of dimensionality in the numerical approximation of the PDEs in (1). We emphasize that, as in the related results in the scientific literature, Theorem 5.1 is a deterministic expressivity result. It establishes the existence of a ResNet with the stated approximation properties, but it does not assert that a prescribed training procedure computes such a ResNet.

## 1.2 Deep learning for high-dimensional PDEs and rigorous approximation results

In recent years, neural networks have been successfully employed for the numerical approximation of solutions to high-dimensional PDEs. For example, the deep BSDE method of E et al. [15, 30] reformulates a semilinear parabolic PDE as a backward stochastic diferential equation and approximates its solution by means of stochastic gradient descent in dimensions of order 100. Further deep-learning-based approximation methods for PDEs include, for instance, the physics-informed neural networks of Raissi et al. [54], the deep Galerkin method of Sirignano & Spiliopoulos [56], and the deep Ritz method of E & Yu [18]. Numerical studies for Kolmogorov equations can, e.g., be found in Beck et al. [4], and we refer to [7] for an overview. These numerical simulations indicate that neural networks possess the flexibility required to approximate solutions of certain high-dimensional PDEs. They do not, however, by themselves establish polynomial bounds for the number of parameters used to describe the approximating networks.

There are now several rigorous results in the scientific literature which prove that neural networks possess the requisite expressive power for various classes of PDEs. The first results of this type concerned linear equations: polynomial parameter bounds for Black–Scholes and Kolmogorov equations [26, 19, 39], including the generalization error [10], space–time approximations [34], option prices under exponential L´evy models [23], and the Poisson equation with Dirichlet boundary conditions [25]; Reisinger & Zhang [55] treat nonsmooth value functions of zero-sum games of nonlinear stif systems.

For nonlinear equations, one of the principal tools is the class of multilevel Picard (MLP) approximations introduced by E et al. [16, 17] and further developed in [37, 38, 6, 8, 35]. MLP approximations are full-history recursive nonlinear Monte Carlo schemes which, for the problem classes and assumptions treated in these works, approximate solutions of semilinear parabolic PDEs at a computational cost growing at most polynomially in the PDE dimension d and the reciprocal $\varepsilon ^ { - 1 }$ of the prescribed approximation accuracy. A principal analytic ingredient of the present article is the result of Hutzenthaler et al. [36]. The authors prove that rectified feedforward neural networks approximate solutions of semilinear heat equations of the form (1) without the curse of dimensionality by showing that one realization of the MLP estimator can be represented exactly by a ReLU network. Truncated MLP approximations for the Allen– Cahn equation are developed in [6], and gradient-dependent nonlinearities in the MLP setting are treated by Hutzenthaler & Kruse [38]; the latter extends MLP convergence to nonlinearities $f = f ( u , \nabla u )$ through a stochastic fixed-point system involving the Bismut–Elworthy–Li formula. The corresponding network-representation theorem for gradient-dependent semilinear heat equations in the feedforward setting was obtained by Neufeld & Nguyen [44]; analogous results for general semilinear PDEs with gradient-dependent nonlinearities and nonconstant coeficients are due to Neufeld et al. [47] at the MLP level.

The class of PDEs covered by the network-representation theorems has been extended in several directions: to general semilinear Kolmogorov PDEs with gradient-independent Lipschitz nonlinearities and Lipschitz drift and difusion coeficients by Cioica-Licht et al. [12]; to $L ^ { p } .$ -errors for arbitrary $p \in ( 0 , \infty )$ and to further activations, including ReLU, leaky ReLU, and softplus, by Ackermann et al. [1] and Neufeld & Nguyen [45], and to space–time domains in [2]; to partial integro-diferential equations (PIDEs) for jump processes, in the linear case by Gonon & Schwab [24] and in the semilinear case by Neufeld & Wu [49] via an MLP scheme, with the corresponding expressivity result in [46] and a random deep splitting method in [48]; to random feature networks for Black–Scholes and exponential L´evy models by Gonon [21]; and to $L ^ { \infty }$ -approximation for the linear heat equation by Gonon et al. [22].

The upper bounds described above are complemented by lower bounds which identify function classes for which the curse of dimensionality cannot be overcome by network approximations with a given architecture. Grohs et al. [27] exhibit a class of high-dimensional target functions for which suficiently deep networks overcome the curse of dimensionality but shallow networks (in particular, single-hidden-layer networks) do not; the depth required for the polynomial parameter bound grows in the dimension. The present result is consistent with this lower bound, since the ResNets $\Psi _ { d , { \varepsilon } }$ produced by Theorem 5.1 are not of uniformly bounded depth. More specifically, the accumulator construction follows the finite MLP recursion tree, whose size grows with d and $\varepsilon ^ { - 1 }$ through the balancing in Lemma 3.5 below. A further samplingcomplexity lower bound of Grohs & Voigtl¨ander [28] shows that, on suitable neural-network approximation spaces, favorable approximation rates cannot in general be realized by deterministic or randomized algorithms using only polynomially many point samples; in the uniform norm, the worst-case sampling complexity sufers from the curse of dimensionality. This result does not contradict our upper bound, because Theorem 5.1 is an expressivity result and not a learning result. It does, however, show that the polynomial existence guarantee in Theorem 5.1 alone does not imply the existence of a polynomial-time training algorithm.

The classical expressivity theory for feedforward networks [33, 13, 40, 43, 53] and the quantitative ReLU rate bounds of [59, 52] provide the general background for the staircase construction in Lemma 4.3.

A recent result of Yang & Pan [58] studies ResNet approximations of high-dimensional stochastic PDEs by means of a splitting-up method and repeated cross-iteration of two ResNets; its stochastic-PDE setting and network construction difer from the deterministic MLP accumulator developed here. To the best of our knowledge, the present article provides the first expressivity result for residual networks, with parameter counts polynomial in d and $\varepsilon ^ { - 1 }$ , for the semilinear heat equations (1). The feedforward results in [36] and [1] treat the same gradientindependent semilinear heat equation, but they require an activation function which represents the identity (such as ReLU, leaky ReLU, or softplus). The residual construction developed below removes this requirement and yields explicit polynomial exponents. In comparison, [44] treats the more general gradient-dependent nonlinearity $f ( u , \nabla u )$ in the feedforward setting, whereas the present article considers a gradient-independent nonlinearity and a residual architecture with a diferent class of admissible activation functions, including bounded monotone sigmoidal activations.

## 1.3 Residual networks and their approximation theory

Residual neural networks were introduced by He et al. [31, 32] and are characterized by the presence of shortcut connections. More specifically, in the case of an identity shortcut, the ith residual update has the form

$$
x _ { i } = x _ { i - 1 } + F _ { i } ( x _ { i - 1 } ) ,
$$

where $F _ { i }$ is the realization of the ith residual branch; the recurrence can be interpreted as one step of an explicit Euler discretization of an ordinary diferential equation, an interpretation developed in [14, 29, 42, 11]. Rigorous approximation results for ResNets include the universal approximation theorem of Lin & Jegelka [41] and the control-theoretic analysis of Tabuada & Gharesifard [57]. A principal architectural ingredient in this article is the result of Baggenstos & Salimova [3], in which the skip-connection recurrence is identified with an Euler–Maruyama discretization and ResNets are shown to overcome the curse of dimensionality for a class of linear Kolmogorov PDEs.

In this article we employ the residual architecture to construct a ResNet which successively accumulates the summands occurring in an MLP approximation. More specifically, each shortcut transmits the spatial variable together with the current value of a scalar accumulator, while each accumulator update adds one summand of the MLP formula; depending on the summand, an update consists of a single residual block or a short concatenation of such blocks. The composition rule in Lemma 2.4(ii) then implies that the parameter count of the resulting ResNet grows additively under concatenation. This construction requires neither an FNN representation of the identity nor a parallelization of networks with diferent widths. Consequently, in contrast to the feedforward constructions as in [36], we do not impose the hypothesis that the activation function represents the identity. Under the abstract surrogate hypothesis in Setting 4.1, part (i) of Theorem 5.1 applies to every activation for which the required data surrogates exist. The explicit constructions in Corollary 4.5 and Lemma 4.6 show that the admissible activations in part (ii) include the admissible sigmoidal activations of Lemma 4.3, such as the logistic function and tanh.

Roughly speaking, our proof of Theorem 5.1 consists of the following steps. First, we employ the stability and complexity estimates for MLP approximations in Corollary 3.4 and Lemma 3.5 to approximate the solution of (1) by an MLP estimator built from neural-network surrogates of the PDE data. This allows us to select one deterministic realization of this estimator with the desired error. Second, the accumulator construction in Proposition 4.9 represents this realization by a ResNet, and the recursion in Lemma 4.10 proves the required polynomial bound for the number of parameters. In addition, Lemma 4.6 provides a concrete class of ridge-sum initial conditions which satisfies the abstract data hypotheses.

## 2 A calculus of feedforward and residual networks

In this section we recall, following [3, 39], the formalism for feedforward and residual networks used throughout this article and the auxiliary results needed below, notably the additivity of ResNet composition with respect to the number of parameters (Lemma 2.4).

Notation conventions. The following symbols are used throughout Section 2 and the subsequent sections.

<table><tr><td> $ { \mathbb { N } } _ { 0 }$ </td><td>N∪ {0}</td></tr><tr><td>|1·1|</td><td>the Euclidean norm;  $\textstyle \| x \| = ( \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } ) ^ { 1 / 2 }$  for  $x \in \mathbb { R } ^ { d }$ </td></tr><tr><td> $\operatorname { I d } _ { n }$ </td><td>the  $n \times n$  identity matrix</td></tr><tr><td> $\mathbf { N }$ </td><td>the set of feedforward neural networks  $( \mathrm { F N N s } ) ;$  see Definition 2.1</td></tr><tr><td> $\textstyle { \mathcal { N } } = \bigcup _ { n \in \mathbb { N } } { \mathcal { N } } _ { n }$ </td><td>the set of residual neural networks (ResNets);  $\textstyle { \mathcal { N } } _ { n }$  is the subset of ResNets of length  $n ;$  see Definition 2.2</td></tr><tr><td> $\theta$ </td><td>a generic FNN,  $\boldsymbol { \theta } = ( ( W _ { 1 } , B _ { 1 } ) , \dots , ( W _ { L } , B _ { L } ) ) \in \mathbf { N }$ </td></tr><tr><td> $\Theta$ </td><td>a generic ResNet,  $\boldsymbol { \Theta } = ( \Gamma _ { 1 } , \theta _ { 1 } , \dots , \Gamma _ { n } , \theta _ { n } ) \in \boldsymbol { N }$ </td></tr><tr><td> $\mathcal { L } ( \boldsymbol { \theta } )$ </td><td>the depth of an  $\mathrm { F N N } ;$  see Definition 2.1</td></tr><tr><td> $\theta _ { i }$ </td><td>the ith residual branch (an FNN) of a ResNet</td></tr><tr><td> $( \Gamma _ { i } , \theta _ { i } )$ </td><td>the ith residual block of a ResNet</td></tr><tr><td> $\Gamma _ { i }$ </td><td>the ith shortcut matrix of a ResNet</td></tr><tr><td> $\mathcal { T } ( \cdot ) , \mathcal { O } ( \cdot )$ </td><td>input and output dimensions (of an FNN or a ResNet); see Definitions 2.1 and 2.2</td></tr><tr><td> $\mathcal { D } ( \cdot )$ </td><td>the architecture (of an FNN or a ResNet); see Definitions 2.1 and 2.2</td></tr><tr><td> $\mathfrak { L } ( \Theta )$ </td><td>the length of a ResNet, i.e. the number of residual blocks; see Defini- tion 2.2</td></tr><tr><td> ${ \mathcal { P } } ( \theta )$ </td><td>the parameter count of an  $\mathrm { F N N } ;$  see (2)</td></tr><tr><td> $\mathfrak { P } ( \Theta )$ </td><td>the parameter count of a ResNet; see (3)</td></tr><tr><td> $\mathcal { R } _ { a } ( \theta )$ </td><td>the realization of an FNN θ with activation  $a ;$  see Definition 2.1</td></tr><tr><td> $\Re _ { a } ( \Theta )$ </td><td>the realization of a ResNet Θ with activation  $a ;$  see Definition 2.3</td></tr><tr><td> $\Theta _ { 2 } * \Theta _ { 1 }$ </td><td>the concatenation of two ResNets; see Lemma 2.4</td></tr><tr><td> $\mathcal { M } _ { a }$ </td><td>the componentwise action of  $^ { a , }$  i.e.  $\mathcal { M } _ { a } ( x _ { 1 } , \ldots , x _ { n } ) = ( a ( x _ { 1 } ) , \ldots , a ( x _ { n } ) )$ </td></tr></table>

## 2.1 Feedforward neural networks (FNNs)

Definition 2.1 (FNNs). Let

$$
\mathbf { N } = \bigcup _ { L \in \mathbb { N } } \bigcup _ { l _ { 0 } , l _ { 1 } , \dotsc , l _ { L } \in \mathbb { N } } \Big ( \sum _ { k = 1 } ^ { L } \big ( \mathbb { R } ^ { l _ { k } \times l _ { k - 1 } } \times \mathbb { R } ^ { l _ { k } } \big ) \Big ) .
$$

For $\boldsymbol { \theta } = ( ( W _ { 1 } , B _ { 1 } ) , \ldots , ( W _ { L } , B _ { L } ) ) \in \times _ { k = 1 } ^ { L } \left( \mathbb { R } ^ { l _ { k } \times l _ { k - 1 } } \times \mathbb { R } ^ { l _ { k } } \right) \subset \mathbf { N }$ define the depth ${ \mathcal { L } } ( \theta ) = L$ , the input/output dimensions $\mathcal { T } ( \theta ) = l _ { 0 } , \mathcal { O } ( \theta ) = l _ { L }$ , the architecture $\mathcal { D } ( \boldsymbol { \theta } ) = ( l _ { 0 } , l _ { 1 } , \ldots , l _ { L } )$ , and the number of parameters

$$
{ \mathcal { P } } ( \theta ) = \sum _ { k = 1 } ^ { L } l _ { k } \left( l _ { k - 1 } + 1 \right) .\tag{2}
$$

Given an activation function $a \in C ( \mathbb { R } , \mathbb { R } )$ , write $\mathcal { M } _ { a } ( y _ { 1 } , \dots , y _ { n } ) : = ( a ( y _ { 1 } ) , \dots , a ( y _ { n } ) )$ for its componentwise action, and define $x _ { k } : = \mathcal M _ { a } ( W _ { k } x _ { k - 1 } + B _ { k } )$ for $k \in \mathbb { N } \cap ( 0 , L )$ recursively from $x _ { 0 } \in \mathbb { R } ^ { l _ { 0 } }$ . The realization function $\mathcal { R } _ { a } ( \theta ) \in C ( \mathbb { R } ^ { l _ { 0 } } , \mathbb { R } ^ { l _ { L } } )$ of θ is then given by $( {  { \mathcal R } } _ { a } \theta ) ( x _ { 0 } ) : =$ $W _ { L } x _ { L - 1 } + B _ { L }$

## 2.2 Residual neural networks (ResNets)

Definition 2.2 (ResNets). The set of ResNets is $\textstyle { \mathcal { N } } = \bigcup _ { n \in \mathbb { N } } { \mathcal { N } } _ { n }$ , where for $n \in \mathbb { N }$

$$
\mathcal { N } _ { n } = \bigcup _ { ( d _ { 0 } , \ldots , d _ { n } ) \in \mathbb { N } ^ { n + 1 } } \Big \{ ( \Gamma _ { 1 } , \theta _ { 1 } , \ldots , \Gamma _ { n } , \theta _ { n } ) : \theta _ { k } \in \mathbb { N } , ~ \mathcal { T } ( \theta _ { k } ) = d _ { k - 1 } , ~ \mathcal { O } ( \theta _ { k } ) = d _ { k } , ~ \Gamma _ { k } \in \mathbb { R } ^ { d _ { k } \times d _ { k - 1 } } \Big \} .
$$

Given a ResNet $\boldsymbol { \Theta } = ( \Gamma _ { 1 } , \theta _ { 1 } , \dots , \Gamma _ { n } , \theta _ { n } ) \in \mathcal { N }$ , we call $\theta _ { 1 } , \ldots , \theta _ { n }$ its residual branches, $\Gamma _ { 1 } , \ldots , \Gamma _ { n }$ its shortcut (skip-connection) matrices, and $( \Gamma _ { i } , \theta _ { i } ) , \ i \in \{ 1 , . . . , n \}$ , its residual blocks. We define its length as $\mathfrak { L } ( \Theta ) = n$ , its input and output dimensions as $\mathcal { I } ( \Theta ) = d _ { 0 }$ and $O ( \Theta ) = d _ { n }$ its architecture as ${ \mathcal { D } } ( \Theta ) = ( d _ { 0 } , \ldots , d _ { n } )$ , and its parameter count as

$$
\mathfrak { P } ( \Theta ) = \sum _ { i = 1 } ^ { n } \big ( \mathcal { P } ( \theta _ { i } ) + d _ { i } d _ { i - 1 } \big ) .\tag{3}
$$

The above formulation follows [3, Definition 3.1].

Definition 2.3 (Realization of a ResNet). Let $a \in C ( \mathbb { R } , \mathbb { R } ) , n \in \mathbb { N } .$ , and let $\Theta = ( \Gamma _ { i } , \theta _ { i } ) _ { i = 1 } ^ { n } \in \mathcal { N } .$ with residual branches $\theta _ { i }$ and shortcut matrices $\Gamma _ { i }$ (cf. Definition 2.2). Its realization $\Re _ { a } ( \Theta )$ is the map $\mathbb { R } ^ { \mathcal { T } ( \theta _ { 1 } ) } \ni x _ { 0 } \mapsto x _ { n } \in \mathbb { R } ^ { \mathcal { O } ( \theta _ { n } ) }$ defined by the recurrence

$$
x _ { i } = \Gamma _ { i } x _ { i - 1 } + ( \mathcal { R } _ { a } \theta _ { i } ) ( x _ { i - 1 } ) , \qquad \forall i \in \{ 1 , \ldots , n \} .\tag{4}
$$

This formulation of the realization recurrence follows [3, Definition 3.2].

When $d _ { 0 } = \cdots = d _ { n }$ and $\Gamma _ { i } = \mathrm { I d } _ { d _ { 0 } }$ for all i, (4) has the additive form $x _ { i } = x _ { i - 1 } + ( \mathcal { R } _ { a } \theta _ { i } ) ( x _ { i - 1 } )$ of an explicit Euler-type update. Such updates encompass Euler–Maruyama steps when the residual branch represents a drift-plus-difusion increment with a fixed Brownian increment; see [3]. Figure 1 illustrates the structure.

$$
x _ { 0 } \xrightarrow [ \ddots ] { } \left( \begin{array} { l } { \theta _ { 1 } } \\ { \theta _ { 1 } } \end{array} \right) \xrightarrow [ \ddots ] { } \left( \begin{array} { l } { \theta _ { 2 } } \\ { \theta _ { 3 } } \end{array} \right) \xrightarrow [ \ddots ] { } \left( \begin{array} { l } { \theta _ { 2 } } \\ { \theta _ { 3 } } \end{array} \right) \xrightarrow [ \ddots ] { } \left( \begin{array} { l } { \iff } \\ { \iff } \end{array} \right) \xrightarrow [ \ddots ] { } x _ { n - 1 } \xrightarrow [ \ddots ] { } \left( \begin{array} { l } { \theta _ { n } } \\ { \theta _ { n } } \end{array} \right) \xrightarrow [ \ddots ] { } \left( \bigoplus \right) \cdots x _ { n - 1 }
$$

Figure 1: Realization of a ResNet. The boxes labeled $\theta _ { i }$ represent the residual branches, and the dashed arrows represent the linear shortcuts $\Gamma _ { i }$ $\mathrm { I f } \ \Gamma _ { i } = \mathrm { I d }$ , then the recurrence has the additive structure of an explicit Euler-type update.

Lemma 2.4 (Composition of ResNets). Let $a \in C ( \mathbb { R } , \mathbb { R } ) , m , n \in \mathbb { N }$ , and let $\Theta _ { 1 } = ( \Gamma _ { 1 } ^ { 1 } , \theta _ { 1 } ^ { 1 } , \dots ,$ $\Gamma _ { n } ^ { 1 } , \theta _ { n } ^ { 1 } )$ and $\Theta _ { 2 } = ( \Gamma _ { 1 } ^ { 2 } , \theta _ { 1 } ^ { 2 } , \dots , \Gamma _ { m } ^ { 2 } , \theta _ { m } ^ { 2 } )$ be ResNets with ${ \mathcal O } ( \theta _ { n } ^ { 1 } ) = { \mathcal T } ( \theta _ { 1 } ^ { 2 } )$ . We define their composition by concatenation,

$$
\begin{array} { r } { \Theta _ { 2 } \ast \Theta _ { 1 } = \big ( \Gamma _ { 1 } ^ { 1 } , \theta _ { 1 } ^ { 1 } , \ldots , \Gamma _ { n } ^ { 1 } , \theta _ { n } ^ { 1 } , \Gamma _ { 1 } ^ { 2 } , \theta _ { 1 } ^ { 2 } , \ldots , \Gamma _ { m } ^ { 2 } , \theta _ { m } ^ { 2 } \big ) \in \pmb { \mathscr N } . } \end{array}
$$

Then

$$
( i ) \quad \Re _ { a } ( \Theta _ { 2 } * \Theta _ { 1 } ) = \Re _ { a } ( \Theta _ { 2 } ) \circ \Re _ { a } ( \Theta _ { 1 } ) , \qquad ( i i ) \quad \Re ( \Theta _ { 2 } * \Theta _ { 1 } ) = \Re ( \Theta _ { 1 } ) + \Re ( \Theta _ { 2 } ) .
$$

Proof of Lemma ${ 2 . 4 } .$ The assertions follow from [3, Definition 3.3 and Lemma 3.4]. For completeness we note that the recurrence (4) on the concatenated ResNet first runs through the blocks of $\Theta _ { 1 }$ , reaching $\mathfrak { R } _ { a } ( \Theta _ { 1 } ) ( x _ { 0 } )$ , and then through the blocks of $\Theta _ { 2 }$ , while (3) adds the two parameter counts. This proves assertions (i)–(ii). □

Lemma 2.5 (One-coordinate lift). Let $a \in C ( \mathbb { R } , \mathbb { R } )$ and let $\boldsymbol { \Theta } = ( ( \Gamma _ { i } , \theta _ { i } ) ) _ { i = 1 } ^ { n } \in \mathcal { N } _ { n }$ have architecture ${ \mathcal { D } } ( \Theta ) = ( d _ { 0 } , \ldots , d _ { n } )$ . Then there exists a ResNet Lift $( \Theta ) \in \mathcal { N } _ { n }$ with the architecture $( d _ { 0 } + 1 , \ldots , d _ { n } + 1 )$ such that

$$
\Re _ { a } ( \mathrm { L i f t } ( \Theta ) ) ( x , s ) = \big ( \Re _ { a } ( \Theta ) ( x ) , s \big ) \qquad ( x \in \mathbb { R } ^ { d _ { 0 } } , ~ s \in \mathbb { R } ) ,\tag{5}
$$

and

$$
\mathfrak { P } ( \mathrm { L i f t } ( \Theta ) ) \le 4 \mathfrak { P } ( \Theta ) .\tag{6}
$$

No assumption on a beyond continuity is required.

Proof of Lemma 2.5. For each $i \in \{ 1 , \ldots , n \}$ , let $\theta _ { i } ^ { \uparrow }$ be an FNN of input dimension $d _ { i - 1 } + 1$ and output dimension $d _ { i } + 1$ whose realization is

$$
( \mathcal { R } _ { a } \theta _ { i } ^ { \uparrow } ) ( z , s ) = \big ( ( \mathcal { R } _ { a } \theta _ { i } ) ( z ) , 0 \big ) .
$$

If $\theta _ { i }$ has depth one, this is obtained by appending one zero column and one zero row to its weight matrix and one zero coordinate to its bias vector; the parameter count then satisfies $\mathcal { P } ( \theta _ { i } ^ { \uparrow } ) = ( d _ { i } + 1 ) ( d _ { i - 1 } + 2 ) \leq 3 d _ { i } ( d _ { i - 1 } + 1 ) = 3 \mathcal { P } ( \theta _ { i } )$ , using $d _ { i } , d _ { i - 1 } \geq 1$ . If its depth L is at least two, write $\mathcal { D } ( \theta _ { i } ) = ( d _ { i - 1 } , m _ { 1 } , \dots , m _ { L - 1 } , d _ { i } )$ , adjoin one zero column to the first weight matrix and one zero row and bias coordinate to the last afine layer, and leave all intermediate layers unchanged; this increases the count by $m _ { 1 } + ( m _ { L - 1 } + 1 ) \leq 2 \mathcal { P } ( \theta _ { i } )$ , since $m _ { 1 } \leq \mathcal { P } ( \theta _ { i } )$ and $m _ { L - 1 } + 1 \le \mathcal { P } ( \theta _ { i } )$ . In either case $\mathcal { P } ( \theta _ { i } ^ { \uparrow } ) \leq 3 \mathcal { P } ( \theta _ { i } )$ . Let

$$
\Gamma _ { i } ^ { \uparrow } = \binom { \Gamma _ { i } } { 0 } \ 0 \in \mathbb { R } ^ { ( d _ { i } + 1 ) \times ( d _ { i - 1 } + 1 ) } .
$$

Since $d _ { i } , d _ { i - 1 } \geq 1$ , the shortcut count satisfies $( d _ { i } + 1 ) ( d _ { i - 1 } + 1 ) \leq 4 d _ { i } d _ { i - 1 }$ . The recurrence (4) for $( ( \Gamma _ { i } ^ { \uparrow } , \theta _ { i } ^ { \uparrow } ) ) _ { i = 1 } ^ { n }$ leaves the last coordinate unchanged and runs the original recurrence in the first coordinates, which proves (5). Summing the blockwise bounds in (3) gives (6). □

## 3 Semilinear heat equations and MLP approximations

In this section we recall, in the form required below, the MLP approximations of [36, 37, 16]. Following the organization of [36, Section 2], we collect the data, the probability space, the stochastic fixed-point equation, and the MLP estimator in Setting 3.1. Every result in this section is formulated in this setting. More specifically, (9) below records the stochastic fixedpoint representation of the considered PDE solution, and (12) defines the approximation which will subsequently be represented by a ResNet. The two quantitative ingredients used in the proof of Theorem 5.1 are the stability estimate in Corollary 3.4 and the complexity estimate in Lemma 3.5 below.

Setting 3.1 (Semilinear heat equation and its MLP approximation, cf. [36, Setting 2.1]). Let $d \in \mathbb { N }$ , T ${ \bf \Phi } , L , B \in ( 0 , \infty ) , p \in [ 1 , \infty ) , q \in [ 2 , \infty )$ , and $\Delta \in ( 0 , 1 ]$ . Let $g _ { 1 } , g _ { 2 } \in C (  { \mathbb { R } } ^ { d } ,  { \mathbb { R } } )$ and $f _ { 1 } , f _ { 2 } \in C ( \mathbb { R } , \mathbb { R } )$ satisfy, for all $v , w \in \mathbb { R }$ and all $\boldsymbol { x } \in \mathbb { R } ^ { d }$ , the Lipschitz bounds and growth bounds

$$
| f _ { i } ( w ) - f _ { i } ( v ) | \leq L | w - v | , \qquad \operatorname* { m a x } \{ | f _ { i } ( 0 ) | , | g _ { i } ( x ) | \} \leq B ( 1 + \| x \| ) ^ { p } , \qquad i \in \{ 1 , 2 \} ,\tag{7}
$$

and the perturbation bound

$$
\operatorname* { m a x } \{ | f _ { 1 } ( v ) - f _ { 2 } ( v ) | , | g _ { 1 } ( x ) - g _ { 2 } ( x ) | \} \leq \Delta { \bigl ( } ( 1 + \| x \| ) ^ { p q } + | v | ^ { q } { \bigr ) } .\tag{8}
$$

Here $( g _ { 1 } , f _ { 1 } )$ represents the original PDE data, whereas $( g _ { 2 } , f _ { 2 } )$ represents perturbed data which will later be chosen as network realizations. If $( g _ { 2 } , f _ { 2 } ) = ( g _ { 1 } , f _ { 1 } )$ , then the perturbation bound holds for every $\Delta \in ( 0 , 1 ]$ . Let ${ \mathfrak { T } } = \bigcup _ { n \in \mathbb { N } } \mathbb { Z } ^ { n }$ , and let $( \Omega , \mathcal { F } , \mathbb { P } )$ be a probability space which supports a standard d-dimensional Brownian motion W, i.i.d. random variables $\mathfrak { u } ^ { \vartheta } , \vartheta \in \mathfrak { T }$ which are uniformly distributed on [0, 1], and independent standard d-dimensional Brownian motions $W ^ { \vartheta } , \vartheta \in \mathfrak { T }$ . Assume that W, $( \mathfrak { u } ^ { \vartheta } ) _ { \vartheta \in \mathfrak { T } }$ , and $( W ^ { \vartheta } ) _ { \vartheta \in \mathfrak { T } }$ are mutually independent. For every $\vartheta \in \mathfrak { T }$ and $t \in [ 0 , T ]$ , let $\mathcal { U } _ { t } ^ { \vartheta } = t + ( T - t ) \mathbf { u } ^ { \vartheta }$

For $i \in \{ 1 , 2 \}$ let $v _ { i } \in C ( [ 0 , T ] \times \mathbb { R } ^ { d } , \mathbb { R } )$ denote the unique at most polynomially growing solution of the stochastic fixed-point equation: for all $s \in [ 0 , T ] , x \in \mathbb { R } ^ { d }$ 2

$$
v _ { i } ( s , x ) = \mathbb { E } \bigg [ g _ { i } \big ( x + \mathbf { W } _ { T - s } \big ) + \int _ { s } ^ { T } f _ { i } \big ( v _ { i } ( t , x + \mathbf { W } _ { t - s } ) \big ) d t \bigg ] .\tag{9}
$$

Here at most polynomially growing means that there exist $c , r \in ( 0 , \infty )$ , possibly depending on $i$ and $d ,$ such that $| v _ { i } ( s , x ) | \leq c ( 1 + \| x \| ) ^ { r }$ for all $( s , x ) \in [ 0 , T ] \times \mathbb { R } ^ { d } ;$ existence and uniqueness in this class are recalled after this setting. Whenever the semilinear heat equation (1) with data $( g _ { i } , f _ { i } )$ possesses an at most polynomially growing solution $\mathsf { u } _ { i }$ in the class

$$
\mathsf { u } _ { i } \in C ( [ 0 , T ] \times \mathbb { R } ^ { d } , \mathbb { R } ) \cap C ^ { 1 , 2 } ( ( 0 , T ] \times \mathbb { R } ^ { d } , \mathbb { R } ) ,\tag{10}
$$

the Feynman–Kac/Duhamel principle (with W the standard Brownian motion of generator $\scriptstyle { \frac { 1 } { 2 } } \Delta$ so that no rescaling is needed) shows that the time reversal $( s , x ) \mapsto \mathbf { u } _ { i } ( T - s , x )$ solves (9); by uniqueness,

$$
v _ { i } ( s , x ) = \mathbf { u } _ { i } ( T - s , x ) \qquad ( s \in [ 0 , T ] , \ x \in \mathbb { R } ^ { d } ) .\tag{11}
$$

In particular $v _ { i } ( T , \cdot ) = g _ { i }$ and, whenever $\mathsf { u } _ { i }$ exists, $v _ { i } ( 0 , \cdot ) = { \mathbf { u } } _ { i } ( T , \cdot )$ is the time-T solution that the main theorem approximates. The regularity class (10) allows for merely continuous initial data $g _ { i }$ that are smoothed instantaneously by the heat flow.

For $i \in \{ 1 , 2 \} , M \in \mathbb { N } , \vartheta \in \mathfrak { T }$ , and $n \in \mathbb { N } _ { 0 } \cup \{ - 1 \}$ define the multilevel Picard estimator $U _ { n , M } ^ { i , \vartheta } \colon [ 0 , T ] \times \mathbb { R } ^ { d } \times \Omega $ R by $U _ { - 1 , M } ^ { i , \vartheta } = U _ { 0 , M } ^ { i , \vartheta } = 0$ and, for $n \in \mathbb { N }$

$$
\begin{array} { l } { { \displaystyle U _ { n , M } ^ { i , \vartheta } ( t , x ) = \frac { 1 } { M ^ { n } } \sum _ { j = 1 } ^ { M ^ { n } } g _ { i } \big ( x + W _ { T } ^ { ( \vartheta , 0 , - j ) } - W _ { t } ^ { ( \vartheta , 0 , - j ) } \big ) } } \\ { { \displaystyle \qquad + \sum _ { l = 0 } ^ { n - 1 } \frac { T - t } { M ^ { n - l } } \sum _ { j = 1 } ^ { M ^ { n - l } } \left[ \big ( f _ { i } \circ U _ { l , M } ^ { i , ( \vartheta , l , j ) } \big ) - \mathbf { 1 } _ { \mathbb { R } } ( l ) \big ( f _ { i } \circ U _ { l - 1 , M } ^ { i , ( \vartheta , - l , j ) } \big ) \right] } } \\ { { \displaystyle \qquad \times \left( { \mathcal { U } _ { t } ^ { ( \vartheta , l , j ) } } , x + W _ { \mathcal { U } _ { t } ^ { ( \vartheta , l , j ) } } ^ { ( \vartheta , l , j ) } - W _ { t } ^ { ( \vartheta , l , j ) } \right) } . } \end{array}\tag{12}
$$

The estimator $U _ { n , M } ^ { i , \vartheta } ( t , \cdot )$ approximates the fixed-point solution $v _ { i } ( t , \cdot )$ ; in particular $U _ { N , M } ^ { i , 0 } ( 0 , \cdot )$ approximates $v _ { i } ( 0 , \cdot )$ . We write $v : = v _ { 1 } , U _ { n , M } ^ { \vartheta } : = U _ { n , M } ^ { 1 , \vartheta } , g : = g _ { 1 }$ , and $f : = f _ { 1 }$ when only the true data are involved, and we retain the notation $U _ { n , M } ^ { 2 , \vartheta }$ for the estimator associated with the surrogate data $( g _ { 2 } , f _ { 2 } )$

For globally Lipschitz nonlinearities $f _ { i }$ and at most polynomially growing $g _ { i } .$ existence and uniqueness of the at most polynomially growing continuous solution $v _ { i }$ of (9) follow from [5, Corollary 3.10]; for the classical parabolic background and the Feynman–Kac representation used in (11) see Friedman [20] for the linear theory, Pardoux & Peng [51] for the connection to backward stochastic diferential equations as a nonlinear Feynman–Kac formula, and [36, Section 2] for the formulation adapted to the present setting. Since $f$ is gradient-independent, the nonlinear operator $F ( v ) ( t , x ) = f ( v ( t , x ) )$ acts pointwise, which is exactly what the MLP recursion exploits.

For every fixed $\omega \in \Omega$ , all Brownian increments and random times in (12) are deterministic, so that $x \mapsto U _ { n , M } ^ { \vartheta } ( 0 , x ) ( \omega )$ is obtained from $g$ and $f$ through finitely many afine shifts, compositions, scalar multiplications, and additions; this is used in Section 4 to represent one realization of the MLP approximation by a ResNet.

## 3.1 Two quantitative inputs

Lemmas 3.2 and 3.3 and Corollary 3.4 specialize [36, Lemmas 2.2–2.3 and Corollary 2.4] to Setting 3.1; the random-time MLP error factor used in Corollary 3.4 comes from [37, Theorem 3.5]. We include the proofs in Section A to record the constants needed below. Lemma 3.5 is the elementary balancing estimate for this error factor.

Lemma 3.2 (A priori moment bound, [36, Lemma 2.2]). In Setting 3.1, the solutions $v _ { 1 } , v _ { 2 }$ of (9) satisfy, for all $i \in \{ 1 , 2 \} , \ x \in \mathbb { R } ^ { d }$

$$
\operatorname* { s u p } _ { t \in [ 0 , T ] } \left( \mathbb { E } \left[ \vert v _ { i } ( t , x + \mathbf { W } _ { t } ) \vert ^ { q } \right] \right) ^ { 1 / q } \leq e ^ { L T } ( T + 1 ) B \Big [ \operatorname* { s u p } _ { t \in [ 0 , T ] } \left( \mathbb { E } \left[ ( 1 + \Vert x + \mathbf { W } _ { t } \Vert ) ^ { p q } \right] \right) ^ { 1 / q } \Big ] .
$$

Lemma 3.3 (PDE stability, [36, Lemma 2.3]). In Setting 3.1, the solutions v<sub>1</sub>, v<sub>2</sub> of (9) satisfy, for all $t \in [ 0 , T ] , x \in \mathbb { R } ^ { d }$

$$
\begin{array} { r l } & { \mathbb { E } \big [ | v _ { 1 } ( t , x + \mathbf { W } _ { t } ) - v _ { 2 } ( t , x + \mathbf { W } _ { t } ) | \big ] } \\ & { \quad \leq \Delta \big ( e ^ { L T } ( T + 1 ) \big ) ^ { q + 1 } ( B ^ { q } + 1 ) \Big ( 1 + \| x \| + \big ( \mathbb { E } [ \| \mathbf { W } _ { T } \| ^ { p q } ] \big ) ^ { 1 / ( p q ) } \Big ) ^ { p q } . } \end{array}
$$

Corollary 3.4 (MLP stability bound, [36, Corollary 2.4]). In Setting 3.1, for $\boldsymbol { x } \in \mathbb { R } ^ { d }$ and $N , M \in \mathbb { N }$ , the surrogate-data estimator $\dot { U } _ { N , M } ^ { 2 , 0 }$ approximates the true solution $v _ { 1 }$ with

$$
\begin{array} { r l r } {  { \big ( \mathbb { E } \big [ | U _ { N , M } ^ { 2 , 0 } ( 0 , x ) - v _ { 1 } ( 0 , x ) | ^ { 2 } \big ] \big ) ^ { 1 / 2 } \leq \big ( e ^ { L T } ( T + 1 ) \big ) ^ { q + 1 } ( B ^ { q } + 1 ) \Big ( \Delta + \frac { e ^ { M / 2 } ( 1 + 2 L T ) ^ { N } } { M ^ { N / 2 } } \Big ) } } \\ & { } & { \times ( 1 + \| x \| + \big ( \mathbb { E } [ \| \mathbf { W } _ { T } \| ^ { p q } ] \big ) ^ { \frac { 1 } { p q } } ) ^ { p q } . } \end{array}\tag{13}
$$

For $M = N$ , the following elementary estimate balances the MLP error factor with a recursive representation factor $( A N ) ^ { N }$

Lemma 3.5 (MLP balancing estimate). Let $L , T \in ( 0 , \infty )$ $A \in [ 1 , \infty )$ , and $\xi \in ( 0 , \infty )$ . Then there exists $C _ { \xi , A } \in ( 0 , \infty )$ , depending only on $L , T , \xi , A$ , such that, for every $\varepsilon \in ( 0 , 1 ]$ , there exists $N \in \mathbb { N } \cap [ 2 , \infty )$ satisfying

$$
\frac { e ^ { N / 2 } ( 1 + 2 L T ) ^ { N } } { N ^ { N / 2 } } \leq \varepsilon\tag{14}
$$

and

$$
( A N ) ^ { N } \leq C _ { \xi , A } \varepsilon ^ { - ( 2 + \xi ) } .\tag{15}
$$

## 4 ResNet representation of MLP estimators

In this section we establish that every fixed realization of the MLP approximation can be represented by a ResNet with a polynomially bounded number of parameters. We first formulate the data approximation hypotheses in Setting 4.1. The one-dimensional approximation result in Lemma 4.3 yields, in Corollary 4.5 and Lemma 4.6, explicit families which satisfy these hypotheses. Thereafter, Lemma 4.8 constructs the elementary residual updates, Proposition 4.9 represents one realization of the MLP approximation, and Lemma 4.10 estimates the number of parameters of the resulting ResNet. The following setting is the residual-network analogue of [3, Setting 4.1] and [36, Theorem 1.1].

Setting 4.1 (Network-representable data). Let $a \in C ( \mathbb { R } , \mathbb { R } )$ , let L, $B \in ( 0 , \infty )$ and $p \in [ 1 , \infty )$ let $f \in C ( \mathbb { R } , \mathbb { R } )$ satisfy $\operatorname { L i p } ( f ) \leq L$ , and, for every d $\ l \in \mathbb { N }$ , let $g _ { d } \in C (  { \mathbb { R } } ^ { d } ,  { \mathbb { R } } )$ . Assume there exist families of FNNs $\left( \mathfrak { f } _ { \delta } \right) _ { \delta \in ( 0 , 1 ] } \subseteq \mathbf { N }$ with $\begin{array} { r } { \mathcal { T } ( \mathfrak { f } _ { \delta } ) = \mathcal { O } ( \mathfrak { f } _ { \delta } ) = 1 } \end{array}$ and, for each $d \in \mathbb { N } , ( \mathfrak { g } _ { d , \delta } ) _ { \delta \in ( 0 , 1 ] } \subseteq \mathbb { N }$ with $\mathcal { T } ( \mathfrak { g } _ { d , \delta } ) = d$ and $\mathcal { O } ( { \mathfrak { g } } _ { d , \delta } ) = 1$ , and constants $\kappa , C _ { g } , L _ { f } \in ( 0 , \infty )$ such that for all $d \in \mathbb { N }$ $\delta \in ( 0 , 1 ] , v , w \in \mathbb { R } , x \in \mathbb { R } ^ { d } \colon$

$$
\left| f ( v ) - ( \mathcal { R } _ { a } \mathfrak { f } _ { \delta } ) ( v ) \right| \leq \delta , \quad | g _ { d } ( x ) - ( \mathcal { R } _ { a \mathfrak { g } d , \delta } ) ( x ) | \leq C _ { g } \delta d ^ { \kappa } ( 1 + \| x \| ^ { \kappa } ) ,\tag{16}
$$

$$
| ( \mathcal { R } _ { a } \mathfrak { f } _ { \delta } ) ( w ) - ( \mathcal { R } _ { a } \mathfrak { f } _ { \delta } ) ( v ) | \le L _ { f } | w - v | , \qquad | ( \mathcal { R } _ { a } \mathfrak { f } _ { \delta } ) ( 0 ) | \le B ,
$$

$$
| ( \mathcal { R } _ { a } \mathfrak { g } _ { d , { \delta } } ) ( x ) | \le B ( 1 + \| x \| ) ^ { p } ,\tag{17}
$$

$$
\begin{array} { r } { \mathcal { P } ( \mathfrak { f } _ { \delta } ) + \mathcal { P } ( \mathfrak { g } _ { d , \delta } ) \le C _ { g } d ^ { \kappa } \delta ^ { - \kappa } . } \end{array}\tag{18}
$$

Remark 4.2. The polynomial bound in (18) expresses that the data f and $g _ { d }$ do not themselves carry the curse of dimensionality. This hypothesis is analogous to the corresponding assumptions in [36, 3]. In contrast to the feedforward arguments in [36, 39], however, we do not require that the activation function a admit an exact FNN representation of the identity. Indeed, the residual updates in Proposition 4.9 are combined by the concatenation rule in Lemma 2.4, and no identity subnetwork is inserted between consecutive updates.

When Settings 3.1 and 4.1 are combined, the bounds in (17) allow the true and surrogate data to be controlled by common constants, uniformly in the approximation accuracy δ. Without a uniform Lipschitz bound for $\mathcal { R } _ { a } ( { \mathfrak { f } } _ { \delta } )$ , the constants in the stability estimate of Corollary 3.4 would depend on $\delta ,$ and the polynomial complexity estimate would in general be lost. We therefore replace L by max $\{ L , L _ { f } \}$ and enlarge $B ,$ if necessary, so that $B$ dominates both the surrogate value bound $B _ { f }$ in Corollary 4.5 and the uniform bound for the g -surrogates in Lemma 4.6. Henceforth, L and B denote these enlarged constants.

We next verify the hypotheses of Setting 4.1 for a concrete class of initial conditions. The class consists of finite sums of ridge functions of the form

$$
g _ { d } ( x ) = \sum _ { j = 1 } ^ { J } \phi _ { j } \big ( \langle \alpha _ { j } ^ { d } , x \rangle \big ) .
$$

Such functions arise, for example, as bounded truncations of European basket option payofs, as single-neuron initial conditions for Allen–Cahn-type equations, and as initial conditions for Fisher–Kolmogorov-type fronts propagating in a fixed direction.

We first establish the one-dimensional approximation result; its final assertion, proved by a variation-adapted staircase, serves the ridge profiles of Lemma 4.6. For $h \colon \mathbb { R }  \mathbb { R }$ and an interval $I \subseteq \mathbb { R }$ we write $\operatorname { T V } ( h ; I )$ for the total variation of $h$ on I, that is, the supremum of $\begin{array} { r } { \sum _ { k } | h ( t _ { k } ) - h ( t _ { k - 1 } ) | } \end{array}$ over all finite increasing sequences $\left( t _ { k } \right)$ in $I ,$ and $\mathrm { T V } ( h ) : = \mathrm { T V } ( h ; \mathbb { R } )$

Lemma 4.3 (One-dimensional sigmoidal approximation). Let $a \in C ^ { 1 } ( \mathbb { R } , \mathbb { R } )$ be nondecreasing and assume that the finite limits

$$
A _ { - } : = \operatorname* { l i m } _ { r \to - \infty } a ( r ) \in \mathbb { R } \qquad a n d \qquad A _ { + } : = \operatorname* { l i m } _ { r \to + \infty } a ( r ) \in \mathbb { R }\tag{19}
$$

exist and satisfy $A _ { - } < A _ { + }$ . Let $\widetilde { a } : = ( a - A _ { - } ) / ( A _ { + } - A _ { - } )$ , and assume that

$$
\mathrm { T V } ( \widetilde { a } ^ { \prime } ) < \infty , \qquad \widetilde { \Lambda } _ { a } : = \int _ { \mathbb { R } } \left| \widetilde { a } ( r ) - \mathbf { 1 } _ { ( 0 , \infty ) } ( r ) \right| d r < \infty .\tag{20}
$$

We call such an a an admissible sigmoidal activation. Then there exists $C _ { a } \in [ 1 , \infty )$ , depending only on a, such that for every Lipschitz function $\phi : \mathbb { R }  \mathbb { R }$ , every $R \in [ 1 , \infty )$ , and every $n \in \mathbb { N }$ there exists an FNN $\phi _ { n , R } \in \mathbf { N }$ satisfying

$$
\begin{array} { r } { \mathcal { D } ( \phi _ { n , R } ) = ( 1 , n , 1 ) , \qquad \mathcal { P } ( \phi _ { n , R } ) = 3 n + 1 , } \end{array}\tag{21}
$$

$$
\operatorname* { s u p } _ { v \in [ - R , R ] } | \phi ( v ) - ( \mathcal { R } _ { a } \phi _ { n , R } ) ( v ) | \leq C _ { a } \mathrm { L i p } ( \phi ) \frac { R } { n } ,\tag{22}
$$

and

$$
\mathrm { L i p } \big ( \mathcal { R } _ { a } \phi _ { n , R } \big ) \leq C _ { a } \mathrm { L i p } ( \phi ) .\tag{23}
$$

$H ,$ in addition, $\phi$ is constant on each of $( - \infty , - R ]$ and $[ R , \infty )$ , then

$$
\operatorname* { s u p } _ { v \in \mathbb { R } } | \phi ( v ) - ( \mathcal { R } _ { a } \phi _ { n , R } ) ( v ) | \leq C _ { a } \mathrm { L i p } ( \phi ) \frac { R } { n } .\tag{24}
$$

Finally, if ϕ has finite total variation, then, for every $\delta \in ( 0 , 1 ]$ , there exist $m \in \mathbb { N }$ and an FNN $\phi _ { \delta } \in \mathbf { N }$ with

$$
\mathcal { D } ( \phi _ { \delta } ) = ( 1 , m , 1 ) , \qquad \operatorname* { s u p } _ { v \in \mathbb { R } } | \phi ( v ) - ( \mathcal { R } _ { a } \phi _ { \delta } ) ( v ) | \leq \delta , \qquad m \leq \operatorname* { m a x } \Big \{ 1 , \Big \lceil \frac { 3 \mathrm { T V } ( \phi ) } { \delta } \Big \rceil - 1 \Big \} .\tag{25}
$$

The proof of Lemma 4.3 is provided in Section B.

Remark 4.4. The logistic function, tanh, the error function, and the algebraic sigmoid $r \mapsto$ $r / \sqrt { 1 + r ^ { 2 } }$ are admissible sigmoidal activations: in each case $\widetilde { a } ^ { \prime }$ is unimodal, hence of bounded variation, and $\widetilde { a }$ approaches its limits at an integrable rate. None of them represents the identity through the two-unit construction of [36, 39]; conversely, ReLU does not satisfy (19) but represents the identity through $r = { \mathrm { R e L U } } ( r ) - { \mathrm { R e L U } } ( - r )$ and is therefore covered by [36].

Corollary 4.5 (Uniformly Lipschitz surrogates for globally Lipschitz truncations). Let a be an admissible sigmoidal activation in the sense of Lemma $4 . 3 ,$ let $L , M _ { f } \in \mathsf { \Gamma } ( 0 , \infty )$ , and let $f \in C ( \mathbb { R } , \mathbb { R } )$ satisfy $\operatorname { L i p } ( f ) \leq L$ and be constant on each $o f \left( - \infty , - M _ { f } \right]$ and $[ M _ { f } , \infty ) - a$ globally Lipschitz truncation. Then there exist a family $( \mathfrak { f } _ { \delta } ) _ { \delta \in ( 0 , 1 ] } \ \subseteq$ N and constants $L _ { f } , B _ { f } , C _ { f } \in$ $( 0 , \infty )$ , where $L _ { f }$ and $C _ { f }$ depend only on $a , L , M _ { f }$ , whereas $\dot { \boldsymbol { B } } _ { f }$ depends only on $| f ( 0 ) |$ , such that for every $\delta \in ( 0 , 1 ]$

$$
\operatorname* { s u p } _ { v \in \mathbb { R } } | f ( v ) - ( \mathcal { R } _ { a } \mathfrak { f } _ { \delta } ) ( v ) | \leq \delta , \quad \mathrm { L i p } ( \mathcal { R } _ { a } \mathfrak { f } _ { \delta } ) \leq L _ { f } , \quad | ( \mathcal { R } _ { a } \mathfrak { f } _ { \delta } ) ( 0 ) | \leq B _ { f } , \quad \mathcal { P } ( \mathfrak { f } _ { \delta } ) \leq C _ { f } \delta ^ { - 1 } .\tag{26}
$$

Moreover, $\mathfrak { f } _ { \delta }$ may be chosen with architecture $( 1 , n _ { f } , 1 )$ and $n _ { f } \le C _ { f } \delta ^ { - 1 }$ . In particular f satisfies the f-part of (16)–(17) with uniform Lipschitz constant $L _ { f }$ , and the f-part of (18) with exponent $\kappa = 1$

Proof of Corollary $4 . 5 .$ Set $R : = M _ { f } + 1$ , and let $C _ { a }$ be the constant in Lemma 4.3. For $\delta \in ( 0 , 1 ]$ , choose

$$
n : = \operatorname* { m a x } \{ 1 , \lceil C _ { a } L R / \delta \rceil \}
$$

and apply Lemma 4.3 to $f .$ . Since $f$ is constant on $( - \infty , - R ]$ and on $[ R , \infty )$ , (24) gives

$$
\operatorname* { s u p } _ { v \in \mathbb { R } } \vert f ( v ) - ( \mathcal { R } _ { a } \mathfrak { f } _ { \delta } ) ( v ) \vert \leq \frac { C _ { a } L R } { n } \leq \delta , \qquad \mathrm { L i p } ( \mathcal { R } _ { a } \mathfrak { f } _ { \delta } ) \leq C _ { a } L .
$$

Consequently,

$$
| ( \mathcal { R } _ { a } \mathfrak { f } _ { \delta } ) ( 0 ) | \le | f ( 0 ) | + 1 .
$$

Moreover, (21) and $\delta \leq 1$ yield

$$
\begin{array} { r } { \mathcal { D } ( \mathfrak { f } _ { \delta } ) = ( 1 , n , 1 ) , \qquad n \leq ( 1 + C _ { a } L R ) \delta ^ { - 1 } , \qquad \mathcal { P } ( \mathfrak { f } _ { \delta } ) = 3 n + 1 \leq ( 4 + 3 C _ { a } L R ) \delta ^ { - 1 } . } \end{array}
$$

The claimed constants may therefore be chosen as $L _ { f } : = C _ { a } L , B _ { f } : = 1 + | f ( 0 ) |$ , and $C _ { f } : =$ $4 + 3 C _ { a } L R$ . These estimates establish the f-parts of $( 1 6 )  ( 1 8 )$ □

Lemma 4.6 (Ridge-sum approximation). Let a be an admissible sigmoidal activation in the sense of Lemma $4 . 3 ,$ let $J \in \mathbb { N } , L _ { \phi } \in ( 0 , \infty )$ , and $M _ { \phi } , T _ { \phi } \in [ 0 , \infty )$ , and let $\phi _ { 1 } , \ldots , \phi _ { J } \in C ( \mathbb { R } , \mathbb { R } )$ satisfy

$$
\mathrm { L i p } ( \phi _ { j } ) \leq L _ { \phi } , \qquad | \phi _ { j } ( 0 ) | \leq M _ { \phi } , \qquad \mathrm { T V } ( \phi _ { j } ) \leq T _ { \phi }\tag{27}
$$

for all $j \in \{ 1 , \dotsc , J \}$ . For every d $\in \mathbb { N } _ { \cdot }$ , let $\alpha _ { 1 } ^ { d } , \ldots , \alpha _ { J } ^ { d } \in \mathbb { R } ^ { d }$ and define

$$
g _ { d } ( x ) : = \sum _ { j = 1 } ^ { J } \phi _ { j } \bigl ( \langle \alpha _ { j } ^ { d } , x \rangle \bigr ) \qquad ( x \in  { \mathbb { R } } ^ { d } ) .\tag{28}
$$

Then there exists $C \in ( 0 , \infty )$ , depending only on J and $T _ { \phi } ,$ such that for all $d \in \mathbb { N }$ and $\delta \in ( 0 , 1 ]$ there is an $F N N \mathfrak { g } _ { d , \delta } \in \mathbf { N }$ of architecture $( d , n _ { d , \delta } , 1 )$ with $n _ { d , \delta } \leq C \delta ^ { - 1 }$ satisfying

$$
\operatorname* { s u p } _ { x \in \mathbb { R } ^ { d } } | g _ { d } ( x ) - ( \mathcal { R } _ { a } \mathfrak { g } _ { d , \delta } ) ( x ) | \leq \delta , \qquad \mathcal { P } ( \mathfrak { g } _ { d , \delta } ) \leq C d \delta ^ { - 1 } ,\tag{29}
$$

and

$$
\operatorname* { s u p } _ { x \in \mathbb { R } ^ { d } } | ( \mathcal { R } _ { a } \mathfrak { g } _ { d , \delta } ) ( x ) | \leq J ( M _ { \phi } + T _ { \phi } ) + 1 .\tag{30}
$$

Proof of Lemma $4 . 6 .$ Fix $d \in \mathbb { N }$ and $\delta \in ( 0 , 1 ]$ , and set $\delta _ { 0 } : = \delta / J$ . By the final assertion of Lemma 4.3, for every $j \in \{ 1 , \dots , J \}$ there exists an FNN $\phi _ { j , \delta _ { 0 } } \in \mathbf { N }$ of architecture $( 1 , m _ { j } , 1 )$ with

$$
\operatorname* { s u p } _ { v \in \mathbb { R } } | \phi _ { j } ( v ) - ( \mathcal { R } _ { a } \phi _ { j , \delta _ { 0 } } ) ( v ) | \leq \delta _ { 0 } , \qquad m _ { j } \leq \operatorname* { m a x } \Big \{ 1 , \Big \lceil \frac { 3 J T _ { \phi } } { \delta } \Big \rceil - 1 \Big \} .
$$

$\mathrm { B y }$ adding zero rows to ${ W } _ { j } ^ { ( 1 ) }$ , zero entries to ${ B } _ { j } ^ { ( 1 ) }$ , and matching zero columns to ${ W } _ { j } ^ { ( 2 ) }$ , which changes neither the realization nor the order of the parameter count, we may assume that all these networks have the common architecture $( 1 , n , 1 )$ with $n : = \operatorname* { m a x } \{ 1 , \lceil 3 J T _ { \phi } / \delta \rceil \}$ . Write $\phi _ { j , \delta _ { 0 } } ~ = ~ \big ( ( W _ { j } ^ { ( 1 ) } , B _ { j } ^ { ( 1 ) } ) , ( W _ { j } ^ { ( 2 ) } , B _ { j } ^ { ( 2 ) } ) \big )$ with $W _ { j } ^ { ( 1 ) } \in \mathbb { R } ^ { n \times 1 }$ , and define an FNN of architecture $( d , J n , 1 )$ by stacking the lifted first layers and concatenating the output layers:

$$
\begin{array} { r } { \mathfrak { g } _ { d , \delta } : = \big ( ( W ^ { ( 1 ) } , B ^ { ( 1 ) } ) , ( W ^ { ( 2 ) } , B ^ { ( 2 ) } ) \big ) , \qquad W ^ { ( 1 ) } : = \left( \begin{array} { c } { W _ { 1 } ^ { ( 1 ) } ( \alpha _ { 1 } ^ { d } ) ^ { \mathsf { T } } } \\ { \vdots } \\ { W _ { J } ^ { ( 1 ) } ( \alpha _ { J } ^ { d } ) ^ { \mathsf { T } } } \end{array} \right) , \qquad B ^ { ( 1 ) } : = \left( \begin{array} { c } { B _ { 1 } ^ { ( 1 ) } } \\ { \vdots } \\ { B _ { J } ^ { ( 1 ) } } \end{array} \right) , } \end{array}
$$

$$
W ^ { ( 2 ) } : = \bigl ( W _ { 1 } ^ { ( 2 ) } \ \cdot \cdot \cdot \ W _ { J } ^ { ( 2 ) } \bigr ) , \qquad B ^ { ( 2 ) } : = \sum _ { j = 1 } ^ { J } B _ { j } ^ { ( 2 ) } .
$$

Since the jth block of hidden units receives the input $W _ { j } ^ { ( 1 ) } \langle \alpha _ { j } ^ { d } , x \rangle + B _ { j } ^ { ( 1 ) }$ , Definition 2.1 gives, for every $\boldsymbol { x } \in \mathbb { R } ^ { d }$ 2

$$
( \mathcal { R } _ { a } \mathfrak { g } _ { d , \delta } ) \ d ( x ) = \sum _ { j = 1 } ^ { J } ( \mathcal { R } _ { a } \phi _ { j , \delta _ { 0 } } ) \ d ( \langle \alpha _ { j } ^ { d } , x \rangle \ d ) ,\tag{31}
$$

and therefore

$$
\operatorname* { s u p } _ { x \in \mathbb { R } ^ { d } } \vert g _ { d } ( x ) - ( \mathcal { R } _ { a } \mathfrak { g } _ { d , \delta } ) ( x ) \vert \leq \sum _ { j = 1 } ^ { J } \operatorname* { s u p } _ { v \in \mathbb { R } } \vert \phi _ { j } ( v ) - ( \mathcal { R } _ { a } \phi _ { j , \delta _ { 0 } } ) ( v ) \vert \leq J \delta _ { 0 } = \delta .\tag{32}
$$

Moreover, $n _ { d , \delta } = J n$ , and since $\delta \leq 1$ gives $n \leq ( 3 J T _ { \phi } + 1 ) \delta ^ { - 1 }$ 2

$$
\mathcal { P } ( \mathfrak { g } _ { d , \delta } ) = J n ( d + 1 ) + ( J n + 1 ) = J n ( d + 2 ) + 1 \le 4 J n d \le 4 J \bigl ( 3 J T _ { \phi } + 1 \bigr ) d \delta ^ { - 1 } ,\tag{33}
$$

which proves (29) with $C : = 4 J ( 3 J T _ { \phi } { + } 1 )$ . Finally, $\begin{array} { r } { \operatorname* { s u p } _ { v \in \mathbb { R } } | \phi _ { j } ( v ) | \leq | \phi _ { j } ( 0 ) | + \mathrm { T V } ( \phi _ { j } ) \leq M _ { \phi } + T _ { \phi } , } \end{array}$ so that (28) and (32) give (30). □

Together with Corollary 4.5, Lemma 4.6 verifies Setting 4.1 with $p = \kappa = 1$ for globally Lipschitz truncations and the ridge-sum initial conditions above: the f-parts of $( 1 6 )  ( 1 8 )$ are (26), and the g-parts follow from (29)–(30) since $1 + \| x \| ^ { \kappa } \geq 1$ and $d \geq 1$ , after enlarging $C _ { g }$ to absorb $C _ { f }$ and $C .$

Remark 4.7 (Breadth and limitations of the ridge-sum class). Lemma 4.6 covers shallownetwork initial conditions $\begin{array} { r } { g _ { d } ( \boldsymbol { x } ) = \sum _ { i = 1 } ^ { J } c _ { j } b ( \langle \alpha _ { j } ^ { d } , \boldsymbol { x } \rangle + \beta _ { j } ) } \end{array}$ with a Lipschitz profile b of finite total variation and J independent of $\dot { d } ,$ including single-index models $( J = 1 )$ and the Allen– Cahn-type datum of Remark 6.1, with no condition on the direction vectors. It does not cover a general radial function $g _ { d } ( x ) \ = \ \psi ( \| x \| )$ or a generic Lipschitz function on $\mathbb { R } ^ { d } \colon$ ; the lower bounds of [27, 28] discussed in Section 1.2 show that such restrictions cannot be removed in general, and verifying Setting 4.1 for further structured classes (tensor products, hierarchical compositions, Barron-type representations) is left to future research. An unbounded basket payof $( K - \langle w ^ { d } , x \rangle ) _ { + }$ is not of finite total variation, but a bounded Lipschitz truncation of it is covered, with no condition on $w ^ { d }$ , provided the truncated profiles have dimension-uniform Lipschitz, value, and total-variation bounds; comparing with the PDE for the original payof then requires a separate Gaussian tail estimate, since agreement of the initial conditions on the test cube alone is not suficient for the nonlocal heat flow.

## 4.1 ResNet representation of an MLP sample

The representation uses the last coordinate of the state as a scalar accumulator. Each accumulator update adds one summand of the MLP formula to this coordinate; depending on the summand, an update consists of a single residual block or a short concatenation of such blocks, and each shortcut transmits the spatial variable. The resulting blocks are combined by concatenation. Consequently, the construction requires no common-width padding.

Lemma 4.8 (Atomic accumulator blocks). Let $a \in C ( \mathbb { R } , \mathbb { R } )$

(a) Let $r , D \in \mathbb { N }$ with $r ~ \leq ~ D$ , let $\varphi \in \textbf { N }$ have input dimension r and scalar output, let $P \in \mathbb { R } ^ { r \times D } , b \in \mathbb { R } ^ { r } , e \in \mathbb { R } ^ { D }$ , and $h \in \mathbb { R }$ . There exists a one-block ResNet $\mathcal { A } _ { \varphi , P , b , e , h } \in \mathcal { N } _ { 1 }$ of input/output dimension D such that

$$
\Re _ { a } ( \mathcal { A } _ { \varphi , P , b , e , h } ) ( z ) = z + h e ( \mathcal { R } _ { a } \varphi ) ( P z + b ) , \qquad z \in \mathbb { R } ^ { D } ,\tag{34}
$$

and

$$
\mathfrak { P } ( \mathcal { A } _ { \varphi , P , b , e , h } ) \le C \bigl ( D \mathcal { P } ( \varphi ) + D ^ { 2 } \bigr )\tag{35}
$$

for a universal $C \in ( 0 , \infty )$ . If, in addition, $\mathcal { D } ( \varphi ) = ( D - 1 , m , 1 )$ and $P = ( \mathrm { I d } _ { D - 1 } \ | \ 0 )$ then the sharper estimate

$$
\mathfrak { P } ( \mathcal { A } _ { \varphi , P , b , e , h } ) \le C \bigl ( \mathcal { P } ( \varphi ) + D ^ { 2 } \bigr )\tag{36}
$$

holds.

(b) For every $d \in \mathbb { N }$ and $b \in \mathbb { R } ^ { d }$ there exist one-block ResNets $\mathcal { T } _ { b }$ and $\mathcal { O } _ { b }$ whose input-output dimensions are, respectively, $( d + 1 , d + 2 )$ and $( d + 2 , d + 1 )$ , such that

$$
\Re _ { a } ( \mathcal { T } _ { b } ) ( x , s ) = ( x + b , 0 , s ) , \qquad \Re _ { a } ( \mathcal { O } _ { b } ) ( y , w , s ) = ( y - b , s ) ,\tag{37}
$$

$$
a n d \mathfrak { P } ( \mathcal { T } _ { b } ) + \mathfrak { P } ( \mathcal { O } _ { b } ) \leq C d ^ { 2 } .
$$

Proof of Lemma $4 . 8 .$ For part (a), write $\mathcal { D } ( \varphi ) = ( r , m _ { 1 } , \dots , m _ { L - 1 } , 1 )$ . If $L = 1$ , then $\mathcal { R } _ { a \varphi } ( v ) =$ $W v + B$ is afine. The map

$$
z \longmapsto h e \bigl ( W ( P z + b ) + B \bigr )
$$

is the realization of an afine FNN $\widehat { \varphi } \colon \mathbb R ^ { D } \to \mathbb R ^ { D }$ with $D ( D + 1 )$ parameters. If $L \geq 2$ , replace the first afine pair $( W _ { 1 } , B _ { 1 } )$ by $( W _ { 1 } P , W _ { 1 } b + B _ { 1 } )$ , replace the last afine pair $( W _ { L } , B _ { L } )$ by $( h e W _ { L } , h e B _ { L } )$ , and leave all intermediate layers unchanged. The resulting FNN $\boldsymbol { \hat { \varphi } } \colon \mathbb { R } ^ { D } \to \mathbb { R } ^ { D }$ has

$$
( \mathcal { R } _ { a } \widehat { \varphi } ) ( z ) = h e ( \mathcal { R } _ { a } \varphi ) ( P z + b ) .
$$

Because $1 \leq r \leq D _ { }$ , the new first layer has at most D times as many parameters as the old first layer, the new last layer has exactly D times as many parameters as the old last layer, and the intermediate layers are unchanged. Hence ${ \mathcal { P } } ( { \widehat { \varphi } } ) \leq D { \mathcal { P } } ( \varphi )$ for $L \geq 2$ , while the afine case is bounded by $D ( D \pm 1 )$ . Let the shortcut of the single residual block be $\mathrm { I d } _ { D }$ . This proves (34) and, after adding the $D ^ { 2 }$ shortcut parameters, (35). If $\mathcal { D } ( \varphi ) = ( D - 1 , m , 1 )$ and $P = ( \mathrm { I d } _ { D - 1 } \mid 0 )$ then $\widehat { \varphi }$ has architecture $( D , m , D )$ and

$$
\mathcal { P } ( \widehat { \varphi } ) = m ( D + 1 ) + D ( m + 1 ) \leq C \big ( \mathcal { P } ( \varphi ) + D ^ { 2 } \big ) .
$$

Adding the shortcut parameters gives (36).

For part (b), use shortcuts

$$
( x , s ) \longmapsto ( x , 0 , s ) , \qquad ( y , w , s ) \longmapsto ( y , s ) ,
$$

and constant residual FNNs with values $( b , 0 , 0 )$ and $( - b , 0 )$ , respectively. The realizations are (37), and the two dense afine blocks and shortcuts contain at most $C d ^ { 2 }$ parameters. □

Proposition 4.9 (MLP-sample accumulator representation). Assume Setting 4.1 and use the probability space and random variables of Setting 3.1. Fix $d \in \mathbb { N } , \delta \in ( 0 , 1 ]$ , M, $N \in \mathbb { N } .$ , and a realization ω $\mathbf { \xi } ) \in \Omega$ . For $\vartheta \in \mathfrak { T }$ and $n \in  { \mathbb { N } } _ { 0 } ,$ , let $U _ { n , M } ^ { \vartheta , \delta }$ denote the MLP estimator (12) obtained by taking $g _ { 2 } = \mathcal { R } _ { a } \mathfrak { g } _ { d , \delta }$ and $f _ { 2 } = \mathcal { R } _ { a } \dag _ { \delta } ;$ thus δ here is the surrogate accuracy and is unrelated to the perturbation parameter $\Delta$ of Setting 3.1. Then there exists a ResNet $\Psi _ { N , M , \delta } ^ { \omega } \in \mathcal { N }$ such that

$$
( \Re _ { a } \Psi _ { N , M , \delta } ^ { \omega } ) ( x ) = U _ { N , M } ^ { 0 , \delta } ( 0 , x ) ( \omega ) , \qquad x \in \mathbb { R } ^ { d } .\tag{38}
$$

More precisely, for every occurrence of a level-n subestimator in the recursion, specified by its multi-index $\vartheta \in \mathfrak { T }$ , initial time $\tau \in [ 0 , T ]$ , and the fixed realization ω, there exists an accumulator $R e s N e t \mathcal { A } ^ { ( n , \emptyset , \tau , \omega ) } \in N$ with $\begin{array} { r } { \boldsymbol { \mathcal { T } } ( \boldsymbol { A } ^ { ( n , \bar { \vartheta } , \tau , \omega ) } ) = \boldsymbol { \mathcal { O } } ( \boldsymbol { A } ^ { ( n , \vartheta , \tau , \omega ) } ) = \boldsymbol { d } + 1 } \end{array}$ and satisfying

$$
\mathfrak { R } _ { a } \Big ( \mathcal { A } ^ { ( n , \vartheta , \tau , \omega ) } \Big ) \left( x , s \right) = \big ( x , s + U _ { n , M } ^ { \vartheta , \delta } ( \tau , x ) ( \omega ) \big ) .\tag{39}
$$

Proof of Proposition $4 . 9 .$ Fix ω and suppress it from the notation. Every randomized time and Brownian increment in (12) is then fixed. We prove (39) by induction on n, uniformly over all occurrences $( \vartheta , \tau )$ ; when no confusion is possible, we suppress these indices from the accumulator notation

For $n = 0$ , the estimator is zero, and the one-block ResNet on $\mathbb { R } ^ { d + 1 }$ with shortcut $\mathrm { I d } _ { d + 1 }$ and zero residual FNN satisfies (39).

Let $n \geq 1$ and assume the assertion for all lower levels. Consider (12) at a fixed time τ . Its first sum contains $M ^ { n }$ summands of the form

$$
M ^ { - n } ( \mathcal { R } _ { a } { \mathfrak { g } } _ { d , \delta } ) ( x + \beta ) .
$$

For each such summand, Lemma $4 . 8 ( \mathrm { a } )$ , applied with $D = d + 1 , r = d , P = ( \mathrm { I d } _ { d } | 0 ) , b = \beta$ $e = e _ { d + 1 }$ , and $h = M ^ { - n }$ , provides a residual block which leaves x unchanged and adds the displayed quantity to the accumulator s.

Next, consider a summand of the form

$$
h \left( \mathcal { R } _ { a } \mathfrak { f } _ { \delta } \right) \left( U _ { l , M } ^ { \vartheta ^ { \prime } , \delta } ( \tau ^ { \prime } , x + b ) \right) ,\tag{40}
$$

where ${ \mathit { l } } \ < \ n , \ \vartheta ^ { \prime }$ is the multi-index of this inner occurrence, and h is the fixed coeficient, positive or negative, appearing in (12). The induction hypothesis establishes the existence of an accumulator ResNet on $\mathbb { R } ^ { d + 1 }$ carrying the multi-index and time of this particular inner estimator. Denote it temporarily by $\mathbf { \mathcal { A } } ^ { ( l ) }$ . The ResNet $\mathcal { T } _ { b }$ maps $( x , s )$ to $( y , w , s ) = ( x + b , 0 , s )$ The one-coordinate lift of $\mathbf { \mathcal { A } } ^ { ( l ) }$ furnished by Lemma 2.5 maps this state to $( y , U _ { l , M } ^ { \vartheta ^ { \prime } , \delta } ( \tau ^ { \prime } , y ) , s )$ Thereafter, Lemma $4 . 8 ( \mathrm { a } )$ , applied with $D = d \div 2 , r = 1 , b = 0$ , with $P$ selecting the coordinate $w ,$ and with $e = e _ { d + 2 }$ , adds $h ( \mathcal { R } _ { a } \mathfrak { f } _ { \delta } ) ( w )$ to the final coordinate. The ResNet $\mathcal { O } _ { b }$ then maps the state to $( x , s + h ( \mathcal { R } _ { a } \mathfrak { f } _ { \delta } ) ( U _ { l , M } ^ { \vartheta ^ { \prime } , \delta } ( \tau ^ { \prime } , x + b ) ) )$ . Consequently, Lemma 2.4 shows that the concatenation of these four ResNets realizes the update associated with (40). The correction terms involving $U _ { l - 1 , M } ^ { \vartheta ^ { \prime \prime } , \delta }$ , which occur only for $l \in \{ 1 , \ldots , n - 1 \}$ , are handled identically, with the multi-index and time belonging to that occurrence.

All the resulting updates have input and output dimension $d + 1$ . Concatenating them therefore adds every summand of (12) to the same accumulator and leaves the spatial variable unchanged. This proves (39) at level n.

Finally, concatenating the linear one-block ResNet $x \mapsto ( x , 0 )$ , the ResNet $\mathcal { A } ^ { ( N , 0 , 0 , \omega ) }$ , and the linear one-block readout $( x , s ) \mapsto s$ yields the ResNet $\Psi _ { N , M , \delta } ^ { \omega }$ which satisfies (38). □

## 4.2 The parameter recursion

For arbitrary surrogate depths define

$$
\widehat { B } _ { d , \delta } : = d \big ( \mathcal { P } ( \mathfrak { g } _ { d , \delta } ) + \mathcal { P } ( \mathfrak { f } _ { \delta } ) \big ) + c _ { 0 } d ^ { 2 } ,\tag{41}
$$

where $c _ { 0 } \in ( 0 , \infty )$ is universal. If $\mathfrak { g } _ { d , \delta }$ has one hidden layer, as in Lemma 4.6, set instead

$$
B _ { d , \delta } : = \mathcal { P } ( \mathfrak { g } _ { d , \delta } ) + d \mathcal { P } ( \mathfrak { f } _ { \delta } ) + c _ { 0 } d ^ { 2 } .\tag{42}
$$

The sharper coeficient of the g-surrogate follows from (36).

Lemma 4.10 (Parameter recursion). Let $d \in \mathbb { N }$ and $\delta \in ( 0 , 1 ]$ . There exist universal constants $c _ { 1 } \in ( 0 , \infty )$ and $K \in [ 1 , \infty )$ and, for every $M \in \mathbb { N } \cap [ 2 , \infty )$ and $n \in  { \mathbb { N } } _ { 0 }$ , a deterministic bound $p _ { n , M } \in \left[ 0 , \infty \right)$ on the number of parameters of every level-n accumulator constructed with branching parameter M in the proof of Proposition $4 . 9 ,$ uniformly over all multi-indices, initial times, Brownian increments, and random times (these values change only the weights and biases, not the architecture), such that, for every $n \in \mathbb { N }$

$$
\begin{array} { l } { \displaystyle { p _ { n , M } \leq c _ { 1 } \bigg [ M ^ { n } \mathcal { B } _ { d , \delta } + \sum _ { l = 0 } ^ { n - 1 } M ^ { n - l } \big ( p _ { l , M } + \mathcal { B } _ { d , \delta } \big ) \bigg ] , } } \\ { \displaystyle { p _ { 0 , M } \leq c _ { 1 } \mathcal { B } _ { d , \delta } . } } \end{array}\tag{43}
$$

Here $\mathcal { B } _ { d , { \delta } } = \widehat { B } _ { d , { \delta } }$ in general. For the one-hidden-layer g-surrogates of Lemma $4 . 6 ,$ one may take $\mathcal { B } _ { d , { \delta } } = B _ { d , { \delta } }$ . Consequently,

$$
\begin{array} { r } { p _ { N , M } \leq c _ { 1 } \mathcal { B } _ { d , \delta } ( K M ) ^ { N } , \qquad M \in \mathbb { N } \cap [ 2 , \infty ) , \ N \in \mathbb { N } , } \end{array}\tag{44}
$$

and the scalar-output ResNet of Proposition 4.9 satisfies the same bound after enlarging $c _ { 1 }$

Proof of Lemma 4.10. Fix $M \in \mathbb { N } \cap [ 2 , \infty )$ By Lemma $4 . 8 ( \mathrm { a } )$ , an update associated with a summand containing g has at most $C B _ { d , \delta }$ parameters. For an update associated with a summand containing $f \circ U _ { l }$ , the initialization and restoration blocks together have at most $C d ^ { 2 }$ parameters, the lifted level-l accumulator has at most $4 p _ { l , M }$ parameters by Lemma $2 . 5 ,$ and the reaction block has at most $C ( d \mathcal { P } ( \mathfrak { f } _ { \delta } ) + d ^ { 2 } )$ parameters. Hence, every such update has at most $C ( p _ { l , M } + { \cal B } _ { d , \delta } )$ parameters.

There are $M ^ { n }$ summands involving g and $M ^ { n - l }$ summands of the first nonlinear type at level l. Reindexing the correction terms with $j = l - 1$ and using $M ^ { n - j - 1 } \leq M ^ { n - j }$ shows that they contribute at most $\begin{array} { r } { \sum _ { i = 0 } ^ { n - 1 } M ^ { n - j } ( p _ { j , M } + \mathcal { B } _ { d , \delta } ) } \end{array}$ . Because concatenation adds parameter counts exactly, Lemma 2.4(ii) gives (43). The identity accumulator in the base case has at most $C d ^ { 2 }$ parameters.

Let $q _ { n , M } : = p _ { n , M } / ( M ^ { n } B _ { d , \delta } )$ . Dividing (43) by $M ^ { n } B _ { d , \delta }$ , noting that $M ^ { n - l } ( p _ { l , M } + B _ { d , \delta } )$ equals $\left( q _ { l , M } + M ^ { - l } \right) M ^ { n } B _ { d , \delta }$ , and using $\begin{array} { r } { \sum _ { l \geq 0 } M ^ { - l } \leq 2 } \end{array}$ yields

$$
q _ { n , M } \leq C \biggl ( 1 + \sum _ { l = 0 } ^ { n - 1 } q _ { l , M } \biggr ) .
$$

If $b _ { n , M } : = 1 + \textstyle \sum _ { l = 0 } ^ { n } q _ { l , M }$ , then the base case of (43) gives $b _ { 0 , M } = 1 + q _ { 0 , M } \leq 1 + c _ { 1 }$ , and $b _ { n , M } \leq ( 1 + C ) b _ { n - 1 , M }$ for $n \geq 1$ . Hence $q _ { n , M } \leq C b _ { n - 1 , M } \leq C ( 1 + c _ { 1 } ) ( 1 + C ) ^ { n - 1 }$ , and (44) follows with $K : = 1 + C$ after changing the universal constant. The two root blocks $x \mapsto ( x , 0 )$ and $( x , s ) \mapsto s$ together have at most $C d ^ { 2 }$ parameters, and this quantity is absorbed by $\quad B _ { d , \delta } . \quad \sqsubseteq$

## 5 Main theorem

Theorem 5.1 (Residual neural networks overcome the curse of dimensionality for semilinear heat equations). Let $L \in ( 0 , \infty )$ , let $a \in C ( \mathbb { R } , \mathbb { R } )$ , and let $f \in C ( \mathbb { R } , \mathbb { R } )$ satisfy $\operatorname { L i p } ( f ) \leq L$ . For every $d \in \mathbb { N }$ let $g _ { d } \in C ( \mathbb { R } ^ { d } , \mathbb { R } )$ and let $u _ { d } \in C ( [ 0 , T ] \times \mathbb { R } ^ { d } , \mathbb { R } ) \cap C ^ { 1 , 2 } ( ( 0 , T ] \times \mathbb { R } ^ { d } , \mathbb { R } )$ be a solution of at most polynomial growing solution $o f \ ( 1 )$ in the sense of (10). Assume that there exist $B \in ( 0 , \infty )$ and $p \in [ 1 , \infty )$ such that

$$
| f ( 0 ) | \leq B , \qquad | g _ { d } ( x ) | \leq B ( 1 + \| x \| ) ^ { p } \qquad ( d \in \mathbb { N } , \ x \in \mathbb { R } ^ { d } ) .
$$

Then the following hold.

(i) Assume that Setting 4.1 holds with the constants $L , B , p$ above and with cost exponent κ. Then there exist $\eta \in ( 0 , \infty )$ and a family of ResNets $( \Psi _ { d , \varepsilon } ) _ { d \in \mathbb { N } , \varepsilon \in ( 0 , 1 ] } \subseteq N$ such that, for all $d \in \mathbb { N }$ and $\varepsilon \in ( 0 , 1 ]$

$$
\begin{array} { l } { \displaystyle \mathfrak { R } _ { a } ( \Psi _ { d , \varepsilon } ) \in C ( \mathbb { R } ^ { d } , \mathbb { R } ) , \qquad \mathfrak { R } ( \Psi _ { d , \varepsilon } ) \leq \eta d ^ { \eta } \varepsilon ^ { - \eta } , } \\ { \displaystyle \left[ \int _ { [ 0 , 1 ] ^ { d } } \vert u _ { d } ( T , x ) - ( \mathfrak { R } _ { a } \Psi _ { d , \varepsilon } ) ( x ) \vert ^ { 2 } \ d x \right] ^ { 1 / 2 } \leq \varepsilon . } \end{array}\tag{45}
$$

(ii) Assume instead that a is an admissible sigmoidal activation in the sense of Lemma $4 . 3 ,$ that f is a globally Lipschitz truncation, and that $( g _ { d } ) _ { d \in \mathbb { N } }$ has the ridge-sum form $\left( 2 7 \right) - \left( 2 8 \right)$ of Lemma 4.6. Then the conclusions of part (i) hold and, for every $\xi > 0$ , the networks may be chosen so that

$$
\mathfrak { P } ( \Psi _ { d , { \varepsilon } } ) \le C _ { \xi } d ^ { 4 + \xi } { \varepsilon } ^ { - ( 3 + \xi ) } \qquad ( d \in \mathbb { N } , \ { \varepsilon } \in ( 0 , 1 ] ) ,\tag{46}
$$

where $C _ { \xi }$ depends only on $\xi$ and on the dimension-independent constants in the hypotheses, and not on $d , \varepsilon ,$ , or the direction vectors $\alpha _ { j } ^ { d }$

Proof of Theorem 5.1. Throughout this proof let $d \in \mathbb { N } , \varepsilon \in ( 0 , 1 ]$ , and $\xi \in ( 0 , \infty )$ . For part (i) it is suficient to choose, for example, $\xi = 1$ , whereas for part (ii) the real number ξ is arbitrary. We denote by $C , C ^ { \prime } , \ldots$ . finite constants which may depend on $T , L , B , p , \kappa , C _ { g } , L _ { f } .$ on the constants of Lemma 4.10, and on ξ, but not on $d , \varepsilon , \rho , \delta , N _ { \mathrm { \scriptsize ~ ; ~ } }$ , or M. Enlarge L to max $\{ L , L _ { f } \}$ and B if necessary, as explained in Remark 4.2, so that the true and surrogate data satisfy the same bounds in Setting 3.1, and let $q : = \operatorname* { m a x } \{ 2 , \kappa / p \}$ , so that $q \geq 2$ and $p q \geq \kappa ;$ Setting 3.1 is instantiated with this moment exponent. Let

$$
\begin{array} { r } { \Pi _ { d } ( x ) : = \Big ( 1 + \| x \| + \big ( \mathbb { E } [ \| \mathbf { W } _ { T } \| ^ { p q } ] \big ) ^ { 1 / ( p q ) } \Big ) ^ { p q } . } \end{array}
$$

Let $C _ { \Pi } \in [ 1 , \infty )$ be an exponent for which there exists $K _ { \Pi } \in [ 1 , \infty )$ , independently of d and ε, such that

$$
\bigl ( e ^ { L T } ( T + 1 ) \bigr ) ^ { q + 1 } ( B ^ { q } + 1 ) \Bigl ( \int _ { [ 0 , 1 ] ^ { d } } \Pi _ { d } ( x ) ^ { 2 } d x \Bigr ) ^ { 1 / 2 } \leq K _ { \Pi } d ^ { C _ { \Pi } } .\tag{47}
$$

The Gaussian moment estimate $( \mathbb { E } [ \| \mathbf { W } _ { T } \| ^ { p q } ] ) ^ { 1 / ( p q ) } \leq C _ { p , q , T } \sqrt { d }$ , together with $\| x \| \leq { \sqrt { d } }$ for $x \in [ 0 , 1 ] ^ { d }$ , shows, in particular, that $C _ { \Pi } = p q / 2$ is admissible.

First, we choose the surrogate and MLP approximation accuracies. Enlarging $C _ { g }$ if necessary, assume $C _ { g } \geq 1$ . By (16), $p q \geq \kappa .$ , and $1 + \| x \| ^ { \kappa } \leq 2 ( 1 + \| x \| ) ^ { p q }$ , the surrogate pair lies within

$$
\Delta _ { \mathrm { p e r t } } : = 2 C _ { g } d ^ { \kappa } \delta\tag{48}
$$

of $( g _ { d } , f )$ in the gauge (8). Let

$$
\rho : = \varepsilon ( 4 K _ { \Pi } d ^ { C _ { \Pi } } ) ^ { - 1 } , \qquad \delta : = \rho ( 2 C _ { g } d ^ { \kappa } ) ^ { - 1 } = \varepsilon ( 8 C _ { g } K _ { \Pi } d ^ { C _ { \Pi } + \kappa } ) ^ { - 1 } .\tag{49}
$$

Then $\Delta _ { \mathrm { p e r t } } = \rho$ and $\delta , \rho \in ( 0 , 1 ]$ . Let K be the universal constant from Lemma 4.10. Lemma 3.5, applied with target $\rho ,$ parameter $\xi ,$ and $A = K$ , ensures that there exists $N = N ( \rho ) \in \mathbb { N } \cap [ 2 , \infty )$ such that

$$
\frac { e ^ { N / 2 } ( 1 + 2 L T ) ^ { N } } { N ^ { N / 2 } } \leq \rho , \qquad ( K N ) ^ { N } \leq C _ { \xi } \rho ^ { - ( 2 + \xi ) } .\tag{50}
$$

We set $M : = N$ . In particular,

$$
( K M ) ^ { N } \leq C d ^ { C _ { \Pi } ( 2 + \xi ) } \varepsilon ^ { - ( 2 + \xi ) } .\tag{51}
$$

Moreover,

$$
\delta \geq C ^ { - 1 } \varepsilon d ^ { - \left( C _ { \Pi } + \kappa \right) } .\tag{52}
$$

Next, we establish the mean-square error estimate. Let $U _ { N , M } ^ { 0 , \delta }$ be the MLP estimator built from the surrogate data, as in Proposition 4.9; when Setting 3.1 is instantiated with these data, this is the estimator $U _ { N , M } ^ { 2 , 0 }$ . Corollary 3.4, applied with perturbation level $\Delta _ { \mathrm { p e r t } } = \rho _ { \mathrm { ; } }$ the identification $v _ { 1 } ( 0 , x ) = u _ { d } ( T , \stackrel {  } { x } )$ from (11) (the theorem hypothesizes $u _ { d }$ in the class (10)), and (50) ensure, for every $\boldsymbol { x } \in \mathbb { R } ^ { d }$ , that

$$
\begin{array} { r } { \big ( \mathbb { E } [ \left| U _ { N , M } ^ { 0 , \delta } ( 0 , x ) - u _ { d } ( T , x ) \right| ^ { 2 } ] \big ) ^ { 1 / 2 } \leq 2 \rho \big ( e ^ { L T } ( T + 1 ) \big ) ^ { q + 1 } ( B ^ { q } + 1 ) \Pi _ { d } ( x ) . } \end{array}
$$

After squaring and integrating over $[ 0 , 1 ] ^ { d }$ , (47) and the definition of $\rho$ show that

$$
\int _ { [ 0 , 1 ] ^ { d } } \mathbb { E } \big [ \Big | U _ { N , M } ^ { 0 , \delta } ( 0 , x ) - u _ { d } ( T , x ) \Big | ^ { 2 } ] d x \leq \frac { \varepsilon ^ { 2 } } { 4 } < \varepsilon ^ { 2 } .\tag{53}
$$

The map $( x , \omega ) \mapsto U _ { N , M } ^ { 0 , \delta } ( 0 , x ) ( \omega )$ is jointly measurable. Indeed, this follows by induction in the finite recursion (12) from the continuity of the surrogate realizations and the joint measurability of the Brownian motions and random times. Hence Tonelli’s theorem gives

$$
\mathbb E \left[ \int _ { [ 0 , 1 ] ^ { d } } \left| U _ { N , M } ^ { 0 , \delta } ( 0 , x ) - u _ { d } ( T , x ) \right| ^ { 2 } d x \right] = \int _ { [ 0 , 1 ] ^ { d } } \mathbb E [ \left| U _ { N , M } ^ { 0 , \delta } ( 0 , x ) - u _ { d } ( T , x ) \right| ^ { 2 } ] d x .
$$

Consequently, there exists $\omega ^ { * } \in \Omega$ such that

$$
\int _ { [ 0 , 1 ] ^ { d } } \left| U _ { N , M } ^ { 0 , \delta } ( 0 , x ) ( \omega ^ { * } ) - u _ { d } ( T , x ) \right| ^ { 2 } d x < \varepsilon ^ { 2 } .\tag{54}
$$

We now represent the selected realization by a ResNet. Proposition 4.9 establishes the existence of a ResNet $\Psi _ { d , \varepsilon } : = \Psi _ { N , M , \delta } ^ { \omega ^ { * } }$ whose realization equals the selected MLP sample. Equation (54) gives the error bound in (45).

It remains to estimate the number of parameters. For general surrogate families, Lemma 4.10 and (41) give

$$
\begin{array} { r } { \mathfrak { P } ( \Psi _ { d , { \varepsilon } } ) \leq C \widehat { B } _ { d , { \delta } } ( K M ) ^ { N } , \qquad \widehat { B } _ { d , { \delta } } \leq C d ^ { \kappa + 1 } { \delta } ^ { - \kappa } + C d ^ { 2 } . } \end{array}
$$

Combining this estimate with (52) and (51) yields

$$
\mathfrak { P } ( \Psi _ { d , \varepsilon } ) \le C _ { 0 } d ^ { \eta _ { d } } \varepsilon ^ { - \eta _ { \varepsilon } } ,
$$

where

$$
\eta _ { d } : = \operatorname* { m a x } \bigl \{ \kappa + 1 + \kappa C _ { \Pi } + \kappa ^ { 2 } + C _ { \Pi } ( 2 + \xi ) , 2 + C _ { \Pi } ( 2 + \xi ) \bigr \} , \qquad \eta _ { \varepsilon } : = \kappa + 2 + \xi ,
$$

and $C _ { 0 } \in ( 0 , \infty )$ is independent of d and $\varepsilon .$ For part (i), let $\xi = 1$ and $\eta : = \operatorname* { m a x } \{ C _ { 0 } , \eta _ { d } , \eta _ { \varepsilon } \}$ This proves the parameter estimate in (45).

Under the hypotheses of part (ii), Corollary 4.5 and Lemma 4.6 verify Setting 4.1 with $\kappa = 1$ and $p = 1$ , so part (i) applies. Every profile satisfies $| \phi _ { j } ( v ) | \leq | \phi _ { j } ( 0 ) | + \mathrm { T V } ( \phi _ { j } ) \leq M _ { \phi } + T _ { \phi }$ , so that $\begin{array} { r } { \operatorname* { s u p } _ { x \in \mathbb { R } ^ { d } } | g _ { d } ( x ) | \leq J ( M _ { \phi } + T _ { \phi } ) } \end{array}$ and, by (30), $\begin{array} { r } { \operatorname* { s u p } _ { x \in \mathbb { R } ^ { d } } | ( \mathcal { R } _ { a } \mathfrak { g } _ { d , \delta } ) ( x ) | \leq J ( M _ { \phi } + T _ { \phi } ) + 1 } \end{array}$ . We therefore run the preceding argument with the explicit common constants

$$
\begin{array} { r l } & { L _ { * } : = \operatorname* { m a x } \{ L , L _ { f } \} , \qquad B _ { * } : = \operatorname* { m a x } \{ B , B _ { f } , J ( M _ { \phi } + T _ { \phi } ) + 1 \} , } \\ & { \qquad \quad \kappa = 1 , \qquad p = 1 , \qquad q = \operatorname* { m a x } \{ 2 , \kappa / p \} = 2 , } \end{array}
$$

in place of $L$ and $B ;$ these depend only on the constants in the hypotheses and not on $d , \varepsilon ,$ or the direction vectors $\alpha _ { j } ^ { d } .$ . By the Gaussian moment estimate following (47), the exponent $C _ { \Pi } = p q / 2 = 1$ is then admissible, and we fix this choice, with $K _ { \Pi }$ depending only on $T , L _ { * } ,$ and $B _ { * }$ . Lemma 4.6 supplies the uniform error and linear cost bounds in (29). By the first estimate in (29) and the f-part of (16), both surrogate errors are bounded by <sup>¯</sup>δ uniformly, and the gauge weight $( 1 + \| x \| ) ^ { p q } + | v | ^ { q }$ in (8) is at least 1. Hence the surrogate pair has perturbation level

$$
\bar { \Delta } _ { \mathrm { p e r t } } : = \bar { \delta } .
$$

Choose

$$
\bar { \delta } : = \rho = \frac { \varepsilon } { 4 K _ { \Pi } d ^ { C _ { \Pi } } } .
$$

Then $\Delta _ { \mathrm { p e r t } } = \rho$ and $\bar { \delta } \in ( 0 , 1 ]$ , so the preceding MLP error estimate, selection argument, and representation argument apply without change. Furthermore, Corollary 4.5, (29), and (42) give

$$
B _ { d , \bar { \delta } } \leq C \bigl ( d \bar { \delta } ^ { - 1 } + d ^ { 2 } \bigr ) \leq C \bigl ( d ^ { 1 + C _ { \Pi } } \varepsilon ^ { - 1 } + d ^ { 2 } \bigr ) \leq C d ^ { 1 + C _ { \Pi } } \varepsilon ^ { - 1 } .
$$

In the last step we used $C _ { \Pi } \geq 1 , d \geq 1$ , and $\varepsilon \leq 1$ . Multiplying this estimate by (51) yields

$$
\mathfrak { P } ( \Psi _ { d , { \varepsilon } } ) \le C _ { \xi } d ^ { 1 + C _ { \Pi } ( 3 + \xi ) } \varepsilon ^ { - ( 3 + \xi ) } = C _ { \xi } d ^ { 4 + \xi } \varepsilon ^ { - ( 3 + \xi ) } ,
$$

since $C _ { \Pi } = 1$ . This proves (46).

Since $[ 0 , 1 ] ^ { d }$ has Lebesgue measure one, H¨older’s inequality shows that the networks of Theorem 5.1 also satisfy $[ \int _ { [ 0 , 1 ] ^ { d } } | u _ { d } ( T , x ) - ( \Re _ { a } \Psi _ { d , \varepsilon } ) ( x ) | ^ { r } d x ] ^ { 1 / r } \leq \varepsilon$ for every $r \in ( 0 , 2 ]$

## 6 Consequences and examples

Remark 6.1 (An Allen–Cahn-type example). Let a be an admissible sigmoidal activation in the sense of Lemma 4.3, let

$$
g _ { d } ( x ) = \operatorname { t a n h } ( \langle \alpha ^ { d } , x \rangle ) , \qquad \alpha ^ { d } \in \mathbb { R } ^ { d } \ \mathrm { a r b i t r a r y } ,
$$

and define the globally Lipschitz truncation

$$
\widetilde f ( r ) = \left\{ \begin{array} { l l } { r - r ^ { 3 } , } & { | r | \leq 1 , } \\ { 0 , } & { | r | > 1 . } \end{array} \right.
$$

Then Lemma 4.6 applies with $J = 1 , L _ { \phi } = 1 , M _ { \phi } = 0$ , and $T _ { \phi } = 2$ . Since $g _ { d }$ takes values in [−1, 1] and the constants ±1 are solutions of (1) because $\widetilde { f } ( \pm 1 ) = 0$ , the parabolic comparison principle implies that the corresponding solution remains in this interval. Consequently, $\widetilde { f }$ agrees with the classical Allen–Cahn reaction $r \mapsto r - r ^ { 3 }$ on the range of the solution, and the truncation level is independent of $d .$

Theorem 5.1(ii) therefore yields the bound (46) for every $\xi > 0$

Remark 6.2 (General measures). With $p , q , \Pi _ { d } ,$ and $C _ { \Pi }$ as in the proof of Theorem 5.1, the Lebesgue measure on $[ 0 , 1 ] ^ { d }$ in (45) may be replaced by Borel probability measures $\nu _ { d }$ with $\begin{array} { r } { \int _ { \mathbb { R } ^ { d } } ( 1 + \| x \| ) ^ { 2 p q } \nu _ { d } ( d x ) \leq C _ { 0 } d ^ { \gamma } } \end{array}$ for some $C _ { 0 } \in ( 0 , \infty )$ and $\gamma \in [ 0 , \infty )$ , the $L ^ { 2 } ( [ 0 , 1 ] ^ { d } )$ -error being replaced by the $L ^ { 2 } ( \nu _ { d } ) – \mathrm { e r r o r : }$ since the Gaussian moment term in $\Pi _ { d }$ is of order $d ^ { 1 / 2 }$ , one has $\begin{array} { r } { \int _ { \mathbb { R } ^ { d } } \Pi _ { d } ( x ) ^ { 2 } \nu _ { d } ( d x ) \leq C d ^ { \operatorname* { m a x } \{ \gamma , p q \} } } \end{array}$ , and the proof applies with $C _ { \Pi } = \textstyle { \frac { 1 } { 2 } } \operatorname* { m a x } \{ \gamma , p q \}$ . In particular, the exponents of part (ii) persist for $\gamma \le 2$

## 7 Conclusion

Theorem 5.1 establishes a deterministic, curse-of-dimensionality-free ResNet expressivity result for semilinear heat equations whose data admit the network surrogates of Setting 4.1; for admissible sigmoidal activations, globally Lipschitz truncations, and the ridge-sum data of Lemma 4.6, it yields the bound $C _ { \xi } d ^ { 4 + \xi } \varepsilon ^ { - ( 3 + \xi ) }$ for every $\xi > 0$ . The key ingredient is the accumulator construction of Proposition 4.9, which represents one realization of an MLP estimator by concatenating residual updates, so that parameter counts add and no identity representation of the activation is needed. Extensions to state-dependent coeficients, gradient-dependent nonlinearities, and space–time approximations, as well as sharper exponents and constructive training procedures, remain open.

## A Proofs of the quantitative MLP inputs

In this appendix we prove Lemmas 3.2, 3.3 and 3.5 and Corollary 3.4 and record the constants used in Theorem 5.1.

Proof of Lemma 3.2. Fix $i \in \{ 1 , 2 \}$ and $x \in \mathbb { R } ^ { d }$ , write $v = v _ { i } , g = g _ { i } , f = f _ { i }$ , and $\Vert Z \Vert _ { L ^ { q } } : =$ $( \mathbb { E } [ | Z | ^ { q } ] ) ^ { 1 / q }$ for a random variable Z, and let

$$
\Phi ( t ) : = \operatorname* { s u p } _ { s \in [ t , T ] } \| v ( s , x + \mathbf { W } _ { s } ) \| _ { L ^ { q } } , \qquad R : = \operatorname* { s u p } _ { s \in [ 0 , T ] } \big ( \mathbb { E } [ ( 1 + \| x + \mathbf { W } _ { s } \| ) ^ { p q } ] \big ) ^ { 1 / q } ,\tag{55}
$$

so that the assertion reads $\Phi ( 0 ) \leq e ^ { L T } ( T + 1 ) B R$ . The Gaussian moments of W and the polynomial growth of v give $R < \infty$ and $\Phi ( 0 ) < \infty$ , and $R \geq 1$ since $1 + \| x + \mathbf { W } _ { s } \| \geq 1$ ; moreover, Φ is nonincreasing, hence measurable, with $\begin{array} { r } { \int _ { 0 } ^ { T } \Phi ( r ) d r \leq T \Phi ( 0 ) < \infty } \end{array}$

Fix $t \in [ 0 , T ]$ . Evaluating (9) at the random point $\boldsymbol { x } + \mathbf { W } _ { t }$ gives, P-almost surely,

$$
v ( t , x + \mathbf { W } _ { t } ) = \mathbb { E } \Big [ g \big ( x + \mathbf { W } _ { t } + \widetilde { \mathbf { W } } _ { T - t } \big ) + \int _ { t } ^ { T } f \big ( v ( r , x + \mathbf { W } _ { t } + \widetilde { \mathbf { W } } _ { r - t } ) \big ) d r \Big | \mathbf { W } _ { t } \Big ] ,\tag{56}
$$

where $\widetilde { \mathbf { W } }$ is an independent copy of the Brownian motion and the conditional expectation is taken over $\widetilde { \mathbf { W } }$ only; the patched process $( \mathbf { W } _ { t } + \widetilde { \mathbf { W } } _ { r - t } ) _ { r \in [ t , T ] }$ has the same law as $( \mathbf { W } _ { r } ) _ { r \in [ t , T ] }$ Taking L<sup>q</sup>-norms in (56) and using conditional Jensen’s inequality $\| \mathbb { E } [ Z \mid \mathbf { W } _ { t } ] \| _ { L ^ { q } } \leq \| Z \| _ { L ^ { q } }$ , this law identity, the triangle inequality, and Minkowski’s integral inequality yield

$$
\| v ( t , x + \mathbf { W } _ { t } ) \| _ { L ^ { q } } \leq \left\| g ( x + \mathbf { W } _ { T } ) \right\| _ { L ^ { q } } + \int _ { t } ^ { T } \left\| f \big ( v ( r , x + \mathbf { W } _ { r } ) \big ) \right\| _ { L ^ { q } } d r .\tag{57}
$$

By the growth bound $| g ( y ) | \leq B ( 1 + \| y \| ) ^ { p }$ and the definition of $R ,$

$$
\begin{array} { r } { \| g ( x + \mathbf { W } _ { T } ) \| _ { L ^ { q } } \leq B \big ( \mathbb { E } \big [ ( 1 + \| x + \mathbf { W } _ { T } \| ) ^ { p q } \big ] \big ) ^ { 1 / q } \leq B R , } \end{array}\tag{58}
$$

and since $| f ( v ) | \leq | f ( 0 ) | + L | v | \leq B + L | v |$ , for every $r \in [ t , T ]$

$$
\begin{array} { r } { \left. f \bigl ( v ( r , x + \mathbf { W } _ { r } ) \bigr ) \right. _ { L ^ { q } } \leq B + L \Phi ( r ) . } \end{array}\tag{59}
$$

Inserting (58) and (59) into (57) and using $R \geq 1$ gives

$$
\| v ( t , x + \mathbf { W } _ { t } ) \| _ { L ^ { q } } \leq B R + B T + L \int _ { t } ^ { T } \Phi ( r ) d r \leq ( T + 1 ) B R + L \int _ { t } ^ { T } \Phi ( r ) d r .\tag{60}
$$

The right-hand side is nondecreasing as t decreases, so taking the supremum over $t \in [ \tau , T ]$ gives, for every $\tau \in [ 0 , T ]$ 2

$$
\Phi ( \tau ) \leq ( T + 1 ) B R + L \int _ { \tau } ^ { T } \Phi ( r ) d r .\tag{61}
$$

Applying the backward Gr¨onwall inequality to (61), with $\alpha = ( T + 1 ) B R$ and $\tau = 0$ , yields $\Phi ( 0 ) \leq e ^ { L T } ( T + 1 ) B R$ , which is the assertion. □

Proof of Lemma 3.3. Fix $\boldsymbol { x } \in \mathbb { R } ^ { d }$ and let

$$
\Pi ( x ) : = \bigl ( 1 + \| x \| + ( \mathbb { E } [ \| \mathbf { W } _ { T } \| ^ { p q } ] ) ^ { 1 / ( p q ) } \bigr ) ^ { p q } , \qquad e ( t ) : = \operatorname* { s u p } _ { s \in [ t , T ] } \mathbb { E } \big [ \left| v _ { 1 } ( s , x + \mathbf { W } _ { s } ) - v _ { 2 } ( s , x + \mathbf { W } _ { s } ) \right| \big ] .\tag{62}
$$

By Lemma 3.2 and Jensen’s inequality, $e ( 0 ) < \infty ,$ , and e is nonincreasing, hence measurable with $\begin{array} { r } { \int _ { 0 } ^ { T } e ( s ) d s \leq T e ( 0 ) < \infty } \end{array}$ . It sufices to prove the stronger bound $e ( 0 ) \leq \Delta ( e ^ { L T } ( T + 1 ) ) ^ { q + 1 } ( B ^ { q } +$ $1 ) \Pi ( x )$

Fix $t \in [ 0 , T ]$ and $\boldsymbol { y } \in \mathbb { R } ^ { d }$ , and let $\widetilde { \mathbf { W } }$ be an independent copy of W. Subtracting the fixedpoint equation (9) for $v _ { 2 }$ from that for $v _ { 1 }$ at $( t , y )$ and adding and subtracting $f _ { 2 } ( v _ { 1 } )$ inside the integrand gives

$$
v _ { 1 } ( t , y ) - v _ { 2 } ( t , y ) = \mathbb { E } \big [ ( g _ { 1 } - g _ { 2 } ) ( y + \widetilde { \mathbf { W } } _ { T - t } ) \big ] + \mathbb { E } \int _ { t } ^ { T } \underbrace { \big [ f _ { 1 } ( v _ { 1 } ) - f _ { 2 } ( v _ { 1 } ) \big ] } _ { \mathrm { d a t a ~ p a r t } } + \underbrace { \big [ f _ { 2 } ( v _ { 1 } ) - f _ { 2 } ( v _ { 2 } ) \big ] } _ { \mathrm { L i p s c h i t z ~ p a r t } } d s ,\tag{63}
$$

where $f _ { i } ( v _ { j } )$ abbreviates $f _ { i } ( v _ { j } ( s , y + \widetilde { \mathbf { W } } _ { s - t } ) )$ . Let

$$
\Lambda : = \operatorname* { s u p } _ { s \in [ 0 , T ] } \mathbb { E } \big [ ( 1 + \| x + \mathbf { W } _ { s } \| ) ^ { p q } + | v _ { 1 } ( s , x + \mathbf { W } _ { s } ) | ^ { q } \big ] ,\tag{64}
$$

which is finite by the Gaussian moments of W and Lemma 3.2; see (67) below. The perturbation bound (8), applied at $z = y + \widetilde { \mathbf { W } } _ { T - t } , v = 0$ for the boundary term and at $z = y + \widetilde { \mathbf { W } } _ { s - t }$ $v = v _ { 1 } ( s , z )$ for the data part, gives

$$
\begin{array} { r l r } & { } & { \Big | ( g _ { 1 } - g _ { 2 } ) ( y + \widetilde { \mathbf { W } } _ { T - t } ) \Big | \leq \Delta ( 1 + \Big \| y + \widetilde { \mathbf { W } } _ { T - t } \Big \| ) ^ { p q } , } \\ & { } & { | f _ { 1 } ( v _ { 1 } ) - f _ { 2 } ( v _ { 1 } ) | \leq \Delta \big ( ( 1 + \Big \| y + \widetilde { \mathbf { W } } _ { s - t } \Big \| \big ) ^ { p q } + | v _ { 1 } | ^ { q } \big ) . } \end{array}
$$

Conditioning on $\mathbf { W } _ { t }$ and setting $\boldsymbol { y } = \boldsymbol { x } + \mathbf { W } _ { t }$ , independence and stationarity of the increments give $x + \mathbf { W } _ { t } + \widetilde { \mathbf { W } } _ { s - t } \overset { d } { = } x + \mathbf { W } _ { s }$ , so that, after taking expectations, the boundary term in (63) is at most $\Delta \Lambda$ and the data integrand is at most $\Delta \Lambda$ for every s. Hence, with $d ( r ) : = \mathbb { E } [ | v _ { 1 } ( r , x + \mathbf { W } _ { r } ) - v _ { 2 } ( r , x + \mathbf { W } _ { r } ) | ]$ , conditioning on $\mathbf { W } _ { r } ,$ applying (63) with initial time $r$ and $\boldsymbol { y } = \boldsymbol { x } + \mathbf { W } _ { r }$ , and using $\operatorname { L i p } ( f _ { 2 } ) \leq L$ yields $\begin{array} { r } { d ( r ) \leq [ 1 + ( T - r ) ] \Delta \Lambda + L \int _ { r } ^ { T } d ( s ) d s \leq } \end{array}$ $\begin{array} { r } { ( T + 1 ) \Delta \Lambda + L \int _ { r } ^ { T } d ( s ) } \end{array}$ ds for every $r \in [ 0 , T ]$ . Since $d \leq e$ and $\textstyle \int _ { r } ^ { T } e \leq \int _ { t } ^ { T }$ e for $r \geq t ,$ taking the supremum over $\dot { \boldsymbol { r } } \in [ t , T ]$ gives

$$
e ( t ) \leq ( T + 1 ) \Delta \Lambda + L \int _ { t } ^ { T } e ( s ) d s \qquad ( t \in [ 0 , T ] ) ,\tag{65}
$$

and the backward Gr¨onwall inequality of the proof of Lemma 3.2 yields

$$
e ( 0 ) \leq ( T + 1 ) \Delta \Lambda e ^ { L T } .\tag{66}
$$

To bound Λ, note that $\mathbb { E } [ ( 1 + \| x + \mathbf { W } _ { s } \| ) ^ { p q } ] \leq \Pi ( x )$ for every $s \in [ 0 , T ]$ , by the triangle inequality, Minkowski’s inequality in $L ^ { p q } ( \Omega , \mathbb { P } )$ , and $\mathbb { E } [ \| \mathbf { W } _ { s } \| ^ { p q } ] \leq \mathbb { E } [ \| \mathbf { W } _ { T } \| ^ { p q } ]$ , while Lemma 3.2 applied to $v _ { 1 }$ gives $\begin{array} { r } { \mathbb { E } [ | v _ { 1 } ( s , x + \mathbf { W } _ { s } ) | ^ { q } ] \leq ( e ^ { L T } ( T + 1 ) B ) ^ { q } \Pi ( x ) } \end{array}$ for every $s \in [ 0 , T ]$ ; hence

$$
\begin{array} { r } { \Lambda \leq \left( ( e ^ { L T } ( T + 1 ) B ) ^ { q } + 1 \right) \Pi ( x ) . } \end{array}\tag{67}
$$

Inserting (67) into (66) and using $( e ^ { L T } ( T + 1 ) B ) ^ { q } + 1 \le ( e ^ { L T } ( T + 1 ) ) ^ { q } ( B ^ { q } + 1 )$ , which holds because $e ^ { L T } ( T + 1 ) \geq 1$ , yields the claimed bound. □

Proof of Corollary $\ 3 . 4 \cdot$ . Fix $x \in \mathbb { R } ^ { d }$ , write $U : = U _ { N , M } ^ { 2 , 0 } ( 0 , x )$ , and let $\Pi ( x )$ be as in (62). Inserting the deterministic number $v _ { 2 } ( 0 , x )$ and using the triangle inequality in $L ^ { 2 } ( \Omega , \mathbb { P } )$ gives

$$
\begin{array} { r } { \bigl ( \mathbb { E } [ | U - v _ { 1 } ( 0 , x ) | ^ { 2 } ] \bigr ) ^ { 1 / 2 } \leq \underbrace { \bigl ( \mathbb { E } [ | U - v _ { 2 } ( 0 , x ) | ^ { 2 } ] \bigr ) ^ { 1 / 2 } } _ { \mathrm { M L P ~ a p p r o x i m a t i o n ~ e r r o r } } + \underbrace { | v _ { 2 } ( 0 , x ) - v _ { 1 } ( 0 , x ) | } _ { \mathrm { b i a s ~ f r o m ~ s u r r o g a t e s } } . } \end{array}\tag{68}
$$

Since $v _ { 1 }$ and $v _ { 2 }$ solve (9) with data difering by at most $\Delta$ in the sense of (8), Lemma 3.3 at $t = 0$ , where $\mathbf { W } _ { 0 } = 0$ , gives

$$
| v _ { 2 } ( 0 , x ) - v _ { 1 } ( 0 , x ) | \leq \Delta ( e ^ { L T } ( T + 1 ) ) ^ { q + 1 } ( B ^ { q } + 1 ) \Pi ( x ) .\tag{69}
$$

The estimator $U _ { N , M } ^ { 2 , 0 }$ is the full-history recursive MLP estimator for the surrogate solution $v _ { 2 }$ and the MLP convergence estimate [37, Theorem 3.5], in the form of [36, Corollary 2.4], controls its full $L ^ { 2 } .$ -error under the Lipschitz and qth-moment hypotheses of Setting 3.1:

$$
\bigl ( \mathbb { E } [ \bigl | U _ { N , M } ^ { 2 , 0 } ( 0 , x ) - v _ { 2 } ( 0 , x ) \bigr | ^ { 2 } ] \bigr ) ^ { 1 / 2 } \leq \frac { e ^ { M / 2 } ( 1 + 2 L T ) ^ { N } } { M ^ { N / 2 } } ( e ^ { L T } ( T + 1 ) ) ^ { q + 1 } ( B ^ { q } + 1 ) \Pi ( x ) .\tag{70}
$$

In (70), the factor $\Pi ( x )$ is obtained by bounding the terminal-data and $f _ { 2 } ( 0 )$ terms, Lemma 3.2 verifying the required $L ^ { 2 } { \mathrm { - i n t e g r a b i l i t y } } ;$ in (69), it arises from the Gaussian growth term together with the moment bound of Lemma 3.2 applied to $_ { v _ { 1 } ; }$ see (67). Inserting (69) and (70) into (68) and collecting the common factor $( e ^ { L T } ( T + 1 ) ) ^ { q + 1 } ( B ^ { q } + 1 ) \dot { \Pi } ( \dot { x } )$ gives (13). The detailed tracking of the constants in (70) is carried out in [36, Corollary 2.4]. □

Proof of Lemma 3.5. Set $\kappa _ { 0 } : = e ^ { 1 / 2 } ( 1 + 2 L T )$ and $E _ { n } : = ( \kappa _ { 0 } / \sqrt { n } ) ^ { n }$ for $n \in \mathbb { N }$ , so that (14) reads $E _ { N } \leq \varepsilon$ . Since $E _ { n } \to 0$ and

$$
{ \frac { n \log ( A n ) } { - \log E _ { n - 1 } } } = { \frac { n \log ( A n ) } { ( n - 1 ) { \bigl ( } { \frac { 1 } { 2 } } \log ( n - 1 ) - \log \kappa _ { 0 } { \bigr ) } } } \longrightarrow 2 \qquad ( n \to \infty ) ,
$$

there exists $N _ { 0 } \in \mathbb { N } \cap [ 2 , \infty )$ , depending only on $L , T , A , \xi$ , such that − log $E _ { n - 1 } > 0$ and this quotient is at most $2 + \xi$ for every $n > N _ { 0 }$ . Let $N \geq N _ { 0 }$ be minimal with $E _ { N } \leq \varepsilon$ . If $N > N _ { 0 }$ 2 then $E _ { N - 1 } > \varepsilon$ by minimality, and hence

$$
( A N ) ^ { N } = \exp \left( N \log ( A N ) \right) \leq E _ { N - 1 } ^ { - ( 2 + \xi ) } < \varepsilon ^ { - ( 2 + \xi ) } .
$$

If $N = N _ { 0 }$ , then $( A N ) ^ { N } = ( A N _ { 0 } ) ^ { N _ { 0 } }$ is a constant depending only on $L , T , A , \xi$ , which is absorbed into $C _ { \xi , A }$ . This establishes (14)–(15). □

## B Proof of the one-dimensional approximation lemma

We carry out the proof of Lemma 4.3. Steps 1–3 are a classical sigmoidal staircase argument with bandwidth equal to the grid spacing; Step 4 proves the finite-total-variation assertion by a variation-adapted staircase with a separated-center bandwidth.

Proof of Lemma 4.3. Since a is nondecreasing, its normalization $\widetilde { a }$ is nondecreasing with limits 0 and 1. Consequently,

$$
0 \leq \widetilde { \boldsymbol { a } } \leq 1 , \qquad \widetilde { \boldsymbol { a } } ^ { \prime } \geq 0 , \qquad \left\| \widetilde { \boldsymbol { a } } ^ { \prime } \right\| _ { L ^ { 1 } ( \mathbb { R } ) } = \mathrm { T V } ( \widetilde { \boldsymbol { a } } ) = 1 .\tag{71}
$$

Step 1: The staircase target. Fix $R \in [ 1 , \infty )$ and $n \in \mathbb { N } .$ , and set $\eta : = 2 R / n$ . For $k \in$ $\{ 0 , 1 , \ldots , n \}$ put $v _ { k } : = - R + k \eta$ . Define the midpoint staircase $S _ { n } \colon \mathbb { R } \to \mathbb { R }$ by

$$
S _ { n } ( v ) : = \phi ( v _ { 0 } ) + \sum _ { k = 1 } ^ { n } \Delta _ { k } \mathbf { 1 } _ { ( c _ { k } , \infty ) } ( v ) , \qquad \Delta _ { k } : = \phi ( v _ { k } ) - \phi ( v _ { k - 1 } ) , \quad c _ { k } : = \frac { v _ { k - 1 } + v _ { k } } { 2 } .\tag{72}
$$

By construction $S _ { n } ( v ) = \phi ( v _ { k } )$ for $v \in \left( c _ { k } , c _ { k + 1 } \right)$ with $c _ { 0 } : = - \infty , c _ { n + 1 } : = + \infty$ , and the staircase is constant outside its central region: $S _ { n } ( v ) = \phi ( v _ { 0 } )$ for $v \leq c _ { 1 }$ and $S _ { n } ( v ) = \phi ( v _ { n } )$ for $v > c _ { n }$ At each jump point $c _ { k } .$ , one has $S _ { n } ( c _ { k } ) = \phi ( v _ { k - 1 } )$ and $| c _ { k } - v _ { k - 1 } | = R / n$ . Since $\phi$ is $\operatorname { L i p } ( \phi ) -$ Lipschitz and $| v - v _ { k } | \le R / n$ for $v \in [ - R , R ] \cap ( c _ { k } , c _ { k + 1 } )$ and every $k \in \{ 0 , \ldots , n \}$ ,

$$
\operatorname* { s u p } _ { v \in [ - R , R ] } | \phi ( v ) - S _ { n } ( v ) | \leq \mathrm { L i p } ( \phi ) { \frac { R } { n } } , \qquad | \Delta _ { k } | \leq \mathrm { L i p } ( \phi ) \eta .\tag{73}
$$

If $\phi$ is constant on $( - \infty , - R ]$ and on $[ R , \infty )$ , then the same estimate holds with $[ - R , R ]$ replaced by R, because $S _ { n } ( v ) = \phi ( v )$ for $v \not \in [ - R , R ]$

Step 2: Sigmoidal smoothing. We use the elementary sampling estimate

$$
\operatorname* { s u p } _ { x \in \mathbb { R } } \sum _ { j \in \mathbb { Z } } | g ( x - j ) | \leq \| g \| _ { L ^ { 1 } ( \mathbb { R } ) } + \mathrm { T V } ( g )\tag{74}
$$

for every integrable function $g \colon  { \mathbb { R } } \ \to \  { \mathbb { R } }$ whose $\mathrm { g i }$ ven pointwise representative has bounded variation. Indeed, by periodicity of the sum it sufices to take $x \in [ 0 , 1 )$ . For every $m \in \mathbb { Z } .$

$$
| g ( x + m ) | \leq \int _ { m } ^ { m + 1 } | g ( r ) | \ d r + \mathrm { T V } \big ( g ; [ m , m + 1 ) \big ) ,
$$

and summing over m proves (74).

Let $H : = { \bf 1 } _ { ( 0 , \infty ) } \ ( \mathrm { s o } \ H ( 0 ) = 0 )$ and $\psi _ { a } : = | \widetilde { a } - H |$ . Since the absolute-value map is 1- Lipschitz, (71) gives

$$
\| \psi _ { a } \| _ { L ^ { 1 } ( \mathbb { R } ) } = \widetilde { \Lambda } _ { a } , \qquad \mathrm { T V } ( \psi _ { a } ) \leq \mathrm { T V } ( \widetilde { a } - H ) \leq \mathrm { T V } ( \widetilde { a } ) + \mathrm { T V } ( H ) = 2 .\tag{75}
$$

Set the bandwidth $h : = \eta = 2 R / n$ and define

$$
Q _ { n , R } ( v ) : = \phi ( v _ { 0 } ) + \sum _ { k = 1 } ^ { n } \Delta _ { k } \widetilde a \left( \frac { v - c _ { k } } { \eta } \right) .\tag{76}
$$

Since $c _ { k } = c _ { 1 } + ( k - 1 ) \eta$ , writing $x : = ( v - c _ { 1 } ) / \eta$ and applying (74) to $\psi _ { a }$ yield, for every $v \in \mathbb { R }$

$$
\begin{array} { l } { \displaystyle \left| Q _ { n , R } ( v ) - S _ { n } ( v ) \right| \leq \sum _ { k = 1 } ^ { n } \left| \Delta _ { k } \right| \psi _ { a } ( x - ( k - 1 ) ) } \\ { \leq \mathrm { L i p } ( \phi ) \eta \sum _ { j \in \mathbb { Z } } \psi _ { a } ( x - j ) \leq \mathrm { L i p } ( \phi ) \eta \big ( \widetilde { \Lambda } _ { a } + 2 \big ) . } \end{array}
$$

Together with (73), this proves

$$
\operatorname* { s u p } _ { v \in [ - R , R ] } | \phi ( v ) - Q _ { n , R } ( v ) | \leq \bigl ( 5 + 2 \widetilde { \Lambda } _ { a } \bigr ) \mathrm { L i p } ( \phi ) \frac { R } { n } .\tag{77}
$$

If ϕ is constant on the two exterior intervals appearing in the lemma, the global version of (73) established in Step 1 and the pointwise estimate above give the same bound with $[ - R , R ]$ replaced by R.

The function $Q _ { n , R }$ is continuously diferentiable. Applying (74) to the continuous representative of $\widetilde { a } ^ { \prime }$ gives

$$
\begin{array} { l } { \displaystyle \left. Q _ { n , R } ^ { \prime } ( v ) \right. \leq \sum _ { k = 1 } ^ { n } \frac { \vert \Delta _ { k } \vert } { \eta } \left. \widetilde { a } ^ { \prime } ( x - ( k - 1 ) ) \right. } \\ { \leq \mathrm { L i p } ( \phi ) \sum _ { j \in \mathbb { Z } } \left. \widetilde { a } ^ { \prime } ( x - j ) \right. \leq \mathrm { L i p } ( \phi ) \big ( 1 + \mathrm { T V } ( \widetilde { a } ^ { \prime } ) \big ) . } \end{array}
$$

Consequently,

$$
\mathrm { L i p } ( Q _ { n , R } ) \leq \big ( 1 + \mathrm { T V } ( \widetilde a ^ { \prime } ) \big ) \mathrm { L i p } ( \phi ) .\tag{78}
$$

Step 3: Realization as an FNN. Define $\phi _ { n , R } : = \left( ( W _ { 1 } , B _ { 1 } ) , ( W _ { 2 } , B _ { 2 } ) \right) \in \mathbf { N }$ with $W _ { 1 } \in \mathbb { R } ^ { n \times 1 }$ $B _ { 1 } \in \mathbb { R } ^ { n } , W _ { 2 } \in \mathbb { R } ^ { 1 \times n } , B _ { 2 } \in \mathbb { R }$ given by

$$
( W _ { 1 } ) _ { k , 1 } : = \frac { 1 } { \eta } , \qquad ( B _ { 1 } ) _ { k } : = - \frac { c _ { k } } { \eta } , \qquad ( W _ { 2 } ) _ { 1 , k } : = \frac { \Delta _ { k } } { A _ { + } - A _ { - } } , \qquad B _ { 2 } : = \phi ( v _ { 0 } ) - \frac { A _ { - } } { A _ { + } - A _ { - } } \sum _ { i = 1 } ^ { n } \Delta _ { j } ,\tag{79}
$$

for $k \in \{ 1 , \ldots , n \}$ . Since $\widetilde { a } = ( a - A _ { - } ) / ( A _ { + } - A _ { - } )$ , Definition 2.1 gives $( \mathcal { R } _ { a } \phi _ { n , R } ) ( v ) = \phi ( v _ { 0 } ) +$ $\begin{array} { r } { \sum _ { k = 1 } ^ { n } \Delta _ { k } \widetilde { a } ( ( v - c _ { k } ) / \eta ) = Q _ { n , R } ( v ) } \end{array}$ for every $v \in \mathbb { R }$ , so that $\mathcal { D } ( \phi _ { n , R } ) = ( 1 , n , 1 )$ and $\mathcal { P } ( \phi _ { n , R } ) =$ $2 n + ( n + 1 ) = 3 n + 1$ . Finally, (77) and (78) give (22)–(23), as well as (24), with, for example,

$$
C _ { a } : = 5 + 2 \widetilde { \Lambda } _ { a } + \mathrm { T V } ( \widetilde { a } ^ { \prime } ) .
$$

Step 4: The finite-total-variation assertion. The notation of Steps 1–3 is not used in this step. Let ϕ have finite total variation $T : = \mathrm { T V } ( \phi )$ , let $\delta \in ( 0 , 1 ]$ , and set $\lambda : = \delta / 3$ and

$$
\varpi _ { a } ( t ) : = \operatorname* { s u p } _ { | r | \geq t } \left| \widetilde { a } ( r ) - \mathbf { 1 } _ { ( 0 , \infty ) } ( r ) \right| \qquad ( t \in ( 0 , \infty ) ) ,
$$

so that $\varpi _ { a } ( t ) \to 0$ as $t  \infty$ by (19). Since $\phi$ is continuous with finite total variation, the finite limits $\scriptstyle \operatorname* { l i m } _ { u \to \pm \infty } \phi ( u )$ exist. Let $V ( v ) : = \operatorname { T V } ( \phi ; ( - \infty , v ] )$ ; by additivity of the total variation, |ϕ(v) − lim $_ { u \to \infty } \phi ( u ) | \ \leq \ \mathrm { T V } ( \phi ; [ v , \infty ) ) = T - V ( v )$ for every $v \in \mathbb { R } .$ . If $T \le \lambda$ , then $\begin{array} { r } { | \phi ( v ) - \operatorname* { l i m } _ { u \to \infty } \phi ( u ) | \le \lambda \le \delta } \end{array}$ for every $v \in \mathbb { R }$ , and the network of architecture $( 1 , 1 , 1 )$ with zero weight matrices, zero hidden bias, and output bias lim $\iota _ { u \to \infty } \phi ( u )$ satisfies (25). Assume now $T > \lambda ;$ then $\phi$ is not constant, so $L : = \mathrm { L i p } ( \phi ) \in ( 0 , \infty )$ . The function V is nondecreasing with lim $\mathfrak { l } _ { v \to - \infty } V ( v ) = 0$ and lim $_ { \cdot v  \infty } V ( v ) = T$ ; moreover, for $x < y .$ , additivity gives $0 \leq V ( y ) - V ( x ) = \mathrm { T V } ( \phi ; [ x , y ] ) \leq L \left( y - x \right)$ , so V is L-Lipschitz and, in particular, continuous. Set $K : = \lceil T / \lambda \rceil \geq 2$ and, for $k \in \{ 1 , \ldots , K - 1 \}$ , let

$$
c _ { k } : = \operatorname* { i n f } \{ v \in \mathbb { R } : V ( v ) \geq k \lambda \} .
$$

Since $k \lambda \le ( K { - } 1 ) \lambda < T$ and V is continuous with the stated limits, each $c _ { k }$ is a real number with $V ( c _ { k } ) = k \lambda$ , and $c _ { 1 } < c _ { 2 } < \cdots < c _ { K - 1 } ,$ , since $c _ { k + 1 } \geq c _ { k }$ and $V ( c _ { k + 1 } ) = ( k + 1 ) \lambda \neq k \lambda = V ( c _ { k } )$ Moreover, the Lipschitz bound for V gives, for every $k \in \{ 1 , \ldots , K - 2 \}$ 2

$$
\lambda = V ( c _ { k + 1 } ) - V ( c _ { k } ) \leq L \left( c _ { k + 1 } - c _ { k } \right) , \qquad { \mathrm { h e n c e } } \qquad c _ { k + 1 } - c _ { k } \geq \lambda / L .\tag{80}
$$

Consider the cells $I _ { 0 } : = ( - \infty , c _ { 1 } ] , I _ { k } : = ( c _ { k } , c _ { k + 1 } ]$ for $k \in \{ 1 , \ldots , K { - } 2 \}$ , and $I _ { K - 1 } : = ( c _ { K - 1 } , \infty )$ and the values $s _ { 0 } : = \phi ( c _ { 1 } ) , s _ { k } : = \phi ( c _ { k + 1 } )$ for $k \in \{ 1 , . . . , K - 2 \}$ , and $s _ { K - 1 } : = \operatorname* { l i m } _ { u \to \infty } \phi ( u )$ The variation of $\phi$ over the closure of each cell is at most λ: it equals $V ( c _ { 1 } ) = \lambda$ for $I _ { 0 }$ , equals $V ( c _ { k + 1 } ) - V ( c _ { k } ) = \lambda$ for the interior cells, and is $T - V ( c _ { K - 1 } ) = T - ( K - 1 ) \lambda \le \lambda$ for $I _ { K - 1 }$ For $v \in I _ { k }$ with $k \in \{ 0 , \ldots , K - 2 \}$ , both v and the sample point defining $s _ { k }$ lie in the closure of $I _ { k } , \mathrm { ~ s o ~ } | \phi ( v ) - s _ { k } | \leq \lambda ;$ for $v \in I _ { K - 1 }$ , the tail bound above gives $| \phi ( v ) - s _ { K - 1 } | \leq T - V ( v ) \leq$ $T - V ( c _ { K - 1 } ) \le \lambda$ . Hence the staircase

$$
S ( v ) : = s _ { 0 } + \sum _ { k = 1 } ^ { K - 1 } \Delta _ { k } \mathbf { 1 } _ { ( c _ { k } , \infty ) } ( v ) , \qquad \Delta _ { k } : = s _ { k } - s _ { k - 1 } ,
$$

satisfies $\begin{array} { r } { \operatorname* { s u p } _ { v \in \mathbb { R } } | \phi ( v ) - S ( v ) | \leq \lambda } \end{array}$ . Moreover, $| \Delta _ { k } | \le \lambda$ for every k by the cellwise variation bounds, while $\begin{array} { r } { \sum _ { k = 1 } ^ { K - 1 } \left| \Delta _ { k } \right| \leq T } \end{array}$ follows from the definition of the total variation applied to the ordered sample points $c _ { 1 } < \dots < c _ { K - 1 }$ , followed by passage to the limit at $+ \infty$

Since $\varpi _ { a } ( t ) \to 0$ as $t \to \infty ,$ , we may choose $\theta \in ( 0 , 1 ]$ , depending only on a and $T / \lambda$ , such that $T \varpi _ { a } ( 1 / ( 2 \theta ) ) \leq \lambda$ . Set $h : = \theta \lambda / L$ and

$$
Q ( v ) : = s _ { 0 } + \sum _ { k = 1 } ^ { K - 1 } \Delta _ { k } \widetilde { a } \big ( ( v - c _ { k } ) / h \big ) \qquad ( v \in \mathbb { R } ) .
$$

As in Step 3, Q is the realization of an FNN $\phi _ { \delta }$ of architecture $( 1 , K - 1 , 1 )$ with first-layer weights $1 / h$ , first-layer biases $- c _ { k } / h$ , output weights $\Delta _ { k } / ( A _ { + } - A _ { - } )$ , and output bias $s _ { 0 } \textrm { -- }$ $\textstyle \sum _ { k = 1 } ^ { \tilde { K } - 1 } \Delta _ { k } \overset { \cdot } { A } _ { - } / ( A _ { + } - \overset { \cdot } { A } _ { - } )$ . We estimate $\begin{array} { r } { | Q ( v ) - S ( v ) | \leq \sum _ { k = 1 } ^ { K - 1 } | \Delta _ { k } | \ \psi \big ( ( v - c _ { k } ) / h \big ) } \end{array}$ , where $\psi : =$ $\left| \widetilde { \boldsymbol { a } } - \mathbf { 1 } _ { ( 0 , \infty ) } \right|$ satisfies $\psi \leq 1$ by (71) and $\psi ( r ) \leq \varpi _ { a } ( | r | )$ for $r \neq 0$ . Fix $v \in \mathbb { R }$ and let $k ^ { * }$ minimize $\left| v - c _ { k } \right|$ over k. By (80), distinct centers are at least $\lambda / L$ apart, and $\begin{array} { r } { | v - c _ { k } | \ge \frac 1 2 \left| c _ { k } - c _ { k ^ { * } } \right| } \end{array}$ for $k \neq k ^ { * }$ by the choice of $k ^ { * } ,$ ; hence $| v - c _ { k } | \ge \lambda / ( 2 L )$ and $\left| v - c _ { k } \right| / h \geq 1 / ( 2 \theta )$ for every $k \neq k ^ { * }$ Consequently,

$$
| Q ( v ) - S ( v ) | \leq | \Delta _ { k ^ { * } } | + \Big ( \sum _ { k \neq k ^ { * } } | \Delta _ { k } | \Big ) \varpi _ { a } \big ( 1 / ( 2 \theta ) \big ) \leq \lambda + T \varpi _ { a } \big ( 1 / ( 2 \theta ) \big ) \leq 2 \lambda .
$$

Together with the staircase bound, $\begin{array} { r } { \operatorname* { s u p } _ { v \in \mathbb { R } } | \phi ( v ) - ( \mathcal { R } _ { a } \phi _ { \delta } ) ( v ) | \leq 3 \lambda = \delta _ { \mathrm { m } } } \end{array}$ , which is (25) with $m = K - 1 \leq \lceil 3 T / \delta \rceil - 1$ □

## Acknowledgements

This project has been partially supported by DFG – Project-ID 499552394 – SFB 1597.

Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

The key ideas of the statements and the proofs of the main results of this work are due to the authors. During the preparation of this work, the authors used OpenAI’s ChatGPT and the AI assistant Claude (Anthropic) to improve the language and readability of the manuscript, to check the consistency of notation, cross-references, to verify bibliographic details against the cited sources, and to flag possible issues concerning mathematical expositions. All suggestions and proposed revisions were critically reviewed by the authors. The authors independently verified all mathematical statements, proofs, and references, made all final decisions concerning the manuscript, and take full responsibility for the content of the published article.

## References

[1] Ackermann, J., Jentzen, A., Kruse, T., Kuckuck, B., and Padgett, J. L. Deep neural networks with ReLU, leaky ReLU, and softplus activation provably overcome the curse of dimensionality for Kolmogorov partial diferential equations with Lipschitz nonlinearities in the L<sup>p</sup>-sense. arXiv preprint (2023). arXiv:2309.13722.

[2] Ackermann, J., Jentzen, A., Kuckuck, B., and Padgett, J. L. Deep neural networks with ReLU, leaky ReLU, and softplus activation provably overcome the curse of dimensionality for space-time solutions of semilinear partial diferential equations. arXiv preprint (2024). arXiv:2406.10876.

[3] Baggenstos, J., and Salimova, D. Approximation properties of residual neural networks for Kolmogorov PDEs. Discrete Contin. Dyn. Syst. Ser. B 28, 5 (2023), 3193–3215.

[4] Beck, C., Becker, S., Grohs, P., Jaafari, N., and Jentzen, A. Solving the Kolmogorov PDE by means of deep learning. J. Sci. Comput. 88 (2021), 73.

[5] Beck, C., Gonon, L., Hutzenthaler, M., and Jentzen, A. On existence and uniqueness properties for solutions of stochastic fixed point equations. Discrete Contin. Dyn. Syst. Ser. B 26, 9 (2021), 4927–4962.

[6] Beck, C., Hornung, F., Hutzenthaler, M., Jentzen, A., and Kruse, T. Overcoming the curse of dimensionality in the numerical approximation of Allen–Cahn PDEs via truncated full-history recursive MLP approximations. J. Numer. Math. 28, 4 (2020), 197–222.

[7] Beck, C., Hutzenthaler, M., Jentzen, A., and Kuckuck, B. An overview on deep learning-based approximation methods for partial diferential equations. Discrete Contin. Dyn. Syst. Ser. B 28, 6 (2023), 3697–3746.

[8] Becker, S., Braunwarth, R., Hutzenthaler, M., Jentzen, A., and von Wurstemberger, P. Numerical simulations for full history recursive multilevel Picard approximations for systems of high-dimensional partial diferential equations. Commun. Comput. Phys. 28, 5 (2020), 2109–2138.

[9] Bellman, R. Dynamic Programming. Princeton University Press, 1957.

[10] Berner, J., Grohs, P., and Jentzen, A. Analysis of the generalization error: empirical risk minimization over deep artificial neural networks overcomes the curse of dimensionality in the numerical approximation of Black–Scholes PDEs. SIAM J. Math. Data Sci. 2, 3 (2020), 631–657.

[11] Chen, R. T. Q., Rubanova, Y., Bettencourt, J., and Duvenaud, D. Neural ordinary diferential equations. In Advances in Neural Information Processing Systems (2018), vol. 31, Curran Associates, Inc., pp. 6571–6583.

[12] Cioica-Licht, P. A., Hutzenthaler, M., and Werner, P. T. Deep neural networks overcome the curse of dimensionality in the numerical approximation of semilinear partial diferential equations. arXiv preprint (2022). arXiv:2205.14398. To appear in Commun. Math. Sci.

[13] Cybenko, G. Approximation by superpositions of a sigmoidal function. Math. Control Signals Systems 2, 4 (1989), 303–314.

[14] E, W. A proposal on machine learning via dynamical systems. Commun. Math. Stat. 5, 1 (2017), 1–11.

[15] E, W., Han, J., and Jentzen, A. Deep learning-based numerical methods for highdimensional parabolic partial diferential equations and backward stochastic diferential equations. Commun. Math. Stat. 5, 4 (2017), 349–380.

[16] E, W., Hutzenthaler, M., Jentzen, A., and Kruse, T. On multilevel Picard numerical approximations for high-dimensional nonlinear parabolic PDEs and high-dimensional nonlinear BSDEs. J. Sci. Comput. 79, 3 (2019), 1534–1571.

[17] E, W., Hutzenthaler, M., Jentzen, A., and Kruse, T. Multilevel Picard iterations for solving smooth semilinear parabolic heat equations. Partial Difer. Equ. Appl. 2 (2021), 80.

[18] E, W., and Yu, B. The deep Ritz method: a deep learning-based numerical algorithm for solving variational problems. Commun. Math. Stat. 6, 1 (2018), 1–12.

[19] Elbrachter, D., Grohs, P., Jentzen, A., and Schwab, C.<sup>¨</sup> DNN expression rate analysis of high-dimensional PDEs: application to option pricing. Constr. Approx. 55, 1 (2022), 3–71.

[20] Friedman, A. Partial Diferential Equations of Parabolic Type. Prentice–Hall, 1964.

[21] Gonon, L. Random feature neural networks learn Black–Scholes type PDEs without curse of dimensionality. J. Mach. Learn. Res. 24, 189 (2023), 1–51.

[22] Gonon, L., Grohs, P., Jentzen, A., Kofler, D., and Si<sup>ˇ</sup> <sup>ˇ</sup>ska, D. Uniform error estimates for artificial neural network approximations for heat equations. IMA J. Numer. Anal. 42, 3 (2022), 1991–2054.

[23] Gonon, L., and Schwab, C. Deep ReLU network expression rates for option prices in high-dimensional, exponential L´evy models. Finance Stoch. 25, 4 (2021), 615–657.

[24] Gonon, L., and Schwab, C. Deep ReLU neural networks overcome the curse of dimensionality for partial integro-diferential equations. Anal. Appl. 21, 1 (2023), 1–47.

[25] Grohs, P., and Herrmann, L. Deep neural network approximation for high-dimensional elliptic PDEs with boundary conditions. IMA J. Numer. Anal. 42, 3 (2022), 2055–2082.

[26] Grohs, P., Hornung, F., Jentzen, A., and von Wurstemberger, P. A proof that artificial neural networks overcome the curse of dimensionality in the numerical approximation of Black–Scholes partial diferential equations. Mem. Amer. Math. Soc. 284, 1410 (2023).

[27] Grohs, P., Ibragimov, S., Jentzen, A., and Koppensteiner, S. Lower bounds for artificial neural network approximations: A proof that shallow neural networks fail to overcome the curse of dimensionality. J. Complexity 77 (2023), 101746.

[28] Grohs, P., and Voigtlander, F. <sup>¨</sup> Proof of the theory-to-practice gap in deep learning via sampling complexity bounds for neural network approximation spaces. Found. Comput. Math. 24, 4 (2024), 1085–1143.

[29] Haber, E., and Ruthotto, L. Stable architectures for deep neural networks. Inverse Problems 34, 1 (2018), 014004.

[30] Han, J., Jentzen, A., and E, W. Solving high-dimensional PDEs using deep learning. Proc. Natl. Acad. Sci. USA 115, 34 (2018), 8505–8510.

[31] He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR) (2016), pp. 770–778.

[32] He, K., Zhang, X., Ren, S., and Sun, J. Identity mappings in deep residual networks. In European Conf. Comput. Vis. (ECCV) (2016), Springer, pp. 630–645.

[33] Hornik, K., Stinchcombe, M., and White, H. Multilayer feedforward networks are universal approximators. Neural Networks 2, 5 (1989), 359–366.

[34] Hornung, F., Jentzen, A., and Salimova, D. Space-time deep neural network approximations for high-dimensional partial diferential equations. J. Comput. Math. 43, 4 (2025), 918–975.

[35] Hutzenthaler, M., Jentzen, A., Kruse, T., and Nguyen, T. A. Multilevel Picard approximations for high-dimensional semilinear second-order PDEs with Lipschitz nonlinearities. arXiv preprint (2020). arXiv:2009.02484.

[36] Hutzenthaler, M., Jentzen, A., Kruse, T., and Nguyen, T. A. A proof that rectified deep neural networks overcome the curse of dimensionality in the numerical approximation of semilinear heat equations. SN Partial Difer. Equ. Appl. 1, 2 (2020), 10.

[37] Hutzenthaler, M., Jentzen, A., Kruse, T., Nguyen, T. A., and von Wurstemberger, P. Overcoming the curse of dimensionality in the numerical approximation of semilinear parabolic partial diferential equations. Proc. R. Soc. A 476 (2020), 20190630.

[38] Hutzenthaler, M., and Kruse, T. Multilevel Picard approximations of highdimensional semilinear parabolic diferential equations with gradient-dependent nonlinearities. SIAM J. Numer. Anal. 58, 2 (2020), 929–961.

[39] Jentzen, A., Salimova, D., and Welti, T. A proof that deep artificial neural networks overcome the curse of dimensionality in the numerical approximation of Kolmogorov partial diferential equations with constant difusion and nonlinear drift coeficients. Commun. Math. Sci. 19, 5 (2021), 1167–1205.

[40] Leshno, M., Lin, V. Y., Pinkus, A., and Schocken, S. Multilayer feedforward networks with a nonpolynomial activation function can approximate any function. Neural Networks 6, 6 (1993), 861–867.

[41] Lin, H., and Jegelka, S. ResNet with one-neuron hidden layers is a universal approximator. In Advances in Neural Information Processing Systems (2018), vol. 31, Curran Associates, Inc., pp. 6169–6178.

[42] Lu, Y., Zhong, A., Li, Q., and Dong, B. Beyond finite layer neural networks: bridging deep architectures and numerical diferential equations. In Proc. 35th Int. Conf. Mach. Learn. (ICML) (2018), vol. 80 of PMLR, pp. 3276–3285.

[43] Mhaskar, H. N. Neural networks for optimal approximation of smooth and analytic functions. Neural Comput. 8, 1 (1996), 164–177.

[44] Neufeld, A., and Nguyen, T. A. Rectified deep neural networks overcome the curse of dimensionality in the numerical approximation of gradient-dependent semilinear heat equations. Commun. Math. Sci. 23, 4 (2025), 883–912.

[45] Neufeld, A., and Nguyen, T. A. Multilevel Picard approximations and deep neural networks with ReLU, leaky ReLU, and softplus activation overcome the curse of dimensionality when approximating semilinear parabolic partial diferential equations in L<sup>p</sup>-sense. J. Comput. Appl. Math. 487 (2026), 117736.

[46] Neufeld, A., Nguyen, T. A., and Wu, S. Deep ReLU neural networks overcome the curse of dimensionality when approximating semilinear partial integro-diferential equations. Anal. Appl. 23, 7 (2025), 1227–1278.

[47] Neufeld, A., Nguyen, T. A., and Wu, S. Multilevel Picard approximations overcome the curse of dimensionality in the numerical approximation of general semilinear PDEs with gradient-dependent nonlinearities. J. Complexity 90 (2025), 101946.

[48] Neufeld, A., Schmocker, P., and Wu, S. Full error analysis of the random deep splitting method for nonlinear parabolic PDEs and PIDEs. Commun. Nonlinear Sci. Numer. Simul. 143 (2025), 108556.

[49] Neufeld, A., and Wu, S. Multilevel Picard approximation algorithm for semilinear partial integro-diferential equations and its complexity analysis. Stoch. Partial Difer. Equ. Anal. Comput. 13, 3 (2025), 1220–1278.

[50] Novak, E., and Wo<sup>´</sup>zniakowski, H. Tractability of Multivariate Problems. Volume I: Linear Information, vol. 6 of EMS Tracts Math. European Mathematical Society, 2008.

[51] Pardoux, E., and Peng, S. <sup>´</sup> Backward stochastic diferential equations and quasilinear parabolic partial diferential equations. In Stoch. Partial Difer. Equ. Appl., vol. 176 of Lect. Notes Control Inf. Sci. Springer, 1992, pp. 200–217.

[52] Petersen, P., and Voigtlander, F.<sup>¨</sup> Optimal approximation of piecewise smooth functions using deep ReLU neural networks. Neural Networks 108 (2018), 296–330.

[53] Pinkus, A. Approximation theory of the MLP model in neural networks. Acta Numer. 8 (1999), 143–195.

[54] Raissi, M., Perdikaris, P., and Karniadakis, G. E. Physics-informed neural networks: a deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations. J. Comput. Phys. 378 (2019), 686–707.

[55] Reisinger, C., and Zhang, Y. Rectified deep neural networks overcome the curse of dimensionality for nonsmooth value functions in zero-sum games of nonlinear stif systems. Anal. Appl. 18, 6 (2020), 951–999.

[56] Sirignano, J., and Spiliopoulos, K. DGM: a deep learning algorithm for solving partial diferential equations. J. Comput. Phys. 375 (2018), 1339–1364.

[57] Tabuada, P., and Gharesifard, B. Universal approximation power of deep residual neural networks through the lens of control. IEEE Trans. Automat. Control 68, 5 (2023), 2715–2728.

[58] Yang, J., and Pan, Q. Numerical analysis of residual neural networks for stochastic partial diferential equations. Rev. R. Acad. Cienc. Exactas F´ıs. Nat. Ser. A Mat. RACSAM 120, 4 (2026), 105.

[59] Yarotsky, D. Error bounds for approximations with deep ReLU networks. Neural Networks 94 (2017), 103–114.