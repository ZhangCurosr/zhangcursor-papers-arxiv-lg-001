# Phases in a class of associative memories via hidden neurons

Toshihiro Ota<sup>1,</sup> <sup>3,</sup> <sup>∗</sup> and Masato Taki<sup>2,</sup> <sup>3,</sup> <sup>†</sup>

<sup>1</sup>CyberAgent AI Lab, Shibuya, Tokyo 150–0002, Japan

<sup>2</sup>Graduate School of Artificial Intelligence and Science, Rikkyo University, Toshima, Tokyo 171–8501, Japan <sup>3</sup>RIKEN iTHEMS, Wako, Saitama 351–0198, Japan

Associative memory in the Hopfield network is attractor dynamics in a disordered many-body system, and higher-order and exponential extensions turn its retrieval update into softmax attention. The polynomial and exponential regimes have been analyzed by diferent methods, with no common architecture in which to ask what fixes the storage scale. In this paper we study the bipartite architecture of Krotov and Hopfield, which we call the class H, whose model is fixed by a Lagrangian for each layer, taking the hidden neurons as the order parameter of retrieval. At polynomial load the replica method yields the replica-symmetric phase diagrams and closed-form capacities, and the crosstalk moment is common to Ising and spherical visible neurons, so their diferences come from the visible entropy. With a softmax hidden layer the load is exponential, and a copy representation maps the thermodynamics onto random-energy-model counting, with paramagnetic, condensed, and frozen phases. Heating destabilizes retrieval by quantized reassignments of attention, and typical Gaussian patterns remain metastable at every load. The regimes difer in their crosstalk statistics, central-limit at polynomial load and large-deviation at exponential load, and the class H splits retrieval into two roles, the visible Lagrangian fixing stability and the hidden one the storage scale, two axes that may also guide the design of new Lagrangians.

## I. INTRODUCTION

Associative memory is a content-addressable mechanism that retrieves a whole memory from a partial cue. The Hopfield network formalized this retrieval as the collective dynamics of many interacting neurons, with the stored patterns realized as attractors of an energy landscape [1]. The network is thereby a disordered many-body system, and the replica method of spin-glass statistical mechanics quantified the competition between the retrieval and the spin-glass phases and the storage capacity of the model [2–4].

This framework has since been extended considerably [5, 6]. Replacing the pairwise interactions by higher-order ones raises the number of retrievable patterns from linear in the number of neurons to polynomial, with a degree that grows with the order of the interaction [7–10], and exponential interactions make it exponentially large in the number of neurons [11]. With a log-sum-exp energy the retrieval update takes the form of softmax attention, placing associative memory in direct correspondence with the attention mechanism of the transformer [12–14].

The two regimes have been analyzed with diferent tools: the replica method at polynomial load, and largedeviation and extreme-value statistics at exponential load, which control both the retrieval thresholds [15] and the finite-temperature transitions [16–18]. What is missing is a setting in which one can ask, within a single architecture, what fixes the storage scale and how the crosstalk statistics change with it. The bipartite architecture of Krotov and Hopfield provides one [19]: a visible and a hidden layer coupled only by pairwise interactions, in which the choice of a Lagrangian for each layer alone determines the model. We call this two-layer family the class H, the k = 2 member of a hierarchical class H<sub>k</sub> of k coupled layers, and describe it in Sec. II. Its three representatives singled out in [19], studied so far as separate models with diferent efective energies, are Model A, with Ising visible neurons, Model B, whose hidden layer is a softmax, and Model C, with spherical visible neurons. We take a further step and treat the hidden neurons themselves as the order parameter of memory retrieval, which describes diferent visible geometries and hidden nonlinearities in one language.

Sec. III treats the statistical mechanics of Models A and C at polynomial load, where the replica method yields the replica-symmetric phase diagrams, with the zero-temperature capacities in closed form. The moment that measures the crosstalk of the non-retrieved patterns is common to the two models, while their retrieval phases difer sharply: the quadratic spherical model is marginal and stores nothing [20], whereas the higher-order interaction restores a retrieval phase. In Sec. IV, we study Model B, whose hidden sector reduces to the attention weights assigned to the stored patterns, so that retrieval is the concentration of attention. Its natural load is exponential, and a representation of the thermodynamics in terms of a temperature-dependent number of copies, each selecting one stored pattern, turns the problem into one of counting with the structure of the random energy model [21, 22]. This yields the paramagnetic, condensed, and frozen phases, and shows that heating destabilizes retrieval not by a smooth erosion of the overlap but by quantized reassignments of attention, the retrieval of a typical Gaussian pattern remaining metastable at every load.

What separates the two regimes is the character of the crosstalk statistics, central-limit and insensitive to the pattern ensemble at polynomial load, large-deviation and ensemble-dependent at exponential load. One consequence is that at exponential load the rare Gaussian patterns of atypically large norm lie below typical retrieval in free energy, so that a typical memory is never the equilibrium phase and the network operates as a metastable device. Underlying both statements is the picture the class H supplies, in which the retrieval problem separates into two roles carried by the two Lagrangians. The visible Lagrangian fixes, through the visible entropy, the stability of retrieval under a given crosstalk, which is what isolates the diference between Models A and C. The hidden Lagrangian fixes the storage scale and the character of the disorder statistics, a polynomial nonlinearity yielding polynomial load and central-limit crosstalk, and the log-sum-exp yielding exponential load and large deviations. In these terms the sequence from the classical Hopfield network through dense associative memory to attention is not a succession of separate theories, but a set of positions on these two axes, compared in the common order-parameter language that the hidden neurons provide.<sup>1</sup>

## II. PRELIMINARIES

To fix notation, in this section we give an overview of the class H, originally proposed in [19], and provide the statistical mechanical setup for our main discussions in the subsequent sections. Details of the class H and of the more general $\mathrm { c l a s s } { - \mathcal { H } _ { k } }$ associative memories are presented in Appendix A.

## A. Overview of the class H

The dynamical variables in this system consist of $N _ { v }$ visible neurons $\boldsymbol { v } ( t ) \ \in \ \mathbb { R } ^ { N _ { v } }$ and $N _ { h }$ hidden neurons $h ( t ) \in \mathbb { R } ^ { N _ { h } }$ , and their interactions are represented by $\xi ^ { ( \dot { h } , v ) } \ \in \ \mathbb { R } ^ { \dot { N _ { h } } \times N _ { \tau } }$ and $ { \boldsymbol { \xi } } ^ { ( v , h ) } \in \mathbb { R } ^ { N _ { v } \times N _ { h } }$ , with the constraint $\xi ^ { ( v , h ) } = ( \xi ^ { ( h , v ) } ) ^ { \top }$ , see $\mathrm { F i g . 1 . }$ . The dynamics of the system is governed by the “Lagrangians” $L _ { v } : \mathbb { R } ^ { N _ { v } }  \mathbb { R }$ and $L _ { h } \colon \mathbb { R } ^ { \smile _ { h } } \to \mathbb { R }$ , which determine the activation functions of the neurons as their gradients,

$$
f = \nabla L _ { h } , \qquad g = \nabla L _ { v } .\tag{1}
$$

The dynamical equations of the system and the energy function are given by

$$
\tau _ { v } \frac { d v ( t ) } { d t } = \frac { \lambda } { \tau _ { h } } \xi ^ { ( v , h ) } f ( h ( t ) ) - v ( t ) ,\tag{2}
$$

$$
\tau _ { h } \frac { d h ( t ) } { d t } = \frac { \lambda } { \tau _ { v } } \xi ^ { ( h , v ) } g ( v ( t ) ) - h ( t ) ,\tag{3}
$$

![](images/d18402cfe1e24060bcbb8cb48350cff4a5482b2c0e4d66aa1f5ec2c4ce3d3511.jpg)  
FIG. 1: The class-H associative memories. The $N _ { v }$ visible neurons and the $N _ { h }$ hidden neurons form a bipartite network with no intralayer connections.

and

$$
\begin{array} { l } { { \displaystyle E _ { \xi } ( v , h ) = \frac { 1 } { \tau _ { v } } \big ( v ^ { \top } g ( v ) - { \cal L } _ { v } ( v ) \big ) } \ ~ } \\ { { \displaystyle ~ + \frac { 1 } { \tau _ { h } } \big ( h ^ { \top } f ( h ) - { \cal L } _ { h } ( h ) \big ) - \frac { \lambda } { \tau _ { h } \tau _ { v } } f ( h ) ^ { \top } \xi ^ { ( h , v ) } g ( v ) } \ ~ } \\ { { \displaystyle ~ = : \frac { 1 } { \tau _ { v } } E _ { v } ( v ) + \frac { 1 } { \tau _ { h } } E _ { h } ( h ) + \frac { \lambda } { \tau _ { h } \tau _ { v } } E _ { \mathrm { i n t } } ( v , h ) } , } \end{array}\tag{4}
$$

where $\tau _ { v }$ and $\tau _ { h }$ are the relaxation time constants of the visible and hidden neurons, respectively, and λ is a coupling constant. The combinations $v ^ { \top } g ( v ) - L _ { v } ( v )$ and $\tilde { h ^ { \top } } f ( \tilde { h } ) - L _ { h } ( h )$ are the Legendre transforms of the Lagrangians evaluated at the activations $g ( v )$ and $f ( h )$ In this sense, $E _ { \xi }$ is the “Hamiltonian” associated with the pair of Lagrangians. In fact, the dynamical equations above can be viewed as one half of Hamilton’s canonical equations restricted to the Legendre constraint surface, which makes the dynamics dissipative, see Appendix A.

Provided that the Hessians of the Lagrangians are positive (semi-)definite, this energy function monotonically decreases along the solution trajectory of the dynamical equations,

$$
\frac { d E _ { \xi } ( v ( t ) , h ( t ) ) } { d t } \leq 0 .\tag{5}
$$

If, in addition, the overall energy function is bounded from below, the trajectory is guaranteed to converge to a fixed-point attractor state, which corresponds to one of the local minima of the energy function. Such fixed points may be identified with the stored memories, and the convergence toward them with memory retrieval.

In the adiabatic limit, $\tau _ { v } \gg \tau _ { h }$ , the hidden neurons relax much faster than the visible ones and can be adiabatically eliminated: taking $\tau _ { h }  0$ in the dynamical equations, we obtain

$$
h ( t ) = \frac { \lambda } { \tau _ { v } } \xi ^ { ( h , v ) } g ( v ( t ) ) ,\tag{6}
$$

so that the hidden neurons instantaneously follow the visible configuration, and their states $h _ { \mu } ( t )$ measure the overlap between the activation of the visible neurons and the patterns $\xi _ { \mu } ^ { ( h , v ) }$ . In this sense, each hidden neuron acts as a feature detector for the corresponding pattern, and the hidden neurons serve as the order parameter of memory retrieval in the system [19].

## B. Partition function

To consider the statistical mechanics of the class H with the energy function Eq. (4), here we introduce a formal partition function in a general setup:

$$
Z _ { \xi } ( \beta ) = \int d v d h \exp ( - \beta E _ { \xi } ( v , h ) ) ,\tag{7}
$$

where $\beta$ is the inverse temperature. This expression is formal: for certain choices of Lagrangians, the energy function has flat directions along which the integral diverges $( \mathrm { e . g . } ,$ , for $L _ { v }$ homogeneous of degree one, $E _ { v }$ vanishes identically and the energy is independent of the overall scale of v), and thus the precise partition function, including the integration domain and measure, is defined for each model in the corresponding sections below.

We can write this partition function as

$$
\begin{array} { l } { { \displaystyle Z _ { \xi } ( \beta ) = \int d v e ^ { - \frac { \beta } { \tau _ { v } } E _ { v } ( v ) } } \ ~ } \\ { { \displaystyle ~ \times \int d h \exp \Biggl \{ - \frac { \beta } { \tau _ { h } } \biggl ( E _ { h } ( h ) + \frac { \lambda } { \tau _ { v } } E _ { \mathrm { i n t } } ( v , h ) \biggr ) \Biggr \} } . } \end{array}\tag{8}
$$

In this form, the adiabatic limit is rephrased as $\beta / \tau _ { h }  \infty$ in which the thermal fluctuations of the hidden neurons are suppressed. By the saddle-point approximation, the h integral then localizes at the stationary point of the integrand, which is given by

$$
h _ { * } = \frac { \lambda } { \tau _ { v } } \xi ^ { ( h , v ) } g ( v ) .\tag{9}
$$

Thus, in the adiabatic limit, the partition function becomes

$$
\begin{array} { l } { { \displaystyle Z _ { \xi } ( \beta ) } } \\ { { \displaystyle \approx \int d v \exp \Biggl \{ - \beta \biggl ( \frac 1 { { \tau _ { v } } } E _ { v } ( v ) - \frac { 1 } { { \tau _ { h } } } L _ { h } ( h _ { * } ) \biggr ) \Biggl \} \biggl ( \frac { 2 \pi } { \beta / { \tau _ { h } } } \biggr ) ^ { N _ { h } / 2 } } } \\ { { \displaystyle = \biggl ( \frac { 2 \pi } { \beta / { \tau _ { h } } } \biggr ) ^ { N _ { h } / 2 } \int d v \exp ( - \beta E _ { \xi } ( v , h _ { * } ) ) . } } \end{array}\tag{10}
$$

Strictly, the Gaussian prefactor also carries the factor (det Hess $L _ { h } ( h _ { * } ) ) ^ { - 1 / 2 }$ , which we suppress at this leading order. Its role is examined for Model B in Appendix C 2.

As discussed in this section, the hidden neurons play the role of the order parameter of memory retrieval in the system. In the following sections, we investigate the properties of Models A, B, and C from the viewpoint of the role of the hidden neurons.

## III. MODELS A AND C

Models A and C are members of a family of models within the class H. Let us consider a family of Lagrangians,<sup>2</sup>

$$
L _ { v } ( v ) = \| v \| _ { p } , \qquad L _ { h } ( h ) = \sum _ { \mu } F ( h _ { \mu } ) ,\tag{11}
$$

where $1 \leq p \leq \infty$ . The corresponding activation functions are

$$
g _ { i } ( v ) = { \frac { \mathrm { s g n } ( v _ { i } ) | v _ { i } | ^ { p - 1 } } { \| v \| _ { p } ^ { p - 1 } } } , \qquad f _ { \mu } ( h ) = F ^ { \prime } ( h _ { \mu } ) .\tag{12}
$$

Since $v ^ { \top } g ( v ) - L _ { v } = 0$ for these Lagrangians, the bare visible neurons vanish in the energy function, that is, the energy depends on v only through the activation $g \colon$

$$
\begin{array} { c } { { E _ { \xi } ( v , h ) = \displaystyle \frac { 1 } { \tau _ { h } } \sum _ { \mu } ( h _ { \mu } F ^ { \prime } ( h _ { \mu } ) - F ( h _ { \mu } ) ) } } \\ { { - \displaystyle \frac { \lambda } { \tau _ { h } \tau _ { v } } \sum _ { \mu , i } F ^ { \prime } ( h _ { \mu } ) \xi _ { \mu i } ^ { ( h , v ) } g _ { i } ( v ) . } } \end{array}\tag{13}
$$

The activation of the visible neurons satisfies $\| g ( v ) \| _ { p ^ { \prime } } =$ 1, where $p ^ { \prime }$ is the Hölder conjugate of p defined by $1 / p +$ $1 / p ^ { \prime } = 1$ . Since the energy is independent of the overall scale of v (cf. Sec. II B), the partition function for this family should be defined with the visible integral restricted to a sphere:

$$
\begin{array} { c l c r } { { \displaystyle Z _ { \xi } ( \beta ) = \int _ { \mathbb { R } ^ { N _ { h } } } d h e ^ { - \frac { \beta } { \tau _ { h } } \sum _ { \mu } \left( h _ { \mu } F ^ { \prime } ( h _ { \mu } ) - F ( h _ { \mu } ) \right) } } } \\ { { \displaystyle ~ \times \int _ { S } d \Omega ( { \bf x } ) \exp \left\{ \frac { \beta \lambda } { \tau _ { h } \tau _ { v } } \sum _ { \mu , i } F ^ { \prime } ( h _ { \mu } ) \xi _ { \mu i } ^ { ( h , v ) } { \bf x } _ { i } \right\} , } } \end{array}\tag{14}
$$

where

$$
S = \mathbb { S } _ { p ^ { \prime } } ^ { N _ { v } - 1 } ( N _ { v } ^ { 1 / p ^ { \prime } } ) : = \Big \{ { \mathbf { x } } \in \mathbb { R } ^ { N _ { v } } \Big | \left\| { \mathbf { x } } \right\| _ { p ^ { \prime } } = N _ { v } ^ { 1 / p ^ { \prime } } \Big \} ,\tag{15}
$$

and dΩ is the standard measure on this sphere for $1 < p <$ ∞, while for $p = 1$ , ∞ the integral reduces to a discrete sum. The radius of S is chosen such that the components are normalized as $\mathrm { x } _ { i } = O ( 1 )$ : in particular, S is the sphere of radius $\sqrt { N _ { v } }$ for $p = 2 ,$ and the integral reduces to the sum over $\mathrm { x } \in \{ \pm 1 \} ^ { N _ { v } }$ for $p = 1$

Models A and C correspond to the cases $p = 1$ and $p = 2$ , respectively. In what follows, we study this family with the hidden Lagrangian specified by ${ \dot { F } } ( x ) = x ^ { k } / { k }$ with a positive even integer $k . ^ { 3 }$ The replica method provides a powerful tool for this purpose. Since the hidden neurons represent the order parameters of memory retrieval in these systems, we first integrate out the visible neurons and then perform the quenched average over the random patterns. We focus on the replica symmetric (RS) solutions of these models.

## A. Model A

The model-A energy function is given by

$$
E _ { \xi } ^ { \mathrm { A } } ( v , h ) = N _ { v } \frac { k - 1 } { \tau _ { h } k } \sum _ { \mu } h _ { \mu } ^ { k } - \frac { \lambda } { \tau _ { h } \tau _ { v } } \sum _ { \mu , i } h _ { \mu } ^ { k - 1 } \xi _ { \mu i } ^ { ( h , v ) } \operatorname { s g n } ( v _ { i } ) ,\tag{16}
$$

where the factor $N _ { v }$ in the first term is introduced for the extensivity of the energy function. This modification is equivalent to the renormalizations $\tau _ { h }  \tau _ { h } / N _ { v }$ and $\lambda  \lambda / N _ { v }$ , which fix the correct normalization of h and do not afect any other property of Model A as an associative memory. From Eq. (14), the partition function for this model reads

$$
\begin{array} { l } { { \displaystyle Z _ { \xi } ^ { \mathrm { A } } ( \beta ) = \int d h \exp \bigg \{ - N _ { v } \gamma _ { k } \sum _ { \mu } m _ { \mu } ^ { k } } } \\ { { \displaystyle ~ + \sum _ { i = 1 } ^ { N _ { v } } \log \bigg ( 2 \cosh \bigg [ \beta _ { k } \sum _ { \mu } \xi _ { i \mu } ^ { ( v , h ) } m _ { \mu } ^ { k - 1 } \bigg ] \bigg ) \bigg \} , } } \end{array}\tag{17}
$$

where

$$
\gamma _ { k } = \frac { k - 1 } { k } \beta _ { k } , \quad \beta _ { k } = \frac { \beta } { \tau _ { h } } \biggl ( \frac { \lambda } { \tau _ { v } } \biggr ) ^ { k } , \quad m _ { \mu } = \frac { h _ { \mu } } { \lambda / \tau _ { v } } ,\tag{18}
$$

and we dropped an irrelevant overall constant. Note that the stationarity of Eq. (17) with respect to $m _ { \mu }$ constrains $m _ { \mu }$ to be the thermal average of the overlap between the pattern $\xi _ { \mu } ^ { ( v , h ) }$ and the visible spins $\operatorname { s g n } ( v _ { i } )$ , in accordance with the general role of the hidden neurons discussed in Sec. II A.

We compute the quenched free energy per visible neuron,

$$
f ^ { \mathrm { A } } ( \beta ) = - \operatorname* { l i m } _ { N _ { v } \to \infty } \frac { 1 } { \beta N _ { v } } \mathbb { E } _ { \xi } \big [ \log Z _ { \xi } ^ { \mathrm { A } } ( \beta ) \big ] ,\tag{19}
$$

by the replica method, with the patterns drawn independently as $\stackrel { \cdot } { \xi } _ { \mu i } ^ { ( h , v ) } = \pm 1$ with equal probability. We work in the high-load regime of dense associative memories [7–9],

$$
N _ { h } = \alpha _ { k } N _ { v } ^ { k - 1 } ,\tag{20}
$$

and consider retrieval states in which a single pattern is condensed, $m _ { 1 } = m = { \cal { O } } ( 1 )$ , while the remaining overlaps are of order $N _ { v } ^ { - 1 / 2 }$ . For $k > 2$ , the analysis is performed in the adiabatic limit $\beta / \tau _ { h }  \infty$ at fixed $\beta _ { k } .$ , in which the hidden neurons are enslaved to the visible configuration. The details of the replica computation are presented in Appendix B 2.

Under the RS ansatz, we eventually obtain the free energy, up to an additive constant independent of the order parameters,

$$
\begin{array} { l } { { \displaystyle \beta f ^ { \mathrm { A } } = \gamma _ { k } m ^ { k } + \frac { \alpha _ { k } \beta _ { k } ^ { 2 } } { 2 } r ( 1 - q ) + \frac { \alpha _ { k } } { 2 } \Psi _ { k } ( q ) } } \\ { { \displaystyle ~ - \int D z \log 2 \cosh \beta _ { k } \big ( m ^ { k - 1 } + \sqrt { \alpha _ { k } r } z \big ) , } } \end{array}\tag{21}
$$

where $D z : = d z e ^ { - z ^ { 2 } / 2 } / \sqrt { 2 \pi }$ is the standard Gaussian measure, q is the Edwards–Anderson order parameter of the visible spins, r measures the strength of the crosstalk noise from the non-condensed patterns, and the noise entropic term reads

$$
\Psi _ { k } ( q ) = \left\{ \begin{array} { l l } { \displaystyle \log [ 1 - \beta _ { k } ( 1 - q ) ] - \frac { \beta _ { k } q } { 1 - \beta _ { k } ( 1 - q ) } , } & { k = 2 , } \\ { \displaystyle \beta _ { k } ^ { 2 } \int _ { 0 } ^ { q } \mathcal { M } _ { k } ( s ) d s , } & { k > 2 , } \end{array} \right.\tag{22}
$$

with $\mathcal { M } _ { k }$ defined in Eq. (26) below. The equations of state for the order parameters follow from the stationarity of the free energy. We then obtain

$$
m = \int D z \operatorname { t a n h } { \beta _ { k } } { \left( m ^ { k - 1 } + { \sqrt { \alpha _ { k } r } } z \right) } ,\tag{23}
$$

$$
q = \int D z \operatorname { t a n h } ^ { 2 } \beta _ { k } \big ( m ^ { k - 1 } + \sqrt { \alpha _ { k } r } z \big ) ,
$$

$$
r = \mathcal { M } _ { k } ( q ) ,\tag{24}
$$

(25)

where

$$
\mathcal { M } _ { k } ( q ) = \left\{ \begin{array} { l l } { \displaystyle q } \\ { \left( 1 - \beta _ { k } ( 1 - q ) \right) ^ { 2 } } & { k = 2 , } \\ { \mathbb { E } \left[ X ^ { k - 1 } Y ^ { k - 1 } \right] , } & { k > 2 , } \end{array} \right.\tag{26}
$$

and $( X , Y )$ in the second line denotes a pair of standard Gaussian variables with correlation $\mathbb { E } [ X Y ] = q .$ . For $k >$ $2 , \mathcal { M } _ { k } ( q )$ is a polynomial in q with positive coeficients, $\mathrm { e . g . } , \mathcal { M } _ { 4 } ( q ) = 9 q + 6 q ^ { 3 }$ , and $\mathcal { M } _ { k } ( 1 ) = ( 2 k - 3 ) ! !$

The two cases in $\operatorname { E q . }$ (26) reflect the fate of the Onsager reaction field. Relative to its bare central-limit scale, the feedback of a non-condensed mode onto itself is of order $\beta _ { k } ( 1 - q ) N _ { v } ^ { - ( k - 2 ) / 2 }$ . For $k = 2$ this feedback is marginal and must be resummed to all orders, which produces the denominator $1 - \beta _ { k } ( 1 - q )$ of the classical result by Amit, Gutfreund, and Sompolinsky (AGS) [2, 3]. For $k > 2$ it vanishes in the thermodynamic limit, so that the noncondensed overlaps behave as bare Gaussian variables. A cavity derivation of this dichotomy, together with the closed polynomial form of $\mathcal { M } _ { k }$ , is given in Appendix B 2 e.

Zero-temperature limit and capacity condition. Sending $\beta _ { k } \to \infty ,$ the overlap satisfies $q \to 1$ while the combination $C : = \beta _ { k } ( 1 - q )$ remains finite, and the equations of state close in the pair $( m , C )$ . The elementary evaluation is given in Appendix $\mathrm { B 2 e . }$ . The retrieval overlap obeys

$$
m = \mathrm { e r f } \left( { \frac { m ^ { k - 1 } } { \sqrt { 2 \alpha _ { k } r } } } \right) ,\tag{27}
$$

the frozen response is

$$
C = \sqrt { \frac { 2 } { \pi \alpha _ { k } r } } \exp \biggl ( { - \frac { m ^ { 2 k - 2 } } { 2 \alpha _ { k } r } } \biggr ) ,\tag{28}
$$

and the crosstalk moment Eq. (26) becomes

$$
r = \left\{ \begin{array} { l l } { ( 1 - C ) ^ { - 2 } , } & { k = 2 , } \\ { ( 2 k - 3 ) ! ! , } & { k > 2 , } \end{array} \right.\tag{29}
$$

where the former reproduces the classical AGS zerotemperature equations [2, 3], and the latter is the $2 ( k - 1 )$ th moment of the standard Gaussian.

In terms of the signal-to-noise ratio $t : = m ^ { k - 1 } / \sqrt { 2 \alpha _ { k } r }$ any retrieval solution with $m > 0$ lies on the parametric curve

$$
\alpha _ { k } ( t ) = \frac { m ( t ) ^ { 2 ( k - 1 ) } } { 2 t ^ { 2 } r ( t ) } , \qquad m ( t ) = \mathrm { e r f } ( t ) ,\tag{30}
$$

where $r ( t ) = ( 2 k - 3 ) ! !$ for $k > 2 ,$ , while for $k = 2$ the reaction field enters through $r ( t ) = [ 1 - C ( t ) ] ^ { - 2 }$ with $C ( t ) = 2 t e ^ { - t ^ { 2 } } / [ \sqrt { \pi } m ( t ) ]$ . Since $\alpha _ { k } ( t )  0$ both as $t \to 0$ and as $t \to \infty$ , the curve attains an interior maximum, and retrieval solutions exist if and only if

$$
\alpha _ { k } \leq \alpha _ { c } : = \operatorname* { m a x } _ { t > 0 } \alpha _ { k } ( t ) ,\tag{31}
$$

where the critical load $\alpha _ { c }$ marks a spinodal at which the stable and unstable retrieval branches merge. For $k = 2$ the maximum occurs at $t _ { c } \simeq 1 . 5 1$ , yielding $\alpha _ { c } \simeq 0 . 1 3 8$ and $m _ { c } \simeq 0 . 9 7$ , in agreement with the classical Hopfield result [2, 3]. For $k > 2$ , the capacity takes the closed form

$$
\alpha _ { c } = \operatorname* { m a x } _ { t > 0 } \frac { [ \mathrm { e r f } ( t ) ] ^ { 2 ( k - 1 ) } } { 2 t ^ { 2 } ( 2 k - 3 ) ! ! } ,\tag{32}
$$

which gives $\alpha _ { c } \simeq 1 . 3 2 \times 1 0 ^ { - 2 }$ for $k = 4$ and $\alpha _ { c } \simeq 1 . 6 7 \times$ $1 0 ^ { - 4 } \mathrm { f o r } k = 6 $ . Note that the roles of the reaction field are opposite in the two cases: for $k = 2 .$ , neglecting it $( C \to 0$ $r  1 )$ would overestimate the capacity by more than a factor of four, giving $2 / \pi \simeq 0 . 6 4$ , whereas for $k > 2$ the reaction-free expression $\mathrm { E q . ~ ( 3 2 ) }$ is not an approximation but the leading-order RS result. The double factorial here is precisely the combinatorial factor that controls the error-free capacity estimate of the dense associative memory [10]. For large $k ,$ the maximum is attained at $t _ { c } \simeq \sqrt { \ln k }$ , so that

$$
\alpha _ { c } \sim \frac { O ( 1 ) } { 2 t _ { c } ^ { 2 } ( 2 k - 3 ) ! ! } .\tag{33}
$$

TABLE I: Critical values of Model A obtained from the RS equations of state: the critical temperature $T _ { c }$ at $\alpha _ { k } = 0$ , the zero-temperature critical load $\alpha _ { c } ,$ and the resulting steepness $T _ { c } / \alpha _ { c }$ of the retrieval boundary.
<table><tr><td> $k$ </td><td> $T _ { c }$ </td><td> $\alpha _ { c }$ </td><td> $T _ { c } / \alpha _ { c }$ </td></tr><tr><td>2</td><td>1</td><td>0.138</td><td>7.2</td></tr><tr><td>4</td><td>0.496</td><td> $1 . 3 2 \times 1 0 ^ { - 2 }$ </td><td>38</td></tr><tr><td>6</td><td>0.423</td><td> $1 . 6 7 \times 1 0 ^ { - 4 }$ </td><td> $2 . 5 \times 1 0 ^ { 3 }$ </td></tr></table>

The signal factor $m _ { c } ^ { 2 ( k - 1 ) }$ remains $O ( 1 )$ , and the collapse of the capacity is driven by the factorial growth of the crosstalk variance.

Critical temperature at $\alpha _ { k } = 0$ and steepening of the boundary. At $\alpha _ { k } = 0$ the crosstalk noise vanishes and the retrieval overlap satisfies

$$
m = \operatorname { t a n h } \left( \beta _ { k } m ^ { k - 1 } \right) ,\tag{34}
$$

independently of $^ r .$ The corresponding critical point is determined by the spinodal condition

$$
1 = \beta _ { k } ( k - 1 ) m ^ { k - 2 } \operatorname { s e c h } ^ { 2 } \bigl ( \beta _ { k } m ^ { k - 1 } \bigr ) ,\tag{35}
$$

and the critical value $T _ { c }$ of the efective temperature $T : =$ $\beta _ { k } ^ { - 1 }$ decreases with k only slowly (see Table I). By contrast, the zero-temperature capacity collapses factorially with $k ,$ so the retrieval boundary connecting $( \alpha _ { k } , T ) = ( 0 , T _ { c } )$ to $( \alpha _ { c } , 0 )$ steepens rapidly:

$$
\left| \frac { d T } { d \alpha _ { k } } \right| \sim \frac { T _ { c } } { \alpha _ { c } } \sim ( 2 k - 3 ) ! !\tag{36}
$$

up to factors varying slowly with $k ,$ and the boundary becomes nearly vertical as k grows.

Physically, the k-body interaction sharpens the retrieval landscape through the signal term $m ^ { k - 1 }$ , while the crosstalk from the non-condensed patterns enters through the $2 ( k - 1 )$ )-th moment of their Gaussian overlaps. Rare large fluctuations dominate this moment, so the cost of storing one additional pattern grows factorially with k. The model is thus robust against thermal noise at $\alpha _ { k } = 0$ but fragile against pattern loading at $\alpha _ { k } > 0$ , an asymmetry that becomes extreme for large k.

Phase diagram. The complete RS phase diagram in the $( \alpha _ { k } , T )$ plane is assembled in Fig. 2. Besides the retrieval states, the equations of state admit an $m = 0$ solution with $q > 0 ;$ , a spin glass sustained by the crosstalk noise alone. Linearizing Eq. (24) and Eq. (25) at $m = 0$ and small q, where $q \simeq \beta _ { k } ^ { 2 } \alpha _ { k } \mathcal { M } _ { k } ( q )$ and $\bar { \mathcal { M } } _ { k } ( q ) \simeq [ ( k - 1 ) ! ! ] ^ { 2 } q$ for $k > 2$ , shows that this solution bifurcates continuously from the paramagnet at

$$
T _ { g } ( \alpha _ { k } ) = \left\{ \begin{array} { l l } { 1 + \sqrt { \alpha _ { k } } , } & { k = 2 , } \\ { ( k - 1 ) ! ! \sqrt { \alpha _ { k } } , } & { k > 2 , } \end{array} \right.\tag{37}
$$

where the $k = 2$ expression follows from the same expansion with the resummed moment $\mathcal { M } _ { 2 }$ and reproduces the T(α): retrieval spinodal (existence edge) ……... Tg(α): RS spin-glass line (continuous)

![](images/124b11131658dcb13591667314ba0fba89f9f5252d78038d3e80fbe398d6cdba.jpg)

![](images/ab82222af3804d56b887d58fff9ab8ba243fa72e734fd2093ea63c54c3224df7.jpg)

![](images/75e4d5b29de36f2ecbe45fdfb4441d63e630ca9d1e2fa673e63d14dfe5a90afe.jpg)  
FIG. 2: RS phase diagram of Model A in the $( \alpha _ { k } , T )$ plane for (a) $k = 2 ,$ (b) $k = 4$ , and (c) $k = 6 .$ , where $T = 1 / \beta _ { k }$ is the efective temperature. In the retrieval phase (R) the retrieval states are the global minima of the free energy. In the region M they persist only as metastable states. The red solid line is the retrieval spinodal $T _ { R } ( \alpha _ { k } )$ , beyond which no retrieval solution of Eq. (23)–Eq. (25) exists. The black solid line is the first-order boundary $T _ { M } ( \alpha _ { k } )$ , on which the retrieval free energy crosses that of the $m = 0$ branch. The dotted line is the continuous spin-glass transition $T _ { g } ( \alpha _ { k } )$ of Eq. (37), separating the paramagnet (P) from the RS spin glass (SG). For $k > 2$ the first-order boundary is reentrant, extending to larger loads at intermediate temperatures than at $T = 0$

AGS spin-glass line [4]. Retrieval solutions exist below the spinodal $T _ { R } ( \alpha _ { k } )$ , which connects $( 0 , T _ { c } )$ to $( \alpha _ { c } , 0 )$ They are the global minima of the free energy only below the first-order boundary $T _ { M } ( \alpha _ { k } )$ , obtained by equating the retrieval free energy Eq. (21) with that of the $m = 0$ branch (the spin glass for $T < T _ { g }$ and the paramagnet above). Between the two lines retrieval survives as a metastable state. For $k = 2$ the transition at $\alpha _ { k } = 0$ is continuous, so the metastable band opens only at finite load and closes at $T = 0$ between $\alpha _ { M } \simeq 0 . 0 5 1$ and $\alpha _ { c } \simeq 0 . 1 3 8$ [4]. For $k > 2$ the transition is first order already at $\alpha _ { k } = 0 .$ , and the band persists down to zero load, where retrieval is metastable against the paramagnet for $T _ { M } ( 0 ) < T < T _ { c }$ . The distinct competitor at small load reflects a hierarchy of local stability: a k-body coupling exerts no mean field on a disordered configuration (the local field involves a product of k − 1 spins), so for $k > 2$ the paramagnet remains locally stable at every temperature, as in the purely ferromagnetic k-spin model, and only the accumulated crosstalk of many patterns, with the small-q variance $\alpha _ { k } [ ( k - 1 ) ! ! ] ^ { 2 } q$ underlying Eq. (37), can freeze the $m = 0$ sector. Since $T _ { g } \to 0$ as $\alpha _ { k }  0$ at small load retrieval competes directly with the paramagnet on a glass-free background, and the boundaries admit simple estimates: balancing the retrieval energy densit $r - 1 / k$ against the paramagnetic entropy log 2, and against the zero-temperature spin-glass energy density $- \sqrt { 2 \alpha _ { k } ( 2 k - 3 ) ! ! / \pi }$ , yields

$$
T _ { M } ( 0 ) = \frac 1 { k \log 2 } , \qquad \alpha _ { M } ( 0 ) \simeq \frac { \pi } { 2 k ^ { 2 } ( 2 k - 3 ) ! ! } ,\tag{38}
$$

where the former holds up to corrections exponentially small in k, and both reproduce the numerical boundaries of Fig. 2 within a few percent. The factorial steepening seen in Table I is directly visible in the figure: as k grows, the retrieval and metastable regions collapse toward the temperature axis, and the spin-glass phase comes to dominate the diagram.

Reentrance of the first-order boundary. For $k > 2$ the first-order boundary in Fig. 2 is visibly reentrant: $\alpha _ { M } ( T )$ exceeds its zero-temperature value at intermediate temperatures, by about 20% for $k = 4$ and 54% for $k = 6$ . The mechanism is the entropic asymmetry between the competing states. In the reentrant window of loads the spin glass has the lower free energy at $T = 0$ but upon heating the $m = 0$ background melts into the paramagnet already at $T _ { g } \propto \sqrt { \alpha _ { k } }$ , whereas the retrieval state, protected by the large local field $m ^ { k - 1 }$ , retains an exponentially small entropy and remains efectively frozen. Retrieval thereby recovers the global minimum in an intermediate window of temperatures, which closes on the scale $T _ { M } ( 0 )$ of Eq. (38). Consistently, the reentrance is nearly absent for $k = 2$ , where $T _ { g } = 1 + \sqrt { \alpha _ { k } }$ holds the background frozen throughout the retrieval region. All lines are computed within the RS ansatz. The spinglass phase and the low-temperature boundaries acquire corrections from replica symmetry breaking (RSB) [23], which are known to be small for the retrieval boundaries at $k = 2$ [4] and are expected to remain so for $k > 2 ~ [ 7 ]$ In particular, the far weaker low-temperature reentrance of the spinodal $T _ { R }$ (below four percent for the values of k shown) is the familiar pathology of the RS solution, which RSB removes for $k = 2$ while raising the capacity slightly to $\alpha _ { c } \simeq 0 . 1 4 4 \ : [ 2 4 ]$ . The reentrance of $T _ { M }$ instead operates at intermediate temperatures through the melting of the background, although its magnitude may shift under RSB.

## B. Model C

Model C is the $p = 2$ member of the family, called the spherical memory model in [19]: the visible degrees of freedom are continuous variables on the sphere $S =$ $\mathbb { S } ^ { N _ { v } - 1 } ( \sqrt { N _ { v } } )$ . The model-C energy function is

$$
E _ { \xi } ^ { \mathrm { C } } ( v , h ) = N _ { v } \frac { k - 1 } { \tau _ { h } k } \sum _ { \mu } h _ { \mu } ^ { k } - \frac { \lambda } { \tau _ { h } \tau _ { v } } \sum _ { \mu , i } h _ { \mu } ^ { k - 1 } \xi _ { \mu i } ^ { ( h , v ) } \mathrm { x } _ { i } ,\tag{39}
$$

where the factor $N _ { v }$ in the first term is the same extensivity insertion as in Model A. The partition function then reads

$$
\begin{array} { l } { { \displaystyle Z _ { \xi } ^ { \mathrm { C } } ( \beta ) = \int d h e ^ { - N _ { v } \gamma _ { k } } \sum _ { \mu } m _ { \mu } ^ { k } } } \\ { { \displaystyle \qquad \times \int _ { S } d \Omega ( { \bf x } ) \exp \Biggl \{ \beta _ { k } \sum _ { \mu , i } m _ { \mu } ^ { k - 1 } \xi _ { \mu i } ^ { ( h , v ) } { } _ { { \bf x } _ { i } } \Biggr \} } , }  \end{array}\tag{40}
$$

with the same $\gamma _ { k } , \beta _ { k }$ , and $m _ { \mu }$ as in Model A. The only change relative to Eq. (17) is that the visible trace runs over the sphere S instead of the hypercube $\left\{ \pm 1 \right\} ^ { N _ { v } }$ . In particular, the stationarity with respect to $m _ { \mu }$ again identifies $m _ { \mu }$ with the overlap between the pattern $\xi _ { \mu } ^ { \bar { ( h , v ) } }$ and the visible configuration x.

The patterns are now drawn independently and uniformly from the same sphere, $\xi _ { \mu } ^ { ( h , v ) } \sim \operatorname { U n i f } ( \bar { S } )$ , and we work in the same high-load regime $N _ { h } = \alpha _ { k } \dot { N } _ { v } ^ { k - 1 }$ with a single condensed pattern. By the rotational invariance of the pattern ensemble and of the visible measure, the condensed pattern can be rotated to $\xi _ { 1 } ^ { ( h , v ) } = ( 1 , \cdot \cdot \cdot , 1 )$ (the continuous analogue of the gauge choice for binary patterns) exactly at finite $N _ { v }$ , while the remaining patterns are asymptotically Gaussian, with fixed-norm corrections that do not afect the RS equations. For $k > 2$ , the analysis is again performed in the adiabatic limit $\beta / \tau _ { h }  \infty$ at fixed $\beta _ { k }$ . The details of the replica computation are presented in Appendix B 3.

In contrast to Model A, the visible trace is a Gaussian integral on the sphere and can be carried out exactly, so no single-site integral survives in the final expressions. Under the RS ansatz, after eliminating the condensed hidden mode and the Lagrange multiplier of the spherical constraint at their saddle points, we obtain the free energy, up to an additive constant independent of the order parameters,

$$
\begin{array} { l } { { \displaystyle \beta f ^ { \mathrm { C } } = - \frac { \beta _ { k } } { k } m ^ { k } + \frac { \alpha _ { k } } { 2 } \Psi _ { k } ( q ) } } \\ { { \displaystyle ~ - \frac { 1 } { 2 } \biggl \{ \log ( 1 - q ) + \frac { q - m ^ { 2 } } { 1 - q } \biggr \} } , } \end{array}\tag{41}
$$

where m is the overlap between the visible configuration and the condensed pattern, q is the Edwards–Anderson order parameter of the spherical visible state, and $\Psi _ { k }$ is the same noise entropic term as in Eq. (21). The equations of state follow from the stationarity of the free energy:

$$
0 = \big ( 1 - \beta _ { k } ( 1 - q ) m ^ { k - 2 } \big ) m ,\tag{42}
$$

$$
\frac { q - m ^ { 2 } } { ( 1 - q ) ^ { 2 } } = \alpha _ { k } \beta _ { k } ^ { 2 } { \mathcal { M } } _ { k } ( q ) ,\tag{43}
$$

where we used $\Psi _ { k } ^ { \prime } ( q ) = \beta _ { k } ^ { 2 } \mathcal { M } _ { k } ( q )$ , which holds uniformly in $k ,$ with the same crosstalk moment $\mathcal { M } _ { k }$ as in $\mathrm { E q . ~ ( 2 6 ) }$ The crosstalk strength $r = \mathcal { M } _ { k } ( q )$ of Eq. (25) thus carries over unchanged. In the gauge $\xi _ { 1 } ^ { ( h , v ) } = ( 1 , \cdot \cdot \cdot , 1 )$ , the RS saddle point gives $\left. \mathbf { x } _ { i } \right. = m$ and $\begin{array} { r } { \frac { 1 } { N _ { v } } \sum _ { i } \left. \mathbf { x } _ { i } \right. ^ { 2 } = q . } \end{array}$ . Since the spherical constraint fixes $\begin{array} { r } { \frac { 1 } { N _ { v } } \sum _ { i } \bigl \langle \mathbf { x } _ { i } ^ { 2 } \bigr \rangle = 1 } \end{array}$ , the combination $\beta _ { k } ( 1 - q )$ is the static susceptibility of the spherical state. The crosstalk moment is the same as in Model A because the non-condensed overlaps $N _ { v } ^ { - 1 / 2 } \sum _ { i } \xi _ { \mu i } ^ { ( h , v ) } \mathrm { x } _ { i }$ obey the same central-limit statistics, governed solely by $q \colon$ the crosstalk is universal across the visible ensembles, and the dichotomy of the Onsager reaction field discussed below Eq. (26) applies without modification. The signal equation Eq. (42), in turn, is of the form familiar from the ferromagnetically biased spherical p-spin model [25], the k-body signal competing with the entropy of the sphere. The case $k = 2$ . For $k = 2$ the signal equation degenerates: a retrieval solution $m > 0$ requires $\beta _ { k } ( 1 - q ) = 1$ which is precisely the marginality condition at which the AGS denominator $1 - \beta _ { k } ( 1 - q )$ of $\mathcal { M } _ { 2 }$ vanishes. The right-hand side of $\operatorname { E q }$ . (43) then diverges, so that no RS retrieval solution exists at any $\alpha _ { k } > 0$ , including $T = 0 \mathrm { : }$ the quadratic spherical model has zero storage capacity. This is the known marginality of the spherical Hopfield model [20], in which a retrieval phase is recovered only after the energy function is stabilized by an additional quartic term. At $\alpha _ { k } = 0 ;$ , Eq. (43) gives $q = m ^ { 2 }$ , and the signal equation yields $m ^ { 2 } = 1 - T$ : retrieval sets in continuously below $\dot { T } _ { c } = 1$   
Zero-temperature capacity for $k \ > \ 2 .$ Sending $\beta _ { k } \ \to \ \infty$ with $C = \beta _ { k } ( 1 - q )$ finite, as in Model A, the signal equation gives $C = m ^ { 2 - k }$ , while $q  1$ and $\mathcal { M } _ { k } ( 1 ) = ( 2 k - 3 ) ! !$ . The noise equation Eq. (43) then closes algebraically (no error function appears, since the visible integral is Gaussian), and any retrieval solution lies on the curve

$$
\alpha _ { k } ( m ) = \frac { \left( 1 - m ^ { 2 } \right) m ^ { 2 ( k - 2 ) } } { ( 2 k - 3 ) ! ! } ,\tag{44}
$$

whose maximum over $0 < m < 1$ is attained at $m _ { c } ^ { 2 } =$ $( k - 2 ) / ( k - 1 )$ and yields the capacity in closed form:

$$
\alpha _ { c } = \frac { ( k - 2 ) ^ { k - 2 } } { ( k - 1 ) ^ { k - 1 } ( 2 k - 3 ) ! ! } ,\tag{45}
$$

again a spinodal at which the stable and unstable retrieval branches merge. This gives $\alpha _ { c } = 4 / 4 0 5 \simeq 9 . 8 8 \times 1 0 ^ { - 3 }$ for $k = 4$ and $\alpha _ { c } \simeq 8 . 6 7 \times 1 0 ^ { - 5 }$ for $k = 6$ . Note that the retrieval overlap at capacity, $m _ { c } \simeq 0 . 8 2$ and 0.89 respectively, lies visibly below the corresponding values 0.92 and 0.96 of Model A: even at zero temperature, the soft spherical spins trade retrieval quality against the crosstalk.

![](images/beac3bfacacafa065307a32edef525b2e1e79221a91219c2258c8a7bd45dccfb.jpg)

![](images/dca0fae6ea60a8b8629b4e3bafb9c195d2ce517459a551f46413a85792a1d874.jpg)  
T(α): retrieval spinodal (existence edge) ……... Tg(α): RS spin-glass line (continuous)  
FIG. 3: RS phase diagram of Model C in the $( \alpha _ { k } , T )$ plane for (a) $k = 2$ , (b) $k = 4$ , and (c) $k = 6 ,$ , in the same conventions as Fig. 2. For $k = 2$ retrieval survives only on the segment $\alpha _ { k } = 0 , T \le 1$ (red line on the vertical axis), reflecting the zero capacity of the spherical model. For $k > 2$ the diagram has the same topology as that of Model A, with uniformly smaller retrieval regions.

TABLE II: Critical values of Model C obtained from the RS equations of state, in the same conventions as Table I. For $k = 2$ retrieval survives only at $\alpha _ { k } = 0$ , so that $\alpha _ { c } = 0$ and the steepness is not defined.
<table><tr><td>k</td><td> $T _ { c }$ </td><td> $\alpha _ { c }$ </td><td> $T _ { c } / \alpha _ { c }$ </td></tr><tr><td>2</td><td>1</td><td>0</td><td>一</td></tr><tr><td>4</td><td>0.25</td><td> $9 . 8 8 \times 1 0 ^ { - 3 }$ </td><td>25</td></tr><tr><td>6</td><td>0.148</td><td> $8 . 6 7 \times 1 0 ^ { - 5 }$ </td><td> $1 . 7 \times 1 0 ^ { 3 }$ </td></tr></table>

Critical temperature and retrieval boundary. At $\alpha _ { k } \ = \ 0$ the equations of state give $q \ = \ m ^ { 2 }$ and $1 \ =$ $\beta _ { k } ( 1 - m ^ { 2 } ) m ^ { k - 2 }$ , whose solvability condition determines, for $k > 2 .$

$$
T _ { c } = \frac { 2 } { k } \bigg ( \frac { k - 2 } { k } \bigg ) ^ { ( k - 2 ) / 2 } .\tag{46}
$$

The transition is again of the spinodal type, with the overlap jumping to $m _ { c } ^ { 2 } = ( k - 2 ) / k$ at $T _ { c }$ . The resulting critical values are collected in Table II. As for Model A, the retrieval boundary steepens as $T _ { c } / \alpha _ { c } \sim ( 2 k - 3 ) ! !$ , driven by the same factorial growth of the crosstalk variance, while every entry is uniformly smaller than its model-A counterpart in Table I.

Phase diagram. The resulting RS phase diagram is shown in Fig. 3. Since the crosstalk moment $\mathcal { M } _ { k }$ is universal, the $m = 0$ sector is governed by the same equations as in Model A, and the continuous spin-glass line $T _ { g } ( \alpha _ { k } )$ of Eq. (37) carries over unchanged. The firstorder boundary $T _ { M } ( \alpha _ { k } )$ again follows by equating the free energy Eq. (41) of the retrieval branch with that of the $m = 0$ branch and, in contrast to Model A, its zerotemperature endpoint is available in closed form: along the capacity curve Eq. (44), the energy balance between the retrieval and spin-glass states reduces to the condition $\sqrt { 1 - m ^ { 2 } } = 1 / ( k - 1 )$ , which yields

$$
\alpha _ { M } ( 0 ) = \frac { k ^ { k - 2 } ( k - 2 ) ^ { k - 2 } } { ( k - 1 ) ^ { 2 k - 2 } ( 2 k - 3 ) ! ! } ,\tag{47}
$$

$\mathrm { i . e . , } \alpha _ { M } ( 0 ) \simeq 5 . 8 5 \times 1 0 ^ { - 3 }$ for $k = 4$ and $3 . 6 0 \times 1 0 ^ { - 5 }$ for $k = 6$ . In contrast to Model A, neither boundary of Model C is reentrant: the zero-temperature slope of the spinodal is strictly negative,

$$
\partial _ { T } \alpha _ { R } \big | _ { T = 0 } \propto \big ( 1 - m _ { c } ^ { 2 } \big ) \mathcal { M } _ { k } ^ { \prime } ( 1 ) - \mathcal { M } _ { k } ( 1 ) < 0 ,\tag{48}
$$

and the first-order boundary is likewise monotone. The reentrance mechanism of Model A is suppressed by the visible entropy: the spherical retrieval state pays the confinement entropy $\begin{array} { r } { - \frac { 1 } { 2 } \log ( 1 - q ) \simeq \frac { 1 } { 2 } } \end{array}$ log $\beta _ { k }$ of the condensate, which grows without bound at low temperature, so it melts together with the $m = 0$ background instead of remaining frozen against it. For $k = 2$ the retrieval phase degenerates to the segment $\alpha _ { k } = 0 , T \le 1$ , the graphical expression of the marginality discussed above, while the spin-glass line, being common to the two models, is unafected by the collapse of the retrieval sector.

The comparison between Models A and C, displayed side by side in Fig. 2 and Fig. 3, thus isolates the role of the visible degrees of freedom: the crosstalk moment $\mathcal { M } _ { k }$ is universal, so all diferences, from the collapse of the $k = 2$ capacity to the absence of the reentrant boundary, originate from the visible entropy. For $k = 2$ , the saturating sign activation of the Ising spins is essential for retrieval: replacing it by the linear spherical activation leaves the retrieval state only marginally confined and destroys the capacity entirely. For $k > 2 ,$ , the k-body signal restores a genuine retrieval phase, though with capacity and critical temperature reduced by $O ( 1 )$ factors relative to Model A at each k. The retrieval landscape of this family is therefore shaped jointly by the sharpness of the visible activation and by the order of the hidden nonlinearity, with the latter dominating at large k through the common factorial collapse of $\operatorname { E q . }$ (32) and Eq. (45).

## IV. MODEL B

Model B is the attention model of the class H [19], defined by the Lagrangians

$$
L _ { v } ( v ) = \frac 1 2 \| v \| ^ { 2 } , \qquad L _ { h } ( h ) = \log \sum _ { \mu } e ^ { h _ { \mu } } ,\tag{49}
$$

with the activation functions $g ( v ) \ = \ v$ and $f ( h ) \ =$ softmax(h), where softmax $( h ) _ { \mu } : = e ^ { h _ { \mu } } / \sum _ { \nu } e ^ { h _ { \nu } }$ . From Eq. (4), the model-B energy function reads

$$
\begin{array} { r l } & { E _ { \xi } ^ { \mathrm { B } } ( v , h ) = \displaystyle \frac { 1 } { 2 \tau _ { v } } \| v \| ^ { 2 } + \frac { 1 } { \tau _ { h } } \bigg ( h ^ { \top } \mathrm { s o f t m a x } ( h ) - \log \sum _ { \mu } e ^ { h _ { \mu } } \bigg ) } \\ & { ~ - \frac { \lambda } { \tau _ { h } \tau _ { v } } \mathrm { s o f t m a x } ( h ) ^ { \top } \xi ^ { ( h , v ) } v . } \end{array}\tag{50}
$$

The hidden Legendre term has an information-theoretic meaning: in terms of the activation $f = \operatorname { s o f t m a x } ( h )$ it equals $\sum _ { \mu } f _ { \mu }$ log $f _ { \mu } ,$ , the negative Shannon entropy of the probability vector $f .$ Since the energy depends on h only through $f$ (the softmax is invariant under the uniform shift $h  h + c { \bf 1 } )$ , we may change variables and regard the energy as a function on $\mathbb { R } ^ { N _ { v } } \times \Delta ^ { N _ { h } }$

$$
E _ { \xi } ^ { \mathrm { B } } ( v , f ) = \frac { \left. v \right. ^ { 2 } } { 2 \tau _ { v } } + \frac { 1 } { \tau _ { h } } \sum _ { \mu } f _ { \mu } \log f _ { \mu } - \frac { \lambda } { \tau _ { h } \tau _ { v } } f ^ { \top } \xi ^ { ( h , v ) } v ,\tag{51}
$$

where $\begin{array} { r } { \Delta ^ { N _ { h } } \ : = \ \left\{ f \in \mathbb { R } ^ { N _ { h } } \Big | f _ { \mu } > 0 , \sum _ { \mu } f _ { \mu } = 1 \right\} } \end{array}$ is the open simplex. In these variables the hidden sector reduces to the normalized weights f that the network assigns to the stored patterns. These are the attention weights of the modern Hopfield network and of the transformer attention mechanism [12–14], and, in accordance with the general discussion of Sec. II A, they constitute the order parameter of memory retrieval: retrieval of the pattern $\mu$ corresponds to the concentration of $f$ at the vertex $e _ { \mu }$ of the simplex.

Following Sec. II B, the partition function of Model B

is defined $\mathrm { b y ^ { 4 } }$

$$
Z _ { \xi } ^ { \mathrm { B } } ( \beta ) = \int _ { \mathbb { R } ^ { N _ { v } } \times \Delta ^ { N _ { h } } } d v d f \exp \bigl ( - \beta E _ { \xi } ^ { \mathrm { B } } ( v , f ) \bigr ) .\tag{52}
$$

In the adiabatic limit $\beta / \tau _ { h }  \infty .$ , the f integral localizes at the stationary point

$$
f ^ { * } = \mathrm { s o f t m a x } \bigg ( \frac { \lambda } { \tau _ { v } } \xi ^ { ( h , v ) } v \bigg ) ,\tag{53}
$$

which is precisely $f ( h _ { * } )$ evaluated at the adiabatic saddle point $h _ { * }$ of Sec. II B, and the partition function reduces to the visible integral Eq. (10) with the efective energy

$$
E _ { \xi } ^ { \mathrm { B } } ( v , h _ { * } ) = \frac { \left. v \right. ^ { 2 } } { 2 \tau _ { v } } - \frac { 1 } { \tau _ { h } } \log \sum _ { \mu } e ^ { \frac { \lambda } { \tau _ { v } } ( \xi ^ { ( h , v ) } v ) _ { \mu } } .\tag{54}
$$

The analysis below depends on the parameters only through the two combinations $\beta / \tau _ { h }$ and $\begin{array} { r l } { \tilde { \beta } } & { { } : = } \end{array}$ $( \beta / \tau _ { v } ) ( \lambda / \tau _ { h } ) ^ { 2 }$ . In parallel with the reduction of the couplings to $\beta _ { k }$ for Models A and C, we fix the normalization $\tau _ { v } = 1$ and $\tau _ { h } = \lambda$ , for which ${ \tilde { \beta } } = \beta \colon$ the efective visible energy in Eq. (54) becomes $\begin{array} { r } { { \frac { 1 } { 2 } } \| v \| ^ { 2 } - { \frac { 1 } { \lambda } } \log \sum _ { \mu } e ^ { \lambda ( \xi ^ { ( h , v ) } v ) _ { \mu } } } \end{array}$ precisely the energy function of the modern Hopfield network [13] in the form whose exponential storage was analyzed in [15]. In this normalization λ controls the sharpness of the attention, while $\beta$ remains the genuine inverse temperature.

As for Models A and C, for Model B the visible sector can be integrated out exactly: the visible Lagrangian is quadratic, so the v integral in $\operatorname { E q . }$ (52) is Gaussian. Carrying it out yields, up to an overall constant,

$$
Z _ { \xi } ^ { \mathrm { B } } ( \beta ) \propto \int _ { \Delta ^ { N _ { h } } } d f e ^ { \Phi ( f ) } ,\tag{55}
$$

$$
\Phi ( f ) = - { \frac { \beta } { \lambda } } \sum _ { \mu } f _ { \mu } \log f _ { \mu } + { \frac { \beta } { 2 } } f ^ { \top } G f ,\tag{56}
$$

where $G : = \xi ^ { ( h , v ) } \xi ^ { ( v , h ) }$ is the Gram matrix of the patterns, $G _ { \mu \nu } = \xi _ { \mu } ^ { ( h , v ) } \cdot \xi _ { \nu } ^ { ( h , v ) }$ . We refer to $\mathrm { E q . ~ ( 5 6 ) }$ as the f representation of Model B: the entire thermodynamics is expressed by the attention weights alone, as a competition between the attention entropy, which favors delocalized attention, and the positive semi-definite energy $f ^ { \top } G f = \left\| \xi ^ { ( v , h ) } f \right\| ^ { 2 }$ which favors concentration.

We draw the patterns independently as Gaussian variables, $\xi _ { \mu i } ^ { ( h , v ) } \ \stackrel { \sim } { \sim } \ N ( 0 , 1 )$ , for which $G _ { \mu \mu } ~ = ~ N _ { v } ( 1 +$ $O ( N _ { v } ^ { - 1 / 2 } ) )$ while $G _ { \mu \nu } = O ( \sqrt { N _ { v } } )$ for $\mu \neq \nu .$ . The two terms of Φ are then of widely diferent orders. The energy is extensive: attention concentrated on a single pattern already gains $\begin{array} { r } { \frac { \beta } { 2 } G _ { \mu \mu } \approx \frac { \beta } { 2 } N _ { v } } \end{array}$ . The entropy, by contrast, is bounded by its value at uniform attention, $\frac { \beta } { \lambda }$ log $N _ { h }$ : it is of order one per stored pattern rather than per neuron. $\mathrm { A }$ genuine competition between the two therefore requires log $N _ { h } \sim N _ { v }$ , and the natural high-load regime of Model B is the exponential load

$$
N _ { h } = e ^ { \alpha N _ { v } } ,\tag{57}
$$

in contrast with the polynomial loads $N _ { h } = \alpha _ { k } N _ { v } ^ { k - 1 }$ of Models A and C. A second contrast concerns the method: the disorder now enters only through the Gram matrix, whose relevant statistics are large deviations of pattern norms and overlaps, so the analysis below proceeds by direct counting arguments rather than by the replica method. The relation between the two approaches is clarified in Sec. IV A 1.

## A. Capacity at zero temperature

Vertex condensation. The geometry of Eq. (56) dictates the structure of the low-temperature states. The energy $\begin{array} { r } { f ^ { \top } G f = \left. \sum _ { \mu } f _ { \mu } \xi _ { \mu } ^ { ( h , v ) } \right. ^ { 2 } } \end{array}$ is a convex function of $f ,$ and a convex function on a compact convex set attains its maximum at an extreme point (Bauer’s maximum principle). On the simplex the extreme points are the vertices. Explicitly,

$$
\left\| \sum _ { \mu } f _ { \mu } \xi _ { \mu } ^ { ( h , v ) } \right\| \leq \sum _ { \mu } f _ { \mu } \left\| \xi _ { \mu } ^ { ( h , v ) } \right\| \leq \operatorname* { m a x } _ { \mu } \left\| \xi _ { \mu } ^ { ( h , v ) } \right\| ,\tag{58}
$$

with equality at the vertex of the maximizing pattern. The entropy opposes this concentration only weakly: spreading the attention uniformly over M patterns reduces the energy gain from $\frac { \beta } { 2 } N _ { v }$ to $\frac { \beta } { 2 } N _ { v } / M$ at leading order, an extensive loss, whereas the entropy gain is merely $\frac { \beta } { \lambda }$ log M. For any subexponential load the entropy is thus negligible at leading order, and the Gibbs measure condenses onto the vertices of the simplex, each vertex describing the retrieval state of one pattern with v localized near it (cf. Eq. (53)). At the exponential load Eq. (57) the competition becomes genuine, but not through interior attention distributions: the measure instead spreads over exponentially many nearly pure vertex states, $\begin{array} { r } { Z _ { \xi } ^ { \mathrm { B } } \approx \sum _ { \mu } \bar { Z _ { \mu } } , \bar { Z _ { \mu } } , } \end{array}$ and the entropy is the counting entropy of this decomposition. The capacity question is whether the state condensed on a given typical pattern survives against this exponentially large background.

Retrieval capacity. Fix the retrieved pattern $\xi _ { 1 } ^ { ( h , v ) }$ 2 of typical norm $\left\| \xi _ { 1 } ^ { ( h , v ) } \right\| ^ { 2 } \approx N _ { v }$ . The stationarity of the efective energy in Eq. (54) is the fixed-point condition $v = \xi ^ { ( v , h ) } f ^ { * } ( \boldsymbol { v } ) ;$ : the visible state is the attention-weighted superposition of the stored patterns. At zero temperature the retrieval state lies at $\bar { v } \approx \xi _ { 1 } ^ { ( h , v ) }$ , up to corrections controlled by the leak of attention computed below, and each pattern feels the attention field $a _ { \mu } = \lambda \xi _ { \mu } ^ { ( h , v ) } \cdot v .$ $\mathrm { A t } \ v = \xi _ { 1 } ^ { ( h , v ) }$ one has $a _ { 1 } = \lambda \Big \| \xi _ { 1 } ^ { ( h , v ) } \Big \| ^ { 2 }$ ≈ $\lambda N _ { v }$ , while for $\mu \geq 2$ , conditioned on $\xi _ { 1 } ^ { ( h , v ) }$ , the overlaps $\xi _ { \mu } ^ { ( h , v ) } \cdot \xi _ { 1 } ^ { ( h , v ) }$ are independent centered Gaussians of variance $\left\| \xi _ { 1 } ^ { ( h , v ) } \right\| ^ { 2 }$ , so that $a _ { \mu } = \lambda \sqrt { N _ { v } } \omega _ { \mu }$ with independent standard Gaussian variables $\omega _ { \mu } .$ . The condensed attention weight $p : = f _ { 1 } ^ { * }$ the model-B analogue of the retrieval overlap m of Models A and C, then obeys

$$
1 - p \le e ^ { - a _ { 1 } } L , \qquad L : = \sum _ { \mu \ge 2 } e ^ { \lambda \sqrt { N _ { v } } \omega _ { \mu } } .\tag{59}
$$

The sum $L$ is evaluated by the counting argument that underlies extreme value statistics: the number of patterns whose variable $\omega _ { \mu }$ reaches the level $x \sqrt { N _ { v } }$ is $e ^ { N _ { v } ( \alpha - x ^ { 2 } / 2 ) }$ to leading exponential order, so the levels up to $x _ { \mathrm { m a x } } = \sqrt { 2 \alpha }$ are populated while higher levels are empty with high probability. Retaining the maximal term,

$$
\frac { \log L } { N _ { v } } = \operatorname* { m a x } _ { 0 \le x \le \sqrt { 2 \alpha } } \left[ \alpha - \frac { x ^ { 2 } } { 2 } + \lambda x \right] = \left\{ \alpha + \frac { \lambda ^ { 2 } } { 2 } , \lambda \le \sqrt { 2 \alpha } , \right.\tag{60}
$$

In the first branch the maximum is attained at the interior point $x ^ { * } = \lambda { : }$ the leak is carried by exponentially many patterns of moderate overlap, and L is self-averaging and coincides with its annealed average. In the second branch the maximum is pinned at the boundary $x _ { \mathrm { m a x } } \colon$ the leak is dominated by the $O ( 1 )$ most aligned patterns, and the sum is frozen. The retrieval state is self-consistent precisely when the leak vanishes, $- \lambda + N _ { v } ^ { - 1 }$ log $L < 0$ Combining the two branches yields the zero-temperature capacity

$$
\alpha _ { c } ( \lambda ) = \left\{ \begin{array} { l l } { { \lambda - \displaystyle \frac { \lambda ^ { 2 } } { 2 } , } } & { { \lambda \leq 1 , } } \\ { { 1 } } & { { } } \\ { { \displaystyle \frac { 1 } { 2 } , } } & { { \lambda \geq 1 , } } \end{array} \right.\tag{61}
$$

the two branches matching continuously at $\lambda = 1$ . Details of these estimates are collected in Appendix C 1.

Retrieval fails in physically distinct ways in the two regimes of Eq. (61). For $\lambda \ < \ 1$ the attention is too soft: retrieval is destroyed by the aggregate crosstalk of exponentially many weakly correlated patterns, and sharpening the attention raises the capacity. For $\lambda > 1$ the crosstalk is dominated by the single most aligned competitor, whose overlap max $z _ { \mu \geq 2 } \xi _ { \mu } ^ { ( h , \stackrel { \smile } { v } ) } \cdot \xi _ { 1 } ^ { ( h , v ) } \approx \sqrt { 2 \alpha } N _ { v }$ matches the signal $\left\| \xi _ { 1 } ^ { ( h , v ) } \right\| ^ { 2 } \approx N _ { v }$ at $\alpha = 1 / 2$ . Since λ multiplies the signal and the competitor field alike, it cancels from this comparison: no attention sharpness overcomes a competitor as aligned as the signal itself, hence the ceiling. The capacity Eq. (61) is the threshold for retrieving a typical pattern. Requiring that all $e ^ { \alpha N _ { v } }$ patterns be retrievable simultaneously is a stricter demand, met only at lower loads [15]. The value of the ceiling is also specific to the Gaussian ensemble, which enters through the counting rate $x ^ { 2 } / 2 \colon$ for patterns drawn on the sphere the ceiling disappears and the capacity continues to grow logarithmically in λ [15], while for binary patterns exponential capacities were established rigorously in [11], and the capacity has been computed for more general ensembles, including patterns drawn from a hidden manifold [26]. This sensitivity is characteristic of the exponential regime: at polynomial load the crosstalk is governed by central-limit statistics and is largely insensitive to the pattern ensemble (Sec. III), whereas at exponential load it is governed by large deviations, which are not universal.

Relation to the random energy model. The leak sum in Eq. (59) is precisely the partition function of a random energy model (REM) with $e ^ { \alpha N _ { v } }$ independent Gaussian energy levels at inverse temperature λ [21, 27], and Eq. (60) is the standard large-deviation evaluation of its free energy [28]. The two branches are the two phases of the REM: the entropy-dominated phase, in which annealed and quenched averages agree, and the condensed phase below the freezing transition $\lambda = \sqrt { 2 \alpha }$ , in which the measure concentrates on finitely many levels. This identification makes the agreement with [15] structural rather than accidental: there, the zero-temperature energy landscape is split into the signal of the retrieved pattern and a noise term recognized as the free energy of an auxiliary REM, and the resulting typical-pattern capacity $\alpha _ { 1 } ( \lambda )$ coincides exactly with Eq. (61). The vertex condensation of the Gibbs measure and the landscape analysis are two routes to the same variational problem.

The same extreme value statistics also fixes the status of the retrieval states in the equilibrium ensemble. At exponential load there exist patterns of atypically large norm, up to $\left. \xi _ { \mu ^ { * } } ^ { ( h , v ) } \right. ^ { 2 } = ( 1 + \varepsilon _ { \operatorname* { m a x } } ) N _ { v }$ with $\varepsilon _ { \operatorname* { m a x } } > 0$ determined by the large deviations of the norm, and the state retrieving such a pattern has energy density $- ( 1 + \varepsilon _ { \mathrm { { m a x } } } ) / 2$ , strictly below the value $- 1 / 2$ of typical retrieval. Typical retrieval is therefore metastable at any exponential load, and Eq. (61) is a spinodal, in the same sense as the capacities of Models A and C. The equilibrium phase structure built on the condensed extreme patterns is the subject of Sec. IV B.

## 1. Copy representation

The f representation admits an equivalent discrete formulation, which both fixes its integration measure and clarifies its relation to the replica method. Consider the adiabatic partition function, the visible integral of the efective energy Eq. (54), at the discrete temperatures for which $n : = \beta / \lambda$ is a positive integer. The logarithm in the exponent then exponentiates into the n-th power of $\begin{array} { r } { \sum _ { \mu } e ^ { \lambda \xi _ { \mu } ^ { ( h , v ) } \cdot v } } \end{array}$ , the power expands by the multinomial theorem into a sum over n-tuples of pattern indices, and the Gaussian v integral gives, up to an overall constant,

$$
Z _ { \xi } ^ { \mathrm { B } } ( \beta ) \propto \sum _ { \mu _ { 1 } , \dots , \mu _ { n } = 1 } ^ { N _ { h } } \exp \{ \frac { \lambda } { 2 n } \biggl | | \sum _ { j = 1 } ^ { n } \xi _ { \mu _ { j } } ^ { ( h , v ) } | | ^ { 2 } \} .\tag{62}
$$

An integer-power representation of this type was introduced in [14]. The hidden-sector fluctuations around the adiabatic saddle point renormalize the exponent as $\beta / \lambda \to \beta / \lambda + N _ { h } / 2$ , together with a shift of the summed patterns, and we relegate these corrections to Appendix C 2.

The expansion $\operatorname { E q . }$ . (62) describes n “copies”, each selecting one stored pattern, which interact through the Gram matrix of the selected patterns. Coarse-graining a copy configuration by its empirical measure $\hat { f } _ { \mu } : = n _ { \mu } / n _ { ; }$ with $n _ { \mu }$ the number of copies selecting the pattern $\mu ,$ the number of configurations in a class is the multinomial coefficient $n ! / \prod _ { \mu } n _ { \mu } !$ , and Stirling’s formula converts Eq. (62) into

$$
Z _ { \xi } ^ { \mathrm { B } } ( \beta ) \propto \sum _ { \hat { f } } \exp \left\{ - \frac { \beta } { \lambda } \sum _ { \mu } \hat { f } _ { \mu } \log \hat { f } _ { \mu } + \frac { \beta } { 2 } \hat { f } ^ { \top } G \hat { f } \right\}\tag{63}
$$

up to subexponential factors, where the sum runs over the grid $\hat { f } \in \Delta ^ { N _ { h } } \cap ( \mathbb { Z } _ { > 0 } / n ) ^ { N _ { h } }$ This is exactly the functional Φ of $\operatorname { E q . } \ ( 5 6 ) \colon$ the copy representation is the $f$ representation with its measure made explicit as a counting measure. This distinction is not innocuous at exponential load, where the simplex has dimension $e ^ { \alpha N _ { v } } -$ 1 and the volume factors of a continuum measure are a priori uncontrolled. The counting measure supplied by the model itself carries no such factors, its only extensive entropy being the choice of the selected patterns.

The physical picture behind $\operatorname { E q . }$ . (62) is transparent. Reversing the Gaussian integration shows that, given a copy configuration, the visible state is Gaussian with mean the centroid $\begin{array} { r } { \frac { 1 } { n } \sum _ { j } \xi _ { \mu _ { j } } ^ { ( h , v ) } } \end{array}$ and variance $1 / \beta$ per component: the n copies are parallel retrieval queries sharing the single context $v ,$ and the temperature enters only through the number of queries, $n = \beta / \lambda$ . Retrieval is the fully aligned configuration in which all copies select the same pattern, and the elementary excitation above it is a single copy defecting to a competitor, which carries the attention quantum $\bar { 1 } / n = \lambda T$ . As $T  0$ the number of copies diverges, the aligned configurations reproduce the vertex condensation described above, and the stability of alignment against single-copy defection reproduces the capacity Eq. (61) (see Appendix C 3).

The copy representation also locates the present analysis relative to the replica method: in the replica approach the patterns are averaged first, at a formal number of replicas continued to zero, whereas here the thermal variable v is integrated out first, at a physical, integer number of copies, so the thermal and quenched averages exchange roles. The efective log det interaction that the pattern average generates among replicas reappears here as the counting entropy of the pattern choices, its Legendre transform (Appendix C 2). Likewise, the symmetric structure of the order parameter, an ansatz in the replica computation, arises here as the exact maximizer within each class of copy configurations, and the frozen branch of $\operatorname { E q . }$ (60) plays the role that one-step RSB (1RSB) plays in the REM. The construction realizes the clone method of Monasson [22], with the number of clones set by the physical temperature rather than introduced as an auxiliary parameter.

## B. Phase diagram at finite temperature

The copy representation reduces the finite-temperature problem to a combinatorial one. The temperature enters only through the copy number $n = \beta / \lambda ,$ , and a configuration is specified by the partition of the n copies into groups occupying distinct patterns. Since the weight in Eq. (62) depends on the occupied patterns only through their Gram matrix, the disorder average reduces to a counting problem: the number of choices of M distinct patterns with prescribed normalized Gram matrix $Q _ { j l } = G _ { \mu _ { j } \mu _ { l } } / N _ { v }$ is $e ^ { N _ { v } ( M \alpha - I _ { M } ( Q ) ) }$ to leading exponential order, where

$$
I _ { M } ( Q ) = \frac { 1 } { 2 } [ \mathrm { T r } Q - \log \operatorname * { d e t } Q - M ]\tag{64}
$$

is the large-deviation rate function of the Gram matrix, the relative entropy between the centered Gaussian ensembles of covariance Q and of unit covariance [28]. For $M \alpha < I _ { M } ( Q )$ no such choice exists with high probability. Maximizing the counting factor times the weight under this existence constraint extends the two branches of Eq. (60) to every configuration class: an annealed branch, attained at the typical Gram matrix of an exponentially tilted ensemble, and a frozen branch, pinned to the most extreme configurations actually present. The optimization over the partition classes is elementary and is carried out in Appendix C 3. Here we summarize the results.

Equilibrium phases. Only the two extreme partitions survive the optimization, all copies on separate patterns or all copies on a single pattern. Measuring the free energy per visible neuron relative to the Gaussian reference $\int \mathop { d v } e ^ { - \beta \| v \| ^ { 2 } / 2 }$ , the resulting branches are

$$
\begin{array} { l } { \displaystyle f _ { \mathrm { P } } = - \frac { \alpha } { \lambda } + \frac { T } { 2 } \log ( 1 - \lambda ) , } \\ { \displaystyle f _ { \mathrm { C } } = - T \alpha + \frac { T } { 2 } \log ( 1 - \beta ) , } \\ { \displaystyle f _ { \mathrm { F } } = - \frac { 1 + \varepsilon _ { \mathrm { m a x } } ( \alpha ) } { 2 } , } \end{array}\tag{65}
$$

and the equilibrium phase at given $( \alpha , \lambda , T )$ is the branch of lowest free energy. In the paramagnetic phase (P) each copy selects its own pattern and contributes the selection entropy α. The visible state is a weak thermal condensate, $\big \langle \big | | v \| \big | ^ { 2 } \big \rangle / N _ { v } = T / ( 1 - \lambda )$ , which vanishes as $T  0$ . The selection entropy itself, however, does not vanish: it enters in proportion to the copy number $n = \beta / \lambda$ and leaves the temperature-independent term $- \alpha / \lambda$ in $f _ { \mathrm { P } } { \mathrm { . } }$ , so that the paramagnet survives down to $T = 0$ and remains the equilibrium phase at loads beyond the zero-temperature intercept $\alpha _ { \mathrm { t h } } ( 0 )$ of the P–F boundary in Fig. $4 ( \mathrm { a } ) \mathrm { : }$ at exponential load, the entropy of pattern selection acts as an energy. In the condensed phase (C) all n copies align on a single pattern, thermally tilted toward the norm $( 1 - \beta ) ^ { - 1 } N _ { v }$ , and the measure is an ergodic mixture over the exponentially many retrieval lumps of Sec. IV A, with entropy density $\alpha - \kappa ( \beta )$ , where $\kappa ( \bar { \beta ) } : = I _ { 1 } ( ( 1 - \beta ) ^ { - 1 } )$ = $\textstyle { \frac { 1 } { 2 } } [ \beta / ( \dot { 1 } - \beta ) + \log ( 1 - \ddot { \beta } ) ]$ is the counting cost of the tilted norm. This entropy vanishes on the freezing line

$$
T _ { f } ( \alpha ) = 1 + \frac { 1 } { \varepsilon _ { \operatorname* { m a x } } ( \alpha ) } ,\tag{66}
$$

where $\varepsilon _ { \mathrm { m a x } } ( \alpha )$ , defined by $I _ { 1 } ( 1 + \varepsilon _ { \mathrm { m a x } } ) = \alpha _ { \mathrm { m } }$ , makes precise the maximal norm excess introduced in Sec. IV A. Below $T _ { f }$ the mixture freezes onto the $O ( 1 )$ patterns of maximal norm: this frozen phase (F) is the retrieval state of the maximum-norm pattern, its free energy is temperature independent, and the transition at $T _ { f }$ is the continuous freezing transition of the REM [21, 27]. The existence conditions of the two annealed branches, $\lambda < 1$ for P and $T > 1$ for $\mathrm { C } ,$ are two instances of a single criterion: a state with attention spread over $n / s$ patterns has inverse participation ratio $\left\| \boldsymbol f \right\| ^ { 2 } = s / n$ , and its visible Gaussian fluctuations are stable only while $\beta \| f \| ^ { 2 } < 1$ , the instability at $\beta \| f \| ^ { 2 } = 1$ being the divergence of the visible integral in the $f$ representation Eq. (56). In particular, for $\lambda \geq 1$ the paramagnetic phase is absent altogether. The first-order boundaries between P and the condensed sector follow by equating the branches of Eq. (65), and the resulting phase diagram is shown in Fig. 4. For $\lambda < 1$ the paramagnetic and the two condensed phases meet at a triple point, located at $( \alpha , T ) \simeq ( 0 . 6 6 , 1 . 3 8 )$ for $\lambda = 0 . 5$ Retrieval branch. Typical retrieval appears in this landscape as a constrained branch. Fixing the attention weight $p = { \hat { f } } _ { 1 }$ condensed on a typical pattern as a reaction coordinate, and optimizing over the destinations and the Gram matrix of the $n ( 1 - p )$ remaining copies, yields the Landau function

$$
f _ { \mathrm { R } } ( p ) = - \frac { \alpha } { \lambda } ( 1 - p ) + \frac { T } { 2 } \log A - \frac { p ^ { 2 } } { 2 A } ,\tag{67}
$$

where $A : = 1 - \lambda ( 1 - p )$ The three terms are the selection entropy of the leaked attention, the determinant of the visible fluctuations softened by the leak, and the signal energy. The leaked copies respond linearly to the retrieval field and align weakly with the retrieved pattern, which amplifies the visible overlap to $m = p / A \geq p .$ . The branch interpolates between the paramagnet, $f _ { \mathrm { R } } ( 0 ) = f _ { \mathrm { P } }$ and pure retrieval, $f _ { \mathrm { R } } ( 1 ) = - \frac { 1 } { 2 }$ , the zero-temperature retrieval energy density of Sec. IV A. Its interior stationary point is a maximum: a barrier between the two endpoints, not a phase. Interior attention distributions therefore never become states at any temperature, which is the finite-temperature form of vertex condensation. The construction is a Franz–Parisi potential [29] whose reference configuration is the stored pattern itself, and the retrieval state at $p = 1$ is a metastable state in the restrictedensemble sense [30].

![](images/72d15e1b2afb88491231b3b690e91bb17b30d38f818a6e9b0e5a6e07e7f2f800.jpg)  
FIG. 4: Finite-temperature phase diagram of Model B in the $( \alpha , T )$ plane at (a) λ = 0.5 and (b) λ = 1.5. Black solid lines are the first-order boundaries between the paramagnetic phase (P) and the condensed sector, obtained by equating the free energies Eq. (65). For $\lambda \geq 1$ the paramagnetic phase is absent, and in (a) the P–F boundary meets $T = 0 \mathrm { a t } \alpha _ { \mathrm { t h } } ( 0 )$ determined by $2 \alpha / \lambda = 1 + \varepsilon _ { \mathrm { m a x } } ( \alpha )$ , see Appendix C 1. The dotted line is the freezing line $\operatorname { E q . }$ (66), the continuous transition on which the entropy of the condensed phase (C) vanishes and the measure freezes onto the maximum-norm patterns (F). It is independent of λ and hence identical in the two panels. The thick red line is the retrieval spinodal, evaluated numerically from the one-quantum criterion of Appendix C 4, whose annealed and frozen branches are Eq. (68) and Eq. (69). The dashed line shows its closed-form asymptotics, and the dash-dotted line is the switch line $I ^ { * } ( \dot { T } )$ between the annealed and extreme-value evaluations of the defection destination. The hatched region is the metastable retrieval region, which terminates at $T = 1 / \lambda ,$ where the attention quantum λT reaches the full attention weight (n = 1).

Finite-temperature capacity. The stability of the retrieval endpoint is not governed by the slope of Eq. (67) at $p = 1$ The attention weight moves on the lattice $p = 1 - k \lambda T$ with $k \in \mathbb { Z } _ { > 0 } \colon$ the elementary excitation is the defection of a single copy, which carries the attention quantum λT, and retrieval survives as long as a single defection raises the free energy. Optimizing the destination of the defecting copy over the patterns present in the ensemble yields the finite-temperature spinodal

$$
\alpha _ { c } ( \lambda , T ) = \frac { \lambda ( 2 - \lambda - \lambda T ) } { 2 ( 1 - \lambda ^ { 2 } T ) } + \frac { 1 } { 2 } \log \bigl ( 1 - \lambda ^ { 2 } T \bigr )\tag{68}
$$

in the annealed regime, in which destinations of the optimal overlap and norm exist in exponential number. For sharp attention, $\lambda ^ { 2 } \gtrsim 2 \alpha$ , they do not, and the defection freezes onto the single most aligned pattern present, as in the frozen branch of $\operatorname { E q . }$ (60). Two efects of order T then favor the defection: the condensate left behind is thinner by one copy, and an atypically large norm of the destination pattern becomes an energy gain in its own right. The zero-temperature ceiling accordingly bends downward,

$$
\alpha _ { \mathrm { c e i l } } ( \lambda , T ) = \frac { 1 } { 2 } \biggl ( 1 - \frac { \lambda T } { 2 } \biggr ) ^ { 2 } + { \cal O } ( T ^ { 2 } ) ,\tag{69}
$$

and the finite-temperature capacity follows $\operatorname { E q . }$ (68) in the annealed regime and $\operatorname { E q }$ . (69) in the frozen regime, with the branch switch at $\lambda ^ { 2 } \approx 2 \alpha$ . As $T \to 0$ the two branches reduce to the zero-temperature capacity Eq. (61). In the opposite direction, the metastable region terminates at $T = 1 / \lambda$ , where the attention quantum λT reaches the full attention weight and a single defection already empties the condensate (see Fig. 4). The approach to this endpoint is controlled by the frozen branch: at small α destinations of the optimal overlap cease to exist, so the annealed expression Eq. (68), whose zero lies below $1 / \lambda$ underestimates the stability of retrieval, and the defection onto the most favorable pattern actually present sustains a narrow metastable strip up to $T = 1 / \bar { \lambda }$ . A continuum treatment of $p$ would overestimate the capacity at any $T > 0 { : }$ it resolves barriers thinner than one attention quantum and counts the selection entropy per infinitesimal attention weight rather than per copy. The thermal destabilization of retrieval thus proceeds by discrete reassignments of attention, not by a smooth erosion of the overlap.

The results above are derived at the integer temperature points $n \in \mathbb { Z } _ { > 0 }$ , but they are not tied to this lattice: replacing the multinomial expansion of Sec. IV A 1 by the generalized binomial expansion of the same integrand extends the sector decomposition, the attention quantum λT, and all the formulas above unchanged to arbitrary real $\beta .$ The complete real-temperature construction of the phase diagram is presented in Appendix C 4.

The phase diagram fixes the status of retrieval at exponential load. Since $\varepsilon _ { \operatorname* { m a x } } > 0$ at any $\alpha > 0$ , the frozen phase lies below pure retrieval, $f _ { \mathrm { F } } < f _ { \mathrm { R } } ( 1 ) = - \frac { 1 } { 2 }$ , at all temperatures: for Gaussian patterns, typical retrieval never becomes the equilibrium phase, and the associative memory operates throughout the hatched region of Fig. 4 as a metastable state, with an escape time exponentially large in $N _ { v } .$ governed in the Arrhenius sense by the one-quantum barrier of Appendix C 4, mapped over the metastable region in Fig. 5. This sharpens the zero-temperature statement of Sec. IV A and contrasts with Models A and C in two respects. First, the mechanism of thermal destabilization is diferent: there the retrieval overlap erodes smoothly through the equations of state, while here the temperature acts solely through the number of copies, and retrieval is undone by quantized reassignments of attention. Second, the scale is diferent: the retrieval boundaries of Models A and C steepen factorially with k (Table I and Table II), whereas the model-B retrieval region retains an extent of order unity in both α and T. Finally, the equilibrium dominance of the frozen phase is a norm-fluctuation efect of the Gaussian ensemble, driven by the same large deviations that set the capacity ceiling: for patterns drawn on the sphere, $\varepsilon _ { \operatorname* { m a x } } \equiv 0$ , the frozen phase loses its advantage, and typical retrieval can compete as a genuine equilibrium phase. The ensemble sensitivity noted below Eq. (61) thus extends from the capacity to the entire equilibrium structure.

## V. DISCUSSION

We have investigated the statistical mechanics of the class-H associative memories [19] through the hidden neurons, the order parameter of memory retrieval. For Models A and C (Sec. III), the replica method at polynomial load yields the RS phase diagrams and closed-form capacities.

Since the crosstalk moment is universal, the model dependence, foremost the zero capacity of the quadratic spherical model, rests on the visible entropy, while the factorial growth of the crosstalk variance collapses both capacities at large k. For Model B (Sec. IV), the hidden sector reduces to the attention weights, the load is exponential, and the copy representation maps the thermodynamics onto REM counting, with paramagnetic, condensed, and frozen phases and typical retrieval metastable, undone thermally by quantized reassignments of attention. The two regimes difer in their crosstalk statistics, centrallimit and largely insensitive to the pattern ensemble at polynomial load, large-deviation and ensemble-dependent at exponential load. Retrieval thus separates into the two roles carried by the Lagrangians of the class H: the visible Lagrangian fixes, through the visible entropy, the stability of retrieval under a given crosstalk, while the hidden Lagrangian fixes the storage scale and with it the character of the crosstalk statistics. The models analyzed here are positions on these two axes rather than separate theories.

Two limitations deserve mention. First, all results for Models A and C are derived within the RS ansatz, the capacities being the spinodals of the RS free energy. The de Almeida–Thouless (AT) stability of the RS saddle points [23] has not been examined, and the spin-glass phases and the low-temperature boundaries acquire corrections from RSB, known to be small for the retrieval boundaries at k = 2 [4, 24] and expected to remain so at $k > 2 ~ [ 7 ]$ . Second, for $k > 2$ and for Model B the hidden sector is treated in the adiabatic limit $\beta / \tau _ { h }  \infty$ , with the hidden neurons enslaved to the visible configuration. At finite $\beta / \tau _ { h }$ the class H is a genuinely two-temperature system, whose phase diagrams away from the adiabatic limit we do not address. The equal-temperature end of this interpolation is the bipartite Gibbs measure of a restricted Boltzmann machine, whose equivalence with the Hopfield model [31] and whose phase diagram for general hidden priors [32] are known.

These limitations mark the first direction for future work, completing the phase diagrams under RSB. For Models A and C this amounts to locating the AT lines and computing the one-step corrections to the spin-glass phase and the retrieval boundaries, for which the techniques developed for the Hopfield model [4, 24], dense networks [33, 34], and bipartite spin glasses [35] apply directly. For Model B the corresponding step is the stability of the sector decomposition against fluctuations between sectors, together with a test of the predicted Ruelle statistics of the attention weights in the frozen phase [36] (Appendix C 4).

A second direction concerns the hidden-sector corrections of Model B. The Gaussian fluctuations of the hidden neurons around the adiabatic saddle point produce the exponent shift $\beta / \lambda  \beta / \lambda + N _ { h } / 2$ and an additive shift of the summed patterns (Appendix C 2), neither of which is small at exponential load. Whether they can be absorbed into a redefinition of the reference measure, leaving the rate-level phase diagram intact, is the main structural question left open by our analysis (Appendix C 4). Since both originate from the zero mode of the softmax, their resolution would clarify how the normalization of the attention weights, shared by the transformer attention and the modern Hopfield network [12–14], afects the thermodynamics beyond the leading order.

The third direction is more open-ended. Regarded as design variables rather than properties of given models, the two axes suggest constructing new Lagrangians, or selecting them with prescribed retrieval properties, in the spirit of recent programs on the design of energy-based architectures [5, 6, 37]. The family generating Models A and C (Appendix B 1) is the smallest setting in which both axes can be varied, and the same question extends to the class $\mathcal { H } _ { k }$ (Appendix A), where additional layers of hidden neurons implement the hierarchical associative memory [38, 39]. Identifying the order parameters of the deeper hidden layers, and the phases they support, would extend the present analysis to genuinely hierarchical models, for which exponential capacities from distributed hidden representations [40] and emergent computations in assemblies of networks [41] have recently been reported.

## Appendix A: Overview of the class- $\cdot \mathcal { H } _ { k }$ associative memories

The class- $\mathbf { \nabla } \cdot \mathcal { H } _ { k }$ associative memories are a slight modification of Krotov’s hierarchical associative memory $[ 3 8 ] , ^ { 5 }$ the modification being that the energy function carries explicit coupling constants $\lambda _ { A + 1 , A }$ and layer-wise weights $1 / \tau _ { A }$ The class $\mathcal { H } _ { k }$ consists of $k \ ( \geq 2 )$ layers, with $N _ { A }$ neurons in layer A $( A = 1 , \ldots , k )$ . This system is a generalization of the model discussed in [19] and is specified by the following components: the neuron states $\boldsymbol { x } ^ { A } ( t ) \in \mathbb { R } ^ { \breve { N } _ { A } }$ in each layer, interaction matrices between the adjacent layers, $\boldsymbol { \xi } ^ { ( A + 1 , A ) } \in \mathbb { R } ^ { \dot { N } _ { A + 1 } \times N _ { A } }$ , and activation functions

$$
g ^ { A } : \mathbb { R } ^ { N _ { A } }  \mathbb { R } ^ { N _ { A } } , \quad A = 1 , \ldots , k ,\tag{A1}
$$

determined through Lagrangians $L ^ { A } : \mathbb { R } ^ { N _ { A } } $ R such that $g ^ { A } = \nabla L ^ { A }$ . Note that the interaction matrices are required to be “symmetric”: $\boldsymbol { \xi } ^ { ( A , A + 1 ) } : = ( \boldsymbol { \xi } ^ { ( A + 1 , A ) } ) ^ { \top }$ . (For more details, see [38, Sec. 3].)

Let us now define an energy function on $\begin{array} { r } { T ^ { * } ( \mathbb { R } ^ { \sum _ { A = 1 } ^ { k } N _ { A } } ) \simeq \prod _ { A = 1 } ^ { k } ( \mathbb { R } ^ { N _ { A } } \times \mathbb { R } ^ { N _ { A } } ) } \end{array}$ by

$$
E ( \{ y ^ { A } \} , \{ x ^ { A } \} ) = \sum _ { A = 1 } ^ { k } { \frac { 1 } { \tau _ { A } } } { \big ( } ( x ^ { A } ) ^ { \top } y ^ { A } - L ^ { A } ( x ^ { A } ) { \big ) } - \sum _ { A = 1 } ^ { k - 1 } { \frac { \lambda _ { A + 1 , A } } { \tau _ { A + 1 } \tau _ { A } } } ( y ^ { A + 1 } ) ^ { \top } \xi ^ { ( A + 1 , A ) } y ^ { A } ,\tag{A2}
$$

where $\lambda _ { A + 1 , A }$ are coupling constants between the $( A + 1 )$ )- and A-th layers and are assumed to be symmetric, $\lambda _ { A + 1 , A } = \lambda _ { A , A + 1 }$ . This energy function may be regarded as a Hamiltonian function on the phase space, with the variables $y ^ { A }$ playing the role of the positions and the neuron states $x ^ { A }$ that of the conjugate momenta. The first sum in E is then the Legendre transform of the Lagrangians. The physical configurations lie on the Legendre constraint surface $\Sigma : = \left\{ y ^ { A } = g ^ { A } ( x ^ { A } ) \right\} _ { A = 1 } ^ { k }$ , which is precisely the locus where the gradients $\nabla _ { x ^ { A } } E = ( y ^ { A } - g ^ { A } ( x ^ { A } ) ) / \tau _ { A }$ vanish, namely, where the position sector of Hamilton’s canonical equations, $d y ^ { A } / d t = \nabla _ { x ^ { A } } E$ , becomes stationary. We then define the dynamical equations of the system as the momentum sector of the canonical equations restricted to the constraint surface Σ, with the convention $\lambda _ { 1 , 0 } \equiv 0$ and $\lambda _ { k + 1 , k } \equiv 0$ :

$$
\begin{array} { l } { \displaystyle \frac { d x ^ { A } ( t ) } { d t } : = - \nabla _ { y ^ { A } } E ( \{ y ^ { A } \} , \{ x ^ { A } \} ) | _ { \{ ( y ^ { A } , x ^ { A } ) = ( g ^ { A } ( x ^ { A } ( t ) ) , x ^ { A } ( t ) ) \} _ { A = 1 } ^ { k } } } \\ { = \displaystyle \frac { 1 } { \tau _ { A } } \bigg ( \frac { \lambda _ { A , A - 1 } } { \tau _ { A - 1 } } \xi ^ { ( A , A - 1 ) } g ^ { A - 1 } \big ( x ^ { A - 1 } ( t ) \big ) + \frac { \lambda _ { A , A + 1 } } { \tau _ { A + 1 } } \xi ^ { ( A , A + 1 ) } g ^ { A + 1 } \big ( x ^ { A + 1 } ( t ) \big ) - x ^ { A } ( t ) \bigg ) , } \end{array}\tag{A3}
$$

(A4)

where $\nabla _ { y ^ { A } }$ are the gradient operators with respect to $y ^ { A }$ , and we used the fact that $( \xi ^ { ( A + 1 , A ) } ) ^ { \top } = \xi ^ { ( A , A + 1 ) }$ . The energy function of the class $\mathcal { H } _ { k }$ , which serves as a Lyapunov function of the dynamical system, is defined by

$$
\begin{array} { l } { { \displaystyle E _ { { \mathcal { H } } _ { k } } \big ( \big \{ x ^ { A } \big \} \big ) : = E ( \big \{ y ^ { A } \big \} , \big \{ x ^ { A } \big \} ) \big \vert _ { \{ y ^ { A } = g ^ { A } ( x ^ { A } ) \} _ { A = 1 } ^ { k } } } } \\ { { \displaystyle \qquad = \sum _ { A = 1 } ^ { k } \frac { 1 } { \tau _ { A } } \big ( ( x ^ { A } ) ^ { \top } g ^ { A } ( x ^ { A } ) - L ^ { A } ( x ^ { A } ) \big ) - \sum _ { A = 1 } ^ { k - 1 } \frac { \lambda _ { A + 1 , A } } { \tau _ { A + 1 } \tau _ { A } } \big ( g ^ { A + 1 } ( x ^ { A + 1 } ) \big ) ^ { \top } \xi ^ { ( A + 1 , A ) } g ^ { A } ( x ^ { A } ) . } } \end{array}\tag{A5}
$$

Since the position sector of the canonical equations is replaced by the constraint $y ^ { A } = g ^ { A } ( x ^ { A } )$ , the flow is not symplectic, and the energy is not conserved along the trajectory. Instead, provided that the Hessians of the Lagrangians are positive (semi-)definite, this energy function monotonically decreases along the solution trajectory of the dynamica equations,

$$
\frac { d E _ { \mathcal { H } _ { k } } ( \{ x ^ { A } ( t ) \} ) } { d t } = - \sum _ { A = 1 } ^ { k } \biggl ( \frac { d x ^ { A } ( t ) } { d t } \biggr ) ^ { \top } \mathrm { H e s s } L ^ { A } ( x ^ { A } ( t ) ) \frac { d x ^ { A } ( t ) } { d t } \leq 0 ,\tag{A6}
$$

which exhibits the dissipative nature of the system. $\operatorname { I f } ,$ in addition, the overall energy function is bounded from below, the trajectory is guaranteed to converge to a fixed-point attractor state, which corresponds to one of the local minima of the energy function. Such fixed points may be identified with the stored memories, and the convergence toward them with memory retrieval. Setting $k = 2$ and identifying $x ^ { 1 } = v$ and $x ^ { 2 } = h$ reproduces the system discussed in the main text. For the two-layer $\left( k = 2 \right)$ case, as in the main text, we drop the subscript k and refer to the model as the class-H associative memories.

## Appendix B: Details on Models A and C

## 1. A family of associative memories that generates Models A and C

In this appendix we characterize the family of visible Lagrangians underlying the construction of Sec. III and explain in what sense Models A and C are its distinguished members. Throughout, the hidden sector is kept general, and it reenters only in the final remark.

Euler’s identity and scale invariance. In the energy function Eq. (4), the visible neurons enter both through the Legendre term $E _ { v } ( v ) = v ^ { \top } g ( v ) - L _ { v } ( v )$ and through the activation $g ( v )$ in the interaction term. The bare visible neurons drop out of the energy precisely when the Legendre term vanishes identically,

$$
\boldsymbol { v } ^ { \intercal } \nabla L _ { v } ( v ) - L _ { v } ( v ) = 0 ,\tag{B1}
$$

in which case the energy depends on v only through $g ( v )$ , as observed in Sec. III. Equation (B1) is Euler’s identity, and by Euler’s theorem on homogeneous functions it holds on an open cone on which $L _ { v }$ is diferentiable if and only if $L _ { v }$ is positively homogeneous of degree one there [43],

$$
L _ { v } ( t v ) = t L _ { v } ( v ) , \qquad t > 0 .\tag{B2}
$$

Diferentiating $\operatorname { E q }$ . (B2) with respect to v shows that the activation is then homogeneous of degree zero, $g ( t v ) = g ( v ) { : }$ it is a pure readout of the direction of $v ,$ insensitive to its scale. This is the family-wide origin of the scale invariance of the energy noted in Sec. III.

Degree-one homogeneity fixes $L _ { v }$ along each ray from the origin once its value on a single cross-section is given. Choosing the Euclidean unit sphere as the cross-section yields the representation

$$
L _ { v } ( v ) = \| v \| \phi \biggl ( \frac { v } { \| v \| } \biggr ) , \qquad \phi \colon \mathbb { S } ^ { N _ { v } - 1 } \to \mathbb { R } ,\tag{B3}
$$

where ∥·∥ is the Euclidean norm. Conversely, every function $\phi$ on the unit sphere defines a positively homogeneous $L _ { v }$ through Eq. (B3). No diferentiability is needed for the representation itself. Smoothness enters only through the correspondence that $L _ { v }$ is $C ^ { 1 }$ away from the origin exactly when $\phi \in C ^ { 1 } ( \mathbb { S } ^ { N _ { v } - 1 } )$ , which is what defines the activation $g = \nabla L _ { v }$ there. In particular, ϕ need not be an elementary function, so the family is genuinely infinite-dimensional. In two dimensions, Eq. (B3) is simply $L _ { v } = r \phi ( \theta )$ in polar coordinates, and the convexity requirement introduced next takes the classical form $\phi ( \theta ) + \phi ^ { \prime \prime } ( \theta ) \ge 0$ of the support-function condition [44].

Convexity and support functions. Within the class H, the Lagrangians are not arbitrary: the monotonic decrease of the energy (the defining property of an associative memory) requires positive semi-definite Hessians, i.e., the Lagrangians must be convex, though not necessarily strictly. For the nondiferentiable members below, convexity itself is the appropriate formulation of the same requirement. A convex, everywhere finite, positively degree-one homogeneous function is precisely a sublinear function, and every sublinear function is the support function of a uniquely determined compact convex set $K \subset \mathbb { R } ^ { N _ { v } } \ \lvert 4 5$ , Sec. 13],

$$
L _ { v } ( v ) = h _ { K } ( v ) : = \operatorname* { m a x } _ { u \in K } u ^ { \top } v , \qquad K = \partial L _ { v } ( 0 ) ,\tag{B4}
$$

where $\partial L _ { v } ( 0 )$ denotes the subdiferential at the origin. The family underlying Models A and C is thus classified by a convex body: the choice of K determines the model. Moreover, wherever $L _ { v }$ is diferentiable, the gradient is the unique point of K at which the linear function $u \mapsto u ^ { \top }$ v attains its maximum [45, Sec. 25], so that

$$
g ( v ) = \nabla h _ { K } ( v ) \in \partial K .\tag{B5}
$$

The efective visible degree of freedom is therefore the activation $u = g ( v )$ , which lives on the boundary of $K \colon$ the direction of v determines the point of ∂K at which the hyperplane with normal v supports $K .$

Since the energy is exactly constant along each open ray $\{ t v : t > 0 \}$ , the radial direction is a flat direction (an exact zero mode) of $\beta E _ { \xi }$ for every member of the family, and the naive visible integral in Eq. (7) diverges in proportion to the infinite radial volume. This divergence is an overall factor, independent of the patterns and of all order parameters, so it factors out of every observable. Removing it in the standard collective-coordinate manner leaves an integral over the space of rays. Any cross-section of the rays represents this quotient, and the natural choice is provided by the activation itself: the visible trace becomes an integral over $u = g ( v ) \in \partial K$ , rescaled so that the components are $O ( 1 )$ as in the main text. This is the general form of the prescription anticipated in Sec. II B and adopted for the family in $\operatorname { E q . }$ (14). One point deserves emphasis: the quotient fixes the domain of the visible trace but not its measure, which must be supplied as part of the model definition. For polytopes (including $p = 1$ , ∞ below) the activation takes finitely many values and the counting measure is canonical, and for $p = 2$ the rotation-invariant measure on the sphere is singled out by symmetry. For the other members inequivalent natural choices coexist (e.g., the surface measure and the cone measure on the dual sphere difer by an explicit density [46]), and in $\operatorname { E q } .$ . (14) the standard surface measure is understood.

The $\ell _ { p }$ family. For $L _ { v } = \| v \| _ { p }$ with $1 \leq p \leq \infty$ , the maximum in $\mathrm { E q . ~ ( B 4 ) }$ is Hölder’s inequality $u ^ { \top } v \leq \| u \| _ { p ^ { \prime } } \| v \| _ { p }$ together with its equality condition, attained precisely at the activation given in Sec. III, and the body is the unit ball of the dual norm,

$$
K = B _ { p ^ { \prime } } : = \Big \{ u \in \mathbb { R } ^ { N _ { v } } \Big | \left\| u \right\| _ { p ^ { \prime } } \leq 1 \Big \} , \qquad { \frac { 1 } { p } } + { \frac { 1 } { p ^ { \prime } } } = 1 ,\tag{B6}
$$

whose boundary carries the constraint $\| g \| _ { p ^ { \prime } } = 1$ of Sec. III. Three cases deserve mention. ${ \mathrm { ( i ) ~ } } p = 2 \colon$ K is the Euclidean ball, which is self-dual. The activation $g ( \dot { v } ) = v / \| v \|$ sweeps the whole unit sphere, the visible degrees of freedom are spherical spins, and the member is Model C. (ii) $\dot { p } = 1 \colon \bar { K } = [ - 1 , 1 ] ^ { N _ { v } }$ is the hypercube, the dual ball of $\ell _ { \infty }$ . Of the coordinate hyperplanes, the activation is $g ( v ) = \operatorname { s g n } ( v )$ componentwise, whose image is the finite vertex set $\left\{ \pm 1 \right\} ^ { N _ { v } }$ The visible degrees of freedom are Ising spins, and the member is Model A, a discrete spin model arising from continuous neurons through a piecewise linear Lagrangian. (iii) $p = \infty \colon K = \mathrm { c o n v } \{ \pm e _ { 1 } , \ldots , \pm e _ { N _ { v } } \}$ is the cross-polytope, with $e _ { i }$ the standard basis vectors. Wherever the largest component $| v _ { i ^ { * } } |$ is unique, $g ( v ) = \mathrm { s g n } ( v _ { i ^ { * } } ) e _ { i ^ { * } }$ , so the visible layer collapses to a single winner-take-all (one-hot) degree of freedom with $2 N _ { v }$ states.

Cases (ii) and (iii) illustrate a general rule. Abbreviating the local field acting on the visible layer in Eq. (14) by $\begin{array} { r } { c _ { i } : = { { \frac { \beta \lambda } { \tau _ { h } \tau _ { v } } } \sum _ { \mu } F ^ { \prime } ( h _ { \mu } ) \xi _ { \mu i } ^ { ( h , v ) } } } \end{array}$ , the visible trace takes the form $\begin{array} { r } { \int _ { S } d \Omega ( \mathbf { x } ) e ^ { c ^ { \top } \mathbf { x } } : } \end{array}$ : a Laplace-type transform of the chosen measure on $\partial K$ . If K is a polytope with vertices $a _ { 1 } , \dots , a _ { M }$ , then $h _ { K } ( v ) = \mathrm { m a x } _ { 1 \leq l \leq M } a _ { l } ^ { \top } v$ , the activation is piecewise constant with values in the vertex set, and the visible trace is a finite exponential sum,

$$
\int _ { S } d \Omega ( { \bf x } ) e ^ { c ^ { \top } { \bf x } } \longrightarrow \sum _ { l = 1 } ^ { M } w _ { l } e ^ { c ^ { \top } a _ { l } } ,\tag{B7}
$$

with weights w<sub>l</sub> given by the counting measure and the vertices rescaled according to the radius convention of Eq. (14). Discrete spin models thus arise within the class H as the polyhedral shapes $K ,$ with no need to postulate discrete variables at the outset. For the hypercube the sum factorizes over the sites, $\begin{array} { r } { \sum _ { \mathbf { x } \in \{ \pm 1 \} ^ { N _ { v } } } e ^ { c ^ { \top } \mathbf { x } } = \prod _ { i } \mathopen { } \mathclose \bgroup \left( e ^ { + } \aftergroup \egroup \right) } \end{array}$ 2 cosh $c _ { i } ,$ , which is the elementary identity behind Eq. (17). For the cross-polytope it is a single sum over sites, $\textstyle \sum _ { i } ( e ^ { c _ { i } } + e ^ { - c _ { i } } )$ up to the radius normalization. For $p = 2$ , on the other hand, the trace is rotation invariant and is evaluated exactly by Gaussian methods in Appendix B 3. Within the $\ell _ { p }$ family, $p = 1$ and $p = 2$ are therefore the two members whose visible trace closes in elementary terms at every $\bar { N _ { v } } .$ . This is the quantitative content of the footnote in Sec. III and the reason the main text focuses on Models A and C.

For the remaining members, general $1 < p < \infty ,$ , no elementary closed form of the visible trace is known, but three standard representations organize the available results. (i) Probabilistic representation. If $\boldsymbol { Y } = ( Y _ { 1 } , \ldots , Y _ { { N _ { v } } } )$ has i.i.d. components with density proportional to $e ^ { - | t | ^ { p ^ { \prime } } }$ , then $Y / \| Y \| _ { p ^ { \prime } }$ is distributed according to the cone measure on the unit dual sphere $\partial B _ { p ^ { \prime } }$ and is independent of $\| Y \| _ { p ^ { \prime } } \ [ 4 7 ] ;$

$$
u \triangleq { \frac { Y } { \| Y \| _ { p ^ { \prime } } } } , \qquad Y _ { i } \sim { \frac { e ^ { - | t | ^ { p ^ { \prime } } } } { 2 \Gamma ( 1 + 1 / p ^ { \prime } ) } } d t \quad { \mathrm { i . i . d . } } ,\tag{B8}
$$

and the same holds for the surface measure after inserting an explicit density [46]. This trades the hard constraint for $N _ { v }$ independent single-site variables coupled only through the norm $\| Y \| _ { p ^ { \prime } }$ (the $\ell _ { p ^ { \prime } }$ analogue of representing the uniform measure on the sphere by a Gaussian vector conditioned on its radius), and is the natural starting point for a saddle-point evaluation of the trace at large $N _ { v }$ . (ii) Mellin–Barnes representation. Resolving the residual radial constraint in Eq. (B8) by a Mellin–Barnes contour integral expresses the trace as a single contour integra over products of Gamma functions, i.e., a representation of Fox-H type, from which asymptotics can be extracted by shifting the contour [48]. (iii) Confluent-hypergeometric special case. When the local field has a single non-vanishing component, $c \parallel e _ { i }$ , the trace collapses to the one-dimensional Beta-family integral $\begin{array} { r } { \int _ { - 1 } ^ { 1 } e ^ { c t } ( 1 - \left| t \right| ^ { p ^ { \prime } } ) ^ { ( N _ { v } - 1 ) / p ^ { \prime } - 1 } d t } \end{array}$ (the one-coordinate marginal of the cone measure), whose term-by-term expansion is a confluent series of Kummer type. For $p ^ { \prime } = 2$ it is precisely a confluent hypergeometric function ${ } _ { 1 } F _ { 1 }$ , equivalently a modified Bessel function [49, Ch. 13], and for rational $p ^ { \prime }$ it reduces to finite combinations of generalized hypergeometric functions [48]. None of these yields an elementary closed form at a generic field: Models A and C are the two exactly solvable members of the family, between which the other $\ell _ { p }$ members interpolate.

Remarks. Three structural comments conclude this subsection. (i) The origin is necessarily singular. If $L _ { v }$ is diferentiable at the origin, the subdiferential $K = \partial L _ { v } ( 0 )$ reduces to the single point $\nabla L _ { v } ( 0 )$ [45, Sec. 25], and Eq. (B4) gives $\boldsymbol { L _ { v } } ( \boldsymbol { v } ) = \boldsymbol { \bar { \nabla } } L _ { v } ( 0 ) ^ { \top } \boldsymbol { \imath }$ : the Lagrangian is linear, the activation is a constant vector, and the interaction energy no longer depends on the visible configuration: no memory is stored. Every nontrivial member of the family is therefore non-diferentiable at $v = 0 \mathrm { : }$ : the kinks of $\left\| \cdot \right\| _ { 1 }$ on the coordinate hyperplanes and the conical point of $\left\| \cdot \right\| _ { 2 }$ at the origin are structural features of scale-free activations rather than accidental features of the two examples. Physically, $v = 0$ carries no direction information, and a pure direction readout cannot be continuously defined there. (ii) Norms and beyond. The support function $h _ { K }$ is a norm exactly when $K$ is centrally symmetric with the origin in its interior [44]. This is the case relevant to the ±-symmetric patterns of the main text. General members allow asymmetric bodies, e.g., $h _ { K } ( v ) = \operatorname* { m a x } _ { l } a _ { l } ^ { \top } v$ with a generic vertex set, for which $L _ { v } ( - v ) \neq L _ { v } ( v )$ and patterns and anti-patterns are treated asymmetrically. The family is in one-to-one correspondence with compact convex bodies, of which the $\ell _ { p }$ balls form a one-parameter slice. This is the precise content of the footnote in Sec. III. (iii) The hidden layer. The criterion Eq. (B1) applies equally to $L _ { h } \mathbf { . }$ : a degree-one homogeneous hidden Lagrangian would remove the bare hidden neurons from Eq. (4) as well. The choice $\begin{array} { r } { \dot { L _ { h } } = \sum _ { \mu } F ( h _ { \mu } ) } \end{array}$ with $F ( x ) = x ^ { k } / k$ is homogeneous of degree $k \neq 1 ,$ so the hidden Legendre term survives (it is precisely the confining term $\begin{array} { r } { N _ { v } \gamma _ { k } \sum _ { \mu } m _ { \mu } ^ { k } } \end{array}$ of Eq. (17)), and the bare hidden neurons remain in the energy, which allows them to act as the order parameters of memory retrieval.

## 2. Model A replica analysis

In this appendix we derive the RS free energy Eq. (21) and the equations of state Eq. (23)–Eq. (25) of Model A. We follow the standard procedure of the replica method [50, 51]: after averaging the replicated partition function over the patterns, the crosstalk noise is characterized by an overlap matrix R and its conjugate ${ \hat { R } } ,$ and the hidden-sector integral factorizes over the non-condensed modes. For $k = 2$ the hidden sector is Gaussian, the computation closes exactly, and it reproduces the AGS theory of the Hopfield model [2–4]. For $k > 2 .$ , in contrast, we show that the non-condensed hidden integral admits no consistent evaluation at finite $\beta _ { k }$ (a genuine pathology of the continuous hidden variables at the load $\tilde { N _ { h } } = \alpha _ { k } N _ { v } ^ { k - 1 }$ , not a technical artifact), so that the analysis must be based on the adiabatic reduction of Sec. II B, for which we then carry out the corrected computation.

## a. Replica setup and order parameters

Throughout we work with the normalized hidden variables $m _ { \mu } = h _ { \mu } / ( \lambda / \tau _ { v } )$ introduced in the main text, and the Jacobian of this change of variables only contributes an irrelevant constant. Introducing the replica index $a = 1 , \ldots , n$ and averaging Eq. (17) over the patterns $\xi _ { \mu i } ^ { ( h , v ) } \sim \operatorname { U n i f } ( \{ \pm 1 \} )$ ), which are independent across the sites $i ,$ we have

$$
\mathbb { E } _ { \xi } \left[ ( Z _ { \xi } ^ { \mathtt { A } } ) ^ { n } \right] = \int \prod _ { a = 1 } ^ { n } d m ^ { a } e ^ { - N _ { v } \gamma _ { k } \sum _ { \mu , a } ( m _ { \mu } ^ { a } ) ^ { k } } \prod _ { i = 1 } ^ { N _ { v } } \mathbb { E } _ { \xi _ { i } } \left[ \prod _ { a = 1 } ^ { n } 2 \cosh \beta _ { k } \bigg ( \sum _ { \mu } \xi _ { i \mu } ^ { ( v , h ) } ( m _ { \mu } ^ { a } ) ^ { k - 1 } \bigg ) \right] ,\tag{B9}
$$

where $\xi _ { i }$ denotes the i-th row of the pattern matrix. Since the pattern distribution is invariant under $\xi _ { i \mu } ^ { ( v , h ) } \to \xi _ { i 1 } ^ { ( v , h ) } \xi _ { i \mu } ^ { ( v , h ) }$ for each i, we may set $\xi _ { i 1 } ^ { ( v , h ) } = 1$ for all i without loss of generality. To describe the retrieval phase, we assume that only the first pattern is condensed,

$$
m ^ { a } : = m _ { 1 } ^ { a } = O ( 1 ) , \qquad m _ { \mu } ^ { a } = O ( N _ { v } ^ { - 1 / 2 } ) \quad ( \mu \geq 2 ) ,\tag{B10}
$$

and collect the non-condensed contributions to the local field into the crosstalk noise

$$
u _ { i } ^ { a } : = \sum _ { \mu \geq 2 } \xi _ { i \mu } ^ { ( v , h ) } ( m _ { \mu } ^ { a } ) ^ { k - 1 } .\tag{B11}
$$

For fixed $\{ m _ { \mu } ^ { a } \}$ , the vectors $u _ { i } = ( u _ { i } ^ { a } ) _ { a = 1 } ^ { n }$ are i.i.d. across the sites with zero mean, and by the central limit theorem they become Gaussian at large $N _ { v }$

$$
u _ { i } \stackrel { \mathrm { d } } {  } { \mathcal { N } } ( 0 , R ) , \qquad R ^ { a b } : = \sum _ { \mu \geq 2 } ( m _ { \mu } ^ { a } ) ^ { k - 1 } ( m _ { \mu } ^ { b } ) ^ { k - 1 } ,\tag{B12}
$$

where the covariance matrix R is of order $N _ { h } \times N _ { v } ^ { - ( k - 1 ) } = \alpha _ { k } = O ( 1 )$ on the assumed scaling of the non-condensed modes. The site average in Eq. (B9) thus becomes

$$
\mathbb { E } _ { \xi _ { i } } \left[ \prod _ { a } 2 \cosh \beta _ { k } \big ( ( m ^ { a } ) ^ { k - 1 } + u _ { i } ^ { a } \big ) \right] \to \Psi ( m , R ) : = \mathbb { E } _ { z \sim \mathcal { N } ( 0 , R ) } \left[ \prod _ { a } 2 \cosh \beta _ { k } \big ( ( m ^ { a } ) ^ { k - 1 } + z ^ { a } \big ) \right] .\tag{B13}
$$

We next promote $R ^ { a b }$ to independent integration variables by inserting the identity

$$
\begin{array} { l } { { \displaystyle 1 = \int \prod _ { a , b } d R ^ { a b } \delta \bigg ( R ^ { a b } - \sum _ { \mu \ge 2 } ( m _ { \mu } ^ { a } ) ^ { k - 1 } ( m _ { \mu } ^ { b } ) ^ { k - 1 } \bigg ) } } \\ { { \displaystyle \propto \int \prod _ { a , b } d R ^ { a b } d \hat { R } ^ { a b } \exp \bigg \{ - \frac { \beta _ { k } ^ { 2 } N _ { v } } { 2 } \sum _ { a , b } \hat { R } ^ { a b } \bigg ( R ^ { a b } - \sum _ { \mu \ge 2 } ( m _ { \mu } ^ { a } ) ^ { k - 1 } ( m _ { \mu } ^ { b } ) ^ { k - 1 } \bigg ) \bigg \} , } } \end{array}\tag{B14}
$$

where the conjugate variables ${ \hat { R } } ^ { a b }$ run along the imaginary axis and take real values at the saddle point, and the prefactor $\beta _ { k } ^ { 2 } N _ { v } / 2$ is chosen for later convenience. The integrals over the non-condensed modes then factorize,

$$
\prod _ { \mu = 2 } ^ { N _ { h } } \int \prod _ { a } d m _ { \mu } ^ { a } \exp \bigg [ - N _ { v } \gamma _ { k } \sum _ { a } ( m _ { \mu } ^ { a } ) ^ { k } + \frac { \beta _ { k } ^ { 2 } N _ { v } } { 2 } \sum _ { a , b } \hat { \cal R } ^ { a b } ( m _ { \mu } ^ { a } ) ^ { k - 1 } ( m _ { \mu } ^ { b } ) ^ { k - 1 } \bigg ] = { \cal T } _ { k } ( \hat { \cal R } ) ^ { N _ { h } - 1 } ,\tag{B15}
$$

with the single-mode integral

$$
\mathcal { Z } _ { k } ( \hat { R } ) : = \int _ { \mathbb { R } ^ { n } } \prod _ { a } d x ^ { a } \exp \bigg [ - N _ { v } \gamma _ { k } \sum _ { a } ( x ^ { a } ) ^ { k } + \frac { \beta _ { k } ^ { 2 } N _ { v } } { 2 } \sum _ { a , b } \hat { R } ^ { a b } ( x ^ { a } ) ^ { k - 1 } ( x ^ { b } ) ^ { k - 1 } \bigg ] ,\tag{B16}
$$

and the replicated partition function reads

$$
\mathbb { E } _ { \xi } \big [ ( Z _ { \xi } ^ { \mathtt { A } } ) ^ { n } \big ] = \int \prod _ { a } d m ^ { a } \int \prod _ { a , b } d R ^ { a b } d \hat { R } ^ { a b } \exp \bigg [ - N _ { v } \gamma _ { k } \sum _ { a } ( m ^ { a } ) ^ { k } - \frac { \beta _ { k } ^ { 2 } N _ { v } } { 2 } \sum _ { a , b } \hat { R } ^ { a b } R ^ { a b } + N _ { v } \log \Psi ( m , R ) \bigg ] Z _ { k } ( \hat { R } ) ^ { N _ { h } - 1 } .\tag{B17}
$$

We now impose the RS ansat $^ { \mathrm { ~ , ~ Z ~ , ~ } }$

$$
m ^ { a } = m , \quad \quad R ^ { a a } = \alpha _ { k } r _ { d } , \quad R ^ { a b } = \alpha _ { k } r ( a \neq b ) , \quad \quad \hat { R } ^ { a a } = \hat { r } _ { d } , \quad \hat { R } ^ { a b } = \hat { r } ( a \neq b ) ,\tag{B18}
$$

where the factors $\alpha _ { k }$ are inserted so that r matches the noise parameter of the main text. To evaluate Ψ, we realize the Gaussian vector $z \sim \mathcal { N } ( 0 , R )$ as

$$
z ^ { a } = \sqrt { \alpha _ { k } r } z + \sqrt { \alpha _ { k } ( r _ { d } - r ) } w ^ { a } , \qquad z , w ^ { a } \sim \mathcal { N } ( 0 , 1 ) \ \mathrm { i . i . d . } ,\tag{B19}
$$

which decomposes the noise into a component z frozen across the replicas and thermal components $w ^ { a }$ independent between the replicas. Using R Dw $2 \cosh ( A + B w ) = 2 e ^ { B ^ { 2 } / 2 }$ cosh A and expanding to first order in n, we obtain

$$
\log \Psi ( m , R ) = n \bigg [ \frac { \beta _ { k } ^ { 2 } \alpha _ { k } } { 2 } ( r _ { d } - r ) + \int D z \log 2 \cosh \beta _ { k } \big ( m ^ { k - 1 } + \sqrt { \alpha _ { k } r } z \big ) \bigg ] + O ( n ^ { 2 } ) .\tag{B20}
$$

Similarly, the conjugate term in Eq. (B17) becomes

$$
- \frac { \beta _ { k } ^ { 2 } N _ { v } } { 2 } \sum _ { a , b } \hat { R } ^ { a b } R ^ { a b } = - \frac { \beta _ { k } ^ { 2 } N _ { v } \alpha _ { k } } { 2 } \big [ n \hat { r } _ { d } r _ { d } + n ( n - 1 ) \hat { r } r \big ] .\tag{B21}
$$

Two of the saddle-point equations can already be read of, because $\mathcal { T } _ { k }$ depends only on the conjugate variables. Stationarity of the exponent with respect to $r _ { d } \ { \mathrm { g i v e s } } ,$ from Eq. (B20) and the conjugate term,

$$
{ \frac { \beta _ { k } ^ { 2 } \alpha _ { k } } { 2 } } - { \frac { \beta _ { k } ^ { 2 } \alpha _ { k } } { 2 } } \hat { r } _ { d } = 0 \qquad \Longrightarrow \qquad \hat { r } _ { d } = 1 .\tag{B22}
$$

Substituting $\hat { r } _ { d } = 1$ back, the two r<sub>d</sub>-dependent terms cancel identically, so that $r _ { d }$ and $\hat { r } _ { d }$ disappear from the free energy altogether. The value of $r _ { d }$ itself is fixed by stationarity with respect to ${ \hat { r } } _ { d } ,$ which identifies $\alpha _ { k } r _ { d }$ with the average of $\breve { \sum } _ { \mu \geq 2 } ( m _ { \mu } ^ { a } ) ^ { 2 ( k - 1 ) }$ in the hidden sector, but it plays no further role. Stationarity with respect to r gives, using Gaussian integration by parts $\begin{array} { r } { \int D z z \operatorname { t a n h } \beta _ { k } ( m ^ { k - 1 } + \sqrt { \alpha _ { k } r } z ) = \beta _ { k } \sqrt { \alpha _ { k } r } \int D z \operatorname { s e c h } ^ { 2 } \beta _ { k } ( m ^ { k - 1 } + \sqrt { \alpha _ { k } r } z ) } \end{array}$ 2

$$
\hat { r } = \int D z \operatorname { t a n h } ^ { 2 } \beta _ { k } \big ( m ^ { k - 1 } + \sqrt { \alpha _ { k } r } z \big ) = q .\tag{B23}
$$

The conjugate variable rˆ is therefore precisely the Edwards–Anderson order parameter q of the visible spins $s _ { i } = \operatorname { s g n } ( v _ { i } )$ the integrand of Eq. (B23) is the squared thermal average $\left. s _ { i } \right. ^ { 2 }$ in the efective single-site measure. Note that Eq. (B22) and Eq. (B23) hold for every k. What distinguishes $k = 2$ from $k > 2$ is the remaining hidden-sector factor $\mathcal { T } _ { k } ( \hat { R } )$ , to which we now turn.

## b. The case k = 2: Gaussian hidden sector and the AGS theory

For $k = 2$ we have $\gamma _ { 2 } = \beta _ { k } / 2$ , and the single-mode integral Eq. (B16) is Gaussian:

$$
\mathcal { Z } _ { 2 } ( \hat { R } ) = \int _ { \mathbb { R } ^ { n } } \prod _ { a } d x ^ { a } \exp \bigg [ - \frac { \beta _ { k } N _ { v } } { 2 } \sum _ { a , b } x ^ { a } \big ( \delta ^ { a b } - \beta _ { k } \hat { R } ^ { a b } \big ) x ^ { b } \bigg ] \propto \operatorname* { d e t } \big ( I _ { n } - \beta _ { k } \hat { R } \big ) ^ { - 1 / 2 } ,\tag{B24}
$$

up to an irrelevant constant. Under the RS ansatz, $\hat { R }$ has the eigenvalue $\hat { r } _ { d } + ( n - 1 ) \hat { r }$ (non-degenerate) and $\hat { r } _ { d } - \hat { r }$ with degeneracy $n - 1$ , so that

$$
\log \operatorname* { d e t } \left( I _ { n } - \beta _ { k } { \hat { R } } \right) = ( n - 1 ) \log \left( 1 - \beta _ { k } { \left( { \hat { r } } _ { d } - { \hat { r } } \right) } \right) + \log \left( 1 - \beta _ { k } { \left( { \hat { r } } _ { d } - { \hat { r } } \right) } - n \beta _ { k } { \hat { r } } \right) .\tag{B25}
$$

Expanding to first order in n and inserting $\hat { r } _ { d } = 1 , \hat { r } = q$ , we obtain

$$
\log \mathcal { Z } _ { 2 } ( \hat { R } ) = - \frac { n } { 2 } \bigg [ \log \big ( 1 - \beta _ { k } ( 1 - q ) \big ) - \frac { \beta _ { k } q } { 1 - \beta _ { k } ( 1 - q ) } \bigg ] + O ( n ^ { 2 } ) = - \frac { n } { 2 } \Psi _ { 2 } ( q ) + O ( n ^ { 2 } ) ,\tag{B26}
$$

which, multiplied by $N _ { h } - 1 \simeq \alpha _ { k } N _ { v }$ , yields precisely the noise entropic term $- \frac { n N _ { v } \alpha _ { k } } { 2 } \Psi _ { 2 } ( q )$ of the main text. Collecting Eq. (B20)–Eq. (B26) in Eq. (B17) and using the replica trick

$$
f ^ { \mathrm { A } } ( \beta ) = - \frac 1 { \beta N _ { v } } \operatorname* { l i m } _ { n \to 0 } \frac 1 n \Big ( \mathbb { E } _ { \xi } \big [ ( Z _ { \xi } ^ { \mathrm { A } } ) ^ { n } \big ] - 1 \Big ) ,\tag{B27}
$$

we arrive at the RS free energy Eq. (21) with $\Psi _ { k } = \Psi _ { 2 }$ . In the assembly, the diagonal terms combine as $\begin{array} { r l r } { \mathrm { ~ } } & { { } } & { \frac { \beta _ { k } ^ { 2 } \alpha _ { k } } { 2 } ( r _ { d } - \hat { r } _ { d } r _ { d } ) = } \end{array}$ 0, which is the cancellation of $r _ { d }$ anticipated above, while the of-diagonal terms produce $\frac { \alpha _ { k } \beta _ { k } ^ { 2 } } { 2 } r ( 1 - q )$ upon using $\hat { r } = q$ . Finally, stationarity with respect to rˆ picks up contributions from the conjugate term and from log $\mathcal { T } _ { 2 } \colon$

$$
\frac { \beta _ { k } ^ { 2 } \alpha _ { k } } { 2 } r = - \alpha _ { k } \frac { \partial } { \partial \hat { r } } \operatorname* { l i m } _ { n \to 0 } \frac { 1 } { n } \log \bar { Z } _ { 2 } = \frac { \alpha _ { k } } { 2 } \frac { \beta _ { k } ^ { 2 } \hat { r } } { \big ( 1 - \beta _ { k } ( \hat { r } _ { d } - \hat { r } ) \big ) ^ { 2 } } \qquad \Longrightarrow \qquad r = \frac { q } { \big ( 1 - \beta _ { k } ( 1 - q ) \big ) ^ { 2 } } = \mathcal { M } _ { 2 } ( q ) ,\tag{B28}
$$

reproducing the AGS equations of state of the Hopfield model near saturation [2–4]. We emphasize that for $k = 2$ no adiabatic assumption has been made: the hidden variables enter quadratically, their Hessian is field-independent, and the Gaussian integration is exact at any $\beta / \tau _ { h }$

## c. The case $k > 2 :$ breakdown of the hidden-sector integral and the adiabatic reduction

For $k > 2$ the integral Eq. (B16) does not admit an analogous evaluation, for a reason that is structural rather than technical. Rescaling $x ^ { a } = N _ { v } ^ { c } y ^ { a }$ , the two terms in the exponent of $\mathcal { T } _ { k }$ scale as

$$
N _ { v } \gamma _ { k } \sum _ { a } ( x ^ { a } ) ^ { k } \sim N _ { v } ^ { 1 + k c } , \qquad \frac { \beta _ { k } ^ { 2 } N _ { v } } { 2 } \sum _ { a , b } \hat { R } ^ { a b } ( x ^ { a } ) ^ { k - 1 } ( x ^ { b } ) ^ { k - 1 } \sim N _ { v } ^ { 1 + 2 ( k - 1 ) c } .\tag{B29}
$$

For $k = 2$ the two exponents coincide for any $c ,$ and the choice $c = - 1 / 2$ makes both $O ( 1 )$ . The $N _ { h } - 1 \simeq \alpha _ { k } N _ { v }$ modes then sum up to an extensive contribution, which is the calculation of Appendix B 2 b. For $k > 2 .$ , however, no choice of c balances the two terms at a nontrivial order: at the central-limit scale $c = - 1 / 2$ , which underlies the ansatz $R = O ( 1 )$ in Eq. (B12), both exponents are negative and the integrand of $\mathcal { T } _ { k }$ becomes flat, so the integral is not confined to this scale. Instead, $\mathcal { T } _ { k }$ is dominated by the bare confinement scale $c = - 1 / k$ , on which the coupling term still vanishes as $N _ { v } ^ { ( 2 - k ) / k }$ . This has two consequences. First, the typical amplitude of a non-condensed mode is $| m _ { \mu } | \sim N _ { v } ^ { - 1 / k } \gg N _ { v } ^ { - 1 / 2 }$ , so that the diagonal noise covariance evaluates to

$$
R ^ { a a } = \sum _ { \mu \geq 2 } ( m _ { \mu } ^ { a } ) ^ { 2 ( k - 1 ) } \sim \alpha _ { k } N _ { v } ^ { k - 1 } \cdot N _ { v } ^ { - 2 ( k - 1 ) / k } = \alpha _ { k } N _ { v } ^ { ( k - 1 ) ( k - 2 ) / k } \longrightarrow \infty ,\tag{B30}
$$

in contradiction with $R = O ( 1 )$ : the crosstalk variance diverges, and the continuous hidden variables cannot sustain the load $N _ { h } = \alpha _ { k } N _ { v } ^ { k - 1 }$ at any finite $\beta _ { k }$ . Second, and equivalently, the $\scriptstyle { \hat { R } } \to \mathrm { d e p e n d e n t }$ part of $( N _ { h } - 1 )$ log $\mathcal { T } _ { k }$ is of order $N _ { v } ^ { ( k ^ { 2 } - 2 k + 2 ) / k }$ , which is superextensive for $k > 2 ,$ so no extensive saddle-point structure exists for the conjugate variables. We also note that this pathology cannot be removed by any limit of the time-scale parameters: restoring the original variables via $h = ( \lambda / \tau _ { v } )$ )m shows that the equilibrium partition function Eq. (14) depends on $\beta / \tau _ { h }$ and $\lambda / \tau _ { v }$ only through the combination $\dot { \beta } _ { k } = ( \beta / \tau _ { h } ) ( \lambda / \tau _ { v } ) ^ { k }$ , so that $^ { \mathfrak { a } } \beta / \tau _ { h } \to \infty$ at fixed $\beta _ { k } ^ { \mathrm { ~ } } { } ^ { \mathclose { ~ } }$ leaves the equilibrium measure unchanged.

For $k > 2$ we therefore define Model A through the adiabatic reduction of Sec. II B, in which the hidden neurons are replaced by their stationary values given the visible configuration, as in Eq. (10). This definition is motivated by the dynamics: in the adiabatic regime $\tau _ { v } \gg \tau _ { h }$ the hidden neurons deterministically track the visible configuration, and their thermal fluctuations (the source of the runaway Eq. (B30)) are switched of. Concretely, writing the partition function with the visible spins $s _ { i } \in \{ \pm 1 \}$ } unintegrated and eliminating each mode $m _ { \mu }$ by its stationarity condition, $k \gamma _ { k } m _ { \mu } ^ { k - 1 } = ( k - 1 ) \beta _ { k } m _ { \mu } ^ { k - 2 } \hat { m } _ { \mu }$ with $\begin{array} { r } { \hat { m } _ { \mu } ( s ) : = \frac { 1 } { N _ { v } } \sum _ { i } \xi _ { \mu i } ^ { ( h , v ) } s _ { i } } \end{array}$ , we obtain $m _ { \mu } = \hat { m } _ { \mu } ( s )$ and

$$
- N _ { v } \gamma _ { k } m _ { \mu } ^ { k } + \beta _ { k } N _ { v } m _ { \mu } ^ { k - 1 } \hat { m } _ { \mu } \Big | _ { m _ { \mu } = \hat { m } _ { \mu } } = \frac { \beta _ { k } } { k } N _ { v } \hat { m } _ { \mu } ^ { k } , \qquad Z _ { \xi } ^ { \mathrm { A , a d } } : = \sum _ { s \in \{ \pm 1 \} ^ { N _ { v } } } \exp { \bigg [ \frac { \beta _ { k } } { k } N _ { v } \sum _ { \mu } \hat { m } _ { \mu } ( s ) ^ { k } \bigg ] } ,\tag{B31}
$$

which is the dense associative memory with polynomial energy [10], also known as the multiconnected network [7]. For the condensed mode this substitution coincides with the exact Laplace evaluation of its integral, whose exponent is $O ( N _ { v } )$ with an $O ( 1 )$ saddle. For $k = 2$ it agrees with the exact Gaussian integration up to a constant, as noted above. The definition is therefore consistent across $k ,$ and the pathology is confined to the thermal fluctuations of the non-condensed continuous modes at $k > 2$

## d. The case $k > 2 \colon$ noise sector from the cumulant expansion

We now carry out the replica analysis of Eq. (B31). Replicating and averaging over the patterns, the condensed and non-condensed sectors factorize:

$$
\mathbb { E } _ { \xi } \Big [ ( Z _ { \xi } ^ { \mathrm { A } , \mathrm { a d } } ) ^ { n } \Big ] = \sum _ { \{ s ^ { a } \} } \exp \Bigg [ \frac { \beta _ { k } } { k } N _ { v } \sum _ { a } \hat { m } _ { 1 } ( s ^ { a } ) ^ { k } \Bigg ] \prod _ { \mu \geq 2 } \mathbb { E } _ { \xi _ { \mu } } \Bigg [ \exp \Bigg ( \varepsilon \sum _ { a } ( \hat { y } _ { \mu } ^ { a } ) ^ { k } \Bigg ) \Bigg ] , \qquad \varepsilon : = \frac { \beta _ { k } } { k } N _ { v } ^ { 1 - k / 2 } , \qquad \varepsilon \in \mathbb { R } ^ { n } .\tag{B32}
$$

where we used the gauge $\xi _ { i 1 } ^ { ( v , h ) } = 1$ and introduced the rescaled non-condensed overlaps

$$
\hat { y } _ { \mu } ^ { a } : = \frac { 1 } { \sqrt { N _ { v } } } \sum _ { i } \xi _ { \mu i } ^ { ( h , v ) } s _ { i } ^ { a } = \sqrt { N _ { v } } \hat { m } _ { \mu } ( s ^ { a } ) .\tag{B33}
$$

For fixed spin configurations, the vector $( \hat { y } _ { \mu } ^ { a } ) _ { a = 1 } ^ { n }$ is a normalized sum of i.i.d. bounded random variables and becomes jointly Gaussian at large $N _ { v }$ by the central limit theorem,

$$
( \hat { y } _ { \mu } ^ { a } ) _ { a } \stackrel { \mathrm { \tiny ~ d } } {  } { \mathcal N } ( 0 , Q ) , \qquad Q _ { a b } = q _ { a b } : = \frac { 1 } { N _ { v } } \sum _ { i } s _ { i } ^ { a } s _ { i } ^ { b } , \qquad q _ { a a } = 1 ,\tag{B34}
$$

with the replica-overlap matrix of the spins as its covariance. Since $\varepsilon  0$ for $k > 2$ , the per-pattern average in $\operatorname { E q . }$ (B32) is evaluated by the cumulant expansion

$$
\log \mathbb { E } _ { \xi _ { \mu } } \Big [ e ^ { \varepsilon \sum _ { a } ( \hat { y } ^ { a } ) ^ { k } } \Big ] = \varepsilon \sum _ { a } \mathbb { E } \big [ ( \hat { y } ^ { a } ) ^ { k } \big ] + \frac { \varepsilon ^ { 2 } } { 2 } \sum _ { a , b } \mathbf { C o v } \big ( ( \hat { y } ^ { a } ) ^ { k } , ( \hat { y } ^ { b } ) ^ { k } \big ) + O ( \varepsilon ^ { 3 } ) .\tag{B35}
$$

The first term is independent of the spin configurations: all single-replica cumulants of ${ \hat { y } } ^ { a }$ are spin-independent, because $\begin{array} { r } { \kappa _ { j } \big ( \hat { y } ^ { a } \big ) = N _ { v } ^ { - j / 2 } \kappa _ { j } ( \xi ) \sum _ { i } ( s _ { i } ^ { a } ) ^ { j } } \end{array}$ vanishes for odd j (symmetry of ξ) and reduces to $N _ { v } ^ { 1 - j / 2 } \kappa _ { j } ( \xi )$ for even ${ \mathrm { ~ \it ~ j ~ } } \left( s _ { i } ^ { 2 } = 1 \right)$ . It therefore contributes a state-independent constant (superextensive, of order $N _ { v } ^ { k / 2 }$ after summing over the patterns, but a pure constant) relative to which we define the free energy (see also Remark 2 below). The second term is governed by the covariance kernel

$$
\Phi _ { k } ( q ) : = { \mathrm { C o v } } { \bigl ( } X ^ { k } , Y ^ { k } { \bigr ) } , \qquad ( X , Y ) { \mathrm { ~ s t a n d a r d ~ G a u s s i a n ~ w i t h ~ c o r r e l a t i o n ~ } } \mathbb { E } [ X Y ] = q ,\tag{B36}
$$

evaluated at $q = q _ { a b }$ . Note that $\Phi _ { k } ( 0 ) = 0$ , and the diagonal terms $a = b$ contribute the q-independent constant nΦ<sub>k</sub>(1), which we drop. Summing over the $\dot { N } _ { h } - 1 \simeq \alpha _ { k } N _ { v } ^ { k - 1 }$ patterns, the noise sector contributes the extensive action

$$
\left( N _ { h } - 1 \right) \frac { \varepsilon ^ { 2 } } { 2 } \sum _ { a \neq b } \Phi _ { k } ( q _ { a b } ) = \frac { \alpha _ { k } \beta _ { k } ^ { 2 } } { 2 k ^ { 2 } } N _ { v } \sum _ { a \neq b } \Phi _ { k } ( q _ { a b } ) ,\tag{B37}
$$

while the third and higher cumulants are subextensive for even $k \geq 4 \ \mathrm { ( R e m a r k \ 2 ) }$ . In this representation the collapse of the noise-sector order parameters is manifest: the covariance $R ^ { a b }$ of $\operatorname { E q } .$ . (B12), evaluated on the enslaved modes $m _ { \mu } = \hat { m } _ { \mu } ( s )$ , self-averages by the law of large numbers to

$$
R ^ { a b } = N _ { v } ^ { - ( k - 1 ) } \sum _ { \mu \geq 2 } ( \hat { y } _ { \mu } ^ { a } ) ^ { k - 1 } ( \hat { y } _ { \mu } ^ { b } ) ^ { k - 1 } \longrightarrow \alpha _ { k } \mathbb { E } \left[ X ^ { k - 1 } Y ^ { k - 1 } \right] \Big | _ { q = q _ { a b } } = \alpha _ { k } \mathcal { M } _ { k } ( q _ { a b } ) ,\tag{B38}
$$

so that, in contrast to the $k = 2$ computation, the pair $( R , { \hat { R } } )$ carries no independent degrees of freedom: the crosstalk noise is entirely slaved to the spin overlap $q _ { a b } .$ . In particular the diagonal element is finite, $\bar { R ^ { a a } }  \alpha _ { k } \mathcal { M } _ { k } ( 1 ) = \alpha _ { k } ( 2 k { - } 3 ) ! !$ in contrast with the divergence Eq. (B30) of the continuous model

The remaining steps are standard. We introduce the condensed overlap and the spin overlaps with their conjugates,

$$
1 = \int \prod _ { a } d m ^ { a } \delta \Big ( N _ { v } m ^ { a } - \sum _ { i } s _ { i } ^ { a } \Big ) , \quad 1 = \int \prod _ { a < b } d q _ { a b } \delta \Big ( N _ { v } q _ { a b } - \sum _ { i } s _ { i } ^ { a } s _ { i } ^ { b } \Big ) ,\tag{B39}
$$

represented with conjugate variables $\tilde { m } ^ { a }$ and $\hat { q } _ { a b }$ as in Eq. (B14), upon which the spin trace factorizes over the sites. Under the RS ansatz $m ^ { a } = m , q _ { a b } = q , \hat { q } _ { a b } = \hat { q } \left( a < b \right)$ , the conjugate $\tilde { m } ^ { a } = \tilde { m }$ is eliminated by its saddle $\tilde { m } = \beta _ { k } m ^ { k - 1 }$ ， and the single-site trace gives the familiar

$$
\operatorname* { l i m } _ { n \to 0 } \frac { 1 } { n } \log \mathrm { T r } _ { s } \exp \left[ \frac { \hat { q } } { 2 } \Big ( \sum _ { a } s ^ { a } \Big ) ^ { 2 } - \frac { n \hat { q } } { 2 } + \tilde { m } \sum _ { a } s ^ { a } \right] = - \frac { \hat { q } } { 2 } + \int D z \log 2 \cosh \big ( \beta _ { k } m ^ { k - 1 } + \sqrt { \hat { q } } z \big ) .\tag{B40}
$$

Stationarity with respect to $\hat { q }$ returns $\begin{array} { r } { q = \int D z \operatorname { t a n h } ^ { 2 } ( \beta _ { k } m ^ { k - 1 } + \sqrt { \hat { q } } z ) } \end{array}$ , while stationarity with respect to $q$ ties the conjugate to the noise kernel:

$$
\hat { q } = \frac { \alpha _ { k } \beta _ { k } ^ { 2 } } { k ^ { 2 } } \Phi _ { k } ^ { \prime } ( q ) .\tag{B41}
$$

The derivative of the kernel is evaluated by Price’s theorem [52]: for jointly Gaussian (X, Y ) with correlation $q ,$

$$
\frac { \partial } { \partial q } \mathbb { E } \left[ X ^ { k } Y ^ { k } \right] = k ^ { 2 } \mathbb { E } \left[ X ^ { k - 1 } Y ^ { k - 1 } \right] .\tag{B42}
$$

A short proof follows from the Hermite expansion: writing $\begin{array} { r } { x ^ { m } = \sum _ { j } c _ { j } ^ { ( m ) } \mathrm { H e } _ { j } ( x ) } \end{array}$ with the probabilists’ Hermite polynomials ${ \mathrm { H e } } _ { j }$ and using the orthogonality $\begin{array} { r } { \mathbb { E } [ \mathrm { H e } _ { i } ( X ) \mathrm { H e } _ { j } ( Y ) ] = \delta _ { i j } j ! q ^ { j } } \end{array}$ , one has $\begin{array} { r } { \mathbb { E } [ X ^ { m } Y ^ { m } ] = \sum _ { j } ( c _ { j } ^ { ( m ) } ) ^ { 2 } j ! q ^ { j } } \end{array}$ Diferentiating term by term and using $k c _ { j - 1 } ^ { ( k - 1 ) } = j c _ { j } ^ { ( k ) }$ , which is the Hermite-coeficient form of $\textstyle { \frac { d } { d x } } x ^ { k } = k x ^ { k - 1 }$ together with $\mathrm { H e } _ { j } ^ { \prime } = j \mathrm { H e } _ { j - 1 }$ , yields Eq. (B42). Combining Eq. (B41) and Eq. (B42) and defining the noise parameter as in Eq. (B38),

$$
\begin{array} { r } { r : = \mathcal { M } _ { k } ( q ) = \mathbb { E } \big [ X ^ { k - 1 } Y ^ { k - 1 } \big ] , \qquad \hat { q } = \alpha _ { k } \beta _ { k } ^ { 2 } \mathcal { M } _ { k } ( q ) = \alpha _ { k } \beta _ { k } ^ { 2 } r , } \end{array}\tag{B43}
$$

the local field becomes $\beta _ { k } m ^ { k - 1 } + \sqrt { \hat { q } } z = \beta _ { k } ( m ^ { k - 1 } + \sqrt { \alpha _ { k } r } z )$ , matching the main text. Assembling all the terms at $n  0$ , the $\Phi _ { k }$ contribution enters the free energy as $\begin{array} { r } { \frac { \alpha _ { k } \beta _ { k } ^ { 2 } } { 2 k ^ { 2 } } \Phi _ { k } ( q ) } \end{array}$ , and since $\Phi _ { k } ( 0 ) = 0$ and $\Phi _ { k } ^ { \prime } = k ^ { 2 } \mathcal { M } _ { k }$

$$
\frac { \alpha _ { k } \beta _ { k } ^ { 2 } } { 2 k ^ { 2 } } \Phi _ { k } ( q ) = \frac { \alpha _ { k } \beta _ { k } ^ { 2 } } { 2 } \int _ { 0 } ^ { q } \mathcal { M } _ { k } ( s ) d s = \frac { \alpha _ { k } } { 2 } \Psi _ { k } ( q ) ,\tag{B44}
$$

which is precisely the noise entropic term of Eq. (21) for $k > 2$ . The terms $\gamma _ { k } m ^ { k }$ (from $\begin{array} { r } { \frac { \beta _ { k } } { k } m ^ { k } - \tilde { m } m } \end{array}$ at $\tilde { m } = \beta _ { k } m ^ { k - 1 } )$ and $\frac { \alpha _ { k } \beta _ { k } ^ { 2 } } { 2 } r ( 1 - q )$ (from $\textstyle { \frac { 9 } { 2 } } ( 1 - q ) )$ complete the RS free energy, up to the additive constant $- \frac { \alpha _ { k } \beta _ { k } ^ { 2 } } { 2 k ^ { 2 } } \Phi _ { k } ( 1 )$ and the state-independent constants discussed above. The equations of state Eq. (23)–Eq. (25) then follow from stationarity, as verified directly in the main text.

## e. Consistency checks and remarks

Remark 1: the Onsager reaction field. The two cases of $\mathcal { M } _ { k }$ can be cross-checked by a cavity argument that does not rely on replicas. In the efective spin model Eq. (B31), adding a single non-condensed mode µ shifts the local field on the spin $s _ { i }$ by $\beta _ { k } \xi _ { i \mu } ^ { ( v , h ) } \hat { m } _ { \mu } ^ { k - 1 }$ , and hence its thermal average by $\delta \langle s _ { i } \rangle = \beta _ { k } ( 1 - \langle s _ { i } \rangle ^ { 2 } ) \xi _ { i \mu } ^ { ( v , h ) } \hat { m } _ { \mu } ^ { k - 1 }$ in linear response. Feeding this back into the overlap of the same mode gives the self-feedback (Onsager reaction)

$$
\delta \hat { m } _ { \mu } = \frac { 1 } { N _ { v } } \sum _ { i } \xi _ { \mu i } ^ { ( h , v ) } \delta \langle s _ { i } \rangle = \beta _ { k } ( 1 - q ) \hat { m } _ { \mu } ^ { k - 1 } , \qquad \mathrm { i . e . } \qquad \delta \hat { y } _ { \mu } = \beta _ { k } ( 1 - q ) N _ { v } ^ { - ( k - 2 ) / 2 } \hat { y } _ { \mu } ^ { k - 1 } .\tag{B45}
$$

For $k = 2$ the feedback is marginal, $O ( 1 )$ , and must be resummed to all orders: with the frozen cavity field $\sqrt { q } z$ the self-consistent solution is $\langle \hat { y } \rangle _ { z } = \sqrt { q } z / ( 1 - \beta _ { k } ( 1 - q ) )$ , so that $\begin{array} { r } { r = \int D z \langle \hat { y } \rangle _ { z } ^ { 2 } = q / ( 1 - \beta _ { k } ( 1 - q ) ) ^ { 2 } = \mathcal { M } _ { 2 } ( q ) } \end{array}$ The geometric resummation is the origin of the AGS denominator, in agreement with Appendix B 2 b. For $k > 2$ the feedback vanishes in the thermodynamic limit, the non-condensed overlaps remain bare Gaussian variables, and $r = \mathcal { M } _ { k } ( q )$ is their bare moment, in agreement with Appendix B 2 d. This is the content of the scaling statement in the main text.

Remark 2: validity of the cumulant expansion. The truncation of $\operatorname { E q }$ . (B35) at second order is controlled as follows. (i) The third cumulant contributes $O ( \varepsilon ^ { 3 } )$ per pattern, hence $O ( N _ { v } ^ { 2 - k / 2 } )$ in total after multiplication by $N _ { h }$ which is subextensive for $k \geq 4 .$ . Since k is even in our setting, this covers all $k > 2 .$ and higher cumulants are smaller still. (ii) The corrections to the joint Gaussianity of $( \hat { y } ^ { a } ) _ { a }$ are of relative order $N _ { v } ^ { - 1 }$ (Edgeworth), and, for ±1 spin and symmetrically distributed patterns, the single-replica corrections are exactly state-independent, as noted below Eq. (B35). The state-dependent cross-replica corrections contribute only at $O ( 1 )$ in total. (iii) The first cumulant produces a state-independent constant of order $N _ { v } ^ { k / 2 }$ , which is superextensive but common to all configurations and all phases, and drops out of every order-parameter equation and free energy diference. (iv) The evenness of k is used twice: it bounds the hidden potential from below, and it guarantees the vanishing of the odd pattern cumulants used in (i) and (ii).

Remark 3: closed form of the noise kernel. The Hermite expansion provides the explicit polynomial form of $\mathcal { M } _ { k }$ Writing $\begin{array} { r } { x ^ { k - 1 } = \sum _ { j } c _ { k , j } \mathrm { H e } _ { j } ( x ) } \end{array}$ with

$$
c _ { k , j } = { \frac { ( k - 1 ) ! } { j ! 2 ^ { ( k - 1 - j ) / 2 } \left( { \frac { k - 1 - j } { 2 } } \right) ! } } , \qquad j \equiv k - 1 { \pmod { 2 } } , \quad 0 \leq j \leq k - 1 ,\tag{B46}
$$

the orthogonality $\mathbb { E } [ \mathrm { H e } _ { i } ( X ) \mathrm { H e } _ { j } ( Y ) ] = \delta _ { i j } j ! q ^ { j }$ gives

$$
\mathcal { M } _ { k } ( q ) = \sum _ { j } c _ { k , j } ^ { 2 } j ! q ^ { j } , \qquad \mathcal { M } _ { k } ( 1 ) = \mathbb { E } \Big [ X ^ { 2 ( k - 1 ) } \Big ] = ( 2 k - 3 ) ! ! ,\tag{B47}
$$

a polynomial with positive coeficients, for instance $\mathcal { M } _ { 4 } ( q ) = 9 q + 6 q ^ { 3 }$ and $\mathcal { M } _ { 6 } ( q ) = 2 2 5 q + 6 0 0 q ^ { 3 } + 1 2 0 q ^ { 5 }$ . The same polynomial kernel appears as the noise covariance in the dynamical mean-field theory of dense associative memories with the polynomial (Krotov–Hopfield-type) energy [53, 54], with the static correspondence between the equal-time correlation and the replica overlap. The monomial kernel ∝ $q ^ { k - 1 }$ familiar from p-spin models arises instead for the diagonal-free (Abbott–Arian-type) variant of the interaction [8, 54], which is a diferent model from ours.

Remark 4: the zero-temperature limit. The zero-temperature equations quoted in Sec. III A follow from Eq. (23)–Eq. (25) by standard manipulations. As $\beta _ { k } $ ∞ at fixed $C = \beta _ { k } ( 1 - q )$ , the hyperbolic tangent in Eq. (23) reduces to the sign of its argument, and $\begin{array} { r } { \int D z \ \mathrm { s g n } ( m ^ { k - 1 } + \sqrt { \alpha _ { k } r } z ) = \mathrm { e r f } \big ( m ^ { k - 1 } / \sqrt { 2 \alpha _ { k } r } \big ) } \end{array}$ yields Eq. (27). For the frozen response, one combines $\begin{array} { r } { 1 - q = \int D z \operatorname { s e c h } ^ { 2 } \beta _ { k } { \left( m ^ { k - 1 } + \sqrt { \alpha _ { k } r } z \right) } } \end{array}$ with $\beta _ { k } \operatorname { s e c h } ^ { 2 } ( \beta _ { k } x ) \to 2 \delta ( x )$ . The delta function picks up the Gaussian density at the zero $z _ { 0 } = - m ^ { k - 1 } / \sqrt { \alpha _ { k } r }$ of the local field, which gives Eq. (28). Finally, $q \to 1$ at fixed $C$ reduces the crosstalk moment Eq. (26) to Eq. (29): for $k = 2$ the denominator survives as $1 - \beta _ { k } ( 1 - q ) \to 1 - C ;$ while for $k > 2$ the polynomial is simply evaluated at $q = 1$

## 3. Model C replica analysis

In this appendix we derive the RS free energy Eq. (41) and the equations of state Eq. (42)–Eq. (43) of Model C. The hidden sector of Model C coincides with that of Model A: after the conjugate insertion, the non-condensed hidden modes factorize into the same single-mode integral Eq. (B16), so the dichotomy established in Appendix B 2 carries over: the hidden sector closes exactly for $k = 2$ and admits no consistent evaluation for $k > 2 .$ where the model is defined by the adiabatic reduction. We exploit this from the outset by working with the formulation in which the hidden variables are eliminated in favor of the pattern overlaps. This formulation is exact for $k = 2$ and definitional for $k > 2 .$ , and it leads directly to the two-parameter free energy quoted in the main text. The computation that retains the hidden variables explicitly is presented in Appendix B 3 e for completeness. What is genuinely new compared with Model A is the visible sector: the spherical trace is Gaussian and is carried out exactly, so that no single-site integra remains in the final expressions.

## a. Overlap formulation and replica setup

We start from the partition function Eq. (40). The flat radial direction of the visible variables has already been removed by the sphere-restricted definition of Eq. (14), and no further regularization is needed. The basic objects are the pattern overlaps of a visible configuration $\mathbf { x } \in S$

$$
\hat { m } _ { \mu } ( \mathrm { x } ) : = \frac { 1 } { N _ { v } } \sum _ { i } \xi _ { \mu i } ^ { ( h , v ) } \mathrm { x } _ { i } .\tag{B48}
$$

For $k = 2$ the hidden variables can be eliminated exactly: $\gamma _ { 2 } = \beta _ { k } / 2$ , each $m _ { \mu }$ appears quadratically in Eq. (40), and the Gaussian integral gives, up to an overall constant,

$$
\int d m _ { \mu } \exp \Big ( - \frac { \beta _ { k } } { 2 } N _ { v } m _ { \mu } ^ { 2 } + \beta _ { k } N _ { v } \hat { m } _ { \mu } ( { \bf x } ) m _ { \mu } \Big ) \propto \exp \Big ( \frac { \beta _ { k } } { 2 } N _ { v } \hat { m } _ { \mu } ( { \bf x } ) ^ { 2 } \Big ) .\tag{B49}
$$

For $k > 2$ the direct evaluation of the hidden sector fails in exactly the manner analyzed in Appendix B 2 c: the non-condensed modes factorize into the single-mode integral $\mathcal { T } _ { k } ( \hat { R } )$ of Eq. (B16), which admits no extensive saddle-point structure, and the crosstalk variance of the continuous modes sufers the runaway Eq. (B30). We return to this formulation in Appendix B 3 e. As for Model A, for $k > 2$ we therefore define Model C through the adiabatic reduction of Sec. II B. The enslaved value $m _ { \mu } = \hat { m } _ { \mu } ( { \bf x } )$ follows from the same per-mode stationarity algebra as in Eq. (B31), and we work with

$$
Z _ { \xi } ^ { \mathrm { C , a d } } : = \int _ { S } d \Omega ( { \bf x } ) \exp { \left[ \frac { \beta _ { k } } { k } N _ { v } \sum _ { \mu } \hat { m } _ { \mu } ( { \bf x } ) ^ { k } \right] } ,\tag{B50}
$$

which is exact for $k = 2$ by $\operatorname { E q } .$ . (B49) and is the adiabatic definition for $k > 2 ,$ so that the analysis below is uniform in $k .$

Replicating Eq. (B50) and averaging over the patterns, we use the rotational invariance of the spherical ensemble to rotate the condensed pattern to $\xi _ { 1 } ^ { ( h , v ) } = ( 1 , \cdot \cdot \cdot , 1 )$ . This gauge choice is exact at finite $N _ { v }$ (Remark 1 below), and the condensed term becomes a function of the visible magnetizations,

$$
\hat { m } _ { 1 } ( { \bf x } ^ { a } ) = \frac { 1 } { N _ { v } } \sum _ { i } { \bf x } _ { i } ^ { a } = : m ^ { a } .\tag{B51}
$$

For the non-condensed patterns we introduce the rescaled overlaps

$$
\hat { y } _ { \mu } ^ { a } : = \frac { 1 } { \sqrt { N _ { v } } } \sum _ { i } \xi _ { \mu i } ^ { ( h , v ) } \mathrm { x } _ { i } ^ { a } = \sqrt { N _ { v } } \hat { m } _ { \mu } ( \mathrm { x } ^ { a } ) .\tag{B52}
$$

Since the spherical ensemble has $\bar { \mathsf { \xi } } \Big [ \xi _ { i \mu } ^ { ( v , h ) } \xi _ { j \mu } ^ { ( v , h ) } \Big ] = \delta _ { i j }$ , the covariance of the overlaps is exactly the replica-overlap matrix of the visible variables,

$$
\mathbb { E } _ { \xi _ { \mu } } \left[ \hat { y } _ { \mu } ^ { a } \hat { y } _ { \mu } ^ { b } \right] = \frac { 1 } { N _ { v } } \sum _ { i } \mathrm { x } _ { i } ^ { a } \mathrm { x } _ { i } ^ { b } = : q _ { a b } , \qquad q _ { a a } = 1 ,\tag{B53}
$$

where the diagonal is fixed by the spherical constraint, the counterpart for continuous spins of $s _ { i } ^ { 2 } = 1$ in Model A. At large $N _ { v }$ the vector $\left( \hat { y } _ { \mu } ^ { a } \right) _ { { \sf ( } }$ <sub>a</sub> becomes jointly Gaussian with covariance matrix $Q = \left( q _ { a b } \right)$ . Moreover, by rotational invariance the exact per-pattern average depends on the visible configurations only through Q, so that all finite- $N _ { v }$ corrections are automatically functions of the order parameters (Remark 1).

## b. Noise sector

The average over one non-condensed pattern produces the noise factor

$$
{ \mathbb E } _ { \xi _ { \mu } } \left[ \exp \left( \varepsilon \sum _ { a } ( \hat { y } _ { \mu } ^ { a } ) ^ { k } \right) \right] , \qquad \varepsilon = \frac { \beta _ { k } } { k } N _ { v } ^ { 1 - k / 2 } ,\tag{B54}
$$

of the same form as in Eq. (B32).

For $k = 2$ we have $\varepsilon = \beta _ { k } / 2 = O ( 1 )$ , and the Gaussian asymptotics of $( \hat { y } _ { \mu } ^ { a } ) _ { a }$ gives the closed form

$$
\mathbb { E } _ { \xi _ { \mu } } [ e ^ { \frac { \beta _ { k } } { 2 } \sum _ { a } ( \hat { y } _ { \mu } ^ { a } ) ^ { 2 } } ]  \operatorname* { d e t } ( I _ { n } - \beta _ { k } Q ) ^ { - 1 / 2 } ,\tag{B55}
$$

convergent for $\beta _ { k } \lambda _ { \operatorname* { m a x } } ( Q ) < 1$ , with $\lambda _ { \mathrm { m a x } }$ the largest eigenvalue of Q. Under the RS ansatz $q _ { a b } = q \ ( a \neq b )$ , the eigenvalues of $Q$ are $1 - q$ with degeneracy $n - 1$ and $1 + ( n - 1 ) q$ , both of which tend to $1 - q$ in the replica limit, so that

$$
\log \operatorname* { d e t } \left( I _ { n } - \beta _ { k } Q \right) = ( n - 1 ) \log \left( 1 - \beta _ { k } ( 1 - q ) \right) + \log \left( 1 - \beta _ { k } ( 1 - q ) - n \beta _ { k } q \right) = n \Psi _ { 2 } ( q ) + O ( n ^ { 2 } ) .\tag{B56}
$$

which is the same algebra as Eq. (B26) evaluated at $( \hat { r } _ { d } , \hat { r } ) = ( 1 , q )$ . Multiplying by $\begin{array} { r } { - \frac 1 2 ( N _ { h } - 1 ) \simeq - \frac { \alpha _ { k } N _ { v } } { 2 } } \end{array}$ , the noise sector contributes $\begin{array} { r l } { ~ } & { { } - \frac { n \alpha _ { k } N _ { v } } { 2 } \Psi _ { 2 } ( q ) } \end{array}$ to the replicated exponent, i.e., $+ \frac { \alpha _ { k } } { 2 } \Psi _ { 2 } ( q )$ to $\beta f ^ { \mathrm { C } } .$

For $k > 2$ we have $\varepsilon \xrightarrow { - } 0$ , and Eq. (B54) is evaluated by the cumulant expansion Eq. (B35), whose structure carries over intact. The first cumulant is state-independent because it depends only on the diagonal $q _ { a a } = 1$ , now enforced by the spherical constraint. The second cumulant is governed by the same covariance kerne $\Phi _ { k } ( q _ { a b } )$ of Eq. (B36). The third and higher cumulants are subextensive for even $k \geq 4$ , with the Model C-specific aspects of the power counting collected in Remark 2. Summing over the $N _ { h } - 1$ patterns, the noise sector contributes the extensive action Eq. (B37), which, by Price’s theorem Eq. (B42), integrates to $\frac { \alpha _ { k } } { 2 } \Psi _ { k } ( q )$ in the free energy, exactly as in Appendix B 2 d.

In both cases, therefore, the noise sector enters $\begin{array} { r } { \beta f ^ { \mathrm { C } } \ \mathrm { a s } \ \frac { \alpha _ { k } } { 2 } \Psi _ { k } ( q ) } \end{array}$ , with the uniform derivative $\Psi _ { k } ^ { \prime } ( q ) = \beta _ { k } ^ { 2 } \mathcal { M } _ { k } ( q )$ quoted in the main text (for $k = 2$ directly from Eq. (B56), for $k > 2$ from Price’s theorem). Note also that the noise factors depend on the visible configurations only through the overlap matrix $Q ,$ , which is precisely the quantity fixed by the conjugate insertions of the next subsection.

## c. Visible Gaussian sector and the reduced free energy

The remaining trace over the visible variables is organized by introducing the condensed overlaps and the replica overlaps with their conjugates,

$$
1 = \int \prod _ { a } d m ^ { a } \delta \Big ( N _ { v } m ^ { a } - \sum _ { i } \mathbf { x } _ { i } ^ { a } \Big ) , \qquad 1 = \int \prod _ { a < b } d q _ { a b } \delta \Big ( N _ { v } q _ { a b } - \sum _ { i } \mathbf { x } _ { i } ^ { a } \mathbf { x } _ { i } ^ { b } \Big ) ,\tag{B57}
$$

represented with conjugate variables $\tilde { m } ^ { a }$ and $\hat { q } _ { a b }$ as in $\operatorname { E q . }$ (B14), together with the spherical constraints

$$
1 = \int _ { c - \mathrm { i } \infty } ^ { c + \mathrm { i } \infty } \prod _ { a } { \frac { d u ^ { a } } { 4 \pi \mathrm { i } } } \exp \bigg [ - { \frac { u ^ { a } } { 2 } } \Big ( \sum _ { i } ( \mathrm { x } _ { i } ^ { a } ) ^ { 2 } - N _ { v } \Big ) \bigg ] , \qquad c > 0 .\tag{B58}
$$

The multiplier $u ^ { a }$ is precisely the conjugate of the diagonal overlap $q _ { a a } \colon$ it absorbs the entire diagonal sector, playing the role of the pair $( r _ { d } , \hat { r } _ { d } )$ in Model $\mathrm { A } ,$ where we found $\hat { r } _ { d } = 1$ and a complete cancellation of $r _ { d }$ (Eq. (B22) and below). After these insertions the visible integral factorizes over the sites. Under the RS ansatz $m ^ { a } = m , \tilde { m } ^ { a } = \tilde { m }$ $u ^ { a } = u , q _ { a b } = q , \hat { q } _ { a b } = \hat { q }$ , the single-site measure is an n-dimensional Gaussian. Decoupling the replica coupling with a frozen Gaussian field, $\begin{array} { r } { { \hat { e } } ^ { { \hat { q } } \sum _ { a < b } { \mathbf { x } } ^ { a } { \mathbf { \check { x } } } ^ { b } } = e ^ { - \frac { n { \hat { q } } } { 2 } } \int D z e ^ { { \sqrt { \hat { q } } } z \sum _ { a } { \mathbf { x } } ^ { a } } } \end{array}$ , each site contributes

$$
\int { \cal D } z \prod _ { a } \int d { \bf x } ^ { a } \exp \Big [ - \frac { { \cal D } } { 2 } ( { \bf x } ^ { a } ) ^ { 2 } + \big ( \tilde { m } + \sqrt { \hat { q } } z \big ) { \bf x } ^ { a } \Big ] , \qquad { \cal D } : = u + \hat { q } ,\tag{B59}
$$

whose logarithm is, expanding to first order in n and dropping constants,

$$
\operatorname * { l i m } _ { n  0 } \frac { 1 } { n } \log ( \mathrm { s i t e ~ f a c t o r } ) = - \frac { 1 } { 2 } \log D + \frac { \tilde { m } ^ { 2 } + \hat { q } } { 2 D } .\tag{B60}
$$

Collecting the condensed term, the conjugate terms, the constraint terms, the noise sector of Appendix B 3 b, and $\operatorname { E q . }$ (B60), we obtain the variational free energy, up to additive constants,

$$
\beta f ^ { \mathrm { { C } } } = - \frac { \beta _ { k } } { k } m ^ { k } + \tilde { m } m - \frac { \hat { q } q } { 2 } + \frac { \alpha _ { k } } { 2 } \Psi _ { k } ( q ) - \frac { D - \hat { q } } { 2 } + \frac { 1 } { 2 } \log D - \frac { \tilde { m } ^ { 2 } + \hat { q } } { 2 D } ,\tag{B61}
$$

where we traded the multiplier $u = D - \hat { q }$ for $D .$

The stationarity conditions of Eq. (B61) are elementary. Variations with respect to m˜ and D give

$$
\tilde { m } = D m , \quad \quad \frac { 1 } { D } + \frac { \tilde { m } ^ { 2 } + \hat { q } } { D ^ { 2 } } = 1 ,\tag{B62}
$$

the latter being the spherical constraint $\begin{array} { r } { \frac { 1 } { N _ { v } } \sum _ { i } \bigl \langle \mathbf { x } _ { i } ^ { 2 } \bigr \rangle = 1 } \end{array}$ , while variation with respect to $\hat { q }$ identifies

$$
q = \frac { \tilde { m } ^ { 2 } + \hat { q } } { D ^ { 2 } } = m ^ { 2 } + \frac { \hat { q } } { D ^ { 2 } } .\tag{B63}
$$

This exhibits $q$ as the Edwards–Anderson order parameter: in the single-site measure, $\left. \mathbf { x } \right. = ( \tilde { m } + \sqrt { \hat { q } } z ) / D$ , so that $\begin{array} { r } { q = \int D z { \langle \mathbf { x } \rangle } ^ { 2 } } \end{array}$ decomposes into the condensed part $m ^ { 2 }$ and the glassy part $\hat { q } / D ^ { 2 }$ . Solving Eq. (B62)–Eq. (B63),

$$
{ \cal D } = \frac { 1 } { 1 - q } , \qquad \tilde { m } = \frac { m } { 1 - q } , \qquad \hat { q } = \frac { q - m ^ { 2 } } { ( 1 - q ) ^ { 2 } } ,\tag{B64}
$$

so that $1 - q = 1 / D$ is the single-site thermal variance left after the condensed and glassy components are removed, and $\beta _ { k } ( 1 - q )$ is the static susceptibility of the spherical state. The remaining two conditions are the equations of state. Variation with respect to q gives, using $\Psi _ { k } ^ { \prime } = \beta _ { k } ^ { 2 } \mathcal { M } _ { k }$ ,

$$
\hat { q } = \alpha _ { k } \beta _ { k } ^ { 2 } \mathcal { M } _ { k } ( q ) \qquad \Longrightarrow \qquad \frac { q - m ^ { 2 } } { ( 1 - q ) ^ { 2 } } = \alpha _ { k } \beta _ { k } ^ { 2 } \mathcal { M } _ { k } ( q ) ,\tag{B65}
$$

which is Eq. (43), and variation with respect to m gives $\tilde { m } = \beta _ { k } m ^ { k - 1 }$ , which combined with $\tilde { m } = m / ( 1 - q )$ yields the signal equation Eq. (42).

Finally, we eliminate the auxiliary variables. Substituting D and m˜ from $\operatorname { E q }$ . (B64) into $\operatorname { E q . }$ (B61), the qˆ-dependent terms cancel identically,

$$
- \frac { \hat { q } q } { 2 } + \frac { \hat { q } } { 2 } - \frac { \hat { q } } { 2 } ( 1 - q ) = 0 ,\tag{B66}
$$

(the three terms coming from the conjugate, constraint, and single-site terms, respectively, so that $\hat { q }$ need not even be substituted), while mm˜ $- \tilde { m } ^ { 2 } / ( 2 D ) = \bar { m } ^ { 2 } / ( 2 ( 1 - q ) )$ , and we arrive at

$$
\beta f ^ { \mathrm { C } } = - \frac { \beta _ { k } } { k } m ^ { k } + \frac { \alpha _ { k } } { 2 } \Psi _ { k } ( q ) - \frac { 1 } { 2 } \biggl [ \log ( 1 - q ) + \frac { q - m ^ { 2 } } { 1 - q } \biggr ] - \frac { 1 } { 2 } ,\tag{B67}
$$

which is the free energy Eq. (41) of the main text, up to the additive constant $- 1 / 2$ . Because the eliminated variables were removed at their exact stationary points, the stationarity of Eq. (B67) in $( m , q )$ reproduces Eq. (42)–Eq. (43), as stated in the main text.

## d. The case k = 2: marginality and zero capacity

For $k = 2$ the two expressions for the conjugate field obtained above, $\tilde { m } = \beta _ { k } m$ and $\tilde { m } = m / ( 1 - q )$ , are compatible only if

$$
\left[ 1 - \beta _ { k } ( 1 - q ) \right] m = 0 .\tag{B68}
$$

A retrieval state thus requires the marginality condition $\beta _ { k } ( 1 - q ) = 1$ , and when it holds the magnitude of m is left undetermined at this order: the condensed direction is a flat direction (zero mode) of the quadratic spherical energy rather than a genuine minimum. At $\alpha _ { k } = 0$ the flatness is lifted by the spherical constraint itself: Eq. (B65) gives $q = m ^ { 2 }$ , and $\ddot { \beta _ { k } } ( 1 - m ^ { 2 } ) = 1$ determines $m ^ { 2 } = 1 - T$ below $T _ { c } = 1$ , as quoted in the main text.

The noise sector exhibits the same marginality from a complementary viewpoint. The exact $k = 2$ noise factor $\operatorname { E q . }$ (B55) is convergent only for $\beta _ { k } \lambda _ { \operatorname* { m a x } } ( Q ) < 1$ , which under the RS ansatz becomes $\beta _ { k } ( 1 - q ) < 1$ in the replica limit. A retrieval state lies precisely at the boundary of this normalizability region, where det $( I _ { n } - \beta _ { k } Q )  0 \colon$ the fluctuations of the non-condensed overlaps soften and condense, which is the standard condensation mechanism of spherical models [55, 56], and the divergence of $\mathcal { M } _ { 2 } ( q ) = q / ( 1 - \beta _ { k } ( 1 - q ) ) ^ { 2 }$ is its precursor. Consequently the noise equation $\operatorname { E q . }$ (43) admits no solution with $m > 0$ at any $\alpha _ { k } > 0$ , including $T = 0$ , and the quadratic spherical model has zero storage capacity, in agreement with the marginal behavior of the spherical Hopfield model [20]. There, a retrieval phase is restored by augmenting the Hamiltonian with a quartic term. Within the class H, the same lifting of the flat direction is provided by the hidden nonlinearity with $k > 2$

## e. Direct computation with continuous hidden variables

For completeness, we sketch the computation that retains the hidden variables explicitly, parallel to Appendix $\mathrm { ~ B 2 a . }$ and show where it connects to the overlap formulation above. Averaging the replicated Eq. (40) over the non-condensed patterns gives the noise coupling $\begin{array} { r } { \frac { \beta _ { k } ^ { 2 } } { 2 } \sum _ { a , b } R ^ { a b } \sum _ { i } \mathbf { x } _ { i } ^ { a } \mathbf { x } _ { i } ^ { b } } \end{array}$ with the same covariance matrix $\begin{array} { r } { R ^ { a b } = \sum _ { \mu \geq 2 } ( m _ { \mu } ^ { a } ) ^ { k - 1 } ( m _ { \mu } ^ { b } ) ^ { k - 1 } } \end{array}$ as in Eq. (B12). Inserting the conjugate pair $( R , { \hat { R } } )$ exactly as in $\operatorname { E q } .$ . (B14), the non-condensed hidden modes factorize into the single-mode integral $\mathcal { T } _ { k } ( \hat { R } )$ of $\operatorname { E q }$ . (B16). With the spherical constraints Eq. (B58) inserted, the visible trace is an unconstrained Gaussian integral,

$$
\int \prod _ { a , i } d \mathbf { x } _ { i } ^ { a } \exp \Bigg [ - \frac { 1 } { 2 } \sum _ { i } \mathbf { x } _ { i } ^ { \top } S \mathbf { x } _ { i } + \beta _ { k } \sum _ { a , i } h _ { a } ^ { k - 1 } \mathbf { x } _ { i } ^ { a } \Bigg ] = ( 2 \pi ) ^ { n N _ { v } / 2 } ( \operatorname* { d e t } S ) ^ { - N _ { v } / 2 } \exp \left( \frac { N _ { v } \beta _ { k } ^ { 2 } } { 2 } b ^ { \top } S ^ { - 1 } b \right) ,\tag{B69}
$$

where $\mathbf { x } _ { i } = ( \mathbf { x } _ { i } ^ { a } ) _ { a } , h _ { a } : = m _ { 1 } ^ { a }$ is the condensed hidden mode, $\pmb { b } = ( h _ { a } ^ { k - 1 } ) _ { a }$ , and $S : = \mathrm { d i a g } ( u ^ { a } ) - \beta _ { k } ^ { 2 } R$ . Under the RS ansatz $( h _ { a } = h , u ^ { a } = u , R ^ { a a } = \alpha _ { k } r _ { d } , R ^ { a b } = \alpha _ { k } r , \hat { R } ^ { a a } = \hat { r } _ { d } , \hat { R } ^ { a b } = \hat { r } )$

$$
S = \tilde { D } I _ { n } - \alpha _ { k } \beta _ { k } ^ { 2 } r { \bf 1 1 } ^ { \top } , \qquad \tilde { D } : = u - \alpha _ { k } \beta _ { k } ^ { 2 } ( r _ { d } - r ) ,\tag{B70}
$$

whose eigenvalues are $\tilde { D }$ (with degeneracy $n - 1 )$ and $\tilde { D } - n \alpha _ { k } \beta _ { k } ^ { 2 } r$ , and the Sherman–Morrison formula gives

$$
\operatorname* { l i m } _ { n \to 0 } \frac { 1 } { n } \log \operatorname* { d e t } S = \log \tilde { D } - \frac { \alpha _ { k } \beta _ { k } ^ { 2 } r } { \tilde { D } } , \qquad \operatorname* { l i m } _ { n \to 0 } \frac { 1 } { n } b ^ { \top } S ^ { - 1 } b = \frac { h ^ { 2 k - 2 } } { \tilde { D } } .\tag{B71}
$$

The stationarity conditions with respect to $r _ { d }$ and u read

$$
\hat { r } _ { d } = \frac { 1 } { \tilde { D } } + \frac { \beta _ { k } ^ { 2 } ( \alpha _ { k } r + h ^ { 2 k - 2 } ) } { \tilde { D } ^ { 2 } } , \qquad \frac { 1 } { \tilde { D } } + \frac { \beta _ { k } ^ { 2 } ( \alpha _ { k } r + h ^ { 2 k - 2 } ) } { \tilde { D } ^ { 2 } } = 1 ,\tag{B72}
$$

so that $\hat { r } _ { d } = 1$ , and all remaining r -dependence then cancels via the shift $u = \tilde { D } + \alpha _ { k } \beta _ { k } ^ { 2 } ( r _ { d } - r )$ , in complete parallel with Model A (Eq. (B22) and below). The stationarity condition with respect to r gives

$$
\hat { r } = \frac { \beta _ { k } ^ { 2 } ( \alpha _ { k } r + h ^ { 2 k - 2 } ) } { \tilde { D } ^ { 2 } } = m ^ { 2 } + \frac { \alpha _ { k } \beta _ { k } ^ { 2 } r } { \tilde { D } ^ { 2 } } = q , \qquad m : = \frac { \beta _ { k } h ^ { k - 1 } } { \tilde { D } } ,\tag{B73}
$$

the Edwards–Anderson overlap of the visible variables, as in Eq. (B23): per site, $m = \langle \mathbf { x } _ { i } \rangle$ is the mean of the visible Gaussian and $\alpha _ { k } \beta _ { k } ^ { 2 } r / \tilde { D } ^ { 2 } = ( S ^ { - 1 } ) _ { a b } ~ ( a \neq b )$ is its frozen covariance, so that $\hat { r } = \langle \mathrm { x } _ { i } ^ { a } \rangle \langle \mathrm { x } _ { i } ^ { b } \rangle + ( S ^ { - 1 } ) _ { a b }$ . For $k = 2$ the hidden factor $\mathcal { T } _ { 2 } ( \hat { R } )$ is evaluated exactly as in Appendix B 2 b and yields the noise term $\begin{array} { r } { \frac { \alpha _ { k } } { 2 } \Psi _ { 2 } ( q ) } \end{array}$ together with $r = \mathcal { M } _ { 2 } ( q )$ (Eq. (B26) and Eq. (B28)). Assembling all terms and eliminating $( h , u )$ reproduces Eq. (B67), with the condensed hidden mode tied to the visible overlap by $m = \beta _ { k } ( 1 - q ) h ^ { k - 1 }$ . Combined with the h-saddle, this relation gives $m = h$ on the retrieval branch: the condensed hidden neuron equals the overlap it detects, in accordance with the general role of the hidden neurons discussed in Sec. II A. For $k > 2$ , this computation terminates at the same obstruction as in Model $\mathrm { A } \colon \mathcal { T } _ { k }$ admits no extensive evaluation (Appendix $\mathrm { B 2 c ) }$ , and one must pass to the adiabatic formulation Eq. (B50), upon which the pair $( R , { \hat { R } } )$ collapses onto the visible overlap, $R ^ { a b }  \alpha _ { k } \mathcal { M } _ { k } ( q _ { a b } )$ as in Eq. (B38), and the computation reduces to the one performed in Appendix B 3 a–Appendix B 3 c.

## f. Remarks

Remark 1: exactness of the gauge rotation and the Gaussian replacement. The rotation that maps the condensed pattern to $( 1 , \ldots , 1 )$ is an orthogonal transformation of $\mathbb R ^ { N _ { v } }$ under which both the spherical pattern ensemble and the visible measure dΩ(x) are invariant. The gauge choice is therefore exact at any finite $N _ { v } ,$ just as the site-wise sign gauge of Model A. The only asymptotic step in the pattern average is the replacement of the remaining spherical patterns by Gaussian vectors. By rotational invariance, the exact average of any function of the overlaps $( \hat { y } _ { \mu } ^ { a } ) _ { a }$ depends on the visible configurations only through the Gram matrix Q. The finite- $N _ { v }$ corrections are therefore automatically functions of the order parameters, and they are suppressed by $O ( N _ { v } ^ { - 1 } )$ relative to the Gaussian leading term, leaving the RS equations unafected.

Remark 2: validity of the cumulant expansion. The power counting of Remark 2 in Appendix B 2 e applies with two modifications. First, the overlaps are bounded, $| \hat { y } _ { \mu } ^ { a } | \le \sqrt { N _ { v } }$ by the Cauchy–Schwarz inequality with $\left. \xi _ { \mu } ^ { ( h , v ) } \right. _ { 2 } = \| \mathbf { x } ^ { a } \| _ { 2 } = \sqrt { N _ { v } }$ , so the per-pattern average $\operatorname { E q . }$ (B54) and all its cumulants are finite. Note that naively replacing yˆ by an exactly Gaussian variable inside the exponential would produce a divergent average for $k > 2$ . The correct order of operations is to expand in cumulants first and to evaluate each cumulant by the Gaussian asymptotics. The divergence of the naive replacement is precisely the $\mathcal { T } _ { k }$ pathology of Appendix B 2 c in another guise. Second, the role played by $s _ { i } ^ { 2 } = 1$ in Model A (the state-independence of the first cumulant) is now played by the spherical constraint, which fixes $q _ { a a } = 1$ exactly. The third cumulant is $O ( N _ { v } ^ { 2 - k / 2 } )$ by the same counting as in Model A, subextensive for even $k \geq 4$

Remark 3: the Onsager reaction field. The cavity argument of Remark 1 in Appendix B 2 e carries over with $s _ { i } \to \mathrm { x } _ { i } \colon$ the linear response of a soft spin to the field shift of one non-condensed mode is controlled by its single-site thermal variance, whose average $\begin{array} { r } { \frac { 1 } { N _ { v } } \sum _ { i } ( \left. \mathrm { x } _ { i } ^ { 2 } \right. - \left. \mathrm { x } _ { i } \right. ^ { 2 } ) = 1 - q } \end{array}$ follows from the spherical constraint. The self-feedback of a non-condensed overlap is again of relative order $\beta _ { k } ( 1 - q ) N _ { v } ^ { - ( k - 2 ) / 2 }$ : marginal at $k = 2$ , where its geometric resummation produces the denominator of $\mathcal { M } _ { 2 }$ (equivalently, the determinant Eq. (B55) resums it exactly), and vanishing for $k > 2 .$ , where $r = \mathcal { M } _ { k } ( q )$ is the bare Gaussian moment. This confirms, by an argument independent of replicas, that the crosstalk moment of Model C coincides with that of Model A, as stated in the main text.

## Appendix C: Details on Model B

This appendix collects the computations behind Sec. IV. Appendix C 1 carries out the zero-temperature estimates underlying the capacity Eq. (61). Appendix C 2 derives the copy representation exactly, including the hidden-sector zero mode and the shifts quoted in Sec. IV A 1, and makes the duality with the replica method quantitative. Appendix C 3 performs the optimization over copy configurations behind the finite-temperature phases, the retrieval branch, and the capacity of Sec. IV B. Appendix C 4 reconstructs the phase diagram at arbitrary real temperature and settles the analytic continuation of the integer-temperature lattice. Throughout we use the normalization of Sec. IV, $\tau _ { v } = 1$ and $\tau _ { h } = \lambda$ , for which ${ \tilde { \beta } } = \beta .$ , with Gaussian patterns at the exponential load Eq. (57). Rates of partition functions are denoted $\begin{array} { r } { \varphi : = \operatorname* { l i m } _ { N _ { v } \to \infty } { N _ { v } ^ { - 1 } } } \end{array}$ log $Z _ { \xi } ^ { \mathrm { B } }$ , measured relative to the Gaussian reference $\hat { \int { d v e ^ { - \beta \| v \| ^ { 2 } / 2 } } }$ as in Sec. IV B, and are related to the free energy densities of the main text by $f = - T \varphi$

## 1. Vertex condensation and the zero-temperature capacity

Vertex condensation. The Gram matrix is positive semi-definite, $x ^ { \top } G x = \left\| \xi ^ { ( v , h ) } x \right\| ^ { 2 } \geq 0$ , so the energy $Q ( f ) : =$ $f ^ { \top } G f$ is convex on the simplex, and a convex function on a compact convex set attains its maximum at an extreme point [45, Sec. 32]. Together with the elementary bound Eq. (58), this places the maximum at the vertex of the pattern of maximal norm, a deterministic fact independent of the pattern ensemble. The entropic competition quoted in Sec. IV A is quantified as follows. For the uniform mixture $\begin{array} { r } { f _ { \mathrm { m i x } } = \frac { 1 } { M } \sum _ { \mu \in S } e _ { \mu } } \end{array}$ over a set S of M patterns,

$$
Q ( f _ { \mathrm { m i x } } ) = \frac { 1 } { M ^ { 2 } } \Big [ \sum _ { \mu \in S } \Big \lVert \xi _ { \mu } ^ { ( h , v ) } \Big \rVert ^ { 2 } + \sum _ { \mu \neq \nu \in S } \xi _ { \mu } ^ { ( h , v ) } \cdot \xi _ { \nu } ^ { ( h , v ) } \Big ] = \frac { N _ { v } } { M } + O \big ( \frac { \sqrt { N _ { v } } } { M } \big ) ,\tag{C1}
$$

since the $M ( M - 1 )$ zero-mean cross terms, each of size $O ( \sqrt { N _ { v } } )$ , add up to $O ( M \sqrt { N _ { v } } )$ typically. Hence

$$
\Phi ( e _ { \mu } ) - \Phi ( f _ { \mathrm { m i x } } ) \approx \frac { \beta } { 2 } N _ { v } \biggl ( 1 - \frac { 1 } { M } \biggr ) - \frac { \beta } { \lambda } \log M ,\tag{C2}
$$

which is positive unless log $M \gtrsim N _ { v }$ : the entropy of an interior mixture can never ofset its extensive energy cost at subexponential load, and the competition becomes genuine only at the load Eq. (57). Even then the competitor is not an interior point of the simplex. The Gibbs measure decomposes into lumps attached to the vertices,

$$
Z _ { \xi } ^ { \mathrm { B } } \approx \sum _ { \mu = 1 } ^ { N _ { h } } Z _ { \mu } , \qquad Z _ { \mu } \asymp e ^ { \frac { \beta } { 2 } \left\| \xi _ { \mu } ^ { ( h , v ) } \right\| ^ { 2 } } ,\tag{C3}
$$

and the entropy is gained by spreading the measure over exponentially many, individually almost pure, lumps. The distinction between a mixed configuration and a mixture of pure states is the same as in spin-glass theory, and the counting of the lumps is precisely the entropy evaluated below. A final caution concerns the continuum measure: the simplex has dimension $N _ { h } { - } 1 = \stackrel { \cdot \mathrm { { \alpha } } } { e } ^ { \alpha N _ { v } } - 1$ , so the flat measure carries volume factors of order log $\mathrm { V o l } ( \Delta ^ { N _ { h } } ) \sim - N _ { h } \log N _ { h }$ doubly exponential in $N _ { v } .$ , which would overwhelm any $e ^ { O ( N _ { v } ) }$ energy unless they cancel exactly. The statements above are therefore formulated either through the adiabatic saddle point Eq. (53), for which vertex condensation is the single-term dominance of the log-sum-exp, or through the discrete copy representation of Appendix C 2, whose counting measure carries no volume factors.

Norm statistics and freezing. The single-pattern rate function quoted throughout the main text follows from Cramér’s theorem [28, 57]. For $\begin{array} { r } { \left. \xi _ { \mu } ^ { ( h , v ) } \right. ^ { 2 } = \sum _ { i } ( \xi _ { \mu i } ^ { ( h , v ) } ) ^ { 2 } } \end{array}$ a sum of $N _ { v }$ i.i.d. squared Gaussians, the cumulant generating function per component is

$$
\Lambda ( t ) : = \log \mathbb { E } e ^ { t \xi ^ { 2 } } = - \frac { 1 } { 2 } \log ( 1 - 2 t ) , \qquad t < \frac { 1 } { 2 } ,\tag{C4}
$$

and the Legendre transform $\mathrm { s u p } _ { t } [ t x - \Lambda ( t ) ]$ , whose stationary point is $t ^ { * } = { \textstyle \frac { 1 } { 2 } } ( 1 - 1 / x )$ , yields

$$
\mathbb { P } \left( \frac { \left. \xi _ { \mu } ^ { ( h , v ) } \right. ^ { 2 } } { N _ { v } } \approx x \right) \asymp e ^ { - N _ { v } I _ { 1 } ( x ) } , \quad I _ { 1 } ( x ) = \frac { x - 1 - \log x } { 2 } ,\tag{C5}
$$

the $M = 1$ case of Eq. (64). The rate $I _ { 1 }$ is convex, vanishes at $x = 1$ , and has the bounded derivative $I _ { 1 } ^ { \prime } ( x ) =$ $\begin{array} { r } { \frac { 1 } { 2 } ( 1 - 1 / x ) < \frac { 1 } { 2 } } \end{array}$ . Writing $\left\| \xi _ { \mu } ^ { ( h , v ) } \right\| ^ { 2 } = ( 1 + \varepsilon _ { \mu } ) N _ { v }$ , the number of patterns at norm excess $\varepsilon$ is

$$
\mathcal { N } ( \varepsilon ) \asymp N _ { h } e ^ { - N _ { v } I _ { 1 } ( 1 + \varepsilon ) } = e ^ { N _ { v } [ \alpha - I _ { 1 } ( 1 + \varepsilon ) ] } .\tag{C6}
$$

For $\alpha > I _ { 1 } ( 1 { + } \varepsilon )$ the occupation numbers of independent patterns concentrate on this value $( \mathrm { V a r / M e a n } ^ { 2 } \asymp e ^ { - N _ { v } [ \alpha - I _ { 1 } ] } \to$ 0 by the second-moment method), while for $\alpha < I _ { 1 } ( 1 + \varepsilon )$ the level is empty with high probability by Markov’s inequality. The maximal norm excess is therefore $\varepsilon _ { \mathrm { m a x } } ( \alpha )$ with $I _ { 1 } ( 1 + \varepsilon _ { \mathrm { m a x } } ) = \alpha _ { \mathrm { m } }$ as stated in Sec. IV B, and $\varepsilon _ { \mathrm { m a x } } \approx 2 \sqrt { \alpha }$ for small $\alpha .$ The lump sum Eq. (C3) then follows by the maximal-term principle,

$$
\frac { 1 } { N _ { v } } \log \sum _ { \mu } e ^ { \frac { \beta } { 2 } \left\| \xi _ { \mu } ^ { ( h , v ) } \right\| ^ { 2 } } = \frac { \beta } { 2 } + \operatorname* { m a x } _ { 0 \leq \varepsilon \leq \varepsilon _ { \mathrm { m a x } } } \left[ \alpha - I _ { 1 } ( 1 + \varepsilon ) + \frac { \beta } { 2 } \varepsilon \right] .\tag{C7}
$$

The interior stationary point $I _ { 1 } ^ { \prime } ( 1 + \varepsilon ^ { * } ) = \beta / 2$ gives $\varepsilon ^ { * } = \beta / ( 1 - \beta )$ , which exists for $\beta < 1$ and lies in the populated range while $I _ { 1 } ( 1 + \varepsilon ^ { * } ) = \kappa ( \beta ) \stackrel { \triangledown } { \le } \alpha$ . Substitution collapses Eq. (C7) to $\begin{array} { r } { \alpha - \frac { 1 } { 2 } \log ( 1 - \beta ) } \end{array}$ , in exact agreement with the annealed average ${ \mathbb E } e ^ { \frac { \beta } { 2 } \chi _ { N _ { v } } ^ { 2 } } = ( 1 - \beta ) ^ { - N _ { v } / 2 }$ per pattern: the sum is carried by exponentially many typical terms and self-averages. Since $I _ { 1 } ^ { \prime } < \frac { 1 } { 2 }$ , for $\beta \geq 1$ the bracket in Eq. (C7) increases monotonically, and for $\kappa ( { \boldsymbol { \beta } } ) > \alpha$ the stationary point leaves the populated range. In either case the maximum is pinned at the boundary $\varepsilon _ { \mathrm { m a x } } .$ where the counting exponent vanishes and

$$
\frac { 1 } { N _ { v } } \log \sum _ { \mu } e ^ { \frac { \beta } { 2 } \left\| \xi _ { \mu } ^ { ( h , v ) } \right\| ^ { 2 } } = \frac { \beta } { 2 } ( 1 + \varepsilon _ { \operatorname* { m a x } } ( \alpha ) ) :\tag{C8}
$$

the sum is dominated by the O(1) patterns of maximal norm, and the quenched value falls below the annealed one, which diverges altogether for $\beta \geq 1$ . This is the freezing mechanism of the REM, for whose rigorous treatment see [58]. These two branches are the free energies $f _ { \mathrm { C } }$ and f of Eq. (65), and the threshold $\beta = 1$ reappears in Appendix C 3 as the stability criterion of the visible fluctuations.

Leak sum and capacity. The stationary points of the efective energy Eq. (54) obey the fixed-point equation $v = \xi ^ { ( v , h ) } f ^ { * } ( v )$ of Sec. IV A. For the retrieval ansatz we evaluate the fields at $v = \dot { \xi } _ { 1 } ^ { ( h , v ) }$ and verify self-consistency afterwards. Conditioned on $\xi _ { 1 } ^ { ( h , v ) }$ , the overlaps $\xi _ { \mu } ^ { ( h , v ) } \cdot \xi _ { 1 } ^ { ( h , v ) }$ for $\mu \geq 2$ are exactly i.i.d. centered Gaussians of variance $\left\| \xi _ { 1 } ^ { ( h , v ) } \right\| ^ { 2 } \approx N _ { v } ,$ , which is the statement $a _ { \mu } = \lambda \sqrt { N _ { v } } \omega _ { \mu }$ used in Eq. (59). The leak sum L is a Boltzmann sum over the linear random energies $\lambda \sqrt { N _ { v } } \omega _ { \mu } ,$ so the counting dichotomy above applies with the Gaussian rate $x ^ { 2 } / 2$ in place of $I _ { 1 } \mathbf { : }$ levels $\omega _ { \mu } \approx x \sqrt { N _ { v } }$ are populated by $e ^ { N _ { v } ( \alpha - x ^ { 2 } / 2 ) }$ patterns up to $x _ { \mathrm { m a x } } = \sqrt { 2 \alpha }$ and empty beyond, which is the content of Eq. (60), with the interior branch again matching the annealed average of L and the boundary branch frozen on the $O ( 1 )$ most aligned patterns. Retrieval is self-consistent when $1 - p \le e ^ { - \lambda N _ { v } } L \to 0$ , i.e. $N _ { v } ^ { - 1 }$ log $L < \lambda$ . On the frozen branch this reads $\lambda { \sqrt { 2 \alpha } } < \lambda$ , i.e. $\alpha < \textstyle { \frac { 1 } { 2 } }$ , while on the annealed branch it reads $\alpha + \lambda ^ { 2 } / 2 < \lambda .$ i.e. $\alpha < \lambda - \lambda ^ { 2 } / 2$ . The branch applicable at the threshold is determined by comparing λ with $\sqrt { 2 \alpha _ { c } } ,$ which reduces to $\lambda \gtrless 1$ , and the two conditions combine into the capacity Eq. (61), continuous at $\lambda = 1$ . The correction to the ansatz is controlled by the same leak: $\begin{array} { r } { \delta : = \bar { v } - \xi _ { 1 } ^ { ( h , v ) } = \sum _ { \mu \geq 2 } f _ { \mu } ^ { \ast } \xi _ { \mu } ^ { ( h , v ) } } \end{array}$ obeys $\| \delta \| \leq ( 1 - p ) \operatorname* { m a x } _ { \mu } \left\| \xi _ { \mu } ^ { ( h , v ) } \right\|$ , exponentially small whenever the leak exponent is negative, so the fixed point survives in a neighborhood of $\xi _ { 1 } ^ { ( h , v ) }$ throughout the retrieval region.

Zero-temperature thermodynamics. At the retrieval saddle the efective energy Eq. (54) is $\begin{array} { r } { E ( \xi _ { 1 } ^ { ( h , v ) } ) \approx \frac { N _ { v } } { 2 } - } \end{array}$ $\begin{array} { r } { \frac { 1 } { \lambda } \cdot \lambda N _ { v } = - \frac { N _ { v } } { 2 } } \end{array}$ , the leak contributing only $e ^ { - O ( N _ { v } ) }$ corrections. The point $v = 0$ is also stationary, $\nabla E | _ { 0 } =$ $- N _ { h } ^ { - 1 } \sum _ { \mu } \xi _ { \mu } ^ { ( h , v ) } = { \cal O } ( \sqrt { N _ { v } / N _ { h } } )$ , with energy $\begin{array} { r } { E ( 0 ) = - \frac { 1 } { \lambda } \log N _ { h } = - \frac { \alpha } { \lambda } N _ { v } } \end{array}$ . Typical retrieval therefore lies below this paramagnetic point only for $\alpha < \lambda / 2$ , and for $\lambda \leq 1$ the window $\lambda / 2 < \alpha < \lambda - \lambda ^ { 2 } / 2$ supports retrieval only as a metastable state. Neither of these is the true zero-temperature equilibrium, however: the retrieval state of the maximal-norm pattern has energy density $- ( 1 + \varepsilon _ { \mathrm { { m a x } } } ) / 2$ , below both, and its leak condition is satisfied wherever the comparison is relevant because the signal is enhanced by the factor $1 + \varepsilon _ { \mathrm { m a x } }$ . Equating it with the paramagnetic value yields the zero-temperature equilibrium boundary $2 \alpha / \lambda = 1 + \varepsilon _ { \mathrm { m a x } } ( \alpha )$ , whose small-λ solution is $\overset { \cdot } { \alpha } \approx \lambda / 2 + \lambda ^ { 3 / 2 } / \sqrt { 2 }$ This is the $T  0$ limit of the P–F boundary derived in Appendix C 3, and the extreme-value enhancement is precisely the amount by which the equilibrium boundary exceeds the naive comparison $\alpha = \lambda / 2$

## 2. Copy representation and hidden-sector corrections

Zero mode of the hidden Hessian and the shifts. The adiabatic evaluation Eq. (10) presumes a positive-definite Hessian of the hidden Lagrangian. For Model B this fails in exactly one direction. At any point of the hidden space,

$$
\operatorname { H e s s } { \big ( } L _ { h } ( h ) { \big ) } = \operatorname { d i a g } ( f ) - f f ^ { \top } , \qquad f = \operatorname { s o f t m a x } ( h ) ,\tag{C9}
$$

and the normalization $\textstyle \sum _ { \mu } f _ { \mu } = 1$ makes the all-one vector an exact zero mode,

$$
\big ( \mathrm { d i a g } ( f ) - f f ^ { \top } \big ) { \bf 1 } = f - f \sum _ { \mu } f _ { \mu } = 0 .\tag{C10}
$$

This is not an accident of the saddle point but the gauge direction of the softmax, which is invariant under the uniform shift $h  h + c { \bf 1 }$ noted in Sec. IV. The energy is exactly constant along 1, not merely flat to quadratic order. The remaining spectrum is well behaved: Eq. (C9) is a rank-one downdate of a positive diagonal matrix, its eigenvalues interlace the softmax weights, and exactly one eigenvalue vanishes while the others remain positive. Its pseudo-determinant (the product of the nonzero eigenvalues, denoted $\operatorname* { d e t } ^ { \prime } )$ follows from the matrix determinant lemma with an ϵ regularization,

$$
\begin{array} { l } { { \displaystyle \operatorname * { d e t } \left( \mathrm { H e s s } + \epsilon I \right) = \left( 1 - \sum _ { \mu } \frac { f _ { \mu } ^ { 2 } } { f _ { \mu } + \epsilon } \right) \prod _ { \mu } ( f _ { \mu } + \epsilon ) } } \\ { { = \left[ N _ { h } \epsilon + O ( \epsilon ^ { 2 } ) \right] \prod _ { \mu } f _ { \mu } \left( 1 + O ( \epsilon ) \right) , } } \end{array}\tag{C11}
$$

so that

$$
\operatorname* { d e t } ^ { \prime } \mathrm { H e s s } \big ( L _ { h } ( h ) \big ) = \operatorname* { l i m } _ { \epsilon  0 } \frac { \operatorname* { d e t } ( \mathrm { H e s s } + \epsilon I ) } { \epsilon } = N _ { h } \prod _ { \mu = 1 } ^ { N _ { h } } f _ { \mu } .\tag{C12}
$$

Expanding the h integral around the adiabatic saddle point along the Hessian eigenbasis, the flat direction contributes a divergent volume $V _ { 0 }$ that is independent of v and of the patterns. Dividing it out and performing the remaining

$N _ { h } - 1$ Gaussian modes yields the corrected version of Eq. (10),

$$
Z _ { \xi } ^ { \mathrm { B } } ( \beta ) / V _ { 0 } = \left( \frac { 2 \pi } { \beta / \lambda } \right) ^ { ( N _ { h } - 1 ) / 2 } \int _ { \mathbb { R } ^ { N _ { v } } } d v e ^ { - \beta E _ { \xi } ^ { \mathrm { B } } ( v , h _ { * } ) } \bigl ( \operatorname* { d e t } ^ { \prime } \mathrm { H e s s } ( L _ { h } ( h _ { * } ) ) \bigr ) ^ { - 1 / 2 } .\tag{C13}
$$

The exponent $( N _ { h } - 1 ) / 2$ in place of $N _ { h } / 2$ , together with the quotient by $V _ { 0 } .$ , is the measure prescription stated in the footnote of Sec. IV: the flat measure df on the simplex is the flat measure dh modulo the gauge orbit.

The determinant factor in $\operatorname { E q . }$ (C13) is itself an exponential of the fields. With $a : = \lambda \xi ^ { ( h , v ) } v$ and $f _ { \nu } ^ { * }$ = softmax $( a ) _ { \nu } =$ $e ^ { a _ { \nu } } / \sum _ { \mu } e ^ { a _ { \mu } }$

$$
\left( \mathrm { d e t ^ { \prime } H e s s } \right) ^ { - 1 / 2 } = N _ { h } ^ { - 1 / 2 } e ^ { - \frac { 1 } { 2 } \sum _ { \nu } a _ { \nu } } \Bigl ( \sum _ { \mu } e ^ { a _ { \mu } } \Bigr ) ^ { N _ { h } / 2 } ,\tag{C14}
$$

so the integrand of $\operatorname { E q . }$ (C13) combines with the power $\displaystyle ( \sum _ { \mu } e ^ { a _ { \mu } } ) ^ { \beta / \lambda }$ of the efective energy Eq. (54) into

$$
\exp \Big [ - \frac { \beta } { 2 } \| v \| ^ { 2 } - \frac { 1 } { 2 } \sum _ { \nu } a _ { \nu } \Big ] \Big ( \sum _ { \mu } e ^ { a _ { \mu } } \Big ) ^ { \beta / \lambda + N _ { h } / 2 } .\tag{C15}
$$

Whenever the combined exponent $\bar { n } : = \beta / \lambda + N _ { h } / 2$ is a positive integer, the multinomial theorem linearizes the power into a sum over n¯-tuples, each term is Gaussian in $v ,$ and completing the square component by component gives, up to the same prefactors,

$$
Z _ { \xi } ^ { \mathrm { B } } ( \beta ) / V _ { 0 } \propto \sum _ { \mu _ { 1 } , \dots , \mu _ { n } } \exp \Big [ \frac { \lambda ^ { 2 } } { 2 \beta } \Big \lVert \tilde { \xi } _ { \{ \mu \} } \Big \rVert ^ { 2 } \Big ] , \quad \tilde { \xi } _ { \{ \mu \} } : = - \frac { 1 } { 2 } \sum _ { \nu = 1 } ^ { N _ { h } } \xi _ { \nu } ^ { ( h , v ) } + \sum _ { j = 1 } ^ { n } \xi _ { \mu _ { j } } ^ { ( h , v ) } .\tag{C16}
$$

Relative to the representation Eq. (62), which coincides with the visible-only computation of [14], the hidden-sector fluctuations thus produce exactly the two corrections quoted in Sec. IV A 1: the exponent shift $\dot { \beta } / \lambda  \beta / \lambda + N _ { h } / 2$ and the additive pattern shift $- \textstyle { \frac { 1 } { 2 } } \sum _ { \nu } \xi _ { \nu } ^ { ( h , v ) }$ . Two features of Eq. (C16) deserve emphasis at exponential load. First, positivity of the temperature imposes $\bar { \beta } / \lambda = \bar { n } - N _ { h } / 2 > 0 :$ the admissible integers satisfy $\bar { n } > N _ { h } / 2$ , so the small copy numbers, in particular $\bar { n } = 1$ , are excluded once $N _ { h }$ is large. Second, the shift is not a perturbation: the components of ${ \begin{array} { l } { { \frac { 1 } { 2 } } \sum _ { \nu } \xi _ { \nu } ^ { ( h , \bar { v } ) } } \end{array} }$ are of order $\sqrt { N _ { h } }$ , so at $N _ { h } = e ^ { \alpha N _ { v } }$ the shifted sum is dominated by the correction term itself. The analysis of the main text is therefore formulated for the adiabatic partition function proper, the leading saddle-point term for which the unshifted expansion Eq. (62) is an exact identity at $n = \beta / \lambda \in \mathbb { Z } _ { > 0 }$ . The corrections $\operatorname { E q } .$ (C16) constitute the complete Gaussian fluctuation content of the hidden sector around that limit, and their consistent treatment at exponential load remains open (see the remarks closing Appendix C 4).

Alignment versus dispersion. Reversing the Gaussian integration that produced $\operatorname { E q } .$ (62) shows that, conditioned on a copy configuration, the visible state is Gaussian around the centroid of the selected patterns,

$$
v \mid \{ \mu _ { j } \} \sim { \mathcal { N } } \Bigl ( { \frac { 1 } { n } } \sum _ { j = 1 } ^ { n } \xi _ { \mu _ { j } } ^ { ( h , v ) } , \ \beta ^ { - 1 } I _ { N _ { v } } \Bigr ) ,\tag{C17}
$$

which is the parallel-query picture of Sec. IV A 1 and will be used repeatedly below to read of visible observables. Expanding the energy of Eq. (62),

$$
\left\| \sum _ { j } \xi _ { \mu _ { j } } ^ { ( h , v ) } \right\| ^ { 2 } = \sum _ { j } \left\| \xi _ { \mu _ { j } } ^ { ( h , v ) } \right\| ^ { 2 } + \sum _ { j \neq l } \left[ N _ { v } \delta _ { \mu _ { j } \mu _ { l } } + O ( \sqrt { N _ { v } } ) \right] ,\tag{C18}
$$

the copies form an n-site system with $N _ { h }$ states per site, a ferromagnetic gain of order $N _ { v }$ for every pair of aligned copies, and random couplings of order $\sqrt { N _ { v } }$ otherwise. The zero-temperature equilibrium of Appendix C 1 can be recovered from this picture by pure counting. The fully aligned configurations $( \mu _ { j } \equiv \mu$ , with $N _ { h }$ choices) carry the exponent $\begin{array} { r } { \frac { \lambda } { 2 n } n ^ { 2 } N _ { v } + \alpha N _ { v } = ( \frac { \beta } { 2 } + \alpha ) N _ { v } } \end{array}$ , while the fully dispersed ones $( n _ { \mu } \in \{ 0 , 1 \}$ , with ≈ $e ^ { n \alpha N _ { v } }$ choices) retain only the diagonal energy, $\bigl ( { \frac { \lambda } { 2 } } + n \alpha \bigr ) N _ { v }$ . The diference,

$$
\Big ( \frac { \beta } { 2 } + \alpha \Big ) - \Big ( \frac { \lambda } { 2 } + n \alpha \Big ) = ( n - 1 ) \Big ( \frac { \lambda } { 2 } - \alpha \Big ) ,\tag{C19}
$$

shows that alignment is favored precisely for $\alpha < \lambda / 2$ , reproducing the zero-temperature equilibrium threshold of Appendix C 1 without any reference to the f representation. The frozen phase corresponds, in the same language, to the co-condensation of all copies onto the few patterns of extreme norm.

Legendre duality and one-step RSB. The qualitative dictionary of Sec. IV A 1 becomes a precise convex duality. On the replica side one computes $\mathbb { E } _ { \xi } \bigg [ ( Z _ { \xi } ^ { \mathrm { B } } ) ^ { r } \bigg ]$ at a formal replica number r. The patterns enter the replicated exponent through $\textstyle \sum _ { i } \xi _ { i } ^ { \top } \Theta \xi _ { i }$ with the tilting matrix $\begin{array} { r } { \mathbf { \bar { \Theta } } \mathbf { \Psi } \mathbf { \Psi } \mathbf { \Psi } \mathbf { \Psi } \Theta : = \frac { \beta } { 2 } \sum _ { a = 1 } ^ { r } f ^ { a } ( f ^ { a } ) ^ { \top } } \end{array}$ , where $\xi _ { i } \in \mathbb { R } ^ { N _ { h } }$ collects the i-th components of all patterns, and the Gaussian average per visible component is the matrix version of Eq. (C4),

$$
\mathbb { E } e ^ { \xi ^ { \top } \Theta \xi } = \operatorname* { d e t } ( I - 2 \Theta ) ^ { - 1 / 2 } = : e ^ { \Lambda ( \Theta ) } ,\tag{C20}
$$

valid for $I - 2 \Theta \succ 0$ . By Sylvester’s identity, det $\begin{array} { r } { ( I _ { N _ { h } } - \beta \sum _ { a } f ^ { a } ( f ^ { a } ) ^ { \top } ) = \operatorname* { d e t } ( I _ { r } - \beta R ) } \end{array}$ with the replica overlap matrix $R ^ { a b } = f ^ { a } \cdot f ^ { b }$ , so the disorder average generates the efective replica coupling $- \frac { N _ { v } } { 2 }$ log det $\left( I _ { r } - \beta R \right)$ . On the copy side the disorder statistics appear instead as the counting rate $I _ { M } ( Q )$ of Eq. (64). The two are a Legendre–Fenchel pair [28, 45],

$$
\begin{array} { r l r } & { } & { I _ { M } ( Q ) = \underset { \Theta } { \operatorname* { s u p } } \big [ \operatorname { T r } ( \Theta Q ) - \Lambda ( \Theta ) \big ] , } \\ & { } & { \Lambda ( \Theta ) = \underset { Q } { \operatorname* { s u p } } \big [ \operatorname { T r } ( \Theta Q ) - I _ { M } ( Q ) \big ] , } \end{array}\tag{C21}
$$

with stationarity $Q ^ { * } = ( I - 2 \Theta ) ^ { - 1 }$ : the optimal Gram matrix is the covariance of the exponentially tilted Gaussian ensemble. The annealed branches of Appendix C 3 realize the second line of Eq. (C21) at the rank-one tilts generated by the copy energy, and the conjugate order parameter that a replica computation introduces through the Fourier representation of δ functions is precisely the tilting matrix Θ: the field of the exponential change of measure that selects which distortion of the pattern statistics dominates. Freezing is the point where the two routes diverge. The Gaussian average Eq. (C20) runs over an unbounded ensemble, whereas only $e ^ { \alpha N _ { v } }$ patterns exist. The correct variational problem is the Fenchel transform constrained to the populated set,

$$
\operatorname* { s u p } _ { Q } \left[ \operatorname { T r } ( \Theta Q ) - I _ { M } ( Q ) \right] \quad \mathrm { s u b j e c t ~ t o } \quad I _ { M } ( Q ) \leq M \alpha .\tag{C22}
$$

When the constraint is active, the Karush–Kuhn–Tucker condition with multiplier $\eta \geq 0$ reads $\Theta = ( 1 + \eta ) \nabla I _ { M } ( Q )$ , i.e.

$$
Q = \left( I - \frac { 2 \Theta } { 1 + \eta } \right) ^ { - 1 } :\tag{C23}
$$

the tilt is renormalized by the factor $( 1 + \eta ) ^ { - 1 } \in ( 0 , 1 ]$ In the scalar case relevant to the frozen phase (tilt $\beta / 2$ boundary condition $I _ { 1 } ( Q ) = \alpha , { \mathrm { i . e . ~ } } Q = 1 + \varepsilon _ { \operatorname* { m a x } } )$ , Eq. (C23) gives $\beta / ( 1 + \eta ) = \varepsilon _ { \mathrm { m a x } } / ( 1 + \varepsilon _ { \mathrm { m a x } } ) = \beta _ { f }$ , so that

$$
\frac { 1 } { 1 + \eta } = \frac { \beta _ { f } } { \beta } = \frac { T } { T _ { f } } ,\tag{C24}
$$

precisely the temperature dependence of the one-step Parisi parameter of the REM [21, 27, 58]: the frozen phase is pinned at the edge of the populated set and behaves as if held at the freezing temperature, which is the origin of the temperature independence of f . $f _ { \mathrm { F } }$

The identification of the freezing prescription with one-step RSB can be checked exactly on the lump sum Eq. (C7), which is a REM in its own right. Computing $\mathbb { E } \tilde { Z } ^ { r }$ for $\begin{array} { r } { \tilde { Z } = \sum _ { \mu } e ^ { \frac { \beta } { 2 } \left\| \xi _ { \mu } ^ { ( h , v ) } \right\| ^ { 2 } } } \end{array}$ with the one-step ansatz, $r / x$ blocks of x replicas co-condensing on distinct patterns, each block contributes $e ^ { \alpha N _ { v } } \mathbb { E } e ^ { \frac { x \beta } { 2 } \chi _ { N _ { v } } ^ { 2 } } = e ^ { N _ { v } [ \alpha - \frac { 1 } { 2 } \log ( 1 - x \beta ) ] }$ , so per replica

$$
\varphi _ { \mathrm { 1 R S B } } ( x ) = \frac { 1 } { x } \Big [ \alpha - \frac { 1 } { 2 } \log ( 1 - x \beta ) \Big ] .\tag{C25}
$$

Extremizing over the block size continued to $x \in ( 0 , 1 ]$ yields $\alpha = \kappa ( x \beta )$ , whose solution is $x ^ { * } \beta = \beta _ { f } , \mathrm { i . e . } x ^ { * } = T / T _ { f } ,$ in agreement with the multiplier Eq. (C24). Substituting back, $\begin{array} { r } { \alpha - \frac 1 2 \log ( 1 - \beta _ { f } ) = \alpha + \frac 1 2 \log ( 1 + \varepsilon _ { \mathrm { m a x } } ) = \varepsilon _ { \mathrm { m a x } } / 2 } \end{array}$ and

$$
\varphi _ { \mathrm { 1 R S B } } ( x ^ { * } ) = \frac { \beta } { \beta _ { f } } \cdot \frac { \varepsilon _ { \mathrm { m a x } } } { 2 } = \frac { \beta } { 2 } \big ( 1 + \varepsilon _ { \mathrm { m a x } } \big ) ,\tag{C26}
$$

which is the frozen counting value Eq. (C8) exactly. For $\beta < \beta _ { f }$ the stationary point falls at $x > 1$ , the extremum over the physical interval is the endpoint $x = 1$ , and the RS evaluation returns the annealed branch, again in agreement.

For the sign conventions of the $r  0$ extremization see [27]. The one-step structure also fixes the weight statistics of the frozen phase: the lump weights $w _ { \mu } = e ^ { \frac { \beta } { 2 } \left\| \xi _ { \mu } ^ { ( h , v ) } \right\| ^ { 2 } } / \tilde { Z }$ follow a Ruelle point process of parameter $x ^ { * } = T / T _ { f } \ [ 3 6 , 5 0 ]$ with participation ratio $\mathbb { E } \bar { \sum _ { \mu } { w _ { \mu } ^ { 2 } } } = \bar { 1 } ^ { \cdot } - T / T _ { f }$ . Since the $w _ { \mu }$ are the attention weights of the f representation, this is a direct quantitative prediction for the attention statistics in the F phase.

Clone method. Finally, the copy representation realizes the clone method of Monasson [22] with a physical clone number. The single-group sector of Appendix C 3 (all n copies on one pattern) has the rate

$$
\varphi = \underset { \varepsilon } { \operatorname* { m a x } } \Big [ \underset { \mathrm { c o m p l e x i t y } } { \underbrace { \alpha - I _ { 1 } ( 1 + \varepsilon ) } } + n \cdot \underbrace { \frac { \lambda } { 2 } ( 1 + \varepsilon ) } _ { \mathrm { p e r - c l o n e ~ g a i n } } \Big ] ,\tag{C27}
$$

which is exactly the clone free energy: the choice of the occupied state (pattern) is counted once for the whole clone collective, while the energy is paid once per clone. By the envelope theorem, $\begin{array} { r } { { \partial \varphi } / { \partial n } = \frac { \lambda } { 2 } ( 1 + \varepsilon ^ { * } ) } \end{array}$ reads of the occupied level and $\varphi - n \partial \varphi / \partial n = \Sigma ( \varepsilon ^ { * } )$ the complexity, so the clone-number derivative extracts the configurational entropy exactly as in the structural-glass application. The distinctive feature of Model B is that the clone number $n = \beta / \lambda$ is not an auxiliary parameter but the physical temperature itself. Scanning n through real values is scanning the temperature, and the analytic control of this continuation is supplied by the generalized binomial expansion of Appendix C 4.

## 3. Finite-temperature analysis in the copy representation

Counting lemma. We first derive the counting statement of Sec. IV B. Fix M distinct indices $ { \boldsymbol Ḋ \mu Ḍ } _ { 1 } , \dots ,  { \boldsymbol Ḋ \mu Ḍ } _ { M }$ . For each visible component i the vector $x _ { i } : = ( \xi _ { \mu _ { 1 } i } ^ { ( h , v ) } , \ldots , \bar { \xi } _ { \mu _ { M } i } ^ { ( h , v ) } ) ^ { \top }$ is standard Gaussian in $\mathbb { R } ^ { M }$ , i.i.d. over $i ,$ and the normalized Gram matrix is the empirical covariance $\begin{array} { r } { \hat { Q } = \frac { 1 } { N _ { v } } \sum _ { i } { x _ { i } x _ { i } ^ { \top } } } \end{array}$ . Its cumulant generating function per component is the matrix average Eq. (C20), and Cramér’s theorem gives $\mathbb P ( \hat { Q } \approx Q ) \asymp e ^ { - N _ { v } I _ { M } ( Q ) }$ with the Legendre transform of Eq. (C21): the stationary point $\Theta ^ { * } = { \textstyle \frac { 1 } { 2 } } \big ( I - Q ^ { - 1 } \big )$ yields $\begin{array} { r } { \operatorname { T r } ( \Theta ^ { * } Q ) = \frac { 1 } { 2 } \operatorname { T r } ( Q - I ) } \end{array}$ and $\Lambda ( \Theta ^ { * } ) = \textstyle { \frac { 1 } { 2 } }$ log det Q, which yields the rate Eq. (64). Multiplying by the $e ^ { M \alpha N _ { v } }$ choices of the indices, and repeating the second-moment and Markov arguments of Appendix C 1, the number of M-tuples with Gram matrix $Q$ concentrates on $e ^ { N _ { v } [ M \alpha - I _ { M } ( Q ) ] }$ when the exponent is positive and vanishes with high probability otherwise. In the eigenbasis of $Q$ the rate decomposes as

$$
I _ { M } ( Q ) = \sum _ { a = 1 } ^ { M } I _ { 1 } ( \Lambda _ { a } ) ,\tag{C28}
$$

with $\Lambda _ { a }$ the eigenvalues of $Q ,$ and this decomposition organizes every computation below.

Partition classes and the bulk branches. Following Sec. IV B, classify the copy configurations by the partition of the n copies over distinct patterns. Consider first the symmetric classes: M groups of equal size $n / M$ , each occupying one of M distinct patterns (the mixed classes relevant for retrieval are treated below, and the remaining channels in Appendix C 4). The number of ways to distribute the copies is subexponential in $N _ { v } .$ , and the weight in Eq. (62) depend on the configuration only through the Gram matrix $Q$ of the occupied patterns, with $\begin{array} { r } { \left\| \sum _ { j } \xi _ { \mu _ { j } } ^ { ( h , v ) } \right\| ^ { 2 } = ( n / M ) ^ { 2 } { \bf 1 } ^ { \top } Q { \bf 1 } N _ { v } } \end{array}$ so the class exponent per visible neuron is

$$
\varphi [ Q ] = M \alpha - I _ { M } ( Q ) + \frac { \beta } { 2 M ^ { 2 } } \mathbf { 1 } ^ { \top } Q \mathbf { 1 } .\tag{C29}
$$

The energy couples only to the coherent quadratic form $\mathbf { 1 } ^ { \top } Q \mathbf { 1 }$ . Writing $w _ { a } : = ( { \bf 1 } \cdot e _ { a } ) ^ { 2 } / M$ for the weights of the eigenvectors of $Q$ in the coherent direction $\bar { ( \sum _ { a } w _ { a } = 1 ) }$ , so that $\begin{array} { r } { \mathbf { 1 } ^ { \top } Q \mathbf { 1 } = \bar { M } \sum _ { a } w _ { a } \bar { \Lambda } _ { a } } \end{array}$ , the convexity and positivity of $I _ { 1 }$ give

$$
I _ { M } ( Q ) = \sum _ { a } I _ { 1 } ( \Lambda _ { a } ) \geq \sum _ { a } w _ { a } I _ { 1 } ( \Lambda _ { a } ) \geq I _ { 1 } \Bigl ( \sum _ { a } w _ { a } \Lambda _ { a } \Bigr ) ,\tag{C30}
$$

with equality if and only if $\mathbf { \delta } _ { \mathbf { 1 } / \sqrt { M } }$ is an eigenvector and all remaining eigenvalues lie at the minimum of $I _ { 1 }$ , i.e. at 1. The symmetric form $Q = ( q _ { d } - q ) I + q \mathbf { 1 1 } ^ { \top }$ with transverse eigenvalue $\Lambda _ { 2 } = q _ { d } - q = 1$ is therefore not an ansatz but the exact maximizer within each class, and the exponent reduces to one variable, the coherent eigenvalue $\Lambda _ { 1 }$

$$
\varphi ( M , \Lambda _ { 1 } ) = M \alpha - I _ { 1 } ( \Lambda _ { 1 } ) + \frac { \beta } { 2 M } \Lambda _ { 1 } .\tag{C31}
$$

For M = 1 this is precisely the lump sum Eq. (C7), of which Eq. (C31) is the generalization to collectively occupied patterns. The maximization over $\Lambda _ { 1 }$ repeats the annealed/frozen dichotomy of Appendix C 1. The interior stationary point $I _ { 1 } ^ { \prime } ( \Lambda _ { 1 } ^ { * } ) \ : = \ : \beta / 2 M$ gives $\Lambda _ { 1 } ^ { * } = ( 1 { \bar { ~ } } \overline { { { - \beta } } } / M ) ^ { - 1 }$ , which exists for $\beta < M$ . At the stationary point the identity $\begin{array} { r } { \frac { \beta } { 2 M } \Lambda _ { 1 } ^ { * } - \frac { 1 } { 2 } ( \Lambda _ { 1 } ^ { * } - 1 ) = 0 } \end{array}$ collapses the exponent to

$$
\varphi _ { \mathrm { a n n } } ( M ) = M \alpha - \frac { 1 } { 2 } \log \Big ( 1 - \frac { \beta } { M } \Big ) , \qquad M \alpha \geq \kappa \Big ( \frac { \beta } { M } \Big ) ,\tag{C32}
$$

in exact agreement with the annealed average of the class partition function (the rank-one tilt has the single nontrivia eigenvalue $\beta / 2 M$ along 1, so the determinant in Eq. (C20) is $( 1 - \beta / M ) ^ { - 1 / 2 } )$ : the annealed branch is the self-averaging regime. When the stationary point violates the counting constraint, or when $\beta \geq M$ , the maximum is pinned at the boundary of the populated set, where the counting exponent vanishes. Since $I _ { 1 }$ is convex with its minimum at $\Lambda = 1$ the counting condition $I _ { 1 } ( \Lambda ) = M \alpha$ has one root below and one above unity, and the exponent Eq. (C31) increases with Λ<sub>1</sub> throughout the populated range, so the boundary is the upper root $\Lambda _ { + } > 1$ , the largest coherent eigenvalue carried by a populated class:

$$
\varphi _ { \mathrm { f r o z } } ( M ) = \frac { \beta } { 2 M } \Lambda _ { + } , \qquad I _ { 1 } ( \Lambda _ { + } ) = M \alpha , \quad \Lambda _ { + } > 1 .\tag{C33}
$$

At $M = 1$ one has $\Lambda _ { + } = 1 + \varepsilon _ { \operatorname* { m a x } } ( \alpha )$ , and Eq. (C33) reduces to Eq. (C8). The optimization over M is now elementary. On the annealed branch, $d \varphi _ { \mathrm { a n n } } / d M = \alpha - \beta / [ 2 M ( M - \beta ) ]$ tends to −∞ as $\bar { M } \to \beta ^ { + }$ and to $\alpha > 0$ at large M: the branch has an interior minimum in M, so its maximum is attained at the endpoints, $M = n$ (all copies dispersed) or $M = 1$ (all copies aligned). On the frozen branch, diferentiating the constraint in Eq. (C33) gives

$$
\frac { d \varphi _ { \mathrm { f r o z } } } { d M } = - \frac { \beta } { 2 M ^ { 2 } } \frac { \Lambda _ { + } \log \Lambda _ { + } } { \Lambda _ { + } - 1 } < 0 ,\tag{C34}
$$

so freezing always prefers full alignment, $M = 1$ . Three bulk branches survive. For $M = n$ (using $\beta / n = \lambda )$ $\begin{array} { r } { \varphi _ { \mathrm { P } } = n \alpha - \frac { 1 } { 2 } \log ( 1 - \lambda ) } \end{array}$ . For $M = 1$ annealed, $\begin{array} { r } { \varphi _ { \mathrm { C } } = \alpha - \frac { 1 } { 2 } \log ( 1 - \beta ) } \end{array}$ with validity $\beta < 1$ and $\alpha \geq \kappa ( \beta )$ . For $M = 1$ frozen, $\begin{array} { r } { \varphi _ { \mathrm { F } } = \frac { \beta } { 2 } ( 1 + \varepsilon _ { \mathrm { m a x } } ) } \end{array}$ . Through $f = - T \varphi$ these are the free energies Eq. (65). The entropy of the C branch is $s _ { \mathrm { C } } = - \partial f _ { \mathrm { C } } \bar { / } \partial T = \alpha - \kappa ( \beta )$ , which vanishes exactly on the freezing line Eq. (66). There $\beta _ { f } = \varepsilon _ { \mathrm { m a x } } / ( 1 + \varepsilon _ { \mathrm { m a x } } )$ and substituting into either branch gives the common value $\varphi = \varepsilon _ { \mathrm { m a x } } / 2$ , so C and F connect continuously, the REM freezing scenario in the form used in Appendix C 2. By the conditional Gaussian Eq. (C17), the F state is the complete retrieval state of the maximal-norm pattern, as asserted in the main text. The P state, by contrast, is a weak thermal condensate: the tilted Gram matrix of the dispersed patterns follows from the stationary covariance $Q ^ { * } = ( I - 2 \Theta ) ^ { - 1 }$ by the Sherman–Morrison formula, $q ^ { * } = \lambda ^ { 2 } \bar { T } / ( 1 - \bar { \lambda } )$ of the diagonal, so the centroid in Eq. (C17) carries $\begin{array} { r } { \left. \frac { 1 } { n } \sum _ { j } \xi _ { \mu _ { j } } ^ { ( h , v ) } \right. ^ { 2 } / N _ { v } = \Lambda _ { 1 } ^ { * } / n = \lambda T / ( 1 - \lambda ) } \end{array}$ , and adding the thermal variance T of the Gaussian fluctuations,

$$
\frac { \left. \| v \| ^ { 2 } \right. } { N _ { v } } = \frac { \lambda T } { 1 - \lambda } + T = \frac { T } { 1 - \lambda } ,\tag{C35}
$$

the value quoted in Sec. IV B: even the paramagnet weakly aligns the selected patterns $\left( q ^ { * } > 0 \right)$ , a linear-response enhancement that vanishes as $T  0$ . Finally, the existence conditions unify: a symmetric class has attention weights $1 / M$ on M patterns, hence inverse participation ratio $\left\| f \right\| ^ { 2 } = 1 / M$ , and the annealed condition $\beta < M$ is $\beta \| f \| ^ { 2 } < 1$ For P this reads $\lambda < 1$ and for C it reads $\beta < 1$ , and the instability at $\beta \| f \| ^ { 2 } = 1$ is the divergence of the visible Gaussian integral in Eq. (56), i.e. the softening of the visible fluctuations by concentrated attention. For $\lambda \geq 1$ the paramagnetic branch does not exist.

Retrieval branch. Condition on the retrieved pattern $\xi _ { 1 } ^ { ( h , v ) }$ , of typical norm, and consider the mixed classes of Sec. IV B: np copies on $\xi _ { 1 } ^ { ( h , v ) }$ and $N ^ { \prime } : = n ( 1 - p )$ copies dispersed on distinct patterns $\mu \neq 1$ , with overlaps $c _ { j } = \xi _ { \mu _ { j } } ^ { ( h , v ) } \cdot \xi _ { 1 } ^ { ( h , v ) } / N _ { v }$ and mutual Gram matrix Q<sup>′</sup>. The energy depends only on $\textstyle \sum _ { j } c _ { j }$ and $\mathbf { 1 } ^ { \top } Q ^ { \prime } \mathbf { 1 }$ , so the Jensen argument Eq. (C30), supplemented by the Cauchy–Schwarz inequality for the $c _ { j } .$ , again makes the uniform and symmetric form $( c _ { j } = c , Q ^ { \prime } = ( q _ { d } - q ) I + q \mathbf { 1 1 } ^ { \top } )$ exact within the family. The counting rate conditioned on $\xi _ { 1 } ^ { ( h , v ) }$ follows from the longitudinal–transverse decomposition $\xi _ { \mu _ { j } } ^ { ( h , v ) } = \ell _ { j } e _ { 1 } + \xi _ { j } ^ { \perp }$ with $e _ { 1 } = \xi _ { 1 } ^ { ( h , v ) } / \left\| \xi _ { 1 } ^ { ( h , v ) } \right\|$ . The longitudinal components are scalar unit Gaussians constrained to $\ell _ { j } = c \sqrt { N _ { v } } ,$ , at rate $c ^ { 2 } / 2$ each, while the transverse vectors are i.i.d. standard Gaussian in the orthogonal complement with mutual overlaps $\xi _ { j } ^ { \perp } \cdot \xi _ { l } ^ { \perp } / N _ { v } = Q _ { j l } ^ { \prime } - c ^ { 2 }$ , at rate

$I _ { N ^ { \prime } } ( Q ^ { \prime } - c ^ { 2 } { \bf 1 1 } ^ { \top } )$ . Adding the two, the longitudinal cost cancels against the trace shift,

$$
\begin{array} { l } { { \displaystyle { I _ { \mathrm { c o n d } } = \frac { N ^ { \prime } c ^ { 2 } } { 2 } + I _ { N ^ { \prime } } \big ( Q ^ { \prime } - c ^ { 2 } { \bf 1 1 } ^ { \top } \big ) } } \ ~ } \\ { { \displaystyle ~ = \frac { 1 } { 2 } \Big [ \mathrm { T r } Q ^ { \prime } - \log \operatorname* { d e t } \big ( Q ^ { \prime } - c ^ { 2 } { \bf 1 1 } ^ { \top } \big ) - N ^ { \prime } \Big ] } , } \end{array}\tag{C36}
$$

so only the Schur complement survives in the log-determinant. Equivalently, apply Eq. (64) to the full $( N ^ { \prime } + 1 )$ -tuple including $\xi _ { 1 } ^ { ( h , v ) }$ and use det $Q _ { \mathrm { f u l l } } = \operatorname* { d e t } ( Q ^ { \prime } - c ^ { 2 } \mathbf { 1 } \mathbf { 1 } ^ { \top } )$ together with the vanishing conditioning cost $I _ { 1 } ( 1 ) = 0$ . In the symmetric form the eigenvalues of $\dot { Q } ^ { \prime } - c ^ { 2 } { \bf 1 1 } ^ { \top }$ are $\tilde { \Lambda _ { 1 } ^ { \prime } } - N ^ { \prime } c ^ { 2 }$ (coherent, $\Lambda _ { 1 } ^ { \prime } \stackrel { - } { = } q _ { d } + ( N ^ { \prime } - \bar { 1 } ) q )$ and $\Lambda _ { 2 } ^ { \prime } = q _ { d } - q$ (transverse), so

$$
I _ { \mathrm { c o n d } } = \left( N ^ { \prime } - 1 \right) I _ { 1 } ( \Lambda _ { 2 } ^ { \prime } ) + \frac { 1 } { 2 } \Bigl [ \Lambda _ { 1 } ^ { \prime } - \log \left( \Lambda _ { 1 } ^ { \prime } - N ^ { \prime } c ^ { 2 } \right) - 1 \Bigr ] ,\tag{C37}
$$

where the trace carries $\Lambda _ { 1 } ^ { \prime }$ but the logarithm its Schur shift: the mismatch is the imprint of the longitudinal cost. The energy exponent follows from $\begin{array} { r } { \left\| n p \xi _ { 1 } ^ { ( h , v ) } + \sum _ { j } \xi _ { \mu _ { j } } ^ { ( h , v ) } \right\| ^ { 2 } / N _ { v } = ( n p ) ^ { 2 } + 2 n p N ^ { \prime } c + N ^ { \prime } \Lambda _ { 1 } ^ { \prime } } \end{array}$ , multiplied by $\lambda / 2 n \colon$

$$
\frac { \beta } { 2 } p ^ { 2 } + \beta p ( 1 - p ) c + \frac { \lambda ( 1 - p ) } { 2 } \Lambda _ { 1 } ^ { \prime } .\tag{C38}
$$

The coeficient of the last term is $\lambda ,$ not $\beta \colon$ the coherent enhancement of a dispersed copy is that of a single copy, and this asymmetry between the condensed group and the leak drives the entire structure below. Assembling the selection entropy $N ^ { \prime } \alpha$ , the rate Eq. (C37), and the energy Eq. (C38), and introducing $A : = ( \Lambda _ { 1 } ^ { \prime } - N ^ { \prime } c ^ { 2 } ) ^ { - 1 }$ , the stationarity conditions are elementary. The transverse eigenvalue carries no energy, so $I _ { 1 } ^ { \prime } ( \Lambda _ { 2 } ^ { \prime } ) = 0$ gives $\Lambda _ { 2 } ^ { \prime } = 1$ . The c derivative balances $- N ^ { \prime } c A$ from the logarithm against $\beta p ( 1 - p ) = \lambda p N ^ { \prime }$ from the cross term, and the $\Lambda _ { 1 } ^ { \prime }$ derivative balances $\textstyle - { \frac { 1 } { 2 } } + { \frac { A } { 2 } }$ against $\lambda ( 1 - p ) / 2$ . Hence

$$
\Lambda _ { 2 } ^ { \prime } = 1 , \qquad A = 1 - \lambda ( 1 - p ) , \qquad c = \frac { \lambda p } { A } .\tag{C39}
$$

Substituting back, the $\Lambda _ { 1 } ^ { \prime }$ terms combine into $\begin{array} { r } { - \frac { A } { 2 } \Lambda _ { 1 } ^ { \prime } = - \frac { 1 } { 2 } - \frac { A } { 2 } N ^ { \prime } c ^ { 2 } } \end{array}$ , the constant cancels, the two c-dependent terms add to $+ \textstyle { \frac { n \lambda ^ { 2 } } { 2 } } p ^ { 2 } ( 1 - p ) / A$ , and the identity $A + \lambda ( 1 - p ) = 1$ collapses the sum with the condensate self-energy into a single term:

$$
\varphi _ { \mathrm { R } } ( p ) = { \frac { \beta } { \lambda } } ( 1 - p ) \alpha - { \frac { 1 } { 2 } } \log A + \frac { \beta } { 2 } \frac { p ^ { 2 } } { A } ,\tag{C40}
$$

which is the Landau function Eq. (67) through $f _ { \mathrm { R } } = - T \varphi _ { \mathrm { R } }$ , with the endpoint values $\varphi _ { \mathrm { R } } ( 0 ) = \varphi _ { \mathrm { P } }$ and $\varphi _ { \mathrm { R } } ( 1 ) = \beta / 2$ quoted in the main text. The visible overlap follows from the centroid Eq. (C17),

$$
m = \frac { v \cdot \xi _ { 1 } ^ { ( h , v ) } } { N _ { v } } = p + \left( 1 - p \right) c = \frac { p } { A } , \qquad c = \lambda m ,\tag{C41}
$$

where the second form uses $A + \lambda ( 1 - p ) = 1 ~ { \mathrm { a g a i n } }$ . The relation $c =$ λm identifies the stationary overlap of the destinations as a linear response to the retrieval field, and the first form of Eq. (C41) is a self-consistent loop, $m = p + \lambda ( 1 - p )$ m: the leak amplifies the visible overlap by the geometric series of the loop gain $\lambda ( 1 - p )$ , and the positivity $A > 0$ of the loop stifness is once more the criterion $\beta \| f \| ^ { 2 } < 1$ , evaluated on the leak weights $\left\| f _ { \mathrm { l e a k } } \right\| ^ { 2 } = ( 1 - p ) / n$ . For later use we record the stationarity of Eq. (C40) in $p .$ . Using $1 / A = ( 1 - \lambda m ) / ( 1 - \lambda )$ , the condition $\partial _ { p } \varphi _ { \mathrm { { R } } } = 0$ closes in the visible overlap alone,

$$
\alpha = F _ { T } ( m ) : = \lambda m \Big ( 1 - \frac { \lambda m } { 2 } \Big ) - \frac { \lambda ^ { 2 } T } { 2 } \frac { 1 - \lambda m } { 1 - \lambda } ,\tag{C42}
$$

and $F _ { T }$ is strictly increasing on $m \in \ [ 0 , 1 ]$ for $\lambda \ < \ 1$ , so the interior stationary point $m ^ { \dagger }$ is unique. It is the barrier of Sec. IV B, with the zero-temperature root $\lambda m ^ { \dagger } = 1 - \sqrt { 1 - 2 \alpha }$ . The continuum endpoint criterion, $\begin{array} { r } { \alpha < F _ { T } ( 1 ) = \lambda - \frac { \lambda ^ { 2 } } { 2 } ( 1 + T ) } \end{array}$ , is what a smooth treatment of $p$ would predict for the spinodal, and the next paragraph corrects it.

One-copy defection and the spinodal. The attention weight moves on the lattice $p = 1 - k / n$ , with spacing $\lambda T$ that is not small at finite temperature, so the local stability of retrieval is decided by the nearest sector, $\varphi _ { \mathrm { R } } ( 1 ) \geq \varphi _ { \mathrm { R } } ( 1 - 1 / n )$ ， not by the endpoint slope. We evaluate the one-copy defection directly, which also serves as an independent check of $\operatorname { E q . }$ (C40). Move one copy from $\xi _ { 1 } ^ { ( h , v ) }$ to a pattern $\mu$ with overlap $x : = \xi _ { \mu } ^ { ( h , v ) } \cdot \xi _ { 1 } ^ { ( h , v ) } / N _ { v }$ and norm $\left. \xi _ { \mu } ^ { ( h , v ) } \right. ^ { 2 } / N _ { v } = 1 + \varepsilon$ Conditioned on $\xi _ { 1 } ^ { ( h , v ) }$ , the pair rate function is the $M = 2$ case of Eq. (64),

$$
I ( x , \varepsilon ) = { \frac { x ^ { 2 } } { 2 } } + I _ { 1 } { \big ( } 1 + \varepsilon - x ^ { 2 } { \big ) } ,\tag{C43}
$$

the longitudinal Gaussian cost plus the transverse norm cost. Note that conditioning on the overlap x makes the typical norm excess $x ^ { 2 }$ , at no additional cost. The energy gain of the defection is

$$
\begin{array} { l } { \displaystyle \Delta ( x , \varepsilon ) = \frac { \lambda } { 2 n } \Big [ ( n - 1 ) ^ { 2 } + 2 ( n - 1 ) x + 1 + \varepsilon - n ^ { 2 } \Big ] } \\ { \displaystyle = \tilde { \lambda } ( x - 1 ) + \frac { \lambda ^ { 2 } T } { 2 } \varepsilon , \qquad \tilde { \lambda } : = \lambda ( 1 - \lambda T ) , } \end{array}\tag{C44}
$$

an exchange term, the loss of alignment with the $n - 1$ copies left behind (the defector does not interact with itself, hence the thinning factor $1 - \lambda T )$ , and a norm term, an energy channel of order T that favors defection toward patterns of large norm. The defection sector carries the exponent $\begin{array} { r } { \operatorname* { s u p } _ { I \leq \alpha } [ \alpha - I ( x , \varepsilon ) + \Delta ( x , \varepsilon ) ] } \end{array}$ . In the annealed regime the interior stationary point is, in the variable $y : = 1 + \varepsilon - x ^ { 2 }$

$$
y ^ { * } = { \frac { 1 } { 1 - \lambda ^ { 2 } T } } , \qquad x ^ { * } = { \frac { \tilde { \lambda } } { 1 - \lambda ^ { 2 } T } } = { \frac { \lambda ( 1 - \lambda T ) } { 1 - \lambda ^ { 2 } T } } ,\tag{C45}
$$

and collecting the terms (the $1 / n ^ { 2 }$ contributions cancel exactly) gives

$$
\Delta \varphi _ { \mathrm { d e f } } = \alpha - \frac { \lambda ( 2 - \lambda - \lambda T ) } { 2 ( 1 - \lambda ^ { 2 } T ) } - \frac { 1 } { 2 } \log \big ( 1 - \lambda ^ { 2 } T \big ) ,\tag{C46}
$$

which coincides exactly with $\varphi _ { \mathrm { R } } ( 1 - 1 / n ) - \varphi _ { \mathrm { R } } ( 1 )$ computed from Eq. (C40): the collective derivation and the singledefection computation, which optimizes the destination geometry independently, agree term by term. Retrieval is locally stable while $\Delta \varphi _ { \mathrm { d e f } } < 0$ , which is the spinodal $\operatorname { E q . }$ (68). Its small- ${ \mathbf - } T$ expansion, $\begin{array} { r } { \alpha _ { c } = \lambda - \frac { \lambda ^ { 2 } } { 2 } - \frac { \lambda ^ { 2 } T } { 2 } \big [ 1 + ( 1 - \lambda ) ^ { 2 } ] + O ( T ^ { 2 } ) } \end{array}$ lies strictly below the continuum criterion below Eq. (C42) except at $\lambda = 1 \colon$ : the continuum treatment resolves barriers thinner than one attention quantum and therefore overestimates the capacity at any $T > 0$ , as stated in the main text. The defection saddle $\operatorname { E q . }$ (C45) also encodes the physics: the optimal destination is slightly aligned with the signal $( x ^ { * } > 0$ , the one-copy version of the linear response $c = \lambda m )$ and slightly long $( y ^ { * } > 1$ , the norm channel opened by the temperature).

Frozen ceiling. The annealed defection assumed that destinations with the optimal $( \boldsymbol { x } ^ { * } , \varepsilon ^ { * } )$ exist, i.e. $I ( x ^ { \ast } , \varepsilon ^ { \ast } ) \leq \alpha$ As $T \to 0 , I ( x ^ { * } , \varepsilon ^ { * } ) \to \lambda ^ { 2 } / 2$ , so for $\alpha \lesssim \lambda ^ { 2 } / 2$ (sharp attention) they do not, and the supremum is attained on the boundary $I = \alpha \colon$ the defection freezes onto the most favorable pattern actually present, the one-copy analogue of the frozen branch of Eq. (60). On the boundary the counting term vanishes and one maximizes $\Delta ( x , \varepsilon )$ alone subject to $I ( x , \varepsilon ) = \alpha$ . The Lagrange conditions, $\partial _ { x } \colon \tilde { \lambda } = \eta x / y$ and $\begin{array} { r } { \partial _ { \varepsilon } \colon \frac { \lambda ^ { 2 } T } { 2 } = \frac { \eta } { 2 } ( 1 - 1 / y ) } \end{array}$ , combine into

$$
y - 1 = { \frac { \lambda T x } { 1 - \lambda T } } :\tag{C47}
$$

the optimal extreme destination shifts toward norm excess as $T$ grows, and returns to the conditionally typical norm $( y  1 )$ at $T = 0$ . Since $I _ { 1 } ( y ) = O ( ( y - 1 ) ^ { 2 } )$ , the transverse cost enters the constraint only at ${ \dot { O } } ( T ^ { 2 } )$ , so $x = \sqrt { 2 \alpha } \left( 1 + O ( T ^ { 2 } ) \right)$ : the extreme overlap itself is temperature independent at this order, and the temperature enters only through the gain,

$$
\Delta = \lambda ( x - 1 ) + \lambda ^ { 2 } T \Bigl [ ( 1 - x ) + \frac { x ^ { 2 } } { 2 } \Bigr ] + { \cal O } ( T ^ { 2 } ) .\tag{C48}
$$

Near the threshold $x \approx 1$ the bracket equals $^ { \frac { 1 } { 2 } , }$ so stability $\Delta < 0$ requires $x < x _ { c } = 1 - \lambda T / 2 + O ( T ^ { 2 } )$ , i.e. $\alpha < x _ { c } ^ { 2 } / 2$ which is the ceiling Eq. (69). The two order $- \mathbf { \tilde { \boldsymbol { T } } }$ mechanisms quoted in the main text are the two terms of Eq. (C44), the exchange thinning of the condensate and the norm channel. The boundary optimization ranges over all patterns actually present, so it automatically includes the defection toward the maximal-norm pattern $( x \approx 0 , \varepsilon = \varepsilon _ { \mathrm { m a x } } )$ , the first step of nucleation toward the $\mathrm { F }$ phase. The solution of $\operatorname { E q . }$ (C47) dominates it at low temperature, so the one-copy instability channels are exhausted by this computation. The finite-temperature capacity thus follows Eq. (68) on the annealed side and $\operatorname { E q . }$ (69) on the frozen side of the branch switch at $\alpha { \stackrel { . } { \approx } } \lambda ^ { 2 } / 2 + { \bar { O } } ( T )$ , as in Sec. IV B. The switch is the finite-temperature continuation of the branch point $\lambda = \sqrt { 2 \alpha }$ of Eq. (60).

## 4. Complete phase diagram for continuous temperature

The results of Appendix C 3 are exact at the integer temperature points $n = \beta / \lambda \in \mathbb { Z } _ { > 0 }$ . In this appendix we reconstruct the phase diagram from the visible representation, which is defined at every real temperature from the outset, and show that all formulas of the main text hold as stated for arbitrary real $\beta . ^ { 6 }$ Along the way the fate of the fluctuation determinant and of the hidden-sector zero mode is clarified, and the analytic continuation of the integer lattice is proved unique. Throughout, $s : = \beta / \lambda$ denotes the real (in the uniqueness argument, complex) continuation of the copy number, with $s = n$ on the lattice.

Constrained partition function. Fix the visible overlap $m = v \cdot \xi _ { 1 } ^ { ( h , v ) } / N _ { v }$ with a typical pattern $\xi _ { 1 } ^ { ( h , v ) }$ as a reaction coordinate and define $\begin{array} { r } { Z ( \boldsymbol { m } ) : = \int \mathcal { D } \boldsymbol { v } \delta \big ( \boldsymbol { v } \cdot \boldsymbol { \xi } _ { 1 } ^ { ( h , v ) } / N _ { v } - m \big ) e ^ { - \beta E ( \boldsymbol { v } ) } } \end{array}$ , with E the efective energy Eq. (54) and $\mathcal { D } \boldsymbol { v }$ the Gaussian reference measure. The orthogonal decomposition $v = m \xi _ { 1 } ^ { ( h , v ) } + v _ { \perp }$ fixes the longitudinal component with unit Jacobian, and the fields become

$$
a _ { 1 } = \lambda N _ { v } m , \qquad a _ { \mu } = \lambda \bigl ( N _ { v } m x _ { \mu } + \xi _ { \mu } ^ { \perp } \cdot v _ { \perp } \bigr ) , \quad \mu \geq 2 ,\tag{C49}
$$

with $x _ { \mu } : = \xi _ { \mu } ^ { ( h , v ) } \cdot \xi _ { 1 } ^ { ( h , v ) } / N _ { v } , \xi _ { \mu } ^ { \perp } : = \xi _ { \mu } ^ { ( h , v ) } - x _ { \mu } \xi _ { 1 } ^ { ( h , v ) }$ , and $y _ { \mu } : = \left\| \xi _ { \mu } ^ { \perp } \right\| ^ { 2 } / N _ { v }$ . Splitting of the signal and writing the leak sum Eq. (59) in the present variables, $\begin{array} { r } { L = \sum _ { \mu \geq 2 } e ^ { a _ { \mu } } } \end{array}$ , together with $u : = e ^ { - a _ { 1 } } L$ ，

$$
Z ( m ) = e ^ { N _ { v } \beta ( m - m ^ { 2 } / 2 ) } \left. ( 1 + u ) ^ { s } \right. _ { \perp } ,\tag{C50}
$$

where $\langle \cdot \rangle _ { \perp }$ averages over $v _ { \perp } \sim \mathcal { N } ( 0 , \beta ^ { - 1 } I _ { N _ { v } - 1 } )$ and is normalized so that the term $u ^ { 0 }$ contributes 1. Conditioned on $\xi _ { 1 } ^ { ( h , v ) }$ , the variables $( x _ { \mu } , y _ { \mu } )$ are independent across $\mu ,$ , with joint rate $I ( x , y ) = { x ^ { 2 } } / { 2 } + I _ { 1 } ( y )$ , which is Eq. (C43) in the variables $y = 1 + \varepsilon - x ^ { 2 }$ . The counting dichotomy of Appendix C 1 applies to them unchanged.

Sector decomposition of the retrieval basin. By Newton’s generalized binomial theorem, for every real (indeed complex) exponent s,

$$
( 1 + u ) ^ { s } = \sum _ { k = 0 } ^ { \infty } \binom { s } { k } u ^ { k } , \qquad \binom { s } { k } = \frac { s ( s - 1 ) \cdot \cdot \cdot ( s - k + 1 ) } { k ! } ,\tag{C51}
$$

convergent for $u < 1$ . In the retrieval basin u is exponentially small, term-by-term evaluation is justified, and the expansion decomposes $Z ( m )$ into sectors labeled by the integer defection number $k ,$ of condensed weight $p _ { k } = 1 - k \lambda T$ the attention lattice and its quantum $\lambda T$ are properties of the model at every real temperature, not artifacts of integer $n .$ Only the combinatorial coeficients $\textstyle { \binom { s } { k } }$ are continued, and they are subexponential in $N _ { v }$ . The convergence condition $u < 1$ is the retrieval condition itself, so the expansion is precisely the excitation theory of the retrieval state. The $k = 0$ sector gives $\varphi _ { 0 } ( m ) = \beta ( m - m ^ { 2 } / 2 )$ , maximal at $m = 1$ with value $\beta / 2$ . For $k = 1$ , using the Gaussian generating function $\langle e ^ { \lambda \xi _ { \mu } ^ { \perp } \cdot v _ { \perp } } \rangle _ { \perp } = e ^ { N _ { v } ( \lambda ^ { 2 } T / 2 ) y _ { \mu } }$ and the counting dichotomy,

$$
\Delta _ { 1 } ( m ) = \operatorname * { s u p } _ { I ( x , y ) \leq \alpha } \Bigl [ \alpha - I ( x , y ) + \lambda m ( x - 1 ) + \frac { \lambda ^ { 2 } T } { 2 } y \Bigr ] .\tag{C52}
$$

The annealed stationary point in y is governed by the identity

$$
- I _ { 1 } ( y _ { \ast } ) + { \frac { t } { 2 } } y _ { \ast } = - { \frac { 1 } { 2 } } \log ( 1 - t ) , \qquad y _ { \ast } = { \frac { 1 } { 1 - t } } ,\tag{C53}
$$

here at $t = \lambda ^ { 2 } T$ , while the stationary overlap is $x _ { * } = \lambda m$ , so that

$$
\Delta _ { 1 } ( m ) = \alpha - \lambda m \Big ( 1 - \frac { \lambda m } { 2 } \Big ) - \frac { 1 } { 2 } \log \big ( 1 - \lambda ^ { 2 } T \big ) ,\tag{C54}
$$

valid while $I ( x _ { * } , y _ { * } ) = \lambda ^ { 2 } m ^ { 2 } / 2 + \kappa ( \lambda ^ { 2 } T ) \le \alpha$ . Maximizing $\varphi _ { 0 } + \Delta _ { 1 }$ over m gives $m _ { 1 } ^ { * } = ( 1 - \lambda T ) / ( 1 - \lambda ^ { 2 } T )$ , which is the overlap amplification $p _ { 1 } / A _ { 1 }$ of Eq. (C41) evaluated in this sector, and the value $\varphi _ { \mathrm { R } } ( 1 - \lambda T )$ , i.e. exactly the copy result Eq. (C40) at $p = 1 - 1 / n$ , now for arbitrary real $\beta .$ In particular the one-quantum stability criterion, hence the spinodal Eq. (68), holds at every real temperature. For general k with distinct destinations, the transverse average factorizes over the $N _ { v } - 1$ transverse coordinates. Per coordinate the destinations contribute $\begin{array} { r } { w = \sum _ { j = 1 } ^ { k } z _ { j } \sim \mathcal { N } ( 0 , k ) } \end{array}$ and

$$
\mathbb { E } e ^ { \frac { \lambda ^ { 2 } T } { 2 } w ^ { 2 } } = \left( 1 - k \lambda ^ { 2 } T \right) ^ { - 1 / 2 } ,\tag{C55}
$$

which resums the destination norms and their mutual response to all orders. The longitudinal tilts contribute $e ^ { N _ { v } \lambda ^ { 2 } m ^ { 2 } / 2 }$ each, independently, since m is held fixed. With the signal loss $e ^ { - k \lambda m N _ { v } }$ and the counting $e ^ { k \alpha N _ { v } }$

$$
\Delta _ { k } ( m ) = k \Big [ \alpha - \lambda m \Big ( 1 - \frac { \lambda m } { 2 } \Big ) \Big ] - \frac { 1 } { 2 } \log \big ( 1 - k \lambda ^ { 2 } T \big ) ,\tag{C56}
$$

and in terms of $p _ { k } = 1 - k \lambda T$ and $A _ { k } = 1 - \lambda ( 1 - p _ { k } ) = 1 - k \lambda ^ { 2 } T _ { \ddagger }$

$$
\begin{array} { l } { \displaystyle { \varphi _ { k } ( m ) = \varphi _ { 0 } ( m ) + \Delta _ { k } ( m ) } } \\ { \displaystyle { \quad = \frac { \beta } { \lambda } ( 1 - p _ { k } ) \alpha - \frac { 1 } { 2 } \log A _ { k } + \beta \Big ( p _ { k } m - \frac { A _ { k } } { 2 } m ^ { 2 } \Big ) . } } \end{array}\tag{C57}
$$

This is the two-variable Landau surface $\varphi ( m , p )$ of the model, derived exactly on the lattice $p _ { k } = 1 - k \lambda T$ at arbitrary real temperature. Maximizing over m returns Eq. (C40), so the entire retrieval branch of Appendix C 3 is recovered without any integer assumption. The m direction is exactly continuous and smooth, while the $p$ direction is quantized: the discreteness of the finite-temperature physics resides solely in the attention coordinate. The validity conditions are the annealed counting condition per quantum, $\alpha \ge \lambda ^ { 2 } m ^ { 2 } / \overset {  } { 2 } + \kappa ( \lambda ^ { 2 } T )$ , and the positivity $A _ { k } > 0$ , i.e. $k < 1 / ( \lambda ^ { 2 } T )$ The physical lattice ends at $p _ { k } \geq 0 ,$ , i.e. $k \leq s .$ , beyond which the binomial coeficients alternate in sign and the bulk is described by the dual expansion below.

Co-defection channels. The $u ^ { k }$ terms with coinciding destination indices describe $j$ copies defecting to the same pattern. The coherent coupling is enhanced by $j ^ { 2 }$ , giving

$$
\Delta _ { j } ^ { \mathrm { c o } } ( m ) = \alpha - j \lambda m + \frac { j ^ { 2 } \lambda ^ { 2 } m ^ { 2 } } { 2 } - \frac { 1 } { 2 } \log { \left( 1 - j ^ { 2 } \lambda ^ { 2 } T \right) } ,\tag{C58}
$$

with tilted overlap $x _ { * } = j \lambda m$ , which requires the existence condition $I = j ^ { 2 } \lambda ^ { 2 } m ^ { 2 } / 2 \le \alpha$ . Formally $\operatorname { E q . }$ (C58) would overtake $j$ separate defections for $\alpha < \lambda ^ { \bar { 2 } } m ^ { 2 }$ , but that region violates the existence condition already at $j = 2$ (which demands $\overset { \cdot } { \alpha } \geq 2 \lambda ^ { 2 } m ^ { 2 } )$ , so the annealed co-defection channel is never dominant where it is valid. The frozen co-defection, j quanta onto the most favorable pattern actually present, gains only $j$ times the single-quantum exchange term plus the $O ( T )$ norm enhancement, and therefore does not destabilize retrieval before the one-copy channel does. The spinodal thus remains the one-copy criterion, min of Eq. (68) and $\operatorname { E q } .$ (69), at every real temperature. The role of the co-defection ladder is instead to provide the nucleation path toward the condensed phases, as shown below. Similarly, the frozen-leak evaluation of Appendix C 3 carries over at fixed m: the boundary optimization gives $\Delta _ { 1 } ^ { \mathrm { f r o z } } ( m ) = \lambda m ( \bar { \sqrt { 2 \alpha } } - 1 ) + \lambda ^ { 2 } T / 2 + O ( T ^ { 2 } )$ , and relaxing m to its optimum $m _ { * } = 1 - \lambda T ( 1 - \sqrt { 2 \alpha } )$ yields

$$
\Delta ^ { \mathrm { f r o z } } = \lambda ( \sqrt { 2 \alpha } - 1 ) + \lambda ^ { 2 } T \bigl [ 1 - \sqrt { 2 \alpha } + \alpha \bigr ] + O ( T ^ { 2 } ) ,\tag{C59}
$$

which agrees term by term with Eq. (C48) at $x = \sqrt { 2 \alpha }$ (the exchange factor $1 - \lambda T$ of the copy computation reappears here as the m-relaxation term: the grouping of terms difers, the total does not), and reproduces the ceiling Eq. (69) at real temperature.

Bulk branches and endpoint closure. Outside the basin $( u > 1 )$ one expands dually, $( e ^ { a _ { 1 } } + L ) ^ { s } = L ^ { s } ( 1 + u ^ { - 1 } ) ^ { s } $ leak-dominated states into which signal is injected. For the paramagnetic branch the annealed leak exponent $N _ { v } ^ { - 1 } \log L = \alpha + \lambda ^ { 2 } \rho ^ { 2 } / 2$ is linear in $\rho ^ { \mathrm { 2 } } = \left\| v \right\| ^ { 2 } / \bar { N } _ { v } .$ , so the visible integral is exactly Gaussian and

$$
\varphi _ { \mathrm { P } } ( m ) = s \alpha - \frac { 1 } { 2 } \log ( 1 - \lambda ) - \frac { \beta ( 1 - \lambda ) } { 2 } m ^ { 2 } ,\tag{C60}
$$

maximal at $m = 0$ where it reproduces $\varphi _ { \mathrm { P } }$ of Appendix C 3 with $n  s$ real. The longitudinal stifness $\beta ( 1 - \lambda )$ is the inverse of the condensate size Eq. (C35). For the condensed sector, a state $v \approx \xi _ { \mu } ^ { ( h , \bar { v } ) }$ with overlap $x = m$ and norm $1 + \varepsilon = m ^ { 2 } + y$ carries

$$
\varphi _ { \mathrm { c o n d } } ( m ) = \operatorname* { s u p } _ { y : \frac { m ^ { 2 } } { 2 } + I _ { 1 } ( y ) \leq \alpha } \Big [ \alpha - \frac { m ^ { 2 } } { 2 } - I _ { 1 } ( y ) + \frac { \beta } { 2 } \big ( m ^ { 2 } + y \big ) \Big ] .\tag{C61}
$$

The interior stationary point, $y _ { * } = ( 1 - \beta ) ^ { - 1 }$ for $\beta < 1$ by Eq. (C53) at $t = \beta$ , gives

$$
\varphi _ { \mathrm { C } } ( m ) = \alpha - \frac { 1 } { 2 } \log ( 1 - \beta ) - \frac { 1 - \beta } { 2 } m ^ { 2 } ,\tag{C62}
$$

whose existence condition $\beta < 1$ appears automatically. The boundary evaluation, $I _ { 1 } ( y ) = \alpha - m ^ { 2 } / 2$ , gives

$$
\varphi _ { \mathrm { F } } ( m ) = \frac { \beta } { 2 } \Bigl [ m ^ { 2 } + 1 + \varepsilon _ { \mathrm { m a x } } \Bigl ( \alpha - \frac { m ^ { 2 } } { 2 } \Bigr ) \Bigr ] , \qquad m ^ { 2 } \le 2 \alpha ,\tag{C63}
$$

where $\varepsilon _ { \operatorname* { m a x } } ( a )$ solves $I _ { 1 } ( 1 + \varepsilon ) = a$ . Since $\partial _ { m } \varphi _ { \mathrm { F } } = - \beta m / \varepsilon _ { \mathrm { m a x } } < 0$ , the frozen branch is maximal at $m = 0 ;$ the extreme-norm pattern is typically orthogonal to $\xi _ { 1 } ^ { ( h , v ) }$ . The freezing line at fixed m is $\kappa ( \beta ) = \alpha - m ^ { 2 } / 2$ , reducing at $m = 0$ to Eq. (66). The decisive structural fact is that the defection lattice closes exactly onto these bulk branches. Substituting the terminus $k = n$ of the distinct-defection ladder (at integer $n ,$ with $p = 0 , k \lambda = \beta _ { \mathrm { { } } }$ and $k \lambda ^ { 2 } T = \lambda )$ into Eq. (C57),

$$
\varphi _ { n } ( m ) = n \alpha - \frac { 1 } { 2 } \log ( 1 - \lambda ) - \frac { \beta ( 1 - \lambda ) } { 2 } m ^ { 2 } = \varphi _ { \mathrm { P } } ( m ) ,\tag{C64}
$$

and the terminus $j = n$ of the co-defection ladder, whose gain $- { \textstyle { \frac { \beta } { 2 } } } m ^ { 2 } + \beta m x + { \textstyle { \frac { \beta } { 2 } } } y$ completes the square as $\frac { \beta } { 2 } \left[ 1 + \varepsilon - \right.$ $( m - x ) ^ { 2 } ]$ , is optimized on the boundary at $x = m$ and closes onto Eq. (C63) (interior stationarity giving Eq. (C62) instead). The retrieval state is thus connected to the bulk phases by two quantized nucleation ladders living on a single surface, distinct defections leading to $\mathrm { P }$ and co-defections to C or F. At non-integer s the terminus $k = s$ is not a lattice point, but the dual expansion supplies the bulk values independently, and they agree with the $p  0$ values of $\operatorname { E q } .$ (C57), so the two expansions connect without a gap.

Fluctuation determinant, zero mode, and uniqueness. Three structural questions raised in Appendix C 2 are settled by this construction.

(a) One determinant, two interpretations. The only Gaussian fluctuation determinant surviving in the entire construction is the transverse visible one, and it admits two equivalent evaluations. Averaging over the destination patterns first, $\begin{array} { r } { \mathbb { E } _ { \xi ^ { \bot } } \exp [ \lambda \sum _ { j } \xi _ { j } ^ { \bot } \cdot v _ { \bot } ] = \exp [ \frac { k \lambda ^ { 2 } } { 2 } \| v _ { \bot } \| ^ { 2 } ] } \end{array}$ , which renormalizes the transverse stifness $\beta  \beta - k \lambda ^ { 2 } = \beta A _ { k }$ The ratio to the reference measure is

$$
\left( \frac { \beta } { \beta - k \lambda ^ { 2 } } \right) ^ { ( N _ { v } - 1 ) / 2 } = A _ { k } ^ { - ( N _ { v } - 1 ) / 2 } \implies - \frac { 1 } { 2 } \log A _ { k } ,\tag{C65}
$$

so the logarithm in $\operatorname { E q } .$ (C56) is the determinant of the visible transverse fluctuations softened by the attention leak, $A _ { k } > 0$ is their stability condition, and $A _ { k } \to 0$ at $k = s$ is the point $\lambda  1 \colon$ : the paramagnetic instability. Averaging over $v _ { \perp }$ first instead gives the Gram-fluctuation evaluation Eq. (C55), whose large-deviation form is the variational problem sup $\begin{array} { r } { { \bf \nabla } _ { Q ^ { \perp } } [ - I _ { k } ( Q ^ { \perp } ) + \frac { \lambda ^ { 2 } T } { 2 } { \bf 1 } ^ { \top } Q ^ { \perp } { \bf 1 } ] } \end{array}$ . The symmetric decomposition puts the transverse eigenvalues at 1 and the coherent one at $\Lambda _ { * } = ( 1 - k \bar { \lambda ^ { 2 } } T ) ^ { - 1 }$ , with the same value by Eq. (C53). The two interpretations are the two sides of the Legendre pair Eq. (C21) realized sector by sector: the energy-side log det of a replica computation and the entropy-side rate function of the counting derivation are one object viewed from its two convex-dual sides.

(b) The fate of the zero mode. No pseudo-determinant appears above, and the reason is instructive. The visible representation is written in terms of the log-sum-exp, i.e. after the softmax gauge orbit of Appendix C 2 has been fixed. The divergent zero-mode volume $V _ { 0 }$ of Eq. (C13) is a configuration-independent constant absorbed into the reference measure. Conversely, a Gaussian fluctuation expansion of the f representation around a retrieval vertex is structurally degenerate: by Eq. (C12), det<sup>′</sup> $\begin{array} { r } { \mathrm { H e s s } = N _ { h } \prod _ { \mu } \bar { f } _ { \mu } \to 0 } \end{array}$ as $f  e _ { 1 }$ , since every non-condensed weight is exponentially small, reflecting the divergence of the entropy curvature $\partial _ { f } ^ { 2 } ( f \log f ) = 1 / f$ at the boundary of the simplex. A Gaussian theory of fluctuations around the vertex therefore does not exist. The correct fluctuation theory is the discrete sector decomposition itself, whose elementary excitations are the quantized defections of attention weight λT and whose exact generating function is the binomial series Eq. (C51). This also sharpens the distinction, noted in Sec. $\mathrm { { I V A 1 } , }$ from a variant model in which the attention entropy is rescaled by hand to be extensive: such a model equilibrates at an interior saddle point with macroscopically spread attention, where $1 / f$ remains finite and the Gaussian expansion is legitimate. The two models difer already at the level of their fluctuation theories, discrete versus Gaussian.

(c) Uniqueness of the continuation. At finite $N _ { v }$ and fixed disorder, $Z ( m ; \beta )$ is defined by the visible integral for every real $\beta > 0$ and is analytic in $\beta .$ The multinomial expansion at integer n and the binomial expansion Eq. (C51) are exact rewritings of this single function on their respective domains, so within the retrieval basin there is no continuation to choose. The only question is the exchange of the thermodynamic limit with the continuation, and in the basin the series Eq. (C51) converges locally uniformly in $\beta _ { ; }$ the dominant sector index is finite, and each $\Delta _ { k }$ is analytic by Eq. (C56), so $\varphi ( m ; \beta )$ is analytic up to branch switches and continuous across them. Agreement with the copy values at the integer points was verified at $\operatorname { E q . }$ (C54). A stronger statement holds: the integer-point data alone already determine the answer. By Carlson’s theorem [59, Sec. 9], a function analytic in $\Re s \geq 0 .$ of exponential type with $| \dot { F } ( s ) | \le C e ^ { \tau | s }$ <sup>|</sup> for some $\tau < \pi ,$ and vanishing on the non-negative integers, vanishes identically. In the basin, $| ( 1 + u ) ^ { s } | ^ { \cdot } = ( 1 + u ) ^ { \mathfrak { R } s } \leq 2 ^ { \mathfrak { R } s }$ , so the candidate functions have type at most log $2 < \pi .$ , and the diference of any two continuations that agree on the lattice is identically zero. The growth condition is what excludes the spurious continuations sin(πs), of type exactly $\pi .$ . With it, the caveat usually attached to integer-power representations is removed for the entire retrieval basin.

TABLE III: Branches of the constrained rate $\varphi ( m )$ at real temperature. The retrieval sectors live on the attention lattice $p _ { k } = 1 - k \lambda T . \mathrm { ~ P } , \mathrm { ~ C } ,$ and F are maximal at $m = 0$ , and R at $p = 1$
<table><tr><td>Branch</td><td> $\varphi ( m )$ </td><td>Validity</td></tr><tr><td>R sectors</td><td>Eq. (C57)</td><td> $\begin{array} { r } { u < 1 , A _ { k } > 0 , \alpha \geq \frac { \lambda ^ { 2 } m ^ { 2 } } { 2 } + \kappa ( \lambda ^ { 2 } T ) } \end{array}$ </td></tr><tr><td>R frozen leak</td><td>Eq. (C59)</td><td> $\begin{array} { r } { u < 1 , \alpha < \frac { \lambda ^ { 2 } m ^ { 2 } } { 2 } + O ( T ^ { 2 } ) } \end{array}$ </td></tr><tr><td>P</td><td>Eq. (C60)</td><td> $\lambda < 1$ </td></tr><tr><td>C</td><td>Eq. (C62)</td><td> $\begin{array} { r } { \beta < 1 , \alpha \ge \kappa ( \beta ) + \frac { m ^ { 2 } } { 2 } } \end{array}$ </td></tr><tr><td>F</td><td> $\operatorname { E q . }$  (C63)</td><td> $m ^ { 2 } \leq 2 \alpha$ </td></tr></table>

Assembly of the phase diagram. Table III collects the branches constructed above with their validity conditions.   
All live on the single m axis and are connected through the defection lattice by the endpoint closure.

The spinodal read of the table is the one-quantum criterion established above, with the annealed and frozen branches $\operatorname { E q . }$ (68) and Eq. (69), now as a statement at every real temperature. The two branches follow from a single expression: eliminating m at its saddle, $1 - m = \lambda T ( 1 - x )$ , reduces the one-quantum criterion to the variational problem

$$
\Delta ( \alpha ; \lambda , T ) = \operatorname* { m a x } _ { I ( x , y ) \leq \alpha } \Big [ \alpha - I ( x , y ) - \lambda ( 1 - x ) + \frac { \lambda ^ { 2 } T } { 2 } \big ( ( 1 - x ) ^ { 2 } + y \big ) \Big ] ,\tag{C66}
$$

with retrieval locally stable while $\Delta \ < \ 0 .$ The interior stationary point of Eq. (C66) reproduces Eq. (68), the boundary evaluation reproduces Eq. $( 6 9 )$ , the switch between the two is the existence line of the annealed destination, $\alpha = I ^ { * } ( T ) : = ( x ^ { * } ) ^ { 2 } / 2 + \kappa ( \lambda ^ { 2 } T )$ with $x ^ { * }$ the annealed saddle Eq. (C45) (the dash-dotted curve of the figures), and the root $\Delta = 0$ is the spinodal drawn in Fig. 4 and Fig. 5. The same surface carries the barrier. The increments of Eq. (C56) in k start at $\alpha - \lambda m ( 1 - \lambda m / 2 )$ and grow convexly through the logarithm, so for $\alpha < \alpha _ { c }$ the sector weights first decrease and then increase along the ladder. The continuum stationary point is $\operatorname { E q . }$ (C42), and the barrier height of the retrieval state is

$$
\Delta \varphi _ { \mathrm { b } } = \frac { \beta } { 2 } - \varphi _ { \mathrm { R } } ( p ^ { \dagger } ) ,\tag{C67}
$$

with $p ^ { \dagger }$ the sector corresponding to the root $m ^ { \dagger }$ of Eq. (C42) through $m ^ { \dagger } = p ^ { \dagger } / A ^ { \dagger }$ . At $T = 0 , \lambda m ^ { \dagger } = 1 - { \sqrt { 1 - 2 \alpha } }$ gives a closed form, and at finite $T$ a single numerical root sufices. In the Arrhenius sense $\tau _ { \mathrm { e s c } } \sim e ^ { N _ { v } \Delta \varphi _ { \mathrm { b } } }$ , the barrier turns the equilibrium construction into the lifetime map of Fig. 5: the contours of $\Delta \varphi _ { \mathrm { E } }$ fan out from the spinodal, on which the barrier vanishes, and grow toward small α and low temperature, so that retrieval is protected by an order-one rate barrier already at moderate depth inside the metastable region. The insets resolve the competition between the two ladders of the endpoint closure at a fixed reference point. For $\lambda = 0 . 5$ the co-defection ladder has both the lower barrier (0.19 against 0.34 at the starred point) and a terminus, the frozen value $\varphi _ { \mathrm { F } }$ , far above the retrieval value $\beta / 2$ , while the distinct ladder terminates at a paramagnetic value below $\beta / 2 \colon$ escape toward P is closed there and opens only at larger $\alpha ,$ , where $\varphi _ { \mathrm { P } }$ exceeds $\varphi _ { \mathrm { R } } .$ , so distinct defections dominate the escape at large load and co-defections at small load and low temperature. For $\lambda = 1 . 5$ the inset shows the distinct ladder running into the divergence of the determinant $- \frac 1 2$ log $A _ { k }$ as $A _ { k } \to 0$ , so the escape proceeds through the co-defection ladder alone. The general statement is given below. The equilibrium boundaries follow by comparing the m-optimized branches. Equating $\operatorname { E q . }$ (C60) and Eq. (C63) at $m = 0$ gives the P–F line

$$
\frac { \alpha } { \lambda } + \frac { T } { 2 } \bigl | \log ( 1 - \lambda ) \bigr | = \frac { 1 + \varepsilon _ { \mathrm { m a x } } ( \alpha ) } { 2 } ,\tag{C68}
$$

whose $T  0$ limit is the zero-temperature equilibrium boundary of Appendix C 1. Equating Eq. (C60) and Eq. (C62) gives the P–C line

$$
\left( s - 1 \right) \alpha = \frac { 1 } { 2 } \log \frac { 1 - \lambda } { 1 - \beta } ,\tag{C69}
$$

![](images/a38aa8014a850ba5af215673ea0ee9157553f520d09bee34bdc780c05ecd6776.jpg)  
FIG. 5: Same as Fig. 4, with the dynamical content of the real-temperature construction overlaid. The blue dashed curves are contours of the escape barrier Eq. (C67), evaluated along the lower of the two escape ladders, at the levels $\Delta \varphi _ { \mathrm { b } } = 0 . 0 5 , 0 . 1 , 0 . 2 , 0 . 5 .$ , and 1. The barrier vanishes on the spinodal and grows toward small α and low temperature. The insets show the constrained rate φ along the two escape ladders at the starred reference points,  
$( \alpha , T ) = ( 0 . 1 5 , 0 . 4 0 )$ in (a) and (0.14, 0.27) in (b), against the transferred attention weight $1 - p \colon$ the distinct-defection ladder (blue, annealed curve with rungs at $p _ { k } = 1 - k \lambda T )$ ends at the paramagnetic value, and the co-defection ladder (green) at the condensed value, whose frozen terminus reproduces $\varphi _ { \mathrm { F } }$ exactly. In (b) the distinct ladder is cut of by the transverse stability condition $A _ { k } > 0$ before reaching its terminus, and only the co-defection ladder connects retrieval to the bulk.

both first order, while C–F is the continuous freezing line Eq. (66). The status of retrieval is unchanged from the main text: $\begin{array} { r } { \varphi _ { \mathrm { F } } - \varphi _ { \mathrm { R } } ( 1 ) = \frac { \beta } { 2 } \varepsilon _ { \mathrm { m a x } } > 0 } \end{array}$ at every $\alpha > 0$ and every temperature, so typical retrieval never becomes the equilibrium phase for Gaussian patterns. For $\lambda \geq 1$ the transverse stability $A _ { k } > 0$ fails at $k = 1 / ( \lambda ^ { 2 } T ) \le s \colon$ the distinct-defection ladder terminates before reaching its P terminus, which is the ladder-level manifestation of the absence of the paramagnetic phase. The equilibrium is then C or F alone, with retrieval metastable below the ceiling Eq. (69). This is the structure displayed in Fig. 4 and, with the barrier contours and escape ladders overlaid, in Fig. 5, now established at all real temperatures: relative to the integer-lattice derivation, every curve is promoted from an interpolation to an exact solution, the restriction that the lattice caps the accessible temperatures at $T = 1 / \lambda$ disappears (so the C region stands for every λ), and the barrier Eq. (C67) adds dynamical information not visible in the equilibrium diagram.

Remarks and open problems. We close with the points left open by the present analysis. (i) Hidden-sector corrections. The shifts of Eq. (C16), the exponent $\beta / \lambda  \beta / \lambda + N _ { h } / 2$ with its constraint $\bar { n } > N _ { h } / 2$ and the pattern shift $\begin{array} { r } { - \frac { 1 } { 2 } \sum _ { \nu } \xi _ { \nu } ^ { ( h , v ) } } \end{array}$ , are not small at exponential load. Whether they can be absorbed as a change of the reference measure within the sector decomposition, leaving the rate-level phase diagram intact, is the main structural question left open. (ii) The basin edge. At $u \to 1$ the binomial coeficients alternate in sign and the series requires resummation. A uniform treatment of the basin edge, e.g. through the Mellin–Barnes representation of $( 1 + u ) ^ { s } ~ [ 6 0 ]$ , would give the precise structure on the spinodal line itself, though the location of the spinodal, fixed by the $k = 0$ versus $k = 1$ comparison, is not afected. (iii) Subexponential corrections. The prefactors of  <sup>s</sup>, the fluctuation determinants collected here only at rate level, and the resulting finite-size shifts of $\alpha _ { c } ,$ of order log $N _ { v } / N _ { v }$ , are relevant to numerical verification of the T slopes in Eq. (68) and Eq. (69). (iv) Stability beyond one step. Within each sector the freezing prescription is exact (Appendix C 2), but fluctuations between sectors (an analogue of the AT condition) and the predicted Ruelle statistics of the F-phase attention weights [36] remain unexamined. (v) Ensemble dependence. For spherical patterns $\varepsilon _ { \operatorname* { m a x } } \equiv 0$ and the F phase loses its advantage, so typical retrieval can equilibrate, while for binary patterns the Gaussian overlap rate $x ^ { 2 } / 2$ is replaced by a binary entropy. Redoing the construction for these ensembles and matching [11, 15] quantitatively would separate what is universal in the phase diagram from what is a norm-fluctuation efect. (vi) Dynamics. The barrier Eq. (C67) and the two escape ladders are equilibrium statements about a reaction coordinate. Connecting them to genuine relaxation dynamics, nucleation prefactors, and the multi-step retrieval dynamics of attention networks is left for future work.

[1] J. J. Hopfield, Neural networks and physical systems with emergent collective computational abilities., Proceedings of the National Academy of Sciences 79, 2554 (1982).

[2] D. J. Amit, H. Gutfreund, and H. Sompolinsky, Spin-glass models of neural networks, Physical Review A 32, 1007 (1985).

[3] D. J. Amit, H. Gutfreund, and H. Sompolinsky, Storing infinite numbers of patterns in a spin-glass model of neural networks, Physical Review Letters 55, 1530 (1985).

[4] D. J. Amit, H. Gutfreund, and H. Sompolinsky, Statistical mechanics of neural networks near saturation, Annals of Physics 173, 30 (1987).

[5] D. Krotov, A new frontier for hopfield networks, Nature Reviews Physics 5, 366 (2023).

[6] D. Krotov, B. Hoover, P. Ram, and B. Pham, Modern methods in associative memory, arXiv preprint arXiv:2507.06211 (2025).

[7] E. Gardner, Multiconnected neural network models, Journal of Physics A: Mathematical and General 20, 3453 (1987).

[8] L. F. Abbott and Y. Arian, Storage capacity of generalized networks, Physical Review A 36, 5091 (1987).

[9] P. Baldi and S. S. Venkatesh, Number of stable points for spin-glasses and neural networks of higher orders, Physical Review Letters 58, 913 (1987).

[10] D. Krotov and J. J. Hopfield, Dense associative memory for pattern recognition, in Advances in Neural Information Processing Systems, Vol. 29 (2016).

[11] M. Demircigil, J. Heusel, M. Löwe, S. Upgang, and F. Vermet, On a model of associative memory with huge storage capacity, Journal of Statistical Physics 168, 288 (2017).

[12] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. u. Kaiser, and I. Polosukhin, Attention is all you need, in Advances in Neural Information Processing Systems, Vol. 30 (2017).

[13] H. Ramsauer, B. Schäfl, J. Lehner, P. Seidl, M. Widrich, L. Gruber, M. Holzleitner, T. Adler, D. Kreil, M. K. Kopp, G. Klambauer, J. Brandstetter, and S. Hochreiter, Hopfield networks is all you need, in International Conference on Learning Representations (2021).

[14] T. Ota and R. Karakida, Attention in a family of Boltzmann machines emerging from modern Hopfield networks, Neural Computation 35, 1463 (2023).

[15] C. Lucibello and M. Mézard, Exponential capacity of dense associative memories, Physical Review Letters 132, 077301 (2024).

[16] F. Koulischer, C. Goemaere, T. van der Meersch, J. Deleu, and T. Demeester, Exploring the temperature-dependent phase transition in modern Hopfield networks, arXiv preprint arXiv:2311.18434 (2023).

[17] J. Y.-C. Hu, D. Wu, and H. Liu, Provably optimal memory capacity for modern Hopfield models: Transformer-compatible dense associative memories as spherical codes, in Advances in Neural Information Processing Systems, Vol. 37 (2024).

[18] T. Petrova, E. Polyachenko, and R. State, Geometric entropy and retrieval phase transitions in continuous thermal dense associative memory, in International Conference on Machine Learning (PMLR, 2026).

[19] D. Krotov and J. J. Hopfield, Large associative memory problem in neurobiology and machine learning, in International Conference on Learning Representations (2021).

[20] D. Bollé, T. M. Nieuwenhuizen, I. Pérez Castillo, and T. Verbeiren, A spherical Hopfield model, Journal of Physics A: Mathematical and General 36, 10269 (2003).

[21] B. Derrida, Random-energy model: An exactly solvable model of disordered systems, Physical Review B 24, 2613 (1981).

[22] R. Monasson, Structural glass transition and the entropy of the metastable states, Physical Review Letters 75, 2847 (1995).

[23] J. R. L. de Almeida and D. J. Thouless, Stability of the Sherrington-Kirkpatrick solution of a spin glass model, Journal of Physics A: Mathematical and General 11, 983 (1978).

[24] A. Crisanti, D. J. Amit, and H. Gutfreund, Saturation level of the Hopfield model for neural network, Europhysics Letters 2, 337 (1986).

[25] A. Crisanti and H.-J. Sommers, The spherical p-spin interaction spin glass model: the statics, Zeitschrift für Physik B Condensed Matter 87, 341 (1992).

[26] B. Achilli, L. Ambrogioni, C. Lucibello, M. Mezard, and E. Ventura, The capacity of modern Hopfield networks under the data manifold hypothesis, in Proceedings of ICLR 2025 workshop: New Frontiers in Associative Memories (2025).

[27] M. Mezard and A. Montanari, Information, physics, and computation (Oxford University Press, 2009).

[28] H. Touchette, The large deviation approach to statistical mechanics, Physics Reports 478, 1 (2009).

[29] S. Franz and G. Parisi, Recipes for metastable states in spin glasses, Journal de Physique I 5, 1401 (1995).

[30] O. Penrose and J. L. Lebowitz, Rigorous treatment of metastable states in the van der Waals-Maxwell theory, Journal of Statistical Physics 3, 211 (1971).

[31] A. Barra, A. Bernacchia, E. Santucci, and P. Contucci, On the equivalence of Hopfield networks and Boltzmann machines, Neural Networks 34, 1 (2012).

[32] A. Barra, G. Genovese, P. Sollich, and D. Tantari, Phase diagram of restricted Boltzmann machines and generalized Hopfield networks with arbitrary priors, Physical Review E 97, 022310 (2018).

[33] L. Albanese, F. Alemanno, A. Alessandrelli, and A. Barra, Replica symmetry breaking in dense Hebbian neural networks, Journal of Statistical Physics 189, 24 (2022).

[34] L. Albanese, A. Alessandrelli, A. Annibale, and A. Barra, Replica symmetry breaking in supervised and unsupervised Hebbian networks, Journal of Physics A: Mathematical and Theoretical 57, 165003 (2024).

[35] G. S. Hartnett, E. Parker, and E. Geist, Replica symmetry breaking in bipartite spin glasses and neural networks, Physical Review E 98, 022116 (2018).

[36] D. Ruelle, A mathematical reformulation of Derrida’s REM and GREM, Communications in Mathematical Physics 108, 225 (1987).

[37] B. Hoover, Y. Liang, B. Pham, R. Panda, H. Strobelt, D. H. Chau, M. Zaki, and D. Krotov, Energy transformer, in Advances in Neural Information Processing Systems, Vol. 36 (2023).

[38] D. Krotov, Hierarchical associative memory, arXiv preprint arXiv:2107.06446 (2021).

[39] R. Karakida, T. Ota, and M. Taki, Hierarchical associative memory, parallelized MLP-Mixer, and symmetry breaking, arXiv preprint arXiv:2406.12220 (2024).

[40] M. Shafiei Kafraj, D. Krotov, and P. E. Latham, A biologically plausible dense associative memory with exponentia capacity, arXiv preprint arXiv:2601.00984 (2026).

[41] E. Agliari, A. Alessandrelli, A. Barra, M. S. Centonze, and F. Ricci-Tersenghi, Networks of neural networks: more is diferent, Neural Networks 194, 108181 (2026).

[42] B. Hoover, D. H. Chau, H. Strobelt, and D. Krotov, A universal abstraction for hierarchical hopfield networks, in The Symbiosis of Deep Learning and Diferential Equations II (2022).

[43] R. Courant and F. John, Introduction to Calculus and Analysis: Volume II (Springer, 1989).

[44] R. Schneider, Convex Bodies: The Brunn–Minkowski Theory, 2nd ed., Encyclopedia of Mathematics and its Applications, Vol. 151 (Cambridge University Press, 2014).

[45] R. T. Rockafellar, Convex Analysis, Princeton Mathematical Series, Vol. 28 (Princeton University Press, 1970).

[46] F. Barthe, O. Guédon, S. Mendelson, and A. Naor, A probabilistic approach to the geometry of the ℓ<sup>n</sup><sub>p</sub> -ball, The Annals of Probability 33, 480 (2005).

[47] G. Schechtman and J. Zinn, On the volume of the intersection of two L<sup>n</sup><sub>p</sub> balls, Proceedings of the American Mathematical Society 110, 217 (1990).

[48] A. P. Prudnikov, Y. A. Brychkov, and O. I. Marichev, Integrals and Series, Vol. 3: More Special Functions (Gordon and Breach Science Publishers, 1990).

[49] F. W. J. Olver, D. W. Lozier, R. F. Boisvert, and C. W. Clark, eds., NIST Handbook of Mathematical Functions (Cambridge University Press, 2010).

[50] M. Mezard, G. Parisi, and M. A. Virasoro, Spin Glass Theory and Beyond: An Introduction to the Replica Method and Its Applications, Vol. 9 (World Scientific Publishing Company, 1987).

[51] H. Nishimori, Statistical Physics of Spin Glasses and Information Processing: An Introduction (Oxford University Press, 2001).

[52] R. Price, A useful theorem for nonlinear devices having Gaussian inputs, IRE Transactions on Information Theory 4, 69 (1958).

[53] K. Mimura, J. Takeuchi, Y. Sumikawa, Y. Kabashima, and A. C. C. Coolen, Dynamical properties of dense associative memory, arXiv preprint arXiv:2506.00851 (2025).

[54] Y. Sumikawa and Y. Kabashima, Testing the role of diagonal interactions in high-order Hopfield models via dynamica mean-field theory, arXiv preprint arXiv:2604.03115 (2026).

[55] T. H. Berlin and M. Kac, The spherical model of a ferromagnet, Physical Review 86, 821 (1952).

[56] J. M. Kosterlitz, D. J. Thouless, and R. C. Jones, Spherical model of a spin-glass, Physical Review Letters 36, 1217 (1976).

[57] A. Dembo and O. Zeitouni, Large Deviations Techniques and Applications, 2nd ed., Stochastic Modelling and Applied Probability, Vol. 38 (Springer, 1998).

[58] A. Bovier, Statistical Mechanics of Disordered Systems: A Mathematical Perspective, Cambridge Series in Statistical and Probabilistic Mathematics, Vol. 18 (Cambridge University Press, 2006).

[59] R. P. Boas, Entire Functions, Pure and Applied Mathematics, Vol. 5 (Academic Press, 1954).

[60] R. B. Paris and D. Kaminski, Asymptotics and Mellin-Barnes Integrals, Encyclopedia of Mathematics and its Applications, Vol. 85 (Cambridge University Press, 2001).