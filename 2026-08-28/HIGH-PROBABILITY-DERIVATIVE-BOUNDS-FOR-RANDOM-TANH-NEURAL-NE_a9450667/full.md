# HIGH PROBABILITY DERIVATIVE BOUNDS FOR RANDOM TANH NEURAL NETWORKS ON A HYPERCUBE

JOSEF DICK, MICHAEL FEISCHL, AND FABIAN ZEHETGRUBER

Abstract. We establish high-probability bounds for mixed input derivatives of wide random neural networks whose activation derivatives satisfy a factorial growth bound. Our main result specializes these estimates to tanh networks with Xavier initialization. A direct deterministic analysis based on Euclidean operator norms of the weight matrices yields derivative bounds that generally grow exponentially with the depth. We show that this growth can be substantially improved for suficiently wide Gaussian networks by isolating the term that is linear in the highest-order derivative and controlling the corresponding tangent directions by measurable finite nets.

For scalar-output tanh networks with Gaussian weights and Xavier initialization, we prove that there exist constants $C , C _ { 0 } , C _ { 1 } > 0$ such that, whenever the common hidden width satisfies $n \geq C \left( L ^ { 3 } n _ { 0 } ^ { 2 } ( 1 + \log n _ { 0 } ) + L ^ { 2 } \left( 1 + \log ( L / \eta ) \right) \right)$ , then, with probability at least $1 - \eta ,$ , the estimate $\begin{array} { r } { | D ^ { u } \mathcal { R } _ { \Phi ^ { ( L ) } } ( x ) | \le C _ { 0 } | u | ! ( C _ { 1 } L ) ^ { | u | - 1 } \prod _ { j \in u } \beta _ { j } ( \eta , n _ { 0 } ) } \end{array}$ holds simultaneously for every non-empty $u \subseteq [ n _ { 0 } ]$ and every $x \in [ 0 , 1 ] ^ { n _ { 0 } }$ . Thus, the first-order derivative bound is independent of the depth, while a square-free mixed derivative of order $| u |$ grows at most polynomially as $L ^ { | u | - 1 }$ , apart from the coordinate factors. As consequences, we obtain high-probability bounds for the Euclidean Lipschitz constant and for weighted Sobolev norms of the network realization. The latter connect the derivative estimates to quasi-Monte Carlo integration and indicate how such regularity can enter the analysis of QMC-based training.

## 1. Introduction

Neural networks are empirically known to be vulnerable to adversarial perturbations: small, deliberately chosen changes of the input can lead to large changes in the output [11, 31]. A natural worst-case measure of this sensitivity is the Euclidean Lipschitz constant. For a scalar-valued map $f : D \subseteq \mathbb { R } ^ { d } \to \mathbb { R }$ , we write

$$
\operatorname { L i p } _ { 2 } ( f ; D ) : = \operatorname* { s u p } _ { \stackrel { x , y \in D } { x \neq y } } { \frac { | f ( x ) - f ( y ) | } { \| x - y \| _ { 2 } } } .
$$

If D is convex and $f$ is continuously diferentiable, then $\begin{array} { r } { \mathrm { L i p } _ { 2 } ( f ; D ) \leq \operatorname* { s u p } _ { x \in D } \| \nabla f ( x ) \| _ { 2 } } \end{array}$ . Thus, for neural networks with a smooth activation function such as tanh, robustness can be studied through bounds on input derivatives. In the worst case the Lipschitz constant behaves like the product of layerwise operator norms. For random neural networks we can hope for substantially smaller derivative bounds with high probability.

Derivative control is useful well beyond adversarial robustness. In scientific computing one often seeks a cheap surrogate for a high-dimensional data-to-observable map. Given $y \in Y \subseteq \mathbb { R } ^ { s }$ one would like to compute $G ( y ) \in \mathbb { R } ^ { N _ { \mathrm { o b s } } }$ , where evaluating $G ( y )$ requires solving a parameterdependent partial diferential equation. Such many-query problems arise, for example, in uncertainty quantification, inverse problems, optimization, and design. When each PDE solve is expensive and the number s of uncertain parameters is large, constructing an accurate surrogate can itself become a high-dimensional problem. Neural-network surrogates trained at low-discrepancy or quasi-Monte

Carlo (QMC) points provide one approach to this problem [20, 24, 26]. Their analysis naturally leads to mixed input derivatives: QMC error bounds and weighted Sobolev norms are governed not only by first derivatives, but by families of mixed derivatives and by their dependence on the active coordinate set. Consequently, high-probability mixed-derivative estimates for randomly initialized smooth networks can simultaneously inform robustness questions and the regularity theory underlying QMC-based approximation and training.

1.1. Related work. Derivative bounds for neural networks arise in several distinct literatures. On the deterministic side, the connection between adversarial robustness and Lipschitz regularity has motivated methods for estimating or certifying Lipschitz constants of neural networks [8, 18, 33], as well as direct certification of higher-order quantities such as Hessians [30]. For smooth approximation, simultaneous approximation of functions and their derivatives by feedforward networks goes back at least to [17], while quantitative approximation in high-order Sobolev norms by tanh networks was studied in [4]. In scientific computing and QMC-based learning, mixed derivatives play a central role in weighted Sobolev estimates and in the analysis of generalization from low-discrepancy point sets [19, 20, 24, 26]. In particular, Keller, Kuo, Nuyens, and Sloan derive explicit mixed-derivative bounds for smooth deep networks under deterministic restrictions on the network parameters [20]. Our deterministic estimates are closest to this last line of work, but are formulated in terms of Euclidean operator norms in preparation for the probabilistic analysis.

A second line of work concerns derivatives of randomly initialized neural networks, with much of the literature focusing on first-order quantities. The singular values and stability of the input-output Jacobian at random initialization have been studied through dynamical-isometry and random-matrix methods [14, 15, 28, 29]. These works reveal in particular how depth, width, initialization, and the activation function afect propagation of gradients through a random network. Closest to the Lipschitz consequence of our results, high-probability upper and lower bounds for Lipschitz constants of random ReLU networks were established in [9], and near-optimal estimates for ℓ<sup>p</sup>- Lipschitz constants of deep random ReLU networks were subsequently obtained in [5].

A third neighboring theory concerns the large- and infinite-width behavior of randomly initialized neural networks. At fixed depth, wide fully connected networks with random weights converge to Gaussian processes [13, 23, 25], and functional versions of this convergence describe the network as a random function, rather than only through its values at finitely many inputs [3]. Quantitative finite-width Gaussian approximations were subsequently obtained in [1, 2]. Particularly close to the present work are the quantitative central limit theorems of Favaro, Hanin, Marinucci, Nourdin, and Peccati [7]. They study deep fully connected networks with Gaussian weights at large but finite width and prove quantitative approximation of the network, together with its input derivatives, by the corresponding infinite-width Gaussian process. Their results apply at any fixed network depth and include functional approximations of the entire random field, rather than only pointwise statements. In particular, for a fixed derivative order and suficiently smooth activation functions, their results provide simultaneous control of the finite-width network and its derivatives through their distance to a smooth limiting Gaussian field. Thus, neither the consideration of higher input derivatives of random smooth networks nor uniformity over a continuum of input points is by itself specific to the present work. The nature of the estimate obtained here is, however, diferent. The results of [7] are Gaussian-approximation theorems: schematically, for fixed depth L and fixed derivative order k, they show that $\mathcal { R } _ { \Phi _ { r } }$ and its derivatives up to order k are close, as random fields, to the corresponding derivatives of an infinite-width Gaussian process, with an error that tends to zero as the width n tends to infinity. Our objective is instead to bound the derivatives of the finite-width network directly and nonasymptotically.

Related finite-width information is obtained from a diferent perspective by Hanin [12], who develops a perturbative 1/n-expansion for joint cumulants of the outputs and derivatives of Gaussian fully connected networks. A notable conclusion of that analysis is that the ratio of depth to width acts as an efective expansion parameter, so that terms nominally of order $n ^ { - r }$ can acquire powers of the depth. This provides complementary evidence that depth and width must be analyzed jointly when studying derivatives of random deep networks. Whereas [12] describes the distributional structure of the network and its derivatives through cumulant expansions, our result provides a direct samplewise high-probability regularity estimate.

At first order, Kim and Yang [21] establish a joint infinite-width Gaussian-process limit for a multilayer perceptron and its input Jacobian, and use this limit to study Jacobian-regularized training. This complements the random-matrix and dynamical-isometry literature on input-output Jacobians [14, 15, 28, 29] and the high-probability Lipschitz estimates for random ReLU networks [5, 9].

To the best of our knowledge, the preceding results do not provide the particular nonasymptotic regularity estimate proved here. More precisely, we are not aware of a finite-width result that gives a single high-probability event on which every non-empty square-free mixed input derivative is bounded simultaneously and uniformly over the input domain, with an explicit product-and-order-dependent bound and with explicit dependence on the derivative order, depth, width, input dimension, and failure probability. This explicit depth dependence is particularly important: a direct deterministic argument based on products of layerwise operator norms generally leads to exponential growth in $L ,$ whereas our probabilistic argument yields the polynomial factor $L ^ { | u | - 1 }$

1.2. Setting and main result. Throughout the paper we consider fully connected neural networks

$$
\varphi ^ { ( \ell ) } = ( W ^ { ( 0 ) } , \ldots , W ^ { ( \ell ) } ) , \qquad \ell \in \mathbb { N } _ { 0 } ,
$$

where, since the biases are fixed rather than part of the parameter variable, the parameter space is

$$
\Theta _ { \ell } : = \prod _ { r = 0 } ^ { \ell } \mathbb { R } ^ { n _ { r + 1 } \times n _ { r } } .
$$

The network has input dimension $n _ { 0 }$ and layer widths $n _ { 1 } , n _ { 2 } , \ldots$ For $x \in \mathbb { R } ^ { n _ { 0 } }$ , define

$$
\mathcal { R } _ { \varphi ^ { ( 0 ) } } ( x ) : = b ^ { ( 1 ) } + W ^ { ( 0 ) } x ,\tag{1}
$$

and, recursively for $\ell \in \mathbb { N }$

$$
\mathcal { R } _ { \varphi _ { i } ^ { ( \ell ) } } ( x ) = b _ { i } ^ { ( \ell + 1 ) } + \sum _ { j = 1 } ^ { n _ { \ell } } W _ { i , j } ^ { ( \ell ) } \sigma \biggl ( \mathcal { R } _ { \varphi _ { j } ^ { ( \ell - 1 ) } } ( x ) \biggr ) , \qquad i = 1 , \ldots , n _ { \ell + 1 } .
$$

The activation $\sigma : \mathbb { R }  \mathbb { R }$ is smooth, and the main probabilistic result is stated for $\sigma = \operatorname { t a n h }$ . The biases are arbitrary but fixed; all estimates below are uniform in their values. For a scalar-output network with $L \geq 2$ hidden layers of common width $n ,$ we assume independent centered Gaussian weights with Xavier scaling [10]:

$$
W _ { i , j } ^ { ( 0 ) } \sim { \mathcal { N } } \bigg ( 0 , \frac { 2 } { n _ { 0 } + n } \bigg ) , \qquad W _ { i , j } ^ { ( \ell ) } \sim { \mathcal { N } } \bigg ( 0 , \frac { 1 } { n } \bigg ) \quad ( 1 \leq \ell \leq L - 1 ) ,
$$

and

$$
W _ { 1 , j } ^ { ( L ) } \sim \mathcal { N } \bigg ( 0 , \frac { 2 } { n + 1 } \bigg ) .
$$

For $j \in [ n _ { 0 } ]$ and $\eta \in ( 0 , 1 )$ , set

$$
\beta _ { j } ( \eta , n _ { 0 } ) : = \sqrt { \frac { 2 } { n _ { 0 } + n } } \left( n + 2 \sqrt { n \log \left( \frac { 4 n _ { 0 } } { \eta } \right) } + 2 \log \left( \frac { 4 n _ { 0 } } { \eta } \right) \right) ^ { 1 / 2 } .
$$

Our main theorem shows that there exist universal constants $C , C _ { 0 } , C _ { 1 } > 0$ such that the width condition

$$
n \geq C \left( L ^ { 3 } n _ { 0 } ^ { 2 } ( 1 + \log n _ { 0 } ) + L ^ { 2 } \left( 1 + \log \frac { L } { \eta } \right) \right)
$$

implies, with probability at least $1 - \eta ,$ that

$$
| D ^ { u } \mathcal { R } _ { \Phi ^ { ( L ) } } ( x ) | \le C _ { 0 } | u | ! ( C _ { 1 } L ) ^ { | u | - 1 } \prod _ { j \in u } \beta _ { j } ( \eta , n _ { 0 } )
$$

holds simultaneously for every non-empty $u \subseteq [ n _ { 0 } ]$ and every $x \in [ 0 , 1 ] ^ { n _ { 0 } }$ . In particular, the firstderivative bound is independent of the depth. After increasing the universal constant in the width condition if necessary, $\beta _ { j } ( \eta , n _ { 0 } ) \leq 3$ , and therefore the same event gives the Euclidean robustness estimate $\mathrm { L i p } _ { 2 } ( \mathcal { R } _ { \Phi ^ { ( L ) } } ; [ 0 , 1 ] ^ { n _ { 0 } } ) \le 3 C _ { 0 } \sqrt { n _ { 0 } }$ , again with no growth in $L$

1.3. Contributions. The proof has four main components.

(1) We first derive deterministic mixed-derivative estimates for smooth neural networks in terms of Euclidean operator norms of the weight matrices. This is the 2-norm analogue of the regularity argument in [20]. As expected, direct multiplication of layerwise matrix norms generally produces an exponential dependence on the depth.

(2) We then isolate, in the Fa\`a di Bruno expansion, the partition consisting of the full derivative set. This contribution is linear in the highest-order derivative and is governed by a tangent vector. Under a suitable tangent condition, a majorant-series argument allows us to replace the exponential depth factor by a polynomial dependence on $L .$

(3) For Gaussian random networks we prove that the required matrix and tangent conditions hold with high probability. The key step is to construct measurable finite nets for the normalized tangent directions and to exploit the independence of the current weight matrix from the preceding layers. This replaces a full operator-norm factor by a layerwise factor of order $1 + L ^ { - 1 }$ along the relevant tangent directions and leads to the width condition stated above.

(4) Finally, we record consequences of the simultaneous derivative event for weighted Sobolev norms, QMC quadrature and lattice-based training, and Euclidean Lipschitz continuity. The QMC interpretation also identifies a limitation of isotropic Xavier initialization: the coordinate factors $\beta _ { j }$ do not decay with $j ,$ so dimension-independent weighted QMC estimates require additional anisotropy or regularization.

The paper is organized as follows. We first establish deterministic derivative estimates and the depth-stable majorant bound under the tangent condition. We then prove the high-probability matrix and tangent estimates for wide Gaussian networks and specialize them to scalar-output Xavier networks. The subsequent section derives the QMC and Lipschitz consequences. Auxiliary analytic, combinatorial, probabilistic, and measurability results are collected in the appendix.

1.4. Notation. We set $\mathbb { N } : = \{ 1 , 2 , \ldots \}$ and $ { \mathbb { N } } _ { 0 } : =  { \mathbb { N } } \cup \lbrace 0 \rbrace$ . For $s \in \mathbb { N }$ , set $[ s ] : = \{ 1 , \ldots , s \}$ . By $e _ { j }$ we denote the jth canonical basis vector, with the ambient dimension clear from the context. The Euclidean norm of a vector is denoted by $\| \cdot \| _ { 2 }$ , and the induced Euclidean operator norm of a matrix by $\| \cdot \| _ { 2 \to 2 }$ . We write $S ^ { m - 1 } : = \{ x \in \mathbb { R } ^ { m } : \| x \| _ { 2 } = 1 \}$

For $a , b \in \mathbb { R } ^ { n }$ , the componentwise, or Hadamard, product is denoted by $a \odot b$ . More generally, for $a ^ { ( i ) } \in \mathbb { R } ^ { n } , i = 1 , \dots , r$ , we write $\textstyle \bigcirc _ { i = 1 } ^ { r } a ^ { ( i ) }$ . For $\boldsymbol { \nu } \in \mathbb { N } _ { 0 } ^ { s }$ , define $\textstyle | \nu | : = \sum _ { i = 1 } ^ { s } \nu _ { i }$ and $\begin{array} { r } { \partial ^ { \nu } : = \Pi _ { j = 1 } ^ { s } \partial _ { j } ^ { \nu _ { j } } } \end{array}$ If $u \subseteq [ s ]$ , we write $\begin{array} { r } { D ^ { u } : = \prod _ { j \in u } \partial _ { j } ; } \end{array}$ in particular, the derivatives $D ^ { u }$ considered in the main probabilistic result are square-free mixed derivatives.

We use the conventions $\begin{array} { r } { \prod _ { i = 1 } ^ { 0 } a _ { i } = \prod _ { v \in \emptyset } a ( v ) = 1 } \end{array}$ and $\begin{array} { r } { \sum _ { i = 1 } ^ { 0 } a _ { i } = \sum _ { v \in \emptyset } a ( v ) = 0 } \end{array}$ , as well as $0 ^ { 0 } = 1$ and $\sigma ^ { ( 0 ) } : = \sigma$ . For a non-empty finite set $u ,$ let $\Pi ( u )$ be the set of all partitions of $u ,$ and set $\Pi ( \emptyset ) : = \{ \emptyset \}$ . For $r \in \{ 1 , \ldots , | u | \}$ , let $\Pi _ { r } ( u )$ be the set of partitions with exactly r blocks. For a polynomial $p$ and $k \in  { \mathbb { N } } _ { 0 }$ , coe $\operatorname { f } _ { k } ( p )$ denotes the coeficient of $z ^ { k }$ . All finite-dimensional matrix and parameter spaces are equipped with their usual Euclidean topology and the corresponding Borel σ-algebra. Products of parameter spaces are equipped with the product topology.

## 2. Derivative bounds of neural networks

Derivative bounds for smooth neural networks were obtained in [20] using the ∞-matrix norm of the weight matrices. We follow the same basic argument, but formulate it in terms of the Euclidean operator norm. This is convenient for Gaussian random matrices and will be used in the probabilistic part of the paper. If $n _ { 0 } , L \in \mathbb { N }$ and $\beta _ { j } , \kappa _ { \ell } > 0$ for every $j \in [ n _ { 0 } ]$ and $\ell \in [ L ]$ , then we define $\Theta _ { L } ^ { \mathrm { m a t } } ( n _ { 0 } , \beta , \kappa )$ as the set of all $\varphi \in \Theta _ { L }$ such that $\| W ^ { ( 0 ) } e _ { j } \| _ { 2 } \le \beta _ { j }$ holds for every $j \in [ n _ { 0 } ]$ and $\| W ^ { ( \ell ) } \| _ { 2 \to 2 } \le \kappa _ { \ell }$ holds for every $\ell \in [ L ]$

We denote by $\kappa = ( \kappa _ { \ell } ) _ { \ell \in [ L ] }$ the tuple of all $\kappa _ { \ell }$ . We introduce notation that allows us to write down the Fa\`a di Bruno formula in a way that is convenient for us. We define the function $\begin{array} { r } { \mathrm { t } : \mathbb { N } _ { 0 } ^ { n _ { 0 } }  \bigcup _ { r \in \mathbb { N } _ { 0 } } \mathbb { N } ^ { r } } \end{array}$ such that for all $\nu \in \mathbb { N } _ { 0 } ^ { n _ { 0 } }$ we define $\mathrm { t } ( \nu ) = : a \in [ n _ { 0 } ] ^ { | \nu | }$ as the unique vector that is nondecreasing and such that $\begin{array} { r } { \partial ^ { \nu } = \prod _ { k = 1 } ^ { | \nu | } \partial _ { a _ { k } } } \end{array}$ . For any $v \subseteq [ | \nu | ]$ we define

$$
( \mu ( \nu , v ) ) _ { j } = | \{ k \in v \mid ( \mathfrak { t } ( \nu ) ) _ { k } = j \} | = \left| ( \mathfrak { t } ( \nu ) ) ^ { - 1 } \left( j \right) \cap v \right| \quad \mathrm { f o r ~ a l l ~ } j = 1 , \dots , n _ { 0 }\tag{2}
$$

and we note that $\textstyle | \mu ( \nu , v ) | = \sum _ { j = 1 } ^ { n _ { 0 } } ( \mu ( \nu , v ) ) _ { j } = | v |$ and $\begin{array} { r } { \sum _ { v \in \pi } \mu ( \nu , v ) = \nu } \end{array}$ . This notation will be useful later.

Throughout this paper we always assume that for every $r \in \mathbb N$ there exists $A _ { r } > 0$ such that $| \sigma ^ { ( r ) } ( x ) | \le A _ { \ i }$ holds for all $x \in \mathbb { R }$ . We define $B _ { 1 } ^ { ( 0 ) } ( \kappa ) : = 1$ and $B _ { q } ^ { ( 0 ) } ( \kappa ) : = 0$ for all $q > 1$ . For all $\dot { \ell } \in [ L ]$ and all $q \in \mathbb { N }$ we define

$$
B _ { q } ^ { ( \ell ) } ( \kappa ) : = \kappa _ { \ell } \sum _ { \pi \in \Pi ( [ q ] ) } A _ { | \pi | } \prod _ { v \in \pi } B _ { | v | } ^ { ( \ell - 1 ) } ( \kappa ) .
$$

For all $\ell \in [ L ]$ [L], all $r \in  { \mathbb { N } } _ { 0 }$ , all $\varphi ^ { ( \ell - 1 ) } \in \Theta _ { \ell - 1 }$ and all $x \in \mathbb { R } ^ { n _ { 0 } }$ we define the vector $h _ { r } ^ { ( \ell ) } ( x , \varphi ^ { ( \ell - 1 ) } ) \in \mathbb { R } ^ { n _ { \ell } }$ by

$$
h _ { r , i } ^ { ( \ell ) } ( x , \varphi ^ { ( \ell - 1 ) } ) : = \sigma ^ { ( r ) } \left( \operatorname { \mathcal { R } } _ { \varphi _ { i } ^ { ( \ell - 1 ) } } ( x ) \right) \quad \mathrm { f o r ~ a l l } \quad i \in \{ 1 , \ldots , n _ { \ell } \} .
$$

Theorem 1. If for every $r \in \mathbb N$ there exists $A _ { r } > 0$ such that $| \sigma ^ { ( r ) } ( x ) | \le A$ <sub>r</sub> holds for all $x \in \mathbb { R }$ then

$$
\left\| \partial ^ { \nu } \mathcal { R } _ { \varphi ^ { ( \ell ) } } ( x ) \right\| _ { 2 } \leq B _ { | \nu | } ^ { ( \ell ) } ( \kappa ) \prod _ { j = 1 } ^ { n _ { 0 } } \beta _ { j } ^ { \nu _ { j } } ,
$$

holds for all $n _ { 0 } \in \mathbb { N }$ , all $\ell \in \{ 0 , \ldots , L \}$ , all $\varphi ^ { ( \ell ) } \in \Theta _ { \ell } ^ { \mathrm { m a t } } ( n _ { 0 } , \beta , \kappa )$ , all $\nu \in \mathbb { N } _ { 0 } ^ { n _ { 0 } } \setminus \{ 0 \}$ , all $x \in \mathbb { R } ^ { n _ { 0 } }$

Proof. We prove the result by induction over the layer index ℓ. Suppose $\ell = 0$ and suppose $\nu \in \mathbb { N } _ { 0 } ^ { n _ { 0 } }$ is such that $\nu _ { k } = 1$ for one $k \in [ n _ { 0 } ]$ and $\nu _ { m } = 0$ for all $m \in [ n _ { 0 } ] \setminus \{ k \}$ . Then

$$
\Big \| \partial ^ { \nu } \mathcal { R } _ { \varphi ^ { ( 0 ) } } ( x ) \Big \| _ { 2 } = \left( \sum _ { i = 1 } ^ { n _ { 1 } } \left| W _ { i , k } ^ { ( 0 ) } \right| ^ { 2 } \right) ^ { 1 / 2 } = \Big \| W ^ { ( 0 ) } e _ { k } \Big \| _ { 2 } \leq \beta _ { k } = B _ { | \nu | } ^ { ( 0 ) } ( \kappa ) \prod _ { j = 1 } ^ { n _ { 0 } } ( \beta _ { j } ) ^ { \nu _ { j } }
$$

$\mathrm { I f } \ | \nu | > 1$ then the derivative is zero and the inequality is trivially true. For the induction step, assume that $\ell \in \mathbb { N }$ and that the induction hypothesis holds for $\ell - 1$ . We have

$$
\begin{array} { r l } & {  \partial ^ { \nu } \mathcal { R } _ { \varphi ^ { ( \varepsilon ) } } ( x )  _ { 2 } =  W ^ { ( \ell ) } \partial ^ { \nu } h _ { 0 } ^ { ( \ell ) } ( x , \varphi ^ { ( \ell - 1 ) } )  _ { 2 } \leq  W ^ { ( \ell ) }  _ { 2  2 }  \partial ^ { \nu } h _ { 0 } ^ { ( \ell ) } ( x , \varphi ^ { ( \ell - 1 ) } )  _ { 2 } } \\ & { \qquad \leq \kappa _ { \ell }  \partial ^ { \nu } h _ { 0 } ^ { ( \ell ) } ( x , \varphi ^ { ( \ell - 1 ) } )  _ { 2 } } \end{array}
$$

The Fa\`a di Bruno formula in Theorem 18 together with the triangle inequality and Lemma 19 gives

$$
\begin{array} { r l } & { \left. \phi ^ { p } h _ { 0 } ^ { ( \ell ) } \left( x , \varphi ^ { ( \ell - 1 ) } \right) \right. _ { 2 } = \Bigg \Vert \displaystyle \sum _ { x \in \Pi ( \{ | x | \} ) } h _ { | x | } ^ { ( \ell ) } \left( x , \varphi ^ { ( \ell - 1 ) } \right) \odot \left( \underset { \nu \in \pi } { \odot } \left( \partial ^ { \mu ( \nu , s ) } \mathcal { R } _ { \varphi ^ { ( \ell - 1 ) } } ( x ) \right) \right) \Bigg \Vert _ { 2 } } \\ & { \qquad \leq \displaystyle \sum _ { x \in \Pi ( \{ | x | \} ) } \left. h _ { | x | } ^ { ( \ell ) } \left( x , \varphi ^ { ( \ell - 1 ) } \right) \odot \left( \underset { \nu \in \pi } { \odot } \left( \partial ^ { \mu ( \nu , s ) } \mathcal { R } _ { \varphi ^ { ( \ell - 1 ) } } ( x ) \right) \right) \right. _ { 2 } } \\ & { \qquad \leq \displaystyle \sum _ { x \in \Pi ( \{ | x | \} ) } \displaystyle \sum _ { i = 1 } ^ { n _ { \ell } } \left( \underset { | x | } { \sum } \left( \sigma ^ { ( | \pi | ) } \left( \mathcal { R } _ { \varphi ^ { ( \ell - 1 ) } } ( x ) \right) \right) ^ { 2 } \underset { \nu \in \pi } { \prod } \left( \partial ^ { \mu ( \nu , s ) } \mathcal { R } _ { \varphi ^ { ( \ell - 1 ) } } ( x ) \right) ^ { 2 } \right) ^ { 1 / 2 } } \\ &  \qquad \leq \displaystyle \sum _ { x \in \Pi ( \{ | x | \} ) } A _ { | x | } \left. \underset { s \in \pi } { \sum } \left( \partial ^ { \mu ( \nu , s ) } \mathcal { R } _ { \varphi ^ { ( \ell - 1 ) } } ( x ) \right) \right. _ { 2 } \leq \displaystyle \sum _ { x \in \Pi ( \{ | x | \} ) } A _ { | x | } \prod _ { \nu \in \pi } \left. \partial ^ { \mu ( \nu , s ) } \mathcal { R } _ { \varphi ^ { ( \ell - 1 ) } } ( x ) \right. _ { \Sigma } \end{array}
$$

Combining everything and inserting the induction hypothesis we obtain

$$
\begin{array} { r l } & { \left\| \partial ^ { \nu } \mathcal { R } _ { \varphi ^ { ( t ) } } ( x ) \right\| _ { 2 } \leq \kappa _ { \ell } \left\| \partial ^ { \nu } h _ { 0 } ^ { ( \ell ) } \left( x , \varphi ^ { ( \ell - 1 ) } \right) \right\| _ { 2 } \leq \kappa _ { \ell } \displaystyle \sum _ { \pi \in \operatorname { I I } ( [ | \nu | ] ) } A _ { | \pi } \prod _ { v \in \pi } \left\| \partial ^ { \mu ( \nu , v ) } \mathcal { R } _ { \varphi ^ { ( \ell - 1 ) } } ( x ) \right\| _ { 2 } } \\ & { \qquad \leq \kappa _ { \ell } \displaystyle \sum _ { \pi \in \operatorname { I I } ( [ | \nu | ] ) } A _ { | \pi } \prod _ { v \in \pi } \left( B _ { | \mu ( \nu , v ) | } ^ { ( \ell - 1 ) } ( \kappa ) \displaystyle \prod _ { j = 1 } ^ { n _ { 0 } } \beta _ { j } ^ { ( \mu ( \nu , v ) ) _ { j } } \right) = \left( \kappa _ { \ell } \sum _ { \pi \in \Pi ( [ | \nu | ] ) } A _ { | \pi } \prod _ { v \in \pi } B _ { | v | } ^ { ( \ell - 1 ) } ( \kappa ) \right) \displaystyle \prod _ { j = 1 } ^ { n _ { 0 } } \beta _ { j } ^ { \nu _ { j } } , } \end{array}
$$

where we used $| \mu ( \nu , v ) | = | v |$ for all $\pi \in \Pi ( [ | \nu | ] )$ and all $v \in \pi$ as well as $\textstyle \sum _ { v \in \pi } \mu ( \nu , v ) = \nu$ □

If we impose a more concrete bound on the derivatives of the activation function that is inspired by the bound from Lemma 20, then we obtain the following result.

Corollary 2. If there exists a constant $C _ { \sigma }$ such that $\left| \sigma ^ { ( r ) } ( x ) \right| \le A _ { r } : = r ! C _ { \sigma } ^ { r - 1 }$ holds for all $r \in \mathbb N$ and all $x \in \mathbb { R }$ , then

$$
\left. D ^ { u } \mathcal { R } _ { \varphi ^ { ( \ell ) } } ( x ) \right. _ { 2 } \leq \left( \prod _ { m = 1 } ^ { \ell } \kappa _ { m } \right) ( | u | ! ) \left( C _ { \sigma } \sum _ { p = 0 } ^ { \ell - 1 } \prod _ { t = 1 } ^ { p } \kappa _ { t } \right) ^ { | u | - 1 } \prod _ { j \in u } \beta _ { j } ,
$$

holds for all $n _ { 0 } , L \in \mathbb { N }$ , all $\ell \in \{ 0 , \ldots , L \}$ , all $\varphi ^ { ( \ell ) } \in \Theta _ { \ell } ^ { \mathrm { m a t } } ( n _ { 0 } , \beta , \kappa )$ , all non-empty $u \subseteq [ n _ { 0 } ]$ and all $x \in \mathbb { R } ^ { n _ { 0 } }$

Proof. We prove the statement using induction over $\ell \in \mathbb { N } _ { 0 }$ . For $\ell = 0$ and $q = 1$ we have $B _ { 1 } ^ { ( 0 ) } ( \kappa ) = 1 \leq 1 = 1 ( 1 ! ) 0 ^ { 0 }$ . For $\ell = 0$ and $q > 1$ we have $B _ { q } ^ { ( 0 ) } ( \kappa ) = 0 \leq 0 = 1 ( q ! ) 0 ^ { q - 1 }$

For the induction step let us suppose $\ell \in \mathbb { N }$ and the induction hypothesis is satisfied for $\ell - 1$ and every $q \in \mathbb { N }$ . By Lemma 21 and the binomial formula we have

$$
\begin{array} { r l } { \bar { H } _ { \mathrm { e f f } } ^ { ( 0 ) } ( i \omega ) = } & { \omega \displaystyle \sum _ { s = 1 } ^ { N } \lambda _ { 1 } \Big [ \underbrace { \prod _ { \sigma } ^ { \sigma + 1 } \omega _ { s } } _ { \mathrm { C _ { 1 } \wedge \sigma } } - \underbrace { \sum _ { s = 1 } ^ { \infty } \lambda _ { 1 } } _ { \mathrm { C _ { 1 } \wedge \sigma } } \underbrace { \prod _ { \sigma } ^ { \sigma + 1 } \omega _ { s } } _ { \mathrm { D _ { \sigma } \wedge \sigma , \sigma } } } \\ & { \le \omega \displaystyle \sum _ { s = 1 } ^ { N } \gamma _ { \sigma } \epsilon _ { s } ^ { - \sigma } \textbf { \sum } _ { \sigma \wedge \sigma } \Bigg ] \Bigg [ \Bigg ( \underset { s = 1 } { \overset { N } { \prod } } \Bigg ) \mathrm { i d } y \Bigg ( \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon } \\ & { = \Bigg ( \underset { s = 1 } { \overset { N } { \prod } } \epsilon _ { s = 1 } \Bigg ) \underset { s = 1 } { \overset { N } { \prod } } \Bigg ) \mathrm { i d } y } \\ & { = \Bigg ( \underset { s = 1 } { \overset { N } { \prod } } \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon ^ { - 1 } \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon ^ { - 1 } \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon ^ { - 1 } \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon } \\ & { - \Bigg ( \underset { s = 1 } { \overset { N } { \prod } } \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon ^ { - 1 } \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon ^ { - 1 } \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon \epsilon } \\ &  = \ \end{array}
$$

An application of Theorem 1 concludes the proof.

If $\kappa _ { 1 } = \cdot \cdot \cdot = \kappa _ { L } > 1$ , then the bound in Corollary 2 contains the term $\begin{array} { r } { \prod _ { \ell = 1 } ^ { L } \kappa _ { \ell } = \kappa _ { 1 } ^ { L } } \end{array}$ , which grows exponentially in the depth L of the neural network. As a remedy we give an additional condition that allows us to get rid of this exponential growth in the depth L of the neural network. For every non-empty $u \subseteq [ n _ { 0 } ]$ and every $\ell \in [ L ]$ we define $G _ { u } ^ { ( \ell ) } : \mathbb { R } ^ { n _ { 0 } } \times \Theta _ { \ell - 1 } \to \mathbb { R } ^ { n _ { \ell } }$ by

$$
G _ { u } ^ { ( \ell ) } \left( x , \varphi ^ { ( \ell - 1 ) } \right) : = h _ { 1 } ^ { ( \ell ) } \left( x , \varphi ^ { ( \ell - 1 ) } \right) \odot D ^ { u } \mathcal { R } _ { \varphi ^ { ( \ell - 1 ) } } ( x ) .
$$

Given $\alpha = ( \alpha _ { \ell } ) _ { \ell \in [ L ] } \in [ 0 , \infty ) ^ { L }$ and $\gamma = ( \gamma _ { \ell } ) _ { \ell \in [ L ] } \in [ 0 , \infty ) ^ { L }$ we define $\Theta _ { L } ^ { \mathrm { t a n g } } ( n _ { 0 } , \beta , \alpha , \gamma )$ as the set of all $\varphi \in \Theta _ { L }$ such that for all non-empty $u \subseteq [ n _ { 0 } ]$ , all $\ell \in [ L ]$ and all $x \in [ 0 , 1 ] ^ { n _ { 0 } }$ the inequality

$$
\left\| W ^ { ( \ell ) } G _ { u } ^ { ( \ell ) } \left( x , \varphi ^ { ( \ell - 1 ) } \right) \right\| _ { 2 } \leq \alpha \ell \left\| G _ { u } ^ { ( \ell ) } \left( x , \varphi ^ { ( \ell - 1 ) } \right) \right\| _ { 2 } + \gamma \ell \prod _ { j \in u } \beta _ { j }\tag{3}
$$

holds. By Lemma 22 the set $\Theta _ { L } ^ { \mathrm { t a n g } } ( n _ { 0 } , \beta , \alpha , \gamma )$ is closed in $\Theta _ { L }$ , and in particular measurable. We define $\Xi _ { 1 } ^ { ( 0 ) } : = 1$ and $\Xi _ { k } ^ { ( 0 ) } : = 0$ for all $k \geq 2$ . Recursively, we define

$$
\Xi _ { k } ^ { ( \ell ) } : = \alpha _ { \ell } A _ { 1 } \Xi _ { k } ^ { ( \ell - 1 ) } + \gamma _ { \ell } + \kappa _ { \ell } \sum _ { \pi \in \Pi ( [ k ] ) , \pi \neq \{ [ k ] \} } A _ { | \pi | } \prod _ { v \in \pi } \Xi _ { | v | } ^ { ( \ell - 1 ) }
$$

for all $k , \ell \in \mathbb { N }$

Remark 3. The assumption $| \sigma ^ { ( r ) } | \le r ! C _ { \sigma } ^ { r - 1 }$ includes $\| \sigma ^ { \prime } \| _ { L ^ { \infty } ( \mathbb { R } ) } \leq 1$ . This normalization is relevant for the depth dependence: the linear tangent contribution propagates with the factors $\alpha \ell \| \sigma ^ { \prime } \| _ { \infty }$ More generally, the same argument applies if the assumed bound on contiguous products is imposed on these combined factors.

Lemma 4. If for every $r \in \mathbb { N }$ there exists $A _ { r } > 0$ such that $| \sigma ^ { ( r ) } ( x ) | \le A$ <sub>r</sub> holds for all $x \in \mathbb { R }$ , then

$$
\left. D ^ { u } \mathcal { R } _ { \varphi ^ { ( \ell ) } } ( x ) \right. _ { 2 } \leq \Xi _ { | u | } ^ { ( \ell ) } \prod _ { j \in u } \beta _ { j }
$$

holds for all $L \in \mathbb { N }$ , for all $\varphi \in \Theta _ { L } ^ { \mathrm { m a t } } ( n _ { 0 } , \beta , \kappa ) \cap \Theta _ { L } ^ { \mathrm { t a n g } } ( n _ { 0 } , \beta , \alpha , \gamma )$ , for all $\ell \in \{ 0 , \ldots , L \}$ , all non-empty $u \subseteq [ n _ { 0 } ]$ and all $x \in [ 0 , 1 ] ^ { n _ { 0 } ^ { - } }$

Proof. The proof is by induction over $\ell \in \{ 0 , \ldots , L \}$ . For $\ell = 0$ the realization of $\varphi ^ { ( \ell ) }$ is an afine function by definition (1). If $j \in [ n _ { 0 } ]$ and $u = \{ j \}$ , then

$$
\left. D ^ { u } \mathcal { R } _ { \varphi ^ { ( 0 ) } } ( x ) \right. _ { 2 } = \left. W ^ { ( 0 ) } e _ { j } \right. _ { 2 } \leq \beta _ { j } = \Xi _ { 1 } ^ { ( 0 ) } \prod _ { i \in u } \beta _ { i }
$$

holds for all $x \in \mathbb { R } ^ { n _ { 0 } }$ . If $| u | > 1$ then $D ^ { u } \mathcal { R } _ { \varphi ^ { ( 0 ) } } ( x ) = 0$ for all $x \in \mathbb { R } ^ { n _ { 0 } }$ and therefore

$$
\left\| D ^ { u } \mathcal { R } _ { \varphi ^ { ( 0 ) } } ( x ) \right\| _ { 2 } = 0 = \Xi _ { | u | } ^ { ( 0 ) } \prod _ { j \in u } \beta _ { j } .
$$

Let $\ell \in [ L ]$ and suppose the statement is true for $\ell - 1$ . The Fa\`a di Bruno formula in Theorem 18 gives for all non-empty $u \subseteq [ n _ { 0 } ]$ that

$$
\begin{array} { r l } & { D ^ { u } \mathcal { R } _ { \varphi ^ { ( \ell ) } } ( x ) = W ^ { ( \ell ) } \displaystyle \sum _ { \pi \in \Pi ( u ) } h _ { | \pi | } ^ { ( \ell ) } ( x ) \odot \left( \displaystyle \bigodot _ { v \in \pi } D ^ { v } \mathcal { R } _ { \varphi ^ { ( \ell - 1 ) } } ( x ) \right) } \\ & { \qquad = W ^ { ( \ell ) } G _ { u } ^ { ( \ell ) } \left( x , \varphi ^ { ( \ell - 1 ) } \right) + W ^ { ( \ell ) } \displaystyle \sum _ { \pi \in \Pi ( u ) , \pi \neq \{ u \} } h _ { | \pi | } ^ { ( \ell ) } ( x ) \odot \left( \displaystyle \bigodot _ { v \in \pi } D ^ { v } \mathcal { R } _ { \varphi ^ { ( \ell - 1 ) } } ( x ) \right) . } \end{array}
$$

Since

$$
\begin{array} { l } { \displaystyle \left\| W ^ { ( \ell ) } G _ { u } ^ { ( \ell ) } \left( x , \varphi ^ { ( \ell - 1 ) } \right) \right\| _ { 2 } \leq \alpha _ { \ell } \left\| G _ { u } ^ { ( \ell ) } \left( x , \varphi ^ { ( \ell - 1 ) } \right) \right\| _ { 2 } + \gamma _ { \ell } \displaystyle \prod _ { j \in u } \beta _ { j } \leq \alpha _ { \ell } A _ { 1 } \left\| D ^ { u } \mathcal { R } _ { \varphi ^ { ( \ell - 1 ) } } ( x ) \right\| _ { 2 } + \gamma _ { \ell } \displaystyle \prod _ { j \in u } \beta _ { j } } \\ { \leq \left( \alpha _ { \ell } A _ { 1 } \Xi _ { | u | } ^ { ( \ell - 1 ) } + \gamma _ { \ell } \right) \displaystyle \prod _ { j \in u } \beta _ { j } , } \end{array}
$$

we obtain by Lemma 19 that

$$
\begin{array} { r l } & { \| D ^ { u } \mathcal { R } _ { \varphi ^ { ( \ell ) } } ( x ) \| _ { 2 } \leq \Big ( \alpha _ { \ell } A _ { 1 } \Xi _ { | u | } ^ { ( \ell - 1 ) } + \gamma _ { \ell } \Big ) \displaystyle \prod _ { j \in u } \beta _ { j } + \| W ^ { ( \ell ) } \| _ { 2  2 } \displaystyle \sum _ { \pi \in \Pi ( u ) , \pi \neq \{ u \} } A _ { | \pi | } \displaystyle \prod _ { v \in \pi } \| D ^ { v } \mathcal { R } _ { \varphi ^ { ( \ell - 1 ) } } ( x ) \| _ { 2 } } \\ & { \qquad \leq \Big ( \alpha _ { \ell } A _ { 1 } \Xi _ { | u | } ^ { ( \ell - 1 ) } + \gamma _ { \ell } \Big ) \displaystyle \prod _ { j \in u } \beta _ { j } + \kappa _ { \ell } \displaystyle \sum _ { \pi \in \Pi ( u ) , \pi \neq \{ u \} } A _ { | \pi | } \displaystyle \prod _ { v \in \pi } ( \Xi _ { | v | } ^ { ( \ell - 1 ) } \displaystyle \prod _ { j \in v } \beta _ { j } ) = \Xi _ { | u | } ^ { ( \ell ) } \displaystyle \prod _ { j \in u } \beta _ { j } . } \end{array}
$$

Proposition 5. $I f \overline { { \kappa } } , \overline { { \alpha } } \geq 1$ and there exists $C _ { \sigma } > 0$ such that $\left| \sigma ^ { ( r ) } ( x ) \right| \le A _ { r } : = r ! C _ { \sigma } ^ { r - 1 }$ holds for all $r \in \mathbb N$ and all $x \in \mathbb { R }$ , then there exist constants $C _ { 0 } , C _ { 1 } > 0$ such that the following holds. If $L \in \mathbb { N }$ and

$$
\kappa _ { \ell } \leq \overline { { \kappa } } , \qquad \prod _ { n = k } ^ { m } \alpha _ { n } \leq \overline { { \alpha } } \quad a n d \quad 0 \leq \gamma _ { \ell } \leq \overline { { \kappa } } / L
$$

holds for all $k , \ell , m \in [ L ]$ with $k \leq m$ , then

$$
\left. D ^ { u } \mathcal { R } _ { \varphi ^ { ( \ell ) } } ( x ) \right. _ { 2 } \leq C _ { 0 } \left( | u | \right) ! \left( C _ { 1 } L \right) ^ { | u | - 1 } \prod _ { j \in u } \beta _ { j } \qquad 
$$

for all $\varphi \in \Theta _ { L } ^ { \mathrm { m a t } } ( n _ { 0 } , \beta , \kappa ) \cap \Theta _ { L } ^ { \mathrm { t a n g } } ( n _ { 0 } , \beta , \alpha , \gamma )$ , all $\ell \in \{ 0 , \ldots , L \}$ , all non-empty $u \subseteq [ n _ { 0 } ]$ and all $x \in [ 0 , 1 ] ^ { n _ { 0 } }$

Proof. We define $C _ { 0 } : = \operatorname* { m a x } \left. 1 , 4 \overline { { \alpha } } ( 1 + 2 \overline { { \kappa } } ) \right.$ and

$$
c : = \operatorname* { m i n } \left\{ 1 , \frac { 1 } { 2 C _ { \sigma } C _ { 0 } } , \frac { 1 } { 4 \overline { { \alpha \kappa } } C _ { \sigma } C _ { 0 } } \right\}
$$

as well as $C _ { 1 } : = 1 / c$ . Those constants are the right choices for the following arguments. Let $L \in \mathbb { N }$ and suppose $0 \leq \gamma _ { \ell } \leq \overline { { \kappa } } / L$ holds for all $\ell \in [ L ]$ . For every $\ell \in \{ 0 , \ldots , L \}$ we define a function $F _ { \ell } : \mathbb { C } \to \mathbb { C }$ by

$$
F _ { \ell } ( z ) : = \sum _ { k = 1 } ^ { n _ { 0 } } \frac { \Xi _ { k } ^ { ( \ell ) } } { k ! } z ^ { k }
$$

By the recursive definition, $\Xi _ { k } ^ { ( \ell ) } \geq 0$ for all $k \in \mathbb N$ and $\ell \in \{ 0 , \ldots , L \}$ . Consequently, all coeficients of $F _ { \ell }$ and of its positive integer powers are non-negative. For all $k \in [ n _ { 0 } ]$ and all $r \in \mathbb { N }$ we have

$$
\mathrm { c o e f f } _ { k } \left( ( F _ { \ell } ( z ) ) ^ { r } \right) = \frac { r ! } { k ! } \sum _ { \substack { \pi \in \Pi ( [ k ] ) , | \pi | = r } } \prod _ { v \in \pi } \Xi _ { | v | } ^ { ( \ell ) } .
$$

Indeed,

$$
k ! \operatorname { c o e f f } _ { k } ( F _ { \ell } ( z ) ^ { r } ) = \sum _ { { j _ { 1 } , \ldots , j _ { r } \geq 1 } \atop { j _ { 1 } + \cdots + j _ { r } = k } } { \frac { k ! } { j _ { 1 } ! \cdot \cdot \cdot j _ { r } ! } } \prod _ { m = 1 } ^ { r } \Xi _ { j _ { m } } ^ { ( \ell ) } .
$$

The right-hand side is the sum over ordered partitions of $[ k ]$ into r non-empty blocks, weighted by the product of the corresponding $\boldsymbol { \Xi } ^ { ( \ell ) }$ -terms. Every unordered partition in $\Pi _ { r } ( [ k ] )$ has exactly r! orderings. Hence, we obtain

$$
\begin{array} { r l } & { \displaystyle \sum _ { \pi \in \Pi ( [ k ] ) , | \pi | \geq 2 } A _ { | \pi | } \prod _ { v \in \pi } | e | = k ! \sum _ { r = 2 } ^ { \infty } \frac { A _ { r } } { k ! } \sum _ { \pi \in \Pi ( [ k ] ) , | \pi | = r \in \pi } \prod _ { v \in \pi } \Xi _ { v | } ^ { ( \ell ) } \leq k ! \sum _ { r = 2 } ^ { \infty } C _ { \sigma } ^ { r - 1 } \left( \frac { r ! } { k ! } \sum _ { \pi \in \Pi ( [ k ] ) , | \pi | = r \in \pi } \prod _ { v \in \pi } \Xi _ { | v | } ^ { ( \ell ) } \right) } \\ & { \quad \quad \quad \quad = k ! \displaystyle \sum _ { r = 2 } ^ { \infty } C _ { \sigma } ^ { r - 1 } \operatorname { c o e f f } _ { k } \big ( ( F _ { \ell } ( z ) ) ^ { r } \big ) = k ! \operatorname { c o e f f } _ { k } \left( \displaystyle \sum _ { r = 2 } ^ { \infty } C _ { \sigma } ^ { r - 1 } \big ( F _ { \ell } ( z ) \big ) ^ { r } \right) . } \end{array}
$$

Notice that the truncation of $F _ { \ell }$ at degree $n _ { 0 }$ causes no loss: for $k \leq n _ { 0 }$ , the coeficient of $z ^ { k }$ in any power $F _ { \ell } ^ { r }$ depends only on coeficients of degrees at most k. We set $y : = c / L$ and claim that $F _ { \ell } ( y ) \le C _ { 0 } y$ holds for all $\ell \in \{ 0 , \ldots , L \}$ . Looking at the definitions of $F _ { 0 }$ and $\Xi _ { k } ^ { ( 0 ) }$ for $k \in  { \mathbb { N } } _ { 0 }$ we immediately observe that $F _ { 0 } ( y ) = y \le C _ { 0 } y$ . This is the base case.

Assume now that $\ell \in \mathbb { N }$ and $F _ { \ell - 1 } ( y ) \le C _ { 0 } y$ holds. From this we obtain that

$$
C _ { \sigma } F _ { \ell - 1 } ( y ) \leq C _ { \sigma } C _ { 0 } y \leq C _ { \sigma } C _ { 0 } c \leq \frac { C _ { \sigma } C _ { 0 } } { 2 C _ { \sigma } C _ { 0 } } = \frac { 1 } { 2 }
$$

and therefore

$$
\sum _ { r = 2 } ^ { \infty } C _ { \sigma } ^ { r - 1 } \left( F _ { \ell - 1 } ( y ) \right) ^ { r } = \frac { C _ { \sigma } \left( F _ { \ell - 1 } ( y ) \right) ^ { 2 } } { 1 - C _ { \sigma } F _ { \ell - 1 } ( y ) } \le 2 C _ { \sigma } \left( F _ { \ell - 1 } ( y ) \right) ^ { 2 } \le 2 C _ { \sigma } C _ { 0 } ^ { 2 } y ^ { 2 } .
$$

Moreover, since $0 < c \leq 1$ and $L \geq 1$ we have $0 < y \le 1$ and therefore $\exp ( y ) - 1 \leq 2 y$ . Consequently, we have

$$
\begin{array} { r l } { F _ { \ell } ( y ) = \displaystyle \sum _ { k = 1 } ^ { n _ { \ell } } \frac { \Xi _ { \ell } ^ { ( \ell ) } } { k ! } y ^ { k } - \displaystyle \sum _ { k = 1 } ^ { n _ { \ell } } \frac { y ^ { k } } { k ! } \left( \alpha _ { \ell } A \mathrm { i } \overline { { \Xi _ { k } ^ { ( \ell - 1 ) } } } + \gamma _ { \ell } + \kappa _ { \ell } \displaystyle \sum _ { \sigma : \ell = 1 } ^ { \infty } A _ { | \tau | } \displaystyle \prod _ { \sigma = 1 } ^ { \ell } \frac { ( \varepsilon _ { \sigma } ^ { ( \ell - 1 ) } ) } { \kappa ! } \right) } & { } \\ { \le \displaystyle \sum _ { k = 1 } ^ { n _ { \ell } } \frac { y ^ { k } } { k ! } \left( \alpha _ { \ell } \overline { { \Xi _ { k } ^ { ( \ell - 1 ) } } } + \frac { \overline { { K } } } { \overline { { L } } } + \overline { { \kappa } } \displaystyle \sum _ { \sigma = 1 \ell ( k ) , \sigma \neq \ell ( k ) } ^ { \infty } A _ { | \tau | } \displaystyle \prod _ { \sigma = 1 } ^ { \ell } \frac { ( \varepsilon _ { \sigma } ^ { ( \ell - 1 ) } ) } { \kappa ! } \right) } & { } \\ { \le \displaystyle \sum _ { k = 1 } ^ { n _ { \ell } } \frac { y ^ { k } } { k ! } \left( \alpha _ { \ell } \overline { { \Xi _ { k } ^ { ( \ell - 1 ) } } } + \frac { \overline { { K } } } { \overline { { L } } } + \overline { { \kappa } } \displaystyle \mathrm { k } \mathrm { k } \log \mathrm { e r } _ { k } \left( \displaystyle \sum _ { \sigma = 2 } ^ { \infty } C _ { \sigma } ^ { 2 - 1 } ( F _ { \ell - 1 } ( z ) ) ^ { \tau } \right) \right) } & { } \\  \le \alpha _ { \ell } F _ { \ell - 1 } ( y ) + \displaystyle \frac { \overline { { K } } } { \overline { { L } } } \displaystyle \sum _ { k = 1 } ^ { n _ { \ell } } \frac  y ^ { k } \end{array}
$$

In the fourth line we used that every coeficient of $F _ { \ell - 1 }$ , and hence of every power $F _ { \ell - 1 } ^ { r } ,$ , is nonnegative. Therefore the sum of the first $n _ { 0 }$ coeficient contributions evaluated at $y > 0$ is bounded by the full power series evaluated at y. If we define $p _ { \ell } : = F _ { \ell } ( y ) / y$ , then we have $p _ { 0 } = 1$ and

$$
p _ { \ell } \leq \alpha _ { \ell } p _ { \ell - 1 } + \left( \frac { 2 \overline { { \kappa } } } { L } + 2 \overline { { \kappa } } C _ { \sigma } C _ { 0 } ^ { 2 } y \right) .
$$

By Proposition 23 we have

$$
\begin{array} { l } { \displaystyle { p _ { \ell } \leq \prod _ { j = 1 } ^ { \ell } \alpha _ { j } + \left( \frac { 2 \overline { { \kappa } } } { L } + 2 \overline { { \kappa } } C _ { \sigma } C _ { 0 } ^ { 2 } y \right) \sum _ { q = 1 } ^ { \ell } \left( \prod _ { j = q + 1 } ^ { \ell } \alpha _ { j } \right) \leq \overline { { \alpha } } + \left( \frac { 2 \overline { { \kappa } } } { L } + 2 \overline { { \kappa } } C _ { \sigma } C _ { 0 } ^ { 2 } y \right) \ell \overline { { \alpha } } } } \\ { \displaystyle { \quad \leq \overline { { \alpha } } ( 1 + 2 \overline { { \kappa } } ) + 2 \overline { { \alpha \kappa } } C _ { \sigma } C _ { 0 } ^ { 2 } c . } } \end{array}
$$

By definition we have $\overline { { \alpha } } ( 1 + 2 \overline { { \kappa } } ) \leq C _ { 0 } / 4$ and 2ακ $C _ { \sigma } C _ { 0 } ^ { 2 } c \leq C _ { 0 } / 2$ and we obtain

$$
\frac { F _ { \ell } ( y ) } { y } = p _ { \ell } \le \overline { { \alpha } } ( 1 + 2 \overline { { \kappa } } ) + 2 \overline { { \alpha \kappa } } C _ { \sigma } C _ { 0 } ^ { 2 } c \le \frac { C _ { 0 } } { 4 } + \frac { C _ { 0 } } { 2 } = \frac { 3 C _ { 0 } } { 4 } \le C _ { 0 }
$$

and $F _ { \ell } ( y ) \le C _ { 0 } y$ . Since y and all coeficients of $F _ { \ell }$ are non-negative we obtain

$$
\frac { \Xi _ { k } ^ { ( \ell ) } } { k ! } y ^ { k } \le \sum _ { j = 1 } ^ { n _ { 0 } } \frac { \Xi _ { j } ^ { ( \ell ) } } { j ! } y ^ { j } = F _ { \ell } ( y ) \le C _ { 0 } y .
$$

Hence,

$$
\Xi _ { k } ^ { ( \ell ) } \leq C _ { 0 } k ! y ^ { 1 - k } = C _ { 0 } k ! \left( { \frac { L } { c } } \right) ^ { k - 1 } = C _ { 0 } k ! ( C _ { 1 } L ) ^ { k - 1 }
$$

An application of Lemma 4 concludes the proof.

Remark 6 (Majorant-series interpretation). The generating function $F _ { \ell }$ is a majorant series for the square-free mixed derivatives considered in this paper. Evaluating it at $y = c / L$ and using $F _ { \ell } ( y ) \le C _ { 0 } y ~ .$ yields

$$
\Xi _ { k } ^ { ( \ell ) } \leq C _ { 0 } k ! y ^ { 1 - k } = C _ { 0 } k ! ( C _ { 1 } L ) ^ { k - 1 } .
$$

This resembles a Cauchy-type coeficient estimate and explains the factor $L ^ { k - 1 }$ . We stress, however, that the argument is a majorant-series argument in the derivative order. Since the theorem only

controls square-free mixed derivatives, it does not by itself assert a holomorphic extension of the network realization to a complex polydisc.

## 3. High probability bounds for random neural networks

In this section we show that the matrix and tangent conditions introduced above hold with high probability for suficiently wide Gaussian neural networks. We fix one probability space $( \Omega , { \mathfrak { F } } , \mathbb { P } )$ and we assume that it is rich enough to accommodate all random variables we introduce. For all $\ell \in  { \mathbb { N } } _ { 0 }$ , all $i \in [ n _ { \ell + 1 } ]$ and all $j \in [ n _ { \ell } ]$ we assume that $X _ { i , j } ^ { ( \ell ) }$ are i.i.d. standard normal random variables. For all $j \in [ n _ { 0 } ]$ let $\tau _ { j } ^ { ( 0 ) } > 0$ and define $\mathcal { W } _ { i , j } ^ { ( 0 ) } : = \tau _ { j } ^ { ( 0 ) } X _ { i , j } ^ { ( 0 ) }$ for all $i \in [ n _ { 1 } ]$ . For all $\ell \in \mathbb { N }$ let $\tau _ { \ell } > 0$ and define $\mathcal { W } ^ { ( \ell ) } : = \tau _ { \ell } X ^ { ( \ell ) }$ . For every $\ell \in  { \mathbb { N } } _ { 0 }$ we introduce a random variable $\Phi ^ { ( \ell ) } : \Omega \to \Theta _ { \ell }$ by $\Phi ^ { ( \ell ) } ( \omega ) : = \left( \mathcal { W } ^ { ( 0 ) } ( \omega ) , \dots , \mathcal { W } ^ { ( \ell ) } ( \omega ) \right)$ , which has values in the space of neural networks. If it is clear that the neural network has depth $L \in \mathbb { N }$ , then we just write $\Phi : \Omega  \Theta _ { L }$

For all $\eta \in ( 0 , 1 )$ , all $L \in \mathbb { N }$ and all $\ell \in \{ 1 , \ldots , L \}$ we define

$$
\kappa _ { \ell } ( \eta , L ) : = \tau _ { \ell } \left( \sqrt { n _ { \ell } } + \sqrt { n _ { \ell + 1 } } + \left( 2 \log \left( \frac { 8 L } { \eta } \right) \right) ^ { 1 / 2 } \right) .
$$

For all $n _ { 0 } \in \mathbb { N }$ , all $j \in [ n _ { 0 } ]$ and all $\eta \in ( 0 , 1 )$ we define

$$
\beta _ { j } ( \eta , n _ { 0 } ) : = \tau _ { j } ^ { ( 0 ) } \left( n _ { 1 } + 2 \sqrt { n _ { 1 } \log \left( \frac { 4 n _ { 0 } } { \eta } \right) } + 2 \log \left( \frac { 4 n _ { 0 } } { \eta } \right) \right) ^ { 1 / 2 } .
$$

For $\eta \in ( 0 , 1 )$ and $n _ { 0 } , L \in \mathbb { N }$ we define the event $\mathcal { E } _ { \mathrm { m a t } } ( \eta , L , n _ { 0 } )$ as the set of all $\omega \in \Omega$ such that $\Phi ( \omega ) \in \Theta _ { L } ^ { \mathrm { m a t } } \left( n _ { 0 } , \beta ( \eta , n _ { 0 } ) , \kappa ( \eta , L ) \right)$

Proposition 7. For all $\eta \in ( 0 , 1 )$ we have $\mathbb { P } \left( \mathcal { E } _ { \mathrm { m a t } } ( \eta , L , n _ { 0 } ) \right) \ge 1 - \eta / 2 .$

Proof. Fix one $\ell \in \{ 1 , \ldots , L \}$ . We apply Proposition 24 to the matrix $\mathscr { W } ^ { ( \ell ) }$ with $t = ( 2 \log { ( 8 L / \eta ) } ) ^ { 1 / 2 }$ We obtain that

$$
\| \mathcal { W } ^ { ( \ell ) } \| _ { 2  2 } \leq \tau _ { \ell } ( \sqrt { n _ { \ell } } + \sqrt { n _ { \ell + 1 } } + ( 2 \log ( \frac { 8 L } { \eta } ) ) ^ { 1 / 2 } ) = \kappa _ { \ell } ( \eta , L )
$$

with probability at least $1 - 2 \exp \left( - t ^ { 2 } / 2 \right) = 1 - 2 \eta / 8 L = 1 - \eta / 4 L$ . The probability that this happens for all $\ell = 1 , \ldots , L$ simultaneously is at least $1 - L \eta / 4 L = 1 - \eta / 4$ . For the first-layer column bounds let us fix $j \in \{ 1 , \dots , n _ { 0 } \}$ . For every $i = 1 , \ldots , n _ { 1 }$ the random variable $\mathcal { W } _ { i , j } ^ { ( 0 ) }$ is Gaussian and by Lemma 25 with $x = \log ( 4 n _ { 0 } / \eta )$ and $\sqrt { a _ { i } } = \tau _ { j } ^ { ( 0 ) }$ for all $i \in \{ 1 , \ldots , n _ { 1 } \}$ we obtain

$$
\mathbb { P } \left( \sum _ { i = 1 } ^ { n _ { 1 } } ( \tau _ { j } ^ { ( 0 ) } ) ^ { 2 } \left( \left( X _ { i , j } ^ { ( 0 ) } \right) ^ { 2 } - 1 \right) \geq 2 ( \tau _ { j } ^ { ( 0 ) } ) ^ { 2 } \left( n _ { 1 } \log \left( \frac { 4 n _ { 0 } } { \eta } \right) \right) ^ { 1 / 2 } + 2 ( \tau _ { j } ^ { ( 0 ) } ) ^ { 2 } \log \left( \frac { 4 n _ { 0 } } { \eta } \right) \right) \leq \frac { \eta } { 4 n _ { 0 } } .
$$

From this we deduce that $\| \mathcal { W } ^ { ( 0 ) } e _ { j } \| _ { 2 } \le \beta _ { j } ( \eta , n _ { 0 } )$ holds with probability at least $1 - \eta / ( 4 n _ { 0 } )$ . The probability that this happens for all $j \in [ n _ { 0 } ]$ simultaneously is at least $1 - \eta / 4$ . If we want all the bounds simultaneously we get a probability of at least $1 - \eta / 2$ □

Proposition 7 together with Theorem 1 immediately gives Lemma 8.

Lemma 8. Suppose that for every $r \in \mathbb N$ there exists $A _ { r } > 0$ such that $| \sigma ^ { ( r ) } ( x ) | \le A _ { r }$ holds for all $x \in \mathbb { R } . \ I f L , n _ { 0 } \in \mathbb { N }$ and $\eta \in ( 0 , 1 )$ , then

$$
\big \| \partial ^ { \nu } \mathcal { R } _ { \Phi ^ { ( \ell ) } } ( x ) \big \| _ { 2 } \leq B _ { | \nu | } ^ { ( \ell ) } \big ( \kappa \big ( \eta , L ) \big ) \prod _ { j = 1 } ^ { n _ { 0 } } ( \beta _ { j } \big ( \eta , n _ { 0 } \big ) \big ) ^ { \nu _ { j } } ,
$$

holds on the event $\mathcal { E } _ { \mathrm { m a t } } ( \eta , L , n _ { 0 } )$ , and in particular it holds at least with probability $1 - \eta / 2$ , for all $\nu \in \mathbb { N } _ { 0 } ^ { n _ { 0 } } \setminus \{ 0 \}$ , all $x \in \mathbb { R } ^ { n _ { 0 } }$ , and all $\ell = 0 , \ldots , L$

Combining Lemma 20 with Corollary 2 gives the following result.

Corollary 9. Suppose there exists a constant $C _ { \sigma }$ such that $\left| \sigma ^ { ( r ) } ( x ) \right| \le A _ { r } : = r ! C _ { \sigma } ^ { r - 1 }$ holds for all $r \in \mathbb N$ and all $x \in \mathbb { R }$ . If $L , n _ { 0 } \in \mathbb { N }$ and $\eta \in ( 0 , 1 )$ , then

$$
\| D ^ { u } \mathcal { R } _ { \Phi ^ { ( \ell ) } } ( x ) \| _ { 2 } \le \left( \prod _ { m = 1 } ^ { \ell } \kappa _ { m } ( \eta , L ) \right) ( | u | ! ) \left( C _ { \sigma } \sum _ { p = 0 } ^ { \ell - 1 } \prod _ { t = 1 } ^ { p } \kappa _ { t } ( \eta , L ) \right) ^ { | u | - 1 } \prod _ { j \in u } \beta _ { j } ( \eta , n _ { 0 } ) ,
$$

holds on the event $\mathcal { E } _ { \mathrm { m a t } } ( \eta , L , n _ { 0 } )$ , and in particular it holds at least with probability $1 - \eta / 2$ , for all non-empty $u \subseteq [ n _ { 0 } ]$ , all $x \in \mathbb { R } ^ { n _ { 0 } }$ , and all $\ell = 0 , \ldots , L$

Let us for a moment assume that $n _ { 1 } = n _ { 2 } = \cdot \cdot \cdot = n _ { L + 1 } = n$ . For the Xavier initialization we obtain

$$
\kappa _ { \ell } ( \eta , L ) : = \frac { \sqrt { 2 } } { \sqrt { 2 n } } \left( 2 \sqrt { n } + \left( 2 \log \left( \frac { 8 L } { \eta } \right) \right) ^ { 1 / 2 } \right) \geq 2
$$

for all $\ell \in [ L ]$ . This means that the bound from Corollary 9 grows exponentially with the depth $L .$ The goal of this section is to get a better bound with high probability. We will see that this is possible at least for very wide neural networks. We assume that $\beta _ { j } > 0$ holds for all $j \in [ n _ { 0 } ]$ . For $L \in \mathbb { N }$ and $\ell \in [ L ]$ we define

$$
\mathcal { A } _ { u } ^ { ( \ell ) } ( v , L , \beta ) : = \left\{ x \in [ 0 , 1 ] ^ { n _ { 0 } } : L \left\| \boldsymbol { G } _ { u } ^ { ( \ell ) } ( x , v ) \right\| _ { 2 } \geq \prod _ { j \in u } \beta _ { j } \right\} ,
$$

where $\emptyset \neq u \subseteq [ n _ { 0 } ]$ and $v \in \Theta _ { \ell - 1 }$ . We define

$$
\mathcal { T } _ { u } ^ { ( \ell ) } \left( v , L , \beta \right) : = \left\{ \left. G _ { u } ^ { ( \ell ) } \left( x , v \right) \right. _ { 2 } ^ { - 1 } G _ { u } ^ { ( \ell ) } \left( x , v \right) \big \vert x \in A _ { u } ^ { ( \ell ) } ( v , L , \beta ) \right\} \subseteq S ^ { n _ { \ell } - 1 }
$$

and

$$
\mathcal { T } _ { \ell } ( v , L , \beta ) : = \bigcup _ { \emptyset \neq u \subseteq [ n _ { 0 } ] } \mathcal { T } _ { u } ^ { ( \ell ) } ( v , L , \beta ) .
$$

Furthermore, we set

$$
\begin{array} { r } { M _ { k } ^ { ( \ell ) } ( \kappa ) : = A _ { 2 } B _ { 1 } ^ { ( \ell - 1 ) } ( \kappa ) B _ { k } ^ { ( \ell - 1 ) } ( \kappa ) + A _ { 1 } B _ { k + 1 } ^ { ( \ell - 1 ) } ( \kappa ) . } \end{array}
$$

Proposition 10. $I f \varepsilon > 0$ and $L , n _ { 0 } \in \mathbb { N }$ , then for every $\ell \in [ L ]$ there exists a finite set that we denote by $\mathcal { T } _ { \ell } \left( \varepsilon , L , n _ { 0 } , \beta , \kappa \right)$ , with

$$
| \mathcal { I } _ { \ell } \left( \varepsilon , L , n _ { 0 } , \beta , \kappa \right) | \leq \sum _ { k = 1 } ^ { n _ { 0 } } \binom { n _ { 0 } } { k } \prod _ { j = 1 } ^ { n _ { 0 } } \left( 1 + \frac { 2 n _ { 0 } L M _ { k } ^ { ( \ell ) } ( \kappa ) \beta _ { j } } { \varepsilon } \right) = : \exp \left( \mathfrak { H } _ { \ell } \left( \varepsilon , L , n _ { 0 } , \kappa , \beta \right) \right)
$$

and for every $i \in \mathcal { I } _ { \ell } \left( \varepsilon , L , n _ { 0 } , \beta , \kappa \right)$ there is a Borel measurable map $\xi _ { \varepsilon , L , n _ { 0 } , \beta , \kappa , i } ^ { ( \ell ) } : \Theta _ { \ell - 1 } \to S ^ { n _ { \ell } - 1 }$ such that for every $v \in \Theta _ { \ell - 1 } ^ { \mathrm { m a t } } ( n _ { 0 } , \beta , \kappa )$ and every $y \in \mathcal { T } _ { \ell } \left( v , L , \beta \right)$ there exists $i \in \mathcal { I } _ { \ell } \left( \varepsilon , L , n _ { 0 } , \beta , \kappa \right)$ such that

$$
\begin{array} { r } { \left\| y - \xi _ { \varepsilon , L , n _ { 0 } , \beta , \kappa , i } ^ { ( \ell ) } ( v ) \right\| _ { 2 } \leq \varepsilon . } \end{array}
$$

Proof. Fix $\ell \in [ L ]$ and a non-empty set $u \subseteq [ n _ { 0 } ]$ . For $j \in [ n _ { 0 } ]$ we define

$$
N _ { u , j } ^ { ( \ell ) } ( n _ { 0 } , L , \kappa , \beta , \varepsilon ) : = \operatorname* { m a x } \left\{ 1 , \left\lceil \frac { 2 n _ { 0 } L M _ { | u | } ^ { ( \ell ) } ( \kappa ) \beta _ { j } } { \varepsilon } \right\rceil \right\}
$$

and

$$
I _ { u , j , i } ^ { ( \ell ) } ( n _ { 0 } , L , \kappa , \beta , \varepsilon ) : = \left[ \frac { i - 1 } { N _ { u , j } ^ { ( \ell ) } ( n _ { 0 } , L , \kappa , \beta , \varepsilon ) } , \frac { i } { N _ { u , j } ^ { ( \ell ) } ( n _ { 0 } , L , \kappa , \beta , \varepsilon ) } \right] , \quad i = 1 , \dots , N _ { u , j } ^ { ( \ell ) } ( n _ { 0 } , L , \kappa , \beta , \varepsilon ) .
$$

If $k _ { j } \in \left[ N _ { u , j } ^ { ( \ell ) } ( n _ { 0 } , L , \kappa , \beta , \varepsilon ) \right]$ for every $j \in [ n _ { 0 } ]$ , then we define

$$
R _ { u , k } ^ { ( \ell ) } \left( n _ { 0 } , L , \kappa , \beta , \varepsilon \right) : = I _ { u , 1 , k _ { 1 } } ^ { ( \ell ) } ( n _ { 0 } , L , \kappa , \beta , \varepsilon ) \times \dots \times I _ { u , n _ { 0 } , k _ { n _ { 0 } } } ^ { ( \ell ) } ( n _ { 0 } , L , \kappa , \beta , \varepsilon )
$$

and

$$
\begin{array} { r } { \mathcal { Q } _ { u } ^ { ( \ell ) } ( n _ { 0 } , L , \kappa , \beta , \varepsilon ) : = \left\{ R _ { u , k } ^ { ( \ell ) } \left( n _ { 0 } , L , \kappa , \beta , \varepsilon \right) \big | \forall j \in [ n _ { 0 } ] : k _ { j } \in \left[ N _ { u , j } ^ { ( \ell ) } ( n _ { 0 } , L , \kappa , \beta , \varepsilon ) \right] \right\} } \end{array}
$$

is a set that covers $[ 0 , 1 ] ^ { n _ { 0 } }$ . The set

$$
\mathcal { I } _ { \ell } \left( \varepsilon , L , n _ { 0 } , \beta , \kappa \right) : = \Big \{ ( u , R ) \ | \ \emptyset \neq u \subseteq [ n _ { 0 } ] , R \in \mathcal { Q } _ { u } ^ { ( \ell ) } ( n _ { 0 } , L , \kappa , \beta , \varepsilon ) \Big \}
$$

satisfies

$$
\begin{array} { r l } & { \displaystyle | \mathcal { I } _ { \ell } ( \varepsilon , L , n _ { 0 } , \beta , \kappa ) | = \sum _ { 0 \neq u \subseteq [ n _ { 0 } ] } \left| \mathcal { Q } _ { u } ^ { ( \ell ) } ( n _ { 0 } , L , \kappa , \beta , \varepsilon ) \right| = \sum _ { \emptyset \neq u \subseteq [ n _ { 0 } ] } \prod _ { j = 1 } ^ { n _ { 0 } } N _ { u , j } ^ { ( \ell ) } ( n _ { 0 } , L , \kappa , \beta , \varepsilon ) } \\ & { \qquad \leq \displaystyle \sum _ { 0 \neq u \subseteq [ n _ { 0 } ] } \prod _ { j = 1 } ^ { n _ { 0 } } \left( 1 + \frac { 2 n _ { 0 } L M _ { | u | } ^ { ( \ell ) } ( \kappa ) \beta _ { j } } { \varepsilon } \right) = \sum _ { k = 1 } ^ { n _ { 0 } } { \binom { n _ { 0 } } { k } } \prod _ { j = 1 } ^ { n _ { 0 } } \left( 1 + \frac { 2 n _ { 0 } L M _ { k } ^ { ( \ell ) } ( \kappa ) \beta _ { j } } { \varepsilon } \right) . } \end{array}
$$

It remains to construct the measurable maps. Fix $i = ( u , R ) \in \mathcal { I } _ { \ell } \left( \varepsilon , L , n _ { 0 } , \beta , \kappa \right)$ and define

$$
\begin{array} { r } { D _ { u , R } ^ { ( \ell ) } \left( L , \beta , \kappa \right) : = \left\{ v \in \Theta _ { \ell - 1 } ^ { \operatorname* { m a t } } ( n _ { 0 } , \beta , \kappa ) \mid \mathcal { A } _ { u } ^ { ( \ell ) } ( v , L , \beta ) \cap R \neq \emptyset \right\} . } \end{array}
$$

The set

$$
\Gamma _ { u , R } ^ { ( \ell ) } : = \left\{ ( v , x ) \in \Theta _ { \ell - 1 } ^ { \mathrm { m a t } } ( n _ { 0 } , \beta , \kappa ) \times R : L \left\| { G } _ { u } ^ { ( \ell ) } ( x , v ) \right\| _ { 2 } \geq \prod _ { j \in u } \beta _ { j } \right\}
$$

is closed in $\Theta _ { \ell - 1 } ^ { \mathrm { m a t } } ( n _ { 0 } , \beta , \kappa ) \times R$ , since $G _ { u } ^ { ( \ell ) } : \Theta _ { \ell - 1 } ^ { \mathrm { m a t } } ( n _ { 0 } , \beta , \kappa ) \times R  \mathbb { R } ^ { n _ { \ell } }$ is a continuous map. Since R is compact we can apply Lemma 28 to obtain that the map

$$
\begin{array} { r } { q _ { u , R , L , \beta , \kappa } ^ { ( \ell ) } : D _ { u , R } ^ { ( \ell ) } ( L , \beta , \kappa )  R \quad \mathrm { g i v e n ~ b y } \quad q _ { u , R , L , \beta , \kappa } ^ { ( \ell ) } ( v ) : = \mathrm { l e x m i n } ( A _ { u } ^ { ( \ell ) } ( v , L , \beta ) \cap R ) } \end{array}
$$

is Borel measurable, where lexmin is the minimum in the lexicographic order. The set $\Theta _ { \ell - 1 } ^ { \mathrm { m a t } } ( n _ { 0 } , \beta , \kappa )$ is closed in $\Theta _ { \ell - 1 }$ , since it is defined by finitely many continuous norm inequalities. Hence, $D _ { u , R } ^ { ( \ell ) } ( L , \beta , \kappa )$ which is closed relative to $\Theta _ { \ell - 1 } ^ { \mathrm { m a t } } ( n _ { 0 } , \beta , \kappa )$ , is a Borel subset of $\Theta _ { \ell - 1 }$ . On $D _ { u , R } ^ { ( \ell ) } ( L , \beta , \kappa )$ , the selected point belongs to the active set, and therefore

$$
\left\| G _ { u } ^ { ( \ell ) } \left( q _ { u , R , L , \beta , \kappa } ^ { ( \ell ) } ( v ) , v \right) \right\| _ { 2 } \geq \frac { 1 } { L } \prod _ { j \in u } \beta _ { j } > 0 .
$$

Consequently, the map $\xi _ { \varepsilon , L , n _ { 0 } , \beta , \kappa , ( u , R ) } ^ { ( \ell ) } : \Theta _ { \ell - 1 } \to S ^ { n _ { \ell } - 1 }$ given by

$$
\xi _ { \varepsilon , L , n _ { 0 } , \beta , \kappa , ( u , R ) } ^ { ( \ell ) } ( v ) : = \left\{ \begin{array} { l l } { \left\| G _ { u } ^ { ( \ell ) } \left( q _ { u , R , L , \beta , \kappa } ^ { ( \ell ) } ( v ) , v \right) \right\| _ { 2 } ^ { - 1 } G _ { u } ^ { ( \ell ) } \left( q _ { u , R , L , \beta , \kappa } ^ { ( \ell ) } ( v ) , v \right) } & { , \mathrm { i f ~ } v \in D _ { u , R } ^ { ( \ell ) } ( L , \beta , \kappa ) } \\ { e _ { 1 } } & { , \mathrm { o t h e r w i s e } } \end{array} \right.
$$

is Borel measurable. If $v \in \Theta _ { \ell - 1 } ^ { \mathrm { m a t } } ( n _ { 0 } , \beta , \kappa )$ and $y \in \mathcal { T } _ { \ell } ( v , L , \beta )$ , then there exists a non-empty $u ~ \subseteq ~ [ n _ { 0 } ]$ and a point $x \ \in \ \mathcal { A } _ { u } ^ { ( \ell ) } ( v , L , \beta )$ such that $y ~ = ~ \left\| G _ { u } ^ { ( \ell ) } \left( x , v \right) \right\| _ { 2 } ^ { - 1 } G _ { u } ^ { ( \ell ) } \left( x , v \right)$ . Choosing $R \in \mathcal { Q } _ { u } ^ { ( \ell ) } ( n _ { 0 } , L , \kappa , \beta , \varepsilon )$ such that $x \in R$ we have by construction $q _ { u , R , L , \beta , \kappa } ^ { ( \ell ) } ( v ) \in \mathcal { A } _ { u } ^ { ( \ell ) } ( v , L , \beta ) \cap R .$ Lemma 30 gives

$$
\begin{array} { r l } { \displaystyle \left\| G _ { u } ^ { ( \ell ) } ( x , v ) - G _ { u } ^ { ( \ell ) } \left( q _ { u , R , L , \beta , \kappa } ^ { ( \ell ) } ( v ) , v \right) \right\| _ { 2 } \leq M _ { | u | } ^ { ( \ell ) } ( \kappa ) \prod _ { j \in u } \beta _ { j } \sum _ { m = 1 } ^ { n _ { 0 } } \beta _ { m } \left| x _ { m } - \left( q _ { u , R , L , \beta , \kappa } ^ { ( \ell ) } ( v ) \right) _ { m } \right| } & { } \\ { \leq { M _ { | u | } ^ { ( \ell ) } ( \kappa ) \prod _ { j \in u } \beta _ { j } \sum _ { m = 1 } ^ { n _ { 0 } } \beta _ { m } \left( N _ { u , m } ^ { ( \ell ) } ( n _ { 0 } , L , \kappa , \beta , \varepsilon ) \right) ^ { - 1 } \leq \frac { \varepsilon } { 2 L } \prod _ { j \in u } \beta _ { j } . } } & { } \end{array}
$$

By Lemma 31 we have

$$
\begin{array} { r l } & { \left. y - \xi _ { \varepsilon , L , n _ { 0 } , \beta , \kappa , ( u , R ) } ^ { ( \ell ) } ( v ) \right. _ { 2 } = \left. \frac { G _ { u } ^ { ( \ell ) } \left( x , v \right) } { \left. G _ { u } ^ { ( \ell ) } \left( x , v \right) \right. _ { 2 } } - \frac { G _ { u } ^ { ( \ell ) } \left( q _ { u , R , L , \beta , \kappa } ^ { ( \ell ) } ( v ) , v \right) } { \left. G _ { u } ^ { ( \ell ) } \left( q _ { u , R , L , \beta , \kappa } ^ { ( \ell ) } ( v ) , v \right) \right. _ { 2 } } \right. _ { 2 } } \\ & { \qquad \leq \frac { 2 \left. G _ { u } ^ { ( \ell ) } \left( x , v \right) - G _ { u } ^ { ( \ell ) } \left( q _ { u , R , L , \beta , \kappa } ^ { ( \ell ) } ( v ) , v \right) \right. _ { 2 } } { \operatorname* { m i n } \left\{ \left. G _ { u } ^ { ( \ell ) } \left( x , v \right) \right. _ { 2 } , \left. G _ { u } ^ { ( \ell ) } \left( q _ { u , R , L , \beta , \kappa } ^ { ( \ell ) } ( v ) , v \right) \right. _ { 2 } \right\} } \leq \varepsilon . } \end{array}
$$

Indeed, both x and $q _ { u , R , L , \beta , \kappa } ^ { ( \ell ) } ( v )$ belong to the active set, so each norm in the denominator is at least $L ^ { - 1 } \prod _ { j \in u } \beta _ { j }$ □

For $\eta \in ( 0 , 1 )$ and $n _ { 0 } , L \in \mathbb { N }$ we define $\alpha _ { \ell } ( L ) : = \tau _ { \ell } \sqrt { n _ { \ell + 1 } } \left( 1 + 1 / L \right)$ and $\gamma _ { \ell } ( \eta , L ) : = \kappa _ { \ell } ( \eta , L ) / L$ We define the tangent event by

$$
\mathcal { E } _ { \mathrm { t a n g } } ( \eta , L , n _ { 0 } ) : = \left\{ \omega \in \Omega : \Phi ( \omega ) \in \Theta _ { L } ^ { \mathrm { t a n g } } ( n _ { 0 } , \beta ( \eta , n _ { 0 } ) , \alpha ( L ) , \gamma ( \eta , L ) ) \right\} .
$$

Theorem 11. Let $\eta \in ( 0 , 1 )$ and $L , n _ { 0 } \in \mathbb { N }$ . If

$$
n _ { \ell + 1 } \geq 8 L ^ { 2 } \left( \mathfrak { H } _ { \ell } \left( \frac { \tau _ { \ell } \sqrt { n _ { \ell + 1 } } } { 2 L \kappa _ { \ell } ( \eta , L ) } , L , n _ { 0 } , \kappa ( \eta , L ) , \beta ( \eta , n _ { 0 } ) \right) + \log \left( \frac { 2 L } { \eta } \right) \right)
$$

holds for all $\ell \in [ L ]$ , then $\mathbb { P } \left( \mathcal { E } _ { \mathrm { m a t } } \left( \eta , L , n _ { 0 } \right) \cap \mathcal { E } _ { \mathrm { t a n g } } \left( \eta , L , n _ { 0 } \right) \right) \geq 1 - \eta$

Proof. Fix $\ell \in [ L ]$ and set $\varepsilon _ { \ell } : = \tau _ { \ell } \sqrt { n _ { \ell + 1 } } / 2 L \kappa _ { \ell } ( \eta , L )$ . By Proposition 10, there is a finite index set $\mathcal { K } _ { \ell } ( \eta , L , n _ { 0 } ) : = \mathcal { I } _ { \ell } \left( \varepsilon _ { \ell } , L , n _ { 0 } , \beta ( \eta , n _ { 0 } \dot { ) } , \kappa ( \eta , L ) \right)$ satisfying

$$
\begin{array} { r } { \left| { K } _ { \ell } ( \eta , L , n _ { 0 } ) \right| \le \exp \left( \mathfrak { H } _ { \ell } \left( \varepsilon _ { \ell } , L , n _ { 0 } , \kappa ( \eta , L ) , \beta ( \eta , n _ { 0 } ) \right) \right) . } \end{array}
$$

For every $i \in \mathcal { K } _ { \ell } ( \eta , L , n _ { 0 } )$ , let $\zeta _ { \eta , L , n _ { 0 } , i } ^ { ( \ell ) } : \Theta _ { \ell - 1 } \longrightarrow S ^ { n _ { \ell } - 1 }$ be the measurable net map from Proposition 10. Thus, for every $v \in \Theta _ { \ell - 1 } ^ { \mathrm { m a t } } \left( n _ { 0 } , \beta ( \eta , n _ { 0 } ) , \kappa ( \eta , L ) \right)$ and every $y \in \mathcal { T } _ { \ell } ( v , L , \beta ( \eta , n _ { 0 } ) )$ ), there exists $i \in \mathcal { K } _ { \ell } ( \eta , L , n _ { 0 } )$ such that

$$
\left\| y - \zeta _ { \eta , L , n _ { 0 } , i } ^ { ( \ell ) } ( v ) \right\| _ { 2 } \leq \varepsilon _ { \ell } .
$$

Define the measurable set

$$
D _ { \ell } ( \eta , L , n _ { 0 } ) : = \bigcup _ { i \in K _ { \ell } ( \eta , L , n _ { 0 } ) } \Big \{ ( v , w ) : \Big \| w \zeta _ { \eta , L , n _ { 0 } , i } ^ { ( \ell ) } ( v ) \Big \| _ { 2 } \geq \tau _ { \ell } \sqrt { n _ { \ell + 1 } } \left( 1 + \frac { 1 } { 2 L } \right) \Big \} ,
$$

where $( v , w ) \in \Theta _ { \ell - 1 } \times \mathbb { R } ^ { n _ { \ell + 1 } \times n _ { \ell } }$ . Indeed, for each $i ,$ the map $( v , w ) \longmapsto w \zeta _ { \eta , L , n _ { 0 } , i } ^ { ( \ell ) } ( v )$ is Borel measurable, since $v \mapsto \zeta _ { \eta , L , n _ { 0 } , i } ^ { ( \ell ) } ( v )$ is Borel and matrix-vector multiplication is continuous. Hence $D _ { \ell } ( \eta , L , n _ { 0 } )$ is Borel as a finite union of Borel sets. Let

$$
\begin{array} { r } { \mathcal { B } _ { \ell } ( \eta , L , n _ { 0 } ) : = \left\{ \omega : \left( \Phi ^ { ( \ell - 1 ) } ( \omega ) , \mathcal { W } ^ { ( \ell ) } ( \omega ) \right) \in D _ { \ell } ( \eta , L , n _ { 0 } ) \right\} . } \end{array}
$$

We first estimate the probability of the corresponding event for a fixed previous-layer parameter v. Fix $i \in \mathcal { K } _ { \ell } ( \eta , L , n _ { 0 } )$ and set $z : = \zeta _ { \eta , L , n _ { 0 } , i } ^ { ( \ell ) } ( v ) \in S ^ { n _ { \ell } - 1 }$ . For $r = 1 , \ldots , n _ { \ell + 1 }$ define

$$
Y _ { r } : = \sum _ { j = 1 } ^ { n _ { \ell } } z _ { j } X _ { r , j } ^ { ( \ell ) } .
$$

Since $z$ is deterministic and has Euclidean norm one, the random variables $Y _ { 1 } , \dots , Y _ { n _ { \ell + 1 } }$ are independent standard normal random variables. Moreover,

$$
\| \mathcal { W } ^ { ( \ell ) } z \| _ { 2 } ^ { 2 } = \tau _ { \ell } ^ { 2 } \sum _ { r = 1 } ^ { n _ { \ell + 1 } } Y _ { r } ^ { 2 } .
$$

Apply Lemma 25 with $a _ { 1 } = \cdot \cdot \cdot = a _ { n _ { \ell + 1 } } = 1$ and $x = n _ { \ell + 1 } / ( 8 L ^ { 2 } )$ . The corresponding deviation threshold is

$$
\frac { n _ { \ell + 1 } } { \sqrt { 2 } L } + \frac { n _ { \ell + 1 } } { 4 L ^ { 2 } } \leq n _ { \ell + 1 } \left[ \left( 1 + \frac { 1 } { 2 L } \right) ^ { 2 } - 1 \right] .
$$

Hence

$$
\mathbb { P } \left( \lVert \mathcal { W } ^ { ( \ell ) } z \rVert _ { 2 } \ge \tau _ { \ell } \sqrt { n _ { \ell + 1 } } \left( 1 + \frac { 1 } { 2 L } \right) \right) \le \exp \left( - \frac { n _ { \ell + 1 } } { 8 L ^ { 2 } } \right) .
$$

Let $g _ { \ell } ( v ) : = \mathbb { E } \left[ \mathbf { 1 } _ { D _ { \ell } ( \eta , L , n _ { 0 } ) } \left( v , \mathcal { W } ^ { ( \ell ) } \right) \right]$ . The union bound and the assumed width condition give

$$
g _ { \ell } ( v ) \leq | \mathcal { K } _ { \ell } ( \eta , L , n _ { 0 } ) | \exp \left( - \frac { n _ { \ell + 1 } } { 8 L ^ { 2 } } \right) \leq \exp \left( \mathfrak { H } _ { \ell } \left( \varepsilon _ { \ell } , L , n _ { 0 } , \kappa ( \eta , L ) , \beta ( \eta , n _ { 0 } ) \right) - \frac { n _ { \ell + 1 } } { 8 L ^ { 2 } } \right) \leq \frac { \eta } { 2 L } .
$$

Since $\Phi ^ { ( \ell - 1 ) }$ and $\mathscr { W } ^ { ( \ell ) }$ are independent, Lemma 32 yields P $\left( \mathcal { B } _ { \ell } ( \eta , L , n _ { 0 } ) \right) = \mathbb { E } \left[ g _ { \ell } \left( \Phi ^ { ( \ell - 1 ) } \right) \right] \leq \eta / ( 2 L )$ Consequently,

$$
\mathbb { P } \left( \bigcup _ { \ell = 1 } ^ { L } \mathcal { B } _ { \ell } ( \eta , L , n _ { 0 } ) \right) \leq \frac { \eta } { 2 } .
$$

Together with Proposition 7, this shows that the event

$$
\mathcal { G } : = \mathcal { E } _ { \operatorname* { m a t } } ( \eta , L , n _ { 0 } ) \cap \bigcap _ { \ell = 1 } ^ { L } \left( \Omega \setminus \mathcal { B } _ { \ell } ( \eta , L , n _ { 0 } ) \right)
$$

has probability at least $1 - \eta$ . It remains to show that $\mathcal { G } \subseteq \mathcal { E } _ { \mathrm { t a n g } } ( \eta , L , n _ { 0 } )$ . Fix $\omega \in { \mathcal { G } } , \ell \in [ L ]$ , a non-empty set $u \subseteq [ n _ { 0 } ]$ , and $x \in [ 0 , 1 ] ^ { n _ { 0 } }$ . If

$$
L \left| \left| G _ { u } ^ { ( \ell ) } \left( x , \Phi ^ { ( \ell - 1 ) } ( \omega ) \right) \right| \right| _ { 2 } < \prod _ { j \in u } \beta _ { j } ( \eta , n _ { 0 } ) ,
$$

then the matrix event gives

$$
\left\| \mathcal { W } ^ { ( \ell ) } ( \omega ) G _ { u } ^ { ( \ell ) } ( x , \Phi ^ { ( \ell - 1 ) } ( \omega ) ) \right\| _ { 2 } \le \kappa \ell ( \eta , L ) \left\| G _ { u } ^ { ( \ell ) } ( x , \Phi ^ { ( \ell - 1 ) } ( \omega ) ) \right\| _ { 2 } \le \frac { \kappa _ { \ell } ( \eta , L ) } { L } \prod _ { j \in u } \beta _ { j } ( \eta , n _ { 0 } ) .
$$

Otherwise, the normalized tangent

$$
q : = \frac { G _ { u } ^ { ( \ell ) } \left( x , \Phi ^ { ( \ell - 1 ) } ( \omega ) \right) } { \left\| G _ { u } ^ { ( \ell ) } \left( x , \Phi ^ { ( \ell - 1 ) } ( \omega ) \right) \right\| _ { 2 } }
$$

belongs to ${ \mathcal { T } } _ { \ell } ( \Phi ^ { ( \ell - 1 ) } ( \omega ) , L , \beta ( \eta , n _ { 0 } ) )$ . Choose a net point $\zeta _ { i }$ with $\| q - \zeta _ { i } \| _ { 2 } \leq \varepsilon _ { \ell }$ . Since ω $\notin B _ { \ell } ( \eta , L , n _ { 0 } )$ and $\omega \in \mathcal { E } _ { \mathrm { m a t } } ( \eta , L , n _ { 0 } )$

$$
\begin{array} { l } { \displaystyle | | \mathcal { W } ^ { ( \ell ) } q | | _ { 2 } \le | | \mathcal { W } ^ { ( \ell ) } \zeta _ { i } | | _ { 2 } + | | \mathcal { W } ^ { ( \ell ) } | | _ { 2 \to 2 } | | q - \zeta _ { i } | | _ { 2 } < \tau _ { \ell } \sqrt { n _ { \ell + 1 } } \left( 1 + \frac { 1 } { 2 L } \right) + \kappa _ { \ell } ( \eta , L ) \varepsilon _ { \ell } } \\ { \displaystyle \qquad = \tau _ { \ell } \sqrt { n _ { \ell + 1 } } \left( 1 + \frac { 1 } { L } \right) . } \end{array}
$$

Multiplication by the tangent norm proves (3) with $\alpha _ { \ell } ( L ) = \tau _ { \ell } \sqrt { n _ { \ell + 1 } } \left( 1 + 1 / L \right)$ and $\gamma _ { \ell } ( \eta , L ) =$ $\kappa _ { \ell } ( \eta , L ) / L$ . Thus, $\omega \in \mathcal { E } _ { \mathrm { t a n g } } ( \eta , L , n _ { 0 } )$ , which completes the proof. □

Theorem 12. $I f \overline { { \kappa } } , \overline { { \alpha } } \geq 1$ and there exists $C _ { \sigma } > 0$ such that $\left| \sigma ^ { ( r ) } ( x ) \right| \le A _ { r } : = r ! C _ { \sigma } ^ { r - 1 }$ holds for all $r \in \mathbb { N }$ and all $x \in \mathbb { R }$ , then there exist constants $C _ { 0 } , C _ { 1 } > 0$ such that the following holds. $I f \eta \in ( 0 , 1 )$ and $L , n _ { 0 } \in \mathbb { N }$ are such that

(1) for all $\ell \in \left[ L \right]$ the inequality $\kappa \iota ( \eta , L ) \leq \overline { { \kappa } }$ holds,

(2) for all $1 \leq k \leq m \leq L$ the inequality

$$
\prod _ { \ell = k } ^ { m } \alpha _ { \ell } ( L ) = \left( 1 + { \frac { 1 } { L } } \right) ^ { m - k + 1 } \prod _ { \ell = k } ^ { m } \tau _ { \ell } { \sqrt { n \ell + 1 } } \leq { \overline { { \alpha } } }
$$

holds, and

(3) for all $\ell \in [ L ]$ the inequality

$$
n _ { \ell + 1 } \geq 8 L ^ { 2 } \left( \mathfrak { H } _ { \ell } \left( \frac { \tau _ { \ell } \sqrt { n _ { \ell + 1 } } } { 2 L \kappa _ { \ell } ( \eta , L ) } , L , n _ { 0 } , \kappa ( \eta , L ) , \beta ( \eta , n _ { 0 } ) \right) + \log \left( \frac { 2 L } { \eta } \right) \right)
$$

holds, then

$$
\mathbb { P } \left( \bigcap _ { \emptyset \neq u \subseteq [ n _ { 0 } ] } \left\{ \operatorname* { s u p } _ { x \in [ 0 , 1 ] ^ { n _ { 0 } } } \Vert D ^ { u } \mathcal { R } _ { \Phi ^ { ( L ) } } ( x ) \Vert _ { 2 } \leq C _ { 0 } | u | ! ( C _ { 1 } L ) ^ { | u | - 1 } \prod _ { j \in u } \beta _ { j } ( \eta , n _ { 0 } ) \right\} \right) \geq 1 - \eta .
$$

Proof. By Proposition 5 we obtain constants $C _ { 0 } , C _ { 1 } > 0$ depending on $\overline { { \kappa } } , \overline { { \alpha } } \geq 1$ and $C _ { \sigma } > 0$ Let $\eta \in ( 0 , 1 )$ and $L , n _ { 0 } \in \mathbb { N }$ satisfying the conditions of this theorem. For all $\ell \in [ L ]$ , we have $\begin{array} { r } { 0 \le \gamma _ { \ell } ( \eta , L ) = \frac { \kappa _ { \ell } ( \eta , L ) } { L } \le \overline { { \kappa } } / L } \end{array}$ and

$$
\prod _ { \ell = k } ^ { m } \alpha _ { \ell } ( L ) = \left( 1 + { \frac { 1 } { L } } \right) ^ { m - k + 1 } \prod _ { \ell = k } ^ { m } \tau _ { \ell } { \sqrt { n _ { \ell + 1 } } } \leq { \overline { { \alpha } } }
$$

For $\omega \in \mathcal { E } _ { \mathrm { m a t } } \left( \eta , L , n _ { 0 } \right) \cap \mathcal { E } _ { \mathrm { t a n g } } \left( \eta , L , n _ { 0 } \right)$ we have

$$
\Phi ^ { ( L ) } ( \omega ) \in \Theta _ { L } ^ { \mathrm { m a t } } \left( n _ { 0 } , \beta ( \eta , n _ { 0 } ) , \kappa ( \eta , L ) \right) \cap \Theta _ { L } ^ { \mathrm { t a n g } } \left( n _ { 0 } , \beta ( \eta , n _ { 0 } ) , \alpha ( L ) , \gamma ( \eta , L ) \right) \cap \Theta _ { L } ^ { \mathrm { t a n g } } \left( n _ { 0 } , \beta ( \eta , n _ { 0 } ) , \gamma ( \eta , L ) \right)
$$

and therefore

$$
\left\| D ^ { u } \mathcal { R } _ { \Phi ^ { ( L ) } ( \omega ) } ( x ) \right\| _ { 2 } \leq C _ { 0 } | u | ! ( C _ { 1 } L ) ^ { | u | - 1 } \prod _ { j \in u } \beta _ { j } ( \eta , n _ { 0 } )
$$

holds for all non-empty $u \subseteq [ n _ { 0 } ]$ and all $x \in [ 0 , 1 ] ^ { n _ { 0 } }$ . Since we have

$$
\begin{array} { r } { \mathbb { P } \left( \mathcal { E } _ { \mathrm { m a t } } \left( \eta , L , n _ { 0 } \right) \cap \mathcal { E } _ { \mathrm { t a n g } } \left( \eta , L , n _ { 0 } \right) \right) \geq 1 - \eta , } \end{array}
$$

by Theorem 11 the claim follows.

Proposition 13. For the activation $\sigma = \operatorname { t a n h }$ , there exist constants $C , C _ { 0 } , C _ { 1 } > 0$ with the following property. Let $\eta \in ( 0 , 1 )$ , let $L \in \mathbb { N } \setminus \{ 1 \}$ , let $n _ { 1 } = \cdot \cdot \cdot = n _ { L } = n$ and $n _ { L + 1 } = 1$ . Assume Xavier initialization, that is,

$$
\tau _ { j } ^ { ( 0 ) } = \sqrt { \frac { 2 } { n _ { 0 } + n } } , \quad f o r \ a l l \ j \in [ n _ { 0 } ] , \quad \tau _ { \ell } ^ { 2 } = \frac { 1 } { n } , \quad f o r \ a l l \ \ell \in [ L - 1 ] \quad a n d \quad \tau _ { L } ^ { 2 } = \frac { 2 } { n + 1 } .
$$

If

$$
n \geq C \left( L ^ { 3 } n _ { 0 } ^ { 2 } \left( 1 + \log ( n _ { 0 } ) \right) + L ^ { 2 } \left( 1 + \log \left( \frac { L } { \eta } \right) \right) \right) ,
$$

then

$$
\mathbb { P } \left( \bigcap _ { \emptyset \neq u \subseteq [ n _ { 0 } ] } \left\{ \operatorname* { s u p } _ { x \in [ 0 , 1 ] ^ { n _ { 0 } } } | D ^ { u } \mathcal { R } _ { \Phi ^ { ( L ) } } ( x ) | \leq C _ { 0 } | u | ! ( C _ { 1 } L ) ^ { | u | - 1 } \prod _ { j \in u } \beta _ { j } ( \eta , n _ { 0 } ) \right\} \right) \geq 1 - \eta .
$$

Proof. By Lemma 20, there exists a constant $C _ { \sigma } > 0$ such that $\begin{array} { r } { \left| \sigma ^ { ( r ) } ( x ) \right| \le A _ { r } : = r ! C _ { \sigma } ^ { r - 1 } } \end{array}$ holds for all $r \in \mathbb N$ and all $x \in \mathbb { R }$ . We define $\overline { { \kappa } } : = 3$ and $\overline { { \alpha } } : = \exp ( 1 )$ . By enlarging the universal constant C in the statement, we may assume that the width condition additionally implies $n \geq 2 \log \left( 1 6 L / \eta \right)$ and $n \geq 2 \log { ( 8 n _ { 0 } / \eta ) }$ . Indeed, the first inequality follows from the $L ^ { 2 } ( 1 + \log ( L / \eta ) )$ term, while the second follows from the combination of the $L ^ { \mathrm { 3 } } n _ { 0 } ^ { 2 } ( 1 + \log n _ { 0 } )$ and $L ^ { 2 } ( 1 + \log ( L / \eta ) )$ terms.

Set $\delta : = \eta / 2$ and $L _ { \mathrm { h } } : = L - 1$ . We will apply Theorem 12 to the hidden subnetwork $\Phi ^ { ( L _ { \mathrm { h } } ) } =$ $( \mathcal { W } ^ { ( 0 ) } , \dots , \mathcal { W } ^ { ( L - 1 ) } )$ . For this hidden subnetwork the depth is $L _ { \mathrm { h } } = L - 1$ , and $n _ { 1 } = \cdot \cdot \cdot = n _ { L _ { \mathrm { h } } + 1 } = n$ Moreover, for every $\ell \in [ L _ { \mathrm { h } } ]$ , we have $\tau _ { \ell } = 1 / \sqrt { n }$ . Hence, $\tau _ { \ell } \sqrt { n _ { \ell + 1 } } = 1$ for all $\ell \in [ L _ { \mathrm { h } } ]$

We first verify the assumptions of Theorem 12 for the hidden subnetwork with failure probability δ. For $\ell \in [ L _ { \mathrm { h } } ]$ , we have

$$
\kappa _ { \ell } ( \delta , L _ { \mathrm { h } } ) = \frac { 1 } { \sqrt { n } } \left( 2 \sqrt { n } + \sqrt { 2 \log \left( \frac { 8 L _ { \mathrm { h } } } { \delta } \right) } \right) = 2 + \sqrt { \frac { 2 \log \left( 1 6 L _ { \mathrm { h } } / \eta \right) } { n } } .
$$

By increasing the constant C in the lower bound on $n ,$ if necessary, we may assume that $n \geq$ 2 log $( 1 6 L _ { \mathrm { h } } / \eta )$ . Consequently, $\kappa _ { \ell } ( \delta , L _ { \mathrm { h } } ) \leq 3 = \overline { { \kappa } }$ for all $\ell \in [ L _ { \mathrm { h } } ]$ . Next, for all $1 \leq k \leq m \leq L _ { \mathrm { h } }$

$$
\left( 1 + \frac { 1 } { L _ { \mathrm { h } } } \right) ^ { m - k + 1 } \prod _ { \ell = k } ^ { m } \tau _ { \ell } \sqrt { n _ { \ell + 1 } } = \left( 1 + \frac { 1 } { L _ { \mathrm { h } } } \right) ^ { m - k + 1 } \le \left( 1 + \frac { 1 } { L _ { \mathrm { h } } } \right) ^ { L _ { \mathrm { h } } } \le e = \overline { { \alpha } } .
$$

Thus the first two assumptions of Theorem 12 are satisfied. It remains to verify the width condition. We first record a simple bound for $\beta _ { j } ( \delta , n _ { 0 } )$ . Since $\tau _ { j } ^ { ( 0 ) } = \sqrt { 2 / ( n _ { 0 } + n ) }$ , we have

$$
( \beta _ { j } ( \delta , n _ { 0 } ) ) ^ { 2 } = \frac { 2 } { n _ { 0 } + n } \left( n + 2 \sqrt { n \log \left( \frac { 4 n _ { 0 } } { \delta } \right) } + 2 \log \left( \frac { 4 n _ { 0 } } { \delta } \right) \right) .
$$

After increasing C again, the assumed lower bound on n implies

$$
n \geq 2 \log \left( \frac { 4 n _ { 0 } } { \delta } \right) = 2 \log \left( \frac { 8 n _ { 0 } } { \eta } \right) .
$$

Therefore, $\beta _ { j } ( \delta , n _ { 0 } ) \leq 3$ for all $j \in [ n _ { 0 } ]$ . Moreover, for every $\ell \in [ L _ { \mathrm { h } } ]$

$$
\varepsilon _ { \ell } : = \frac { \tau _ { \ell } \sqrt { n _ { \ell + 1 } } } { 2 L _ { \mathrm { h } } \kappa _ { \ell } ( \delta , L _ { \mathrm { h } } ) } = \frac { 1 } { 2 L _ { \mathrm { h } } \kappa _ { \ell } ( \delta , L _ { \mathrm { h } } ) } \geq \frac { 1 } { 6 L _ { \mathrm { h } } } .
$$

We now estimate the entropy term $\mathfrak { H } _ { \ell } \left( \varepsilon _ { \ell } , L _ { \mathrm { h } } , n _ { 0 } , \kappa ( \delta , L _ { \mathrm { h } } ) , \beta ( \delta , n _ { 0 } ) \right)$ . Since $\kappa _ { \ell } ( \delta , L _ { \mathrm { h } } ) \leq 3$ , the deterministic exponential derivative bound gives, for all $q \in \mathbb { N }$ and all $r \leq L _ { \mathrm { h } }$ 2

$$
B _ { q } ^ { ( r ) } ( \kappa ( \delta , L _ { \mathrm { h } } ) ) \leq 3 ^ { r } q ! \left( C _ { \sigma } \sum _ { p = 0 } ^ { r - 1 } 3 ^ { p } \right) ^ { q - 1 } \leq q ! C _ { \sigma } ^ { q - 1 } 3 ^ { r q } .
$$

Hence, for $1 \leq k \leq n _ { 0 }$ and $1 \leq \ell \leq L _ { \mathrm { h } }$

$$
M _ { k } ^ { ( \ell ) } ( \kappa ( \delta , L _ { \mathrm { h } } ) ) = A _ { 2 } B _ { 1 } ^ { ( \ell - 1 ) } B _ { k } ^ { ( \ell - 1 ) } + A _ { 1 } B _ { k + 1 } ^ { ( \ell - 1 ) } \leq C ( k + 1 ) ! C ^ { k } 3 ^ { \ell ( k + 1 ) } .
$$

Here and below, C denotes a constant depending only on $\sigma = \operatorname { t a n h }$ , and it may be enlarged finitely many times. Thus, log $\left( 1 + M _ { k } ^ { ( \ell ) } ( \kappa ( \delta , L _ { \mathrm { h } } ) ) \right) \le C \left( L n _ { 0 } + n _ { 0 } \log ( e n _ { 0 } ) \right)$ uniformly in $1 \leq k \leq n _ { 0 }$ and $1 \leq \ell \leq L _ { \mathrm { h } }$ . Using the definition of ${ \mathfrak H } _ { \ell }$ and $\begin{array} { r } { \sum _ { k = 1 } ^ { n _ { 0 } } \binom { n _ { 0 } } { k } \leq 2 ^ { n _ { 0 } } } \end{array}$ , together with $\beta _ { j } ( \delta , n _ { 0 } )  \leq 3$ and $\varepsilon _ { \ell } ^ { - 1 } \leq 6 L _ { \mathrm { h } }$ , we obtain

$$
\begin{array} { r l } & { \mathfrak { H } _ { \ell } \left( \varepsilon _ { \ell } , L _ { \mathrm { h } } , n _ { 0 } , \kappa ( \delta , L _ { \mathrm { h } } ) , \beta ( \delta , n _ { 0 } ) \right) \le \log \left( \displaystyle \sum _ { k = 1 } ^ { n _ { 0 } } \binom { n _ { 0 } } { k } \displaystyle \prod _ { j = 1 } ^ { n _ { 0 } } \left( 1 + C L ^ { 2 } n _ { 0 } M _ { k } ^ { ( \ell ) } ( \kappa ( \delta , L _ { \mathrm { h } } ) ) \right) \right) } \\ & { \qquad \le n _ { 0 } \log 2 + n _ { 0 } \displaystyle \operatorname* { m a x } _ { 1 \le k \le n _ { 0 } } \log \left( 1 + C L ^ { 2 } n _ { 0 } M _ { k } ^ { ( \ell ) } \right) } \\ & { \qquad \le C \left( L n _ { 0 } ^ { 2 } + n _ { 0 } ^ { 2 } \log ( e n _ { 0 } ) + n _ { 0 } \log ( e L ) \right) \le C L n _ { 0 } ^ { 2 } \left( 1 + \log ( n _ { 0 } ) \right) . } \end{array}
$$

Therefore, after increasing the constant C in the statement if necessary, the assumed lower bound

$$
n \geq C \left( L ^ { 3 } n _ { 0 } ^ { 2 } ( 1 + \log n _ { 0 } ) + L ^ { 2 } \left( 1 + \log \left( { \frac { L } { \eta } } \right) \right) \right)
$$

implies, for all $\ell \in [ L _ { \mathrm { h } } ]$ 2

$$
n \geq 8 L _ { \mathrm { h } } ^ { 2 } \left( \mathfrak { H } _ { \ell } \left( \varepsilon _ { \ell } , L _ { \mathrm { h } } , n _ { 0 } , \kappa ( \delta , L _ { \mathrm { h } } ) , \beta ( \delta , n _ { 0 } ) \right) + \log \left( \frac { 2 L _ { \mathrm { h } } } { \delta } \right) \right) .
$$

Thus, all assumptions of Theorem 12 are satisfied for the hidden subnetwork with depth $L _ { \mathrm { h } }$ and failure probability δ. Hence, there exists an event $E _ { \mathrm { h } }$ such that $\mathbb { P } ( E _ { \mathrm { h } } ) \geq 1 - \delta = 1 - \eta / 2$ and on $E _ { \mathrm { h } }$

$$
\| D ^ { v } \mathcal { R } _ { \Phi ^ { ( L - 1 ) } } ( x ) \| _ { 2 } \le C _ { 0 } ^ { \mathrm { h } } | v | ! \left( C _ { 1 } ^ { \mathrm { h } } L \right) ^ { | v | - 1 } \prod _ { j \in v } \beta _ { j } ( \delta , n _ { 0 } )
$$

holds simultaneously for all non-empty $v \subseteq [ n _ { 0 } ]$ and all $x \in [ 0 , 1 ] ^ { n _ { 0 } }$ . We now estimate the scalar readout layer. Since $n _ { L + 1 } = 1$ , the matrix ${ \mathcal { W } } ^ { ( L ) }$ is a row vector in $\mathbb { R } ^ { 1 \times n }$ . By Xavier initialization,

$$
\mathcal { W } ^ { ( L ) } = \sqrt { \frac { 2 } { n + 1 } } \left( X _ { 1 , 1 } ^ { ( L ) } , \dots , X _ { 1 , n } ^ { ( L ) } \right) .
$$

By Lemma 25, applied with $a _ { 1 } = \cdots = a _ { n } = 1$ and $x = \log \left( 4 / \eta \right)$ , we obtain, with probability at least $1 - \eta / 2$

$$
\sum _ { i = 1 } ^ { n } \left( X _ { 1 , i } ^ { ( L ) } \right) ^ { 2 } \leq n + 2 { \sqrt { n \log \left( { \frac { 4 } { \eta } } \right) + 2 \log \left( { \frac { 4 } { \eta } } \right) } } .
$$

The lower bound on $n ,$ , after increasing C if necessary, implies $n \geq 2 \log { ( 4 / \eta ) }$ . Therefore, on an event $E _ { \mathrm { o u t } }$ with $\mathbb { P } ( E _ { \mathrm { o u t } } ) \ge 1 - \eta / 2$ , we have

$$
 \mathcal { W } ^ { ( L ) }  _ { 2  2 } = \sqrt { \frac { 2 } { n + 1 } } ( \sum _ { i = 1 } ^ { n } ( X _ { 1 , i } ^ { ( L ) } ) ^ { 2 } ) ^ { 1 / 2 } \leq \sqrt { 2 ( 1 + 2 \sqrt { \frac { \log ( 4 / \eta ) } { n } } + 2 \frac { \log ( 4 / \eta ) } { n } ) } \leq 3 .
$$

By the union bound, $\mathbb { P } ( E _ { \mathrm { h } } \cap E _ { \mathrm { o u t } } ) \geq 1 - \eta$ . We now work on the event $E _ { \mathrm { h } } \cap E _ { \mathrm { o u t } }$ . Fix $\emptyset \neq u \subseteq [ n _ { 0 } ]$ and $x \in [ 0 , 1 ] ^ { n _ { 0 } }$ , and write $q : = | u |$ . Since the final layer is scalar, $\mathcal { R } _ { \Phi ^ { ( L ) } } ( x ) = b ^ { ( L + 1 ) } + \mathcal { W } ^ { ( L ) } \sigma \left( \mathcal { R } _ { \Phi ^ { ( L - 1 ) } } ( x ) \right)$ . The multivariate Fa\`a di Bruno formula yields

$$
D ^ { u } \mathcal { R } _ { \Phi ^ { ( L ) } } ( x ) = \mathcal { W } ^ { ( L ) } \sum _ { \pi \in \Pi ( u ) } h _ { | \pi | } ^ { ( L ) } ( x , \Phi ^ { ( L - 1 ) } ) \odot \bigodot \bigodot _ { v \in \pi } D ^ { v } \mathcal { R } _ { \Phi ^ { ( L - 1 ) } } ( x ) .
$$

Taking absolute values and using $\| \mathcal { W } ^ { ( L ) } \| _ { 2  2 } \le 3$ , the derivative bound $\| h _ { | \pi | } ^ { ( L ) } ( x , \Phi ^ { ( L - 1 ) } ) \| _ { \infty } \leq A _ { | \pi | }$ 9 and Lemma 19, we obtain

$$
\vert D ^ { u } \mathcal { R } _ { \Phi ^ { ( L ) } } ( x ) \vert \leq 3 \sum _ { \pi \in \Pi ( u ) } A _ { | \pi | } \prod _ { v \in \pi } \Vert D ^ { v } \mathcal { R } _ { \Phi ^ { ( L - 1 ) } } ( x ) \Vert _ { 2 } .
$$

Using the hidden-layer bound on $E _ { \mathrm { h } }$ , this gives

$$
\vert D ^ { u } \mathcal { R } _ { \Phi ^ { ( L ) } } ( x ) \vert \leq 3 \prod _ { j \in u } \beta _ { j } ( \delta , n _ { 0 } ) \sum _ { \pi \in \Pi ( u ) } A _ { \vert \pi \vert } \left( C _ { 0 } ^ { \mathrm { h } } \right) ^ { \vert \pi \vert } \left( C _ { 1 } ^ { \mathrm { h } } L \right) ^ { q - \vert \pi \vert } \prod _ { v \in \pi } \vert v \vert ! .
$$

Grouping the partitions according to $r = | \pi |$ , using $A _ { r } \leq r ! C _ { \sigma } ^ { r - 1 }$ , and applying Lemma 21, we get

$$
\begin{array} { l } { { \displaystyle | D ^ { u } \mathcal { R } _ { \Phi ^ { ( L ) } } ( x ) | \le 3 q ! \prod _ { j \in u } \beta _ { j } ( \delta , n _ { 0 } ) \sum _ { r = 1 } ^ { q } { \binom { q - 1 } { r - 1 } } C _ { \sigma } ^ { r - 1 } \left( C _ { 0 } ^ { \mathrm { h } } \right) ^ { r } \left( C _ { 1 } ^ { \mathrm { h } } L \right) ^ { q - r } } } \\ { { \displaystyle \qquad = 3 C _ { 0 } ^ { \mathrm { h } } q ! \left( C _ { 1 } ^ { \mathrm { h } } L + C _ { \sigma } C _ { 0 } ^ { \mathrm { h } } \right) ^ { q - 1 } \prod _ { j \in u } \beta _ { j } ( \delta , n _ { 0 } ) . } } \end{array}
$$

Since $L \geq 2$ , we have $C _ { 1 } ^ { \mathrm { h } } L + C _ { \sigma } C _ { 0 } ^ { \mathrm { h } } \leq \left( C _ { 1 } ^ { \mathrm { h } } + C _ { \sigma } C _ { 0 } ^ { \mathrm { h } } \right) L$ . Thus,

$$
| D ^ { u } \mathcal { R } _ { \Phi ^ { ( L ) } } ( x ) | \leq 3 C _ { 0 } ^ { \mathrm { h } } q ! \left[ \left( C _ { 1 } ^ { \mathrm { h } } + C _ { \sigma } C _ { 0 } ^ { \mathrm { h } } \right) L \right] ^ { q - 1 } \prod _ { j \in u } \beta _ { j } ( \delta , n _ { 0 } ) .
$$

Finally, we replace $\delta = \eta / 2$ by η in the β-factor. Since $n _ { 0 } \geq 1$ and $\eta \in ( 0 , 1 )$

$$
\log \left( \frac { 8 n _ { 0 } } { \eta } \right) = \log \left( \frac { 4 n _ { 0 } } { \eta } \right) + \log 2 \le 2 \log \left( \frac { 4 n _ { 0 } } { \eta } \right) .
$$

Define $f ( t ) : = n + 2 { \sqrt { n t } } + 2 t$ . Since f is increasing and $f ( 2 t ) \leq 2 f ( t )$ for $t \geq 0$ , we obtain $\beta _ { j } ( \delta , n _ { 0 } ) ^ { 2 } \leq 2 \beta _ { j } ( \eta , n _ { 0 } ) ^ { 2 }$ . Hence, $\beta _ { j } ( \delta , n _ { 0 } ) \leq \sqrt { 2 } \beta _ { j } ( \eta , n _ { 0 } )$ and

$$
\prod _ { j \in u } \beta _ { j } ( \delta , n _ { 0 } ) \leq 2 ^ { q / 2 } \prod _ { j \in u } \beta _ { j } ( \eta , n _ { 0 } ) .
$$

Since $2 ^ { q / 2 } = \sqrt { 2 } ( \sqrt { 2 } ) ^ { q - 1 }$ , we may take, for example, $C _ { 0 } : = 3 \sqrt { 2 } C _ { 0 } ^ { \mathrm { h } }$ and $C _ { 1 } : = \sqrt { 2 } \left( C _ { 1 } ^ { \mathrm { h } } + C _ { \sigma } C _ { 0 } ^ { \mathrm { h } } \right)$ Then

$$
| D ^ { u } \mathcal { R } _ { \Phi ^ { ( L ) } } ( x ) | \leq C _ { 0 } q ! ( C _ { 1 } L ) ^ { q - 1 } \prod _ { j \in u } \beta _ { j } ( \eta , n _ { 0 } ) .
$$

This estimate holds simultaneously for all non-empty $u \subseteq [ n _ { 0 } ]$ and all $x \in [ 0 , 1 ] ^ { n _ { 0 } }$ on the event $E _ { \mathrm { h } } \cap E _ { \mathrm { o u t } }$ , which has probability at least $1 - \eta$ . This proves the proposition. □

## 4. Consequences for quasi-Monte Carlo methods and Lipschitz continuity

The estimate in Proposition 13 is uniform in the input and has the product-and-order-dependent form that is commonly used in the analysis of quasi-Monte Carlo methods. In this section we first record the resulting weighted Sobolev and quadrature estimates. We then derive a high-probability bound for the Lipschitz constant and compare its dimension dependence with the estimates for random ReLU networks in [5]. Throughout the section, the network has scalar output and satisfies the assumptions of Proposition 13.

For every non-empty $u \subseteq [ n _ { 0 } ]$ , define

$$
B _ { u } ( \eta ) : = C _ { 0 } | u | ! ( C _ { 1 } L ) ^ { | u | - 1 } \prod _ { j \in u } \beta _ { j } ( \eta , n _ { 0 } ) .
$$

The conclusion of Proposition 13 can then be written without any ambiguity concerning the simultaneous quantifiers as

$$
\mathbb { P } \left( \operatorname* { m a x } _ { \substack { \emptyset \neq = [ n _ { 0 } ] } x \in [ 0 , 1 ] ^ { n _ { 0 } } } \frac { | D ^ { u } \mathcal { R } _ { \Phi ^ { ( L ) } } ( x ) | } { B _ { u } ( \eta ) } \leq 1 \right) \geq 1 - \eta .\tag{4}
$$

All estimates below are consequences of the event in (4). Let $s : = n _ { 0 }$ , and let $\gamma = ( \gamma _ { u } ) _ { \emptyset \neq u \subseteq [ s ] }$ be a family of positive weights. For a suficiently smooth function $f : [ 0 , 1 ] ^ { s }  \mathbb { R }$ , define

$$
| f | _ { \mathcal { H } _ { s , \gamma } } ^ { 2 } : = \sum _ { \emptyset \neq u \subseteq [ s ] } \frac { 1 } { \gamma _ { u } } \int _ { [ 0 , 1 ] ^ { s } } | D ^ { u } f ( x ) | ^ { 2 } d x .\tag{5}
$$

This is a first-order Sobolev seminorm with dominating mixed smoothness. The same argument also applies to the usual anchored and unanchored variants, since the lower-dimensional averaging operators in those norms are bounded by the corresponding uniform derivative bounds.

Corollary 14. Under the assumptions of Proposition 13, for every family of positive weights $\gamma$ we have

$$
\mathbb { P } ( | \mathcal { R } _ { \Phi ^ { ( L ) } } | _ { \mathcal { H } _ { n _ { 0 } , \gamma } } \le ( \sum _ { \emptyset \neq u \subseteq [ n _ { 0 } ] } \frac { B _ { u } ( \eta ) ^ { 2 } } { \gamma _ { u } } ) ^ { 1 / 2 } ) \ge 1 - \eta .
$$

Moreover, the same probability event works simultaneously for every deterministic choice of the positive weights γ.

Proof. On the event in (4), for every non-empty $u \subseteq [ n _ { 0 } ]$

$$
\int _ { [ 0 , 1 ] ^ { n _ { 0 } } } | D ^ { u } \mathcal { R } _ { \Phi ^ { ( L ) } } ( x ) | ^ { 2 } d x \leq B _ { u } ( \eta ) ^ { 2 } ,
$$

because $[ 0 , 1 ] ^ { n _ { 0 } }$ has unit volume. Substitution into (5) proves the estimate. Since the event in (4) does not depend on $\gamma ,$ the estimate holds there for every positive weight family. □

For $s = n _ { 0 }$ , write

$$
I _ { s } ( f ) : = \int _ { [ 0 , 1 ] ^ { s } } f ( x ) d x .
$$

Let $N \geq 2$ , let $z \in \mathbb { Z } ^ { s }$ have components coprime to $N$ , and let $\pmb { \Delta }$ be uniformly distributed on $[ 0 , 1 ] ^ { s }$ The randomly shifted rank-one lattice rule is

$$
Q _ { N , z , \Delta } ( f ) : = \frac { 1 } { N } \sum _ { { k = 0 } \atop { 2 0 } } ^ { N - 1 } f \left( \left\{ \frac { k z } { N } + \Delta \right\} \right) ,
$$

where braces denote the componentwise fractional part. The shift is taken independently of the random network parameters.

Corollary 15 (Randomly shifted lattice rule). Assume the hypotheses of Proposition 13, let N be a power of a prime, and let $\lambda \in ( 1 / 2 , 1 )$ . Define

$$
\varrho _ { 1 } ( \lambda ) : = { \frac { 2 \zeta ( 2 \lambda ) } { ( 2 \pi ) ^ { 2 \lambda } } } , \qquad c _ { \lambda } : = 2 ^ { \lambda } \varrho _ { 1 } ( \lambda ) ,
$$

and choose the product-and-order-dependent weights

$$
\gamma _ { u } ^ { \mathrm { N N } } : = \left( \frac { B _ { u } ( \eta ) ^ { 2 } } { c _ { \lambda } ^ { | u | } } \right) ^ { 1 / ( 1 + \lambda ) } \quad ( \emptyset \neq u \subseteq [ s ] ) , \qquad \gamma _ { \emptyset } ^ { \mathrm { N N } } : = 1 .\tag{6}
$$

Then a generating vector z can be constructed by a component-by-component algorithm such that, with probability at least 1 − η over the random network parameters,

$$
\left( \mathbb { E } _ { \Delta } \left| I _ { s } ( \mathcal { R } _ { \Phi ^ { ( L ) } } ) - Q _ { N , z , \Delta } ( \mathcal { R } _ { \Phi ^ { ( L ) } } ) \right| ^ { 2 } \right) ^ { 1 / 2 } \leq \left( \frac { 2 } { N } \right) ^ { 1 / ( 2 \lambda ) } \left( \sum _ { \theta \not = u \subseteq [ s ] } B _ { u } ( \eta ) ^ { \frac { 2 \lambda } { 1 + \lambda } } c _ { \lambda } ^ { \frac { | u | } { 1 + \lambda } } \right) ^ { \frac { 1 + \lambda } { 2 \lambda } } .\tag{7}
$$

Thus the displayed quantity is a root-mean-square error conditional on a network realization in the event (4); the only expectation in (7) is over the independent random shift.

Proof. Work on the derivative event in (4). Corollary 14 gives

$$
\left| \mathcal { R } _ { \Phi ^ { ( L ) } } \right| _ { \mathcal { H } _ { s , \gamma } } ^ { 2 } \le \sum _ { \emptyset \ne u \subseteq [ s ] } \frac { B _ { u } ( \eta ) ^ { 2 } } { \gamma _ { u } } .\tag{8}
$$

The nonconstant part of the standard unanchored first-order Sobolev norm is bounded by this full $L _ { 2 }$ mixed-derivative seminorm. Indeed, Jensen’s inequality yields, for every nonempty $u \subseteq [ s ]$ 2

$$
\int _ { [ 0 , 1 ] ^ { | u | } } \left| \int _ { [ 0 , 1 ] ^ { s - | u | } } D ^ { u } f ( x ) d x _ { - u } \right| ^ { 2 } d x _ { u } \leq \int _ { [ 0 , 1 ] ^ { s } } | D ^ { u } f ( x ) | ^ { 2 } d x .
$$

The standard randomly shifted lattice-rule estimate in this non-periodic space, together with the CBC construction for prime-power N, states [19, Section 2.1(a)] that

$$
\mathrm { r . \ m . s . } _ { \Delta } e _ { N } ^ { \mathrm { w o r } } \leq \left( \frac { 2 } { N } \sum _ { \substack { \emptyset \neq u \subseteq [ s ] } } \gamma _ { u } ^ { \lambda } c _ { \lambda } ^ { | u | } \right) ^ { 1 / ( 2 \lambda ) } .
$$

Multiplying this estimate by the square root of the right-hand side of (8) and substituting (6) makes both sums equal to

$$
\sum _ { \varnothing \ne u \subseteq [ s ] } B _ { u } ( \eta ) ^ { \frac { 2 \lambda } { 1 + \lambda } } c _ { \lambda } ^ { \frac { | u | } { 1 + \lambda } } .
$$

Thus the weights balance the Sobolev-norm and worst-case-error sums and give (7). Since the derivative event has probability at least $1 - \eta$ , the asserted conditional RMS statement follows. □

The product-and-order-dependent structure can be written as

$$
b _ { j } ^ { \mathrm { N N } } : = C _ { 1 } L \beta _ { j } ( \eta , n _ { 0 } ) , \qquad B _ { u } ( \eta ) = \frac { C _ { 0 } } { C _ { 1 } L } | u | ! \prod _ { j \in u } b _ { j } ^ { \mathrm { N N } } .
$$

The following consequence is genuinely uniform in the input dimension.

Corollary 16 (Dimension-independent RMS rate). Fix $\eta \in ( 0 , 1 )$ and $L \geq 2$ . Suppose there is an infinite sequence $( \bar { b } _ { j } ) _ { j \geq 1 }$ , independent of s, such that for every dimension under consideration

$$
b _ { j } ^ { \mathrm { N N } } \leq \bar { b } _ { j } \quad ( j \leq s ) , \qquad \sum _ { j \geq 1 } \bar { b } _ { j } ^ { p ^ { * } } < \infty
$$

for some $p ^ { * } \in ( 0 , 1 )$ . For any $\varepsilon \in ( 0 , 1 / 2 )$ , choose

$$
\lambda = \left\{ \begin{array} { l l } { \displaystyle \frac { 1 } { 2 - 2 \varepsilon } , } & { p ^ { * } \in ( 0 , 2 / 3 ] , } \\ { \displaystyle \frac { p ^ { * } } { 2 - p ^ { * } } , } & { p ^ { * } \in ( 2 / 3 , 1 ) . } \end{array} \right.\tag{9}
$$

Then the CBC lattice rules of Corollary 15 satisfy, with probability at least $1 - \eta$ over the network parameters,

$$
\mathrm { r . \ m . s . } _ { \Delta } | I _ { s } ( { \mathcal R } _ { \Phi ^ { ( L ) } } ) - Q _ { N , z , \Delta } ( { \mathcal R } _ { \Phi ^ { ( L ) } } ) | = { \mathcal O } ( N ^ { - r } ) , \qquad r = \operatorname* { m i n } \left( 1 - \varepsilon , \frac { 1 } { p ^ { * } } - \frac { 1 } { 2 } \right) .
$$

The implied constant may depend on $C _ { 0 } / ( C _ { 1 } L ) , p ^ { * } , \varepsilon ,$ and the fixed majorant $( \bar { b } _ { j } ) _ { j \geq 1 }$ (and hence on η and L through the chosen majorant), but it is independent of $s , N$ , the shift, and the particular network realization in the derivative event.

Proof. Set $q : = 2 \lambda / ( 1 + \lambda ) < 1$ . The choices in (9) ensure $q \geq p ^ { * }$ . Hence $\begin{array} { r } { \sum _ { j \geq 1 } \bar { b } _ { j } ^ { q } < \infty \colon } \end{array}$ only finitely many $\bar { b } _ { j }$ exceed one, and on the remaining indices $\bar { b } _ { j } ^ { q } \leq \bar { b } _ { j } ^ { p ^ { * } }$ . With

$$
a _ { j } : = c _ { \lambda } ^ { 1 / ( 1 + \lambda ) } \bar { b } _ { j } ^ { q } , \qquad D : = \frac { C _ { 0 } } { C _ { 1 } L } ,
$$

the sum in (7) is bounded, uniformly in s, by

$$
D ^ { q } \sum _ { k \geq 1 } ( k ! ) ^ { q } \sum _ { \stackrel { u \subset \mathbb { N } } { | u | = k } } \prod _ { j \in u } a _ { j } \leq D ^ { q } \sum _ { k \geq 1 } ( k ! ) ^ { q - 1 } \left( \sum _ { j \geq 1 } a _ { j } \right) ^ { k } < \infty .
$$

The first inequality uses $\begin{array} { r } { \sum _ { | u | = k } \prod _ { j \in u } a _ { j } \leq k ! ^ { - 1 } ( \sum _ { j } a _ { j } ) ^ { k } } \end{array}$ , and the final series converges by the ratio test because $q < 1$ . Corollary 15 therefore gives the stronger dimension-independent rate $N ^ { - 1 / ( 2 \lambda ) }$ In the two cases in (9), the exponent $1 / ( 2 \lambda )$ equals respectively $1 - \varepsilon$ and $1 / p ^ { * } - 1 / 2$ . In either case, $r \leq 1 / ( 2 \lambda )$ , so the stronger estimate implies $\mathcal { O } ( N ^ { - r } )$ . □

We emphasize three limitations of the QMC interpretation. First, Proposition 13 concerns the random network at initialization. It does not imply that the same event remains valid after unrestricted training. A complete generalization theorem therefore requires either parameter regularization, as in [20], or a stability estimate showing that the training trajectory remains in a region where the derivative bounds persist. Second, isotropic Xavier initialization makes the quantities $\beta _ { j } ( \eta , n _ { 0 } )$ comparable for all $j$ . Hence, the summability assumptions needed for dimension independent QMC constants are not uniform as $n _ { 0 } \to \infty$ . Such estimates require an anisotropic initialization or regularization that enforces coordinate decay. Third, the present theorem controls only square-free mixed derivatives. This is suficient for first-order spaces of dominating mixed smoothness, but not for higher-order periodic Korobov spaces, which require repeated derivatives and a periodic architecture.

We next turn to the first-order consequence of (4). For $p \in [ 1 , \infty ]$ , let $p ^ { \prime }$ be its H¨older conjugate and define

$$
\mathrm { L i p } _ { \ell ^ { p } } ( f ; [ 0 , 1 ] ^ { n _ { 0 } } ) : = \operatorname* { s u p } _ { \stackrel { x , y \in [ 0 , 1 ] ^ { n _ { 0 } } } { x \neq y } } { \frac { | f ( x ) - f ( y ) | } { \| x - y \| _ { \ell ^ { p } } } } .
$$

Corollary 17 (High-probability Lipschitz estimate). Under the assumptions of Proposition 13, for every $p \in [ 1 , \infty ]$ ，

$$
\mathbb { P } \left( \operatorname* { s u p } _ { x \in [ 0 , 1 ] ^ { n _ { 0 } } } \| \nabla \mathcal { R } _ { \Phi ^ { ( L ) } } ( x ) \| _ { \ell ^ { p ^ { \prime } } } \leq C _ { 0 } \| \beta ( \eta , n _ { 0 } ) \| _ { \ell ^ { p ^ { \prime } } } \right) \geq 1 - \eta .\tag{10}
$$

On the same event,

$$
\begin{array} { r } { \mathrm { L i p } _ { \ell ^ { p } } \left( \mathcal { R } _ { \Phi ^ { ( L ) } } ; [ 0 , 1 ] ^ { n _ { 0 } } \right) \leq C _ { 0 } \left\| \beta ( \eta , n _ { 0 } ) \right\| _ { \ell ^ { p ^ { \prime } } } . } \end{array}\tag{11}
$$

In particular, after increasing the universal constant in the width condition $i f$ necessary, $\beta _ { j } ( \eta , n _ { 0 } ) \leq 3$ for all $j \in [ n _ { 0 } ]$ , and hence

$$
\mathbb { P } \left( \operatorname* { s u p } _ { x \in [ 0 , 1 ] ^ { n _ { 0 } } } \| \nabla { \mathcal { R } } _ { \Phi ^ { ( L ) } } ( x ) \| _ { \ell ^ { p ^ { \prime } } } \leq 3 C _ { 0 } n _ { 0 } ^ { 1 - 1 / p } \right) \geq 1 - \eta .
$$

Consequently, with probability at least $1 - \eta ,$ we have $\mathrm { L i p } _ { \ell ^ { p } } \left( \mathcal { R } _ { \Phi ^ { ( L ) } } ; [ 0 , 1 ] ^ { n _ { 0 } } \right) \le 3 C _ { 0 } n _ { 0 } ^ { 1 - 1 / p }$

Proof. Taking $u = \{ j \}$ in (4) gives $\begin{array} { r } { \operatorname* { s u p } _ { x \in [ 0 , 1 ] ^ { n _ { 0 } } } | \partial _ { j } \mathcal { R } _ { \Phi ^ { ( L ) } } ( x ) | \leq C _ { 0 } \beta _ { j } ( \eta , n _ { 0 } ) } \end{array}$ for every $j \in [ n _ { 0 } ]$ on an event of probability at least $1 - \eta .$ . Taking the ℓ<sup>p</sup> -norm over the coordinates proves (10). Since the cube is convex, the fundamental theorem of calculus along the line segment from x to y, followed by H¨older’s inequality, gives

$$
| \mathcal { R } _ { \Phi ^ { ( L ) } } ( x ) - \mathcal { R } _ { \Phi ^ { ( L ) } } ( y ) | \leq \operatorname* { s u p } _ { z \in [ 0 , 1 ] ^ { n _ { 0 } } } \Vert \nabla \mathcal { R } _ { \Phi ^ { ( L ) } } ( z ) \Vert _ { \ell ^ { p ^ { \prime } } } \Vert x - y \Vert _ { \ell ^ { p } } .
$$

This proves (11). The last claim follows from $\| \beta ( \eta , n _ { 0 } ) \| _ { \ell ^ { p ^ { \prime } } } \leq 3 n _ { 0 } ^ { 1 / p ^ { \prime } } = 3 n _ { 0 } ^ { 1 - 1 / p } .$

It is instructive to compare Corollary 17 with the results of Dirksen, Finke, Geuchen, St¨oger, and Voigtlaender [5]. They consider random ReLU networks $\Psi : \mathbb { R } ^ { n _ { 0 } }  \mathbb { R }$ with $L$ hidden layers of width n and a variant of He initialization. However, we need to be careful if we compare those results with ours. Their estimates are global estimates on $\mathbb { R } ^ { d }$ , the activation functions are diferent, and the output normalizations are not the same: the final-layer weights in [5] are standard Gaussian, while our scalar Xavier readout has variance $2 / ( n + 1 )$ . Moreover, our result is only an upper bound, whereas [5] also establishes lower bounds.

## Acknowledgements

Funding was received from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (Grant agreement No. 101125225). Funding was also received from the OeAD under the Marietta Blau-Grant.

## Declaration on generative AI assistance

Generative AI tools were used for research assistance. In particular, AI assistance contributed to the formulation of mathematical ideas used in the paper, including Proposition 5 and Proposition 10. AI tools were also used for literature search and for drafting and revising parts of the exposition. All definitions, theorem statements, proofs, calculations, and references included in the manuscript were independently checked, revised, and approved by the authors. The authors assume responsibility for all content.

## Appendix A. Auxiliary results

The following form of the multivariate Fa\`a di Bruno formula follows from [16, Propositions 1 and 2].

Theorem 18. Suppose $\sigma : \mathbb { R }  \mathbb { R }$ and $g : \mathbb { R } ^ { s }  \mathbb { R }$ are two smooth functions. $I f \nu \in \mathbb { N } _ { 0 } ^ { s }$ , then

$$
\partial ^ { \nu } \left( \sigma \circ g \right) ( x ) = \sum _ { \pi \in \Pi ( [ | \nu | ] ) } \sigma ^ { ( | \pi | ) } \left( g ( x ) \right) \prod _ { v \in \pi } \partial ^ { \mu ( \nu , v ) } g ( x ) ,
$$

holds for all $x \in \mathbb { R } ^ { s }$ , where $\mu ( \nu , v )$ is defined in (2).

If $u = \{ j _ { 1 } < \cdot \cdot \cdot < j _ { q } \} \subseteq [ n _ { 0 } ]$ and $\nu = \mathbf { 1 } _ { u }$ , then the order-preserving bijection $r \mapsto j _ { r }$ from $[ q ]$ onto u induces a bijection between $\Pi ( [ q ] )$ and $\Pi ( u )$ . Under this identification, $\partial ^ { \mu ( \nu , v ) } = D ^ { v }$ . Hence the Fa\`a di Bruno formula takes the particularly simple square-free form

$$
D ^ { u } ( \sigma \circ g ) ( x ) = \sum _ { \pi \in \Pi ( u ) } \sigma ^ { ( | \pi | ) } ( g ( x ) ) \prod _ { v \in \pi } D ^ { v } g ( x ) .
$$

Lemma 19. Let $k , n \in \mathbb { N }$ . If $a ^ { ( i ) } \in \mathbb { R } ^ { n }$ for all $i \in [ k ]$ then

$$
\left\| \bigcirc _ { i = 1 } ^ { k } a ^ { ( i ) } \right\| _ { 2 } \leq \prod _ { i = 1 } ^ { k } \left\| a ^ { ( i ) } \right\| _ { 2 }
$$

holds.

Proof. We have

$$
\left\| \sum _ { i = 1 } ^ { k } a ^ { ( i ) } \right\| _ { 2 } ^ { 2 } = \sum _ { j = 1 } ^ { n } \prod _ { i = 1 } ^ { k } \left| a _ { j } ^ { ( i ) } \right| ^ { 2 } \leq \sum _ { j = 1 } ^ { n } \left( \left| a _ { j } ^ { ( 1 ) } \right| ^ { 2 } \prod _ { i = 2 } ^ { k } \left\| a ^ { ( i ) } \right\| _ { 2 } ^ { 2 } \right) = \left( \prod _ { i = 2 } ^ { k } \left\| a ^ { ( i ) } \right\| _ { 2 } ^ { 2 } \right) \sum _ { j = 1 } ^ { n } \left| a _ { j } ^ { ( 1 ) } \right| ^ { 2 } = \prod _ { i = 1 } ^ { k } \left\| a ^ { ( i ) } \right\| _ { 2 } ^ { 2 } .
$$

Taking square roots gives the claim.

Lemma 20. For all $r \in \mathbb N$ we $h a v e \left| \operatorname { t a n h } ^ { ( r ) } ( x ) \right| \leq r ! \left( { 1 6 } / { \pi ^ { 2 } } \right) ^ { r - 1 }$ for all $x \in \mathbb { R }$

Proof. For every $x \in \mathbb { R }$ we define the closed disk $D ( x ) : = \{ z \in \mathbb { C } : | x - z | \leq \pi / 4 \}$ and we have

$$
D ( x ) \subseteq M : = \left\{ z \in \mathbb { C } : | \operatorname { I m } ( z ) | < { \frac { \pi } { 2 } } \right\}
$$

If a, $b \in \mathbb { R }$ and $a + i b = z \in M$ then

$$
{ \begin{array} { r l } & { | \sinh ( z ) | ^ { 2 } = ( \sinh ( a ) \cos ( b ) ) ^ { 2 } + ( \cosh ( a ) \sin ( b ) ) ^ { 2 } = ( \sinh ( a ) ) ^ { 2 } \left( 1 - ( \sin ( b ) ) ^ { 2 } \right) + ( \cosh ( a ) \sin ( b ) ) ^ { 2 } } \\ & { \qquad = ( \sinh ( a ) ) ^ { 2 } + ( \sin ( b ) ) ^ { 2 } \left( ( \cosh ( a ) ) ^ { 2 } - ( \sinh ( a ) ) ^ { 2 } \right) = ( \sinh ( a ) ) ^ { 2 } + ( \sin ( b ) ) ^ { 2 } } \end{array} }
$$

and

$$
\begin{array} { r l r } & { } & { | \cosh ( z ) | ^ { 2 } = ( \cosh ( a ) \cos ( b ) ) ^ { 2 } + ( \sinh ( a ) \sin ( b ) ) ^ { 2 } = ( \cosh ( a ) ) ^ { 2 } \left( 1 - ( \sin ( b ) ) ^ { 2 } \right) + ( \sinh ( a ) \sin ( b ) ) ^ { 2 } } \\ & { } & { = ( \cosh ( a ) ) ^ { 2 } - ( \sin ( b ) ) ^ { 2 } \left( ( \cosh ( a ) ) ^ { 2 } - ( \sinh ( a ) ) ^ { 2 } \right) = ( \cosh ( a ) ) ^ { 2 } - ( \sin ( b ) ) ^ { 2 } . \qquad } \end{array}
$$

If $| b | \leq \pi / 4$ then $\left( \sin ( b ) \right) ^ { 2 } \leq 1 / 2$ and

$$
\begin{array} { r l } & { | \sinh ( z ) | ^ { 2 } = ( \sinh ( a ) ) ^ { 2 } + ( \sin ( b ) ) ^ { 2 } \leq ( \sinh ( a ) ) ^ { 2 } + 1 - ( \sin ( b ) ) ^ { 2 } } \\ & { \qquad = ( \cosh ( a ) ) ^ { 2 } - ( \sin ( b ) ) ^ { 2 } = | \cosh ( z ) | ^ { 2 } . } \end{array}
$$

In this case we obtain $| \mathrm { t a n h } ( z ) | ^ { 2 } = | \sinh ( z ) | ^ { 2 } / | \cosh ( z ) | ^ { 2 } \leq 1$ . By Cauchy’s estimate we have $\left| \operatorname { t a n h } ^ { ( r ) } ( x ) \right| \leq r ! ( 4 / \pi ) ^ { r }$ for all $x \in \mathbb { R }$ and all $r \in \mathbb N$ . Since the holomorphic extension agrees with

the real-valued function on the real axis, its complex derivatives restricted to R coincide with the corresponding real derivatives. For $r = 1$ we have $\left. \operatorname { t a n h } ^ { \prime } ( x ) \right. \leq 1 = 1 ! \left( { 1 6 } / { \pi ^ { 2 } } \right) ^ { 0 }$ . For $r \geq 2$ we have $2 r - 2 \geq r$ and since $4 / \pi > 1$ we obtain

$$
\left| \operatorname { t a n h } ^ { ( r ) } ( x ) \right| \leq r ! \left( { \frac { 4 } { \pi } } \right) ^ { r } \leq r ! \left( { \frac { 4 } { \pi } } \right) ^ { 2 ( r - 1 ) } = r ! \left( { \frac { 1 6 } { \pi ^ { 2 } } } \right) ^ { r - 1 } .
$$

Lemma 21. For all $r \in \mathbb N$ and all finite sets $u \neq \emptyset$ we have

$$
\sum _ { \pi \in \Pi _ { r } ( u ) } \prod _ { v \in \pi } | v | ! = { \frac { | u | ! } { r ! } } { \binom { | u | - 1 } { r - 1 } } .
$$

Proof. We prove the identity by strong induction on $m : = | { \boldsymbol { u } } | .$ , simultaneously for every $r \in \mathbb N .$ . If $m = 1$ , then for $r = 1$ both sides equal one, whereas for $r > 1$ both sides vanish.

Suppose that $n \geq 1$ and that the identity holds for every nonempty finite set of size at most n and every block number in N. Let u be a finite set with $\lvert u \rvert = n + 1$ , and fix $r \in \mathbb { N }$ . If $r > n + 1$ then $\Pi _ { r } ( u ) = \emptyset$ and $\binom { n } { r - 1 } = 0$ , so both sides vanish. $\textit { I f } r = 1$ , then $\Pi _ { 1 } ( u ) = \{ \{ u \} \}$ , and both sides equal $( n + 1 ) !$ !. It remains to consider $2 \leq r \leq n + 1$

Fix $x \in u$ and set $v : = u \setminus \{ x \}$ . Define

$$
A : = \{ \pi \in \Pi _ { r } ( u ) : \{ x \} \in \pi \} , \qquad B : = \{ \pi \in \Pi _ { r } ( u ) : \{ x \} \notin \pi \} .
$$

Then A and B are disjoint and $\Pi _ { r } ( u ) = A \cup B$ . Define $f : A \to \Pi _ { r - 1 } ( v )$ by $f ( \pi ) : = \pi \setminus \{ \{ x \} \}$ . This is a bijection with inverse $f ^ { - 1 } ( \rho ) : = \rho \cup \{ \{ x \} \}$ . The induction hypothesis, applied to the set v and the block number $r - 1$ , gives

$$
\sum _ { \pi \in A } \prod _ { w \in \pi } | w | ! = { \frac { n ! } { ( r - 1 ) ! } } { \binom { n - 1 } { r - 2 } } .
$$

Next, define

$$
C : = \{ ( \rho , y ) : \rho \in \Pi _ { r } ( v ) , \ y \in \rho \} .
$$

For $\pi \in B$ , let $h ( \pi ) : = w$ , where w is the unique block of π containing x, and define $g : B  C$ by

$$
g ( \pi ) : = \left( { \bigl ( } \pi \setminus \{ h ( \pi ) \} { \bigr ) } \cup \{ h ( \pi ) \setminus \{ x \}  , h ( \pi ) \setminus \{ x \} \right) .
$$

The map g is a bijection with inverse

$$
g ^ { - 1 } ( \rho , y ) : = ( \rho \setminus \{ y \} ) \cup \{ y \cup \{ x \} \} .
$$

The induction hypothesis, now applied to the set v and the block number $^ { r , }$ yields

$$
\sum _ { \pi \in B } \prod _ { w \in \pi } | w | ! = \sum _ { \rho \in \Pi _ { r } ( v ) } \sum _ { y \in \rho } ( | y | + 1 ) ! \prod _ { w \in \rho } | w | ! = \sum _ { \rho \in \Pi _ { r } ( v ) } \left( \prod _ { w \in \rho } | w | ! \right) \sum _ { y \in \rho } ( | y | + 1 ) = ( n + r ) \frac { n ! } { r ! } \binom { n - 1 } { r - 1 } .
$$

Combining the preceding identities, we obtain

$$
\sum _ { \pi \in \Pi _ { r } ( u ) } \prod _ { w \in \pi } | w | ! = { \frac { n ! } { ( r - 1 ) ! } } { \binom { n - 1 } { r - 2 } } + ( n + r ) { \frac { n ! } { r ! } } { \binom { n - 1 } { r - 1 } } = { \frac { ( n + 1 ) ! } { r ! } } { \binom { n } { r - 1 } } = { \frac { | u | ! } { r ! } } { \binom { | u | - 1 } { r - 1 } } .
$$

This proves the identity for every $r \in \mathbb { N }$ and completes the strong induction.

Lemma 22. For fixed $n _ { 0 } , L , \beta , \alpha , \gamma .$ , the set $\Theta _ { L } ^ { \mathrm { t a n g } } ( n _ { 0 } , \beta , \alpha , \gamma )$ is closed in $\Theta _ { L }$ . In particular, if $\Phi : \Omega  \Theta _ { L }$ is measurable, then $\{ \Phi \in \Theta _ { L } ^ { \mathrm { t a n g } } ( n _ { 0 } , \beta , \alpha , \gamma ) \}$ is an event.

Proof. For $\ell \in [ L ]$ and $\emptyset \neq u \subseteq [ n _ { 0 } ]$ , define

$$
F _ { \ell , u } ( \varphi , x ) : = \left\| W ^ { ( \ell ) } G _ { u } ^ { ( \ell ) } ( x , \varphi ^ { ( \ell - 1 ) } ) \right\| _ { 2 } - \alpha _ { \ell } \left\| G _ { u } ^ { ( \ell ) } ( x , \varphi ^ { ( \ell - 1 ) } ) \right\| _ { 2 } - \gamma _ { \ell } \prod _ { j \in u } \beta _ { j } .
$$

The map $( \varphi , x ) \mapsto F _ { \ell , u } ( \varphi , x )$ is continuous. Since $[ 0 , 1 ] ^ { n _ { 0 } }$ is compact, the map

$$
\varphi \longmapsto \operatorname* { m a x } _ { x \in [ 0 , 1 ] ^ { n _ { 0 } } } F _ { \ell , u } ( \varphi , x )
$$

is continuous. Hence,

$$
\left\{ \varphi : \operatorname* { m a x } _ { x \in [ 0 , 1 ] ^ { n _ { 0 } } } F _ { \ell , u } ( \varphi , x ) \leq 0 \right\}
$$

is closed. Taking the finite intersection over ℓ and non-empty $u \subseteq [ n _ { 0 } ]$ proves the claim. □

Proposition 23. Let $N \in  { \mathbb { N } }$ , let $a _ { 0 } , \dots , a _ { N } \in \mathbb { R }$ , and let $f _ { 0 } , \dots , f _ { N - 1 } , g _ { 0 } , \dots , g _ { N - 1 } \in [ 0 , \infty )$ . If $a _ { n + 1 } \leq f _ { n } a _ { n } + g _ { n }$ holds for every $n = 0 , \ldots , N - 1$ , then

$$
a _ { n } \leq \left( \prod _ { p = 0 } ^ { n - 1 } f _ { p } \right) a _ { 0 } + \sum _ { m = 0 } ^ { n - 1 } \left( \prod _ { p = m + 1 } ^ { n - 1 } f _ { p } \right) g _ { m }
$$

for every $n = 0 , \ldots , N$

Proof. The assertion follows by induction over n. The case $n = 0$ is immediate. If the estimate holds for n, then multiplication by $f _ { n } \geq 0$ and the recurrence inequality give

$$
a _ { n + 1 } \leq f _ { n } a _ { n } + g _ { n } \leq \left( \prod _ { p = 0 } ^ { n } f _ { p } \right) a _ { 0 } + \sum _ { m = 0 } ^ { n - 1 } \left( \prod _ { p = m + 1 } ^ { n } f _ { p } \right) g _ { m } + g _ { n } ,
$$

which is the desired formula for $n + 1$ , using the convention for empty products.

We use the following singular-value estimate from [32, Corollary 5.35].

Proposition 24. If A is an $m \times n$ random matrix whose entries $A _ { i , j }$ are independent standard normal random variables, then for every $t \geq 0$ we have

$$
\sqrt { m } - \sqrt { n } - t \leq s _ { \operatorname* { m i n } } ( A ) \leq s _ { \operatorname* { m a x } } ( A ) \leq \sqrt { m } + \sqrt { n } + t
$$

with probability at least $1 - 2 \exp ( - t ^ { 2 } / 2 )$ . Here $s _ { \mathrm { m i n } } ( A )$ and $s _ { \mathrm { m a x } } ( A )$ denote respectively the smallest and the largest singular value of A.

This means in particular that the matrix norm

$$
\| A \| _ { 2  2 } \leq { \sqrt { m } } + { \sqrt { n } } + t
$$

is satisfied with probability at least $1 - 2 \exp ( - t ^ { 2 } / 2 )$

We also use the following concentration inequality from [22, Lemma 1].

Lemma 25. Let $n \in \mathbb { N }$ and $X _ { 1 } , \ldots , X _ { n }$ be $i . i . d .$ standard normal random variables. If $a \in \mathbb { R } ^ { n }$ is a vector with non-negative entries, then

$$
\mathbb { P } \left( \sum _ { i = 1 } ^ { n } a _ { i } \left( X _ { i } ^ { 2 } - 1 \right) \geq 2 \| a \| _ { 2 } \sqrt { x } + 2 \| a \| _ { \infty } x \right) \leq \exp ( - x )
$$

holds for all $x > 0$

By [27, Lemma 26.8] we have

Lemma 26. Let X and Y be topological spaces with Y compact, and consider the product space $X \times Y . \ I f x \in X$ and U is an open set in the product topology containing $\{ x \} \times Y$ , then there exists an open subset V of X such that $\{ x \} \times Y \subseteq V \times Y \subseteq U \subseteq X \times Y$

As a consequence of the tube Lemma 26 we get the following result.

Corollary 27. If X and Y are topological spaces and Y is compact, then the projection $p : X \times Y $ X given by $p ( x , y ) = x$ is a closed map.

Proof. Let A be any closed set in $X \times Y$ and define $U : = ( X \times Y ) \setminus A$ . Suppose $x \in X \setminus p ( A )$ This means that for all $y \in Y$ we have $( x , y ) \in U$ and consequently, $\{ x \} \times Y \subseteq U$ . By the Tube Lemma 26 there exists an open subset V of X such that $\{ x \} \times Y \subseteq V \times Y \subseteq U \subseteq X \times Y$ . Since $\{ x \} \subseteq V$ and $V \cap p ( A ) = \emptyset$ we conclude that $p ( A )$ is a closed set in X. □

Lemma 28. Let S be a topological space, let $d \in \mathbb { N }$ , and let $C \subseteq [ 0 , 1 ] ^ { d }$ be non-empty and compact. Suppose that $\Gamma \subseteq S \times C$ is closed. For $s \in S$ , define $\Gamma _ { s } : = \{ x \in C : ( s , x ) \in \Gamma \}$ and $D : = \{ s \in S : \Gamma _ { s } \neq \emptyset \}$ . Then D is closed. Moreover, every $\Gamma _ { s } , s \in D$ , has a unique lexicographically smallest element, and the map $\lambda : D \to C$ given by $\lambda ( s ) : = \operatorname { l e x m i n } \Gamma _ { s }$ is Borel measurable.

Proof. Since C is compact and Γ is closed, the projection $D = \mathrm { p r o j } _ { S } ( \Gamma )$ is closed by Corollary 27. For $n \in  { \mathbb { N } } _ { 0 }$ , let

$$
\mathcal { T } _ { n } : = \left\{ \left[ \frac { r - 1 } { 2 ^ { n } } , \frac { r } { 2 ^ { n } } \right] : r = 1 , \ldots , 2 ^ { n } \right\} .
$$

If

$$
I = \left[ \frac { r - 1 } { 2 ^ { n } } , \frac { r } { 2 ^ { n } } \right] \in \mathcal { I } _ { n } ,
$$

define its left and right children by

$$
I ^ { \mathrm { L } } : = \left[ { \frac { r - 1 } { 2 ^ { n } } } , { \frac { 2 r - 1 } { 2 ^ { n + 1 } } } \right] \quad { \mathrm { a n d } } \quad I ^ { \mathrm { R } } : = \left[ { \frac { 2 r - 1 } { 2 ^ { n + 1 } } } , { \frac { r } { 2 ^ { n } } } \right] .
$$

We first record a measurability observation. Suppose that $1 \leq j \leq d$ , and that $J _ { i } : D  \mathbb { Z } _ { m _ { i } } ,$ for $i = 1 , \ldots , j$ are Borel measurable maps. Then the set

$$
E ( J _ { 1 } , \dots , J _ { j } ) : = \left\{ s \in D : \Gamma _ { s } \cap \left( J _ { 1 } ( s ) \times \dots \times J _ { j } ( s ) \times [ 0 , 1 ] ^ { d - j } \right) \neq \emptyset \right\}
$$

is Borel measurable. Indeed, each $J _ { i }$ has finite range. Hence,

$$
E ( J _ { 1 } , \ldots , J _ { j } ) = \bigcup _ { A _ { 1 } , \ldots , A _ { j } } \left[ \left( \bigcap _ { i = 1 } ^ { j } J _ { i } ^ { - 1 } ( \{ A _ { i } \} ) \right) \cap \left\{ s \in D : \Gamma _ { s } \cap \left( A _ { 1 } \times \cdots \times A _ { j } \times [ 0 , 1 ] ^ { d - j } \right) \neq \emptyset \right\} \right] ,
$$

where $A _ { i } \in \mathbb { Z } _ { m _ { i } }$ and the union is finite. For fixed $A _ { 1 } , \dotsc , A _ { j }$ , set $K : = C \cap \left( A _ { 1 } \times \cdots \times A _ { j } \times [ 0 , 1 ] ^ { d - j } \right)$ Then K is compact, and the corresponding hit set is $\mathrm { p r o j } _ { D } ( \Gamma \cap ( D \times \dot { K } ) )$ . Hence, it is closed by Corollary 27. Therefore, $E ( J _ { 1 } , \dots , J _ { j } )$ is Borel.

We now construct, successively for $j = 1 , \ldots , d ,$ , Borel measurable interval-valued maps $I _ { j , n } : D \to$ $\mathcal { I } _ { n }$ for $n \in  { \mathbb { N } } _ { 0 }$ , such that, for every $s \in D$ , we have $I _ { j , n + 1 } ( s ) \subseteq I _ { j , n } ( s )$ and diam $( I _ { j , n } ( s ) ) = 2 ^ { - n }$

Suppose first that the interval maps $I _ { i , n }$ have already been constructed for every $i < j$ and every $n \in  { \mathbb { N } } _ { 0 }$ , and define $a _ { i } ( s )$ as the unique element of $\cap _ { n = 0 } ^ { \infty } I _ { i , n } ( s )$ for all $i < j$ . For $j = 1$ , there are no preceding coordinates.

Set $I _ { j , 0 } ( s ) : = [ 0 , 1 ]$ for every $s \in D$ . Suppose $I _ { j , n }$ has been constructed. Let $L _ { j , n } ( s ) : = I _ { j , n } ( s ) ^ { \mathrm { L } }$ and $R _ { j , n } ( s ) : = I _ { j , n } ( s ) ^ { \mathrm { R } }$ . Both maps have finite range and are Borel measurable. Define

$$
H _ { j , n } : = { \Big \{ } s \in D : \exists x \in \Gamma _ { s } { \mathrm { ~ s u c h ~ t h a t ~ } } x _ { i } = a _ { i } ( s ) { \mathrm { ~ f o r ~ } } i < j { \mathrm { ~ a n d ~ } } x _ { j } \in L _ { j , n } ( s ) { \Big \} } .
$$

We claim that $H _ { j , n }$ is Borel. In fact,

$$
H _ { j , n } = \bigcap _ { N = 0 } ^ { \infty } E \left( I _ { 1 , N } , \ldots , I _ { j - 1 , N } , L _ { j , n } \right) ,
$$

where for $j = 1$ the preceding interval maps are omitted. To verify this equality, the inclusion from left to right is immediate. Conversely, suppose s belongs to every set on the right. Then, for every $N$ , the set

$$
K _ { N } : = \Gamma _ { s } \cap \left( I _ { 1 , N } ( s ) \times \cdots \times I _ { j - 1 , N } ( s ) \times L _ { j , n } ( s ) \times [ 0 , 1 ] ^ { d - j } \right)
$$

is non-empty and compact. The sequence $( K _ { N } ) _ { N \ge 0 }$ is decreasing. Hence compactness gives $\cap _ { N = 0 } ^ { \infty } K _ { N } \neq \emptyset$ . Every point in this intersection satisfies $x _ { i } = a _ { i } ( s )$ for $i < j$ and $x _ { j } \in L _ { j , n } ( s )$ . Thus, $s \in H _ { j , n }$ , proving the claim.

We now define

$$
I _ { j , n + 1 } ( s ) : = \left\{ { L _ { j , n } ( s ) , \mathrm { \quad } s \in H _ { j , n } , } \right.
$$

Since $H _ { j , n }$ is Borel and the two child maps are Borel, $I _ { j , n + 1 }$ is Borel measurable. For fixed $s \in D$ define $K _ { s } ^ { j - 1 } : = \{ x \in \Gamma _ { s } : x _ { i } = a _ { i } ( s )$ for $i < j \}$ . Inductively, this is a non-empty compact set. Hence the continuous coordinate projection $x \mapsto x _ { j }$ attains its minimum on $K _ { s } ^ { j - 1 }$ . Denote that minimum by $a _ { j } ( s ) : = \operatorname* { m i n } \{ x _ { j } : x \in K _ { s } ^ { j - 1 } \}$

We claim that $a _ { j } ( s ) \in I _ { j , n } ( s )$ holds for every n. This is clear for $n = 0$ . Suppose it holds for $n .$ . If the left child contains a point of $K _ { s } ^ { j - 1 }$ , then, since $a _ { j } ( s )$ is the minimum of the $j \mathrm { - t h }$ coordinates on $K _ { s } ^ { j - 1 }$ , it also belongs to the left child. If the left child contains no such point, then $a _ { j } ( s )$ belongs to the right child. Thus the claim follows by induction.

Since the intervals are nested and have lengths tending to zero, $\begin{array} { r } { \bigcap _ { n = 0 } ^ { \infty } I _ { j , n } ( s ) = \{ a _ { j } ( s ) \} . \mathrm { ~ I f ~ } u _ { j , n } ( s ) } \end{array}$ denotes the right endpoint of $I _ { j , n } ( s )$ , then $u _ { j , n }$ is Borel measurable and $\begin{array} { r } { a _ { j } ( s ) = \operatorname* { l i m } _ { n  \infty } u _ { j , n } ( s ) } \end{array}$ Therefore, $a _ { j } : D \to [ 0 , 1 ]$ is Borel measurable.

This completes the recursive construction for every $j = 1 , \ldots , d .$ . Define $\lambda ( s ) : = { \big ( } a _ { 1 } ( s ) , \ldots , a _ { d } ( s ) { \big ) }$ Since each coordinate $a _ { j }$ is Borel measurable, the map $\lambda : D \to [ 0 , 1 ] ^ { d }$ is Borel measurable.

Finally, define recursively $K _ { s } ^ { 0 } : = \Gamma _ { s }$ and $K _ { s } ^ { j } : = { \Big \{ } x \in K _ { s } ^ { j - 1 } : x _ { j } = a _ { j } ( s ) { \Big \} }$ . Each $K _ { s } ^ { j }$ is non-empty and compact. The set $K _ { s } ^ { d }$ contains exactly the point $\lambda ( s )$ , so $\lambda ( s ) \in \Gamma _ { s } \subseteq C$ . By construction, $a _ { 1 } ( s )$ is the smallest first coordinate in $\Gamma _ { s } ;$ among points having that first coordinate, $a _ { 2 } ( s )$ is the smallest second coordinate; and so on. Consequently, $\lambda ( s ) = \operatorname { l e x m i n } \Gamma _ { s }$ . The lexicographically smallest element is necessarily unique. □

Lemma 29. Let $s , d \in \mathbb { N }$ and $F \in C ^ { 1 } \left( [ 0 , 1 ] ^ { s } ; \mathbb { R } ^ { d } \right)$ . If there exist constants $c _ { 1 } , \ldots , c _ { s } \geq 0$ such that $\| \partial _ { m } F ( z ) \| _ { 2 } \leq c _ { m }$ holds for all $z \in [ 0 , 1 ] ^ { s }$ and all $m \in [ s ]$ , then

$$
\| F ( x ) - F ( y ) \| _ { 2 } \leq \sum _ { m = 1 } ^ { s } c _ { m } \left| x _ { m } - y _ { m } \right|
$$

holds for all x, $, y \in [ 0 , 1 ] ^ { s }$

Proof. By the fundamental theorem of calculus we have

$$
\begin{array} { r l } {  { \| F ( y ) - F ( x ) \| _ { 2 } = \| \int _ { 0 } ^ { 1 } { D F ( x + t ( y - x ) ) ( y - x ) d t } \| _ { 2 } } } \\ & { \leq \int _ { 0 } ^ { 1 } \| \displaystyle \sum _ { m = 1 } ^ { s } ( y _ { m } - x _ { m } ) \partial _ { m } F ( x + t ( y - x ) ) \| _ { 2 } d t } \\ & { \leq \int _ { 0 } ^ { 1 } ( \displaystyle \sum _ { m = 1 } ^ { s } | y _ { m } - x _ { m } | \| \partial _ { m } F ( x + t ( y - x ) ) \| _ { 2 } ) d t \leq \displaystyle \sum _ { m = 1 } ^ { s } c _ { m } | y _ { m } - x _ { m } | . } \end{array}
$$

Lemma 30. Let $\ell \in \mathbb { N }$ , and let $u \subseteq [ n _ { 0 } ]$ be nonempty. $I f v \in \Theta _ { \ell - 1 } ^ { \mathrm { m a t } } ( n _ { 0 } , \beta , \kappa )$ , then

$$
\left\| G _ { u } ^ { ( \ell ) } ( x , v ) - G _ { u } ^ { ( \ell ) } ( y , v ) \right\| _ { 2 } \leq M _ { | u | } ^ { ( \ell ) } ( \kappa ) \prod _ { j \in u } \beta _ { j } \sum _ { m = 1 } ^ { n _ { 0 } } \beta _ { m } \left| x _ { m } - y _ { m } \right|
$$

holds for all $x , y \in [ 0 , 1 ] ^ { n _ { 0 } }$ , where $\begin{array} { r } { M _ { k } ^ { ( \ell ) } ( \kappa ) : = A _ { 2 } B _ { 1 } ^ { ( \ell - 1 ) } ( \kappa ) B _ { k } ^ { ( \ell - 1 ) } ( \kappa ) + A _ { 1 } B _ { k + 1 } ^ { ( \ell - 1 ) } ( \kappa ) . } \end{array}$

Proof. For $m \in [ n _ { 0 } ]$ we have

$$
\begin{array} { r } { \partial _ { m } G _ { u } ^ { ( \ell ) } ( x , v ) = h _ { 2 } ^ { ( \ell ) } ( x , v ) \odot \partial _ { m } \mathcal { R } _ { v } ( x ) \odot D ^ { u } \mathcal { R } _ { v } ( x ) + h _ { 1 } ^ { ( \ell ) } ( x , v ) \odot \partial _ { m } D ^ { u } \mathcal { R } _ { v } ( x ) . } \end{array}
$$

Since $v \in \Theta _ { \ell - 1 } ^ { \mathrm { m a t } } ( n _ { 0 } , \beta , \kappa )$ , Theorem 1 and Lemma 19 give

$$
\begin{array} { r l } {  { \Big \| \partial _ { m } G _ { u } ^ { ( \ell ) } ( x , v ) \Big \| _ { 2 } \le A _ { 2 } \| \partial _ { m } \mathcal { R } _ { \nu } ( x ) \| _ { 2 } \| D ^ { u } \mathcal { R } _ { v } ( x ) \| _ { 2 } + A _ { 1 } \| \partial _ { m } D ^ { u } \mathcal { R } _ { v } ( x ) \| _ { 2 } } } \\ & { \le A _ { 2 } B _ { 1 } ^ { ( \ell - 1 ) } ( \kappa ) \beta _ { m } B _ { | u | } ^ { ( \ell - 1 ) } ( \kappa ) \prod _ { j \in u } \beta _ { j } + A _ { 1 } B _ { | u | + 1 } ^ { ( \ell - 1 ) } ( \kappa ) \beta _ { m } \prod _ { j \in u } \beta _ { j } = M _ { | u | } ^ { ( \ell ) } ( \kappa ) \beta _ { m } \prod _ { j \in u } \beta _ { j } . } \end{array}
$$

The asserted Lipschitz estimate follows from Lemma 29.

Lemma 31. $I f \left( E , \| \cdot \| \right)$ is a normed vector space, then

$$
\left\| { \frac { a } { \| a \| } } - { \frac { b } { \| b \| } } \right\| \leq { \frac { 2 \| b - a \| } { \operatorname* { m i n } \{ \| a \| , \| b \| \} } }
$$

holds for all $a , b \in E \setminus \{ 0 \}$

Proof. By the triangle inequality we have

$$
\left\| { \frac { a } { \| a \| } } - { \frac { b } { \| b \| } } \right\| = \left\| { \frac { a - b } { \| a \| } } + b \left( { \frac { 1 } { \| a \| } } - { \frac { 1 } { \| b \| } } \right) \right\| \leq { \frac { \| a - b \| } { \| a \| } } + \| b \| \left| { \frac { 1 } { \| a \| } } - { \frac { 1 } { \| b \| } } \right| = { \frac { \| a - b \| } { \| a \| } } + { \frac { \| \| b \| - \| a \| \| } { \| a \| } } .
$$

By the reverse triangle inequality we have $| \| b \| - \| a \| | \leq \| b - a \|$ and consequently

$$
\left\| { \frac { a } { \| a \| } } - { \frac { b } { \| b \| } } \right\| \leq { \frac { \| a - b \| } { \| a \| } } + { \frac { \| b \| - \| a \| } { \| a \| } } \leq { \frac { 2 \left\| a - b \right\| } { \| a \| } } \leq { \frac { 2 \| b - a \| } { \operatorname* { m i n } \{ \| a \| , \| b \| \} } } .
$$

The following conditional-expectation identity is standard; see [6, Chapter 4].

Lemma 32. Let X and Y be independent random variables taking values in Euclidean spaces, and let f be a bounded Borel-measurable function on the corresponding product space. Define $g ( x ) : = \mathbb { E } [ f ( x , Y ) ]$ . Then

$$
\mathbb { E } \big [ f ( X , Y ) \mid \sigma ( X ) \big ] = g ( X )
$$

almost surely.

[1] Krishnakumar Balasubramanian, Larry Goldstein, Nathan Ross, and Adil Salim. “Gaussian Random Field Approximation via Stein’s Method with Applications to Wide Random Neural Networks”. In: Applied and Computational Harmonic Analysis 72 (2024), p. 101668.

[2] Andrea Basteri and Dario Trevisan. “Quantitative Gaussian Approximation of Randomly Initialized Deep Neural Networks”. In: Machine Learning 113.9 (2024), pp. 6373–6393.

[3] Daniele Bracale, Stefano Favaro, Sandra Fortini, and Stefano Peluchetti. “Large-Width Functional Asymptotics for Deep Gaussian Neural Networks”. In: International Conference on Learning Representations. 2021. arXiv: 2102.10307.

[4] Tim De Ryck, Samuel Lanthaler, and Siddhartha Mishra. “On the Approximation of Functions by Tanh Neural Networks”. In: Neural Networks 143 (2021), pp. 732–750.

[5] Sjoerd Dirksen, Patrick Finke, Paul Geuchen, Dominik St¨oger, and Felix Voigtlaender. Near-Optimal Estimates for the ℓ<sup>p</sup>-Lipschitz Constants of Deep Random ReLU Neural Networks. 2025. arXiv: 2506.19695.

[6] Rick Durrett. Probability: Theory and Examples. 5th ed. Cambridge University Press, 2019.

[7] Stefano Favaro, Boris Hanin, Domenico Marinucci, Ivan Nourdin, and Giovanni Peccati. “Quantitative CLTs in Deep Neural Networks”. In: Probability Theory and Related Fields 191.3–4 (2025), pp. 933–977.

[8] Mahyar Fazlyab, Alexander Robey, Hamed Hassani, Manfred Morari, and George J. Pappas. “Eficient and Accurate Estimation of Lipschitz Constants for Deep Neural Networks”. In: Advances in Neural Information Processing Systems 32. 2019.

[9] Paul Geuchen, Dominik St¨oger, Thomas Telaar, and Felix Voigtlaender. “Upper and Lower Bounds for the Lipschitz Constant of Random Neural Networks”. In: Information and Inference: A Journal of the IMA 14.2, iaaf009 (2025). doi: 10.1093/imaiai/iaaf009.

[10] Xavier Glorot and Yoshua Bengio. “Understanding the Dificulty of Training Deep Feedforward Neural Networks”. In: Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics. Vol. 9. Proceedings of Machine Learning Research. 2010, pp. 249– 256.

[11] Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy. “Explaining and Harnessing Adversarial Examples”. In: International Conference on Learning Representations. 2015. arXiv: 1412.6572.

[12] Boris Hanin. “Random Fully Connected Neural Networks as Perturbatively Solvable Hierarchies”. In: Journal of Machine Learning Research 25.267 (2024), pp. 1–58. url: https: //jmlr.org/papers/v25/23-0643.html.

[13] Boris Hanin. “Random Neural Networks in the Infinite Width Limit as Gaussian Processes”. In: The Annals of Applied Probability 33.6A (2023), pp. 4798–4819. doi: 10.1214/22-AAP1933.

[14] Boris Hanin. “Which Neural Net Architectures Give Rise to Exploding and Vanishing Gradients?” In: Advances in Neural Information Processing Systems 31. 2018.

[15] Boris Hanin and Mihai Nica. “Products of Many Large Random Matrices and Gradients in Deep Neural Networks”. In: Communications in Mathematical Physics 376.1 (2020), pp. 287– 322.

[16] Michael Hardy. “Combinatorics of Partial Derivatives”. In: The Electronic Journal of Combinatorics 13.1 (2006), R1. doi: 10.37236/1027.

[17] Kurt Hornik, Maxwell Stinchcombe, and Halbert White. “Universal Approximation of an Unknown Mapping and Its Derivatives Using Multilayer Feedforward Networks”. In: Neural Networks 3.5 (1990), pp. 551–560.

[18] Matt Jordan and Alexandros G. Dimakis. “Exactly Computing the Local Lipschitz Constant of ReLU Networks”. In: Advances in Neural Information Processing Systems 33. 2020.

[19] Alexander Keller, Frances Y. Kuo, Dirk Nuyens, and Ian H. Sloan. “Lattice-Based Deep Neural Networks: Regularity and Tailored Regularization”. In: Monte Carlo and Quasi-Monte Carlo 2024. Ed. by Christiane Lemieux and Ben Feng. Vol. 522. Springer Proceedings in Mathematics & Statistics. Cham: Springer Nature Switzerland, 2026, pp. 41–72. doi: 10.1007/978-3-032- 10590-5\_2.

[20] Alexander Keller, Frances Y. Kuo, Dirk Nuyens, and Ian H. Sloan. “Regularity and Tailored Regularization of Deep Neural Networks, with Application to Parametric PDEs in Uncertainty Quantification”. In: Mathematics of Computation (June 23, 2026). doi: 10.1090/mcom/4220.

[21] Taeyoung Kim and Hongseok Yang. “An Infinite-Width Analysis on the Jacobian-Regularised Training of a Neural Network”. In: Proceedings of the 41st International Conference on Machine Learning. Vol. 235. Proceedings of Machine Learning Research. 2024, pp. 24584–24657. url: https://proceedings.mlr.press/v235/kim24ah.html.

[22] B´eatrice Laurent and Pascal Massart. “Adaptive Estimation of a Quadratic Functional by Model Selection”. In: The Annals of Statistics 28.5 (2000), pp. 1302–1338.

[23] Jaehoon Lee, Yasaman Bahri, Roman Novak, Samuel S. Schoenholz, Jefrey Pennington, and Jascha Sohl-Dickstein. “Deep Neural Networks as Gaussian Processes”. In: International Conference on Learning Representations. 2018. arXiv: 1711.00165.

[24] Marcello Longo, Siddhartha Mishra, T. Konstantin Rusch, and Christoph Schwab. “Higher-Order Quasi-Monte Carlo Training of Deep Neural Networks”. In: SIAM Journal on Scientific Computing 43.6 (2021), A3938–A3966. doi: 10.1137/20M1369373.

[25] Alexander G. de G. Matthews, Mark Rowland, Jiri Hron, Richard E. Turner, and Zoubin Ghahramani. “Gaussian Process Behaviour in Wide Deep Neural Networks”. In: International Conference on Learning Representations. 2018. arXiv: 1804.11271.

[26] Siddhartha Mishra and T. Konstantin Rusch. “Enhancing Accuracy of Deep Learning Algorithms by Training with Low-Discrepancy Sequences”. In: SIAM Journal on Numerical Analysis 59.3 (2021), pp. 1811–1834. doi: 10.1137/20M1344883.

[27] James R. Munkres. Topology. 2nd ed. Upper Saddle River, NJ: Prentice Hall, 2000.

[28] Leonid Pastur and Victor Slavin. “On Random Matrices Arising in Deep Neural Networks: General IID Case”. In: Random Matrices: Theory and Applications 12.1, 2250046 (2023). doi: 10.1142/S2010326322500460.

[29] Jefrey Pennington, Samuel S. Schoenholz, and Surya Ganguli. “Resurrecting the Sigmoid in Deep Learning through Dynamical Isometry: Theory and Practice”. In: Advances in Neural Information Processing Systems 30. 2017.

[30] Sina Sharifi and Mahyar Fazlyab. Provable Bounds on the Hessian of Neural Networks: Derivative-Preserving Reachability Analysis. 2024. arXiv: 2406.04476.

[31] Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. “Intriguing Properties of Neural Networks”. In: International Conference on Learning Representations. 2014. arXiv: 1312.6199.

[32] Roman Vershynin. “Introduction to the Non-Asymptotic Analysis of Random Matrices”. In: Compressed Sensing: Theory and Applications. Ed. by Yonina C. Eldar and Gitta Kutyniok. Cambridge University Press, 2012, pp. 210–268.

[33] Aladin Virmaux and Kevin Scaman. “Lipschitz Regularity of Deep Neural Networks: Analysis and Eficient Estimation”. In: Advances in Neural Information Processing Systems 31. 2018.