# Nearly Tight Rademacher Bounds for Sparsely Activated Neural Networks

Xiaoyu Li<sup>1</sup> Zhizhou Sha<sup>2</sup> Jiaojiao Jiang<sup>1</sup> Junbin Gao<sup>3</sup> Andi Han<sup>3</sup>

<sup>1</sup>University of New South Wales <sup>2</sup>University of Texas at Austin <sup>3</sup>University of Sydney

{xiaoyu.li2,jiaojiao.jiang}@unsw.edu.au zhizhousha@utexas.edu {junbin.gao,andi.han}@sydney.edu.au

## Abstract

An input may activate few hidden units even when diferent inputs collectively use an entire network. We study the statistical complexity of this input-dependent sparsity in the one-hidden-layer ReLU model of Awasthi et al. (COLT 2024). For width �, at most � active units per input, and efective weight and bias bounds �, �, every size-� sample in the class’s fixed radius-� input domain satisfies <sub>R</sub> <sub>(�)</sub> <sub>≤</sub> <sub>��</sub> <sub>�</sub> <sub>min{�,</sub> <sub>√︁��/�</sub> <sub>log</sub>3/2<sub>(2�)}</sub> <sub>+ ��/</sub>√<sub>�.</sub> <sub>A</sub> <sub>support-preserving</sub> <sub>cover</sub> <sub>and</sub> <sub>a</sub> <sub>single</sub> <sub>normalized</sub> chaining argument remove the previous explicit dimension factor, up to logarithms. Lower bounds on appropriate i.i.d. marginals match up to those logarithms, showing how changing active units across inputs retains a width dependence. The input domain matters: zero-bias networks sparse on the entire <sub>ball</sub> <sub>have</sub> <sub>at</sub> <sub>most</sub> <sub>2�</sub> <sub>nonzero</sub> <sub>units</sub> <sub>and</sub> <sub>complexity</sub> <sub>�(��</sub> <sub>�/</sub>√<sub>�),</sub> <sub>whereas</sub> <sub>bias</sub> <sub>bounds</sub> <sub>comparable</sub> <sub>to</sub> � � restore the worst-case rate on that same domain in only logarithmic dimension. A spherical-cap construction proves the latter claim without assuming sparsity merely on the sampling support. For a specified normalized bounded loss and biases comparable to ��, we also obtain agnostic minimax excess-risk bounds of order min{1, √︁�/(��)} up to logarithms.

Keywords: Activation sparsity, Rademacher complexity, metric entropy, neural networks, generalization

## 1 Introduction

Activation sparsity limits how many hidden units a network uses on an input, while allowing the identities of those units to change across inputs. It limits the number of nonzero contributions without simply reducing the number of available parameters. The statistical question is whether this input-dependent constraint controls a network’s capacity uniformly over all admissible activation patterns.

Awasthi et al. (2024), henceforth ADKM, formulated this question for one-hidden-layer ReLU networks. Their model is motivated by sparse hidden activations, but imposes an exact mathematical promise: at most � of the � hidden units are active on every point in a prescribed input set. For bounded weights and inputs, they obtained a Rademacher bound with dependence $\widetilde { O } ( \sqrt { s n k / m } )$ , where � is the input dimension, and <sub>conjectured</sub> <sub>that</sub> <sub>the</sub> <sub>factor</sub> √<sub>�</sub> <sub>could</sub> <sub>be</sub> <sub>removed</sub> <sub>(Conjecture</sub> <sub>20).</sub> <sub>The</sub> <sub>challenge</sub> <sub>is</sub> <sub>to</sub> <sub>retain</sub> <sub>the</sub> <sub>sparsity</sub> improvement in width while avoiding a count of all halfspace activation patterns.

We establish the dimension-free rate up to log ${ \boldsymbol { \beta } } ^ { 3 / 2 } ( 2 m )$ and ask when its remaining width dependence is necessary. Our analysis separates three questions: the worst-case capacity of the fixed-domain class, the efect of the domain on admissible activation regions, and the agnostic learning dificulty under a specified loss. The answers distinguish sparse computation on each input from a globally small collection of useful neurons.

## 1.1 Model and results

Let $n , s , m \geq 1 , 1 \leq k \leq s .$ and $W , B , R \geq 0$ . Fix a nonempty set $\mathcal { X } _ { 0 } \subseteq \{ x \in \mathbb { R } ^ { n } : \| x \| _ { 2 } \leq R \}$ before sampling.   
Write $\sigma ( z ) = \operatorname* { m a x } \{ z , 0 \}$ .

Definition 1 (ADKM’s class on a fixed input set). The class $\mathcal { H } _ { n , s , k } ^ { W , B } ( X _ { 0 } )$ consists of restrictions to $\chi _ { 0 }$ of functions

$$
h ( x ) = \sum _ { j = 1 } ^ { s } u _ { j } \sigma ( \langle w _ { j } , x \rangle - b _ { j } )
$$

admitting a representation with

$$
\| u \| _ { \infty } \operatorname* { m a x } _ { j } \| w _ { j } \| _ { 2 } \leq W , \qquad \| u \| _ { \infty } \operatorname* { m a x } _ { j } | b _ { j } | \leq B , \qquad \# \{ j : \langle w _ { j } , x \rangle > b _ { j } \} \leq k \quad ( x \in \mathcal { X } _ { 0 } ) .
$$

The input set is part of the class definition. Upper bounds hold on every $S = ( x _ { 1 } , \ldots , x _ { m } ) \in X _ { 0 } ^ { m }$ ; a lower-bound construction is allowed to choose $\chi _ { 0 }$ and a distribution on it. Requiring sparsity on a larger set can give a smaller class. No assertion below identifies the support-wide class with a class chosen after seeing a sample.

For independent uniform signs $\zeta _ { i } \in \{ - 1 , 1 \}$ , define

$$
\mathcal { R } _ { \mathcal { H } } ( S ) = \frac { 1 } { m } \mathbb { E } _ { \zeta } \operatorname* { s u p } _ { h \in \mathcal { H } } \sum _ { i = 1 } ^ { m } \zeta _ { i } h ( x _ { i } ) , \qquad A = W R , \quad \ell _ { m } = \log ( 2 m ) , \quad \rho _ { m } = \frac { 1 } { m } \mathbb { E } \left| \sum _ { i = 1 } ^ { m } \zeta _ { i } \right| .
$$

All logarithms are natural unless indicated otherwise. The elementary moment bound in Lemma 9 gives $1 / \sqrt { 3 m } \leq \rho _ { m } \leq 1 / \sqrt { m }$

Theorem 2 (Dimension-free upper bound with separated bias). There is a universal constant $C \geq 1$ such that, for all parameters and input sets above and every $S \in { \mathbb X } _ { 0 } ^ { m }$

$$
\mathcal { R } _ { \mathcal { H } _ { n , s , k } ^ { W , B } ( \chi _ { 0 } ) } ( S ) \leq C A \operatorname* { m i n } \left\{ k , \sqrt { \frac { s k } { m } } \ell _ { m } ^ { 3 / 2 } \right\} + k B \rho _ { m } .\tag{1}
$$

In addition, $\mathcal { R } _ { \mathcal { H } _ { n , s , k } ^ { W , B } ( \chi _ { 0 } ) } \left( S \right) \leq k ( A + B ) \ a n d \mathcal { R } _ { \mathcal { H } _ { n , s , k } ^ { W , B } ( \chi _ { 0 } ) } ( S ) \leq 2 s ( A + B ) / \sqrt { m } .$

Thus one may always take the smallest of the three upper bounds. The last one is useful in the dense regime $k = s ,$ , where it avoids a logarithmic loss. Dimension-free means that � does not appear separately from $R ;$ on the unnormalized Boolean cube, $R = { \sqrt { n } }$ still carries a dimension dependence.

Theorem 3 (Lower bound on i.i.d. samples). Let $q = \operatorname* { m i n } \{ \lfloor s / k \rfloor , m \}$ and suppose $n \geq q$ . There are a finite input set $\chi _ { 0 }$ in the radius-� ball and a marginal $D _ { x }$ on it such that

$$
\mathbb { E } _ { S \sim D _ { x } ^ { m } } \mathcal { R } _ { \mathcal { H } _ { n , s , k } ^ { W , B } ( \chi _ { 0 } ) } ( S ) \geq \frac { 1 } { 4 \sqrt { 2 } } \left( A \operatorname* { m i n } \left\{ k , \sqrt { \frac { s k } { m } } \right\} + \frac { k B } { \sqrt { m } } \right) .\tag{2}
$$

The same expression lower-bounds the supremum over all admissible input sets and samples.

For $n \geq q$ and $m \geq s / k$ , these results determine the worst-case scaling $\widetilde { \Theta } ( A \sqrt { s k / m } + k B / \sqrt { m } )$ , with logarithmic slack only in the weight term. For $m \leq s / k$ and suficiently large �, the weight term saturates at ��. The bounds cover arbitrary $s , k ,$ � without divisibility assumptions. They concern class complexity; Theorem 3 alone is not a lower bound on every learning algorithm.

The domain can change the answer. On the entire Euclidean ball, zero-bias �-sparsity forces at most 2� nonzero units, giving the smaller rate $\Theta ( k A / \sqrt { m } )$ in the worst case (Proposition 10). Positive thresholds restore the width dependence: Theorem 12 realizes the lower rate on that same full-ball class, using separated spherical caps in only logarithmic dimension when $B \geq ( { \sqrt { 3 } } / 2 ) A$ . Thus the lower bound need not rely on imposing sparsity only at isolated sampling locations.

A same-loss learning characterization. We derive general bounded-loss guarantees and, separately, prove an agnostic minimax result for the normalized class and a fixed bounded linear loss. Under the explicit dimension and comparable-bias conditions of Theorem 14, its expected excess-risk rate is $\widetilde { \Theta } ( \operatorname* { m i n } \{ 1 , \sqrt { s / ( k m ) } \} )$ . This lower bound is established directly in the learning experiment, not inferred from Rademacher complexity. It is not a claim of optimality for every loss or for realizable learning.

Comparison with the original bound. ADKM’s equation (15) has the displayed dependence $( W R +$ $B ) \sqrt { s n k \log ( k m ( R + B ) ) } / \sqrt { m }$ in its parameter regime. Their conjectured dependence is $( W R + B ) \sqrt { s k / m }$ Equation (1) removes the explicit $\sqrt { n }$ at a logarithmic cost and improves the bias coeficient from $\sqrt { s k }$ to $k .$ The logarithm-free form of their conjecture remains unresolved here. Bounds based only on the sum of the individual neuron complexities give $O ( s ( A + B ) / \sqrt { m } )$ ; the point is to combine norm control with the shared activation budget.

## 1.2 Technique overview

The upper bound uses the total activation budget without fixing an activation pattern. The lower bounds then ask how many independently signed neuron clusters that budget permits, first on a discrete domain and then on the entire ball.

From a network to one normalized neuron process. By positive homogeneity, absorb each outer weight’s magnitude into its neuron, leaving an outer sign and parameters $\lVert v _ { j } \rVert _ { 2 } \leq W , | \beta _ { j } | \leq B . \operatorname { I f } I _ { j }$ is its active sample set, double-counting gives $\begin{array} { r } { \sum _ { j } | I _ { j } | \le k m } \end{array}$ . Let $\mathcal { F } _ { t }$ be the sample-output vectors of one neuron active on at most � coordinates, and put $\begin{array} { r } { Q = \operatorname* { m a x } _ { 1 \leq t \leq m } t ^ { - 1 / 2 } \operatorname* { s u p } _ { f \in \mathcal { F } _ { t } } | \langle \zeta , f \rangle | } \end{array}$ |. Then

$$
\sum _ { j } \left| \sum _ { i } \zeta _ { i } \sigma ( \langle v _ { j } , x _ { i } \rangle - \beta _ { j } ) \right| \leq Q \sum _ { j } { \sqrt { | I _ { j } | } } \leq Q { \sqrt { s k m } } .
$$

Inactive units contribute zero. After division by � and averaging, Lemma 4 bounds the network complexity by $\sqrt { s k / m } \mathbb { E } Q$ . The displayed inequality holds for each sign vector: it does not require choosing the active sets before observing the signs. The remaining task is to control all activation counts simultaneously.

A cover that preserves inactive coordinates. A direct afine approximation followed by ReLU can turn zero coordinates into small positive ones, losing the sparse support in its Euclidean error. We instead shift every approximating pre-activation downward. $\operatorname { I f } \| g - { \widehat { g } } \| _ { \infty } \leq \delta ,$ then ${ \widehat { f } } = \sigma ( { \widehat { g } } - \delta )$ vanishes wherever $f = \sigma ( g )$ vanishes and difers from $f$ by at most 2� elsewhere. Hence $\| f - \widehat { f } \| _ { 2 } \leq 2 \delta \sqrt { t }$ for $f \in \mathcal { F } _ { t }$ . Applying this transformation to a dimension-free afine cover gives Lemma 5, without counting halfspace activation patterns. The centers are allowed outside the parameter-constrained class; only their approximation error matters.

One chain, rather than separate bounds at each count. Normalize $\mathcal { F } _ { t }$ by $\sqrt { t }$ and take the symmetric union

$$
T = \bigcup _ { t = 1 } ^ { m } \left( t ^ { - 1 / 2 } \mathcal { F } _ { t } \cup - t ^ { - 1 / 2 } \mathcal { F } _ { t } \right) , \qquad Q = \operatorname* { s u p } _ { z \in T } \langle \zeta , z \rangle .
$$

The normalization removes � from the cover’s scale-dependent entropy; selecting a member of the union costs only an additive log(2�). A single finite chaining argument therefore bounds the expectation of the maximum itself, without interchanging maximum and expectation. Truncating the chain at radius at most $( A + B ) / \sqrt { m }$ controls its residual by $A + B$ . The afine entropy contributes $\sqrt { \log ( 2 m ) }$ , and summing over scales contributes another logarithm, giving $\mathbb { E } Q = O ( ( A + B ) \log ^ { 3 / 2 } ( 2 m ) )$ . This identifies the remaining logarithmic loss: removing the lower-order union cost alone would not remove it.

Clipping thresholds separates the constant part. The preceding argument initially charges both � and � at the network rate. To refine it, clip $\beta$ to $\beta ^ { \circ } \in [ - A , A ]$ and use the identity

$$
\sigma ( \langle v , x \rangle - \beta ) = \sigma ( \langle v , x \rangle - \beta ^ { \circ } ) + ( - \beta - A ) _ { + } \qquad ( \| x \| _ { 2 } \leq R ) .
$$

Clipping creates no new activation. Every nonzero constant correction comes from a unit previously active everywhere, so at most � corrections occur. Thus $h = h _ { 0 } + c ,$ with clipped bias bound min $\{ A , B \}$ and $| c | \leq k ( B - A )$ <sub>+</sub> (Lemma 8). Applying the chain and the envelope bound to $h _ { 0 } .$ , and the exact constant-class complexity to �, yields Theorem 2.

Independent clusters and the geometry of the domain. For the discrete lower bound, use $q \ =$ min $\{ \lfloor s / k \rfloor , m \}$ orthogonal input directions and � parallel neurons per direction. Each cluster carries its own sign. Under uniform i.i.d. sampling, its signed occupancy has absolute expectation of order ${ \sqrt { m / q } } ,$ giving the weight lower bound without balanced-count or divisibility assumptions. The separate constant subclass supplies the bias term. On the full ball, however, a generic antipodal pair counts every nonzero zero-bias neuron, forcing at most 2� in total. To recover independent clusters there, we use disjoint spherical caps. Directions with pairwise inner product at most $1 / 2 _ { ; }$ , together with threshold $\tau A$ for $\tau = \sqrt { 3 } / 2$ , prevent two clusters from activating at any point of the ball. A random-sign packing supplies the directions in $O ( \log ( 2 q ) )$ dimensions. When $B \geq \tau A$ , each cap center still has output amplitude $( 1 - \tau ) k A .$ , so the same occupancy argument applies (Theorem 12).

Turning sign encoding into a learning lower bound. A complexity lower bound alone does not establish learning hardness. Under the conditions of Theorem 14, normalization by $k ( A + B )$ makes the cap construction’s sign amplitude a positive universal constant. Assign binary labels small unknown means $\delta \theta _ { b }$ at the � cap centers. Adjacent sign choices have joint-sample KL divergence at most $O ( m \delta ^ { 2 } / q )$ . Choosing � as a suficiently small constant times $\sqrt { q / m }$ keeps adjacent distributions hard to distinguish. Total variation then limits any learner’s correlation with the signs, including randomized and improper learners. For the specified bounded linear loss, this gives an expected excess-risk lower bound of order ${ \sqrt { q / m } } .$ . Approximate ERM and the complexity upper bound give the corresponding upper rate for that same loss and normalized class; Appendix B supplies the testing calculation.

## 1.3 Related work

Our comparison uses the support-wide class and product norms of Awasthi et al. (2024). Their expressivity and computational results are distinct from the capacity question studied here. Norm-based analyses such as

Neyshabur et al. (2015); Golowich et al. (2018) control complexity without imposing this activation promise. We use an elementary neuronwise bound as the dimension-free baseline rather than asserting that all such norm-based results have identical specializations.

Other sparsity notions lead to diferent guarantees. Galanti et al. (2023) study compositionally sparse networks, in which each neuron has a limited number ofinputs. Muthukumar and Sulam (2023) use activation stability and sensitivity analysis to obtain predictor-dependent, derandomized PAC-Bayes guarantees across multiple layers. Our result is a uniform complexity bound for an entire one-layer class with a fixed activation promise; it does not require stability under parameter perturbations.

Scale-sensitive covers of norm-constrained linear classes have a long history (Zhang, 2002). The afine cover used here is an application of dual Sudakov (Pajor and Tomczak-Jaegermann, 1986; Ledoux and Talagrand, 1991); the chain uses the familiar multiscale principle of Dudley (1967). The contribution of the covering step is its compatibility with sparse supports. We include the finite chaining argument and a proof of the required geometric covering estimate so that the proof interfaces can be checked directly.

## 2 A dimension-free upper bound

We first prove the bound with scale $M = A + B$ . We then separate the contribution of large biases. Vectors of sample evaluations use the unnormalized Euclidean norm in $\mathbb { R } ^ { m }$ . The covering number $N ( T , \epsilon , \parallel \cdot \parallel )$ permits centers in the ambient vector space; all covers used below are finite and deterministic once the sample is fixed.

## 2.1 Reduction and the activation budget

Positive homogeneity gives the exact reduced representation

$$
h ( x ) = \sum _ { j = 1 } ^ { s } \varepsilon _ { j } \sigma ( \langle v _ { j } , x \rangle - \beta _ { j } ) , \qquad \varepsilon _ { j } \in \{ - 1 , 1 \} , \quad \| v _ { j } \| _ { 2 } \le W , \quad | \beta _ { j } | \le B .\tag{3}
$$

For $u _ { j } \neq 0 _ { : }$ , take $v _ { j } = | u _ { j } | w _ { j } , \beta _ { j } = | u _ { j } | b _ { j }$ , and $\varepsilon _ { j } = \mathrm { s i g n } ( u _ { j } )$ . For $u _ { j } = 0$ , take $v _ { j } = 0 , \beta _ { j } = 0$ and $\varepsilon _ { j } = 1$ ; this only decreases the activation count. Conversely, (3) is realized in Definition 1 by $u _ { j } = \varepsilon _ { j } , w _ { j } = v _ { j }$ , and $b _ { j } = \beta _ { j }$ . Sparsity does not depend on the outer signs.

For $t \in [ m ]$ , let

$$
\mathcal { F } _ { t } = \left\{ ( \sigma ( \langle v , x _ { i } \rangle - \beta ) ) _ { i = 1 } ^ { m } : \| v \| _ { 2 } \leq W , ~ \vert \beta \vert \leq B , ~ \# \{ i : \langle v , x _ { i } \rangle > \beta \} \leq t \right\} ,
$$

and set $P ( t ) = \operatorname* { s u p } _ { f \in { \mathcal { F } } _ { t } } | \langle \zeta , f \rangle |$ and $Q = \operatorname* { m a x } _ { t \in [ m ] } P ( t ) / \sqrt { t } .$ . Each $f \in \mathcal { F } _ { t }$ has entries in [0, �] and satisfies $\| { f } \| _ { 2 } \leq M { \sqrt { t } }$

Lemma 4 (Sample relaxation and budget). For every $S \in { \mathbb X } _ { 0 } ^ { m }$

$$
\mathcal { R } _ { \mathcal { H } _ { n , s , k } ^ { W , B } ( \chi _ { 0 } ) } ( S ) \leq \sqrt { \frac { s k } { m } } \mathbb { E } _ { \zeta } Q .\tag{4}
$$

Proof. Enlarge the reduced parameter set by requiring sparsity only on �. For fixed parameters, maximizing over the outer signs yields $\textstyle \sum _ { j } | N _ { j } |$ , where $\begin{array} { r } { N _ { j } = \sum _ { i } \zeta _ { i } \sigma ( \langle \upsilon _ { j } , x _ { i } \rangle - \beta _ { j } ) } \end{array}$ . Let $t _ { j } = \# \{ i : \left. v _ { j } , x _ { i } \right. > \beta _ { j } \}$ . A unit with $t _ { j } = 0$ contributes zero; otherwise $| N _ { j } | \leq Q \sqrt { t _ { j } }$ . Double-counting active sample-unit pairs gives $\begin{array} { r } { \sum _ { j } t _ { j } \le k m } \end{array}$ Therefore, for every sign vector and admissible configuration,

$$
\sum _ { j } | N _ { j } | \leq Q \sum _ { j } { \sqrt { t _ { j } } } \leq Q { \sqrt { s \sum _ { j } t _ { j } } } \leq Q { \sqrt { s \ k m } } .
$$

Take the supremum, divide by �, and average over the signs.

## 2.2 Covering sparse outputs

The afine evaluation class ${ \mathcal { A } } = \{ ( \langle v , x _ { i } \rangle - \beta ) _ { i } : \| v \| _ { 2 } \leq W , \ | \beta | \leq B \}$ obeys

$$
\log N ( \mathcal { A } , \delta , \| \cdot \| _ { \infty } ) \leq 8 0 \frac { A ^ { 2 } \log ( 2 m ) } { \delta ^ { 2 } } + \log ( 1 + 4 B / \delta ) , \qquad \delta > 0 .\tag{5}
$$

For completeness, Lemma 15 proves this estimate using a Gaussian packing argument. Restricting to span $\left\{ x _ { i } \right\}$ deals with the fact that max<sub>�</sub> $| \langle v , x _ { i } \rangle |$ need only be a seminorm on the original space.

Lemma 5 (Shifted cover). For $t \in [ m ]$ and $\epsilon > 0$

$$
\log N ( \mathcal { F } _ { t } , \epsilon , \| \cdot \| _ { 2 } ) \leq 3 2 0 \frac { A ^ { 2 } t \log ( 2 m ) } { \epsilon ^ { 2 } } + \log ( 1 + 8 B \sqrt { t } / \epsilon ) .\tag{6}
$$

Proof. Put $\delta = \epsilon / ( 2 \sqrt { t } )$ and cover $\mathcal { A }$ in sup norm at radius $\delta .$ For $f = \sigma ( g ) \in \mathcal { F } _ { t }$ , choose a center $\widehat g$ with $\| g - \widehat g \| _ { \infty } \leq \delta$ and use $\widehat { f } = \sigma ( \widehat { g } - \delta ) . \operatorname { I f } g _ { i } \leq 0 .$ , then $\widehat { g } _ { i } - \delta \leq 0 .$ , so $f _ { i } = { \widehat { f _ { i } } } = 0$ . On the other coordinates, the 1-Lipschitz property of ReLU gives $\vert f _ { i } - \widehat { f _ { i } } \vert \leq 2 \delta$ . Thus $\| f - { \widehat { f } } \| _ { 2 } \leq 2 \delta { \sqrt { t } } = \epsilon$ . The transformed cover has no more centers than the afine cover; substituting $\delta$ in (5) proves the claim. The centers need not satisfy the bias constraint, which is why we use external covers. □

## 2.3 One chain over all normalized scales

Define the symmetric set

$$
T = \bigcup _ { t = 1 } ^ { m } \left( t ^ { - 1 / 2 } \mathcal { F } _ { t } \cup - t ^ { - 1 / 2 } \mathcal { F } _ { t } \right) .
$$

It contains $0 ,$ has radius at most $M ,$ and satisfies $Q = \operatorname* { s u p } _ { z \in T } \langle \zeta , z \rangle$ . By scaling Lemma 5 and taking the union of $2 m$ covers, we get the uniform estimate

$$
\log N ( T , \epsilon , \| \cdot \| _ { 2 } ) \leq \ell _ { m } + 3 2 0 { \frac { A ^ { 2 } \ell _ { m } } { \epsilon ^ { 2 } } } + \log ( 1 + 8 B / \epsilon ) .\tag{7}
$$

The entropy cost of selecting an activation count is only log $\left( 2 m \right)$ , separate from the $\epsilon ^ { - 2 }$ term. No comparison of the expectations of diferent $P ( t )$ is needed.

Lemma 6 (Finite Rademacher chaining). Let $T \subseteq \mathbb { R } ^ { m }$ have radius at most $D > 0$ and put $r _ { j } = D 2 ^ { - j }$ . For an integer $J \geq 1$ , suppose external $r _ { j }$ -nets exist with cardinalities $N _ { j }$ and log $N _ { j } \leq H _ { j }$ , where $0 \leq H _ { 1 } \leq \cdots \leq H _ { J }$ Then

$$
\mathbb { E } \operatorname* { s u p } _ { z \in T } \langle \zeta , z \rangle \leq \sqrt { m } r _ { J } + 6 \sum _ { j = 1 } ^ { J } r _ { j } \sqrt { H _ { j } } .\tag{8}
$$

Proof. Choose deterministic net maps $\pi _ { j }$ with $\| z - \pi _ { j } z \| _ { 2 } \leq r _ { j }$ and set $\pi _ { 0 } z = 0 , N _ { 0 } = 1$ . The edge set $E _ { j } = \left\{ \pi _ { j } z - \pi _ { j - 1 } z : z \in T \right\}$ has at most $N _ { j } N _ { j - 1 }$ elements, each with norm at most $r _ { j } + r _ { j - 1 } = 3 r _ { j }$ . For every deterministic vector �,

$$
\mathbb { E } e ^ { \lambda \langle \zeta , a \rangle } = \prod _ { i } \mathrm { c o s h } ( \lambda a _ { i } ) \leq e ^ { \lambda ^ { 2 } \| a \| _ { 2 } ^ { 2 } / 2 } .
$$

The exponential-moment maximal inequality therefore gives

$$
\mathbb { E } \operatorname* { m a x } _ { a \in E _ { j } } \langle \zeta , a \rangle \le 3 r _ { j } \sqrt { 2 \log ( N _ { j } N _ { j - 1 } ) } \le 6 r _ { j } \sqrt { H _ { j } } .
$$

If the edge set is a singleton, its expected maximum is zero, so the same inequality applies. Telescope from 0 to $\pi _ { J } z$ and use $\langle \zeta , z - \pi _ { J } z \rangle \le \sqrt { m } r _ { J }$ for the residual. Taking the supremum and expectation proves (8). The process is defined at every point of ${ \mathbb R } ^ { m }$ , so external centers cause no change to the argument. □

Proposition 7 (Unseparated bound). There is a universal constant $C _ { 0 } \geq 1$ such that

$$
\begin{array} { r } { \mathbb { E } Q \leq C _ { 0 } M \ell _ { m } ^ { 3 / 2 } , \qquad \mathcal { R } _ { \mathcal { H } _ { n , s , k } ^ { W , B } ( \chi _ { 0 } ) } ( S ) \leq C _ { 0 } M \sqrt { s k / m } \ell _ { m } ^ { 3 / 2 } . } \end{array}
$$

Proof. If $M = 0$ , both quantities vanish. Otherwise use Lemma 6 with $D = M$ and $J = \operatorname* { m a x } \{ 1 , \lceil \log _ { 2 } \sqrt { m } \rceil \}$ }. Then $\sqrt { m } r _ { J } \leq M$ and $J \leq \ell _ { m } / \log 2$ . Since $B \leq M ,$ (7) permits

$$
H _ { j } = \ell _ { m } + 3 2 0 A ^ { 2 } \ell _ { m } / r _ { j } ^ { 2 } + \left( j + 4 \right) \log 2 .
$$

Subadditivity of the square root, $\begin{array} { r } { \sum _ { j \geq 1 } 2 ^ { - j } = 1 } \end{array}$ , and $\begin{array} { r } { \sum _ { j \geq 1 } 2 ^ { - j } \sqrt { j + 4 } \leq \sum _ { j \geq 1 } 2 ^ { - j } \big ( j + 4 \big ) = 6 } \end{array}$ give

$$
\begin{array} { r l r } {  { \mathbb { E } Q \le M + 6 M \sqrt { \ell _ { m } } + 6 \sqrt { 3 2 0 } A \sqrt { \ell _ { m } } J + 3 6 M \sqrt { \log { 2 } } } } \\ & { } & { \le C _ { 0 } M \ell _ { m } ^ { 3 / 2 } . \quad \quad } \end{array}
$$

For example, one may take

$$
C _ { 0 } = \frac { 6 \sqrt { 3 2 0 } + 6 } { \log 2 } + \frac { 1 + 3 6 \sqrt { \log 2 } } { ( \log 2 ) ^ { 3 / 2 } } .
$$

The comparison uses only $A \leq M$ and $\ell _ { m } \geq \log 2$ , including $m = 1$ . Lemma 4 yields the network bound. □

## 2.4 Large biases contribute constants

Lemma 8 (Bias clipping). Put $b _ { 0 } = \operatorname* { m i n } \{ B , A \}$ and $d = ( B - A ) .$ <sub>+</sub>. Every $h \in \mathcal { H } _ { n , s , k } ^ { W , B } ( X _ { 0 } )$ can be written on $\chi _ { 0 }$ as $h = h _ { 0 } + c ,$ , where

$$
h _ { 0 } \in \mathcal { H } _ { n , s , k } ^ { W , b _ { 0 } } ( \boldsymbol { X } _ { 0 } ) , \qquad | c | \leq k d .
$$

Proof. Use (3) and clip each $\beta _ { j }$ to $[ - A , A ]$ . For $a \in \left[ - A , A \right]$ and every $\beta \in \mathbb { R }$ , the three cases $\beta < - A$ $- A \leq \beta \leq A$ , and $\beta > A$ give

$$
\sigma ( a - \beta ) = \sigma ( a - \beta ^ { \circ } ) + ( - \beta - A ) _ { + } , \qquad \beta ^ { \circ } = \operatorname* { m a x } \{ - A , \operatorname* { m i n } \{ \beta , A \} \} .\tag{9}
$$

Clipping creates no active unit at any input in the ball. A unit with $\beta < - A$ was strictly active everywhere before clipping, and a unit with $\beta > A$ stays inactive. Hence $\begin{array} { r } { h _ { 0 } = \sum _ { j } \varepsilon _ { j } \sigma ( \langle v _ { j } , x \rangle - \beta _ { j } ^ { \circ } ) } \end{array}$ belongs to the stated class. Every nonzero summand of $\begin{array} { r } { c = \sum _ { j } \varepsilon _ { j } \bigl ( - \beta _ { j } - A \bigr ) _ { + } } \end{array}$ comes from a unit active everywhere. Because $\chi _ { 0 }$ is nonempty, there can be at most � such units, each contributing at most � in absolute value. □

## 2.5 Proof of Theorem 2

Proof. The complexity of constants in $[ - k d , k d ]$ is $k d \rho _ { m }$ . Subadditivity and Lemma 8 give the slightly sharper intermediate bound

$$
\mathcal { R } _ { \mathcal { H } _ { n , s , k } ^ { W , B } ( \chi _ { 0 } ) } ( S ) \leq ( A + b _ { 0 } ) \operatorname* { m i n } \{ k , C _ { 0 } \sqrt { s k / m } \ell _ { m } ^ { 3 / 2 } \} + k d \rho _ { m } .\tag{10}
$$

Here the two estimates for the clipped class are its pointwise envelope and Proposition 7. Using $A + b _ { 0 } \leq 2 A$ $d \leq B _ { i }$ , and $C _ { 0 } \geq 1$ proves (1) with $C = 2 C _ { 0 }$ . This also covers $A = 0 ;$ the clipped class is then zero and the original class consists exactly of constants in $[ - k B , k B ]$

The original envelope is $| h ( x ) | \leq k M$ , proving the first additional bound. For the second, ignore sparsity and bound each signed neuron separately. Since the unsigned neuron class contains zero, sign symmetry and scalar contraction imply

$$
\mathbb { E } \operatorname* { s u p } _ { \| v \| \leq W , | \beta | \leq B } \left. \sum _ { i } \zeta _ { i } \sigma ( \langle v , x _ { i } \rangle - \beta ) \right. \leq 2 \left( W \mathbb { E } \left\| \sum _ { i } \zeta _ { i } x _ { i } \right\| _ { 2 } + B \mathbb { E } \left. \sum _ { i } \zeta _ { i } \right. \right) \leq 2 M \sqrt { m } .
$$

Summing over � neurons proves $2 s M / \sqrt { m }$

## 3 Lower bounds on the same class

The lower bound uses parallel neurons within each cluster and orthogonal directions between clusters. Parallelism allows � units to each attain value � on a single input while respecting the radius constraint. Orthogonality makes the output values of diferent clusters independent parameters on the constructed support.

Lemma ${ \bf 9 } \left( { \mathrm { A } } \right.$ moment estimate). $I f Z$ has finite fourth moment and $\mathbb { E } Z ^ { 2 } > 0$ , then

$$
\mathbb { E } | Z | \geq \frac { ( \mathbb { E } Z ^ { 2 } ) ^ { 3 / 2 } } { ( \mathbb { E } Z ^ { 4 } ) ^ { 1 / 2 } } .
$$

In particular, $\mathbb { E } | \sum _ { i = 1 } ^ { m } \zeta _ { i } | \geq \sqrt { m / 3 } .$

Proof. Hölder applied to $| Z | ^ { 2 } = | Z | ^ { 2 / 3 } | Z | ^ { 4 / 3 }$ gives $\mathbb { E } Z ^ { 2 } \le ( \mathbb { E } | Z | ) ^ { 2 / 3 } ( \mathbb { E } | Z | ^ { 4 } ) ^ { 1 / 3 }$ . For the sign sum, $\mathbb { E } Z ^ { 2 } = m$ and $\mathbb { E } Z ^ { 4 } = 3 m ^ { 2 } - 2 m \le 3 m ^ { 2 }$ □

## 3.1 Proof of Theorem 3

Proof. First suppose $A > 0$ . Choose orthonormal vectors $e _ { 1 } , \ldots , e _ { q } \in \mathbb { R } ^ { n }$ and let $D _ { x }$ be uniform on $X _ { 0 } =$ $\{ R e _ { 1 } , . . . , R e _ { q } \}$ . Use � clusters of � neurons each, with weight � $\dot { \boldsymbol { e _ { b } } }$ and zero bias in cluster �. There are at most � neurons; pad with inactive zero units when necessary. On $R e _ { b }$ , only cluster � is active. Choosing a common outer sign $\theta _ { b } \in \{ - 1 , 1 \}$ in that cluster realizes $h _ { \theta } ( R e _ { b } ) = k A \theta _ { b }$

Write $X _ { i } = R e _ { J _ { i } }$ , where the $J _ { i }$ are i.i.d. uniform on [�] and independent of the Rademacher signs. For each realized sample, optimizing over the cluster signs gives

$$
\mathcal { R } _ { \mathcal { H } _ { n , s , k } ^ { W , B } ( \chi _ { 0 } ) } ( S ) \geq \frac { k A } { m } \sum _ { b = 1 } ^ { q } \mathbb { E } _ { \zeta } \left| \sum _ { i = 1 } ^ { m } \zeta _ { i } { 1 \{ J _ { i } = b \} } \right| .\tag{11}
$$

For a fixed $b ,$ set $\begin{array} { r } { Z _ { b } = \sum _ { i } \zeta _ { i } { 1 } \{ J _ { i } = b \} } \end{array}$ and $\lambda = m / q \geq 1$ . Independence and centering give

$$
\mathbb { E } Z _ { b } ^ { 2 } = \lambda , \qquad \mathbb { E } Z _ { b } ^ { 4 } = \frac { m } { q } + \frac { 3 m ( m - 1 ) } { q ^ { 2 } } \le \lambda + 3 \lambda ^ { 2 } \le 4 \lambda ^ { 2 } .
$$

Lemma 9, followed by (11), yields

$$
\mathbb { E } _ { S } \mathcal { R } _ { \mathcal { H } _ { n , s , k } ^ { W , B } ( \chi _ { 0 } ) } ( S ) \ge \frac { k A } { 2 } \sqrt { q / m } \ge \frac { A } { 2 \sqrt { 2 } } \operatorname* { m i n } \{ k , \sqrt { s k / m } \} .\tag{12}
$$

For the last step, $\lfloor s / k \rfloor \ge s / ( 2 k )$ since $s / k \geq 1$ , and hence $q \geq \frac { 1 } { 2 } \operatorname* { m i n } \{ s / k , m \}$

The same class on the same support also contains the two constants $\pm k B \colon$ use $k$ zero-weight units with bias −� and a common outer sign. Thus for every sample,

$$
\mathcal { R } _ { \mathcal { H } _ { n , s , k } ^ { W , B } ( \chi _ { 0 } ) } ( S ) \geq k B \rho _ { m } \geq \frac { k B } { \sqrt { 3 m } } .\tag{13}
$$

The two constructions are alternative subclasses; they need not fit simultaneously into one network. Taking the larger of (12) and (13) and using max $\{ a , b \} \ge ( a + b ) / 2$ proves (2). If $A = 0$ , take $X _ { 0 } = \{ 0 \}$ and use only the constant subclass; the weight term is zero. Finally, a supremum over samples is at least their expectation under this marginal. □

Interpretation of the regimes. When $m \leq s / k$ , we may use one cluster per sample-sized support point; random repetitions only change constants in the lower bound. When $m \geq s / k$ , the $\lfloor s / k \rfloor$ clusters are repeatedly observed and their signed sums have square-root fluctuations. The bias subclass has just one freely chosen sign, giving $m ^ { - 1 / 2 }$ decay independently of the number of available clusters. This explains the distinct width dependence in the upper bound. The dimensional condition is used to supply orthogonal cluster directions; it is not an assumption of Theorem $^ { 2 , }$ and no claim of sharpness for every smaller fixed dimension is made.

## 4 When does sparsity remove the width dependence?

The lower bound in Section 3 allows diferent neuron clusters to act independently on a discrete input set. Does its width dependence survive if sparsity is required throughout a full-dimensional convex domain? The answer depends on whether activation thresholds are available. Write $\mathbb { B } _ { R } ^ { n } = \{ x \in \mathbb { R } ^ { n } : \| x \| _ { 2 } \leq R \}$ and suppose $R > 0$ throughout this section.

Proposition 10 (Zero-threshold width collapse). Every zero-bias network that is �-sparse on $\mathbb { B } _ { R } ^ { n }$ has a reduced representation with at most min $\{ s , 2 k \}$ nonzero units. Consequently, for every $S \in ( \mathbb { B } _ { R } ^ { n } ) ^ { m }$

$$
\mathcal { R } _ { \mathcal { H } _ { n , s , k } ^ { W , 0 } ( \mathbb { B } _ { R } ^ { n } ) } ( S ) \leq \frac { 4 k A } { \sqrt { m } } .
$$

The supremum over such samples is at least $k A \rho _ { m }$ , so its order is $k A / \sqrt { m }$

Proof. Use the reduced parameters in (3) and discard $v _ { j } = 0$ units. Choose a unit direction outside the finitely many hyperplanes $\langle v _ { j } , x \rangle = 0$ . Each remaining unit is strictly active at exactly one of the two antipodal radius-� points. Both points lie in the required input domain, so the number of remaining units is at most 2�. The neuronwise estimate in Theorem 2 then gives the upper bound. For the lower bound, repeat $R e _ { 1 }$ and use � parallel units with weight $W e _ { 1 }$ and a common outer sign. These networks are globally �-sparse and take values $\pm k A$ on the sample. The case $W = 0$ is immediate. □

Central symmetry alone is insuficient. On the finite set $\{ \pm R e _ { b } : b \in [ q ] \}$ , the zero-bias clusters from Section 3 remain admissible: most units can vanish on a given antipodal pair. The full-ball argument uses a pair that avoids every nonzero neuron’s zero hyperplane simultaneously.

For a polytope $X _ { 0 } = \mathrm { c o n v } ( V )$ with finitely many vertices, a related count holds even with biases. Every unit active somewhere is active at a vertex, since an afine function attains its maximum there. Hence at most $k | V |$ units are nonzero on the domain, giving the neuronwise bound $2 k | V | ( A + B ) / \sqrt { m }$ . In particular, on an interval $( n = 1 )$ , the width dependence disappears for every bias bound, not only for $B = 0$

Positive thresholds can instead isolate disjoint caps of the ball. Set

$$
\tau = { \frac { \sqrt { 3 } } { 2 } } , \qquad d _ { q } = \operatorname* { m i n } \{ q , \lceil 1 6 \log ( 2 q ) \rceil \} .
$$

The following elementary packing sufices; we do not need sharp spherical-code bounds.

Lemma 11 (Separated directions). $H n \geq d _ { q }$ , there are unit vectors $u _ { 1 } , . . . , u _ { q } \in \mathbb { R } ^ { n }$ with $\left. u _ { b } , u _ { c } \right. \leq 1 / 2$ for $b \neq c .$

Proof. If $n \geq q .$ , take orthogonal vectors. Otherwise $n \geq 1 6 \log ( 2 q )$ . Take independent vectors uniform on $\{ \pm n ^ { - 1 / 2 } \} ^ { n }$ . For a fixed pair, their inner product has the law of $n ^ { - 1 } \sum _ { r = 1 } ^ { n } \xi _ { r }$ for independent uniform signs. The bound $\mathbb { E } e ^ { \lambda \sum _ { r } \xi _ { r } } \le \bar { e } ^ { n \lambda ^ { 2 } / 2 }$ gives $\mathbb { P } [ \left. u _ { b } , u _ { c } \right. > 1 / 2 ] \le e ^ { - n / 8 }$ . A union bound over pairs has probability at most $q ( q - 1 ) e ^ { - n / 8 } / 2 < 1$ . Thus a configuration with the required separation exists. □

Theorem 12 (A full-ball lower bound in logarithmic dimension). Let $q = \operatorname* { m i n } \{ \lfloor s / k \rfloor , m \}$ and suppose $n \geq d _ { q }$ Put $a = \operatorname* { m i n } \{ A , B / \tau \}$ . There is a fixed marginal $D _ { x }$ on $\mathbb { B } _ { R } ^ { n }$ such that

$$
\mathbb { E } _ { S \sim D _ { x } ^ { m } } \mathcal { R } _ { \mathcal { H } _ { n , s , k } ^ { W , B } ( \mathbb { B } _ { R } ^ { n } ) } ( S ) \geq c _ { \mathrm { c a p } } \left( a \operatorname* { m i n } \{ k , \sqrt { s k / m } \} + \frac { k B } { \sqrt { m } } \right) , \qquad c _ { \mathrm { c a p } } = \frac { 1 - \tau } { 4 \sqrt { 2 } } .\tag{14}
$$

In particular, $i f B \geq \tau A$ , the rate of Theorem 3 holds for the full-ball class itself, with only $O ( \log ( 2 q ) )$ dimensions suficient.

Proof. Choose directions from Lemma 11 and put $D _ { x }$ uniform on $\{ R u _ { 1 } , . . . , R u _ { q } \}$ . Suppose first that $a > 0$ In cluster $b ,$ place � copies of the unit

$$
v _ { b } = ( a / R ) u _ { b } , \qquad \beta _ { b } = \tau a ,
$$

with a common outer sign $\theta _ { b } .$ . These parameters obey the norm and bias bounds. If two distinct clusters were active at an input $x \in \mathbb { B } _ { R } ^ { n }$ , then

$$
\langle u _ { b } + u _ { c } , x \rangle > 2 \tau R , \qquad \| u _ { b } + u _ { c } \| _ { 2 } ^ { 2 } = 2 + 2 \langle u _ { b } , u _ { c } \rangle \leq 3 = 4 \tau ^ { 2 } ,
$$

contradicting Cauchy–Schwarz. Thus the network is �-sparse at every point of the ball, not only at the sampling locations. At $R u _ { b } .$ its value is $k a ( 1 - \tau ) \theta _ { b } ;$ all other clusters are inactive because $1 / 2 < \tau .$ . Pad with inactive units to width �.

The moment calculation in (12), with amplitude $a ( 1 - \tau )$ replacing �, now gives a lower bound of $a ( 1 - \tau ) \operatorname* { m i n } \{ k , \sqrt { s k / m } \} / ( 2 \sqrt { 2 } )$ . The same full-ball class contains the constants ±��, contributing $k B \rho _ { m }$ Taking the larger of these two subclass bounds proves (14). If $a = 0$ , the constant subclass alone proves the assertion. □

What changes between the regimes? The zero-bias proposition turns pointwise sparsity into a global bound on the number of useful units. The cap construction defeats that implication by assigning diferent clusters disjoint activation regions. It also shows that a finite-support marginal does not require a function class whose sparsity promise is restricted to that finite support. For $m \geq s / k$ , the comparison is summarized in Table 1. The value �� is suficient for the construction, not a proved critical threshold.

The logarithmic dimension order is necessary for constant-amplitude sign encoding. Indeed, every full-ball network is �� -Lipschitz: along any segment, the finitely many afine pieces have gradients bounded by ��. When $A + B > 0$ , normalization by $k ( A + B )$ gives a function that is at most 1/�-Lipschitz. If all signs on � inputs can be realized at amplitude $\alpha > 0$ , each pair of inputs must be at distance at least 2��. Disjoint open balls of radius �� around them lie inside the ball of radius $( 1 + \alpha ) R ,$ , so volume comparison gives $q \leq ( 1 + 1 / \alpha ) ^ { n }$ . This proves an $\Omega ( \log q )$ requirement when � is constant; it is a statement about this encoding, not a necessary dimension condition for every complexity lower bound.

Table 1: Worst-case empirical complexity; $\mathbf { \nabla } \cdot \boldsymbol { g } = \lfloor s / k \rfloor$ and $m \geq s / k$ . The lower bounds also hold in expectation for an appropriate i.i.d. marginal. Θe permits logarithms in $m ;$ the middle row has none.
<table><tr><td>Sparsity domain and bias Dimension Complexity</td><td></td><td></td></tr><tr><td>Finite set,  $B = 0$ </td><td> $n \geq g$ </td><td> $\widetilde { \Theta } ( A \sqrt { s k / m } )$ </td></tr><tr><td>Entire ball,  $B = 0$ </td><td> $n \geq 1$ </td><td> $\Theta ( k A / \sqrt { m } )$ </td></tr><tr><td>Entire ball,  $B \geq \tau A$ </td><td> $n \geq d _ { g }$ </td><td> $\widetilde { \Theta } ( A \sqrt { s k / m } + k B / \sqrt { m } )$ </td></tr></table>

## 5 Learning consequences

Fix the class ${ \mathcal { H } } = { \mathcal { H } } _ { n , s , k } ^ { W , B } ( X _ { 0 } )$ before drawing data. In this section $\chi _ { 0 }$ is Borel, � is a probability distribution on $\chi _ { 0 } \times \mathbb { R }$ , and the loss $\ell : \mathbb { R } \times \mathbb { R } \to [ 0 , b _ { \ell } ]$ is jointly Borel measurable and �-Lipschitz in its first argument. Write $\mathcal { L } _ { D } ( h ) = \mathbb { E } \ell ( h ( X ) , Y )$ and $\begin{array} { r } { \widehat { \mathcal { L } } _ { S } ( h ) = m ^ { - 1 } \sum _ { i } \ell ( h ( X _ { i } ) , Y _ { i } ) } \end{array}$ . Put

$$
U _ { m } = \operatorname* { m i n } \left\{ k ( A + B ) , \frac { 2 s ( A + B ) } { \sqrt { m } } , \thinspace C A \operatorname* { m i n } \{ k , \sqrt { s k / m } \ell _ { m } ^ { 3 / 2 } \} + k B \rho _ { m } \right\} .\tag{15}
$$

This is a deterministic upper bound on the empirical and expected complexity of the fixed class, for every admissible marginal.

Corollary 13 (Agnostic and realizable guarantees). For any $\eta > 0 .$ , there exists a measurable �-approximate empirical risk minimizer $\widehat { h } \in \mathcal { H }$ . For every $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta$ over $S \sim D ^ { m }$ it satisfies

$$
\mathcal { L } _ { D } ( \widehat { h } ) - \operatorname* { i n f } _ { h \in \mathcal { H } } \mathcal { L } _ { D } ( h ) \leq 4 L U _ { m } + 2 b _ { \ell } \sqrt { \frac { \log ( 2 / \delta ) } { 2 m } } + \eta .\tag{16}
$$

$I f Y = h ^ { \star } ( X )$ almost surely for some $h ^ { \star } \in { \mathcal { H } }$ and $\ell ( y , y ) = 0 .$ , then with probability at least $1 - \delta _ { : }$

$$
\mathcal { L } _ { D } ( \widehat { h } ) \leq 2 L U _ { m } + b _ { \ell } \sqrt { \frac { \log ( 1 / \delta ) } { 2 m } } + \eta .\tag{17}
$$

These statements give statistical guarantees, without an eficient optimization claim.

Proof. The parameter space in (3) with the support-wide sparsity constraint is compact. Indeed, having at most � strictly positive afine values at each fixed � is a closed condition, and arbitrary intersections of these conditions remain closed in the bounded parameter box. The finitely many outer sign vectors preserve compactness. The parameter-to-function map is continuous in the uniform norm on $X _ { 0 } ,$ , since $\left\| { \boldsymbol { x } } \right\| _ { 2 } \leq R .$ . Thus $\mathcal { H }$ is separable in that norm. Choose a fixed countable dense subclass. Lipschitz continuity of the loss makes its empirical and population infima agree with those of H. Selecting the first element of this subclass with empirical loss within � of its countable infimum defines a measurable approximate ERM. This is an existence argument, not an efective search procedure.

Symmetrization, scalar contraction, and bounded diferences $\mathrm { g i v e , }$ simultaneously for all $h \in { \mathcal { H } }$ , the one-sided bound

$$
\begin{array} { r } { \mathcal { L } _ { D } ( h ) - \widehat { \mathcal { L } } _ { S } ( h ) \leq 2 L \mathbb { E } _ { S ^ { \prime } } \mathcal { R } _ { \mathcal { H } } ( S ^ { \prime } ) + b _ { \ell } \sqrt { \log ( 1 / \delta ) / ( 2 m ) } . } \end{array}
$$

Here $S ^ { \prime }$ is an independent input sample. This is the expected-complexity form of the standard Rademacher generalization inequality (Bartlett and Mendelson, 2002; Mohri et al., 2018). The countable dense subclass justifies the suprema, and subtracting $\ell ( 0 , Y _ { i } )$ before contraction does not change expected Rademacher complexity. Apply the same argument in the other direction and use a union bound to obtain uniform absolute deviation at most $2 L U _ { m } + b _ { \ell } \sqrt { \log ( 2 / \delta ) / ( 2 m ) }$ . The approximate ERM inequality then proves (16). In the realizable case, $\widehat { \mathcal { L } } _ { S } ( \widehat { h } ) \leq \eta$ almost surely; the single one-sided event proves (17). □

## 5.1 Agnostic minimax risk for a normalized bounded loss

We now fix one loss and match upper and lower bounds in the same statistical experiment. Suppose $A > 0$ set $g = \lfloor s / k \rfloor$ , and normalize the full-ball class as

$$
\begin{array} { r } { \overline { { \mathcal { H } } } = \{ h / ( k ( A + B ) ) : h \in \mathcal { H } _ { n , s , k } ^ { W , B } ( { \mathbb { B } } _ { R } ^ { n } ) \} \subseteq [ - 1 , 1 ] ^ { { \mathbb { B } } _ { R } ^ { n } } . } \end{array}
$$

For $y \in \{ - 1 , 1 \}$ , use the bounded linear loss

$$
\begin{array} { r } { \ell _ { \mathrm { l i n } } ( t , y ) = \frac 1 2 \big ( 1 - y \mathrm { c l i p } ( t ) \big ) , \qquad \mathrm { c l i p } ( t ) = \mathrm { m a x } \{ - 1 , \mathrm { m i n } \{ t , 1 \} \} . } \end{array}
$$

It is 1/2-Lipschitz in � and takes values in [0, 1]; replacing � by $\operatorname { c l i p } ( y )$ extends it to all real labels. Let

$$
\mathfrak { E } _ { m } = \operatorname* { i n f } _ { \widehat { f } } \operatorname* { s u p } _ { D } \left( \mathbb { E } _ { S \sim D ^ { m } } \mathcal { L } _ { D } ( \widehat { f } ) - \operatorname* { i n f } _ { f \in \overline { { \mathcal { H } } } } \mathcal { L } _ { D } ( f ) \right) ,
$$

where � ranges over all Borel distributions on $\mathbb { B } _ { R } ^ { n } \times \{ - 1 , 1 \}$ . The infimum allows measurable, randomized and improper learning rules; their internal randomness is included in the expectation. The rule receives the sample, not the unknown distribution. No computation restriction is imposed.

Theorem 14 (Agnostic minimax rate). Suppose $A > 0 , \tau A \leq B \leq A$ , and $n \geq d _ { g } ,$ , where $\tau , d _ { g }$ are defined in Section 4. There are universal constants $c _ { 1 } , C _ { 1 } > 0$ such that, for all $m \geq 1$

$$
c _ { 1 } \operatorname* { m i n } \{ 1 , \sqrt { s / ( k m ) } \} \leq { \mathfrak { E } } _ { m } \leq C _ { 1 } \operatorname* { m i n } \{ 1 , \sqrt { s / ( k m ) } \log ^ { 3 / 2 } ( 2 m ) \} .\tag{18}
$$

The upper bound follows from approximate empirical risk minimization in the normalized class. For the lower bound, the spherical-cap construction realizes all independent signs on $q = \operatorname* { m i n } \{ g , m \}$ inputs with a constant normalized amplitude. Give the labels small, unknown signed means on those inputs. Adjacent sign choices have small joint-sample divergence, so no learning rule can reliably recover their correlations. Appendix B makes this testing argument explicit, including improper and randomized rules.

For this loss and model, (18) yields an expected-excess-risk sample-size order $\widetilde { \Theta } ( s / ( k \varepsilon ^ { 2 } ) )$ for suficiently small �. The ratio $s / k$ reflects the explicit output normalization by $k ( A + B )$ ; it is not an unnormalized improvement from �� to $s / k$ . The theorem is loss-specific and agnostic. It does not establish a matching lower bound for the realizable guarantee above or for every bounded Lipschitz loss.

## 6 Scope and remaining questions

Activation sparsity does not by itself identify a globally small subnetwork. The full-ball comparison makes the distinction precise: zero thresholds force at most 2� nonzero units, whereas positive thresholds can isolate many independently signed caps, even in logarithmic dimension. Meanwhile, biases larger than the linear pre-activation range add only constants. These are diferent roles of thresholds, not conflicting statements about bias complexity.

The logarithmic gap remains. Our afine cover contributes $\sqrt { \log ( 2 m ) }$ , and summing the entropy bound over scales contributes another logarithm. The finite union over activation counts has only a lower-order cost. Removing that union cost alone cannot prove the logarithm-free conjecture. A direct process estimate, or an analysis retaining the simultaneous feasibility of all neuron configurations, may avoid losses introduced by the present relaxation.

The domain results leave a more geometric question: how does complexity interpolate between the zero-bias collapse and the full-ball cap construction? The suficient value $B = ( \sqrt { 3 } / 2 ) W R$ is not shown to be a critical threshold. The constants in the spherical-code dimension estimate are not sharp. Understanding small positive thresholds and fixed low dimension would explain more than improving a universal upper bound alone.

Finally, the agnostic minimax characterization concerns one explicitly normalized bounded loss and comparable bias and weight scales. It does not settle realizable learning or every loss function. All classes here impose sparsity on a fixed input domain before data are drawn; observed training-set sparsity is not a substitute for that promise. Distribution-dependent sparsity and multilayer extensions require additional arguments.

## References

Pranjal Awasthi, Nishanth Dikkala, Pritish Kamath, and Raghu Meka. Learning neural networks with sparse activations. In Proceedings of the 37th Conference on Learning Theory, volume 247 of Proceedings of Machine Learning Research, pages 406–425. PMLR, 2024. URL https://proceedings.mlr.press/v247/awasthi24a.html.

Peter L. Bartlett and Shahar Mendelson. Rademacher and Gaussian complexities: Risk bounds and structural results. Journal ofMachine Learning Research, 3:463–482, 2002. URL https://www.jmlr.org/papers/v3/bar tlett02a.html.

Richard M. Dudley. The sizes of compact subsets of Hilbert space and continuity of Gaussian processes. Journal ofFunctional Analysis, 1(3):290–330, 1967. doi: 10.1016/0022-1236(67)90017-1.

Tomer Galanti, Mengjia Xu, Liane Galanti, and Tomaso Poggio. Norm-based generalization bounds for sparse neural networks. In Advances in Neural Information Processing Systems, volume 36, pages 42482– 42501. Curran Associates, Inc., 2023. doi: 10.52202/075280-1843. URL https://proceedings.neurips.cc/p aper\_files/paper/2023/hash/8493e190ff1bbe3837eca821190b61ff-Abstract-Conference.html.

Noah Golowich, Alexander Rakhlin, and Ohad Shamir. Size-independent sample complexity of neural networks. In Proceedings ofthe 31st Conference on Learning Theory, volume 75 of Proceedings ofMachine Learning Research, pages 297–299. PMLR, 2018. URL https://proceedings.mlr.press/v75/golowich18a.html.

Michel Ledoux and Michel Talagrand. Probability in Banach Spaces: Isoperimetry and Processes, volume 23 of Ergebnisse der Mathematik und ihrer Grenzgebiete. Springer-Verlag, 1991.

Mehryar Mohri, Afshin Rostamizadeh, and Ameet Talwalkar. Foundations of Machine Learning. MIT Press, second edition, 2018.

Ramchandran Muthukumar and Jeremias Sulam. Sparsity-aware generalization theory for deep neural networks. In Proceedings ofthe 36th Conference on Learning Theory, volume 195 of Proceedings ofMachine Learning Research, pages 5311–5342. PMLR, 2023. URL https://proceedings.mlr.press/v195/muthukumar2 3a.html.

Behnam Neyshabur, Ryota Tomioka, and Nathan Srebro. Norm-based capacity control in neural networks. In Proceedings of the 28th Conference on Learning Theory, volume 40 of Proceedings of Machine Learning Research, pages 1376–1401. PMLR, 2015. URL https://proceedings.mlr.press/v40/Neyshabur15.html.

Alain Pajor and Nicole Tomczak-Jaegermann. Subspaces of small codimension of finite-dimensional Banach spaces. Proceedings ofthe American Mathematical Society, 97(4):637–642, 1986. doi: 10.1090/S0002-9939-1 986-0845980-8

Alexandre B. Tsybakov. Introduction to Nonparametric Estimation. Springer Series in Statistics. Springer, 2009. doi: 10.1007/b13794.

Tong Zhang. Covering number bounds of certain regularized linear function classes. Journal of Machine Learning Research, 2:527–550, 2002. URL https://www.jmlr.org/papers/v2/zhang02b.html.

## A The afine covering estimate

We give the finite-dimensional argument underlying (5). It is the usual Gaussian packing proof of dual Sudakov (Pajor and Tomczak-Jaegermann, 1986; Ledoux and Talagrand, 1991), specialized to the sample norm.

Lemma 15 (Samplewise afine covering). For every radius-� sample, every � , $B \geq 0$ , and every $\delta > 0$ , the afine evaluation class satisfies (5).

Proof. We first establish the linear bound. Let $\begin{array} { r } { V \ = \ \mathrm { s p a n } \{ x _ { 1 } , . . . , x _ { m } \} , \ p ( v ) \ = \ \operatorname* { m a x } _ { i } | \langle v , x _ { i } \rangle | _ { \mathbf { \Omega } } } \end{array}$ , and $L _ { X } \ =$ max<sub>�</sub> $\| x _ { i } \| _ { 2 }$ . Orthogonal projection onto � preserves the sample evaluations and decreases the Euclidean norm. If $W = 0$ or $L _ { X } = 0$ , the linear class is a singleton. Otherwise $\boldsymbol { p }$ is a norm on �. Let � be a standard Gaussian in $V , a = \mathbb { E } p ( G ) > 0$ , and $K = \{ v : p ( v ) \leq 1 \}$ .

For any centrally symmetric Borel set $K ^ { \prime }$ and $z \in V$ , Gaussian density integration gives

$$
\gamma ( K ^ { \prime } + z ) = e ^ { - \| z \| _ { 2 } ^ { 2 } / 2 } \int _ { K ^ { \prime } } e ^ { - \langle z , y \rangle } d \gamma ( y ) \geq e ^ { - \| z \| _ { 2 } ^ { 2 } / 2 } \gamma ( K ^ { \prime } ) .\tag{19}
$$

Indeed, by symmetry the integral equals that of cosh $( \langle z , y \rangle )$ , which is at least $\gamma ( K ^ { \prime } )$ . This argument applies in the Euclidean space � with its own Gaussian density.

Fix a covering radius $r > 0 ,$ . Choose a maximal finite collection $v _ { 1 } , \dotsc , v _ { N } \in W B _ { 2 } ( V )$ with pairwise $\mathcal { P }$ -distance strictly larger than �. Such a collection exists because the ball is compact. Its closed �-balls cover the ball, and the sets $\boldsymbol { v } _ { j } + \left( \boldsymbol { r } / 2 \right) \boldsymbol { K }$ are disjoint. Put $\lambda = 4 a / r$ . The dilated sets remain disjoint, and Markov’s inequality gives $\gamma ( ( \lambda r / 2 ) K ) = \mathbb { P } [ \ p ( G ) \leq 2 a ] \geq 1 / 2$ . Using (19) and $\| v _ { j } \| _ { 2 } \leq W _ { : }$

$$
1 \geq \sum _ { j = 1 } ^ { N } \gamma ( \lambda v _ { j } + ( \lambda r / 2 ) K ) \geq \frac { N } { 2 } e ^ { - \lambda ^ { 2 } W ^ { 2 } / 2 } .
$$

It follows that log $N \leq \log 2 + 8 W ^ { 2 } a ^ { 2 } / r ^ { 2 } . \mathrm { ~ I f ~ } r \geq W L _ { X }$ , the single center 0 sufices instead. Otherwise, $a \ge \sqrt { 2 / \pi } L _ { X }$ , by choosing an input attaining $L _ { X }$ and taking the expected absolute value of its Gaussian projection. Hence log $2 \leq ( \pi \log 2 / 2 ) W ^ { 2 } a ^ { 2 } / r ^ { 2 }$ , and in both cases

$$
\log N ( W B _ { 2 } ( V ) , r , p ) \leq 1 0 W ^ { 2 } a ^ { 2 } / r ^ { 2 } .\tag{20}
$$

Each of the 2� Gaussian variables $\pm \langle G , x _ { i } \rangle$ has variance at most $R ^ { 2 }$ . The exponential-moment maximal inequality gives $a \leq R { \sqrt { 2 \log ( 2 m ) } }$ . Setting $r = \delta / 2$ in (20) bounds the logarithm of the linear covering number by $8 0 A ^ { 2 } \log ( 2 m ) / \delta ^ { 2 }$ . Finally, a grid at spacing $\delta / 2$ , starting at −� and ending at the last such grid point not exceeding $B ,$ covers $[ - B , B ]$ at radius $\delta / 2$ using at most $\lfloor 4 B / \delta \rfloor + 1$ points. Adding this grid to the linear cover makes a �-cover of afine evaluations. Cardinalities multiply, proving (5). □

Measurability and zero scales. For each fixed sample and �, the set of $( v , \beta )$ with at most � positive sample pre-activations is closed in the compact parameter box: the violating set is the finite union of conditions under which $t + 1$ specified coordinates are strictly positive. Its continuous image $\mathcal { F } _ { t }$ is compact and contains zero. The normalized union � in Section 2 is a finite union of compact sets. Its suprema are finite; the sign probability space is itself finite. The proof handles $M = 0$ before invoking a positive covering radius and chooses at least one chaining level even when $m = 1$

## B Proof of the agnostic minimax bound

We prove Theorem 14 for its normalized class and bounded linear loss.

Upper bound. Use a measurable �-approximate empirical risk minimizer in $\overline { { \mathcal { H } } }$ , whose existence follows from the compactness argument for Corollary 13. The class is symmetric and contains zero. On its prediction range, $\ell _ { \mathrm { l i n } } ( f ( x ) , y ) - 1 / 2 = - y f ( x ) / 2$ . Symmetrization and sign symmetry therefore give

$$
\mathbb { E } \operatorname* { s u p } _ { f \in \overline { { \mathcal { H } } } } | \mathcal { L } _ { D } ( f ) - \widehat { \mathcal { L } } _ { S } ( f ) | \leq \mathbb { E } _ { S ^ { \prime } } \mathcal { R } _ { \overline { { \mathcal { H } } } } ( S ^ { \prime } ) \leq \frac { U _ { m } } { k ( A + B ) } .
$$

Multiplying the Rademacher signs by the labels preserves their conditional law; the loss’s factor $1 / 2$ cancels the symmetrization factor 2. Approximate ERM thus has expected excess risk at most $2 U _ { m } / ( k ( A + B ) ) + \eta$ uniformly in �. Using (15), $s \geq k$ , and $\rho _ { m } \leq m ^ { - 1 / 2 }$ , and taking the better of this bound and the trivial bound 1, proves the upper inequality in (18). Letting $\eta \downarrow 0$ is legitimate in the infimum over learning rules; no exact measurable minimizer is needed.

Lower bound: one fixed family before sampling. Put $q = \operatorname* { m i n } \{ g , m \}$ . Since $d _ { q } \leq d _ { g }$ , choose $q$ cap directions as in Theorem 12. Its construction with $a = A$ yields functions $f _ { \theta } \in \overline { { \mathcal { H } } }$ such that

$$
f _ { \theta } ( R u _ { b } ) = \alpha \theta _ { b } , \qquad \theta \in \{ - 1 , 1 \} ^ { q } , \qquad \alpha = \frac { ( 1 - \tau ) A } { A + B } \geq \frac { 1 - \tau } { 2 } .
$$

Set $\delta _ { 0 } = ( \alpha / 4 ) \sqrt { q / m } \le 1 / 4$ . For each fixed �, define $D _ { \theta }$ by drawing � uniformly from the cap centers and setting

$$
\mathbb { P } _ { \theta } [ Y = 1 \mid X = R u _ { b } ] = \frac { 1 + \delta _ { 0 } \theta _ { b } } { 2 } .
$$

This family is fixed before sampling. Let $P _ { \theta } = D _ { \theta } ^ { m }$ , and let $\theta ^ { ( b ) }$ flip coordinate �. The common input marginal gives

$$
\begin{array} { l l l } { \displaystyle \mathrm { K L } ( P _ { \theta } | | P _ { \theta ^ { ( b ) } } ) = \frac { m } { q } \delta _ { 0 } \log \frac { 1 + \delta _ { 0 } } { 1 - \delta _ { 0 } } } \\ { \displaystyle \qquad \leq \frac { 4 m \delta _ { 0 } ^ { 2 } } { q } . } \end{array}
$$

Here log( $( 1 + t ) / ( 1 - t ) ) \leq 4 t$ for $0 \leq t \leq 1 / 2$ , for example by integrating $2 / ( 1 - t ^ { 2 } ) \leq 4$ . Pinsker’s inequality implies

$$
\mathrm { T V } ( P _ { \theta } , P _ { \theta ^ { ( b ) } } ) \leq \delta _ { 0 } \sqrt { 2 m / q } = \frac { \alpha } { 2 \sqrt { 2 } } \leq \frac { \alpha } { 2 } .
$$

Fix any learning rule, including an improper or randomized one, and write $T _ { b } = \mathrm { c l i p } ( \widehat { f } ( R u _ { b } ) ) \in [ - 1 , 1 ]$ Averaging over adjacent pairs and using the bounded-observable characterization of total variation,

$$
2 ^ { - q } \sum _ { \theta } \theta _ { b } \mathbb { E } _ { \theta } T _ { b } \leq \frac { \alpha } { 2 } .
$$

For deterministic rules, pair each � with $\theta _ { b } = 1$ with its flipped neighbor and use $( \mathbb { E } _ { \theta } T _ { b } - \mathbb { E } _ { \theta ^ { ( b ) } } T _ { b } ) / 2 \le$ $\mathrm { T V } \big ( P _ { \theta } , P _ { \theta ^ { ( b ) } } \big )$ . Randomized post-processing cannot increase total variation, so the same inequality applies.

The comparator $f _ { \theta }$ has risk $1 / 2 - \delta _ { 0 } \alpha / 2$ . Thus the average, over this fixed family, of the learning rule’s expected excess risk is at least

$$
\begin{array} { r l } & { 2 ^ { - q } \displaystyle \sum _ { \theta } \left( \mathbb { E } _ { \theta } \mathcal { L } _ { D _ { \theta } } ( \widehat f ) - \operatorname* { i n f } _ { f \in \overline { { \mathcal { H } } } } \mathcal { L } _ { D _ { \theta } } ( f ) \right) \geq \frac { \delta _ { 0 } \alpha } { 2 } - \frac { \delta _ { 0 } } { 2 q } \displaystyle \sum _ { b = 1 } ^ { q } 2 ^ { - q } \sum _ { \theta } \theta _ { b } \mathbb { E } _ { \theta } T _ { b } } \\ & { \qquad \geq \frac { \delta _ { 0 } \alpha } { 4 } = \frac { \alpha ^ { 2 } } { 1 6 } \sqrt { q / m } . } \end{array}
$$

A supremum over � is at least this average. Finally, $\begin{array} { r } { q \ge \frac { 1 } { 2 } } \end{array}$ min $\{ s / k , m \}$ and $\alpha \geq ( 1 - \tau ) / 2 .$ , so one may take $c _ { 1 } = ( 1 - \tau ) ^ { 2 } / ( 6 4 \sqrt { 2 } )$ . This Assouad-type argument (Tsybakov, 2009, Chapter 2) directly bounds excess risk for the full-ball class, rather than converting a complexity lower bound.

## AI Disclosure

The original manuscript, including its activation-budget reduction and margin-shifted covering argument, was written by the authors. During subsequent revisions, we used OpenAI Codex, including GPT-5.6 Solar and GPT-6 Astra, to assist with proof checking, mathematical revisions, literature lookup, and exposition.

Specifically, the AI-assisted revisions replaced the per-scale concentration and dyadic peeling steps with a single chaining argument over the union of normalized single-neuron classes (Section 2); introduced the threshold-clipping decomposition that isolates the constant contribution and removes the width and logarithmic factors from the bias term (Lemma 8); extended the clustered lower-bound construction to i.i.d. samples without divisibility assumptions (Theorem 3); added the zero-bias width-collapse result and the disjoint-cap lower bound on the entire ball in logarithmic dimension (Section 4); and reformulated the learning lower bound as an Assouad-type argument for the normalized class under a specified bounded linear loss, so that the learning upper and lower bounds concern the same statistical model (Theorem 14). AI tools also assisted with edge-case checks, finite-instance verification scripts, and expansion of the technique overview. We take full responsibility for all content in the final manuscript, including AI-assisted material, and for its correctness, originality, attribution, and citations.