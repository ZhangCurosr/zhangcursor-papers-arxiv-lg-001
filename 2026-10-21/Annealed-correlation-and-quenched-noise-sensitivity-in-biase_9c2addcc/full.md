# Annealed correlation and quenched noise sensitivity in biased deep neural networks

Johan Jonasson<sup>\*†</sup>

October 21, 2026

## Abstract

In [1], fully connected deep neural networks with layers of width n, i.i.d. standard Gaussian weights, sign activations and no bias terms were shown to be annealed and quenched noise sensitive with respect to any joint probability distribution Q<sup>(n)</sup> on the input ω and its perturbation η satisfying a mild positive-correlation condition and Q<sup>(n)</sup>(η ∈ {ω, −ω}) → 0. Here we add independent Gaussian biases of the natural order and show that quenched noise sensitivity is retained over a broad range of depths, indeed under the slightly weaker assumption that Q<sup>(n)</sup>(ω = η) → 0. Thus, subject to the positive-correlation condition, quenched noise sensitivity holds under the minimal requirement that the perturbed input is asymptotically unlikely to coincide with the original input. At the same time, the annealed covariance remains bounded away from zero and is entirely accounted for by fluctuations in the network-dependent output means.

## 1 Introduction

This paper addresses a question left open in [1]. The driving force there was to investigate how sensitive binary deep neural net (DNN) classifiers are to input noise, i.e. if small changes in a typical input make the classifier change its mind. It is a well-known observation that neural nets used in practice are typically sensitive to such perturbations. For example a DNN that distinguishes cars from aeroplanes can be very certain that an image that clearly is of a car is in fact a car, but after well-chosen perturbations indistinguishable to the human eye, the DNN suddenly believes that the image is an aeroplane.

The simplest model studied in [1], and, with some modifications, also the model considered here, is a fully connected network in which every hidden layer has width n and the depth is $T _ { n }$ . Asymptotics when n grows to infinity will be in focus. The network weights $\Theta _ { t , i , j }$ are assumed to be i.i.d. standard Gaussian. Unlike in [1], each node will also have a bias term that is Gaussian with mean 0. The given weights constitute a typical initialization before training with data starts. The input will always be a binary string and the output activation will always be the sign function. In practice this activation is seldom used because it makes training hard, but could be argued to be the most natural one if such considerations were not needed.

To make the setup precise, let $\Theta = \{ \Theta _ { 1 } , \dots , \Theta _ { T _ { n } } \}$ be i.i.d. random $n \times n$ matrices, where in each $\Theta _ { t }$ , the elements $\Theta _ { t } ( i , j )$ are i.i.d. standard Gaussian. Let $B \ = \ ( \mathbf { b } _ { 1 } , \ldots , \mathbf { b } _ { T _ { n } } )$ be independent Gaussian n-vectors with mean 0 and covariance $c _ { n } n I ,$ where $c _ { n }$ is assumed to satisfy $1 / n \ \leq \ c _ { n } \ \leq \ \bar { c } \ < \ \infty$ . Let $\omega = \omega _ { 0 } \in \{ - 1 , 1 \} ^ { n }$ be the input. Then for $t = 1 , \ldots , T _ { n }$ , let

$$
\omega _ { t } = \mathrm { s i g n } ( \Theta _ { t } \omega _ { t - 1 } + \mathbf { b } _ { t } ) ,
$$

where the sign operator is taken pointwise. Finally, writing $\Phi = ( \Theta , B , \theta , b )$ , let the output of the DNN be

$$
f _ { \Phi } ^ { ( n , T _ { n } ) } ( \omega ) = \mathrm { s i g n } ( \pmb { \theta } ^ { \top } \omega _ { T _ { n } } + b ) ,
$$

where $\pmb \theta \sim N ( 0 , I _ { n } )$ and $b \sim N ( 0 , c _ { n } n )$ . (i.e. $f _ { n , T _ { n } }$ is constructed by feeding the penultimate vector $\omega _ { T _ { n } }$ into another independent neuron of the same kind as all the other neurons before it). We always assume that n is odd. Denote the law of $f _ { \Phi } ^ { ( n , T _ { n } ) } \log \mathbb { P } _ { f } ^ { ( n ) }$

In [1], it was shown that for $c _ { n } = 0$ , the sequence $\left\{ f _ { \Phi } ^ { ( n , T _ { n } ) } \right\}$ is noise sensitive in a very strong sense: It is annealed noise sensitive with respect to any joint distribution $\mathbb { Q } ^ { ( n ) } = \mathbb { P } _ { \omega , \eta } ^ { ( n ) }$ on the input ω and its perturbation η for which these are whp not identical or antipodal: $\mathbb { Q } ^ { ( n ) } ( \pmb { \eta } \in \{ \omega , - \omega \} )  0$ , i.e. for such $\mathbb { Q } ^ { ( n ) }$

$$
\operatorname* { l i m } _ { n \to \infty } \mathbb { C } \mathrm { o v } _ { ( f , ( \omega , \eta ) ) \sim \mathbb { P } _ { f } ^ { ( n ) } \times \mathbb { Q } ^ { ( n ) } } ( f ( \omega ) , f ( \eta ) ) = 0 .
$$

If $\mathbb { Q } ^ { ( n ) }$ also satisfies the positive-correlation condition

$$
\mathbb { C } \mathrm { o v } _ { ( \omega , \pmb { \eta } ) \sim \mathbb { Q } ^ { ( n ) } } ( h ( \pmb { \omega } ) , h ( \pmb { \eta } ) ) \ge 0
$$

for all Boolean functions $h ,$ , then this entails that $\left\{ f _ { \Phi } ^ { ( n , T _ { n } ) } \right\}$ is also quenched noise sensitive with respect to $\mathbb { Q } ^ { ( n ) }$ : for any $\delta > 0$ , there exists $N \ < \ \infty$ such that

whenever $n \geq N$

$$
\mathbb { P } _ { f } ^ { ( n ) } \left( \mathbb { C } \mathrm { o v } _ { \mathbb { Q } ^ { ( n ) } } ( f ( \omega ) , f ( \pmb { \eta } ) ) > \delta \right) < \delta .
$$

(This quenched noise-sensitivity result was in fact not explicitly stated in [1], but is immediate from what was done there.) The positive-correlation condition on $\mathbb { Q } ^ { ( n ) }$ holds very generally, for example if $\mathbb { Q } ^ { ( n ) }$ is symmetric and the (automatically reversible) Markov kernel $K$ given by

$$
K ( \mathbf { x } , \mathbf { y } ) = \mathbb { Q } ^ { ( n ) } ( \pmb { \eta } = \mathbf { y } \vert \pmb { \omega } = \mathbf { x } )
$$

has only nonnegative eigenvalues, which immediately gives $\mathbb { C } \mathrm { o v } _ { \mathbb { Q } ^ { ( n ) } } ( h ( \omega ) , h ( \pmb { \eta } ) ) \ge$ 0. A particular class of such distributions is given by taking $( \omega , \eta )$ as consecutive states of any lazy reversible Markov chain on $\{ - 1 , 1 \} ^ { n }$ at stationarity. Another such class is given by taking states two steps apart for any reversible stationary Markov chain on $\{ - 1 , 1 \} ^ { n }$

In other words, [1] established noise sensitivity for $c _ { n } = 0$ for a very broad class of input and perturbation distributions. As we shall now see, when adding bias vectors to the model as above, quenched noise sensitivity remains for $T _ { n }$ in a window $[ K _ { n } , e ^ { \rho { \sqrt { n } } } ]$ for $K _ { n } /$ log log $n  \infty$ and some $\rho > 0$ . Indeed it does so in an even stronger form: We retain the positive-correlation assumption on $\mathbb { Q } ^ { ( n ) }$ but, in the biased setting considered below, replace the non-antipodal condition by the weaker assumption $\mathbb { Q } ^ { ( n ) } ( \omega = \pmb { \eta } )  0 \mathrm { ~ a s ~ } n  \infty$ . The exclusion of antipodal input pairs is no longer required because the bias destroys the absorbing antipodal state.

We write $\mathbb { P } _ { \omega } ^ { ( n ) }$ and $\mathbb { P } _ { \eta } ^ { ( n ) }$ for the two marginals of $\mathbb { Q } ^ { ( n ) }$ . Before stating the main result, we need the following.

## Lemma 1.1 Consider

$$
g _ { c } ( x ) = \frac { 2 } { \pi } \arcsin \sqrt { \frac { x } { 1 + c } } , 0 \leq c \leq \bar { c } , 0 \leq x \leq 1 .
$$

Then $g _ { c }$ has a unique fixed point $v _ { c }$ in $( 0 , 1 )$ , for which $\frac { 4 } { \pi ^ { 2 } ( 1 + c ) } \leq v _ { c } \leq 1 / 2$ , and for any sufficiently small ϵ, there exists α $> 0$ such that $g _ { c } ( x ) \geq x +$ α on $[ \epsilon , v _ { c } - \epsilon ]$ and $g _ { c } ( x ) \leq x - \alpha$ on $[ v _ { c } + \epsilon , 1 - \epsilon ]$ uniformly over $c \in [ 0 , \bar { c } ]$

Theorem 1.2 Let $\mathbb { Q } ^ { ( n ) }$ and $\Phi \ : = \ : ( \Theta , B , \pmb { \theta } , b )$ be as above, and let $g _ { c }$ be as in Lemma 1.1. Assume that $K _ { n } / \log \log n \to \infty$ . Then there exists $\rho > 0$ such that whenever

$$
K _ { n } \leq T _ { n } \leq e ^ { \rho { \sqrt { n } } } ,
$$

and $\mathbb { P } _ { f } ^ { ( n ) }$ is the resulting law of $f _ { \Phi } ^ { ( n , T _ { n } ) }$ , thefollowing conclusions hold.

(i) Quenched noise sensitivity: We have

$$
\mathbb { C } \mathrm { o v } _ { \mathbb { Q } ^ { ( n ) } } ( f ( \omega ) , f ( \pmb { \eta } ) ) \xrightarrow { \mathbb { P } _ { f } ^ { ( n ) } } 0
$$

$a s n  \infty ,$ , in particular $\{ f _ { \Phi } ^ { ( n , T _ { n } ) } \}$ is quenched noise sensitive with respect to $\mathbb { Q } ^ { ( n ) }$

(ii) Annealed correlation: $H c _ { n } = c > 0$ , then, with $v _ { c }$ being the unique fixed point of $g _ { c } ,$

$$
\begin{array} { r l } & { \underset { n  \infty } { \operatorname* { l i m } } \mathbb { C } \mathrm { o v } _ { \mathbb { P } _ { f } ^ { ( n ) } \times \mathbb { Q } ^ { ( n ) } } ( f ( \omega ) , f ( \eta ) ) = \underset { n  \infty } { \operatorname* { l i m } } \mathbb { C } \mathrm { o v } _ { \mathbb { P } _ { f } ^ { ( n ) } } ( \mathbb { E } _ { \mathbb { Q } ^ { ( n ) } } [ f ( \omega ) ] , \mathbb { E } _ { \mathbb { Q } ^ { ( n ) } } [ f ( \eta ) ] ) } \\ & { \qquad = 1 - 2 v _ { c } . } \end{array}
$$

(iii) Variance of the conditional output mean: $H ,$ in addition to the conditions in (ii), $\begin{array} { r } { \sum _ { \mathbf { x } \in \{ - 1 , 1 \} ^ { n } } \mathbb { P } _ { \omega } ^ { ( n ) } ( \mathbf { x } ) ^ { 2 } \to 0 } \end{array}$ as $n  \infty ,$ , then

$$
\operatorname* { l i m } _ { n  \infty } \mathbb { V } \mathrm { a r } _ { \mathbb { P } _ { f } ^ { ( n ) } } ( \mathbb { E } _ { \mathbb { P } _ { \omega } ^ { ( n ) } } [ f ( \omega ) ] ) = 1 - 2 v _ { c } .
$$

## 2 Proofs

The following lemma is a different formulation of Lemma 3.1 in [1].

Lemma 2.1 Let X and Y be independent Gaussian random variables with mean 0 and variances a and b respectively. Then

$$
\mathbb { P } ( \operatorname { s i g n } ( X - Y ) \neq \operatorname { s i g n } ( X + Y ) ) = { \frac { 2 } { \pi } } \arcsin { \sqrt { \frac { b } { a + b } } } .
$$

A consequence of Lemma 2.1 is that if ω and η are two bit strings of dimension n that differ at nx coordinates, then

$$
\mathbb { P } ( \mathrm { s i g n } ( ( \Theta _ { t } ) _ { i , \cdot } \omega + b _ { t } ( i ) ) \neq \mathrm { s i g n } ( ( \Theta _ { t } ) _ { i , \cdot } \eta + b _ { t } ( i ) ) ) = \frac { 2 } { \pi } \arcsin \sqrt { \frac { x } { 1 + c _ { n } } } = g _ { c _ { n } } ( x )
$$

for $g _ { c }$ as in Lemma 1.1, where $( \Theta _ { t } ) _ { i , }$ <sub>·</sub> denotes the ith row of $\Theta _ { t }$ .

Proof of Lemma 1.1: It is obvious that $g _ { c }$ is nonnegative, increasing, and of order $\sqrt { x }$ as $x \to 0$ . Also,

$$
1 - g _ { c } ( 1 - x ) \geq g _ { c } ( x ) \vee { \frac { 2 } { \pi } } \arctan \sqrt { c } ,
$$

where the first inequality follows from $g _ { c } ( x ) + g _ { c } ( 1 - x ) \leq 1$ . Moreover,

$$
g _ { c } ^ { \prime \prime } ( x ) = - \frac { 1 + c - 2 x } { 2 \pi ( x ( 1 + c - x ) ) ^ { 3 / 2 } } ,
$$

which shows that $h _ { c } ( x ) : = g _ { c } ( x ) - x$ is concave on $[ 0 , a ]$ and convex on $[ a , 1 ]$ where $a = 1 \wedge ( 1 + c ) / 2$ . Let

$$
0 < \epsilon < \operatorname* { m i n } \left\{ \frac { 2 } { \pi ^ { 2 } ( 1 + \bar { c } ) } , \frac { 1 } { 4 } \right\} .
$$

Then

$$
- h _ { c } ( 1 - \epsilon ) \ge h _ { c } ( \epsilon ) \ge \frac { 2 } { \pi } \sqrt { \frac { \epsilon } { 1 + c } } - \epsilon \ge \frac { 2 } { \pi } \sqrt { \frac { \epsilon } { 1 + \bar { c } } } - \epsilon > 0 .
$$

We claim that $h _ { c }$ has a unique zero $v _ { c } \in ( 0 , 1 )$ , which satisfies $v _ { c } \in [ \epsilon , 1 - \epsilon ] ,$ $h _ { c } > 0$ on $( 0 , v _ { c } )$ , and $h _ { c } < 0 \mathrm { o n } ( v _ { c } , 1 )$ . That $v _ { c } \in [ \epsilon , 1 - \epsilon ]$ follows from the inequalities displayed. To see the other two conclusions, use that $h _ { c } ( a ) \geq 0$ or $h _ { c } ( a ) < 0$ . If the former holds, then concavity and $h _ { c } ( \epsilon ) > 0$ prove that $h _ { c }$ has no zero on $[ \epsilon , a )$ , and convexity and $h _ { c } ( 1 - \epsilon ) < 0$ prove that there is a unique zero in $[ a , 1 - \epsilon ]$ . If the latter holds, concavity and $h _ { c } ( \epsilon ) > 0$ prove a unique zero in $[ \epsilon , a ]$ and convexity and $h _ { c } ( 1 - \epsilon ) < 0$ prove the absence of a zero in $[ a , 1 - \epsilon ]$

Knowing that $v _ { c }$ is unique and using that arcsin $u \geq$ u and $g _ { c } ( 1 / 2 ) \leq g _ { 0 } ( 1 / 2 ) =$ $1 / 2$ , it is now easy to show that $\frac { 4 } { \pi ^ { 2 } ( 1 + c ) } \leq v _ { c } \leq 1 / 2$ . This also means, by the choice of $\epsilon ,$ that the two intervals $[ \epsilon , v _ { c } - \epsilon ]$ and $\left[ v _ { c } + \epsilon , 1 - \epsilon \right]$ are nonempty. Since $v _ { c }$ is the unique solution of $\Psi ( x ) = c$ for a continuous strictly decreasing function $\Psi , v _ { c }$ is continuous in $c .$ Hence the sets

$$
A _ { - } = \{ ( c , x ) : 0 \leq c \leq \bar { c } , \epsilon \leq x \leq v _ { c } - \epsilon \}
$$

and

$$
A _ { + } = \{ ( c , x ) : 0 \leq c \leq \bar { c } , v _ { c } + \epsilon \leq x \leq 1 - \epsilon \}
$$

are compact. Moreover, $h _ { c } ( x ) > 0$ on A<sub>−</sub> and $h _ { c } ( x ) < 0$ on $A _ { + }$ . Hence

$$
\alpha : = \operatorname* { m i n } \left\{ \operatorname* { m i n } _ { A _ { - } } h _ { c } ( x ) , \operatorname* { m i n } _ { A _ { + } } ( - h _ { c } ( x ) ) \right\} > 0 ,
$$

and $g _ { c } ( x ) \geq x +$ α on $[ \epsilon , v _ { c } - \epsilon ]$ and $g _ { c } ( x ) \leq x - \alpha$ on $[ v _ { c } + \epsilon , 1 - \epsilon ]$ uniformly over $c \in [ 0 , \bar { c } ]$ ✷

Proof of Theorem 1.2: Let $\omega = \omega _ { 0 }$ and $\eta = \eta _ { 0 }$ be two inputs into the DNN $f _ { \Phi } ^ { ( n , T _ { n } ) }$ defined above. Let $W _ { t }$ be the set of coordinates where $\omega _ { t }$ and $\eta _ { t }$ differ, and let $D _ { t } = | W _ { t } | / n$ . Conditionally on $D _ { t - 1 } = x$ , the disagreement indicators in layer

t are i.i.d. Bernoulli with parameter $g _ { c _ { n } } ( x )$ , so that in particular $W _ { t }$ is invariant under permutations and $n D _ { t } \mid D _ { t - 1 } = x \sim B i n ( n , g _ { c _ { n } } ( x ) )$ .

On $[ 1 / n , \epsilon ] { . }$ , a lower bound $g _ { c _ { n } } ( x ) \geq 2 C \sqrt { x }$ applies uniformly. By standard Chernoff bounds, there then exists $\rho > 0$ such that for $x \in [ 1 / n , \epsilon ]$

$$
\mathbb { P } _ { f } ^ { ( n ) } ( D _ { t + 1 } \leq C \sqrt { x } \mid D _ { t } = x ) \leq e ^ { - 3 \rho \sqrt { n } } .
$$

Hence, for n large and $0 ~ < ~ D _ { 0 } ~ < ~ \epsilon , ~ D _ { t }$ exceeds ϵ within $O ( \log \log n )$ steps with probability at least $1 - e ^ { - 2 \rho { \sqrt { n } } }$ , by a union bound over layers. By identical arguments, if $D _ { 0 } > 1 - \epsilon$ , then $D _ { t }$ goes below $1 - \epsilon$ within time $O ( \log \log n )$ with probability at least $1 - e ^ { - 2 \rho { \sqrt { n } } }$

Using further Chernoff bounds, union bounds over layers, and Lemma 1.1, it follows that for $t _ { 0 } ( n ) / \log \log n \to \infty$ and $t _ { 1 } ( n ) = e ^ { \rho { \sqrt { n } } }$ and sufficiently small $\varphi > 0$

$$
\mathbb { P } _ { f } ^ { ( n ) } \left( \exists t \in [ t _ { 0 } ( n ) , t _ { 1 } ( n ) ] : D _ { t } \notin \left( ( 1 - \varphi ) v _ { c _ { n } } , ( 1 + \varphi ) v _ { c _ { n } } \right) \right) < e ^ { - \rho \sqrt { n } } .
$$

Hence for $T _ { n } \in [ t _ { 0 } ( n ) , t _ { 1 } ( n ) ]$

$$
\mathbb { P } _ { f } ^ { ( n ) } \left( D _ { T _ { n } } \notin ( 1 \pm \varphi ) v _ { c _ { n } } \right) < e ^ { - \rho \sqrt { n } } .
$$

From this we then get for sufficiently large n

$$
\begin{array} { r l } & { \mathbb { P } _ { f } ^ { ( n ) } ( f ( \omega ) \neq f ( \eta ) ) = \mathbb { E } _ { \mathbb { P } _ { f } ^ { ( n ) } } \left[ \mathbb { P } _ { f } ^ { ( n ) } ( f ( \omega ) \neq f ( \eta ) \mid D _ { T _ { n } } ) \right] } \\ & { \quad \quad \quad \quad \in [ - e ^ { - \rho \sqrt { n } } + g _ { c _ { n } } ( ( 1 - \varphi ) v _ { c _ { n } } ) , e ^ { - \rho \sqrt { n } } + g _ { c _ { n } } ( ( 1 + \varphi ) v _ { c _ { n } } ) ] } \\ & { \quad \quad \quad \in ( 1 \pm \varphi ) v _ { c _ { n } } . } \end{array}
$$

By the symmetry of $\Phi , \mathbb { E } _ { \mathbb { P } _ { f } ^ { ( n ) } } [ f ( \omega ) ] = 0$ , and it follows that

$$
\mathbb { C } \mathrm { o v } _ { \mathbb { P } _ { f } ^ { ( n ) } } ( f ( \omega ) , f ( \eta ) ) \in 1 - ( 1 \pm \varphi ) 2 v _ { c _ { n } } .
$$

Since $\mathbb { Q } ^ { ( n ) } ( \omega = \pmb { \eta } )  0$ , it also follows that for sufficiently large n,

$$
( \mathbb { P } _ { f } ^ { ( n ) } \times \mathbb { Q } ^ { ( n ) } ) \left( f ( \omega ) \neq f ( \pmb { \eta } ) \right) \in ( 1 \pm 2 \varphi ) v _ { c _ { n } }
$$

and

$$
\mathbb { C } \mathrm { o v } _ { \mathbb { P } _ { f } ^ { ( n ) } \times \mathbb { Q } ^ { ( n ) } } \left( f ( \omega ) , f ( \pmb { \eta } ) \right) \in 1 - ( 1 \pm 2 \varphi ) 2 v _ { c _ { n } } .
$$

Next observe that

$$
\begin{array} { r l } { \mathbb { C } \mathrm { o v } _ { \mathbb { P } _ { f } ^ { ( n ) } } \left( \mathbb { E } _ { \mathbb { Q } ^ { ( n ) } } \left[ f ( \omega ) \right] , \mathbb { E } _ { \mathbb { Q } ^ { ( n ) } } \left[ f ( \eta ) \right] \right) = } & { \displaystyle \sum _ { \mathbf { x } , \mathbf { y } \in \{ - 1 , 1 \} ^ { n } } \mathbb { P } _ { \omega } ^ { ( n ) } ( \mathbf { x } ) \mathbb { P } _ { \eta } ^ { ( n ) } ( \mathbf { y } ) \mathbb { C } \mathrm { o v } _ { \mathbb { P } _ { f } ^ { ( n ) } } ( f ( \mathbf { x } ) , f ( \mathbf { y } ) ) } \\ & { \qquad \geq 1 - ( 1 + \varphi ) 2 v _ { c _ { n } } . } \end{array}
$$

The same calculation, replacing $\mathbb { P } _ { \eta } ^ { ( n ) }$ with $\mathbb { P } _ { \omega } ^ { ( n ) }$ and under the additional assumption $\begin{array} { r } { \sum _ { \mathbf { x } \in \{ - 1 , 1 \} ^ { n } } \mathbb { P } _ { \omega } ^ { ( n ) } ( \mathbf { x } ) ^ { 2 } \to 0 } \end{array}$ , gives

$$
\mathbb { V } \mathrm { a r } _ { \mathbb { P } _ { f } ^ { ( n ) } } \left( \mathbb { E } _ { \mathbb { P } _ { \omega } ^ { ( n ) } } \left[ f ( \omega ) \right] \right) \in 1 - ( 1 \pm 2 \varphi ) 2 v _ { c _ { n } } ,
$$

proving (iii).

By the positive-correlation condition on $\mathbb { Q }$ and the covariance decomposition formula,

$$
\begin{array} { r l } & { 0 \leq \mathbb { E } _ { \mathbb { P } _ { f } ^ { ( n ) } } \left[ \mathbb { C } \mathrm { o v } _ { \mathbb { Q } ^ { ( n ) } } ( f ( \omega ) , f ( \eta ) ) \right] } \\ & { \quad = \mathbb { C } \mathrm { o v } _ { \mathbb { P } _ { f } ^ { ( n ) } \times \mathbb { Q } ^ { ( n ) } } \left( f ( \omega ) , f ( \eta ) \right) - \mathbb { C } \mathrm { o v } _ { \mathbb { P } _ { f } ^ { ( n ) } } \left( \mathbb { E } _ { \mathbb { Q } ^ { ( n ) } } \left[ f ( \omega ) \right] , \mathbb { E } _ { \mathbb { Q } ^ { ( n ) } } \left[ f ( \eta ) \right] \right) } \\ & { \quad \leq 3 \varphi . } \end{array}
$$

By Markov’s inequality, it follows for any $\delta > 0$ , by taking $\varphi > 0$ sufficiently small and n sufficiently large,

$$
\mathbb { P } _ { f } ^ { ( n ) } \left( \mathbb { C } \mathrm { o v } _ { \mathbb { Q } ^ { ( n ) } } ( f ( \omega ) , f ( \pmb { \eta } ) ) > \delta \right) < \delta .
$$

This proves (i), and in combination with the previous lower bound on the second term, it also proves (ii). ✷

## References

[1] J. Jonasson, J. E. Steif and O. Zetterqvist, Noise sensitivity and stability of deep neural networks for binary classification, Stochastic Processes and their Applications 165 (2023), 130-167.