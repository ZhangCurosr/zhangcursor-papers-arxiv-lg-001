# CONVERGENCE RATES FOR THE RMSPROP OPTIMIZER WITH FULL CONTROL OF THE HYPERPARAMETERS

STEFFEN DEREICH AND ARNULF JENTZEN

Abstract. Stochastic gradient descent (SGD) optimization methods are the standard methods to train artificial intelligence (AI) systems. Often the standard SGD method is not the used training algorithm but instead one considers suitable adaptive variants of SGD in which the learning rates are for each network parameter separately adjusted adaptively during the training process. Popular adaptive SGD methods include the RMSprop, the Adam, and the AdamW optimizers, where the adaptivity parts in Adam and AdamW basically just coincide with RMSprop.

Such adaptive methods involve several hyperparameters including the regularization parameter ε (which ensures that one does not divide by zero during the training process and is often chosen to be very close to zero such as $1 0 ^ { - 8 }$ in PyTorch by default) as well as the second moment decay parameter β (which is often chosen to be very close to 1 such as 0.99 (RMSprop) and 0.999 (Adam and AdamW) in PyTorch by default). Despite the extensive literature on the theoretical analysis of such adaptive methods, it remains an open research problem to provide error estimates for such methods with the error constants being provably not exploding but uniformly bounded with the respect to the hyperparameters approaching the critical regions, even in the situation of convex stochastic optimization problems (SOPs).

It is the key contribution of this work to essentially solve this problem for the RMSprop optimizer. Specifically, our main result provides bounds the expectation of the stopped evaluation of the objective function at the RMSprop optimization process from above by the sum of an initialization term that decays exponentially in the training time, a stochastic approximation remainder of order $\gamma _ { n } ,$ and a memory error of order $( 1 - \beta ) ^ { 2 }$ with the error constants being uniformly controlled over all admissible choices of the step sizes $( \gamma _ { n } ) _ { n \in \mathbb { N } }$ , the second moment decay parameter $\beta$ and the regularization parameter $\varepsilon \in [ 0 , 1 ]$ . In particular, we emphasize that our error analysis also directly covers the unregularized regime in which the regularity parameter vanishes $\varepsilon = 0 .$ . The non-asymptotic optimization error estimates that we establish hold not just for all suficiently large number of gradient steps but instead hold for every gradient step $n \in \mathbb { N } = \{ 1 , 2 , 3 , . . . \}$ (starting at the very first gradient step) with all error constants being explicitly specified.

The key innovative new feature in the proof of our error analysis are suitable inverse moment bounds for the second moment process in the RMSprop method. We also illustrate the general theory of this work in the case of several example SOPs.

## Contents

1. Introduction 2   
1.1. The RMSprop method 3   
1.2. Main result: Error analysis for RMSprop with full control of the hyperparameters 4   
1.3. Example: RMSprop for least squares with full control of the hyperparameters 5   
1.4. Short literature review and key novelty of this work 6   
1.5. Overview and structure of this work 7   
1.6. Use of large language models 7   
2. The abstract recursion estimate in Proposition 2.1 7   
3. Analysis of the property in item (iii) of Proposition 2.1 10   
4. Invserve moment estimates for the second moment process $( v _ { n } ) _ { n \in \mathbb { N } _ { 0 } }$ in the RMSprop method 12   
∈5. Technical estimates for the property in item (ii) of Proposition 2.1 17   
6. Initialisation estimates 19   
7. Combining the estimates before and after the coordinate startup times 22   
8. Criteria and examples for regular minibatch innovations 27   
8.1. A small-ball criterion for regularity 27   
8.2. Strong convexity in the data variable 27   
8.3. Nonlinear regression 30   
References 34

## 1. Introduction

Artificial Intelligence (AI) systems are trained by means of stochastic gradient descent (SGD) optimization methods [33]. The standard SGD optimization method is often not the employed training scheme for large scale AI systems but instead one often considers suitable adaptive variants of standard SGD such as the root mean square propagation SGD (RMSprop) [35], the adaptive moment estimation SGD (Adam) [25], and the Adam with decoupled weight decay (AdamW) [28] optimizers [21].

In the RMSprop method essentially every coordinate of the update is divided by the square root of a geometric average of the square of all previously seen stochastic gradients and the adaptivity parts in Adam and AdamW are essentially nothing else but the RMSprop method. Such adaptive SGD optimization methods are based on several hyperparameters in the optimizers including

● the sequence of learning rates $( \gamma _ { n } ) _ { n \in \mathbb { N } }$ (the sequence of step sizes),

∈● the regularitzation parameter ε (which ensures that one does not divide by zero in each update) as well as

● the second moment decay parameter $\beta$ (which, loosely speaking, models how much weight the recent stochastic gradients get within the geometric average of the square of all previously seen stochastic gradients).

The regularization parameter $\varepsilon$ is often chosen to be very close to zero such as $1 0 ^ { - 8 }$ in $\mathrm { P y }$ Torch by default and the second moment decay parameter $\beta$ is often chosen to very close to 1 such as 0.99 (RMSprop) and 0.999 (Adam and AdamW) in PyTorch by default.

Despite the already existing extensive literature on the theoretical analysis of such adaptive SGD optimization methods and the high importance of such methods, it remains an open problem of research to prove (or disprove) rigorous error estimates for such optimizers with the error constants being provably not exploding but uniformly bounded with the respect to the hyperparameters approaching the critical regions, even in the situation of strongly convex stochastic optimization problems (SOPs).

It is the key contribution of this work to essentially solve this problem for the RMSprop optimizer. Specifically, in our main result in Theorem 1.1 below we bound the expectation of a stopped evaluation of the objective function at the RMSprop optimization process (which controls the squared mean square strong optimization error in the special case of strongly convex SOPs) from above by the sum of

● an initialization term that decays exponentially in the training time $\begin{array} { r } { t _ { n } = \sum _ { k = 1 } ^ { n } \gamma _ { k } } \end{array}$

● a stochastic approximation remainder of order $\gamma _ { n } ,$ , and

● a memory error of order $( 1 - \beta ) ^ { 2 }$

with the error constants being uniformly controlled over all admissible choices of the step sizes $( \gamma _ { n } ) _ { n \in \mathbb { N } }$ , the second moment decay parameter $\beta$ and the regularization parameter $\varepsilon \in [ 0 , 1 ]$ ∈. Particularly, we highlight that our error analysis also directly covers the unregularized regime in which the regularity parameter completely vanishes $\varepsilon = 0$

1.1. The RMSprop method. In order to formulate Theorem 1.1, we present the considered mathematica setup and we briefly recall the RMSprop method.

Let d $\in \mathbb { N } = \{ 1 , 2 , 3 , \dots \}$ , let (U, U) be a measurable space, let $X \colon \mathbb { R } ^ { d } \times \mathcal { U }  \mathbb { R } ^ { d }$ be product measurable, let $( \Omega , \mathcal { F } , \mathbb { P } )$ be a probability space, and let U be an U-valued random variable. We call the pair $( X , U )$ an innovation. The RMSprop algorithm driven by $( X , U )$ depends on the following additional parameters:

(i) $\beta \in [ 0 , 1 )$ (second moment decay parameter ), $\varepsilon \in [ 0 , \infty )$ (regularization parameter ),

(ii) a non-increasing $( 0 , \infty )$ -valued sequence $( \gamma _ { n } ) _ { n \in \mathbb { N } }$ (step-sizes/learning rates), and

(iii) $\theta _ { 0 } \in \mathbb { R } ^ { d }$ (initial state/value).

Let $U _ { n } , n \in \mathbb { N } ,$ be independent copies of $U$ and let $( \mathcal { F } _ { n } ) _ { n \in \mathbb { N } _ { 0 } }$ be the filtration generated by these copies, that is, for every $n \in  { \mathbb { N } } _ { 0 }$ let

$$
\mathcal { F } _ { n } = \sigma ( U _ { 1 } , U _ { 2 } , \ldots , U _ { n } ) .\tag{1}
$$

The associated RMSprop algorithm is the $( \mathcal { F } _ { n } ) _ { n \in \mathbb { N } _ { 0 } }$ -adapted process $( \theta _ { n } ) _ { n \in  { \mathbb { N } } _ { 0 } }$ defined recursively, for every $n \in \mathbb { N } .$ by

$$
\theta _ { n } = \theta _ { n - 1 } + \gamma _ { n } \bigl [ \varepsilon + \sqrt { v _ { n } } \bigr ] ^ { - 1 } X \bigl ( \theta _ { n - 1 } , U _ { n } \bigr ) ,\tag{2}
$$

where all operations are applied coordinatewise to vectors and where, for every $n \in \mathbb { N } .$ , we have that

$$
v _ { n } = \frac { 1 - \beta } { 1 - \beta ^ { n } } \sum _ { k = 0 } ^ { n - 1 } \beta ^ { k } X ( \theta _ { n - 1 - k } , U _ { n - k } ) ^ { 2 } .\tag{3}
$$

Equivalently, we have for every $n \in \mathbb { N }$ that

$$
v _ { n } = \frac { \beta ( 1 - \beta ^ { n - 1 } ) } { 1 - \beta ^ { n } } v _ { n - 1 } + \frac { 1 - \beta } { 1 - \beta ^ { n } } X ( \theta _ { n - 1 } , U _ { n } ) ^ { 2 } .\tag{4}
$$

where $\theta _ { 0 }$ is the predetermined initial state and where $v _ { 0 } = 0 \in \mathbb { R } ^ { d }$ . If $\varepsilon = 0$ , all coordinatewise normalized innovations are understood with the convention

$$
0 ^ { - 1 } 0 = \frac { 0 } { 0 } = 0 .\tag{5}
$$

We call the mapping $f _ { X } \colon  { \mathbb { R } ^ { d } } \to  { \mathbb { R } ^ { d } }$ given, for every $\theta \in \mathbb { R } ^ { d }$ , by $f _ { X } ( \theta ) = \mathbb { E } [ X ( \theta , U ) ]$ the vector field associated with the innovation $( X , U )$

A key hypothesis in our error analysis is the following non-degeneracy assumption, which we formalize in Definition 4.1 below. For suitable $c , q > 0$ we have for every $i \in \{ 1 , 2 , \ldots , d \}$ that the Laplace transform of $X ^ { ( i ) } ( \theta , U ) ^ { 2 }$ satisfies

$$
\mathbb { E } \big [ e ^ { - \lambda X ^ { ( i ) } ( \theta , U ) ^ { 2 } } \big ] \leq \exp ( - \operatorname* { m i n } \{ c \lambda , q \} ) .\tag{6}
$$

Loosely speaking, this hypothesis ensures for every component of the innovation that the probability that this component vanishes is not too large.

In our error analysis result for the RMSprop method in Theorem 1.1 below we employ this assumption together with locally $L ^ { \infty }$ -bounded innovations and a local Polyak– Lojasiewicz (PL) inequality (see (11) below). The additional condition in (11) prevents abrupt decreases of the step-size when measured with respect to the training time $\begin{array} { r } { t _ { n } = \sum _ { k = 1 } ^ { n } \gamma _ { k } } \end{array}$ for every $n \in  { \mathbb { N } } _ { 0 }$ . The condition in (11) is satisfied, in particular, in the situation of =the usual polynomial learning rate schedules.

1.2. Main result: Error analysis for RMSprop with full control of the hyperparameters. The following non-asymptotic error analysis result for the RMSprop method is a consequence of the general recursive error estimate in Theorem 7.1, which is stated and proved in Section 7 below. In Theorem 1.2 in Subsection 1.3 below we illustrate the conclusion of Theorem 1.1 in the example of linear least squares (example application of Theorem 1.1).

Theorem 1.1 (Error analysis for RMSprop with full control of the hyperparameters). Let $\rho , c , q , C _ { X } , L _ { f } , C _ { \mathrm { L o j } } , C _ { \mathrm { d e c } } \in ( 0 , \infty )$ . Set

$$
\eta = [ 2 C _ { \mathrm { L o j } } ( C _ { X } + 1 ) ] ^ { - 1 } .\tag{7}
$$

Then there exist $C _ { 0 } , C _ { 1 } \in ( 0 , \infty )$ such that the following is true. Let $\beta \in [ 1 / 2 , 1 ) , \varepsilon \in [ 0 , 1 ]$ satisfy

$$
q + 3 \ln \beta \geq \rho .\tag{8}
$$

Let $V \subseteq \mathbb { R } ^ { d }$ be measurable, let $( X , U )$ be $( c , q )$ -regular on V in the sense of Definition 4.1 and assume $f o r$ every $\theta \in V , i \in \{ 1 , \ldots , d \}$ that

$$
\| X ^ { ( i ) } ( \theta , U ) \| _ { L ^ { \infty } } \leq C _ { X } .\tag{9}
$$

Let $F \colon \mathbb { R } ^ { d } \to [ 0 , \infty )$ have a globally L<sub>f</sub>-Lipschitz continuous derivative and assume for every $\theta \in V$ that

$$
f _ { X } ( \theta ) = - \nabla F ( \theta ) \qquad a n d \qquad F ( \theta ) \leq C _ { \mathrm { L o j } } | f _ { X } ( \theta ) | ^ { 2 } .\tag{10}
$$

Let $( \gamma _ { n } ) _ { n \in \mathbb { N } }$ be decreasing and $\left( 0 , C _ { \mathrm { L o j } } ( C _ { X } + \varepsilon ) \right]$ -valued and assume for every $j , n \in \mathbb { N }$ with $1 < j \leq n$ that

$$
\begin{array} { r } { \gamma _ { j } \leq \gamma _ { n } C _ { \mathrm { d e c } } \exp \bigl ( \frac \eta 2 \sum _ { k = j + 1 } ^ { n } \gamma _ { k } \bigr ) . } \end{array}\tag{11}
$$

Let $\theta _ { 0 } ~ \in ~ \mathbb { R } ^ { d }$ and let $( \theta _ { n } ) _ { n \in  { \mathbb { N } } _ { 0 } }$ be the RMSprop algorithm driven by $( X , U )$ with parameter tuple $( \beta , \varepsilon , ( \gamma _ { n } ) _ { n \in \mathbb { N } } , \theta _ { 0 } )$ ∈defined above. Define $\tau = \operatorname* { i n f } \{ n \in \mathbb { N } _ { 0 } { : } \theta _ { n } \notin V \}$ . Then one has for every $n \in \mathbb { N }$ that

$$
\begin{array} { r } { \mathbb { E } \big [ F ( \theta _ { n } ) \mathbb { 1 } _ { \{ r \geq n \} } \big ] \leq \exp \big ( - \eta \sum _ { k = 2 } ^ { n } \gamma _ { k } \big ) \left[ F ( \theta _ { 0 } ) ^ { 1 / 2 } + \gamma _ { 1 } ( 2 ^ { - 1 } L _ { f } d ) ^ { 1 / 2 } \right] ^ { 2 } + C _ { 0 } \gamma _ { n } + C _ { 1 } ( 1 - \beta ) ^ { 2 } . } \end{array}\tag{12}
$$

Theorem 1.1 follows from an application of the general recursive error estimate in Theorem 7.1, which is stated and proved in Section 7 below. We refer to the end of Section 7 for the detailed proof of Theorem 1.1.

In (12) we bound the expectation of the τ -stopped evaluation of the objective function at the RMSprop optimization process from above by the sum

(i) of the initialization term that converges to zero as the training time $t _ { n } = \Sigma _ { k = 1 } ^ { n }$ γ<sub>k</sub> converges to infinity in the number of RMSprop steps n,

(ii) of the term $C _ { 0 } \gamma _ { n }$ (the usual stochastic approximation error) that converges to zero as the number of RMSprop steps n goes to infinity, and

(iii) of the term $C _ { 1 } ( 1 - \beta ) ^ { 2 }$ that converges to zero as the second moment decay parameter $\beta$ approaches 1.

The convergence speed $C _ { 0 } \gamma _ { n }$ according to item (ii) is the usual stochastic approximation error term that also appears in the mean square error analysis of the standard SGD method and is optimal and cannot be improved (cf., for instance, [23]). In the case of Adam numerical simulations and analytical considerations suggest that the power two in the error term $C _ { 1 } ( 1 - \beta ) ^ { 2 }$ can also not be improved (cf., for example, [7])

We highlight that, except of $C _ { 0 }$ and $C _ { 1 }$ , all constants in Theorem 1.1 are explicitly specified and, actually, even $C _ { 0 }$ and $C _ { 1 }$ can be explicitly specified (see Remark 7.2 in Section 7 below). In particular, Remark 7.2 shows that $C _ { 0 }$ and $C _ { 1 }$ in Theorem 1.1 can be chosen as

$$
C _ { 0 } = 2 ^ { 1 3 } C _ { \mathrm { d e c } } d ( L _ { f } + C _ { X } + 1 ) ^ { 7 } ( \eta ^ { - 1 } + 1 ) \biggl [ \sum _ { r = 0 } ^ { \infty } \frac { ( c ^ { - 3 } + 1 ) ( r + 1 ) ^ { 3 } } { \exp ( \operatorname * { m i n } \{ \rho , 2 ^ { - 4 } q \} r ) } \biggr ]\tag{13}
$$

and

$$
C _ { 1 } = 2 ^ { 1 3 } d L _ { f } C _ { X } ^ { 6 } \eta ^ { - 2 } \Bigg [ \sum _ { r = 0 } ^ { \infty } \frac { c ^ { - 3 } ( r + 1 ) ^ { 3 } } { \exp ( \rho r ) } \Bigg ] .\tag{14}
$$

We refer to (206), (208), and (210) in Remark 7.2 for details. We also emphasize that all of the error constants in (12) are completely independent of $\beta \in [ 1 / 2 , 1 )$ and also completely independent of $\varepsilon \in [ 0 , 1 ]$ and we highlight that even in the unregularized regime $\varepsilon = 0$ the conclusion of (12) remains valid under the zero-over-zero convention in (5) above.

In Theorem 1.1 the initial value of the RMSprop optimization process is a purely deterministic vector $\theta _ { 0 } \in \mathbb { R } ^ { d }$ instead of an $\mathbb { R } ^ { d } .$ -valued random variable. However, under the assumption that this $\mathbb { R } ^ { d _ { - } }$ -valued random variable is independent from the driving random variables $U _ { n } , n \in \mathbb { N } ,$ , we have that Theorem 1.1 directly implies the corresponding statement but with a stochastic initial state of the RMSprop process as the error constants and parameters $\eta , C _ { 0 } , C _ { 1 } , L _ { f }$ , and d in (12) are all completely independent of the initial value $\theta _ { 0 }$

We also point out that in Theorem 1.1 we do not require that the learning rates $\gamma _ { n } , n \in \mathbb { N }$ , converge to zero and the inequality in (12) holds true even if the initial learning rates $( \gamma _ { n } ) _ { n \in \mathbb { N } }$ are bounded from above by $C _ { \mathrm { L o j } } C _ { X }$ ∈and non-increasing, even though it is well-known that RMSprop and other SGD optimization methods fail to converge if the learning rates do not converge to zero [8].

We also note that the error estimate in (12) in Theorem 1.1 is restricted to the event $\{ \tau \geq n \}$ and is thus only applicable until the optimization process $( \theta _ { n } ) _ { n \in  { \mathbb { N } } _ { 0 } }$ has left the measurable set V. We think of the set V ∈as a suficiently large compact ball (see, for instance, (17) in Theorem 1.2 below) and we refer, for example, to [6] and [9, Theorem 2.10] for results in the literature that establish a priori bounds for RMSprop and other related adaptive SGD optimization methods (particularly Adam) which ensure that the considered optimization process remains in a suficiently large compact ball.

1.3. Example: RMSprop for least squares with full control of the hyperparameters. In the following result, Theorem 1.2 below, we illustrate the conclusion of Theorem 1.1 above in the example of linear least squares. In Section 8 below we show in detail how Theorem 1.1 can be applied to prove Theorem 1.2.

Theorem 1.2 (RMSprop for linear least squares). Let $( \Omega , \mathcal { F } , \mathbb { P } )$ be a probability space, let $\kappa \in ( 0 , \infty )$ $m , d , M \in \mathbb { N } , \ \vartheta \in \mathbb { R } ^ { d }$ , let $A \in \mathbb { R } ^ { m \times d }$ have no zero columns, let $K \subseteq \mathbb { R } ^ { m }$ be compact, let $( Y _ { n , r } ) _ { ( n , r ) \in \mathbb { N } ^ { 2 } }$ be K-valued i.i.d. random variables with a bounded Lebesgue density, let $L \colon \mathbb { R } ^ { d } \times \mathbb { R } ^ { m }  \mathbb { R }$ satisfy for all $\theta \in \mathbb { R } ^ { d }$ $y \in \mathbb { R } ^ { m }$ that

$$
L ( \theta , y ) = \left| A \theta - y \right| ^ { 2 }\tag{15}
$$

and let $F : \mathbb { R } ^ { d }  \mathbb { R }$ satisfy for all $\theta \in \mathbb { R } ^ { d }$ that $\begin{array} { r } { F ( \theta ) = \mathbb { E } [ L ( \theta , Y _ { 1 , 1 } ) ] - \operatorname* { i n f } _ { \vartheta \in \mathbb { R } ^ { d } } \mathbb { E } [ L ( \vartheta , Y _ { 1 , 1 } ) ] } \end{array}$ . Then there exists $\eta \in ( 0 , \infty )$ such that for every ${ \mathfrak { C } } \in ( 0 , \infty )$ there exists $c \in \mathbb { R }$ ∈such that for all $\beta \in [ \mathsf { 1 / 2 } , 1 ) , \ \varepsilon \in [ 0 , 1 ]$ , all $\mathbb { R } ^ { d } .$ valued stochastic processes $( w _ { n } ) _ { n \in \mathbb { N } _ { 0 } }$ and $( \theta _ { n } ) _ { n \in  { \mathbb { N } } _ { 0 } } ,$ , and every $( 0 , \mathfrak { C } ]$ -valued decreasing $( \gamma _ { n } ) _ { n \in \mathbb { N } }$ with the property that for every $n \in \mathbb { N } , \ j \in \mathbb { N } \cap [ 1 , n ] , \ i \in \{ 1 , 2 , . . . , d \}$ it holds that

$$
w _ { 0 } = 0 , \qquad w _ { n } ^ { ( i ) } = \beta w _ { n - 1 } ^ { ( i ) } + ( 1 - \beta ) \big [ M ^ { - 1 } \Sigma _ { r = 1 } ^ { M } \big ( \nabla _ { \theta _ { i } } L \big ) \big ( \theta _ { n - 1 } , Y _ { n , r } \big ) \big ] ^ { 2 } ,\tag{16}
$$

$$
\begin{array} { r } { \theta _ { 0 } = \vartheta , \qquad \theta _ { n } ^ { ( i ) } = \theta _ { n - 1 } ^ { ( i ) } - \gamma _ { n } \big [ \varepsilon + ( 1 - \beta ^ { n } ) ^ { - 1 / 2 } ( w _ { n } ^ { ( i ) } ) ^ { 1 / 2 } \big ] ^ { - 1 } M ^ { - 1 } \sum _ { r = 1 } ^ { M } \bigl ( \nabla _ { \theta _ { i } } L \bigr ) \bigl ( \theta _ { n - 1 } , Y _ { n , r } \bigr ) } \end{array}
$$

and $\begin{array} { r } { \gamma _ { j } \leq \mathfrak { C } \exp \bigl ( \frac { \eta } { 2 } \sum _ { k = j + 1 } ^ { n } \gamma _ { k } \bigr ) \gamma _ { n } } \end{array}$ we have for every $n \in \mathbb { N }$ that

$$
\begin{array} { r } { \mathbb { E } \big [ F ( \theta _ { n } ) \mathbb { 1 } _ { \cap _ { j = 0 } ^ { n - 1 } \{ | \theta _ { j } | \leq \kappa \} } \big ] \leq \exp \big ( - \eta \sum _ { k = 2 } ^ { n } \gamma _ { k } \big ) \left( \gamma _ { 1 } ( d \| A ^ { \boldsymbol { \mathsf { T } } } A \| ) ^ { 1 / 2 } + | F ( \theta _ { 0 } ) | ^ { 1 / 2 } \right) ^ { 2 } + c \gamma _ { n } + c ( 1 - \beta ) ^ { 2 } . } \end{array}\tag{17}
$$

In Section 8 we rigorously show how Theorem 1.2 can be deduced from Theorem 1.1 in Subsection 1.2 above. In addition, in Section 8 we also illustrate the conclusion of Theorem 1.1 in the case of two further classes of example SOPs, namely, risk-sensitive exponential tilting with coordinate gradients that are strongly convex in the data variable and compactly supported nonlinear regression with vector-valued responses.

We point out that in Theorem 1.2 we do not assume that $A ^ { \mathsf { T } } A$ is invertible and we note that the objective function $F : \mathbb { R } ^ { d }  \mathbb { R }$ in Theorem 1.2 is convex but not necessarily strongly convex.

1.4. Short literature review and key novelty of this work. Nowadays there are a large number of research works that study adaptive gradient based optimization methods theoretically. In particular,

● we refer, for instance, to [4, Theorem 3.1], [16, Subsection 2.4], [24, Subsection 6.4], [27, Theorem 3.6 and Corollary 3.7], [34, Subsection 4.2], [37, Theorem 4.1], [39, Theorem 1, Corollary 2, and Corollary 3], [40, Theorem 1 and Theorem 3], [42, Sections 4 and 5], and [43, Section 3] for error, complexity, and convergence analyses for the RMSprop method and minor modifications of it when applied to SOPs,

● we refer, for example, to [2, 11, 12, 20, 29] for works that study RMSprop in the deterministic/full-batch setting, and

● we refer, for instance, to [13, 14, 22] for lower bounds and non-convergence results to global minimizers for RMSprop in the training of artificial neural networks (ANNs).

For online learning problems we also refer, for example, to [31, Theorem 4.1, Section 4, and Assumptions A1–A2] for upper bounds for the (averaged) regret of a modified variant of the RMSprop method in which the second moment parameter $\beta$ is not constant but converging with appropriate speed of convergence to 1 during the training procedure and in which the regularization parameter ε is not constant but converging with appropriate speed of convergence to 0 during the training procedure. We also refer, for instance, to [30, Theorem 4.2, Theorem C.2, and Theorem C.5] for approximation results for RMSprop applied to SOPs using solution processes of stochastic diferential equations (SDEs).

Furthermore, we refer, for example, to [1, 3, 5, 7, 10, 19, 26, 32, 36, 38, 41] for minimized/ergodic/non-ergodic error, gradient, and complexity estimates for the Adam optimizer (which is basically nothing else but the combination of RMSprop and momentum) and modified variants of it and we refer, for instance, to [7, 18, 32] for non-convergence results for Adam due to an internal error of the optimizer. RMSprop also admits such an internal error and this is the reason why the additional error term $C _ { 1 } ( 1 - \beta ) ^ { 2 }$ in (12) in Theorem 1.1 above can, in general, not be avoided. We also refer, for example, to [17] and the references therein for error estimates for other related adaptive SGD methods such as the adaptive gradient SGD (Adagrad) optimizer [15]. Further works on the theoretical analysis of SGD optimization methods can, for instance, also be found in [7, Section 1.3].

The key innovative new contribution of this work is that we establish convergence rates and error estimates for RMSprop with full control of the hyperparameters. Specifically, to the best of our knowledge, Theorem 1.1 is the first result in the scientific literature that establishes convergence of the RMSprop method for SOPs with the error constants being uniformly controlled with respect to the hyperparameters, particularly, with respect to the second moment decay parameter $\beta$ and the regularization parameter ε. This work even covers the completely unregularized case $\varepsilon = 0$

This kind of error analysis is very relevant for the application of adaptive optimizers in practically relevant regimes as if the error constants in error estimates are not uniformly bounded with respect to the hyperparameters but instead grow, for example, quadratically in the reciprocal $\varepsilon ^ { - 1 }$ of the regularization parameter ε and if the regularization parameter is chosen to be $\varepsilon = 1 0 ^ { - 8 }$ (default value in PyTorch), then the error constant would be as large as $1 0 ^ { 1 6 }$ and would make the error estimate essentially not useable in practically relevant regimes.

On a technical level, the key innovative contribution of this work – which allows us to prove convergence rates with full control of the hyperparameters – is to establish and employ new upper bounds for the inverse moments of the i-th component of the second moment process $( v _ { n } ) _ { n \in \mathbb { N } _ { 0 } }$ in (4) in the RMSprop method at time n ∈ N ∈after the second moment process has become suficiently large, that is, restricted to the event $\left\{ \sigma ^ { ( i ) } \le n \right\}$ where $\sigma ^ { ( i ) } = \operatorname* { i n f } \{ n \in \mathbb { N } { : } v _ { n } ^ { ( i ) } \geq \iota \}$ for an arbitrary given real constant $\iota \in ( 0 , \infty )$ . Such inverse moment estimates are established in Section 4 (see Lemma 4.4 and Lemma 4.5 in Section 4) and are then used extensively in the proof of Theorem 7.1 in Section 7 and, thereby, also in the proof of Theorem 1.1 (see the end of Section 7).

Such inverse moment bounds for $( v _ { n } ) _ { n \in \mathbb { N } _ { 0 } }$ cannot in general hold on the whole probability space: before a component of $v _ { n }$ ∈has reached a non-degenerate level (that is, has become greater than or equal to some suficiently large real threshold $\iota \in ( 0 , \infty ) )$ , it may vanish and its inverse moments may explode and fail to exist. We therefore introduce the random, coordinatewise startup time $\boldsymbol { \sigma } ^ { ( i ) }$ at which this level is first reached (at which the i-th component of the second moment process $( v _ { n } ^ { ( i ) } ) _ { n \in \mathbb { N } _ { 0 } }$ has become greater than or equal to a predetermined real threshold ι).

On the complement of the event $\{ \sigma ^ { ( i ) } \leq n \}$ we use elementary estimates for $F ( \theta _ { n } ) \mathbb { 1 } _ { \{ \tau \geq n \} }$ and establish that the probability of the event $( \Omega \backslash \{ \sigma ^ { ( i ) } ~ \leq ~ n \} ) \cap \{ \tau ~ \geq ~ n \} ~ = ~ \{ \tau ~ \geq ~ n , \sigma ^ { ( i ) } ~ > ~ n \}$ ≥is exponentially small in n (see Proposition 6.3 in Section 6 below).

Performing this early-late decomposition separately for every coordinate and at every iteration produces a single recursion that is valid from the first step on. To the best of our knowledge, this combination of post-startup inverse moment estimates with a diferent pre-startup argument is a new proof mechanism in the analysis of stochastic gradient methods.

## 1.5. Overview and structure of this work. The remainder of the article is organized as follows.

● In Proposition 2.1 in Section 2 we derive under the abstract assumptions in items (i)–(iii) a one-step recursion for the scalar quantity $\psi _ { n } = \mathbb { E } [ F ( \theta _ { n } ) \mathbb { 1 } _ { \{ \tau \geq n \} } ]$ (expectation of the τ-stopped evaluation of the objective function $F \colon \mathbb { R } ^ { d } \to \lceil 0 , \infty \ )$ ≥at the RMSprop optimization process $\theta _ { n }$ after n steps) for $n \in \mathbb { N } .$ In Proposition 2.1 the events $G _ { n , i } \in \mathcal { F } _ { n - 1 } , n \in \mathbb { N } , i \in \left\{ 1 , \ldots , d \right\}$ , are general predictable events but in our later application of Proposition 2.1 the event $G _ { n , i }$ will coincide with the event $\{ \sigma ^ { ( i ) } < n \}$ where $\sigma ^ { ( i ) } = \operatorname* { i n f } \{ n \in \mathbb { N } { : } v _ { n } ^ { ( i ) } \geq ( \kappa _ { 0 } ) ^ { - 1 } 2 c \}$ is the stopping time describing the first time where the i-th component of the second moment process $( v _ { n } ^ { ( i ) } ) _ { n \in \mathbb { N } _ { 0 } }$ in the RMSprop method is suficiently large (greater than or equal to the constant $2 c ( \kappa _ { 0 } ) ^ { - 1 } )$

● In Section 3 we show that suitable inverse moment bounds for the i-th component of the second moment process $( v _ { n } ^ { ( i ) } ) _ { n \in \mathbb { N } _ { 0 } }$ in the RMSprop method (see (35) in Proposition 3.1 in Section 3) are suficient ∈to ensure that the property in item (iii) in Proposition 2.1 is satisfied (estimates for the covariance correction).

● In Section 4 we then establish such inverse moment bounds restricted to the events $G _ { n , i } , n \in \mathbb { N }$ , i ∈ $\{ 1 , \ldots , d \}$ (on the complement $\Omega \backslash G _ { n , i }$ of the event $G _ { n , i }$ the inverse moment bounds do in general not hold).

● In Section 5 we provide suficient conditions to ensure that the property in item (ii) of Proposition 2.1 is satisfied (estimates for the quadratic Taylor remainder).

● In Section 6 we show that the probability of the event $( \Omega \backslash G _ { n , i } ) \cap \{ \tau \geq n \}$ is exponentially small in n.

In Section 7 we combine Proposition 2.1 from Section 2 with the estimates from Sections 3–6 to establish the key error estimate in Theorem 7.1. In addition, in Section 7 we also apply Theorem 7.1 to deduce Theorem 1.1 in this introductory section. In the proof of Theorem 7.1 we apply Proposition 2.1 with the coordinate events $G _ { n , i }$ . The contributions on their complements are controlled by the estimates from Section 6, while the findings of Sections 3–5 ensure that the assumptions in items (ii) and (iii) of Proposition 2.1 are satisfied.

● In Section 8 we illustrate Theorem 1.1 by means of several examples. In particular, in Section 8 we show how Theorem 1.1 can be used to prove Theorem 1.2.

1.6. Use of large language models. The key ideas of the statements and the proofs of the main results of this work are due to the authors. GPT 5.5 and GPT 5.6 have been substantially employed to formulate and work out the statements and the proofs in LaTeX according to the strategies of the authors. GPT 5.5 and GPT 5.6 have also supported us in creating the review in Subsection 1.4 and the explanatory texts in the remainder of this work. The authors take full responsibility for each of the statements/arguments/sentences made in this work.

## 2. The abstract recursion estimate in Proposition 2.1

This section derives an abstract one-step recursion for the stopped Lyapunov values $F ( \theta _ { n } )$ . Starting from the smoothness inequality, we separate the coercive drift, the covariance correction caused by the dependence

of $v _ { n }$ on the current innovation, and the quadratic Taylor remainder. The resulting scalar recursion reduces the proof of the main result to estimates for these three contributions.

Proposition 2.1. Let $L _ { f } , c _ { v } , c _ { \mathrm { L o j } } , C _ { \mathrm { L o j } } \in ( 0 , \infty )$ $n _ { 0 } \in \mathbb { N } _ { : }$ , let $( C _ { X } ( n ) ) _ { n > n _ { 0 } }$ and $( C _ { \beta } ( n ) ) _ { n > n _ { 0 } }$ be $( 0 , \infty )$ -valued sequences and let $V \subseteq \mathbb { R } ^ { d }$ be measurable. Define

$$
\tau : = \operatorname* { i n f } \{ n \in \mathbb { N } \cap [ n _ { 0 } , \infty ) : \theta _ { n } \notin V \} .\tag{18}
$$

For every n $\epsilon \mathbb { N } \cap ( n _ { 0 } , \infty ) , \ i \in \{ 1 , . . . , d \}$ let $G _ { n , i } \in \mathcal { F } _ { n - 1 }$ . Suppose that $F \colon \mathbb { R } ^ { d } \to [ 0 , \infty )$ has an $L _ { f } – L i p s c h i t z$ continuous derivative and assume for all $\theta \in V$ that

$$
\begin{array} { r } { f _ { X } ( \theta ) = - \nabla F ( \theta ) \qquad \mathrm { ~ } a n d \qquad c _ { \mathrm { L o j } } | f _ { X } ( \theta ) | ^ { 2 } \leq F ( \theta ) \leq C _ { \mathrm { L o j } } | f _ { X } ( \theta ) | ^ { 2 } . } \end{array}\tag{19}
$$

Assume moreover that all quantities appearing below are well-defined and integrable to the powers used there on the corresponding stopped events. $I f \varepsilon = 0$ , inverse factors and conditional covariances appearing together with $\mathbb { 1 } _ { G _ { n , i } }$ are evaluated on $\{ \tau \geq n \} \cap G _ { n , i }$ and extended by zero to its complement. Assume that, for every $n \in \mathbb { N } \cap \left( n _ { 0 } , \infty \right)$ , the following conditions<sup>1</sup> hold:

$$
\begin{array} { r l } & { ( i ) \mathbb { E } [ \mathbb { 1 } _ { \{ \tau \geq n \} } \sum _ { i = 1 } ^ { d } \mathbb { 1 } _ { G _ { n , i } } ( \sqrt { v _ { n } ^ { ( i ) } } + \varepsilon ) ^ { - 1 } | f _ { X } ^ { ( i ) } ( \theta _ { n - 1 } ) | ^ { 2 } ] \ge c _ { v } \mathbb { E } [ \mathbb { 1 } _ { \{ \tau \geq n \} } \sum _ { i = 1 } ^ { d } \mathbb { 1 } _ { G _ { n , i } } | f _ { X } ^ { ( i ) } ( \theta _ { n - 1 } ) | ^ { 2 } ] , } \end{array}
$$

(ii) $\begin{array} { r } { \mathbb { E } \big [ \mathbb { 1 } _ { \{ \tau \geq n \} } \sum _ { i = 1 } ^ { d } \mathbb { 1 } _ { G _ { n , i } } \big ( \sqrt { v _ { n } ^ { ( i ) } + \varepsilon } \big ) ^ { - 2 } X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) ^ { 2 } \big ] \leq C _ { X } ( n ) . } \end{array}$ , and

$$
\begin{array} { r } { \mathbb { E } [ \mathbb { 1 } _ { \{ \tau \geq n \} } \sum _ { i = 1 } ^ { d } \mathbb { 1 } _ { G _ { n , i } } \mathrm { c o v } ( ( \sqrt { v _ { n } ^ { ( i ) } } + \varepsilon ) ^ { - 1 } , X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) | \mathcal { F } _ { n - 1 } ) ^ { 2 } ] ^ { 1 / 2 } \leq C _ { \beta } ( n ) . } \end{array}
$$

For every $n \in \mathbb { N } \cap \left( n _ { 0 } , \infty \right)$ , set

$$
\psi _ { n } = \mathbb { E } \big [ \mathbb { 1 } _ { \{ \tau \geq n \} } F ( \theta _ { n } ) \big ] \qquad a n d \qquad \widetilde { \psi } _ { n } = \mathbb { E } \big [ \mathbb { 1 } _ { \{ \tau \geq n \} } F ( \theta _ { n - 1 } ) \big ] .\tag{20}
$$

Then one has for every $n \in \mathbb { N } \cap \left( n _ { 0 } , \infty \right)$ that

$$
\begin{array} { l } { \displaystyle \psi _ { n } \leq \left( 1 - \frac { c _ { \nu } } { C _ { \mathrm { L o j } } } \gamma _ { n } \right) \widetilde { \psi } _ { n } + \frac { C _ { \beta } ( n ) } { c _ { \mathrm { L o j } } ^ { 1 / 2 } } \gamma _ { n } \widetilde { \psi } _ { n } ^ { 1 / 2 } + \frac { L _ { f } C _ { X } ( n ) } { 2 } \gamma _ { n } ^ { 2 } } \\ { \displaystyle \qquad + c _ { \upsilon } \gamma _ { n } \mathbb { E } \Big [ \mathbb { 1 } _ { \{ \tau \geq n \} } \sum _ { i = 1 } ^ { d } \mathbb { I } _ { G _ { n , i } ^ { c } } \big [ f _ { X } ^ { ( i ) } ( \theta _ { n - 1 } ) \big ] ^ { 2 } \Big ] + \gamma _ { n } \mathbb { E } \Big [ \mathbb { 1 } _ { \{ \tau \geq n \} } \sum _ { i = 1 } ^ { d } \mathbb { I } _ { G _ { n , i } ^ { c } } \big ] f _ { X } ^ { ( i ) } ( \theta _ { n - 1 } ) ( \sqrt { v _ { n } ^ { ( i ) } } + \varepsilon ) ^ { - 1 } X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) \big ] \Big ] } \\ { \displaystyle \qquad + \frac { L _ { f } \gamma _ { n } ^ { 2 } } { 2 } \mathbb { E } \Big [ \mathbb { 1 } _ { \{ \tau \geq n \} } \sum _ { i = 1 } ^ { d } \mathbb { I } _ { G _ { n , i } ^ { c } } ( \sqrt { v _ { n } ^ { ( i ) } } + \varepsilon ) ^ { - 2 } X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) ^ { 2 } \Big ] . } \end{array}\tag{21}
$$

=<sub>In</sub> <sub>particular,</sub> <sub>one</sub> <sub>has</sub> <sub>for</sub> <sub>every</sub> <sub>n</sub> <sub>∈</sub> $\mathbb { N } \cap \left( n _ { 0 } , \infty \right)$ that

$$
\begin{array} { l } { \displaystyle \psi _ { n } \leq \left( 1 - \frac { c _ { v } } { 2 C _ { [ \alpha _ { 0 } ] } } \gamma _ { n } \right) \widetilde { \psi } _ { n } + \frac { C _ { \beta } ( n ) ^ { 2 } C _ { \mathrm { I } \alpha _ { 0 } } } { 2 c _ { v } c _ { \mathrm { I } \alpha _ { 0 } } } \gamma _ { n } + \frac { L _ { f } C _ { X } ( n ) } { 2 } \gamma _ { n } ^ { 2 } } \\ { \displaystyle \quad + c _ { v } \gamma _ { n } \mathbb { E } \Big [ \mathbb { 1 } _ { \{ \tau \geq n \} } \sum _ { i = 1 } ^ { d } \mathbb { I } _ { G _ { n , i } ^ { c } } \big | \int _ { X } ^ { ( i ) } ( \theta _ { n - 1 } ) \big | ^ { 2 } \Big ] + \gamma _ { n } \mathbb { E } \Big [ \mathbb { 1 } _ { \{ \tau \geq n \} } \sum _ { i = 1 } ^ { d } \mathbb { I } _ { G _ { n , i } ^ { c } } \int _ { X } ^ { ( i ) } ( \theta _ { n - 1 } ) \big ( \sqrt { v _ { n } ^ { ( i ) } } + \varepsilon \big ) ^ { - 1 } X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) \big | \Big ] } \\ { \displaystyle \quad + \frac { L _ { f } \gamma _ { n } ^ { 2 } } { 2 } \mathbb { E } \Big [ \mathbb { 1 } _ { \{ \tau \geq n \} } \sum _ { i = 1 } ^ { d } \mathbb { I } _ { G _ { n , i } ^ { c } } \big ( \sqrt { v _ { n } ^ { ( i ) } } + \varepsilon \big ) ^ { - 2 } X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) ^ { 2 } \Big ] . } \end{array}\tag{22}
$$

Proof. Fix $n > n _ { 0 }$ . Since $\tau$ is the first exit time from V after time $n _ { 0 } .$ , the event $\{ \tau \geq n \}$ implies $\theta _ { n - 1 } \in V$ . Since $\nabla F$ is L<sub>f</sub>-Lipschitz continuous and $f _ { X } = - \nabla F$ on V we get that for all $x \in V$ and $\boldsymbol { y } \in \mathbb { R } ^ { d }$

$$
F ( y ) \leq F ( x ) - \langle f _ { X } ( x ) , y - x \rangle + \frac { L _ { f } } { 2 } | y - x | ^ { 2 } .\tag{23}
$$

Using that

$$
\theta _ { n } = \theta _ { n - 1 } + \gamma _ { n } \bigl ( \sqrt { v _ { n } } + \varepsilon \bigr ) ^ { \circ ( - 1 ) } \circ X \bigl ( \theta _ { n - 1 } , U _ { n } \bigr ) ,\tag{24}
$$

Combining (23) and (24), we conclude that on the event $\{ \tau \geq n \}$

$$
\begin{array} { r l } & { F ( \theta _ { n } ) \le F ( \theta _ { n - 1 } ) - \gamma _ { n } \displaystyle \sum _ { i = 1 } ^ { d } ( \sqrt { v _ { n } ^ { ( i ) } } + \varepsilon ) ^ { - 1 } f _ { X } ^ { ( i ) } ( \theta _ { n - 1 } ) X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) } \\ & { \qquad + \frac { L _ { f } \gamma _ { n } ^ { 2 } } { 2 } \displaystyle \sum _ { i = 1 } ^ { d } ( \sqrt { v _ { n } ^ { ( i ) } } + \varepsilon ) ^ { - 2 } X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) ^ { 2 } . } \end{array}\tag{25}
$$

The events $G _ { n , i }$ and $\{ \tau \geq n \}$ are $\mathcal { F } _ { n - 1 ^ { - } } \mathrm { m e a s u r a b l e }$ . Moreover, $\mathbb { E } [ X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) | \mathcal { F } _ { n - 1 } ] = f _ { X } ^ { ( i ) } ( \theta _ { n - 1 } )$ . Hence, multiplying (25) by $\mathbb { 1 } _ { \{ \tau \geq n \} }$ , using the conditional covariance decomposition on $G _ { n , i } .$ , and taking expectations, we arrive at

$$
\begin{array} { r l } & { \psi _ { n } \leq \widetilde { \psi } _ { n } - \gamma _ { n } \mathbb { E } \Big [ \mathbb { 1 } _ { \{ \tau > n \} } \underset { i = 1 } { \overset { d } { \sum } } \mathbb { 1 } _ { G _ { n , i } } ( \sqrt { v _ { n } ^ { ( i ) } } + \varepsilon ) ^ { - 1 } | f _ { x } ^ { ( i ) } ( \theta _ { n - 1 } ) | ^ { 2 } \Big ] } \\ & { \qquad - \gamma _ { n } \mathbb { E } \Big [ \mathbb { 1 } _ { \{ \tau > n \} } \underset { i = 1 } { \overset { d } { \sum } } \mathbb { 1 } _ { G _ { n , i } } , f _ { x } ^ { ( i ) } ( \theta _ { n - 1 } ) \mathrm { c o v } ( ( \sqrt { v _ { n } ^ { ( i ) } } + \varepsilon ) ^ { - 1 } , X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) | \mathcal { F } _ { n - 1 } ) \Big ] } \\ & { \qquad + \gamma _ { n } \mathbb { E } \Big [ \mathbb { 1 } _ { \{ \tau > n \} } \underset { i = 1 } { \overset { d } { \sum } } \mathbb { 1 } _ { G _ { n , i } ^ { \varepsilon } } | f _ { X } ^ { ( i ) } ( \theta _ { n - 1 } ) ( \sqrt { v _ { n } ^ { ( i ) } } + \varepsilon ) ^ { - 1 } X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) | \Big ] } \\ & { \qquad + \frac { L _ { f } \gamma _ { n } ^ { 2 } } { 2 } \mathbb { E } \Big [ \mathbb { 1 } _ { \{ \tau > n \} } \underset { i = 1 } { \overset { d } { \sum } } \mathbb { 1 } _ { G _ { n , i } } ( \sqrt { v _ { n } ^ { ( i ) } } + \varepsilon ) ^ { - 2 } X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) ^ { 2 } \Big ] } \\ &  \qquad + \frac { L _ { f } \gamma _ { n } ^ { 2 } } { 2 } \mathbb { E } \Big [ \mathbb { 1 } _ { \{ \tau > n \} } \underset { i = 1 }  \overset \end{array}\tag{26}
$$

Next, we estimate the drift, covariance, and quadratic terms on the right-hand side. By assumption (i) and the upper Lojasiewicz bound, we have

$$
\begin{array} { r l r } {  { \mathbb { E } [ \mathbb { 1 } _ { \{ \tau \geq n \} } \sum _ { i = 1 } ^ { d } \mathbb { 1 } _ { G _ { n , i } } ( \sqrt { v _ { n } ^ { ( i ) } } + \varepsilon ) ^ { - 1 } | f _ { X } ^ { ( i ) } ( \theta _ { n - 1 } ) | ^ { 2 } ] } } \\ & { } & { \geq c _ { v } \mathbb { E } [ \mathbb { 1 } _ { \{ \tau \geq n \} } \sum _ { i = 1 } ^ { d } \mathbb { 1 } _ { G _ { n , i } } | f _ { X } ^ { ( i ) } ( \theta _ { n - 1 } ) | ^ { 2 } ] } \\ & { } & { \geq \frac { c _ { v } } { C _ { \mathrm { L o j } } } \widetilde { \psi } _ { n } - c _ { v } \mathbb { E } \Big [ \mathbb { 1 } _ { \{ \tau \geq n \} } \sum _ { i = 1 } ^ { d } \mathbb { 1 } _ { G _ { n , i } ^ { c } } | f _ { X } ^ { ( i ) } ( \theta _ { n - 1 } ) | ^ { 2 } \Big ] . } \end{array}\tag{27}
$$

=<sub>Next,</sub> <sub>we</sub> <sub>estimate</sub> <sub>the</sub> <sub>covariance</sub> <sub>term.</sub> <sub>As</sub> <sub>a</sub> <sub>consequence</sub> <sub>of</sub> <sub>the</sub> <sub>Cauchy–Schwarz</sub> <sub>inequality,</sub> <sub>we</sub> <sub>have</sub>

$$
\begin{array} { r l } & { \left| \mathbb { E } \Big [ \mathbb { 1 } _ { \{ \tau \geq n \} } \sum _ { i = 1 } ^ { d } \mathbb { 1 } _ { G _ { n , i } } f _ { X } ^ { ( i ) } ( \theta _ { n - 1 } ) \mathrm { c o v } ( ( \sqrt { v _ { n } ^ { ( i ) } } + \varepsilon ) ^ { - 1 } , X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) | \mathcal { F } _ { n - 1 } ) \Big ] \right| } \\ & { \leq \mathbb { E } \Big [ \mathbb { 1 } _ { \{ \tau \geq n \} } | f _ { X } ( \theta _ { n - 1 } ) | ^ { 2 } \Big ] ^ { 1 / 2 } \mathbb { E } \Big [ \mathbb { 1 } _ { \{ \tau \geq n \} } \sum _ { i = 1 } ^ { d } \mathbb { 1 } _ { G _ { n , i } } \mathrm { c o v } ( ( \sqrt { v _ { n } ^ { ( i ) } } + \varepsilon ) ^ { - 1 } , X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) | \mathcal { F } _ { n - 1 } ) ^ { 2 } \Big ] ^ { 1 / 2 } . } \end{array}\tag{28}
$$

By assumption (iii), the second factor is bounded by $C _ { \beta } ( n )$ . Moreover, on $\{ \tau \geq n \}$ the lower Lojasiewicz bound yields

$$
| f _ { X } ( \theta _ { n - 1 } ) | ^ { 2 } \leq c _ { \mathrm { L o j } } ^ { - 1 } F ( \theta _ { n - 1 } ) ,\tag{29}
$$

so that (20) and (29) give

$$
\mathbb { E } \big [ \mathbb { 1 } _ { \{ \tau \geq n \} } | f _ { X } ( \theta _ { n - 1 } ) | ^ { 2 } \big ] \leq c _ { \mathrm { L o j } } ^ { - 1 } \widetilde { \psi } _ { n } .\tag{30}
$$

Consequently, we get with (28) and (30) that

$$
\mathbb { E } \Big [ \mathbb { 1 } _ { \{ \tau \geq n \} } \sum _ { i = 1 } ^ { d } \mathbb { 1 } _ { G _ { n , i } } f _ { X } ^ { ( i ) } ( \theta _ { n - 1 } ) \mathrm { c o v } ( ( \sqrt { v _ { n } ^ { ( i ) } } + \varepsilon ) ^ { - 1 } , X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) | \mathcal { F } _ { n - 1 } ) \Big ] \geq - \frac { C _ { \beta } ( n ) } { c _ { \mathrm { L o j } } ^ { 1 / 2 } } \widetilde { \psi } _ { n } ^ { 1 / 2 } .\tag{31}
$$

By assumption (ii),

$$
\mathbb { E } \Big [ \mathbb { 1 } _ { \{ \tau \geq n \} } \sum _ { i = 1 } ^ { d } \mathbb { 1 } _ { G _ { n , i } } ( \sqrt { v _ { n } ^ { ( i ) } } + \varepsilon ) ^ { - 2 } X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) ^ { 2 } \Big ] \leq C _ { X } ( n ) .\tag{32}
$$

Combining (26) with (27), (31), and (32) proves (21). For the second estimate, we use Young’s inequality in the form

$$
\frac { C _ { \beta } ( n ) } { c _ { \mathrm { L o j } } ^ { 1 / 2 } } \widetilde { \psi } _ { n } ^ { 1 / 2 } \leq \frac { c _ { v } } { 2 C _ { \mathrm { L o j } } } \widetilde { \psi } _ { n } + \frac { C _ { \beta } ( n ) ^ { 2 } C _ { \mathrm { L o j } } } { 2 c _ { v } c _ { \mathrm { L o j } } } ,\tag{33}
$$

which, together with (21) and (33), proves (22).

## 3. Analysis of the property in item (iii) of Proposition 2.1

This section controls the covariance correction appearing in item (iii) of Proposition 2.1. A conditional Lipschitz argument bounds each coordinatewise covariance in terms of a conditional third moment of the innovation and a negative third moment of $v _ { n - 1 }$ . An auxiliary estimate for the resulting β-dependent quotient then distinguishes the initial $n ^ { - 1 }$ -regime from the asymptotic $1 - \beta { \mathrm { - r e g i m e } }$

Proposition 3.1. Let $\beta \in [ 1 / 2 , 1 ) , \ \varepsilon \in [ 0 , \infty ) , \ i \in \{ 1 , 2 , \ldots , d \} , \ n \in \mathbb { N } \backslash \{ 1 \}$ , let $A _ { n } \in \mathcal { F } _ { n - 1 }$ , and let $C _ { X } ^ { ( 3 ) } , C _ { v } ^ { ( - 3 ) }$ ∈ $( 0 , \infty )$ . Assume that

$$
\mathbb { E } \big [ | X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) | ^ { 3 } | \mathcal { F } _ { n - 1 } \big ] \leq C _ { X } ^ { ( 3 ) } \qquad o n ~ A _ { n }\tag{34}
$$

and

$$
\mathbb { E } \big [ \mathbb { 1 } _ { A _ { n } } ( v _ { n - 1 } ^ { ( i ) } ) ^ { - 3 } \big ] \leq C _ { v } ^ { ( - 3 ) } .\tag{35}
$$

The negative-moment assumption implies that $v _ { n - 1 } ^ { ( i ) } > 0$ almost surely on $A _ { n } . ~ H \varepsilon = 0$ , the conditional covariances in the conclusion are therefore evaluated on $A _ { n }$ −and extended by zero to $\Omega \backslash A _ { n }$ . Then one has

$$
\mathbb { E } \Big [ \mathbb { 1 } _ { A _ { n } } \mathrm { c o v } ( ( \sqrt { v _ { n } ^ { ( i ) } } + \varepsilon ) ^ { - 1 } , X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) | \mathcal { F } _ { n - 1 } ) ^ { 2 } \Big ] ^ { 1 / 2 } \le 4 C _ { X } ^ { ( 3 ) } ( C _ { v } ^ { ( - 3 ) } ) ^ { 1 / 2 } \frac { 1 - \beta } { 1 - \beta ^ { n - 1 } } .\tag{36}
$$

Proof. Set

$$
Y _ { n } ^ { ( i ) } = \frac { \beta ( 1 - \beta ^ { n - 1 } ) } { 1 - \beta ^ { n } } v _ { n - 1 } ^ { ( i ) } = \kappa _ { n } v _ { n - 1 } ^ { ( i ) } \qquad \mathrm { a n d } \qquad Z _ { n } = \frac { 1 - \beta } { 1 - \beta ^ { n } } .\tag{37}
$$

Then

$$
v _ { n } ^ { ( i ) } = Y _ { n } ^ { ( i ) } + Z _ { n } X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) ^ { 2 } .\tag{38}
$$

The negative-moment assumption gives $v _ { n - 1 } ^ { ( i ) } > 0$ almost surely on $A _ { n } .$ , and hence $Y _ { n } ^ { ( i ) } > 0$ there. In what follows, all conditional identities involving inverse factors are understood on $A _ { n }$ . The function

$$
h : ( 0 , \infty ) \to ( 0 , \infty ) , \ y \mapsto h ( y ) = { \frac { 1 } { \sqrt { y } + \varepsilon } }\tag{39}
$$

is for every $y _ { 0 } \in ( 0 , \infty )$ Lipschitz continuous on $[ y _ { 0 } , \infty )$ with constant ${ \frac { 1 } { 2 } } y _ { 0 } ^ { - 3 / 2 }$ . Using that $Y _ { n } ^ { ( i ) }$ is $\mathcal { F } _ { n - 1 }$ <sub>1</sub>-measurable, we conclude on $A _ { n }$ that

$$
\begin{array} { r l } &  \begin{array} { r l } & { \mathrm { l c o v } ( ( \sqrt { v _ { n } ^ { ( i ) } } + \varepsilon ) ^ { - 1 } , X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) | \mathcal { F } _ { n - 1 } ) | } \\ & { = | \cos ( h ( Y _ { n } ^ { ( i ) } + Z _ { n } X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) ^ { 2 } ) , X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) | \mathcal { F } _ { n - 1 } ) | } \\ & { = | \mathbb { E } [ ( h ( Y _ { n } ^ { ( i ) } + Z _ { n } X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) ^ { 2 } ) - h ( Y _ { n } ^ { ( i ) } ) ) \big ( X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) - \int _ { X } ^ { ( i ) } ( \theta _ { n - 1 } ) \big ) | \mathcal { F } _ { n - 1 } ] | } \\ & { \le \frac { Z _ { n } } { 2 ( Y _ { n } ^ { ( i ) } ) ^ { 3 / 2 } } \mathbb { E } [ X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) ^ { 2 } \Big | X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) - f _ { X } ^ { ( i ) } ( \theta _ { n - 1 } ) \Big | | \mathcal { F } _ { n - 1 } ] | } \\ &  \le \frac { Z _ { n } } { 2 ( Y _ { n } ^ { ( i ) } ) ^ { 3 / 2 } } \Big ( \mathbb { E } \big [ X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) ^ { 3 } \big | ^ { 3 } \mathcal { F } _ { n - 1 } \big ] + \mathbb { E } \big [ | X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) | ^ { 2 } | \mathcal { F } _ { n - 1 } \big ] \mathbb { E } \big [ | X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) | | \end{array} \end{array}\tag{40}
$$

On $A _ { n } ,$ , the assumption on the third moment therefore implies

$$
| \mathrm { c o v } ( ( \sqrt { v _ { n } ^ { ( i ) } } + \varepsilon ) ^ { - 1 } , X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) | \mathcal { F } _ { n - 1 } ) | ^ { 2 } \le Z _ { n } ^ { 2 } ( C _ { X } ^ { ( 3 ) } ) ^ { 2 } ( Y _ { n } ^ { ( i ) } ) ^ { - 3 } .\tag{41}
$$

Multiplying by $\mathbb { 1 } _ { A _ { n } }$ and taking expectations yields

$$
\mathbb E \Big [ \mathbb { 1 } _ { A _ { n } } \mathrm { c o v } ( ( \sqrt { v _ { n } ^ { ( i ) } } + \varepsilon ) ^ { - 1 } , X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) | \mathcal F _ { n - 1 } ) ^ { 2 } \Big ] \leq Z _ { n } ^ { 2 } ( C _ { X } ^ { ( 3 ) } ) ^ { 2 } \mathbb E [ \mathbb { 1 } _ { A _ { n } } ( Y _ { n } ^ { ( i ) } ) ^ { - 3 } ] .\tag{42}
$$

Since $Y _ { n } ^ { ( i ) } = \kappa _ { n } v _ { n - 1 } ^ { ( i ) }$ , we obtain

$$
\mathbb { E } \big [ \mathbb { 1 } _ { A _ { n } } ( Y _ { n } ^ { ( i ) } ) ^ { - 3 } \big ] = \Big ( \frac { 1 - \beta ^ { n } } { \beta \big ( 1 - \beta ^ { n - 1 } \big ) } \Big ) ^ { 3 } \mathbb { E } \big [ \mathbb { 1 } _ { A _ { n } } ( v _ { n - 1 } ^ { ( i ) } ) ^ { - 3 } \big ] \leq \Big ( \frac { 1 - \beta ^ { n } } { \beta \big ( 1 - \beta ^ { n - 1 } \big ) } \Big ) ^ { 3 } C _ { v } ^ { ( - 3 ) } .\tag{43}
$$

Hence, combining (42) and (43) and using the definition of $Z _ { n }$ ,

$$
\begin{array} { r l } & { \mathbb { E } \Big [ \mathbb { 1 } _ { A _ { n } } \mathrm { c o v } \big ( ( \sqrt { v _ { n } ^ { ( i ) } } + \varepsilon ) ^ { - 1 } , X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) \big | \mathcal { F } _ { n - 1 } \big ) ^ { 2 } \Big ] \leq ( C _ { X } ^ { ( 3 ) } ) ^ { 2 } C _ { v } ^ { ( - 3 ) } \Big ( \displaystyle \frac { 1 - \beta } { 1 - \beta ^ { n } } \Big ) ^ { 2 } \Big ( \displaystyle \frac { 1 - \beta ^ { n } } { \beta \big ( 1 - \beta ^ { n - 1 } \big ) } \Big ) ^ { 3 } } \\ & { \qquad = ( C _ { X } ^ { ( 3 ) } ) ^ { 2 } C _ { v } ^ { ( - 3 ) } \beta ^ { - 3 } \displaystyle \frac { \big ( 1 - \beta \big ) ^ { 2 } \big ( 1 - \beta ^ { n } \big ) } { ( 1 - \beta ^ { n - 1 } ) ^ { 3 } } . } \end{array}\tag{44}
$$

By Lemma 4.3, we have

$$
\frac { 1 - \beta ^ { n } } { n } \leq \frac { 1 - \beta ^ { n - 1 } } { n - 1 } , \mathrm { s o ~ t h a t } \frac { 1 - \beta ^ { n } } { 1 - \beta ^ { n - 1 } } \leq \frac { n } { n - 1 } .\tag{45}
$$

we infer, using $\beta \in [ \textstyle { \frac { 1 } { 2 } } , 1 )$ and $n \geq 2$ , that

$$
\beta ^ { - 3 } \frac { 1 - \beta ^ { n } } { 1 - \beta ^ { n - 1 } } \leq 8 \cdot 2 = 1 6 .\tag{46}
$$

Taking square roots, we finish the proof with the conclusion that

$$
\mathbb { E } \Big [ \mathbb { 1 } _ { A _ { n } \mathrm { c o v } } ( ( \sqrt { v _ { n } ^ { ( i ) } } + \varepsilon ) ^ { - 1 } , X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) | \mathcal { F } _ { n - 1 } ) ^ { 2 } \Big ] ^ { 1 / 2 } \le 4 C _ { X } ^ { ( 3 ) } ( C _ { v } ^ { ( - 3 ) } ) ^ { 1 / 2 } \frac { 1 - \beta } { 1 - \beta ^ { n - 1 } } .\tag{47}
$$

The latter proposition incorporates an n-dependent term $\frac { 1 - \beta } { 1 - \beta ^ { n - 1 } }$ . This is analysed in the next lemma.

Lemma 3.2. Let $n \in \mathbb { N } , \ \beta \in [ 0 , 1 )$ . Then one has that

$$
\frac { 1 - \beta } { 1 - \beta ^ { n } } \leq \left\{ \begin{array} { l l } { \frac { 2 } { n } , } & { i f n \leq ( 1 - \beta ) ^ { - 1 } } \\ { \frac { 1 - \beta } { 1 - e ^ { - 1 } } , } & { i f n \geq ( 1 - \beta ) ^ { - 1 } } \end{array} \right.\tag{48}
$$

Proof. Set $x = 1 - \beta \in ( 0 , 1 ]$ . Then

$$
{ \frac { 1 - \beta } { 1 - \beta ^ { n } } } = { \frac { x } { 1 - ( 1 - x ) ^ { n } } } .\tag{49}
$$

Assume first that $n \leq ( 1 - \beta ) ^ { - 1 }$ , that is, $n x \leq 1$ . Since $( 1 - x ) ^ { n } \leq e ^ { - n x }$ and $1 - e ^ { - y } \ge y / 2$ for every $y \in [ 0 , 1 ]$ , we obtain

$$
1 - \left( 1 - x \right) ^ { n } \geq 1 - e ^ { - n x } \geq \frac { n x } { 2 } , \mathrm { { w h i c h ~ y i e l d s } } \frac { 1 - \beta } { 1 - \beta ^ { n } } = \frac { x } { 1 - \left( 1 - x \right) ^ { n } } \leq \frac { 2 } { n } .\tag{50}
$$

This proves the first assertion.

Now assume that $n \geq ( 1 - \beta ) ^ { - 1 }$ , that is, $n x \ge 1$ . Then

$$
\beta ^ { n } = ( 1 - x ) ^ { n } \leq e ^ { - n x } \leq e ^ { - 1 } \quad { \mathrm { a n d , ~ h e n c e , } } \quad { \frac { 1 - \beta } { 1 - \beta ^ { n } } } \leq { \frac { 1 - \beta } { 1 - e ^ { - 1 } } } .\tag{51}
$$

This proves the second assertion.

## 4. Invserve moment estimates for the second moment process $( v _ { n } ) _ { n \in \mathbb { N } _ { 0 } }$ in the RMSprop method

This section establishes the negative moment estimates needed for the covariance and quadratic error bounds. We introduce the (c, q)-regularity condition, derive conditional Laplace-transform estimates for the stopped conditioner, and convert these estimates into bounds for arbitrary negative moments. Since such bounds need not hold initially, the final result is formulated on the event that the corresponding coordinatewise startup time has already occurred.

Definition 4.1. Let $c , q \in ( 0 , \infty )$ . We say that an innovation $( X , U )$ is $( c , q )$ -regular on a set $\nu \subseteq \mathbb { R } ^ { d }$ if and only if for all $\theta \in \mathcal { V } , \ i \in \{ 1 , \dots , d \} , \ \lambda \in [ 0 , \infty )$ we have that

$$
- \ln \mathbb { E } \big [ e ^ { - \lambda X ^ { ( i ) } ( \theta , U ) ^ { 2 } } \big ] \geq \big ( c \lambda \big ) \wedge q .\tag{52}
$$

First we will deduce that (c, q)-regular innovations do allow us to estimate the exponential moments of $v _ { n }$

Proposition 4.2. Let $( X , U )$ be a $( c , q ) \ – r e g u l a r$ innovation on a measurable set V and let $\left( \theta _ { n } \right)$ be an RMSprop algorithm with arbitrary parameter tuple $( \beta , \varepsilon , ( \gamma _ { n } ) , \theta _ { 0 } )$ as in the introduction. Let $n _ { 0 } \in  { \mathbb { N } } _ { 0 }$ and let τ be the stopping time given by

$$
\tau = \operatorname* { i n f } \{ n \geq n _ { 0 } : \theta _ { n } \notin \mathcal { V } \} .\tag{53}
$$

One has for every $n \in \mathbb { N } _ { 0 } \cap [ n _ { 0 } , \infty ) , i \in \{ 1 , \dots , d \} , \lambda \in [ 0 , \infty )$ that

$$
- \ln \mathbb { E } \big [ e ^ { - \lambda ( v _ { n } ^ { ( i ) } + \mathbf { 1 } _ { \{ n \leq \tau \} ^ { c } } \times \infty ) } | \mathcal { F } _ { n _ { 0 } } \big ] \geq \sum _ { k = n _ { 0 } + 1 } ^ { n } \Big ( c \frac { 1 - \beta } { 1 - \beta ^ { n } } \beta ^ { n - k } \lambda \Big ) \wedge q + \frac { 1 - \beta ^ { n _ { 0 } } } { 1 - \beta ^ { n } } \beta ^ { n - n _ { 0 } } \lambda v _ { n _ { 0 } } ^ { ( i ) } .\tag{54}
$$

In the case $n = n _ { 0 } = 0$ , the last term on the right-hand side is understood as zero.

Proof. Fix $i \in \{ 1 , \ldots , d \} , \lambda \in [ 0 , \infty )$ . The statement is trivial for $n = n _ { 0 }$ . We thus fix $n \in \mathbb { N }$ with $n > n _ { 0 }$ and note that

$$
\begin{array} { l } { { \displaystyle v _ { n } ^ { ( i ) } = \frac { 1 - \beta } { 1 - \beta ^ { n } } \sum _ { k = 1 } ^ { n } \beta ^ { n - k } X ^ { ( i ) } ( \theta _ { k - 1 } , U _ { k } ) ^ { 2 } } } \\ { { \displaystyle \ = \sum _ { k = n _ { 0 } + 1 } ^ { n } \frac { 1 - \beta } { 1 - \beta ^ { n } } \beta ^ { n - k } X ^ { ( i ) } ( \theta _ { k - 1 } , U _ { k } ) ^ { 2 } + v _ { n _ { 0 } } ^ { ( i ) } \frac { 1 - \beta ^ { n _ { 0 } } } { 1 - \beta ^ { n } } \beta ^ { n - n _ { 0 } } . } } \end{array}\tag{55}
$$

We let for $k = n _ { 0 } + 1 , \ldots , n$

$$
Z _ { k } : = \left\{ \begin{array} { l l } { \frac { 1 - \beta } { 1 - \beta ^ { n } } \beta ^ { n - k } X ^ { ( i ) } ( \theta _ { k - 1 } , U _ { k } ) ^ { 2 } , } & { \mathrm { ~ i f ~ } k \le \tau , } \\ { \infty , } & { \mathrm { ~ e l s e . } } \end{array} \right.\tag{56}
$$

Then one has

$$
v _ { n } ^ { ( i ) } + \mathbb { 1 } _ { \{ n \leq \tau \} ^ { c } } \infty = \sum _ { k = n _ { 0 } + 1 } ^ { n } Z _ { k } + v _ { n _ { 0 } } ^ { ( i ) } \frac { 1 - \beta ^ { n _ { 0 } } } { 1 - \beta ^ { n } } \beta ^ { n - n _ { 0 } } .\tag{57}
$$

Moreover,

$$
\begin{array} { r l } & { \mathbb { E } [ e ^ { - \lambda \sum _ { k = n _ { 0 } + 1 } ^ { n } Z _ { k } } \vert \mathcal { F } _ { n - 1 } ] = e ^ { - \lambda \sum _ { k = n _ { 0 } + 1 } ^ { n - 1 } Z _ { k } } \mathbb { 1 } _ { \left\{ n \leq \tau \right\} } \mathbb { E } \Big [ \exp \Big \{ - \lambda \frac { 1 - \beta } { 1 - \beta ^ { n } } X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) ^ { 2 } \Big \} \Big \vert \mathcal { F } _ { n - 1 } \Big ] } \\ & { \qquad \leq e ^ { - \lambda \sum _ { k = n _ { 0 } + 1 } ^ { n - 1 } Z _ { k } } \exp \Big \{ - \Big ( c \frac { 1 - \beta } { 1 - \beta ^ { n } } \lambda \Big ) \wedge q \Big \} . } \end{array}\tag{58}
$$

Iterating yields that

$$
\mathbb { E } \big [ e ^ { - \lambda \sum _ { k = n _ { 0 } + 1 } ^ { n } Z _ { k } } \big | \mathcal { F } _ { n _ { 0 } } \big ] \leq \exp \Big \{ - \sum _ { k = n _ { 0 } + 1 } ^ { n } \Big ( c \frac { 1 - \beta } { 1 - \beta ^ { n } } \beta ^ { n - k } \lambda \Big ) \wedge q \Big \} .\tag{59}
$$

Consequently, one has that

$$
\begin{array} { l } { \displaystyle \mathbb { E } \big [ e ^ { - \lambda ( v _ { n } ^ { ( i ) } + \mathbf { 1 } _ { \{ n \leq \tau \} } \mathrm { c o s } ) } | \mathcal { F } _ { n _ { 0 } } \big ] = e ^ { - \lambda v _ { n _ { 0 } } ^ { ( i ) } \frac { 1 - \beta ^ { n _ { 0 } } } { 1 - \beta ^ { n } } \beta ^ { n - n _ { 0 } } } \mathbb { E } \big [ e ^ { - \lambda \sum _ { k = n _ { 0 } + 1 } ^ { n } Z _ { k } } | \mathcal { F } _ { n _ { 0 } } \big ] } \\ { \displaystyle \leq \exp \Big \{ - \sum _ { k = n _ { 0 } + 1 } ^ { n } \big ( c \frac { 1 - \beta } { 1 - \beta ^ { n } } \beta ^ { n - k } \lambda \big ) \wedge q - \frac { 1 - \beta ^ { n _ { 0 } } } { 1 - \beta ^ { n } } \beta ^ { n - n _ { 0 } } \lambda v _ { n _ { 0 } } ^ { ( i ) } \Big \} . } \end{array}\tag{60}
$$

This finishes the proof.

Next, we will deduce a negative moment estimate for random variables satisfying an exponential moment estimate like (54). The moment estimate will make use of the following technical result.

Lemma 4.3. Let $\beta \in [ 0 , 1 )$ . The function

$$
f : ( 0 , \infty ) \to ( 0 , \infty ) , \qquad f ( x ) = { \frac { 1 - \beta ^ { x } } { x } }\tag{61}
$$

is decreasing. In particular, for every $n \in \mathbb { N }$

$$
{ \frac { 1 - \beta } { 1 - \beta ^ { n } } } \geq { \frac { 1 } { n } } .\tag{62}
$$

Proof. If $\beta = 0$ , then $f ( x ) = x ^ { - 1 }$ for every $x > 0$ , so monotonicity is clear. Now suppose that $\beta \in ( 0 , 1 )$ . For $x > 0$ , one has

$$
f ^ { \prime } ( x ) = { \frac { - x \beta ^ { x } \ln \beta - 1 + \beta ^ { x } } { x ^ { 2 } } } .\tag{63}
$$

Letting $y = - x \ln \beta > 0$ , the numerator becomes

$$
( y + 1 ) e ^ { - y } - 1 .\tag{64}
$$

Since $e ^ { y } > 1 + y$ for all $y > 0$ , we obtain $( y + 1 ) e ^ { - y } < 1$ . Hence the numerator is strictly negative, so that $f ^ { \prime } ( x ) < 0$ for all $x > 0$ . This proves the monotonicity assertion.

For every $n \in \mathbb { N } .$ , monotonicity implies that

$$
{ \frac { 1 - \beta ^ { n } } { n } } = f ( n ) \leq f ( 1 ) = 1 - \beta ,\tag{65}
$$

which implies the stated inequality.

Lemma 4.4 (Inverse moment estimates). Let $c , q , \rho \in ( 0 , \infty ) , p \in [ 1 , \infty ) , \beta \in ( 0 , 1 ) , n , n _ { 0 } \in \mathbb { N } \ s a t i s f y$

$$
q + p \ln \beta \geq \rho\tag{66}
$$

and $n > n _ { 0 }$ . Let $V _ { n } ^ { * }$ be a non-negative random variable which satisfies for all $\lambda \in [ 0 , \infty )$ that

$$
- \ln \mathbb { E } \big [ e ^ { - \lambda V _ { n } ^ { * } } \big ] \geq \sum _ { k = n _ { 0 } + 1 } ^ { n } \Big ( c \frac { 1 - \beta } { 1 - \beta ^ { n } } \beta ^ { n - k } \lambda \Big ) \wedge q + c \frac { 1 - \beta ^ { n _ { 0 } } } { 1 - \beta ^ { n } } \beta ^ { n - n _ { 0 } } \lambda .\tag{67}
$$

Then

$$
\mathbb { E } [ ( V _ { n } ^ { \ast } ) ^ { - p } ] \leq \frac { 1 } { c ^ { p } } D _ { \rho , p } ,\tag{68}
$$

where

$$
D _ { \rho , p } : = \sum _ { r = 0 } ^ { \infty } ( r + 1 ) ^ { p } e ^ { - \rho r } < \infty .\tag{69}
$$

Proof. We can assume without loss of generality that $c = 1$ . Note that for every $x \ge 0$

$$
x ^ { - p } = { \frac { 1 } { \Gamma ( p ) } } \int _ { 0 } ^ { \infty } e ^ { - x \lambda } \lambda ^ { p - 1 } \mathrm { d } \lambda\tag{70}
$$

so that by Fubini’s theorem

$$
\mathbb { E } [ ( V _ { n } ^ { * } ) ^ { - p } ] = \frac { 1 } { \Gamma ( p ) } \mathbb { E } \Big [ \int _ { 0 } ^ { \infty } e ^ { - V _ { n } ^ { * } \lambda } \lambda ^ { p - 1 } \mathrm { d } \lambda \Big ] = \frac { 1 } { \Gamma ( p ) } \int _ { 0 } ^ { \infty } \mathbb { E } \big [ e ^ { - \lambda V _ { n } ^ { * } } \big ] \lambda ^ { p - 1 } \mathrm { d } \lambda .\tag{71}
$$

To bound the integral we distinguish cases and choose

$$
C _ { 1 } = \frac { q ( 1 - \beta ^ { n } ) } { 1 - \beta } \quad \mathrm { a n d } \quad C _ { 2 } = \frac { q ( 1 - \beta ^ { n } ) } { ( 1 - \beta ) \beta ^ { n - n _ { 0 } - 1 } } = C _ { 1 } \beta ^ { - ( n - n _ { 0 } - 1 ) } .\tag{72}
$$

Thus

$$
\begin{array}{c} \mathbb { E } [ ( V _ { n } ^ { \ast } ) ^ { - p } ] = \frac { 1 } { \Gamma ( p ) } \int _ { 0 } ^ { C _ { 1 } } \mathbb { E } [ e ^ { - \lambda V _ { n } ^ { \ast } } ] \lambda ^ { p - 1 } \mathrm { d } \lambda + \frac { 1 } { \Gamma ( p ) } \int _ { C _ { 1 } } ^ { C _ { 2 } } \mathbb { E } [ e ^ { - \lambda V _ { n } ^ { \ast } } ] \lambda ^ { p - 1 } \mathrm { d } \lambda  \\ { + \frac { 1 } { \Gamma ( p ) } \int _ { C _ { 2 } } ^ { \infty } \mathbb { E } [ e ^ { - \lambda V _ { n } ^ { \ast } } ] \lambda ^ { p - 1 } \mathrm { d } \lambda . \quad \quad \quad \quad \quad \quad \quad \quad \quad } \end{array}\tag{73}
$$

1) First, let us estimate the contribution of $\left[ C _ { 1 } , C _ { 2 } \right)$ by slicing this interval. If

$$
C _ { 1 } < C _ { 2 } ,\tag{74}
$$

equivalently if $n \geq n _ { 0 } + 2$ , then for $r = 1 , \ldots , n - n _ { 0 } - 1$ define

$$
I _ { r } : = \big [ \beta ^ { - ( r - 1 ) } C _ { 1 } , \beta ^ { - r } C _ { 1 } \big ) .\tag{75}
$$

These slices form a disjoint partition of $[ C _ { 1 } , C _ { 2 } )$ . For every $\lambda \in I _ { r }$ one has

$$
\boldsymbol { \beta } ^ { - ( r - 1 ) } \boldsymbol { C } _ { 1 } \le \boldsymbol { \lambda } < \boldsymbol { \beta } ^ { - r } \boldsymbol { C } _ { 1 } \quad \mathrm { ~ o r , ~ e q u i v a l e n t l y , ~ } \quad \boldsymbol { \beta } ^ { - ( r - 1 ) } \boldsymbol { q } \le \frac { 1 - \beta } { 1 - \beta ^ { n } } \boldsymbol { \lambda } < \boldsymbol { \beta } ^ { - r } \boldsymbol { q } .\tag{76}
$$

Hence, for $k \in \{ n _ { 0 } + 1 , . . . , n \}$ , one has

$$
\frac { 1 - \beta } { 1 - \beta ^ { n } } \beta ^ { n - k } \lambda \left\{ \begin{array} { l l } { { < q } } & { { \mathrm { i f ~ } k \in \{ n _ 0 + 1 , \ldots , n - r \} , } } \\ { { \geq q } } & { { \mathrm { i f ~ } k \in \{ n - r + 1 , \ldots , n \} , } } \end{array} \right.\tag{77}
$$

so that

$$
\begin{array} { l } { \displaystyle \sum _ { k = n _ { 0 } + 1 } ^ { n } \left( \frac { 1 - \beta } { 1 - \beta ^ { n } } \beta ^ { n - k } \lambda \right) \wedge q = \displaystyle \sum _ { k = n _ { 0 } + 1 } ^ { n - r } \frac { 1 - \beta } { 1 - \beta ^ { n } } \beta ^ { n - k } \lambda + r q } \\ { = \displaystyle \frac { \beta ^ { r } - \beta ^ { n - n _ { 0 } } } { 1 - \beta ^ { n } } \lambda + r q . } \end{array}\tag{78}
$$

Combining this with (67), we obtain

$$
- \ln { \mathbb E } \left[ e ^ { - \lambda V _ { n } ^ { \ast } } \right] \ge r q + \lambda \frac { \beta ^ { r } - \beta ^ { n } } { 1 - \beta ^ { n } } = r q + a _ { n , r } \lambda ,\tag{79}
$$

where

$$
a _ { n , r } : = \frac { \beta ^ { r } - \beta ^ { n } } { 1 - \beta ^ { n } } = \beta ^ { r } \frac { 1 - \beta ^ { n - r } } { 1 - \beta ^ { n } } .\tag{80}
$$

Therefore, we obtain that

$$
\begin{array} { l } { \displaystyle J _ { r } : = \frac { 1 } { \Gamma ( p ) } \int _ { I _ { r } } \mathbb { E } [ e ^ { - \lambda V _ { n } ^ { \star } } ] \lambda ^ { p - 1 } \mathrm { d } \lambda \le \frac { e ^ { - r q } } { \Gamma ( p ) } \int _ { \beta ^ { - ( r - 1 ) } C _ { 1 } } ^ { \beta ^ { - r } C _ { 1 } } e ^ { - a _ { n , r } \lambda } \lambda ^ { p - 1 } \mathrm { d } \lambda } \\ { \displaystyle \ = \frac { e ^ { - r q } } { \Gamma ( p ) } a _ { n , r } ^ { - p } \int _ { a _ { n , r } \beta ^ { - ( r - 1 ) } C _ { 1 } } ^ { a _ { n , r } \beta ^ { - r } C _ { 1 } } e ^ { - u } u ^ { p - 1 } \mathrm { d } u } \\ { \displaystyle \ \le \frac { e ^ { - r q } } { \Gamma ( p ) } a _ { n , r } ^ { - p } \int _ { 0 } ^ { \infty } e ^ { - u } u ^ { p - 1 } \mathrm { d } u = e ^ { - r q } a _ { n , r } ^ { - p } } \\ { \displaystyle = e ^ { - r ( q + p \ln \beta ) } \left( \frac { 1 - \beta ^ { n } } { 1 - \beta ^ { n - r } } \right) ^ { p } . } \end{array}\tag{81}
$$

Using Lemma 4.3 and that $n - r \geq 1$ , we get that

$$
\frac { 1 - \beta ^ { n } } { 1 - \beta ^ { n - r } } \leq \frac { n } { n - r } \leq r + 1 .\tag{82}
$$

Consequently, we can conclude with assumption (66) that

$$
J _ { r } \leq ( r + 1 ) ^ { p } e ^ { - r \left( q + p \ln \beta \right) } \leq ( r + 1 ) ^ { p } e ^ { - \rho r } .\tag{83}
$$

Hence the whole contribution of the middle interval is bounded by

$$
\frac { 1 } { \Gamma ( p ) } \int _ { C _ { 1 } } ^ { C _ { 2 } } \mathbb { E } [ e ^ { - \lambda V _ { n } ^ { \ast } } ] \lambda ^ { p - 1 } \mathrm { d } \lambda = \sum _ { r = 1 } ^ { n - n _ { 0 } - 1 } J _ { r } \leq \sum _ { r = 1 } ^ { n - n _ { 0 } - 1 } ( r + 1 ) ^ { p } e ^ { - \rho r } ,\tag{84}
$$

while for $n = n _ { 0 } + 1$ the interval $[ C _ { 1 } , C _ { 2 } )$ is empty.

2) Next, let $\lambda \geq C _ { 2 }$ . Then all terms in the sum are capped at $q ,$ so

$$
- \ln \mathbb { E } \left[ e ^ { - \lambda V _ { n } ^ { \ast } } \right] \geq ( n - n _ { 0 } ) q + \frac { 1 - \beta ^ { n _ { 0 } } } { 1 - \beta ^ { n } } \beta ^ { n - n _ { 0 } } \lambda .\tag{85}
$$

Therefore,

$$
\begin{array} { r l r } {  { \frac { 1 } { \Gamma ( p ) } \int _ { C _ { 2 } } ^ { \infty } \mathbb { E } [ e ^ { - \lambda V _ { n } ^ { * } } ] \lambda ^ { p - 1 } \mathrm { d } \lambda \leq \frac { e ^ { - ( n - n _ { 0 } ) q } } { \Gamma ( p ) } \int _ { C _ { 2 } } ^ { \infty } \exp ( - \frac { 1 - \beta ^ { n _ { 0 } } } { 1 - \beta ^ { n } } \beta ^ { n - n _ { 0 } } \lambda ) \lambda ^ { p - 1 } \mathrm { d } \lambda } } \\ & { } & { \leq \frac { e ^ { - ( n - n _ { 0 } ) q } } { \Gamma ( p ) } \int _ { 0 } ^ { \infty } \exp ( - \frac { 1 - \beta ^ { n _ { 0 } } } { 1 - \beta ^ { n } } \beta ^ { n - n _ { 0 } } \lambda ) \lambda ^ { p - 1 } \mathrm { d } \lambda } \\ & { } & { = ( \frac { 1 - \beta ^ { n } } { ( 1 - \beta ^ { n _ { 0 } } ) \beta ^ { n - n _ { 0 } } } ) ^ { p } e ^ { - ( n - n _ { 0 } ) q } , } \end{array}\tag{86}
$$

where we have used (70) again. Applying Lemma 4.3, we obtain

$$
\frac { 1 - \beta ^ { n } } { n } \leq \frac { 1 - \beta ^ { n _ { 0 } } } { n _ { 0 } } ~ \mathrm { o r , ~ e q u i v a l e n t l y , } ~ \frac { 1 - \beta ^ { n } } { 1 - \beta ^ { n _ { 0 } } } \leq \frac { n } { n _ { 0 } } .\tag{87}
$$

Hence

$$
\frac { 1 } { \Gamma ( p ) } \int _ { C _ { 2 } } ^ { \infty } \mathbb { E } [ e ^ { - \lambda V _ { n } ^ { \ast } } ] \lambda ^ { p - 1 } \mathrm { d } \lambda \leq \left( \frac { n } { n _ { 0 } } \right) ^ { p } \exp \left( - ( n - n _ { 0 } ) ( q + p \ln \beta ) \right) .\tag{88}
$$

We conclude as above and use that $\begin{array} { r } { \frac { n } { n _ { 0 } } \leq n - n _ { 0 } + 1 } \end{array}$ which implies that

$$
\frac { 1 } { \Gamma ( p ) } \int _ { C _ { 2 } } ^ { \infty } \mathbb { E } [ e ^ { - \lambda V _ { n } ^ { \ast } } ] \lambda ^ { p - 1 } \mathrm { d } \lambda \leq ( n - n _ { 0 } + 1 ) ^ { p } e ^ { - \rho ( n - n _ { 0 } ) }\tag{89}
$$

Thus together with (84) we get that

$$
\frac { 1 } { \Gamma ( p ) } \int _ { C _ { 1 } } ^ { \infty } \mathbb { E } [ e ^ { - \lambda V _ { n } ^ { \ast } } ] \lambda ^ { p - 1 } \mathrm { d } \lambda \leq D _ { \rho , p } .\tag{90}
$$

3) Finally, let $\lambda \in [ 0 , C _ { 1 } ]$ . Then none of the summands are capped, and therefore

$$
\begin{array} { l } { \displaystyle - \ln \mathbb { E } \left[ e ^ { - \lambda V _ { n } ^ { * } } \right] \geq \sum _ { k = n _ { 0 } + 1 } ^ { n } \frac { 1 - \beta } { 1 - \beta ^ { n } } \beta ^ { n - k } \lambda + \frac { 1 - \beta ^ { n _ { 0 } } } { 1 - \beta ^ { n } } \beta ^ { n - n _ { 0 } } \lambda } \\ { \displaystyle \qquad = \lambda \frac { 1 - \beta ^ { n - n _ { 0 } } } { 1 - \beta ^ { n } } + \lambda \frac { \beta ^ { n - n _ { 0 } } - \beta ^ { n } } { 1 - \beta ^ { n } } = \lambda . } \end{array}\tag{91}
$$

Thus

$$
\frac { 1 } { \Gamma ( p ) } \int _ { 0 } ^ { C _ { 1 } } \mathbb { E } [ e ^ { - \lambda V _ { n } ^ { * } } ] \lambda ^ { p - 1 } \mathrm { d } \lambda \leq \frac { 1 } { \Gamma ( p ) } \int _ { 0 } ^ { \infty } e ^ { - \lambda } \lambda ^ { p - 1 } \mathrm { d } \lambda = 1 .\tag{92}
$$

Note that the first summand of the series defining $D _ { \rho , p }$ is one and thus we get with (84) and (89) that

$$
\frac { 1 } { \Gamma ( p ) } \int _ { 0 } ^ { \infty } \mathbb { E } [ e ^ { - \lambda V _ { n } ^ { * } } ] \lambda ^ { p - 1 } \mathrm { d } \lambda \leq \sum _ { r = 0 } ^ { n - n _ { 0 } } ( r + 1 ) ^ { p } e ^ { - \rho r } \leq D _ { \rho , p } .\tag{93}
$$

Lemma 4.5. Let ${ c , q , \rho \in ( 0 , \infty ) , p \in [ 1 , \infty ) , \mathrm { ~ } \beta \in ( 0 , 1 ) }$ satisfy

$$
q + p \ln \beta \geq \rho ,\tag{94}
$$

let $\nu \subseteq \mathbb { R } ^ { d }$ be measurable, and let (X, U) be (c, q)-regular on V in the sense of Definition 4.1. Consider for each $i \in \{ 1 , \ldots , d \}$ the stopping time

$$
\begin{array} { r } { \bar { \tau } = \operatorname* { i n f } \big \{ n \in \mathbb { N } _ { 0 } : \theta _ { n } \notin \mathcal { V } \big \} \qquad { a n d } \qquad \sigma ^ { ( i ) } = \operatorname* { i n f } \big \{ n \in \mathbb { N } : v _ { n } ^ { ( i ) } \geq c \big \} . } \end{array}\tag{95}
$$

Then for every $i \in \{ 1 , \ldots , d \} , n \in \mathbb { N }$ one has

$$
\mathbb E \big [ ( v _ { n } ^ { ( i ) } ) ^ { - p } \mathbb { 1 } _ { \{ \sigma ^ { ( i ) } \leq n \} } \mathbb { 1 } _ { \{ \bar { \tau } \geq n \} } \big ] \leq c ^ { - p } D _ { \rho , p } ,\tag{96}
$$

where $D _ { \rho , p }$ is given by (69).

Proof. Fix $i \in \{ 1 , \ldots , d \}$ and consider for all $n \in \mathbb { N }$

$$
V _ { n } ^ { ( i ) } = v _ { n } ^ { ( i ) } + \mathbb { 1 } _ { \{ \bar { \tau } < n \} } \infty .\tag{97}
$$

We use the recursion (4) to show that by induction for $m \in \{ 1 , \ldots , n \}$ one has

$$
- \ln \mathbb { E } \big [ \exp \{ - \lambda V _ { n } ^ { ( i ) } \} | \mathcal { F } _ { m } \big ] \ge \sum _ { k = m + 1 } ^ { n } \bigg ( c \frac { 1 - \beta } { 1 - \beta ^ { n } } \beta ^ { n - k } \lambda \bigg ) \wedge q + \frac { 1 - \beta ^ { m } } { 1 - \beta ^ { n } } \beta ^ { n - m } \lambda V _ { m } ^ { ( i ) } .\tag{98}
$$

Indeed, if we set

$$
a _ { n } = { \frac { \beta ( 1 - \beta ^ { n - 1 } ) } { 1 - \beta ^ { n } } } \qquad { \mathrm { a n d } } \qquad b _ { n } = { \frac { 1 - \beta } { 1 - \beta ^ { n } } } ,\tag{99}
$$

then on the event $\{ \bar { \tau } \geq n \}$ one has

$$
V _ { n } ^ { ( i ) } = a _ { n } V _ { n - 1 } ^ { ( i ) } + b _ { n } X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) ^ { 2 } .\tag{100}
$$

Since both sides vanish on the complementary event after applying exp(−λ⋅), this implies

$$
\mathbb E \big [ e ^ { - \lambda V _ { n } ^ { ( i ) } } | \mathcal F _ { n - 1 } \big ] = e ^ { - a _ { n } \lambda V _ { n - 1 } ^ { ( i ) } } \mathbb { 1 } _ { \{ \bar { \tau } \geq n \} } \mathbb E \Big [ e ^ { - b _ { n } \lambda X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) ^ { 2 } } \Big | \mathcal F _ { n - 1 } \Big ]\tag{101}
$$

$$
\leq e ^ { - a _ { n } \lambda V _ { n - 1 } ^ { ( i ) } } \exp \bigl ( - ( c b _ { n } \lambda ) \wedge q \bigr ) ,
$$

where we used the $( c , q )$ -regularity on the event $\left\{ \bar { \tau } \geq n \right\} \subseteq \left\{ \theta _ { n - 1 } \in \mathcal { V } \right\}$ . Therefore,

$$
- \ln \mathbb { E } [ e ^ { - \lambda V _ { n } ^ { ( i ) } } | \mathcal { F } _ { n - 1 } ] \ge \Big ( c \frac { 1 - \beta } { 1 - \beta ^ { n } } \lambda \Big ) \wedge q + \frac { \beta ( 1 - \beta ^ { n - 1 } ) } { 1 - \beta ^ { n } } \lambda V _ { n - 1 } ^ { ( i ) } ,\tag{102}
$$

and iteration yields the displayed estimate, since

$$
\prod _ { j = m + 1 } ^ { n } { \frac { \beta ( 1 - \beta ^ { j - 1 } ) } { 1 - \beta ^ { j } } } = { \frac { 1 - \beta ^ { m } } { 1 - \beta ^ { n } } } \beta ^ { n - m } .\tag{103}
$$

Consequently, on the even $\{ \sigma ^ { ( i ) } = m \}$ , the bound $V _ { m } ^ { ( i ) } \geq c$ holds, with the value +∞ allowed if $\bar { \tau } < m$ and, hence, on this event

$$
- \ln \mathbb { E } \big [ \exp \{ - \lambda V _ { n } ^ { ( i ) } \} \big | \mathcal { F } _ { m } \big ] \ge \sum _ { k = m + 1 } ^ { n } \bigg ( c \frac { 1 - \beta } { 1 - \beta ^ { n } } \beta ^ { n - k } \lambda \bigg ) \wedge q + c \frac { 1 - \beta ^ { m } } { 1 - \beta ^ { n } } \beta ^ { n - m } \lambda .\tag{104}
$$

If $m = n ,$ , then on $\{ \sigma ^ { ( i ) } = m \}$ we have $v _ { n } ^ { ( i ) } \geq c .$ Since the first summand in the definition of $D _ { \rho , p }$ is equal to one, this gives on the latter event that

$$
\mathbb E [ \mathbb { 1 } _ { \{ \bar { \tau } \geq n \} } ( v _ { n } ^ { ( i ) } ) ^ { - p } | \mathcal F _ { n } ] = \mathbb { 1 } _ { \{ \bar { \tau } \geq n \} } ( v _ { n } ^ { ( i ) } ) ^ { - p } \leq c ^ { - p } \leq c ^ { - p } D _ { \rho , p } .\tag{105}
$$

If $m < n$ , then we apply Lemma 4.4 for the conditional distribution of $V _ { n } ^ { ( i ) }$ given $\mathcal { F } _ { m }$ and get that on $\{ \sigma ^ { ( i ) } = m \}$

$$
\mathbb { E } \big [ \mathbb { 1 } _ { \{ \bar { \tau } \geq n \} } ( v _ { n } ^ { ( i ) } ) ^ { - p } | \mathcal { F } _ { m } \big ] = \mathbb { E } \big [ ( V _ { n } ^ { ( i ) } ) ^ { - p } | \mathcal { F } _ { m } \big ] \leq c ^ { - p } D _ { \rho , p } .\tag{106}
$$

The statement follows by decomposing the event $\{ \sigma ^ { ( i ) } \leq n \}$ according to the hitting time:

$$
\mathbb { E } \big [ \mathbb { 1 } _ { \{ \sigma ^ { ( i ) } \leq n \} } \mathbb { 1 } _ { \{ \bar { \tau } \geq n \} } ( v _ { n } ^ { ( i ) } ) ^ { - p } \big ] = \sum _ { m = 1 } ^ { n } \mathbb { E } \big [ \mathbb { 1 } _ { \{ \sigma ^ { ( i ) } = m \} } \mathbb { E } \big [ \mathbb { 1 } _ { \{ \bar { \tau } \geq n \} } ( v _ { n } ^ { ( i ) } ) ^ { - p } \big | \mathcal { F } _ { m } \big ] \big ] \leq c ^ { - p } D _ { \rho , p } .\tag{107}
$$

## 5. Technical estimates for the property in item (ii) of Proposition 2.1

This section controls the quadratic Taylor remainder appearing in item (ii) of Proposition 2.1. The inverse squared conditioner is a decreasing function of the current squared innovation, which yields a nonpositive conditional covariance and permits the two factors to be separated. Combining this observation with a conditional second-moment bound for the innovation and a negative first-moment bound for $v _ { n - 1 }$ gives the required coordinatewise estimate.

Lemma 5.1. One has for all $n \in \mathbb { N } , i \in \{ 1 , \dots , d \}$ that

$$
\mathrm { c o v } \big ( ( [ v _ { n } ^ { ( i ) } ] ^ { 1 / 2 } + \varepsilon ) ^ { - 2 } , X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) ^ { 2 } | \mathcal { F } _ { n - 1 } \big ) \leq 0\tag{108}
$$

on the event where the covariance is well-defined.

Proof. Fix $n \in \mathbb { N }$ and $i \in \{ 1 , \ldots , d \}$ and recall that

$$
v _ { n } ^ { ( i ) } = \frac { \beta ( 1 - \beta ^ { n - 1 } ) } { 1 - \beta ^ { n } } v _ { n - 1 } ^ { ( i ) } + \frac { 1 - \beta } { 1 - \beta ^ { n } } X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) ^ { 2 } .\tag{109}
$$

In terms of the non-negative random variables

$$
A = \frac { \beta ( 1 - \beta ^ { n - 1 } ) } { 1 - \beta ^ { n } } v _ { n - 1 } ^ { ( i ) } , \qquad B = \frac { 1 - \beta } { 1 - \beta ^ { n } } \quad \mathrm { ~ a n d ~ } \quad Y = X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) ^ { 2 }\tag{110}
$$

define the random ${ \mathcal { F } } _ { n }$ <sub>1</sub>-measurable extended-valued function $G ( x ) = ( \sqrt { A + B x } + \varepsilon ) ^ { - 2 }$ for $x \in [ 0 , \infty )$ , where the value is $+ \infty$ if the denominator vanishes. Then

$$
( \sqrt { v _ { n } ^ { ( i ) } } + \varepsilon ) ^ { - 2 } = \frac { 1 } { ( \sqrt { A + B Y } + \varepsilon ) ^ { 2 } } = G ( Y ) .\tag{111}
$$

For every $m \in \mathbb { N } ,$ set

$$
G _ { m } ( x ) = \frac { 1 } { ( \sqrt { A + B x } + \varepsilon + m ^ { - 1 } ) ^ { 2 } } , \qquad x \in [ 0 , \infty ) .\tag{112}
$$

The function $G _ { m }$ is bounded and decreasing. If $Y ^ { \prime }$ is conditionally on $\mathcal { F } _ { n - 1 }$ independent of Y and identically distributed as $Y$ , then

$$
2 \mathrm { c o v } ( G _ { m } ( Y ) , Y | \mathcal { F } _ { n - 1 } ) = \mathbb { E } [ ( G _ { m } ( Y ) - G _ { m } ( Y ^ { \prime } ) ) ( Y - Y ^ { \prime } ) | \mathcal { F } _ { n - 1 } ] \leq 0 .\tag{113}
$$

On the event where cov $\cdot ( G ( Y ) , Y | \mathcal { F } _ { n - 1 } )$ is well-defined, the random variables entering its definition are integrable. Since $G _ { m } \uparrow G$ , conditional monotone convergence yields

$$
\mathrm { c o v } ( G ( Y ) , Y | { \mathcal F } _ { n - 1 } ) = \operatorname* { l i m } _ { m \to \infty } \mathrm { c o v } ( G _ { m } ( Y ) , Y | { \mathcal F } _ { n - 1 } ) \leq 0 .\tag{114}
$$

This finishes the proof.

Proposition 5.2. Let $\beta \in [ 1 / 2 , 1 ) , \varepsilon \in [ 0 , \infty ) , n \in \mathbb { N } \backslash \{ 1 \} , i \in \{ 1 , \ldots , d \} , l e t A _ { n } \in \mathcal { F } _ { n - 1 }$ , and let $C _ { X } ^ { ( 2 ) } , C _ { v } ^ { ( - 1 ) } \in ( 0 , \infty )$ Assume that

$$
\mathbb { E } [ ( X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) ) ^ { 2 } | \mathcal { F } _ { n - 1 } ] \le C _ { X } ^ { ( 2 ) } \qquad o n A _ { n }\tag{115}
$$

and

$$
\mathbb { E } \big [ \mathbb { 1 } _ { A _ { n } } ( v _ { n - 1 } ^ { ( i ) } ) ^ { - 1 } \big ] \leq C _ { v } ^ { ( - 1 ) } .\tag{116}
$$

The negative-moment assumption implies that $v _ { n - 1 } ^ { ( i ) } > 0$ almost surely on $A _ { n } . ~ H \varepsilon = 0 $ , inverse factors used in the proof are evaluated on $A _ { n }$ −and extended by zero to $A _ { n } ^ { c }$ . Then one has

$$
\begin{array} { r } { \mathbb { E } \Big [ \mathbb { 1 } _ { A _ { n } } ( \sqrt { v _ { n } ^ { ( i ) } } + \varepsilon ) ^ { - 2 } X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) ^ { 2 } \Big ] \leq 4 C _ { X } ^ { ( 2 ) } C _ { v } ^ { ( - 1 ) } . } \end{array}\tag{117}
$$

Proof. The negative-moment assumption implies that $v _ { n - 1 } ^ { ( i ) } > 0$ almost surely on $A _ { n }$ . Since $\beta \geq \frac { 1 } { 2 }$ and $n \geq 2$ the recursion for $v _ { n }$ also gives $v _ { n } ^ { ( i ) } > 0$ almost surely on $A _ { n }$ . We extend all inverse factors by zero to $A _ { n } ^ { c }$ . By Lemma 5.1, one has

$$
\mathbb { I } _ { A _ { n } } \mathbb { E } [ ( \sqrt { v _ { n } ^ { ( i ) } } + \varepsilon ) ^ { - 2 } X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) ^ { 2 } | \mathcal { F } _ { n - 1 } ] \le \mathbb { I } _ { A _ { n } } \mathbb { E } [ ( \sqrt { v _ { n } ^ { ( i ) } } + \varepsilon ) ^ { - 2 } | \mathcal { F } _ { n - 1 } ] \mathbb { E } [ X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) ^ { 2 } | \mathcal { F } _ { n - 1 } ] .\tag{118}
$$

The conditional second-moment assumption implies that, on $A _ { n }$

$$
\mathbb { E } [ X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) ^ { 2 } | \mathcal { F } _ { n - 1 } ] \le C _ { X } ^ { ( 2 ) } .\tag{119}
$$

Since $\frac { \beta ( 1 - \beta ^ { n - 1 } ) } { 1 - \beta ^ { n } } \leq 1$ , the recursion for $v _ { n }$ yields

$$
v _ { n } ^ { ( i ) } \geq \frac { \beta \big ( 1 - \beta ^ { n - 1 } \big ) } { 1 - \beta ^ { n } } v _ { n - 1 } ^ { ( i ) } .\tag{120}
$$

Thus

$$
\begin{array} { r l } & { \mathbb { 1 } _ { A _ { n } } \mathbb { E } \big [ ( \sqrt { v _ { n } ^ { ( i ) } } + \varepsilon ) ^ { - 2 } | \mathcal { F } _ { n - 1 } \big ] \le \mathbb { 1 } _ { A _ { n } } \mathbb { E } \big [ ( v _ { n } ^ { ( i ) } ) ^ { - 1 } | \mathcal { F } _ { n - 1 } \big ] } \\ & { \qquad \le \mathbb { 1 } _ { A _ { n } } \displaystyle \frac { 1 - \beta ^ { n } } { \beta \big ( 1 - \beta ^ { n - 1 } \big ) } ( v _ { n - 1 } ^ { ( i ) } ) ^ { - 1 } . } \end{array}\tag{121}
$$

Lemma 4.3 and $\beta \geq { \frac { 1 } { 2 } }$ imply that

$$
{ \frac { 1 - \beta ^ { n } } { \beta ( 1 - \beta ^ { n - 1 } ) } } \leq { \frac { 1 } { \beta } } { \frac { n } { n - 1 } } \leq 4 .\tag{122}
$$

Combining (118), (119), (121), and (122) gives

$$
\mathbb { E } \big [ \mathbb { 1 } _ { A _ { n } } ( \sqrt { v _ { n } ^ { ( i ) } } + \varepsilon ) ^ { - 2 } X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) ^ { 2 } \big ] \leq 4 C _ { X } ^ { ( 2 ) } \mathbb { E } \big [ \mathbb { 1 } _ { A _ { n } } ( v _ { n - 1 } ^ { ( i ) } ) ^ { - 1 } \big ] \leq 4 C _ { X } ^ { ( 2 ) } C _ { v } ^ { ( - 1 ) } .\tag{123}
$$

This proves (117).

## 6. Initialisation estimates

This section estimates the time required for each coordinate of the conditioner to enter the regime where negative moments are available. We first convert the conditional Laplace-transform bounds into lower-tail estimates and then iterate these estimates over blocks whose lengths are comparable to the memory scale $( 1 - \beta ) ^ { - 1 }$ . This yields an exponential bound, uniform in $\beta ,$ , for the probability that a coordinate has not started before the process exits the prescribed region.

Lemma 6.1. Let $c , q \in ( 0 , \infty ) , \ \beta \in ( 0 , 1 ) , \ n \in \mathbb { N }$ . Let $V _ { n } ^ { * }$ be a nonnegative random variable which satisfies for all $\lambda \in [ 0 , \infty )$ that

$$
- \ln \mathbb { E } \big [ e ^ { - \lambda V _ { n } ^ { * } } \big ] \geq \sum _ { k = 1 } ^ { n } \operatorname* { m i n } \Big \{ q , c \frac { 1 - \beta } { 1 - \beta ^ { n } } \beta ^ { n - k } \lambda \Big \} .\tag{124}
$$

Then

$$
\mathbb { P } \big ( V _ { n } ^ { * } \le c / 2 \big ) \le \exp \Big ( - \frac { q } { 2 } \frac { 1 - \beta ^ { n } } { 1 - \beta } \Big ) .\tag{125}
$$

Proof. Set

$$
\lambda _ { * } = \frac { q ( 1 - \beta ^ { n } ) } { c ( 1 - \beta ) } .\tag{126}
$$

By the exponential Chebyshev inequality, we have that

$$
\mathbb { P } ( V _ { n } ^ { \ast } \leq c / 2 ) \leq e ^ { \lambda _ { \ast } c / 2 } \mathbb { E } [ e ^ { - \lambda _ { \ast } V _ { n } ^ { \ast } } ] .\tag{127}
$$

By (124),

$$
\mathbb { E } \bigl [ e ^ { - \lambda _ { * } V _ { n } ^ { * } } \bigr ] \leq \exp \Bigl ( - \sum _ { k = 1 } ^ { n } \Bigl ( c \frac { 1 - \beta } { 1 - \beta ^ { n } } \beta ^ { n - k } \lambda _ { * } \Bigr ) \wedge q \Bigr ) .\tag{128}
$$

Now, for every $k \in \{ 1 , \ldots , n \}$ , one has

$$
c \frac { 1 - \beta } { 1 - \beta ^ { n } } \beta ^ { n - k } \lambda _ { * } = q \beta ^ { n - k } \leq q .\tag{129}
$$

Hence all terms inside the minimum are unsaturated and therefore

$$
\sum _ { k = 1 } ^ { n } \Bigl ( c { \frac { 1 - \beta } { 1 - \beta ^ { n } } } \beta ^ { n - k } \lambda _ { * } \Bigr ) \wedge q = q \sum _ { k = 1 } ^ { n } \beta ^ { n - k } = q { \frac { 1 - \beta ^ { n } } { 1 - \beta } } .\tag{130}
$$

Consequently, we get with (127) that

$$
\mathbb { P } \big ( V _ { n } ^ { * } \le c / 2 \big ) \le \exp \Big ( \lambda _ { * } c / 2 - q \frac { 1 - \beta ^ { n } } { 1 - \beta } \Big ) = \exp \Big ( - \frac { q } { 2 } \frac { 1 - \beta ^ { n } } { 1 - \beta } \Big ) ,\tag{131}
$$

which proves the claim.

Lemma 6.2. Let $c , q \in ( 0 , \infty ) , \ \beta \in ( 0 , 1 ) , n _ { 0 } , n \in \mathbb { N }$ with $n _ { 0 } < n$ . Let $V _ { n } ^ { * }$ be a nonnegative random variable which satisfies for all $\lambda \in [ 0 , \infty )$ that

$$
- \ln \mathbb { E } \big [ e ^ { - \lambda V _ { n } ^ { * } } \big ] \geq \sum _ { k = n _ { 0 } + 1 } ^ { n } \operatorname* { m i n } \Big \{ q , c \frac { 1 - \beta } { 1 - \beta ^ { n } } \beta ^ { n - k } \lambda \Big \} .\tag{132}
$$

Then

$$
\mathbb { P } \Big ( V _ { n } ^ { * } \leq \frac { c } { 2 } \frac { 1 - \beta ^ { n - n _ { 0 } } } { 1 - \beta ^ { n } } \Big ) \leq \exp \Big ( - \frac { q } { 2 } \frac { 1 - \beta ^ { n - n _ { 0 } } } { 1 - \beta } \Big ) .\tag{133}
$$

If, in addition, $\begin{array} { r } { n - n _ { 0 } \geq \frac { 1 } { 2 } ( 1 - \beta ) ^ { - 1 } } \end{array}$ , then

$$
\mathbb { P } \Big ( V _ { n } ^ { * } \leq \frac { c } { 2 } \big ( 1 - e ^ { - 1 / 2 } \big ) \Big ) \leq \exp \Big ( - \frac { q } { 2 } \frac { 1 - e ^ { - 1 / 2 } } { 1 - \beta } \Big ) .\tag{134}
$$

Proof. Set

$$
\lambda _ { * } = \frac { q ( 1 - \beta ^ { n } ) } { c ( 1 - \beta ) } .\tag{135}
$$

By the exponential Chebyshev inequality, we obtain

$$
\mathbb { P } \Big ( V _ { n } ^ { * } \leq \frac { c } { 2 } \frac { 1 - \beta ^ { n - n _ { 0 } } } { 1 - \beta ^ { n } } \Big ) \leq \exp \Big ( \lambda _ { * } \frac { c } { 2 } \frac { 1 - \beta ^ { n - n _ { 0 } } } { 1 - \beta ^ { n } } \Big ) \mathbb { E } [ e ^ { - \lambda _ { * } V _ { n } ^ { * } } ] .\tag{136}
$$

By (132),

$$
\mathbb { E } \big [ e ^ { - \lambda _ { * } V _ { n } ^ { * } } \big ] \leq \exp \Big ( - \sum _ { k = n _ { 0 } + 1 } ^ { n } \Big ( c \frac { 1 - \beta } { 1 - \beta ^ { n } } \beta ^ { n - k } \lambda _ { * } \Big ) \wedge q \Big ) .\tag{137}
$$

Now, for every $k \in \{ n _ { 0 } + 1 , . . . , n \}$ , one has

$$
c \frac { 1 - \beta } { 1 - \beta ^ { n } } \beta ^ { n - k } \lambda _ { * } = q \beta ^ { n - k } \leq q .\tag{138}
$$

Hence all terms inside the minimum are unsaturated and therefore

$$
\sum _ { k = n _ { 0 } + 1 } ^ { n } \Bigl ( c \frac { 1 - \beta } { 1 - \beta ^ { n } } \beta ^ { n - k } \lambda _ { * } \Bigr ) \wedge q = q \sum _ { k = n _ { 0 } + 1 } ^ { n } \beta ^ { n - k } = q \frac { 1 - \beta ^ { n - n _ { 0 } } } { 1 - \beta } .\tag{139}
$$

Consequently,

$$
\begin{array} { l } { \displaystyle \mathbb { P } \Big ( V _ { n } ^ { * } \leq \frac { c } { 2 } \frac { 1 - \beta ^ { n - n _ { 0 } } } { 1 - \beta ^ { n } } \Big ) \leq \exp \Big ( \lambda _ { * } \frac { c } { 2 } \frac { 1 - \beta ^ { n - n _ { 0 } } } { 1 - \beta ^ { n } } - q \frac { 1 - \beta ^ { n - n _ { 0 } } } { 1 - \beta } \Big ) } \\ { \displaystyle \qquad = \exp \Big ( - \frac { q } { 2 } \frac { 1 - \beta ^ { n - n _ { 0 } } } { 1 - \beta } \Big ) , } \end{array}\tag{140}
$$

which proves the first assertion.

For the second assertion, note that $\begin{array} { r } { n - n _ { 0 } \geq \frac { 1 } { 2 } ( 1 - \beta ) ^ { - 1 } } \end{array}$ and hence

$$
\begin{array} { r } { \beta ^ { n - n _ { 0 } } \leq e ^ { - ( 1 - \beta ) ( n - n _ { 0 } ) } \leq e ^ { - 1 / 2 } . } \end{array}\tag{141}
$$

Therefore

$$
\frac { 1 - \beta ^ { n - n _ { 0 } } } { 1 - \beta ^ { n } } \geq 1 - \beta ^ { n - n _ { 0 } } \geq 1 - e ^ { - 1 / 2 }\tag{142}
$$

and

$$
\Big \{ V _ { n } ^ { * } \leq \frac { c } { 2 } \big ( 1 - e ^ { - 1 / 2 } \big ) \Big \} \subseteq \Big \{ V _ { n } ^ { * } \leq \frac { c } { 2 } \frac { 1 - \beta ^ { n - n _ { 0 } } } { 1 - \beta ^ { n } } \Big \} .\tag{143}
$$

Applying the first estimate finishes the proof:

$$
\mathbb { P } \Big ( V _ { n } ^ { * } \leq \frac { c } { 2 } \big ( 1 - e ^ { - 1 / 2 } \big ) \Big ) \leq \exp \Big ( - \frac { q } { 2 } \frac { 1 - e ^ { - 1 / 2 } } { 1 - \beta } \Big ) .\tag{144}
$$

Proposition 6.3. Let $c , q \in ( 0 , \infty ) , \beta \in [ 1 / 2 , 1 )$ , let $\mathcal { V } \subseteq \mathbb { R } ^ { d }$ be measurable, and let $( X , U )$ be $( c , q )$ -regular on V in the sense of Definition 4.1. Consider the stopping time

$$
\begin{array} { r } { \bar { \tau } = \operatorname* { i n f } \big \{ n \in \mathbb { N } _ { 0 } { : } \theta _ { n } \not \in \mathcal { V } \big \} \qquad a n d \qquad \sigma ^ { ( i ) } = \operatorname* { i n f } \big \{ n \in \mathbb { N } { : } v _ { n } ^ { ( i ) } \geq \frac { c ( 1 - e ^ { - 1 / 2 } ) } { 2 } \big \} . } \end{array}\tag{145}
$$

Then for all $n \in \mathbb { N }$ one has

$$
\mathbb { P } \big ( \bar { \tau } \geq n , \sigma ^ { ( i ) } > n \big ) \leq \exp \Bigl ( - \frac { q n ( 1 - e ^ { - 1 / 2 } ) } { 4 } \Bigr ) .\tag{146}
$$

Proof. Set

$$
\eta = \exp \Bigl ( - \frac { q } { 2 } \frac { 1 - e ^ { - 1 / 2 } } { 1 - \beta } \Bigr ) .\tag{147}
$$

First suppose that $n \leq ( 1 - \beta ) ^ { - 1 }$ . Then Proposition 4.2 with $n _ { 0 } = 0$ and Lemma 6.1 imply that

$$
\begin{array} { l } { \displaystyle \mathbb { P } \big ( \bar { \tau } \geq n , \sigma ^ { ( i ) } > n \big ) \leq \mathbb { P } \Big ( \bar { \tau } \geq n , v _ { n } ^ { ( i ) } \leq \frac { 1 - e ^ { - 1 / 2 } } { 2 } c \Big ) } \\ { \leq \mathbb { P } \Big ( \bar { \tau } \geq n , v _ { n } ^ { ( i ) } \leq \frac { c } { 2 } \Big ) \leq \exp \Big ( - \displaystyle \frac { q } { 2 } \frac { 1 - \beta ^ { n } } { 1 - \beta } \Big ) . } \end{array}\tag{148}
$$

For $n = 1$ , this already implies the claim. Thus we may assume that $n \geq 2$ . Since

$$
{ \frac { 1 - \beta ^ { n } } { 1 - \beta } } = \sum _ { k = 0 } ^ { n - 1 } \beta ^ { k } \geq n \beta ^ { n - 1 } ,\tag{149}
$$

and since $n \leq ( 1 - \beta ) ^ { - 1 }$ implies $\begin{array} { r } { \beta \ge 1 - \frac { 1 } { n } } \end{array}$ , we get for $n \geq 2$ that

$$
\beta ^ { n - 1 } \geq \left( 1 - { \frac { 1 } { n } } \right) ^ { n - 1 } = \left( \left( 1 + { \frac { 1 } { n - 1 } } \right) ^ { n - 1 } \right) ^ { - 1 } \geq e ^ { - 1 } \geq { \frac { 1 } { 4 } } .\tag{150}
$$

Consequently, by (148), (149), and (150),

$$
\mathbb { P } \big ( \bar { \tau } \geq n , \sigma ^ { ( i ) } > n \big ) \leq \exp \Bigl ( - \frac { q } { 8 } n \Bigr ) \leq \exp \Bigl ( - \frac { q ( 1 - e ^ { - 1 / 2 } ) } { 4 } n \Bigr ) .\tag{151}
$$

Now suppose that $n > ( 1 - \beta ) ^ { - 1 }$ and set

$$
L = \Big \lceil { \frac { 1 } { 2 ( 1 - \beta ) } } \Big \rceil \quad \mathrm { { ~ a n d ~ } } \quad J = \Big \lfloor { \frac { n } { L } } \Big \rfloor .\tag{152}
$$

Since $\beta \geq \frac { 1 } { 2 }$ , one has

$$
{ \frac { 1 } { 2 ( 1 - \beta ) } } \leq L \leq { \frac { 1 } { 1 - \beta } } .\tag{153}
$$

Moreover, $J \geq 1$ and

$$
J \geq { \frac { n } { 2 L } } \geq { \frac { n ( 1 - \beta ) } { 2 } } .\tag{154}
$$

Setting, for all $j = 2 , \dots , J ,$

$$
t _ { 0 } = 0 \mathrm { a n d } t _ { j } = n - J L + j L ,\tag{155}
$$

we obtain a partition

$$
0 = t _ { 0 } < t _ { 1 } < \cdots < t _ { J } = n\tag{156}
$$

such that all $j = 1 , \dots , J$

$$
t _ { j } - t _ { j - 1 } \geq L \geq \frac { 1 } { 2 ( 1 - \beta ) } .\tag{157}
$$

We now show by induction that

$$
\mathbb { P } \big ( \bar { \tau } \geq t _ { j } , \sigma ^ { ( i ) } > t _ { j } \big ) \leq \eta ^ { j } , \qquad j \in \{ 1 , \ldots , J \} .\tag{158}
$$

For $j = 1$ , Proposition 4.2 with $n _ { 0 } = 0$ and Lemma 6.1 give

$$
\begin{array} { r l r } {  { \mathbb P \big ( \bar { \tau } \geq t _ { 1 } , \sigma ^ { ( i ) } > t _ { 1 } \big ) \leq \mathbb P \Big ( \bar { \tau } \geq t _ { 1 } , v _ { t _ { 1 } } ^ { ( i ) } \leq \frac { 1 - e ^ { - 1 / 2 } } { 2 } c \Big ) } } \\ & { } & { \leq \mathbb P \Big ( \bar { \tau } \geq t _ { 1 } , v _ { t _ { 1 } } ^ { ( i ) } \leq \frac { c } { 2 } \Big ) \leq \exp \Big ( - \frac { q } { 2 } \frac { 1 - \beta ^ { t _ { 1 } } } { 1 - \beta } \Big ) . } \end{array}\tag{159}
$$

Since $\begin{array} { r } { t _ { 1 } \ge \frac { 1 } { 2 ( 1 - \beta ) } } \end{array}$ , we have

$$
\beta ^ { t _ { 1 } } = \exp \{ t _ { 1 } \ln \beta \} \le \exp \{ - t _ { 1 } ( 1 - \beta ) \} \le e ^ { - 1 / 2 }\tag{160}
$$

and hence

$$
\mathbb { P } \big ( \bar { \tau } \geq t _ { 1 } , \sigma ^ { ( i ) } > t _ { 1 } \big ) \leq \eta .\tag{161}
$$

Next let $j \in \{ 2 , \dots , J \}$ and assume that

$$
\mathbb { P } \big ( \bar { \tau } \geq t _ { j - 1 } , \sigma ^ { ( i ) } > t _ { j - 1 } \big ) \leq \eta ^ { j - 1 } .\tag{162}
$$

Set $A _ { j - 1 } = \left\{ \bar { \tau } > t _ { j - 1 } , \sigma ^ { ( i ) } > t _ { j - 1 } \right\}$ . Then

$$
\mathbb { P } \big ( \bar { \tau } \geq t _ { j } , \sigma ^ { ( i ) } > t _ { j } \big ) = \mathbb { E } \Big [ \mathbb { 1 } _ { A _ { j - 1 } } \mathbb { P } \big ( \bar { \tau } \geq t _ { j } , \sigma ^ { ( i ) } > t _ { j } \big | \mathcal { F } _ { t _ { j - 1 } } \big ) \Big ] .\tag{163}
$$

Define the shifted exit time

$$
\bar { \tau } ^ { ( j ) } = \operatorname* { i n f } \{ m \in \mathbb { N } _ { 0 } : m \geq t _ { j - 1 } \mathrm { ~ a n d ~ } \theta _ { m } \notin \mathcal { V } \} .\tag{164}
$$

On $A _ { j - 1 }$ this shifted stopping time agrees with $\bar { \tau } \ \mathrm { u p }$ to the interval $[ t _ { j - 1 } , t _ { j } ]$ . Proposition 4.2, applied condi tionally on $\mathcal { F } _ { t _ { j - 1 } }$ with initial time $t _ { j - 1 }$ and exit time $\bar { \tau } ^ { \left( j \right) }$ , implies that the random variable

$$
V _ { t _ { j } } ^ { * } = v _ { t _ { j } } ^ { ( i ) } + \mathbb { 1 } _ { \{ t _ { j } \leq \bar { \tau } ^ { ( j ) } \} ^ { c } } \infty\tag{165}
$$

satisfies the Laplace bound in Lemma 6.2. Since

$$
t _ { j } - t _ { j - 1 } \geq \frac { 1 } { 2 ( 1 - \beta ) } ,\tag{166}
$$

Lemma 6.2 yields, on $A _ { j - 1 }$ 2

$$
\mathbb { P } \Big ( V _ { t _ { j } } ^ { \ast } \leq \frac { c } { 2 } \big ( 1 - e ^ { - 1 / 2 } \big ) \Big | \mathcal { F } _ { t _ { j - 1 } } \Big ) \leq \eta .\tag{167}
$$

Moreover, on $A _ { j - 1 }$

$$
\big \{ \bar { \tau } \geq t _ { j } , \sigma ^ { ( i ) } > t _ { j } \big \} \subseteq \Big \{ V _ { t _ { j } } ^ { * } \leq \frac { c } { 2 } \big ( 1 - e ^ { - 1 / 2 } \big ) \Big \} .\tag{168}
$$

Consequently, by (163), (167), and (168),

$$
\begin{array} { r } { \mathbb { P } \big ( \bar { \tau } \geq t _ { j } , \sigma ^ { ( i ) } > t _ { j } \big ) \leq \eta \mathbb { P } \big ( A _ { j - 1 } \big ) \leq \eta \mathbb { P } \big ( \bar { \tau } \geq t _ { j - 1 } , \sigma ^ { ( i ) } > t _ { j - 1 } \big ) \leq \eta ^ { j } . } \end{array}\tag{169}
$$

This proves (158).

Since $t _ { J } = n$ , we conclude with (154) that

$$
\mathbb { P } \big ( \bar { \tau } \ge n , \sigma ^ { ( i ) } > n \big ) \le \eta ^ { J } \le \exp \Big ( - \frac { q } { 2 } \frac { 1 - e ^ { - 1 / 2 } } { 1 - \beta } \frac { n \big ( 1 - \beta \big ) } { 2 } \Big ) = \exp \Big ( - \frac { q \big ( 1 - e ^ { - 1 / 2 } \big ) } { 4 } n \Big ) ,\tag{170}
$$

which proves the claim.

## 7. Combining the estimates before and after the coordinate startup times

This section assembles the preceding estimates into the final error bound. At every iteration, each coordinate is split according to whether its startup time has occurred: the negative-moment, covariance, and quadratic estimates are used in the late regime, while direct self-normalization and the startup tail bound control the early regime. Together, these estimates yield the general recursion and, after convolution in numerical time, the main theorem with constants uniform in $\beta$ and ε.

Theorem 7.1. Let $\beta \in [ 1 / 2 , 1 ) , \varepsilon \in [ 0 , \infty ) , \rho , c , q , C _ { X } , L _ { f } , C _ { \mathrm { L o j } } \in ( 0 , \infty )$ , let $V \subseteq \mathbb { R } ^ { d }$ be measurable, and suppose that the following is true:

(a) It holds that

$$
q + 3 \ln \beta \geq \rho .\tag{171}
$$

(b) $( X , U )$ is $( c , q )$ -regular on V in the sense of Definition 4.1 and uniformly bounded: for every $\theta \in V$ $i \in \{ 1 , \ldots , d \}$ one has

$$
\| X ^ { ( i ) } ( \theta , U ) \| _ { L ^ { \infty } } \leq C _ { X } .\tag{172}
$$

(c) There exists $F \colon \mathbb { R } ^ { d } \to [ 0 , \infty )$ with globally L<sub>f</sub>-Lipschitz continuous derivative such that for every $\theta \in V$ one has

$$
f _ { X } \left( \theta \right) = - \nabla F \left( \theta \right)\tag{173}
$$

and

$$
F ( \theta ) \leq C _ { \mathrm { L o j } } | f _ { X } ( \theta ) | ^ { 2 } .\tag{174}
$$

(d) The step-size sequence $( \gamma _ { n } ) _ { n \in \mathbb { N } }$ is $( 0 , \infty )$ -valued and satisfies for all $n \in \mathbb { N } \backslash \{ 1 \}$ that

$$
\gamma _ { n } \leq C _ { \mathrm { L o j } } ( C _ { X } + \varepsilon ) .\tag{175}
$$

Set

$$
\kappa _ { 0 } = \frac { 4 } { 1 - e ^ { - 1 / 2 } } , ~ \ ~ \ ~ \ \kappa _ { 1 } = \frac { 1 } { 2 C _ { \mathrm { L o j } } ( C _ { X } + \varepsilon ) } ,\tag{176}
$$

$$
\kappa _ { 2 } = \frac { 8 \kappa _ { 0 } ^ { 3 } d L _ { f } C _ { X } ^ { 6 } C _ { \mathrm { { L o j } } } ( C _ { X } + \varepsilon ) } { c ^ { 3 } } D _ { \rho , 3 } , \qquad \kappa _ { 3 } = \frac { \kappa _ { 0 } d L _ { f } C _ { X } ^ { 2 } } { c } D _ { \rho , 1 } ,
$$

and, for every $n \in \mathbb { N } \backslash \{ 1 \}$ ,

$$
R _ { n } ^ { \mathrm { e a r l y } } = d \Big ( 2 C _ { X } n ^ { 1 / 2 } + \frac { L _ { f } } { 2 } n \gamma _ { n } \Big ) e ^ { - q ( n - 1 ) / \kappa _ { 0 } } \gamma _ { n } .\tag{177}
$$

For every $n \in  { \mathbb { N } } _ { 0 }$ set $\begin{array} { r } { t _ { n } = \sum _ { k = 1 } ^ { n } \gamma _ { k } } \end{array}$ . Then one has for every $n \in \mathbb { N }$ that

$$
\begin{array} { l } { \mathbb { E } \big [ F ( \theta _ { n } ) \mathbb { 1 } _ { \{ \tau \geq n \} } \big ] } \\ { \leq \big [ F ( \theta _ { 0 } ) ^ { 1 / 2 } + ( 2 ^ { - 1 } d L _ { f } ) ^ { 1 / 2 } \gamma _ { 1 } \big ] ^ { 2 } e ^ { \kappa _ { 1 } ( t _ { 1 } - t _ { n } ) } + \displaystyle \sum _ { j = 2 } ^ { n } \Big [ \kappa _ { 2 } \big [ ( j - 1 ) ^ { - 1 } \vee ( 1 - \beta ) \big ] ^ { 2 } \gamma _ { j } + \kappa _ { 3 } \gamma _ { j } ^ { 2 } + R _ { j } ^ { \mathrm { e a r l y } } \Big ] e ^ { \kappa _ { 1 } ( t _ { j } - t _ { n } ) } , } \end{array}\tag{178}
$$

where τ = inf $\{ n \in \mathbb { N } _ { 0 } { : \theta _ { n } } \not \in V \}$

Proof. If $\theta _ { 0 } \notin V .$ , then $\tau = 0$ and the left-hand side of (178) vanishes. Hence we may assume without loss of generality that $\theta _ { 0 } \in V$ . Set

$$
c _ { v } = ( C _ { X } + \varepsilon ) ^ { - 1 } , \qquad c _ { \mathrm { L o j } } = ( 2 L _ { f } ) ^ { - 1 } .\tag{179}
$$

For every $n \in  { \mathbb { N } } _ { 0 }$ set

$$
\varphi _ { n } = \mathbb E \bigl [ F ( \theta _ { n } ) \mathbb { 1 } _ { \{ \tau \geq n \} } \bigr ] .\tag{180}
$$

For every $x \in \mathbb { R } ^ { d } .$ , the $L _ { f }$ -smoothness inequality, applied with $h = - L _ { f } ^ { - 1 } \nabla F ( x )$ , and the nonnegativity of F give

$$
\begin{array} { r } { 0 \leq F ( x - L _ { f } ^ { - 1 } \nabla F ( x ) ) \leq F ( x ) - \frac { 1 } { 2 L _ { f } } | \nabla F ( x ) | ^ { 2 } , \qquad | \nabla F ( x ) | ^ { 2 } \leq 2 L _ { f } F ( x ) . } \end{array}\tag{181}
$$

Thus the choice of $c _ { \mathrm { L o j } }$ in (179) is admissible by (173) and (181). For every $i \in \{ 1 , \ldots , d \}$ define $\sigma ^ { ( i ) } = \operatorname* { i n f } \{ n \in$ N<sub>∶</sub> $v _ { n } ^ { ( i ) } \geq 2 c / \kappa _ { 0 } \}$ . For every $n \geq 2$ set

$$
h _ { n } = \left( n - 1 \right) ^ { - 1 } \vee \left( 1 - \beta \right) , p _ { n } = e ^ { - q ( n - 1 ) / \kappa _ { 0 } } , A _ { n } = \left\{ \tau \geq n \right\} ,\tag{182}
$$

$$
G _ { n , i } = \{ \sigma ^ { ( i ) } \leq n - 1 \} , \qquad Y _ { n , i } = \big ( \sqrt { v _ { n } ^ { ( i ) } } + \varepsilon \big ) ^ { - 1 } X ^ { ( i ) } \big ( \theta _ { n - 1 } , U _ { n } \big ) .
$$

By (182), the events $A _ { n }$ and $G _ { n , i }$ are ${ \mathcal { F } } _ { n }$ <sub>1</sub>-measurable. On $G _ { n , i }$ , let

$$
\xi _ { n , i } = \mathrm { c o v } \big ( ( \sqrt { v _ { n } ^ { ( i ) } } + \varepsilon ) ^ { - 1 } , X ^ { ( i ) } ( \theta _ { n - 1 } , U _ { n } ) \mid \mathcal { F } _ { n - 1 } \big ) .\tag{183}
$$

When $\varepsilon = 0 .$ , inverse factors and covariances used with $\mathbb { 1 } _ { G _ { n , i } }$ are extended by zero to $G _ { n , i } ^ { c }$

Step 1. Late coordinates satisfy uniform moment bounds, while early coordinates are exponentially rare.

On $A _ { n } , ~ ( 4 ) , ~ ( 1 7 2 )$ , and (182) give $0 \leq v _ { n } ^ { ( i ) } \leq C _ { X } ^ { 2 }$ ; hence, by (179), $( \sqrt { v _ { n } ^ { ( i ) } } + \varepsilon ) ^ { - 1 } \ge c _ { v }$ on $A _ { n } \cap G _ { n , i }$ . Put $\widetilde { c } = 2 c / \kappa _ { 0 }$ . Since $\widetilde { c } < c , \left( \boldsymbol { X } , \boldsymbol { U } \right)$ is also $( \widetilde { c } , q )$ -regular, and the corresponding startup time in Lemma 4.5 is $\boldsymbol { \sigma } ^ { ( i ) }$ Since $A _ { n } \cap G _ { n , i } \subseteq \left\{ \tau \geq n - 1 , \sigma ^ { ( i ) } \leq n - 1 \right\}$ and since $q + r \ln \beta \geq \rho$ for every $r \in \{ 1 , 3 \}$ , Lemma 4.5 gives, for every $r \in \{ 1 , 3 \}$

$$
\mathbb { E } \big [ \mathbb { 1 } _ { A _ { n } \cap G _ { n , i } } ( v _ { n - 1 } ^ { ( i ) } ) ^ { - r } \big ] \leq \widetilde { c } ^ { r } D _ { \rho , r } .\tag{184}
$$

Moreover,

$$
{ \frac { 1 - \beta } { 1 - \beta ^ { n - 1 } } } \leq 2 h _ { n }\tag{185}
$$

follows from Lemma 3.2 and (182). Combining (184) and (185) with Proposition 5.2 and Proposition 3.1 and (172) yields

$$
\mathbb { E } \Big [ \mathbb { 1 } _ { A _ { n } } \sum _ { i = 1 } ^ { d } \mathbb { 1 } _ { G _ { n , i } } Y _ { n , i } ^ { 2 } \Big ] \leq \frac { 2 d \kappa _ { 0 } C _ { X } ^ { 2 } } { c } D _ { \rho , 1 } , \qquad \mathbb { E } \Big [ \mathbb { 1 } _ { A _ { n } } \sum _ { i = 1 } ^ { d } \mathbb { 1 } _ { G _ { n , i } } \xi _ { n , i } ^ { 2 } \Big ] ^ { 1 / 2 } \leq 8 \sqrt { d } C _ { X } ^ { 3 } \Big ( \frac { \kappa _ { 0 } } { 2 c } \Big ) ^ { 3 / 2 } D _ { \rho , 3 } ^ { 1 / 2 } h _ { n } .\tag{186}
$$

Moreover, (4) and (182) give $| Y _ { n , i } | \leq n ^ { 1 / 2 }$ , while Proposition 6.3 and (182) give $\mathbb { P } ( A _ { n } \cap G _ { n , i } ^ { c } ) \ \leq \ p _ { n }$ . Since $| f _ { X } ^ { ( i ) } | \leq C _ { X }$ on V, we obtain

$$
\mathbb { E } \Big [ \mathbb { 1 } _ { A _ { n } } \sum _ { i = 1 } ^ { d } \mathbb { 1 } _ { G _ { n , i } ^ { c } } | f _ { X } ^ { ( i ) } ( \theta _ { n - 1 } ) | ^ { 2 } \Big ] \leq d C _ { X } ^ { 2 } p _ { n } , \quad \mathbb { E } \Big [ \mathbb { 1 } _ { A _ { n } } \sum _ { i = 1 } ^ { d } \mathbb { 1 } _ { G _ { n , i } ^ { c } } | f _ { X } ^ { ( i ) } ( \theta _ { n - 1 } ) Y _ { n , i } | \Big ] \leq d C _ { X } n ^ { 1 / 2 } p _ { n } ,\tag{187}
$$

$$
\quad { \mathrm { a n d } } \quad \quad \mathbb { E } { \Big [ } \mathbb { 1 } _ { A _ { n } } \sum _ { i = 1 } ^ { d } \mathbb { 1 } _ { G _ { n , i } ^ { c } } Y _ { n , i } ^ { 2 } { \Big ] } \leq d n p _ { n } .
$$

Step 2. The coordinate estimates and Proposition 2.1 yield a single Lyapunov recursion.

The lower bound for the inverse factor on $A _ { n } \cap G _ { n , i }$ proves condition $( i )$ of Proposition 2.1. Conditions (ii) and (iii) are given by (186), while the required Lojasiewicz bounds follow from (174), (173), (181), and (179). Since $\theta _ { 0 } \in V$ , the stopping times agree for $n _ { 0 } = 1$ . Hence (22), applied with $n _ { 0 } = 1$ , together with (187) and the definitions in (176), gives

$$
\begin{array} { r l r } {  { \varphi _ { n } \le \Big ( 1 - \frac { c _ { v } } { 2 C _ { \mathrm { L o j } } } \gamma _ { n } \Big ) \mathbb { E } \big [ \mathbb { 1 } _ { A _ { n } } F \big ( \theta _ { n - 1 } \big ) \big ] + \kappa _ { 2 } h _ { n } ^ { 2 } \gamma _ { n } + \kappa _ { 3 } \gamma _ { n } ^ { 2 } } } \\ & { } & { \quad + d \big ( c _ { v } C _ { X } ^ { 2 } + C _ { X } n ^ { 1 / 2 } \big ) p _ { n } \gamma _ { n } + \frac { d L _ { f } } { 2 } n p _ { n } \gamma _ { n } ^ { 2 } . \quad } \end{array}\tag{188}
$$

By (175), the coeficient in the first line is nonnegative, and $\mathbb { E } [ \mathbb { 1 } _ { A _ { n } } F ( \theta _ { n - 1 } ) ] \leq \varphi _ { n - 1 }$ . Since (179) gives $c _ { v } C _ { X } ^ { 2 } \leq$ $C _ { X } \leq C _ { X } n ^ { 1 / 2 }$ , (177) now yields

$$
\varphi _ { n } \leq \bigl ( 1 - \kappa _ { 1 } \gamma _ { n } \bigr ) \varphi _ { n - 1 } + \kappa _ { 2 } h _ { n } ^ { 2 } \gamma _ { n } + \kappa _ { 3 } \gamma _ { n } ^ { 2 } + R _ { n } ^ { \mathrm { e a r l y } } .\tag{189}
$$

Step 3. The scalar recursion can be iterated. For every $i \in \{ 1 , \ldots , d \}$ , equations (4) and (5), applied at $n = 1$ give the first line below; together with (24), this yields the second:

$$
\left| ( \sqrt { v _ { 1 } ^ { ( i ) } } + \varepsilon ) ^ { - 1 } X ^ { ( i ) } ( \theta _ { 0 } , U _ { 1 } ) \right| \le 1 , \qquad \left| \theta _ { 1 } - \theta _ { 0 } \right| \le \sqrt { d } \gamma _ { 1 } .\tag{190}
$$

Equations (23), (173), (181), and (190) yield

$$
F ( \theta _ { 1 } ) \leq F ( \theta _ { 0 } ) + \left| \nabla F ( \theta _ { 0 } ) \right| \left| \theta _ { 1 } - \theta _ { 0 } \right| + \frac { L _ { f } } { 2 } \left| \theta _ { 1 } - \theta _ { 0 } \right| ^ { 2 } \leq \left[ F ( \theta _ { 0 } ) ^ { 1 / 2 } + \Big ( \frac { d L _ { f } } { 2 } \Big ) ^ { 1 / 2 } \gamma _ { 1 } \right] ^ { 2 } .\tag{191}
$$

By (180), $\varphi _ { 1 } \leq \mathbb { E } [ F ( \theta _ { 1 } ) ]$ ; hence (191) controls $\varphi _ { 1 }$ by the initial term in (178). Finally, (175) gives $\kappa _ { 1 } \gamma _ { n } \leq 1 / 2$ Iterating (189) from time 1 and using $1 - x \leq e ^ { - x }$ proves (178). □

Proof of Theorem 1.1. Fix $\varepsilon \in [ 0 , 1 ]$ . The constant $\kappa _ { 1 }$ from Theorem 7.1 is

$$
\kappa _ { 1 } = { \left[ { 2 C _ { \mathrm { L o j } } ( C _ { X } + \varepsilon ) } \right] } ^ { - 1 } .\tag{192}
$$

The assumptions of Theorem 7.1 are satisfied. Since $\varepsilon \leq 1 , ( 7 )$ and (192) give

$$
\eta \leq \kappa _ { 1 } \leq \big [ 2 C _ { \mathrm { L o j } } C _ { X } \big ] ^ { - 1 } .\tag{193}
$$

Set $\overline { { { \gamma } } } = C _ { \mathrm { L o j } } ( C _ { X } + 1 ) = ( 2 \eta ) ^ { - 1 }$ . Then the assumption that $( \gamma _ { n } ) _ { n \in \mathbb { N } }$ is $\left( 0 , C _ { \mathrm { L o } } ( C _ { X } + \varepsilon ) \right]$ -valued and the fact that $\varepsilon \leq 1$ show that, for every $j \geq 2$

$$
\gamma _ { j } \leq { \overline { { \gamma } } } .\tag{194}
$$

By (176), the constant $\kappa _ { 2 }$ satisfies

$$
\kappa _ { 2 } \leq \frac { 8 \kappa _ { 0 } ^ { 3 } d L _ { f } C _ { X } ^ { 6 } C _ { \mathrm { L o j } } ( C _ { X } + 1 ) } { c ^ { 3 } } D _ { \rho , 3 } = : \bar { \kappa } _ { 2 } ,\tag{195}
$$

whereas, by (176) and (177), $\kappa _ { 3 }$ and $R _ { j } ^ { \mathrm { e a r l y } }$ do not depend on ε. Since $\kappa _ { 1 } \geq \eta$ and $e ^ { \kappa _ { 1 } \gamma _ { 1 } } e ^ { - \kappa _ { 1 } t _ { n } } = e ^ { - \kappa _ { 1 } \left( t _ { n } - t _ { 1 } \right) }$ , the general estimate (178) implies, for every $n \geq 2$

$$
\begin{array} { r l r } {  { \mathbb { E } \big [ \mathbb { 1 } _ { \{ \tau \geq n \} } F ( \theta _ { n } ) \big ] \leq \bigg [ F ( \theta _ { 0 } ) ^ { 1 / 2 } + \Big ( \frac { d L _ { f } } { 2 } \Big ) ^ { 1 / 2 } \gamma _ { 1 } \bigg ] ^ { 2 } e ^ { - \eta ( t _ { n } - t _ { 1 } ) } } } \\ & { } & { + \displaystyle \sum _ { j = 2 } ^ { n } \bigg [ \kappa _ { 2 } \Big ( \frac { 1 } { j - 1 } \vee \big ( 1 - \beta \big ) \Big ) ^ { 2 } \gamma _ { j } + \kappa _ { 3 } \gamma _ { j } ^ { 2 } + R _ { j } ^ { \mathrm { e a r l y } } \bigg ] e ^ { - \eta ( t _ { n } - t _ { j } ) } . } \end{array}\tag{196}
$$

For $a \in ( 0 , \infty )$ , the monotonicity of $( \gamma _ { j } ) _ { j \in \mathbb { N } }$ and (194) give

$$
\sum _ { j = 2 } ^ { n } \gamma _ { j } e ^ { - a \left( t _ { n } - t _ { j } \right) } \leq e ^ { a \overline { { \gamma } } } \sum _ { j = 2 } ^ { n } \int _ { t _ { j - 1 } } ^ { t _ { j } } e ^ { - a \left( t _ { n } - s \right) } \mathrm { d } s \leq \frac { e ^ { a \overline { { \gamma } } } } { a } .\tag{197}
$$

Since $\begin{array} { r } { ( \frac { 1 } { j - 1 } \vee ( 1 - \beta ) ) ^ { 2 } \le ( j - 1 ) ^ { - 2 } + ( 1 - \beta ) ^ { 2 } } \end{array}$ , (11) and (197), applied with $a = \eta ,$ yield

$$
\begin{array} { r l } { \displaystyle \sum _ { j = 2 } ^ { n } \Bigl ( \frac { 1 } { j - 1 } \vee ( 1 - \beta ) \Bigr ) ^ { 2 } \gamma _ { j } e ^ { - \eta ( t _ { n } - t _ { j } ) } \le ( 1 - \beta ) ^ { 2 } \displaystyle \sum _ { j = 2 } ^ { n } \gamma _ { j } e ^ { - \eta ( t _ { n } - t _ { j } ) } + C _ { \operatorname* { d e c } } \gamma _ { n } \sum _ { j = 2 } ^ { n } \frac { e ^ { - \eta ( t _ { n } - t _ { j } ) / 2 } } { ( j - 1 ) ^ { 2 } } } & { } \\ { \le \displaystyle \frac { e ^ { 1 / 2 } } { \eta } \left( 1 - \beta \right) ^ { 2 } + C _ { \operatorname* { d e c } } \gamma _ { n } \sum _ { j = 2 } ^ { n } \frac { 1 } { ( j - 1 ) ^ { 2 } } } & { } \\ { } & { = \displaystyle \frac { e ^ { 1 / 2 } } { \eta } \left( 1 - \beta \right) ^ { 2 } + \frac { \pi ^ { 2 } } { 6 } C _ { \operatorname* { d e c } } \gamma _ { n } . } \end{array}\tag{198}
$$

Similarly, (11) and (197), now applied with $a = \eta / 2$ , give

$$
\sum _ { j = 2 } ^ { n } \gamma _ { j } ^ { 2 } e ^ { - \eta ( t _ { n } - t _ { j } ) } \leq C _ { \mathrm { d e c } } \gamma _ { n } \sum _ { j = 2 } ^ { n } \gamma _ { j } e ^ { - \eta ( t _ { n } - t _ { j } ) / 2 } \leq \frac { 2 C _ { \mathrm { d e c } } e ^ { 1 / 4 } } { \eta } \gamma _ { n } .\tag{199}
$$

Equations (177) and (194) give

$$
R _ { j } ^ { \mathrm { e a r l y } } \leq d \left( 2 C _ { X } j ^ { 1 / 2 } + \frac { L _ { f } } { 4 \eta } j \right) e ^ { - q ( j - 1 ) / \kappa _ { 0 } } \gamma _ { j } .\tag{200}
$$

The constant

$$
C _ { \mathrm { e a r l y } } ^ { \mathrm { t i m e } } = d C _ { \mathrm { d e c } } \sum _ { j = 2 } ^ { \infty } \left( 2 C _ { X } j ^ { 1 / 2 } + { \frac { L _ { f } } { 4 \eta } } j \right) e ^ { - q ( j - 1 ) / \kappa _ { 0 } }\tag{201}
$$

is finite. Consequently, (11) and (200) imply

$$
\sum _ { j = 2 } ^ { n } R _ { j } ^ { \mathrm { e a r l y } } e ^ { - \eta ( t _ { n } - t _ { j } ) } \le d C _ { \mathrm { d e c } } \gamma _ { n } \sum _ { j = 2 } ^ { n } \left( 2 C _ { X } j ^ { 1 / 2 } + \frac { L _ { f } } { 4 \eta } j \right) e ^ { - q ( j - 1 ) / \kappa _ { 0 } } e ^ { - \eta ( t _ { n } - t _ { j } ) / 2 } \le C _ { \mathrm { e a r l y } } ^ { \mathrm { t i m e } } \gamma _ { n } .\tag{202}
$$

For $n = 1$ , (12) follows directly from (178). For $n \geq 2 .$ , equations (196), (195), (198), (199), and (201)–(202) prove (12) with the choices

$$
C _ { 0 } = \frac { \pi ^ { 2 } } { 6 } \overline { { { \kappa } } } _ { 2 } C _ { \mathrm { d e c } } + \frac { 2 \kappa _ { 3 } C _ { \mathrm { d e c } } e ^ { 1 / 4 } } { \eta } + C _ { \mathrm { e a r l y } } ^ { \mathrm { t i m e } } , \qquad C _ { 1 } = \frac { \overline { { { \kappa } } } _ { 2 } e ^ { 1 / 2 } } { \eta } ,\tag{203}
$$

which do not depend on $\varepsilon$ or $\beta .$

Remark 7.2 (Non-asymptotic error analysis with fully explicit error constants). In Theorem 1.1 all constants, except of $C _ { 0 }$ and $C _ { 1 }$ , are explicitly specified and, actually, even $C _ { 0 }$ and $C _ { 1 }$ in Theorem 1.1 can be explicitly specified (see (203) in the proof of Theorem 1.1) and explicitly estimated from above. This is precisely the subject of this remark. More formally, combining (7), (69), (176), (195), (201), and (203) shows that

$$
\begin{array} { r l } & { C _ { 0 } = 6 ^ { - 1 } \pi ^ { 2 } \overline { { \kappa } } _ { 2 } C _ { \mathrm { d e c } } + \eta ^ { - 1 } 2 \kappa _ { 3 } C _ { \mathrm { d e c } } e ^ { 1 / 4 } + C _ { \mathrm { e a r l y } } ^ { \mathrm { t i m e } } } \\ & { \quad = 6 ^ { - 1 } \pi ^ { 2 } C _ { \mathrm { d e c } } \bigg [ \frac { 8 \kappa _ { 0 } ^ { 3 } L _ { f } C _ { X } ^ { 6 } C _ { \mathrm { L o j } } ( C _ { X } + 1 ) d } { c ^ { 3 } } \bigg ] D _ { \rho , 3 } + \eta ^ { - 1 } 2 \kappa _ { 3 } C _ { \mathrm { d e c } } \exp \bigl ( 1 / 4 \bigr ) } \\ & { \quad \quad + C _ { \mathrm { d e c } } d \Bigg [ \displaystyle \sum _ { j = 2 } ^ { \infty } \Big ( 2 C _ { X } j ^ { 1 / 2 } + \frac { L _ { f } } { 4 \eta } j \Big ) \exp \bigl ( - q ( j - 1 ) / \kappa _ { 0 } \bigr ) \Bigg ] . } \end{array}\tag{204}
$$

Therefore, we obtain

$$
\begin{array} { l } { { \displaystyle { C _ { 0 } = 6 ^ { - 1 } \pi ^ { 2 } C _ { \mathrm { d e c } } \bigg [ \frac { 4 \kappa _ { 0 } ^ { 3 } L _ { f } C _ { X } ^ { 6 } d } { c ^ { 3 } \eta } \bigg ] \bigg [ \sum _ { r = 0 } ^ { \infty } \frac { ( r + 1 ) ^ { 3 } } { \exp ( \rho r ) } \bigg ] + \bigg [ \frac { 2 \kappa _ { 0 } L _ { f } C _ { X } ^ { 2 } d } { \eta c } \bigg ] \bigg [ \sum _ { r = 0 } ^ { \infty } \frac { ( r + 1 ) } { \exp ( \rho r ) } \bigg ] C _ { \mathrm { d e c } } \exp \big ( 1 / 4 \big ) } } } \\ { { \displaystyle ~ + C _ { \mathrm { d e c } } d \bigg [ \sum _ { i = 2 } ^ { \infty } \bigg ( 2 C _ { X } j ^ { 1 / 2 } + \frac { L _ { f } j } { 4 \eta } \bigg ) \exp \big ( - q ( j - 1 ) / \kappa _ { 0 } \big ) \bigg ] . } } \end{array}\tag{205}
$$

Hence, we obtain

$$
\begin{array} { l } { { \displaystyle C _ { 0 } = C _ { \mathrm { d e c } } d \Bigg [ \sum _ { r = 0 } ^ { \infty } \frac { 2 ^ { 7 } \pi ^ { 2 } c ^ { - 3 } \eta ^ { - 1 } L _ { f } C _ { X } ^ { 6 } ( r + 1 ) ^ { 3 } } { 3 ( 1 - \exp ( - 1 / 2 ) ) ^ { 3 } \exp ( \rho r ) } + \sum _ { r = 0 } ^ { \infty } \frac { 8 \eta ^ { - 1 } c ^ { - 1 } L _ { f } C _ { X } ^ { 2 } ( r + 1 ) } { ( 1 - \exp ( - 1 / 2 ) ) \exp ( \rho r - 1 / 4 ) } } } \\ { { \displaystyle \qquad + \sum _ { j = 2 } ^ { \infty } \Big ( 2 C _ { X } j ^ { 1 / 2 } + \frac { L _ { f } j } { 4 \eta } \Big ) \exp \bigl ( - q ( j - 1 ) ( 1 - \exp ( - 1 / 2 ) ) / 4 \bigr ) \Bigg ] . } } \end{array}\tag{206}
$$

Using $2 7 \pi ^ { 2 } 3 ^ { - 1 } ( 1 - \exp ( - 1 / 2 ) ) ^ { - 3 } \leq 2 ^ { 7 } ( 6 3 ) = 2 ^ { 7 } ( 2 ^ { 6 } - 1 ) = 2 ^ { 1 3 } - 2 ^ { 7 } , \ ( 1 - \exp ( - 1 / 2 ) ) ^ { - 1 } \leq 4 = 2 ^ { 2 } , \ j ^ { 1 / 2 } \leq j ,$ $[ \exp ( - 1 / 4 ) ] ^ { - 1 } \le \exp ( 1 / 2 ) \le 2 , ( 1 - \exp ( - 1 / 2 ) ) / 4 \ge 2 ^ { - 4 }$ , and the change of index $r = j - 1$ in the last series of (206), we obtain

$$
C _ { 0 } \le C _ { \mathrm { d e c } } d \bigl ( L _ { f } + C _ { X } + 1 \bigr ) ^ { 7 } \bigl ( \eta ^ { - 1 } + 1 \bigr ) \left[ \sum _ { r = 0 } ^ { \infty } \left( \frac { \bigl ( \bigl ( 2 ^ { 1 3 } - 2 ^ { 7 } \bigr ) c ^ { - 3 } + 2 ^ { 6 } c ^ { - 1 } \bigr ) \bigl ( r + 1 \bigr ) ^ { 3 } } { \exp ( \rho r ) } + \frac { 2 ^ { 2 } \bigl ( r + 1 \bigr ) } { \exp \bigl ( 2 ^ { - 4 } q r \bigr ) } \right) \right] .\tag{207}
$$

Combining this with the fact that $2 ^ { 6 } c ^ { - 1 } + 2 ^ { 2 } \leq \bigl ( 2 ^ { 6 } + 2 ^ { 2 } \bigr ) \bigl ( c ^ { - 3 } + 1 \bigr ) \leq 2 ^ { 7 } \bigl ( c ^ { - 3 } + 1 \bigr )$ and the fact that for all $r \in \mathbb { N } _ { 0 }$ it holds that $( r + 1 ) \le ( r + 1 ) ^ { 3 }$ proves that

$$
C _ { 0 } \le 2 ^ { 1 3 } C _ { \mathrm { d e c } } d \bigl ( L _ { f } + C _ { X } + 1 \bigr ) ^ { 7 } \bigl ( \eta ^ { - 1 } + 1 \bigr ) \Bigg [ \sum _ { r = 0 } ^ { \infty } \frac { \bigl ( c ^ { - 3 } + 1 \bigr ) \bigl ( r + 1 \bigr ) ^ { 3 } } { \exp \bigl ( \operatorname* { m i n } \bigl \{ \rho , 2 ^ { - 4 } q \bigr \} r \bigr ) } \Bigg ] .\tag{208}
$$

In addition, combining (7), (69), (176), (195), and (203) shows that

$$
\begin{array} { l } { { C _ { 1 } = \overline { { { \kappa } } } _ { 2 } \eta ^ { - 1 } e ^ { 1 / 2 } = \left( c ^ { - 3 } 8 \kappa _ { 0 } ^ { 3 } d L _ { f } C _ { X } ^ { 6 } C _ { \mathrm { I o j } } ( C _ { X } + 1 ) D _ { \rho , 3 } \right) \eta ^ { - 1 } \exp ( 1 / 2 ) = \left( c ^ { - 3 } 4 \kappa _ { 0 } ^ { 3 } d L _ { f } C _ { X } ^ { 6 } D _ { \rho , 3 } \right) \eta ^ { - 2 } \exp ( 1 / 2 ) } } \\ { { \displaystyle = \left[ \frac { 4 \kappa _ { 0 } ^ { 3 } \exp \left( 1 / 2 \right) L _ { f } C _ { X } ^ { 6 } d } { c ^ { 3 } \eta ^ { 2 } } \right] \left[ \displaystyle \sum _ { r = 0 } ^ { \infty } ( r + 1 ) ^ { 3 } e ^ { - \rho r } \right] = \left[ \frac { 4 \kappa _ { 0 } ^ { 3 } \exp \left( 1 / 2 \right) L _ { f } C _ { X } ^ { 6 } d } { c ^ { 3 } \eta ^ { 2 } } \right] \left[ \displaystyle \sum _ { r = 0 } ^ { \infty } ( r + 1 ) ^ { 3 } e ^ { - \rho r } \right] } } \\ { { \displaystyle = \left[ \frac { 4 ^ { 4 } \exp \left( 1 / 2 \right) L _ { f } C _ { X } ^ { 6 } d } { c ^ { 3 } \eta ^ { 2 } \left( 1 - \exp \left( - 1 / 2 \right) \right) ^ { 3 } } \right] \left[ \displaystyle \sum _ { r = 0 } ^ { \infty } \frac { \left( r + 1 \right) ^ { 3 } } { \exp \left( \rho r \right) } \right] = \left[ \frac { 2 ^ { 8 } \exp \left( \rho + 1 / 2 \right) L _ { f } C _ { X } ^ { 6 } d } { c ^ { 3 } \eta ^ { 2 } \left( 1 - \exp \left( - 1 / 2 \right) \right) ^ { 3 } } \right] \left[ \displaystyle \sum _ { r = 1 } ^ { \infty } \frac { r ^ { 3 } } { \exp \left( \rho r \right) } \right] } . } \end{array}\tag{209}
$$

Using exp $( 1 / 2 ) [ 1 - \exp ( - 1 / 2 ) ] ^ { - 3 } \leq 2 ^ { 1 5 }$ we hence obtain that

$$
C _ { 1 } = \bigg [ \frac { 2 ^ { 8 } \exp ( \rho + 1 / 2 ) L _ { f } C _ { X } ^ { 6 } d } { c ^ { 3 } \eta ^ { 2 } \big ( 1 - \exp ( - 1 / 2 ) \big ) ^ { 3 } } \bigg ] \bigg [ \sum _ { r = 0 } ^ { \infty } \frac { r ^ { 3 } } { \exp ( \rho r ) } \bigg ] \leq \sum _ { r = 0 } ^ { \infty } \frac { 2 ^ { 1 3 } \eta ^ { - 2 } c ^ { - 3 } L _ { f } C _ { X } ^ { 6 } d r ^ { 3 } } { \exp ( \rho r - \rho ) } = \sum _ { r = 0 } ^ { \infty } \frac { 2 ^ { 1 3 } d L _ { f } C _ { X } ^ { 6 } \eta ^ { - 2 } c ^ { - 3 } ( r + 1 ) ^ { 3 } } { \exp ( \rho r ) } .\tag{210}
$$

In (206) and (210) we provide exact and explicit representations of the error constants $C _ { 0 }$ and $C _ { 1 }$ in (12) in Theorem 1.1 in terms of the parameters $d , \rho , c , q , C _ { X } , L _ { f } , C _ { \mathrm { d e c } } , \eta$ in Theorem 1.1 (see $( 7 ) )$ and in (208) and (210) we provide short upper bounds for the error constants $C _ { 0 }$ and $C _ { 1 }$ in (12) in Theorem 1.1 in terms of the parameters $d , \rho , c , q , C _ { X } , L _ { f } , C _ { \mathrm { d e c } } , \eta$ in Theorem 1.1 (see (7)).

## 8. Criteria and examples for regular minibatch innovations

We conclude with suficient conditions under which the regularity assumption in Definition 4.1 can be verified for minibatch stochastic gradients. We first reduce regularity to a uniform small-ball estimate. We then establish such an estimate for strongly convex coordinate gradients and apply the resulting criterion to a risk-sensitive objective. Finally, we treat nonlinear regression by a separate argument based on conditional density bounds and negative moments.

8.1. A small-ball criterion for regularity. Let $M \in \mathbb { N } .$ , let $U _ { 1 } , \dots , U _ { M }$ be independent copies of an $\mathbb { R } ^ { m } .$ valued random variable $U ,$ and set $\mathbf { U } = \left( U _ { 1 } , \dots , U _ { M } \right)$ . For a loss $\ell : \mathbb { R } ^ { d } \times \mathbb { R } ^ { m } $ R that is diferentiable in its first argument, the natural minibatch innovation is defined, for every $\theta \in \mathbb { R } ^ { d }$ and $i \in \{ 1 , \ldots , d \}$ , by

$$
X _ { M } ^ { ( i ) } ( \theta , { \bf U } ) = - \frac { 1 } { M } \sum _ { r = 1 } ^ { M } \frac { \partial \ell } { \partial \theta _ { i } } ( \theta , U _ { r } ) .\tag{211}
$$

The following elementary small-ball criterion is convenient.

Lemma 8.1. Let $a \in ( 0 , \infty ) , p \in ( 0 , 1 ]$ , let $\nu \subseteq \mathbb { R } ^ { d }$ , and let $( X , U )$ be an innovation which satisfies for every $\theta \in \mathcal { V } , \ i \in \{ 1 , \dotsc , d \}$ that

$$
\mathbb { P } \big ( | X ^ { ( i ) } ( \theta , U ) | \ge a \big ) \ge p .\tag{212}
$$

Then $( X , U )$ is (c, q)-regular on V with

$$
c = p ( 1 - e ^ { - 1 } ) a ^ { 2 } \qquad a n d \qquad q = p ( 1 - e ^ { - 1 } ) .\tag{213}
$$

Proof. Fix $\theta \in \mathcal { V } , i \in \{ 1 , \ldots , d \}$ , and $\lambda \in [ 0 , \infty )$ . Assumption (212) gives

$$
\begin{array} { r } { \mathbb { E } \Big [ e ^ { - \lambda X ^ { ( i ) } ( \theta , U ) ^ { 2 } } \Big ] \leq 1 - p + p e ^ { - \lambda a ^ { 2 } } = 1 - p \big ( 1 - e ^ { - \lambda a ^ { 2 } } \big ) . } \end{array}\tag{214}
$$

Since $- \ln ( 1 - x ) \geq x$ for $x \in [ 0 , 1 )$ and $1 - e ^ { - x } \geq ( 1 - e ^ { - 1 } ) ( x \wedge 1 )$ for $x \in [ 0 , \infty )$ , (214) yields

$$
\begin{array} { r l r } {  { - \ln \mathbb { E } \big [ e ^ { - \lambda X ^ { ( i ) } ( \theta , U ) ^ { 2 } } \big ] \geq p \big ( 1 - e ^ { - \lambda a ^ { 2 } } \big ) } } \\ & { } & \\ & { } & { \geq p \big ( 1 - e ^ { - 1 } \big ) \big ( \big ( \lambda a ^ { 2 } \big ) \wedge 1 \big ) = \big ( c \lambda \big ) \wedge q , } \end{array}\tag{215}
$$

where c and q are given by (213).

8.2. Strong convexity in the data variable. Our first application uses strong convexity in the data variable. The next estimate combines this property with an upper density bound and will be used to prove Proposition 8.3. The bounded-support assumption provides a uniform bound on the volume of boundary layers of convex sublevel sets. This is needed because an upper density bound alone controls small balls but does not uniformly control boundary layers at arbitrarily high levels.

Lemma 8.2 (Anti-concentration of strongly convex functions). Let $\mu , B , R \in ( 0 , \infty )$ , and suppose that U has a Lebesgue density g on $\mathbb { R } ^ { m }$ such that

$$
\| g \| _ { L ^ { \infty } } \leq B \qquad a n d \qquad \mathbb { P } \big ( | U | < R \big ) = 1 .\tag{216}
$$

Let $\varphi : \mathbb { R } ^ { m }  \mathbb { R }$ be µ-strongly convex. Let $\omega _ { m }$ denote the Lebesgue volume of the Euclidean unit ball in $\mathbb { R } ^ { m }$ . For every $\varepsilon \in ( 0 , \infty )$ , set

$$
\begin{array} { r } { \delta _ { \varepsilon } = 2 \sqrt { \frac { \varepsilon } { \mu } } . } \end{array}\tag{217}
$$

Then, for every $a \in \mathbb { R }$ , one has

$$
\begin{array} { r } { \mathbb { P } \big ( \vert \varphi ( U ) - a \vert \le \varepsilon \big ) \le B \omega _ { m } \big ( \delta _ { \varepsilon } ^ { m } + ( R + 2 \delta _ { \varepsilon } ) ^ { m } - ( R + \delta _ { \varepsilon } ) ^ { m } \big ) . } \end{array}\tag{218}
$$

Proof. Since a finite-valued strongly convex function on $\mathbb { R } ^ { m }$ is continuous and coercive, $\varphi$ has a unique minimizer $u _ { * } .$ . Set $\varphi _ { * } = \varphi ( u _ { * } )$ . First suppose that $a - \varepsilon < \varphi _ { * }$ . If the event in (218) is nonempty, then $a + \varepsilon < \varphi _ { * } + 2 \varepsilon$ . Strong convexity gives that for all $u \in \mathbb { R } ^ { m }$

$$
\varphi ( u ) \geq \varphi _ { * } + \frac \mu 2 | u - u _ { * } | ^ { 2 } .\tag{219}
$$

It follows from (217) and (219) that

$$
\left\{ u \in \mathbb { R } ^ { m } : | \varphi ( u ) - a | \leq \varepsilon \right\} = \left\{ u \in \mathbb { R } ^ { m } : \varphi ( u ) \leq \varphi _ { * } + 2 \varepsilon \right\} \subseteq B ( u _ { * } , \delta _ { \varepsilon } ) .\tag{220}
$$

Therefore, (216) and (220) give

$$
\mathbb { P } \big ( | \varphi ( U ) - a | \le \varepsilon \big ) \le \mathbb { P } \big ( U \in B ( u _ { * } , \delta _ { \varepsilon } ) \big ) \le B \omega _ { m } \delta _ { \varepsilon } ^ { m } .\tag{221}
$$

Now suppose that $t = a - \varepsilon \geq \varphi _ { * }$ , and set $K _ { t } = \{ u \in \mathbb { R } ^ { m } : \varphi ( u ) \leq t \}$ . The set $K _ { t }$ is nonempty, closed, and convex. Fix $u \in \mathbb { R } ^ { m } \setminus K _ { t }$ and let v be its Euclidean projection onto $K _ { t }$ . Then $\varphi ( v ) = t$ . For every $\alpha \in ( 0 , 1 )$ , the point $( 1 - \alpha ) v +$ αu lies outside $K _ { t }$ , and strong convexity yields

$$
t < \varphi \big ( ( 1 - \alpha ) v + \alpha u \big ) \leq ( 1 - \alpha ) \varphi ( v ) + \alpha \varphi ( u ) - \frac { \mu } { 2 } \alpha ( 1 - \alpha ) | u - v | ^ { 2 } .\tag{222}
$$

Using $\varphi ( v ) = t$ in (222), subtracting t, and dividing by $\alpha > 0$ show that, for every $\alpha \in ( 0 , 1 )$ ,

$$
\varphi ( u ) - t > \frac { \mu } { 2 } ( 1 - \alpha ) | u - v | ^ { 2 } .\tag{223}
$$

Letting α decrease to zero in (223) and recalling that $| u - v | = \mathrm { d i s t } ( u , K _ { t } )$ prove that, for every $\iota \in \mathbb { R } ^ { m } \setminus K _ { t }$

$$
\varphi ( u ) - t \geq \frac { \mu } { 2 } \operatorname { d i s t } ( u , K _ { t } ) ^ { 2 } .\tag{224}
$$

Consequently, (217) and (224) imply, up to a Lebesgue-null boundary,

$$
\left\{ u \in \mathbb { R } ^ { m } : | \varphi ( u ) - a | \leq \varepsilon \right\} \subseteq \left( K _ { t } + \delta _ { \varepsilon } B ( 0 , 1 ) \right) \setminus K _ { t } .\tag{225}
$$

Set $L _ { t } = K _ { t } \cap B ( 0 , R + \delta _ { \varepsilon } )$ . By (216), only this bounded part of $K _ { t }$ can contribute to the event in (225); more precisely,

$$
\big ( ( K _ { t } + \delta _ { \varepsilon } B ( 0 , 1 ) ) \setminus K _ { t } \big ) \cap B ( 0 , R ) \subseteq \left( L _ { t } + \delta _ { \varepsilon } B ( 0 , 1 ) \right) \setminus L _ { t } .\tag{226}
$$

Since $L _ { t } \subseteq B ( 0 , R + \delta _ { \varepsilon } )$ , the Steiner formula and monotonicity of the intrinsic volumes, together with (226), yield

$$
\left| \left( ( K _ { t } + \delta _ { \varepsilon } B ( 0 , 1 ) ) \setminus K _ { t } \right) \cap B ( 0 , R ) \right| \leq \omega _ { m } \left( ( R + 2 \delta _ { \varepsilon } ) ^ { m } - ( R + \delta _ { \varepsilon } ) ^ { m } \right) .\tag{227}
$$

Equations (216), (225), and (227) prove (218).

Proposition 8.3 (Strongly convex coordinate gradients). Let $m , d , M \in \mathbb { N } , \ \mu , B , R \in ( 0 , \infty )$ , and let $\nu \subseteq \mathbb { R } ^ { d }$ Let U be $a n \mathbb { R } ^ { m }$ -valued random variable with a Lebesgue density g satisfying

$$
\| g \| _ { L ^ { \infty } } \leq B , \qquad a n d \qquad \mathbb { P } { \bigl ( } | U | < R { \bigr ) } = 1 .\tag{228}
$$

Let $U _ { 1 } , \dots , U _ { M }$ be independent copies of $U ,$ and set $\mathbf { U } = \left( U _ { 1 } , \dots , U _ { M } \right)$ . Let $\ell : \mathbb { R } ^ { d } \times \mathbb { R } ^ { m }  \mathbb { R }$ be diferentiable in its first argument. For every $\theta \in \mathcal { V } , \ i \in \{ 1 , \dots , d \}$ , define

$$
\varphi _ { \theta , i } \colon \mathbb { R } ^ { m }  \mathbb { R } , \qquad \varphi _ { \theta , i } ( u ) = \frac { \partial \ell } { \partial \theta _ { i } } ( \theta , u ) ,\tag{229}
$$

and assume that $\varphi _ { \boldsymbol { \theta } , i }$ is µ-strongly convex. For every $\theta \in \mathcal { V } , \ i \in \{ 1 , \dots , d \}$ , define

$$
X _ { M } ^ { ( i ) } ( \theta , { \bf U } ) = - \frac { 1 } { M } \sum _ { r = 1 } ^ { M } \frac { \partial \ell } { \partial \theta _ { i } } ( \theta , U _ { r } ) .\tag{230}
$$

Let $\omega _ { m }$ denote the Lebesgue volume of the Euclidean unit ball in $\mathbb { R } ^ { m }$ , and set

$$
C _ { m } = 1 + m 3 ^ { m - 1 }\tag{231}
$$

and

$$
s _ { 0 } = \mu \operatorname* { m i n } \left\{ \frac { R ^ { 2 } } { 4 } , \frac { 1 } { 1 6 B ^ { 2 } \omega _ { m } ^ { 2 } C _ { m } ^ { 2 } R ^ { 2 m - 2 } } \right\} .\tag{232}
$$

Then the minibatch innovation $( X _ { M } , { \bf U } )$ is $( c _ { M } , q )$ -regular on V with

$$
c _ { M } = { \frac { ( 1 - e ^ { - 1 } ) s _ { 0 } ^ { 2 } } { 2 M ^ { 2 } } } \qquad a n d \qquad q = { \frac { 1 - e ^ { - 1 } } { 2 } } .\tag{233}
$$

$H ,$ in addition, the coordinate derivatives of ℓ are uniformly bounded on $\mathcal { V } \times \left\{ u \in \mathbb { R } ^ { m } \colon | u | < R \right\}$ , then assumption (9) in Theorem 1.1 is satisfied for $X _ { M }$

Proof. For every $s \in ( 0 , s _ { 0 } ]$ , set $\begin{array} { r } { \delta _ { s } = 2 \sqrt { \frac { s } { \mu } } } \end{array}$ . Equation (232) then gives $\delta _ { s } \leq R$ . Hence

$$
\begin{array} { c } { \delta _ { s } ^ { m } + ( R + 2 \delta _ { s } ) ^ { m } - ( R + \delta _ { s } ) ^ { m } \leq R ^ { m - 1 } \delta _ { s } + m ( R + 2 \delta _ { s } ) ^ { m - 1 } \delta _ { s } } \\ { \leq C _ { m } R ^ { m - 1 } \delta _ { s } . } \end{array}\tag{234}
$$

$\mathrm { B y }$ (228), Lemma 8.2 applies to every $\varphi _ { \boldsymbol { \theta } , i }$ . Equations (218), (232), and (234) give

$$
\operatorname* { s u p } _ { \theta \in \mathcal { V } , \ 1 \leq i \leq d } \mathbb { P } \big ( | \varphi _ { \theta , i } ( U ) - a | \leq s _ { 0 } \big ) \leq B \omega _ { m } C _ { m } R ^ { m - 1 } \delta _ { s _ { 0 } } = 2 B \omega _ { m } C _ { m } R ^ { m - 1 } \sqrt { \frac { s _ { 0 } } { \mu } } \leq \frac { 1 } { 2 } .\tag{235}
$$

Fix $\theta \in \mathcal { V }$ and $i \in \{ 1 , \ldots , d \}$ . By (230) and (229), conditional on $U _ { 2 } , \dots , U _ { M }$ , the event

$$
\left\{ \left| X _ { M } ^ { ( i ) } ( \theta , { \bf U } ) \right| \le \frac { s _ { 0 } } { M } \right\}\tag{236}
$$

is an interval event of width $2 s _ { 0 }$ for $\varphi _ { \theta , i } ( U _ { 1 } )$ , whose center may depend on $U _ { 2 } , \dots , U _ { M }$ . Therefore, (235) and (236) imply

$$
\mathbb { P } \left( \left| X _ { M } ^ { ( i ) } ( \theta , \mathbf { U } ) \right| \geq \frac { s _ { 0 } } { M } \right) \geq \frac { 1 } { 2 } .\tag{237}
$$

Lemma 8.1, applied with $a = s _ { 0 } / M$ and $p = 1 / 2$ , proves the asserted regularity constants. The boundedness assertion follows directly from (230). □

The preceding criterion applies naturally to risk-sensitive objectives in which the parameter changes the exponential weighting of a random energy. Unlike feature-mean estimation, the resulting optimizer depends on the Laplace transform of the energy and hence on more than one fixed moment.

Example 8.4 (Risk-sensitive exponential tilting). Let $\bar { \theta } , \mu , B , R \in ( 0 , \infty )$ , assume (216), and let $\mathcal { V } \subseteq [ 0 , \bar { \theta } ] ^ { d }$ be nonempty and compact. Let $G \colon  { \mathbb { R } ^ { d } } \to  { \mathbb { R } }$ be continuously diferentiable. For every $i \in \{ 1 , \ldots , d \}$ , let $\varphi _ { i } : \mathbb { R } ^ { m } $ $[ 0 , \infty )$ be twice continuously diferentiable and µ-strongly convex. Consider the risk-sensitive loss

$$
\ell _ { \mathrm { r i s k } } ( \theta , u ) = G ( \theta ) + \sum _ { i = 1 } ^ { d } e ^ { \theta _ { i } \varphi _ { i } ( u ) } .\tag{238}
$$

For every $\theta \in \mathcal { V } , i \in \{ 1 , . . . , d \} , u \in \mathbb { R } ^ { m }$ , one has

$$
\frac { \partial \ell _ { \mathrm { r i s k } } } { \partial \theta _ { i } } ( \theta , u ) = \frac { \partial G } { \partial \theta _ { i } } ( \theta ) + \varphi _ { i } ( u ) e ^ { \theta _ { i } \varphi _ { i } ( u ) } .\tag{239}
$$

For symmetric matrices $A , B \in \mathbb { R } ^ { m \times m }$ , write $A \succeq B$ if $A - B$ is positive semidefinite. Moreover, for every $\theta \in \mathcal V$ $i \in \{ 1 , \ldots , d \} , u \in \mathbb { R } ^ { m }$ , one has

$$
\begin{array} { r l } & { \mathrm { H e s s } _ { u } \Big ( \varphi _ { i } ( u ) e ^ { \theta _ { i } \varphi _ { i } ( u ) } \Big ) = e ^ { \theta _ { i } \varphi _ { i } ( u ) } \big ( 1 + \theta _ { i } \varphi _ { i } ( u ) \big ) \mathrm { H e s s } \varphi _ { i } ( u ) } \\ & { \qquad + \theta _ { i } e ^ { \theta _ { i } \varphi _ { i } ( u ) } \big ( 2 + \theta _ { i } \varphi _ { i } ( u ) \big ) \nabla \varphi _ { i } ( u ) \nabla \varphi _ { i } ( u ) ^ { \top } } \\ & { \qquad \succeq \mu I _ { m } . } \end{array}\tag{240}
$$

Indeed, $\theta _ { i } , \varphi _ { i } ( u ) \geq 0$ and Hess $\varphi _ { i } ( u ) \geq \mu I _ { m }$ . Equations (239) and (240) show that every coordinate gradient of (238) is µ-strongly convex as a function of u. Therefore, Proposition 8.3 applies. In particular, with $s _ { 0 }$ defined in (232), the minibatch innovation driven by (238) is $( c _ { M } , q ) \mathrm { - r e g u l a r }$ on V, where $c _ { M }$ and q are given in (233).

The continuity assumptions and the compactness of $\mathcal { V } \times B ( 0 , R )$ also imply that the coordinate gradients in (239) are uniformly bounded on this set. Hence assumption (9) in Theorem 1.1 is satisfied by the resulting minibatch innovation.

For every $\theta \in \mathcal V$ , the associated population objective is given by

$$
F _ { \mathrm { r i s k } } \mathopen { } \mathclose \bgroup \left( \theta \aftergroup \egroup \right) = G \mathopen { } \mathclose \bgroup \left( \theta \aftergroup \egroup \right) + \sum _ { i = 1 } ^ { d } \mathbb { E } \mathopen { } \mathclose \bgroup \left[ e ^ { \theta _ { i } \varphi _ { i } \mathopen { } \mathclose \bgroup \left( U \aftergroup \egroup \right) } \aftergroup \egroup \right] .\tag{241}
$$

The gradient of $F _ { \mathrm { r i s k } }$ is the negative of the mean innovation. For example, if $\begin{array} { r } { G ( \theta ) = - \sum _ { i = 1 } ^ { d } b _ { i } \theta _ { i } } \end{array}$ for constants $b _ { 1 } , \dots , b _ { d } \in \mathbb { R }$ , every interior critical point satisfies, for every $i \in \{ 1 , \ldots , d \}$ ,

$$
\mathbb { E } \Big [ \varphi _ { i } ( U ) e ^ { \theta _ { i } \varphi _ { i } ( U ) } \Big ] = b _ { i } .\tag{242}
$$

Thus the optimized parameters are exponential-tilting parameters determined by the Laplace transforms of the random energies $\varphi _ { i } ( U )$

8.3. Nonlinear regression. We finally consider nonlinear regression with vector-valued responses. Let $k \in \mathbb { N }$ let $\mathcal { Z } \subseteq \mathbb { R } ^ { k }$ be a Borel set, let $U = ( Z , Y )$ take values in ${ \mathcal { Z } } \times \mathbb { R } ^ { m }$ , and let $( \Gamma ^ { \theta } { : } \theta \in \mathbb { R } ^ { d } )$ be a family of measurable maps from Z to $\mathbb { R } ^ { m }$ that is diferentiable with respect to θ. For every $\theta \in \mathbb { R } ^ { d } , z \in \mathcal { Z } , y \in \mathbb { R } ^ { m }$ , and $i \in \{ 1 , \ldots , d \}$ the quadratic loss and its associated coordinate innovation are given by

$$
\begin{array} { l } { \displaystyle \ell _ { \mathrm { r e g } } \big ( \theta , \big ( z , y \big ) \big ) = \frac { 1 } { 2 } \left| \Gamma ^ { \theta } ( z ) - y \right| ^ { 2 } , } \\ { \displaystyle X ^ { ( i ) } ( \theta , ( z , y ) ) = - \left. \frac { \partial \Gamma ^ { \theta } } { \partial \theta _ { i } } ( z ) , \Gamma ^ { \theta } ( z ) - y \right. . } \end{array}\tag{243}
$$

The following result first bounds negative moments of the scalar residual in the direction selected by the derivative of the regression function. A reverse H¨older inequality turns this into a uniform positive lower bound for the first absolute moment. Compactness provides a uniform upper bound for the innovation, and the two moment bounds then yield regularity.

Proposition 8.5 (Compactly supported nonlinear regression). Let $\nu \subseteq \mathbb { R } ^ { d }$ be nonempty and compact, let $r \in ( 0 , 1 / 2 ) , \delta \in ( 0 , \infty )$ , and suppose that there exist nonempty compact sets $K _ { Z } \subseteq { \mathcal { Z } }$ and $K _ { Y } \subseteq \mathbb { R } ^ { m }$ and a constant $B \in ( 0 , \infty )$ with the following properties:

(i) The random variables are compactly supported in the sense that

$$
\mathbb { P } \big ( Z \in K _ { Z } , Y \in K _ { Y } \big ) = 1 .\tag{244}
$$

(ii) For $\mathbb { P } _ { Z } - a l m o s t$ every $z \in K _ { Z }$ , the conditional law of Y given $Z = z$ has a Lebesgue density $p _ { z }$ satisfying

$$
\| p _ { z } \| _ { L ^ { \infty } } \leq B .\tag{245}
$$

(iii) The map $( \theta , z ) \mapsto \Gamma ^ { \theta } ( z )$ and, for every $i \in \{ 1 , \ldots , d \}$ , the map $\begin{array} { r } { ( \theta , z ) \mapsto \frac { \partial \Gamma ^ { \theta } } { \partial \theta _ { i } } ( z ) } \end{array}$ are continuous on $\mathcal { V } \times K _ { Z }$

(iv) The coordinate derivatives satisfy the uniform moment lower bound

$$
D _ { r } = \operatorname* { i n f } _ { \theta \in \mathcal { V } } \operatorname* { i n f } _ { 1 \leq i \leq d } \mathbb { E } \left[ \left| \frac { \partial \Gamma ^ { \theta } } { \partial \theta _ { i } } ( Z ) \right| ^ { r } \right] \geq \delta .\tag{246}
$$

Choose $R \in ( 0 , \infty )$ such that $K _ { Y } \ \subseteq \ B ( 0 , R )$ , let $\omega _ { m - 1 }$ denote the volume of the unit ball in $\mathbb { R } ^ { m - 1 }$ , use the convention $\omega _ { 0 } = 1$ , and set

$$
\begin{array} { l } { \displaystyle \mu _ { r } = \frac { \delta ^ { 1 / r } } { 2 B \omega _ { m - 1 } R ^ { m - 1 } } \left( \frac { 1 - 2 r } { 1 - r } \right) ^ { ( 1 - r ) / r } , } \\ { \displaystyle C _ { X } = \operatorname* { m a x } _ { \theta \in \mathcal { V } , \ z \in K _ { Z } , \ y \in K _ { Y } } \left. \left. \frac { \partial \Gamma ^ { \theta } } { \partial \theta _ { i } } ( z ) , \Gamma ^ { \theta } ( z ) - y \right. \right. . } \end{array}\tag{247}
$$

Then $\mu _ { r } , C _ { X } \in ( 0 , \infty )$ . The innovation (X, U) defined in (243) is $\left( c _ { r } , q _ { r } \right)$ -regular on V with

$$
c _ { r } = { \frac { ( 1 - e ^ { - 1 } ) \mu _ { r } ^ { 3 } } { 8 C _ { X } } } , \qquad q _ { r } = { \frac { ( 1 - e ^ { - 1 } ) \mu _ { r } } { 2 C _ { X } } } .\tag{248}
$$

Proof. Since $r \in ( 0 , 1 / 2 )$ , one has $r / ( 1 - r ) \in ( 0 , 1 )$ . We first prove the uniform first-moment lower bound that drives the argument. Fix $\theta \in \mathcal V$ and $i \in \{ 1 , \ldots , d \}$ . In every normalized derivative below, the quotient is understood as an arbitrary fixed unit vector in $\mathbb { R } ^ { m }$ whenever $\partial \Gamma ^ { \theta } / \partial \theta _ { i }$ vanishes. With this convention the normalized derivative is measurable, has unit norm, and its product with $| \partial \Gamma ^ { \theta } / \partial \theta _ { i } |$ remains the original derivative.

By (244), the conditional density $p _ { z }$ vanishes almost everywhere outside $K _ { Y }$ for $\mathbb { P } _ { Z } .$ -almost every $z \in K _ { Z }$ Fix such a z. After an orthogonal change of coordinates, Fubini’s theorem shows that the conditional density of the residual projected onto the normalized direction specified above is bounded by $B \omega _ { m - 1 } R ^ { m - 1 }$ : the density $p _ { z }$ is bounded by $B$ according to (245), and every hyperplane slice of $K _ { Y } \subseteq B ( 0 , R )$ has $( m - 1 )$ -dimensional Hausdorf measure at most $\omega _ { m - 1 } R ^ { m - 1 }$ . Consequently, for every $h \in ( 0 , \infty )$

$$
\mathbb { P } \bigg ( \bigg | \bigg | \frac { \partial \Gamma ^ { \theta } / \partial \theta _ { i } ( z ) } { | \partial \Gamma ^ { \theta } / \partial \theta _ { i } ( z ) | } , \Gamma ^ { \theta } ( z ) - Y \bigg \rangle \bigg | \leq h \bigg | Z = z \bigg ) \leq 2 B \omega _ { m - 1 } R ^ { m - 1 } h .\tag{249}
$$

Letting h decrease to zero in (249) shows that the projected residual is conditionally nonzero almost surely. For the fixed $\theta , i ,$ and $z ,$ denote its absolute value by W and set $C _ { \mathrm { p r } } = 2 B \omega _ { m - 1 } R ^ { m - 1 }$ . We get with (249) that

$$
\begin{array} { l } { \mathbb { E } [ |  \frac { \partial \Gamma ^ { \theta } / \partial \theta _ { i } ( z ) } { | \partial \Gamma ^ { \theta } / \partial \theta _ { i } ( z ) | } , \Gamma ^ { \theta } ( z ) - Y  | ^ { - r / ( 1 - r ) } | Z = z ] = \int _ { 0 } ^ { \infty } \mathbb { P } \big ( W ^ { - r / ( 1 - r ) } \geq t \big | Z = z \big ) \mathrm { d } t } \\ { = \displaystyle \int _ { 0 } ^ { \infty } \mathbb { P } \big ( W \leq t ^ { - ( 1 - r ) / r } \big | Z = z \big ) \mathrm { d } t \leq \int _ { 0 } ^ { C _ { \operatorname* { p r } } ^ { r / ( 1 - r ) } } 1 \mathrm { d } t + C _ { \operatorname* { p r } } \int _ { C _ { \operatorname* { p r } } ^ { r / ( 1 - r ) } } ^ { \infty } t ^ { - ( 1 - r ) / r } \mathrm { d } t } \\ { = C _ { \operatorname* { p r } } ^ { r / ( 1 - r ) } + \displaystyle \frac { r } { 1 - 2 r } C _ { \operatorname* { p r } } ^ { r / ( 1 - r ) } = \frac { 1 - r } { 1 - 2 r } ( 2 B \omega _ { m - 1 } R ^ { m - 1 } ) ^ { r / ( 1 - r ) } . } \end{array}\tag{250}
$$

Taking expectations in (250) and using (244) give

$$
\mathbb { E } \left[ \left| \left. \frac { \partial \Gamma ^ { \theta } / \partial \theta _ { i } ( Z ) } { | \partial \Gamma ^ { \theta } / \partial \theta _ { i } ( Z ) | } , \Gamma ^ { \theta } ( Z ) - Y \right. \right| ^ { - r / ( 1 - r ) } \right] \leq \frac { 1 - r } { 1 - 2 r } \left( 2 B \omega _ { m - 1 } R ^ { m - 1 } \right) ^ { r / ( 1 - r ) } .\tag{251}
$$

The reverse H¨older inequality used below follows from the ordinary H¨older inequality with conjugate exponents $1 / r$ and $1 / ( 1 - r )$ . Equations (243), (246), and (251) therefore yield

$$
\begin{array} { r l } & { \mathbb { E } \Big [ \big | X ^ { ( i ) } ( \theta , U ) \big | \Big ] = \mathbb { E } \Bigg [ \Bigg | \frac { \partial \Gamma ^ { \theta } } { \partial \theta _ { i } } ( Z ) \Bigg | \Bigg | \Bigg \langle \frac { \partial \Gamma ^ { \theta } / \partial \theta _ { i } ( Z ) } { | \partial \Gamma ^ { \theta } / \partial \theta _ { i } ( Z ) | } , \Gamma ^ { \theta } ( Z ) - Y \Bigg \rangle \Bigg | \Bigg ] } \\ & { \qquad \geq \mathbb { E } \Bigg [ \Bigg | \frac { \partial \Gamma ^ { \theta } } { \partial \theta _ { i } } ( Z ) \Bigg | ^ { r } \Bigg ] ^ { 1 / r } \mathbb { E } \Bigg [ \Bigg | \Bigg \langle \frac { \partial \Gamma ^ { \theta } / \partial \theta _ { i } ( Z ) } { | \partial \Gamma ^ { \theta } / \partial \theta _ { i } ( Z ) | } , \Gamma ^ { \theta } ( Z ) - Y \Bigg \rangle \Bigg | ^ { - r / ( 1 - r ) } \Bigg ] ^ { - ( 1 - r ) / r } } \\ & { \qquad \geq \delta ^ { 1 / r } \left[ \frac { 1 - r } { 1 - 2 r } \left( 2 B \omega _ { m - 1 } R ^ { m - 1 } \right) ^ { r / ( 1 - r ) } \right] ^ { - ( 1 - r ) / r } = \mu _ { r } , } \end{array}\tag{252}
$$

where the last identity follows from (247). Since $\theta$ and $i$ were arbitrary, (252) holds uniformly in $\theta \in \mathcal V$ and $i \in \{ 1 , \ldots , d \}$

The continuity assumptions and the compactness of $\mathcal { V } \times K _ { Z } \times K _ { Y }$ imply that the maximum defining $C _ { X }$ in (247) exists and is finite. Moreover, by (244) and (243), for every $\theta \in \mathcal { V }$ and $i \in \{ 1 , \ldots , d \}$ one has, almost surely,

$$
\left| X ^ { ( i ) } ( \theta , U ) \right| \leq C _ { X } .\tag{253}
$$

Equations (253) and (252) first show that $C _ { X } \geq \mu _ { r } > 0$ . They also imply that, for every $\theta \in \mathcal V$ and $i \in \{ 1 , \ldots , d \}$

$$
\mu _ { r } \leq \mathbb { E } \Big [ | X ^ { ( i ) } ( \theta , U ) | \Big ] \leq \frac { \mu _ { r } } { 2 } + C _ { X } \mathbb { P } \Big ( | X ^ { ( i ) } ( \theta , U ) | \geq \frac { \mu _ { r } } { 2 } \Big ) .\tag{254}
$$

It follows from (254) that, for every $\theta \in \mathcal { V }$ and $i \in \{ 1 , \ldots , d \}$ ，

$$
\mathbb { P } \bigg ( | X ^ { ( i ) } ( \theta , U ) | \geq \frac { \mu _ { r } } { 2 } \bigg ) \geq \frac { \mu _ { r } } { 2 C _ { X } } .\tag{255}
$$

Since $\mu _ { r } / 2 \in ( 0 , \infty )$ and $\mu _ { r } / ( 2 C _ { X } ) \in \mathsf { ( 0 , 1 ] }$ , equation (255) and Lemma 8.1 apply with threshold $\mu _ { r } / 2$ and probability $\mu _ { r } / ( 2 C _ { X } )$ . The constants in (213) are then

$$
\frac { \mu _ { r } } { 2 C _ { X } } \big ( 1 - e ^ { - 1 } \big ) \left( \frac { \mu _ { r } } { 2 } \right) ^ { 2 } = \frac { \big ( 1 - e ^ { - 1 } \big ) \mu _ { r } ^ { 3 } } { 8 C _ { X } } = c _ { r } \quad \mathrm { ~ a n d ~ } \quad \frac { \mu _ { r } } { 2 C _ { X } } \big ( 1 - e ^ { - 1 } \big ) = \frac { \big ( 1 - e ^ { - 1 } \big ) \mu _ { r } } { 2 C _ { X } } = q _ { r } ,\tag{256}
$$

where the last identities follow from (248). This proves the claim.

Proof of Theorem 1.2. Minibatch innovation and regularity. For every n ∈ N, set $\mathbf { Y } _ { n } = ( Y _ { n , 1 } , \ldots , Y _ { n , M } )$ and, for every $i \in \{ 1 , \ldots , d \}$ , set

$$
G _ { n } ^ { ( i ) } = \frac { 1 } { M } \sum _ { r = 1 } ^ { M } ( \nabla _ { \theta _ { i } } L ) ( \theta _ { n - 1 } , Y _ { n , r } ) .\tag{257}
$$

For $\mathbf { y } = ( y _ { 1 } , \dots , y _ { M } ) \in ( \mathbb { R } ^ { m } ) ^ { M }$ , set $\begin{array} { r } { \overline { { y } } _ { M } = M ^ { - 1 } \sum _ { r = 1 } ^ { M } y _ { r } } \end{array}$ and define the innovation

$$
X _ { M } ( \theta , \mathbf { y } ) = - \frac { 1 } { M } \sum _ { r = 1 } ^ { M } \nabla _ { \theta } L ( \theta , y _ { r } ) = 2 A ^ { \mathsf { T } } ( \overline { { y } } _ { M } - A \theta ) .\tag{258}
$$

If $v _ { n } = w _ { n } / ( 1 - \beta ^ { n } )$ for $n \in \mathbb { N }$ and $v _ { 0 } = 0$ , then (16) is precisely the RMSprop algorithm driven by $( X _ { M } , \mathbf { Y } _ { 1 } )$ with parameter tuple $( \beta , \varepsilon , ( \gamma _ { n } ) _ { n \in \mathbb { N } } , \theta _ { 0 } )$

Let $p$ ∈denote a bounded Lebesgue density of $Y _ { 1 , 1 }$ . The average $\begin{array} { r } { \overline { { Y } } _ { 1 , M } = M ^ { - 1 } \sum _ { r = 1 } ^ { M } Y _ { 1 , r } } \end{array}$ is supported by conv(K). Its density is

$$
p _ { M } ( y ) = M ^ { m } p ^ { * M } ( M y ) .\tag{259}
$$

Young’s convolution inequality gives $\| p _ { M } \| _ { L ^ { \infty } } \leq M ^ { m } \| p \| _ { L ^ { \infty } } < \infty$ . Fix $r \in \left( 0 , 1 / 2 \right)$ and set

$$
\begin{array} { r l r } & { V _ { \kappa } = \displaystyle \{ \theta \in \mathbb R ^ { d } : | \theta | \leq \kappa \} , \qquad K _ { Y } = \mathrm { c o n v } ( K ) , \qquad B = M ^ { m } \| p \| _ { L ^ { \infty } } , } & \\ & { \delta = \displaystyle \operatorname* { m i n } _ { 1 \leq i \leq d } | A e _ { i } | ^ { r } > 0 . } & \end{array}\tag{260}
$$

Apply Proposition 8.5 with $\mathcal { Z } = K _ { Z } = \{ 0 \} \subseteq \mathbb { R } , Z = 0 , \Gamma ^ { \theta } ( 0 ) = A \theta$ , response $\overline { { Y } } _ { 1 , M }$ , and the parameters in (260). Its derivative condition holds because

$$
\operatorname* { i n f } _ { \theta \in V _ { \kappa } } \mathbb { E } \Bigg [ \Bigg | \frac { \partial \Gamma ^ { \theta } } { \partial \theta _ { i } } ( Z ) \Bigg | ^ { r } \Bigg ] = \delta .\tag{261}
$$

$$
| X _ { M } ^ { ( i ) } ( \theta , { \bf Y } _ { 1 } ) | \le C _ { X } \quad \mathrm { a l m o s t ~ s u r e l y } .
$$

Proposition 8.5 and (261) establish regularity and boundedness for the innovation in (258) without the factor 2. Scaling by 2 changes regularity parameters $( c , q )$ into $( 4 c , q )$ and doubles the uniform bound. Hence there exist $c _ { \mathrm { l i n } } , q _ { \mathrm { l i n } } , C _ { X } \in ( 0 , \infty )$ such that $( X _ { M } , \mathbf { Y } _ { 1 } )$ is $( c _ { \mathrm { l i n } } , q _ { \mathrm { l i n } } )$ -regular on $V _ { \kappa }$ and, for every $\theta \in V _ { \kappa }$ and $i \in \{ 1 , \ldots , d \}$

(262)

Since A has no zero column, it is nonzero. Define $\lambda _ { * }$ as the smallest positive eigenvalue of $A ^ { \mathsf { T } } A$ . Since an upper bound remains valid when its constant is increased, we may and do assume that

$$
C _ { X } \geq 4 \lambda _ { * } \Gamma .\tag{263}
$$

Choose $\beta _ { 0 } \in ( \operatorname* { m a x } \{ 1 / 2 , e ^ { - q _ { \mathrm { l i n } } / 3 } \} , 1 )$ and set $\rho = q _ { \mathrm { l i n } } + 3 \ln \beta _ { 0 } > 0$

Objective geometry and uniform constants. Let $\theta _ { * } \in \mathbb { R } ^ { d }$ solve the normal equation $A ^ { \mathsf { T } } A \theta _ { * } = A ^ { \mathsf { T } } \mathbb { E } [ Y _ { 1 , 1 } ]$ . Such a solution exists because range $( A ^ { \mathsf { T } } ) = \operatorname { r a n g e } ( A ^ { \mathsf { T } } A )$ , and it minimizes the population risk. Then

$$
F ( \theta ) = | A ( \theta - \theta _ { * } ) | ^ { 2 } .\tag{264}
$$

Equations (258) and (264) show that $f _ { X _ { M } } = - \nabla F$ and, by the spectral theorem, that, for every $\theta \in \mathbb { R } ^ { d }$

$$
F ( \theta ) \leq \frac { 1 } { 4 \lambda _ { * } } | f _ { X _ { M } } ( \theta ) | ^ { 2 } .\tag{265}
$$

Moreover, $\nabla F$ is globally $2 \| A ^ { \mathsf { T } } A \| { - }  { \mathrm { I } }$ ipschitz continuous. Set

$$
C _ { \mathrm { L o j } } = \frac { 1 } { 4 \lambda _ { * } } , \qquad \eta = \frac { 1 } { 2 C _ { \mathrm { L o j } } ( C _ { X } + 1 ) } .\tag{266}
$$

Thus $\eta$ depends only on the fixed problem data. Now fix ${ \mathfrak { C } } \in ( 0 , \infty )$ , set $C _ { \mathrm { d e c } } = \mathfrak { C }$ , and let $C _ { 0 } , C _ { 1 } \in ( 0 , \infty )$ be the constants furnished by Theorem 1.1 for

$$
\begin{array} { r } { c = c _ { \mathrm { l i n } } , \qquad q = q _ { \mathrm { l i n } } , \qquad L _ { f } = 2 \| A ^ { \mathsf { T } } A \| , } \end{array}\tag{267}
$$

and the fixed values of $\rho , C _ { X } , C _ { \mathrm { L o j } } , C _ { \mathrm { d e c } }$ . In particular, $C _ { 0 }$ and $C _ { 1 }$ are independent of $\theta _ { 0 }$ , the particular step-size sequence, $\beta ,$ and ε.

Fix such a step-size sequence, $\beta \in [ 1 / 2 , 1 ) , \varepsilon \in [ 0 , 1 ]$ , and $\theta _ { 0 } \in \mathbb { R } ^ { d }$ , and let $( \theta _ { n } ) _ { n \in  { \mathbb { N } } _ { 0 } }$ be the process in the statement. Set $\tau = \operatorname* { i n f } \left\{ j \in \mathbb { N } _ { 0 } : | \theta _ { j } | > \kappa \right\}$ and, for every $n \in  { \mathbb { N } } _ { 0 }$ , set

$$
{ \mathcal { T } } _ { n } = \{ \tau \geq n \} = \bigcap _ { j = 0 } ^ { n - 1 } \{ | \theta _ { j } | \leq \kappa \} .\tag{268}
$$

Here $\mathcal { T } _ { 0 } = \Omega$ by the empty-intersection convention.

The case $n = 1$ is immediate. Indeed, by (257), $w _ { 1 } ^ { ( i ) } = ( 1 - \beta ) | G _ { 1 } ^ { ( i ) } | ^ { 2 }$ and (5) give $\vert \theta _ { 1 } - \theta _ { 0 } \vert \leq \sqrt { d } \gamma _ { 1 }$ . Thus (264) and the triangle inequality give

$$
F ( \theta _ { 1 } ) ^ { 1 / 2 } \leq F ( \theta _ { 0 } ) ^ { 1 / 2 } + \| A ^ { \mathsf { T } } A \| ^ { 1 / 2 } | \theta _ { 1 } - \theta _ { 0 } | \leq F ( \theta _ { 0 } ) ^ { 1 / 2 } + \sqrt { d \| A ^ { \mathsf { T } } A \| } \gamma _ { 1 } .\tag{269}
$$

Consequently, (269) yields

$$
\begin{array} { r } { \mathbb E \big [ F ( \theta _ { 1 } ) \mathbb { 1 } _ { \mathcal { T } _ { 1 } } \big ] \leq \Big ( F ( \theta _ { 0 } ) ^ { 1 / 2 } + \sqrt { d \| A ^ { \top } A \| } \gamma _ { 1 } \Big ) ^ { 2 } . } \end{array}\tag{270}
$$

The regime $\beta \in \left[ \beta _ { 0 } , 1 \right)$ . For every $n \geq 2$ , the range of the step sizes and (263) give

$$
\gamma _ { n } \leq \Gamma \leq C _ { \mathrm { L o j } } C _ { X } \leq C _ { \mathrm { L o j } } ( C _ { X } + \varepsilon ) .\tag{271}
$$

Moreover, the assumed step-size comparison implies, for every $j , n \in \mathbb { N }$ with $j \leq n$ , that

$$
\gamma _ { j } \leq C _ { \mathrm { d e c } } \gamma _ { n } \exp \left( \frac \eta 2 \sum _ { k = j + 1 } ^ { n } \gamma _ { k } \right) .\tag{272}
$$

If $\beta \in \left[ \beta _ { 0 } , 1 \right)$ , the assumptions of Theorem 1.1 follow from (262), (265), the preceding step-size bounds, and (271); moreover, $q _ { \mathrm { l i n } } + 3 \ln \beta \geq \rho .$ . Thus Theorem 1.1 and (268) show that, for every $n \geq 2$

$$
\mathbb { E } \big [ F ( \theta _ { n } ) \mathbb { 1 } _ { \mathcal { T } _ { n } } \big ] \leq \Big ( F ( \theta _ { 0 } ) ^ { 1 / 2 } + \sqrt { d \| A ^ { \top } A \| } \gamma _ { 1 } \Big ) ^ { 2 } \exp \left( - \eta \sum _ { k = 2 } ^ { n } \gamma _ { k } \right) + C _ { 0 } \gamma _ { n } + C _ { 1 } ( 1 - \beta ) ^ { 2 } .\tag{273}
$$

The regime $\beta \in \left[ 1 / 2 , \beta _ { 0 } \right]$ . The recursion for $w _ { n } ^ { ( i ) }$ and (257) imply $w _ { n } ^ { ( i ) } \geq ( 1 - \beta ) | G _ { n } ^ { ( i ) } | ^ { 2 }$ . Hence, for every $n \in \mathbb { N }$ and $i \in \{ 1 , \ldots , d \}$ ,

$$
\left| \left[ \varepsilon + ( 1 - \beta ^ { n } ) ^ { - 1 / 2 } ( w _ { n } ^ { ( i ) } ) ^ { 1 / 2 } \right] ^ { - 1 } G _ { n } ^ { ( i ) } \right| \leq \sqrt { \frac { 1 - \beta ^ { n } } { 1 - \beta } } \leq \frac { 1 } { \sqrt { 1 - \beta _ { 0 } } } .\tag{274}
$$

The first inequality in (274) also holds when its denominator vanishes, by the convention in (5).

Set

$$
R = \kappa + \frac { \sqrt { d } \mathfrak { e } } { \sqrt { 1 - \beta _ { 0 } } } , \qquad C _ { F } = \operatorname* { s u p } _ { | \theta | \leq R } F ( \theta ) < \infty .\tag{275}
$$

For every $n \geq 2 .$ , one has $| \theta _ { n - 1 } | \leq \kappa$ on $\mathcal { I } _ { n }$ . Consequently, the bound $\gamma _ { n } \leq \mathfrak { C }$ and (274) imply

$$
\left| \theta _ { n } \right| \leq \left| \theta _ { n - 1 } \right| + { \frac { { \sqrt { d } } \gamma _ { n } } { { \sqrt { 1 - \beta _ { 0 } } } } } \leq R \qquad { \mathrm { o n ~ } } \mathcal { T } _ { n } .\tag{276}
$$

Since $( 1 - \beta ) ^ { 2 } \geq ( 1 - \beta _ { 0 } ) ^ { 2 }$ , it follows from (276) that, for every $n \geq 2$

$$
\mathbb { E } \big [ F ( \theta _ { n } ) \mathbb { 1 } _ { \mathbb { Z } _ { n } } \big ] \leq C _ { F } \leq \frac { C _ { F } } { ( 1 - \beta _ { 0 } ) ^ { 2 } } ( 1 - \beta ) ^ { 2 } .\tag{277}
$$

Conclusion. The first term on the right-hand side of (17) is nonnegative. Therefore, (270), (273), and (277) prove (17) for every $n \in \mathbb { N }$ and $\beta \in \left[ 1 / 2 , 1 \right)$ with

$$
c = \operatorname* { m a x } \left\{ C _ { 0 } , C _ { 1 } , \frac { C _ { F } } { ( 1 - \beta _ { 0 } ) ^ { 2 } } \right\} .\tag{278}
$$

This constant may depend on C, but is independent of $\theta _ { 0 }$ , the particular step-size sequence, $\beta ,$ and ε, as required. □

Acknowledgements. This work has been partially funded by the National Science Foundation of China (NSFC) under grant number W2531010. This work has also been partially funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy EXC 2044-390685587, Mathematics M¨unster: Dynamics-Geometry-Structure.

## References

[1] Anas Barakat and Pascal Bianchi. Convergence and dynamical behavior of the Adam algorithm for nonconvex stochastic optimization. SIAM J. Optim., 31(1):244–274, 2021.

[2] Bilel Bensaid, Ga¨el Po¨ette, and Rodolphe Turpault. Convergence of the Iterates for Momentum and RMSProp for Local Smooth Functions: Adaptation is the Key. arXiv:2407.15471, 2024.

[3] Xiangyi Chen, Sijia Liu, Ruoyu Sun, and Mingyi Hong. On the Convergence of A Class of Adam-Type Algorithms for Non-Convex Optimization. In International Conference on Learning Representations (ICLR), 2019.

[4] Soham De, Anirbit Mukherjee, and Enayat Ullah. Convergence guarantees for RMSProp and ADAM in non-convex optimization and an empirical comparison to Nesterov acceleration. arXiv:1807.06766, 2018.

[5] Alexandre D´efossez, Leon Bottou, Francis Bach, and Nicolas Usunier. A Simple Convergence Proof of Adam and Adagrad. Transactions on Machine Learning Research, 2022

[6] Stefen Dereich, Thang Do, and Arnulf Jentzen. Uniform a priori bounds and error analysis for the Adam stochastic gradient descent optimization method. arXiv:2603.18899, 2026.

[7] Stefen Dereich, Thang Do, Arnulf Jentzen, and Philippe von Wurstemberger. Adam symmetry theorem: characterization of the convergence of the stochastic Adam optimizer. arXiv:2511.06675, 2025.

[8] Stefen Dereich, Robin Graeber, and Arnulf Jentzen. Non-convergence of Adam and other adaptive stochastic gradient descent optimization methods for non-vanishing learning rates. arXiv:2407.08100, 2024.

[9] Stefen Dereich, Robin Graeber, Arnulf Jentzen, and Adrian Riekert. Asymptotic stability properties and a priori bounds for Adam and other gradient descent optimization methods. arXiv:2509.10476, 2026.

[10] Stefen Dereich and Arnulf Jentzen. Convergence rates for the Adam optimizer. arXiv:2407.21078, 2024.

[11] Stefen Dereich, Arnulf Jentzen, and Adrian Riekert. Sharp higher order convergence rates for the Adam optimizer. arXiv:2504.19426, 2025.

[12] Naum Dimitrieski, Maria Christine Honecker, Carsten Scherer, and Christian Ebenbauer. Global Stability and Step Size Robustness of RMSProp. arXiv:2603.15823, 2026.

[13] Thang Do, Sonja Hannibal, and Arnulf Jentzen. Non-convergence to global minimizers in data driven supervised deep learning: Adam and stochastic gradient descent optimization provably fail to converge to global minimizers in the training of deep neural networks with ReLU activation. J. Math. Anal. Appl., 564(2):Paper No. 130724, 2026.

[14] Thang Do, Arnulf Jentzen, and Adrian Riekert. Non-convergence to the optimal risk for Adam and stochastic gradient descent optimization in the training of deep neural networks. arXiv:2503.01660, 2025.

[15] John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. J. Mach. Learn. Res., 12:2121–2159, 2011.

[16] S´ebastien Gadat and Ioana Gavra. Asymptotic study of stochastic adaptive algorithms in non-convex landscape. J. Mach. Learn. Res., 23:Paper No. [228], 54, 2022.

[17] Antoine Godichon-Baggioni and Pierre Tarrago. Non asymptotic analysis of Adaptive stochastic gradient algorithms and applications. arXiv:2303.01370, 2023.

[18] Steven Heilman and Sampad Mohanty. On the Convergence of Adam, Revisited. arXiv:2607.03519, 2026.

[19] Yusu Hong and Junhong Lin. On Convergence of Adam for Stochastic Optimization under Relaxed Assumptions. In Advances in Neural Information Processing Systems, volume 37, pages 10827–10877, 2024.

[20] Shokhrukh Ibragimov and Arnulf Jentzen. Unified convergence analysis for gradient descent optimization methods in the training of deep neural networks. arXiv:2607.04233, 2026.

[21] Arnulf Jentzen, Benno Kuckuck, and Philippe von Wurstemberger. Mathematical Introduction to Deep Learning: Methods, Implementations, and Theory. arXiv:2310.20360, 2023.

[22] Arnulf Jentzen and Adrian Riekert. Non-convergence to global minimizers for Adam and stochastic gradient descent optimization and constructions of local minimizers in the training of artificial neural networks. SIAM/ASA J. Uncertain. Quantif., 13(3):1294–1333, 2025.

[23] Arnulf Jentzen and Philippe von Wurstemberger. Lower error bounds for the stochastic gradient descent optimization algorithm: sharp convergence rates for slowly and fast decaying learning rates. J. Complexity, 57:101438, 16, 2020.

[24] Ruinan Jin and Xiaoyu Wang. Asymptotic Convergence and Stability of Adaptive Gradient Methods in Smooth Non-convex Optimization. arXiv:2601.01853, 2026.

[25] Diederik P. Kingma and Jimmy Ba. Adam: A Method for Stochastic Optimization. arXiv:1412.6980, 2014.

[26] Haochuan Li, Alexander Rakhlin, and Ali Jadbabaie. Convergence of Adam Under Relaxed Assumptions. arXiv:2304.13972, 2024.

[27] Jinlan Liu, Dongpo Xu, Huisheng Zhang, and Danilo Mandic. On hyper-parameter selection for guaranteed convergence of RMSProp. Cognitive Neurodynamics, 18(6):3227–3237, 2024.

[28] Ilya Loshchilov and Frank Hutter. Decoupled Weight Decay Regularization. arXiv:1711.05101, 2017.

[29] Chao Ma, Lei Wu, and Weinan E. A Qualitative Study of the Dynamic Behavior for Adaptive Gradient Algorithms. arXiv:2009.06125, 2020.

[30] Sadhika Malladi, Kaifeng Lyu, Abhishek Panigrahi, and Sanjeev Arora. On the SDEs and Scaling Rules for Adaptive Gradient Algorithms. arXiv:2205.10287, 2022.

[31] Mahesh Chandra Mukkamala and Matthias Hein. Variants of RMSProp and Adagrad with Logarithmic Regret Bounds. arXiv:1706.05507, 2017.

[32] Sashank J. Reddi, Satyen Kale, and Sanjiv Kumar. On the Convergence of Adam and Beyond. arXiv:1904.09237, 2019.

[33] Sebastian Ruder. An overview of gradient descent optimization algorithms. arXiv:1609.04747, 2016.

[34] Naichen Shi, Dawei Li, Mingyi Hong, and Ruoyu Sun. RMSprop Converges with Proper Hyper-Parameter. In International Conference on Learning Representations (ICLR), 2021.

[35] Tijmen Tieleman and Geofrey Hinton. Lecture 6.5—RMSProp: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural Networks for Machine Learning, 4(2):26–31, 2012.

[36] Bohan Wang, Jingwen Fu, Huishuai Zhang, Nanning Zheng, and Wei Chen. Closing the Gap Between the Upper Bound and the Lower Bound of Adam’s Iteration Complexity. arXiv:2310.17998, 2023.

[37] Dongpo Xu, Shengdong Zhang, Huisheng Zhang, and Danilo P. Mandic. Convergence of the RMSProp deep learning method with penalty for nonconvex optimization. Neural Networks, 139:17–23, 2021.

[38] Yaxin Yu, Long Chen, and Minfu Feng. Adam-SHANG: A Convergent Adam-Type Method for Stochastic Smooth Convex Optimization. arXiv:2605.12878, 2026.

[39] Manzil Zaheer, Sashank Reddi, Devendra Sachan, Satyen Kale, and Sanjiv Kumar. Adaptive Methods for Nonconvex Optimization. In Advances in Neural Information Processing Systems, volume 31, 2018.

[40] Qi Zhang, Yi Zhou, and Shaofeng Zou. Convergence Guarantees for RMSProp and Adam in Generalized-smooth Non-convex Optimization with Afine Noise Variance. Transactions on Machine Learning Research, 2025.

[41] Yushun Zhang, Congliang Chen, Naichen Shi, Ruoyu Sun, and Zhi-Quan Luo. Adam Can Converge Without Any Modification On Update Rules. arXiv:2208.09632, 2022.

[42] Dongruo Zhou, Jinghui Chen, Yuan Cao, Ziyan Yang, and Quanquan Gu. On the Convergence of Adaptive Gradient Methods for Nonconvex Optimization. Transactions on Machine Learning Research, 2024.

[43] Fangyu Zou, Li Shen, Zequn Jie, Weizhong Zhang, and Wei Liu. A Suficient Condition for Convergences of Adam and RMSProp. arXiv:1811.09358, 2018.

Steffen Dereich<sub>,</sub> Institute for Mathematical Stochastics<sub>,</sub> Faculty of Mathematics and Computer Science<sub>,</sub> University of Munster<sub>,</sub> Germany¨

Email address: steffen.dereich@uni-muenster.de

Arnulf Jentzen<sub>,</sub> School of Data Science and School of Artificial Intelligence<sub>,</sub> The Chinese University of Hong Kong, Shenzhen (CUHK-Shenzhen), China & Applied Mathematics: Institute for Analysis and Numerics, Faculty of Mathematics and Computer Science<sub>,</sub> University of Munster<sub>,</sub> Germany¨

Email address: ajentzen@cuhk.edu.cn; ajentzen@uni-muenster.de