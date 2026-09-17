# Revisiting Distributed Sign-Based Variance Reduction

Wei Jiang<sup>1,2</sup> Zechao Li<sup>1</sup> Lijun Zhang<sup>2,3</sup>

<sup>1</sup>School of Computer Science and Engineering, Nanjing University of Science and Technology, China

<sup>2</sup>National Key Laboratory for Novel Software Technology,Nanjing University, China

<sup>3</sup>School of Artificial Intelligence, Nanjing University, China

## Abstract

Sign-based methods reduce communication costs in distributed environments, but aggregating local signs can introduce bias when data are heterogeneous. As a result, existing sign-based variance reduction methods fail to obtain the optimal convergence rates. In this paper, we solve this problem and obtain optimal rates for both nonconvex stochastic and finite-sum optimization. We first give a counterexample showing that majority voting can fail to approach stationary points even with exact local gradients. Motivated by this limitation, we propose tracking the global gradient at the server through unbiased compression of recursive gradient increments. As a result, we can obtain the convergence rates of $\bar { O } ( \sqrt { d / K } + \sqrt { d } ( a / ( n \bar { K } ) ) ^ { 1 / 3 } )$ for the $\ell _ { 1 } { \mathrm { - n o r m } }$ and $O ( \sqrt { a / K } + \sqrt { a } / ( n K ) ^ { 1 / 3 } )$ for the ℓ<sub>2</sub>-norm. Here, K is the iteration number, n is the number of workers, d is the dimension, and $a = 1 + \omega _ { : }$ with ω denoting the compressor’s relative variance. For finite-sum problems with M components, we combine periodic exact gradient refreshes with compressed component-gradient diferences. The resulting total sample complexities are $O ( M + d \bar { \sqrt { { a } M } } \epsilon ^ { - 2 } )$ and $O ( M \bar { + } a \sqrt { M } \epsilon ^ { - 2 } )$ for $\ell _ { 1 }$ and $\ell _ { 2 }$ gradient norms at most ϵ, matching the corresponding bounds in centralized settings.

## 1 Introduction

In this paper, we focus on smooth nonconvex optimization in the distributed setting:

$$
\operatorname* { m i n } _ { x \in \mathbb { R } ^ { d } } f ( x ) = \frac { 1 } { n } \sum _ { j = 1 } ^ { n } f _ { j } ( x ) , \qquad f _ { j } ( x ) = \mathbb { E } _ { \xi \sim \mathcal { D } _ { j } } F _ { j } ( x ; \xi ) .\tag{1}
$$

In this setting, each worker $j \in \{ 1 , 2 , \cdots , n \}$ accesses its own data distribution $\mathcal { D } _ { j }$ , and these data distributions may difer for diferent nodes. Such heterogeneous objectives arise when data are partitioned arbitrarily across machines. We also consider the finite-sum formulation, in which every worker has a fixed dataset of m components, and the problem can be written in the form:

$$
\operatorname* { m i n } _ { x \in \mathbb { R } ^ { d } } f ( x ) = \frac { 1 } { n } \sum _ { j = 1 } ^ { n } f _ { j } ( x ) , \qquad f _ { j } ( x ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } f _ { j , i } ( x ) .\tag{2}
$$

Here $M = n m$ is the total number of components across the system. Our goal is to find stationary points while reducing the information sent between workers and the parameter server.

Sign-based methods are highly attractive for this purpose since a vector of signs only takes one bit per coordinate (Bernstein et al., 2018, 2019). Later studies show that the momentum technique improves the small-batch convergence guarantees (Safaryan and Richtárik, 2021; Jiang et al., 2025), and variance reduction further improves the quality of the gradient estimate. In particular, the previous centralized variance reduction method SSVR attains an $\ell _ { 1 }$ convergence rate of $O ( \sqrt { d } K ^ { - 1 / 3 } )$ under mean-squared smoothness, and SSVR-FS attains a total gradient complexity of $O ( M + d \sqrt { M } \epsilon ^ { - 2 } )$ for $\ell _ { 1 }$ accuracy ϵ on a centralized dataset of M components (Jiang et al., 2024).

These results motivate extending the benefits of variance reduction to the distributed problem. Such an extension is not automatic. In a majority-vote method, each worker first converts its local estimate into a sign vector, and the server takes the majority of those signs. For diferent local objectives, this nonlinear aggregation may not align with the gradient of their average. SSVR-MV (Jiang et al., 2024) provides two options with diferent server updates. Option 1 uses deterministic majority voting and has an $\ell _ { 1 }$ guarantee $O ( { \sqrt { d / K } } + d / { \sqrt { n } } )$ with an error floor. Option 2 uses randomized worker and server signs and obtains an $\ell _ { 2 }$ rate $O ( d ^ { 1 / 4 } K ^ { - 1 / 4 } )$ . SSVR-MV Option 2 also requires projecting each local estimate onto a Euclidean ball before worker-side sign quantization, and its $O ( K ^ { - 1 / 4 } )$ guarantee leaves a gap to the centralized variance-reduced $O ( K ^ { - 1 / 3 } )$ rate.

In this paper, we obtain improved convergence rates with new algorithms. We assume an unbiased relative-variance compressor that introduces noise proportional to the input magnitude, allowing more general compressors. We then let the server retain a global gradient estimate. Workers send compressed recursive increments, and the server accumulates these messages before choosing a global update direction. Our recursion compresses an increment consisting of a scaled gradient and a same-sample gradient diference, whose second moment is controlled by the parameter choices. Finally, we use deterministic signs for the $\ell _ { 1 }$ criterion and unbiased compression for the $\ell _ { 2 }$ guarantee.

For the finite-sum problem, we periodically compute the exact global gradient and use compressed component-gradient diferences for steps between checkpoints. The same component is evaluated at two successive iterates, so component smoothness controls the update noise even when local gradients are large or heterogeneous. The resulting total component-gradient complexities are $O ( M + d \sqrt { a M } \epsilon ^ { - 2 } )$ for $\ell _ { 1 } { \mathrm { - n o r m } }$ and $O ( M + a \sqrt { M } \epsilon ^ { - 2 } )$ for $\ell _ { 2 } { \mathrm { - n o r m } }$ , where $a = 1 +$ ω and $\omega$ is the compressor’s relative variance. These have the same oracle orders as the centralized SSVR-FS algorithm and Euclidean variance-reduced methods, respectively.

Our contributions. We summarize the results and contributions of this paper below.

• We first analyze the obstruction to local majority voting. We give a three-worker counterexample in Proposition 1, which sufers a nonzero average-gradient floor despite exact local gradients. It isolates the bias of the final vote and motivates tracking the global gradient before taking signs.

• For stochastic problems, we attain $O ( \sqrt { d / K } + \sqrt { d } ( a / ( n K ) ) ^ { 1 / 3 } )$ in the $\ell _ { 1 }$ criterion, removing the previous error floor. Then, we obtain $O ( \sqrt { a / K } + \sqrt { a } / ( n K ) ^ { 1 / 3 } )$ for the $\ell _ { \mathrm { { 2 } ^ { - } } } \mathrm { { n o r m } }$ . This improves the horizon dependence of previous methods.

• We also obtain the finite-sum bounds for the sign-based distributed setting. Exact refreshes and compressed component diferences remove the need for a bounded gradient assumption. We obtain total oracle complexities $O ( M + d \sqrt { a M } \epsilon ^ { - 2 } )$ and $O ( M + a \sqrt { M } \epsilon ^ { - 2 } )$ for the $\ell _ { 1 }$ and $\ell _ { 2 }$ criteria, matching the corresponding centralized rates.

## 2 Assumptions

We now specify the assumptions used. Write $g _ { j } ( x ; \boldsymbol { \xi } ) = \nabla F _ { j } ( x ; \boldsymbol { \xi } )$ for a stochastic gradient and $\Delta _ { f } = f ( x _ { 1 } ) - f _ { * }$ for the initial objective gap, where $f _ { * }$ is a lower bound of the objective $f .$

We first give the following assumption on the objective function for the stochastic problem (1).

Assumption 1. The objective satisfies $f \geq f _ { * } > - \infty$ . For every worker and all $x , y ,$ we have

$$
\begin{array} { r } { \mathbb { E } _ { \xi } g _ { j } ( x ; \xi ) = \nabla f _ { j } ( x ) , } \end{array}\tag{3}
$$

$$
\begin{array} { r } { \mathbb { E } _ { \xi } \| g _ { j } ( x ; \xi ) \| _ { 2 } ^ { 2 } \le H ^ { 2 } , } \end{array}\tag{4}
$$

$$
\begin{array} { r } { \mathbb { E } _ { \xi } \| g _ { j } ( x ; \xi ) - g _ { j } ( y ; \xi ) \| _ { 2 } ^ { 2 } \leq L ^ { 2 } \| x - y \| _ { 2 } ^ { 2 } . } \end{array}\tag{5}
$$

Remark: Here, Jensen’s inequality implies that $f _ { j }$ and $f$ are also L-smooth. Besides, inequality (4) can be replaced with bounded oracle variance $\sigma ^ { 2 }$ and a true-gradient bound $\| \nabla f _ { j } ( x ) \| _ { 2 } \leq G _ { 2 }$ which imply inequality (4) with $H ^ { 2 } = \sigma ^ { 2 } + G _ { 2 } ^ { 2 }$

Next, we give the general compression assumption below.

Assumption 2 (Relative-variance compressor). For every input $v ,$ the output $Q ( v )$ satisfies

$$
\begin{array} { r } { \mathbb { E } [ Q ( v ) \mid v ] = v , \qquad \mathbb { E } [ \| Q ( v ) - v \| _ { 2 } ^ { 2 } \mid v ] \le \omega \| v \| _ { 2 } ^ { 2 } . } \end{array}\tag{6}
$$

Note that unbiasedness also gives

$$
\mathbb { E } [ \| Q ( v ) \| _ { 2 } ^ { 2 } \mid v ] \leq ( 1 + \omega ) \| v \| _ { 2 } ^ { 2 } .\tag{7}
$$

Remark: The general compression assumption includes the usual scaled stochastic sign. For $v \neq 0$ let $\rho ( v ) = \| v \| _ { \infty }$ and draw independent signs with $\operatorname* { P r } ( S _ { k } = 1 \mid v ) = ( 1 + { v _ { k } } / { \rho ( v ) } ) / 2$ . Then

$$
Q ( v ) = \rho ( v ) S , \qquad Q ( 0 ) = 0 ,\tag{8}
$$

is unbiased and satisfies $1 + \omega = d$ since $\begin{array} { r } { \mathbb { E } [ \| Q ( v ) - v \| _ { 2 } ^ { 2 } \mid v ] = d \| v \| _ { \infty } ^ { 2 } - \| v \| _ { 2 } ^ { 2 } \leq ( d - 1 ) \| v \| _ { 2 } ^ { 2 } . } \end{array}$

Finally, we list assumptions for the finite-sum problem (2).

Assumption 3. The average objective is lower bounded and each component satisfies

$$
\begin{array} { r } { \| \nabla f _ { j , i } ( x ) - \nabla f _ { j , i } ( y ) \| _ { 2 } \leq L \| x - y \| _ { 2 } \quad f o r \ a l l \ x , y , j , i . } \end{array}\tag{9}
$$

The finite-sum results use Assumptions $2$ and 3, requiring no bound on gradients, oracle variance, or gradient heterogeneity.

## 3 The proposed method

We begin with the earlier SSVR-MV method and explain the bias created by its majority vote. This motivates a global gradient estimate at the server, followed by either a sign update for the $\ell _ { 1 }$ criterion or an unbiased compressed update for the $\ell _ { 2 }$ criterion.

## 3.1 SSVR-MV and the limitation of its majority voting

The earlier method. SSVR-MV, introduced by Jiang et al. (2024), combines a local variancereduced estimator with bidirectional sign communication. Specifically, each worker $j$ first initializes $v _ { 1 } ^ { j } = g _ { j } ( x _ { 1 } ; \xi _ { 1 } ^ { j } )$ and, for $t \geq 2$ , computes

$$
v _ { t } ^ { j } = g _ { j } ( x _ { t } ; \xi _ { t } ^ { j } ) + ( 1 - \beta ) [ v _ { t - 1 } ^ { j } - g _ { j } ( x _ { t - 1 } ; \xi _ { t } ^ { j } ) ] .\tag{10}
$$

The two stochastic gradients in this update use the same sample. For a fixed radius $R > 0$ and an input satisfying $\| v \| _ { \infty } \leq R$ , define the random sign vector $S _ { R } ( v )$ coordinatewise by

$$
\operatorname* { P r } ( S _ { R } ( v ) _ { k } = + 1 \mid v ) = { \frac { 1 + v _ { k } / R } { 2 } } , \qquad \operatorname* { P r } ( S _ { R } ( v ) _ { k } = - 1 \mid v ) = { \frac { 1 - v _ { k } / R } { 2 } } .\tag{11}
$$

Thus the signs estimate the scaled input without any bias. In SSVR-MV (Option 1), worker $j$ sends $q _ { t } ^ { j } = S _ { R } ( v _ { t } ^ { j } )$ , and the server broadcasts their deterministic majority vote:

$$
s _ { t } = \mathrm { S i g n } \left( \frac { 1 } { n } \sum _ { j = 1 } ^ { n } q _ { t } ^ { j } \right) , \qquad x _ { t + 1 } = x _ { t } - \eta s _ { t } .\tag{12}
$$

The setting of their Theorem 3 uses $\| g _ { j } ( x ; \xi ) \| _ { \infty } \le G , \beta = 1 / 2 , \eta = \mathcal { O } ( ( d K ) ^ { - 1 / 2 } )$ , and $R = 4 G$ ， keeping the inputs inside the radius. The server forms a new vote at every iteration.

Where the bias enters. The worker signs are unbiased for their scaled inputs, but the majority operation is nonlinear. This can be seen exactly with three workers. For any $q _ { 1 } , q _ { 2 } , q _ { 3 } \in \{ - 1 , 1 \}$ ，

$$
\mathrm { S i g n } ( q _ { 1 } + q _ { 2 } + q _ { 3 } ) = { \frac { q _ { 1 } + q _ { 2 } + q _ { 3 } - q _ { 1 } q _ { 2 } q _ { 3 } } { 2 } } .\tag{13}
$$

Writing $m _ { j } = \mathbb { E } q _ { j }$ , with all expectations conditional on these inputs, we obtain

$$
\begin{array} { r l } { \mathbb { E } \operatorname { S i g n } ( q _ { 1 } + q _ { 2 } + q _ { 3 } ) = \displaystyle \frac { 1 } { 2 } \left( \sum _ { j = 1 } ^ { 3 } \mathbb { E } q _ { j } - \mathbb { E } [ q _ { 1 } q _ { 2 } q _ { 3 } ] \right) } & { } \\ { \displaystyle } & { = \displaystyle \frac { 1 } { 2 } ( m _ { 1 } + m _ { 2 } + m _ { 3 } - m _ { 1 } m _ { 2 } m _ { 3 } ) . } \end{array}\tag{14}
$$

$\mathrm { I f ~ } ( m _ { 1 } , m _ { 2 } , m _ { 3 } ) = ( u , u , - 2 u )$ with $0 < u \le 1 / 2$ , the average sign mean is zero, whereas the expected majority vote is $u ^ { 3 } > 0$ . Unbiased worker messages therefore can produce a biased vote.

Proposition 1 (A gradient floor with exact local estimates). There exist three lower-bounded smooth one-dimensional local objectives with gradients bounded by $G = 1$ such that SSVR-MV Option 1, initialized at a global minimizer and using exact gradients, satisfies

$$
\operatorname* { l i m } _ { K \to \infty } \operatorname* { i n f } _ { \mathbb { E } | f ^ { \prime } ( x _ { \tau } ) | \geq } \frac { 1 } { 1 5 4 2 } > 0\tag{15}
$$

for $R = 4$ and step size $\eta = \Theta ( K ^ { - 1 / 2 } )$ , where $\tau$ is independent and uniform on $\{ 1 , \ldots , K \}$

Construction and intuition. Take

$$
f _ { 1 } ( x ) = f _ { 2 } ( x ) = { \textstyle { \frac { 1 } { 2 } } } \log \cosh x + { \textstyle { \frac { 1 } { 4 } } } x , \qquad f _ { 3 } ( x ) = { \textstyle { \frac { 1 } { 2 } } } \log \cosh x - { \textstyle { \frac { 1 } { 2 } } } x .
$$

Their average is $\begin{array} { r } { f ( x ) = \frac { 1 } { 2 } } \end{array}$ log cosh $x ,$ minimized at $x = 0$ . At this point the worker sign means are $( u , u , - 2 u )$ with $u = 1 / 1 6 .$ so (14) gives a nonzero expected vote despite $f ^ { \prime } ( 0 ) = 0$ . Appendix A proves that this bias produces the stated gradient floor. □

## 3.2 Tracking the global gradient through compressed increments

We retain the variance-reduction idea in (10), but change what workers transmit and what the server remembers. For $t \geq 2$ , worker $j$ forms the increment

$$
\begin{array} { r l } & { h _ { t } ^ { j } = g _ { j } ( x _ { t } ; \xi _ { t } ^ { j } ) - ( 1 - \beta ) g _ { j } ( x _ { t - 1 } ; \xi _ { t } ^ { j } ) } \\ & { \quad = \beta g _ { j } ( x _ { t } ; \xi _ { t } ^ { j } ) + ( 1 - \beta ) [ g _ { j } ( x _ { t } ; \xi _ { t } ^ { j } ) - g _ { j } ( x _ { t - 1 } ; \xi _ { t } ^ { j } ) ] . } \end{array}\tag{16}
$$

Algorithm 1 DVR-Sign / DVR-Q   
Require: $\overline { { x _ { 1 } , K , B _ { 0 } , \eta , \beta } } ,$ compressor ${ \overline { { Q . } } }$   
1: Each worker draws $B _ { 0 }$ samples at $x _ { 1 }$   
2: Server initializes $\begin{array} { r } { z _ { 1 } = ( n B _ { 0 } ) ^ { - 1 } \sum _ { j , b } Q ( g _ { j } ( x _ { 1 } ; \xi _ { 1 , b } ^ { j } ) ) } \end{array}$   
3: for $t = 1 , \ldots , K$ do   
4: if $t \geq 2$ then   
5: Each worker draws a sample $\xi _ { t } ^ { j }$ , forms (16), and sends $Q ( h _ { t } ^ { j } )$   
6: Server updates $z _ { t }$ by (17).   
7: end if   
8: if variant is DVR-Sign then   
9: Server broadcasts $s _ { t } = \mathrm { S i g n } ( z _ { t } )$ to all workers.   
10: else   
11: Server broadcasts $s _ { t } = Q ( z _ { t } )$ to all workers.   
12: end if   
13: All workers and the server set $\boldsymbol { x } _ { t + 1 } = \boldsymbol { x } _ { t } - \eta \boldsymbol { s } _ { t } .$   
14: end for   
15: Draw an independent uniform $\tau \in \{ 1 , \ldots , K \}$ and return $x _ { \tau } .$

The worker sends an encoding of $Q ( h _ { t } ^ { j } )$ and the server then accumulates the increments:

$$
z _ { t } = ( 1 - \beta ) z _ { t - 1 } + \frac { 1 } { n } \sum _ { j = 1 } ^ { n } Q ( h _ { t } ^ { j } ) .\tag{17}
$$

The diference from (12) is substantive: the server retains $z _ { t - 1 }$ and corrects it with numerical increments before choosing its update direction. In particular, it does not take a majority vote of independently randomized local signs.

The increment decomposition is central to the analysis. The fresh-gradient term is multiplied by $\beta ,$ while mean-squared smoothness controls the same-sample diference. These two terms give

$$
\begin{array} { r } { \mathbb { E } _ { t } \| h _ { t } ^ { j } \| _ { 2 } ^ { 2 } \leq 2 \beta ^ { 2 } H ^ { 2 } + 2 L ^ { 2 } \| x _ { t } - x _ { t - 1 } \| _ { 2 } ^ { 2 } . } \end{array}\tag{18}
$$

Here $\mathbb { E } _ { t }$ conditions on the history before the current worker samples and compression calls. Unlike repeatedly compressing a full local estimate, compressing this increment introduces noise that decreases with smaller $\beta$ and smaller model movements. The two methods below control this movement diferently. At initialization, every worker compresses $B _ { 0 }$ independent sample gradients separately. The whole algorithm is specified in Algorithm 1.

## 3.3 The compressed update for the server

For $\ell _ { 1 }$ measure For the $\ell _ { 1 }$ criterion, the natural direction is $s _ { t } = \mathrm { S i g n } ( z _ { t } )$ . With an exact estimate $z _ { t } = \nabla f ( x _ { t } )$ , its inner product with the gradient equals $\| \nabla f ( x _ { t } ) \| _ { 1 }$ . With an approximate estimate, the corresponding inequality is

$$
- \langle g , \mathrm { S i g n } ( z ) \rangle \leq - \| g \| _ { 1 } + 2 \| z - g \| _ { 1 } \leq - \| g \| _ { 1 } + 2 { \sqrt { d } } \| z - g \| _ { 2 } .\tag{19}
$$

Thus the deterministic sign update converts a bound on the server’s tracking error into an $\ell _ { 1 }$ stationarity guarantee. We call this algorithm $D V R  – S i g n \colon$ distributed variance reduction with sign updates.

For $\ell _ { 2 }$ measure An $\ell _ { 1 } { \mathrm { - n o r m } }$ guarantee already implies an $\ell _ { 2 } \cdot$ -norm guarantee through $\| g \| _ { 2 } \leq \| g \| _ { 1 }$ However, this transfer retains the $\sqrt { d }$ tracking-error coeficient in (19). We therefore use the unbiased compressor already defined in Assumption 2 for the server broadcast as well:

$$
s _ { t } = Q ( z _ { t } ) , \qquad x _ { t + 1 } = x _ { t } - \eta s _ { t } .\tag{20}
$$

We call this method $D V R – Q .$ . Conditional on the history $\mathcal { G } _ { t }$ containing $z _ { t }$ and all current worker messages, fresh server randomness gives

$$
\begin{array} { r } { \mathbb { E } [ s _ { t } \mid \mathcal { G } _ { t } ] = z _ { t } , \qquad \mathbb { E } [ \| s _ { t } \| _ { 2 } ^ { 2 } \mid \mathcal { G } _ { t } ] \leq ( 1 + \omega ) \| z _ { t } \| _ { 2 } ^ { 2 } . } \end{array}\tag{21}
$$

With an exact estimate, the expected gradient inner product is $\| \nabla f ( x _ { t } ) \| _ { 2 } ^ { 2 }$ . For an approximate estimate, the elementary identity

$$
- 2 \langle g , z \rangle = - \| g \| _ { 2 } ^ { 2 } - \| z \| _ { 2 } ^ { 2 } + \| z - g \| _ { 2 } ^ { 2 }\tag{22}
$$

connects descent to squared Euclidean tracking error. Its negative $\| \boldsymbol { z } \| _ { 2 } ^ { 2 }$ term is essential: it absorbs the additional tracking error caused by the random step length in (21). The two methods thus use diferent directions for diferent criteria. Deterministic signs give the $\ell _ { 1 }$ inner product in (19), while unbiased Q updates give squared $\ell _ { 2 }$ descent without its explicit $\sqrt { d }$ conversion.

## 4 Stochastic guarantees

We first analyze DVR-Sign under the $\ell _ { 1 }$ stationarity criterion and then analyze DVR-Q directly in the squared Euclidean norm. The two methods share the server recursion, but their model movements and descent arguments difer. Write $a = 1 + \omega$ and $c = a / n$ . Denote $e _ { t } = z _ { t } - \nabla f ( x _ { t } )$ $E _ { t } = \mathbb { E } \Vert e _ { t } \Vert _ { 2 } ^ { 2 }$ , and $\begin{array} { r } { \overline { { E } } _ { K } = K ^ { - 1 } \sum _ { t = 1 } ^ { K } E _ { t } } \end{array}$

## 4.1 The $\ell _ { 1 }$ guarantee for DVR-Sign

Theorem 2. Under Assumptions 1 and ${ \mathcal { Q } } ,$ for any $\eta > 0 , \beta \in ( 0 , 1 ]$ , and integers K, $B _ { 0 } \geq 1$ , the DVR-Sign variant of Algorithm 1 satisfies

$$
E _ { 1 } \le c H ^ { 2 } / B _ { 0 } , \qquad E _ { t } \le ( 1 - \beta ) E _ { t - 1 } + 2 c ( \beta ^ { 2 } H ^ { 2 } + L ^ { 2 } \eta ^ { 2 } d ) \quad ( t \ge 2 ) ,\tag{23}
$$

$$
\overline { { E } } _ { K } \leq \frac { c H ^ { 2 } } { B _ { 0 } \beta K } + 2 c \left( H ^ { 2 } \beta + \frac { L ^ { 2 } \eta ^ { 2 } d } { \beta } \right) ,\tag{24}
$$

$$
\mathbb { E } \| \nabla f ( x _ { \tau } ) \| _ { 1 } \leq \frac { \Delta _ { f } } { \eta K } + \frac { L \eta d } { 2 } + 2 \sqrt { d \overline { { E } } _ { K } } .\tag{25}
$$

Balancing these efects gives the following result.

Corollary 3. Under the assumptions of Theorem ${ \mathcal { Q } } ,$ choose

$$
u _ { 1 } = \frac { 1 } { \sqrt { K } + c ^ { 1 / 3 } K ^ { 2 / 3 } } , \qquad \beta = u _ { 1 } , \quad \eta = \frac { H u _ { 1 } } { L \sqrt { d } } , \quad B _ { 0 } = \left\lceil \frac { 1 } { u _ { 1 } ^ { 2 } K } \right\rceil .\tag{26}
$$

Then DVR-Sign satisfies

$$
\begin{array} { r l } & { \mathbb { E } \| \nabla f ( x _ { \tau } ) \| _ { 1 } \leq \sqrt { d } \left[ \left( \frac { L \Delta _ { f } } { H } + \frac { H } { 2 } \right) K ^ { - 1 / 2 } + \left( \frac { L \Delta _ { f } } { H } + 2 \sqrt { 5 } H \right) \left( \frac { c } { K } \right) ^ { 1 / 3 } \right] } \\ & { \quad \quad = O \left( \sqrt { \displaystyle \frac { d } { K } } + \sqrt { d } \left( \frac { 1 + \omega } { n K } \right) ^ { 1 / 3 } \right) . } \end{array}\tag{27}
$$

Remark: To achieve $\mathbb { E } \| \nabla f ( x _ { \tau } ) \| _ { 1 } \leq \epsilon$ , the required number of model updates is

$$
K = O \left( 1 + \frac { d } { \epsilon ^ { 2 } } + \frac { a d ^ { 3 / 2 } } { n \epsilon ^ { 3 } } \right) .
$$

## 4.2 The $\ell _ { 2 }$ guarantee for DVR-Q

The unbiased compressor $Q ( z _ { t } )$ has conditional mean $z _ { t }$ , and its conditional second moment is at most $( 1 + \omega ) \| z _ { t } \| _ { 2 } ^ { 2 }$ . The descent analysis therefore controls the squared gradient norm and retains a negative tracker-norm term. This term absorbs the tracking error caused by moving the model, which is the key diference from the sign analysis.

Theorem 4. Under Assumptions 1 and ${ \mathcal { Q } } ,$ for any $\eta > 0 , \beta \in ( 0 , 1 ]$ , and integers $K , B _ { 0 } \geq 1$ , the DVR-Q variant of Algorithm 1 satisfies

$$
\begin{array} { r l } & { \displaystyle \mathbb { E } \| \nabla f ( x _ { \tau } ) \| _ { 2 } ^ { 2 } + \left( 1 - L a \eta - \frac { 2 c a L ^ { 2 } \eta ^ { 2 } } { \beta } \right) \frac { 1 } { K } \sum _ { t = 1 } ^ { K } \mathbb { E } \| z _ { t } \| _ { 2 } ^ { 2 } } \\ & { \quad \quad \leq \displaystyle \frac { 2 \Delta _ { f } } { \eta K } + \frac { c H ^ { 2 } } { B _ { 0 } \beta K } + 2 c \beta H ^ { 2 } . } \end{array}\tag{28}
$$

In particular, if $L a \eta \leq 1 / 2$ and $\beta \geq 4 c a L ^ { 2 } \eta ^ { 2 }$ , the tracker-norm term on the left is nonnegative and can be dropped.

Corollary 5. Under the assumptions of Theorem $^ { 4 , }$ choose

$$
r = \left( \frac { K } { n ^ { 2 } } \right) ^ { 1 / 3 } , \qquad \eta = \frac { 1 } { 2 L a ( 1 + r ) } , \quad \beta = \frac { 1 } { n ( 1 + r ) ^ { 2 } } , \quad B _ { 0 } = \lceil 1 + r \rceil .\tag{29}
$$

Then DVR-Q satisfies

$$
\begin{array} { l } { \displaystyle \mathbb { E } \| \nabla f ( x _ { \tau } ) \| _ { 2 } \leq \sqrt { 4 L \Delta _ { f } + H ^ { 2 } } \sqrt { \frac { a } { K } } + \sqrt { 4 L \Delta _ { f } + 3 H ^ { 2 } } \frac { \sqrt { a } } { ( n K ) ^ { 1 / 3 } } } \\ { \displaystyle = O \left( \sqrt { \frac { 1 + \omega } { K } } + \frac { \sqrt { 1 + \omega } } { ( n K ) ^ { 1 / 3 } } \right) . } \end{array}\tag{30}
$$

Remark: To achieve $\mathbb { E } \| \nabla f ( x _ { \tau } ) \| _ { 2 } \leq \epsilon$ , the required number of model updates is

$$
K = O \left( 1 + \frac { a } { \epsilon ^ { 2 } } + \frac { a ^ { 3 / 2 } } { n \epsilon ^ { 3 } } \right) .\tag{31}
$$

## 5 The proposed methods for finite-sum problems

For the finite-sum problem (2), an occasional full pass through the data can replace the stochastic correction used by DVR-Sign and DVR-Q. We compute exact gradients at checkpoints, followed by compressed component-gradient diferences. We use a deterministic sign for the $\ell _ { 1 }$ criterion and employ the unbiased compressor Q for the $\ell _ { 2 }$ criterion.

Fix an integer refresh period $q = m$ . At times $t = 1 + k q$ , every worker computes and sends its local full gradient, and the server resets $z _ { t } = \nabla f ( x _ { t } )$ . At other times, each worker independently draws a uniform component $i _ { t } ^ { j } \in \{ 1 , \dots , m \}$ and sends a compressed paired diference, using the same component at both iterates. The server uses

Algorithm 2 DVR-Sign-FS and DVR-Q-FS   
Require: $x _ { 1 } , K , \eta ,$ refresh period q, and compressor $Q .$   
1: for $t = 1 , \ldots , K$ do   
2: if $t = 1 +$ kq for an integer $k \geq 0$ then   
3: Each worker computes and sends $\begin{array} { r } { \nabla f _ { j } ( x _ { t } ) = m ^ { - 1 } \sum _ { i = 1 } ^ { m } \nabla f _ { j , i } ( x _ { t } ) } \end{array}$   
4: Server sets $\begin{array} { r } { z _ { t } = n ^ { - 1 } \sum _ { j = 1 } ^ { n } \nabla f _ { j } ( x _ { t } ) } \end{array}$   
5: else   
6: Every worker j draws an independent uniform $i _ { t } ^ { j } \in \{ 1 , \dots , m \}$   
7: Every worker forms y<sup>j</sup> in (32) and sends $Q ( y _ { t } ^ { j } )$   
8: Server sets $\begin{array} { r } { z _ { t } = z _ { t - 1 } + n ^ { - 1 } \sum _ { j = 1 } ^ { n } Q ( y _ { t } ^ { j } ) } \end{array}$   
9: end if   
10: if DVR-Sign-FS then   
11: Server broadcasts $s _ { t } = \mathrm { S i g n } ( z _ { t } )$   
12: else   
13: Server broadcasts $s _ { t } = Q ( z _ { t } )$   
14: end if   
15: Set $x _ { t + 1 } = x _ { t } - \eta s _ { t } .$   
16: end for   
17: Draw an independent uniform $\tau \in \{ 1 , \ldots , K \}$ and return $x _ { \tau } .$

$$
y _ { t } ^ { j } = \nabla f _ { j , i _ { t } ^ { j } } ( x _ { t } ) - \nabla f _ { j , i _ { t } ^ { j } } ( x _ { t - 1 } ) , \qquad z _ { t } = z _ { t - 1 } + { \frac { 1 } { n } } \sum _ { j = 1 } ^ { n } Q ( y _ { t } ^ { j } ) .\tag{32}
$$

The server retains $z _ { t }$ itself for the next estimator update. Component smoothness gives $\| y _ { t } ^ { j } \| _ { 2 } \leq$ $L \| x _ { t } - x _ { t - 1 } \| _ { 2 }$ , which controls the increment without requiring bounded gradients or bounded heterogeneity. The whole method is described in Algorithm 2.

## 5.1 Deterministic sign updates for the $\ell _ { 1 }$ guarantee

The first version uses $s _ { t } = \mathrm { S i g n } ( z _ { t } )$ . A refresh eliminates the estimation error and only the $q - 1$ intervening updates can contribute to its variance. A refresh costs M component-gradient evaluations, whereas an ordinary update costs $2 n$ . Choosing $q = m$ makes the amortized refresh cost $M / q = n$ of the same order as the ordinary update cost.

## Theorem 6. Under Assumptions 2 and 3, let $a = 1 + \omega$ . For DVR-Sign-FS, define

$$
J _ { 1 } = \frac { d } { 2 } + 2 d \sqrt { \frac { a ( q - 1 ) } { n } } .\tag{33}
$$

For an independent uniform $\tau \in \{ 1 , \ldots , K \}$ 2

$$
\mathbb { E } \| z _ { t } - \nabla f ( x _ { t } ) \| _ { 2 } ^ { 2 } \leq \frac { a L ^ { 2 } \eta ^ { 2 } d ( q - 1 ) } { n } ,\tag{34}
$$

$$
\mathbb { E } \| \nabla f ( x _ { \tau } ) \| _ { 1 } \leq \frac { \Delta _ { 0 } } { \eta K } + L \eta J _ { 1 } .\tag{35}
$$

Corollary 7. Under the assumptions of Theorem $\delta ,$ set $q = m$ . We ensure $\mathbb { E } \| \nabla f ( x _ { \tau } ) \| _ { 1 } \leq \epsilon .$ , with

$$
\eta = \frac { \epsilon } { 2 L J _ { 1 } } , \quad K = O \left( 1 + \frac { L \Delta _ { 0 } d } { \epsilon ^ { 2 } } \left[ 1 + \sqrt { \frac { a ( m - 1 ) } { n } } \right] \right) .\tag{36}
$$

$$
N _ { \mathrm { g r a d } } = O \biggl ( M + \frac { L \Delta _ { 0 } d } { \epsilon ^ { 2 } } [ n + \sqrt { a M } ] \biggr ) .\tag{37}
$$

Remark: For $n \leq O ( a m )$ , the total oracle order is $\begin{array} { r } { N _ { \mathrm { g r a d } } = O \Big ( M + \frac { d \sqrt { a M } } { \epsilon ^ { 2 } } \Big ) } \end{array}$

## 5.2 Unbiased compressed updates for the $\ell _ { 2 }$ guarantee

The second version keeps the same refreshes and estimator, but uses $s _ { t } = Q ( z _ { t } )$ , where $Q$ satisfies Assumption 2 with $a = 1 + \omega$ . Unlike a sign step of fixed length, this update has conditional second moment at most $a \eta ^ { 2 } \| z _ { t } \| _ { 2 } ^ { 2 }$ . Unbiasedness also makes the expected descent depend on $\langle \nabla f ( x _ { t } ) , z _ { t } \rangle$ These two facts allow the tracking error to be absorbed into the descent inequality and yield a squared Euclidean stationarity guarantee.

Theorem 8. Under Assumptions 2 and 3, for $a = 1 + \omega$ , define

$$
J _ { Q } = a \left( 1 + \sqrt { \frac { q - 1 } { n } } \right) , \qquad \eta = \frac { 1 } { 2 L J _ { Q } } .\tag{38}
$$

Then an independent uniform $\tau \in \{ 1 , \ldots , K \}$ satisfies

$$
\mathbb { E } \| \nabla f ( x _ { \tau } ) \| _ { 2 } ^ { 2 } \leq \frac { 4 L \Delta _ { 0 } J _ { Q } } { K } , \qquad \mathbb { E } \| \nabla f ( x _ { \tau } ) \| _ { 2 } \leq 2 \sqrt { \frac { L \Delta _ { 0 } J _ { Q } } { K } } .\tag{39}
$$

Corollary 9. Under the assumptions of Theorem $\delta ,$ set $q = m$ . We ensure $\mathbb { E } \| \nabla f ( x _ { \tau } ) \| _ { 2 } \leq \epsilon .$ , with

$$
K = O \left( 1 + \frac { a L \Delta _ { 0 } } { \epsilon ^ { 2 } } \left[ 1 + \sqrt { \frac { m - 1 } { n } } \right] \right) ,\tag{40}
$$

$$
N _ { \mathrm { g r a d } } = O \bigg ( M + \frac { a L \Delta _ { 0 } } { \epsilon ^ { 2 } } [ n + \sqrt { M } ] \bigg ) .\tag{41}
$$

Remark: For $n \leq O ( { \sqrt { m } } )$ , this total oracle order $\begin{array} { r } { N _ { \mathrm { g r a d } } = O \bigl ( M + \frac { a \sqrt { M } } { \epsilon ^ { 2 } } \bigr ) } \end{array}$ matches the centralized Euclidean benchmark of SPIDER and PAGE (Fang et al., 2018; Li et al., 2021).

## References

Jeremy Bernstein, Yu-Xiang Wang, Kamyar Azizzadenesheli, and Animashree Anandkumar. signSGD: Compressed optimisation for non-convex problems. In Proceedings of the 35th International Conference on Machine Learning, pages 560–569, 2018.

Jeremy Bernstein, Jiawei Zhao, Kamyar Azizzadenesheli, and Anima Anandkumar. signSGD with majority vote is communication eficient and fault tolerant. In International Conference on Learning Representations, 2019.

Cong Fang, Chris Junchi Li, Zhouchen Lin, and Tong Zhang. SPIDER: Near-optimal non-convex optimization via stochastic path-integrated diferential estimator. In Advances in Neural Information Processing Systems, 2018.

Wei Jiang, Sifan Yang, Wenhao Yang, and Lijun Zhang. Eficient sign-based optimization: Accelerating convergence via variance reduction. In Advances in Neural Information Processing Systems, 2024.

Wei Jiang, Dingzhi Yu, Sifan Yang, Wenhao Yang, and Lijun Zhang. Improved analysis for sign-based methods with momentum updates. arXiv preprint arXiv:2507.12091, 2025.

Zhize Li, Hongyan Bao, Xiangliang Zhang, and Peter Richtarik. PAGE: A simple and optimal probabilistic gradient estimator for nonconvex optimization. In Proceedings of the 38th International Conference on Machine Learning, pages 6286–6295, 2021.

Mher Safaryan and Peter Richtárik. Stochastic sign descent methods: New algorithms and better theory. In Proceedings of the 38th International Conference on Machine Learning, pages 9224–9234, 2021.

## A Proof of Proposition 1

We first prove Proposition 1 for the SSVR-MV Option 1. Use the three one-dimensional functions

$$
\begin{array} { r } { f _ { 1 } ( x ) = f _ { 2 } ( x ) = \frac 1 2 \log \cosh x + \frac 1 4 x , \qquad f _ { 3 } ( x ) = \frac 1 2 \log \cosh x - \frac 1 2 x . } \end{array}\tag{A.1}
$$

Since cosh x $\geq e ^ { | x | } / 2$ , we have log cosh x $\geq | x | ~ .$ − log 2. It follows that

$$
\begin{array} { r } { f _ { 1 } ( x ) = f _ { 2 } ( x ) \geq \frac { 1 } { 2 } | x | + \frac { 1 } { 4 } x - \frac { 1 } { 2 } \log 2 \geq - \frac { 1 } { 2 } \log 2 , \quad f _ { 3 } ( x ) \geq \frac { 1 } { 2 } ( | x | - x ) - \frac { 1 } { 2 } \log 2 \geq - \frac { 1 } { 2 } \log 2 . } \end{array}
$$

Thus each local objective is lower bounded. Their derivatives are

$$
\begin{array} { r } { f _ { 1 } ^ { \prime } ( x ) = f _ { 2 } ^ { \prime } ( x ) = \frac { 1 } { 2 } \operatorname { t a n h } x + \frac { 1 } { 4 } , \qquad f _ { 3 } ^ { \prime } ( x ) = \frac { 1 } { 2 } \operatorname { t a n h } x - \frac { 1 } { 2 } . } \end{array}
$$

Because | tanh $x | \le 1$ , all three derivatives have absolute value at most $G = 1$ . Moreover, $f _ { j } ^ { \prime \prime } ( x ) =$ $1 / ( 2 \cosh ^ { 2 } x ) \in ( 0 , 1 / 2 ]$ , so local objectives are $1 / 2 \cdot$ -smooth. Their average and its derivative are

$$
f ( x ) = { \textstyle \frac { 1 } { 2 } } \log \cosh x , \qquad h ( x ) : = f ^ { \prime } ( x ) = { \textstyle \frac { 1 } { 2 } } \operatorname { t a n h } x .
$$

In particular, we observe that $x _ { 1 } = 0$ is a global minimizer and $| h ( x ) | \leq 1 / 2$ . Take the deterministic oracle $g _ { j } ( x ; \xi ) = f _ { j } ^ { \prime } ( x )$ whose variance is zero.

Exact initialization gives $v _ { 1 } ^ { j } = f _ { j } ^ { \prime } ( x _ { 1 } )$ . If $v _ { t - 1 } ^ { j } = f _ { j } ^ { \prime } ( x _ { t - 1 } )$ , the local recursion gives

$$
v _ { t } ^ { j } = f _ { j } ^ { \prime } ( x _ { t } ) + ( 1 - \beta ) [ f _ { j } ^ { \prime } ( x _ { t - 1 } ) - f _ { j } ^ { \prime } ( x _ { t - 1 } ) ] = f _ { j } ^ { \prime } ( x _ { t } ) .
$$

Induction therefore proves exact local estimates at every iteration for every $\beta \in ( 0 , 1 ]$ . The signs $q _ { t } ^ { j } = S _ { 4 } ( f _ { j } ^ { \prime } ( x _ { t } ) )$ are well defined because $| f _ { j } ^ { \prime } ( x _ { t } ) | \leq 1 < 4$

Next, to keep the following scalar calculation short, write $u = 1 / 1 6$ and $z = h ( x ) / 4$ . The three sign means at $x _ { t } = x$ are $z + u , z + u , z - 2 u$ . Expanding the product,

$$
( z + u ) ^ { 2 } ( z - 2 u ) = ( z ^ { 2 } + 2 u z + u ^ { 2 } ) ( z - 2 u ) = z ^ { 3 } - 3 u ^ { 2 } z - 2 u ^ { 3 } .
$$

Consequently the expected vote, denoted by $M ( x )$ , is

$$
M ( x ) : = \mathbb { E } [ s _ { t } \mid x _ { t } = x ] = { \frac { 3 z - ( z + u ) ^ { 2 } ( z - 2 u ) } { 2 } } = { \frac { 3 ( 1 + u ^ { 2 } ) z - z ^ { 3 } + 2 u ^ { 3 } } { 2 } } .\tag{A.2}
$$

At the minimizer, $h ( 0 ) = z = 0$ and $\boldsymbol { M } ( 0 ) = \boldsymbol { u } ^ { 3 } > 0$ . The following derivative bound controls how much this bias can change when the true gradient is small:

$$
\frac { d M } { d h } = \frac { 3 } { 8 } ( 1 + u ^ { 2 } - z ^ { 2 } ) , \qquad 0 < \frac { d M } { d h } \le \frac { 3 } { 8 } ( 1 + u ^ { 2 } ) = \frac { 7 7 1 } { 2 0 4 8 } .\tag{A.3}
$$

Indeed $| z | = | h | / 4 \leq 1 / 8$ , so $1 + u ^ { 2 } - z ^ { 2 } > 0$ . The mean value theorem now gives

$$
| M ( x ) - u ^ { 3 } | \leq { \frac { 7 7 1 } { 2 0 4 8 } } | h ( x ) | .\tag{A.4}
$$

Next, we control the iterate without assuming it is bounded. Since tanh $1 > 1 / 2 , x \ge 1$ implies that $z \geq u ,$ , and $x \le - 1$ implies $z \le - u$ . The polynomial in (A.2) is increasing in z on $[ - 1 / 8 , 1 / 8 ]$ Its values at u and −u are, respectively, $( 3 u + 4 u ^ { 3 } ) / 2 > 0$ and $- 3 u / 2 < 0$ . Thus $x M ( x ) \geq 0$ for $| x | \geq 1$ . For $| x | < 1$ , the fact that $M ( x )$ is the expectation of a sign gives $| M ( x ) | \leq 1$ and hence $x M ( x ) \geq - 1$ . Combining the two regions yields $x M ( x ) \geq - 1$ for every x.

The update $x _ { t + 1 } = x _ { t } - \eta s _ { t }$ and $s _ { t } ^ { 2 } = 1$ give

$$
\mathbb { E } [ x _ { t + 1 } ^ { 2 } \mid x _ { t } ] = x _ { t } ^ { 2 } - 2 \eta x _ { t } M ( x _ { t } ) + \eta ^ { 2 } .
$$

Take expectations and sum over $t = 1 , \ldots , K$ . Since $x _ { 1 } = 0$ 9

$$
\mathbb { E } x _ { K + 1 } ^ { 2 } = \sum _ { t = 1 } ^ { K } ( - 2 \eta \mathbb { E } [ x _ { t } M ( x _ { t } ) ] + \eta ^ { 2 } ) \le 2 \eta K + \eta ^ { 2 } K .\tag{A.5}
$$

All moments are finite, since for a fixed horizon, $| x _ { t } | \leq ( t - 1 ) \eta$ on every sample path.

Taking expectations gives $\mathbb { E } x _ { t + 1 } - \mathbb { E } x _ { t } = - \eta \mathbb { E } M ( x _ { t } )$ . After summing and using $x _ { 1 } = 0$

$$
\frac { 1 } { K } \sum _ { t = 1 } ^ { K } \mathbb { E } M ( x _ { t } ) = - \frac { \mathbb { E } x _ { K + 1 } } { \eta K } .
$$

Cauchy–Schwarz and (A.5) imply

$$
\left| \frac { 1 } { K } \sum _ { t = 1 } ^ { K } \mathbb { E } M ( x _ { t } ) \right| \leq \frac { \sqrt { \mathbb { E } x _ { K + 1 } ^ { 2 } } } { \eta K } \leq \sqrt { \frac { 2 } { \eta K } + \frac { 1 } { K } } .\tag{A.6}
$$

This argument controls the average vote even though individual iterates need not converge.

Finally, the triangle inequality and (A.6) give

$$
\begin{array} { r l r } {  { u ^ { 3 } \leq | \frac { 1 } { K } \sum _ { t = 1 } ^ { K } \mathbb { E } M ( x _ { t } ) | + | u ^ { 3 } - \frac { 1 } { K } \sum _ { t = 1 } ^ { K } \mathbb { E } M ( x _ { t } ) | } } \\ & { } & { \leq \sqrt { \frac { 2 } { \eta K } + \frac { 1 } { K } } + \frac { 1 } { K } \sum _ { t = 1 } ^ { K } \mathbb { E } | u ^ { 3 } - M ( x _ { t } ) | } \\ & { } & { \leq \sqrt { \frac { 2 } { \eta K } + \frac { 1 } { K } } + \frac { 7 7 1 } { 2 0 4 8 } \frac { 1 } { K } \sum _ { t = 1 } ^ { K } \mathbb { E } | h ( x _ { t } ) | . } \end{array}
$$

By independent uniform output selection, the final average is $\mathbb { E } | f ^ { \prime } ( x _ { \tau } ) |$ . Since $u ^ { 3 } = 1 / 4 0 9 6$ rearranging gives the finite-horizon bound

$$
\mathbb { E } | f ^ { \prime } ( x _ { \tau } ) | \ge \frac { 2 0 4 8 } { 7 7 1 } \left[ \frac { 1 } { 4 0 9 6 } - \sqrt { \frac { 2 } { \eta K } + \frac { 1 } { K } } \right] .\tag{A.7}
$$

For $\eta = \Theta ( K ^ { - 1 / 2 } )$ , we have $\eta K  \infty$ , so the square-root term tends to zero. Taking the lower limit and using $2 0 4 8 / ( 7 7 1 \cdot 4 0 9 6 ) = 1 / 1 5 4 2$ proves (15).

## B Proofs for the stochastic results

We first prove a tracking estimate that retains the actual model movement. Throughout, $a = 1 + \omega$ $c = a / n$ , K counts model updates, and $B _ { 0 }$ is the initialization batch per worker.

## B.1 Moments and server tracking

For $t \geq 2$ , let $\mathcal { F } _ { t - 1 }$ contain all randomness generated before the current worker samples and compression calls. Write $\mathbb { E } _ { t } [ \cdot ] = \mathbb { E } [ \cdot \mid \mathcal { F } _ { t - 1 } ]$ . First, unbiased compression controls second moments. Conditional on an input $v ,$ expansion of the squared norm gives

$$
\begin{array} { r } { \mathbb { E } [ \| Q ( v ) \| _ { 2 } ^ { 2 } \mid v ] = \| v \| _ { 2 } ^ { 2 } + 2 \langle v , \mathbb { E } [ Q ( v ) - v \mid v ] \rangle + \mathbb { E } [ \| Q ( v ) - v \| _ { 2 } ^ { 2 } \mid v ] \leq a \| v \| _ { 2 } ^ { 2 } . } \end{array}
$$

The middle term is zero by unbiasedness. Applying $\begin{array} { r } { \| u + v \| _ { 2 } ^ { 2 } \leq 2 \| u \| _ { 2 } ^ { 2 } + 2 \| v \| _ { 2 } ^ { 2 } } \end{array}$ to the decomposition in (16), we obtain

$$
\begin{array} { r l } & { \mathbb { E } _ { t } \| h _ { t } ^ { j } \| _ { 2 } ^ { 2 } \leq 2 \beta ^ { 2 } \mathbb { E } _ { t } \| g _ { j } ( x _ { t } ; \xi _ { t } ^ { j } ) \| _ { 2 } ^ { 2 } + 2 ( 1 - \beta ) ^ { 2 } \mathbb { E } _ { t } \| g _ { j } ( x _ { t } ; \xi _ { t } ^ { j } ) - g _ { j } ( x _ { t - 1 } ; \xi _ { t } ^ { j } ) \| _ { 2 } ^ { 2 } } \\ & { \qquad \leq 2 \beta ^ { 2 } H ^ { 2 } + 2 L ^ { 2 } \| x _ { t } - x _ { t - 1 } \| _ { 2 } ^ { 2 } . } \end{array}\tag{B.1}
$$

The last inequality uses mean-squared smoothness and $( 1 - \beta ) ^ { 2 } \leq 1$

Next, we control the initial error. Write $Y _ { j , b } = Q ( g _ { j } ( x _ { 1 } ; \xi _ { 1 , b } ^ { j } ) )$ for the message from sample b at worker $j .$ . Repeated conditioning gives

$$
\mathbb { E } Y _ { j , b } = \nabla f _ { j } ( x _ { 1 } ) , \qquad \mathbb { E } \| Y _ { j , b } \| _ { 2 } ^ { 2 } \leq a H ^ { 2 } .
$$

Its centered second moment therefore satisfies

$$
\begin{array} { r } { \mathbb { E } \| Y _ { j , b } - \nabla f _ { j } ( x _ { 1 } ) \| _ { 2 } ^ { 2 } = \mathbb { E } \| Y _ { j , b } \| _ { 2 } ^ { 2 } - \| \nabla f _ { j } ( x _ { 1 } ) \| _ { 2 } ^ { 2 } \leq a H ^ { 2 } . } \end{array}
$$

Distinct initialization messages use independent samples and compression calls. For $( j , b ) \neq ( k , r )$ 2 their centered cross term is consequently

$$
\begin{array} { r } { \mathbb { E } \langle Y _ { j , b } - \nabla f _ { j } ( x _ { 1 } ) , Y _ { k , r } - \nabla f _ { k } ( x _ { 1 } ) \rangle = 0 . } \end{array}
$$

Expanding the squared norm of the average gives

$$
E _ { 1 } = \frac { 1 } { n ^ { 2 } B _ { 0 } ^ { 2 } } \sum _ { j = 1 } ^ { n } \sum _ { b = 1 } ^ { B _ { 0 } } \mathbb { E } \| Y _ { j , b } - \nabla f _ { j } ( x _ { 1 } ) \| _ { 2 } ^ { 2 } \leq \frac { n B _ { 0 } a H ^ { 2 } } { n ^ { 2 } B _ { 0 } ^ { 2 } } = \frac { c H ^ { 2 } } { B _ { 0 } } .
$$

This explains why each initialization sample is compressed separately. We now derive the tracking recursion. For $t \geq 2$ , define the centered decoded error

$$
\delta _ { t } ^ { j } = Q ( h _ { t } ^ { j } ) - [ \nabla f _ { j } ( x _ { t } ) - ( 1 - \beta ) \nabla f _ { j } ( x _ { t - 1 } ) ] .
$$

Unbiasedness of the oracle and of the compressor implies

$$
\mathbb { E } _ { t } Q ( h _ { t } ^ { j } ) = \mathbb { E } _ { t } h _ { t } ^ { j } = \nabla f _ { j } ( x _ { t } ) - ( 1 - \beta ) \nabla f _ { j } ( x _ { t - 1 } ) , \qquad \mathbb { E } _ { t } \delta _ { t } ^ { j } = 0 .
$$

Subtracting $\nabla f ( x _ { t } )$ from (17) gives the exact identity

$$
e _ { t } = ( 1 - \beta ) e _ { t - 1 } + \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \delta _ { t } ^ { j } .\tag{B.2}
$$

The conditional squared norm expands as

$$
\begin{array} { l } { \displaystyle \mathbb { E } _ { t } \| e _ { t } \| _ { 2 } ^ { 2 } = ( 1 - \beta ) ^ { 2 } \| e _ { t - 1 } \| _ { 2 } ^ { 2 } + \frac { 2 ( 1 - \beta ) } { n } \sum _ { j } \langle e _ { t - 1 } , \mathbb { E } _ { t } \delta _ { t } ^ { j } \rangle } \\ { \displaystyle \qquad + \frac { 1 } { n ^ { 2 } } \sum _ { j } \mathbb { E } _ { t } \| \delta _ { t } ^ { j } \| _ { 2 } ^ { 2 } + \frac { 2 } { n ^ { 2 } } \sum _ { j < k } \mathbb { E } _ { t } \langle \delta _ { t } ^ { j } , \delta _ { t } ^ { k } \rangle . } \end{array}\tag{B.3}
$$

The second term is zero because $e _ { t - 1 }$ is fixed under conditioning. The last term is zero because the current worker errors are conditionally independent and centered. For each remaining variance, centering and (B.1) give

$$
\begin{array} { r l } & { \mathbb { E } _ { t } \| \delta _ { t } ^ { j } \| _ { 2 } ^ { 2 } = \mathbb { E } _ { t } \| Q ( h _ { t } ^ { j } ) \| _ { 2 } ^ { 2 } - \| \nabla f _ { j } ( x _ { t } ) - ( 1 - \beta ) \nabla f _ { j } ( x _ { t - 1 } ) \| _ { 2 } ^ { 2 } } \\ & { \qquad \leq a \mathbb { E } _ { t } \| h _ { t } ^ { j } \| _ { 2 } ^ { 2 } \leq 2 a \beta ^ { 2 } H ^ { 2 } + 2 a L ^ { 2 } \| x _ { t } - x _ { t - 1 } \| _ { 2 } ^ { 2 } . } \end{array}\tag{B.4}
$$

Taking full expectations and using $( 1 - \beta ) ^ { 2 } \leq 1 - \beta$ yields

$$
\begin{array} { r } { E _ { t } \leq ( 1 - \beta ) E _ { t - 1 } + 2 c \beta ^ { 2 } H ^ { 2 } + 2 c L ^ { 2 } \mathbb { E } \| x _ { t } - x _ { t - 1 } \| _ { 2 } ^ { 2 } . } \end{array}\tag{B.5}
$$

For completeness, summing this recursion from $t = 2$ to $K$ gives

$$
\begin{array} { r l r } {  { \beta \sum _ { t = 1 } ^ { K } E _ { t } \le E _ { 1 } - ( 1 - \beta ) E _ { K } + 2 c \beta ^ { 2 } H ^ { 2 } ( K - 1 ) + 2 c L ^ { 2 } \sum _ { t = 1 } ^ { K - 1 } \mathbb { E } \| x _ { t + 1 } - x _ { t } \| _ { 2 } ^ { 2 } } } \\ & { } & { \le E _ { 1 } + 2 c \beta ^ { 2 } H ^ { 2 } K + 2 c L ^ { 2 } \displaystyle \sum _ { t = 1 } ^ { K - 1 } \mathbb { E } \| x _ { t + 1 } - x _ { t } \| _ { 2 } ^ { 2 } . } \end{array}\tag{B.6}
$$

After division by $\beta K$ , this proves

$$
\overline { { E } } _ { K } \leq \frac { c H ^ { 2 } } { B _ { 0 } \beta K } + 2 c \beta H ^ { 2 } + \frac { 2 c L ^ { 2 } } { \beta K } \sum _ { t = 1 } ^ { K - 1 } \mathbb { E } \| x _ { t + 1 } - x _ { t } \| _ { 2 } ^ { 2 } .\tag{B.7}
$$

For $\mathrm { D V R - S i g n , ~ } \| x _ { t + 1 } - x _ { t } \| _ { 2 } ^ { 2 } = \eta ^ { 2 } d$ . Substituting this identity into (B.5) and (B.7) proves (23) and (24). The DVR-Q specialization is given separately in Appendix B.3.

## B.2 The $\ell _ { 1 }$ guarantee

Let $g$ be a true gradient and z an arbitrary estimate. If $\mathrm { S i g n } ( g _ { k } ) \ne \mathrm { S i g n } ( z _ { k } )$ and $g _ { k } \neq 0 ,$ , then $z _ { k }$ is on the opposite side of zero or is zero, and therefore $\left| g _ { k } \right| \le \left| g _ { k } - z _ { k } \right|$ . Coordinates with $g _ { k } = 0$ contribute zero. It follows that

$$
\begin{array} { r l r } {  { \| g \| _ { 1 } - \langle g , \mathrm { S i g n } ( z ) \rangle = 2 \sum _ { k = 1 } ^ { d } \vert g _ { k } \vert \mathbf { 1 } \{ \mathrm { S i g n } ( g _ { k } ) \neq \mathrm { S i g n } ( z _ { k } ) \} } } \\ & { } & { \leq 2 \sum _ { k = 1 } ^ { d } \vert g _ { k } - z _ { k } \vert = 2 \| g - z \| _ { 1 } \leq 2 \sqrt { d } \| g - z \| _ { 2 } . } \end{array}\tag{B.8}
$$

Next, we sum the objective decreases. Mean-squared smoothness and Jensen’s inequality give

$$
\| \nabla f _ { j } ( x ) - \nabla f _ { j } ( y ) \| _ { 2 } = \| \mathbb { E } [ g _ { j } ( x ; \xi ) - g _ { j } ( y ; \xi ) ] \| _ { 2 } \le \big ( \mathbb { E } \| g _ { j } ( x ; \xi ) - g _ { j } ( y ; \xi ) \| _ { 2 } ^ { 2 } \big ) ^ { 1 / 2 } \le L \| x - y \| _ { 2 } .
$$

Averaging over workers shows that $f$ is also L-smooth. Applying its smoothness inequality to the deterministic sign update gives

$$
\begin{array} { r l } & { f ( x _ { t + 1 } ) \leq f ( x _ { t } ) - \eta \langle \nabla f ( x _ { t } ) , \mathrm { S i g n } ( z _ { t } ) \rangle + \displaystyle \frac { L \eta ^ { 2 } } { 2 } \| \mathrm { S i g n } ( z _ { t } ) \| _ { 2 } ^ { 2 } } \\ & { \qquad \leq f ( x _ { t } ) - \eta \| \nabla f ( x _ { t } ) \| _ { 1 } + 2 \eta \sqrt { d } \| e _ { t } \| _ { 2 } + \displaystyle \frac { L \eta ^ { 2 } d } { 2 } . } \end{array}\tag{B.9}
$$

Taking expectations and summing from $t = 1$ to $K ,$ , we have

$$
\eta \sum _ { t = 1 } ^ { K } \mathbb { E } \| \nabla f ( x _ { t } ) \| _ { 1 } \le \Delta _ { f } + 2 \eta \sqrt { d } \sum _ { t = 1 } ^ { K } \mathbb { E } \| e _ { t } \| _ { 2 } + \frac { K L \eta ^ { 2 } d } { 2 } .
$$

For the error sum, Jensen’s inequality followed by Cauchy–Schwarz gives

$$
\frac { 1 } { K } \sum _ { t = 1 } ^ { K } \mathbb { E } \| e _ { t } \| _ { 2 } \leq \frac { 1 } { K } \sum _ { t = 1 } ^ { K } \sqrt { E _ { t } } \leq \frac { 1 } { K } \sqrt { K \sum _ { t = 1 } ^ { K } E _ { t } } = \sqrt { \overline { { E } } _ { K } } .
$$

Dividing the previous descent inequality by ηK proves

$$
\frac { 1 } { K } \sum _ { t = 1 } ^ { K } \mathbb { E } \| \nabla f ( x _ { t } ) \| _ { 1 } \leq \frac { \Delta _ { f } } { \eta K } + \frac { L \eta d } { 2 } + 2 \sqrt { d \overline { { E } } _ { K } } .\tag{B.10}
$$

Because $\tau$ is uniform on $\{ 1 , \ldots , K \}$ and independent of the run, its expected gradient norm equals the average on the left. This proves Theorem 2. For Corollary 3, choose

$$
u _ { 1 } = \frac { 1 } { \sqrt { K } + c ^ { 1 / 3 } K ^ { 2 / 3 } } , \qquad \beta = u _ { 1 } , \quad \eta = \frac { H u _ { 1 } } { L \sqrt { d } } , \quad B _ { 0 } = \left\lceil \frac { 1 } { u _ { 1 } ^ { 2 } K } \right\rceil .\tag{B.11}
$$

These are admissible because $0 < u _ { 1 } \le K ^ { - 1 / 2 } \le 1$ and $B _ { 0 } \geq 1$ . Inserting the parameters into (24) controls each contribution separately:

$$
\frac { c H ^ { 2 } } { B _ { 0 } \beta K } \leq c H ^ { 2 } u _ { 1 } , \qquad 2 c H ^ { 2 } \beta = 2 c H ^ { 2 } u _ { 1 } , \qquad \frac { 2 c L ^ { 2 } \eta ^ { 2 } d } \beta = 2 c H ^ { 2 } u _ { 1 } .
$$

Thus $\overline { { E } } _ { K } \leq 5 c H ^ { 2 } u _ { 1 }$ , and (B.10) becomes

$$
\mathbb { E } \| \nabla f ( x _ { \tau } ) \| _ { 1 } \leq \sqrt { d } \left[ \frac { L \Delta _ { f } } { H u _ { 1 } K } + \frac { H u _ { 1 } } { 2 } + 2 \sqrt { 5 } H \sqrt { c u _ { 1 } } \right] .
$$

The definition of $u _ { 1 }$ gives

$$
\frac { 1 } { u _ { 1 } K } = K ^ { - 1 / 2 } + c ^ { 1 / 3 } K ^ { - 1 / 3 } , \qquad u \le K ^ { - 1 / 2 } , \qquad u _ { 1 } \le c ^ { - 1 / 3 } K ^ { - 2 / 3 } .
$$

In particular, the third inequality implies $\sqrt { c u _ { 1 } } \leq c ^ { 1 / 3 } K ^ { - 1 / 3 }$ . Substitution proves the explicit bound

$$
\mathbb { E } \| \nabla f ( x _ { \tau } ) \| _ { 1 } \leq \sqrt { d } \left[ \left( \frac { L \Delta _ { f } } { H } + \frac { H } { 2 } \right) K ^ { - 1 / 2 } + \left( \frac { L \Delta _ { f } } { H } + 2 \sqrt { 5 } H \right) \left( \frac { c } { K } \right) ^ { 1 / 3 } \right] .\tag{B.12}
$$

Keeping $L , H , \Delta _ { f }$ fixed and recalling $c = ( 1 + \omega ) / n$ gives (27).

Finally, we specify initialization and a target accuracy. Squaring the denominator of $u _ { 1 }$ yields the exact batch size

$$
B _ { 0 } = \left\lceil ( 1 + c ^ { 1 / 3 } K ^ { 1 / 6 } ) ^ { 2 } \right\rceil = \Theta ( 1 + c ^ { 2 / 3 } K ^ { 1 / 3 } ) .\tag{B.13}
$$

To verify the order including the ceiling, use $1 + b ^ { 2 } \leq ( 1 + b ) ^ { 2 } \leq 2 ( 1 + b ^ { 2 } )$ for $b \geq 0$ , and $y \le \lceil y \rceil \le y + 1 \le 2 y$ $y \geq 1$ . For any $\epsilon > 0$ , the following explicit integer horizon sufices:

$$
K = \operatorname* { m a x } \{ 1 ,  [ { \frac { 4 d ( L \Delta _ { f } / H + H / 2 ) ^ { 2 } } { \epsilon ^ { 2 } } } ] ,  [ { \frac { 8 c d ^ { 3 / 2 } ( L \Delta _ { f } / H + 2 { \sqrt { 5 } } H ) ^ { 3 } } { \epsilon ^ { 3 } } } ] \} .\tag{B.14}
$$

The second entry makes the first term of (B.12) at most $\epsilon / 2 \AA$ square that desired inequality and solve for K. The third entry does the same for the second term by cubing it. Their sum is therefore at most ϵ. With fixed $L , H , \Delta _ { f }$

$$
K = { \cal O } \left( 1 + \frac { d } { \epsilon ^ { 2 } } + \frac { c d ^ { 3 / 2 } } { \epsilon ^ { 3 } } \right) .\tag{B.15}
$$

## B.3 The $\ell _ { 2 }$ guarantee

Let $\mathcal { G } _ { t }$ denote the history after the server has formed $z _ { t }$ and before drawing that downlink. Thus

$$
\mathbb { E } [ Q ( z _ { t } ) \mid { \mathcal { G } } _ { t } ] = z _ { t } , \qquad \mathbb { E } [ \| Q ( z _ { t } ) \| _ { 2 } ^ { 2 } \mid { \mathcal { G } } _ { t } ] \leq a \| z _ { t } \| _ { 2 } ^ { 2 } .\tag{B.16}
$$

For clarity, write $Z _ { t } = \mathbb { E } \| z _ { t } \| _ { 2 } ^ { 2 }$ only within this proof. The actual movement satisfies

$$
\begin{array} { r } { \mathbb { E } \| x _ { t + 1 } - x _ { t } \| _ { 2 } ^ { 2 } = \eta ^ { 2 } \mathbb { E } \| Q ( z _ { t } ) \| _ { 2 } ^ { 2 } \leq a \eta ^ { 2 } Z _ { t } . } \end{array}
$$

Inserting it into the general tracking bounds gives

$$
E _ { t } \leq ( 1 - \beta ) E _ { t - 1 } + 2 c \beta ^ { 2 } H ^ { 2 } + 2 c a L ^ { 2 } \eta ^ { 2 } Z _ { t - 1 } \quad ( t \geq 2 ) ,\tag{B.17}
$$

$$
\overline { { E } } _ { K } \leq \frac { c H ^ { 2 } } { B _ { 0 } \beta K } + 2 c \beta H ^ { 2 } + \frac { 2 c a L ^ { 2 } \eta ^ { 2 } } { \beta K } \sum _ { t = 1 } ^ { K - 1 } Z _ { t } .\tag{B.18}
$$

Unlike the sign update, this movement becomes small when the tracker becomes small. We must retain its dependence on $Z _ { t }$ in the descent calculation.

By L-smoothness and (B.16),

$$
\mathbb { E } [ f ( x _ { t + 1 } ) \mid \mathcal { G } _ { t } ] \leq f ( x _ { t } ) - \eta \langle \nabla f ( x _ { t } ) , z _ { t } \rangle + \frac { L a \eta ^ { 2 } } { 2 } \| z _ { t } \| _ { 2 } ^ { 2 } .\tag{B.19}
$$

The exact inner-product identity

$$
2 \langle \nabla f ( x _ { t } ) , z _ { t } \rangle = \| \nabla f ( x _ { t } ) \| _ { 2 } ^ { 2 } + \| z _ { t } \| _ { 2 } ^ { 2 } - \| e _ { t } \| _ { 2 } ^ { 2 }
$$

then gives, after taking full expectations,

$$
\mathbb { E } f ( x _ { t + 1 } ) \leq \mathbb { E } f ( x _ { t } ) - \frac { \eta } { 2 } \mathbb { E } \| \nabla f ( x _ { t } ) \| _ { 2 } ^ { 2 } - \frac { \eta } { 2 } ( 1 - L a \eta ) Z _ { t } + \frac { \eta } { 2 } E _ { t } .\tag{B.20}
$$

Summing from $t = 1$ to $K$ , using $\mathbb { E } f ( x _ { K + 1 } ) \ge f _ { * }$ , and dividing by $\eta K / 2$ yields

$$
\frac { 1 } { K } \sum _ { t = 1 } ^ { K } \mathbb { E } \| \nabla f ( x _ { t } ) \| _ { 2 } ^ { 2 } + \frac { 1 - L a \eta } { K } \sum _ { t = 1 } ^ { K } Z _ { t } \leq \frac { 2 \Delta _ { f } } { \eta K } + \overline { { E } } _ { K } .\tag{B.21}
$$

Substitute (B.18) and use $\begin{array} { r } { \sum _ { t = 1 } ^ { K - 1 } Z _ { t } \leq \sum _ { t = 1 } ^ { K } Z _ { t } } \end{array}$ . Moving this last contribution to the left gives

$$
\begin{array} { r l r } {  { \displaystyle \frac { 1 } { K } \sum _ { t = 1 } ^ { K } \mathbb { E } \| \nabla f ( { \boldsymbol x } _ { t } ) \| _ { 2 } ^ { 2 } + ( 1 - L a \eta - \frac { 2 c a L ^ { 2 } \eta ^ { 2 } } { \beta } ) \frac { 1 } { K } \sum _ { t = 1 } ^ { K } Z _ { t } } } \\ & { } & { \le \frac { 2 \Delta _ { f } } { \eta K } + \frac { c H ^ { 2 } } { B _ { 0 } \beta K } + 2 c \beta H ^ { 2 } . } \end{array}\tag{B.22}
$$

The independent uniform output $\tau$ turns the first average into $\mathbb { E } \| \nabla f ( x _ { \tau } ) \| _ { 2 } ^ { 2 }$ . If $L a \eta \le 1 / 2$ and $\beta \geq 4 c a L ^ { 2 } \eta ^ { 2 }$ , both subtracted terms in the tracker coeficient are at most $1 / 2$ , so the coeficient is nonnegative. This proves Theorem 4. The cancellation explains why we did not replace $Z _ { t }$ by a gradient-magnitude bound earlier.

We next prove Corollary 5. Recall $c = a / n$ and choose

$$
r = ( K / n ^ { 2 } ) ^ { 1 / 3 } , \qquad \eta = \frac { 1 } { 2 L a ( 1 + r ) } , \quad \beta = \frac { 1 } { n ( 1 + r ) ^ { 2 } } , \quad B _ { 0 } = \lceil 1 + r \rceil .\tag{B.23}
$$

These choices satisfy $0 < \beta \leq 1$ for every $n , K \geq 1$ . Direct substitution gives

$$
L a \eta = \frac { 1 } { 2 ( 1 + r ) } , \qquad 4 c a { L ^ { 2 } } { \eta ^ { 2 } } = \frac { 1 } { n ( 1 + r ) ^ { 2 } } = \beta .
$$

The tracker coeficient in (28) is consequently

$$
1 - \frac { 1 } { 2 ( 1 + r ) } - \frac { 1 } { 2 } = \frac { r } { 2 ( 1 + r ) } \geq 0 .
$$

We can drop that term and bound the three remaining terms separately:

$$
\begin{array} { r l r } {  { \frac { 2 \Delta _ { f } } { \eta K } = \frac { 4 L a \Delta _ { f } ( 1 + r ) } { K } , } } \\ & { \frac { c H ^ { 2 } } { B _ { 0 } \beta K } = \frac { a H ^ { 2 } ( 1 + r ) ^ { 2 } } { B _ { 0 } K } \leq \frac { a H ^ { 2 } ( 1 + r ) } { K } , } \\ & { 2 c \beta H ^ { 2 } = \frac { 2 a H ^ { 2 } } { n ^ { 2 } ( 1 + r ) ^ { 2 } } = \frac { 2 a H ^ { 2 } r ^ { 3 } } { K ( 1 + r ) ^ { 2 } } \leq \frac { 2 a H ^ { 2 } r } { K } . } \end{array}\tag{B.24}
$$

The second bound uses $B _ { 0 } \geq 1 + r$ . The last uses $r ^ { 3 } = K / n ^ { 2 }$ and $r ^ { 2 } \leq ( 1 + r ) ^ { 2 }$ . Adding the bounds and using $r / K = ( n K ) ^ { - 2 / 3 }$ proves

$$
\mathbb { E } \| \nabla f ( x _ { \tau } ) \| _ { 2 } ^ { 2 } \leq a \left[ \frac { 4 L \Delta _ { f } + H ^ { 2 } } { K } + \frac { 4 L \Delta _ { f } + 3 H ^ { 2 } } { ( n K ) ^ { 2 / 3 } } \right] .\tag{B.25}
$$

Jensen’s inequality and ${ \sqrt { u + v } } \leq { \sqrt { u } } + { \sqrt { v } }$ for $u , v \geq 0$ give

$$
\begin{array} { r l } & { \mathbb { E } \| \nabla f ( x _ { \tau } ) \| _ { 2 } \leq \sqrt { \mathbb { E } \| \nabla f ( x _ { \tau } ) \| _ { 2 } ^ { 2 } } } \\ & { \qquad \leq \sqrt { 4 L \Delta _ { f } + H ^ { 2 } } \sqrt { \displaystyle \frac { a } { K } } + \sqrt { 4 L \Delta _ { f } + 3 H ^ { 2 } } \frac { \sqrt { a } } { ( n K ) ^ { 1 / 3 } } . } \end{array}\tag{B.26}
$$

This proves (30). To obtain $\mathbb { E } \| \nabla f ( x _ { \tau } ) \| _ { 2 } ^ { 2 } \le \epsilon ^ { 2 }$ , and hence $\mathbb { E } \| \nabla f ( x _ { \tau } ) \| _ { 2 } \leq \epsilon .$ , it sufices to choose

$$
K = \operatorname* { m a x } \Bigg \{ 1 , \left\lceil \frac { 2 a ( 4 L \Delta _ { f } + H ^ { 2 } ) } { \epsilon ^ { 2 } } \right\rceil , \left\lceil \frac { [ 2 a ( 4 L \Delta _ { f } + 3 H ^ { 2 } ) ] ^ { 3 / 2 } } { n \epsilon ^ { 3 } } \right\rceil \Bigg \} .\tag{B.27}
$$

The second entry makes the first term in (B.25) at most $\epsilon ^ { 2 } / 2$ . Raising the desired bound on its second term to the power $3 / 2$ gives the third entry. Their sum is at most $\epsilon ^ { 2 }$ . For fixed $L , H , \Delta _ { f }$ the resulting update and per-worker sample complexities are

$$
K = O \left( 1 + \frac { a } { \epsilon ^ { 2 } } + \frac { a ^ { 3 / 2 } } { n \epsilon ^ { 3 } } \right) .\tag{B.28}
$$

## C Proofs for finite sums

## C.1 Exact computation

For K updates and an integer refresh period $q \geq 1$ , the times $1 , 1 + q , 1 + 2 q , . .$ . give

$$
r = 1 + \left\lfloor { \frac { K - 1 } { q } } \right\rfloor = \left\lceil { \frac { K } { q } } \right\rceil .
$$

Each refresh evaluates every component once for $M = n m$ evaluations. At each of the $K - r$ ordinary updates, every worker evaluates one component at two points. Therefore

$$
N _ { \mathrm { g r a d } } = M r + 2 n ( K - r ) , \qquad N _ { \mathrm { g r a d } , j } = m r + 2 ( K - r ) .\tag{C.1}
$$

For $q = m$ , the inequality $r \leq 1 + K / m$ gives

$$
N _ { \mathrm { g r a d } } \leq M \left( 1 + { \frac { K } { m } } \right) + 2 n K = M + 3 n K .\tag{C.2}
$$

## C.2 Tracking error between exact refreshes

Lemma C.1. For the estimator (32), the error is zero at each refresh $t _ { 0 } = 1 + k q$ . For $t _ { 0 } \leq t < t _ { 0 } + q ,$ we have

$$
\mathbb { E } \| z _ { t } - \nabla f ( x _ { t } ) \| _ { 2 } ^ { 2 } \leq \frac { a L ^ { 2 } } { n } \sum _ { s = t _ { 0 } + 1 } ^ { t } \mathbb { E } \| x _ { s } - x _ { s - 1 } \| _ { 2 } ^ { 2 } .\tag{C.3}
$$

Deterministic sign updates therefore give the bound a $L ^ { 2 } \eta ^ { 2 } d ( q - 1 ) / n$ . For updates $x _ { t + 1 } = x _ { t } - \eta Q ( z _ { t } )$ 2 with a fresh common downlink satisfying the same compressor assumption,

$$
\frac { 1 } { K } \sum _ { t = 1 } ^ { K } \mathbb { E } \| z _ { t } - \nabla f ( x _ { t } ) \| _ { 2 } ^ { 2 } \leq \frac { a ^ { 2 } L ^ { 2 } \eta ^ { 2 } ( q - 1 ) } { n } \frac { 1 } { K } \sum _ { t = 1 } ^ { K } \mathbb { E } \| z _ { t } \| _ { 2 } ^ { 2 } .\tag{C.4}
$$

Proof. At an ordinary iteration, condition on the history through $x _ { t }$ and before the current component indices and compression calls. Denote this conditional expectation by $\mathbb { E } _ { t } ;$ the points $x _ { t } , x _ { t - 1 }$ and the previous estimate $z _ { t - 1 }$ are fixed under this conditioning. Write

$$
\begin{array} { r } { { X } _ { j } = Q \big ( \nabla f _ { j , i _ { t } ^ { j } } ( x _ { t } ) - \nabla f _ { j , i _ { t } ^ { j } } ( x _ { t - 1 } ) \big ) , } \\ { { \mu } _ { j } = \mathbb { E } _ { t } { X } _ { j } = \nabla f _ { j } ( x _ { t } ) - \nabla f _ { j } ( x _ { t - 1 } ) . } \end{array}
$$

Uniform component sampling and unbiased compression give the expression for $\mu _ { j }$ . The same component is evaluated at both points. Thus

$$
\mathbb { E } _ { t } \Vert X _ { j } \Vert _ { 2 } ^ { 2 } \leq \frac { a } { m } \sum _ { i = 1 } ^ { m } \Vert \nabla f _ { j , i } ( x _ { t } ) - \nabla f _ { j , i } ( x _ { t - 1 } ) \Vert _ { 2 } ^ { 2 } \leq a L ^ { 2 } \Vert x _ { t } - x _ { t - 1 } \Vert _ { 2 } ^ { 2 } .
$$

The first inequality uses (7); the second is component smoothness. Since $X _ { j } - \mu _ { j }$ is centered,

$$
\begin{array} { r } { \mathbb { E } _ { t } \| X _ { j } - \mu _ { j } \| _ { 2 } ^ { 2 } = \mathbb { E } _ { t } \| X _ { j } \| _ { 2 } ^ { 2 } - \| \mu _ { j } \| _ { 2 } ^ { 2 } \leq a L ^ { 2 } \| x _ { t } - x _ { t - 1 } \| _ { 2 } ^ { 2 } . } \end{array}
$$

Note that the component indices and compressor calls are conditionally independent across all n workers. For $j \neq k$ , this independence and centering imply

$$
\begin{array} { r } { \mathbb { E } _ { t } \langle X _ { j } - \mu _ { j } , X _ { k } - \mu _ { k } \rangle = \langle \mathbb { E } _ { t } ( X _ { j } - \mu _ { j } ) , \mathbb { E } _ { t } ( X _ { k } - \mu _ { k } ) \rangle = 0 . } \end{array}
$$

Consequently,

$$
\mathbb { E } _ { t } \left. \frac { 1 } { n } \sum _ { j = 1 } ^ { n } ( X _ { j } - \mu _ { j } ) \right. ^ { 2 } = \frac { 1 } { n ^ { 2 } } \sum _ { j = 1 } ^ { n } \mathbb { E } _ { t } \Vert X _ { j } - \mu _ { j } \Vert _ { 2 } ^ { 2 } \leq \frac { a L ^ { 2 } } { n } \Vert x _ { t } - x _ { t - 1 } \Vert _ { 2 } ^ { 2 } .
$$

The averaging is over all workers and $\begin{array} { r } { n ^ { - 1 } \sum _ { j } \mu _ { j } = \nabla f ( x _ { t } ) - \nabla f ( x _ { t - 1 } ) } \end{array}$ exactly.

Next, let $e _ { t } = z _ { t } - \nabla f ( x _ { t } )$ . The estimator recursion gives

$$
e _ { t } = e _ { t - 1 } + \frac { 1 } { n } \sum _ { j = 1 } ^ { n } ( X _ { j } - \mu _ { j } ) .
$$

The new sum has zero conditional mean, and $e _ { t - 1 }$ is fixed under $\mathbb { E } _ { t }$ . Expanding the square yields

$$
\begin{array} { r l r } {  { \mathbb { E } _ { t } \| e _ { t } \| _ { 2 } ^ { 2 } = \| e _ { t - 1 } \| _ { 2 } ^ { 2 } + 2  e _ { t - 1 } , \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \mathbb { E } _ { t } ( X _ { j } - \mu _ { j } )  + \mathbb { E } _ { t } \| \frac { 1 } { n } \sum _ { j = 1 } ^ { n } ( X _ { j } - \mu _ { j } ) \| ^ { 2 } } } \\ & { } & { \leq \| e _ { t - 1 } \| _ { 2 } ^ { 2 } + \frac { a L ^ { 2 } } { n } \| x _ { t } - x _ { t - 1 } \| _ { 2 } ^ { 2 } . } \end{array}
$$

At a refresh, $z _ { t _ { 0 } } = \nabla f ( x _ { t _ { 0 } } )$ , so $e _ { t _ { 0 } } = 0$ on every sample path. Taking total expectations and applying the recursion successively at $t _ { 0 } + 1 , \ldots , t$ gives

$$
\mathbb { E } \| e _ { t } \| _ { 2 } ^ { 2 } \leq \mathbb { E } \| e _ { t _ { 0 } } \| _ { 2 } ^ { 2 } + \frac { a L ^ { 2 } } { n } \sum _ { s = t _ { 0 } + 1 } ^ { t } \mathbb { E } \| x _ { s } - x _ { s - 1 } \| _ { 2 } ^ { 2 } .
$$

This proves (C.3). For DVR-Sign-FS, each summand is $\eta ^ { 2 } d ,$ and there are at most $q - 1$ summands.

For DVR-Q-FS, condition on the history $\mathcal { G } _ { s }$ after all uplinks in iteration s and before the server’s new compression call. Then

$$
\begin{array} { r } { \mathbb { E } [ \| x _ { s + 1 } - x _ { s } \| _ { 2 } ^ { 2 } \mid \mathcal { G } _ { s } ] = \eta ^ { 2 } \mathbb { E } [ \| Q ( z _ { s } ) \| _ { 2 } ^ { 2 } \mid \mathcal { G } _ { s } ] \le a \eta ^ { 2 } \| z _ { s } \| _ { 2 } ^ { 2 } . } \end{array}
$$

Substituting its total expectation in (C.3) yields

$$
\mathbb { E } \Vert e _ { t } \Vert _ { 2 } ^ { 2 } \leq \frac { a ^ { 2 } L ^ { 2 } \eta ^ { 2 } } { n } \sum _ { s = t _ { 0 } } ^ { t - 1 } \mathbb { E } \Vert z _ { s } \Vert _ { 2 } ^ { 2 } .
$$

Let $t _ { 1 } = \operatorname* { m i n } \{ t _ { 0 } + q - 1 , K \}$ be the last iteration in this interval between refreshes. Reversing the order of the finite sums gives

$$
\begin{array} { r } { \displaystyle \sum _ { t = t _ { 0 } } ^ { t _ { 1 } } \mathbb { E } \| e _ { t } \| _ { 2 } ^ { 2 } \leq \frac { a ^ { 2 } L ^ { 2 } \eta ^ { 2 } } { n } \displaystyle \sum _ { s = t _ { 0 } } ^ { t _ { 1 } - 1 } ( t _ { 1 } - s ) \mathbb { E } \| z _ { s } \| _ { 2 } ^ { 2 } } \\ { \leq \frac { a ^ { 2 } L ^ { 2 } \eta ^ { 2 } ( q - 1 ) } { n } \displaystyle \sum _ { s = t _ { 0 } } ^ { t _ { 1 } } \mathbb { E } \| z _ { s } \| _ { 2 } ^ { 2 } . } \end{array}
$$

These intervals are disjoint. Summing over them and dividing by K proves (C.4).

## C.3 Descent for the $\ell _ { 1 }$ criterion

Proof of Theorem 6. The deterministic sign $s _ { t } = \mathrm { S i g n } ( z _ { t } )$ has squared norm $d .$ Thus the deterministic sign case of Lemma C.1 proves (34). To translate this into stationarity, smoothness gives

$$
f ( x _ { t + 1 } ) \leq f ( x _ { t } ) - \eta \langle \nabla f ( x _ { t } ) , \mathrm { S i g n } ( z _ { t } ) \rangle + \frac { L \eta ^ { 2 } d } { 2 } .
$$

By (19),

$$
\langle \nabla f ( x _ { t } ) , \mathrm { S i g n } ( z _ { t } ) \rangle \geq \| \nabla f ( x _ { t } ) \| _ { 1 } - 2 \sqrt { d } \| z _ { t } - \nabla f ( x _ { t } ) \| _ { 2 } .
$$

Substituting and rearranging before taking expectations gives

$$
\eta \mathbb { E } \| \nabla f ( x _ { t } ) \| _ { 1 } \le \mathbb { E } f ( x _ { t } ) - \mathbb { E } f ( x _ { t + 1 } ) + 2 \eta \sqrt { d } \mathbb { E } \| z _ { t } - \nabla f ( x _ { t } ) \| _ { 2 } + \frac { L \eta ^ { 2 } d } { 2 } .
$$

Sum this inequality over $t = 1 , \ldots , K$ . The objective terms telescope to $f ( x _ { 1 } ) - \mathbb { E } f ( x _ { K + 1 } ) \leq$ $f ( x _ { 1 } ) - f _ { * } = \Delta _ { f } \leq \Delta _ { 0 }$ . After division by $\eta K$ , we obtain

$$
\begin{array} { r l } {  { \frac { 1 } { K } \sum _ { t = 1 } ^ { K } \mathbb { E } \| \nabla f ( { x } _ { t } ) \| _ { 1 } \le \frac { \Delta _ { 0 } } { \eta K } + \frac { L \eta d } { 2 } + \frac { 2 \sqrt { d } } { K } \sum _ { t = 1 } ^ { K } \mathbb { E } \| { z } _ { t } - \nabla f ( { x } _ { t } ) \| _ { 2 } } } \\ & { \le \frac { \Delta _ { 0 } } { \eta K } + \frac { L \eta d } { 2 } + 2 \sqrt { d } \sqrt { \frac { a L ^ { 2 } \eta ^ { 2 } d ( q - 1 ) } { n } } } \\ & { = \frac { \Delta _ { 0 } } { \eta K } + L \eta [ \frac { d } { 2 } + 2 d \sqrt { \frac { a ( q - 1 ) } { n } } ] . } \end{array}
$$

In the second line, $\begin{array} { r } { \mathbb { E } \| z _ { t } - \nabla f ( x _ { t } ) \| _ { 2 } \leq \sqrt { \mathbb { E } \| z _ { t } - \nabla f ( x _ { t } ) \| _ { 2 } ^ { 2 } } } \end{array}$ and Lemma C.1 bound each summand.   
Uniformity of $\tau$ implies that the average on the left equals $\mathbb { E } \Vert \nabla f ( x _ { \tau } ) \Vert _ { 1 }$ . This proves (35).

Finally, $J _ { 1 } \geq d / 2 > 0$ , so the positive stepsize is well defined and $L \eta J _ { 1 } = \epsilon / 2$ . If $\Delta _ { 0 } > 0$ , the choice of K gives

$$
\frac { \Delta _ { 0 } } { \eta K } = \frac { 2 L \Delta _ { 0 } J _ { 1 } } { \epsilon K } \le \frac { \epsilon } { 2 } .
$$

Complexity Set $q = m$ . The target-accuracy choice gives, for every $\epsilon > 0$

$$
K \leq 1 + \frac { 4 L \Delta _ { 0 } J _ { 1 } } { \epsilon ^ { 2 } } = 1 + \frac { 4 L \Delta _ { 0 } d } { \epsilon ^ { 2 } } \left[ \frac { 1 } { 2 } + 2 \sqrt { \frac { a ( m - 1 ) } { n } } \right] .
$$

For the total oracle count, multiply the coeficient by n:

$$
n J _ { 1 } = \frac { d n } { 2 } + 2 d \sqrt { a n ( m - 1 ) } \le \frac { d n } { 2 } + 2 d \sqrt { a M } .\tag{C.5}
$$

The inequality uses $n ( m - 1 ) \leq n m = M$ . Substituting the iteration bound into (C.2) now gives

$$
\begin{array} { l } { \displaystyle { N _ { \mathrm { g r a d } } \leq M + 3 n + \frac { 1 2 L \Delta _ { 0 } n J _ { 1 } } { \epsilon ^ { 2 } } } } \\ { \displaystyle { \leq 4 M + \frac { 1 2 L \Delta _ { 0 } d } { \epsilon ^ { 2 } } \left[ \frac { n } { 2 } + 2 \sqrt { a M } \right] . } } \end{array}
$$

Here $n \leq M$ absorbs the extra update arising from the ceiling.

## C.4 Descent for the $\ell _ { 2 }$ criterion

Proof of Theorem 8. Condition on the post-uplink history $\mathcal { G } _ { t }$ , which includes $x _ { t }$ and $z _ { t }$ and precedes the server’s new call to $Q .$ . Unbiasedness and the second-moment bound give

$$
\mathbb { E } [ Q ( z _ { t } ) \mid { \mathcal { G } } _ { t } ] = z _ { t } , \qquad \mathbb { E } [ \| Q ( z _ { t } ) \| _ { 2 } ^ { 2 } \mid { \mathcal { G } } _ { t } ] \leq a \| z _ { t } \| _ { 2 } ^ { 2 } .
$$

Since $x _ { t + 1 } = x _ { t } - \eta Q ( z _ { t } )$ , smoothness implies

$$
\mathbb { E } [ f ( x _ { t + 1 } ) \mid \mathcal { G } _ { t } ] \leq f ( x _ { t } ) - \eta \langle \nabla f ( x _ { t } ) , z _ { t } \rangle + \frac { L a \eta ^ { 2 } } { 2 } \| z _ { t } \| _ { 2 } ^ { 2 } .
$$

The inner product has the exact decomposition

$$
2 \langle \nabla f ( { x } _ { t } ) , { z } _ { t } \rangle = \| \nabla f ( { x } _ { t } ) \| _ { 2 } ^ { 2 } + \| { z } _ { t } \| _ { 2 } ^ { 2 } - \| { z } _ { t } - \nabla f ( { x } _ { t } ) \| _ { 2 } ^ { 2 } .
$$

Substitute this identity, take total expectations, and rearrange:

$$
\frac { \eta } { 2 } \mathbb { E } \| \nabla f ( x _ { t } ) \| _ { 2 } ^ { 2 } + \frac { \eta } { 2 } ( 1 - L a \eta ) \mathbb { E } \| z _ { t } \| _ { 2 } ^ { 2 } \leq \mathbb { E } f ( x _ { t } ) - \mathbb { E } f ( x _ { t + 1 } ) + \frac { \eta } { 2 } \mathbb { E } \| z _ { t } - \nabla f ( x _ { t } ) \| _ { 2 } ^ { 2 } .
$$

Sum the last inequality over $t = 1 , \ldots , K$ and divide by $\eta K / 2$ . The objective values telescope, and $f ( x _ { 1 } ) - \mathbb { E } f ( x _ { K + 1 } ) \leq \Delta _ { 0 }$ . Applying (C.4) to the right side gives

$$
\begin{array} { r l } & { \cfrac { 1 } { K } \displaystyle \sum _ { t = 1 } ^ { K } \mathbb { E } \| \nabla f ( x _ { t } ) \| _ { 2 } ^ { 2 } + \left[ 1 - L a \eta - \frac { a ^ { 2 } L ^ { 2 } \eta ^ { 2 } ( q - 1 ) } { n } \right] \frac { 1 } { K } \displaystyle \sum _ { t = 1 } ^ { K } \mathbb { E } \| z _ { t } \| _ { 2 } ^ { 2 } } \\ & { \qquad \leq \frac { 2 \Delta _ { 0 } } { \eta K } . } \end{array}\tag{C.6}
$$

Write $s = \sqrt { ( q - 1 ) / n } \geq 0$ . The choice in (38) gives

$$
L a \eta + \frac { a ^ { 2 } L ^ { 2 } \eta ^ { 2 } ( q - 1 ) } { n } = \frac { 1 } { 2 ( 1 + s ) } + \frac { s ^ { 2 } } { 4 ( 1 + s ) ^ { 2 } } = \frac { 1 } { 4 } + \frac { 1 } { 4 ( 1 + s ) ^ { 2 } } \le \frac { 1 } { 2 } .
$$

Thus the bracket in (C.6) is at least $1 / 2$ , and we may discard its nonnegative contribution. Since $2 \Delta _ { 0 } / ( \eta K ) = 4 L \Delta _ { 0 } J _ { Q } / K$ , we obtain the squared-norm bound in (39). The independent uniform choice of τ identifies its expectation with the averaged left side. Cauchy–Schwarz then gives

$$
\mathbb { E } \| \nabla f ( x _ { \tau } ) \| _ { 2 } \leq \sqrt { \mathbb { E } \| \nabla f ( x _ { \tau } ) \| _ { 2 } ^ { 2 } } \leq 2 \sqrt { \frac { L \Delta _ { 0 } J _ { Q } } { K } } .
$$

Complexity For $q = m$ , we have $J _ { Q } = a ( 1 + \sqrt { ( m - 1 ) / n } )$ . Hence

$$
K \leq 1 + \frac { 4 L \Delta _ { 0 } J _ { Q } } { \epsilon ^ { 2 } } = 1 + \frac { 4 a L \Delta _ { 0 } } { \epsilon ^ { 2 } } \left[ 1 + \sqrt { \frac { m - 1 } { n } } \right] ,
$$

which proves (40). To count gradients, use

$$
n J _ { Q } = a \left[ n + \sqrt { n ( m - 1 ) } \right] \leq a ( n + \sqrt { M } ) .
$$

Then (C.2) yields

$$
\begin{array} { r l r } {  { N _ { \mathrm { g r a d } } \le M + 3 n + \frac { 1 2 L \Delta _ { 0 } n J _ { Q } } { \epsilon ^ { 2 } } } } \\ & { } & { \le 4 M + \frac { 1 2 a L \Delta _ { 0 } } { \epsilon ^ { 2 } } ( n + \sqrt { M } ) . } \end{array}
$$