# One-shot Distributed Learning via Weighted Averaging with Server-side Information

Pavle Tepavceviˇ c, Dušan Jakoveti´ c, Miloš Savi´ c´

Department of Mathematics and Informatics, Faculty of Sciences, University of Novi Sad, Serbia {pavle.tepavcevic, dusan.jakovetic, svc}@dmi.uns.ac.rs

Abstract—Distributed learning has drawn considerable attention due to its capability of learning from distributed data without aggregating raw observations. However, frequent exchange of model updates can become the main bottleneck of communication. Therefore, one-shot approaches where each worker fits a local model and sends it to the server to aggregate emerge. It is known that averaging can achieve the so-called pooled-data mean squared error (MSE) order of a hypothetical node that has access to data from the server and all the workers, and performs empirical risk minimization (ERM) on it. However, when the number of workers increases too fast compared to the local sample size, nonzero local estimation bias prevents averaging from achieving this order.

In this paper, we study whether independent server data can overcome this limitation. We consider m workers with n observations each and a server with p observations, all i.i.d., giving $N = m n { + } p$ observations in total. Under certain regularity conditions, a pooled-data ERM achieves MSE of order $\bar { \mathcal { O } } ( N ^ { - 1 } )$ We analyze deterministic nonnegative averaging of convex-loss M-estimators, with equal worker weights and the server’s own estimator included. Under our assumptions, including nonzero local bias of order $n ^ { - \alpha } , \alpha \geq 1$ , sample-size weighted averaging achieves the pooled-data MSE order whenever ${ \bar { m } } = { \mathcal { O } } ( n ^ { 2 \alpha ^ { - 1 } } )$ Above this worker scale, it does so if and only if $p = \Omega ( n m$ $m / n ^ { 2 \alpha - 1 } )$ . Hence, server-only estimation, whose MSE is of order $\mathcal { O } ( p ^ { - 1 } )$ , can achieve pooled-data order while sample-size weighting fails. Analytical examples and synthetic experiments illustrate when averaging matches the pooled-data MSE order, complemented by CIFAR-10 experiments comparing aggregation strategies with linear and nonlinear classifiers.

Index Terms—one-shot distributed learning, M-estimation, order-optimality, server-side information, weighted averaging.

## I. INTRODUCTION

Smart devices and IoT systems produce data from various locations. Collection of all this data at a central server may be expensive or undesirable due to their large number and privacy concerns [1]. This gives rise to learning techniques that operate on data where it resides. Distributed learning involves cooperation between multiple computing nodes in training [2]. Federated learning, a special case of distributed learning, is a framework where devices keep their raw training data and collaborate by sharing local updates of models [3].

Communication is still a big issue, especially in situations where training involves many rounds of model updates transmission over limited bandwidth channels [3]. The one-shot averaging framework that we consider involves local fitting of models and transmission of their parameters to the server only once [2], [4]. It eliminates the need for many communication rounds, but aggregation of local models can suffer from loss of statistical efficiency compared to joint fitting of a model on all data.

We consider m workers with n observations each and a server with an independent dataset of size p, that is, there are $N = m n + p { \mathrm { ~ i . i . d } } .$ . observations in total. In addition to aggregation of models fitted by the workers, the server fits its own estimator based on its $p$ observations. The benchmark is a hypothetical pooled-data ERM that performs fitting on all N observations.

Key research question. Is it possible for the independent server data to help one-shot weighted averaging achieve the MSE order of the pooled-data, and what size should the server data be?

## A. Related work

Zhang et al. [2] analyze averaging independently fitted worker M-estimators, without an independent server estimator. Their bound has order $\mathcal { O } ( ( m n ) ^ { - 1 } + n ^ { - 2 } )$ , guaranteeing the pooled-data MSE order for $m = \mathcal { O } ( n )$ when dimension and regularity constants are fixed. They also consider a bootstrap subsampling-based bias correction which improves on simple averaging achieving MSE of order $\mathcal { O } ( ( m n ) ^ { - 1 } + n ^ { - 3 } )$ This approach is more robust to parallelization, achieving the pooled-data MSE order for $m = \mathcal { O } ( n ^ { 2 } )$

Rosenblatt and Nadler [4] establish first-order equivalence between worker averaging and pooled estimation as local sample size grows with both dimension and the number of workers fixed. This does not cover arbitrary growth of m with n.

Armacki et al. [5] study one-shot learning with heterogeneous data by clustering local models and averaging within clusters. We instead study i.i.d. data with an independent server dataset.

Dobriban and Sheng [6] consider weighted one-shot ridge regression in a high-dimensional random effects model. The weights which minimize their error may sum to a value greater than one due to shrinkage. Our weights are nonnegative and sum to one, and we also examine the sample size of the independent server data.

Trusted server information has already been explored in robust learning. FLTrust [7] evaluates and normalizes client updates based on the server dataset, while Zeno [8] uses the server-side loss estimate to rank the gradients and Fed-Greed [9] uses the server-side loss to select the client models. In each of these approaches, the issue of faulty or adversarial clients is handled using iterative training. We instead use server data to fit an estimator included in a one-shot average, under i.i.d. sampling and without adversarial clients.

## B. Contributions

The main contributions are:

• We derive the MSE order of sample-size weighting, which assigns weight $n / N$ to each worker estimator and $p / N$ to the server estimator. We establish necessary and sufficient conditions for attaining the pooled-data MSE order, characterizing the required server sample size.

• We identify regimes where server-only estimation attains the pooled-data MSE order while sample-size weighting does not, showing that incorporating worker models can worsen the MSE order.

• We provide analytical examples distinguishing zero bias from bias of orders $n ^ { - 1 }$ and $n ^ { - 2 }$ , and compare aggregation strategies across sample-size regimes through synthetic and CIFAR-10 experiments.

Paper organization. Section II describes the model and the assumptions. The results are presented in Section III. Sections IV and V give analytical and numerical illustrations. The appendix contains the proofs that are not included in the main text. The rest of this section introduces the notation used in this paper.

Notation. Notations $\| \ \cdot \ \| _ { 2 }$ and $\| { \bf \nabla } \cdot { \bf \nabla } \| _ { F }$ stand for Euclidean/spectral and Frobenius norms; $\langle \cdot , \cdot \rangle$ is the Euclidean inner product. Notation $A \succeq B$ is used for positive semidefiniteness of matrix $A - B ; A \succ 0$ is positive definiteness of $A ; ~ I _ { d }$ is the $d \times d$ identity matrix. Notations $\nabla , \nabla ^ { 2 }$ , Cov, and int stand for gradient, Hessian, covariance, and interior, respectively. For two nonnegative sequences $\{ a _ { n } \} _ { n \in \mathbb { N } }$ and $\{ b _ { n } \} _ { n \in \mathbb { N } } ,$ notation $a _ { n } = \mathcal { O } ( b _ { n } ) ( a _ { n } = \Omega ( b _ { n } ) )$ means that there exists a constant $C _ { 1 } > 0$ and $n _ { 1 } ~ \in$ N such that $a _ { n } \leq C _ { 1 } b _ { n }$ $( a _ { n } \geq C _ { 1 } b _ { n } )$ holds for all $n \geq n _ { 1 }$ . The relations $a _ { n } = o ( b _ { n } )$ and $a _ { n } = \omega ( b _ { n } )$ are defined by the requirement that the above inequalities hold for any $C _ { 1 } > 0$ , with $n _ { 1 }$ depending on $C _ { 1 }$ We define $\Theta ( b _ { n } ) : = \mathcal { O } ( b _ { n } ) \cap \Omega ( b _ { n } )$

## II. PROBLEM FORMULATION AND ASSUMPTIONS

Let $X \sim P$ and let $f ( \theta ; x )$ be convex in $\theta \in \Theta \subseteq \mathbb { R } ^ { d }$ , with $\mathbb { E } | f ( \theta ; X ) | < \infty$ . We assume that the population risk has a unique minimizer:

$$
\theta ^ { * } = \arg \operatorname* { m i n } _ { \theta \in \Theta } F _ { 0 } ( \theta ) , \qquad F _ { 0 } ( \theta ) = \mathbb { E } [ f ( \theta ; X ) ] .\tag{1}
$$

An M-estimator $\widehat { \theta } _ { k }$ is obtained by empirical risk minimization (ERM), which minimizes the average loss over k independent observations. Each worker computes:

$$
\theta _ { i } \in \arg \operatorname* { m i n } _ { \theta \in \Theta } \frac { 1 } { n } \sum _ { j = 1 } ^ { n } f ( \theta ; X _ { i j } ) , \qquad i = 1 , \ldots , m .\tag{2}
$$

If $p > 0$ , the server computes:

$$
\theta _ { 0 } \in \arg \operatorname* { m i n } _ { \theta \in \Theta } \frac { 1 } { p } \sum _ { j = 1 } ^ { p } f ( \theta ; X _ { 0 j } ) .\tag{3}
$$

When $p = 0$ , the server estimator is omitted. All the data points are independent samples from distribution $P .$ Since $f ( \cdot ; x )$ and Θ are convex (see Assumption 1 ahead), each node has to solve a convex optimization problem based on its data points alone, and hence any local minimum is a global minimum.

Each worker sends its parameter vector once. The server computes its own estimator and forms

$$
\bar { \theta } = c _ { 1 } \sum _ { i = 1 } ^ { m } \theta _ { i } + c _ { 0 } \theta _ { 0 } , \quad c _ { 0 } , c _ { 1 } \geq 0 , \quad m c _ { 1 } + c _ { 0 } = 1 .\tag{4}
$$

Weights are deterministic and may depend on $m , n , p .$ We compare two choices:

TABLE I  
AVERAGING STRATEGIES, WHERE $N = m n + p .$
<table><tr><td rowspan=1 colspan=1>Strategy</td><td rowspan=1 colspan=1>c0</td><td rowspan=1 colspan=1> $c _ { 1 }$ </td></tr><tr><td rowspan=1 colspan=1>Server only $( p > 0 )$ </td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>Sample-size weighting</td><td rowspan=1 colspan=1> $\overline { { p / N } }$ </td><td rowspan=1 colspan=1> $\overline { { n / N } }$ </td></tr></table>

The server forms a weighted average of the worker estimators and, when $p > 0 ,$ , its own estimator (Figure 1).

![](images/9885237fcc1a9b72a4b83ad42f572c3bd50bc59a5cbfc60af8f20fb6cead3c8e.jpg)  
Fig. 1. The workers send their local estimates to the server, which combines them with its own estimate to obtain θ.

The hypothetical pooled estimator has access to every raw observation:

$$
\hat { \theta } _ { \mathrm { p o o l } } \in \arg \operatorname* { m i n } _ { \theta \in \Theta } \frac { 1 } { N } \left( \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { n } f ( \theta ; X _ { i j } ) + \sum _ { j = 1 } ^ { p } f ( \theta ; X _ { 0 j } ) \right) ,\tag{5}
$$

The MSE of $\hat { \theta } _ { \mathrm { p o o l } }$ with respect to $\theta ^ { * }$ under Assumptions 1– 3 is of order $\bar { \mathcal { O } } ( N ^ { - 1 } )$ [2, Lemma 8]. This pooled-data rate defines our benchmark.

Definition 1 (Order-optimality). A strategy with estimator $\bar { \theta }$ is order-optimal $i f \colon$

$$
\mathbb { E } \Vert \bar { \theta } - \theta ^ { * } \Vert _ { 2 } ^ { 2 } = \mathcal { O } \bigg ( \frac { 1 } { N } \bigg ) .
$$

Therefore, order optimality is equivalent to matching the MSE rate of the pooled data. As an example, both $2 / N$ and $1 0 / N$ are order-optimal, regardless of whether their constant terms match. This definition does not imply that the weights must be such that the MSE is minimized for any given sample size or that they are optimal among all estimators.

We make the following assumptions:

Assumption 1 (Parameters). The set $\Theta \subset \mathbb { R } ^ { d }$ is compact and convex, with $\theta ^ { \ast } \in$ int Θ and radius $R = \operatorname* { m a x } _ { \theta \in \Theta } \| \theta - \theta ^ { * } \| _ { 2 } .$

Assumption 2 (Local strong convexity). The population risk is twice differentiable and $\nabla ^ { 2 } F _ { 0 } ( \theta ^ { * } ) \succeq \lambda I _ { d }$ for some $\lambda > 0$

Assumption 3 (Smoothness). Let $U = \left\{ \theta : \| \theta - \theta ^ { * } \| _ { 2 } \leq \rho \right\} \subseteq$ Θ for some $\rho > 0$ . First and second derivatives of f exist on U, and finite constants $G , H$ satisfy

$$
\begin{array} { r } { \mathbb { E } \| \nabla f ( \theta ; X ) \| _ { 2 } ^ { 8 } \le G ^ { 8 } , } \\ { \mathbb { E } \| \nabla ^ { 2 } f ( \theta ; X ) - \nabla ^ { 2 } F _ { 0 } ( \theta ) \| _ { 2 } ^ { 8 } \le H ^ { 8 } } \end{array}
$$

for all $\theta \in U$ . For every x and $\theta , \theta ^ { \prime } \in U$

$$
\begin{array} { r } { \| \nabla ^ { 2 } f ( \theta ^ { \prime } ; x ) - \nabla ^ { 2 } f ( \theta ; x ) \| _ { 2 } \leq L ( x ) \| \theta ^ { \prime } - \theta \| _ { 2 } . } \end{array}
$$

For some finite L, $\begin{array} { r c l } { \mathbb { E } [ L ( X ) ^ { 8 } ] } & { \leq } & { L ^ { 8 } } \end{array}$ and $\mathbb { E } [ ( L ( X ) ~ -$ $\mathbb { E } L ( X ) ) ^ { 8 } ] \leq L ^ { 8 }$

Assumption 4 (Gradient covariance). The covariance matrix $K = \operatorname { C o v } ( \nabla f ( \theta ^ { * } ; X ) )$ is finite and positive definite.

Assumption 5 (Nonzero asymptotic bias). For the k-sample empirical risk minimizer $\widehat { \theta } _ { k }$ defined above, there exist $\alpha \geq 1$ and a constant vector $b \neq 0$ such that:

$$
\operatorname* { l i m } _ { k \to \infty } k ^ { \alpha } \mathbb { E } [ \widehat { \theta } _ { k } - \theta ^ { * } ] = b .
$$

Assumptions 1–3 are the regularity conditions used in Zhang et al. [2]; Assumption 4 does not allow degenerate gradient noise.

Assumption 5 restricts our consideration to the cases where the local estimators have a nonzero leading bias term of order $k ^ { - \alpha }$ , with $\alpha \geq 1 . \mathrm { ~ } \mathrm { S o } .$ , it excludes unbiased estimators. Relaxing the requirement $\alpha \geq 1$ would not create new admissible models, because the bias bound in the proof of Theorem 1 of Zhang et al. [2, Appendix B.0.2], under Assumptions 1–3, gives

$$
\left. \mathbb { E } [ \hat { \theta } _ { k } - \theta ^ { * } ] \right. _ { 2 } = O ( k ^ { - 1 } ) .
$$

Hence, for any fixed $\alpha < 1$ , it follows that

$$
\begin{array} { r } { k ^ { \alpha } \left\| \mathbb { E } [ \widehat { \theta } _ { k } - \theta ^ { * } ] \right\| _ { 2 } = O ( k ^ { \alpha - 1 } ) \longrightarrow 0 . } \end{array}
$$

This contradicts the requirement $b \neq 0 .$ . That is why it is enough to keep $\alpha \geq 1$ in Assumption 5. Section IV presents examples for this assumption with $\alpha = 1$ and $\alpha = 2 ,$ , as well as an example when this assumption does not hold showing why the nonzero-bias condition is needed for the necessity results for sample-size weighting.

In order to study the impact of data partitioning alone, we fix the distribution, loss function, dimensionality, and the constants in the assumptions. All asymptotic statements take $n  \infty$ , with integer sequences $m \ = \ m ( n ) \ \geq \ 2$ and $p = p ( n ) \geq 0$ . We consider stable scaling regimes in which

$$
p , \qquad { \frac { m } { n ^ { 2 \alpha - 1 } } } , \qquad { \frac { p } { m n } } , \qquad { \frac { p n ^ { 2 \alpha - 2 } } { m ^ { 2 } } }
$$

each have a limit in $[ 0 , + \infty ]$ . The presence of such a condition helps to classify asymptotic regimes by avoiding oscillatory scaling; all our results are stated under this assumption.

The convergence of these sequences guarantees that the system is in some well-defined asymptotic regime. For example, convergence of $m / n ^ { 2 \alpha - 1 }$ separates the cases $m = \mathcal { O } ( n ^ { 2 \alpha - 1 } )$ and $m \bar { = } \omega ( n ^ { 2 \alpha - 1 } )$ . Without this restriction, a sequence may alternate between regimes and require separate conclusions along different subsequences. Growth does not need to be monotone; it is allowed to have fluctuations, as long as the four sequences above keep converging to their respective limits.

## III. RESULTS

We first characterize sample-size weighting, then compare it with server-only estimation and explain the bias mechanism.

## A. Sample-size weighting

The following theorem applies to the local estimators in (2) and the average in (4). Recall that order-optimality (Definition 1) means achieving MSE $\mathcal { O } ( N ^ { - 1 } )$

Theorem 1 (Sample-weighted strategy). Under Assumptions 1–5, let $c _ { 0 } = p / N$ and $c _ { 1 } = n / N$ . Then

$$
\mathrm { M S E } ( \bar { \theta } ) = \Theta \bigg ( \frac { 1 } { N } + \frac { m ^ { 2 } } { N ^ { 2 } n ^ { 2 \alpha - 2 } } \bigg ) .\tag{6}
$$

Sample-size weighting is order-optimal if and only if $m ^ { 2 } / n ^ { 2 \alpha - 2 } = \mathcal { O } ( \breve { N } )$

Corollary 1 (Server sample-size requirement). Under Assumptions 1–5:

1) If $m \ = \ O ( n ^ { 2 \alpha - 1 } )$ , sample-size weighting is orderoptimal for every $p \geq 0$

2) If $m \ = \ \omega ( n ^ { 2 \alpha - 1 } )$ , it is order-optimal if and only if $p = \Omega ( n m \cdot m / n ^ { 2 \alpha - 1 } ) .$

Corollary 2 (Server-only versus sample-size weighting). Under Assumptions 1–5, if

$$
m = \omega ( n ^ { 2 \alpha - 1 } ) , \quad p = \Omega ( n m ) , \quad p = o ( n m \cdot m / n ^ { 2 \alpha - 1 } ) ,
$$

server-only estimation is order-optimal, whereas sample-size weighting is not.

B. Implications for weighted averaging

Rewriting (6) gives:

$$
\mathrm { M S E } ( \bar { \theta } ) = \Theta \left( \frac { 1 } { N } + \left( \frac { m n } { N } \right) ^ { 2 } \frac { 1 } { n ^ { 2 \alpha } } \right) .
$$

The factor $m n / N$ is the total worker weight. More server data reduce the workers’ bias contribution, but this reduction must keep pace with the improving $N ^ { - 1 }$ benchmark. Table II summarizes the two strategies.

When $m = \omega ( n ^ { 2 \alpha - 1 } )$ , the required server sample-size scale for sample-size weighting is larger than that for server-only estimation by the diverging factor $m / n ^ { 2 \alpha - 1 }$ . For example, if $\begin{array} { r c l } { p } & { = } & { \Theta ( m n ) } \end{array}$ , the server stores a constant fraction of all samples, which makes the server-only estimation orderoptimal. In contrast, sample-size weighting provides a nonvanishing total weight to the worker estimators. This leads to the contribution of the common local bias into the squared error of the higher order than $N ^ { - 1 }$ , thus making it impossible for order-optimality.

TABLE II  
NECESSARY AND SUFFICIENT CONDITIONS FOR ORDER-OPTIMALITY, UNDER ASSUMPTIONS 1–5.
<table><tr><td rowspan=1 colspan=1>Strategy</td><td rowspan=1 colspan=1> $\overline { { m = \mathcal { O } ( n ^ { 2 \alpha - 1 } ) } }$ </td><td rowspan=1 colspan=1> $m = \omega ( n ^ { 2 \alpha - 1 } )$ </td></tr><tr><td rowspan=1 colspan=1>Server only</td><td rowspan=1 colspan=1> $p = \Omega ( m n )$ </td><td rowspan=1 colspan=1> $\overline { { p = \Omega ( m n ) } }$ </td></tr><tr><td rowspan=1 colspan=1>Sample-size weights</td><td rowspan=1 colspan=1> $\mathrm { A n y } \ p \geq 0$ </td><td rowspan=1 colspan=1> $\begin{array} { r } { \overline { { p = \Omega \big ( n m \cdot \frac { m } { n ^ { 2 \alpha - 1 } } \big ) } } } \end{array}$ </td></tr></table>

The above statements apply to the averaging procedures considered in this paper; other one-shot aggregation procedures could be more successful. Having several rounds of communication, it is possible, for example, to implement FedAvg [3] under certain convergence assumptions in order to approximate the hypothetical ERM procedure using the information provided by both workers and the server. Therefore, the above results describe an inefficiency of sample-size weighting in a one-shot framework without claiming the ineffectiveness of federated learning.

Proofs and further details are given in Appendices A and B.

## IV. INSIGHTS THROUGH ANALYTICAL EXAMPLES

These scalar examples permit explicit calculations of local estimators and their bias. Although simple, they clarify the role of Assumption 5 in the results of Section III and show how the bias exponent determines the worker and server samplesize requirements for order-optimality. Table III records the assumptions they satisfy.

TABLE III  
ASSUMPTIONS SATISFIED BY THE ANALYTICAL EXAMPLES.
<table><tr><td rowspan=1 colspan=1>Setting</td><td rowspan=1 colspan=1>Assumptions1-4</td><td rowspan=1 colspan=1>Assumption5</td><td rowspan=1 colspan=1>α</td></tr><tr><td rowspan=1 colspan=1>Bernoulli sample mean withsquared loss</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Misspecified least squares</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>Squared loss with a quarticpenalty</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>2</td></tr></table>

## A. Bernoulli sample mean with squared loss

Let X ∼ Bernoulli(q), $0 ~ < ~ q ~ < ~ 1$ , and $f ( \theta ; X ) ~ =$ $( \theta - X ) ^ { 2 } / 2$ on $\Theta = [ - 1 , 2 ]$ . The population minimizer is $\theta ^ { * } = q$ and the empirical-risk minimizer is $\hat { \theta } _ { n } = \bar { X } _ { n }$ . It is unbiased, and sample-size weighting equals pooled ERM exactly, with MSE $q ( 1 - q ) / N$ for every number of workers. Thus Assumptions 1–4 do not imply Assumption 5. Corollary 1 does not apply to this example: without local bias, samplesize weighting can remain order-optimal for arbitrarily many workers, regardless of the server sample size.

## B. Misspecified least squares

Let Z equal 1 or 2 with equal probability, $Y = Z ^ { 3 }$ , and $f ( \theta ; ( Z , Y ) ) ~ = ~ ( \theta Z - Y ) ^ { 2 } / 2$ on $\Theta = [ 0 , 5 ]$ . No single θ represents the conditional mean $Z ^ { 3 }$ at both values of $Z .$ The minimizers and bias are

$$
\theta ^ { * } = \frac { 1 7 } { 5 } , \qquad \hat { \theta } _ { n } = \frac { \sum _ { j = 1 } ^ { n } Z _ { j } ^ { 4 } } { \sum _ { j = 1 } ^ { n } Z _ { j } ^ { 2 } } ,
$$

$$
\mathbb { E } [ \hat { \theta } _ { n } - \theta ^ { * } ] = - \frac { 7 2 } { 1 2 5 n } + \mathcal { O } ( n ^ { - 3 / 2 } ) .
$$

Here $\alpha = 1$ , so Corollary 1 gives a critical worker scale of n for sample-size weighting and requires $p = \Omega ( m ^ { 2 } )$ when $m = \omega ( n )$ . In particular, $m = \omega ( n )$ and $p = \Theta ( m n )$ yield the gap identified in Section III: server-only estimation is orderoptimal, whereas sample-size weighting is not. This example shows that even a squared-loss problem can exhibit the serveronly advantage because its local estimators are biased relative to the population minimizer.

## C. Squared loss with a quartic penalty

Let $X \in [ - 1 , 1 ]$ have zero mean and nonzero third moment, and set

$$
\begin{array} { r } { f ( \theta ; X ) = \frac 1 2 ( \theta - X ) ^ { 2 } + \frac 1 4 \theta ^ { 4 } , \qquad \Theta = [ - 1 , 1 ] . } \end{array}
$$

Then $\theta ^ { * } = 0 , \hat { \theta } _ { n } + \hat { \theta } _ { n } ^ { 3 } = \bar { X } _ { n } .$ , and

$$
\mathbb { E } [ \hat { \theta } _ { n } - \theta ^ { * } ] = - \frac { \mathbb { E } [ X ^ { 3 } ] } { n ^ { 2 } } + \mathcal { O } ( n ^ { - 5 / 2 } ) .
$$

A concrete choice is $\mathbb { P } ( X = - 1 / 3 ) = 2 / 3$ and $\mathbb { P } ( X = 2 / 3 ) =$ $1 / 3 ,$ , giving $b = - 2 / 2 7$

Here $\alpha = 2 .$ , so Corollary 1 guarantees order-optimality of sample-size weighting for any $p \geq 0$ when $m = \mathcal { O } ( n ^ { 3 } )$ . When $m = \omega ( n ^ { 3 } )$ , it is order-optimal if and only if $p = \dot { \Omega } ( \dot { m } ^ { 2 } / n ^ { 2 } )$ Compared with the previous example, faster bias decay raises the critical worker scale from n $\tan ^ { 3 }$ , permitting more workers without an additional server sample-size requirement.

The calculations and assumption checks are in Appendix C.

## V. NUMERICAL EXPERIMENTS

We compare sample-size weighting, server-only estimation, and pooled fitting across sample-size regimes, focusing on aggregation rather than absolute classification accuracy.

## A. Synthetic examples: bias and server sample size

We use the models of Section IV, with $q ~ = ~ 1 / 3$ for the Bernoulli mean and the stated quartic distribution. For $n ~ \in ~ \{ 8 , 1 6 , 3 2 , 6 4 , 1 2 8 , 2 5 6 \}$ , local moments are computed from binomial statistics, omitting tail probability at most $1 0 ^ { - 1 8 }$ . Independence gives the aggregation MSE. These are numerical expectations, not Monte Carlo estimates. Figure 2 reports N MSE, which remains bounded for order-optimal strategies as $n \to \infty$

Panels (a)–(c) fix $m = n ^ { 2 } , p = n ^ { 3 }$ . Sample-size weighting equals pooled ERM for the unbiased mean, loses orderoptimality for least squares $( \alpha = 1 )$ , and retains it for the quartic model $( \alpha = 2 )$ . Server-only estimation is order-optimal throughout; the least-squares gap illustrates Corollary 2.

![](images/34f427ad8684da0eb632b75590f268b9dd81e9df6e86554db005c519c25f0842.jpg)  
Fig. 2. Scaled MSE for the analytical examples. Gray lines indicate predicted growth rates. In (a), sample-size weighting and pooled ERM coincide. Vertical scales differ.

For least squares with $m = n ^ { 2 }$ , Theorem 1 predicts scaled MSE of orders $n , n ^ { 1 / 2 }$ , and 1 for $p = n ^ { 3 } , p = \lceil n ^ { 7 / 2 } \rceil$ , and $p = n ^ { 4 }$ (panels (b), (e), (f)). Panel (d), with $m = p = n .$ instead shows order-optimal sample-size weighting and growing server-only scaled MSE.

## B. CIFAR-10: linear and nonlinear classifiers

CIFAR-10 [10] images use unit-norm, 512-dimensional features from a frozen ImageNet-pretrained ResNet-18 [11]. Worker and server data are sampled independently with replacement from the 50,000 training images, with identical draws across methods within each repetition. The 10,000 test images are used only for evaluation.

Ridge uses mean half squared loss on one-hot labels with penalty $1 0 ^ { - 2 } \| W \| _ { F } ^ { 2 } / 2$ and is solved directly. Linear and nonlinear cross-entropy heads use penalty $1 0 ^ { - 4 } \| \theta \| _ { 2 } ^ { 2 } / 2$ and 400 full-batch Adam updates with learning rate 0.01. The nonlinear head is 512→128→10 with a ReLU hidden layer. For each cross-entropy head and repetition, local and pooled fits share the initialization and training budget. Aggregation averages fitted parameters.

Ridge parameter MSE uses the ridge minimizer on the full training pool as reference, distinct from the pooled fit on sampled data. Results average 20 repetitions for ridge and 10 per cross-entropy head; bands are pointwise 95% Student-t confidence intervals for the means.

Increasing workers. With $m n = 3 2 7 6 8 , p = 8 1 9 2 .$ , increasing m from 2 to 512 causes ridge sample-size-weighted MSE to surpass server-only MSE (Figure 3(a)). Nonlinear weighted accuracy falls from 85.59% to 80.66% (panel (c)); at $m = 5 1 2 ,$ server-only accuracy is 2.99 percentage points higher (95% paired confidence interval: [2.56, 3.42]). Linear cross-entropy likewise favors server-only estimation at $m = 5 1 2 \colon 8 3 . 1 4 \%$ versus 80.06% weighted accuracy.

Increasing server data. Panels (b) and (d) fix mn = 8192, $m = 1 2 8 .$ , and vary $p / ( m n )$ from 1/64 to 16. Ridge samplesize weighting initially outperforms server-only estimation, but the ordering reverses as p grows. Nonlinear weighted accuracy rises from 75.53% to 86.38%, approaching pooled accuracy 86.44%.

Doubling training to 800 updates in three repetitions preserves the mean nonlinear server-only advantage at $m = 5 1 2$ without certifying convergence. We establish neither Assumption 5 nor a bias exponent for CIFAR-10. The nonlinear head lies outside the convex theory; optimization and parameter alignment may affect its results.

## VI. CONCLUSION

Under the stated assumptions, local estimation bias determines the important worker scale $n ^ { 2 \alpha - 1 }$ . When $\begin{array} { r l } { m } & { { } = } \end{array}$ $\mathcal { O } ( n ^ { 2 \alpha - 1 } )$ , sample-size weighting achieves the pooled-data MSE order for every server sample size. When m = $\omega ( n ^ { 2 \alpha - 1 } )$ , it does so if and only if $p = \Omega ( n m \cdot m / n ^ { 2 \alpha - 1 } )$ That means that including worker estimators through samplesize weighting can prevent order-optimality even when serveronly estimation achieves it.

Synthetic experiments illustrate the predicted regimes and the role of the bias exponent. CIFAR-10 experiments show related behavior: sample-size weighting can underperform server-only estimation with both linear and nonlinear classifiers. The nonlinear results illustrate the relevance of this comparison beyond convex models, but do not establish an extension of the theoretical guarantees.

![](images/f6fe68024d34fafaf1f088454959ba6bda28a4c65001838d5abc24f228078319.jpg)  
(c) Nonlinear: increasing workers

![](images/c56949ec6de252a9cf459882d6db77f237bfdff8f075d1486a94ed0da20eef99.jpg)

![](images/4c4c637d57c84a68654e3bbae43471dd6c15d6fe5e000be0d9d1d080a7feaa61.jpg)

(d) Nonlinear: increasing server data  
![](images/c43559b115b5e7bd811086c293137264f6d6671aff76b95693f31ac41449b282.jpg)  
Fig. 3. CIFAR-10: ridge parameter MSE (top) and nonlinear-head test accuracy (bottom). Left: mn = 32768, p = 8192, varying m. Right: mn = 8192, m = 128, varying p. Bands show 95% confidence intervals.

These conclusions concern the specified one-shot averaging rules under i.i.d. sampling and the stated regularity and bias assumptions. They do not exclude better aggregation rules or methods using multiple communication rounds. Heterogeneous data and computational costs remain outside the analysis.

## ACKNOWLEDGMENT

This paper gives a subset of content from Pavle Tepavceviˇ c’s´ master thesis, One-shot Distributed Learning with Server-side Information, completed in 2026 at the Faculty of Sciences, University of Novi Sad, under the supervision of professor Dušan Jakovetic. The authors acknowledge funding from the´ Horizon Europe programme through the TURING project under grant agreement No. 101215032.

## REFERENCES

[1] W. Shi, J. Cao, Q. Zhang, Y. Li, and L. Xu, “Edge computing: Vision and challenges,” IEEE Internet of Things Journal, vol. 3, no. 5, pp. 637-646, 2016.

[2] Y. Zhang, J. C. Duchi, and M. J. Wainwright, “Communication-Efficient Algorithms for Statistical Optimization,” Journal of Machine Learning Research, vol. 14, pp. 3321-3363, 2013.

[3] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. Agüera y Arcas, “Communication-Efficient Learning of Deep Networks from Decentralized Data,” in Proc. 20th International Conference on Artificial Intelligence and Statistics (AISTATS), PMLR, vol. 54, pp. 1273-1282, 2017.

[4] J. Rosenblatt and B. Nadler, "On the optimality of averaging in distributed statistical learning," Information and Inference, vol. 5, pp. 379- 404, 2016.

[5] A. Armacki, D. Bajovic, D. Jakoveti ´ c, and S. Kar, “A One-Shot´ Framework for Distributed Clustered Learning in Heterogeneous Environments,” IEEE Transactions on Signal Processing, vol. 72, pp. 636- 651, 2024.

[6] E. Dobriban and Y. Sheng, "WONDER: Weighted One-shot Distributed Ridge Regression in High Dimensions," Journal of Machine Learning Research, vol. 21, no. 66, pp. 1-52, 2020.

[7] X. Cao, M. Fang, J. Liu, and N. Z. Gong, “FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping,” in Proc. Network and Distributed System Security Symposium (NDSS), 2021.

[8] C. Xie, S. Koyejo, and I. Gupta, “Zeno: Distributed Stochastic Gradient Descent with Suspicion-based Fault-tolerance,” in Proc. 36th International Conference on Machine Learning (ICML), PMLR, vol. 97, pp. 6893-6901, 2019.

[9] E. Kritharakis, A. Makris, D. Jakovetic, and K. Tserpes, “FedGreed:´ A Byzantine-Robust Loss-Based Aggregation Method for Federated Learning,” in Proc. 3rd International Conference on Federated Learning Technologies and Applications (FLTA), 2025.

[10] A. Krizhevsky, “Learning Multiple Layers of Features from Tiny Images,” University of Toronto, Tech. Rep., 2009.

[11] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” in Proc. CVPR, pp. 770-778, 2016.

[12] G. B. Folland, Real Analysis: Modern Techniques and Their Applications, 2nd ed., John Wiley & Sons, 1999.

[13] R. Durrett, Probability: Theory and Examples, 5th ed., Cambridge University Press, 2019.

[14] A. W. van der Vaart, Asymptotic Statistics, Cambridge Series in Statistical and Probabilistic Mathematics, Cambridge University Press, 1998.

[15] G. W. Oehlert, "A Note on the Delta Method," The American Statistician, vol. 46, no. 1, pp. 27-29, 1992.

# APPENDIX A MSE AND BIAS BOUNDS FOR LOCAL ESTIMATORS

All statements below use the notation and asymptotic setting of Section II. For n i.i.d. observations, we write:

$$
F _ { n } ( \theta ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } f ( \theta ; X _ { i } ) , \qquad \hat { \theta } _ { n } \in \arg \operatorname* { m i n } _ { \theta \in \Theta } F _ { n } ( \theta ) .
$$

Theorem A.1 (Squared Bias Bounds for M-estimators). Under Assumption 5,

$$
\| \mathbb { E } [ \hat { \theta } _ { n } - \theta ^ { * } ] \| _ { 2 } ^ { 2 } = \Theta ( n ^ { - 2 \alpha } ) .
$$

Proof. From Assumption 5 we have:

$$
\begin{array} { r } { ( \exists b \neq 0 ) \quad n ^ { \alpha } \operatorname { \mathbb { E } } [ \hat { \theta } _ { n } - \theta ^ { * } ]  b \qquad \mathrm { a s ~ } n  \infty . } \end{array}
$$

Hence:

$$
\begin{array} { r } { ( \forall \varepsilon > 0 ) ( \exists N \in \mathbb N ) ( \forall n \in \mathbb Z _ { \ge N } ) \quad \| n ^ { \alpha } \mathbb E \big [ \widehat \theta _ { n } - \theta ^ { * } \big ] - b \| _ { 2 } \le \varepsilon . } \end{array}
$$

$$
\| b \| _ { 2 } - \varepsilon \leq \| n ^ { \alpha } \mathbb { E } \big [ \widehat { \theta } _ { n } - \theta ^ { * } \big ] \| _ { 2 } \leq \| b \| _ { 2 } + \varepsilon .
$$

$$
\begin{array} { r } { \big ( \| b \| _ { 2 } - \varepsilon \big ) n ^ { - \alpha } \leq \big \| \mathbb { E } \big [ \widehat { \theta } _ { n } - \theta ^ { * } \big ] \big \| _ { 2 } \leq \big ( \| b \| _ { 2 } + \varepsilon \big ) n ^ { - \alpha } . } \end{array}
$$

Now let $\varepsilon = \| b \| _ { 2 } / 2$ . From $b \neq 0$ , we have $\| \boldsymbol { b } \| _ { 2 } > 0$

$$
\begin{array} { r } { \left( \| b \| _ { 2 } - \varepsilon \right) ^ { 2 } n ^ { - 2 \alpha } \leq \left\| \mathbb { E } \big [ \hat { \theta } _ { n } - \theta ^ { * } \big ] \right\| _ { 2 } ^ { 2 } \leq \big ( \| b \| _ { 2 } + \varepsilon \big ) ^ { 2 } n ^ { - 2 \alpha } . } \end{array}
$$

For all n big enough,

$$
\frac { \| b \| _ { 2 } ^ { 2 } } { 4 } n ^ { - 2 \alpha } \leq \left\| \mathbb { E } \big [ \hat { \theta } _ { n } - \theta ^ { * } \big ] \right\| _ { 2 } ^ { 2 } \leq \frac { 9 \| b \| _ { 2 } ^ { 2 } } { 4 } n ^ { - 2 \alpha } .
$$

Then by definition of Θ we have:

$$
\left. \mathbb { E } \big [ \hat { \theta } _ { n } - \theta ^ { * } \big ] \right. _ { 2 } ^ { 2 } = \Theta \big ( 1 / n ^ { 2 \alpha } \big ) .
$$

Assumptions 1-3 are the same as Assumptions 1-3 in [2]. That means that plugging in $k = 2$ in Lemma 8 from that paper yields the following theorem:

Theorem A.2 (Upper bound of MSE for M-estimators). Under Assumptions $^ { l - 3 , }$

$$
\mathbb { E } [ \| \hat { \theta } _ { n } - \theta ^ { * } \| _ { 2 } ^ { 2 } ] = \mathcal { O } ( n ^ { - 1 } ) .
$$

To prove Theorem A.3 some well known theorems will be needed. Here, we just state them. Their proofs can be found in the corresponding literature.

Lemma A.1 (Dominated Convergence Theorem for Differentiation). Let $U \subset \mathbb { R } ^ { d }$ be an open set, and let $f : \mathcal { X } \times U  \mathbb { R }$ be a function such that $x \mapsto f ( x , \theta )$ is integrable for each $\theta \in U$ . Suppose that the gradient $\nabla _ { \boldsymbol { \theta } } f ( \boldsymbol { x } , \boldsymbol { \theta } )$ exists for almost all $x \in \mathcal { X }$ and for all $\theta \in U ,$ , and there exists an integrable function $g : \mathcal { X }  \mathbb { R } _ { > 0 } \ ( i . e . , \ \mathbb { E } [ g ( X ) ] < \infty )$ such that $\| \nabla _ { \theta } f ( x , \theta ) \| _ { 2 } \leq g ( x )$ for all $\theta \in U$ . Then, the expectation $\mathbb { E } [ f ( X , \theta ) ]$ is differentiable, and the derivative and expectation can be exchanged [12 multidimensional generalization of Theorem 2.27(b)]:

$$
\nabla _ { \boldsymbol { \theta } } \mathbb { E } [ f ( \boldsymbol { X } , \boldsymbol { \theta } ) ] = \mathbb { E } [ \nabla _ { \boldsymbol { \theta } } f ( \boldsymbol { X } , \boldsymbol { \theta } ) ] .
$$

Lemma A.2 (Jensen’s Inequality). Let X be a random variable and $\varphi : \mathbb { R }  \mathbb { R }$ be a convex function. $I f \mathbb { E } [ | X | ] < \infty$ and $\mathbb { E } [ | \varphi ( X ) | ] < \infty$ , then the following inequality holds [13, Theorem 1.6.2]:

$$
\varphi ( \mathbb { E } [ X ] ) \leq \mathbb { E } [ \varphi ( X ) ] .
$$

Lemma A.3 (Continuous Mapping Theorem). Let $X _ { n }$ be a sequence of random vectors taking values in $\mathbb { R } ^ { k }$ , and let $g : \mathbb { R } ^ { k } $ $\mathbb { R } ^ { m }$ be a measurable function. Let $C _ { g }$ be the set of points where $g$ is continuous. $I f X _ { n } \ { \xrightarrow { p } } X$ (convergence in probability) and $\mathbb { P } ( X \in C _ { q } ) = 1$ , then $g ( X _ { n } ) \ { \overset { p } { \to } } \ g ( X )$ . The identical implication holds for convergence in distribution $( \xrightarrow { d } )$ and almost sure convergence $( { \xrightarrow { a . s . } } ) ~ I l 4 ,$ Theorem 2.3].

Lemma A.4 (Portmanteau’s Lemma). Let $X _ { n }$ and X be random vectors. The sequence $X _ { n }$ converges in distribution to X (denoted $X _ { n } \stackrel { d } { \to } X )$ if and only iffor every bounded and continuous function $f ,$ the following holds [14, Lemma 2.2]:

$$
\operatorname* { l i m } _ { n \to \infty } \mathbb { E } [ f ( X _ { n } ) ] = \mathbb { E } [ f ( X ) ] .
$$

Theorem A.3 (Lower bound of MSE for M-estimators). Under Assumptions $I { - } 4 ,$

$$
\mathbb { E } [ \| \hat { \theta } _ { n } - \theta ^ { * } \| _ { 2 } ^ { 2 } ] = \Omega ( n ^ { - 1 } ) .
$$

Proof. $X _ { 1 } , X _ { 2 } , \ldots , X _ { n }$ are i.i.d. random variables from the data distribution. To prove the statement it is sufficient to lower bound the expected mean squared error $n \mathbb { E } [ \lVert \hat { { \boldsymbol { \theta } } } _ { n } - { \boldsymbol { \theta } } ^ { * } \rVert _ { 2 } ^ { 2 } ]$

From Assumption 1 we have $\theta ^ { \ast } \in$ int Θ. Since $\theta ^ { * }$ minimizes $F _ { 0 } ( \theta ) = \mathbb { E } [ f ( \theta ; X ) ]$ we know $\nabla F _ { 0 } ( \theta ^ { * } ) = 0$ . To apply Lemma A.1, define:

$$
g ( x ) : = \| \nabla f ( \theta ^ { * } ; x ) \| _ { 2 } + \rho \big ( \| \nabla ^ { 2 } f ( \theta ^ { * } ; x ) \| _ { 2 } + \rho L ( x ) \big ) .
$$

For $\theta \in U$ , Assumption 3 gives

$$
\begin{array} { r l } & { \| \nabla ^ { 2 } f ( \theta ; x ) \| _ { 2 } \leq \| \nabla ^ { 2 } f ( \theta ^ { * } ; x ) \| _ { 2 } + \| \nabla ^ { 2 } f ( \theta ; x ) - \nabla ^ { 2 } f ( \theta ^ { * } ; x ) \| _ { 2 } } \\ & { \qquad \leq \| \nabla ^ { 2 } f ( \theta ^ { * } ; x ) \| _ { 2 } + L ( x ) \| \theta - \theta ^ { * } \| _ { 2 } } \\ & { \qquad \leq \| \nabla ^ { 2 } f ( \theta ^ { * } ; x ) \| _ { 2 } + \rho L ( x ) . } \end{array}
$$

Line from $\theta ^ { * }$ to θ is in U. Hence:

$$
\nabla f ( \theta ; x ) - \nabla f ( \theta ^ { * } ; x ) = \int _ { 0 } ^ { 1 } \nabla ^ { 2 } f \big ( \theta ^ { * } + t ( \theta - \theta ^ { * } ) ; x \big ) ( \theta - \theta ^ { * } ) d t .
$$

Which gives:

$$
\begin{array} { r l } & { \| \nabla f ( \theta ; x ) \| _ { 2 } \leq \| \nabla f ( \theta ^ { * } ; x ) \| _ { 2 } + \| \theta - \theta ^ { * } \| _ { 2 } \big ( \| \nabla ^ { 2 } f ( \theta ^ { * } ; x ) \| _ { 2 } + \rho L ( x ) \big ) } \\ & { \qquad \leq \| \nabla f ( \theta ^ { * } ; x ) \| _ { 2 } + \rho \big ( \| \nabla ^ { 2 } f ( \theta ^ { * } ; x ) \| _ { 2 } + \rho L ( x ) \big ) } \\ & { \qquad = g ( x ) . } \end{array}
$$

Moment bounds in Assumption 3 give:

$$
\begin{array} { r } { \mathbb { E } [ g ( X ) ] \leq G + \rho \big ( \| \nabla ^ { 2 } F _ { 0 } ( { \theta ^ { * } } ) \| _ { 2 } + H + \rho L \big ) < \infty . } \end{array}
$$

Hence from Lemma A.1, applied on int $U ,$ , we have:

$$
\boxed { \mathbb { E } [ \nabla f ( \theta ^ { * } ; X ) ] = \nabla F _ { 0 } ( \theta ^ { * } ) = 0 . }\tag{A.1}
$$

The displayed Hessian bound is also integrable:

$$
\begin{array} { r } { \mathbb { E } \big [ \| \nabla ^ { 2 } f ( \theta ^ { * } ; X ) \| _ { 2 } + \rho L ( X ) \big ] \leq \| \nabla ^ { 2 } F _ { 0 } ( \theta ^ { * } ) \| _ { 2 } + H + \rho L < \infty . } \end{array}
$$

For $\theta \in$ int U and $j , k = 1 , \ldots , d ,$

$$
\begin{array} { r l r } {  {  \frac { \partial ^ { 2 } f ( { \boldsymbol { \theta } } ; { \boldsymbol { x } } ) } { \partial \theta _ { k } \partial \theta _ { j } }  \leq \| \nabla ^ { 2 } f ( { \boldsymbol { \theta } } ; { \boldsymbol { x } } ) \| _ { 2 } } } \\ & { } & { \leq \| \nabla ^ { 2 } f ( { \boldsymbol { \theta } } ^ { * } ; { \boldsymbol { x } } ) \| _ { 2 } + \rho L ( { \boldsymbol { x } } ) . } \end{array}
$$

Hence Lemma A.1 applied to each component of $\nabla f$ gives:

$$
\begin{array} { r l r } & { } & { \frac { \partial ^ { 2 } F _ { 0 } ( \boldsymbol { \theta } ) } { \partial \theta _ { k } \partial \theta _ { j } } = \frac { \partial } { \partial \theta _ { k } } \mathbb { E } \biggl [ \frac { \partial f ( \boldsymbol { \theta } ; \boldsymbol { X } ) } { \partial \theta _ { j } } \biggr ] } \\ & { } & { = \mathbb { E } \biggl [ \frac { \partial ^ { 2 } f ( \boldsymbol { \theta } ; \boldsymbol { X } ) } { \partial \theta _ { k } \partial \theta _ { j } } \biggr ] . } \end{array}
$$

Evaluating at $\theta = \theta ^ { * }$ gives

$$
\boxed { \mathbb { E } [ \nabla ^ { 2 } f ( \theta ^ { * } ; X ) ] = \nabla ^ { 2 } F _ { 0 } ( \theta ^ { * } ) . }
$$

Since $\theta ^ { * }$ is strictly in the interior of $\Theta ,$ the distance to the boundary ∂Θ is strictly positive. Let $\delta = \operatorname* { m i n } ( \rho , \operatorname { d i s t } ( \theta ^ { * } , \partial \Theta ) ) > 0$ where $\rho$ is the radius from Assumption 3. Let $V = \{ \theta \in \Theta \mid \| \theta - \theta ^ { * } \| _ { 2 } \leq \delta / 2 \}$ . Then, $V \subset$ int Θ and $V \subset U$

We partition the parameter space into two events based on the empirical estimator $\ddot { \theta } _ { n } \mathrm { : }$

• Case $\mathbf { A } \colon { \hat { \theta } } _ { n } \not \in V$ . Here:

$$
\lVert \hat { { \boldsymbol { \theta } } } _ { n } - { \boldsymbol { \theta } } ^ { * } \rVert _ { 2 } > \frac { \delta } { 2 } .
$$

• Case B: $\hat { \theta } _ { n } \in V$ . Because $V \subset$ int $\Theta , \hat { \theta } _ { n }$ is in the interior of the parameter space. Since $F _ { n }$ is differentiable and $\hat { \theta } _ { n }$ minimizes $F _ { n }$ , we have $\nabla F _ { n } ( \widehat { \theta } _ { n } ) = 0$

Applying Taylor’s Theorem with integral remainder to the vector-valued function $\nabla F _ { n }$ , we get:

$$
0 = \nabla F _ { n } ( { \hat { \theta } } _ { n } ) = \nabla F _ { n } ( \theta ^ { * } ) + \left( \int _ { 0 } ^ { 1 } \nabla ^ { 2 } F _ { n } ( \theta ^ { * } + t ( { \hat { \theta } } _ { n } - \theta ^ { * } ) ) d t \right) ( { \hat { \theta } } _ { n } - \theta ^ { * } ) .
$$

Let $\begin{array} { r } { H _ { n } = \int _ { 0 } ^ { 1 } \nabla ^ { 2 } F _ { n } ( \theta ^ { * } + t ( \hat { \theta } _ { n } - \theta ^ { * } ) ) d t } \end{array}$ . Rearranging gives us $\nabla F _ { n } ( \theta ^ { * } ) = - H _ { n } ( { \hat { \theta } } _ { n } - \theta ^ { * } )$ . Applying the sub-multiplicative property of the spectral norm we get:

$$
\lVert \nabla F _ { n } ( { \boldsymbol { \theta } } ^ { * } ) \rVert _ { 2 } \leq \lVert H _ { n } \rVert _ { 2 } \lVert { \hat { \boldsymbol { \theta } } } _ { n } - { \boldsymbol { \theta } } ^ { * } \rVert _ { 2 } .\tag{A.2}
$$

To upper bound $\Vert H _ { n } \Vert _ { 2 }$ , we use the triangle inequality:

$$
\begin{array} { l } { \displaystyle \| H _ { n } \| _ { 2 } \leq \| \nabla ^ { 2 } F _ { 0 } ( { \boldsymbol { \theta } } ^ { * } ) \| _ { 2 } + \| \nabla ^ { 2 } F _ { n } ( { \boldsymbol { \theta } } ^ { * } ) - \nabla ^ { 2 } F _ { 0 } ( { \boldsymbol { \theta } } ^ { * } ) \| _ { 2 } } \\ { \displaystyle \qquad + \int _ { 0 } ^ { 1 } \| \nabla ^ { 2 } F _ { n } ( { \boldsymbol { \theta } } ^ { * } + t ( \hat { { \boldsymbol { \theta } } } _ { n } - { \boldsymbol { \theta } } ^ { * } ) ) - \nabla ^ { 2 } F _ { n } ( { \boldsymbol { \theta } } ^ { * } ) \| _ { 2 } d t . } \end{array}
$$

Since $\hat { \theta } _ { n } \in V \subset U$ , the line segment connecting $\theta ^ { * }$ and $\hat { \theta } _ { n }$ lies entirely in U. By the Lipschitz property of the Hessian (Assumption 3), for any $\theta \in U$ , we have $\begin{array} { r } { \| \nabla ^ { 2 } F _ { n } ( \boldsymbol { \theta } ) - \nabla ^ { 2 } F _ { n } ( \boldsymbol { \theta } ^ { * } ) \| _ { 2 } \leq \frac { 1 } { n } \sum _ { i = 1 } ^ { \bar { n } } L ( X _ { i } ) \| \boldsymbol { \theta } - \boldsymbol { \theta } ^ { * } \| _ { 2 } } \end{array}$ . Let $\begin{array} { r } { L _ { n } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } L ( X _ { i } ) } \end{array}$ . The integral term is bounded by $L _ { n } \| \hat { \theta } _ { n } - \theta ^ { * } \| _ { 2 } \leq L _ { n } \rho .$

Let $M _ { 0 } = \lVert \nabla ^ { 2 } F _ { 0 } ( \theta ^ { * } ) \rVert _ { 2 }$ . By Assumption 2 we know that $F _ { 0 }$ is twice differentiable, hence $\nabla ^ { 2 } F _ { 0 } ( \theta ^ { * } )$ is a finite $d \times d$ real matrix, hence $M _ { 0 } < \infty$ . Assumption 2 also gives $M _ { 0 } \geq \lambda > 0$ . Let $D _ { n } = \| \nabla ^ { 2 } F _ { n } ( \theta ^ { * } ) - \nabla ^ { 2 } F _ { 0 } ( \theta ^ { * } ) \| _ { 2 }$ . We define the random sequence:

$$
Y _ { n } = M _ { 0 } + D _ { n } + L _ { n } \rho .
$$

In Case B, $\| H _ { n } \| _ { 2 } \leq Y _ { n }$ , which by inequality A.2 implie $\begin{array} { r } { \| \hat { { \boldsymbol \theta } } _ { n } - { \boldsymbol \theta } ^ { * } \| _ { 2 } \geq \frac { \| \nabla F _ { n } ( { \boldsymbol \theta } ^ { * } ) \| _ { 2 } } { Y _ { n } } } \end{array}$

Combining Cases A and B gives a bound that holds for every n:

$$
\| \hat { { \boldsymbol \theta } } _ { n } - { \boldsymbol \theta } ^ { * } \| _ { 2 } \geq \operatorname* { m i n } \left( \frac { \delta } { 2 } , \frac { \| \nabla F _ { n } ( { \boldsymbol \theta } ^ { * } ) \| _ { 2 } } { Y _ { n } } \right)\tag{A.3}
$$

$$
\implies n \| \hat { { \boldsymbol \theta } } _ { n } - { \boldsymbol \theta } ^ { * } \| _ { 2 } ^ { 2 } \geq \operatorname* { m i n } \left( n \frac { \delta ^ { 2 } } { 4 } , \frac { \| \sqrt { n } \nabla F _ { n } ( { \boldsymbol \theta } ^ { * } ) \| _ { 2 } ^ { 2 } } { Y _ { n } ^ { 2 } } \right) .\tag{A.4}
$$

Let $W _ { n } \ = \ \sqrt { n } \nabla F _ { n } ( \theta ^ { * } )$ . Because $\mathbb { E } [ \nabla f ( \theta ^ { * } ; X ) ] = 0$ and its covariance matrix $K \succ 0$ is finite (Assumption 4), the multivariate Central Limit Theorem states that:

$$
W _ { n } \stackrel { d } { \to } W \sim { \mathcal { N } } ( 0 , K ) .
$$

By the Weak Law of Large Numbers, whose required finite first absolute moment condition follows from Assumption 3 together with Jensen’s inequality, since for each relevant quantity Y,

$$
\mathbb { E } | Y | \le \left( \mathbb { E } | Y | ^ { 8 } \right) ^ { 1 / 8 } < \infty
$$

whenever Assumption 3 gives $\mathbb { E } | Y | ^ { 8 } < \infty$ we have:

$$
D _ { n } \stackrel { P } { \to } 0 \qquad \mathrm { a n d } \qquad L _ { n } \stackrel { P } { \to } \mathbb { E } [ L ( X ) ] .
$$

Therefore, by the Continuous mapping theorem:

$$
Y _ { n } \stackrel { P } { \longrightarrow } M _ { 0 } + \mathbb { E } [ L ( X ) ] \rho \equiv y _ { 0 } .
$$

Note that $y _ { 0 } \ge M _ { 0 } \ge \lambda > 0$ by Assumption 2. Thus, by Slutsky’s Theorem:

$$
Z _ { n } \equiv \frac { \| { \cal W } _ { n } \| _ { 2 } ^ { 2 } } { Y _ { n } ^ { 2 } } \stackrel { d } {  } \frac { \| { \cal W } \| _ { 2 } ^ { 2 } } { y _ { 0 } ^ { 2 } } \equiv Z .
$$

For any constant $C > 0$ , there exists an integer N such that for all $n \geq N , n { \frac { \delta ^ { 2 } } { 4 } } \geq C$ . Therefore, for all $n \geq N .$ , the inequality (A.4) implies:

$$
\begin{array} { r } { n \| \hat { \theta } _ { n } - \theta ^ { * } \| _ { 2 } ^ { 2 } \geq \operatorname* { m i n } ( C , Z _ { n } ) . } \end{array}
$$

Taking expectations on both sides preserves the inequality:

$$
\begin{array} { r } { n \mathbb { E } [ \| \hat { \theta } _ { n } - \theta ^ { * } \| _ { 2 } ^ { 2 } ] \ge \mathbb { E } [ \operatorname* { m i n } ( C , Z _ { n } ) ] . } \end{array}
$$

The function $g ( x ) = \operatorname* { m i n } ( C , x )$ is continuous and bounded on $[ 0 , \infty )$ and $Z _ { n } , Z \ge 0 .$ . By the Portmanteau’s Lemma for weak convergence (since $Z _ { n } \stackrel { d } { \to } Z )$ , we have:

$$
\operatorname* { l i m } _ { n \to \infty } \mathbb { E } [ \operatorname* { m i n } ( C , Z _ { n } ) ] = \mathbb { E } [ \operatorname* { m i n } ( C , Z ) ] .
$$

Since $K \succ 0 .$ , the limiting distribution $W \sim \mathcal { N } ( 0 , K )$ is a non-degenerate multivariate Gaussian, which implies $\mathbb { P } ( \| W \| _ { 2 } >$ $0 ) = 1$ . Consequently, $Z > 0$ almost surely, ensuring that:

$$
c \equiv \mathbb { E } [ \operatorname* { m i n } ( C , Z ) ] > 0 .
$$

It follows immediately that:

$$
\operatorname* { l i m } _ { n \to \infty } \operatorname* { i n f } n \mathbb { E } [ \| \hat { \theta } _ { n } - \theta ^ { * } \| _ { 2 } ^ { 2 } ] \geq c > 0 .
$$

By definition, this proves that $\mathbb { E } [ \lVert \hat { { \boldsymbol { \theta } } } _ { n } - { \boldsymbol { \theta } } ^ { * } \rVert _ { 2 } ^ { 2 } ] = \Omega ( 1 / n )$

Combining the bounds of Theorems A.1, A.2, and A.3 we get the following corollary.

Corollary A.1 (MSE and Squared Bias Bounds for M-estimators). Under Assumptions $I { - } 5 ,$

$$
\begin{array} { r } { \mathbb { E } [ \| \hat { { \boldsymbol \theta } } _ { n } - { \boldsymbol \theta } ^ { * } \| _ { 2 } ^ { 2 } ] = \Theta ( n ^ { - 1 } ) , \qquad \| \mathbb { E } [ \hat { { \boldsymbol \theta } } _ { n } - { \boldsymbol \theta } ^ { * } ] \| _ { 2 } ^ { 2 } = \Theta ( n ^ { - 2 \alpha } ) . } \end{array}
$$

## APPENDIX BSAMPLE-SIZE WEIGHTING

Theorem B.1 (Optimality of the Server Estimator). Under Assumptions 1–4, let $p > 0 .$ . Server-only estimation, with $c _ { 0 } = 1$ and $c _ { 1 } = 0 ,$ , is order-optimal if and only if $p = \Omega ( m n )$

Proof. (⇐) Suppose $p = \Omega ( n m )$ . Let $N = n m + p$ be the total amount of data. The parameter θ is considered order-optimal if its MSE is bounded as $\mathbb { E } [ \lVert \overline { { \theta } } - \theta ^ { * } \rVert _ { 2 } ^ { 2 } ] = \mathcal { O } ( \frac { 1 } { N } )$ . We will show this holds for the specified choice of weights.

We choose the weights $c _ { 0 } = 1$ and $c _ { 1 } = 0$ . This choice satisfies the condition $m c _ { 1 } + c _ { 0 } = 1$ . With these weights, the weighted average estimator simplifies to:

$$
\overline { { \theta } } = c _ { 1 } \sum _ { i = 1 } ^ { m } \theta _ { i } + c _ { 0 } \theta _ { 0 } = 0 \cdot \sum _ { i = 1 } ^ { m } \theta _ { i } + 1 \cdot \theta _ { 0 } = \theta _ { 0 }
$$

From Theorem A.2 we have:

$$
\mathbb { E } [ \| \overline { { \theta } } - \theta ^ { * } \| _ { 2 } ^ { 2 } ] = \mathbb { E } [ \| \theta _ { 0 } - \theta ^ { * } \| _ { 2 } ^ { 2 } ] = \mathcal { O } \left( \frac { 1 } { p } \right) .
$$

We are given the assumption $p = \Omega ( n m )$ , therefore the total number of data points $N = n m + p$ is of the same order as $p \mathrm { : }$

$$
\begin{array} { r l r } { n m + p = \mathcal { O } ( p ) } & { { } \mathrm { a n d } \quad p = \mathcal { O } ( n m + p ) } & { \implies \mathcal { O } \left( \frac { 1 } { p } \right) = \mathcal { O } \left( \frac { 1 } { n m + p } \right) } \end{array}
$$

Hence, the MSE of the estimator $\overline { { \theta } }$ with $c _ { 0 } = 1$ and $c _ { 1 } = 0$ matches the order-optimal rate:

$$
\mathbb { E } [ \lVert \overline { { \theta } } - \theta ^ { * } \rVert _ { 2 } ^ { 2 } ] = \mathcal { O } \left( \frac { 1 } { N } \right)
$$

(⇒) Suppose that the server-only estimation is order-optimal, i.e.,

$$
\mathbb { E } [ \| \theta _ { 0 } - \theta ^ { * } \| _ { 2 } ^ { 2 } ] = \mathcal { O } ( N ^ { - 1 } ) .
$$

From the fact that $p$ has a limit we have that either $p \to \infty \ \mathrm { o r } \ p$ is eventually a fixed positive integer.

If $p$ was fixed zero MSE would give us $\theta _ { 0 } = \theta ^ { * }$ a.s. Since $\theta ^ { \ast } \in$ int Θ, empirical optimality would give $\begin{array} { r } { \nabla F _ { p } ( \theta ^ { * } ) = 0 \ \mathrm { a . s . } } \end{array}$ contradicting Assumption 4:

$$
\mathrm { C o v } ( \nabla F _ { p } ( { \boldsymbol { \theta } } ^ { * } ) ) = \frac { K } { p } \succ 0 .
$$

This means that for fixed $p$ the server MSE is a positive constant. Since $N = m n + p  \infty$ we have:

$$
N \mathbb { E } [ \| \theta _ { 0 } - \theta ^ { * } \| _ { 2 } ^ { 2 } ] \longrightarrow \infty ,
$$

which is a contradiction with order-optimality assumption. Hence $p  \infty$

Theorem A.3 gives $\mathbb { E } [ \| \theta _ { 0 } - \theta ^ { * } \| _ { 2 } ^ { 2 } ] = \Omega ( p ^ { - 1 } )$ ). But assumed order-optimality gives $\mathbb { E } [ \| \theta _ { 0 } - \theta ^ { * } \| _ { 2 } ^ { 2 } ] = \mathcal { O } ( N ^ { - 1 } )$ . That means:

$$
{ \frac { 1 } { p } } = { \mathcal { O } } \left( { \frac { 1 } { N } } \right) \implies N = { \mathcal { O } } ( p ) \implies m n = { \mathcal { O } } ( p ) \implies p = \Omega ( m n ) .
$$

□

Proof of Theorem 1. Case $I \colon p  \infty$

$$
\begin{array} { c } { { \displaystyle \mathbb { E } [ \| \bar { \theta } - \theta ^ { * } \| _ { 2 } ^ { 2 } ] = \displaystyle \frac { m n ^ { 2 } } { ( n m + p ) ^ { 2 } } \mathbb { E } [ \| \theta _ { 1 } - \theta ^ { * } \| _ { 2 } ^ { 2 } ] + \displaystyle \frac { p ^ { 2 } } { ( n m + p ) ^ { 2 } } \mathbb { E } [ \| \theta _ { 0 } - \theta ^ { * } \| _ { 2 } ^ { 2 } ] } } \\ { { \displaystyle \phantom { \displaystyle \mathbb { E } [ \| \bar { \theta } - \theta ^ { * } \| _ { 2 } ^ { 2 } ] = } + \displaystyle \frac { m ( m - 1 ) n ^ { 2 } } { ( n m + p ) ^ { 2 } } \| \mathbb { E } [ \theta _ { 1 } - \theta ^ { * } ] \| _ { 2 } ^ { 2 } + \displaystyle \frac { 2 m n p } { ( n m + p ) ^ { 2 } } \left. \mathbb { E } [ \theta _ { 1 } - \theta ^ { * } ] , \mathbb { E } [ \theta _ { 0 } - \theta ^ { * } ] \right. . } } \end{array}
$$

From Corollary A.1 we have:

$$
\frac { m n ^ { 2 } } { ( n m + p ) ^ { 2 } } \mathbb { E } [ \| \theta _ { 1 } - \theta ^ { * } \| _ { 2 } ^ { 2 } ] + \frac { p ^ { 2 } } { ( n m + p ) ^ { 2 } } \mathbb { E } [ \| \theta _ { 0 } - \theta ^ { * } \| _ { 2 } ^ { 2 } ] = \Theta \bigg ( \frac { n m + p } { ( n m + p ) ^ { 2 } } \bigg ) = \Theta \bigg ( \frac { 1 } { n m + p } \bigg ) .
$$

From Theorem A.1,

$$
\frac { m ( m - 1 ) n ^ { 2 } } { ( n m + p ) ^ { 2 } } \| \mathbb { E } [ \theta _ { 1 } - \theta ^ { * } ] \| _ { 2 } ^ { 2 } = \Theta \bigg ( \frac { m ^ { 2 } } { ( n m + p ) ^ { 2 } n ^ { 2 \alpha - 2 } } \bigg ) .
$$

Also,

$$
\left| \frac { 2 m n p } { ( n m + p ) ^ { 2 } } \left. \mathbb { E } [ \theta _ { 1 } - \theta ^ { * } ] , \mathbb { E } [ \theta _ { 0 } - \theta ^ { * } ] \right. \right| \overset { \mathrm { c . s } } { \leq } \frac { 2 m n p } { ( n m + p ) ^ { 2 } } \| \mathbb { E } [ \theta _ { 1 } - \theta ^ { * } ] \| _ { 2 } \| \mathbb { E } [ \theta _ { 0 } - \theta ^ { * } ] \| _ { 2 }
$$

$$
\mathrm { { T h } } \underset { = } { \overset { \mathrm { \scriptsize { R m . } } } { = } } \cup \left( \underbrace { n m } _ { \overset { n m } { \underbrace { n m + p } } } \underbrace { p ^ { 1 - \alpha } } _ { \leq 1 } \frac { 1 } { ( n m + p ) n ^ { \alpha } } \right) \subseteq \mathcal O \left( \frac { 1 } { ( n m + p ) n ^ { \alpha } } \right) \subseteq o \left( \frac { 1 } { n m + p } \right) .
$$

This term may be negative, but its absolute value is $o ( ( n m + p ) ^ { - 1 } )$ so it cannot cancel the positive lower bound from the first two terms. Combining these results we get:

$$
\mathbb { E } [ \| \overline { { \theta } } - \theta ^ { * } \| _ { 2 } ^ { 2 } ] = \Theta \bigg ( \frac { 1 } { n m + p } + \frac { m ^ { 2 } } { ( n m + p ) ^ { 2 } n ^ { 2 \alpha - 2 } } \bigg ) .
$$

Case $2 \colon p \geq 1$ is fixed. By Assumption 1,

$$
\begin{array} { r } { \mathbb { E } [ \| \theta _ { 0 } - \theta ^ { * } \| _ { 2 } ^ { 2 } ] \le R ^ { 2 } , \qquad \| \mathbb { E } [ \theta _ { 0 } - \theta ^ { * } ] \| _ { 2 } \le R . } \end{array}
$$

Since $N = m n + p = \Theta ( m n )$ , the same expansion gives:

$$
\frac { m n ^ { 2 } } { N ^ { 2 } } \mathbb { E } [ \| \theta _ { 1 } - \theta ^ { * } \| _ { 2 } ^ { 2 } ] = \Theta \Big ( \frac { m n } { N ^ { 2 } } \Big ) = \Theta ( N ^ { - 1 } ) ,
$$

$$
\frac { p ^ { 2 } } { N ^ { 2 } } \mathbb { E } [ \| \theta _ { 0 } - \theta ^ { * } \| _ { 2 } ^ { 2 } ] \le \frac { p ^ { 2 } R ^ { 2 } } { N ^ { 2 } } = \mathcal { O } ( N ^ { - 2 } ) ,
$$

$$
\frac { m ( m - 1 ) n ^ { 2 } } { N ^ { 2 } } \| \mathbb { E } [ \theta _ { 1 } - \theta ^ { * } ] \| _ { 2 } ^ { 2 } = \Theta \bigg ( \frac { m ^ { 2 } } { N ^ { 2 } n ^ { 2 \alpha - 2 } } \bigg ) ,
$$

$$
\begin{array} { r } { \displaystyle \Big | \frac { 2 m n p } { N ^ { 2 } } \left. \mathbb { E } [ \theta _ { 1 } - \theta ^ { * } ] , \mathbb { E } [ \theta _ { 0 } - \theta ^ { * } ] \right. \Big | \leq \frac { 2 m n p R } { N ^ { 2 } } \| \mathbb { E } [ \theta _ { 1 } - \theta ^ { * } ] \| _ { 2 } } \\ { = \mathcal { O } \Big ( \frac { 1 } { N n ^ { \alpha } } \Big ) = o ( N ^ { - 1 } ) . } \end{array}
$$

Hence,

$$
\mathbb { E } [ \| \overline { { \theta } } - \theta ^ { * } \| _ { 2 } ^ { 2 } ] = \Theta \bigg ( \frac { 1 } { N } + \frac { m ^ { 2 } } { N ^ { 2 } n ^ { 2 \alpha - 2 } } \bigg ) .
$$

Case 3: $p = 0$ . Then $c _ { 0 } = 0 , c _ { 1 } = 1 / m$ , and $N = m n$ . Independence gives

$$
\begin{array} { r } { \mathbb { E } [ \| \overline { { \theta } } - \theta ^ { * } \| _ { 2 } ^ { 2 } ] = \frac { 1 } { m } \mathbb { E } [ \| \theta _ { 1 } - \theta ^ { * } \| _ { 2 } ^ { 2 } ] + \frac { m - 1 } { m } \| \mathbb { E } [ \theta _ { 1 } - \theta ^ { * } ] \| _ { 2 } ^ { 2 } } \\ { = \Theta \bigg ( \frac { 1 } { m n } + \frac { 1 } { n ^ { 2 \alpha } } \bigg ) = \Theta \bigg ( \frac { 1 } { N } + \frac { m ^ { 2 } } { N ^ { 2 } n ^ { 2 \alpha - 2 } } \bigg ) . } \end{array}
$$

In all three cases, the MSE is $\mathcal { O } ( ( n m + p ) ^ { - 1 } )$ if and only if

$$
{ \frac { m ^ { 2 } } { ( n m + p ) ^ { 2 } n ^ { 2 \alpha - 2 } } } = { \cal O } \bigg ( { \frac { 1 } { n m + p } } \bigg ) \Longleftrightarrow { \frac { m ^ { 2 } } { n ^ { 2 \alpha - 2 } } } = { \cal O } ( n m + p ) .
$$

Proof of Corollary 1. If $m = \mathcal { O } ( n ^ { 2 \alpha - 1 } )$ , then

$$
\frac { m ^ { 2 } } { n ^ { 2 \alpha - 2 } } = \mathcal { O } ( n m ) \implies \frac { m ^ { 2 } } { n ^ { 2 \alpha - 2 } } = \mathcal { O } ( n m + p ) .
$$

Hence by Theorem 1 the first statement is true.

If $m = \omega ( n ^ { 2 \alpha - 1 } )$ , then

$$
\frac { n m } { m ^ { 2 } / n ^ { 2 \alpha - 2 } } = \frac { n ^ { 2 \alpha - 1 } } { m } \longrightarrow 0 .
$$

Hence,

$$
\frac { m ^ { 2 } } { n ^ { 2 \alpha - 2 } } = \mathcal { O } ( n m + p ) \iff \frac { m ^ { 2 } } { n ^ { 2 \alpha - 2 } } = \mathcal { O } ( p ) \iff p = \Omega \left( \frac { m ^ { 2 } } { n ^ { 2 \alpha - 2 } } \right) .
$$

Hence by Theorem 1 the second statement is true.

Proof of Corollary 2.

$p = \Omega ( n m ) \stackrel { \mathrm { T h m . ~ B . 1 } } { \Longrightarrow }$ server-only weighting achieves order-optimal MSE.

From $\begin{array} { r } { p = o ( n m \cdot \frac { m } { n ^ { 2 \alpha - 1 } } ) } \end{array}$ we have $\begin{array} { r } { p \neq \Omega \Big ( n m \cdot \frac { m } { n ^ { 2 \alpha - 1 } } \Big ) } \end{array}$ . Hence:

$$
m = \omega ( n ^ { 2 \alpha - 1 } ) \wedge p \neq \Omega \Big ( n m \cdot \frac { m } { n ^ { 2 \alpha - 1 } } \Big ) \ \stackrel { \mathrm { C o r . } _ { * } } { = } \quad
$$

sample-size weighting does not

achieve order-optimal MSE.

## APPENDIX C ANALYTICAL EXAMPLE CALCULATIONS

A. An unbiased sample mean

For the model in Section IV-A,

$$
\begin{array} { r } { F _ { 0 } ( \theta ) = \frac 1 2 \big ( ( \theta - q ) ^ { 2 } + q ( 1 - q ) \big ) , \qquad F _ { n } ^ { \prime } ( \theta ) = \theta - \bar { X } _ { n } . } \end{array}
$$

Thus $\theta ^ { * } = q$ and $\hat { \theta } _ { n } = \bar { X } _ { n } \in [ 0 , 1 ] \subset \Theta = [ - 1 , 2 ]$ . Since $\mathbb { E } [ \bar { X } _ { n } ] = q$ and $\operatorname { V a r } ( { \bar { X } } _ { n } ) = q ( 1 - q ) / n$ , the estimator is unbiased. Sample-size weighting gives

$$
\overline { { \theta } } = \frac { 1 } { N } \left( \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { n } X _ { i j } + \sum _ { j = 1 } ^ { p } X _ { 0 j } \right) = \hat { \theta } _ { \mathrm { p o o l } } , \qquad \mathbb { E } [ ( \overline { { \theta } } - \theta ^ { * } ) ^ { 2 } ] = \frac { q ( 1 - q ) } { N } .
$$

The server sum is omitted when $p = 0$

Assumption 1 holds because Θ is compact and convex, with q in its interior. The sample Hessian and the population Hessian both equal 1. The gradient $\theta - X$ is bounded on Θ, and the Hessian is constant. This verifies Assumptions 2 and 3. At $\theta ^ { * } = q$

$$
K = \mathbb { E } [ ( q - X ) ^ { 2 } ] = ( 1 - q ) q ^ { 2 } + q ( q - 1 ) ^ { 2 } = q ( 1 - q ) > 0 ,
$$

so Assumption 4 also holds. Assumption 5 fails because the bias is zero.

## B. Misspecified least squares

For the model in Section IV-B,

$$
F _ { 0 } ^ { \prime } ( \theta ) = \mathbb { E } [ Z ^ { 2 } ] \theta - \mathbb { E } [ Z ^ { 4 } ] = \frac { 5 } { 2 } \theta - \frac { 1 7 } { 2 } , \qquad F _ { n } ^ { \prime } ( \theta ) = \frac { 1 } { n } \sum _ { j = 1 } ^ { n } ( \theta Z _ { j } ^ { 2 } - Z _ { j } ^ { 4 } ) .
$$

Consequently,

$$
\theta ^ { * } = \frac { 1 7 } { 5 } , \qquad \hat { \theta } _ { n } = \frac { \sum _ { j = 1 } ^ { n } Z _ { j } ^ { 4 } } { \sum _ { j = 1 } ^ { n } Z _ { j } ^ { 2 } } .
$$

The empirical minimizer is a weighted average of the values $Z _ { j } ^ { 2 }$ , with positive weights proportional to $Z _ { j } ^ { 2 }$ . It therefore lies in $[ 1 , 4 ] \subset \Theta = [ 0 , 5 ]$ , so the constraint is inactive.

The gradient is $\theta Z ^ { 2 } - Z Y$ , and the Hessian is $Z ^ { 2 }$ . The population Hessian equals $5 / 2 > 0$ . The data are bounded, and the sample Hessian does not depend on θ. At $\theta ^ { * } = 1 7 / 5$ , the gradient takes the values $1 2 / 5$ and $- 1 2 / 5$ with equal probability. Its mean is zero, so

$$
K = { \frac { 1 } { 2 } } \left( { \frac { 1 2 } { 5 } } \right) ^ { 2 } + { \frac { 1 } { 2 } } \left( - { \frac { 1 2 } { 5 } } \right) ^ { 2 } = { \frac { 1 4 4 } { 2 5 } } > 0 .
$$

These facts verify Assumptions 1–4.

We calculate the bias by a Taylor expansion. This is the usual method for approximating moments of a smooth function of a sample mean; see Oehlert [15]. Write

$$
\overline { { Z ^ { 2 } } } _ { n } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } Z _ { i } ^ { 2 } .
$$

Since $Z _ { i } ^ { 4 } = 5 Z _ { i } ^ { 2 } - 4$ , the estimator becomes

$$
\hat { \theta } _ { n } = 5 - \frac { 4 } { Z ^ { 2 } { } _ { n } } .
$$

The variables $Z _ { i } ^ { 2 } - 5 / 2$ are independent and take the values $- 3 / 2$ and $3 / 2$ with equal probability. Thus

$$
\mathbb { E } \Big [ \overline { { Z ^ { 2 } } } _ { n } \Big ] = \frac { 5 } { 2 } , \qquad \mathbb { E } \Bigg [ \bigg ( \overline { { Z ^ { 2 } } } _ { n } - \frac { 5 } { 2 } \bigg ) ^ { 2 } \Bigg ] = \frac { 9 } { 4 n } .
$$

In the fourth moment, only products with all four indices equal or with two pairs of equal indices have nonzero expectation.   
Any other product contains an index that appears only once. Independence then gives a factor with expectation zero.

There are n terms with all four indices equal. Each has expectation $( 3 / 2 ) ^ { 4 } = 8 1 / 1 6$ . For two pairs of equal indices, there are $\binom { n } { 2 }$ choices of distinct indices. Each pair can be arranged in the four positions in ${ \binom { 4 } { 2 } } = 6$ ways. For $i \neq j$ , independence gives

$$
\mathbb { E } \left[ \left( Z _ { i } ^ { 2 } - \frac 5 2 \right) ^ { 2 } \left( Z _ { j } ^ { 2 } - \frac 5 2 \right) ^ { 2 } \right] = \mathbb { E } \left[ \left( Z _ { i } ^ { 2 } - \frac 5 2 \right) ^ { 2 } \right] \mathbb { E } \left[ \left( Z _ { j } ^ { 2 } - \frac 5 2 \right) ^ { 2 } \right] = \left( \frac 9 4 \right) ^ { 2 } .
$$

This gives

$$
\begin{array} { r l } { \mathbb { E } \bigg [ \bigg ( \overline { { Z ^ { 2 } } } _ { n } - \frac { 5 } { 2 } \bigg ) ^ { 4 } \bigg ] = \frac { 1 } { n ^ { 4 } } \mathbb { E } \bigg [ \bigg ( \sum _ { i = 1 } ^ { n } \bigg ( Z _ { i } ^ { 2 } - \frac { 5 } { 2 } \bigg ) \bigg ) ^ { 4 } \bigg ] } & { } \\ & { = \frac { 1 } { n ^ { 4 } } \bigg [ \displaystyle \sum _ { i = 1 } ^ { n } \mathbb { E } \bigg [ \bigg ( Z _ { i } ^ { 2 } - \frac { 5 } { 2 } \bigg ) ^ { 4 } \bigg ] } \\ & { \qquad + \ 6 \ \sum _ { i \in \mathcal { S } ( \mathbb { Z } ) } \mathbb { E } \bigg [ \bigg ( Z _ { i } ^ { 2 } - \frac { 5 } { 2 } \bigg ) ^ { 2 } \bigg ( Z _ { j } ^ { 2 } - \frac { 5 } { 2 } \bigg ) ^ { 2 } \bigg ] \bigg ] } \\ & { = \frac { 1 } { n ^ { 4 } } \bigg [ \displaystyle n \frac { 8 1 } { 1 6 } + 6 \bigg ( \int _ { 0 } ^ { n } \frac { 8 1 } { 1 6 } \bigg ) } \\ & { = \frac { 8 1 } { 1 6 n ^ { 4 } } \big [ \displaystyle n + 3 n ( n - 1 ) \big ] } \\ & { = \frac { 8 1 } { 1 6 n ^ { 4 } } \big ( 2 \big ) \ln - 2 \bigg ) . } \end{array}
$$

Expanding $5 - 4 / \overline { { Z ^ { 2 } } } _ { r }$ around $\overline { { Z ^ { 2 } } } _ { n } = 5 / 2$ gives

$$
{ \hat { \theta } } _ { n } = { \frac { 1 7 } { 5 } } + { \frac { 1 6 } { 2 5 } } \left( { \overline { { Z ^ { 2 } } } } _ { n } - { \frac { 5 } { 2 } } \right) - { \frac { 3 2 } { 1 2 5 } } \left( { \overline { { Z ^ { 2 } } } } _ { n } - { \frac { 5 } { 2 } } \right) ^ { 2 } + R _ { n } .
$$

The absolute value of the third derivative of this expression with respect to its argument is at most 24 on [1, 4]. Since $\overline { { Z ^ { 2 } } } _ { n } \in [ 1 , 4 ]$ , Taylor’s remainder satisfies

$$
\left| R _ { n } \right| \leq 4 \left| { \overline { { Z ^ { 2 } } } } _ { n } - { \frac { 5 } { 2 } } \right| ^ { 3 } .
$$

By Cauchy–Schwarz,

$$
\begin{array} { r l } & { \mathbb { E } \left[ \left| \overline { { Z ^ { 2 } } } _ { n } - \frac { 5 } { 2 } \right| ^ { 3 } \right] \leq \sqrt { \mathbb { E } \left[ \left( \overline { { Z ^ { 2 } } } _ { n } - \frac { 5 } { 2 } \right) ^ { 2 } \right] } \mathbb { E } \left[ \left( \overline { { Z ^ { 2 } } } _ { n } - \frac { 5 } { 2 } \right) ^ { 4 } \right] } \\ & { \qquad = O ( n ^ { - 3 / 2 } ) . } \end{array}
$$

Hence $\mathbb { E } [ | R _ { n } | ] = O ( n ^ { - 3 / 2 } )$ . Taking expectations in the expansion gives

$$
\mathbb { E } [ \hat { \theta } _ { n } - \theta ^ { * } ] = - \frac { 3 2 } { 1 2 5 } \frac { 9 } { 4 n } + O ( n ^ { - 3 / 2 } ) = - \frac { 7 2 } { 1 2 5 n } + O ( n ^ { - 3 / 2 } ) .
$$

All five assumptions hold, with $\alpha = 1$ and $b = - 7 2 / 1 2 5$ . This is the bias relative to the best linear predictor, $\theta ^ { * } = 1 7 / 5$ Squared loss can therefore give either an unbiased estimator or a biased one.

## C. Squared loss with a quartic penalty

The sample gradient and Hessian are

$$
\frac { \partial f } { \partial \theta } ( \theta ; X ) = \theta + \theta ^ { 3 } - X , \qquad \frac { \partial ^ { 2 } f } { \partial \theta ^ { 2 } } ( \theta ; X ) = 1 + 3 \theta ^ { 2 } .
$$

Since $\mathbb { E } [ X ] = 0$ , the population derivative is $F _ { 0 } ^ { \prime } ( \theta ) = \theta + \theta ^ { 3 }$ . Its unique zero is $\theta ^ { * } = 0$ , which lies in the interior of Θ. The Hessian is at least 1. This verifies Assumptions 1 and 2.

For the gradient, $| \theta | \leq 1$ and $| X | \le 1$ give

$$
\left| { \frac { \partial f } { \partial \theta } } ( \theta ; X ) \right| = | \theta + \theta ^ { 3 } - X | \leq | \theta | + | \theta | ^ { 3 } + | X | \leq 3 .
$$

Its eighth absolute moment is therefore at most $3 ^ { 8 }$

For any $\theta , \theta ^ { \prime } \in [ - 1 , 1 ]$ , the Hessian satisfies

$$
\begin{array} { r l } { \displaystyle \left. \frac { \partial ^ { 2 } f } { \partial \theta ^ { 2 } } ( \theta ^ { \prime } ; X ) - \frac { \partial ^ { 2 } f } { \partial \theta ^ { 2 } } ( \theta ; X ) \right. = 3 \lvert ( \theta ^ { \prime } ) ^ { 2 } - \theta ^ { 2 } \rvert } & { } \\ { = 3 \lvert \theta ^ { \prime } - \theta \rvert \lvert \theta ^ { \prime } + \theta \rvert } & { } \\ { \leq 6 \lvert \theta ^ { \prime } - \theta \rvert . } \end{array}
$$

It is therefore 6-Lipschitz on Θ. The constant 6 also satisfies the moment conditions on the Lipschitz constant in Assumption 3. The Hessian does not depend on X, so the population Hessian is also $F _ { 0 } ^ { \prime \prime } ( \theta ) = 1 + 3 \theta ^ { 2 }$ . Thus the centered Hessian is

$$
\frac { \partial ^ { 2 } f } { \partial \theta ^ { 2 } } ( \theta ; X ) - F _ { 0 } ^ { \prime \prime } ( \theta ) = ( 1 + 3 \theta ^ { 2 } ) - ( 1 + 3 \theta ^ { 2 } ) = 0 .
$$

Its eighth absolute moment is zero. This verifies Assumption 3.

At $\theta ^ { * } = 0 .$ , the gradient equals −X. Since $\mathbb { E } [ X ] = 0 .$

$$
K = \operatorname { V a r } ( - X ) = \operatorname { V a r } ( X ) = \operatorname { \mathbb { E } } [ X ^ { 2 } ] - ( \operatorname { \mathbb { E } } [ X ] ) ^ { 2 } = \operatorname { \mathbb { E } } [ X ^ { 2 } ] .
$$

If $\mathbb { E } [ X ^ { 2 } ] = 0$ , then $X = 0$ almost surely. This would imply $\mathbb { E } [ X ^ { 3 } ] = 0$ , contrary to our choice of distribution. Hence $K > 0$ which verifies Assumption 4.

Write ${ \overline { { X } } } _ { n } = n ^ { - 1 } \sum _ { i = 1 } ^ { n } X _ { i }$ . The empirical derivative is strictly increasing. At the endpoints it satisfies $F _ { n } ^ { \prime } ( - 1 ) = - 2 - { \overline { { X } } } _ { n } <$ 0 and $F _ { n } ^ { \prime } ( 1 ) = 2 - { \overline { { X } } } _ { n } > 0$ . The empirical minimizer therefore lies in $( - 1 , 1 )$ for every sample and satisfies

$$
\widehat { \theta } _ { n } + \widehat { \theta } _ { n } ^ { 3 } = \overline { { { X } } } _ { n } .
$$

This gives $\lvert \hat { \theta } _ { n } \rvert \leq \lvert \overline { { X } } _ { n } \rvert$ and $\overline { { { X } } } _ { n } - \hat { \theta } _ { n } = \hat { \theta } _ { n } ^ { 3 }$ . Factoring the difference of cubes yields

$$
\begin{array} { r l } & { \left| \hat { \theta } _ { n } - \overline { { X } } _ { n } + \overline { { X } } _ { n } ^ { 3 } \right| = \left| \overline { { X } } _ { n } ^ { 3 } - \hat { \theta } _ { n } ^ { 3 } \right| } \\ & { \qquad = | \overline { { X } } _ { n } - \hat { \theta } _ { n } | \left| \overline { { X } } _ { n } ^ { 2 } + \overline { { X } } _ { n } \hat { \theta } _ { n } + \hat { \theta } _ { n } ^ { 2 } \right| } \\ & { \qquad \leq 3 | \overline { { X } } _ { n } | ^ { 5 } . } \end{array}
$$

Independence and $\mathbb { E } [ X ] = 0$ imply

$$
\mathbb { E } [ \overline { { X } } _ { n } ] = 0 , \qquad \mathbb { E } [ \overline { { X } } _ { n } ^ { 3 } ] = \frac { n \mathbb { E } [ X ^ { 3 } ] } { n ^ { 3 } } = \frac { \mathbb { E } [ X ^ { 3 } ] } { n ^ { 2 } } .
$$

In the cubic expansion, only terms with all three indices equal have nonzero expectation.

To bound the fifth absolute moment, expand the sixth power of $\textstyle \sum _ { i = 1 } ^ { n } X _ { i } . \mathrm { ~ A ~ }$ product has expectation zero if an index appears only once. The remaining products contain at most three distinct indices. There are $O ( n ^ { 3 } )$ such products, and each has absolute value at most one. Dividing by $n ^ { 6 }$ gives $\mathbb { E } [ \overline { { X } } _ { n } ^ { 6 } ] = O ( n ^ { - 3 } )$ . Since the function $t \mapsto t ^ { 5 / 6 }$ is concave on $[ 0 , \infty )$ Jensen’s inequality gives

$$
\begin{array} { r } { \mathbb { E } [ | \overline { { X } } _ { n } | ^ { 5 } ] = \mathbb { E } \Big [ ( \overline { { X } } _ { n } ^ { 6 } ) ^ { 5 / 6 } \Big ] } \\ { \leq \big ( \mathbb { E } [ \overline { { X } } _ { n } ^ { 6 } ] \big ) ^ { 5 / 6 } . } \end{array}
$$

The sixth-moment bound means that $\mathbb { E } [ \overline { { X } } _ { n } ^ { 6 } ] \leq C n ^ { - 3 }$ for some constant $C > 0$ and all sufficiently large n. Hence

$$
\mathbb { E } [ | \overline { { X } } _ { n } | ^ { 5 } ] \le ( C n ^ { - 3 } ) ^ { 5 / 6 } = C ^ { 5 / 6 } n ^ { - 5 / 2 } = O ( n ^ { - 5 / 2 } ) .
$$

The absolute value of an expectation is at most the expectation of the absolute value. The difference-of-cubes bound therefore gives

$$
\begin{array} { r l } & { \left| \mathbb { E } [ \widehat \theta _ { n } ] - \mathbb { E } [ \overline { { X } } _ { n } ] + \mathbb { E } [ \overline { { X } } _ { n } ^ { 3 } ] \right| \leq \mathbb { E } \Big [ \Big | \widehat \theta _ { n } - \overline { { X } } _ { n } + \overline { { X } } _ { n } ^ { 3 } \Big | \Big ] } \\ & { \qquad \leq 3 \mathbb { E } [ | \overline { { X } } _ { n } | ^ { 5 } ] } \\ & { \qquad = O ( n ^ { - 5 / 2 } ) . } \end{array}
$$

Substituting $\mathbb { E } [ \overline { { X } } _ { n } ] = 0$ and $\mathbb { E } [ \overline { { X } } _ { n } ^ { 3 } ] = \mathbb { E } [ X ^ { 3 } ] / n ^ { 2 }$ gives

$$
\bigg \vert \mathbb { E } [ \hat { \theta } _ { n } ] + \frac { \mathbb { E } [ X ^ { 3 } ] } { n ^ { 2 } } \bigg \vert = O ( n ^ { - 5 / 2 } ) .
$$

Since $\theta ^ { * } = 0$ , this is equivalent to

$$
\mathbb { E } [ \hat { \theta } _ { n } - \theta ^ { * } ] = - \frac { \mathbb { E } [ X ^ { 3 } ] } { n ^ { 2 } } + O ( n ^ { - 5 / 2 } ) .
$$

All five assumptions hold, with

$$
\alpha = 2 , \qquad b = - \mathbb { E } [ X ^ { 3 } ] \neq 0 .
$$

For the concrete distribution in Section IV-C,

$$
\mathbb { E } [ X ^ { 3 } ] = { \frac { 2 } { 3 } } \left( - { \frac { 1 } { 3 } } \right) ^ { 3 } + { \frac { 1 } { 3 } } \left( { \frac { 2 } { 3 } } \right) ^ { 3 } = { \frac { 2 } { 2 7 } } , \qquad b = - { \frac { 2 } { 2 7 } } .
$$

## APPENDIX D NUMERICAL EXPERIMENT DETAILS

## A. Synthetic calculations

The synthetic experiments in Section V use the three models of Section IV, with $q = 1 / 3$ for the Bernoulli mean and $\mathbb { P } ( X = - 1 / 3 ) = 2 / 3 , \mathbb { P } ( X = 2 / 3 ) = 1 / 3$ for the quartic example. Each local estimator is determined by a binomial count: the number of ones for the sample mean, the number of observations with $Z = 2$ for least squares, and the number with $X = 2 / 3$ for the quartic model. The corresponding success probabilities are $1 / 3 , 1 / 2 ,$ and $1 / 3$

We sum over these counts to compute the finite-sample moments of the local estimators. Binomial probabilities are evaluated using probability ratios around the mode. A Hoeffding bound selects a retained range with total omitted probability at most $1 0 ^ { - 1 8 }$ ; retained probabilities are renormalized. For the quartic model, twelve Newton iterations solve $\widehat { \theta } _ { n } + \widehat { \theta } _ { n } ^ { 3 } = \bar { X } _ { n }$ . The identity $\mathbb { E } [ \hat { \theta } _ { n } ] = - \mathbb { E } [ \hat { \theta } _ { n } ^ { 3 } ]$ reduces cancellation in its small bias. The least-squares bias calculation likewise uses the symmetry of its binomial distribution. The unbiased mean uses its exact bias and variance.

Independence and the weights $c _ { 0 } = p / N , c _ { 1 } = n / N$ then determine the aggregate MSE, including the worker–server bias cross term. Server-only and pooled MSE are obtained from the same local-moment calculations at sample sizes p and N. Thus the curves use numerical finite-sample expectations, rather than Monte Carlo averages or substitution into the asymptotic rate formula.

The grid is $n \in \{ 8 , 1 6 , 3 2 , 6 4 , 1 2 8 , 2 5 6 \}$ . The three models first use $m = n ^ { 2 } , p = n ^ { 3 }$ . The remaining least-squares regimes are $( m , p ) = ( n , n ) , ( n ^ { 2 } , \lceil n ^ { 7 / 2 } \rceil )$ , and $( n ^ { 2 } , n ^ { 4 } )$ . Separate Monte Carlo checks use 20,000 repetitions at $n = 8 , 1 6$ in all six regimes; they do not supply the plotted curves. The plotted quantity is N MSE, and the reference slopes are the theoretical predictions.

## B. CIFAR-10 data and representation

The experiments condition on the fixed pool of 50,000 CIFAR-10 training images, the frozen feature extractor, and the 10,000-image test set. Worker and server observations are drawn independently with replacement from the training pool. The same draws are used across aggregation strategies and classifier heads within a repetition. As m changes, the same ordered worker sample is repartitioned into equal-sized local datasets. As p changes, server datasets are nested prefixes of a separately sampled stream. Repetitions use independent training draws. The test set is used only for evaluation.

The extractor is Torchvision’s ResNet-18 with IMAGENET1K\_V1 weights and the final classification layer replaced by the identity. Its evaluation transforms resize images to 256 pixels on the shorter side, apply a central $2 2 4 \times 2 2 4$ crop, and use ImageNet channel normalization. Each resulting 512-dimensional feature vector is normalized to unit Euclidean norm. The extractor remains in evaluation mode and is not trained on CIFAR-10. No data augmentation is used.

## C. Local fitting and aggregation

Ridge classifiers minimize the mean half squared error against one-hot labels plus 0.01/2 times the squared Frobenius norm of the coefficient matrix. They have no intercept. The penalty coefficient is fixed in the average-loss objective at every sample size. Fits use a direct linear-system solve in double precision, using the primal or dual system according to its dimension. Ridge parameter MSE is the mean squared Frobenius distance to the ridge minimizer over the complete training pool, without division by parameter dimension. This reference is distinct from the pooled estimator fitted to the sampled N observations in each repetition.

The linear cross-entropy head is affine. The nonlinear head has a 512 → 128 → 10 architecture with a ReLU hidden layer and biases in both layers. Both use mean cross-entropy plus $1 0 ^ { - 4 } / 2$ times the squared Euclidean norm of all parameters, including biases, as the training objective. Each fit uses 400 full-batch Adam updates with learning rate 0.01. When observations are processed in chunks, gradients are accumulated over the complete local dataset before an update. Pooled fits use the same optimization budget.

Within each head and repetition, workers, server, and pooled fits start from identical parameters. The linear head starts at zero. Nonlinear weights are drawn from zero-mean Gaussian distributions with variances 2/512 and 1/128 in the first and second layers, respectively; biases start at zero. Nonlinear initializations vary across repetitions. Sample-size weighting averages corresponding fitted parameters, including both nonlinear layers and their biases. Common initialization does not guarantee that learned hidden units remain aligned. There is no further optimization after aggregation.

## D. Experimental grids and uncertainty

The fixed-budget study uses $m n \ = \ 3 2 7 6 8$ and $p \ = \ 8 1 9 2$ . Ridge uses $m ~ \in ~ \{ 2 , 4 , 8 , 1 6 , 3 2 , 6 4 , 1 2 8 , 2 5 6 , 5 1 2 \}$ ; the cross-entropy heads use $m ~ \in ~ \{ 2 , 8 , 3 2 , 1 2 8 , 5 1 2 \}$ . The server-size study uses $m n ~ = ~ 8 1 9 2 , ~ m ~ \in ~ \{ 8 , 1 2 8 , 5 1 2 \}$ , and $p \in \{ 1 2 8 , 5 1 2 , 2 0 4 8 , 8 1 9 2 , 3 2 7 6 8 , 1 3 1 0 7 2 \}$ . The main-text server-size panels report $m = 1 2 8$ . There are 20 repetitions for ridge and 10 for each cross-entropy head. Sampling and initialization use base seed 20261022 with separate random streams.

Reported confidence intervals are approximate pointwise 95% Student-t intervals for mean metrics over repetitions. Their half-width is the corresponding Student-t quantile times the sample standard deviation divided by the square root of the repetition count. Comparisons between methods use differences within the same repetition. These intervals describe variation conditional on the fixed training pool, feature extractor, and test set.

Accuracy is the fraction of correctly classified test images. Only ridge is assigned parameter MSE. The saved predictionloss metric is one-half the mean sum of squared output errors against one-hot labels: it uses raw ridge scores and softmax probabilities for the cross-entropy heads. It is distinct from parameter MSE and, for the cross-entropy heads, from the training loss.

The longer-training check uses the first three repetitions of the fixed-budget study at $m = 8 , 5 1 2$ , comparing 400 and 800 updates for both cross-entropy heads on the same data and initialization. This diagnostic does not certify convergence or global optimality. No bias exponent or verification of Assumption 5 is claimed for CIFAR-10; the nonlinear comparison lies outside the convex theory.