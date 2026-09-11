# Generalization Analysis of Distributed Kernel-based Robust Gradient Descent Algorithms<sup>†</sup>

Jun-Yi Meng<sup>1</sup>, Yuan Mao<sup>2</sup> and Zheng-Chu Guo<sup>1</sup>

<sup>1</sup> School of Mathematical Sciences, Zhejiang University, Hangzhou 310058, P. R. China

<sup>2</sup> College of Informatics, Huazhong Agricultural University, Wuhan 430070, P. R. China

## Abstract

In this paper, we investigate the generalization performance of distributed gradient descent algorithms in a reproducing kernel Hilbert space under a robust loss function l . By exploiting the spectral characterization of gradient descent together with the intrinsic properties of robust loss functions, we establish optimal learning rates for the distributed kernel-based robust gradient descent (DKRGD) algorithm with an appropriately chosen scale parameter σ. The proposed parameter choice of σ simultaneously alleviates the saturation phenomenon and guarantees statistical robustness. A key technical contribution is a novel error analysis that provides substantially sharper bounds for products of operators, thereby significantly relaxing existing restrictions on the maximum number of local machines while retaining optimal learning rates. Finally, we develop a communication-eficient strategy that further improves the convergence performance of DKRGD.

Keywords: Learning theory, Distributed learning, Robust regression, Gradient descent, Communication Mathematics Subject Classification (2020): 68T05, 68Q32, 68W15, 62G08, 62G35, 62J02

## Introduction

With the rapid growth of data across a wide range of applications, performing machine learning tasks on a single machine is becoming increasingly impractical. Meanwhile, privacy constraints and data isolation often prevent data collected by diferent organizations from being directly pooled. For example, patient records may be stored separately across hospitals, while financial institutions typically maintain their risk-management data independently. Thesev considerations have motivated significant interest in distributed learning, where statistical models are constructed from data stored across multiple local machines without requiring direct access to the entire data set. Among vari-<sub>a</sub> ous distributed learning paradigms, the “divide-and-conquer” approach has attracted particular attention due to its conceptual simplicity and computational eficiency [1, 2].

A substantial body of work has investigated the statistical performance of distributed kernel-based learning algo rithms. For distributed regularized least-squares algorithms, optimal learning rates in expectation were established in [1, 2]. Subsequently, [3] and [4] extended the analysis to distributed spectral algorithms and derived optimal learning rates in expectation. More recently, [5] refined the theoretical analysis of distributed kernel ridge regression (DKRR), establishing optimal learning rates both in expectation and with high probability.

Despite these advances, the general frameworks developed in [3, 4] exhibit a saturation phenomenon with respect to the maximal admissible number of local machines. More precisely, once the regularity of the regression function exceeds a certain threshold, further improvements in regularity no longer allow for an increase in the number of local machines while preserving the optimal learning rate. Moreover, the results in [5] are restricted to the regularity range $\textstyle { \frac { 1 } { 2 } } \leq r \leq 1$ . To alleviate this saturation phenomenon, [6] and [7] exploited the specific spectral structure of gradient descent algorithms to derive improved bounds on the maximum number of local machines.

Another important issue in statistical learning is robustness against outliers and heavy-tailed observations. Leastsquares estimators can be sensitive to atypical observations because large residuals are penalized disproportionately heavily. To improve robustness, [8, 9, 10] considered a family of robust loss functions of the form $\begin{array} { r } { l _ { \sigma } ( u ) = G \left( \frac { u ^ { 2 } } { \sigma ^ { 2 } } \right) } \end{array}$ where G is a windowing function and $\sigma > 0$ is a scale parameter. Appropriate choices of G lead to a broad class of robust loss functions that reduce the influence of large residuals while maintaining desirable statistical properties.

These developments naturally raise the question of whether robust kernel-based gradient methods can be efectively combined with distributed learning. In particular, it is important to understand whether a distributed robust gradient descent algorithm can simultaneously retain statistical robustness, achieve optimal learning rates, and accommodate a suficiently large number of local machines. The last issue is especially important because the restriction on the number of machines directly determines the extent to which a distributed method can benefit from increasing computationa resources.

In this paper, we study the generalization performance of DKRGD in a reproducing kernel Hilbert space (RKHS). Each local machine applies a robust gradient descent algorithm to its own data, and the resulting local estimators are subsequently aggregated to form a global estimator. Our analysis characterizes the statistical performance of the last iterate under standard source and capacity conditions, and reveals the interaction among the regularity of the regression function, the capacity of the hypothesis space, the robust scale parameter σ, and the number of local machines.

The main contributions of this paper are summarized as follows.

First, we establish a generalization theory for DKRGD both in expectation and with high probability. With appropriate choices of the early-stopping time and the scale parameter σ, the proposed algorithm achieves the optimal learning rates determined by the regularity of the regression function and the capacity of the underlying RKHS. In particular, the scale parameter can be selected so that the optimal learning rates do not deteriorate due to the additional error induced by the robust loss, while the estimator retains the robustness associated with the loss function $l _ { \sigma }$

Second, we develop a refined error analysis that yields sharper estimates for the operator products arising in the study of distributed gradient descent. The resulting bounds substantially relax the restriction on the maximum number of local machines. In particular, the improved analysis alleviates the saturation phenomenon that arises in existing general analyses of distributed spectral algorithms when the regularity of the regression function becomes suficiently high.

Third, we develop a DKRGD algorithm with communication based on a Newton–Raphson-type correction. The proposed communication strategy further enlarges the admissible number of local machines while preserving the optimal learning rate with high probability. Moreover, the maximum number of local machines increases with the number of communication rounds, revealing a favorable trade-of between communication cost and distributed scalability.

The remainder of this paper is organized as follows. Section 2 outlines the regression setting and robust loss functions, followed by a detailed description of DKRGD and its communication strategy. Under mild assumptions, Section 3 establishes optimal learning rates for DKRGD both with high probability and in expectation, and optimal learning rates for DKRGD with communication under high probability. Additionally, it discusses the corresponding restrictions on the number of local machines. Section 4 discusses and compares the theoretical results with related distributed learning methods. Section 5 presents the error decomposition and auxiliary results used in the analysis. Finally, the proofs of the main results are provided in Section 6.

## 2 Problem Setting and Distributed Kernel-based Robust Gradient Descent

In this paper, we consider the nonparametric regression problem based on i.i.d. samples $D = \{ x _ { i } , y _ { i } \} _ { i = 1 } ^ { | D | }$ drawn from an unknown distribution $\rho$ on $Z : = \mathcal { X } \times \mathcal { Y }$ , where input space $\mathcal { X }$ is a separable metric space and $\mathcal { V } \subset \mathbb { R }$ is the output space. The training data are assumed to be generated according to the model $Y = f _ { \rho } ( X ) + \epsilon$ , where $\mathbb { E } [ \epsilon | X ] = 0$ $X$ is the explanatory variable and $Y$ denotes the corresponding response variable. The regression function $f _ { \rho }$ is defined as

$$
f _ { \rho } ( x ) = \int _ { \mathcal { V } } y \mathrm { d } \rho ( y | x ) , \quad x \in \mathcal { X } ,
$$

where $\rho ( y | x )$ denotes the conditional distribution at x induced by $\rho .$ The goal of regression is to recover $f _ { \rho }$ from the observed random samples.

Our objective is to investigate the generalization performance of robust gradient descent algorithms in a distributed learning setting. To this end, we adopt the class of robust loss functions introduced in [8], defined b

$$
l _ { \sigma } ( u ) = G ( \frac { u ^ { 2 } } { \sigma ^ { 2 } } ) ,
$$

where $G \colon \mathbb { R } _ { + } \to \mathbb { R }$ is a windowing function and $\sigma > 0$ is a scale parameter controlling the degree of robustness. Hereafter, the windowing function G is assumed to satisfy the following two conditions, which are

$$
G _ { + } ^ { \prime } ( 0 ) > 0 , \ C _ { G } : = \ \operatorname* { s u p } _ { s \in ( 0 , \infty ) } | G ^ { \prime } ( s ) | \quad \mathrm { w i t h } \ G ^ { \prime } ( s ) > 0 \ \mathrm { f o r } \ s > 0 ,\tag{1}
$$

and there exists some $p \geq 0$ and $c _ { p } > 0$ such that

$$
\left| { G } ^ { \prime } ( s ) - { G } _ { + } ^ { \prime } ( 0 ) \right| \le c _ { p } | s | ^ { p } , \quad \forall s > 0 .\tag{2}
$$

By choosing diferent windowing functions $G ,$ the general form $l _ { \sigma } ( u ) = G ( u ^ { 2 } / \sigma ^ { 2 } )$ encompasses a variety of commonly used robust loss functions. Several representative examples for regression are presented below, where $\mathbb { I } _ { A }$ denotes the indicator function of a set A.

Example 1. Huber’s loss combines the advantages of mean squared error and mean absolute error [11]. It takes a quadratic form for small errors and a linear form for larger errors. The balance between two terms is controlled by a threshold parameter $\sigma ,$ providing robustness to outliers while maintaining precision.

$$
l _ { \sigma } ( u ) = \mathbb { I } _ { \{ | u | \leq \sigma \} } \frac { u ^ { 2 } } { 2 \sigma ^ { 2 } } + \mathbb { I } _ { \{ | u | > \sigma \} } \left( \frac { | u | } { \sigma } - \frac 1 2 \right) , G ( s ) = \mathbb { I } _ { \{ s \leq 1 \} } \frac { s } { 2 } + \mathbb { I } _ { \{ s > 1 \} } \left( \sqrt { s } - \frac 1 2 \right) , p = 0 , c _ { p } = \frac { 1 } { 2 } .
$$

Example 2. Fair loss focuses on balancing treatment for larger errors through a piecewise linear modeling approach It imposes a smaller penalty on larger errors compared to mean squared error, making it suitable for scenarios where equal treatment of errors across diferent ranges is desired.

$$
l _ { \sigma } ( u ) = \frac { \lvert u \rvert } { \sigma } - \log \left( 1 + \frac { \lvert u \rvert } { \sigma } \right) , G ( s ) = \sqrt { s } - \log ( 1 + \sqrt { s } ) , p = \frac { 1 } { 2 } , c _ { p } = \frac { 1 } { 2 } .
$$

Both loss functions above are convex, and their curves for diferent values of $\sigma$ are shown in Figure 1.

![](images/97e4f589f605dbce035afae8093e5e0b57557a43ca9f60d61a13f149ae8319af.jpg)  
(a)

![](images/df7c9e7a862c73be2e4ccde77ce520cc482ab0939c30bd9f69b346cda3650fd8.jpg)  
(b)  
Figure. 1: Two convex types of $l _ { \sigma }$ with diferent σ. (a) Huber Loss, (b) Fair Loss.

Note that the loss function is not necessarily convex; two illustrative nonconvex losses are given below, and their plots for diferent values of σ are shown in Figure 2.

Example 3. Cauchy loss, motivated by the Cauchy distribution, is designed to aggressively suppress extreme errors.   
Compared to Huber’s loss, Cauchy loss has a slower decay in the tails, leading to a stronger suppression of outliers.

$$
l _ { \sigma } ( u ) = \log \left( 1 + \frac { u ^ { 2 } } { 2 \sigma ^ { 2 } } \right) , G ( s ) = \log \left( 1 + \frac { s } { 2 } \right) , p = 1 , c _ { p } = \frac { 1 } { 4 } .
$$

Example 4. Welsch loss can be induced by the well-known correntropy loss, which resembles mean squared error for small errors and adopts a square root form for larger errors [12]. The roots of the second derivative of Welsch loss are located at $u = \pm \sigma$ , indicating that the loss function is concave for $| u | > \sigma$ and convex for $| u | < \sigma$ . Therefore, adjusting scaling parameter σ allows rejecting outliers while keeping a similar prediction accuracy as that of least squares loss.

$$
l _ { \sigma } ( u ) = 1 - \exp \left( - \frac { u ^ { 2 } } { 2 \sigma ^ { 2 } } \right) , G ( s ) = 1 - \exp \left( - \frac { s } { 2 } \right) , p = 1 , c _ { p } = \frac { 1 } { 4 } .
$$

![](images/896fa015ae78b8c119911759bda1392469fabc921647963a1dac14963d345f42.jpg)  
(a)

![](images/6e31b07a5f972953050f9e8f3ec733f868a21572e271eabaa93230ee041b5757.jpg)  
(b)  
Figure. 2: Two nonconvex types of $l _ { \sigma }$ with diferent σ. (a) Cauchy Loss, (b) Welsch Loss.

## 2.1 Distributed Kernel-based Robust Gradient Descent

In this subsection, we introduce the DKRGD algorithm. We begin with some notation that will be used throughout the subsequent analysis.

Denote by $K \colon \mathcal { X } \times \mathcal { X }  \mathbb { R } \mathrm { ~ a ~ }$ Mercer kernel and $( \mathcal { H } _ { K } , \| \cdot \| _ { K } )$ the corresponding RKHS. Given i.i.d samples $D =$ $\{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { | D | }$ drawn from $\rho ,$ the empirical risk is defined by a robust loss function $l _ { \sigma }$ as

$$
\mathcal { E } _ { l _ { \sigma } } = \frac { 1 } { | D | } \sum _ { ( x _ { i } , y _ { i } ) \in D } l _ { \sigma } \left( f ( x _ { i } ) - y _ { i } \right) ,\tag{3}
$$

where $| D |$ is the cardinality of $D .$ . Then a robust estimator can be obtained by minimizing the empirical risk $\mathcal { E } _ { l _ { \sigma } }$ over $\mathcal { H } _ { K }$ . Let $K _ { x } \colon \mathcal { X } $ R be the function defined by $K _ { x } ( \cdot ) = K ( x , \cdot )$ for $x \in \mathcal { X }$ . By the reproducing property of kernel K and proposition 2.1 in [13], we can compute the gradient of (3) and further define the kernel-based robust gradient descent algorithms as follows with $f _ { 1 , D } = 0$ and

$$
f _ { t + 1 , D } = f _ { t , D } - \frac { \eta } { | D | } \sum _ { i = 1 } ^ { | D | } G ^ { \prime } ( \xi _ { t , D , \sigma } ( z _ { i } ) ) ( f _ { t , D } ( x _ { i } ) - y _ { i } ) K _ { x _ { i } } , \quad \forall t \geqslant 1 ,\tag{4}
$$

where $\eta > 0$ is the step size, $z _ { i } = ( x _ { i } , y _ { i } )$ , and

$$
\xi _ { t , D , \sigma } ( z _ { i } ) = \frac { ( f _ { t , D } ( x _ { i } ) - y _ { i } ) ^ { 2 } } { \sigma ^ { 2 } } .
$$

We next extend (4) to a distributed learning setting. Specifically, the full data set D is partitioned into m disjoint subsets $\{ D _ { j } \} _ { j = 1 } ^ { m }$ such that $\textstyle D = \bigcup _ { j = 1 } ^ { m } D _ { j }$ and $D _ { i } \cap D _ { j } = \emptyset$ for $i \neq j$ . On the j-th local machine, the robust gradient descent iteration (4) is performed using only the local sample $D _ { j }$ , yielding the local estimator $f _ { t , D _ { j } }$ . Subsequently, the local estimators are aggregated through an average weighted by the local sample sizes to produce a global estimator

$$
\bar { f } _ { t , D } = \sum _ { j = 1 } ^ { m } \frac { | D _ { j } | } { | D | } f _ { t , D _ { j } } .\tag{5}
$$

When each local machine performs T gradient descent iterations, our analysis focuses on the last iterate $\bar { f } _ { T + 1 , D }$ , where $T = T ( | D | ) : \mathbb { N }  \mathbb { N }$ serves as an early-stopping rule to prevent overfitting.

The distributed estimator (5) follows the standard divide-and-conquer paradigm: the local machines perform their computations independently, and communication is required only for the final aggregation of the local estimators. Although such a strategy is simple and communication-eficient, weighted aggregation alone may fail to fully compensate for the statistical loss caused by splitting the full sample across multiple local machines [5]. Consequently, to retain the optimal learning rate, the number of local machines typically has to satisfy a restrictive upper bound.

A natural approach to alleviating this restriction is to allow additional communication among the local machines. Existing distributed learning methods achieve this by exchanging various types of information, including data [14], gradients [5, 15, 16], and local estimators [17, 18, 19]. In particular, [5] developed a Newton–Raphson-based commu nication strategy for DKRR, which substantially relaxed the restriction on the admissible number of local machines Motivated by this idea, we develop an analogous communication strategy for DKRGD in the next subsection.

One of the main objectives of this paper is to establish optimal learning rates for the distributed estimator (5), both in expectation and with high probability. By exploiting the concentration inequality developed in [20], our analysis allows a substantially larger number of local machines than that permitted by existing analyses. Furthermore we introduce a DKRGD algorithm with communication that further relaxes the restriction on the number of local machines, without requiring additional unlabeled data as in [6, 21].

## 2.2 DKRGD with Communication

In this subsection, we propose a novel communication strategy for DKRGD with the aim of further relaxing the restriction on the number of local machines. Our construction is motivated by the Newton–Raphson-type communication strategy introduced in [5]. To facilitate the development of the algorithm, we first derive two operator representations of the robust gradient descent iteration (4).

Define the empirical integral operator $L _ { K , D }$ as follows,

$$
L _ { K , D } ( f ) = \frac { 1 } { | D | } \sum _ { ( x _ { i } , y _ { i } ) \in D } f ( x _ { i } ) K _ { x _ { i } } , \qquad \forall f \in \mathcal { H } _ { K } ,
$$

and denote

$$
E _ { t , D , \sigma } = \frac { 1 } { | D | } \sum _ { ( x _ { i } , y _ { i } ) \in D } ( G _ { + } ^ { \prime } ( 0 ) - G ^ { \prime } ( \xi _ { t , D , \sigma } ( z _ { i } ) ) ) ( f _ { t , D } ( x _ { i } ) - y _ { i } ) K _ { x _ { i } } ,\tag{6}
$$

then the algorithm (4) can be rewritten as

$$
\begin{array} { l } { { f _ { t + 1 , D } = f _ { t , D } - \displaystyle \frac { \eta } { | D | } \sum _ { i = 1 } ^ { | D | } G _ { + } ^ { \prime } ( 0 ) ( f _ { t , D } ( x _ { i } ) - y _ { i } ) K _ { x _ { i } } + \eta E _ { t , D , \sigma } } } \\ { { = ( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } ) f _ { t , D } + \eta G _ { + } ^ { \prime } ( 0 ) \hat { f } _ { K , D } + \eta E _ { t , D , \sigma } , } } \end{array}
$$

where I denotes the identity operator, and $\begin{array} { r } { \hat { f } _ { K , D } = \frac { 1 } { | D | } \sum _ { ( x _ { i } , y _ { i } ) \in D } y _ { i } K _ { x _ { i } } } \end{array}$

By adding and subtracting the population integral operator $L _ { K }$ , defined in (13), we also obtain

$$
f _ { t + 1 , D } = \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) f _ { t , D } + \eta G _ { + } ^ { \prime } ( 0 ) ( L _ { K } - L _ { K , D } ) f _ { t , D } + \eta G _ { + } ^ { \prime } ( 0 ) \hat { f } _ { K , D } + \eta E _ { t , D , \sigma } .
$$

Then by induction, we can derive two representations for $f _ { t + 1 , D }$ as

$$
f _ { t + 1 , D } = \sum _ { i = 1 } ^ { t } \eta G _ { + } ^ { \prime } ( 0 ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { t - i } \hat { f } _ { K , D } + \sum _ { i = 1 } ^ { t } \eta \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { t - i } E _ { i , D , \sigma } ,\tag{7}
$$

and

$$
\begin{array} { l } { { f _ { t + 1 , D } = \displaystyle \sum _ { i = 1 } ^ { t } \eta G _ { + } ^ { \prime } ( 0 ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { t - i } \hat { f } _ { K , D } + \displaystyle \sum _ { i = 1 } ^ { t } \eta G _ { + } ^ { \prime } ( 0 ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { t - i } \left( L _ { K } - L _ { K , D } \right) f _ { i , D } } } \\ { { \displaystyle \qquad + \sum _ { i = 1 } ^ { t } \eta \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { t - i } E _ { i , D , \sigma } } . } \end{array}\tag{8}
$$

Based on the representation (7), we can design a communication strategy for DKRGD by the Newton-Raphson iteration applied in [5]. For the sake of convenience, we introduce a polynomial denoted by

$$
g _ { t } ( x ) = \sum _ { i = 1 } ^ { t } \eta G _ { + } ^ { \prime } ( 0 ) \left( 1 - \eta G _ { + } ^ { \prime } ( 0 ) x \right) ^ { t - i } , \quad \forall t > 0 ,\tag{9}
$$

and $g _ { t } \big ( L _ { K , D } \big )$ is defined by the spectral calculus. The invertibility of $g _ { t } \big ( L _ { K , D } \big )$ can be guaranteed by the specific choice of $\eta ,$ , which will be given in the main results. Then for any $f \in \mathcal { H } _ { K }$ , we can rewrite $f _ { t + 1 , D }$ as

$$
\begin{array} { l } { f _ { t + 1 , D } = g _ { t } ( L _ { K , D } ) \hat { f } _ { K , D } + \displaystyle \sum _ { i = 1 } ^ { t } \eta \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { t - i } E _ { i , D , \sigma } } \\ { = f - g _ { t } ( L _ { K , D } ) \left[ g _ { t } ^ { - 1 } ( L _ { K , D } ) f - \hat { f } _ { K , D } - g _ { t } ^ { - 1 } ( L _ { K , D } ) \displaystyle \sum _ { i = 1 } ^ { t } \eta \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { t - i } E _ { i , D , \sigma } \right] , } \end{array}\tag{10}
$$

where the last equation can be seen as the well known Newton-Raphson iteration. Further, a variant of the global gradient for the empirical risk (3) on $\mathcal { H } _ { K }$ over $f$ can be expressed as

$$
G _ { t + 1 , D } ( f ) = g _ { t } ^ { - 1 } ( L _ { K , D } ) f - \hat { f } _ { K , D } - g _ { t } ^ { - 1 } ( L _ { K , D } ) \sum _ { i = 1 } ^ { t } \eta \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { t - i } E _ { i , D , \sigma } ,\tag{11}
$$

where $E _ { i , D , \sigma }$ remains invariant to the change of $f .$ . In distributed setting, denote the global estimator (5) without communication by $\begin{array} { r } { \bar { f } _ { t + 1 , D } ^ { 0 } : = \bar { f } _ { t + 1 , D } = \sum _ { j = 1 } ^ { m } \frac { | D _ { j } | } { | D | } f _ { t + 1 , D _ { j } } } \end{array}$ . Then by applying (10) to each $D _ { j }$ , we can get

$$
\tilde { f } _ { t + 1 , D } ^ { 0 } = f - \sum _ { j = 1 } ^ { m } \left| \frac { D _ { j } } { | D | } g _ { t } ( L _ { K , D _ { j } } ) \left[ g _ { t } ^ { - 1 } ( L _ { K , D _ { j } } ) f - \hat { f } _ { K , D _ { j } } - g _ { t } ^ { - 1 } ( L _ { K , D _ { j } } ) \sum _ { i = 1 } ^ { t } \eta \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } \right) ^ { t - i } F _ { 3 , D _ { j } , \sigma } \right] \right| .
$$

For any $l \in  { \mathbb { N } } _ { + }$ , let

$$
\beta _ { D _ { j } , l - 1 } = g _ { t } ( L _ { K , D _ { j } } ) G _ { t + 1 , D } ( \bar { f } _ { t + 1 , D } ^ { l - 1 } ) .
$$

Then we can use the Newton-Raphson iteration to design our communication strategy as

$$
\begin{array} { l } { { \displaystyle { \bar { f } _ { t + 1 , D } ^ { l } = \bar { f } _ { t + 1 , D } ^ { l - 1 } - \sum _ { j = 1 } ^ { m } \frac { | D _ { j } | } { | D | } \beta _ { D _ { j } , l - 1 } } } } \\ { { \displaystyle ~ = \bar { f } _ { t + 1 , D } ^ { l - 1 } - \sum _ { j = 1 } ^ { m } \frac { | D _ { j } | } { | D | } g _ { t } ( L _ { K , D _ { j } } ) \left[ g _ { t } ^ { - 1 } ( L _ { K , D } ) \bar { f } _ { t + 1 , D } ^ { l - 1 } - \hat { f } _ { K , D } \right] } } \\ { { \displaystyle ~ + \sum _ { j = 1 } ^ { m } \frac { | D _ { j } | } { | D | } g _ { t } ( L _ { K , D _ { j } } ) g _ { t } ^ { - 1 } ( L _ { K , D } ) \sum _ { i = 1 } ^ { t } \eta \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { t - i } E _ { i , D , \sigma } } . } \end{array}\tag{12}
$$

Thus, each communication round applies a Newton–Raphson-type correction to the current global estimator by com bining information from the local empirical operators. As will be shown in the subsequent analysis, these additional communication rounds enable DKRGD to tolerate a substantially larger number of local machines while retaining the desired generalization performance.

It is worth noting that the global quantity $G _ { t + 1 , D }$ in (11) cannot be recovered by simply taking a weighted average of its local counterparts in general. Consequently, an exact implementation of (12) requires the estimation or communication of certain global operator quantities. Developing more eficient aggregation and operator-estimation procedures for this communication scheme is an interesting direction for future work.

## 3 Main Results

In this section, we present generalization bounds for DKRGD both with high probability and in expectation. We then investigate DKRGD with communication and show that, under suitable regularity conditions, communication can further relax the restriction on the number of local machines while preserving the optimal learning rate. To state the main results, we first introduce several standard assumptions concerning the boundedness of the output, the regularity of the regression function, and the capacity of the RKHS $\mathcal { H } _ { K }$

Throughout this paper, we assume that the kernel K is a Mercer kernel and $\mathcal { X }$ is compact, then K is bounded and define $\begin{array} { r } { \kappa = \sqrt { \operatorname* { s u p } _ { x \in \mathcal { X } } K ( x , x ) } < \infty } \end{array}$ . Without loss of generality, we will assume $\kappa \geq 1$ in the subsequent analysis. Let $\rho _ { X }$ be the marginal distribution of $\rho$ and $L _ { \rho _ { X } } ^ { 2 }$ be the Hilbert space of $\rho _ { X }$ square integrable functions on $\mathcal { X } ,$ , with norm denoted $\mathrm { b y } \| \cdot \| _ { \rho } .$ . The kernel K induces the integral operator $L _ { K }$ defined on $L _ { \rho _ { X } } ^ { 2 }$ (or the RKHS $\mathcal { H } _ { K } )$ by

$$
L _ { K } ( f ) = \int _ { \mathcal { X } } f ( x ) K _ { x } \mathrm { d } \rho _ { X } , \quad f \in L _ { \rho _ { X } } ^ { 2 } \ ( \mathrm { o r } \ f \in \mathcal { H } _ { K } ) .\tag{13}
$$

We first impose a standard boundedness condition on the output variable.

Assumption 1. There exists a constant $M > 0$ such that $| y | \le M$ almost surely with respect to $\rho .$

The boundedness of the outputs is a common assumption in the analysis of regression problems, which is also adopted in [3, 8]. Although it is slightly stricter than some moment conditions in more general settings [22], our analysis in this paper can be easily extended to the case by assuming moment conditions on the outputs.

The next assumption characterizes the regularity of the regression function.

Assumption 2. There exist $r > 0$ and $u _ { \rho } \in L _ { \rho _ { X } } ^ { 2 }$ such that

$$
f _ { \rho } = L _ { K } ^ { r } u _ { \rho } ,\tag{14}
$$

where $L _ { K } ^ { r }$ denotes the r-th power of $L _ { K } \colon L _ { \rho _ { X } } ^ { 2 } \to L _ { \rho _ { X } } ^ { 2 }$ as a compact and positive operator.

The regularity condition (14) of the regression function is a common assumption in the analysis of kernel-based learning algorithms [3, 5, 8, 22], which is also known as the source condition in the context of inverse problems. It states that $f _ { \rho }$ lies in the range of $L _ { K } ^ { r }$ , and the special case $r = 1 / 2$ corresponds to the situation where $f _ { \rho } \in \mathcal { H } _ { K }$ Intuitively, larger values of $r$ indicate higher regularity of $f _ { \rho } ,$ potentially resulting in improved learning rates.

The complexity of the hypothesis space $\mathcal { H } _ { K }$ with respect to the measure $\rho _ { X }$ is measured by the efective dimension defined as

$$
\begin{array} { r } { \mathcal { N } ( \lambda ) = \mathrm { T r } \left( ( \lambda I + L _ { K } ) ^ { - 1 } L _ { K } \right) , \quad \lambda > 0 . } \end{array}
$$

And the capacity assumption is given by the polynomial decay of the efective dimension.

Assumption 3. There exists some $s \in ( 0 , 1 ]$ and a constant $C _ { 0 } \geq 1$ which is independent of λ, such that

$$
\mathcal { N } ( \lambda ) \leq C _ { 0 } \lambda ^ { - s } , \quad \forall \lambda > 0 .\tag{15}
$$

Condition (15) with $s = 1$ is always satisfied by taking the constant $C _ { 0 } = \mathrm { T r } ( L _ { K } ) \leqslant \kappa ^ { 2 }$ . Now we are ready to present the main results, the first of which to be proved in section 6 exhibits optimal learning rates for DKRGD with high probability. Note that the generalization performance of target function estimation is measured by the $\rho -$ distance between the estimator and the regression function, i.e., $\left\| \bar { f } _ { T + 1 , D } - f _ { \rho } \right\| _ { \rho }$

Theorem 1. Let $0 < \delta < 1 , p > 0$ and $\begin{array} { r } { 0 < \eta \leq \frac { 1 } { \kappa ^ { 2 } \operatorname* { m a x } \{ G _ { + } ^ { \prime } ( 0 ) , C _ { G } \} } } \end{array}$ . Under Assumption 1-3 with $r > { \frac { 1 } { 2 } }$ and $0 < s \le 1$ , if $\lambda = | D | ^ { - { \frac { 1 } { 2 r + s } } } , T = \Big \lceil | D | ^ { \frac { 1 } { 2 r + s } } \Big \rceil , | D _ { 1 } | = \cdot \cdot \cdot = | D _ { m } |$ and

$$
m \leq { \frac { | D | ^ { \frac { 2 r + s - 1 } { 4 r + 2 s } } } { \left( \left( \log | D | \right) ^ { 5 } + 1 \right) ^ { 2 } } } ,\tag{16}
$$

then with confidence at least $1 - \delta$ , there holds

$$
\left\| \bar { f } _ { T + 1 , D } - f _ { \rho } \right\| _ { \rho } \leq \tilde { C } _ { 1 } \operatorname* { m a x } \left\{ | D | ^ { - \frac { r } { 2 r + s } } , \frac { | D | ^ { \frac { p + 1 } { 2 r + s } } } { \sigma ^ { 2 p } } \right\} \left( \log \frac { 4 8 } { \delta } \right) ^ { 6 } ,\tag{17}
$$

where $\tilde { C } _ { 1 }$ is a constant independent of |D| or $m _ { ; }$ , and will be given explicitly in the proof.

Here and throughout the paper, ⌈x⌉ denotes the smallest integer greater than or equal to x. As shown in Theorem 1, by choosing $\sigma \geq | D | ^ { \frac { p + 1 + r } { ( 2 r + s ) 2 p } }$ and neglecting the logarithmic factor, the high-probability upper bound in (17) yields the optimal learning rate $O ( | D | ^ { - \frac { r } { 2 r + s } } )$ in $L _ { \rho _ { X } } ^ { 2 }$ , which achieves the mini-max lower bound proved in [22, 23].

Remark 1. Theorem 1 establishes a high-probability learning rate for DKRGD under the regularity condition $r > { \frac { 1 } { 2 } }$ of $f _ { \rho } .$ . In the boundary case $\begin{array} { r } { r = { \frac { 1 } { 2 } } } \end{array}$ , where $f _ { \rho } \in \mathcal { H } _ { K }$ , the resulting bound incurs an additional logarithmic factor $\log | D |$ compared with (17). The logarithmic factor arises at the final stage of the error aggregation. Since the proof follows the same line as that of Theorem 1, the details are omitted.

We next turn to the generalization performance of DKRGD in expectation. Compared with the high-probability analysis, the expectation bound allows a weaker restriction on the number of local machines.

Theorem 2. Let $p > 0$ and $\begin{array} { r } { 0 < \eta \le \frac { 1 } { \kappa ^ { 2 } \operatorname* { m a x } \{ G _ { + } ^ { \prime } ( 0 ) , C _ { G } \} } } \end{array}$ . Under Assumption 1-3 with $r > \frac { 1 } { 2 }$ and $0 ~ < ~ s ~ \le ~ 1$ , if $\lambda = | D | ^ { - { \frac { 1 } { 2 r + s } } } , T = \Big \lceil | D | ^ { \frac { 1 } { 2 r + s } } \Big \rceil , | D _ { 1 } | = \cdot \cdot \cdot = | D _ { m } |$ and

$$
m \leq \left\{ \begin{array} { l l } { \vert D \vert ^ { \frac { 2 r + \frac { s } { 2 } - 1 } { 2 r + s } } \left( \left( \log \vert D \vert \right) ^ { 2 } + 1 \right) ^ { - 1 } , } & { \frac { 1 } { 2 } < r \leq 1 , } \\ { \vert D \vert ^ { \frac { 2 r - 1 } { 2 r + s } } \left( \left( \log \vert D \vert \right) ^ { 2 } + 1 \right) ^ { - 1 } , } & { 1 < r \leq \frac { 3 } { 2 } , } \\ { \vert D \vert ^ { \frac { 2 } { 2 r + s } } \left( \left( \log \vert D \vert \right) ^ { 2 } + 1 \right) ^ { - 1 } , } & { \frac { 3 } { 2 } < r \leq \frac { 5 - s } { 2 } , } \\ { \vert D \vert ^ { \frac { 2 r + s - 1 } { 4 r + 2 s } } ( ( \log \vert D \vert ) ^ { 8 } + 1 ) ^ { - 1 } , } & { r > \frac { 5 - s } { 2 } , } \end{array} \right.\tag{18}
$$

then

$$
\mathbb { E } \left[ \left. \bar { f } _ { T + 1 , D } - f _ { \rho } \right. _ { \rho } ^ { 2 } \right] \leq \tilde { C } _ { 2 } \operatorname* { m a x } \left\{ \left. D \right. ^ { - \frac { 2 r } { 2 r + s } } , \frac { \left. D \right. ^ { \frac { 2 p + 2 } { 2 r + s } } } { \sigma ^ { 4 p } } \right\} ,\tag{19}
$$

where ${ \tilde { C } } _ { 2 }$ is a constant independent $o f \left| D \right| \ o r \ m$ , and will be given explicitly in the proof.

Analogously, Theorem 2 establishes that with the choice of $\sigma \geq | D | ^ { \frac { p + 1 + r } { ( 2 r + s ) 2 p } }$ , one can obtain the optimal learning rate $O ( | D | ^ { - \frac { 2 r } { 2 r + s } } )$ from the expected error bound in (19). As a comparison between Theorem 1 and Theorem 2, we can see that the upper bound of (16) is tighter than that of (18) for all possible values of $r ,$ which shows a stricter restriction on m to guarantee the optimal learning rate with high probability than that in expectation. Note that when the regularity of $f _ { \rho }$ exceeds a certain level, the restriction on m in Theorem 1 and Theorem 2 reduce to the same order $| D | ^ { \frac { 2 r + s - 1 } { 4 r + 2 s } }$ , difering only by a logarithmic factor.

Remark 2. An logarithmic factor also arises in Theorem 2 for the expected error bound. At the critical level $\begin{array} { r } { r = { \frac { 1 } { 2 } } } \end{array}$ the resulting bound incurs an additional $\log ^ { 2 } | D |$ factor compared with (19). This quadratic logarithmic factor results from converting the high-probability estimate into an expectation bound: the logarithmic factor in the high-probability estimate appears squared when moving from tail probability to expectation bounds. The derivation closely parallels that of Theorem 2 and is omitted for brevity.

We finally consider DKRGD equipped with the communication strategy introduced in Subsection 2.2. The following theorem quantifies how communication can enlarge the admissible number of local machines while retaining the optimal high-probability learning rate.

Theorem 3. Let $\begin{array} { r } { 0 < \delta < 1 , \ : p > 0 , \ : 0 < \eta < \frac { 1 } { \kappa ^ { 2 } \operatorname* { m a x } \{ G _ { + } ^ { \prime } ( 0 ) , C _ { G } \} } } \end{array}$ and $l \in \mathbb { N } _ { + }$ . Under Assumption 1-3 with $r > \frac { 1 } { 2 }$ and $0 < s \leq 1 , \ i f \lambda = | D | ^ { - { \frac { 1 } { 2 r + s } } } , T = \Big \lceil | D | ^ { \frac { 1 } { 2 r + s } } \Big \rceil , | D _ { 1 } | = \cdot \cdot \cdot = | D _ { m } |$ and

$$
m \leq \operatorname* { m i n } \left\{ | D | ^ { \frac { 2 r - 1 } { 2 r + s } } , | D | ^ { \frac { ( 2 r - 1 ) ( l + 1 ) + s } { ( 2 r + s ) ( l + 2 ) } } \right\} \left( \left( \log | D | \right) ^ { 7 } + 1 \right) ^ { - 1 } ,\tag{20}
$$

then with confidence at least $1 - \delta ,$ there holds

$$
\left\| \bar { f } _ { T + 1 , D } ^ { l } - f _ { \rho } \right\| _ { \rho } \leq \tilde { C } _ { 3 } \operatorname* { m a x } \left\{ | D | ^ { - \frac { r } { 2 r + s } } , \frac { | D | ^ { \frac { p + 1 } { 2 r + s } } } { \sigma ^ { 2 p } } \right\} \left( \log \frac { 9 6 } { \delta } \right) ^ { 6 + 3 l } ,\tag{21}
$$

where ${ \tilde { C } } _ { 3 }$ is a constant independent of |D| or m, and will be given explicitly in the proof.

It should be pointed that the optimal learning rate $O ( | D | ^ { - \frac { r } { 2 r + s } } )$ can be also obtained by setting $\sigma \geq | D |$ <sup>p+1+r</sup>(2r+s)2p in (21). By comparing (16) with (20), the proposed communication strategy relaxes the restriction on m from order $| D | ^ { \frac { 2 r + s - 1 } { 4 r + 2 s } }$ to order min $\left\{ | D | ^ { \frac { 2 r - 1 } { 2 r + s } } , | D | ^ { \frac { ( 2 r - 1 ) ( l + 1 ) + s } { ( 2 r + s ) ( l + 2 ) } } \right\}$ up to a logarithmic factor. Further, the upper bound of m in (20) is increasing with respect to the number of communications l, and it tends to order $| D | ^ { \frac { 2 r - 1 } { 2 r + s } }$ up to a logarithmic factor as $l  \infty$ , which is significantly larger than the upper bound of m in (16) and coincides with that in [2, 3, 4, 6]. Although this limit is not better than that in Theorem 2 for every value of $r ,$ it should be highlighted that the optimal learning rate in Theorem 3 is achieved with high probability, which is stronger than that in expectation in Theorem 2. To conclude, Theorem 3 conducts the generalization analysis with high probability for DKRGD with communication strategy, which significantly relaxes the restriction on the number of local machines to ensure optimal learning rates, and thus enhances the practical applicability of DKRGD in distributed learning.

## 4 Related Work

As a special instance of spectral algorithms, gradient descent (GD) has received extensive attention in the literature. A comprehensive overview of gradient-based optimization methods is provided in [24], covering widely used variants of gradient descent, such as mini-batch and stochastic gradient descent, adaptive optimization methods including Adam and AdaMax, and distributed stochastic optimization frameworks such as Hogwild! and TensorFlow. In the context of nonparametric learning, the statistical properties of gradient descent have been rigorously investigated in [21, 25, 26], where optimal learning rates were established under suitable regularity and capacity conditions. More recently, [6] studied kernel-based gradient descent in a distributed setting and established optimal learning rates in probability. An important observation therein is that the intrinsic regularization properties of gradient descent can be exploited to alleviate the saturation phenomenon that typically arises in distributed spectral algorithms.

Despite these favorable theoretical properties, standard gradient descent based on the squared loss is generally sensitive to outliers and heavy-tailed noise, and its performance may deteriorate substantially in such settings. Consequently, several approaches have been developed to improve its robustness. Early-stopping strategies, which serve as an implicit regularization mechanism for controlling overfitting to noisy observations, have been extensively studied in [13, 27]. Alternatively, robust loss functions can be employed in place of the squared loss. In particular, [8] investigated gradient descent equipped with a broad class of robust losses, while [28] analyzed the convergence behavior of kernel gradient descent under the maximum correntropy criterion.

Motivated by these developments in robust gradient descent and distributed learning [5, 6, 8, 9, 10], we investigate the generalization performance of DKRGD, a problem that has received comparatively little theoretical attention to date. A particularly relevant starting point is [8], which demonstrates that robust gradient descent, combined with an appropriate early-stopping rule and a suitable choice of the scale parameter $\sigma ,$ achieves optimal convergence rates in $L _ { \rho _ { X } } ^ { 2 }$ . More precisely, under Assumptions 1–2 and an eigenvalue decay condition slightly stronger than Assumption 3, the following estimate holds with confidence at least 1 − δ:

$$
\| f _ { T + 1 } - f _ { \rho } \| _ { \rho } \leq \mathcal { O } \left( \operatorname* { m a x } \left\{ | D | ^ { - \frac { r } { 2 r + s } } , \frac { | D | ^ { \frac { p + 1 } { 2 r + s } } } { \sigma ^ { 2 p } } \right\} \left[ \log ( 6 / \delta ) \right] ^ { 3 } \right) .
$$

Consequently, choosing $\sigma \geq | D | ^ { \frac { p + 1 + r } { ( 2 r + s ) 2 p } }$ yields the optimal convergence rate

$$
O ( | D | ^ { - \frac { r } { 2 r + s } } )
$$

in the $L _ { \rho _ { X } } ^ { 2 }$ -norm. Under Assumptions 2–3 and a condition implied by Assumption 1, optimal learning rate in probability for robust gradient descent algorithms was established in [6]: if $r > \frac { 1 } { 2 }$ and $m \leq | D | ^ { \frac { r - \frac { 1 } { 2 } } { 2 r + s } } / [ ( \log | D | ) ^ { 5 } + 1 ]$ , then with

confidence at least $1 - \delta$ there holds

$$
\left\| \bar { f } _ { T + 1 , D } - f _ { \rho } \right\| _ { \rho } \leq \mathcal { O } \Bigl ( | D | ^ { - \frac { r } { 2 r + s } } \left[ \log ( 1 2 / \delta ) \right] ^ { 4 } \Bigr ) .
$$

This result shows that optimal learning rates can be retained in the distributed setting, provided that the number of local machines is suficiently controlled.

In contrast, by deriving sharper estimates for certain operator products through the concentration inequality developed in [20], we establish optimal learning rates in probability for DKRGD while substantially relaxing the restriction on the number of local machines. More specifically, the admissible number m of local machines can be enlarged to $| D | ^ { \frac { 2 r + s - 1 } { 4 r + 2 s } } / [ ( \log | D | ) ^ { 5 } + 1 ] ^ { 2 }$ . Moreover, the result of [6] permits a nontrivial number of local machines only when $\begin{array} { r } { r > \frac { 1 } { 2 } } \end{array}$ , and therefore does not cover the boundary case $\begin{array} { r } { r = { \frac { 1 } { 2 } } } \end{array}$ . This limitation is largely overcome by Theorem 1, although an additional logarithmic factor log $| D |$ appears in the resulting learning rate.

As for the optimal learning rates in expectation, we further loosen the restriction on m. Compared with [3], the maximum number m is relaxed slightly from $| D | ^ { \frac { 2 r - 1 } { 2 r + s } }$ to $| D | ^ { \frac { 2 r + \frac { s } { 2 } - 1 } { 2 r + s } } [ \left( \log | D | \right) ^ { 2 } + 1 ] ^ { - 1 }$ when $\textstyle { \frac { 1 } { 2 } } \leq r \leq 1$ . In addition, the saturation phenomenon inherent to regularized least squares [3] is efectively mitigated by leveraging the results in Theorem 1 and the standard tail-expectation formula $\begin{array} { r } { \mathbb { E } [ \xi ] = \int _ { 0 } ^ { \infty } \mathrm { P r o b } ( \xi > t ) \mathrm { d } t } \end{array}$ . Specifically, while maintaining optimal learning rates in expectation, the admissible upper bound on the number of local machines can scale up with increasing r when $r > ( 5 - s ) / 2$

In distributed learning, numerous studies have focused on relaxing the restriction on the number of local machines by utilizing unlabeled data [6, 21] or introducing communication strategies [5, 14, 16]. Such approaches can compensate for the statistical loss caused by data partitioning and thereby improve the practical applicability of distributed algorithms. For nonparametric regression, the most relevant work is [5], in which a communication strategy based on the Newton-Raphson iteration is proposed for distributed kernel ridge regression. Inspired by this approach, we propose a communication-enhanced version of DKRGD that further relaxes the restriction on the number of local machines without requiring additional unlabeled data. Although our condition on the number of local machines for attaining the optimal learning rate is slightly more restrictive than that obtained in [5], our approach ofers two complementary advantages. First, owing to the intrinsic regularization properties of gradient descent, our theoretical results extend beyond the usual range $1 / 2 \le r \le 1$ . More precisely, while the results in [5] are restricted to $1 / 2 \le r \le 1$ Theorem 3 applies to the entire regime $r \ > \ \frac 1 2$ . Second, the use of a robust loss makes the resulting distributed algorithm more resistant to outliers and heavy-tailed noise. It would also be interesting to investigate whether the proposed communication strategy can be adapted to other distributed learning frameworks, such as deep distributed convolutional neural networks [29] and distributed gradient descent functional learning [30].

## 5 Error decomposition

In this section, we conduct the error analysis of DKRGD using the integral operator approach [3, 5, 6, 31, 32]. We first introduce several preliminary lemmas and propositions that will be repeatedly used in the subsequent analysis. We then derive two distinct error decompositions for DKRGD, which serve as the basis for establishing generalization bounds both with high probability and in expectation, respectively. Finally, we develop a corresponding error decomposition for DKRGD with communication and use it to derive a high-probability generalization bound.

## 5.1 Preliminaries

As the intermediate function in the error decomposition, we first introduce the data-free sequence $\{ f _ { t } \} _ { t \ge 1 }$ which can be regarded as the limit of $\{ f _ { t , D } \} _ { t \geq 1 }$ when the sample size $| D |$ goes to infinity, and then we will derive two representations for $f _ { t + 1 }$ by using the integral operator approach. Specifically, it is defined as follows,

$$
f _ { t + 1 } = f _ { t } - \eta \int _ { \mathcal { Z } } G ^ { \prime } ( \xi _ { t , \sigma } ( z ) ) ( f _ { t } ( x ) - y ) K _ { x } \mathrm { d } \rho ,\tag{22}
$$

where $f _ { 1 } = 0 , 1 \leq t \leq T$ and $\begin{array} { r } { \xi _ { t , \sigma } ( z ) = \frac { ( y - f _ { t } ( x ) ) ^ { 2 } } { \sigma ^ { 2 } } , ( x , y ) \in D } \end{array}$ . For the simplicity of subsequent calculations, we can further rewrite $f _ { t + 1 }$ as

$$
\begin{array} { l } { f _ { t + 1 } = f _ { t } - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } ( f _ { t } - f _ { \rho } ) + \eta \displaystyle \int _ { \mathcal { Z } } ( G _ { + } ^ { \prime } ( 0 ) - G ^ { \prime } ( \xi _ { t , \sigma } ( z ) ) ) ( f _ { t } ( x ) - y ) K _ { x } \mathrm { d } \rho } \\ { \mathrm { ~ } } \\ { = \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) f _ { t } + \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } f _ { \rho } + \eta E _ { t , \sigma } , } \end{array}
$$

and

$$
f _ { t + 1 } = \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) f _ { t } + \eta G _ { + } ^ { \prime } ( 0 ) ( L _ { K , D } - L _ { K } ) f _ { t } + \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } f _ { \rho } + \eta E _ { t , \sigma } ,
$$

where $\begin{array} { r } { E _ { t , \sigma } = \int _ { \mathcal Z } ( G _ { + } ^ { \prime } ( 0 ) - G ^ { \prime } ( \xi _ { t , \sigma } ( z ) ) ) ( f _ { t } ( x ) - y ) K _ { x } \mathrm { d } \rho . } \end{array}$

Analogous to Equation (7) and (8), we can employ mathematical induction to derive the following two representations for $f _ { t + 1 }$ as

$$
f _ { t + 1 } = \sum _ { i = 1 } ^ { t } \eta G _ { + } ^ { \prime } ( 0 ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { t - i } L _ { K } f _ { \rho } + \sum _ { i = 1 } ^ { t } \eta \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { t - i } E _ { i , \sigma } ,\tag{23}
$$

and

$$
\begin{array} { l } { { f _ { t + 1 } = \displaystyle \sum _ { i = 1 } ^ { t } \eta G _ { + } ^ { \prime } ( 0 ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { t - i } L _ { K } f _ { \rho } + \displaystyle \sum _ { i = 1 } ^ { t } \eta G _ { + } ^ { \prime } ( 0 ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { t - i } \left( L _ { K , D } - L _ { K } \right) f _ { 2 } + \displaystyle \sum _ { i = 1 } ^ { t } \eta G _ { - } ^ { \prime } ( 0 ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { t - i } \left( L _ { K , D } - L _ { K } \right) f _ { 2 } } } \\ { { \displaystyle \qquad + \displaystyle \sum _ { i = 1 } ^ { t } \eta \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { t - i } E _ { i , \sigma } } . } \end{array}\tag{24}
$$

By combining (7) and (24), it is easy to obtain the representation for the diference between $f _ { T + 1 , D }$ and $f _ { T + 1 }$ as

$$
f _ { T + 1 , D } - f _ { T + 1 } = \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime \prime } ( 0 ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { T - i } \left\{ \hat { f } _ { K , D } - L _ { K } f _ { \rho } + \frac { 1 } { G _ { + } ^ { \prime } ( 0 ) } ( E _ { i , D , \sigma } - E _ { i , \sigma } ) + ( L _ { K } - L _ { K , D } ) f _ { i } \right\} .\tag{25}
$$

Moreover, we can combine (8) and (23) to yield another representation for $f _ { T + 1 , D } - f _ { T + 1 }$ as

$$
f _ { Y + 1 , D } - f _ { Y + 1 } = \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { T - i } \left\{ \hat { f } _ { K , D } - L _ { K } f _ { \rho } + \frac { 1 } { G _ { + } ^ { \prime } ( 0 ) } ( E _ { i , D , \sigma } - E _ { i , \sigma } ) + ( L _ { K } - L _ { K , D } ) f _ { i , D } \right\} .\tag{26}
$$

Building on the operator representations introduced above, we now establish key estimates for operator products and the data-free iterative sequence $\{ f _ { t } \} _ { t \ge 1 }$ , which form the foundation of our subsequent error analysis. Throughout this paper, we simply denote the operator norm as $\| \cdot \|$ . The following two lemmas are obtained by adapting the proof in [8, Prop. 4.1, Thm. 1(i)] to accommodate our constant step-size η. For brevity, detailed proofs are deferred to the Appendix.

Lemma 1. Define $\{ f _ { t } \} _ { t \ge 1 }$ as (22) and $\begin{array} { r } { f _ { 1 } = 0 . ~ I f 0 < \eta \le \frac { 1 } { \kappa ^ { 2 } \operatorname* { m a x } \{ G _ { + } ^ { \prime } ( 0 ) , C _ { G } \} } } \end{array}$ , then

$$
\| f _ { t } \| _ { K } \leq M \sqrt { \eta C _ { G } ( t - 1 ) } \leq \frac { M } { \kappa } ( t - 1 ) ^ { \frac { 1 } { 2 } } , \quad \forall t \geq 1 .\tag{27}
$$

Lemma 2. For $\begin{array} { r } { \lambda > 0 , 0 < \eta \leq \frac { 1 } { \kappa ^ { 2 } \operatorname* { m a x } \{ G _ { + } ^ { \prime } ( 0 ) , C _ { G } \} } } \end{array}$ , and $T \in \mathbb { N }$ , we have

$$
\begin{array} { r l } & { \quad \operatorname* { m a x } \left\{ \left\| \displaystyle \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) ( \lambda I + L _ { K } ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { T - i } \right\| , \left\| \displaystyle \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime \prime } ( 0 ) ( \lambda I + L _ { K , D } ) \left( I - \eta G _ { + } ^ { \prime \prime } ( 0 ) L _ { K , D } \right) ^ { T - i } \right\| \right\} } \\ & { \le \eta G _ { + } ^ { \prime } ( 0 ) \lambda T + 1 . } \end{array}
$$

In the end of this subsection, we derive an error decomposition based on (25), which is crucial for the subsequent error analysis of DKRGD in both the high-probability and expectation settings.

Proposition 1. Let $\begin{array} { r } { \lambda > 0 , 0 < \eta \leq \frac { 1 } { \kappa ^ { 2 } \operatorname* { m a x } \{ G _ { + } ^ { \prime } ( 0 ) , C _ { G } \} } } \end{array}$ , and $T \in \mathbb N$ . If Assumption 2 holds for $r \geq { \frac { 1 } { 2 } }$ , we have

$$
\begin{array} { l } { \displaystyle \left\| { ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } ( f _ { T + 1 , D } - f _ { T + 1 } ) } \right\| _ { K } \le ( 1 + \eta G _ { + } ^ { \prime } ( 0 ) \lambda T ) A _ { D , \lambda } ^ { 2 } ( \mathcal { P } _ { D , \lambda } + \mathcal { Q } _ { D , \lambda } \left\| f _ { \rho } \right\| _ { K } ) } \\ { \displaystyle \qquad + \sum _ { i = 1 } ^ { T } \eta \left\| { ( \lambda I + L _ { K , D } ) ^ { \frac { 1 } { 2 } } \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { T - i } } \right\| A _ { D , \lambda } \left\| E _ { i , D , \sigma } - E _ { i , \sigma } \right\| _ { K } } \\ { \displaystyle \qquad + \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \left\| { ( \lambda I + L _ { K , D } ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { T - i } } \right\| A _ { D , \lambda } ^ { 2 } \mathcal { Q } _ { D , \lambda } \left\| f _ { i } - f _ { \rho } \right\| _ { K } , } \end{array}\tag{28}
$$

where

$$
\begin{array} { r l } & { { \cal A } _ { D , \lambda } = \Big \| ( \lambda I + { \cal L } _ { K } ) ^ { \frac { 1 } { 2 } } ( \lambda I + { \cal L } _ { K , D } ) ^ { - \frac { 1 } { 2 } } \Big \| , } \\ & { { \cal P } _ { D , \lambda } = \Big \| ( \lambda I + { \cal L } _ { K } ) ^ { - \frac { 1 } { 2 } } ( \hat { f } _ { K , D } - { \cal L } _ { K } f _ { \rho } ) \Big \| _ { K } , } \\ & { { \cal Q } _ { D , \lambda } = \Big \| ( \lambda I + { \cal L } _ { K } ) ^ { - \frac { 1 } { 2 } } ( { \cal L } _ { K } - { \cal L } _ { K , D } ) \Big \| . } \end{array}
$$

Proof. With the representation (25) for $f _ { T + 1 , D } - f _ { T + 1 }$ , we have

$$
\begin{array} { r l } & { \| ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } ( f _ { 1 ^ { 1 } \bot , D } - f _ { \Gamma ^ { \prime } \bot } ) \| _ { K } \leq \mathcal { A } f _ { \kappa , \lambda } \Bigg \| \displaystyle \sum _ { i = 1 } ^ { \infty } \eta \mathcal { C } _ { + } ^ { \frac { 1 } { 2 } } ( \mathbb { I } ) ( \lambda I + L _ { K , D } ) ( I - \eta \mathcal { C } _ { + } ^ { \frac { 1 } { 2 } } ( 0 ) I \mathbb { R } _ { K , D } ) ^ { \frac { 1 } { 2 } - \lambda } , } \\ & { \qquad \quad ( \lambda I - L _ { K , D } ) ^ { - \frac { 1 } { 2 } } \Bigg \{ \displaystyle \hat { f } _ { k , D } - I _ { K , D } f _ { + } \frac { R _ { i , D , D } - E _ { k , i , \omega } } { \sqrt { \pi } } + ( I _ { K } - L _ { K , D } ) f _ { k } \Bigg \} \Bigg \| _ { K } } \\ & { \leq \mathcal { A } _ { \sigma , \lambda } \Bigg \{ \Bigg \| \displaystyle \sum _ { i = 1 } ^ { \infty } \eta ( \lambda I + L _ { K , D } ) ( I - \eta \mathcal { C } _ { + } ^ { \frac { 1 } { 2 } } ( 0 ) L _ { K , D } ) ^ { \frac { 1 } { 2 - \alpha } } ( \lambda I + L _ { K , D } ) ^ { - \frac { 1 } { 2 } } ( \mathbb { E } _ { i , \sigma , - } ^ { \frac { 1 } { 2 } } ( \hat { E } _ { i , \sigma , - } ^ { \frac { 1 } { 2 } } L _ { K } ) \Bigg \| _ { K } } \\ &  \qquad + \| \displaystyle \sum _ { i = 1 } ^ { \infty } \eta \mathcal { C } _ { - } ^ { \frac { 1 } { 2 } } ( 0 ) ( \lambda I + L _ { K , D } ) ( I - \eta \mathcal { C } _ { + } ^ { \frac { 1 } { 2 } } ( 0 ) L _ { K , D } ) ^ { \frac { 1 } { 2 - \alpha } } ( \lambda I + L _ { K , D } ) ^  - \end{array}\tag{29}
$$

For $A _ { 1 }$ , applying the triangle inequality of K-norm, we have

$$
A _ { 1 } \leq \sum _ { i = 1 } ^ { T } \eta \left\| ( \lambda I + L _ { K , D } ) ^ { \frac { 1 } { 2 } } \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { T - i } \right\| \| E _ { i , D , \sigma } - E _ { i , \sigma } \| _ { K } .
$$

To estimate $A _ { 2 } ,$ Lemma $2 \textrm { y }$ ields

$$
\begin{array} { r l } & { A _ { 2 } \leq \left\| \displaystyle \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) ( \lambda I + L _ { K , D } ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { T - i } \right\| \left\| ( \lambda I + L _ { K , D } ) ^ { - \frac { 1 } { 2 } } ( \hat { f } _ { K , D } - L _ { K } f _ { \rho } ) \right\| _ { K } } \\ & { \quad \leq ( 1 + \eta G _ { + } ^ { \prime } ( 0 ) \lambda T ) \left\| ( \lambda I + L _ { K , D } ) ^ { - \frac { 1 } { 2 } } ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } \right\| \left\| ( \lambda I + L _ { K } ) ^ { - \frac { 1 } { 2 } } ( \hat { f } _ { K , D } - L _ { K } f _ { \rho } ) \right\| _ { K } } \\ & { \quad = ( 1 + \eta G _ { + } ^ { \prime } ( 0 ) \lambda T ) \mathcal { A } _ { D , \lambda } \mathcal { P } _ { D , \lambda } , } \end{array}
$$

where the last equality holds since $\| L _ { 1 } L _ { 2 } \| = \| ( L _ { 1 } L _ { 2 } ) ^ { * } \| = \| L _ { 2 } L _ { 1 } \|$ for any self-adjoint operators $L _ { 1 }$ and $L _ { 2 }$ on Hilbert spaces. Concerning $A _ { 3 }$ , by the decomposition $f _ { i } = f _ { i } - f _ { \rho } + f _ { \rho }$ , we have

$$
\begin{array} { l } { { \displaystyle { \cal A } _ { 3 } \leq \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \left\| \left( \lambda I + L _ { K , D } \right) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { T - i } \right\| A _ { D , \lambda } Q _ { D , \lambda } \| f _ { i } - f _ { \rho } \| _ { K } } } \\ { { \displaystyle ~ + \left\| \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) ( \lambda I + L _ { K , D } ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { T - i } \right\| A _ { D , \lambda } Q _ { D , \lambda } \| f _ { \rho } \| _ { K } } } \\ { { \displaystyle \leq \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \left\| \left( \lambda I + L _ { K , D } \right) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { T - i } \right\| A _ { D , \lambda } Q _ { D , \lambda } \| f _ { i } - f _ { \rho } \| _ { K } + \left( 1 + \eta G _ { + } ^ { \prime } ( 0 ) \lambda T \right) A _ { D , \lambda } Q _ { D , \lambda } \| f _ { \rho } \| _ { K } . } } \end{array}
$$

Finally, the proof of Proposition 1 is completed by substituting the bounds on $A _ { 1 } , A _ { 2 }$ , and $A _ { 3 }$ into (29).

## 5.2 Error decomposition I for DKRGD

To derive the generalization bounds in probability, we establish an error decomposition for DKRGD that will be used to prove Theorem 1. Specifically, by introducing the data-free sequence $\{ f _ { t } \} _ { t \ge 1 }$ , we decompose the error $\left\| \bar { f } _ { T + 1 , D } - f _ { \rho } \right\| _ { \rho }$ into two parts: $\left\| \bar { f } _ { T + 1 , D } - f _ { T + 1 } \right\| _ { \rho }$ and $\| f _ { T + 1 } - f _ { \rho } \| _ { \rho }$ . These two components are then estimated in the following two propositions, respectively.

Proposition 2. For $\begin{array} { r } { \lambda > 0 , 0 < \eta \leq \frac { 1 } { \kappa ^ { 2 } \operatorname* { m a x } \{ G _ { + } ^ { \prime } ( 0 ) , C _ { G } \} } } \end{array}$ , and $T \in \mathbb { N }$ . If Assumption 2 holds for $\begin{array} { r } { r \geq \frac { 1 } { 2 } } \end{array}$ , we have

$$
\begin{array} { r l } { \Big \| \bar { f } _ { T + 1 , D } - f _ { T + 1 } \Big \| _ { \rho } \leq ( 1 + \eta G _ { + } ^ { \prime } ( 0 ) \lambda T ) ( \mathcal { P } _ { D , \lambda } + \mathcal { Q } _ { D , \lambda } \| f _ { \rho } \| _ { K } ) } & { } \\ { \quad } & { \quad + \displaystyle \sum _ { i = 1 } ^ { T } \eta \left\| ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { T - i } \right\| \displaystyle \sum _ { j = 1 } ^ { m } \frac { | D _ { j } | } { | D | } \left\| E _ { i , D _ { j } , \sigma } - E _ { i , \sigma } \right\| _ { K } } \\ { \quad } & { \quad + \displaystyle \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \left\| \big ( \lambda I + L _ { K } \big ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { T - i } \right\| \mathcal { Q } _ { D , \lambda } \| f _ { i } - f _ { \rho } \| _ { K } + \displaystyle \operatorname* { m a x } _ { 1 \leq j \leq m } K _ { D _ { j } , \lambda } , } \end{array}
$$

where

$$
\begin{array} { r l } {  { K _ { D _ { j } , \lambda } = \sum _ { i = 2 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \| ( \lambda I + L _ { K } ) ( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } ) ^ { T - i } \| \mathcal { R } _ { D _ { j } , \lambda } A _ { D _ { j } , \lambda } . } } \\ & { \quad \quad \quad \quad \quad \quad \Bigg \{ ( 1 + \eta G _ { + } ^ { \prime } ( 0 ) \lambda ( i - 1 ) ) A _ { D _ { j } , \lambda } ( \mathcal { P } _ { D _ { j } , \lambda } + \mathcal { Q } _ { D _ { j } , \lambda } | | f _ { \rho } | | _ { K } ) } \\ & { \quad \quad \quad \quad \quad + \displaystyle \sum _ { l = 1 } ^ { i - 1 } \eta \| ( \lambda I + L _ { K , D _ { j } } ) ^ { \frac { 1 } { 2 } } ( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } ) ^ { i - l - 1 } \| \| E _ { l , D _ { j } , \sigma } - E _ { l , \sigma } \| _ { K } } \\ & { \quad \quad \quad \quad + \displaystyle \sum _ { l = 1 } ^ { i - 1 } \eta G _ { + } ^ { \prime } ( 0 ) \| ( \lambda I + L _ { K , D _ { j } } ) ( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } ) ^ { i - l - 1 } \| \mathcal { A } _ { D _ { j } , \lambda } \mathcal { Q } _ { D _ { j } , \lambda } \| f _ { l } - f _ { \rho } \| _ { K } \Bigg \} , } \end{array}
$$

and

$$
\mathscr { R } _ { D , \lambda } : = \left\| ( \lambda I + L _ { K } ) ^ { - \frac 1 2 } ( L _ { K } - L _ { K , D } ) ( \lambda I + L _ { K } ) ^ { - \frac 1 2 } \right\| .
$$

Proof. Based on the definition (5) of $\bar { f } _ { t , D }$ , we can apply (26) to $D _ { j }$ for each fixed $j \in { 1 , . . . , m }$ to have that

$$
\begin{array} { l } { \displaystyle \bar { f } _ { T + 1 , D } - f _ { T + 1 } = \sum _ { j = 1 } ^ { m } \frac { \left| D _ { j } \right| } { \left| D \right| } \left( f _ { T + 1 , D _ { j } } - f _ { T + 1 } \right) } \\ { = \displaystyle \sum _ { j = 1 } ^ { m } \frac { \left| D _ { j } \right| } { \left| D \right| } \left\{ \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } \left( 0 \right) \left( I - \eta G _ { + } ^ { \prime } \left( 0 \right) L _ { K } \right) ^ { T - i } \left[ \hat { f } _ { K , D _ { j } } - L _ { K } f _ { \rho } + \frac { 1 } { G _ { + } ^ { \prime } \left( 0 \right) } \left( E _ { i , D _ { j } , \sigma } - E _ { i , \sigma } \right) + \left( L _ { K } - L _ { K , D _ { j } } \right) f _ { i , D _ { j } } \right] \right\} . } \end{array}
$$

Since $\begin{array} { r } { \sum _ { j = 1 } ^ { m } \frac { | D _ { j } | } { | D | } = 1 } \end{array}$ and

$$
\sum _ { j = 1 } ^ { m } \frac { | D _ { j } | } { | D | } \hat { f } _ { K , D _ { j } } = \sum _ { j = 1 } ^ { m } \frac { | D _ { j } | } { | D | } \frac { 1 } { | D _ { j } | } \sum _ { ( x _ { i } , y _ { i } ) \in D _ { j } } y _ { i } K _ { x _ { i } } = \frac { 1 } { | D | } \sum _ { ( x _ { i } , y _ { i } ) \in D } y _ { i } K _ { x _ { i } } = \hat { f } _ { K , D } ,
$$

we have

$$
\begin{array} { r l } { \displaystyle \left\| \bar { f } _ { T + 1 , D } - f _ { T + 1 } \right\| _ { \rho } \leq } & { \displaystyle \left\| \displaystyle \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { T - i } ( \hat { f } _ { K , D } - L _ { K } f _ { \rho } ) \right\| _ { \rho } } \\ & { \quad \quad \quad + \displaystyle \left\| \displaystyle \sum _ { i = 1 } ^ { T } \eta \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { T - i } \displaystyle \sum _ { j = 1 } ^ { m } \frac { | D _ { j } | } { | D | } ( E _ { i , D _ { j } , \sigma } - E _ { i , \sigma } ) \right\| _ { \rho } } \\ & { \quad \quad \quad + \displaystyle \left\| \displaystyle \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { T - i } \displaystyle \sum _ { j = 1 } ^ { m } \frac { | D _ { j } | } { | D | } ( L _ { K } - L _ { K , D _ { j } } ) f _ { i , D _ { j } } \right\| _ { \rho } } \\ & { = : I _ { 1 } + I _ { 2 } + I _ { 3 } . } \end{array}\tag{30}
$$

Since $\left\| f \right\| _ { \rho } = \left\| L _ { K } ^ { \frac 1 2 } f \right\| _ { K }$ for any $f \in { \mathcal { H } } _ { K } { \mathrm { ~ a n d ~ } } \left\| L _ { K } ^ { \frac { 1 } { 2 } } ( \lambda I + L _ { K } ) ^ { - { \frac { 1 } { 2 } } } \right\| \leq 1$ for any $\lambda > 0$ , we have

$$
\begin{array} { r l } & { I _ { 1 } = \left\| \displaystyle \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) ( \lambda I + L _ { K } ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { T - i } \right\| \left\| ( \lambda I + L _ { K } ) ^ { - 1 } ( \hat { f } _ { K , D } - L _ { K } f _ { \rho } ) \right\| _ { \rho } } \\ & { \quad \leq \left\| \displaystyle \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) ( \lambda I + L _ { K } ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { T - i } \right\| \left\| ( \lambda I + L _ { K } ) ^ { - \frac 1 2 } ( \hat { f } _ { K , D } - L _ { K } f _ { \rho } ) \right\| _ { K } } \\ & { \quad \leq ( 1 + \eta G _ { + } ^ { \prime } ( 0 ) \lambda T ) \mathcal { P } _ { D , \lambda } , } \end{array}\tag{31}
$$

where the last inequality holds by Lemma 2. For $I _ { 2 } .$ it’s easy to derive that

$$
I _ { 2 } \leq \sum _ { i = 1 } ^ { T } \eta \left\| ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { T - i } \right\| \sum _ { j = 1 } ^ { m } \frac { | D _ { j } | } { | D | } \left\| E _ { i , D _ { j } , \sigma } - E _ { i , \sigma } \right\| _ { K } .\tag{32}
$$

For $I _ { 3 } ,$ , it can be bounded by decomposing $f _ { i , D _ { j } }$ as $f _ { i , D _ { j } } = f _ { i , D _ { j } } - f _ { i } + f _ { i } - f _ { \rho } + f _ { \rho }$ as follows

$$
\begin{array} { l } { { I _ { 3 } \le \displaystyle \left\| \displaystyle \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) ( \lambda I + L _ { K } ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { T - i } \displaystyle \sum _ { j = 1 } ^ { m } \frac { | D _ { j } | } { | D | } ( \lambda I + L _ { K } ) ^ { - \frac 1 2 } ( L _ { K } - L _ { K , D _ { j } } ) ( f _ { i , D _ { j } } - f _ { i } ) \right\| _ { K } } } \\ { { + \left\| \displaystyle \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) ( \lambda I + L _ { K } ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { T - i } ( \lambda I + L _ { K } ) ^ { - \frac 1 2 } ( L _ { K } - L _ { K , D } ) ( f _ { i } - f _ { \rho } ) \right\| _ { K } } } \\ { { + \left\| \displaystyle \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) ( \lambda I + L _ { K } ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { T - i } ( \lambda I + L _ { K } ) ^ { - \frac 1 2 } ( L _ { K } - L _ { K , D } ) f _ { \rho } \right\| _ { K } } } \\ { { = : I _ { 3 , 1 } + I _ { 3 , 2 } + I _ { 3 , 3 } , } } \end{array}\tag{33}
$$

where $I _ { 3 , 2 }$ and $I _ { 3 , 3 }$ is derived from the fact that for any $f \in \mathcal { H } _ { K }$ , there holds

$$
\sum _ { j = 1 } ^ { m } \frac { | D _ { j } | } { | D | } L _ { K , D _ { j } } f = \sum _ { j = 1 } ^ { m } \frac { | D _ { j } | } { | D | } \frac { 1 } { | D _ { j } | } \sum _ { x _ { i } \in D _ { j } } f ( x _ { i } ) K _ { x _ { i } } = \frac { 1 } { | D | } \sum _ { x _ { i } \in D } f ( x _ { i } ) K _ { x _ { i } } = L _ { K , D } f .
$$

For $I _ { 3 , 2 }$ , since $f _ { i } \in \mathcal { H } _ { K }$ and Assumption 2 with $r \geq { \frac { 1 } { 2 } }$ implies $f _ { \rho } \in \mathcal { H } _ { K }$ , then

$$
I _ { 3 , 2 } \leq \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \left\| \left( \lambda I + L _ { K } \right) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { T - i } \right\| \mathcal { Q } _ { D , \lambda } \left\| f _ { i } - f _ { \rho } \right\| _ { K } .\tag{34}
$$

Analogously, $I _ { 3 , 3 }$ can be bounded by Lemma 2 as

$$
\begin{array} { r l } & { I _ { 3 , 3 } \leq \left\| \displaystyle \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) ( \lambda I + L _ { K } ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { T - i } \right\| \left\| ( \lambda I + L _ { K } ) ^ { - \frac 1 2 } ( L _ { K } - L _ { K , D } ) \right\| \| f _ { \rho } \| _ { K } } \\ & { \quad = ( 1 + \eta G _ { + } ^ { \prime } ( 0 ) \lambda T ) Q _ { D , \lambda } \| f _ { \rho } \| _ { K } . } \end{array}\tag{35}
$$

To bound ${ { I } _ { 3 , 1 } }$ , we use $f _ { 1 } = f _ { 1 , D } = 0$ and Jensen’s inequality to obtain

$$
\begin{array} { l } { { \displaystyle { \cal I } _ { 3 , 1 } \leq \sum _ { i = 2 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \left\| \left( \lambda I + L _ { K } \right) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { T - i } \right\| \left\| \displaystyle { \sum _ { j = 1 } ^ { m } \frac { | D _ { j } | } { | D | } ( \lambda I + L _ { K } ) ^ { - \frac { 1 } { 2 } } ( L _ { K } - L _ { K , D _ { j } } ) ( f _ { i , D _ { j } } - f _ { i } ) } \right\| _ { K } } } \\ { { \displaystyle \quad \leq \sum _ { j = 1 } ^ { m } \displaystyle { \frac { | D _ { j } | } { | D | } \displaystyle \sum _ { i = 2 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \left\| \left( \lambda I + L _ { K } \right) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { T - i } \right\| } . } } \\ { { \displaystyle \quad \quad { \left\| \left( \lambda I + L _ { K } \right) ^ { - \frac { 1 } { 2 } } ( L _ { K } - L _ { K , D _ { j } } ) ( \lambda I + L _ { K } ) ^ { - \frac { 1 } { 2 } } \right\| \left\| \left( \lambda I + L _ { K } \right) ^ { \frac { 1 } { 2 } } ( f _ { i , D _ { j } } - f _ { i } ) \right\| _ { K } } } } \\ { { \displaystyle \quad \leq \displaystyle \operatorname* { m a x } _ { 1 \leq j \leq m } \displaystyle { \sum _ { i = 2 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \left\| \left( \lambda I + L _ { K } \right) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { T - i } \right\| \mathcal { R } _ { D , \lambda } \left\| \left( \lambda I + L _ { K } \right) ^ { \frac { 1 } { 2 } } ( f _ { i , D _ { j } } - f _ { i } ) \right\| _ { K } } . } } \end{array}
$$

Recall the proof of Proposition 1 with D and $T + 1$ replaced by $D _ { j }$ and i, we have

$$
\begin{array} { r l } { \displaystyle \left\| ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } ( f _ { i , D _ { j } } - f _ { i } ) \right\| _ { K } \le ( 1 + \eta G _ { + } ^ { \prime } ( 0 ) \lambda ( i - 1 ) ) A _ { D _ { j , \lambda } } ^ { 2 } ( { \mathcal P } _ { D _ { j } , \lambda } + { \mathcal Q } _ { D _ { j } , \lambda } \| f _ { \rho } \| _ { K } ) } & { } \\ { \displaystyle } & { \qquad + \sum _ { l = 1 } ^ { i - 1 } \eta \left\| ( \lambda I + L _ { K , D _ { j } } ) ^ { \frac { 1 } { 2 } } \left( I - \eta G _ { + } ^ { \prime \prime } ( 0 ) L _ { K , D _ { j } } \right) ^ { i - l - 1 } \right\| A _ { D _ { j } , \lambda } \left\| E _ { l , D _ { j } , \sigma } - E _ { l , \sigma } \right\| _ { K } } \\ { \displaystyle } & { \qquad + \sum _ { l = 1 } ^ { i - 1 } \eta G _ { + } ^ { \prime } ( 0 ) \left\| ( \lambda I + L _ { K , D _ { j } } ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } \right) ^ { i - l - 1 } \right\| A _ { D _ { j } , \lambda } ^ { 2 } { \mathcal Q } _ { D _ { j } , \lambda } \| f _ { l } - f _ { \rho } \| _ { K } . } \end{array}
$$

It follows that

$$
\begin{array} { r l } {  { { \cal I } _ { 3 , 1 } \leq \operatorname* { m a x } _ { 1 \leq j \leq m } \sum _ { i = 2 } ^ { T } \eta G _ { + } ^ { \prime \prime } ( 0 ) \| ( \lambda T + L _ { K } ) ( T - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } ) ^ { T - i } \| \mathcal { R } _ { D , \lambda } \mathcal { A } _ { D , \lambda } , } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & & { \quad \quad \quad \quad \quad \quad \quad \quad + \sum _ { l = 1 } ^ { m } \eta \| ( \lambda T + L _ { K , D _ { s } } ) ^ { \frac { 1 } { 2 } } ( T - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { s } } ) ^ { i - l - 1 } \| \| \mathcal { R } _ { t , D _ { j } , \sigma } - E _ { t , \sigma } \| _ { K } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ &  \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \ \end{array}\tag{36}
$$

By substituting (34), (35) and (36) into (33), we have

$$
\begin{array} { l } { { \displaystyle I _ { 3 } \le \operatorname* { m a x } _ { 1 \le j \le m } \kappa _ { D _ { j } , \lambda } + ( 1 + \eta G _ { + } ^ { \prime } ( 0 ) \lambda T ) \mathcal { Q } _ { D , \lambda } \| f _ { \rho } \| _ { K } } } \\ { { \displaystyle \qquad + \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \left\| \big ( \lambda I + L _ { K } \big ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { T - i } \right\| \mathcal { Q } _ { D , \lambda } \| f _ { i } - f _ { \rho } \| _ { K } . } } \end{array}\tag{37}
$$

Finally, the proof of Proposition 2 is completed by substituting (31), (32) and (37) into (30).

We are now ready to bound the generalization error $\| f _ { T + 1 } - f _ { \rho } \| _ { \rho } ,$ as formalized in the following proposition. The proof adapts the analytical framework established in [13, 33] to our setting.

Proposition 3. Define $\{ f _ { t } \} _ { t \ge 1 }$ as (22) and $f _ { 1 } = 0$ . If Assumption 2 holds for $r \geq { \frac { 1 } { 2 } }$ , then for any $t \geq 2$ , we have

$$
\| f _ { t } - f _ { \rho } \| _ { K } \leq \left( \frac { r - \frac { 1 } { 2 } } { e \eta G _ { + } ^ { \prime } ( 0 ) } \right) ^ { r - \frac { 1 } { 2 } } \| u _ { \rho } \| _ { \rho } ( t - 1 ) ^ { - ( r - \frac { 1 } { 2 } ) } + \sum _ { i = 1 } ^ { t - 1 } \eta \| E _ { i , \sigma } \| _ { K } ,\tag{38}
$$

and

$$
\| f _ { t } - f _ { \rho } \| _ { \rho } \leq \left( \frac { r } { e \eta G _ { + } ^ { \prime } ( 0 ) } \right) ^ { r } \| u _ { \rho } \| _ { \rho } ( t - 1 ) ^ { - r } + \sum _ { i = 1 } ^ { t - 1 } \eta \left\| ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { t - i - 1 } \right\| \| E _ { i , \sigma } \| _ { K } .\tag{39}
$$

Proof. Firstly, for the sake of convenience, we adopt the notation $g _ { t } ( \cdot )$ defined in (9). Since for any $t \geqslant 2$

$$
\begin{array} { l } { { g _ { t - 1 } ( x ) \cdot x = \displaystyle \sum _ { k = 1 } ^ { t - 1 } \eta G _ { + } ^ { \prime } ( 0 ) x \left( 1 - \eta G _ { + } ^ { \prime } ( 0 ) x \right) ^ { t - k - 1 } = \displaystyle \sum _ { k = 1 } ^ { t - 1 } \left( 1 - ( 1 - \eta G _ { + } ^ { \prime } ( 0 ) x ) \right) \left( 1 - \eta G _ { + } ^ { \prime } ( 0 ) x \right) ^ { t - k - 1 } } } \\ { { \displaystyle \qquad = \displaystyle \sum _ { k = 1 } ^ { t - 1 } \left[ \left( 1 - \eta G _ { + } ^ { \prime } ( 0 ) x \right) ^ { t - k - 1 } - \left( 1 - \eta G _ { + } ^ { \prime } ( 0 ) x \right) ^ { t - k } \right] = 1 - \left( 1 - \eta G _ { + } ^ { \prime } ( 0 ) x \right) ^ { t - 1 } , } } \end{array}\tag{40}
$$

we have that for any adjoint operator $A$

$$
I - g _ { t - 1 } ( A ) A = \left( I - \eta G _ { + } ^ { \prime } ( 0 ) A \right) ^ { t - 1 } .
$$

With the representation (23) of $f _ { t }$ , we can rewrite $f _ { t } - f _ { \rho }$ as

$$
\begin{array} { l } { { f _ { t } - f _ { \rho } = g _ { t - 1 } ( L _ { K } ) L _ { K } f _ { \rho } - f _ { \rho } + \displaystyle \sum _ { i = 1 } ^ { t - 1 } \eta \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { t - i - 1 } E _ { i , \sigma } } } \\ { { = - \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { t - 1 } f _ { \rho } + \displaystyle \sum _ { i = 1 } ^ { t - 1 } \eta \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { t - i - 1 } E _ { i , \sigma } . } } \end{array}
$$

Further, it can be derived that

$$
\begin{array} { r l } & { \| f _ { t } - f _ { \rho } \| _ { K } \leq \| \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { t - 1 } f _ { \rho } \| _ { K } + \displaystyle \sum _ { i = 1 } ^ { t - 1 } \eta \left\| \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { t - i - 1 } E _ { i , \sigma } \right\| _ { K } } \\ & { \qquad \leq \left\| L _ { K } ^ { r - \frac { 1 } { 2 } } \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { t - 1 } \right\| \| u _ { \rho } \| _ { \rho } + \displaystyle \sum _ { i = 1 } ^ { t - 1 } \eta \left\| \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { t - i - 1 } \right\| \| E _ { i , \sigma } \| _ { K } } \\ & { \qquad \leq \left\| L _ { K } ^ { r - \frac { 1 } { 2 } } \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { t - 1 } \right\| \| u _ { \rho } \| _ { \rho } + \displaystyle \sum _ { i = 1 } ^ { t - 1 } \eta \| E _ { i , \sigma } \| _ { K } , } \end{array}
$$

where the second inequality holds by Assumption 2 with $\begin{array} { r } { r \geq \frac { 1 } { 2 } } \end{array}$ and $\| u _ { \rho } \| _ { \rho } < \infty$ , and the last inequality holds by the fact that $\begin{array} { r } { \left. \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { t - i - 1 } \right. \leq 1 } \end{array}$ for any $i \in \{ 1 , . . . , t - 1 \}$

Since $L _ { K }$ is a compact and positive operator, it admits a spectral decomposition with non-negative eigenvalues $\{ \gamma _ { j } \} _ { j = 1 } ^ { \infty }$ . Then

$$
\begin{array} { r l } { \left\| { L _ { K } ^ { r - \frac { 1 } { 2 } } \left( I - \eta { G _ { + } ^ { \prime } } ( 0 ) L _ { K } \right) ^ { t - 1 } } \right\| \le \underset { j \ge 1 } { \operatorname* { s u p } } \gamma _ { j } ^ { r - \frac { 1 } { 2 } } \left( 1 - \eta { G _ { + } ^ { \prime } } ( 0 ) \gamma _ { j } \right) ^ { t - 1 } } & { } \\ & { \qquad = \underset { j \ge 1 } { \operatorname* { s u p } } \exp \left\{ ( t - 1 ) \log ( 1 - \eta { G _ { + } ^ { \prime } } ( 0 ) \gamma _ { j } ) + ( r - \frac { 1 } { 2 } ) \log \gamma _ { j } \right\} } \\ & { \qquad \le \underset { j \ge 1 } { \operatorname* { s u p } } \exp \left\{ - ( t - 1 ) \eta { G _ { + } ^ { \prime } } ( 0 ) \gamma _ { j } + ( r - \frac { 1 } { 2 } ) \log \gamma _ { j } \right\} . } \end{array}
$$

For the function

$$
f ( \gamma ) = - ( t - 1 ) \eta G _ { + } ^ { \prime } ( 0 ) \gamma + ( r - \frac 1 2 ) \log \gamma , \quad \gamma > 0 ,
$$

it is easy to know that the maximum is attained at $\begin{array} { r } { \gamma ^ { * } = \frac { r - \frac { 1 } { 2 } } { ( t - 1 ) \eta G _ { + } ^ { \prime } ( 0 ) } } \end{array}$ and

$$
f ( \gamma ^ { * } ) = - ( r - \frac { 1 } { 2 } ) + ( r - \frac { 1 } { 2 } ) \log ( r - \frac { 1 } { 2 } ) - ( r - \frac { 1 } { 2 } ) \log [ ( t - 1 ) \eta G _ { + } ^ { \prime } ( 0 ) ] .
$$

Hence, it yields that

$$
\Big \| { L _ { K } ^ { r - \frac { 1 } { 2 } } } \left( I - \eta { G _ { + } ^ { \prime } } ( 0 ) L _ { K } \right) ^ { t - 1 } \Big \| \le \exp \{ f ( \gamma ^ { * } ) \} = \left( \frac { r - \frac { 1 } { 2 } } { e \eta { G _ { + } ^ { \prime } } ( 0 ) } \right) ^ { r - \frac { 1 } { 2 } } ( t - 1 ) ^ { - ( r - \frac { 1 } { 2 } ) } .
$$

Finally, we have

$$
\| f _ { t } - f _ { \rho } \| _ { K } \leq \left( \frac { r - \frac { 1 } { 2 } } { e \eta G _ { + } ^ { \prime } ( 0 ) } \right) ^ { r - \frac { 1 } { 2 } } \| u _ { \rho } \| _ { \rho } ( t - 1 ) ^ { - ( r - \frac { 1 } { 2 } ) } + \sum _ { i = 1 } ^ { t - 1 } \eta \| E _ { i , \sigma } \| _ { K } .
$$

On the other hand, by following a similar proof strategy, we can likewise obtain

$$
\begin{array} { r l } & { \| f _ { t } - f _ { \rho } \| _ { \rho } \leq \| L _ { K } ^ { r } \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { t - 1 } \| \| u _ { \rho } \| _ { \rho } + \displaystyle \sum _ { i = 1 } ^ { t - 1 } \eta \left\| ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { t - i - 1 } \right\| \| E _ { i , \sigma } \| _ { K } } \\ & { \qquad \leq \left( \frac { r } { e \eta G _ { + } ^ { \prime } ( 0 ) } \right) ^ { r } \| u _ { \rho } \| _ { \rho } ( t - 1 ) ^ { - r } + \displaystyle \sum _ { i = 1 } ^ { t - 1 } \eta \left\| ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { t - i - 1 } \right\| \| E _ { i , \sigma } \| _ { K } . } \end{array}
$$

This completes the proof of Proposition 3.

Finally, by combining these two propositions with the triangle inequality and letting $t = T + 1$ , we can easily derive the following error decomposition for DKRGD.

Proposition 4. For $\begin{array} { r } { \lambda > 0 , 0 < \eta \leq \frac { 1 } { \kappa ^ { 2 } \operatorname* { m a x } \{ G _ { + } ^ { \prime } ( 0 ) , C _ { G } \} } } \end{array}$ , and $T \in \mathbb { N }$ . If Assumption 2 holds for $\begin{array} { r } { r \geq \frac { 1 } { 2 } } \end{array}$ , we have

$$
\begin{array} { r l } { \displaystyle \big \| \bar { f } _ { T + 1 , D } - f _ { \rho } \big \| _ { \rho } \leq ( 1 + \eta G _ { + } ^ { \prime } ( 0 ) \lambda T ) ( \mathcal { P } _ { D , \lambda } + \mathcal { Q } _ { D , \lambda } \| f _ { \rho } \| _ { K } ) } & { } \\ { \displaystyle \qquad + \sum _ { i = 1 } ^ { T } \eta \| ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } ( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } ) ^ { T - i } \| \displaystyle \sum _ { j = 1 } ^ { m } \frac { | D _ { j } | } { | D | } \| E _ { i , D _ { j } , \sigma } - E _ { i , \sigma } \| _ { K } } & { } \\ { \displaystyle \qquad + \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \| \big ( \lambda I + L _ { K } \big ) ( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \big ) ^ { T - i } \| \mathcal { Q } _ { D , \lambda } \| f _ { i } - f _ { \rho } \| _ { K } + \operatorname* { m a x } _ { 1 \leq j \leq m } \kappa _ { D _ { j } , \lambda } } & { } \\ { \displaystyle \qquad + ( \frac { r } { e \eta G _ { + } ^ { \prime } ( 0 ) } ) ^ { r } \| u _ { \rho } \| _ { \rho } T ^ { - r } + \displaystyle \sum _ { i = 1 } ^ { T } \eta \| \big ( \lambda I + L _ { K } \big ) ^ { \frac { 1 } { 2 } } ( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } ) ^ { T - i } \| \| E _ { i , \sigma } \| _ { K } . } & { } \end{array}\tag{41}
$$

## 5.3 Error decomposition II for DKRGD

As established in [6], gradient descent can be interpreted as a special instance of spectral algorithms. To derive the generalization error bounds in expectation for DKRGD, we adopt the general error decomposition framework provided in [3].

Lemma 3. Let $\begin{array} { r } { \bar { f } _ { T + 1 , D } = \sum _ { j = 1 } ^ { m } \frac { | D _ { j } | } { | D | } f _ { T + 1 , D _ { j } } } \end{array}$ , we have

$$
\mathbb { E } \left[ \left. \bar { f } _ { T + 1 , D } - f _ { \rho } \right. _ { \rho } ^ { 2 } \right] \leq \sum _ { j = 1 } ^ { m } \frac { | D _ { j } | ^ { 2 } } { | D | ^ { 2 } } \mathbb { E } \left[ \left. f _ { T + 1 , D _ { j } } - f _ { \rho } \right. _ { \rho } ^ { 2 } \right] + \sum _ { j = 1 } ^ { m } \frac { | D _ { j } | } { | D | } \left. \mathbb { E } \left[ f _ { T + 1 , D _ { j } } \right] - f _ { \rho } \right. _ { \rho } ^ { 2 } .\tag{42}
$$

To estimate the first term on the right-hand side of (42), we introduce the following proposition.

Proposition 5. For $\begin{array} { r } { \lambda > 0 , 0 < \eta \leq \frac { 1 } { \kappa ^ { 2 } \operatorname* { m a x } \{ G _ { + } ^ { \prime } ( 0 ) , C _ { G } \} } } \end{array}$ , and $T \in \mathbb { N }$ . If Assumption $\mathcal { Z }$ holds for $r \geq { \frac { 1 } { 2 } }$ , we have

$$
\begin{array} { l } { { \displaystyle \| f _ { T + 1 , D _ { j } } - f _ { \rho } \| _ { \rho } \leq ( 1 + \eta G _ { + } ^ { \prime } ( 0 ) \lambda T ) A _ { D _ { j , \lambda } } ^ { 2 } ( \mathcal { P } _ { D _ { j , \lambda } } + \mathcal { Q } _ { D _ { j , \lambda } } \| f _ { \rho } \| _ { K } ) } } \\ { { \displaystyle \qquad + \sum _ { i = 1 } ^ { T } \eta \left\| ( \lambda I + L _ { K , D _ { j } } ) ^ { \frac { 1 } { 2 } } \left( I - \eta G _ { + } ^ { \prime \prime } ( 0 ) L _ { K , D _ { j } } \right) ^ { T - i } \right\| A _ { D _ { j , \lambda } } \left\| E _ { i , D _ { j } , \sigma } - E _ { i , \sigma } \right\| _ { K } } } \\ { { \displaystyle \qquad + \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \left\| ( \lambda I + L _ { K , D _ { j } } ) \left( I - \eta G _ { + } ^ { \prime \prime } ( 0 ) L _ { K , D _ { j } } \right) ^ { T - i } \right\| A _ { D _ { j , \lambda } } ^ { 2 } \mathcal { Q } _ { D _ { j , \lambda } } \| f _ { i } - f _ { \rho } \| _ { K } } } \\ { { \displaystyle \qquad + \left( \frac { r } { e \eta G _ { + } ^ { \prime } ( 0 ) } \right) ^ { r } \| u _ { \rho } \| _ { \rho } T ^ { - r } + \sum _ { i = 1 } ^ { T } \eta \left\| ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { T - i } \right\| \left\| E _ { i , \sigma } \right\| _ { K } . } } \end{array}\tag{43}
$$

Proof. We first decompose the error into the following two parts,

$$
f _ { T + 1 , D _ { j } } - f _ { \rho } = f _ { T + 1 , D _ { j } } - f _ { T + 1 } + f _ { T + 1 } - f _ { \rho } .
$$

The estimates of $\| f _ { T + 1 } - f _ { \rho } \| _ { \rho }$ is guaranteed by(39) in Proposition $s ,$

$$
\big \lVert f _ { T + 1 } - f _ { \rho } \big \rVert _ { \rho } \leq \left( \frac { r } { e \eta G _ { + } ^ { \prime } ( 0 ) } \right) ^ { r } \left. u _ { \rho } \right. _ { \rho } T ^ { - r } + \sum _ { i = 1 } ^ { T } \eta \left. \left( \lambda I + L _ { K } \right) ^ { \frac { 1 } { 2 } } \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { T - i } \right. \left. E _ { i , \sigma } \right. _ { K } .
$$

For the estimates of $\| f _ { T + 1 , D _ { j } } - f _ { T + 1 } \| _ { \rho }$ , applying (28) with D replaced by $D _ { j }$ yields that

$$
\begin{array} { l } { \displaystyle \| f _ { T + 1 , D _ { j } } - f _ { T + 1 } \| _ { \rho } \leq \Big \| ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } ( f _ { T + 1 , D _ { j } } - f _ { T + 1 } ) \Big \| _ { K } } \\ { \displaystyle \leq ( 1 + \eta G _ { + } ^ { \prime } ( 0 ) \lambda T ) A _ { D _ { j } , \lambda } ^ { 2 } ( \mathcal { P } _ { D _ { j } , \lambda } + \mathcal { Q } _ { D _ { j } , \lambda } \| f _ { \rho } \| _ { K } ) } \\ { \displaystyle \qquad + \sum _ { i = 1 } ^ { T } \eta \left\| ( \lambda I + L _ { K , D _ { j } } ) ^ { \frac { 1 } { 2 } } \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } \right) ^ { T - i } \right\| \mathcal { A } _ { D _ { j } , \lambda } \left\| E _ { i , D _ { j } , \sigma } - E _ { i , \sigma } \right\| _ { K } } \\ { \displaystyle \qquad + \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \left\| ( \lambda I + L _ { K , D _ { j } } ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } \right) ^ { T - i } \right\| \mathcal { A } _ { D _ { j } , \lambda } ^ { 2 } \mathcal { Q } _ { D _ { j } , \lambda } \| f _ { i } - f _ { \rho } \| _ { K } . } \end{array}
$$

Finally, the proof of Proposition 5 is completed by combining the bounds on $\left\| f _ { T + 1 , D _ { j } } - f _ { T + 1 } \right\| _ { \rho }$ and $\| f _ { T + 1 } - f _ { \rho } \| _ { \rho }$ with the triangle inequality $\begin{array} { r } { \| f _ { T + 1 , D _ { j } } - f _ { \rho } \| _ { \rho } \leq \left\| f _ { T + 1 , D _ { j } } - f _ { T + 1 } \right\| _ { \rho } + \| f _ { T + 1 } - f _ { \rho } \| _ { \rho } . } \end{array}$ □

The second term on the right-hand side of (42) requires careful treatment. To establish a rigorous bound, we introduce a semi-supervised learning version of the local estimator $f _ { T + 1 , D _ { j } }$ . Specifically, define

$$
f _ { T + 1 , D } ^ { * } : = \mathbb { E } ^ { * } \left[ f _ { T + 1 , D } \right] = \mathbb { E } \left[ f _ { T + 1 , D } | D ( x ) \right] ,
$$

where $\mathbb { E } ^ { * }$ denotes the conditional expectation with respect to the output y given the fixed input data $D ( x ) = \{ x _ { i } \} _ { i = 1 } ^ { | D | }$ Since $\mathbb { E } ^ { * } [ y _ { i } ] = f _ { \rho } ( x _ { i } )$ , it follows from (7) that

$$
f _ { T + 1 , D } ^ { * } = \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { T - i } L _ { K , D } f _ { \rho } + \sum _ { i = 1 } ^ { T } \eta \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { T - i } \mathbb { E } ^ { * } [ E _ { i , D , \sigma } ] .\tag{44}
$$

For any fixed $j \in \{ 1 , 2 , \dots m \}$ , from Jensen’s inequality, we have

$$
\begin{array} { r } { \left\| \mathbb { E } [ f _ { T + 1 , D _ { j } } ] - f _ { \rho } \right\| _ { \rho } = \left\| \mathbb { E } [ \mathbb { E } ^ { * } [ f _ { T + 1 , D _ { j } } ] - f _ { \rho } ] \right\| _ { \rho } \leq \mathbb { E } \left[ \left\| f _ { T + 1 , D _ { j } } ^ { * } - f _ { \rho } \right\| _ { \rho } \right] . } \end{array}
$$

Hence, to estimate $\left\| \mathbb E [ f _ { T + 1 , D _ { j } } ] - f _ { \rho } \right\| _ { \rho } ,$ , it sufices to derive the bound of $\left. f _ { T + 1 , D _ { j } } ^ { * } - f _ { \rho } \right. _ { \rho }$ . With D replaced by $D _ { j }$ in (44), it follows from (40) that

$$
\begin{array} { l } { f _ { T + 1 , D _ { j } } ^ { * } - f _ { \rho } = \left( \displaystyle \sum _ { i = 1 } ^ { T } \eta { G } _ { + } ^ { \prime } \left( 0 \right) \left( I - \eta { G } _ { + } ^ { \prime } \left( 0 \right) L _ { K , D _ { j } } \right) ^ { T - i } L _ { K , D _ { j } } - I \right) f _ { \rho } + \displaystyle \sum _ { i = 1 } ^ { T } \eta \left( I - \eta { G } _ { + } ^ { \prime } \left( 0 \right) L _ { K , D _ { j } } \right) ^ { T - i } \mathbb E ^ { * } [ E _ { i , D _ { j } , \sigma } ] } \\ { \displaystyle \quad = - \left( I - \eta { G } _ { + } ^ { \prime } \left( 0 \right) L _ { K , D _ { j } } \right) ^ { T } f _ { \rho } + \displaystyle \sum _ { i = 1 } ^ { T } \eta \left( I - \eta { G } _ { + } ^ { \prime } \left( 0 \right) L _ { K , D _ { j } } \right) ^ { T - i } \mathbb E ^ { * } [ E _ { i , D _ { j } , \sigma } ] . } \end{array}
$$

Further, since (14) holds for $r \geq { \frac { 1 } { 2 } }$ , we can derive the following error decomposition

$$
\begin{array} { r l r } {  { \| f _ { T + 1 , D _ { p } } ^ { * } - f _ { \rho } \| _ { \rho } \leq \| ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } ( I - \eta G _ { + } ^ { \prime \prime } ( 0 ) L _ { K , D _ { s } } ) ^ { T } f _ { \rho } \| _ { K } + \| \sum _ { i = 1 } ^ { T } \eta ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } ( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { s } } ) ^ { T - i } \mathbb { E } ^ { * } [ E _ { i , D _ { j } , \sigma } ] \| _ { K } } } \\ & { } & { \leq { \mathcal A } _ { D , \lambda } \| ( \lambda I + L _ { K , D _ { j } } ) ^ { \frac { 1 } { 3 } } ( I - \eta G _ { + } ^ { \prime \prime } ( 0 ) L _ { K , D _ { s } } ) ^ { T } L _ { K } ^ { - \frac { 1 } { 2 } } \| \| u _ { \rho } \| _ { \rho } } \\ & { } & { \quad + \displaystyle \sum _ { i = 1 } ^ { T } \eta \| ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } ( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { s } } ) ^ { T - i } \| \| \mathbb { E } ^ { * } [ E _ { i , D _ { j } , \sigma } ] \| _ { K } . } \end{array}
$$

Now we are in a position to estimate the above two terms separately, where the corresponding bounds are derived for diferent ranges of r.

Case 1: $\textstyle { \frac { 1 } { 2 } } \leq r \leq { \frac { 3 } { 2 } }$ . When $\textstyle { \frac { 1 } { 2 } } \leq r \leq 1$ , by rewriting $L _ { K } ^ { r - { \frac { 1 } { 2 } } }$ as ${ \cal L } _ { K } ^ { r - { \frac { 1 } { 2 } } } = ( \lambda I + L _ { K , D _ { j } } ) ^ { r - { \frac { 1 } { 2 } } } ( \lambda I + L _ { K , D _ { j } } ) ^ { - ( r - { \frac { 1 } { 2 } } ) } ( \lambda I +$ $L _ { K } ) ^ { r - \frac { 1 } { 2 } } ( \lambda I + L _ { K } ) ^ { - ( r - \frac { 1 } { 2 } ) } L _ { K } ^ { r - \frac { 1 } { 2 } }$ , we can derive

$$
\begin{array} { r l } & { \left\| f _ { T + 1 , D _ { s } } ^ { * } - f _ { \rho } \right\| _ { \rho } \leq A _ { D _ { 2 } , \lambda } \bigg \| ( \lambda I + L _ { K , D _ { 2 } } ) ^ { \frac { 1 } { 2 } } \left( I - \eta \mathcal { G } _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { 2 } } \right) ^ { T } ( \lambda I + L _ { K , D _ { 2 } } ) ^ { r - \frac { 1 } { 2 } } . } \\ & { \qquad ( \lambda I + L _ { K , D _ { s } } ) ^ { - ( r - \frac { 1 } { 2 } ) } ( \lambda I + L _ { K } ) ^ { r - \frac { 1 } { 2 } } ( \lambda I + L _ { K } ) ^ { - ( r - \frac { 1 } { 2 } ) } L _ { K } ^ { r - \frac { 1 } { 2 } } \bigg \| \| u _ { \rho } \| _ { \rho } } \\ & { \qquad + \displaystyle \sum _ { i = 1 } ^ { T } \eta \left\| ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } \left( I - \eta \mathcal { G } _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { s } } \right) ^ { T - i } \right\| \| \mathbb { E } ^ { \kappa } [ E _ { i , D _ { s } , \tau } ] \| _ { K } } \\ & { \leq \left\| ( \lambda I + L _ { K , D _ { s } } ) ^ { r } \left( I - \eta \mathcal { G } _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { s } } \right) ^ { T } \right\| \mathcal { A } _ { D _ { \rho } , \lambda } ^ { 2 r } \| u _ { \rho } \| _ { \rho } } \\ & { \qquad + \displaystyle \sum _ { i = 1 } ^ { T } \eta \left\| ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } \left( I - \eta \mathcal { G } _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { s } } \right) ^ { T - i } \right\| \left\| \mathbb { E } ^ { \kappa } [ E _ { i , D _ { s } , \tau } ] \right\| _ { K } , } \end{array}\tag{46}
$$

where the last inequality follows from the fact that for two positive operators $A , B$ on a Hilbert space and $s \in [ 0 , 1 ]$ there holds [34]

$$
\| A ^ { s } B ^ { s } \| \leq \| A B \| ^ { s } .\tag{47}
$$

When $\textstyle 1 < r \leq { \frac { 3 } { 2 } }$ , considering that $2 r - 1 > 1$ , we can employ an approach analogous to the case of $\textstyle { \frac { 1 } { 2 } } \leq r \leq 1$ to obtain

$$
\begin{array} { r l } {  { \| f _ { T + 1 , D _ { j } } ^ { * } - f _ { \rho } \| _ { \rho } \leq A _ { D _ { j } , \lambda } \| ( \lambda I + L _ { K , D _ { j } } ) ^ { \frac { 1 } { 2 } } ( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } ) ^ { T } ( \lambda I + L _ { K , D _ { j } } ) ^ { r - \frac { 1 } { 2 } } \| \tilde { \mathcal { A } } _ { D _ { j } , \lambda } ^ { r - \frac { 1 } { 2 } } \| u _ { \rho } \| _ { \rho } } } \\ & { \qquad + \displaystyle \sum _ { i = 1 } ^ { T } \eta \| ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } ( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } ) ^ { T - i } \| \| \mathbb { E } ^ { * } [ k _ { i , D _ { j } , \sigma } ] \| _ { K } } \\ & { \leq \| ( \lambda I + L _ { K , D _ { j } } ) ^ { r } ( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } ) ^ { T } \| \tilde { \mathcal { A } } _ { D _ { j } , \lambda } ^ { r } \| u _ { \rho } \| _ { \rho } } \\ & { \qquad + \displaystyle \sum _ { i = 1 } ^ { T } \eta \| ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } ( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } ) ^ { T - i } \| \| \mathbb { E } ^ { * } [ E _ { i , D _ { j } , \sigma } ] \| _ { K } . } \end{array}\tag{48}
$$

where

$$
\tilde { \mathcal { A } } _ { D , \lambda } = \| ( \lambda I + L _ { K , D } ) ^ { - 1 } ( \lambda I + L _ { K } ) \| ,
$$

and the last inequality holds by the result that $\mathcal { A } _ { D _ { j } , \lambda } \leq \tilde { \mathcal { A } } _ { D _ { j } , \lambda } ^ { \frac { 1 } { 2 } }$

Case 2. $\textstyle { \frac { 3 } { 2 } } < r \leq { \frac { 5 } { 2 } }$ . Note that $r - { \textstyle { \frac { 1 } { 2 } } } > 1$ , hence the property in (47) cannot be applied directly. To address this issue, we adopt an operator decomposition technique as follows

$$
( \lambda I + L _ { K } ) ^ { r - \frac { 1 } { 2 } } = ( \lambda I + L _ { K } ) ^ { r - \frac { 1 } { 2 } } - ( \lambda I + L _ { K , D _ { j } } ) ^ { r - \frac { 1 } { 2 } } + ( \lambda I + L _ { K , D _ { j } } ) ^ { r - \frac { 1 } { 2 } } .
$$

By (45), we have

$$
\begin{array} { l } { \| f _ { T + 1 , D _ { j } } ^ { * } - f _ { \rho } \| _ { \rho } \leq A _ { D _ { j } , \lambda } \Big \| ( \lambda I + L _ { K , D _ { j } } ) ^ { \frac { 1 } { 2 } } ( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } ) ^ { T } \Big ( ( \lambda I + L _ { K , D _ { j } } ) ^ { r - \frac { 1 } { 2 } } - ( \lambda I + L _ { K } ) ^ { r - \frac { 1 } { 2 } } \Big ) \| \| u _ { \rho } \| _ { \rho } } \\ { \qquad + A _ { D _ { j } , \lambda } \Big \| ( \lambda I + L _ { K , D _ { j } } ) ^ { r } ( I - \eta G _ { + } ^ { \prime } ( 0 ) I _ { K , D _ { j } } ) ^ { T } \Big \| \| u _ { \rho } \| _ { \rho } } \\ { \qquad + \displaystyle \sum _ { i = 1 } ^ { T } \eta \| ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } ( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } ) ^ { T - i } \| \| \mathbb { E } ^ { s } [ E _ { i , D _ { j } , \sigma } ] \| _ { K } . } \end{array}\tag{49}
$$

To estimate the first term of the right-hand side of (49), we introduce an operator decomposition in [3]

$$
\begin{array} { r l } & { \quad ( \lambda I + L _ { K , D _ { g } } ) ^ { r - \frac { 1 } { 2 } } - ( \lambda I + L _ { K } ) ^ { r - \frac { 1 } { 2 } } } \\ & { = ( \lambda I + L _ { K , D _ { g } } ) \left( ( \lambda I + L _ { K , D _ { g } } ) ^ { r - \frac { 3 } { 2 } } - ( \lambda I + L _ { K , D _ { g } } ) ^ { - 1 } ( \lambda I + L _ { K } ) ^ { r - \frac { 1 } { 2 } } \right) } \\ & { = ( \lambda I + L _ { K , D _ { g } } ) \left( ( ( \lambda I + L _ { K , D _ { g } } ) ^ { r - \frac { 3 } { 2 } } - ( \lambda I + L _ { K } ) ^ { r - \frac { 3 } { 2 } } ) + ( ( \lambda I + L _ { K } ) ^ { - 1 } - ( \lambda I + L _ { K , D _ { g } } ) ^ { - 1 } ) ( \lambda I + L _ { K } ) ^ { r - \frac { 1 } { 2 } } \right) . } \end{array}\tag{50}
$$

Combining (50) with (49), we have

$$
\begin{array} { l } { \displaystyle \Big \| f _ { T + 1 , D _ { j } } ^ { * } - f _ { \rho } \Big \| _ { \rho } \leq \mathcal { I } _ { D _ { j } , \lambda } + \mathcal { A } _ { D _ { j } , \lambda } \Big \| \big ( \lambda I + L _ { K , D _ { j } } \big ) ^ { r } \big ( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } \big ) ^ { T } \Big \| \| u _ { \rho } \| _ { \rho } } \\ { \displaystyle \qquad + \sum _ { i = 1 } ^ { T } \eta \left\| \big ( \lambda I + L _ { K } \big ) ^ { \frac { 1 } { 2 } } \big ( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } \big ) ^ { T - i } \right\| \| \mathbb { E } ^ { * } [ E _ { i , D _ { j } , \sigma } ] \| _ { K } , } \end{array}\tag{51}
$$

where

$$
\begin{array} { r l } & { \mathcal { I } _ { D _ { j } , \lambda } : = \mathcal { A } _ { D _ { j } , \lambda } \bigg \| \big ( \lambda I + L _ { K , D _ { j } } \big ) ^ { \frac { 1 } { 2 } } \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } \right) ^ { T } ( \lambda I + L _ { K , D _ { j } } ) \cdot } \\ & { \qquad \left( \big ( \lambda I + L _ { K , D _ { j } } \big ) ^ { r - \frac { 3 } { 2 } } - ( \lambda I + L _ { K } ) ^ { r - \frac { 3 } { 2 } } \right) \bigg \| \| u _ { \rho } \| _ { \rho } } \\ & { \qquad + \mathcal { A } _ { D _ { j } , \lambda } \bigg \| \big ( \lambda I + L _ { K , D _ { j } } \big ) ^ { \frac { 1 } { 2 } } \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } \right) ^ { T } ( \lambda I + L _ { K , D _ { j } } ) \cdot } \\ & { \qquad \big ( ( \lambda I + L _ { K } ) ^ { - 1 } - ( \lambda I + L _ { K , D _ { j } } ) ^ { - 1 } \big ) \left( \lambda I + L _ { K } \right) ^ { r - \frac { 1 } { 2 } } \bigg \| \| u _ { \rho } \| _ { \rho } . } \end{array}
$$

In the following, we will estimate $\mathcal { I } _ { D _ { j } , \lambda }$ for diferent ranges of r to derive the error decomposition for $\| f _ { T + 1 , D _ { j } } ^ { * } - f _ { \rho } \| _ { \rho } .$ When $\textstyle { \frac { 3 } { 2 } } < r \leq { \frac { 5 } { 2 } }$ , we have $\textstyle 0 < r - { \frac { 3 } { 2 } } \leq 1$ . With the fact that

$$
( \lambda I + L _ { K , D _ { j } } ) ^ { r - \frac { 3 } { 2 } } - ( \lambda I + L _ { K } ) ^ { r - \frac { 3 } { 2 } } = ( \lambda I + L _ { K , D _ { j } } ) ^ { r - \frac { 3 } { 2 } } \left( I - ( \lambda I + L _ { K , D _ { j } } ) ^ { - ( r - \frac { 3 } { 2 } ) } ( \lambda I + L _ { K } ) ^ { r - \frac { 3 } { 2 } } \right) ,
$$

and (47), we can derive

$$
\begin{array} { r l } & { \mathcal { I } _ { D _ { j } , \lambda } \leq \mathcal { A } _ { D _ { j } , \lambda } \left\| { ( \lambda I + L _ { K , D _ { j } } ) ^ { r } \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } \right) ^ { T } } \right\| \left( 1 + \left\| { ( \lambda I + L _ { K , D _ { j } } ) ^ { - ( r - \frac { 3 } { 2 } ) } ( \lambda I + L _ { K } ) ^ { r - \frac { 3 } { 2 } } } \right\| \right) \left\| u _ { \rho } \right\| _ { \rho } } \\ & { \qquad + \ A _ { D _ { j } , \lambda } ^ { 2 } \left\| { ( \lambda I + L _ { K , D _ { j } } ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } \right) ^ { T } } \right\| \mathcal { Q } _ { D , \lambda } \left\| { ( \lambda I + L _ { K } ) ^ { r - \frac { 3 } { 2 } } } \right\| \| u _ { \rho } \| _ { \rho } } \\ & { \leq \tilde { A } _ { D _ { j } , \lambda } ^ { \frac { 1 } { 2 } } \left\| { ( \lambda I + L _ { K , D _ { j } } ) ^ { r } \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } \right) ^ { T } } \right\| \left( 1 + \tilde { A } _ { D _ { j } , \lambda } ^ { r - \frac { 3 } { 2 } } \right) \| u _ { \rho } \| _ { \rho } } \\ & { \qquad + \tilde { A } _ { D _ { j } , \lambda } \left\| { ( \lambda I + L _ { K , D _ { j } } ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } \right) ^ { T } } \right\| \mathcal { Q } _ { D , \lambda } \left\| { ( \lambda I + L _ { K } ) ^ { r - \frac { 3 } { 2 } } } \right\| \| u _ { \rho } \| _ { \rho } , } \end{array}\tag{52}
$$

where the first inequality follows from the fact that

$$
\begin{array} { r } { ( \lambda I + L _ { K , D _ { j } } ) ^ { \frac { 1 } { 2 } } \left( ( \lambda I + L _ { K } ) ^ { - 1 } - ( \lambda I + L _ { K , D _ { j } } ) ^ { - 1 } \right) ( \lambda I + L _ { K } ) = ( \lambda I + L _ { K , D _ { j } } ) ^ { - \frac { 1 } { 2 } } ( L _ { K , D _ { j } } - L _ { K } ) . } \end{array}
$$

Putting (52) into (51), we have

$$
\begin{array} { r l } { \big \| f _ { T + 1 , D _ { j } } ^ { * } - f _ { \rho } \big \| _ { \rho } \leq \Big ( \tilde { A } _ { D _ { j } , \lambda } ^ { \frac { 1 } { 2 } } + \tilde { A } _ { D _ { j } , \lambda } ^ { r - 1 } \Big ) \left\| \big ( \lambda I + L _ { K , D _ { j } } \big ) ^ { r } \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } \right) ^ { T } \right\| \| u _ { \rho } \| _ { \rho } } & { } \\ { + \tilde { A } _ { D _ { j } , \lambda } Q _ { D _ { j } , \lambda } \left\| \big ( \lambda I + L _ { K , D _ { j } } \big ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } \right) ^ { T } \right\| \left\| \big ( \lambda I + L _ { K } \big ) ^ { r - \frac { 3 } { 2 } } \right\| \| u _ { \rho } \| _ { \rho } } & { } \\ { + \tilde { A } _ { D _ { j } , \lambda } ^ { \frac { 1 } { 2 } } \left\| \big ( \lambda I + L _ { K , D _ { j } } \big ) ^ { r } \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } \right) ^ { T } \right\| \| u _ { \rho } \| _ { \rho } } & { } \\ { + \displaystyle \sum _ { i = 1 } ^ { T } \eta \left\| \big ( \lambda I + L _ { K } \big ) ^ { \frac { 1 } { 2 } } \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } \right) ^ { T - i } \right\| \left\| \mathbb { E } ^ { * } [ E _ { i , D _ { j } , \sigma } ] \right\| _ { K } . } & { } \end{array}\tag{53}
$$

Case 3. $r > \frac { 5 } { 2 }$ . Now we have $r - \mathrm { \frac { 3 } { 2 } } > 1$ and (47) cannot be adopted directly. Lemma A.6 in [34] states that for any two positive self-adjoint operators A, B on a Hilbert space satisfying $\| A \| \leq c$ and $\| B \| \leq c$ for some constant $c > 0$ , there holds that for all $s \geq 1$

$$
\| A ^ { s } - B ^ { s } \| \leq s c ^ { s - 1 } \| A - B \| .
$$

Recall that $\| L _ { K } \| \le \kappa ^ { 2 }$ and $\| L _ { K , D _ { j } } \| \le \kappa ^ { 2 }$ . To estimate $\mathcal { I } _ { D _ { j } , \lambda }$ in (51), we substitute $A = \lambda I + L _ { K , D _ { i } } , B = \lambda I + L _ { K }$ and $\textstyle s = r - { \frac { 3 } { 2 } }$ into the above inequality to yield

$$
\begin{array} { l } { { \mathcal { I } _ { D _ { j } , \lambda } \leq \left( r - \frac { 3 } { 2 } \right) \kappa ^ { 2 r - 5 } \tilde { A } _ { D _ { j } , \lambda } ^ { \frac { 1 } { 2 } } \left. ( \lambda I + L _ { K , D _ { j } } ) ^ { \frac { 3 } { 2 } } \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } \right) ^ { T } \right. \left. L _ { K } - L _ { K , D _ { j } } \right. \left. u _ { \rho } \right. _ { \rho } } } \\ { { \quad \quad \quad + \tilde { A } _ { D _ { j } , \lambda } \left. ( \lambda I + L _ { K , D _ { j } } ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } \right) ^ { T } \right. \mathcal { Q } _ { D _ { j } , \lambda } \left. ( \lambda I + L _ { K } ) ^ { r - \frac { 3 } { 2 } } \right. \left. u _ { \rho } \right. _ { \rho } . } } \end{array}
$$

By putting the above bound into (51), we have

$$
\begin{array} { r l } { \big \| f _ { T + 1 , D _ { j } } ^ { * } - f _ { \rho } \big \| _ { \rho } \leq } & { \bigg ( r - \frac { 3 } { 2 } \bigg ) \kappa ^ { 2 r - 5 } \bar { \mathcal { A } } _ { D _ { j } , \lambda } ^ { \frac { 1 } { 2 } } \bigg \| \big ( \lambda I + L _ { K , D _ { j } } \big ) ^ { \frac { 3 } { 2 } } \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } \right) ^ { T } \bigg \| \| L _ { K } - L _ { K , D _ { j } } \| \| u _ { \rho } \| _ { \rho } } \\ & { \qquad + \tilde { \mathcal { A } } _ { D _ { j } , \lambda } \mathcal { Q } _ { D _ { j } , \lambda } \bigg \| \big ( \lambda I + L _ { K , D _ { j } } \big ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } \right) ^ { T } \bigg \| \big \| \big ( \lambda I + L _ { K } \big ) ^ { r - \frac { 3 } { 2 } } \bigg \| \| u _ { \rho } \| _ { \rho } } \\ & { \qquad + \bar { \mathcal { A } } _ { D _ { j } , \lambda } ^ { \frac { 1 } { 2 } } \bigg \| \big ( \lambda I + L _ { K , D _ { j } } \big ) ^ { r } \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } \right) ^ { T } \bigg \| \| u _ { \rho } \| _ { \rho } } \\ & { \qquad + \displaystyle \sum _ { i = 1 } ^ { T } \eta \left\| \big ( \lambda I + L _ { K } \big ) ^ { \frac { 1 } { 2 } } \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } \right) ^ { T - i } \right\| \big \| \mathbb { E } ^ { * } [ E _ { i , D _ { j } , \sigma } ] \big \| _ { K } . } \end{array}\tag{54}
$$

Thus, we have completed the error decomposition of $\left. f _ { T + 1 , D _ { j } } ^ { * } - f _ { \rho } \right. _ { \rho }$

## 5.4 Error decomposition for DKRGD with communication

In this subsection, we derive an error decomposition for DKRGD with communication based on (12). Firstly, we adopt $f _ { t + 1 , D }$ as an intermediate function to derive the following error decomposition

$$
\left\| \bar { f } _ { T + 1 , D } ^ { l } - f _ { \rho } \right\| _ { \rho } \leq \left\| \bar { f } _ { T + 1 , D } ^ { l } - f _ { T + 1 , D } \right\| _ { \rho } + \left\| f _ { T + 1 , D } - f _ { \rho } \right\| _ { \rho } .
$$

The bound on $\| f _ { T + 1 , D } - f _ { \rho } \| _ { \rho }$ has been derived in [8] with the polynomial decay assumption on the eigenvalues of the integral operator $L _ { K }$ , in contrast to which we present a more general bound adapted to the capacity assumption as

follows. By substituting $D _ { j }$ with D in (43), we can follow a proof strategy analogous to that of Proposition 5 to derive

$$
\begin{array} { l } { { \| f _ { T + 1 , D } - f _ { \rho } \| _ { \rho } \leq \left( 1 + \eta G _ { + } ^ { \prime } ( 0 ) \lambda T \right) A _ { D , \lambda } ^ { 2 } ( \mathcal { P } _ { D , \lambda } + \mathcal { Q } _ { D , \lambda } \| f _ { \rho } \| _ { K } ) } } \\ { { \qquad + \displaystyle \sum _ { i = 1 } ^ { T } \eta \left\| \left( \lambda I + L _ { K , D } \right) ^ { \frac { 1 } { 2 } } \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { T - i } \right\| A _ { D , \lambda } \left\| E _ { i , D , \sigma } - E _ { i , \sigma } \right\| _ { K } } } \\ { { \qquad + \displaystyle \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \left\| \left( \lambda I + L _ { K , D } \right) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { T - i } \right\| A _ { D , \lambda } ^ { 2 } \mathcal { Q } _ { D , \lambda } \| f _ { i } - f _ { \rho } \| _ { K } } } \\ { { \qquad + \left( \displaystyle \frac { r } { e \eta G _ { + } ^ { \prime } ( 0 ) } \right) ^ { r } \left\| u _ { \rho } \right\| _ { \rho } T ^ { - r } + \displaystyle \sum _ { i = 1 } ^ { T } \eta \left\| \left( \lambda I + L _ { K } \right) ^ { \frac { 1 } { 2 } } \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { T - i } \right\| \left\| E _ { i , \sigma } \right\| _ { K } . } } \end{array}\tag{55}
$$

Next, we will derive the error decomposition for $\left\| \bar { f } _ { T + 1 , D } ^ { l } - f _ { T + 1 , D } \right\| _ { \rho }$ . Due to (7) and (12), we get

$$
\begin{array} { l } { f _ { T + 1 , D } - \bar { f } _ { T + 1 , D } ^ { 2 } = - g _ { T } ( L _ { K , D } ) [ g _ { T } ^ { - 1 } ( I _ { K , D } ) \bar { f } _ { T + 1 , D } ^ { - 1 } - \bar { f } _ { K , D } ] + \displaystyle \sum _ { 1 = 1 } ^ { T } \eta ( I - \eta \bar { G } _ { + } ^ { \prime } ( 0 ) I _ { K , D } ) ^ { T - \delta } E _ { I , D } , } \\ { \displaystyle \qquad + \displaystyle \sum _ { i = 1 } ^ { m } [ \frac { D _ { \delta } } { | D | } g _ { T } ( L _ { K , D } ) [ g _ { T } ^ { - 1 } ( L _ { K , D } ) \bar { f } _ { T + 1 , D } ^ { \bar { I } _ { I } - 1 } - \bar { f } _ { K , D } ]  } \\ { \displaystyle \qquad - \sum _ { j = 1 } ^ { m } \frac { | D _ { \delta } | } { | D | } g _ { T } ( L _ { K , D } ) g _ { T } ^ { - 1 } ( L _ { K , D } ) \displaystyle \sum _ { i = 1 } ^ { T } \eta ( I - \eta \bar { G } _ { + } ^ { \prime } ( 0 ) L _ { K , D } ) ^ { T - i } E _ { I , i , D , \sigma } } \\ { \displaystyle \qquad = \displaystyle \sum _ { j = 1 } ^ { m } \frac { | D _ { \delta } | } { | D | } ( g _ { T } ( L _ { K , D _ { i } } ) - g _ { T } ( L _ { K , D } ) ) [ g _ { T } ^ { - 1 } ( L _ { K , D } ) \bar { f } _ { T + 1 , D } ^ { \bar { I } _ { 1 } - 1 } - \hat { f } _ { K , D } ] } \\  \displaystyle \qquad + \sum _ { j = 1 } ^ { m } \frac { | D _ { \delta } | } { | D | } ( g _ { T } ( L _ { K , D _ { i } } ) - g _ { T } ( L _ { K , D _ { j } } ) ) g _ { T } ^ { - 1 } ( L _ { K , D } ) \displaystyle \sum _ { i = 1 } \end{array}
$$

In order to simplify the above decomposition and represent $f _ { T + 1 , D } - \bar { f } _ { T + 1 , D } ^ { l }$ by ${ f _ { T + 1 , D } - \bar { f } _ { T + 1 , D } ^ { l - 1 } \ \mathrm { o r } \ \bar { f } _ { T + 1 , D } ^ { l - 1 } - f _ { T + 1 , D } } ^ { l } $ we rewrite $g _ { T } ^ { - 1 } ( L _ { K , D } ) \bar { f } _ { T + 1 , D } ^ { l - 1 } - \hat { f } _ { K , D }$ as

$$
\begin{array} { r l } { g _ { T } ^ { - 1 } ( L _ { K , D } ) \bar { f } _ { T + 1 , D } ^ { l - 1 } - \hat { f } _ { K , D } = g _ { T } ^ { - 1 } ( L _ { K , D } ) \left[ \bar { f } _ { T + 1 , D } ^ { l - 1 } - g _ { T } ( L _ { K , D } ) \hat { f } _ { K , D } \right] } & { } \\ & { \qquad = g _ { T } ^ { - 1 } ( L _ { K , D } ) \left[ \bar { f } _ { T + 1 , D } ^ { l - 1 } - f _ { T + 1 , D } + f _ { T + 1 , D } - g _ { T } ( L _ { K , D } ) \hat { f } _ { K , D } \right] } \\ & { \qquad = g _ { T } ^ { - 1 } ( L _ { K , D } ) \left[ \bar { f } _ { T + 1 , D } ^ { l - 1 } - f _ { T + 1 , D } + \displaystyle \sum _ { i = 1 } ^ { T } \eta \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { T - i } E _ { i , D , \sigma } \right] , } \end{array}
$$

and further derive that

$$
\begin{array} { l } { { f _ { T + 1 , D } - \bar { f } _ { T + 1 , D } ^ { l } = \displaystyle \sum _ { j = 1 } ^ { m } \frac { | D _ { j } | } { | D | } \left( \theta r ( L _ { K , D _ { 2 } } ) - g _ { T } ( L _ { K , D } ) \right) g _ { T } ^ { - 1 } ( L _ { K , D } ) \left[ \bar { f } _ { T + 1 , D } ^ { l - 1 } - f _ { T + 1 , D } \right] } } \\ { { \displaystyle \qquad + \sum _ { j = 1 } ^ { m } \frac { | D _ { j } | } { | D | } \left( g _ { T } ( L _ { K , D _ { 3 } } ) - g _ { T } ( L _ { K , D } ) \right) g _ { T } ^ { - 1 } ( L _ { K , D } ) \sum _ { i = 1 } ^ { T } \eta \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { T - i } E _ { i , D , \sigma } } } \\ { { \displaystyle \qquad + \sum _ { j = 1 } ^ { m } \frac { | D _ { j } | } { | D | } \left( g _ { T } ( L _ { K , D } ) - g _ { T } ( L _ { K , D _ { j } } ) \right) g _ { T } ^ { - 1 } ( L _ { K , D } ) \sum _ { i = 1 } ^ { T } \eta \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { T - i } E _ { i , D , \sigma } } } \\ { { \displaystyle \qquad = \sum _ { j = 1 } ^ { m } \frac { | D _ { j } | } { | D | } \left( g _ { T } ( L _ { K , D _ { 2 } } ) - g _ { T } ( L _ { K , D } ) \right) g _ { T } ^ { - 1 } ( L _ { K , D } ) \left( \bar { f } _ { T + 1 , D } ^ { l - 1 } - f _ { T + 1 , D } \right) . } } \end{array}\tag{56}
$$

Note that it follows from the definition (9) of $g _ { T } ( \cdot )$ that

$$
\begin{array} { c } { { \displaystyle g _ { T } ( L _ { K , D _ { j } } ) - g _ { T } ( L _ { K , D } ) } } \\ { { \displaystyle = \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } \right) ^ { T - i } - \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { T - i } } } \\ { { \displaystyle = \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \left( \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } \right) ^ { T - i } - \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { T - i } \right) } } \end{array}\tag{57}
$$

To obtain a more refined estimate of $f _ { T + 1 , D } - \bar { f } _ { T + 1 , D } ^ { l } .$ , we need to adjust $\left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } \right) ^ { T - i } - \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { T - i }$ in (57). To this end, we first introduce the following property that for two operators $X , Y$ and $n \in \mathbb { N } _ { + }$ , there holds

$$
\begin{array} { l } { { X ^ { n } - Y ^ { n } = X ^ { n } - Y X ^ { n - 1 } + Y X ^ { n - 1 } - Y ^ { 2 } X ^ { n - 2 } + \cdot \cdot \cdot + Y ^ { n - 1 } X - Y ^ { n } } } \\ { { \ \qquad = ( X - Y ) X ^ { n - 1 } + Y ( X - Y ) X ^ { n - 2 } + \cdot \cdot \cdot + Y ^ { n - 1 } ( X - Y ) } } \\ { { \ \qquad = \cdot \cdot \cdot = \displaystyle \sum _ { k = 1 } ^ { n } Y ^ { k - 1 } ( X - Y ) X ^ { n - k } . } } \end{array}
$$

By applying the above property to $X = I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { i } } , Y = I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D }$ and $n = T - i$ , we can deduce that

$$
\begin{array} { l } { { \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } \right) ^ { T - i } - \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { T - i } } } \\ { { \displaystyle = \sum _ { k = 1 } ^ { T - i } \eta G _ { + } ^ { \prime } ( 0 ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { k - 1 } \left( L _ { K , D } - L _ { K , D _ { j } } \right) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } \right) ^ { T - i - k } } } \\ { { \displaystyle = \sum _ { k = i + 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { k - i - 1 } \left( L _ { K , D } - L _ { K , D _ { j } } \right) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } \right) ^ { T - k } . } } \end{array}
$$

Based on the results above, we can obtain an estimate of $\left\| \left( g _ { T } ( L _ { K , D _ { j } } ) - g _ { T } ( L _ { K , D } ) \right) g _ { T } ^ { - 1 } ( L _ { K , D } ) \right\|$ as follows

$$
\begin{array} { l } { { \displaystyle \left\| \left( g _ { t } ( L _ { K , D _ { j } } ) - g _ { t } ( L _ { K , D } ) \right) g _ { T } ^ { - 1 } ( L _ { K , D } ) \right\| = \left\| g _ { T } ^ { - 1 } ( L _ { K , D } ) \left( g _ { t } ( L _ { K , D _ { j } } ) - g _ { t } ( L _ { K , D } ) \right) \right\| } }  \\ { { \displaystyle = \left\| g _ { T } ^ { - 1 } ( L _ { K , D } ) \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \left( \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } \right) ^ { T - i } - \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { T - i } \right) \right\| } } \\ { { \displaystyle = \left\| g _ { T } ^ { - 1 } ( L _ { K , D } ) \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \sum _ { k = i + 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { k - i - 1 } \left( L _ { K , D } - L _ { K , D _ { j } } \right) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } \right) ^ { T - k } \right\| , } } \end{array}
$$

where the first equality follows from the fact that $g _ { T } ( L _ { K , D _ { j } } ) - g _ { T } ( L _ { K , D } )$ and $g _ { T } ^ { - 1 } ( L _ { K , D } )$ are self-adjoint operators. Furthermore, we can interchange the order of summation to obtain

$$
\begin{array} { r l } & { \quad \| ( g _ { t } ( L _ { K , D _ { j } } ) - g _ { t } ( L _ { K , D } ) ) g _ { T } ^ { - 1 } ( L _ { K , D } ) \| } \\ & { = \| \displaystyle \sum _ { k = 2 } ^ { T } \eta C _ { + } ^ { \prime } ( 0 ) g _ { T } ^ { - 1 } ( L _ { K , D } ) \displaystyle \sum _ { i = 1 } ^ { k - 1 } \eta C _ { + } ^ { \prime } ( 0 ) ( I - \eta C _ { + } ^ { \prime } ( 0 ) L _ { K , D } ) ^ { k - i - 1 } ( L _ { K , D } - L _ { K , D _ { j } } ) ( I - \eta C _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } ) ^ { T - k } \| } \\ & { \leq \displaystyle \sum _ { k = 2 } ^ { T } \eta C _ { + } ^ { \prime } ( 0 ) \| g _ { T } ^ { - 1 } ( L _ { K , D } ) \displaystyle \sum _ { i = 1 } ^ { k - 1 } \eta C _ { + } ^ { \prime } ( 0 ) ( I - \eta C _ { + } ^ { \prime \prime } ( 0 ) [ L _ { K , D } ) ^ { k - i - 1 } ( L _ { K , D } - L _ { K , D _ { j } } ) ( I - \eta C _ { + } ^ { \prime \prime } ( 0 ) L _ { K , D _ { j } } ) ^ { T - k } \| } \\ & { \leq \displaystyle \sum _ { k = 2 } ^ { T } \eta C _ { + } ^ { \prime } ( 0 ) \| g _ { T } ^ { - 1 } ( L _ { K , D } ) \displaystyle \sum _ { i = 1 } ^ { k - 1 } \eta C _ { + } ^ { \prime \prime } ( 0 ) ( I - \eta C _ { + } ^ { \prime \prime } ( 0 ) [ L _ { K , D } ) ^ { k - i - 1 } ] \| ( I _ { K , D } - L _ { K , D _ { j } } ) ( I - \eta C _ { + } ^ { \prime \prime } ( 0 ) L _ { K , D _ { j } } ) ^ { T - k } \| } \\ &  \leq \displaystyle \sum _  \end{array}
$$

where the last inequality follows from the fact that for any $2 \leq k \leq T$

$$
\left\| g _ { T } ^ { - 1 } ( L _ { K , D } ) \sum _ { i = 1 } ^ { k - 1 } \eta G _ { + } ^ { \prime } ( 0 ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { k - i - 1 } \right\| \leq 1 .
$$

Since $L _ { K , D } - L _ { K , D _ { j } }$ can be decomposed as $L _ { K , D } - L _ { K } + L _ { K } - L _ { K , D _ { j } }$ , we see

$$
\begin{array} { r l r } {  { \| ( g _ { t } ( L _ { K , D _ { s } } ) - g _ { t } ( L _ { K , D _ { s } } ) ) g _ { T } ^ { - 1 } ( L _ { K , D } ) \| } } \\ & { \leq \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \| ( L _ { K , D } - L _ { K } ) ( \lambda I + L _ { K } ) ^ { - \frac { 1 } { 2 } } \| \| ( ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } ( \lambda I + L _ { K , D _ { s } } ) ^ { - \frac { 1 } { 2 } } \| \| ( \lambda I + L _ { K , D _ { s } } ) ^ { \frac { 1 } { 2 } } ( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { s } } ) ^ { T - i } \| } \\ & { + \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \| ( L _ { K } - L _ { K , D _ { s } } ) ( \lambda I + L _ { K } ) ^ { - \frac { 1 } { 2 } } \| \| ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } ( \lambda I + L _ { K , D _ { s } } ) ^ { - \frac { 1 } { 2 } } \| \| ( \lambda I + L _ { K , D _ { s } } ) ^ { \frac { 1 } { 2 } } ( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { s } } ) ^ { T - i } \| } \\ & { } & { \quad - ( Q _ { D , \lambda } + Q _ { D , \lambda } ) A _ { D , \lambda } \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \| ( \lambda I + L _ { K , D _ { s } } ) ^ { \frac { 1 } { 2 } } ( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { s } } ) ^ { T - i } \| . } \end{array}
$$

Finally, it can be derived by combining (56) and (58) that

$$
\left. f _ { T + 1 , D } - \bar { f } _ { T + 1 , D } ^ { l } \right. _ { \rho } \leq \sum _ { j = 1 } ^ { m } \frac { | D _ { j } | } { | D | } \left. \left( g _ { T } ( L _ { K , D _ { j } } ) - g _ { T } ( L _ { K , D } ) \right) g _ { T } ^ { - 1 } ( L _ { K , D } ) \right. \left. \bar { f } _ { T + 1 , D } ^ { l - 1 } - f _ { T + 1 , D } \right. _ { \rho }
$$

$$
\leq \left\{ \sum _ { j = 1 } ^ { m } \frac { | D _ { j } | } { | D | } \left( \mathcal { Q } _ { D , \lambda } + \mathcal { Q } _ { D _ { j } , \lambda } \right) A _ { D _ { j } , \lambda } \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \left\| \left( \lambda I + L _ { K , D _ { j } } \right) ^ { \frac { 1 } { 2 } } \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } \right) ^ { T - i } \right\| \right\} \left\| \bar { f } _ { T + 1 , D } ^ { i - 1 } - f _ { T + 1 , D } \right\| _ { \rho }\tag{59}
$$

$$
\leq \left\{ \sum _ { j = 1 } ^ { m } \frac { \vert D _ { j } \vert } { \vert D \vert } ( \mathcal { Q } _ { D , \lambda } + \mathcal { Q } _ { D _ { j } , \lambda } ) \mathcal { A } _ { D _ { j } , \lambda } \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \left. \left( \lambda I + L _ { K , D _ { j } } \right) ^ { \frac { 1 } { 2 } } \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D _ { j } } \right) ^ { T - i } \right. \right\} ^ { i } \left. \tilde { f } _ { T + 1 , D } ^ { 0 } - f _ { T + 1 , D } \right. _ { \rho } .
$$

Recall that $\bar { f } _ { T + 1 , D } ^ { 0 } = \bar { f } _ { T + 1 , D }$ . This decomposes the bound on $\left\| f _ { T + 1 , D } - \bar { f } _ { T + 1 , D } ^ { l } \right\| _ { \rho }$ into the product of an operator norm depending on l and the term $\left\| \bar { f } _ { T + 1 , D } - f _ { T + 1 , D } \right\| _ { \rho }$ . The latter can be further bounded via Proposition 2. Finally, the error decomposition for $\| f _ { T + 1 , D } - f _ { \rho } \| _ { \rho }$ is completed by combining (55) and (59).

## 6 Proofs of Main Results

In this section, we present the proofs of the main results in Section 3. To this end, we need to introduce some preliminary estimates on the quantities defined in Section 5, which are demonstrated in the following lemmas.

## 6.1 Preliminary Lemmas

Lemma 4. Let $\{ z _ { i } = ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { | D | }$ be drawn independently according to $\rho$ and $0 < \delta < 1$ . Under Assumption 1, each of the following estimates holds with confidence at least 1 − δ:

$$
A _ { D , \lambda } = \left\| ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } ( \lambda I + L _ { K , D } ) ^ { - \frac { 1 } { 2 } } \right\| \leq \left( \frac { \mathcal { B } _ { | D | , \lambda } } { \sqrt { \lambda } } + 1 \right) \log \frac { 2 } { \delta } ,\tag{60}
$$

$$
\tilde { \mathcal { A } } _ { D , \lambda } = \left\| ( \lambda I + L _ { K } ) ( \lambda I + L _ { K , D } ) ^ { - 1 } \right\| \leq \left( \frac { \mathcal { B } _ { | D | , \lambda } } { \sqrt { \lambda } } + 1 \right) ^ { 2 } \left( \log \frac { 2 } { \delta } \right) ^ { 2 } ,\tag{61}
$$

$$
\mathcal { P } _ { D , \lambda } = \Big | \Big | ( \lambda I + L _ { K } ) ^ { - \frac { 1 } { 2 } } \big ( \hat { f } _ { K , D } - L _ { K } f _ { \rho } \big ) \Big | \Big | _ { K } \le \mathcal { B } _ { | D | , \lambda } \log \frac { 2 } { \delta } ,\tag{62}
$$

$$
\mathcal { Q } _ { D , \lambda } = \left| \left| ( \lambda I + L _ { K } ) ^ { - \frac { 1 } { 2 } } ( L _ { K } - L _ { K , D } ) \right| \right| \leq \mathcal { B } _ { | D | , \lambda } \log \frac { 2 } { \delta } ,\tag{63}
$$

$$
\| L _ { K } - L _ { K , D } \| \leq \| L _ { K } - L _ { K , D } \| _ { H S } \leq \frac { 4 \kappa ^ { 2 } } { \sqrt { | D | } } \log \frac { 2 } { \delta } ,\tag{64}
$$

where $\begin{array} { r } { \beta _ { | D | , \lambda } = \frac { 2 \kappa } { \sqrt { | D | } } \left\{ \frac { \kappa } { \sqrt { | D | \lambda } } + \sqrt { \mathcal { N } ( \lambda ) } \right\} } \end{array}$ and $\| \cdot \| _ { H S }$ denotes the norm of $\mathrm { H S } ( \mathcal { H } _ { K } )$ , the Hilbert space of all Hilbert-Schmidt operators on $\mathcal { H } _ { K }$

The aforementioned bounds are well-established in the literature. Specifically, the proofs for (60), (63), and (64) follow from [2, 22, 35], while (61) and (62) can be found in [3] and [22], respectively. It is crucial to note that the high-probability bounds in Lemma 4 do not hold simultaneously. Consequently, we must invoke the union bound in the subsequent analysis to control the overall failure probability. This necessity is reflected in the accumulation of constants within the logarithmic terms of the final error bound.

By leveraging a new concentration inequality for self-adjoint operators in [20], the following bound on $\mathcal { R } _ { D , \lambda }$ in Proposition 2 are proved in [5], which contributes to loosening the restriction on the number of local machines m.

Lemma 5. Let $0 < \delta \leq 1 , \ i f 0 < \lambda \leq 1$ and $\mathcal { N } ( \lambda ) > 1$ , then there holds with confidence at least $1 - \delta$ that

$$
\begin{array} { r l } & { \mathcal { R } _ { D , \lambda } = \displaystyle \lVert ( \lambda I + L _ { K } ) ^ { - \frac { 1 } { 2 } } ( L _ { K } - L _ { K , D } ) ( \lambda I + L _ { K } ) ^ { - \frac { 1 } { 2 } } \rVert } \\ & { \quad \quad \le \operatorname* { m a x } \left\{ \displaystyle \frac { \kappa ^ { 2 } + 1 } { 3 } , 2 \sqrt { \kappa ^ { 2 } + 1 } \right\} \left\{ \displaystyle \frac { 1 + \log \mathcal { N } ( \lambda ) } { \lambda | D | } + \sqrt { \displaystyle \frac { 1 + \log \mathcal { N } ( \lambda ) } { \lambda | D | } } \right\} \log \frac { 4 } { \delta } } \\ & { \quad = \cdot \mathcal { R } _ { | D | , \lambda } \log \frac { 4 } { \delta } . } \end{array}
$$

Furthermore, leveraging Lemma 5, we refine the bound on $\mathcal { A } _ { D , \lambda }$ in (60) as follows.

Lemma 6. Let $0 < \delta \leq 1 , \ i f 0 < \lambda \leq 1$ and $\mathcal { N } ( \lambda ) > 1$ , then there holds with confidence at least $1 - \delta$ that

$$
A _ { D , \lambda } \leq \left( 1 + \mathcal { R } _ { | D | , \lambda } ^ { 2 } \left( \frac { \mathcal { B } _ { | D | , \lambda } } { \sqrt { \lambda } } + 1 \right) ^ { 2 } + \mathcal { R } _ { | D | , \lambda } \right) ^ { \frac { 1 } { 2 } } \left( \log \frac { 8 } { \delta } \right) ^ { 2 } = : A _ { | D | , \lambda } \left( \log \frac { 8 } { \delta } \right) ^ { 2 } .
$$

Proof. To derive a tighter upper bound for $\mathcal { A } _ { D , \lambda }$ than (60), consider

$$
\begin{array} { r l } & { { \cal A } _ { D , \lambda } = \left\| ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } ( \lambda I + L _ { K , D } ) ^ { - \frac { 1 } { 2 } } \right\| = \left\| \left[ ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } ( \lambda I + L _ { K , D } ) ^ { - 1 } ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } \right] ^ { \frac { 1 } { 2 } } \right\| } \\ & { \qquad \leq \left\| ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } ( \lambda I + L _ { K , D } ) ^ { - 1 } ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } \right\| ^ { \frac { 1 } { 2 } } , } \end{array}\tag{65}
$$

then we only need to estimate $\left\| ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } ( \lambda I + L _ { K , D } ) ^ { - 1 } ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } \right\| .$

It is derived by Lemma 16 in [2] that for any two invertible operators A and B on a Banach space, there holds

$$
A ^ { - 1 } - B ^ { - 1 } = B ^ { - 1 } ( B - A ) B ^ { - 1 } ( B - A ) A ^ { - 1 } + B ^ { - 1 } ( B - A ) B ^ { - 1 } .
$$

With the above second order decomposition of inverse operator diference, we have

$$
B ^ { \frac { 1 } { 2 } } A ^ { - 1 } B ^ { \frac { 1 } { 2 } } = B ^ { \frac { 1 } { 2 } } \left( A ^ { - 1 } - B ^ { - 1 } + B ^ { - 1 } \right) B ^ { \frac { 1 } { 2 } } = B ^ { - { \frac { 1 } { 2 } } } ( B - A ) B ^ { - 1 } ( B - A ) A ^ { - 1 } B ^ { \frac { 1 } { 2 } } + B ^ { - { \frac { 1 } { 2 } } } ( B - A ) B ^ { - { \frac { 1 } { 2 } } } + I .\tag{66}
$$

Inserting $A = \lambda I + L _ { K , D }$ and $B = \lambda I + L _ { K }$ into (66), we obtain

$$
\begin{array} { c } { { ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } ( \lambda I + L _ { K , D } ) ^ { - 1 } ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } = ( \lambda I + L _ { K } ) ^ { - \frac { 1 } { 2 } } ( L _ { K } - L _ { K , D } ) ( \lambda I + L _ { K } ) ^ { - 1 } ( L _ { K } - L _ { K , D } ) ( \lambda I + L _ { K , D } ) ^ { - 1 } } } \\ { { { } } } \\ { { ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } + ( \lambda I + L _ { K } ) ^ { - \frac { 1 } { 2 } } ( L _ { K } - L _ { K , D } ) ( \lambda I + L _ { K } ) ^ { - \frac { 1 } { 2 } } + I , } } \end{array}
$$

and further derive that

$$
\begin{array} { r l } & { \quad \quad \left\| ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } ( \lambda I + L _ { K , D } ) ^ { - 1 } ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } \right\| } \\ & { \leq \left\| ( \lambda I + L _ { K } ) ^ { - \frac { 1 } { 2 } } ( L _ { K } - L _ { K , D } ) ( \lambda I + L _ { K } ) ^ { - \frac { 1 } { 2 } } ( \lambda I + L _ { K } ) ^ { - \frac { 1 } { 2 } } ( L _ { K } - L _ { K , D } ) ( \lambda I + L _ { K } ) ^ { - \frac { 1 } { 2 } } ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } ( \lambda I + L _ { K , D } ) ^ { - 1 } \right. } \\ & { \qquad \left. ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } + ( \lambda I + L _ { K } ) ^ { - \frac { 1 } { 2 } } ( L _ { K } - L _ { K , D } ) ( \lambda I + L _ { K } ) ^ { - \frac { 1 } { 2 } } + I \right\| } \end{array}
$$

$$
\leq 1 + \mathcal { R } _ { D , \lambda } ^ { 2 } \mathcal { A } _ { D , \lambda } ^ { 2 } + \mathcal { R } _ { D , \lambda } .
$$

Then for $0 < \delta < 1$ , by Lemma 4 and Lemma 5 and scaling 2δ to $\delta ,$ there holds with confidence at least 1−δ that

$$
\begin{array} { r l r } {  { \| { ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } ( \lambda I + L _ { K , D } ) ^ { - 1 } ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } } \| \leq 1 + \mathcal { R } _ { | D | , \lambda } ^ { 2 } ( \frac { \mathcal { B } _ { | D | , \lambda } } { \sqrt { \lambda } } + 1 ) ^ { 2 } ( \log \frac { \mathtt { B } } { \delta } ) ^ { 2 } ( \log \frac { 4 } { \delta } ) ^ { 2 } + \mathcal { R } _ { | D | , \lambda } \log \frac { 8 } { \delta } } } \\ & { } & { \leq ( 1 + \mathcal { R } _ { | D | , \lambda } ^ { 2 } ( \frac { \mathcal { B } _ { | D | , \lambda } } { \sqrt { \lambda } } + 1 ) ^ { 2 } + \mathcal { R } _ { | D | , \lambda } ) ( \log \frac { 8 } { \delta } ) ^ { 4 } . \qquad } \end{array}\tag{67}
$$

Finally, the proof of Lemma 6 is completed by substituting (67) into (65).

To derive explicit learning rates based on Proposition 4, it is essential to first estimate the operator norms appearing in the upper bounds, specifically for $\left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { T - i }$ and $\left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { T - i }$ . These estimates are provided in the following two lemmas.

Lemma 7. For any integer $T \geq 3$ and $0 < \lambda \leq 1$ , we have

$$
\sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \left\| ( \lambda I + L _ { K , D } ) ^ { \frac { 1 } { 2 } } \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { T - i } \right\| \leq C _ { \eta } ^ { \prime } \left( \lambda ^ { \frac { 1 } { 2 } } T + T ^ { \frac { 1 } { 2 } } \right) ,\tag{68}
$$

and

$$
\sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \left\| \left( \lambda I + L _ { K , D } \right) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { T - i } \right\| \leq 2 \eta G _ { + } ^ { \prime } ( 0 ) \lambda T + \left( 2 \eta G _ { + } ^ { \prime } ( 0 ) \kappa ^ { 2 } + \frac 4 e \right) \log T ,\tag{69}
$$

where $\begin{array} { r } { C _ { \eta } ^ { \prime } = \sqrt { 2 } \eta ( 1 + \kappa ) G _ { + } ^ { \prime } ( 0 ) + 2 \sqrt { 2 } \left( \frac { \eta G _ { + } ^ { \prime } ( 0 ) } { e } \right) ^ { \frac { 1 } { 2 } } } \end{array}$ . The same bounds hold with $L _ { K , D }$ replaced by $L _ { K }$

Proof. By setting $\theta = 0$ and $\eta _ { 1 } = \eta$ , Equation (21) in [8, Prop. 4.2] implies that for any $\tau , \lambda > 0$ and $1 \leqslant i < T$ , there holds

$$
\left\| { ( \lambda I + L _ { K , D } ) ^ { \tau } \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { T - i } } \right\| \leq 2 ^ { \tau } \left( \lambda ^ { \tau } + \left( \frac { \tau } { e G _ { + } ^ { \prime } ( 0 ) } \right) ^ { \tau } \left( \eta ( T - i ) \right) ^ { - \tau } \right) .\tag{70}
$$

To derive the first inequality in Lemma 7, we can set $\begin{array} { r } { \tau = \frac { 1 } { 2 } } \end{array}$ in the above inequality to obtain

$$
\sum _ { i = 1 } ^ { T - 1 } \eta G _ { + } ^ { \prime } ( 0 ) \left. \left( \lambda I + L _ { K , D } \right) ^ { \frac { 1 } { 2 } } \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { T - i } \right. \le \sqrt { 2 } \eta G _ { + } ^ { \prime } ( 0 ) \left( \sum _ { i = 1 } ^ { T - 1 } \lambda ^ { \frac { 1 } { 2 } } + \sum _ { i = 1 } ^ { T - 1 } \left( \frac { 1 } { e G _ { + } ^ { \prime } ( 0 ) } \right) ^ { \frac { 1 } { 2 } } ( \eta ( T - i ) ) ^ { - \frac { 1 } { 2 } } \right) .
$$

Since

$$
\sum _ { i = 1 } ^ { T - 1 } ( T - i ) ^ { - \frac { 1 } { 2 } } = \sum _ { i = 1 } ^ { T - 1 } i ^ { - \frac { 1 } { 2 } } \leq 1 + \int _ { 1 } ^ { T - 1 } x ^ { - \frac { 1 } { 2 } } d x \leq 2 T ^ { \frac { 1 } { 2 } } ,
$$

we have

$$
\sum _ { i = 1 } ^ { T - 1 } \eta G _ { + } ^ { \prime } ( 0 ) \left\| ( \lambda I + L _ { K , D } ) ^ { \frac { 1 } { 2 } } \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { T - i } \right\| \leq \sqrt { 2 } \eta G _ { + } ^ { \prime } ( 0 ) \lambda ^ { \frac { 1 } { 2 } } T + 2 \sqrt { 2 } \left( \frac { \eta G _ { + } ^ { \prime } ( 0 ) } { e } \right) ^ { \frac { 1 } { 2 } } T ^ { \frac { 1 } { 2 } } .
$$

This together with the fact that $\begin{array} { r } { \left\| { ( \lambda I + L _ { K , D } ) ^ { \frac { 1 } { 2 } } \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { 0 } } \right\| = \left\| { ( \lambda I + L _ { K , D } ) ^ { \frac { 1 } { 2 } } } \right\| \le ( \lambda + \kappa ^ { 2 } ) ^ { \frac { 1 } { 2 } } \leqslant \sqrt { 2 } \kappa } \end{array}$ completes the proof of the first inequality in Lemma 7.

By setting $\tau = 1$ in (70), we can derive that

$$
\begin{array} { r l } & { \displaystyle \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \left\| \left( \lambda I + L _ { K , D } \right) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { T - i } \right\| } \\ & { \le \eta G _ { + } ^ { \prime } ( 0 ) \left\| \lambda I + L _ { K , D } \right\| + 2 \eta G _ { + } ^ { \prime } ( 0 ) \left( \displaystyle \sum _ { i = 1 } ^ { T - 1 } \lambda + \displaystyle \sum _ { i = 1 } ^ { T - 1 } \left( \displaystyle \frac { 1 } { e G _ { + } ^ { \prime } ( 0 ) } \right) ( \eta ( T - i ) ) ^ { - 1 } \right) } \\ & { \le 2 \eta G _ { + } ^ { \prime } ( 0 ) \kappa ^ { 2 } + 2 \eta G _ { + } ^ { \prime } ( 0 ) \lambda T + \displaystyle \frac { 2 } { e } \sum _ { i = 1 } ^ { T - 1 } ( T - i ) ^ { - 1 } } \\ & { \le 2 \eta G _ { + } ^ { \prime } ( 0 ) \kappa ^ { 2 } + 2 \eta G _ { + } ^ { \prime } ( 0 ) \lambda T + \displaystyle \frac { 4 } { e } \log T \le 2 \eta G _ { + } ^ { \prime } ( 0 ) \lambda T + \left( 2 \eta G _ { + } ^ { \prime } ( 0 ) \kappa ^ { 2 } + \displaystyle \frac { 4 } { e } \right) \log T , } \end{array}
$$

which completes the proof of the second inequality in Lemma 7. Finally, the same bounds apply to the population operator $L _ { K }$ , as the preceding derivations rely solely on the operator being positive, compact, and satisfying $\| \cdot \| \leq \kappa ^ { 2 }$ 2 common to both operators. □

Lemma 8. For any integer $T \geq 3 , 0 < \lambda \leq 1$ and $\begin{array} { r } { r \geq \frac { 1 } { 2 } } \end{array}$ , we have

$$
\begin{array} { l } { \displaystyle \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \left\| ( \lambda I + L _ { K , D } ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { T - i } \right\| i ^ { - ( r - \frac { 1 } { 2 } ) } } \\ { \le \left( \eta G _ { + } ^ { \prime } ( 0 ) ( 1 + \kappa ^ { 2 } ) + ( 2 \eta G _ { + } ^ { \prime } ( 0 ) + \frac { 2 } { e } ) C _ { r , 1 } \right) \operatorname* { m a x } \left\{ \lambda T ^ { - ( r - \frac { 3 } { 2 } ) } \log T , \lambda , T ^ { - ( r - \frac { 1 } { 2 } ) } \log T , T ^ { - 1 } \right\} , } \end{array}\tag{71}
$$

where $C _ { r , 1 }$ is a constant given by (73). The same bound holds with $L _ { K , D }$ replaced by $L _ { K }$

Proof. From (70) with $\tau = 1$ , we get for every $1 \leq i < T$

$$
\bigl \| ( \lambda I + L _ { K , D } ) ( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } ) ^ { T - i } \bigr \| \leq 2 \left( \lambda + \frac { 1 } { e G _ { + } ^ { \prime } ( 0 ) \eta ( T - i ) } \right) .
$$

For the index $i = T$ , we have $\eta G _ { + } ^ { \prime } ( 0 ) \| \lambda I + L _ { K , D } \| T ^ { - ( r - \frac { 1 } { 2 } ) } \leq \eta G _ { + } ^ { \prime } ( 0 ) ( \lambda + \kappa ^ { 2 } ) T ^ { - ( r - \frac { 1 } { 2 } ) }$ and thus

$$
\begin{array} { l } { \displaystyle \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \left\| ( \lambda I + L _ { K , D } ) ( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } ) ^ { T - i } \right\| i ^ { - ( r - \frac { 1 } { 2 } ) } } \\ { \displaystyle \quad \leq \eta G _ { + } ^ { \prime } ( 0 ) ( \lambda + \kappa ^ { 2 } ) T ^ { - ( r - \frac { 1 } { 2 } ) } + 2 \eta G _ { + } ^ { \prime } ( 0 ) \lambda \sum _ { i = 1 } ^ { T - 1 } i ^ { - ( r - \frac { 1 } { 2 } ) } + \frac { 2 } { e } \sum _ { i = 1 } ^ { T - 1 } \frac { i ^ { - ( r - \frac { 1 } { 2 } ) } } { T - i } . } \end{array}\tag{72}
$$

The bound for the first sum follows immediately from Lemma A.1 in [7] that

$$
\sum _ { i = 1 } ^ { T - 1 } i ^ { - ( r - \frac { 1 } { 2 } ) } \leq \left\{ \begin{array} { l l } { \frac { 2 T ^ { - ( r - \frac { 3 } { 2 } ) } } { 3 - 2 r } , } & { \frac { 1 } { 2 } < r < \frac { 3 } { 2 } , } \\ { 2 \log T , } & { r = \frac { 3 } { 2 } , } \\ { \frac { 2 r - 1 } { 2 r - 3 } , } & { r > \frac { 3 } { 2 } . } \end{array} \right.
$$

To bound $\begin{array} { r } { \sum _ { i = 1 } ^ { T - 1 } \left( T - i \right) ^ { - 1 } i ^ { - ( r - \frac { 1 } { 2 } ) } } \end{array}$ , notice that when $\begin{array} { r } { 1 \leq i \leq \frac { T } { 2 } } \end{array}$ 2

$$
\sum _ { 1 \le i \le \frac { T } { 2 } } \left( T - i \right) ^ { - 1 } i ^ { - \left( r - \frac { 1 } { 2 } \right) } \le 2 T ^ { - 1 } \sum _ { 1 \le i \le \frac { T } { 2 } } i ^ { - \left( r - \frac { 1 } { 2 } \right) } \le \left\{ \begin{array} { l l } { \frac { 4 T ^ { - \left( r - \frac { 1 } { 2 } \right) } } { 3 - 2 r } , } & { \frac { 1 } { 2 } \le r < \frac { 3 } { 2 } , } \\ { 4 T ^ { - 1 } \log T , } & { r = \frac { 3 } { 2 } , } \\ { \frac { 2 \left( 2 r - 1 \right) } { 2 r - 3 } T ^ { - 1 } , } & { r > \frac { 3 } { 2 } , } \end{array} \right.
$$

and for $\begin{array} { r } { \frac { T } { 2 } < i \leq T - 1 } \end{array}$

$$
\sum _ { \frac { T } { 2 } < i \leq T - 1 } \left( T - i \right) ^ { - 1 } i ^ { - \left( r - \frac { 1 } { 2 } \right) } \leq 2 ^ { r - \frac { 1 } { 2 } } T ^ { - \left( r - \frac { 1 } { 2 } \right) } \sum _ { \frac { T } { 2 } < i \leq T - 1 } \left( T - i \right) ^ { - 1 } \leq 2 ^ { r - \frac { 1 } { 2 } } T ^ { - \left( r - \frac { 1 } { 2 } \right) } \log T .
$$

Therefore, we have

$$
\sum _ { i = 1 } ^ { T - 1 } \left( T - i \right) ^ { - 1 } i ^ { - ( r - \frac { 1 } { 2 } ) } \leq C _ { r , 1 } \operatorname* { m a x } \left\{ T ^ { - ( r - \frac { 1 } { 2 } ) } \log T , T ^ { - 1 } \right\} ,
$$

where $C _ { r , 1 }$ is a constant given by

$$
\begin{array} { r } { C _ { r , 1 } = \left\{ \begin{array} { l l } { \frac { 4 } { 3 - 2 r } + 2 ^ { r - \frac { 1 } { 2 } } , } & { \frac { 1 } { 2 } \leq r < \frac { 3 } { 2 } , } \\ { 6 , } & { r = \frac { 3 } { 2 } , } \\ { \frac { 2 ( 2 r - 1 ) } { 2 r - 3 } + 2 ^ { r - \frac { 1 } { 2 } } , } & { r > \frac { 3 } { 2 } . } \end{array} \right. } \end{array}\tag{73}
$$

To ensure consistency in the constant factors across our theoretical results, we alternatively derive

$$
\sum _ { i = 1 } ^ { T - 1 } i ^ { - ( r - \frac { 1 } { 2 } ) } \leq C _ { r , 1 } \operatorname* { m a x } \left\{ T ^ { - ( r - \frac { 3 } { 2 } ) } \log T , 1 \right\} .
$$

Finally, putting the bound on each sum term into (72), we have

$$
\begin{array} { l } { { \displaystyle \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \left\| ( \lambda I + L _ { K , D } ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { T - i } \right\| i ^ { - ( r - \frac { 1 } { 2 } ) } } \ ~ } \\ { { \le \eta G _ { + } ^ { \prime } ( 0 ) ( \lambda + \kappa ^ { 2 } ) T ^ { - ( r - \frac { 1 } { 2 } ) } + 2 \eta G _ { + } ^ { \prime } ( 0 ) C _ { r , 1 } \operatorname* { m a x } \left\{ \lambda T ^ { - ( r - \frac { 3 } { 2 } ) } \log T , \lambda \right\} + \displaystyle \frac { 2 } { e } C _ { r , 1 } \operatorname* { m a x } \left\{ T ^ { - ( r - \frac { 1 } { 2 } ) } \log T , T ^ { - 1 } \right\} } \ } \\ { { \le \left( \eta G _ { + } ^ { \prime } ( 0 ) ( 1 + \kappa ^ { 2 } ) + 2 \eta G _ { + } ^ { \prime } ( 0 ) C _ { r , 1 } + \displaystyle \frac { 2 } { e } C _ { r , 1 } \right) \operatorname* { m a x } \left\{ \lambda T ^ { - ( r - \frac { 3 } { 2 } ) } \log T , \lambda , T ^ { - ( r - \frac { 1 } { 2 } ) } \log T , T ^ { - 1 } \right\} } , } \end{array}
$$

which completes the proof of Lemma 8.

In the end of this subsection, we aim at estimating $E _ { t , \sigma } , \mathbb { E } ^ { * } [ E _ { t , D , \sigma } ]$ and the diference between $E _ { t , D , \sigma }$ and $E _ { t , \sigma }$ respectively.

Lemma 9. Under Assumption 1, we have that for any $1 \leq t \leq T$

$$
\| E _ { t , \sigma } \| _ { K } \leq \frac { \kappa c _ { p } ( M + \kappa \| f _ { t } \| _ { K } ) ^ { 2 p + 1 } } { \sigma ^ { 2 p } } \leq \kappa c _ { p } ( 2 M ) ^ { 2 p + 1 } \frac { t ^ { p + \frac { 1 } { 2 } } } { \sigma ^ { 2 p } } ,\tag{74}
$$

$$
\Vert \mathbb { E } ^ { * } [ E _ { t , D , \sigma } ] \Vert _ { K } \leq \kappa c _ { p } \left( 2 M \right) ^ { 2 p + 1 } \frac { t ^ { p + \frac { 1 } { 2 } } } { \sigma ^ { 2 p } } ,\tag{75}
$$

$$
\| E _ { t , D , \sigma } - E _ { t , \sigma } \| _ { K } \leq 2 \kappa c _ { p } \left( 2 M \right) ^ { 2 p + 1 } \frac { t ^ { p + \frac { 1 } { 2 } } } { \sigma ^ { 2 p } } .\tag{76}
$$

Proof. Recall that $\begin{array} { r } { E _ { t , \sigma } = \int _ { \mathcal { Z } } \left( G _ { + } ^ { \prime } ( 0 ) - G ^ { \prime } ( \xi _ { t , \sigma } ( z ) ) \right) \left( f _ { t } ( x ) - y \right) K _ { x } \mathrm { d } \rho } \end{array}$ and $\begin{array} { r } { \xi _ { t , \sigma } ( z ) = \frac { ( y - f _ { t } ( x ) ) ^ { 2 } } { \sigma ^ { 2 } } } \end{array}$ . Since $E _ { t , \sigma }$ is the expectation of $E _ { t , D , \sigma }$ with respect to the data D, it follows that

$$
\| E _ { t , \sigma } \| _ { K } \leq \int _ { \mathcal { Z } } \left| G _ { + } ^ { \prime } ( 0 ) - G ^ { \prime } ( \xi _ { t , \sigma } ( z ) ) \right| | y - f _ { t } ( x ) | \| K _ { x } \| _ { K } d \rho \leq \frac { \kappa c _ { p } ( M + \kappa \| f _ { t } \| _ { K } ) ^ { 2 p + 1 } } { \sigma ^ { 2 p } } ,
$$

where the last inequality is due to condition (2) of the function G. Combining the above inequality with the bound on $\| f _ { t } \| _ { K }$ in Lemma 1 yields (74).

The uniform bound on $E _ { t , D , \sigma }$ defined in (6) has been established in [8, Proposition 5.1], based on which we can further set $\eta _ { 1 } = \eta$ and $\theta = 0$ to derive that

$$
\| E _ { t , D , \sigma } \| _ { K } \le \frac { \kappa c _ { p } ( M + \kappa \| f _ { t , D } \| _ { K } ) ^ { 2 p + 1 } } { \sigma ^ { 2 p } } \le \kappa c _ { p } ( 2 M ) ^ { 2 p + 1 } \frac { t ^ { p + \frac { 1 } { 2 } } } { \sigma ^ { 2 p } } .
$$

Then (76) can be proved by the triangle inequality $\| E _ { t , D , \sigma } - E _ { t , \sigma } \| _ { K } \leq \| E _ { t , D , \sigma } \| _ { K } + \| E _ { t , \sigma } \| _ { K }$

Finally, (75) follows from the Jensen’s inequality for conditional expectation and the bound on $\| E _ { t , D , \sigma } \| _ { K }$ . This completes the proof of Lemma 9. □

## 6.2 Proof of Theorem 1

We now proceed to prove Theorem 1 by leveraging the error decomposition in Proposition 4. Specifically, we bound each term on the right-hand side of (41) individually, and aggregate these estimates to obtain the desired learning rates.

For the first term of (41), combining Assumption 2, (62) and (63) in Lemma 4 yields that with confidence at least $1 - \delta .$ , there holds

$$
\mathcal { P } _ { D , \lambda } + \mathcal { Q } _ { D , \lambda } \| f _ { \rho } \| _ { K } \leq \big ( 1 + \kappa ^ { 2 r - 1 } \| u _ { \rho } \| _ { \rho } \big ) \mathcal { B } _ { | D | , \lambda } \log \frac { 4 } { \delta } .\tag{77}
$$

We then turn to estimate the second and third terms. Lemma 9 provides a uniform bound for $\| E _ { t , D , \sigma } - E _ { t , \sigma } \| _ { K }$ for any $t \in \{ 1 , \ldots , T \}$ . Note that $T = \lceil \lambda ^ { - 1 } \rceil \in \mathbb { N }$ , we have $1 \leq \lambda T < 2$ . By combining Lemma 7 and substituting D with $D _ { j }$ in Lemma 9, we have

$$
\sum _ { i = 1 } ^ { T } \eta \left\| ( \lambda I + L _ { K } ) ^ { \frac { 1 } { 2 } } \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { T - i } \right\| \sum _ { j = 1 } ^ { m } \frac { | D _ { j } | } { | D | } \left\| E _ { i , D _ { j } , \sigma } - E _ { i , \sigma } \right\| _ { K }
$$

$$
\leq \underset { 1 \leq j \leq m } { \operatorname* { m a x } } \sum _ { i = 1 } ^ { T } \eta \left. \left( \lambda I + L _ { K } \right) ^ { \frac { 1 } { 2 } } \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { T - i } \right. \left. E _ { i , D _ { j } , \sigma } - E _ { i , \sigma } \right. _ { K }\tag{78}
$$

$$
\leq \frac { C _ { \eta } ^ { \prime } } { G _ { + } ^ { \prime } ( 0 ) } \left( \lambda ^ { \frac 1 2 } T + T ^ { \frac 1 2 } \right) 2 \kappa c _ { p } \left( 2 M \right) ^ { 2 p + 1 } \frac { T ^ { p + \frac 1 2 } } { \sigma ^ { 2 p } } = C _ { r , 2 } \frac { T ^ { p + 1 } } { \sigma ^ { 2 p } } ,
$$

where $\begin{array} { r } { C _ { r , 2 } : = 2 ^ { \frac { 5 } { 2 } } \frac { C _ { \eta } ^ { \prime } } { G _ { + } ^ { \prime } ( 0 ) } \kappa c _ { p } \left( 2 M \right) ^ { 2 p + 1 } } \end{array}$ is a constant independent of $T , \lambda$ and $\sigma .$ Moreover, by (70) with $\tau = 1$ , we have

$$
\begin{array} { l } { { \displaystyle \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \left\| ( \lambda I + L _ { K } ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { T - i } \right\| \| f _ { i } - f _ { \rho } \| _ { K } } } \\ { { \displaystyle \le 2 \eta G _ { + } ^ { \prime } ( 0 ) \left( \lambda + \frac { 1 } { e G _ { + } ^ { \prime } ( 0 ) \eta ( T - 1 ) } \right) \kappa ^ { 2 r - 1 } \| u _ { \rho } \| _ { \rho } + \displaystyle \sum _ { i = 2 } ^ { T } \eta G _ { + } ^ { \prime \prime } ( 0 ) \left\| \left( \lambda I + L _ { K } \right) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { T - i } \right\| \| f _ { i } - f _ { \rho } \| _ { K } . } } \end{array}
$$

From (38), Lemma 7 and Lemma 8, there holds

$$
\begin{array} { l } { { \displaystyle \sum _ { i = 2 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \left\| \left( \lambda I + L _ { K } \right) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { T - i } \right\| \left\| f _ { i } - f _ { \rho } \right\| _ { K } } } \\ { { \displaystyle \leq \sum _ { i = 2 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \left\| \left( \lambda I + L _ { K } \right) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { T - i } \right\| \left[ \left( \frac { r - \frac { 1 } { 2 } } { e \eta G _ { + } ^ { \prime } ( 0 ) } \right) ^ { r - \frac { 1 } { 2 } } \| u _ { \rho } \| _ { \rho } ( i - 1 ) ^ { - ( r - \frac { 1 } { 2 } ) } + \sum _ { l = 1 } ^ { i - 1 } \eta \| E _ { l , \sigma } \| _ { K } \right] } } \\ { { \displaystyle \leq \left( \frac { r - \frac { 1 } { 2 } } { e \eta G _ { + } ^ { \prime } ( 0 ) } \right) ^ { r - \frac { 1 } { 2 } } \| u _ { \rho } \| _ { \rho } \left( \eta G _ { + } ^ { \prime } ( 0 ) ( 1 + \kappa ^ { 2 } ) + \left( 2 \eta G _ { + } ^ { \prime } ( 0 ) + \frac { 2 } { e } \right) C _ { r , 1 } \right) \operatorname* { m a x } \left\{ \lambda T ^ { - ( r - \frac { 3 } { 2 } ) } \log T , \lambda , T ^ { - ( r - \frac { 1 } { 2 } ) } \log T , T ^ { - 1 } \right\} } } \\   \displaystyle ~ + \left( 2 \eta G _ { + } ^ { \prime } ( 0 ) \lambda T + \left( 2 \eta G _ { + } ^ { \prime } ( 0 ) \kappa ^ { 2 } + \frac { 4 } { e } \right) \log T \right) \eta \kappa c _ { p } \left( 2 M \right) ^ { 2 p + \frac { 1 } { 2 } } \frac { T ^ { p + \frac { 3 } { 2 } } } { \sigma ^ { 2 p } } \end{array}
$$

Therefore, we obtain that

$$
\sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) \left\| \left( \lambda I + L _ { K } \right) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \right) ^ { T - i } \right\| \| f _ { i } - f _ { \rho } \| _ { K } \leq C _ { r , 3 } \operatorname* { m a x } \left\{ T ^ { - ( r - \frac { 1 } { 2 } ) } \log T , T ^ { - 1 } , \frac { T ^ { p + \frac { 3 } { 2 } } \log T } { \sigma ^ { 2 p } } \right\} .\tag{79}
$$

$$
\begin{array} { r } { \mathrm { w h e r e ~ } C _ { r , 3 } : = 2 \left( \frac { r - \frac { 1 } { 2 } } { e \eta G _ { + } ^ { r } ( 0 ) } \right) ^ { r - \frac { 1 } { 2 } } \| u _ { \rho } \| _ { \rho } \left( \eta G _ { + } ^ { r } ( 0 ) ( 1 + \kappa ^ { 2 } + 2 C _ { r , 1 } ) + \frac { 2 C _ { r , 1 } } { \epsilon } \right) + \left( 2 \eta G _ { + } ^ { r } ( 0 ) ( 1 + \kappa ^ { 2 } ) + \frac { 4 } { \epsilon } \right) \eta \kappa c _ { p } \left( 2 M \right) ^ { 2 p + 1 } . } \end{array}
$$

For any fixed $j \in \{ 1 , \dots , m \}$ , by applying Lemma 4-6 with D replaced by $D _ { j }$ , we can derive that with confidence at least $\begin{array} { r } { 1 - \frac { \delta } { m } } \end{array}$ , there holds

$$
\mathcal { R } _ { D _ { j } , \lambda } A _ { D _ { j } , \lambda } ^ { 2 } \left( \mathcal { P } _ { D _ { j } , \lambda } + \mathcal { Q } _ { D _ { j } , \lambda } \lVert f _ { \rho } \rVert _ { K } \right) \leq \left( 1 + \kappa ^ { 2 r - 1 } \lVert u _ { \rho } \rVert _ { \rho } \right) \mathcal { R } _ { | D _ { j } | , \lambda } \mathcal { B } _ { | D _ { j } | , \lambda } A _ { | D _ { j } | , \lambda } ^ { 2 } \left( \log \frac { 1 6 m } \delta \right) ^ { 6 } .
$$

Since Lemma 7 and Lemma 8 also hold with $L _ { K }$ replaced by $L _ { K , D }$ , we have the same bound in (78) and (79) when $L _ { K }$ is replaced by $L _ { K , D }$ . Therefore, combining (78), (79) and the above result, we can derive that with confidence at

least $\begin{array} { r } { 1 - \frac { \delta } { m } } \end{array}$ , there holds

$$
\begin{array} { r l } & { K _ { D , \lambda } \leq \left( 4 \eta G _ { + } ^ { \prime } ( 0 ) + 2 \eta G _ { + } ^ { \prime } ( 0 ) \kappa ^ { 2 } + 4 / e \right) \log T \cdot \left( 1 + 2 \eta G _ { + } ^ { \prime } ( 0 ) \right) \left( 1 + \kappa ^ { 2 r - 1 } \| u _ { \rho } \| _ { \rho } \right) \widetilde { K } _ { | D _ { \xi } | , \lambda } B _ { | D _ { \xi } | , \lambda } A _ { | D _ { \xi } | , \lambda } ^ { 2 } \left( \log \frac { 1 6 m } { \delta } \right) ^ { \otimes } } \\ & { \quad + \left( 4 \eta G _ { + } ^ { \prime } ( 0 ) + 2 \eta G _ { + } ^ { \prime } ( 0 ) \kappa ^ { 2 } + 4 / e \right) \log T \cdot C _ { r , 2 } \frac { T ^ { p + 1 } } { \sigma ^ { 2 p } } \widetilde { \mathcal { K } } _ { | D _ { \xi } | , \lambda } A _ { | D _ { \xi } | , \lambda } \left( \log \frac { 1 6 m } { \delta } \right) ^ { 3 } } \\ & { \quad + \left( 4 \eta G _ { + } ^ { \prime } ( 0 ) + 2 \eta G _ { + } ^ { \prime } ( 0 ) \kappa ^ { 2 } + 4 / e \right) \log T \cdot \mathcal { K } _ { | D _ { \xi } | , \lambda } B _ { | D _ { \xi } | , \lambda } A _ { | D _ { \xi } | , \lambda } ^ { 2 } \left( \log \frac { 1 6 m } { \delta } \right) ^ { 6 } C _ { r , 3 } \operatorname* { m a x } \left\{ T ^ { - ( \ell - \frac { 1 } { 3 } ) } \log T , T ^ { - 1 } , \frac { T ^ { p + \frac { 3 } { 2 } } \log T } { \sigma ^ { 2 p } } \right\} } \\ & { \leq C _ { r , 2 } ^ { \prime } \mathcal { R } _ { | D _ { \xi } | , \lambda } A _ { | D _ { \xi } | , \lambda } \frac { T ^ { p + 1 } } { \sigma ^ { 2 p } } \log T \left( \log \frac { 1 6 m } { \delta } \right) ^ { 3 } } \end{array}
$$

where $C _ { r , 2 } ^ { \prime } : = \left( 4 \eta G _ { + } ^ { \prime } ( 0 ) + 2 \eta G _ { + } ^ { \prime } ( 0 ) \kappa ^ { 2 } + 4 / e \right) C _ { r , 2 } \mathrm { ~ a n d ~ } C _ { r , 3 } ^ { \prime } : = \left( 4 \eta G _ { + } ^ { \prime } ( 0 ) + 2 \eta G _ { + } ^ { \prime } ( 0 ) \kappa ^ { 2 } + 4 / e \right) \left( 1 + 2 \eta G _ { + } ^ { \prime } ( 0 ) \right) \left( 1 + \kappa ^ { 2 \tau - 1 } \right)$ $\| u _ { \rho } \| _ { \rho } ) + \left( 4 \eta G _ { + } ^ { \prime } ( 0 ) + 2 \eta G _ { + } ^ { \prime } ( 0 ) \kappa ^ { 2 } + 4 / e \right) C _ { r , 3 } .$

Plugging the above estimates into (41), with confidence at least 1−δ, we have

$$
\begin{array} { r l } & { \left. \bar { f } _ { T + 1 , D } - f _ { \rho } \right. _ { \rho } \leq ( 1 + 2 \eta G _ { + } ^ { \prime } ( 0 ) ) ( 1 + \kappa ^ { 2 r - 1 } \lVert u _ { \rho } \rVert _ { \rho } ) \mathcal { B } _ { | D | , \lambda } \log \frac { 1 2 } { \delta } + \left( \frac { r } { e \eta G _ { + } ^ { \prime } ( 0 ) } \right) ^ { r } \lVert u _ { \rho } \rVert _ { \rho } T ^ { - r } } \\ & { \qquad + \frac { 3 C _ { r , 2 } } { 2 } \frac { T ^ { p + 1 } } { \sigma ^ { 2 p } } + C _ { r , 3 } \mathcal { B } _ { | D | , \lambda } \operatorname* { m a x } \left\{ T ^ { - ( r - \frac { 1 } { 2 } ) } \log T , T ^ { - r - 1 } , \frac { T ^ { p + \frac { 3 } { 2 } } \log T } { \sigma ^ { 2 p } } \right\} \log \frac { 6 } { \delta } } \\ & { \qquad + C _ { r , 2 } ^ { \prime } \underset { 1 \leq j \leq m } { \operatorname* { m a x } } \left\{ \mathcal { R } _ { | D _ { j } | , \lambda } \mathcal { A } _ { | D _ { j } | , \lambda } \right\} \frac { T ^ { p + 1 } } { \sigma ^ { 2 p } } \log T \left( \log \frac { 4 8 m } { \delta } \right) ^ { 3 } } \\ & { \qquad + C _ { r , 3 } ^ { \prime } \underset { 1 \leq j \leq m } { \operatorname* { m a x } } \left\{ \mathcal { R } _ { | D _ { j } | , \lambda } \mathcal { B } _ { | D _ { j } | , \lambda } \mathcal { A } _ { | D _ { j } | , \lambda } ^ { 2 } \right\} \operatorname* { m a x } \left\{ \log T , T ^ { - ( r - \frac { 1 } { 2 } ) } ( \log T ) ^ { 2 } , \frac { T ^ { p + \frac { 3 } { 2 } } ( \log T ) ^ { 2 } } { \sigma ^ { 2 p } } \right\} \left( \log \frac { 4 8 m } { \delta } \right) ^ { 6 } . } \end{array}\tag{80}
$$

Note that the condition on m in (16) implies

$$
m | D | ^ { - \frac { 4 r + 2 s - 1 } { 4 r + 2 s } } \leq \sqrt { m } | D | ^ { - \frac { r } { 2 r + s } } \qquad \mathrm { a n d } \qquad m | D | ^ { - \frac { 2 r + s - 1 } { 2 r + s } } \leq \sqrt { m } | D | ^ { - \frac { 2 r + s - 1 } { 4 r + 2 s } } .\tag{81}
$$

Let $\lambda = | D | ^ { - \frac { 1 } { 2 r + s } }$ , then we can adopt Assumption 3 and $| D _ { 1 } | = \cdots = | D _ { m } |$ to obtain that

$$
\mathcal { B } _ { | D | , \lambda } \le 2 \kappa \left( \kappa | D | ^ { - \frac { 4 r + 2 s - 1 } { 4 r + 2 s } } + \sqrt { C _ { 0 } } | D | ^ { - \frac { r } { 2 r + s } } \right) \le C _ { 0 , 1 } | D | ^ { - \frac { r } { 2 r + s } } ,
$$

and for any $j \in \{ 1 , \dots , m \}$

$$
\mathcal { B } _ { | D _ { j } | , \lambda } \leq 2 \kappa \left( \kappa m | D | ^ { - \frac { 4 r + 2 s - 1 } { 4 r + 2 s } } + \sqrt { C _ { 0 } m } | D | ^ { - \frac { r } { 2 r + s } } \right) \leq C _ { 0 , 1 } \sqrt { m } | D | ^ { - \frac { r } { 2 r + s } } ,
$$

$$
\begin{array} { r l } & { \mathcal { R } _ { | D _ { j } | , \lambda } \leq \operatorname* { m a x } \left\{ \displaystyle \frac { \kappa ^ { 2 } + 1 } { 3 } , 2 \sqrt { \kappa ^ { 2 } + 1 } \right\} \left\{ \displaystyle \frac { 2 s } { 2 r + s } m | D | ^ { - \frac { 2 r + s - 1 } { 2 r + s } } \log | D | + \sqrt { \frac { 2 s } { 2 r + s } m | D | ^ { - \frac { 2 r + s - 1 } { 2 r + s } } \log | D | } \right\} } \\ & { \qquad \leq C _ { 0 , 2 } \sqrt { m } | D | ^ { - \frac { 2 r + s - 1 } { 4 r + 2 s } } \log | D | , } \end{array}
$$

where $C _ { 0 , 1 } : = 2 \kappa ( \kappa + \sqrt { C _ { 0 } } )$ , and $\begin{array} { r } { C _ { 0 , 2 } = 2 \operatorname* { m a x } \left\{ \frac { \kappa ^ { 2 } + 1 } { 3 } , 2 \sqrt { \kappa ^ { 2 } + 1 } \right\} \sqrt { \frac { 2 s } { 2 r + s } } . } \end{array}$

Additionally, by leveraging the above results, we can derive the following bound for $\mathcal { A } _ { | D | , \lambda } \colon$

$$
\begin{array} { r l } & { \mathcal A _ { | D _ { j } | , \lambda } ^ { 2 } = 1 + \frac { \mathcal R _ { | D _ { j } | , \lambda } ^ { 2 } \mathcal B _ { | D _ { j } | , \lambda } ^ { 2 } } { \lambda } + 2 \frac { \mathcal R _ { | D _ { j } | , \lambda } ^ { 2 } \mathcal B _ { | D _ { j } | , \lambda } } { \sqrt \lambda } + \mathcal R _ { | D _ { j } | , \lambda } ^ { 2 } + \mathcal R _ { | D _ { j } | , \lambda } } \\ & { \qquad \leq 1 + C _ { 0 , 1 } ^ { 2 } C _ { 0 , 2 } ^ { 2 } m ^ { 2 } | D | ^ { - \frac { 4 r + s - 2 } { 2 r + s } } \left( \log | D | \right) ^ { 2 } + 2 C _ { 0 , 1 } C _ { 0 , 2 } ^ { 2 } m ^ { \frac 3 2 } | D | ^ { - \frac { 3 r + s - \frac 3 2 } { 2 r + s } } \left( \log | D | \right) ^ { 2 } } \\ & { \qquad + C _ { 0 , 2 } ^ { 2 } m | D | ^ { - \frac { 2 r + s - 1 } { 2 r + s } } \left( \log | D | \right) ^ { 2 } + C _ { 0 , 2 } \sqrt { m } | D | ^ { - \frac { 2 r + s - 1 } { 4 r + 2 s } } \log | D | } \\ & { \qquad \leq C _ { 0 , 3 } + 1 . } \end{array}
$$

where $C _ { 0 , 3 } = C _ { 0 , 1 } ^ { 2 } C _ { 0 , 2 } ^ { 2 } + 2 C _ { 0 , 1 } C _ { 0 , 2 } ^ { 2 } + C _ { 0 , 2 } ^ { 2 } + C _ { 0 , 2 }$ and the last inequality follows from the condition on m in (16).

Finally, putting the aforementioned results, $\lambda = | D | ^ { - { \frac { 1 } { 2 r + s } } }$ and $T = \left\lceil | D | ^ { \frac { 1 } { 2 r + s } } \right\rceil$ into (80), we can derive that, with confidence at least 1−δ, there holds

$$
\begin{array} { r l } & { \| \bar { f } _ { T + 1 , D } - f _ { \rho } \| _ { \rho } \leq \bigg [ C _ { 0 , 1 } ( 1 + 2 \eta G _ { + } ^ { \prime } ( 0 ) ) ( 1 + \kappa ^ { 2 r - 1 } \| u _ { \rho } \| _ { \rho } ) + \bigg ( \frac { r } { e \eta G _ { + } ^ { \prime } ( 0 ) } \bigg ) ^ { r } \| u _ { \rho } \| _ { \rho } \bigg ] | D | ^ { - \frac { r } { 2 r + s } } \log { \frac { 1 2 } { \delta } } } \\ & { \quad + 3 \cdot 2 ^ { p } C _ { r , 2 } \frac { | D | ^ { \frac { p + 1 } { 2 r } } } { \sigma ^ { 2 p } } + 2 ^ { p + \frac { 3 } { 2 } } C _ { r , 3 } C _ { 0 , 1 } | D | ^ { - \frac { \nu } { 2 r + s } } \operatorname* { m a x } \{ | D | ^ { - \frac { r - \frac { 1 } { 2 r + s } } } \log { | D | } , | D | ^ { - \frac { 1 } { 2 r + s } } , \frac { | D | ^ { \frac { p + 3 } { 2 r } } \log { | D | } } { \sigma ^ { 2 p } } \} \log { \frac { 6 } { \delta } } } \\ & { \quad + 2 ^ { p + 1 } C _ { r , 2 } ^ { \prime } ( C _ { 0 , 3 } + 1 ) ^ { \frac { 1 } { 2 } } C _ { 0 , 2 } \sqrt { m } | D | ^ { - \frac { 2 r + s - 1 } { 4 r + 2 s } } ( \log { | D | } ) ^ { 2 } \frac { | D | ^ { \frac { p + 1 } { 2 r + s } } } { \sigma ^ { 2 p } } ( \log { \frac { 4 8 m } { \delta } } ) ^ { 3 } } \\ &  \quad + 2 ^ { p + \frac { 3 } { 2 } } C _ { r , 3 } ^ { \prime } C _ { 0 , 1 } C _ { 0 , 2 } ( C _ { 0 , 3 } + 1 ) m | D | ^ { - \frac { 4 r - 1 } { 4 r + 2 s } } \operatorname* { m a x } \{ ( \log  | D |  \end{array}
$$

Note that

$$
\log ^ { 6 } { \frac { 4 8 m } { \delta } } \leq 2 ^ { 6 } \left( \log ^ { 6 } { \frac { 4 8 } { \delta } } + \log ^ { 6 } m \right) \leq 2 ^ { 6 } \log ^ { 6 } { \frac { 4 8 } { \delta } } \left( \log ^ { 6 } | D | + 1 \right) ,
$$

and $D ^ { - \alpha } \log | D | = \alpha ^ { - 1 } D ^ { - \alpha } \log | D | ^ { \alpha } \leq \alpha ^ { - 1 }$ with $\begin{array} { r } { \alpha = \frac { r - \frac { 1 } { 2 } } { 2 r + s } } \end{array}$ . Since $r > { \frac { 1 } { 2 } }$ , by combining the condition on m in (16), we have with confidence at least 1−δ,

$$
\left\| \bar { f } _ { T + 1 , D } - f _ { \rho } \right\| _ { \rho } \leq \tilde { C } _ { 1 } \operatorname* { m a x } \left\{ | D | ^ { - \frac { r } { 2 r + s } } , \frac { | D | ^ { \frac { p + 1 } { 2 r + s } } } { \sigma ^ { 2 p } } \right\} \left( \log \frac { 4 8 } { \delta } \right) ^ { 6 } ,
$$

where $\begin{array} { r } { \bar { C } _ { 1 } : = C _ { 0 , 1 } ( 1 + 2 \eta C _ { + } ^ { \prime } ( 0 ) ) ( 1 + \kappa ^ { 2 r - 1 } \| u _ { \rho } \| _ { \rho } ) + \left( \frac { r } { \omega \eta C _ { \nu } ^ { \prime } ( 0 ) } \right) ^ { r } \| u _ { \rho } \| _ { \rho } + 3 \cdot 2 ^ { p } C _ { r , 2 } + 2 ^ { p + \frac { 3 } { 2 } } C _ { r , 3 } C _ { 0 , 1 } \frac { 4 r + 2 s } { 2 r - 1 } + 2 ^ { p + 5 } C _ { r , 2 } ^ { \prime } ( C _ { 0 , 3 } + } \end{array}$ $\begin{array} { r } { 1 ) ^ { \frac { 1 } { 2 } } C _ { 0 , 2 } + 2 ^ { p + \frac { 1 5 } { 2 } } C _ { r , 3 } ^ { \prime } C _ { 0 , 1 } C _ { 0 , 2 } ( C _ { 0 , 3 } + 1 ) \frac { 4 r + 2 s } { 2 r - 1 } } \end{array}$ . This completes the proof of Theorem 1. □

## 6.3 Proof of Theorem 2

In this subsection, we aim to derive the explicit learning rates in expectation for the DKRGD based on the error   
decomposition in Subsection 5.3. To achieve this, we first establish the error bounds in probability for $\left\| f _ { T + 1 , D _ { j } } - f _ { \rho } \right\| _ { \rho }$   
and $\left. f _ { T + 1 , D _ { j } } ^ { * } - f _ { \rho } \right. _ { \rho }$ for each $j \in \{ 1 , \dots , m \}$ , and then obtain their expected error bounds by leveraging the formula ρ

$$
\mathbb { E } [ \xi ] = \int _ { 0 } ^ { \infty } { \mathrm { P r o b } } ( \xi > t ) { \mathrm { d } } t ,\tag{82}
$$

for nonnegative random variables $\xi .$

For the nonnegative random variable $\xi = \left\| \bar { f } _ { T + 1 , D } - f _ { \rho } \right\| _ { \rho } ^ { 2 } ,$ (17) in Theorem 1 immediately yields that, for any u satisfying $\begin{array} { r } { u \geq \tilde { C } _ { 1 } ^ { 2 } \left( \log 4 8 \right) ^ { 1 2 } \operatorname* { m a x } \left\{ | D | ^ { - \frac { r } { 2 r + s } } , \frac { | D | ^ { \frac { p + 1 } { 2 r + s } } } { \sigma ^ { 2 p } } \right\} ^ { 2 } } \end{array}$ , there holds

$$
\mathrm { P r o b } ( \xi > u ) = \mathrm { P r o b } ( \sqrt { \xi } > \sqrt { u } ) \le 4 8 \exp \left\{ - \left( { \tilde { C } } _ { 1 } \right) ^ { - \frac { 1 } { 6 } } \operatorname* { m a x } \left\{ | D | ^ { - \frac { r } { 2 r + 5 } } , \frac { | D | ^ { \frac { p + 1 } { 2 r + 5 } } } { \sigma ^ { 2 p } } \right\} ^ { - \frac { 1 } { 6 } } u ^ { \frac { 1 } { 1 2 } } \right\} .
$$

Then applying the formula $\begin{array} { r } { \mathbb { E } [ \xi ] = \int _ { 0 } ^ { \infty } \mathrm { P r o b } ( \xi > t ) \mathrm { d } t \mathrm { ~ t o ~ } \xi . } \end{array}$ we have

$$
\begin{array} { r l } & { \mathbb { E } [ \left\| \bar { f } _ { T + 1 , D } - f _ { \rho } \right\| _ { \rho } ^ { 2 } ] } \\ & { \le \tilde { C } _ { 1 } ^ { 2 } \left( \log 4 8 \right) ^ { 1 2 } \operatorname* { m a x } \left\{ | D | ^ { - \frac { r } { 2 r + s } } , \frac { | D | ^ { \frac { p + 1 } { 2 r + s } } } { \sigma ^ { 2 p } } \right\} ^ { 2 } + 4 8 \int _ { 0 } ^ { \infty } \exp \left\{ - ( \tilde { C } _ { 1 } ) ^ { - \frac { 1 } { 6 } } \operatorname* { m a x } \left\{ | D | ^ { - \frac { r } { 2 r + s } } , \frac { | D | ^ { \frac { p + 1 } { 2 r + s } } } { \sigma ^ { 2 p } } \right\} ^ { - \frac { 1 } { 6 } } u ^ { \frac { 1 } { 1 2 } } \right\} \mathrm { d } u } \\ & { = \tilde { C } _ { 1 } ^ { 2 } \left( \log 4 8 \right) ^ { 1 2 } \operatorname* { m a x } \left\{ | D | ^ { - \frac { r } { 2 r + s } } , \frac { | D | ^ { \frac { p + 1 } { 2 r + s } } } { \sigma ^ { 2 p } } \right\} ^ { 2 } + 5 7 6 \tilde { C } _ { 1 } ^ { 2 } \operatorname* { m a x } \left\{ | D | ^ { - \frac { r } { 2 r + s } } , \frac { | D | ^ { \frac { p + 1 } { 2 r + s } } } { \sigma ^ { 2 p } } \right\} ^ { 2 } \int _ { 0 } ^ { \infty } t ^ { 1 1 } e ^ { - t } \mathrm { d } t } \\ & { = \tilde { C } _ { 1 } ^ { 2 } \left[ \left( \log 4 8 \right) ^ { 1 2 } + 5 7 6 \Gamma ( 1 2 ) \right] \operatorname* { m a x } \left\{ | D | ^ { - \frac { 2 r } { 2 r + s } } , \frac { | D | ^ { \frac { 2 p + 2 } { 2 r + s } } } { \sigma ^ { 4 p } } \right\} . } \end{array}\tag{83}
$$

We now turn to estimate the term $\left\| f _ { T + 1 , D _ { j } } - f _ { \rho } \right\| _ { \rho }$ . By substituting the bounds in Lemma 4 and Lemma 6 into Proposition 5 and employing the intermediate results in proving Theorem 1, we can derive that with confidence at least 1−3δ, there holds

$$
\begin{array} { r l } & { \left\| f _ { T + 1 , D _ { j } } - f _ { \rho } \right\| _ { \rho } \leq ( 1 + \eta G _ { + } ^ { \prime } ( 0 ) \lambda T ) ( 1 + \kappa ^ { 2 r - 1 } \| u _ { \rho } \| _ { \rho } ) \mathcal { A } _ { | D _ { j } | , \lambda } ^ { 2 } \mathcal { B } _ { | D _ { j } | , \lambda } \left( \log \frac { 8 } { \delta } \right) ^ { 5 } } \\ & { \qquad + C _ { r , 2 } \left( \mathcal { A } _ { | D _ { j } | , \lambda } + \displaystyle \frac { 1 } { 2 } \right) \frac { T ^ { p + 1 } } { \sigma ^ { 2 p } } \left( \log \frac { 8 } { \delta } \right) ^ { 2 } + \left( \frac { r } { e \eta G _ { + } ^ { \prime } ( 0 ) } \right) ^ { r } \| u _ { \rho } \| _ { \rho } T ^ { - r } } \\ & { \qquad + C _ { r , 3 } \mathcal { A } _ { | D _ { j } | , \lambda } ^ { 2 } \mathcal { B } _ { | D _ { j } | , \lambda } \operatorname* { m a x } \left\{ T ^ { - ( r - \frac { 1 } { 2 } ) } \log T , T ^ { - 1 } , \frac { T ^ { p + \frac { 3 } { 2 } } \log T } { \sigma ^ { 2 p } } \right\} \left( \log \frac { 8 } { \delta } \right) ^ { 5 } . } \end{array}\tag{84}
$$

Since the condition on m in (18) also implies (81), we can adopt the bounds on $B _ { | D _ { j } | , \lambda } , \mathcal { R } _ { | D _ { j } | , \lambda }$ and $\boldsymbol { A } _ { | D _ { j } | , \boldsymbol { \lambda } }$ mentioned in the proof of Theorem 1 to further derive explicit rate in probability for $\left\| f _ { T + 1 , D _ { j } } - f _ { \rho } \right\| _ { \rho }$ . Since $\lambda = | D | ^ { - \frac { 1 } { 2 r + s } }$ and $T = \left\lceil | D | ^ { \frac { 1 } { 2 r + s } } \right\rceil$ , scaling 3δ to δ yields that with confidence at least 1−δ, there holds

$$
\left\| f _ { T + 1 , D _ { j } } - f _ { \rho } \right\| _ { \rho } \leq \tilde { C } _ { r , 2 } \operatorname* { m a x } \left\{ \mathcal { B } _ { | D _ { j } | , \lambda } , \frac { T ^ { p + 1 } } { \sigma ^ { 2 p } } , \lambda ^ { r } , \mathcal { B } _ { | D _ { j } | , \lambda } \frac { T ^ { p + \frac { 3 } { 2 } } \log T } { \sigma ^ { 2 p } } \right\} \left( \log \frac { 2 4 } { \delta } \right) ^ { 5 } ,
$$

where $\begin{array} { r } { \tilde { C } _ { r , 2 } : = \left( ( 1 + 2 \eta G _ { + } ^ { \prime } ( 0 ) ) ( 1 + \kappa ^ { 2 r - 1 } \| u _ { \rho } \| _ { \rho } ) + \frac { 3 } { 2 } C _ { r , 2 } + \frac { 2 C _ { r , 3 } } { 2 r - 1 } \right) ( C _ { 0 , 3 } + 1 ) + \left( \frac { r } { e \eta G _ { + } ^ { \prime } ( 0 ) } \right) ^ { r } \| u _ { \rho } \| _ { \rho } . } \end{array}$ . Then, we can easily apply (82) to obtain that for any $j \in \{ 1 , \dots , m \}$

$$
\begin{array} { r l } { \displaystyle \mathbb { E } \left[ \left\| f _ { T + 1 , D _ { j } } - f _ { \rho } \right\| _ { \rho } ^ { 2 } \right] \leq \tilde { C } _ { r , 2 } ^ { 2 } \operatorname* { m a x } \left\{ \mathcal { B } _ { | D _ { j } | , \lambda , } \frac { T ^ { p + 1 } } { \sigma ^ { 2 p } } , \lambda ^ { r } , \mathcal { B } _ { | D _ { j } | , \lambda } \frac { T ^ { p + \frac { 3 } { 2 } } \log T } { \sigma ^ { 2 p } } \right\} ^ { 2 } ( \log 2 4 ) ^ { 1 0 } } & { } \\ { \displaystyle + 2 4 \int _ { 0 } ^ { \infty } \exp \left\{ - ( \tilde { C } _ { r , 2 } ) ^ { - \frac { 1 } { 5 } } \operatorname* { m a x } \left\{ \mathcal { B } _ { | D _ { j } | , \lambda , } \frac { T ^ { p + 1 } } { \sigma ^ { 2 p } } , \lambda ^ { r } , \mathcal { B } _ { | D _ { j } | , \lambda } \frac { T ^ { p + \frac { 3 } { 2 } } \log T } { \sigma ^ { 2 p } } \right\} ^ { - \frac { 1 } { 5 } } u ^ { \frac { 1 } { 1 0 } } \right\} \mathrm { d } u } & { } \\ { \displaystyle = \tilde { C } _ { r , 2 } ^ { 2 } \left[ ( \log 2 4 ) ^ { 1 0 } + 2 4 0 \Gamma ( 1 0 ) \right] \operatorname* { m a x } \left\{ \mathcal { B } _ { | D _ { j } | , \lambda , } \frac { T ^ { p + 1 } } { \sigma ^ { 2 p } } , \lambda ^ { r } , \mathcal { B } _ { | D _ { j } | , \lambda } \frac { T ^ { p + \frac { 3 } { 2 } } \log T } { \sigma ^ { 2 p } } \right\} ^ { 2 } . } \end{array}\tag{85}
$$

In the following, we will estimate the term $\left. f _ { T + 1 , D _ { j } } ^ { * } - f _ { \rho } \right. _ { \rho }$ for each $j \in \{ 1 , \dots , m \}$ by applying the derived decomposition of $f _ { T + 1 , D _ { j } } ^ { * } - f _ { \rho }$ for diferent cases of r in Subsection 5.3. By invoking Equation (22) in [8, Prop. 4.2] under the constant step size η, we obtain that for any $\tau , \lambda > 0$ , there holds

$$
\begin{array} { r } { \left\| ( \lambda I + L _ { K , D } ) ^ { \tau } \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { T } \right\| = C _ { \tau } \left( \lambda ^ { \tau } + T ^ { - \tau } \right) . } \end{array}
$$

where $\begin{array} { r } { C _ { \tau } : = 2 ^ { \tau } \left[ 1 + \left( \frac { \tau } { e G _ { + } ^ { \prime } ( 0 ) } \right) ^ { \tau } \eta ^ { - \tau } \right] } \end{array}$ . For brevity, we denote $C _ { r } : = C _ { \tau } | _ { \tau = r }$

For the regime $\textstyle { \frac { 1 } { 2 } } < r \leq 1$ , by substituting the corresponding estimates in Lemma $6 ,$ Lemma 7 and Lemma 9 into (46), we obtain that with confidence at least $1 - \delta$

$$
\Big \| f _ { T + 1 , D _ { j } } ^ { * } - f _ { \rho } \Big \| _ { \rho } \leq C _ { r } \left( \lambda ^ { r } + T ^ { - r } \right) A _ { | D _ { j } | , \lambda } ^ { 2 r } \left( \log \frac \otimes \delta \right) ^ { 4 r } \| u _ { \rho } \| _ { \rho } + \frac { C _ { \eta } ^ { \prime } } { G _ { + } ^ { \prime } ( 0 ) } \left( \lambda ^ { \frac 1 2 } T + T ^ { \frac 1 2 } \right) \kappa c _ { p } \left( 2 M \right) ^ { 2 p + 1 } \frac { T ^ { p + \frac 1 2 } } { \sigma ^ { 2 p } } .
$$

For $\lambda = | D | ^ { - \frac { 1 } { 2 r + s } }$ and $T = \left\lceil | D | ^ { \frac { 1 } { 2 r + s } } \right\rceil$ , combining the partition size m specified in (18) and the bound on $\boldsymbol { A } _ { | D _ { j } | , \boldsymbol { \lambda } }$ , the preceding inequality simplifies to the following high-probability guarantee:

$$
\left\| f _ { T + 1 , D _ { j } } ^ { * } - f _ { \rho } \right\| _ { \rho } \leq C _ { r , 4 } \operatorname* { m a x } \left\{ \lambda ^ { r } , \frac { T ^ { p + 1 } } { \sigma ^ { 2 p } } \right\} \left( \log \frac { 8 } { \delta } \right) ^ { 4 r } ,
$$

where $\begin{array} { r } { C _ { r , 4 } : = 2 C _ { r } ( C _ { 0 , 3 } + 1 ) ^ { r } \left. u _ { \rho } \right. _ { \rho } + \frac { C _ { \eta } ^ { \prime } } { G _ { \bot } ^ { \prime } ( 0 ) } ( \sqrt { 2 } + 1 ) \kappa c _ { p } \left( 2 M \right) ^ { 2 p + 1 } } \end{array}$ . Finally, applying the tail-expectation identity (82) to the random variabl $\xi = \left\| f _ { T + 1 , D _ { j } } ^ { * } - f _ { \rho } \right\| _ { \rho }$ yields the expected error bound

$$
\mathbb { E } \left[ \left. f _ { T + 1 , D _ { j } } ^ { * } - f _ { \rho } \right. _ { \rho } \right] \leq C _ { r , 4 } \left[ ( \log 8 ) ^ { 4 r } + 3 2 r \Gamma ( 4 r ) \right] \operatorname* { m a x } \left\{ \lambda ^ { r } , \frac { T ^ { p + 1 } } { \sigma ^ { 2 p } } \right\} .
$$

For $\textstyle 1 < r \leq { \frac { 3 } { 2 } }$ , combining (48), Lemma 4, Lemma 7 and Lemma 9, it follows that with confidence at least $1 - \delta .$

$$
\Big \| f _ { T + 1 , D _ { j } } ^ { * } - f _ { p } \Big \| _ { \rho } \leq C _ { r } \left( \lambda ^ { r } + T ^ { - r } \right) \left( \frac { \mathcal { B } _ { | D _ { j } | , \lambda } } { \sqrt { \lambda } } + 1 \right) ^ { 2 r } \left( \log \frac 2 \delta \right) ^ { 2 r } \| u _ { \rho } \| _ { \rho } + \frac { C _ { \eta } ^ { r } } { G _ { + } ^ { r } ( 0 ) } \left( \lambda ^ { \frac 1 2 } T + T ^ { \frac 1 2 } \right) \kappa c _ { p } \left( 2 M \right) ^ { 2 p + 1 } \frac { T ^ { p + \frac 1 2 } } { \sigma ^ { 2 p } } .
$$

Note that $\lambda = | D | ^ { - \frac { 1 } { 2 r + s } }$ and $T = \left\lceil | D | ^ { \frac { 1 } { 2 r + s } } \right\rceil$ . When $r > 1$ and m satisfies (18), we have

$$
\frac { \mathcal { B } _ { | D _ { j } | , \lambda } } { \sqrt { \lambda } } + 1 \leq C _ { 0 , 1 } \sqrt { m } | D | ^ { - \frac { r - \frac { 1 } { 2 } } { 2 r + s } } + 1 \leq C _ { 0 , 1 } + 1 .
$$

Then, we have that with confidence at least $1 { - } \delta .$ there holds

$$
\left\| f _ { T + 1 , D _ { j } } ^ { * } - f _ { \rho } \right\| _ { \rho } \leq C _ { r , 5 } \operatorname* { m a x } \left\{ \lambda ^ { r } , \frac { T ^ { p + 1 } } { \sigma ^ { 2 p } } \right\} \left( \log \frac 2 \delta \right) ^ { 2 r } .
$$

where $\begin{array} { r } { C _ { r , 5 } : = 2 C _ { r } ( C _ { 0 , 1 } + 1 ) ^ { 2 r } \left\| u _ { \rho } \right\| _ { \rho } + \frac { C _ { \eta } ^ { \prime } } { G _ { \pm } ^ { \prime } ( 0 ) } ( \sqrt { 2 } + 1 ) \kappa c _ { p } \left( 2 M \right) ^ { 2 p + 1 } } \end{array}$ . Further, we can also derive the expected error bound

$$
\begin{array} { r } { \mathbb { E } \left[ \left. f _ { T + 1 , D _ { j } } ^ { * } - f _ { \rho } \right. _ { \rho } \right] \leq C _ { r , 5 } \left[ ( \log 2 ) ^ { 2 r } + 4 r \Gamma ( 2 r ) \right] \operatorname* { m a x } \left\{ \lambda ^ { r } , \frac { T ^ { p + 1 } } { \sigma ^ { 2 p } } \right\} . } \end{array}
$$

For the case of $\textstyle { \frac { 3 } { 2 } } < r \leq { \frac { 5 } { 2 } }$ , putting the estimates in Lemma 4, Lemma 7 and Lemma 9 into (53) yields that with confidence at least $1 - \delta .$

$$
\begin{array} { r l } & { \left\| f _ { T + 1 , D _ { j } } ^ { * } - f _ { p } \right\| _ { \rho } \leq C _ { r } \left( \lambda ^ { r } + T ^ { - r } \right) \left\| u _ { \rho } \right\| _ { \rho } \left( \frac { B _ { | D _ { j } | , \lambda } } { \sqrt { \lambda } } + 1 \right) \left( 1 + \left( \frac { B _ { | D _ { j } | , \lambda } } { \sqrt { \lambda } } + 1 \right) ^ { 2 r - 3 } \right) \left( \log \frac { 4 } { \delta } \right) ^ { 3 } } \\ & { \qquad + C _ { 1 } \left( \lambda + T ^ { - 1 } \right) \left\| u _ { \rho } \right\| _ { \rho } ( 1 + \kappa ^ { 2 } ) \left( \frac { B _ { | D _ { j } | , \lambda } } { \sqrt { \lambda } } + 1 \right) ^ { 2 } B _ { | D _ { j } | , \lambda } \left( \log \frac { 4 } { \delta } \right) ^ { 3 } } \\ & { \qquad + C _ { r } \left( \lambda ^ { r } + T ^ { - r } \right) \left( \frac { B _ { | D _ { j } | , \lambda } } { \sqrt { \lambda } } + 1 \right) \left\| u _ { \rho } \right\| _ { \rho } \log \frac { 4 } { \delta } + \frac { C _ { \eta } ^ { \prime } } { G _ { + } ^ { \prime } ( 0 ) } \left( \lambda ^ { \frac { 1 } { 2 } } T + T ^ { \frac { 1 } { 2 } } \right) \kappa c _ { p } \left( 2 M \right) ^ { 2 p + 1 } \frac { T ^ { p + \frac { 1 } { 2 } } } { \sigma ^ { 2 p } } . } \end{array}
$$

Analogously, for $\lambda = | D | ^ { - \frac { 1 } { 2 r + s } }$ and $T = \left\lceil | D | ^ { \frac { 1 } { 2 r + s } } \right\rceil$ , we have that with confidence at least $1 { - } \delta .$

$$
\left\| f _ { T + 1 , D _ { j } } ^ { * } - f _ { \rho } \right\| _ { \rho } \leq C _ { r , 6 } \operatorname* { m a x } \left\{ \lambda ^ { r } , \lambda \mathcal { B } _ { | D _ { j } | , \lambda } , \frac { T ^ { p + 1 } } { \sigma ^ { 2 p } } \right\} \left( \log \frac { 4 } { \delta } \right) ^ { 3 } .
$$

where $\begin{array} { r } { C _ { r , 6 } : = 6 C _ { r } \left\| u _ { \rho } \right\| _ { \rho } ( C _ { 0 , 1 } + 1 ) ^ { 2 r - 2 } + 2 C _ { 1 } \left\| u _ { \rho } \right\| _ { \rho } ( 1 + \kappa ^ { 2 } ) ( C _ { 0 , 1 } + 1 ) ^ { 2 } + \frac { C _ { \eta } ^ { \prime } } { G _ { \pm } ^ { \prime } ( 0 ) } ( \sqrt { 2 } + 1 ) \kappa c _ { p } \left( 2 M \right) ^ { 2 p + 1 } } \end{array}$ . Then it can be easily derived that

$$
\begin{array} { r } { \mathbb { E } \left[ \left\| f _ { T + 1 , D _ { j } } ^ { * } - f _ { \rho } \right\| _ { \rho } \right] \leq C _ { r , 6 } \left[ ( \log 4 ) ^ { 3 } + 1 2 \Gamma ( 3 ) \right] \operatorname* { m a x } \left\{ \lambda ^ { r } , \lambda \mathcal { B } _ { | D _ { j } | , \lambda } , \frac { T ^ { p + 1 } } { \sigma ^ { 2 p } } \right\} . } \end{array}
$$

When $\textstyle r > { \frac { 5 } { 2 } }$ , by (54), Lemma 4, Lemma 7 and Lemma 9, with confidence at least $1 - \delta .$ , there holds

$$
\begin{array} { r l } & { \left\| f _ { T + 1 , D _ { j } } ^ { * } - f _ { p } \right\| _ { \rho } \leq \left( r - \frac { 3 } { 2 } \right) \kappa ^ { 2 r - 5 } C _ { \frac { 3 } { 2 } } \left( \lambda ^ { \frac { 3 } { 2 } } + T ^ { - \frac { 3 } { 2 } } \right) \frac { 4 \kappa ^ { 2 } } { \sqrt { | D _ { j } | } } \| u _ { \rho } \| _ { \rho } \left( \frac { B _ { | D _ { j } | , \lambda } } { \sqrt { \lambda } } + 1 \right) \left( \log \frac { 6 } { \delta } \right) ^ { 2 } } \\ & { \qquad + C _ { 1 } ( 1 + \kappa ^ { 2 } ) ^ { r - \frac { 3 } { 2 } } \| u _ { \rho } \| _ { \rho } \left( \lambda + T ^ { - 1 } \right) \left( \frac { B _ { | D _ { j } | , \lambda } } { \sqrt { \lambda } } + 1 \right) ^ { 2 } B _ { | D _ { j } | , \lambda } \left( \log \frac { 6 } { \delta } \right) ^ { 3 } } \\ & { \qquad + C _ { r } \left( \lambda ^ { r } + T ^ { - r } \right) \left( \frac { B _ { | D _ { j } | , \lambda } } { \sqrt { \lambda } } + 1 \right) \| u _ { \rho } \| _ { \rho } \log \frac { 6 } { \delta } + \frac { C _ { \eta } ^ { \prime } } { G _ { + } ^ { \prime } ( 0 ) } \left( \lambda ^ { \frac { 1 } { 2 } } T + T ^ { \frac { 1 } { 2 } } \right) \kappa c _ { p } \left( 2 M \right) ^ { 2 p + 1 } \frac { T ^ { p + \frac { 1 } { 2 } } } { \sigma ^ { 2 p } } , } \end{array}
$$

Note that with the partition size m specified in (18), we also have $\lambda ^ { - \frac { 1 } { 2 } } \mathcal { B } _ { | D _ { j } | , \lambda } \le C _ { 0 , 1 } + 1$ . Hence, for $\lambda = | D | ^ { - \frac { 1 } { 2 r + s } }$ and $T = \left\lceil | D | ^ { \frac { 1 } { 2 r + s } } \right\rceil$ , there holds with confidence at least 1−δ that

$$
\left\| f _ { T + 1 , D _ { j } } ^ { * } - f _ { \rho } \right\| _ { \rho } \leq C _ { r , 7 } \operatorname* { m a x } \left\{ \lambda ^ { r } , \frac { \lambda ^ { \frac { 3 } { 2 } } } { \sqrt { | D _ { j } | } } , \lambda B _ { | D _ { j } | , \lambda } , \frac { T ^ { p + 1 } } { \sigma ^ { 2 p } } \right\} \left( \log \frac { 6 } { \delta } \right) ^ { 3 } .
$$

where $\begin{array} { r } { C _ { r , 7 } : = \| u _ { \rho } \| _ { \rho } ( C _ { 0 , 1 } + 1 ) ^ { 2 } \left[ ( 8 r - 1 2 ) \kappa ^ { 2 r - 3 } C _ { \frac { 3 } { 2 } } + 2 C _ { 1 } ( 1 + \kappa ^ { 2 } ) ^ { r - \frac { 3 } { 2 } } + 2 C _ { r } \right] + \frac { C _ { r } ^ { \prime } } { G _ { + } ^ { \prime } ( 0 ) } ( \sqrt { 2 } + 1 ) \kappa c _ { p } \left( 2 M \right) ^ { 2 p + 1 } } \end{array}$ . Further, we can apply (82) to obtain

$$
\begin{array} { r } { \mathbb { E } \left[ \left\| f _ { T + 1 , D _ { j } } ^ { * } - f _ { \rho } \right\| _ { \rho } \right] \leq C _ { r , 7 } \left[ \left( \log 6 \right) ^ { 3 } + 1 8 \Gamma ( 3 ) \right] \operatorname* { m a x } \left\{ \lambda ^ { r } , \frac { \lambda ^ { \frac { 3 } { 2 } } } { \sqrt { | D _ { j } | } } , \lambda \mathcal { B } _ { | D _ { j } | , \lambda } , \frac { T ^ { p + 1 } } { \sigma ^ { 2 p } } \right\} . } \end{array}
$$

Before concluding the proof, we remark that for $\textstyle r > { \frac { 5 } { 2 } }$ , the constraint on partition size m derived from the above bound is more stringent than condition (18). Specifically, ensuring $\begin{array} { r } { \frac { \lambda ^ { \frac { 3 } { 2 } } } { \sqrt { | D _ { j } | } } = \mathcal { O } ( | D | ^ { - \frac { 2 r } { 2 r + s } } ) } \end{array}$ requires $m \leq | D | ^ { \frac { 3 + s } { 2 r + s } }$ , which imposes a tighter restriction on m for lager r. To overcome this bottleneck, we employ (83) to derive the explicit learning rates in Theorem 2 for $r > { \frac { 5 } { 2 } }$ . Conversely, for $\begin{array} { r } { \frac { 5 - s } { 2 } < r \leq \frac { 5 } { 2 } } \end{array}$ , condition (16) is less restrictive than (18); thus, we also apply (83) to establish the rates. This distinction clarifies why the regularity regimes in (18) difer from those implied by the intermediate technical conditions.

Finally, to derive the explicit learning rates in Theorem 2, we need the following results

$$
\lambda ^ { 2 } \mathcal { B } _ { | D _ { j } | , \lambda } ^ { 2 } = C _ { 0 , 1 } ^ { 2 } m | D | ^ { - \frac { 2 r + 2 } { 2 r + s } } \leq C _ { 0 , 1 } ^ { 2 } | D | ^ { - \frac { 2 r } { 2 r + s } } ,
$$

and

$$
\sum _ { j = 1 } ^ { m } \frac { | D _ { j } | ^ { 2 } } { | D | ^ { 2 } } \mathcal { B } _ { | D _ { j } | , \lambda } ^ { 2 } = \sum _ { j = 1 } ^ { m } \frac { | D _ { j } | ^ { 2 } } { | D | ^ { 2 } } C _ { 0 , 1 } ^ { 2 } m | D | ^ { - \frac { 2 r } { 2 r + s } } \leq C _ { 0 , 1 } ^ { 2 } | D | ^ { - \frac { 2 r } { 2 r + s } } ,
$$

which can be verified by the partition size m specified in (18). Since $\left\| \mathbb { E } \left[ f _ { T + 1 , D _ { j } } \right] - f _ { \rho } \right\| _ { \rho } \leq \mathbb { E } \left[ \left\| f _ { T + 1 , D _ { j } } ^ { * } - f _ { \rho } \right\| _ { \rho } \right]$ , it follows from (42) that

$$
\mathbb { E } \left[ \left. \bar { f } _ { T + 1 , D } - f _ { \rho } \right. _ { \rho } ^ { 2 } \right] \leq \sum _ { j = 1 } ^ { m } \frac { | D _ { j } | ^ { 2 } } { | D | ^ { 2 } } \mathbb { E } \left[ \left. f _ { T + 1 , D _ { j } } - f _ { \rho } \right. _ { \rho } ^ { 2 } \right] + \sum _ { j = 1 } ^ { m } \frac { | D _ { j } | } { | D | } \left\{ \mathbb { E } \left[ \left. f _ { T + 1 , D _ { j } } ^ { * } - f _ { \rho } \right. _ { \rho } \right] \right\} ^ { 2 } .
$$

For $\textstyle { \frac { 1 } { 2 } } < r \leq { \frac { 5 - s } { 2 } }$ , combining (85) and the above bounds on $\mathbb { E } \left[ \left. f _ { T + 1 , D _ { j } } ^ { * } - f _ { \rho } \right. _ { \rho } \right]$ yields that

$$
\begin{array} { l } { { \displaystyle \mathbb { E } \Big [ \big \| \bar { f } _ { T + 1 , D } - f _ { \rho } \big \| _ { \rho } ^ { 2 } \Big ] } } \\ { { \displaystyle \le A _ { r } \sum _ { j = 1 } ^ { m } \frac { \big | D _ { j } \big | ^ { 2 } } { | D | ^ { 2 } } \operatorname* { m a x } \left\{ { \mathcal { B } _ { | D _ { j } | , \lambda } , \frac { T ^ { p + 1 } } { \sigma ^ { 2 p } } } , \lambda ^ { r } , { \mathcal { B } _ { | D _ { j } | , \lambda } \frac { T ^ { p + \frac { 3 } { 2 } } \log T } { \sigma ^ { 2 p } } } \right\} ^ { 2 } + B _ { r } \sum _ { j = 1 } ^ { m } \frac { \big | D _ { j } \big | } { | D | } \operatorname* { m a x } \left\{ \lambda ^ { r } , \frac { T ^ { p + 1 } } { \sigma ^ { 2 p } } , \lambda { \mathcal { B } _ { | D _ { j } | , \lambda } } \right\} ^ { 2 } } } \\ { { \displaystyle \le 2 ^ { 2 p + 2 } C _ { 0 , 1 } ^ { 2 } \left( A _ { r } \frac { 8 \big ( 2 r + s \big ) ^ { 2 } } { ( 2 r - 1 ) ^ { 2 } } + B _ { r } \right) \operatorname* { m a x } \left\{ \big | D | ^ { - \frac { 2 r } { 2 r + s } } , \frac { T ^ { 2 p + 2 } } { \sigma ^ { 4 p } } \right\} , } } \end{array}
$$

where $A _ { r } : = \tilde { C } _ { r , 2 } ^ { 2 } \left[ \left( \log { 2 4 } \right) ^ { 1 0 } + 2 4 0 \Gamma ( 1 0 ) \right] \mathrm { ~ a n d ~ } B _ { r } : = C _ { r , 4 } ^ { 2 } \left[ \left( \log { 8 } \right) ^ { 4 r } + 3 2 r \Gamma ( 4 r ) \right] ^ { 2 } + C _ { r , 5 } ^ { 2 } \left[ \left( \log { 2 } \right) ^ { 3 } + 6 \Gamma ( 3 ) \right] ^ { 2 } + C _ { r , 6 } ^ { 2 } \left[ \left( \log { 6 } \right) ^ { 3 } + \left( \log { 7 } \right) ^ { 4 } \right]$ $\boldsymbol { 1 8 \Gamma ( 3 ) } \boldsymbol { \ ] } ^ { 2 }$

The proof is completed by combining (83) for $\textstyle r > { \frac { 5 - s } { 2 } }$ and the above bound for $\textstyle { \frac { 1 } { 2 } } < r \leq { \frac { 5 - s } { 2 } }$ , where

$$
\tilde { C } _ { 2 } : = \tilde { C } _ { 1 } ^ { 2 } \big [ ( \log 4 8 ) ^ { 1 2 } + 5 7 6 \Gamma ( 1 2 ) \big ] + 2 ^ { 2 p + 2 } C _ { 0 , 1 } ^ { 2 } \left( A _ { r } \frac { 8 ( 2 r + s ) ^ { 2 } } { ( 2 r - 1 ) ^ { 2 } } + B _ { r } \right) .
$$

## 6.4 Proof of Theorem 3

To prove Theorem 3, we adopt the error decomposition in Subsection 5.4 and establish the high-probability bounds for $\left\| \bar { f } _ { T + 1 , D } ^ { l } - f _ { T + 1 , D } \right\| _ { \rho }$ and $\| f _ { T + 1 , D } - f _ { \rho } \| _ { \rho }$ , respectively.

First, we can directly apply the derived high-probability bounds for $\left\| f _ { T + 1 , D _ { j } } - f _ { \rho } \right\| _ { \rho }$ to estimate $\| f _ { T + 1 , D } - f _ { \rho } \| _ { \rho }$ ρ by substituting $D _ { j }$ with D in (84). Specifically, there holds with confidence at least $1 - \delta$ that

$$
\begin{array} { r l } & { \displaystyle \| f _ { T + 1 , D } - f _ { \rho } \| _ { \rho } \leq ( 1 + \eta G _ { + } ^ { \prime } ( 0 ) \lambda T ) ( 1 + \kappa ^ { 2 r - 1 } \| u _ { \rho } \| _ { \rho } ) \mathcal { A } _ { | D | , \lambda } ^ { 2 } \mathcal { B } _ { | D | , \lambda } \left( \log \frac { 2 4 } { \delta } \right) ^ { 5 } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad + C _ { r , 2 } \left( \mathcal { A } _ { | D | , \lambda } + \frac { 1 } { 2 } \right) \frac { T ^ { p + 1 } } { \sigma ^ { 2 p } } \left( \log \frac { 2 4 } { \delta } \right) ^ { 2 } + \left( \frac { r } { e \eta G _ { + } ^ { \prime } ( 0 ) } \right) ^ { r } \| u _ { \rho } \| _ { \rho } T ^ { - r } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad + C _ { r , 3 } \mathcal { A } _ { | D | , \lambda } ^ { 2 } \mathcal { B } _ { | D | , \lambda } \operatorname* { m a x } \left\{ T ^ { - ( r - \frac { 1 } { 2 } ) } \log T , T ^ { - 1 } , \frac { T ^ { p + \frac { 3 } { 2 } } \log T } { \sigma ^ { 2 p } } \right\} \left( \log \frac { 2 4 } { \delta } \right) ^ { 5 } . } \end{array}
$$

Since $\lambda = | D | ^ { - { \frac { 1 } { 2 r + s } } }$ and $T = \left\lceil | D | ^ { \frac { 1 } { 2 r + s } } \right\rceil$ , it can be directly derived that $B _ { | D | , \lambda } \leq C _ { 0 , 1 } | D | ^ { - { \frac { r } { 2 r + s } } }$ and $\mathcal { A } _ { | D | , \lambda } ^ { 2 } \le C _ { 0 , 3 } + 1$ by taking $m = 1$ on the bounds on $B _ { | D _ { j } | , \lambda } , \mathcal { R } _ { | D _ { j } | , \lambda }$ and $\boldsymbol { A } _ { | D _ { j } | , \boldsymbol { \lambda } }$ in the proof of Theorem 1. Hence, we have that with confidence at least $1 - \delta$

$$
\| f _ { T + 1 , D } - f _ { \rho } \| _ { \rho } \leq \tilde { C } _ { r , 3 } \operatorname* { m a x } \left\{ | D | ^ { - \frac { r } { 2 r + s } } , \frac { | D | ^ { \frac { p + 1 } { 2 r + s } } } { \sigma ^ { 2 p } } \right\} \left( \log \frac { 2 4 } { \delta } \right) ^ { 5 } ,\tag{86}
$$

where $\begin{array} { r } { \tilde { C } _ { r , 3 } : = \left( C _ { 0 , 1 } ( 1 + 2 \eta G _ { + } ^ { \prime } ( 0 ) ) ( 1 + \kappa ^ { 2 r - 1 } \| u _ { \rho } \| _ { \rho } ) + 3 \cdot 2 ^ { p } C _ { r , 2 } + 2 ^ { p + \frac { 3 } { 2 } } C _ { 0 , 1 } C _ { r , 3 } \frac { 4 r + 2 s } { 2 r - 1 } \right) ( C _ { 0 , 3 } + 1 ) + \left( \frac { r } { e \eta G _ { + } ^ { \prime } ( 0 ) } \right) ^ { r } \| u _ { \rho } \| _ { \rho } } \end{array}$ . It is evident that this result difers from that in [8] only by a constant factor.

Now we turn to estimate $\left. \bar { f } _ { T + 1 , D } ^ { l } - f _ { T + 1 , D } \right. _ { \rho }$ . By (59), we first need to estimate the bound of $\left. \bar { f } _ { T + 1 , D } ^ { 0 } - f _ { T + 1 , D } \right. _ { \rho } ,$ which equals to $\left\| \bar { f } _ { T + 1 , D } - f _ { T + 1 , D } \right\| _ { \rho }$ . For convenience, the decomposition $\bar { f } _ { T + 1 , D } - f _ { T + 1 , D } = \bar { f } _ { T + 1 , D } - f _ { T + 1 } + f _ { T + 1 } -$ $f _ { T + 1 , D }$ is adopted, which allows us to separately estimate the two terms $\left\| \bar { f } _ { T + 1 , D } - f _ { T + 1 } \right\| _ { \rho }$ and $\| f _ { T + 1 , D } - f _ { T + 1 } \| _ { \rho }$

To estimate $\bar { f } _ { T + 1 , D } - f _ { T + 1 }$ , by Proposition 2 and the proof of Theorem 1, we have that with confidence at least

## 1−δ, there holds

$$
\begin{array} { r l } & { \left\| \bar { f } _ { T + 1 , D } - f _ { T + 1 } \right\| _ { \rho } \leq \big ( 1 + 2 \eta G _ { + } ^ { \prime } ( 0 ) \big ) ( 1 + \kappa ^ { 2 r - 1 } \| u _ { \rho } \| _ { \rho } ) B _ { | D | , \lambda } \log \frac { 1 2 } { \delta } } \\ & { \qquad + C _ { r , 2 } \frac { T ^ { p + 1 } } { \sigma ^ { 2 p } } + C _ { r , 3 } \mathcal { B } _ { | D | , \lambda } \operatorname* { m a x } \left\{ T ^ { - ( r - \frac { 1 } { 2 } ) } \log T , T ^ { - 1 } , \frac { T ^ { p + \frac { 3 } { 2 } } \log T } { \sigma ^ { 2 p } } \right\} \log \frac { 6 } { \delta } } \\ & { \qquad + C _ { r , 2 } ^ { \prime } \frac { \operatorname* { m a x } } { 1 \leq j \leq m } \left\{ \mathcal { R } _ { | D _ { j } | , \lambda } A _ { | D _ { j } | , \lambda } \right\} \frac { T ^ { p + 1 } } { \sigma ^ { 2 p } } \log T \left( \log \frac { 4 8 m } { \delta } \right) ^ { 3 } } \\ & { \qquad + C _ { r , 3 } ^ { \prime } \frac { \operatorname* { m a x } } { 1 \leq j \leq m } \left\{ \mathcal { R } _ { | D _ { j } | , \lambda } B _ { | D _ { j } | , \lambda } A _ { | D _ { j } | , \lambda } ^ { 2 } \right\} \operatorname* { m a x } \left\{ \log T , T ^ { - ( r - \frac { 1 } { 2 } ) } ( \log T ) ^ { 2 } , \frac { T ^ { p + \frac { 3 } { 2 } } ( \log T ) ^ { 2 } } { \sigma ^ { 2 p } } \right\} \left( \log \frac { 4 8 m } { \delta } \right) ^ { 6 } . } \end{array}
$$

Since the condition (20) still implies (81), we can apply the bounds on $B _ { | D _ { j } | , \lambda } , \mathcal { R } _ { | D _ { j } | , \lambda }$ and $\mathcal { A } _ { | D _ { j } | , \lambda }$ in the proof of Theorem 1 to further simplify the above bound as

$$
\begin{array} { l } { \displaystyle \left\| \bar { f } _ { T + 1 , D } - f _ { T + 1 } \right\| _ { \rho } } \\ { \displaystyle \leq \tilde { C } _ { 3 , 1 } \operatorname* { m a x } \left\{ \mathcal { B } _ { | D | , \lambda } , \frac { T ^ { p + 1 } } { \sigma ^ { 2 p } } \right\} \left( \log \frac { 4 8 m } { \delta } \right) ^ { 6 } + \tilde { C } _ { 3 , 2 } \displaystyle \underset { 1 \leq j \leq m } { \operatorname* { m a x } } \left\{ \mathcal { R } _ { | D _ { j } | , \lambda } \mathcal { B } _ { | D _ { j } | , \lambda } \right\} \log T \left( \log \frac { 4 8 m } { \delta } \right) ^ { 6 } , } \end{array}\tag{87}
$$

where $\begin{array} { r } { \tilde { C } _ { 3 , 1 } : = ( 1 + 2 \eta G _ { + } ^ { \prime } ( 0 ) ) ( 1 + \kappa ^ { 2 r - 1 } \| u _ { \rho } \| _ { \rho } ) + C _ { r , 2 } + \sqrt { 2 } C _ { r , 3 } C _ { 0 , 1 } \frac { 4 r + 2 s } { 2 r - 1 } + C _ { r , 2 } ^ { \prime } C _ { 0 , 2 } ( C _ { 0 , 3 } + 1 ) ^ { \frac { 1 } { 2 } } + \sqrt { 2 } C _ { r , 3 } ^ { \prime } C _ { 0 , 1 } C _ { 0 , 2 } ( C _ { 0 , 3 } + 1 ) } \end{array}$ and $\begin{array} { r } { \tilde { C } _ { 3 , 2 } : = \frac { 2 } { 2 r - 1 } C _ { r , 3 } ^ { \prime } ( C _ { 0 , 3 } + 1 ) } \end{array}$

Since for each $f \in \mathcal { H } _ { K }$ , there holds $\left\| f \right\| _ { \rho } = \left\| L _ { K } ^ { \frac 1 2 } f \right\| _ { K }$ . Considering that $f _ { T + 1 , D } , f _ { T + 1 } \in \mathcal { H } _ { K }$ , we have

$$
\begin{array} { r } { \left\| f _ { T + 1 , D } - f _ { T + 1 } \right\| _ { \rho } = \left\| L _ { K } ^ { \frac 1 2 } ( f _ { T + 1 , D } - f _ { T + 1 } ) \right\| _ { K } \leq \left\| ( \lambda I + L _ { K } ) ^ { \frac 1 2 } ( f _ { T + 1 , D } - f _ { T + 1 } ) \right\| _ { K } . } \end{array}
$$

By Proposition 1, the proof of Theorem 1 and the partition size m specified in (20), we can obtain that with confidence at least 1−δ, there holds

$$
\begin{array} { r l } & { \left\| f _ { T + 1 , D } - f _ { T + 1 } \right\| _ { \rho } \leq ( 1 + 2 \eta G _ { + } ^ { \prime } ( 0 ) ) A _ { D , \lambda } ^ { 2 } ( \mathcal { P } _ { D , \lambda } + \mathcal { Q } _ { D , \lambda } \| f _ { \rho } \| _ { K } ) } \\ & { \qquad + C _ { r , 2 } \mathcal { A } _ { D , \lambda } \frac { T ^ { p + 1 } } { \sigma ^ { 2 p } } + C _ { r , 3 } \mathcal { A } _ { D , \lambda } ^ { 2 } \mathcal { Q } _ { D , \lambda } \operatorname* { m a x } \left\{ T ^ { - ( r - \frac { 1 } { 2 } ) } \log T , T ^ { - 1 } , \frac { T ^ { p + \frac { 3 } { 2 } } \log T } { \sigma ^ { 2 p } } \right\} } \\ & { \qquad \leq \tilde { C } _ { 3 , 3 } \operatorname* { m a x } \left\{ \mathcal { B } _ { | D | , \lambda } , \frac { T ^ { p + 1 } } { \sigma ^ { 2 p } } \right\} \left( \log \frac { 2 4 } { \delta } \right) ^ { 5 } , } \end{array}\tag{88}
$$

$$
\begin{array} { r } { \mathrm { w h e r e ~ } \tilde { C } _ { 3 , 3 } : = ( 1 + 2 \eta G _ { + } ^ { \prime } ( 0 ) ) ( 1 + \kappa ^ { 2 r - 1 } \| u _ { \rho } \| _ { \rho } ) ( C _ { 0 , 3 } + 1 ) + C _ { r , 2 } ( C _ { 0 , 3 } + 1 ) ^ { \frac { 1 } { 2 } } + \sqrt { 2 } C _ { r , 3 } C _ { 0 , 1 } \frac { 4 r + 2 s } { 2 r - 1 } ( C _ { 0 , 3 } + 1 ) . } \end{array}
$$

With the triangle inequality, we combine (87) and (88)to derive that with confidence at least 1−2δ, there holds

$$
\begin{array} { r l } & { \left\| \bar { f } _ { T + 1 , D } ^ { 0 } - f _ { T + 1 , D } \right\| _ { \rho } = \left\| \bar { f } _ { T + 1 , D } - f _ { T + 1 , D } \right\| _ { \rho } \leq \left\| \bar { f } _ { T + 1 , D } - f _ { T + 1 } \right\| _ { \rho } + \left\| f _ { T + 1 , D } - f _ { T + 1 } \right\| _ { \rho } } \\ & { \qquad \leq ( \tilde { C } _ { 3 , 1 } + \tilde { C } _ { 3 , 3 } ) \operatorname* { m a x } \left\{ \mathcal { B } _ { | D | , \lambda } , \frac { T ^ { p + 1 } } { \sigma ^ { 2 p } } \right\} \left( \log \frac { 4 8 m } { \delta } \right) ^ { 6 } } \\ & { \qquad + \tilde { C } _ { 3 , 2 } \underset { 1 \leq j \leq m } { \operatorname* { m a x } } \left\{ \mathcal { R } _ { | D _ { j } | , \lambda } \mathcal { B } _ { | D _ { j } | , \lambda } \right\} \log T \left( \log \frac { 4 8 m } { \delta } \right) ^ { 6 } . } \end{array}
$$

Then we can adopt (59) to bound $\left. \bar { f } _ { T + 1 , D } ^ { l } - f _ { T + 1 , D } \right. _ { \rho }$ . Note that bounding the sum term in (59) relies on the highprobability bounds of $\mathcal { Q } _ { \vert D \vert , \lambda } , \mathcal { Q } _ { \vert D _ { j } \vert , \lambda }$ and $\mathcal { A } _ { | D _ { j } | , \lambda }$ , which are also analyzed in deriving the bound of $\left. \bar { f } _ { T + 1 , D } ^ { 0 } - f _ { T + 1 , D } \right. _ { \rho } .$

Hence, by Lemma 4 and Lemma $6 ,$ we have that with confidence at least 1−2δ, there holds

$$
\begin{array} { r l } & { \quad \| f _ { T + 1 , D } - \bar { f } _ { T + 1 , D } ^ { l } \| _ { \rho } \leq \underset { 1 \leq j \leq m } { \operatorname* { m a x } } ( ( \mathcal { Q } _ { D , \lambda } + \mathcal { Q } _ { D _ { j } , \lambda } ) \mathcal { A } _ { D _ { j } , \lambda } C _ { \eta } ^ { \prime } ( \lambda ^ { \frac { 1 } { 2 } } T + T ^ { \frac { 1 } { 2 } } ) ) ^ { l } \| \bar { f } _ { T + 1 , D } ^ { \perp } - f _ { T + 1 , D } \| _ { \rho } } \\ & { \leq ( 4 + 2 \sqrt { 2 } ) ^ { l } ( C _ { \eta } ^ { \prime } ) ^ { l } ( C _ { 0 , 3 } + 1 ) ^ { \frac { l } { 2 } } \lambda ^ { - \frac { l } { 2 } } \underset { 1 \leq j \leq m } { \operatorname* { m a x } } B _ { | D _ { j } | , \lambda } ^ { l } \cdot ( \tilde { C } _ { 3 , 1 } + \tilde { C } _ { 3 , 3 } ) \operatorname* { m a x } \{ \mathcal { B } _ { | D | , \lambda } , \frac { T ^ { p + 1 } } { \sigma ^ { 2 p } } \} ( \log \frac { 4 8 m } { \delta } ) ^ { 6 + 3 l } } \\ & { \qquad + ( 4 + 2 \sqrt { 2 } ) ^ { l } ( C _ { \eta } ^ { \prime } ) ^ { l } ( C _ { 0 , 3 } + 1 ) ^ { \frac { l } { 2 } } \lambda ^ { - \frac { l } { 2 } } \underset { 1 \leq j \leq m } { \operatorname* { m a x } } B _ { | D _ { j } | , \lambda } ^ { l } \cdot \tilde { C } _ { 3 , 2 } \underset { 1 \leq j \leq m } { \operatorname* { m a x } } \{ \mathcal { R } _ { | D _ { j } | , \lambda } \mathcal { B } _ { | D _ { j } | , \lambda } \} \log T ( \log \frac { 4 8 m } { \delta } ) ^ { 6 + 3 l } . } \\ &  \leq \tilde { C } _ { 3 , 4 } \operatorname* { m a x } \{  \end{array}\tag{89}
$$

$$
\begin{array} { r } { \mathrm { w h e r e ~ } \tilde { C } _ { 3 , 4 } : = ( 4 + 2 \sqrt { 2 } ) ^ { l _ { 2 } 6 + 3 l } ( C _ { \eta } ^ { \prime } ) ^ { l } C _ { 0 , 1 } ^ { l } ( C _ { 0 , 3 } + 1 ) ^ { \frac { i } { 2 } } \left[ ( C _ { 0 , 1 } + 2 ^ { p + 1 } ) ( \tilde { C } _ { 3 , 1 } + \tilde { C } _ { 3 , 3 } ) + C _ { 0 , 1 } C _ { 0 , 2 } \tilde { C } _ { 3 , 2 } \right] . } \end{array}
$$

It is observed that the derivation of both high-probability upper bounds for $\| f _ { T + 1 , D } - f _ { T + 1 } \| _ { \rho }$ and $\| f _ { T + 1 , D } - f _ { \rho } \| _ { \rho }$ relies on the estimation of the bounds of $\mathcal { P } _ { | D | , \lambda } , \mathcal { Q } _ { | D | , \lambda }$ and $\mathcal { A } _ { | D | , \lambda }$ . Consequently, these two bounds hold simultaneously. Applying the triangle inequality and scaling 2δ to $\delta ,$ we can combine (86) and (89) to derive that with confidence at least $1 - \delta$ , there holds

$$
\left\| \bar { f } _ { T + 1 , D } ^ { l } - f _ { \rho } \right\| _ { \rho } \leq \tilde { C } _ { 3 } \operatorname* { m a x } \left\{ | D | ^ { - \frac { r } { 2 r + s } } , \frac { | D | ^ { \frac { p + 1 } { 2 r + s } } } { \sigma ^ { 2 p } } \right\} \left( \log \frac { 9 6 } { \delta } \right) ^ { 6 + 3 l } .
$$

where $\tilde { C } _ { 3 } = \tilde { C } _ { 3 , 4 } + \tilde { C } _ { r , 3 }$ . This completes the proof of Theorem 3.

## Appendix

In this appendix, we give the proof of Lemma 1 and Lemma 2.

Proof of Lemma 1. We prove (27) by induction, $f _ { 1 } = 0$ is obviously satisfies the inequality (27). Then for $t = 2$ (22) indicates that

$$
\| f _ { 2 } \| _ { K } = \left\| - \eta \int _ { \mathcal { Z } } G ^ { \prime } ( \xi _ { t , \sigma } ( z ) ) ( - y ) K _ { x } \mathrm { d } \rho \right\| _ { K } \leq \eta \kappa M C _ { G } \leq M \sqrt { \eta C _ { G } } \leq \frac { M } { \kappa } ,
$$

which means that the inequality (27) also holds for $t = 2$ . When $t > 2 ,$ we denote $\begin{array} { r } { H _ { t } = \int _ { \mathcal { Z } } G ^ { \prime } ( \xi _ { t , \sigma } ( z ) ) ( f _ { t } ( x ) - y ) K _ { x } \mathrm { d } \rho } \end{array}$ and have that

$$
\begin{array} { l } { \displaystyle \| f _ { t + 1 } \| _ { \kappa } ^ { 2 } = \| f _ { t } - \eta H _ { t } \| _ { \kappa } ^ { 2 } = \| f _ { t } \| _ { \kappa } ^ { 2 } - 2 \eta \langle f _ { t } , H _ { t } \rangle _ { \kappa } + \eta ^ { 2 } \| H _ { t } \| _ { \kappa } ^ { 2 } } \\ { \displaystyle \leq \| f _ { t } \| _ { \kappa } ^ { 2 } + \int _ { \mathcal { Z } } \left( - 2 \eta G ^ { \prime } ( \xi _ { t , \sigma } ( z ) ) ( f _ { t } ( x ) - y ) f _ { t } ( x ) + \eta ^ { 2 } \kappa ^ { 2 } G ^ { \prime } ( \xi _ { t , \sigma } ( z ) ) ^ { 2 } ( f _ { t } ( x ) - y ) ^ { 2 } \right) \mathrm { d } \rho } \\ { \displaystyle = \| f _ { t } \| _ { \kappa } ^ { 2 } + \int _ { \mathcal { Z } } \eta \left( \eta \kappa ^ { 2 } G ^ { \prime } ( \xi _ { t , \sigma } ( z ) ) ( f _ { t } ( x ) - y ) ^ { 2 } - 2 ( f _ { t } ( x ) - y ) f _ { t } ( x ) \right) G ^ { \prime } ( \xi _ { t , \sigma } ( z ) ) \mathrm { d } \rho } \\ { \displaystyle = \| f _ { t } \| _ { \kappa } ^ { 2 } + \int _ { \mathcal { Z } } \eta \left\{ ( \eta \kappa ^ { 2 } G ^ { \prime } ( \xi _ { t , \sigma } ( z ) ) - 2 ) \left( ( f _ { t } ( x ) - y ) - \frac { y } { \eta \kappa ^ { 2 } G ^ { \prime } ( \xi _ { t , \sigma } ( z ) ) - 2 } \right) ^ { 2 } + \frac { y ^ { 2 } } { 2 - \eta \kappa ^ { 2 } G ^ { \prime } ( \xi _ { t , \sigma } ( z ) ) } \right\} G ^ { \prime } ( \xi _ { t , \sigma } ( z ) ) \mathrm { d } \rho , } \end{array}\tag{90}
$$

where the first inequality is due to the Jensen’s inequality and the fact that $K ( x , x ^ { \prime } ) \le \kappa ^ { 2 }$ for any $x , x ^ { \prime } \in \mathcal { X }$ , and the last equality is due to the completion of squares.

Since $G ^ { \prime } ( \xi _ { t , \sigma } ( z ) ) \leq C _ { G }$ , we have that $\eta \kappa ^ { 2 } G ^ { \prime } ( \xi _ { t , \sigma } ( z ) ) \leq 1$ , and further $\eta \kappa ^ { 2 } G ^ { \prime } ( \xi _ { t , \sigma } ( z ) ) - 2 \leq 0$ and $2 - \eta \kappa ^ { 2 } G ^ { \prime } ( \xi _ { t , \sigma } ( z ) ) \geq$ 1 for any $z \in { \mathcal { Z } }$ . Putting the above bound with $| y | \le M$ and the induction assumption $\| f _ { t } \| _ { K } ^ { 2 } \leq \eta C _ { G } M ^ { 2 } ( t - 1 )$ into (90) yields that

$$
\| f _ { t + 1 } \| _ { K } ^ { 2 } \leq \| f _ { t } \| _ { K } ^ { 2 } + \eta C _ { G } M ^ { 2 } \leq \eta C _ { G } M ^ { 2 } t ,
$$

and thus

$$
\| f _ { t + 1 } \| _ { K } \leq M \sqrt { \eta C _ { G } t } \leq \frac { M } { \kappa } t ^ { \frac { 1 } { 2 } } .
$$

This completes the proof of Lemma 1.

Proof of Lemma 2. With the choice of η, we have

$$
\begin{array} { r } { \Big \| \big ( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K } \big ) ^ { T - i } \Big \| \leq 1 , \quad \forall i = 0 , 1 , \cdots , T . } \end{array}
$$

Based on the proof of Theorem 1 (i) in [8, Page 15], it can be derived with $\theta = 0$ and $\eta _ { 1 }$ replaced by η that

$$
\left\| \sum _ { i = 1 } ^ { T } \eta G _ { + } ^ { \prime } ( 0 ) ( \lambda I + L _ { K , D } ) \left( I - \eta G _ { + } ^ { \prime } ( 0 ) L _ { K , D } \right) ^ { T - i } \right\| \leq \eta G _ { + } ^ { \prime } ( 0 ) \lambda T + 1 .
$$

Analogously, we can replace $\boldsymbol { L } _ { K , D }$ with $L _ { K }$ to obtain the same bound of the operator norm concerning $L _ { K }$ . Therefore, the proof is completed.

## Reference

[1] Zhang Y, Duchi J, Wainwright M, Divide and conquer kernel ridge regression: A distributed algorithm with minimax optimal rates, Journal of Machine Learning Research, 16(1):3299-3340, 2015.

[2] Lin S B, Guo X, Zhou D X, Distributed learning with least square regularization, Journal of Machine Learning Research, 18(92):1-31, 2017.

[3] Guo Z C, Lin S B, Zhou D X, Learning theory of distributed spectral algorithms, Inverse Problems, 33(7):074009, 2017.

[4] Mücke N, Blanchard G, Parallelizing spectrally regularized kernel algorithms, Journal of Machine Learning Research, 19(1):1069-1097, 2018.

[5] Lin S B, Wang D, Zhou D X, Distributed Kernel Ridge Regression with Communications, Journal of Machine Learning Research, 21(93):1-38, 2020.

[6] Lin S B, Zhou D X, Distributed kernel-based gradient descent algorithms, Constructive Approximation, 47(2): 249-276, 2018.

[7] Hu T, Wu Q, Zhou D X, Distributed kernel gradient descent algorithm for minimum error entropy principle, Applied and Computational Harmonic Analysis, 49(1):229-256, 2020.

[8] Guo Z C, Hu T, Shi L, Gradient descent for robust kernel-based regression, Inverse Problems, 34(6):065009, 2018.

[9] Hong Q, Guo Z C, Robust kernel-based gradient descent with random features, Advances in Computational Mathematics, 51(6):1-63, 2025.

[10] Guo Z C, Christmann A, Shi L, Optimality of Robust Online Learning, Foundations of Computational Mathe matics, 24(5):1455-1483, 2024.

[11] Huber P J, Robust estimation of a location parameter, Breakthroughs in Statistics: Methodology and Distribution, Springer, 492-518, 1992.

[12] Holland P, Welsch R E, Robust regression using iteratively reweighted least-squares, Communications in Statistics - Theory and Methods, 6(9):813-827, 1977.

[13] Yao Y, Rosasco L, Caponnetto A, On early stopping in gradient descent learning, Constructive Approximation, 26:289-315, 2007.

[14] Bellet A, Liang Y, Garakani A B, et al., A distributed Frank-Wolfe algorithm for communication-eficient sparse learning, Proceedings of the 2015 SIAM International Conference on Data Mining, 478-486, 2015.

[15] Zeng J, Yin W, On nonconvex decentralized gradient descent, IEEE Transactions on signal processing, 66(11): 2834-2848, 2018.

[16] Yin R, Wang W, Meng D, Distributed nyström kernel learning with communications, International Conference on Machine Learning, 12019-12028, 2021.

[17] Huang C, Huo X, A distributed one-step estimator, Mathematical Programming, 174:41-76, 2019.

[18] Liu J, Shi L, Distributed learning with discretely observed functional data, Inverse Problems, 41(4):045006, 2025.

[19] Wang B, Hu T, Lei L, Distributed Robust Algorithms with Dependent Sampling, Mathematics, 13(23):3813, 2025.

[20] Minsker S, On some extensions of Bernstein’s inequality for self-adjoint operators, Statistics & Probability Letters, 127:111-119, 2017.

[21] Caponnetto A, Yao Y, Cross-validation based adaptation for regularization operators in learning theory, Analysis and Applications, 8(02):161-183, 2010.

[22] Caponnetto A, De Vito E, Optimal rates for the regularized least-squares algorithm, Foundations of Computational Mathematics, 7:331-368, 2007.

[23] Steinwart I, Hush D R, Scovel C, Optimal Rates for Regularized Least Squares Regression, Proceedings of the 22nd Annual Conference on Learning Theory, 79-93, 2009.

[24] Ruder S, An overview of gradient descent optimization algorithms, arXiv preprint arXiv:1609.04747, 2016.

[25] Bauer F, Pereverzev S, Rosasco L, On regularization algorithms in learning theory, Journal of complexity, 23(1): 52-72, 2007.

[26] Dicker L H, Foster D P, Hsu D, Kernel ridge vs. principal component regression: Minimax bounds and the qualification of regularization operators, Electronic Journal of Statistics, 11(1):1022-1047, 2017.

[27] Raskutti G, Wainwright M J, Yu B, Early stopping and non-parametric regression: an optimal data-dependent stopping rule, Journal of Machine Learning Research, 15(1):335-366, 2014.

[28] Hu T, Kernel-based maximum correntropy criterion with gradient descent method, Communications on Pure & Applied Analysis, 19(8):4159-4177, 2020.

[29] Zhou D X, Deep distributed convolutional neural networks: Universality, Analysis and Applications, 16(6): 895-919, 2018.

[30] Yu Z, Fan J, Shi Z, et al., Distributed Gradient Descent for Functional Learning, IEEE Transactions on Information Theory, 70(9):6547-6571, 2024.

[31] Smale S, Zhou D X, Learning theory estimates via integral operators and their approximations, Constructive Approximation, 26(2):153-172, 2007.

[32] Guo Z C, Shi L, Optimal rates for coeficient-based regularized regression, Applied and Computational Harmonic Analysis, 47(3):662-701, 2019.

[33] Engl H W, Hanke M, Neubauer A, Regularization of inverse problems, Springer Science & Business Media, 1996.

[34] Blanchard G, Krämer N, Optimal learning rates for Kernel Conjugate Gradient regression, Advances in Neural Information Processing Systems, 226-234, 2010.

[35] Zhou D X, The covering number in learning theory, Journal of Complexity, 18(3):739-767, 2002.