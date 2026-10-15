# CONVERGENCE RATES OF GAUSSIAN PROCESS THOMPSON SAMPLING FOR EFFICIENT GLOBAL OPTIMIZATION

Haowei Wang<sup>∗</sup> National University of Singapore Singapore

Yilin Zheng National University of Singapore Singapore

October 15, 2026

## ABSTRACT

Efficient global optimization aims to identify a global optimum of an expensive black-box function within a limited evaluation budget N. We investigate the convergence rates of Gaussian process Thompson sampling (GP-TS) under noiseless observations, assuming a fixed objective function in a reproducing kernel Hilbert space (RKHS). For Matérn kernels with smoothness parameter ν in dimension d, we establish a worst-case expected simple-regret upper bound of $O ( N ^ { - \nu / d } \sqrt { \log N } )$ for both discrete GP-TS on sufficiently fine grids and its continuous counterpart. For the continuous implementation, we develop a nested-grid argument to control posterior sample paths over the entire domain. We further establish matching lower bounds of $\Omega ( N ^ { - 5 / 2 } \sqrt { \log N } )$ for both implementations under a fixed one-dimensional Matérn-5/2 kernel, demonstrating that the logarithmic penalty is intrinsic to standard GP-TS in this setting. Motivated by this limitation, we propose Gaussian Process Capped Thompson Sampling (GP-CTS), which caps posterior sample paths using a valid RKHS norm upper bound while preserving optimism. This modification successfully eliminates the logarithmic penalty and achieves the minimax-optimal rate $\Theta ( N ^ { - \nu / d } )$ for both discrete and continuous implementations.

Keywords: Efficient global optimization; noiseless observation; Thompson sampling; reproducing kernel Hilbert space; simple regret; upper and lower bounds.

## 1 Introduction

We consider efficient global optimization of an expensive deterministic black-box function, where the objective is to identify a near-optimal solution using a limited number of function evaluations[1]. In such problems, evaluating the objective requires running expensive computer simulations [6, 10]. Many of these simulations are deterministic, making the observation model noiseless [10]. Moreover, it typically aims to identify a single high-quality solution as the final recommendation. Poor intermediate evaluations can be acceptable if they help improve the final recommendation, which is different from the setting of cumulative regret[11]. When observations are noiseless, the best queried point can be directly identified from the observed values. Therefore, best-query simple regret is a natural performance measure for studying the convergence rates of efficient global optimization algorithms [1].

Bayesian Optimization (BO) provides a widely used framework for efficient global optimization by using a surrogate Gaussian Process (GP) model to guide sequential evaluations [11, 2, 4]. Expected Improvement (EI) [6, 1, 4] is one of the most widely studied acquisition functions. In particular, Bull [1] investigated simple regret upper bounds for EI under Reproducing Kernel Hilbert Space (RKHS) assumptions with noiseless observations. These results demonstrate how kernel smoothness affects the convergence rates.

Thompson Sampling (TS) [2] is another widely used acquisition strategy, with the associated algorithm called GP-TS. At each iteration, GP-TS draws a function from the current GP posterior and selects its maximizer as the next query point. Its simplicity and randomized sampling mechanism have made it useful in applications such as parallel BO [7] and scalable optimization methods such as TuRBO [3].

Despite its practical use, the available simple regret bounds for GP-TS remain comparatively limited. Much of the existing theoretical analysis of GP-TS has focused on cumulative regret, which measures the total loss incurred over all evaluations. Simple regret upper bounds can then be obtained by dividing cumulative regret bounds by the evaluation budget. Although this approach provides valid bounds, the resulting rates can be loose because cumulative regret also accounts for poor intermediate evaluations that do not affect the best queried point. This distinction is particularly important for smooth kernels, as discussed in the related work below. Therefore, a direct analysis of simple regret is needed to better characterize the convergence rates of GP-TS. This motivates the central question:What convergence rates can GP-TS achieve in efficient global optimization when performance is measured by best-query simple regret?

We address this question under the noiseless frequentist RKHS setting, where the objective function is fixed instead of sampled from the GP prior. Under a polynomial Kolmogorov-width condition with decay exponent $\alpha > 0$ and regularity assumptions, we establish a general simple regret upper bound of $\widetilde O ( N ^ { - \alpha } )$ , where $N$ denotes the evaluation budget. Our analysis combines conditional optimism of posterior samples with a deterministic bound on posterior uncertainty along arbitrary subsequences of queries. For the continuous implementation, the main challenge is to control posterior sample paths over the entire input domain. We address this challenge using a nested-grid argument without restricting the algorithm to a finite decision set. For Matérn kernels with smoothness parameter ν in dimension $d ,$ we have $\alpha = \nu / d$ . Both exact maximization of Gaussian vectors on sufficiently fine finite grids and exact continuous posterior sample path maximization achieve the worst-case expected simple regret upper bound $O ( N ^ { - \nu / d } \sqrt { \log N } )$ for all $\nu > 0 .$ . This bound recovers the full smoothness exponent $\nu / d$ and is near-optimal in terms of the minimax convergence rates[1].

We next investigate whether the remaining $\sqrt { \log N }$ factor is an artifact of the upper bound analysis or an inherent limitation of GP-TS itself. Establishing a lower bound is challenging because the query points are determined by maxima of correlated Gaussian sample paths, whose behavior changes as observations are collected. To address this challenge, we analyze Gaussian derivative processes to control sampled values near the true maximizer and compare them with Gaussian fluctuations over a suboptimal region. For a fixed one-dimensional Matérn- $- 5 / 2$ kernel, we establish matching worst-case expected lower bounds of $\Omega ( N ^ { - 5 / 2 } \sqrt { \log N } )$ for both discrete and continuous implementations of GP-TS. For the discrete implementation, we further show that the discretization error is of smaller order than the lower bound. Therefore, in this setting, the additional $\sqrt { \log N }$ factor reflects the intrinsic behavior of GP-TS itself instead of the discretization or the upper-bound analysis. In particular, the result identifies a logarithmic loss relative to the optimal minimax convergence rates[1].

This logarithmic factor during the analysis motivates controlling excessive positive sample deviations while retaining the optimism used in our upper bound analysis. We thus propose Gaussian Process Capped Thompson Sampling (GP-CTS), which caps each sampled function by a deterministic upper envelope obtained from a known valid RKHS norm upper bound. The cap preserves optimism and gives simple regret upper bounds of $O ( N ^ { - \alpha } )$ without a logarithmic factor. For Matérn kernels, both discrete and continuous GP-CTS therefore can attain the minimax rate of $\Theta ( N ^ { - \nu / d } )$

Our contributions are summarized as follows:

1). Simple-regret upper bounds for GP-TS. We establish direct simple-regret upper bounds for both discrete and continuous implementations of GP-TS. For Matérn kernels with any smoothness parameter $\nu > 0 .$ , we obtain a worst-case expected simple-regret upper bound of $O ( N ^ { - \nu / d } \sqrt { \log N } )$ . Compared with previously reported bounds for GP-TS-based methods under noiseless observations, our results improve the logarithmic dependence when $1 < \nu \leq d$ and the polynomial convergence rate when $\nu > d .$ . Moreover, our analysis accommodates exact maximization of posterior sample paths over a continuous domain, without restricting the algorithm to a finite decision set.

2). Matching lower bounds and an intrinsic limitation of GP-TS. To the best of our knowledge, we establish the first matching lower bounds on the worst-case expected simple regret of GP-TS in the noiseless frequentist RKHS setting. Specifically, for a fixed one-dimensional $\mathrm { { M a t e r n – 5 / 2 } }$ kernel, we prove lower bounds of $\Omega ( N ^ { - 5 / 2 } \sqrt { \log N } )$ for both discrete and continuous implementations. Together with our upper bounds, these results establish the tight rate $\Theta ( N ^ { - 5 / 2 } \sqrt { \log N } )$ , demonstrating that the logarithmic factor is intrinsic to standard GP-TS in this setting instead of an artifact of the upper-bound analysis.

3). Eliminating the logarithmic penalty through capped posterior sampling. Motivated by this limitation, we propose Gaussian Process Capped Thompson Sampling (GP-CTS), which caps posterior sample paths using a known valid RKHS norm upper bound. The cap preserves optimism while controlling positive sample deviations, thereby eliminating the logarithmic penalty. We establish the minimax-optimal convergence rates $\Theta ( N ^ { - \nu / d } )$ for Matérn kernels in both discrete and continuous implementations.

## 1.1 Related Work

Convergence rates in efficient global optimization. Bull [1] considered noiseless observations and established simple regret upper bounds for EI under fixed GP models and showed that a modified EI algorithm can attain the optimal polynomial convergence rate for Matérn kernels. More recently, Vázquez and Petit [14] established the optimal simple regret upper bound $O ( N ^ { - \nu / d } )$ for fixed-prior EI under Matérn RKHS assumptions. Other GP based algorithms, including maximum variance reduction and phased elimination, have also been analyzed in the noiseless setting [5]. These results provide important benchmarks for studying the convergence of GP-TS. However, convergence guarantees for other acquisition strategies do not determine the rates achieved by GP-TS itself.

Table 1: Comparison of worst-case expected simple regret bounds under noiseless observations. The previous bounds are for GP-BTS with fixed delay Q and finite decision sets. The last column reports the bounds for standard GP-TS.
<table><tr><td>Simple regret Matérn kernel</td><td></td><td>Takeno and Iwazaki [12]</td><td>Ours</td></tr><tr><td rowspan="3">Upper bound</td><td> $1 < \nu < d$ </td><td> $O ( N ^ { - \nu / d } ( \log N ) ^ { \nu / d + 1 / 2 } )$ </td><td> $O ( N ^ { - \nu / d } \sqrt { \log N } )$ </td></tr><tr><td> $\nu = d > 1$ </td><td> $O ( N ^ { - 1 } ( \log N ) ^ { 5 / 2 } )$ </td><td> $O ( N ^ { - 1 } { \sqrt { \log N } } )$ </td></tr><tr><td> $\nu > d$ </td><td> $O ( N ^ { - 1 } { \sqrt { \log N } } )$ </td><td> $O ( N ^ { - \nu / d } \sqrt { \log N } )$ </td></tr><tr><td>Lower bound</td><td> $\nu = 5 / 2 , d = 1$ </td><td></td><td> $\Omega ( N ^ { - 5 / 2 } \sqrt { \log N } )$ </td></tr></table>

Frequentist regret analysis of GP-TS. Chowdhury and Gopalan [2] established cumulative regret upper bounds for GP-TS in the frequentist RKHS setting with noisy observations. Their analysis uses posterior sampling over finite decision sets and bounds cumulative regret through maximum information gain. More recently, Takeno and Iwazaki [12] studied GP Batched Thompson Sampling (GP-BTS) under both noisy and noiseless observations. In the noiseless setting, their simple regret upper bounds are obtained by dividing cumulative regret bounds by the evaluation budget. Among the available frequentist analyses of GP-TS based methods, their noiseless results provide a particularly relevant comparison with our work. Table 1 compares our simple regret bounds with those reported by Takeno and Iwazaki [12] for Matérn kernels under noiseless observations. It is clear that our upper bounds improve the reported logarithmic factors when $1 < \nu \leq d$ and the polynomial convergence rate when $\nu > d .$ . Furthermore, we establish a matching lower bound for the fixed one-dimensional Matérn- $- 5 / 2$ kernel, which is not provided in the previous work. The logarithmic factors reported by Takeno and Iwazaki [12] are subject to the qualification discussed in their Remark 2.2. Roy et al. [9] also developed a frequentist analysis of GP-TS using fractional GP posteriors over continuous domains, focusing on cumulative regret under noisy observations. Our work instead studies best-query simple regret under noiseless observations.

## 2 Background

We consider black-box optimization problems with noiseless observations over a compact domain $\mathcal { X } \subset \mathbb { R } ^ { d }$ . For simplicity, throughout the paper we take $\mathcal { X } = [ 0 , 1 ] ^ { d }$ . At round n, a sequential algorithm selects a query point $x _ { n } \in \mathcal { X }$ and observes

$$
y _ { n } = f ( x _ { n } ) , \quad n = 1 , 2 , \ldots , N ,\tag{1}
$$

where $f : \mathcal { X } \to \mathbb { R }$ is an unknown but fixed black-box objective function.

Without loss of generality, we consider maximization and define

$$
f ^ { * } = \operatorname* { m a x } _ { x \in \mathcal { X } } f ( x ) , \quad x ^ { * } \in \arg \operatorname* { m a x } _ { x \in \mathcal { X } } f ( x ) ,\tag{2}
$$

where one maximizer $x ^ { * }$ is fixed if the maximizer is not unique. Our primary performance metric is the best-query simple regret

$$
r _ { N } ^ { \mathrm { b e s t } } = f ^ { * } - \operatorname* { m a x } _ { 1 \leq n \leq N } f ( x _ { n } ) .\tag{3}
$$

Since all observations are exact, the best queried point can be identified directly from the observed function values. All probabilities and expectations in our theoretical analysis are taken with respect to the algorithmic randomness of the algorithm under consideration, for a fixed objective $\dot { f } .$

## 2.1 GP posterior and its RKHS Geometry

We use a zero-mean working GP prior $\mathcal { G P } ( 0 , k ( \cdot , \cdot ) )$ , where $k : \mathcal { X } \times \mathcal { X } $ R is the kernel function. All kernel parameters remain fixed, and no positive nugget is added to the Gram matrix. The working GP model does not assume that the fixed objective f is drawn from this prior. For $X _ { n } = ( x _ { 1 } , \ldots , x _ { n } )$ , define

$$
K _ { n } = [ k ( x _ { i } , x _ { j } ) ] _ { i , j = 1 } ^ { n } , \quad \mathbf { k } _ { n } ( x ) = ( k ( x _ { 1 } , x ) , \ldots , k ( x _ { n } , x ) ) ^ { \top } , \quad \mathbf { y } _ { n } = ( f ( x _ { 1 } ) , \ldots , f ( x _ { n } ) ) ^ { \top } .\tag{4}
$$

The GP posterior formulas follow from Gaussian conditioning [8]:

$$
\begin{array} { r l r } & { } & { \mu _ { n } ( x ) = { \bf k } _ { n } ( x ) ^ { \top } K _ { n } ^ { \dagger } { \bf y } _ { n } , } \\ & { } & { c _ { n } ( x , x ^ { \prime } ) = k ( x , x ^ { \prime } ) - { \bf k } _ { n } ( x ) ^ { \top } K _ { n } ^ { \dagger } { \bf k } _ { n } ( x ^ { \prime } ) , } \\ & { } & { s _ { n } ( x ) = \sqrt { c _ { n } ( x , x ) } . \qquad } \end{array}\tag{5}
$$

Here $K _ { n } ^ { \dagger }$ is the Moore–Penrose inverse, equal to $K _ { n } ^ { - 1 }$ when $K _ { n }$ is nonsingular. For $n = 0 .$ , set $\mu _ { 0 } = 0 , c _ { 0 } = k$ , and $s _ { 0 } ( x ) = \sqrt { k ( x , x ) }$ . The corresponding subspace and projection are $W _ { 0 } = \{ 0 \}$ and $P _ { 0 } = 0$

For $f \in \mathcal { H } _ { k }$ , these posterior quantities also admit useful geometric interpretations in the RKHS $\mathcal { H } _ { k }$ associated with kernel k. Let $k _ { x } : = k ( x , \cdot ) , W _ { n } : =$ span $\{ k _ { x _ { 1 } } , \ldots , k _ { x _ { n } } \}$ , and let $P _ { n } : { \bar { \mathcal { H } } } _ { k } \to W _ { n }$ denote the orthogonal projection operator, and let I be the identity operator on $\mathcal { H } _ { k }$ . Then

$$
\begin{array} { r } { \mu _ { n } = P _ { n } f , \quad \quad \quad } \\ { c _ { n } ( x , x ^ { \prime } ) = \langle ( I - P _ { n } ) k _ { x } , ( I - P _ { n } ) k _ { x ^ { \prime } } \rangle _ { \mathcal { H } _ { k } } , } \\ { s _ { n } ( x ) = \| ( I - P _ { n } ) k _ { x } \| _ { \mathcal { H } _ { k } } \quad \quad } \end{array}\tag{6}
$$

In particular, once the RKHS norm of f is bounded by $B > 0$ , the following deterministic bound holds

$$
| f ( x ) - \mu _ { n } ( x ) | \leq B s _ { n } ( x ) .\tag{7}
$$

The residual vector at round $n , \mathrm { i . e . , } ( I - P _ { n - 1 } ) k _ { x _ { \tau } }$ is also called the sequential innovation [14] of the newly queried kernel section $k _ { x _ { n } }$ , and its RKHS norm is exactly $s _ { n - 1 } ( x _ { n } )$

This connection between posterior uncertainty and approximation geometry will be central to our analysis. We refer readers to Appendix A for the corresponding derivations, including singular Gram matrices.

## 2.2 Kernel Approximation Geometry

Our analysis requires additional definitions related to kernel sections. Define $\mathcal { K } _ { k } = \{ k _ { x } : x \in \mathcal { X } \} \subset \mathcal { H } _ { k }$ . For an integer $m \geq 1$ , the Kolmogorov m-width of $\kappa _ { k }$ is defined as

$$
w _ { m } ( k , \mathcal { X } ) : = \operatorname* { i n f } _ { V \subset \mathcal { H } _ { k } \operatorname* { l i n e a r } _ { x \in \mathcal { X } } } \operatorname* { s u p } _ { \mathrm { d i s t } _ { \mathcal { H } _ { k } } ( k _ { x } , V ) . }\tag{8}
$$

Intuitively, the quantity $w _ { m } ( k , \mathcal { X } )$ measures the smallest worst-case error achievable when the entire family of kernel sections is approximated by an m-dimensional subspace $V \subset \mathcal { H } _ { k }$ .

We further define the canonical kernel pseudo-metric

$$
d _ { k } ( x , x ^ { \prime } ) : = \| k _ { x } - k _ { x ^ { \prime } } \| _ { \mathcal H _ { k } } = \sqrt { k ( x , x ) + k ( x ^ { \prime } , x ^ { \prime } ) - 2 k ( x , x ^ { \prime } ) } ,\tag{9}
$$

which characterizes the local variation of kernel sections with respect to the input.

## 2.3 Regularity Assumptions

We state the following regularity assumptions. Each theoretical result specifies which of them is required.

Assumption 1. The kernel k is continuous, symmetric, and positive semidefinite on $\mathcal { X } \ \times \ \mathcal { X }$ , and satisfies $\begin{array} { r } { \operatorname* { s u p } _ { x \in \mathcal { X } } k ( x , x ) \le 1 } \end{array}$ . The objective function satisfies $f \in \mathcal { H } _ { k }$ and $\| \bar { \boldsymbol { f } } \| _ { \mathcal { H } _ { k } } \le B$ for some fixed $B > 0$

The kernel and B are fixed independently of the evaluation budget. This assumption implies that f is continuous, the maximum $f ^ { \ast } \mathrm { e x i s t s }$ , and $| f ( x ) | \overset { } { \leq } B$ , so $\mathbf { \dot { 0 } } \leq r _ { N } ^ { \mathrm { b e s t } } \leq 2 B$

Assumption 2. There exist constants $A > 0$ and $\alpha > 0$ such that

$$
w _ { m } ( k , \mathcal { X } ) \leq A m ^ { - \alpha } , \quad m \geq 1 .\tag{10}
$$

Assumption 2 characterizes the approximation complexity of the kernel-section family $\kappa _ { k }$ . It does not impose any space-filling, separation, or quasi-uniformity condition on the actual query points. The approximating subspace need not be spanned by previously queried kernel sections and need not be constructed by the algorithm.

Assumption 3. There exist constants $L _ { k } > 0$ and $\zeta \in ( 0 , 1 ]$ such that

$$
d _ { k } \left( x , x ^ { \prime } \right) \leq L _ { k } \left. x - x ^ { \prime } \right. _ { 2 } ^ { \zeta } , \quad x , x ^ { \prime } \in \mathcal { X } .\tag{11}
$$

Assumption 3 is used to control finite-grid approximation errors and continuous sample-path increments. Consequently,

$$
| f ( x ) - f ( x ^ { \prime } ) | \leq B L _ { k } \| x - x ^ { \prime } \| _ { 2 } ^ { \zeta } .\tag{12}
$$

Proposition 1. Let k be a stationary isotropic Matérn kernel with fixed lengthscale $\ell > 0$ and smoothness parameter $\nu > 0 ,$ , restricted to $\mathcal { X } = [ 0 , 1 ] ^ { d }$ and normalized such that $k ( x , \overset { \cdot } { x } ) \leq 1 , \ \overset { \cdot } { \forall } x \in \mathcal { X }$ . Then Assumption 2 holds with $\alpha = \nu / d ,$ and Assumption 3 holdsfor any $0 < \zeta < \mathrm { m i n } \{ \nu , 1 \}$ and one may take $\zeta = 1$ whenever $\nu > 1$

Proposition 1 follows from the spectral properties of the underlying stationary Matérn kernel on $\mathbb { R } ^ { d } .$ , whose spectral density satisfies $\rho ( \omega ) \asymp ( 1 + \| \omega \| _ { 2 } ^ { 2 } ) ^ { - ( \nu + \bar { d } / 2 ) }$ . A direct verification is provided in Appendix A.1. Classical analyses such as Bull [1] instead impose spectral and local regularity conditions on stationary kernels and derive the corresponding approximation properties. Proposition 1 connects these two viewpoints for Matérn kernels.

## 2.4 Gaussian-Process Thompson Sampling

Let $\mathcal { F } _ { n - 1 }$ denote the history available before round n, including all previous query points, exact observations, and algorithmic randomness. We distinguish two implementations of GP-TS according to the decision domain D, which is fixed within a run. All posterior draws use fresh randomization and have the stated conditional law given the complete history. The sampling scale $v > 0$ is fixed throughout the theoretical analysis.

Discrete Finite Decision Domain I $\mathbf { \dot { \hat { \rho } } } \mathcal { D } = \mathcal { G } = \left\{ z _ { 1 } , \dots , z _ { M } \right\} \subset \mathcal { X }$ is a nonempty deterministic finite set, GP-TS draws a finite-dimensional Gaussian random vector

$$
\left( G _ { n } ( z _ { 1 } ) , \ldots , G _ { n } ( z _ { M } ) \right) ^ { \top } | \mathcal { F } _ { n - 1 } \sim \mathcal { N } \left( \mu _ { n - 1 } ( \mathcal { G } ) , v ^ { 2 } c _ { n - 1 } ( \mathcal { G } , \mathcal { G } ) \right) .\tag{13}
$$

GP-TS then selects its maximizing coordinate as the next query point. No continuous sample path is required. On $\mathcal { G } _ { : }$ write $Z _ { n } = ( G _ { n } - \mu _ { n - 1 } ) / v$ for the corresponding centered residual vector.

Continuous Decision Domain $\operatorname { I f } { \mathcal { D } } = { \mathcal { X } } ,$ the main analysis of continuous domain uses an exact continuous posterior sample path. Specifically, at round n, GP-TS draws a continuous sample path of

$$
Z _ { n } \mid { \mathcal { F } } _ { n - 1 } \sim { \mathcal { G P } } ( 0 , c _ { n - 1 } ) ,\tag{14}
$$

and constructs

$$
G _ { n } ( x ) = \mu _ { n - 1 } ( x ) + v Z _ { n } ( x ) , \qquad x \in \mathcal { X } .\tag{15}
$$

GP-TS then selects a global maximizer of the sample path. The choice $v = 1$ corresponds to the unscaled working posterior.

For either decision domain, we allow a deterministic global optimization tolerance $\eta _ { n } \geq 0$ and select a measurable point $x _ { n } \in \mathcal { D }$ satisfying

$$
G _ { n } ( x _ { n } ) \geq \operatorname* { s u p } _ { x \in \mathcal { D } } G _ { n } ( x ) - \eta _ { n } .\tag{16}
$$

Exact maximization corresponds to $\eta _ { n } = 0$ . In the discrete decision domain, the sampled Gaussian vector can always be maximized exactly, so we set $\eta _ { n } = 0$ . The notation is retained only for consistency with the analysis of continuous domain.

We use fixed measurable tie-breaking rules, and their existence for continuous sample paths on the compact domain is verified in Appendix A. Algorithm 1 combines the two implementations in one procedure.

These two implementations correspond to different decision domains but follow the same posterior-sampling principle. In the subsequent analysis, we treat them separately. For the discrete decision domain, the main additional issue is the discretization error relative to the original continuous domain. For the continuous decision domain, the main additional issue is controlling the sample path over an uncountable domain. We will show that, after accounting for these differences, both implementations achieve the same dominant simple-regret rate. For either implementation, N counts all function evaluations.

Algorithm 1 Gaussian-Process Thompson Sampling (GP-TS)   
Require: Domain $x ;$ decision domain $\mathcal { D } = \mathcal { G }$ or $x ;$ kernel $k ;$ fixed $v > 0 ;$ budget $N \geq 1 ;$ tolerances $( \eta _ { n } ) _ { n = 1 } ^ { N }$   
Ensure: A best queried point ${ \widehat { x } } _ { N } .$   
1: Initialize ${ { X } _ { 0 } } \mathrm { ~ \bar { = ~ } ( ) ~ }$ and $\mathbf { y } _ { 0 } = ( ) ;$ set $\mu _ { 0 } = 0$ and $c _ { 0 } = k .$   
2: for $n = 1 , \ldots , \tilde { N }$ do   
3: Compute $( \mu _ { n - 1 } , c _ { n - 1 } , s _ { n - 1 } )$ from $\left( X _ { n - 1 } , \mathbf { y } _ { n - 1 } \right)$ using (5).   
4: if $\mathcal { D } = \mathcal { G } = \{ z _ { 1 } , . . . , z _ { M } \}$ then   
5: Draw $g _ { n } \stackrel { \cdot } { \sim } \mathcal { N } ( \mu _ { n - 1 } ( \mathcal { G } ) , v ^ { 2 } c _ { n - 1 } ( \mathcal { G } , \mathcal { G } ) )$ conditionally on ${ \mathcal { F } } _ { n - 1 } .$   
6: Set $G _ { n } ( z _ { j } ) = ( g _ { n } ) _ { j }$ for $j = 1 , \dots , M$   
7: else   
8: Draw a fresh continuous $Z _ { n } \mid \mathcal { F } _ { n - 1 } \sim \mathcal { G P } ( 0 , c _ { n - 1 } )$   
9: Set $G _ { n } = \mu _ { n - 1 } + v Z _ { n }$ on X.   
10: end if   
11: Select a measurable $x _ { n } \in \mathcal { D }$ satisfying $\begin{array} { r } { G _ { n } ( x _ { n } ) \geq \operatorname* { s u p } _ { x \in \mathcal { D } } G _ { n } ( x ) - \eta _ { n } } \end{array}$   
12: Observe $y _ { n } = f ( x _ { n } ) ;$ append $x _ { n }$ to $X _ { n - 1 }$ and $y _ { n } \ \mathrm { t o } \ \mathbf { y } _ { n - 1 }$ to obtain $X _ { n } , \mathbf { y } _ { n }$   
13: end for   
14: Choose $j _ { N } \in \arg \operatorname* { m a x } _ { 1 \leq j \leq N } y _ { j }$ using a fixed tie-breaking rule.   
15: return $\widehat { x } _ { N } = x _ { j _ { N } }$

## 3 Simple Regret Upper Bound of GP-TS

In this section, we derive simple-regret rates for both the discrete and continuous implementations of GP-TS.

## 3.1 Two Key Lemmas

Before presenting the main results, we introduce two key lemmas that characterize, respectively, the decay of posterior uncertainty along arbitrary query subsequences and the frequency of optimistic posterior samples. We first restate a consequence of the width-to-separation geometry of Vázquez and Petit [14].

Lemma 1. Under Assumptions 1 and 2, there is a constant $C _ { \mathrm { s e p } } > 0$ depending only on A and $\alpha ,$ such that for every $N \geq 1$ , every sequence $x _ { 1 } , \dotsc , x _ { N } \in \mathcal { X }$ , and every nonempty index set ${ \hat { I } } \subseteq \{ 1 , \ldots , { \dot { N } } \}$ with $| I | = m$

$$
\operatorname* { m i n } _ { n \in I } s _ { n - 1 } ( x _ { n } ) \leq C _ { \mathrm { s e p } } m ^ { - \alpha } .\tag{17}
$$

One admissible constant is

$$
C _ { \mathrm { s e p } } = e ^ { ( 1 + 2 \alpha ) / 2 } \sqrt { \operatorname* { m a x } \{ 1 , 2 A ^ { 2 } 3 ^ { 2 \alpha } \} } .\tag{18}
$$

Consequently, for every $u > 0 ,$

$$
\# \{ n \leq N : s _ { n - 1 } ( x _ { n } ) > u \} \leq \left\lceil \left( \frac { C _ { \mathrm { s e p } } } { u } \right) ^ { 1 / \alpha } \right\rceil .\tag{19}
$$

Lemma 1 is deterministic and holds simultaneously for every subset of a realized algorithmic trajectory. Thus I could be defined after all sampled functions have been observed. No union bound over index sets is needed. It neither controls $\mathrm { s u p } _ { x } s _ { N } ( x )$ nor implies space filling, and it does not assert $s _ { n - 1 } ( x _ { n } ) \leq C _ { \mathrm { s e p } } n ^ { - \alpha }$ at every round. Its proof is given in Appendix B.1.

Lemma 2. Let $( \mathcal { F } _ { n } )$ be a filtration and let $O _ { n } \in \{ 0 , 1 \}$ be an ${ \mathcal { F } } _ { n }$ -measurable indicator. If

$$
\mathbb { P } ( O _ { n } = 1 \mid { \mathcal F } _ { n - 1 } ) \ge p \quad a . s . , \quad 0 < p \le 1 ,\tag{20}
$$

then, for every $N \geq 1$

$$
\mathbb { P } \left( \sum _ { n = 1 } ^ { N } O _ { n } < \frac { p N } { 2 } \right) \le e ^ { - p N / 8 } .\tag{21}
$$

For our analysis, fix a deterministic comparator $x ^ { \dag } \in \mathcal { X }$ , taken as $x _ { g } ^ { * }$ in the discrete case and $x ^ { * }$ in the continuous case, and define the indicator $O _ { n } = { \bf 1 } \{ G _ { n } ( x ^ { \dagger } ) \geq f ( x ^ { \dagger } ) \}$ . Assumption 1, together with the fresh conditional Gaussian sampling rule in Section 2.4, implies (20) with $p = \Phi \dot { ( - B / v ) }$ , where Φ is the standard normal distribution function.

When $s _ { n - 1 } ( x ^ { \dagger } ) > 0$ , the required standardized Gaussian threshold is at most $B / v$ . When the standard deviation is zero, exact interpolation gives $G _ { n } ( x ^ { \dagger } ) = f ( x ^ { \dagger } )$ almost surely, hence $O _ { n } = 1$ . Choosing $v = B$ makes $p$ the numerical constant $\Phi ( - 1 )$ . With $v = 1$ , p remains strictly positive for every fixed RKHS ball, although it can be small when B is large. We retain this dependence explicitly instead of treating p as universal. Appendix B.2 proves Lemma 2 and verifies this optimism probability.

We are now ready to establish the main simple-regret guarantees for GP-TS. We first consider the discrete decision domain and then extend the analysis to the continuous setting.

## 3.2 Simple Regret Upper Bound for the Discrete Decision Domain

Fix a nonempty deterministic finite set $\mathcal G \subset \mathcal X$ of cardinality M. Define

$$
f _ { \mathcal { G } } ^ { * } = \operatorname* { m a x } _ { x \in \mathcal { G } } f ( x ) , \quad x _ { \mathcal { G } } ^ { * } \in \arg \operatorname* { m a x } _ { x \in \mathcal { G } } f ( x ) , \quad b _ { \mathcal { G } } = f ^ { * } - f _ { \mathcal { G } } ^ { * } .\tag{22}
$$

The bias $b _ { \mathcal { G } }$ measures discretization of the continuous objective domain into a finite decision set. It is distinct from posterior-sampling approximation and global optimization errors, which can also arise on a finite decision set.

Theorem 1. Under Assumptions 1 and 2, run Algorithm 1 on G for N rounds withfixed $v > 0 ,$ ,fresh exact conditional draws, and the global selection rule (16). For $0 ^ { - } < \delta < 1$ , define $p = \Phi ( - B / v )$ and thefinite-set Gaussian-envelope coefficient $u _ { N , M , \delta } : = \sqrt { 2 \log ( 4 M N / \delta ) } . \ I f N \geq 8 \log ( 2 / \delta ) / p ,$ then,for everyfixed f in the radius-B RKHS ball, with probability at leas $1 - \delta ,$

$$
r _ { N } ^ { \mathrm { b e s t } } \leq b _ { \mathcal { G } } + C _ { \mathrm { s e p } } ( B + v u _ { N , M , \delta } ) \left( \frac { p N } { 2 } \right) ^ { - \alpha } + \operatorname* { m a x } _ { 1 \leq n \leq N } \eta _ { n } .\tag{23}
$$

Proof Idea The finite decision set allows a uniform Gaussian deviation bound over all rounds and candidate points. Lemma 1 and Lemma 2 then imply the existence of an optimistic round with posterior uncertainty of order $N ^ { - \alpha }$ Combining these two ingredients yields the stated bound. The complete proof is deferred to Appendix B.3.

The optimization error term $\eta _ { n }$ is not essential in the discrete decision-domain setting. Since the sampled Gaussian vector is finite-dimensional, its maximizing coordinate can in principle be identified exactly, in which case one may simply set $\eta _ { n } = 0$ . We retain $\eta _ { n }$ only to keep the notation consistent with the continuous-domain analysis.

Next we provide the Matérn kernel specialization under Theorem 1.

To control the discretization bias term $b _ { \mathcal { G } }$ , we define the following regular grid

$$
\nonumber Q _ { J } = \{ 0 , 1 / J , \ldots , 1 \} ^ { d } , \quad M _ { J } = ( J + 1 ) ^ { d } , \quad h _ { J } = \frac { \sqrt { d } } { J } ,\tag{24}
$$

which means that each coordinate axis is uniformly partitioned into J subintervals.

Under Assumption 3 and (12), we have $b _ { \mathcal { Q } _ { J } } \leq B L _ { k } h _ { J } ^ { \zeta }$ . For a given budget $N ,$ , choose

$$
\begin{array} { r } { J _ { N } = \left\lceil \sqrt { d } N ^ { \alpha / \zeta } \right\rceil , \quad \mathcal { G } _ { N } = \mathcal { Q } _ { J _ { N } } . } \end{array}\tag{25}
$$

Then

$$
b _ { { \mathcal { G } } _ { N } } \leq B L _ { k } N ^ { - \alpha } , \quad | { \mathcal { G } } _ { N } | \leq C _ { d } N ^ { \alpha d / \zeta } ,\tag{26}
$$

where one may take $C _ { d } = ( \sqrt { d } + 2 ) ^ { d }$

Corollary 1. For the Matérn kernel in Proposition 1, run Algorithm 1 on the grid in (25) with $\alpha = \nu / d ,$ any fixed $v > 0$ , and exact maximization $( \eta _ { n } = 0 )$ . Set $p = \Phi ( - B / v )$ . There exists a constant $C > 0 ;$ , independent off and $\delta ,$ such that,for everyfixed f in the radius-B RKHS ball and every $0 < \delta < 1$ , whenever $N \geq 2$ and $N \geq 8 \log ( 2 / \delta ) / p ,$

$$
\begin{array} { r } { \mathbb { P } _ { f } \left( r _ { N } ^ { \mathrm { b e s t } } \le C N ^ { - \nu / d } \sqrt { \log ( N / \delta ) } \right) \ge 1 - \delta . } \end{array}\tag{27}
$$

Moreover, there exist constants $C ^ { \prime } > 0$ and an integer $N _ { 0 } \geq 2 ,$ independent of the particular $f ,$ such that for all $N \geq N _ { 0 } ,$

$$
\operatorname* { s u p } _ { \| f \| _ { \mathcal { H } _ { k } } \leq B } \mathbb { E } _ { f } [ r _ { N } ^ { \mathrm { b e s t } } ] \leq C ^ { \prime } N ^ { - \nu / d } \sqrt { \log N } .\tag{28}
$$

The constants $C , C ^ { \prime } , N _ { 0 }$ in Corollary 1 may depend on $B , v , k .$ , d and the chosen ζ. The proof is given in Appendix B.4. A deterministic global-optimization error can be added to the right-hand side as in Theorem 1. For a fixed Matérn RKHS ball, the minimax simple-regret rate under noiseless evaluations is $N ^ { - \nu / d } \ [ 1 ]$ . Therefore, Corollary 1 matches the optimal polynomial exponent and is near-optimal up to a $\sqrt { \log N }$ factor. The best-query comparison is justified in Appendix B.4.

## 3.3 Simple Regret Upper Bound for the Continuous Decision Domain

We now consider the continuous implementation of GP-TS with $\mathcal { D } = \mathcal { X }$ . Unlike in the discrete case, the maximizer $x _ { n }$ is selected from an uncountable domain using the current posterior sample path. Therefore, the finite-set Gaussian union bound used in Theorem 1 is no longer directly applicable.

To control the sampled function over the continuous domain, we first establish a uniform envelope for the posterior residual process. We still use the regular grid defined in (24). Define

$$
u _ { N , J , \delta } ^ { \mathrm { c } } = \sqrt { 2 \log \frac { 8 N M _ { J } } { \delta } } , \quad \rho _ { \zeta } = 2 ^ { - \zeta } ,\tag{29}
$$

and

$$
C _ { 1 } ( \zeta ) = \frac { 1 } { 1 - \rho _ { \zeta } } , \quad C _ { 2 } ( d , \zeta ) = \sqrt { 2 ( d + 1 ) \log 2 } \frac { \rho _ { \zeta } } { ( 1 - \rho _ { \zeta } ) ^ { 2 } } .\tag{30}
$$

Lemma 3. Under Assumptions 1 and 3, let $Z _ { n }$ befresh continuous posterior residuals with the conditional law (14). For any integers $J , N \ge 1$ and $0 < \delta < 1$ , with probability at least $1 - \delta / 2 ,$ , simultaneously for all $1 \leq n \leq N$ and $x \in { \mathcal { X } } ,$

$$
| Z _ { n } ( x ) | \leq u _ { N , J , \delta } ^ { \mathrm { c } } s _ { n - 1 } ( x ) + L _ { k } h _ { J } ^ { \zeta } \left[ C _ { 1 } ( \zeta ) u _ { N , J , \delta } ^ { \mathrm { c } } + C _ { 2 } ( d , \zeta ) \right] .\tag{31}
$$

In Lemma 3, the first term is analogous to the Gaussian envelope in the discrete case, while the second term controls the increments between the proof grid and arbitrary points in the continuous domain. The result follows from the contraction

$$
{ \mathbb { E } } \left[ \left( Z _ { n } ( x ) - Z _ { n } \left( x ^ { \prime } \right) \right) ^ { 2 } \mid \mathcal { F } _ { n - 1 } \right] \leq d _ { k } \left( x , x ^ { \prime } \right) ^ { 2 }\tag{32}
$$

together with a dyadic chaining argument. The proof is provided in Appendix B.5. The grid is used only in the proof and does not restrict the continuous algorithm. Using this envelope, the same optimistic-subsequence argument as in the discrete case gives the following result.

Theorem 2. Under Assumptions $I , 2 ,$ and 3, run Algorithm 1 on Xfor N rounds withfixed $v > 0 ,$ ,fresh exact continuous posterior draws, and the global selection rule (16). For $0 < \delta < 1$ , set $p = \Phi ( - B / v ) . \ : I f N \geq 8 \log ( 2 / \delta ) / p ,$ , then,for each integer $J \geq 1$ and eachfixed f in the radius-B RKHS ball, with probability at least $1 - \delta ,$

$$
r _ { N } ^ { \mathrm { b e s t } } \leq C _ { \mathrm { s e p } } ( B + v u _ { N , J , \delta } ^ { \mathrm { c } } ) \left( \frac { p N } { 2 } \right) ^ { - \alpha } + v L _ { k } h _ { J } ^ { \zeta } \left[ C _ { 1 } ( \zeta ) u _ { N , J , \delta } ^ { \mathrm { c } } + C _ { 2 } ( d , \zeta ) \right] + \operatorname* { m a x } _ { 1 \leq n \leq N } \eta _ { n } .\tag{33}
$$

Proof sketch The proof follows the same structure as Theorem 1. Lemma 2 provides a linear number of optimistic rounds, while Lemma 1 guarantees that one of these rounds has posterior uncertainty of order $N ^ { - \alpha }$ . The continuous envelope in Lemma 3 replaces the finite Gaussian union bound used in the discrete case. The complete proof is given in Appendix B.6.

Similar to the discrete case, we can still derive the Matérn kernel specialization under Theorem 2.

Corollary 2. For the Matérn kernel in Proposition 1, run Algorithm 1 on X with any fixed $v > 0 ,$ fresh exact continuous posterior draws, and exact maximization $( \eta _ { n } = 0 )$ . The high-probability and worst-case expected bounds in Corollary 1 hold under the same respective budget conditions, with possibly different constants.

The proof of Corollary 2 is given in Appendix B.7. Notably, Theorem 2 considers an exact continuous posterior sample path and a global η<sub>n</sub>-approximate maximizer. Practical implementations based on finite-dimensional sample-path approximations introduce additional sampling errors that are not covered by these results.

Remark 1. Both continuous and sufficiently fine discrete implementations achieve the same near-minimax convergence ratesfor Matérn kernels. However, Theorems 1 and 2 and their corresponding corollaries do not themselves prove that <sub>Matérn</sub> <sub>GP-TS</sub> <sub>must</sub> <sub>pay</sub> √<sub>log</sub> <sub>N.</sub> <sub>Therefore,</sub> <sub>we</sub> <sub>further</sub> <sub>discuss</sub> <sub>whether</sub> <sub>the</sub> <sub>logarithmic</sub> <sub>factor</sub> <sub>is</sub> <sub>genuine</sub> <sub>or</sub> <sub>not</sub> <sub>in</sub> <sub>the</sub> next section

## 4 Logarithmic Factor from Gaussian Overshoot of GP-TS

The upper bounds in Section 3 leave open whether the logarithmic factor is necessary. We address this question using a fixed one-dimensional Matérn-5/2 kernel, establishing lower bounds for both discrete and continuous implementations of GP-TS. The choice $d = 1$ provides a transparent interval geometry for analyzing the query sequence, while the widely used $\nu = 5 / 2$ admits twice continuously differentiable sample paths and convenient RKHS representations for our theoretical analysis. Notably, this path regularity also holds for other Matérn kernels with $\nu > 2$ . These choices allow us to distinguish the loss caused by Gaussian posterior overshoot from discretization error while keeping the lower-bound construction easy to understand.

## 4.1 Matérn Construction and Key Estimates

Fix $B , v > 0$ and

$$
k ( x , y ) = \left( 1 + | x - y | + { \frac { | x - y | ^ { 2 } } { 3 } } \right) e ^ { - | x - y | } , \qquad x , y \in [ 0 , 1 ] .\tag{34}
$$

This is the normalized Matérn- $- 5 / 2$ kernel with fixed lengthscale ${ \sqrt { 5 } } .$ Let $\left( J _ { N } ^ { \mathrm { l b } } \right)$ be a deterministic sequence of positive integers, and define the regular decision grids

$$
\gamma _ { \cal N } = ( J _ { \cal N } ^ { \mathrm { l b } } ) ^ { - 1 } , \qquad { \mathcal { G } } _ { \cal N } ^ { \mathrm { l b } } = { \mathcal { Q } } _ { { \cal J } _ { \cal N } ^ { \mathrm { l b } } } = \{ j \gamma _ { \cal N } : 0 \leq j \leq J _ { \cal N } ^ { \mathrm { l b } } \} .\tag{35}
$$

We consider the resolution regime

$$
\gamma _ { N } = \mathcal { O } ( N ^ { - 5 / 4 } ) , \qquad \log ( J _ { N } ^ { \mathrm { l b } } + 1 ) = \mathcal { O } ( \log N ) .\tag{36}
$$

For example, $J _ { N } ^ { \mathrm { l b } } = \lceil N ^ { 5 / 4 } \rceil$ satisfies these conditions, as does the finer choice $\lceil N ^ { 5 / 2 } \rceil$ obtained from (25) with $d = 1$ and $\zeta = 1$ . The grid is fixed within each run and is the same for every objective in the RKHS ball.

The first condition in (36) keeps the discretization bias at the minimax scale. Indeed, this kernel satisfies $\| f ^ { \prime \prime } \| _ { \infty } \leq B$ whenever $\| f \| _ { \mathcal { H } _ { k } } \le B$ . Since the grid contains both boundary points, Taylor’s formula at an interior maximizer gives

$$
\operatorname* { s u p } _ { \| f \| _ { \mathcal { H } _ { k } } \le B } b _ { \mathcal { G } _ { N } ^ { \mathrm { l b } } } ( f ) \le \frac { B } { 8 } \gamma _ { N } ^ { 2 } = \mathcal { O } ( N ^ { - 5 / 2 } ) .\tag{37}
$$

The second condition keeps the finite-set Gaussian coefficient in Theorem 1 of order $\sqrt { \log N }$ . Thus the upper and lower bounds can be compared in the same discretization regime. No particular grid cardinality or alignment with the maximizer is required. The grid estimates are verified in Appendix C.3.

Choose a fixed smooth cutoff $\chi ,$ equal to 1 on $[ 0 , 1 / 8 ]$ and 0 on $[ 1 / 2 , 1 ]$ , with $0 \leq \chi \leq 1$ and a smooth compactly supported extension to R. Choose a fixed $\kappa > 0$ such that

$$
\kappa \big ( 2 \| \chi x \| _ { \mathcal H _ { k } } + \| \chi x ^ { 2 } \| _ { \mathcal H _ { k } } \big ) \leq B / 2 ,
$$

and consider

$$
f _ { \varepsilon } ( x ) = \kappa \chi ( x ) ( 2 \varepsilon x - x ^ { 2 } ) , \qquad 0 < \varepsilon < 1 / 1 6 .\tag{38}
$$

This family has a quadratic peak with maximizer ε and value $\kappa \varepsilon ^ { 2 }$ , together with a zero-valued plateau $\mathcal { P } = [ 3 / 4 , 1 ]$ . Its members lie in the span of two fixed smooth RKHS functions, so their norms remain uniformly bounded as the positive peak narrows. We choose $\varepsilon = \varepsilon _ { N }$ below, independently of the algorithmic randomization.

The same objective family is used for both implementations. To state their common trajectory estimates, let $\mathcal { D } _ { N }$ denote either $\check { \mathcal { G } } _ { N } ^ { \mathrm { l b } } \mathrm { o r } \check { [ 0 , 1 ] }$ , fixed within a run. GP-TS uses fresh exact samples and exact maximization on the chosen domain. In the discrete case, represent each Gaussian vector as the restriction of an auxiliary $C ^ { 2 }$ posterior path. In the continuous case, use the $C ^ { 2 }$ version of the sampled path itself.

Let $\mathcal { E } _ { \partial } = \{ x _ { 1 } = 0 \}$ . Its probability is at least a fixed $q _ { \partial } > 0 \colon$ a prior path with strictly negative derivative has its maximum at 0 on either domain. Conditional on this event, $G _ { n } ( 0 ) = f _ { N } ( 0 ) = 0 \mathrm { f o r } n \ge 2 .$ . Appendix C verifies this probability bound.

The following lemmas describe the two competing sampled-value scales along each implementation’s own trajectory. Constants may depend on the fixed problem parameters and, in the discrete case, the constants in (36), but not on $\dot { N }$ the admissible $\varepsilon _ { N }$ , or the realized history. No grid conditions are required in the continuous case.

Lemma 4 (Persistent plateau overshoot). For the kernel (34), let GP-TS operate on either $\mathcal { D } _ { N } = [ 0 , 1 ]$ or a fixed grid sequence satisfying (35)–(36). Let $f _ { N } = f _ { \varepsilon _ { N } }$ satisfy $\kappa \varepsilon _ { N } ^ { 2 } \leq N ^ { - 5 / 2 } \sqrt { \log N }$ . There exist $b _ { * } \in ( 0 , 1 ]$ and an integer m<sub>∗</sub> such that,for all sufficiently large N,

$$
\mathbb { P } _ { f _ { N } } \left( \operatorname* { m a x } _ { z \in \mathcal { D } _ { N } \cap \mathcal { P } } G _ { n } ( z ) > b _ { * } n ^ { - 5 / 2 } \sqrt { \log n } f o r e \nu e r y m _ { * } < n \leq N \Big | \mathcal { E } _ { \partial } \right) \geq \frac { 7 } { 8 } .\tag{39}
$$

The proof derives plateau coverage from the GP-TS rule and the two key lemmas of Section 3. It then normalizes posterior coordinates by the lengths of the actual observation intervals. The resulting covariance bounds yield enough Gaussian variation for persistent overshoot. Details are provided in Appendix C.1.

Lemma ${ \boldsymbol { 5 } } \left( { \mathrm { I } } \right.$ Local observations and curvature). Under the same conditions, there existfixed constants $A _ { \mathrm { l o c } } , C _ { g } , L , c > 0 ,$ with $L \geq \operatorname* { m a x } \{ 1 , \kappa \}$ , such that thefollowing holdsfor all sufficiently large N. Define

$$
h _ { N } = \frac { A _ { \mathrm { l o c } } } { \log N } , \qquad m _ { N } = \lceil N ^ { 1 / 4 } \rceil ,\tag{40}
$$

and let $\tau _ { N } ^ { \mathrm { l o c } }$ be the first round by which two distinct query locations in $( 0 , h _ { N } )$ have been observed. Then

$$
\begin{array} { r } { \mathbb { P } _ { f _ { N } } ( \tau _ { N } ^ { \mathrm { l o c } } > m _ { N } \mid \mathcal { E } _ { \partial } ) \leq e ^ { - c N ^ { 3 / 1 6 } } , } \end{array}\tag{41}
$$

$$
\mathbb { P } _ { f _ { N } } \left( \operatorname* { m a x } _ { 2 \leq n \leq N } \| G _ { n } ^ { \prime \prime } \| _ { \infty } > C _ { g } \sqrt { \log N } \bigg | \mathcal { E } _ { \partial } \right) \leq N ^ { - 2 } ,\tag{42}
$$

$$
\mathbb { P } _ { f _ { N } } \left( \tau _ { N } ^ { \mathrm { l o c } } \le m _ { N } , \operatorname* { m a x } _ { m _ { N } < n \le N } \| G _ { n } ^ { \prime \prime } \| _ { \infty , [ 0 , h _ { N } ] } > L \bigg | \mathcal { E } _ { \partial } \right) \le N ^ { - 2 } .\tag{43}
$$

The derivatives refer to the posterior paths, used only as auxiliary variables in the discrete implementation.

The two local observations, together with the boundary observation, reduce the local second-derivative variance through a three-zero Sobolev estimate. Gaussian small-ball and shift bounds establish the required visits, and the proof is provided in Appendix C.2. The quantities $h _ { N }$ and $m _ { N }$ organize the proof instead of the algorithm. These local observations need not be near-optimal: $\varepsilon _ { N } = o ( h _ { N } ) $ ), so the local neighborhood is much wider than the eventual near-optimal region.

## 4.2 Discrete Matérn Logarithmic Lower Bound

Choose $b _ { * }$ and $L$ from the preceding lemmas and set

$$
b _ { N } = b _ { * } N ^ { - 5 / 2 } \sqrt { \log N } , \qquad \varepsilon _ { N } ^ { 2 } = \frac { b _ { N } } { 1 6 L } , \qquad f _ { N } = f _ { \varepsilon _ { N } } .\tag{44}
$$

For sufficiently large N, this choice satisfies both lemmas. Outside the near-optimal region

$$
\begin{array} { r } { A _ { N } = ( \varepsilon _ { N } / 2 , 3 \varepsilon _ { N } / 2 ) , } \end{array}\tag{45}
$$

the objective gap is at least $\kappa \varepsilon _ { N } ^ { 2 } / 4$

Theorem 3 (Discrete Matérn logarithmic lower bound). Fix $B , v > 0 ,$ the kernel $( 3 4 ) ,$ , and any deterministic regulargrid sequence satisfying (35)–(36). Run Algorithm 1 on these grids withfresh exact conditional Gaussian vectors and exact maximization. There exist constants $c _ { \mathrm { l b } } , q _ { \mathrm { l b } } > 0$ and an integer $N _ { 0 }$ such that,for every $N \geq N _ { 0 } ,$ , the deterministic objective in (44) belongs to the radius-B RKHS ball and satisfies

$$
\mathbb { P } _ { f _ { N } } \left( r _ { N } ^ { \mathrm { b e s t } } \ge c _ { \mathrm { l b } } N ^ { - 5 / 2 } \sqrt { \log N } \right) \ge q _ { \mathrm { l b } } .\tag{46}
$$

Consequently, for this family of discrete GP-TS policies,

$$
\operatorname* { s u p } _ { \| \boldsymbol { f } \| _ { \mathcal { H } _ { k } } \leq B } \mathbb { E } _ { \boldsymbol { f } } [ r _ { N } ^ { \mathrm { b e s t } } ] = \Theta \left( N ^ { - 5 / 2 } \sqrt { \log N } \right) .\tag{47}
$$

The constants may depend on thefixed problem and resolution parameters, but not on N. The objective f isfixed before the algorithm is run and may vary with the budget within the same RKHS ball.

Proof sketch On $\mathcal { E } _ { \partial }$ , a grid maximizer $x _ { n } \in A _ { N }$ must beat its two neighboring grid points. If the auxiliary path has second derivative bounded by $L _ { n }$ on the relevant interval, Taylor’s formula gives

$$
\begin{array} { r } { | G _ { n } ^ { \prime } ( x _ { n } ) | \leq \frac { 1 } { 2 } L _ { n } \gamma _ { N } , \qquad G _ { n } ( x _ { n } ) \leq \frac { 1 } { 2 } L _ { n } ( x _ { n } ^ { 2 } + \gamma _ { N } x _ { n } ) . } \end{array}\tag{48}
$$

Lemma $^ { 5 }$ controls this local sampled value, while Lemma 4 supplies a larger value on the plateau. A separate argument treats the first finitely many rounds. These estimates give a positive probability that all queries avoid $\mathcal { A } _ { N }$ , and the details are provided in Appendix C.3.

By (37), the uniform discretization bias is $o ( N ^ { - 5 / 2 } \sqrt { \log N } )$ . Hence the logarithmic lower bound persists after accounting for the approximation of the continuous objective domain. The condition (36) is sufficient, instead of a claim about the coarsest possible grid.

## 4.3 Continuous Matérn Logarithmic Lower Bound

We next consider exact posterior-path maximization over [0, 1]. The objective family and the two trajectory estimates above remain applicable, but all posterior quantities must now be evaluated along the continuous algorithm’s own query sequence. The result is not obtained by identifying a grid maximizer with a continuous maximizer or by taking a limit of the discrete theorem.

Choose $b _ { * }$ and L from the continuous versions of Lemmas 4 and 5, and define $b _ { N } , \varepsilon _ { N }$ , and $f _ { N }$ by (44). The constants may differ from those used for the discrete implementation, while the kernel and the form of the objective family are unchanged.

Theorem 4 (Continuous Matérn logarithmic lower bound). Fix $B , v \ > \ 0$ and the kernel (34). Run Algorithm 1 on $\mathcal { D } = [ 0 , 1 ]$ with fresh exact continuous posterior samples and exact global maximization. There exist constants $c _ { \mathrm { l b } } ^ { \mathrm { c } } , q _ { \mathrm { l b } } ^ { \mathrm { c } } > 0$ and an integer $N _ { \mathrm { 0 } } ^ { \mathrm { c } }$ , depending only on the fixed problem parameters, such thatfor every $N \geq N _ { 0 } ^ { \mathrm { c } }$ , the deterministic objective just defined belongs to the radius-B RKHS ball and satisfies

$$
\mathbb { P } _ { f _ { N } } \left( r _ { N } ^ { \mathrm { b e s t } } \geq c _ { \mathrm { l b } } ^ { \mathrm { c } } N ^ { - 5 / 2 } \sqrt { \log N } \right) \geq q _ { \mathrm { l b } } ^ { \mathrm { c } } .\tag{49}
$$

Consequently,

$$
\operatorname* { s u p } _ { \| \boldsymbol { f } \| _ { \mathcal { H } _ { k } } \leq B } \mathbb { E } _ { f } [ r _ { N } ^ { \mathrm { b e s t } } ] = \Theta \left( N ^ { - 5 / 2 } \sqrt { \log N } \right) .\tag{50}
$$

The objective may vary with $N$ within the same fixed RKHS ball, but is chosen before the algorithm is run. No decision grid is involved in this result.

Proof sketch If a continuous sampled maximizer lies in $\mathcal { A } _ { N }$ , it is an interior maximizer of a $C ^ { 2 }$ path and satisfies $G _ { n } ^ { \prime } ( x _ { n } ) = 0 $ . On $\mathcal { E } _ { \partial }$ , Taylor’s formula therefore gives

$$
\begin{array} { r } { G _ { n } ( x _ { n } ) \leq \frac { 1 } { 2 } L _ { n } x _ { n } ^ { 2 } , } \end{array}\tag{51}
$$

whenever $| G _ { n } ^ { \prime \prime } | \leq L _ { n }$ on $[ 0 , x _ { n } ]$ . The plateau estimate and the local curvature estimate exclude such a maximizer in the intermediate and later rounds. A separate derivative-density argument treats the initial fixed-length prefix. Appendix C.4 gives the complete argument along the continuous trajectory and combines it with Corollary 2 for the matching upper bound.

Remark 2. Theorems 3 and 4 identify a worst-case limitation ofexact GP-TSfor thefixed one-dimensional Matérn-5/2 kernel. In both implementations, the mechanism is competition between Gaussian extremes on the suboptimal plateau and sampled values near the true maximizer. The discrete theorem additionally controls decision-grid bias, whereas the continuous theorem compares maxima over the entire domain.

To summarize, the upper and lower bounds distinguish a genuine limitation of standard GP-TS from a loss caused only by the analysis. This motivates capping positive sample deviations while preserving comparator optimism. In the next section, we show that this modification attains the Matérn minimax rate when a valid RKHS norm upper bound is supplied.

## 5 GP-CTS and Minimax Convergence Rates

The analyses in Sections 3 and 4 provide an important insight into the additional $\sqrt { \log N }$ factor in the convergence rates of GP-TS. The upper bound introduces this factor when controlling positive Gaussian deviations at the selected query, while the lower bound shows that Gaussian overshoot over a suboptimal region can indeed cause such a loss. These results suggest limiting excessive positive sample deviations while preserving optimism at a fixed comparator. Motivated by this observation, we propose Gaussian Process Capped Thompson Sampling (GP-CTS). We first introduce the algorithm and then establish its simple regret bounds.

## 5.1 Algorithmic Details of GP-CTS

The deterministic RKHS inequality (7) provides an upper bound on the objective at every point. Using this bound to cap a posterior sample prevents excessively large sampled values without changing whether the sample is optimistic at a fixed comparator. To construct the cap, GP-CTS requires a known valid RKHS norm upper bound B. For uniform guarantees over the radius-B RKHS ball in Assumption 1, we assume that $\overline { { B } } \geq B > 0$ is fixed independently of the evaluation budget and the particular objective. The algorithm does not need the exact value of $\| f \| _ { \mathcal { H } _ { k } }$

Algorithm 2 Gaussian Process Capped Thompson Sampling (GP-CTS)   
Require: Domain $x ,$ decision domain $\mathcal { D } = \mathcal { G }$ or X, fixed kernel k, fixed $v > 0 ,$ , known valid ${ \overline { { B } } } \geq B ,$ , budget $N \geq 1$   
deterministic tolerances $( \eta _ { n } ) _ { n = 1 } ^ { N }$ with $\eta _ { n } = 0$ for $\mathcal { D } = \mathcal { G } .$   
Ensure: A best queried point ${ \widehat { x } } _ { N } .$   
1: Initialize ${ { X } _ { 0 } } \mathrm { ~ \bar { = ~ } ( ) ~ }$ and $\mathbf { y } _ { 0 } = \left( \begin{array} { r l } \end{array} \right)$ . Set $\mu _ { 0 } = 0$ and $c _ { 0 } = k .$   
2: for $n = 1 , \ldots , \bar { N }$ do   
3: Compute $( \mu _ { n - 1 } , c _ { n - 1 } , s _ { n - 1 } )$ from $\left( X _ { n - 1 } , \mathbf { y } _ { n - 1 } \right)$ using (5).   
4: if $\mathcal { D } = \mathcal { G } = \{ z _ { 1 } , . . . , z _ { M } \}$ then   
5: Draw $g _ { n } \stackrel { \cdot } { \sim } \mathcal { N } ( \mu _ { n - 1 } ( \mathcal { G } ) , v ^ { 2 } c _ { n - 1 } ( \mathcal { G } , \mathcal { G } ) )$ conditionally on $\mathcal { F } _ { n - 1 }$   
6: Set $G _ { n } ( z _ { j } ) = ( g _ { n } ) _ { j } \mathrm { ~ f o r ~ } j = 1 , \ldots , M .$   
7: else   
8: Draw a fresh continuous $Z _ { n } \mid { \mathcal { F } } _ { n - 1 } \sim { \mathcal { G P } } ( 0 , c _ { n - 1 } ) .$   
9: Set $G _ { n } = \mu _ { n - 1 } + v Z _ { n }$ on X .   
10: end if   
11: Set $U _ { n - 1 } = \mu _ { n - 1 } + \overline { { B } } s _ { n - 1 }$ and $H _ { n } = \operatorname* { m i n } \{ G _ { n } , U _ { n - 1 } \}$ on $\mathcal { D } .$   
12: Select a measurable $x _ { n } \in \mathcal { D }$ satisfying $H _ { n } ( x _ { n } ) \geq$ sup<sub>x∈D</sub> $H _ { n } ( x ) - \eta _ { n }$   
13: Observe $y _ { n } = f ( x _ { n } )$ and append $x _ { n } , y _ { n }$ to obtain $X _ { n } , \mathbf { y } _ { n } .$   
14: end for   
15: Choose $j _ { N } \in \arg \operatorname* { m a x } _ { 1 \leq j \leq N } y _ { j }$ using a fixed tie-breaking rule.   
16: return $\widehat { x } _ { N } = x _ { j _ { N } } .$

For $n \geq 0 ,$ , define the upper envelope

$$
U _ { n } ( x ) = \mu _ { n } ( x ) + \overline { { B } } s _ { n } ( x ) , \qquad x \in \mathcal { X } .\tag{52}
$$

By (7), $f ( x ) \leq U _ { n } ( x )$ holds for every realized history and every $x \in { \mathcal { X } } .$ . All posterior quantities in this section are computed from the queries and exact observations generated by GP-CTS itself. The posterior update formulas are the same as those of GP-TS, but the two algorithms need not generate the same query sequence.

At round $n ,$ draw $G _ { n }$ using the same conditional sampling rule as in Section 2.4, with the fixed scale $v > 0$ . Define the capped decision function

$$
\begin{array} { r l } & { H _ { n } ( x ) = \operatorname* { m i n } \{ G _ { n } ( x ) , U _ { n - 1 } ( x ) \} } \\ & { \qquad = \mu _ { n - 1 } ( x ) + \operatorname* { m i n } \{ v Z _ { n } ( x ) , \overline { { B } } s _ { n - 1 } ( x ) \} , \qquad x \in \mathcal { D } . } \end{array}\tag{53}
$$

The cap changes a sampled value only when it exceeds the deterministic upper envelope. In particular, negative sample deviations are left unchanged. GP-CTS selects a measurable point $x _ { n } \in \mathcal { D }$ satisfying

$$
H _ { n } ( x _ { n } ) \geq \operatorname* { s u p } _ { x \in \mathcal { D } } H _ { n } ( x ) - \eta _ { n } ,\tag{54}
$$

where $( \eta _ { n } )$ is a deterministic sequence of nonnegative global optimization tolerances. It then observes $y _ { n } = f ( x _ { n } )$ and updates the GP posterior using (5). The tolerance in (54) concerns maximization of $H _ { n } .$ , not the uncapped sample $G _ { n }$ For the discrete implementation, draw a joint Gaussian vector on ${ \mathcal { G } } ,$ cap each coordinate by its corresponding value of $U _ { n - 1 }$ , and select a maximizing coordinate. As in the discrete implementation of $\mathrm { G P - T { \bar { S } } }$ , we set $\eta _ { n } = 0$ . For the continuous implementation, maximize $H _ { n }$ over X. If $G _ { n }$ has continuous paths, then $H _ { n }$ is continuous because $\mu _ { n - 1 }$ and $s _ { n - 1 }$ are continuous. The measurable selection construction in Appendix A therefore also applies to $H _ { n }$

In general, $H _ { n }$ is not a Gaussian process. GP-CTS caps the function used for selecting the query, instead of replacing the GP posterior with a different model. The next round still uses a fresh Gaussian sample from the posterior conditioned on the actual observations. Algorithm 2 summarizes both implementations.

## 5.2 Theoretical Analysis

We first establish the two properties of the cap used in the regret analysis. Since $U _ { n - 1 }$ lies above the objective, capping cannot remove optimism at a fixed comparator. At the same time, $H _ { n } \leq U _ { n - 1 }$ bounds every positive sample deviation in terms of the posterior standard deviation.

Lemma 6. Under Assumption $^ { l , }$ let ${ \overline { { B } } } \geq B$ and define $U _ { n - 1 }$ and $H _ { n }$ by (52) and (53). For every fixed comparator $x ^ { \dagger } \in { \mathcal { D } }$

$$
\{ H _ { n } ( x ^ { \dagger } ) \geq f ( x ^ { \dagger } ) \} = \{ G _ { n } ( x ^ { \dagger } ) \geq f ( x ^ { \dagger } ) \} .\tag{55}
$$

Moreover, for every $x \in \mathcal { D }$ , every realized history, and every sample realization,

$$
H _ { n } ( x ) - f ( x ) \leq U _ { n - 1 } ( x ) - f ( x ) \leq ( B + { \overline { { B } } } ) s _ { n - 1 } ( x ) .\tag{56}
$$

The event identity preserves the conditional optimism probability $p = \Phi ( - B / v )$ under exact Gaussian sampling. The second property is a one-sided deterministic bound. It does not bound $| H _ { n } ( x ) - { \dot { f } } ( x ) |$ , since negative sample deviations remain uncapped. Only the positive deviation is needed in the regret comparison. The proof is given in Appendix D.1. To state the result for both decision domains, define

$$
f _ { \mathcal { D } } ^ { * } = \operatorname* { m a x } _ { x \in \mathcal { D } } f ( x ) , \qquad b _ { \mathcal { D } } = f ^ { * } - f _ { \mathcal { D } } ^ { * } , \qquad \bar { b } _ { \mathcal { D } } = \operatorname* { s u p } _ { \| f \| _ { \mathcal { H } _ { k } } \leq B } b _ { \mathcal { D } } ( f ) .\tag{57}
$$

Here $b _ { \mathcal { D } } ( f )$ makes the dependence on the objective explicit. We have $b _ { \mathcal { D } } = b _ { \mathcal { G } }$ for $\mathcal { D } = \mathcal { G }$ and $b _ { \mathcal { D } } = 0$ for $\mathcal { D } = \mathcal { X }$ . Set

$$
p = \Phi ( - B / v ) , \qquad C _ { \mathrm { c a p } } = ( B + \overline { { B } } ) C _ { \mathrm { s e p } } ( 2 / p ) ^ { \alpha } .\tag{58}
$$

Theorem 5. Under Assumptions 1 and 2, run Algorithm 2for N rounds withfixed $v > 0 ,$ , a knownfixed ${ \overline { { B } } } \geq B ,$ fresh exact conditional draws, and the measurable global selection rule (54). On the continuous domain, assume that the posterior samples admit continuous paths and use those versions. For everyfixed f in the radius-B RKHS ball and every integer $\bar { N } \geq 1$

$$
\mathbb { P } _ { f } \left( r _ { N } ^ { \mathrm { b e s t } } > b _ { \mathcal { D } } + C _ { \mathrm { c a p } } N ^ { - \alpha } + \operatorname* { m a x } _ { 1 \leq n \leq N } \eta _ { n } \right) \leq e ^ { - p N / 8 } .\tag{59}
$$

Consequently,

$$
\operatorname* { s u p } _ { \| f \| _ { \mathcal { H } _ { k } } \leq B } \mathbb { E } _ { f } [ r _ { N } ^ { \mathrm { b e s t } } ] \leq \bar { b } _ { \mathcal { D } } + C _ { \mathrm { c a p } } N ^ { - \alpha } + \operatorname* { m a x } _ { 1 \leq n \leq N } \eta _ { n } + 2 B e ^ { - p N / 8 } .\tag{60}
$$

Proof Sketch Lemma 6 preserves optimism and bounds the regret on each optimistic round by $( B + \overline { { { B } } } ) s _ { n - 1 } ( x _ { n } ) + \eta _ { n } .$ relative to the best point in D. Lemma 2 supplies a linear number of optimistic rounds with high probability. Lemma 1 then guarantees that at least one of those queries has posterior standard deviation of order $N ^ { - \widetilde { \alpha } }$ . This proves the bound without a uniform Gaussian deviation event. The complete proof is provided in Appendix D.2.

The cap replaces the factor $B + v u$ from the GP-TS analysis, where u is the relevant Gaussian-envelope coefficient, with the fixed factor $B + { \overline { { B } } }$ . Consequently, the continuous GP-CTS analysis does not require Lemma 3 or its nestedgrid argument. Assumption 3 is not used in this regret comparison, although it is sufficient to establish continuous sample-path existence and to control finite-grid bias. With exact maximization on X, (59) gives $r _ { N } ^ { \mathrm { b e s t } } \leq C _ { \mathrm { c a p } } N ^ { - \alpha }$ with probability at least $1 - \delta$ whenever $0 < \bar { \delta } < 1$ and $N \geq 8 \log ( 1 / \delta ) / p$

We next specialize the bound to Matérn kernels. The same finite grids as in Section 3 are sufficient because their uniform discretization bias is already of order $N ^ { - \nu / d }$

Corollary 3. For thefixed Matérn kernel in Proposition 1, run Algorithm 2 withfixed $v > 0 ,$ , a knownfixed ${ \overline { { B } } } \geq B$ fresh exact conditional samples, and exact maximization. On $\mathcal { D } = \mathcal { G } _ { N }$ from (25) with $\alpha = \nu / d ,$ or on $\mathcal { D } = \mathcal { X }$

$$
\operatorname* { s u p } _ { \| f \| _ { \mathcal { H } _ { k } } \leq B } \mathbb { E } _ { f } [ r _ { N } ^ { \mathrm { b e s t } } ] = O ( N ^ { - \nu / d } ) .\tag{61}
$$

More generally, the same upper bound holds for any deterministic sequence of nonempty finite decision sets whose uniform bias is $O ( N ^ { - \nu / d } )$ . This matches the minimax worst-case expected simple regret rate over the fixed Matérn RKHS ball.

The proof is given in Appendix D.3. The minimax lower bound used in this comparison is the classical bound for the function class, recalled in Appendix B.4, instead of the lower bound specific to standard GP-TS in Section 4.

Remark 3. For Matérn kernels, GP-CTS attains the minimax worst-case expected convergence rate $\Theta ( N ^ { - \nu / d } ) f o r$ both discrete and continuous implementations. In particular, for the fixed one-dimensional Matérn-5/2 kernel, this eliminates the log N penalty shown to be intrinsic to the standard GP-TS. This improvement requires a known valid RKHS norm upper bound B, which is an additional input compared with standard GP-TS. Afixed conservative choice of B affects the constants but not the convergence rate, although it may influence finite-budget performance. The guarantees retain exact posterior sampling and account for global optimization errors through the explicit tolerance in (54).

## 6 Conclusions

This paper investigates the convergence rates of Gaussian Process Thompson Sampling (GP-TS) in efficient global optimization under noiseless observations. We first establish direct simple regret upper bounds for both discrete and continuous implementations of GP-TS. Our analysis combines conditional optimism with a deterministic bound on posterior uncertainty along arbitrary subsequences of queries. For Matérn kernels with smoothness parameter ν in dimension d, we obtain the worst-case expected simple regret upper bound $O ( N ^ { - \nu / d } \sqrt { \log N } )$ for both implementations, recovering the full smoothness exponent $\nu / d .$

We then investigate whether the additional log N factor is necessary. For a fixed one-dimensional Matérn-5/2 kernel, we establish matching worst-case expected simple regret lower bounds for both implementations. The analysis shows that positive Gaussian fluctuations over a suboptimal region can dominate sampled values near the true maximizer. These results demonstrate that, in this setting, the logarithmic factor is inherent to standard GP-TS instead of an artifact of the upper-bound analysis.

Motivated by these findings, we propose Gaussian Process Capped Thompson Sampling (GP-CTS), which caps posterior sample paths using a deterministic RKHS upper envelope. Given a valid RKHS norm upper bound, the cap preserves optimism while controlling positive sample deviations. This allows us to remove the logarithmic factor and establish the minimax worst-case expected simple regret rate $\Theta ( N ^ { - \nu / d } )$ for Matérn kernels in both discrete and continuous implementations.

Several questions remain. Extending the logarithmic lower bounds beyond the present Matérn parameters would clarify how widely the identified mechanism applies. For GP-CTS, obtaining a valid norm upper bound without prior knowledge of the function class remains an additional challenge.

## References

[1] Adam D. Bull. Convergence rates of efficient global optimization algorithms. Journal of Machine Learning Research, 12(88):2879–2904, 2011. URL http://jmlr.org/papers/v12/bull11a.html.

[2] Sayak Ray Chowdhury and Aditya Gopalan. On kernelized multi-armed bandits. In International Conference on Machine Learning, pages 844–853. PMLR, 2017.

[3] David Eriksson, Michael Pearce, Jacob Gardner, Ryan D Turner, and Matthias Poloczek. Scalable global optimization via local bayesian optimization. Advances in neural information processing systems, 32, 2019.

[4] Peter I Frazier. A tutorial on bayesian optimization. arXiv preprint arXiv:1807.02811, 2018.

[5] Shogo Iwazaki and Shion Takeno. Improved regret analysis in gaussian process bandits: Optimality for noiseless reward, rkhs norm, and non-stationary variance. arXiv preprint arXiv:2502.06363, 2025.

[6] Donald R Jones, Matthias Schonlau, and William J Welch. Efficient global optimization of expensive black-box functions. Journal ofGlobal optimization, 13:455–492, 1998.

[7] Kirthevasan Kandasamy, Akshay Krishnamurthy, Jeff Schneider, and Barnabás Póczos. Parallelised bayesian optimisation via thompson sampling. In International conference on artificial intelligence and statistics, pages 133–142. PMLR, 2018.

[8] Carl Edward Rasmussen and Christopher K. I. Williams. Gaussian processesfor machine learning. Adaptive computation and machine learning. MIT Press, 2006. ISBN 026218253X. URL https://www.worldcat.org/ oclc/61285753.

[9] Somjit Roy, Prateek Jaiswal, Anirban Bhattacharya, Debdeep Pati, and Bani K Mallick. Frequentist regret analysis of gaussian process thompson sampling via fractional posteriors. arXiv preprint arXiv:2602.14472, 2026.

[10] Jerome Sacks, William J Welch, Toby J Mitchell, and Henry P Wynn. Design and analysis of computer experiments. Statistical science, 4(4):409–423, 1989.

[11] Niranjan Srinivas, Andreas Krause, Sham M Kakade, and Matthias Seeger. Gaussian process optimization in the bandit setting: No regret and experimental design. arXiv preprint arXiv:0912.3995, 2009.

[12] Shion Takeno and Shogo Iwazaki. Improved regret analysis for parallel gaussian process bandit optimization. arXiv preprint arXiv:2608.16492, 2026.

[13] Aad W Van Der Vaart, J Harry Van Zanten, et al. Reproducing kernel hilbert spaces of gaussian priors. IMS Collections, 3:200–222, 2008.

[14] Emmanuel Vázquez and Sébastien Petit. Simple-regret rates and minimax optimality of fixed-prior expected improvement in matérn and squared-exponential rkhss. CoRR, abs/2607.29245, 2026. doi: 10.48550/ARXIV. 2607.29245. URL https://doi.org/10.48550/arXiv.2607.29245.

## A Proofs for Section 2

Posterior identities in (5)–(7). Fix a finite design $X _ { n }$ , regardless of how it was selected. If $a \in$ ker $K _ { n } .$ , then

$$
\bigg \| \sum _ { i = 1 } ^ { n } a _ { i } k _ { x _ { i } } \bigg \| _ { \mathcal { H } _ { k } } ^ { 2 } = a ^ { \top } K _ { n } a = 0 .\tag{A.1}
$$

Hence $\textstyle \sum _ { i } a _ { i } k _ { x _ { i } } = 0$ . Taking its inner product with $k _ { x }$ and with f gives $a ^ { \top } \mathbf { k } _ { n } ( x ) = 0$ and $a ^ { \top } \mathbf { y } _ { n } = 0$ . Since $K _ { n }$ is symmetric,

$$
\mathbf { k } _ { n } ( x ) , \mathbf { y } _ { n } \in \operatorname { r a n } K _ { n } = ( \ker K _ { n } ) ^ { \perp } .\tag{A.2}
$$

In particular, $K _ { n } K _ { n } ^ { \dagger } \mathbf y _ { n } = \mathbf y _ { n }$ . Let $\begin{array} { r } { g = \sum _ { i = 1 } ^ { n } ( K _ { n } ^ { \dagger } \mathbf { y } _ { n } ) _ { i } k _ { x _ { i } } \in W _ { n } } \end{array}$ . It interpolates the observations, and therefore

$$
\langle f - g , k _ { x _ { j } } \rangle _ { \mathcal { H } _ { k } } = f ( x _ { j } ) - g ( x _ { j } ) = 0 , \qquad j = 1 , \dotsc , n .
$$

Thus $g = P _ { n } f _ { \mathrm { : } }$ , establishing $\mu _ { n } = P _ { n } f$ even when $K _ { n }$ is singular.

Applying this argument with $f = k _ { x }$ gives

$$
P _ { n } k _ { x } = \sum _ { i = 1 } ^ { n } ( K _ { n } ^ { \dagger } \mathbf { k } _ { n } ( x ) ) _ { i } k _ { x _ { i } } .\tag{A.3}
$$

Since $P _ { n }$ is self-adjoint and idempotent,

$$
\begin{array} { r l } & { \langle ( I - P _ { n } ) k _ { x } , ( I - P _ { n } ) k _ { x ^ { \prime } } \rangle _ { \mathcal { H } _ { k } } = \langle k _ { x } , k _ { x ^ { \prime } } \rangle _ { \mathcal { H } _ { k } } - \langle P _ { n } k _ { x } , k _ { x ^ { \prime } } \rangle _ { \mathcal { H } _ { k } } } \\ & { \qquad = k ( x , x ^ { \prime } ) - \mathbf { k } _ { n } ( x ) ^ { \top } K _ { n } ^ { \dagger } \mathbf { k } _ { n } ( x ^ { \prime } ) } \\ & { \qquad = c _ { n } ( x , x ^ { \prime } ) . } \end{array}\tag{A.4}
$$

Setting $x ^ { \prime } = x$ proves $s _ { n } ( x ) = \| ( I - P _ { n } ) k _ { x } \| _ { \mathcal { H } _ { k } }$ . The reproducing property then gives

$$
\begin{array} { r l } & { | f ( x ) - \mu _ { n } ( x ) | = | \langle f , ( I - P _ { n } ) k _ { x } \rangle _ { \mathcal { H } _ { k } } | } \\ & { \qquad \leq \| f \| _ { \mathcal { H } _ { k } } \| ( I - P _ { n } ) k _ { x } \| _ { \mathcal { H } _ { k } } \leq B s _ { n } ( x ) . } \end{array}\tag{A.5}
$$

At an observed point, $k _ { x _ { i } } \in W _ { n }$ , so both $s _ { n } ( x _ { i } )$ and the interpolation error are zero. The same identities hold for $n = 0$ using $P _ { 0 } = 0$

Monotonicity and increment identities used below. For finite-dimensional subspaces $V \subseteq W \subset \mathcal { H } _ { k }$

$$
\mathrm { d i s t } _ { \mathscr { H } _ { k } } ( g , W ) \leq \mathrm { d i s t } _ { \mathscr { H } _ { k } } ( g , V ) .\tag{A.6}
$$

Thus additional exact observations cannot increase posterior standard deviation, including when observations are repeated. The reverse triangle inequality also yields

$$
| s _ { n } ( x ) - s _ { n } ( x ^ { \prime } ) | \leq \| ( I - P _ { n } ) ( k _ { x } - k _ { x ^ { \prime } } ) \| _ { \mathcal { H } _ { k } } \leq d _ { k } ( x , x ^ { \prime } ) .\tag{A.7}
$$

For a centered conditional posterior process $Z$ with covariance $c _ { n } .$

$$
\begin{array} { r l } & { \mathrm { V a r } ( Z ( x ) - Z ( x ^ { \prime } ) \mid \mathcal { F } _ { n } ) = c _ { n } ( x , x ) + c _ { n } ( x ^ { \prime } , x ^ { \prime } ) - 2 c _ { n } ( x , x ^ { \prime } ) } \\ & { \qquad = \| ( I - P _ { n } ) ( k _ { x } - k _ { x ^ { \prime } } ) \| _ { \mathcal { H } _ { k } } ^ { 2 } \le d _ { k } ( x , x ^ { \prime } ) ^ { 2 } . } \end{array}\tag{A.8}
$$

Here the posterior draw is fresh after $\mathcal { F } _ { n }$ . The statement is conditional on the realized history. Finally,

$$
| f ( x ) - f ( x ^ { \prime } ) | = | \langle f , k _ { x } - k _ { x ^ { \prime } } \rangle _ { \mathcal { H } _ { k } } | \leq B d _ { k } ( x , x ^ { \prime } ) .
$$

This proves (12). Kernel continuity implies $d _ { k } ( x , x ^ { \prime } )  0$ as $x ^ { \prime }  x ,$ so f is continuous and attains its maximum on X. Also, $| f ( x ) | \leq B { \sqrt { k ( x , x ) } } \leq B$ , proving $r _ { N } ^ { \mathrm { b e s t } } \leq 2 B$

## A.1 Proposition 1: Matérn Kernel Assumptions

Proof. We verify the two assumptions through a spectral representation. The construction concerns approximation spaces only and does not prescribe the algorithm’s query points.

Spectral representation. For fixed $\nu , \ell > 0$ , the underlying stationary Matérn kernel has representation

$$
\begin{array} { c } { \displaystyle \boldsymbol { k } ( \boldsymbol { x } , \boldsymbol { y } ) = \int _ { \mathbb { R } ^ { d } } e ^ { i \omega \cdot ( \boldsymbol { x } - \boldsymbol { y } ) } \rho ( \omega ) d \omega , } \\ { \displaystyle \rho ( \omega ) = c _ { d , \nu , \ell } \left( \frac { 2 \nu } { \ell ^ { 2 } } + \| \omega \| _ { 2 } ^ { 2 } \right) ^ { - ( \nu + d / 2 ) } , } \end{array}\tag{A.9}
$$

where the positive constant incorporates the fixed amplitude and the Fourier convention [8]. In particular,

$$
\int _ { \mathbb { R } ^ { d } } \rho ( \omega ) d \omega = k ( x , x ) \leq 1 , \qquad \rho ( \omega ) \leq C ( 1 + \| \omega \| _ { 2 } ^ { 2 } ) ^ { - ( \nu + d / 2 ) } .\tag{A.10}
$$

All constants in this proof may depend on the fixed kernel and d.

Consider the real Hilbert space

$$
{ \mathcal { L } } _ { \rho } = \{ g \in L ^ { 2 } ( \rho ; \mathbb { C } ) : g ( - \omega ) = { \overline { { g ( \omega ) } } } { \mathrm { ~ a l m o s t ~ e v e r y w h e r e } } \} ,
$$

with inner product Re $\int g \overline { { { h } } } \rho .$ The feature $\phi _ { x } ( \omega ) = e ^ { i \omega \cdot x }$ satisfies $\langle \phi _ { x } , \phi _ { y } \rangle _ { \mathcal { L } _ { \rho } } = k ( x , y )$ . Thus $U k _ { x } = \phi _ { x }$ extends to an isometric isomorphism from $\mathcal { H } _ { k }$ onto

$$
\begin{array} { r } { S = \overline { { \operatorname { s p a n } _ { \mathbb { R } } \{ \phi _ { x } : x \in \mathcal { X } \} } } \subseteq \mathcal { L } _ { \rho } . } \end{array}\tag{A.11}
$$

Finite-dimensional approximation. Choose an integer $r > \nu$ and write

$$
D _ { r } = \# \{ \beta \in \mathbb { N } _ { 0 } ^ { d } : | \beta | \leq r - 1 \} = { \binom { d + r - 1 } { d } } .
$$

Partition $[ 0 , 1 ] ^ { d }$ into $J ^ { d }$ cubes of side length $1 / J ,$ and let $\mathcal { Z } _ { J }$ consist of their lower corners. Every $x \in \mathcal { X }$ admits a corner z with $\| x - z \| _ { 2 } \leq { \sqrt { d } } / J .$ . Define the ambient approximation space

$$
\begin{array} { r } { \widetilde { V } _ { J } = \operatorname { s p a n } _ { \mathbb { R } } \left\{ \mathbf { 1 } _ { \{ \| \omega \| _ { 2 } \leq J \} } e ^ { i \omega \cdot z } ( i \omega ) ^ { \beta } : z \in \mathcal { Z } _ { J } , \ | \beta | \leq r - 1 \right\} . } \end{array}\tag{A.12}
$$

Every spanning function belongs to $\mathcal { L } _ { \rho } .$ , since it has compact frequency support and Hermitian symmetry. Moreover, dim $\widetilde V _ { J } \leq D _ { r } J ^ { d }$

For the corner associated with $x ,$ set

$$
q _ { x , J } ( \omega ) = \mathbf { 1 } _ { \{ \| \omega \| _ { 2 } \leq J \} } e ^ { i \omega \cdot z } \sum _ { a = 0 } ^ { r - 1 } \frac { ( i \omega \cdot ( x - z ) ) ^ { a } } { a ! } .
$$

The multinomial expansion shows that $q _ { x , J } \in \widetilde { V } _ { J }$ . For real u, the integral Taylor remainder gives

$$
\left| e ^ { i u } - \sum _ { a = 0 } ^ { r - 1 } \frac { ( i u ) ^ { a } } { a ! } \right| \leq \frac { | u | ^ { r } } { r ! } .
$$

Consequently,

$$
\begin{array} { r l } & { \displaystyle \| \phi _ { x } - q _ { x , J } \| _ { \mathcal L _ { \rho } } ^ { 2 } \leq C _ { d , r } J ^ { - 2 r } \int _ { \| \omega \| _ { 2 } \leq J } \| \omega \| _ { 2 } ^ { 2 r } \rho ( \omega ) d \omega } \\ & { \qquad + \displaystyle \int _ { \| \omega \| _ { 2 } > J } \rho ( \omega ) d \omega . } \end{array}\tag{A.13}
$$

By polar integration and (A.10),

$$
\begin{array} { l } { \displaystyle \int _ { \| \omega \| _ { 2 } > J } \rho ( \omega ) d \omega \le C \displaystyle \int _ { J } ^ { \infty } s ^ { - 2 \nu - 1 } d s \le C ^ { \prime } J ^ { - 2 \nu } , } \\ { \displaystyle \int _ { \| \omega \| _ { 2 } \le J } \| \omega \| _ { 2 } ^ { 2 r } \rho ( \omega ) d \omega \le C \left( 1 + \displaystyle \int _ { 1 } ^ { J } s ^ { 2 r - 2 \nu - 1 } d s \right) \le C ^ { \prime } J ^ { 2 r - 2 \nu } . } \end{array}\tag{A.14}
$$

The second inequality uses $r > \nu$ . Substitution into (A.13) gives a uniform error at most $C J ^ { - \nu }$

The approximants in $\widetilde { V } _ { J }$ need not lie in S. Let $P _ { S }$ be its orthogonal projection and set

$$
{ \cal V } _ { \cal J } = { \cal U } ^ { - 1 } ( P _ { S } \widetilde { \cal V } _ { \cal J } ) \subset \mathcal { H } _ { k } .
$$

Its dimension is at most $D _ { r } J ^ { d }$ . Because $\phi _ { x } \in S$ , projection cannot increase the error:

$$
\| \phi _ { x } - P _ { S } q _ { x , J } \| _ { \mathcal { L } _ { \rho } } \leq \| \phi _ { x } - q _ { x , J } \| _ { \mathcal { L } _ { \rho } } .
$$

Hence

$$
\operatorname* { s u p } _ { x \in \mathcal { X } } \mathrm { d i s t } _ { \mathcal { H } _ { k } } ( k _ { x } , V _ { J } ) \le C J ^ { - \nu } .\tag{A.15}
$$

For $m \geq D _ { r }$ , take $J = \lfloor ( m / D _ { r } ) ^ { 1 / d } \rfloor$ . Then $D _ { r } J ^ { d } \leq m$ and $\begin{array} { r } { J \ge \frac { 1 } { 2 } ( m / D _ { r } ) ^ { 1 / d } ; } \end{array}$ , so

$$
w _ { m } ( k , \mathcal { X } ) \leq C 2 ^ { \nu } D _ { r } ^ { \nu / d } m ^ { - \nu / d } .\tag{A.16}
$$

For $1 \leq m < D _ { r }$ , the zero subspace gives $w _ { m } \leq 1$ . Enlarging the constant covers these finitely many m and proves the polynomial-width assumption with $\alpha = \nu / d .$

Canonical-metric regularity. Take $0 < \zeta \leq 1$ with $\zeta < \nu .$ . The elementary inequality $| e ^ { i u } - 1 | \leq 2 | u | ^ { \zeta }$ follows by treating $| u | \leq 1$ and $| u | > 1$ separately. Therefore,

$$
\begin{array} { r l r } {  { d _ { k } ( x , y ) ^ { 2 } = \int _ { \mathbb { R } ^ { d } } | e ^ { i \omega \cdot x } - e ^ { i \omega \cdot y } | ^ { 2 } \rho ( \omega ) d \omega } } \\ & { } & { \leq 4 \| x - y \| _ { 2 } ^ { 2 \zeta } \int _ { \mathbb { R } ^ { d } } \| \omega \| _ { 2 } ^ { 2 \zeta } \rho ( \omega ) d \omega . } \end{array}\tag{A.17}
$$

The last integral is finite: its radial tail is bounded by a constant multiple of $\textstyle \int _ { 1 } ^ { \infty } s ^ { 2 \zeta - 2 \nu - 1 } d s$ , which converges when $\zeta < \nu .$ Taking square roots proves Assumption 3 for every stated $\zeta ,$ including $\dot { \zeta } = 1 \operatorname { i f } \nu > 1$ □

Continuous samples and measurable selection in Algorithm 1. For a centered $\mathrm { G P } \Xi$ with a kernel satisfying Assumption 3, Gaussian moments give, for every $q > 0$

$$
\begin{array} { r } { \mathbb { E } | \Xi ( x ) - \Xi ( y ) | ^ { q } = c _ { q } d _ { k } ( x , y ) ^ { q } \leq c _ { q } L _ { k } ^ { q } \| x - y \| _ { 2 } ^ { q \zeta } . } \end{array}
$$

Choose $q > d / \zeta$ . The parameter-dimensional Kolmogorov–Chentsov continuity theorem supplies a continuous modification on $[ 0 , 1 ] ^ { d }$ . Given a realized history, let $\Xi _ { n }$ be a fresh such prior draw independent of that history and define

$$
Z _ { n } ( x ) = \Xi _ { n } ( x ) - \mathbf { k } _ { n - 1 } ( x ) ^ { \top } K _ { n - 1 } ^ { \dag } \Xi _ { n } ( X _ { n - 1 } ) .\tag{A.18}
$$

Its conditional finite-dimensional distributions are centered Gaussian. Using $K _ { n - 1 } ^ { \dagger } K _ { n - 1 } K _ { n - 1 } ^ { \dagger } = K _ { n - 1 } ^ { \dagger }$ , direct covariance expansion gives

$$
\operatorname { C o v } ( Z _ { n } ( x ) , Z _ { n } ( x ^ { \prime } ) \mid \mathcal { F } _ { n - 1 } ) = k ( x , x ^ { \prime } ) - \mathbf { k } _ { n - 1 } ( x ) ^ { \top } K _ { n - 1 } ^ { \dagger } \mathbf { k } _ { n - 1 } ( x ^ { \prime } ) = c _ { n - 1 } ( x , x ^ { \prime } ) .
$$

The correction is a finite linear combination of continuous kernel sections, so the posterior paths remain continuous. This verifies compatibility of the continuous sampling model with the stated regularity. It is not a finite-feature implementation theorem.

For a finite decision set, a fixed ordering resolves ties measurably. For the continuous domain, consider $C ( \mathcal { X } )$ with its supremum-norm Borel sigma-algebra. Given $g \in C ( \mathcal X )$ , set $m ( g ) = \operatorname* { m a x } _ { \mathcal { X } } g .$ Among the closed dyadic subcubes in a fixed order, select the first $C$ satisfying max<sub>C</sub> $\operatorname { \mu } _ { \boldsymbol { g } } = m ( \boldsymbol { g } )$ , and repeat among its children. The selected nested cubes have diameters tending to zero and each contains a maximizer. Their centers converge to a point $x ( g )$ satisfying $g ( x ( g ) ) = m ( g )$ . For each fixed compact cube C, the map $g \mapsto \operatorname* { m a x } _ { C } g$ is continuous. Thus each finite choice is Borel measurable, and so is the limiting selector $x ( g )$ . This proves the required existence without claiming computational efficiency.

## B Proofs for Section 3

## B.1 Lemma 1: Arbitrary-Subsequence Innovation Bound

Proof. We give the width-to-separation argument in the precise form used here, following the geometry of Vázquez and Petit [14].

Gram eigenvalue tails. Take arbitrary $z _ { 1 } , \ldots , z _ { m } \in \mathcal { X }$ , let $g _ { i } = k _ { z _ { i } }$ , and form $K = [ \langle g _ { i } , g _ { j } \rangle _ { \mathcal { H } _ { k } } ] _ { i , i = 1 } ^ { m }$ . Write its eigenvalues as $\lambda _ { 1 } \geq \cdot \cdot \cdot \geq \lambda _ { m } \geq 0$ . For $1 \leq \ell < m$ and any $\epsilon > 0 .$ , Assumption 2 provides a subspace V of dimension at most ℓ such that

$$
\operatorname* { s u p } _ { x \in \mathcal { X } } \mathrm { d i s t } _ { \mathcal { H } _ { k } } ( k _ { x } , V ) \leq A \ell ^ { - \alpha } + \epsilon .
$$

Let $P$ project onto V . The finite-rank operator $\begin{array} { r } { S g = \sum _ { i = 1 } ^ { m } \langle g , g _ { i } \rangle _ { \mathcal { H } _ { k } } g _ { i } } \end{array}$ has the same nonzero eigenvalues as $K$ For orthonormal eigenvectors $e _ { j }$ corresponding to its positive eigenvalues, put $a _ { j } = \| P e _ { j } \| _ { \mathcal { H } _ { k } } ^ { 2 }$ . Then $0 \leq a _ { j } \leq 1$ $\textstyle \sum _ { j } a _ { j } \leq \ell ,$ and

$$
\sum _ { i = 1 } ^ { m } \| ( I - P ) g _ { i } \| _ { \mathcal { H } _ { k } } ^ { 2 } = \sum _ { j } \lambda _ { j } ( 1 - a _ { j } ) .
$$

The decreasing ordering of the eigenvalues implies $\begin{array} { r } { \sum _ { j } \lambda _ { j } a _ { j } \leq \sum _ { j = 1 } ^ { \ell } \lambda _ { j } } \end{array}$ . For example, when ℓ is below the rank,

$$
\begin{array} { r l } & { \displaystyle \sum _ { j } \lambda _ { j } a _ { j } \leq \sum _ { j \leq \ell } \lambda _ { j } a _ { j } + \lambda _ { \ell } \sum _ { j > \ell } a _ { j } } \\ & { \qquad \leq \displaystyle \sum _ { j \leq \ell } \bigl [ \lambda _ { j } a _ { j } + \lambda _ { \ell } ( 1 - a _ { j } ) \bigr ] \leq \sum _ { j \leq \ell } \lambda _ { j } . } \end{array}
$$

If ℓ is at least the rank, the same assertion is immediate. Thus

$$
\sum _ { j > \ell } \lambda _ { j } \leq \sum _ { i = 1 } ^ { m } \| ( I - P ) g _ { i } \| _ { \mathcal { H } _ { k } } ^ { 2 } \leq m ( A \ell ^ { - \alpha } + \epsilon ) ^ { 2 } .
$$

Letting $\epsilon \downarrow 0$ gives

$$
\sum _ { j > \ell } \lambda _ { j } \le m A ^ { 2 } \ell ^ { - 2 \alpha } .\tag{B.1}
$$

No attainment of the Kolmogorov-width infimum is required.

Determinant bound. For $j \geq 2 ,$ , take $\ell = \lfloor j / 2 \rfloor$ . Since $j - \ell \geq j / 2$ and $\ell \geq j / 3 ,$

$$
( j - \ell ) \lambda _ { j } \leq \sum _ { r = \ell + 1 } ^ { j } \lambda _ { r } \leq m A ^ { 2 } \ell ^ { - 2 \alpha } .
$$

It follows that $\lambda _ { j } ~ \le ~ 2 A ^ { 2 } 3 ^ { 2 \alpha } m j ^ { - ( 1 + 2 \alpha ) }$ . For $j = 1$ , the diagonal bound gives $\lambda _ { 1 } \leq \operatorname { t r } K \leq m$ . With $C _ { 0 } =$ max $\left\{ 1 , 2 A ^ { 2 } 3 ^ { 2 \alpha } \right\}$ , we therefore have

$$
\lambda _ { j } \le C _ { 0 } m j ^ { - ( 1 + 2 \alpha ) } , \qquad 1 \le j \le m .
$$

Using m! $\ge ( m / e ) ^ { m }$

$$
\begin{array} { l } { \displaystyle \operatorname* { d e t } K \leq \frac { ( C _ { 0 } m ) ^ { m } } { ( m ! ) ^ { 1 + 2 \alpha } } } \\ { \leq \left( C _ { 0 } e ^ { 1 + 2 \alpha } m ^ { - 2 \alpha } \right) ^ { m } = \left( C _ { \mathrm { s e p } } ^ { 2 } m ^ { - 2 \alpha } \right) ^ { m } , } \end{array}\tag{B.2}
$$

where $C _ { \mathrm { s e p } }$ is the constant in (18). This also covers $m = 1$ and singular Gram matrices.

Ordered innovations and actual histories. Define

$$
\widetilde { s } _ { j } = \mathrm { d i s t } _ { \mathcal { H } _ { k } } \left( g _ { j } , \mathrm { s p a n } \{ g _ { 1 } , \dots , g _ { j - 1 } \} \right) ,
$$

with the empty span interpreted as {0}. Gram–Schmidt gives

$$
\operatorname * { d e t } { K } = \prod _ { j = 1 } ^ { m } \widetilde { s } _ { j } ^ { 2 } .\tag{B.3}
$$

If the sections are linearly dependent, the first dependent section has zero innovation and both sides are zero. The minimum is bounded by the geometric mean, so

$$
\operatorname* { m i n } _ { 1 \leq j \leq m } \widetilde s _ { j } \leq ( \operatorname* { d e t } K ) ^ { 1 / ( 2 m ) } \leq C _ { \mathrm { s e p } } m ^ { - \alpha } .\tag{B.4}
$$

Now write $I = \{ n _ { 1 } < \cdots < n _ { m } \}$ and set $z _ { j } = x _ { n _ { j } }$ . The preceding subsequence span satisfies

$$
\operatorname { s p a n } \{ k _ { x _ { n _ { 1 } } } , \dots , k _ { x _ { n _ { j - 1 } } } \} \subseteq W _ { n _ { j } - 1 } .
$$

By $( \mathrm { A } . 6 ) , s _ { n _ { i } - 1 } ( x _ { n _ { i } } ) \leq \widetilde { s } _ { j }$ . Taking minima and applying (B.4) proves (17). Every step holds for any realized sequence and subset. Random or post hoc selection of I requires no additional probability bound.

For (19), take $I = \{ n \leq N : s _ { n - 1 } ( x _ { n } ) > u \} . \mathrm { I f } m = | I | > 0 ,$ then

$$
u < \operatorname* { m i n } _ { n \in I } s _ { n - 1 } ( x _ { n } ) \leq C _ { \mathrm { s e p } } m ^ { - \alpha } ,
$$

which gives $m < ( C _ { \mathrm { s e p } } / u ) ^ { 1 / \alpha }$ and hence the stated ceiling bound. The case $I = \emptyset$ is immediate.

## B.2 Lemma 2: Conditional Lower-Tail Bound and Optimism

Proof. Set $q _ { n } = \mathbb { P } ( O _ { n } = 1 \mid { \mathcal { F } } _ { n - 1 } ) \geq p .$ For $\theta > 0 .$ , since $e ^ { - \theta } < 1$

$$
\mathbb { E } [ e ^ { - \theta O _ { n } } \mid \mathcal { F } _ { n - 1 } ] = 1 - q _ { n } + q _ { n } e ^ { - \theta } \leq 1 - p + p e ^ { - \theta } .\tag{B.5}
$$

Let $\textstyle S _ { n } = \sum _ { j = 1 } ^ { n } O _ { j }$ and $S _ { 0 } = 0$ . The tower property gives

$$
\begin{array} { r l } & { \mathbb { E } e ^ { - \theta S _ { n } } = \mathbb { E } \left[ e ^ { - \theta S _ { n - 1 } } \mathbb { E } ( e ^ { - \theta O _ { n } } \mid \mathcal { F } _ { n - 1 } ) \right] } \\ & { \qquad \leq ( 1 - p + p e ^ { - \theta } ) \mathbb { E } e ^ { - \theta S _ { n - 1 } } . } \end{array}
$$

Iteration yields $\mathbb { E } e ^ { - \theta S _ { N } } \le ( 1 - p + p e ^ { - \theta } ) ^ { N }$ . Choose $\theta = \log 2$ and apply Markov’s inequality:

$$
\begin{array} { r l } & { \mathbb { P } ( S _ { N } < p N / 2 ) \le e ^ { ( \log 2 ) p N / 2 } \mathbb { E } e ^ { - ( \log 2 ) S _ { N } } } \\ & { \qquad \le e ^ { ( \log 2 ) p N / 2 } ( 1 - p / 2 ) ^ { N } } \\ & { \qquad \le \exp \left( - \frac { 1 - \log 2 } { 2 } p N \right) \le e ^ { - p N / 8 } . } \end{array}\tag{B.6}
$$

The last steps use $1 - z \leq e ^ { - z }$ and $( 1 - \log 2 ) / 2 > 1 / 8$ . No independence across rounds is used.

Application to the optimistic rounds of GP-TS. Fix a comparator $x ^ { \dagger }$ independently of the current and past algorithmic randomization. Conditional on $\mathcal { F } _ { n - 1 }$ , write $a = \mu _ { n - 1 } ( x ^ { \dagger } )$ and $s = s _ { n - 1 } ( x ^ { \dagger } )$ . If $s > 0 ,$ , the exact conditional sampling rule gives $G _ { n } ( x ^ { \dagger } ) = a + v s \xi$ in conditional distribution, where $\xi \sim \mathcal { N } ( 0 , 1 )$ . By $( 7 ) , f ( x ^ { \dagger } ) - a \leq$ $\boldsymbol { B s }$ . Hence

$$
\begin{array} { r l } & { \mathbb { P } ( G _ { n } ( x ^ { \dagger } ) \geq f ( x ^ { \dagger } ) \mid \mathcal { F } _ { n - 1 } ) = 1 - \Phi \left( \frac { f ( x ^ { \dagger } ) - a } { v s } \right) } \\ & { \qquad \geq 1 - \Phi ( B / v ) = \Phi ( - B / v ) . } \end{array}\tag{B.7}
$$

If $s = 0 ,$ , interpolation implies $a = f ( x ^ { \dagger } )$ , and the centered Gaussian residual is zero almost surely. The optimism probability is then 1. Since f and the finite decision set are fixed, both $x ^ { * }$ and $x _ { g } ^ { * }$ are valid comparators. The same Gaussian argument cannot be applied directly at the current sample-dependent maximizer $x _ { n }$

## B.3 Theorem 1: Discrete Decision Domain

Proof. Write $u = u _ { N , M , \delta } .$ . For $\xi \sim \mathcal { N } ( 0 , 1 ) , \mathbb { E } e ^ { \lambda \xi } = e ^ { \lambda ^ { 2 } / 2 }$ and Markov’s inequality give $\mathbb { P } ( | \xi | > u ) \le 2 e ^ { - u ^ { 2 } / 2 }$ for $u \geq 0$ . For each fixed $x \in { \mathcal { G } }$ , apply this tail bound conditionally on $\mathcal { F } _ { n - 1 }$ to the centered residual:

$$
\begin{array} { r } { \mathbb { P } \left( | Z _ { n } ( x ) | > u s _ { n - 1 } ( x ) ~ | ~ \mathcal { F } _ { n - 1 } \right) \le 2 e ^ { - u ^ { 2 } / 2 } . } \end{array}
$$

When $s _ { n - 1 } ( x ) = 0$ , the residual is zero almost surely and the bound remains valid without division.

Define

$$
\begin{array} { r } { \mathcal { E } _ { \mathrm { e n v } } = \{ \forall 1 \leq n \leq N , x \in \mathcal { G } : | Z _ { n } ( x ) | \leq u s _ { n - 1 } ( x ) \} . } \end{array}
$$

Integrate over histories and take a union bound over the NM pairs:

$$
\mathbb { P } _ { f } ( \mathcal { E } _ { \mathrm { e n v } } ^ { c } ) \le 2 N M e ^ { - u ^ { 2 } / 2 } = \delta / 2 .\tag{B.8}
$$

On ${ \mathcal { E } } _ { \mathrm { e n v } } .$

$$
| G _ { n } ( x ) - f ( x ) | \leq | \mu _ { n - 1 } ( x ) - f ( x ) | + v | Z _ { n } ( x ) | \leq ( B + v u ) s _ { n - 1 } ( x )\tag{B.9}
$$

holds simultaneously for all n and $x \in { \mathcal { G } }$

Fix $x ^ { \dagger } = x _ { \mathcal { G } } ^ { * }$ and define $I _ { N } = \{ n \leq N : G _ { n } ( x _ { \mathcal { G } } ^ { * } ) \geq f _ { \mathcal { G } } ^ { * } \}$ . Equation (B.7) and Lemma 2 imply

$$
\begin{array} { r } { \mathbb { P } _ { f } ( | I _ { N } | < p N / 2 ) \le e ^ { - p N / 8 } \le \delta / 2 , } \end{array}
$$

where the budget condition gives the final inequality. Thus $\mathcal { E } _ { \mathrm { e n v } } \cap \{ | I _ { N } | \geq p N / 2 \}$ has probability at least $1 - \delta .$ On this intersection, every $n \in { \cal I } _ { N }$ satisfies

$$
\begin{array} { r l } { f _ { \mathcal { G } } ^ { * } - f ( x _ { n } ) \le G _ { n } ( x _ { \mathcal { G } } ^ { * } ) - f ( x _ { n } ) } & { } \\ { \le G _ { n } ( x _ { n } ) - f ( x _ { n } ) + \eta _ { n } } & { } \\ { \le ( B + v u ) s _ { n - 1 } ( x _ { n } ) + \eta _ { n } . } \end{array}\tag{B.10}
$$

The second line uses (16). The final line uses the uniform envelope before substituting the random $x _ { n }$ . Lemma 1 gives an index $n _ { \mathrm { o } } \in I _ { N }$ such that

$$
\begin{array} { r } { s _ { n _ { \mathrm { 0 } } - 1 } ( x _ { n _ { \mathrm { 0 } } } ) \leq C _ { \mathrm { s e p } } | I _ { N } | ^ { - \alpha } \leq C _ { \mathrm { s e p } } ( p N / 2 ) ^ { - \alpha } . } \end{array}
$$

Consequently,

$$
\begin{array} { r l r } {  { r _ { N } ^ { \mathrm { b e s t } } = b _ { \mathcal { G } } + f _ { \mathcal { G } } ^ { * } - \underset { 1 \leq n \leq N } { \operatorname* { m a x } } f ( x _ { n } ) } } \\ & { } & { \leq b _ { \mathcal { G } } + f _ { \mathcal { G } } ^ { * } - f ( x _ { n _ { \circ } } ) } \\ & { } & { \leq b _ { \mathcal { G } } + C _ { \mathrm { s e p } } ( B + v u ) ( p N / 2 ) ^ { - \alpha } + \underset { 1 \leq n \leq N } { \operatorname* { m a x } } \eta _ { n } . } \end{array}
$$

This is (23). The two events need not be independent, and the algorithm need not identify $n _ { \circ }$

## B.4 Corollary 1: Discrete Matérn Rate

Proof. We first control the grid bias and then derive the expected-regret bound.

Grid bias and cardinality. Choose a grid representative $z \in \mathcal { Q } _ { J }$ for $x ^ { * }$ with $\| x ^ { * } - z \| _ { 2 } \leq h _ { J }$ . By (12),

$$
0 \leq b _ { \mathcal Q _ { J } } \leq f ( x ^ { * } ) - f ( z ) \leq B L _ { k } h _ { J } ^ { \zeta } .
$$

The choice in (25) implies $h _ { J _ { N } } \leq N ^ { - \alpha / \zeta }$ , and hence $b g _ { N } \le B L _ { k } N ^ { - \alpha }$ uniformly over the RKHS ball. Also,

$$
J _ { N } + 1 \le ( \sqrt { d } + 2 ) N ^ { \alpha / \zeta } , \qquad | { \mathcal { G } } _ { N } | \le ( \sqrt { d } + 2 ) ^ { d } N ^ { \alpha d / \zeta } .
$$

With $C _ { d } = ( \sqrt { d } + 2 ) ^ { d }$

$$
u _ { N , | \mathcal { G } _ { N } | , \delta } ^ { 2 } \le 2 \log \frac { 4 C _ { d } } { \delta } + 2 \left( 1 + \frac { \alpha d } { \zeta } \right) \log N \le C \log ( N / \delta ) , \qquad N \ge 2 .\tag{B.11}
$$

Substitute these bounds into Theorem 1, set $\eta _ { n } = 0$ , and use $\alpha = \nu / d$ from Proposition 1. Since $\sqrt { \log ( N / \delta ) } \geq \sqrt { \log 2 } .$ the constant $B$ term is absorbed into the same rate. This proves (27) with constants independent of f and δ.

Worst-case expected regret. Set $\alpha = \nu / d$ and choose

$$
\delta _ { N } = N ^ { - ( \alpha + 1 ) } , \qquad N \ge 2 .
$$

The budget condition becomes

$$
N \geq { \frac { 8 } { p } } \bigl [ \log 2 + ( \alpha + 1 ) \log N \bigr ] .\tag{B.12}
$$

Since $N /$ log $N \to \infty$ , there is an integer $N _ { 0 } \geq 2$ for which (B.12) holds for all $N \geq N _ { 0 }$ . For definiteness, $N _ { 0 }$ can be any integer from which this inequality holds for every subsequent integer. It depends on p and $\alpha ,$ , not on the particular $f .$ The high-probability bound and $r _ { N } ^ { \mathrm { b e s t } } \leq 2 B$ give

$$
\begin{array} { r l } & { \mathbb { E } _ { f } [ r _ { N } ^ { \mathrm { b e s t } } ] \leq C N ^ { - \alpha } \sqrt { \log ( N / \delta _ { N } ) } + 2 B \delta _ { N } } \\ & { \qquad \leq C ^ { \prime } N ^ { - \alpha } \sqrt { \log N } + 2 B N ^ { - ( \alpha + 1 ) } } \\ & { \qquad \leq C ^ { \prime \prime } N ^ { - \alpha } \sqrt { \log N } . } \end{array}\tag{B.13}
$$

All constants and $N _ { 0 }$ are uniform over $\| f \| _ { \mathcal { H } _ { k } } \le B$ . Taking the supremum over this ball proves (28). This does not require a single probability event that covers every f simultaneously. □

Minimax comparison under best-query loss. Let $\Re _ { N } ^ { \mathrm { r e c } }$ be the infimum of worst-case expected regret over sequential algorithms with N noiseless evaluations and any data-measurable final recommendation. Let $\Re _ { N } ^ { \mathrm { b e s t } }$ impose the additional requirement that the output is a queried point. For $N \geq 2 .$

$$
\Re _ { N } ^ { \mathrm { r e c } } \leq \Re _ { N } ^ { \mathrm { b e s t } } \leq \Re _ { N - 1 } ^ { \mathrm { r e c } } .
$$

The first inequality follows from inclusion of algorithm classes. For the second, use $N - 1$ evaluations to obtain a recommendation, query it in the final round, and return the best observed point. An arbitrarily near-optimal recommendation algorithm suffices before taking the infimum. The Matérn benchmark of Bull [1] therefore gives $\mathfrak { R } _ { N } ^ { \mathrm { b e s t } } \asymp N ^ { - \nu / d }$ as well. Corollary 1 matches this polynomial exponent but retains $\sqrt { \log N }$

Classical Matérn minimax lower bound. For completeness, we give the classical packing argument underlying the Matérn benchmark of Bull [1]. This is a lower bound for the problem class, not a new GP-TS-specific lower bound. Let $s = \nu + d / 2$ . Under the Fourier convention $\begin{array} { r } { { \widehat { g } } ( \omega ) = \int g ( x ) { \dot { e } } ^ { - i \omega \cdot x } d x } \end{array}$ , the whole-space Matérn norm has the form

$$
\| g \| _ { \mathcal H _ { k } ( \mathbb R ^ { d } ) } ^ { 2 } = C _ { k } \int _ { \mathbb R ^ { d } } | \widehat g ( \omega ) | ^ { 2 } ( a _ { k } + \| \omega \| _ { 2 } ^ { 2 } ) ^ { s } d \omega , \qquad a _ { k } , C _ { k } > 0 .\tag{B.14}
$$

The RKHS norm of a restriction to X is no larger than the norm of any whole-space extension.

Choose a nonnegative $\psi \in C _ { c } ^ { \infty } ( ( 0 , 1 ) ^ { d } )$ with max $\psi = 1$ and define

$$
g _ { h , z } ( x ) = c B h ^ { \nu } \psi ( ( x - z ) / h ) , \qquad 0 < h \leq 1 .
$$

Its Fourier transform is $c B h ^ { \nu + d } e ^ { - i \omega \cdot z } \widehat { \psi } ( h \omega )$ . Since $2 s = 2 \nu + d ,$ the change of variables $\xi = h \omega$ in (B.14) gives

$$
\begin{array} { r l r } {  { \| g _ { h , z } \| _ { \mathcal { H } _ { k } ( \mathbb { R } ^ { d } ) } ^ { 2 } = C _ { k } c ^ { 2 } B ^ { 2 } h ^ { 2 \nu + d - 2 s } \int _ { \mathbb { R } ^ { d } } | \widehat { \psi } ( \xi ) | ^ { 2 } ( a _ { k } h ^ { 2 } + \| \xi \| _ { 2 } ^ { 2 } ) ^ { s } d \xi } } \\ & { } & { \leq C _ { \psi } c ^ { 2 } B ^ { 2 } . } \end{array}\tag{B.15}
$$

The constant is finite because $\widehat { \psi }$ decays faster than every polynomial. Choosing a sufficiently small fixed $c > 0$ puts all restricted bumps in the radius-B RKHS ball.

For a budget N, take $J = \lceil ( 4 N ) ^ { 1 / d } \rceil$ and $h = J ^ { - 1 }$ . Place one translated bump inside each of the $M = J ^ { d } \geq 4 N$ disjoint grid cells. Their maximum values all equal $c B h ^ { \nu } \asymp N ^ { - \nu / d }$ . Choose the true objective uniformly among the bumps solely for a lower-bound averaging argument, and this choice is not a prior used by the algorithm.

Fix an arbitrary algorithm’s random seed and follow its all-zero observation history. Its N queries enter at most N bump supports, and its final recommendation enters at most one more. Before the active support is first queried, the true history coincides with this zero history. Therefore, even for adaptive algorithms, the probability over the uniform bump index that a query or final recommendation hits the active support is at most $( N + 1 ) \dot { / } M \leq 1 / \dot { 2 }$ . On the complementary event, the recommended value is zero and the regret equals the bump height. Averaging over the algorithm’s seed and then using that the maximum is at least the average yields

$$
\operatorname* { i n f } _ { \pmb { \mathscr { A } } } \operatorname* { s u p } _ { \| \pmb { \mathscr { H } } _ { k } \leq \boldsymbol { B } } \mathbb { E } _ { \boldsymbol { f } } ^ { \pmb { A } } [ \pmb { f } ^ { * } - \pmb { f } ( \widehat { \mathscr { x } } _ { N } ) ] \geq c ^ { \prime } N ^ { - \nu / d } .\tag{B.16}
$$

Here $\mathcal { A }$ ranges over randomized sequential algorithms with N noiseless queries and a data-measurable final recommen dation. The argument allows the algorithm to know B and to recommend an unqueried point. It therefore also applies when the output is restricted to the best queried point.

## B.5 Lemma 3: Continuous Residual Envelope

Proof. Fix integers $J , N \ge 1$ and $0 < \delta < 1$ , and abbreviate $u = u _ { N , J , \delta } ^ { \mathrm { c } }$ . We use the conditional increment contraction in (A.8), with the posterior based on the first $n - 1$ observations.

Nested grids. Define

$$
\mathcal { Q } _ { J , j } = \left\{ \frac { i } { J 2 ^ { j } } : i = 0 , \ldots , J 2 ^ { j } \right\} ^ { d } , \qquad j = 0 , 1 , \ldots .
$$

Then $\mathcal { Q } _ { J , 0 } = \mathcal { Q } _ { J }$ and $| \mathcal { Q } _ { J , i } | \leq M _ { J } 2 ^ { d j }$ . Let $\pi _ { j } { \mathrm { . } }$ x round each coordinate downward to the corresponding grid, preserving the endpoint 1. For $z \in \mathcal { Q } _ { J , j }$ , its parent is $\pi _ { j - 1 } z .$ . The maps satisfy

$$
\begin{array} { r l r } & { } & { \pi _ { j - 1 } ( \pi _ { j } x ) = \pi _ { j - 1 } x , \quad \| z - \pi _ { j - 1 } z \| _ { 2 } \leq h _ { J } 2 ^ { - j } , } \\ & { } & { \| x - \pi _ { 0 } x \| _ { 2 } \leq h _ { J } , \quad \quad \quad \pi _ { j } x \to x . \quad } \end{array}\tag{B.17}
$$

Coarse-grid event. At every fixed $z \in \mathcal { Q } _ { J }$ , the conditional Gaussian tail bound used in the discrete theorem gives

$$
\begin{array} { r } { \mathbb { P } _ { f } \left( \exists 1 \leq n \leq N , z \in \mathcal { Q } _ { J } : | Z _ { n } ( z ) | > u s _ { n - 1 } ( z ) \right) \leq 2 N M _ { J } e ^ { - u ^ { 2 } / 2 } = \delta / 4 . } \end{array}
$$

Call the complementary event ${ \mathcal { E } } _ { 0 }$

Refinement increments. For $j \geq 1$ , set

$$
a _ { j } = \sqrt { u ^ { 2 } + 2 ( d + 1 ) j \log 2 } .
$$

By (A.8), Assumption 3, and (B.17),

$$
\mathrm { V a r } \left( Z _ { n } ( z ) - Z _ { n } ( \pi _ { j - 1 } z ) \mid \mathcal { F } _ { n - 1 } \right) \le L _ { k } ^ { 2 } h _ { J } ^ { 2 \zeta } 2 ^ { - 2 j \zeta } .\tag{B.18}
$$

Taking conditional Gaussian tails, integrating over histories, and summing over n and the grid nodes gives

$$
\begin{array} { r l } & { \mathbb { P } _ { f } \Big ( \exists 1 \leq n \leq N , z \in \mathcal { Q } _ { J , j } : } \\ & { \qquad | Z _ { n } ( z ) - Z _ { n } ( \pi _ { j - 1 } z ) | > L _ { k } h _ { J } ^ { \zeta } 2 ^ { - j \zeta } a _ { j } \Big ) } \\ & { \qquad \leq 2 N M _ { J } 2 ^ { d j } e ^ { - a _ { j } ^ { 2 } / 2 } = \displaystyle \frac { \delta } { 4 } 2 ^ { - j } . } \end{array}\tag{B.19}
$$

A zero-variance increment satisfies the bound automatically. Let ${ \mathcal { E } } _ { 1 }$ be the event that all refinement bounds hold. Since $\textstyle \sum _ { j \geq 1 } 2 ^ { - j } = 1$ , we have

$$
\mathbb { P } _ { f } ( \mathcal { E } _ { 0 } \cap \mathcal { E } _ { 1 } ) \geq 1 - \frac { \delta } { 2 } .\tag{B.20}
$$

Neither the increments nor the rounds need to be independent.

Extension to the continuum. Work on this intersection and the probability-one event of continuity of the finitely many paths. For every $x \in \mathcal { X }$ , telescoping along $\pi _ { j } x$ gives

$$
| Z _ { n } ( x ) - Z _ { n } ( \pi _ { 0 } x ) | \leq L _ { k } h _ { J } ^ { \zeta } \sum _ { j = 1 } ^ { \infty } 2 ^ { - j \zeta } a _ { j } .\tag{B.21}
$$

The extension holds simultaneously for all x by continuity. No union bound over an uncountable set is taken. Set $\rho = 2 ^ { - \zeta }$ and $c _ { d } = \sqrt { 2 ( d + 1 ) \log { 2 } }$ . Since $a _ { j } \le u + c _ { d } \sqrt { j } \le u + c _ { d } j$

$$
\sum _ { j = 1 } ^ { \infty } \rho ^ { j } a _ { j } \le u \frac { \rho } { 1 - \rho } + c _ { d } \frac { \rho } { ( 1 - \rho ) ^ { 2 } } .
$$

Also, (A.7) and Assumption 3 imply $s _ { n - 1 } ( \pi _ { 0 } x ) \leq s _ { n - 1 } ( x ) + L _ { k } h _ { J } ^ { \zeta }$ . Combining with the coarse-grid event yields

$$
\begin{array} { l } { \displaystyle | Z _ { n } ( x ) | \leq u s _ { n - 1 } ( \pi _ { 0 } x ) + L _ { k } h _ { J } ^ { \zeta } \left[ u \frac { \rho } { 1 - \rho } + c _ { d } \frac { \rho } { ( 1 - \rho ) ^ { 2 } } \right] } \\ { \displaystyle \leq u s _ { n - 1 } ( x ) + L _ { k } h _ { J } ^ { \zeta } \left[ \frac { u } { 1 - \rho } + c _ { d } \frac { \rho } { ( 1 - \rho ) ^ { 2 } } \right] . } \end{array}
$$

The coefficients are exactly $C _ { 1 } ( \zeta )$ and $C _ { 2 } ( d , \zeta )$ in (30). This proves (31), without standardizing by a possibly vanishing $s _ { n - 1 } ( x )$ □

## B.6 Theorem 2: Continuous Decision Domain

Proof. Fix J. At a fixed true maximizer $x ^ { * }$ , define $I _ { N } = \{ n \leq N : G _ { n } ( x ^ { * } ) \geq f ^ { * } \}$ . By (B.7) and Lemma 2,

$$
\begin{array} { r } { \mathbb { P } _ { f } ( | I _ { N } | < p N / 2 ) \le e ^ { - p N / 8 } \le \delta / 2 . } \end{array}
$$

Intersect the complementary event with the event in Lemma 3. Their intersection has probability at least $1 - \delta .$ . Write

$$
D _ { J } = L _ { k } h _ { J } ^ { \zeta } \left[ C _ { 1 } ( \zeta ) u _ { N , J , \delta } ^ { \mathrm { c } } + C _ { 2 } ( d , \zeta ) \right] .
$$

For every $n \in I _ { N }$ , optimism, the global selection rule, and the continuous envelope imply

$$
\begin{array} { r l } & { f ^ { * } - f ( x _ { n } ) \leq G _ { n } ( x ^ { * } ) - f ( x _ { n } ) } \\ & { \qquad \leq G _ { n } ( x _ { n } ) - f ( x _ { n } ) + \eta _ { n } } \\ & { \qquad \leq ( B + v u _ { N , J , \delta } ^ { \mathrm { c } } ) s _ { n - 1 } ( x _ { n } ) + v D _ { J } + \eta _ { n } . } \end{array}\tag{B.22}
$$

The last bound is valid at the sample-dependent $x _ { n }$ because the envelope holds simultaneously for all x. Lemma 1 supplies $n _ { \mathrm { o } } \in I _ { N }$ with

$$
\begin{array} { r } { s _ { n _ { \circ } - 1 } ( x _ { n _ { \circ } } ) \leq C _ { \mathrm { s e p } } ( p N / 2 ) ^ { - \alpha } . } \end{array}
$$

Apply (B.22) at this index and use $r _ { N } ^ { \mathrm { b e s t } } \leq f ^ { * } - f ( x _ { n _ { \circ } } )$ to prove (33). Unlike in the finite decision case, there is no objective discretization bias. The grid used in this proof does not constrain the queries. □

## B.7 Corollary 2: Continuous Matérn Rate

Proof. Use Proposition 1 to set $\alpha = \nu / d$ and select a permissible ζ. In Theorem $^ { 2 , }$ choose the proof-grid resolution $J = J _ { N }$ from (25). Then $h _ { J } ^ { \zeta } \leq N ^ { - \alpha }$ , and the cardinality calculation in Appendix B.4 gives

$$
u _ { N , J , \delta } ^ { \mathrm { c } } = \mathcal { O } \big ( \sqrt { \log ( N / \delta ) } \big ) , \qquad N \geq 2 .
$$

With $\eta _ { n } = 0 .$ , both non-optimization terms in (33) are bounded by a fixed constant times $N ^ { - \alpha } \sqrt { \log ( N / \delta ) }$ . This proves the high-probability claim under the stated budget condition.

For the expectation, take $\delta _ { N } = N ^ { - ( \alpha + 1 ) }$ and an $N _ { 0 }$ satisfying (B.12) for every $N \geq N _ { 0 }$ . Repeat (B.13), using $r _ { N } ^ { \mathrm { b e s t } } \leq 2 \dot { B }$ on the failure event. The constants are uniform over the fixed RKHS ball, proving the worst-case expected bound. The choice of $J _ { N }$ is entirely analytical, and the continuous algorithm does not need to know the budget in order to implement its per-round rule □

## C Proofs for Section 4

We prove the shared trajectory estimates for each exact implementation separately, with $\mathcal { D } _ { N }$ as defined in Section 4.1. For discrete selection, a fresh exact continuous posterior path restricted to the decision grid has exactly the required conditional Gaussian-vector law. Its other values and derivatives are auxiliary proof variables. For continuous selection, the algorithm maximizes the path over [0, 1]. In both cases, the filtration includes the complete past, including any earlier auxiliary paths, and fresh draws retain their stated conditional law given that past. Every posterior uses its own algorithm’s actual queries and observations.

Fixed-kernel identities. For (34), the Fourier convention $\begin{array} { r } { \widehat { g } ( \omega ) = \int g ( x ) e ^ { - i \omega x } } \end{array}$ dx gives

$$
\begin{array} { l } { { \displaystyle \widehat { k } ( \omega ) = \frac { 1 6 } { 3 ( 1 + \omega ^ { 2 } ) ^ { 3 } } , } } \\ { { \displaystyle \| g \| _ { \mathcal { H } _ { k } ( \mathbb { R } ) } ^ { 2 } = \frac { 3 } { 1 6 } \left( \| g \| _ { 2 } ^ { 2 } + 3 \| g ^ { \prime } \| _ { 2 } ^ { 2 } + 3 \| g ^ { \prime \prime } \| _ { 2 } ^ { 2 } + \| g ^ { \prime \prime \prime } \| _ { 2 } ^ { 2 } \right) . } } \end{array}\tag{C.1}
$$

The first identity follows by integrating the three terms of $( 1 + | x | + | x | ^ { 2 } / 3 ) e ^ { - | x | }$ . The second follows by Parseval’s identity and the spectral RKHS norm. Restriction to [0, 1] takes the minimum extension norm. In particular,

$$
\| u ^ { \prime \prime \prime } \| _ { L ^ { 2 } ( 0 , 1 ) } \leq \frac { 4 } { \sqrt { 3 } } \| u \| _ { \mathcal { H } _ { k } } , \qquad \| u ^ { \prime \prime } \| _ { \infty } \leq \| u \| _ { \mathcal { H } _ { k } } .\tag{C.2}
$$

The second bound uses the second-derivative representer and $\partial _ { x } ^ { 2 } \partial _ { y } ^ { 2 } k ( x , y ) | _ { y = x } = 1$

A useful elementary estimate is the following. If $u \in H ^ { 3 } ( [ a , b ] )$ vanishes at three distinct points of that interval, and $L _ { 0 } = b - a . $ , then

$$
\| u ^ { ( r ) } \| _ { \infty , [ a , b ] } \leq L _ { 0 } ^ { 5 / 2 - r } \| u ^ { \prime \prime \prime } \| _ { L ^ { 2 } ( a , b ) } , \qquad r = 0 , 1 , 2 .\tag{C.3}
$$

Indeed, Rolle’s theorem gives two zeros of $u ^ { \prime }$ and one zero of $u ^ { \prime \prime }$ . Integrating $u ^ { \prime \prime \prime }$ from the last zero and applying Cauchy–Schwarz bounds $\| u ^ { \prime \prime } \| _ { \infty }$ by $\sqrt { L _ { 0 } } \lVert u ^ { \prime \prime \prime } \rVert _ { 2 }$ . Two further integrations from zeros of $u ^ { \prime }$ and u prove the other bounds. No lower bound on the spacing of the three zeros is required.

The centered posterior RKHS after any finite history $X$ is

$$
\mathcal { R } _ { X } = \{ u \in \mathcal { H } _ { k } : u | _ { X } = 0 \} = W _ { X } ^ { \perp }
$$

with its inherited norm. Its evaluation representers are $( I - P _ { X } ) k _ { x }$ , and its reproducing kernel is $c _ { X }$ . This also follows directly from Appendix A. Posterior derivative variances are the squared dual norms of derivative evaluations on $\mathcal { R } _ { X }$ Thus projection contraction gives

$$
\begin{array} { r l } & { \operatorname { V a r } ( Z _ { n } ^ { \prime \prime } ( x ) \mid \mathcal { F } _ { n - 1 } ) \leq 1 , } \\ & { \operatorname { V a r } ( Z _ { n } ^ { \prime \prime } ( x ) - Z _ { n } ^ { \prime \prime } ( y ) \mid \mathcal { F } _ { n - 1 } ) \leq C _ { k } | x - y | . } \end{array}\tag{C.4}
$$

For the second inequality, the prior increment variance is $\begin{array} { r } { ( 2 \pi ) ^ { - 1 } \int | e ^ { i \omega x } - e ^ { i \omega y } | ^ { 2 } \omega ^ { 4 } \widehat { k } ( \omega ) } \end{array}$ dω. Splitting at $| \omega | = 1$ and $| \omega | = | x - y | ^ { - 1 }$ , and using $| e ^ { i u } - 1 | \leq \operatorname* { m i n } \{ | u | , 2 \}$ and $\omega ^ { 4 } \widehat { k } ( \omega ) = O ( \omega ^ { - 2 } )$ , bounds this by $C _ { k } | x - y |$

A smooth prior representation and the boundary event. Let $Y = ( W , W ^ { \prime } , W ^ { \prime \prime } )$ be the stationary solution of the spatial linear SDE

$$
\begin{array} { l } { { d Y _ { 0 } = Y _ { 1 } d x , \qquad d Y _ { 1 } = Y _ { 2 } d x , } } \\ { { d Y _ { 2 } = - ( Y _ { 0 } + 3 Y _ { 1 } + 3 Y _ { 2 } ) d x + \displaystyle \frac { 4 } { \sqrt { 3 } } d B _ { x } . } } \end{array}\tag{C.5}
$$

Here $B _ { x }$ is a standard Brownian motion in the spatial variable, not an algorithmic time index. Its stationary initial covariance is

$$
\Sigma = \left( \begin{array} { c c c } { { 1 } } & { { 0 } } & { { - 1 / 3 } } \\ { { 0 } } & { { 1 / 3 } } & { { 0 } } \\ { { - 1 / 3 } } & { { 0 } } & { { 1 } } \end{array} \right) , \qquad \operatorname* { d e t } \Sigma = 8 / 2 7 > 0 .\tag{C.6}
$$

The drift has characteristic polynomial $( s + 1 ) ^ { 3 }$ , and its first-component spectral density is $1 6 / [ 3 ( 1 + \omega ^ { 2 } ) ^ { 3 } ]$ . This verifies the covariance (34). The first component has $C ^ { 2 }$ paths because the third component is continuous. Finite noiseless kernel corrections preserve $C ^ { 2 }$ paths, including for singular observation Gram matrices.

There is a fixed positive probability that $W ^ { \prime } < - 1 / 2$ on $[ 0 , 1 ]$ . One direct verification uses the deterministic state path $( q , q ^ { \prime } , q ^ { \prime \prime } )$ for $q ( x ) = - x$ . Start near $( 0 , - 1 , 0 )$ , which has positive probability under $\left( \mathbf { C } . 6 \right)$ , and require the Brownian path to stay in a sufficiently small uniform neighborhood of the control $b ( 0 ) = 0 , b ^ { \prime } ( x ) = ( \sqrt { 3 } / 4 ) ( - 3 - x )$ . The control has finite Cameron–Martin energy, so this tube has positive probability. This follows from the symmetric-set shift inequality of Van Der Vaart et al. [13, Lemma 5.2] and the positive centered Brownian small-ball probability. The initial state is independent of the future Brownian increments. Continuity of the linear solution map in the initial state and the driving path then gives the claimed derivative inequality. On this event, the unique maximizer over [0, 1], and over every decision grid containing 0, is 0. Consequently,

$$
\mathbb { P } ( \mathcal { E } _ { \partial } ) \ge q _ { \partial } > 0\tag{C.7}
$$

uniformly in $N .$ . Conditioning on this first-round event does not change the prescribed law of any future fresh posterior draw.

## C.1 Lemma 4: Persistent Plateau Overshoot

Proof. All probability bounds below are conditional on $\mathcal { E } _ { \partial }$ unless a full-history conditional probability is displayed. Constants may depend on $k , B , v , \chi .$ , κ and, for discrete selection, the fixed constants in (36), but not on $N ,$ , the particular admissible $\varepsilon _ { N }$ , or the realized history.

Coverage derived from the actual queries. Set

$$
\sigma _ { m } = \operatorname* { m a x } _ { z \in \mathscr { D } _ { N } \cap \mathscr { P } } s _ { m } ( z ) , \qquad \rho _ { m } = \operatorname* { s u p } _ { x \in \mathscr { P } } \mathrm { d i s t } ( x , X _ { m } ) , \qquad \Delta _ { N } = \kappa \varepsilon _ { N } ^ { 2 } .
$$

We first show that there are fixed $C , c > 0$ such that for $2 \leq m \leq N$

$$
\begin{array} { r } { \mathbb { P } _ { f _ { N } } \left( \sigma _ { m } > \Delta _ { N } + C m ^ { - 5 / 2 } \sqrt { \log ( m + 1 ) } \middle | \mathcal { E } _ { \partial } \right) \leq e ^ { - c ( m - 1 ) } . } \end{array}\tag{C.8}
$$

Let $p _ { * } : = \Phi ( - ( B + 1 ) / v ) > 0$ . Before round $n ,$ select a maximizer z of $z _ { n }$ $s _ { n - 1 }$ over $\mathcal { D } _ { N } \cap \mathcal { P }$ , using the finite ordering or a fixed measurable continuous-domain selector. It is measurable with respect to the past, before the current draw. Since $f _ { N } ( z _ { n } ) = 0$ , the interpolation bound gives $\mu _ { n - 1 } ( z _ { n } ) \geq - B \sigma _ { n - 1 }$ . Therefore

$$
\begin{array} { r } { \mathbb { P } _ { f _ { N } } ( G _ { n } ( z _ { n } ) \geq \sigma _ { n - 1 } \mid \mathcal { F } _ { n - 1 } ) \geq p _ { * } . } \end{array}\tag{C.9}
$$

If $\sigma _ { n - 1 } = 0$ , the event holds surely.

Fix m for this argument. Apply the proof of Lemma 3 to one fresh auxiliary path conditional on its complete past, with proof-grid resolution $\lceil m ^ { 7 / \bar { 2 } } \rceil$ , confidence parameter $p _ { * }$ , and $\zeta = 1$ . There are deterministic

$$
u _ { m } \leq C { \sqrt { \log ( m + 1 ) } } , \qquad d _ { m } \leq C m ^ { - 7 / 2 } { \sqrt { \log ( m + 1 ) } }
$$

such that the event $\mathcal V _ { n ; m } = \{ \forall x \in [ 0 , 1 ] : | Z _ { n } ( x ) | \leq u _ { m } s _ { n - 1 } ( x ) + d _ { m } \}$ has conditional failure probability at most $p _ { * } / 2 .$ . For instance, the coarse-grid coefficient is $\sqrt { 2 \log ( 8 ( \lceil m ^ { 7 / 2 } \rceil + 1 ) / p _ { * } ) }$ . The proof is valid conditionally because all increment bounds are pathwise in the past design. This analytical grid is distinct from the actual decision domain and does not constrain either implementation.

Define $Q _ { n ; m } = \mathbf { 1 } \{ G _ { n } ( z _ { n } ) \geq \sigma _ { n - 1 } \} \mathbf { 1 } _ { \mathcal { V } _ { n ; m } }$ . Subtracting the two conditional probabilities gives $\mathbb { E } ( Q _ { n ; m } \mid \mathcal { F } _ { n - 1 } ) \geq$ $p _ { * } / 2 .$ . Independence of these events is unnecessary. Lemma 2, applied to rounds $2 , \ldots , m$ , shows that at least $p _ { * } ( m - 1 ) / 4$ of these indicators equal one, except with probability $e ^ { - p _ { * } ( m - 1 ) / 1 6 }$ . On each such round,

$$
\begin{array} { r l } & { \sigma _ { m } \leq \sigma _ { n - 1 } \leq G _ { n } ( z _ { n } ) \leq G _ { n } ( x _ { n } ) } \\ & { \qquad \leq \Delta _ { N } + ( B + v u _ { m } ) s _ { n - 1 } ( x _ { n } ) + v d _ { m } . } \end{array}\tag{C.10}
$$

The comparison $G _ { n } ( z _ { n } ) \leq G _ { n } ( x _ { n } )$ uses maximization over the actual domain $\mathcal { D } _ { N }$ , so it holds for both implementations. Apply Lemma 1 to this successful subsequence and select its smallest-innovation round. Substitution into (C.10) proves (C.8).

If $x \in \mathcal { P }$ is at distance $r > 0$ from all observations, $\mathbf { u s e } \ z = x$ for the continuous domain. For the discrete domain, choose $z \in \mathcal { P } \cap \mathcal { G } _ { N } ^ { \mathrm { l b } }$ within $\gamma _ { N }$ of $x ,$ and first consider $r \geq 2 \gamma _ { N }$ . In either case, a smooth bump centered at $z ,$ of width a fixed small multiple of min $\{ r , r _ { 0 } \}$ , avoids all observations. Equation (C.1) normalizes its RKHS norm to at most one while retaining its value at least c min $\{ r , r _ { 0 } \} ^ { 5 / 2 }$ . Restriction of a whole-space bump handles the endpoint 1 as well. The residual dual-norm formula bounds $s _ { m } ( z )$ below by this value. In the discrete case with $r < 2 \gamma _ { N }$ , the distance is already bounded by the mesh error. In the continuous case there is no mesh error. Since $\Delta _ { N } \le N ^ { - 5 / 2 } \sqrt { \log N }$ , the variance bound consequently implies that, for fixed $C _ { H } , m _ { c }$ and $m _ { c } \le m \le N$

$$
\mathbb { P } _ { f _ { N } } \left( \rho _ { m } > H _ { m } \mid \mathcal { E } _ { \partial } \right) \le e ^ { - c ( m - 1 ) } , \qquad H _ { m } = C _ { H } m ^ { - 1 } [ \log ( m + 1 ) ] ^ { 1 / 5 } .\tag{C.11}
$$

Here $m ^ { - 5 / 2 } \sqrt { \log ( m + 1 ) }$ is decreasing for m $\geq 2$ . For discrete selection, (36) gives $N \gamma _ { N } = \mathcal { O } ( N ^ { - 1 / 4 } ) \to 0 , \operatorname { s o } 2 \gamma _ { N }$ is absorbed into $H _ { m }$ uniformly for $m _ { c } \le m \le N$ . For continuous selection, no such term occurs. By first choosing $m _ { c }$ large, the fixed cutoff $r _ { 0 }$ in the bump estimate causes no problem.

Usable intervals in an irregular design. Fix a history before round n satisfying $\rho _ { n - 1 } \leq H$ , with $H \leq L _ { P } / 1 6$ and $L _ { P } = 1 / 4$ . Sort its distinct plateau observations as $y _ { 1 } < \cdots < y _ { \ell }$ , and set $h _ { i } = y _ { i + 1 } - y _ { i }$ . Coverage implies

$$
y _ { 1 } \leq 3 / 4 + 2 H , \qquad y _ { \ell } \geq 1 - H , \qquad h _ { i } \leq 2 H , \qquad \sum _ { i } h _ { i } \geq L _ { P } - 3 H .
$$

Call an internal interval good if at least one neighboring interval has length at most $2 h _ { i }$ . Otherwise call it bad. The left neighbor of a bad interval is not bad and has length greater than $2 h _ { i }$ . This is an injective assignment, with only the first interval as a possible boundary exception. If $S _ { G } , S _ { B }$ are the total internal good and bad lengths, then

$$
S _ { G } + S _ { B } \geq L _ { P } - 7 H , \qquad S _ { B } \leq S _ { G } / 2 + H .
$$

Thus $S _ { G } \geq ( 2 / 3 ) ( L _ { P } - 8 H ) \geq L _ { P } / 3$ . Put $a = L _ { P } / 1 2$ and discard good intervals shorter than $a / n$ . There are fewer than n intervals in total, so at most length a is discarded. The retained length is at least $L _ { P } / 4$ . Since each retained interval has length at most 2H, their number K satisfies

$$
K \ge \frac { L _ { P } } { 8 H } , \qquad h _ { i } \ge a / n .\tag{C.12}
$$

Join each retained interval to a neighboring interval of length at most $2 h _ { i }$ . The resulting patch $J _ { i }$ has length at most $3 h _ { i }$ and contains three distinct observed locations. The patches have overlap multiplicity at most three, because each consists of only two adjacent basic intervals.

Normalized posterior covariance. For continuous selection, let $z _ { i }$ be the exact midpoint of each retained interval. For discrete selection, choose $z _ { i } \in \mathcal { G } _ { N } ^ { \mathrm { l b } }$ nearest to that midpoint. Since $h _ { i } \geq a / n$ and ${ \cal N } \gamma _ { N }  0 .$ , in the latter case $\gamma _ { N } / h _ { i } \le N \gamma _ { N } / a \to 0$ uniformly over $n \leq N$ . Thus, in either implementation, $z _ { i } \in \mathcal { D } _ { N } \cap \mathcal { P }$ lies at distance at least $h _ { i } / 4$ from both endpoints for all sufficiently large $N$ . Put $D _ { h } = \mathrm { d i a g } ( h _ { i } ^ { 5 / 2 } )$ . There are fixed constants $0 < c _ { 0 } \le C _ { 0 }$ such that

$$
c _ { 0 } I _ { K } \preceq D _ { h } ^ { - 1 } [ c _ { n - 1 } ( z _ { i } , z _ { j } ) ] _ { i , j = 1 } ^ { K } D _ { h } ^ { - 1 } \preceq C _ { 0 } I _ { K } .\tag{C.13}
$$

To prove the lower bound, take a fixed smooth bump ψ supported in $( - 1 / 4 , 1 / 4 )$ , with $\psi ( 0 ) = 1$ , and, for $a ^ { \prime } \in \mathbb { R } ^ { K }$ define

$$
u _ { a ^ { \prime } } ( x ) = \sum _ { i } a _ { i } ^ { \prime } h _ { i } ^ { 5 / 2 } \psi ( ( x - z _ { i } ) / h _ { i } ) .
$$

The supports are disjoint and avoid every observation. Scaling in (C.1) gives $\| u _ { a ^ { \prime } } \| _ { \mathcal { H } _ { k } } ^ { 2 } \le C _ { \psi } \| a ^ { \prime } \| _ { 2 } ^ { 2 }$ and $u _ { a ^ { \prime } } ( z _ { i } ) / h _ { i } ^ { 5 / 2 } =$ $a _ { i } ^ { \prime } .$ For the normalized evaluation operator $T : u \in \mathcal { R } _ { X _ { n - 1 } } \mapsto ( u ( z _ { i } ) / h _ { i } ^ { 5 / 2 } ) \colon$ , the middle matrix in (C.13) is $T T ^ { * }$ Using $u _ { a ^ { \prime } }$ in the dual norm of $T ^ { * } a ^ { \prime }$ yields $\| T ^ { * } a ^ { \prime } \| ^ { 2 } \geq C _ { \psi } ^ { - 1 } \| a ^ { \prime } \| _ { 2 } ^ { 2 } .$

For the upper bound, each residual function u has three zeros in $J _ { i } .$ . Equations (C.3) and (C.2) give

$$
\sum _ { i } { \frac { | u ( z _ { i } ) | ^ { 2 } } { h _ { i } ^ { 5 } } } \leq C \sum _ { i } \int _ { J _ { i } } | u ^ { \prime \prime \prime } | ^ { 2 } \leq C ^ { \prime } \| u ^ { \prime \prime \prime } \| _ { 2 } ^ { 2 } \leq C _ { 0 } \| u \| _ { \mathcal { H } _ { k } } ^ { 2 } .\tag{C.14}
$$

This bounds $\| T \| ^ { 2 }$ and proves the other side of (C.13). Also, $\mu _ { n - 1 }$ vanishes at every observed plateau point because it interpolates $f _ { N } = 0$ there. Apply the same three-zero estimate and $\| \mu _ { n - 1 } \| _ { \mathcal { H } _ { k } } \leq \dot { B }$ to obtain

$$
| \mu _ { n - 1 } ( z _ { i } ) | / h _ { i } ^ { 5 / 2 } \leq C _ { \mu } B .\tag{C.15}
$$

No posterior coordinates have been assumed independent, and no lower bound on the ratio of neighboring gap lengths is used.

A summable single-round failure probability. On a history satisfying (C.11) with $m = n - 1$ , equations (C.12)– (C.13) give $K \ge c n / [ \log ( n + 1 ) ] ^ { \bar { 1 } / 5 } \ge n ^ { 1 / 2 }$ for sufficiently large n, and $K \leq n$ . Under the conditional law, decompose

$$
\begin{array} { r } { ( Z _ { n } ( z _ { i } ) / h _ { i } ^ { 5 / 2 } ) _ { i = 1 } ^ { K } \stackrel { d } { = } \sqrt { c _ { 0 } } \xi + Y , \quad \xi \sim \mathcal { N } ( 0 , I _ { K } ) , \quad 0 \preceq \mathrm { C o v } ( Y ) \preceq C _ { 0 } I _ { K } , } \end{array}
$$

with ξ and Y conditionally independent. Diagonalizing $\operatorname { C o v } ( Y )$ gives $\mathbb { E } \exp ( \| Y \| _ { 2 } ^ { 2 } / ( 4 C _ { 0 } ) ) \le 2 ^ { K / 2 }$ and hence $\mathbb { P } ( \| Y \| _ { 2 } ^ { 2 } > 4 C _ { 0 } K ) \le e ^ { - K / 2 }$ . On the complementary event, at least $K / 2$ coordinates satisfy $Y _ { i } \ge - \sqrt { 8 C _ { 0 } }$ . On those coordinates,

$$
G _ { n } ( z _ { i } ) \geq v \sqrt { c _ { 0 } } h _ { i } ^ { 5 / 2 } ( \xi _ { i } - A _ { 0 } ) , \qquad A _ { 0 } = \frac { C _ { \mu } B / v + \sqrt { 8 C _ { 0 } } } { \sqrt { c _ { 0 } } } .
$$

Choose $b _ { * } = \operatorname* { m i n } \{ 1 , v { \sqrt { c _ { 0 } } } a ^ { 5 / 2 } / 8 \}$ . For large $n , \sqrt { \log K } - A _ { 0 } \geq \sqrt { \log n } / 8$ . Conditional on $Y ,$ any one of these retained coordinates exceeding $\sqrt { \log K }$ gives $G _ { n } ( z _ { i } ) > b _ { * } n ^ { - 5 / 2 } \sqrt { \log n }$ . The scalar Gaussian lower-tail estimate $1 - \Phi ( u ) \geq c u ^ { - 1 } e ^ { - u ^ { 2 } / 2 }$ for u $\iota \geq 1$ therefore gives

$$
\begin{array} { r l } {  { \mathbb { P } _ { f _ { N } } ( \operatorname* { m a x } _ { z \in \mathscr { D } _ { N } \cap \mathscr { P } } G _ { n } ( z ) \leq b _ { * } n ^ { - 5 / 2 } \sqrt { \log n } \bigg | \mathcal { F } _ { n - 1 } ) } } \\ & { \leq e ^ { - K / 2 } + \Phi ( \sqrt { \log K } ) ^ { K / 2 } } \\ & { \leq e ^ { - n ^ { 1 / 2 } / 2 } + \exp ( - c _ { 1 } \frac { n ^ { 1 / 4 } } { \sqrt { \log n } } ) = : q _ { n } . } \end{array}\tag{C.16}
$$

For completeness, the scalar tail estimate follows by integrating the normal density on $[ u , u + 1 / u ]$ . In particular, $\textstyle \sum _ { n } q _ { n } < \infty$

First exclude coverage failures using (C.11). Then integrate (C.16) on each past-measurable coverage event. A union bound yields

$$
\mathbb { P } _ { f _ { N } } \left( \exists n \in \{ m _ { * } + 1 , \dots , N \} : \operatorname* { m a x } _ { z \in \mathcal { D } _ { N } \cap \mathbb { P } } G _ { n } ( z ) \leq b _ { * } n ^ { - 5 / 2 } \sqrt { \log n } \bigg | \mathcal { E } _ { \partial } \right) \leq \sum _ { m \geq m _ { * } } e ^ { - c ( m - 1 ) } + \sum _ { n > m _ { * } } q _ { n } .
$$

Choosing a fixed sufficiently large m<sub>∗</sub> makes the right side at most $1 / 8 .$ . At no point was a conditional Gaussian law used after conditioning on a future good event. This proves (39). □

## C.2 Lemma 5: Local Observations and Curvature

Proof. We first establish a quantitative prior small-ball bound, then use it to control actual local visits. These events are analytical tools, so the algorithm does not test them or reject samples.

A quantitative small-ball bound. For the prior W in (C.5), there are fixed $C _ { \mathrm { s b } } , r _ { 0 } > 0$ such that

$$
\begin{array} { r } { \mathbb { P } ( \| W \| _ { \infty , [ 0 , 1 ] } \le r ) \ge \exp ( - C _ { \mathrm { s b } } r ^ { - 2 / 5 } ) , \qquad 0 < r \le r _ { 0 } . } \end{array}\tag{C.17}
$$

Here is a direct verification. On a spatial block of length $\lambda \leq 1$ , rescale time to $u \in [ 0 , 1 ]$ and state to $R _ { 0 } = \lambda ^ { - 5 / 2 } Y _ { 0 }$ $R _ { 1 } = \lambda ^ { - 3 / 2 } Y _ { 1 } , R _ { 2 } = \lambda ^ { - 1 / 2 } Y _ { 2 }$ . The equation becomes

$$
d { \cal R } _ { 0 } = { \cal R } _ { 1 } d u , d { \cal R } _ { 1 } = { \cal R } _ { 2 } d u , d { \cal R } _ { 2 } = - ( \lambda ^ { 3 } { \cal R } _ { 0 } + 3 \lambda ^ { 2 } { \cal R } _ { 1 } + 3 \lambda { \cal R } _ { 2 } ) d u + \frac { 4 } { \sqrt { 3 } } d \widetilde { \cal B } _ { u } .\tag{C.18}
$$

Uniformly over $R ( 0 ) \in [ - 1 , 1 ] ^ { 3 }$ and $0 < \lambda \leq 1$ , there are fixed $p _ { 0 } > 0 , M \ge 1$ such that, with probability at least $p _ { 0 }$ the block satisfies $| \dot { R } _ { 0 } ( u ) | \leq \dot { M }$ throughout and $R ( 1 ) \in [ - 1 , 1 ] ^ { 3 }$ . To verify this uniform statement, choose a quintic Hermite polynomial q with $( q , q ^ { \prime } , q ^ { \prime \prime } ) ( 0 ) = R ( 0 )$ and $( q , q ^ { \dot { \prime } } , q ^ { \prime \prime } ) ( \dot { 1 } ) = 0$ . Its derivatives through order three are bounded uniformly on the compact initial-state set. The control

$$
b ^ { \prime } ( u ) = \frac { \sqrt { 3 } } { 4 } \big ( q ^ { \prime \prime \prime } ( u ) + 3 \lambda q ^ { \prime \prime } ( u ) + 3 \lambda ^ { 2 } q ^ { \prime } ( u ) + \lambda ^ { 3 } q ( u ) \big ) , \qquad b ( 0 ) = 0 ,
$$

has uniformly bounded energy and makes $( q , q ^ { \prime } , q ^ { \prime \prime } )$ the controlled trajectory. If $A _ { \lambda }$ is the drift matrix in (C.18) and $e = ( 0 , 0 , 4 / \sqrt { 3 } ) ^ { \top }$ , integration by parts shows that the trajectory difference driven by $d = \widetilde { B } - b$ is

$$
e d ( u ) + \int _ { 0 } ^ { u } A _ { \lambda } e ^ { A _ { \lambda } ( u - s ) } e d ( s ) d s .
$$

Its uniform norm is bounded by a fixed multiple of $\| d \| _ { \infty } ,$ , because $A _ { \lambda }$ is uniformly bounded. A fixed small driving tube therefore gives the asserted block event. The symmetric-set Gaussian shift inequality yields

$$
\begin{array} { r } { \mathbb { P } ( \| \widetilde { B } - b \| _ { \infty } \le \eta ) \ge e ^ { - \| b ^ { \prime } \| _ { 2 } ^ { 2 } / 2 } \mathbb { P } ( \| \widetilde { B } \| _ { \infty } \le \eta ) \ge p _ { 0 } > 0 . } \end{array}
$$

The centered probability is positive for fixed $\eta > 0$ . We also refer readers to see Van Der Vaart et al. [13, Section 5] for the Gaussian support and shift facts.

By (C.6), the probability that the initial state lies in $| Y _ { 0 } | \le \lambda ^ { 5 / 2 } , | Y _ { 1 } | \le \lambda ^ { 3 / 2 } , | Y _ { 2 } | \le \lambda ^ { 1 / 2 }$ is at least $c \lambda ^ { 9 / 2 }$ . Iterating the block event using independent Brownian increments gives

$$
\begin{array} { r } { \mathbb { P } ( \| W \| _ { \infty } \le M \lambda ^ { 5 / 2 } ) \ge c \lambda ^ { 9 / 2 } p _ { 0 } ^ { \lceil 1 / \lambda \rceil } . } \end{array}
$$

The final block may extend beyond 1 before restricting back to [0, 1]. Set $\lambda = ( r / M ) ^ { 2 / 5 }$ and absorb the polynomial prefactor into the exponential to obtain (C.17).

A local visit from an arbitrary actual history. Let $0 < h < 1 / 8$ , with $\varepsilon _ { N } = o ( h )$ and $f _ { N } ^ { * } \leq h ^ { 2 } / 2$ . In the discrete case additionally require $\gamma _ { N } \leq h / 1 2 8$ . Suppose the current history contains at most one distinct point in $( 0 , h )$ . Choose three disjoint intervals of length a fixed multiple of h inside $( h / 8 , 7 h / 8 )$ . At least one contains no observation. For continuous selection take its midpoint z, while for discrete selection take a grid point z bounded away from its endpoints by a fixed multiple of $h .$ Center a smooth nonnegative bump $u _ { h }$ at z, with support inside that empty interval and height $( \dot { \kappa } + 2 ) h ^ { 2 }$ . Then

$$
u _ { h } | _ { X _ { n - 1 } } = 0 , \qquad \| u _ { h } \| _ { \mathcal { H } _ { k } } ^ { 2 } \le C h ^ { - 1 } , \qquad ( f _ { N } + u _ { h } ) ( z ) \ge 2 h ^ { 2 } .\tag{C.19}
$$

The norm bound follows from (C.1). Outside the bump support, $f _ { N } + u _ { h } = f _ { N } \leq h ^ { 2 } / 2$ . Thus the tube event

$$
\| G _ { n } - ( f _ { N } + u _ { h } ) \| _ { \infty , \mathcal { D } _ { N } } \leq h ^ { 2 } / 4
$$

forces every maximizer over $\mathcal { D } _ { N }$ into that empty interval and makes the next selected point a new local query.

The shift $a _ { X } = ( f _ { N } - \mu _ { n - 1 } + u _ { h } ) / v$ belongs to $\mathcal { R } _ { X _ { n - } }$ and satisfies $\| a _ { X } \| _ { \mathcal { H } _ { L } } ^ { 2 } \le C ( 1 + h ^ { - 1 } )$ . For any finite set $Q \subseteq { \mathcal { D } } _ { N }$ , the restriction of $a x$ belongs to the Cameron–Martin space of the posterior vector, with norm no larger than its residual RKHS norm. Apply the symmetric-set shift inequality to that vector, then Anderson’s inequality to its centered box. Since $k ( Q , Q ) { \stackrel { - } { - } } { \dot { c } } _ { n - 1 } ( { \dot { Q } } , Q )$ is positive semidefinite, the posterior centered-box probability is at least the corresponding prior-vector probability, which is at least the full-prior small-ball probability. For the finite implementation take $\overset { \cdot } { Q } = \overset { \cdot } { \mathcal { G } } _ { N } ^ { \mathrm { l b } }$ . For the continuous implementation take an increasing sequence of finite sets with dense union in [0, 1]. Continuity of the sampled path and of $f _ { N } + u _ { h }$ identifies the intersection of their tube events with the supremum-norm tube, so continuity of probability from above preserves the same lower bound. These Gaussian inequalities also apply to singular covariance matrices on their supports [13]. Hence

P<sub>f</sub> (next query is a new point in $( 0 , h ) \mid { \mathcal { F } } _ { n - 1 } )$

$$
\begin{array} { r l } & { \geq e ^ { - C ( 1 + h ^ { - 1 } ) } \mathbb { P } ( \| W \| _ { \infty } \leq h ^ { 2 } / ( 4 v ) ) } \\ & { \geq \exp [ - C _ { 1 } ( 1 + h ^ { - 1 } ) - C _ { 2 } h ^ { - 4 / 5 } ] \geq e ^ { - C _ { \mathrm { v i s i t } } / h } . } \end{array}\tag{C.20}
$$

All constants are uniform over the actual finite history, regardless of the number of observations outside $( 0 , h )$ . The bump and shift depend only on the past, which is legitimate inside a conditional probability calculation.

Choose a fixed $A _ { \mathrm { l o c } } \geq \operatorname* { m a x } \{ 1 6 C _ { \mathrm { v i s i t } } , 1 \}$ and define $h _ { N } , m _ { N }$ as in (40). The assumed bound on ε<sub>N</sub> ensures $\varepsilon _ { N } = o ( h _ { N } )$ and $f _ { N } ^ { * } \le h _ { N } ^ { 2 } / 2$ for large $N$ . In the discrete case the resolution condition gives $\gamma _ { N } / h _ { N } = \mathcal { O } ( N ^ { - 5 / 4 } \log N ) \to 0 ,$ so $\gamma _ { N } \leq h _ { N } / 1 2 8$ for all sufficiently large N. Continuous selection requires no mesh condition. Until two distinct local observations have been obtained, the next round has conditional success probability at least $N ^ { - 1 / 1 6 }$ . After two have been obtained, set the counting indicator to one. On $\{ \tau _ { N } ^ { \mathrm { l o c } } > m _ { N } \}$ , the sum of these indicators over rounds $2 , \ldots , m _ { N }$ is at most one. For large N, this is smaller than $N ^ { - 1 / 1 6 } ( m _ { N } - 1 ) / 2$ . Lemma 2 therefore gives

$$
\begin{array} { r } { \mathbb { P } _ { f _ { N } } \mathopen { } \mathclose \bgroup \left( \tau _ { N } ^ { \mathrm { l o c } } > m _ { N } \mid \mathcal { E } _ { \partial } \aftergroup \egroup \right) \leq \exp [ - N ^ { - 1 / 1 6 } \mathopen { } \mathclose \bgroup \left( m _ { N } - 1 \aftergroup \egroup \right) / 8 ] \leq e ^ { - c N ^ { 3 / 1 6 } } , } \end{array}
$$

proving (41).

Global and local derivative control. Equation (C.4) and a scalar Gaussian dyadic argument give, for $0 < \delta < 1 / 2$

$$
\begin{array} { r } { \mathbb { P } \left( \| Z _ { n } ^ { \prime \prime } \| _ { \infty } > C ( 1 + \sqrt { \log ( 1 / \delta ) } ) \Big | \mathcal { F } _ { n - 1 } \right) \leq \delta . } \end{array}\tag{C.21}
$$

For clarity, bound the value at 0 by a Gaussian tail. At refinement level j, there are at most $2 ^ { j } + 1$ dyadic nodes and each parent increment has standard deviation at most $C 2 ^ { - j / 2 }$ . Use thresholds proportional to $2 ^ { - j / 2 } \sqrt { \log ( C / \delta ) + 2 j \log 2 }$ Their failure probabilities sum to at most $\delta ,$ , and their thresholds sum to $C ^ { \prime } ( 1 + \sqrt { \log ( 1 / \delta ) } )$ . Continuity extends the bound from the countable nodes to the whole interval. This is the same telescoping argument as Appendix B.5, applied to the derivative process without normalization by posterior standard deviation. Taking $\delta = N ^ { \bar { - } \bar { 3 } }$ , summing over at most N rounds, and using $\| \mu _ { n - 1 } ^ { \prime \prime } \| _ { \infty } \leq B$ proves (42) for a fixed $C _ { g }$

Once the history contains 0 and two distinct points of $( 0 , h _ { N } )$ , every $u \in { \mathcal { R } } _ { X _ { n - 1 } }$ has three zeros in $[ 0 , h _ { N } ]$ . Therefore

$$
\| u ^ { \prime \prime } \| _ { \infty , [ 0 , h _ { N } ] } \leq C \sqrt { h _ { N } } \| u \| _ { \mathcal { H } _ { k } } , \qquad \operatorname* { s u p } _ { x \in [ 0 , h _ { N } ] } \mathrm { V a r } ( Z _ { n } ^ { \prime \prime } ( x ) \mid \mathcal { F } _ { n - 1 } ) \leq C ^ { 2 } h _ { N } .
$$

After rescaling $x = h _ { N } u$ and the process by $h _ { N } ^ { - 1 / 2 }$ , the pointwise variances and increment bounds are uniform on $[ 0 , 1 ]$ The preceding dyadic argument consequently gives

$$
\| Z _ { n } ^ { \prime \prime } \| _ { \infty , [ 0 , h _ { N } ] } \leq C \sqrt { h _ { N } } ( 1 + \sqrt { \log ( 1 / \delta ) } )
$$

with conditional failure probability at most $\delta$ whenever the past has the three required observations. Set $\delta = N ^ { - 3 }$ and sum over $m _ { N } < n \le N$ . Since $h _ { N } \log N = A _ { \mathrm { l o c } }$ , the amplitude is bounded by a fixed constant. Choose $L \geq \operatorname* { m a x } \{ 1 , \kappa \}$ large enough also to dominate $B + C ^ { \prime } v \sqrt { A _ { \mathrm { l o c } } }$ . For $n > m _ { N }$ , the event $\{ \tau _ { N } ^ { \mathrm { l o c } } \leq m _ { N } \}$ is already past-measurable, so integrating the conditional tail on this event proves (43). No future hitting event is used to change a Gaussian conditional law. □

## C.3 Theorem 3: Discrete Matérn Logarithmic Lower Bound

Proof. First fix the kernel, $B , v , \chi , \kappa$ . Lemma 4 supplies $b _ { * } \le 1 , m _ { * }$ . Lemma 5 supplies fixed $A _ { \mathrm { l o c } } , C _ { g } , L$ , with $L \geq \operatorname* { m a x } \{ 1 , \kappa \}$ . Only then choose the budget-dependent $\varepsilon _ { N }$ and $f _ { N }$ by (44). This order prevents any dependence of the fixed constants on the eventual random trajectory or on $N$ . For large $N , \| f _ { N } \| _ { \mathcal { H } _ { k } } \le \hat { B / 2 } , \Delta _ { N } = \bar { \kappa } \varepsilon _ { N } ^ { 2 } \overset { \le } { \le } b _ { N } / 1 6 \leq$ $N ^ { - 5 / 2 } \sqrt { \log N }$ , and

$$
\begin{array} { r l } & { \varepsilon _ { N } \asymp N ^ { - 5 / 4 } ( \log N ) ^ { 1 / 4 } , } \\ & { \frac { \gamma _ { N } } { \varepsilon _ { N } } = \mathcal { O } ( ( \log N ) ^ { - 1 / 4 } ) \to 0 , \qquad \frac { \varepsilon _ { N } } { h _ { N } } \to 0 . } \end{array}\tag{C.22}
$$

The resolution condition also gives ${ \cal N } \gamma _ { N }  0 { . }$ , as used in the plateau proof, and $\gamma _ { N } / h _ { N }  0 .$ , as used in the local-visit proof. Thus both preceding lemmas apply. The quadratic identity $f _ { N } ( \bar { x } ) = \kappa [ \varepsilon _ { N } ^ { 2 } - ( x - \varepsilon _ { N } ) ^ { 2 } ]$ on $[ 0 , 1 / 8 ]$ , and $f _ { N } \leq 0$ outside its positive excursion, prove the near-optimal-set assertion following (45).

Discrete optimality gives a derivative comparison. Let $g \in C ^ { 2 } ( [ 0 , 1 ] )$ and let an interior grid point x maximize its grid values. Both $x - \gamma _ { N }$ and $x + \gamma _ { N }$ are grid points. Taylor’s formula applied in the two directions yields

$$
| g ^ { \prime } ( x ) | \leq \frac { K \gamma _ { N } } { 2 }\tag{C.23}
$$

whenever $| g ^ { \prime \prime } | \le K$ between the neighboring points. I $\textrm { f } g ( 0 ) = 0$ and the same bound holds on $[ 0 , x ]$ , Taylor’s formula between x and 0 then gives

$$
g ( x ) \leq | g ^ { \prime } ( x ) | x + \frac { K } { 2 } x ^ { 2 } \leq \frac { K } { 2 } ( x ^ { 2 } + \gamma _ { N } x ) .
$$

Neither calculation assumes that x is a maximizer over the continuum.

(C.24)

The initial fixed-length prefix. We must control rounds $2 , \ldots , m _ { * }$ , before the plateau estimate is used. Conditional on $\mathcal { E } _ { \partial }$ , we show

$$
\mathbb { P } _ { f _ { N } } ( \exists 2 \leq n \leq m _ { * } : x _ { n } \in { \cal A } _ { N } \mid \mathcal { E } _ { \partial } ) = o ( 1 ) .\tag{C.25}
$$

Suppose the past has no positive observation below R, but includes 0. Let $u _ { R } ( x ) = x \vartheta ( x / R )$ , where ϑ is a fixed smooth cutoff supported in $( - 1 , 1 )$ and equal to one near zero. Then $u _ { R }$ vanishes at every observation, $u _ { R } ^ { \prime } ( 0 ) = 1$ , and (C.1) gives $\| u _ { R } \| _ { \mathcal { H } _ { k } } \le C R ^ { - 3 / 2 }$ . The derivative residual dual norm therefore implies

$$
\operatorname { s d } ( Z _ { n } ^ { \prime } ( 0 ) \mid { \mathcal { F } } _ { n - 1 } ) \geq c R ^ { 3 / 2 } .\tag{C.26}
$$

This remains true with repeated observations at zero.

For $1 \leq n \leq m _ { * }$ , set $a _ { n } = 2 ^ { n - m _ { * } - 2 }$ and $r _ { n } = N ^ { - a _ { n } }$ . Then $a _ { n } = 2 a _ { n - 1 }$ and $r _ { m _ { * } } = N ^ { - 1 / 4 }$ . After the first query at zero, there is no positive observation below r . Consider the first round $n \in \{ 2 , \ldots , m _ { * } \}$ that violates $x _ { n } \notin ( 0 , r _ { n } )$ , and intersect with the global curvature event in (42). All preceding positive observations are at least $r _ { n - 1 }$ . Equation (C.23) and the global sampled-path curvature bound imply

$$
| { G } _ { n } ^ { \prime } ( 0 ) | \le C _ { g } \sqrt { \log N } ( r _ { n } + \gamma _ { N } / 2 ) \le 2 C _ { g } r _ { n } \sqrt { \log N }
$$

for large N, since $r _ { n } \ge r _ { m _ { * } } = N ^ { - 1 / 4 }$ and $\gamma _ { N } / r _ { m _ { * } } = \mathcal { O } ( N ^ { - 1 } )$ . By (C.26), the conditional Gaussian density of $G _ { n } ^ { \prime } ( 0 )$ is at most $C r _ { n - 1 } ^ { - 3 / 2 }$ , uniformly in its conditional mean. Integrating that density over the displayed interval bounds the probability of this first violation, intersected with the curvature event, by

$$
C \sqrt { \log N } r _ { n } r _ { n - 1 } ^ { - 3 / 2 } = C \sqrt { \log N } N ^ { - a _ { n - 1 } / 2 } = o ( 1 ) .\tag{C.27}
$$

We use event inclusion here, not a Gaussian law conditioned on the curvature event. Summing over the fixed number of rounds and adding its $N ^ { - 2 }$ failure probability proves that, with conditional probability $1 - o ( \bar { 1 } )$ ), every positive query in the prefix is at least $N ^ { - 1 / 4 }$ . By (C.22), $3 \varepsilon _ { N } / 2 = o ( N ^ { - 1 / 4 } )$ ), proving (C.25).

Intersecting the trajectory events. Write $b _ { n } = b _ { * } n ^ { - 5 / 2 } { \sqrt { \log n } }$ for $n \geq 2$ . Conditional on $\mathcal { E } _ { \partial }$ , intersect the following: the prefix avoidance event, the plateau event of Lemma $^ { 4 , }$ the global curvature event, $\tau _ { N } ^ { \mathrm { l o c } } \leq m _ { N }$ , and the subsequent local curvature event. Their total failure probability is at most

$$
o ( 1 ) + \frac { 1 } { 8 } + 2 N ^ { - 2 } + e ^ { - c N ^ { 3 / 1 6 } } < \frac { 1 } { 2 }
$$

for sufficiently large N. Independence is not required.

For $m _ { * } < n \le m _ { N }$ , suppose $x _ { n } \in A _ { N }$ . The global curvature bound and (C.24) imply

$$
G _ { n } ( x _ { n } ) \leq C _ { g } \sqrt { \log N } \left( \frac { 9 } { 8 } \varepsilon _ { N } ^ { 2 } + \frac { 3 } { 4 } \gamma _ { N } \varepsilon _ { N } \right) .
$$

The plateau sampled maximum is larger than $b _ { n } \geq b _ { m _ { N } }$ , whereas

$$
{ \frac { \sqrt { \log N } ( \varepsilon _ { N } ^ { 2 } + \gamma _ { N } \varepsilon _ { N } ) } { b _ { m _ { N } } } } = O ( N ^ { - 1 5 / 8 } \sqrt { \log N } ) \to 0 .
$$

For large N, the plateau grid therefore contains a strictly better sampled value, contradicting grid maximality.

For $m _ { N } < n \leq N ,$ , a point in $\mathcal { A } _ { N }$ and its neighboring grid points lie inside $[ 0 , h _ { N } ]$ . The local curvature bound instead gives

$$
\begin{array} { c l c r } { { G _ { n } ( x _ { n } ) \leq L \left( \displaystyle \frac { 9 } { 8 } \varepsilon _ { N } ^ { 2 } + \displaystyle \frac { 3 } { 4 } \gamma _ { N } \varepsilon _ { N } \right) } } \\ { { = b _ { N } \left( \displaystyle \frac { 9 } { 1 2 8 } + \displaystyle \frac { 3 \gamma _ { N } } { 6 4 \varepsilon _ { N } } \right) < b _ { N } \leq b _ { n } . } } \end{array}\tag{C.28}
$$

This again contradicts the plateau event. The function $n ^ { - 5 / 2 } { \sqrt { \log n } }$ is decreasing for n $\geq 2 .$ , justifying both comparisons of $b _ { n }$ . The prefix has already been excluded, and $x _ { 1 } = 0 \notin \mathcal { A } _ { N }$ . Thus all N queries avoid $\mathcal { A } _ { N }$ on the intersection, and

$$
r _ { N } ^ { \mathrm { b e s t } } \geq \frac { \kappa \varepsilon _ { N } ^ { 2 } } { 4 } = \frac { \kappa b _ { * } } { 6 4 L } N ^ { - 5 / 2 } \sqrt { \log N } .
$$

Using (C.7) proves (46), with $c _ { \mathrm { l b } } = \kappa b _ { * } / ( 6 4 L )$ and $q _ { \mathrm { l b } } = q _ { \partial } / 2$ . It yields the lower-bound direction of (47) upon taking expectations.

Quadratic grid bias and matching upper bound. We verify the discretization statement uniformly over the whole fixed RKHS ball, not only on the constructed objectives. By (C.2), every such f satisfies $\| f ^ { \prime \prime } \| _ { \infty } \leq \dot { B }$ . If a boundary point is a global maximizer, it belongs to the grid and the bias is zero. Otherwise, take an interior maximizer $x ^ { * }$ and a nearest grid point z, so $f ^ { \prime } ( x ^ { * } ) = 0$ and $| x ^ { * } - z | \le \gamma _ { N } / 2$ . Taylor’s formula gives

$$
0 \le b _ { { \mathcal G } _ { N } ^ { \mathrm { l b } } } ( f ) \le f ( x ^ { * } ) - f ( z ) \le \frac { B } { 2 } | x ^ { * } - z | ^ { 2 } \le \frac { B } { 8 } \gamma _ { N } ^ { 2 } .\tag{C.29}
$$

This proves (37). In particular, the bias is $\mathcal { O } ( N ^ { - 5 / 2 } ) = o ( N ^ { - 5 / 2 } \sqrt { \log N } )$ under (36).

The grid has $M _ { N } = J _ { N } ^ { \mathrm { l b } } + 1$ points, with log $M _ { N } = \mathcal { O } ( \log N )$ . Apply Theorem 1 with exact maximization and $\delta _ { N } = N ^ { - 7 / 2 }$ . Its budget condition holds for all sufficiently large N, and

$$
u _ { N , M _ { N } , \delta _ { N } } ^ { 2 } = 2 \log ( 4 N M _ { N } / \delta _ { N } ) = \mathcal { O } ( \log N ) .
$$

Using (C.29) and the same expectation argument as in Corollary 1 yields

$$
\begin{array} { l } { { \displaystyle \operatorname* { s u p } _ { \| f \| _ { \varkappa _ { k } } \leq B } \mathbb { E } _ { f } [ r _ { N } ^ { \mathrm { b e s t } } ] \leq \frac { B } { 8 } \gamma _ { N } ^ { 2 } + C N ^ { - 5 / 2 } \sqrt { \log N } + 2 B N ^ { - 7 / 2 } } } \\ { ~ } \\ { { \displaystyle = \mathcal { O } ( N ^ { - 5 / 2 } \sqrt { \log N } ) } . } \end{array}
$$

This proves (47) for every fixed regular-grid sequence in the stated resolution regime. The polynomial-cardinality condition is used in this matching upper bound. The lower-bound trajectory estimates use the mesh comparisons established above. □

## C.4 Theorem 4: Continuous Matérn Logarithmic Lower Bound

Proof. All samples, posterior quantities, and events in this proof refer to the actual continuous-domain algorithm. The shared proofs in Appendices C.1 and C.2 apply to this trajectory: the variance comparator is chosen over the continuous plateau, comparison coordinates are exact interval midpoints, and the local-visit event is a supremum-norm tube over [0, 1]. Neither a discrete trajectory nor a limit of grid maximizers is used.

Parameters and objective. Fix $k , B , v , \chi , \kappa ,$ then choose the constants $b _ { * } , m _ { * } , A _ { \mathrm { l o c } } , C _ { q } , L$ from the continuous versions of the two lemmas. Increase $m _ { * }$ to at least 2 if necessary and retain $b _ { * } \leq 1 , L \geq \operatorname* { m a x } \{ 1 , \kappa \}$ . Only then set

$$
b _ { N } = b _ { * } N ^ { - 5 / 2 } \sqrt { \log N } , \qquad \varepsilon _ { N } ^ { 2 } = b _ { N } / ( 1 6 L ) , \qquad f _ { N } ( x ) = \kappa \chi ( x ) ( 2 \varepsilon _ { N } x - x ^ { 2 } ) .
$$

For large $N , \| f _ { N } \| _ { \mathcal { H } _ { k } } \leq B / 2 , f _ { N } ( 0 ) = 0 .$ , and

$$
f _ { N } ^ { * } = \Delta _ { N } = \kappa \varepsilon _ { N } ^ { 2 } \le b _ { N } / 1 6 , \qquad \varepsilon _ { N } \asymp N ^ { - 5 / 4 } ( \log N ) ^ { 1 / 4 } , \qquad \varepsilon _ { N } / h _ { N } \to 0 .
$$

The maximizer is $\varepsilon _ { N }$ . Outside $\mathcal { A } _ { N } = ( \varepsilon _ { N } / 2 , 3 \varepsilon _ { N } / 2 )$ , the quadratic identity near zero and the nonpositive values outside the positive excursion give

$$
f _ { N } ^ { * } - f _ { N } ( x ) \geq \kappa \varepsilon _ { N } ^ { 2 } / 4 .\tag{C.30}
$$

The boundary event has probability at least $q _ { \partial } > 0$ by (C.7). Conditional on this event, $G _ { n } ( 0 ) = 0$ for all $n \geq 2$

Initial fixed-length prefix. We first show, for this continuous trajectory, that

$$
\mathbb { P } _ { f _ { N } } ( \exists 2 \leq n \leq m _ { * } : x _ { n } \in { \cal A } _ { N } \mid \mathcal { E } _ { \partial } ) = o ( 1 ) .\tag{C.31}
$$

If the past includes zero but no positive location below R, the same residual test function $u _ { R } ( x ) = x \vartheta ( x / R )$ used in (C.26) gives

$$
\operatorname { s d } ( Z _ { n } ^ { \prime } ( 0 ) \mid { \mathcal { F } } _ { n - 1 } ) \geq c R ^ { 3 / 2 } .
$$

This statement depends only on the actual observed locations and also holds when zero has been queried repeatedly. For $1 \leq n \leq m _ { * }$ , take $a _ { n } = 2 ^ { n - m _ { * } - 2 }$ and $r _ { n } = N ^ { - a _ { n } }$ . Consider the first round $n \geq 2$ with $0 < x _ { n } < r _ { n }$ . Before that round, every positive observation is at least $r _ { n - 1 }$ . On the global curvature event of Lemma $5 ,$ interior maximality gives $G _ { n } ^ { \prime } ( x _ { n } ) = \mathrm { \bar { 0 } }$ and therefore

$$
| G _ { n } ^ { \prime } ( 0 ) | \leq C _ { g } r _ { n } { \sqrt { \log N } } .
$$

Given the past, the Gaussian density of $G _ { n } ^ { \prime } ( 0 )$ is bounded by $C r _ { n - 1 } ^ { - 3 / 2 }$ , uniformly in its mean. The event of this first violation intersected with the curvature event is thus contained in an interval event with conditional probability at most

$$
C \sqrt { \log N } r _ { n } r _ { n - 1 } ^ { - 3 / 2 } = C \sqrt { \log N } N ^ { - a _ { n - 1 } / 2 } = o ( 1 ) .
$$

We bound this interval under the original conditional Gaussian law. We do not first condition on a future curvature event. Sum over the fixed number of rounds and add the $N ^ { - 2 }$ curvature-failure probability. With conditional probability $1 - o ( 1 )$ , every positive query in this prefix is at least $r _ { m _ { * } } = N ^ { - 1 / 4 }$ . Since $3 \bar { \varepsilon } _ { N } / 2 = \overset { \cdot } { o } \overset { \cdot } { N } ^ { - 1 / 4 } )$ , this proves (C.31).

Joint trajectory event. Write $b _ { n } = b _ { * } n ^ { - 5 / 2 } \sqrt { \log n }$ for $n \geq 2$ . Intersect the prefix-avoidance event with the continuous plateau event in Lemma 4, the global curvature event, the event $\tau _ { N } ^ { \mathrm { l o c } } \bar { \leq } m _ { N }$ , and the subsequent local curvature event. Conditional on $\mathcal { E } _ { \partial }$ , their total failure probability is bounded by

$$
o ( 1 ) + \frac { 1 } { 8 } + 2 N ^ { - 2 } + e ^ { - c N ^ { 3 / 1 6 } } < \frac { 1 } { 2 }
$$

for sufficiently large N. No independence between these events is required.

Intermediate and later rounds. Work on this intersection. Suppose first that $m _ { * } < n \le m _ { N }$ and $x _ { n } \in A _ { N }$ . The point $x _ { n }$ is interior, so $G _ { n } ^ { \prime } ( x _ { n } ) = 0$ . Taylor’s formula from $x _ { n }$ to the observed anchor zero gives

$$
| G _ { n } ( x _ { n } ) | \leq \frac { C _ { g } \sqrt { \log N } } { 2 } x _ { n } ^ { 2 } \leq \frac { 9 } { 8 } C _ { g } \sqrt { \log N } \varepsilon _ { N } ^ { 2 } .
$$

The plateau maximum exceeds $b _ { n } \geq b _ { m _ { N } }$ , while

$$
{ \frac { \sqrt { \log N } \varepsilon _ { N } ^ { 2 } } { b _ { m _ { N } } } } = O ( N ^ { - 1 5 / 8 } \sqrt { \log N } ) \to 0 .
$$

This contradicts global maximality for all sufficiently large N.

For m $\smash { p < n \leq N }$ , the entire segment $[ 0 , x _ { n } ]$ is contained in $[ 0 , h _ { N } ]$ whenever $x _ { n } \in \mathcal { A } _ { N }$ . The local curvature bound instead gives

$$
| G _ { n } ( x _ { n } ) | \leq { \frac { L } { 2 } } x _ { n } ^ { 2 } \leq { \frac { 9 } { 8 } } L \varepsilon _ { N } ^ { 2 } = { \frac { 9 } { 1 2 8 } } b _ { N } < b _ { N } \leq b _ { n } < \operatorname* { m a x } _ { x \in \mathcal { P } } G _ { n } ( x ) .\tag{C.32}
$$

This again contradicts maximality. The monotonicity of $n ^ { - 5 / 2 } { \sqrt { \log n } }$ for $n \geq 2$ justifies both comparisons with $b _ { n }$

Regret and expectation. The first query is zero on $\mathcal { E } _ { \partial }$ , the fixed prefix has been excluded, and the two comparisons exclude every remaining query from $\mathcal { A } _ { N }$ . Equation (C.30) therefore yields

$$
r _ { N } ^ { \mathrm { b e s t } } \geq \frac { \kappa \varepsilon _ { N } ^ { 2 } } { 4 } = \frac { \kappa b _ { * } } { 6 4 L } N ^ { - 5 / 2 } \sqrt { \log N }
$$

on an event of unconditional probability at least $q _ { \partial } / 2 .$ . This proves (49), with $c _ { \mathrm { l b } } ^ { \mathrm { c } } = \kappa b _ { * } / ( 6 4 L )$ and $q _ { \mathrm { l b } } ^ { \mathrm { c } } = q _ { \partial } / 2$ . Taking expectations gives the lower-bound direction of (50). Corollary 2, specialized to the same fixed kernel, supplies its upper-bound direction. There is no decision-grid bias in this comparison. □

## D Proofs for Section 5

All posterior quantities and conditional probabilities in this appendix refer to the query locations and exact observations generated by GP-CTS. The algorithm uses the same conditional Gaussian sampling law as GP-TS before applying the cap. Lemmas 1 and 2 therefore apply to its own trajectory.

## D.1 Lemma 6: Optimism Preservation and Positive Sample Deviations

Proof. Fix a realized history. By (7) and ${ \overline { { B } } } \geq B$

$$
U _ { n - 1 } ( x ) = \mu _ { n - 1 } ( x ) + { \overline { { B } } } s _ { n - 1 } ( x ) \geq f ( x ) \qquad { \mathrm { f o r ~ e v e r y ~ } } x \in { \mathcal { D } } .
$$

For real numbers $g , u , a$ with $u \geq a ,$ the inequality min $\{ g , u \} \ge a$ holds if and only if $g \geq a$ . Substituting $g = G _ { n } ( x ^ { \dagger } )$ $u = U _ { n - 1 } ( x ^ { \dagger } )$ , and $a = f ( x ^ { \dagger } )$ proves (55). The equality of events does not require independence or a Gaussian assumption.

For every $x \in \mathcal { D }$

$$
\begin{array} { r l } {  { H _ { n } ( x ) - f ( x ) \le U _ { n - 1 } ( x ) - f ( x ) } } \\ & { = \mu _ { n - 1 } ( x ) - f ( x ) + \overline { B } s _ { n - 1 } ( x ) } \\ & { \le ( B + \overline { B } ) s _ { n - 1 } ( x ) . } \end{array}\tag{D.1}
$$

This proves (56) for every sample realization. Under exact Gaussian sampling, the calculation in (B.7) and the event identity yield

$$
\begin{array} { r } { \mathbb { P } _ { f } ( H _ { n } ( x ^ { \dagger } ) \geq f ( x ^ { \dagger } ) \mid \mathcal { F } _ { n - 1 } ) \geq p = \Phi ( - B / v ) . } \end{array}
$$

If the comparator has zero posterior variance, exact interpolation gives $G _ { n } ( x ^ { \dagger } ) = H _ { n } ( x ^ { \dagger } ) = f ( x ^ { \dagger } )$ almost surely. The deviation bound remains one-sided, and no control of the negative tail is needed. □

## D.2 Theorem 5: Simple Regret Bound for GP-CTS

Proof. Fix a deterministic comparator $x ^ { \dagger } \in \arg \operatorname* { m a x } _ { x \in \mathcal { D } } f ( x )$ , using a fixed rule if necessary. Such a maximizer exists because D is either finite or equal to the compact domain X, and f is continuous. Define the random index set

$$
I _ { N } = \{ n \leq N : G _ { n } ( x ^ { \dagger } ) \geq f ( x ^ { \dagger } ) \} .
$$

The comparator is fixed for the given objective and decision domain. Conditional on the history generated by GP-CTS, fresh exact Gaussian sampling and the interpolation inequality give the same optimism probability as (B.7). Lemma 2 therefore implies

$$
\mathbb { P } _ { f } ( | I _ { N } | < p N / 2 ) \le e ^ { - p N / 8 } .\tag{D.2}
$$

On the complementary event, $I _ { N }$ is nonempty. For every $n \in { \cal I } _ { N }$ , Lemma $^ 6$ and the selection rule (54) give

$$
\begin{array} { r l } { f _ { \mathcal { D } } ^ { * } - f ( x _ { n } ) \le H _ { n } ( x ^ { \dagger } ) - f ( x _ { n } ) } & { } \\ { \le H _ { n } ( x _ { n } ) - f ( x _ { n } ) + \eta _ { n } } & { } \\ { \le ( B + \overline { { B } } ) s _ { n - 1 } ( x _ { n } ) + \eta _ { n } . } \end{array}\tag{D.3}
$$

The last inequality holds simultaneously at all points for every sample realization, so it can be applied directly at the sample-dependent query $x _ { n }$

Lemma 1 supplies an index $n _ { \mathrm { o } } \in I _ { N }$ such that

$$
s _ { n _ { 0 } - 1 } ( x _ { n _ { 0 } } ) \leq C _ { \mathrm { s e p } } | I _ { N } | ^ { - \alpha } \leq C _ { \mathrm { s e p } } ( p N / 2 ) ^ { - \alpha } .
$$

That lemma holds for every realized query sequence and every subset of its indices. Its use does not require $I _ { N }$ to be independent of the samples or the queries.

Using the best queried point and adding the decision-domain bias gives

$$
\begin{array} { r l } & { r _ { N } ^ { \mathrm { b e s t } } \leq b _ { \mathcal { D } } + f _ { \mathcal { D } } ^ { * } - f ( x _ { n _ { \circ } } ) } \\ & { \qquad \leq b _ { \mathcal { D } } + ( B + \overline { { B } } ) C _ { \mathrm { s e p } } ( p N / 2 ) ^ { - \alpha } + \underset { 1 \leq n \leq N } { \operatorname* { m a x } } \eta _ { n } } \\ & { \qquad = b _ { \mathcal { D } } + C _ { \mathrm { c a p } } N ^ { - \alpha } + \underset { 1 \leq n \leq N } { \operatorname* { m a x } } \eta _ { n } . } \end{array}\tag{D.4}
$$

Together with (D.2), this proves (59) for every $N \geq 1$ . The only excluded event is that there are too few optimistic rounds. No Gaussian envelope event is required in either decision domain.

For the expectation, the deterministic bound $0 \leq r _ { N } ^ { \mathrm { b e s t } } \leq 2 B$ implies

$$
\mathbb { E } _ { f } [ r _ { N } ^ { \mathrm { b e s t } } ] \leq b _ { \mathcal { D } } ( f ) + C _ { \mathrm { c a p } } N ^ { - \alpha } + \operatorname* { m a x } _ { 1 \leq n \leq N } \eta _ { n } + 2 B e ^ { - p N / 8 } .
$$

The constants, cap input, and tolerance schedule are fixed uniformly over the radius-B RKHS ball. Taking the supremum over that ball and using (57) proves (60). This step does not require a single probability event that holds simultaneously for every objective. □

## D.3 Corollary 3: Matérn Minimax Convergence Rate

Proof. We first specialize the upper bound to the discrete domain and then to the continuous domain. We finally apply the classical minimax lower bound already established in Appendix B.4.

Discrete implementation. Proposition 1 gives $\alpha = \nu / d .$ For $\mathcal { D } = \mathcal { G } _ { N }$ from (25), the uniform bias calculation in Appendix B.4 yields

$$
\bar { b } \varsigma _ { N } \leq B L _ { k } N ^ { - \nu / d } .
$$

With exact maximization, Theorem 5 therefore gives

$$
\operatorname* { s u p } _ { \| f \| _ { \mathcal { H } _ { k } } \leq B } \mathbb { E } _ { f } [ r _ { N } ^ { \mathrm { b e s t } } ] \leq ( B L _ { k } + C _ { \mathrm { c a p } } ) N ^ { - \nu / d } + 2 B e ^ { - p N / 8 } = O ( N ^ { - \nu / d } ) .\tag{D.5}
$$

Since $p > 0$ is fixed, the exponentially decreasing term is bounded by a constant multiple of $N ^ { - \nu / d }$ . For any other deterministic sequence of finite decision sets with $\bar { b } _ { \mathcal { D } } = O ( N ^ { - \nu / d } )$ , the same substitution proves the asserted extension. No bound on the cardinality is used in this regret argument.

Continuous implementation. For $\mathcal { D } = \mathcal { X }$ , we have $\bar { b } _ { \mathcal { D } } = 0$ . The continuous sample-path requirement follows from Proposition 1 and the construction in Appendix A. With exact maximization, Theorem 5 gives

$$
\operatorname* { s u p } _ { \| f \| _ { \mathcal { H } _ { k } } \leq B } \mathbb { E } _ { f } [ r _ { N } ^ { \mathrm { b e s t } } ] \leq C _ { \mathrm { c a p } } N ^ { - \nu / d } + 2 B e ^ { - p N / 8 } = O ( N ^ { - \nu / d } ) .\tag{D.6}
$$

There is no discretization bias or chaining remainder in this bound.

Minimax comparison. The classical lower bound (B.16) applies to every randomized sequential algorithm with N noiseless queries and a data-measurable final recommendation. Its proof allows knowledge of the function-class radius and arbitrary query locations. It therefore also applies to GP-CTS with a fixed valid cap input B, in both the discrete and continuous implementations. Fixing B uniformly over the RKHS ball does not reveal which objective in the lower bound construction is used.

The recommendation returned by GP-CTS is the best queried point. Thus (B.16) supplies a lower bound of order $N ^ { - \nu / d }$ on its worst-case expected simple regret. Combining this with (D.5) and (D.6) proves minimax optimality in order over the fixed Matérn RKHS ball. This is the same benchmark as in Bull [1]. The argument does not imply minimax optimality for a general kernel satisfying only the upper bound on Kolmogorov widths. □