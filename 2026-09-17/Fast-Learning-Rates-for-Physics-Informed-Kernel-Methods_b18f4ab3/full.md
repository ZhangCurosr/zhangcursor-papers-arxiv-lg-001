# Fast Learning Rates for Physics-Informed Kernel Methods

Luc Brogat-Motte<sup>1</sup>, Joachim Bona-Pellissier<sup>2</sup>, Giacomo Meanti<sup>2</sup>, Lorenzo Rosasco<sup>1,2</sup>

<sup>1</sup>SAIL Unit, Istituto Italiano di Tecnologia, Genoa, Italy <sup>2</sup>MaLGa Center, DIBRIS, Università degli Studi di Genova, Genoa, Italy luc.brogatmotte@iit.it, joachim.bona@edu.unige.it, giacomo.meanti@edu.unige.it, lorenzo.rosasco@unige.it

## Abstract

In physics-informed machine learning, a target function u<sup>∗</sup> is learned from noisy value observations $y _ { i } = u ^ { * } ( x _ { i } ) + \varepsilon _ { i } ,$ together with diferential information, given either by noisy observations $d _ { j } = ( D u ^ { * } ) ( z _ { j } ) + \xi _ { j }$ or by a known physical constraint $D u ^ { * } = v$ . We consider the setting where D is a linear diferential operator and analyze a physics-informed kernel estimator uˆ combining n value observations and m diferential observations. In this context, we ask how much can diferential information improve predictions, and how does this improvement depend quantitatively on n, m, and D. We prove finite-sample bounds, supported by numerical simulations, revealing a two-regime structure for the prediction error. When m is limited, the rate depends jointly on n and m; when m exceeds a problem-dependent threshold, the rate saturates and matches the oracle rate obtained when the perfect constraint $D \hat { u } = D u ^ { * }$ is imposed. Examples are discussed for Sobolev spaces which are reproducing kernel Hilbert spaces and include partial Laplacian constraints on the torus and gradient observations on bounded domains. These examples illustrate the range of possible learning rate improvements – from the standard nonparametric $n ^ { - 1 / 4 }$ to the parametric rate $n ^ { - 1 / 2 }$ . Finally, we derive physically consistent rates in a stronger norm that jointly controls the errors in uˆ and Duˆ.

## 1 Introduction

Supervised learning algorithms seek to estimate an unknown target function from noisy observations of its values. In many applications arising in physics, engineering, and scientific computing, however, one has access not only to value observations, but also to physical information involving derivatives or other diferential quantities associated with the target function [1]. This additional information is often specified through functional constraints, such as gradients, Partial Diferential Equation (PDE) residuals, or conservation laws [2–4]. Concrete examples include learning interatomic potentials, where energies and forces of atomic systems provide coupled value and gradient information through $F = - \nabla E [ 5 , 6 ] ;$ ; 3D reconstruction problems, where surface normals give diferential information about the underlying shape [7, 8]; or computational cardiology, where physical models involve diferential constraints from fluid dynamics [9, 10].

In this paper we investigate physics-informed machine learning from the point of view of statistical learning theory. While there exists a comprehensive theory describing the statistical performance of learning from value observations [11, 12], much less is known about how diferential information afects learning rates and sample complexity. We are particularly interested in quantitative results of how diferential information improves prediction: what is the improvement it can provide? how many diferential observations are needed for this gain to appear?

We study this question for target functions in a reproducing kernel Hilbert space (RKHS) H, which makes precise analyses feasible. We assume that there exists a target function $u ^ { \ast } \in \mathcal { H }$ and that, in addition to noisy value observation

$$
y _ { i } = u ^ { * } ( x _ { i } ) + \varepsilon _ { i } , \qquad i = 1 , \dots , n ,
$$

also noisy diferential quantities are available

$$
d _ { j } = ( D u ^ { * } ) ( z _ { j } ) + \xi _ { j } , \qquad j = 1 , \ldots , m ,
$$

where D is a linear diferential operator.

Our contributions are as follows:

• We derive finite-sample bounds that quantitatively describe how the prediction error depends on sample sizes $n , m$ , diferential operator D, and regularization parameters $\lambda , \gamma$

• We introduce a novel capacity assumption for physics-informed learning. This assumption decomposes the learning problem into a component that can be learned from diferential data and a residual component not seen by D. It then quantifies how fast the first component can be learned from physics data and how much complexity remains in the residual one. We illustrate both quantities with two interpretable Sobolev RKHS examples, showing how they depend on the choice of D.

• Under this assumption, we obtain learning rates by deriving the best choice of λ and $\gamma .$ . The rates exhibit two regimes: a diferential-limited regime, where the error decreases jointly with n and m, and a saturation regime, where the rate matches the one obtained when the physical constraint is known exactly. In the Sobolev space examples, the saturated prediction rates range from $n ^ { - 1 / 4 }$ to the parametric rate $n ^ { - 1 / \hat { 2 } }$ . Simulations further illustrate the predicted improvements and two-regime behavior.

• Finally, we show that the same estimator is physically consistent beyond value prediction. We prove learning rates in a stronger norm that jointly controls the errors in uˆ and Duˆ, ruling out accurate value prediction without learning the diferential structure.

## 1.1 Related work

Settings. Learning from both function values and derivatives (or, more generally, diferential operators) is classically known as Hermite-Birkhof interpolation, whose study in a statistical setting dates back at least to the seminal work of Kimeldorf and Wahba [13]. This setting has recently resurfaced in machine learning under the names Sobolev training [14] and physics-informed learning [15]. Incorporating diferential information, whether in discrete or continuous form, can serve various purposes. Diferential operators can be used to regularize the learning procedure: penalties of the form $\textstyle \int ( D u ( x ) ) ^ { 2 } .$ dx have been well studied for spline smoothing [16], inverse problems [17, 18], and machine learning [19]. Diferential operators can also carry information about the data distribution.

This is the case in semi-supervised learning, where the operator D is learned from unlabeled data [20– 24]. Finally, diferential constraints can come from the physical knowledge of the object of study itself, as is often the case in scientific applications [1]. Typically taking the form of a PDE, such constraints provide complementary data, either on the same domain as the value measurements [25] or on a diferent domain. A classical example of the latter is the one where $D u ^ { * }$ is known inside a domain Ω while $u ^ { * }$ is known only on the boundary $\partial \Omega .$ , which is well studied in the PDE literature [26].

Methods. Deep learning-based algorithms to tackle problems with diferential constraints include the Deep Ritz Method [27], neural operators [28, 29], Universal Diferential Equations [4]. Closest to our approach are Physics-Informed Neural Networks (PINNs) [2], which minimize a loss function similar to ours. Kernel methods have also been considered to tackle the problems described above, from splines [13] to RBF collocation [30–33], semi-supervised learning [22] and, more recently, in scientific machine learning applications [34–38]. Our estimator belongs to this kernel-based family: it combines the usual kernel ridge loss on values with an empirical loss on diferential observations $D u ( z _ { j } )$ . An orthogonal class of methods that is commonly used to estimate PDE solutions are finite elements/volumes [39, 25].

Theory. Many convergence results exist for learning PDE solutions with kernels [40, 33, 41, 42] or neural networks [43–46]. In the settings considered, data observations are only available on the boundary of the input domain (and not inside the domain). Such settings difer from ours and are closer to classical numerical methods for PDEs such as finite diferences or finite element methods. Instead, we consider settings in which the value and diferential observations lie in the same domain, and discuss the existing results for such settings, which are commonly of two types. A first type is concerned with physical consistency: showing that a physics-informed estimator uˆ converges to the target $u ^ { * }$ in a physically consistent, stronger than $L ^ { 2 }$ sense [47–49]. In Section $3 . 4 ,$ we provide a result of this nature with learning rates in a norm which controls both uˆ and Duˆ. In particular, Doumèche et al. [47] shows, for a broad class of diferential operators that PINNs are physically consistent. For kernel estimators, Shi et al. [48] and ul Abdeen et al. [49] show that learning with gradients allows to obtain convergence rates in the $H ^ { 1 }$ norm. A second type of result focuses on how the physics-informed penalty benefits the $L ^ { 2 }$ convergence rates, which we show in Section 3.3. Previously, Fisher et al. [50] studied the impact of gradient information on $L ^ { 2 }$ performance using a random feature model, and highlighted the existence of a regime where incorporating gradient information may hurt the prediction accuracy. Arnone et al. [51] investigated a regression estimator regularized with a 2-dimensional elliptic PDE, when $u ^ { * }$ belongs to the Sobolev space $H ^ { 2 }$ . Their estimator converges at least at the standard rate for $H ^ { 2 }$ functions $( n ^ { - 2 / 3 } )$ , and can reach the faster $n ^ { - 4 / 5 }$ rate when the PDE constraint is satisfied by the target. Doumèche et al. [37] reformulated empirical risk minimization with linear diferential constraints as kernel regression in a physics-informed RKHS, yielding theoretical evidence that physical constraints can improve convergence rates, characterized through the efective dimension of the new kernel. Such improvements are illustrated for the setting of $D = \partial _ { x }$ with homogeneous constraints $( D u ^ { * } = 0 )$ Subsequent works [52, 53] focus on approximations and fast implementations, while preserving the underlying convergence rates. Overall, these theoretical results either study an ideal estimator for which continuous constraints are enforced (which typically is not computable in closed form) [51, 37], or tie the number of value and diferential data points (m = n) [48–50]. In contrast, we provide results for any linear D which hold for arbitrary finite values of m and $n ,$ explicitly dealing with approximately enforced constraints.

Finally, we must mention works which characterized the statistical performance of kernel regression [11, 54, 55], which we extend to handle physical constraints. They introduce the key assumptions (source and capacity conditions) under which rates can be derived and the notion of efective dimension which controls the dificulty of the learning problem.

## 2 Physics-informed kernel regression

## 2.1 Problem setup

We now give a formal description of the learning problem. Let $\mathcal { X }$ be a measurable space and let $u ^ { * } : \mathcal { X }  \mathbb { R }$ denote the target function. We will consider linear diferential operators D of order $s \geq 1$ , which take the form

$$
D = \sum _ { | \alpha | \leq s } c _ { \alpha } \partial ^ { \alpha } ,
$$

with coeficients $c _ { \alpha } \in \mathbb { R }$ . We will assume (in (A2)) that $D u ^ { * }$ is well defined. Let $\rho$ and $\rho _ { D }$ be distributions over X. We consider value and diferential observations

$$
\begin{array} { l l } { { y _ { i } = u ^ { * } ( x _ { i } ) + \varepsilon _ { i } , } } & { { x _ { i } \stackrel { i . i . d . } { \sim } \rho , ~ i = 1 , \ldots , n , } } \\ { { d _ { j } = ( D u ^ { * } ) ( z _ { j } ) + \xi _ { j } , } } & { { z _ { j } \stackrel { i . i . d . } { \sim } \rho _ { D } , ~ j = 1 , \ldots , m , } } \end{array}
$$

where $( \varepsilon _ { i } ) _ { i }$ and $( \xi _ { j } ) _ { j }$ are real-valued noise variables. In general, $\rho$ and $\rho _ { D }$ can be diferent, reflecting the fact that value and diferential information may be available on diferent regions of the domain.

## 2.2 Physics informed kernel estimator

Let $( \mathcal { H } , \langle \cdot , \cdot \rangle _ { \mathcal { H } } )$ be an RKHS with associated kernel $k : \mathcal { X } \times \mathcal { X } \to \mathbb { R }$ . Given parameters $\lambda > 0$ and $\gamma > 0$ , the regularized empirical risk is defined, for $u \in \mathcal { H } .$ , as

$$
\widehat { R } ( u ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( u ( x _ { i } ) - y _ { i } \right) ^ { 2 } + \gamma \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \left( D u ( z _ { j } ) - d _ { j } \right) ^ { 2 } + \lambda \| u \| _ { \mathcal { H } } ^ { 2 } ,\tag{1}
$$

and the Physics Informed Kernel MethodS (PIKS) [38] estimator is the result of an optimization procedure in H

$$
\widehat { u } : = \operatorname * { a r g m i n } _ { u \in \mathcal { H } } \widehat { R } ( u ) .
$$

A small deviation from the estimator defined in [38] is the introduction of the scaling parameter $\gamma .$ The parameter λ controls the regularity of the estimator, while γ controls the relative weight assigned to derivative observations. Setting $\gamma = 0$ recovers standard kernel ridge regression with value observations.

Since $\lambda > 0 , \tilde { R }$ is a strictly convex quadratic functional on H, hence the minimizer uˆ exists and is unique. Moreover, the estimator admits a finite-dimensional representer form. In particular,

$$
\hat { u } ( \cdot ) = \sum _ { i = 1 } ^ { n } \alpha _ { i } k ( x _ { i } , \cdot ) + \sum _ { j = 1 } ^ { m } \beta _ { j } D _ { 1 } k ( z _ { j } , \cdot ) ,
$$

where the coeficients $\alpha \in \mathbb { R } ^ { n }$ and $\beta \in \mathbb { R } ^ { m }$ are the solutions to a linear system involving the value, mixed, and diferential Gram matrices. The explicit system is given in Appendix A.1.

## 3 Main results

In this section, we present learning rates for PIKS and quantify the impact of derivative information. After listing the assumptions in Section 3.1, we state a general finite sample result in Section 3.2, from which, in Section 3.3, we derive learning rates, emphasizing the existence of two regimes. In Section 3.4 we study the rates in a stronger norm involving both u and Du.

## 3.1 Assumptions

Assumptions (A1) to (A5) are standard and ensure that derivative evaluations are well defined in the RKHS and that both value and derivative observations are generated by the same underlying target function. The main novel assumption is (A6), which quantifies how much of the function space is visible through derivative information.

Assumption (A1) (Linear diferential operator). We consider linear diferential operators D of the form $\begin{array} { r } { D = \sum _ { | \alpha | \leq s } c _ { \alpha } \partial ^ { \alpha } } \end{array}$ for some integer $s \geq 1$ and coeficients $c _ { \alpha } \in \mathbb { R }$

Typical examples include directional derivatives, Laplacians, and more general linear partial diferential operators. The order s determines the degree of smoothness required in the kernel such that derivative evaluations are well defined.

Assumption (A2) (Kernel regularity). We assume that the kernel k is of class $C ^ { s , s }$ on $\mathcal { X } \times \mathcal { X }$ meaning that all mixed derivatives $\partial _ { x } ^ { \alpha } \partial _ { x ^ { \prime } } ^ { \beta } k ( x , x ^ { \prime } )$ with $| \alpha | , | \beta | \leq s$ exist and are continuous.

Under assumptions (A1) and (A2), the diferential operator D can be applied to the kernel and gives rise to well-defined derivative representers in the RKHS. More precisely, let $\phi ( x ) : = k ( x , \cdot )$ be the RKHS feature map. Since derivative evaluations are continuous linear functionals on H [56, Corollary 4.36], and defining $D \phi ( x ) : = D k ( x , \cdot )$ , where $D$ acts on the first argument of $k ,$ one has $D \phi ( x ) \in \mathcal { H }$ and the reproducing property holds on both the kernel and its derivatives:

$$
u ( x ) = \langle u , \phi ( x ) \rangle _ { \mathcal { H } } , \qquad D u ( z ) = \langle u , D \phi ( z ) \rangle _ { \mathcal { H } } .
$$

Assumption (A3) (Boundedness of features and outputs). There exist constants $\kappa _ { V } , \kappa _ { D } , L _ { V } , L _ { D } >$ 0 such that, almost surely,

$$
\| \phi ( \boldsymbol { x } ) \| _ { \mathcal { H } } \le \kappa _ { V } , \qquad \| D \phi ( \boldsymbol { z } ) \| _ { \mathcal { H } } \le \kappa _ { D } , \qquad | \boldsymbol { y } | \le L _ { V } , \qquad | d | \le L _ { D } .
$$

Such boundedness assumptions are standard in statistical analyses of kernel methods and are satisfied under assumptions (A1) and (A2) when X is compact.

Assumption (A4) (Attainability). We assume that the regression function $u ^ { * } ( x ) : = \mathbb { E } [ y \mid x ]$ belongs to H. Equivalently, there exists $w ^ { \ast } \in \mathcal { H }$ such that

$$
\boldsymbol { u } ^ { * } ( \boldsymbol { x } ) = \langle \boldsymbol { w } ^ { * } , \phi ( \boldsymbol { x } ) \rangle _ { \mathcal { H } } .
$$

This is a standard assumption for well-specified problems: it states that the target function belongs to the RKHS induced by the kernel, so it can be represented by the model class under consideration.

Assumption (A5) (Consistency of derivative observations). We assume that the derivative observations are unbiased measurements of the derivative of the regression function, in the sense that

$$
\mathbb { E } [ d \mid z ] = D u ^ { * } ( z ) .
$$

This assumption states that, when $D u ^ { * }$ is known, it can be interpreted as having a physical constraint which is well-specified. It is also referred to as using a perfect measure $\rho _ { D }$ [48].

To quantify both the efective dimension of the hypothesis space and its alignment with value and derivative observations, we introduce the following covariance operators

$$
\Sigma : = \mathbb { E } [ \phi ( x ) \otimes \phi ( x ) ] , \qquad \Sigma _ { D } : = \mathbb { E } [ D \phi ( z ) \otimes D \phi ( z ) ] ,
$$

where, for $a , b \in { \mathcal { H } }$ , the rank-one operator $a \otimes b : \mathcal { H } \to \mathcal { H }$ is defined as $( a \otimes b ) w : = \langle w , b \rangle _ { \mathcal { H } } a$ . For self-adjoint operators $A , B ,$ , we write $A \preceq B$ when $B - A$ is positive semi-definite.

Assumption (A6) (Value-derivative capacity decomposition). Assume that there exist positive semi-definite operators $\Sigma _ { 1 } , \Sigma _ { 2 } \succeq 0$ such that

$$
\Sigma = \Sigma _ { 1 } + \Sigma _ { 2 } ,
$$

and that there exist constants $c _ { 1 } , c _ { 2 } > 0$ and $\alpha , r \in [ 0 , 1 ]$ such that, for all $\lambda \in ( 0 , 1 ]$

$$
\begin{array} { r } { \mathrm { T r } \big ( \Sigma _ { 1 } ( \Sigma _ { 1 } + \lambda I ) ^ { - 1 } \big ) \ \leq \ c _ { 1 } \lambda ^ { - \alpha } , \qquad \mathrm { T r } \big ( ( \Sigma _ { D } + \lambda I ) ^ { - 1 } \Sigma _ { 2 } \big ) \ \leq \ c _ { 2 } \lambda ^ { - r } . } \end{array}\tag{2}
$$

This assumption decomposes the geometry of function values into directions that are invisible, or only weakly visible, to the operator D (captured by $\Sigma _ { 1 } )$ and directions that are detectable through diferential observations (captured by $\Sigma _ { 2 } )$ . The exponent α governs the efective dimension of the invisible component, while r measures how well the detectable component is separated from the null-space of $D .$ . In the limiting case $\Sigma _ { 1 } = \Sigma$ and $\Sigma _ { 2 } = 0$ , the assumption reduces to the usual efective-dimension condition of standard kernel regression. Together, these conditions formalize how diferential information can reduce the statistical complexity of learning function values via capacity reduction, depending on interactions between kernel, diferential operator, and data distribution.

## 3.2 Finite-sample bounds

Theorem 3.1 (High-probability finite-sample bounds for PIKS). Under assumptions $( A I )$ to (A5), let $\lambda > 0 , \gamma > 0$ and let $\delta \in ( 0 , 1 / 2 )$ . Define

$$
\Sigma _ { \gamma } : = \Sigma + \gamma \Sigma _ { D } , \qquad \mathcal { B } ( \lambda , \gamma ) : = \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } w ^ { * } \big \| _ { \mathcal { H } } ,
$$

and the (value and derivative) capacity terms

$$
\mathcal { N } _ { V } ( \lambda , \gamma ) : = \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } \Sigma ^ { 1 / 2 } \big \| _ { \mathrm { H S } } , \qquad \mathcal { N } _ { D } ( \lambda , \gamma ) : = \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } \Sigma _ { D } ^ { 1 / 2 } \big \| _ { \mathrm { H S } } .
$$

If

$$
\frac { 3 6 } { n } \log \frac { 2 n } { \delta } \leq \lambda \leq \| \Sigma \| _ { \infty } a n d \frac { 3 6 } { m } \log \frac { 2 m } { \delta } \leq \lambda \gamma ^ { - 1 } \leq \| \Sigma _ { D } \| _ { \infty } ,\tag{3}
$$

then there exists a constant $c > 0$ , depending only on $\kappa _ { V } , \kappa _ { D } , L _ { V } , L _ { D }$ and $\| u ^ { * } \| _ { \mathcal { H } }$ , but not on $\lambda , \gamma , n , m , \delta$ , such that, with probability at least $1 - 2 \delta$

$$
\mathbb { E } \left[ \left( \hat { u } ( x ) - u ^ { * } ( x ) \right) ^ { 2 } \right] ^ { \frac { 1 } { 2 } } \leq c \log \frac { 4 } { \delta } \left[ \frac { 1 } { \sqrt { \lambda } } \left( \frac { 1 } { n } + \frac { \gamma } { m } \right) + \frac { {  { \mathcal N } } _ { V } ( \lambda , \gamma ) } { \sqrt { n } } + \gamma \frac { {  { \mathcal N } } _ { D } ( \lambda , \gamma ) } { \sqrt { m } } + \lambda B ( \lambda , \gamma ) \right] .\tag{4}
$$

![](images/f93da6bbc100750e2ce3566ed5662851310284962dd81560397cdcb702710769.jpg)  
Figure 1: $( l e f t )$ Rate-regime changes in the $( \alpha , r )$ plane. On the lower-right of each contour line the problem parameters are such that m is larger than the critical threshold. (right) Log-linear plot of the expected error as a function of n and α for fixed $m , r .$ . For smaller α a larger $m / n$ ratio is required to be in the fast saturated regime.

Sketch of the proof. The proof stems from decomposing the error into separate approximation and estimation terms, and controlling the stochastic parts through concentration inequalities for covariance operators in Hilbert spaces. The main novelty compared to the classical kernel ridge regression analysis lies in handling of the derivative observations. We introduce the covariance operator $\Sigma _ { \gamma } = \Sigma + \gamma \Sigma _ { D }$ , and decompose all deviation terms into value and derivative contributions while carefully tracking their dependence on $\gamma , n ,$ , and $m$ . Once these additional decompositions are established, the remainder of the argument closely parallels the standard KRR proof [11]. □

The bound in Theorem 3.1 separates the contributions of value observations, derivative observations, and regularization bias. The quantities $\mathcal { N } _ { V } ( \lambda , \gamma )$ and $\mathcal { N } _ { D } ( \lambda , \gamma )$ play the role of efective dimensions associated with the value and derivative components of the problem. They control the corresponding variance contributions, while the bias term $\lambda B ( \lambda , \gamma )$ captures the approximation error induced by regularization. The contributions of $\mathcal { N } _ { V } ( \lambda , \gamma )$ and $\mathcal { N } _ { D } ( \lambda , \gamma )$ interact through the shared parameters $\lambda$ and $\gamma .$ . Increasing $\gamma$ places more weight on the derivative constraints, which reduces the efective complexity of the value component by shrinking the set of admissible functions in directions that are observable through $D _ { \mathbf { \lambda } }$ , but also amplifies the variance contribution coming from noisy derivative observations. Setting $\gamma = 0$ recovers the standard KRR bound [11]. Setting $m = \infty$ instead recovers the rates of Doumèche et al. [37].

## 3.3 Learning rate acceleration from diferential information

We now instantiate the finite-sample bound under assumption (A6) and optimize over the regularization parameters λ and $\gamma$ to precisely characterize the learning rates of the PIKS estimator.

Corollary 3.1 (Learning rates for PIKS). Under assumptions $( A I )$ to $( A \theta )$ , there exist choices of regularization parameters $\lambda = \lambda ( n , m )$ and $\gamma = \gamma ( n , m )$ , given explicitly in the proof and which

depend polynomially on $n , ~ m$ , log n, and log m, such that, with high probability,

$$
\begin{array} { r } { \mathbb { E } \left[ \left( \hat { u } ( x ) - u ^ { * } ( x ) \right) ^ { 2 } \right] ^ { 1 / 2 } \lesssim \left\{ \begin{array} { l l } { n ^ { - \frac { 1 } { 4 } } m ^ { - \frac { 1 - r } { 8 } } , } & { m \lesssim m _ { \mathrm { c r i t } } , } \\ { n ^ { - \frac { 1 } { 2 ( 1 + \alpha ) } } , } & { m \gtrsim m _ { \mathrm { c r i t } } , } \end{array} \right. } \end{array}
$$

where $m _ { \mathrm { c r i t } } \asymp n ^ { \frac { 2 ( 1 - \alpha ) } { ( 1 + \alpha ) ( 1 - r ) } }$

Here and throughout, $a ( n , m ) ~ \lesssim ~ b ( n , m )$ means that $a ( n , m ) ~ \leq ~ c b ( n , m )$ for some $c > 0$ independent of n and $m _ { : }$ up to logarithmic factors in $n , m .$ , and $\delta ^ { - 1 }$ (the high-probability parameter). The error bound reveals two regimes depending on the amount of diferential samples $m ,$ relative to the number of function-value samples n as depicted also in Fig. 1:

• Diferential-limited regime $( m \lesssim m _ { \mathrm { c r i t } } )$ . With low $m ,$ the error decreases as both n and m increase. In particular as the number of diferential observations $m$ increases, the estimation error along directions captured by $\Sigma _ { 2 }$ decreases. The rate of decrease depends on how well the $\Sigma _ { 2 }$ directions are aligned with the diferential covariance $\Sigma _ { D }$ . The exponent r quantifies this efect, with smaller values of r leading to faster decay of the error with respect to $m$

• Saturation threshold $( m \asymp m _ { \mathrm { c r i t } } )$ . Increasing $m ,$ a threshold is reached where diferential and value contributions to the error are of the same order:

$$
n ^ { - \frac { 1 } { 4 } } m ^ { - \frac { 1 - r } { 8 } } \asymp n ^ { - \frac { 1 } { 2 ( 1 + \alpha ) } } ,
$$

which yields $m _ { \mathrm { c r i t } } \asymp n ^ { \frac { 2 ( 1 - \alpha ) } { ( 1 + \alpha ) ( 1 - r ) } }$ . The dependence of $m _ { \mathrm { c r i t } }$ on α and $r$ follows from this relation: smaller values of r lead to a slower decay of the diferential term in $m _ { : }$ , and smaller values of α lead to a faster decay of the value term in $n ,$ so that a larger m is required to balance the two contributions.

• Saturation regime $( m \gtrsim m _ { \mathrm { c r i t } } )$ . Once m exceeds the threshold, the rate saturates: increasing the number of diferential observations only reduces lower-order terms, but it no longer improves the leading error term. At this point, the error on the diferential-accessible component is below the error driven by $\Sigma _ { 1 }$ , which captures directions that are either in the null-space of $D _ { \mathbf { \lambda } }$ and therefore invisible to diferential observations, or only weakly visible through them. These directions can be estimated only with value samples. In this regime, the leading rate matches the physical oracle rate obtained when the diferential information is known exactly (see Appendix A.8).

## 3.4 Learning rates in physically consistent norms

In many physics-informed settings, respecting the diferential structure is as important as minimizing prediction error. For instance, in learning interatomic potentials, predicting diferential quantities such as atomic forces is essential for molecular dynamics simulations [57]. However, improved learning rates for function values do not by themselves ensure that uˆ captures this structure. Indeed, convergence in value alone does not control derivatives: on any bounded domain, $u _ { \varepsilon } ( x ) = \varepsilon \sin ( \varepsilon ^ { - 2 } x )$ satisfies $\| u _ { \varepsilon } \| _ { L ^ { 2 } } \to 0 { \mathrm { ~ a s ~ } } \varepsilon \to 0$ , whereas for $D = \partial _ { x } , D u _ { \varepsilon } ( x ) = \varepsilon ^ { - 1 } \cos ( \varepsilon ^ { - 2 } x )$ and $\| D u _ { \varepsilon } \| _ { L ^ { 2 } } \to \infty$ We therefore study convergence in a stronger $L ^ { 2 } { \mathrm { - t } } .$ ype norm that jointly controls function-value and diferential errors.

Corollary 3.2 (Learning rates in the physically consistent norm). Under assumptions $( A I )$ to $( A \theta )$ there exist choices of regularization parameters $\lambda = \lambda ( n , m )$ and $\gamma = \gamma ( n , m )$ , given explicitly in the proof with $\gamma \geq 1$ such that, with high probability,

$$
\begin{array} { r } { \mathbb { E } \left[ \left( \widehat { u } ( x ) - u ^ { * } ( x ) \right) ^ { 2 } + \left( D \widehat { u } ( z ) - D u ^ { * } ( z ) \right) ^ { 2 } \right] ^ { 1 / 2 } \lesssim \left\{ \begin{array} { l l } { n ^ { - \frac { 1 } { 4 } } m ^ { - \frac { 1 - r } { 8 } } + m ^ { - 1 / 4 } , } & { m \lesssim m _ { \mathrm { c r i t } } } \\ { n ^ { - \frac { 1 } { 2 ( 1 + \alpha ) } } + m ^ { - 1 / 4 } , } & { m \gtrsim m _ { \mathrm { c r i t } } , } \end{array} \right. } \end{array}
$$

where $m _ { \mathrm { c r i t } } \asymp n ^ { \frac { 2 ( 1 - \alpha ) } { ( 1 + \alpha ) ( 1 - r ) } }$

The stronger norm bounds rule out the possibility that the PIKS estimator has a good prediction accuracy on $u ^ { * }$ while failing to accurately learn $D u ^ { * }$ . This can be interpreted as a form of physical consistency: the learned function respects both the values and the diferential structure of the target. Compared with the value-only rates in Corollary 3.1, this stronger guarantee incurs an additional term $m ^ { - 1 / 4 }$ , which depends only on the number of derivative observations and therefore cannot be reduced by increasing n alone. This is a reasonable price to pay in settings where diferential predictions are themselves of interest or are used in downstream tasks.

## 4 Examples

Here we illustrate the improved rates of Corollary 3.1 in concrete learning settings. We focus in Section 4.1 on Sobolev spaces on the torus where Laplacian information can provide large benefits, and in Section 4.2 on bounded domains, which are particularly relevant for their connection to PDEs.

## 4.1 Example 1: Sobolev spaces on the torus

We define the periodic Matérn kernel on $ { \mathbb { T } } ^ { d } = [ 0 , 1 ] ^ { d }$ with Fourier expansion

$$
k ( x - x ^ { \prime } ) = \sum _ { k \in \mathbb { Z } ^ { d } } \mu _ { k } e ^ { 2 \pi i k \cdot ( x - x ^ { \prime } ) } , \qquad \mu _ { k } \asymp ( 1 + \vert k \vert ^ { 2 } ) ^ { - s } ,
$$

for some $\begin{array} { r } { s > \frac { d } { 2 } + 2 } \end{array}$ , where $\asymp$ denotes the equality up to positive multiplicative constants. The associated RKHS H consists of functions $\begin{array} { r } { f ( x ) = \sum _ { k \in \mathbb { Z } ^ { d } } c _ { k } e ^ { 2 \pi i k \cdot x } } \end{array}$ with norm

$$
\| f \| _ { \mathcal { H } } ^ { 2 } = \sum _ { k \in \mathbb { Z } ^ { d } } \frac { | c _ { k } | ^ { 2 } } { \mu _ { k } } \asymp \sum _ { k \in \mathbb { Z } ^ { d } } ( 1 + | k | ^ { 2 } ) ^ { s } | c _ { k } | ^ { 2 } ,
$$

where $c _ { k } \in \mathbb { C }$ are the Fourier coeficients. H is norm-equivalent to the Sobolev space $H ^ { s } ( \mathbb { T } ^ { d } )$

Proposition 4.1 (Capacity decomposition for Sobolev spaces). Let $\mathcal { X } = \mathbb { T } ^ { d }$ , let k be a periodic Matérn kernel of smoothness $s > \frac { d } { 2 } + 2$ , and let $\begin{array} { r } { D = \sum _ { i \in S } \partial _ { z _ { i } } ^ { 2 } } \end{array}$ be a partial Laplacian with $S \subseteq$ $\{ 1 , \ldots , d \}$ . Assume that value and diferential samples are drawn uniformly on $\mathbb { T } ^ { d }$ . Then the value covariance Σ admits a decomposition $\Sigma = \Sigma _ { 1 } + \Sigma _ { 2 }$ satisfying assumption $( A \theta )$ , with exponents

$$
\alpha = \frac { d - | S | } { 2 s } , \qquad r = \frac { d } { 2 s } .
$$

Referring back to Corollary 3.1, this Sobolev example provides a functional-analytic interpretation of assumption (A6) and of the learning rates. For small $m ,$ the learning rate coincides with that of estimating a function in $H ^ { s } ( \mathbb { T } ^ { d } )$ from value observations, reflecting the full d-dimensional complexity of the data. When m exceeds the saturation threshold, all n function-value data points can be used to learn the nullspace of the partial Laplacian $D ,$ which depends only on $d _ { 0 } = d - | S |$ variables. The saturated rate therefore matches the minimax rate for Sobolev regression on $\mathbb { T } ^ { d _ { 0 } }$ , making explicit how the diferential constraint reduces the problem dimensionality.

## 4.2 Example 2: Gradients on bounded domain

Consider now the case when D is the gradient operator ∇, and the RKHS H is a Sobolev space of smoothness $\begin{array} { r } { s > \frac { d } { 2 } + 1 } \end{array}$ on domain $\mathcal { X } \subset \mathbb { R } ^ { d }$ . We consider the reproducing kernel k associated with H to be a Matérn kernel of smoothness s [58].

Note that, while in Sections 2 and 3 the analysis is limited to the case of scalar-valued $D ,$ it still holds for vector-valued operators. Denoting by $D _ { i }$ each scalar component of D, we decompose the diferential-data covariance as $\begin{array} { r } { \Sigma _ { D } : = \sum _ { \ell = 1 } ^ { k } \Sigma _ { D _ { \ell } } } \end{array}$ , where $\Sigma _ { D _ { \ell } } : = \mathbb { E } [ D _ { \ell } \phi ( z ) \otimes D _ { \ell } \phi ( z ) ]$ ]. Provided this new $\Sigma _ { D }$ satisfies assumption (A6) (which is unchanged), Corollary 3.1 still holds. We give more details on the extension to the vector-valued case in Appendix A.7.

When $D = \nabla$ , we have $\begin{array} { r } { \Sigma _ { D } = \mathbb E \left| \sum _ { i = 1 } ^ { d } \partial _ { i } \phi ( z ) \otimes \partial _ { i } \phi ( z ) \right| } \end{array}$ . The efect of learning with diferential constraints is shown in the following proposition.

Proposition 4.2. Let $\mathcal { X } \subset \mathbb { R } ^ { d }$ be a bounded, connected and Lipschitz domain. Fix $\begin{array} { r } { s > \frac { d } { 2 } + 1 } \end{array}$ . Let $\mathscr { H } = H ^ { s } ( \Omega )$ and let $D = \nabla$ be the gradient operator. Then the value covariance operator $\Sigma$ admits a decomposition $\Sigma = \Sigma _ { 1 } + \Sigma _ { 2 }$ satisfying assumption $( A \theta )$ , with exponents

$$
\alpha = 0 , \qquad r = \frac { d } { 2 ( s - 1 ) } .
$$

The saturated rate from Corollary 3.1 is obtained as long as m $\geq m _ { c r i t } = n ^ { \frac { 2 ( s - 1 ) } { ( s - 1 ) - d / 2 } } \geq n ^ { 2 }$ . Then,

$$
\begin{array} { r } { \mathbb { E } \left[ \left( \hat { u } ( x ) - u ^ { * } ( x ) \right) ^ { 2 } \right] ^ { 1 / 2 } \lesssim n ^ { - \frac { 1 } { 2 ( 1 + \alpha ) } } , } \end{array}
$$

which proves that the gradient information allows to obtain the parametric rate $n ^ { - { \frac { 1 } { 2 } } }$

## 5 Numerical experiments

We show how the diferent rate-regimes of Corollary 3.1 look like in practice through simulations on two learning problems. We will show i) the error saturation efect when m increases and ii) the efect of decreasing the capacity of $\Sigma _ { 1 }$ (decreasing α) via the partial Laplacian. More details on the implementation are available in Appendix C.

Saturation efect on Matérn data. We sample synthetic data on the 2d disk according to the following function, letting $x _ { \mathrm { s u p p } }$ be a support point in the domain and $p : = \| x - x _ { \mathrm { s u p p } } \|$

$$
u ^ { * } ( x ) = \left( 1 - \| x \| ^ { 2 } \right) \left[ \left( 1 + \sqrt { 5 } p + ( 5 / 3 ) p ^ { 2 } \right) \exp \left( - \sqrt { 5 } p \right) \right] .
$$

![](images/9411dafb10b491b49333f6fdab1df4e9a1952ee5117d8bf2dd852235a0887b41.jpg)  
Figure 2: Theoretical and experimental rates for fixed $n ,$ increasing m showing the saturation efect with gradient information on Matérn data.

![](images/ce2a66f3fca5c64c5d5efbdda1884d7566d9d98378a376c22605b5e064b10f0d.jpg)  
Figure 3: Test error rates increasing n and partial Laplacian dimensions |S|. Dashed lines show a best fit for the rates of Proposition 4.1. On the right are the inferred α exponents.

$u ^ { * }$ is in $H ^ { s }$ with $s = 3$ , and we will use the corresponding Matérn $\left( \nu = 5 / 2 \right)$ kernel. In Fig. 2 we show that, for fixed n and increasing $m _ { : }$ , the error rates switch between the two regimes of Corollary 3.1: an initial phase in which $m < m _ { \mathrm { c r i t } }$ and the error decreases as m grows, followed by a phase in which m has grown beyond the threshold and the error has reached a saturation with respect to m in which additional data does not improve accuracy. Here we used $D = \nabla$ and plotted empirical rates using PIKS next to the theoretical results from Section 4.2. Despite some diferences – notably the experimental curves saturate at similar values of m – due to the unknown problem-dependent constants afecting the rates, the saturation efect is clear as well as the dependence of the initial rate on both n and m.

Partial Laplacian. We now consider Sobolev spaces on the torus with partial Laplacian information. Unlike the example in Section 4.1, we approximate the infinite-dimensional Fourier construction by truncating to a finite number of frequencies. With $\mathcal { X } = \mathbb { T } ^ { 4 }$ , we randomly sample frequency vectors $k _ { \ell } \in \mathbb { Z } ^ { 4 }$ such that a single dimension is active (non-zero) at a time. Writing $\phi _ { \ell } ( x ) = \sqrt { 2 } \cos ( 2 \pi \langle k _ { \ell } , x \rangle )$ for the feature-map, target function and kernel are defined as

$$
u ^ { * } ( x ) = u _ { 0 } + \sum _ { \ell = 1 } ^ { F } c _ { \ell } \phi _ { \ell } ( x ) , \qquad k ( x , x ^ { \prime } ) = 1 + \sum _ { \ell = 1 } ^ { F } \mu _ { \ell } \phi _ { \ell } ( x ) \phi _ { \ell } ( x ^ { \prime } ) ,
$$

with $F$ frequency coeficients $c _ { \ell }$ and $\mu _ { \ell }$ which depend on a smoothness parameter of the problem. $k _ { \ell }$ are sampled such that $u ^ { * }$ decomposes into frequency blocks $u ^ { * } ( x ) = u _ { 0 } + u _ { 1 } ( x ) + u _ { 2 } ( x ) + u _ { 3 } ( x ) + u _ { 4 } ( x )$ each of which only depends on a single dimension of x and cannot be observed unless the partial Laplacian $\begin{array} { r } { D _ { s } = \sum _ { i = 1 } ^ { s } \partial _ { z _ { i } } ^ { 2 } } \end{array}$ includes that specific dimension. This creates a simple setting in which the incrementing the partial Laplacian order $s ,$ increases the amount of information which can be learned with the diferential data. Moreover the Sobolev capacity decomposition from Proposition 4.1 remains valid after truncation, with constants independent of $F .$

In Fig. 3 we observe the learning rate in n, as the type of diferential data changes: from having access to function data only $( D _ { 0 } )$ to having access to the full Laplacian $( D _ { 4 } )$ . The number of diferential points m is set as a constant factor of n. Best fits are obtained for the theoretical rates of Theorem 3.1 to obtain the exponent term of the error rate in n. Despite the high variance of the experiments, due to the noise $\varepsilon , \xi$ and the random sampling of points $x _ { i } , z _ { j } .$ it is evident that with each increase in the amount of diferential information |S| the learning rate increases. From the fitted curves we can infer the values of α which – as predicted by Proposition 4.1 – have a linear relationship with the number of partial Laplacian dimensions (denoted by |S|), despite the mismatch caused by the finite-dimensional setting.

## 6 Conclusion and research directions

In this paper, we provide a precise characterization of how kernel regression benefits from learning with both function-value and diferential observations. Our analysis quantifies how the prediction error depends jointly on the number of value samples, the number of diferential samples, and the structure of the underlying diferential operator. The resulting rates reveal two regimes: a diferential-limited regime, in which the error decreases with both types of samples, and a saturation regime, in which the leading rate matches the physical oracle rate attainable with exact diferential information. We also establish convergence in a stronger norm that controls both function-value and diferential errors. Several questions remain open: are these rates minimax optimal? Can similar gains be obtained for nonlinear diferential operators or misspecified physical constraints? Is it possible to preserve the statistical gains with approximate algorithms which are more eficient?

## Acknowledgements

This material is based upon work supported by the European Commission (Horizon Europe grant ELIAS 101120237), and the Ministry of Education, University and Research (FARE grant ML4IP R205T7J2KP).

## References

[1] Salvatore Cuomo, Vincenzo Schiano Di Cola, Fabio Giampaolo, Gianluigi Rozza, Maziar Raissi, and Francesco Piccialli. Scientific machine learning through physics–informed neural networks: Where we are and what’s next. Journal of Scientific Computing, 92(3), 2022.

[2] Maziar Raissi, Paris Perdikaris, and George E Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations. Journal of Computational Physics, 378:686–707, 2019.

[3] George Em Karniadakis, Ioannis G Kevrekidis, Lu Lu, Paris Perdikaris, Sifan Wang, and Liu Yang. Physics-informed machine learning. Nature Reviews Physics, 3(6), 2021.

[4] Christopher Rackauckas, Yingbo Ma, Julius Martensen, Collin Warner, Kirill Zubov, Rohit Supekar, Dominic Skinner, Ali Ramadhan, and Alan Edelman. Universal diferential equations for scientific machine learning. arXiv preprint arXiv:2001.04385, 2020.

[5] Jörg Behler and Michele Parrinello. Generalized neural-network representation of high-dimensional potential-energy surfaces. Physical review letters, 98(14):146401, 2007.

[6] Albert P. Bartók, Mike C. Payne, Risi Kondor, and Gábor Csányi. Gaussian approximation potentials: The accuracy of quantum mechanics, without the electrons. Phys. Rev. Lett., 104, 2010. doi: 10.1103/ PhysRevLett.104.136403.

[7] Amos Gropp, Lior Yariv, Niv Haim, Matan Atzmon, and Yaron Lipman. Implicit geometric regularization for learning shapes. In Hal Daumé III and Aarti Singh, editors, Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 3789–3799. PMLR, 13–18 Jul 2020. URL https://proceedings.mlr.press/v119/gropp20a.html.

[8] Aarya Patel, Hamid Laga, and Ojaswa Sharma. Normal-guided detail-preserving neural implicit function for high-fidelity 3D surface reconstruction. Proceedings of the ACM on Computer Graphics and Interactive Techniques, 8(1), 2025. doi: https://doi.org/10.1145/3728293.

[9] Alfio Quarteroni, Paola Gervasio, and Francesco Regazzoni. Combining physics-based and data-driven models: advancing the frontiers of research with scientific machine learning. Mathematical Models and Methods in Applied Sciences, 35(4):905–1071, 2025. doi: 10.1142/S0218202525500125.

[10] Tommaso Botarelli, Marco Fanfani, Paolo Nesi, and Lorenzo Pinelli. Using physics-informed neural networks for solving navier-stokes equations in fluid dynamic complex scenarios. Engineering Applications of Artificial Intelligence, 148, 2025. doi: https://doi.org/10.1016/j.engappai.2025.110347.

[11] Andrea Caponnetto and Ernesto De Vito. Optimal rates for the regularized least-squares algorithm. Foundations of Computational mathematics, 7(3):331–368, 2007.

[12] László Györfi, Michael Kohler, Adam Krzyżak, and Harro Walk. A distribution-free theory of nonparametric regression. Springer, 2002.

[13] George Kimeldorf and Grace Wahba. Some results on Tchebychefian spline functions. Journal of mathematical analysis and applications, 33(1):82–95, 1971.

[14] Wojciech M Czarnecki, Simon Osindero, Max Jaderberg, Grzegorz Swirszcz, and Razvan Pascanu. Sobolev training for neural networks. Advances in neural information processing systems, 30, 2017.

[15] Maziar Raissi, Paris Perdikaris, and George Em Karniadakis. Physics informed deep learning (part i): Data-driven solutions of nonlinear partial diferential equations, 2017.

[16] Grace Wahba. Spline models for observational data. SIAM, 1990.

[17] Martin Hanke. Regularization with diferential operators: an iterative approach. Numerical functional analysis and optimization, 13(5-6):523–540, 1992.

[18] Heinz Werner Engl, Martin Hanke, and Andreas Neubauer. Regularization of inverse problems, volume 375. Springer Science & Business Media, 1996.

[19] T. Poggio and F. Girosi. Regularization algorithms for learning that are equivalent to multilayer networks. Science, 247(4945), 1990. doi: 10.1126/science.247.4945.978.

[20] Xiaojin Zhu, Zoubin Ghahramani, and John D Laferty. Semi-supervised learning using Gaussian fields and harmonic functions. In Proceedings of the 20th International conference on Machine learning (ICML-03), pages 912–919, 2003.

[21] Dengyong Zhou and Bernhard Schölkopf. Regularization on discrete spaces. In Joint Pattern Recognition Symposium, pages 361–368. Springer, 2005.

[22] Mikhail Belkin, Partha Niyogi, and Vikas Sindhwani. Manifold regularization: A geometric framework for learning from labeled and unlabeled examples. Journal of machine learning research, 7(11), 2006.

[23] Dejan Slepcev and Matthew Thorpe. Analysis of p-Laplacian regularization in semisupervised learning. SIAM Journal on Mathematical Analysis, 51(3):2085–2120, 2019.

[24] Vivien Cabannes, Loucas Pillaud-Vivien, Francis Bach, and Alessandro Rudi. Overcoming the curse of dimensionality with Laplacian regularization in semi-supervised learning. Advances in Neural Information Processing Systems, 34:30439–30451, 2021.

[25] Laura M Sangalli. Spatial regression with partial diferential equation regularisation. International Statistical Review, 89(3):505–531, 2021.

[26] Lawrence C Evans. Partial diferential equations, volume 19. American mathematical society, 2010.

[27] Bing Yu et al. The deep Ritz method: a deep learning-based numerical algorithm for solving variational problems. Communications in Mathematics and Statistics, 6(1), 2018.

[28] Lu Lu, Pengzhan Jin, Guofei Pang, Zhongqiang Zhang, and George Em Karniadakis. Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. Nature machine intelligence, 3(3), 2021.

[29] Zongyi Li, Nikola Borislavov Kovachki, Kamyar Azizzadenesheli, Burigede liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Fourier neural operator for parametric partial diferential equations. In International Conference on Learning Representations, 2021. URL https://openreview. net/forum?id=c8P9NQVtmnO.

[30] Edward J Kansa. Multiquadrics—a scattered data approximation scheme with applications to computational fluid-dynamics—ii solutions to parabolic, hyperbolic and elliptic partial diferential equations. Computers & mathematics with applications, 19(8-9):147–161, 1990.

[31] Wu Zongmin. Hermite-Birkhof interpolation of scattered data by radial basis functions. Approximation Theory and its Applications, 8(2):1–10, 1992.

[32] Gregory E Fasshauer. Solving partial diferential equations by collocation with radial basis functions. In Proceedings of Chamonix, volume 1997, pages 1–8, 1996.

[33] Holger Wendland. Scattered data approximation, volume 17. Cambridge university press, 2004.

[34] Houman Owhadi. Bayesian numerical homogenization. Multiscale Modeling & Simulation, 13(3): 812–828, 2015.

[35] Maziar Raissi, Paris Perdikaris, and George Em Karniadakis. Machine learning of linear diferential equations using Gaussian processes. Journal of Computational Physics, 348:683–693, 2017.

[36] Yifan Chen, Bamdad Hosseini, Houman Owhadi, and Andrew M Stuart. Solving and learning nonlinear PDEs with Gaussian processes. Journal of Computational Physics, 447:110668, 2021.

[37] Nathan Doumèche, Francis Bach, Gérard Biau, and Claire Boyer. Physics-informed machine learning as a kernel method. In The Thirty Seventh Annual Conference on Learning Theory, pages 1399–1450. PMLR, 2024.

[38] Joachim Bona-Pellissier, Giacomo Meanti, Matteo Santacesaria, and Lorenzo Rosasco. Piks: Universal physics-informed kernel methods. arXiv preprint arXiv:2607.27062, 2026.

[39] Laura Azzimonti, Laura M Sangalli, Piercesare Secchi, Maurizio Domanin, and Fabio Nobile. Blood flow velocity field estimation via spatial regression with PDE penalization. Journal of the American Statistical Association, 110(511):1057–1071, 2015.

[40] Carsten Franke and Robert Schaback. Solving partial diferential equations by collocation using radial basis functions. Applied Mathematics and computation, 93(1):73–82, 1998.

[41] Robert Schaback and Holger Wendland. Kernel techniques: from machine learning to meshless methods. Acta numerica, 15, 2006.

[42] Pau Batlle, Yifan Chen, Bamdad Hosseini, Houman Owhadi, and Andrew M Stuart. Error analysis of kernel/GP methods for nonlinear and parametric PDEs. Journal of Computational Physics, 520, 2025.

[43] Yeonjong Shin, Jerome Darbon, and George Em Karniadakis. On the convergence of physics informed neural networks for linear second-order elliptic and parabolic type PDEs. Communications in Computational Physics, 28(5), 2020.

[44] Siddhartha Mishra and Roberto Molinaro. Estimates on the generalization error of physics-informed neural networks for approximating PDEs. IMA Journal of Numerical Analysis, 43(1):1–43, 2023.

[45] Yeonjong Shin, Zhongqiang Zhang, and George Em Karniadakis. Error estimates of residual minimization using neural networks for linear PDEs. Journal of Machine Learning for Modeling and Computing, 4 (4), 2023.

[46] Tim De Ryck, Ameya D Jagtap, and Siddhartha Mishra. Error estimates for physics-informed neural networks approximating the Navier–Stokes equations. IMA Journal of Numerical Analysis, 44(1): 83–119, 2024.

[47] Nathan Doumèche, Gérard Biau, and Claire Boyer. On the convergence of PINNs. Bernoulli, 31(3): 2127 – 2151, 2025. doi: 10.3150/24-BEJ1799.

[48] Lei Shi, Xin Guo, and Ding-Xuan Zhou. Hermite learning with gradient data. Journal of computational and applied mathematics, 233(11):3046–3059, 2010.

[49] Zain ul Abdeen, Ruoxi Jia, Vassilis Kekatos, and Ming Jin. A theoretical analysis of using gradient data for Sobolev training in RKHS. IFAC-PapersOnLine, 56(2), 2023. doi: https://doi.org/10.1016/j. ifacol.2023.10.1491. 22nd IFAC World Congress.

[50] Katharine E Fisher, Matthew TC Li, Youssef Marzouk, and Timo Schorlepp. Precise asymptotic analysis of Sobolev training for random feature models. arXiv preprint arXiv:2511.03050, 2025.

[51] Eleonora Arnone, Alois Kneip, Fabio Nobile, and Laura M Sangalli. Some first results on the consistency of spatial regression with partial diferential equation regularization. Statistica Sinica, 32(1):209–238, 2022.

[52] Nathan Doumèche, Francis Bach, Gérard Biau, and Claire Boyer. Physics-informed kernel learning. Journal of Machine Learning Research, 26(124):1–39, 2025.

[53] Nathan Doumèche, Francis Bach, Gérard Biau, and Claire Boyer. Fast kernel methods: Sobolev, physics-informed, and additive models. arXiv preprint arXiv:2509.02649, 2025.

[54] Ingo Steinwart, Don R Hush, Clint Scovel, et al. Optimal rates for regularized least squares regression. In COLT, pages 79–93, 2009.

[55] Simon Fischer and Ingo Steinwart. Sobolev norm learning rates for regularized least-squares algorithms. Journal of Machine Learning Research, 21(205):1–38, 2020.

[56] Ingo Steinwart and Andreas Christmann. Support vector machines. Springer Science & Business Media, 2008.

[57] Frank Noé, Alexandre Tkatchenko, Klaus-Robert Müller, and Cecilia Clementi. Machine learning for molecular simulation. Annual review of physical chemistry, 71(1), 2020.

[58] Christopher KI Williams and Carl Edward Rasmussen. Gaussian processes for machine learning, volume 2. MIT press Cambridge, MA, 2006.

[59] Alessandro Rudi, Guillermo D Canas, and Lorenzo Rosasco. On the sample complexity of subspace learning. Advances in Neural Information Processing Systems, 26, 2013.

[60] Luc Brogat-Motte, Riccardo Bonalli, and Alessandro Rudi. Learning controlled stochastic diferential equations. arXiv preprint arXiv:2411.01982, 2024.

## A Proofs

This section provides the proofs of the main results. We first derive the closed-form expressions of the empirical and population minimizers in Section A.1, then establish an error decomposition in Section A.2, derive high-probability bounds for its constituent terms in Section $\mathrm { A . 3 , }$ and finally combine these results to obtain finite-sample guarantees and optimized learning rates in Sections A.4 and A.5.

## A.1 Closed forms

We derive here the closed-form expressions of the empirical minimizer of the regularized empirical risk $\widehat { R }$ introduced in Section 2,

$$
\widehat { R } ( u ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( u ( x _ { i } ) - y _ { i } \right) ^ { 2 } + \gamma \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \left( D u ( z _ { j } ) - d _ { j } \right) ^ { 2 } + \lambda \| u \| _ { \mathcal { H } } ^ { 2 } ,
$$

and of the minimizer of its population counterpart

$$
R _ { \lambda , \gamma } ( u ) = \mathbb { E } \big [ ( u ( x ) - y ) ^ { 2 } \big ] + \gamma \mathbb { E } \big [ ( D u ( z ) - d ) ^ { 2 } \big ] + \lambda \| u \| _ { \mathcal { H } } ^ { 2 } ,
$$

where the expectations are taken over $x \sim \rho , z \sim \rho _ { D }$ , and the corresponding observation noises. The empirical minimizer is given in operator form in Lemma A.1 and in finite-dimensional form in Lemma A.2. The population minimizer is given in Lemma A.3.

Lemma A.1 (Empirical solution). Let $\lambda > 0$ and $\gamma > 0$ . Define

$$
{ \widehat { \Sigma } } : = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \phi ( x _ { i } ) \otimes \phi ( x _ { i } ) , \qquad { \widehat { \Sigma } } _ { D } : = { \frac { 1 } { m } } \sum _ { j = 1 } ^ { m } D \phi ( z _ { j } ) \otimes D \phi ( z _ { j } ) ,
$$

$$
\widehat V : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } y _ { i } \phi ( x _ { i } ) , \qquad \widehat V _ { D } : = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } d _ { j } D \phi ( z _ { j } ) .
$$

Then $\widehat { R }$ has a unique minimizer $\hat { u } \in \mathcal H$ of the form

$$
\begin{array} { r } { \hat { u } ( x ) = \langle \widehat { w } , \phi ( x ) \rangle _ { \mathcal { H } } , \qquad \widehat { w } = \left( \widehat { \Sigma } + \gamma \widehat { \Sigma } _ { D } + { \lambda } I \right) ^ { - 1 } \big ( \widehat { V } + \gamma \widehat { V } _ { D } \big ) . } \end{array}
$$

Proof. Using the representation $u ( x ) = \langle w , \phi ( x ) \rangle _ { \mathcal { H } } , D u ( z ) = \langle w , D \phi ( z ) \rangle _ { \mathcal { H } }$ , we write $\widehat { R } ( u ) = \widehat { R } ( w )$ as

$$
\widehat { R } ( w ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( \langle w , \phi ( x _ { i } ) \rangle _ { \mathcal { H } } - y _ { i } \right) ^ { 2 } + \gamma \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \left( \langle w , D \phi ( z _ { j } ) \rangle _ { \mathcal { H } } - d _ { j } \right) ^ { 2 } + \lambda \langle w , w \rangle _ { \mathcal { H } } .
$$

Expanding the least-squares terms in ${ \mathcal { H } } ,$ we obtain for the value part

$$
\begin{array} { r l } { \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \big ( \langle w , \phi ( x _ { i } ) \rangle - y _ { i } \big ) ^ { 2 } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \Big ( \langle w , \phi ( x _ { i } ) \rangle ^ { 2 } - 2 y _ { i } \langle w , \phi ( x _ { i } ) \rangle + y _ { i } ^ { 2 } \Big ) } & { } \\ { \displaystyle } & { = \left. w , \Big ( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \phi ( x _ { i } ) \otimes \phi ( x _ { i } ) \Big ) w \right. _ { \mathcal { H } } - 2 \left. \frac { 1 } { n } \sum _ { i = 1 } ^ { n } y _ { i } \phi ( x _ { i } ) , w \right. _ { \mathcal { H } } + \mathrm { c o n s t } } \\ { \displaystyle } & { = \langle w , \widehat { \Sigma } w \rangle _ { \mathcal { H } } - 2 \langle \widehat { V } , w \rangle _ { \mathcal { H } } + \mathrm { c o n s t } , } \end{array}
$$

and for the derivative part

$$
\gamma \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \left( \langle w , D \phi ( z _ { j } ) \rangle - d _ { j } \right) ^ { 2 } = \gamma \left( \langle w , \widehat { \Sigma } _ { D } w \rangle _ { \mathcal { H } } - 2 \langle \widehat { V } _ { D } , w \rangle _ { \mathcal { H } } + \mathrm { c o n s t } \right) .
$$

Adding the regularization term $\lambda \langle w , w \rangle _ { \mathcal { H } }$ , we get

$$
\widehat { R } ( w ) = \left. w , ( \widehat { \Sigma } + \gamma \widehat { \Sigma } _ { D } + \lambda I ) w \right. _ { \mathcal { H } } - 2 \big \langle \widehat { V } + \gamma \widehat { V } _ { D } , w \big \rangle _ { \mathcal { H } } + \mathrm { c o n s t } .
$$

Since $\widehat { \Sigma }$ and $\widehat { \Sigma } _ { D }$ are positive self-adjoint and $\lambda > 0$ , the operator $\widehat { \Sigma } + \gamma \widehat { \Sigma } _ { D } + \lambda I$ is strictly positive. Thus $\widehat { R } ( w )$ is a strictly convex quadratic functional of $w ,$ whose unique minimizer wb is characterized by the normal equation

$$
\left( \widehat { \Sigma } + \gamma \widehat { \Sigma } _ { D } + \lambda I \right) \widehat { w } = \widehat { V } + \gamma \widehat { V } _ { D } ,
$$

which yields

$$
\widehat { w } = \big ( \widehat { \Sigma } + \gamma \widehat { \Sigma } _ { D } + \lambda I \big ) ^ { - 1 } \big ( \widehat { V } + \gamma \widehat { V } _ { D } \big ) .
$$

The corresponding minimizer in function space is $\hat { u } ( x ) = \langle \widehat { w } , \phi ( x ) \rangle _ { \mathcal { H } }$

Lemma A.2 (Finite-dimensional form of the PIKS estimator). Let $\lambda > 0$ and $\gamma > 0$ . Define the Gram matrices

$$
\begin{array} { c c } { { ( K _ { X X } ) _ { i i ^ { \prime } } = k ( x _ { i } , x _ { i ^ { \prime } } ) , } } & { { ( K _ { X Z } ) _ { i j } = D _ { 1 } k ( z _ { j } , x _ { i } ) , } } \\ { { { } } } & { { { } } } \\ { { ( K _ { Z X } ) _ { j i } = D _ { 2 } k ( x _ { i } , z _ { j } ) , } } & { { ( K _ { Z Z } ^ { D } ) _ { j j ^ { \prime } } = D _ { 1 } D _ { 2 } k ( z _ { j } , z _ { j ^ { \prime } } ) . } } \end{array}
$$

Then the PIKS estimator can be computed as

$$
\hat { u } ( \cdot ) = \frac { 1 } { \sqrt { n } } \sum _ { i = 1 } ^ { n } \alpha _ { i } k ( x _ { i } , \cdot ) + \sqrt { \frac { \gamma } { m } } \sum _ { j = 1 } ^ { m } \beta _ { j } D _ { 1 } k ( z _ { j } , \cdot ) , \qquad \theta = \left[ { \beta } \right] ,
$$

where $\boldsymbol { \theta } \in \mathbb { R } ^ { n + m }$ solves

$$
\begin{array} { r } { \left( \left[ \sqrt { \frac { 1 } { n m } } K _ { X X } \right. \left. \sqrt { \frac { \gamma } { n m } } K _ { X Z } \right] + \lambda I \right) \theta = \left[ \sqrt [ y / \sqrt { n } ] { n } \right] . } \end{array}
$$

Proof. Let $( e _ { i } ) _ { i = 1 } ^ { n }$ and $( f _ { j } ) _ { j = 1 } ^ { m }$ denote the canonical bases of $\mathbb { R } ^ { n }$ and $\mathbb { R } ^ { m }$ , and define

$$
\Phi e _ { i } = k ( x _ { i } , \cdot ) , \qquad D \Phi f _ { j } = D _ { 1 } k ( z _ { j } , \cdot ) .
$$

Set

$$
\begin{array} { r } { \Gamma _ { \gamma } = \Big [ \frac { 1 } { \sqrt { n } } \Phi \quad \sqrt { \frac { \gamma } { m } } D \Phi \Big ] , \qquad b _ { \gamma } = \Big [ \frac { y / \sqrt { n } } { \sqrt { \gamma } d / \sqrt { m } } \Big ] . } \end{array}
$$

Then

$$
\widehat { R } ( u ) = \| \Gamma _ { \gamma } ^ { * } u - b _ { \gamma } \| _ { 2 } ^ { 2 } + \lambda \| u \| _ { \mathcal { H } } ^ { 2 } .
$$

The optimality condition gives

$$
( \Gamma _ { \gamma } \Gamma _ { \gamma } ^ { \ast } + \lambda I ) \hat { u } = \Gamma _ { \gamma } b _ { \gamma } .
$$

Using

$$
( \Gamma _ { \gamma } \Gamma _ { \gamma } ^ { \ast } + \lambda I ) ^ { - 1 } \Gamma _ { \gamma } = \Gamma _ { \gamma } ( \Gamma _ { \gamma } ^ { \ast } \Gamma _ { \gamma } + \lambda I ) ^ { - 1 } ,
$$

we get

$$
\hat { u } = \Gamma _ { \gamma } \theta , \qquad \theta = ( \Gamma _ { \gamma } ^ { \ast } \Gamma _ { \gamma } + \lambda I ) ^ { - 1 } b _ { \gamma } .
$$

Finally,

$$
\Gamma _ { \gamma } ^ { \ast } \Gamma _ { \gamma } = \left[ \begin{array} { c c } { { \frac { 1 } { n } K _ { X X } } } & { { \sqrt { \frac { \gamma } { n m } } K _ { X Z } } } \\ { { \sqrt { \frac { \gamma } { n m } } K _ { Z X } } } & { { \frac { \gamma } { m } K _ { Z Z } ^ { D } } } \end{array} \right] ,
$$

which gives the claimed system. Moreover, since

$$
\hat { u } = \Gamma _ { \gamma } \theta = \frac { 1 } { \sqrt { n } } \Phi \alpha + \sqrt { \frac { \gamma } { m } } D \Phi \beta ,
$$

we obtain

$$
\hat { u } ( \cdot ) = \frac { 1 } { \sqrt { n } } \sum _ { i = 1 } ^ { n } \alpha _ { i } k ( x _ { i } , \cdot ) + \sqrt { \frac { \gamma } { m } } \sum _ { j = 1 } ^ { m } \beta _ { j } D _ { 1 } k ( z _ { j } , \cdot ) .
$$

Lemma A.3 (Population solution). Let $x \sim \rho$ and $z \sim \rho _ { D }$ , and let the expectations involving y and d be taken with respect to the joint laws induced by the observation models. Define

$$
\Sigma : = \mathbb { E } _ { x \sim \rho } \big [ \phi ( x ) \otimes \phi ( x ) \big ] , \qquad \Sigma _ { D } : = \mathbb { E } _ { z \sim \rho _ { D } } \big [ D \phi ( z ) \otimes D \phi ( z ) \big ] ,
$$

$$
V : = \mathbb { E } { \bigl [ } y \phi ( x ) { \bigr ] } , \qquad V _ { D } : = \mathbb { E } { \bigl [ } d D \phi ( z ) { \bigr ] } .
$$

Then $R _ { \lambda , \gamma }$ has a unique minimizer $u _ { \lambda , \gamma } \in \mathcal { H }$ of the form

$$
u _ { \lambda , \gamma } ( x ) = \langle w _ { \lambda , \gamma } , \phi ( x ) \rangle _ { \mathcal { H } } , \qquad w _ { \lambda , \gamma } = \big ( \Sigma + \gamma \Sigma _ { D } + \lambda I \big ) ^ { - 1 } ( V + \gamma V _ { D } ) .
$$

Proof. The derivations are the same as in the empirical case, (Lemma A.1) with empirical averages replaced by expectations and $\widehat { \Sigma } , \widehat { \Sigma } _ { D } , \widehat { V } , \widehat { V } _ { D }$ replaced by $\Sigma , \Sigma _ { D } , V , V _ { D }$ , and with the same quadratic expansion of $R _ { \lambda , \gamma } ( w )$

## A.2 Error decomposition

We next establish a standard error decomposition adapted to our setting. The idea is to separate the error into a bias term and an estimation term, controlled by the deviations of the empirical covariance operators from their population counterparts.

Lemma A.4 (Error decomposition). Let

$$
\Sigma _ { \gamma } : = \Sigma + \gamma \Sigma _ { D } , \qquad \widehat { \Sigma } _ { \gamma } : = \widehat { \Sigma } + \gamma \widehat { \Sigma } _ { D } , \qquad V _ { \gamma } : = V + \gamma V _ { D } , \qquad \widehat { V } _ { \gamma } : = \widehat { V } + \gamma \widehat { V } _ { D } ,
$$

and define

$$
\begin{array} { r } { w _ { \lambda , \gamma } : = ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 } V _ { \gamma } , \qquad \widehat { w } : = ( \widehat \Sigma _ { \gamma } + \lambda I ) ^ { - 1 } \widehat V _ { \gamma } , \qquad u ^ { * } ( x ) = \langle w ^ { * } , \phi ( x ) \rangle _ { \mathcal { H } } . } \end{array}
$$

Assume Assumption (A4) holds. Define the quantities

$$
\begin{array} { r l } & { \Delta _ { \Sigma , 1 } : = \| ( \Sigma _ { \gamma } + \lambda I ) ^ { 1 / 2 } ( \widehat \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } \| } \\ & { \Delta _ { \Sigma , 2 } : = \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \Sigma _ { \gamma } - \widehat \Sigma _ { \gamma } ) \big \| } \\ & { \Delta _ { V } : = \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \widehat V _ { \gamma } - V _ { \gamma } ) \big \| } \\ & { \mathcal B ( \lambda , \gamma ) : = \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } w ^ { * } \big \| . } \end{array}
$$

Then

$$
\begin{array} { r } { \| \Sigma ^ { 1 / 2 } ( \widehat { w } - w ^ { * } ) \| \leq \Delta _ { \Sigma , 1 } ^ { 2 } ( \Delta _ { V } + \left\| w ^ { * } \right\| \Delta _ { \Sigma , 2 } ) + \lambda B ( \lambda , \gamma ) . } \end{array}
$$

Proof. We first split the error into an estimation and a bias term:

$$
\| \Sigma ^ { 1 / 2 } ( \widehat { w } - w ^ { * } ) \| \leq \| \Sigma ^ { 1 / 2 } ( \widehat { w } - w _ { \lambda , \gamma } ) \| + \| \Sigma ^ { 1 / 2 } ( w _ { \lambda , \gamma } - w ^ { * } ) \| .
$$

We now bound each term.

Bias term. Under the Assumption (A4) we have

$$
V = \Sigma w ^ { \ast } , \qquad V _ { D } = \Sigma _ { D } w ^ { \ast } ,
$$

hence

$$
V _ { \gamma } = V + \gamma V _ { D } = ( \Sigma + \gamma \Sigma _ { D } ) w ^ { * } = \Sigma _ { \gamma } w ^ { * } .
$$

Therefore

$$
\begin{array} { r l } & { w _ { \lambda , \gamma } - w ^ { * } = ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 } V _ { \gamma } - w ^ { * } } \\ & { \qquad = ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 } \Sigma _ { \gamma } w ^ { * } - w ^ { * } } \\ & { \qquad = \bigl [ ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 } \Sigma _ { \gamma } - I \bigr ] w ^ { * } } \\ & { \qquad = - \lambda ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 } w ^ { * } . } \end{array}
$$

As a consequence,

$$
\begin{array} { r } { \| \Sigma ^ { 1 / 2 } ( w _ { \lambda , \gamma } - w ^ { * } ) \| = \lambda \big \| \Sigma ^ { 1 / 2 } ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 } w ^ { * } \big \| \leq \lambda \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } w ^ { * } \big \| , } \end{array}
$$

where we used $\| \Sigma ^ { 1 / 2 } ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } \| \le 1$ since $\Sigma \preccurlyeq \Sigma _ { \gamma } + \lambda I$

Estimation term. We have

$$
\widehat { w } - w _ { \lambda , \gamma } = ( \widehat { \Sigma } _ { \gamma } + \lambda I ) ^ { - 1 } ( \widehat { V } _ { \gamma } - V _ { \gamma } ) + \Big [ ( \widehat { \Sigma } _ { \gamma } + \lambda I ) ^ { - 1 } - ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 } \Big ] V _ { \gamma } ,
$$

and, using $A ^ { - 1 } - B ^ { - 1 } = A ^ { - 1 } ( B - A ) B ^ { - 1 } \ \mathrm { w i t h } \ A = \widehat { \Sigma } _ { \gamma } + \lambda I , \ B = \Sigma _ { \gamma } + \lambda I ,$

$$
\begin{array} { r } { ( \widehat { \Sigma } _ { \gamma } + \lambda I ) ^ { - 1 } - ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 } = ( \widehat { \Sigma } _ { \gamma } + \lambda I ) ^ { - 1 } ( \Sigma _ { \gamma } - \widehat { \Sigma } _ { \gamma } ) ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 } . } \end{array}
$$

Multiplying by $\Sigma ^ { 1 / 2 }$ and inserting $( \Sigma _ { \gamma } + \lambda I ) ^ { \pm 1 / 2 }$ , we obtain

$$
\begin{array} { r l } & { \| \Sigma ^ { 1 / 2 } ( \widehat { w } - w _ { \lambda , \gamma } ) \| \leq \big \| \Sigma ^ { 1 / 2 } ( \widehat { \Sigma } _ { \gamma } + \lambda I ) ^ { - 1 / 2 } \big \| \big \| ( \widehat { \Sigma } _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \Sigma _ { \gamma } + \lambda I ) ^ { 1 / 2 } \big \| } \\ & { \qquad \times \Big [ \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \widehat { V } _ { \gamma } - V _ { \gamma } ) \big \| } \\ & { \qquad + \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \Sigma _ { \gamma } - \widehat { \Sigma } _ { \gamma } ) \big \| \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 } V _ { \gamma } \big \| \Big ] . } \end{array}
$$

Then,

$$
\begin{array} { r l } & { \left\| { \Sigma ^ { 1 / 2 } ( \widehat { \Sigma } _ { \gamma } + \lambda I ) ^ { - 1 / 2 } } \right\| = \left\| { \Sigma ^ { 1 / 2 } ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \Sigma _ { \gamma } + \lambda I ) ^ { 1 / 2 } ( \widehat { \Sigma } _ { \gamma } + \lambda I ) ^ { - 1 / 2 } } \right\| } \\ & { \qquad \leq \left\| { \Sigma ^ { 1 / 2 } ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } } \right\| \left\| { ( \Sigma _ { \gamma } + \lambda I ) ^ { 1 / 2 } ( \widehat { \Sigma } _ { \gamma } + \lambda I ) ^ { - 1 / 2 } } \right\| } \\ & { \qquad \leq \left\| { ( \Sigma _ { \gamma } + \lambda I ) ^ { 1 / 2 } ( \widehat { \Sigma } _ { \gamma } + \lambda I ) ^ { - 1 / 2 } } \right\| , } \end{array}
$$

where we used again that $\| \Sigma ^ { 1 / 2 } ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } \| \le 1$

Moreover, using $V _ { \gamma } = \Sigma _ { \gamma } w ^ { * }$ , we have

$$
\begin{array} { r l } & { \left\| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 } V _ { \gamma } \right\| = \left\| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 } \Sigma _ { \gamma } w ^ { * } \right\| } \\ & { \qquad \leq \left\| w ^ { * } \right\| . } \end{array}
$$

Conclusion. We define

$$
\begin{array} { r l } & { \Delta _ { \Sigma , 1 } : = \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { 1 / 2 } ( \widehat \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } \big \| } \\ & { \Delta _ { \Sigma , 2 } : = \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \Sigma _ { \gamma } - \widehat \Sigma _ { \gamma } ) \big \| } \\ & { \Delta _ { V } : = \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \widehat V _ { \gamma } - V _ { \gamma } ) \big \| } \\ & { B ( \lambda , \gamma ) : = \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } w ^ { * } \big \| . } \end{array}
$$

Collecting the terms, we obtain

$$
\begin{array} { r } { \| \Sigma ^ { 1 / 2 } ( \widehat { w } - w _ { \lambda , \gamma } ) \| \leq \Delta _ { \Sigma , 1 } ^ { 2 } ( \Delta _ { V } + \| w ^ { * } \| \Delta _ { \Sigma , 2 } ) , } \end{array}
$$

and

$$
\| \Sigma ^ { 1 / 2 } ( w _ { \lambda , \gamma } - w ^ { \ast } ) \| \leq \lambda B ( \lambda , \gamma ) ,
$$

which together yields

$$
\begin{array} { r } { \| \Sigma ^ { 1 / 2 } ( \widehat { w } - w ^ { * } ) \| \leq \Delta _ { \Sigma , 1 } ^ { 2 } ( \Delta _ { V } + \left\| w ^ { * } \right\| \Delta _ { \Sigma , 2 } ) + \lambda B ( \lambda , \gamma ) . } \end{array}
$$

## A.3 High-probability bounds for the error decomposition terms $\Delta _ { \Sigma , 1 }$ $\Delta _ { \Sigma , 2 }$ , and $\Delta _ { V }$

We next control the random quantities appearing in the error decomposition of Lemma A.4. Compared with the standard proof for kernel ridge regression, the only additional dificulty is the presence of the derivative term and the need to keep track of its dependence on γ. This leads us to decompose both covariance and moment deviations into value and derivative contributions. The remainder of the argument then follows the standard kernel ridge regression proof strategy: we derive high-probability bounds for $\Delta _ { V }$ and $\Delta _ { \Sigma , 2 }$ , and control the multiplicative factor $\Delta _ { \Sigma , 1 }$ through the auxiliary normalized covariance deviation

$$
\Delta _ { \Sigma , 3 } : = \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \Sigma _ { \gamma } - \widehat \Sigma _ { \gamma } ) ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } \big \| .
$$

The key point is that when $\Delta _ { \Sigma , 3 } < 1$ , the empirical covariance operator remains well conditioned relative to $\Sigma _ { \gamma } + \lambda I$ , which yields a bound on $\Delta _ { \Sigma , 1 }$

Lemma A.5 (High-probability bound on $\Delta _ { \Sigma , 3 } )$ . We define

$$
\Delta _ { \Sigma , 3 } : = \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \Sigma _ { \gamma } - \widehat \Sigma _ { \gamma } ) ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } \big \| .
$$

Let $\delta \in ( 0 , 1 )$ and $\lambda > 0$ satisfy

$$
\frac { 3 6 } { n } \log \frac { n } { \delta } \leq \lambda \leq \| \Sigma \| _ { \infty } , { \frac { 3 6 } { m } } \log \frac { m } { \delta } \leq \lambda \gamma ^ { - 1 } \leq \| \Sigma _ { D } \| _ { \infty } .
$$

Then, with probability at least $1 - 2 \delta$ 2

$$
\Delta _ { \Sigma , 3 } ~ \leq ~ \frac { 1 } { 2 } .
$$

Proof. We first relate $\Delta _ { \Sigma , 3 }$ to the value and derivative parts separately. Using the decomposition

$$
\Sigma _ { \gamma } - \widehat \Sigma _ { \gamma } = ( \Sigma - \widehat \Sigma ) + \gamma ( \Sigma _ { D } - \widehat \Sigma _ { D } ) ,
$$

and the triangle inequality, we obtain

$$
\begin{array} { r l } & { \Delta _ { \Sigma , 3 } = \Big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \Sigma _ { \gamma } - \widehat \Sigma _ { \gamma } ) ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } \Big \| } \\ & { \qquad \leq \Big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \Sigma - \widehat \Sigma ) ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } \Big \| + \gamma \Big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \Sigma _ { D } - \widehat \Sigma _ { D } ) ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } \Big \| . } \end{array}
$$

For the first term, write

$$
( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \Sigma - \widehat \Sigma ) ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } = A ( \Sigma + \lambda I ) ^ { - 1 / 2 } ( \Sigma - \widehat \Sigma ) ( \Sigma + \lambda I ) ^ { - 1 / 2 } A ^ { * } ,
$$

where

$$
A : = ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \Sigma + \lambda I ) ^ { 1 / 2 } .
$$

From $\Sigma \preccurlyeq \Sigma _ { \gamma }$ we have $( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 } \preccurlyeq ( \Sigma + \lambda I ) ^ { - 1 }$ , hence $\| A \| \leq 1$ . Therefore

$$
\begin{array} { r } { \left\| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \Sigma - \widehat \Sigma ) ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } \right\| \leq \left\| ( \Sigma + \lambda I ) ^ { - 1 / 2 } ( \Sigma - \widehat \Sigma ) ( \Sigma + \lambda I ) ^ { - 1 / 2 } \right\| . } \end{array}
$$

For the second term we argue analogously, but comparing with $\gamma \Sigma _ { D }$ . Using $\gamma \Sigma _ { D } \preccurlyeq \Sigma _ { \gamma }$ we obtain

$$
\begin{array} { r } { \Big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \Sigma _ { D } - \widehat \Sigma _ { D } ) ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } \Big \| \leq \big \| ( \gamma \Sigma _ { D } + \lambda I ) ^ { - 1 / 2 } ( \Sigma _ { D } - \widehat \Sigma _ { D } ) ( \gamma \Sigma _ { D } + \lambda I ) ^ { - 1 / 2 } \big \| , } \end{array}
$$

and hence

$$
\begin{array} { r } { \Delta _ { \Sigma , 3 } \leq \| ( \Sigma + \lambda I ) ^ { - 1 / 2 } ( \Sigma - \widehat \Sigma ) ( \Sigma + \lambda I ) ^ { - 1 / 2 } \| + \gamma \| ( \gamma \Sigma _ { D } + \lambda I ) ^ { - 1 / 2 } ( \Sigma _ { D } - \widehat \Sigma _ { D } ) ( \gamma \Sigma _ { D } + \lambda I ) ^ { - 1 / 2 } \| . } \end{array}
$$

We now control each term probabilistically using Lemma 3.6 in [59]. Although the lemma yields a bound of $1 / 2 ,$ this constant can be reduced by rescaling the regularization parameter and applying the same proof. In particular, applying the proof of Lemma 3.6 with target bound $1 / 4$ , and using the admissibility conditions

$$
\frac { 3 6 } { n } \log \frac { n } { \delta } \leq \lambda \leq \| \Sigma \| _ { \infty } , \gamma \frac { 3 6 } { m } \log \frac { m } { \delta } \leq \lambda \leq \gamma \| \Sigma _ { D } \| _ { \infty } ,
$$

we obtain that, with probability at least $1 - \delta .$

$$
\begin{array} { r } { \big \| ( \Sigma + \lambda I ) ^ { - 1 / 2 } ( \Sigma - \widehat \Sigma ) ( \Sigma + \lambda I ) ^ { - 1 / 2 } \big \| \ < \ \frac 1 4 , } \end{array}
$$

and, again with probability at least $1 - \delta .$

$$
\begin{array} { r } { \gamma \big \| ( \gamma \Sigma _ { D } + \lambda I ) ^ { - 1 / 2 } ( \Sigma _ { D } - \widehat \Sigma _ { D } ) ( \gamma \Sigma _ { D } + \lambda I ) ^ { - 1 / 2 } \big \| \ < \ \frac { 1 } { 4 } . } \end{array}
$$

Here the improvement from $\textstyle { \frac { 1 } { 2 } }$ to $\textstyle { \frac { 1 } { 4 } }$ comes from running the same concentration proof with the smaller target threshold $\textstyle { \frac { 1 } { 4 } } .$ , at the cost of replacing the constant 9 in the lower admissibility condition by 36.

By a union bound, both events hold simultaneously with probability at least $1 - 2 \delta$ , and on this event we have

$$
\begin{array} { r } { \Delta _ { \Sigma , 3 } \ \leq \ \frac { 1 } { 4 } + \frac { 1 } { 4 } \ = \ \frac { 1 } { 2 } . } \end{array}
$$

This concludes the proof.

Lemma A.6 (Bound on $\Delta _ { \Sigma , 1 } )$ . Recall

$$
\Delta _ { \Sigma , 1 } : = \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { 1 / 2 } ( \widehat { \Sigma } _ { \gamma } + \lambda I ) ^ { - 1 / 2 } \big \| .
$$

We have

$$
\Delta _ { \Sigma , 1 } \leq ( 1 - \Delta _ { \Sigma , 3 } ) ^ { - 1 / 2 } .
$$

Proof. We have

$$
\Delta _ { \Sigma , 1 } ^ { 2 } = \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { 1 / 2 } ( \widehat { \Sigma } _ { \gamma } + \lambda I ) ^ { - 1 } ( \Sigma _ { \gamma } + \lambda I ) ^ { 1 / 2 } \big \| .
$$

Define

$$
\widehat { B } : = ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \Sigma _ { \gamma } - \widehat { \Sigma } _ { \gamma } ) ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } .
$$

By definition of $\Delta _ { \Sigma , 3 }$

$$
\| \widehat { B } \| = \Delta _ { \Sigma , 3 } .
$$

We can rewrite

$$
\widehat { \Sigma } _ { \gamma } + \lambda I = \Sigma _ { \gamma } + \lambda I - ( \Sigma _ { \gamma } - \widehat { \Sigma } _ { \gamma } ) = ( \Sigma _ { \gamma } + \lambda I ) ^ { 1 / 2 } ( I - \widehat { B } ) ( \Sigma _ { \gamma } + \lambda I ) ^ { 1 / 2 } ,
$$

so

$$
( \widehat { \Sigma } _ { \gamma } + \lambda I ) ^ { - 1 } = ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( I - \widehat { B } ) ^ { - 1 } ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } .
$$

Plugging this into the expression for $\Delta _ { \Sigma , 1 } ^ { 2 }$ gives

$$
\Delta _ { \Sigma , 1 } ^ { 2 } = \| ( I - \widehat { B } ) ^ { - 1 } \| .
$$

$\mathrm { I f } \ \| \widehat { B } \| = \Delta _ { \Sigma , 3 } < 1$ , then the spectrum of $\widehat { B }$ lies in $[ - \Delta _ { \Sigma , 3 } , \Delta _ { \Sigma , 3 } ]$ , hence the spectrum of $( I - \widehat { B } ) ^ { - 1 }$ is contained in $\{ ( 1 - \mu ) ^ { - 1 } : | \mu | \leq \Delta _ { \Sigma , 3 } \}$ , and therefore

$$
\Delta _ { \Sigma , 1 } ^ { 2 } = \Vert ( I - \widehat { B } ) ^ { - 1 } \Vert \leq ( 1 - \Vert \widehat { B } \Vert ) ^ { - 1 } = ( 1 - \Delta _ { \Sigma , 3 } ) ^ { - 1 } .
$$

Lemma A.7 (High-probability bound on $\Delta _ { V } )$ . Recall

$$
\Delta _ { V } : = \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \widehat { V } _ { \gamma } - V _ { \gamma } ) \big \| .
$$

Then

$$
\Delta _ { V } \leq \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \widehat { V } - V ) \big \| + \gamma \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \widehat { V } _ { D } - V _ { D } ) \big \| .
$$

For any $\delta \in ( 0 , 1 )$ , with probability at least $1 - 2 \delta$ 2

$$
\begin{array} { c } { \displaystyle \Delta _ { V } \leq 2 \beta _ { V } \frac { \log ( 2 / \delta ) } { n } + \sqrt { 2 } \sigma _ { V } \sqrt { \frac { \log ( 2 / \delta ) } { n } } } \\ { \displaystyle + \gamma \Bigg ( 2 \beta _ { D } \frac { \log ( 2 / \delta ) } { m } + \sqrt { 2 } \sigma _ { D } \sqrt { \frac { \log ( 2 / \delta ) } { m } } \Bigg ) , } \end{array}
$$

where the explicit constants are

$$
\begin{array} { r } { \beta _ { V } : = 2 \lambda ^ { - 1 / 2 } L _ { V } \kappa _ { V } , \qquad \sigma _ { V } ^ { 2 } : = L _ { V } ^ { 2 } \kappa _ { V } ^ { 2 } \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } \Sigma ^ { 1 / 2 } \big \| _ { \mathrm { H S } } ^ { 2 } , } \end{array}
$$

$$
\beta _ { D } : = 2 \lambda ^ { - 1 / 2 } L _ { D } \kappa _ { D } , \quad \quad \sigma _ { D } ^ { 2 } : = L _ { D } ^ { 2 } \kappa _ { D } ^ { 2 } \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } \Sigma _ { D } ^ { 1 / 2 } \big \| _ { \mathrm { H S } } ^ { 2 } .
$$

Proof. The decomposition

$$
{ \widehat V } _ { \gamma } - { V } _ { \gamma } = ( { \widehat V } - V ) + \gamma ( { \widehat V } _ { D } - V _ { D } )
$$

immediately gives

$$
\Delta _ { V } \leq \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \widehat { V } - V ) \big \| + \gamma \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \widehat { V } _ { D } - V _ { D } ) \big \| .
$$

Value part. Write

$$
( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \widehat { V } - V ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } M ( x _ { i } , y _ { i } ) , \qquad M ( x , y ) : = ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( y \phi ( x ) - V ) .
$$

Since $| y | \le L _ { V }$ and $\| \phi ( x ) \| \leq \kappa _ { V }$

$$
\| y \phi ( x ) - V \| \leq L _ { V } \kappa _ { V } + \| V \| \leq 2 L _ { V } \kappa _ { V } ,
$$

hence

$$
\| M ( x , y ) \| \leq \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } \| 2 L _ { V } \kappa _ { V } \leq 2 \lambda ^ { - 1 / 2 } L _ { V } \kappa _ { V } = : \beta _ { V } .
$$

The second moment satisfies

$$
\begin{array} { r } { \mathbb { E } \| M ( x , y ) \| ^ { 2 } \leq L _ { V } ^ { 2 } \kappa _ { V } ^ { 2 } \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } \Sigma ^ { 1 / 2 } \| _ { \mathrm { H S } } ^ { 2 } = : \sigma _ { V } ^ { 2 } . } \end{array}
$$

Applying Hilbert-space Bernstein (Prop. 7.15 of [60]) gives, with probability at least $1 - \delta .$

$$
\big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \widehat { V } - V ) \big \| \le 2 \beta _ { V } \frac { \log ( 2 / \delta ) } { n } + \sqrt { 2 } \sigma _ { V } \sqrt { \frac { \log ( 2 / \delta ) } { n } } .
$$

Derivative part. Exactly the same argument with

$$
M _ { D } ( z , d ) : = ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( d D \phi ( z ) - V _ { D } )
$$

yields

$$
\| M _ { D } ( z , d ) \| \le 2 \lambda ^ { - 1 / 2 } L _ { D } \kappa _ { D } = : \beta _ { D } ,
$$

and

$$
\sigma _ { D } ^ { 2 } = L _ { D } ^ { 2 } \kappa _ { D } ^ { 2 } \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } \Sigma _ { D } ^ { 1 / 2 } \| _ { \mathrm { H S } } ^ { 2 } .
$$

Bernstein again gives, with probability at least $1 - \delta _ { \colon }$

$$
\big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \widehat { V } _ { D } - V _ { D } ) \big \| \le 2 \beta _ { D } \frac { \log ( 2 / \delta ) } { m } + \sqrt { 2 } \sigma _ { D } \sqrt { \frac { \log ( 2 / \delta ) } { m } } .
$$

Conclusion. A union bound yields the stated bound with probability at least $1 - 2 \delta$ □

Lemma A.8 (High-probability bound on $\Delta _ { \Sigma , 2 } )$ . Recall

$$
\Delta _ { \Sigma , 2 } : = \bigl \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \Sigma _ { \gamma } - \widehat { \Sigma } _ { \gamma } ) \bigr \| , \qquad \Sigma _ { \gamma } = \Sigma + \gamma \Sigma _ { D } , \quad \widehat { \Sigma } _ { \gamma } = \widehat { \Sigma } + \gamma \widehat { \Sigma } _ { D } .
$$

Then

$$
\Delta _ { \Sigma , 2 } \ \leq \ \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \Sigma - \widehat \Sigma ) \big \| \ + \ \gamma \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \Sigma _ { D } - \widehat \Sigma _ { D } ) \big \| .
$$

Moreover, for any $\delta \in ( 0 , 1 )$ , with probability at least $1 - 2 \delta$

$$
\Delta _ { \Sigma , 2 } \leq 2 \beta _ { \Sigma } \frac { \log ( 2 / \delta ) } { n } + \sqrt { 2 } \sigma _ { \Sigma } \sqrt { \frac { \log ( 2 / \delta ) } { n } } + \gamma \Bigg ( 2 \beta _ { \Sigma _ { D } } \frac { \log ( 2 / \delta ) } { m } + \sqrt { 2 } \sigma _ { \Sigma _ { D } } \sqrt { \frac { \log ( 2 / \delta ) } { m } } \Bigg ) ,
$$

where

$$
\beta _ { \Sigma } : = \lambda ^ { - 1 / 2 } 2 \kappa _ { V } ^ { 2 } , \qquad \sigma _ { \Sigma } ^ { 2 } : = 4 \kappa _ { V } ^ { 2 } \left\| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } { \Sigma ^ { 1 / 2 } } \right\| _ { \mathrm { H S } } ^ { 2 } ,
$$

$$
\beta _ { \Sigma _ { D } } : = \lambda ^ { - 1 / 2 } 2 \kappa _ { D } ^ { 2 } , \qquad \sigma _ { \Sigma _ { D } } ^ { 2 } : = 4 \kappa _ { D } ^ { 2 } \left. ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } \Sigma _ { D } ^ { 1 / 2 } \right. _ { \mathrm { H S } } ^ { 2 } .
$$

Proof. The decomposition

$$
\Sigma _ { \gamma } - \widehat \Sigma _ { \gamma } = ( \Sigma - \widehat \Sigma ) + \gamma ( \Sigma _ { D } - \widehat \Sigma _ { D } )
$$

and the triangle inequality give

$$
\begin{array} { r l } & { \Delta _ { \Sigma , 2 } = \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \Sigma _ { \gamma } - \widehat \Sigma _ { \gamma } ) \big \| } \\ & { \qquad \leq \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \Sigma - \widehat \Sigma ) \big \| + \gamma \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \Sigma _ { D } - \widehat \Sigma _ { D } ) \big \| . } \end{array}
$$

We now bound these two terms separately, with the same Hilbert-space Bernstein argument as in Lemma $\mathrm { A . 7 } _ { }$ , but now in the Hilbert space of Hilbert-Schmidt operators and with $y \phi ( x )$ replaced by $\phi ( x ) \otimes \phi ( x )$

Value part. Recall

$$
\Sigma = \mathbb { E } [ \phi ( x ) \otimes \phi ( x ) ] , \qquad { \widehat { \Sigma } } = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \phi ( x _ { i } ) \otimes \phi ( x _ { i } ) .
$$

Define

$$
U ( x ) : = \Sigma - \phi ( x ) \otimes \phi ( x ) , \qquad M _ { \Sigma } ( x ) : = ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } U ( x ) .
$$

Then $\mathbb { E } [ U ( x ) ] = 0$ , and

$$
( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \Sigma - \widehat \Sigma ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } M _ { \Sigma } ( x _ { i } ) .
$$

We view $M _ { \Sigma } ( x )$ as a random element of the Hilbert space of Hilbert-Schmidt operators. We first bound its norm. Using the boundedness $\| \phi ( x ) \| \leq \kappa _ { V }$ we have

$$
\| \phi ( \boldsymbol { x } ) \otimes \phi ( \boldsymbol { x } ) \| _ { \mathrm { H S } } = \| \phi ( \boldsymbol { x } ) \| ^ { 2 } \leq \kappa _ { V } ^ { 2 } , \qquad \| \Sigma \| _ { \mathrm { H S } } \leq \mathbb { E } \| \phi ( \boldsymbol { x } ) \otimes \phi ( \boldsymbol { x } ) \| _ { \mathrm { H S } } \leq \kappa _ { V } ^ { 2 } .
$$

Thus

$$
\begin{array} { r } { \| U ( x ) \| _ { \mathrm { H S } } \leq \| \phi ( x ) \otimes \phi ( x ) \| _ { \mathrm { H S } } + \| \Sigma \| _ { \mathrm { H S } } \leq 2 \kappa _ { V } ^ { 2 } , } \end{array}
$$

and since $\| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } \| \le \lambda ^ { - 1 / 2 }$

$$
\begin{array} { r } { \| M _ { \Sigma } ( x ) \| _ { \mathrm { H S } } \leq \lambda ^ { - 1 / 2 } 2 \kappa _ { V } ^ { 2 } = \beta _ { \Sigma } . } \end{array}
$$

Next, we bound the second moment. Using that $\Sigma _ { \gamma } + \lambda I$ is self-adjoint and positive, we have

$$
\begin{array} { r } { \Vert M _ { \Sigma } ( x ) \Vert _ { \mathrm { H S } } ^ { 2 } = \left. ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } U ( x ) \right. _ { \mathrm { H S } } ^ { 2 } = \mathrm { T r } \Big ( U ( x ) ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 } U ( x ) \Big ) . } \end{array}
$$

Take expectation and use linearity of the trace:

$$
\begin{array} { r } { \mathbb { E } \| M _ { \Sigma } ( x ) \| _ { \mathrm { H S } } ^ { 2 } = \mathrm { T r } \Big ( ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 } \mathbb { E } \big [ U ( x ) ^ { 2 } \big ] \Big ) . } \end{array}
$$

We now bound $\mathbb { E } [ U ( x ) ^ { 2 } ]$ in terms of Σ. First note that

$$
U ( x ) ^ { 2 } = { \bigl ( } \Sigma - \phi ( x ) \otimes \phi ( x ) { \bigr ) } ^ { 2 } \preccurlyeq 2 \Sigma ^ { 2 } + 2 { \bigl ( } \phi ( x ) \otimes \phi ( x ) { \bigr ) } ^ { 2 } ,
$$

by the elementary operator inequality $( A - B ) ^ { 2 } = A ^ { 2 } + B ^ { 2 } - A B - B A \prec 2 A ^ { 2 } + 2 B ^ { 2 }$ for self-adjoint $A , B$

Moreover,

$$
\bigl ( \phi ( x ) \otimes \phi ( x ) \bigr ) ^ { 2 } = \| \phi ( x ) \| ^ { 2 } \phi ( x ) \otimes \phi ( x ) \prec \kappa _ { V } ^ { 2 } \phi ( x ) \otimes \phi ( x ) ,
$$

and $\begin{array} { r } { \Sigma ^ { 2 } \prec \| \Sigma \| \Sigma \prec \kappa _ { V } ^ { 2 } \Sigma . } \end{array}$ , since $\| \Sigma \| \le \kappa _ { V } ^ { 2 }$ for a bounded feature map. Combining these,

$$
U ( x ) ^ { 2 } \precsim 2 \kappa _ { V } ^ { 2 } \phi ( x ) \otimes \phi ( x ) + 2 \kappa _ { V } ^ { 2 } \Sigma .
$$

Taking expectations yields

$$
\begin{array} { r } { \mathbb { E } [ U ( x ) ^ { 2 } ] \preccurlyeq 2 \kappa _ { V } ^ { 2 } \mathbb { E } [ \phi ( x ) \otimes \phi ( x ) ] + 2 \kappa _ { V } ^ { 2 } \Sigma = 4 \kappa _ { V } ^ { 2 } \Sigma . } \end{array}
$$

Plugging this into the expression for the second moment, we get

$$
\begin{array} { r l } & { \mathbb { E } \| M _ { \Sigma } ( x ) \| _ { \mathrm { H S } } ^ { 2 } \leq \mathrm { T r } \Big ( ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 } 4 \kappa _ { V } ^ { 2 } \Sigma \Big ) } \\ & { \qquad = 4 \kappa _ { V } ^ { 2 } \mathrm { T r } \Big ( ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } \Sigma ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } \Big ) } \\ & { \qquad = 4 \kappa _ { V } ^ { 2 } \left\| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } { \Sigma ^ { 1 / 2 } } \right\| _ { \mathrm { H S } } ^ { 2 } . } \end{array}
$$

Thus we set

$$
\sigma _ { \Sigma } ^ { 2 } : = 4 \kappa _ { V } ^ { 2 } \left\| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } \Sigma ^ { 1 / 2 } \right\| _ { \mathrm { H S } } ^ { 2 }
$$

and obtain

$$
\mathbb { E } \| M _ { \Sigma } ( x ) \| _ { \mathrm { H S } } ^ { 2 } \leq \sigma _ { \Sigma } ^ { 2 } .
$$

We can now apply the Hilbert-space Bernstein inequality (Proposition 7.15 in [60]) in the Hilbert space of Hilbert-Schmidt operators to the empirical average $\textstyle { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } M _ { \Sigma } ( x _ { i } )$ . This gives: for any $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta$ ，

$$
\big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \Sigma - \widehat \Sigma ) \big \| _ { \mathrm { H S } } \ \leq \ 2 \beta _ { \Sigma } \frac { \log ( 2 / \delta ) } { n } + \sqrt 2 \sigma _ { \Sigma } \sqrt { \frac { \log ( 2 / \delta ) } { n } } .
$$

Since $\| T \| \leq \| T \| _ { \mathrm { H S } }$ for every Hilbert-Schmidt operator $T ,$ , the same bound holds for

$$
\big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \Sigma - \widehat { \Sigma } ) \big \| .
$$

Derivative part. The argument is identical with

$$
\Sigma _ { D } = \mathbb { E } [ D \phi ( z ) \otimes D \phi ( z ) ] , \qquad { \widehat { \Sigma } } _ { D } = { \frac { 1 } { m } } \sum _ { j = 1 } ^ { m } D \phi ( z _ { j } ) \otimes D \phi ( z _ { j } ) .
$$

Define

$$
U _ { D } ( z ) : = \Sigma _ { D } - D \phi ( z ) \otimes D \phi ( z ) , \qquad M _ { \Sigma _ { D } } ( z ) : = ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } U _ { D } ( z ) .
$$

Then

$$
( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \Sigma _ { D } - \widehat \Sigma _ { D } ) = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } M _ { \Sigma _ { D } } ( z _ { j } ) ,
$$

with $\mathbb { E } [ M _ { \Sigma _ { D } } ( z ) ] = 0$

Using $\| D \phi ( z ) \| \le \kappa _ { D }$ we obtain

$$
\| U _ { D } ( z ) \| _ { \mathrm { H S } } \le 2 \kappa _ { D } ^ { 2 } , \qquad \| M _ { \Sigma _ { D } } ( z ) \| _ { \mathrm { H S } } \le \lambda ^ { - 1 / 2 } 2 \kappa _ { D } ^ { 2 } = \beta _ { \Sigma _ { D } } ,
$$

and, by the same operator inequalities as above,

$$
\begin{array} { r } { \mathbb { E } \| M _ { \Sigma _ { D } } ( z ) \| _ { \mathrm { H S } } ^ { 2 } \leq 4 \kappa _ { D } ^ { 2 } \left\| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } { \Sigma _ { D } ^ { 1 / 2 } } \right\| _ { \mathrm { H S } } ^ { 2 } = : \sigma _ { \Sigma _ { D } } ^ { 2 } . } \end{array}
$$

Another application of the Hilbert-space Bernstein inequality in the Hilbert space of Hilbert-Schmidt operators yields, with probability at least $1 - \delta$ ，

$$
\big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \Sigma _ { D } - \widehat \Sigma _ { D } ) \big \| _ { \mathrm { H S } } \ \le \ 2 \beta _ { \Sigma _ { D } } \frac { \log ( 2 / \delta ) } { m } + \sqrt 2 \sigma _ { \Sigma _ { D } } \sqrt { \frac { \log ( 2 / \delta ) } { m } } .
$$

Again the same bound holds for

$$
\begin{array} { r } { \left\| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \Sigma _ { D } - \widehat { \Sigma } _ { D } ) \right\| . } \end{array}
$$

Conclusion. We have shown that, with probability at least $1 - \delta .$

$$
\big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \Sigma - \widehat \Sigma ) \big \| \leq 2 \beta _ { \Sigma } \frac { \log ( 2 / \delta ) } { n } + \sqrt 2 \sigma _ { \Sigma } \sqrt { \frac { \log ( 2 / \delta ) } { n } } ,
$$

and, with probability at least $1 - \delta$ ，

$$
\bigl \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \Sigma _ { D } - \widehat \Sigma _ { D } ) \bigr \| \leq 2 \beta _ { \Sigma _ { D } } \frac { \log ( 2 / \delta ) } { m } + \sqrt 2 \sigma _ { \Sigma _ { D } } \sqrt { \frac { \log ( 2 / \delta ) } { m } } .
$$

By a union bound, both events hold simultaneously with probability at least $1 - 2 \delta$ , and combining these with the initial decomposition of $\Delta _ { \Sigma , 2 }$ yields the claimed high-probability bound. □

## A.4 Finite-sample bounds

Combining Lemma A.4 with the concentration results of the previous subsection yields the following finite-sample bound.

Theorem A.9 (High-probability finite-sample bounds for PIKS). Let $\lambda > 0 , \gamma > 0$ and let $\delta \in ( 0 , 1 / 2 )$ . Assume that

$$
\frac { 3 6 } { n } \log \frac { 2 n } { \delta } \leq \lambda \leq \| \Sigma \| _ { \infty } , { \frac { 3 6 } { m } } \log \frac { 2 m } { \delta } \leq \lambda \gamma ^ { - 1 } \leq \| \Sigma _ { D } \| _ { \infty } .\tag{5}
$$

Define

$$
\Sigma _ { \gamma } : = \Sigma + \gamma \Sigma _ { D } , \qquad B ( \lambda , \gamma ) : = \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } w ^ { * } \big \| ,
$$

and the (value and derivative) capacity terms

$$
\begin{array} { r } {  { \mathcal { N } } _ { V } ( \lambda , \gamma ) : = \bigl \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } \Sigma ^ { 1 / 2 } \bigr \| _ { \mathrm { H S } } , \qquad { \mathcal { N } } _ { D } ( \lambda , \gamma ) : = \bigl \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } \Sigma _ { D } ^ { 1 / 2 } \bigr \| _ { \mathrm { H S } } . } \end{array}
$$

Then there exists a constant $c > 0$ , depending only on $\kappa _ { V } , \kappa _ { D } , L _ { V } , L _ { D }$ and $\| w ^ { * } \|$ , but not on $\lambda , \gamma , n , m , \delta$ , such that, with probability at least $1 - 2 \delta$

$$
\big \| \Sigma ^ { 1 / 2 } ( \widehat w - w ^ { * } ) \big \| \ \leq \ c \bigg [ \lambda ^ { - 1 / 2 } \Big ( \frac { \log ( 4 / \delta ) } { n } + \gamma \frac { \log ( 4 / \delta ) } { m } \Big ) + \Big ( \frac { \mathcal { N } _ { V } ( \lambda , \gamma ) } { \sqrt { n } } + \gamma \frac { \mathcal { N } _ { D } ( \lambda , \gamma ) } { \sqrt { m } } \Big ) \sqrt { \log \frac { 4 } { \delta } } + \lambda B ( \lambda , \gamma ) \bigg ] .\tag{6}
$$

Proof. From the error decomposition Lemma A.4, we have

$$
\| \Sigma ^ { 1 / 2 } ( \widehat { w } - w ^ { * } ) \| \ \le \ \Delta _ { \Sigma , 1 } ^ { 2 } \big ( \Delta _ { V } + \| w ^ { * } \| \Delta _ { \Sigma , 2 } \big ) \ + \ \lambda \mathcal { B } ( \lambda , \gamma ) ,
$$

where

$$
\begin{array} { r l } & { \Delta _ { \Sigma , 1 } : = \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { 1 / 2 } ( \widehat \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } \big \| , \quad \Delta _ { \Sigma , 2 } : = \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \Sigma _ { \gamma } - \widehat \Sigma _ { \gamma } ) \big \| , } \\ & { \qquad \Delta _ { V } : = \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \widehat V _ { \gamma } - V _ { \gamma } ) \big \| . } \end{array}
$$

Step 1: bound on $\Delta _ { \Sigma , 1 }$ . Recall

$$
\Delta _ { \Sigma , 3 } : = \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } ( \Sigma _ { \gamma } - \widehat \Sigma _ { \gamma } ) ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } \big \| .
$$

Applying Lemma A.5 with confidence parameter $\eta ,$ and the condition (3), we obtain an event of probability at least $1 - 2 \eta$ on which $\Delta _ { \Sigma , 3 } \leq 1 / 2$ . On this event, Lemma A.6 yields

$$
\Delta _ { \Sigma , 1 } ^ { 2 } \leq ( 1 - \Delta _ { \Sigma , 3 } ) ^ { - 1 } \leq 2 .
$$

Step 2: bounds on $\Delta _ { V }$ and $\Delta _ { \Sigma , 2 }$ . We use Lemmas $\mathrm { A . 7 }$ and $\mathrm { A } . 8 .$ From Lemma $\mathrm { A . 7 } _ { }$ , for any $\eta \in ( 0 , 1 )$ , with probability at least $1 - 2 \eta$

$$
\Delta _ { V } \ \leq \ \underbrace { \Big ( 2 \beta _ { V } \frac { \log ( 2 / \eta ) } { n } + \sqrt { 2 } \sigma _ { V } \sqrt { \frac { \log ( 2 / \eta ) } { n } } \Big ) } _ { \mathrm { v a l u e ~ p a r t } } + \gamma \underbrace { \Big ( 2 \beta _ { D } \frac { \log ( 2 / \eta ) } { m } + \sqrt { 2 } \sigma _ { D } \sqrt { \frac { \log ( 2 / \eta ) } { m } } \Big ) } _ { \mathrm { d e r i v a t i v e ~ p a r t } } ,
$$

with

$$
\beta _ { V } = \lambda ^ { - 1 / 2 } 2 L _ { V } \kappa _ { V } , \qquad \sigma _ { V } ^ { 2 } = L _ { V } ^ { 2 } \kappa _ { V } ^ { 2 } \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } { \Sigma ^ { 1 / 2 } \big \| _ { \mathrm { H S } } ^ { 2 } } ,
$$

$$
\beta _ { D } = \lambda ^ { - 1 / 2 } 2 L _ { D } \kappa _ { D } , \qquad \sigma _ { D } ^ { 2 } = L _ { D } ^ { 2 } \kappa _ { D } ^ { 2 } \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } \Sigma _ { D } ^ { 1 / 2 } \big \| _ { \mathrm { H S } } ^ { 2 } .
$$

Hence there exist constants $a _ { V } , b _ { V } , a _ { D } , b _ { D } > 0$ depending only on $\left( \kappa _ { V } , L _ { V } \right)$ and $\left( \kappa _ { D } , L _ { D } \right)$ such that

$$
\Delta _ { V } \ \leq \ \lambda ^ { - 1 / 2 } \Big ( a _ { V } \frac { \log ( 2 / \eta ) } { n } + \gamma a _ { D } \frac { \log ( 2 / \eta ) } { m } \Big ) + \Big ( b _ { V } \mathcal { N } _ { V } ( \lambda , \gamma ) \sqrt { \frac { \log ( 2 / \eta ) } { n } } + \gamma b _ { D } \mathcal { N } _ { D } ( \lambda , \gamma ) \sqrt { \frac { \log ( 2 / \eta ) } { m } } \Big ) .
$$

Similarly, from Lemma $_ { \mathrm { A . 8 , } }$ for the same confidence parameter, with probability at least $1 - 2 \eta$

$$
\Delta _ { \Sigma , 2 } \leq \lambda ^ { - 1 / 2 } \Big ( \widetilde a _ { V } \frac { \log ( 2 / \eta ) } { n } + \gamma \widetilde a _ { D } \frac { \log ( 2 / \eta ) } { m } \Big ) + \Big ( \widetilde b _ { V } \mathcal { N } _ { V } ( \lambda , \gamma ) \sqrt { \frac { \log ( 2 / \eta ) } { n } } + \gamma \widetilde b _ { D } \mathcal { N } _ { D } ( \lambda , \gamma ) \sqrt { \frac { \log ( 2 / \eta ) } { m } } \Big ) ,
$$

for some constants $\tilde { a } _ { V } , \tilde { a } _ { D } , \tilde { b } _ { V } , \tilde { b } _ { D } > 0$ depending only on $\kappa _ { V }$ and $\kappa _ { D } .$

Combining these two bounds, we obtain constants $A _ { 1 } , A _ { 2 } , B _ { 1 } , B _ { 2 } \ > \ 0$ , depending only on $\left( \kappa _ { V } , \kappa _ { D } , L _ { V } , L _ { D } \right)$ , such that

$$
\Delta _ { V } \ \leq \ \lambda ^ { - 1 / 2 } \Big ( A _ { 1 } \frac { \log ( 2 / \eta ) } { n } + A _ { 2 } \gamma \frac { \log ( 2 / \eta ) } { m } \Big ) + \Big ( B _ { 1 } N _ { V } ( \lambda , \gamma ) \sqrt { \frac { \log ( 2 / \eta ) } { n } } + B _ { 2 } \gamma N _ { D } ( \lambda , \gamma ) \sqrt { \frac { \log ( 2 / \eta ) } { m } } \Big ) ,\tag{7}
$$

and

$$
\Delta _ { \Sigma , 2 } \leq \lambda ^ { - 1 / 2 } \Big ( A _ { 1 } \frac { \log ( 2 / \eta ) } { n } + A _ { 2 } \gamma \frac { \log ( 2 / \eta ) } { m } \Big ) + \Big ( B _ { 1 } \Lambda _ { V } ( \lambda , \gamma ) \sqrt { \frac { \log ( 2 / \eta ) } { n } } + B _ { 2 } \gamma \Lambda _ { D } ( \lambda , \gamma ) \sqrt { \frac { \log ( 2 / \eta ) } { m } } \Big ) ,\tag{8}
$$

where we have simply taken maxima of the corresponding constants from $\Delta _ { V }$ and $\Delta _ { \Sigma , 2 }$

Step 3: combine all bounds. Set $\eta : = \delta / 3$ and apply Lemmas A.5, A.7, and A.8 with confidence parameter η. Each holds with probability at least $1 - 2 \eta$ , so by a union bound, with probability at least

$$
1 - 3 \cdot 2 \eta = 1 - 2 \delta ,
$$

we simultaneously have

$$
\Delta _ { \Sigma , 1 } ^ { 2 } \leq 2 , \quad \mathrm { a n d \ t h e \ b o u n d s \ ( 7 ) } { - } ( 8 ) \ \mathrm { f o r } \ \Delta _ { V } , \Delta _ { \Sigma , 2 } .
$$

On this event,

$$
\begin{array} { r l } & { \| \Sigma ^ { 1 / 2 } ( \widehat { w } - w ^ { * } ) \| \leq \Delta _ { \Sigma , 1 } ^ { 2 } \big ( \Delta _ { V } + \| w ^ { * } \| \Delta _ { \Sigma , 2 } \big ) + \lambda \mathcal { B } ( \lambda , \gamma ) } \\ & { \qquad \leq 2 \big ( \Delta _ { V } + \| w ^ { * } \| \Delta _ { \Sigma , 2 } \big ) + \lambda \mathcal { B } ( \lambda , \gamma ) . } \end{array}
$$

Substituting the bounds (7)-(8), we obtain

$$
\begin{array} { r l r } {  { \| \Sigma ^ { 1 / 2 } ( \widehat { w } - w ^ { * } ) \| \le 2 ( 1 + \| w ^ { * } \| ) [ \lambda ^ { - 1 / 2 } \Big ( A _ { 1 } \frac { \log ( 2 / \eta ) } { n } + A _ { 2 } \gamma \frac { \log ( 2 / \eta ) } { m } \Big )  } } \\ & { } & { \quad \quad \quad \quad \quad \quad \quad  + ( B _ { 1 } \Lambda _ { V } ( \lambda , \gamma ) \sqrt { \frac { \log ( 2 / \eta ) } { n } } + B _ { 2 } \gamma \Lambda _ { D } ( \lambda , \gamma ) \sqrt { \frac { \log ( 2 / \eta ) } { m } } ) ] + \lambda B ( \lambda , \gamma ) . } \end{array}
$$

Since $\eta = \delta / 3$ , we have $\log ( 2 / \eta ) = \log ( 6 / \delta ) \leq c _ { 1 } \log ( 4 / \delta )$ for some constant $c _ { 1 } > 0$ . Absorbing the constants $2 ( 1 + \| w ^ { * } \| ) A _ { 1 } , 2 ( 1 + \| w ^ { * } \| ) A _ { 2 } , 2 ( 1 + \| w ^ { * } \| ) B _ { 1 } , 2 ( 1 + \| w ^ { * } \| ) B _ { 2 }$ and $\sqrt { c _ { 1 } }$ into a single constant $c > 0$ depending only on $\kappa _ { V } , \kappa _ { D } , L _ { V } , L _ { D }$ and $\| w ^ { * } \|$ , we rewrite the bound as

$$
\| \Sigma ^ { 1 / 2 } ( \widehat w - w ^ { * } ) \| \ \leq \ c \bigg [ \lambda ^ { - 1 / 2 } \Big ( \frac { \log ( 4 / \delta ) } { n } + \gamma \frac { \log ( 4 / \delta ) } { m } \Big ) + \Big ( \frac { \mathcal { N } _ { V } ( \lambda , \gamma ) } { \sqrt { n } } + \gamma \frac { \mathcal { N } _ { D } ( \lambda , \gamma ) } { \sqrt { m } } \Big ) \sqrt { \log \frac { 4 } { \delta } } + \lambda B ( \lambda , \gamma ) \bigg ] ,
$$

which is exactly (4).

## A.5 Learning rate acceleration from diferential information

We now instantiate the finite-sample bound under the Assumption (A6) and optimize over $\lambda , \gamma$

Corollary A.1 (Learning rates for PIKS). Assume Assumptions $( A 1 ) – ( A 6 )$ hold. Then there exist choices of regularization parameters $\lambda = \lambda ( n , m )$ and $\gamma = \gamma ( n , m )$ , given explicitly in the proof and depending polynomially on n, m, log n, and log m, such that, with high probability:

$$
\begin{array} { r } { \| \Sigma ^ { 1 / 2 } ( \widehat { w } - w ^ { * } ) \| \lesssim \left\{ \begin{array} { l l } { n ^ { - \frac { 1 } { 4 } } m ^ { - \frac { 1 - r } { 8 } } , } & { m \lesssim n ^ { \frac { 2 ( 1 - \alpha ) } { ( 1 + \alpha ) ( 1 - r ) } } , } \\ { n ^ { - \frac { 1 } { 2 ( 1 + \alpha ) } } , } & { m \gtrsim n ^ { \frac { 2 ( 1 - \alpha ) } { ( 1 + \alpha ) ( 1 - r ) } } . } \end{array} \right. } \end{array}
$$

Proof. The proof proceeds in four steps.

Step 1: Capacity bounds. As a consequence of Assumption (A6), using $\Sigma _ { \gamma } + \lambda I \succeq \Sigma _ { 1 } + \lambda I$ and $\Sigma _ { \gamma } + \lambda I \succeq \gamma \Sigma _ { D } + \lambda I$ , we obtain

$$
\mathcal { N } _ { V } ( \lambda , \gamma ) ^ { 2 } = \operatorname { T r } \bigl ( \Sigma _ { 1 } ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 } \bigr ) + \operatorname { T r } \bigl ( \Sigma _ { 2 } ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 } \bigr ) \ \lesssim \ \lambda ^ { - \alpha } + \gamma ^ { r - 1 } \lambda ^ { - r } .\tag{9}
$$

Moreover, since $\Sigma _ { \gamma } + \lambda I \succeq \lambda I .$

$$
\begin{array} { r } { \mathcal { N } _ { D } ( \lambda , \gamma ) ^ { 2 } \leq \lambda ^ { - 1 } \operatorname { T r } ( \Sigma _ { D } ) \lesssim \lambda ^ { - 1 } . } \end{array}
$$

Step 2: Error bound. From Assumption (A4), the model is well specified, i.e. $\| w ^ { * } \| _ { \mathcal { H } } < + \infty$ Then

$$
\begin{array} { r } { \mathcal { B } ( \lambda , \gamma ) ^ { 2 } = \langle w ^ { * } , ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 } w ^ { * } \rangle \ \leq \ \lambda ^ { - 1 } \| w ^ { * } \| _ { \mathcal { H } } ^ { 2 } , \qquad \lambda \mathcal { B } ( \lambda , \gamma ) \ \lesssim \ \sqrt { \lambda } . } \end{array}
$$

Throughout the remainder of the proof, we restrict to regularization parameters satisfying

$$
\operatorname* { m a x } \biggl \{ \frac { \log n } { n } , \ \gamma \frac { \log m } { m } \biggr \} \ \stackrel { } { \sim } \ \lambda \ \stackrel { } { \sim } \ \gamma .
$$

This ensures the admissibility condition of Theorem 3.1. Under this regime, using ${ \sqrt { a + b } } \leq { \sqrt { a } } + { \sqrt { b } } .$ the finite-sample bound simplifies to

$$
\| \Sigma ^ { 1 / 2 } ( \widehat { w } - w ^ { * } ) \| \lesssim \frac { \lambda ^ { - \alpha / 2 } } { \sqrt { n } } + \frac { \gamma ^ { \frac { r - 1 } { 2 } } \lambda ^ { - r / 2 } } { \sqrt { n } } + \frac { \gamma \lambda ^ { - 1 / 2 } } { \sqrt { m } } + \sqrt { \lambda } .\tag{10}
$$

This bound makes explicit how derivative information can reduce the efective value complexity through the aligned component $\Sigma _ { 2 } .$ , while simultaneously introducing a variance cost that grows with $\gamma$ through the derivative observations.

Step 3: Optimization over $\gamma$ . For fixed $\lambda ,$ only the middle two terms in (10) depend on γ. Increasing γ reduces the variance associated with estimating the function values in directions constrained by the derivative operator (by shrinking the contribution of the corresponding eigenspaces), but simultaneously amplifies the variance coming from the noisy estimation of derivative information. The optimal choice of $\gamma$ is obtained by balancing these two efects by solving

$$
{ \frac { \gamma ^ { \frac { r - 1 } { 2 } } \lambda ^ { - r / 2 } } { \sqrt { n } } } = { \frac { \gamma \lambda ^ { - 1 / 2 } } { \sqrt { m } } } .
$$

This is equivalent to

$$
\gamma ^ { \frac { r - 3 } { 2 } } = \left( \frac { n } { m } \right) ^ { 1 / 2 } \lambda ^ { \frac { r - 1 } { 2 } } ,
$$

which yields

$$
\gamma ^ { \star } ( \lambda ) \asymp \Big ( \frac { m } { n } \Big ) ^ { \frac { 1 } { 3 - r } } \lambda ^ { \frac { 1 - r } { 3 - r } } .\tag{11}
$$

Substituting (11) into (10), the two γ–dependent terms become equal and we obtain

$$
\lVert \Sigma ^ { 1 / 2 } ( \widehat { w } - w ^ { * } ) \rVert \ \lesssim \ A ( \lambda ) \ + \ B ( \lambda ) \ + \ \sqrt { \lambda } ,\tag{12}
$$

where

$$
A ( \lambda ) : = \frac { \lambda ^ { - \alpha / 2 } } { \sqrt { n } } , \qquad B ( \lambda ) : = m ^ { - \frac { 1 - r } { 2 ( 3 - r ) } } n ^ { - \frac { 1 } { 3 - r } } \lambda ^ { - \frac { 1 + r } { 2 ( 3 - r ) } } .
$$

Substituting $\gamma ^ { \star } ( \lambda ) \asymp ( m / n ) ^ { \frac { 1 } { 3 - r } } \lambda ^ { \frac { 1 - r } { 3 - r } }$ into the working regime max{log n/n, γ log m/m} $\lesssim \lambda \lesssim \gamma$ yields the equivalent λ-only conditions

$$
\operatorname* { m a x } \biggr \{ \frac { \log n } { n } , ~ \Big ( \frac { m } { n } \Big ) ^ { \frac { 1 } { 2 } } \Big ( \frac { \log m } { m } \Big ) ^ { \frac { 3 - r } { 2 } } \biggr \} ~ \lesssim ~ \lambda ~ \lesssim ~ \sqrt { \frac { m } { n } } .
$$

Step $\it 4 .$ Optimization over λ and regime analysis. We choose λ by balancing the bias $\sqrt { \lambda }$ with one of the variance terms in (12).

(i) Balance with A(λ). We solve

$$
{ \sqrt { \lambda } } = { \frac { \lambda ^ { - \alpha / 2 } } { \sqrt { n } } } ,
$$

which gives

$$
\lambda _ { A } ^ { \star } \asymp n ^ { - \frac { 1 } { 1 + \alpha } } , \qquad \sqrt { \lambda _ { A } ^ { \star } } \asymp A ( \lambda _ { A } ^ { \star } ) \asymp n ^ { - \frac { 1 } { 2 ( 1 + \alpha ) } } .\tag{13}
$$

(ii) Balance with $B ( \lambda )$ . We solve

$$
{ \sqrt { \lambda } } = m ^ { - { \frac { 1 - r } { 2 ( 3 - r ) } } } n ^ { - { \frac { 1 } { 3 - r } } } \lambda ^ { - { \frac { 1 + r } { 2 ( 3 - r ) } } } ,
$$

which gives

$$
\lambda _ { B } ^ { \star } \asymp m ^ { - \frac { 1 - r } { 4 } } n ^ { - \frac { 1 } { 2 } } , \qquad \sqrt { \lambda _ { B } ^ { \star } } \asymp B ( \lambda _ { B } ^ { \star } ) \asymp n ^ { - \frac { 1 } { 4 } } m ^ { - \frac { 1 - r } { 8 } } .\tag{14}
$$

Validity of the regimes. The bound (12) is controlled by the dominant variance term at the chosen λ.

Saturation regime. We choose $\lambda = \lambda _ { A } ^ { \star }$ when

$$
B ( \lambda _ { A } ^ { \star } ) \ \lesssim \ A ( \lambda _ { A } ^ { \star } ) .
$$

Substituting $\lambda _ { A } ^ { \star } \asymp n ^ { - 1 / ( 1 + \alpha ) }$ into $B ( \lambda )$ gives

$$
B ( \lambda _ { A } ^ { \star } ) \asymp m ^ { - \frac { 1 - r } { 2 ( 3 - r ) } } n ^ { - \frac { 1 + 2 \alpha - r } { 2 ( 1 + \alpha ) ( 3 - r ) } } .
$$

Hence

$$
B ( \lambda _ { A } ^ { \star } ) \lesssim A ( \lambda _ { A } ^ { \star } )
$$

is equivalent to

$$
m \gtrsim n ^ { \frac { 2 ( 1 - \alpha ) } { ( 1 + \alpha ) ( 1 - r ) } } .
$$

Derivative-limited regime. We choose $\lambda = \lambda _ { B } ^ { \star }$ when

$$
A ( \lambda _ { B } ^ { \star } ) \ \lesssim \ B ( \lambda _ { B } ^ { \star } ) .
$$

which yields the same transition value for m.

Moreover, to ensure our working regime condition, we take the regularization parameter to be clipped as

$$
\lambda : = \operatorname* { m a x } \left\{ \lambda ^ { \star } , \frac { \log n } { n } , \gamma ^ { \star } ( \lambda ^ { \star } ) \frac { \log m } { m } \right\} ,
$$

where $\lambda ^ { \star } \in \{ \lambda _ { A } ^ { \star } , \lambda _ { B } ^ { \star } \}$ is the optimizer in the corresponding regime. One can check that this choice of λ satisfies the admissibility conditions, including the upper condition $\lambda \lesssim \gamma$ , in both regimes. Since the variance terms are nonincreasing in λ, clipping only increases the bias term, yielding an additional lower-order contribution of order $\sqrt { \log n / n } + \sqrt { \gamma \log m / m }$

Resulting rate. Let

$$
m _ { \mathrm { c r i t } } \ \asymp \ n ^ { \frac { 2 ( 1 - \alpha ) } { ( 1 + \alpha ) ( 1 - r ) } } .
$$

The optimized learning rate is

$$
\begin{array} { r } { \| \Sigma ^ { 1 / 2 } ( \widehat { w } - w ^ { * } ) \| \ \lesssim \ \left\{ \begin{array} { l l } { n ^ { - \frac { 1 } { 4 } } m ^ { - \frac { 1 - r } { 8 } } , } & { m \lesssim m _ { \mathrm { c r i t } } , } \\ { n ^ { - \frac { 1 } { 2 ( 1 + \alpha ) } } , } & { m \gtrsim m _ { \mathrm { c r i t } } . } \end{array} \right. } \end{array}
$$

## A.6 Learning rates in physically consistent norms

We finally show that the estimator also converges in the stronger norm induced by $\Sigma + \Sigma _ { D }$ , which jointly controls the prediction error and the error on the diferential quantities.

Corollary A.2 (Learning rates in the physically consistent norm). Assume Assumptions $( A 1 ) – ( A 6 )$ hold. Then there exist choices of regularization parameters $\lambda = \lambda ( n , m )$ and $\gamma = \gamma ( n , m )$ such that, with high probability,

$$
\begin{array} { r } { \| ( \Sigma + \Sigma _ { D } ) ^ { 1 / 2 } ( \widehat { w } - w ^ { * } ) \| \ \lesssim \ \left\{ \begin{array} { l l } { n ^ { - \frac { 1 } { 4 } } m ^ { - \frac { 1 - r } { 8 } } + m ^ { - 1 / 4 } , } & { m \lesssim m _ { \mathrm { c r i t } } , } \\ { n ^ { - \frac { 1 } { 2 ( 1 + \alpha ) } } + m ^ { - 1 / 4 } , } & { m \gtrsim m _ { \mathrm { c r i t } } , } \end{array} \right. } \end{array}
$$

where

$$
m _ { \mathrm { c r i t } } \ \asymp \ n ^ { \frac { 2 ( 1 - \alpha ) } { ( 1 + \alpha ) ( 1 - r ) } } .
$$

Proof. Since $\gamma \geq 1$ , we have

$$
\Sigma _ { \gamma } = \Sigma + \gamma \Sigma _ { D } \succeq \Sigma + \Sigma _ { D } ,
$$

and therefore

$$
\begin{array} { r } { \| ( \Sigma + \Sigma _ { D } ) ^ { 1 / 2 } ( \widehat { w } - w ^ { * } ) \| \ \leq \ \| \Sigma _ { \gamma } ^ { 1 / 2 } ( \widehat { w } - w ^ { * } ) \| . } \end{array}
$$

Finite-sample bound. Repeating the proof of Theorem 3.1 with $\Sigma ^ { 1 / 2 }$ replaced by $\Sigma _ { \gamma } ^ { 1 / 2 }$ yields the same finite-sample bound, since both the estimation and bias terms are handled exactly as before, but using $\| \Sigma _ { \gamma } ^ { 1 / 2 } ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } \| \le 1$ instead of $\| \Sigma ^ { 1 / 2 } ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } \| \le 1$ . Then, the same argument as in the proof of Corollary 3.1 gives

$$
\| \Sigma _ { \gamma } ^ { 1 / 2 } ( \widehat { w } - w ^ { * } ) \| \lesssim \frac { \lambda ^ { - \alpha / 2 } } { \sqrt { n } } + \frac { \gamma ^ { \frac { r - 1 } { 2 } } \lambda ^ { - r / 2 } } { \sqrt { n } } + \frac { \gamma \lambda ^ { - 1 / 2 } } { \sqrt { m } } + \sqrt { \lambda } ,
$$

up to logarithmic factors and lower-order terms.

The optimization over $\gamma$ is identical to that of Corollary 3.1, except that we now impose the constraint $\gamma \geq 1$ . Let

$$
\gamma ^ { \star } ( \lambda ) \asymp \Big ( \frac { m } { n } \Big ) ^ { \frac { 1 } { 3 - r } } \lambda ^ { \frac { 1 - r } { 3 - r } }
$$

denote the unconstrained optimizer from Corollary 3.1. Under the constraint $\gamma \geq 1$ , we therefore choose

$$
\gamma ( \lambda ) \asymp \operatorname* { m a x } \{ 1 , \gamma ^ { \star } ( \lambda ) \} .
$$

Optimizing $\gamma$ and λ. When $\gamma ^ { \star } ( \lambda ) \geq 1$ , we recover exactly the same bound as in Corollary 3.1. When $\gamma ^ { \star } ( \lambda ) < 1$ , the constraint is active and we set $\gamma = 1$ , which yields the additional term

$$
{ \frac { \lambda ^ { - 1 / 2 } } { \sqrt { m } } } .
$$

Balancing this term with the bias term $\sqrt { \lambda }$ gives

$$
\lambda \asymp m ^ { - 1 / 2 } ,
$$

and therefore an additional contribution of order $m ^ { - 1 / 4 }$

Therefore, the optimal regularization parameter is

$$
\lambda \asymp \operatorname* { m a x } \{ \lambda _ { A } ^ { \star } , \lambda _ { B } ^ { \star } , m ^ { - 1 / 2 } \} ,
$$

where $\lambda _ { A } ^ { \star }$ and $\lambda _ { B } ^ { \star }$ are the optimizers obtained in the proof of Corollary 3.1. This yields

$$
\begin{array} { r } { \| ( \Sigma + \Sigma _ { D } ) ^ { 1 / 2 } ( \widehat { w } - w ^ { * } ) \| \lesssim \left\{ \begin{array} { l l } { n ^ { - \frac { 1 } { 4 } } m ^ { - \frac { 1 - r } { 8 } } + m ^ { - 1 / 4 } , } & { m \lesssim m _ { \mathrm { c r i t } } , } \\ { n ^ { - \frac { 1 } { 2 ( 1 + \alpha ) } } + m ^ { - 1 / 4 } , } & { m \gtrsim m _ { \mathrm { c r i t } } , } \end{array} \right. } \end{array}
$$

as claimed.

## A.7 Adaptation to the vector-valued case

The setting of this paper considers scalar-valued diferential operators, for simplicity of the argument.   
The adaptation to vector-valued operator is straightforward as we detail in this section.

Consider a vector-valued diferential operator $D = \left( D _ { 1 } , \ldots , D _ { k } \right)$ , where each $D _ { i }$ is of the form given in Assumption (A1). Then, for any $u \in \mathcal H$ and any $x \in \mathcal { X }$ , Du(x) is a vector in $\mathbb { R } ^ { k }$ . The diferential data points $d _ { j }$ are also in $\mathbb { R } ^ { k }$ and we replace the risk (1) by the following one:

$$
\widehat { R } ( u ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( u ( x _ { i } ) - y _ { i } \right) ^ { 2 } + \gamma \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \| D u ( z _ { j } ) - d _ { j } \| _ { 2 } ^ { 2 } + \lambda \| u \| _ { \mathcal { H } } ^ { 2 } .\tag{15}
$$

We can define component-wise feature maps $D _ { \ell } \phi ( z )$ , and define the diferential covariance operator as

$$
\Sigma _ { D } : = \mathbb { E } _ { z \sim \rho _ { D } } \left[ \sum _ { \ell = 1 } ^ { k } D _ { \ell } \phi ( z ) \otimes D _ { \ell } \phi ( z ) \right] .\tag{16}
$$

The key point is that the analysis is carried out using covariance operators Σ and $\Sigma _ { D } ,$ and the key assumption, Assumption (A6), is an abstract condition on Σ and $\Sigma _ { D } .$ , regardless of their expressions. As such, even if going from scalar-valued to vector-valued changes the expression of the covariance operator $\Sigma _ { D }$ as we just saw, as long as one guarantees that Assumption (A6) is satisfied, the rest of the analysis holds. The other assumptions must be adapted in a straightforward way: the smoothness s required in Assumption (A2) is the maximum of the orders of each $D _ { \ell } ,$ , and for Assumption (A3) we need the boundedness of each feature $\| D _ { \ell } \phi ( z ) \| _ { \mathcal { H } }$

For the proofs, we also adapt $V _ { D } , \widehat { \Sigma } _ { D } , \widehat { V } _ { D }$ by summing over $\ell :$

$$
V _ { D } : = \mathbb { E } \left[ \sum _ { \ell = 1 } ^ { k } d _ { \ell } D _ { \ell } \phi ( z ) \right]
$$

$$
\widehat { \Sigma } _ { D } : = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \sum _ { \ell = 1 } ^ { k } D _ { \ell } \phi ( z _ { j } ) \otimes D _ { \ell } \phi ( z _ { j } )
$$

$$
\widehat { V } _ { D } : = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \sum _ { \ell = 1 } ^ { k } d _ { j , \ell } D _ { \ell } \phi ( z _ { j } )
$$

Then, the concentration inequalities are still valid and the proofs are the same.

## A.8 Learning rates with exact physical constraint

We derive the PIKS rate in the idealized setting where the diferential information is known exactly. In this physical oracle setting, the value observations remain empirical, while the diferential part of the risk is replaced by its population counterpart:

$$
\widehat { R } _ { \mathrm { o r } } ( u ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( u ( x _ { i } ) - y _ { i } \right) ^ { 2 } + \gamma \mathbb { E } _ { z \sim \rho _ { D } } \left[ \left( D u ( z ) - D u ^ { * } ( z ) \right) ^ { 2 } \right] + \lambda \| u \| _ { \mathcal { H } } ^ { 2 } .
$$

Equivalently, in operator form, this replaces $\widehat { \Sigma } _ { D }$ and $\widehat { V } _ { D }$ by their population counterparts $\Sigma _ { D }$ and $V _ { D }$

Corollary A.3 (Oracle learning rate). Assume Assumptions $( A 1 ) – ( A 6 )$ hold, with $r < 1 \AA$ . Then there exist choices of $\lambda = \lambda ( n )$ and $\gamma = \gamma ( n )$ such that, with high probability,

$$
\| \Sigma ^ { 1 / 2 } ( \widehat { w } _ { \mathrm { o r } } - w ^ { * } ) \| \ \lesssim \ n ^ { - \frac { 1 } { 2 ( 1 + \alpha ) } } .
$$

Proof. The oracle estimator has the closed form

$$
\widehat { w } _ { \mathrm { o r } } = ( \widehat { \Sigma } + \gamma \Sigma _ { D } + \lambda I ) ^ { - 1 } ( \widehat { V } + \gamma V _ { D } ) .
$$

Let

$$
\widehat { \Sigma } _ { \gamma } ^ { \mathrm { o r } } : = \widehat { \Sigma } + \gamma \Sigma _ { { D } } , \qquad \widehat { V } _ { \gamma } ^ { \mathrm { o r } } : = \widehat { V } + \gamma V _ { { D } } , \qquad \Sigma _ { \gamma } : = \Sigma + \gamma \Sigma _ { { D } } .
$$

Under Assumptions (A4) and (A5),

$$
V = \Sigma w ^ { * } , \qquad V _ { D } = \Sigma _ { D } w ^ { * } , \qquad V + \gamma V _ { D } = \Sigma _ { \gamma } w ^ { * } .
$$

The proof of Theorem 3.1 applies verbatim, except that the diferential part is no longer empirical. Hence the only stochastic deviations are

$$
\widehat { \Sigma } - \Sigma , \qquad \widehat { V } - V ,
$$

and all terms involving $\widehat { \Sigma } _ { D } - \Sigma _ { D }$ and $\widehat { V } _ { D } - V _ { D }$ disappear. Up to logarithmic factors,

$$
\lVert \Sigma ^ { 1 / 2 } ( \widehat { w } _ { \mathrm { o r } } - w ^ { * } ) \rVert \stackrel { < } { \sim } \frac { 1 } { n \sqrt { \lambda } } + \frac { \mathcal { N } _ { V } ( \lambda , \gamma ) } { \sqrt { n } } + \lambda \mathcal { B } ( \lambda , \gamma ) ,
$$

where

$$
\begin{array} { r } { \mathcal { N } _ { V } ( \lambda , \gamma ) = \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } { \Sigma ^ { 1 / 2 } } \big \| _ { \mathrm { H S } } , \qquad \mathcal { B } ( \lambda , \gamma ) = \big \| ( \Sigma _ { \gamma } + \lambda I ) ^ { - 1 / 2 } w ^ { * } \big \| . } \end{array}
$$

By Assumption (A6), as in the proof of Corollary 3.1,

$$
\begin{array} { r } { \mathcal { N } _ { V } ( \lambda , \gamma ) ^ { 2 } \ \lesssim \ \lambda ^ { - \alpha } + \gamma ^ { r - 1 } \lambda ^ { - r } . } \end{array}
$$

Moreover, by attainability,

$$
\lambda B ( \lambda , \gamma ) \lesssim { \sqrt { \lambda } } .
$$

Therefore,

$$
\| \Sigma ^ { 1 / 2 } ( \widehat { w } _ { \mathrm { o r } } - w ^ { * } ) \| \lesssim \frac { 1 } { n \sqrt { \lambda } } + \frac { \lambda ^ { - \alpha / 2 } } { \sqrt { n } } + \frac { \gamma ^ { \frac { r - 1 } { 2 } } \lambda ^ { - r / 2 } } { \sqrt { n } } + \sqrt { \lambda } .
$$

Since the diferential information is known at the population level, there is no derivative-sample variance term increasing with γ. We choose γ large enough so that

$$
\gamma ^ { r - 1 } \lambda ^ { - r } \lesssim \lambda ^ { - \alpha } .
$$

Equivalently, when $r > \alpha$ , it sufices to take

$$
\gamma \gtrsim \lambda ^ { - \frac { r - \alpha } { 1 - r } } ,
$$

while when $r \leq \alpha .$ , any $\gamma \gtrsim 1$ is enough. With this choice,

$$
\big \| \Sigma ^ { 1 / 2 } \big ( \widehat { w } _ { \mathrm { o r } } - w ^ { * } \big ) \big \| \ \lesssim \ \frac { 1 } { n \sqrt \lambda } + \frac { \lambda ^ { - \alpha / 2 } } { \sqrt { n } } + \sqrt \lambda .
$$

Balancing the dominant variance term with the bias,

$$
{ \frac { \lambda ^ { - \alpha / 2 } } { \sqrt { n } } } = { \sqrt { \lambda } } ,
$$

gives

$$
\lambda \asymp n ^ { - \frac { 1 } { 1 + \alpha } } .
$$

For this choice,

$$
{ \frac { \lambda ^ { - \alpha / 2 } } { \sqrt { n } } } \asymp { \sqrt { \lambda } } \asymp n ^ { - { \frac { 1 } { 2 ( 1 + \alpha ) } } } ,
$$

while

$$
{ \frac { 1 } { n { \sqrt { \lambda } } } } = n ^ { - 1 + { \frac { 1 } { 2 ( 1 + \alpha ) } } }
$$

is lower order since $\alpha > 0$ . Hence

$$
\begin{array} { r } { \| \Sigma ^ { 1 / 2 } ( \widehat { w } _ { \mathrm { o r } } - w ^ { * } ) \| \ \lesssim \ n ^ { - \frac { 1 } { 2 ( 1 + \alpha ) } } . } \end{array}
$$

## B Examples

We now illustrate Assumption (A6) on concrete examples where the decomposition can be computed explicitly.

## B.1 Partial Laplacian and periodic Sobolev RKHS

Lemma B.1 (Capacity decomposition for Sobolev spaces). Let $\mathcal { X } = \mathbb { T } ^ { d }$ , let k be a Matérn kernel of smoothness $s > d / 2$ , and let $\textstyle D = \sum _ { i \in S } \partial _ { x _ { i } } ^ { 2 }$ be a partial Laplacian with $S \subseteq \{ 1 , \ldots , d \}$ . Assume that value and diferential samples are drawn uniformly on T<sup>d</sup>.

Then the value covariance operator Σ admits a decomposition $\Sigma = \Sigma _ { 1 } + \Sigma _ { 2 }$ satisfying Assumption (A6), with exponents

$$
\alpha = \frac { d - | S | } { 2 s } , \qquad r = \frac { d } { 2 s } .
$$

Proof. Step 1: diagonalization in the Fourier basis. Let D be a constant-coeficient diferential operator of order q, of the form

$$
D = \sum _ { | \alpha | \leq q } c _ { \alpha } \partial ^ { \alpha } , \qquad \partial ^ { \alpha } = \partial _ { x _ { 1 } } ^ { \alpha _ { 1 } } \cdot \cdot \cdot \partial _ { x _ { d } } ^ { \alpha _ { d } } ,
$$

with real coeficients $c _ { \alpha }$ . Its Fourier symbol is given by

$$
P ( k ) = \sum _ { | \alpha | \leq q } c _ { \alpha } ( 2 \pi i k ) ^ { \alpha } , \qquad k \in \mathbb { Z } ^ { d } ,
$$

so that $D e ^ { 2 \pi i k \cdot x } = P ( k ) e ^ { 2 \pi i k \cdot x }$

Since the kernel k is translation invariant and both x and z are sampled uniformly on $\mathbb { T } ^ { d }$ , the covariance operators

$$
\Sigma = \mathbb { E } [ \phi ( x ) \otimes \phi ( x ) ] , \qquad \Sigma _ { D } = \mathbb { E } [ D \phi ( z ) \otimes D \phi ( z ) ]
$$

are convolution operators and are therefore diagonal in the Fourier basis $\{ e _ { k } ( x ) : = e ^ { 2 \pi i k \cdot x } \} _ { k \in \mathbb { Z } ^ { d } }$ The corresponding eigenvalues are

$$
\sigma _ { k } = \mu _ { k } , \qquad \tau _ { k } = | P ( k ) | ^ { 2 } \mu _ { k } .
$$

Step 2: partial Laplacian and visible/invisible frequencies. For $\textstyle D = \sum _ { i \in S } \partial _ { x _ { i } } ^ { 2 }$ , the symbol is

$$
P ( k ) = - ( 2 \pi ) ^ { 2 } \| k _ { S } \| ^ { 2 } , \qquad \mathrm { s o } \qquad \tau _ { k } = ( 2 \pi ) ^ { 4 } \| k _ { S } \| ^ { 4 } \mu _ { k } ,
$$

where $k _ { S }$ denotes the restriction of k to coordinates in S. Define the index sets

$$
\mathcal { Z } : = \{ k \in \mathbb { Z } ^ { d } : ~ k _ { S } = 0 \} , \qquad \mathcal { Z } ^ { c } : = \mathbb { Z } ^ { d } \setminus \mathcal { Z } .
$$

Thus $\mathcal { Z }$ consists of frequencies invisible to $D$ (since $P ( k ) = 0 )$ , whereas $\mathcal { Z } ^ { c }$ corresponds to frequencies detectable by D.

Step 3: the decomposition $\Sigma = \Sigma _ { 1 } + \Sigma _ { 2 }$ . Define

$$
\Sigma _ { 1 } : = \sum _ { k \in \mathcal { Z } } \mu _ { k } e _ { k } \otimes e _ { k } , \qquad \Sigma _ { 2 } : = \sum _ { k \in \mathcal { Z } ^ { c } } \mu _ { k } e _ { k } \otimes e _ { k } .
$$

Then $\Sigma = \Sigma _ { 1 } + \Sigma _ { 2 }$ and $\Sigma _ { 1 } , \Sigma _ { 2 } \succeq 0 .$

Step 4: capacity of the invisible component $\Sigma _ { 1 }$ . Let $d _ { 0 } = d - | \boldsymbol { S } |$ and write $k _ { - S } \in \mathbb { Z } ^ { d - S }$ for the subvector of coordinates outside S. The map $k _ { - S } \mapsto ( k _ { - S } , 0 _ { S } )$ is a bijection $\mathbb { Z } ^ { d - S } \to \mathcal { Z }$ and $| ( k _ { - S } , 0 _ { S } ) | = | k _ { - S } |$ , hence

$$
\sum _ { k \in \mathcal { Z } } \frac { \mu _ { k } } { \mu _ { k } + \lambda } = \sum _ { k - s \in \mathbb { Z } ^ { d - S } } \frac { \mu _ { ( k _ { - } s , 0 s ) } } { \mu _ { ( k _ { - } s , 0 s ) } + \lambda } , \qquad \mu _ { ( k _ { - } s , 0 s ) } \asymp ( 1 + | k _ { - } s | ^ { 2 } ) ^ { - s } .
$$

Therefore, by a standard comparison between lattice sums and integrals, for all $\lambda \in ( 0 , 1 ]$

$$
\mathrm { T r } \bigl ( \Sigma _ { 1 } ( \Sigma _ { 1 } + \lambda I ) ^ { - 1 } \bigr ) = \sum _ { k \in \mathcal { Z } } \frac { \mu _ { k } } { \mu _ { k } + \lambda } \ \lesssim \ \lambda ^ { - \frac { d _ { 0 } } { 2 s } } .
$$

This verifies the first trace condition with exponent $\alpha = d _ { 0 } / ( 2 s )$

Step 5: alignment of $\Sigma _ { 2 }$ with $\Sigma _ { D }$ . On $\mathcal { Z } ^ { c }$ we have $\| k _ { S } \| \geq 1$ and hence $\tau _ { k } = ( 2 \pi ) ^ { 4 } \| k _ { S } \| ^ { 4 } \mu _ { k } \geq$ $c \mu _ { k }$ for some $c > 0$ . Therefore, for all $t \in ( 0 , 1 ]$

$$
\mathrm { T r } \big ( ( \Sigma _ { D } + t I ) ^ { - 1 } \Sigma _ { 2 } \big ) = \sum _ { k \in \mathcal { Z } ^ { c } } \frac { \mu _ { k } } { \tau _ { k } + t } \ \leq \ \sum _ { k \in \mathcal { Z } ^ { c } } \frac { \mu _ { k } } { c \mu _ { k } + t } .
$$

Using $\mu _ { k } \asymp ( 1 + | k | ^ { 2 } ) ^ { - s }$ and standard comparison between lattice sums and integrals gives

$$
\sum _ { k \in \mathbb { Z } ^ { d } } \frac { \mu _ { k } } { \mu _ { k } + t } \ \lesssim \ t ^ { - \frac { d } { 2 s } } , \qquad t \in ( 0 , 1 ] ,
$$

and hence

$$
\mathrm { T r } \left( ( \Sigma _ { { D } } + t I ) ^ { - 1 } \Sigma _ { 2 } \right) ~ \lesssim ~ t ^ { - \frac { d } { 2 s } } .
$$

This verifies the second trace condition with exponent $r = d / ( 2 s )$

Combining Steps 3-5 yields Assumption (A6) with the claimed exponents.

## B.2 Gradients

Let $\Omega \subset \mathbb { R } ^ { d }$ be a bounded, connected, Lipschitz domain. Fix $\begin{array} { r } { s > \frac { d } { 2 } + 1 } \end{array}$ . Let $\mathscr { H } = H ^ { s } ( \Omega )$ , with norm $\| \cdot \| _ { \mathcal { H } }$ equivalent to the standard $H ^ { s } ( \Omega )$ norm. We denote by K the reproducing kernel associated to $\mathcal { H } .$ , and by $\phi : \Omega \to \mathcal { H }$ the feature map defined, for all $x \in \Omega$ , by $\phi ( x ) = K ( x , \cdot ) \in \mathcal { H }$ . Choose $\rho = \rho _ { D }$ the uniform distribution on $\Omega .$

## B.2.1 Covariance operator decomposition

Definition of Σ Define the covariance operator

$$
\Sigma : = \mathbb { E } _ { x \sim \rho } [ \phi ( x ) \otimes \phi ( x ) ] .
$$

We have, for all $u \in \mathcal H$

$$
\begin{array} { r } { \langle \Sigma u , u \rangle _ { \mathcal { H } } = \| u \| _ { L ^ { 2 } ( \rho ) } ^ { 2 } . } \end{array}\tag{17}
$$

Definition of $\Sigma _ { D }$ Define the gradients covariance operator

$$
\Sigma _ { D } = \mathbb { E } _ { z \sim \rho _ { D } } \left[ \sum _ { i = 1 } ^ { d } \partial _ { i } \phi ( z ) \otimes \partial _ { i } \phi ( z ) \right] .
$$

We have, for all $u \in \mathcal H$

$$
\langle \Sigma _ { D } u , u \rangle _ { \mathcal { H } } = \| \nabla u \| _ { L ^ { 2 } ( \rho ) } ^ { 2 } .\tag{18}
$$

Definition of $\Sigma _ { 1 }$ Denote by $\phi _ { \Omega } = \mathbb { E } _ { x \sim \rho } [ \phi ( x ) ] \in \mathcal { H }$ the representer of the averaging functional, so that

$$
\begin{array} { r } { \langle u , \phi _ { \Omega } \rangle _ { \mathcal { H } } = \mathbb { E } _ { x \sim \rho } [ u ( x ) ] = : u _ { \Omega } . } \end{array}
$$

Denote

$$
\Sigma _ { 1 } = \phi _ { \Omega } \otimes \phi _ { \Omega } .
$$

Then for all $u \in \mathcal H$

$$
\langle \Sigma _ { 1 } u , u \rangle _ { \mathcal { H } } = u _ { \Omega } ^ { 2 } .\tag{19}
$$

Definition of $\Sigma _ { 2 }$ Next, define the bounded operator

$$
A : \mathcal { H } \to L ^ { 2 } ( \rho ) , \qquad A u : = u - u _ { \Omega } ,
$$

Set

$$
\Sigma _ { 2 } : = A ^ { * } A ,
$$

so that for all $u \in \mathcal H$

$$
\langle \Sigma _ { 2 } u , u \rangle _ { \mathcal { H } } = \| u - u _ { \Omega } \| _ { L ^ { 2 } ( \rho ) } ^ { 2 } = \mathrm { V a r } _ { x \sim \rho } ( u ( x ) ) .\tag{20}
$$

Proposition B.2 (Decomposition of Σ). One has

$$
\Sigma = \Sigma _ { 1 } + \Sigma _ { 2 } .
$$

Proof. Let $u \in \mathcal H$ . By the variance decomposition formula,

$$
\begin{array} { r } { \mathbb { E } _ { { x } \sim \rho } \big [ u ( x ) ^ { 2 } \big ] = \Big ( \mathbb { E } _ { { x } \sim \rho } [ u ( x ) ] \Big ) ^ { 2 } + \operatorname { V a r } _ { x \sim \rho } ( u ( x ) ) . } \end{array}
$$

Equivalently,

$$
\begin{array} { r } { \| u \| _ { L ^ { 2 } ( \rho ) } ^ { 2 } = u _ { \Omega } ^ { 2 } + \| u - u _ { \Omega } \| _ { L ^ { 2 } ( \rho ) } ^ { 2 } = u _ { \Omega } ^ { 2 } + \mathrm { V a r } _ { x \sim \rho } ( u ( x ) ) . } \end{array}
$$

Using (17), (19), and (20), we obtain

$$
\langle \Sigma u , u \rangle _ { \mathcal { H } } = \langle \Sigma _ { 1 } u , u \rangle _ { \mathcal { H } } + \langle \Sigma _ { 2 } u , u \rangle _ { \mathcal { H } }
$$

for every $u \in \mathcal H$

Since $\Sigma , \Sigma _ { 1 }$ , and $\Sigma _ { 2 }$ are bounded self-adjoint operators on H, equality of their quadratic forms implies, by polarization, that

$$
\Sigma = \Sigma _ { 1 } + \Sigma _ { 2 } .
$$

## B.2.2 Computation of the coeficients

Let us first bound the efective dimension

$$
\begin{array} { r } { \mathcal { N } _ { 1 } ( \lambda ) : = \mathrm { T r } \left( \Sigma _ { 1 } ( \Sigma _ { 1 } + \lambda I ) ^ { - 1 } \right) . } \end{array}
$$

Since $\Sigma _ { 1 }$ has rank one, the efective dimension $\mathcal { N } _ { 1 } ( \lambda )$ is uniformly bounded. In particular, it satisfies

$$
\mathcal { N } _ { 1 } ( \lambda ) \leq C _ { \alpha } \lambda ^ { - \alpha } , \qquad \lambda \in ( 0 , 1 ] ,
$$

with

$$
\boxed { \alpha = 0 } .
$$

Let us now bound

$$
\mathrm { T r } \big ( ( \Sigma _ { D } + \lambda I ) ^ { - 1 } \Sigma _ { 2 } \big ) .
$$

Since Ω is bounded, connected, and Lipschitz, the Poincaré–Wirtinger inequality [26] holds: there exists $C > 0$ such that, for every $u \in H ^ { 1 } ( \Omega )$

$$
\| u - u _ { \Omega } \| _ { L ^ { 2 } ( \rho ) } \leq C \| \nabla u \| _ { L ^ { 2 } ( \rho ) } .
$$

Therefore, for every $u \in \mathcal { H } .$

$$
\begin{array} { r l r } & { } & { \langle \Sigma _ { 2 } u , u \rangle _ { \mathcal { H } } = \| u - u _ { \Omega } \| _ { L ^ { 2 } ( \rho ) } ^ { 2 } } \\ & { } & { \qquad \leq C ^ { 2 } \| \nabla u \| _ { L ^ { 2 } ( \rho ) } ^ { 2 } } \\ & { } & { \qquad = C ^ { 2 } \langle \Sigma _ { D } u , u \rangle _ { \mathcal { H } } . } \end{array}
$$

Equivalently, in the Loewner order,

$$
0 \preceq \Sigma _ { 2 } \preceq C ^ { 2 } \Sigma _ { D } .
$$

It follows that

$$
\begin{array} { r } { \mathrm { T r } \big ( ( \Sigma _ { D } + \lambda I ) ^ { - 1 } \Sigma _ { 2 } \big ) \leq C ^ { 2 } \mathrm { T r } \big ( \Sigma _ { D } ( \Sigma _ { D } + \lambda I ) ^ { - 1 } \big ) . } \end{array}
$$

We now bound the last trace. Let $( \eta _ { j } ) _ { j \ge 1 }$ be the nonzero eigenvalues of $\Sigma _ { D }$ , arranged in nonincreasing order. Since $\Sigma _ { D }$ is the covariance operator associated with the map

$$
u \longmapsto \nabla u , \qquad H ^ { s } ( \Omega ) \to L ^ { 2 } ( \rho ; \mathbb { R } ^ { d } ) ,
$$

the standard eigenvalue estimate for Sobolev embeddings on bounded Lipschitz domains gives

$$
\eta _ { j } \leq C _ { D } j ^ { - 2 ( s - 1 ) / d } .
$$

Set

$$
\beta : = \frac { 2 ( s - 1 ) } { d } .
$$

Because $\begin{array} { r } { s > \frac { d } { 2 } + 1 } \end{array}$ , we have $\beta > 1$ . Hence, for $0 < \lambda \leq 1$ ，

$$
\begin{array} { r l r } {  { \mathrm { T r } \big ( \Sigma _ { D } ( \Sigma _ { D } + \lambda I ) ^ { - 1 } \big ) = \sum _ { j \geq 1 } \frac { \eta _ { j } } { \eta _ { j } + \lambda } } } \\ & { } & { \quad \leq \sum _ { j \geq 1 } \frac { C _ { D } j ^ { - \beta } } { C _ { D } j ^ { - \beta } + \lambda } } \\ & { } & { \quad \lesssim \lambda ^ { - 1 / \beta } . } \end{array}
$$

Since $1 / \beta = d / ( 2 ( s - 1 ) )$ , we obtain

$$
\mathrm { T r } \big ( ( \Sigma _ { D } + \lambda I ) ^ { - 1 } \Sigma _ { 2 } \big ) \lesssim \lambda ^ { - d / ( 2 ( s - 1 ) ) } .
$$

Thus one may take

$$
\boxed { r = \cfrac { d } { 2 ( s - 1 ) } . }
$$

In particular, since $\begin{array} { r } { s > \frac { d } { 2 } + 1 } \end{array}$ , one has $r < 1$

## C Detailed experimental setup

In this section we provide more details about the setup used for the experiments of Section 5.

Implementation To maximize the flexibility of our system, we implemented the PIKS estimator using the Jax framework, which allows to eficiently compute arbitrary derivatives. While very flexible, any automatic diferentiation framework does introduce some computational overhead and requires particular care when implementing kernel functions. For example, Matérn kernels require diferentiating through a square-root which is numerically unstable at 0. We introduce a small additive nugget term to ensure stability everywhere. The kernel solver is a straightforward implementation of the equations in Lemma A.2 which yield a space complexity of $O ( ( n + m ) ^ { 2 } )$ and time complexity of $O ( ( n + m ) ^ { 3 } )$

Bounded domain gradient example We repeated the experiment 10 times to obtain standard deviations reported in Fig. 2. We set the variance on the diferential data to be low in order to avoid having to scale m too much before seeing a saturation efect. All data points were uniformly sampled on a unit disk with 1000 samples used for validation (to select hyperparameters) and 10000 for computing the test error.

Partial Laplacian example We give a few more details on the kernel and target functions for the example on Sobolev spaces on the torus with the partial Laplacian operator. Here we used $\mathcal { X } = \mathbb { T } ^ { 4 }$ . We sampled uniformly at random $F = 6 4$ frequency vectors $k _ { \ell } \in \mathbb { Z } ^ { 4 }$ such that for $\ell \in [ 0 , 1 5 ]$ $k _ { \ell } \sim [ \mathcal { U } ( [ - 8 , 8 ] \backslash \{ 0 \} ) , 0 , 0 , 0 ] ;$ for $\ell \in [ 1 6 , 3 1 ] , k _ { \ell } \sim [ 0 , \mathcal { U } ( [ - 8 , 8 ] \backslash \{ 0 \} ) ]$ ] and so on (only one dimension active for each vector). For feature map $\phi _ { \ell } ( x ) = \sqrt { 2 } \cos ( 2 \pi \langle k _ { \ell } , x \rangle )$ , target function and kernel are defined as

$$
u ^ { * } ( x ) = u _ { 0 } + \sum _ { \ell = 1 } ^ { F } c _ { \ell } \phi _ { \ell } ( x ) , \qquad k ( x , x ^ { \prime } ) = 1 + \sum _ { \ell = 1 } ^ { F } \mu _ { \ell } \phi _ { \ell } ( x ) \phi _ { \ell } ( x ^ { \prime } ) ,
$$

with the smoothness coeficients

$$
\mu _ { \ell } \asymp ( 1 + \| k _ { \ell } \| ^ { 2 } ) ^ { - \beta } , \quad c _ { \ell } \asymp ( 1 + \| k _ { \ell } \| ^ { 2 } ) ^ { - \delta } , \quad \beta > 1 , \quad \delta > \frac { 1 } { 2 } .
$$

In particular, we set $\beta = \delta = 4 . 1$ to have a well-specified problem in a smooth enough space. We use additive Gaussian noise, with standard-deviation equal to 10% to the range of the data. This was done because the diferent data components have vastly diferent numerical ranges, and setting a fixed variance would have introduced artifacts in the results.

The best-fit lines of Fig. 3 are obtained by least-squares regression of the experimental data against the function $a n ^ { b }$ to find the exponential coeficient b. Then, using Corollary 3.1, we get $\begin{array} { r } { \alpha = - \frac { 1 } { 2 b } - 1 } \end{array}$

Hyperparameters The two hyperparameters of PIKS (λ and $\gamma )$ , as well as the hyperparameters of the kernel (notably the length-scale of the Matérn kernel) were determined by a coarse grid-search using small, noisy validation sets to estimate their performance at generalization time.