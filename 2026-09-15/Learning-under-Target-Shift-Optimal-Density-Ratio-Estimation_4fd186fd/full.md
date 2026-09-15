# Learning under Target Shift: Optimal Density Ratio Estimation and Importance-Weighted Regression

Ren-Rui Liu<sup>1</sup>

Zheng-Chu Guo<sup>1,∗</sup>

## Abstract

We study density ratio estimation and importance-weighted regression under target shift with continuous outputs. Under target shift, the conditional distribution of the inputs given the outputs remains invariant across the training and test distributions, while the output marginal distribution may change. Although this problem has been extensively studied for discrete outputs, the continuous setting is substantially less understood: the importance weights are determined by an unknown density ratio function, for which existing estimation methods lack explicit finite-sample convergence rates. We propose a spectral regularization method in a reproducing kernel Hilbert space (RKHS) for estimating the continuous density ratio from labeled training samples and unlabeled test inputs. Under a source condition with regularity parameter $\iota > 0 ,$ we establish high-probability finite-sample guarantees and show that the estimator achieves the capacity-independent minimax-optimal RKHS-norm rate $O ( n _ { \eta } ^ { - \iota / ( 2 \iota + 2 ) } )$ . We then incorporate the estimated density ratio into importance-weighted regression and characterize the propagation of density-ratio estimation error to the final predictor. When suficiently many samples are available for density ratio estimation, the resulting regression estimator attains the minimaxoptimal rates of standard kernel regression. These results establish a finite-sample theory for continuous density ratio estimation and importance-weighted learning under target shift.

Keywords: Learning theory, density ratio estimation, target shift, importance weighting, reproducing kernel Hilbert space, spectral algorithm

## 1 Introduction

Classical supervised learning relies on the fundamental assumption that training and test data are drawn from the same probability distribution. Under this assumption, minimizing the empirical risk on training samples yields a predictor that generalizes well to unseen test data with high probability (Vapnik, 1998). In virtually all realistic deployment scenarios, however, this assumption is violated: the training and test distributions difer—a situation broadly termed distribution shift (Qui˜nonero-Candela et al., 2008; Storkey, 2008). When this shift occurs, the empirical risk no longer accurately approximates the test risk, and models trained via standard empirical risk minimization can sufer severe degradation in predictive performance.

A principled and widely adopted strategy for correcting distribution shift is importance weighting (Shimodaira, 2000). The key idea is to reweight each training example by the density ratio (the ratio of the test density to the training density evaluated at that example, formally defined in (3)), thereby constructing an unbiased estimator of the test-expected risk from training data alone. Consequently, learning under distribution shift can be reduced to the auxiliary problem of estimating the unknown density ratio from finite samples.

Among the various forms of distribution shift, two have received particular attention because they induce tractable factorizations of the density ratio. Under covariate shift, the conditional distribution of the output given the input is assumed invariant across domains, while the marginal distribution of the inputs may change (Shimodaira, 2000; Sugiyama et al., 2012). In this case, the density ratio then depends only on the input, and a rich literature has developed methods for its estimation and analyzed the resulting importance-weighted estimators, including sharp theoretical guarantees for kernel methods (Gizewski et al., 2022; Ma et al., 2023; Gogolashvili et al., 2023; Guo and Shi, 2025; Fan et al., 2025). Under target shift, the roles of the input and output are reversed: the conditional distribution of the input given the output remains invariant, while the marginal distribution of the output changes between the training and test domains (Saerens et al., 2002; Zhang et al., 2013). This setting arises naturally when the output can be viewed as a cause of the observed input features. For example, in medical diagnosis, a disease (output) causes symptoms (inputs); the prevalence of the disease may difer across hospitals, yet the distribution of symptoms given the disease remains the same. Similarly, in econometric forecasting, a policy intervention (output) drives economic indicators (inputs); the policy mix may shift over time, while the conditional distribution of the indicators given the policy stays invariant. Under target shift, the density ratio therefore reduces to a function of the output alone, namely the ratio of the test to the training marginal density of the outputs.

The vast majority of existing work on target shift has focused on the discrete case, where the importance weight reduces to a finite-dimensional vector and a mature literature provides eficient algorithms with strong theoretical guarantees (Saerens et al., 2002; Lipton et al., 2018; Azizzade nesheli et al., 2019; Alexandari et al., 2020; Garg et al., 2020). In the continuous setting, by contrast, the importance weight becomes an infinite-dimensional function whose estimation typically involves solving an ill-posed integral equation. Despite its practical importance, density ratio estimation under continuous target shift remains comparatively underexplored. A few pioneering algorithms have been proposed (Zhang et al., 2013; Nguyen et al., 2016; Kim et al., 2024), but a finite-sample theory with explicit, optimized convergence rates for the estimated density ratio is still lacking. In particular, Kim et al. (2024) established consistency, but the resulting rates depend on bandwidth and regularization parameters that are not optimized with respect to the sample size; More recently, Gogolashvili (2026) derived minimax-optimal rates for importance-weighted regression under continuous target shift, but assumed that the true density ratio is known. Thus, establishing explicit finite-sample convergence rates for estimating the density ratio itself remains an important open problem.

Contributions. This work develops a finite-sample learning theory for continuous density ratio estimation under target shift and investigates how density ratio estimation error propagates to downstream importance-weighted regression. Our main contributions are summarized as follows.

• A spectral framework for continuous density ratio estimation. We propose a spectral algorithm in an RKHS for estimating the density ratio between the output marginal distributions using labeled training samples and unlabeled test inputs. By casting density ratio estimation as a regularized operator equation, the proposed framework accommodates a broad class of spectral regularization methods.

• Minimax-optimal finite-sample guarantees for density ratio estimation. We establish explicit finite-sample convergence bounds for the proposed density ratio estimator and show that, under suitable source conditions, it achieves the minimax-optimal convergence rate in the RKHS norm. To the best of our knowledge, this provides the first such finitesample guarantee for continuous density ratio estimation under target shift, going beyond existing results that either assume the density ratio to be known or establish only asymptotic consistency.

• Learning theory for importance-weighted regression with an estimated density ratio. We incorporate the estimated density ratio into an importance-weighted spectral algorithm for the downstream regression problem and explicitly characterize the propagation of density ratio estimation error into the final prediction error. Our analysis identifies two statistical regimes: when suficiently many samples are allocated to density ratio estimation, the regression estimator retains the minimax-optimal learning rates of standard kernel regression; when the available sample size is insuficient, density ratio estimation error becomes dominant and leads to slower convergence rates.

• A quantitative characterization of the interaction between the two learning stages. Our analysis explicitly characterizes how the sample size for density ratio estimation, the regularity of both the density ratio and the target function, and the sample size for downstream regression jointly determine the overall convergence rate. These results provide a rigorous theoretical foundation for importance weighting under continuous target shift and ofer quantitative guidance on the sample requirements for density ratio estimation relative to those for downstream regression.

Paper organization. The remainder of this paper is organized as follows: Section 2 introduces the problem setting, presents the main assumptions, and states the main theoretical results; Section 3 provides a literature review and comparative analysis with existing works; Section 4 contains the proofs of the main theorems.

## 2 Problem Setting and Main Results

We begin by considering the regression setting considered in this paper. Let $\boldsymbol { \mathcal { X } } \subseteq \mathbb { R } ^ { d }$ and $\mathcal { V } \subseteq \mathbb { R }$ be compact input and output spaces, respectively, and consider the squared loss. Our goal is to learn a predictor from training data $\{ ( \pmb { x } _ { j } , y _ { j } ) \} _ { j = 1 } ^ { n _ { f } } \subseteq \mathcal { X } \times \mathcal { Y }$ that achieves good generalization performance under a test distribution $p _ { \mathcal { X } \times \mathcal { Y } } ^ { \mathrm { t e } } ( \pmb { x } , y )$ , when the training data are generated from a potentially diferent distribution $p _ { \mathcal { X } \times \mathcal { Y } } ^ { \mathrm { t r } } ( \pmb { x } , y )$ . Both the training and test distributions are unknown and can only be accessed through finite samples.

For a predictor $f ,$ its generalization performance is measured by the expected risk under the test distribution:

$$
\mathbb { E } _ { ( \pmb { x } , \pmb { y } ) \sim p _ { \pmb { \chi } \times \mathscr { y } } ^ { \mathrm { t e } } } \big [ ( \mathscr { y } - f ( \pmb { x } ) ) ^ { 2 } \big ] .\tag{1}
$$

The minimizer of the expected risk is the regression function $f _ { \star } ^ { \mathrm { t e } }$ , defined as

$$
f _ { \star } ^ { \mathrm { t e } } ( { \pmb x } ) = \mathbb { E } _ { { \pmb y } \sim p _ { { \mathscr y } | { \mathscr x } } ^ { \mathrm { t e } } } [ { \pmb y } \mid { \pmb x } ] ,
$$

where $p _ { \mathcal { V } | \mathcal { X } } ^ { \mathrm { t e } } ( y \mid x )$ denotes the test-conditional distribution of y given x. Hence, learning a predictor with small test risk amounts to accurately estimating $f _ { \star } ^ { \mathrm { t e } }$ . Since the test distribution $p _ { \mathcal { X } \times \mathcal { Y } } ^ { \mathrm { t e } }$ is unknown, the expected risk (1) cannot be computed directly. Instead, we approximate it by the empirical risk over the training sample:

$$
\frac { 1 } { n _ { f } } \sum _ { j = 1 } ^ { n _ { f } } ( y _ { j } - f ( \pmb { x } _ { j } ) ) ^ { 2 } .\tag{2}
$$

In the standard learning setting, where the training and test distributions coincide, i.e., $( p _ { \mathcal { X } \times \mathcal { Y } } ^ { \mathrm { t r } } =$ $p _ { \mathcal { X } \times \mathcal { Y } } ^ { \mathrm { t e } } )$ , the empirical risk (2) is an unbiased estimator of the expected risk, and minimizing it often provides a consistent estimator of $f _ { \star } ^ { \mathrm { t e } }$ under suitable conditions (Vapnik, 1998). Under distribution $s h i f t .$ however, $p _ { \mathcal { X } \times \mathcal { Y } } ^ { \mathrm { t r } } \neq p _ { \mathcal { X } \times \mathcal { Y } } ^ { \mathrm { t e } }$ and (2) becomes a biased estimator. Consequently, minimizing the unweighted empirical risk may lead to a predictor that generalizes poorly under the test distribution.

To correct this bias, we can reweight each training pair by the density ratio between the test and training distributions. Define the density ratio $\eta _ { \star }$ as

$$
\eta _ { \star } ( x , y ) = { \frac { p _ { \mathcal { X } \times \mathcal { Y } } ^ { \mathrm { t e } } ( x , y ) } { p _ { \mathcal { X } \times \mathcal { Y } } ^ { \mathrm { t r } } ( x , y ) } } ,\tag{3}
$$

and form the weighted empirical risk

$$
\frac { 1 } { n _ { f } } \sum _ { j = 1 } ^ { n _ { f } } \eta _ { \star } ( x _ { j } , y _ { j } ) ( y _ { j } - f ( x _ { j } ) ) ^ { 2 } ,\tag{4}
$$

then (4) is again an unbiased estimator of the expected risk (1). This approach is known as importance weighting (Shimodaira, 2000). In this paper, we focus exclusively on target shift, where the marginal distributions of $y$ difer between training and test, while the conditional distribution of x given y remains invariant. That is,

$$
p _ { \chi \times \mathcal { Y } } ^ { \mathrm { t r } } ( \pmb { x } , y ) = p _ { \chi | \mathcal { Y } } ( \pmb { x } \mid y ) p _ { \mathcal { Y } } ^ { \mathrm { t r } } ( y ) , \quad p _ { \mathcal { X } \times \mathcal { Y } } ^ { \mathrm { t e } } ( \pmb { x } , y ) = p _ { \mathcal { X } | \mathcal { Y } } ( \pmb { x } \mid y ) p _ { \mathcal { Y } } ^ { \mathrm { t e } } ( y ) ,
$$

with $p _ { \mathcal { V } } ^ { \mathrm { t r } } \neq p _ { \mathcal { V } } ^ { \mathrm { t e } }$ and $p _ { \mathcal { X } | \mathcal { Y } }$ identical across both distributions. Under target shift, the density ratio simplifies to a function of $y$ only:

$$
\eta _ { \star } ( x , y ) = \eta _ { \star } ( y ) = \frac { p _ { y } ^ { \mathrm { t e } } ( y ) } { p _ { y } ^ { \mathrm { t r } } ( y ) } .\tag{5}
$$

In practice, the density ratio $\eta _ { \star }$ is unknown and need to be estimated from data. In classification, where Y is discrete, $\eta _ { \star }$ reduces to a finite-dimensional vector and can, in principle, be estimated directly from labeled samples from the training and test distributions. In regression, however, $\eta _ { \star }$ is an unknown function over a continuous output space. Direct estimation of $p _ { \mathcal { V } } ^ { \mathrm { t e } } / p _ { \mathcal { V } } ^ { \mathrm { t r } }$ would therefore require suficiently many labeled test samples, which are often unavailable or prohibitively expensive to obtain. To avoid this requirement, we assume access only to labeled training samples $\{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n _ { \eta } }$ 1 drawn from $p _ { \mathcal { X } \times \mathcal { Y } } ^ { \mathrm { t r } }$ , and unlabeled data $\{ { \pmb x } _ { i } ^ { \prime } \} _ { i = 1 } ^ { n _ { \eta } }$ drawn from the test marginal distribution $p _ { \mathcal { X } } ^ { \mathrm { t e } }$ . Our goal is to exploit the dependence between the inputs and outputs, together with the target-shift assumption, to estimate the continuous density ratio $\eta _ { \star }$ without requiring labeled test data.

In the following sections, we first develop a method for estimating the continuous density ratio $\eta _ { \star }$ , and then incorporate the resulting estimate into the importance-weighting framework to learn the regression function $f _ { \star } ^ { \mathrm { t e } }$

## 2.1 Our Estimation Methods

We estimate the density ratio $\eta _ { \star }$ from a labeled sample $\{ ( \pmb { x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n _ { \eta } } \sim p _ { \mathcal { X } \times \mathcal { Y } } ^ { \mathrm { t r } }$ and an unlabeled sample $\{ \pmb { x } _ { i } ^ { \prime } \} _ { i = 1 } ^ { n _ { \eta } } \sim p _ { \mathcal { X } } ^ { \mathrm { t e } }$ . To construct the estimator, we first introduce the underlying function spaces. Let $K _ { \mathcal { X } } \colon \mathcal { X } \times \mathcal { X } \ \to \ \mathbb { R } _ { + }$ and $K _ { \mathcal { y } } \colon \mathcal { y } \times \mathcal { y }  \mathbb { R } _ { + }$ be symmetric, positive-definite, and continuous kernels. These kernels induce two reproducing kernel Hilbert spaces (RKHSs), denoted by $\mathcal { H } _ { \mathcal { X } }$ and $\mathcal { H } _ { \mathcal { Y } }$ , whose elements satisfy the reproducing properties:

$$
\langle f , K _ { \mathcal { X } } ( \cdot , \pmb { x } ) \rangle _ { \mathcal { H } _ { \mathcal { X } } } = f ( \pmb { x } ) , \quad \forall f \in \mathcal { H } _ { \mathcal { X } } , \quad \forall \pmb { x } \in \mathcal { X } ;
$$

$$
\langle h , K _ { \mathcal { Y } } ( \cdot , y ) \rangle _ { \mathcal { H } _ { \mathcal { Y } } } = h ( y ) , \quad \forall h \in \mathcal { H } _ { \mathcal { Y } } , \quad \forall y \in \mathcal { Y } .
$$

Next, we assume that the joint distributions on $\mathcal { X } \times \mathcal { V }$ factorize as follows:

$$
p _ { \mathcal { X } \times \mathcal { Y } } ^ { \mathrm { t r } } ( \pmb { x } , y ) = p _ { \mathcal { X } | \mathcal { Y } } ( \pmb { x } \mid y ) p _ { \mathcal { Y } } ^ { \mathrm { t r } } ( y ) , \quad p _ { \mathcal { X } \times \mathcal { Y } } ^ { \mathrm { t e } } ( \pmb { x } , y ) = p _ { \mathcal { X } | \mathcal { Y } } ( \pmb { x } \mid y ) p _ { \mathcal { Y } } ^ { \mathrm { t e } } ( y ) ,
$$

$$
\begin{array} { r } { p _ { \mathcal { X } \times \mathcal { Y } } ^ { \mathrm { t r } } ( \pmb { x } , y ) = p _ { \mathcal { Y } | \mathcal { X } } ^ { \mathrm { t r } } ( y \mid \pmb { x } ) p _ { \mathcal { X } } ^ { \mathrm { t r } } ( \pmb { x } ) , \quad p _ { \mathcal { X } \times \mathcal { Y } } ^ { \mathrm { t e } } ( \pmb { x } , y ) = p _ { \mathcal { Y } | \mathcal { X } } ^ { \mathrm { t e } } ( y \mid \pmb { x } ) p _ { \mathcal { X } } ^ { \mathrm { t e } } ( \pmb { x } ) . } \end{array}
$$

The first factorization reflects the target shift assumption, under which the conditional distribution $p _ { \mathcal { X } | \mathcal { Y } }$ remains invariant; the second is the standard chain-rule factorization, and $p _ { \mathcal { V } | \mathcal { X } } ^ { \mathrm { t r } }$ does not necessarily equal $p _ { \mathcal { V } | \mathcal { X } } ^ { \mathrm { t e } }$ . Using these factorizations, we derive a key relation for the density ratio $\eta _ { \star } ( y ) = p _ { y } ^ { \mathrm { t e } } ( y ) / p _ { y } ^ { \mathrm { t r } } ( y )$

$$
\begin{array} { r l } & { \frac { p _ { \mathcal { X } } ^ { \mathrm { t e } } ( { \pmb x } ) } { p _ { \mathcal { X } } ^ { \mathrm { t r } } ( { \pmb x } ) } = \int _ { { \mathscr { y } } } \frac { p _ { \mathcal { X } \times \mathcal { Y } } ^ { \mathrm { t e } } ( { \pmb x } , { \pmb y } ) } { p _ { \mathcal { X } } ^ { \mathrm { t r } } ( { \pmb x } ) } { \mathrm { d } } { \pmb y } = \int _ { { \mathscr { y } } } \frac { p _ { \mathcal { X } | \mathcal { Y } } ( { \pmb x } \mid { \pmb y } ) p _ { \mathcal { Y } } ^ { \mathrm { t e } } ( { \pmb y } ) } { p _ { \mathcal { X } } ^ { \mathrm { t r } } ( { \pmb x } ) } { \mathrm { d } } { \pmb y } } \\ & { \qquad = \int _ { { \mathscr { y } } } \frac { p _ { \mathcal { Y } } ^ { \mathrm { t e } } ( { \pmb y } ) } { p _ { \mathcal { Y } } ^ { \mathrm { t r } } ( { \pmb y } ) } \cdot \frac { p _ { \mathcal { X } | \mathcal { Y } } ( { \pmb x } \mid { \pmb y } ) p _ { \mathcal { Y } } ^ { \mathrm { t r } } ( { \pmb y } ) } { p _ { \mathcal { X } } ^ { \mathrm { t r } } ( { \pmb x } ) } { \mathrm { d } } { \pmb y } = \int _ { { \mathscr { y } } } \eta _ { \star } ( { \pmb y } ) \cdot p _ { \mathcal { Y } | \mathcal { X } } ^ { \mathrm { t r } } ( { \pmb y } \mid { \pmb x } ) { \mathrm { d } } { \pmb y } . } \end{array}
$$

Multiplying both sides by $K _ { \mathcal { X } } ( \cdot , \pmb { x } ) p _ { \mathcal { X } } ^ { \mathrm { t r } } ( \pmb { x } )$ and integrating over X yields

$$
\begin{array} { r l } { \displaystyle \int _ { \mathcal { X } } K _ { \mathcal { X } } ( \cdot , \boldsymbol { x } ) p _ { \mathcal { X } } ^ { \mathrm { t e } } ( \boldsymbol { x } ) \mathrm { d } \boldsymbol { x } = \int _ { \mathcal { X } } \frac { p _ { \mathcal { X } } ^ { \mathrm { t e } } ( \boldsymbol { x } ) } { p _ { \mathcal { X } } ^ { \mathrm { t r } } ( \boldsymbol { x } ) } K _ { \mathcal { X } } ( \cdot , \boldsymbol { x } ) p _ { \mathcal { X } } ^ { \mathrm { t r } } ( \boldsymbol { x } ) \mathrm { d } \boldsymbol { x } } & { } \\ { = \displaystyle \int _ { \mathcal { X } } \left( \int _ { \mathcal { D } } \eta _ { \star } ( \boldsymbol { y } ) \cdot p _ { \mathcal { Y } | \mathcal { X } } ^ { \mathrm { t r } } ( \boldsymbol { y } \mid \boldsymbol { x } ) \mathrm { d } \boldsymbol { y } \right) K _ { \mathcal { X } } ( \cdot , \boldsymbol { x } ) p _ { \mathcal { X } } ^ { \mathrm { t r } } ( \boldsymbol { x } ) \mathrm { d } \boldsymbol { x } } & { } \\ { = \displaystyle \int _ { \mathcal { X } \times \mathcal { Y } } \eta _ { \star } ( \boldsymbol { y } ) K _ { \mathcal { X } } ( \cdot , \boldsymbol { x } ) p _ { \mathcal { X } \times \mathcal { Y } } ^ { \mathrm { t r } } ( \boldsymbol { x } , \boldsymbol { y } ) \mathrm { d } \boldsymbol { x } \mathrm { d } \boldsymbol { y } . } \end{array}
$$

Equivalently, we obtain the following identity:

$$
\begin{array} { r } { \mathbb { E } _ { ( \pmb { x } , \pmb { y } ) \sim p _ { \pmb { \chi } \times \pmb { y } } ^ { \mathrm { t r } } } [ \eta _ { \star } ( \pmb { y } ) K _ { \mathcal { X } } ( \cdot , \pmb { x } ) ] = \mathbb { E } _ { \pmb { x } ^ { \prime } \sim p _ { \pmb { \chi } } ^ { \mathrm { t e } } } [ K _ { \mathcal { X } } ( \cdot , \pmb { x } ^ { \prime } ) ] . } \end{array}\tag{6}
$$

Let $\pmb { u } _ { \mathbf { 1 } } ^ { \mathrm { t e } }$ be the kernel mean embedding of the test marginal distribution $p _ { \mathcal { X } } ^ { \mathrm { t e } } , \mathrm { i . e . }$

$$
\begin{array} { r } { \pmb { u } _ { 1 } ^ { \mathrm { t e } } = \mathbb { E } _ { \pmb { x } ^ { \prime } \sim p _ { \mathcal { X } } ^ { \mathrm { t e } } } \big [ K _ { \mathcal { X } } ( \cdot , \pmb { x } ^ { \prime } ) \big ] , } \end{array}
$$

and introduce the following cross-covariance operators $^ 1 { : }$

$$
\begin{array} { r } { U _ { \mathcal { Y X } } = \mathbb { E } _ { ( x , y ) \sim p _ { \mathcal { X } \times \mathcal { Y } } ^ { \mathrm { t r } } } [ K _ { \mathcal { Y } } ( \cdot , y ) \otimes K _ { \mathcal { X } } ( \cdot , x ) ] , \quad U _ { \mathcal { X Y } } = \mathbb { E } _ { ( x , y ) \sim p _ { \mathcal { X } \times \mathcal { Y } } ^ { \mathrm { t r } } } [ K _ { \mathcal { X } } ( \cdot , x ) \otimes K _ { \mathcal { Y } } ( \cdot , y ) ] . } \end{array}
$$

Assuming that $\eta _ { \star } \in \mathcal { H } _ { \mathcal { Y } }$ , the identity (6) then simplifies to the operator equation

$$
U _ { \mathcal { X } \mathcal { Y } } \eta _ { \star } = \pmb { u } _ { \mathbf { 1 } } ^ { \mathrm { t e } } .\tag{7}
$$

We exploit (7) to construct an estimator for $\eta _ { \star }$ . The empirical counterparts of the involved quantities are

$$
\widehat { \pmb { u } } _ { 1 } ^ { \mathrm { t e } } = \frac { 1 } { n _ { \eta } } \sum _ { i = 1 } ^ { n _ { \eta } } K _ { \mathcal { X } } ( \cdot , \pmb { x } _ { i } ^ { \prime } ) ,\tag{8}
$$

and

$$
\widehat { U } _ { \mathcal { Y X } } = \frac { 1 } { n _ { \eta } } \sum _ { i = 1 } ^ { n _ { \eta } } K _ { \mathcal { Y } } ( \cdot , y _ { i } ) \otimes K _ { \mathcal { X } } ( \cdot , { \pmb x } _ { i } ) , \quad \widehat { U } _ { \mathcal { X Y } } = \frac { 1 } { n _ { \eta } } \sum _ { i = 1 } ^ { n _ { \eta } } K _ { \mathcal { X } } ( \cdot , { \pmb x } _ { i } ) \otimes K _ { \mathcal { Y } } ( \cdot , y _ { i } ) .\tag{9}
$$

Solving the regularized least-squares problem

$$
\underset { \eta \in \mathcal { H } _ { \mathcal { y } } } { \arg \operatorname* { m i n } } \biggl \| \widehat { U } _ { \mathcal { X } \mathcal { Y } } \eta - \widehat { \pmb { u } } _ { 1 } ^ { \mathrm { t e } } \biggr \| _ { \mathcal { H } _ { \mathcal { x } } } ^ { 2 } + \mu \| \eta \| _ { \mathcal { H } _ { \mathcal { y } } } ^ { 2 } ,
$$

yields the kernel ridge regression estimator $\widehat { \eta } _ { \mu } ^ { \mathrm { K R R } }$ of $\eta _ { \star }$ :

$$
\widehat { \eta } _ { \mu } ^ { \mathrm { K R R } } = ( \widehat { T } + \mu I _ { \mathcal { V } } ) ^ { - 1 } \widehat { U } _ { \mathcal { V X } } \widehat { \mathbf { u } } _ { \mathbf { 1 } } ^ { \mathrm { t e } } ,
$$

where $\mu > 0$ is the regularization parameter and $\widehat { T } = \widehat { U } _ { y x } \widehat { U } _ { x y }$ . The idea of kernel ridge regression extends naturally to a broader class of regularization methods, known as spectral algorithms (de Vito et al., 2005; Lo Gerfo et al., 2008; Bauer et al., 2007). A spectral algorithm modifies the spectrum of the underlying operator by applying a scalar function to each eigenvalue, shrinking the contributions associated with small eigenvalues to control the efective complexity of the hypothesis space. This function is termed the filter function and completely characterizes the regularization strategy.

Definition 1 (Filter functions). A family of functions $g _ { \mu } \colon [ 0 , \kappa ^ { 2 } ] \to [ 0 , \infty )$ , parameterized by $\mu > 0$ constitutes filter functions if:

• There exists $E \geq 0$ such that for all $c \in [ 0 , 1 ]$

$$
\operatorname* { s u p } _ { t \in [ 0 , \kappa ^ { 2 } ] } t ^ { c } g _ { \mu } ( t ) \leq E \cdot \mu ^ { c - 1 } .\tag{10}
$$

<sup>1</sup>For $f _ { 0 } \in \mathcal { H } _ { 1 }$ and $h _ { 0 } \in \mathcal { H } _ { 2 }$ , the tensor product $f _ { 0 } \otimes h _ { 0 }$ defines a rank-1 operator as

$$
f _ { 0 } \otimes h _ { 0 } \colon \quad \mathcal { H } _ { 2 } \to \mathcal { H } _ { 1 } , \quad h \mapsto \left. h _ { 0 } , h \right. _ { \mathcal { H } _ { 2 } } f _ { 0 } .
$$

• There exist $\tau \geq 1$ and $F > 0$ such that for all $c \in [ 0 , \tau ]$

$$
\operatorname* { s u p } _ { t \in [ 0 , \kappa ^ { 2 } ] } t ^ { c } | 1 - t g _ { \mu } ( t ) | \leq F \cdot \mu ^ { c } .\tag{11}
$$

Condition (10) ensures that the regularized inverse remains bounded, guaranteeing numerical stability. Condition (11) controls the approximation error by requiring the residual $| 1 - t g _ { \mu } ( t ) |$ to decay at a rate governed by $\mu .$ The parameter $\tau _ { : }$ , known as the qualification of the regularization method, determines the maximum degree of source smoothness that the algorithm can efectively handle. This framework, originally developed for solving ill-posed linear inverse problems (Engl and Ramlau, 2015), has been adapted to the learning theory (Guo et al., 2017; Fan et al., 2025; Liu and Guo, 2025; Liu et al., 2026) and encompasses a variety of regularization methods, including:

• Kernel ridge regression: $g _ { \mu } ^ { \mathrm { K R R } } ( t ) = ( t + \mu ) ^ { - 1 }$ , with qualification $\tau = 1$ and constants $E =$ $F = 1$ ; here $\mu$ serves as the regularization parameter.

• Early-stopped gradient flow: $g _ { \mu } ^ { \operatorname { G F } } ( t ) = t ^ { - 1 } ( 1 - \mathrm { e } ^ { - t / \mu } )$ , which achieves arbitrary qualification $\tau \geq 1$ with $E = 1 , F = ( \tau / \mathrm { e } ) ^ { \tau }$ ; the stopping time corresponds to $1 / \mu$

• Spectral cutof: $g _ { \mu } ^ { \mathrm { C U T } } ( t ) = t ^ { - 1 } \mathbf { 1 } _ { t \geq \mu }$ , with arbitrary $\tau \geq 1$ and $E = F = 1$ ; the cutof threshold is $\mu .$

Since the input and output spaces are compact, the kernels are bounded, and we set

$$
\operatorname* { s u p } _ { \pmb { x } \in \mathscr { X } } K _ { \mathscr { X } } ( \pmb { x } , \pmb { x } ) \leq \kappa _ { \mathscr { X } } ^ { 2 } , \quad \operatorname* { s u p } _ { y \in \mathscr { Y } } K _ { \mathscr { Y } } ( y , y ) \leq \kappa _ { \mathscr { Y } } ^ { 2 } .
$$

Consequently, the operator norms

$$
\begin{array} { r } { \begin{array} { r l } { \lVert U _ { y x } \rVert _ { \mathcal { H } _ { x }  \mathcal { H } _ { y } } , } & { { }  \widehat { U } _ { y x }  _ { \mathcal { H } _ { x }  \mathcal { H } _ { y } } , \quad \lVert U _ { x y } \rVert _ { \mathcal { H } _ { y }  \mathcal { H } _ { x } } , } \end{array} \quad  \widehat { U } _ { x y }  _ { \mathcal { H } _ { y }  \mathcal { H } _ { x } } } \end{array} \begin{array} { r l } {  \widehat { U } _ { x y }  _ { \mathcal { H } _ { y }  \mathcal { H } _ { x } } } & { { }  \widehat { U } _ { x y }  _ { \mathcal { H } _ { y }  \mathcal { H } _ { x } } } \end{array}
$$

are all bounded above by $\kappa _ { \mathcal { X } } \kappa _ { \mathcal { Y } }$ . Therefore $\| \widehat { T } \| _ { \mathcal { H } _ { \mathcal { y } }  \mathcal { H } _ { \mathcal { y } } } = \| \widehat { U } _ { \mathcal { X } \mathcal { Y } } \widehat { U } _ { \mathcal { Y } \mathcal { X } } \| _ { \mathcal { H } _ { \mathcal { y } }  \mathcal { H } _ { \mathcal { y } } } \leq ( \kappa _ { \mathcal { X } } \kappa _ { \mathcal { Y } } ) ^ { 2 }$ . Hence, κ in Definition 1 can be taken as $\kappa = \kappa _ { \mathcal { X } } \kappa _ { \mathcal { Y } }$ , and the spectral algorithm yields the following estimator of $\eta _ { \star }$ :

$$
\widehat { \eta } _ { \mu } = g _ { \mu } ( \widehat { T } ) \widehat { U } _ { \mathcal { Y X } } \widehat { \mathbf { u } } _ { 1 } ^ { \mathrm { t e } } .\tag{12}
$$

Finally, since the true density ratio is nonnegative, we refine our estimate by retaining only its positive part:

$$
\widetilde { \eta } _ { \mu } = \operatorname* { m a x } \{ \widehat { \eta } _ { \mu } , 0 \} .
$$

Our final density ratio estimator is therefore $\widetilde { \eta } _ { \mu }$

With the density ratio estimator established, we now estimate the regression function $f _ { \star } ^ { \mathrm { t e } }$ , using an independent sample $\left\{ ( \pmb { x } _ { j } , y _ { j } ) \right\} _ { j = 1 } ^ { n _ { f } } \sim p _ { \mathcal { X } \times \mathcal { Y } } ^ { \mathrm { t r } }$ . Recalling that $f _ { \star } ^ { \mathrm { t e } } ( { \pmb x } ) = \mathbb { E } _ { y \sim p _ { y | \mathcal { X } } ^ { \mathrm { t e } } } [ y \mid { \pmb x } ]$ , we have the

key relation:

$$
\begin{array} { r l } & { \displaystyle \int _ { \mathcal X } f _ { \star } ^ { \mathrm { t e } } ( { \pmb x } ) K _ { \mathcal X } ( \cdot , { \pmb x } ) p _ { \mathcal X } ^ { \mathrm { t e } } ( { \pmb x } ) \mathrm d { \pmb x } = \int _ { \mathcal X } \left( \int _ { \mathcal Y } y p _ { \mathcal y | \mathcal X } ^ { \mathrm { t e } } ( { \pmb y } \mid { \pmb x } ) \mathrm d { \pmb y } \right) K _ { \mathcal X } ( \cdot , { \pmb x } ) p _ { \mathcal X } ^ { \mathrm { t e } } ( { \pmb x } ) \mathrm d { \pmb x } } \\ & { \qquad \quad = \int _ { \mathcal X \times \mathcal Y } y K _ { \mathcal X } ( \cdot , { \pmb x } ) p _ { \mathcal X } ^ { \mathrm { t e } } ( { \pmb x } , { \pmb y } ) \mathrm d { \pmb x } \mathrm d { \pmb y } . } \end{array}
$$

That is,

$$
\begin{array} { r } { { \mathbb E } _ { { \pmb x } \sim p _ { \pmb \chi } ^ { \mathrm { t e } } } \big [ f _ { \star } ^ { \mathrm { t e } } ( { \pmb x } ) K _ { \mathcal { X } } ( \cdot , { \pmb x } ) \big ] = { \mathbb E } _ { ( { \pmb x } , { \pmb y } ) \sim p _ { \mathscr { X } \times \mathscr { Y } } ^ { \mathrm { t e } } } [ { \pmb y } K _ { \mathcal { X } } ( \cdot , { \pmb x } ) ] . } \end{array}
$$

To simplify notation, define the response kernel embedding on the test domain as ${ v } _ { y } ^ { \mathrm { t e } }$ :

$$
{ \pmb v } _ { { \pmb y } } ^ { \mathrm { t e } } = \mathbb { E } _ { ( { \pmb x } , { \pmb y } ) \sim p _ { \mathcal { X } \times \mathcal { Y } } ^ { \mathrm { t e } } } [ { \pmb y } K _ { \mathcal { X } } ( \cdot , { \pmb x } ) ] ,
$$

and the covariance operator on the test domain as $V _ { \mathcal { X } \mathcal { X } }$

$$
V _ { \mathcal { X X } } = \mathbb { E } _ { \pmb { x } \sim p _ { \mathcal { X } } ^ { \mathrm { t e } } } [ K _ { \mathcal { X } } ( \cdot , \pmb { x } ) \otimes K _ { \mathcal { X } } ( \cdot , \pmb { x } ) ] .
$$

Then, assuming $f _ { \star } ^ { \mathrm { t e } } \in \mathcal { H } _ { \mathcal { X } }$ , we obtain

$$
V _ { \mathcal { X X } } f _ { \star } ^ { \mathrm { t e } } = v _ { y } ^ { \mathrm { t e } } .\tag{13}
$$

To estimate $f _ { \star } ^ { \mathrm { t e } }$ using (13), we introduce the auxiliary operator and quantity

$$
W _ { \mathcal { X X } } = \mathbb { E } _ { ( x , y ) \sim p _ { \mathcal { X } \times \mathcal { Y } } ^ { \mathrm { t r } } } [ \widetilde { \eta } _ { \mu } ( y ) K _ { \mathcal { X } } ( \cdot , x ) \otimes K _ { \mathcal { X } } ( \cdot , x ) ] , \quad w _ { y } = \mathbb { E } _ { ( x , y ) \sim p _ { \mathcal { X } \times \mathcal { Y } } ^ { \mathrm { t r } } } [ \widetilde { \eta } _ { \mu } ( y ) y K _ { \mathcal { X } } ( \cdot , x ) ] .
$$

These are the importance-weighted versions of $V _ { \mathcal { X } \mathcal { X } }$ and $v _ { y } ^ { \mathrm { t e } }$ . Their empirical counterparts are

$$
\widehat { W } \chi \chi = \frac { 1 } { n _ { f } } \sum _ { j = 1 } ^ { n _ { f } } \widetilde { \eta } _ { \mu } ( y _ { j } ) K \chi ( \cdot , \pmb { x } _ { j } ) \otimes K \chi ( \cdot , \pmb { x } _ { j } ) , \quad \widehat { w } _ { y } = \frac { 1 } { n _ { f } } \sum _ { j = 1 } ^ { n _ { f } } \widetilde { \eta } _ { \mu } ( y _ { j } ) y _ { j } K \chi ( \cdot , \pmb { x } _ { j } ) .
$$

Applying a spectral algorithm then gives an estimator of $f _ { \star } ^ { \mathrm { t e } }$

$$
\widehat { f } _ { \lambda } = g _ { \lambda } ( \widehat { W } _ { \mathcal { X } \mathcal { X } } ) \widehat { \pmb { w } } _ { y } ,\tag{14}
$$

where $g _ { \lambda }$ is a filter function from Definition 1 with regularization parameter $\lambda > 0$ . Note that as $n _ { \eta }$ increases, with a suitable choice of $\mu ,$ the estimate $\widehat { \eta } _ { \mu }$ (and hence $\widetilde { \eta } _ { \mu } = \operatorname* { m a x } \{ \widehat { \eta } _ { \mu } , 0 \} )$ converges to $\eta _ { \star }$ with high probability; consequently, $\widehat { W } _ { \mathcal { X } \mathcal { X } }$ converges to $V _ { \mathcal { X } \mathcal { X } }$ with high probability. Therefore, without loss of generality, we may set κ in Definition 1 as $\kappa _ { \mathcal { X } } ^ { 2 }$ here, since $\| V \chi \chi \| _ { \mathcal { H } _ { \mathcal { X } } \to \mathcal { H } _ { \mathcal { X } } } \le \kappa _ { \mathcal { X } } ^ { 2 }$

## 2.2 Main Results

In this subsection, we derive high-probability convergence guarantees for the density ratio estimator $\widehat { \eta } _ { \mu }$ and the subsequent regression function estimator $\widehat { f } _ { \lambda }$

We first analyze the convergence of $\widehat { \eta } _ { \mu }$ to the true density ratio $\eta _ { \star }$ . Recall that $\widehat { \eta } _ { \mu } = g _ { \mu } ( \widehat { T } ) \widehat { U } _ { \mathcal { Y } \mathcal { X } } \widehat { \mathbf { u } } _ { 1 } ^ { \mathrm { t e } }$ is defined by the spectral regularization scheme (12), using samples $\{ ( { \pmb x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n _ { \eta } } \sim p _ { \mathcal { X } \times \mathcal { Y } } ^ { \mathrm { t r } }$ and $\{ \pmb { x } _ { i } ^ { \prime } \} _ { i = 1 } ^ { n _ { \eta } } \sim p _ { \mathcal { X } } ^ { \mathrm { t e } }$ . Here, $g _ { \mu }$ is the filter function introduced in Definition 1 with regularization parameter $\mu > 0$ , and $\widehat { T } = \widehat { U } _ { y x } \widehat { U } _ { x y }$ . The quantities $\widehat { \mathbf { u } } _ { 1 } ^ { \mathrm { t e } } , \widehat { U } _ { \mathcal { Y X } }$ and $\widehat { U } _ { X \mathcal { Y } }$ are defined by (8) and (9) respectively.

To obtain a quantitative convergence rate, we impose a source (or regularity) condition on $\eta _ { \star }$ relative to the operator underlying the estimation problem. Specifically, we assume that $\eta _ { \star }$ lies in the range of a fractional power of $T = U _ { \mathcal { V X } } U _ { \mathcal { X Y } }$

Assumption 1 (Source condition of $\eta _ { \star } )$ . Let τ be the qualification parameter from Definition 1 and let $\eta _ { \star } = p _ { y } ^ { \mathrm { t e } } / p _ { y } ^ { \mathrm { t r } }$ denote the density ratio. There exist $\iota \in ( 0 , \tau ]$ and $\varpi _ { \star } \in \mathcal { H } _ { \mathcal { Y } }$ such that

$$
\eta _ { \star } = T ^ { \iota } \varpi _ { \star } .
$$

In Assumption 1, the exponent ι quantifies the regularity of $\eta _ { \star }$ with respect to the spectral decomposition of $T _ { i }$ , with larger values of ι corresponding to stronger regularity. The restriction $\iota \leq \tau$ ensures that the qualification of the filter $g _ { \mu }$ is suficient to exploit the prescribed source condition without saturation. Source conditions of this form are standard in the analysis of kernelbased learning and spectral regularization methods (Smale and Zhou, 2003; Cucker and Zhou, 2007; Caponnetto and de Vito, 2007).

Remark. Assumption 1 not only quantifies the regularity of $\eta _ { \star }$ , but also ensures its identifiability from the data. The fundamental relationship (7),

$$
U _ { \mathcal { X } \mathcal { Y } } \eta _ { \star } = \pmb { u } _ { \mathbf { 1 } } ^ { \mathrm { t e } } ,
$$

generally admits infinitely many solutions because $U _ { \mathcal { X } \mathcal { Y } }$ may possess a nontrivial null space. However, the source condition $\eta _ { \star } = T ^ { \iota } \varpi$ <sub>⋆</sub> with $T = U _ { y x } U _ { x y }$ forces $\eta _ { \star }$ to be orthogonal to that null space: for any $h \in \mathcal { H } _ { \mathcal { Y } }$ satisfying $U _ { \mathcal { X } \mathcal { Y } } h = 0$ , we have $T h = 0$ , hence $T ^ { \iota } h = 0$ , and therefore

$$
\langle \eta _ { \star } , h \rangle _ { \mathcal { H } _ { \mathcal { Y } } } = \langle T ^ { \iota } \varpi _ { \star } , h \rangle _ { \mathcal { H } _ { \mathcal { Y } } } = \langle \varpi _ { \star } , T ^ { \iota } h \rangle _ { \mathcal { H } _ { \mathcal { Y } } } = 0 .
$$

Thus $\eta _ { \star }$ coincides exactly with the unique minimal-norm solution of $U _ { \mathcal { X } \mathcal { Y } } \eta = u _ { 1 } ^ { \mathrm { t e } }$ . This property guarantees that the true density ratio can in principle be recovered by regularization algorithms that promote small norms, and it underpins the convergence analysis that follows.

In kernel-based learning theory, it is customary to study both the RKHS-norm and the ${ \mathcal { L } } ^ { 2 } .$ -norm. These two norms are linked through the covariance operator

$$
U _ { \mathcal { V V } } = \mathbb { E } _ { ( \pmb { x } , \pmb { y } ) \sim p _ { \mathcal { X } \times \mathcal { Y } } ^ { \mathrm { t r } } } [ K _ { \mathcal { V } } ( \cdot , \pmb { y } ) \otimes K _ { \mathcal { Y } } ( \cdot , \pmb { y } ) ] .
$$

Indeed, for every $h \in \mathcal { H } _ { \mathcal { Y } }$ the isometric property $\| h \| _ { \mathcal { L } ^ { 2 } ( \mathcal { V } , p _ { y } ^ { \mathrm { t r } } ) } = \| U _ { y y } ^ { 1 / 2 } h \| _ { \mathcal { H } _ { y } }$ holds. A standard route to $\scriptstyle { \mathcal { L } } ^ { 2 } - { \mathrm { n o r m } }$ bounds is therefore to write

$$
\begin{array} { r } { \| \widehat { \eta } _ { \mu } - \eta _ { \star } \| _ { \mathcal { L } ^ { 2 } ( \mathcal { V } , p _ { \mathcal { V } } ^ { \mathrm { t r } } ) } = \left\| U _ { \mathcal { V } \mathcal { V } } ^ { 1 / 2 } \left( \widehat { \eta } _ { \mu } - \eta _ { \star } \right) \right\| _ { \mathcal { H } _ { \mathcal { V } } } , } \end{array}\tag{15}
$$

and then control the RKHS-norm on the right-hand side. In the present setting, the estimator $\widehat { \eta } _ { \mu }$ is built from the empirical operator $\widehat { T } = \widehat { U } _ { \mathcal { Y X } } \widehat { U } _ { \mathcal { X Y } } ;$ by concentration, its error naturally decomposes along the spectral calculus of $T = U _ { y x } U _ { x y }$ . To translate the right-hand side of (15) into a form that can be handled via such spectral estimates—for instance, $\| T ^ { \alpha / 2 } \left( \widehat { \eta } _ { \mu } - \eta _ { \star } \right) \| _ { \mathcal { H } _ { \mathcal { Y } } } -$ one would need an inequality of the form

$$
U _ { y y } \preceq c T ^ { \alpha } , \quad c > 0 , \quad \alpha > 0 ,\tag{16}
$$

relating the two positive self-adjoint operators on $\mathcal { H } _ { \mathcal { Y } }$ . (The notation $\preceq$ means that $c T ^ { \alpha } - U y y$ is positive semidefinite.)

However, while the elementary bound $T \preceq \kappa _ { \mathcal { X } } ^ { 2 } U _ { \mathcal { Y } \mathcal { Y } }$ always holds <sup>2</sup>, the reverse inequality does not hold in general. The operator T captures only the variability in $\mathcal { V }$ that remains distinguishable after conditioning on $\mathcal { X } .$ . If two distinct targets $y _ { 1 } \neq y _ { 2 }$ give rise to nearly indistinguishable conditional distributions $p _ { \mathcal { X } | \mathcal { Y } } ( \cdot \mid y _ { 1 } ) \approx p _ { \mathcal { X } | \mathcal { Y } } ( \cdot \mid y _ { 2 } )$ , then $U _ { \mathcal { X } \mathcal { Y } }$ maps the corresponding directions in $\mathcal { H } _ { \mathcal { Y } }$ to almost the same point in $\mathcal { H } _ { \mathcal { X } }$ , so that $T$ strongly suppresses them. In contrast, $U _ { \mathcal { V } \mathcal { V } }$ weights every direction according to the marginal $p _ { \mathcal { Y } } ^ { \mathrm { t r } }$ , irrespective of its discriminability from $\mathcal { X } .$ . Consequently, the spectrum of $T$ may be markedly smaller than that of $U _ { \mathcal { V } \mathcal { V } }$ , and a reverse domination of the form $U y y \preceq c T ^ { \alpha }$ cannot be guaranteed without strong additional assumptions.

Without a reverse bound of the type (16), decay rates for $\lVert T ^ { \alpha / 2 } \left( \widehat { \eta } _ { \mu } - \eta _ { \star } \right) \rVert _ { \mathcal { H } _ { \ 3 } }$ cannot be converted into meaningful rates for $\| U _ { y y } ^ { 1 / 2 } \left( \widehat { \eta } _ { \mu } - \eta _ { \star } \right) \| _ { \mathcal { H } _ { y } } = \| \widehat { \eta } _ { \mu } - \eta _ { \star } \| _ { \mathcal { L } ^ { 2 } ( \mathcal { V } , p _ { y } ^ { \mathrm { t r } } ) }$ . For this reason, the theorem that follows provides convergence guarantees only in the $\mathcal { H } _ { \mathrm { { y } - \mathrm { { n o r m } } } }$

Theorem 1. Suppose that Assumption 1 holds with $\iota \in ( 0 , \tau ]$ and set the regularization parameter $\mu = n _ { \eta } ^ { - 1 / ( 2 \iota + 2 ) }$ . Then, for any $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta ,$ if the sample size $n _ { \eta }$ satisfies

$$
n _ { \eta } \ge \left( 4 0 ( \kappa _ { \mathcal { X } } \kappa _ { \mathcal { Y } } ) ^ { 2 } \log \frac { 4 } { \delta } \right) ^ { \frac { 2 \iota + 2 } { \iota } } ,\tag{17}
$$

the density ratio estimator $\widehat { \eta } _ { \mu }$ defined in (12) obeys

$$
\| \eta _ { \star } - \widehat \eta _ { \mu } \| _ { \mathcal { H } _ { \mathcal { Y } } } \le \varDelta _ { \eta } \cdot n _ { \eta } ^ { - \frac \iota { 2 \iota + 2 } } \log \frac 4 \delta ,
$$

where $\varDelta _ { \eta }$ is a constant independent of $n _ { \eta }$ and $\delta ,$ specified in (27).

Theorem 1 shows that the density ratio estimator $\widehat { \eta } _ { \mu }$ attains the convergence rate $O ( n _ { \eta } ^ { - \iota / ( 2 \iota + 2 ) } )$ in the RKHS-norm. According to Caponnetto and de Vito (2007), this rate matches the capacityindependent minimax optimal rate under a source condition with parameter $\iota > 0$

Having obtained an estimator $\widehat { \eta } _ { \mu }$ of the density ratio, we next incorporate it into the importanceweighting framework to estimate the target regression function $f _ { \star } ^ { \mathrm { t e } }$ . To ensure nonnegativity of the estimated importance weights, we define the truncated estimator: $\widetilde { \eta } _ { \mu } = \operatorname* { m a x } \{ \widehat { \eta } _ { \mu } , 0 \}$ . The regression function estimator is then defined as in (14):

$$
\widehat { f } _ { \lambda } = g _ { \lambda } ( \widehat { W } _ { \mathcal { X } \mathcal { X } } ) \widehat { \pmb { w } } _ { y } ,
$$

where $g _ { \lambda }$ is the filter function from Definition 1 with regularization parameter $\lambda > 0$ , and

$$
\widehat { W } \chi \chi = \frac { 1 } { n _ { f } } \sum _ { j = 1 } ^ { n _ { f } } \widetilde { \eta } _ { \mu } ( y _ { j } ) K \chi ( \cdot , \pmb { x } _ { j } ) \otimes K \chi ( \cdot , \pmb { x } _ { j } ) , \quad \widehat { \pmb { w } } _ { y } = \frac { 1 } { n _ { f } } \sum _ { j = 1 } ^ { n _ { f } } \widetilde { \eta } _ { \mu } ( y _ { j } ) y _ { j } K \chi ( \cdot , \pmb { x } _ { j } ) .
$$

To analyze the convergence of $\widehat { f } _ { \lambda }$ , we impose a source condition that characterizes the regularity of the regression function $f _ { \star } ^ { \mathrm { t e } }$ , analogously to Assumption 1.

Assumption 2 (Source condition of $f _ { \star } ^ { \mathrm { t e } } )$ . Let τ be the qualification parameter from Definition 1. There exists $r \in ( 0 , \tau - 1 / 2 ]$ such that

$$
f _ { \star } ^ { \mathrm { t e } } = V _ { \mathcal { X } \mathcal { X } } ^ { r } \psi _ { \star }
$$

holds for some $\psi _ { \star } \in \mathcal { H } _ { \mathcal { X } }$

We are now ready to state the convergence guarantees for the regression estimator.

Theorem 2. Let $\delta \in ( 0 , 1 )$ . Assume that the conditions of Theorem 1 hold with $\iota \in ( 0 , \tau ]$ , and that Assumption 2 holds with $r \in ( 0 , \tau - 1 / 2 ]$ . In addition to (17), assume that the sample sizes $n _ { \eta } , n _ { f }$ and the regularization parameter λ satisfy

$$
\left\{ \begin{array} { l l } { \displaystyle \lambda n _ { \eta } ^ { \frac { \iota } { 2 \iota + 2 } } \geq 4 \kappa _ { \chi } ^ { 2 } \kappa y \varDelta _ { \eta } \log \frac 8 \delta , } \\ { \quad } \\ { \displaystyle \lambda n _ { f } ^ { 1 / 2 } \geq 4 0 M _ { \eta } \kappa _ { \chi } ^ { 2 } \log \frac 8 \delta . } \end{array} \right.\tag{18}
$$

Then, with probability at least $1 - \delta$ , the estimator $\widehat { f } _ { \lambda }$ achieves the following error bounds:

1. $I f n _ { \eta } ^ { 2 \iota / ( 2 \iota + 2 ) } < n _ { f } .$ , choosing

$$
\lambda = n _ { \eta } ^ { - \frac { \iota } { 2 \iota + 2 } \cdot \frac { 1 } { r + 1 } } ,
$$

yields

$$
\left\| f _ { \star } ^ { \mathrm { t e } } - \widehat f _ { \lambda } \right\| _ { \mathcal { L } ^ { 2 } ( \mathcal { X } , p _ { \mathcal { X } } ^ { \mathrm { t e } } ) } \leq \varDelta _ { f } \cdot n _ { \eta } ^ { - \frac { \iota } { 2 \iota + 2 } \cdot \frac { r + 1 / 2 } { r + 1 } } \log \frac { 8 } { \delta } , \quad \left\| f _ { \star } ^ { \mathrm { t e } } - \widehat f _ { \lambda } \right\| _ { \mathcal { H } _ { \mathcal { X } } } \leq \varDelta _ { f } \cdot n _ { \eta } ^ { - \frac { \iota } { 2 \iota + 2 } \cdot \frac { r } { r + 1 } } \log \frac { 8 } { \delta } .
$$

2. $I f n _ { \eta } ^ { 2 \iota / ( 2 \iota + 2 ) } \geq n _ { f }$ , choosing

$$
\lambda = n _ { f } ^ { - \frac { r } { 2 r + 2 } } ,
$$

yields

$$
\left\| f _ { \star } ^ { \mathrm { t e } } - \widehat f _ { \lambda } \right\| _ { \mathcal L ^ { 2 } ( \mathcal { X } , p _ { \mathcal X } ^ { \sf t e } ) } \leq \varDelta _ { f } \cdot n _ { f } ^ { - \frac { r + 1 / 2 } { 2 r + 2 } } \log \frac 8 \delta , \quad \left\| f _ { \star } ^ { \mathrm { t e } } - \widehat f _ { \lambda } \right\| _ { \mathcal { H } _ { \mathcal { X } } } \leq \varDelta _ { f } \cdot n _ { f } ^ { - \frac { r } { 2 r + 2 } } \log \frac 8 \delta .
$$

Here, $\varDelta _ { f }$ is a constant given in (37) that is independent of $n _ { \eta } , ~ n _ { f }$ , and $\delta .$

According to Caponnetto and de Vito (2007), under a source condition with parameter $r > 0$ the capcity-independent minimax optimal convergence rates are $O ( n _ { f } ^ { - ( r + 1 / 2 ) / ( 2 r + 2 ) } )$ in the $\textstyle { \mathcal { L } } ^ { 2 } .$ norm and $O ( n _ { f } ^ { - r / ( 2 r + 2 ) } )$ in the RKHS-norm. Hence, Theorem 2 shows that when $n _ { \eta } ^ { 2 \iota / ( 2 \iota + 2 ) } \geq n _ { f } .$ the estimator $\widehat { f } _ { \lambda }$ attains these minimax optimal rates. In contrast, when $n _ { \eta } ^ { 2 \iota / ( 2 \iota + 2 ) } < n _ { f }$ , only suboptimal rates are obtained. Because $2 \iota / ( 2 \iota + 2 ) < 1$ for all $\iota > 0$ , the favorable regime requires the sample size $n _ { \eta }$ for density ratio estimation to scale polynomially with the sample size $n _ { f }$ for regression estimation. This condition is somewhat stringent, as both steps rely on costly labeled sample pairs.

## 3 Related Work and Discussions

In this section, we review the literature on density ratio estimation and target shift, discuss the implications of our main results, and conclude with limitations of our analysis and several directions for future research.

## 3.1 Density Ratio Estimation

Estimating the density ratio between two probability distributions is a fundamental problem in many areas of machine learning (Sugiyama et al., 2012). A naive approach estimates the numerator and denominator densities separately and then takes their ratio, typically using parametric models or nonparametric techniques such as Kernel Density Estimation (KDE) (Sheather and Jones, 1991).

However, this indirect method sufers from the curse of dimensionality and amplifies estimation errors, particularly in regions where the denominator density is near zero. To overcome these limitations, the prevailing paradigm has shifted toward direct methods that avoid estimating the individual densities.

Early direct methods recast the problem as probabilistic classification. By training a binary classifier (e.g., logistic regression) to discriminate between samples from the two distributions, the density ratio can be recovered from the resulting class-posterior probabilities (Qin, 1998; Bicke et al., 2009). In parallel, moment matching techniques have emerged as a powerful alternative, with Kernel Mean Matching (KMM) (Gretton et al., 2008) being the most prominent example. KMM circumvents explicit density modeling by directly computing the density ratio that align the kernel mean embeddings of the two distributions in a Reproducing Kernel Hilbert Space (RKHS). The elegance of this principle has motivated a wealth of recent methodological and theoretical advances in density ratio estimation (Gizewski et al., 2022, 2026; Myleiko and Solodky, 2025; Liu et al., 2026).

A more general and theoretically richer framework for direct estimation is based on divergence minimization and ratio fitting. Methods in this family optimize the ratio by minimizing a chosen statistical divergence. Prominent examples include the Kullback-Leibler Importance Estimation Procedure (KLIEP) (Sugiyama et al., 2008), which minimizes the KL divergence, and Least-Squares Importance Fitting (LSIF) (Kanamori et al., 2009), which minimizes the squared loss.

More recently, the literature addresses high-dimensional and highly discrepant scenarios through structural and architectural innovations. This progress includes telescoping and path-based estimators that bridge large density chasms (Rhodes et al., 2020; Choi et al., 2022), as well as methods that explicitly handle unbounded density ratios (Feng et al., 2024; Xu et al., 2025; Zheng et al., 2026; Liu et al., 2026).

Following this line of research, we exploit the identity $U _ { \mathcal { X } \mathcal { Y } } \eta _ { \star } = u _ { 1 } ^ { \mathrm { t e } }$ derived in (7), where $u _ { 1 } ^ { \mathrm { t e } }$ is the kernel mean embedding of the test input distribution. Estimating the density ratio $\eta _ { \star }$ therefore reduces to solving a regularized operator equation, which we solve using a spectral algorithm. This construction can be viewed as a structured variant of kernel mean matching adapted to the target shift setting, and it naturally unifies several common regularization schemes.

## 3.2 Discrete and Continuous Target Shift

Under target shift, the conditional distribution of the input given the output remains invariant across domains, while the marginal distribution of the output changes. As a consequence, the density ratio depends only on the output variable, substantially simplifying the structure of the distribution shift. This setting has motivated a considerable body of work, particularly in classification and, more recently, in regression.

Most existing studies on target shift have focused on the discrete case, i.e., classification problems in which the output variable takes values in a finite set. In this setting, the importance weights reduce to a finite-dimensional vector indexed by the class labels, making both estimation and theoretical analysis considerably more tractable. Early work by Saerens et al. (2002) introduced an Expectation-Maximization (EM) algorithm to estimate the target label distribution, assuming a predictor with well-calibrated class probabilities. Lipton et al. (2018) relaxed this calibration requirement by proposing Black Box Shift Estimation (BBSE), which solves a linear system based on the confusion matrix of an arbitrary predictor, and established consistency along with finitesample error bounds. Building on this framework, Azizzadenesheli et al. (2019) added regularization to improve stability with limited data and derived generalization bound for importance-weighted classifiers under label shift without labeled test data. From an empirical perspective, Alexandari et al. (2020) showed that combining modern calibration techniques (e.g., bias-corrected temperature scaling) with the EM algorithm often yields better performance. Subsequently, Garg et al. (2020) unified these approaches, revealing that confusion matrix estimators implicitly perform a coarse calibration, and proved that maximum likelihood estimation is consistent under the same invertibility conditions as BBSE when a calibrated predictor is used. Taken together, these developments provide a mature theoretical and algorithmic framework for discrete target shift, where the finite-dimensional structure of the label space greatly facilitates both estimator construction and statistical analysis.

In the continuous setting, corresponding to regression problems, the importance weight becomes an unknown function that need to be estimated from data, and its characterization leads to an infinite-dimensional integral equation that is inherently ill-posed (Kress, 2014). Consequently, small perturbations in the data may induce large deviations in an unregularized solution, making regularization essential. Despite the practical relevance of regression under target shift, the continuous setting has received comparatively limited attention. The existing literature is largely algorithmic. Zhang et al. (2013) extended kernel mean matching to target shift by matching the kernel mean embedding of the test input distribution with that of a reweighted training distribution, leading to a quadratic program that estimates the density ratio at the observed training outputs. Nguyen et al. (2016) instead proposed estimating the density ratio function directly by minimizing an $\textstyle { \mathcal { L } } ^ { 2 } .$ -based discrepancy between the corresponding distributions, resulting in a constrained optimization problem. More recently, Kim et al. (2024) developed a nonparametric regularized approach that formulates density ratio estimation as a Tikhonov-regularized integral equation, with the unknown distributions approximated by kernel density estimators, and established consistency of the resulting estimator.

These pioneering studies demonstrate the practical feasibility of adaptation under continuous target shift. Nevertheless, a substantial theoretical gap remains: existing methods do not provide explicit sample-size-dependent convergence rates for the estimated density ratio, let alone establish their optimality. For example, although Kim et al. (2024) proved consistency and derived convergence bounds depending on the bandwidth and regularization parameters, these parameters were not optimized with respect to the sample size, and no rate-optimality result was established. In parallel, Gogolashvili (2026) recently derived minimax-optimal rates for importance-weighted kernel ridge regression under target shift, but assumed that the true density ratio is known and thus did not address the density ratio estimation problem. Consequently, a rigorous finite-sample theory for continuous density ratio estimation, providing explicit convergence rates together with an understanding of their optimality, has remained lacking.

## 3.3 Discussions on Our Results

Our results address the theoretical gap identified above by providing explicit finite-sample convergence guarantees for continuous density ratio estimation under target shift. In particular, Theorem 1 shows that the proposed spectral estimator achieves the capacity-independent minimaxoptimal rate $O ( n _ { \eta } ^ { - \iota / ( 2 \iota + 2 ) } )$ in the RKHS-norm for estimating the unknown density ratio. Here $\iota \in ( 0 , \tau ]$ is the regularity parameter (Assumption 1) and $n _ { \eta }$ denotes the common size of the labeled training sample and the unlabeled test sample used for ratio estimation. To the best of our knowledge, this provides the first explicit rate-optimal finite-sample guarantee for density ratio estimation under continuous target shift, thereby strengthening the existing consistency results.

Theorem 2 further quantifies how the estimated ratio afects the downstream importanceweighted regression estimator. When the sample size used for density ratio estimation is suficiently large relative to that used for regression, namely, ${ n _ { \eta } ^ { 2 \iota / ( 2 \iota + 2 ) } \geq n _ { f } }$ , the resulting regression estimator achieves the capacity-independent minimax-optimal rates for standard kernel regression: $O ( n _ { f } ^ { - ( r + 1 / 2 ) / ( 2 r + 2 ) } )$ in the $\scriptstyle { \mathcal { L } } ^ { 2 } - { \mathrm { n o r m } }$ and $O ( n _ { f } ^ { - r / ( 2 r + 2 ) } )$ in the RKHS-norm, where $r \in ( 0 , \tau - 1 / 2 ]$ is the regularity parameter of the regression function specified in (Assumption 2). In the opposite regime, the overall convergence rate is limited by the accuracy of the estimated density ratio and is therefore slower than the minimax-optimal regression rate. These results provide a finite-sample theoretical foundation for importance-weighted regression under continuous target shift and clarify how the statistical accuracy of density ratio estimation afects downstream prediction.

In addition, our analysis also has several limitations that point to directions for future research.

• Assumption 1 requires the density ratio to lie in the range of a fractional power of $T =$ $U _ { \mathcal { Y X } } U _ { \mathcal { X } \mathcal { Y } }$ , which excludes components of $\mathcal { H } _ { \mathcal { Y } }$ that are only weakly coupled to $\mathcal { H } _ { \mathcal { X } }$ through the cross-covariance structure. This condition, while essential for our analysis, is more restrictive than typical regularity assumptions in standard kernel regression. It therefore remains an interesting question whether alternative estimation principles can achieve comparable convergence guarantees under weaker regularity conditions.

• As discussed in Section 2.2, our convergence guarantees for the density ratio are established in the RKHS norm, whereas corresponding rates in the ${ \mathcal { L } } ^ { 2 } .$ -norm are not derived. Establishing ${ \mathcal { L } } ^ { 2 } .$ -convergence rates, and determining whether they are minimax optimal, is an important direction for future work.

• Theorem 2 shows that the regression estimator achieves minimax-optimal rates only when ${ n } _ { \eta } ^ { 2 \iota / ( 2 \iota + 2 ) } \geq n _ { f } ,$ i.e., the sample size used for density ratio estimation need to scale polynomially with the regression sample size. In particular, increasing $n _ { \eta }$ requires additional labeled training samples, which may be costly to obtain. Developing more sample-eficient density ratio estimation and importance-weighting procedures that relax this sample-size requirement is therefore an important direction for future research.

## 4 Proofs of Main Results

This section contains the proofs of Theorem 1 and Theorem 2. The argument is based on an error decomposition: the target quantity is split into several components, each component is bounded separately by an auxiliary proposition, and the resulting bounds are combined to obtain the final estimate.

## 4.1 Proof of Theorem 1

In this subsection, we prove our first main result, Theorem 1. We begin by deriving an error decomposition for $\| \eta _ { \star } - \widehat { \eta } _ { \mu } \| _ { \mathcal { H } _ { \mathcal { y } } }$ . To this end, we introduce the auxiliary function

$$
\eta _ { \mu } = g _ { \mu } ( T ) T \eta _ { \star } = g _ { \mu } ( T ) U _ { \mathcal { V X } } U _ { \mathcal { X Y } } \eta _ { \star } = g _ { \mu } ( T ) U _ { \mathcal { V X } } { \bf u _ { 1 } ^ { \mathrm { t e } } } ,
$$

where the equality $U _ { \mathcal { X } \mathcal { Y } } \eta _ { \star } = u _ { 1 } ^ { \mathrm { t e } }$ follows from (7). Then we can write

$$
\eta _ { \star } - { \widehat \eta } _ { \mu } = ( \eta _ { \star } - \eta _ { \mu } ) + ( \eta _ { \mu } - { \widehat \eta } _ { \mu } ) .
$$

Recalling that $\widehat { \eta } _ { \mu } = g _ { \mu } ( \widehat { T } ) \widehat { U } _ { \mathcal { Y } \mathcal { X } } \widehat { \mathbf { u } } _ { 1 } ^ { \mathrm { t e } }$ , we expand $\eta _ { \mu } - \widehat { \eta } _ { \mu }$ as

$$
\begin{array} { r l } & { \eta _ { \mu } - \widehat { \eta } _ { \mu } = \eta _ { \mu } - g _ { \mu } ( \widehat { T } ) \widehat { U } _ { \mathcal { V } \mathcal { X } } \widehat { u } _ { \mathbf { i } } ^ { \mathrm { t e } } } \\ & { \qquad = \left( I _ { \mathcal { V } } - g _ { \mu } ( \widehat { T } ) \widehat { T } + g _ { \mu } ( \widehat { T } ) \widehat { T } \right) \eta _ { \mu } - g _ { \mu } ( \widehat { T } ) \widehat { U } _ { \mathcal { V } \mathcal { X } } \widehat { u } _ { \mathbf { i } } ^ { \mathrm { t e } } } \\ & { \qquad = g _ { \mu } ( \widehat { T } ) \left( \widehat { T } \eta _ { \mu } - \widehat { U } _ { \mathcal { V } \mathcal { X } } \widehat { u } _ { \mathbf { i } } ^ { \mathrm { t e } } \right) + \left( I _ { \mathcal { V } } - g _ { \mu } ( \widehat { T } ) \widehat { T } \right) \eta _ { \mu } , } \end{array}
$$

where $I _ { \mathcal { Y } }$ denotes the identity map on $\mathcal { H } _ { \mathcal { Y } }$ . The term $\widehat { T } \eta _ { \mu } - \widehat { U } _ { \mathcal { Y X } } \widehat { \mathbf { u } } _ { 1 } ^ { \mathrm { t e } }$ can be further decomposed as

$$
\widehat { T } \eta _ { \mu } - \widehat { U } _ { \mathcal { Y } \mathcal { X } } \widehat { \mathbf { u } } _ { \mathbf { 1 } } ^ { \mathrm { t e } } = \Big ( \big ( \widehat { T } \eta _ { \mu } - \widehat { U } _ { \mathcal { Y } \mathcal { X } } \widehat { \mathbf { u } } _ { \mathbf { 1 } } ^ { \mathrm { t e } } \big ) - \big ( T \eta _ { \mu } - U _ { \mathcal { Y } \mathcal { X } } \mathbf { u } _ { \mathbf { 1 } } ^ { \mathrm { t e } } \big ) \Big ) + \big ( T \eta _ { \mu } - U _ { \mathcal { Y } \mathcal { X } } \mathbf { u } _ { \mathbf { 1 } } ^ { \mathrm { t e } } \big ) .
$$

Summarizing, we obtain

$$
\begin{array} { r } { \| \eta _ { \star } - \widehat \eta _ { \mu } \| _ { \mathcal { H } _ { y } } \leq A _ { 1 } + A _ { 2 } + A _ { 3 } + A _ { 4 } , } \end{array}
$$

where

$$
\begin{array} { r } { A _ { 1 } = \left. \eta _ { \star } - \eta _ { \mu } \right. _ { \mathcal { H } _ { \mathcal { V } } } , \quad A _ { 2 } = \left. g _ { \mu } ( \widehat { T } ) \left( ( \widehat { T } \eta _ { \mu } - \widehat { U } _ { \mathcal { V X } } \widehat { u } _ { 1 } ^ { \mathrm { t e } } ) - ( T \eta _ { \mu } - U _ { \mathcal { V X } } u _ { 1 } ^ { \mathrm { t e } } ) \right) \right. _ { \mathcal { H } _ { \mathcal { V } } } , } \end{array}\tag{19}
$$

$$
\begin{array} { r } { A _ { 3 } = \left\| g _ { \mu } ( \widehat { T } ) \left( T \eta _ { \mu } - U _ { \mathcal { V X } } u _ { 1 } ^ { \mathrm { t e } } \right) \right\| _ { \mathcal { H } _ { \mathcal { V } } } , \quad A _ { 4 } = \left\| \left( I _ { \mathcal { V } } - g _ { \mu } ( \widehat { T } ) \widehat { T } \right) \eta _ { \mu } \right\| _ { \mathcal { H } _ { \mathcal { V } } } . } \end{array}
$$

We now bound $A _ { 1 } , A _ { 2 } , A _ { 3 }$ , and $A _ { 4 }$ individually via separate propositions; the bounds will later be combined. As a preliminary tool, we first establish a proposition controlling empirical averages, which will be used repeatedly throughout the proof.

Proposition 1. For any $\delta \in ( 0 , 1 )$ , the following bounds hold simultaneously with probability at least $1 - \delta$

$$
\begin{array} { r } { \left\| { U _ { \mathcal { V X } } - \widehat { U } _ { \mathcal { V X } } } \right\| _ { \mathcal { H } _ { \mathcal { X } } \to \mathcal { H } _ { \mathcal { V } } } \le 1 0 \kappa _ { \mathcal { X } } \kappa _ { \mathcal { V } } \cdot n _ { \eta } ^ { - 1 / 2 } \log \frac { 4 } { \delta } , \quad \left\| { u _ { 1 } ^ { \mathrm { t e } } - \widehat { u } _ { 1 } ^ { \mathrm { t e } } } \right\| _ { \mathcal { H } _ { \mathcal { X } } } \le 1 0 \kappa _ { \mathcal { X } } \cdot n _ { \eta } ^ { - 1 / 2 } \log \frac { 4 } { \delta } . } \end{array}
$$

Proof. Define the random variables

$$
\xi _ { 1 } ( x , y ) = K _ { \mathcal { V } } ( \cdot , y ) \otimes K _ { \mathcal { X } } ( \cdot , \pmb { x } ) , \quad ( \pmb { x } , y ) \sim p _ { \mathcal { X } \times \mathcal { Y } } ^ { \mathrm { t r } } ,
$$

and

$$
\xi _ { 2 } ( \pmb { x } ^ { \prime } ) = K _ { \mathcal { X } } ( \cdot , \pmb { x } ^ { \prime } ) , \quad \pmb { x } ^ { \prime } \sim p _ { \mathcal { X } } ^ { \mathrm { t e } } ,
$$

and set $\xi _ { 1 , i } = \xi _ { 1 } ( \pmb { x } _ { i } , y _ { i } ) , \xi _ { 2 , i } = \xi _ { 2 } ( \pmb { x } _ { i } ^ { \prime } ) \ \mathrm { f o r } \ 1 \leq i \leq n _ { \eta }$ . Then

$$
U _ { \mathcal { V X } } - \widehat { U } _ { \mathcal { V X } } = \mathbb { E } _ { ( \pmb { x } , \pmb { y } ) \sim p _ { \mathcal { X } \times \mathcal { Y } } ^ { \mathrm { t r } } } [ \xi _ { 1 } ( \pmb { x } , \pmb { y } ) ] - \frac { 1 } { n _ { \eta } } \sum _ { i = 1 } ^ { n _ { \eta } } \xi _ { 1 , i } ,
$$

$$
{ \pmb u } _ { { \bf 1 } } ^ { \mathrm { t e } } - \widehat { { \pmb u } } _ { { \bf 1 } } ^ { \mathrm { t e } } = \mathbb { E } _ { { \pmb x } ^ { \prime } \sim p _ { \mathcal { X } } ^ { \mathrm { t e } } } \big [ \xi _ { 2 } ( { \pmb x } ^ { \prime } ) \big ] - \frac { 1 } { n _ { \eta } } \sum _ { i = 1 } ^ { n _ { \eta } } \xi _ { 2 , i } .
$$

By construction, the random variables satisfy the following uniform bounds:

$$
\operatorname* { s u p } _ { \substack { \boldsymbol { x } \in \mathcal { H } _ { \boldsymbol { x } } , \boldsymbol { y } \in \mathcal { H } _ { \mathcal { V } } } } \| \xi _ { 1 } ( \boldsymbol { x } , \boldsymbol { y } ) \| _ { \mathrm { H S } ( \mathcal { H } _ { \boldsymbol { x } } , \mathcal { H } _ { \mathcal { V } } ) } \leq \kappa _ { \mathcal { X } } \kappa _ { \mathcal { V } } , \quad \operatorname* { s u p } _ { \boldsymbol { x } ^ { \prime } \in \mathcal { X } } \big \| \xi _ { 2 } ( \boldsymbol { x } ^ { \prime } ) \big \| _ { \mathcal { H } _ { \mathcal { X } } } \leq \kappa _ { \mathcal { X } } ,
$$

where $\mathrm { H S } ( \mathcal { H } _ { \mathcal { X } } , \mathcal { H } _ { \mathcal { Y } } )$ denotes the space of Hilbert–Schmidt operators from $\mathcal { H } _ { \mathcal { X } }$ to $\mathcal { H } _ { \mathcal { Y } }$ . Applying Lemma 1, each of the two inequalities

$$
\begin{array} { r } { \Big \lVert U y x - \widehat { U } y x \Big \rVert _ { \mathcal { H } _ { x } \to \mathcal { H } _ { y } } \le \Big \lVert U  y x - \widehat { U } y x \Big \rVert _ { \mathrm { H S } ( \mathcal { H } _ { x } , \mathcal { H } _ { y } ) } \le 1 0 \kappa _ { \mathcal { X } } \kappa _ { \mathcal { Y } } \cdot n _ { \eta } ^ { - 1 / 2 } \log \frac { 4 } { \delta } , } \end{array}
$$

$$
\big \| \boldsymbol { u } _ { 1 } ^ { \mathrm { t e } } - \widehat { \boldsymbol { u } } _ { 1 } ^ { \mathrm { t e } } \big \| _ { \mathcal { H } _ { \mathcal { X } } } \leq 1 0 \kappa _ { \mathcal { X } } \cdot n _ { \eta } ^ { - 1 / 2 } \log \frac { 4 } { \delta } ,
$$

holds with probability at least $1 - \delta / 2$ . By the union bound, both bounds hold simultaneously with probability at least $1 - \delta$ . This completes the proof. □

Now we present Proposition 2, which provides an upper bound for $A _ { 1 }$ in (19).

Proposition 2. Suppose that Assumption 1 holds with $\iota \in ( 0 , \tau ]$ . Then the following bound holds:

$$
A _ { 1 } = \| \eta _ { \star } - \eta _ { \mu } \| _ { \mathcal { H } _ { \mathcal { Y } } } \leq F \| \varpi _ { \star } \| _ { \mathcal { H } _ { \mathcal { Y } } } \cdot \mu ^ { \iota } .
$$

Proof. Recall that $\eta _ { \mu } = g _ { \mu } ( T ) T \eta _ { \star }$ , and by Assumption 1 we have $\eta _ { \star } = T ^ { \iota } \varpi _ { \star }$ . It follows that

$$
\begin{array} { r l } & { A _ { 1 } = \| \eta _ { \star } - \eta _ { \mu } \| _ { \mathcal H _ { y } } = \| ( I _ { \mathcal V } - g _ { \mu } ( T ) T ) \eta _ { \star } \| _ { \mathcal H _ { y } } = \| ( I _ { \mathcal V } - g _ { \mu } ( T ) T ) T ^ { \iota } \varpi _ { \star } \| _ { \mathcal H _ { y } } } \\ & { \quad \quad \leq \| ( I _ { \mathcal V } - g _ { \mu } ( T ) T ) T ^ { \iota } \| _ { \mathcal H _ { y } \to \mathcal H _ { y } } \cdot \| \varpi _ { \star } \| _ { \mathcal H _ { y } } . } \end{array}
$$

Using the filter function property in (11), we obtain the following bound on the operator norm:

$$
\| ( I _ { \mathcal { V } } - g _ { \mu } ( T ) T ) T ^ { \iota } \| _ { \mathcal { H } _ { \mathcal { V } } \to \mathcal { H } _ { \mathcal { V } } } \le \operatorname* { s u p } _ { 0 \le t \le ( \kappa _ { \mathcal { X } } \kappa _ { \mathcal { V } } ) ^ { 2 } } | 1 - g _ { \mu } ( t ) t | t ^ { \iota } \le F \cdot \mu ^ { \iota } .
$$

Consequently,

$$
A _ { 1 } \leq \| ( I _ { \mathcal { V } } - g _ { \mu } ( T ) T ) T ^ { \iota } \| _ { \mathcal { H } _ { \mathcal { V } }  \mathcal { H } _ { \mathcal { V } } } \cdot \| \varpi _ { \star } \| _ { \mathcal { H } _ { \mathcal { V } } } \leq F \| \varpi _ { \star } \| _ { \mathcal { H } _ { \mathcal { V } } } \cdot \mu ^ { \iota } ,
$$

which completes the proof.

The bound for $A _ { 2 }$ in (19) is given in Proposition 3.

Proposition 3. Suppose that Assumption 1 holds with $\iota \in \mathsf { \Gamma } ( 0 , \tau ]$ , then for any $\delta \in ( 0 , 1 )$ , the following bounds hold with probability at least $1 - \delta$

$$
\begin{array} { r l } & { A _ { 2 } = \left\| g _ { \mu } ( \widehat { T } ) \left( ( \widehat { T } \eta _ { \mu } - \widehat { U } _ { \mathcal { Y X } } \widehat { u } _ { 1 } ^ { \mathrm { t e } } ) - ( T \eta _ { \mu } - U _ { \mathcal { Y X } } u _ { 1 } ^ { \mathrm { t e } } ) \right) \right\| _ { \mathcal { H } _ { \mathcal { Y } } } } \\ & { \qquad \leq 4 0 E ^ { 2 } ( \kappa _ { \mathcal { X } } \kappa _ { \mathcal { Y } } ) ^ { 2 \iota + 2 } \| \varpi _ { \star } \| _ { \mathcal { H } _ { \mathcal { Y } } } \cdot \mu ^ { - 1 } n _ { \eta } ^ { - 1 / 2 } \log \frac { 4 } { \delta } . } \end{array}
$$

Proof. Observing that

$$
\begin{array} { r l } & { \quad ( \widehat { T } \eta _ { \mu } - \widehat { U } _ { \mathcal { Y X } } \widehat { \mathbf { u } } _ { \mathbf { 1 } } ^ { \mathrm { t e } } ) - \left( T \eta _ { \mu } - U _ { \mathcal { Y X } } \mathbf { u } _ { \mathbf { 1 } } ^ { \mathrm { t e } } \right) } \\ & { \quad \quad \quad ( \widehat { T } - T ) \eta _ { \mu } + \left( U _ { \mathcal { Y X } } - \widehat { U } _ { \mathcal { Y X } } \right) \mathbf { u } _ { \mathbf { 1 } } ^ { \mathrm { t e } } + \widehat { U } _ { \mathcal { Y X } } \mathbf { ( } \mathbf { u } _ { \mathbf { 1 } } ^ { \mathrm { t e } } - \widehat { \mathbf { u } } _ { \mathbf { 1 } } ^ { \mathrm { t e } } \mathbf { ) } , } \end{array}
$$

we can bound $A _ { 2 }$ as

$$
A _ { 2 } \leq \| g _ { \mu } ( \widehat { T } ) \| _ { \mathcal { H } _ { \mathcal { Y } }  \mathcal { H } _ { \mathcal { Y } } } \cdot ( A _ { 2 , 1 } + A _ { 2 , 2 } + A _ { 2 , 3 } ) ,
$$

where

$$
A _ { 2 , 1 } = \Big \| ( \widehat { T } - T ) \eta _ { \mu } \Big \| _ { \mathcal { H } _ { y } } , \quad A _ { 2 , 2 } = \Big \| ( U y x - \widehat { U } y x ) u _ { 1 } ^ { \mathrm { t e } } \Big \| _ { \mathcal { H } _ { y } } , \quad A _ { 2 , 3 } = \Big \| \widehat { U } y x \big ( u _ { 1 } ^ { \mathrm { t e } } - \widehat { u } _ { 1 } ^ { \mathrm { t e } } \big ) \Big \| _ { \mathcal { H } _ { y } } .
$$

By the filter function property in (10), we have

$$
\begin{array} { r } { \left\| g _ { \mu } ( \widehat { T } ) \right\| _ { \mathcal { H } _ { \mathcal { y } } \to \mathcal { H } _ { \mathcal { y } } } \le \underset { 0 \le t \le ( \kappa _ { \mathcal { X } } \kappa _ { \mathcal { Y } } ) ^ { 2 } } { \operatorname* { s u p } } g _ { \mu } ( t ) \le E \cdot \mu ^ { - 1 } . } \end{array}\tag{20}
$$

We now bound $A _ { 2 , 1 } , A _ { 2 , 2 }$ , and $A _ { 2 , 3 }$ as follows:

1. For $A _ { 2 , 1 }$ , note that $T - \widehat { T }$ can be expressed as

$$
T - \widehat { T } = U _ { y \chi } U _ { \chi y } - \widehat { U } _ { y \chi } \widehat { U } _ { \chi y } = U _ { y \chi } \left( U _ { \chi y } - \widehat { U } _ { \chi y } \right) + \left( U _ { y \chi } - \widehat { U } _ { y \chi } \right) \widehat { U } _ { \chi y } .
$$

Consequently,

$$
\begin{array} { r l } & { { A _ { 2 , 1 } } = \| ( \widehat { T } - T ) \eta _ { \mu } \| _ { \mathcal { H } _ { y } } \leq \| \widehat { T } - T \| _ { \mathcal { H } _ { y }  \mathcal { H } _ { y } } \cdot \| \eta _ { \mu } \| _ { \mathcal { H } _ { y } } } \\ & { \qquad \leq ( \| U _ { \mathcal { V } \mathcal { X } } ( U _ { \mathcal { X } \mathcal { Y } } - \widehat { U } _ { \mathcal { X } \mathcal { Y } } ) \| _ { \mathcal { H } _ { y }  \mathcal { H } _ { y } } + \| ( U _ { \mathcal { V } \mathcal { X } } - \widehat { U } _ { \mathcal { Y } \mathcal { X } } ) \widehat { U } _ { \mathcal { X } \mathcal { Y } } \| _ { \mathcal { H } _ { y }  \mathcal { H } _ { y } } ) \| \eta _ { \mu } \| _ { \mathcal { H } _ { y } } } \\ & { \qquad \leq 2 \kappa _ { \mathcal { X } \mathcal { K } \mathcal { Y } } \cdot \| U _ { \mathcal { V } \mathcal { X } } - \widehat { U } _ { \mathcal { Y } \mathcal { X } } \| _ { \mathcal { H } _ { \mathcal { X } }  \mathcal { H } _ { \mathcal { Y } } } \cdot \| \eta _ { \mu } \| _ { \mathcal { H } _ { y } } . } \end{array}
$$

In the last inequality, we used the bounds $\| U _ { \mathcal { V X } } \| _ { \mathcal { H } _ { \mathcal { X } }  \mathcal { H } _ { \mathcal { V } } } \le \kappa _ { \mathcal { X } } \kappa _ { \mathcal { Y } }$ and $\| \widehat { U } _ { \mathcal { X } \mathcal { Y } } \| _ { \mathcal { H } _ { \mathcal { Y } }  \mathcal { H } _ { \mathcal { X } } } \le \kappa _ { \mathcal { X } } \kappa y .$ together with the identity

$$
\begin{array} { r } { \left\| U _ { \mathcal { X } \mathcal { Y } } - \widehat { U } _ { \mathcal { X } \mathcal { Y } } \right\| _ { \mathcal { H } _ { \mathcal { Y } } \mathcal { H } _ { \mathcal { X } } } = \left\| U _ { \mathcal { Y } \mathcal { X } } - \widehat { U } _ { \mathcal { Y } \mathcal { X } } \right\| _ { \mathcal { H } _ { \mathcal { X } } \mathcal { H } _ { \mathcal { Y } } } . } \end{array}
$$

By Proposition 1, we have

$$
\left\| U _ { \mathcal { V X } } - \widehat { U } _ { \mathcal { V X } } \right\| _ { \mathcal { H } _ { \mathcal { X } } \mathcal { H } _ { \mathcal { Y } } } \le 1 0 \kappa _ { \mathcal { X } } \kappa _ { \mathcal { Y } } \cdot n _ { \eta } ^ { - 1 / 2 } \log \frac { 4 } { \delta }
$$

holds with probability at least $1 - \delta .$ . Furthermore, under Assumption 1 with $0 < \iota \leq \tau$ , we

have

$$
\begin{array} { r l } & { \| \eta _ { \mu } \| _ { \mathcal { H } _ { \mathcal { Y } } } = \left\| g _ { \mu } ( T ) T ^ { \iota + 1 } \varpi _ { \star } \right\| _ { \mathcal { H } _ { \mathcal { Y } } } \leq \| g _ { \mu } ( T ) T \| _ { \mathcal { H } _ { \mathcal { Y } } \to \mathcal { H } _ { \mathcal { Y } } } \cdot \| T ^ { \iota } \| _ { \mathcal { H } _ { \mathcal { Y } } \to \mathcal { H } _ { \mathcal { Y } } } \cdot \| \varpi _ { \star } \| _ { \mathcal { H } _ { \mathcal { Y } } } } \\ & { \qquad \leq E ( \kappa _ { \mathcal { X } } \kappa _ { \mathcal { Y } } ) ^ { 2 \iota } \| \varpi _ { \star } \| _ { \mathcal { H } _ { \mathcal { Y } } } . } \end{array}
$$

The last inequality follows from $\| T \| _ { \mathcal { H } _ { y } \to \mathcal { H } _ { y } } \le ( \kappa _ { \mathcal { X } } \kappa _ { \mathcal { Y } } ) ^ { 2 }$ and, by the filter function property in (10),

$$
\| g _ { \mu } ( T ) T \| _ { \mathcal { H } y \to \mathcal { H } y } \le \operatorname* { s u p } _ { 0 \le t \le ( \kappa _ { \mathcal { X } } \kappa _ { \mathcal { Y } } ) ^ { 2 } } g _ { \mu } ( t ) t \le E .
$$

Combining these estimates yields

$$
\begin{array} { r l } & { { A } _ { 2 , 1 } \leq 2 \kappa _ { \mathcal { X } } \kappa _ { \mathcal { Y } } \cdot \left\| { U } _ { \mathcal { Y } \mathcal { X } } - \widehat { U } _ { \mathcal { Y } \mathcal { X } } \right\| _ { \mathcal { H } _ { \mathcal { X } } \to \mathcal { H } _ { \mathcal { Y } } } \cdot \| \eta _ { \mu } \| _ { \mathcal { H } _ { \mathcal { Y } } } } \\ & { \qquad \leq 2 0 E ( \kappa _ { \mathcal { X } } \kappa _ { \mathcal { Y } } ) ^ { 2 \iota + 2 } \| \varpi _ { \star } \| _ { \mathcal { H } _ { \mathcal { Y } } } \cdot n _ { \eta } ^ { - 1 / 2 } \log \frac { 4 } { \delta } . } \end{array}\tag{21}
$$

2. Since the bound $\| u _ { 1 } ^ { \mathrm { t e } } \| _ { \mathcal { H } _ { \mathcal { X } } } \leq \kappa _ { \mathcal { X } }$ holds trivially, applying Proposition 1 gives

$$
\begin{array} { r l } & { { A _ { 2 , 2 } } = \| ( U _ { \mathcal { V X } } - \widehat { U } _ { \mathcal { V X } } ) { u _ { \mathbf { 1 } } ^ { \mathrm { t e } } } \| _ { \mathcal { H } _ { \mathcal { V } } } \leq \| { U _ { \mathcal { V X } } - \widehat { U } _ { \mathcal { V X } } } \| _ { \mathcal { H } _ { \mathcal { X } }  \mathcal { H } _ { \mathcal { Y } } } \cdot \| { u _ { \mathbf { 1 } } ^ { \mathrm { t e } } } \| _ { \mathcal { H } _ { \mathcal { X } } } } \\ & { \phantom { { A _ { 2 , 2 } } } \leq 1 0 \kappa _ { \mathcal { X } } ^ { 2 } \kappa _ { \mathcal { V } } \cdot n _ { \eta } ^ { - 1 / 2 } \log \frac { 4 } { \delta } . } \end{array}\tag{22}
$$

3. For $A _ { 2 , 3 }$ , using the bound $\lVert \widehat { U } _ { y x } \rVert _ { \mathcal { H } _ { x }  \mathcal { H } _ { y } } \leq \kappa _ { \mathcal { X } } \kappa _ { \mathcal { Y } }$ together with Proposition 1 yields

$$
\begin{array} { r l } & { { A _ { 2 , 3 } } = \| \widehat { U } _ { \mathcal { V X } } ( { { u _ { 1 } ^ { \mathrm { t e } } } - \widehat { { u } } _ { 1 } ^ { \mathrm { t e } } } ) \| _ { \mathcal { H } _ { \mathcal { V } } } \leq \| \widehat { U } _ { \mathcal { V X } } \| _ { \mathcal { H } _ { \mathcal { X } }  \mathcal { H } _ { \mathcal { V } } } \cdot \| { { u _ { 1 } ^ { \mathrm { t e } } } - \widehat { { u } } _ { 1 } ^ { \mathrm { t e } } } \| _ { \mathcal { H } _ { \mathcal { X } } } } \\ & { \qquad \leq 1 0 \kappa _ { \mathcal { X } } ^ { 2 } \kappa _ { \mathcal { V } } \cdot { n _ { \eta } ^ { - 1 / 2 } } \log \frac { 4 } { \delta } . } \end{array}\tag{23}
$$

Combining the estimates in (20), (21), (22), and (23) yields

$$
\begin{array} { r l } & { \boldsymbol { A } _ { 2 } \le \left\| g _ { \boldsymbol { \mu } } ( \widehat { T } ) \right\| _ { \mathcal { H } _ { \mathcal { V } } \to \mathcal { H } _ { \mathcal { V } } } \cdot ( \boldsymbol { A } _ { 2 , 1 } + \boldsymbol { A } _ { 2 , 2 } + \boldsymbol { A } _ { 2 , 3 } ) } \\ & { \qquad \le 4 0 E ^ { 2 } ( \kappa \chi \kappa y ) ^ { 2 \iota + 2 } \| \boldsymbol { \varpi } _ { \star } \| _ { \mathcal { H } _ { \mathcal { V } } } \cdot \boldsymbol { \mu } ^ { - 1 } n _ { \eta } ^ { - 1 / 2 } \log \frac { 4 } { \delta } , } \end{array}
$$

which holds with probability at least 1 − δ. This completes the proof.

Next, we bound $A _ { 3 }$ in (19) using Proposition 4.

Proposition 4. Suppose that Assumption 1 holds with $\iota \in ( 0 , \tau ]$ , and that the bounds in Proposition 1 are satisfied. If the sample size $n _ { \eta }$ and the regularization parameter µ satisfy

$$
\mu n _ { \eta } ^ { 1 / 2 } \geq 4 0 ( \kappa _ { \mathcal { X } } \kappa _ { \mathcal { Y } } ) ^ { 2 } \log \frac { 4 } { \delta } ,
$$

then the following bound holds:

$$
\begin{array} { r } { A _ { 3 } = \Big \| g _ { \mu } ( \widehat { T } ) \left( T \eta _ { \mu } - U _ { \mathcal { V X } } u _ { 1 } ^ { \mathrm { t e } } \right) \Big \| _ { \mathcal { H } _ { \mathcal { Y } } } \leq 4 E F \| \varpi _ { \star } \| _ { \mathcal { H } _ { \mathcal { Y } } } \cdot \mu ^ { \iota } . } \end{array}
$$

Proof. Inserting the identities $( \widehat { T } + \mu I _ { \mathcal { Y } } ) ( \widehat { T } + \mu I _ { \mathcal { Y } } ) ^ { - 1 }$ and $( T + \mu I _ { \mathcal { Y } } ) ( T + \mu I _ { \mathcal { Y } } ) ^ { - 1 }$ , we decompose $A _ { 3 }$ as

$$
{ A _ { 3 } } \leq { A _ { 3 , 1 } } \cdot { A _ { 3 , 2 } } \cdot { A _ { 3 , 3 } } ,
$$

where

$$
\begin{array} { r } { A _ { 3 , 1 } = \Big \| g _ { \mu } ( \widehat { T } ) \left( \widehat { T } + \mu I _ { \mathcal { V } } \right) \Big \| _ { \mathcal { H } _ { \mathcal { V } } \to \mathcal { H } _ { \mathcal { V } } } , \quad A _ { 3 , 2 } = \Big \| ( \widehat { T } + \mu I _ { \mathcal { V } } ) ^ { - 1 } \left( T + \mu I _ { \mathcal { V } } \right) \Big \| _ { \mathcal { H } _ { \mathcal { V } } \to \mathcal { H } _ { \mathcal { V } } } , } \end{array}
$$

$$
A _ { 3 , 3 } = \left. ( T + \mu I _ { \mathcal { V } } ) ^ { - 1 } \left( T \eta _ { \mu } - U _ { \mathcal { V } \mathcal { X } } \boldsymbol { u } _ { \mathbf { 1 } } ^ { \mathrm { t e } } \right) \right. _ { \mathcal { H } _ { \mathcal { V } } } .
$$

We now bound $A _ { 3 , 1 } , A _ { 3 , 2 }$ , and $A _ { 3 , 3 }$ as follows:

1. By the filter function property in (10), we obtain

$$
A _ { 3 , 1 } = \Big \| g _ { \mu } ( \widehat { T } ) ( \widehat { T } + \mu I _ { \mathcal { V } } ) \Big \| _ { \mathcal { H } _ { \mathcal { V } } \to \mathcal { H } _ { \mathcal { V } } } \leq \operatorname* { s u p } _ { 0 \leq t \leq ( \kappa _ { \mathcal { X } } \kappa _ { \mathcal { V } } ) ^ { 2 } } g _ { \mu } ( t ) ( t + \mu )\tag{24}
$$

$$
\leq E + E \cdot \mu ^ { - 1 } \cdot \mu = 2 E .
$$

2. Since

$$
\widehat T + \mu I _ { \mathcal V } = ( T + \mu I _ { \mathcal V } ) \left( I _ { \mathcal V } - ( T + \mu I _ { \mathcal V } ) ^ { - 1 } \left( T - \widehat T \right) \right) ,
$$

we have

$$
\begin{array} { r } { A _ { 3 , 2 } = \Big \| ( \widehat { T } + \mu I _ { \mathcal { V } } ) ^ { - 1 } \left( T + \mu I _ { \mathcal { V } } \right) \Big \| _ { \mathcal { H } _ { \mathcal { V } } \to \mathcal { H } _ { \mathcal { V } } } = \Big \| \Big ( I _ { \mathcal { V } } - ( T + \mu I _ { \mathcal { V } } ) ^ { - 1 } \left( T - \widehat { T } \right) \Big ) ^ { - 1 } \Big \| _ { \mathcal { H } _ { \mathcal { V } } \to \mathcal { H } _ { \mathcal { V } } } . } \end{array}
$$

To bound the norm of the inverse, we first estimate $\| ( T + \mu I _ { \mathcal { V } } ) ^ { - 1 } \left( T - \widehat { T } \right) \| _ { \mathcal { H } _ { \mathcal { V } } \to \mathcal { H } _ { \mathcal { V } } }$ . As shown in the proof of Proposition 3 (when bounding $A _ { 2 , 1 } )$ , we have

$$
\begin{array} { r } { \left\| \widehat { T } - T \right\| _ { \mathcal { H } _ { y } \to \mathcal { H } _ { y } } \le 2 \kappa _ { \mathcal { X } } \kappa _ { \mathcal { Y } } \cdot \left\| U _ { \mathcal { Y } \mathcal { X } } - \widehat { U } _ { \mathcal { Y } \mathcal { X } } \right\| _ { \mathcal { H } _ { \mathcal { X } } \to \mathcal { H } _ { \mathcal { Y } } } \le 2 0 ( \kappa _ { \mathcal { X } } \kappa _ { \mathcal { Y } } ) ^ { 2 } \cdot n _ { \eta } ^ { - 1 / 2 } \log \frac { 4 } { \delta } , } \end{array}
$$

under the assumption that the bounds in Proposition 1 hold. Thus,

$$
\begin{array} { r l } & { \| ( T + \mu I _ { \mathcal { Y } } ) ^ { - 1 } ( T - \widehat { T } ) \| _ { \mathcal { H } _ { \mathcal { Y } }  \mathcal { H } _ { \mathcal { Y } } } \leq \| ( T + \mu I _ { \mathcal { Y } } ) ^ { - 1 } \| _ { \mathcal { H } _ { \mathcal { Y } }  \mathcal { H } _ { \mathcal { Y } } } \cdot \| \widehat { T } - T \| _ { \mathcal { H } _ { \mathcal { Y } }  \mathcal { H } _ { \mathcal { Y } } } } \\ & { \qquad \leq 2 0 ( \kappa _ { \mathcal { X } } \kappa _ { \mathcal { Y } } ) ^ { 2 } \cdot \mu ^ { - 1 } n _ { \eta } ^ { - 1 / 2 } \log { \frac { 4 } { \delta } } , } \end{array}
$$

where we use $\| ( T + \mu I _ { \mathcal { V } } ) ^ { - 1 } \| _ { \mathcal { H } _ { \mathcal { V } } \to \mathcal { H } _ { \mathcal { V } } } \le \mu ^ { - 1 }$ in the last inequality.

If the sample size $n _ { \eta }$ and the regularization parameter µ satisfy

$$
\mu n _ { \eta } ^ { 1 / 2 } \geq 4 0 ( \kappa _ { \mathcal { X } } \kappa _ { \mathcal { Y } } ) ^ { 2 } \log \frac { 4 } { \delta } ,
$$

then

$$
\begin{array} { r } { \left\| ( T + \mu I _ { \mathcal { V } } ) ^ { - 1 } \left( T - \widehat { T } \right) \right\| _ { \mathcal { H } _ { \mathcal { V } } \to \mathcal { H } _ { \mathcal { V } } } \le 1 / 2 . } \end{array}
$$

By the Neumann series expansion, we obtain

$$
A _ { 3 , 2 } = \bigg \| \Big ( I _ { \mathcal { V } } - ( T + \mu I _ { \mathcal { V } } ) ^ { - 1 } ( T - \widehat { T } ) \Big ) ^ { - 1 } \bigg \| _ { \mathcal { H } _ { \mathcal { V } } \to \mathcal { H } _ { \mathcal { V } } } \leq \frac { 1 } { 1 - 1 / 2 } = 2 .\tag{25}
$$

3. For $A _ { 3 , 3 } ,$ we use the relation

$$
T \eta _ { \mu } - U _ { \mathcal { V X } } \mathbf { \Psi } u _ { 1 } ^ { \mathrm { t e } } = T \eta _ { \mu } - U _ { \mathcal { V X } } U _ { \mathcal { X Y } } \eta _ { \star } = T \eta _ { \mu } - T \eta _ { \star }
$$

to obtain

$$
\begin{array} { r l } & { { \cal A } _ { 3 , 3 } = \left\| { ( T + \mu I _ { \mathcal { V } } ) ^ { - 1 } \left( T \eta _ { \mu } - U _ { \mathcal { V X } } { \pmb u } _ { 1 } ^ { \mathrm { t e } } \right) } \right\| _ { \mathcal { H } _ { \mathcal { V } } } = \left\| { ( T + \mu I _ { \mathcal { V } } ) ^ { - 1 } T \left( \eta _ { \mu } - \eta _ { \star } \right) } \right\| _ { \mathcal { H } _ { \mathcal { V } } } } \\ & { \qquad \leq \left\| { ( T + \mu I _ { \mathcal { V } } ) ^ { - 1 } T } \right\| _ { \mathcal { H } _ { \mathcal { V } } \to \mathcal { H } _ { \mathcal { V } } } \cdot \| \eta _ { \mu } - \eta _ { \star } \| _ { \mathcal { H } _ { \mathcal { V } } } \leq { F } \| \varpi _ { \star } \| _ { \mathcal { H } _ { \mathcal { V } } } \cdot \mu ^ { \iota } . } \end{array}\tag{26}
$$

In the last inequality, we use $\| ( T + \mu I _ { \mathcal { V } } ) ^ { - 1 } T \| _ { \mathcal { H } _ { \mathcal { V } } \to \mathcal { H } _ { \mathcal { V } } } \le 1$ and the bound for $\| \eta _ { \mu } - \eta _ { \star } \| _ { \mathcal { H } _ { \mathcal { Y } } }$ from Proposition 2.

Combining the estimates in (24), (25), and (26), we conclude that when

$$
\mu n _ { \eta } ^ { 1 / 2 } \geq 4 0 ( \kappa _ { \mathcal { X } } \kappa _ { \mathcal { Y } } ) ^ { 2 } \log \frac { 4 } { \delta } ,
$$

the following inequality holds:

$$
\begin{array} { r } { A _ { 3 } \leq A _ { 3 , 1 } \cdot A _ { 3 , 2 } \cdot A _ { 3 , 3 } \leq 4 E F \| \varpi _ { \star } \| _ { \mathcal H _ { y } } \cdot \mu ^ { \iota } . } \end{array}
$$

This completes the proof.

Finally, Proposition 5 provides a bound for $A _ { 4 }$ in (19).

Proposition 5. Assume that the bounds in Proposition 1 hold. Then we have

$$
\begin{array} { r l } & { A _ { 4 } = \left\| ( I _ { \mathcal { V } } - g _ { \mu } ( \widehat { T } ) \widehat { T } ) \eta _ { \mu } \right\| _ { \mathcal { H } _ { \mathcal { V } } } } \\ & { \quad \leq E F \| \varpi _ { \star } \| _ { \mathcal { H } _ { \mathcal { V } } } \left( \mu ^ { \iota } + 2 0 ( \iota + 1 ) ( \kappa _ { \mathcal { X } } \kappa _ { \mathcal { Y } } ) ^ { 2 \iota } \cdot n _ { \eta } ^ { - \frac { \operatorname* { m i n } \{ \iota , 1 \} } { 2 } } \right) \log \frac { 4 } { \delta } . } \end{array}
$$

Proof. Since $\eta _ { \mu } = g _ { \mu } ( T ) T \eta _ { \star } = g _ { \mu } ( T ) T ^ { \iota + 1 } \varpi _ { \star }$ , we have

$$
\begin{array} { r l } & { \boldsymbol { A } _ { 4 } = \Big \| ( I _ { \mathcal { V } } - g _ { \mu } ( \widehat { T } ) \widehat { T } ) \eta _ { \mu } \Big \| _ { \mathcal { H } _ { \mathcal { V } } } = \Big \| ( I _ { \mathcal { V } } - g _ { \mu } ( \widehat { T } ) \widehat { T } ) g _ { \mu } ( T ) T ^ { \iota + 1 } \varpi _ { \star } \Big \| _ { \mathcal { H } _ { \mathcal { V } } } } \\ & { \quad \leq \Big \| ( I _ { \mathcal { V } } - g _ { \mu } ( \widehat { T } ) \widehat { T } ) T ^ { \iota } \Big \| _ { \mathcal { H } _ { \mathcal { V } } \to \mathcal { H } _ { \mathcal { V } } } \cdot \| g _ { \mu } ( T ) T \| _ { \mathcal { H } _ { \mathcal { V } } \to \mathcal { H } _ { \mathcal { V } } } \cdot \| \varpi _ { \star } \| _ { \mathcal { H } _ { \mathcal { V } } } } \\ & { \quad \leq \Big \| ( I _ { \mathcal { V } } - g _ { \mu } ( \widehat { T } ) \widehat { T } ) T ^ { \iota } \Big \| _ { \mathcal { H } _ { \mathcal { V } } \to \mathcal { H } _ { \mathcal { V } } } \cdot E \cdot \| \varpi _ { \star } \| _ { \mathcal { H } _ { \mathcal { V } } } , } \end{array}
$$

where the filter function property (10) gives

$$
\| g _ { \mu } ( T ) T \| _ { \mathcal { H } _ { \mathcal { y } } \to \mathcal { H } _ { \mathcal { y } } } \le \operatorname* { s u p } _ { 0 \le t \le ( \kappa _ { \mathcal { X } } \kappa _ { \mathcal { y } } ) ^ { 2 } } g _ { \mu } ( t ) t \le E .
$$

To bound the remaining operator norm, we add and subtract $\widehat { T } ^ { \iota }$ :

$$
\begin{array} { r l } & { \quad \| ( I _ { \mathcal { V } } - g _ { \mu } ( \widehat { T } ) \widehat { T } ) T ^ { \iota } \| _ { \mathcal { H } _ { \mathcal { V } }  \mathcal { H } _ { \mathcal { V } } } } \\ & { \leq \| ( I _ { \mathcal { V } } - g _ { \mu } ( \widehat { T } ) \widehat { T } ) \widehat { T } ^ { \iota } \| _ { \mathcal { H } _ { \mathcal { V } }  \mathcal { H } _ { \mathcal { V } } } + \| ( I _ { \mathcal { V } } - g _ { \mu } ( \widehat { T } ) \widehat { T } ) \| _ { \mathcal { H } _ { \mathcal { V } }  \mathcal { H } _ { \mathcal { V } } } \cdot \| T ^ { \iota } - \widehat { T } ^ { \iota } \| _ { \mathcal { H } _ { \mathcal { V } }  \mathcal { H } _ { \mathcal { V } } } } \\ & { \leq F ( \mu ^ { \iota } + \| T ^ { \iota } - \widehat { T } ^ { \iota } \| _ { \mathcal { H } _ { \mathcal { V } }  \mathcal { H } _ { \mathcal { V } } } ) . } \end{array}
$$

Here we used the filter function property (11) to obtain

$$
\begin{array} { r } { \left\| ( I _ { \mathcal { V } } - g _ { \mu } ( \widehat { T } ) \widehat { T } ) \widehat { T } ^ { * } \right\| _ { \mathcal { H } _ { \mathcal { V } } \to \mathcal { H } _ { \mathcal { V } } } \le \underset { 0 \le t \le ( \kappa _ { \mathcal { X } } \kappa _ { \mathcal { V } } ) ^ { 2 } } { \operatorname* { s u p } } | 1 - g _ { \mu } ( t ) t | t ^ { \iota } \le F \cdot \mu ^ { \iota } , } \end{array}
$$

$$
\begin{array} { r } { \Big \| ( I _ { \mathcal { V } } - g _ { \mu } ( \widehat { T } ) \widehat { T } ) \Big \| _ { \mathcal { H } _ { \mathcal { V } } \to \mathcal { H } _ { \mathcal { V } } } \leq \operatorname* { s u p } _ { 0 \leq t \leq ( \kappa _ { \mathcal { X } } \kappa _ { \mathcal { V } } ) ^ { 2 } } | 1 - g _ { \mu } ( t ) t | \leq F . } \end{array}
$$

For the term $\| T ^ { \iota } - \widehat { T } ^ { \iota } \| _ { \mathcal { H } _ { y } \to \mathcal { H } _ { y } }$ , we invoke Lemma 2, which yields

$$
\begin{array} { r } { \left\| T ^ { \iota } - \widehat { T } ^ { \iota } \right\| _ { \mathcal { H } _ { \mathcal { V } } \to \mathcal { H } _ { \mathcal { V } } } \leq \left\{ \begin{array} { l l } { \left\| T - \widehat { T } \right\| _ { \mathcal { H } _ { \mathcal { V } } \to \mathcal { H } _ { \mathcal { V } } } ^ { \iota } , } & { \iota \in ( 0 , 1 ] ; } \\ { \iota ( \kappa _ { \mathcal { X } } \kappa _ { \mathcal { V } } ) ^ { 2 \iota - 2 } \left\| T - \widehat { T } \right\| _ { \mathcal { H } _ { \mathcal { V } } \to \mathcal { H } _ { \mathcal { V } } } , } & { \iota > 1 . } \end{array} \right. } \end{array}
$$

As established in the proof of Proposition 3 (when bounding ${ \cal A } _ { 2 , 1 } )$ , under the assumptions of Proposition 1 we have

$$
\left\| \widehat { T } - T \right\| _ { \mathcal { H } _ { \mathcal { V } } \to \mathcal { H } _ { \mathcal { V } } } \le 2 0 ( \kappa _ { \mathcal { X } } \kappa _ { \mathcal { Y } } ) ^ { 2 } \cdot n _ { \eta } ^ { - 1 / 2 } \log \frac { 4 } { \delta } .
$$

Combining this with the above case distinction, gives

$$
\| T ^ { \iota } - \widehat { T } ^ { \iota } \| _ { \mathcal { H } _ { \mathcal { Y } }  \mathcal { H } _ { \mathcal { Y } } } \leq 2 0 ( \iota + 1 ) ( \kappa _ { \mathcal { X } } \kappa _ { \mathcal { Y } } ) ^ { 2 \iota } \cdot n _ { \eta } ^ { - \frac { \operatorname* { m i n } \{ \iota , 1 \} } { 2 } } \log \frac { 4 } { \delta } .
$$

Putting everything together, we obtain

$$
\begin{array} { r l } & { { A _ { 4 } } \leq \| ( I _ { \mathcal { V } } - g _ { \mu } ( \widehat { T } ) \widehat { T } ) T ^ { \iota } \| _ { \mathcal { H } _ { \mathcal { V } }  \mathcal { H } _ { \mathcal { V } } } \cdot E \cdot \| \varpi _ { \star } \| _ { \mathcal { H } _ { \mathcal { V } } } } \\ & { \quad \leq E F \| \varpi _ { \star } \| _ { \mathcal { H } _ { \mathcal { V } } } ( \mu ^ { \iota } + \| T ^ { \iota } - \widehat { T } ^ { \iota } \| _ { \mathcal { H } _ { \mathcal { V } }  \mathcal { H } _ { \mathcal { V } } } ) } \\ & { \quad \leq E F \| \varpi _ { \star } \| _ { \mathcal { H } _ { \mathcal { V } } } ( \mu ^ { \iota } + 2 0 ( \iota + 1 ) ( \kappa _ { \mathcal { X } } \kappa _ { \mathcal { V } } ) ^ { 2 \iota } \cdot n _ { \eta } ^ { - \frac { \operatorname* { m i n } \{ \iota , 1 \} } { 2 } } ) \log \frac { 4 } { \delta } , } \end{array}
$$

which completes the proof.

With the bounds for $A _ { 1 } , A _ { 2 } , A _ { 3 }$ , and $A _ { 4 }$ established, we are now ready to prove Theorem 1.

Proof of Theorem 1. We set the regularization parameter as $\mu = n _ { \eta } ^ { - 1 / ( 2 \iota + 2 ) }$ . Consequently,

$$
\operatorname* { m a x } \biggl \{ \mu ^ { \iota } , \mu ^ { - 1 } n _ { \eta } ^ { - 1 / 2 } , n _ { \eta } ^ { - \frac { \operatorname* { m i n } \{ \iota , 1 \} } { 2 } } \biggr \} \leq n _ { \eta } ^ { - \frac { \iota } { 2 \iota + 2 } } .
$$

Furthermore, for any $\delta \in ( 0 , 1 )$ , if the sample size $n _ { \eta }$ is suficiently large such that

$$
n _ { \eta } \ge \left( 4 0 ( \kappa _ { \mathcal { X } } \kappa _ { \mathcal { Y } } ) ^ { 2 } \log \frac { 4 } { \delta } \right) ^ { \frac { 2 \iota + 2 } { \iota } } ,
$$

then, on the event described in Proposition 1 (which holds with probability at least $1 - \delta )$ , the bounds in Propositions 2–5 hold simultaneously. This yields the following inequalities:

$$
\begin{array} { r l } & { A _ { 1 } \le F \| \varpi _ { * } \| _ { \mathcal { H } _ { \mathcal { V } } } \cdot n _ { \eta } ^ { - \frac { \iota } { 2 \iota + 2 } } , \quad A _ { 2 } \le 4 0 E ^ { 2 } ( \kappa \chi \kappa y ) ^ { 2 \iota + 2 } \| \varpi _ { * } \| _ { \mathcal { H } _ { \mathcal { V } } } \cdot n _ { \eta } ^ { - \frac { \iota } { 2 \iota + 2 } } \log \frac { 4 } { \delta } , } \\ & { A _ { 3 } \le 4 E F \| \varpi _ { * } \| _ { \mathcal { H } _ { \mathcal { V } } } \cdot n _ { \eta } ^ { - \frac { \iota } { 2 \iota + 2 } } , \quad A _ { 4 } \le E F \| \varpi _ { * } \| _ { \mathcal { H } _ { \mathcal { V } } } \left( 1 + 2 0 ( \iota + 1 ) ( \kappa _ { \mathcal { X } } \kappa _ { \mathcal { V } } ) ^ { 2 \iota } \right) n _ { \eta } ^ { - \frac { \iota } { 2 \iota + 2 } } \log \frac { 4 } { \delta } . } \end{array}
$$

Therefore, the error $\| \eta _ { \star } - \widehat { \eta } _ { \mu } \| _ { \mathcal { H } _ { \mathcal { y } } }$ can be bounded as

$$
\| \eta _ { \star } - \widehat { \eta } _ { \mu } \| _ { \mathcal { H } _ { \mathcal { V } } } \leq A _ { 1 } + A _ { 2 } + A _ { 3 } + A _ { 4 } \leq \Delta _ { \eta } \cdot n _ { \eta } ^ { - \frac { \iota } { 2 \iota + 2 } } \log \frac { 4 } { \delta } ,
$$

where

$$
\begin{array} { r l } & { \varDelta _ { \eta } = F \| \varpi _ { \star } \| _ { \mathcal { H } _ { \mathcal { Y } } } + 4 0 E ^ { 2 } ( \kappa _ { \mathcal { X } } \kappa _ { \mathcal { Y } } ) ^ { 2 \iota + 2 } \| \varpi _ { \star } \| _ { \mathcal { H } _ { \mathcal { Y } } } } \\ & { \qquad + 4 E F \| \varpi _ { \star } \| _ { \mathcal { H } _ { \mathcal { Y } } } + E F \| \varpi _ { \star } \| _ { \mathcal { H } _ { \mathcal { Y } } } \left( 1 + 2 0 ( \iota + 1 ) ( \kappa _ { \mathcal { X } } \kappa _ { \mathcal { Y } } ) ^ { 2 \iota } \right) } \end{array}\tag{27}
$$

is a constant independent of $n _ { \eta }$ or $\delta .$ . Hence, the theorem holds.

## 4.2 Proof of Theorem 2

Having proved Theorem 1, we know that for any $\iota \in ( 0 , \tau - 1 / 2 ] , \widehat { \eta } _ { \mu }$ converges to $\eta _ { \star }$ with high probability. Therefore, our density ratio estimator $\widetilde { \eta } _ { \mu } = \operatorname* { m a x } \{ \widehat { \eta } _ { \mu } , 0 \}$ satisfies the uniform bound:

$$
\begin{array} { r l } { \displaystyle \operatorname* { s u p } _ { y \in \mathcal { V } } | \eta _ { \star } ( y ) - \widetilde { \eta } _ { \mu } ( y ) | \leq \displaystyle \operatorname* { s u p } _ { y \in \mathcal { V } } | \eta _ { \star } ( y ) - \widehat { \eta } _ { \mu } ( y ) | \leq \kappa y \cdot \| \eta _ { \star } - \widehat { \eta } _ { \mu } \| _ { \mathcal { H } _ { y } } } & { } \\ { \leq \kappa y \Delta _ { \eta } \cdot n _ { \eta } ^ { - \frac { \iota } { 2 \iota + 2 } } \log \frac { 4 } { \delta ^ { \prime } } , } & { } \end{array}\tag{28}
$$

with probability at least $1 - \delta ^ { \prime }$ . The first inequality holds beacuse $\eta _ { \star }$ is nonnegative.

Now, we use $\widetilde { \eta } _ { \mu }$ as a weighting function to construct an estimator $\widehat { f } _ { \lambda }$ of the regression function $f _ { \star } ^ { \mathrm { t e } }$ via (14), and analyze its convergence behavior. In this subsection, we establish bounds for $\| f _ { \star } ^ { \mathrm { t e } } - \widehat { f } _ { \lambda } \| _ { \mathcal { L } ^ { 2 } ( \mathcal { X } , p _ { \mathcal { X } } ^ { \mathrm { t e } } ) }$ and $\| f _ { \star } ^ { \mathrm { t e } } - \widehat { f } _ { \lambda } \| _ { \mathcal { H } _ { \mathcal { X } } }$ . To unify the analysis for both norms, recall that for any $f \in \mathcal { H } _ { \mathcal { X } }$

$$
\left\| f \right\| _ { \mathcal { L } ^ { 2 } ( \mathcal { X } , p _ { \mathcal { X } } ^ { \mathrm { t e } } ) } = \left\| V _ { \mathcal { X } \mathcal { X } } ^ { 1 / 2 } f \right\| _ { \mathcal { H } _ { \mathcal { X } } } ,
$$

where $V _ { \mathcal { X X } } = \mathbb { E } _ { { \pmb x } \sim p _ { \mathcal { X } } ^ { \mathrm { t e } } } [ K _ { \mathcal { X } } ( \cdot , { \pmb x } ) \otimes K _ { \mathcal { X } } ( \cdot , { \pmb x } ) ]$ is the covariance operator on the test domain. Consequently, it sufices to bound

$$
\left\| V _ { \mathcal { X } \mathcal { X } } ^ { \frac { 1 - \gamma } { 2 } } \left( f _ { \star } ^ { \mathrm { t e } } - \widehat { f } _ { \lambda } \right) \right\| _ { \mathcal { H } _ { \mathcal { X } } }
$$

for $\gamma \in [ 0 , 1 ]$ , and then specialize to $\gamma = 0$ or 1 to obtain the respective norms. In addition, we assume without loss of generality that

$$
\operatorname* { s u p } _ { y \in \mathcal { V } } | \eta _ { \star } ( y ) | \leq M _ { \eta } , \quad \operatorname* { s u p } _ { y \in \mathcal { V } } | \widetilde { \eta } _ { \mu } ( y ) | \leq M _ { \eta } , \quad \operatorname* { s u p } _ { y \in \mathcal { V } } | y | \leq M _ { y } .
$$

We introduce the auxiliary function

$$
f _ { \lambda } = g _ { \lambda } ( V _ { \mathcal { X X } } ) V _ { \mathcal { X X } } f _ { \star } ^ { \mathrm { t e } } = g _ { \lambda } ( V _ { \mathcal { X X } } ) \pmb { v } _ { y } ^ { \mathrm { t e } } ,
$$

where the relation V<sub>XX</sub> $f _ { \star } ^ { \mathrm { t e } } = v _ { y } ^ { \mathrm { t e } }$ is established in (13). By the triangle inequality,

$$
\left\| V _ { \chi \chi } ^ { \frac { 1 - \gamma } { 2 } } \left( f _ { \star } ^ { \mathrm { t e } } - \widehat f _ { \lambda } \right) \right\| _ { \mathcal { H } _ { \chi } } \leq \left\| V _ { \chi \chi } ^ { \frac { 1 - \gamma } { 2 } } \left( f _ { \star } ^ { \mathrm { t e } } - f _ { \lambda } \right) \right\| _ { \mathcal { H } _ { \chi } } + \left\| V _ { \chi \chi } ^ { \frac { 1 - \gamma } { 2 } } \left( f _ { \lambda } - \widehat f _ { \lambda } \right) \right\| _ { \mathcal { H } _ { \chi } } .
$$

For the second term, we have

$$
\begin{array} { r l } & { \quad \| V _ { \mathcal { X } \mathcal { X } } ^ { \frac { 1 - \gamma } { 2 } } ( f _ { \lambda } - \widehat { f } _ { \lambda } ) \| _ { \mathcal { H } _ { \mathcal { X } } } } \\ & { \leq \| V _ { \mathcal { X } \mathcal { X } } ^ { \frac { 1 - \gamma } { 2 } } ( \widehat { W } _ { \mathcal { X } \mathcal { X } } + \lambda I _ { \mathcal { X } } ) ^ { - 1 / 2 } \| _ { \mathcal { H } _ { \mathcal { X } }  \mathcal { H } _ { \mathcal { X } } } \cdot \| ( \widehat { W } _ { \mathcal { X } \mathcal { X } } + \lambda I _ { \mathcal { X } } ) ^ { 1 / 2 } ( f _ { \lambda } - \widehat { f } _ { \lambda } ) \| _ { \mathcal { H } _ { \mathcal { X } } } . } \end{array}
$$

Recalling that $\widehat { f } _ { \lambda } = g _ { \lambda } ( \widehat { W } _ { \mathcal { X } \mathcal { X } } ) \widehat { \pmb { w } } _ { y }$ , we decompose $f _ { \lambda } - \widehat { f } _ { \lambda }$ as

$$
{ f _ { \lambda } - \widehat { f } _ { \lambda } = g _ { \lambda } ( \widehat { W } _ { \mathcal { X } \mathcal { X } } ) ( \widehat { W } _ { \mathcal { X } \mathcal { X } } f _ { \lambda } - \widehat { w } _ { y } ) + ( I _ { \mathcal { X } } - g _ { \lambda } ( \widehat { W } _ { \mathcal { X } \mathcal { X } } ) \widehat { W } _ { \mathcal { X } \mathcal { X } } ) f _ { \lambda } } ,
$$

which implies

$$
\begin{array} { r l } { \left\| { ( \widehat { W } _ { \mathcal { X } \mathcal { X } } + \lambda I _ { \mathcal { X } } ) ^ { 1 / 2 } ( f _ { \lambda } - \widehat { f } _ { \lambda } ) } \right\| _ { \mathcal { H } _ { \mathcal { X } } } \leq \left\| { ( \widehat { W } _ { \mathcal { X } \mathcal { X } } + \lambda I _ { \mathcal { X } } ) ^ { 1 / 2 } g _ { \lambda } ( \widehat { W } _ { \mathcal { X } \mathcal { X } } ) ( \widehat { W } _ { \mathcal { X } \mathcal { X } } f _ { \lambda } - \widehat { w } _ { y } ) } \right\| _ { \mathcal { H } _ { \mathcal { X } } } } & { } \\ { + \left\| { ( \widehat { W } _ { \mathcal { X } \mathcal { X } } + \lambda I _ { \mathcal { X } } ) ^ { 1 / 2 } ( I _ { \mathcal { X } } - g _ { \lambda } ( \widehat { W } _ { \mathcal { X } \mathcal { X } } ) \widehat { W } _ { \mathcal { X } \mathcal { X } } ) f _ { \lambda } } \right\| _ { \mathcal { H } _ { \mathcal { X } } } . } & { } \end{array}
$$

Combining these results yields the error decomposition

$$
\begin{array} { r } { \left\| V _ { \mathcal { X } \mathcal { X } } ^ { \frac { 1 - \gamma } { 2 } } \left( f _ { \star } ^ { \mathrm { t e } } - \widehat { f } _ { \lambda } \right) \right\| _ { \mathcal { H } _ { \mathcal { X } } } \le B _ { 1 } + B _ { 2 } ( B _ { 3 } + B _ { 4 } ) , } \end{array}
$$

where

$$
B _ { 1 } = \| V _ { \mathcal { X } } ^ { \frac { 1 - \gamma } { 2 } } ( f _ { \star } ^ { \mathrm { t e } } - f _ { \lambda } ) \| _ { \mathcal { H } _ { \mathcal { X } } } , \quad B _ { 2 } = \| V _ { \mathcal { X } \mathcal { X } } ^ { \frac { 1 - \gamma } { 2 } } ( \widehat { W } _ { \mathcal { X } \mathcal { X } } + \lambda I _ { \mathcal { X } } ) ^ { - 1 / 2 } \| _ { \mathcal { H } _ { \mathcal { X } }  \mathcal { H } _ { \mathcal { X } } } ,
$$

$$
B _ { 3 } = \Big \| ( \widehat { W } \chi \chi + \lambda I \chi ) ^ { 1 / 2 } g _ { \lambda } ( \widehat { W } \chi \chi ) ( \widehat { W } \chi \chi \ : f _ { \lambda } - \widehat { w } _ { y } ) \Big \| _ { \mathcal { H } _ { \chi } } ,\tag{29}
$$

$$
B _ { 4 } = \Bigl \| ( \widehat { W } \chi \chi + \lambda I \chi ) ^ { 1 / 2 } \left( I _ { \chi } - g _ { \lambda } ( \widehat { W } \chi \chi ) \widehat { W } \chi \chi \right) f _ { \lambda } \Bigr \| _ { \mathcal { H } _ { \chi } } .
$$

We now bound $B _ { 1 } , B _ { 2 } , B _ { 3 }$ , and $B _ { 4 }$ individually via separate propositions, and later combine the estimates. As a first step, we give a proposition that controls empirical averages and will be used repeatedly.

Proposition 6. Assume that the error bound (28) for the density ratio estimator $\widetilde { \eta } _ { \mu }$ holds. Then for any $\delta \in ( 0 , 1 )$ , the following bounds hold simultaneously with probability at least $1 - \delta .$

$$
\Big \| { \cal V } \chi \chi - \widehat W \chi \chi \Big \| _ { \mathscr { H } _ { \chi } \to \mathscr { H } _ { \chi } } \leq \kappa _ { \chi } ^ { 2 } \kappa y \varDelta _ { \eta } \cdot n _ { \eta } ^ { - \frac { \iota } { 2 \iota + 2 } } \log \frac { 4 } { \delta ^ { \prime } } + 1 0 M _ { \eta } \kappa _ { \chi } ^ { 2 } \cdot n _ { f } ^ { - 1 / 2 } \log \frac { 4 } { \delta } ,
$$

$$
\begin{array} { r } { \left\| { ( W _ { \mathcal { X } \mathcal { X } } f _ { \lambda } - w _ { y } ) - ( \widehat { W } _ { \mathcal { X } \mathcal { X } } f _ { \lambda } - \widehat { w } _ { y } ) } \right\| _ { \mathcal { H } _ { \mathcal { X } } } \leq 1 0 M _ { \eta } ( F \kappa _ { \mathcal { X } } \| \psi _ { * } \| _ { \mathcal { H } _ { \mathcal { X } } } + 2 M _ { y } ) \kappa _ { \mathcal { X } } \cdot n _ { f } ^ { - 1 / 2 } \log \frac { 4 } { \delta } . } \end{array}
$$

Proof. We bound these two quantities separately.

1. To bound $\| V _ { \mathcal { X X } } - \widehat { W } _ { \mathcal { X X } } \| _ { \mathcal { H } _ { \mathcal { X } } \to \mathcal { H } _ { \mathcal { X } } } .$ we decompose it as

$$
\begin{array} { r } {  V \chi - \widehat { W } \chi \chi  _ { \mathcal { H } _ { \chi }  \mathcal { H } _ { \chi } } \leq  V \chi \chi - W \chi \chi  _ { \mathcal { H } _ { \chi }  \mathcal { H } _ { \chi } } +  W \chi \chi - \widehat { W } \chi \chi  _ { \mathcal { H } _ { \chi }  \mathcal { H } _ { \chi } } . } \end{array}
$$

Since

$$
\begin{array} { r l } & { V _ { \mathcal { X X } } = \mathbb { E } _ { \pmb { x } \sim p _ { \mathcal { X } } ^ { \mathrm { t e } } } [ K _ { \mathcal { X } } ( \cdot , \pmb { x } ) \otimes K _ { \mathcal { X } } ( \cdot , \pmb { x } ) ] = \mathbb { E } _ { ( \pmb { x } , \pmb { y } ) \sim p _ { \mathcal { X } \times \mathcal { Y } } ^ { \mathrm { t e } } } [ K _ { \mathcal { X } } ( \cdot , \pmb { x } ) \otimes K _ { \mathcal { X } } ( \cdot , \pmb { x } ) ] } \\ & { \qquad = \mathbb { E } _ { ( \pmb { x } , \pmb { y } ) \sim p _ { \mathcal { X } \times \mathcal { Y } } ^ { \mathrm { t r } } } [ \eta _ { \star } ( \pmb { y } ) K _ { \mathcal { X } } ( \cdot , \pmb { x } ) \otimes K _ { \mathcal { X } } ( \cdot , \pmb { x } ) ] , } \end{array}
$$

then the first term $\| V _ { \mathcal { X X } } - W _ { \mathcal { X X } } \| _ { \mathcal { H } _ { \mathcal { X } } \to \mathcal { H } _ { \mathcal { X } } }$ can be expressed as

$$
\begin{array} { r l } & { \lVert V _ { \mathcal { X } \mathcal { X } } - W _ { \mathcal { X } \mathcal { X } } \rVert _ { \mathcal { H } _ { \mathcal { X } \to \mathcal { H } _ { \mathcal { X } } } } = \Big \lVert \mathbb { E } _ { ( x , y ) \sim p _ { \mathcal X \times \mathcal { Y } } ^ { \mathrm { t r } } } [ ( \eta _ { \star } ( y ) - \widetilde { \eta } _ { \mu } ( y ) ) K _ { \mathcal { X } } ( \cdot , \pmb { x } ) \otimes K _ { \mathcal { X } } ( \cdot , \pmb { x } ) ] \Big \rVert _ { \mathcal { H } _ { \mathcal { X } \to \mathcal { H } _ { \mathcal { X } } } } } \\ & { \qquad \le \underset { y \in \mathcal { Y } } { \operatorname* { s u p } } \lvert \eta _ { \star } ( y ) - \widetilde { \eta _ { \mu } } ( y ) \rvert \cdot \mathbb { E } _ { ( x , y ) \sim p _ { \mathcal X \times \mathcal { Y } } ^ { \mathrm { t r } } } \Big [ \lVert K _ { \mathcal { X } } ( \cdot , \pmb { x } ) \otimes K _ { \mathcal { X } } ( \cdot , \pmb { x } ) \rVert _ { \mathcal { H } _ { \mathcal { X } \to \mathcal { H } _ { \mathcal { X } } } } \Big ] } \\ & { \qquad \le \kappa _ { \mathcal { X } } ^ { 2 } \kappa _ { \mathcal { Y } } \Delta _ { \eta } \cdot n _ { \eta } ^ { - \frac { \varepsilon } { 2 i + 2 } } \log \frac { 4 } { \delta ^ { \prime } } . } \end{array}
$$

In the last inequality, we use (28) and the fact that $\begin{array} { r } { \operatorname* { s u p } _ { { \pmb x } \in { \pmb X } } \| K _ { { \pmb X } } ( \cdot , { \pmb x } ) \otimes K _ { { \pmb X } } ( \cdot , { \pmb x } ) \| _ { \mathcal { H } _ { \pmb X }  \mathcal { H } _ { \pmb X } } \le \kappa _ { \pmb X } ^ { 2 } } \end{array}$ For the second term, $\| W \chi \chi - \widehat { W } \chi \chi \| _ { \mathcal { H } _ { \chi } \to \mathcal { H } _ { \chi } }$ , define the random variable

$$
\zeta _ { 1 } ( x , y ) = \widetilde { \eta } _ { \mu } ( y ) K \chi ( \cdot , \pmb { x } ) \otimes K \chi ( \cdot , \pmb { x } ) , \quad ( \pmb { x } , y ) \sim p _ { \chi \times y } ^ { \mathrm { t r } } ,
$$

and let $\zeta _ { 1 , j } = \zeta _ { 1 } ( x _ { j } , y _ { j } )$ for $1 \leq j \leq n _ { f }$ . Then

$$
W _ { \mathcal { X } \mathcal { X } } - \widehat { W } _ { \mathcal { X } \mathcal { X } } = \mathbb { E } _ { ( \pmb { x } , \pmb { y } ) \sim p _ { \mathcal { X } \times \mathcal { Y } } ^ { \mathrm { t r } } } [ \zeta _ { 1 } ( \pmb { x } , \pmb { y } ) ] - \frac { 1 } { n _ { f } } \sum _ { j = 1 } ^ { n _ { f } } \zeta _ { 1 , j } .
$$

Since we assume $\mathrm { s u p } _ { y \in \mathcal { V } } | \widetilde { \eta } _ { \mu } ( y ) | \le M _ { \eta }$ , the Hilbert-Schmidt norm of $\zeta _ { 1 } ( x , y )$ is uniformly bounded by $M _ { \eta } \kappa _ { \chi } ^ { 2 }$ . Applying Lemma 1, we obtain, with probability at least $1 - \delta / 2$

$$
\begin{array} { r l } & { \left\| W _ { \mathcal { X } \mathcal { X } } - \widehat { W } _ { \mathcal { X } \mathcal { X } } \right\| _ { \mathcal { H } _ { \mathcal { X } } \to \mathcal { H } _ { \mathcal { X } } } \le \left\| \mathbb { E } _ { ( \pmb { x } , \pmb { y } ) \sim p _ { \mathcal { X } \times \mathcal { Y } } ^ { \bot } } [ \zeta _ { 1 } ( \pmb { x } , \pmb { y } ) ] - \frac { 1 } { n _ { f } } \sum _ { j = 1 } ^ { n _ { f } } \zeta _ { 1 , j } \right\| _ { \mathrm { H S } ( \mathcal { H } _ { \mathcal { X } } , \mathcal { H } _ { \mathcal { X } } ) } } \\ & { \qquad \le 1 0 M _ { \eta } \kappa _ { \mathcal { X } } ^ { 2 } \cdot n _ { f } ^ { - 1 / 2 } \log \frac { 4 } { \delta } . } \end{array}
$$

Combining these two bounds yields

$$
\begin{array} { r l } & { \left\| { V _ { \mathcal { X } \mathcal { X } } - \widehat { W } _ { \mathcal { X } \mathcal { X } } } \right\| _ { \mathcal { H } _ { \mathcal { X } } \to \mathcal { H } _ { \mathcal { X } } } \le \| { V _ { \mathcal { X } \mathcal { X } } - W _ { \mathcal { X } \mathcal { X } } } \| _ { \mathcal { H } _ { \mathcal { X } } \to \mathcal { H } _ { \mathcal { X } } } + \left\| { W _ { \mathcal { X } \mathcal { X } } - \widehat { W } _ { \mathcal { X } \mathcal { X } } } \right\| _ { \mathcal { H } _ { \mathcal { X } } \to \mathcal { H } _ { \mathcal { X } } } } \\ & { \qquad \le \kappa _ { \mathcal { X } } ^ { 2 } \kappa _ { \mathcal { Y } } \varDelta _ { \eta } \cdot n _ { \eta } ^ { - \frac { \iota } { 2 \iota + 2 } } \log \frac { 4 } { \delta ^ { \prime } } + 1 0 M _ { \eta } \kappa _ { \mathcal { X } } ^ { 2 } \cdot n _ { f } ^ { - 1 / 2 } \log \frac { 4 } { \delta } . } \end{array}\tag{30}
$$

2. Next, we bound $\lVert ( W _ { \mathcal { X X } } f _ { \lambda } - w _ { y } ) - ( \widehat { W } _ { \mathcal { X X } } f _ { \lambda } - \widehat { w } _ { y } ) \rVert _ { \mathcal { H } _ { \mathcal { X } } }$ . Define

$$
\zeta _ { 2 } ( { \pmb x } , y ) = { \widetilde \eta } _ { \mu } ( y ) ( f _ { \lambda } ( { \pmb x } ) - y ) K _ { \mathcal { X } } ( \cdot , { \pmb x } ) , \quad ( { \pmb x } , y ) \sim p _ { \mathcal { X } \times \mathcal { Y } } ^ { \mathrm { t r } }
$$

and set $\zeta _ { 2 , j } = \zeta _ { 2 } ( x _ { j } , y _ { j } )$ for $1 \leq j \leq n _ { f }$ . Then

$$
( W _ { \mathcal { X X } } f _ { \lambda } - { \pmb w } _ { y } ) - ( \widehat { W } _ { \mathcal { X X } } f _ { \lambda } - \widehat { { \pmb w } } _ { y } ) = { \mathbb E } _ { ( { \pmb x } , y ) \sim p _ { \mathscr { X } \times \mathscr { Y } } ^ { \bot \bot } } [ \zeta _ { 2 } ( { \pmb x } , y ) ] - \frac { 1 } { n _ { f } } \sum _ { j = 1 } ^ { n _ { f } } \zeta _ { 2 , j } .
$$

To obtain a uniform bound for $\| \zeta _ { 2 } ( \pmb { x } , y ) \| _ { \mathcal { H } _ { \mathcal { X } } }$ , note that $\begin{array} { r } { \operatorname* { s u p } _ { y \in \mathcal { V } } | y | \le M _ { y } } \end{array}$ implies $\begin{array} { r } { \operatorname* { s u p } _ { x \in \mathcal { X } } | f _ { \star } ^ { \mathrm { t e } } ( { \pmb x } ) | \leq } \end{array}$ $M _ { y }$ . Thus for all $( { \pmb x } , { \pmb y } ) \in \mathcal { X } \times \mathcal { Y }$

$$
| f _ { \lambda } ( { \pmb x } ) - { \pmb y } | \leq | f _ { \lambda } ( { \pmb x } ) - f _ { \star } ^ { \mathrm { t e } } ( { \pmb x } ) | + | f _ { \star } ^ { \mathrm { t e } } ( { \pmb x } ) - { \pmb y } | \leq | f _ { \lambda } ( { \pmb x } ) - f _ { \star } ^ { \mathrm { t e } } ( { \pmb x } ) | + 2 M _ { { \pmb y } } ,
$$

and we further bound $| f _ { \lambda } ( { \pmb x } ) - f _ { \star } ^ { \mathrm { t e } } ( { \pmb x } ) |$ using the reproducing property:

$$
\begin{array} { r } { | f _ { \lambda } ( \pmb { x } ) - f _ { \star } ^ { \mathrm { t e } } ( \pmb { x } ) | = \left| \left. f _ { \lambda } - f _ { \star } ^ { \mathrm { t e } } , K _ { \mathcal { X } } ( \cdot , \pmb { x } ) \right. _ { \mathcal { H } _ { \mathcal { X } } } \right| \leq \kappa _ { \mathcal { X } } \cdot \left\| f _ { \lambda } - f _ { \star } ^ { \mathrm { t e } } \right\| _ { \mathcal { H } _ { \mathcal { X } } } \leq F \kappa _ { \mathcal { X } } \| \psi _ { \star } \| _ { \mathcal { H } _ { \mathcal { X } } } , } \end{array}
$$

where the bound for $\| f _ { \lambda } - f _ { \star } ^ { \mathrm { t e } } \| _ { \mathcal { H } _ { \mathcal { X } } }$ follows from Proposition 7 with $\gamma = 1$ (note that $\lambda \leq 1 )$ . Consequently,

$$
\begin{array} { r l } { \displaystyle \underset { ( \boldsymbol { x } , \boldsymbol { y } ) \in \mathcal { K } \times \mathcal { Y } } { \operatorname* { s u p } } \| \zeta _ { 2 } ( \boldsymbol { x } , \boldsymbol { y } ) \| _ { \mathcal { H } _ { \boldsymbol { x } } } = \underset { ( \boldsymbol { x } , \boldsymbol { y } ) \in \mathcal { X } \times \mathcal { Y } } { \operatorname* { s u p } } \| \widetilde { \eta } _ { \mu } ( \boldsymbol { y } ) ( f _ { \boldsymbol { \lambda } } ( \boldsymbol { x } ) - \boldsymbol { y } ) K _ { \mathcal { X } } ( \cdot , \boldsymbol { x } ) \| _ { \mathcal { H } _ { \boldsymbol { x } } } } & { } \\ { \le \underset { ( \boldsymbol { x } , \boldsymbol { y } ) \in \mathcal { X } \times \mathcal { Y } } { \operatorname* { s u p } } | \widetilde { \eta } _ { \mu } ( \boldsymbol { y } ) | \cdot | f _ { \boldsymbol { \lambda } } ( \boldsymbol { x } ) - \boldsymbol { y } | \cdot \| K _ { \mathcal { X } } ( \cdot , \boldsymbol { x } ) \| _ { \mathcal { H } _ { \boldsymbol { x } } } } & { } \\ { \le M _ { \eta } ( F \kappa _ { \boldsymbol { \chi } } \| \psi _ { \star } \| _ { \mathcal { H } _ { \boldsymbol { \mathcal { X } } } } + 2 M _ { \boldsymbol { y } } ) \kappa _ { \mathcal { X } } . } & { } \end{array}
$$

Applying Lemma 1 again, we obtain, with probability at least $1 - \delta / 2$

$$
\begin{array} { r l } { \displaystyle \left\| \left( W _ { \mathcal { X } \mathcal { X } } f _ { \lambda } - w _ { \mathcal { Y } } \right) - \left( \widehat { W } _ { \mathcal { X } \mathcal { X } } f _ { \lambda } - \widehat { w } _ { \mathcal { Y } } \right) \right\| _ { \mathcal { H } _ { \mathcal { X } } } = \left\| \mathbb { E } _ { ( \boldsymbol { x } , \boldsymbol { y } ) \sim p _ { \mathcal { X } \times \mathcal { Y } } ^ { \mathrm { t r } } } [ \zeta _ { 2 } ( \boldsymbol { x } , \boldsymbol { y } ) ] - \frac { 1 } { n _ { f } } \sum _ { j = 1 } ^ { n _ { f } } \zeta _ { 2 , j } \right\| _ { \mathcal { H } _ { \mathcal { X } } } } & { } \\ { \leq 1 0 M _ { \eta } ( F \kappa _ { \mathcal { X } } \| \psi _ { \star } \| _ { \mathcal { H } _ { \mathcal { X } } } + 2 M _ { \mathcal { Y } } ) \kappa _ { \mathcal { X } } \cdot n _ { f } ^ { - 1 / 2 } \log \frac { 4 } { \delta } . } \end{array}\tag{31}
$$

The proposition follows from (30) and (31) via the union bound.

Next, we present an upper bound for $B _ { 1 }$ in (29) via Proposition 7.

Proposition 7. Suppose that Assumption 2 holds with $r \in ( 0 , \tau - 1 / 2 ]$ . Then, for any $\gamma \in [ 0 , 1 ]$ the following bound holds:

$$
B _ { 1 } = \left\| V _ { \mathcal { X } \mathcal { X } } ^ { \frac { 1 - \gamma } { 2 } } \left( f _ { \star } ^ { \mathrm { t e } } - f _ { \lambda } \right) \right\| _ { \mathcal { H } _ { \mathcal { X } } } \le F \| \psi _ { \star } \| _ { \mathcal { H } _ { \mathcal { X } } } \cdot \lambda ^ { r + \frac { 1 - \gamma } { 2 } } .
$$

Proof. Recall that $f _ { \lambda } = g _ { \lambda } ( V _ { \mathcal { X } \mathcal { X } } ) V _ { \mathcal { X } \mathcal { X } } f _ { \star } ^ { \mathrm { t e } }$ and $f _ { \star } ^ { \mathrm { t e } } = V _ { \mathcal { X } \mathcal { X } } ^ { r } \psi _ { \star }$ . Using analogous reasoning to the proof of Proposition 2, we obtain

$$
\begin{array} { r l } & { B _ { 1 } = \left\| V _ { \mathcal { X } \mathcal { X } } ^ { \frac { 1 - \gamma } { 2 } } \left( f _ { \star } ^ { \mathrm { t e } } - f _ { \lambda } \right) \right\| _ { \mathcal { H } _ { \mathcal { X } } } = \left\| V _ { \mathcal { X } \mathcal { X } } ^ { \frac { 1 - \gamma } { 2 } } \left( I _ { \mathcal { X } } - g _ { \lambda } ( V _ { \mathcal { X } \mathcal { X } } ) V _ { \mathcal { X } \mathcal { X } } \right) V _ { \mathcal { X } \mathcal { X } } ^ { T } \psi _ { \star } \right\| _ { \mathcal { H } _ { \mathcal { X } } } } \\ & { \quad \leq \left\| V _ { \mathcal { X } \mathcal { X } } ^ { r + \frac { 1 - \gamma } { 2 } } \left( I - g _ { \lambda } ( V _ { \mathcal { X } \mathcal { X } } ) V _ { \mathcal { X } \mathcal { X } } \right) \right\| _ { \mathcal { H } _ { \mathcal { X } \mathcal { X } } \mathcal { H } _ { \mathcal { X } } } \cdot \| \psi _ { \star } \| _ { \mathcal { H } _ { \mathcal { X } } } \leq F \| \psi _ { \star } \| _ { \mathcal { H } _ { \mathcal { X } } } \cdot \lambda ^ { r + \frac { 1 - \gamma } { 2 } } , } \end{array}
$$

where the last inequality follows from the filter function property (11).

We bound $B _ { 2 }$ in (29) by Proposition 8.

Proposition 8. Assume that the bounds in Proposition 6 hold. If the sample sizes $n _ { \eta } , ~ n _ { f }$ and the

regularization parameter λ satisfy

$$
\left\{ \begin{array} { l l } { \displaystyle \lambda n _ { \eta } ^ { \frac { \iota } { 2 \iota + 2 } } \geq 4 \kappa _ { \mathcal { X } } ^ { 2 } \kappa y \varDelta _ { \eta } \log \frac { 4 } { \delta ^ { \prime } } , } \\ { \quad } \\ { \displaystyle \lambda n _ { f } ^ { 1 / 2 } \geq 4 0 M _ { \eta } \kappa _ { \mathcal { X } } ^ { 2 } \log \frac { 4 } { \delta } , } \end{array} \right.
$$

then for any $\gamma \in [ 0 , 1 ]$ , there holds

$$
B _ { 2 } =  V _ { \mathcal { X } \mathcal { X } } ^ { \frac { 1 - \gamma } { 2 } } ( \widehat { W } \mathcal { X } \mathcal { X } + \lambda I _ { \mathcal { X } } ) ^ { - 1 / 2 }  _ { \mathcal { H } _ { \mathcal { X } }  \mathcal { H } _ { \mathcal { X } } } \leq \sqrt { 2 } \lambda ^ { - \gamma / 2 } .
$$

Proof. To bound $B _ { 2 }$ , we decompose it as follows:

$$
\begin{array} { r l } & { B _ { 2 } = \left\| V _ { \mathcal { X } \mathcal { X } } ^ { \frac { 1 - \gamma } { 2 } } \left( \widehat { W } _ { \mathcal { X } \mathcal { X } } + \lambda I _ { \mathcal { X } } \right) ^ { - 1 / 2 } \right\| _ { \mathcal { H } _ { \mathcal { X } \to \mathcal { H } _ { \mathcal { X } } } } \overset { \mathrm { ( a ) } } { \leq } \left\| V _ { \mathcal { X } \mathcal { X } } ^ { 1 - \gamma } \left( \widehat { W } _ { \mathcal { X } \mathcal { X } } + \lambda I _ { \mathcal { X } } \right) ^ { - 1 } \right\| _ { \mathcal { H } _ { \mathcal { X } } \to \mathcal { H } _ { \mathcal { X } } } ^ { 1 / 2 } } \\ & { \quad \leq \left\| V _ { \mathcal { X } \mathcal { X } } ^ { 1 - \gamma } \left( V _ { \mathcal { X } \mathcal { X } } + \lambda I _ { \mathcal { X } } \right) ^ { - 1 } \right\| _ { \mathcal { H } _ { \mathcal { X } \to \mathcal { H } _ { \mathcal { X } } } } ^ { 1 / 2 } . \left\| \left( V _ { \mathcal { X } \mathcal { X } } + \lambda I _ { \mathcal { X } } \right) \left( \widehat { W } _ { \mathcal { X } \mathcal { X } } + \lambda I _ { \mathcal { X } } \right) ^ { - 1 } \right\| _ { \mathcal { H } _ { \mathcal { X } } \to \mathcal { H } _ { \mathcal { X } } } ^ { 1 / 2 } } \end{array}
$$

$$
\stackrel { \mathrm { ( b ) } } { \leq } \lambda ^ { - \gamma / 2 } \cdot \| ( V _ { \mathcal { X X } } + \lambda I _ { \mathcal { X } } ) ( \widehat { W } _ { \mathcal { X X } } + \lambda I _ { \mathcal { X } } ) ^ { - 1 } \| _ { \mathcal { H } _ { \mathcal { X } }  \mathcal { H } _ { \mathcal { X } } } ^ { 1 / 2 } .
$$

In step (a) we apply the Cordes inequality (Cordes, 1987, Lemma 5.1); step (b) follows from the uniform bound

$$
\left\| V _ { \mathcal { X } \mathcal { X } } ^ { 1 - \gamma } \left( V _ { \mathcal { X } \mathcal { X } } + \lambda I _ { \mathcal { X } } \right) ^ { - 1 } \right\| _ { \mathcal { H } _ { \mathcal { X } } \to \mathcal { H } _ { \mathcal { X } } } ^ { 1 / 2 } \le \left( \operatorname* { s u p } _ { t \ge 0 } \frac { t ^ { 1 - \gamma } } { t + \lambda } \right) ^ { 1 / 2 } \le \lambda ^ { - \gamma / 2 } .
$$

It remains to bound the operator norm of $( V \chi \chi + \lambda I \chi ) \left( \widehat { W } \chi \chi + \lambda I \chi \right) ^ { - 1 }$ . Observe that

$$
\widehat { W } \chi \chi + \lambda I \chi = \left( I \chi - \left( V \chi \chi - \widehat { W } \chi \chi \right) \left( V \chi \chi + \lambda I \chi \right) ^ { - 1 } \right) ^ { - 1 } ( V \chi \chi + \lambda I \chi ) ,
$$

which implies

$$
( V \chi \chi + \lambda I \chi ) \left( \widehat { W } \chi \chi + \lambda I \chi \right) ^ { - 1 } = \left( I \chi - ( V \chi \chi - \widehat { W } \chi \chi ) \left( V \chi \chi + \lambda I \chi \right) ^ { - 1 } \right) ^ { - 1 } .
$$

The bound for $\| V _ { \mathcal { X } \mathcal { X } } - \widehat { W } _ { \mathcal { X } \mathcal { X } } \| _ { \mathcal { H } _ { \mathcal { X } }  \mathcal { H } _ { \mathcal { X } } }$ is given in Proposition 6. Thus

$$
\begin{array} { r l } & { \quad \| ( V _ { \mathcal { X } \mathcal { X } } - \widehat { W } _ { \mathcal { X } \mathcal { X } } ) ( V _ { \mathcal { X } \mathcal { X } } + \lambda I _ { \mathcal { X } } ) ^ { - 1 } \| _ { \mathcal { H } _ { \mathcal { X } }  \mathcal { H } _ { \mathcal { X } } } } \\ & { \leq \| V _ { \mathcal { X } \mathcal { X } } - \widehat { W } _ { \mathcal { X } \mathcal { X } } \| _ { \mathcal { H } _ { \mathcal { X } }  \mathcal { H } _ { \mathcal { X } } } \cdot \| ( V _ { \mathcal { X } \mathcal { X } } + \lambda I _ { \mathcal { X } } ) ^ { - 1 } \| _ { \mathcal { H } _ { \mathcal { X } }  \mathcal { H } _ { \mathcal { X } } } } \\ & { \leq \kappa _ { \mathcal { X } } ^ { 2 } \kappa _ { \mathcal { Y } } \varDelta _ { \eta } \cdot \lambda ^ { - 1 } n _ { \eta } ^ { - \frac { \iota } { 2 ^ { \iota + 2 } } } \log \frac { 4 } { \delta ^ { \prime } } + 1 0 M _ { \eta } \kappa _ { \mathcal { X } } ^ { 2 } \cdot \lambda ^ { - 1 } n _ { f } ^ { - 1 / 2 } \log \frac { 4 } { \delta } . } \end{array}
$$

If the sample sizes $n _ { \eta } , ~ n _ { f }$ and the regularization parameter λ satisfy

$$
\left\{ \begin{array} { l l } { \displaystyle \lambda n _ { \eta } ^ { \frac { \iota } { 2 \iota + 2 } } \geq 4 \kappa _ { \mathcal { X } } ^ { 2 } \kappa y \varDelta _ { \eta } \log \frac { 4 } { \delta ^ { \prime } } , } \\ { \quad } \\ { \displaystyle \lambda n _ { f } ^ { 1 / 2 } \geq 4 0 M _ { \eta } \kappa _ { \mathcal { X } } ^ { 2 } \log \frac { 4 } { \delta } , } \end{array} \right.
$$

then

$$
\Big \| ( V _ { \mathcal { X } \mathcal { X } } - \widehat { W } _ { \mathcal { X } \mathcal { X } } ) ( V _ { \mathcal { X } \mathcal { X } } + \lambda I _ { \mathcal { X } } ) ^ { - 1 } \Big \| _ { \mathcal { H } _ { \mathcal { X } } \to \mathcal { H } _ { \mathcal { X } } } \leq \frac { 1 } { 4 } + \frac { 1 } { 4 } = \frac { 1 } { 2 } .
$$

This ensures that the Neumann series converges:

$$
\begin{array} { r l } & { ~ \| ( V _ { \mathcal { X } \mathcal { X } } + \lambda I _ { \mathcal { X } } ) ( \widehat { W } _ { \mathcal { X } \mathcal { X } } + \lambda I _ { \mathcal { X } } ) ^ { - 1 } \| _ { \mathcal { H } _ { \mathcal { X } }  \mathcal { H } _ { \mathcal { X } } } } \\ & { = \| ( I _ { \mathcal { X } } - ( V _ { \mathcal { X } \mathcal { X } } - \widehat { W } _ { \mathcal { X } \mathcal { X } } ) ( V _ { \mathcal { X } \mathcal { X } } + \lambda I _ { \mathcal { X } } ) ^ { - 1 } ) ^ { - 1 } \| _ { \mathcal { H } _ { \mathcal { X } }  \mathcal { H } _ { \mathcal { X } } } \leq \frac { 1 } { 1 - 1 / 2 } = 2 . } \end{array}
$$

Combining these estimates, we conclude

$$
B _ { 2 } \leq \lambda ^ { - \gamma / 2 } \cdot \left\| ( V _ { \mathcal { X } \mathcal { X } } + \lambda I _ { \mathcal { X } } ) ( \widehat { W } _ { \mathcal { X } \mathcal { X } } + \lambda I _ { \mathcal { X } } ) ^ { - 1 } \right\| _ { \mathcal { H } _ { \mathcal { X } } \to \mathcal { H } _ { \mathcal { X } } } ^ { 1 / 2 } \leq \sqrt { 2 } \lambda ^ { - \gamma / 2 } .
$$

The following proposition bounds $B _ { 3 }$ in (29).

Proposition 9. Assume that the bounds in Proposition 6 hold. If the sample sizes $n _ { \eta } , ~ n _ { f }$ and the regularization parameter λ satisfy

$$
\left\{ \begin{array} { l l } { \displaystyle \lambda n _ { \eta } ^ { \frac { \iota } { 2 \iota + 2 } } \geq 4 \kappa _ { \mathcal { X } } ^ { 2 } \kappa y \varDelta _ { \eta } \log \frac { 4 } { \delta ^ { \prime } } , } \\ { \quad } \\ { \displaystyle \lambda n _ { f } ^ { 1 / 2 } \geq 4 0 M _ { \eta } \kappa _ { \mathcal { X } } ^ { 2 } \log \frac { 4 } { \delta } , } \end{array} \right.
$$

then there holds

$$
\begin{array} { r l } & { B _ { 3 } = \Big \| ( \widehat { W } \chi \chi + \lambda I _ { \chi } ) ^ { 1 / 2 } g _ { \lambda } ( \widehat { W } \chi \chi ) \big ( \widehat { W } \chi \chi fint _ { \lambda } - \widehat { w } _ { y } \big ) \Big \| _ { \mathscr { H } _ { \chi } } } \\ & { \quad \leq 2 0 E M _ { \eta } \big ( F \kappa _ { \chi } \| \psi _ { \star } \| _ { \mathscr { H } _ { \chi } } + 2 M _ { y } \big ) \kappa _ { \chi } \cdot \lambda ^ { - 1 / 2 } n _ { f } ^ { - 1 / 2 } \log \frac { 4 } { \delta } } \\ & { \qquad + 2 E \kappa _ { \chi } \kappa _ { \mathscr { H } } \Big ( F \kappa _ { \chi } \| \psi _ { \star } \| _ { \mathscr { H } _ { \chi } } + 2 M _ { y } \Big ) \varDelta _ { \eta } \cdot \lambda ^ { - 1 / 2 } n _ { \eta } ^ { - \frac { \iota } { 2 \iota + 2 } } \log \frac { 4 } { \delta ^ { \prime } } } \\ & { \qquad + 2 \sqrt { 2 } E F \| \psi _ { \star } \| _ { \mathscr { H } _ { \chi } } \cdot \lambda ^ { r + \frac { 1 } { 2 } } . } \end{array}
$$

Proof. By adding and subtracting $\left( W _ { \mathcal { X X } } f _ { \lambda } - { w _ { y } } \right)$ , we decompose $B _ { 3 }$ as

$$
B _ { 3 } \leq B _ { 3 , 1 } + B _ { 3 , 2 } ,
$$

where

$$
B _ { 3 , 1 } = \Big \| ( \widehat { W } x x + \lambda I { x } ) ^ { 1 / 2 } g _ { \lambda } ( \widehat { W } x x ) \left( ( \widehat { W } x x f _ { \lambda } - \widehat { w } _ { y } ) - ( W x x f _ { \lambda } - w _ { y } ) \right) \Big \| _ { \mathscr { H } _ { x } } ,
$$

$$
B _ { 3 , 2 } = \Big \| ( \widehat W \chi \chi + \lambda I \chi ) ^ { 1 / 2 } g _ { \lambda } ( \widehat W \chi \chi ) \left( W \chi \chi f _ { \lambda } - w _ { y } \right) \Big \| _ { \mathscr { H } _ { \chi } } .
$$

These two terms are bounded separately as follows:

1. For $B _ { 3 , 1 }$ , by the filter function property (10), we have

$$
\begin{array} { r } { \| ( \widehat W _ { \mathcal { X } \mathcal { X } } + \lambda I _ { \mathcal { X } } ) ^ { 1 / 2 } g _ { \lambda } ( \widehat W _ { \mathcal { X } \mathcal { X } } ) \| _ { \mathcal { H } _ { \mathcal { X } }  \mathcal { H } _ { \mathcal { X } } } \leq 2 E \cdot \lambda ^ { - 1 / 2 } . } \end{array}
$$

This, together with the bounds from Proposition 6, yields

$$
\begin{array} { r l } & { { B _ { 3 , 1 } } \leq \left\| { ( \widehat W \chi \chi + \lambda I \chi ) ^ { 1 / 2 } g _ { \lambda } ( \widehat W \chi \chi ) } \right\| _ { \mathcal { H } _ { \chi \to \mathcal { H } _ { \chi } } } \cdot \left\| { ( \widehat W \chi \chi f _ { \lambda } - \widehat w _ { y } ) - ( W \chi \chi f _ { \lambda } - w _ { y } ) } \right\| _ { \mathcal { H } _ { \chi } } } \\ & { \qquad \leq 2 0 E M _ { \eta } ( F \kappa _ { \chi } \| \psi _ { * } \| _ { \mathcal { H } _ { \chi } } + 2 M _ { y } ) \kappa _ { \chi } \cdot \lambda ^ { - 1 / 2 } n _ { f } ^ { - 1 / 2 } \log \frac { 4 } { \delta } . } \end{array}\tag{32}
$$

2. Observe that

$$
\begin{array} { r l } & { W _ { \mathcal { X } \mathcal { X } } f _ { \lambda } - \pmb { w } _ { y } = \mathbb { E } _ { ( \pmb { x } , y ) \sim p _ { \mathcal { X } \times \mathcal { X } } ^ { \mathrm { t r } } } [ \widetilde { \eta } _ { \mu } ( y ) ( f _ { \lambda } ( \pmb { x } ) - y ) K _ { \mathcal { X } } ( \cdot , \pmb { x } ) ] } \\ & { \quad \quad \quad \quad = \mathbb { E } _ { ( \pmb { x } , y ) \sim p _ { \mathcal { X } \times \mathcal { X } } ^ { \mathrm { t r } } } [ ( \widetilde { \eta } _ { \mu } ( y ) - \eta _ { \star } ( y ) ) ( f _ { \lambda } ( \pmb { x } ) - y ) K _ { \mathcal { X } } ( \cdot , \pmb { x } ) ] } \\ & { \quad \quad \quad \quad \quad + \mathbb { E } _ { ( \pmb { x } , y ) \sim p _ { \mathcal { X } \times \mathcal { Y } } ^ { \mathrm { t r } } } [ \eta _ { \star } ( y ) ( f _ { \lambda } ( \pmb { x } ) - y ) K _ { \mathcal { X } } ( \cdot , \pmb { x } ) ] . } \end{array}
$$

Furthermore, since $f _ { \star } ^ { \mathrm { t e } } ( { \pmb x } ) = \mathbb { E } _ { y \sim p _ { y | \mathcal { X } } ^ { \mathrm { t e } } } [ y \mid { \pmb x } ]$ , we have

$$
\begin{array} { r l } & { \quad \mathbb { E } _ { ( \pmb { x } , \pmb { y } ) \sim p _ { \pmb { \chi } \times \pmb { \chi } } ^ { \mathrm { t r } } } \big [ \eta _ { \star } ( \pmb { y } ) \big ( f _ { \lambda } ( \pmb { x } ) - \pmb { y } \big ) K _ { \mathcal { X } } ( \cdot , \pmb { x } ) \big ] } \\ & { = \mathbb { E } _ { ( \pmb { x } , \pmb { y } ) \sim p _ { \pmb { \chi } \times \pmb { \chi } } ^ { \mathrm { t e } } } \big [ \big ( f _ { \lambda } ( \pmb { x } ) - \pmb { y } \big ) K _ { \mathcal { X } } ( \cdot , \pmb { x } ) \big ] = \mathbb { E } _ { \pmb { x } \sim p _ { \pmb { \chi } } ^ { \mathrm { t e } } } \big [ \big ( f _ { \lambda } ( \pmb { x } ) - f _ { \star } ^ { \mathrm { t e } } ( \pmb { x } ) \big ) K _ { \mathcal { X } } ( \cdot , \pmb { x } ) \big ] } \\ & { = V _ { \mathcal { X } \mathcal { X } } \big ( f _ { \lambda } - f _ { \star } ^ { \mathrm { t e } } \big ) . } \end{array}
$$

Therefore, $B _ { 3 , 2 }$ can be further decomposed as

$$
\begin{array} { r l } & { \displaystyle B _ { 3 , 2 } \leq 2 E \cdot \lambda ^ { - 1 / 2 } \cdot \Big | \Big | \mathbb { E } _ { ( \pmb { x } , \pmb { y } ) \sim p _ { x \times \mathscr { X } } ^ { \mathrm { t r } } } [ ( \widetilde { \eta } _ { \mu } ( \pmb { y } ) - \eta _ { \star } ( \pmb { y } ) ) ( f _ { \lambda } ( \pmb { x } ) - \pmb { y } ) K _ { \mathscr { X } } ( \cdot , \pmb { x } ) ] \Big | \Big | _ { \mathscr { H } _ { \pmb { \mathscr { X } } } } } \\ & { \qquad + \left\| ( \widehat { W } _ { \pmb { \mathscr { X } } \pmb { \mathscr { X } } } + \lambda I _ { \mathscr { X } } ) ^ { 1 / 2 } g _ { \lambda } ( \widehat { W } _ { \mathscr { X } \mathscr { X } } ) V _ { \mathscr { X } \mathscr { X } } \left( f _ { \lambda } - f _ { \star } ^ { \mathrm { t e } } \right) \right\| _ { \mathscr { H } _ { \pmb { \mathscr { X } } } } , } \end{array}\tag{33}
$$

where the factor $2 E \cdot \lambda ^ { - 1 / 2 }$ follows from bounding $\| ( \widehat { W } \chi \chi + \lambda I \chi ) ^ { 1 / 2 } g _ { \lambda } ( \widehat { W } \chi \chi ) \| _ { \mathcal { H } _ { \chi } \chi \chi } \chi _ { \chi }$

For the first term in (33), recall that in the proof of Proposition 6, we derive

$$
\operatorname* { s u p } _ { ( \pmb { x } , \pmb { y } ) \in \mathcal { X } \times \mathcal { Y } } | f _ { \lambda } ( \pmb { x } ) - \pmb { y } | \leq F \kappa \chi \| \psi _ { \star } \| _ { \mathcal { H } _ { \mathcal { X } } } + 2 M _ { \pmb { y } } .
$$

This bound, together with (28), give

$$
\begin{array} { r l } & { \quad \left\| \mathbb { E } _ { ( \pmb { x } , \pmb { y } ) \sim p _ { \pmb { x } \times \pmb { y } } ^ { \mathrm { t r } } } [ ( \widetilde { \eta } _ { \mu } ( \pmb { y } ) - \eta _ { \star } ( \pmb { y } ) ) ( f _ { \lambda } ( \pmb { x } ) - \pmb { y } ) K \pmb { \chi } ( \cdot , \pmb { x } ) ] \right\| _ { \mathcal { H } _ { \pmb { \chi } } } } \\ & { \le \kappa _ { \pmb { \chi } \kappa \mathscr { y } } \left( F \kappa _ { \pmb { \chi } } \| \psi _ { \star } \| _ { \mathcal { H } _ { \pmb { \chi } } } + 2 M _ { \pmb { y } } \right) \Delta _ { \eta } \cdot n _ { \eta } ^ { - \frac { \iota } { 2 \iota + 2 } } \log \frac { 4 } { \delta ^ { \prime } } . } \end{array}\tag{34}
$$

For the second term in (33), we decompose it as follows:

$$
\begin{array} { r l } & { \qquad \ldots \leq \| ( \widehat { W } _ { \mathcal { X } \mathcal { X } } + \lambda I _ { \mathcal { X } } ) ^ { 1 / 2 } g _ { \lambda } ( \widehat { W } _ { \mathcal { X } \mathcal { X } } ) ( \widehat { W } _ { \mathcal { X } \mathcal { X } } + \lambda I _ { \mathcal { X } } ) ^ { 1 / 2 } \| _ { \mathcal { H } _ { \mathcal { X } }  \mathcal { H } _ { \mathcal { X } } } } \\ & { \qquad \cdot \| ( \widehat { W } _ { \mathcal { X } \mathcal { X } } + \lambda I _ { \mathcal { X } } ) ^ { - 1 / 2 } V _ { \mathcal { X } \mathcal { X } } ^ { 1 / 2 } \| _ { \mathcal { H } _ { \mathcal { X } }  \mathcal { H } _ { \mathcal { X } } } \cdot \| V _ { \mathcal { X } \mathcal { X } } ^ { 1 / 2 } ( f _ { \lambda } - f _ { \star } ^ { \mathrm { t e } } ) \| _ { \mathcal { H } _ { \mathcal { X } } } } \\ & { \leq 2 E \cdot \| ( \widehat { W } _ { \mathcal { X } \mathcal { X } } + \lambda I _ { \mathcal { X } } ) ^ { - 1 / 2 } V _ { \mathcal { X } \mathcal { X } } ^ { 1 / 2 } \| _ { \mathcal { H } _ { \mathcal { X }  \mathcal { H } _ { \mathcal { X } } } } \cdot F \| \psi _ { \star } \| _ { \mathcal { H } _ { \mathcal { X } } } \cdot \lambda ^ { r + \frac { 1 } { 2 } } , } \end{array}
$$

where the last inequality follows from the filter function property (10) and Proposition 7 (with $\gamma = 0 )$ . It remains to bound $\| ( \widehat { W } \chi \chi + \lambda I _ { \mathcal { X } } ) ^ { - 1 / 2 } V _ { \mathcal { X } \mathcal { X } } ^ { 1 / 2 } \| _ { \mathcal { H } _ { \mathcal { X } } \to \mathcal { H } _ { \mathcal { X } } }$ . By the Cordes inequality (Cordes, 1987, Lemma 5.1), we have

$$
\begin{array} { r l } { \left\| { ( \widehat { W } _ { \mathcal { X } \mathcal { X } } + \lambda I _ { \mathcal { X } } ) ^ { - 1 / 2 } } V _ { \mathcal { X } \mathcal { X } } ^ { 1 / 2 } \right\| _ { \mathcal { H } _ { \mathcal { X } \mathcal { X } } } = \left\| { V _ { \mathcal { X } \mathcal { X } } ^ { 1 / 2 } ( \widehat { W } _ { \mathcal { X } \mathcal { X } } + \lambda I _ { \mathcal { X } } ) ^ { - 1 / 2 } } \right\| _ { \mathcal { H } _ { \mathcal { X } \mathcal { X } } } } & { } \\ { \leq \left\| { V _ { \mathcal { X } \mathcal { X } } ( \widehat { W } _ { \mathcal { X } \mathcal { X } } + \lambda I _ { \mathcal { X } } ) ^ { - 1 } } \right\| _ { \mathcal { H } _ { \mathcal { X } \mathcal { X } } } ^ { 1 / 2 } . } \end{array}
$$

Moreover, since

$$
\left( V \chi \chi + \lambda I \chi \right) \left( \widehat { W } \chi \chi + \lambda I \chi \right) ^ { - 1 } - V \chi \chi \left( \widehat { W } \chi \chi + \lambda I \chi \right) ^ { - 1 } = \lambda ( \widehat { W } \chi \chi + \lambda I \chi ) ^ { - 1 }
$$

is a positive operator, we obtain

$$
\begin{array} { r } { \left\| ( \widehat W \chi \chi + \lambda I \chi ) ^ { - 1 } V \chi \chi \right\| _ { \mathcal H _ { \mathcal X } \to \mathcal H _ { \mathcal X } } \leq \left\| ( V \chi \chi + \lambda I \chi ) \left( \widehat W \chi \chi + \lambda I \chi \right) ^ { - 1 } \right\| _ { \mathcal H _ { \mathcal X } \to \mathcal H _ { \mathcal X } } . } \end{array}
$$

As shown in Proposition $8 ,$ , under the assumed conditions on $n _ { \eta } , ~ n _ { f }$ , and $\lambda ,$ the right-hand side is bounded by 2. Combining these, we get

$$
\begin{array} { r l } & { \quad \left\| ( \widehat { W } x x + \lambda I x ) ^ { 1 / 2 } g _ { \lambda } ( \widehat { W } x x ) V x x \left( f _ { \lambda } - f _ { \star } ^ { \mathrm { t e } } \right) \right\| _ { \mathcal { H } _ { x } } } \\ & { \leq 2 E \cdot \left\| ( \widehat { W } x x + \lambda I _ { \mathcal { X } } ) ^ { - 1 / 2 } V _ { \mathcal { X } \mathcal { X } } ^ { 1 / 2 } \right\| _ { \mathcal { H } _ { \mathcal { X } \to \mathcal { H } _ { \mathcal { X } } } } \cdot F \| \psi _ { \star } \| _ { \mathcal { H } _ { \mathcal { X } } } \cdot \lambda ^ { r + \frac { 1 } { 2 } } \leq 2 \sqrt { 2 } E F \| \psi _ { \star } \| _ { \mathcal { H } _ { \mathcal { X } } } \cdot \lambda ^ { r + \frac { 1 } { 2 } } . } \end{array}\tag{35}
$$

Substituting (34) and (35) into (33) yields a bound for $B _ { 3 , 2 }$

$$
\begin{array} { l } { { B _ { 3 , 2 } \leq 2 E \kappa _ { \mathcal { X } } \kappa _ { \mathcal { Y } } \left( F \kappa _ { \mathcal { X } } \| \psi _ { \star } \| _ { \mathcal { H } _ { \mathcal { X } } } + 2 M _ { \mathcal { Y } } \right) \varDelta _ { \eta } \cdot \lambda ^ { - 1 / 2 } n _ { \eta } ^ { - \frac { \iota } { 2 \iota + 2 } } \log \frac { 4 } { \delta ^ { \prime } } } } \\ { { \qquad + 2 \sqrt { 2 } E F \| \psi _ { \star } \| _ { \mathcal { H } _ { \mathcal { X } } } \cdot \lambda ^ { r + \frac { 1 } { 2 } } . } } \end{array}\tag{36}
$$

Finally, combining (32) and (36), we bound $B _ { 3 }$ as

$$
\begin{array} { r l } & { B _ { 3 } \le B _ { 3 , 1 } + B _ { 3 , 2 } \le 2 0 E M _ { \eta } ( F \kappa _ { \mathcal { X } } \| \psi _ { \star } \| _ { \mathcal { H } _ { \mathcal { X } } } + 2 M _ { y } ) \kappa _ { \mathcal { X } } \cdot \lambda ^ { - 1 / 2 } n _ { f } ^ { - 1 / 2 } \log \frac { 4 } { \delta } } \\ & { \qquad + 2 E \kappa _ { \mathcal { X } } \kappa _ { \mathcal { Y } } \left( F \kappa _ { \mathcal { X } } \| \psi _ { \star } \| _ { \mathcal { H } _ { \mathcal { X } } } + 2 M _ { y } \right) \varDelta _ { \eta } \cdot \lambda ^ { - 1 / 2 } n _ { \eta } ^ { - \frac { \iota } { 2 \iota + 2 } } \log \frac { 4 } { \delta ^ { \prime } } } \\ & { \qquad + 2 \sqrt { 2 } E F \| \psi _ { \star } \| _ { \mathcal { H } _ { \mathcal { X } } } \cdot \lambda ^ { r + \frac { 1 } { 2 } } . } \end{array}
$$

This completes the proof.

The following proposition bounds the last term $B _ { 4 }$ in (29).

Proposition 10. Assume that the bounds in Proposition 6 hold. Then we have

$$
\begin{array} { r l } & { B _ { 4 } = \left\| ( \widehat { W } \chi x + \lambda I _ { \mathcal { X } } ) ^ { 1 / 2 } \left( I _ { \mathcal { X } } - g _ { \lambda } ( \widehat { W } \chi x ) \widehat { W } x x \right) f _ { \lambda } \right\| _ { \mathscr { H } _ { \mathcal { X } } } } \\ & { \quad \leq 2 E F \| \psi _ { \star } \| _ { \mathscr { H } _ { \lambda } } \cdot \lambda ^ { r + \frac { 1 } { 2 } } } \\ & { \qquad + 2 0 ( r + 1 ) E F M _ { \eta } ^ { r } \kappa _ { \mathcal { X } } ^ { 2 r } \kappa y \| \psi _ { \star } \| _ { \mathscr { H } _ { \mathcal { X } } } \varDelta _ { \eta } \left( \lambda ^ { 1 / 2 } n _ { \eta } ^ { - \frac { \frac { \epsilon } { 2 + 2 } \cdot \operatorname* { m i n } \{ r , 1 \} } { 2 \dot { \delta } ^ { \prime } } } \log \frac { 4 } { \delta ^ { \prime } } + \lambda ^ { 1 / 2 } n _ { f } ^ { - \frac { \operatorname* { m i n } \{ r , 1 \} } { 2 } } \log \frac { 4 } { \delta } \right) . } \end{array}
$$

Proof. We begin by recalling that $f _ { \lambda } = g _ { \lambda } ( V _ { \lambda \chi } ) V _ { \lambda \chi } f _ { \star } ^ { \mathrm { t e } }$ and $f _ { \star } ^ { \mathrm { t e } } = V _ { \mathcal { X } \mathcal { X } } ^ { r }$ ψ<sub>⋆</sub>:

$$
\begin{array} { r l } & { B _ { 4 } = \left\| ( \widehat { W } \chi \chi + \lambda I \chi ) ^ { 1 / 2 } \left( I \chi - g _ { \lambda } ( \widehat { W } \chi \chi ) \widehat { W } \chi \chi \right) f _ { \lambda } \right\| _ { \mathcal { H } _ { \chi } } } \\ & { \quad = \left\| ( \widehat { W } \chi \chi + \lambda I \chi ) ^ { 1 / 2 } \left( I \chi - g _ { \lambda } ( \widehat { W } \chi \chi ) \widehat { W } \chi \chi \right) g _ { \lambda } ( V \chi \chi ) V _ { \chi \chi } ^ { r + 1 } \psi _ { \star } \right\| _ { \mathcal { H } _ { \chi } } } \\ & { \quad \leq E \| \psi _ { \star } \| _ { \mathcal { H } _ { \chi } } \cdot \left\| ( \widehat { W } _ { \chi \chi } + \lambda I \chi ) ^ { 1 / 2 } \left( I _ { \chi } - g _ { \lambda } ( \widehat { W } \chi \chi ) \widehat { W } \chi \chi \right) V _ { \chi \chi } ^ { r } \right\| _ { \mathcal { H } _ { \chi \to \mathcal { H } _ { \chi } } } . } \end{array}
$$

In the last step, we apply the filter function property (10) to obtain $\| g _ { \lambda } ( V _ { \mathcal { X } \mathcal { X } } ) V _ { \mathcal { X } \mathcal { X } } \| _ { \mathcal { H } _ { \mathcal { X } } \to \mathcal { H } _ { \mathcal { X } } } \le E$ To proceed, we add and subtract $\widehat { W } _ { \mathcal { X } \mathcal { X } } ^ { r }$

$$
\begin{array} { r l } & { \quad \left\| ( \widehat { W } _ { \mathcal { X } \mathcal { X } } + \lambda I _ { \mathcal { X } } ) ^ { 1 / 2 } \left( I _ { \mathcal { X } } - g _ { \lambda } ( \widehat { W } _ { \mathcal { X } \mathcal { X } } ) \widehat { W } _ { \mathcal { X } \mathcal { X } } \right) V _ { \mathcal { X } \mathcal { X } } ^ { r } \right\| _ { \mathcal { H } _ { \mathcal { X } \mathcal { X } } \mathcal { H } _ { \mathcal { X } } } } \\ & { \leq \left\| ( \widehat { W } _ { \mathcal { X } \mathcal { X } } + \lambda I _ { \mathcal { X } } ) ^ { 1 / 2 } \left( I _ { \mathcal { X } } - g _ { \lambda } ( \widehat { W } _ { \mathcal { X } \mathcal { X } } ) \widehat { W } _ { \mathcal { X } \mathcal { X } } \right) \widehat { W } _ { \mathcal { X } \mathcal { X } } ^ { r } \right\| _ { \mathcal { H } _ { \mathcal { X } \mathcal { X } } } } \\ & { \quad \quad + \left\| ( \widehat { W } _ { \mathcal { X } \mathcal { X } } + \lambda I _ { \mathcal { X } } ) ^ { 1 / 2 } \left( I _ { \mathcal { X } } - g _ { \lambda } ( \widehat { W } _ { \mathcal { X } \mathcal { X } } ) \widehat { W } _ { \mathcal { X } \mathcal { X } } \right) \right\| _ { \mathcal { H } _ { \mathcal { X } \mathcal { X } } \mathcal { H } _ { \mathcal { X } } } \cdot \left\| V _ { \mathcal { X } \mathcal { X } } ^ { r } - \widehat { W } _ { \mathcal { X } \mathcal { X } } ^ { r } \right\| _ { \mathcal { H } _ { \mathcal { X } \mathcal { X } } \mathcal { H } _ { \mathcal { X } } } } \\ & { \leq 2 F \cdot \lambda ^ { r + \frac 1 2 } + 2 F \cdot \lambda ^ { 1 / 2 } \cdot \left\| V _ { \mathcal { X } \mathcal { X } } ^ { r } - \widehat { W } _ { \mathcal { X } \mathcal { X } } ^ { r } \right\| _ { \mathcal { H } _ { \mathcal { X } \mathcal { X } } } , } \end{array}
$$

where the last inequality follows from the filter function property (11). Next, we bound $\parallel V _ { \mathcal { X } \mathcal { X } } ^ { r } -$ $\widehat { W } _ { \mathcal { X } \mathcal { X } } ^ { r } \Vert _ { \mathcal { H } _ { \mathcal { X } }  \mathcal { H } _ { \mathcal { X } } }$ using Lemma 2. Under our assumptions, we have

$$
\operatorname* { m a x } \biggl \{ \| V \chi \chi \| _ { \mathcal { H } _ { \mathcal { X } } \to \mathcal { H } _ { \mathcal { X } } } , \left\| \widehat { W } _ { \mathcal { X } \mathcal { X } } \right\| _ { \mathcal { H } _ { \mathcal { X } } \to \mathcal { H } _ { \mathcal { X } } } \biggr \} \leq M _ { \eta } \kappa _ { \mathcal { X } } ^ { 2 } ,
$$

which implies

$$
\begin{array} { r } { \| { V _ { \mathcal { X } \mathcal { X } } ^ { r } - \widehat { W } _ { \mathcal { X } \mathcal { X } } ^ { r } } \| _ { \mathcal { H } _ { \mathcal { X } }  \mathcal { H } _ { \mathcal { X } } } \leq \{ \begin{array} { l l } { \| { V x x - \widehat { W } x x } \| _ { \mathcal { H } _ { \mathcal { X } }  \mathcal { H } _ { \mathcal { X } } } ^ { r } , } & { r \in ( 0 , 1 ] ; } \\ { r ( M _ { \eta } \kappa _ { \mathcal { X } } ^ { 2 } ) ^ { r - 1 } \| { V _ { \mathcal { X } \mathcal { X } } - \widehat { W } _ { \mathcal { X } \mathcal { X } } } \| _ { \mathcal { H } _ { \mathcal { X } }  \mathcal { H } _ { \mathcal { X } } } , } & { r > 1 . } \end{array}  } \end{array}
$$

The bound for $\| V _ { \mathcal { X } \mathcal { X } } - \widehat { W } _ { \mathcal { X } \mathcal { X } } \| _ { \mathcal { H } _ { \mathcal { X } }  \mathcal { H } _ { \mathcal { X } } }$ is given in Proposition 6:

$$
\Big \| { \cal V } \chi \chi - \widehat W \chi \chi \Big \| _ { \mathscr { H } _ { \mathcal { X } } \to \mathscr { H } _ { \chi } } \le \kappa _ { \chi } ^ { 2 } \kappa y \varDelta _ { \eta } \cdot n _ { \eta } ^ { - \frac { \iota } { 2 \iota + 2 } } \log \frac { 4 } { \delta ^ { \prime } } + 1 0 M _ { \eta } \kappa _ { \chi } ^ { 2 } \cdot n _ { f } ^ { - 1 / 2 } \log \frac { 4 } { \delta } .
$$

Consequently,

$$
\begin{array} { r l } & { \quad \left\| V _ { \mathcal { X } \mathcal { X } } ^ { r } - \widehat { W } _ { \mathcal { X } \mathcal { X } } ^ { r } \right\| _ { \mathcal { H } _ { \mathcal { X } \to \mathcal { H } _ { \mathcal { X } } } } } \\ & { \leq ( r + 1 ) ( M _ { \eta } \kappa _ { \mathcal { X } } ^ { 2 } ) ^ { r - 1 } \left( \kappa _ { \mathcal { X } ^ { \kappa } \mathcal { Y } } ^ { 2 } \varDelta _ { \eta } \cdot n _ { \eta } ^ { - \frac { \iota } { 2 \iota + 2 } } \log \frac { 4 } { \delta ^ { \prime } } + 1 0 M _ { \eta } \kappa _ { \mathcal { X } } ^ { 2 } \cdot n _ { f } ^ { - 1 / 2 } \log \frac { 4 } { \delta } \right) ^ { \operatorname* { m i n } \{ r , 1 \} } } \\ & { \leq 1 0 ( r + 1 ) ( M _ { \eta } \kappa _ { \mathcal { X } } ^ { 2 } ) ^ { r } \kappa _ { \mathcal { Y } } \varDelta _ { \eta } \left( n _ { \eta } ^ { - \frac { \iota } { 2 \iota + 2 } \cdot \operatorname* { m i n } \{ r , 1 \} } \log \frac { 4 } { \delta ^ { \prime } } + n _ { f } ^ { - \frac { \operatorname* { m i n } \{ r , 1 \} } { 2 } } \log \frac { 4 } { \delta } \right) . } \end{array}
$$

Combining these estimates yields

$$
\begin{array} { r l } & { B _ { 4 } \leq E \| \psi _ { \star } \| _ { \mathcal { H } _ { x } } \cdot \Big \| ( \widehat { W } x x + \lambda I _ { \mathcal { X } } ) ^ { 1 / 2 } \left( I _ { \mathcal { X } } - g _ { \lambda } ( \widehat { W } x x ) \widehat { W } x x \right) V _ { \mathcal { X } \mathcal { X } } ^ { r } \Big \| _ { \mathcal { H } _ { x } \to \mathcal { H } _ { x } } } \\ & { \quad \leq 2 E F \| \psi _ { \star } \| _ { \mathcal { H } _ { x } } \left( \lambda ^ { r + \frac { 1 } { 2 } } + \lambda ^ { 1 / 2 } \cdot \Big \| V _ { \mathcal { X } \mathcal { X } } ^ { r } - \widehat { W } _ { \mathcal { X } \mathcal { X } } ^ { r } \Big \| _ { \mathcal { H } _ { x } \to \mathcal { H } _ { \mathcal { X } } } \right) } \\ & { \quad \leq 2 E F \| \psi _ { \star } \| _ { \mathcal { H } _ { \mathcal { X } } } \cdot \lambda ^ { r + \frac { 1 } { 2 } } } \\ & { \quad \quad + 2 0 ( r + 1 ) E F M _ { \eta } ^ { r } \kappa _ { \mathcal { X } } ^ { 2 r } \kappa y \| \psi _ { \star } \| _ { \mathcal { H } _ { x } } \varDelta _ { \eta } \left( \lambda ^ { 1 / 2 } n _ { \eta } ^ { - \frac { \iota } { 2 \iota + 2 } \cdot \operatorname* { m i n } \{ r , 1 \} } \log \frac { 4 } { \delta ^ { \prime } } + \lambda ^ { 1 / 2 } n _ { f } ^ { - \frac { \operatorname* { m i n } \{ r , 1 \} } { 2 } } \log \frac { 4 } { \delta } \right) . } \end{array}
$$

This completes the proof of the theorem.

We now proceed to prove Theorem 2.

Proof of Theorem 2. By setting $\delta ^ { \prime } \mathit { \Delta } = \delta / 2$ , the density ratio error bound in (28) holds with probability at least $1 - \delta / 2 \cdot$

$$
\operatorname* { s u p } _ { y \in \mathcal { V } } | \eta _ { \star } ( y ) - \widetilde { \eta } _ { \mu } ( y ) | \leq \kappa _ { \mathcal { V } } \varDelta _ { \eta } \cdot n _ { \eta } ^ { - \frac { \iota } { 2 \iota + 2 } } \log \frac { 8 } { \delta } .
$$

Similarly, applying Proposition 6 with $\delta / 2$ in place of $\delta ,$ the bounds in Proposition 6 hold with probability at least $1 - \delta / 2$ . Consequently, if the sample sizes $n _ { \eta } , ~ n _ { f }$ and the regularization parameter λ satisfy

$$
\left\{ \begin{array} { l l } { \displaystyle \lambda n _ { \eta } ^ { \frac { \iota } { 2 \iota + 2 } } \geq 4 \kappa _ { \chi } ^ { 2 } \kappa y \varDelta _ { \eta } \log \frac 8 \delta , } \\ { \quad } \\ { \displaystyle \lambda n _ { f } ^ { 1 / 2 } \geq 4 0 M _ { \eta } \kappa _ { \chi } ^ { 2 } \log \frac 8 \delta , } \end{array} \right.
$$

then all bounds from Propositions 7–10 hold simultaneously with probability at least $1 - \delta \colon$

$$
\begin{array} { r l } & { B _ { 1 } \leq F \| \psi _ { k } \| _ { \mathcal H _ { X } } \cdot \chi ^ { + \frac { 1 - \eta } { 2 } } , \quad B _ { 2 } \leq \sqrt { 2 } \lambda ^ { - \gamma / 2 } , } \\ & { B _ { 3 } \leq 2 0 E \mathcal M _ { \eta } ( F \kappa _ { X } \| \psi _ { k } \| _ { \mathcal H _ { X } } + 2 \mathcal M _ { y } ) \kappa _ { X } \cdot \lambda ^ { - 1 / 2 } n _ { f } ^ { - 1 / 2 } \log \frac 8 \delta } \\ & { \quad + 2 E \kappa _ { X } \kappa _ { \mathcal H } \left( F \kappa _ { X } \| \psi _ { k } \| _ { \mathcal H _ { X } } + 2 \mathcal M _ { y } \right) \varDelta _ { \eta } \cdot \lambda ^ { - 1 / 2 } n _ { \eta } ^ { - 4 } \log \frac 8 \delta } \\ & { \quad + 2 \sqrt { 2 } E F \| \psi _ { k } \| _ { \mathcal H _ { X } } \cdot \chi ^ { + \frac { 1 } { 2 } } , } \\ & { B _ { 4 } \leq 2 E F \| \psi _ { k } \| _ { \mathcal H _ { X } } \cdot \chi ^ { + \frac { 1 } { 2 } } } \\ & { \quad + 2 0 ( r + 1 ) E F \mathcal M _ { \eta } ^ { r } \kappa _ { X } ^ { 2 r } \kappa _ { \mathcal H } \| \psi _ { * } \| _ { \mathcal H _ { x } } \varDelta _ { \eta } \left( \lambda ^ { 1 / 2 } n _ { \eta } ^ { - 4 , \operatorname* { m i n } \{ r , 1 \} } \log \frac 8 \delta + \lambda ^ { 1 / 2 } n _ { f } ^ { - \frac { \operatorname* { m i n } \{ r , 1 \} } } \log \frac 8 \delta \right) , } \end{array}
$$

where we write $A = \iota / ( 2 \iota + 2 )$ for brevity.

We now choose the regularization parameter $\lambda$ as

$$
\lambda = \left\{ { \begin{array} { l l } { n _ { \eta } ^ { - { \frac { A } { r + 1 } } } , } & { { \mathrm { i f ~ } } n _ { \eta } ^ { 2 A } < n _ { f } ; } \\ { } & { } \\ { n _ { f } ^ { - { \frac { 1 } { 2 r + 2 } } } , } & { { \mathrm { i f ~ } } n _ { \eta } ^ { 2 A } \geq n _ { f } . } \end{array} } \right.
$$

Under this choice, a direct calculation shows that

$$
\operatorname* { m a x } \biggl \{ \lambda ^ { - 1 / 2 } n _ { f } ^ { - 1 / 2 } , \lambda ^ { - 1 / 2 } n _ { \eta } ^ { - A } , \lambda ^ { 1 / 2 } n _ { \eta } ^ { - A \cdot \operatorname* { m i n } \{ r , 1 \} } , \lambda ^ { 1 / 2 } n _ { f } ^ { - \frac { \operatorname* { m i n } \{ r , 1 \} } { 2 } } \biggr \} \leq \lambda ^ { r + \frac { 1 } { 2 } } .
$$

Consequently, we obtain

$$
\left\| V _ { \mathcal { X } \mathcal { X } } ^ { \frac { 1 - \gamma } { 2 } } \left( f _ { \star } ^ { \mathrm { t e } } - \widehat { f } _ { \lambda } \right) \right\| _ { \mathcal { H } _ { \mathcal { X } } } \leq B _ { 1 } + B _ { 2 } ( B _ { 3 } + B _ { 4 } ) \leq \varDelta _ { f } \cdot \lambda ^ { r + \frac { 1 - \gamma } { 2 } } \log \frac { 8 } { \delta } ,
$$

where

$$
\begin{array} { r l } & { \varDelta _ { f } = \left( ( 4 + 2 \sqrt { 2 } ) E + 1 \right) F \| \psi _ { \star } \| _ { \mathcal { H } _ { \mathcal { X } } } + 2 0 \sqrt { 2 } E M _ { \eta } ( F \kappa _ { \mathcal { X } } \| \psi _ { \star } \| _ { \mathcal { H } _ { \mathcal { X } } } + 2 M _ { y } ) \kappa _ { \mathcal { X } } } \\ & { \qquad + \left. 2 \sqrt { 2 } E \kappa _ { \mathcal { X } } \kappa _ { \mathcal { Y } } \left( F \kappa _ { \mathcal { X } } \| \psi _ { \star } \| _ { \mathcal { H } _ { \mathcal { X } } } + 2 M _ { y } \right) \varDelta _ { \eta } \right. } \\ & { \qquad + \left. 4 0 \sqrt { 2 } ( r + 1 ) E F M _ { \eta } ^ { r } \kappa _ { \mathcal { X } } ^ { 2 r } \kappa _ { \mathcal { Y } } \| \psi _ { \star } \| _ { \mathcal { H } _ { \mathcal { X } } } \varDelta _ { \eta } \right. } \end{array}\tag{37}
$$

is a constant independent of $n _ { \eta } , n _ { f }$ , or $\delta .$

Specializing to $\gamma = 0$ or 1 yields bounds for $\| f _ { \star } ^ { \mathrm { t e } } - \widehat { f } _ { \lambda } \| _ { \mathcal { L } ^ { 2 } ( \mathcal { X } , p _ { \mathcal { X } } ^ { \mathrm { t e } } ) }$ and $\| f _ { \star } ^ { \mathrm { t e } } - \widehat { f } _ { \lambda } \| _ { \mathcal { H } _ { \mathcal { X } } }$ , respectively:

$$
\begin{array} { r } { \left\| f _ { \star } ^ { \mathrm { t e } } - \widehat f _ { \lambda } \right\| _ { \mathcal L ^ { 2 } ( \mathcal X , p _ { \mathcal X } ^ { \mathrm { t e } } ) } \leq \varDelta _ { f } \cdot \log \frac 8 \delta \cdot \left\{ n _ { \eta } ^ { - A \frac { r + 1 / 2 } { r + 1 } } , \quad \mathrm { i f ~ } n _ { \eta } ^ { 2 A } < n _ { f } ; \right. } \end{array}
$$

and

$$
\left\| f _ { \star } ^ { \mathrm { t e } } - \widehat f _ { \lambda } \right\| _ { \mathcal { H } _ { \chi } } \leq \varDelta _ { f } \cdot \log \frac 8 \delta \cdot \left\{ \begin{array} { l l } { { n } _ { \eta } ^ { - A \frac { r } { r + 1 } } , } & { \mathrm { i f ~ } { n } _ { \eta } ^ { 2 A } < n _ { f } ; } \\ { \qquad } \\ { { n } _ { f } ^ { - \frac { r } { 2 r + 2 } } , } & { \mathrm { i f ~ } { n } _ { \eta } ^ { 2 A } \geq n _ { f } . } \end{array} \right.
$$

We have thus established the theorem.

## 4.3 Auxiliary Lemmas

The following lemmas are used throughout the proofs.

Lemma 1 (Caponnetto and de Vito, 2007, Proposition 2). Let $\xi$ be a random variable taking values in a separable Hilbert space $\mathcal { H }$ . Assume that $\| \xi \| _ { \mathcal { H } } \le G$ holds almost surely for some constant $G > 0$ . Then, for any i.i.d. sample $\{ \xi _ { i } \} _ { i = 1 } ^ { n }$ and any $\delta \in ( 0 , 1 )$ ,

$$
\left. \mathbb { E } [ \xi ] - \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \xi _ { i } \right. _ { \mathcal { H } } \leq 1 0 U n ^ { - 1 / 2 } \log \frac { 2 } { \delta }
$$

holds with probability at least $1 - \delta .$

Lemma 2 (Blanchard and Kr¨amer, 2010, Lemma E.3). Let A and B be positive self-adjoint operators on a separable Hilbert space $\mathcal { H }$ such that max $\{ \| A \| _ { \mathcal { H } \to \mathcal { H } } , \| B \| _ { \mathcal { H } \to \mathcal { H } } \} \le U$ . Then, for any $c \geq 0$ 2

$$
\begin{array} { r } { \| A ^ { c } - B ^ { c } \| _ { \mathcal H \to \mathcal H } \le \left\{ \begin{array} { l l } { \| A - B \| _ { \mathcal H \to \mathcal H } ^ { c } , } & { c \le 1 ; } \\ { \qquad \lfloor c U ^ { c - 1 } \| A - B \| _ { \mathcal H \to \mathcal H } , } & { c > 1 . } \end{array} \right. } \end{array}
$$

## References

Amr Alexandari, Anshul Kundaje, and Avanti Shrikumar. Maximum likelihood with bias-corrected calibration is hard-to-beat at label shift adaptation. In The 37th International Conference on Machine Learning, volume 119, pages 222–232. Proceedings of Machine Learning Research, 2020.

Kamyar Azizzadenesheli, Anqi Liu, Fanny Yang, and Animashree Anandkumar. Regularized learning for domain adaptation under label shifts. In The 7th International Conference on Learning Representations, 2019.

Frank Bauer, Sergei Pereverzyev, and Lorenzo Rosasco. On regularization algorithms in learning theory. Journal of Complexity, 23(1):52–72, 2007.

Stefen Bickel, Michael Br¨uckner, and Tobias Schefer. Discriminative learning under covariate shift. Journal of Machine Learning Research, 10(75):2137–2155, 2009.

Gilles Blanchard and Nicole Kr¨amer. Optimal learning rates for kernel conjugate gradient regression. In Advances in Neural Information Processing Systems, volume 23. Curran Associates, Inc., 2010.

Andrea Caponnetto and Ernesto de Vito. Optimal rates for the regularized least-squares algorithm. Foundations of Computational Mathematics, 7(3):331–368, 2007.

Kristy Choi, Chenlin Meng, Yang Song, and Stefano Ermon. Density ratio estimation via infinitesimal classification. In The 25th International Conference on Artificial Intelligence and Statistics, volume 151, pages 2552–2573. Proceedings of Machine Learning Research, 2022.

Heinz O. Cordes. Spectral Theory of Linear Diferential Operators and Comparison Algebras. Cambridge University Press, 1987.

Felipe Cucker and Ding-Xuan Zhou. Learning Theory: An Approximation Theory Viewpoint. Cambridge University Press, 2007.

Ernesto de Vito, Lorenzo Rosasco, Andrea Caponnetto, Umberto de Giovannini, and Francesca Odone. Learning from examples as an inverse problem. Journal of Machine Learning Research, 6(30):883–904, 2005.

Heinz W. Engl and Ronny Ramlau. Regularization of inverse problems. In Encyclopedia of applied and computational mathematics, pages 1233–1241. Springer, 2015.

Jun Fan, Zheng-Chu Guo, and Lei Shi. Spectral algorithms under covariate shift. arXiv preprint, abs/2504.12625, 2025.

Xingdong Feng, Xin He, Yuling Jiao, Lican Kang, and Caixing Wang. Deep nonparametric quantile regression under covariate shift. Journal of Machine Learning Research, 25(385):1–50, 2024.

Saurabh Garg, Yifan Wu, Sivaraman Balakrishnan, and Zachary Lipton. A unified view of label shift estimation. In Advances in Neural Information Processing Systems, volume 33, pages 3290– 3300. Curran Associates, Inc., 2020.

Elke R. Gizewski, Lukas Mayer, Bernhard A. Moser, Duc Hoan Nguyen, Sergiy Pereverzyev Jr., Sergei Pereverzyev, Natalia Shepeleva, and Werner Zellinger. On a regularization of unsupervised domain adaptation in RKHS. Applied and Computational Harmonic Analysis, 57:201–227, 2022.

Elke R. Gizewski, Shuai Lu, Stephanie Mangesius, Duc Hoan Nguyen, and Sergei Pereverzyev. The impact of smoothness of kernels and target functions on unsupervised covariate shift adaptation in rkhs. Applied and Computational Harmonic Analysis, 83:101866, 2026.

Davit Gogolashvili. Importance weighting correction of regularized least-squares for target shift. arXiv preprint, abs/2210.09709, 2026.

Davit Gogolashvili, Matteo Zecchin, Motonobu Kanagawa, Marios Kountouris, and Maurizio Filippone. When is importance weighting correction needed for covariate shift adaptation? arXiv preprint, abs/2303.04020, 2023.

Arthur Gretton, Alex Smola, Jiayuan Huang, Marcel Schmittfull, Karsten Borgwardt, and Bernhard Sch¨okopf. Covariate shift by kernel mean matching. In Dataset Shift in Machine Learning. Massachusetts Institute of Technology Press, 2008.

Zheng-Chu Guo and Lei Shi. Online learning algorithms tackling covariate shift. Advances in Computational Mathematics, 51(6):60, 2025.

Zheng-Chu Guo, Shao-Bo Lin, and Ding-Xuan Zhou. Learning theory of distributed spectral algorithms. Inverse Problems, 33(7):074009, 2017.

Takafumi Kanamori, Shohei Hido, and Masashi Sugiyama. A least-squares approach to direct importance estimation. Journal of Machine Learning Research, 10(48):1391–1445, 2009.

Hwanwoo Kim, Xin Zhang, Jiwei Zhao, and Qinglong Tian. ReTaSA: a nonparametric functional estimation approach for addressing continuous target shift. In The 12th International Conference on Learning Representations, 2024.

Rainer Kress. Linear Integral Equations. Springer, 2014.

Zachary Lipton, Yu-Xiang Wang, and Alexander Smola. Detecting and correcting for label shift with black box predictors. In The 35th International Conference on Machine Learning, volume 80, pages 3122–3130. Proceedings of Machine Learning Research, 2018.

Ren-Rui Liu and Zheng-Chu Guo. Spectral algorithms in misspecified regression: convergence under covariate shift. arXiv preprint, abs/2509.05106, 2025.

Ren-Rui Liu, Jun Fan, Lei Shi, and Zheng-Chu Guo. Unbounded density ratio estimation and its application to covariate shift adaptation. arXiv preprint, abs/2603.29725, 2026.

Laura Lo Gerfo, Lorenzo Rosasco, Francesca Odone, Ernesto de Vito, and Alessandro Verri. Spectral algorithms for supervised learning. Neural Computation, 20(7):1873–1897, 2008.

Cong Ma, Reese Pathak, and Martin J. Wainwright. Optimally tackling covariate shift in RKHSbased nonparametric regression. The Annals of Statistics, 51(2):738–761, 2023.

Hanna L. Myleiko and Sergei G. Solodky. On recovering the Radon-Nikodym derivative under the big data assumption. Journal of Complexity, page 102001, 2025.

Tuan Duong Nguyen, Marthinus Christofel, and Masashi Sugiyama. Continuous target shift adaptation in supervised learning. In Asian Conference on Machine Learning, volume 45, pages 285–300. Proceedings of Machine Learning Research, 2016.

Jing Qin. Inferences for case-control and semiparametric two-sample density ratio models. Biometrika, 85(3):619–630, 1998.

Joaquin Qui˜nonero-Candela, Masashi Sugiyama, Anton Schwaighofer, and Neil D. Lawrence. Dataset Shift in Machine Learning. Massachusetts Institute of Technology Press, 2008.

Benjamin Rhodes, Kai Xu, and Michael Gutmann. Telescoping density-ratio estimation. In Advances in Neural Information Processing Systems, volume 33, pages 4905–4916. Curran Associates, Inc., 2020.

Marco Saerens, Patrice Latinne, and Christine Decaestecker. Adjusting the outputs of a classifier to new a priori probabilities: a simple procedure. Neural Computation, 14(1):21–41, 2002.

Simon J Sheather and Michael C Jones. A reliable data-based bandwidth selection method for kernel density estimation. Journal of the Royal Statistical Society: Series B (Methodological), 53 (3):683–690, 1991.

Hidetoshi Shimodaira. Improving predictive inference under covariate shift by weighting the loglikelihood function. Journal of Statistical Planning and Inference, 90(2):227–244, 2000.

Steve Smale and Ding-Xuan Zhou. Estimating the approximation error in learning theory. Analysis and Applications, 1(01):17–41, 2003.

Amos Storkey. When training and test sets are diferent: characterizing learning transfer. In Dataset Shift in Machine Learning. Massachusetts Institute of Technology Press, 2008.

Masashi Sugiyama, Taiji Suzuki, Shinichi Nakajima, Hisashi Kashima, Paul Von B¨unau, and Motoaki Kawanabe. Direct importance estimation for covariate shift adaptation. Annals of the Institute of Statistical Mathematics, 60(4):699–746, 2008.

Masashi Sugiyama, Taiji Suzuki, and Takafumi Kanamori. Density Ratio Estimation in Machine Learning. Cambridge University Press, 2012.

Vladimir N. Vapnik. Statistical Learning Theory. John Wiley & Sons, Inc., 1998.

Shuntuo Xu, Zhou Yu, and Jian Huang. Estimating unbounded density ratios: Applications in error control under covariate shift. arXiv preprint, abs/2504.01031, 2025.

Kun Zhang, Bernhard Sch¨olkopf, Krikamol Muandet, and Zhikun Wang. Domain adaptation under target and conditional shift. In The 30th International Conference on Machine Learning, volume 28, pages 819–827. Proceedings of Machine Learning Research, 2013.

Siming Zheng, Guohao Shen, Yuanyuan Lin, and Jian Huang. Error analysis for deep relu feedforward density-ratio estimation with bregman divergence. Journal of Machine Learning Research, 27(15):1–60, 2026.