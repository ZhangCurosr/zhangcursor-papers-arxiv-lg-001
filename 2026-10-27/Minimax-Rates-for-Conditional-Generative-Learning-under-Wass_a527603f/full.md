# Minimax Rates for Conditional Generative Learning under Wasserstein Loss

Sungwon Park<sup>∗</sup>, Kyoung-Sook Moon<sup>†</sup> , and Hongjoong Kim<sup>∗</sup>

Abstract. We study minimax rates for conditional generative learning under integrated Wasserstein loss, with H¨older continuous conditional laws and an unrestricted covariate distribution. We evaluate unrestricted generators through their conditional output distributions. For expected Wasserstein loss, the minimax rate agrees with the rate under uniform covariates and follows from balancing localization bias against unconditional distribution estimation error. Higher moments of the loss can instead be dominated by unobserved covariate regions, which force an additional risk contribution of order $n ^ { - 1 }$ , where n is the sample size. An adaptive localization argument gives matching upper bounds. The results cover unrestricted response laws and possibly vanishing Besov densities for $W _ { 1 }$ and extend to higher Wasserstein orders under uniform response density bounds and integrability conditions. A multiscale lower bound closes the logarithmic gap in unconditional $W _ { 1 }$ minimax rates in dimension two for both unrestricted distributions and Besov density classes. We also establish minimax rates for conditional Brenier maps under density, integrability, and uniform curvature conditions.

Key words. conditional generative learning; conditional distribution estimation; Wasserstein distance; minimax rates; statistical optimal transport

MSC codes. 62C20, 62G05, 49Q22

1. Introduction. Many prediction problems require a conditional distribution rather than a conditional mean, for example when the response is intrinsically random or multimodal. Such problems arise in probabilistic forecasting [42, 44], simulation-based Bayesian inference [38, 24], and probabilistic surrogate modeling for scientific systems [12, 39]. We formulate conditional generative learning as nonparametric estimation of conditional output distributions and study its minimax rates under Wasserstein loss.

Let $( X _ { 1 } , Y _ { 1 } ) , \dots , ( X _ { n } , Y _ { n } )$ be i.i.d. observations with $X \in [ 0 , 1 ] ^ { d _ { x } }$ and $Y \in [ 0 , 1 ] ^ { d _ { y } }$ , and let

$$
x \longmapsto Q _ { x } = P _ { Y | X = x }
$$

be a regular conditional distribution of $Y$ given $X = x$ . We integrate $W _ { p } ^ { \kappa } ( \widehat { Q } _ { x } , Q _ { x } )$ with respect to the marginal law $P _ { X }$ . We impose no density, doubling, or lower mass condition on $P _ { X }$ , which may be discrete, continuous, singular, or highly nonuniform. For $p = 1$ we allow unrestricted distributions and densities in Besov classes that need not be bounded away from zero. For $1 < p < \infty$ we assume that the response densities are uniformly bounded away from zero and infinity, while conditional Brenier maps are treated under a separate geometric stability class.

For the uniform covariate distribution, the benchmark rate follows from the usual local bias–variance balance. Pooling observations over a covariate neighborhood of diameter $h$ incurs H¨older error of order $h ^ { \beta }$ , while the efective local sample size is $n h ^ { d _ { x } }$ . The stochastic term is therefore the minimax error for estimating one distribution from $n h ^ { d _ { x } }$ observations. Balancing these terms defines $A _ { n } ( \tau )$ , with $\tau = 0$ for unrestricted distributions and $\tau > 0$ for densities in a Besov class.

Our first result shows that $A _ { n } ( \tau )$ remains sharp for integrated Wasserstein loss over all Borel covariate marginals. Weighting local errors by cell probability allows them to be summed uniformly over $P _ { X }$ . Higher moments behave diferently: an atom of mass of order $n ^ { - 1 }$ is unobserved with probability bounded away from zero, forcing an additional $n ^ { - 1 }$ contribution to the integrated $W _ { p } ^ { \kappa }$ risk. The resulting rate is $A _ { n } ( \tau ) ^ { \kappa } \vee n ^ { - 1 }$

For $p > 1$ , a comparison with a negative Sobolev norm gives the local distribution estimation rate under uniform response density bounds. The corresponding bounds extend to every fixed finite moment. The choices $\kappa = 1$ and $\kappa = p$ give Wasserstein loss and its pth power, respectively.

The conditional results rely on sharp unconditional distribution estimation rates. We prove the missing $W _ { 1 }$ lower bound of order $\sqrt { \log m / m }$ for unrestricted distributions and Besov density classes in dimension two. A single resolution gives only $m ^ { - 1 / 2 }$ ; the logarithmic factor comes from combining perturbations across a logarithmic number of admissible resolutions.

A conditional generator transforms a reference noise law into a conditional response law. We also study conditional Brenier maps, the quadratic optimal transport maps from normalized Lebesgue measure $\rho$ on the response cube. Uniform curvature bounds transfer squared $W _ { 2 }$ error for the target distribution to squared $L ^ { 2 } ( \rho )$ map error. A construction based on inverse potentials gives the matching lower bound, yielding the minimax map rate $A _ { n } ( \tau ) ^ { 2 }$

1.1. Related work. Existing minimax results for conditional density and distribution estimation use several notions of loss and regularity. Efromovich [22] develops minimax theory for conditional density estimation under fixed and random designs, and Brunel, Comte, and Lacour [11] study conditional distribution functions. Li, Neykov, and Balakrishnan [32] obtain minimax rates without a marginal density assumption under total variation regularity in the covariate, while Bilodeau, Foster, and Roy [8] derive rates from empirical Hellinger entropy. This setting is also related to distribution-free nonparametric regression [25]; for example, Kohler and Krzy˙zak [30] show that irregular random designs can create an $n ^ { - 1 }$ barrier in a noiseless interpolation problem.

The closest Wasserstein results difer mainly in their assumptions on the covariate and response distributions. Dombry, Modeste, and Pic [20] prove universal consistency in $W _ { p }$ and obtain $W _ { 1 }$ minimax rates for univariate responses on particular classes without restricting the covariate marginal. Pic et al. [41] study minimax rates under the CRPS. For multivariate responses, B´en´ezet, Cheng, and Jaimungal [7] analyze local empirical estimators under a uniform lower bound on the covariate law relative to Lebesgue measure. Tang and Yang [53] obtain minimax rates, up to logarithmic factors, under H¨older integral probability metrics for smooth density models and distributions supported on manifolds. We impose H¨older continuity directly on $x \mapsto Q _ { x }$ and obtain matching multivariate Wasserstein minimax bounds over all Borel covariate marginals.

Wasserstein-based conditional generators and conditional difusion models are studied in [34, 52, 48]. Kumar, Yang, and Lin [31] analyze likelihood-based conditional generative models. Chemseddine et al. [13] study conditional Wasserstein distances. Tamano and Imaizumi [51] derive minimax bounds for causal conditional distribution learning under a regular target design and a smooth latent pushforward model. Balakrishnan [3] shows that estimating a valid transport map includes estimating its induced distribution. Our results give analogous statistical limits for conditional generators on the H¨older classes considered here.

For the unconditional distribution estimation problem we use empirical Wasserstein bounds $[ 2 3 , \ 4 6 , \ 5 8 , \ 9 ]$ and results for smooth densities under Wasserstein and related adversarial losses [47, 55, 33, 37, 18]. Dedecker and Merlev\`ede [16] study higher moments of empirical $W _ { 1 }$ errors, and Bobkov and Ledoux [9] give a Fourier analytic proof of the critical two-dimensional empirical rate. Niles-Weed and Berthet [37] obtain the optimal polynomial rates for smooth densities and identify the role of a density lower bound for $p > 1$ . Their lower bound leaves a logarithmic gap in the critical two-dimensional case. Divol [18] gives the corresponding critical upper rate for densities on smooth manifolds without boundary. Section 4 establishes the matching critical lower bound for unrestricted distributions and the Besov classes considered here.

The Brenier map result builds on minimax and plugin estimation of smooth transport maps [28, 35, 19, 56] and on recent stability and minimax comparison results [4, 3]. Covariatedependent constructions appear in [49, 50, 27], including the conditional Brenier estimator of Baptista et al. [5] and conditional map error bounds of Al-Jarrah et al. [2]. Combining these stability results with our localization bounds gives an upper bound uniform over all $P _ { X }$ . The inverse potential construction gives the matching conditional lower bound in the stated class.

Section 2 states the model and main results. Section 3 proves the localization argument for unrestricted covariate distributions, Section 4 gives the multiscale response lower bound, and Section 5 treats conditional Brenier maps. The appendices contain the remaining unconditional estimates, adaptive localization, bounds for higher Wasserstein orders, measurability arguments, and class verifications.

2. Main results. We define the losses and regularity classes, then state the minimax rates for conditional distributions and Brenier maps.

2.1. Statistical model and losses. Let $d _ { x } , d _ { y } \geq 1 , \chi = [ 0 , 1 ] ^ { d _ { x } }$ , and $\mathcal { V } = [ 0 , 1 ] ^ { d _ { y } }$ . Suppose that $( X _ { 1 } , Y _ { 1 } ) , \dots , ( X _ { n } , Y _ { n } )$ are i.i.d. with joint law P. We denote the marginal law of $X$ by $P _ { X }$ , write $S _ { X } = \mathrm { s u p p } ( P _ { X } )$ , and fix a Borel regular conditional distribution

$$
Q _ { x } = P _ { Y | X = x }
$$

of $Y$ given $X = x$

For probability measures $\mu , \nu$ on $[ 0 , 1 ] ^ { d }$ , let $\Pi ( \mu , \nu )$ denote the probability measures on $[ 0 , 1 ] ^ { d } \times [ 0 , 1 ] ^ { d }$ with marginals $\mu$ and $\nu .$ For $1 \leq p < \infty ,$ define

$$
W _ { p } ^ { p } ( \mu , \nu ) = \operatorname* { i n f } _ { \pi \in \Pi ( \mu , \nu ) } \int | y - y ^ { \prime } | ^ { p } \pi ( d y , d y ^ { \prime } ) .\tag{2.1}
$$

Thus $W _ { p } ^ { p } ( \mu , \nu )$ is the minimum expected transportation cost over all couplings of $\mu$ and $\nu .$ For $p = 1$ , the Kantorovich–Rubinstein theorem gives

$$
W _ { 1 } ( \mu , \nu ) = \operatorname * { s u p } _ { \operatorname { L i p } ( f ) \leq 1 } \left| \int f d ( \mu - \nu ) \right| .\tag{2.2}
$$

For a finite signed measure $\sigma$ with total mass zero, we similarly set

$$
\| \sigma \| _ { \mathrm { K R } } = \operatorname* { s u p } _ { \mathrm { L i p } ( f ) \leq 1 } \left| \int f d \sigma \right| .\tag{2.3}
$$

Hence $W _ { 1 } ( \mu , \nu ) = \| \mu - \nu \| _ { \mathrm { K R } }$ . We use this dual representation in the unconditional lower bound of Section 4.

For $1 \leq p < \infty , 1 \leq \kappa < \infty$ , and an estimator $x \mapsto { \widehat { Q } } _ { x } .$ , define the integrated $W _ { p } ^ { \kappa }$ risk by

$$
\mathcal { R } _ { n , p , \kappa } ( \widehat { Q } ; P ) = \mathbb { E } _ { P ^ { \otimes n } } \left[ \int _ { \mathcal { X } } W _ { p } ^ { \kappa } ( \widehat { Q } _ { x } , Q _ { x } ) P _ { X } ( d x ) \right] .\tag{2.4}
$$

For a class $\mathcal { P }$ of joint distributions, the corresponding minimax risk is

$$
\mathcal { R } _ { n , p , \kappa } ^ { \star } ( \mathcal { P } ) = \operatorname* { i n f } _ { \widehat { Q } } \operatorname* { s u p } _ { P \in \mathcal { P } } \mathcal { R } _ { n , p , \kappa } ( \widehat { Q } ; P ) .\tag{2.5}
$$

The κth root of the risk in (2.4) is the $L ^ { \kappa }$ norm of the Wasserstein error over the training sample and an independent test covariate. The infimum is over probability kernels measurable jointly in the training sample and the query covariate.

A measurable conditional generator driven by independent noise induces such a kernel. Conversely, since $\mathcal { V }$ is a standard Borel space, every such kernel can be realized by a measurable function of the training sample, the query covariate, and an independent uniform random variable [29, Lemma 3.22]. Thus conditional distribution estimation and conditional generative learning have the same minimax risk when generators are unrestricted and evaluated through their output laws.

For the integrated $W _ { p } ^ { \kappa }$ risk, the cases $\kappa = 1$ and $\kappa = p$ give ordinary $W _ { p }$ loss and the usual $W _ { p } ^ { p }$ transport cost, respectively. For $p = 1$ the results hold on the broad distribution classes introduced below, whereas for $1 < p < \infty$ ∞ they use the density bounds in Assumption 2.3. All regularity assumptions on $x \mapsto Q _ { x }$ are imposed only on $S _ { X } .$ , since values of a conditional version outside the support do not afect (2.4).

For conditional map estimation, let $\rho$ be normalized Lebesgue measure on $\mathcal { V } .$ For a measurable map $T : \mathcal { V }  \mathcal { V } , T _ { \# } \rho$ denotes the law of $T ( Z )$ for $Z \sim \rho .$ . Since $\rho$ is absolutely continuous, the quadratic optimal transport map $T _ { x } = \nabla \varphi _ { x }$ from ρ to $Q _ { x }$ exists and is unique $\rho \mathrm { - }$ -almost everywhere, with $\varphi _ { x }$ convex $[ 1 0 , 4 5 ]$ . Map estimators $\widehat { T } _ { x } : \mathcal { V } \to \mathcal { V }$ are based on the training sample and are required to be measurable jointly in the sample and covariate as $L ^ { 2 } ( \rho )$ -valued functions. Define their integrated squared $L ^ { 2 } ( \rho )$ risk by

$$
\mathcal { R } _ { n } ^ { \operatorname* { m a p } } ( \widehat { T } ; P ) = \mathbb { E } _ { P ^ { \otimes n } } \left[ \int _ { \mathcal { X } } \| \widehat { T } _ { x } - T _ { x } \| _ { L ^ { 2 } ( \rho ) } ^ { 2 } P _ { X } ( d x ) \right] .\tag{2.6}
$$

We use $a \vee b = \operatorname* { m a x } \{ a , b \}$ and $t _ { + } = \operatorname* { m a x } \{ t , 0 \}$ . We write $a \lesssim b$ when $a \leq C b$ for a constant $C$ independent of both the sample size and the covariate distribution, and $a \asymp b$ when $a \lesssim b$ and $b \lesssim a$ . The dimensions and model parameters are fixed in their stated ranges. Implicit constants and sample-size thresholds may depend on them, on each fixed loss exponent $\kappa ,$ and on the chosen wavelet system.

2.2. Regularity classes. The assumptions separate regularity in the covariate from regularity of the response law. In all four classes below, $P _ { X }$ ranges over all Borel probability measures on X. Fix $0 < \beta \leq 1$ and $0 < L _ { X } < \infty$ . For a specified Wasserstein order $1 \leq p < \infty$ , we use the covariate H¨older condition

$$
W _ { p } ( Q _ { x } , Q _ { x ^ { \prime } } ) \leq L _ { X } | x - x ^ { \prime } | ^ { \beta } , \qquad x , x ^ { \prime } \in S _ { X } .\tag{2.7}
$$

Each assumption below specifies the order p and any additional regularity of the response law.

Assumption 2.1 (Unrestricted response $W _ { 1 } \ \mathsf { m o d e l } )$ There is a Borel version $x \mapsto Q _ { x }$ of the conditional distribution for which (2.7) holds with $p = 1$ . No density or response smoothness condition is imposed on the individual distributions $Q _ { x }$

Denote this class by $\mathcal { P } _ { \beta } ^ { \mathrm { f r e e } }$

For densities with smoothness $\tau > 0$ , let $1 \leq r , v \leq \infty$ and use the standard Besov space $B _ { r , v } ^ { \tau } ( [ 0 , 1 ] ^ { d } ) ;$ ; see [14, 54]. Here $\tau$ is the smoothness index in the response variable, while r and v determine aggregation within and across resolutions. We use the equivalent characterization by boundary wavelets recalled in Appendix A. For each response dimension $d ,$ fix $L _ { Y } > \| 1 \| _ { B _ { r , v } ^ { \tau } ( [ 0 , 1 ] ^ { d } ) }$ , taking $d = d _ { y }$ in the conditional models.

Assumption 2.2 (Besov density model for $W _ { 1 } )$ There is a jointly Borel function $( x , y ) \mapsto$ $q _ { x } ( y )$ such that $Q _ { x } = q _ { x } d y \ f o r \ x \in S _ { X }$ , (2.7) holds with $p = 1$ , and

$$
\operatorname* { s u p } _ { x \in S _ { X } } \| q _ { x } \| _ { B _ { r , v } ^ { \tau } ( \mathcal { V } ) } \leq L _ { Y } .\tag{2.8}
$$

No pointwise lower or upper bound on $q _ { x }$ is assumed.

Denote the resulting class by $\mathcal { P } _ { \beta , \tau , r , v } ^ { B }$ . For the associated unconditional problems, let $\mathcal { Q } _ { d } ^ { \mathrm { f r e e } }$ be the class of all Borel probability measures on $[ 0 , 1 ] ^ { d }$ and put

$$
\mathcal { Q } _ { d , \tau , r , v } ^ { B } = \left\{ q d y : q \geq 0 , \int q = 1 , \| q \| _ { B _ { r , v } ^ { \tau } } \leq L _ { Y } \right\} .\tag{2.9}
$$

For higher Wasserstein orders we assume that the conditional densities are uniformly bounded away from zero and infinity. Fix $1 < p < \infty , \tau \geq 0 , 1 \leq r , v \leq \infty$ , and constants $0 < \underline { { q } } < 1 <$ $\overline { { q } } < \infty$

Assumption 2.3 $( W _ { p }$ density bounds). There is a jointly Borel function $( x , y ) \mapsto q _ { x } ( y )$ such that $Q _ { x } = q _ { x } d y \ f o r \ x \in S _ { X }$ , (2.7) holds for the fixed order $1 < p < \infty$ , and

$$
\underline { { q } } \le q _ { x } ( y ) \le \overline { { q } } , \qquad x \in S _ { X } , \quad f o r \ a . e . \ y \in \mathcal { V } .\tag{2.10}
$$

When $\tau > 0$ , we also impose (2.8) and require $r \geq p \ i f d _ { y } > 2 ;$ when $\tau = 0$ , no Besov condition is imposed and the indices r, v are ignored.

Denote this class by $\mathcal { P } _ { \beta , \tau , r , v } ^ { ( p ) } ( \underline { { q } } , \overline { { q } } )$ , and let $\mathcal { Q } _ { d , \tau , r , v } ^ { ( p ) } ( \underline { { q } } , \overline { { q } } )$ be the corresponding unconditional distribution class on $[ 0 , 1 ] ^ { d }$ , with $r \geq p$ when $\tau > 0$ and $d > 2$ . The lower density bound is used in the comparison with a negative Sobolev norm for $W _ { p } ,$ and the upper bound controls the stochastic wavelet error. These conditions are suficient for the argument below and are not claimed to be minimal.

For the conditional Brenier map class, fix $\tau \geq 0 , 1 \leq r , v \leq \infty , 1 < \overline { { q } } < \infty .$ and $\Lambda > 1$

Assumption 2.4 (Brenier map regularity). There is a jointly Borel function $( x , y ) \mapsto q _ { x } ( y )$ such that $Q _ { x } = q _ { x } d y$ for every $x \in S _ { X }$ , (2.7) holds with $p = 2$ , and

$$
0 < q _ { x } ( y ) \leq \overline { { q } } , \qquad x \in S _ { X } , \quad f o r \ a . e . \ y \in \mathcal { V } .\tag{2.11}
$$

When $\tau > 0$ , we also impose $\left( 2 . 8 \right)$ and require $r \geq 2$ if $d _ { y } > 2 _ { z }$ ; when $\tau = 0$ , the Besov indices are ignored. In addition, $\varphi _ { x } \in C ^ { 1 , 1 } ( \mathcal { V } )$ and $\mathrm { L i p } ( \nabla \varphi _ { x } ) \leq \Lambda$ uniformly in $x \in S _ { X }$

Denote the resulting class by $\mathcal { P } _ { \beta , \tau , r , v } ^ { \mathrm { B r } } ( \overline { { q } } , \Lambda )$

The parameters $\beta$ and τ play diferent roles. The former controls the approximation error from pooling nearby covariates, whereas the latter determines the dificulty of estimating a response distribution from a local sample.

2.3. Conditional distribution estimation. We begin with the unconditional distribution estimation problem. Let $a _ { d , \tau } ( m )$ denote the minimax $W _ { 1 }$ rate for estimating a distribution on $[ 0 , 1 ] ^ { d }$ from m observations, with $\tau = 0$ for unrestricted laws and $\tau > 0$ for the Besov class above. When $\tau = 0$ , this notation is only a convention for the unrestricted response model and does not impose a $B _ { r , v } ^ { 0 }$ condition. We use the following expression, extended to real $m \geq 1$ for localization:

$$
a _ { d , \tau } ( m ) = \left\{ \begin{array} { l l } { m ^ { - 1 / 2 } , } & { d = 1 , } \\ { \{ \log ( e m ) / m \} ^ { 1 / 2 } , } & { d = 2 , } \\ { m ^ { - ( \tau + 1 ) / ( 2 \tau + d ) } , } & { d > 2 . } \end{array} \right.\tag{2.12}
$$

The rate is parametric in dimension one, has the critical logarithmic correction in dimension two, and improves with smoothness in the response variable when $d > 2$ . We use this benchmark throughout the conditional analysis.

Except for the critical two-dimensional lower bound, these rates follow from existing results on empirical Wasserstein convergence and smooth density estimation. Theorem 4.1 in Section 4 states the unconditional benchmark and establishes the critical two-dimensional lower bound; Appendix B extends the first-moment bounds to every fixed finite $L ^ { \kappa }$ risk.

Under the uniform covariate distribution, a cell of side length $h$ has expected sample count $n h ^ { d _ { x } }$ . Pooling within the cell incurs H¨older approximation error of order $h ^ { \beta }$ , while Theorem 4.1 gives distribution estimation error $a _ { d _ { y } , \tau } ( n h ^ { d _ { x } } )$ . We define $A _ { n } ( \tau )$ as the common order obtained by balancing

$$
h ^ { \beta } \asymp a _ { d _ { y } , \tau } ( n h ^ { d _ { x } } ) .\tag{2.13}
$$

Writing

$$
\gamma ( \tau ) = \left\{ 2 + \frac { d _ { x } } { \beta } + \frac { ( d _ { y } - 2 ) _ { + } } { \tau + 1 } \right\} ^ { - 1 } ,\tag{2.14}
$$

we have

$$
A _ { n } ( \tau ) = \left\{ \begin{array} { l l } { n ^ { - \gamma ( \tau ) } , } & { d _ { y } \neq 2 , } \\ { \{ \log ( e n ) / n \} ^ { \gamma ( \tau ) } , } & { d _ { y } = 2 . } \end{array} \right.\tag{2.15}
$$

For unrestricted responses, we use these definitions with $\tau = 0$

With an unrestricted covariate distribution, the localization scale remains $A _ { n } ( \tau )$ , but higher moments introduce an additional term.

Theorem 2.5 (Conditional $W _ { 1 }$ minimax rates). Consider the classes defined by Assumptions 2.1 and 2.2, respectively, with fixed model parameters. Fix $1 \leq \kappa < \infty$ . Then, for all suficiently large $n ,$

(2.16)

$$
\mathcal { R } _ { n , 1 , \kappa } ^ { \star } ( \mathcal { P } _ { \beta } ^ { \mathrm { f r e e } } ) \asymp A _ { n } ( 0 ) ^ { \kappa } \vee n ^ { - 1 } ,\tag{2.17}
$$

$$
\begin{array} { r } { \mathcal { R } _ { n , 1 , \kappa } ^ { \star } ( \mathcal { P } _ { \beta , \tau , r , v } ^ { B } ) \asymp A _ { n } ( \tau ) ^ { \kappa } \vee n ^ { - 1 } , \qquad \tau > 0 . } \end{array}
$$

The first term is necessary already under the uniform covariate distribution, whereas a two-point covariate distribution with an atom of mass $n ^ { - 1 }$ forces the second term. For $\kappa = 1$ the additional $n ^ { - 1 }$ term is of lower order and the theorem gives the ordinary integrated $W _ { 1 }$ rate $A _ { n } ( \tau )$

Writing

$$
\kappa _ { c } ( \tau ) = \gamma ( \tau ) ^ { - 1 } = 2 + \frac { d _ { x } } { \beta } + \frac { ( d _ { y } - 2 ) _ { + } } { \tau + 1 } ,\tag{2.18}
$$

the term $A _ { n } ( \tau ) ^ { \kappa }$ dominates the minimax risk for $\kappa < \kappa _ { c } ( \tau )$ , while $n ^ { - 1 }$ dominates for $\kappa > \kappa _ { c } ( \tau )$ The polynomial orders agree at the boundary. When $d _ { y } = 2$ , the logarithmic correction $\mathrm { g i }$ ves $A _ { n } ( \tau ) ^ { \kappa _ { c } } = \log ( e n ) / n$

Table 1  
Unconditional and conditional rates. The response dimension determines the critical behavior, while the covariate dimension enters through localization.
<table><tr><td>Regime</td><td>Unconditional rate  $a _ { d _ { y } , \tau } ( m )$ </td><td>Conditional rate  $A _ { n } ( \tau )$ </td></tr><tr><td> $d _ { y } = 1$ </td><td> $m ^ { - 1 / 2 }$ </td><td> $n ^ { - \beta / ( 2 \beta + d _ { x } ) }$ </td></tr><tr><td> $d _ { y } = 2$ </td><td> $\{ \log ( e m ) / m \} ^ { 1 / 2 }$ </td><td> $\{ \log ( e n ) / n \} ^ { \beta / ( 2 \beta + d _ { x } ) }$ </td></tr><tr><td> $d _ { y } > 2 , \tau = 0$ </td><td>m−  $- 1 / d _ { y }$ </td><td> $n ^ { - \beta / ( \beta d _ { y } + d _ { x } ) }$ </td></tr><tr><td> $d _ { y } > 2 , \tau > 0$ </td><td> $m ^ { - ( \tau + 1 ) / ( 2 \tau + d _ { y } ) }$ </td><td> $_ { \lambda ^ { ' } } { - \beta ( \tau + 1 ) } / [ \beta ( 2 \tau { + } d _ { y } ) { + } d _ { x } ( \tau { + } 1 ) ]$  m</td></tr></table>

Table 1 collects the unconditional and conditional rates. Positive Besov smoothness does not change the leading order when $d _ { y } \leq 2 ;$ for $d _ { y } > 2$ it improves the polynomial rate through the usual gain of one derivative under Wasserstein loss.

Figure 1 illustrates the conditional grid estimator with X uniform on [0, 1] and conditional densities $q _ { x } ( y ) = 1 + 0 . 8 \sin ( 2 \pi x ) ( 2 y _ { 1 } - 1 )$ on $[ 0 , 1 ] ^ { d }$ , where $d = d _ { y } \in \{ 1 , 2 , 4 \}$ . Panel (a) shows the integrated conditional estimation error, which includes localization bias. Panel (b) instead compares the empirical cell distribution with its pooled population law, isolating the sampling error in (2.13). If $K _ { d } ( n )$ is the number of covariate cells, the expected local sample count is $m _ { n , d } = n / K _ { d } ( n )$ . After multiplication by $\sqrt { m _ { n , d } } .$ , the two-dimensional sampling curve increases slowly, while the one-dimensional curve remains approximately constant. These fixed-law simulations do not establish a minimax logarithmic exponent; Section 4 proves the critical lower bound.

![](images/66f7d3ca9d08244c578bd3d01111e731e6cc0d989ae479e79c24f29fba4517a7.jpg)

![](images/f24978fa1bae26fb9cdf7f7bc1640c15e59a94e04c1f9f8169c6a05dea7a6c61.jpg)  
d<sub>y</sub> = 1 d<sub>y</sub> = 2 d<sub>y</sub> = 4 Critical reference

Figure 1. Conditional distribution estimation for $q _ { x } ( y ) = 1 + 0 . 8 \sin ( 2 \pi x ) ( 2 y _ { 1 } - 1 )$ with X uniform on $[ 0 , 1 ]$ . Panel (a) shows mean integrated $W _ { 1 }$ error of the grid empirical estimator in response dimensions $d = d _ { y } \in \{ 1 , 2 , 4 \}$ . Panel $( b )$ shows the mean sampling loss $W _ { 1 } ( \widehat { Q } _ { I } , Q _ { I } )$ for $d = 1 , 2$ , where $Q _ { I }$ is the population law pooled over the query cell. In panel $( b ) _ { i }$ each curve is multiplied by $\sqrt { m _ { n , d } } ,$ with $m _ { n , d } = n / K _ { d } ( n )$ , and normalized at $n = 5 1 2$ . The dotted reference is $\{ \log ( e m _ { n , 2 } ) / \log ( e m _ { 5 1 2 , 2 } ) \} ^ { 1 / 2 }$ , and the horizontal reference is one. Shaded bands are pointwise 95% bootstrap intervals incorporating numerical transport bounds. Appendix G.1 gives the bandwidths, repetition counts, and evaluation procedures.

The localization argument is not specific to $W _ { 1 }$ . For the unconditional class associated with Assumption 2.3, the minimax rate, with $\widehat { Q }$ based on m i.i.d. responses, is

$$
\operatorname* { i n f } _ { \widehat { Q } } \operatorname* { s u p } _ { Q \in \mathcal { Q } _ { d _ { y } , \tau , r , v } ^ { ( p ) } ( \underline { { q } } , \overline { { q } } ) } \left\{ \mathbb { E } _ { Q ^ { \otimes m } } W _ { p } ^ { \kappa } ( \widehat { Q } , Q ) \right\} ^ { 1 / \kappa } \asymp a _ { d _ { y } , \tau } ( m )\tag{2.19}
$$

for all suficiently large m. Appendix E proves (2.19) for every fixed finite κ using a one-sided comparison with a negative Sobolev norm. Appendix B gives the corresponding moment bounds for $W _ { 1 }$

Theorem 2.6 (Conditional $W _ { p }$ minimax rates). Consider the class defined by Assumption 2.3 with fixed model parameters. Fix $1 \leq \kappa < \infty$ . Then, for all suficiently large $n ,$

$$
\begin{array} { r } { \mathcal { R } _ { n , p , \kappa } ^ { \star } \big ( \mathcal { P } _ { \beta , \tau , r , v } ^ { ( p ) } ( \underline { { q } } , \overline { { q } } ) \big ) \asymp A _ { n } ( \tau ) ^ { \kappa } \vee n ^ { - 1 } . } \end{array}\tag{2.20}
$$

In particular, $\kappa = 1$ gives integrated $W _ { p }$ risk of order $A _ { n } ( \tau )$ , while $\kappa = p$ gives integrated $W _ { p } ^ { p }$ risk of order $A _ { n } ( \tau ) ^ { p } \lor n ^ { - 1 }$

The proof uses the localization argument of Appendix D. Mixtures preserve the density and Besov constraints, and convexity of $\boldsymbol { W } _ { p } ^ { p }$ controls the pooling bias. For the lower bound, Proposition $\mathrm { C . 4 ( i i ) }$ controls the $W _ { p }$ distance from each positive density in the hypercube family to the uniform law, and $W _ { p } \geq W _ { 1 }$ preserves the separation.

2.4. Conditional Brenier map estimation. For the map risk (2.6), Assumption 2.4 provides uniform curvature control of the conditional Brenier maps. Lemma F.1 in Appendix F shows

that, uniformly over $x \in S _ { X }$ , it implies

$$
\lambda _ { 0 } I \preceq D ^ { 2 } \varphi _ { x } \preceq \lambda I \quad \mathrm { f o r ~ a . e . ~ } z \in \mathcal { V } , \qquad \lambda _ { 0 } = ( \overline { { q } } \Lambda ^ { d _ { y } - 1 } ) ^ { - 1 } , \qquad q _ { x } ( y ) \geq \Lambda ^ { - d _ { y } } \quad \mathrm { f o r ~ a . e . ~ } y \in \mathcal { V } .
$$

The curvature assumption therefore supplies a uniform density lower bound.

Theorem 2.7 (Conditional Brenier map minimax rate). Consider the class defined by Assumption 2.4 with fixed model parameters. Then, for all suficiently large $n ,$

$$
\operatorname* { i n f } _ { \widehat { T } } \operatorname* { s u p } _ { P \in \mathcal { P } _ { \beta , \tau , r , v } ^ { \mathrm { B r } } ( \overline { { q } } , \Lambda ) } \mathcal { R } _ { n } ^ { \mathrm { m a p } } ( \widehat { T } ; P ) \asymp A _ { n } ( \tau ) ^ { 2 } .\tag{2.21}
$$

Brenier stability converts the $p = \kappa = 2$ distributional rate to squared $L ^ { 2 } ( \rho )$ map error. Since $d _ { x } \geq 1$ gives $\gamma ( \tau ) < 1 / 2$ , the additional $n ^ { - 1 }$ term is of lower order than $A _ { n } ( \tau ) ^ { 2 }$ . Section 5 proves the matching lower bound using inverse potentials.

3. Conditional localization with unrestricted covariate distributions. The key step is to localize an unconditional distribution estimator and weight each local error by the probability of its covariate cell. This mass weighting removes the need for a lower mass condition on $P _ { X }$ for ordinary integrated Wasserstein loss. We first give the fixed grid argument for $W _ { 1 }$ . For higher moments, the concavity argument for a fixed grid is no longer suficient in every regime, and Appendix D replaces it with an adaptive dyadic partition.

3.1. Localization on a fixed grid. Let Π be a finite measurable partition of $\mathcal { X } .$ . For $B \in \Pi$ write

$$
\pi _ { B } = P _ { X } ( B ) , \qquad Q _ { B } = P _ { Y | X \in B }
$$

when $\pi _ { B } > 0$ . On an occupied cell, estimate $Q _ { B }$ from the responses whose covariates fall in $B ;$ if fewer than two observations fall in the cell, use a fixed admissible distribution instead. Introduce the capped rate function

$$
r ( t ) = \left\{ { \begin{array} { l l } { 1 , } & { 0 \leq t < 1 , } \\ { a _ { d _ { y } , \tau } ( t ) , } & { t \geq 1 , } \end{array} } \right.
$$

with $\tau = 0$ for unrestricted responses.

The count in cell $B$ is binomial with mean nπ<sub>B</sub>. Concentration and monotonicity of the unconditional rate imply

$$
\mathbb { E } W _ { 1 } ( \widehat { Q } _ { B } , Q _ { B } ) \lesssim r ( n \pi _ { B } ) .\tag{3.1}
$$

For $x \in B \cap S _ { X }$ , convexity of $W _ { 1 }$ under mixtures together with the conditional H¨older condition gives

$$
W _ { 1 } ( Q _ { B } , Q _ { x } ) \leq \int _ { B } W _ { 1 } ( Q _ { u } , Q _ { x } ) P _ { X } ( d u \mid B ) \lesssim \mathrm { d i a m } ( B ) ^ { \beta } .\tag{3.2}
$$

It follows that, for a regular partition $\Pi _ { h }$ with $M \asymp h ^ { - d _ { x } }$ cells of side length $h ,$

$$
\mathbb { E } \int W _ { 1 } ( \widehat { Q } _ { x } , Q _ { x } ) P _ { X } ( d x ) \lesssim h ^ { \beta } + \sum _ { B \in \Pi _ { h } } \pi _ { B } r ( n \pi _ { B } ) .\tag{3.3}
$$

The corresponding partition inequality, including the treatment of small binomial cell counts, is proved in Proposition D.2.

To control the stochastic term, let $F ( t ) = t r ( t )$ . For each unconditional rate in $( 2 . 1 2 ) , F$ is concave on $[ 0 , \infty )$ . In the power-law cases, $F ( t ) = t$ for $0 \leq t < 1$ and $F ( t ) = t ^ { 1 - \alpha }$ for $t \geq 1$ with $0 < \alpha < 1$ . For $d _ { y } = 2$ , the upper branch is $F ( t ) = \sqrt { t \log ( e t ) }$ , which is also concave. In each case, the branches meet continuously at 1, and the right derivative there is no larger than the left derivative. Jensen’s inequality gives

$$
\begin{array} { r l r } {  { \sum _ { B \in \Pi _ { h } } \pi _ { B } r ( n \pi _ { B } ) = \displaystyle \frac { 1 } { n } \sum _ { B \in \Pi _ { h } } F ( n \pi _ { B } ) } } \\ & { } & { \leq \displaystyle \frac { M } { n } F ( n / M ) = r ( n / M ) . } \end{array}\tag{3.4}
$$

Equations (3.3) and (3.4) imply

$$
\operatorname* { s u p } _ { P } \mathcal { R } _ { n , 1 , 1 } ( \widehat { Q } ; P ) \lesssim h ^ { \beta } + r ( n / M ) ,\tag{3.5}
$$

uniformly over all Borel $P _ { X }$ . At the balancing bandwidth, $n / M \asymp n h ^ { d _ { x } } \to \infty$ , so $r ( n / M ) \asymp$ $a _ { d _ { y } , \tau } ( n h ^ { d _ { x } } )$ . Balancing the two terms gives the upper bound of order $A _ { n } ( \tau )$

3.2. Matching lower bound. For the lower bound it is enough to take $P _ { X }$ uniform. Fix h and choose $M _ { x } \times h ^ { - d _ { x } }$ disjoint smooth covariate bumps with supports of diameter $O ( h )$ . Place an independent copy of the hypercube family from Proposition C.4 in each bump, attenuating the perturbation by the bump height outside its plateau. With local sample size $m \asymp n h ^ { d _ { x } }$ assume

$$
a _ { d _ { y } , \tau } ( m ) \lesssim h ^ { \beta } .
$$

The attenuation estimate then enforces the conditional $W _ { 1 }$ H¨older condition. The Hellinger bound for a change in one sign keeps the total variation distance between neighboring n-sample laws bounded away from one. Summing over the covariate blocks produces integrated $W _ { 1 }$ separation of order $a _ { d _ { y } , \tau } ( m )$ . At the balancing bandwidth (2.13), this gives

$$
\begin{array} { r } { \mathcal { R } _ { n , 1 , 1 } ^ { \star } \gtrsim A _ { n } ( \tau ) . } \end{array}
$$

Proposition D.3 gives the full replicated hypercube argument, completing the proof of Theorem 2.5 for $\kappa = 1$

3.3. Higher moments and rare covariate regions. For fixed finite $\kappa ,$ the local $W _ { 1 } ^ { \kappa }$ risk is of order $a _ { d _ { y } , \tau } ( m ) ^ { \kappa }$ in the two $W _ { 1 }$ classes. Under Assumption 2.3, the local $W _ { p } ^ { \kappa }$ risk has the same order. The dificulty is therefore not the local estimation problem but the aggregation of these errors over cells with very diferent probabilities under $P _ { X }$ . The fixed grid partition inequality remains valid, but the Jensen argument used above is not sharp for all $\kappa .$

For $p > 1$ , cellwise mixtures preserve the density bounds and the Besov condition. If $x \in B$ joint convexity of $\boldsymbol { W } _ { p } ^ { p }$ and (2.7) give

$$
W _ { p } ( Q _ { B } , Q _ { x } ) ^ { p } \leq \int _ { B } W _ { p } ( Q _ { u } , Q _ { x } ) ^ { p } P _ { X } ( d u \mid B ) \lesssim \mathrm { d i a m } ( B ) ^ { \beta p } ,
$$

so the pooling bias is again of order diam $( B ) ^ { \beta }$ . The corresponding statement for $W _ { 1 }$ follows directly from mixture convexity.

The proof of Proposition D.4 uses sample splitting and an adaptive dyadic partition. The partition adapts to local sample occupancy, with the regularity parameters $\beta$ and τ fixed and known. The first subsample, of size $n _ { 0 } \asymp n .$ , accepts a level-j cell when its count is at least $T _ { j }$ . The geometrically growing thresholds in Appendix D.3 satisfy $a _ { d _ { y } , \tau } ( T _ { j } ) \lesssim 2 ^ { - j \beta }$ . A cell B is misclassified if it is accepted despite $n _ { 0 } P _ { X } ( B ) < T _ { j } / 2$ , or rejected despite $n _ { 0 } P _ { X } ( B ) > 2 T _ { j }$ Chernof bounds, weighted by cell probabilities and summed over the $2 ^ { j d _ { x } }$ cells at each level, give

$$
\mathbb { E } \sum _ { \substack { B \mathrm { ~ m i s c l a s s i f i e d } } } P _ { X } ( B ) \lesssim \frac { 1 } { n } \sum _ { j \geq 0 } 2 ^ { j d _ { x } } T _ { j } e ^ { - c T _ { j } } \lesssim n ^ { - 1 } .
$$

The first rejected cells form a disjoint terminal partition, and estimation uses their accepted parents. A terminal cell is treated as bad when it or its parent is misclassified. Misclassified terminal cells are charged to their own masses. The terminal children assigned to any misclassified parent have total probability at most that parent’s probability. Thus the expected mass of all bad terminal cells is $O ( n ^ { - 1 } )$ . Since Wasserstein loss is bounded on the response cube, their risk contribution has the same order.

On the remaining terminal cells, independence of the second subsample and the local moment bounds control distribution estimation error, while the conditional H¨older condition controls pooling bias. Write $\mathcal { L } _ { \mathrm { g o o d } }$ for these cells and $h _ { L } = 2 ^ { - j ( L ) }$ for their side lengths. Both error terms have conditional κth moments of order at most $h _ { L } ^ { \beta \kappa }$ given the first subsample. Correct rejection also gives $P _ { X } ( L ) \lesssim T _ { j ( L ) } / n$

The thresholds in Appendix D.3 satisfy

$$
2 ^ { j d _ { x } } T _ { j } \asymp 2 ^ { j \beta \kappa _ { c } ( \tau ) } ( 1 + j ) ^ { 1 \{ d _ { y } = 2 \} } .
$$

Choose $j _ { * }$ so that $2 ^ { j _ { * } d _ { x } } T _ { j _ { * } } \asymp n$ and put $h _ { * } = 2 ^ { - j _ { * } }$ . Then $h _ { * } ^ { \beta } \asymp A _ { n } ( \tau )$ . Good terminal cells finer than $h _ { * }$ have total probability at most one, so their contribution is at most $C h _ { * } ^ { \beta \kappa }$ . For the coarser cells,

$$
\begin{array} { r l r } {  { \sum _ { L \in \mathcal { L } _ { \mathrm { g o o d } } } P _ { X } ( L ) h _ { L } ^ { \beta \kappa } \lesssim \frac { 1 } { n } \sum _ { \stackrel { L \in \mathcal { L } _ { \mathrm { g o o d } } } { j ( L ) \leq j _ { * } } } | L | \cdot 2 ^ { j ( L ) ( d _ { x } - \beta \kappa ) } T _ { j ( L ) } } } \\ & { } & { \leq \frac { 1 } { n } \underset { 0 \leq j \leq j _ { * } } { \sum } 2 ^ { j ( d _ { x } - \beta \kappa ) } T _ { j } \lesssim n ^ { - 1 } + h _ { * } ^ { \beta \kappa } , } \end{array}
$$

where $| L | = h _ { L } ^ { d _ { x } }$ and disjointness gives $\begin{array} { r } { \sum _ { L } | L | \leq 1 . \ \mathrm { I f } \ \kappa \leq \kappa _ { c } ( \tau ) } \end{array}$ , the maximum is bounded by a constant times its value at $j _ { * } \mathrm { ; ~ i f ~ } \kappa > \kappa _ { c } ( \tau )$ , exponential decay makes it bounded independently of n. At $\kappa = \kappa _ { c } ( \tau )$ , the endpoint bound retains the logarithmic factor when $d _ { y } = 2$ . Together with the expected bad-cell mass of order $n ^ { - 1 }$ , this gives, uniformly over all Borel $P _ { X }$

$$
\operatorname* { s u p } _ { P } \mathcal { R } _ { n , p , \kappa } ( \widehat { Q } ; P ) \lesssim A _ { n } ( \tau ) ^ { \kappa } + n ^ { - 1 } ,\tag{3.6}
$$

with $p = 1$ in the unrestricted and $W _ { 1 }$ Besov classes.

The two terms arise from diferent lower bounds. Under the uniform covariate distribution, replication of the distribution family gives $A _ { n } ( \tau ) ^ { \kappa }$ . For $p > 1$ , the positive family remains inside the prescribed density bounds and $W _ { p } \geq W _ { 1 }$ preserves its separation. The second term comes from a two-point covariate law with an atom of mass $n ^ { - 1 }$ carrying two separated distribution alternatives. With probability bounded away from zero that atom is not observed, forcing risk of order $n ^ { - 1 }$ . Section D.3 gives both constructions.

3.4. Numerical illustration: rare covariate events. To illustrate the efect of rare covariates on higher moments in Theorem 2.5, we take $d _ { x } = d _ { y } = 1$ and consider

$$
q _ { x , s } ( y ) = 1 + \alpha s x ( 2 y - 1 ) , \qquad x , y \in [ 0 , 1 ] , \quad s \in \{ - 1 , 1 \} , \quad \alpha = 0 . 6 5 .\tag{3.7}
$$

The conditional densities are bounded between 0.35 and 1.65, and the corresponding laws are Lipschitz in x under $W _ { 1 }$ . We compare uniform covariates with

$$
P _ { X , n } = ( 1 - n ^ { - 1 } ) \delta _ { 0 } + n ^ { - 1 } \delta _ { 1 } .
$$

Only the covariate distribution varies with n, as permitted by the minimax supremum over unrestricted $P _ { X }$ . Here $\beta = 1$ , so $A _ { n } = n ^ { - 1 / 3 }$ and $\kappa _ { c } = 3$ . We report the integrated risks for $\kappa = 1$ and $\kappa = 4$

We compare the $\mathrm { g r i d }$ , occupancy, and empirical nearest neighbor estimators with a conditional normalizing flow using a monotone rational-quadratic spline [21]. The flow is fitted by likelihood and evaluated under the same Wasserstein loss. We use $n = { \bar { 2 } } ^ { 8 } , \ldots , 2 ^ { 1 5 }$ and average the risks over the two signs in (3.7). Appendix G.2 gives the estimator definitions, tuning parameters, and simulation procedures.

Over the sample sizes in Figure 2, the fourth-moment risks decrease more slowly under rare covariates than under uniform covariates. The gray segments show the first-moment benchmark $n ^ { - 1 / 3 }$ and, for the fourth moment, the uniform covariate benchmark $n ^ { - 4 / 3 }$ and the rare atom scale $n ^ { - 1 }$ . These powers describe minimax benchmarks over the full model classes; individual estimators can behave diferently on the fixed family in (3.7).

A $\mathrm { ~ t ~ } n = 2 ^ { 1 5 }$ under rare covariates, most fourth-moment error of the occupancy estimator arises when the rare atom is observed but only sparsely. The missed event establishes the lower bound, while the measured risk also includes this estimation error. This distinction also holds for the normalizing flow; Appendix G.2 gives the risk decomposition and corresponding contributions.

4. Unconditional Wasserstein estimation and multiscale lower bounds. The conditional lower bound uses an unconditional family of distributions that can be attenuated across covariate neighborhoods without changing its rate. Appendix B gives the unconditional upper bounds and their extension to fixed finite moments, and Appendix C constructs the family used for the lower bound. A single multiscale potential construction gives the rates for $d = 1$ ， $d = 2$ , and $d > 2$

Theorem 4.1 (Unconditional $W _ { 1 }$ minimax rates). Let $d \geq 1$ and fix a finite $1 \leq \kappa < \infty$ . In the smooth case, also $\mathit { f i x } \tau > 0$ and $1 \leq r , v \leq \infty$ . Then, for all suficiently large m,

$$
\operatorname* { i n f } _ { \widehat { Q } } \operatorname* { s u p } _ { Q \in \mathcal { Q } _ { d } ^ { \mathrm { f r e e } } } \mathbb { E } W _ { 1 } ^ { \kappa } ( \widehat { Q } , Q ) \asymp a _ { d , 0 } ( m ) ^ { \kappa } ,\tag{4.1}
$$

![](images/6dca1fed3fc1c5f04a25f1f870a509a5e10c5d06d029404e78b6a09869c29c0e.jpg)

![](images/ee6be1a1f2112992930516f15993cc0b4fa0463cf05e6495288ece8506a451b9.jpg)

(c) Uniform, κ = 4  
![](images/4ac77ea63eca68725ddbb6240c9e5b279300694d40f274c9adda32167c6aaa1f.jpg)

(d) Rare atom, κ = 4  
![](images/af100656e77d56f7a40e9497232fd0eb1c5e746657aaf265f75a1136969c33ef.jpg)  
Grid Occupancy kNN Normalizing flow

Figure 2. Conditional distribution learning for (3.7). The $l e f t$ and right columns use uniform and rare covariates, respectively. Panels $( a )$ and $( b )$ show mean integrated $W _ { 1 }$ loss, and panels $( c )$ and $( d )$ show mean integrated $W _ { 1 } ^ { 4 }$ loss, averaged over the two signs. All axes are logarithmic, with common vertical limits within each row. Gray dashed segments indicate reference powers with arbitrary vertical scaling. Shaded bands are pointwise 95% bootstrap intervals over independent training datasets, with stratified resampling for rare covariates. Appendix G.2 gives the repetition counts and integration procedures.

$$
\operatorname* { i n f } _ { \widehat { Q } } \operatorname* { s u p } _ { Q \in \mathcal Q _ { d , \tau , r , v } ^ { B } } \mathbb E W _ { 1 } ^ { \kappa } ( \widehat { Q } , Q ) \asymp a _ { d , \tau } ( m ) ^ { \kappa } , \qquad \tau > 0 .\tag{4.2}
$$

Here and below, constants are allowed to depend on the fixed value of $\kappa ,$ but not on $m$

For the upper bound, the empirical measure attains the unrestricted rate in every dimension and the smooth rate when $d \leq 2$ . When $d > 2$ and $\tau > 0$ , we truncate the empirical wavelet expansion at level $J$ and apply an approximate projection in the KR norm onto probability laws. The resulting estimator satisfies

$$
\begin{array} { r } { \mathbb { E } W _ { 1 } ( \widehat { Q } _ { m } , Q ) \lesssim 2 ^ { - J ( \tau + 1 ) } + m ^ { - 1 / 2 } 2 ^ { J ( d / 2 - 1 ) } . } \end{array}
$$

The first term is the approximation bias, with the gain of one derivative coming from the Lipschitz test functions in KR duality; the second is the stochastic error. Choosing $2 ^ { \bar { J ( 2 \tau + d ) } } \asymp m$ gives the rate $a _ { d , \tau } ( m )$ . No pointwise density bounds are used. Appendix B proves these estimates and extends the upper bounds to every fixed finite moment.

The lower bound requires response laws that are dificult to distinguish statistically but well separated under Wasserstein loss. We combine perturbations across resolutions to accumulate separation while keeping neighboring alternatives dificult to distinguish. We use the multiscale potential

$$
U _ { \theta } = \varepsilon \sum _ { j = j _ { 0 } } ^ { J } \sum _ { k \in K _ { j } } \theta _ { j , k } 2 ^ { - 2 j } \psi _ { j , k } , \qquad q _ { \theta } = 1 - \Delta U _ { \theta } , \qquad \varepsilon = c m ^ { - 1 / 2 } .\tag{4.3}
$$

Here $| \mathcal { K } _ { j } | \asymp 2 ^ { j d }$ and the retained wavelets are supported in a fixed compact subset of $( 0 , 1 ) ^ { d } .$ . The factor $2 ^ { - 2 j }$ compensates for the two derivatives in the Laplacian. Thus the density perturbations have comparable testing cost across scales. At resolution j, each wavelet perturbation in the potential has gradient $L ^ { 2 }$ norm of order $\varepsilon { 2 ^ { - j } }$ . With $\tau = 0$ for the unrestricted branch, the upper resolution J is chosen maximal subject to

$$
\varepsilon 2 ^ { J ( \tau + d / 2 ) } \leq c _ { 0 } .
$$

Appendix C shows that this choice ensures positivity, the Besov constraint when $\tau > 0$ , and the one-coordinate testing bound. The dimensional transition is encoded by

$$
B _ { J } = \sum _ { j = j _ { 0 } } ^ { J } 2 ^ { j ( d - 2 ) } \asymp \left\{ \begin{array} { l l } { 1 , } & { d = 1 , } \\ { J , } & { d = 2 , } \\ { 2 ^ { J ( d - 2 ) } , } & { d > 2 . } \end{array} \right.
$$

This sum determines the rate in each response dimension.

It remains to convert the energy of the multiscale potential into Wasserstein separation. The required estimate is a lower bound for the Kantorovich–Rubinstein norm.

Proposition 4.2 (Potential lower bound for the KR norm). Let d $\geq 1$ and $U \in C _ { c } ^ { 2 } ( ( 0 , 1 ) ^ { d } )$ Set $g = - \Delta U$ . Then

$$
\| g d y \| _ { \mathrm { K R } } \geq c _ { d } \frac { \| \nabla U \| _ { L ^ { 2 } } ^ { 3 } } { \| \nabla U \| _ { L ^ { 4 } } ^ { 2 } } ,\tag{4.4}
$$

with the right-hand side understood to be zero if $g = 0$

The proof is based on quantitative Sobolev Lipschitz truncation [1, 17, 26]. A related argument for the empirical matching lower bound appears in [9, Section 7]. Extend U by zero and truncate it at gradient level λ. If $E _ { 2 } = \| \nabla U \| _ { 2 } ^ { 2 }$ and $E _ { 4 } = \| \nabla U \| _ { 4 } ^ { 4 }$ , the truncation gives a Lipschitz function $\phi _ { \lambda }$ satisfying

$$
\int ( - \Delta U ) \phi _ { \lambda } \ge E _ { 2 } - C E _ { 4 } \lambda ^ { - 2 } , \qquad \mathrm { L i p } ( \phi _ { \lambda } ) \lesssim \lambda .
$$

Choose $\lambda ^ { 2 } \asymp E _ { 4 } / E _ { 2 }$ and rescale the test function to have Lipschitz constant one. This establishes (4.4); the full proof is given in Appendix C.

For two vertices $\theta , \theta ^ { \prime }$ , set $U = U _ { \theta } - U _ { \theta ^ { \prime } }$ and let $H _ { j }$ be the number of difering signs at resolution $j .$ . Define

$$
S = \sum _ { j = j _ { 0 } } ^ { J } 2 ^ { - 2 j } H _ { j } .\tag{4.5}
$$

The square function bounds give

$$
\| \nabla U \| _ { 2 } ^ { 2 } \gtrsim \varepsilon ^ { 2 } S , \qquad \| \nabla U \| _ { 4 } ^ { 4 } \lesssim \varepsilon ^ { 4 } B _ { J } S .
$$

Applying Proposition 4.2 now gives

$$
W _ { 1 } ( Q _ { \theta } , Q _ { \theta ^ { \prime } } ) \gtrsim \varepsilon S / \sqrt { B _ { J } } .\tag{4.6}
$$

Coordinatewise testing yields $\operatorname { \mathbb { E } } S \gtrsim B _ { J }$ . Since $2 ^ { J } \asymp m ^ { 1 / ( 2 \tau + d ) }$ , with $\tau = 0$ in the unrestricted branch,

$$
\varepsilon \sqrt { B _ { J } } \asymp \left\{ \begin{array} { l l } { m ^ { - 1 / 2 } , } & { d = 1 , } \\ { \{ \log ( e m ) / m \} ^ { 1 / 2 } , } & { d = 2 , } \\ { m ^ { - ( \tau + 1 ) / ( 2 \tau + d ) } , } & { d > 2 , } \end{array} \right.\tag{4.7}
$$

which is of order $a _ { d , \tau } ( m )$ . The testing argument therefore proves the first-moment lower bounds in Theorem 4.1. Jensen’s inequality gives the corresponding lower bounds for every fixed finite $\kappa .$ Appendix C also verifies the attenuation estimates needed for the conditional lower bound.

The critical case is $d = 2$ . Here each active resolution contributes at the same order to $B _ { J }$ so $B _ { J } \asymp J \asymp \log ( e m )$ and the accumulation across scales produces the factor $\sqrt { \log m }$ . A family supported on one resolution yields only the parametric separation $m ^ { - 1 / 2 }$ . Besov smoothness limits the finest scale but not this accumulation.

Figure 3 illustrates this accumulation using two-dimensional cosine densities, a numerical analogue of the interior-wavelet construction above. Combining levels gives greater mean transport separation than using the finest level alone at $J = 2 , 3 , 4 , 5$ in panel (a). Panel (b) uses the corresponding $\sqrt { J }$ normalization. These five resolutions, with varying amplitudes, are insuficient to determine the asymptotic scaling. Appendix G.3 gives the construction and numerical transport bounds.

5. Conditional Brenier map estimation. Theorem 2.7 follows from stability for the upper bound and an inverse potential construction for the lower bound.

5.1. Stability and the minimax upper bound. Lemma F.1 specializes the fixed-source stability bound in [4, Corollary 1] to Assumption 2.4. Uniformly over $x \in S _ { X }$ , for every probability law $\widetilde { Q }$ on $\mathcal { V }$

$$
\| T _ { \rho \to \widetilde { Q } } - T _ { x } \| _ { L ^ { 2 } ( \rho ) } ^ { 2 } \le \frac { \Lambda } { \lambda _ { 0 } } W _ { 2 } ^ { 2 } ( Q _ { x } , \widetilde { Q } ) ,\tag{5.1}
$$

where $\lambda _ { 0 } = ( \overline { { q } } \Lambda ^ { d _ { y } - 1 } ) ^ { - 1 }$ . The corresponding minimax comparison is made explicit in [3].

![](images/894b577bdf71e64fcb2d881281df927065d6135b7ecc26dc1cdabb61104bff97.jpg)

![](images/df8d69454b55ce3012f95f44dee5aa8d3b31bc4ed3fb0c1cdced4121a3e1e798.jpg)  
Levels 1, …,J Level J only Numerical bounds  
Figure 3. Transport separation $f o r$ cosine densities using multiple levels or the finest level alone, with amplitude $\varepsilon _ { J } = 0 . 0 2 5 2 \AA ^ { - 2 J }$ . Panels $( a )$ and (b) show $W _ { 1 } / \varepsilon _ { J }$ and $W _ { 1 } / ( \varepsilon _ { J } \sqrt { J } )$ . Markers and lines show means of numerical interval midpoints; black segments give mean lower and upper bounds. Shaded bands incorporate these bounds and pointwise 95% bootstrap uncertainty over 32 independent pairs of sign realizations at each resolution. Appendix G.3 gives the transport computations and numerical bounds.

Apply Theorem 2.6 with $p = \kappa = 2$ and density bounds $\underline { { q } } = \Lambda ^ { - d _ { y } }$ and q. Assumption 2.4 and the density lower bound in Section 2.4 place the Brenier class inside the corresponding distribution class. There is therefore a measurable distribution estimator $\widehat { Q }$ such that, for the plugin map $\widehat { T } _ { x } = T _ { \rho  \widehat { Q } _ { x } }$ , Lemma F.1 gives

$$
\operatorname* { s u p } _ { P \in \mathcal { P } _ { \beta , \tau , r , v } ^ { \mathrm { B r } } ( \overline { { q } } , \Lambda ) } \mathcal { R } _ { n } ^ { \mathrm { m a p } } ( \widehat { T } ; P ) \lesssim A _ { n } ( \tau ) ^ { 2 } + n ^ { - 1 } \asymp A _ { n } ( \tau ) ^ { 2 } .\tag{5.2}
$$

The final comparison follows from $\gamma ( \tau ) < 1 / 2$ , since $d _ { x } \geq 1$ . Only the true conditional map needs the curvature bounds in the stability lemma; the estimated law $\widehat { Q } _ { x }$ need not lie in the Brenier class and may be discrete. Appendix F records measurability of the plugin estimator. This proves the upper bound in Theorem 2.7 for the full class.

5.2. Lower bound from inverse potentials. For the lower bound, we use the multiscale potentials $U _ { \theta }$ and the spectral sum $B _ { J }$ from Section 4, with $d = d _ { y }$ , but parameterize the inverse map. Define

$$
S _ { U } = \mathrm { I d } + \nabla U , \qquad T _ { U } = S _ { U } ^ { - 1 } , \qquad q _ { U } = \mathrm { d e t } ( I + D ^ { 2 } U ) .
$$

The target density is then an explicit determinant, avoiding composition with an inverse map. For suficiently small Hessian norm, Appendix F.2 shows that $T _ { U }$ is a Brenier map in the class of Assumption 2.4, the squared Hellinger distance between neighboring distributions is $O ( \varepsilon ^ { 2 } )$ and

$$
\lVert \boldsymbol { T } _ { U } - \boldsymbol { T } _ { V } \rVert _ { L ^ { 2 } ( \rho ) } ^ { 2 } \asymp \lVert \nabla U - \nabla V \rVert _ { 2 } ^ { 2 } .
$$

Embedding independent copies of this family in covariate neighborhoods of diameter h gives the three constraints

$$
n h ^ { d _ { \boldsymbol { x } } } \varepsilon ^ { 2 } \stackrel { < } { \sim } 1 , \qquad \varepsilon 2 ^ { J ( \tau + d _ { \boldsymbol { y } } / 2 ) } \stackrel { < } { \sim } 1 , \qquad \varepsilon \sqrt { B _ { J } } \stackrel { < } { \sim } h ^ { \beta } .\tag{5.3}
$$

They correspond, respectively, to testing, smoothness in the response variable, and the conditional $W _ { 2 }$ H¨older condition. Taking $\varepsilon \asymp ( n h ^ { { d _ { x } } } ) ^ { - 1 / 2 }$ and the largest admissible J gives

$$
\varepsilon \sqrt { B _ { J } } \asymp a _ { d _ { y } , \tau } ( n h ^ { d _ { x } } ) .
$$

Balancing as in (2.13) gives $h ^ { \beta } \asymp A _ { n } ( \tau )$ and squared $L ^ { 2 } ( \rho )$ map separation $\varepsilon ^ { 2 } B _ { J } \asymp A _ { n } ( \tau ) ^ { 2 }$ Appendix F.2 verifies the determinant identity, class membership, separation, and testing bounds, completing the lower bound in Theorem 2.7.

6. Discussion. The minimax results separate response distribution estimation from localization in the covariate. Weighting local errors by cell probabilities preserves the ordinary integrated Wasserstein rate $A _ { n } ( \tau )$ uniformly over unrestricted covariate distributions. Higher moments can instead be dominated by rare unobserved regions, which contribute the additional $n ^ { - 1 }$ term to the integrated $W _ { p } ^ { \kappa }$ risk. This limitation applies to conditional generators even when the response laws are simple.

For comparison, a covariate density bounded below by a fixed $c _ { X } \in ( 0 , 1 ]$ removes the rare region penalty. In the distribution classes considered here, the minimax integrated $W _ { p } ^ { \kappa }$ risk is then of order $A _ { n } ( \tau ) ^ { \kappa }$ for every fixed finite κ. Regular grid cells have probability at least $c _ { X } h ^ { d _ { x } }$ , so Proposition D.2 gives the upper bound by balancing localization bias against error at sample scale $n h ^ { d _ { x } }$ . The lower bound under uniform covariates remains applicable.

Within the Brenier class, uniform curvature bounds convert squared $W _ { 2 }$ estimation error into squared $L ^ { 2 } ( \rho )$ map error of order $A _ { n } ( \tau ) ^ { 2 }$ . For smooth classes with $d _ { y } > 2$ , the sharp upper bound uses a wavelet distribution estimator and an approximate metric projection; computational complexity bounds for this construction remain open. Attaining these minimax rates with computationally eficient parameterized generators remains a separate problem.

Further statistical questions concern adaptation to unknown $\beta$ and $\tau$ and extension to unbounded response spaces under suitable tail conditions. The density bounds for $p > 1$ and the curvature assumptions for Brenier maps are suficient conditions. It remains open how far these conditions can be relaxed without changing the rates.

Declarations. The authors used OpenAI’s GPT-5.6 and GPT-6 to assist with proof development, literature searches, numerical experiments, and manuscript editing. The authors directed the research, independently verified all AI-assisted work, and determined the final formulation and presentation of the results. The authors take full responsibility for all content.

Acknowledgments. This work of Kim was supported by the National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT) (RS-2026-25598375).

Appendix A. Preliminaries. We fix the wavelet conventions used below and record the transport comparison and measurable projection used in the upper bounds. Implicit constants may depend on the fixed structural parameters and wavelet systems, but not on the sample size, resolution level, cell mass, or hypercube signs.

For each fixed $\tau \geq 0$ , choose an integer $K > \tau + 2$ and fix a compactly supported tensor-product Cohen–Daubechies–Vial orthonormal boundary wavelet basis of $L ^ { 2 } ( [ 0 , 1 ] ^ { d } )$ whose regularity and number of vanishing moments exceed K. We use the basis without imposed boundary conditions: the scaling spaces reproduce afine functions, and every detail wavelet, including those at the boundary, has zero moments against afine functions; see [15, Theorem 4.4] and [14, Section 2.12]. For the multiscale lower-bound construction, we retain only interior wavelets supported in a fixed compact subset of $( 0 , 1 ) ^ { d }$ ; zero extension then permits the standard positive-order square function estimates.

Let Φ denote the finite scaling family and $\Psi _ { j } , j \geq j _ { 0 }$ , the detail families on $[ 0 , 1 ] ^ { d }$ . If $\alpha _ { f }$ and $\beta _ { f , j }$ are the associated coeficients, then the Besov norm is equivalent to

$$
\lVert \alpha _ { f } \rVert _ { \ell _ { r } } + \left. \left( 2 ^ { j \{ \tau + d ( 1 / 2 - 1 / r ) \} } \lVert \beta _ { f , j } \rVert _ { \ell _ { r } } \right) _ { j \ge j _ { 0 } } \right. _ { \ell _ { v } } .\tag{A.1}
$$

Write $\Omega _ { d } = [ 0 , 1 ] ^ { d }$ . For a mean-zero function $u \in L ^ { p } ( \Omega _ { d } ) , 1 < p < \infty$ , set

$$
\| u \| _ { - 1 , p } = \operatorname* { s u p } \left\{ \left| \int _ { \Omega _ { d } } u \phi \right| : \phi \in W ^ { 1 , p ^ { \prime } } ( \Omega _ { d } ) , \ \| \nabla \phi \| _ { L ^ { p ^ { \prime } } } \leq 1 \right\} , \qquad { \frac { 1 } { p } } + { \frac { 1 } { p ^ { \prime } } } = 1 .\tag{A.2}
$$

Constants are immaterial because u has zero mean.

Lemma A.1 (Flux representation). For every mean-zero $\boldsymbol { \mathscr { x } } \in L ^ { p } ( \Omega _ { d } )$

$$
\| u \| _ { - 1 , p } = \operatorname* { m i n } \left\{ \| V \| _ { L ^ { p } } : \nabla \cdot V = u \ i n \ d i s t r i b u t i o n s \ a n d V \cdot n = 0 \ o n \ \partial \Omega _ { d } \right\} .
$$

Proof. Equip $W ^ { 1 , p ^ { \prime } } ( \Omega _ { d } ) / \mathbb { R }$ with the norm $\| \nabla \phi \| _ { p ^ { \prime } }$ . The functional $\left[ \phi \right] \mapsto - \int u \phi$ has norm $\| u \| _ { - 1 , p } .$ . Hahn–Banach and $L ^ { p } { - } L ^ { p ^ { \prime } }$ duality give a vector field V with $\| V \| _ { p } = \| u \| _ { - 1 , p }$ and

$$
\int \boldsymbol { V } \cdot \nabla \phi = - \int \boldsymbol { u } \phi , \qquad \phi \in W ^ { 1 , p ^ { \prime } } ( \Omega _ { d } ) .
$$

This is the weak formulation of $\nabla \cdot V = u$ with zero normal trace. The converse follows from H¨older’s inequality. ■

Proposition A.2 (One-sided transport comparison). Let $1 < p <$ ∞ and let $f , q \in L ^ { p } ( \Omega _ { d } )$ be probability densities with $q \geq { \underline { { q } } } > 0$ almost everywhere. Then

$$
\begin{array} { r } { W _ { p } ( f d y , q d y ) \leq p \underline { { q } } ^ { - ( p - 1 ) / p } \| f - q \| _ { - 1 , p } . } \end{array}\tag{A.3}
$$

Proof. Apply Lemma A.1 to $u = f - q$ and choose V with $\| V \| _ { p } = \| f - q \| _ { - 1 , p } .$ . Set

$$
\begin{array} { r } { \rho _ { t } = ( 1 - t ^ { p } ) f + t ^ { p } q , \qquad j _ { t } = p t ^ { p - 1 } V , \qquad 0 \le t \le 1 . } \end{array}
$$

Then $\partial _ { t } \rho _ { t } + \nabla \cdot \boldsymbol { j } _ { t } = 0$ and $\rho _ { t } \geq t ^ { p } q$ . The dynamic formulation of $W _ { p }$ [6, 45] gives

$$
W _ { p } ^ { p } ( f d y , q d y ) \leq p ^ { p } \int _ { \Omega _ { d } } | V | ^ { p } q ^ { 1 - p } \leq p ^ { p } \underline { { q } } ^ { 1 - p } \| V \| _ { p } ^ { p } .
$$

Taking the pth root proves (A.3). This is a one-sided comparison between $W _ { p }$ and a negative Sobolev norm under a density lower bound; for the quadratic case see also [40], and for finite Wasserstein orders see [37].

We also use the following measurable approximate projection. Let d be induced by a norm on diferences in an afine separable space, let C be the admissible set, and fix a countable d-dense family $\{ f _ { 1 } , f _ { 2 } , \ldots \} \subset { \mathcal { C } }$ . Given $\widetilde { f }$ and $\delta > 0$ , select the first index i such that

$$
d ( \widetilde f , f _ { i } ) \leq \operatorname* { i n f } _ { j } d ( \widetilde f , f _ { j } ) + \delta .
$$

Whenever the distances are measurable, this rule is measurable and, for every $f \in { \mathcal { C } }$

$$
d ( \widehat { f } , f ) \leq 2 d ( \widetilde { f } , f ) + \delta .\tag{A.4}
$$

We take $\delta = m ^ { - 1 }$ . For the $W _ { 1 }$ estimator, d is the KR norm and rational finitely supported probability laws provide a countable dense family. For the $W _ { p }$ estimator, d is the Neumann negative norm $\| \cdot \| _ { - 1 , p }$ on the separable set of $L ^ { p }$ probability densities. In each case the preliminary estimator depends on finitely many empirical wavelet coeficients, so its distance to each fixed candidate is Borel. The projected $W _ { p }$ estimator need not satisfy the density bounds imposed on the true law; those bounds are used only in the one-sided comparison between Wasserstein distance and a negative Sobolev norm and the variance estimate. The projection defines a measurable statistical estimator.

Appendix B. Unconditional upper bounds in $W _ { 1 }$ . The unconditional upper bounds in Theorem 4.1 use the empirical measure for unrestricted responses and for smooth classes in dimensions one and two. In the smooth high-dimensional case, a truncated wavelet estimator suppresses fine-scale stochastic error and gains one derivative under the Kantorovich–Rubinstein metric.

Proposition B.1 (Unconditional upper bound). Fix $1 \leq \kappa < \infty$ . Let m $\geq 1$ be an integer and let $Y _ { 1 } , \dots , Y _ { m }$ be $i . i . d .$ with law $Q$ on $[ 0 , 1 ] ^ { d }$ . In the unrestricted model, there is an estimator satisfying

$$
\left\{ \mathbb { E } W _ { 1 } ^ { \kappa } ( \widehat { Q } _ { m } , Q ) \right\} ^ { 1 / \kappa } \lesssim _ { \kappa } a _ { d , 0 } ( m ) .
$$

If instead $Q \in \mathcal { Q } _ { d , \tau , r , v } ^ { B }$ , there is an estimator such that

$$
\bigl \{ \mathbb { E } W _ { 1 } ^ { \kappa } ( \widehat { Q } _ { m } , Q ) \bigr \} ^ { 1 / \kappa } \lesssim _ { \kappa } a _ { d , \tau } ( m ) .
$$

In dimensions $d \leq 2$ , the empirical distribution itself attains this rate on the smooth class.

The smooth high-dimensional argument uses the following negative-order wavelet estimate.

Lemma B.2 (Endpoint wavelet upper bound). Let w be a mean-zero integrable function on $[ 0 , 1 ] ^ { d }$ , with coeficients in the boundary wavelet basis $( \alpha _ { w } , \beta _ { w , \psi } )$ . Then

$$
\| w d y \| _ { \mathrm { K R } } \leq C \left\{ \| \alpha _ { w } \| _ { \ell ^ { 1 } } + \sum _ { j \geq j _ { 0 } } 2 ^ { - j ( 1 + d / 2 ) } \| \beta _ { w , j } \| _ { \ell ^ { 1 } } \right\} .\tag{B.1}
$$

Proof. Because $w$ has mean zero, a 1-Lipschitz test function $\varphi$ may be normalized by $\varphi ( y _ { 0 } ) = 0$ . The finite scaling contribution is then uniformly bounded. For a detail wavelet $\psi$ at level $j ,$ choose $y _ { \psi } \in \operatorname { s u p p } ( \psi )$ . Using $\textstyle \int \psi = 0$

$$
| \langle \varphi , \psi \rangle | = \left| \int ( \varphi ( y ) - \varphi ( y _ { \psi } ) ) \psi ( y ) d y \right| \leq C 2 ^ { - j } \| \psi \| _ { 1 } \leq C 2 ^ { - j ( 1 + d / 2 ) } .
$$

Applying this coeficient estimate to finite projections onto the boundary wavelet basis of $\varphi$ and passing to the uniform limit proves (B.1). ■

Proof of Proposition B.1. We first prove the assertions for $\kappa = 1$ . For the unrestricted branch, let $\begin{array} { r } { Q _ { m } = m ^ { - 1 } \sum _ { i = 1 } ^ { m } \delta _ { Y _ { i } } } \end{array}$ . Standard empirical Wasserstein estimates [23, 58] give

$$
\mathbb { E } W _ { 1 } ( Q _ { m } , Q ) \lesssim a _ { d , 0 } ( m ) ,
$$

where the two-dimensional case has the critical rate $\sqrt { \log ( e m ) / m }$ by [9, Theorem $3 ]$ . Since ${ a _ { d , \tau } = a _ { d , 0 } }$ for $d \leq 2$ , this also proves the smooth upper bound in low dimensions.

Suppose now that $d > 2$ and $\tau > 0$ . Let Φ and $\Psi _ { j }$ be the boundary wavelet families fixed in Appendix $\mathrm { A }$ and define the empirical projection

$$
\widetilde { q } _ { J } = \sum _ { \zeta \in \Phi } \widehat { \alpha } _ { \zeta } \zeta + \sum _ { j = j _ { 0 } } ^ { J } \sum _ { \psi \in \Psi _ { j } } \widehat { \beta } _ { \psi } \psi .
$$

The coeficients are empirical wavelet averages and hence unbiased for the coeficients of $q .$ Write $q _ { J } = \mathbb { E } \widetilde { q } _ { J }$ for the population projection. Because constants belong to the scaling space, $\textstyle \int { \widetilde { q } } J = 1$

The characterization by boundary wavelets implies

$$
\| \beta _ { q , j } \| _ { \ell ^ { 1 } } \leq C 2 ^ { - j \tau } 2 ^ { j d / 2 } ,
$$

and Lemma B.2 converts this into

$$
\| ( q - q _ { J } ) d y \| _ { \mathrm { K R } } \leq C 2 ^ { - J ( \tau + 1 ) } .\tag{B.2}
$$

For the variance term, finite overlap and $\textstyle \int q = 1$ give

$$
\sum _ { \psi \in \Psi _ { j } } \operatorname { V a r } ( \widehat { \beta } _ { \psi } ) \leq \frac { 1 } { m } \int \sum _ { \psi \in \Psi _ { j } } | \psi ( y ) | ^ { 2 } q ( y ) d y \lesssim \frac { 2 ^ { j d } } { m } .
$$

Cauchy–Schwarz therefore yields

$$
\mathbb { E } \Vert \widehat { \beta } _ { j } - \beta _ { q , j } \Vert _ { \ell ^ { 1 } } \leq | \Psi _ { j } | ^ { 1 / 2 } \left\{ \sum _ { \psi \in \Psi _ { j } } \mathrm { V a r } ( \widehat { \beta } _ { \psi } ) \right\} ^ { 1 / 2 } \lesssim m ^ { - 1 / 2 } 2 ^ { j d } .
$$

No density upper bound is used in this calculation. Another application of Lemma B.2 yields

$$
\begin{array} { r } { \mathbb { E } \| \{ \widetilde { q } _ { J } - q _ { J } \} d y \| _ { \mathrm { K R } } \lesssim m ^ { - 1 / 2 } 2 ^ { J ( d / 2 - 1 ) } . } \end{array}\tag{B.3}
$$

Choose $J = J _ { m }$ so that $2 ^ { J ( 2 \tau + d ) } \asymp m$ . Then (B.2)–(B.3) give

$$
\mathbb { E } \| ( \widetilde { q } _ { J } - q ) d y \| _ { \mathrm { K R } } \lesssim a _ { d , \tau } ( m ) .
$$

Applying the measurable approximate metric projection rule $\left( \mathrm { A . 4 } \right)$ in the KR metric to the set of probability laws, with $\delta = m ^ { - 1 }$ , gives a probability estimator satisfying

$$
W _ { 1 } ( \widehat { Q } _ { m } , Q ) \leq 2 \| ( \widetilde { q } _ { J } - q ) d y \| _ { \mathrm { K R } } + m ^ { - 1 } ,
$$

establishing the smooth high-dimensional upper bound for suficiently large $m .$

We now extend these bounds to fixed finite moments. For the empirical estimators above, let $Z _ { m } = W _ { 1 } ( Q _ { m } , Q )$ . Replacing one observation changes $Z _ { m }$ by at most ${ \sqrt { d } } / m$ , by coupling the unchanged empirical atoms. For the smooth high-dimensional estimator, set $Z _ { m } = | | ( \widetilde { q } \ j - q ) d \ j | | _ { \mathrm { K R } }$ . Lemma B.2 and finite overlap of the wavelet supports show that replacing one observation changes $Z _ { m }$ by at most

$$
\frac { C } { m } \left\{ 1 + \sum _ { j = j _ { 0 } } ^ { J } 2 ^ { - j ( 1 + d / 2 ) } 2 ^ { j d / 2 } \right\} \leq \frac { C ^ { \prime } } { m } .
$$

The constants are independent of J. The bounded diferences inequality [36] and integration of its upper tail therefore give, in each case,

$$
\begin{array} { r } { \| Z _ { m } \| _ { L ^ { \kappa } ( \mathbb { P } ) } \leq \mathbb { E } Z _ { m } + C _ { \kappa } m ^ { - 1 / 2 } \lesssim _ { \kappa } a _ { d , \tau } ( m ) , } \end{array}
$$

where $\tau = 0$ in the unrestricted model. The final inequality uses the preceding first-moment bounds and $a _ { d , \tau } ( m ) \geq m ^ { - 1 / 2 }$ . In the smooth high-dimensional case, (A.4) transfers this bound to $W _ { 1 } ( \widehat { Q } _ { m } , Q )$ . Thus the same estimators satisfy

$$
\bigl \{ \mathbb { E } W _ { 1 } ^ { \kappa } ( \widehat { Q } _ { m } , Q ) \bigr \} ^ { 1 / \kappa } \lesssim _ { \kappa } a _ { d , \tau } ( m ) .\tag{B.4}
$$

In the smooth high-dimensional case, use a fixed probability law on the cube for the finitely many remaining positive sample sizes. Since W<sub>1</sub> is bounded by the cube diameter and $a _ { d , \tau } ( m ) > 0$ , increasing the constant extends (B.4) to every integer $m \geq 1 .$ , as required in Proposition D.2.

Appendix C. Unconditional lower bounds. A single compactly supported potential hypercube is used in every response dimension. It supplies the testing, separation, and attenuation estimates required below, with the dimensional transition encoded by the spectral sum

$$
B J = \sum _ { j = j _ { 0 } } ^ { J } 2 ^ { j ( d - 2 ) } .\tag{C.1}
$$

Thus $B _ { J } \asymp 1$ when $d = 1 , B _ { J } \asymp J$ when $d = 2 ,$ and $B _ { J } \asymp 2 ^ { J ( d - 2 ) }$ when $d > 2 .$

C.1. Testing and metric decoding. Write $a _ { m } : = a _ { d , \tau } ( m )$ . For probability densities $f , g ,$ we use $\begin{array} { r } { H ^ { 2 } ( f d y , g d y ) = \int ( \sqrt { f } - \sqrt { g } ) ^ { 2 } d y } \end{array}$ for the squared Hellinger distance. We combine a metric decoder with a hypercube of distributions. One-coordinate neighbors are statistically close, whereas accumulated coordinate errors yield $W _ { 1 }$ separation of order $a _ { m }$

Lemma C.1 (Hypercube decoder). Let $( \mathcal { M } , d )$ be a metric space, let $\{ P _ { \theta } : \theta \in \{ - 1 , 1 \} ^ { M } \}$ be a family of experiments, and associate to each vertex a point $f _ { \theta } \in \mathcal { M }$ . Assume that every single-coordinate flip satisfies

$$
\mathrm { T V } ( P _ { \theta } , P _ { \theta ^ { ( u ) } } ) \leq \delta _ { \mathrm { T V } } < 1 .
$$

Given any estimator ${ \widehat { f } } ,$ choose a measurable nearest vertex ${ \widehat { \theta } } ,$ using deterministic tie breaking. Then

$$
d ( f _ { \theta } , f _ { \widehat { \theta } } ) \leq 2 d ( f _ { \theta } , { \widehat { f } } ) .
$$

Let π be the uniform prior on the hypercube, and let $\mathbb { E } _ { \pi }$ denote expectation over $\theta \sim \pi$ and the data drawn from $P _ { \theta }$ . Then, for every coordinate $u ,$

$$
\mathbb { E } _ { \pi } { \mathbf { 1 } } \{ \widehat { \theta } _ { u } \neq \theta _ { u } \} \geq \frac { 1 - \delta _ { \mathrm { T V } } } { 2 } .\tag{C.2}
$$

Proof. Nearest vertex optimality and the triangle inequality give the first claim. For the second, fix u and pair every vertex with its uth-coordinate flip. The decision $\widehat { \theta } _ { u }$ tests the two paired experiments, so the sum of its two error probabilities is at least $1 - \mathrm { T V } \big ( P _ { \theta } , P _ { \theta ^ { ( u ) } } \big )$ Averaging over the pairs proves (C.2). ■

C.2. Multiscale potential estimates. The same analytic argument applies in every dimension. We use quantitative Lipschitz truncation together with the standard positive-order wavelet square function equivalence.

Proof of Proposition $4 . 2 .$ Extend U by zero to $\mathbb { R } ^ { d }$ . Compact support gives $\textstyle \int g = 0$ and permits integration by parts against Lipschitz test functions.

The case $U = 0$ is immediate. Otherwise write $\begin{array} { r } { E _ { 2 } = \int | \nabla U | ^ { 2 } } \end{array}$ and $\begin{array} { r } { E _ { 4 } = \int | \nabla U | ^ { 4 } } \end{array}$ . For any $\lambda > 0$ , apply [17, Theorem 2.3] on the cube with an amplitude cutof larger than $\| U \| _ { \infty }$ . After zero extension and a fixed rescaling of the truncation parameter, this gives a function $\phi$ and an exceptional set $E = \{ U \neq \phi \}$ satisfying

$$
\mathrm { L i p } ( \phi ) \leq \lambda , \qquad | E | \leq C _ { d } E _ { 4 } \lambda ^ { - 4 } , \qquad \nabla \phi = \nabla U \quad \mathrm { a . e . ~ o n ~ } E ^ { c } .
$$

The amplitude cutof contributes no exceptional set. The bound for $| E |$ follows from $L ^ { 4 }$ boundedness of the Hardy–Littlewood maximal operator applied to $\nabla U$ , and the gradient agreement follows from locality of weak gradients. H¨older’s inequality gives

$$
\begin{array} { r l } & { \displaystyle \left. \int _ { E } \nabla U \cdot ( \nabla \phi - \nabla U ) \right. \leq E _ { 4 } ^ { 1 / 2 } \lvert E \rvert ^ { 1 / 2 } + \lambda E _ { 4 } ^ { 1 / 4 } \lvert E \rvert ^ { 3 / 4 } } \\ & { \qquad \leq C E _ { 4 } \lambda ^ { - 2 } . } \end{array}
$$

Integration by parts therefore yields

$$
\int ( - \Delta U ) \phi \geq E _ { 2 } - C E _ { 4 } \lambda ^ { - 2 } .
$$

Choose $\lambda ^ { 2 } = C _ { 0 } E _ { 4 } / E _ { 2 }$ with $C _ { 0 }$ suficiently large. Then $\int g \phi \ge E _ { 2 } / 2$ , and $\phi / \lambda$ is an admissible test function for the KR norm. Hence

$$
\| ( - \Delta U ) d y \| _ { \mathrm { K R } } \geq c _ { d } \frac { E _ { 2 } ^ { 3 / 2 } } { E _ { 4 } ^ { 1 / 2 } } ,
$$

which proves (4.4).

Lemma C.2 (Positive-order square function for the interior family). Let $1 < s <$ ∞ and

$$
V = \sum _ { j = j _ { 0 } } ^ { J } \sum _ { k \in \mathcal { K } _ { j } } c _ { j , k } \psi _ { j , k } .
$$

All retained d-dimensional wavelets are supported in a common compact cube $K _ { 0 } \Subset ( 0 , 1 ) ^ { d }$ Then

$$
\| \nabla V \| _ { L ^ { s } ( [ 0 , 1 ] ^ { d } ) } \asymp _ { s } \left\| \left( \sum _ { j , k } 2 ^ { 2 j } | c _ { j , k } | ^ { 2 } | \psi _ { j , k } | ^ { 2 } \right) ^ { 1 / 2 } \right\| _ { L ^ { s } ( [ 0 , 1 ] ^ { d } ) } ,\tag{C.3}
$$

where the constants do not depend on J or on the coeficient array.

Proof. Let $\widetilde { V }$ be the zero extension of V to $\mathbb { R } ^ { d }$ . Every retained wavelet is an interior translate of a whole-space wavelet, so its whole-space coeficient is $c _ { j , k }$ and all other coeficients, including the scaling coeficients, vanish. The order-one wavelet characterization gives

$$
\| \widetilde V \| _ { W ^ { 1 , s } ( { \mathbb R } ^ { d } ) } \asymp _ { s } \left\| \left( \sum _ { j , k } 2 ^ { 2 j } | c _ { j , k } | ^ { 2 } | \psi _ { j , k } | ^ { 2 } \right) ^ { 1 / 2 } \right\| _ { L ^ { s } ( { \mathbb R } ^ { d } ) } ,
$$

see [14, Remark 3.8.5, equations (3.8.42)–(3.8.43)]. The fixed basis has regularity greater than one and at least two vanishing moments, as required at order one.

Poincar´e’s inequality bounds $\| V \| _ { L ^ { s } }$ by $C _ { s } \| \nabla V \| _ { L ^ { s } }$ , since V vanishes near the boundary. The zero extension and the square function both vanish outside the response cube, giving (C.3). The constants depend only on the fixed wavelet system, the exponent $s ,$ and the fixed support region; in particular, they are independent of J and the coeficient array.

The gradient estimates below give Wasserstein separation through Proposition 4.2. The derivative estimates ensure positivity and Besov regularity, while the one-coordinate Hessian estimate controls the Hellinger distance between neighboring alternatives.

Lemma C.3 (Multiscale potential bounds). Let

$$
U _ { \theta } = \varepsilon \sum _ { j = j _ { 0 } } ^ { J } \sum _ { k \in \mathcal { K } _ { j } } \theta _ { j , k } 2 ^ { - 2 j } \psi _ { j , k } , \qquad \theta _ { j , k } \in \{ - 1 , 1 \} ,
$$

where $| \mathcal { K } _ { j } | \asymp 2 ^ { j d }$ and the wavelets are the interior family fixed above. Then

(C.4)

$$
\| \nabla U _ { \theta } \| _ { 2 } ^ { 2 } \lesssim \varepsilon ^ { 2 } B _ { J } ,\tag{C.5}
$$

$$
\| D ^ { \ell + 2 } U _ { \theta } \| _ { \infty } \lesssim _ { \ell } \varepsilon 2 ^ { J ( d / 2 + \ell ) } , \qquad \ell \in \{ 0 , \ldots , K - 2 \} .
$$

If two vertices disagree in $H _ { j }$ coordinates at resolution $j$ and

$$
S = \sum _ { j = j _ { 0 } } ^ { J } 2 ^ { - 2 j } H _ { j } ,
$$

then, for $V = U _ { \theta } - U _ { \theta ^ { \prime } }$

(C.6)

$$
\begin{array} { r l } & { \| \nabla V \| _ { 2 } ^ { 2 } \gtrsim \varepsilon ^ { 2 } S , } \\ & { \| \nabla V \| _ { 4 } ^ { 4 } \lesssim \varepsilon ^ { 4 } B _ { J } S . } \end{array}\tag{C.7}
$$

For a single-coordinate $\mathit { f i p }$ , one also has

$$
\| D ^ { 2 } ( U _ { \theta } - U _ { \theta ^ { ( j , k ) } } ) \| _ { 2 } ^ { 2 } \lesssim \varepsilon ^ { 2 } .\tag{C.8}
$$

Proof. Lemma C.2 with $s = 2$ gives the first and third estimates. The derivative estimate (C.5) follows from finite overlap together with $\| D ^ { \ell + 2 } \psi _ { j , k } \| _ { \infty } \lesssim 2 ^ { j ( d / 2 + \ell + 2 ) }$

To prove (C.7), write

$$
N _ { d } ( y ) = \sum _ { j = j _ { 0 } } ^ { J } \sum _ { k \in E _ { j } } 2 ^ { - 2 j } | \psi _ { j , k } ( y ) | ^ { 2 } ,
$$

where $E _ { j }$ denotes the set of difering coordinates. Lemma C.2 gives $\begin{array} { r } { \| \nabla V \| _ { 4 } ^ { 4 } \lesssim \varepsilon ^ { 4 } \int N _ { d } ^ { 2 } . } \end{array}$ . Since $\textstyle \int N _ { d } = { \dot { S } }$ by $L ^ { 2 }$ normalization and finite overlap implies $N _ { d } \lesssim B _ { J }$ pointwise, $\bar { \int { \cal N } } _ { d } ^ { 2 } \lesssim { \cal B } _ { J } S$ For a one-coordinate flip, $\| D ^ { 2 } \psi _ { j , k } \| _ { 2 } \asymp 2 ^ { 2 j }$ together with the coeficient $2 ^ { - 2 j }$ gives the Hessian estimate.

For $q _ { \theta } = 1 - \Delta U _ { \theta }$ , Proposition 4.2 and (C.6)–(C.7) give

$$
W _ { 1 } ( Q _ { \theta } , Q _ { \theta ^ { \prime } } ) \geq c \frac { \varepsilon } { \sqrt { B _ { J } } } \sum _ { j = j _ { 0 } } ^ { J } 2 ^ { - 2 j } H _ { j } .\tag{C.9}
$$

## C.3. A unified hypercube family.

Proposition C.4 (Least favorable hypercube with localization control). Fix $d \ge 1 , \tau \ge 0$ and $1 \leq r , v \leq \infty$ , and abbreviate $a _ { m } = a _ { d , \tau } ( m )$ $I f \tau > 0$ , use the fixed Besov slack from Assumption 2.2. For every suficiently large m, there is a finite hypercube of distributions

$$
\{ Q _ { \theta } ^ { ( m ) } : \theta \in \{ - 1 , 1 \} ^ { M _ { m } } \} ,
$$

with densities $q _ { \theta } ^ { ( m ) }$ having the following properties.

(i) every density $q _ { \theta } ^ { ( m ) }$ and every attenuated density $1 + t ( q _ { \theta } ^ { ( m ) } - 1 ) , 0 \leq t \leq 1$ , is bounded between fixed positive constants and can be made uniformly as close to the uniform density as required by choosing the fixed cutof constant suficiently small. When $\tau > 0$ , the entire attenuated family lies in the prescribed $B _ { r , v } ^ { \tau }$ ball;

(ii) with $Q _ { \theta , t }$ denoting the attenuated law, for every fixed $1 \leq p < \infty$

$$
W _ { p } ( Q _ { \theta , t } , Q _ { \theta , t ^ { \prime } } ) \leq C _ { p } a _ { m } | t - t ^ { \prime } | , \qquad W _ { p } ( Q _ { \theta , t } , \mathrm { U n i f } ) \leq C _ { p } a _ { m } .
$$

(iii) every one-coordinate $\mathit { f l i p }$ satisfies

$$
H ^ { 2 } ( Q _ { \theta } ^ { ( m ) } , Q _ { \theta ^ { ( u ) } } ^ { ( m ) } ) \leq C / m ;
$$

(iv) if a decoder has coordinate error probability bounded below by a fixed $\eta > 0$ on every coordinate, then

$$
\mathbb { E } _ { \pi } W _ { 1 } ( Q _ { \theta } ^ { ( m ) } , Q _ { \widehat { \theta } } ^ { ( m ) } ) \geq c _ { \eta } a _ { m } .
$$

Proof. Set $\varepsilon = c m ^ { - 1 / 2 }$ and use the potential family of Lemma C.3. For $\tau \geq 0$ , with $\tau = 0$ interpreted as the unrestricted branch, choose J maximal subject to

$$
\varepsilon 2 ^ { J ( \tau + d / 2 ) } \leq c _ { 0 } ,\tag{C.10}
$$

where $c _ { 0 } , c > 0$ are fixed constants chosen below. Thus

$$
2 ^ { J } \asymp m ^ { 1 / ( 2 \tau + d ) } .\tag{C.11}
$$

Define $q _ { \theta } = 1 - \Delta U _ { \theta }$ . Since every $U _ { \theta }$ is supported in $\textstyle K _ { 0 } , \int q _ { \theta } = 1$ . By (C.5),

$$
\| q _ { \theta } - 1 \| _ { \infty } \lesssim \varepsilon 2 ^ { J d / 2 } .
$$

Thus (C.10) ensures uniform lower and upper bounds on the densities.

When $\tau > 0$ , choose an integer ℓ with $\tau < \ell \leq K - 2$ . The same derivative estimate yields

$$
\| D ^ { \ell } ( q _ { \theta } - 1 ) \| _ { \infty } \lesssim _ { \ell } \varepsilon 2 ^ { J ( d / 2 + \ell ) } .
$$

Because $q _ { \theta } - 1$ is supported a fixed positive distance from the boundary, its zero extension obeys

$$
\| \Delta _ { u } ^ { \ell } ( q _ { \theta } - 1 ) \| _ { L ^ { r } } \lesssim \varepsilon 2 ^ { J d / 2 } \operatorname* { m i n } \{ 1 , ( 2 ^ { J } | u | ) ^ { \ell } \} , \qquad 1 \leq r \leq \infty .
$$

The characterization by finite diferences of $B _ { r , v } ^ { \tau }$ now gives

$$
\| q _ { \theta } - 1 \| _ { B _ { r , v } ^ { \tau } } \lesssim \varepsilon 2 ^ { J ( \tau + d / 2 ) } .\tag{C.12}
$$

Multiplying $U _ { \theta }$ by any $t \in [ 0 , 1 ]$ leaves these estimates unchanged up to constants, which proves part $\mathrm { ( i ) }$

Under a one-coordinate flip, the density diference is $- \Delta ( U _ { \theta } - U _ { \theta ^ { ( u ) } } )$ . Its $L ^ { 2 }$ norm is $O ( \varepsilon )$ by (C.8); uniform positivity then gives the Hellinger bound in part (iii). Choose $c _ { 0 }$ suficiently small for the prescribed density bounds and, when $\tau > 0$ , the Besov slack. With $c _ { 0 }$ fixed, choose c suficiently small that

$$
\operatorname { T V } \bigl ( Q _ { \theta } ^ { \otimes m } , Q _ { \theta ^ { ( u ) } } ^ { \otimes m } \bigr ) \leq 1 / 2 .
$$

The constants in the preceding bounds are independent of J and $\theta ,$ so these choices satisfy the density, regularity, and testing constraints simultaneously.

For the localization estimate, integration by parts together with (C.4) yields

$$
\begin{array} { r } { \| ( q _ { \theta } - 1 ) d y \| _ { \mathrm { K R } } \le \| \nabla U _ { \theta } \| _ { 1 } \le \| \nabla U _ { \theta } \| _ { 2 } \lesssim \varepsilon \sqrt { B _ { J } } . } \end{array}
$$

Replacing $U _ { \theta }$ by $( t - t ^ { \prime } ) U _ { \theta }$ proves part (ii) for $p = 1$ . For fixed $p > 1$ , all attenuated densities are bounded below by a common positive constant by part (i). Integration by parts gives

$$
\begin{array} { r } { \| - ( t - t ^ { \prime } ) \Delta U _ { \theta } \| _ { - 1 , p } \leq | t - t ^ { \prime } | \| \nabla U _ { \theta } \| _ { L ^ { p } } . } \end{array}
$$

Lemma C.2 and finite overlap give $\| \nabla U _ { \theta } \| _ { L ^ { p } } \lesssim _ { p } \varepsilon \sqrt { B _ { J } }$ : the square function in (C.3) is pointwise at most $C \varepsilon \sqrt { B _ { J } }$ . Proposition A.2 therefore gives

$$
W _ { p } ( Q _ { \theta , t } , Q _ { \theta , t ^ { \prime } } ) \lesssim _ { p } | t - t ^ { \prime } | \varepsilon \sqrt { B _ { J } } , \qquad W _ { p } ( Q _ { \theta , t } , \mathrm { U n i f } ) \lesssim _ { p } \varepsilon \sqrt { B _ { J } } .
$$

Thus part (ii) holds for every fixed finite $p .$ Moreover, (C.1) and (C.11) give

$$
\varepsilon \sqrt { B _ { J } } \asymp \left\{ \begin{array} { l l } { m ^ { - 1 / 2 } , } & { d = 1 , } \\ { \{ \log ( e m ) / m \} ^ { 1 / 2 } , } & { d = 2 , } \\ { m ^ { - ( \tau + 1 ) / ( 2 \tau + d ) } , } & { d > 2 , } \end{array} \right.\tag{C.13}
$$

which is of order $a _ { m }$ by construction.

For part (iv), let $H _ { j }$ count coordinate errors at level $j$ and put $\begin{array} { r } { S = \sum _ { j } 2 ^ { - 2 j } H _ { j } } \end{array}$ . The coordinatewise testing assumption implies

$$
\mathbb { E } _ { \pi } S \gtrsim _ { \eta } \sum _ { j } 2 ^ { - 2 j } | { \mathcal { K } } _ { j } | \asymp B _ { J } .
$$

Equation (C.9) then gives

$$
\mathbb { E } _ { \pi } W _ { 1 } ( Q _ { \theta } , Q _ { \widehat { \theta } } ) \gtrsim _ { \eta } \varepsilon \sqrt { B _ { J } } \asymp a _ { m } ,
$$

proving (iv).

## C.4. Completion of the response theorem.

Proof of Theorem $4 . 1 .$ For the upper bound, Proposition B.1 and (B.4) give the asserted κth-moment rate in both the unrestricted and smooth classes.

For the minimax lower bound, use the hypercube in Proposition C.4 and decode any estimator to a nearest vertex in $W _ { 1 }$ . The one-coordinate Hellinger estimate and Lemma C.1 ensure a uniformly positive coordinate error probability. Hence Proposition $\mathrm { C . 4 ( i v ) }$ gives

$$
\begin{array} { r } { \mathbb { E } _ { \pi } W _ { 1 } ( Q _ { \theta } , Q _ { \widehat { \theta } } ) \geq c a _ { d , \tau } ( m ) . } \end{array}
$$

Nearest vertex optimality implies $W _ { 1 } ( Q _ { \theta } , Q _ { \widehat { \theta } } ) \leq 2 W _ { 1 } ( Q _ { \theta } , \widehat { Q } )$ . Jensen’s inequality therefore yields a Bayes lower bound of order $a _ { d , \tau } ( m ) ^ { \kappa }$ under the uniform prior. Since minimax risk dominates this Bayes risk, the required lower bound follows.

Appendix D. Conditional localization with unrestricted covariate distributions. We prove the partition bound used in Section 3, then derive the lower bound under uniform covariates and the adaptive upper bound for higher moments. The argument applies to the two $W _ { 1 }$ classes and, with the unconditional estimates of Appendix E, to the $W _ { p }$ density classes. A separate two-point construction gives the rare region lower bound.

D.1. Partition bound weighted by cell probability. Let Π be any finite measurable partition of X. For each $B \in \Pi$ , set

$$
\pi _ { B } = P _ { X } ( B ) , \qquad N _ { B } = \sum _ { i = 1 } ^ { n } \mathbf { 1 } \{ X _ { i } \in B \} , \qquad Q _ { B } = P _ { Y | X \in B } \quad ( \pi _ { B } > 0 ) .
$$

For an unconditional rate $a = a _ { d _ { y } , \tau }$ and a fixed finite $1 \leq \kappa < \infty$ , define the capped moment rate

$$
r _ { \kappa } ( t ) = \left\{ { 1 , \begin{array} { l l } { { 0 \leq t < 1 , } } & { { 0 \leq t < 1 , } } \\ { { a ( t ) ^ { \kappa } , } } & { { t \geq 1 . } } \end{array} } \right.\tag{D.1}
$$

Changing the value on a bounded interval afects only constants.

Lemma D.1 (Binomial transfer). Let $N \sim \mathrm { B i n } ( n , \pi )$ and $\lambda = n \pi$ . For every fixed finite $1 \leq \kappa < \infty$

$$
\mathbb { E } [ a ( N ) ^ { \kappa } \mathbf { 1 } \{ N \geq 2 \} + \mathbf { 1 } \{ N < 2 \} ] \leq C _ { \kappa } r _ { \kappa } ( \lambda ) ,\tag{D.2}
$$

where $a ( N )$ on $N \geq 2$ denotes any of the unconditional rates in (2.12) evaluated at the integer $N _ { ☉ }$

Proof. For bounded λ, the result follows after increasing $C _ { \kappa }$ . Assume henceforth that $\lambda \geq 4$ and split according to $N \geq \lambda / 2$ . The explicit rates in (2.12) are nonincreasing and satisfy $a ( \lambda / 2 ) \leq { \sqrt { 2 } } a ( \lambda )$ . Thus, on $\{ N \geq \lambda / 2 \}$ ,

$$
a ( N ) ^ { \kappa } \leq a ( \lambda / 2 ) ^ { \kappa } \leq 2 ^ { \kappa / 2 } a ( \lambda ) ^ { \kappa } .
$$

On $\{ N < \lambda / 2 \}$ , the integrand in (D.2) is at most one, whereas Chernof’s inequality gives $\operatorname* { P r } ( N < \lambda / 2 ) \leq e ^ { - c \lambda }$ . Since $a ( \lambda ) \geq \lambda ^ { - 1 / 2 }$ , for fixed finite κ we have $e ^ { - c \lambda } \leq C _ { \kappa } \lambda ^ { - \kappa / \bar { 2 } } \leq C _ { \kappa } \bar { a ( \lambda ) } ^ { \kappa }$ Combining the two events proves (D.2).

Proposition D.2 (Partition bound weighted by cell probability). Fix $1 \leq p < \infty$ and finite $1 \leq \kappa < \infty$ . Assume the distribution class is closed under cellwise mixtures and $t h a t ,$ from m $\geq 2 \ i . i . d .$ . responses, one has an estimator satisfying

$$
\begin{array} { r } { \operatorname* { s u p } _ { Q } \mathbb { E } W _ { p } ^ { \kappa } ( \widehat { Q } _ { m } , Q ) \leq C _ { \kappa } a _ { d _ { y } , \tau } ( m ) ^ { \kappa } . } \end{array}
$$

Apply this estimator on each cell with at least two observations, and use any fixed admissible fallback otherwise. Then

$$
\begin{array} { r l r } {  { \mathbb { E } \int W _ { p } ^ { \kappa } ( \widehat { Q } _ { x } , Q _ { x } ) P _ { X } ( d x ) } } \\ & { } & { \leq C _ { \kappa } \displaystyle \sum _ { B : \pi _ { B } > 0 } \{ \int _ { B } W _ { p } ^ { \kappa } ( Q _ { x } , Q _ { B } ) P _ { X } ( d x ) + \pi _ { B } r _ { \kappa } ( n \pi _ { B } ) \} . } \end{array}\tag{D.3}
$$

Proof. Conditional on $N _ { B } = k$ , the responses in cell B are i.i.d. with distribution $Q _ { B }$ . The assumed bound for distribution estimation, followed by Lemma D.1, therefore yields

$$
\mathbb { E } W _ { p } ^ { \kappa } ( \widehat { Q } _ { B } , Q _ { B } ) \leq C _ { \kappa } r _ { \kappa } ( n \pi _ { B } ) .
$$

For every $\kappa \geq 1$

$$
W _ { p } ^ { \kappa } ( \widehat { Q } _ { B } , Q _ { x } ) \leq 2 ^ { \kappa - 1 } \{ W _ { p } ^ { \kappa } ( \widehat { Q } _ { B } , Q _ { B } ) + W _ { p } ^ { \kappa } ( Q _ { B } , Q _ { x } ) \} .
$$

Multiplying by $\pi _ { B }$ , integrating over the conditional distribution of x in $B ,$ , and summing over the partition yields (D.3). In the applications below, the required mixture closure is immediate for unrestricted laws; for smooth distribution classes it follows from convexity of the Besov constraint, together with preservation of any stated pointwise density bounds under mixtures.

## D.2. Replication under the uniform covariate distribution.

Proposition D.3 (Replicated hypercube lower bound at $\kappa = 1 )$ . Fix $\tau \geq 0$ and $1 \leq r , v \leq \infty$ Put $a _ { m } = a _ { d _ { y } , \tau } ( m )$ and take the hypercube family $\{ Q _ { \theta } ^ { ( m ) } \}$ from Proposition $C . 4 \cdot$ . There exist constants $A _ { 0 } , c _ { 0 } > 0$ such that, for suficiently small h and suficiently large $m$

$$
m = \lfloor A _ { 0 } n h ^ { d _ { x } } \rfloor , \qquad a _ { m } \leq c _ { 0 } h ^ { \beta }\tag{D.4}
$$

imply

$$
\mathcal { R } _ { n , 1 , 1 } ^ { \star } \geq c a _ { m }\tag{D.5}
$$

for the corresponding conditional model. This lower bound already holds when $P _ { X }$ is uniform on $\mathcal { X }$

Proof. Choose $M _ { x } \asymp h ^ { - d _ { x } }$ smooth covariate bumps $b _ { a }$ with supports of diameter $O ( h )$ separated by distances of order $h$ . Let $U _ { a }$ be plateau regions with $| U _ { a } | \asymp h ^ { d _ { x } }$ . For an array $\Theta = ( \theta _ { 1 } , \dots , \theta _ { M _ { x } } )$ , set

$$
q _ { \Theta , x } ( y ) = 1 + \sum _ { a } b _ { a } ( x ) ( q _ { \theta _ { a } } ^ { ( m ) } ( y ) - 1 ) .
$$

The supports are disjoint, so at most one bump is active at any covariate value. We verify that the resulting conditional law has the required H¨older modulus. For a vertex $\theta$ and $t \in [ 0 , 1 ]$ write

$$
q _ { \theta , t } = 1 + t ( q _ { \theta } ^ { ( m ) } - 1 ) , \qquad Q _ { \theta , t } = q _ { \theta , t } d y .
$$

Part (ii) of Proposition C.4 yields

$$
W _ { 1 } ( Q _ { \theta , t } , Q _ { \theta , t ^ { \prime } } ) \leq C a _ { m } | t - t ^ { \prime } | .
$$

The scaled bumps may be chosen so that

$$
| b _ { a } ( x ) - b _ { a } ( x ^ { \prime } ) | \leq C \operatorname* { m i n } \{ 1 , | x - x ^ { \prime } | / h \} \leq C h ^ { - \beta } | x - x ^ { \prime } | ^ { \beta } .
$$

If $x , x ^ { \prime }$ lie in the same active bump, or if one point lies immediately outside it so that the corresponding attenuation is zero, the path estimate above and $a _ { m } \leq c _ { 0 } h ^ { \beta }$ imply

$$
W _ { 1 } ( Q _ { \Theta , x } , Q _ { \Theta , x ^ { \prime } } ) \leq C a _ { m } h ^ { - \beta } | x - x ^ { \prime } | ^ { \beta } \leq C c _ { 0 } | x - x ^ { \prime } | ^ { \beta } .
$$

If the points belong to distinct active bumps, their supports are separated by at least $c h .$ . Since each conditional law is within $C a _ { m }$ of the uniform baseline by Proposition C.4(ii),

$$
W _ { 1 } ( Q _ { \Theta , x } , Q _ { \Theta , x ^ { \prime } } ) \leq C a _ { m } \leq C c _ { 0 } h ^ { \beta } \leq C ^ { \prime } c _ { 0 } | x - x ^ { \prime } | ^ { \beta } .
$$

Choose the fixed response-family constants as in Proposition C.4, so that every attenuated density remains in the prescribed class. With these constants fixed, choose $c _ { 0 }$ suficiently small that the H¨older constants in both cases are no larger than $L _ { X }$

For the testing scale, flipping one coordinate in block a changes the joint law of one observation by squared Hellinger distance at most $C h ^ { d _ { x } } / m$ . With the preceding constants fixed, choose $A _ { 0 }$ suficiently large. Then $m = \lfloor A _ { 0 } n h ^ { d _ { x } } \rfloor$ and Hellinger tensorization keep adjacent n-sample total variation distances uniformly below one for suficiently large m. For each a, decode an arbitrary estimator by nearest vertex in the integrated $W _ { 1 }$ metric

$$
d _ { a } ( Q , Q ^ { \prime } ) = \int _ { U _ { a } } W _ { 1 } ( Q _ { x } , Q _ { x } ^ { \prime } ) d x .\tag{D.6}
$$

The testing lemma gives a uniformly positive coordinate error probability. Because the response vertex is constant on each plateau $U _ { a }$ , Proposition C.4(iv) gives

$$
\begin{array} { r } { \mathbb { E } _ { \pi } d _ { a } ( Q _ { \theta _ { a } } ^ { ( m ) } , Q _ { \widehat { \theta _ { a } } } ^ { ( m ) } ) \geq c | U _ { a } | a _ { m } . } \end{array}
$$

Nearest vertex optimality bounds the decoded distance by twice the estimator error integrated over $U _ { a }$ . Summing over a and using $M _ { x } | U _ { a } | \asymp 1$ gives a lower bound $c a _ { m }$ for the integrated $W _ { 1 }$ risk, proving (D.5). ■

D.3. Higher moments and adaptive localization. The fixed grid proof is useful for $\kappa = 1$ , but for larger κ the function $t \mapsto t r _ { \kappa } ( t )$ need not be concave. The following adaptive construction removes this restriction without imposing regularity on $P _ { X }$

Proposition D.4 (Adaptive localization). Fix $1 \leq p < \infty$ and a finite $1 \leq \kappa < \infty$ . For $p = 1$ take either conditional distribution class in Theorem 2.5. For $p > 1$ , take the distribution class with densities bounded away from zero and infinity in Theorem 2.6. There exists a conditional estimator satisfying

$$
\operatorname* { s u p } _ { P } \mathcal { R } _ { n , p , \kappa } ( \widehat { Q } ; P ) \lesssim A _ { n } ( \tau ) ^ { \kappa } + n ^ { - 1 }\tag{D.7}
$$

for every integer $n \geq 1$ and every Borel covariate distribution $P _ { X }$

Proof. We first bound the expected mass of terminal cells afected by incorrect acceptance or rejection. We then control the risk on the remaining cells using the local moment bounds and disjointness of the terminal partition.

It sufices to prove the bound for suficiently large $n .$ For the finitely many smaller positive sample sizes, use a fixed probability law for every query covariate; boundedness of $W _ { p } ^ { \kappa }$ and positivity of $A _ { n } ( \tau ) ^ { \kappa } + n ^ { - 1 }$ give the same upper bound after increasing the constant. Split the data into two subsamples of sizes $n _ { 0 }$ and $n _ { 1 }$ with $n _ { 0 } \asymp n _ { 1 } \asymp n$ . Construct a dyadic partition of $\mathcal { X }$ from the first subsample. A level-j cube has side length $h _ { j } = 2 ^ { - j }$ . Define

$$
T _ { j } = \left\lceil C _ { 0 } 2 ^ { j b } ( 1 + j ) ^ { \eta } \right\rceil ,
$$

where

$$
( b , \eta ) = \left\{ \begin{array} { l l } { ( 2 \beta , 0 ) , } & { d _ { y } = 1 , } \\ { ( 2 \beta , 1 ) , } & { d _ { y } = 2 , } \\ { \big ( \beta ( 2 \tau + d _ { y } ) / ( \tau + 1 ) , 0 \big ) , } & { d _ { y } > 2 . } \end{array} \right.
$$

By construction, $a ( T _ { j } ) \lesssim h _ { j } ^ { \beta }$ . In the $p > 1$ case, use Proposition E.1 for the local distribution estimator; cellwise mixtures preserve the density and Besov constraints, and joint convexity of $W _ { p } ^ { p }$ gives the same $O ( h _ { j } ^ { \beta } )$ ) pooling bias. Declare a cell accepted when its count in the first subsample is at least $T _ { j }$ . For large $n ,$ the root is accepted, whereas $T _ { j } > n$ at suficiently fine levels; hence every dyadic chain has a deepest accepted cell. Use that cell for each query point. Acceptance is inherited by ancestors because counts decrease and thresholds increase down a chain. Fit the distribution estimator using only the second subsample in the selected cell, making the selected cell independent of the estimation sample.

Let $\mathcal { L }$ be the disjoint partition formed by the first rejected cells. If $x \in L \in \mathcal { L }$ , its pooling cell is the accepted parent $A ( L )$ ; parent and child side lengths and thresholds difer only by constants, which we suppress. If a level-j cell satisfies $n _ { 0 } P _ { X } ( B ) < T _ { j } / 2$ , Chernof’s inequality bounds its acceptance probability by $e ^ { - c T _ { j } }$ . Multiplying by its mass, summing over the $O ( 2 ^ { j d _ { x } } )$ cells at level $j ,$ and then over $j$ gives

$$
\frac { C } { n } \sum _ { j \geq 0 } 2 ^ { j d _ { x } } T _ { j } e ^ { - c T _ { j } } \lesssim n ^ { - 1 } .
$$

Likewise, when $n _ { 0 } P _ { X } ( B ) > 2 T _ { j }$ , the rejection probability is at most $e ^ { - c n _ { 0 } P _ { X } ( B ) }$ . The bound $t e ^ { - c t } \lesssim \dot { T } _ { j } e ^ { - c ^ { \prime } T _ { j } }$ for $t > 2 T _ { j }$ gives another total contribution of order $n ^ { - 1 }$ . Let B denote the sum of $P _ { X } { \left( B \right) }$ over dyadic cells B that are accepted in the first case or rejected in the second. The preceding bounds give $\mathbb { E } \mathfrak { B } \lesssim n ^ { - 1 }$ . Define

$$
{ \mathcal { L } } _ { \mathrm { g o o d } } = \{ L \in { \mathcal { L } } : L { \mathrm { ~ a n d ~ } } A ( L ) { \mathrm { ~ a r e ~ n o t ~ m i s c l a s s i f i e d } } \} .
$$

For each accepted parent $B ,$ , the terminal cells with $A ( L ) = B$ are disjoint subsets of $B .$ Consequently, for every realization of the first subsample,

$$
\begin{array} { r l } & { \displaystyle \sum _ { L \in \mathcal { L } \backslash \mathcal { L } _ { \mathrm { g o o d } } } P _ { X } ( L ) \leq \sum _ { L \in \mathcal { L } } P _ { X } ( L ) + \sum _ { \begin{array} { l } { B \mathrm { ~ a c c e p t e d ~ } } \end{array} } \sum _ { L \in \mathcal { L } } P _ { X } ( L ) } \\ & { \qquad \leq \mathfrak { B } . } \end{array}
$$

Since $W _ { p } ^ { \kappa } \leq d _ { y } ^ { \kappa / 2 }$ on the response cube, these terminal cells contribute at most CEB to the integrated risk. For $L \in \mathcal { L } _ { \mathrm { g o o d } }$ , a terminal cell and its accepted parent satisfy

$$
P _ { X } ( L ) \lesssim T _ { j ( L ) } / n , \qquad n _ { 0 } P _ { X } ( A ( L ) ) \gtrsim T _ { j ( L ) } .\tag{D.8}
$$

The second inequality also implies $n _ { 1 } P _ { X } ( A ( L ) ) \gtrsim T _ { j ( L ) }$ . Conditional on the first subsample, the count in the second subsample in $A ( L )$ is binomial with this mean. Lemma D.1, together with $a ( T _ { j ( L ) } ) \lesssim h _ { L } ^ { \beta }$ , bounds the distribution estimation term by $C h _ { L } ^ { \beta \kappa }$ , including the fallback event. The conditional H¨older condition gives the same order for the pooling bias. Hence

$$
\mathcal { R } _ { n , p , \kappa } ( \widehat { Q } ; P ) \lesssim \mathbb { E } \sum _ { L \in \mathcal { L } _ { \mathrm { g o o d } } } P _ { X } ( L ) h _ { L } ^ { \beta \kappa } + \mathbb { E } \mathfrak { B } .\tag{D.9}
$$

Choose $j _ { * }$ so that

$$
2 ^ { j _ { * } ( d _ { x } + b ) } ( 1 + j _ { * } ) ^ { \eta } \asymp n , \qquad h _ { * } = 2 ^ { - j _ { * } } .\tag{D.10}
$$

At this scale $h _ { * } ^ { \beta } \asymp A _ { n } ( \tau )$ . Cells in $\mathcal { L } _ { \mathrm { g o o d } }$ finer than $h _ { * }$ have total probability under $P _ { X }$ at most one and hence contribute at most $h _ { * } ^ { \beta \kappa }$ . For the coarser cells in $\mathcal { L } _ { \mathrm { g o o d } } , ( \mathrm { D } . 8 )$ gives

$$
\sum _ { \stackrel { L \in \mathcal { L } _ { \mathrm { g o o d } } } { j ( L ) \leq j _ { * } } } P _ { X } ( L ) h _ { L } ^ { \beta \kappa } \stackrel { < } { \sim } \frac { 1 } { n } \sum _ { \stackrel { L \in \mathcal { L } _ { \mathrm { g o o d } } } { j ( L ) \leq j _ { * } } } | L | h _ { L } ^ { \beta \kappa - b - d _ { x } } ( 1 + j ( L ) ) ^ { \eta } .
$$

Here $| L | = h _ { L } ^ { d _ { x } }$ is the Euclidean volume of $L .$ . Disjointness gives $\begin{array} { r } { \sum _ { L \in \mathcal { L } _ { \mathrm { g o o d } } } | L | \le 1 } \end{array}$ , so the preceding display is at most

$$
\frac { C } { n } \operatorname* { m a x } _ { 0 \leq j \leq j _ { * } } 2 ^ { j ( d _ { x } + b - \beta \kappa ) } ( 1 + j ) ^ { \eta } \lesssim n ^ { - 1 } + h _ { * } ^ { \beta \kappa } .
$$

If $d _ { x } + b - \beta \kappa \geq 0$ , the maximum occurs at $j _ { * }$ and (D.10) gives the second term. If $d _ { x } + b - \beta \kappa < 0$ the exponentially decreasing factor makes the maximum bounded independently of $n _ { \mathrm { : } }$ giving the first term. This includes the critical case without summing over resolutions. Together with $\mathbb { E } \mathfrak { B } \lesssim n ^ { - 1 }$ and $h _ { * } ^ { \beta } \asymp A _ { n } ( \tau )$ , substitution into (D.9) proves (D.7).

Completion of the proofs of Theorems 2.5 and 2.6. The two lower bounds are complementary. The uniform covariate distribution and the replicated family give

$$
\begin{array} { r } { \mathcal { R } _ { n , p , \kappa } ^ { \star } \gtrsim A _ { n } ( \tau ) ^ { \kappa } , } \end{array}\tag{D.11}
$$

using Jensen’s inequality together with the first-moment decoding bound. For $p > 1$ , the hard family lies inside the prescribed density bounds, Proposition $\mathrm { C . 4 ( i i ) }$ gives the required $W _ { p }$ attenuation estimate, and $W _ { p } \geq W _ { 1 }$ preserves the separation.

For the second lower bound, take $n \geq 2$ , choose two covariate points $x _ { 0 } , x _ { 1 }$ a fixed distance apart and let

$$
P _ { X } = ( 1 - n ^ { - 1 } ) \delta _ { x _ { 0 } } + n ^ { - 1 } \delta _ { x _ { 1 } } .
$$

Put the uniform distribution at $x _ { 0 }$ and choose two fixed smooth perturbations of it, $Q _ { + } , Q .$ at $x _ { 1 }$ , with $\Delta = W _ { p } ( Q _ { + } , Q _ { - } ) > 0$ small enough to satisfy the conditional H¨older condition, any imposed Besov bound, and, when $p > 1$ , the stated density bounds. With probability $( 1 - n ^ { - 1 } ) ^ { n } \geq 1 / 4$ , no observation has covariate $x _ { 1 }$ , in which case the two data-generating laws coincide. For any estimated distribution $R ,$ the triangle inequality gives

$$
\frac { W _ { p } ^ { \kappa } ( R , Q _ { + } ) + W _ { p } ^ { \kappa } ( R , Q _ { - } ) } { 2 } \geq ( \Delta / 2 ) ^ { \kappa } .
$$

Hence

$$
\begin{array} { r } { \mathcal { R } _ { n , p , \kappa } ^ { \star } \gtrsim n ^ { - 1 } . } \end{array}\tag{D.12}
$$

Together, (D.7) and (D.12), with the lower bound under the uniform covariate distribution above, establish Theorems 2.5 and 2.6 for every fixed finite κ. ■

Appendix E. Bounds for higher Wasserstein orders. We prove the $W _ { p }$ bound used in Theorem 2.6.

E.1. $W _ { p }$ distribution estimation. We use the negative norm (A.2) and the transport comparison in Proposition A.2. Orthonormality across resolutions controls the empirical wavelet divergence field and gives the sharp logarithmic factor in dimension two.

Proposition E.1 $( W _ { p }$ unconditional rates). Consider $\mathcal { Q } _ { d , \tau , r , v } ^ { ( p ) } ( \underline { { q } } , \overline { { q } } )$ as defined in Section ${ \it 2 . 2 , }$ with fixed model parameters. Fix $1 \leq \kappa < \infty$ . Then, for all suficiently large m,

$$
\operatorname* { i n f } _ { \widehat { Q } } \operatorname* { s u p } _ { Q \in \mathcal { Q } _ { d , \tau , r , v } ^ { ( p ) } ( \underline { { q } } , \overline { { q } } ) } \left\{ \mathbb { E } W _ { p } ^ { \kappa } ( \widehat { Q } , Q ) \right\} ^ { 1 / \kappa } \asymp a _ { d , \tau } ( m ) .\tag{E.1}
$$

Proof. Let $Y _ { 1 } , \dots , Y _ { m }$ be i.i.d. with density $q ,$ let $q { \mathrm { } } J$ be the population wavelet projection through level $J ,$ , and let $\widetilde { q } { \boldsymbol { J } }$ be the corresponding estimator with empirical wavelet coeficients. The scaling space contains constants, so $\textstyle \int { \widetilde { q } } J = 1$

Let $\widehat { q } _ { J }$ be the measurable approximate projection of $\widetilde { q } _ { J }$ onto the set of $L ^ { p }$ probability densities, using the rule of Appendix A in the norm $\| \cdot \| _ { - 1 , p }$ with tolerance $m ^ { - 1 }$ , and write ${ \widehat { Q } } _ { J } = { \widehat { q } } _ { J } d y$ . Since the true density is admissible, $\left( \mathrm { A . 4 } \right)$ and Proposition A.2 give

$$
\left\{ \mathbb { E } W _ { p } ^ { \kappa } ( \widehat { Q } _ { J } , Q ) \right\} ^ { 1 / \kappa } \lesssim \| q - q _ { J } \| _ { - 1 , p } + \left\{ \mathbb { E } \| \widetilde { q } _ { J } - q _ { J } \| _ { - 1 , p } ^ { \kappa } \right\} ^ { 1 / \kappa } + m ^ { - 1 } .
$$

For each tensor-product wavelet $\psi \in \Psi _ { j }$ , choose a coordinate i containing a detail factor and set

$$
V _ { \psi } ( z ) = e _ { i } \int _ { 0 } ^ { z _ { i } } \psi ( z _ { 1 } , \dots , z _ { i - 1 } , s , z _ { i + 1 } , \dots , z _ { d } ) d s .
$$

The detail factor has zero integral, so its primitive vanishes beyond its support. Thus div $V _ { \psi } = \psi$ and $V _ { \psi } \cdot n = 0$ on the boundary of the cube. These fields have support diameter $O ( 2 ^ { - j } )$ bounded overlap at each level, and $\| V _ { \psi } \| _ { \infty } \lesssim 2 ^ { j ( d / 2 - 1 ) }$ . This is the wavelet field construction of [37, Section 3.1, Lemmas 3 and 5]. Finite overlap gives

$$
\left\| \sum _ { \psi \in \Psi _ { j } } c _ { \psi } V _ { \psi } \right\| _ { L ^ { p } } \lesssim 2 ^ { - j } 2 ^ { j d ( 1 / 2 - 1 / p ) } \| c \| _ { \ell ^ { p } } .
$$

The density upper bound implies $| \beta _ { q , \psi } | \leq \overline { { q } } \| \psi \| _ { 1 } \lesssim 2 ^ { - j d / 2 }$ , hence $\| \beta _ { q , j } \| _ { \ell ^ { p } } \lesssim 2 ^ { j d ( 1 / p - 1 / 2 ) }$ When $d > 2$ and $\tau > 0$ , the Besov coeficient bound (A.1) and the class restriction $r \geq p$ instead give $\| \beta _ { q , j } \| _ { \ell ^ { p } } \lesssim 2 ^ { - j \tau } 2 ^ { j d ( 1 / p - 1 / 2 ) }$ . The fields $\begin{array} { r } { \sum _ { j > J } \sum _ { \psi \in \Psi _ { j } } \beta _ { q , \psi } V _ { \psi } } \end{array}$ therefore converge absolutely in $L ^ { p }$ . Since the boundary wavelet projections converge to q in $L ^ { p } ,$ integration by parts for the finite sums passes to every test function in $W ^ { 1 , p ^ { \prime } } ( \Omega _ { d } )$ . The limiting field has divergence $q - q _ { J }$ and zero normal flux in this weak sense. Consequently,

$$
\lVert q - q _ { J } \rVert _ { - 1 , p } \lesssim \sum _ { j > J } 2 ^ { - j } 2 ^ { j d ( 1 / 2 - 1 / p ) } \lVert \beta _ { q , j } \rVert _ { \ell ^ { p } } ,
$$

which yields

$$
\begin{array} { r } { \| q - q _ { J } \| _ { - 1 , p } \lesssim \left\{ 2 ^ { - J } , \begin{array} { l l } { d \leq 2 \mathrm { ~ o r ~ } \tau = 0 , } \\ { 2 ^ { - J ( \tau + 1 ) } , } & { d > 2 , \tau > 0 . } \end{array} \right. } \end{array}\tag{E.2}
$$

For the empirical detail error, define

$$
K _ { J } ( z , y ) = \sum _ { j = j _ { 0 } } ^ { J } \sum _ { \psi \in \Psi _ { j } } V _ { \psi } ( z ) \psi ( y ) , \qquad G _ { J } ( z ) = \frac { 1 } { m } \sum _ { a = 1 } ^ { m } \{ K _ { J } ( z , Y _ { a } ) - \mathbb { E } K _ { J } ( z , Y _ { a } ) \} .
$$

The divergence of $G _ { J }$ is the detail part of $\widetilde { q } _ { J } - q _ { J }$ , and $G _ { J }$ has zero normal component. Write $\begin{array} { r } { B _ { J } = \sum _ { j = j _ { 0 } } ^ { J } 2 ^ { j ( d - 2 ) } } \end{array}$ . Orthonormality in y and finite overlap give, for each $z ,$

$$
\int _ { \Omega _ { d } } | K _ { J } ( z , y ) | ^ { 2 } d y = \sum _ { j = j _ { 0 } } ^ { J } \sum _ { \psi \in \Psi _ { j } } | V _ { \psi } ( z ) | ^ { 2 } \lesssim B _ { J } .
$$

Also, $\begin{array} { r } { \sum _ { j = j _ { 0 } } ^ { J } \sum _ { \psi \in \Psi _ { i } } | \psi ( y ) | ^ { 2 } \lesssim 2 ^ { J d } } \end{array}$ , so Cauchy–Schwarz gives $| K _ { J } ( z , y ) | ^ { 2 } \lesssim B _ { J } 2 ^ { J d }$ . Set $t =$ max $\{ 2 , p , \kappa \}$ and $\dot { Z _ { J } } ( z , Y ) = K _ { J } ( z , Y ) - \mathbb { E } K _ { J } ( z , Y )$ . Under $q \leq \overline { { q } }$ , these bounds imply

$$
\begin{array} { r } { { \mathbb E } | Z _ { J } ( z , Y ) | ^ { 2 } \lesssim B _ { J } , \qquad { \mathbb E } | Z _ { J } ( z , Y ) | ^ { t } \lesssim _ { t } B _ { J } ^ { t / 2 } 2 ^ { J d ( t / 2 - 1 ) } . } \end{array}
$$

Rosenthal’s inequality [43], applied componentwise to the independent observations, therefore gives

$$
\| G _ { J } ( z ) \| _ { L ^ { t } ( \mathbb { P } ) } \lesssim _ { p , \kappa } \left( \frac { B _ { J } } { m } \right) ^ { 1 / 2 } \left\{ 1 + \left( \frac { 2 ^ { J d } } { m } \right) ^ { 1 / 2 - 1 / t } \right\} \lesssim _ { p , \kappa } \left( \frac { B _ { J } } { m } \right) ^ { 1 / 2 } , \qquad 2 ^ { J d } \leq m .
$$

For $t = 2$ , the same bound follows from the variance identity. Since $t \geq$ max $\{ p , \kappa \}$ , Minkowski’s inequality gives

$$
\left\| \| G _ { J } \| _ { L ^ { p } ( d z ) } \right\| _ { L ^ { \kappa } ( \mathbb { P } ) } \leq \left\| \| G _ { J } \| _ { L ^ { t } ( \mathbb { P } ) } \right\| _ { L ^ { p } ( d z ) } \lesssim _ { p , \kappa } \left( \frac { B _ { J } } { m } \right) ^ { 1 / 2 } .
$$

The scaling part of $\widetilde { q } _ { J } - q _ { J }$ has mean zero because both projections have integral one and every detail wavelet has zero integral. Poincar´e’s inequality bounds its negative norm by its $L ^ { p }$ norm. The scaling family is fixed and finite, and its bounded functions have empirical coeficient errors of order $m ^ { - 1 / 2 }$ in every fixed finite moment. Combining this bound with the divergence field $G _ { J }$ yields

$$
\left\{ \mathbb { E } \Vert \widetilde { q } _ { J } - q _ { J } \Vert _ { - 1 , p } ^ { \kappa } \right\} ^ { 1 / \kappa } \lesssim _ { p , \kappa } \left[ \frac { 1 } { m } \left( 1 + \sum _ { j = j _ { 0 } } ^ { J } 2 ^ { j ( d - 2 ) } \right) \right] ^ { 1 / 2 } .\tag{E.3}
$$

For $d \leq 2$ , take $J = \left\lfloor ( \log _ { 2 } m ) / 2 \right\rfloor$ . Substituting (E.2) and (E.3) into the preceding error decomposition gives $m ^ { - 1 / 2 }$ when $d = 1$ and $\{ \log ( e m ) / m \} ^ { 1 / 2 }$ when $d = 2$ . For $d > 2$ , take $J = \left\lfloor ( \log _ { 2 } m ) / ( 2 \tau + d ) \right\rfloor$ , which balances the bias $2 ^ { - J ( \tau + 1 ) }$ and stochastic error $m ^ { - 1 / 2 } 2 ^ { J ( \dot { d } - 2 ) / 2 }$ including when $\tau = 0$ . Both choices satisfy $2 ^ { J d } \leq m$ and yield the upper bound $C _ { p , \kappa } a _ { d , \tau } ( m )$

For the finitely many positive sample sizes for which $J < j _ { 0 }$ , use a fixed probability law on the cube. Boundedness of $W _ { p }$ then extends the moment upper bound to every integer $m \geq 1$ after increasing the constant.

For the lower bound, use the hypercube in Proposition C.4. Part (i) places the family in the prescribed density bounds and Besov ball after decreasing the fixed perturbation amplitude if necessary. Since $W _ { p } \geq W _ { 1 }$ , the separation in part (iv) gives a first-moment Bayes lower bound of order $a _ { d , \tau } ( m )$ . Jensen’s inequality raises this to order $a _ { d , \tau } ( m ) ^ { \kappa }$ for the $W _ { p } ^ { \kappa }$ risk. This proves (E.1) and hence the unconditional statement (2.19). ■

Appendix F. Conditional Brenier map estimation. We establish the stability and measurability needed for the upper bound in Theorem 2.7, then prove the lower bound using inverse potentials.

## F.1. Stability and measurability.

Lemma F.1 (Uniform Brenier stability). Fix $P \in \mathcal { P } _ { \beta , \tau , r , v } ^ { \mathrm { B r } } ( \overline { { q } } , \Lambda )$ and write

$$
\lambda _ { 0 } = ( \overline { { { q } } } \Lambda ^ { d _ { y } - 1 } ) ^ { - 1 } .
$$

Then, uniformly in $x \in S _ { X }$

$$
q _ { x } ( y ) \geq \Lambda ^ { - d _ { y } } \quad f o r \ a . e . \ y \in \mathcal { Y } ,\tag{F.1}
$$

and for every probability law $\widetilde { Q }$ on $\mathcal { V }$

$$
\| T _ { \rho \to \widetilde { Q } } - T _ { x } \| _ { L ^ { 2 } ( \rho ) } ^ { 2 } \le \frac { \Lambda } { \lambda _ { 0 } } W _ { 2 } ^ { 2 } ( \widetilde { Q } , Q _ { x } ) .\tag{F.2}
$$

In addition, every measurable map $S : \mathcal { V } \to \mathcal { V }$ satisfies

$$
W _ { 2 } ^ { 2 } ( S _ { \# } \rho , Q _ { x } ) \leq \lVert S - T _ { x } \rVert _ { L ^ { 2 } ( \rho ) } ^ { 2 } .\tag{F.3}
$$

Proof. By the almost everywhere change of variables identity for the Brenier map [45],

$$
q _ { x } ( T _ { x } ( z ) ) \operatorname* { d e t } D T _ { x } ( z ) = 1 .
$$

Since $0 \preceq D T _ { x } \preceq \Lambda I$ and $q _ { x } \leq \overline { { q } } .$ , we have det $D T _ { x } \geq \overline { { q } } ^ { - 1 }$ almost everywhere, so the smallest eigenvalue of $D T _ { x }$ is at least $\lambda _ { 0 }$ . Thus the Brenier potential is uniformly λ -strongly convex and Λ-smooth. The map $T _ { x }$ is continuous, and its compact image has full $Q _ { x } { \mathrm { - m a s s } } ;$ because $q _ { x } > 0$ almost everywhere, that image must be the whole cube. Since the preceding bounds make $T _ { x }$ bi-Lipschitz, det $D T _ { x } \leq \Lambda ^ { d _ { y } }$ in the same identity gives (F.1).

Equation (F.2) now follows from [4, Corollary 1], with fixed source $\rho ,$ strong convexity constant $\lambda _ { 0 }$ , and smoothness constant Λ. The estimated target $\widetilde { Q }$ may be any probability law on the cube. Finally, $( S , T _ { x } ) _ { \# } \rho$ couples $S _ { \# } \rho$ and $Q _ { x } .$ , which proves (F.3). ■

For the plugin estimators, consider the Brenier map operator

$$
\mathsf { B } : \mathcal { P } ( \mathcal { Y } ) \to L ^ { 2 } ( \rho ; \mathbb { R } ^ { d _ { \boldsymbol { y } } } ) , \qquad \mathsf { B } ( \nu ) = T _ { \rho \to \nu } .
$$

If $\nu _ { k } \to \nu$ in $W _ { 2 }$ , qualitative stability of optimal maps [57, Corollary 5.23] gives ${ \sf B } ( \nu _ { k } ) \to { \sf B } ( \nu )$ in ρ-probability, since $\rho$ is absolutely continuous. All maps take values in the cube, so the convergence also holds in $L ^ { 2 } ( \rho )$ . Thus B is continuous and Borel measurable. The distribution estimator in Theorem 2.6 is measurable by construction. It follows by composition that the plugin estimator $\widehat { T } _ { x } = \mathsf { B } ( \widehat { Q } _ { x } )$ is measurable as an $L ^ { 2 } ( \rho )$ -valued function.

F.2. Transport families from inverse potentials. For the lower bound, we parameterize the inverse Brenier map directly. The target density is then an explicit determinant, avoiding composition with an inverse map.

Lemma F.2 (Brenier chart based on inverse potentials). Let $U , V \in C _ { c } ^ { 2 } ( ( 0 , 1 ) ^ { d } )$ and suppose

$$
\| D ^ { 2 } U \| _ { \mathrm { o p , \infty } } \vee \| D ^ { 2 } V \| _ { \mathrm { o p , \infty } } \le \eta < 1 .
$$

Define

$$
S _ { U } = \mathrm { I d } + \nabla U , \qquad T _ { U } = S _ { U } ^ { - 1 } , \qquad q _ { U } = \mathrm { d e t } ( I + D ^ { 2 } U ) .
$$

Then the following statements hold.

(i) $S _ { U }$ is a difeomorphism of the cube onto itself, $T _ { U }$ is the gradient of a convex potential, and

$$
( T _ { U } ) _ { \# } \rho = q _ { U } d y .
$$

Moreover

$$
( 1 - \eta ) ^ { d } \leq q _ { U } \leq ( 1 + \eta ) ^ { d } , \qquad ( 1 + \eta ) ^ { - 1 } I \preceq D T _ { U } \preceq ( 1 - \eta ) ^ { - 1 } I .
$$

(ii) The corresponding target laws satisfy

$$
H ^ { 2 } ( q _ { U } d y , q _ { V } d y ) \lesssim _ { \eta } \| D ^ { 2 } ( U - V ) \| _ { 2 } ^ { 2 } .\tag{F.4}
$$

(iii) The Brenier maps satisfy the two-sided comparison

$$
\lVert \boldsymbol { T } \boldsymbol { U } - \boldsymbol { T } _ { V } \rVert _ { L ^ { 2 } ( \rho ) } ^ { 2 } \asymp _ { \eta } \lVert \nabla \boldsymbol { U } - \nabla \boldsymbol { V } \rVert _ { 2 } ^ { 2 } .\tag{F.5}
$$

Proof. Extend U by zero to $\mathbb { R } ^ { d }$ . Then $\Psi _ { U } ( y ) = | y | ^ { 2 } / 2 + U ( y )$ is globally $( 1 - \eta )$ -strongly convex and quadratic outside a compact set. For every $z \in \mathbb { R } ^ { d } .$ , the function $\Psi _ { U } ( y ) - z \cdot y$ has a unique minimizer, so $S _ { U } = \nabla \Psi _ { U }$ is a bijection of $\mathbb { R } ^ { d }$ . Since $D S _ { U } = I + D ^ { 2 } U$ is positive definite, the local inverse theorem makes $S _ { U }$ a global $C ^ { 1 }$ difeomorphism. Since $S _ { U }$ fixes every point outside the cube, injectivity and surjectivity imply that it maps the cube onto itself; see also [28, Lemma 38]. Its inverse is $T _ { U } = \nabla \Psi _ { U } ^ { * }$ and is therefore a Brenier map. Change of variables gives

$$
( S _ { U } ) _ { \# } ( q _ { U } d y ) = \rho ,
$$

which is equivalent to $( T _ { U } ) _ { \# } \rho = q _ { U } d y$ . The eigenvalue bounds on $I + D ^ { 2 } U$ immediately give the stated determinant and curvature bounds.

The determinant is Lipschitz on the compact matrix ball $\{ A : \| A \| _ { \mathrm { o p } } \leq \eta \}$ . Together with uniform positivity, this gives

$$
H ^ { 2 } ( q _ { U } d y , q _ { V } d y ) \lesssim \| q _ { U } - q _ { V } \| _ { 2 } ^ { 2 } \lesssim \| D ^ { 2 } ( U - V ) \| _ { 2 } ^ { 2 } ,
$$

and hence (F.4).

For (F.5), change variables $z = S _ { U } ( y )$ and put $y ^ { \prime } = S _ { V } ^ { - 1 } ( S _ { U } ( y ) )$ . Then

$$
S _ { V } ( y ) - S _ { V } ( y ^ { \prime } ) = \nabla V ( y ) - \nabla U ( y ) .
$$

Uniform bi-Lipschitz control of $S _ { V }$ gives

$$
\begin{array} { r } { | y - y ^ { \prime } | \asymp _ { \eta } | \nabla U ( y ) - \nabla V ( y ) | . } \end{array}
$$

Since $q _ { U } \asymp 1$ 2

$$
\| T _ { U } - T _ { V } \| _ { L ^ { 2 } ( \rho ) } ^ { 2 } = \int | y - y ^ { \prime } | ^ { 2 } q _ { U } ( y ) d y \asymp _ { \eta } \| \nabla U - \nabla V \| _ { 2 } ^ { 2 } .
$$

Proof of Theorem 2.7 (lower bound). For the map lower bound, fix model parameters in the ranges of Assumption 2.4 and take $P _ { X }$ uniform on $[ 0 , 1 ] ^ { d _ { x } }$ . Use the covariate bumps from Appendix D.2: $M _ { x } \times h ^ { - d _ { x } }$ smooth functions $0 \leq b _ { a } \leq 1$ with supports of diameter $O ( h )$ separated by distances comparable to $h ,$ plateau regions of volume comparable to $h ^ { d _ { x } }$ , and H¨older moduli bounded by $C h ^ { - \beta }$ . For each $a ,$ let $\theta _ { a }$ index an independent copy of the potential family in the response dimension in Lemma C.3, and set

$$
U _ { \Theta , x } ( y ) = \sum _ { a } b _ { a } ( x ) U _ { \theta _ { a } } ( y ) , \qquad S _ { \Theta , x } = \mathrm { I d } + \nabla U _ { \Theta , x } , \qquad T _ { \Theta , x } = S _ { \Theta , x } ^ { - 1 } ,\tag{F.6}
$$

with target density

$$
q _ { \Theta , x } ( y ) = \operatorname* { d e t } \{ I + D _ { y } ^ { 2 } U _ { \Theta , x } ( y ) \} .\tag{F.7}
$$

Set

$$
B _ { J } = \sum _ { j = j _ { 0 } } ^ { J } 2 ^ { j ( d _ { y } - 2 ) } .
$$

We verify membership in the Brenier class. By Lemma C.3,

$$
\lVert D _ { y } ^ { 2 } U _ { \Theta , x } \rVert _ { \infty } \lesssim \varepsilon 2 ^ { J d _ { y } / 2 } .
$$

If the fixed quantity $\varepsilon 2 ^ { J d _ { y } / 2 }$ is suficiently small, Lemma F.2 supplies the density and curvature bounds. When $\tau > 0$ , choose an integer ℓ with $\tau < \ell \leq K - 2$ . Because the determinant is polynomial in the Hessian and the Hessians stay in a fixed small matrix ball,

$$
\begin{array} { r } { \| q _ { \Theta , x } - 1 \| _ { \infty } \lesssim \varepsilon 2 ^ { J d _ { y } / 2 } , \qquad \| D _ { y } ^ { \ell } ( q _ { \Theta , x } - 1 ) \| _ { \infty } \lesssim _ { \ell } \varepsilon 2 ^ { J ( d _ { y } / 2 + \ell ) } . } \end{array}
$$

$\mathrm { A s }$ the perturbation is supported in the fixed interior cube $K _ { 0 } .$ , the characterization by finite diferences implies

$$
\| q _ { \Theta , x } - 1 \| _ { B _ { r , v } ^ { \tau } } \lesssim \varepsilon 2 ^ { J ( \tau + d _ { y } / 2 ) } , \qquad 1 \leq r , v \leq \infty .\tag{F.8}
$$

Thus, with the convention $\tau = 0$ when no additional smoothness in the response variable is imposed, it is enough to impose

$$
\varepsilon 2 ^ { J ( \tau + d _ { y } / 2 ) } \leq c _ { 0 } .\tag{F.9}
$$

The conditional $W _ { 2 }$ regularity follows from the coupling induced by the common source and (F.5). Lemma C.3 gives $\lVert \nabla U _ { \theta _ { a } } \rVert _ { 2 } \lesssim \varepsilon \sqrt { B _ { J } }$ . If $x , x ^ { \prime }$ are in the same active covariate bump,

$$
W _ { 2 } ( Q _ { \Theta , x } , Q _ { \Theta , x ^ { \prime } } ) \leq \lVert T _ { \Theta , x } - T _ { \Theta , x ^ { \prime } } \rVert _ { 2 } \lesssim \varepsilon \sqrt { B _ { J } } h ^ { - \beta } | x - x ^ { \prime } | ^ { \beta } .
$$

The same estimate holds when, say, $x ^ { \prime }$ lies outside all bump supports, since $b _ { a } ( x ^ { \prime } ) = 0$ . If both points lie outside, the conditional laws coincide. For points in diferent active bumps, the supports are separated by a distance comparable to $h ,$ and both maps lie within $C \varepsilon \sqrt { B _ { J } }$ of the identity. The conditional H¨older bound therefore follows provided

$$
\varepsilon { \sqrt { B _ { J } } } \leq c _ { 1 } h ^ { \beta } .\tag{F.10}
$$

Flipping one coordinate in block a changes $D _ { y } ^ { 2 } U _ { \Theta , x }$ by an $L ^ { 2 }$ term of size $O ( \varepsilon b _ { a } ( x ) )$ . Lemma F.2(ii) consequently gives the one-sample testing estimate

$$
H ^ { 2 } ( P _ { \Theta } , P _ { \Theta ^ { ( a , j , k ) } } ) \lesssim h ^ { d _ { x } } \varepsilon ^ { 2 } .\tag{F.11}
$$

On the plateau of covariate bump $^ { a , }$ Lemma F.2(iii), together with (C.6), gives the following separation for arbitrary vertices:

$$
\int _ { \mathcal { X } } \| T _ { \Theta , x } - T _ { \Theta ^ { \prime } , x } \| _ { L ^ { 2 } ( \rho ) } ^ { 2 } d x \gtrsim h ^ { d _ { x } } \varepsilon ^ { 2 } \sum _ { a , j , k } 2 ^ { - 2 j } \mathbf { 1 } \{ \theta _ { a , j , k } \neq \theta _ { a , j , k } ^ { \prime } \} .\tag{F.12}
$$

Decode any map estimator using the metric

$$
d ( T , T ^ { \prime } ) = \left\{ \int _ { \mathcal { X } } \| T _ { x } - T _ { x } ^ { \prime } \| _ { L ^ { 2 } ( \rho ) } ^ { 2 } d x \right\} ^ { 1 / 2 } .
$$

If $n h ^ { d _ { x } } \varepsilon ^ { 2 }$ is bounded by a suficiently small constant, Hellinger tensorization and Lemma C.1 give a uniformly positive coordinate error probability. Since $M _ { x } \times h ^ { - d _ { x } }$ covariate blocks are present, (F.12) implies

$$
\operatorname* { i n f } _ { \widehat { T } } \operatorname* { s u p } _ { \Theta } \mathbb { E } \int _ { \mathcal { X } } \| \widehat { T } _ { x } - T _ { \Theta , x } \| _ { L ^ { 2 } ( \rho ) } ^ { 2 } d x \gtrsim \varepsilon ^ { 2 } B _ { J } .\tag{F.13}
$$

Take

$$
\varepsilon \asymp ( n h ^ { d _ { x } } ) ^ { - 1 / 2 }
$$

and choose J as large as allowed by (F.9). As in (C.13),

$$
\varepsilon \sqrt { B _ { J } } \asymp a _ { d _ { y } , \tau } ( n h ^ { d _ { x } } ) .
$$

Choosing h according to (2.13) also enforces (F.10), and gives

$$
\varepsilon ^ { 2 } B _ { J } \asymp A _ { n } ( \tau ) ^ { 2 } .
$$

The fixed construction constants can be chosen small enough that the density upper bound and the prescribed map curvature constant hold uniformly. This proves the map lower bound in every response dimension with the same family based on inverse potentials. ■

Appendix G. Details of the numerical experiments. This appendix gives the numerical procedures for Sections 2, 3, and 4. For the conditional estimation and multiscale illustrations, the reported repetitions use seeds independent of the preliminary simulations. The experiment code and simulation records are not publicly available.

Pointwise 95% percentile bootstrap intervals use 4,000 resamples of independent repetitions, preserving paired signs, all queries within a dataset, and pairing between estimators evaluated on the same repetitions. Rare covariate simulations are resampled within count strata before recombination with exact probabilities.

For losses evaluated using numerical lower and upper bounds, plotted curves use mean interval midpoints. The shaded interval runs from the 2.5th percentile of the bootstrap mean lower endpoints to the 97.5th percentile of the bootstrap mean upper endpoints. These intervals combine approximate pointwise bootstrap uncertainty with numerical bracketing. The probabilistic transport lower bounds in Subsection G.1 have an additional failure probability $\delta ,$ separate from the bootstrap level. Quadrature and resolution comparisons are sensitivity checks rather than uniform error bounds.

G.1. Conditional estimation across response dimensions. For Figure 1, X is uniform on [0, 1], and the conditional response density on $[ 0 , 1 ] ^ { d }$ is

$$
q _ { x } ( y ) = 1 + \theta ( x ) ( 2 y _ { 1 } - 1 ) , \qquad \theta ( x ) = 0 . 8 \sin ( 2 \pi x ) .
$$

The remaining response coordinates are independent uniforms. These densities satisfy $0 . 2 \leq$ $q _ { x } ~ \leq ~ 1 . 8$ . Projection onto the first coordinate and a coupling with identical remaining coordinates give $W _ { 1 } ( Q _ { x } , Q _ { x ^ { \prime } } ) = \vert \theta ( x ) - \theta ( x ^ { \prime } ) \vert / 6$ , so the conditional laws are Lipschitz with a constant independent of d.

For $n = 2 ^ { 9 } , \ldots , 2 ^ { 1 5 }$ , the grid estimator partitions the covariate interval into $K _ { d } ( n )$ equal cells, where

$$
K _ { 1 } ( n ) = { \lceil n ^ { 1 / 3 } \rceil } , \qquad K _ { 2 } ( n ) = \left\lceil { \{ n / \log ( e n ) \} ^ { 1 / 3 } } \right\rceil , \qquad K _ { 4 } ( n ) = { \lceil n ^ { 1 / 5 } \rceil } .
$$

It uses the empirical response distribution in the query cell, with the uniform distribution as the empty-cell fallback. The bandwidth for $d = 4$ uses the unrestricted response rate; this example does not assess the gain from response smoothness. We use 2,048, 128, and 24 independent repetitions in dimensions 1, 2, and 4, respectively.

Each repetition uses an independent uniform query. Conditional on its cell $I ,$ the retained count has distribution $\mathrm { B i n } ( n , 1 / K _ { d } ( n ) )$ , and the retained responses are independent with density $1 + \bar { \theta } _ { I } ( 2 y _ { 1 } - 1 )$ , where $\begin{array} { r } { \bar { \theta } _ { I } = | I | ^ { - 1 } \int _ { I } \theta ( x ) } \end{array}$ dx. We simulate this count and the retained responses directly, which has the same law as retaining observations from a full i.i.d. training sample. The cell parameter is evaluated analytically, and the first response coordinate is generated by inverting $F _ { I } ( t ) = ( 1 - \bar { \theta } _ { I } ) t + \bar { \theta } _ { I } t ^ { 2 }$ . The estimator itself uses only the retained responses. Averaging the losses over independent training samples and queries estimates the expected integrated risk. Common random variables couple the simulations across sample sizes, and bootstrap resampling preserves each complete trajectory. No empty query cell occurred in the reported repetitions.

Panel (a) evaluates losses against the population conditional law $Q _ { x }$ . Panel (b) uses the same empirical cell distributions but evaluates their sampling error against $Q _ { I } .$ , the population law pooled over the query cell. The population pooling bias is $W _ { 1 } ( Q _ { I } , Q _ { x } ) = \vert \bar { \theta } _ { I } - \theta ( x ) \vert / 6$ For K equal cells, its integrated value is asymptotic to $2 / ( 1 5 K )$ and contributes to the total error in panel (a).

Scalar losses use analytic quantile integration. In dimensions two and four, partition the target cube into cells of side $1 / M$ and compute their exact masses, centroids $c _ { C } ,$ and within-cell variances $v _ { C }$ . A balanced transport linear program with source atoms $Y _ { i }$ and costs $\sqrt { | Y _ { i } - c _ { C } | ^ { 2 } + v _ { C } }$ gives an upper bound: independent lifting within each target cell and Jensen’s inequality produce a feasible continuous transport plan, as in Subsection G.3.

For the lower bound, target dual values $b _ { C }$ define $a _ { i } = \mathrm { m i n } _ { C } \{ | Y _ { i } - c _ { C } | - b _ { C } \}$ . These values are 1-Lipschitz on the source atoms, and $\phi ( z ) = \mathrm { m a x } _ { i } \{ a _ { i } - | z - Y _ { i } | \}$ is a globally 1-Lipschitz extension. Thus $\begin{array} { r } { N ^ { - 1 } \sum _ { i } a _ { i } - \int \phi d Q _ { x } } \end{array}$ is a lower bound when the cell contains $N$ observations. To evaluate the integral, partition the target cube into strata of side $1 / H$ with exact probabilities $w _ { S } = Q _ { x } ( S )$ , and draw one independent observation $Z _ { S }$ from each conditional law $Q _ { x } ( \cdot \mid S )$ . The draws are independent of the training data and the computed potential. Subtracting

$$
\left\{ \frac { d \log ( 1 / \delta ) } { 2 H ^ { 2 } } \sum _ { S } w _ { S } ^ { 2 } \right\} ^ { 1 / 2 } , \qquad \delta = 1 0 ^ { - 7 } ,
$$

from $\begin{array} { r } { N ^ { - 1 } \sum _ { i } a _ { i } - \sum _ { S } w _ { S } \phi ( Z _ { S } ) } \end{array}$ gives a lower bound with failure probability at most $\delta ,$ by Hoefding’s inequality and the diameter $\sqrt { d } / H$ of each stratum. The same construction with

$Q _ { I }$ in place of $Q _ { x }$ gives the sampling loss bounds in panel (b), using independent integration draws.

We use $( M , H ) = ( 1 2 8 , 2 5 6 )$ in dimension two and (12, 16) in dimension four. Primal and dual objectives are checked to agree within $1 0 ^ { - 7 }$ , and a rounding allowance of $1 0 ^ { - 9 }$ is added to the upper bound and subtracted from the lower bound. Scalar losses are checked against independent numerical quadrature, and the continuous-target bounds are checked on examples with known transport costs. For panel (a) at $n = 2 ^ { 1 5 }$ , the diference between the mean upper and lower bounds is $3 . 0 \%$ and 15.7% of the mean upper bound in dimensions two and four, respectively.

Panel (b) multiplies each mean sampling loss by $\sqrt { m _ { n , d } }$ , where $m _ { n , d } = n / K _ { d } ( n )$ , and normalizes it at $n = 5 1 2$ . Writing ${ \bar { L } } _ { n } ^ { * }$ and ${ \bar { U } } _ { n } ^ { * }$ for the bootstrap mean lower and upper endpoints, the normalized interval endpoints are

$$
\sqrt { \frac { m _ { n , d } } { m _ { 5 1 2 , d } } } \frac { \bar { L } _ { n } ^ { * } } { \bar { U } _ { 5 1 2 } ^ { * } } , \qquad \sqrt { \frac { m _ { n , d } } { m _ { 5 1 2 , d } } } \frac { \bar { U } _ { n } ^ { * } } { \bar { L } _ { 5 1 2 } ^ { * } } .
$$

Their 2.5th and 97.5th percentiles define the shaded band. At the normalization point, the curve and both endpoints are set to one. The reference curve is $\{ \log ( e m _ { n , 2 } ) / \log ( e m _ { 5 1 2 , 2 } ) \} ^ { 1 / 2 }$ . These fixed-law simulations do not determine the full conditional minimax logarithmic exponent.

G.2. Scalar distribution estimation. The grid estimator uses $\lceil n ^ { 1 / 3 } \rceil$ equal covariate cells and the empirical response distribution in each occupied cell. The empirical nearest neighbor estimator uses

$$
k = \operatorname* { m i n } \{ n , \operatorname* { m a x } ( 2 , \lceil C n ^ { 2 / 3 } \rceil ) \} , \qquad C = 4 .
$$

The occupancy estimator divides the sample into two halves. For each query, the first half selects the deepest dyadic cell at level $j$ such that every ancestor, at its respective level $l \leq j$ contains at least $\lceil 0 . 5 2 ^ { 2 l } \rceil$ observations. The empirical distribution of second half responses in the selected cell is then used for estimation. The empirical estimators use the uniform response distribution when an estimation cell is empty.

For the family in (3.7), the conditional distribution function is

$$
F _ { x , s } ( y ) = ( 1 - \alpha s x ) y + \alpha s x y ^ { 2 } .
$$

Responses are generated by inversion, with common uniform random variables for the two signs. For uniform covariates, we use 256 independent repetitions per sample size, each evaluating both signs, with the nearest neighbor constant fixed beforehand. Empirical $W _ { 1 }$ losses use analytic quantile integration, checked independently through the distribution functions.

Covariate integration uses exact probabilities for the two-point distribution. Under uniform covariates, grid and occupancy losses use 16 Gauss points on each localization interval, and nearest neighbor losses use 2,048 midpoints. Doubling these resolutions for the first 12 datasets at each sample size changes the mean fourth-moment risk by at most 0.1945% in the cases checked.

For the rare covariate design, let $\begin{array} { r } { K = \sum _ { i = 1 } ^ { n } \mathbf { 1 } \{ X _ { i } = 1 \} } \end{array}$ count observations at the rare atom in the full sample, before sample splitting. For the empirical estimators, we simulate datasets conditional on $K = k$ , for $k = 0 , \ldots , 1 6$ , placing the k rare observations uniformly among sample indices. Each sample size uses 5,632 conditional repetitions: 512 for each $k = 0 , \ldots , 8$ and 128 for each $k = 9 , \ldots , 1 6$ . Conditional mean losses are weighted by exact $\mathrm { B i n } ( n , 1 / n )$ probabilities. The omitted tail contribution is bounded by the maximal loss multiplied by $\operatorname* { P r } ( K > 1 6 )$ . The calculation is checked against 1,024 independent samples from the original joint law per sample size.

The normalizing flow has eight spline intervals on $[ 0 , 1 ]$ and a uniform base distribution. A network with two hidden layers of 16 units and SiLU activations takes $2 x - 1$ as input and supplies the spline widths, heights, and derivatives, each bounded below by $1 0 ^ { - 3 }$ . The inverse spline generates responses.

Each fit starts from the uniform distribution. We use 80% of the observations for likelihood fitting and the remaining 20% for validation. Adam uses batches of 256 observations sampled with replacement, 1,600 updates, an $\ell _ { 2 }$ penalty of $1 0 ^ { - 3 }$ , and a learning rate that decreases by a cosine schedule from 0.002 to $2 \times 1 0 ^ { - 5 }$ . Gradient norms are clipped at 5. The initial model and checkpoints at every 40 updates are compared by validation likelihood. The two signs are fitted separately with paired initializations, data splits, and minibatch indices.

We estimate flow risks from 256 paired datasets per sample size under uniform covariates. Under rare covariates, we use 128 repetitions conditional on $K = k$ for each $k = 0 , \ldots , 4 .$ , 64 for each $k = 5 , \ldots , 8$ , and 16 for each $k = 9 , \ldots , 1 6 .$ , giving 1,024 stratified datasets per sample size. The weighting and tail treatment are the same as for the empirical estimators.

For flow losses, we integrate the absolute diference between fitted and true cumulative distribution functions using 64 Gauss nodes on each spline interval. Covariate integration uses 64 Gauss nodes under uniform covariates and exact masses for the two-point design. We double both resolutions for the first eight uniform datasets at each sample size and the first eight datasets in each rare stratum $K = 0 , 1 , 2$ . The largest relative change in a mean fourth-moment risk is 0.0003% in these checks. When $K = 0$ , the two signs give identical training data and, under the paired training randomness, identical fitted flows.

Under uniform covariates, the sharp decrease in the flow’s fourth-moment risk in Figure 2(c) coincides with validation selecting the uniform initialization less often: it is selected in 53.1% of fits at $n = 2 5 6 , 2 . 5 \%$ at $n = 2 { , } 0 4 8$ , and none at $n = 4 { , } 0 9 6$

For the two-point distribution, write $\ell _ { a } = W _ { 1 } ( \hat { Q } _ { a } , Q _ { a , s } ) , a \in \{ 0 , 1 \}$ . With E including the average over signs, the occupancy risk decomposes as

$$
\mathcal { R } _ { n , 1 , 4 } = ( 1 - n ^ { - 1 } ) \mathbb { E } [ \ell _ { 0 } ^ { 4 } ] + n ^ { - 1 } \mathbb { E } [ \ell _ { 1 } ^ { 4 } \mathbf { 1 } \{ K = 0 \} ] + n ^ { - 1 } \mathbb { E } [ \ell _ { 1 } ^ { 4 } \mathbf { 1 } \{ K \geq 1 \} ] .\tag{G.1}
$$

On $K = 0 .$ the data cannot distinguish the two conditional laws at the rare atom. Their $W _ { 1 }$ distance is $\alpha / 3$ , so the triangle inequality and convexity imply the lower bound

$$
B _ { n } = \frac { 1 } { n } \left( 1 - \frac { 1 } { n } \right) ^ { n } \left( \frac { \alpha } { 6 } \right) ^ { 4 }\tag{G.2}
$$

for the second term in (G.1). An estimator using the uniform distribution at the unobserved atom attains this bound. An estimator that pools observations from a parent cell may incur a larger loss; accordingly, the dotted curve in Figure G.1 is a lower bound rather than the measured contribution.

At $n = 2 ^ { 1 5 }$ , the rare atom contributes 99.46% and 98.87% of the fourth-moment risks of the occupancy estimator and normalizing flow, respectively. The corresponding contributions from the unobserved rare atom are 8.69% and 36.39% of the total risks. The fitted flow laws at the rare atom remain close to uniform even when that atom is observed, so their risk also includes estimation error beyond the missed-event contribution.

![](images/0b8ec9debf3ab670b65ef2579b06761594127f3b40c4c29621fe1295430f3d22.jpg)  
Figure G.1. Fourth-moment risk of the occupancy estimator under rare covariates. The three contributions in (G.1) are multiplied by n. Both axes are logarithmic. The dotted curve is the lower bound $n B _ { n }$ from (G.2). Shaded bands are pointwise 95% bootstrap intervals, using 5,632 stratified repetitions at each sample size.

G.3. Multiscale transport separation. For the constructions in Figure 3, let $\psi _ { k }$ be the nonconstant orthonormal cosine modes on $[ 0 , 1 ] ^ { 2 }$ , and define

$$
I _ { j } = \{ k : 2 ^ { j - 1 } \leq \operatorname* { m a x } ( k _ { 1 } , k _ { 2 } ) < 2 ^ { j } \} , \qquad \varepsilon _ { J } = 0 . 0 2 5 2 ^ { - 2 J } .
$$

For each $J \in \{ 1 , 2 , 3 , 4 , 5 \}$ , we generate 32 independent pairs of sign vectors and form

$$
q ^ { ( a ) } = 1 + \varepsilon _ { J } \sum _ { k \in I } \sigma _ { k } ^ { ( a ) } \psi _ { k } , \qquad a \in \{ 1 , 2 \} .
$$

The finest level construction uses $I = I _ { J }$ , and the multiscale construction uses $\begin{array} { r } { I = \bigcup _ { j = 1 } ^ { J } I _ { j } } \end{array}$ The two constructions share signs on their common modes and coincide at $J = 1$ . The modes have zero mean, so the densities integrate to one. The cosine family permits exact cell averages in the transport calculations.

Both constructions satisfy $0 . 9 5 \leq q ^ { ( a ) } \leq 1$ .05 and $\varepsilon _ { J } 2 ^ { j } | I \cap I _ { j } | ^ { 1 / 2 } \leq 0 . 0 2 5 \sqrt { 3 } / 2$ for $1 \leq j \leq J$ giving positivity and the spectral block bound at $\tau = 1$

We compute exact cell averages on an $M ^ { 2 } \ { \mathrm { g r i d } }$ , with $M = 2 5 6$ for $J \le 4$ and $M = 5 1 2$ for $J = 5$ , and double M for the first 8 pairs. For $g = q ^ { ( 1 ) } - q ^ { ( 2 ) }$ , let $z _ { C }$ be the center of cell $C$ and set

$$
g _ { M } = \sum _ { C } \left( \int _ { C } g ( y ) d y \right) \delta _ { z _ { C } } .
$$

Moving the signed mass in each cell to its center gives

$$
\| g d y - g _ { M } \| _ { \mathrm { K R } } \leq \frac { \sqrt { 2 } } { 2 M } \| g \| _ { 1 } \leq \frac { \sqrt { 2 } } { 2 M } \| g \| _ { 2 } = \frac { \sqrt { 2 } } { 2 M } \| \delta \| _ { 2 } = : \eta ,
$$

where $\delta$ contains the diferences in the orthonormal cosine coeficients, including $\varepsilon _ { J } .$

Let $[ L _ { G } , U _ { G } ]$ be dual and primal bounds for transport of $g _ { M }$ on the grid with eight neighbors. Horizontal and vertical edges have cost $1 / M _ { ; }$ , and diagonal edges have cost $\sqrt { 2 } / M$ The Euclidean distance $d _ { E }$ and graph distance $d _ { G }$ satisfy

$$
d _ { E } \leq d _ { G } \leq \xi d _ { E } , \qquad \xi = \sqrt { 1 + ( \sqrt { 2 } - 1 ) ^ { 2 } } .
$$

Indeed, for coordinate displacements $a \geq b \geq 0$ , the graph distance is $a + ( \sqrt { 2 } - 1 ) b ;$ Cauchy– Schwarz gives the upper comparison. Thus $L _ { G } / \xi$ and $U _ { G }$ bound the Euclidean transport cost of $g _ { M }$

We also compute bounds $[ L _ { E } , U _ { E } ]$ by grouping the positive and negative masses of $g _ { M }$ After normalizing each part to a probability measure, we recursively split the group with largest mass-weighted spatial variance along its largest-variance coordinate at a weighted median, using at most 512 groups for each measure. Each group is represented by its mass, centroid, and mean squared distance from the centroid. If $c _ { i } , d _ { j }$ are the centroids, $v _ { i } , w _ { j }$ the within-group variances, and $\pi _ { i j }$ a feasible transport plan between group masses, independent lifting within each group pair and Jensen’s inequality give the upper bound

$$
\sum _ { i , j } \pi _ { i j } \sqrt { | c _ { i } - d _ { j } | ^ { 2 } + v _ { i } + w _ { j } } .
$$

For the lower bound, target dual values $b _ { j }$ from the centroid transport problem define the globally 1-Lipschitz function $\begin{array} { r } { \phi ( z ) = \operatorname* { m i n } _ { j } \{ | z - d _ { j } | - b _ { j } \} } \end{array}$ . We integrate $\phi$ against the original discrete measures. The bounds are multiplied by the common transported mass. Atoms of normalized mass at most $1 0 ^ { - 1 5 }$ are omitted during grouping; their total mass is charged at the cube diameter, and residual and rounding allowances are added to the upper bound and subtracted from the lower bound.

Combining the discrete bounds and the signed discretization error yields

$$
\operatorname* { m a x } \{ 0 , L _ { G } / \xi - \eta , L _ { E } - \eta \} \le W _ { 1 } ( q ^ { ( 1 ) } d y , q ^ { ( 2 ) } d y ) \le \operatorname* { m i n } \{ U _ { G } + \eta , U _ { E } + \eta \} .\tag{G.3}
$$

Optimization uses a relative gap target of $1 \%$ and a limit of 12,000 iterations; the corrected bounds are retained even when the target gap is not achieved. Graph divergence residuals are corrected using a discrete Neumann Poisson solve, with any remaining residual charged at the graph diameter. Dual potentials are rescaled to satisfy the edge constraints. Centroid transport plans are accepted only when the maximum marginal residual is at most $1 0 ^ { - 8 }$ , with the resulting allowance included in the bounds. Costs, residuals, and feasibility are checked in double precision.

Independent checks against exact linear programming solutions cover 6 random cases and 2 multiscale cases. The largest relative graph primal–dual gap among the primary runs is $8 . 4 \%$ In the doubled-grid checks, interval midpoints change by at most 2.4%, and all corresponding intervals overlap.

For $J = 2 , 3 , 4 , 5$ , the intervals for the mean diference in separation between the multiscale and finest level constructions, normalized by $\varepsilon _ { J } .$ , lie strictly above zero after incorporating both numerical bounds and bootstrap uncertainty. $\mathrm { A t } ~ J = 1$ , the multiscale and finest level constructions coincide for each paired sign realization, so the diference in their true separations is zero. The positive width of its numerical interval comes from subtracting the separately computed lower and upper bounds.

## REFERENCES

[1] E. Acerbi and N. Fusco, An approximation lemma for $W ^ { 1 , p }$ functions, in Material Instabilities in Continuum Mechanics (Edinburgh, 1985–1986), Oxford University Press, New York, 1988, pp. 1–5.

[2] M. Al-Jarrah, B. Hosseini, N. Jin, M. Martino, and A. Taghvaei, Error analysis of triangular optimal transport maps for filtering. arXiv preprint arXiv:2510.19283, 2025, https://arxiv.org/abs/ 2510.19283.

[3] S. Balakrishnan, The fundamental limits of valid transport map estimation. arXiv preprint arXiv:2606.30574, 2026, https://arxiv.org/abs/2606.30574.

[4] S. Balakrishnan and T. Manole, Stability bounds for smooth optimal transport maps and their statistical implications, Electronic Journal of Statistics, 20 (2026), pp. 3078–3105.

[5] R. Baptista, A.-A. Pooladian, M. Brennan, Y. Marzouk, and J. Niles-Weed, Conditional simulation via entropic optimal transport: Toward non-parametric estimation of conditional Brenier maps, in Proceedings of the 28th International Conference on Artificial Intelligence and Statistics (AISTATS), vol. 258 of Proceedings of Machine Learning Research, PMLR, 2025, pp. 4807–4815.

[6] J.-D. Benamou and Y. Brenier, A computational fluid mechanics solution to the Monge–Kantorovich mass transfer problem, Numerische Mathematik, 84 (2000), pp. 375–393.

[7] C. Ben<sup>´</sup> ezet, Z. Cheng, and S. Jaimungal <sup>´</sup> , Learning conditional distributions on continuous spaces, Journal of Machine Learning Research, 26 (2025), pp. 1–64.

[8] B. Bilodeau, D. J. Foster, and D. M. Roy, Minimax rates for conditional density estimation via empirical entropy, Annals of Statistics, 51 (2023), pp. 762–790.

[9] S. G. Bobkov and M. Ledoux, A simple Fourier analytic proof of the AKT optimal matching theorem, Annals of Applied Probability, 31 (2021), pp. 2567–2584.

[10] Y. Brenier, Polar factorization and monotone rearrangement of vector-valued functions, Communications on Pure and Applied Mathematics, 44 (1991), pp. 375–417.

[11] E. Brunel, F. Comte, and C. Lacour, Minimax estimation of the conditional cumulative distribution function, Sankhy¯a A, 72 (2010), pp. 293–330.

[12] C. Bulte, P. Scholl, and G. Kutyniok <sup>¨</sup> , Probabilistic neural operators for functional uncertainty quantification, Transactions on Machine Learning Research, (2025).

[13] J. Chemseddine, P. Hagemann, G. Steidl, and C. Wald, Conditional Wasserstein distances with applications in Bayesian OT flow matching, Journal of Machine Learning Research, 26 (2025), pp. 1–47.

[14] A. Cohen, Numerical Analysis of Wavelet Methods, vol. 32 of Studies in Mathematics and its Applications, North-Holland, Amsterdam, 2003.

[15] A. Cohen, I. Daubechies, and P. Vial, Wavelets on the interval and fast wavelet transforms, Applied and Computational Harmonic Analysis, 1 (1993), pp. 54–81.

[16] J. Dedecker and F. Merlevede<sup>\`</sup> , Behavior of the Wasserstein distance between the empirical and the marginal distributions of stationary α-dependent sequences, Bernoulli, 23 (2017), pp. 2083–2127.

[17] L. Diening, J. Malek, and M. Steinhauer<sup>´</sup> , On Lipschitz truncations of Sobolev functions (with variable exponent) and their selected applications, ESAIM: Control, Optimisation and Calculus of Variations, 14 (2008), pp. 211–232.

[18] V. Divol, Measure estimation on manifolds: an optimal transport approach, Probability Theory and Related Fields, 183 (2022), pp. 581–647.

[19] V. Divol, J. Niles-Weed, and A.-A. Pooladian, Optimal transport map estimation in general function spaces, Annals of Statistics, 53 (2025), pp. 963–988.

[20] C. Dombry, T. Modeste, and R. Pic, Stone’s theorem for distributional regression in Wasserstein distance, Journal of Nonparametric Statistics, 37 (2025), pp. 430–452.

[21] C. Durkan, A. Bekasov, I. Murray, and G. Papamakarios, Neural spline flows, in Advances in Neural Information Processing Systems 32 (NeurIPS), 2019, pp. 7511–7522.

[22] S. Efromovich, Conditional density estimation in a regression setting, Annals of Statistics, 35 (2007), pp. 2504–2535.

[23] N. Fournier and A. Guillin, On the rate of convergence in Wasserstein distance of the empirical measure, Probability Theory and Related Fields, 162 (2015), pp. 707–738.

[24] D. S. Greenberg, M. Nonnenmacher, and J. H. Macke, Automatic posterior transformation for likelihood-free inference, in Proceedings of the 36th International Conference on Machine Learning (ICML), vol. 97 of Proceedings of Machine Learning Research, PMLR, 2019, pp. 2404–2414.

[25] L. Gyorfi, M. Kohler, A. Krzy <sup>¨</sup> zak, and H. Walk <sup>˙</sup> , A Distribution-Free Theory of Nonparametric Regression, Springer Series in Statistics, Springer, New York, 2002.

[26] P. Haj lasz, Sobolev spaces on an arbitrary metric space, Potential Analysis, 5 (1996), pp. 403–415.

[27] B. Hosseini, A. W. Hsu, and A. Taghvaei, Conditional optimal transport on function spaces, SIAM/ASA Journal on Uncertainty Quantification, 13 (2025), pp. 304–338.

[28] J.-C. Hutter and P. Rigollet <sup>¨</sup> , Minimax estimation of smooth optimal transport maps, Annals of Statistics, 49 (2021), pp. 1166–1194.

[29] O. Kallenberg, Foundations of Modern Probability, Springer, New York, 2 ed., 2002, https://doi.org/10. 1007/978-1-4757-4015-8.

[30] M. Kohler and A. Krzyzak <sup>˙</sup> , Optimal global rates of convergence for interpolation problems with random design, Statistics & Probability Letters, 83 (2013), pp. 1871–1879.

[31] S. Kumar, Y. Yang, and L. Lin, A likelihood based approach to distribution regression using conditional deep generative models, in Proceedings of the 42nd International Conference on Machine Learning (ICML), vol. 267 of Proceedings of Machine Learning Research, PMLR, 2025, pp. 31964–31990.

[32] M. Li, M. Neykov, and S. Balakrishnan, Minimax optimal conditional density estimation under total variation smoothness, Electronic Journal of Statistics, 16 (2022), pp. 3937–3972.

[33] T. Liang, How well generative adversarial networks learn distributions, Journal of Machine Learning Research, 22 (2021), pp. 1–41.

[34] S. Liu, X. Zhou, Y. Jiao, and J. Huang, Wasserstein generative learning of conditional distribution. arXiv preprint arXiv:2112.10039, 2021, https://arxiv.org/abs/2112.10039.

[35] T. Manole, S. Balakrishnan, J. Niles-Weed, and L. Wasserman, Plugin estimation of smooth optimal transport maps, Annals of Statistics, 52 (2024), pp. 966–998.

[36] C. McDiarmid, On the method of bounded diferences, in Surveys in Combinatorics, 1989, J. Siemons, ed., vol. 141 of London Mathematical Society Lecture Note Series, Cambridge University Press, 1989, pp. 148–188, https://doi.org/10.1017/CBO9781107359949.008.

[37] J. Niles-Weed and Q. Berthet, Minimax estimation of smooth densities in Wasserstein distance, Annals of Statistics, 50 (2022), pp. 1519–1540.

[38] G. Papamakarios and I. Murray, Fast ε-free inference of simulation models with Bayesian conditional density estimation, in Advances in Neural Information Processing Systems 29 (NIPS), 2016, pp. 1028– 1036.

[39] S. Park, A. Zhou, H. Kim, and A. Barati Farimani, Generative neural operators through difusion last layer, in Proceedings of the 43rd International Conference on Machine Learning (ICML), vol. 306 of Proceedings of Machine Learning Research, PMLR, 2026, pp. 97234–97262.

[40] R. Peyre<sup>´</sup>, Comparison between W<sub>2</sub> distance and H<sup>˙</sup> <sup>−1</sup> norm, and localization of Wasserstein distance, ESAIM: Control, Optimisation and Calculus of Variations, 24 (2018), pp. 1489–1501.

[41] R. Pic, C. Dombry, P. Naveau, and M. Taillardat, Distributional regression and its evaluation with the CRPS: Bounds and convergence of the minimax risk, International Journal of Forecasting, 39 (2023), pp. 1564–1572.

[42] S. Rasp and S. Lerch, Neural networks for postprocessing ensemble weather forecasts, Monthly Weather Review, 146 (2018), pp. 3885–3900.

[43] H. P. Rosenthal, On the subspaces of L<sup>p</sup> (p > 2) spanned by sequences of independent random variables, Israel Journal of Mathematics, 8 (1970), pp. 273–303, https://doi.org/10.1007/BF02771562.

[44] D. Salinas, V. Flunkert, J. Gasthaus, and T. Januschowski, DeepAR: Probabilistic forecasting with autoregressive recurrent networks, International Journal of Forecasting, 36 (2020), pp. 1181–1191.

[45] F. Santambrogio, Optimal Transport for Applied Mathematicians: Calculus of Variations, PDEs, and Modeling, vol. 87 of Progress in Nonlinear Diferential Equations and Their Applications, Birkh¨auser, Cham, 2015.

[46] S. Singh and B. Poczos<sup>´</sup> , Minimax distribution estimation in Wasserstein distance. arXiv preprint arXiv:1802.08855, 2018, https://arxiv.org/abs/1802.08855.

[47] S. Singh, A. Uppal, B. Li, C.-L. Li, M. Zaheer, and B. Poczos<sup>´</sup> , Nonparametric density estimation under adversarial losses, in Advances in Neural Information Processing Systems 31 (NeurIPS), 2018.

[48] S. Song, T. Wang, G. Shen, Y. Lin, and J. Huang, Wasserstein generative regression, Journal of the Royal Statistical Society Series B: Statistical Methodology, 88 (2026), pp. 330–351.

[49] E. G. Tabak, G. Trigila, and W. Zhao, Conditional density estimation and simulation through optimal transport, Machine Learning, 109 (2020), pp. 665–688.

[50] E. G. Tabak, G. Trigila, and W. Zhao, Data driven conditional optimal transport, Machine Learning, 110 (2021), pp. 3135–3155.

[51] S. Tamano and M. Imaizumi, Extended Wasserstein-GAN approach to causal distribution learning: Density-free estimation and minimax optimality, in Advances in Neural Information Processing Systems, 2026. To appear.

[52] R. Tang, L. Lin, and Y. Yang, Conditional difusion models are minimax-optimal and manifold-adaptive for conditional distribution estimation, in International Conference on Learning Representations (ICLR), 2025.

[53] R. Tang and Y. Yang, Minimax optimal rates for regression on manifolds and distributions. arXiv preprint arXiv:2506.07504, 2025, https://arxiv.org/abs/2506.07504.

[54] H. Triebel, Function Spaces and Wavelets on Domains, vol. 7 of EMS Tracts in Mathematics, European Mathematical Society, Z¨urich, 2008.

[55] A. Uppal, S. Singh, and B. Poczos<sup>´</sup> , Nonparametric density estimation & convergence rates for GANs under Besov IPM losses, in Advances in Neural Information Processing Systems 32 (NeurIPS), 2019.

[56] A. Vacher, B. Muzellec, F. Bach, F.-X. Vialard, and A. Rudi, Optimal estimation of smooth transport maps with kernel SoS, SIAM Journal on Mathematics of Data Science, 6 (2024), pp. 311–342.

[57] C. Villani, Optimal Transport: Old and New, vol. 338 of Grundlehren der mathematischen Wissenschaften, Springer, 2009, https://doi.org/10.1007/978-3-540-71050-9.

[58] J. Weed and F. Bach, Sharp asymptotic and finite-sample rates of convergence of empirical measures in Wasserstein distance, Bernoulli, 25 (2019), pp. 2620–2648.