# Semi-Supervised Classification with Informative Missing Labels in Weibull Mixture Models

Jinran Wu The University of Queensland, Australia jinran.wu@uq.edu.au

You-Gan Wang The University of Queensland, Australia you-gan.wang@uq.edu.au

Geoffrey J. McLachlan The University of Queensland, Australia g.mclachlan@uq.edu.au

## ABSTRACT

We consider semi-supervised classification from a partially classified sample arising from a twocomponent Weibull mixture. The feature is observed for all data, whereas some class labels are missing. The probability of a missing label is modelled as a function of classification uncertainty, giving a feature-dependent missing-at-random (MAR) mechanism that shares parameters with the Weibull-mixture classifier. The missing-label indicators can therefore provide information about the classifier in addition to the observed features and available class labels. Under a common Weibull shape, a Bayes’ rule has at most one positive decision boundary, which is unique when the rule is nonconstant; under unequal shapes, it can have two. We characterise these decision regions, derive the Fisher information for the classifier after adjustment for nuisance parameters in the missingness model, and obtain a decision-boundary expansion of the expected error rate of the plug-in sample rule relative to the Bayes error. The expansion yields classification-specific asymptotic relative efficiency formulas for the one- and two-boundary cases and shows that a positive-definite increase in Fisher information is sufficient, but not necessary, for a smaller first-order expected error rate. Numerical studies and a semi-synthetic analysis based on hard-drive failure data illustrate potential reductions in expected error rate and improvements in decision-boundary estimation from modelling feature-dependent label missingness.

Keywords label missingness · missing at random · Weibull mixture · Fisher information · asymptotic relative efficiency

## 1 Introduction

We consider semi-supervised classification for positive-valued data modelled by a two-component Weibull mixture. The Weibull distribution is widely used for lifetime and reliability data because its shape parameter accommodates a broad range of failure-rate behaviour (Lai et al., 2011; Menon, 1963; Zhao et al., 2011). Weibull mixtures allow for latent heterogeneity and multiple failure mechanisms and have a substantial statistical literature (Bucar et al.ˇ , 2004; Farcomeni and Nardi, 2010; Jewell, 1982; Jiang and Murthy, 1998; Panteleeva et al., 2015; Tsionas, 2002; Woodward and Gunst, 1987). In a classification setting, Ahmad and Abd-Elrahman (1994) studied a nonlinear discriminant function for a two-component Weibull mixture using classified and unclassified observations. Here our interest is in the additional structure that arises when class labels are missing in a way that depends on the observed feature.

In likelihood-based semi-supervised learning, unlabelled data can contribute information through the marginal distribution of the observed feature (Ahfock and McLachlan, 2023; Dempster et al., 1977; McLachlan et al., 2019). When the probability that a class label is missing depends on the observed feature, the missing-label pattern may itself be informative for the classifier. This possibility has been studied in Gaussian discrimination and more general likelihood-based and semi-supervised settings (Ahfock and McLachlan, 2020, 2023; Lyu et al., 2024; Sportisse et al.,

2023; Wu et al., 2026). We therefore use a feature-dependent missing-at-random (MAR) model in which the missingness probability depends on classification uncertainty and shares parameters with the Weibull-mixture classifier. The present focus is on how this informative-missingness principle interacts with the non-Gaussian decision geometry of Weibull mixtures and, in turn, how the resulting information affects classification error through estimation of the Bayes decision boundaries.

The Weibull model gives a classification geometry that differs from the usual Gaussian case. Under a common component shape, the log-posterior odds are affine after a monotone power transformation, so a Bayes’ rule has at most one positive decision boundary, which is unique when the rule is nonconstant. Under unequal component shapes, the log-posterior odds are nonlinear and can have two positive roots. One class may therefore occupy a bounded central interval and the other the two tails, or the roles may be reversed. This one- versus two-boundary structure is central to the classification-specific efficiency analysis developed below.

Our contributions are fourfold. First, we characterise the Bayes decision regions for common- and unequal-shape Weibull mixtures, including the two-boundary geometry that can arise under unequal shapes. Second, under a sharedparameter MAR label-missingness model, we derive the Fisher information for the classifier after adjustment for the nuisance parameters in the missingness mechanism, thereby separating information lost through unobserved labels from information gained through the missing-label indicators. Third, we derive a decision-boundary representation of the first-order expected error rate of the plug-in sample rule relative to the Bayes error. The expansion depends on estimation uncertainty only in parameter directions that perturb the decision boundaries, yields classification-specific asymptotic relative efficiency formulas for both the one- and two-boundary cases, and shows that a positive-definite increase in Fisher information is sufficient, but not necessary, for a smaller first-order expected error rate. Fourth, numerical simulations and a semi-synthetic study based on hard-drive failure data illustrate the finite-sample consequences for expected error rate and decision-boundary estimation. Relative to the Gaussian and more general informativelabel-missingness settings cited above, the emphasis here is therefore on the way Weibull-specific decision geometry translates information in the missing-label pattern into classification efficiency.

## 2 A two-component mixture of Weibull distributions for classification

## 2.1 Model and Bayes’ rule

Let $Y > 0$ be a scalar feature and let $Z \in \{ 1 , 2 \}$ denote its class. The class prior probabilities are

$$
\Pr ( Z = i ) = \pi _ { i } , \qquad \pi _ { i } > 0 , \qquad \pi _ { 1 } + \pi _ { 2 } = 1 .
$$

Conditional on $Z = i ,$ , assume

$$
f _ { i } ( y ; k _ { i } , \lambda _ { i } ) = \frac { k _ { i } } { \lambda _ { i } } \left( \frac { y } { \lambda _ { i } } \right) ^ { k _ { i } - 1 } \exp \left\{ - \left( \frac { y } { \lambda _ { i } } \right) ^ { k _ { i } } \right\} , \qquad y > 0 ,
$$

where $k _ { i } > 0$ and $\lambda _ { i } > 0$ are the shape and scale parameters. The corresponding distribution function is

$$
F _ { i } ( y ) = 1 - \exp \left\{ - \left( \frac { y } { \lambda _ { i } } \right) ^ { k _ { i } } \right\} .
$$

The marginal density of $Y$ is the two-component Weibull mixture

$$
p _ { Y } ( y ; \theta ) = \pi _ { 1 } f _ { 1 } ( y ) + \pi _ { 2 } f _ { 2 } ( y ) ,
$$

where θ denotes the Weibull-mixture parameter vector.

The posterior probability of membership in class 1 is

$$
\tau _ { 1 } ( y ; \theta ) = \operatorname* { P r } ( Z = 1 \mid Y = y ) = { \frac { \pi _ { 1 } f _ { 1 } ( y ) } { \pi _ { 1 } f _ { 1 } ( y ) + \pi _ { 2 } f _ { 2 } ( y ) } } , \qquad \log \mathrm { i t } \{ \tau _ { 1 } ( y ; \theta ) \} = \delta ( y ; \theta ) ,
$$

where $\tau _ { 2 } = 1 - \tau _ { 1 }$ and

$$
\begin{array} { l } { \displaystyle \delta ( y ; \theta ) = \log \frac { \pi _ { 1 } f _ { 1 } ( y ) } { \pi _ { 2 } f _ { 2 } ( y ) } } \\ { = \log \frac { \pi _ { 1 } } { \pi _ { 2 } } + \log \frac { k _ { 1 } } { k _ { 2 } } - k _ { 1 } \log \lambda _ { 1 } + k _ { 2 } \log \lambda _ { 2 } + ( k _ { 1 } - k _ { 2 } ) \log y - \left( \frac { y } { \lambda _ { 1 } } \right) ^ { k _ { 1 } } + \left( \frac { y } { \lambda _ { 2 } } \right) ^ { k _ { 2 } } . } \end{array}\tag{1}
$$

Under zero-one loss, the Bayes’ rule is

$$
\mathcal { C } _ { \theta } ( y ) = \left\{ \begin{array} { l l } { 1 , } & { \delta ( y ; \theta ) > 0 , } \\ { 2 , } & { \delta ( y ; \theta ) < 0 . } \end{array} \right.
$$

Ties occur only at roots of the log-posterior odds and hence on a set of probability zero in the cases considered below. A Bayes decision boundary is a positive root $y _ { b }$ of the log-posterior odds function,

$$
\begin{array} { r } { \delta ( y _ { b } ; \theta ) = 0 \quad \Longleftrightarrow \quad \pi _ { 1 } f _ { 1 } ( y _ { b } ) = \pi _ { 2 } f _ { 2 } ( y _ { b } ) \quad \Longleftrightarrow \quad \tau _ { 1 } ( y _ { b } ; \theta ) = \tau _ { 2 } ( y _ { b } ; \theta ) = \frac { 1 } { 2 } . } \end{array}
$$

For the asymptotic analysis in Section $^ { 4 , }$ we restrict attention to simple roots, which locally separate regions assigned to different classes; a double root is instead a nonregular touching point. Throughout the paper, “decision boundary” is used in this classification sense.

For numerical and asymptotic calculations, we parameterise the original constrained parameters so that the transformed parameters range over the real line. Define

$$
\rho = \log ( \pi _ { 1 } / \pi _ { 2 } ) , \qquad \alpha _ { i } = \log \lambda _ { i } , \qquad \nu _ { i } = \log k _ { i } .
$$

Then $\mathrm { l o g i t } ( \pi _ { 1 } ) = \rho , \lambda _ { i } = e ^ { \alpha _ { i } }$ , and $k _ { i } = e ^ { \nu _ { i } }$ . For later use, set

$$
a _ { i } ( y ) = \log y - \alpha _ { i } , \qquad r _ { i } ( y ) = \exp \{ k _ { i } a _ { i } ( y ) \} = \left( \frac { y } { \lambda _ { i } } \right) ^ { k _ { i } } .
$$

The log-posterior odds function can be written as

$$
\delta ( y ; \theta _ { U } ) = \rho + \nu _ { 1 } - \nu _ { 2 } + k _ { 1 } a _ { 1 } ( y ) - k _ { 2 } a _ { 2 } ( y ) - r _ { 1 } ( y ) + r _ { 2 } ( y ) ,
$$

where

$$
\theta _ { U } = ( \rho , \alpha _ { 1 } , \nu _ { 1 } , \alpha _ { 2 } , \nu _ { 2 } ) ^ { \top }
$$

for the unequal-shape model.

## 2.2 Common shape

When $k _ { 1 } = k _ { 2 } = k ,$ let ν = log k. The constrained parameter vector is

$$
\theta _ { C } = ( \rho , \alpha _ { 1 } , \alpha _ { 2 } , \nu ) ^ { \top } .
$$

The log-posterior odds function simplifies to

$$
\begin{array} { c } { { \delta _ { C } ( y ; \theta _ { C } ) = \rho + k \log ( \lambda _ { 2 } / \lambda _ { 1 } ) + \left( \lambda _ { 2 } ^ { - k } - \lambda _ { 1 } ^ { - k } \right) y ^ { k } } } \\ { { = A + B y ^ { k } , } } \end{array}
$$

where

$$
A = \rho + k \log ( \lambda _ { 2 } / \lambda _ { 1 } ) , \qquad B = \lambda _ { 2 } ^ { - k } - \lambda _ { 1 } ^ { - k } .
$$

Consequently, the log-posterior odds function is affine in the transformed statistic $X = Y ^ { k }$ , but is generally nonlinear in the original observation $Y .$ . Only when $k = 1$ is it affine in Y itself. Since $y ^ { k }$ is strictly increasing on $( 0 , \infty )$ , a nonconstant Bayes’ rule has a unique positive decision boundary if and only if $A B < 0 .$ , in which case

$$
y _ { 0 } = \left( - \frac { A } { B } \right) ^ { 1 / k } .
$$

When $A B < 0$ , the Bayes error is available in closed form. If $B > 0 ,$

$$
\mathrm { e r r } _ { C } ( \theta _ { C } ) = \pi _ { 1 } F _ { 1 } ( y _ { 0 } ) + \pi _ { 2 } \{ 1 - F _ { 2 } ( y _ { 0 } ) \} ,
$$

and if $B < 0 .$

$$
\mathrm { e r r } _ { C } ( \theta _ { C } ) = \pi _ { 1 } \{ 1 - F _ { 1 } ( y _ { 0 } ) \} + \pi _ { 2 } F _ { 2 } ( y _ { 0 } ) .
$$

If no positive decision boundary exists, the Bayes’ rule assigns every $y > 0$ to the same class. Its error is $\pi _ { 2 }$ when class 1 is always selected and $\pi _ { 1 }$ when class 2 is always selected.

## 2.3 Unequal shapes

When $k _ { 1 } \neq k _ { 2 }$ , the log-posterior odds function is nonlinear in both y and log y. Its derivative is

$$
\delta _ { y } ( y ; \theta _ { U } ) = \frac { k _ { 1 } - k _ { 2 } - k _ { 1 } r _ { 1 } ( y ) + k _ { 2 } r _ { 2 } ( y ) } { y } .
$$

The next result describes the structure of the Bayes decision regions.

Proposition 1 (Unequal-shape decision boundaries). Suppose $\boldsymbol { k } _ { 1 } \neq \boldsymbol { k } _ { 2 }$

(a) The equation $\delta ( y ; \theta _ { U } ) = 0$ has at most two positive roots.

(b) $I f k _ { 1 } > k _ { 2 } ,$ , then $\delta ( y ) \to - \infty$ as y ↓ 0 and as $y  \infty$ . The log-posterior oddsfunction has a unique global maximum. Hence it has either no positive root, one double root, or two simple roots $0 < y _ { 1 } < y _ { 2 }$ . In the two-root case, class 1 is selected on $( y _ { 1 } , y _ { 2 } )$ and class 2 in the two tails.

(c) If $k _ { 1 } < k _ { 2 } ,$ , then $\delta ( y ) $ +∞ as y ↓ 0 and as $y  \infty$ . The log-posterior oddsfunction has a unique global minimum. Hence it has either no positive root, one double root, or two simple roots $0 < y _ { 1 } < y _ { 2 }$ . In the two-root case, class 2 is selected on $( y _ { 1 } , y _ { 2 } )$ and class 1 in the two tails.

The proof is given in Appendix A. This structure simplifies root finding: locate the unique stationary point of the log-posterior odds on the log scale, then bracket at most one root on each side. The result also supplies the decisionboundary geometry used in the ARE calculation. Figure 1 illustrates these two forms of Bayes decision geometry: a single decision boundary under the common-shape model and two decision boundaries under the unequal-shape model.

![](images/766305dd5a52cc5918652c5602810c7b8ae201ca4fb891a25431629e0e02decd.jpg)  
(a)

![](images/3108f3e787b30349d6327d4cbb30ce1e949fb9538833e44adb444982b9a4b7ff.jpg)  
(b)  
Figure 1: Bayes decision regions for representative two-component Weibull mixtures. Panel (a) shows the commonshape case with $\pi _ { 1 } = \pi _ { 2 } = 0 . 5 , ( \lambda _ { 1 } , \lambda _ { 2 } ) = ( 1 , 3 )$ , and $k _ { 1 } = k _ { 2 } = 1 . 5$ . The Bayes’ rule has a single positive decision boundary at $y _ { b } = 1 . 6 0 8 8$ . Panel (b) shows the unequal-shape case with $\pi _ { 1 } = \pi _ { 2 } = 0 . 5 , ( \lambda _ { 1 } , \lambda _ { 2 } ) = ( 1 , 1 . 5 )$ , and $( k _ { 1 } , k _ { 2 } ) = ( 1 , 2 )$ . The Bayes’ rule has two positive decision boundaries, $y _ { 1 } = 0 . 6 9 5 7$ and $y _ { 2 } = 2 . 9 8 5 5$ . The dashed vertical lines mark the roots of $\delta ( y ; \theta ) = 0 ,$ equivalently the points at which $\pi _ { 1 } f _ { 1 } ( y ) = \pi _ { 2 } f _ { 2 } ( y )$ . The shaded bands indicate the corresponding Bayes decision regions.

If $k _ { 1 } > k _ { 2 }$ and two roots exist, the Bayes error is

$$
\mathrm { e r r } _ { U } ( \theta _ { U } ) = \pi _ { 1 } \{ F _ { 1 } ( y _ { 1 } ) + 1 - F _ { 1 } ( y _ { 2 } ) \} + \pi _ { 2 } \{ F _ { 2 } ( y _ { 2 } ) - F _ { 2 } ( y _ { 1 } ) \} .
$$

If $k _ { 1 } < k _ { 2 }$ and two roots exist,

$$
\mathrm { e r r } _ { U } ( \theta _ { U } ) = \pi _ { 1 } \{ F _ { 1 } ( y _ { 2 } ) - F _ { 1 } ( y _ { 1 } ) \} + \pi _ { 2 } \{ F _ { 2 } ( y _ { 1 } ) + 1 - F _ { 2 } ( y _ { 2 } ) \} .
$$

When no positive roots exist, the Bayes’ rule assigns all $y > 0$ to the same class, and its error is the prior probability of the class that is never selected. The double-root case is excluded from the asymptotic boundary theory because $\delta _ { y } = 0$ at the touching point.

## 2.4 Identifiability and label switching

A finite mixture of univariate Weibull densities is identifiable up to a permutation of its components under the usual distinct-component conditions (Panteleeva et al., 2015). In an entirely unlabelled sample, this leaves the familiar label-switching ambiguity. Partial class labels resolve that ambiguity provided both classes have positive probability of appearing among the labelled observations.

Assumption 1 (Identifiability conditions). The component parameter pairs $( k _ { 1 } , \lambda _ { 1 } )$ and $( k _ { 2 } , \lambda _ { 2 } )$ are distinct, $0 ~ <$ $\pi _ { i } < 1$ , and each class has positive probability of appearing among the labelled observations. In addition, $\delta ( Y ; \theta ) ^ { 2 }$ is nondegenerate under the mixture, as required to identify both coefficients in the missingness model introduced below.

The distinct-component condition identifies the mixture up to permutation, while the labelled observations resolve the component-label permutation.

## 3 Fisher information under the label-missingness model

Here we consider a label-missingness mechanism for which the conditional probability of a missing label depends on the observed feature $Y$ . Thus, the mechanism is MAR (Rubin, 1976). We retain the label-missingness mechanism in the statistical model because its linear predictor can depend on the same parameter vector that determines the log-posterior odds and hence Bayes’ rule.

## 3.1 Likelihoods for the partially classified sample

Let $( Y _ { j } , Z _ { j } ) , j = 1 , \dots , n$ , be independent observations from the two-component mixture of Weibull distributions. Let

$$
M _ { j } = \left\{ { \begin{array} { l l } { 1 , } & { Z _ { j } { \mathrm { ~ i s ~ m i s s i n g } } , } \\ { 0 , } & { Z _ { j } { \mathrm { ~ i s ~ o b s e r v e d } } . } \end{array} } \right.
$$

The partially classified observation is $O _ { j } = ( Y _ { j } , M _ { j } , ( 1 - M _ { j } ) Z _ { j } )$ . Write $z _ { i j } = 1 \mathrm { i f } Z _ { j } = i$ and 0 otherwise. The log likelihood formed from the features and their class labels, where available, while ignoring the label-missingness mechanism, is

$$
\ell _ { \mathrm { i g } } ( \theta ) = \sum _ { j = 1 } ^ { n } ( 1 - M _ { j } ) \sum _ { i = 1 } ^ { 2 } z _ { i j } \log \{ \pi _ { i } f _ { i } ( Y _ { j } ) \} + \sum _ { j = 1 } ^ { n } M _ { j } \log p _ { Y } ( Y _ { j } ; \theta ) .
$$

If the sample were completely classified, the corresponding CC log likelihood would be

$$
\ell _ { { \mathrm { C C } } } ( \theta ) = \sum _ { j = 1 } ^ { n } \sum _ { i = 1 } ^ { 2 } z _ { i j } \log \{ \pi _ { i } f _ { i } ( Y _ { j } ) \} .\tag{2}
$$

We assume conditional independence of the missing-label indicator and the class label given the feature:

$$
\operatorname* { P r } ( M = 1 \mid Y = y , Z = z ) = \operatorname* { P r } ( M = 1 \mid Y = y ) = q ( y ; \theta , \xi ) .
$$

The missingness model is

$$
\begin{array} { r } { \log \mathrm { i t } \{ q ( y ; \theta , \xi ) \} = \eta ( y ; \theta , \xi ) = \xi _ { 0 } + \xi _ { 1 } \delta ( y ; \theta ) ^ { 2 } , } \end{array}\tag{3}
$$

where $\xi = ( \xi _ { 0 } , \xi _ { 1 } ) ^ { \top }$ . The case of interest is $\xi _ { 1 } < 0 ;$ , for which labels are most likely to be missing near a Bayes decision boundary. In estimation, however, $\xi _ { 1 }$ is allowed to vary over R; the inequality $\xi _ { 1 } < 0$ specifies the regime of primary interest rather than an optimisation constraint. When $\xi _ { 1 } \neq 0$ , the missing-label probability varies with the observed feature and the mechanism is MAR. The case $\xi _ { 1 } = 0$ gives a constant missing-label probability and serves as an interior benchmark.

Although the mechanism is MAR in the feature-dependent case, (3) is a shared-parameter model: q depends on the same θ as the class densities and $\mathrm { B a y e s } ^ { \prime }$ rule. Hence the parameter-distinctness condition ordinarily used for likelihood ignorability is not satisfied (Lu and Copas, 2004; Rubin, 1976). The missing-label pattern can therefore provide information about the log-posterior odds function and the resulting Bayes’ rule, so the label-missingness mechanism is retained in the likelihood.

The Bernoulli log likelihood for the missing-label indicators is

$$
\ell _ { \mathrm { m i s s } } ( \theta , \xi ) = \sum _ { j = 1 } ^ { n } \left[ M _ { j } \log q ( Y _ { j } ; \theta , \xi ) + ( 1 - M _ { j } ) \log \{ 1 - q ( Y _ { j } ; \theta , \xi ) \} \right] .
$$

Combining these contributions gives the full log likelihood formed from the partially classified sample and its missinglabel indicators

$$
\ell _ { \mathrm { f u l l } } ( \theta , \xi ) = \ell _ { \mathrm { i g } } ( \theta ) + \ell _ { \mathrm { m i s s } } ( \theta , \xi ) .\tag{4}
$$

We use the subscripts CC, ig, and full for the estimator based on the completely classified sample, the estimator based on the likelihood that ignores the label-missingness mechanism, and the estimator based on the full likelihood, respectively. Equivalently, a labelled observation contributes

$$
\{ 1 - q ( y ; \theta , \xi ) \} \pi _ { z } f _ { z } ( y ) ,
$$

and an unlabelled observation contributes

$$
q ( y ; \theta , \xi ) p _ { Y } ( y ; \theta ) .
$$

Because logit $\{ \tau _ { 1 } ( y ) \} ~ = ~ \delta ( y )$ , entropy is an even, strictly decreasing function of $| \delta ( y ) |$ . Hence $\delta ( y ) ^ { 2 }$ provides a smooth measure of classification certainty and attains its minimum at every Bayes decision boundary. Related uncertainty-dependent missing-label mechanisms have been used in likelihood-based semi-supervised classification (Ahfock and McLachlan, 2023; Lyu et al., 2024). The expected missing-label fraction is

$$
{ \gamma } ( \theta , \xi ) = \operatorname { E } \{ q ( Y ; \theta , \xi ) \} .
$$

All expectations below are with respect to the true mixture density unless stated otherwise.

## 3.2 Log-posterior-odds derivatives and information in the completely classified sample

Let

$$
g ( y ; \theta ) = \frac { \partial \delta ( y ; \theta ) } { \partial \theta } .
$$

For the unequal-shape parametrisation $\theta _ { U } = ( \rho , \alpha _ { 1 } , \nu _ { 1 } , \alpha _ { 2 } , \nu _ { 2 } ) ^ { \top }$

$$
g _ { U } ( y ) = \left( \begin{array} { c } { { 1 } } \\ { { k _ { 1 } ( r _ { 1 } - 1 ) } } \\ { { 1 + k _ { 1 } a _ { 1 } ( 1 - r _ { 1 } ) } } \\ { { k _ { 2 } ( 1 - r _ { 2 } ) } } \\ { { - 1 + k _ { 2 } a _ { 2 } ( r _ { 2 } - 1 ) } } \end{array} \right) .
$$

For the common-shape parametrisation $\theta _ { C } = ( \rho , \alpha _ { 1 } , \alpha _ { 2 } , \nu ) ^ { \top }$

$$
g _ { C } ( y ) = \left( \begin{array} { c } { { 1 } } \\ { { k ( r _ { 1 } - 1 ) } } \\ { { k ( 1 - r _ { 2 } ) } } \\ { { k \{ a _ { 1 } ( 1 - r _ { 1 } ) - a _ { 2 } ( 1 - r _ { 2 } ) \} } } \end{array} \right) .
$$

These derivatives also determine the information in both the class labels and the missing-label indicators.

Let $\gamma _ { E }$ denote Euler’s constant and define

$$
C _ { W } = \frac { \pi ^ { 2 } } { 6 } + ( 1 - \gamma _ { E } ) ^ { 2 } .
$$

For $Y \sim \mathrm { { W e i b u l l } } ( k , \lambda )$ , the transformation

$$
R = ( Y / \lambda ) ^ { k }
$$

has the standard exponential distribution. Under the log-scale/log-shape parametrisation $( \alpha , \nu )$ , the class-conditional scores are

$$
S _ { \alpha } = k ( R - 1 ) , \qquad S _ { \nu } = 1 + \log R ( 1 - R ) .
$$

It follows that the per-observation Weibull information matrix for $( \alpha , \nu )$ is

$$
W ( k ) = \binom { k ^ { 2 } } { k ( \gamma _ { E } - 1 ) } \quad k ( \gamma _ { E } - 1 ) \biggr ) .
$$

For unequal shapes, let

$$
\theta _ { U } = \left( \rho , \alpha _ { 1 } , \nu _ { 1 } , \alpha _ { 2 } , \nu _ { 2 } \right) ^ { \top } .
$$

The per-observation Fisher information in the completely classified sample is block diagonal:

$$
\begin{array} { r } { \mathcal { I } _ { \mathrm { C C } , U } ( \theta _ { U } ) = \mathrm { d i a g } \{ \pi _ { 1 } \pi _ { 2 } , \pi _ { 1 } W ( k _ { 1 } ) , \pi _ { 2 } W ( k _ { 2 } ) \} . } \end{array}
$$

For a common shape, the per-observation information for $\theta _ { C } = ( \rho , \alpha _ { 1 } , \alpha _ { 2 } , \nu ) ^ { \top }$ is

$$
\mathcal { I } _ { \mathrm { C C } , C } = \left( \begin{array} { c c c c } { \pi _ { 1 } \pi _ { 2 } } & { 0 } & { 0 } & { 0 } \\ { 0 } & { \pi _ { 1 } k ^ { 2 } } & { 0 } & { \pi _ { 1 } k ( \gamma _ { E } - 1 ) } \\ { 0 } & { 0 } & { \pi _ { 2 } k ^ { 2 } } & { \pi _ { 2 } k ( \gamma _ { E } - 1 ) } \\ { 0 } & { \pi _ { 1 } k ( \gamma _ { E } - 1 ) } & { \pi _ { 2 } k ( \gamma _ { E } - 1 ) } & { C _ { W } } \end{array} \right) .
$$

The Fisher information in the completely classified sample is $\mathcal { T } _ { \mathrm { C C } } = n \mathcal { I } _ { \mathrm { C C } }$

The quantities $\mathcal { T } _ { \mathrm { l o s t } } , \mathcal { T } _ { \mathrm { i g } }$ , and $\mathcal { I } _ { \mathrm { f u l l } }$ considered below require one-dimensional integration because the posterior probabilities and missingness weights involve both components.

## 3.3 Information loss and gain from the label-missingness mechanism

For one observation, the log likelihood that ignores the label-missingness mechanism can be written as

$$
\ell _ { \mathrm { i g } , 1 } ( \theta ) = \log p ( Y , Z ; \theta ) - M \log p ( Z \mid Y ; \theta ) .\tag{5}
$$

Conditional on $Y = y$ , the label is Bernoulli with success probability $\tau _ { 1 } ( y )$ , where logit $\{ \tau _ { 1 } ( y ) \} = \delta ( y )$ . Therefore, the conditional Fisher information about θ carried by $Z ,$ given $Y = y ,$ , is

$$
\mathcal { I } _ { Z | Y } ( y ; \theta ) = \tau _ { 1 } ( y ) \tau _ { 2 } ( y ) g ( y ; \theta ) g ( y ; \theta ) ^ { \top } .
$$

Because $M \perp Z \mid Y ,$

$$
\mathcal { I } _ { \mathrm { l o s t } } ( \theta , \xi ) = \mathrm { E } \left[ q ( Y ; \theta , \xi ) \tau _ { 1 } ( Y ; \theta ) \tau _ { 2 } ( Y ; \theta ) g ( Y ; \theta ) g ( Y ; \theta ) ^ { \top } \right]\tag{6}
$$

is the per-observation information lost through unobserved labels. Thus

$$
\mathcal { I } _ { \mathrm { i g } } ( \theta , \xi ) = \mathcal { I } _ { \mathrm { C C } } ( \theta ) - \mathcal { I } _ { \mathrm { l o s t } } ( \theta , \xi ) .\tag{7}
$$

Although $\ell _ { \mathrm { i g } } ( \theta )$ does not contain $\xi ,$ the matrix $\mathcal { I } _ { \mathrm { i g } }$ depends on ξ through the distribution of M. Under the sharedparameter model, $\ell _ { \mathrm { i g } }$ is an estimating criterion rather than the full observed-data log likelihood. Nevertheless, its score satisfies an information equality at the true parameter, which justifies the inverse-information covariance used below.

Lemma 1 (Information equality for the ignoring estimator). Let

$$
S _ { Y } ( \theta ) = { \frac { \partial } { \partial \theta } } \log p _ { Y } ( Y ; \theta ) , \qquad S _ { Z | Y } ( \theta ) = { \frac { \partial } { \partial \theta } } \log p ( Z \mid Y ; \theta ) ,
$$

and let

$$
\psi _ { \mathrm { i g } } ( \theta ) = S _ { Y } ( \theta ) + ( 1 - M ) S _ { Z | Y } ( \theta )
$$

be the score of $\ell _ { \mathrm { i g , 1 } }$ . Under the conditional-independence assumption $M \perp Z \mid Y$ and the usual differentiability and moment conditions,

$$
- \operatorname { E } \left\{ { \frac { \partial \psi _ { \mathrm { i g } } ( \theta ) } { \partial \theta ^ { \top } } } \right\} = \operatorname { E } \{ \psi _ { \mathrm { i g } } ( \theta ) \psi _ { \mathrm { i g } } ( \theta ) ^ { \top } \} = \mathcal { I } _ { \mathrm { i g } } ( \theta , \xi ) .\tag{8}
$$

If, in addition, $\widehat { \theta } _ { \mathrm { i g } } \ \xrightarrow { p } \theta , \ \mathcal { J } _ { \mathrm { i g } }$ is nonsingular, and the standard local regularity conditions for M-estimation hold, then

$$
\begin{array} { r } { \sqrt { n } ( \widehat { \theta } _ { \mathrm { i g } } - \theta ) \stackrel { d } {  } N \Big ( 0 , \mathcal { T } _ { \mathrm { i g } } ^ { - 1 } \Big ) . } \end{array}\tag{9}
$$

A proof is given in Appendix B.

Let

$$
\begin{array} { r } { s ( y ; \theta ) = \delta ( y ; \theta ) ^ { 2 } , \qquad x ( y ; \theta ) = { \binom { 1 } { s ( y ; \theta ) } } , \qquad w ( y ; \theta , \xi ) = q ( y ) \{ 1 - q ( y ) \} . } \end{array}
$$

The gradient of the missingness linear predictor is

$$
\frac { \partial \eta } { \partial \theta } = 2 \xi _ { 1 } \delta ( y ; \theta ) g ( y ; \theta ) \equiv u ( y ; \theta , \xi ) , \qquad \frac { \partial \eta } { \partial \xi } = x ( y ; \theta ) .
$$

The per-observation Fisher information from $M \mid Y$ for $( \theta ^ { \top } , \xi ^ { \top } ) ^ { \top }$ has the following blocks, with their dependence on $( \theta , \xi )$ suppressed for brevity:

$$
\begin{array} { r } { B _ { \theta \theta } = \mathrm { E } \{ w ( Y ) u ( Y ) u ( Y ) ^ { \top } \} , } \\ { B _ { \theta \xi } = \mathrm { E } \{ w ( Y ) u ( Y ) x ( Y ) ^ { \top } \} , } \\ { B _ { \xi \xi } = \mathrm { E } \{ w ( Y ) x ( Y ) x ( Y ) ^ { \top } \} . } \end{array}
$$

When $\xi$ is estimated jointly with $\theta ,$ the information about θ contributed by the missing-label indicators, after adjustment for $\xi ,$ is the Schur complement

$$
\mathcal { I } _ { \mathrm { m i s s } } ^ { \mathrm { a d j } } ( \theta , \xi ) = B _ { \theta \theta } - B _ { \theta \xi } B _ { \xi \xi } ^ { - 1 } B _ { \xi \theta } .\tag{10}
$$

This positive-semidefinite matrix is the part of the missingness information about θ that remains after adjusting for the nuisance parameters $( \xi _ { 0 } , \xi _ { 1 } )$

Theorem 1 (Information under the full likelihood). Assume the model is identifiable and that the true parameter lies in an identifiable regular interior neighbourhood. Assume further that differentiation may be interchanged with inte gration, $B _ { \xi \xi }$ is nonsingular, the relevantfull-likelihood and completely classified likelihood maximisers are consistent for the true parameter, and the standard local regularity conditions for maximum-likelihood estimation hold. The per-observation Fisher informationfor θ, after adjustmentfor the nuisance parameter $\xi ,$ is

$$
\mathcal { I } _ { \mathrm { f u l l } } ( \theta , \xi ) = \mathcal { I } _ { \mathrm { C C } } ( \theta ) - \mathcal { I } _ { \mathrm { l o s t } } ( \theta , \xi ) + \mathcal { I } _ { \mathrm { m i s s } } ^ { \mathrm { a d j } } ( \theta , \xi ) .\tag{11}
$$

Consequently,

$$
\mathcal { I } _ { \mathrm { f u l l } } ( \theta , \xi ) - \mathcal { I } _ { \mathrm { C C } } ( \theta ) = \mathcal { I } _ { \mathrm { m i s s } } ^ { \mathrm { a d j } } ( \theta , \xi ) - \mathcal { I } _ { \mathrm { l o s t } } ( \theta , \xi ) .
$$

If

$$
\mathcal { T } _ { \mathrm { m i s s } } ^ { \mathrm { a d j } } ( \theta , \xi ) - \mathcal { T } _ { \mathrm { l o s t } } ( \theta , \xi ) > 0 ,\tag{12}
$$

and both $\mathcal { I } _ { \mathrm { f u l l } } ( \theta , \xi )$ and ${ \mathcal { I } } _ { \mathrm { C C } } ( \theta )$ are positive definite, then the estimator based on the full likelihood has a smaller asymptotic covariance matrix than the maximum-likelihood estimator based on the completely classified sample.

A derivation is given in Appendix B. In subsequent displays, parameter arguments of the information matrices are suppressed when no ambiguity arises.

Remark 1 (Role of the missingness parameters). If ξ were known, the missingness information would be $B _ { \theta \theta }$ ; estimating ξ subtracts the nuisance adjustment $B _ { \theta \xi } B _ { \xi \xi } ^ { - 1 } B _ { \xi \theta } . \operatorname { I f } \xi _ { 1 } = 0$ , then $\mathcal { I } _ { \mathrm { m i s s } } ^ { \mathrm { a d j } } = 0 \mathrm { a n d } \mathcal { I } _ { \mathrm { f u l l } } = \mathcal { I } _ { \mathrm { C C } } - \mathcal { I } _ { \mathrm { l o s t } } \leq \mathcal { I } _ { \mathrm { C C } }$ . Thus, a constant missing-label probability cannot offset the loss of class labels; any such offset requires feature-dependent MAR label missingness that is informative about θ through the shared-parameter structure.

Remark 2 (Augmented benchmark based on the completely classified sample). Consider the augmented experiment in which $( Y , Z , M )$ is observed. After adjustment for the nuisance parameter $\xi ,$ its information about θ is

$$
\mathcal { T } _ { \mathrm { C C } } ^ { \mathrm { a u g } } = \mathcal { I } _ { \mathrm { C C } } + \mathcal { I } _ { \mathrm { m i s s } } ^ { \mathrm { a d j } } .
$$

Consequently,

$$
\mathcal { T } _ { \mathrm { f u l l } } = \mathcal { T } _ { \mathrm { C C } } ^ { \mathrm { a u g } } - \mathcal { T } _ { \mathrm { l o s t } } \leq \mathcal { T } _ { \mathrm { C C } } ^ { \mathrm { a u g } } .
$$

Thus, the partially classified sample analysed with the full likelihood cannot dominate the augmented experiment that observes $( Y , Z , \dot { M } )$

Remark 3. For any integrable function $h .$

$$
\operatorname { E } \{ h ( Y ) \} = \sum _ { i = 1 } ^ { 2 } \pi _ { i } \int _ { 0 } ^ { \infty } h \left( \lambda _ { i } r ^ { 1 / k _ { i } } \right) e ^ { - r } \mathrm { d } r .
$$

Thus all required expectations, including $\gamma ,$ the Bayes error, and the ARE, can be evaluated by one-dimensional Gauss–Laguerre quadrature.

## 4 Expected error rate of the plug-in sample rule and asymptotic relative efficiency

## 4.1 Expected error rate of the plug-in sample rule

For an estimator ${ \widehat { \theta } } _ { s } .$ the plug-in sample rule is $\mathcal { C } _ { \widehat { \theta } _ { s } }$ . Its expected error rate, averaged over repeated training samples, is ${ \mathrm { E } } \{ R ( { \widehat { \theta } } _ { s } ; \theta ) \}$ , where θ denotes the data-generating parameter. Because the Bayes error is fixed by $\theta ,$ estimators can be compared through

$$
\begin{array} { r } { \operatorname { E } \{ R ( \widehat { \theta } _ { s } ; \theta ) \} - \operatorname { e r r } ( \theta ) , \qquad s \in \{ \mathrm { C C } , \mathrm { i g } , \mathrm { f u l l } \} , } \end{array}
$$

which gives the usual sample-size interpretation of asymptotic relative efficiency (Ahfock and McLachlan, 2020;   
Efron, 1975).

## 4.2 Decision-boundary expansion

For the local expansion, let $\widetilde { \theta }$ denote a generic parameter value used to form the classification rule, while θ denotes the data-generating parameter. The error rate of the corresponding rule is

$$
\begin{array} { r } { R ( \widetilde { \theta } ; \theta ) = \pi _ { 1 } \operatorname* { P r } \{ \delta ( Y ; \widetilde { \theta } ) < 0 \mid Z = 1 \} + \pi _ { 2 } \operatorname* { P r } \{ \delta ( Y ; \widetilde { \theta } ) > 0 \mid Z = 2 \} , } \end{array}
$$

where the probabilities are evaluated under the data-generating parameter $\theta . \mathrm { \ A t \ } \widetilde { \theta } = \theta .$ , this equals the Bayes error $\operatorname { e r r } ( \theta )$ . Throughout this section, δ denotes the log-posterior odds log $\{ \pi _ { 1 } f _ { 1 } / ( \pi _ { 2 } f _ { 2 } ) \}$ rather than an arbitrary rescaling of the decision score. Let

$$
B ( \theta ) = \{ y > 0 : \delta ( y ; \theta ) = 0 \}
$$

denote the Bayes decision boundaries.

Assumption 2 (Stable simple boundaries). The Bayes’ rule is nonconstant and $B ( \theta ) = \left\{ y _ { 1 } , \ldots , y _ { K } \right\}$ is finite. Each boundary lies in $( 0 , \infty )$ and is simple, $\delta _ { y } ( y _ { j } ; \theta ) \neq 0$ . In a neighbourhood of $\theta , \delta ( y ; \widetilde { \theta } )$ is twice continuously differentiable in $( y , \widetilde { \theta } )$ , the number of roots remains K, and the roots can be represented by differentiable functions $y _ { j } ( { \widetilde { \theta } } )$ that do not merge or escape to 0 or ∞.

For the common-shape model, $K = 1$ when $A B \ < \ 0 ;$ this structure is locally stable whenever A and B remain bounded away from zero. For the unequal-shape model, the two-boundary structure is locally stable when the unique extremum is strictly separated from zero, and both roots are simple. Tangent roots, root mergers, and constant Bayes rules are nonregular cases and are excluded from the expansion below.

Theorem 2 (Decision-boundary representation of the error-rate Hessian). Under Assumption 2, with $p _ { Y }$ continuous at the boundaries,

$$
\left. \frac { \partial R ( \widetilde { \theta } ; \theta ) } { \partial \widetilde { \theta } } \right| _ { \widetilde { \theta } = \theta } = 0 ,
$$

and

$$
H _ { R } ( \theta ) : = \left. \frac { \partial ^ { 2 } R ( \widetilde { \theta } ; \theta ) } { \partial \widetilde { \theta } \partial \widetilde { \theta } ^ { \top } } \right| _ { \widetilde { \theta } = \theta } = \sum _ { j = 1 } ^ { K } \frac { p _ { Y } ( y _ { j } ; \theta ) } { 2 | \delta _ { y } ( y _ { j } ; \theta ) | } g ( y _ { j } ; \theta ) g ( y _ { j } ; \theta ) ^ { \top } .\tag{13}
$$

Let $\widehat { \theta }$ satisfy

$$
\begin{array} { r } { \sqrt { n } ( \widehat { \theta } - \theta ) \stackrel { d } {  } N ( 0 , V ) , \qquad n \operatorname { E } \{ ( \widehat { \theta } - \theta ) ( \widehat { \theta } - \theta ) ^ { \top } \} \longrightarrow V , } \end{array}\tag{14}
$$

and suppose that,for some $\epsilon > 0 ,$

$$
\operatorname { E } \| { \sqrt { n } } ( { \widehat { \theta } } - \theta ) \| ^ { 2 + \epsilon } = O ( 1 ) .\tag{15}
$$

Then

$$
\operatorname { E } \{ R ( \widehat { \theta } ; \theta ) \} - \operatorname { e r r } ( \theta ) = \frac { 1 } { 4 n } \sum _ { j = 1 } ^ { K } \frac { p _ { Y } ( y _ { j } ; \theta ) } { | \delta _ { y } ( y _ { j } ; \theta ) | } g ( y _ { j } ; \theta ) ^ { \top } V g ( y _ { j } ; \theta ) + o ( n ^ { - 1 } ) .\tag{16}
$$

For a correctly specified regular maximum-likelihood estimator satisfying the stated moment conditions, $V = \mathcal { I } ( \theta ) ^ { - 1 }$ so

$$
\operatorname { E } \{ R ( \widehat { \theta } ; \theta ) \} - \operatorname { e r r } ( \theta ) = \frac { 1 } { 4 n } \sum _ { j = 1 } ^ { K } \frac { p _ { Y } ( y _ { j } ; \theta ) } { | \delta _ { y } ( y _ { j } ; \theta ) | } g ( y _ { j } ; \theta ) ^ { \top } \mathcal { I } ( \theta ) ^ { - 1 } g ( y _ { j } ; \theta ) + o ( n ^ { - 1 } ) .\tag{17}
$$

For the estimator that ignores the missingness mechanism, the same specialisation holds with $\mathcal { I } = \mathcal { I } _ { \mathrm { i g } }$ by Lemma 1.

The weight $p _ { Y } ( y _ { j } ) / | \delta _ { y } ( y _ { j } ) |$ measures the probability mass near boundary $y _ { j }$ relative to the local steepness of the log-posterior odds. The quadratic form measures uncertainty in the estimated log-posterior odds at that boundary.

## 4.3 Asymptotic relative efficiency

For an estimator $s \in \{ \mathrm { C C } , \mathrm { i g } , \mathrm { f u l l } \}$ , define the leading excess-error coefficient

$$
C _ { s } ( \theta ) = \frac { 1 } { 4 } \sum _ { j = 1 } ^ { K } \omega _ { j } g _ { j } ^ { \top } V _ { s } g _ { j } , \qquad \omega _ { j } = \frac { p _ { Y } ( y _ { j } ; \theta ) } { | \delta _ { y } ( y _ { j } ; \theta ) | } , \qquad g _ { j } = g ( y _ { j } ; \theta ) ,\tag{18}
$$

so that $\operatorname { E } \{ R ( { \widehat { \theta } } _ { s } ; \theta ) \} - \operatorname { e r r } ( \theta ) = C _ { s } ( \theta ) / n + o ( n ^ { - 1 } )$ . Under the usual regularity conditions for the completely classified and full maximum-likelihood estimators, together with Lemma 1 for the ignoring estimator, the asymptotic covariance matrices are

$$
V _ { \mathrm { C C } } = \mathcal { I } _ { \mathrm { C C } } ^ { - 1 } , \qquad V _ { \mathrm { i g } } = \mathcal { I } _ { \mathrm { i g } } ^ { - 1 } , \qquad V _ { \mathrm { f u l l } } = \mathcal { I } _ { \mathrm { f u l l } } ^ { - 1 } .
$$

For $s \in \{ \mathrm { i g } , \mathrm { f u l l } \}$ , define the asymptotic relative efficiency with respect to the estimator based on the completely classified sample by

$$
\mathrm { A R E } _ { s : \mathrm { C C } } ( \theta ) = \frac { C _ { \mathrm { C C } } ( \theta ) } { C _ { s } ( \theta ) } .\tag{19}
$$

Thus, $\mathrm { A R E } _ { s : \mathrm { C C } } > 1$ means that estimator s has a smaller first-order expected error rate above the Bayes error than the estimator based on the completely classified sample. The main theoretical comparison below takes $s = \mathrm { f u l l }$

Corollary 1 (Common-shape asymptotic relative efficiency). Assume the conditions of Theorem 2. Under a commonshape two-component Weibull mixture, suppose $A B < 0 ,$ , so that Bayes’ rule has the unique simple decision boundary y . For the full estimator relative to the estimator based on the completely classified sample,

$$
\mathrm { A R E } _ { \mathrm { f u l l : C C } } ^ { \mathrm { C } } = \frac { g _ { C } ( y _ { 0 } ) ^ { \top } \mathcal { T } _ { \mathrm { C C , C } } ^ { - 1 } g _ { C } ( y _ { 0 } ) } { g _ { C } ( y _ { 0 } ) ^ { \top } \mathcal { T } _ { \mathrm { f u l l , } C } ^ { - 1 } g _ { C } ( y _ { 0 } ) } .\tag{20}
$$

Proof. For the common-shape model, $K = 1$ in (18). Substituting $V _ { \mathrm { C C } } = \mathcal { I } _ { \mathrm { C C } , C } ^ { - 1 }$ and $V _ { \mathrm { f u l l } } = \{ \mathcal { T } _ { \mathrm { f u l l } , C } \} ^ { - 1 }$ into (19), the common boundary factor $\omega _ { 0 } / 4$ cancels, yielding (20). □

The subscript C denotes the common-shape parametrisation. The information matrices in (20) must be obtained from the constrained common-shape likelihood; they are not obtained by setting $\nu _ { 1 } = \nu _ { 2 }$ after inverting the unequal-shape information matrix.

Corollary 2 (Unequal-shape asymptotic relative efficiency). Assume the conditions of Theorem 2. Under an unequalshape two-component Weibull mixture, suppose the Bayes’ rule has two simple decision boundaries $0 < y _ { 1 } < y _ { 2 }$ . For the full estimator relative to the estimator based on the completely classified sample,

$$
\begin{array} { r } { \mathrm { A R E } _ { \mathrm { f u l l : C C } } ^ { \mathrm { U } } = \frac { \displaystyle \sum _ { j = 1 } ^ { 2 } \omega _ { j } g _ { U } ( y _ { j } ) ^ { \top } \mathcal { I } _ { \mathrm { C C , } U } ^ { - 1 } g _ { U } ( y _ { j } ) } { \displaystyle \sum _ { j = 1 } ^ { 2 } \omega _ { j } g _ { U } ( y _ { j } ) ^ { \top } \mathcal { I } _ { \mathrm { f u l l } , U } ^ { - 1 } g _ { U } ( y _ { j } ) } . } \end{array}\tag{21}
$$

Proof. For the unequal-shape model, $K = 2$ in (18). Substitution of the covariance matrices for the completely classified and full-likelihood estimators into (19) gives (21). Unlike the single-boundary case, the two boundary weights generally do not cancel. □

The positive-definiteness condition in (12) is sufficient for an ARE above one, but it is stronger than necessary: only covariance in the parameter directions that perturb the Bayes decision boundaries enters the first-order errorrate comparison.

## 5 Numerical simulations

## 5.1 Simulation setup

We conducted numerical simulations for six two-component Weibull mixture models with equal class probabilities, $\pi _ { 1 } = \pi _ { 2 } = 0 . 5$ , sample size $n = 5 0 0$ , and $B = 5 0 0$ Monte Carlo replications. Table 1 summarises the data-generating models. Designs $\mathrm { C } 1 { - } \mathrm { C } 3$ impose a common Weibull shape parameter and therefore have one positive Bayes decision boundary, whereas designs U1–U3 allow unequal shape parameters and have two.

In each simulation setting, class labels were removed according to

$$
\begin{array} { r } { \log \mathrm { i t } \{ q ( y ; \theta , \xi ) \} = \xi _ { 0 } + \xi _ { 1 } \delta ( y ; \theta ) ^ { 2 } , } \end{array}
$$

where $\delta ( y ; \theta )$ is the log-posterior odds defined in (1). Because $\delta ( y ; \theta ) = 0$ at each Bayes decision boundary, $\delta ( y ; \theta ) ^ { 2 }$ provides a smooth measure of classification certainty and attains its minimum at the boundaries. We considered

$$
\xi _ { 1 } \in \{ 0 , - 0 . 5 , - 1 , - 2 \} .
$$

Table 1: Weibull-mixture models used in the numerical simulations.
<table><tr><td>Design</td><td>Shape model</td><td> $( \lambda _ { 1 } , \lambda _ { 2 } )$ </td><td> $( k _ { 1 } , k _ { 2 } )$ </td><td>Bayes error</td><td>Decision boundary or boundaries</td></tr><tr><td>C1</td><td>Common</td><td>(1, 3)</td><td>(1,1)</td><td>0.3075</td><td>1.6479</td></tr><tr><td>C2</td><td>Common</td><td>(1,3)</td><td>(1.5, 1.5)</td><td>0.2274</td><td>1.6088</td></tr><tr><td>C3</td><td>Common</td><td>(1,3)</td><td>(3,3)</td><td>0.0758</td><td>1.5070</td></tr><tr><td>U1</td><td>Unequal</td><td>(1,1.5)</td><td>(0.7, 1.5)</td><td>0.3266</td><td>0.4882, 3.4944</td></tr><tr><td>U2</td><td>Unequal</td><td>(1, 1.5)</td><td>(1,2)</td><td>0.3304</td><td>0.6957, 2.9855</td></tr><tr><td>U3</td><td>Unequal</td><td>(1, 1.5)</td><td>(1.5, 3)</td><td>0.3051</td><td>0.9185, 2.5641</td></tr></table>

Note. The common-shape models have one positive decision boundary, whereas the unequal-shape models have two positive simple decision boundaries.

For $\xi _ { 1 } < 0$ , label missingness is MAR because the missing-label probability depends on the observed feature; more negative values correspond to stronger concentration of missing labels near the Bayes decision boundaries. The value $\xi _ { 1 } = 0$ provides the constant-probability benchmark. For each value of $\xi _ { 1 }$ , the intercept $\xi _ { 0 }$ was calibrated to satisfy

$$
{ \mathrm { E } } \{ q ( Y ; \theta , \xi ) \} = \gamma , \qquad \gamma \in \{ 0 . 1 0 , 0 . 3 0 , 0 . 5 0 \} .
$$

The resulting expected missing-label proportions were 10%, 30%, and 50%. The negative values of $\xi _ { 1 }$ therefore generate increasingly feature-dependent MAR label missingness. The simulation study is intended to assess finitesample performance rather than to provide a numerical verification of the asymptotic ARE formulas. The six Weibull models, four values of $\xi _ { 1 }$ , and three missing-label proportions yielded 72 simulation settings.

We compared three estimators: $\mathrm { C C } ,$ obtained from the likelihood for the completely classified sample in (2); ig, obtained from the likelihood in Section 3.1 that ignores the label-missingness mechanism; and full, obtained from the full likelihood in (4), which jointly estimates θ and $( \xi _ { 0 } , \xi _ { 1 } )$ . For each fitted model, we evaluated the error rate of the plug-in rule directly under the data-generating distribution, avoiding Monte Carlo error from a finite test sample.

For method $s \in \{ \mathrm { C C } , \mathrm { i g } , \mathrm { f u l l } \}$ , the mean excess error rate was defined as

$$
\overline { { E } } _ { s } = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \left[ R \left( \widehat { \theta } _ { s } ^ { ( b ) } ; \theta \right) - \mathrm { e r r } ( \theta ) \right] .
$$

The empirical relative efficiency of method $s ,$ relative to the estimator based on the completely classified sample, was calculated as

$$
\widehat { \mathrm { R E } } _ { s : \mathrm { C C } } = \frac { \overline { E } _ { \mathrm { C C } } } { \overline { E } _ { s } } .
$$

A value greater than one indicates that method s has a smaller mean excess error rate than the benchmark based on the completely classified sample.

Remark 4 (Computation). Each likelihood was maximised numerically from five starting values. Among the runs returning a finite objective value, the solution with the largest log likelihood was retained. A fit was classified as converged when the numerical optimiser returned a successful convergence status. Convergence rates ranged from 98% to 100% across all methods and simulation settings, with most settings attaining 100% convergence. Nonconverged and failed fits were retained in the simulation records but excluded from the Monte Carlo summaries, which were calculated from the successfully converged replications.

## 5.2 Expected error-rate efficiency

Figure 2 summarises the empirical relative efficiencies across the six designs, and Tables 2 and 3 report the corresponding numerical values. When $\xi _ { 1 } = 0 ,$ , the full and ig estimators perform similarly. As the MAR dependence on the observed feature strengthens with decreasing $\xi _ { 1 }$ , the full estimator shows progressively larger gains over the ig estimator.

The gain from modelling the missing-label mechanism becomes more pronounced as the missing-label proportion increases. Under the common-shape designs, the full estimator attains empirical relative efficiencies above six in several settings with strong MAR dependence on the observed feature. The unequal-shape designs exhibit the same qualitative pattern, although the gains are generally more moderate. By contrast, the empirical relative efficiency of the ig estimator remains below one in nearly all settings and decreases as the missing-label proportion increases.

![](images/f25e0a9256156c4b1163497b5ce2a2d84466cc891bd1c579a9746de6c9ef450c.jpg)  
Figure 2: Empirical relative efficiency relative to the estimator based on the completely classified sample (CC). Panel (A) shows the common-shape designs and Panel (B) the unequal-shape designs. Blue and orange curves correspond to the ig and full estimators, respectively. Line types indicate the missing-label proportions (10%, 30%, and 50%). The horizontal dashed line marks relative efficiency equal to one.

## 5.3 Parameter estimation

Figure 3 provides complementary evidence from parameter estimation. It reports empirical root mean squared errors for the mixing proportion and the Weibull scale and shape parameters. For visual clarity, the RMSEs are averaged over designs C1–C3 within the common-shape family and over designs U1–U3 within the unequal-shape family. At $\xi _ { 1 } = 0$ , the full and ig estimators have broadly similar accuracy. As the MAR dependence on classification uncertainty strengthens, the full estimator generally has smaller RMSE, particularly at the larger missing-label proportions.

These reductions in parameter-estimation RMSE occur in both model families. The CC estimator provides a baseline reference, while the full estimator often approaches or improves upon the CC RMSE when MAR label missingness is concentrated near the decision boundary.

## 5.4 Decision-boundary estimation

The primary inferential target is the plug-in classification rule rather than the model parameters alone. We therefore evaluated the RMSE of the estimated Bayes decision boundaries. Figure 4 indicates that the full estimator generally yields smaller boundary RMSE than the ig estimator. The difference increases as the MAR dependence on classification uncertainty strengthens and as the missing-label proportion rises.

Table 2: Empirical relative efficiencies under the common-shape Weibull models.
<table><tr><td rowspan="2">Design</td><td rowspan="2"> $\xi _ { 1 }$ </td><td colspan="2">10% missing</td><td colspan="2">30% missing</td><td colspan="2">50% missing</td></tr><tr><td>ig</td><td>full</td><td>ig</td><td>full</td><td> $\mathrm { i g }$ </td><td>full</td></tr><tr><td rowspan="4">C1</td><td>0.0</td><td>0.918</td><td>0.917</td><td>0.752</td><td>0.757</td><td>0.503</td><td>0.500</td></tr><tr><td>-0.5</td><td>0.916</td><td>1.060</td><td>0.719</td><td>1.129</td><td>0.411</td><td>0.929</td></tr><tr><td>-1.0</td><td>0.931</td><td>1.350</td><td>0.686</td><td>1.883</td><td>0.404</td><td>1.973</td></tr><tr><td>-2.0</td><td>0.915</td><td>2.196</td><td>0.659</td><td>4.269</td><td>0.401</td><td>4.877</td></tr><tr><td rowspan="4">C2</td><td>0.0</td><td>0.925</td><td>0.925</td><td>0.774</td><td>0.775</td><td>0.506</td><td>0.505</td></tr><tr><td>-0.5</td><td>0.945</td><td>1.138</td><td>0.692</td><td>1.466</td><td>0.384</td><td>1.493</td></tr><tr><td>-1.0</td><td>0.916</td><td>1.562</td><td>0.581</td><td>2.959</td><td>0.401</td><td>3.228</td></tr><tr><td>-2.0</td><td>0.851</td><td>2.699</td><td>0.591</td><td>5.931</td><td>0.397</td><td>6.579</td></tr><tr><td rowspan="4">C3</td><td>0.0</td><td>0.941</td><td>0.941</td><td>0.831</td><td>0.831</td><td>0.680</td><td>0.680</td></tr><tr><td>-0.5</td><td>0.726</td><td>1.870</td><td>0.528</td><td>2.569</td><td>0.440</td><td>2.259</td></tr><tr><td>-1.0</td><td>0.676</td><td>3.292</td><td>0.457</td><td>4.210</td><td>0.425</td><td>3.473</td></tr><tr><td>-2.0</td><td>0.709</td><td>6.131</td><td>0.485</td><td>6.586</td><td>0.444</td><td>4.886</td></tr></table>

Table 3: Empirical relative efficiencies under the unequal-shape Weibull models.
<table><tr><td rowspan="2">Design</td><td rowspan="2"> $\xi _ { 1 }$ </td><td colspan="2">10% missing</td><td colspan="2">30% missing</td><td colspan="2">50% missing</td></tr><tr><td>ig</td><td>full</td><td>ig</td><td>full</td><td>ig</td><td>full</td></tr><tr><td rowspan="4">U1</td><td>0.0</td><td>0.918</td><td>0.916</td><td>0.727</td><td>0.718</td><td>0.533</td><td>0.524</td></tr><tr><td>-0.5</td><td>0.919</td><td>1.019</td><td>0.741</td><td>1.017</td><td>0.529</td><td>1.004</td></tr><tr><td>-1.0</td><td>0.912</td><td>1.202</td><td>0.723</td><td>1.451</td><td>0.521</td><td>1.862</td></tr><tr><td>-2.0</td><td>0.929</td><td>1.729</td><td>0.696</td><td>3.070</td><td>0.540</td><td>4.443</td></tr><tr><td rowspan="4">U2</td><td>0.0</td><td>0.917</td><td>0.901</td><td>0.720</td><td>0.706</td><td>0.547</td><td>0.524</td></tr><tr><td>-0.5</td><td>0.925</td><td>1.054</td><td>0.729</td><td>1.013</td><td>0.551</td><td>1.041</td></tr><tr><td>-1.0</td><td>0.920</td><td>1.251</td><td>0.694</td><td>1.582</td><td>0.555</td><td>1.853</td></tr><tr><td>-2.0</td><td>0.909</td><td>1.945</td><td>0.687</td><td>3.378</td><td>0.509</td><td>4.127</td></tr><tr><td rowspan="4">U3</td><td>0.0</td><td>0.907</td><td>0.894</td><td>0.711</td><td>0.703</td><td>0.549</td><td>0.540</td></tr><tr><td>-0.5</td><td>0.920</td><td>1.081</td><td>0.682</td><td>1.060</td><td>0.524</td><td>1.069</td></tr><tr><td>-1.0</td><td>0.907</td><td>1.353</td><td>0.667</td><td>1.792</td><td>0.487</td><td>2.121</td></tr><tr><td>-2.0</td><td>0.896</td><td>2.220</td><td>0.657</td><td>3.987</td><td>0.475</td><td>5.273</td></tr></table>

The common-shape designs contain a single positive decision boundary, whereas the unequal-shape designs contain two. In the unequal-shape case, the full estimator reduces RMSE for both boundaries, with particularly clear gains for the second boundary under strong MAR label missingness. These results indicate that modelling the missing-label mechanism improves estimation of the classification rule, not merely estimation of the mixture parameters.

## 6 A semi-synthetic study using hard-drive failure data

We analysed Backblaze failure records for two frequently observed hard-drive product models Arslan and Zeydan (2020). For drive j, the observed lifetime $Y _ { j }$ was the positive-valued feature and $\bar { Z } _ { j } \in \{ 1 , 2 \}$ denoted drive type, with ST4000DM000 coded as class 1 and ST8000DM002 as class 2. The objective was therefore to classify drive type among failed drives, not to predict future drive failure. After excluding the separate committee-training subset described in Appendix D and two observations with zero recorded lifetime, the analysis sample contained n = 194 drives: 98 from class 1 and 96 from class 2.

To emulate an expert-labelling setting in which ambiguous observations are more likely to remain unlabelled, we constructed three nested label-missingness regimes retaining approximately 60%, 70%, and 80% of the class labels. These regimes represent stylised junior, intermediate, and senior levels of expert assessment. Labels were preferentially retained for observations that were easier to classify, whereas observations nearer the decision boundary were more likely to remain unlabelled. Details are given in Appendix D.

![](images/8e5960e6e0205eda4ea5b9bd0e652755316b4090cb8adc688ca2b36f8dc34572.jpg)  
Figure 3: RMSEs of the estimated model parameters. Panel (A) summarises the common-shape models, displaying the mixing proportion, two scale parameters, and the common shape parameter. Panel (B) summarises the unequalshape models, displaying the mixing proportion, two scale parameters, and two component-specific shape parameters. RMSEs are averaged over the three designs within each shape-model family. Black, blue, and orange curves correspond to the CC, ig, and full estimators, respectively. Line types indicate the missing-label proportions.

Using the completely classified sample as the benchmark, we compared the common- and unequal-shape Weibull mixtures. BIC favoured the common-shape model (2845.142 versus 2847.951), and the likelihood-ratio test did not reject the common-shape restriction $( \chi _ { 1 } ^ { 2 } = 2 . 4 5 9 , p = 0 . 1 1 7 )$ . We therefore used the common-shape model in this study. At each expert level, we compared the completely classified estimator (CC), the estimator that ignored the missingness mechanism (ig), and the full-likelihood estimator (full). The latter used the MAR working model in (3), with missingness depending on the observed lifetime through the squared log-posterior odds.

(A) Common-shape models: Boundary  
![](images/611d2d3e887e6110a2a8ed28b2c40057e6be4832b1e6e67ddedc08b38f179578.jpg)

(B) Unequal-shape models: Boundaries 1 and 2  
![](images/0e3cf90ccd723d97667b7e7723cca83c40a15fa9867079f07afb0444a37ba9da.jpg)  
Method CC ig full Missing-label proportion 10% 30% 50%  
Figure 4: RMSEs of the estimated Bayes decision boundaries. Panel (A) shows the single decision boundary for the common-shape models, whereas Panel (B) shows the two decision boundaries for the unequal-shape models. Black, blue, and orange curves correspond to the CC, ig, and full estimators, respectively. Line types indicate the missinglabel proportions.

Labels were more likely to be absent near the estimated Bayes decision boundary. The CC estimate was ${ \widehat { y } } _ { 0 , \mathrm { C C } } =$ 287.94, whereas the ig estimates were substantially lower. Incorporating the feature-dependent missing-label pattern moved the full estimates closer to the CC benchmark.

Predictive misclassification error was estimated by repeated stratified five-fold cross-validation. The same folds were used for all three expert regimes, so the cross-validated CC result was identical across regimes. The full estimator had the lowest mean estimated error rate at every expert level. Relative to ig, the mean error rate decreased by 0.076, 0.075, and 0.053 for the junior, intermediate, and senior experts, respectively; the corresponding decreases relative to

CC were 0.036, 0.036, and 0.030. The estimated decision boundaries and cross-validated error rates are summarised in Table 4.

Table 4: Estimated decision boundaries and mean estimated error rates. Error rates were evaluated using five repetitions of five-fold cross-validation.
<table><tr><td></td><td colspan="3">Decision boundary</td><td colspan="3">Estimated error rate</td></tr><tr><td>Expert</td><td>CC</td><td>ig</td><td>full</td><td>CC</td><td>ig</td><td>full</td></tr><tr><td>Junior</td><td>287.94</td><td>158.91</td><td>328.71</td><td>0.427</td><td>0.467</td><td>0.391</td></tr><tr><td>Intermediate</td><td>287.94</td><td>150.83</td><td>343.38</td><td>0.427</td><td>0.466</td><td>0.391</td></tr><tr><td>Senior</td><td>287.94</td><td>180.19</td><td>322.93</td><td>0.427</td><td>0.450</td><td>0.397</td></tr></table>

Decision boundaries were estimated by fitting each method to the full analysis sample. Error rates are one minus the corresponding classification accuracies and are averaged over five repetitions of stratified five-fold cross-validation, with all models refitted in the corresponding training folds.

Therefore, modelling the feature-dependent missing-label mechanism produced decision-boundary estimates closer to the completely classified benchmark and reduced the cross-validated error rate in this example.

## 7 Discussion

We have studied semi-supervised classification from a two-component Weibull mixture under a shared-parameter MAR label-missingness model. The Weibull setting leads to one positive decision boundary under a nonconstant common-shape Bayes’ rule and can lead to two under unequal shapes. After adjustment for nuisance parameters in the missingness model, the Fisher information and the expected error rate of the plug-in sample rule can both be expressed in forms that make the missing-label pattern explicit. The decision-boundary expansion shows why improved estimation of all model parameters is not required for a classification gain: the first-order error rate depends on uncertainty only in directions that perturb the decision boundaries. The simulations and the semi-synthetic analysis based on hard-drive failure data support this distinction. Extensions to nonregular boundary configurations, censored Weibull data, more flexible missingness models, and multiclass settings remain useful directions for further work (Ahfock and McLachlan, 2023; Wu et al., 2026).

## A Decision-boundary proof

ProofofProposition 1. Set $t = \log y$ and $\phi ( t ) = \delta ( e ^ { t } ; \theta _ { U } )$ . Then

$$
\begin{array} { r l } & { \phi ^ { \prime } ( t ) = k _ { 1 } - k _ { 2 } - k _ { 1 } \exp \{ k _ { 1 } ( t - \alpha _ { 1 } ) \} + k _ { 2 } \exp \{ k _ { 2 } ( t - \alpha _ { 2 } ) \} , } \\ & { \phi ^ { \prime \prime } ( t ) = - k _ { 1 } ^ { 2 } \exp \{ k _ { 1 } ( t - \alpha _ { 1 } ) \} + k _ { 2 } ^ { 2 } \exp \{ k _ { 2 } ( t - \alpha _ { 2 } ) \} . } \end{array}
$$

The equation $\phi ^ { \prime \prime } ( t ) = 0$ has exactly one solution because the ratio of its two exponential terms is strictly monotone when $\boldsymbol { k } _ { 1 } \neq \boldsymbol { k } _ { 2 }$

$\mathrm { I f } \ k _ { 1 } > k _ { 2 }$ , then $\phi ^ { \prime \prime } ( t ) > 0$ for sufficiently negative t and $\phi ^ { \prime \prime } ( t ) < 0$ for sufficiently positive t. Hence $\phi ^ { \prime }$ first increases and then decreases. Moreover,

$$
\operatorname* { l i m } _ { t \to - \infty } \phi ^ { \prime } ( t ) = k _ { 1 } - k _ { 2 } > 0 , \qquad \operatorname* { l i m } _ { t \to \infty } \phi ^ { \prime } ( t ) = - \infty .
$$

Therefore $\phi ^ { \prime }$ has exactly one zero and $\phi$ has a unique maximum. From (1),

$$
\operatorname * { l i m } _ { t \to - \infty } \phi ( t ) = - \infty , \qquad \operatorname * { l i m } _ { t \to \infty } \phi ( t ) = - \infty .
$$

Thus $\phi$ has zero, one tangential zero at its unique maximum, or two simple zeros, and it is positive only between two simple zeros. It remains to verify that the tangential zero, when present, is exactly double. Let $t _ { \star }$ denote the unique stationary point and write

$$
r _ { i } ^ { \star } = \exp \{ k _ { i } ( t _ { \star } - \alpha _ { i } ) \} > 0 .
$$

If both $\phi ^ { \prime } ( t _ { \star } ) = 0$ and $\phi ^ { \prime \prime } ( t _ { \star } ) = 0$ , then

$$
k _ { 1 } - k _ { 2 } - k _ { 1 } r _ { 1 } ^ { \star } + k _ { 2 } r _ { 2 } ^ { \star } = 0 , \qquad - k _ { 1 } ^ { 2 } r _ { 1 } ^ { \star } + k _ { 2 } ^ { 2 } r _ { 2 } ^ { \star } = 0 .
$$

The second equation gives $r _ { 2 } ^ { \star } = ( k _ { 1 } ^ { 2 } / k _ { 2 } ^ { 2 } ) r _ { 1 } ^ { \star }$ , and substitution into the first gives

$$
\left( k _ { 1 } - k _ { 2 } \right) \left( 1 + \frac { k _ { 1 } } { k _ { 2 } } r _ { 1 } ^ { \star } \right) = 0 ,
$$

which is impossible because $k _ { 1 } \neq k _ { 2 }$ and $r _ { 1 } ^ { \star } > 0$ . Hence $\phi ^ { \prime \prime } ( t _ { \star } ) \neq 0$ . Therefore, i $\dot { { \bf \varphi } } \phi ( t _ { \star } ) = 0 ,$ , then $t _ { \star }$ is a zero of multiplicity exactly two. Since the map $y = e ^ { t }$ is a smooth one-to-one reparameterization with nonzero derivative, the corresponding positive root of $\delta ( y )$ also has multiplicity exactly two.

If $k _ { 1 } < k _ { 2 }$ , the signs reverse: $\phi ^ { \prime }$ first decreases and then increases,

$$
\operatorname* { l i m } _ { t \to - \infty } \phi ^ { \prime } ( t ) = k _ { 1 } - k _ { 2 } < 0 , \qquad \operatorname* { l i m } _ { t \to \infty } \phi ^ { \prime } ( t ) = + \infty ,
$$

so $\phi$ has a unique minimum. Also $\phi ( t ) \to + \infty$ in both tails. The same argument shows that a tangential zero at the minimum has multiplicity exactly two. The stated root count and class regions follow. □

## B Derivation of the information identity

Proof of Lemma 1. From (5),

$$
\ell _ { \mathrm { i g } , 1 } ( \theta ) = \log p _ { Y } ( Y ; \theta ) + ( 1 - M ) \log p ( Z \mid Y ; \theta ) ,
$$

so

$$
\psi _ { \mathrm { i g } } ( \theta ) = S _ { Y } ( \theta ) + ( 1 - M ) S _ { Z | Y } ( \theta ) .
$$

Conditional score identities give

$$
\operatorname { E } \{ S _ { Z | Y } ( \theta ) \mid Y \} = 0 , \qquad - \operatorname { E } \{ { \frac { \partial S _ { Z | Y } ( \theta ) } { \partial \theta ^ { \top } } } | Y \} = { \mathcal { I } } _ { Z | Y } ( Y ; \theta ) .
$$

Because $M \perp Z \mid Y ,$

$$
\operatorname { E } \{ ( 1 - M ) S _ { Z \mid Y } ( \theta ) \mid Y \} = \{ 1 - q ( Y ) \} \operatorname { E } \{ S _ { Z \mid Y } ( \theta ) \mid Y \} = 0 .
$$

Hence the cross terms between $S _ { Y }$ and $( 1 - M ) S _ { Z | Y }$ vanish. Since $( 1 - M ) ^ { 2 } = 1 - M$

$$
\begin{array} { r l r } & { } & { \mathrm { E } \{ \psi _ { \mathrm { i g } } \psi _ { \mathrm { i g } } ^ { \top } \} = \mathcal { I } _ { Y } + \mathrm { E } \big [ \{ 1 - q ( Y ) \} \mathcal { I } _ { Z | Y } ( Y ; \theta ) \big ] , } \\ & { } & { - \mathrm { E } \left\{ \frac { \partial \psi _ { \mathrm { i g } } } { \partial \theta ^ { \top } } \right\} = \mathcal { I } _ { Y } + \mathrm { E } \big [ \{ 1 - q ( Y ) \} \mathcal { I } _ { Z | Y } ( Y ; \theta ) \big ] , } \end{array}
$$

where $\mathcal { T } _ { Y }$ is the Fisher information in the marginal feature density $p _ { Y }$ . The complete-data score decomposition

$$
\log p ( Y , Z ; \theta ) = \log p _ { Y } ( Y ; \theta ) + \log p ( Z \mid Y ; \theta )
$$

gives

$$
\mathcal { I } _ { \mathrm { C C } } = \mathcal { I } _ { Y } + \mathrm { E } \{ \mathcal { I } _ { Z | Y } ( Y ; \theta ) \} .
$$

Therefore,

$$
\mathcal { I } _ { Y } + \mathrm { E } \big [ \{ 1 - q ( Y ) \} \mathcal { I } _ { Z | Y } ( Y ; \theta ) \big ] = \mathcal { I } _ { \mathrm { C C } } - \mathrm { E } \{ q ( Y ) \mathcal { I } _ { Z | Y } ( Y ; \theta ) \} = \mathcal { I } _ { \mathrm { i g } } ,
$$

by $( 6 ) – ( 7 )$ . Thus the sensitivity and variability matrices of the ignoring score coincide, proving (8). Under the additional consistency, nonsingularity, and standard local M-estimation regularity conditions stated in Lemma 1, the usual asymptotic linearization applies. Its sandwich covariance reduces to

$$
\mathcal { T } _ { \mathrm { i g } } ^ { - 1 } \mathcal { I } _ { \mathrm { i g } } \mathcal { I } _ { \mathrm { i g } } ^ { - 1 } = \mathcal { I } _ { \mathrm { i g } } ^ { - 1 } ,
$$

which proves (9).

□

Derivationfor Theorem 1. Since

$$
\ell _ { \mathrm { f u l l } } ( \theta , \xi ) = \ell _ { \mathrm { i g } } ( \theta ) + \ell _ { \mathrm { m i s s } } ( \theta , \xi ) ,
$$

linearity of the expected negative Hessian implies that the information contributions from the ignored likelihood and the Bernoulli missingness likelihood add, while the blocks involving $\xi$ arise only from the missingness likelihood. By Lemma 1 and (7), the θθ block contributed by $\ell _ { \mathrm { i g } }$ is $\mathcal { I } _ { \mathrm { i g } }$ . Hence the joint information matrix for $( \breve { \theta } ^ { \top } , \xi ^ { \top } ) ^ { \top }$ is

$$
\left( \begin{array} { c c } { { \mathcal { I } } _ { \mathrm { i g } } + B _ { \theta \theta } } & { B _ { \theta \xi } } \\ { B _ { \xi \theta } } & { B _ { \xi \xi } } \end{array} \right) .
$$

Profiling out the nuisance parameter $\xi$ gives the Schur complement of $B _ { \xi \xi }$ :

$$
\mathcal { I } _ { \mathrm { i g } } + B _ { \theta \theta } - B _ { \theta \xi } B _ { \xi \xi } ^ { - 1 } B _ { \xi \theta } .
$$

Substituting (7) and (10) yields

$$
\mathcal { I } _ { \mathrm { f u l l } } = \mathcal { I } _ { \mathrm { C C } } - \mathcal { I } _ { \mathrm { l o s t } } + \mathcal { I } _ { \mathrm { m i s s } } ^ { \mathrm { a d j } } ,
$$

which proves (11). Moreover,

$$
\mathcal { I } _ { \mathrm { f u l l } } - \mathcal { I } _ { \mathrm { C C } } = \mathcal { I } _ { \mathrm { m i s s } } ^ { \mathrm { a d j } } - \mathcal { I } _ { \mathrm { l o s t } } .
$$

Thus, if (12) holds and both $\mathcal { I } _ { \mathrm { f u l l } }$ and $\mathcal { I } _ { \mathrm { C C } }$ are positive definite, then

$$
\mathcal { I } _ { \mathrm { f u l l } } > \mathcal { I } _ { \mathrm { C C } } .
$$

Inversion reverses the Loewner order, giving

$$
\mathcal { T } _ { \mathrm { f u l l } } ^ { - 1 } < \mathcal { T } _ { \mathrm { C C } } ^ { - 1 } .
$$

□

## C Proof of the expected error-rate expansion

Proof of Theorem 2. Write

Since

$$
h ( y ) = \pi _ { 1 } f _ { 1 } ( y ) - \pi _ { 2 } f _ { 2 } ( y ) .
$$

$$
\tau _ { 1 } ( y ; \theta ) = \frac { e ^ { \delta ( y ; \theta ) } } { 1 + e ^ { \delta ( y ; \theta ) } } ,
$$

we have

$$
\begin{array} { c } { { h ( y ) = p _ { Y } ( y ; \theta ) \{ 2 \tau _ { 1 } ( y ; \theta ) - 1 \} } } \\ { { = p _ { Y } ( y ; \theta ) \operatorname { t a n h } \left\{ \displaystyle { \frac { \delta ( y ; \theta ) } { 2 } } \right\} . } } \end{array}
$$

Let $y _ { j }$ be a Bayes decision boundary. Then $\delta ( y _ { j } ; \theta ) = 0$ , and hence $h ( y _ { j } ) = 0$ . Because $p _ { Y } ( \cdot ; \theta )$ is continuous at $y _ { j }$ , $\delta ( \cdot ; \breve { \theta } )$ is differentiable at $y _ { j }$ , and

$$
\operatorname { t a n h } ( u / 2 ) = u / 2 + o ( u ) \qquad { \mathrm { a s ~ } } u \to 0 ,
$$

it follows that

$$
h ( y ) = \frac { p _ { Y } ( y _ { j } ; \theta ) } { 2 } \delta _ { y } ( y _ { j } ; \theta ) ( y - y _ { j } ) + o ( | y - y _ { j } | ) \qquad \mathrm { a s } ~ y \to y _ { j } .\tag{22}
$$

In particular, h is differentiable at $y _ { j }$ , with

$$
h ^ { \prime } ( y _ { j } ) = \frac { p _ { Y } ( y _ { j } ; \theta ) } { 2 } \delta _ { y } ( y _ { j } ; \theta ) .
$$

By Assumption 2 and the implicit function theorem, for $\widetilde { \theta }$ in a neighbourhood of $\theta ,$ the boundary $y _ { j }$ extends to a differentiable root $y _ { j } ( { \widetilde { \theta } } )$ satisfying

$$
\delta \{ y _ { j } ( \widetilde { \theta } ) ; \widetilde { \theta } \} = 0 .
$$

A first-order expansion about $( y _ { j } , \theta )$ gives

$$
y _ { j } ( \widetilde { \theta } ) - y _ { j } = - \frac { g ( y _ { j } ; \theta ) ^ { \top } ( \widetilde { \theta } - \theta ) } { \delta _ { y } ( y _ { j } ; \theta ) } + o ( \Vert \widetilde { \theta } - \theta \Vert ) .\tag{23}
$$

The excess error rate relative to the Bayes error admits the representation

$$
R ( \widetilde { \theta } ; \theta ) - R ( \theta ; \theta ) = \int _ { 0 } ^ { \infty } | h ( y ) | { \bf 1 } \left\{ \mathrm { s i g n } \delta ( y ; \widetilde { \theta } ) \neq \mathrm { s i g n } \delta ( y ; \theta ) \right\} \mathrm { d } y .
$$

By the stability and simplicity of the Bayes decision boundaries, for $\widetilde { \theta }$ sufficiently close to $\theta ,$ disagreement between the two rules occurs only on the disjoint intervals having endpoints $y _ { j }$ and $y _ { j } ( { \widetilde { \theta } } )$

Using (22), we obtain

$$
\begin{array} { r l } & { \int _ { \operatorname* { m i n } \{ y _ { j } , y _ { j } ( \widetilde { \theta } ) \} } ^ { \operatorname* { m a x } \{ y _ { j } , y _ { j } ( \widetilde { \theta } ) \} } | h ( y ) | \mathrm { d } y } \\ & { \quad = \frac { p _ { Y } \left( y _ { j } ; \theta \right) | \delta _ { y } \left( y _ { j } ; \theta \right) | } { 4 } \{ y _ { j } ( \widetilde { \theta } ) - y _ { j } \} ^ { 2 } + o \left( \{ y _ { j } ( \widetilde { \theta } ) - y _ { j } \} ^ { 2 } \right) } \\ & { \quad = \frac { p _ { Y } \left( y _ { j } ; \theta \right) } { 4 | \delta _ { y } ( y _ { j } ; \theta ) | } \left\{ g ( y _ { j } ; \theta ) ^ { \top } ( \widetilde { \theta } - \theta ) \right\} ^ { 2 } + o ( \Vert \widetilde { \theta } - \theta \Vert ^ { 2 } ) , } \end{array}
$$

where the second equality follows from (23).

Summing over the finitely many boundaries yields

$$
R ( \widetilde { \boldsymbol { \theta } } ; \boldsymbol { \theta } ) - \mathrm { e r r } ( \boldsymbol { \theta } ) = \frac { 1 } { 2 } ( \widetilde { \boldsymbol { \theta } } - \boldsymbol { \theta } ) ^ { \top } H _ { R } ( \boldsymbol { \theta } ) ( \widetilde { \boldsymbol { \theta } } - \boldsymbol { \theta } ) + o ( \Vert \widetilde { \boldsymbol { \theta } } - \boldsymbol { \theta } \Vert ^ { 2 } ) ,\tag{24}
$$

where

$$
H _ { R } ( \theta ) = \sum _ { j = 1 } ^ { K } \frac { p _ { Y } ( y _ { j } ; \theta ) } { 2 | \delta _ { y } ( y _ { j } ; \theta ) | } g ( y _ { j } ; \theta ) g ( y _ { j } ; \theta ) ^ { \top } .
$$

This proves (13) and also shows that

$$
\left. \frac { \partial R ( \widetilde { \theta } ; \theta ) } { \partial \widetilde { \theta } } \right| _ { \widetilde { \theta } = \theta } = 0 .
$$

Now set

$$
\Delta _ { n } = { \widehat { \theta } } - \theta .
$$

By root-n consistency, $\Delta _ { n } = O _ { p } ( n ^ { - 1 / 2 } )$ , and therefore (24) gives

$$
R ( \widehat { \theta } ; \theta ) - \mathrm { e r r } ( \theta ) = \frac { 1 } { 2 } \Delta _ { n } ^ { \top } H _ { R } ( \theta ) \Delta _ { n } + o _ { p } ( n ^ { - 1 } ) .\tag{25}
$$

To justify taking expectations in (25), fix a radius $r > 0$ such that the local expansion (24) holds whenever $\| \Delta \| \leq r ,$ and write

$$
\displaystyle { \mathcal { A } } _ { n } = \{ \| \Delta _ { n } \| \leq r \} .
$$

Define

$$
\mathrm { r e m } ( \Delta ) = R ( \theta + \Delta ; \theta ) - \mathrm { e r r } ( \theta ) - \frac { 1 } { 2 } \Delta ^ { \top } H _ { R } ( \theta ) \Delta .
$$

By (24),

$$
\mathrm { r e m } ( \Delta ) = o ( \| \Delta \| ^ { 2 } ) \qquad \mathrm { a s } \Delta \to 0 .
$$

Hence there exists a nonnegative deterministic function $\eta ,$ with $\eta ( t )  0$ as $t \downarrow 0$ , such that

$$
| \operatorname { r e m } ( \Delta ) | \leq \eta ( \| \Delta \| ) \| \Delta \| ^ { 2 } , \qquad \| \Delta \| \leq r .
$$

Condition (15) implies that $\{ n \| \Delta _ { n } \| ^ { 2 } \}$ is uniformly integrable, while root-n consistency gives $\eta (  \Delta _ { n }  )  0$ in probability. Hence

$$
n \operatorname { E } [ | \operatorname { r e m } ( \Delta _ { n } ) | \mathbf { 1 } _ { \mathcal { A } _ { n } } ] \longrightarrow 0 .
$$

Moreover, Markov’s inequality and (15) give

$$
\begin{array} { r } { \operatorname* { P r } ( \mathcal { A } _ { n } ^ { c } ) \leq r ^ { - ( 2 + \epsilon ) } \operatorname { E } \| \Delta _ { n } \| ^ { 2 + \epsilon } = O ( n ^ { - 1 - \epsilon / 2 } ) = o ( n ^ { - 1 } ) . } \end{array}
$$

Because

$$
0 \leq R ( \widehat { \theta } ; \theta ) - \operatorname { e r r } ( \theta ) \leq 1 ,
$$

the contribution of $\mathcal { A } _ { n } ^ { c }$ to the expected excess risk is $o ( n ^ { - 1 } )$ . Uniform integrability of $n \| \Delta _ { n } \| ^ { 2 }$ also gives

$$
\operatorname { E } \big [ \| \Delta _ { n } \| ^ { 2 } \mathbf { 1 } _ { \mathcal { A } _ { n } ^ { c } } \big ] = o ( n ^ { - 1 } ) ,
$$

so the quadratic term may likewise be restricted to $\mathcal { A } _ { n }$ without changing its expectation at order $n ^ { - 1 }$ . Consequently,

$$
\begin{array} { l } { { \displaystyle \mathrm { E } \{ R ( \widehat \theta ; \theta ) \} - \mathrm { e r r } ( \theta ) = \frac { 1 } { 2 } \mathrm { t r } \left[ H _ { R } ( \theta ) \mathrm { E } \{ \Delta _ { n } \Delta _ { n } ^ { \top } \} \right] + o ( n ^ { - 1 } ) } } \\ { ~ } \\ { { \displaystyle ~ = \frac { 1 } { 2 n } \mathrm { t r } \{ H _ { R } ( \theta ) V \} + o ( n ^ { - 1 } ) } , } \end{array}
$$

where the second equality follows from (14). Substitution of (13) proves (16). For a correctly specified regular maximum-likelihood estimator, setting $V = \mathcal { T } ( \theta ) ^ { - 1 }$ gives (17); for the ignoring estimator, Lemma 1 gives the corresponding result with $V = \mathcal { I } _ { \mathrm { i g } } ^ { - 1 }$ □

## D Hard-drive expert-label construction and diagnostics

A separate, fully labelled training subset was used to fit a committee of seven classifiers: logistic regression, linear discriminant analysis, regularised discriminant analysis, a support vector machine, random forest, k-nearest neighbours, and naive Bayes. To avoid information leakage, this training subset was excluded from the subsequent Weibull-mixture analysis. For observation j, we constructed the following committee-based ranking score:

$$
\kappa _ { j } = a _ { j } + \left| \overline { { p } } _ { 2 j } - \frac { 1 } { 2 } \right| - s _ { 2 j } ,
$$

where $a _ { j }$ denotes the proportion of classifiers agreeing on the predicted class, $\overline { { p } } _ { 2 j }$ is the mean predicted probability of class 2 across the committee, and $s _ { 2 j }$ is the corresponding between-classifier standard deviation. Thus, a larger value of $\kappa _ { j }$ reflects stronger agreement, a mean prediction farther from the classification threshold of $1 / 2 .$ , and less variation among the committee members. The score is used only to rank observations for the semi-synthetic label-missingness construction.

Observations were ranked in decreasing order of $\kappa _ { j } .$ . Labels were then released for the highest-ranked 60%, 70%, and 80% of observations to form the junior, intermediate, and senior expert regimes, respectively. This construction represents the idea that more experienced experts can assign labels to a larger proportion of ambiguous observations. The three labelled sets were nested, and the resulting missing-label fractions were 0.402, 0.304, and 0.201, respectively.

For expert regime $e \in \{ \mathrm { J } , \mathrm { I } , \mathrm { S } \}$ , let $M _ { j } ^ { ( e ) }$ denote the missing-label indicator for observation $j ,$ and let $\widehat { \theta } _ { \mathrm { C C } }$ denote the common-shape CC estimate obtained from the fully labelled analysis sample. We then fitted the descriptive diagnostic model

$$
\log \mathrm { i t } \left\{ \mathrm { P r } \Big ( M _ { j } ^ { ( e ) } = 1 \mid Y _ { j } \Big ) \right\} = \beta _ { 0 e } + \beta _ { 1 e } \delta _ { C } \Big ( Y _ { j } ; \widehat { \theta } _ { \mathrm { C C } } \Big ) ^ { 2 } .
$$

The estimated slopes were negative in all three regimes (Table 5), consistent with observations closer to the estimated Bayes decision boundary being more likely to remain unlabelled. These estimates are used only as descriptive diagnostics; their magnitudes are not directly comparable across regimes because the regimes have different overall missing-label fractions.

Table 5: Descriptive association between squared log-posterior odds and label missingness.
<table><tr><td>Expert</td><td>Missing fraction</td><td> $\widehat { \beta } _ { 1 e }$ </td></tr><tr><td>Junior</td><td>0.402</td><td>-14.157</td></tr><tr><td>Intermediate</td><td>0.304</td><td>-11.910</td></tr><tr><td>Senior</td><td>0.201</td><td>-32.957</td></tr></table>

The class-specific cross-validation results are reported in Table 6. The same folds were used for all three expert regimes, so the CC rates were identical across regimes. Relative to ig, full increased class 2 sensitivity but reduced class 1 specificity. Its lower overall estimated error rate therefore did not arise from uniform improvements in both classes.

Table 6: Mean class-specific rates from five repetitions of five-fold cross-validation.
<table><tr><td></td><td colspan="3">Class 2 sensitivity</td><td colspan="3">Class 1 specificity</td></tr><tr><td>Expert</td><td>CC</td><td>ig</td><td>full</td><td>CC</td><td>ig</td><td>full</td></tr><tr><td>Junior</td><td>0.633</td><td>0.355</td><td>0.735</td><td>0.514</td><td>0.708</td><td>0.485</td></tr><tr><td>Intermediate</td><td>0.633</td><td>0.346</td><td>0.748</td><td>0.514</td><td>0.718</td><td>0.473</td></tr><tr><td>Senior</td><td>0.633</td><td>0.412</td><td>0.721</td><td>0.514</td><td>0.685</td><td>0.487</td></tr></table>

## References

Ahfock, D. and McLachlan, G. J. (2020). An apparent paradox: A classifier based on a partially classified sample may have smaller expected error rate than that if the sample were completely classified. Statistics and Computing, 30(6):1779–1790.

Ahfock, D. and McLachlan, G. J. (2023). Semi-supervised learning of classifiers from a statistical perspective: A brief review. Econometrics and Statistics, 26:124–138.

Ahmad, K. and Abd-Elrahman, A. (1994). Updating a nonlinear discriminant function estimated from a mixture of two Weibull distributions. Mathematical and Computer Modelling, 19(11):41–51.

Arslan, S. S. and Zeydan, E. (2020). On the distribution modeling of heavy-tailed disk failure lifetime in big data centers. IEEE Transactions on Reliability, 70(2):507–524.

Bucar, T., Nagode, M., and Fajdiga, M. (2004). Reliability approximation using finite Weibull mixture distributions.ˇ Reliability Engineering & System Safety, 84(3):241–251.

Dempster, A. P., Laird, N. M., and Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. Journal of the Royal Statistical Society: Series B (Methodological), 39(1):1–22.

Efron, B. (1975). The efficiency of logistic regression compared to normal discriminant analysis. Journal of the American Statistical Association, 70(352):892–898.

Farcomeni, A. and Nardi, A. (2010). A two-component Weibull mixture to model early and late mortality in a Bayesian framework. Computational Statistics & Data Analysis, 54(2):416–428.

Jewell, N. P. (1982). Mixtures of exponential distributions. The Annals of Statistics, pages 479–484.

Jiang, R. and Murthy, D. (1998). Mixture of Weibull distributions—parametric characterization of failure rate function. Applied Stochastic Models and Data Analysis, 14(1):47–65.

Lai, C., Murthy, D., and Xie, M. (2011). Weibull distributions. Wiley Interdisciplinary Reviews: Computational Statistics, 3(3):282–287.

Lu, G. and Copas, J. B. (2004). Missing at random, likelihood ignorability and model completeness. Annals of Statistics, pages 754–765.

Lyu, Z., Ahfock, D., Thompson, R., and McLachlan, G. J. (2024). Semi-supervised Gaussian mixture modelling with a missing-data mechanism in R. Australian & New Zealand Journal ofStatistics, 66(2):146–162.

McLachlan, G. J., Lee, S. X., and Rathnayake, S. I. (2019). Finite mixture models. Annual Review of Statistics and its Application, 6(1):355–378.

Menon, M. (1963). Estimation of the shape and scale parameters of the Weibull distribution. Technometrics, 5(2):175– 182.

Panteleeva, O. V., Gonzalez, E. G., Huerta, H. V., and Alva, J. A. V. (2015). Identifiability and comparison of estima-´ tion methods on Weibull mixture models. Communications in Statistics-Simulation and Computation, 44(7):1879– 1900.

Rubin, D. B. (1976). Inference and missing data. Biometrika, 63(3):581–592.

Sportisse, A., Schmutz, H., Humbert, O., Bouveyron, C., and Mattei, P.-A. (2023). Are labels informative in semisupervised learning? Estimating and leveraging the missing-data mechanism. In International Conference on Machine Learning, pages 32521–32539. PMLR.

Tsionas, E. G. (2002). Bayesian analysis of finite mixtures of Weibull distributions. Communications in Statistics Theory and Methods, 31(1):37–48.

Woodward, W. A. and Gunst, R. F. (1987). Using mixtures of Weibull distributions to estimate mixing proportions. Computational Statistics & Data Analysis, 5(3):163–176.

Wu, J., Wang, Y.-G., and McLachlan, G. J. (2026). Informative missingness and its implications in semi-supervised learning. The Innovation Informatics, 2:100033.

Zhao, Y., Lee, A. H., Yau, K. K., and McLachlan, G. J. (2011). Assessing the adequacy of Weibull survival models: A simulated envelope approach. Journal ofApplied Statistics, 38(10):2089–2097.