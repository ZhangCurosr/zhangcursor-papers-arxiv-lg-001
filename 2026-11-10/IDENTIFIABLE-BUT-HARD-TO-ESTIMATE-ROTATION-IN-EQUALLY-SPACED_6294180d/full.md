# IDENTIFIABLE BUT HARD TO ESTIMATE: ROTATION IN EQUALLY SPACED OVERCOMPLETE ICA<sup>\*</sup>

A PREPRINT

Wenjun Xiao<sup>1†</sup> Yinghao Fang<sup>2†</sup> Hanxin Chen<sup>3</sup> Sihan Jia<sup>4</sup> Vince D. Calhoun<sup>5</sup> Yuda Bi<sup>4‡</sup>

<sup>1</sup>The George Washington University <sup>2</sup>University of Health and Rehabilitation Sciences <sup>3</sup>University of California, San Diego <sup>4</sup>Georgia State University <sup>5</sup>Georgia Institute of Technology

## ABSTRACT

How many observations are needed to estimate a common rotation in two-dimensional overcomplete independent component analysis? We study equally spaced mixing columns with known source count, common source law, and column norm. For a fixed non-Gaussian fourth-order Gram–Charlier source law, the rotation is identifiable modulo the column spacing. Yet the sample size for fixed relative angular accuracy grows exponentially with the source count. We characterize this sample size, up to absolute constants, by the reciprocal of the squared Hellinger distance between two rotations separated by half the column spacing. An explicit radially weighted phase estimator attains this scale uniformly over the rotation family, using the observation radius to prevent cancellation in the mean of an angular harmonic. We also determine the exact exponential decay rate of singleobservation total variation for the same pair. This gives bounds on the sample size exponent; its exact value remains open. Finally, a noisy binary-source example shows how an exact zero line of the observed characteristic function can have an exponentially small normal derivative.

## 1 Introduction

Consider ten independent sources mixed into two observed coordinates along equally spaced directions. This is an overcomplete independent component analysis (ICA) model: there are more sources than observed coordinates. The source count, common source law, and column norm are known, leaving the common rotation angle as the only unknown parameter. A Gram–Charlier source law of excess kurtosis four makes the angle identifiable modulo the 18<sup>◦</sup> spacing. Suppose the angular error must be strictly below 4.5<sup>◦</sup>, with probability at least 75% at every rotation. Numerical evaluation of the theoretical lower bound (2) gives approximately 60.3 million observations for this guarantee. Every unit projection has excess kurtosis 0.6, so the difficulty persists despite visible non-Gaussianity.

This example separates two questions about ICA. Population identifiability asks whether the observation law determines the mixing parameter, up to the model’s unavoidable ambiguities. Statistical estimation asks how many observations are needed to distinguish parameters at a specified accuracy and confidence. Non-Gaussianity is central to classical ICA (Comon, 1994; Hyvärinen and Oja, 2000), but its presence alone does not quantify the second question. In the opening example, fourth moments reveal non-Gaussianity while carrying no information about the common rotation.

We study this distinction in a family that isolates the effect of the mixing geometry. The observed dimension stays equal to two, the source law is fixed as the source count q grows, and the columns have norm ${ \sqrt { 2 / q } } .$ . The observation covariance is therefore the identity at every rotation. All model ingredients except the common angle are supplied to the estimator. A lower bound in this setting describes difficulty that remains even with that information; it does not rely on having to learn the source law or separate several unknown mixing parameters.

The accuracy scale is also part of the question. Rotating by one column spacing leaves the observation law unchanged, so the target is an angle modulo $\pi / q$ . We require error strictly below one quarter of that spacing, with success probability at least $3 / 4$ at every rotation. This keeps the required resolution fixed relative to the identifiable parameter range as the columns become more closely spaced. The results below concern this uniform, fixed-relative-accuracy target.

## 1.1 Results and mechanism

For a known Gram–Charlier source law, Theorem 1 characterizes the required sample size up to absolute constants for every source count. A test between two rotations separated by half the column spacing gives the lower bound: it is proportional to the reciprocal of their single-observation squared Hellinger distance. A radially weighted phase estimator achieves the same scale uniformly over the full rotation family. Thus, for this model and loss, the difficulty of estimating a continuous parameter is captured by the separation of one explicit pair of observation laws.

The connection between the lower and upper bounds comes from the angular structure of the density. Equal spacing cancels all nonconstant angular frequencies below $2 q .$ For Gram–Charlier sources, the remaining nonconstant terms have frequencies $2 q$ and $4 q ,$ and the phase of the first identifies the rotation. Its amplitude depends on radius and can change sign, so unweighted averaging can cancel contributions that carry the same rotation information. Weights computed from the known model align these contributions. The radial distribution itself is independent of the rotation, but the observed radius indicates how to use the angular component. The second moment of the conditional harmonic amplitude then controls both the estimator’s error and the Hellinger separation.

Theorem 2 quantifies how the observation laws approach one another as the source count grows. For the same halfspacing pair, it gives the exact exponential decay rate of single-observation total variation (TV). Combined with Theorem 1, this yields sample-size bounds that are exponential in $q ,$ despite the fixed observed dimension and the onedimensional unknown parameter. The exact sample-size exponent remains open: total variation depends on the first absolute radial moment of the harmonic amplitude, whereas squared Hellinger distance is comparable to its second moment. The exact TV rate therefore provides exponent bounds, rather than an exact asymptotic formula for the required sample size.

A complementary binary-source example examines an exact zero line of the observed characteristic function. Keeping the equally spaced geometry, we show that the normal derivative along such a line can be exponentially small. This describes the sensitivity of a population feature, and motivates the local analyses in the appendices. The global rotation sample-complexity theorem and these local moment-based analyses have different targets and assumptions; Table 3 records the distinctions.

## 1.2 Related work

ICA identifiability and recovery. Conditions for population identifiability, separability, and uniqueness in linear ICA are studied by Eriksson and Koivunen (2004); Wang and Seigal (2026) treats identifiability in overcomplete models. These results concern what can be determined from the observation law. Recovery guarantees additionally depend on the source and mixing assumptions. Fourier PCA provides finite-sample recovery guarantees under quantitative conditions on tensorized mixing columns and source moments (Goyal et al., 2014). The semidefinite approach studies recovery through the subspace spanned by rank-one column outer products, with a population analysis for random directions (Podosinnikova et al., 2019). For square mixing, polynomial-time recovery with unknown Gaussian noise is possible under source and conditioning assumptions (Arora et al., 2012; Belkin et al., 2013). Our family has a known source law and known relative column directions, leaving only a common rotation to estimate. Its lower bound concerns this restricted target, while its matching upper bound uses the supplied model information.

An earlier information-theoretic lower bound. Anderson et al. (2014, Theorem 4) construct ICA pairs in fixed dimension d from disjoint subsets of $4 k ^ { 2 }$ random directions. The sources follow instance-dependent scaled Poisson laws, and the observations include Gaussian noise with polynomially bounded variance. The laws are exponentially close in $( k / \log k ) ^ { 1 / d }$ . Their theorem gives a testing lower bound. Here the observed dimension is two, the common source law is fixed as q grows, and there is no additive observation noise. We determine an exact TV exponent for the equally spaced family and construct an estimator matching its Hellinger scale. These are different geometric constructions and asymptotic parameters, rather than directly comparable bounds for the same model.

Moment information and statistical separation. In high-noise multireference alignment, informative moment orders govern sample complexity (Perry et al., 2019; Bandeira et al., 2019); estimation under group actions relates such rates to invariant polynomials (Bandeira et al., 2023). The connection here is the loss of low-order information through symmetry. Our varying parameter is the source count at fixed observed covariance, rather than the noise level in an alignment model. Related distinctions between identifiability and estimation rates arise in finite mixtures: strong and weak identifiability control how distributional error translates into parameter error (Ho and Nguyen, 2016b;a), and local minimax rates depend on overfitting (Heinrich and Kahn, 2018). In the present ICA family, the source law remains fixed and the changing mixing geometry makes separated rotations difficult to distinguish.

Organization. Sections 2–3 establish the rotation bounds and explain the roles of radial weighting and symmetry. Section 4 studies sensitivity at a characteristic-function zero. Section 5 discusses the unresolved sample-size exponent, departures from equal spacing, and learning the source law. Appendix A contains the rotation proofs, while Appendices B–D develop the complementary local estimation analyses.

## 2 Rotation identifiability and sample complexity

Let $q \geq 2$ be known. Write $\boldsymbol { u } ( \theta ) = ( \cos \theta , \sin \theta ) ^ { T }$ and $u ^ { \perp } ( \theta ) = ( - \sin \theta , \cos \theta ) ^ { T }$ . We observe n independent draws of $\bar { X _ { \mathrm { ~ } } } = A _ { \theta } S$ , where

$$
A _ { \theta } = \sqrt { 2 / q \left[ u ( \theta ) , u ( \theta + \pi / q ) , \dots , u ( \theta + ( q - 1 ) \pi / q ) \right] }
$$

and $\boldsymbol { S } = ( s _ { 0 } , \ldots , s _ { q - 1 } ) ^ { T }$ has independent coordinates with the same symmetric, unit-variance law $f .$ The source law, spacing, and common column norm are known. The only unknown parameter is the common rotation $\theta .$ The normalization gives $A _ { \theta } A _ { \theta } ^ { T } = I _ { 2 }$ . Write $P _ { \theta }$ for the law of X and $p _ { \theta }$ for its density when it exists.

A rotation by $\pi / q$ permutes the columns up to sign. Because the sources are symmetric and identically distributed, this leaves the observation law unchanged. We therefore estimate the rotation modulo one column spacing, using the distance

$$
d _ { q } ( \theta , \theta ^ { \prime } ) = \operatorname* { m i n } _ { k \in \mathbb { Z } } | \theta - \theta ^ { \prime } - k \pi / q | , \qquad \varrho = \pi / ( 2 q ) .
$$

Thus $\varrho$ is the largest separation on the parameter circle.

For a given source law, let $n ^ { * } ( q )$ be the smallest sample size for which an estimator succeeds with probability at least $3 / 4$ for every $\theta .$ Success means $d _ { q } ( \hat { \theta } , \theta ) < \pi / { ( 4 q ) }$ . The target is one quarter of the column spacing. Equivalently, success requires circular error strictly below $\pi / 2$ in the phase coordinate 2qθ (mod 2π). This fixed phase accuracy makes the statistical target comparable across source counts, even as the columns become more closely spaced. Set $n ^ { * } ( q ) = \infty$ if the guarantee is impossible. The model is overcomplete for $q > 2$ and includes the square case $q = 2$

## 2.1 Which rotations have different observation laws?

Covariance alone cannot distinguish the rotations: it is $I _ { 2 }$ at every angle. With standard Gaussian sources, even the full observation law is the same, $\mathcal { N } ( 0 , I _ { 2 } )$ . To obtain an identifiable family, consider the Gram–Charlier densities

$$
\phi _ { a } ( s ) = \phi ( s ) \bigl [ 1 + a ( s ^ { 4 } - 6 s ^ { 2 } + 3 ) / 2 4 \bigr ] , \qquad 0 < a \leq 4 ,
$$

where $\phi$ is the standard normal density and a is the excess kurtosis. The polynomial $s ^ { 4 } - 6 s ^ { 2 } + 3$ has minimum $_ { - 6 }$ and mean zero under the standard Gaussian law, so $0 < a \le 4$ gives nonnegative densities that integrate to one.

The density relative to the standard bivariate Gaussian is a polynomial of degree at most $4 q$ . Rotation by $\pi / q$ restricts its angular frequencies to multiples of $2 q .$ , and reflection symmetry leaves cosine terms. Thus only $2 q$ and 4q can be nonconstant angular frequencies. Rotating the mixing columns shifts their phases; we track the first harmonic to identify the rotation. In polar coordinates $x = r u ( \psi )$ ,

$$
p _ { \theta } ( r u ( \psi ) ) = \phi _ { \mathrm { N , 2 } } ( r u ( 0 ) ) \left[ A _ { 0 } ( r ) + 2 A _ { 1 } ( r ) \cos t _ { \theta } + 2 A _ { 2 } ( r ) \cos ( 2 t _ { \theta } ) \right] , \qquad t _ { \theta } = 2 q ( \psi - \theta ) .\tag{1}
$$

Here $\phi _ { \mathrm { N } , 2 }$ is the standard bivariate Gaussian density and the $A _ { j }$ are real polynomials. In the bracket, $A _ { 0 } ( r )$ is the angular average, while $2 A _ { 1 } ( r )$ and $2 A _ { 2 } ( r )$ are the coefficients of the two cosine terms.

Lemma 2 shows that $A _ { 1 }$ is nonzero. At any radius with $A _ { 1 } ( r ) ~ \neq ~ 0 .$ , the coefficient of $e ^ { 2 i q \psi }$ in the bracket is $A _ { 1 } ( r ) e ^ { - 2 i q \theta }$ . Equality of observation laws implies equality of the continuous densities and hence of these coefficients. Therefore $P _ { \theta } = P _ { \theta ^ { \prime } }$ ′ if and only if $\theta = \theta ^ { \prime }$ modulo $\pi / q$ . The phase of this coefficient thus identifies the common rotation and provides the basis for the estimator below.

## 2.2 The sample size for uniform rotation estimation

We compare two rotations separated by half the column spacing. Their single-observation squared Hellinger distance is

$$
H _ { q } ^ { 2 } = \int _ { \mathbb { R } ^ { 2 } } \left( \sqrt { p _ { 0 } ( x ) } - \sqrt { p _ { \varrho } ( x ) } \right) ^ { 2 } d x .
$$

This convention has no factor of $1 / 2 .$ . Two-point testing is a standard route to estimation lower bounds (Yu, 1997; Tsybakov, 2009); Hellinger moduli also connect testing and estimation in other models (Donoho and Liu, 1991). Here the half-spacing distance governs estimation over the full parameter circle.

Theorem 1 (Uniform rotation estimation). Let $q \geq 2$ and let the common source density be $\phi _ { a }$ for a known $a \in ( 0 , 4 ] .$ Then

$$
\frac { 1 } { 4 H _ { q } ^ { 2 } } \ \leq \ n ^ { * } ( q ) \ \leq \ \left\lceil \frac { 6 4 } { H _ { q } ^ { 2 } } \right\rceil .\tag{2}
$$

Theorem 1 relates estimation over the full parameter circle to testing two half-spacing rotations. It determines $n ^ { * } ( q )$ up to universal multiplicative constants, whose ratio is 256 apart from integer rounding.

The estimator weights the observed harmonic by its conditional amplitude. Define

$$
B ( r ) = \frac { 2 A _ { 1 } ( r ) } { A _ { 0 } ( r ) } , \qquad M _ { q } = \mathbb { E } [ B ( R ) ^ { 2 } ] , \qquad \hat { \theta } = \frac { 1 } { 2 q } \arg \left[ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } B ( R _ { i } ) e ^ { 2 i q \psi _ { i } } \right] \quad ( \mathrm { m o d } \pi / q ) .
$$

Here $R \ : = \ : \| X \| .$ , and $( R _ { i } , \psi _ { i } )$ are the polar coordinates of observation $X _ { i }$ . Values at $R _ { i } ~ = ~ 0$ or a zero average are arbitrary. Weighting by $B ( R )$ aligns the conditional harmonic means across radii and prevents sign cancellation in their average. The amplitude second moment $M _ { q }$ controls both the estimator’s concentration and the Hellinger separation.

Proof. Radial weights. The radial law is common to all rotations. Lemma 2 gives $A _ { 2 } ( r ) > 0$ for $r > 0 .$ . At $t _ { \theta } = \pi / 2 ,$ nonnegativity in (1) then gives $A _ { 0 } ( r ) \geq 2 A _ { 2 } ( r ) > 0$ . Let $R = \| X \|$ . For $r > 0$ , set $B ( r ) = 2 A _ { 1 } ( r ) / A _ { 0 } ( r )$ and $C ( r ) \stackrel { - } { = } 2 A _ { 2 } \stackrel { \cdot } { ( r ) } / A _ { 0 } ( r )$ . The conditional density of $\psi \in [ 0 , 2 \pi )$ given $R = r$ is

$$
( 1 + B ( r ) \cos t _ { \theta } + C ( r ) \cos ( 2 t _ { \theta } ) ) / ( 2 \pi ) .
$$

We have $0 \leq C \leq 1$ , and evaluating this density at $t _ { \theta } = 0 , \pi \operatorname { y i e l d s } \ | B | \leq 1 + C \leq 2 .$

The radial law has positive density on $( 0 , \infty )$ and $A _ { 1 }$ is a nonzero polynomial. Thus $M _ { q } = \mathbb { E } [ B ( R ) ^ { 2 } ] > 0 .$

Half-spacing comparison. The half-spacing rotation changes the sign of the $2 q$ harmonic and leaves the 4q harmonic unchanged. Fix r and write $t = 2 q \psi$ . Separate the unchanged part, $U = 1 + C ( r )$ cos(2t), from the sign-changing part, $\overset { \vartriangle } { \boldsymbol { V } } = \boldsymbol { B } ( \boldsymbol { r } )$ cos t. The two conditional densities are then $( { \dot { U } } + V ) / ( 2 \pi )$ and $( U - \dot { V } ) / ( 2 \pi )$ . Rationalizing the difference of their square roots gives

$$
\frac { V ^ { 2 } } { U } \leq \left( \sqrt { U + V } - \sqrt { U - V } \right) ^ { 2 } \leq \frac { 2 V ^ { 2 } } { U } ,
$$

where the quotients are taken to be zero when $U = 0$ , since then $V = 0$ . The bounds $\cos ^ { 2 } t \le U \le 2$ follow from $0 \leq C \leq 1$ . Integrating with respect to $d \psi / ( 2 \pi )$ and the common radial law yields

$$
M _ { q } / 4 \le H _ { q } ^ { 2 } \le 2 M _ { q } .\tag{3}
$$

Lower bound. The strict success regions around 0 and ϱ are disjoint. Deciding for 0 exactly when the estimate lies in the success region around 0 yields a test with each error probability at most $\bar { 1 / 4 }$ . Thus $\mathrm { T V } ( P _ { 0 } ^ { \otimes n } , P _ { \varrho } ^ { \otimes n } ) \geq 1 / 2$ . Here $\mathrm { T V } ( P , Q ) = \operatorname* { s u p } _ { E } \left| P ( E ) - Q ( E ) \right|$ | is total variation.

The Hellinger convention above gives $\mathrm { T V } ( P , Q ) \leq H ( P , Q )$ . For the product laws,

$$
H ^ { 2 } ( P _ { 0 } ^ { \otimes n } , P _ { \varrho } ^ { \otimes n } ) = 2 [ 1 - ( 1 - H _ { q } ^ { 2 } / 2 ) ^ { n } ] \leq n H _ { q } ^ { 2 } .
$$

Combining these inequalities gives the lower bound in (2).

Upper bound. Use $Z = B ( R ) e ^ { 2 i q \psi }$ and take $\hat { \theta } = \arg ( \bar { Z } ) / ( 2 q )$ modulo $\pi / q$ , where $\bar { Z }$ is the sample mean. Assign an arbitrary value if $\bar { Z } = 0$ . The conditional density gives $\mathbb { E } _ { \theta } Z = ( M _ { q } / 2 ) e ^ { 2 i q \theta }$ and $\mathbb { E } _ { \theta } | Z | ^ { 2 } = M _ { q }$

The sample mean satisfies $\mathbb { E } _ { \theta } | \bar { Z } - \mathbb { E } _ { \theta } Z | ^ { 2 } \le M _ { q } / n$ . Chebyshev’s inequality therefore bounds the probability of $| \bar { Z } - \mathbb { E } _ { \theta } Z | > { \big | } \mathbb { E } _ { \theta } Z { \big | } / { \sqrt { 2 } }$ by $8 / ( n M _ { q } )$ . For $n \ge 3 2 / M _ { q }$ , this probability is at most $1 / 4$ , uniformly over θ. On the complementary event the phase error is at most $\pi / 4 ,$ giving $d _ { q } ( \hat { \theta } , \theta ) \leq \pi / ( 8 q )$ . By $( 3 ) , \ \lceil 6 4 / H _ { q } ^ { 2 } \rceil$ observations suffice. □

At $a = ~ 4$ and $q \ = \ 1 0 .$ , radial and Cartesian calculations give $H _ { 1 0 } ^ { 2 } \simeq 4 . 1 4 5 1 9 \times 1 0 ^ { - 9 }$ . Numerical evaluation of the theoretical lower bound in (2) gives approximately 60.3 million observations. Appendix A.7 compares the two integration methods and reports numerical refinement checks. Figure 1 shows finite-q distances; Section 3.3 studies their exponential rates.

![](images/05b33bb328f4168595d9718b02b11d9ce129f0dd206c5f732a510e9ccfdee423.jpg)

(b) Squared Hellinger distance  
![](images/a53ba48280ff7c835235c80abddf8e87d5b3c51b67e6641a314931adcceb1d11.jpg)  
Figure 1: Single-observation distances between half-spacing rotations. Left: total variation (Theorem 2). Right: squared Hellinger distance (Theorem 1). The highlighted point has $a = 4$ and $q = 1 0$ . Numerical evaluation of the theoretical lower bound gives approximately 60.3 million observations. The requirement is angular error strictly below $4 . 5 ^ { \circ }$ with probability at least $3 / 4$ at every rotation. Both vertical axes are logarithmic.

Computing the estimator. Given $q$ and $^ { a , }$ construct $A _ { 0 }$ and $A _ { 1 }$ from the Hermite representation in Lemma $2 .$ Evaluate $\bar { B ( R _ { i } ) }$ for each observation, form the weighted phase average displayed above, and reduce its argument divided by 2q modulo $\pi / q .$

## 2.3 What the radial weights do

At radius $r ,$ the conditional mean of $e ^ { 2 i q \psi } \mathrm { i s } ( B ( r ) / 2 ) e ^ { 2 i q \theta }$ . Without weights, the coefficient of $e ^ { 2 i q \theta }$ in the mean is $\mathbb { E } [ B ( R ) ] / 2$ , which can be small if the contributions at different radii cancel. A negative conditional amplitude reverses the mean direction of the harmonic even though the mixing rotation is the same.

After weighting by $B ( R )$ , the mean is $( \mathbb { E } [ B ( R ) ^ { 2 } ] / 2 ) e ^ { 2 i q \theta }$ and the second absolute moment is $\mathbb { E } [ B ( R ) ^ { 2 } ]$ . The weights correct the sign and emphasize radii with stronger harmonics. Radius alone carries no rotation information, because its law is common to all rotations.

Figure 2 illustrates the effect of weighting at $a = 4$ and $q = 3 , 4 , 5$ , where the population unweighted mean is a positive multiple of $e ^ { 2 i q \theta }$ . Rotation equivariance lets us fix $\theta = 0 ,$ . The two estimators use the same observations within each replicate, and the success event is the one defining $n ^ { * } ( q )$ . Near $n H _ { q } ^ { 2 } = 0 . 5$ , the weighted estimator succeeds in 75–78% of trials across the three settings. The corresponding range for the unweighted estimator is 48–65%.

## 3 Why rotation is hard to estimate

## 3.1 What symmetry cancels

Cumulant tensors of order below $2 q$ are rotation invariant whenever they exist. Odd orders vanish by symmetry. At even order k, the sum $\sum _ { j } ( a _ { j } ^ { T } t ) ^ { k }$ contains only angular frequencies that are multiples of $2 q$ as a function of $\theta ,$ where $a _ { j }$ are the columns of $A _ { \theta }$ . Below the first permitted frequency, only the constant angular term can remain.

For $q \geq 3$ and unit $t ,$ the fourth cumulant of $t ^ { T } X$ $3 a / ( 2 q )$ under $\phi _ { a }$ , giving 0.6 in the opening example. The zero-lag fourth cumulant used in tensor-based ICA (Cardoso, 1991; De Lathauwer et al., 2007) therefore contains no

![](images/cac240cba74ca6b39eec8ddc4f330ce3b23486320461de8788a229904fcd0c05.jpg)  
Figure 2: Success of weighted and unweighted phase estimation at $a = 4 .$ . The horizontal axis shows $n H _ { q } ^ { 2 }$ on a logarithmic scale. A uniform random guess succeeds with probability $1 / 2 ,$ because the success region covers half the parameter circle. The estimation target is success probability $3 / 4$ . The error bars are pointwise 95% Wilson intervals. Section A.9 gives the simulation protocol.

rotation information here. Methods using temporal diversity, such as FOBIUM (Ferréol et al., 2005), exploit additional structure absent from this independent-sample model.

Under $\phi _ { a } .$ , cumulants above order two are nonzero only at multiples of four. The first cumulant tensor that depends on the rotation is therefore of order 2q for even q and 2q + 2 for odd q, as shown in Appendix A. The observation density has lowest nonconstant angular frequency 2q in both cases.

## 3.2 The exact total variation exponent

In high-noise multireference alignment, sample complexity depends on informative moment orders (Perry et al., 2019; Bandeira et al., 2019), which can change with the shift distribution (Abbe et al., 2019). The group-action framework connects these rates to separating invariant degrees (Bandeira et al., 2023). In our model, increasing the source count raises the informative angular frequency to $2 q$ and reduces each column norm to $\sqrt { 2 / q }$ . The next theorem quantifies the resulting separation of the full observation laws for the pair used in Theorem 1.

Theorem 2 (Exact total variation exponent). Fix $a \ \in \ ( 0 , 4 ]$ in the common rotation model with Gram–Charlier sources. With $\varrho = \pi / ( 2 q )$ and the constant $\varkappa _ { a } > 0$ defined by the one-dimensional maximization in (10),

$$
\operatorname* { l i m } _ { q \to \infty } - \frac { 1 } { q } \log \mathrm { T V } ( P _ { 0 } , P _ { \varrho } ) = \varkappa _ { a } .
$$

For $a = 4 , \varkappa _ { 4 } \simeq 1 . 5 2 8 8$ . The general expression is $\varkappa _ { a } = - \operatorname* { m a x } _ { v > 0 } e _ { a } ( v )$ . The function $e _ { a } ( \boldsymbol { v } )$ describes the exponential rate of the surviving angular harmonic’s amplitude at frequency radius $v { \sqrt { q / 2 } }$ , after its oscillatory factor has been separated. Equation (10) gives this rate explicitly by factoring the characteristic function on a circle in frequency space. Maximizing over the radius identifies the slowest single-observation TV decay.

A real-frequency evaluation gives a lower bound on total variation. The oscillatory factor can vanish at individual radii. The proof avoids these zeros by choosing a nearby radius where the amplitude has the same exponential rate. For the upper bound, Plancherel’s identity controls the density difference on a ball through an integrated squared difference in frequency space. A tail estimate handles the region outside the ball. These two bounds have the same exponential rate. Appendix A, Proposition 2, gives the bounds for finite source counts and the limiting argument.

## 3.3 What this gives for the sample size exponent

Integrating the absolute difference between the two conditional angle densities gives $\mathrm { T V } ( P _ { 0 } , P _ { \varrho } ) = ( 2 / \pi ) \mathbb { E } | B ( R ) |$ The squared Hellinger distance is comparable to $\mathbb { E } [ B ( R ) ^ { 2 } ]$ by (3). Thus the exact TV rate describes a first absolute moment, while the sample size depends on the rate of the second moment.

Using $\mathrm { T V } ^ { 2 } \leq H _ { q } ^ { 2 } \leq 2 \mathrm { T V }$ in Theorems 1 and 2 gives a lower exponent of $\varkappa _ { a }$ and an upper exponent of $2 \varkappa _ { a }$

The finite-q TV bounds in Appendix A give log $n ^ { * } ( q ) \geq \varkappa _ { a } q - O ( \log q )$ . For $a = 4 .$ , numerical evaluation of a proved complex-frequency bound improves the upper exponent from $2 \varkappa _ { 4 } \simeq 3 . 0 5 7 7$ to approximately 2.3689. Appendix A.7 gives the auxiliary estimator, the exact bound at fixed parameters, and the parameter optimization. The exact limiting sample exponent remains open; the conjecture is stated there.

## 3.4 A total variation bound for other symmetric source laws

The symmetry argument extends beyond the Gram–Charlier family, even when the observation density does not have a finite angular expansion. To control all surviving harmonics, we impose a growth condition on the source transform. Let the common symmetric unit-variance source law $f$ satisfy $\mathbb { E } e ^ { r | s | } < \infty$ for every $r > 0$ . Define its growth relative to the Gaussian transform by

$$
\mu _ { f } = \operatorname* { s u p } _ { z \in \mathbb { C } \setminus \{ 0 \} } \frac { \log \left| e ^ { - z ^ { 2 } / 2 } \mathbb { E } e ^ { z s } \right| } { | z | ^ { 2 } } , \qquad s \sim f .
$$

If $\mu _ { f } < 1 / 2$ and $q \geq 1 / ( 1 - 2 \mu _ { f } )$ , Proposition 1 gives

$$
\mathrm { T V } ( P _ { 0 } , P _ { \varrho } ) \leq 4 \sqrt { \frac { 2 q } { 1 - 2 \mu _ { f } } } ( 2 \mu _ { f } ) ^ { q } .\tag{4}
$$

At the half-spacing rotation, only the odd multiples of angular frequency $2 q$ contribute to the density difference. A complex rotation of the Gaussian-normalized transform bounds the weighted sum of squared Hermite coefficients at these frequencies. This controls the density difference beyond the finitely many cumulants that agree, complementing finite-order moment matching arguments for Gaussian mixtures (Wu and Yang, 2020).

The condition on $\mu _ { f }$ controls source tails and characteristic-function decay as well as moment existence. It holds for $\phi _ { a }$ and the Gaussian-smoothed binary law below (Appendix A). For the larger source class, (4) establishes closeness of the observation laws; identifiability and a matching estimation upper bound require separate arguments.

## 4 Sensitivity at a characteristic-function zero

Characteristic-function ICA uses derivatives (Yeredor, 2000; Comon and Rajih, 2006), including in two-sensor overcomplete identification (Taleb, 2001). We examine the lines of zeros produced by source characteristic functions. Each line is normal to the corresponding mixing column. The question is how rapidly the observed characteristic function changes across an exact zero line.

The Gram–Charlier characteristic function has no real zeros. We therefore keep the equally spaced geometry and use binary sources with optional Gaussian smoothing: $s _ { j } = b B _ { j } + \tau Z _ { j }$ . The $B _ { j }$ are equiprobable signs and the $Z _ { j }$ are standard normals, all mutually independent. We take $b > 0$ and $\tau \geq 0$ , with $\begin{array} { r } { b ^ { \top } + \tau ^ { 2 } = \bar { 1 } } \end{array}$ to give unit source variance. When $\tau > 0 .$ , the transform growth parameter satisfies $2 \mu _ { f } = b ^ { 2 } < 1$ , so (4) applies for sufficiently large $q .$ The observed characteristic function (CF) is

$$
\Phi _ { x } ( \omega ) = \mathbb { E } e ^ { i \omega ^ { T } X } = e ^ { - \tau ^ { 2 } \left\| \omega \right\| ^ { 2 } / 2 } \prod _ { j = 0 } ^ { q - 1 } \cos ( b a _ { j } ^ { T } \omega ) .\tag{5}
$$

The first positive zero line associated with $a _ { 0 }$ is $b a _ { 0 } ^ { T } \omega = \pi / 2$ . It lies at distance $\rho _ { 0 } = \pi \sqrt { q } / ( 2 \sqrt { 2 } b )$ from the origin. Because the characteristic function is identically zero along this line, its tangential derivative is zero. Its gradient therefore measures sensitivity in the normal direction.

On the line, differentiation can only contribute through the vanishing cosine factor. The other cosines have magnitude at most one, and the Gaussian factor is largest at the point nearest the origin. Consequently

$$
\| \nabla _ { \omega } \Phi _ { x } ( \omega ) \| \le b \sqrt { 2 / q } \exp \left( - \frac { \pi ^ { 2 } \tau ^ { 2 } q } { 1 6 b ^ { 2 } } \right) .\tag{6}
$$

Bound (6) captures the Gaussian attenuation. The cosine product causes further decay and provides most of the decay exponent at 10 dB, where $b ^ { 2 } / \tau ^ { 2 } = 1 0$ . To separate these contributions, record the normal slope as a function of position along the line. Let

$$
S _ { q } ( \nu ) = \| \nabla _ { \omega } \Phi _ { x } ( \rho _ { 0 } u ( \theta ) + \nu u ^ { \perp } ( \theta ) ) \| , \qquad \gamma _ { \mathrm { b i n } } = - \frac { 1 } { \pi } \int _ { 0 } ^ { \pi } \log \cos \Bigl ( \frac { \pi } { 2 } \cos \varphi \Bigr ) d \varphi .
$$

The calculation in Appendix A.8 shows that at the point nearest the origin,

$$
\operatorname * { l i m } _ { q \to \infty } q ^ { - 1 } \log S _ { q } ( 0 ) = - \gamma _ { \mathrm { b i n } } - \frac { \pi ^ { 2 } \tau ^ { 2 } } { 1 6 b ^ { 2 } } .
$$

The same limit holds for the maximum normal slope on any fixed bounded segment centered at that point. The binary product contributes $\gamma _ { \mathrm { b i n } } \simeq 1 . 5 0 6 1$ to the decay exponent per source, and this contribution remains when $\tau = 0$ . At $\dot { b } ^ { 2 } / \tau ^ { 2 } = 1 0 ( 1 0 { } \mathrm { d B } )$ , the Gaussian contribution is about 0.0617 per source.

An angular search uses a different derivative because changing the candidate angle moves the evaluation point in frequency space. Hold the observation law fixed and vary the candidate angle in $\bar { \rho _ { 0 } } u ( \theta ^ { \prime } ) + \nu u ^ { \perp } ( \theta ^ { \prime } )$ , keeping $\rho _ { 0 }$ and ν fixed. $\mathbf { A } \mathbf { t } \mathbf { \nabla } \theta ^ { \prime } = \theta$ , only the normal component of the frequency displacement contributes, so

$$
| \frac { d } { d \theta ^ { \prime } } \Phi _ { x } ( \rho _ { 0 } u ( \theta ^ { \prime } ) + \nu u ^ { \perp } ( \theta ^ { \prime } ) ) | _ { \theta ^ { \prime } = \theta } | = | \nu | S _ { q } ( \nu ) .
$$

It vanishes at $\nu = 0$ . Figure 3(a) in Appendix B compares the angular derivative at $\nu = 1$ with (6); at this offset, its magnitude equals the normal slope $S _ { q } \bar { ( 1 ) }$ . The proved exponential rate applies at $\nu = 0$ and to the maximum over a fixed bounded segment; rates at individual nonzero offsets may differ.

Appendix B studies local estimation using a known real zero of the target source CF and an isolating angle–offset window, fixed or supplied independently of the estimation sample. With finitely many frequencies, identification additionally requires the population criterion to vanish only at the target. At fixed model size and grid, local variance depends on moment derivatives and the full covariance of sampled $\mathrm { C F }$ values; slopes alone do not establish a minimax rate.

## 5 Discussion

The rotation model separates identifiability from the amount of information available in a sample. The phase of the $2 q$ harmonic identifies the common rotation, while the radial second moment of its conditional amplitude determines the sample size. Radius alone is uninformative about rotation, yet it tells the estimator how to combine angular observations without cancellation. Theorem 1 makes this distinction quantitative through a guarantee uniform over the parameter circle, at fixed accuracy relative to the column spacing. The guarantee uses the known source law and geometry; its matching upper bound does not establish the same performance when either must also be learned. The construction is explicit, but the theorem does not bound the numerical precision or computational cost needed to evaluate the radial weights as q grows.

The unresolved sample-size exponent. The exact total variation exponent leaves a specific statistical question unresolved. Total variation measures the first absolute moment of the conditional harmonic amplitude, whereas squared Hellinger distance is comparable to its second moment. Determining the decay of the former therefore does not determine the sample-size exponent. The complex-frequency bound narrows the available upper bound, but the existence and value of the limiting sample-size exponent remain open. The conjecture in Appendix A.7 asks whether optimizing that construction loses no exponential factor relative to the radial estimator. Resolving it would identify the exponent of a sample size already characterized, for each q, up to absolute constants. The finite-range fits support investigation of this question without establishing an asymptotic limit.

Departures from equal spacing. The role of symmetry also raises questions about how the difficulty changes when the geometry is perturbed.

Departures from equal spacing can make the covariance informative. Let the known offsets be $\alpha _ { j } = j \pi / q + \epsilon _ { j }$ and take mixing columns $\sqrt { 2 / q } u ( \theta + \alpha _ { j } )$ . Their second angular component is $z _ { 2 } = ( 2 / q ) \sum _ { j } e ^ { 2 i \alpha _ { j } }$ , which gives $\mathbb { E } [ ( X _ { 1 } + i X _ { 2 } ) ^ { 2 } ] = e ^ { 2 i \theta } z _ { 2 }$ . When $z _ { 2 } \neq 0$ , the covariance identifies θ modulo π. For the Gram–Charlier family, estimate this expectation by the sample mean of $( X _ { 1 } + i X _ { 2 } ) ^ { 2 }$ . Divide the sample mean by the known $z _ { 2 }$ and halve its phase to estimate θ modulo $\pi .$ . This estimator achieves angular accuracy ε at fixed confidence using $\bar { O ( | z _ { 2 } | ^ { - 2 } \varepsilon ^ { - 2 } ) }$ observations. The statistic’s variance is bounded by $\mathbb { E } \Vert X \Vert ^ { 4 }$ , uniformly bounded under this normalization. Indeed, the covariance Σ has trace two and $\mathbb { E } \| X \| ^ { 4 } = ( \mathrm { t r } \Sigma ) ^ { 2 ^ { \circ } } + 2 \mathrm { t r } ( \Sigma ^ { 2 } ) + 4 a / q$

This calculation identifies a mechanism by which known perturbations can restore information at a low moment order. Its usefulness depends on the size of $z _ { 2 } \colon$ a nonzero value establishes identification through covariance, but the stated covariance-based sample bound deteriorates as that value approaches zero. The calculation does not characterize the transition between covariance-based estimation and the higher angular harmonics near equal spacing.

For a larger model with unknown offsets, the two-point lower bound transfers when the loss restricts to $d _ { q }$ on the exactly spaced subclass, under the same accuracy and success requirements. If θ remains a separate target parameter, the parameterization must distinguish the common rotation from a common shift of all offsets.

The transferred bound is a worst-case statement for a model containing that subclass. It does not say that every perturbed configuration is equally difficult, or supply an estimator for jointly unknown directions. Studying such models requires specifying which geometric features are known and how the target is represented before comparing rates.

Learning the source law. In the equally spaced model, E $\| X \| ^ { 4 } = 8 + 4 a / q \quad$ , so the radial law identifies a even when it is unknown. Whether an estimate of a can be used in the radial weights while preserving the sample size guarantee remains open.

The issue is how error in the estimated source parameter affects the weighted harmonic when its mean is small. Identifying a from radial observations does not by itself control this effect. More generally, the total variation bound for other symmetric source laws establishes closeness of observation laws under its transform-growth condition; identifiability and a matching upper bound still require separate arguments.

Local sensitivity and global estimation. The characteristic-function zero example addresses a complementary local question. An exact zero line identifies a geometric feature of the population transform, while its normal slope can be exponentially small. The local estimation calculations additionally require an isolating search window and, for a finite grid, a unique population minimum. The resulting estimator covariance depends on both moment derivatives and the joint fluctuations of the sampled transform values. A slope calculation alone therefore gives neither a global minimax rate nor a guarantee of locating the correct line. Choosing informative frequencies, accounting for their covariance, and obtaining bounds uniform in growing model size are natural next steps. These questions connect the local calculations to the uniform rotation problem while preserving their different assumptions and limits.

## References

E. Abbe, T. Bendory, W. Leeb, J. M. Pereira, N. Sharon, and A. Singer, “Multireference alignment is easier with an aperiodic translation distribution,” IEEE Trans. Inf. Theory, 65(6):3565–3584, 2019.

S.-I. Amari and J.-F. Cardoso, “Blind source separation—Semiparametric statistical approach,” IEEE Trans. Signal Process., 45(11):2692–2700, 1997.

J. Anderson, M. Belkin, N. Goyal, L. Rademacher, and J. Voss, “The more, the merrier: The blessing of dimensionality for learning large Gaussian mixtures,” in Proc. Conf. Learn. Theory, 2014, pp. 1135–1164.

D. W. K. Andrews, “Examples of L<sup>2</sup>-complete and boundedly-complete distributions,” J. Econometrics, 199(2):213– 220, 2017.

S. Arora, R. Ge, A. Moitra, and S. Sachdeva, “Provable ICA with unknown Gaussian noise, with implications for Gaussian mixtures and autoencoders,” in Adv. Neural Inf. Process. Syst., vol. 25, 2012, pp. 2375–2383.

A. S. Bandeira, J. Niles-Weed, and P. Rigollet, “Optimal rates of estimation for multi-reference alignment,” Math. Stat. Learn., 2(1):25–75, 2019.

A. S. Bandeira, B. Blum-Smith, J. Kileel, J. Niles-Weed, A. Perry, and A. S. Wein, “Estimation under group actions: Recovering orbits from invariants,” Appl. Comput. Harmon. Anal., 66:236–319, 2023.

V. Bargmann, “On a Hilbert space of analytic functions and an associated integral transform. Part I,” Commun. Pure Appl. Math., 14(3):187–214, 1961.

M. Belkin, L. Rademacher, and J. Voss, “Blind signal separation in the presence of Gaussian noise,” in Proc. Conf. Learn. Theory, Proc. Mach. Learn. Res., vol. 30, 2013, pp. 270–287.

P. J. Bickel, C. A. J. Klaassen, Y. Ritov, and J. A. Wellner, Efficient and Adaptive Estimation for Semiparametric Models. Baltimore, MD, USA: Johns Hopkins University Press, 1993.

L. D. Brown, T. T. Cai, and A. DasGupta, “Interval estimation for a binomial proportion,” Statist. Sci., 16(2):101–133, 2001.

J.-F. Cardoso, “Super-symmetric decomposition of the fourth-order cumulant tensor. Blind identification of more sources than sensors,” in Proc. IEEE ICASSP, 1991, pp. 3109–3112.

J.-F. Cardoso, “Dependence, correlation and Gaussianity in independent component analysis,” J. Mach. Learn. Res., 4:1177–1203, 2003.

M. Carrasco and J.-P. Florens, “Generalization of GMM to a continuum of moment conditions,” Econometric Theory, 16(6):797–834, 2000.

A. Chen and P. J. Bickel, “Efficient independent component analysis,” Ann. Statist., 34(6):2825–2855, 2006.

P. Comon, “Independent component analysis, A new concept?,” Signal Process., 36(3):287–314, 1994.

P. Comon and M. Rajih, “Blind identification of under-determined mixtures based on the characteristic function,” Signal Process., 86(9):2271–2281, 2006.

L. De Lathauwer, J. Castaing, and J.-F. Cardoso, “Fourth-order cumulant-based blind identification of underdetermined mixtures,” IEEE Trans. Signal Process., 55(6):2965–2973, 2007.

P. Delsarte, J. M. Goethals, and J. J. Seidel, “Spherical codes and designs,” Geom. Dedicata, 6(3):363–388, 1977.

D. L. Donoho and R. C. Liu, “Geometrizing rates of convergence, II,” Ann. Statist., 19(2):633–667, 1991.

J. Eriksson and V. Koivunen, “Identifiability, separability, and uniqueness of linear ICA models,” IEEE Signal Process. Lett., 11(7):601–604, 2004.

A. Ferréol, L. Albera, and P. Chevalier, “Fourth-order blind identification of underdetermined mixtures of sources (FOBIUM),” IEEE Trans. Signal Process., 53(5):1640–1653, 2005.

A. Feuerverger and P. McDunnough, “On the efficiency of empirical characteristic function procedures,” J. R. Stat. Soc. Ser. B, 43(1):20–27, 1981.

A. Feuerverger and P. McDunnough, “On statistical transform methods and their efficiency,” Canad. J. Statist., 12(4):303–317, 1984.

A. Feuerverger and R. A. Mureika, “The empirical characteristic function and its applications,” Ann. Statist., 5(1):88– 97, 1977.

N. Goyal, S. Vempala, and Y. Xiao, “Fourier PCA and robust tensor decomposition,” in Proc. ACM Symp. Theory Comput., 2014, pp. 584–593.

L. P. Hansen, “Large sample properties of generalized method of moments estimators,” Econometrica, 50(4):1029– 1054, 1982.

P. Heinrich and J. Kahn, “Strong identifiability and optimal minimax rates for finite mixture estimation,” Ann. Statist., 46(6A):2844–2870, 2018.

N. Ho and X. Nguyen, “Convergence rates of parameter estimation for some weakly identifiable finite mixtures,” Ann. Statist., 44(6):2726–2755, 2016a.

N. Ho and X. Nguyen, “On strong identifiability and convergence rates of parameter estimation in finite mixtures,” Electron. J. Stat., 10(1):271–307, 2016b.

Y. Hu and J.-L. Shiu, “A simple test of completeness in a class of nonparametric specification,” Econometric Rev., 41(4):373–399, 2022.

A. Hyvärinen and E. Oja, “Independent component analysis: Algorithms and applications,” Neural Netw., 13(4– 5):411–430, 2000.

L. Mattner, “Completeness of location families, translated moments, and uniqueness of charges,” Probab. Theory Relat. Fields, 92:137–149, 1992.

L. Mattner, “Some incomplete but boundedly complete location families,” Ann. Statist., 21(4):2158–2162, 1993.

W. K. Newey and D. McFadden, “Large sample estimation and hypothesis testing,” in Handbook of Econometrics, vol. 4, R. F. Engle and D. L. McFadden, Eds. Amsterdam, The Netherlands: Elsevier, 1994, pp. 2111–2245.

A. Perry, J. Weed, A. S. Bandeira, P. Rigollet, and A. Singer, “The sample complexity of multireference alignment,” SIAM J. Math. Data Sci., 1(3):497–517, 2019.

A. Podosinnikova, A. Perry, A. S. Wein, F. Bach, A. d’Aspremont, and D. Sontag, “Overcomplete independent component analysis via SDP,” in Proc. AISTATS, 2019, pp. 2583–2592.

J. Rapp, R. M. A. Dawson, and V. K. Goyal, “Estimation from quantized Gaussian measurements: When and how to use dither,” IEEE Trans. Signal Process., 67(13):3424–3438, 2019.

L. Schuchman, “Dither signals and their effect on quantization noise,” IEEE Trans. Commun. Technol., 12(4):162–165, 1964.

A. Taleb, “An algorithm for the blind identification of N independent signals with 2 sensors,” in Proc. Sixth Int. Symp. Signal Process. Appl. (ISSPA), vol. 1, 2001, pp. 5–8.

A. B. Tsybakov, Introduction to Nonparametric Estimation. New York, NY, USA: Springer, 2009.

A. W. van der Vaart, “Semiparametric models: an evaluation,” Statistica Neerlandica, 49:111–125, 1995.

A. W. van der Vaart, Asymptotic Statistics, Cambridge Series in Statistical and Probabilistic Mathematics, vol. 3. Cambridge, UK: Cambridge University Press, 1998.

K. Wang and A. Seigal, “Identifiability of overcomplete independent component analysis,” Bernoulli, 32(3):1713– 1736, 2026.

E. B. Wilson, “Probable inference, the law of succession, and statistical inference,” J. Amer. Statist. Assoc., 22(158):209–212, 1927.

Y. Wu and P. Yang, “Optimal estimation of Gaussian mixtures via denoised method of moments,” Ann. Statist., 48(4):1981–2007, 2020.

A. Yeredor, “Blind source separation via the second characteristic function,” Signal Process., 80(5):897–902, 2000.

B. Yu, “Assouad, Fano, and Le Cam,” in Festschrift for Lucien Le Cam, D. Pollard, E. Torgersen, and G. L. Yang, Eds. New York, NY, USA: Springer, 1997, pp. 423–435.

## A Proofs for the rotation model

The proofs begin with the angular density used in Theorem 1. The Hermite representation converts transform differences into density differences, while the angular decomposition isolates the terms that distinguish the two rotations. Together they give the general total-variation bound. For Gram–Charlier sources, factoring the characteristic function on a circle gives the exact TV exponent in Theorem 2 and the sample-size exponent bounds in Section 3.3. Section A.8 treats the normal derivative in the binary-source example of Section 4.

## A.1 Notation and transform representation

Let $q \geq 2$ . The independent sources have the same symmetric, unit-variance law $f$ with $\mathbb { E } e ^ { r | s | } < \infty$ for all $r > 0$ Write the mixing matrix as

$$
A _ { \theta } = r _ { q } [ u ( \alpha _ { 0 } ) , \ldots , u ( \alpha _ { q - 1 } ) ] , \qquad r _ { q } = \sqrt { 2 / q } ,
$$

with $\alpha _ { j } = \theta { + } j \pi / q$ and columns $a _ { j } = r _ { q } u ( \alpha _ { j } )$ . Since $\begin{array} { r } { \sum _ { j } u ( \alpha _ { j } ) u ( \alpha _ { j } ) ^ { T } = ( q / 2 ) I _ { 2 } } \end{array}$ for $q \ge 2 , A _ { \theta } A _ { \theta } ^ { T } = I _ { 2 }$ . A rotation by $\pi / q$ permutes the columns up to sign, so the distance between configurations is $d _ { q } ( \theta , \theta ^ { \prime } ) = \mathrm { d i s t } ( \theta - \theta ^ { \prime } , ( \pi / q ) \mathbb { Z } )$ We write $\varrho = \pi / ( 2 q )$ for the largest such distance. The comparison below uses $A _ { 0 }$ and $A _ { \varrho } .$

Let $P _ { \theta }$ denote the law of $X = A _ { \theta } S$ and $p _ { \theta }$ its density when it exists. Write $\phi _ { \mathrm { N } , d }$ for the standard normal density on $\mathbb { R } ^ { d }$ and cum (s) for the kth cumulant of s. The characteristic function of X is denoted by $\Phi _ { x } ;$ lowercase x also denotes a realized value or a function argument.

For a symmetric source let $m _ { f } ( u ) = e ^ { - u ^ { 2 } / 2 } \mathbb { E } e ^ { u s }$ , an even entire function with $m _ { f } ( u ) = 1 + O ( u ^ { 4 } )$ . For $\phi _ { a } .$ $m _ { f } ( u ) = 1 + a u ^ { 4 } / 2 4$ . Since $A _ { \theta } A _ { \theta } ^ { T } = I _ { 2 }$ , independence gives

$$
\mathbb { E } e ^ { t ^ { T } X } = e ^ { t ^ { T } t / 2 } P _ { A } ( t ) , \qquad P _ { A } ( t ) = \prod _ { j = 0 } ^ { q - 1 } m _ { f } ( a _ { j } ^ { T } t ) .\tag{7}
$$

The factor $P _ { A }$ carries the dependence on the source law and mixing matrix. Under the finiteness condition below, the Hermite transform of $P _ { A }$ gives the density ratio relative to the standard Gaussian density. For a multi-index $\alpha =$ $\left( \alpha _ { 1 } , \alpha _ { 2 } \right)$ , write $t ^ { \alpha } = t _ { 1 } ^ { \alpha _ { 1 } } t _ { 2 } ^ { \alpha _ { 2 } }$ and $\alpha ! = \alpha _ { 1 } ! \alpha _ { 2 } !$ . Let H map $t ^ { \alpha }$ to $\mathrm { H e } _ { \alpha } ( x ) = \mathrm { H e } _ { \alpha _ { 1 } } ( x _ { 1 } ) \mathrm { H e } _ { \alpha _ { 2 } } ( x _ { 2 } )$ , where $\mathrm { H e } _ { k }$ denotes the probabilists’ Hermite polynomial. For $\begin{array} { r } { P = \sum _ { \alpha } p _ { \alpha } t ^ { \alpha } } \end{array}$ , define the squared Fischer norm by $\begin{array} { r } { \| P \| _ { \mathrm { F } } ^ { 2 } = \sum _ { \alpha } \alpha ! | p _ { \alpha } | ^ { 2 } } \end{array}$ Hermite expansions also describe local non-Gaussian geometry in ICA (Cardoso, 2003). The next lemma identifies the squared Fischer norm with the Gaussian $L ^ { 2 }$ norm of the Hermite series and with an average over complex Gaussian arguments.

Lemma 1 (Hermite representation). $I f \| P _ { A } \| _ { \mathrm { F } } < \infty$ , then X has density $\phi _ { \mathrm { N } , 2 } r _ { A }$ , where $r _ { A } = \mathcal { H } P _ { A } \in L ^ { 2 } ( \phi _ { \mathrm { N } , 2 } )$ . For two matrices satisfying this condition, $\begin{array} { r } { \int ( r _ { A } - r _ { A ^ { \prime } } ) ^ { 2 } \phi _ { \mathrm { N } , 2 } = \| P _ { A } - P _ { A ^ { \prime } } \| _ { \mathrm { F } } ^ { 2 } } \end{array}$ . Let W have independent circular complex Gaussian entries with $\mathbb { E } | W _ { i } | ^ { 2 } = 1$ . Then $\| P \| _ { \mathrm { F } } ^ { 2 } = \mathbb { E } | P ( W ) | ^ { 2 } f o r$ r every entire $P ,$ , with both sides allowed to be infinite.

Proof. Let $p _ { \alpha }$ be the Taylor coefficients of $P _ { A }$ . The generating function of the Hermite polynomials, $e ^ { t ^ { T } x - t ^ { T } t / 2 } =$ $\begin{array} { r } { \sum _ { \alpha } \bar { \mathrm { H e } } _ { \alpha } ( x ) \bar { t } ^ { \alpha } / \alpha ! } \end{array}$ , and dominated convergence give $\mathbf { \bar { E H e } } _ { \alpha } ( \bar { X } ) \mathbf { \Psi } = \alpha ! p _ { \alpha }$ . Since $\mathrm { H e } _ { k } ( y ) \stackrel { \cdot } { = } \mathbb { E } ( y + i Z ) ^ { k }$ for $Z \sim$ $\overline { { \mathcal { N } } } ( 0 , 1 )$ , the series is dominated by

$$
\sum _ { \alpha } | \mathrm { H e } _ { \alpha } ( x ) | | t ^ { \alpha } | / \alpha ! \leq \prod _ { i } \mathbb { E } e ^ { | t _ { i } | ( | x _ { i } | + | Z _ { i } | ) } ,
$$

which is integrable because $\mathbb { E } e ^ { c \| X \| } < \infty$ for all $c > 0$

We next construct the candidate density from these Hermite moments. Since $\begin{array} { r } { \sum _ { \alpha } \alpha ! | p _ { \alpha } | ^ { 2 } < \infty . } \end{array}$ , the series $\tilde { r } \ =$ $\textstyle \sum _ { \alpha } p _ { \alpha } \mathrm { H e } _ { \alpha }$ converges in $L ^ { 2 } ( \phi _ { \mathrm { N } , 2 } )$ and $\begin{array} { r } { \int \mathrm { H e } _ { \alpha } \tilde { r } \phi _ { \mathrm { N } , 2 } = \alpha ! p _ { \alpha } } \end{array}$ . Cauchy–Schwarz gives $\begin{array} { r } { \int e ^ { c \| x \| } | \tilde { r } ( x ) | \phi _ { \mathrm { N } , 2 } ( x ) d x \ \leq } \end{array}$ $\| \tilde { r } \| _ { L ^ { 2 } ( \phi _ { \mathrm { N } , 2 } ) } ( \mathbb { E } e ^ { 2 c \| G \| } ) ^ { 1 / 2 } < \infty$ for $G \sim \mathcal { N } ( 0 , I _ { 2 } )$ . Thus the finite signed measure $\nu = P _ { \theta } - \tilde { r } \phi _ { \mathrm { N , 2 } }$ has all exponential moments and vanishing polynomial moments. Its Laplace transform is entire and all its Taylor coefficients vanish, so $\nu = 0$ . Hence the candidate density represents the observation law.

Orthogonality of the $\mathrm { H e } _ { \alpha }$ gives the $L ^ { 2 }$ identity. For the complex Gaussian representation, E[W<sup>α</sup>W<sup>¯</sup> <sup>α</sup> ] = α! $\delta _ { \alpha \alpha ^ { \prime } }$ Integrating over polydiscs and then applying monotone convergence gives $\begin{array} { r } { \mathbb E | \dot { \boldsymbol { P } } ( { \boldsymbol { W } } ) | ^ { 2 } = \sum _ { \alpha } ^ { * } \alpha ! | { \boldsymbol { p } } _ { \alpha } | ^ { 2 } } \end{array}$ , the Fischer norm representation of Bargmann (1961). □

## A.2 The Gram–Charlier angular density

For the Gram–Charlier source, the transform factor is a finite polynomial. Its degree and symmetries restrict the angular terms in the density. Its highest-degree part supplies a nonzero coefficient that makes the density depend on the rotation.

Lemma 2 (Angular density). For $q \ \geq \ 2$ and $0 < a \le 4 ,$ the observation density has the form (1). Let $K =$ $( a / ( 9 6 q ^ { 2 } ) ) ^ { q }$ . Then $A _ { 2 } ( r ) \stackrel { \cdot } { = } K r ^ { 4 q }$ and the coefficient $o f r ^ { 4 q }$ in $A _ { 1 } ( r ) i s - 4 ( - 1 ) ^ { q } K$

Proof. Here $\begin{array} { r } { P _ { A _ { \theta } } ( t ) = \prod _ { j = 0 } ^ { q - 1 } [ 1 + a ( a _ { j } ^ { T } t ) ^ { 4 } / 2 4 ] } \end{array}$ , a polynomial of degree 4q. The Hermite transform in Lemma 1 is $\exp ( - \Delta _ { t } / 2 )$ on polynomials, where $\Delta _ { t }$ is the Laplacian. It commutes with rotations and reflections and preserves the highest-degree homogeneous part. $\mathbf { A } \mathbf { t } \theta = 0$ , rotation by $\pi / q$ and reflection about the first coordinate axis permute the unsigned columns. The rotation symmetry restricts angular frequencies to multiples of $2 q$ . The degree bound excludes frequencies above $4 q ,$ , and reflection symmetry excludes sine terms. Thus only the frequencies $0 , 2 q$ , and $4 q$ remain. Rotating by θ gives the phase in (1).

To compute the highest-degree coefficients, put $z = t _ { 1 } + i t _ { 2 }$ for real t. Then $a _ { i } ^ { T } t = ( 2 q ) ^ { - 1 / 2 } e ^ { - i j \pi / q } ( z + e ^ { 2 i j \pi / q } \bar { z } )$ at $\theta = 0$ . The product of the fourth powers of the phases equals one, and the product over roots of unity gives the top homogeneous term $K [ z ^ { q } - ( - 1 ) ^ { \dot { q } } \bar { z } ^ { q } ] ^ { 4 }$ . The coefficients of $e ^ { 4 i q \psi }$ and $e ^ { 2 i q \psi }$ in this term are respectively $\check { K } r ^ { 4 q }$ and $- \bar { 4 } ( - 1 ) ^ { q } \bar { K } r ^ { 4 q }$ . Lower degrees cannot contribute to the 4q harmonic or change the coefficient at degree 4q of the $2 q$ harmonic. □

## A.3 Angular harmonics and factorization on a circle

Decomposing the transform into angular frequencies makes the effect of rotation explicit. We will bound the components that remain after subtracting the two configurations. Write $\zeta = t _ { 1 } + i t _ { 2 }$ and $\zeta ^ { \prime } = t _ { 1 } - i t _ { 2 }$ , and expand an entire function as $\begin{array} { r } { P = \sum _ { k , l } \tilde { p } _ { k l } \zeta ^ { k } \zeta ^ { \prime l } } \end{array}$ . Define its angular component at frequency ℓ by $\begin{array} { r } { \Pi _ { \ell } \hat { P } = \hat { \sum } _ { k - l = \ell } \hat { \tilde { p } _ { k l } } \zeta ^ { k } \zeta ^ { \prime l } } \end{array}$ For complex $t , \zeta$ and $\zeta ^ { \prime }$ are separate coordinates and need not be complex conjugates. For real $t = \| t \| u ( \psi )$ $\zeta ^ { k } \zeta ^ { \prime l } = \hat { \| } t \| ^ { k + l } e ^ { i ( k - l ) \psi }$

To bound the high angular frequencies, we rescale the two complex coordinates in opposite directions. For $\vartheta \in \mathbb { R }$ let $D _ { \vartheta }$ act on $\mathbb { C } ^ { 2 } \mathfrak { b } \overline { { \mathbf { y } } } ( \zeta , \breve { \zeta ^ { \prime } } ) \mapsto ( e ^ { \hat { \vartheta } } \zeta , e ^ { - \vartheta } \zeta ^ { \prime } )$ , a rotation by the imaginary angle −iϑ. Part (c) below uses this rescaling to bound the sum of squared Fischer norms above a chosen angular frequency.

Lemma 3 (Angular harmonics). (a) The monomials $\zeta ^ { k } \zeta ^ { \prime l }$ are orthogonal for the Fischer inner product, with $\| \zeta ^ { k } \zeta ^ { \prime l } \| _ { \mathrm { F } } ^ { 2 } = \mathrm { \dot { 2 } } ^ { k + \breve { l } } k ! l !$ . Hence $\begin{array} { r } { \| \dot { P } \| _ { \mathrm { F } } ^ { 2 } = \sum _ { \ell } \| \Pi _ { \ell } P \| _ { \mathrm { F } } ^ { 2 } } \end{array}$

$( b ) \ f r m _ { f }$ is even, $\Pi _ { \ell } P _ { A _ { \theta } } = 0$ unless 2q divides ℓ, and $\Pi _ { \ell } P _ { A _ { \theta + \psi } } = e ^ { - i \ell \psi } \Pi _ { \ell } P _ { A _ { \theta } }$ . In particular $P _ { A _ { 0 } } - P _ { A _ { \varrho } } =$ $2 \textstyle \sum _ { m \ o d \mathrm { d } } \Pi _ { 2 q m } P _ { A _ { 0 } }$

(c) For $\vartheta \geq 0$ and every integer $\begin{array} { r } { \ell _ { 0 } \geq 0 , \sum _ { \ell > \ell _ { 0 } } \| \Pi _ { \ell } P \| _ { \mathrm { F } } ^ { 2 } \leq e ^ { - 2 \ell _ { 0 } \vartheta } \mathbb { E } | P ( D _ { \vartheta } W ) | ^ { 2 } } \end{array}$ , and symmetrically for $\ell \leq - \ell _ { 0 }$ with $D _ { - \vartheta }$

Proof. (a) The variables $\zeta = W _ { 1 } + i W _ { 2 }$ and $\zeta ^ { \prime } = W _ { 1 } - i W _ { 2 }$ are independent circular complex Gaussians of variance 2. Thus $\mathbb { E } [ \zeta ^ { k } \zeta ^ { \prime l } \overline { { \zeta ^ { k ^ { \prime } } \zeta ^ { \prime l ^ { \prime } } } } ] = 2 ^ { k + l } k ! l ! \delta _ { k k ^ { \prime } } \delta _ { l l ^ { \prime } }$

(b) Under $R _ { \psi } ,$ , the coordinates transform as $( \zeta , \zeta ^ { \prime } ) \mapsto ( e ^ { i \psi } \zeta , e ^ { - i \psi } \zeta ^ { \prime } )$ . Consequently, $\Pi _ { \ell } ( P \circ R _ { - \psi } ) = e ^ { - i \ell \psi } \Pi _ { \ell } P$ Since ${ P _ { A _ { \theta + \psi } } } ~ = ~ P _ { A _ { \theta } } \circ R _ { - \psi }$ , the stated phase relation follows. Moreover $\dot { R } _ { \pi / q } a \dot { } { } _ j = \dot { a } _ { j + 1 }$ for $j ~ < ~ q - 1$ and

$R _ { \pi / q } a _ { q - 1 } = - a _ { 0 }$ , so $P _ { A _ { \theta } }$ is invariant under $R _ { \pi / q }$ when $m _ { f }$ is even, and $\Pi _ { \ell } P _ { A _ { \theta } } = 0$ unless $e ^ { - i \ell \pi / q } = 1$ . For $\psi = \varrho$ and $\ell = 2 q m , e ^ { - i \ell \varrho } = ( - 1 ) ^ { m }$

$$
\begin{array} { r } { ( \mathrm { c } ) \mathrm { B y } ( \mathrm { a } ) , \mathbb { E } | P ( D _ { \vartheta } W ) | ^ { 2 } = \sum _ { \ell } e ^ { 2 \ell \vartheta } \| \Pi _ { \ell } P \| _ { \mathrm { F } } ^ { 2 } \geq e ^ { 2 \ell _ { 0 } \vartheta } \sum _ { \ell \geq \ell _ { 0 } } \| \Pi _ { \ell } P \| _ { \mathrm { F } } ^ { 2 } . } \end{array}
$$

For Gram–Charlier sources, factoring the source polynomial gives the angular dependence explicitly. The resulting formula supplies both the amplitude for the exact TV exponent and the Fourier coefficient for rotation estimation. The exponential moment condition $\mathbb { E } e ^ { c \| X \| } < \infty$ for all c implies that $\Phi _ { x }$ extends to an entire function on $\mathbb { C } ^ { 2 }$ . Symmetry of X then gives E $\cos ( z ^ { T } X ) = \Phi _ { x } ( z )$ for $z \in \mathbb { C } ^ { 2 }$ . For $\phi _ { a }$ the source CF is $\chi ( u ) = e ^ { - u ^ { 2 } / 2 } ( 1 + a u ^ { 4 } / 2 4 )$ , and on the circle $\omega = \widetilde { v } \sqrt { q / 2 } u ( \psi )$ we have $a _ { j } ^ { T } \omega = \tilde { v } \cos ( \psi - \alpha _ { j } )$

Lemma 4 (Circle factorization). Let the source law be $\phi _ { a }$ and let $q \geq 2 .$ Choose $\tilde { v } \in \mathbb { C } \backslash$ {0} with $| \arg \tilde { v } | < \pi / 4$ Using the principalfourth root, define $y _ { 1 } = ( a \tilde { v } ^ { 4 } / 2 4 ) ^ { - 1 / 4 } e ^ { i \pi / 4 }$ and set $y _ { 2 } = i y _ { 1 } . \ F o r \ k = 1 , 2$ , let $w _ { k }$ be the unique root of $\begin{array} { r } { \dot { } w ^ { 2 } - 2 y _ { k } w + 1 = 0 } \end{array}$ inside the unit disc. Set $\dot { U } _ { k } = w _ { k } ^ { 2 q }$ and $\Psi = 2 q ( \psi - \theta )$ . Then

$$
\begin{array} { r l } & { \Phi _ { x } \big ( \tilde { v } \sqrt { q / 2 } u ( \psi ) \big ) } \\ & { \quad = \frac { e ^ { - q \tilde { v } ^ { 2 } / 4 } \prod _ { k = 1 , 2 } ( 1 + U _ { k } ^ { 2 } - 2 U _ { k } \cos \Psi ) } { [ ( 1 + w _ { 1 } ^ { 2 } ) ( 1 + w _ { 2 } ^ { 2 } ) ] ^ { 2 q } } . } \end{array}\tag{8}
$$

For real $\tilde { v } = v > 0 , w _ { 2 } = - \bar { w } _ { 1 }$ . With $w = w _ { 1 } , U = w ^ { 2 q } \ a n d \ \Lambda _ { 0 } ( v ) = - v ^ { 2 } / 4 - 4 \log | 1 + w ^ { 2 } | , \Phi _ { x } = e ^ { q \Lambda _ { 0 } ( v ) } | 1 +$ $U ^ { 2 } - 2 U \cos \Psi | ^ { 2 }$ , and

$$
\begin{array} { r l } & { \Phi _ { x } \big \vert _ { \theta = 0 } - \Phi _ { x } \big \vert _ { \theta = \varrho } } \\ & { \qquad = - 8 e ^ { q \Lambda _ { 0 } ( v ) } ( 1 + \vert U \vert ^ { 2 } ) \operatorname { R e } ( U ) \cos ( 2 q \psi ) . } \end{array}\tag{9}
$$

Proof. Put $c = a \tilde { v } ^ { 4 } / 2 4$ . The roots of $1 + c y ^ { 4 }$ are $\pm y _ { 1 }$ and ±y<sub>2</sub>, and $\textstyle 1 + c y ^ { 4 } = \prod _ { \pm , k } ( 1 \mp y / y _ { k } )$ because $y _ { 1 } ^ { 2 } y _ { 2 } ^ { 2 } =$ $- y _ { 1 } ^ { 4 } = 1 / c$ . Since $| \arg \tilde { v } | < \pi / 4 , \arg y _ { 1 } \in ( 0 , \pi / 2 )$ and $\arg y _ { 2 } \in ( \pi / 2 , \pi )$ , so no root lies in $[ - 1 , 1 ]$ . For such $y ,$ the two roots of $w ^ { 2 } - 2 y w + 1 = 0$ are reciprocal; denote the one inside the unit disc by w. The corresponding root for $- y \mathrm { i } \mathbf { s } - w$

This choice of root rewrites each source factor in terms of angular exponentials. Using $2 y w = 1 + w ^ { 2 } \mathrm { ~ g ~ }$ ives 1 − cos $x / y = ( 1 - w e ^ { i x } ) ( 1 - w e ^ { - i x } ) / ( 1 + w ^ { 2 } )$ . Hence $( 1 - \cos x / y ) \overset { \vartriangle } { ( 1 + \cos x / y ) } = ( 1 - w ^ { 2 } e ^ { \overset { \vartriangle } { 2 i } x } ) ( 1 - w ^ { 2 } e ^ { - 2 i x } ) / \overset { \vartriangle } { ( 1 + w ^ { 2 } ) ^ { 2 } }$ For $x = \psi - \alpha _ { j }$ , we have $e ^ { 2 i x } = e ^ { 2 i ( \psi - \theta ) } e ^ { - 2 \pi i j / q }$ . Applying $\Pi _ { j = 0 } ^ { q - 1 } ( 1 - z e ^ { - 2 \pi i j / q } ) = 1 - z ^ { q }$ to the two factors yields $\prod _ { j } ( 1 - \cos ^ { 2 } ( \psi - \alpha _ { j } ) / y ^ { 2 } ) = ( 1 + U ^ { 2 } - 2 U \cos \Psi ) / ( 1 + w ^ { 2 } ) ^ { 2 q }$ . The Gaussian factors give $\begin{array} { r l } { \prod _ { j } e ^ { - \tilde { v } ^ { 2 } \cos ^ { 2 } ( \psi - \alpha _ { j } ) / 2 } = } \end{array}$ $e ^ { - q \tilde { v } ^ { 2 } / 4 }$ , since $\begin{array} { r } { \sum _ { j } e ^ { 2 i \alpha _ { j } } = 0 } \end{array}$

For real $v , y _ { 2 } = - \bar { y } _ { 1 } , \mathrm { s o } \ w _ { 2 } = - \bar { w } _ { 1 }$ and the two factors are complex conjugate. Expanding $| 1 + U ^ { 2 } - 2 U \cos \Psi | ^ { 2 }$ , the shift $\theta \mapsto \theta + \varrho$ changes the sign of cos Ψ only, which gives (9). □

On every circle $\Phi _ { x }$ is thus a trigonometric polynomial with the frequencies $0 , \pm 2 q , \pm 4 q$ . Fourier inversion preserves these angular frequencies, so $p _ { 0 } - p _ { \varrho }$ is a single harmonic of frequency $2 q$

The exact exponent proof also needs the root’s phase to vary with the radius. For real $v > 0 , y _ { 1 } = ( 2 4 / ( a v ^ { 4 } ) ) ^ { 1 / 4 } e ^ { i \pi / 4 }$ and the root $w = w ( v )$ of $w ^ { 2 } - 2 y _ { 1 } w + 1 = 0$ inside the unit disc is an analytic function of y off $[ - 1 , \dot { } 1 ]$ . It is therefore real-analytic in $v > 0 . \mathrm { A s } v  0 , y _ { 1 } $ ∞ along the ray $e ^ { i \pi / 4 }$ and $w \sim 1 / ( 2 y _ { 1 } )$ , so $| w | \to 0$ and arg $w \to - \pi / 4$ . As $v  \infty , y _ { 1 }  0$ and $w  - i .$ . In particular arg w is not constant

## A.4 Cancellation of low-order cumulants

The same rotational symmetry explains why low-order moments cannot distinguish the configurations. The order k cumulant tensor of X is cum $_ { \cdot k } ( s ) \sum _ { j } a _ { j } ^ { \otimes k }$ , which vanishes for odd k by source symmetry. For $t = \| t \| u ( \psi )$ the geometric factor has the associated form $\begin{array} { r } { T _ { k } ( t ) = \sum _ { j } ( a _ { j } ^ { T } t ) ^ { k } = r _ { q } ^ { k } \| t \| ^ { k } \sum _ { j } \cos ^ { k } ( \alpha _ { j } - \psi ) } \end{array}$ . For even k, expand $\begin{array} { r } { \cos ^ { k } x = 2 ^ { - k } \sum _ { r = 0 } ^ { k } \binom { k } { r } e ^ { i ( k - 2 r ) x } } \end{array}$ . The sum $\textstyle \sum _ { j = 0 } ^ { q - 1 } e ^ { i ( k - { \bar { 2 r } } ) ( \theta + j \pi / q - \psi ) }$ vanishes unless 2q divides $k - 2 r$ . Hence $T _ { k }$ does not depend on θ for even $k < 2 q$ . For $k = 2 q$ its θ-dependent part is $2 ^ { 1 - 2 q } q r _ { q } ^ { 2 q } \lVert t \rVert ^ { 2 q } \cos ( 2 q ( \theta - \psi ) )$ ), so the order-2q cumulants of $P _ { 0 }$ and $P _ { \varrho }$ differ whenever $\mathrm { c u m } _ { 2 q } ( s ) \ne 0$

Equivalently, the two unit orientations of each of the q lines form a spherical $( 2 q - 1 )$ -design on the circle (Delsarte et al., 1977). The cancellation at order k requires only that the symmetric source law have a finite kth moment. Adding independent Gaussian noise with fixed covariance preserves the cancellation.

For the Gram–Charlier law, the nonzero cumulants above order two occur at multiples of four, by the series for $\log ( 1 + a t ^ { 4 } / 2 4 )$ . The first such multiple at least $2 q$ is $2 q$ for even q and $2 q + 2$ for odd $q .$ . The $\cos ^ { k }$ expansion at this order has a positive coefficient at frequency $2 q .$ , which survives the sum.

## A.5 Exponential total variation bound

The next bound combines the absence of low angular frequencies with a Gaussian growth envelope for the transform over the entire complex plane. Let $\begin{array} { r } { \mu _ { f } = \operatorname* { s u p } _ { u \in \mathbb { C } \backslash \{ 0 \} } | u | ^ { - 2 } \log | m _ { f } ( u ) | } \end{array}$ . On the real and imaginary axes, this definition gives $\mathbb { E } e ^ { u s } \le e ^ { ( 1 / 2 + \mu _ { f } ) u ^ { 2 } }$ and $| \mathbb { E } e ^ { i y s } | \le e ^ { - ( 1 / \overset { . . . } { 2 } - \mu _ { f } ) y ^ { 2 } }$ , respectively. Thus $\mu _ { f } < 1 / 2$ requires both Gaussian-type tails and CF decay. The Gaussian law has $\mu _ { f } = 0$

Proposition 1. Let $q \geq 2$ and suppose the common symmetric, unit-variance source law f has all exponential moments and satisfies $\mu _ { f } < 1 / 2 .$ . For every $z \in [ 1 , 1 / ( 2 \mu _ { f } ) )$ (with upper endpoint +∞ $i f \mu _ { f } = 0 )$

$$
\begin{array} { r l r } {  { \mathrm { T V } ( P _ { 0 } , P _ { \varrho } ) \leq \frac { 1 } { 2 } \| r _ { A _ { 0 } } - r _ { A _ { \varrho } } \| _ { L ^ { 2 } ( \phi _ { \mathrm { N } , 2 } ) } } } \\ & { } & { \leq \frac { \sqrt { 2 } z ^ { - q } } { \sqrt { ( 1 - 2 \mu _ { f } z ) ( 1 - 2 \mu _ { f } / z ) } } , } \end{array}
$$

and $i f q \ge 1 / ( 1 - 2 \mu _ { f } )$ , then $\mathrm { T V } ( P _ { 0 } , P _ { \varrho } ) \le 4 \sqrt { 2 q / ( 1 - 2 \mu _ { f } ) } ( 2 \mu _ { f } ) ^ { q } .$

Proof. We first bound the transform after rescaling the two complex coordinates. Choose $\vartheta \geq 0$ such that $2 \mu _ { f } e ^ { 2 \vartheta } < 1$ this condition is automatic when $\mu _ { f } ~ = ~ 0$ . The variables $\zeta \doteq W _ { 1 } + i W _ { 2 }$ and $\zeta ^ { \prime } = W _ { 1 } - i W _ { 2 }$ are independent circular complex Gaussians with variance 2. The factors of $P _ { A _ { 0 } } ( D _ { \vartheta } W )$ are $m _ { f } ( u _ { j } )$ , where $u _ { j } = ( r _ { q } / 2 ) ( e ^ { \vartheta - i \alpha _ { j } } \zeta +$ $e ^ { - \vartheta + i \alpha _ { j } } \zeta ^ { \prime } )$ . By the definition of $\begin{array} { r } { \mu _ { f } , | P _ { A _ { 0 } } ( D _ { \vartheta } W ) | ^ { 2 } \leq \exp ( 2 \bar { \mu _ { f } } \sum _ { j } | u _ { j } | ^ { 2 } ) } \end{array}$ , and because $\begin{array} { r } { \sum _ { j } \overset { \cdot } { e } ^ { 2 i \alpha _ { j } } = 0 , \ : \sum _ { j } | u _ { j } | ^ { 2 } = } \end{array}$ $\frac { 1 } { 2 } ( e ^ { 2 \vartheta } | \zeta | ^ { 2 } + e ^ { - 2 \vartheta } | \zeta ^ { \prime } | ^ { 2 } )$ Under the chosen restriction $2 \mu _ { f } e ^ { 2 \vartheta } < \bar { 1 }$ , the exponential moments of the independent variables $| \zeta | ^ { 2 }$ and $| \zeta ^ { \prime } | ^ { 2 }$ , each of mean 2, give $\mathbb { E } | P _ { A _ { 0 } } ( D _ { \pm \vartheta } \dot { W } ) | ^ { 2 } \le [ ( 1 - 2 \mu _ { f } e ^ { 2 \vartheta } ) ( 1 - 2 \mu _ { f } e ^ { - 2 \vartheta } ) ] ^ { - 1 }$ Setting Setting $\vartheta = 0$ in this estimate verifies the integrability needed for the Hermite representation: in this estimate verifies the integrability needed for the Hermite representation: $\| P _ { A _ { 0 } } \| _ { \mathrm { F } } < \infty$ . Lemma 3 .Lemma3 gives $\| P _ { A _ { \varrho } } \| _ { \mathrm { F } } = \| P _ { A _ { 0 } } \| _ { \mathrm { F } }$ , so Lemma 1 applies to both matrices. By Cauchy–Schwarz, Lemma 1, and Lemma 3(b),

$$
\mathrm { T V } \leq \frac { 1 } { 2 } \| P _ { A _ { 0 } } - P _ { A _ { \varrho } } \| _ { \mathrm { F } } = \| \sum _ { m \mathrm { o d d } } \Pi _ { 2 q m } P _ { A _ { 0 } } \| _ { \mathrm { F } } .
$$

Apply Lemma $3 ( \mathrm { c } )$ with $\ell _ { 0 } = 2 q$ to the positive and negative frequencies separately. The square of the last norm is at most $e ^ { - 4 q \vartheta } ( \mathbb { E } | P _ { A _ { 0 } } ( D _ { \vartheta } W ) | ^ { 2 } + \hat { \mathbb { E } } | P _ { A _ { 0 } } ( \dot { D } _ { - \vartheta } W ) | ^ { 2 } )$ . Setting $z = e ^ { 2 \vartheta }$ gives the first bound.

For the second bound, take $z = ( 1 - 1 / q ) / ( 2 \mu _ { f } ) \geq 1$ and use $1 - 2 \mu _ { f } / z \ge 1 - 2 \mu _ { f }$ and $( 1 - 1 / q ) ^ { - q } \leq 4 . ( \mathrm { I f } \ \mu _ { f } = 0 .$ then $| m _ { f } | \leq 1$ on $\mathbb { C } ,$ so $m _ { f } \equiv 1 , f$ is Gaussian and $P _ { 0 } = P _ { \varrho } . )$ □

Checking the growth condition. For $\phi _ { a }$ , the inequality $| 1 + a u ^ { 4 } / 2 4 | \le 1 + a | u | ^ { 4 } / 2 4$ is attained when $u ^ { 4 } > 0$ With $x = \sqrt { a / 2 4 } | u | ^ { 2 }$ , this gives $\mu _ { \phi _ { a } } = \mu _ { * } \sqrt { a / 2 4 }$ , where $\textstyle \mu _ { * } = \operatorname* { m a x } _ { x > 0 } x ^ { - 1 } \log ( 1 + x ^ { 2 } ) \simeq 0 . 8 0 4 7$ . In particular, $2 \mu _ { \phi _ { a } } \leq 0 . 6 \dot { 5 } 8$ for $a \leq 4$

Gaussian smoothing and rescaling also connect the laws within the Gram–Charlier family. For $a < 4 ,$ , let $c _ { a } ^ { 2 } = \sqrt { a } / 2$ and take independent $s _ { 4 } \sim \phi _ { 4 }$ and $g \sim \mathcal { N } ( 0 , 1 )$ . The variable $c _ { a } s _ { 4 } + \sqrt { 1 - c _ { a } ^ { 2 } } g$ has law $\phi _ { a }$ , as follows from its moment generating function $e ^ { u ^ { 2 } / 2 } ( 1 + a u ^ { 4 } / 2 4 )$ . Because $A _ { \theta } A _ { \theta } ^ { T } = I _ { 2 }$ , the $\phi _ { a }$ observation law is obtained by scaling the $\phi _ { 4 }$ observations by $c _ { a }$ and adding independent Gaussian noise with covariance $( 1 - c _ { a } ^ { 2 } ) I _ { 2 }$

For a second example satisfying the growth condition, take $s = \sqrt { 1 - \tau ^ { 2 } } s ^ { \prime } + \tau g$ , where $s ^ { \prime }$ is an equiprobable sign, $g$ is an independent standard Gaussian, and $0 < \tau < 1$ . Then $m _ { f } ( u ) = \cosh ( \sqrt { 1 - \tau ^ { 2 } } u ) e ^ { - ( 1 - \tau ^ { 2 } ) u ^ { 2 } / 2 }$ . For real $x , y ,$ the bounds | cosh $( x + i y ) | \leq$ cosh x and cosh $x \leq e ^ { x ^ { 2 } / 2 }$ give

$$
| m _ { f } ( x + i y ) | \leq \cosh ( \sqrt { 1 - \tau ^ { 2 } } x ) e ^ { - ( 1 - \tau ^ { 2 } ) ( x ^ { 2 } - y ^ { 2 } ) / 2 } \leq e ^ { ( 1 - \tau ^ { 2 } ) y ^ { 2 } / 2 } .
$$

Therefore $2 \mu _ { f } \le 1 - \tau ^ { 2 } < 1$ . Equality holds by taking $u = 2 \pi i / \sqrt { 1 - \tau ^ { 2 } }$ . This source law also has real CF zeros at $( k + 1 / 2 ) \pi / \sqrt { 1 - \tau ^ { 2 } } , k \in \mathbb { Z } .$ , so the same model exhibits both the TV bound and the zero lines.

## A.6 The exact TV exponent for Gram–Charlier sources

Use the scaled frequency radius v, root $w ( v )$ , and factor $\Lambda _ { 0 } ( v )$ from Lemma 4. The difference in (9) separates an exponential amplitude from an oscillating cosine. Let $e _ { a } ( \boldsymbol { v } )$ be the amplitude’s exponent at radius v, and let $\varkappa _ { a }$ be the negative of its maximum. Proposition 2 identifies $\varkappa _ { a }$ as the exact TV decay exponent.

$$
e _ { a } ( v ) = \Lambda _ { 0 } ( v ) + 2 \log | w ( v ) | ,\tag{10}
$$

$$
\varkappa _ { a } = - \operatorname* { m a x } _ { v > 0 } e _ { a } ( v ) .
$$

Since $1 + w ^ { 2 } = 2 y _ { 1 } w , e _ { a } ( v ) = - v ^ { 2 } / 4 + \log ( a v ^ { 4 } / 3 8 4 ) + 2 \log ( 1 / | w ( v ) | )$ . The maximum exists because $e _ { a }$ is continuous on $( 0 , \infty )$ and tends $\mathrm { t o } - \infty$ at both endpoints.

Proposition 2. For $\phi _ { a } , a \in ( 0 , 4 ] ;$ , and every $q \geq 2$

$$
\begin{array} { r l } & { \mathrm { T V } ( P _ { 0 } , P _ { \varrho } ) \geq 4 \underset { v > 0 } { \operatorname* { m a x } } e ^ { q e _ { a } ( v ) } | \cos ( 2 q \arg w ( v ) ) | , } \\ & { \mathrm { T V } ( P _ { 0 } , P _ { \varrho } ) \leq \big ( 1 + T _ { q } e ^ { \varkappa _ { a } } \sqrt { 8 q I _ { a } } \big ) e ^ { - \varkappa _ { a } q } , } \end{array}
$$

with $\begin{array} { r } { I _ { a } ~ = ~ \int _ { 0 } ^ { \infty } v e ^ { 2 e _ { a } ( v ) } d v } \end{array}$ and $T _ { q } ^ { 2 } = ( 2 + 4 \mu _ { \phi _ { a } } ) ( \varkappa _ { a } q + \log 8 ) / \cos ^ { 2 } ( \pi / 8 )$ . The constant $I _ { a }$ is finite: $e _ { a } ( v ) =$ 2 log $v + O ( 1 )$ as $v  0$ and $e _ { a } ( v ) = - v ^ { 2 } / 4 + 4 \log v + O ( 1 )$ as $v  \infty .$ Hence $q ^ { - 1 } \log \mathrm { T V } ( P _ { 0 } , P _ { \varrho } ) \to - \varkappa _ { a }$

Proof. For real $\omega , | \Phi _ { 0 } ( \omega ) - \Phi _ { \varrho } ( \omega ) | \leq 2 \mathrm { T V }$ . At ψ = 0, (9) gives $\left| \Phi _ { 0 } - \Phi _ { \varrho } \right| = 8 e ^ { q \Lambda _ { 0 } ( v ) } ( 1 + | U | ^ { 2 } ) | \mathrm { R e } U | \ge$ $8 e ^ { q e _ { a } ( v ) } | \cos ( 2 q \arg w )$ |. Maximizing over v gives the lower bound.

The cosine may vanish where $e _ { a }$ is maximized. We therefore choose a nearby radius where its absolute value is at least $1 / 2$ . Let $\bar { v } _ { a }$ maximize $e _ { a }$ and fix $0 ~ < ~ h ~ < ~ \bar { v } _ { a }$ Since arg w is real-analytic and not constant, its range on $[ \bar { v } _ { a } - \dot { h } , \bar { v } _ { a } + h ]$ is an interval of some length $\ell _ { h } > 0$ . Every interval on which | cos $t | < 1 / 2$ has length at most $\pi / 3$ Thus, if $2 q \ell _ { h } > \pi / 3$ , some $v \in [ \bar { v } _ { a } - h , \bar { v } _ { a } + h ]$ satisfies $| \cos ( 2 q \arg w ( v ) ) | \stackrel { \cdot } { \geq } 1 / 2$ . The lower bound then gives $\begin{array} { r } { q ^ { - 1 } \log \mathrm { T V } \geq \operatorname* { m i n } _ { | v - \bar { v } _ { a } | \leq h } e _ { a } ( v ) + q ^ { - 1 } } \end{array}$ log 2. Taking $q  \infty$ and then $h  0$ yields lim inf $q ^ { - 1 } \log \mathrm { T V } \geq - \varkappa _ { a }$ Combining this limit inferior with the TV upper bound in Proposition 1 gives $\varkappa _ { a } \geq - \log ( 2 \mu _ { \phi _ { a } } ) > 0$

For the upper bound, we split the density integral into a tail and a bounded disc. Let $\mu = \mu _ { \phi _ { c } }$ and $T > 0$ . The two configurations differ only by rotation, so $\| X \|$ has the same law under both. Thus $\begin{array} { r } { \frac { 1 } { 2 } \int _ { \| \boldsymbol { x } \| > T } | p _ { 0 } - p _ { \varrho } | \leq \mathrm { P r } _ { 0 } \{ \| \boldsymbol { X } \| > } \end{array}$ $T \}$ . The moment bound $\mathbb { E } e ^ { \lambda ^ { T } X } \ \le \ e ^ { ( 1 / 2 + \mu ) \| \lambda \| ^ { 2 } }$ gives a Gaussian tail bound in every unit direction. The eight directions $u ( k \pi / 4 ) , k = 0 , \ldots , 7 .$ , satisfy max $0 { \le } k { \le } 7 { u } ( k \pi / 4 ) ^ { T } x \ge \| x \| \cos ( \pi / 8 )$ . A union bound therefore gives $\operatorname* { P r } \{ \| X \| > \dot { T } \} \overset { } { \underset { {  } } { \sim } } \tilde { 8 } \exp ( - \dot { T } ^ { 2 } \cos ^ { 2 } ( \pi / 8 ) \dot { / } ( 2 + 4 \ddot { \mu } ) ) . \mathbf { \dot { A } } \mathbf { t } \overset { } { T } = \dot { T } _ { q }$ , this upper bound equals $e ^ { - \varkappa _ { a } q }$

Inside the disc, we use the circle formula to bound the Fourier transform of the density difference. By the Cauchy– Schwarz inequality and Plancherel’s identity, $\begin{array} { r } { \frac 1 2 \int _ { \lVert x \rVert \leq T } | p _ { 0 } - p _ { \varrho } | \leq T \lVert \Phi _ { 0 } - \Phi _ { \varrho } \rVert _ { L ^ { 2 } } / ( \hat { 4 } \sqrt { \pi } ) } \end{array}$ . In the coordinates $\omega =$ $v \sqrt { q / 2 } u ( \psi )$ , the area element is $( q / 2 )$ v dv dψ. Using $\begin{array} { r } { ( 9 ) , \int _ { 0 } ^ { 2 \pi } \cos ^ { 2 } ( 2 q \psi ) d \psi = \pi } \end{array}$ , and $( 1 + | U | ^ { 2 } ) ^ { 2 } ( { \mathrm { R e } } U ) ^ { 2 } \leq 4 | w | ^ { 4 q }$ gives

$$
\| \Phi _ { 0 } - \Phi _ { \varrho } \| _ { L ^ { 2 } } ^ { 2 } \leq 1 2 8 \pi q \int _ { 0 } ^ { \infty } v e ^ { 2 q e _ { a } ( v ) } d v \leq 1 2 8 \pi q e ^ { - 2 ( q - 1 ) \varkappa _ { a } } I _ { a } .
$$

Combining the disc and tail bounds gives the stated upper bound and lim sup $q ^ { - 1 }$ log $\mathrm { T V } \leq - \varkappa _ { a }$

For $a \ = \ 4$ , numerical evaluation gives $\varkappa _ { 4 } \simeq 1 . 5 2 8 8$ . If the observation is $Y = X + \varepsilon$ with independent $\varepsilon \sim$ $\mathcal { N } ( 0 , \sigma ^ { 2 } I _ { 2 } )$ , the same exponent formula applies with a replaced by $a / ( 1 + \sigma ^ { 2 } ) ^ { 2 }$ . Indeed, $Y / \sqrt { 1 + \sigma ^ { 2 } }$ has the noiseless observation law with source density $\phi _ { a / ( 1 + \sigma ^ { 2 } ) ^ { 2 } }$

## A.7 Sample complexity of rotation estimation

The TV bound gives a necessary sample size. A phase estimator at complex frequencies gives a sufficient sample size and hence an upper bound on the sample-size exponent. Lower bound. Since $\bar { d _ { q } } ( 0 , \varrho ) = \bar { \pi } / ( 2 q )$ , the test that decides $\theta = 0$ when $d _ { q } ( \hat { \theta } , 0 ) < \pi / ( 4 q )$ errs under $\theta = 0$ only if $d _ { q } ( \hat { \theta } , 0 ) \geq \pi / { ( 4 q ) }$ and under $\theta = \varrho$ only if $d _ { q } ( { \hat { \theta } } , \varrho ) \geq \pi / ( 4 q )$ The sum of its two error probabilities is at least $1 - \mathrm { T V } ( P _ { 0 } ^ { \otimes n } , P _ { \varrho } ^ { \otimes n } ) \geq 1 - n \mathrm { T V } ( P _ { 0 } , P _ { \varrho } )$ (Le Cam’s two-point bound (Tsybakov, 2009)), so success with probability $3 / 4$ at both rotations requires $n \geq 1 / ( 2 \mathrm { T V } )$ . Proposition 2 then gives log $\cdot n ^ { * } ( q ) \geq \varkappa _ { a } q - O ( \log q )$

Theorem 1 already determines the sample size up to absolute constants at each fixed q through $H _ { q } ^ { 2 }$ . The construction below gives an explicit exponent bound as q grows.

Upper bound with a known source law. To obtain an explicit exponential bound in $q ,$ we estimate the rotation from the phase of one angular Fourier coefficient at complex frequencies. Write $\tilde { v } = v e ^ { i \xi }$ with radius parameter $v > 0$ and phase $\xi$ satisfying $| \xi | < \pi / 4$ . The parameter ϑ below controls a contour shift; θ remains the unknown rotation. The angular average below extracts the Fourier coefficient whose phase records the rotation. Let $N _ { \mathrm { a n g } } = 4 q , \varphi _ { k } = \pi k / N _ { \mathrm { a n g } }$ and $\tilde { \omega } _ { k } = \tilde { v } \sqrt { q / 2 } u ( \varphi _ { k } ) \in \mathbb { C } ^ { 2 }$ . Define

$$
T _ { \mathrm { c f } } ( x ) = \frac { 1 } { N _ { \mathrm { a n g } } } \sum _ { k = 0 } ^ { N _ { \mathrm { a n g } } - 1 } \cos ( \tilde { \omega } _ { k } ^ { T } x ) e ^ { 2 i q \varphi _ { k } } , \qquad \hat { T } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } T _ { \mathrm { c f } } ( X _ { i } ) .
$$

Using $w _ { k }$ and $U _ { k }$ from Lemma $^ { 4 , }$ define the coefficient at zero rotation by

$$
m _ { q } = - [ U _ { 1 } ( 1 + U _ { 2 } ^ { 2 } ) + U _ { 2 } ( 1 + U _ { 1 } ^ { 2 } ) ] e ^ { - q \tilde { v } ^ { 2 } / 4 } [ ( 1 + w _ { 1 } ^ { 2 } ) ( 1 + w _ { 2 } ^ { 2 } ) ] ^ { - 2 q } .
$$

When $m _ { q } \ \ne \ 0$ , set $\hat { \theta } = ( 2 q ) ^ { - 1 } \arg ( \hat { T } / m _ { q } )$ modulo $\pi / q$ . Assign an arbitrary value when $\hat { T } = 0$ . Let $\tilde { \Lambda } ( u ) =$ $\Lambda _ { 0 } ( u ) + u ^ { 2 } / 2$ . The proposition compares the signal mean with the statistic’s second moment to control phase error uniformly over rotations. Part (d) makes the exponent explicit when one root contribution dominates the other.

Proposition 3. Let the source be $\phi _ { a } , a \in ( 0 , 4 ]$

(a) $\mathbb { E } _ { \theta } \hat { T } = m _ { q } e ^ { 2 i q \theta }$ for every θ.

(b) For every $\vartheta > 0$ , uniformly over θ,

$$
\mathbb { E } _ { \theta } | T _ { \mathrm { c f } } ( X ) | ^ { 2 } \le \bar { V } _ { q } : = 1 6 ( 1 - e ^ { - 4 q \vartheta } ) ^ { - 2 } \exp \{ q [ - 4 \vartheta + \tilde { \Lambda } ( 2 v L _ { \xi , \vartheta } ) ] \} ,
$$

where $L _ { \xi , \vartheta } = ( \sin ^ { 2 } \xi \cosh ^ { 2 } \vartheta + \cos ^ { 2 } \xi \sinh ^ { 2 } \vartheta ) ^ { 1 / 2 } .$

(c) $I f m _ { q } \ne 0$ and $n \geq 8 \bar { V } _ { q } / | m _ { q } | ^ { 2 } ;$ , then $\operatorname* { P r } _ { \theta } \{ d _ { q } ( \hat { \theta } , \theta ) \leq \pi / ( 8 q ) \} \geq 3 / 4 .$ for every θ.

(d) Suppose $| w _ { 1 } | > | w _ { 2 } | , | U _ { 2 } / U _ { 1 } | \le 1 / 8 ,$ , and $| U _ { 1 } | \leq 1 / 2 .$ Then $\bar { V } _ { q } / | m _ { q } | ^ { 2 } \leq 6 4 ( 1 - e ^ { - 4 q \vartheta } ) ^ { - 2 } e ^ { q \Gamma _ { a } ( v , \xi , \vartheta ) }$ , where

$$
\begin{array} { c } { { \Gamma _ { a } ( v , \xi , \vartheta ) = - 4 \vartheta + \tilde { \Lambda } ( 2 v L _ { \xi , \vartheta } ) - 2 \ell _ { \xi } , } } \\ { { \ell _ { \xi } = - \operatorname { R e } ( \tilde { v } ^ { 2 } ) / 4 - 2 \log | 1 + w _ { 1 } ^ { 2 } | - 2 \log | 1 + w _ { 2 } ^ { 2 } | + 2 \log | w _ { 1 } | . } } \end{array}
$$

Proof. Signal mean. (a) The mean $\operatorname { \mathbb { E } } \cos ( \tilde { \omega } _ { k } ^ { T } X ) = \Phi _ { x } ( \tilde { \omega } _ { k } )$ is given by (8) at $\psi = \varphi _ { k }$ . As a function of angle, this is a trigonometric polynomial with frequencies $0 , \pm 2 q , \pm 4 q$ . For $\ell \in \{ 0 , \pm 2 q , \pm 4 q \}$ , the average of $e ^ { i ( \ell + 2 q ) \varphi _ { k } }$ over the 4q grid points is zero unless $\ell = - 2 q$ . Thus the discrete average extracts only that coefficient, giving $m _ { q } e ^ { 2 i q \theta }$

Second-moment bound. (b) Write $x = \| x \| u ( \psi _ { x } )$ and $z = \tilde { v } \sqrt { q / 2 } \| x \|$ . The Jacobi–Anger expansion shows that the discrete average retains $J _ { 2 q j } ( z )$ with $j \equiv - 1$ (mod 4), each with a coefficient of modulus one. The absolute values of these indices are $2 q , \bar { 6 } q , 1 0 q , . . . ,$ each occurring once. Since $| J _ { - n } ( z ) | = | J _ { n } ( z ) |$ , we obtain $| T _ { \mathrm { c f } } ( x ) | \le$ $\sum _ { j \ge 1 \mathrm { ~ o d d } } \left| J _ { 2 q j } ( z ) \right|$ . Let $\begin{array} { r } { I _ { 0 } ( r ) = ( 2 \pi ) ^ { - 1 } \int _ { - \pi } ^ { \pi } e ^ { r \cos t } } \end{array}$ dt denote the modified Bessel function of order zero. The contour shift provides geometric decay in the Bessel index, so the retained terms can be summed. Moving the contour in $\begin{array} { r } { J _ { n } ( z ) \dot { = } ( 2 \pi ) ^ { - 1 } \int _ { - \pi } ^ { \pi } e ^ { i ( z \sin t - n t ) } d t } \end{array}$ to $t - i \vartheta$ gives $| J _ { n } ( z ) | \leq e ^ { - n \vartheta } I _ { 0 } ( | z | L _ { \xi , \vartheta } )$ . Summing over positive odd indices therefore yields $| T _ { \mathrm { c f } } ( x ) | \le I _ { 0 } ( c \| x \| ) e ^ { - 2 q \vartheta } / ( 1 - e ^ { - 4 q \vartheta } )$ , where $c = v \sqrt { q / 2 } L _ { \xi , \vartheta }$

To bound the remaining factor, write $\begin{array} { r } { I _ { 0 } ( c \| x \| ) = ( 2 \pi ) ^ { - 1 } \int _ { 0 } ^ { 2 \pi } e ^ { c u ( t ) ^ { T } x } d t } \end{array}$ . Then $\mathbb { E } [ I _ { 0 } ( c \| X \| ) ^ { 2 } ]$ is an average of $\mathbb { E } e ^ { \lambda ^ { T } X }$ over vectors $\lambda = c [ u ( t ) + u ( s ) ]$ . All such vectors satisfy $\left\| \lambda \right\| \leq 2 c$ . By symmetry, the moment generating function along each ray is the expectation of a hyperbolic cosine and is nondecreasing with the radius. For real $\lambda , ( 7 )$ gives $\mathbb { E } e ^ { \lambda ^ { T } X } = e ^ { \| \lambda \| ^ { 2 } } \Phi _ { x } ( \lambda )$ because the Gram–Charlier quartic factor is unchanged under $t \mapsto i t$ . On the boundary $\| \lambda \| = 2 c ,$ apply Lemma 4 at the real radius parameter $2 v L _ { \xi , \vartheta }$ . Since $| U | < \dot { 1 } , | 1 + U ^ { 2 } - 2 U$ cos $\Psi | ^ { 2 } \leq 1 6$ , and the moment generating function there is bounded by $1 6 e ^ { q \tilde { \Lambda } ( 2 v L _ { \xi , \vartheta } ) }$

Phase error and parameter bounds. $\mathrm { ( c ) } \mathrm { I f } | \hat { T } - \mathbb { E } \hat { T } | \leq | m _ { q } | / \sqrt { 2 } ,$ the angle error satisfies $d _ { q } ( \hat { \theta } , \theta ) \leq \pi / ( 8 q )$ . Chebyshev’s inequality gives $\operatorname* { P r } _ { \theta } \{ | \hat { T } - \mathbb { E } _ { \theta } \hat { T } | > | m _ { q } | / \sqrt 2 \} \le 2 \bar { V } _ { q } / ( n | m _ { q } | ^ { 2 } ) \le 1 / 4$

(d) The required lower bound on the signal amplitude follows from the reverse triangle inequality:

$$
| U _ { 1 } ( 1 + U _ { 2 } ^ { 2 } ) + U _ { 2 } ( 1 + U _ { 1 } ^ { 2 } ) | \ge | U _ { 1 } | ( 1 - | U _ { 2 } | ^ { 2 } ) - | U _ { 2 } | ( 1 + | U _ { 1 } | ^ { 2 } ) \ge | U _ { 1 } | / 2 .
$$

Hence $| m _ { q } | \geq e ^ { q \ell _ { \xi } } / 2$

For each fixed admissible triple, log $n ^ { * } ( q ) \leq \Gamma _ { a } ( v , \xi , \vartheta ) q + O ( 1 )$ as $q \to \infty$ . Define

$$
\Gamma _ { a } = \operatorname* { i n f } _ { \substack { v , \vartheta > 0 , \vert \xi \vert < \pi / 4 } } \Gamma _ { a } ( v , \xi , \vartheta ) .
$$

Taking the limsup for each fixed triple and then the infimum gives lim $\begin{array} { r } { \operatorname* { s u p } _ { q \to \infty } q ^ { - 1 } \log n ^ { * } ( q ) \le \Gamma _ { a } } \end{array}$ . For each such triple the remaining conditions in part (d) hold for all sufficiently large q.

For $a = 4$ , evaluate Proposition 3 at $( v , \xi , \vartheta ) = ( 2 . 0 8 9 9 , 0 . 1 8 1 1 , 0 . 2 6 4 1 )$ . This gives $\Gamma _ { 4 } ( v , \xi , \vartheta ) \simeq 2 . 3 6 8 8 4 7 3 5 ;$ numerical checks indicate that part (d)’s conditions hold for $q \ \geq \ 9 .$ . Whenever those exact inequalities hold, the finite-q guarantee is

$$
n ^ { * } ( q ) \leq 5 1 2 ( 1 - e ^ { - 4 q \vartheta } ) ^ { - 2 } e ^ { q \Gamma _ { 4 } ( v , \xi , \vartheta ) } + 1 .
$$

For any $0 < a \le 4$ , multiplying this v by $( 4 / a ) ^ { 1 / 4 }$ leaves $a \tilde { v } ^ { 4 }$ and both roots $w _ { k }$ unchanged, and provides a finite upper exponent. $\mathrm { A t } a = 1$ , the fixed triple (1.64571694, 0.32111649, 0.47791005) gives $\bar { \Gamma _ { 1 } } ( v , \xi , \vartheta ) \bar { \simeq } 3 . 9 1 1 2 7 6 8 4 .$ These computations approximate bounds at fixed parameters; they do not certify decimal endpoints or the parameter infimum. Combining the bounds gives

$$
\varkappa _ { a } \leq \operatorname* { l i m } _ { q  \infty } \operatorname* { i n f } _ { q } \frac { \log n ^ { * } ( q ) } { q } \leq \operatorname* { l i m } _ { q  \infty } \operatorname* { s u p } _ { q } \frac { \log n ^ { * } ( q ) } { q } \leq \operatorname* { m i n } \{ 2 \varkappa _ { a } , \Gamma _ { a } \} .
$$

Conjecture. The existence of the limit of $q ^ { - 1 } \log n ^ { * } ( q )$ is open. For each fixed $a \in ( 0 , 4 ]$ , we conjecture convergence to $\Gamma _ { a } ,$ , equivalently the same limit for $q ^ { - 1 } \log ( 1 / H _ { q } ^ { 2 } )$ . The radial and complex-frequency estimators extract the same angular harmonic. The unresolved step is to show that the optimized complex-frequency signal and second moment bounds lose no exponential factor relative to the radial estimator. The following fits describe the behavior over a finite range of $q .$

Finite-range exponent fits. For each row of Table 1, we fit log $H _ { q } ^ { 2 } = - \widehat { \gamma } q + c \log q + d$ by ordinary least squares on all integer $q$ in the indicated interval. The fitted coefficients $\widehat { \gamma }$ are close to the fixed-parameter values of Proposition 3 reported above. The table also gives the maximum absolute log residual. Comparing the overlapping windows checks sensitivity to the fitting range. Because the windows overlap, they provide no independent asymptotic validation; the limiting rate and subexponential correction remain open.

Table 1: Finite-range fits to squared Hellinger distances; both windows end at $q = 5 6$ . Residuals are in natural-log units. The last column reports the numerical upper-bound evaluations from Proposition 3 at the fixed parameters given above.
<table><tr><td>a  $q _ { \mathrm { m i n } }$ </td><td></td><td> $\widehat { \gamma }$ </td><td></td><td>C Max. log residual</td><td>Theoretical upper bound (numerically evaluated)</td></tr><tr><td>1</td><td>24</td><td>3.91120</td><td>0.4940</td><td> $3 . 1 6 \times 1 0 ^ { - 5 }$ </td><td>3.91128</td></tr><tr><td>1</td><td>30</td><td>3.91121</td><td>0.4946</td><td> $1 . 2 3 \times 1 0 ^ { - 5 }$ </td><td>3.91128</td></tr><tr><td>4</td><td>24</td><td>2.36864</td><td>0.4864</td><td> $5 . 6 9 \times 1 0 ^ { - 4 }$ </td><td>2.3689</td></tr><tr><td>4</td><td>30</td><td>2.36873</td><td>0.4900</td><td> $1 . 0 3 \times 1 0 ^ { - 4 }$ </td><td>2.3689</td></tr></table>

Computing the plotted distances. Equation (1) reduces the integrals for the half-spacing pair to a radial integral with an angular average. We construct the radial polynomials with $6 0 + 3 q$ digits of working precision. To avoid cancellation between nearly equal square roots, the angular Hellinger integrand is evaluated in the equivalent form $[ 2 V / ( \sqrt { U + V } + \sqrt { U - V } ) ] ^ { 2 }$ , with $U = 1 + C ( r ) \cos ( 2 t )$ and $V = B ( r ) \cos t .$ The radial integrands are multiplied by $e ^ { 5 q }$ for the squared Hellinger distance and $M _ { q }$ , and by $e ^ { 3 q }$ for total variation during quadrature; these factors are removed afterward.

We check convergence at $a \in \{ 0 . 5 , 1 , 2 , 3 , 4 \}$ and $q \in \{ 3 , 1 0 , 2 0 , 5 6 \}$ . The refined calculation uses $1 0 0 + 3 q$ digits for the coefficients instead of $6 0 + 3 q$ , and 384 angular nodes instead of 192. It also extends the radial cutoff in $u \bar { = } r ^ { 2 } / 2$ from $1 8 q + 4 0$ to $2 4 q + 6 0$ . Table 2 shows two cases. Across all 20 comparisons, the largest relative change in $H _ { q } ^ { 2 }$ is $2 . 4 0 \times 1 0 ^ { - 1 5 }$ (rounded up). The full distance data cover $q = 3 , \ldots , 5 6$

${ \mathrm { A t ~ } } a = 4 { \mathrm { . } }$ , independent Cartesian integration with product Hermite polynomials gives $H _ { 1 0 } ^ { 2 } \simeq 4 . 1 4 5 1 8 5 2 2 1 6 2 5 2 3 9 \times$ $1 0 ^ { - 9 }$ . The relative difference from the radial value is $2 . 4 8 \times 1 0 ^ { - 1 4 }$ (rounded up).

Table 2: Radial refinement at $a = 4$ . Relative change is $| H _ { \mathrm { r e f i n e d } } ^ { 2 } - H _ { \mathrm { b a s e } } ^ { 2 } | / H _ { \mathrm { r e f i n e d } } ^ { 2 }$ ; the first row agrees at stored precision. These are consistency checks, not certified error bounds.
<table><tr><td>q</td><td>Refined  $H _ { q } ^ { 2 }$ </td><td>Relative change</td></tr><tr><td>10</td><td> $4 . 1 4 5 1 8 5 2 2 1 6 2 5 1 3 6 \times 1 0 ^ { - 9 }$ </td><td>No change at stored precision</td></tr><tr><td>56</td><td> $4 . 4 4 3 4 0 2 9 7 1 6 7 9 8 7 2 \times 1 0 ^ { - 5 6 }$ </td><td> $2 . 0 4 \times \bar { 1 0 } ^ { - 1 6 }$ </td></tr></table>

## A.8 Normal sensitivity of a binary-source zero line

In the binary-source example, the normal derivative measures how the CF changes across a zero line. We now determine its exponential decay rate as the number of sources grows. In the binary model of Section 4, take $b > 0 , \tau \geq 0 .$ and $b ^ { 2 } + \tau ^ { 2 } = 1$ . Put $\omega _ { q } ( \dot { \nu } ) = \rho _ { 0 } u ( \theta ) + \nu u ^ { \perp } ( \theta )$ and $S _ { q } ( \bar { \nu } ) = \lVert \nabla _ { \omega } \Phi _ { x } ( \omega _ { q } ( \bar { \nu } ) ) \rVert$ . For any fixed $L \geq 0 .$

$$
\operatorname * { l i m } _ { q \to \infty } \frac { 1 } { q } \log \operatorname * { s u p } _ { | \nu | \leq L } S _ { q } ( \nu ) = - \gamma _ { \mathrm { b i n } } - \frac { \pi ^ { 2 } \tau ^ { 2 } } { 1 6 b ^ { 2 } } , \qquad \gamma _ { \mathrm { b i n } } = - \frac { 1 } { \pi } \int _ { 0 } ^ { \pi } \log \cos \left( \frac { \pi } { 2 } \cos \varphi \right) d \varphi .\tag{11}
$$

The same limit holds at $\nu = 0$ , the point on the target line nearest the origin. More precisely, log $S _ { q } ( 0 ) = - q ( \gamma _ { \mathrm { b i n } } +$ $\pi ^ { 2 } \tau ^ { 2 } / ( 1 6 b ^ { 2 } ) ) + \textstyle { \frac { 3 } { 2 } } \log q + O ( 1 )$

Proof. On the target zero line, differentiating any other factor leaves the target factor equal to zero. Differentiating the target factor at its first zero therefore gives

$$
S _ { q } ( \nu ) = b \sqrt { 2 / q } e ^ { - \tau ^ { 2 } ( \rho _ { 0 } ^ { 2 } + \nu ^ { 2 } ) / 2 } \prod _ { j = 1 } ^ { q - 1 } \left| \cos \left( \frac { \pi } { 2 } \cos \varphi _ { j } + b \sqrt { 2 / q } \nu \sin \varphi _ { j } \right) \right| , \qquad \varphi _ { j } = j \pi / q .
$$

At $\nu = 0$ , the cosine is positive between the endpoints and has a quadratic zero at each endpoint. We separate these endpoint zeros from the smooth part before estimating the logarithm of the product. The function $h ( \varphi ) =$ log[cos $( ( \dot { \pi } / 2 ) \cos \varphi ) / \sin ^ { 2 } \varphi ]$ extends smoothly to both endpoints, with value $\log ( \pi / 4 )$ . Its sum over the grid differs from $q / \pi$ times its integral by $O ( 1 )$ . Using $\textstyle \prod _ { j = 1 } ^ { q - 1 }$ sin $\varphi _ { j } = q / 2 ^ { q - 1 }$ and $\textstyle \int _ { 0 } ^ { \pi }$ log sin $\varphi d \varphi = - \pi \log 2 \mathrm { g i v e s } - q \gamma _ { \mathrm { b i n } } +$ $2 \log q + O ( 1 )$ for the log product. Together with the Gaussian factor and $b { \sqrt { 2 / q } }$ this proves the expansion at $\nu = 0$ For the uniform upper bound over $| \nu | \leq L$ , we must also allow perturbations that move cosine factors through zero. We therefore truncate the logarithm before passing to a Riemann integral. Fix $K > 0$ and use $f _ { K } ( t ) =$ max $\mathbf { \partial } \cdot \left\{ \log | \cos t | , - K \right\}$ , assigning the value −K at zeros of cos t. The truncated function is continuous, even at the zeros of the cosine. The argument perturbation is uniformly at most $b { \sqrt { 2 / q } } L$ . For fixed $K$ , uniform continuity of $f _ { K }$ and convergence of the Riemann sums imply that the average of the truncated terms converges, uniformly over $\begin{array} { r } { \vert \dot { \nu } \vert \leq L , \operatorname { t o } \pi ^ { - 1 } \int _ { 0 } ^ { \pi } f _ { K } \vert ( \pi / 2 ) } \end{array}$ cos φ) dφ. Let $K  \infty$ to remove the truncation. Monotone convergence applied to the negatives of the truncated terms bounds the limsup of the log product divided by $q \mathrm { \Delta b y - { \gamma _ { \mathrm { b i n } } } }$ . The Gaussian factor contributes at most $- \pi ^ { 2 } \tau ^ { 2 } / ( 1 6 b ^ { 2 } )$ to the limit, and $\nu = 0$ gives the matching lower bound. □

Integrating the smooth remainder h gives $\gamma _ { \mathrm { b i n } } \simeq 1$ .50606. A $\cdot b ^ { 2 } / \tau ^ { 2 } = 1 0$ the Gaussian contribution is about 0.06169. Equation (11) gives a rate for the largest normal slope on a bounded segment and allows individual offsets to have smaller slopes.

## A.9 Finite-sample illustration

Simulation protocol. We use $a = 4$ and $q \in \{ 3 , 4 , 5 \}$ . For each q and each $x \in \{ 0 . 0 5 , 0 . 1 5 , 0 . 5 , 1 . 5 , 4 \}$ , we round $x / H _ { q } ^ { 2 }$ to the nearest positive integer to obtain n. Figure 2 plots the resulting $n H _ { q } ^ { 2 }$ . We draw 400 independent samples of size n at each pair $( q , n )$ and evaluate both estimators on each sample. $\mathbf { A } \mathbf { t } \theta = 0$ , the unweighted statistic has mean $\begin{array} { r } { \mathbb { E } _ { 0 } [ e ^ { 2 i q \psi } ] = \int _ { 0 } ^ { \infty } A _ { 1 } ( \sqrt { 2 u } ) e ^ { - u } d u } \end{array}$ . This mean is positive at all three values of $q .$

To sample $\phi _ { 4 }$ , we draw from the proposal with density $\phi ( s ) ( s ^ { 4 } + 9 ) / 1 2$ and accept with probability $( s ^ { 2 } - 3 ) ^ { 2 } / ( s ^ { 4 } + 9 )$ The proposal is a mixture, with weights $3 / 4$ and $1 / 4$ , of a standard normal variable and a chi variable with five degrees of freedom multiplied by an independent equiprobable sign.

We record the sample sizes, seed, success counts for each estimator, and counts of paired outcomes on which the estimators disagree. Pointwise nominal 95% Wilson score intervals (Wilson, 1927; Brown et al., 2001) quantify the Monte Carlo uncertainty of each estimator’s success proportion.

## B Refining a labeled mixing column using characteristic-function zeros

We refine a labeled mixing column using a known real zero of its source characteristic function and a search window that isolates the target. We first identify the target line (Section B.1), then give conditions for a unique minimum using a finite frequency grid (Section B.3), and finally derive the local covariance (Section B.4). Table 3 compares the targets, supplied information, and limits used in the paper. The source family here differs from the Gram–Charlier family in Theorem 1, whose characteristic function is strictly positive on the real axis.

Table 3: Overview of the statistical targets, supplied information, and limiting regimes.
<table><tr><td>Analysis</td><td>Source family and supplied information</td><td>Target</td><td>Scope and limit</td></tr><tr><td>Theorems 1-2</td><td>Gram-Charlier; 2  $q ,$  source law, spacing and column norm known</td><td>Common rotation modulo  $\pi / q$ </td><td>Global, finite  $n , q ;$  exponents at fixed  $a , q \to \infty$ </td></tr><tr><td>Appendix B</td><td>Independent sources; known target real zero; fixed or independent isolating window; unique grid</td><td>Labeled direction and zero-line offset</td><td>Local; fixed model and grid,  $n \to \infty$ </td></tr><tr><td>Appendix C</td><td>minimum Two sources under Assumption A; target label known; source laws, scales and second direction unknown</td><td>Labeled direction</td><td>Local risk; fixed finite jump paths before  $n  \infty ,$ </td></tr><tr><td>Appendix D</td><td>Known Gram–Charlier law and equal geometry; one complex zero and an isolating angle interval</td><td>Common rotation near the true angle</td><td> $t = n ^ { - 1 / 2 }$  Local; first  $n \to \infty$  at fixed  $q ,$  then variance coefficient as  $q \to \infty$ </td></tr></table>

## B.1 The target line and the search region

Write $X = a _ { 1 } s _ { 1 } + \eta$ , where $a _ { 1 } ~ = ~ r _ { 1 } u ( \theta _ { 1 } )$ is the target column and $r _ { 1 } ~ > ~ 0 .$ . The aggregate interference $\eta$ is independent of the target symbol $s _ { 1 }$ . For finitely many mutually independent sources with characteristic functions (CFs) $\chi _ { j } ( t ) = \mathbb { E } e ^ { i t s _ { j } }$

$$
\Phi _ { x } ( \omega ) = \prod _ { j } \chi _ { j } ( a _ { j } ^ { T } \omega ) .\tag{12}
$$

Moment invariance under independent interference. A source zero yields a moment that remains zero after independent interference is added. Take $c ^ { T } a _ { 1 } = 1$ and $d ^ { T } a _ { 1 } = 0$ . The first projection retains the target symbol with unit coefficient; the second removes it. I $\mathbb { E } h ( s _ { 1 } + t ) = 0$ for every real t, then

$$
\mathbb { E } [ h ( c ^ { T } X ) g ( d ^ { T } X ) ] = 0\tag{13}
$$

whenever the product is absolutely integrable. Indeed, by independence and Fubini’s theorem, integrating the product over $s _ { 1 }$ first gives $g ( d ^ { T } \eta ) \mathbb { E } _ { s _ { 1 } } [ h ( \bar { s } _ { 1 } + \check { c } ^ { T } \eta ) ] = 0$ at every shift, for almost every $\eta .$ For a real zero $\omega _ { 0 }$ of $\chi _ { 1 }$ , take $h ( s ) = e ^ { i \omega _ { 0 } s }$ , which has zero mean at every shift, and any bounded $g .$ Such a nonzero bounded h obstructs bounded completeness of the location family (Mattner, 1993; Hu and Shiu, 2022); completeness in larger function classes requires separate conditions (Mattner, 1992; Andrews, 2017). A related CF-zero cancellation removes input dependence in subtractively dithered quantization under nonoverload conditions (Schuchman, 1964; Rapp et al., 2019).

Parametrize a line by $\omega ( \theta , \rho , \nu ) = \rho u ( \theta ) + \nu u ^ { \perp } ( \theta )$ , with $\rho > 0$ . Its offset from the origin is $\rho ,$ and ν measures position along the line. The offset ρ is distinct from the half-spacing angle $\varrho$ in the rotation model. A known positive real zero $\omega _ { 0 }$ of $\chi _ { 1 }$ fixes the relation $\rho _ { 1 } = \omega _ { 0 } / r _ { \perp }$ between the target offset and the unknown amplitude $r _ { 1 }$

Using $c = u ( \theta _ { 1 } ) / r _ { 1 } , d = u ^ { \perp } ( \theta _ { 1 } )$ and $g ( z ) = e ^ { i \nu z }$ in (13) shows that the observed CF vanishes along $\omega ( \theta _ { 1 } , \rho _ { 1 } , \nu )$ Other sources can also contribute zero lines, as the binary target and discrete interferer do in Figure 3(b). The following restrictions on the search region identify the line associated with the labeled target.

Corollary 1 (Identifying a target zero line). For mutually independent sources, the real zero set $o f \Phi _ { x }$ is the union of the hyperplanes $a _ { j } ^ { \bar { T } } \omega = t ,$ where t ranges over the real zeros of the corresponding source CF. In two dimensions, assume that no source $C F$ vanishes on an open interval. Work in a local chart for directions modulo π, with $\rho > 0 .$ Let the search region R contain $( \theta _ { 1 } , \rho _ { 1 } )$ and exclude the directions of other sources with real zeros. Assume that it also excludes every other offset ρ satisfying $\chi _ { 1 } ( \rho r _ { 1 } ) = 0 .$ . Then, for every $\varepsilon > 0 , \left( \theta _ { 1 } , \rho _ { 1 } \right)$ is the unique point in R whose candidate line satisfies $\Phi _ { x } ( \omega ( \theta , \rho , \nu \bar { ) } ) = 0 f o r \ : a l l \ : | \nu | < \varepsilon .$

![](images/53a438e44e88cc249f5ce96a0c1aeffaae5d1f5a6d9ad773f12d911935a92f0a.jpg)

![](images/9a305ca570f06d4255b4f814bbe306b1d8e74b725819540442bbe8a7aa798458.jpg)  
Figure 3: (a) Magnitude of the angular derivative at the first zero line of the equally spaced noisy binary model, at $\mathit { \bar { b } } ^ { 2 } / \tau ^ { 2 } = 1 0$ and a fixed tangential offset $\nu = 1$ . The dashed curve is the bound from (6), which retains only Gaussian damping. The exact rate in (11) concerns the maximum normal slope over a bounded segment centered at the point nearest the origin. (b) Population CF magnitude for the labeled-target example with Gaussian, four-level pulse-amplitude-modulated (PAM-4) and impulsive interference. The marked lines are the first positive and negative zeros from the binary target and the PAM-4 interferer. The color scale is floored at $1 0 ^ { - 6 }$ for display.

Proof. The union follows from (12). A finite union of closed sets with empty interior cannot cover an open interval. If the product vanishes along an interval, some factor must vanish on a subinterval. Its argument must be constant in $\nu ,$ so its column is parallel to the line normal. The angular and offset restrictions then select the target pair, using the fact that real CF zeros occur in sign pairs. □

In the equally spaced example, a connected search window containing the target and excluding both neighboring directions places every candidate direction within one spacing of $\theta _ { 1 }$ . Within the window we refine the direction and offset using characteristic-function moments evaluated on candidate lines. The local precision depends on both the slopes of these moments and their covariance at a fixed model size and a fixed number of frequency samples.

In Figure 3(a), the trial frequency is $\rho _ { 0 } u ( \theta ^ { \prime } ) + \nu u ^ { \perp } ( \theta ^ { \prime } )$ . At the true angle the angular derivative has magnitude $| \nu |$ times the normal slope because the characteristic function has zero tangential derivative on the line.

## B.2 Radial aliases for discrete sources

Corollary 1 identifies candidate zero lines. Here we describe their offsets for the discrete source laws used in the zero-line example.

For an equiprobable binary target $s _ { 1 } ~ = ~ \pm 1$ with amplitude $r _ { 1 }$ , the first positive zero of $\chi _ { 1 } ( \rho r _ { 1 } ) = \cos ( \rho r _ { 1 } )$ is $\rho _ { 1 } ~ = ~ \pi / ( 2 r _ { 1 } )$ . All positive zeros have offsets $\rho ~ = ~ ( 2 k + 1 ) \rho _ { 1 }$ , with $k = 0 , 1 , \ldots$ Under the parametrization $\rho = \pi / ( 2 r )$ , these offsets correspond to amplitudes $r _ { 1 } / ( 2 k + 1 )$ . An amplitude range $[ r _ { - } , r _ { + } ] \ni r _ { 1 }$ with $r _ { + } < 3 r _ { - }$ excludes all higher aliases $( k \geq 1 )$ . A PAM-4 interferer with levels $\{ \pm 1 , \pm 3 \} c$ has CF $\chi ( t ) = \cos ( c t ) \cos ( 2 c t )$ . Its zero lines have normal $a _ { j }$ , so excluding $\theta _ { j }$ from the search window excludes these lines.

## B.3 Identification and consistency with finitely many frequencies

The characteristic-function zero-line estimator (CFZ) uses the known $\omega _ { 0 }$ and a compact angle–offset region $\mathcal { R } .$ . It replaces the whole-line condition by sampled CF moments, as in empirical characteristic-function inference (Feuerverger and Mureika, 1977; Feuerverger and McDunnough, 1981). The frequency grid is finite; the search over R can be con-

tinuous. Write $\begin{array} { r } { \hat { \Phi } _ { x } ( \omega ) = n ^ { - 1 } \sum _ { i } e ^ { i \omega ^ { T } X _ { i } } } \end{array}$ and set

$$
N ( \theta , \rho ) = \sum _ { k = - L } ^ { L } | \hat { \Phi } _ { x } ( \omega ( \theta , \rho , k \hat { v } / L ) ) | ^ { 2 } ,
$$

where the fixed integer $L \geq 1$ determines the number of grid points and $\hat { v } > 0$ estimates the half-width v. In this appendix, v measures width along the line; it denotes a frequency radius in Appendix A. CFZ minimizes

$$
R ( \theta , \rho ) = \frac { 2 N ( \theta , \rho ) } { D ( \theta , \rho ) } , \qquad D ( \theta , \rho ) = N ( \theta , ( 1 - \delta ) \rho ) + N ( \theta , ( 1 + \delta ) \rho )\tag{14}
$$

for a fixed relative offset $\delta \in ( 0 , 1 )$ . The denominator uses two lines flanking the candidate line to account for CF decay away from the origin. Even when Corollary 1 identifies a unique complete zero line, a different candidate can have every grid point at a crossing with other zero lines. Finite-grid identification must exclude this possibility.

Identification. The required extra condition is that no nontarget candidate has all of its grid moments equal to zero. Let $\xi = ( \theta , \rho )$ and $\xi _ { 0 } = ( \theta _ { 1 } , \rho _ { 1 } )$ . Write $N _ { 0 } , D _ { 0 }$ for the population versions of $N , D$ in (14) at the limiting width $v > 0$ Suppose $D _ { 0 } > 0$ on the compact search region R. Under Corollary 1, it suffices to show that every nontarget segment meets fewer than $2 L + 1$ source zero lines. Each crossing covers at most one grid point, so at least one moment is nonzero on every nontarget segment. All moments vanish at $\xi _ { 0 } ;$ hence $N _ { 0 }$ vanishes only there, and $R _ { 0 } = 2 N _ { 0 } / D _ { 0 }$ has a unique minimum at the target.

A grid alias. Let two independent equiprobable binary sources have columns $a _ { 1 } = ( 1 , 0 ) ^ { T }$ and $a _ { 2 } = ( 2 , 1 ) ^ { T }$ . An independent Gaussian source with column $( 1 , - 1 ) ^ { T }$ makes the mixture overcomplete and adds no CF zeros. Take offsets $\rho \in [ \pi / 5 , 3 \pi / 5 ]$ and a sufficiently small angle window around $\theta _ { 1 } = 0$ that excludes the binary interferer’s direction. The only complete zero line in this region is $( 0 , \pi / 2 )$ . Choose the fixed width $v = L \pi$ , so that the grid points are $\nu _ { k } = k \pi$ for $k = - L , \ldots , L .$ . At the nontarget candidate $( 0 , \pi / 4 )$ , every grid point satisfies $a _ { 2 } ^ { T } \omega ( 0 , \pi / 4 , \nu _ { k } ) =$ $\pi / 2 + k \pi$ , so its grid energy is also zero. Nevertheless, for $\delta = 0 . 2$ the population denominator is positive throughout the offset interval at $\theta = 0 ,$ . To see this, take $\nu = 0$ . The smaller offset $0 . 8 \rho$ has a nonzero target factor; its interferer factor vanishes only at $\rho = 5 \pi / 1 6$ . At that offset, the larger frequency $1 . 2 \rho = 3 \pi / 8$ gives two nonzero factors. By continuity and compactness, positivity persists on a sufficiently small angle window. Thus a positive denominator and continuous-line identification do not by themselves ensure a unique grid minimum.

Consistency. Uniform convergence transfers the unique population minimum to consistent sample minimizers. Treat the search region as fixed; when it is random, condition on the independent side information that supplies it. For fixed L, the empirical CF converges uniformly on compact frequency sets by the uniform law of large numbers for bounded continuous parametric moments (Newey and McFadden, 1994, Lemma 2.4). If $\hat { v }  _ { P } v > 0$ , continuity of the population CF then gives uniform convergence of the grid energies on $\mathcal { R }$ . The positive minimum of $D _ { 0 }$ ensures that the sample denominator is uniformly positive with probability tending to one. The sample criterion converges uniformly to $R _ { 0 } ,$ , so a unique population minimum implies consistency of exact global minimizers and of approximate global minimizers with objective gap $o _ { P } ( 1 )$ ).

## B.4 Moment sensitivity and asymptotic covariance

Local precision depends on the derivatives of the grid moments with respect to angle and offset, and on the covariance of their sample averages. The next proposition gives the derivative at the target and the asymptotic covariance for a consistent estimator. The latter result keeps the observation law and grid fixed and specifies the required objective and stationarity rates.

Proposition 4 (Moment sensitivity and asymptotic covariance). In two dimensions, suppose $s _ { 1 }$ is independent of η. Assume that $\chi _ { 1 }$ has a simple positive real zero ω<sub>0</sub>: $\chi _ { 1 } ( \omega _ { 0 } ) = 0$ and $\chi _ { 1 } ^ { \prime } ( \omega _ { 0 } ) \neq 0 .$ . Write $m ( \theta , \rho ; \nu ) = \bar { \Phi _ { x } } ( \omega ( \theta , \rho , \nu ) )$ and $\xi _ { 0 } = ( \theta _ { 1 } , \rho _ { 1 } ) . A t \xi _ { 0 }$ the derivative row is

$$
( \partial _ { \theta } m , \partial _ { \rho } m ) = c ( \nu ) ( - \nu , 1 ) , \qquad c ( \nu ) = r _ { 1 } \chi _ { 1 } ^ { \prime } ( \omega _ { 0 } ) \Phi _ { \eta } ( \omega ( \theta _ { 1 } , \rho _ { 1 } , \nu ) ) ,
$$

where $\Phi _ { \eta }$ is the CF of the aggregate interference.

For the limiting covariance of CFZ, let $n  \infty$ with the observation law and L fixed. Assume $\xi _ { 0 } \in$ int R and $D _ { 0 } ( \xi _ { 0 } ) > 0$ . Require $\mathbb { E } \| X \| < \infty$ and $\hat { v } - v = O _ { P } ( n ^ { - 1 / 2 } )$ . Stack the real and imaginary population grid moments at $\xi _ { 0 }$ with width v. Let their Jacobian G have full column rank, and let Ω be the covariance of their single-observation summands. For a consistent sequence $\hat { \xi }$ satisfying

$$
R ( \hat { \xi } ) = O _ { P } ( n ^ { - 1 } ) , \qquad \| \nabla R ( \hat { \xi } ) \| = o _ { P } ( n ^ { - 1 / 2 } ) ,
$$

with derivatives taken at fixed estimated width, $\sqrt { n } ( \hat { \xi } - \xi _ { 0 } )$ converges in distribution to a centered normal with covariance

$$
( G ^ { T } G ) ^ { - 1 } G ^ { T } \Omega G ( G ^ { T } G ) ^ { - 1 } .
$$

These conditions on the estimator holdfor an exact global minimizer if $D _ { 0 } > 0$ on $\mathcal { R }$ and $N _ { 0 }$ vanishes only at $\xi _ { 0 }$ . For a random search region the conclusion is conditional on the independent side information.

Proof. By independence, m is the product of $\chi _ { 1 } ( a _ { 1 } ^ { T } \omega )$ and $\Phi _ { \eta } ( \omega )$ . At $\xi _ { 0 }$ the first factor vanishes. Its linear expansion and continuity of the second factor suffice to differentiate the product there. Since $a _ { 1 } ^ { T } \omega = r _ { 1 } [ \rho \cos ( \theta - \theta _ { 1 } ) - \nu \sin ( \theta -$ $\theta _ { 1 } ) ]$ , the row is $c ( \nu ) ( - \nu , 1 )$ . Two distinct $\nu _ { k }$ with $c ( \nu _ { k } ) \neq 0$ give full column rank over the reals.

We next account for estimating the grid width before linearizing the criterion. Let $\bar { m } ( \xi , v ^ { \prime } )$ stack the real and imaginary sample moments at width $v ^ { \prime }$ and let M<sup>¯</sup> be its derivative with respect to $\xi .$ The derivative envelopes in $\xi$ and $v ^ { \prime }$ are bounded by a constant times $\| x \|$ on compact parameter and width sets. The first-moment assumption therefore gives local uniform convergence of the derivatives. The population moments vanish at $\xi _ { 0 }$ for every width, so their population width derivative is zero there. Consequently

$$
\sqrt { n } [ \bar { m } ( { \xi } _ { 0 } , \hat { v } ) - \bar { m } ( { \xi } _ { 0 } , v ) ] = o _ { P } ( 1 ) .
$$

The bounded fixed-grid moments satisfy the multivariate central limit theorem and $R ( \xi _ { 0 } ) = O _ { P } ( n ^ { - 1 } )$ . Under the global identification conditions an exact minimizer is consistent, lies in the interior with probability tending to one, satisfies $R ( \hat { \xi } ) \leq R ( \xi _ { 0 } )$ , and has $\nabla R ( { \hat { \xi } } ) = 0$

At a consistent estimator satisfying the stated rates, the denominator derivative contributes only a smaller-order term. Differentiating (14) at fixed estimated width gives

$$
\begin{array} { r } { \bar { M } ( \hat { \xi } , \hat { v } ) ^ { T } \bar { m } ( \hat { \xi } , \hat { v } ) = \frac { 1 } { 4 } D ( \hat { \xi } ) \nabla R ( \hat { \xi } ) + \frac { 1 } { 4 } R ( \hat { \xi } ) \nabla D ( \hat { \xi } ) = o _ { P } ( n ^ { - 1 / 2 } ) , } \end{array}
$$

since $D$ is bounded and $\nabla D = O _ { P } ( 1 )$ near the target. Expanding the moments along the segment from $\xi _ { 0 }$ to $\hat { \xi }$ and using uniform derivative convergence gives

$$
( G ^ { T } G + o _ { P } ( 1 ) ) ( \hat { \xi } - \xi _ { 0 } ) = - G ^ { T } \bar { m } ( \xi _ { 0 } , v ) + o _ { P } ( n ^ { - 1 / 2 } ) .
$$

Inversion and the central limit theorem give the stated covariance.

The local conclusion also applies to an exact minimizer on a smaller compact neighborhood of $\xi _ { 0 }$ where the denominator is positive and the target is the unique population zero. Such a neighborhood exists when $D _ { 0 } ( \xi _ { 0 } ) > 0$ , the moments are differentiable, and $G$ has full column rank. Figure $4 ( \mathrm { a } )$ describes the local solution near the target $\xi _ { 0 }$ Applying the covariance formula to a numerical solution requires the stated residual rate; a fixed optimization tolerance is insufficient.

The derivative and covariance calculations also apply to bounded functions of the projection along the line. The following formulas cover these more general moments and weighted estimation.

Derivative for a general moment. Assume E∥ $X \parallel < \infty$ , and let g and its derivative be bounded. Define $m _ { g } ( \theta , \rho ) =$ $\mathbb { E } [ e ^ { i \rho u ( \theta ) ^ { T } X } g ( u ^ { \perp } ( \theta ) ^ { T } X ) ]$ . Write $w = u ( \theta _ { 1 } ) ^ { T } r$ and $z = u ^ { \perp } ( \theta _ { 1 } ) ^ { T } \eta$ for the two interference projections. At the target pair,

$$
\begin{array} { r l } & { \partial _ { \theta } m _ { g } = i r _ { 1 } \chi _ { 1 } ^ { \prime } ( \omega _ { 0 } ) \mathbb { E } [ g ^ { \prime } ( z ) e ^ { i \rho _ { 1 } w } ] , } \\ & { \partial _ { \rho } m _ { g } = r _ { 1 } \chi _ { 1 } ^ { \prime } ( \omega _ { 0 } ) \mathbb { E } [ g ( z ) e ^ { i \rho _ { 1 } w } ] . } \end{array}\tag{15}
$$

Covariance with $a$ general weight. Let $G$ be the Jacobian of the stacked real and imaginary population moments and Ω the covariance of their sample summands at the target. Under the usual local linearization for a consistent generalized method of moments estimator (Hansen, 1982; Newey and McFadden, 1994), the limiting covariance of the $\sqrt { n } -$ -scaled error is

$$
K _ { W } ^ { - 1 } G ^ { T } W \Omega W G K _ { W } ^ { - 1 } , \qquad K _ { W } = G ^ { T } W G ,
$$

where the limiting weight $W$ is symmetric positive definite and G has full column rank. For moments formed with $g _ { k }$ and $g _ { l } .$ , each covariance entry has absolute value at most sup $| g _ { k } |$ sup $| g _ { l } |$ . These formulas use a fixed grid. Extending to continuum GMM requires covariance-operator analysis (Carrasco and Florens, 2000).

Derivation. Let $y _ { \boldsymbol { \theta } } = u ( \boldsymbol { \theta } ) ^ { T } X$ and $z _ { \theta } = u ^ { \perp } ( \theta ) ^ { T } X$ . Then $\partial _ { \theta } y _ { \theta } = z _ { \theta }$ and $\partial _ { \theta } z _ { \theta } = - y _ { \theta }$ , so for $g$ bounded with a bounded derivative and $\mathbb { E } \| X \| < \infty ,$ , dominated convergence gives

$$
\begin{array} { r } { \partial _ { \theta } m _ { g } = \mathbb { E } \left[ i \rho z _ { \theta } e ^ { i \rho y _ { \theta } } g ( z _ { \theta } ) \right] - \mathbb { E } \left[ y _ { \theta } e ^ { i \rho y _ { \theta } } g ^ { \prime } ( z _ { \theta } ) \right] , } \end{array}
$$

$$
\begin{array} { r } { \partial _ { \rho } m _ { g } = \mathbb E \left[ i y _ { \theta } e ^ { i \rho y _ { \theta } } g ( z _ { \theta } ) \right] . } \end{array}
$$

![](images/6a99feaec78e73be3668c478adb7a57cb1b7aa2e7e869e40abd992572d6277d4.jpg)

(b) Unequal symbol probabilities  
![](images/37cb2de2f626001e0b1d431e99644e914d4a8abb29383e9c0424b455e3d32731.jpg)  
Figure 4: CFZ sensitivity calculated from the population distribution. (a) Predicted angular standard deviation as the PAM-4 direction approaches the target, with $n = 1 0 ^ { 4 }$ and per-interferer variance $P = 0 . 5$ . The label Target-branch SD refers to the local minimizer near the target. Shading marks where the competing direction enters the search window. (b) Angular displacement of the population minimizer near the target as the binary symbols become imbalanced. The curve is compared with its quadratic approximation at $P = 0 . 2 5$

At $( \theta _ { 1 } , \rho _ { 1 } )$ , write $y = r _ { 1 } s _ { 1 } + w$ . The target $s _ { 1 }$ is independent of $( w , z )$ . Finite first moments of $s _ { 1 }$ and $\eta$ follow from $\mathbb { E } \| \dot { X } \| < \infty$ by restricting the other independent term to an event on which its norm is bounded. Conditioning on $( w , z )$ gives

$$
\begin{array} { r l } & { \mathbb { E } [ e ^ { i \rho _ { 1 } y } \tilde { g } ( w , z ) ] } \\ & { \qquad = \chi _ { 1 } ( \omega _ { 0 } ) \mathbb { E } [ e ^ { i \rho _ { 1 } w } \tilde { g } ( w , z ) ] = 0 , } \\ & { \mathbb { E } [ y e ^ { i \rho _ { 1 } y } \tilde { g } ( z ) ] } \\ & { \qquad = r _ { 1 } \mathbb { E } [ s _ { 1 } e ^ { i \omega _ { 0 } s _ { 1 } } ] \mathbb { E } [ e ^ { i \rho _ { 1 } w } \tilde { g } ( z ) ] } \\ & { \qquad + \chi _ { 1 } ( \omega _ { 0 } ) \mathbb { E } [ w e ^ { i \rho _ { 1 } w } \tilde { g } ( z ) ] } \\ & { \qquad = - i r _ { 1 } \chi _ { 1 } ^ { \prime } ( \omega _ { 0 } ) \mathbb { E } [ e ^ { i \rho _ { 1 } w } \tilde { g } ( z ) ] . } \end{array}
$$

Here we used $\mathbb { E } [ s e ^ { i \omega s } ] = - i \chi ^ { \prime } ( \omega )$ . Substituting $\tilde { g } ( w , z ) = i \rho _ { 1 } z g ( z ) , \tilde { g } = g ^ { \prime }$ and $\tilde { g } = g$ gives (15).

Moment covariance. At the true parameter, let $g _ { k } ( X ) = e ^ { i \rho _ { 1 } y + i \nu _ { k } z }$ , where $y = u ( \theta _ { 1 } ) ^ { T } X$ and $z = u ^ { \perp } ( \theta _ { 1 } ) ^ { T } X$ . Then $\mathbb { E } g _ { k } = 0 , | g _ { k } | = 1$ , and

$$
\begin{array} { r } { \mathbb { E } [ g _ { k } \overline { { g _ { l } } } ] = \mathbb { E } [ e ^ { i ( \nu _ { k } - \nu _ { l } ) z } ] , \qquad \mathbb { E } [ g _ { k } g _ { l } ] = \chi _ { 1 } ( 2 \omega _ { 0 } ) \mathbb { E } [ e ^ { 2 i \rho _ { 1 } w + i ( \nu _ { k } + \nu _ { l } ) z } ] . } \end{array}
$$

When the source CFs are specified, both expectations can be evaluated using the product formula (12). To recover the covariance of the stacked real and imaginary parts, use Re a Re $\begin{array} { r } { b = \frac { 1 } { 2 } \operatorname { R e } ( \bar { a } b + \bar { a } \bar { b } ) } \end{array}$ and Im a Im $\begin{array} { r } { b = \frac { 1 } { 2 } \operatorname { R e } ( a \bar { b } - a b ) } \end{array}$ The cross-covariance follows from Re a Im $\begin{array} { r } { b = \frac { 1 } { 2 } \operatorname { I m } ( a b - a \bar { b } ) } \end{array}$

Damping. Each added independent interferer multiplies $c ( \nu )$ by a factor of modulus at most one. It can therefore reduce sensitivity while leaving the total variance of a single complex moment unchanged. The real and imaginary parts of each $g _ { k }$ have variances $\textstyle { \frac { 1 } { 2 } } ( 1 \pm \operatorname { R e } \mathbb { E } g _ { k } ^ { 2 } )$ , whose sum is $\mathbb { E } | g _ { k } | ^ { 2 } = 1$ for every interference law. If Ω is positive definite and $W = \Omega ^ { - 1 }$ , the first-order variance of $\hat { \theta }$ is at least $\lambda _ { \operatorname* { m i n } } ( \Omega ) / ( n \| G _ { \theta } \| ^ { 2 } )$ . Here $\begin{array} { r } { \| G _ { \theta } \| ^ { 2 } = \sum _ { k } \nu _ { k } ^ { 2 } | c ( \nu _ { k } ) | ^ { 2 } } \end{array}$ measures sensitivity to the angle.

## B.5 A small imbalance in a binary target

The local covariance above assumes a target law with an exact real zero. We now examine how a small binary imbalance changes the population criterion, keeping the grid, search region and interference law fixed. Write $X =$ $a _ { 1 } s _ { 1 } + \eta _ { \ast }$ , with $s _ { 1 }$ independent of $\eta .$ Let $\operatorname* { P r } ( s _ { 1 } = \pm 1 ) = ( 1 \pm \beta ) / 2$ , and denote the CF of η by Ψ. Then

$$
| \Phi _ { \beta } ( \omega ) | ^ { 2 } = \left[ \cos ^ { 2 } ( a _ { 1 } ^ { T } \omega ) + \beta ^ { 2 } \sin ^ { 2 } ( a _ { 1 } ^ { T } \omega ) \right] | \Psi ( \omega ) | ^ { 2 } .
$$

Thus the population numerator and denominator in (14) are $N _ { \beta } = N _ { 0 } + \beta ^ { 2 } N _ { 1 }$ and $D _ { \beta } = D _ { 0 } + \beta ^ { 2 } D _ { 1 }$ , where $N _ { 1 } , D _ { 1 }$ use the sine-squared term on the corresponding lines.

Lemma 5 (Local effect of binary imbalance). Suppose $D _ { 0 } > 0$ on a compact search region and $R _ { 0 } = 2 N _ { 0 } / D _ { 0 }$ has a unique interior minimizer ${ \xi } _ { 0 } = ( \theta _ { 1 } , \rho _ { 1 } )$ with positive definite Hessian $\dot { K _ { 0 } } = \nabla ^ { 2 } R _ { 0 } ( \xi _ { 0 } ) . \ I f { \cal N } _ { 0 } , { \cal N } _ { 1 } , D _ { 0 } , D _ { 1 } ^ { ' }$ are three times continuously differentiable near $\xi _ { 0 } ,$ then,for sufficiently small $| \beta | ,$ , the population minimizer satisfies

$$
\begin{array} { l } { { \displaystyle \xi _ { \beta } = \xi _ { 0 } - \beta ^ { 2 } K _ { 0 } ^ { - 1 } \nabla Q ( \xi _ { 0 } ) + O ( \beta ^ { 4 } ) } , } \\ { { \displaystyle Q = \frac { 2 ( N _ { 1 } D _ { 0 } - N _ { 0 } D _ { 1 } ) } { D _ { 0 } ^ { 2 } } . } } \end{array}
$$

Proof. Expanding the ratio gives $R _ { \beta } = R _ { 0 } + \beta ^ { 2 } Q + O ( \beta ^ { 4 } )$ . Apply the implicit function theorem to $\nabla R _ { \beta } = 0$ with parameter $\beta ^ { 2 }$ . Uniform convergence on the compact region keeps the minimizer near $\xi _ { 0 }$ □

When the coefficient is nonzero, the population minimizer shifts by order $\beta ^ { 2 } ;$ ; the coefficient may also vanish.

## B.6 Deterministic zero-line calculations

Figure 3(b) uses a unit-amplitude binary target at $1 7 ^ { \circ }$ and three independent interferers. The Gaussian interferer at $6 2 ^ { \circ }$ has variance $P .$ . The PAM-4 interferer at $1 0 7 ^ { \circ }$ has equiprobable levels $\{ \pm 1 , \pm 3 \} \sqrt { P / 5 }$ . The impulsive interferer at $1 5 2 ^ { \circ }$ is zero with probability 0.95 and otherwise Gaussian with variance $P / 0 . 0 5$ . Independent Gaussian sensor noise has covariance $0 . \dot { 1 } \bar { I } _ { 2 }$ . The figure uses $P = 0 . 2 5$ and balanced target symbols.

The sensitivity calculations use the angle window $[ 9 ^ { \circ } , 2 5 ^ { \circ } ]$ and amplitude interval [0.5, 2]. We set $L = 6$ and $\delta = 0 . 2 ,$ and use width $v = 1 . 3 4 9 / \mathrm { I Q R } ( u ^ { \perp } ( 1 7 ^ { \circ } ) ^ { T } X )$ . The projection uses the specified target direction and removes the target contribution. The projected Gaussian mixture is symmetric and has a known CDF, so the population interquartile range is twice the positive 0.75 quantile. With this projection fixed, the density is positive and continuous at the quartiles. The sample counterpart therefore has the width rate in Proposition 4.

Grid identification at the original directions. For balanced symbols with the PAM-4 direction at $1 0 7 ^ { \circ }$ and $P \in$ {0.25, 0.5}, the population denominator is positive and the target is the unique grid minimum. The verification uses the intersection count from Section B.3, followed by the denominator and rank checks. First, the projected Gaussian sensor noise bounds the probability of any interval by the mass of a centered noise interval of the same length. Writing $\Phi _ { \mathrm { N } }$ for the standard normal CDF gives $\mathrm { I Q R } ( u ^ { \perp } ( 1 7 ^ { \circ } ) ^ { T } X ) \ge 2 \sqrt { 0 . 1 } \Phi _ { \mathrm { N } } ^ { - 1 } ( 0 . 7 5 )$ ). Consequently the grid segment has length $2 v < 6 . 3 4$ . Transverse intersections with target zero lines are separated by at least $\pi / \sin ( 8 ^ { \circ } ) { \stackrel { - } { > } } 2 2$ , and those with PAM-4 lines by at least $\pi / ( 4 { \sqrt { P / 5 } } ) > 2 . 4 8$ . A noncoincident segment can therefore meet at most one target line and three PAM-4 lines. Gaussian and impulsive factors have no real zeros. A parallel candidate can coincide only with the target’s first zero line within the search window.

For the denominator, the two flanking offsets have ratio $3 / 2$ , so they cannot both coincide with target zeros, whose offsets are odd multiples of $\rho _ { 1 }$ . At least one flank then has positive grid energy. For the rank condition at the target, at most three grid points are PAM-4 zeros, leaving more than two nonzero rows at distinct $\nu _ { k }$

In Figure 4(a) we vary the PAM-4 direction and retain the balanced target. The predicted local standard deviation rises from $\mathrm { { \bar { 0 . 7 1 4 ^ { \circ } } } }$ at $9 0 ^ { \circ }$ separation to $0 . 8 8 7 ^ { \circ }$ at $1 8 ^ { \circ }$ separation. The local calculation remains defined in the shaded range, where choosing the solution near the target requires distinguishing it from competing solutions.

In Figure $4 ( \mathbf { b } )$ we hold the original directions fixed and change the target’s symbol probabilities. Since the grid projection removes the target contribution, the population grid stays fixed across these probabilities. The quadratic approximation in Lemma 5 is $3 . 4 4 1 4 \beta ^ { 2 }$ degrees for this example. $\dot { \mathrm { A t } } \beta = 0 . 3$ the numerical minimizer near the target is displaced by 0.3244<sup>◦</sup>, illustrating a population shift that persists as n grows.

## B.7 Annihilators from higher-order and complex zeros

Real zeros supply the bounded exponential moments used above. Higher-order and complex zeros also yield function whose expectations vanish under every real shift, subject to the integrability conditions below. These functions enter the information calculation in Appendix C. For a density $f _ { : }$ , let $\begin{array} { r } { \bar { \chi _ { f } } ( \omega ) = \int e ^ { i \omega s } f ( s ) d s } \end{array}$ . Extend this definition to complex ω wherever $\begin{array} { r } { \int e ^ { - \operatorname { I m } ( \omega ) s } f ( s ) d s < \infty } \end{array}$ . An admissible zero ω $\in \mathbb { C }$ of multiplicity m is defined by the following two conditions. First, $s ^ { j } e ^ { i \omega s } \in L ^ { 2 } ( f )$ for $0 \leq j \leq m - 1$ . Second, $\chi _ { f } ^ { ( k ) } ( \omega ) = 0 \mathrm { f o r } k < m$ and $\chi _ { f } ^ { ( m ) } ( \omega ) \neq 0$ , with derivatives taken along the real direction.

Lemma 6. If ω is an admissible zero of multiplicity m, then the functions $h _ { j } ( s ) = s ^ { j } e ^ { i \omega s }$ satisfy $\mathbb { E } h _ { j } ( s + d ) = 0 f$ or every $d \in \mathbb { R }$ and $0 \leq j < m$ , where $s \sim f .$ For the densities used in Appendix $C ,$ these functions belong to $\dot { \mathcal { E } } ( f )$

Proof. For each d the polynomial expansion of $( s + d ) ^ { j }$ and the assumed integrability give

$$
\int h _ { j } ( s ) f ( s - d ) d s = e ^ { i \omega d } \sum _ { k \leq j } { \binom { j } { k } } d ^ { j - k } ( - i ) ^ { k } \chi _ { f } ^ { ( k ) } ( \omega ) = 0 .
$$

Integrability. The moment identity in (13) requires $\mathbb { E } | h ( c ^ { T } X ) g ( d ^ { T } X ) | < \infty$ . For bounded $g$ and $h ( s ) = e ^ { i \omega _ { 0 } s }$ at a real zero, this is automatic. For $h ( s ) ~ \bar { = } ~ s ^ { j } e ^ { i \dot { \omega } _ { 0 } s }$ , a sufficient condition at an admissible zero is $\mathbb { E } [ ( 1 +$ $| c ^ { T } \eta | ^ { j } ) e ^ { - \mathrm { I m } ( \omega _ { 0 } ) c ^ { T } \eta } ] < \infty$ , by independence and the binomial formula. For a real zero this is a polynomial moment condition. For a complex zero the sign of Im $\omega _ { 0 }$ determines which tail requires exponential control.

## C Local information in the rare-jump experiment

This appendix studies local information about one labeled direction in a two-source model. Consider the baseline $X = u ( \theta _ { 1 } ) s _ { 1 } + u ( \theta _ { 2 } ) s _ { 2 } ,$ , with independent sources and distinct directions modulo π. Both source laws and scales are unknown. The target is the labeled angle $\theta _ { 1 } ;$ the second angle $\theta _ { 2 }$ is a nuisance parameter. We ask how much information remains when rare independent shifts are also allowed, evaluating information at unit source variances.

A local path $\xi \in \mathcal { T }$ varies both angles and source laws and adds a fixed finite list of independent jumps. At path coordinate $t \geq 0$ , jump j has fixed displacement $d _ { j } \in \mathbb { R } ^ { 2 }$ and probability $t \lambda _ { j }$ , where $\lambda _ { j } \geq 0$ is fixed. Write $P _ { t } ^ { \xi }$ for the one-observation law and $\theta _ { 1 , t } ^ { \xi }$ for its labeled first direction. The local scale in the risk bound is $t = n ^ { - 1 / 2 }$ . The proof first specifies the admissible changes in source laws and angles, then projects out the nuisance scores (Bickel et al., 1993, Ch. 3). Quadratic-mean expansions yield local Gaussian experiments (van der Vaart, 1998, Ch. 7). We derive the risk bound from finite sets of feasible one-sided paths, so the jump intensities remain nonnegative.

## C.1 Source assumptions and shift scores

Let f have mean $\bar { s } _ { f }$ and variance $\sigma _ { f } ^ { 2 }$ . Its location score is $\psi _ { f } = - f ^ { \prime } / f$ , with information $\begin{array} { r } { J ( f ) = \int \psi _ { f } ^ { 2 } f . } \end{array}$ We use the complex Hilbert space $L ^ { 2 } ( f )$ with inner product $\begin{array} { r } { \langle u , v \rangle _ { f } = \int u \bar { v } f ; } \end{array}$ its mean-zero subspace is $L _ { 0 } ^ { 2 } ( f ) \stackrel { \cdot } { = } \{ u \in L ^ { 2 } ( f ) $ $\textstyle \int u f = 0 \}$ . The scores are real, while the complex space also accommodates the Fourier functions used below.

Assumption A. (i) $f > 0$ on R, f is absolutely continuous, $\sigma _ { f } ^ { 2 } < \infty , J ( f ) < \infty$ and $\textstyle { \int s ^ { 2 } \psi _ { f } ^ { 2 } f < \infty }$

(ii) $\textstyle { \int f ( s - d ) ^ { 2 } / f ( s ) }$ ds $< \infty$ for every $d \in \mathbb { R } .$

(iii) $\begin{array} { r } { \operatorname* { s u p } _ { | t | \leq \tau } \int f ^ { \prime } ( s - t ) ^ { 2 } / f ( s ) d s < } \end{array}$ ∞ for some $\tau > 0$ . Assumption A holds for the Gaussian law, $\phi _ { a }$ with $0 \leq a < 4$ finite Gaussian mixtures, the logistic, Laplace and hyperbolic secant laws, Student t laws with more than two degrees of freedom, and constellations in Gaussian noise.

To identify the score components that survive the added shifts, we use functions whose expectation vanishes under every translation of a source law. For a density f satisfying Assumption $\mathrm { \Delta A ( i ) { - } ( i i ) }$ , define

$$
\mathcal { E } ( f ) = \left\{ h \in L ^ { 2 } ( f ) : \int h ( s ) f ( s - d ) d s = 0 \mathrm { f o r e v e r y } d \in \mathbb { R } \right\} .\tag{16}
$$

By the shift-ratio condition, each translated expectation is a bounded linear functional on $L ^ { 2 } ( f )$ , so $\mathcal { E } ( f )$ is closed. To describe the score directions generated by shifts, define $v _ { d } ( s ) = f ( s - d ) / f ( s ) - 1$ . Under $\mathbf { A } ( \mathrm { i i } )$ , these functions lie in $L _ { 0 } ^ { 2 } ( f )$ . Let $\mathcal { U } ( f )$ be their closed span. Since $\begin{array} { r } { \langle h , v _ { d } \rangle _ { f } = \int h ( s ) f ( s - \check { d } ) \dot { d s } - \int h f } \end{array}$ , a mean-zero function is orthogonal to $\mathcal { U } ( f )$ exactly when its expectation vanishes under every translate of $f .$ Therefore $L _ { 0 } ^ { 2 } ( f ) = \mathcal { U } ( f ) \oplus \mathcal { E } ( f )$ with $\breve { \mathcal { E } } ( f )$ defined in (16). The choice $\bar { d } = 0$ gives $\mathcal { E } ( f ) \subset L _ { 0 } ^ { 2 } ( f )$ . Because $f$ is real, $\mathcal { E } ( f )$ is closed under conjugation. The share

$$
S ( f ) = \frac { \| \Pi _ { \mathcal { E } ( f ) } ( s - \bar { s } _ { f } ) \| _ { f } ^ { 2 } } { \sigma _ { f } ^ { 2 } }
$$

is the fraction of source variance in the component orthogonal to the shift-score space. It is invariant under location and scale changes. Nuisance projection removes the component of the target source in $\mathcal { U } ( f )$ , leaving the variance fraction $S ( f )$ in the information formula.

Proposition 5 (Information retained under rare shifts). Suppose both source laws satisfy Assumption A. Let $\Delta =$ $\theta _ { 2 } - \theta _ { 1 }$ . The efficient information for $\theta _ { 1 }$ with and without the additional shifts is, respectively,

$$
\tilde { I } = \frac { S ( f _ { 1 } ) J ( f _ { 2 } ) } { \sin ^ { 2 } \Delta } , \qquad I = \frac { J ( f _ { 2 } ) - 1 / J ( f _ { 1 } ) } { \sin ^ { 2 } \Delta } .
$$

$I f \tilde { I } > 0 ,$ , every estimator sequence $\widehat { \theta } _ { 1 , n }$ satisfies

$$
\operatorname* { s u p } _ { \stackrel { F C T } { 0 < | F | < \infty } } \operatorname* { l i m i n f } _ { n  \infty } \operatorname* { m a x } _ { \xi \in F } \mathbb { E } _ { ( P _ { 1 / \sqrt { n } } ^ { \xi } ) ^ { \otimes n } } \Big [ n ( \widehat { \theta } _ { 1 , n } - \theta _ { 1 , 1 / \sqrt { n } } ^ { \xi } ) ^ { 2 } \Big ] \geq \frac { 1 } { \widetilde { I } } .
$$

The bound also applies to any larger model containing these labeled local paths.

1. $S ( f ) \leq 1 - 1 / ( J ( f ) \operatorname { V a r } ( s ) ) .$

2. $S ( f ) > 0 i f \chi _ { f }$ has an admissible zero of finite multiplicity as defined in Appendix B.

3. $S ( f ) = 0 f o r$ Gaussian and Laplace laws.

## C.2 Proof of the local information formula

The proof proceeds in three steps. We establish a quadratic-mean expansion for simultaneous changes in the angles, marginal laws, and jump probabilities. We then project out the nuisance scores and use the remaining score to derive the risk bound for finite sets of feasible paths.

Write $\Delta = \theta _ { 2 } - \theta _ { 1 }$ and $A = [ u ( \theta _ { 1 } ) , u ( \theta _ { 2 } ) ]$ , so that $s = A ^ { - 1 } x$ and det $A = \sin \Delta$ . Angle scores. The log density is $\log p ( x ) = - \log | \sin \Delta | + \log f _ { 1 } ( s _ { 1 } ) + \log f _ { 2 } ( s _ { 2 } )$ . Using $\partial s / \partial \theta _ { 1 } = - A ^ { - 1 } u ^ { \perp } ( \theta _ { 1 } ) s _ { 1 } = ( \cot \Delta s _ { 1 } , - s _ { 1 } / \sin \Delta )$ to differentiate it gives

$$
\ell _ { \theta _ { 1 } } = \cot \Delta \left( 1 - s _ { 1 } \psi _ { 1 } ( s _ { 1 } ) \right) + s _ { 1 } \psi _ { 2 } ( s _ { 2 } ) / \sin \Delta ,
$$

and symmetrically $\ell _ { \theta _ { 2 } } = - \cot \Delta \left( 1 - s _ { 2 } \psi _ { 2 } ( s _ { 2 } ) \right) - s _ { 2 } \psi _ { 1 } ( s _ { 1 } ) / \sin \Delta$ , where $\psi _ { i } = \psi _ { f _ { i } }$

Local paths. A path $\xi \in \mathcal { T }$ has $\theta _ { i , t } = \theta _ { i } + t \gamma _ { i }$ and $f _ { i , t } = f _ { i } ( 1 + t b _ { i } )$ , where $\gamma _ { i } \in \mathbb { R }$ and $b _ { i }$ is smooth, bounded, has bounded derivative, and satisfies $\textstyle \int b _ { i } f _ { i } = 0$ . It adds a finite list $( d _ { j } , \lambda _ { j } ) _ { j = 1 } ^ { K }$ of independent jumps with probabilities $t \lambda _ { j }$ for $t \geq 0$ sufficiently small. The angles stay in a common chart with distinct directions modulo $\pi ;$ both source means and scales can vary. For small $t ,$ the tilted densities satisfy Assumption A. Bounded tilts preserve positivity and (ii). To verify (i) and (iii), differentiate $f _ { i } ( 1 + t b _ { i } )$ and use

$$
\left\| \frac { f _ { i } ( \cdot - r ) } { \sqrt { f _ { i } } } \right\| _ { 2 } \leq 1 + | r | \operatorname* { s u p } _ { | z | \leq \tau } \left\| \frac { f _ { i } ^ { \prime } ( \cdot - z ) } { \sqrt { f _ { i } } } \right\| _ { 2 } , \qquad | r | \leq \tau ,
$$

which follows from absolute continuity and Minkowski’s inequality. Smooth bounded mean-zero tilts are dense in the respective $L _ { 0 } ^ { 2 } ( f _ { i } )$ spaces. Their score span closes to $\mathcal { M } = \dot { ( } L _ { 0 } ^ { 2 } ( \dot { f } _ { 1 } ) \otimes 1 ) \oplus ( 1 \otimes L _ { 0 } ^ { 2 } ( f _ { 2 } ) )$ , which includes the scale scores $1 - s _ { i } \psi _ { i } ( s _ { i } )$

Joint angle and marginal perturbations. We first verify quadratic-mean differentiability without the jumps, allowing the mixing matrix and source densities to change together. Write $g _ { t } ( s ) = \sqrt { f _ { 1 , t } ( s _ { 1 } ) f _ { 2 , t } ( s _ { 2 } ) }$ and let $U _ { t }$ be the pushforward isometry on square roots,

$$
( U _ { t } g ) ( \boldsymbol { x } ) = \vert \operatorname* { d e t } A _ { t } \vert ^ { - 1 / 2 } g ( A _ { t } ^ { - 1 } \boldsymbol { x } ) , \qquad A _ { t } = [ u ( \theta _ { 1 , t } ) , u ( \theta _ { 2 , t } ) ] .
$$

A change of variables shows that $U _ { t }$ is strongly continuous on continuous compactly supported functions. Isometry and density extend this property to $\bar { L ^ { 2 } }$ . The product square root satisfies $\begin{array} { r } { g _ { t } = g _ { 0 } + \frac { t } { 2 } ( b _ { 1 } + b _ { 2 } ) g _ { 0 } + o _ { L ^ { 2 } } ( t ) } \end{array}$ , where $b _ { i }$ is evaluated at $s _ { i }$ . For the angle derivative, $\mathrm { A } ( \mathrm { i } )$ gives $\partial _ { i } g _ { 0 }$ and $s _ { j } \partial _ { i } g _ { 0 }$ in $L ^ { 2 }$ for every $i , j$ by independence. On smooth compactly supported $^ { g , }$ differentiation gives

$$
\frac { d } { d t } U _ { t } g = U _ { t } \big [ - { \textstyle \frac { 1 } { 2 } } \mathrm { t r } ( A _ { t } ^ { - 1 } \dot { A } _ { t } ) g - \nabla g ^ { T } A _ { t } ^ { - 1 } \dot { A } _ { t } s \big ] .
$$

Its $L ^ { 2 }$ norm is bounded locally in t by a constant times $\begin{array} { r } { \| g \| _ { 2 } + \sum _ { i , j } \| s _ { j } \partial _ { i } g \| _ { 2 } } \end{array}$ . Cutoff and then mollification approxi mate $g _ { 0 }$ in this norm. The cutoff derivative contributes only a bounded multiplier on the tails, and on each compact set the coordinate weights are bounded. Integrating the uniform norm bound on difference quotients extends the derivative formula to $g _ { 0 }$ . Combine this formula with the product expansion and strong continuity on $( b _ { 1 } + b _ { 2 } ) g _ { 0 }$ . The density before adding jumps, $p _ { t } = ( U _ { t } g _ { t } ) ^ { 2 }$ , then satisfies

$$
\begin{array} { r } { \sqrt { p _ { t } } = \sqrt { p } + \frac { t } { 2 } \ell _ { 0 } \sqrt { p } + o _ { L ^ { 2 } } ( t ) , \qquad \ell _ { 0 } ( x ) = \gamma _ { 1 } \ell _ { \theta _ { 1 } } ( x ) + \gamma _ { 2 } \ell _ { \theta _ { 2 } } ( x ) + b _ { 1 } ( s _ { 1 } ) + b _ { 2 } ( s _ { 2 } ) , } \end{array}
$$

where $p = p _ { 0 }$ and the scores on the right use $s = A ^ { - 1 } x$

Adding the jumps. We now extend the expansion to the jump mixture as t approaches zero from above. Writing the square root of the mixture as a vector norm lets us control the shifted components before differentiating. Let $w _ { J } ( t )$ be the probability of the subset $J$ of jumps and $\begin{array} { r } { d _ { J } = \sum _ { j \in J } d _ { j } } \end{array}$ . The full density is $\begin{array} { r } { q _ { t } = \sum _ { J } w _ { J } ( t ) p _ { t } ( \cdot \bar { - } d _ { J } ) } \end{array}$ . Write its square root as the pointwise Euclidean norm of the vector with entries $\sqrt { w _ { J } ( t ) } \sqrt { p _ { t } ( \cdot - d _ { J } ) }$ . By the reverse triangle inequality, replacing the shifted $\sqrt { p _ { t } }$ by the correspondingly shifted $\sqrt { p }$ in all nonempty components changes this norm in $L ^ { 2 }$ by at most $\begin{array} { r } { ( \sum _ { J \neq \emptyset } w _ { J } ( t ) \dot { ) } ^ { 1 / 2 } \| \sqrt { p _ { t } } - \sqrt { p } \| _ { 2 } = O ( t ^ { 3 / 2 } ) } \end{array}$ . The same inequality permits replacing the unshifted square root by its first-order expansion with $o ( t )$ error. The resulting norm, divided by ${ \sqrt { p } } ,$ is

$$
\sqrt { w _ { \mathcal { Q } } ( t ) ( 1 + t \ell _ { 0 } / 2 ) ^ { 2 } + \sum _ { J \neq \mathcal { Q } } w _ { J } ( t ) r _ { J } } , \qquad r _ { J } ( x ) = \frac { p ( x - d _ { J } ) } { p ( x ) } .
$$

Every $r _ { J }$ belongs to $L ^ { 2 } ( P _ { 0 } )$ by A(ii) and independence in source coordinates. The weights obey $w _ { \emptyset } ( t ) = 1 -$ $t \textstyle \sum _ { j } \lambda _ { j } + O ( t ^ { 2 } ) , w _ { \{ j \} } ( t ) = t \lambda _ { j } + O ( t ^ { 2 } )$ , and $w _ { J } ( t ) = O ( t ^ { 2 } )$ for $\left| J \right| \geq 2$ . The displayed norm has pointwise derivative $\begin{array} { r } { \frac 1 2 [ \ell _ { 0 } + \sum _ { j } \lambda _ { j } ( r _ { \{ j \} } - 1 ) ] } \end{array}$ at zero. To obtain the $L ^ { 2 } ( P _ { 0 } )$ derivative, first drop the term $t \ell _ { 0 } / 2$ inside the first vector coordinate, which changes the norm by at most $t | \ell _ { 0 } | / 2$ . Then use $| \sqrt { y } - 1 | \leq | y - 1 |$ for $y \geq 0$ . These bounds dominate the difference quotient by $\begin{array} { r } { \vert \ell _ { 0 } \vert / 2 + C ( 1 + \sum _ { J \neq \emptyset } r _ { J } ) } \end{array}$ for a finite path-dependent $C$ . Dominated convergence proves the joint one-sided quadratic-mean expansion.

Nuisance span. We determine which parts of the angle score lie in the closed linear span of the marginal and jump scores. The later risk argument enforces the nonnegative jump intensities. For a single jump with $\ddot { \delta } = A ^ { - 1 } \dot { d } .$ , the score per unit intensity is $v _ { 1 , \delta _ { 1 } } + v _ { 2 , \delta _ { 2 } } + v _ { 1 , \delta _ { 1 } } v _ { 2 , \delta _ { 2 } }$ , where $v _ { i , a } = f _ { i } ( \cdot - a ) / f _ { i } - 1$ . The closed linear span of jump and marginal scores is $\mathcal { M } \oplus ( \mathcal { U } ( f _ { 1 } ) \hat { \otimes } \mathcal { U } ( f _ { 2 } ) )$ , where $\hat { \otimes }$ is the closed Hilbert tensor product. Subtracting the additive marginal terms leaves every elementary product $v _ { 1 , a } v _ { 2 , b }$ since the displacements are unrestricted.

Efficient score. Project the target-angle score onto the orthogonal complement of this nuisance space. Write $h _ { i } =$ $\Pi _ { \mathcal { E } ( f _ { i } ) } ( s _ { i } - \bar { s } _ { f _ { i } } )$ . The constant component of $s _ { 1 }$ , multiplied by $\psi _ { 2 } ,$ , is a marginal score. The $\mathcal { U } ( f _ { 1 } )$ component of $s _ { 1 } .$ multiplied by $\psi _ { 2 }$ , lies in the tensor nuisance space by Lemma 7. The residual is $\tilde { \ell } = h _ { 1 } \psi _ { 2 } /$ sin ∆. Both factors have mean zero and $h _ { 1 } \perp \mathcal { U } ( f _ { 1 } )$ , giving orthogonality to the entire nuisance space above.

It remains to remove the second-angle nuisance direction. The projected second-angle score is $- h _ { 2 } \psi _ { 1 } /$ sin $\Delta$ . Its inner product with $\tilde { \ell }$ vanishes since $\langle h _ { 1 } , \psi _ { 1 } \rangle ~ = ~ 0$ . The remaining score therefore has information $\tilde { I } ~ = ~ \| \tilde { \ell } \| ^ { 2 } ~ =$ $S ( f _ { 1 } ) { \dot { \sigma } _ { f _ { 1 } } } ^ { 2 } J ( f _ { 2 } ) / \sin ^ { 2 } \Delta$ in this jump experiment.

Without jumps, removing the projection onto M leaves $\left( { { s _ { 1 } } - { { \bar { s } } _ { f _ { 1 } } } } \right) { \psi _ { 2 } } /$ sin $\Delta$ and $- ( s _ { 2 } - \bar { s } _ { f _ { 2 } } ) \psi _ { 1 } /$ sin ∆. By Lemma $^ { 7 , }$ their inner product is $- 1 / \sin ^ { 2 } \Delta$ . Projecting off the second residual therefore gives $I = [ \bar { J } ( f _ { 2 } ) - 1 / J ( f _ { 1 } ) ] / \sin ^ { 2 } \Delta$ at unit variances, in agreement with the classical semiparametric ICA calculation (Amari and Cardoso, 1997; Chen and Bickel, 2006).

Local risk bound. The scores of $\tau$ form a convex cone. Angles and marginal tilts can be added with signed coefficients, and finite lists ofjumps can be concatenated and their intensities multiplied by nonnegative constants. For each $\xi \in \mathcal { T }$ the first-angle derivative is $\gamma _ { 1 }$ . When $\tilde { I } > 0 .$ , it equals the inner product of the joint score of ξ with $\tilde { \ell } / \tilde { I }$ . Indeed, this function has inner product one with $\ell _ { \theta _ { 1 } }$ and zero with the second-angle and other nuisance scores. It lies in the closed linear span of the tangent cone and has variance $1 / \tilde { I }$

We derive the risk bound on one-sided paths through Gaussian limit experiments; see the finite-set formulation in van der Vaart (1995, Theorem 2). Write C for the real feasible score cone and $g = \tilde { \ell } / \tilde { I }$ . We first work with the truncated loss $L _ { B } ( z ) = \operatorname* { m i n } \{ z ^ { 2 } , B \}$ , then recover squared loss after passing to the Gaussian experiment.

First fix finitely many paths with scores $h _ { 1 } , \ldots , h _ { m } \in \mathcal { C }$ and first-angle velocities $\begin{array} { r } { \boldsymbol { a } _ { i } ~ = ~ \langle \boldsymbol { g } , \boldsymbol { h } _ { i } \rangle } \end{array}$ . Let $\Lambda _ { n , i }$ be their likelihood ratios relative to $P _ { 0 } ^ { \otimes n }$ . The one-sided quadratic-mean expansions give joint convergence to $\Lambda _ { i } = \exp \{ W ( h _ { i } ) - \| h _ { i } \| ^ { 2 } / 2 \}$ . Here W is a centered Gaussian process with covariance $\mathbb { E } [ \hat { W } ( h ) W ( k ) ] \hat { = } \langle h , k \rangle$ For any prior weights $\pi _ { i } ,$ the maximum $L _ { B }$ -risk of any estimator is at least

$$
\mathbb { E } _ { 0 } \operatorname* { i n f } _ { d \in \mathbb { R } } \sum _ { i = 1 } ^ { m } \pi _ { i } \Lambda _ { n , i } L _ { B } ( d - a _ { i } ) .
$$

Here $d = \sqrt { n } (  { \hat { \theta } } _ { 1 , n } - \theta _ { 1 } )$ , and the path centers are exactly $\theta _ { 1 } + a _ { i } / \sqrt { n }$ . The infimum can be restricted to the interval spanned by the $a _ { i }$ . This infimum is a continuous function of the likelihood-ratio vector, bounded by $B \textstyle \sum _ { i } \pi _ { i } \Lambda _ { n , i }$

The likelihood ratios are uniformly integrable: they are nonnegative, have expectation one, and their limiting lognormal variables also have expectation one. Consequently the displayed expectation converges to the Bayes risk in the corresponding finite Gaussian shift experiment.

Next we compare the feasible cone with a finite-dimensional Gaussian shift experiment. Let V be any finitedimensional space spanned by feasible scores. Every finite set $H \subset V$ can be translated into C. Indeed, write its elements as $\begin{array} { r } { \bar { h _ { i } } = \sum _ { i } ^ { - } a _ { i j } c _ { j } } \end{array}$ in a basis of feasible scores $c _ { j } .$ , and choose $M _ { j } \geq \operatorname* { m a x } _ { i } ( - a _ { i j } , 0 )$ . Then $\begin{array} { r } { c = \sum _ { i } M _ { j } c _ { j } \in \mathcal { C } } \end{array}$ and $h _ { i } + c \in { \mathcal { C } }$ for every i. Translating a Gaussian shift and its linear target by c leaves its Bayes risk unchanged. Thus finite subsets of C give at least the Bayes risks of all finite priors in the Gaussian shift $Y \overset { \cdot } { = } h + Z$ on V, with $Z \sim { \mathcal { N } } ( 0 , I _ { V } )$ and target $\left. g _ { V } , h \right.$ , where $g _ { V } = \Pi _ { V } g$

To bound these risks from below, start with a Gaussian prior and approximate it by finite priors. Under $h \sim$ $\mathcal { N } ( 0 , \tau ^ { 2 } I _ { V } )$ , the conditional distribution of the target is normal with variance $\tau ^ { 2 } \lVert g _ { V } \rVert ^ { 2 } / ( 1 + \dot { \tau } ^ { 2 } )$ . Its conditional mean minimizes $L _ { B } – \mathrm { r i s k }$ , since each centered interval has maximal probability when centered at the mean of a normal law. The Bayes risk is therefore

$$
\mathbb { E } L _ { B } \left( \frac { \tau \| g _ { V } \| } { \sqrt { 1 + \tau ^ { 2 } } } Z _ { 0 } \right) , \qquad Z _ { 0 } \sim \mathcal { N } ( 0 , 1 ) .
$$

This risk is approximable by finite priors: truncate the prior to a compact set, then quantize it. Truncation changes bounded risks by a quantity tending to zero. Moving every support point by at most ε changes risks uniformly over decision rules by at most $B \varepsilon / 2 + 2 \sqrt { B } \| g _ { V } \| \varepsilon$ , using the total-variation bound between unit-covariance Gaussian shifts and the Lipschitz constant $2 \sqrt { B }$ of $L _ { B }$ . Every such finite prior and its translated path set is fixed before $n \to \infty$

It remains to remove the prior and finite-dimensional approximations. First make the finite-prior approximation arbitrarily accurate, then let $\tau  \infty$ , and finally take finite-dimensional V with $g _ { V }  g$ . The supremum over finite path sets of the limiting lower $L _ { B }$ -risk is therefore at least $\mathbb { E } L _ { B } ( \| g \| Z _ { 0 } )$ . The squared risk dominates this bound for every B. Finally, monotone convergence as $B  \infty$ gives $\begin{array} { r } { \| g \| ^ { 2 } = 1 / \tilde { I } , } \end{array}$ , with exactly the quantifier order in Proposition 5. The same restriction argument applies to any larger model containing the same labeled local paths.

## C.3 Properties of the translate-annihilator space

We now establish the projection properties used above. The location score belongs to $\mathcal { U } ( f )$ , which bounds the retained variance from above. An admissible characteristic-function zero gives a function in $\mathcal { E } ( f )$ with nonzero covariance with the source, proving that the retained variance is positive.

Lemma 7. Under Assumption A, $\psi _ { f } \in \mathcal { U } ( f )$ and $\langle s - \bar { s } _ { f } , \psi _ { f } \rangle _ { f } = 1$ . Hence $S ( f ) \leq 1 - 1 / ( \sigma _ { f } ^ { 2 } J ( f ) )$

Proof. Let $h \in { \mathcal { E } } ( f )$ and set $\begin{array} { r } { q _ { d } = [ f ( \cdot - d ) - f ] / ( d f ) = - d ^ { - 1 } \int _ { 0 } ^ { d } f ^ { \prime } ( \cdot - t ) / f d t } \end{array}$ . Assumption A(iii) and Minkowski’s inequality bound these quotients in $L ^ { 2 } ( f )$ for $| d | \leq \tau$ . Absolute continuity and Lebesgue differentiation give $q _ { d } \to \psi _ { f }$ almost everywhere as $d  0 . \mathrm { ~ A ~ }$ bounded sequence that converges almost everywhere converges weakly in $L ^ { 2 } ( f )$ to the same limit, so $\langle h , \psi _ { f } \rangle$ = lim $\langle h , q _ { d } \rangle =$ lim $\begin{array} { r } { d ^ { - 1 } \int h ( s ) [ \tilde { f ( s - d ) } - f ( s ) ] d s = 0 . } \end{array}$ Hence $\psi _ { f } ~ \bot ~ \mathcal { E } ( f )$ , and $\psi _ { f } \in L _ { 0 } ^ { 2 } ( f )$ gives $\psi _ { f } \in \mathcal { U } ( f )$ . Integration by parts gives $\langle s - \bar { s } _ { f } , \psi _ { f } \rangle = 1$ , so by the Cauchy–Schwarz inequality $\| \dot { \Pi } _ { \mathcal { U } } ( s - \bar { s } _ { f } ) \| ^ { 2 } \geq 1 / J ( f )$ and $S ( f ) \le 1 - 1 / ( \sigma _ { f } ^ { 2 } J ( f ) )$ □

By the Cramér–Rao inequality $\sigma _ { f } ^ { 2 } J ( f ) \geq 1$ , with equality only for the Gaussian law.

Proposition 6 (Positivity). $I f \chi _ { f }$ has an admissible zero ω of finite multiplicity m, then $S ( f ) > 0$

Proof. By Lemma $6 , h = s ^ { m - 1 } e ^ { i \omega s } \in \mathcal { E } ( f )$ , and

$$
\begin{array} { c } { \langle s - \bar { s } _ { f } , h \rangle _ { f } = \mathbb { E } [ s ^ { m } e ^ { - i \bar { \omega } s } ] - \bar { s } _ { f } \mathbb { E } [ s ^ { m - 1 } e ^ { - i \bar { \omega } s } ] } \\ { = ( - i ) ^ { m } \chi _ { f } ^ { ( m ) } ( - \bar { \omega } ) = i ^ { m } \overline { { \chi _ { f } ^ { ( m ) } ( \omega ) } } \neq 0 ; } \end{array}
$$

the second term vanishes because $\chi _ { \scriptscriptstyle f } ^ { ( m - 1 ) } ( - \bar { \omega } ) = ( - 1 ) ^ { m - 1 } \chi _ { \scriptscriptstyle f } ^ { ( m - 1 ) } ( \omega ) = 0 .$ . Hence $\Pi _ { \mathcal { E } ( f ) } \big ( s - \bar { s } _ { f } \big ) \neq 0$ . The mth derivative exists: the Cauchy–Schwarz inequality makes $| s | ^ { m } | e ^ { i \omega s } | = | s | | s ^ { m - 1 } e ^ { i \omega s } |$ integrable. □

Proposition 7 (Gaussian and Laplace laws). $\mathcal { E } ( f ) = \{ 0 \}$ , hence $S ( f ) = 0 ,$ , for the Gaussian and the Laplace law.

Proof. For a Gaussian density and $h \in { \mathcal { E } } ( \phi )$ , the Weierstrass transform $\begin{array} { r } { K ( d ) = \int h ( s ) \phi ( s - d ) } \end{array}$ ds vanishes identically. This transform is injective on $L ^ { 2 } ( \phi )$ by uniqueness of the two-sided Laplace transform of $h \phi$ . Hence $h = 0$ . For a Laplace density $f = ( 2 b ) ^ { - 1 } e ^ { - | { \boldsymbol { s } } | / b } , ( 1 - b ^ { 2 } \partial ^ { 2 } ) f = \delta$ in the sense of distributions, so $h \in { \mathcal { E } } ( f )$ gives $h \ =$ $( 1 - \mathsf { \bar { b } } ^ { 2 } \partial _ { d } ^ { 2 } ) ( h \star \bar { f } ) \stackrel { \cdot } { = } 0$ . Here the distributional convolution identity follows by testing against compactly supported smooth functions and using $f ( s - d ) / f ( s ) \leq e ^ { | d | / b }$ to justify the exchange of integrals. □

More generally, $S ( f ) = 0$ if and only if $s - \bar { s } _ { f }$ is orthogonal to the full annihilator space.

## C.4 Closed form for Gram–Charlier laws

For the strictly positive polynomial perturbations below, the roots of a polynomial determine the finite-dimensional space $\mathcal { E } ( f )$ . A Gram matrix then computes the projection defining $S ( f )$

Theorem 3. Let $f = \phi g ,$ , where $\begin{array} { r } { g = \sum _ { k = 0 } ^ { M } g _ { k } \mathrm { H e } _ { k } , g _ { 0 } = 1 , g _ { M } \neq 0 , } \end{array}$ , and $g \geq c > 0 .$ . Define $\begin{array} { r } { G ( z ) = \sum _ { k } g _ { k } z ^ { k } } \end{array}$ . Then $\chi _ { f } ( \omega ) = e ^ { - \omega ^ { 2 } / 2 } G ( i \omega )$ and $\mathbb { E } _ { f } e ^ { z s } = M _ { f } ( z ) : = e ^ { z ^ { 2 } / 2 } G ( z )$ . If G has the roots $\lambda _ { \varrho }$ with multiplicities $m _ { \varrho } ,$ then $\mathcal { E } ( f ) =$ span $\{ s ^ { j } e ^ { \lambda _ { \theta } s } : 0 \leq j < m _ { \varrho } \}$ , of dimension M. For simple roots, $S ( f ) = { b ^ { H } \Gamma ^ { - 1 } b } / { \sigma _ { f } ^ { 2 } }$ with $\Gamma _ { i j } = M _ { f } ( \lambda _ { j } + \bar { \lambda } _ { i } )$ and $b _ { i } = M _ { f } ^ { \prime } ( \bar { \lambda } _ { i } )$

Proof. Since $g \ge c , L ^ { 2 } ( f ) \subset L ^ { 2 } ( \phi )$ . For $\begin{array} { r } { h \in L ^ { 2 } ( f ) , K ( d ) = \int h ( s ) \phi ( s - d ) } \end{array}$ ds is entire, because $\begin{array} { r } { \int | h | | \phi ( \cdot - d ) | \leq } \end{array}$ $\begin{array} { r } { \| h \| _ { L ^ { 2 } ( \phi ) } ( \int | \phi ( s - d ) | ^ { 2 } / \phi ( s ) d s ) ^ { 1 / 2 } < \infty } \end{array}$ for every d $\in \mathbb { C }$ . Rodrigues’ formula gives He<sub>k</sub> $( s - d ) \phi ( s - d ) = \partial _ { d } ^ { k } \phi ( s - d )$ hence $\begin{array} { r } { \int h ( s ) f ( s - d ) d s = G ( D ) K ( d ) } \end{array}$ . Therefore $h \in { \mathcal { E } } ( f )$ if and only if $G ( D ) K \equiv 0$ . The kernel of $G ( D )$ consists of the exponential polynomials $\textstyle \sum _ { \varrho } P _ { \varrho } ( d ) e ^ { \lambda _ { \varrho } d }$ with deg $P _ { \varrho } < m _ { \varrho }$ . The Weierstrass transform is injective on $L ^ { 2 } ( \phi )$ It maps $s ^ { j } e ^ { \lambda s } \mathrm { t o } e ^ { \lambda d + \lambda ^ { 2 } / 2 }$ times a monic polynomial of degree j in d. Thus the preimage of the kernel is the stated span, which lies in $L ^ { 2 } ( f )$ . Finally $\langle e ^ { \lambda _ { j } s } , e ^ { \lambda _ { i } s } \rangle _ { f } = \mathbb { E } e ^ { ( \lambda _ { j } + \bar { \lambda } _ { i } ) s }$ and $\langle s , e ^ { \lambda _ { i } s } \rangle _ { f } = \mathbb { E } [ s e ^ { \bar { \lambda } _ { i } s } ]$ , and the projection of $s - \bar { s } _ { f }$ equals that of s because constants are orthogonal to $\mathcal { E } ( f )$

## C.5 A 10 dB example

For identical unit-variance source laws with $J = J ( f ) > 1$ , the fraction of information retained is $\tilde { I } / I = S ( f ) / ( 1 -$ $1 / J ^ { 2 } )$ . Take $Y = m B + Z$ , where B is an equiprobable sign and $Z \sim \mathcal { N } ( 0 , 1 )$ is independent. Set $m ^ { 2 } = 1 0$ To evaluate $^ { J , }$ standardize $Y$ to unit variance; scale invariance allows us to compute $S ( f )$ in the original units. The source $\operatorname { C F } \operatorname { i s } e ^ { - \omega ^ { 2 } / 2 } \cos ( m \omega )$ , with zeros $\omega _ { k } = ( \pi / 2 + k \pi ) / m$ . Projection onto $\{ \sin ( \omega _ { k } Y ) : 0 \leq k < 6 4 \}$ gives a lower bound on $S ( f )$ , since these functions belong to the full translate-annihilator space. Evaluating the sine Gram matrix from the CF and the location information by quadrature gives $S ( f ) > 0 . 9 0 4 7$ and $\tilde { I } / I > 0 . 9 1 2 6$ . The resulting information bound concerns the rare-jump experiment; attaining it with unknown source laws and an unknown second direction remains open.

## D Local variance from one complex-frequency moment

Empirical Fourier and moment-generating transforms provide a common estimation framework (Feuerverger and Mc Dunnough, 1984). We use one complex-frequency moment to relate sensitivity to local asymptotic variance. First take $n $ ∞ at fixed $q ;$ then study how the resulting variance coefficient grows with q.

Fix $0 < a \le 4$ in the Gram–Charlier rotation model of Theorem 1. The target column is $a _ { 0 } = \sqrt { 2 / q } u ( \theta )$ . The source $\operatorname { C F } \chi ( t ) = e ^ { - t ^ { 2 } / 2 } ( 1 + a t ^ { 4 } / 2 4 )$ has the simple zero $t _ { 0 } = ( 2 4 / a ) ^ { 1 / 4 } e ^ { i \pi / 4 }$ . For $z \in \mathbb { C }$ , consider the sample mean of

$$
\begin{array} { l } { { Z ( \theta ^ { \prime } ) = e ^ { i \omega ( { \theta ^ { \prime } } ) ^ { T } X } , } } \\ { { \omega ( { \theta ^ { \prime } } ) = \sqrt { q / 2 } [ t _ { 0 } u ( { \theta ^ { \prime } } ) + z u ^ { \perp } ( { \theta ^ { \prime } } ) ] . } } \end{array}
$$

Expectations are taken under the fixed true law $P _ { \theta }$ , while the candidate frequency varies with $\theta ^ { \prime }$ . At the true angle, the target source factor gives $\mathbb { E } Z ( \theta ) = 0$ . The source exponential moments ensure integrability at these complex frequencies.

To apply the product formula, write the frequency seen by each source as $t ( \varphi ) = t _ { 0 } \cos \varphi + z \sin \varphi$ and $t _ { j } = t ( j \pi / q )$ The index $j = 0$ denotes the target; the other indices denote the remaining equally spaced columns. For fixed $z \neq 0$

assume $\chi ( t ( \varphi ) ) \neq 0$ for $0 < \varphi < \pi$ . Then

$$
D _ { q } = \partial _ { \theta ^ { \prime } } \mathbb { E } Z ( \theta ^ { \prime } ) | _ { \theta ^ { \prime } = \theta } = - z \chi ^ { \prime } ( t _ { 0 } ) \prod _ { j = 1 } ^ { q - 1 } \chi ( t _ { j } ) \neq 0 .
$$

For example, $z = \pm i t _ { 0 }$ is excluded because $t ( \pi / 2 )$ is another zero.

For fixed $q ,$ choose a small closed interval around θ on which it is the unique population root; $D _ { q } \neq 0$ guarantees that such an interval exists. Minimize the squared modulus of the sample mean there. The exponential tails give uniform laws of large numbers for the moment and its first two derivatives on this interval. Because the angle is real, the leastsquares condition projects the complex sample fluctuation onto the derivative direction. Expanding this condition to first order yields ${ \sqrt { n } } ( { \hat { \theta } } - \theta ) \Rightarrow { \mathcal { N } } ( 0 , V _ { q } )$ with

$$
V _ { q } = { \frac { \mathbb { E } | Z | ^ { 2 } + \mathrm { R e } ( e ^ { - 2 i \arg D _ { q } } \mathbb { E } Z ^ { 2 } ) } { 2 | D _ { q } | ^ { 2 } } } , \qquad Z = Z ( \theta ) .
$$

This expression uses both real and imaginary parts with equal weight. Write $M ( u ) = e ^ { u ^ { 2 } / 2 } ( 1 + a u ^ { 4 } / 2 4 )$ . Independence and symmetry give $\begin{array} { r } { \mathbb E | Z | ^ { 2 } = et { } { ' } \prod _ { j } \mathbf { \bar { \phi } } _ { M } ( 2 \operatorname { I m } t _ { j } ) } \end{array}$ and $\begin{array} { r } { \mathbb { E } Z ^ { 2 } = \prod _ { j } \chi ( 2 t _ { j } ) } \end{array}$ . Each factor satisfies $| \chi ( 2 t _ { j } ) | \leq$ $M ( 2 \operatorname { I m } t _ { j } )$ . The target factor gives the uniform estimate

$$
{ \frac { | \mathbb { E } Z ^ { 2 } | } { \mathbb { E } | Z | ^ { 2 } } } \leq { \frac { | \chi ( 2 t _ { 0 } ) | } { M ( 2 \operatorname { I m } t _ { 0 } ) } } = 3 e ^ { - { \sqrt { 2 4 / a } } } \leq 3 e ^ { - { \sqrt { 6 } } } < 1 .
$$

Thus the second complex moment cannot cancel the positive term $\mathbb { E } | Z | ^ { 2 }$ in the numerator: $V _ { q }$ is bounded above and below by fixed positive multiples of $\mathbb { E } | Z | ^ { 2 } / | D _ { q } | ^ { 2 }$

To determine the growth of the variance coefficient, we collect each nontarget source’s contribution to the logarithm of this ratio. Let

$$
\begin{array} { l } { { F _ { a } ( t ) = | t | ^ { 2 } + \log \left( 1 + \frac { 2 a } { 3 } ( \operatorname { I m } t ) ^ { 4 } \right) - 2 \log \left| 1 + \frac { a } { 2 4 } t ^ { 4 } \right| , } } \\ { { \displaystyle \gamma ( z ) = \pi ^ { - 1 } \int _ { 0 } ^ { \pi } F _ { a } ( t ( \varphi ) ) d \varphi . } } \end{array}
$$

Since $F _ { a } ( t ) = \log M ( 2 \mathrm { I m } t ) - 2 \log | \chi ( t ) |$ , log $\begin{array} { r } { V _ { q } = \sum _ { j = 1 } ^ { q - 1 } F _ { a } ( t _ { j } ) + O _ { a , z } ( 1 ) } \end{array}$ . To approximate this sum by an integral, first remove the singularities at the endpoints. The zeros at ±t are simple, with $t ^ { \prime } ( 0 ) = z$ and $t ^ { \prime } ( \pi ) = - z$ Hence $F _ { a } ( t ( \varphi ) ) + 2$ log sin $\varphi$ extends smoothly to $[ 0 , \pi ]$ . For this smooth function, the uniform-grid sum differs from q times the integral average by $O ( 1 )$ . We account for the subtracted term exactly: $\begin{array} { r } { \prod _ { j = 1 } ^ { q - 1 } \sin ( j \pi / q ) = q / 2 ^ { q - 1 } } \end{array}$ and $\textstyle \int _ { 0 } ^ { \pi }$ log sin $\varphi d \varphi = - \pi \log 2$ . It follows that

$$
\log V _ { q } = q \gamma ( z ) - 2 \log q + O _ { a , z } ( 1 ) .
$$

In the fixed-q normal approximation, a standard error of $\pi / ( 4 q )$ corresponds to $n = V _ { q } ( 4 q / \pi ) ^ { 2 }$ . A growing-q samplesize guarantee would require bounds uniform in both n and q.

Choosing the frequency and combining several moments remain open problems beyond this calculation on a fixed local interval.