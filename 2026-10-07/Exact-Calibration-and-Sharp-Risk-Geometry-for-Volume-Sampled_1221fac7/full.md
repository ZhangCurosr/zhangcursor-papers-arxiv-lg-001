# Exact Calibration and Sharp Risk Geometry for Volume-Sampled Ridge Regression

Kihun Rhee rheekh00@snu.ac.kr

October 2026

## Abstract

We study ridge regression from exactly s distinct rows of a fixed design. Responses are fixed, and only the subset is random. The determinant law and selected ridge fit share one positive definite penalty. Established mean identities and exponential-family duality give the unique penalty that matches a prescribed full-data ridge fit in expectation. It exists exactly when s exceeds the target’s efective dimension.

Our main result concerns centered covariance risk normalized by full-data penalized loss. For balanced signed coordinate replicas, a strict sector inequality gives the sharp risk and all maximizing responses at every budget from the dimension to one below the row count. This holds for any nonzero positive semidefinite query. With the target and query fixed, the maximizing response space is unchanged across these budgets. For general designs, we characterize attainment of a leave-one-out envelope. For existing real equiangular tight frames, flat row query energy characterizes when every nonzero residual response maximizes at two deletions. At three deletions, we give the sharp risk and complete maximizing space for isotropic queries, using unequal triangle weights.

The balanced geometry yields a same-sample unbiased ridge–Horvitz–Thompson mixture with lower sharp risk and an exact mean-share improvement boundary. Under full recalibration after feature changes, we prove quadratic regret from searching the complete old maximizing space and a query-uniform bound on the mixture’s risk gain. The strongest sector inequalities have exact computer-assisted proofs.

## 1 Introduction

A sampled regression fit may match a full-data fit in expectation and still have substantial variance. Two questions therefore arise. Can an exact row budget preserve a prescribed ridge target? If so, which responses make the remaining covariance risk largest?

We study these questions for a fixed design and fixed responses. We sample distinct original rows under a determinant law. The sampling law and selected ridge fit share one positive definite penalty. The full-data target, which we call Full, has its own prescribed penalty. We choose the sampled penalty from features alone so that the selected fit has mean Full for every response. We call this choice calibration.

We first derive calibration from established selected-inverse identities and exponential-family duality. This gives the basis for our covariance analysis. Calibration exists exactly when the row budget exceeds the efective dimension of the target. When it exists, the penalty is unique. The result applies to any positive definite target. It also allows feasible budgets below the feature dimension. Thus a fixed number of observed labels can preserve a target even when individual sampled designs are rank deficient.

Calibration makes the next question precise. For any fixed query inputs, expected squared-loss excess over Full is a covariance trace. We normalize it by Full’s penalized loss and maximize over nonzero responses. Our goal is both the sharp value and the complete set of responses that attain it. These are diferent tasks from bounding a coeficient covariance matrix or locating its zero-variance responses.

A balanced replica design has the same number of signed copies of each coordinate vector. After undoing the row signs, each response group splits into its mean and zero-sum contrasts. Our main technical result is a strict inequality between the contrast and mean sectors. It holds at every budget from the dimension to one below the row count. The strict coeficient inequality forces every worst-case response to have zero group means. The resulting deficit gives the sharp risk and every maximizing response for each nonzero positive semidefinite query, including all ties. The maximizing space in the original response coordinates stays fixed as the budget changes, with the target and query held fixed. The proof of the strict inequality is exact and computer-assisted. A separate quantitative bound describes the minimum-budget, small-penalty corner. An explicit lower-budget phase change shows why the budget condition matters.

Two complementary regimes extend the geometric picture. For general designs, leave-one-out sampling gives an explicit risk envelope. A row-dependence condition determines exactly when it is attained. For an existing real equiangular tight frame (ETF), flat query energy means that the query assigns the same squared length to every frame row. Residual responses are orthogonal to the columns of the design. At two deletions, flat query energy is necessary and suficient for every nonzero residual response to maximize risk. For isotropic queries at three deletions, we find the sharp value and complete residual maximizing space. The proof uses the actual unequal triangle weights. These are separate exact classifications with distinct assumptions. Table 1 records their scopes.

Table 1: Exact response geometry under the calibrated law. The target and physical query are fixed while the response varies.
<table><tr><td>Design and budget</td><td>Target and query</td><td>Conclusion</td></tr><tr><td>General full rank;  $s = m - 1$ </td><td>Any SPD target; any nonzero PSD query</td><td>Envelope and exact attainment test. The sharp value is smaller when the test fails.</td></tr><tr><td>Balanced replicas; d ≤ s &lt; nd</td><td>Positive scalar target; any nonzero PSD query</td><td>Sharp value and complete maximizing space at every stated budget.</td></tr><tr><td>Existing real ETF; s = m − 2</td><td>Positive scalar target; any nonzero PSD query</td><td>Flat row energy iff every residual maximizes. Exact value and space under flatness.</td></tr><tr><td>Existing real ETF; s = m − 3</td><td>Positive scalar target; isotropic query; d, m − d ≥ 3</td><td>Sharp value and complete residual maximizing space under unequal triangle weights.</td></tr></table>

The geometry has two further uses. First, it guides estimator choice on the same sampled rows. We mix calibrated ridge with a Horvitz–Thompson fit. One feature-selected interval of positive weights improves the sharp risk for every nonzero positive semidefinite query at a balanced base. The maximizing space remains unchanged there. This does not contradict calibration uniqueness: the mixture lies outside the matched ridge family. We also solve the response optimization under a lower bound on the fraction of Full loss carried by group means. The resulting phase boundary states when all-query improvement is possible within the chosen weight interval, which keeps the contrast gap positive. An exact adverse query gives the corresponding failure mechanism.

Second, the full maximizing space controls what happens when features change. We recalibrate the penalty and update the law, selected fits and Full loss. Reoptimizing over the complete old maximizing space has quadratic restriction regret near a balanced base. Keeping only one old maximizing vector can incur linear regret. Enlarging the space to all base contrasts gives a radius that is uniform over normalized positive semidefinite queries. A separate argument preserves the mixture’s gain with error proportional to both the feature change and the mixture weight. This conclusion compares two moving sharp optima. It does not require their maximizing responses to coincide.

Mathematical contributions. Calibration fixes the sampling procedure for the covariance question. It follows from established mean identities and duality after a spectral reduction. The main technical work proves strict sector inequalities and resolves response equality in the stated balanced, LOO and ETF regimes. The mixture and mean-share results turn this geometry into an estimator comparison. The perturbation results then control the fully recalibrated procedure. The mean-map, Rayleigh-quotient and generic spectral tools are standard. The required covariance operators, sign arguments and coupled estimates are specific to the sampling procedure studied here.

Foundations and related work. The sampling law is a fixed-size determinantal point process (Kulesza and Taskar, 2012). Regularized determinant designs are established (Derezi´nski et al., 2020b). A selectedinverse identity gives the fixed-penalty ridge mean (Schreurs et al., 2020). We invert that mean at a hard row budget using finite exponential-family duality (Wainwright and Jordan, 2008). Exact ridge debiasing also exists for surrogate sketches with random row count (Derezi´nski et al., 2020a). Here the number of distinct original rows is exact, and the same penalty enters sampling and fitting. Fixed-size DPPs with prescribed row inclusions also have a survey-sampling precedent (Loonis and Mary, 2019). Our calibration instead matches the full ridge response operator.

Volume sampling has exact fixed-response OLS identities (Derezi´nski and Warmuth, 2018). Normalized worst-response OLS design is studied by Derezi´nski et al. (2019), and OLS covariance contact by Rhee (2026). Our denominator is the positive penalized Full ridge loss. Our covariance is centered over the subset draw with deterministic responses. Classical frame erasure results provide the pair and triangle geometry (Goyal et al., 2001; Bodmann and Paulsen, 2005). They concern diferent losses from the centered ridge risk studied here. Likewise, Horvitz–Thompson weighting (Horvitz and Thompson, 1952), estimator averaging (Lavancier and Rochet, 2016), and spectral perturbation (Yu et al., 2015) provide foundations for our consequences. Appendix I makes these source relations explicit.

Reading order. Section 2 establishes the matched sampling procedure. Section 3 defines normalized covariance risk, compares feasible interior budgets, and gives the general leave-one-out equality test. Section 4 proves the all-budget balanced classification. Section 5 treats two deletion budgets on existing real ETFs. Sections 6 and 7 develop the estimator and perturbation consequences. The appendices contain exact sign certificates, explicit bounds, finite floating-point evaluations, and source comparisons.

## 2 Exact target calibration

## 2.1 Fixed target and fixed row budget

Let $\boldsymbol { X } \in \mathbb { R } ^ { m \times d }$ have full column rank, $d \geq 1$ , with rows $x _ { i } ^ { \top }$ , and put $G = X ^ { \top } X$ . The response $y \in \mathbb { R } ^ { m }$ is fixed. Fix a positive definite target penalty $A _ { 0 }$ and define

$$
w _ { 0 } = ( G + A _ { 0 } ) ^ { - 1 } X ^ { \top } y ,\tag{2.1}
$$

$$
\begin{array} { r } { J _ { 0 } ( X , y ) = \underset { w } { \operatorname* { m i n } } \{ \| y - X w \| ^ { 2 } + w ^ { \top } A _ { 0 } w \} = y ^ { \top } K _ { X } y , } \end{array}\tag{2.2}
$$

$$
K _ { X } = ( I + X A _ { 0 } ^ { - 1 } X ^ { \top } ) ^ { - 1 } .\tag{2.3}
$$

Thus $J _ { 0 } ( X , y ) > 0$ for $y \ne 0$ . This denominator is the minimized penalized full-data objective.

For an integer $1 \leq s \leq$ m and a positive definite matrix R, draw exactly s distinct original rows using

$$
\begin{array} { c } { { p _ { s , R } ( S ) = \displaystyle \frac { \operatorname* { d e t } ( X _ { S } ^ { \top } X _ { S } + R ) } { Z _ { s } ( G , R ) } , } } \\ { { w _ { S } = ( X _ { S } ^ { \top } X _ { S } + R ) ^ { - 1 } X _ { S } ^ { \top } y _ { S } . } } \end{array}\tag{2.4}
$$

Write $G _ { S } = X _ { S } ^ { \top } X _ { S }$ . Here $Z _ { s }$ sums the numerator over all size-s subsets. The same R enters the law and the fit. Only S is random. We call R calibrated when $\mathbb { E } w _ { S } = w _ { 0 }$ for every y, with R independent of y.

## 2.2 The inverse calibration theorem

The law in (2.4) is a fixed-size determinantal point process with L-ensemble kernel $I + X R ^ { - 1 } X ^ { \top }$ (Kulesza and Taskar, 2012, Section 5.1). Indeed, its principal determinant on S, multiplied by det R, is det $( X _ { S } ^ { \top } X _ { S } +$ R). We ask which matched penalty recovers the prescribed full-data operator at an exact original-row budget.

Theorem 2.1 (Exact calibration). Let X have full column rank and $A _ { 0 } \succ 0$ . For $1 \leq s \leq m$ , a calibrated $R _ { s } \succ 0$ exists if and only if

$$
s > d _ { \mathrm { e f f } } ( A _ { 0 } ) : = \operatorname { t r } \{ G ( G + A _ { 0 } ) ^ { - 1 } \} .\tag{2.5}
$$

It is unique among all positive definite matrices and depends only on $G , A _ { 0 } , m , s$ . At s = m, $R _ { m } = A _ { 0 }$ For every feasible $s < m$

$$
0 \prec R _ { s } \preceq \frac { s } { m } A _ { 0 } \prec A _ { 0 } .\tag{2.6}
$$

If $d \leq s < m$ , then

$$
\frac { s - d + 1 } { m - d + 1 } A _ { 0 } \preceq R _ { s } \preceq \frac { s } { m } A _ { 0 } .\tag{2.7}
$$

Both inequalities in (2.7) are strict when d $> 1$ $I f A _ { 0 }$ is scalar, $R _ { s }$ commutes with $G .$

The strict threshold can allow $s < d _ { \colon }$ , so sampled designs need not have full column rank. Uniqueness concerns the joint law-and-fit requirement. It is not identification of a determinantal kernel from subset probabilities. Exact ridge debiasing with scaled regularization and an efective-dimension threshold also occurs in random-size surrogate sketches (Derezi´nski et al., 2020a, Theorems 1 and 6). Here the number of distinct, unrescaled original rows is exactly s.

The calibration theorem follows from two established results after a spectral reduction. The forward mean formula follows from the fixed-size selected-inverse identity of Schreurs et al. (2020, Lemma 1 and Corollary 1). The inverse mean map is a direct finite-support instance of Wainwright and Jordan (2008, Proposition 3.2 and Theorem 3.3). The reduction identifies the spectral support and its feasible mean set. The budget threshold, matrix uniqueness and penalty bounds then follow. We give the derivation to make these steps and their assumptions explicit. A change of coeficient coordinates reduces any SPD target to the identity penalty. We retain the matrix form to state the answer in the original coordinates.

All gradients of functions of a symmetric matrix use the trace inner product. We first establish the needed mean formula without assumptions on simultaneous diagonalization.

Lemma 2.2 (Determinant and mean identities). For $R \succ 0$

$$
\sum _ { s = 0 } ^ { m } t ^ { s } Z _ { s } ( G , R ) = ( 1 + t ) ^ { m } \operatorname* { d e t } \left( R + { \frac { t } { 1 + t } } G \right) ,\tag{2.8}
$$

$$
\mathbb { E } _ { s , R } w _ { S } = \{ \nabla _ { G } \log Z _ { s } ( G , R ) \} X ^ { \top } y .\tag{2.9}
$$

Proof. Expand the determinant into distinct rank-one row contributions. A term using k rows belongs to ${ \binom { m - k } { s - k } }$ size-s subsets. Summing in s proves (2.8). For the mean, allow nonsymmetric rankone contributions and replace $x _ { i } x _ { i } ^ { \top }$ by $x _ { i } ( x _ { i } + t y _ { i } v ) ^ { \top }$ . On a selected subset the derivative at zero is det $( G _ { S } + R ) v ^ { \top } ( G _ { S } + R ) ^ { - 1 } X _ { S } ^ { \top } y _ { S }$ . On the total Gram matrix the derivative is $\boldsymbol { v } ^ { \intercal } ( \boldsymbol { \nabla } _ { G } \boldsymbol { Z _ { s } } ) \boldsymbol { X } ^ { \intercal } \boldsymbol { y }$ . Equating these derivatives and dividing by $Z _ { s }$ proves (2.9). □

For $q _ { j } > 0 ;$ , put $\begin{array} { r } { c _ { k } = \binom { m - k } { s - k } } \end{array}$ , with impossible binomial coeficients equal to zero, and define

$$
P _ { s } ( q ) = \sum _ { J \subseteq [ d ] } c _ { | J | } \prod _ { j \in J } q _ { j } , \qquad \rho _ { j } = q _ { j } \partial _ { q _ { j } } \log P _ { s } ( q ) .\tag{2.10}
$$

Under the auxiliary law with mass proportional to $c _ { | J | } \prod _ { j \in J } q _ { j } , \rho _ { j }$ is the probability that $j \in J$ . These are spectral-coordinate means, not original-row inclusion probabilities.

Lemma 2.3 (Finite inverse moment map). Let $1 \leq s \leq$ m and m $\geq d .$ The gradient of $A _ { s } ( z ) =$ log $P _ { s } ( e ^ { z } )$ , with $P _ { s }$ in (2.10), is a bijection onto $\begin{array} { r } { \{ \rho : 0 < \rho _ { j } < 1 , \ \sum _ { j } \rho _ { j } < s \} } \end{array}$

Proof. The distribution on $J \subseteq [ d ]$ with mass proportional to $c _ { \vert J \vert } \exp ( \sum _ { j \in J } z _ { j } )$ has support all zero-one vectors with at most s ones. Its convex hull is ${ \mathcal { C } } _ { s } = \{ \rho : 0 \leq \rho _ { j } \leq 1 , \ { \overline { { \sum _ { j } \rho _ { j } } } } \leq s \}$ . To check the hull, any vertex with two fractional coordinates admits a feasible perturbation; a vertex with one fractional coordinate cannot satisfy an active integral sum constraint and also admits a perturbation. Thus every vertex is a stated zero-one vector.

The gradient of $A _ { s }$ is the mean indicator, and its Hessian is the indicator covariance. A zero-variance linear combination must agree on the empty set and every singleton, so all its coeficients are zero. The Hessian is therefore positive definite. Every finite parameter gives positive probability to every support point, so its mean is interior. Conversely, for an interior $\rho ,$ a ball of radius $\delta > 0$ about $\rho$ lies in $\mathcal { C } _ { s }$ . The support function then gives

$$
A _ { s } ( z ) - \rho ^ { \top } z \geq \delta \| z \| + \operatorname* { m i n } _ { k : c _ { k } > 0 } \log c _ { k } .
$$

This function is coercive and strictly convex. Its unique minimizer solves $\nabla A _ { s } ( z ) = \rho .$ . This also proves injectivity. When a sum constraint is redundant, the displayed strict inequality remains true through $0 < \rho _ { j } < 1$ □

Proof of Theorem 2.1. Use coeficient coordinates $\widetilde { w } = G ^ { 1 / 2 } w$ and design $\tilde { X } = X G ^ { - 1 / 2 }$ , whose Gram matrix is I. Write

$$
C = G ^ { - 1 / 2 } R G ^ { - 1 / 2 } , \qquad D = G ^ { 1 / 2 } ( G + A _ { 0 } ) ^ { - 1 } G ^ { 1 / 2 } , \qquad 0 \to D \to I .
$$

If $C = U \mathrm { d i a g } ( c _ { j } ) U ^ { \top }$ , put $q _ { j } = 1 / c _ { j }$ . The determinant expansion gives $Z _ { s } ( I , C ) = \operatorname* { d e t } ( C ) P _ { s } ( q )$ , and Lemma 2.2 gives the mean operator

$$
\Phi _ { s } ( C ) \widetilde { X } ^ { \top } , \qquad \Phi _ { s } ( C ) = U \mathrm { d i a g } ( \rho _ { j } ( q ) ) U ^ { \top } .
$$

The desired operator is $D \widetilde { X } ^ { \top }$ . Full row rank of $\widetilde { X } ^ { \top }$ makes calibration equivalent to $\Phi _ { s } ( C ) = D$ . Thus every solution $C$ commutes with $D .$ . In a simultaneous eigenbasis, Lemma 2.3 gives exactly one positive eigenvalue vector $q$ if and only if $\mathrm { t r } D < s$ . Equal target eigenvalues force equal q coordinates, since swapping them preserves the unique moment-map solution. There is no freedom to rotate a diferent penalty inside a repeated eigenspace. This proves all-matrix uniqueness and (2.5). Equivalently, the exact identity

$$
\rho _ { i } - \rho _ { j } = \frac { q _ { i } - q _ { j } } { P _ { s } ( q ) } \sum _ { J \subseteq [ d ] \backslash \{ i , j \} } c _ { | J | + 1 } \prod _ { k \in J } q _ { k }
$$

has a strictly positive final factor, whose empty term is $c _ { 1 } > 0$ . Unwhitening gives $R _ { s } = G ^ { 1 / 2 } C G ^ { 1 / 2 }$ . For scalar $A _ { 0 } , D$ is a spectral function of $G ,$ , so $R _ { s }$ commutes with G.

For the bounds, write $\textstyle q _ { J } = \prod _ { k \in J } q _ { k }$ and split $P _ { s } = B _ { j } + q _ { j } A _ { j }$ . Then

$$
\frac { \rho _ { j } } { 1 - \rho _ { j } } = q _ { j } \frac { A _ { j } } { B _ { j } } , \qquad \frac { A _ { j } } { B _ { j } } = \frac { \sum _ { J \mathcal { J } j } c _ { | J | } q _ { J } ( s - | J | ) / ( m - | J | ) } { \sum _ { J \mathcal { J } j } c _ { | J | } q _ { J } } .
$$

Terms with zero coeficient are omitted. $\mathrm { A t } \ s < \ m$ all denominators here are positive; this weighted average is at most $s / m$ . For $s \geq d$ it lies between $( s - d + 1 ) / ( m - d + 1 )$ and $s / m$ . The eigenvalues of the whitened target penalty are $( 1 - \rho _ { j } ) / \rho _ { j }$ , and the ratio of the calibrated to target eigenvalues is $A _ { j } / B _ { j }$ . Congruence proves (2.6) and (2.7). If $d > 1$ and $d \leq s < m .$ , the positive weights at $| J | = 0$ and $| J | = d - 1$ have distinct ratios, so both sandwich inequalities are strict. ${ \mathrm { A t ~ } } s = m$ the fit is deterministic; equality of its operator with the target and full rank give $R _ { m } = A _ { 0 }$ directly. □

## 2.3 Singleton and noncommuting examples

Corollary 2.4 (Two closed forms). $I f s = 1$ is feasible, then

$$
R _ { 1 } = { \frac { 1 - d _ { \mathrm { e f f } } ( A _ { 0 } ) } { m } } ( G + A _ { 0 } ) .\tag{2.11}
$$

For $d = 1$ and $A _ { 0 } = \lambda _ { 0 } > 0$ , every budget has $r _ { s } = s \lambda _ { 0 } / m$

Proof. For $\begin{array} { r } { s = 1 , P _ { 1 } ( q ) = m + \sum _ { j } q _ { j } } \end{array}$ . Hence $q _ { j } = m \rho _ { j } / ( 1 - \sum _ { k } \rho _ { k } )$ . In the whitened coordinates of the theorem, this says $C = ( 1 - \mathrm { t r } D ) D ^ { - 1 } / m$ . Congruence by $G ^ { 1 / 2 }$ proves (2.11). For $d = 1 , P _ { s } ( q ) = c _ { 0 } + c _ { 1 } q$ and $c _ { 1 } / c _ { 0 } = s / m$ . The odds equation in the theorem’s proof gives $r _ { s } = s \lambda _ { 0 } / m$ □

Take $m = 4$ , rows (1, 0), (1, 1), (0, 1), (0, 0), and

$$
G = { \binom { 2 } { 1 } } \ 1  ) , \qquad A _ { 0 } = { \binom { 3 } { 1 } } \ 1 ) .
$$

Then $\begin{array} { r } { ( G + A _ { 0 } ) ^ { - 1 } = \frac { 1 } { 2 6 } \left( \begin{array} { l l } { 6 } & { - 2 } \\ { - 2 } & { 5 } \end{array} \right) } \end{array}$ , so $d _ { \mathrm { e f f } } = 9 / 1 3 < 1$ and $R _ { 1 } = ( G + A _ { 0 } ) / 1 3$ . Thus one sampled row can preserve this two-dimensional target. Also $( G R _ { 1 } - R _ { 1 } G ) _ { 1 2 } = 1 / 1 3 \neq 0 \ :$ . The penalty need not commute with G when the target does not. The simultaneous spectral reduction occurs after Gram whitening.

## 3 General covariance structure and leave-one-out equality

## 3.1 Risk, response operators, and equality

Write $\Sigma _ { s } ( y ) = \operatorname { C o v } _ { S } ( w _ { S } )$ under the calibrated law. Fix a query metric $W \succeq 0$ in the original coeficient coordinates. We call this the physical query and hold it fixed as the response varies. For $W = Q ^ { \top } Q$ and any fixed query response $y _ { Q }$ , calibration gives

$$
\begin{array} { r } { { \mathbb E } \| Q w _ { S } - y _ { Q } \| ^ { 2 } - \| Q w _ { 0 } - y _ { Q } \| ^ { 2 } = \operatorname { t r } ( W \Sigma _ { s } ( y ) ) . } \end{array}\tag{3.1}
$$

Indeed, expand around $Q w _ { 0 } - y _ { Q }$ . The cross term has expectation zero, and the remaining quadratic term is the covariance trace. Our sharp risk is

$$
h _ { s } ( X , W ) = \operatorname* { s u p } _ { y \neq 0 } \frac { \mathrm { t r } ( W \Sigma _ { s } ( y ) ) } { J _ { 0 } ( X , y ) } .\tag{3.2}
$$

We write $h _ { s } ( X )$ or $h _ { s }$ when the fixed query is clear. This supremum introduces no stochastic model for labels.

Each selected fit is linear in y. If $F _ { S } y = w _ { S }$ and $F _ { 0 } = ( G + A _ { 0 } ) ^ { - 1 } X ^ { \top }$ , define the positive semidefinite response matrix

$$
\begin{array} { r } { C _ { s } ( X , W ) = \mathbb { E } ( F _ { S } - F _ { 0 } ) ^ { \top } W ( F _ { S } - F _ { 0 } ) , \qquad \mathrm { t r } ( W \Sigma _ { s } ( y ) ) = y ^ { \top } C _ { s } ( X , W ) y . } \end{array}\tag{3.3}
$$

This is an m-by-m matrix for scalar query risk. The coeficient covariance $\Sigma _ { s } ( y )$ is a diferent, d-by-d matrix. The normalized operator is

$$
M _ { s } ( X , W ) = K _ { X } ^ { - 1 / 2 } C _ { s } ( X , W ) K _ { X } ^ { - 1 / 2 } , \qquad h _ { s } ( X , W ) = \lambda _ { \operatorname* { m a x } } ( M _ { s } ( X , W ) ) .\tag{3.4}
$$

For a proposed bound $\rho ,$ the standard quadratic-form test is $\Delta _ { \rho } = \rho K _ { X } - C _ { s } ( X , W ) \succeq 0$ . A nonzero response attains that bound exactly when it belongs to ker $\Delta _ { \rho }$ . If this kernel is zero, the bound is strict. The results below identify these operators and kernels under stated sampling and design assumptions.

## 3.2 A common energy across feasible budgets

Before resolving equality at one budget, we ask whether zero variance can occur on diferent response spaces at diferent feasible interior budgets. For $d \geq 2$ , a common energy shows that these spaces are the same for total covariance or a fixed positive definite query. The comparison constants need not be sharp.

Lemma 3.1 (Zero total variance). Let $d \ge 2 , 1 \le s < m$ , and $R \succ 0$ . Under any law positive on every size-s subset, the selected ridge fits satisfy

$$
\mathrm { C o v } ( w _ { S } ) = 0 \quad \Longleftrightarrow \quad x _ { i } y _ { i } = 0 \quad f o r e v e r y i .
$$

Thus the zero-variance responses are exactly those supported on zero rows. This statement does not require calibration.

Proof. Zero covariance makes all fits the same vector w. Set $v _ { i } = x _ { i } ( y _ { i } - x _ { i } ^ { \top } w )$ . The normal equations give $\textstyle \sum _ { i \in S } v _ { i } = R u$ for every size-s subset. For any two indices, choose $s - 1$ other indices and exchange the remaining index. This shows that all $v _ { i }$ agree. Their common value lies in every row line. Those lines have zero intersection because the rows span at least two dimensions. Hence $v _ { i } = 0$ and $R w = 0$ , so $w = 0$ and $x _ { i } y _ { i } = 0$ . The converse follows because every selected right-hand side is then zero. □

Define the full-data residual and the common energy by

$$
e _ { 0 } = y - X w _ { 0 } , \qquad b _ { i } = x _ { i } ( e _ { 0 } ) _ { i } , \qquad \bar { b } = \frac { A _ { 0 } w _ { 0 } } { m } , \qquad \mathcal { E } _ { 0 } ( y ) = \sum _ { i } \lVert b _ { i } - \bar { b } \rVert ^ { 2 } .\tag{3.5}
$$

The full normal equation gives $\sum _ { i } b _ { i } = A _ { 0 } w _ { 0 }$ . For $d \geq 2$ , put

$$
L _ { x } ^ { 2 } = \operatorname* { m a x } _ { i } \lVert x _ { i } \rVert ^ { 2 } , \qquad \mu _ { X } = \frac { \mathrm { t r } G - \lambda _ { \mathrm { m a x } } ( G ) } { L _ { x } ^ { 2 } } > 0 .
$$

At each feasible interior budget define

$$
\begin{array} { r l r l r } { a _ { s } = \displaystyle { \frac { s ( m - s ) } { m ( m - 1 ) } } , } & { \quad } & { \kappa _ { s } = \displaystyle { \frac { \operatorname* { d e t } ( G + R _ { s } ) } { \operatorname* { d e t } R _ { s } } } , } & { \quad } & { \Lambda _ { s } = \lambda _ { \mathrm { m a x } } ( G + R _ { s } ) , } \\ { r _ { \mathrm { m i n } , s } = \lambda _ { \mathrm { m i n } } ( R _ { s } ) , } & { \quad } & { \mathcal { D } _ { s } = \displaystyle { \frac { s } { m } I - R _ { s } A _ { 0 } ^ { - 1 } } , } & \\ { \quad } & { \quad } & { \quad } & { \bar { c } _ { s } = \displaystyle { \frac { a _ { s } } { \kappa _ { s } \Lambda _ { s } ^ { 2 } } } , } & { \quad } & { \bar { c } _ { s } = \displaystyle { \frac { \kappa _ { s } } { r _ { \mathrm { m i n } , s } ^ { 2 } } } \left( a _ { s } + \displaystyle { \frac { m ^ { 2 } \| \mathcal { D } _ { s } \| _ { \mathrm { o p } } ^ { 2 } } { \mu _ { X } } } \right) . } \end{array}\tag{3.6}
$$

These symbols are scalar comparison constants, separate from $C _ { s } , M _ { s }$ , and the sharp risk $h _ { s }$

Proposition 3.2 (Common-energy comparison). Let $d \geq 2$ . At every feasible $1 \leq s < m$

$$
\begin{array} { r } { 0 < \underline { { c } } _ { s } \leq \overline { { c } } _ { s } < \infty , \qquad \underline { { c } } _ { s } \mathcal { E } _ { 0 } ( y ) \leq \mathrm { t r } \Sigma _ { s } ( y ) \leq \overline { { c } } _ { s } \mathcal { E } _ { 0 } ( y ) . } \end{array}\tag{3.7}
$$

Consequently, for any two feasible interior budgets $s , t ,$

$$
\frac { \underline { { c } } _ { s } } { \overline { { c } } _ { t } } \mathrm { t r } \Sigma _ { t } ( y ) \leq \mathrm { t r } \Sigma _ { s } ( y ) \leq \frac { \overline { { c } } _ { s } } { \underline { { c } } _ { t } } \mathrm { t r } \Sigma _ { t } ( y ) .\tag{3.8}
$$

For a fixed $W \succ 0$ , the same comparison holds for $\mathrm { t r } ( W \Sigma _ { s } )$ after multiplying the lower constant $b y$ $\lambda _ { \operatorname* { m i n } } ( W ) / \lambda _ { \operatorname* { m a x } } ( W )$ and the upper constant by its reciprocal. The energy ${ \mathcal { E } } _ { 0 }$ vanishes exactly on the zerorow response space.

Proof. Set $\delta _ { S } = w _ { S } - w _ { 0 }$ . The selected normal equation gives

$$
( G _ { S } + R _ { s } ) \delta _ { S } = \sum _ { i \in S } b _ { i } - R _ { s } w _ { 0 } = : t _ { S } .
$$

Let $U _ { s }$ be the uniform law on size-s subsets. Write $\begin{array} { r } { t _ { S } = \sum _ { i \in S } \bigl ( b _ { i } - \bar { b } \bigr ) + [ ( s / m ) A _ { 0 } - R _ { s } ] w _ { 0 } } \end{array}$ . Uniform inclusion probabilities are $s / m$ for one index and $s ( s - 1 ) / [ m ( m - 1 ) ]$ for two distinct indices. Since $\begin{array} { r } { \sum _ { i } ( b _ { i } - \bar { b } ) = 0 } \end{array}$ , expansion yields

$$
\begin{array} { r } { \mathbb { E } _ { U _ { s } } \| t _ { S } \| ^ { 2 } = a _ { s } \mathcal { E } _ { 0 } + \| [ ( s / m ) A _ { 0 } - R _ { s } ] w _ { 0 } \| ^ { 2 } . } \end{array}\tag{3.9}
$$

For a nonzero row, $b _ { i }$ belongs to its row line. Projection onto the orthogonal complement of that line therefore gives

$$
\| b _ { i } - \bar { b } \| ^ { 2 } \geq \frac { \| x _ { i } \| ^ { 2 } \| \bar { b } \| ^ { 2 } - ( x _ { i } ^ { \top } \bar { b } ) ^ { 2 } } { L _ { x } ^ { 2 } } .
$$

For a zero row the right side is zero. Summing and using the largest eigenvalue of $G$ gives

$$
\mathcal { E } _ { 0 } \geq \mu _ { X } \| \bar { b } \| ^ { 2 } , \qquad \| A _ { 0 } w _ { 0 } \| ^ { 2 } \leq \frac { m ^ { 2 } } { \mu _ { X } } \mathcal { E } _ { 0 } .
$$

Thus (3.9) lies between $a _ { s } \mathcal { E } _ { 0 }$ and $( a _ { s } + m ^ { 2 } \Vert \mathcal { D } _ { s } \Vert _ { \mathrm { o p } } ^ { 2 } / \mu _ { X } ) \mathcal { E } _ { 0 }$

The order bounds $R _ { s } \preceq G _ { S } + R _ { s } \preceq G + R _ { s }$ imply

$$
\frac { \| t _ { S } \| } { \Lambda _ { s } } \leq \| \delta _ { S } \| \leq \frac { \| t _ { S } \| } { r _ { \operatorname* { m i n } , s } } .
$$

Every determinant weight lies between det $R _ { s }$ and det $\left( G + R _ { s } \right)$ . Its ratio to the average weight therefore lies between $\kappa _ { s } ^ { - 1 }$ and $\kappa _ { s } ,$ , so $\kappa _ { s } ^ { - 1 } \leq p _ { s , R _ { s } } ( S ) / U _ { s } ( S ) \leq \kappa _ { s }$ . Calibration gives $\mathbb { E } _ { p } \delta _ { S } = 0$ and hence $\operatorname { t r } \Sigma _ { s } =$ $\mathbb { E } _ { p } \Vert \delta _ { S } \Vert ^ { 2 }$ . Combining the bounds proves (3.7). Combining budgets proves (3.8); the extreme eigenvalues of W prove its query version. Lemma 3.1 identifies the common zero space. □

Why singular queries need a separate analysis. Take two copies of each coordinate row $e _ { 1 } , e _ { 2 }$ , fix $A _ { 0 } = ( 5 4 / 1 3 ) I$ , and put $y = ( 1 , 1 , 1 , 1 ) ^ { \top } , u = ( 1 , 1 ) ^ { \top }$ . The calibrated penalties at sizes two and three are

$$
R _ { 2 } = r _ { 2 } I , \quad r _ { 2 } = { \frac { 7 + { \sqrt { 2 8 3 } } } { 1 3 } } , \qquad R _ { 3 } = 3 I .
$$

Here the full target coeficient is $( G + A _ { 0 } ) ^ { - 1 } X ^ { \top } y = ( 1 3 / 8 0 ) X ^ { \top } y .$ To verify calibration, at size two the count patterns $( 2 , 0 ) , ( 0 , 2 ) , ( 1 , 1 )$ have weights $r ( r + 2 ) , r ( r + 2 ) , 4 ( r + 1 ) ^ { 2 }$ . Symmetry within each group gives the all-response mean coeficient

$$
{ \frac { 3 r + 2 } { 6 r ^ { 2 } + 1 2 r + 4 } } .
$$

Equating it to $1 3 / 8 0$ gives $3 9 r ^ { 2 } - 4 2 r - 5 4 = 0 .$ , with positive root $r _ { 2 }$ . At size three, each omission has equal probability, and the mean coeficient is $[ 1 / ( r + 1 ) + 2 / ( r + 2 ) ] / 4$ , which equals $1 3 / 8 0$ at $r = 3 .$ . For

size three, $u ^ { \top } w _ { S } = 1 / 4 + 2 / 5 = 1 3 / 2 0$ deterministically. At size two it takes the distinct values $2 / ( 2 + r _ { 2 } )$ and $2 / ( 1 + r _ { 2 } )$ , each with positive probability. Thus

$$
u ^ { \top } \Sigma _ { 3 } ( y ) u = 0 , \qquad u ^ { \top } \Sigma _ { 2 } ( y ) u > 0 .
$$

No positive lower comparison from size two to size three can hold for the singular query $W = u \boldsymbol { u } ^ { \intercal }$ Calibration alone does not provide response-uniform two-sided Loewner comparisons.

## 3.3 General-design leave-one-out geometry

Let $m > d \geq 1$ and $s = m - 1$ , with arbitrary $A _ { 0 } \succ 0$ and its calibrated $R \succ 0$ . This budget is feasible because $d _ { \mathrm { e f f } } ( A _ { 0 } ) < d \leq m - 1$ . Write $B = ( G + R ) ^ { - 1 } , L = I - X B X ^ { \top }$ , and abbreviate $K _ { X }$ by K. For a fixed nonzero $W \succeq 0$ , define

$$
\begin{array} { l l } { \ell _ { i } = x _ { i } ^ { \top } B x _ { i } , \qquad v _ { i } = x _ { i } ^ { \top } B W B x _ { i } , \qquad D = \displaystyle \sum _ { i } ( 1 - \ell _ { i } ) , } \\ { \omega _ { i } = \displaystyle \frac { v _ { i } } { D ( 1 - \ell _ { i } ) } , \qquad \rho = \operatorname* { m a x } _ { i } \omega _ { i } , \qquad M = \{ i : \omega _ { i } = \rho \} . } \end{array}\tag{3.10}
$$

These scores combine query energy with the actual omission probabilities. A residual response satisfies $X ^ { \top } y = 0 ;$ for such a response, $w _ { 0 } = 0$ and $J _ { 0 } ( y ) = \| y \| ^ { 2 }$

Theorem 3.3 (LOO envelope and attainment). Under the preceding assumptions, $\rho > 0$ . The risk satisfies $h _ { m - 1 } ( X , W ) \leq \rho$ . The equality kernel at this envelope is

$$
\ker ( \rho K - C _ { m - 1 } ) = \{ y : X ^ { \top } y = 0 , \ y _ { i } = 0 \ f o r \ i \not \in M \} .\tag{3.11}
$$

Thus $h _ { m - 1 } ( X , W ) = \rho$ if and only if the rows $X _ { M }$ are linearly dependent. When this holds, (3.11) is the complete maximizing space, of dimension $| M | - \mathrm { r a n k } ( X _ { M } )$ . If those rows are independent, $h _ { m - 1 } ( X , W ) <$ $\rho .$

Proof. Every deleted normal matrix $\boldsymbol { G } + \boldsymbol { R } - \boldsymbol { x } _ { i } \boldsymbol { x } _ { i } ^ { \top }$ is positive definite. The determinant lemma and the rank-one inverse identity give

$$
0 \leq \ell _ { i } < 1 , \qquad p _ { i } = \frac { 1 - \ell _ { i } } { D } , \qquad w _ { - i } = B X ^ { \top } y - \frac { B x _ { i } ( L y ) _ { i } } { 1 - \ell _ { i } } .\tag{3.12}
$$

Also $D = m - \operatorname { t r } ( G B ) > m - d \geq 1$ . The mean of the deleted correction is $D ^ { - 1 } B X ^ { \top } L y$ . Since $X ^ { \top } L = R B X ^ { \top }$ , all-response calibration and full row rank of $X ^ { \top }$ give

$$
( G + A _ { 0 } ) ^ { - 1 } = B - D ^ { - 1 } B R B , \qquad K - L = D ^ { - 1 } X B R B X ^ { \top } .\tag{3.13}
$$

The second matrix is positive semidefinite with kernel ker $( X ^ { \top } )$ . No commutation is used here.

Taking the correction’s second moment in (3.12) and subtracting its mean square gives the centered response operator

$$
\begin{array} { r } { C _ { m - 1 } = L \mathrm { d i a g } ( \omega _ { i } ) L - D ^ { - 2 } L X B W B X ^ { \top } L . } \end{array}\tag{3.14}
$$

Woodbury gives $L = ( I + X R ^ { - 1 } X ^ { \top } ) ^ { - 1 } , \mathrm { { s o } } \ 0 \prec L \preceq I$ . Spectral calculus for this matrix gives $L - L ^ { 2 } \succeq 0$ Therefore

$$
K - L ^ { 2 } = ( K - L ) + ( L - L ^ { 2 } ) \succeq 0 , \qquad \ker ( K - L ^ { 2 } ) = \ker ( X ^ { \top } ) .\tag{3.15}
$$

Since the vectors $B x _ { i }$ span coeficient space and $W \neq 0 .$ , at least one $v _ { i } = \| W ^ { 1 / 2 } B x _ { i } \| ^ { 2 }$ is positive. Hence $\rho > 0$ . Combining the identities gives the exact deficit

$$
\rho K - C _ { m - 1 } = \rho ( K - L ^ { 2 } ) + L \mathrm { d i a g } ( \rho - \omega _ { i } ) L + D ^ { - 2 } L X B W B X ^ { \top } L .\tag{3.16}
$$

All three terms are positive semidefinite. The first forces $X ^ { \top } y = 0$ at equality. Then $L y = y ,$ the last term vanishes, and the middle term vanishes exactly when y is supported on M. This proves (3.11). Such responses identify with ker $( X _ { M } ^ { \top } )$ , giving the dimension and attainment criterion. If the dimension is zero, the deficit is positive definite. Compactness of $\{ y : y ^ { \top } K y = 1 \}$ then gives $h _ { m - 1 } ( X , W ) < \rho .$ □

The theorem includes tied scores, singular queries, and zero rows. A zero row has score zero and cannot belong to M. When the envelope is unattained, the theorem does not compute the smaller sharp value or classify its maximizers.

Corollary 3.4 (Combined balance and the coloop qualification). If $v _ { i } / ( 1 - \ell _ { i } ) ~ = ~ a$ for every row, then $a > 0 , h _ { m - 1 } ( X , W ) = a / D ,$ and the complete maximizing space is ker $( X ^ { \top } )$ . Conversely, $i f$ every nonzero residual response globally maximizes risk, then $\omega _ { i } = h _ { m - 1 } ( X , W )$ on each non-coloop row. Here a coloop is a row whose deletion lowers column rank. $I f$ there are no coloops, the following conditions are equivalent: constant scores; every nonzero residual maximizes; and the complete maximizing space is ker $( X ^ { \top } )$ ).

Proof. Constant scores make M contain every row, and $m > d$ guarantees a nonzero residual. The theorem proves the forward statement. For the converse write $h = h _ { m - 1 } ( X , W )$ . At any global maximizer $z ,$ Rayleigh stationarity gives $C _ { m - 1 } z = h K z$ . For residual z, $\begin{array} { r } { L z = K z = z , } \end{array}$ and the centered term in (3.14) vanishes. Hence $L \mathrm { d i a g } ( \omega _ { i } ) z = h z$ . Multiplying by $L ^ { - 1 }$ gives $\mathrm { d i a g } ( \omega _ { i } ) z = h z$ . Row i is not a coloop exactly when some residual has $z _ { i } \neq 0$ : a dependence with that nonzero coeficient expresses row i using the others, and conversely. Thus $\omega _ { i } = h$ on every such row. With no coloops this proves constancy and the stated equivalence. □

Attainment on a proper residual subspace. Take two copies of $e _ { 1 }$ , followed by two copies of $e _ { 2 }$ and set

$$
R = \mathrm { d i a g } ( 1 , 2 ) , \qquad A _ { 0 } = \mathrm { d i a g } ( 7 / 5 , 2 0 / 7 ) , \qquad W = I .
$$

Substitution in (3.13) verifies calibration. Here $D = 1 7 / 6$ and $\left( \omega _ { i } \right) = \left( 1 / 1 7 , 1 / 1 7 , 1 / 3 4 , 1 / 3 4 \right)$ . Thus $h _ { 3 } = 1 / 1 7$ and the complete maximizing space is span $\{ ( 1 , - 1 , 0 , 0 ) ^ { \top } \}$ . The scores need not be constant for some residuals to maximize.

An unattained envelope. For $X = ( 1 , 2 , 3 ) ^ { \top } , R = W = 1$ , and $A _ { 0 } = 3 / 2 , ( 3 . 1 3 )$ holds and the scores are $( 1 / 4 3 4 , 4 / 3 4 1 , 3 / 6 2 )$ . The unique maximal row is nonzero, hence independent. Therefore $h _ { 2 } < 3 / 6 2$

Why the coloop condition matters. Take

$$
X = \left( { \begin{array} { l l } { 1 } & { 0 } \\ { 1 } & { 0 } \\ { 0 } & { 1 } \end{array} } \right) , \qquad R = W = I , \qquad A _ { 0 } = \mathrm { d i a g } ( 5 / 3 , 7 / 4 ) .
$$

The scores are $( 1 / 1 1 , 1 / 1 1 , 3 / 1 1 )$ , but (3.14) gives

$$
{ \frac { 1 } { 1 1 } } K - C _ { 2 } = \left( { 7 / 3 6 3 \atop 0 } \begin{array} { c c c } { { 7 / 3 6 3 } } & { { 0 } } & { { 0 } } \\ { { 7 / 3 6 3 } } & { { 7 / 3 6 3 } } & { { 0 } } \\ { { 0 } } & { { 0 } } & { { 1 / 1 2 1 } } \end{array} \right) .
$$

This matrix is positive semidefinite with kernel ker $( X ^ { \top } ) = \operatorname { s p a n } \{ ( 1 , - 1 , 0 ) ^ { \top } \}$ . Every residual therefore maximizes at $1 / 1 1 < \rho$ . The third row is a coloop and escapes the converse’s constraint.

## 3.4 Coeficient-matrix contact is a diferent equality question

The preceding theorem fixes W and varies the response. A coeficient-matrix ceiling instead gives a simultaneous bound for all coeficient query directions at each response.

Proposition 3.5 (Calibrated LOO matrix contact). Under the LOO assumptions above, put $H = G + R$ and $t _ { \ell } = \operatorname* { m a x } _ { i } \ell _ { i } / ( 1 - \ell _ { i } )$ . Then

$$
\Sigma _ { m - 1 } ( y ) \preceq J _ { 0 } ( y ) \frac { t _ { \ell } } { D } H ^ { - 1 } .\tag{3.17}
$$

For $y \ne 0$ , the kernel of the diference in (3.17) is nonzero exactly when $X ^ { \top } y = 0$ and all rows in supp(y) are signed copies of one nonzero vector v with globally maximal calibrated leverage $v ^ { \top } H ^ { - 1 } v$ . In that case the complete kernel in the original coeficient query coordinates is span(v); otherwise it is zero.

Appendix A gives the full positive semidefinite decomposition and the kernel argument. The leverage is evaluated at the calibrated penalty $R ,$ not at the prescribed target $A _ { 0 }$ . For any $W \succeq 0 , v _ { i } \leq \ell _ { i } \operatorname { t r } ( W B )$ so

$$
\rho \leq { \frac { t _ { \ell } } { D } } \operatorname { t r } ( W B ) .
$$

Thus tracing the matrix ceiling can lose the sharp fixed-query value. Its contact kernel is also separate from the zero-total-variance space of Lemma 3.1.

A noncommuting example with unequal omission probabilities. Consider

$$
X = \left( \begin{array} { c c } { { 1 } } & { { 1 } } \\ { { 0 } } & { { 1 } } \\ { { 3 / 5 } } & { { 7 / 5 } } \\ { { - 3 / 5 } } & { { - 7 / 5 } } \end{array} \right) , \qquad R = \left( \begin{array} { c c } { { 1 } } & { { 1 } } \\ { { 1 } } & { { 2 } } \end{array} \right) ,\tag{3.18}
$$

$$
A _ { 0 } = { \frac { 1 } { 1 1 2 5 } } \left( { 1 6 0 7 \quad 1 5 8 3 } \right) , \qquad W = { \frac { 1 } { 5 } } \left( { 2 8 \quad 5 2 } \right) .
$$

The matrices requiring positive definiteness have positive diagonal entries and determinants. They satisfy (3.13). Direct substitution gives

$$
\begin{array} { c } { ( \ell _ { i } ) = ( 4 1 / 1 0 0 , 1 7 / 5 0 , 1 / 4 , 1 / 4 ) , \qquad ( v _ { i } ) = ( 5 9 / 1 0 0 , 3 3 / 5 0 , 3 / 4 , 3 / 4 ) , } \\ { D = 1 1 / 4 , \qquad ( p _ { i } ) = ( 5 9 , 6 6 , 7 5 , 7 5 ) / 2 7 5 . } \end{array}
$$

Every score is $4 / 1 1$ , although neither leverage nor query energy is constant. Hence the sharp value is $4 / 1 1$ and the complete maximizing space is spanned by $( - 3 / 5 , - 4 / 5 , 1 , 0 ) ^ { \top }$ and $( 3 / 5 , 4 / 5 , 0 , 1 ) ^ { \top }$ . There are three distinct row lines in dimension two, so no invertible feature change makes this a coordinate-replica design. The upper-right entries of $G R - R G , G A _ { 0 } - A _ { 0 } G$ , and $R A _ { 0 } - A _ { 0 } R$ are, respectively, $- 3 8 / 2 5$ $- 8 3 6 / 3 7 5$ , and −38/1125. Finally, $t _ { \ell } = 4 1 / 5 9$ and t $\operatorname { r } ( W B ) = 4$ . The traced matrix ceiling is therefore 656/649, strictly larger than the sharp value $4 / 1 1$ under the same law, target, and query.

## 4 Balanced geometry across budgets

## 4.1 Balanced replicas across budgets

Repeated-coordinate designs and countwise ridge calculations have a precedent in Derezi´nski and Warmuth (2018, proof of Theorem 18). Here we use them to study centered covariance over the calibrated subset draw. The key step is the strict sector ordering; the equality space follows from the resulting positive-semidefinite deficit. Suppose now that $m = n d ,$ , with $n , d \geq 2$ , and group $j$ contains n rows $x _ { i } = \varepsilon _ { i } e _ { j }$ , where $\varepsilon _ { i } \in \{ - 1 , 1 \}$ . Fix $A _ { 0 } = \lambda _ { 0 } I , \lambda _ { 0 } > 0$ . Symmetry and Theorem 2.1 give $R _ { s } = r _ { s } I$ for every $d \leq s < m$ . Here $q _ { j }$ denotes a signed group mean. For i in group j, write

$$
\varepsilon _ { i } y _ { i } = q _ { j } + z _ { i } , \qquad \sum _ { i \in j } z _ { i } = 0 ,\tag{4.1}
$$

$$
E _ { j } = \sum _ { i \in j } z _ { i } ^ { 2 } , \qquad b = \frac { n \lambda _ { 0 } } { n + \lambda _ { 0 } } .
$$

Then $w _ { 0 } = n q / ( n + \lambda _ { 0 } )$ and $\begin{array} { r } { J _ { 0 } = \sum _ { i } E _ { j } + b \| q \| ^ { 2 } } \end{array}$

The group counts $K _ { j } = | S \cap j |$ have the global law

$$
\mathbb { P } ( K = k ) \propto \prod _ { j = 1 } ^ { d } { \binom { n } { k _ { j } } } ( r _ { s } + k _ { j } ) , \sum _ { j } k _ { j } = s .\tag{4.2}
$$

There are no fixed group quotas. Set $f ( k ) = k / ( r _ { s } + k )$ and define

$$
\alpha _ { s } = \mathbb { E } \frac { K _ { 1 } ( n - K _ { 1 } ) } { n ( n - 1 ) ( r _ { s } + K _ { 1 } ) ^ { 2 } } ,\tag{4.3}
$$

$$
\beta _ { s } = \frac { \mathbb { E } ( f ( K _ { 1 } ) - f ( K _ { 2 } ) ) ^ { 2 } } { 2 b } ,\tag{4.4}
$$

$$
\tau _ { s } = - \frac { \operatorname { C o v } ( f ( K _ { 1 } ) , f ( K _ { 2 } ) ) } { b } .\tag{4.5}
$$

Conditional on $K .$ , the selected rows are independent uniform subsets within their groups. If $T _ { j }$ is the selected sum of signed contrasts, then

$$
\operatorname { \mathbb { E } } ( T _ { j } \mid K ) = 0 , \qquad \operatorname { V a r } ( T _ { j } \mid K ) = { \frac { K _ { j } ( n - K _ { j } ) } { n ( n - 1 ) } } E _ { j } , \qquad ( w _ { S } ) _ { j } = f ( K _ { j } ) q _ { j } + { \frac { T _ { j } } { r _ { s } + K _ { j } } } .
$$

Total covariance and exchangeability give

$$
\Sigma _ { s } ( y ) = \alpha _ { s } \mathrm { d i a g } ( E _ { j } ) + b \beta _ { s } \mathrm { d i a g } ( q _ { j } ^ { 2 } ) - b \tau _ { s } q q ^ { \top } .\tag{4.6}
$$

The $\alpha _ { \varepsilon }$ term comes from sampling within groups. The $\beta _ { s }$ and $\tau _ { s }$ terms come from random group counts. The next theorem shows that contrasts have a strictly larger risk coeficient than group means at every stated budget.

Theorem 4.1 (Sharp query risk and all maximizers). For the balanced design above, every $d \leq s <$ nd satisfies $\alpha _ { s } > \beta _ { s }$ and $\tau _ { s } > 0$ . For any fixed nonzero $W \succeq 0$ , put $c = \operatorname* { m a x } _ { j } W _ { j j }$ and $I _ { \operatorname* { m a x } } = \{ j : W _ { j j } = c \}$ Then

$$
h _ { s } ( X , W ) = \alpha _ { s } c .\tag{4.7}
$$

The nonzero maximizing responses are exactly the nonzero elements of

$$
\mathcal { E } = \{ y : q = 0 , \ z _ { i } = 0 \ f o r \ i \ i n \ g r o u p s \ o u t s i d e \ I _ { \operatorname* { m a x } } \} .\tag{4.8}
$$

This raw response space has dimension $( n - 1 ) | I _ { \operatorname* { m a x } } |$ and is the same for every stated budget at the fixed target and query.

Proof. The strict signs determine the sharp value and every maximizing response through the exact deficit

$$
\begin{array} { l } { { \alpha _ { s } c J _ { 0 } - \mathrm { t r } ( W \Sigma _ { s } ) = \displaystyle \alpha _ { s } \sum _ { j } ( c - W _ { j j } ) E _ { j } } } \\ { { \mathrm { } ~ + \displaystyle b \sum _ { j } ( \alpha _ { s } c - \beta _ { s } W _ { j j } ) q _ { j } ^ { 2 } } } \\ { { \mathrm { } ~ + b \tau _ { s } q ^ { \top } W q . } } \end{array}\tag{4.9}
$$

Every mean coeficient is strictly positive. Equality therefore forces all group means to vanish and all contrasts to lie in $I _ { \mathrm { m a x } }$ . Conversely those responses attain the bound. Of-diagonal entries of W afect individual risks but neither the sharp value nor this space.

The hard step is the strict sign $\alpha _ { s } > \beta _ { s }$ , including middle budgets. Its proof is computer-assisted and exact. Deletion identities reduce it to a reciprocal moment. A variance bound places an auxiliary mean above a quadratic root. Two reciprocal lower bounds cover diferent parts of the root domain, and finite integer polynomial certificates prove the needed signs and monotonicity on that entire domain. Here are the precise reductions used by that sign proof. For $g ( x ) = ( 1 + x ) ^ { n - 1 } [ r + ( r + n ) x ]$ , put

$$
Z = [ x ^ { s } ] g ( x ) ^ { d } , \quad Q = [ x ^ { s } ] ( 1 + x ) ^ { d ( n - 1 ) + 1 } [ r + ( r + n ) x ] ^ { d - 1 } ,
$$

$$
T = [ x ^ { s - 1 } ] ( 1 + x ) ^ { d ( n - 1 ) } [ r + ( r + n ) x ] ^ { d - 2 } ,
$$

$$
A = [ x ^ { s - 1 } ] g ( x ) ^ { d - 1 } \sum _ { j = 0 } ^ { n - 2 } { \binom { n - 2 } { j } } { \frac { x ^ { j } } { r + 1 + j } } .
$$

Deletion of one or two determinant factors gives $\alpha = A / Z$ and $\beta = [ n T - ( n - 1 ) A ] / [ ( n + r ) Q ]$ . Thus it sufices to prove $A \{ ( n + r ) Q + ( n - 1 ) Z \} > n Z T$ . For $n \geq 3 , d \geq 4$ , select $s - 1$ items with product weights from $d ( n - 1 )$ weight-one items and $d - 2$ items of weight $( r + n ) / r$ . For disjoint ordinary blocks of sizes $n - 2 , n ,$ , with counts $U , V ,$ , coeficient extraction gives $A / T = \mathbb { E } [ ( r + V ) / ( r + 1 + U ) ]$ . Two lower bounds are used: the pointwise integer bound

$$
\frac { 1 } { r + 1 + U } \geq \frac { r + 2 - U } { ( r + 1 ) ( r + 2 ) }
$$

and weighted Cauchy. The variance of the number of unselected exceptional items places its mean above the positive root of an explicit quadratic. At that root, three rational cells cover the full domain. On the two lower cells the integer bound has positive residual; on the upper cell Cauchy does. Exact positivecoeficient expansions also prove the monotonicity needed to pass from the root to the actual mean. Appendix B gives every defining polynomial, the cell maps and their inverses, both finite seams, and the minimum-budget face. It also proves the separate $n = 2$ and $d = 2 , 3$ cases, and the direct LOO endpoint. These cases cover exactly the theorem’s domain. Newton’s coeficient inequality gives strict negative covariance of the increasing count scores, hence $\tau > 0 ;$ its boundary-aware conditional argument is included there. All large coeficient checks use exact integer arithmetic. The preceding deficit then proves the theorem. □

For example, when $n = d = 2 , s = 3 .$ , and $\lambda _ { 0 } = 1 0 / 7$ , calibration gives $r _ { s } = 1$ and $\alpha _ { s } = 1 / 1 6$ . For $W = I ,$ all responses $( a , - a , b , - b )$ maximize the ratio in the positive-copy realization. The sharp value is $1 / 1 6 ,$ , while the isotropic matrix ceiling $\Sigma _ { s } \preceq \alpha _ { s } J _ { 0 } I$ gives the larger trace bound $1 / 8 .$ . At s = nd every covariance is zero and every nonzero response maximizes. The theorem excludes that endpoint and makes no assertion of monotone risk across budgets or of optimality over other estimators.

## 4.2 A quantitative margin at the minimum budget

The sign above holds for every positive penalty. Near zero penalty at $s = d ,$ its size also has a simple bound.

Proposition 4.2 (Minimum-budget margin). In the balanced model with $n , d \geq 2$ and $s = d ,$ , if the calibrated raw penalty satisfies $0 < r _ { d } \leq 1 / ( 4 d ^ { 2 } )$ , then

$$
\alpha _ { d } - \beta _ { d } \geq \frac { 1 } { 4 [ d ( n - 1 ) + 1 ] } .
$$

For fixed $n , d ,$ as the raw penalty $r \downarrow 0$ and the induced target also tends to zero,

$$
\alpha \longrightarrow \frac { 1 } { n } , \qquad \beta \longrightarrow \frac { ( n - 1 ) ( d - 1 ) } { n [ d ( n - 1 ) + 1 ] } , \qquad \frac { \lambda _ { 0 } } { r } \longrightarrow d ( n - 1 ) + 1 .
$$

Proof. Put $M = d ( n - 1 )$ and use the deletion quantities $A , Q , T ,$ Z from the preceding proof at $s = d .$ In the auxiliary experiment, the number H of unselected exceptional items has mass proportional to

$$
{ \binom { d - 2 } { j } } { \binom { M } { j + 1 } } \left( { \frac { r } { r + n } } \right) ^ { j } , \qquad 0 \leq j \leq d - 2 .
$$

The ordinary selected count is $1 + H$ . Write $h = \mathbb { E } H$ , and let $U , V$ count disjoint ordinary blocks of sizes $n - 2 , n$ . Define

$$
\begin{array} { r l } & { Q _ { n } = r n d ( n d - 1 ) + n d ( M + 1 ) + n ^ { 2 } d h , } \\ & { Z _ { n } = r Q _ { n } + n d [ r ( n d - 1 ) + n ( 1 + h ) ] , } \\ & { H _ { n } = ( n + r ) Q _ { n } + ( n - 1 ) Z _ { n } , } \\ & { \quad B = M r + n ( 1 + h ) , \quad \quad C = M r ( d - 1 ) - ( 1 + h ) r ( d - 1 ) . } \end{array}
$$

Adding the deleted factors and taking ordinary factorial moments gives

$$
{ \frac { Q } { T } } = { \frac { Q _ { n } } { d M } } , \quad { \frac { Z } { T } } = { \frac { Z _ { n } } { d M } } , \quad \mathbb { E } ( r + V ) = { \frac { B } { M } } , \quad \mathbb { E } [ ( r + V ) U ] = { \frac { ( n - 2 ) C } { M ( M - 1 ) } } .
$$

These are also the specialization $u = 1$ of (B.7); they remain valid when $n = 2$ or $d = 2$ . The integer reciprocal bound gives

$$
\frac { A } { T } \geq \frac { ( r + 2 ) ( M - 1 ) B - ( n - 2 ) C } { M ( M - 1 ) ( r + 1 ) ( r + 2 ) } .
$$

Consequently, if

$$
P ( h ) = [ ( r + 2 ) ( M - 1 ) B - ( n - 2 ) C ] H _ { n } - n ( r + 1 ) ( r + 2 ) M ( M - 1 ) Z _ { n } ,
$$

then

$$
\alpha - \beta \ge \frac { d P ( h ) } { ( M - 1 ) ( r + 1 ) ( r + 2 ) ( n + r ) Q _ { n } Z _ { n } } .\tag{4.10}
$$

The short expansions in Appendix B.8 show that $P ( h ) = P ( 0 ) + p _ { 1 } h + p _ { 2 } h ^ { 2 }$ , with $p _ { 1 } , p _ { 2 } > 0$ , and

$$
P ( 0 ) \geq d n ^ { 4 } ( M - 1 ) [ 2 - d ^ { 2 } ( 3 r + 4 r ^ { 2 } + r ^ { 3 } ) ] .
$$

On the stated window the subtracted expression is at most $3 / 4 + 1 / 1 6 + 1 / 1 0 2 4 < 1$ . Hence $P ( h ) \geq$ $d n ^ { 4 } ( M - 1 )$ .

The adjacent mass ratio for H gives the drift identity

$$
r ( d - 2 ) ( M - 1 ) = r ( n d - 2 ) h + n h + n \mathbb { E } H ^ { 2 } .
$$

Since $H ^ { 2 } \geq H$ , we obtain $0 \leq h \leq r ( d - 2 ) ( M - 1 ) / ( 2 n ) \leq r d ^ { 2 } / 2 \leq 1 / 8$ . For $Q _ { 0 } = n d ( M + 1 )$ and $Z _ { 0 } = n ^ { 2 } d ,$ it follows that

$$
Q _ { n } / Q _ { 0 } = 1 + \frac { r ( n d - 1 ) + n h } { M + 1 } \leq 1 + 2 r + r ( d - 2 ) / 2 \leq 1 + r d \leq 9 / 8 ,
$$

$$
Z _ { n } / Z _ { 0 } = 1 + h + \frac { r ( n d - 1 ) } { n } + \frac { r ( M + 1 ) } { n } \frac { Q _ { n } } { Q _ { 0 } } \leq 1 + \frac { 1 } { 8 } + \frac { 1 7 } { 8 } r d < \frac { 3 } { 2 } ,
$$

$$
{ \frac { ( r + 1 ) ( r + 2 ) ( n + r ) } { 2 n } } \leq { \frac { 1 7 } { 1 6 } } \left( { \frac { 3 3 } { 3 2 } } \right) ^ { 2 } = { \frac { 1 8 5 1 3 } { 1 6 3 8 4 } } .
$$

Here $( n d - 1 ) / ( M + 1 ) \leq 2 , ( M + 1 ) / n \leq d , r d \leq 1 / 8 .$ , and $r \leq 1 / 1 6$ were used. Substitution into (4.10) yields

$$
\alpha - \beta \ge \frac { 1 } { M + 1 } \frac { 1 / 2 } { ( 1 8 5 1 3 / 1 6 3 8 4 ) ( 9 / 8 ) ( 3 / 2 ) } = \frac { 1 3 1 0 7 2 } { 4 9 9 8 5 1 ( M + 1 ) } > \frac { 1 } { 4 ( M + 1 ) } .
$$

Finally, the coeficient definitions at $r = 0$ give $Z = n ^ { d } , A = n ^ { d - 1 } , Q = n ^ { d - 1 } ( M + 1 )$ , and $T = n ^ { d - 2 } M$ Their positive denominators give the asserted limits, including $\alpha - \beta \to 1 / ( M + 1 )$ . Since $b = n r Q / Z$ and $\lambda _ { 0 } = n b / ( n - b )$ , also $\lambda _ { 0 } / r \to M + 1$ □

The variance normalization b vanishes at this boundary. Setting all count scores to one before normalizing would lose the nonzero limit of $\beta .$ . The proposition gives a finite bound only when the calibrated penalty meets its stated window.

## 4.3 Why the budget restriction matters

Calibration can exist below $d .$ The strict sector order can then reverse. Take two positive copies of each coordinate in dimension three, and sample $s = 2$ . The count patterns are permutations of $( 2 , 0 , 0 )$ and $( 1 , 1 , 0 )$ . Their individual unnormalized masses are respectively $r ^ { 2 } ( r + 2 )$ and $4 r ( r + 1 ) ^ { 2 }$ . Thus, with $A _ { r } = 5 r ^ { 2 } + 1 0 r + 4$

$$
Z = 3 r A _ { r } , \qquad \rho = \mathbb { E } f ( K _ { 1 } ) = \frac { 2 ( 5 r + 4 ) } { 3 A _ { r } } , \qquad \lambda _ { 0 } = \frac { 2 ( 1 - \rho ) } { \rho } = \frac { 1 5 r ^ { 2 } + 2 0 r + 4 } { 5 r + 4 } .
$$

For example, the numerator for $\mathbb { E } f ( K _ { 1 } ) { \mathrm { ~ i s ~ } } 2 r ^ { 2 } + 8 r ( r + 1 )$ : one pattern has $K _ { 1 } = 2$ and two have $K _ { 1 } = 1$ The map $r \mapsto \lambda _ { 0 }$ increases from 1 to infinity, since its derivative is $( 7 5 r ^ { 2 } + 1 2 0 r + 6 0 ) / ( 5 r + 4 ) ^ { 2 } > 0$ . This agrees with the feasibility condition $2 > 6 / ( 2 + \lambda _ { 0 } )$

The same six patterns give

$$
\mathbb { E } f ( K _ { 1 } ) ^ { 2 } = \frac { 4 r ^ { 2 } / ( r + 2 ) + 8 r } { Z } , \qquad \mathbb { E } [ f ( K _ { 1 } ) f ( K _ { 2 } ) ] = \frac { 4 r } { Z } .
$$

Only the two patterns with $K _ { 1 } = 1$ contribute to α. Using $b = 2 ( 1 - \rho )$ in the coeficient definitions therefore gives

$$
\begin{array} { c } { \displaystyle \alpha = \frac { 4 } { 3 A _ { r } } , \qquad \beta = \frac { 4 ( r + 1 ) } { ( r + 2 ) ( 1 5 r ^ { 2 } + 2 0 r + 4 ) } , } \\ { \displaystyle \tau = \frac { 4 ( 5 r ^ { 2 } + 5 r + 2 ) } { 3 A _ { r } ( 1 5 r ^ { 2 } + 2 0 r + 4 ) } , } \\ { \displaystyle \beta - \alpha = - \frac { 4 ( 5 r ^ { 2 } + 2 r - 4 ) } { 3 ( r + 2 ) A _ { r } ( 1 5 r ^ { 2 } + 2 0 r + 4 ) } . } \end{array}
$$

For the physical rank-one query $W = v v ^ { \top } , v = ( 1 , 1 , 1 ) ^ { \top }$ , the contrast coeficient is $\alpha .$ The normalized mean operator is $\beta I - \tau v v ^ { \top }$ , whose largest eigenvalue is $\beta \mathrm { o n } v ^ { \perp }$ . Hence the sharp risk is $\operatorname* { m a x } ( \alpha , \beta )$ . The transition occurs at

$$
r _ { * } = ( \sqrt { 2 1 } - 1 ) / 5 , \qquad \lambda _ { * } = ( 8 + 2 \sqrt { 2 1 } ) / 5 .
$$

For $1 < \lambda _ { 0 } < \lambda _ { * }$ , all maximizers are pure means with $q \perp v$ . At equality, they are all sums of arbitrary contrasts and means with $q \perp v .$ For $\lambda _ { 0 } ~ > ~ \lambda _ { * }$ , precisely the pure contrasts maximize. Indeed the remaining mean eigenvalue is $\beta - 3 \tau < \beta$ . This exact phase change occurs inside the feasible calibrated domain. It does not contradict Theorem 4.1, whose budgets satisfy $s \geq d .$

## 5 Frame geometry at two and three deletions

## 5.1 Two deletions and flat physical queries

At two deletions, we ask which queries make every nonzero residual response worst case on an equiangular tight frame. Assume that X is an existing real unit-row equiangular tight frame (ETF), with $d \geq 2$ and $m \geq d + 2$ . Thus, for $g = m / d$

$$
X ^ { \top } X = g I , \qquad ( x _ { i } ^ { \top } x _ { j } ) ^ { 2 } = \mu ^ { 2 } = { \frac { g - 1 } { m - 1 } } \quad ( i \neq j ) .\tag{5.1}
$$

Fix $A _ { 0 } = \lambda _ { 0 } I , \lambda _ { 0 } > 0$ , and use $s = m - 2$ . Every such target has a unique calibrated penalty rI, also unique among SPD penalties. Under this law, deleting any pair of distinct original rows has the same probability. This uses the classical ETF pair geometry (Bodmann and Paulsen, 2005); it does not assume that ETFs exist for all $( m , d )$

Put $b = ( g + r ) ^ { - 1 } , a = 1 - b , \delta = a ^ { 2 } - b ^ { 2 } \mu ^ { 2 }$ , and $N = { \binom { m } { 2 } }$ . The residual coeficient is

$$
c _ { R } = \frac { b ^ { 2 } \{ ( m - 2 ) ( a ^ { 2 } + b ^ { 2 } \mu ^ { 2 } ) + 2 a b ( g - 2 ) \} } { N \delta ^ { 2 } } .\tag{5.2}
$$

Theorem 5.1 (ETF query criterion). Under these assumptions, fix any nonzero $W \succeq 0$ and put $w =$ $\operatorname { t r } ( W ) / d .$ . The following are equivalent:

1. $x _ { i } ^ { \top } W x _ { i } = w$ for every original row;

2. every nonzero response in ker $( X ^ { \top } )$ globally maximizes risk;

3. the complete maximizing space is ker $( X ^ { \top } )$ .

When they hold, $h _ { m - 2 } ( X , W ) = w c _ { R }$ , and the maximizing space has dimension $m - d .$ . Singular queries are included.

Proof. Calibration and the centered operator. Assume the ETF conditions of Theorem 5.1. Put $T = X X ^ { \top } , P = T / g , P _ { \perp } = I - P , q = \mu ^ { 2 } = ( g - 1 ) / ( m - 1 )$ , and $h = g - 1$ . Here $q$ is the squared ETF coherence, not a group mean. Let $k = m - d \geq 2$ . Then

$$
h ^ { 2 } - q = \frac { k \{ k ( m - 1 ) - d \} } { d ^ { 2 } ( m - 1 ) } > 0 , \qquad k ( m - 1 ) - d = ( k - 1 ) m > 0 .\tag{5.3}
$$

The Gram matrix after deleting two rows has eigenvalues $g$ with multiplicity $d - 2$ , and $h + \mu , h - \mu$ . All are positive.

For a positive raw penalty r, define

$$
b = ( g + r ) ^ { - 1 } , \quad a = 1 - b , \quad \ell = 1 - b g = b r , \quad \delta = a ^ { 2 } - b ^ { 2 } q , \quad N = { \binom { m } { 2 } } , \quad L = I - b T = P _ { \perp } + \ell P .\tag{5.4}
$$

For a deleted pair $D _ { 0 } = \{ i , j \}$ let $E _ { D _ { 0 } }$ select its two coordinates, $X _ { D _ { 0 } } = E _ { D _ { 0 } } X$ , and $t = T _ { i j }$ . Its deletion matrix is

$$
Q _ { D _ { 0 } } = I _ { 2 } - b E _ { D _ { 0 } } T E _ { D _ { 0 } } ^ { \top } = \left( \begin{array} { c c } { a } & { - b t } \\ { - b t } & { a } \end{array} \right) , \qquad Q _ { D _ { 0 } } ^ { - 1 } = \delta ^ { - 1 } \left( \begin{array} { c c } { a } & { b t } \\ { b t } & { a } \end{array} \right) .\tag{5.5}
$$

The deletion matrix $Q _ { D _ { 0 } }$ is positive definite. Every retained determinant is $( g + r ) ^ { d } \delta$ , so the actual law is uniform on the N original pairs. This is the pair geometry associated with ETFs (Bodmann and Paulsen, 2005); it does not use a rotation of the response coordinates. The fit can be written as

$$
\begin{array} { r } { w _ { S } = b X ^ { \top } y - z _ { D _ { 0 } } ( y ) , \qquad z _ { D _ { 0 } } ( y ) = b X _ { D _ { 0 } } ^ { \top } Q _ { D _ { 0 } } ^ { - 1 } E _ { D _ { 0 } } L y . } \end{array}\tag{5.6}
$$

Counting diagonal and of-diagonal pair occurrences $\mathrm { g i }$ ves

$$
\begin{array} { c } { { \displaystyle \sum _ { D _ { 0 } } E _ { D _ { 0 } } ^ { \top } Q _ { D _ { 0 } } ^ { - 1 } E _ { D _ { 0 } } = \delta ^ { - 1 } \{ [ ( m - 1 ) a - b ] I + b T \} , } } \\ { { { } } } \\ { { e = ( m - 1 ) a + b ( g - 1 ) , \qquad \eta = \displaystyle \frac { b \ell e } { N \delta } > 0 , } } \\ { { { } } } \\ { { \mathbb { E } z _ { D _ { 0 } } ( y ) = \eta X ^ { \top } y , \qquad \mathbb { E } w _ { S } = f ( r ) X ^ { \top } y , \quad f ( r ) = b - \eta . } } \end{array}\tag{5.7}
$$

To identify $f ,$ apply this mean identity to $y = X v$ and take traces of the mean fit map. The retained spectrum above gives

$$
f ( r ) = { \frac { 1 } { m } } \left\{ { \frac { ( d - 2 ) g } { g + r } } + { \frac { h + \mu } { h + \mu + r } } + { \frac { h - \mu } { h - \mu + r } } \right\} .\tag{5.8}
$$

It decreases continuously and strictly from $1 / g$ to zero. Hence every prescribed $\lambda _ { 0 } > 0$ has exactly one positive scalar solution of $f ( r ) = 1 / ( g + \lambda _ { 0 } )$ . Theorem 2.1 gives uniqueness among all SPD penalties because $s = m - 2 \geq d$ is feasible. At this calibration,

$$
k _ { F } = 1 - g f = \frac { \lambda _ { 0 } } { g + \lambda _ { 0 } } = \ell + g \eta > 0 , \qquad K = P _ { \perp } + k _ { F } P .\tag{5.9}
$$

Thus this construction reaches a prescribed target, not just a target chosen after the raw penalty.

The operator for any symmetric query. Let $H = X W X ^ { \top } , h _ { i } = H _ { i i }$ , and $D _ { h } = \mathrm { d i a g } ( h _ { i } )$ . Define

$$
\alpha = a ^ { 2 } ( m - 2 ) + 2 a b ( g - 2 ) - 2 b ^ { 2 } q , \quad \beta = b ^ { 2 } q , \quad \gamma = a b , \quad \theta = a ^ { 2 } + b ^ { 2 } q .\tag{5.10}
$$

The coeficient covariance is

$$
\mathrm { C o v } ( w _ { S } ( y ) ) = \frac { 1 } { N } \sum _ { D _ { 0 } } z _ { D _ { 0 } } ( y ) z _ { D _ { 0 } } ( y ) ^ { \top } - \eta ^ { 2 } X ^ { \top } y y ^ { \top } X .\tag{5.11}
$$

For a pair $D _ { 0 } = \{ i , j \}$ , signed multiplication in (5.5) gives

$$
\begin{array} { l } { { ( Q _ { D _ { 0 } } ^ { - 1 } H _ { D _ { 0 } D _ { 0 } } Q _ { D _ { 0 } } ^ { - 1 } ) _ { 1 1 } = \frac { a ^ { 2 } h _ { i } + 2 a b t H _ { i j } + b ^ { 2 } q h _ { j } } { \delta ^ { 2 } } , } } \\ { { ( Q _ { D _ { 0 } } ^ { - 1 } H _ { D _ { 0 } D _ { 0 } } Q _ { D _ { 0 } } ^ { - 1 } ) _ { 1 2 } = \frac { a b t ( h _ { i } + h _ { j } ) + \theta H _ { i j } } { \delta ^ { 2 } } . } } \end{array}\tag{5.12}
$$

The second diagonal entry swaps $i , j$ . Since $T H = H T = g H$ , summing the first line over $j \neq i$ has numerator $[ a ^ { 2 } ( m - 1 ) + 2 a b ( g - 1 ) - b ^ { 2 } q ] h _ { i } + b ^ { 2 } q \mathrm { t r } ( H )$ . Each of-diagonal entry occurs in one pair. Therefore

$$
\begin{array} { c } { { M _ { W } = \displaystyle \frac { \alpha D _ { h } + \beta \mathrm { t r } ( H ) I + \gamma ( T D _ { h } + D _ { h } T ) + \theta H } { N \delta ^ { 2 } } , } } \\ { { C = b ^ { 2 } L M _ { W } L - \eta ^ { 2 } H . } } \end{array}\tag{5.13}
$$

This is the response operator, not the coeficient covariance matrix. The last subtraction centers at the actual mean in (5.7). Before imposing any flatness, its cross block is

$$
P _ { \perp } C P = \frac { b ^ { 2 } \ell ( \alpha + g \gamma ) } { N \delta ^ { 2 } } P _ { \perp } D _ { h } P .\tag{5.14}
$$

Indeed H and T kill residuals, while L acts as the identity on residuals and as ℓ on the fitted space.

Flat-query deficit, strict signs, and converse. If $h _ { i } = w = \mathrm { t r } ( W ) / d$ for every row, then $D _ { h } = w I$ and $\operatorname { t r } ( H ) = m w$ . Define

$$
\begin{array} { c } { { A = \alpha + m \beta = ( m - 2 ) ( a ^ { 2 } + b ^ { 2 } q ) + 2 a b ( g - 2 ) , } } \\ { { { } } } \\ { { c _ { R } = \displaystyle \frac { b ^ { 2 } A } { N \delta ^ { 2 } } , \qquad u = \displaystyle \frac { b ^ { 2 } \ell ^ { 2 } ( A + 2 g a b ) } { N \delta ^ { 2 } } , } } \\ { { { } \tau = \displaystyle \frac { b ^ { 2 } \ell ^ { 2 } ( e ^ { 2 } - N \theta ) } { N ^ { 2 } \delta ^ { 2 } } , \qquad \xi = c _ { R } k _ { F } - u . } } \end{array}\tag{5.15}
$$

Substitute $D _ { h } = w I$ in (5.13) and use $L H = H L = \ell H$ . The result is

$$
C = w c _ { R } { \cal P } _ { \perp } + w u { \cal P } - \tau { \cal H } , \qquad w c _ { R } { \cal K } - C = w \xi { \cal P } + \tau { \cal H } .\tag{5.16}
$$

The coeficient $c _ { R }$ is (5.2). It is also the isotropic residual coeficient: its numerator is $( m - 1 ) [ a ^ { 2 } + b ( 2 a +$ $b ) q ] - [ a ^ { 2 } + 2 a b + b ^ { 2 } q ] = A$ , using $( m - 1 ) q = g - 1$

We now prove all strict signs needed for suficiency and necessity. The assumptions imply

$$
m \ge 4 , \qquad h = g - 1 \ge \frac { 2 } { m - 2 } > 0 , \qquad m h > 2 , \qquad a = b ( h + r ) > b h , \qquad 0 < \ell < 1 .\tag{5.17}
$$

First,

$$
A - 2 a \ell = a [ ( m - 4 ) a + 2 b ( 2 g - 3 ) ] + ( m - 2 ) b ^ { 2 } q > 0 .\tag{5.18}
$$

To check the sign even if $2 g - 3 < 0$ , the expression in square brackets is at least $b [ ( m - 4 ) h + 2 ( 2 g - 3 ) ] =$ $b ( m h - 2 ) > 0$ . Since $k _ { F } = \ell + g \eta > \ell ;$ it follows that

$$
\xi = \frac { b ^ { 2 } \{ A ( k _ { F } - \ell ^ { 2 } ) - 2 g a b \ell ^ { 2 } \} } { N \delta ^ { 2 } } > \frac { b ^ { 3 } g \ell ( A - 2 a \ell ) } { N \delta ^ { 2 } } > 0 .\tag{5.19}
$$

In particular $A > 0$ , so $c _ { R } > 0$

For the second sign put $t _ { 0 } = a / b = h + r > h .$ . Then

$$
\frac { e ^ { 2 } - N \theta } { b ^ { 2 } } = F ( t _ { 0 } ) , \qquad F ( t ) = \frac { ( m - 1 ) ( m - 2 ) } { 2 } t ^ { 2 } + 2 ( m - 1 ) t h + h ^ { 2 } - \frac { m h } { 2 } .\tag{5.20}
$$

This polynomial increases strictly for $t \geq h > 0$ , and $F ( h ) = m h [ ( m + 1 ) h - 1 ] / 2 > 0$ . Hence $\tau > 0$ Finally,

$$
\frac { \alpha + g \gamma } { b ^ { 2 } } = G ( t _ { 0 } ) , \qquad G ( t ) = ( m - 2 ) t ^ { 2 } + ( 3 h - 1 ) t - \frac { 2 h } { m - 1 } .\tag{5.21}
$$

For $t \ge h , G ^ { \prime } ( t ) \ge 2 ( m - 2 ) h + 3 h - 1 \ge 3 + 3 h > 0$ . Also $G ( h ) = h [ ( m + 1 ) h - 1 - 2 / ( m - 1 ) ] > 0 .$ because $( m + 1 ) h \geq 2 ( m + 1 ) / ( m - 2 ) > 2$ and $1 + 2 / ( m - 1 ) \le 5 / 3$ . Thus $\alpha + g \gamma > 0$ and the scalar in (5.14) is strictly positive. These signs hold for all positive raw penalties in the stated domain.

For $W \succeq 0 , W \neq 0$ , we have $H \succeq 0$ and $w > 0$ . The first term in (5.16) has kernel $\ker ( X ^ { \top } )$ and the second term vanishes on that space. Consequently

$$
\ker ( w c _ { R } K - C ) = \ker ( X ^ { \top } ) .\tag{5.22}
$$

There are $m - d > 0$ residual dimensions. They attain $w c _ { R } .$ while every response with a fitted component has strictly smaller normalized risk. The argument does not invert $W .$ , so singular queries are included. For $W = w I$ , writing $c _ { F } = u - g \tau $ gives the isotropic sector gap $c _ { R } k _ { F } - c _ { F } = \xi + g \tau > 0$ . Positivity of covariance also gives $c _ { F } \geq 0$ . Thus no separate sign certificate for that special case is needed here.

For necessity, suppose every nonzero residual is a global maximizer, with sharp value $h _ { \star }$ . Since $K \succ 0$ full Rayleigh stationarity gives $C z = h _ { \star } K z = h _ { \star } z$ on residuals. Hence $P C P _ { \perp } = 0$ , and by symmetry $P _ { \perp } C P = 0$ . The positive coeficient in (5.14) implies $P _ { \perp } D _ { h } P = P D _ { h } P _ { \perp } = 0$ . Thus $D _ { h }$ commutes with

P. Entrywise, $( h _ { i } - h _ { j } ) T _ { i j } / g = 0$ . Every of-diagonal $T _ { i j }$ is nonzero since $q > 0 ,$ so all $h _ { i }$ agree. Their sum is $g \operatorname { t r } ( W ) = m w ;$ therefore each is w. This proves the equivalence in Theorem 5.1. Constancy of a residual compression alone would not prove this converse; the cross block is essential.

The flat symmetric query class has $d ( d + 1 ) / 2 - m$ traceless directions. Indeed the row projectors have positive definite Frobenius Gram matrix $( 1 - q ) I + q \mathbf { 1 } \mathbf { 1 } ^ { \top }$ , so they are independent; their sum is gI (Fickus et al., 2021, Section 5.1). Appendix C.1 gives the full dimension argument and an actual (16, 6) frame with a rank-three nonisotropic flat query. Failure of flatness excludes the claim that every residual maximizes. It does not classify individual residual maximizers or the nonflat sharp value. For the same flat query and prescribed target, single deletion also has constant row scores, so Theorem 3.3 gives the same complete maximizing space. The penalties are separately calibrated.

## 5.2 Three deletions under unequal determinant weights

We now keep isotropic queries and allow one more deletion. Pair deletion was uniform because every pair has the same squared inner product. At three deletions the signed triangle product changes the determinant. The two triangle classes and their incidence counts are classical ETF geometry (Bodmann and Paulsen, 2005, Section 5.2, Lemma 5.19 and Proposition 5.20). Their determinant masses follow from the triple spectra. We retain those unequal masses in the centered covariance calculation. The key step is the strict normalized sector gap that identifies every maximizing response.

The formulas below summarize the first two resolvent moments under this weighted law. For isotropic queries, they give one covariance coeficient for residual responses and one for fitted responses. Under the assumptions below, the theorem proves a strict gap after normalization by the Full loss. Thus the residual space is exactly the maximizing space.

Assume an existing real unit-row ETF with $d \geq 3$ and $k = m - d \geq 3$ . Keep $g = m / d , T = X X ^ { \top }$ $P = T / g$ , and $q = k / [ d ( m - 1 ) ]$ and $\mu = { \sqrt { q } } > 0 .$ . Fix $A _ { 0 } = \lambda _ { 0 } I , \lambda _ { 0 } > 0$ , and $W = w I , w > 0$ . At $s = m - 3$ the same positive scalar penalty rI is used in the determinant law and the selected fit. Define

$$
z = g + r , \quad h = z - 1 , \quad t = \frac { 2 ( g - 1 ) ( g - 2 ) } { ( m - 1 ) ( m - 2 ) } , \quad D = h ^ { 3 } - 3 h q - t ,
$$

$$
H _ { 1 } = \frac { d - 3 } { z } + \frac { 3 ( h ^ { 2 } - q ) } { D } , \qquad H _ { 2 } = \frac { d - 3 } { z ^ { 2 } } + \frac { 3 [ h ( h ^ { 2 } - q ) + 3 t ] } { ( h ^ { 2 } - 4 q ) D } ,\tag{5.23}
$$

$$
M = d - r H _ { 1 } , \quad B = d - 2 r H _ { 1 } + r ^ { 2 } H _ { 2 } , \quad k _ { F } = 1 - \frac { M } { d } ,
$$

$$
c _ { R } = { \frac { H _ { 1 } - r H _ { 2 } - B / g } { k } } , \qquad c _ { F } = { \frac { B } { m } } - { \frac { g M ^ { 2 } } { m ^ { 2 } } } .\tag{5.24}
$$

The symbol h in these formulas is a scalar resolvent parameter, not a sharp risk. All quantities are evaluated at the unique calibrated root $M ( r ) / m = 1 / ( g + \lambda _ { 0 } )$

Theorem 5.2 (Isotropic ETF geometry at three deletions). Under the preceding assumptions, every prescribed $\lambda _ { 0 } > 0$ has exactly one calibrated positive scalar penalty, also unique among all SPD penalties. The centered isotropic response operator is

$$
C _ { m - 3 } ( X , I ) = c _ { R } ( I - P ) + c _ { F } P , \qquad K _ { X } = ( I - P ) + k _ { F } P , \quad k _ { F } = \frac { \lambda _ { 0 } } { g + \lambda _ { 0 } } .
$$

For every positive $r , c _ { R } k _ { F } - c _ { F } > 0$ and $c _ { F } \geq 0$ . Consequently

$$
h _ { m - 3 } ( X , w I ) = w c _ { R } , \qquad w c _ { R } K _ { X } - C _ { m - 3 } ( X , w I ) = w ( c _ { R } k _ { F } - c _ { F } ) P .\tag{5.25}
$$

The complete maximizing response space is $\ker ( X ^ { \top } )$ , of dimension $m - d .$ Every response with a nonzero fitted component has strictly smaller normalized risk.

Proof. Triangle classes and positivity. For a deleted triple $D _ { 0 } = \{ i , j , l \}$ put $e = { \mathrm { s i g n } } ( T _ { i j } T _ { j l } T _ { l i } ) \in$ $\{ - 1 , 1 \}$ . A diagonal sign switch makes the three of-diagonal entries of its local Gram $J = E _ { D _ { 0 } } T E _ { D _ { 0 } } ^ { \top }$ all equal to $e \mu .$ . Thus J has eigenvalues $1 + 2 e \mu , 1 - e \mu , 1 - e \mu$ . The retained Gram has eigenvalues $g$ repeated $d - 3$ times, $g - 1 - 2 e \mu$ once, and $g - 1 + e \mu$ twice. Therefore

$$
\operatorname * { d e t } ( G _ { S } + r I ) = z ^ { d - 3 } \delta _ { e } , \qquad \delta _ { e } = ( h - 2 e \mu ) ( h + e \mu ) ^ { 2 } = h ^ { 3 } - 3 h q - 2 e \mu ^ { 3 } .\tag{5.26}
$$

The (i, j) entry of $T ^ { 2 } = g T$ , multiplied by $T _ { i j }$ , gives

$$
\sum _ { l \neq i , j } T _ { i j } T _ { i l } T _ { j l } = ( g - 2 ) q .
$$

Hence the number of type-e triples through every pair, through every vertex, and in total is respectively

$$
n _ { e } = \frac { m - 2 + e ( g - 2 ) / \mu } { 2 } , \quad \nu _ { e } = \frac { ( m - 1 ) n _ { e } } { 2 } , \quad N _ { e } = \frac { m ( m - 1 ) n _ { e } } { 6 } .\tag{5.27}
$$

Both classes occur, since

$$
( m - 2 ) ^ { 2 } q - ( g - 2 ) ^ { 2 } = \frac { m ^ { 2 } ( d - 1 ) ( k - 1 ) } { d ^ { 2 } ( m - 1 ) } > 0 .
$$

With $N = { \binom { m } { 3 } }$ , summing (5.26) yields

$$
Z _ { s } ( r ) = N z ^ { d - 3 } D , \qquad \mathbb { P } ( S = D _ { 0 } ^ { c } ) = { \frac { \delta _ { e } } { N D } } .\tag{5.28}
$$

All triples in one class have the same individual probability. This probability difers between the two classes.

Every retained Gram is positive definite even at $r = 0$ . For $k \geq 4$

$$
( g - 1 ) ^ { 2 } - 4 q = \frac { k [ ( k - 4 ) d + k ( k - 1 ) ] } { d ^ { 2 } ( m - 1 ) } > 0 .
$$

For $k = 3$ , complete the columns of $X / { \sqrt { g } }$ to an orthogonal matrix and multiply its remaining columns by $\sqrt { m / k }$ . This gives a unit-row ETF in dimension three. Its row projectors have positive definite Frobenius Gram matrix $( 1 - q ^ { \prime } ) I + q ^ { \prime } \mathbf { 1 } \mathbf { 1 } ^ { \intercal }$ , with $q ^ { \prime } < 1$ , so they are linearly independent in the six-dimensional space of symmetric 3 by 3 matrices. Therefore $m \leq 6$ . Since $d \geq 3$ , the only possible pair is $d = k = 3 ,$ , where $( g - 1 ) ^ { 2 } - 4 q = 1 / 5$ . Thus $g - 1 > 2 \mu$ in every actual case. In particular $z , h \pm 2 \mu , \delta _ { e } , D$ , and $h ^ { 2 } - 4 q$ are positive. This is the usual Naimark-complement and projector-dimension argument; it does not assert existence at other formal parameter pairs.

Calibration with the target fixed first. Expanding the selected determinants into original-row minors and counting the supersets of each minor gives

$$
Z _ { s } ( X , r ) = \sum _ { j = 0 } ^ { d } { \binom { m - j } { s - j } } r ^ { d - j } e _ { j } ( X ^ { \top } X ) .\tag{5.29}
$$

At $X ^ { \top } X = g I$ , the matrix derivative of log $Z _ { s }$ with respect to the Gram is scalar. Diferentiating instead with respect to each original row identifies the entire mean map as $f ( \boldsymbol { r } ) \boldsymbol { X } ^ { \top }$ . Taking the trace of its product with X gives

$$
m f ( r ) = \mathbb { E } \operatorname { t r } [ G _ { S } ( G _ { S } + r I ) ^ { - 1 } ] = d - r Z _ { s } ^ { \prime } ( r ) / Z _ { s } ( r ) = M ( r ) .
$$

Every coeficient in (5.29) is positive, because $s = m - 3 \geq d .$ Under the distribution on j proportional to those terms, $M = \mathbb { E } j$ and

$$
\frac { d M } { d \log r } = - \mathrm { V a r } ( j ) < 0 , \qquad M ( 0 + ) = d , \quad M ( + \infty ) = 0 .
$$

Thus $M / m = 1 / { ( g + \lambda _ { 0 } ) }$ has a unique positive solution for every fixed positive target. Theorem 2.1 gives uniqueness among SPD penalties. In particular $k _ { F } = 1 - M / d = r H _ { 1 } / d > 0$

The centered operator has two sectors. Put $L = I - T / z$ and $Q _ { D _ { 0 } } = I _ { 3 } - J / z$ . Solving the retained normal equation gives

$$
w _ { S } = z ^ { - 1 } X ^ { \top } y - z _ { D _ { 0 } } ( y ) , \qquad z _ { D _ { 0 } } ( y ) = z ^ { - 1 } X _ { D _ { 0 } } ^ { \top } Q _ { D _ { 0 } } ^ { - 1 } E _ { D _ { 0 } } L y .
$$

The mean correction is $( z ^ { - 1 } - f ) X ^ { \top } y$ . To compute its second moment, put $a = e \mu$ and define

$$
\begin{array} { l } { { A _ { e } = \displaystyle \frac { z ^ { 2 } ( 1 + a - 2 q ) ( a - 2 h ) } { ( h - 2 a ) ^ { 2 } ( h + a ) ^ { 2 } } , } } \\ { { B _ { e } = \displaystyle \frac { z ^ { 2 } ( h ^ { 2 } + 2 h - a + 2 q ) } { ( h - 2 a ) ^ { 2 } ( h + a ) ^ { 2 } } . } } \end{array}
$$

Interpolation of $z ^ { 2 } x / ( z - x ) ^ { 2 }$ at $1 + 2 a , 1 - a$ gives $Q _ { D _ { 0 } } ^ { - 1 } J Q _ { D _ { 0 } } ^ { - 1 } = A _ { e } I _ { 3 } + B _ { e } J$ . Its diagonal entries are $A _ { e } + B _ { e }$ , and its $( i , j )$ entry is $B _ { e } T _ { i j }$ . Using the exact probabilities and the incidences in (5.27), its embedded average is $A _ { * } I + B _ { * } T$ , where

$$
\begin{array} { l } { { \displaystyle { A _ { * } = \frac { 1 } { N D } \sum _ { e = \pm 1 } n _ { e } \delta _ { e } \left[ \frac { m - 1 } { 2 } A _ { e } + \frac { m - 3 } { 2 } B _ { e } \right] } , } } \\ { { \displaystyle { B _ { * } = \frac { 1 } { N D } \sum _ { e = \pm 1 } n _ { e } \delta _ { e } B _ { e } } . } } \end{array}
$$

Subtracting the actual correction mean square yields

$$
C = z ^ { - 2 } L ( A _ { * } I + B _ { * } T ) L - ( z ^ { - 1 } - f ) ^ { 2 } T .\tag{5.30}
$$

Since $T$ is zero on residuals and equals $g I$ on fitted responses, this proves the asserted two-sector form.   
No transitivity of a frame symmetry group is needed.

The retained spectra and actual class weights give $H _ { j } = \mathbb { E } \operatorname { t r } ( G _ { S } + r I ) ^ { - j } , j = 1 , 2$ . For example,

$$
\begin{array} { c } { { \delta _ { e } \left[ \displaystyle \frac { 1 } { h - 2 a } + \displaystyle \frac { 2 } { h + a } \right] = 3 ( h ^ { 2 } - q ) , } } \\ { { \delta _ { e } \left[ \displaystyle \frac { 1 } { ( h - 2 a ) ^ { 2 } } + \displaystyle \frac { 2 } { ( h + a ) ^ { 2 } } \right] = \displaystyle \frac { 3 ( h ^ { 2 } - 2 h a + 3 q ) } { h - 2 a } . } } \end{array}
$$

Averaging the second line gives $3 [ h ( h ^ { 2 } - q ) + 3 t ] / [ ( h ^ { 2 } - 4 q ) D ]$ , proving (5.23). In particular $H _ { 2 }$ is not simply $- H _ { 1 } ^ { \prime }$ ; the law changes with r.

For the original-coordinate fit map $F _ { S } = ( G _ { S } + r I ) ^ { - 1 } X _ { S } ^ { \top } E _ { S }$ , let $O = \mathbb { E } F _ { S } ^ { \top } F _ { S }$ . Then

$$
\mathrm { t r } O = H _ { 1 } - r H _ { 2 } , \qquad \mathrm { t r } ( P O ) = B / g , \qquad C = O - f ^ { 2 } T .
$$

The first identity follows from $F _ { S } F _ { S } ^ { \top } = ( G _ { S } + r I ) ^ { - 1 } G _ { S } ( G _ { S } + r I ) ^ { - 1 }$ ; the second uses $F _ { S } X = ( G _ { S } +$ $r I ) ^ { - 1 } G _ { S }$ . Dividing the residual and fitted traces by k and d proves (5.24). Actual covariance gives $c _ { F } \geq 0$ . The term $g M ^ { 2 } / m ^ { 2 }$ is the necessary fitted mean-square subtraction.

Strict gap and complete equality. Appendix C.2 constructs an integer polynomial $\mathcal { H } ( r , d , k )$ directly from these moments and proves the identity

$$
c _ { R } k _ { F } - c _ { F } = \frac { 3 d ^ { 3 } r \mathcal { H } } { m \mathcal { Z } ^ { 2 } \mathcal { E } \mathcal { F } ^ { 2 } } ,
$$

with all denominator factors positive. After $d = u + 3 , k = v + 4$ , the polynomial has 581 positive integer coeficients, including a positive constant. The appendix gives its defining expression, the exact division and the common-denominator derivation. The ancillary checker reconstructs the polynomial before comparing every coeficient. For the only remaining actual case $d = k = 3$ , the gap is

$$
\frac { r ( 5 r ^ { 2 } + 1 0 r + 4 ) ( 5 r ^ { 3 } + 2 5 r ^ { 2 } + 3 0 r + 6 ) } { 2 ( r + 1 ) ^ { 2 } ( 5 r ^ { 2 } + 1 0 r + 1 ) ( 5 r ^ { 2 } + 1 0 r + 2 ) ^ { 2 } } > 0 .
$$

This is an exact algebraic proof on the whole stated domain. Finally, for $y = y _ { R } + y _ { F }$ with residual and fitted components, the risk ratio is

$$
\frac { w [ c _ { R } \| y _ { R } \| ^ { 2 } + c _ { F } \| y _ { F } \| ^ { 2 } ] } { \| y _ { R } \| ^ { 2 } + k _ { F } \| y _ { F } \| ^ { 2 } } .
$$

Its deficit from $w c _ { R }$ times the denominator is $w ( c _ { R } k _ { F } - c _ { F } ) \lVert y _ { F } \rVert ^ { 2 }$ . The coeficient is strictly positive, and $k > 0$ ensures nonzero residuals. This proves both the sharp value and every equality case. □

For a concrete unequal-weight example, use the (16, 6) frame in Appendix C.1. At $r = 1 / 3$ there are 320 positive and 240 negative triangles. Each positive triangle has probability 49/27680, while each negative triangle has probability 5/2768. Equations (5.23)–(5.24) give

$$
\lambda _ { 0 } = \frac { 2 8 7 1 2 } { 6 3 9 6 9 } , \quad c _ { R } = \frac { 4 6 8 6 3 } { 8 8 5 7 6 0 } , \quad c _ { F } = \frac { 1 2 9 1 5 8 5 } { 1 6 5 4 9 5 3 9 8 4 } , \quad c _ { R } k _ { F } - c _ { F } = \frac { 4 5 2 9 1 0 5 2 1 } { 6 6 1 9 8 1 5 9 3 6 0 } > 0 .
$$

In this example, the displayed $\lambda _ { 0 }$ is the prescribed target for every subset. The theorem separately proves calibration for every prescribed positive target.

Combining the pair, triple, and LOO results gives the same complete space ker $( X ^ { \top } )$ at $s = m -$ $1 , m - 2 , m - 3$ , for the same existing frame, scalar target, and isotropic query in the triple theorem’s domain. Each budget uses its own calibrated penalty. The sharp values need not agree or be monotone. The three-deletion theorem makes no assertion for nonisotropic queries or for other budgets.

## 6 A same-sample ridge–HT mixture

At a balanced design, the strict gap between contrast and mean risk lets us change the estimator while keeping the same sampled rows and target. We use this freedom to lower the worst-response risk for every nonzero positive semidefinite query. Calibration uniqueness fixes the penalty within the matched ridge family. It does not rule out a diferent unbiased estimator on the same subset. In the balanced model, let $a = n / ( n + \lambda _ { 0 } ) , \mu = s / d .$ , and $c _ { H } = a / \mu$ . Each original row has inclusion probability $s / m$ Thus the estimate

$$
( w _ { H } ) _ { j } = c _ { H } \sum _ { i \in S \cap j } \varepsilon _ { i } y _ { i }
$$

and the matched ridge estimate $w _ { R }$ both have mean w under the same determinant law. Consider $w ( t ) = ( 1 - t ) w _ { R } + t w _ { H }$

Put $u _ { t } ( k ) = ( 1 - t ) / ( r _ { s } + k ) + t c _ { H }$ and $g _ { t } ( k ) = k u _ { t } ( k )$ . Define $\alpha _ { s } ( t ) , \beta _ { s } ( t ) , \tau _ { s } ( t )$ by (4.3)–(4.5), replacing $1 / ( r _ { s } + k )$ by $u _ { t } ( k )$ and $f ( k )$ by $g _ { t } ( k )$ . They are the contrast and mean coeficients in the same covariance decomposition (4.6) for $w ( t )$ , and equal the ridge coeficients at $t = 0$ . Their formulas are given in (6.1) below.

Proposition 6.1 (A finite improving mixture). Under the assumptions of Theorem $4 . 1 ,$ an explicit $T _ { s } > 0$ , depending only on the features, target, and budget, satisfies the following. For every $0 < t \leq T _ { s }$ and every nonzero $W \succeq 0$ ，

$$
\operatorname* { s u p } _ { y \neq 0 } \frac { \operatorname { t r } ( W \operatorname { C o v } ( w ( t ) ) ) } { J _ { 0 } ( y ) } = \alpha _ { s } ( t ) \operatorname* { m a x } _ { j } W _ { j j }
$$

The complete maximizing raw response space remains $\mathcal { E }$

Proof. Fix a balanced design, $d \leq s < n d .$ , and its exactly calibrated $r = r _ { s } > 0$ . All expectations below use the actual count law (4.2). Put $a = n / ( n + \lambda _ { 0 } ) , b = n ( 1 - a ) , \mu = s / d$ , and $c _ { H } = a / \mu$ . Define

$$
\begin{array} { c } { { h ( k ) = \displaystyle \frac { 1 } { r + k } , \qquad f ( k ) = k h ( k ) , } } \\ { { \delta ( k ) = c _ { H } - h ( k ) , \qquad v ( k ) = k \delta ( k ) , } } \\ { { h _ { t } ( k ) = h ( k ) + t \delta ( k ) , \qquad g _ { t } ( k ) = f ( k ) + t v ( k ) . } } \end{array}
$$

The mixture is $\begin{array} { r } { ( w ( t ) ) _ { j } = h _ { t } ( K _ { j } ) \sum _ { i \in S \cap j } \varepsilon _ { i } y _ { i } } \end{array}$ . By symmetry every original row has inclusion probability $s / ( n d )$ . Hence $\mathbb { E } w _ { H } = a q = w _ { 0 }$ , and linearity proves $\mathbb { E } w ( t ) = w _ { 0 }$ for every real t.

## 6.1 The covariance quadratics

Conditional on all counts, the selected contrast sums are independent across groups, have mean zero, and variance $K _ { j } ( n - K _ { j } ) E _ { j } / [ n ( n - 1 ) ]$ . Thus, with $\omega ( k ) = k ( n - k ) / [ n ( n - 1 ) ]$ 2

$$
\begin{array} { r l } & { \alpha ( t ) = \mathbb { E } [ \omega ( K _ { 1 } ) h _ { t } ( K _ { 1 } ) ^ { 2 } ] , } \\ & { \beta ( t ) = \frac { \mathbb { E } [ \left( g _ { t } ( K _ { 1 } ) - g _ { t } ( K _ { 2 } ) \right) ^ { 2 } ] } { 2 b } , \qquad \tau ( t ) = - \frac { \mathrm { C o v } \left( g _ { t } ( K _ { 1 } ) , g _ { t } ( K _ { 2 } ) \right) } { b } , } \\ & { \mathrm { C o v } ( w ( t ) ) = \alpha ( t ) \mathrm { d i a g } ( E _ { j } ) + b \beta ( t ) \mathrm { d i a g } ( q _ { j } ^ { 2 } ) - b \tau ( t ) q q ^ { \top } . } \end{array}\tag{6.1}
$$

This calculation includes the covariance between the ridge and inclusion-weighted estimates on their shared subset. Their cross contrast coeficient is $c _ { H } \mathbb { E } [ \omega ( K _ { 1 } ) h ( K _ { 1 } ) ]$ ]; their mean cross covariance is $\operatorname { C o v } ( f ( K _ { i } ) , c _ { H } K _ { j } )$ . Exchangeability makes that cross covariance symmetric. No independence between the two estimates is assumed.

For $0 \leq t \leq 1$ , the score is strictly increasing:

$$
\frac { d } { d k } g _ { t } ( k ) = \frac { ( 1 - t ) r } { ( r + k ) ^ { 2 } } + t c _ { H } > 0 .
$$

The actual-law conditional ordering proved at the start of Appendix B therefore gives $\tau ( t ) > 0$ throughout this interval.

The positive marginal support of $K _ { 1 }$ is ma $\mathfrak { c } ( 1 , s - ( d - 1 ) n ) , \dotsc , \operatorname* { m i n } ( n , s )$ . It contains two consecutive positive integers: if its upper endpoint is $n _ { : }$ , then $n - 1$ is possible since $s < n d ;$ otherwise the upper endpoint is $s \geq 2$ and the lower positive endpoint is one. Consider the size-biased probability $\nu ( k ) =$ $k \mathbb { P } ( K _ { 1 } = k ) / \mu$ . Calibration says

$$
\mathbb { E } _ { \nu } h ( K _ { 1 } ) = \frac { \mathbb { E } f ( K _ { 1 } ) } { \mu } = c _ { H } .
$$

Diferentiating the finite sum defining α gives

$$
\alpha ^ { \prime } ( 0 ) = - \frac { 2 \mu } { n ( n - 1 ) } \mathrm { C o v } _ { \nu } ( ( n - K _ { 1 } ) h ( K _ { 1 } ) , h ( K _ { 1 } ) ) < 0 .\tag{6.2}
$$

Both functions inside this covariance are strictly decreasing on positive counts, with derivatives $- ( n +$ $r ) / ( r + k ) ^ { 2 }$ and $- 1 / ( r + k ) ^ { 2 }$ . The independent-copy covariance identity and the two distinct positive support points make the covariance strictly positive.

Put

$$
A = \alpha ( 0 ) , \quad D = - \alpha ^ { \prime } ( 0 ) > 0 , \quad Q = \mathbb { E } [ \omega ( K _ { 1 } ) \delta ( K _ { 1 } ) ^ { 2 } ] .
$$

Then $\alpha ( t ) = A - D t + Q t ^ { 2 }$ . Also $Q > 0 { : }$ if it vanished, δ would vanish on every positive-ω atom, making the derivative zero in contradiction to (6.2). Similarly,

$$
\begin{array} { r l } & { B _ { 0 } = \beta ( 0 ) , } \\ & { B _ { 1 } = \mathbb { E } [ ( f ( K _ { 1 } ) - f ( K _ { 2 } ) ) ( v ( K _ { 1 } ) - v ( K _ { 2 } ) ) ] / b , } \\ & { B _ { 2 } = \mathbb { E } [ ( v ( K _ { 1 } ) - v ( K _ { 2 } ) ) ^ { 2 } ] / ( 2 b ) \geq 0 , } \\ & { \beta ( t ) = B _ { 0 } + B _ { 1 } t + B _ { 2 } t ^ { 2 } . } \end{array}
$$

Theorem 4.1 supplies the strict gap $g = A - B _ { 0 } > 0$ . Define the finite step

$$
H _ { * } = D + | B _ { 1 } | + Q + B _ { 2 } , \qquad T _ { s } = \operatorname* { m i n } \left\{ { \frac { 1 } { 2 } } , { \frac { D } { 2 Q } } , { \frac { g } { 2 H _ { * } } } \right\} > 0 .\tag{6.3}
$$

For $0 < t \leq T _ { s }$

$$
\alpha ( t ) \le A - D t / 2 < A , \qquad \alpha ( t ) - \beta ( t ) \ge g - [ D + | B _ { 1 } | + Q + B _ { 2 } ] t \ge g / 2 > 0 ,
$$

where $t \leq 1$ bounds the quadratic terms in the second inequality. The gap, rather than a limiting continuity argument alone, gives this explicit interval.

For $c = \operatorname* { m a x } _ { j } W _ { j j } > 0$ , the exact deficit is

$$
\begin{array} { l } { \alpha ( t ) c J _ { 0 } - \mathrm { t r } ( W \operatorname { C o v } ( w ( t ) ) ) = \alpha ( t ) \displaystyle \sum _ { j } ( c - W _ { j j } ) E _ { j } } \\ { \displaystyle \qquad + b \displaystyle \sum _ { j } [ \alpha ( t ) c - \beta ( t ) W _ { j j } ] q _ { j } ^ { 2 } + b \tau ( t ) q ^ { \top } W q . } \end{array}
$$

Every mean coeficient is at least $b c [ \alpha ( t ) - \beta ( t ) ] > 0$ . Also $\alpha ( t ) > 0$ , since an intermediate count has positive probability and $h _ { t }$ is positive. Equality therefore holds exactly on the same nonzero contrast space E. This proves Proposition 6.1, including arbitrary singular and nondiagonal PSD queries and all ties. For a finite grid of budgets, min<sub>s</sub> $T _ { s } > 0$ supplies one common weight.

We call $0 < t \leq T _ { s }$ the protected interval: the contrast risk falls and the contrast–mean gap stays positive there. The mixture uses the same sample, labels, target and law. The weight uses features only. It lowers each query’s worst-response risk, but can raise an individual response’s risk. The inclusionweighted endpoint uses the Horvitz–Thompson principle (Horvitz and Thompson, 1952). Combining dependent unbiased estimates is a standard construction (Lavancier and Rochet, 2016). Appendix D gives count formulas and an adverse response.

## 6.2 An exact boundary under a mean-share constraint

Unrestricted maximizing responses have $q = 0$ , and hence $w _ { 0 } = 0$ . To examine responses with a nonzero Full mean, define the deterministic loss share

$$
\eta ( y ) = \frac { b \| q \| ^ { 2 } } { J _ { 0 } ( y ) } , \qquad \mathcal { C } _ { \delta } = \big \{ y \neq 0 : \eta ( y ) \geq \delta \big \} , \quad 0 < \delta \leq 1 .
$$

This share measures the fraction of Full loss carried by the group means. It is a response restriction, not a population signal-to-noise ratio. For each fixed query, let $H _ { \delta } ( t , W )$ be the supremum of normalized mixture risk over $\mathcal { C } _ { \delta }$

Proposition 6.2 (Mean-share risk and all equality cases). For $0 \leq t \leq 1$ and $0 \neq W \succeq 0$ , put

$$
c = \operatorname* { m a x } _ { j } W _ { j j } , \quad p _ { t } = \alpha ( t ) c , \quad B _ { t } ( W ) = \beta ( t ) \mathrm { d i a g } ( W _ { j j } ) - \tau ( t ) W , \quad \ell _ { t } = \lambda _ { \operatorname* { m a x } } ( B _ { t } ( W ) ) .
$$

The sharp value at exact share $\eta ( y ) = \delta$ is $( 1 - \delta ) p _ { t } + \delta \ell _ { t }$ , and

$$
H _ { \delta } ( t , W ) = \operatorname* { m a x } \{ \ell _ { t } , ( 1 - \delta ) p _ { t } + \delta \ell _ { t } \} .\tag{6.4}
$$

At exact share $0 < \delta < 1$ , all maximizers have q in the complete top eigenspace of $B _ { t } ( W )$ , contrasts supported only in groups with $W _ { j j } = c ,$ and $\begin{array} { r } { \sum _ { i } E _ { j } = ( 1 - \delta ) b \vert \vert { q } \vert \vert ^ { 2 } / \delta } \end{array}$ $A t \delta = 1$ they are nonzero pure means in that eigenspace. For the lower-bound-share class, $p _ { t } > \ell _ { t }$ forces share δ; $p _ { t } < \ell _ { t }$ forces share one; and $p _ { t } = \ell _ { t }$ allows every share in $[ \delta , 1 ]$ , with these same conditions on each nonzero sector. For $W = 0$ , every admissible response has zero risk.

Proof. At fixed share η, the contrast and mean Rayleigh bounds give $( 1 - \eta ) p _ { t } + \eta \ell _ { t }$ . Both bounds are attained independently: $n \geq 2$ supplies contrasts in every maximal group, and the mean block has a top eigenvector. Equality in their sum requires equality in each sector of positive energy. The function is afine in η. Its maximum on [δ, 1] and all its equality cases follow by its slope. These sets are energy-constrained cones, generally not linear spaces. □

Theorem 6.3 (Exact all-query improvement phase). Use the feature coeficients and protected interval $0 < t \leq T = T _ { s }$ in (6.3). Set

$$
K _ { \delta } = ( 1 - \delta ) D - \delta B _ { 1 } , \qquad L _ { \delta } = ( 1 - \delta ) Q + \delta B _ { 2 } \geq 0 .
$$

For a fixed protected weight,

$$
H _ { \delta } ( t , W ) < H _ { \delta } ( 0 , W ) \quad f o r \ e v e r y \ 0 \neq W \succeq 0 \quad \Longleftrightarrow \quad K _ { \delta } > L _ { \delta } t .
$$

Some protected positive weight works exactly when $K _ { \delta } > 0$ . In that case one may take $t _ { \delta } =$ min $\{ T , K _ { \delta } / ( 2 L _ { \delta } ) \}$ if $L _ { \delta } > 0 ,$ and $t _ { \delta } = T$ otherwise. Every protected weight works exactly when $K _ { \delta } > L _ { \delta } T . I f K _ { \delta } \leq 0 .$ , no positive scalar weight in this mixture family improves every query, even outside the protected interval.

Proof. On $[ 0 , T ] , \alpha ( t ) > \beta ( t )$ and $\tau ( t ) > 0$ , so $p _ { t } > \ell _ { t }$ and (6.4) uses share δ. Put $\gamma ( t ) = \beta ( t ) - d \tau ( t )$ Exchangeability gives

$$
\gamma ( t ) = \frac { \mathrm { V a r } ( \sum _ { j } g _ { t } ( K _ { j } ) ) } { d b } = ( 1 - t ) ^ { 2 } \gamma ( 0 ) \geq 0 ,
$$

since $\textstyle \sum _ { j } K _ { j } = s$ is fixed. Define $P _ { W } = \mathrm { d i a g } ( W _ { j j } ) - W / d$ and $R _ { W } = W / d .$ . Both are PSD: the Gram representation of W and Cauchy–Schwarz give $W \preceq d \mathrm { d i a g } ( W _ { j j } )$ . Thus $B _ { t } ( W ) = \beta ( t ) P _ { W } + \gamma ( t ) R _ { W }$ and $P _ { W } + R _ { W } \preceq c I$

Write $x = A - \alpha ( t ) > 0$ and $y = \beta ( t ) - B _ { 0 }$ . If $y \geq 0$ , the preceding identity gives $\ell _ { t } - \ell _ { 0 } \leq y c$ and hence

$$
H _ { \delta } ( 0 , W ) - H _ { \delta } ( t , W ) \geq c \{ ( 1 - \delta ) x - \delta y \} = c t ( K _ { \delta } - L _ { \delta } t ) .
$$

A flat rank-one query $W = v v ^ { \top } , v _ { j } \in \{ - 1 , 1 \}$ , attains this bound: its mean top eigenspace is $v ^ { \perp }$ and $\ell _ { t } = \beta ( t )$ . This proves necessity and suficiency when $y \geq 0$

If $y \textless 0$ , set $\rho = \mathrm { m a x } \{ \beta ( t ) / B _ { 0 } , ( 1 - t ) ^ { 2 } \} < 1$ when $\gamma ( 0 ) > 0$ , and $\rho = \beta ( t ) / B _ { 0 } < 1$ otherwise. Here ${ B _ { 0 } } ~ > ~ 0$ because unequal counts have positive probability and f is strictly increasing. We have $B _ { t } ( W ) \preceq \rho B _ { 0 } ( W )$ . Also $\ell _ { 0 } \geq ( B _ { 0 } - \tau ( 0 ) ) c > 0$ , since $B _ { 0 } - \tau ( 0 ) = ( ( d - 1 ) B _ { 0 } + \gamma ( 0 ) ) / d > 0$ . Consequently $\ell _ { t } < \ell _ { 0 }$ for every nonzero query. Both sectors strictly improve, including the pure-mean class $\delta = 1$ . The scalar condition holds as well because $( 1 - \delta ) x - \delta y > 0$

The weight conclusions follow from $L _ { \delta } \geq 0$ . For the final obstruction use the flat query and one baseline attaining response of share $\delta ,$ with mean orthogonal to v. For every real $t > 0$ its risk change equals $- K _ { \delta } t + L _ { \delta } t ^ { 2 } \ge 0$ . The endpoint supremum is at least this response’s risk. Thus it cannot fall below the baseline supremum. □

A useful uniform window follows directly. Put $L _ { + } = \operatorname* { m a x } \{ 0 , B _ { 1 } + B _ { 2 } T \}$ and $\delta _ { \mathrm { s a f e } } = D / ( D + 2 L _ { + } ) > 0$ Since $A - \alpha ( t ) \geq D t / 2$ and $B _ { t } ( W ) - B _ { 0 } ( W ) \preceq t L _ { + } c I .$ , every $0 < \delta \leq \delta _ { \mathrm { s a f e } } / 2$ and $0 < t \leq T$ obeys

$$
H _ { \delta } ( 0 , W ) - H _ { \delta } ( t , W ) \geq D t c / 4 .\tag{6.5}
$$

For finitely many budgets, minima give a common positive share window and weight. This does not imply monotonicity with budget.

For the exact example $n = d = s = 2 , r = 1 , \lambda _ { 0 } = 1 2 / 5$ , Appendix D gives

$$
\alpha ( t ) = \frac { ( 1 1 - t ) ^ { 2 } } { 1 3 3 1 } , \quad \beta ( t ) = \frac { ( 1 1 + 4 t ) ^ { 2 } } { 2 1 7 8 } , \quad T = \frac { 8 4 7 } { 3 1 1 6 } .
$$

Some positive weight improves every query exactly when $\delta < 9 / 3 1$ . At $t = 1 / 4$ , the exact threshold is $\delta < 7 8 3 / 2 8 0 7 . \mathrm { ~ A t }$ share $1 / 2$ , the flat-query sharp risk instead increases by 1241/383328. These follow by substitution in $- K _ { \delta } t + L _ { \delta } t ^ { 2 }$ . They delimit this scalar mixture family; they do not classify other estimators or perturbed mean-share classes.

## 7 Stability under full recalibration

We now return to the unrestricted response class $y \ne 0$ . The mean-share restriction in Section 6 is not imposed here. We first treat arbitrary full-column-rank designs, then return to the balanced designs of Theorem 4.1. Their full equality space has a use when the exact row structure is broken. Keep the raw response coordinates, target $A _ { 0 }$ , and physical query W fixed, and replace X by $X + H$ . At each design recompute the full fit, calibrated penalty, sampling law, and selected fit. The question is whether searching only the old maximizing responses still finds nearly the worst response for the perturbed design. The answer is yes to second order, provided we search the whole space and use the new Full loss.

Use the response matrix and normalized operator from (3.3)–(3.4) at each design. All operator norms below are spectral norms. We first bound the recalibrated operator, then study response-space restriction, and finally compare the two estimator risks.

## 7.1 General-design sensitivity

Theorem 7.1 (Sensitivity of the fully recalibrated procedure). Fix a full-column-rank $\boldsymbol { X } \in \mathbb { R } ^ { m \times d }$ , an SPD target $A _ { 0 } , \ a$ fixed PSD physical query W, and $d \leq s < m$ $\mathit { P u t } \ \epsilon = \| \mathit { H A } _ { 0 } ^ { - 1 / 2 } \| _ { F }$ and recompute the $F u l l ~ f i t ,$ calibrated penalty, determinant law and selected fit at $X + H$ . The explicit constants in (E.1)–(E.4) give, $f o r \epsilon \leq r _ { 0 }$

$$
\begin{array} { r } { \| M _ { s } ( X + H , W ) - M _ { s } ( X , W ) \| \leq \eta _ { s } ( \epsilon ) \leq \mathcal { M } _ { s } \epsilon , \qquad | h _ { s } ( X + H , W ) - h _ { s } ( X , W ) | \leq \eta _ { s } ( \epsilon ) . } \end{array}
$$

For every fixed nonzero raw response, $e ^ { - 2 \epsilon } \le J _ { 0 } ( X + H , y ) / J _ { 0 } ( X , y ) \le e ^ { 2 \epsilon }$ . These constants require no enumeration of row subsets.

Proof. Put $Z = X A _ { 0 } ^ { - 1 / 2 } , V = A _ { 0 } ^ { - 1 / 2 } W A _ { 0 } ^ { - 1 / 2 }$ and $P = A _ { 0 } ^ { - 1 / 2 } R _ { s } A _ { 0 } ^ { - 1 / 2 }$ . Along the feature segment, use the explicit Gram box $[ \ell , u ]$ and constants of Appendix E. Calibration has marked-set activation means

$\rho _ { j } = \lambda _ { j } / ( 1 + \lambda _ { j } )$ and odds $q _ { j } = \lambda _ { j } / p _ { j }$ . Its log-coordinate Hessian is $\mathrm { C o v } ( \mathbf { 1 } _ { J } )$ . Select s slots with weights $1 + q _ { j }$ on feature slots and one on the others, then independently mark a selected feature slot with probability $q _ { j } / ( 1 + q _ { j } )$ . Conditional variance gives

$$
\mathrm { C o v } \bigl ( \mathbf { 1 } _ { J } \bigr ) \succeq \mathrm { d i a g } \{ \rho _ { j } / ( 1 + q _ { j } ) \} \succeq \kappa _ { s } I .
$$

The explicit $\kappa _ { s } > 0$ in (E.2) follows from the penalty sandwich and slot inclusion bounds. Integrating this Hessian bound along a coordinate swap bounds the spectral divided diferences as well. Thus it includes all symmetric matrix directions and repeated eigenvalues. Inverting the moment map gives $\| \dot { P } \| _ { F } \le L _ { X } \epsilon .$

For $F _ { S } = ( Z ^ { \top } D _ { S } Z + P ) ^ { - 1 } Z ^ { \top } D _ { S } - ( I + Z ^ { \top } Z ) ^ { - 1 } Z ^ { \top }$ and $T = ( I + Z Z ^ { \top } ) ^ { 1 / 2 }$ , ordered diferentiation of the ridge fits and the equation $T \dot { T } + \dot { T } T = E Z ^ { \top } + Z E ^ { \top }$ give

$$
\| F _ { S } T \| \le B _ { F } t , \qquad \| ( F _ { S } T ) ^ { \bullet } \| \le C _ { F } \epsilon .
$$

The log derivative of each unnormalized determinant weight is at most $L _ { p } \epsilon$ . The normalized endpoint laws therefore have total variation at most tanh $( L _ { p } \epsilon / 2 )$ , as proved in (E.8). Since the risk atom is PSD and at most $\| V \| B _ { F } ^ { 2 } t ^ { 2 } I$ , integrating its derivative and changing its law yield

$$
\begin{array} { r } { \| M _ { s } ( X + H ) - M _ { s } ( X ) \| \le \| V \| \{ 2 B _ { F } t C _ { F } \epsilon + B _ { F } ^ { 2 } t ^ { 2 } \operatorname { t a n h } ( L _ { p } \epsilon / 2 ) \} = \eta _ { s } ( \epsilon ) . } \end{array}
$$

The Rayleigh principle gives the sharp-value bound. Finally $K = ( I { + } Z Z ^ { \top } ) ^ { - 1 }$ obeys $| \partial _ { x } \log ( y ^ { \top } K y ) | \leq 2 \epsilon$ along the segment, using $\| K ^ { 1 / 2 } Z \| \le 1$ and $\| K ^ { 1 / 2 } \| \le 1$ . Integration gives the loss ratio. Appendix E supplies the stated constants and each derivative and likelihood bound. □

## 7.2 Restriction to the old complete maximizing space

Theorem 7.2 (Recalibrated transfer). Let $X _ { 0 }$ be a balanced design as in Theorem 4.1, and fix nonzero $W \succeq 0$ . Let E be its complete maximizing raw response space. For each $d \leq s <$ nd, there are explicit positive constants $r , L _ { s } , v , \gamma _ { s }$ such that, for $\epsilon = \lVert H A _ { 0 } ^ { - 1 / 2 } \rVert _ { F } \leq r ,$ all recalibrated objects exist and

$$
\| M _ { s } ( X _ { 0 } + H ) - M _ { s } ( X _ { 0 } ) \| \le \eta _ { s } ( \epsilon ) \le L _ { s } \epsilon .\tag{7.1}
$$

Here $\gamma _ { s }$ is the gap below the full top eigenspace at $X _ { 0 } . \ I f \eta _ { s } < \gamma _ { s } / 2$ and $\omega = v \epsilon < 1$ , put

$$
\theta _ { s } = \frac { \eta _ { s } } { \gamma _ { s } - \eta _ { s } } + \frac { \omega } { 1 - \omega } .
$$

Then $| h _ { s } ( X _ { 0 } + H ) - h _ { s } ( X _ { 0 } ) | \le \eta _ { s }$ and

$$
\begin{array} { r l } & { 0 \le h _ { s } ( X _ { 0 } + H ) - \underset { 0 \neq y \in \mathcal { E } } { \operatorname* { s u p } } \frac { y ^ { \top } C _ { s } ( X _ { 0 } + H ) y } { y ^ { \top } K _ { X _ { 0 } + H } y } } \\ & { \le \left( h _ { s } ( X _ { 0 } ) + \eta _ { s } \right) \operatorname* { m i n } \{ 1 , \theta _ { s } ^ { 2 } \} . } \end{array}\tag{7.2}
$$

One positive radius makes these conditions hold at every stated budget.

The normalized coordinate is $u = K _ { X } ^ { 1 / 2 } y$ , so keeping the raw space E fixed means searching $K _ { X } ^ { 1 / 2 } { \mathcal { E } }$ at the new design. The first term in $\theta _ { s }$ controls movement of the top eigenspace; the second controls this change of normalization. The right side of (7.2) is $O ( \epsilon ^ { 2 } )$ at a fixed base. The optimum itself can move by $O ( \epsilon )$ . The response must be reoptimized within the entire old space, including all ties, using the perturbed denominator. A selected old maximizing vector need not have quadratic regret.

## 7.3 The balanced gap and the unchanged raw space

At the balanced base, contrasts in group $j$ are normalized eigenvectors with eigenvalue $\alpha _ { s } W _ { j j }$ , while the normalized mean block is

$$
B _ { \mathrm { m e a n } , s } = \beta _ { s } \mathrm { d i a g } ( W _ { j j } ) - \tau _ { s } W .
$$

To check this normalization, use the orthonormal signed group-mean coordinates ${ \sqrt { n } } q$ . The loss matrix equals $b / n$ on that sector, so division by the loss changes its risk block into the matrix above. On all contrast directions, $K _ { X _ { 0 } } = I .$ . Thus the full top eigenspace is precisely the raw space $\mathcal { E }$ in (4.8). Its gap is explicitly

$$
\gamma _ { s } = \alpha _ { s } c - \operatorname* { m a x } \left\{ \alpha _ { s } \operatorname* { m a x } _ { j \notin I _ { \mathrm { m a x } } } W _ { j j } , \lambda _ { \operatorname* { m a x } } ( B _ { \mathrm { m e a n } , s } ) \right\} > 0 ,\tag{7.3}
$$

where an absent contrast term is omitted. Positivity follows from $B _ { \mathrm { m e a n } , s } ~ \preceq ~ \beta _ { s } c I$ and $\alpha _ { s } > \beta _ { s }$ . In particular a lower bound is the minimum of $c ( \alpha _ { s } - \beta _ { s } )$ and any present positive contrast gap.

Let $M = M _ { s } ( X _ { 0 } ) , M ^ { \prime } = M _ { s } ( X _ { 0 } + H ) , \eta = \eta _ { s } ,$ , and $h ^ { \prime } = \lambda _ { \operatorname* { m a x } } ( M ^ { \prime } )$ . The variational characterization gives $| h ^ { \prime } - h _ { s } ( X _ { 0 } ) | \leq \eta$ . If $u ^ { \prime }$ is a unit top eigenvector of $M ^ { \prime }$ and P is the orthogonal projector onto $\mathcal { E } ,$ projecting its eigenvalue equation onto $\mathcal { E } ^ { \bot }$ gives

$$
\| ( I - P ) u ^ { \prime } \| \leq \frac { \eta } { \gamma _ { s } - \eta } .
$$

Indeed the spectrum of $M$ on that complement is at most $h _ { s } ( X _ { 0 } ) - \gamma _ { s }$ , while $h ^ { \prime } \geq h _ { s } ( X _ { 0 } ) - \eta$ . This estimate applies to every new top eigenvector, regardless of ties.

The normalized response coordinate is $u = K _ { X } ^ { 1 / 2 } y = T ^ { - 1 } y$ . Thus the unchanged raw space $\mathcal { E }$ has normalized image $( T ^ { \prime } ) ^ { - 1 } \mathcal { E }$ at the endpoint. The inverse identity and $T , T ^ { \prime } \succeq I$ give

$$
\begin{array} { r } { \| ( T ^ { \prime } ) ^ { - 1 } - T ^ { - 1 } \| \le \| T ^ { \prime } - T \| \le v \epsilon = \omega . } \end{array}
$$

Since $T ^ { - 1 } = I$ on $\mathcal { E } ,$ the map $( T ^ { \prime } ) ^ { - 1 } | _ { \varepsilon }$ difers from the identity by at most ω. Its least singular value is at least $1 - \omega$ . Resolving its parallel and perpendicular components shows that the projector onto $( T ^ { \prime } ) ^ { - 1 } \mathcal { E }$ difers from $P$ by at most $\omega / ( 1 - \omega )$ (the two spaces have the same dimension). Thus a unit new maximizer has distance at most

$$
\theta _ { s } = \frac { \eta } { \gamma _ { s } - \eta } + \frac { \omega } { 1 - \omega }
$$

from that endpoint normalized space.

If its projection is nonzero, normalize this projection to obtain $v ^ { \prime } .$ . For a top eigenvector $u ^ { \prime }$ of a symmetric positive semidefinite $M ^ { \prime } .$

$$
\begin{array} { r } { \boldsymbol { h } ^ { \prime } - ( \boldsymbol { v } ^ { \prime } ) ^ { \top } \boldsymbol { M } ^ { \prime } \boldsymbol { v } ^ { \prime } \leq ( \boldsymbol { h } ^ { \prime } - \lambda _ { \operatorname* { m i n } } ( \boldsymbol { M } ^ { \prime } ) ) [ 1 - | \langle \boldsymbol { u } ^ { \prime } , \boldsymbol { v } ^ { \prime } \rangle | ^ { 2 } ] \leq h ^ { \prime } \theta _ { s } ^ { 2 } . } \end{array}
$$

If the projection is zero use the trivial upper bound $h ^ { \prime } .$ . Optimizing over the endpoint normalized image is exactly optimizing over unchanged raw $y \in \mathcal { E }$ with the endpoint loss denominator. Since $h ^ { \prime } \leq h _ { s } ( X _ { 0 } ) + \eta$ this proves (7.2).

At the balanced base $\sigma = z = \sqrt { n / \lambda _ { 0 } }$ , so the constants above are explicit. Over the finite grid $d \leq s < n d .$ , the radius

$$
r _ { * } = \operatorname* { m i n } _ { d \le s < n d } \left\{ r _ { 0 } , \frac { 1 } { 2 v } , \frac { \gamma _ { s } } { 4 \mathcal { M } _ { s } } \right\} > 0
$$

gives $\omega \leq 1 / 2$ and $\eta _ { s } \leq \gamma _ { s } / 4$ at every size. It proves the asserted common local order. No uniformity is asserted as dimension grows, the target penalty vanishes, or a query gap closes.

## 7.4 A larger space with a query-uniform radius

The old space can have a small external gap when two query diagonals are nearly tied. Retaining more contrast groups removes this dependence. For $I \subseteq \{ 1 , \ldots , d \}$ containing every maximal query group, let $\mathcal { E } _ { I }$ be the unchanged raw contrasts supported on I. It has dimension $( n - 1 ) | I |$ . The base top-to-complement gap is at least

$$
\gamma _ { I } = \operatorname* { m i n } \{ \alpha _ { s } ( c - \operatorname* { m a x } _ { j \notin I } W _ { j j } ) , ( \alpha _ { s } - \beta _ { s } ) c \} > 0 ,
$$

with the first term omitted if I contains every group. This follows from the exact contrast eigenvalues and $\beta _ { s } \mathrm { d i a g } ( W _ { j j } ) - \tau _ { s } W \preceq \beta _ { s } c I$ . For $\mathcal { E } _ { \mathrm { a l l } } = \ker X _ { 0 } ^ { \top }$ it gives $\gamma _ { \mathrm { a l l } } = ( \alpha _ { s } - \beta _ { s } ) c$ . For $I _ { \vartheta } = \{ j : W _ { j j } \geq ( 1 - \vartheta ) c \}$ $0 < \vartheta \leq 1$ , it gives $\gamma _ { I } \geq c \operatorname* { m i n } \{ \alpha _ { s } \vartheta , \alpha _ { s } - \beta _ { s } \}$

Theorem 7.3 (Enlarged contrast-space transfer). Keep the balanced base, target and fixed physical query, and recalibrate exactly at $X _ { 1 } ~ = ~ X _ { 0 } + H$ . Write $K _ { 1 } \ = \ K _ { X _ { 1 } } , \ M _ { 1 } \ = \ \bar { K } _ { 1 } ^ { - 1 / 2 } C _ { s } ( \bar { X } _ { 1 } , \bar { W } ) \bar { K } _ { 1 } ^ { - 1 / 2 }$ and $h _ { 1 } = \lambda _ { \operatorname* { m a x } } ( M _ { 1 } )$ . Use the finite surrogate Mf and error $e = e _ { \mathrm { e n d } }$ from $( \mathrm { F } . 1 ) \mathrm { - } ( \mathrm { F } . 5 )$ , or any valid symmetric surrogate with $\| M _ { 1 } - \widetilde M \| \leq e$ . Let U and Q be orthonormal bases of $K _ { 1 } ^ { 1 / 2 } \mathcal { E } _ { I }$ and its orthogonal complement. Define

$$
\begin{array} { r } { a = \lambda _ { \operatorname* { m a x } } ( U ^ { \top } \widetilde { M } U ) , \quad d _ { 0 } = \lambda _ { \operatorname* { m a x } } ( Q ^ { \top } \widetilde { M } Q ) , \quad b = \| Q ^ { \top } \widetilde { M } U \| , \quad g = a - d _ { 0 } - 2 e , \quad B = b + e . } \end{array}
$$

$H h _ { 1 , I }$ is the supremum over nonzero raw $y \in \mathcal { E } _ { I }$ with the endpoint loss denominator, then, when $g > 0$

$$
0 \le h _ { 1 } - h _ { 1 , I } \le \frac { 2 B ^ { 2 } } { \sqrt { g ^ { 2 } + 4 B ^ { 2 } } + g } \le \frac { B ^ { 2 } } { g } .\tag{7.4}
$$

At every endpoint an alternative bound is $h _ { 1 } - h _ { 1 , I } \leq \lambda _ { \operatorname* { m a x } } ( \widetilde { M } ) - a + 2 \epsilon$ . Every response of true sharp deficit $h _ { 1 } - r _ { 1 } ( y ) \leq \rho$ , where $r _ { 1 } ( y ) = y ^ { \top } C _ { s } ( X _ { 1 } , W ) y / ( y ^ { \top } K _ { 1 } y )$ , obeys, $i f g > 0$

$$
\operatorname* { m i n } _ { v \in \mathcal { E } _ { I } } \sqrt { \frac { ( y - v ) ^ { \top } K _ { 1 } ( y - v ) } { y ^ { \top } K _ { 1 } y } } \leq \operatorname* { m i n } \left\{ 1 , \frac { B + \sqrt { B ^ { 2 } + g \rho } } { g } \right\} .\tag{7.5}
$$

For an exact maximizer the sharper bound is $B / \sqrt { g ^ { 2 } + B ^ { 2 } }$ . For the full contrast space, one explicit positive feature radius works for all nonzero PSD queries after normalization ma ${ \mathrm { x } } _ { j } W _ { j j } = 1$ , and all budgets $d \leq s < m$ . On that ball the regret is $O ( \epsilon ^ { 2 } )$ and the leakage is $O ( \epsilon + { \sqrt { \rho } } )$ , uniformly over those queries and budgets. The same holds for each fixed threshold $\vartheta > 0$ with its query-dependent space $\mathcal { E } _ { I _ { \vartheta } }$

Proof. Let $A = \lambda _ { \operatorname* { m a x } } ( U ^ { \top } M _ { 1 } U ) , D = \lambda _ { \operatorname* { m a x } } ( Q ^ { \top } M _ { 1 } Q )$ and $b _ { 1 } = \lVert Q ^ { \top } M _ { 1 } U \rVert$ . Compression and the variational principle give $A - D \geq g$ and $b _ { 1 } \leq B$ . For a unit vector $x = U u + Q v$ 2

$$
x ^ { \top } M _ { 1 } x \leq A \| u \| ^ { 2 } + D \| v \| ^ { 2 } + 2 b _ { 1 } \| u \| \| v \| .
$$

The largest eigenvalue of the corresponding two-by-two matrix, minus A, is $2 b _ { 1 } ^ { 2 } / ( \sqrt { ( A - D ) ^ { 2 } + 4 b _ { 1 } ^ { 2 } } + A -$ $D )$ . It decreases with the gap and increases with the cross norm. This proves $( 7 . 4 )$ , the classical block Rayleigh bound (Li and Li, 2005). Weyl’s inequality proves the alternative bound. Here $A = h _ { 1 , I }$ because the trial space is $K _ { 1 } ^ { 1 / 2 } \mathcal { E } _ { I }$

For leakage put $t = \| v \|$ . Since $h _ { 1 } \geq A$

$$
h _ { 1 } - x ^ { \top } M _ { 1 } x \geq ( A - D ) t ^ { 2 } - 2 b _ { 1 } t { \sqrt { 1 - t ^ { 2 } } } \geq g t ^ { 2 } - 2 B t .
$$

Solving this quadratic proves (7.5). For a top eigenvector, its complementary eigen-equation gives $\left( h _ { 1 } - \right.$ $D ) t \leq b _ { 1 } \sqrt { 1 - t ^ { 2 } }$ , proving the sharper bound, including ties. A surrogate deficit at most r implies true deficit at most $r + 2 e$ . Thus the leakage statement also has an evaluable surrogate test.

For the uniform claim, Appendix F proves $\| M _ { 1 } - M _ { 0 } \| \le L h$ and $e \le \bar { e } ( h ) = O ( h ^ { 2 } )$ , with $h \ =$ $\epsilon + \| P _ { 1 } - P _ { 0 } \|$ and constants independent of normalized W. It bounds the moving-space projector by $f ( h ) = O ( h )$ and proves

$$
g \geq \gamma - ( \alpha _ { s } + \gamma ) f ( h ) ^ { 2 } - 2 L h - 4 \bar { e } ( h ) , \quad B \leq L h + 2 \alpha _ { s } f ( h ) + 2 \bar { e } ( h ) .
$$

The explicit dyadic choice (F.11) makes $g \ge \gamma / 2 > 0$ on a whole ball. The full-space gap is $\gamma = \alpha _ { s } - \beta _ { s }$ or use the threshold gap above. Exact calibration sensitivity gives $h \leq ( 1 + L _ { X , s } ) \epsilon$ . The finite minimum (F.12) then proves the common feature radius, quadratic regret and leakage orders. □

This argument uses the top value in the retained space, not its lowest value. Some retained contrast eigenvalues may lie below mean eigenvalues. There is no assertion that the entire retained space is a separated spectral cluster. The full contrast space has dimension $m - d$ and need not be small. Its common radius removes dependence on internal query gaps; it does not make the sharp risk itself vary quadratically.

## 7.5 The mixture gain under the same perturbation

The preceding results approximate a response search. We now bound the change of the estimator gain itself. This is a separate argument. At $X \ = \ X _ { 0 } + H$ , use the exactly calibrated ridge fit and the same-subset inclusion-weighted estimate

$$
\ b w _ { \ b { H } } = \ b { L } _ { 0 } ( \ b { X } ) \mathrm { d i a g } ( \ b { 1 } _ { S } / \pi ( \ b { X } ) ) \ b { y } , \qquad \ b { L } _ { 0 } ( \ b { X } ) = ( \ b { X } ^ { \top } \ b { X } + \lambda _ { 0 } \ b { I } ) ^ { - 1 } \ b { X } ^ { \top } .
$$

Here $\pi _ { i } ( X )$ is the actual inclusion probability under the recalibrated law. Both means equal the new Full fit. Let $h _ { t } ( X , W )$ be the mixture’s sharp risk over unrestricted $y \ne 0$ , using the new Full loss; put $h _ { R } = h _ { 0 }$ . Keep the base feature-selected weight interval $0 < t \leq T _ { s }$

Theorem 7.4 (A local bound on the mixture gain). Fix a balanced base, target and budget from Proposition 6.1. Write $\alpha ( t ) = A - D t + Q t ^ { 2 }$ and $T = T _ { s }$ . There are explicit $r _ { 1 } , C > 0$ , independent of query and weight, such that for $\epsilon = \| H / \sqrt { \lambda _ { 0 } } \| _ { F } \leq r _ { 1 }$ , every nonzero PSD W and every $0 < t \leq T$ satisfy, with $c = \operatorname* { m a x } _ { j } W _ { j j }$

$$
| h _ { R } ( X , W ) - h _ { t } ( X , W ) - t ( D - Q t ) c | \leq C t \epsilon c .\tag{7.6}
$$

Thus $\epsilon \leq \operatorname* { m i n } \{ r _ { 1 } , D / ( 4 C ) \}$ gives gain at least $D t c / 4 > 0$

The factor t comes from the coupled same-subset operators and their base deficits. Two separate Lipschitz bounds on the sharp risks would lose this factor. The following proof includes both comparisons.

## 7.6 The coupled operator bound

Use the physical coeficients $A , D , Q , B _ { 0 } , B _ { 1 } , B _ { 2 }$ and $T = T _ { s }$ from Section $6 ;$ set $g = A - B _ { 0 } > 0$ and $\tau ( t ) = \tau _ { 0 } + \tau _ { 1 } t + \tau _ { 2 } t ^ { 2 }$ . The coeficients of τ are obtained by expanding (6.1). Explicitly, with $f , v , b$ from Section 6,

$$
\begin{array} { r l } & { \tau _ { 0 } = - \operatorname { C o v } ( f ( K _ { 1 } ) , f ( K _ { 2 } ) ) / b , \qquad \tau _ { 2 } = - \operatorname { C o v } ( v ( K _ { 1 } ) , v ( K _ { 2 } ) ) / b , } \\ & { \tau _ { 1 } = - \{ \operatorname { C o v } ( f ( K _ { 1 } ) , v ( K _ { 2 } ) ) + \operatorname { C o v } ( v ( K _ { 1 } ) , f ( K _ { 2 } ) ) \} / b . } \end{array}
$$

These are physical coeficients; they are not divided by $\lambda _ { 0 } .$

Set $Z _ { 0 } = X _ { 0 } / \sqrt { \lambda _ { 0 } } , E = H / \sqrt { \lambda _ { 0 } }$ , and $Z _ { x } = Z _ { 0 } + x E .$ . In (E.1)–(E.3) take $A _ { 0 } = \lambda _ { 0 } I$ and evaluate all constants at $X _ { 0 }$ . To avoid confusing the metric bound there with the mixture weight, denote that bound by $\theta = \sqrt { 1 + v ^ { 2 } }$ , and set

$$
\begin{array} { r } { q _ { * } = L _ { p } , \qquad r _ { 1 } = \operatorname* { m i n } \{ r _ { 0 } , ( 2 q _ { * } ) ^ { - 1 } \} , \qquad \pi _ { * } = ( s / m ) e ^ { - 1 } . } \end{array}
$$

Equation (E.7) gives $P _ { x } = R _ { s } ( X _ { 0 } + x H ) / \lambda _ { 0 } \succeq p I$ and $\| \dot { P } _ { x } \| _ { F } \leq L _ { X } \epsilon$ throughout this ball. Let $p _ { x } ( S )$ be the normalized subset probability. The log-determinant bound in Appendix E implies

$$
| \partial _ { x } \log p _ { x } ( S ) | \leq 2 q _ { * } \epsilon , \qquad | \dot { \pi } _ { i } | = | \mathbb { E } _ { x } [ \mathbf { 1 } _ { i \in S } \partial _ { x } \log p _ { x } ( S ) ] | \leq 2 q _ { * } \epsilon \pi _ { i } .
$$

All inclusions are positive. Their base values are $s / m$ , so integration and diferentiation of the reciprocal give

$$
\pi _ { i } ( x ) \geq ( s / m ) e ^ { - 2 q _ { * } \epsilon } \geq \pi _ { * } , \qquad | \partial _ { x } \pi _ { i } ^ { - 1 } | \leq 2 q _ { * } \epsilon / \pi _ { * } .\tag{7.7}
$$

This step uses the changing inclusions, rather than their base values.

Write $D _ { S } = \mathrm { d i a g } ( { \bf 1 } _ { S } )$ and

$$
\begin{array} { r l } & { \quad A _ { S } = Z _ { x } ^ { \top } D _ { S } Z _ { x } + P _ { x } , \quad L _ { S } = A _ { S } ^ { - 1 } Z _ { x } ^ { \top } D _ { S } , \quad L = ( I + Z _ { x } ^ { \top } Z _ { x } ) ^ { - 1 } Z _ { x } ^ { \top } , } \\ & { I _ { S } = \mathrm { d i a g } ( \mathbf { 1 } _ { S } / \pi ) , \quad F _ { R } = L _ { S } - L , \quad F _ { H } = L ( I _ { S } - I ) , \quad G = F _ { H } - F _ { R } . } \end{array}
$$

The physical output maps are these maps divided by $\sqrt { \lambda _ { 0 } }$ . The fit bounds from Appendix E and (7.7) yield the following atom and derivative bounds:

$$
\begin{array} { l l } { { B _ { R } = ( 2 \sqrt { p } ) ^ { - 1 } + 1 / 2 , ~ } } & { { ~ D _ { R } = 5 / ( 4 p ) + 5 / 4 + L _ { X } / ( 2 p ^ { 3 / 2 } ) , } } \\ { { B _ { H } = ( 2 \pi _ { * } ) ^ { - 1 } , ~ } } & { { ~ D _ { H } = ( 5 / 4 + q _ { * } ) / \pi _ { * } , } } \\ { { B _ { G } = B _ { R } + B _ { H } , ~ } } & { { ~ D _ { G } = D _ { R } + D _ { H } . } } \end{array}
$$

Here $\| F _ { R } \| \le B _ { R } , \| \dot { F } _ { R } \| \le D _ { R } \epsilon$ , and likewise for $F _ { H } , G$ . For example, $\| I _ { S } - I \| \le 1 / \pi _ { * } , \| L \| \le 1 / 2$ , and $\lVert \dot { L } \rVert \leq 5 \epsilon / 4$ give the H bounds. With $T _ { x } = ( I + Z _ { x } Z _ { x } ^ { \top } ) ^ { 1 / 2 }$ , the Sylvester calculation in Appendix E gives $\| T _ { x } \| \leq \theta$ and $\| \dot { T } _ { x } \| \le v \epsilon$ . Hence, for $U _ { R } = F _ { R } T _ { x } , U _ { G } = G T _ { x }$ , put

$$
a _ { R } = \theta B _ { R } , \quad d _ { R } = \theta D _ { R } + v B _ { R } , \quad \quad a _ { G } = \theta B _ { G } , \quad d _ { G } = \theta D _ { G } + v B _ { G } .
$$

These bound $U _ { R } , U _ { G }$ and their derivatives, with a factor ϵ on each derivative bound.

Exact calibration and inclusion weighting give the same mean $L y$ in target coordinates. Their sharedsubset covariance therefore gives

$$
M _ { t } = M _ { R } + t \mathcal { I } _ { t } , \qquad \mathcal { I } _ { t } = \mathbb { E } _ { x } [ U _ { R } ^ { \top } V U _ { G } + U _ { G } ^ { \top } V U _ { R } + t U _ { G } ^ { \top } V U _ { G } ] , \quad V = W / \lambda _ { 0 } .\tag{7.8}
$$

This polynomial identity also defines $\mathcal { I } _ { 0 }$ without division by zero. Since $\| V \| \le d c / \lambda _ { 0 }$ , define

$$
\begin{array} { l } { { \displaystyle { \cal L } _ { R } = \frac { d } { \lambda _ { 0 } } ( 2 a _ { R } d _ { R } + 2 q _ { * } a _ { R } ^ { 2 } ) , } } \\ { { \displaystyle { \cal L } _ { D } = \frac { d } { \lambda _ { 0 } } \{ 2 ( d _ { R } a _ { G } + a _ { R } d _ { G } ) + 2 T a _ { G } d _ { G } + 2 q _ { * } ( 2 a _ { R } a _ { G } + T a _ { G } ^ { 2 } ) \} , \qquad L _ { M } = L _ { R } + T L _ { D } . } } \end{array}\tag{7.9}
$$

Diferentiating the finite expectation in (7.8) diferentiates both its atom and its probability. The first terms in (7.9) bound atom derivatives; the terms with $q _ { * }$ bound the normalized score contribution. The cross atom may be indefinite, so we use its full norm. Integration gives, simultaneously for $0 \leq t \leq T$

$$
\| E _ { A } \| \leq L _ { R } \epsilon c , \quad \| E _ { B } \| \leq L _ { M } \epsilon c , \quad \| E _ { B } - E _ { A } \| \leq t L _ { D } \epsilon c ,\tag{7.10}
$$

where $E _ { A } = M _ { R } ( X , W ) - M _ { R } ( X _ { 0 } , W )$ and $E _ { B } = M _ { t } ( X , W ) - M _ { t } ( X _ { 0 } , W )$ . In particular the last bound retains t.

## 7.7 Comparing two moving sharp risks

We give the elementary spectral step explicitly. Suppose symmetric operators A, B have largest eigenvalues $a , b ,$ and write $H _ { A } = a I - \mathsf { A } , H _ { B } = b I - \mathsf { B }$ , and $N = H _ { A } - H _ { B }$ . If

$$
- k t H _ { A } \preceq N \preceq k t H _ { A } , \quad \quad - k t H _ { B } \preceq N \preceq k t H _ { B } ,
$$

then arbitrary symmetric changes $E _ { A } , E _ { B }$ satisfy

$$
\begin{array} { r } { | \lambda _ { \operatorname* { m a x } } ( \mathsf { A } + E _ { A } ) - \lambda _ { \operatorname* { m a x } } ( \mathsf { B } + E _ { B } ) - ( a - b ) | \leq \| E _ { B } - E _ { A } \| + 2 k t \operatorname* { m a x } \{ \| E _ { A } \| , \| E _ { B } \| \} . } \end{array}\tag{7.11}
$$

Indeed, any unit top eigenvector u of $\mathsf { A } + E _ { A }$ obeys u<sup>⊤</sup> $H _ { A } u \leq 2 \| E _ { A } \|$ by the Rayleigh principle. Evaluate $\mathsf { B } + E _ { B } = \mathsf { A } + E _ { A } - ( a - b ) I + N + E _ { B } - E _ { A }$ at u for one bound. Evaluate the reversed equality at a unit top eigenvector of $\mathsf { B } + E _ { B }$ , using $v ^ { \top } H _ { B } v \leq 2 \| E _ { B } \|$ , for the other. This argument allows multiple top eigenvalues.

Apply this to the two base operators. Their top values are Ac and $\alpha ( t ) c ,$ where $\alpha ( t ) = A - D t + Q t ^ { 2 }$ On contrast group j,

$$
H _ { A } = A ( c - W _ { j j } ) I , \quad H _ { B } = \alpha ( t ) ( c - W _ { j j } ) I , \quad N = t ( D - Q t ) ( c - W _ { j j } ) I .
$$

Moreover $\alpha ( t ) \geq ( 1 - T ) ^ { 2 } A \colon$ its defining square contains the nonnegative summands $( 1 - t ) h ( k )$ and tc<sub>H</sub>. On the mean sector, $H _ { A } \succeq g c I$ and $H _ { B } \succeq ( g / 2 ) c I$ by Section 6. The mean part of N has norm at most $t ( D + m _ { * } ) c$ , where

$$
m _ { * } = | B _ { 1 } | + T B _ { 2 } + d ( | \tau _ { 1 } | + T | \tau _ { 2 } | ) , \qquad k = \operatorname * { m a x } \left\{ \frac { D } { ( 1 - T ) ^ { 2 } A } , \frac { 2 ( D + m _ { * } ) } { g } \right\} .
$$

Thus both comparisons required for (7.11) hold. A tied contrast diagonal makes all three corresponding deficits zero; no lower bound on a query diagonal gap is used. Equations (7.10) and (7.11) prove Theorem 7.4 with

$$
C = L _ { D } + 2 k L _ { M } , \qquad r _ { \mathrm { g a i n } } = \operatorname* { m i n } \{ r _ { 1 } , D / ( 4 C ) \} .
$$

Since $D - Q t \geq D / 2$ on [0, T], this radius gives the stated gain. All constants depend only on the fixed base, target, and budget.

## 7.8 Exact finite neighborhoods

The explicit constants above prove a positive local radius. Appendix G sharpens them by transporting the calibration Jacobian and the probability-weighted derivative energies from the base. This certifies a radius of 1/1000 in $\| H / \sqrt { \lambda _ { 0 } } \| _ { F }$ for three specified matched bases:

$$
( n , d , s ; r , \lambda _ { 0 } ) = ( 2 , 2 , 2 ; 2 / 3 , 5 / 3 ) , \quad ( 2 , 2 , 3 ; 5 / 4 , 5 5 / 3 1 ) , \quad ( 2 , 3 , 3 ; 3 / 2 , 3 6 3 / 1 0 1 ) .
$$

For every point in each closed ball, every nonzero PSD query and every $0 < t \leq T _ { s }$ , the two-sided modulus (7.6) and the positive gain $D t c / 4$ hold. Table 2 states the exact weight endpoints and strict rational gate bounds. Its proof gives all transport formulas and rational enclosure rules. The finite coeficient and energy records are reconstructible by the supplied ancillary scripts. These statements assume exact endpoint calibration. They certify these three neighborhoods, not approximate calibration software or prediction performance on a data distribution.

## 8 Scope and open questions

Established mean identities and duality give exact target calibration at a fixed row budget. The feasibility threshold and unique matched penalty provide the basis for covariance comparisons. The main analysis proves strict sector inequalities and resolves response equality in the stated design regimes. Complete response geometry then answers a stronger question than a matrix ceiling alone. In the stated exact regimes, it identifies the sharp risk and all responses that attain it.

The exact regimes have distinct boundaries. General LOO geometry gives an attainment test, but does not compute the smaller optimum when its envelope is strict. The balanced theorem covers all budgets from the dimension upward. Its lower-budget phase example prevents a blanket extension below that range. The ETF pair theorem characterizes flat queries. The triple theorem currently requires an isotropic query. Neither implies an all-budget ETF classification or optimality over designs.

These results leave both conceptual and extension questions. Can structural inequalities replace the longer coeficient certificates? At three ETF deletions, does flat query energy sufice beyond isotropic queries? The constructive conclusions suggest further questions. Can one classify general all-budget sharp risk without replication or frame structure? Which nonflat ETF queries admit residual maximizers? Can the exact mean-share phase be transported through full recalibration? Can useful query-uniform radii be obtained without retaining all base contrasts? The present results isolate the operators, gaps and equality conditions that these extensions must address.

## AI use statement

We used generative AI tools to assist with mathematical exploration, formulating and developing proofs, literature search, experiment design, implementation, and interpretation of results. We also used these tools to draft and edit text. The workflow included separate proof checks and independent numerical checking implementations. The author is responsible for the final claims, proofs, references, code, and presentation.

## References

So Anzai and Hideitsu Hino. Determinantal point process approximation under positive and negative dependence. arXiv:2607.18849, 2026.

S. Arati, P. Devaraj, and Shankhadeep Mondal. Optimal dual frames and dual pairs for probability modelled erasures using weighted average of operator norm and spectral radius. arXiv:2411.00502, 2024.

Bernhard G. Bodmann and Vern I. Paulsen. Frames, graphs and erasures. Linear Algebra and its Applications, 404:118–146, 2005.

Micha l Derezi´nski, Kenneth L. Clarkson, Michael W. Mahoney, and Manfred K. Warmuth. Minimax experimental design: Bridging the gap between statistical and worst-case approaches to least squares regression. In Proceedings of the Thirty-Second Conference on Learning Theory, volume 99 of PMLR, pages 1050–1069, 2019.

Micha l Derezi´nski and Manfred K. Warmuth. Reverse iterative volume sampling for linear regression. Journal of Machine Learning Research, 19(23):1–39, 2018.

Micha l Derezi´nski, Burak Bartan, Mert Pilanci, and Michael W. Mahoney. Debiasing distributed second order optimization with surrogate sketching and scaled regularization. In Advances in Neural Information Processing Systems, volume 33, 2020.

Micha l Derezi´nski, Feynman Liang, and Michael W. Mahoney. Bayesian experimental design using regularized determinantal point processes. In Proceedings of AISTATS, volume 108 of PMLR, pages 3197– 3207, 2020.

Matthew Fickus, John Jasper, Dustin G. Mixon, and Jesse D. Peterson. Hadamard equiangular tight frames. Applied and Computational Harmonic Analysis, 50:281–302, 2021.

Vivek K. Goyal, Jelena Kovaˇcevi´c, and Jonathan A. Kelner. Quantized frame expansions with erasures. Applied and Computational Harmonic Analysis, 10(3):203–233, 2001.

Hideitsu Hino and Keisuke Yano. From DPPs to k-DPPs: identifiability analysis via spectral decomposition. arXiv:2605.25526, 2026.

D. G. Horvitz and D. J. Thompson. A generalization of sampling without replacement from a finite universe. Journal of the American Statistical Association, 47(260):663–685, 1952.

Alex Kulesza and Ben Taskar. Determinantal point processes for machine learning. Foundations and Trends in Machine Learning, 5(2–3):123–286, 2012.

Thomas Lam, Alexander Postnikov, and Pavlo Pylyavskyy. Schur positivity and Schur log-concavity. American Journal of Mathematics, 129(6):1611–1622, 2007.

Fr´ed´eric Lavancier and Paul Rochet. A general procedure to combine estimators. Computational Statistics & Data Analysis, 94:175–192, 2016. doi:10.1016/j.csda.2015.08.001.

Chi-Kwong Li and Ren-Cang Li. A note on eigenvalues of perturbed Hermitian matrices. Linear Algebra and its Applications, 395:183–190, 2005.

Vincent Loonis and Xavier Mary. Determinantal sampling designs. Journal of Statistical Planning and Inference, 199:60–88, 2019. doi:10.1016/j.jspi.2018.05.005. Theorem numbers refer to the expanded author manuscript.

Kihun Rhee. A geometric phase boundary for volume-sampled linear readouts. arXiv:2608.26877, 2026.

Joachim Schreurs, Micha¨el Fanuel, and Johan A. K. Suykens. Ensemble kernel methods, implicit regularization and determinantal point processes. arXiv:2006.13701, 2020.

Martin J. Wainwright and Michael I. Jordan. Graphical models, exponential families, and variational inference. Foundations and Trends in Machine Learning, 1(1–2):1–305, 2008.

Yi Yu, Tengyao Wang, and Richard J. Samworth. A useful variant of the Davis–Kahan theorem for statisticians. Biometrika, 102(2):315–323, 2015.

## A The calibrated LOO coeficient-matrix ceiling

We prove Proposition 3.5. Keep its calibrated R, put $H = G + R$ , and define

$$
\begin{array} { r } { w = H ^ { - 1 } X ^ { \top } y , \qquad e = y - X w , \qquad J _ { R } = \Vert e \Vert ^ { 2 } + w ^ { \top } R w , \qquad a _ { i } = H ^ { - 1 / 2 } x _ { i } , \qquad c _ { i } = 1 - \Vert a _ { i } \Vert ^ { 2 } = 1 - \ell _ { i } . } \end{array}
$$

Here w is the full-data fit at the raw calibrated penalty R. It is distinct from the prescribed target $w _ { 0 }$ Also put $U = X H ^ { - 1 / 2 }$ and $V = H ^ { 1 / 2 } \Sigma _ { m - 1 } ( y ) H ^ { 1 / 2 }$ . Since $R \prec A _ { 0 }$ by (2.6), inverse order gives

$$
J _ { 0 } - J _ { R } = \boldsymbol { y } ^ { \top } \boldsymbol { X } \{ \boldsymbol { H } ^ { - 1 } - ( \boldsymbol { G } + \boldsymbol { A } _ { 0 } ) ^ { - 1 } \} \boldsymbol { X } ^ { \top } \boldsymbol { y } \geq 0 , \qquad J _ { 0 } = J _ { R } \iff \boldsymbol { X } ^ { \top } \boldsymbol { y } = 0 .\tag{A.1}
$$

The inverse diference is positive definite. The deletion identities (3.12), followed by subtraction of the mean square, give

$$
V = { \frac { 1 } { D } } \sum _ { i } { \frac { e _ { i } ^ { 2 } } { c _ { i } } } a _ { i } a _ { i } ^ { \top } - { \frac { 1 } { D ^ { 2 } } } ( \boldsymbol { U } ^ { \top } \boldsymbol { e } ) ( \boldsymbol { U } ^ { \top } \boldsymbol { e } ) ^ { \top } .
$$

Thus the exact whitened deficit is

$$
\begin{array} { l } { { J _ { 0 } \frac { t _ { \ell } } { D } I - V = \displaystyle \frac { t _ { \ell } } { D } ( J _ { 0 } - J _ { R } + w ^ { \top } R w ) I } } \\ { { \displaystyle \quad \quad + \frac { 1 } { D } \sum _ { i } e _ { i } ^ { 2 } \left( t _ { \ell } I - \frac { a _ { i } a _ { i } ^ { \top } } { c _ { i } } \right) + \frac { 1 } { D ^ { 2 } } ( U ^ { \top } e ) ( U ^ { \top } e ) ^ { \top } . } } \end{array}\tag{A.2}
$$

Each term is positive semidefinite: $t _ { \ell } > 0$ by full column rank, $c _ { i } > 0$ , and the sole nonzero eigenvalue of $a _ { i } a _ { i } ^ { \top } / c _ { i } { \mathrm { ~ i s ~ } } \ell _ { i } / ( 1 - \ell _ { i } ) \leq t _ { \ell }$ . Congruence proves the ceiling.

If $w \ne 0 ,$ , then $w ^ { \top } R w > 0$ , so the first term of (A.2) is positive definite and its kernel is zero. If $w = 0$ , then $X ^ { \top } y = 0 , e = y , J _ { 0 } = J _ { R } .$ , and the first and last terms vanish. For each active index $y _ { i } \neq 0$ the bracket has a nonzero kernel only if $\ell _ { i } / ( 1 - \ell _ { i } ) = t _ { \ell }$ . Its kernel is then $\operatorname { s p a n } ( a _ { i } )$ . A common nonzero kernel therefore requires every active row to have maximal leverage and all active $a _ { i }$ to lie on one line. Equal leverage on that line means equal nonzero lengths, so these vectors are signed copies of one vector $a .$ Multiplication by $H ^ { 1 / 2 }$ makes the original active rows signed copies of $v = H ^ { 1 / 2 } a$ . Conversely, these conditions make every active bracket vanish on $^ { a , }$ and their common kernel is exactly its span. Finally, if $\Delta = J _ { 0 } ( t _ { \ell } / D ) H ^ { - 1 } - \Sigma _ { m - 1 } ( y )$ , then

$$
\ker \Delta = H ^ { 1 / 2 } \ker ( H ^ { 1 / 2 } \Delta H ^ { 1 / 2 } ) .
$$

The raw coeficient query kernel is therefore precisely span(v), as claimed.

## B Exact balanced sign algebra

This appendix proves the strict sign for every positive raw penalty r. Calibration then chooses $\boldsymbol { r } = \boldsymbol { r } _ { s } ,$ with the same fixed target at each budget. In this appendix only, put $\rho = \mathbb { E } [ K _ { 1 } / ( r + K _ { 1 } ) ]$ ] and $b = n ( 1 - \rho )$ The induced target is $\lambda _ { 0 } = n ( 1 - \rho ) / \rho > 0$

We first prove $\tau > 0$ and reduce $\alpha > \beta$ to a reciprocal-moment inequality. The strict comparison has three cases: $n = 2$ in any dimension (Appendix B.3); $d = 2 , 3$ for the remaining replica counts $( \mathrm { A p p e n d i x ~ B . 4 } )$ ; and $n \geq 3 , d \geq 4 \ \mathrm { ( A p p e n d i x \ B . 5 ) }$ . In the last case, two reciprocal lower bounds cover the root domain. Polynomial certificates prove the root signs and the monotonicity needed to pass from the root to the actual mean. The LOO endpoint has a separate direct formula. The final deficit identifies all equality responses.

## B.1 Covariance and the sign of the cross term

Conditional on the counts, subsets in diferent groups are independent and uniform. If $T _ { j }$ is the sum of selected signed contrasts in group $j ,$ then

$$
\operatorname { \mathbb { E } } [ T _ { j } \mid K ] = 0 , \qquad \operatorname { V a r } ( T _ { j } \mid K ) = { \frac { K _ { j } ( n - K _ { j } ) } { n ( n - 1 ) } } E _ { j } , \qquad ( w _ { S } ) _ { j } = f ( K _ { j } ) q _ { j } + { \frac { T _ { j } } { r + K _ { j } } } .
$$

Total covariance and exchangeability prove (4.6), with the coeficients (4.3). In particular, $\beta$ is the transverse eigenvalue of the count-score covariance divided by $b ;$ it is not its diagonal variance divided by b.

We give a pairwise dependence argument that will also serve the mixture. Let $\textstyle { a _ { k } = { \binom { n } { k } } ( r + k ) }$ and

$$
g ( x ) = \sum _ { k = 0 } ^ { n } { a _ { k } x ^ { k } } = ( 1 + x ) ^ { n - 1 } [ r + ( r + n ) x ] .
$$

For $d \geq 3 ,$ , put $H _ { \ell } = [ x ^ { \ell } ] g ( x ) ^ { d - 2 }$ . This polynomial has only negative real roots and positive coeficients on its full support $0 , \ldots , ( d - 2 ) n$ . Newton’s coeficient inequality implies ordinary log-concavity of $\left( H _ { \ell } \right)$ The conditional mass of $K _ { 2 } = \ell$ given $K _ { 1 } = k$ is proportional to $a _ { \ell } H _ { s - k - \ell }$ . For adjacent possible $k , k + 1$ the ratio of the new to old conditional masses is proportional to

$$
{ \frac { H _ { s - k - 1 - \ell } } { H _ { s - k - \ell } } } .
$$

Log-concavity makes this ratio nonincreasing in ℓ. Both support endpoints max $( 0 , s - k - ( d - 2 ) n )$ and min $( n , s - k )$ move down as k increases. At a new lower point the ratio is infinite; at a lost upper point it is zero. Thus the ordering includes the support boundaries: the conditional law of $K _ { 2 }$ decreases stochastically with k. For $d = 2 , K _ { 2 } = s - K _ { 1 }$ gives the same conclusion directly.

These conditional laws are distinct because their means are $( s - k ) / ( d - 1 )$ . Consequently, for a strictly increasing function $v , \mathbb { E } [ v ( K _ { 2 } ) \mid K _ { 1 } = k ]$ strictly decreases on the possible k. For any strictly increasing $u ,$ the independent-copy identity

$$
2 \operatorname { C o v } ( u ( K _ { 1 } ) , v ( K _ { 2 } ) ) = \mathbb { E } [ ( u ( K _ { 1 } ) - u ( K _ { 1 } ^ { \prime } ) ) ( \mathbb { E } [ v ( K _ { 2 } ) \mid K _ { 1 } ] - \mathbb { E } [ v ( K _ { 2 } ) \mid K _ { 1 } ^ { \prime } ] ) ]
$$

is strictly negative when $1 \leq s < n d$ , since $K _ { 1 }$ is nonconstant. Here the prime denotes an independent copy. Taking $u = v = f$ proves $\tau > 0$ . This argument uses the actual determinant count law.

## B.2 Deletion identities and a reciprocal reduction

Write $M = d ( n - 1 )$ and define

$$
\begin{array} { l } { { Z = [ x ^ { s } ] g ( x ) ^ { d } , } } \\ { { Q = [ x ^ { s } ] ( 1 + x ) ^ { M + 1 } [ r + ( r + n ) x ] ^ { d - 1 } , } } \\ { { T = [ x ^ { s - 1 } ] ( 1 + x ) ^ { M } [ r + ( r + n ) x ] ^ { d - 2 } , } } \end{array}
$$

$$
A = [ x ^ { s - 1 } ] g ( x ) ^ { d - 1 } \sum _ { j = 0 } ^ { n - 2 } { \binom { n - 2 } { j } } { \frac { x ^ { j } } { r + 1 + j } } .
$$

All four are positive for d $\leq s <$ nd. Direct deletion of determinant factors gives

$$
\alpha = \frac { A } { Z } , \qquad \beta = \frac { n T - ( n - 1 ) A } { ( n + r ) Q } .\tag{B.1}
$$

For details, put ${ \cal R } _ { 1 } = Z \mathbb { E } f ( K _ { 1 } ) , C _ { 1 2 } = Z \mathbb { E } [ f ( K _ { 1 } ) f ( K _ { 2 } ) ]$ , and $D _ { 1 } = Z \mathbb { E } [ K _ { 1 } / ( r + K _ { 1 } ) ^ { 2 } ]$ . The binomial identities give

$$
Z - R _ { 1 } = r Q , \qquad ( n + r ) D _ { 1 } - R _ { 1 } = n ( n - 1 ) A , \qquad n R _ { 1 } - ( n + r ) C _ { 1 2 } = n ^ { 2 } r T .
$$

The transverse covariance numerator is $R _ { 1 } - r D _ { 1 } - C _ { 1 2 }$ , while $b = n r Q / Z$ . These prove (B.1).

For an auxiliary proof experiment, select $s - 1$ items with product weights from nd − 2 items: M ordinary items of weight one and $d - 2$ exceptional items of weight $( r + n ) / r$ . Let U and V count disjoint ordinary blocks of sizes $n - 2$ and n. Expanding the coeficient generating functions, including the two distinguished blocks, gives

$$
{ \frac { A } { T } } = \mathbb { E } { \frac { r + V } { r + 1 + U } } .\tag{B.2}
$$

For example the factor for the U block after division by $r { + } 1 { + } U$ is $\textstyle \sum _ { j = 0 } ^ { n - 2 } { \binom { n - 2 } { j } } x ^ { j } / ( r + 1 + j )$ ; multiplication by $r + V$ on the other block gives $g ( x )$ , leaving $g ( x ) ^ { d - 2 }$ from the remaining ordinary and exceptional blocks. The normalizing polynomial is the one defining T. Hence

$$
\alpha > \beta \quad \Longleftrightarrow \quad A \{ ( n + r ) Q + ( n - 1 ) Z \} > n Z T .\tag{B.3}
$$

## B.3 Two replicas in any dimension

Here $n = 2$ . We first record the coeficient comparison needed for this case. For $a \ge 1 , D \ge 1$ , define

$$
b _ { j } = [ x ^ { j } ] ( 1 + x ) ^ { D - 1 } ( 1 + a x ) ^ { D } , \qquad c _ { j } = [ x ^ { j } ] ( 1 + x ) ^ { D } ( 1 + a x ) ^ { D + 1 } .
$$

Then, for $0 \le k \le D$

$$
b _ { k } c _ { k } \geq b _ { k - 1 } c _ { k + 1 } .\tag{B.4}
$$

Here and below coeficients outside support are zero. We include the argument and its Schur-function foundation explicitly. At a positive alphabet, the two sequences

$$
A _ { j } = e _ { j } ^ { 2 } - e _ { j - 1 } e _ { j + 1 } = s _ { ( 2 ^ { j } ) } , \qquad B _ { j } = e _ { j } e _ { j - 1 } - e _ { j - 2 } e _ { j + 1 } = s _ { ( 2 ^ { j - 1 } , 1 ) }
$$

are log-concave in $j .$ This is the specialization of the rounded-partition Schur log-concavity theorem (Lam et al., 2007, Theorem 11), applied to $( j - 1 , j - 1 ) , ( j + 1 , j + 1 )$ and $( j - 1 , j - 2 ) , ( j + 1 , j )$ , followed by the Schur involution and the dual Jacobi–Trudi identity. The second assertion is used for $j \geq 2$ , with $B _ { 0 } = 0$

Put $q = \sqrt { a } , L = D - 1$ , and let $e _ { j }$ be the coeficients of $( 1 + q x ) ^ { L } ( 1 + q ^ { - 1 } x ) ^ { L }$ . This reciprocal alphabet gives $e _ { j } = e _ { 2 L - j } , A _ { j } = A _ { 2 L - j }$ , and $B _ { j } = B _ { 2 L + 1 - j }$ . The positive parts of both sequences are therefore nondecreasing up to their midpoint, by log-concavity and symmetry. Set $\boldsymbol { v } _ { j } = \boldsymbol { e } _ { j } + q \boldsymbol { e } _ { j - 1 }$ and $\Delta _ { j } ( v ) = v _ { j } ^ { 2 } - v _ { j - 1 } v _ { j + 1 }$ . Expansion gives

$$
\Delta _ { j } ( v ) = A _ { j } + q B _ { j } + q ^ { 2 } A _ { j - 1 } .
$$

Thus $\Delta _ { j } ( v ) \geq \Delta _ { j - 1 } ( v )$ for $1 \leq j \leq L ;$ at $j = L + 1$ the diference is $( q ^ { 2 } - 1 ) ( A _ { L } - A _ { L - 1 } ) \geq 0$ . The case $L = 0$ uses $A _ { 0 } = 1 , A _ { - 1 } = B _ { 0 } = B _ { 1 } = 0$ . Since $b _ { j } = a ^ { j / 2 } v _ { j }$ , this gives $\Delta _ { k } ( b ) \geq a \Delta _ { k - 1 } ( b )$ . Finally, $c _ { j } = b _ { j } + ( 1 + a ) b _ { j - 1 } + a b _ { j - 2 }$ implies $b _ { k } c _ { k } - b _ { k - 1 } c _ { k + 1 } = \Delta _ { k } ( b ) - a \Delta _ { k - 1 } ( b )$ . The case $k = 0$ is immediate, completing (B.4).

For the application define $E _ { j } ( M , D ) = [ x ^ { j } ] ( 1 + x ) ^ { M } ( 1 + a x ) ^ { D }$ , and put

$$
\begin{array} { r } { P _ { * } = E _ { s - 2 } ( d - 1 , d - 2 ) , \quad T _ { * } = E _ { s - 1 } ( d , d - 2 ) , \quad U _ { * } = E _ { s - 1 } ( d , d - 1 ) , \quad Q _ { * } = E _ { s } ( d + 1 , d - 1 ) . } \end{array}
$$

These numbers are positive. Coeficient complementation says $E _ { j } ( M , D ) = a ^ { j - M } E _ { M + D - j } ( D , M )$ . With $t = 2 d - s , k = t - 1 , D = d - 1$ , and $\ell = d - 1 - t$ , it gives

$$
P _ { * } = a ^ { \ell } b _ { k } , \quad T _ { * } = a ^ { \ell } ( b _ { k } + a b _ { k - 1 } ) , \quad U _ { * } = a ^ { \ell } c _ { k + 1 } , \quad Q _ { * } = a ^ { \ell } ( c _ { k + 1 } + a c _ { k } ) .
$$

Consequently $P _ { * } Q _ { * } \ge U _ { * } T _ { * }$ by (B.4). Now take $a = ( r + 2 ) / r > 1$ . The deletion quantities obey

$$
Z = r ^ { d } [ Q _ { * } + ( a - 1 ) U _ { * } ] , \quad Q = r ^ { d - 1 } Q _ { * } , \quad T = r ^ { d - 2 } T _ { * } , \quad A = \frac { r ^ { d - 1 } [ T _ { * } + ( a - 1 ) P _ { * } ] } { r + 1 } .
$$

The diference in (B.3) is

$$
\frac { 2 r ^ { 2 d - 2 } ( a - 1 ) } { a + 1 } \{ a ( P _ { * } Q _ { * } - U _ { * } T _ { * } ) + P _ { * } Q _ { * } + ( a - 1 ) P _ { * } U _ { * } \} > 0 .
$$

This proves the strict sign for every $d \leq s < 2 d .$

## B.4 Two and three dimensions

We use a simple reciprocal bound. If $0 \le U \le L , \mu = \mathbb { E } U < L , v = \mathrm { V a r } U .$ and $a > 0$ , put $u _ { 0 } =$ $\mu - v / ( L - \mu )$ . Tilting the law by $( L - U ) / ( L - \mu )$ makes its mean $u _ { 0 } \in [ 0 , L ]$ . The identity

$$
\frac { 1 } { a + U } = \frac { 1 } { a + L } + \frac { L - U } { ( a + L ) ( a + U ) }
$$

and Jensen’s inequality under this tilted law imply

$$
\mathbb { E } \frac { 1 } { a + U } \geq \frac { 1 } { a + L } + \frac { L - \mu } { ( a + L ) ( a + u _ { 0 } ) } .\tag{B.5}
$$

For $d = 2 .$ , divide the deletion normalizers by $\binom { 2 n } { s }$ and write

$$
z = r ^ { 2 } + r s + \frac { n s ( s - 1 ) } { 2 ( 2 n - 1 ) } , \quad q = r + s / 2 , \quad t = \frac { s ( 2 n - s ) } { 2 n ( 2 n - 1 ) } , \quad h = ( n + r ) q + ( n - 1 ) z .
$$

The auxiliary U is hypergeometric with population $2 n - 2 .$ marked count $n - 2 ,$ , and sample size $s - 1$ and $V = s - 1 - U$ . Its mean and variance are

$$
\mu = \frac { ( n - 2 ) ( s - 1 ) } { 2 n - 2 } , \qquad v = \frac { ( s - 1 ) ( n - 2 ) n ( 2 n - 1 - s ) } { ( 2 n - 2 ) ^ { 2 } ( 2 n - 3 ) } .
$$

Use $L = s - 1$ in (B.5). Then $u _ { 0 } = ( n - 2 ) ( s - 2 ) / ( 2 n - 3 )$ and

$$
\mathbb { E } \frac { 1 } { r + 1 + U } \geq L _ { 0 } : = \frac { 1 } { r + s } + \frac { n ( s - 1 ) } { 2 ( n - 1 ) ( r + s ) ( r + 1 + u _ { 0 } ) } .
$$

Put $R _ { 0 } = ( 2 r + s ) L _ { 0 } - 1$ . By (B.2), $A / T \geq R _ { 0 }$ and $\alpha - \beta = t \{ ( A / T ) h - n z \} / [ z ( n + r ) q ]$ . Set $u = n - 2 .$ $w = s - 2$ , and $\mathscr { D } = 4 ( n - 1 ) ( 2 n - 1 ) ( r + s ) [ ( 2 n - 3 ) r + ( n - 2 ) s + 1 ]$ . Direct expansion gives $\mathcal { D } ( R _ { 0 } h - n z ) =$ $\textstyle \sum _ { i = 0 } ^ { 3 } C _ { i } r ^ { i }$ , where

$$
\begin{array} { r l } & { C _ { 0 } = n ^ { 2 } s ^ { 2 } ( s - 1 ) [ 2 u ^ { 2 } + 5 u + 2 + ( u + 1 ) w ] , } \\ & { C _ { 1 } = n s [ 4 u ^ { 3 } + 3 0 u ^ { 2 } + 5 0 u + 1 8 + ( 8 u ^ { 3 } + 5 0 u ^ { 2 } + 8 2 u + 3 3 ) w + ( 6 u ^ { 2 } + 1 6 u + 9 ) w ^ { 2 } ] , } \\ & { C _ { 2 } = 2 [ 1 2 u ^ { 3 } + 5 8 u ^ { 2 } + 7 8 u + 2 6 + ( 4 u ^ { 4 } + 4 0 u ^ { 3 } + 1 2 9 u ^ { 2 } + 1 5 2 u + 5 1 ) w + ( 6 u ^ { 3 } + 2 9 u ^ { 2 } + 4 1 u + 1 6 ) w ^ { 2 } ] , } \\ & { C _ { 3 } = 2 ( 2 n - 3 ) ( 2 n - 1 ) [ 2 + ( u + 3 ) w ] . } \end{array}
$$

All are positive for $n , s \geq 2 .$ . All cleared factors are positive and $t > 0$ for $s < 2 n$ , proving the sign, including the deterministic auxiliary endpoint.

For $d = 3 ,$ , put $M = 3 ( n - 1 )$ . Conditional on k selected ordinary items in the auxiliary experiment, U is hypergeometric on the ordinary population. Its tilted mean is $( n - 2 ) ( k - 1 ) / ( M - 1 )$ . Thus for $k \geq 1$ 2

$$
\begin{array} { r l } & { L _ { k } = \displaystyle \frac { 1 } { r + 1 + k } + \frac { k ( M - n + 2 ) } { M ( r + 1 + k ) [ r + 1 + ( n - 2 ) ( k - 1 ) / ( M - 1 ) ] } , } \\ & { \ell _ { k } = \frac { [ r ( M + 2 ) + n ( k + 1 ) ] L _ { k } - n } { M - n + 2 } \le { \mathbb E } \left[ \frac { r + V } { r + 1 + U } \bigg | k \right] . } \end{array}
$$

The second formula follows from $\operatorname { \mathbb { E } } [ V \mid U , k ] = n ( k - U ) / ( M - n + 2 )$ . Divide $Z , Q , T$ by $\binom { 3 n } { s }$ and write

$$
\begin{array} { r l } & { z = r ^ { 3 } + s r ^ { 2 } + \frac { n s ( s - 1 ) r } { 3 n - 1 } + \frac { n ^ { 2 } s ( s - 1 ) ( s - 2 ) } { 3 ( 3 n - 1 ) ( 3 n - 2 ) } \cdot } \\ & { q = r ^ { 2 } + 2 s r / 3 + \frac { n s ( s - 1 ) } { 3 ( 3 n - 1 ) } , } \\ & { t _ { 0 } = \frac { s ( 3 n - s ) } { 3 n ( 3 n - 1 ) } , \qquad t _ { 1 } = \frac { s ( s - 1 ) ( 3 n - s ) } { 3 n ( 3 n - 1 ) ( 3 n - 2 ) } , } \\ & { t = r t _ { 0 } + n t _ { 1 } , \qquad h = ( n + r ) q + ( n - 1 ) z , } \\ & { a = t _ { 1 } \left[ ( r + n ) \ell _ { s - 2 } + \frac { r ( 3 n - s - 1 ) } { s - 1 } \ell _ { s - 1 } \right] . } \end{array}
$$

There is one exceptional item. Its two possible selection states have unnormalized masses $\left( r + n \right) \left( { \cal M } _ { \bf { s } - \bf { 2 } } \right)$ and $r \binom { M } { s - 1 }$ . The conditional bounds therefore give $A / ( \boldsymbol { \mathsf { \Sigma } } _ { s } ^ { 3 n } ) \geq a$ . At $s = 3 n - 1$ , the second mass is zero and its impossible conditional component is omitted.

Define

$$
\begin{array} { c } { \displaystyle \mathcal { D } = 2 7 n ( n - 1 ) ( 3 n - 2 ) ^ { 2 } ( 3 n - 1 ) ^ { 2 } ( r + s ) ( r + s - 1 ) } \\ { \displaystyle \cdot [ ( 3 n - 4 ) r + ( n - 2 ) s + n ] [ ( 3 n - 4 ) r + ( n - 2 ) s + 2 ] , } \\ { P ( n , s , r ) = \displaystyle \frac { \mathcal { D } ( a h - n z t ) } { s ( 3 n - s ) } . } \end{array}
$$

Exact rational cancellation makes P an integer polynomial. Expansion of $P ( 2 + u , 3 + w , r )$ has 342 nonzero coeficients, all positive, and constant 9216. This is the small-dimension polynomial certificate. It is reconstructed from the displayed rational formulas by clearing denominators, exact division, substitution, and coeficient collection; no sampled values of $n , s , r$ enter the check. The public anc/balanced/three\_ coordinate\_certificate.py checker verifies the full formal identity and every coeficient. For additional fingerprints,

$$
\begin{array} { l } { { [ r ^ { 0 } ] P = n ^ { 4 } s ^ { 2 } ( s - 2 ) ( s - 1 ) ^ { 3 } ( n s + n - 2 s ) ( 3 n ^ { 2 } + 2 n s - 1 0 n - 2 s + 6 ) , } } \\ { { [ r ^ { 7 } ] P = 3 ( n - 1 ) ( 3 n - 4 ) ^ { 2 } ( 3 n - 2 ) ( 3 n - 1 ) ( 3 n s - 6 n + 2 s ) . } } \end{array}
$$

These two coeficients do not replace the complete coeficient check. All factors in D and $s ( 3 n - s )$ are positive in $3 \leq s < 3 n$ . Hence $a h - n z t > 0$ , and

$$
\alpha - \beta \geq \frac { a h - n z t } { z ( n + r ) q } > 0 .
$$

## B.5 The general domain: a variance-root split

It remains to prove the sign for $n \geq 3 , d \geq 4$ . First let $d \leq s \leq n d - 2$ and put

$$
M = d ( n - 1 ) , \quad D = d - 2 , \quad N = M + D , \quad u = s - d + 1 , \quad L = M - u .
$$

In the auxiliary experiment, let H count unselected exceptional items. Its mass is proportional to

$$
{ \binom { D } { k } } { \binom { M } { u + k } } r ^ { k } ( r + n ) ^ { D - k } .
$$

The selected ordinary count is $u + H$ . Put $h = \mathbb { E } H$ . Detailed balance between adjacent masses gives

$$
r D L = r N h + n u h + n \mathbb { E } H ^ { 2 } .
$$

Distinct exceptional inclusion indicators have nonpositive covariance. Indeed, deleting the two items leaves three elementary symmetric coeficients $a , b , c ;$ their covariance is a positive multiple of $a c - b ^ { 2 } \leq 0$ by Newton’s inequality. Complements have the same covariance. Therefore Var $H \leq h ( 1 - h / D )$ , and

$$
q ( h ) : = n ( D - 1 ) h ^ { 2 } + D [ r N + n ( u + 1 ) ] h - r D ^ { 2 } L \geq 0 .\tag{B.6}
$$

Here $0 < h < D$ and $L > 0$ . The polynomial q is strictly increasing on the nonnegative line and has a unique positive root $\eta ,$ with $h \geq \eta > 0$ . Since $q ( D ) = D ^ { 2 } [ n ( D + u ) + r ( N - L ) ] > 0$ , also $\eta < D$

The following polynomials use h as a formal variable:

$$
\begin{array} { r l } & { Q _ { n } = r m ( m - 1 ) + n s ( M + 1 ) + n m h , \qquad m = n d , \quad s = u + d - 1 , } \\ & { Z _ { n } = r Q _ { n } + n s [ r ( m - 1 ) + n ( u + h ) ] , } \\ & { H _ { n } = ( n + r ) Q _ { n } + ( n - 1 ) Z _ { n } , } \\ & { \quad B = M r + n ( u + h ) , } \\ & { C = M r ( s - 1 ) + ( u + h ) [ n ( u - 1 ) - r ( d - 1 ) ] . } \end{array}
$$

At the actual mean they satisfy

$$
{ \frac { Q } { T } } = { \frac { Q _ { n } } { s ( L + 1 ) } } , \quad { \frac { Z } { T } } = { \frac { Z _ { n } } { s ( L + 1 ) } } , \quad \mathbb { E } ( r + V ) = { \frac { B } { M } } , \quad \mathbb { E } [ ( r + V ) U ] = { \frac { ( n - 2 ) C } { M ( M - 1 ) } } .\tag{B.7}
$$

Here is a derivation that keeps the second moment explicit. If $e _ { j }$ are the elementary symmetric coeficients of the auxiliary weights, counting selected inverse weights and unselected weights gives

$$
\frac { e _ { s } } { e _ { s - 1 } } = { \frac { r L + n h } { r s } } , \qquad \frac { e _ { s - 2 } } { e _ { s - 1 } } = \frac { r ( s - 1 ) + n ( u + h ) } { ( r + n ) ( L + 1 ) } .
$$

Multiplication by the two missing linear factors gives $Q / T$ and $Z / T$ . Conditional ordinary factorial moments give

$$
\mathbb { E } [ ( r + V ) U ] = \frac { n - 2 } { M ( M - 1 ) } \{ r ( M - 1 ) ( u + h ) + n \mathbb { E } [ ( u + H ) ( u + H - 1 ) ] \} .
$$

The drift identity above reduces the braces to $C .$ . This proves (B.7) without replacing a random variable by its mean inside a reciprocal.

Let $\theta = n Z _ { n } / H _ { n }$ . The sign condition (B.3) is now $A / T > \theta$ . For integer $U \geq 0$

$$
{ \frac { 1 } { r + 1 + U } } - { \frac { r + 2 - U } { ( r + 1 ) ( r + 2 ) } } = { \frac { U ( U - 1 ) } { ( r + 1 + U ) ( r + 1 ) ( r + 2 ) } } \geq 0 .
$$

Multiplication by $r + V$ gives the lattice bound below. Weighted Cauchy gives the other bound:

$$
\frac { A } { T } \geq L _ { 1 } = \frac { ( r + 2 ) ( M - 1 ) B - ( n - 2 ) C } { M ( M - 1 ) ( r + 1 ) ( r + 2 ) } ,
$$

$$
\frac { A } { T } \geq L _ { C } = \frac { ( M - 1 ) B ^ { 2 } } { M [ ( r + 1 ) ( M - 1 ) B + ( n - 2 ) C ] } .
$$

At the actual mean the latter denominator is positive because it is a positive multiple of $\mathbb { E } [ ( r + V ) ( r +$ $1 + U ) ]$ ]. Define polynomial residuals

$$
P ( h ) = [ ( r + 2 ) ( M - 1 ) B - ( n - 2 ) C ] H _ { n } - n ( r + 1 ) ( r + 2 ) M ( M - 1 ) Z _ { n } ,\tag{B.8}
$$

$$
S ( h ) = ( M - 1 ) B ^ { 2 } H _ { n } - n M Z _ { n } [ ( r + 1 ) ( M - 1 ) B + ( n - 2 ) C ] .\tag{B.9}
$$

At the actual mean, $P ( h ) > 0$ or $S ( h ) > 0$ proves the desired sign.

Put $t = \eta / D , z = u + \eta .$ , and $\delta = M ( 1 - t ) - z$ . Substitution into $q ( \eta ) = 0$ gives

$$
r = \frac { n t ( z + 1 - t ) } { \delta } , \qquad 1 < z < M , \quad 0 < t \leq \frac { z - 1 } { D } , \quad t < 1 - \frac { z } { M } .\tag{B.10}
$$

Indeed $0 < t < 1 , z > 1$ , and the numerator is positive, so $\delta > 0$ . The two upper bounds meet at $z _ { c } = M ( d - 1 ) / N$ , and $1 < z _ { c } < d < M$ because $z _ { c } - 1 = D ( M - 1 ) / N$ and $d - z _ { c } = d ( n + d - 3 ) / N$

## B.6 Finite polynomial certificate and full domain coverage

The exact certificate proves four statements: P increases strictly on $h \geq 0$ when $1 \leq u \leq d ; S$ increases strictly at fixed u on $0 \leq h < D$ , $\iota + h \geq d ; P ( \eta ) > 0$ when $z < d ;$ and $S ( \eta ) > 0$ when $z \geq d .$ . The following formulas specify the entire certificate and make it independently reconstructible.

First substitute $h = D t , u = z - D t$ , and the value of r in (B.10), and define

$$
\begin{array} { r } { P ^ { \ast } ( n , d , z , t ) = \delta ^ { 3 } P ( D t ) / n ^ { 3 } , \qquad S ^ { \ast } ( n , d , z , t ) = \delta ^ { 3 } S ( D t ) / n ^ { 4 } . } \end{array}
$$

Exact expansion verifies both divisions and gives integer polynomials of total degree seven in $( z , t )$ . Their individual z, t degrees are also seven. For the following three rational cells write $z = z _ { n } / z _ { d } , t = t _ { n } / t _ { d } .$ with $v , w \geq 0 ;$

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1> $z _ { n }$ </td><td rowspan=1 colspan=1> $z _ { d }$ </td><td rowspan=1 colspan=1> $t _ { n }$ </td><td rowspan=1 colspan=1> $t _ { d }$ </td><td rowspan=1 colspan=1>kernel</td></tr><tr><td rowspan=1 colspan=1>LowUpperHigh</td><td rowspan=1 colspan=1> $N ( 1 + w ) + D ( M - 1 ) w$  $M ( d - 1 ) + d N w$  $d [ 1 + ( n - 1 ) w ]$ </td><td rowspan=1 colspan=1> $N ( 1 + w )$  $N ( 1 + w )$  $1 + w$ </td><td rowspan=1 colspan=1>(M − 1)wv[M N(1 + w) − M(d − 1) − dNw]v(n − 2)v</td><td rowspan=1 colspan=1> $N ( 1 + w ) ( 1 + v )$  $M N ( 1 + w ) ( 1 + v )$  $( n - 1 ) ( 1 + w ) ( 1 + v )$ </td><td rowspan=1 colspan=1> $\overline { { P ^ { * } } }$  $P ^ { * }$  $S ^ { * }$ </td></tr></table>

Every denominator is positive. In each cell expand $t _ { d } ^ { 7 } \mathrm { k e r n e l } ( n , d , z _ { n } / z _ { d } , t _ { n } / t _ { d } )$ and then put $n = 3 + x , d =$ $4 + y .$ To do this without rational arithmetic, write $t _ { d } = z _ { d } \ell .$ Each kernel monomial $a _ { i j } ( n , d ) z ^ { i } t ^ { j }$ contributes

$$
a _ { i j } ( n , d ) ( z _ { n } \ell ) ^ { i } t _ { n } ^ { j } t _ { d } ^ { 7 - i - j } ;
$$

all exponents are nonnegative by the total-degree check. The complete coeficient checks are summarized here.
<table><tr><td>polynomial</td><td>nonnegative variables</td><td>nonzero coefficients</td><td>constant</td></tr><tr><td> $\overline { { [ h ] P , ~ d = u + v } }$   $[ h ^ { 2 } ] P , \ d = u + v$ </td><td> $\overline { { ( n - 3 , u - 1 , v , r ) } }$ </td><td>234</td><td>432</td></tr><tr><td rowspan="3"> $( 1 + v ) ^ { 2 } \partial _ { h } S , \ u = d + z - h , \ h = D v / ( 1 + v ) \big | ( n - 3 , d - 4 , r , z , v )$  Low cell</td><td> $( n - 3 , u - 1 , v , r )$ </td><td>108</td><td>270</td></tr><tr><td></td><td>840</td><td>412776</td></tr><tr><td>(n − 3, d − 4, v, w)</td><td>12144</td><td>576240000000</td></tr><tr><td>Upper cell High cell</td><td> $( n - 3 , d - 4 , v , w )$  (n − 3, d − 4, v, w)</td><td>27907 5256</td><td>2953583794675777536 35782656</td></tr></table>

The first two rows prove monotonicity of $P ,$ and the third proves monotonicity of S. The Low and Upper cells prove the root sign of $P ;$ the High cell proves the root sign of S. Every nonzero coeficient is a positive integer. For the third row, diferentiate at fixed $u , r$ before substituting. The first two rows use $s = u + d - 1$ throughout. The checker expands the defining polynomials, compares the entire coeficient dictionaries (including absent monomials), and verifies exact division by the stated factors. Thus the 46,489 entries certify polynomial identities and positivity on unbounded nonnegative orthants. Testing the signs of supplied entries without reconstructing the polynomials would not sufice.

For clarity, the rational cells cover the whole root domain. For $1 < z < z _ { c }$ with $t < ( z - 1 ) / D$ , their inverse coordinates are

$$
w = \frac { z - 1 } { z _ { c } - z } , \qquad v = \frac { t } { ( z - 1 ) / D - t } .
$$

For $z _ { c } \leq z < d .$ , use $w = ( z - z _ { c } ) / ( d - z )$ and $v = t / ( 1 - z / M - t )$ . For $d \leq z < M$ , use $w = ( z - d ) / ( M - z )$ and the same expression for v. The seam $z = z _ { c }$ belongs to Upper at $w = 0$ , and $z = d$ belongs to High at $w = 0$ . No infinite-w argument is needed.

The remaining face is $t = ( z - 1 ) / D$ , or $u = 1$ . It necessarily has $z < z _ { c }$ and is the Low cell’s $v  \infty$ face. The Low polynomial has degree seven in v. Its coeficient of $v ^ { 7 }$ has 1,504 positive coeficients in $( n - 3 , d - 4 , w )$ and constant 576240000000. Dividing by $t _ { d } ^ { 7 }$ and taking this limit preserves a strictly positive value. The other upper face $t = 1 - z / M$ is impossible for finite $r > 0$ because $\delta > 0$

We can now transport the residual signs. If $u + \eta < d ,$ the root certificate gives $P ( \eta ) > 0$ , and $u \leq d$ makes P increasing on $[ \eta , h ]$ . Hence $P ( h ) > 0$ . If $u + \eta \ge d .$ every $x \in [ \eta , h ]$ has $0 \leq x < D$ and $u + x \geq d .$ The third coeficient row makes S increasing there, so $S ( h ) \geq S ( \eta ) > 0$ . Only now do we divide by the positive actual-mean Cauchy denominator. Both cases prove $\alpha > \beta .$

At the remaining endpoint $s = n d - 1$ , the omitted row is uniform, so $K _ { 1 } = n - 1$ with probability $1 / d$ and otherwise $K _ { 1 } = n$ . Direct substitution into the coeficient definitions gives

$$
\alpha = \frac { 1 } { d n ( r + n - 1 ) ^ { 2 } } , \qquad \frac { \beta } { \alpha } = \frac { d r ( r + n - 1 ) } { ( r + n ) [ d ( r + n - 1 ) + 1 ] } < 1 .
$$

This deals with the endpoint without a limiting root argument.

## B.7 Certificate status and geometric conclusion

The public files anc/balanced/sign\_certificate.py and anc/balanced/coefficient\_certificate. json reconstruct and check all six general-domain polynomial identities using exact integer arithmetic. The separate anc/balanced/three\_coordinate\_certificate.py checker reconstructs the 342- coeficient identity above. Both checks are finite algebraic proofs of parameter identities; original-row or count enumerations in the checkers are additional finite corroboration. The formulas in this appendix specify what those programs verify, including all boundary cases and signs of removed factors.

The union of the two-replica, two- and three-dimensional, and general arguments is exactly $n , d \geq 2 .$ $d \leq s < n d , r > 0$ . At each calibrated $r _ { s }$ , the same target gives the common b and $J _ { 0 }$ of (4.1). The deficit (4.9) now proves the value and complete equality space in Theorem 4.1. Indeed $\beta \geq 0$ and $\alpha c - \beta W _ { j j } \geq c ( \alpha - \beta ) > 0$ . Nonzero contrasts in a maximizing group exist because $n \geq 2$ . This also shows that no further equality vectors appear for singular queries, nonzero of-diagonal entries, or tied maximal diagonals.

## B.8 Short algebra for the quantitative margin

Use the notation of Proposition 4.2. Put $E = d ( n ^ { 2 } - 2 ) - 2 n + 2 > 0 ;$ ; indeed $E \geq 2 ( n ^ { 2 } - n - 1 ) > 0$ Direct expansion of its displayed quadratic $P ( h )$ gives

$$
p _ { 2 } = d n ^ { 2 } ( n r + 2 n - 1 ) [ 2 n ( M - 1 ) + r E ] ,
$$

$$
p _ { 1 } = d n [ 2 n ^ { 2 } ( 3 n - 1 ) ( M - 1 ) + n r B _ { 1 } + r ^ { 2 } A _ { 1 } + n r ^ { 3 } ( n d - 1 ) E ] .
$$

For $a = n - 2 \geq 0$ and $e = d - 2 \geq 0$ , the two remaining polynomials are

$$
\begin{array} { l } { B _ { 1 } = 2 e ^ { 2 } ( a + 1 ) ( 2 a ^ { 2 } + 8 a + 7 ) + e ( 2 3 a ^ { 3 } + 1 0 3 a ^ { 2 } + 1 4 2 a + 5 8 ) } \\ { \qquad + 2 ( 1 5 a ^ { 3 } + 5 9 a ^ { 2 } + 7 1 a + 2 2 ) , } \\ { A _ { 1 } = e ^ { 2 } n ( 2 a + 3 ) ( 2 a ^ { 2 } + 8 a + 5 ) + e ( 1 8 a ^ { 4 } + 1 2 2 a ^ { 3 } + 2 9 0 a ^ { 2 } + 2 8 1 a + 8 8 ) } \\ { \qquad + 2 ( 1 0 a ^ { 4 } + 6 1 a ^ { 3 } + 1 3 1 a ^ { 2 } + 1 1 3 a + 2 9 ) . } \end{array}
$$

Every displayed factor is nonnegative and both constant terms are positive, so $p _ { 1 } , p _ { 2 } > 0$ . The constant term is

$$
P ( 0 ) = d n ( M - 1 ) [ 2 n ^ { 3 } - n C _ { 1 } r - C _ { 2 } r ^ { 2 } - ( n d - 1 ) C _ { 3 } r ^ { 3 } ] ,
$$

where

$$
\begin{array} { l } { { C _ { 1 } = 3 d ^ { 2 } n ^ { 2 } - 5 d ^ { 2 } n + 2 d ^ { 2 } - 6 d n ^ { 2 } + 5 d n - 2 d - 2 n ^ { 2 } + 4 n - 2 , } } \\ { { C _ { 2 } = 4 d ^ { 2 } n ^ { 3 } - 7 d ^ { 2 } n ^ { 2 } + 2 d ^ { 2 } n - 8 d n ^ { 3 } + 9 d n ^ { 2 } - 3 d n + 4 n ^ { 2 } - 4 n + 2 , } } \\ { { C _ { 3 } = d n ^ { 2 } - d n - d - 2 n ^ { 2 } + 2 n . } } \end{array}
$$

We have $C _ { 1 } \leq 3 d ^ { 2 } n ^ { 2 }$ , since the diference is $d ^ { 2 } ( 5 n - 2 ) + d ( 6 n ^ { 2 } - 5 n + 2 ) + 2 ( n - 1 ) ^ { 2 } > 0$ . Also $C _ { 2 } \leq 4 d ^ { 2 } n ^ { 3 }$ the diference is

$$
d ^ { 2 } ( 7 n ^ { 2 } - 2 n ) + d ( 8 n ^ { 3 } - 9 n ^ { 2 } + 3 n ) - 4 n ^ { 2 } + 4 n - 2 > 0 .
$$

Its first and last terms together are at least $2 4 n ^ { 2 } - 4 n - 2 > 0$ , and its middle term is positive. Finally $C _ { 3 } \leq d n ^ { 2 }$ implies $( n d - 1 ) C _ { 3 } \leq d ^ { 2 } n ^ { 3 }$ . No lower bound on the $C _ { i }$ is needed. These estimates prove the bound on $P ( 0 )$ used in the proposition. All identities are short polynomial expansions; this margin proof does not use the large all-parameter sign certificate.

## C Frame examples and exact coeficient algebra

## C.1 Query class, a singular example, and the second budget

The row projectors $x _ { i } x _ { i } ^ { \top }$ have Frobenius Gram matrix $( 1 - q ) I + q \mathbf { 1 } \mathbf { 1 } ^ { \top } \succ 0$ . Their linear independence is a classical ETF fact (Fickus et al., 2021, Section 5.1). Let $\mathcal { Q } _ { 0 }$ be the orthogonal complement of their span in the symmetric matrices. It has dimension $d ( d + 1 ) / 2 - m$ . Since $\textstyle \sum _ { i } x _ { i } x _ { i } ^ { \top } = g I$ , every $V \in \mathcal { Q } _ { 0 }$ has trace zero. The symmetric flat-energy query class is exactly

$$
\{ w I + V : w \in \mathbb { R } , \ V \in \mathcal { Q } _ { 0 } \} .\tag{C.1}
$$

If $m < d ( d + 1 ) / 2 .$ , small nonzero multiples of any nonzero V give nonisotropic positive definite queries $I + \epsilon V$ . When $m = d ( d + 1 ) / 2$ , only scalar queries have flat row energy. Thus the (6, 3) ETF gives no larger query class, whereas an existing (16, 6) ETF has five traceless directions. This dimension count is a consequence of the classical projector fact, not a new independence theorem.

Here is an explicit (16, 6) example. Index the features by the edges 12, 13, 14, 23, 24, 34 of the complete graph on four vertices. At each vertex place the four sign triples

$$
( 1 , 1 , 1 ) , \quad ( 1 , - 1 , - 1 ) , \quad ( - 1 , 1 , - 1 ) , \quad ( - 1 , - 1 , 1 )
$$

on its incident edges in the listed order, with zeros elsewhere. Stack the rows to form $Z ,$ , and put $X = Z / \sqrt { 3 }$ . Then $Z ^ { \top } Z = 8 I$ , each row has squared norm three, and distinct row inner products have

squared value one. This verifies that X is a unit-row ETF with $g = 8 / 3$ and $q = 1 / 9$ . Fix

$$
W = \left( \begin{array} { c c c c c c } { { 1 } } & { { 0 } } & { { 0 } } & { { 0 } } & { { 0 } } & { { 1 } } \\ { { 0 } } & { { 2 } } & { { 0 } } & { { 0 } } & { { 2 } } & { { 0 } } \\ { { 0 } } & { { 0 } } & { { 3 } } & { { 3 } } & { { 0 } } & { { 0 } } \\ { { 0 } } & { { 0 } } & { { 3 } } & { { 3 } } & { { 0 } } & { { 0 } } \\ { { 0 } } & { { 2 } } & { { 0 } } & { { 0 } } & { { 2 } } & { { 0 } } \\ { { 1 } } & { { 0 } } & { { 0 } } & { { 0 } } & { { 0 } } & { { 1 } } \end{array} \right) .\tag{C.2}
$$

Its eigenvalues are 0, 0, 0, 2, 4, 6. Of-diagonal couplings join disjoint edges, which never share a row support; the diagonal weights sum to six on every vertex star. Thus $x _ { i } ^ { \top } W x _ { i } = 2 = w$ for every row. This is a singular nonisotropic query in the physical coordinates.

For the prescribed target $\lambda _ { 0 } = 1$ , the calibrated raw penalty is the unique positive root $r _ { \star } \in ( 4 / 5 , 9 / 1 0 )$ of

$$
2 1 6 r ^ { 3 } + 6 0 3 r ^ { 2 } + 3 2 r - 5 7 6 = 0 .\tag{C.3}
$$

Indeed (5.8) yields

$$
f ( r ) - \frac { 3 } { 1 1 } = - \frac { 2 1 6 r ^ { 3 } + 6 0 3 r ^ { 2 } + 3 2 r - 5 7 6 } { 8 8 ( r + 2 ) ( 3 r + 4 ) ( 3 r + 8 ) } .\tag{C.4}
$$

The root and interval are exact; no rounded penalty defines this model. The sharp value is $2 c _ { R } ( r _ { \star } )$ ， where

$$
c _ { R } ( r ) = \frac { 2 1 r ^ { 2 } + 7 2 r + 6 4 } { 2 0 ( r + 2 ) ^ { 2 } ( 3 r + 4 ) ^ { 2 } } ,\tag{C.5}
$$

and all ten residual dimensions maximize.

More generally, keep the ETF, flat physical query, and prescribed target fixed, and consider LOO with its separately calibrated raw penalty $r _ { 1 } I .$ . Single deletion has equal determinant masses and scalar mean coeficient

$$
f _ { 1 } ( r _ { 1 } ) = \frac { 1 } { m } \left\{ \frac { ( d - 1 ) g } { g + r _ { 1 } } + \frac { g - 1 } { g - 1 + r _ { 1 } } \right\} .\tag{C.6}
$$

It decreases strictly from $1 / g$ to zero, so it reaches the same target uniquely. Put $b _ { 1 } = ( g + r _ { 1 } ) ^ { - 1 }$ and $a _ { 1 } = 1 - b _ { 1 }$ . The LOO leverage is $b _ { 1 }$ and $x _ { i } ^ { \top } ( b _ { 1 } I ) W ( b _ { 1 } I ) x _ { i } = b _ { 1 } ^ { 2 } w$ . Therefore all LOO scores equal $w b _ { 1 } ^ { 2 } / ( m a _ { 1 } ^ { 2 } )$ , and Theorem 3.3 gives

$$
h _ { m - 1 } ( X , W ) = \frac { w } { m ( g + r _ { 1 } - 1 ) ^ { 2 } } , \qquad \mathrm { c o m p l e t e ~ m a x i m i z i n g ~ s p a c e = k e r ( } X ^ { \top } ) .\tag{C.7}
$$

This proves persistence of the full space at these two budgets only. In the explicit example $r _ { 1 }$ solves $4 8 r _ { 1 } ^ { 2 } + 4 3 r _ { 1 } - 8 0 = 0 \quad$ , so it difers from $r _ { \star }$ . No monotonicity of values, other-budget conclusion, ETF design optimality, or classification of nonflat queries is asserted.

## C.2 Reconstruction of the three-deletion gap

This appendix proves the strict sign used in Theorem 5.2. Its coeficient certificate is finite, but it covers the unbounded parameter domain. No grid of penalties or dimensions is used.

Put $m = d + k$ and introduce the following integer polynomials:

$$
\begin{array} { r l } & { v = d r + k , \qquad { \mathcal { Z } } = d r + m , } \\ & { { \mathcal { Q } } = ( m - 1 ) v ^ { 2 } - d k , \qquad { \mathcal { E } } = ( m - 1 ) v ^ { 2 } - 4 d k , } \\ & { { \mathcal { F } } = ( m - 1 ) ( m - 2 ) v ^ { 3 } - 3 d k ( m - 2 ) v - 2 d k ( k - d ) , } \\ & { { \mathcal { U } } = ( d - 3 ) { \mathcal { F } } + 3 ( m - 2 ) { \mathcal { Z } } { \mathcal { Q } } , } \\ & { { \mathcal { R } } = ( m - 2 ) v { \mathcal { Q } } + 6 d k ( k - d ) , } \\ & { { \mathcal { V } } = ( d - 3 ) { \mathcal { E } } { \mathcal { F } } + 3 ( m - 1 ) { \mathcal { Z } } ^ { 2 } { \mathcal { R } } . } \end{array}
$$

Direct substitution in (5.23) gives

$$
\begin{array} { l } { { \mathcal { E } = d ^ { 2 } ( m - 1 ) ( h ^ { 2 } - 4 q ) > 0 , \qquad \mathcal { F } = d ^ { 3 } ( m - 1 ) ( m - 2 ) D > 0 , } } \\ { { \displaystyle H _ { 1 } = \frac { d \mathcal { U } } { \mathcal { Z } \mathcal { F } } , \qquad H _ { 2 } = \frac { d ^ { 2 } \mathcal { V } } { \mathcal { Z } ^ { 2 } \mathcal { E } \mathcal { F } } . } } \end{array}\tag{C.8}
$$

Define H by exact polynomial division:

$$
3 d ^ { 2 } k \mathcal { H } = \mathcal { E } ( \mathcal { Z } + m r ) \mathcal { U } ^ { 2 } - d \mathcal { Z } \mathcal { E } \mathcal { F } \mathcal { U } - r d \mathcal { V } ( \mathcal { U } + k \mathcal { F } ) .\tag{C.9}
$$

Here $k \mathcal { F }$ is an ordinary product of the codimension and a polynomial; it is diferent from the fitted-loss coeficient $k _ { F }$ . The right side is divisible by $3 d ^ { 2 } k$ in the integer polynomial ring, and the quotient has degree seven in r.

To verify its statistical role, set

$$
\mathcal { L } = \mathcal { Z } \mathcal { F } - r \mathcal { U } , \qquad \mathcal { B } = \mathcal { Z } ^ { 2 } \mathcal { E } \mathcal { F } - 2 r \mathcal { Z } \mathcal { E } \mathcal { U } + r ^ { 2 } d \mathcal { V } .
$$

Equations (5.24) and (C.8) imply

$$
\begin{array} { l } { { M = \displaystyle \frac { d { \mathcal { L } } } { { \mathcal { Z } } { \mathcal { F } } } } , \quad B = \displaystyle \frac { d { \mathcal { B } } } { { \mathcal { Z } } ^ { 2 } { \mathcal { E } } { \mathcal { F } } } , \quad k _ { F } = \displaystyle \frac { r { \mathcal { U } } } { { \mathcal { Z } } { \mathcal { F } } } , } \\ { { c _ { R } = \displaystyle \frac { d [ m { \mathcal { Z } } { \mathcal { E } } { \mathcal { U } } - m r d { \mathcal { V } } - d { \mathcal { B } } ] } { m k { \mathcal { Z } } ^ { 2 } { \mathcal { E } } { \mathcal { F } } } } , } \\  { c _ { F } = \displaystyle \frac { d [ { \mathcal { B } } { \mathcal { F } } - { \mathcal { E } } { \mathcal { L } } ^ { 2 } ] } { m { \mathcal { Z } } ^ { 2 } { \mathcal { E } } { \mathcal { F } } ^ { 2 } } . } \end{array}
$$

On the common denominator mk $\cdot \mathcal { Z } ^ { 3 } \mathcal { E } \mathcal { F } ^ { 2 }$ , the gap numerator is d times

$$
r \mathcal { U } [ m \mathcal { Z } \mathcal { E } \mathcal { U } - m r d \mathcal { V } - d \mathcal { B } ] - k \mathcal { Z } [ B \mathcal { F } - \mathcal { E } \mathcal { L } ^ { 2 } ] .
$$

Using ${ \mathcal { Z } } = d r + m$ and $m = d + k$ , this bracket is exactly rZ times the right side of (C.9). Therefore

$$
c _ { R } k _ { F } - c _ { F } = \frac { 3 d ^ { 3 } r \mathcal { H } } { m \mathcal { Z } ^ { 2 } \mathcal { E } \mathcal { F } ^ { 2 } } .\tag{C.10}
$$

All factors removed from the denominator are strictly positive on the actual-frame domain proved in the main text.

For $d \geq 3 , k \geq 4$ , substitute $d = u + 3 , k = v _ { 0 } + 4 .$ . The complete expansion of $\mathcal { H } ( r , u + 3 , v _ { 0 } + 4 )$ has 581 nonzero coeficients, all positive integers. The exact coeficient data are in anc/etf\_triple/gap\_ coefficients.json. Each entry is $[ i , j , l , c _ { i j l } ]$ for $c _ { i j l } r ^ { i } u ^ { j } v _ { 0 } ^ { l }$ . The following table provides checks on the expansion; the full artifact supplies every coeficient.

<table><tr><td>Power of r</td><td>Nonzero coefficients</td><td>Minimum</td><td>Constant in  $( u , v _ { 0 } )$ </td></tr><tr><td>0</td><td>69</td><td>1</td><td>145212480</td></tr><tr><td>1</td><td>75</td><td>6</td><td>678003984</td></tr><tr><td>2</td><td>78</td><td>15</td><td>1273128192</td></tr><tr><td>3</td><td>79</td><td>20</td><td>1253473704</td></tr><tr><td>4</td><td>78</td><td>15</td><td>702853200</td></tr><tr><td>5</td><td>75</td><td>6</td><td>225144360</td></tr><tr><td>6</td><td>70</td><td>1</td><td>38141280</td></tr><tr><td>7</td><td>57</td><td>1</td><td>2624400</td></tr></table>

This proves $\mathcal { H } > 0$ for $r > 0 , u , v _ { 0 } \ge 0$ , including the shifted boundary. The exact reconstruction has four steps: form the displayed integer polynomials; perform the division in (C.9); substitute $d = u + 3 , k =$ $v _ { 0 } + 4 ;$ and collect all monomials. The public program anc/etf\_triple/check.py performs these steps, verifies (C.8) and (C.10), and compares the whole coeficient list, including its support, with the artifact. Merely testing signs of stored numbers would not prove the identity. The program also derives the local energy and weighted moments and enumerates all 560 original-row triples of an actual (16, 6) ETF. Those finite model checks support the derivation; the polynomial identity proves its uniform sign.

The only remaining actual case is $d = k = 3$ . Substitution in the compact moment formulas gives

$$
c _ { R } k _ { F } - c _ { F } = \frac { r ( 5 r ^ { 2 } + 1 0 r + 4 ) ( 5 r ^ { 3 } + 2 5 r ^ { 2 } + 3 0 r + 6 ) } { 2 ( r + 1 ) ^ { 2 } ( 5 r ^ { 2 } + 1 0 r + 1 ) ( 5 r ^ { 2 } + 1 0 r + 2 ) ^ { 2 } } > 0 .
$$

The checker verifies this identity separately. The domain argument in the main proof excludes every other actual $k = 3$ pair. Thus these two calculations cover every existing ETF in Theorem 5.2.

## D Mixture coeficient evaluation and examples

## D.1 Evaluating the coeficients

Four scalar count moments sufice given exact calibration:

$$
U = \mathbb { E } K _ { 1 } ^ { 2 } , \quad C = \mathbb { E } [ K _ { 1 } f ( K _ { 1 } ) ] , \quad F = \mathbb { E } f ( K _ { 1 } ) ^ { 2 } , \quad P = \mathbb { E } [ f ( K _ { 1 } ) f ( K _ { 2 } ) ] .
$$

The fixed total count gives $C _ { \times } = ( s a - C ) / ( d - 1 ) = \mathbb { E } [ f ( K _ { 1 } ) K _ { 2 } ]$ and $U _ { \times } = ( s \mu - U ) / ( d - 1 ) = \mathbb { E } [ K _ { 1 } K _ { 2 } ]$ Set

$$
L = { \frac { n a - C } { n ( n - 1 ) } } , \qquad N = { \frac { n \mu - U } { n ( n - 1 ) } } .
$$

Then all required coeficients are

$$
\begin{array} { l } { { \ A = \displaystyle \frac { n a - ( n + r ) F } { r n ( n - 1 ) } , \qquad D = 2 ( A - c _ { H } L ) , \qquad Q = A - 2 c _ { H } L + c _ { H } ^ { 2 } N , } } \\ { { \ } } \\ { { B _ { 0 } = ( F - P ) / b , } } \\ { { B _ { 1 } = 2 [ c _ { H } ( C - C _ { \times } ) - ( F - P ) ] / b , } } \\ { { B _ { 2 } = [ c _ { H } ^ { 2 } ( U - U _ { \times } ) - 2 c _ { H } ( C - C _ { \times } ) + ( F - P ) ] / b , } } \\ { { \ } } \\ { { \tau ( t ) = \{ a ^ { 2 } - [ ( 1 - t ) ^ { 2 } P + 2 t ( 1 - t ) c _ { H } C _ { \times } + t ^ { 2 } c _ { H } ^ { 2 } U _ { \times } ] \} / b . } } \end{array}
$$

These follow by expanding (6.1); for the first identity use $n f ( k ) - ( n + r ) f ( k ) ^ { 2 } = r k ( n - k ) / ( r + k ) ^ { 2 }$ . For numerical evaluation, the direct nonnegative sums can avoid subtractive cancellation.

The generating polynomial $g ( x )$ in Appendix B gives marginal weights $a _ { k } [ x ^ { s - k } ] g ( x ) ^ { d - 1 } / Z$ and pair weights $a _ { k } a _ { \ell } [ x ^ { s - k - \ell } ] g ( x ) ^ { d - 2 } / Z$ . Repeated convolution truncated at degree $s ,$ followed by these sums, takes $O ( d n s + n ^ { 2 } )$ arithmetic operations and $O ( s + n )$ working memory. This count excludes calibration, subset generation, bit complexity, and numerical stability. An approximately supplied penalty does not inherit the exact mean statement proved here.

## D.2 A concrete improvement and its adverse response

Take $n = d = s = 2 , r = 1$ , so $a = 5 / 1 1 , \lambda _ { 0 } = 1 2 / 5 .$ , and $b = 1 2 / 1 1$ . Each mixed-group original subset has probability $2 / 1 1$ , and each full-single-group subset has probability $3 / 2 2$ . Direct evaluation gives

$$
\alpha ( t ) = { \frac { ( 1 1 - t ) ^ { 2 } } { 1 3 3 1 } } , \qquad \beta ( t ) = { \frac { ( 1 1 + 4 t ) ^ { 2 } } { 2 1 7 8 } } .
$$

Here $D = 2 / 1 2 1 , Q = 1 / 1 3 3 1 , B _ { 0 } = 1 / 1 8 , B _ { 1 } = 4 / 9 9 , B _ { 2 } = 8 / 1 0 8 9$ , and (6.3) gives $T _ { s } = 8 4 7 / 3 1 1 6 > 1 / 4$ At $t = 1 / 4$ , every query’s sharp risk drops by the factor $1 8 4 9 / 1 9 3 6$ . Yet for the positive-copy response $y = ( 1 , 1 , - 1 , - 1 )$ and $W = ( 1 , 1 ) ( 1 , 1 ) ^ { \top }$ , the normalized risk increases from $1 / 1 8$ to $8 / 1 2 1$ . This is a pure transverse mean response. Thus a smaller supremum does not imply pointwise risk dominance.

More generally, for $0 \leq t \leq 1$ ，

$$
\operatorname* { s u p } _ { 0 \neq W \succeq 0 } \operatorname* { s u p } _ { y \neq 0 } \frac { \mathrm { t r } ( W \mathrm { C o v } ( w ( t ) ) ) } { J _ { 0 } ( y ) \operatorname* { m a x } _ { j } W _ { j j } } = \operatorname* { m a x } \{ \alpha ( t ) , \beta ( t ) \} .
$$

The upper bound follows from $\tau ( t ) \geq 0$ and the two response blocks. Contrasts attain $\alpha ( t )$ ; a flat rankone query $W = v v ^ { \top } , v _ { j } \in \{ - 1 , 1 \}$ , and nonzero means $q \perp v$ attain $\beta ( t )$ . Hence optimizing this scalar mixture family amounts to minimizing the maximum of two convex quadratics. Its best weight can difer from the conservative step above, and can introduce mean maximizers. This does not optimize over al estimators or laws.

## E Explicit recalibration and transfer bounds

This appendix first gives a perturbation bound for general full-rank designs at every $d \leq s < m$ . It then uses the balanced base’s complete equality space. All constants concern a fixed target and query.

## E.1 Fixed target coordinates and explicit constants

$$
Z = X A _ { 0 } ^ { - 1 / 2 } , \quad E = H A _ { 0 } ^ { - 1 / 2 } , \quad B = Z ^ { \top } Z , \quad \mathsf { P } _ { s } = A _ { 0 } ^ { - 1 / 2 } R _ { s } A _ { 0 } ^ { - 1 / 2 } , \quad V = A _ { 0 } ^ { - 1 / 2 } W A _ { 0 } ^ { - 1 / 2 } .
$$

For a diagonal row selector $D _ { S }$ , define

$$
\begin{array} { r l r l r l } & { A _ { S } = Z ^ { \top } D _ { S } Z + \mathsf { P } _ { \ast } , \qquad } & & { L _ { S } = A _ { S } ^ { - 1 } Z ^ { \top } D _ { S } , \qquad } & & { L _ { 0 } = ( I + B ) ^ { - 1 } Z ^ { \top } , } \\ & { F _ { S } = L _ { S } - L _ { 0 } , \qquad } & & { C _ { s } = \mathbb { E } [ F _ { S } ^ { \top } V F _ { S } ] , \qquad } & & { T = ( I + Z Z ^ { \top } ) ^ { 1 / 2 } = K _ { X } ^ { - 1 / 2 } . } \end{array}
$$

Thus the physical risk is $y ^ { \top } C _ { s } y$ and $M _ { s } = T C _ { s } T$ . A prime on a whole object denotes its value at $Z + E$ Let $z = \| Z \| , \sigma = \sigma _ { \operatorname* { m i n } } ( Z ) > 0$ , and choose

$$
r _ { 0 } = \sigma / 2 , \quad \ell = \sigma ^ { 2 } / 4 , \quad v = z + r _ { 0 } , \quad u = v ^ { 2 } , \quad t = \sqrt { 1 + v ^ { 2 } } .\tag{E.1}
$$

If $\epsilon = \| E \| _ { F } \leq r _ { 0 }$ , every $Z + x E , 0 \leq x \leq 1$ , has Gram spectrum in $[ \ell , u ]$ . For each $d \leq s < m$ , set

$$
a _ { s } = \frac { s - d + 1 } { m - d + 1 } , \qquad b _ { s } = \frac { s } { m } , \qquad q _ { - } = \frac { \ell } { b _ { s } } , \quad q _ { + } = \frac { u } { a _ { s } } ,
$$

$$
\kappa _ { s } = \frac { b _ { s } q _ { - } } { ( 1 + q _ { - } ) ( 1 + q _ { + } ) } , \qquad \mathscr { L } _ { s } = b _ { s } \left[ \frac { 1 } { \ell } + \frac { 1 } { \kappa _ { s } ( 1 + \ell ) ^ { 2 } } \right] ,\tag{E.2}
$$

$$
p = a _ { s } , \quad L _ { X } = 2 v \mathcal { L } _ { s } , \quad B _ { F } = \frac { 1 } { 2 \sqrt { p } } + \frac { 1 } { 2 } ,
$$

$$
C _ { D } = { \frac { 5 } { 4 p } } + { \frac { 5 } { 4 } } + { \frac { L _ { X } } { 2 p ^ { 3 / 2 } } } , \qquad C _ { F } = t C _ { D } + B _ { F } v , \qquad L _ { p } = \sqrt { d } \left( { \frac { 2 } { \sqrt { p } } } + { \frac { L _ { X } } { p } } \right) .\tag{E.3}
$$

These are positive finite constants determined by $X , A _ { 0 } , m , s .$ A convenient explicit operator bound is

$$
\begin{array} { r } { \eta _ { s } ( \epsilon ) = \left\| V \right\| \left[ 2 B _ { F } t C _ { F } \epsilon + B _ { F } ^ { 2 } t ^ { 2 } \operatorname { t a n h } ( L _ { p } \epsilon / 2 ) \right] \leq \mathcal { M } _ { s } \epsilon , \quad \mathcal { M } _ { s } = \left\| V \right\| \left[ 2 B _ { F } t C _ { F } + \frac { 1 } { 2 } B _ { F } ^ { 2 } t ^ { 2 } L _ { p } \right] . } \end{array}\tag{E.4}
$$

In Theorem 7.2 take $r = r _ { 0 }$ and $L _ { s } = \mathcal { M } _ { s }$ . These constants require no sum over original-row subsets.

## E.2 Curvature of calibration, including matrix directions

Recall the spectral partition $P _ { s } ( q )$ from (2.10). Its log-coordinate Hessian is $ { \mathcal { C } } =  { \mathrm { C o v } } ( \mathbf { 1 } _ { J } )$ . There is a useful conditional experiment for this distribution. Select s of m slots with product weights $1 + q _ { j }$ on the first d slots and one on the others. Conditional on those slots, mark each selected feature slot $j$ independently with probability $q _ { j } / ( 1 + q _ { j } )$ . Summing over the unmarked selected companions shows that the marked set has mass proportional to $c _ { \lvert J \rvert } q _ { J }$ , as required. Total conditional variance therefore gives

$$
{ \mathcal { C } } \succeq \mathrm { d i a g } \left( { \frac { \rho _ { j } } { 1 + q _ { j } } } \right) .\tag{E.5}
$$

For weights bounded above by $w _ { \mathrm { m a x } }$ , the selection probability of a slot of weight w is $w e _ { s - 1 } / ( e _ { s } + w e _ { s - 1 } )$ Counting extensions of $( s - 1 )$ -subsets gives $s e _ { s } \ \le \ ( m - s ) w _ { \mathrm { m a x } } e _ { s - 1 }$ , so this probability is at least $( s / m ) w / w _ { \mathrm { m a x } }$ . On the box $q _ { - } \leq q _ { j } \leq q _ { + }$ , (E.5) consequently implies

$$
\mathcal { C } \succeq \frac { ( s / m ) q _ { - } } { ( 1 + q _ { - } ) ( 1 + q _ { + } ) } I = \kappa _ { s } I .\tag{E.6}
$$

The slightly conservative factor $( 1 + q _ { - } )$ in this bound is harmless.

For a symmetric matrix $Q$ with eigenvalues $z _ { j }$ , define A (Q) = log $P _ { s } ( e ^ { z _ { 1 } } , \dots , e ^ { z _ { d } } )$ . The diagonal part of its Hessian is C. Its of-diagonal coeficients are $( \rho _ { i } - \rho _ { j } ) / ( z _ { i } - z _ { j } )$ , with continuous limits at equality. To bound them, join the vector z to the vector obtained by swapping coordinates $i , j$ . The segment stays in the same log box. Integrating (E.6) along it gives

$$
2 ( \rho _ { i } - \rho _ { j } ) ( z _ { i } - z _ { j } ) \geq 2 \kappa _ { s } ( z _ { i } - z _ { j } ) ^ { 2 } .
$$

The usual second derivative formula for a spectral function now proves $\nabla ^ { 2 } \mathcal { A } _ { s } \succeq \kappa _ { s } I$ on all symmetric directions in Frobenius norm, including repeated eigenvalues.

In target coordinates the calibrated $\mathsf { P } _ { s }$ commutes with $B .$ For $B = U \mathrm { d i a g } ( \lambda _ { j } ) U ^ { \top }$ , its eigenvalues $p _ { j }$ satisfy

$$
\rho _ { j } = \frac { \lambda _ { j } } { 1 + \lambda _ { j } } , \qquad q _ { j } = \frac { \lambda _ { j } } { p _ { j } } , \qquad \nabla _ { z } \log P _ { s } ( e ^ { z } ) | _ { z = \log q } = \rho .
$$

The sandwich (2.7) gives $a _ { s } I \preceq \mathsf { P } _ { s } \preceq b _ { s } I$ , placing $q _ { j }$ in the box used above. The matrix target moment is $D = B ( I + B ) ^ { - 1 }$ , with $\dot { D } = ( I + B ) ^ { - 1 } \dot { B } ( I + B ) ^ { - 1 }$ . The inverse Hessian bound gives $\| \dot { Q } \| _ { F } ~ \leq$ $\| \dot { B } \| _ { F } / [ \kappa _ { s } ( 1 + \ell ) ^ { 2 } ]$ for the log-parameter matrix $Q .$ . On diagonal directions, $\dot { p } _ { j } = ( p _ { j } / \lambda _ { j } ) \dot { \lambda } _ { j } - p _ { j } \dot { z } _ { j }$ . This yields the bound $\mathcal { L } _ { s }$ in (E.2).

For completeness the of-diagonal bound does not assume a fixed eigenbasis. If $\lambda _ { i } > \lambda _ { j }$ , the exchange identity in Section 2 gives $q _ { i } > q _ { j }$ , and

$$
p _ { i } - p _ { j } = { \frac { \lambda _ { i } - \lambda _ { j } } { q _ { i } } } - p _ { j } { \frac { q _ { i } - q _ { j } } { q _ { i } } } , \qquad { \frac { q _ { i } - q _ { j } } { q _ { i } } } \leq \log q _ { i } - \log q _ { j } .
$$

The same swap argument bounds the inverse moment divided diference by $1 / \kappa _ { s }$ , while $\big ( \rho _ { i } - \rho _ { j } \big ) / ( \lambda _ { i } - \lambda _ { j } ) =$ $1 / [ ( 1 + \lambda _ { i } ) ( 1 + \lambda _ { j } ) ]$ . Thus the absolute penalty divided diference is at most $\mathcal { L } _ { s }$ . Diagonal and of-diagonal directions are orthogonal in Frobenius geometry, and continuity covers ties. We have proved

$$
\| D \mathsf { P } _ { s } ( B ) [ \dot { B } ] \| _ { F } \leq \mathcal { L } _ { s } \| \dot { B } \| _ { F } .\tag{E.7}
$$

Existence of the derivative follows locally from the positive matrix Hessian and the inverse-function theorem. Uniqueness of calibration patches these local maps along the full segment.

## E.3 Tracking the fit, law, and loss

Along $Z + x E , \| \dot { B } \| _ { F } \leq 2 v \epsilon$ , so (E.7) gives $\| \dot { \mathsf { P } } _ { s } \| _ { F } \leq L _ { X } \epsilon$ . Each $A _ { S } \succeq p I$ . Factoring through the penalized ridge operator and using $\textstyle \operatorname* { s u p } _ { x \geq 0 } x / ( 1 + x ^ { 2 } ) = 1 / 2$ yields

$$
\| L _ { S } \| \leq ( 2 \sqrt { p } ) ^ { - 1 } , \qquad \| L _ { 0 } \| \leq 1 / 2 , \qquad \| F _ { S } \| \leq B _ { F } .
$$

For the selected rows, diferentiation gives

$$
\dot { L } _ { S } = A _ { S } ^ { - 1 } E _ { S } ^ { \top } ( I - Z _ { S } L _ { S } ) - L _ { S } E _ { S } L _ { S } - A _ { S } ^ { - 1 } \dot { \mathsf { P } } _ { s } L _ { S } ,
$$

with the outside row selector restored when $L _ { S }$ is viewed as a d by m operator. The selected smoother is positive semidefinite and at most I. Thus the first two terms are bounded by $5 \epsilon / ( 4 p )$ and the last by $L _ { X } \epsilon / ( 2 p ^ { 3 / 2 } )$ . The full-fit derivative is at most $5 \epsilon / 4$ . Hence $\| \dot { F } _ { S } \| \le C _ { D } \epsilon$

The equation $T \dot { T } + \dot { T } T = E Z ^ { \top } + Z E ^ { \top }$ , with $T \succeq I .$ , has its integral solution $\begin{array} { r } { \dot { T } = \int _ { 0 } ^ { \infty } e ^ { - a T } ( E Z ^ { \top } + } \end{array}$ $Z E ^ { \top } ) e ^ { - a T } d a$ . Consequently $\lVert \dot { T } \rVert \leq v \epsilon$ , and

$$
\| F _ { S } T \| \le B _ { F } t , \qquad \| ( F _ { S } T ) ^ { \bullet } \| \le C _ { F } \epsilon .
$$

The derivative of an unnormalized log weight obeys

$$
\begin{array} { l } { \displaystyle \left. \frac { d } { d x } \log \operatorname* { d e t } A _ { S } \right. \leq 2 \| Z _ { S } A _ { S } ^ { - 1 / 2 } \| _ { F } \| E _ { S } A _ { S } ^ { - 1 / 2 } \| _ { F } + \| A _ { S } ^ { - 1 } \| _ { F } \| \dot { \mathsf { P } } _ { s } \| _ { F } } \\ { \leq L _ { p } \epsilon . } \end{array}
$$

Thus its endpoint ratio lies in $[ e ^ { - L _ { p } \epsilon } , e ^ { L _ { p } \epsilon } ]$ , and the normalized likelihood ratio lies in $[ e ^ { - 2 L _ { p } \epsilon } , e ^ { 2 L _ { p } \epsilon } ]$ . In addition,

$$
\mathrm { T V } ( p _ { s } ^ { \prime } , p _ { s } ) \le \operatorname { t a n h } ( L _ { p } \epsilon / 2 ) .\tag{E.8}
$$

To verify this sharper statement, if a positive unnormalized density ratio lies in $[ A , B ]$ and has mean $\mu ,$ convexity of $( x - \mu ) .$ <sub>+</sub> bounds its normalized total variation by $( B - \mu ) ( \mu - A ) / [ \mu ( B - A ) ]$ ]. Maximizing over µ gives $( { \sqrt { B } } - { \sqrt { A } } ) / ( { \sqrt { B } } + { \sqrt { A } } )$ , which proves (E.8).

Each normalized risk atom $( F _ { S } T ) ^ { \top } V ( F _ { S } T )$ lies between zero and $\| V \| B _ { F } ^ { 2 } t ^ { 2 } I$ . Its derivative norm is at most $2 \| V \| B _ { F } t C _ { F } \epsilon$ . The change in the expectation of an atom in $[ 0 , B I ]$ is at most B times total variation: apply the scalar bound to each unit quadratic form. Integrating the atom derivative and then changing the law proves

$$
\lVert M _ { s } ^ { \prime } - M _ { s } \rVert \leq \eta _ { s } ( \epsilon )
$$

with (E.4). This accounts for both centering at the new full fit and changing normalization.

For the same nonzero raw response at both designs there is also the useful loss comparison

$$
e ^ { - 2 \epsilon } \leq \frac { y ^ { \top } K _ { X + H } y } { y ^ { \top } K _ { X } y } \leq e ^ { 2 \epsilon } .\tag{E.9}
$$

Indeed, along the path,

$$
\left| \frac { d } { d x } \log ( y ^ { \top } K y ) \right| = \frac { | y ^ { \top } K ( E Z ^ { \top } + Z E ^ { \top } ) K y | } { y ^ { \top } K y } \leq 2 \epsilon ,
$$

using $\| K ^ { 1 / 2 } Z \| \le 1$ and $\| K ^ { 1 / 2 } \| \le 1$ . Integrating proves (E.9).

## E.4 Why reoptimization and the linear risk term matter

The distinction is visible in an exact four-row model. Take $n = d = 2 , s = 3 , \lambda _ { 0 } = 1 0 / 7 .$ , and $W = e _ { 1 } e _ { 1 } ^ { \top }$ Scale every row by ${ \sqrt { t } } .$ . The calibrated penalty solves $4 r ( t ) ^ { 2 } + ( 6 t - 3 \lambda _ { 0 } ) r ( t ) - 4 t \lambda _ { 0 } = 0 . \mathrm { A t } t = 1$ $r = 1 , r ^ { \prime } ( 1 ) = - 1 / 3 4$ , and $h ( t ) = t / [ 4 ( r ( t ) + t ) ^ { 2 } ]$ has derivative $h ^ { \prime } ( 1 ) = 1 / 5 4 4$ . The strict base gap keeps this contrast branch maximal locally. Thus the optimum can have nonzero first derivative. At the same base with $W = I$ , scale only the first coordinate group by ${ \sqrt { t } } .$ Write the calibrated penalty as $\mathrm { d i a g } ( r _ { 1 } ( t ) , r _ { 2 } ( t ) )$ . Define

$$
A = ( r _ { 1 } + t ) ( r _ { 2 } + 2 ) , \qquad B = ( r _ { 1 } + 2 t ) ( r _ { 2 } + 1 ) , \qquad Z = A + B .
$$

The two count types (1, 2) and (2, 1) have probabilities $A / Z$ and $B / Z .$ . With $\lambda = 1 0 / 7$ , the calibration equations are

$$
{ \frac { A / ( r _ { 1 } + t ) + 2 B / ( r _ { 1 } + 2 t ) } { Z } } = { \frac { 2 } { 2 t + \lambda } } , \qquad { \frac { 2 A / ( r _ { 2 } + 2 ) + B / ( r _ { 2 } + 1 ) } { Z } } = { \frac { 2 } { 2 + \lambda } } .
$$

The two contrast risk values are

$$
\alpha _ { 1 } ( t ) = \frac { t A } { 2 Z ( r _ { 1 } + t ) ^ { 2 } } , \qquad \alpha _ { 2 } ( t ) = \frac { B } { 2 Z ( r _ { 2 } + 1 ) ^ { 2 } } .
$$

Diferentiating the two calibration equations at $t = r _ { 1 } = r _ { 2 } = 1$ gives

$$
( r _ { 1 } ^ { \prime } , r _ { 2 } ^ { \prime } ) = \frac { 1 } { 1 2 2 4 } ( - 1 , - 3 5 ) , \qquad ( \alpha _ { 1 } ^ { \prime } , \alpha _ { 2 } ^ { \prime } ) = \frac { 1 } { 1 1 7 5 0 4 } ( - 5 8 9 , 8 0 5 ) .
$$

Thus $\alpha _ { 1 } ^ { \prime } - \alpha _ { 2 } ^ { \prime } = - 4 1 / 3 4 5 6$ . For $t > 1$ suficiently close to one, an old maximizer from the first group has linear regret. Reoptimization over the complete tied space is essential to the quadratic statement.

## F A finite remainder and a common contrast-space radius

This appendix supplies the finite approximation used by the enlarged-space theorem. It retains the linear fit change, the centered determinant score, and the exact change in the Full fit. Its error is second order locally. The construction also explains which exact mean correction would be needed for a supplied penalty that is not calibrated. The main theorem uses exact calibration.

## F.1 The common finite surrogate

Fix the balanced base and a budget $d \leq s < m$ . Work in target coordinates:

$$
Z = X _ { 0 } / \sqrt { \lambda _ { 0 } } , \quad E = H / \sqrt { \lambda _ { 0 } } , \quad Z _ { 1 } = Z + E , \quad V = W / \lambda _ { 0 } , \quad P _ { 0 } = p I , \quad \Delta P = P _ { 1 } - p I .
$$

Here $p = r _ { s } / \lambda _ { 0 } > 0$ . The endpoint is full column rank and $P _ { 1 }$ is SPD. In the exactly calibrated case $P _ { 1 } = R _ { s } ( X _ { 0 } + H ) / \lambda _ { 0 }$ . Use this same $P _ { 1 }$ in its law and fit. For each original-row subset define

$$
\begin{array} { r l } & { A _ { S } = Z ^ { \top } D _ { S } Z + p I , } \\ & { L _ { S } = A _ { S } ^ { - 1 } Z ^ { \top } D _ { S } , } \\ & { G _ { S } = Z ^ { \top } D _ { S } E + E ^ { \top } D _ { S } Z + \Delta P , } \\ & { L _ { \ast , i } = ( I + Z _ { i } ^ { \top } Z _ { i } ) ^ { - 1 } Z _ { i } ^ { \top } , } \\ & { Q _ { S } = J _ { S } ^ { \top } V J _ { S } , } \end{array} \qquad \begin{array} { r l } & { A _ { 1 , S } = Z _ { 1 } ^ { \top } D _ { S } Z _ { 1 } + P _ { 1 } , } \\ & { L _ { 1 , S } = A _ { 1 , S } ^ { - 1 } Z _ { 1 } ^ { \top } D _ { S } , } \\ & { D _ { S } ^ { \mathrm { f i } } = A _ { S } ^ { - 1 } ( E ^ { \top } D _ { S } - G _ { S } L _ { S } ) , } \\ & { J _ { S } = L _ { S } - L _ { \ast , 0 } + D _ { S } ^ { \mathrm { f i } } , } \\ & { t _ { S } = \mathrm { t r } ( A _ { S } ^ { - 1 } G _ { S } ) . } \end{array}
$$

In these formulas $Z _ { 0 } = Z$ . Write $\mathbb { E } _ { i }$ for the base and endpoint laws, $\overline { { Q } } = \mathbb { E } _ { 0 } Q _ { S } , \bar { t } = \mathbb { E } _ { 0 } t _ { S }$ , and

$$
N ( Z , P ) = \sum _ { | S | = s } \operatorname * { d e t } ( P + Z ^ { \top } D _ { S } Z ) , \quad \mu = \frac { N ( Z _ { 1 } , P _ { 1 } ) } { N ( Z , p I ) } , \quad \bar { w } = \mu - 1 - \bar { t } , \quad \Delta = L _ { * , 1 } - L _ { * , 0 } .
$$

The complete surrogate is

$$
\widetilde { C } = \overline { { Q } } + \frac { \mathbb { E } _ { 0 } [ t _ { S } Q _ { S } ] - \bar { t } \overline { { Q } } } { \mu } - \Delta ^ { \top } V \Delta .\tag{F.1}
$$

The endpoint covariance is $C _ { 1 } = \mathbb { E } _ { 1 } [ ( L _ { 1 , S } - L _ { * , 1 } ) ^ { \top } V ( L _ { 1 , S } - L _ { * , 1 } ) ]$ when the endpoint is exactly calibrated.   
Put $T _ { 1 } = ( I + Z _ { 1 } Z _ { 1 } ^ { \top } ) ^ { 1 / 2 }$ and $\widetilde { M } = T _ { 1 } \widetilde { C } T _ { 1 }$ . The surrogate itself need not be PSD.

Here is the exact error identity. Define

$$
R _ { S } = L _ { 1 , S } - L _ { S } - D _ { S } ^ { \mathrm { f i t } } , \qquad w _ { S } = { \frac { \mathrm { d e t } A _ { 1 , S } } { \mathrm { d e t } A _ { S } } } - 1 - t _ { S } .
$$

The endpoint density is

$$
\frac { p _ { 1 } ( S ) } { p _ { 0 } ( S ) } = 1 + \frac { t _ { S } - \bar { t } } { \mu } + \frac { w _ { S } - \bar { w } } { \mu } .
$$

Since exact calibration gives $\mathbb { E } _ { 1 } L _ { 1 , S } = L _ { * , 1 }$ , expansion around $L _ { * , 0 }$ yields

$$
C _ { 1 } - \widetilde { C } = \frac { \mathbb { E } _ { 0 } [ ( w _ { S } - \bar { w } ) Q _ { S } ] } { \mu } + \mathbb { E } _ { 1 } [ J _ { S } ^ { \top } V R _ { S } + R _ { S } ^ { \top } V J _ { S } ] + \mathbb { E } _ { 1 } [ R _ { S } ^ { \top } V R _ { S } ] .\tag{F.2}
$$

Thus the finite Full shift cancels exactly. Omitting it from the surrogate would alter the error identity.

## F.2 Ordered resolvent bounds at an SPD endpoint

All norms below are operator norms. Let $e _ { s } ^ { \mathrm { r o w } }$ denote the sum of the s largest squared row norms of E. Put $c _ { 0 } = 1 / \lambda _ { 0 }$ and

$$
\begin{array} { l l } { { k _ { - } = \operatorname* { m a x } \{ 0 , s - n ( d - 1 ) \} , } } & { { \qquad k _ { + } = \operatorname* { m i n } \{ n , s \} , } } \\ { { a _ { - } = p + c _ { 0 } k _ { - } , } } & { { \qquad a _ { + } = p + c _ { 0 } k _ { + } , } } \\ { { \phantom { a _ { + } = a _ { - } ^ { - 1 / 2 } , } } } & { { \qquad u _ { * } = \sqrt { c _ { 0 } k _ { + } / ( p + c _ { 0 } k _ { + } ) } , } } \\ { { u _ { - } = \sqrt { c _ { 0 } k _ { - } / a _ { - } } , } } & { { \qquad a _ { * } ^ { 2 } = \operatorname* { m i n } \{ e _ { s } ^ { \mathrm { r o w } } , \| E \| ^ { 2 } \} , } } \\ { { v _ { * } ^ { 2 } = a _ { * } ^ { 2 } / a _ { - } , } } & { { \qquad \tau _ { * } = e _ { s } ^ { \mathrm { r o w } } / a _ { - } , } } \\ { { g _ { * } = 2 u _ { * } v _ { * } + \| \Delta P \| / a _ { - } , } } & { { \qquad \delta _ { * } = g _ { * } + v _ { * } ^ { 2 } . } } \end{array}
$$

Choose a valid $l > 0$ with $P _ { 1 } \succeq l I$ . Define

$$
\begin{array} { r l } & { \ell _ { \mathrm { p e n } } = l / a _ { + } , \qquad \ell _ { \mathrm { g r a m } } = \ell _ { \mathrm { p e n } } + ( u _ { - } - v _ { * } ) _ { + } ^ { 2 } , } \\ & { \ell _ { \mathrm { r a y } } = \displaystyle \operatorname* { m i n } _ { x \in [ c _ { 0 } k _ { - } , c _ { 0 } k _ { + } ] } \frac { l + ( \sqrt { x } - a _ { * } ) _ { + } ^ { 2 } } { p + x } , } \\ & { \ell _ { * } = \operatorname* { m a x } \{ \ell _ { \mathrm { g r a m } } , \ell _ { \mathrm { r a y } } , 1 - g _ { * } \} > 0 , \qquad \kappa _ { * } = 1 / \ell _ { * } , } \\ & { \beta _ { * } = \operatorname* { m i n } \left\{ \delta _ { * } \kappa _ { * } , \operatorname* { m a x } \left( \left| \ell _ { * } ^ { - 1 } - 1 \right| , \left| ( 1 + \delta _ { * } ) ^ { - 1 } - 1 \right| \right) \right\} , } \\ & { r _ { \mathrm { p t } } = b _ { * } \{ \beta _ { * } ( v _ { * } + g _ { * } u _ { * } ) + \kappa _ { * } v _ { * } ^ { 2 } u _ { * } \} . } \end{array}
$$

This scalar minimum is explicit. If $a _ { * } > 0 ,$ , clip

$$
t _ { * } = \frac { - ( p - l - a _ { * } ^ { 2 } ) + \sqrt { ( p - l - a _ { * } ^ { 2 } ) ^ { 2 } + 4 a _ { * } ^ { 2 } p } } { 2 a _ { * } }
$$

to $[ \sqrt { c _ { 0 } k _ { - } } , \sqrt { c _ { 0 } k _ { + } } ]$ and evaluate at $x = t _ { * } ^ { 2 } . \mathrm { ~ H ~ } a _ { * } = 0$ , use the lower endpoint when $p \geq l$ and the upper endpoint otherwise. To verify the rule, write $t = { \sqrt { x } }$ . Below $^ { a _ { * } }$ the ratio decreases. Above $^ { a _ { * } }$ its derivative has the sign of $a _ { * } t ^ { 2 } + ( p - l - a _ { * } ^ { 2 } ) t - a _ { * } p $ , whose unique positive root exceeds $^ { a _ { * } }$

For each subset suppress the subscript and set

$$
\begin{array} { c } { { U = A ^ { - 1 / 2 } Z ^ { \top } D _ { S } , \qquad Y = A ^ { - 1 / 2 } E ^ { \top } D _ { S } , } } \\ { { B _ { 1 } = U Y ^ { \top } + Y U ^ { \top } + A ^ { - 1 / 2 } \Delta P A ^ { - 1 / 2 } , \qquad B _ { 2 } = Y Y ^ { \top } , } } \\ { { B = B _ { 1 } + B _ { 2 } , \qquad H _ { B } = ( I + B ) ^ { - 1 } . } } \end{array}
$$

Then $\left\| U \right\| \le u _ { * } , \left\| Y \right\| \le v _ { * }$ , tr $B _ { 2 } \leq \tau _ { * } , \| B _ { 1 } \| \leq g _ { * }$ <sub>∗</sub>, and $\| B \| \leq \delta _ { * }$ . The trace and operator bounds on $B _ { 2 }$ are diferent; τ<sub>∗</sub> cannot be replaced by $v _ { * } ^ { 2 }$

The penalty and Gram parts give $I + B \succeq \ell _ { \mathrm { g r a m } } I$ . For the Rayleigh bound, any nonzero $h$ has $x = \| D _ { S } Z h \| ^ { 2 } / \| h \| ^ { 2 } \in [ c _ { 0 } k _ { - } , c _ { 0 } k _ { + } ]$ , and

$$
\begin{array} { r } { h ^ { \top } A _ { 1 } h \geq \{ l + ( \sqrt { x } - a _ { * } ) _ { + } ^ { 2 } \} \| h \| ^ { 2 } , \qquad h ^ { \top } A h = ( p + x ) \| h \| ^ { 2 } . } \end{array}
$$

Also $I + B \succeq ( 1 - g _ { * } ) I$ . Consequently

$$
\ell _ { * } I \preceq I + B \preceq ( 1 + \delta _ { * } ) I , \quad \| H _ { B } \| \leq \kappa _ { * } , \quad \| H _ { B } - I \| \leq \beta _ { * } .
$$

The last bound follows both from $H _ { B } - I = - H _ { B } B$ and from its eigenvalues. There is no assumption $\delta _ { * } < 1$ here. The ordered resolvent identity is

$$
R = A ^ { - 1 / 2 } \{ ( H _ { B } - I ) ( Y - B _ { 1 } U ) - H _ { B } B _ { 2 } U \} .\tag{F.3}
$$

Indeed $L _ { 1 } = A ^ { - 1 / 2 } H _ { B } ( U + Y )$ and $L + D ^ { \mathrm { f i t } } = A ^ { - 1 / 2 } ( U + Y - B _ { 1 } U )$ ; subtract and use $H _ { B } - I =$ $- H _ { B } ( B _ { 1 } + B _ { 2 } )$ . This proves $\| R \| \leq r _ { \mathrm { p t } }$ without commuting any matrices.

## F.3 Residual moments and the finite normalized error

Retain the two PSD residual moments

$$
T _ { S } ^ { \mathrm { r e s } } = Y - B _ { 1 } U = A ^ { 1 / 2 } D ^ { \mathrm { f i t } } , \qquad W _ { S } ^ { \mathrm { r e s } } = Y ^ { \top } U = D _ { S } E L _ { S } .
$$

For each of the three PSD atoms $F _ { S } \in \{ Q _ { S } , ( T _ { S } ^ { \mathrm { r e s } } ) ^ { \top } T _ { S } ^ { \mathrm { r e s } } , ( W _ { S } ^ { \mathrm { r e s } } ) ^ { \top } W _ { S } ^ { \mathrm { r e s } } \}$ , define $\bar { F } = \mathbb { E } _ { 0 } F _ { S }$ and $F ^ { \mathrm { s c } } =$ $\mathbb { E } _ { 0 } [ t _ { S } F _ { S } ]$ . Put

$$
\rho _ { * } = \tau _ { * } + \sum _ { j = 2 } ^ { d } { \binom { d } { j } } \delta _ { * } ^ { j } , \quad a _ { \mathrm { d e n } } = \frac { \rho _ { * } + | \bar { w } | } { \mu } , \quad q _ { \mathrm { d e n } } = \frac { ( 1 + \delta _ { * } ) ^ { d } } { \mu } ,
$$

and define the scalar upper bound

$$
\mathcal { U } ( \bar { F } , F ^ { \mathrm { s c } } ) = \operatorname* { m i n } \left\{ q _ { \mathrm { d e n } } \| \bar { F } \| , \lambda _ { \operatorname* { m a x } } \left( \bar { F } + \frac { F ^ { \mathrm { s c } } - \bar { t } \bar { F } } { \mu } + a _ { \mathrm { d e n } } \bar { F } \right) \right\} .
$$

Both entries are nonnegative. Expansion of $\operatorname* { d e t } ( I + B )$ gives $| w _ { S } | \le \rho ,$ <sub>∗</sub> and $0 < \operatorname* { d e t } ( I + B ) \le ( 1 + \delta _ { * } ) ^ { d }$ The exact density formula therefore proves the PSD inequalities

$$
\mathbb { E } _ { 1 } F _ { S } \preceq q _ { \mathrm { d e n } } { \bar { F } } , \qquad \mathbb { E } _ { 1 } F _ { S } \preceq { \bar { F } } + ( F ^ { \mathrm { s c } } - { \bar { t } } { \bar { F } } ) / \mu + a _ { \mathrm { d e n } } { \bar { F } } .
$$

Take their scalar norm bounds; no matrix minimum is used. Let $q _ { J } , q _ { T } , q _ { W }$ be these three values and put

$$
\begin{array} { r l } { r _ { \mathrm { m s } } = b _ { * } \{ \beta _ { * } \sqrt { q _ { T } } + \kappa _ { * } v _ { * } \sqrt { q _ { W } } \} , \quad } & { \quad \quad \quad e _ { R } = \| V \| \operatorname* { m i n } \{ r _ { \mathrm { m s } } ^ { 2 } , r _ { \mathrm { p t } } ^ { 2 } \} , } \\ { D _ { \mathrm { e n d } } = a _ { \mathrm { d e n } } \| \bar { Q } \| + 2 \sqrt { q _ { J } e _ { R } } + e _ { R } , \quad } & { \quad \quad \quad e _ { \mathrm { e n d } } = ( 1 + \| Z _ { 1 } \| ^ { 2 } ) D _ { \mathrm { e n d } } . } \end{array}\tag{F.4}
$$

Then

$$
\| C _ { 1 } - \widetilde { C } \| \le D _ { \mathrm { e n d } } , \qquad \| M _ { 1 } - \widetilde { M } \| \le e _ { \mathrm { e n d } } .\tag{F.5}
$$

To prove this, rewrite (F.3) as $R = A ^ { - 1 / 2 } \{ ( H _ { B } { - } I ) T ^ { \mathrm { r e s } } { - } H _ { B } Y W ^ { \mathrm { r e s } } \}$ . For every unit vector $x ,$ Minkowski’s inequality in endpoint probability space gives

$$
\begin{array} { r } { \sqrt { \mathbb { E } _ { 1 } \| R x \| ^ { 2 } } \leq b _ { * } \{ \beta _ { * } \sqrt { \mathbb { E } _ { 1 } \| T ^ { \mathrm { r e s } } x \| ^ { 2 } } + \kappa _ { * } v _ { * } \sqrt { \mathbb { E } _ { 1 } \| W ^ { \mathrm { r e s } } x \| ^ { 2 } } \} \leq r _ { \mathrm { m s } } . } \end{array}
$$

The pointwise alternative gives $r _ { \mathrm { p t } }$ under the same probability law, with no extra density factor. Thus $0 \preceq \mathbb { E } _ { 1 } R ^ { \top } V R \preceq e _ { R } I$ . Quadratic-form Cauchy–Schwarz bounds the cross term in (F.2) by $2 { \sqrt { q _ { J } e _ { R } } }$ . Its density term lies between $- a _ { \mathrm { d e n } } \bar { Q }$ and $a _ { \mathrm { d e n } } \bar { Q }$ . Adding proves the raw bound. Congruence by $T _ { 1 }$ proves the normalized bound since $\| T _ { 1 } \| ^ { 2 } = 1 + \| Z _ { 1 } \| ^ { 2 }$ . It also bounds every raw response’s error divided by its endpoint Full loss.

All moments above are explicit finite sums. They can also be reconstructed by coeficient extraction without enumerating count configurations. For r distinct specified row labels in a group and $q$ inverse factors, use the polynomial

$$
F _ { r , q } ( z ) = \sum _ { k = 0 } ^ { n } { \binom { n } { k } } ( p + c _ { 0 } k ) ^ { 1 - q } \frac { ( k ) _ { r } } { ( n ) _ { r } } z ^ { k } .
$$

An impossible row-label request is zero before any factorial division. The expectation of a product with group profiles $( r _ { j } , q _ { j } )$ is $\begin{array} { r } { \left[ z ^ { s } \right] \prod _ { j } F _ { r _ { j } , q _ { j } } ( z ) } \end{array}$ divided by $[ z ^ { s } ] F _ { 0 , 0 } ( z ) ^ { d }$ . Repeated row selectors collapse to one selector. Expand $J _ { S } , D _ { S } ^ { \mathrm { f i t } }$ and $t _ { S }$ from their displayed formulas, then apply this rule term by term. This reconstructs $\bar { Q }$ and its score moment. For the residual moment, the factor A in $( D ^ { \mathrm { f i t } } ) ^ { \top } A D ^ { \mathrm { f i t } }$ cancels one inverse factor. Its selector and inverse degrees are at most (4, 3), or (5, 4) after score insertion. If column a is in group h with signed whitened amplitude $\alpha _ { a }$ , then

$$
( W _ { S } ^ { \mathrm { r e s } } ) _ { i a } = { \bf 1 } _ { i } { \bf 1 } _ { a } E _ { i h } \alpha _ { a } / ( p + c _ { 0 } K _ { h } ) ,
$$

which gives degrees at most (3, 2), or (4, 3) with the score. These formulas specify every residual moment used in (F.4); an endpoint covariance is not an input.

For completeness, the supplied-penalty correction is also exact. If $\bar { L } _ { 1 } = \mathbb { E } _ { 1 } L _ { 1 , S }$ and $B _ { \mathrm { m e a n } } = \bar { L } _ { 1 } - L _ { * , 1 }$ subtract $\Delta ^ { \top } V B _ { \mathrm { m e a n } } + B _ { \mathrm { m e a n } } ^ { \top } V \Delta$ from (F.1) for Full-centered MSE, and subtract $B _ { \mathrm { m e a n } } ^ { \top } V B _ { \mathrm { m e a n } }$ as well for centered covariance. Their true risks receive exactly the same corrections, so the remainder (F.4) is unchanged. The mean can be computed exactly: with $G = Z _ { 1 } ^ { \top } Z _ { 1 }$ and $\begin{array} { r } { c _ { j } = \binom { m - j } { s - j } } \end{array}$ ,

$$
N _ { 1 } = \sum _ { j } c _ { j } [ t ^ { j } ] \operatorname * { d e t } ( P _ { 1 } + t G ) , \qquad \bar { L } _ { 1 } = \frac { \sum _ { j } c _ { j } [ t ^ { j } ] \{ t \operatorname { a d j } ( P _ { 1 } + t G ) \} } { N _ { 1 } } Z _ { 1 } ^ { \top } .
$$

The determinant identity follows by counting how many s-sets contain each row minor. Diferentiating it with respect to $Z _ { 1 }$ , while holding $P _ { 1 }$ fixed, gives the mean formula. Expanding the squared fits gives the corrections above. Under exact calibration $B _ { \mathrm { m e a n } } = 0$ . This distinction prevents a supplied fit’s mean defect from being silently called sampling covariance.

## F.4 Explicit majorants and a terminating radius choice

Normalize c = max<sub>j</sub> $W _ { j j } = 1$ . Then $\| W \| \leq d$ and $\| V \| \le w = d / \lambda _ { 0 }$ . Fix a budget and put

$$
z = \sqrt { n / \lambda _ { 0 } } , \quad r _ { 0 } = \operatorname* { m i n } \{ z / 2 , p / 2 , 1 \} , \quad p _ { f } = p / 2 , \quad z _ { + } = z + r _ { 0 } , \quad t _ { + } = \sqrt { 1 + z _ { + } ^ { 2 } } .
$$

For $h = \| E \| _ { F } + \| \Delta P \| \le r _ { 0 }$ , the straight feature/penalty path is full rank with penalty at least $p _ { f } I .$ Define

$$
\begin{array} { c c } { { b _ { 0 } = ( 2 \sqrt { p } ) ^ { - 1 } + 1 / 2 , } } & { { \qquad b _ { f } = ( 2 \sqrt { p _ { f } } ) ^ { - 1 } + 1 / 2 , } } \\ { { \phantom { b _ { 0 } = } C _ { D } = 5 / ( 4 p _ { f } ) + 5 / 4 + 1 / ( 2 p _ { f } ^ { 3 / 2 } ) , } } & { { \qquad C _ { F } = t _ { + } C _ { D } + b _ { f } z _ { + } , } } \\ { { L _ { \mathrm { l a w } } = 2 \sqrt { d } / \sqrt { p _ { f } } + d / p _ { f } , } } & { { \qquad A _ { M } = 2 b _ { f } t _ { + } C _ { F } + b _ { f } ^ { 2 } t _ { + } ^ { 2 } L _ { \mathrm { l a w } } / 2 , } } \\ { { Q _ { B } = C _ { F } + b _ { f } t _ { + } L _ { \mathrm { l a w } } , } } & { { \qquad L = w ( A _ { M } + Q _ { B } ^ { 2 } r _ { 0 } ) . } } \end{array}
$$

Then, uniformly over the normalized PSD queries,

$$
\begin{array} { r } { \| M _ { 1 } - M _ { 0 } \| \le \eta ( h ) : = L h . } \end{array}\tag{F.6}
$$

Here this statement can refer either to centered covariance or Full-centered MSE with a supplied penalty. To check it, the selected and Full derivative bounds from Appendix E give $\| \dot { F } _ { S } \| \le C _ { D } h$ for $F _ { S } = L _ { S } - L ,$ ∗ and $\| F _ { S } \| \le b _ { f }$ . The root derivative gives $\| ( F _ { S } T ) ^ { \bullet } \| \le C _ { F } h$ . The feature and penalty parts of the log-determinant derivative are at most $2 { \sqrt { d } } \| E \| _ { F } / { \sqrt { p _ { f } } }$ and $d \| \Delta P \| / p _ { f }$ . Thus total variation is at most $L _ { \mathrm { l a w } } h / 2$ by (E.8). The PSD MSE atoms give the bound wA<sub>M</sub>h. At the base $\mathbb { E } _ { 0 } F _ { S } = 0$ , so $\| \mathbb { E } _ { 1 } F _ { 1 , S } T _ { 1 } \| \le$ $Q _ { B } h$ . Centering subtracts a PSD mean square of norm at most $\because Q _ { B } ^ { 2 } h ^ { 2 }$ . This proves (F.6), including along a path that is not itself calibrated.

The endpoint remainder has an explicit common majorant. For $0 \leq h \leq r _ { 0 }$ put

$$
\begin{array} { l l l } { { v _ { h } = h / \sqrt { p } , ~ } } & { { ~ u _ { h } = 2 v _ { h } + h / p , ~ } } & { { ~ \delta _ { h } = u _ { h } + v _ { h } ^ { 2 } , } } \\ { { { } } } & { { { } } } & { { } } \\ { { \rho _ { h } = v _ { h } ^ { 2 } + \displaystyle \sum _ { j = 2 } ^ { d } { \binom { d } { j } } \delta _ { h } ^ { j } , ~ } } & { { ~ r _ { h } = \frac { \delta _ { h } \left( v _ { h } + u _ { h } \right) + v _ { h } ^ { 2 } } { \sqrt { p } \left( 1 - \delta _ { h } \right) } , ~ } } & { { ~ } } \\ { { { } } } & { { { } } } & { { } } \\ { { q _ { h } = \left( \displaystyle \frac { 1 + \delta _ { h } } { 1 - \delta _ { h } } \right) ^ { d } , ~ } } & { { ~ d _ { h } = \left( v _ { h } + u _ { h } \right) / \sqrt { p } , ~ } } & { { ~ j _ { h } = b _ { 0 } + d _ { h } . } } \end{array}
$$

When $\delta _ { h } < 1$ , define

$$
\bar { e } ( h ) = [ 1 + ( z + h ) ^ { 2 } ] w \left\{ \frac { 2 \rho _ { h } j _ { h } ^ { 2 } } { ( 1 - \delta _ { h } ) ^ { d } } + 2 \sqrt { q _ { h } } j _ { h } r _ { h } + r _ { h } ^ { 2 } \right\} .\tag{F.7}
$$

Then $e _ { \mathrm { e n d } } ~ \le ~ \bar { e } ( h )$ , and $\bar { e } ( h ) = O ( h ^ { 2 } )$ . Indeed $\| A ^ { - 1 } \| \le 1 / p , \| U \| \le 1 , \| Y \| \le v _ { h }$ , tr $Y Y ^ { \top } \le v _ { h } ^ { 2 }$ , and $\| B _ { 1 } \| \le u _ { h }$ . The Neumann bound now gives $\| R _ { S } \| \le r _ { h }$ , while $\mu \geq ( 1 - \delta _ { h } ) ^ { d }$ and $| \bar { w } | \le \rho _ { h }$ . Also $\| J _ { S } \| \leq j _ { h }$ and the density is at most $q _ { h }$ . These bounds control the three terms of (F.2) by the three terms in braces. They also dominate the individual choices in (F.4): its pointwise alternative is no larger than $r _ { h }$ , and its first PSD averaging alternative gives $q _ { J } \leq q _ { h } w j _ { h } ^ { 2 }$ . This proves the stated domination, not just the existence of another second-order error estimate. Every scalar expression in (F.7) is nondecreasing on this domain.

Let ${ \mathcal { E } } _ { I }$ be one of the spaces in Theorem 7.3, and let $P _ { 0 } ^ { I } , P _ { 1 } ^ { I }$ project onto $\mathcal { E } _ { I }$ and $K _ { 1 } ^ { 1 / 2 } \mathcal { E } _ { I }$ . Since $K _ { 0 } ^ { 1 / 2 } = I$ on every contrast, the metric-root bound and its inverse identity give

$$
\| P _ { 1 } ^ { I } - P _ { 0 } ^ { I } \| \le f ( h ) : = \frac { \omega ( h ) } { 1 - \omega ( h ) } , \qquad \omega ( h ) = z _ { + } h < 1 .\tag{F.8}
$$

This follows by resolving the parallel and perpendicular parts of $K _ { 1 } ^ { 1 / 2 } | _ { \varepsilon _ { I } ; }$ its least singular value is at least $1 - \omega ( h )$ . Write $h _ { 0 } = \alpha _ { s }$ and let $\gamma$ be the uniform base top-to-complement gap. When $f ( h ) < 1$ , projection of a base top vector onto the endpoint trial space has base Rayleigh quotient at least $h _ { 0 } ( 1 - f ( h ) ^ { 2 } )$ . A unit vector in its complement has at most $f ( h ) ^ { 2 }$ energy in $\mathcal { E } _ { I }$ , so its base quotient is at most $h _ { 0 } - \gamma + \gamma f ( h ) ^ { 2 }$ Therefore the finite quantities from Theorem 7.3 satisfy

$$
g \ge \gamma - ( h _ { 0 } + \gamma ) f ( h ) ^ { 2 } - 2 \eta ( h ) - 4 \bar { e } ( h ) ,
$$

$$
B \leq \eta ( h ) + 2 h _ { 0 } f ( h ) + 2 \bar { e } ( h ) .\tag{F.9}
$$

(F.10)

The four remainder terms in the first line account for two compressed surrogate errors and the explicit 2e margin. For the second line, subtract the zero base cross block. The two projector changes contribute at most $2 h _ { 0 } f ( h )$ ; the operator change and surrogate error give the rest.

For all contrasts use $\gamma = \alpha _ { s } - \beta _ { s } > 0$ . For a fixed threshold $\vartheta > 0$ , use $\gamma = \operatorname* { m i n } \{ \alpha _ { s } \vartheta , \alpha _ { s } - \beta _ { s } \} > 0$ Choose the first integer $j \geq 0$ such that $h _ { j } = r _ { 0 } 2 ^ { - j }$ satisfies

$$
\delta _ { h _ { j } } < 1 , \quad \omega ( h _ { j } ) < 1 , \quad f ( h _ { j } ) < 1 , \quad ( h _ { 0 } + \gamma ) f ( h _ { j } ) ^ { 2 } + 2 \eta ( h _ { j } ) + 4 \bar { e } ( h _ { j } ) \le \gamma / 2 .\tag{F.11}
$$

The search terminates because every error term tends to zero. Monotonicity makes this a suficient closed ball in all joint directions. On it $g \geq \gamma / 2 , B = O ( h )$ , and the main theorem gives quadratic regret and leakage $O ( h + { \sqrt { \rho } } )$ for every response of deficit at most $\rho .$ None of these constants uses a query diagonal gap.

Finally, use the calibration constants from Appendix E. At the balanced base, set

$$
r _ { \mathrm { c a l } , s } = \frac { 1 } { 2 } \sqrt { n / \lambda _ { 0 } } , \qquad L _ { X , s } = 2 v { \mathcal L } _ { s } ,
$$

where v and $\mathcal { L } _ { s }$ are defined in (E.1)–(E.3) of that appendix. These are separate from this appendix’s joint radius $r _ { 0 } .$ . On the feature ball of radius $r _ { \mathrm { c a l } , s }$ , exact calibration gives $\| \Delta P \| \leq \| \Delta P \| _ { F } \leq L _ { X , s } \epsilon .$ Consequently the explicit positive feature radius

$$
r _ { \mathrm { u n i f } } = \operatorname* { m i n } _ { d \leq s < m } \left\{ r _ { \mathrm { c a l } , s } , \frac { h _ { j , s } } { 1 + L _ { X , s } } \right\}\tag{F.12}
$$

works for all budgets and all queries with $\operatorname* { m a x } _ { j } W _ { j j } = 1$ . For a general scale $c > 0 .$ risk and error bounds scale by c and leakage uses the normalized deficit $\rho / c .$ . This radius is not uniform as $n , d , \lambda _ { 0 }$ vary, as the sector gap vanishes, or as $\vartheta$ tends to zero. No arithmetic or calibration solver error is implicit in it; those would require additional verified error bounds. The finite endpoint formulas remain valid outside this suficient neighborhood, though their bounds may be loose.

## G The gain under recalibration and exact local certificates

Section 7 proves Theorem 7.4 with explicit constants. The shared-subset operator identity keeps a factor t in the perturbation bound; a comparison of the two base deficits then controls the diference of sharp risks. Section 6 gives the separate positive mean-share result at the balanced base. The following subsections sharpen the constants for the three exact certificates in Table 2. Throughout, $n , d \geq 2 , d \leq s < n d ,$ and the base consists of signed unit coordinate replicas. The physical target is the fixed matrix $\lambda _ { 0 } I , \lambda _ { 0 } > 0$ Every endpoint uses its unique exactly calibrated penalty in both the original-row determinant law and the ridge fit. Only the subset is random. The query is any fixed nonzero PSD matrix. All unmarked norms are operator norms; derivatives of the calibration map use Frobenius norm on their input and output.

## G.1 A sharper finite spectral bound

To certify a radius R, we need bounds $L _ { R } , L _ { D } , L _ { M }$ on the entire ball. The spectral gates below state what those bounds must satisfy. We then construct them in this order: (G.5) places the feature ball inside the calibration region; (G.9) computes derivative energies at the base; and (G.14) extends the derivative bounds over the ball. Equations (G.11) and (G.8) turn these bounds into $L _ { R } , L _ { D } , L _ { M }$ , which enter (G.2).

The following refinement of the two-eigenvector argument will be used for the exact certificates. Suppose constants $L _ { R } , L _ { D } , L _ { M } = L _ { R } + T L _ { D }$ satisfy (7.10) on a proposed radius R. They may be sharper than (7.9). Write $f ( t ) = D - Q t$ and $\gamma _ { 0 } = \gamma ( 0 )$ . Choose

$$
\begin{array} { r l } { 0 < g _ { T } \leq \displaystyle \operatorname* { m i n } _ { [ 0 , T ] } ( \alpha - \beta ) , \qquad } & { k _ { C } \geq \displaystyle \operatorname* { s u p } _ { [ 0 , T ] } f / \alpha , } \\ { a _ { - } \geq \displaystyle \operatorname* { s u p } _ { [ 0 , T ] } \operatorname* { m a x } \{ 0 , - j _ { A \beta } , - j _ { A \gamma } \} , \qquad } & { b _ { + } \geq \displaystyle \operatorname* { s u p } _ { [ 0 , T ] } \operatorname* { m a x } \{ 0 , j _ { B \beta } , j _ { B \gamma } \} , } \end{array}
$$

where

$$
\begin{array} { l l } { { j _ { A \beta } = B _ { 1 } + B _ { 2 } t + f B _ { 0 } / A , ~ } } & { { ~ j _ { A \gamma } = ( - 2 + t ) \gamma _ { 0 } + f \gamma _ { 0 } / A , } } \\ { { { } } } & { { { } } } \\ { { j _ { B \beta } = B _ { 1 } + B _ { 2 } t + f \beta / \alpha , ~ } } & { { ~ j _ { B \gamma } = ( - 2 + t ) \gamma _ { 0 } + f \gamma / \alpha . } } \end{array}
$$

Let $\Pi _ { M }$ denote the fixed base mean-sector projector. With the base deficits used above, direct subtraction gives

$$
N = t ( f / A ) H _ { A } + R _ { A } = t ( f / \alpha ) H _ { B } + R _ { B } , \quad R _ { A } \succeq - t c a _ { - } \Pi _ { M } , \quad R _ { B } \preceq t c b _ { + } \Pi _ { M } .
$$

For clarity, the remainders vanish on contrasts. On means they are $R _ { A } / t = j _ { A \beta } P _ { W } + j _ { A \gamma } R _ { W }$ and $R _ { B } / t = j _ { B \beta } P _ { W } + j _ { B \gamma } R _ { W } ;$ ; their bounds use $P _ { W } + R _ { W } \preceq c I$

Put $e _ { R } = L _ { R } \epsilon , e _ { M } = L _ { M } \epsilon$ . If $\boldsymbol { e } _ { R } < \boldsymbol { g }$ and $e _ { M } \ < g _ { T }$ , projecting the two endpoint top-eigenvector equations onto the base mean sector gives

$$
\left\| \Pi _ { M } u \right\| \le \frac { e _ { R } } { g - e _ { R } } , \qquad \left\| \Pi _ { M } v \right\| \le \frac { e _ { M } } { g _ { T } - e _ { M } } .
$$

Indeed the ridge eigenvalue is at least $( A { - } e _ { R } ) c ,$ whereas its base mean block is at most $B _ { 0 } c I ;$ the mixture argument uses $\beta ( t ) c I .$ . The forcing terms have norms at most $e _ { R } c , e _ { M } c .$ . With $\mathcal { G } = h _ { R } - h _ { t } - t f c ,$ the same two variational comparisons as before are

$$
\mathcal { G } \leq \| E _ { B } - E _ { A } \| - u ^ { \top } N u , \qquad \mathcal { G } \geq - \| E _ { B } - E _ { A } \| - v ^ { \top } N v .
$$

In the first, discard the nonnegative proportional $H _ { A }$ term. In the second, use $v ^ { \top } H _ { B } v \le 2 e _ { M } c$ . This proves

$$
| \mathcal { G } | \leq t c \Phi ( \epsilon ) , \quad \Phi ( \epsilon ) = L _ { D } \epsilon + \operatorname* { m a x } \left\{ a _ { - } \left( \frac { L _ { R } \epsilon } { g - L _ { R } \epsilon } \right) ^ { 2 } , 2 k _ { C } L _ { M } \epsilon + b _ { + } \left( \frac { L _ { M } \epsilon } { g _ { T } - L _ { M } \epsilon } \right) ^ { 2 } \right\} .\tag{G.1}
$$

For fixed constants, $\Phi ( \epsilon ) / \epsilon$ is nondecreasing: each nonlinear term divided by ϵ is a nonnegative multiple of $\epsilon / ( g - L \epsilon ) ^ { 2 }$ . Therefore the gates

$$
L _ { R } R < g , \qquad L _ { M } R < g _ { T } , \qquad \Phi ( R ) \le D / 4\tag{G.2}
$$

give (7.6) on the entire ball with $C = \Phi ( R ) / R$ and a gain of at least $D t c / 4$

All scalar choices can be made with rational polynomial operations. Take $\alpha _ { \mathrm { m i n } } = \mathrm { m i n } _ { [ 0 , T ] } \alpha , g _ { T } =$ min $_ { [ 0 , T ] } ( \alpha - \beta )$ , and $k _ { C } = D / \alpha _ { \mathrm { m i n } }$ . A quadratic minimum uses its endpoints and any interior vertex. The two $j _ { A }$ are afine, so their endpoints give $a _ { - } .$ . For $b _ { + }$ , multiply each $j _ { B }$ by $\alpha _ { \mathrm { { ; } } }$ , take the maximum of zero and all Bernstein coeficients of the resulting polynomials on $[ 0 , T ]$ , then divide by $\alpha _ { \mathrm { m i n } }$ . Specifically, for $\begin{array} { r } { p ( t ) = \sum _ { j = 0 } ^ { N } p _ { j } t ^ { j } } \end{array}$ , those coeficients are $\begin{array} { r } { \sum _ { j = 0 } ^ { i } p _ { j } T ^ { j } \binom { i } { j } / \binom { N } { j } , \ 0 \stackrel { . . } { \ } \stackrel { . } { \ } i \leq N } \end{array}$ . The nonnegative Bernstein basis sums to one, so this is a uniform bound.

## G.2 A finite calibration derivative diference

We supply the local calibration bound needed by the sharper certificate; a bound on two derivative norms alone would not sufice. In this and the remaining subsections, write $\zeta = n / \lambda _ { 0 }$ (distinct from the

mean-loss coeficient $b ) , z = \sqrt { \zeta } , p _ { 0 } = r / \lambda _ { 0 }$ , and $q _ { 0 } ^ { \mathrm { c a l } } = \zeta / p _ { 0 }$ . Here rI is the physical base penalty. The marked-set distribution in Section 2 has cardinality K with weights

$$
{ \binom { d } { k } } { \binom { m - k } { s - k } } ( q _ { 0 } ^ { \mathrm { c a l } } ) ^ { k } , \quad 0 \leq k \leq d .
$$

Define its positive curvatures and the base derivative coeficients by

$$
\begin{array} { r l r l r l } & { \lambda _ { \parallel } = \mathrm { V a r } ( K ) / d , \quad } & & { \lambda _ { \perp } = \mathbb { E } [ K ( d - K ) ] / [ d ( d - 1 ) ] , \quad } & & { \mu _ { * } = \operatorname* { m i n } \{ \lambda _ { \parallel } , \lambda _ { \perp } \} , } \\ & { t _ { j } = p _ { 0 } / \zeta - p _ { 0 } / [ ( 1 + \zeta ) ^ { 2 } \lambda _ { j } ] , \quad } & & { L _ { 0 } ^ { \mathrm { c a l } } = \operatorname* { m a x } _ { j } \lvert t _ { j } \rvert . } \end{array}
$$

The marked covariance has eigenvalues $\lambda _ { \parallel }$ on scalar directions and $\lambda _ { \perp }$ on traceless diagonal directions. Diferentiating the moment equation gives

$$
D P ( \zeta I ) [ F ] = t _ { \perp } \{ F - \mathrm { t r } ( F ) I / d \} + t _ { \parallel } \mathrm { t r } ( F ) I / d .\tag{G.3}
$$

Orthogonal equivariance extends the diagonal formula to all symmetric $F .$ Positivity of both curvatures follows from the full support of the marked-set law.

Choose a log radius $a > 0$ , upper bounds $E _ { a } \geq e ^ { a }$ and $\chi \geq e ^ { 2 { \sqrt { d } } a }$ , and

$$
0 < \delta \leq \operatorname* { m i n } \{ \zeta / 2 , \mu _ { * } a ( 1 + \zeta / 2 ) ^ { 2 } / ( 2 \chi ) \} .
$$

Set $\ell = \zeta - \delta , u = \zeta + \delta$ and define

$$
\begin{array} { r l r } & { R _ { \mathrm { m a x } } = ( 1 + \ell ) ^ { - 2 } , \ } & { R _ { 0 } = ( 1 + \zeta ) ^ { - 2 } , \ } \\ & { \Delta R _ { * } = \operatorname* { m a x } \{ R _ { \mathrm { m a x } } - R _ { 0 } , R _ { 0 } - ( 1 + u ) ^ { - 2 } \} , \ } & { \Delta P _ { * } = p _ { 0 } \operatorname* { m a x } \{ E _ { a } u / \zeta - 1 , 1 - \ell / ( E _ { a } \zeta ) \} , \ } \\ & { J _ { * } = \frac { E _ { a } - 1 } { q _ { 0 } ^ { \mathrm { G a l } } } + \frac { \Delta P _ { * } \chi R _ { \mathrm { m a x } } } { \mu _ { * } } + \frac { p _ { 0 } ( \chi - 1 ) R _ { \mathrm { m a x } } } { \mu _ { * } } + \frac { p _ { 0 } \Delta R _ { * } } { \mu _ { * } } , \ } \\ & { L _ { \mathrm { c a l } } = L _ { 0 } ^ { \mathrm { c a l } } + J _ { * } , \ } & { \ } \\ & { p = \operatorname* { m a x } \{ ( s - d + 1 ) / ( m - d + 1 ) , \ell / ( q _ { 0 } ^ { \mathrm { c a l } } E _ { a } ) \} . } \end{array}
$$

We claim, throughout $\| B - \zeta I \| _ { F } \leq \delta$

$$
\begin{array} { r } { P ( B ) \succeq p I , \qquad \| D P ( B ) - D P ( \zeta I ) \| _ { F \to F } \leq J _ { * } . } \end{array}\tag{G.4}
$$

Here are details that also cover repeated spectra. Within the radius-a log-parameter ball about $\log ( q _ { 0 } ^ { \mathrm { c a l } } ) I .$ each marked log weight changes by at most $\sqrt { d } a$ . Thus its normalized density ratio lies between $\chi ^ { - 1 }$ and $\chi .$ The identity $\operatorname { V a r } ( v ^ { \top } \mathbf { 1 } _ { J } ) = \operatorname* { m i n } _ { c } \mathbb { E } ( v ^ { \top } \mathbf { 1 } _ { J } - c ) ^ { 2 }$ places its covariance $\mathcal { C }$ between $\chi ^ { - 1 } \mathcal { C } _ { 0 }$ and $\chi \mathcal { C } _ { 0 }$ Coordinate swaps, as in the proof of $\left( \mathrm { E . 6 } \right)$ , extend the lower Hessian bound $\mu _ { * } / \chi$ to symmetric matrix directions. The target moment $B ( I + B ) ^ { - 1 }$ changes by at most $\delta / ( 1 + \zeta / 2 ) ^ { 2 } \leq \mu _ { * } a / ( 2 \chi )$ . On the log-ball boundary, the radial component of the gradient of the convex calibration objective is therefore at least $( \mu _ { * } / \chi ) a ^ { 2 } - ( \mu _ { * } a / ( 2 \chi ) ) a > 0$ . Its minimum on the closed ball is interior; uniqueness identifies it with the calibrated solution. In particular every odds eigenvalue lies in $[ q _ { 0 } ^ { \mathrm { c a l } } / E _ { a } , q _ { 0 } ^ { \mathrm { c a l } } E _ { a } ]$ , which proves the penalty floor.

In Gram eigenvalue coordinates $b _ { j }$ , the calibration Jacobian is

$$
J _ { \mathrm { c a l } } = \mathrm { d i a g } ( 1 / q _ { j } ) - \mathrm { d i a g } ( p _ { j } ) \mathcal { C } ^ { - 1 } \mathrm { d i a g } ( ( 1 + b _ { j } ) ^ { - 2 } ) .
$$

Write $J _ { \mathrm { c a l } , 0 }$ for its value at the balanced base. The covariance comparison gives $\| { \mathcal C } ^ { - 1 } \| \le \chi / \mu .$ <sub>∗</sub> and $\| \mathcal C ^ { - 1 } - \mathcal C _ { 0 } ^ { - 1 } \| \le ( \chi - 1 ) / \mu _ { * }$ . Subtract the baseline Jacobian by changing, in order, $1 / q _ { j } , p _ { j } , \mathcal { C } ^ { - 1 }$ , and the final diagonal factor. Their four bounds are exactly the four terms of $J _ { \ast } , \mathrm { s o } \ \| J _ { \mathrm { c a l } } - J _ { \mathrm { c a l , 0 } } \| \leq J _ { \ast }$ . For two

distinct Gram eigenvalues, join their vector to its coordinate swap. This segment stays in the Gram ball. With $e = ( e _ { i } - e _ { j } ) / \sqrt { 2 }$ , equivariance and integration give

$$
\frac { p _ { i } ( b ) - p _ { j } ( b ) } { b _ { i } - b _ { j } } - t _ { \perp } = \int _ { 0 } ^ { 1 } e ^ { \top } \{ J _ { \mathrm { c a l } } ( b _ { \mathrm { s w a p } } + x ( b - b _ { \mathrm { s w a p } } ) ) - J _ { \mathrm { c a l , 0 } } \} e d x .
$$

Thus the of-diagonal derivative coeficients satisfy the same bound. The diagonal and of-diagonal symmetric subspaces are orthogonal and invariant, so their bounds combine by a maximum. The baseline map (G.3) is independent of eigenbasis, and continuity covers repeated eigenvalues. This proves (G.4).

For a feature radius R impose

$$
R < z , \qquad ( 2 z + R ) R \leq \delta .\tag{G.5}
$$

Let $v = z + R , L _ { X } = 2 v L _ { \mathrm { c a l } } , L _ { X 0 } = 2 z L _ { 0 } ^ { \mathrm { c a l } }$ . The entire feature ball is full rank, lies in the Gram ball, and, for every unit Frobenius feature direction, satisfies

$$
\| \dot { P } \| _ { F } \leq L _ { X } , \qquad \| \dot { P } - \dot { P } _ { 0 } \| _ { F } \leq \Delta \dot { P } _ { * } = 2 v J _ { * } + 2 L _ { 0 } ^ { \mathrm { c a l } } R .\tag{G.6}
$$

The last inequality adds and subtracts $D P ( \zeta I ) [ Z ^ { \top } E + E ^ { \top } Z ]$ and uses the input diference bound 2R.   
Thus it covers noncommuting feature changes as well.

## G.3 Derivative energies of the probability-weighted atoms

We now construct finite constants for (7.10). At a design Z in the ball, keep $A _ { S } , L _ { S } , L , I _ { S } , T _ { Z }$ as defined above, and include physical scaling in the atoms:

$$
u _ { R , S } = ( L _ { S } - L ) T _ { Z } / \sqrt { \lambda _ { 0 } } , \qquad u _ { G , S } = ( L I _ { S } - L _ { S } ) T _ { Z } / \sqrt { \lambda _ { 0 } } .
$$

Dots in this subsection are derivatives in a unit Frobenius direction E. If $a _ { S } = \operatorname { t r } ( A _ { S } ^ { - 1 } { \dot { A } } _ { S } )$ and $\ell _ { S } =$ $a _ { S } - \mathbb { E } _ { p } a _ { S }$ , exact calibration gives

$$
\ell _ { S } = 2 \operatorname { t r } ( ( L _ { S } - L ) E ) + \operatorname { t r } \{ ( A _ { S } ^ { - 1 } - \mathbb { E } _ { p } A _ { S } ^ { - 1 } ) { \dot { P } } \} , \quad \eta _ { i } : = \partial \log \pi _ { i } = \mathbb { E } _ { p } [ \ell _ { S } \mid i \in S ] .\tag{G.7}
$$

Indeed the design part of $a _ { S }$ is $2 \operatorname { t r } ( L _ { S } E )$ and $\mathbb { E } _ { p } L _ { S } = L$ . Also $\dot { \pi } _ { i } = \mathbb { E } _ { p } [ { \mathbf { 1 } _ { i } } \ell _ { S } ]$ and $\dot { I } _ { S } = - I _ { S } \mathrm { d i a g } ( \eta _ { i } )$

Stack $\sqrt { p _ { S } } W ^ { 1 / 2 } u _ { R , S }$ into a linear response map $\boldsymbol { B } _ { R }$ and define $B _ { G }$ likewise. Its derivative stacks $\sqrt { p _ { S } } W ^ { 1 / 2 } V _ { R , S }$ , where

$$
V _ { R , S } = \dot { u } _ { R , S } + \ell _ { S } u _ { R , S } / 2 , \qquad V _ { G , S } = \dot { u } _ { G , S } + \ell _ { S } u _ { G , S } / 2 .
$$

Then $M _ { R } = B _ { R } ^ { \top } B _ { R }$ and $\mathcal { T } _ { t } = \mathcal { B } _ { R } ^ { \top } \mathcal { B } _ { G } + \mathcal { B } _ { G } ^ { \top } \mathcal { B } _ { R } + t \mathcal { B } _ { G } ^ { \top } \mathcal { B } _ { G }$ . If their norms are bounded by $\sqrt { c } r _ { * } , \sqrt { c } g _ { * }$ and derivative norms by $\sqrt { c } v _ { R } , \sqrt { c } v _ { G }$ , product diferentiation proves

$$
L _ { R } = 2 r _ { * } v _ { R } , \qquad L _ { D } = 2 ( v _ { R } g _ { * } + r _ { * } v _ { G } ) + 2 T g _ { * } v _ { G } , \qquad L _ { M } = L _ { R } + T L _ { D } .\tag{G.8}
$$

Let $E _ { a } , 1 \leq a \leq m d .$ be all coordinate feature directions. At the base define the two positive matrices

$$
H _ { R } = \sum _ { a } \mathbb { E } _ { 0 } [ V _ { R } ( E _ { a } ) ^ { \top } V _ { R } ( E _ { a } ) ] , \qquad H _ { G } = \sum _ { a } \mathbb { E } _ { 0 } [ V _ { G } ( E _ { a } ) ^ { \top } V _ { G } ( E _ { a } ) ] ,\tag{G.9}
$$

$$
v _ { R 0 } = \sqrt { d \| H _ { R } \| } , \qquad v _ { G 0 } = \sqrt { d \| H _ { G } \| } .
$$

For a unit direction $\begin{array} { r } { E = \sum _ { a } e _ { a } E _ { a } } \end{array}$ , vector Cauchy–Schwarz gives $\begin{array} { r } { V ( E ) ^ { \top } V ( E ) \preceq \sum _ { a } V ( E _ { a } ) ^ { \top } V ( E _ { a } ) } \end{array}$ Together with $W \preceq d c I$ , this proves the claimed base derivative bounds.

These finite matrices are particularly simple. Put $\Pi _ { M } = Z _ { 0 } Z _ { 0 } ^ { \top } / \zeta$ , $\mathrm { I } _ { C } = I - \Pi _ { M }$ and $\theta _ { 0 } = \sqrt { 1 + \zeta }$ Each H in (G.9) has

$$
H = h _ { C } \Pi _ { C } + h _ { M } \Pi _ { M } , \qquad \lVert H \rVert = \operatorname* { m a x } \{ h _ { C } , h _ { M } \} .\tag{G.10}
$$

To see this, first remove row signs by an orthogonal diagonal matrix. Within-group row permutations fix the base and force each diagonal block to be a linear combination of I and $\mathbf { 1 1 } ^ { \top }$ , and each of-diagonal group block to be constant. Flip one whole row group and its matching feature coordinate. This signed permutation fixes the base and forces of-diagonal blocks of H to vanish. Group permutations identify the remaining coeficients. The complete coordinate-direction sum is invariant under these orthogonal changes of direction basis. All row operations here are signed permutations, so they preserve or relabel the original-row law. Conjugating back restores arbitrary row signs.

For an explicit evaluation of (G.9), use (G.3), (G.7), and

$$
\begin{array} { r l r l } & { \dot { A } _ { S } = E ^ { \top } D _ { S } Z _ { 0 } + Z _ { 0 } ^ { \top } D _ { S } E + \dot { P } , \quad } & & { \dot { L } _ { S } = A _ { S } ^ { - 1 } ( E ^ { \top } D _ { S } - \dot { A } _ { S } L _ { S } ) , } \\ & { \dot { L } = ( I + \zeta I ) ^ { - 1 } ( E ^ { \top } - \dot { B } L ) , \quad } & & { \dot { B } = Z _ { 0 } ^ { \top } E + E ^ { \top } Z _ { 0 } , } \\ & { \dot { T } = \frac 1 2 \Pi _ { C } \dot { U } \Pi _ { C } + \frac { \Pi _ { C } \dot { U } \Pi _ { M } + \Pi _ { M } \dot { U } \Pi _ { C } } { 1 + \theta _ { 0 } } + \frac { \Pi _ { M } \dot { U } \Pi _ { M } } { 2 \theta _ { 0 } } , \quad } & & { \dot { U } = E Z _ { 0 } ^ { \top } + Z _ { 0 } E ^ { \top } . } \end{array}
$$

Here $\dot { U } = \partial ( T _ { Z } ^ { 2 } )$ . Diferentiate each atom as a product and add its half-score term. For the rational fixtures below, every squared energy lies in $\mathbb { Q } ( \theta _ { 0 } )$ . Indeed $Z _ { 0 } = U / \sqrt { \lambda _ { 0 } }$ for an integer replica matrix $U ;$ the selected and Full maps have this same scalar factor, while $\dot { L } _ { S } , \dot { L }$ are rational and $\dot { T }$ has that factor times a matrix in $\mathbb { Q } ( \theta _ { 0 } )$ . Thus each $V$ is $1 / \sqrt { \lambda _ { 0 } }$ times a matrix in that field. No numerical derivative or eigenvalue is needed.

Record the finite base norms

$$
l _ { 0 } = \operatorname* { m a x } _ { S } \| L _ { S , 0 } \| , \quad u _ { R 0 } = \operatorname* { m a x } _ { S } \| u _ { R , S , 0 } \| , \quad u _ { G 0 } = \operatorname* { m a x } _ { S } \| u _ { G , S , 0 } \| , \quad q _ { 0 } = \operatorname* { m a x } _ { S } \sqrt { \sum _ { a } \ell _ { S , 0 } ( E _ { a } ) ^ { 2 } } .
$$

The row Grams of these base fit and atom maps are diagonal because the groups have disjoint supports. Their norms and $q _ { 0 }$ therefore admit exact square-root bounds. Finally the base stack norms are at most $\sqrt { c A }$ and $\sqrt { c \operatorname* { m a x } \{ Q , B _ { 2 } , \gamma _ { 0 } \} }$ . For the latter, the $G$ mean block is $B _ { 2 } P _ { W } + \gamma _ { 0 } R _ { W }$ and its contrast coeficient is $Q .$ . Both mean coeficients are nonnegative. Once $v _ { R } , v _ { G }$ hold on the ball, integration gives valid stack bounds

$$
r _ { * } = \sqrt { A } + R v _ { R } , \qquad g _ { * } = \sqrt { \mathrm { m a x } \{ Q , B _ { 2 } , \gamma _ { 0 } \} } + R v _ { G } .\tag{G.11}
$$

## G.4 Finite transport of derivative energies

The following formulas bound diferences from the base at every point in the radius-R ball, in every unit feature direction. Constants are fixed on that ball. In addition to (G.5), use

$$
\begin{array} { r l r } & { \theta = \sqrt { 1 + v ^ { 2 } } , \quad \quad \pi _ { 0 } = s / m , \quad \quad q _ { \mathrm { s c o r e } } = \sqrt { d } \{ 2 / \sqrt { p } + L _ { X } / p \} , } & \\ & { \pi _ { f } = \pi _ { 0 } e ^ { - 2 q _ { \mathrm { s c o r e } } R } , \quad \quad h = e ^ { 2 q _ { \mathrm { s c o r e } } R } - 1 , \quad \quad d _ { p } = e ^ { q _ { \mathrm { s c o r e } } R } . } & \end{array}
$$

The determinant derivative bounds prove $| a _ { S } | \leq q _ { \mathrm { s c o r e } } , | \ell _ { S } | \leq 2 q _ { \mathrm { s c o r e } }$ and $e ^ { - 2 q _ { \mathrm { s c o r e } } R } \leq p _ { S } / p _ { S , 0 } \leq e ^ { 2 q _ { \mathrm { s c o r e } } R }$ They also prove $\pi _ { i } \geq \pi _ { f } > 0$ . Set

$$
D _ { R } = 5 / ( 4 p ) + 5 / 4 + L _ { X } / ( 2 p ^ { 3 / 2 } ) , \quad f _ { 0 } = z / ( 1 + z ^ { 2 } ) , \quad b _ { R 0 } = \sqrt { \lambda _ { 0 } } u _ { R 0 } , \quad b _ { G 0 } = \sqrt { \lambda _ { 0 } } u _ { G 0 } .
$$

The last two constants bound the raw base atoms since $\| T _ { 0 } ^ { - 1 } \| \leq 1$ . Use the following nonnegative bounds, with $\Delta \dot { P } _ { * }$ from (G.6):

$$
\begin{array} { r l r l } & { \Delta A = ( 2 z + R + L _ { X } ) R , } & & { \Delta I _ { A } = \Delta A / ( p p _ { 0 } ) , } \\ & { \Delta L = ( D _ { R } - 5 / 4 ) R , } & & { \Delta F = 5 R / 4 , } \\ & { \Delta \dot { L } = \Delta I _ { A } [ 1 + ( 2 z + L _ { X 0 } ) l _ { 0 } ] } \\ & { \qquad + \left\{ ( 2 R + \Delta \dot { P } _ { * } ) l _ { 0 } + ( 2 v + L _ { X } ) \Delta L \right\} / p , } \\ & { \mathsf { f } _ { \mathrm { f l o o r } } = 1 + ( z - R ) ^ { 2 } , \qquad } & & { \Delta I _ { F } = ( 2 z + R ) R / [ f _ { \mathrm { f l o o r } } ( 1 + z ^ { 2 } ) ] , } \\ & { \Delta \dot { F } = \Delta I _ { F } ( 1 + 2 z f _ { 0 } ) + \{ 2 R f _ { 0 } + 2 v \Delta F \} / f _ { \mathrm { f l o o r } } . } \end{array}\tag{G.12}
$$

Here $\Delta I _ { A }$ bounds the selected inverse diference, $\Delta L$ the selected fit diference, and $\Delta F$ the Full fit diference. Resolvent subtraction gives $\Delta I _ { A }$ . Integrating the selected and Full derivatives gives $\Delta L , \Delta F$ For the derivative diference, write $\dot { L } _ { S } ~ = ~ A _ { S } ^ { - 1 } C _ { S }$ with $\begin{array} { r } { C _ { S } = E ^ { \top } D _ { S } - \dot { A } _ { S } L _ { S } } \end{array}$ . Subtract as $( A _ { S } ^ { - 1 } -$ $A _ { S , 0 } ^ { - 1 } ) C _ { S , 0 } + A _ { S } ^ { - 1 } ( C _ { S } - C _ { S , 0 } )$ . The base factor has norm at most $1 + ( 2 z + L _ { X 0 } ) l _ { 0 }$ and the other diference at most $( 2 R + \Delta \dot { P } _ { * } ) l _ { 0 } + ( 2 v + L _ { X } ) \Delta L$ . This proves the selected formula, preserving matrix order. The Full formula is the same subtraction with constant penalty I, endpoint inverse floor $f _ { \mathrm { f l o o r } }$ , and base fit norm $f _ { 0 }$

For the centered score and inclusions, define

$$
\begin{array} { l c r l } & { \Delta a = \sqrt { d } \{ \Delta I _ { A } ( 2 z + L _ { X 0 } ) + ( 2 R + \Delta \dot { P } _ { * } ) / p \} , } \\ & { \Delta \ell = 2 \Delta a + h q _ { 0 } , \quad \quad \quad \quad \quad \quad q = q _ { 0 } + \Delta \ell , } \\ & { \Delta \eta = \Delta \ell + 2 h q _ { 0 } \pi _ { 0 } / \pi , \quad \quad \quad \quad \Delta I = h / \pi _ { f } , \quad \quad B _ { I } = 1 / \pi _ { f } , } \\ & { \Delta \dot { I } = \Delta I q _ { 0 } + B _ { I } \Delta \eta , \quad \quad \quad \quad \quad \Delta H = \Delta F B _ { I } + f _ { 0 } \Delta I , \quad \quad } \\ & { \Delta \dot { H } = \Delta \dot { F } B _ { I } + ( 5 / 4 ) \Delta I + \Delta F q B _ { I } + f _ { 0 } \Delta \dot { I } . } \end{array}\tag{G.13}
$$

Trace Cauchy–Schwarz bounds $| a _ { S } - a _ { S , 0 } |$ by $\Delta a$ . Centering costs twice this bound. The remaining law change is $( \mathbb { E } _ { p } - \mathbb { E } _ { 0 } ) \ell _ { S , 0 }$ , whose absolute value is at most $h q _ { 0 }$ . This proves the bound on $\Delta \ell$ and $| \ell _ { S } | \le q$ For $\eta _ { i } = \mathbb { E } _ { p } [ \ell \mid i \in S ]$ , first change the score inside the endpoint conditional expectation, at cost $\Delta \ell .$ Then change the conditional average of the base score. Its numerator change is at most $h \pi _ { 0 } q _ { 0 }$ , and its denominator change costs another $h \pi _ { 0 } q _ { 0 } / \pi _ { f }$ after division. This proves $\Delta \eta ;$ the entire score diference is not divided by the inclusion floor. Since $| \pi _ { i } - \pi _ { 0 } | \leq h \pi _ { 0 } .$ , reciprocals difer by at most $\Delta I .$ Subtracting $\dot { I } _ { S } = - I _ { S } \mathrm { d i a g } ( \eta _ { i } )$ gives $\Delta \dot { I } .$ The last two formulas follow by subtracting $L I _ { S }$ and its derivative as ordered products. Here $\| \dot { I } _ { S } \| \le q / \pi _ { f }$ and the base Full derivative is at most $5 / 4$

Finally define the metric and atom diferences

$$
\begin{array} { l l } { { \Delta T = v R , } } & { { \Delta \dot { T } = R + z \Delta T , } } \\ { { \Delta F _ { n } = \Delta L + \Delta F , } } & { { \Delta G = \Delta H + \Delta L , } } \\ { { \Delta u _ { n } = ( \Delta F _ { R } \theta + b _ { R 0 } \Delta T ) / \sqrt { \lambda _ { 0 } } , } } & { { \Delta u _ { G } = ( \Delta G \theta + b _ { G 0 } \Delta T ) / \sqrt { \lambda _ { 0 } } , } } \\ { { d _ { R 0 } = 5 / 4 + 5 / ( 4 p _ { 0 } ) + L _ { X 0 } / ( 2 p _ { 0 } ^ { 3 / 2 } ) , } } & { { d _ { G 0 } = ( 5 / 4 ) / \pi _ { 0 } + j _ { 0 } q _ { 0 } / \pi _ { 0 } + d _ { R 0 } - 5 / 4 , } } \\ { { \Delta { \dot { u } _ { R } } = \{ ( \Delta \dot { L } + \Delta \dot { F } ) \theta + d _ { R 0 } \Delta T + \Delta F _ { R } v + b _ { R 0 } \Delta \dot { T } \} / \sqrt { \lambda _ { 0 } } , } } & { { } } \\ { { \Delta { \dot { u } _ { G } } = \{ ( \Delta \dot { H } + \Delta \dot { L } ) \theta + d _ { G 0 } \Delta T + \Delta G v + b _ { G 0 } \Delta \dot { T } \} / \sqrt { \lambda _ { 0 } } , } } & { { } } \\ { { x _ { R } = \Delta { \dot { u } _ { R } } + ( \Delta \dot { \ell } / 2 ) u _ { R 0 } + ( q / 2 ) \Delta u _ { R } , } } & { { x _ { G } = \Delta { \dot { u } _ { G } } + ( \Delta \dot { \ell } / 2 ) u _ { G 0 } + ( q / 2 ) \Delta u _ { G } , } } \\ { { v _ { R } = d _ { p } v _ { R 0 } + \sqrt { d } x _ { R } , } } & { { v _ { G } = d _ { p } v _ { G 0 } + \sqrt { d } x _ { G } . } } \end{array}
$$

To check the metric bound, subtract the $\mathrm { S y } \mathbf { \dot { \Omega } }$ lvester equations for $\dot { T }$ and $\dot { T } _ { 0 }$ . Since both metrics are at least I, the Sylvester inverse has norm at most $1 / 2$ . The right side consists of the derivative of $Z Z ^ { \top } - Z _ { 0 } Z _ { 0 } ^ { \top }$ of norm at most $2 R ,$ and two products of the metric diference with $\dot { T } _ { 0 }$ , of total norm at most $2 z \Delta T .$ . This gives $\Delta \dot { T }$ . Integrating $\lVert \dot { T } \rVert \leq v$ gives ∆T. No commutation is assumed. Subtracting the products $F _ { R } T _ { Z }$ $G T _ { Z }$ and their derivatives proves the four atom bounds. The base HT derivative uses the actual base inclusion derivative, bounded by ${ { q } _ { 0 } } / { { \pi } _ { 0 } }$ , in $d _ { G 0 }$ . Subtracting the half-score product then gives $x _ { R } , x _ { G }$ as bounds for $V _ { R } - V _ { R , 0 }$ and $V _ { G } - V _ { G , 0 }$ . In the derivative stack, split ${ \sqrt { p _ { S } } } V _ { S }$ into its base-atom part and its diference part. The first has norm at most $d _ { p }$ times its base stack norm since $p _ { S } / p _ { S , 0 } \leq d _ { p } ^ { 2 } ;$ the second is bounded using $W \preceq d c I$ . The triangle inequality proves the final two formulas in $\left( \mathrm { G . 1 4 } \right)$

Equations (G.14), (G.11) and (G.8) now give finite constants satisfying (7.10) on the entire ball. Together with (G.5) and (G.2), they prove both the two-sided gain modulus and strict improvement. This derivation has included calibration, fits, law, actual inclusions, and the Full metric at every point of the ball.

## G.5 Three exact radius certificates

Table 2 gives three matched bases. Each row uses its own interval $0 < t \leq T$ from (6.3), and all have $R = 1 / 1 0 0 0$ in $\| H / \sqrt { \lambda _ { 0 } } \| _ { F }$ . For every dense or sparse change in that closed ball, every nonzero PSD query and every $0 < t \leq T$ 2

$$
| h _ { R } - h _ { t } - t ( D - Q t ) c | \leq C t \epsilon c , \qquad h _ { R } - h _ { t } \geq D t c / 4 > 0 .
$$

The penalty at each endpoint is assumed exactly calibrated. The proof does not certify an approximate calibration solver.

Table 2: Exact suficient neighborhoods. The penalty rI is the matched base penalty, and $\lambda _ { 0 } I$ is the fixed physical Full target. The entries in the last column are strict rational upper bounds for $\Phi ( R )$ , not rounded decimal estimates.
<table><tr><td> $( n , d , s )$ </td><td> $r$ </td><td> $\lambda _ { 0 }$ </td><td> $T$ </td><td>R</td><td> $\Phi ( R )$  upper bound</td></tr><tr><td>(2,2,2)</td><td> $2 / 3$ </td><td> $5 / 3$ </td><td>1089/3890</td><td>1/1000</td><td>291/100000</td></tr><tr><td>(2, 2, 3)</td><td> $5 / 4$ </td><td> $5 5 / 3 1$ </td><td>28665/123952</td><td>1/1000</td><td>111/100000</td></tr><tr><td>(2, 3, 3)</td><td> $3 / 2$ </td><td>363/101</td><td>98972519/449051012</td><td>1/1000</td><td>209/100000</td></tr></table>

We describe the exact calculation so that the table is reproducible from finite sums and inequalities. Use $a = 1 / 1 0 0$ in Section G.2. Compute the physical mixture coeficients by the count sums in Appendix D, and the marked moments by its cardinality distribution above. The exact target identity $\mathbb { E } [ K _ { 1 } / ( r + K _ { 1 } ) ] = n / ( n + \lambda _ { 0 } )$ checks matching at the base. Sum over the $\binom { m } { s }$ original-row subsets and all md coordinate directions using (G.7) and (G.9). Store each energy as a pair of rational coeficients in $\mathbb { Q } ( { \sqrt { 1 + n / \lambda _ { 0 } } } )$ and check its full matrix form (G.10). The exact coeficients, energy pairs, scalar polynomial bounds, all transport constants, and rational gate slacks are supplied in gain certificates.json, with evaluator gain certificates.py and separate exact check gain independent checks.py in the reproducibility package. They are inputs and outputs of the formulas above, not substituted proof premises.

For completeness, the enclosure rules use only integer and rational arithmetic. For $x \geq 0 .$ , let $k =$ $\lfloor { \sqrt { x } } 1 0 ^ { 1 8 } \rfloor$ , found by integer squares. Then $k / 1 0 ^ { 1 8 } \leq \sqrt { x } \leq ( k + 1 ) / 1 0 ^ { 1 8 }$ , with equality used when exact. Respect the sign of the second coeficient when enclosing a quadratic field element. For $0 \leq x < 2 0$ use

$$
e ^ { x } \leq \sum _ { j = 0 } ^ { 1 8 } \frac { x ^ { j } } { j ! } + \frac { x ^ { 1 9 } / 1 9 ! } { 1 - x / 2 0 } .
$$

Table 3: Sharp-risk reduction from mixing on 140 structured target-size points. The thresholds summarize this fixed grid.
<table><tr><td>Weight</td><td>Median</td><td> $\geq 1 \%$ </td><td> $\geq 5 \%$ </td></tr><tr><td>Theorem  $T$ </td><td>1.24%</td><td>80</td><td>16</td></tr><tr><td>Common  $T$ </td><td>0.98%</td><td>70</td><td>12</td></tr><tr><td>Optimal mixture</td><td>4.50%</td><td>121</td><td>62</td></tr></table>

Successive tail ratios are at most $x / 2 0$ , which proves this bound. Round positive upper quantities upward and floors downward on the same rational grid. Take

$$
\chi \geq e ^ { 2 \sqrt { d } a } , \quad E _ { a } \geq e ^ { a } , \quad \delta \leq \operatorname* { m i n } \{ \zeta / 2 , \mu _ { * } a ( 1 + \zeta / 2 ) ^ { 2 } / ( 2 \chi ) \}
$$

with these directed bounds, and check (G.5) exactly. The lower bound for $f _ { \mathrm { f l o o r } } = 1 + ( z - R ) ^ { 2 }$ uses a lower bound for $z > R$ The upper bound for $f _ { 0 }$ uses an upper bound for z only in its numerator and the exact denominator $1 + \zeta$ . Keep $J _ { * }$ separately; subtracting rounded derivative norms would not bound their diference. Apply the same directed operations to (G.12)–(G.14) and (G.1). All calibration, rank, penalty, inclusion, leakage, and gain gates are strictly satisfied for the three rows. In particular the rational bounds in Table 2 are below each row’s exact $D / 4$

This evaluator enumerates $6 + 4 + 2 0$ labelled subsets and 8, 8, 18 coordinate directions, respectively. For general parameters this is a combinatorial calculation. With $N = { \binom { m } { s } }$ , storing dense selectors and base atoms uses $O ( N ( m ^ { 2 } + d m + d ^ { 2 } ) )$ scalar storage, in addition to energy accumulation. Rational bit costs and evaluation of endpoint inclusions remain separate. The certificates cover precisely these matched bases and weights $0 < t \leq T$ . They do not certify the earlier feature grid or establish practical prediction gains, responsewise dominance, or an optimal radius.

## H Finite evaluations and reproducibility

## H.1 Finite evaluation of the mixture

We evaluate the constructive mixture without changing the sampling law or label budget. The outcome is sharp query risk normalized by Full ridge loss. All calculations use finite sums or count formulas with 80-digit arithmetic. We recompute four prespecified cases at 120-digit precision. These are floating evaluations, not interval certificates. The protocol below specifies the inputs and controls; the supplement provides code and per-case results.

Structured designs. We vary $n \in \{ 2 , 3 , 4 , 8 \} , \ d \in \{ 2 , 4 , 8 \}$ and $\lambda _ { 0 } / n \in \{ . 0 1 , . 1 , 1 , 1 0 \}$ . Minimum, middle and leave-one-out sizes give 140 distinct target-size points. The queries are identity, one coordinate, flat rank one and tilted rank one. We also evaluate the query-uniform criterion max $\{ \alpha ( t ) , \beta ( t ) \}$ Table 3 compares the theorem’s weight T, a common weight, and the query-uniform optimum within the mixture family. For each fixed $( n , d , \lambda _ { 0 } )$ , the common weight is the smallest theorem weight $T _ { s }$ among that triple’s listed sample sizes. All three reported weights improve on ridge for the four tested queries at every grid point. The explicit weight is conservative: its median gain is 1.24%, compared with 4.50% for the optimum.

Strong controls. Improvement over ridge is not dominance over other procedures. Across the 560 physical query cases, T wins 402 comparisons with same-law HT, 432 with uniform-subset HT, and 297 with query-aware quota sampling. The latter two change the sampling law. For coordinate queries, quota sampling has zero risk at 92 of the 140 points. Thus an all-query improvement within a fixed law does not settle query-specific sampling design.

Perturbed designs. For a four-row, two-feature base, we perturb features in two directions at signed magnitudes .001, .01 and .1. Three target penalties and two sizes give 72 nonzero design points, each with four queries. We recalibrate the whole procedure and retain the base mixture weights. The base T improves all 288 query cases; its smallest gain is 0.018%. The base optimal weight loses four cases, by up to 0.350%. The gain theorem in Section 7 gives a separate local guarantee. It does not certify this grid: its suficient neighborhoods and improving weight intervals have their own assumptions. The exact certificates in Appendix G use three diferent matched bases and targets. These evaluations do not test the general-design LOO or ETF geometry.

## H.2 Protocol and complete comparisons

The structured calculation uses 48 fixed-target families. For each $( n , d )$ , the size set is $\{ d , d + \lfloor ( n d -$ $d ) / 2 ] , n d - 1 \}$ with duplicates removed. There are 140 target-size points. The physical queries are $I ,$ $e _ { 1 } e _ { 1 } ^ { \top } , \mathbf { 1 1 } ^ { \top }$ and $v v ^ { \top }$ , where $v _ { j } = j / d { \mathrm { . } }$ . Each has maximum diagonal entry one. The query-uniform criterion is separate; it is not a fifth physical query.

The local weight is T from Section 6. The common weight is the minimum of these weights over the three-size grid of one target family. The optimized weight minimizes max $\{ \alpha ( t ) , \beta ( t ) \}$ over [0, 1]. It need not minimize risk for each known query. The same-law HT control uses the original determinant sample. Uniform-subset HT samples uniformly among all size-s subsets. Quota sampling independently selects $k _ { j }$ rows uniformly within group $j ,$ with $1 \leq k _ { j } \leq n$ and $\textstyle \sum _ { j } k _ { j } = s$ . Its allocation minimizes the sharp query risk over these integer choices. Its risk is

$$
\operatorname* { m a x } _ { j } W _ { j j } \frac { a ^ { 2 } ( n - k _ { j } ) } { n k _ { j } ( n - 1 ) } , \qquad a = \frac { n } { n + \lambda _ { 0 } } .
$$

The code evaluates this optimization by a finite recurrence. A separate threshold-feasibility calculation verifies every retained optimum.

Table 4: Mixture wins/ties/losses against strong controls. Each row has 140 paired target-size cases. Quota sampling uses the query and changes the law.
<table><tr><td>Weight</td><td>Query</td><td>Same-law HT</td><td>Uniform HT</td><td>Quota</td></tr><tr><td> $T$ </td><td>Identity</td><td> $8 2 / 0 / 5 8$ </td><td>93/0/47</td><td>116/0/24</td></tr><tr><td rowspan="7">Optimal</td><td>Coordinate</td><td>82/0/58</td><td>93/0/47</td><td>30/0/110</td></tr><tr><td>Flat</td><td>140/0/0</td><td>140/0/0</td><td>116/0/24</td></tr><tr><td>Tilted</td><td>98/0/42</td><td>106/0/34</td><td>35/0/105</td></tr><tr><td>Identity</td><td>85/0/55</td><td>94/0/46</td><td>130/0/10</td></tr><tr><td>Coordinate</td><td>85/0/55</td><td>94/0/46</td><td>44/0/96</td></tr><tr><td>Flat</td><td>140/0/0</td><td>140/0/0</td><td>130/0/10</td></tr><tr><td>Tilted</td><td>103/0/37</td><td>111/0/29</td><td>51/0/89</td></tr></table>

For perturbations, $X _ { 0 } = ( e _ { 1 } , e _ { 1 } , e _ { 2 } , e _ { 2 } ) ^ { \top }$ and $\lambda _ { 0 } \in \{ . 2 , 2 , 2 0 \} , s \in \{ 2 , 3 \}$ . Define

$$
H _ { 1 } = \frac { 1 } { \sqrt { 6 } } \left( \begin{array} { c c } { { 0 } } & { { 1 } } \\ { { 1 } } & { { - 1 } } \\ { { - 1 } } & { { 0 } } \\ { { 1 } } & { { 1 } } \end{array} \right) , \qquad H _ { 2 } = \frac { 1 } { \sqrt { 5 } } \left( \begin{array} { c c } { { 1 } } & { { 0 } } \\ { { 0 } } & { { - 1 } } \\ { { 1 } } & { { 1 } } \\ { { - 1 } } & { { 0 } } \end{array} \right) .
$$

We evaluate $X _ { 0 } + \varepsilon H _ { i }$ for $\varepsilon \in \{ \pm . 0 0 1 , \pm . 0 1 , \pm . 1 \}$ . Both directions break exact replication; $H _ { 1 }$ preserves Gram isotropy while changing its scale, whereas $H _ { 2 }$ also changes anisotropy. Here |ε| is the raw Frobenius perturbation size; the target-whitened size in Theorem 7.2 is $| \varepsilon | / \sqrt { \lambda _ { 0 } }$ . At each point, the code reconstructs Full, the calibrated positive definite penalty, every subset probability and fit, the unequal row-inclusion probabilities, and the native normalization. The same-law HT operator is $( G + A _ { 0 } ) ^ { - 1 } X ^ { \top } \mathrm { d i a g } ( \mathbf { 1 } _ { i \in S } / \pi _ { i } )$ The raw mean-squared-error operator includes any finite mean defect; covariance and squared bias are retained separately. Each risk is a largest generalized eigenvalue. Physical queries and target penalties stay fixed.

The 72 nonzero points give 288 query cases per method. Six unperturbed bases add 24 cases, kept separate from perturbation counts. The four optimized-weight losses all have $\lambda _ { 0 } ~ = ~ 2 0 , ~ s ~ = ~ 2$ , flat query and $| \varepsilon | = . 1$ . Their reductions are −0.248281% and −0.179110% for $H _ { 1 }$ , and −0.349723% and −0.046204% for $H _ { 2 }$ . Against same-law HT, the base $T$ wins 108/288 cases and the base optimum wins 124/288. The structured optimum lies at a contrast–mean crossing; its finite perturbation losses do not establish a causal explanation or rule out improvement on a smaller neighborhood.

Arithmetic and reproducibility. The calculations use Python 3.13.11 and mpmath 1.3.0 on a CPU, without random seeds, model training or external datasets. Primary precision is 80 decimal digits. The scalar calibration residual gate is $1 0 ^ { - 6 0 }$ ; bisection has a 400-step cap. Four prespecified cases are replayed at 120 digits with agreement tolerance $1 0 ^ { - 5 5 }$ : structured $( n , d , \lambda _ { 0 } / n , s )$ equal to $( 2 , 2 , . 1 , 2 )$ and (8, 8, 10, 63), and perturbed $( \lambda _ { 0 } / 2 , s , H , \varepsilon )$ equal to $( 1 , 2 , H _ { 1 } , . 1 )$ and $( 1 0 , 3 , H _ { 2 } , - . 1 )$ . The structured replays include their target families for the common weight. All four replays pass. Wins and ties use $1 0 ^ { - 3 0 } \operatorname* { m a x } \lbrace 1 , | r _ { 1 } | , | r _ { 2 } | \rbrace$ . There are no statistical error bars. The full run retains 6,148 method rows with no unresolved cases.

The supplementary package separates exact polynomial checks from these floating calculations. It includes synthetic inputs in code, all per-case risks, a complete rerun driver and a separate riskreconstruction checker. Long coeficient arrays and method-row records are supplied as machine-readable files. No figure renderer is needed to reproduce the numerical claims.

## I Source relations

Law and mean. The identity de $( X _ { S } ^ { \top } X _ { S } + R ) = \operatorname* { d e t } ( R ) \operatorname* { d e t } ( I + X _ { S } R ^ { - 1 } X _ { S } ^ { \top } )$ ) identifies the law as a k-DPP on $I + X R ^ { - 1 } X ^ { \top }$ (Kulesza and Taskar, 2012, Section 5). Conditioning a uniform-base regularized DPP on its size gives the same law (Derezi´nski et al., 2020b, Definition 2). The selected-inverse expectation of Schreurs et al. (2020, Lemma 1 and Corollary 1) gives the fixed-penalty ridge mean by the Woodbury identity.

Inverse calibration. Our calibration theorem combines this mean identity with the interior meanmap theorem for minimal exponential families (Wainwright and Jordan, 2008, Proposition 3.2 and Theorem 3.3). The spectral expansion identifies a finite family with a truncated-cube mean set. Its interior condition gives the efective-dimension threshold. Uniqueness lifts from the mean vector to the shared matrix penalty. The penalty bounds follow from binomial-coeficient ratios. Thus calibration is a modelspecific consequence of these established results. Arbitrary SPD target scope follows by a change of coeficient coordinates. Loonis and Mary (2019, Theorem 4.1) construct fixed-size projection DPPs with prescribed original-row inclusion probabilities. That inverse-design result concerns row marginals. Here calibration matches the selected ridge response operator. Anzai and Hino (2026, Theorem 1) use this mean-map principle to match item inclusion probabilities in an unconditioned DPP with fixed diversity. Here the matched moments are latent spectral activations under an exact row budget, and they determine the ridge mean. Our uniqueness concerns the penalty used jointly in sampling and fitting. It is not identification of a generic k-DPP kernel from its law; such kernels have scale and other invariances (Hino and Yano, 2026).

Exact debiasing. Derezi´nski et al. (2020a, Theorems 1 and 6) obtain exact ridge debiasing with scaled regularization under surrogate sketching. Their row count is random. We instead use exactly s distinct original rows and the same matrix penalty in sampling and fitting, with an arbitrary SPD target. The efective-dimension threshold and debiasing goal have precedents.

Covariance geometry. Derezi´nski and Warmuth (2018) establish selected-OLS identities under ordinary volume sampling and study a regularized reverse-deletion procedure. At positive penalty, that sequential procedure generally has a diferent endpoint law from our direct determinant law. The proof of their Theorem 18 also uses repeated coordinate rows and group counts for noise-averaged ridge prediction risk. Our result instead concerns fixed-response covariance under the direct calibrated subset law and its complete equality space. Rhee (2026) studies fixed-design OLS covariance attainment. The normalized worst-response OLS design criterion of Derezi´nski et al. (2019, Definitions 6 and 8) is another close precedent. It divides by residual energy and optimizes over designs and estimators. Here one law and estimator are fixed for each comparison; the response is deterministic, covariance is centered over the subset draw, W is a fixed physical coeficient query, and positive Full ridge loss normalizes every nonzero response. We give complete equality spaces on balanced replicas at the stated budgets, an exact attainability test for a general-design leave-one-out envelope, and a flat-query criterion on existing real ETFs at two deletions. For isotropic queries, we also give sharp risk at three deletions. The leave-one-out envelope can be strict, and the ETF criterion does not classify the risk of every nonflat query.

Frame erasures and ETF structure. Goyal et al. (2001) study one-erasure average and worst reconstruction MSE under zero-mean, equal-variance, uncorrelated quantization noise. Their estimator uses linear dual-frame reconstruction. Bodmann and Paulsen (2005) study frame reconstruction-error operator norms, including the pair-uniform geometry of equiangular tight frames. Probability-weighted erasure criteria for dual frames also appear in Arati et al. (2024). These are diferent losses from the determinant-weighted, centered covariance divided by Full ridge loss here. In particular, an ETF calculation within a fixed frame class is not a claim that ETFs optimize our risk over designs: Bodmann and Paulsen (2005, Proposition 3.8) show that two-uniform frames can maximize their average two-erasure error measure. The independence of the projectors $x _ { i } x _ { i } ^ { \top }$ for noncollinear equiangular lines is established in Fickus et al. (2021, Section 5.1). The dimension of the flat-energy physical-query space follows from this rank fact and tightness; our two-deletion result instead concerns the covariance deficit and all its equality responses. Flat energy $x _ { i } ^ { \top } W x _ { i } = { \mathrm { t r } } ( W ) / d$ is not the source’s notion of a flat ETF.

Three ETF deletions. The triangle classes and their erasure spectra are classical (Bodmann and Paulsen, 2005). Their unequal determinant weights follow directly from those spectra. Our theorem uses the weights in the mean and centered second moment. It proves a strict normalized sector gap and identifies every maximizing response. It does not assume uniform triples or claim frame-design optimality.

Estimator mixing and restricted responses. Horvitz–Thompson weighting (Horvitz and Thompson, 1952) and dependent vector estimator averaging (Lavancier and Rochet, 2016) are established methods. Loonis and Mary (2019, Theorem 5.1) optimize fixed linear survey weights for specified auxiliary variables, using first- and second-order inclusion probabilities. Our selected ridge inverse depends on the realized subset. We use the contrast derivative and strict sector gap to choose one feature-only interval of scalar mixture weights. It improves every nonzero PSD query’s sharp risk at the balanced base and preserves the complete maximizing space. The mean-share value follows by optimizing the contrast and mean blocks separately. The further comparison gives the exact all-query improvement phase on the protected interval and an attaining adverse query. It concerns deterministic responses and this mixture family.

Recalibrated transfer. Classical eigenspace bounds (Yu et al., 2015) apply after we control the calibration, subset law, fit and Full normalization. The full equality space then yields the raw-response restriction theorem. The individual top-Ritz bound is also classical (Li and Li, 2005, Theorem 2). We combine it with the moving Full metric and explicit remainders. Retaining all base contrasts removes query-gap dependence at the cost of a larger space. The gain theorem compares two moving sharp risks under the same recalibrated procedure, including its actual inclusion probabilities. Its coupled depen dence on the mixture weight comes from bounding the shared-sample diference. This is stronger than qualitative gain persistence at one fixed positive weight, which follows from continuity.