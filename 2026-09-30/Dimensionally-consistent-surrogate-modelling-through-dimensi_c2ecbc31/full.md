# Dimensionally consistent surrogate modelling through dimensional analysis and harmonic expansions

Ernest Tarrus<sup>1</sup> and Hector Gisbert<sup>1</sup>

1\* Universidad Europea de Valencia, Escuela de Ciencias, Ingenier´ıa y Dise˜no, C/ de Guillem de Castro, 175, Extramurs, Valencia, 46008, Valencia, Spain

Corresponding Author: Hector Gisbert.

Contributing authors: ernesttarrus@gmail.com; hector.gisbert@universidadeuropea.es;

## Abstract

Dimensional homogeneity is a fundamental constraint on physically meaningful models, requiring invariance under changes of units. We present a data-driven method for constructing surrogate models that satisfy this constraint at the level of the hypothesis class. Starting from a dimension matrix of measured variables, the method derives Buckingham Π-groups, constructs admissible dimensional prefactors, and approximates the remaining dimensionless dependence using truncated harmonic expansions on normalized invariant domains. Once the prefactor and dictionary are fixed, the coeficients are obtained from a regularized linear regression problem. We test the approach on the simple pendulum, Planck’s black-body law, the double-pendulum Lyapunov field, and an experimental COBE/FIRAS black-body spectrum dataset. The results show that dimensional constraints improve conditioning, robustness to noise, and sample eficiency relative to unconstrained baselines, while the choice of dictionary becomes important in non-periodic or multi-invariant settings. The learned expressions are explicit and inexpensive to evaluate, which makes them useful as surrogate models for structured physical problems.

Keywords: Dimensional analysis; Buckingham Π theorem; Unit-equivariant learning; Dimensionally consistent surrogate modelling; Regularized regression; Harmonic feature expansions; Invariant representations

## 1 Introduction

Dimensional analysis plays a central role in the physical sciences. At a minimum, it provides a consistency criterion: physically meaningful equations must be dimensionally homogeneous, so that their form is preserved under changes of units [1, 2]. More importantly, it reveals the internal structure of physical models. By exploiting scaling relations among variables, dimensional analysis reduces the efective number of degrees of freedom and identifies the dimensionless combinations that govern the remaining behavior, as formalized by the Buckingham Π theorem [2–5]. This result is also known as the Vaschy–Buckingham theorem, since Vaschy stated an equivalent formulation in 1892 [6]; we follow the standard convention and refer to it as the Buckingham Π theorem throughout. The resulting invariants provide natural coordinates for the problem and clarify which combinations of variables may influence the observable of interest.

The growing interest in data-driven modelling of physical systems has renewed the relevance of such structural constraints. Symbolic regression seeks explicit analytic expressions directly from data [7, 8], while sparse-library approaches identify compact models from prescribed candidate terms [9, 10]. More broadly, physics-informed learning introduces prior physical knowledge into the learning process in order to improve generalization, robustness, and interpretability [11]. In all these settings, one basic requirement is often overlooked or only weakly enforced: the learned model should remain meaningful under arbitrary changes of units. A predictor that depends on whether length is expressed in meters or centimeters is not physically admissible. This suggests that dimensional consistency should not be treated merely as an a posteriori check, but rather imposed directly at the level of the hypothesis class.

Recent work has begun to incorporate dimensional structure more explicitly into machine-learning pipelines. Villar et al. formulate learning with physical quantities as a problem of exact units equivariance, showing how dimensional analysis can be used to construct invariant representations before inference [12]. Bakarji et al. develop dimensionally consistent learning with Buckingham Π, including constrained fitting and data-driven identification of dimensionless groups and scaling parameters [13]. Xie et al. propose a framework for discovering dominant dimensionless numbers and governing relations from limited measurements [14]. Related ideas have also been explored in data-driven dimensional analysis for complex flow systems, where the goal is to identify informative nondimensional groups directly from simulation or experimental data [15]. Taken together, these works show that dimensional structure is not only a consistency requirement, but also a useful inductive bias for modelling physical relations from data.

In this work, we construct surrogate models from data under explicit dimensional constraints. Starting from the dimensional structure of the measured variables, we form independent Buckingham invariants and identify the admissible dimensional scal ings compatible with the target quantity. The predictor is written as a dimensional prefactor multiplied by a dimensionless modulation on invariant space, and the modulation is approximated with a truncated harmonic dictionary. With this choice, fitting the coeficients reduces to regularized linear regression, for example Ridge, Lasso, or Elastic Net [16–19].

Our approach is related to this previous literature, and several of its individual ingredients are classical. Buckingham dimensional analysis, monomial dimensional prefactors, harmonic/Fourier approximations, and Ridge regularization are all wellestablished tools. The contribution of the present work is to use these components as a fixed hypothesis class for surrogate modelling: an admissible dimensional prefactor is selected from the afine space $A { \boldsymbol { \beta } } = \mathbf { b }$ , while the remaining dimensionless response is approximated by a finite dictionary on Buckingham invariant space.

This difers from previous dimensionally aware learning methods in scope and emphasis. Existing approaches often focus on discovering dimensionless groups from data, embedding dimensional constraints into flexible black-box regressors, or enforcing units equivariance in broad model classes [12–14]. Here, the admissible dimensional structure and the approximation space are specified before fitting. The model is therefore less general than a symbolic discovery engine, but its assumptions and fitted coeficients are directly inspectable. In this sense, our framework is also distinct from symbolic-regression systems such as AI Feynman, which search over broader symbolic expression spaces rather than working within a structured approximation class determined a priori by dimensional admissibility [8].

The main contributions of this work are therefore the following. First, we formulate dimensionally constrained surrogate modelling as a unit-equivariant regression problem determined by the dimension matrix of the variables. Second, we construct an explicit prefactor-times-invariants hypothesis class in which the admissible dimensional scaling is separated from the residual dimensionless response. Third, we study finite dictionaries on invariant space, including one-dimensional, phase-combination, and separable tensor-product forms, and discuss when the associated periodic continuation is appropriate. Fourth, we evaluate the resulting models against unconstrained and dimensionally aware baselines, including alternative non-periodic basis choices, on synthetic and experimental benchmark data.

The benchmarks are chosen to separate four efects: recovery of a dimensional skeleton, approximation of a nontrivial dimensionless modulation, validation on measured data, and dictionary design in a multi-invariant setting. Across these cases, the experiments track how dimensional constraints afect conditioning, robustness, and sample eficiency, and how dictionary geometry matters once the invariant space is multidimensional.

The remainder of the paper is organized as follows. Section 2 introduces the dimensional formulation of the learning problem. Section 3 develops the unit-consistent hypothesis class, including admissible prefactors, invariant coordinates, and harmonic dictionaries. Section 4 summarizes the full learning pipeline. Sections 5, 6, and 7 present the benchmarks and discuss the numerical results. Finally, Section 8 concludes with the main implications and possible extensions of the framework.

## 2 Dimensional structure of the learning problem

We begin by formalizing the dimensional structure that constrains the admissible form of the target response. The aim of this section is to introduce the dimension matrix, the

notion of unit equivariance, and the reduction to dimensionless invariant coordinates provided by the Buckingham Π theorem.

## 2.1 Variables, dimensions, and the dimension matrix

Consider a physical system described by measured variables ${ \bf x } = ( x _ { 1 } , \dots , x _ { N } )$ and a target quantity $y ,$ related through an unknown response function

$$
y = f ( x _ { 1 } , \ldots , x _ { N } ) = f ( \mathbf { x } ) .\tag{1}
$$

In practice, this relation may also depend on additional quantities that are unobserved, externally controlled, or efectively fixed over the dataset under consideration. We do not model such variables explicitly and assume that, on the domain of interest, their efect is either negligible or approximately constant.

Let $B = \{ D _ { 1 } , \ldots , D _ { R } \}$ be a basis of fundamental dimensions, such as mass, length, and time. Each input variable $x _ { j }$ is assigned a dimensional signature

$$
\left[ x _ { j } \right] = \prod _ { r = 1 } ^ { R } ( D _ { r } ) ^ { a _ { r j } } , \qquad a _ { r j } \in \mathbb { Q } , \qquad j = 1 , \ldots , N ,\tag{2}
$$

and the target variable satisfies

$$
[ y ] = \prod _ { r = 1 } ^ { R } ( D _ { r } ) ^ { b _ { r } } , \qquad b _ { r } \in \mathbb { Q } .\tag{3}
$$

The exponents of the input variables are collected in the dimension matrix

$$
A = \left( a _ { r j } \right) \in \mathbb { Q } ^ { R \times N } ,\tag{4}
$$

whose j-th column encodes the dimensional signature of $x _ { j }$ . This representation allows the dimensional structure of the problem to be handled algebraically. In particular, linear relations among the columns of A correspond to dimensionless combinations of the inputs, while relations involving the target signature $\mathbf { b } = ( b _ { 1 } , \ldots , b _ { R } ) ^ { \top }$ determine the admissible dimensional scalings of the response.

## 2.2 Unit equivariance

A physically meaningful model must preserve its form under changes of units. This requirement is naturally expressed as an equivariance condition under the scaling group $G \ : = \ : ( \mathbb { R } _ { > 0 } ) ^ { R }$ . For $\lambda = ( \lambda _ { 1 } , \ldots , \lambda _ { R } ) \in G$ , the induced action on each input variable $x _ { j }$ is

$$
x _ { j } \mapsto \left( \prod _ { r = 1 } ^ { R } ( \lambda _ { r } ) ^ { a _ { r j } } \right) x _ { j } , \qquad j = 1 , \ldots , N ,\tag{5}
$$

while the target transforms as

$$
y \mapsto \left( \prod _ { r = 1 } ^ { R } ( \lambda _ { r } ) ^ { b _ { r } } \right) y .\tag{6}
$$

Accordingly, the response function (1) is physically admissible only if it satisfies

$$
f \left( \left( \prod _ { r = 1 } ^ { R } ( \lambda _ { r } ) ^ { a _ { r 1 } } \right) x _ { 1 } , \cdots , \left( \prod _ { r = 1 } ^ { R } ( \lambda _ { r } ) ^ { a _ { r N } } \right) x _ { N } \right) = \left( \prod _ { r = 1 } ^ { R } ( \lambda _ { r } ) ^ { b _ { r } } \right) f ( { \bf x } ) \mathrm { ~ . ~ }\tag{7}
$$

Equation (7) is the mathematical expression of dimensional homogeneity. It states that the learned response must transform exactly as the target quantity under any admissible rescaling of the fundamental units. A formal equivariance theorem for this class of monomial-times-invariant predictors is established in the broader machinelearning setting by [12]; the diferential form of the constraint specific to our setup is derived in Appendix A.

## 2.3 Dimensionless invariants and admissible prefactors

Dimensional analysis implies that the unit-equivariance condition strongly restricts the admissible form of the response. A monomial

$$
\pi ( \mathbf { x } ) = \prod _ { j = 1 } ^ { N } ( x _ { j } ) ^ { \gamma _ { j } }\tag{8}
$$

is dimensionless if and only if its exponent vector $\gamma \in \mathbb { Q } ^ { N }$ satisfies

$$
\mathbf { \nabla } A \boldsymbol { \gamma } = \mathbf { 0 } \mathrm { ~ . ~ }\tag{9}
$$

Thus, dimensionless combinations of the inputs are in one-to-one correspondence with the kernel of the dimension matrix. If $\{ \bar { \gamma } ^ { ( m ) } \} _ { m = } ^ { N _ { \Pi } }$ <sub>1</sub> is a basis of kerpAq, then the associated Buckingham invariants are

$$
\pi _ { m } ( { \bf x } ) = \prod _ { j = 1 } ^ { N } ( x _ { j } ) ^ { \gamma _ { j } ^ { ( m ) } } , \qquad m = 1 , \ldots , N _ { \Pi } ,\tag{10}
$$

with $N _ { \Pi } = \dim \ker ( A ) = N - \operatorname { r a n k } ( A )$ . In parallel, admissible dimensional prefactors are obtained by solving

$$
A \beta = \mathbf { b } \ ,\tag{11}
$$

where $\mathbf { b } = ( b _ { 1 } , \ldots , b _ { R } ) ^ { \top }$ is the dimensional signature of the target. We denote the solution set by $S _ { b } = \{ \beta \in \mathbb { Q } ^ { N } \mid A \beta = \mathbf { b } \}$ . Any solution $\beta \in S _ { b }$ defines a monomial

$$
P _ { \beta } ( \mathbf { x } ) = \prod _ { j = 1 } ^ { N } x _ { j } ^ { \beta _ { j } }\tag{12}
$$

with the same physical dimensions as $y .$ If $\beta _ { 0 }$ is one particular solution, then every other solution has the form

$$
\begin{array} { r } { \beta = \beta _ { 0 } + { \bf v } , \qquad { \bf v } \in \ker ( A ) \ , } \end{array}\tag{13}
$$

where $\boldsymbol { S } _ { b }$ is an afine subspace parallel to ker $( A )$ , with dimension $N _ { \Pi } =$ dim ker $\cdot ( A )$ The vector v should therefore be understood as an arbitrary element of this kernel, not as a single fixed displacement, so diferent admissible prefactors difer only by multiplication of a dimensionless monomial.<sup>1</sup>

Whenever there exists $\beta \in \mathbb { Q } ^ { N }$ such that $A \beta = \mathbf { b } ,$ the Buckingham Π theorem implies that any dimensionally homogeneous response can be written as a dimensional prefactor times a dimensionless function of the invariants. In the finite model class used here, this motivates predictors of the form

$$
y = \sum _ { \beta \in \mathcal { T } } P _ { \beta } ( \mathbf { x } ) \Phi _ { \beta } \bigl ( \pi _ { 1 } ( \mathbf { x } ) , \ldots , \pi _ { N _ { \Pi } } ( \mathbf { x } ) \bigl ) ,\tag{14}
$$

where $\mathcal { T } \subset S _ { b }$ denotes a finite set of admissible candidate exponent vectors selected from the afine solution space of $A { \boldsymbol { \beta } } = \mathbf { b }$ . Thus, the sum is not taken over the full continuous solution space, but only over the finite prefactor dictionary used in the regression model. Here, $P _ { \beta }$ carries the physical dimensions of the target $y$ and $\Phi _ { \beta }$ is dimensionless. Diferent admissible choices of $\beta$ correspond to equivalent factorizations of the same response, since any change in the prefactor can be absorbed into the dimensionless modulation $\Phi _ { \beta }$

In the experiments reported here, each benchmark employs a single monomial prefactor, so the sum contains one term:

$$
y = P _ { \beta _ { 0 } } ( \mathbf { x } ) \Phi _ { \beta _ { 0 } } \bigl ( \pi _ { 1 } ( \mathbf { x } ) , \ldots , \pi _ { N _ { \Pi } } ( \mathbf { x } ) \bigr ) .
$$

In the next section, we turn this structural representation into a concrete hypothesis class by specifying how the dimensionless modulation is approximated and how the resulting model is fitted from observations.

## 3 Dimensionally consistent hypothesis class

The dimensional analysis developed in Section 2 fixes the admissible dimensional structure of the predictor, but not the form of the remaining dimensionless dependence. We now specify this dependence by writing the target response as a dimensional prefactor $P _ { \beta }$ multiplied by a modulation $\Phi _ { \beta }$ on invariant space, and then approximating that modulation with harmonic dictionaries.

## 3.1 Prefactor-times-invariants representation

Let $\lbrace \pi _ { m } \rbrace _ { m = } ^ { N _ { \Pi } }$ be a basis of independent Buckingham invariants and let $P _ { \beta }$ be any admissible prefactor satisfying $\left[ P _ { \beta } \right] = \left[ y \right]$ . We seek predictors of the form

$$
g ( \mathbf { x } ) = \sum _ { \beta \in \mathcal { T } } P _ { \beta } ( \mathbf { x } ) \Phi _ { \beta } \big ( \pi _ { 1 } ( \mathbf { x } ) , \ldots , \pi _ { N _ { \Pi } } ( \mathbf { x } ) \big ) ,\tag{15}
$$

where $\Phi _ { \beta }$ are dimensionless modulation functions to be learned from data. This representation directly enforces unit equivariance. Indeed, the prefactor carries the full dimensional signature of the target, whereas the invariants remain unchanged under unit transformations. As a result, any predictor of the form (15) satisfies the scaling law in Eq. (7) by construction.

## 3.2 Harmonic approximation on invariant space

The remaining task is to approximate the dimensionless modulation functions $\Phi _ { \beta }$ on the invariant coordinates. Since the Buckingham invariants need not be periodic or bounded on their raw scale, we first map them to a normalized compact domain. For each invariant $\pi _ { m } .$ , we define

$$
u _ { m } ( \pi _ { m } ) = 2 \pi \left( \frac { \pi _ { m } - \pi _ { m } ^ { \mathrm { m i n } } } { \pi _ { m } ^ { \mathrm { m a x } } - \pi _ { m } ^ { \mathrm { m i n } } } \right) , \qquad m = 1 , \hdots , N _ { \Pi } ,\tag{16}
$$

If $\pi _ { m } ^ { \mathrm { m a x } } = \pi _ { m } ^ { \mathrm { m i n } }$ for a given invariant, the corresponding coordinate has zero empirical variation and the afine normalization is not defined. In that degenerate case, we set $u _ { m } = 0$ and remove all non-constant harmonic features associated with that invariant. This reflects the fact that no dependence on that coordinate can be inferred from the available data. For notational convenience, the dependence of $\pi _ { m }$ on x is left implicit, so we write $u _ { m } ( \pi _ { m } )$ in place of $u _ { m } ( \pi _ { m } ( \mathbf { x } ) )$ q. Furthermore, we write $\Phi ( u _ { 1 } , \dots , u _ { N _ { \Pi } } ) \equiv$ $\Phi _ { \beta } ( u _ { 1 } ( \pi _ { 1 } ) , \dots , u _ { N _ { \Pi } } ( \pi _ { N _ { \Pi } } ) )$ for the modulation expressed in normalized coordinates; this is the same dimensionless factor as in (14), with the invariants $\pi _ { m }$ replaced by their afinely rescaled counterparts $u _ { m }$

Here, $\pi _ { m } ^ { \mathrm { m i n } }$ and $\pi _ { m } ^ { \mathrm { m a x } }$ are estimated from the dataset or prescribed from prior physical constraints. This maps the observed invariant domain into $[ 0 , 2 \pi ] ^ { N _ { \Pi } }$ . We then approximate $\Phi _ { \beta }$ by a truncated harmonic expansion in the rescaled variables $u = \left( u _ { 1 } , \ldots , u _ { N _ { \Pi } } \right)$

The harmonic dictionary is a modelling choice, not a consequence of dimensional consistency. It is natural when the invariants have angular or phase-like character, and the truncation order has a simple spectral interpretation. It also leaves a standard regularized linear regression problem for the coeficients.

Other basis choices are possible within the same dimensional formulation. For bounded non-periodic invariant domains, polynomial bases such as Chebyshev or Legendre features may be more appropriate. For localized or non-smooth modulations, splines or radial basis functions could also be used. These alternatives would keep the same prefactor and Buckingham invariants, changing only the dictionary used for the residual dimensionless modulation.

The afine embedding in Eq. (16) introduces a periodic continuation of the learned dimensionless modulation. For example, in the double-pendulum benchmark the relevant invariant coordinates are the angles $\theta _ { 1 } , \theta _ { 2 } \in [ - \pi , \pi ]$ , so identifying the endpoints is consistent with the geometry of the problem.

By contrast, in the black-body benchmark the invariant $\pi _ { 1 } ~ = ~ k _ { B } T / ( h \nu )$ is not periodic. In that case, the harmonic dictionary should be understood as a finite spectral surrogate on the sampled interval, not as a physically periodic representation of the Planck modulation. If the endpoint values of the modulation difer substantially, the periodic extension can introduce boundary artefacts and slower convergence, analogous to Gibbs-type efects. For this reason, we explicitly compare the harmonic dictionary with a Chebyshev polynomial dictionary in the black-body benchmark. Chebyshev features provide a natural non-periodic basis on a bounded interval and therefore serve as a diagnostic of whether the harmonic continuation is limiting the approximation.

## 3.2.1 One-dimensional case

If $N _ { \Pi } = 1$ , we use

$$
\Phi ( u _ { 1 } ) \approx w _ { 0 } + \sum _ { n = 1 } ^ { N _ { f } } \Bigl [ w _ { n } ^ { c } \cos ( n u _ { 1 } ) + w _ { n } ^ { s } \sin ( n u _ { 1 } ) \Bigr ] ,\tag{17}
$$

where $N _ { f }$ is the maximum retained frequency. In the multi-invariant case, the truncation orders per coordinate are written $\{ N _ { f } ^ { ( m ) } \} _ { m = 1 } ^ { N _ { \Pi } } ;$ when a uniform order is used across all coordinates it is abbreviated to the scalar $N _ { f } \ { \mathrm { ( i . e . } } \ N _ { f } ^ { ( m ) } = N _ { f }$ for all m).

## 3.2.2 Multi-invariant phase-combination dictionary

For $N _ { \Pi } > 1$ , a direct multidimensional extension of the one-dimensional harmonic expansion is obtained by introducing a non-negative integer multi-index

$$
{ \mathbf { n } } = ( n _ { 1 } , . . . , n _ { N _ { \Pi } } ) \in ( \mathbb { Z } _ { \geqslant 0 } ) ^ { N _ { \Pi } } ,\tag{18}
$$

and defining the corresponding phase through the dot product between the multiindex vector and the invariant-coordinate vector ${ \mathbf { n } } \cdot u = n _ { 1 } u _ { 1 } + \cdot \cdot \cdot + n _ { N _ { \Pi } } u _ { N _ { \Pi } }$ . The dimensionless modulation is then approximated as

$$
\Phi ( u ) \approx w _ { 0 } + \sum _ { \mathbf { n } \in \mathcal { T } _ { \mathbf { n } } } \Big [ w _ { \mathbf { n } } ^ { c } \cos ( \mathbf { n } \cdot u ) + w _ { \mathbf { n } } ^ { s } \sin ( \mathbf { n } \cdot u ) \Big ] ,\tag{19}
$$

where ${ \mathcal { T } } _ { \mathbf { n } } \subset ( { \mathbb { Z } } _ { \geqslant 0 } ) ^ { N _ { \Pi } } \backslash \{ { \mathbf { 0 } } \}$ is a finite truncation set. The main feature of this basis is that all invariant coordinates are coupled through a single phase. This makes the representation compact and natural, but it may also limit flexibility when the target function exhibits strongly directional or anisotropic structure. In such settings, the approximation may benefit from a separable tensor-product basis, which resolves the behavior along diferent invariant directions more explicitly.

## 3.2.3 Separable tensor-product dictionary

In multi-invariant problems, especially when the response surface is anisotropic, it is often preferable to use a separable tensor-product basis. For each coordinate $u _ { m } .$ define the one-dimensional basis set $\mathcal { F } _ { m } = \{ \psi _ { k } ^ { ( m ) } \} _ { k = 0 } ^ { 2 N _ { f } ^ { ( m ) } }$ ordered as:

$$
\psi _ { 0 } ^ { ( m ) } ( u _ { m } ) = 1 , \qquad \psi _ { 2 n - 1 } ^ { ( m ) } ( u _ { m } ) = \cos ( n u _ { m } ) , \qquad \psi _ { 2 n } ^ { ( m ) } ( u _ { m } ) = \sin ( n u _ { m } ) ,\tag{20}
$$

for $n = 1 , \dots , N _ { t } ^ { ( m ) }$ . A separable feature is then uniquely identified by an integer multi-index $\mathbf { k } = \left( \omega _ { \mathrm { 1 } } , \ldots , k _ { N _ { \Pi } } \right)$ :

$$
\Psi _ { \mathbf { k } } ( u ) = \prod _ { m = 1 } ^ { N _ { \Pi } } \psi _ { k _ { m } } ^ { ( m ) } ( u _ { m } ) , \qquad \mathbf { k } \in \mathcal { T } _ { \mathbf { k } } ,\tag{21}
$$

where $\mathcal { T } _ { \boldsymbol { \mathbf { k } } } \subseteq \times _ { m = 1 } ^ { N _ { \Pi } } \{ 0 , 1 , \dots , 2 N _ { f } ^ { ( m ) } \}$ is the retained set of basis-function index vectors. The modulation is approximated as

$$
\Phi ( u ) \approx \sum _ { \mathbf { k } \in \mathcal { T } _ { \mathbf { k } } } w _ { \mathbf { k } } \Psi _ { \mathbf { k } } ( u ) ,\tag{22}
$$

where the constant term corresponds to $\mathbf { k } = \mathbf { 0 }$ , giving $\boldsymbol { w _ { 0 } } \Psi _ { \mathbf { 0 } } ( u ) = \boldsymbol { w _ { \mathbf { 0 } } }$ . This tensorproduct representation allows diferent truncation orders in diferent coordinates and separates marginal efects from interaction terms. As shown later in the doublependulum benchmark, Section 6.3, this distinction matters in multi-invariant settings with directional structure.

## 3.3 Learning by regularized regression

We define the feature vector $\varphi ( \mathbf { x } ) \in \mathbb { R } ^ { p }$ as the concatenation of the basis functions for each of the K candidate monomial terms:

$$
\varphi ( \mathbf x ) = \left[ \varphi _ { 1 } ( \mathbf x ) ^ { \top } , \varphi _ { 2 } ( \mathbf x ) ^ { \top } , \dots , \varphi _ { K } ( \mathbf x ) ^ { \top } \right] ^ { \top } ,\tag{23}
$$

where each block $\varphi _ { \ell } ( \mathbf { x } )$ contains the features associated with the ℓ-th monomial. Here, $P _ { \ell } \equiv P _ { \beta _ { \ell } }$ corresponds to the ℓ-th admissible exponent vector $\beta \in { \mathcal { T } }$ (where $\mathcal { T } \subset S _ { b } )$ identified in the prefactor search.

The construction of the feature elements is governed by the specific geometry of the selected dictionary. In the case of a phase-combination dictionary defined by multiindices $\mathbf { n } \in \mathcal { Z } _ { \mathbf { n } }$ , the feature set comprises the constant dimensional term $P _ { \ell } ( \mathbf { x } )$ and the corresponding trigonometric pairs $P _ { \ell } ( { \bf x } ) \cos ( { \bf n } \cdot { \boldsymbol u } )$ and $P _ { \ell } ( \mathbf { x } ) \sin ( \mathbf { n } \cdot u )$ . Alternatively, when using a separable tensor-product dictionary, the features are indexed by k P $\scriptstyle { \mathcal { T } } _ { \mathbf { k } }$ and defined as $\varphi _ { \ell , { \bf k } } ( { \bf x } ) = P _ { \ell } ( { \bf x } ) \Psi _ { \bf k } ( u )$ , where $\Psi _ { \mathbf { k } }$ represents the product of the independent one-dimensional marginal basis functions associated with each invariant coordinate.

For a dataset of $N _ { \mathrm { d a t } }$ observations and a fixed candidate prefactor $\beta ,$ we construct the design matrix $\mathcal { X } _ { \beta } \in \mathbb { R } ^ { N _ { \mathrm { d a t } } \times p }$ such that $[ \mathcal { X } _ { \beta } ] _ { i , \cdot } = \varphi _ { \beta } ( \mathbf { x } ^ { ( i ) } ) ^ { \top }$ , and define the target vector $\mathbf { y } ^ { \top } = [ y ^ { ( 1 ) } , \ldots , y ^ { ( N _ { \mathrm { d a t } } ) } ] ^ { \top }$ . The predictor $g ( \mathbf { x } ) = \mathbf { w } ^ { \top } \boldsymbol { \varphi } ( \mathbf { x } )$ is linear in the global coeficient vector $\mathbf { w } \in \mathbb { R } ^ { p }$ . Scalar weights for individual dictionary modes are written as follows: $w _ { n } ^ { c } , w _ { n } ^ { s }$ for cosine and sine modes in the one-dimensional case; $w _ { \bf n } ^ { c } , w _ { \bf n } ^ { s }$ for the phase-combination basis indexed by the frequency vector n; and $w _ { \mathbf { k } }$ for the separable tensor-product basis, where each index k already identifies a specific trigonometric factor through the basis functions $\psi _ { k _ { m } } ^ { ( m ) }$ defined in (20). The coeficients are estimated by minimizing the Ridge [17] objective:

$$
\hat { \mathbf { w } } _ { \beta } = \mathop { \arg \operatorname* { m i n } } _ { \mathbf { w } \in \mathbb { R } ^ { p } } \left[ \| \mathbf { y } - \boldsymbol { \chi } _ { \beta } \mathbf { w } \| _ { 2 } ^ { 2 } + \mu \| \mathbf { w } \| _ { 2 } ^ { 2 } \right] .\tag{24}
$$

The $\ell _ { 2 }$ penalty damps high-frequency coeficients, and the objective remains quadratic in the weights. The implementation details and cross-validation protocol used in the numerical experiments are described in Section 5.1.

## 3.3.1 Feature count and truncation

The dimensionality p of the regression problem is a direct consequence of the truncation strategy employed in the invariant space. For a phase-combination dictionary with rectangular truncation and frequency limit $N _ { f } ,$ the number of non-zero multiindices is $\left| \mathcal { T } _ { \mathbf { n } } \right| = ( N _ { f } + 1 ) ^ { N _ { \Pi } } - 1$ , resulting in $p = \overset { \cdot } { K } ( 2 ( N _ { f } + 1 ) ^ { N _ { \Pi } } - 1 )$ parameters. This exponential scaling with $N _ { \Pi }$ is the primary bottleneck for complex systems. To overcome this, we adopt a total-degree truncation $\| \mathbf { k } \| _ { 1 } \leqslant N _ { \mathrm { t o t } }$ . The number of nonnegative integer multi-indices satisfying this bound is given by the binomial coeficient $\binom { N _ { \mathrm { t o t } } + N _ { \Pi } } { N _ { \Pi } }$ . Including the constant term and the sine-cosine pairs, the total parameter count grows as:

$$
p = K \left( 2 \binom { N _ { \mathrm { t o t } } + N _ { \mathrm { I I } } } { N _ { \mathrm { I I } } } - 1 \right) \approx \mathcal { O } \left( K \cdot \frac { N _ { \mathrm { t o t } } ^ { N _ { \mathrm { I I } } } } { N _ { \mathrm { I I } } ! } \right) .\tag{25}
$$

This specific combinatorics formula applies to the phase-combination basis under total-degree truncation. By contrast, for a separable tensor-product dictionary without symmetry reductions, the feature count scales as $\begin{array} { r } { p = K \prod _ { m = 1 } ^ { N _ { \Pi } } ( 2 N _ { f } ^ { ( m ) } + 1 ) } \end{array}$ . This follows because each invariant coordinate contributes one constant feature plus $N _ { f } ^ { ( m ) }$ cosine modes and $N _ { f } ^ { ( m ) }$ sine modes, giving $2 N _ { f } ^ { ( m ) } + 1$ one-dimensional basis functions per coordinate. The separable dictionary is obtained by taking all products of these one-dimensional factors, hence the product over m. In both cases, the model is linear in the fitted weights.

![](images/a43afda9f4de0af1298a3a1a93c9bfda9b43fd768a375e96c9ecfa2e840396cf.jpg)  
Fig. 1: Algorithmic pipeline. Box 2c illustrates the two harmonic dictionary formulations evaluated in the experiments: separable tensor-product (option A) and phase-combination (option B) dictionaries.

## 4 Algorithmic pipeline

We now summarize the computational pipeline that turns raw measurements into an explicit, dimensionally consistent predictor. The purpose of this section is not to repeat the mathematical construction of Section 3, but to make explicit the operational workflow and the computational scaling of the diferent stages of the method.

Concretely, the algorithm proceeds by (i) constructing independent Π groups, (ii) building dimensionally admissible prefactors with the dimensions of y, (iii) expanding the unknown dimensionless modulation in a truncated dictionary on invariant space, and (iv) fitting the resulting linear-in-parameters model by regularized regression.

Algorithm 1 Unit-equivariant learning via tensor-product harmonic dictionaries   
Require: Dataset $\mathcal D \ = \ \{ ( \mathbf x ^ { ( i ) } , y ^ { ( i ) } ) \} _ { i = 1 } ^ { N _ { \mathrm { d a t } } }$ ; Truncation orders $\{ N _ { f } ^ { ( m ) } \} _ { m = 1 } ^ { N _ { \Pi } }$ (for separable   
basis, index set $\begin{array} { r } { \mathcal { T } _ { \mathbf { k } } ; } \end{array}$ for phase-combination basis, index set $\begin{array} { r } { \mathcal { T } _ { \mathbf { n } } ) ; } \end{array}$ Regularization grid Λ.   
Ensure: Fitted analytic predictor $\hat { g } ( \mathbf x )$   
Phase 1: Dimensionless Manifold Mapping   
1: Compute basis $\{ \gamma ^ { ( m ) } \} _ { m = 1 } ^ { N _ { \Pi } }$ for kerpAq.   
2: Construct independent invariants: $\begin{array} { r } { \pi _ { m } ( \mathbf { x } ) = \prod _ { j = 1 } ^ { N } x _ { j } ^ { \gamma _ { j } ^ { ( m ) } } } \end{array}$ for $m = 1 , \ldots , N _ { \Pi } .$   
3: Compute afine embedding to map $\pi ( \mathbf { x } ^ { ( i ) } )$ to periodic coordinates $u ^ { ( i ) } \in [ 0 , 2 \pi ] ^ { N _ { \Pi } }$   
Phase 2: Tensor-Product Dictionary Construction   
4: Initialize dictionary $\scriptstyle { \mathcal { L } } _ { \mathbf { k } }$ based on symmetry reductions (e.g., retaining only even-parity   
tensor-product features).   
5: for each coordinate m $\dot { \in } \left\{ 1 , \dots , N _ { \Pi } \right\}$ do   
6: Generate 1D marginal basis: $\mathcal { F } _ { m } = \{ 1 \} \cup \{ \cos ( n u _ { m } ) , \sin ( n u _ { m } ) \} _ { n = 1 } ^ { N _ { f } ^ { ( m ) } }$   
7: end for   
8: Construct separable basis $\Psi _ { \mathbf { k } } ( u )$ via tensor product: $\bigotimes _ { m = 1 } ^ { N _ { \Pi } } \mathcal { F } _ { m } .$   
9: (Optional) Apply total-degree truncation (see section $3 . 3 . 1 \rangle$ : restrict to $\| \mathbf { k } \| _ { 1 } \leqslant N _ { \mathrm { t o t } }$   
Phase 3: Structural Optimization & Regression   
10: Identify afine subspace of admissible exponents: $S _ { b } = \{ \beta \in \mathbb { Q } ^ { N } \mid A \beta = \mathbf { b } \}$   
11: Generate search grid $\mathcal { T } \subset S _ { b }$ for candidate dimensional prefactors.   
12: for each candidate exponent vector $\beta \in { \mathcal { T } }$ do   
13: Construct dimensional skeleton $P _ { \beta } ( \mathbf { x } )$   
14: Build design matrix $\chi _ { \beta }$ with elements $[ \mathcal { X } _ { \beta } ] _ { i , { \bf k } } = P _ { \beta } ( { \bf x } ^ { ( i ) } ) \Psi _ { \bf k } ( u ^ { ( i ) } )$ for $\mathbf { k } \in \mathcal { T } _ { \mathbf { k } }$   
15: Solve regularized regression via cross-validation over $\Lambda { : }$   
$\hat { \mathbf { w } } _ { \beta } = \mathop { \operatorname { a r g m i n } } _ { \mathbf { w } } \| \mathbf { y } - \boldsymbol { \chi } _ { \beta } \mathbf { w } \| _ { 2 } ^ { 2 } + \mu \| \mathbf { w } \| _ { 2 } ^ { 2 }$   
w   
16: Record cross-validation metric $\mathcal { M } ( \beta ) \ ( \mathrm { e . g . , } \ R ^ { 2 }$ or negative MSE).   
17: end for   
18: Select optimal structural skeleton: $\beta ^ { * } = \operatorname { a r g m a x } _ { \beta \in \mathcal { T } } \mathcal { M } ( \beta )$   
Phase 4: Symbolic Recovery   
19: return $\begin{array} { r } { \hat { g } ( \mathbf { x } ) = P _ { \beta ^ { * } } ( \mathbf { x } ) \sum _ { \mathbf { k } \in \mathcal { T } _ { \mathbf { k } } } \hat { w } _ { \beta ^ { * } , \mathbf { k } } \Psi _ { \mathbf { k } } ( u ( \mathbf { x } ) ) } \end{array}$

Remark. If a phase-combination basis is used instead of the separable tensor-product dictionary, the dictionary-construction step is replaced by the direct evaluation of the features $w _ { \mathbf { n } } ^ { c } \cos ( \mathbf { n } \cdot u )$ and $w _ { \mathbf { n } } ^ { s } \sin ( \mathbf { n } \cdot u )$ for $\mathbf { n } \in \mathcal { L } _ { \mathbf { n } } .$ where $\begin{array} { r } { \mathbf { n } \cdot u = \sum _ { m } n _ { m } u _ { m } } \end{array}$

## 4.1 Computational complexity and scalability

The prefactor-selection step requires evaluating the Ridge regression with leave-oneout cross-validation (RidgeCV) fitting procedure for each point on a grid over the afine subspace $\boldsymbol { S } _ { b }$ . For a system with $N _ { \Pi }$ free exponents in the prefactor family, the grid has $\mathcal { O } ( M ^ { N _ { \Pi } } )$ evaluations, where M is the number of grid points per dimension. Each evaluation involves a RidgeCV fit with cost $\mathcal { O } ( N _ { \mathrm { d a t } } p ^ { 2 } + p ^ { 3 } )$ , where $p$ is the number of dictionary features. The total cost therefore scales as $\mathcal { O } \big ( \hat { M } ^ { \hat { N } _ { \Pi } } \big ( N _ { \mathrm { d a t } } \hat { p ^ { 2 } } + p ^ { 3 } \big ) \big )$ ˘. The cost $\mathcal { O } ( N _ { \mathrm { d a t } } p ^ { 2 } + p ^ { 3 } )$ arises from the standard regularized least-squares solution. Forming the covariance matrix $\chi ^ { \top } \mathcal { X }$ requires $\mathcal { O } ( N _ { \mathrm { d a t } } p ^ { 2 } )$ operations, and the subsequent inversion or decomposition of this $p \times p$ matrix scales as $\mathcal { O } ( p ^ { 3 } )$ . Since RidgeCV employs an eficient leave-one-out cross-validation scheme that reuses the same matrix decomposition for multiple values of the regularization parameter $\mu .$ , the overall complexity per grid point remains dominated by these two terms [20]. For the benchmarks considered in this work $( N _ { \Pi } \leqslant 2 )$ , this exhaustive scan is computationally feasible. In practical terms, the current implementation is intended for low-dimensional invariant spaces, roughly $N _ { \Pi } \leqslant 3$ , with moderate truncation orders and prefactor grids. For $N _ { \Pi } \geqslant 4$ the combination of the prefactor grid and high-dimensional dictionaries becomes the main computational bottleneck, since both the number of candidate prefactors and the number of fitted features can grow rapidly.

Several strategies can mitigate this limitation. On the prefactor side, the bruteforce scan over $\boldsymbol { S _ { b } }$ could be replaced by gradient-based optimization over the exponent parameters [21], coarse-to-fine hierarchical grid searches [22], or Bayesian optimization over the prefactor space [23]. On the dictionary side, sparse total-degree truncations, adaptive frequency selection, or regularization paths could be used to avoid fitting the full dictionary. These extensions are outside the scope of the present benchmarks, but they provide natural routes for scaling the method to higher-dimensional invariant spaces.

## 5 Experimental setup

We evaluate the method on three synthetic benchmarks and one experimental black-body spectrum dataset. The experiments are designed to examine dimensional constraints, dictionary choice, and prefactor selection in diferent regimes: a case in which dimensional analysis nearly determines the response, a case with a strongly non-polynomial modulation, and a case with a structured multi-invariant response surface.

## 5.1 General protocol

For each benchmark, we construct a dataset of input-output pairs $\begin{array} { r l } { \mathcal { D } } & { { } = } \end{array}$ $\{ ( \mathbf { x } ^ { ( i ) } , y ^ { ( i ) } ) \} _ { i = 1 } ^ { N _ { \mathrm { d a t } } }$ , and divide it into training and test subsets in order to evaluate out-ofsample performance. When the dimensional prefactor is not unique, prefactor selection is treated as an outer model-selection loop: each candidate prefactor defines a different feature map, the corresponding harmonic coeficients are fitted by regularized regression, and the final model is chosen according to validation performance.

Predictive performance is quantified primarily through the coeficient of determination $R ^ { 2 }$ and the mean squared error (MSE),

$$
R ^ { 2 } = 1 - \frac { \sum _ { i = 1 } ^ { N _ { \mathrm { t e s t } } } ( y ^ { ( i ) } - \hat { y } ^ { ( i ) } ) ^ { 2 } } { \sum _ { i = 1 } ^ { N _ { \mathrm { t e s t } } } ( y ^ { ( i ) } - \bar { y } ) ^ { 2 } } , \qquad \mathrm { M S E } = \frac { 1 } { N _ { \mathrm { t e s t } } } \sum _ { i = 1 } ^ { N _ { \mathrm { t e s t } } } ( y ^ { ( i ) } - \hat { y } ^ { ( i ) } ) ^ { 2 } ,\tag{26}
$$

where $\boldsymbol y ^ { ( i ) }$ denotes the true target, $\hat { y } ^ { ( i ) }$ the prediction, y¯ the empirical mean of the test targets, and $N _ { \mathrm { t e s t } }$ the size of the evaluation set.

Unless otherwise stated, the fitting stage is carried out using regularized linear models implemented in scikit-learn [22]. Throughout all experiments, RidgeCV is used by default. The regularization parameter µ is selected by cross-validation within the training set only. Across all experiments, we study the framework under variations in four factors:

• the dimensional complexity of the benchmark;

• the number of retained invariant coordinates;

• the richness of the harmonic dictionary;

• the sampling density and the level of additive noise.

This makes it possible to assess separately the efects of dimensional structure, approximation bias, and statistical variability.

## 5.2 Benchmarks

The benchmarks considered in this work are: (i) a small-angle simple pendulum, used to isolate recovery of the dimensional scaling; (ii) synthetic black-body radiation at fixed frequency, used to test a one-dimensional but transcendental residual modulation; (iii) the COBE/FIRAS black-body spectrum, used as a real-data validation and as a comparison between harmonic and Chebyshev features; and (iv) the doublependulum Lyapunov field, used to test dictionary geometry, tensor-product structure and symmetry reduction in a multi-invariant setting.

Taken together, the experiments are designed to answer four questions: whether the framework recovers the correct dimensional scaling, how prefactor selection interacts with approximation error, how dictionary geometry afects multi-invariant learning, and how regularization and symmetry reduction improve robustness.

## 6 Results

In this section, we report the results for each benchmark.

## 6.1 Simple pendulum

We begin with the simple pendulum as a dimensional-skeleton benchmark. With the release angle fixed, the learning task reduces to recovering the scaling $\tau \propto \sqrt { L / g }$ and testing its robustness under diferent sampling and noise regimes.

## 6.1.1 Physical setting and dimensional structure

We consider a simple pendulum consisting of a point mass m suspended by a rigid massless rod of length L in a uniform gravitational field $g .$ . For release angle $\theta _ { 0 }$ , the period is described by

$$
\tau = 2 \pi \sqrt { \frac { L } { g } } \frac { 2 } { \pi } { \mathcal K } \bigg ( \sin \frac { \theta _ { 0 } } { 2 } \bigg ) ,\tag{27}
$$

where K is the complete elliptic integral of the first kind

$$
K ( x ) = \int _ { 0 } ^ { \pi / 2 } \frac { d \theta } { \sqrt { 1 - x \sin ^ { 2 } \theta } } .
$$

In the small-angle regime $\theta _ { 0 } ~ \ll ~ 1$ , this reduces to the familiar approximation $\begin{array} { r } { \tau \approx 2 \pi \sqrt { \frac { L } { g } } } \end{array}$ . We model the period τ as a function of the variables $( L , g , m , \theta _ { 0 } )$ . The associated dimension matrix (if we let $B = \{ { \sf M } , { \sf L } , { \sf T } \} )$ is

$$
A = \left( { \begin{array} { l l l } { 0 } & { 0 } & { 1 } & { 0 } \\ { 1 } & { 1 } & { 0 } & { 0 } \\ { 0 } & { - 2 } & { 0 } & { 0 } \end{array} } \right) .\tag{28}
$$

Since the angle is already dimensionless and the mass does not afect the period law, the invariant structure is particularly simple. Solving $A \gamma = \mathbf { 0 }$ yields a one-dimensional invariant space, and a natural choice is $\pi _ { 1 } = \theta _ { 0 }$ . Likewise, solving $A { \boldsymbol { \beta } } = \mathbf { b }$ for the target signature of $\tau$ gives the dimensional prefactor $\begin{array} { r } { P = \sqrt { \frac { L } { g } } } \end{array}$ , up to multiplication by powers of the invariant, which can be absorbed into the dimensionless modulation. The resulting model therefore takes the form

$$
\tau = \Phi ( \theta _ { 0 } ) \sqrt { \frac { L } { g } } .\tag{29}
$$

## 6.1.2 Model and data generation

For this benchmark, the predictor is written as

$$
\hat { \tau } = \sqrt { \frac { L } { g } } \left[ w _ { 0 } + \sum _ { n = 1 } ^ { N _ { f } } \left( w _ { n } ^ { c } \cos ( n u _ { 1 } ) + w _ { n } ^ { s } \sin ( n u _ { 1 } ) \right) \right] ,\tag{30}
$$

where $u _ { 1 }$ is the afine rescaling of $\theta _ { 0 }$ to the interval r0, 2πs. For these experiments, the period of the pendulum is assumed to be $\tau ~ = ~ 2 \pi { \sqrt { \frac { L } { g } } }$ . In this scenario, the dimensionless modulation is constant, $\Phi ( \theta _ { 0 } ) = 2 \pi$ . The experiment therefore mainly tests the efect of using the dimensional prefactor; the harmonic part of the model is exercised in the black-body and double-pendulum benchmarks, where the modulation is non-trivial.

Synthetic data are generated from the exact pendulum law under a two-level sampling procedure. First, $n _ { L }$ distinct pendulum lengths are drawn uniformly from the interval r0.1, 2.0s m. For each sampled length, M repeated measurements are produced by adding Gaussian noise $\mathcal { N } ( 0 , \sigma _ { \mathrm { a b s } } ^ { 2 } )$ to the ground-truth period, where $\sigma _ { \mathrm { a b s } }$ is an absolute noise level in seconds. The repeated measurements are then averaged before regression. This setup allows us to vary independently the measurement noise, the number of distinct length values, and the number of repeated observations.

To examine the stability of the framework, we consider six representative scenarios combining low and high noise with sparse and dense sampling. As a baseline comparison, we also report fits obtained with a standard Minuit [24] parametric estimator.

## 6.1.3 Results

The numerical results are summarized in Table 1. In low-noise regimes, both approaches recover the expected scaling with high accuracy. In particular, when the sampling density is suficiently large, the learned coeficient is essentially indistinguishable from the theoretical small-angle value $2 \pi \approx 6 . 2 8 3 2$

For comparison with the theoretical small-angle scaling, we report the coeficient (c) and intercept (d) obtained from an auxiliary post hoc fit of the form $\tau = c \sqrt { L / g } + d$

Table 1: Simple-pendulum results comparing regularized regression and Minuit. The theoretical small-angle coeficient is $2 \pi \approx 6 .$ .2832.
<table><tr><td>Scenario</td><td> $\sigma _ { \mathrm { a b s } }$ </td><td> $( n _ { L } , M )$ </td><td>Method</td><td>Coeff. (c)</td><td>Intercept (d)</td><td>Error (%)</td></tr><tr><td>Low Noise</td><td>0.02</td><td>(5,5)</td><td>RidgeCV Minuit</td><td>6.2539 6.2451 ± 0.0287</td><td>0.0061 0.0079 ± 0.0011</td><td>0.47 0.61</td></tr><tr><td>High Density</td><td>0.02</td><td>(50,50)</td><td>RidgeCV Minuit</td><td>6.2836 6.2836 ± 0.0142</td><td>0.0007 0.0006 ± 0.0053</td><td>0.01 0.01</td></tr><tr><td>High Noise</td><td>0.6</td><td>(5,5)</td><td>RidgeCV Minuit</td><td>5.4052 5.1418 ± 0.8622</td><td>0.1822 0.2382 ± 0.3164</td><td>13.97 18.16</td></tr><tr><td>High Density Noise</td><td>0.6</td><td>(50,50)</td><td>RidgeCV Minuit</td><td>6.29028 6.2916 ± 0.4266</td><td>0.01799 0.0182 ± 0.1588</td><td>0.10 0.20</td></tr><tr><td>Mixed (Low nL)</td><td>0.6</td><td>(5,50)</td><td>RidgeCV Minuit</td><td>6.5910 6.5941 ± 1.1526</td><td>-0.0997 -0.0981 ± 0.4282</td><td>4.90 4.95</td></tr><tr><td>Mixed (Low M)</td><td>0.6</td><td>(50,5)</td><td>RidgeCV Minuit</td><td>6.55672 6.5411 ± 0.3375</td><td>-0.08879 −0.1078 ± 0.1217</td><td>4.52 4.11</td></tr></table>

The main qualitative trend is that the dimensionally constrained regression model remains accurate whenever either the sampling density or the amount of averaging is suficient. Under sparse and noisy conditions, both methods deteriorate, but the regularized estimator is generally more stable. This is particularly visible in the highnoise, low-sampling setting, where Ridge regression yields a smaller error than the baseline parametric fit. The efect is consistent with the variance-reduction role of the $\ell _ { 2 }$ penalty.

These trends are illustrated in Figure 2. Increasing either the number of sampled lengths or the number of repeated measurements is enough to recover the correct scaling even when the noise level is large.

![](images/e7aa62791b3f03a04e3667873482077f10298dc24f18a26e9224dcc15f01af52.jpg)  
(a) Low noise; $\begin{array} { r l } { \sigma _ { \mathrm { a b s } } } & { { } = } \end{array}$ 0.02, $n _ { L } = M = 5$

![](images/f09b555a60f1cb3f710f3c2932b4532e83d3e6f17840f5fded9910c32c2bd4e7.jpg)  
(b) High density; $\sigma _ { \mathrm { a b s } } =$ 0.02, $n _ { L } = M = 5 0$

![](images/2589049a2f234913a687df9ee357534bb0314fca52f82668dcf605393b504d9a.jpg)  
(c) High noise; $\sigma _ { \mathrm { a b s } } =$ 0.6, $n _ { L } = M = 5$

![](images/9087bdb43b68d0ae16fac8fe39acc99753822425b6b36564df455c407a3ec8e0.jpg)  
(d) High density noise;  
σ<sub>abs</sub> “ 0.6, $n _ { L } = M =$ 50.

![](images/61b2b21c7356206f09569897685c43e691a8169b34b9e2cd7fafcbfad0997321.jpg)  
(e) Mixed (low n<sub>L</sub>);  
σ “ 0.6, n<sub>L</sub> “ 5, M “ 50.

![](images/30e61f8fb82489949b97601d3d5b297279364310a1920e5bfb285c6c682e52ae.jpg)  
(f) Mixed (low M);  
$\sigma _ { \mathrm { a b s } } ~ = ~ 0 . 6 ,$ $n _ { L } \ = \ 5 0 ,$ M “ 5.  
Fig. 2: Regression results for the simple pendulum period under diferent noise and sampling regimes. The red dashed line indicates the theoretical small-angle coeficient 2π.

## 6.1.4 Interpretation

This benchmark behaves as expected in the simplest setting. The dimensional prefactor recovers the scaling $\sqrt { L / g }$ , and the residual dimensionless dependence is constant in the small-angle approximation. The simple pendulum therefore serves mainly as a

baseline check: when dimensional analysis almost determines the response, the model reproduces the expected scaling with little data.

## 6.1.5 Comparison with an unconstrained baseline

To quantify the benefit of imposing dimensional consistency, we compare the proposed model against an unconstrained power-law regressor with the same functional form. The dimensional model fixes the exponent from physics:

$$
\hat { \tau } = w _ { 0 } ^ { ( \mathrm { D i m } ) } + w _ { 1 } ^ { ( \mathrm { D i m } ) } \sqrt { \frac { L } { g } } ,\tag{31}
$$

with 2 free parameters $( w _ { 0 } , w _ { 1 } )$ . The unconstrained analog would be

$$
\hat { \tau } = w _ { 0 } ^ { \mathrm { ( U n c ) } } + w _ { 1 } ^ { \mathrm { ( U n c ) } } L ^ { \alpha } g ^ { \beta } ,
$$

Since $g$ is constant, we absorb it into $w _ { 1 }$ . So the final unconstrained polynomial is of the form:

$$
\begin{array} { r } { \hat { \tau } = w _ { 0 } ^ { \mathrm { ( U n c ) } } + w _ { 1 } ^ { \mathrm { ( U n c ) } } L ^ { \alpha } , } \end{array}\tag{32}
$$

with 3 parameters $( w _ { 0 } , w _ { 1 } , \alpha )$ . Both models are fitted entirely via Minuit [24]. For the dimensional model, Minuit directly minimizes the mean squared error (MSE) over $( w _ { 0 } , w _ { 1 } )$ on the training set. For the unconstrained model, we use a nested 80/20 train– validation split of the training data: Minuit optimizes all three parameters $( \alpha , w _ { 0 } , w _ { 1 } )$ simultaneously against the validation MSE. After convergence, the final out-of-sample $R ^ { 2 }$ is evaluated on the held-out test set using the discovered parameters. Both models use identical starting values $( w _ { 0 } , w _ { 1 } ) = ( 0 , 0 )$ , and for the unconstrained model α is initialized at 0 and constrained to $[ - 2 , 2 ]$ , with no prior knowledge of the physical exponent. All experiments use 30 independent 70/30 train/test trials.

In this comparison, fixing $\alpha = 1 / 2$ removes the nonlinear optimization over the exponent and leaves a two-parameter linear fit. This allows the dimensional model to work from as few as $n _ { L } = 5$ data points. The unconstrained model requires a validation split to guide Minuit over α and fails for $n _ { L } < 1 2$ , where the split leaves too few points to form reliable estimates.

As shown in Table 2, the unconstrained model requires substantial data before the exponent converges: at $n _ { L } = 1 2 , \hat { \alpha } = 0 . 7 9 6 \pm 0 . 1 4 0$ is far from the physical value and the out-of-sample $R ^ { 2 }$ is only 0.31, reflecting the dificulty of jointly resolving three parameters from a validation set of just two points. With increasing $n _ { L }$ , αˆ drifts steadily toward $1 / 2$ , reaching $0 . 5 5 6 \pm 0 . 0 9 9$ at $n _ { L } = 3 5$ and $0 . 5 2 6 \pm 0 . 0 3 2$ at $n _ { L } = 1 0 0$ while the $R ^ { 2 }$ rises to 0.999. The dimensional model, by contrast, reaches $R ^ { 2 } \approx 0 . 9 9 9$ from $n _ { L } = 5$ onward. These results confirm that the dimensional constraint is suficient but not strictly necessary for this benchmark, a blind 3-parameter search eventually converges to the correct exponent, but requires an order of magnitude more data and passes through a regime of poor generalization where the joint optimization over $( \alpha , w _ { 0 } , w _ { 1 } )$ is ill-conditioned. The dimensional model’s advantage is therefore one of sample eficiency, robustness and correctness.

Table 2: Dimensional versus unconstrained powerlaw regression for the simple pendulum. Mean outof-sample $R ^ { 2 }$ and recovered exponent $\hat { \alpha }$ over 30 independent trials. Both models fitted with Minuit.
<table><tr><td>nL</td><td> $R ^ { 2 } \ \mathrm { ( D i m . ) }$ </td><td> $R ^ { 2 } \ \mathrm { ( U n c . ) }$ </td><td> $\hat { \alpha } \pm \sigma _ { \alpha }$ </td></tr><tr><td>5</td><td>0.9982</td><td></td><td></td></tr><tr><td>8</td><td>0.9977</td><td></td><td></td></tr><tr><td>12</td><td>0.9985</td><td>0.3061</td><td> $0 . 7 9 6 \pm 0 . 1 4 0$ </td></tr><tr><td>20</td><td>0.9991</td><td>0.9667</td><td> $0 . 6 7 8 \pm 0 . 1 3 2$ </td></tr><tr><td>35</td><td>0.9995</td><td>0.9930</td><td> $0 . 5 5 6 \pm 0 . 0 9 9$ </td></tr><tr><td>60</td><td>0.9994</td><td>0.9983</td><td> $0 . 5 2 8 \pm 0 . 0 3 4$ </td></tr><tr><td>100</td><td>0.9995</td><td>0.9988</td><td> $0 . 5 2 6 \pm 0 . 0 3 2$ </td></tr></table>

The trends are illustrated in Figure 3. The dimensional model reaches $R ^ { 2 } \approx 0 . 9 9 9$ with as few as $n _ { L } = 5$ lengths, whereas the unconstrained model requires $n _ { L } \geqslant 1 2$ to produce any result and $n _ { L } \geqslant 6 0$ to approach the dimensional accuracy level. At small $n _ { L }$ , the simultaneous optimization over $( \alpha , w _ { 0 } , w _ { 1 } )$ is severely ill-conditioned, producing αˆ far from $1 / 2$ and low out-of-sample $R ^ { 2 } ;$ the convergence toward the physical exponent is gradual, requiring an order of magnitude more data than the dimensional model.

![](images/0f98b1db535add4921cd9c7e419335b3be6d1dc3520c4d4e97b8279cf6734a24.jpg)

![](images/b8c10676f13758f979bca14adfd8e37a50e7134d0f61e8e7614d6363a2224106.jpg)  
Fig. 3: Dimensional vs. unconstrained power-law baseline on the simple pendulum. Left: out-of-sample $R ^ { 2 }$ as a function of $n _ { L }$ . Right: recovered exponent αˆ vs. $n _ { L }$ , skipping $n _ { L } < 2 0 ;$ ; the dashed line marks the dimensional value $\alpha = 1 / 2$ . Error bars denote ˘1 standard deviation over 30 independent trials. Both models fitted with Minuit.

## 6.2 Black-body radiation

As a second benchmark, we consider Planck’s black-body law at fixed frequency. Here the dimensional reduction still yields a one-dimensional invariant, but the remaining dependence is transcendental; the benchmark therefore tests how prefactor selection interacts with approximation error and noise.

## 6.2.1 Physical setting and dimensional structure

In frequency form, Planck’s law for the spectral radiance of a black body is

$$
B _ { \nu } ( T ) = { \frac { 2 h \nu ^ { 3 } } { c ^ { 2 } } } { \frac { 1 } { e ^ { h \nu / ( k _ { B } T ) } - 1 } } ,\tag{33}
$$

where $\nu$ is the frequency, $T$ is the absolute temperature, h is Planck’s constant, c is the speed of light, and $k _ { B }$ is Boltzmann’s constant. In the experiments below, ν is treated as a fixed physical parameter, and the dependence on temperature is learned from data. The variables are ordered as $\left( \nu , T , h , c , k _ { B } \right)$ , and the dimension matrix with respect to $B = \{ \mathsf { M } , \mathsf { L } , \mathsf { T } , \mathsf { K } \}$ is:

$$
A = \left( { \begin{array} { r r r r } { 0 } & { 0 } & { 1 } & { 0 } & { 1 } \\ { 0 } & { 0 } & { 2 } & { 1 } & { 2 } \\ { - 1 } & { 0 } & { - 1 } & { - 1 } & { - 2 } \\ { 0 } & { 1 } & { 0 } & { 0 } & { - 1 } \end{array} } \right) .\tag{34}
$$

Solving the homogeneous system $A \gamma = \mathbf { 0 }$ yields a one-dimensional invariant space. A natural generator gives $\begin{array} { r } { \pi _ { 1 } = \frac { k _ { B } T } { h \nu } } \end{array}$ . Thus, after dimensional reduction, the problem becomes one-dimensional. The admissible prefactors are obtained from the afine system $A \beta =$ b. Setting $\beta = \beta _ { 0 } + \alpha \gamma ^ { ( 1 ) }$ for the particular solution $\beta _ { 0 } = ( 3 , 0 , 1 , - 2 , 0 ) ^ { \top }$ and null-space vector $\gamma ^ { ( 1 ) } = ( - 1 , 1 , - 1 , 0 , 1 ) ^ { \intercal }$ , the admissible prefactor family becomes

$$
P _ { \alpha } = \nu ^ { 3 - \alpha } T ^ { \alpha } h ^ { 1 - \alpha } c ^ { - 2 } k _ { B } ^ { \alpha } .\tag{35}
$$

The physically correct choice corresponds to $\alpha = 0$ , namely $P _ { 0 } = \nu ^ { 3 } h c ^ { - 2 }$ , so the exact law can be written as

$$
B _ { \nu } = P _ { 0 } \Phi ( \pi _ { 1 } ) ,\tag{36}
$$

with $\begin{array} { c } { { \Phi ( \pi _ { 1 } ) ~ = ~ \frac { 2 } { e ^ { 1 / \pi _ { 1 } } - 1 } } } \end{array}$ . This benchmark is therefore especially informative for the prefactor scan. $\operatorname { A n y }$ deviation from $\alpha = 0$ can then be interpreted as evidence of finite-sample efects, dictionary bias, or noise sensitivity.

## 6.2.2 Model and experimental regimes

For this benchmark, the predictor is written as

$$
\hat { B } _ { \nu } = P _ { \alpha } \left[ w _ { 0 } + \sum _ { n = 1 } ^ { N _ { f } } \left( w _ { n } ^ { c } \cos ( n u _ { 1 } ) + w _ { n } ^ { s } \sin ( n u _ { 1 } ) \right) \right] ,\tag{37}
$$

where $u _ { 1 }$ is the rescaling of the invariant $\pi _ { 1 }$ to the interval $[ 0 , 2 \pi ]$

To examine separately the roles of prefactor selection and dictionary expressivity, we consider four regimes obtained by combining two prefactor-search ranges with two truncation orders:

• Regime A: broad prefactor grid, α <sub>P</sub> $[ - 5 , 5 ]$ , with low truncation $N _ { f } = 5 ;$

• Regime B: broad prefactor grid, α $\mathrm { \prime } \in \left[ - 5 , 5 \right]$ , with high truncation $N _ { f } = 6 0 ;$

• Regime C: narrow prefactor grid, $\alpha \in \lceil - 0 . 5 , 0 . 5 \rceil$ , with low truncation $N _ { f } = 5 ;$

• Regime D: narrow prefactor grid, $\alpha \in [ - 0 . 5 , 0 . 5 ]$ , with high truncation $N _ { f } = 6 0 .$

These four configurations make it possible to distinguish errors due to insuficient harmonic expressivity from errors due to imperfect identification of the dimensional scaling.

Synthetic datasets are generated by evaluating Planck’s law over the temperature range ${ \cal T } \in \mathrm { ( 0 , 1 0 0 0 0 ) } \mathrm { \ K } .$ at fixed frequency. The number of samples is varied from 50 to 5000 in ten steps. In the noisy experiments, additive Gaussian noise $\mathcal { N } ( 0 , ( \sigma _ { \mathrm { r e l } } \mathrm { s t d } ( B _ { \nu } ) ) ^ { 2 } )$ is introduced at three relative levels, $\sigma _ { \mathrm { r e l } } \in \{ 0 . 0 2 , 0 . 2 , 0 . 6 \}$ , where std $\left( \boldsymbol { B } _ { \nu } \right)$ is the standard deviation of the noiseless signal over the sampled temperature range.

## 6.2.3 Noiseless results

The noiseless results for the four regimes are reported in Tables 3–6. The main conclusion is that the correct dimensional prefactor is recovered.

## Regime A: broad grid, low truncation.

With a small dictionary, the model can fit the data well, but the prefactor scan may be biased for very small sample sizes. As shown in Table 3, the method initially favors $\alpha = 1$ , and only for suficiently large datasets does it converge to the correct value α “ 0. This indicates that a rigid harmonic approximation can push part of the residual curvature into the dimensional prefactor.

Table 3: Black-body Regime A: broad prefactor grid $( \alpha \in [ - 5 , 5 ] )$ , low truncation $( N _ { f } = 5 )$
<table><tr><td> $N _ { \mathrm { d a t } }$ </td><td>Best </td><td>Best  $R ^ { 2 }$ </td><td>Best MSE</td></tr><tr><td>50</td><td>1.0</td><td>0.999938</td><td>9.707071e-20</td></tr><tr><td>600</td><td>1.0</td><td>0.999970</td><td>4.605817e-20</td></tr><tr><td>1150</td><td>0.0</td><td>0.999998</td><td>3.589867e-21</td></tr><tr><td>1700</td><td>0.0</td><td>0.999998</td><td>3.440877e-21</td></tr><tr><td>2250</td><td>0.0</td><td>0.999998</td><td>3.339133e-21</td></tr><tr><td>2800</td><td>0.0</td><td>0.999998</td><td>3.255943e-21</td></tr><tr><td>3350</td><td>0.0</td><td>0.999998</td><td>3.182589e-21</td></tr><tr><td>3900</td><td>0.0</td><td>0.999998</td><td>3.115358e-21</td></tr><tr><td>4450</td><td>0.0</td><td>0.999998</td><td>3.051995e-21</td></tr><tr><td>5000</td><td>0.0</td><td>0.999998</td><td>2.992223e-21</td></tr></table>

## Regime B: broad grid, high truncation.

Increasing the truncation order largely removes that bias. As shown in Table 4, the correct value $\alpha = 0$ is recovered already at moderate sample size, and the resulting fit is essentially exact throughout the range. This confirms that suficient expressivity in the dimensionless modulation helps disentangle functional approximation error from dimensional scaling.

Table 4: Black-body Regime B: broad prefactor grid $( \alpha \in [ - 5 , 5 ] )$ , high truncation $( N _ { f } = 6 0 )$
<table><tr><td> $N _ { \mathrm { d a t } }$ </td><td>Best </td><td>Best  $R ^ { 2 }$ </td><td>Best MSE</td></tr><tr><td>50</td><td>0.5</td><td>1.0</td><td>6.046916e-28</td></tr><tr><td>600</td><td>0.0</td><td>1.0</td><td>2.807961e-28</td></tr><tr><td>1150</td><td>0.0</td><td>1.0</td><td>2.365705e-28</td></tr><tr><td>1700</td><td>0.0</td><td>1.0</td><td>2.157930e-28</td></tr><tr><td>2250</td><td>0.0</td><td>1.0</td><td>2.016073e-28</td></tr><tr><td>2800</td><td>0.0</td><td>1.0</td><td>1.906302e-28</td></tr><tr><td>3350</td><td>0.0</td><td>1.0</td><td>1.815028e-28</td></tr><tr><td>3900</td><td>0.0</td><td>1.0</td><td>1.737311e-28</td></tr><tr><td>4450</td><td>0.0</td><td>1.0</td><td>1.669088e-28</td></tr><tr><td>5000</td><td>0.0</td><td>1.0</td><td>1.608124e-28</td></tr></table>

## Regime C: narrow grid, low truncation.

When both the prefactor range and the harmonic basis are restricted, small but persistent deviations from the theoretical scaling remain visible; see Table 5. In particular, the scan tends to settle near $\alpha = 0 . 0 5$ for larger sample sizes. This suggests that when the dictionary is too rigid, the optimization may partially compensate by distorting the prefactor.

## Regime D: narrow grid, high truncation.

With a richer harmonic basis, the prefactor again stabilizes close to the theoretical value; see Table 6. Small oscillations around $\alpha = 0$ remain for some intermediate sample sizes, but they correspond to numerically indistinguishable fits and do not afect the overall conclusion.

Taken together, the noiseless experiments show that prefactor recovery is highly robust provided that the harmonic dictionary is expressive enough. When the dictionary is too restrictive, part of the residual functional error leaks into the prefactor scan.

## 6.2.4 Results with noise

We now repeat the same four regimes under additive Gaussian noise with levels $\sigma _ { \mathrm { r e l } } ~ \in ~ \{ 0 . 0 2 , 0 . 2 , 0 . 6 \}$ . The full numerical results are summarized in Figure B1 of

Table 5: Black-body Regime C: narrow prefactor grid $( \alpha \in [ - 0 . 5 , 0 . 5 ] )$ ), low truncation $( N _ { f } = 5 )$
<table><tr><td> $N _ { \mathrm { d a t } }$ </td><td>Best </td><td>Best  $R ^ { 2 }$ </td><td>Best MSE</td></tr><tr><td>50</td><td>0.05</td><td>0.999879</td><td> $1 . 8 9 4 5 2 5 \mathrm { e } { \mathrm { - } } 1 9$ </td></tr><tr><td>600</td><td>-0.05</td><td>0.999977</td><td>3.485987e-20</td></tr><tr><td>1150</td><td>0.00</td><td>0.999998</td><td>3.589867e-21</td></tr><tr><td>1700</td><td>0.00</td><td>0.999998</td><td>3.440877e-21</td></tr><tr><td>2250</td><td>0.00</td><td>0.999998</td><td>3.339133e-21</td></tr><tr><td>2800</td><td>0.00</td><td>0.999998</td><td>3.255943e-21</td></tr><tr><td>3350</td><td>0.00</td><td>0.999998</td><td>3.182589e-21</td></tr><tr><td>3900</td><td>0.05</td><td>0.999998</td><td>2.954565e-21</td></tr><tr><td>4450</td><td>0.05</td><td>0.999998</td><td>2.900151e-21</td></tr><tr><td>5000</td><td>0.05</td><td>0.999998</td><td> $2 . 8 4 7 7 4 1 \mathrm { e } { - } 2 1$ </td></tr></table>

Table 6: Black-body Regime D: narrow prefactor grid $( \alpha ~ \in ~ [ - 0 . 5 , 0 . 5 ] )$ high truncation $( N _ { f } = 6 0 )$
<table><tr><td> $N _ { \mathrm { d a t } }$ </td><td>Best </td><td>Best  $R ^ { 2 }$ </td><td>Best MSE</td></tr><tr><td>50</td><td>0.50</td><td>1.0</td><td>6.046916e-28</td></tr><tr><td>600</td><td>0.00</td><td>1.0</td><td>2.807961e-28</td></tr><tr><td>1150</td><td>0.00</td><td>1.0</td><td>2.365705e-28</td></tr><tr><td>1700</td><td>-0.10</td><td>1.0</td><td>2.033666e-28</td></tr><tr><td>2250</td><td>-0.05</td><td>1.0</td><td>1.958330e-28</td></tr><tr><td>2800</td><td>0.00</td><td>1.0</td><td>1.906302e-28</td></tr><tr><td>3350</td><td>0.00</td><td>1.0</td><td>1.815028e-28</td></tr><tr><td>3900</td><td>0.00</td><td>1.0</td><td>1.737311e-28</td></tr><tr><td>4450</td><td>0.00</td><td>1.0</td><td>1.669088e-28</td></tr><tr><td>5000</td><td>0.00</td><td>1.0</td><td> $1 . 6 0 8 1 2 4 \mathrm { { e } { - } 2 8 }$ </td></tr></table>

Appendix B. The main conclusion is that the dimensional component of the model remains substantially more stable than the functional one.

Across all regimes, the recovered exponent stays at or very near the theoretical value $\alpha = 0$ , even when the quality of the fitted modulation degrades at high noise. The broad-grid regimes recover $\alpha = 0$ almost uniformly, while the narrow-grid regimes display only small deviations of order $\pm 0 . 0 5 \mathrm { o r } \pm 0 . 1 0 $ . This indicates that the dimensional skeleton acts as a robust low-complexity structural constraint, much less sensitive to noise than the detailed harmonic content of the fitted function.

A second clear trend is the expected bias-variance trade-of associated with the truncation order. Low-order dictionaries are more robust but may introduce approximation bias, which can then distort the estimated prefactor. High-order dictionaries remove most of that bias but may begin to overfit noise. This is most visible in Regime B at $\sigma _ { \mathrm { r e l } } = 0 . 6$ , where the correct prefactor is still recovered but the fitted modulation starts to track noise fluctuations.

Another useful diagnostic is the saturation of out-of-sample $R ^ { 2 }$ at moderate and high noise levels. Writing the noisy target as $y = y _ { \mathrm { t r u e } }$ \`ε with $\varepsilon \sim \mathcal { N } ( 0 , \sigma _ { \mathrm { r e l } } ^ { 2 } \operatorname { v a r } ( y _ { \mathrm { t r u e } } ) )$ , and normalizing the noiseless signal to unit variance so that $\mathrm { v a r } ( y _ { \mathrm { t r u e } } ) = 1$ , even a perfect predictor of $y _ { \mathrm { t r u e } }$ is bounded by

$$
R _ { \infty } ^ { 2 } = \frac { 1 } { 1 + \sigma _ { \mathrm { r e l } } ^ { 2 } } .\tag{38}
$$

This saturation value assumes that the fitted model is an unbiased predictor of the noiseless signal. Therefore, systematic deviations below $R _ { \infty } ^ { 2 }$ should not be attributed solely to observation noise; they also indicate approximation bias, model misspecification, or an insuficiently expressive dictionary. This gives $R _ { \infty } ^ { 2 } \approx 0 . 9 9 9$ for $\sigma _ { \mathrm { r e l } } = 0 . 0 2$ and $R _ { \infty } ^ { 2 } \approx 0 . 7 3 5$ for $\sigma _ { \mathrm { r e l } } = 0 . 6$ , which agrees closely with the empirical plateaus seen in the noisy experiments. See figures 4 and 5.

![](images/f6ea4987351583fdec229a7c3796838626ab3b154d99ed8fbdf8e6d5abe3931a.jpg)  
Fig. 4: Regime B $( N _ { f } = 6 0 , \alpha \in [ - 5 , 5 ] )$ , low noise $( \sigma _ { \mathrm { r e l } } = 0 . 0 2 , N _ { \mathrm { d a t } } = 5 0 0 0 )$ : nearperfect reconstruction of the Planck curve.

## 6.2.5 Interpretation

The black-body benchmark illustrates a useful distinction between the dimensional and functional parts of the model. The dimensional skeleton is recovered more robustly than the detailed modulation, even when the latter is only approximately captured or partially overfitted.

![](images/16a54c9564dd5e802b011c3372dc44f7e4bb4dc29377c89c31b0c4aa2965a466.jpg)  
Fig. 5: Example of functional overfitting in the high-noise regime (Regime B, $\sigma _ { \mathrm { r e l } } =$ 0.6, $N _ { f } = 6 0 )$ . The prefactor scan still identifies the correct dimensional scaling, but the fitted modulation begins to track noise fluctuations.

At the same time, the experiments show that prefactor selection and functional approximation are not fully separable. When the harmonic basis is too rigid, approximation bias can leak into the prefactor scan, producing small but systematic deviations from the theoretical scaling. When the basis is suficiently expressive, the scan stabilizes near the correct value and the remaining error is concentrated in the dimensionless fit itself.

For this reason, the black-body benchmark provides a more stringent validation of the framework than the simple pendulum. It shows that dimensional consistency remains informative and robust even when the residual law is highly non-polynomial, while also making clear that the success of prefactor recovery depends on matching the dictionary complexity to the structure of the dimensionless modulation.

## 6.2.6 Validation on COBE/FIRAS experimental data

We validate the framework on real experimental data: the cosmic microwave background (CMB) monopole spectrum measured by the FIRAS instrument on board the COBE satellite [25]. The dataset provides the spectral radiance $B _ { \nu }$ at 43 frequency points in the range ν P r68, 639s GHz, at the fixed CMB temperature $T = 2 . 7 2 5 \mathrm { K }$ . The dimensional structure is constructed exactly as in the synthetic benchmark: the invari ant is $\pi _ { 1 } = k _ { B } T / h \nu$ and the admissible prefactor family is $P _ { \alpha } = \nu ^ { 3 - \alpha } T ^ { \alpha } h ^ { 1 - \alpha } c ^ { - 2 } k _ { B } ^ { \alpha }$ We use RidgeCV with 5-fold cross-validation. The single-fit validation shown in Figure 6 uses $N _ { f } = 3 0$ and a 30/13 train–test split, approximately a $7 0 / 3 0$ split. For the bootstrap analysis in Figure 7, we use $N _ { f } = 8$ harmonic terms. This lower truncation remains suficiently expressive for the FIRAS spectrum while reducing overfitting artifacts under resampling, which is important given the limited number of data points.

A single train/test split selects $\hat { \alpha } = 0 . 4$ with $R _ { \mathrm { t e s t } } ^ { 2 } = 0 . 9 9 9 9 9 8 ;$ ; the theoretical $\alpha = 0$ yields $R _ { \mathrm { t e s t } } ^ { 2 } = 0$ .999982 on the same split, the diference is negligible $( \Delta R ^ { 2 } \approx 1 0 ^ { - 5 } )$ . To assess whether $\hat { \alpha } = 0 . 4$ is a statistical fluctuation, we repeat the experiment over 500 classic bootstrap samples (with replacement), training each on the in-bag observations and testing on the out-of-bag points (Figure 7). The bootstrap distribution yields $\hat { \alpha } = 0 . 1 5 { \pm } 0 . 3 6 .$ concentrated near zero over the scan range $[ - 1 , 1 ]$ , confirming that the prefactor exponent α is compatible with zero and is not sharply resolved from only 43 data points at fixed temperature. Furthermore, $\alpha = 0$ produces a mean out-of-bag $R ^ { 2 }$ of 0.976 across bootstrap samples, higher than the $R ^ { 2 } = 0 . 9 4 7$ achieved by the sampleselected αˆ. This suggests that the data are too scarce to resolve the prefactor exponent sharply, and that additional measurements would likely concentrate the estimate more strongly around $\hat { \alpha } = 0$ . The fit for $\alpha = 0$ is shown in Figure 6.

![](images/1da603c75046144ae26a8dfe069fa42a2bf5cae6b29292e6277892cd0fb7008f.jpg)

![](images/1a60ae134a20c55a01d529b45ac1a0a8fdba88de8d3512db9c11091a0e756a69.jpg)  
Fig. 6: Fit of the dimensionally constrained model to COBE/FIRAS CMB monopole spectrum data (43 points, $\alpha = 0 , N _ { f } = 3 0 )$ . Left: spectral radiance $B _ { \nu }$ as a function of frequency; the theoretical Planck curve at $T = 2 . 7 2 5 \mathrm { K }$ (dashed) is overlaid with the FIRAS measurements (blue circles) and the dimensional fit (red line). Right: pull distribution $( B _ { \nu } - \hat { B } _ { \nu } ) / \sigma$ in units of the FIRAS measurement uncertainty σ.

## 6.2.7 Harmonic versus Chebyshev basis

The choice of harmonic expansions for the dimensionless modulation deserves justification, since other bases such as Chebyshev polynomials, splines, or radial basis functions could also be employed. We compare the harmonic basis against Chebyshev polynomials mapped to r´1, 1s on the black-body benchmark, using the same number of basis functions and the same RidgeCV pipeline. Figure 8 shows the results.

Bootstrap distribution of  
![](images/9ecfd790c992e012d7f1a28e5aeb4baf286ca274a199dde09df1e3ea31c7c558.jpg)  
Fig. 7: Bootstrap distribution of the recovered prefactor exponent αˆ over 500 classic bootstrap samples (with replacement) of the 43 FIRAS data points, using $N _ { f } = 8$ harmonic terms. The theoretical value $\alpha = 0$ (solid black line) and the bootstrap result $\hat { \alpha } = 0 . 1 5 \pm 0 . 3 6$ (dashed red line) are indicated. The distribution is concentrated near zero with a modest spread $( \sigma \approx 0 . 3 6 )$

With $N _ { f } = 3$ to $N _ { f } = 5 0$ harmonic terms, the out-of-sample $R ^ { 2 }$ remains stable at 0.9996, while the Chebyshev basis degrades from 0.9997 at low order to 0.866 at $N _ { f } = 5 0$ . The Chebyshev expansion sufers from Runge’s phenomenon when approximating the transcendental modulation $\Phi ( \pi _ { 1 } ) = 2 / ( e ^ { 1 / \pi _ { 1 } } - 1 )$ : high-order algebraic polynomials oscillate strongly near the boundaries of the domain, whereas the harmonic expansion captures the exponential structure with a smooth spectral basis. Furthermore, Chebyshev requires substantially more training samples to generalize: with 20 to 30 training points, the harmonic model attains $R ^ { 2 } > 0 . 9 9 9$ while Chebyshev remains below 0.82.

## 6.2.8 Comparison with Villar et al. (2023)

We compare our harmonic-based approach with a Villar et al.–style baseline inspired by the units-equivariant framework of Villar et al. [12]. In this baseline, dimensional analysis is used to construct dimensionless inputs, and inference is then performed in the resulting dimensionless space using a standard machine-learning model. Both approaches share the same dimensional reduction: the target is divided by the dimensional prefactor $P _ { 0 } ~ = ~ \nu ^ { 3 } h / c ^ { 2 }$ , and the regression is performed on the invariant $\pi _ { 1 } = k _ { B } T / h \nu$ . Our method uses a harmonic expansion with $N _ { f } = 8$ terms and RidgeCV with cross-validation, yielding a linear-in-parameters predictor with explicit harmonic coeficients. The Villar-style baseline employs a multi-layer perceptron (MLP) with three hidden layers of 128, 128, and 64 units and ReLU activations, trained on the same invariant coordinate.

![](images/5576d06f675be91d864e765f681abe7144414b6f528c0aaa0686af143263c921.jpg)  
Fig. 8: Comparison of harmonic and Chebyshev polynomial bases on the black-body benchmark. Left: out-of-sample $R ^ { 2 }$ as a function of the truncation order $N _ { f }$ , with both bases using the same number of features $\left( 1 + 2 N _ { f } \right)$ . Right: sample eficiency at fixed $N _ { f } = 8 .$ , showing that the harmonic expansion generalizes reliably from as few as 20 training points while Chebyshev requires at least 100.

The results are summarized in Table 7 and Figure 9. Both methods achieve high accuracy, but the harmonic expansion consistently outperforms the ML ${ \mathrm { . P } } ,$ particularly at low sample sizes $( R ^ { 2 } = 0 . 9 9 9 9 9 6$ vs. 0.989688 at $N = 3 0$ training points) and under noise (e.g., $R ^ { 2 } = 0 . 7 2 1$ vs. 0.683 at $\sigma _ { \mathrm { r e l } } = 0 . 6 )$ . The harmonic model uses a convex fitting objective and explicit coeficients for the dimensionless response. The MLP is more flexible, but in this experiment it requires more data to generalize and involves a non-convex optimization problem.

Table 7: Comparison between the proposed harmonic model and a Villar et al.–style MLP baseline on the black-body benchmark $\begin{array} { r l r } { ( N _ { f } } & { { } = } & { 8 } \end{array}$ , 70/30 train/test split).
<table><tr><td>σrel</td><td>Harmonic (ours)</td><td>MLP on π1 (Villar-style baseline)</td></tr><tr><td>0.00</td><td>1.000000</td><td>0.987803</td></tr><tr><td>0.02</td><td>0.999652</td><td>0.987709</td></tr><tr><td>0.10</td><td>0.990905</td><td>0.980457</td></tr><tr><td>0.30</td><td>0.923498</td><td>0.891541</td></tr><tr><td>0.60</td><td>0.721088</td><td>0.683265</td></tr></table>

![](images/39c39f6a22faa605bded6567c9e05512b3c331fbf476ffc6abc15020de6dce52.jpg)

![](images/a3f7b021740099e8c2f02b77cbf4508bd416ba5ee56f3ae2f10cca992327138c.jpg)  
Fig. 9: Comparison of our harmonic expansion method against a Villar et al.style MLP operating on the same dimensionless invariant space. Left: out-of-sample $R ^ { 2 }$ as a function of training dataset size (noiseless). Right: out-of-sample $R ^ { 2 }$ under increasing additive noise $( N = 2 0 0 )$ . The harmonic model matches or exceeds the MLP in these tests while using a linear fit in the coeficients.

This comparison should not be interpreted as a replacement for the more general units-equivariant framework of Villar et al., but as a controlled baseline within the same dimensionless input space. The purpose is to isolate the efect of replacing a flexible black-box regressor by an explicit harmonic dictionary. In this setting, the harmonic model keeps the same dimensional reduction but uses a smaller explicit dictionary instead of a neural regressor.

Taken together, these comparisons provide complementary ablations of the proposed framework. The unconstrained pendulum baseline removes the dimensional reduction and fits the raw variables directly, testing the efect of imposing Buckingham structure and an admissible dimensional prefactor. The Chebyshev comparison keeps the same dimensional prefactor and invariant coordinate but replaces the harmonic dictionary by a non-periodic polynomial basis. Finally, the Villar et al.–style baseline keeps the same dimensionless input space but replaces the explicit harmonic dictionary by a flexible black-box regressor. Together, these comparisons separate the efects of dimensional reduction, prefactor structure, dictionary choice, and regression architecture.

## 6.3 Double pendulum Lyapunov field

As a third and substantially more demanding benchmark, we consider the maximal Lyapunov exponent field of the double pendulum. The target is a structured twodimensional response surface with strong anisotropy and sharp transitions, which makes it a stringent test of dictionary design on invariant space.

## 6.3.1 Physical setting and dimensional structure

The double pendulum consists of two point masses $m _ { 1 }$ and $m _ { 2 }$ connected by rigid massless rods of lengths $l _ { 1 }$ and $l _ { 2 }$ in a uniform gravitational field $g .$ . Its angular equations

of motion are nonlinear and strongly coupled:

$$
\ddot { \theta } _ { 1 } = \frac { - g ( 2 m _ { 1 } + m _ { 2 } ) \sin \theta _ { 1 } - m _ { 2 } g \sin ( \theta _ { 1 } - 2 \theta _ { 2 } ) - 2 m _ { 2 } \sin ( \theta _ { 1 } - \theta _ { 2 } ) \left( \dot { \theta } _ { 2 } ^ { 2 } l _ { 2 } + \dot { \theta } _ { 1 } ^ { 2 } l _ { 1 } \cos ( \theta _ { 1 } - \theta _ { 2 } ) \right) } { l _ { 1 } \left( 2 m _ { 1 } + m _ { 2 } - m _ { 2 } \cos ( 2 \theta _ { 1 } - 2 \theta _ { 2 } ) \right) } \sin \theta _ { 1 } ,
$$

$$
\ddot { \theta } _ { 2 } = \frac { 2 \sin ( \theta _ { 1 } - \theta _ { 2 } ) \left( \dot { \theta } _ { 1 } ^ { 2 } l _ { 1 } ( m _ { 1 } + m _ { 2 } ) + g ( m _ { 1 } + m _ { 2 } ) \cos \theta _ { 1 } + \dot { \theta } _ { 2 } ^ { 2 } l _ { 2 } m _ { 2 } \cos ( \theta _ { 1 } - \theta _ { 2 } ) \right) } { l _ { 2 } \left( 2 m _ { 1 } + m _ { 2 } - m _ { 2 } \cos ( 2 \theta _ { 1 } - 2 \theta _ { 2 } ) \right) } .\tag{39}
$$

To quantify sensitivity to initial conditions, we study the maximal Lyapunov exponent

$$
\lambda _ { \operatorname* { m a x } } = \operatorname* { l i m } _ { t \to \infty } \operatorname* { l i m } _ { \| \delta _ { 0 } \| \to 0 } \frac { 1 } { t } \log \frac { \| \delta ( t ) \| } { \| \delta _ { 0 } \| } .\tag{40}
$$

Numerically, $\lambda _ { \mathrm { m a x } }$ is estimated from the divergence of nearby trajectories initialized with a small angular perturbation. The resulting map $( \theta _ { 1 } , \theta _ { 2 } ) \mapsto \lambda _ { \mathrm { { m a x } } } ( \theta _ { 1 } , \theta _ { 2 } )$ defines a structured response surface over the angular configuration space. We use the dimensional basis $B = \{ { \sf M } , { \sf L } , { \sf T } \}$ . The dimension matrix for the dimensional variables $( m _ { 1 } , m _ { 2 } , l _ { 1 } , l _ { 2 } , g )$ is

$$
A = { \binom { 1 \ 1 \ 0 \ 0 \ 0 } { 0 \ 0 \ 1 \ 1 } } .\tag{41}
$$

Since the angular variables are already dimensionless, Buckingham’s theorem gives $N _ { \Pi } \quad = \quad 7 \ - \ 3 \ = \quad 4$ independent invariants when the full variable set $( m _ { 1 } , m _ { 2 } , l _ { 1 } , l _ { 2 } , g , \theta _ { 1 } , \theta _ { 2 } )$ is considered. Solving $A \gamma = \mathbf { 0 }$ yields,

$$
\pi _ { 1 } = \theta _ { 1 } , \qquad \pi _ { 2 } = \theta _ { 2 } , \qquad \pi _ { 3 } = \frac { m _ { 2 } } { m _ { 1 } } , \qquad \pi _ { 4 } = \frac { l _ { 2 } } { l _ { 1 } } .\tag{42}
$$

For the target signature $[ \lambda _ { \mathrm { m a x } } ] = T ^ { - 1 }$ , solving $A \beta = \mathbf { b }$ yields the admissible prefactor family

$$
P _ { \eta , \xi } = \left( \frac { m _ { 2 } } { m _ { 1 } } \right) ^ { \eta } \left( \frac { l _ { 2 } } { l _ { 1 } } \right) ^ { - \xi } \sqrt { \frac { g } { l _ { 1 } } } .\tag{43}
$$

In the symmetric baseline configuration, where $m _ { 1 } = m _ { 2 }$ and $l _ { 1 } = l _ { 2 }$ , this reduces to the unique minimal prefactor $\begin{array} { r } { P = \sqrt { \frac { g } { l _ { 1 } } } } \end{array}$ . Accordingly, the remaining learning task is concentrated entirely in the two-dimensional angular modulation.

## 6.3.2 Data generation and benchmark configurations

Two numerical configurations are considered.

• In the symmetric baseline case, we fix $m _ { 1 } = m _ { 2 } = 1$ kg, $l _ { 1 } = l _ { 2 } = 2 0 0$ m, and sample the Lyapunov field on a 401 ˆ 401 grid over $( \theta _ { 1 } , \theta _ { 2 } ) \in [ - \pi , \pi ] ^ { 2 }$

• In the asymmetric case, unequal masses are used and the field is sampled on a $2 0 1 \times 2 0 1$ grid, while the integration protocol remains unchanged. This second configuration is included to verify that the advantage of the separable dictionary is not specific to the symmetric geometry of the baseline field.

For both settings, the learned model has the generic form

$$
\hat { \lambda } _ { \mathrm { m a x } } = P _ { \eta , \xi } \Phi ( \theta _ { 1 } , \theta _ { 2 } ) ,\tag{44}
$$

with $P = \sqrt { g / l _ { 1 } }$ in the symmetric case. Since the nontrivial dependence is genuinely two-dimensional, this benchmark is the natural place to compare the two harmonic constructions introduced earlier.

## 6.3.3 Phase-combination dictionary

We first use the phase-combination basis

$$
\Phi ( u _ { 1 } , u _ { 2 } ) = w _ { 0 } + \sum _ { \mathbf { n } \neq \mathbf { 0 } } \Bigl [ w _ { \mathbf { n } } ^ { c } \cos ( n _ { 1 } u _ { 1 } + n _ { 2 } u _ { 2 } ) + w _ { \mathbf { n } } ^ { s } \sin ( n _ { 1 } u _ { 1 } + n _ { 2 } u _ { 2 } ) \Bigr ] ,\tag{45}
$$

where $( u _ { 1 } , u _ { 2 } )$ are the rescaled angular invariants. This basis is straightforward to implement, but it couples both directions through a single phase and therefore has limited flexibility when the response surface exhibits diferent structures along different angular directions. The corresponding performance in the symmetric baseline configuration is summarized in Table 8. Throughout this benchmark, $N _ { f }$ denotes the per-coordinate truncation order used uniformly across both angular invariant directions $( \mathrm { i . e . ~ } N _ { f } ^ { ( 1 ) } = N _ { f } ^ { ( 2 ) } = N _ { f } )$

Table 8: Performance of the phasecombination dictionary for the symmetric baseline configuration $( m _ { 1 } =$ $m _ { 2 } , l _ { 1 } = l _ { 2 } )$
<table><tr><td>Truncation  $\left( N _ { f } \right)$ </td><td> $R ^ { 2 }$ </td><td>MSE</td></tr><tr><td>5</td><td>0.791778</td><td>0.517102</td></tr><tr><td>20</td><td>0.816322</td><td>0.456150</td></tr><tr><td>30</td><td>0.823415</td><td>0.438534</td></tr></table>

The improvement with truncation order is modest, and the fit saturates near $R ^ { 2 }$ « 0.82. This suggests that the main limitation is not the number of frequencies alone, but the geometry of the basis itself.

## 6.3.4 Separable tensor-product dictionary

We next replace the phase-combination basis by the separable tensor-product dictionary $\begin{array} { r l r } { \Phi ( u _ { 1 } , u _ { 2 } ) } & { = } & { \sum _ { { \bf k } \in { \mathcal T } _ { { \bf k } } } w _ { { \bf k } } \Psi _ { { \bf k } } ( u _ { 1 } , u _ { 2 } ) } \end{array}$ , where each feature is a product of one-dimensional trigonometric factors in $u _ { 1 }$ and $u _ { 2 }$ . This separates the two angular directions before interaction terms are introduced. The results for the symmetric baseline configuration are given in Table 9.

Table 9: Performance of the separable tensor-product dictionary for the symmetric baseline configuration $( m _ { 1 } = m _ { 2 } , l _ { 1 } = l _ { 2 } )$
<table><tr><td>Truncation  $\left( N _ { f } \right)$ </td><td> $R ^ { 2 }$ </td><td>MSE</td></tr><tr><td>5</td><td>0.860839</td><td>0.345595</td></tr><tr><td>20</td><td>0.933313</td><td>0.165611</td></tr></table>

At the same nominal truncation level, the separable basis outperforms the phasecombination model. This indicates that the limitation of the simpler representation is geometric: the Lyapunov field is anisotropic, and the dictionary must resolve diferent directional structures. The ten largest fitted coeficients for the $N _ { f } = 2 0$ separable model are shown in Table 10. For readability we write $C _ { k _ { 1 } , k _ { 2 } }$ for the coeficient of $\cos ( k _ { 1 } u _ { 1 } ) \cos ( k _ { 2 } u _ { 2 } )$ and $S _ { k _ { 1 } , k _ { 2 } }$ for the coeficient of sin $\left( k _ { 1 } u _ { 1 } \right) \sin ( k _ { 2 } u _ { 2 } )$ ; each corresponds uniquely to a scalar weight $w _ { \mathbf { k } }$ from the tensor-product expansion of Section 3. The subscript pair $( k _ { 1 } , k _ { 2 } )$ gives the angular-frequency index along the $\theta _ { 1 }$ and $\theta _ { 2 }$ invariant directions, respectively. Only even-parity products survive in the parity-reduced model (Table 12), so no mixed sin $( k _ { 1 } u _ { 1 } )$ cospk<sub>2</sub>u<sub>2</sub>q or cos $( k _ { 1 } u _ { 1 } )$ sin $( k _ { 2 } u _ { 2 } )$ terms appear.

Table 10: Ten largest coeficients for the separable model at $N _ { f } = 2 0$
<table><tr><td>Term</td><td>Value</td></tr><tr><td> $C _ { 1 , 0 }$  (cos-cos)</td><td>-7.976002</td></tr><tr><td> $S _ { 1 , 1 }$  (sin-sin)  $C _ { 0 , 1 }$  (cos-cos)</td><td>-3.137987 -2.709306</td></tr><tr><td> $S _ { 1 , 2 }$  (sin-sin) (sin-sin)</td><td>2.003437</td></tr><tr><td> $S _ { 3 , 2 }$   $C _ { 2 , 3 }$  (cos-cos) (cos-cos)</td><td>1.498303</td></tr><tr><td></td><td>-1.259975 1.184035</td></tr><tr><td> $C _ { 2 , 1 }$ </td><td></td></tr><tr><td> $C _ { 2 , 0 }$  (cos-cos)</td><td>1.168022</td></tr><tr><td>(sin-sin)</td><td></td></tr><tr><td> $S _ { 1 , 3 }$ </td><td>-0.956398</td></tr><tr><td> $C _ { 3 , 1 }$  (cos-cos)</td><td>-0.908308</td></tr></table>

## 6.3.5 Parity reduction

Inspection of the fitted coeficients and of the Lyapunov field itself reveals a strong approximate symmetry under the joint reflection $( \theta _ { 1 } , \theta _ { 2 } ) \ \mapsto \ ( - \theta _ { 1 } , - \theta _ { 2 } )$ This motivates imposing the parity reduction, retaining only the even-parity blocks cos $( k _ { 1 } u _ { 1 } )$ cos $( k _ { 2 } u _ { 2 } )$ and $\sin ( k _ { 1 } u _ { 1 } ) \sin ( k _ { 2 } u _ { 2 } )$ . The resulting performance is summarized in Table 11.

Table 11: Performance of parity-reduced dictionaries for the symmetric baseline configuration $( m _ { 1 } = m _ { 2 } , l _ { 1 } = l _ { 2 } )$
<table><tr><td>Dictionary Type</td><td> $N _ { f }$ </td><td> $R ^ { 2 }$ </td><td>MSE</td></tr><tr><td>Separable + Parity</td><td>20</td><td>0.933290</td><td>0.165668</td></tr><tr><td>Phase-Comb + Parity</td><td>40</td><td>0.826657</td><td>0.430482</td></tr><tr><td>Phase-Comb + Parity</td><td>50</td><td>0.828744</td><td>0.425301</td></tr><tr><td>Separable + Parity</td><td>35</td><td>0.950272</td><td>0.123495</td></tr></table>

Parity reduction alone does not resolve the limitations of the phase-combination basis. The best result is obtained by the parity-reduced separable model, which reaches $R ^ { 2 } = 0 . 9 5 0 2 7 2$ at $N _ { f } = 3 5$ . Thus, symmetry is useful only after the dictionary geometry is already well matched to the target field. For completeness, the ten largest coeficients of the best model are reported in Table 12.

Table 12: Ten largest coeficients for the parityreduced separable model at $N _ { f } = 3 5$
<table><tr><td>Term</td><td>Value</td></tr><tr><td> $C _ { 1 , 0 }$  (cos-cos)</td><td>-7.972290</td></tr><tr><td> $S _ { 1 , 1 }$  (sin-sin)  $C _ { 0 , 1 }$  (cos-cos)</td><td>-3.137987 -2.708697</td></tr><tr><td> $S _ { 1 , 2 }$  (sin-sin)  $S _ { 3 , 2 }$  (sin-sin)</td><td>2.003437</td></tr><tr><td> $C _ { 2 , 3 }$  (cos-cos)  $C _ { 2 , 1 }$  (cos-cos)</td><td>1.498303 -1.260299</td></tr><tr><td></td><td>1.187112</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td> $C _ { 2 , 0 }$  (cos-cos)</td><td>1.163263</td></tr><tr><td> $S _ { 1 , 3 }$  (sin-sin)</td><td>-0.956398</td></tr><tr><td> $C _ { 3 , 1 }$  (cos-cos)</td><td>-0.910151</td></tr></table>

Figure 10 shows the progression in reconstruction quality across the diferent dictionary choices.

![](images/f0be8b3900f026f99d504656782b47ae96f8f2e2755d147460266f2926a6674c.jpg)  
(a) Ground-truth Lyapunov field.

![](images/ed456597611fa47fcac750668cc5826fdcfcb6135831bc264c0050f9145fa607.jpg)  
(b) Phase-combination fit at $N _ { f } = 2 0$

![](images/48e4149e4276ebbe889ae858b7e6ae35fdb657be08767f5458061c418334ec0b.jpg)  
(c) Separable fit at $N _ { f } = 2 0 .$

![](images/05f85d3b33e4e4348fdb288a4c961a80acb34bfceee9a932dea7cfc1a5edf6a0.jpg)  
(d) Parity-reduced separable fit at $N _ { f } = 3 5$  
Fig. 10: Progressive improvement in the reconstruction of the double-pendulum Lyapunov field. The separable tensor-product dictionary is essential for resolving the anisotropic angular structure, while parity reduction improves eficiency without degrading accuracy.

## 6.3.6 Asymmetric case

We also consider an asymmetric mass configuration, for which the ratio invariant $m _ { 2 } / m _ { 1 }$ is no longer trivial. In this setting, the prefactor family

$$
P _ { \eta , \xi } = \left( \frac { m _ { 2 } } { m _ { 1 } } \right) ^ { \eta } \left( \frac { l _ { 2 } } { l _ { 1 } } \right) ^ { - \xi } \sqrt { \frac { g } { l _ { 1 } } }\tag{46}
$$

contains nontrivial free parameters. Once the ratio invariants are fixed numerically, however, constant multiplicative factors are readily absorbed into the harmonic coeficients, so predictive performance depends much more strongly on the dictionary choice than on the precise prefactor exponent. The corresponding results are reported in Table 13.

Table 13: Results for the asymmetric-mass double-pendulum configuration.
<table><tr><td>Dictionary Type</td><td> $N _ { f }$ </td><td> $R ^ { 2 }$ </td><td>MSE</td></tr><tr><td>Phase-Combination</td><td>5</td><td>0.704248</td><td>3.395539</td></tr><tr><td>Phase-Combination</td><td>20</td><td>0.717002</td><td>3.249113</td></tr><tr><td> ${ \mathrm { S e p a r a b l e } } + { \mathrm { P a r i t y } }$ </td><td>5</td><td>0.908818</td><td>1.046867</td></tr><tr><td> ${ \mathrm { S e p a r a b l e } } + { \mathrm { P a r i t y } }$ </td><td>20</td><td>0.942478</td><td>0.660411</td></tr></table>

The same qualitative pattern persists: the phase-combination basis remains limited, whereas the separable parity-reduced model stays highly efective. This confirms that the advantage of the separable construction is not specific to the symmetric baseline case. Figure 11 illustrates the reconstruction quality in the asymmetric setting.

## 6.3.7 Interpretation

The double-pendulum benchmark separates two aspects of the method. The dimensional prefactor fixes the overall scaling, but the accuracy is controlled by the dictionary used for the dimensionless modulation. Phase-combination harmonic features saturate early because they do not match the anisotropic structure of the Lyapunov field, whereas separable tensor-product dictionaries resolve the angular directions independently.

Symmetry reduction gives an additional reduction in feature count, but its efect is secondary to the choice of dictionary. On the symmetric benchmark, the phasecombination model plateaus near $R ^ { 2 } \approx 0 . 8 2$ , whereas the parity-reduced separable model reaches $R ^ { 2 } = 0 . 9 5$

## 6.3.8 Harmonic versus Chebyshev basis

We also compare the separable harmonic dictionary against a separable Chebyshev polynomial basis on the same angular domain (Figure 12). Chebyshev polynomials are orthogonal on $[ - 1 , 1 ]$ , whereas the angular coordinates $\theta _ { 1 } , \theta _ { 2 }$ are naturally periodic on $[ - \pi , \pi ]$ ; mapping the angles to r´1, 1s introduces an artificial discontinuity at the domain boundaries. At matched truncation order $N _ { f } = 8 .$ , the harmonic model achieves $R ^ { 2 } = 0$ .882 against $R ^ { 2 } = 0 . 8 4 8$ for Chebyshev, and the gap persists with increasing $N _ { f } \ ( R ^ { 2 } = 0 . 9 0 7 \ \mathrm { v s . } \ 0 . 8 6 9$ at $N _ { f } = 1 2 )$ . This is consistent with the angular nature of the invariant coordinates in this benchmark.

![](images/8df1f60606f42e030db5d949fa5144369ef6260a3e3fb37b4a509e1d4f08d8e4.jpg)  
(a) Ground-truth Lyapunov field.

![](images/ef544bdd35879ae6a3c4c03b15986bbc1520494fbcb38976b0846e074b448483.jpg)  
(b) Phase-combination fit at $N _ { f } = 2 0$

![](images/3b72cfe90e72cd7256ead9f5f5155692d51da39dbea9773c0cbb6886926a1aa4.jpg)  
(c) Parity-reduced separable fit at $N _ { f } = 5 .$

![](images/57bce78c3a4f99267d30c1243b441631cb1efea61e9429f0313507e6ef77df1f.jpg)  
(d) Parity-reduced separable fit at $N _ { f } = 2 0$  
Fig. 11: Reconstruction quality for the asymmetric double-pendulum configuration. The separable tensor-product basis again provides a substantially better approximation than the phase-combination basis.

![](images/227ea2af2d5e10e7ddac5a47ab5707c6eca49d30a4cb73c3c88cc63a7517ae79.jpg)

![](images/ae579aca3053c800c034b56f3006bb4e74a7b6349b7832b433bce2a0a89549e3.jpg)

![](images/2cdbaae65d340e1896ffde2ecb1bcec3a2dfb6c8356adc01abfa2c56f26dff4d.jpg)  
Fig. 12: Comparison of separable harmonic and separable Chebyshev dictionaries on the double-pendulum Lyapunov field. Left: ground truth. Center: harmonic fit at $N _ { f } = 8 .$ . Right: Chebyshev fit at $N _ { f } = 8$ . The harmonic basis captures both marginal and interaction angular structure more faithfully.

## 6.3.9 Convergence with truncation order

The results reported in Tables 8–11 are limited to three values of $N _ { f } \in \{ 5 , 2 0 , 3 5 \}$ . To assess whether $R ^ { 2 } \approx 0 . 9 5$ at $N _ { f } = 3 5$ represents saturation or a transient plateau, we extend the scan to $N _ { f } = 6 0$ using the parity-reduced separable model with a train/test split (Figure 13). The training $\bar { R } ^ { 2 }$ increases monotonically with $N _ { f } .$ , reaching 0.991 at $N _ { f } = 6 0$ , as expected from a model with growing expressivity. The out-of-sample $R ^ { 2 }$ however, peaks at $N _ { f } \approx 3 0 \ ( R ^ { 2 } = 0 . 9 3 0 )$ and then declines to 0.633 at $N _ { f } = 6 0$ due to overfitting with the available number of training samples. The observed plateau in the earlier tables is therefore not a property of the harmonic basis but rather reflects the finite sample budget: the model saturates when the number of features approaches the efective number of training degrees of freedom. This behavior is consistent with standard bias–variance trade-of expectations for regularized regression.

## 6.3.10 Spatial distribution of the prediction error

The aggregate metrics in Tables 8–11 abstract away the spatial structure of the error. Figure 14 shows heatmaps of $| \hat { \lambda } _ { \mathrm { m a x } } - \lambda _ { \mathrm { m a x } } |$ over the angular domain for the parity-reduced separable model at three truncation orders. The error is systematically concentrated along the boundaries between regular and chaotic regions of the phase space, where the Lyapunov field exhibits sharp gradients. At $N _ { f } = 5$ , errors of order Op1q span large portions of the domain; at $N _ { f } = 2 0$ , the high-error regions shrink to thin filaments along the separatrix; at $N _ { f } = 3 5$ , errors are further reduced in magnitude but remain localized at the same interface structures. This confirms that the primary challenge for the harmonic approximation is not uniform field complexity but the resolution of narrow transition zones, for which higher-order harmonic terms provide diminishing returns once the dominant interface width is captured.

![](images/4b1fc99f09f060319b119beb677030adf68049ed5081fb76801734bcbc321ed2.jpg)  
Fig. 13: Train and test $R ^ { 2 }$ as a function of the truncation order $N _ { f }$ for the parityreduced separable model on an $8 0 / 2 0$ split of 8000 subsampled grid points. Test performance peaks at $N _ { f } \approx 3 0$ and decreases at higher $N _ { f }$ due to overfitting, indicating that the plateau reported in the earlier tables is a consequence of finite training samples rather than a limitation of the harmonic representation.

## 7 Discussion

The three benchmarks probe complementary aspects of the proposed framework: a low-dimensional law largely determined by dimensional analysis, a one-dimensional but transcendental modulation with a nontrivial prefactor family, and a structured two-dimensional response surface with partial chaos. Taken together, they show that dimensional consistency is useful, but that the residual dimensionless dependence still has to be represented appropriately. The unconstrained pendulum baseline in Section 6.1 quantifies the sample-eficiency gain: by fixing the exponent from dimensional analysis, the proposed model avoids the ill-conditioned joint optimization over $( \alpha , w _ { 0 } , w _ { 1 } )$ and reaches high accuracy with far fewer data points.

A first observation is that the dimensional component is more stable than the functional one. In the black-body benchmark, for example, the correct prefactor is recovered even in regimes where the harmonic approximation begins to overfit noise. This suggests that the admissible prefactor family acts as a low-complexity constraint, whereas the fitted dimensionless modulation carries the higher-variance part of the approximation.

A second observation is that dictionary geometry becomes important as soon as the invariant space has dimension greater than one. In the double-pendulum benchmark, phase-combination harmonic features saturate at comparatively modest accuracy, whereas separable tensor-product dictionaries yield a large improvement. The reason is geometric rather than purely spectral: the Lyapunov field is anisotropic, and a basis that couples all directions through a single phase is not well adapted to that structure. By contrast, separable dictionaries expose marginal and interaction efects directly and allow the representation to follow the directional organization of the target surface.

![](images/4919e65ed2686b5b286c7acdb18852c5d3c49a96d970deb212abb43f62f872ae.jpg)  
Fig. 14: Heatmaps of the absolute prediction error $| \hat { \lambda } _ { \mathrm { m a x } } - \lambda _ { \mathrm { m a x } } |$ for the parity-reduced separable model at three truncation orders. Errors concentrate at the boundaries between regular and chaotic regions and narrow progressively with increasing $N _ { f } .$ , but the interface structure persists.

The experiments also show that symmetry information can be incorporated in a useful way. In the double-pendulum case, imposing joint parity reduces the number of fitted coeficients without degrading accuracy, and in the best models it improves eficiency at essentially no cost in expressivity on the physically relevant subspace. This illustrates the complementary roles of dimensional analysis and geometric symmetry: the former constrains the admissible scaling structure, while the latter restricts the dimensionless modulation within invariant space.

A further methodological point is that prefactor selection and functional approximation cannot be treated as fully independent. When the harmonic dictionary is too rigid, residual approximation error may leak into the prefactor scan, leading to small but systematic distortions of the recovered dimensional scaling. The black-body benchmark makes this interaction particularly clear. Conversely, when the dictionary is suficiently expressive, the scan stabilizes near the physically correct prefactor. Prefactor selection should therefore be viewed not merely as a dimensional bookkeeping device, but as part of the overall bias-variance trade-of of the model.

Several limitations should also be kept in mind. First, although the present work is primarily based on synthetic benchmarks, the COBE/FIRAS validation in Section 6.2 provides a real-data check: the bootstrap prefactor scan remains compatible with the expected scaling, but also shows the limited statistical resolution of such a small measured dataset.

Overall, dimensional analysis defines a compact physically admissible hypothesis space. The experiments also show that, once multiple invariants are present, the learning stage depends strongly on matching the dictionary to the geometry of the response surface.

Regarding the choice of basis, harmonic dictionaries work well for the transcendental and periodic invariant domains studied here. The Chebyshev comparisons in Sections 6.2 and 6.3 are useful ablations, but they do not outperform the harmonic construction on these benchmarks.

## 8 Conclusion

We have introduced a data-driven method for learning physical responses under explicit dimensional constraints. Starting from the dimension matrix of the variables, the method constructs admissible dimensional prefactors and independent Buckingham invariants, and then learns the remaining dimensionless dependence through truncated harmonic expansions on invariant space. Once the dictionary is fixed, the fitting problem is linear in the coeficients and can be regularized by standard methods.

The numerical experiments cover three qualitatively diferent settings. It recovers the expected pendulum scaling, identifies the correct black-body dimensional skeleton even under approximation error and noise, and shows that separable tensor-product dictionaries with symmetry reduction are essential for the double-pendulum Lyapunov field. The convergence and error-map analyses further indicate that the remaining error is mostly a finite-sample issue concentrated near regular-to-chaotic transition regions.

More broadly, the study highlights three methodological points. First, dimensional consistency is a useful inductive bias for data-driven law discovery. Second, the separation between dimensional scaling and dimensionless modulation suggests a path to extensions in which the dimensional skeleton is fixed by prior knowledge while the modulation is refined as more data become available. Third, once the invariant space has dimension greater than one, the representation chosen for the dimensionless modulation becomes a central design decision.

The present work focuses on problems with low-dimensional invariant spaces, where prefactor scans and moderately rich harmonic dictionaries remain tractable. Extending the framework to larger invariant dimensions will require more structured approximation strategies, such as sparse truncations, adaptive dictionaries, or low-rank tensor constructions. The results obtained here suggest that combining dimensional analysis with explicit spectral approximation is a practical route toward learning physically admissible laws from data.

## Data availability

The datasets generated and/or analysed during the current study are available from the corresponding author on reasonable request. This includes the raw synthetic datasets generated for the controlled benchmarks and the processed data used to produce the figures and tables. The COBE/FIRAS spectrum used for the real-data validation is a public dataset originally reported by Fixsen et al. [25]; the processed values used in the present analysis are also available from the corresponding author on reasonable request.

## Funding

This work was supported by the C´atedra Fundaci´on ASISA–UEM de Ciencias de la Salud, under internal project code P2025-15CA.

## Author Contributions

Ernest Tarrus: Writing - original draft, validation, software, methodology, investigation, formal analysis, and conceptualization. Hector Gisbert: Writing - original draft, visualization, supervision, methodology, investigation, formal analysis, and conceptualization.

## Appendix A Diferential form of unit equivariance

For $\lambda = ( \lambda _ { 1 } , \ldots , \lambda _ { R } ) \in ( \mathbb { R } _ { > 0 } ) ^ { R }$ , introduce logarithmic coordinates

$$
t _ { r } = \log \lambda _ { r } , \qquad r = 1 , \ldots , R .\tag{A1}
$$

Under the induced change of units, each input variable transforms as

$$
x _ { j } ( t ) = x _ { j } \exp \left( \sum _ { r = 1 } ^ { R } a _ { r j } t _ { r } \right) , \qquad j = 1 , \ldots , N .\tag{A2}
$$

The unit-equivariance condition (7) can then be written as

$$
f { \bigl ( } x _ { 1 } ( t ) , \ldots , x _ { N } ( t ) { \bigr ) } = \exp \left( \sum _ { r = 1 } ^ { R } b _ { r } t _ { r } \right) f ( x _ { 1 } , \ldots , x _ { N } ) .\tag{A3}
$$

Diferentiating with respect to $t _ { r }$ at $t = 0$ gives

$$
\sum _ { j = 1 } ^ { N } \frac { \partial f } { \partial x _ { j } } \frac { \partial x _ { j } } { \partial t _ { r } } \bigg | _ { t = 0 } = b _ { r } f ( x _ { 1 } , \ldots , x _ { N } ) .\tag{A4}
$$

Since

$$
\left. \frac { \partial x _ { j } } { \partial t _ { r } } \right| _ { t = 0 } = a _ { r j } x _ { j } ,\tag{A5}
$$

we obtain the system

$$
\sum _ { j = 1 } ^ { N } a _ { r j } x _ { j } { \frac { \partial f } { \partial x _ { j } } } = b _ { r } f , \qquad r = 1 , \ldots , R .\tag{A6}
$$

This is the diferential form of dimensional homogeneity. For a monomial ansatz

$$
f ( \mathbf { x } ) = \prod _ { j = 1 } ^ { N } x _ { j } ^ { \beta _ { j } } ,\tag{A7}
$$

substituting into (A6) yields

$$
\sum _ { j = 1 } ^ { N } a _ { r j } \beta _ { j } = b _ { r } , \qquad r = 1 , \ldots , R ,\tag{A8}
$$

that is, the algebraic system $A { \boldsymbol { \beta } } = \mathbf { b }$ . Thus, for monomial prefactors, the diferential and algebraic formulations of unit equivariance are equivalent.

## Appendix B Summary of black-body radiation with noise

The full numerical results of the four noisy regimes described in Section 6.2 are summarized in Figure B1. The top row shows the recovered prefactor exponent αˆ as a function of the dataset size $N _ { \mathrm { d a t } }$ for each regime and noise level. Across all regimes, the method recovers values very close to the theoretical exponent $\alpha = 0$ , with deviations that stay within ˘0.1 except at the smallest sample sizes. The bottom row shows the corresponding out-of-sample $R ^ { 2 }$ . For the low-noise regime $( \sigma _ { \mathrm { r e l } } = 0 . 0 2 )$ 2 $R ^ { 2 }$ remains above 0.999 at all dataset sizes and both $N _ { f } ~ = ~ 5$ (Regime $\mathrm { A } , \mathrm { C } )$ and $N _ { f } = 6 0$ (Regime B, D). At higher noise levels, the low- $N _ { f }$ model maintains $R ^ { 2 } \approx 0 . 9 6$ $( \sigma _ { \mathrm { r e l } } = 0 . 2 )$ and $R ^ { 2 } \approx 0 . 7 3 \ ( \sigma _ { \mathrm { r e l } } = 0 . 6 )$ , while the $\mathrm { h i g h } { - } \dot { N _ { f } }$ model exhibits overfitting at small $N _ { \mathrm { d a t } }$ (e.g., $R ^ { 2 } = 0 . 7 2$ at $N _ { \mathrm { d a t } } = 3 3 5 0$ for Regime B, $\sigma _ { \mathrm { r e l } } = 0 . 6 )$ . These results confirm that the dimensional component (the prefactor exponent α) is substantially more stable than the functional component $( R ^ { 2 }$ from the harmonic fit) and that the qualitative behavior is robust to the chosen scan range for $\alpha$

![](images/4d49a96e6157fdd5fddcf8337b9a03858282ba610b7580e0308f781c1226ac44.jpg)  
Fig. B1: Summary of the noisy black-body experiments across the four regimes. Top: recovered prefactor exponent αˆ vs. $N _ { \mathrm { d a t } }$ . Bottom: best $R ^ { 2 }$ vs. $N _ { \mathrm { d a t } }$ . Three noise levels are shown per panel. The prefactor exponent remains robust across all regimes, while $R ^ { 2 }$ degrades with noise and, for large- $N _ { f }$ models, also with small $N _ { \mathrm { d a t } }$

## References

[1] Bridgman, P.W.: Dimensional Analysis. Yale University Press, New Haven (1922)

[2] Barenblatt, G.I.: Scaling. Cambridge Texts in Applied Mathematics, vol. 34. Cambridge University Press, Cambridge (2003). https://doi.org/10.1017/ CBO9780511814921

[3] Buckingham, E.: On physically similar systems; illustrations of the use of dimensional equations. Physical Review 4(4), 345–376 (1914)

[4] Barenblatt, G.I.: Scaling, Self-similarity, and Intermediate Asymptotics. Cambridge University Press, Cambridge (1996)

[5] Sedov, L.I.: Similarity and Dimensional Methods in Mechanics. CRC Press, Boca Raton (1993)

[6] Vaschy, A.: Sur les lois de similitude en physique. Annales t´el´egraphiques 19, 25–28 (1892)

[7] Schmidt, M., Lipson, H.: Distilling free-form natural laws from experimental data. Science 324(5923), 81–85 (2009) https://doi.org/10.1126/science.1165893

[8] Udrescu, S.-M., Tegmark, M.: Ai feynman: A physics-inspired method for symbolic regression. Science Advances 6(16), 2631 (2020) https://doi.org/10.1126/ sciadv.aay2631

[9] Brunton, S.L., Proctor, J.L., Kutz, J.N.: Discovering governing equations from data by sparse identification of nonlinear dynamical systems. Proceedings of the National Academy of Sciences 113(15), 3932–3937 (2016) https://doi.org/10. 1073/pnas.1517384113

[10] Rudy, S.H., Brunton, S.L., Proctor, J.L., Kutz, J.N.: Data-driven discovery of partial diferential equations. Science Advances 3(4), 1602614 (2017) https://doi. org/10.1126/sciadv.1602614

[11] Raissi, M., Perdikaris, P., Karniadakis, G.E.: Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations. Journal of Computational Physics 378, 686–707 (2019) https://doi.org/10.1016/j.jcp.2018.10.045

[12] Villar, S., Yao, W., Hogg, D.W., Blum-Smith, B., Dumitrascu, B.: Dimensionless machine learning: Imposing exact units equivariance. Journal of Machine Learning Research 24(109), 1–32 (2023)

[13] Bakarji, J., Callaham, J., Brunton, S.L., Kutz, J.N.: Dimensionally consistent learning with Buckingham pi. Nature Computational Science 2(12), 834–844 (2022) https://doi.org/10.1038/s43588-022-00355-5

[14] Xie, X., Samaei, A., Guo, J., Liu, W.K., Gan, Z.: Data-driven discovery of dimensionless numbers and governing laws from scarce measurements. Nature Communications 13, 7562 (2022) https://doi.org/10.1038/s41467-022-35084-w

[15] Jofre, L., Rosario, Z.R., Iaccarino, G.: Data-driven dimensional analysis of heat transfer in irradiated particle-laden turbulent flow. International Journal of Multiphase Flow 125, 103198 (2020) https://doi.org/10.1016/j.ijmultiphaseflow.2019. 103198

[16] Tikhonov, A.N., Arsenin, V.Y.: Solutions of Ill-Posed Problems. Winston and Sons, Washington, DC (1977)

[17] Hoerl, A.E., Kennard, R.W.: Ridge regression: Biased estimation for nonorthogonal problems. Technometrics 12(1), 55–67 (1970) https://doi.org/10.1080/ 00401706.1970.10488634

[18] Tibshirani, R.: Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society: Series B (Methodological) 58(1), 267–288 (1996) https: //doi.org/10.1111/j.2517-6161.1996.tb02080.x

[19] Zou, H., Hastie, T.: Regularization and variable selection via the elastic net. Journal of the Royal Statistical Society: Series B 67(2), 301–320 (2005) https: //doi.org/10.1111/j.1467-9868.2005.00503.x

[20] Golub, G.H., Van Loan, C.F.: Matrix Computations, 4th edn. Johns Hopkins University Press, Baltimore, MD (2013)

[21] Bengio, Y.: Gradient-based optimization of hyper-parameters. Neural computation 12(8), 1889–1900 (2000)

[22] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., Duchesnay, E.: Scikit-learn: Machine learning in Python. Journal of Machine Learning Research 12, 2825–2830 (2011)

[23] Frazier, P.I.: A tutorial on bayesian optimization. arXiv preprint arXiv:1807.02811 (2018)

[24] James, F., Roos, M.: Minuit: A system for function minimization and analysis of the parameter errors and correlations. Computer Physics Communications 10(6), 343–367 (1975) https://doi.org/10.1016/0010-4655(75)90039-9

[25] Fixsen, D.J., Cheng, E.S., Gales, J.M., Mather, J.C., Shafer, R.A., Wright, E.L.: The cosmic microwave background spectrum from the full COBE FIRAS data set. The Astrophysical Journal 473, 576–587 (1996)