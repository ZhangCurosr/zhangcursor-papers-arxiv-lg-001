# SHARP MIXED SPECTRAL BARRON REGULARITY OF COULOMBIC MANY-ELECTRON WAVE FUNCTIONS

PINGBING MING AND HAO YU

Abstract. We establish sharp mixed spectral Barron regularity for eigenfunctions of molecular Coulomb Hamiltonians. The mixed norm is a Fourier $L ^ { 1 }$ norm with one isotropic weight and coordinate-product weights, and therefore detects regularity invisible to the isotropic Barron scale. For a nonempty set I of electron indices on which the wave function is antisymmetric, we derive an explicit admissible region for the isotropic order s and the coordinate orders $\alpha , \beta .$ . This region is optimal as a uniform statement over the class of clamped-nuclei Coulomb Hamiltonians. For fixed-spin components with two occupied spin blocks, it reduces to $s + \alpha + \beta < 1 ;$ in the fully spin-polarized class it reduces to $s + \alpha < 1$ . In particular, if $\mathcal { T } _ { \sigma }$ denotes the family of occupied same-spin blocks determined by $\sigma ,$ then every fixed-spin spatial component $\psi _ { \sigma }$ satisfies, for every $0 \leq \alpha < 1$

$$
\left( \sum _ { I \in \mathcal { Z } _ { \sigma } } \prod _ { i \in I } \langle \xi _ { i } \rangle ^ { \alpha } \right) \widehat { \psi _ { \sigma } } \in L ^ { 1 } ( \mathbb { R } ^ { 3 N } ) .
$$

For a fully spin-polarized state, $\mathcal { T } _ { \sigma } = \{ \{ 1 , . . . , N \} \}$

## 1. Introduction

Let $N \geq 1$ electrons move in the field of $L \geq 1$ clamped nuclei with positive charges $Z _ { 1 } , \dots , Z _ { L }$ at distinct positions $R _ { 1 } , \ldots , R _ { L } \in \mathbb { R } ^ { 3 }$ . In atomic units, the Hamiltonian is

$$
H = - \frac { 1 } { 2 } \sum _ { i = 1 } ^ { N } \Delta _ { x _ { i } } + V , \qquad V ( x ) = - \sum _ { i = 1 } ^ { N } \sum _ { \nu = 1 } ^ { L } \frac { Z _ { \nu } } { | x _ { i } - R _ { \nu } | } + \sum _ { 1 \leq i < j \leq N } \frac { 1 } { | x _ { i } - x _ { j } | } .\tag{1.1}
$$

The operator H denotes the self-adjoint Friedrichs realization of the diferential expression in (1.1). The Coulomb singularities occur in three-dimensional collision variables even though the configuration space has dimension 3N. This structure is responsible for the mixed regularity of electronic wave functions and, at the same time, for the failure of direct isotropic estimates to describe all available smoothness.

The classical mixed-Sobolev theory began with Yserentant’s work and its subsequent refinements. Kreusler and Yserentant proved that general electronic eigenfunctions have fractional product regularity of every order below $3 / 4 \ [ 5$ , Theorem 5.1]; their model in Section 6 shows that this threshold is sharp in that scale. Yserentant proved full first-order mixed-Sobolev regularity for an explicitly cusp-regularized eigenfunction [11, Theorem 6.6]. More recently, Meng used the Pauli principle to prove sharper spin-dependent mixed-Sobolev estimates [6].

Spectral Barron regularity is a diferent, Fourier $L ^ { 1 }$ notion. It controls absolute Fourier moments rather than square-integrable derivatives. Yserentant proved that Coulombic electronic eigenfunctions belong to the isotropic spectral Barron space $B ^ { s }$ for every $s < 1$ [12, Theorem 4.7]; the hydrogen ground state shows that $s = 1$ cannot be reached in general. A general Fourier–Lebesgue theory for many-particle Schrödinger eigenfunctions was developed in our earlier work [7, Section 2.2]. The present paper refines these isotropic conclusions by retaining coordinate-product weights and by using cancellation forced by the Pauli principle.

Our main contributions are as follows.

(i) We introduce mixed spectral Barron spaces with an isotropic weight and coordinateproduct weights. We prove index-set-dependent mixed Barron regularity for Coulombic eigenfunctions and, for fixed-spin components, simultaneous regularity with respect to every occupied spin block.

(ii) We prove that the resulting mixed-regularity regions are uniformly sharp over the class of clamped-nuclei Coulomb Hamiltonians. Hydrogen, helium, and lithium ground states show that none of the defining inequalities of these parameter regions can be relaxed uniformly over this class.

The paper is organized as follows. Section 2 defines the spaces and states the main results. Section 3 proves the weighted Coulomb estimates. Section 4 derives the mixed regularity gain of the resolvent. Section 5 completes the proof by a contraction argument. Section 6 contains the proofs of the sharpness results. Section 7 summarizes the results and identifies directions for further work. Appendix A contains the auxiliary space-comparison results and their proofs.

## 2. Setting and main results

For $f \in { \mathcal { S } } ( \mathbb { R } ^ { d } )$ , we use the unitary Fourier transform

$$
{ \widehat { f } } ( \xi ) = ( 2 \pi ) ^ { - d / 2 } \int _ { \mathbb { R } ^ { d } } e ^ { - i x \cdot \xi } f ( x ) \ \mathrm { d } x
$$

and extend it to $S ^ { \prime } ( \mathbb { R } ^ { d } )$ by duality. Write $\langle z \rangle = ( 1 + | z | ^ { 2 } ) ^ { 1 / 2 }$ . For $\xi = ( \xi _ { 1 } , \dots , \xi _ { N } ) \in (  { \mathbb { R } } ^ { 3 } ) ^ { N }$ and a nonempty set $I \subseteq \{ 1 , \ldots , N \}$ , define

$$
w _ { s , I } ^ { \alpha , \beta } ( \boldsymbol { \xi } ) = \langle \boldsymbol { \xi } \rangle ^ { s } \prod _ { i \in I } \langle \xi _ { i } \rangle ^ { \alpha } \prod _ { j \notin I } \langle \xi _ { j } \rangle ^ { \beta } .\tag{2.1}
$$

Definition 2.1 (Mixed spectral Barron space). For $s , \alpha , \beta \geq 0$ , define

$$
\begin{array} { r } { \mathcal { B } _ { I ; \alpha , \beta } ^ { s } ( \mathbb { R } ^ { 3 N } ) : = \left\{ f \in \mathcal { S } ^ { \prime } ( \mathbb { R } ^ { 3 N } ) \Big | w _ { s , I } ^ { \alpha , \beta } \widehat { f } \in L ^ { 1 } ( \mathbb { R } ^ { 3 N } ) \right\} , } \end{array}
$$

equipped with the norm

$$
\| f \| _ { \mathcal B _ { I ; \alpha , \beta } ^ { s } } : = \int _ {  { \mathbb { R } } ^ { 3 N } } w _ { s , I } ^ { \alpha , \beta } ( \boldsymbol { \xi } ) | \widehat { f } ( \boldsymbol { \xi } ) | \mathrm { ~ d } \boldsymbol { \xi } .
$$

For $\alpha = \beta = 0$ , this is the usual spectral Barron space $B ^ { s }$ . Unlike a single isotropic power, the product weights record simultaneous growth in several electron momenta.

We say that u is antisymmetric with respect to I if

$$
u ( \ldots , x _ { i } , \ldots , x _ { j } , \ldots ) = - u ( \ldots , x _ { j } , \ldots , x _ { i } , \ldots ) \quad ( i , j \in I , \ i \neq j ) .
$$

For $| I | = 1$ this condition is void.

Theorem 2.2 (Index-set-dependent mixed Barron regularity). Let $\psi \in H ^ { 1 } ( \mathbb { R } ^ { 3 N } ) \ \backslash$ {0} satisfy $H \psi = E \psi$ for some $E \in \mathbb { R }$ , and suppose that $\psi$ is antisymmetric with respect to a nonempty set $I \subseteq \{ 1 , \ldots , N \}$ . Let $s , \alpha , \beta \geq 0$ . If

$$
\left\{ \begin{array} { l l } { s + \alpha < 1 , } & { I ^ { c } = \emptyset , } \\ { s + \alpha + \beta < 1 , } & { | I ^ { c } | = 1 , } \\ { s + \operatorname* { m a x } \{ \alpha + \beta , 2 \beta \} < 1 , } & { | I ^ { c } | \geq 2 . } \end{array} \right.\tag{2.2}
$$

Then $\psi \in B _ { I ; \alpha , \beta } ^ { s } ( \mathbb { R } ^ { 3 N } )$

The family of parameter regions (2.2), with N and the nonempty set I allowed to vary, is uniformly optimal over the class of clamped-nuclei Coulomb Hamiltonians; see Corollary 2.8.

For the spin formulation, let $q \in \mathbb { N }$ be the number of spin states, let $\Psi ( x , \sigma )$ be a fermionic electronic eigenfunction, and fix $\sigma = ( \sigma _ { 1 } , \ldots , \sigma _ { N } ) \in \{ 1 , \ldots , q \} ^ { N }$ . Define

$$
\psi _ { \sigma } ( x ) = \Psi ( x , \sigma ) , \qquad I _ { \ell } ( \sigma ) = \{ i \mid \sigma _ { i } = \ell \} , \qquad \mathcal { Z } _ { \sigma } = \{ I _ { \ell } ( \sigma ) \mid I _ { \ell } ( \sigma ) \neq \emptyset \} .
$$

For distinct $i , j ,$ , let $U _ { i j }$ exchange $x _ { i }$ and $x _ { j }$ . If $i , j \in I _ { \ell } ( \sigma )$ , the corresponding spin transposition leaves $\sigma$ unchanged, and the Pauli principle gives

$$
U _ { i j } \psi _ { \sigma } = - \psi _ { \sigma } \qquad ( i , j \in I _ { \ell } ( \sigma ) , i \ne j ) .\tag{2.3}
$$

Thus $\psi _ { \sigma }$ is antisymmetric with respect to every occupied spin block $I \in { \mathcal { T } } _ { \sigma } .$ ; see also [6, $( 1 . 8 ) \ – ( 1 . 9 )$ and Remark 1.3].

Theorem 2.3 (Spin-partition refinement). Let $\psi = \psi _ { \sigma }$ be a fixed-spin spatial component of an electronic eigenfunction, and let $s , \alpha , \beta \geq 0$ . If

$$
\left\{ \begin{array} { l l } { s + \alpha < 1 , } & { | \mathcal { T } _ { \sigma } | = 1 , } \\ { s + \alpha + \beta < 1 , } & { | \mathcal { T } _ { \sigma } | = 2 , } \\ { s + \operatorname* { m a x } \{ \alpha + \beta , 2 \beta \} < 1 , } & { | \mathcal { T } _ { \sigma } | \geq 3 . } \end{array} \right.\tag{2.4}
$$

Then

$$
\psi \in \bigcap _ { I \in \mathcal { T } _ { \sigma } } B _ { I ; \alpha , \beta } ^ { s } ( \mathbb { R } ^ { 3 N } ) .\tag{2.5}
$$

For two spin states, Lemma A.1 shows that, at fixed total order $s + \alpha + \beta < 1$ , the conclusion of Theorem 2.3 is strongest when $s = \beta = 0$ . This choice yields

Corollary 2.4 (Electronic wave functions). $L e t \psi = \psi _ { \sigma }$ be a fixed-spin spatial component of an electronic eigenfunction. For every $0 \leq \alpha < 1$ ，

$$
\int _ { \mathbb { R } ^ { 3 N } } \left( \sum _ { I \in \mathcal { T } _ { \sigma } } \prod _ { i \in I } \langle \xi _ { i } \rangle ^ { \alpha } \right) | \widehat \psi ( \xi ) | \ \mathrm { d } \xi < \infty .\tag{2.6}
$$

If the state is fully spin-polarized, the sum in (2.6) consists of the single product over all electron coordinates.

The above corollary expresses the Pauli gain as simultaneous product-weighted Fourier integrability over each same-spin block. In a fully polarized state, every electron pair benefits from the antisymmetric diference-kernel estimate.

For $s , \alpha , \beta \geq 0$ , define

$$
\chi _ { \sigma } ^ { s , \alpha , \beta } = \bigcap _ { I \in \mathcal { I } _ { \sigma } } \mathcal { B } _ { I ; \alpha , \beta } ^ { s } ( \mathbb { R } ^ { 3 N } ) , \qquad \| f \| _ { \mathcal { X } _ { \sigma } ^ { s , \alpha , \beta } } = \sum _ { I \in \mathcal { I } _ { \sigma } } \| f \| _ { \mathcal { B } _ { I ; \alpha , \beta } ^ { s } } .\tag{2.7}
$$

The following lemma gives the sharp comparison with isotropic spectral Barron spaces.   
Its proof is deferred to Appendix A.

Lemma 2.5 (Sharp comparison with isotropic spectral Barron spaces). Let $\begin{array} { r l } { n _ { \sigma } } & { { } = } \end{array}$ $\operatorname* { m a x } _ { I \in { \mathcal { I } } _ { \sigma } } | I |$ and $0 \leq \alpha < 1$ . Then, for every $t \geq 0$

$$
\begin{array} { r l } & { \mathcal { B } ^ { t } ( \mathbb { R } ^ { 3 N } ) \hookrightarrow \mathcal { X } _ { \sigma } ^ { 0 , \alpha , 0 } \Longleftrightarrow t \geq \alpha n _ { \sigma } , } \\ & { \mathcal { X } _ { \sigma } ^ { 0 , \alpha , 0 } \hookrightarrow \mathcal { B } ^ { t } ( \mathbb { R } ^ { 3 N } ) \Longleftrightarrow t \leq \alpha . } \end{array}\tag{2.8}
$$

In particular,

$$
B ^ { \alpha n _ { \sigma } } ( \mathbb { R } ^ { 3 N } ) \hookrightarrow \mathcal { X } _ { \sigma } ^ { 0 , \alpha , 0 } \hookrightarrow B ^ { \alpha } ( \mathbb { R } ^ { 3 N } ) .
$$

If $\alpha > 0$ and $n _ { \sigma } > 1$ , both inclusions are strict. ${ \cal I } f \alpha = 0 \ o r \ n _ { \sigma } = 1$ , then $\mathcal { X } _ { \sigma } ^ { 0 , \alpha , 0 } = \mathcal { B } ^ { \alpha } ( \mathbb { R } ^ { 3 N } )$ with equivalent norms.

By Corollary 2.4 and the second embedding in (2.8), $\psi \in B ^ { \alpha } ( \mathbb { R } ^ { 3 N } )$ for every $0 \leq \alpha < 1$ recovering the sharp isotropic Barron regularity established in [12, Theorem 4.7] and [7, Corollary 2.7] without a dimension-dependent constant. If $0 < \alpha < 1$ and an occupied spin block contains at least two electrons, Lemma 2.5 shows that the mixed product-moment conclusion is strictly stronger than ordinary isotropic Barron regularity.

2.1. Mixed Fourier–Lebesgue interpolation. We combine the Fourier $L ^ { 1 }$ estimates with Meng’s mixed $L ^ { 2 }$ regularity to obtain an interpolated Fourier–Lebesgue scale. Meng’s fixed-spin result gives mixed-Sobolev regularity under $\alpha < 5 / 4 , \beta < 3 / 4$ , and $\alpha + \beta < 3 / 2$ [6, Corollary 2.4]. The two estimates encode diferent integrability and decay properties. Since both hold for the same eigenfunction, interpolation of their weighted Fourier norms yields the following intermediate regularity for every $1 < p < 2$

Corollary 2.6 (Mixed Fourier–Lebesgue interpolation). Let $\psi = \psi _ { \sigma }$ be a fixed-spin spatial component of an electronic eigenfunction and fix $I \in \mathcal { T } _ { \sigma }$ . Let $s _ { 0 } , \alpha _ { 0 } , \beta _ { 0 } \geq 0$ satisfy the applicable condition in (2.4), with $s , \alpha , \beta$ replaced by $s _ { 0 } , \alpha _ { 0 } , \beta _ { 0 }$ . Let $0 \leq \alpha _ { 2 } < 5 / 4$ and $0 \leq \beta _ { 2 } < 3 / 4$ satisfy $\alpha _ { 2 } + \beta _ { 2 } < 3 / 2$ . For $1 < p < 2$ , set $\theta = 2 ( 1 - 1 / p )$ and

$$
s _ { \theta } = ( 1 - \theta ) s _ { 0 } + \theta , \qquad \alpha _ { \theta } = ( 1 - \theta ) \alpha _ { 0 } + \theta \alpha _ { 2 } , \qquad \beta _ { \theta } = ( 1 - \theta ) \beta _ { 0 } + \theta \beta _ { 2 } .
$$

Then

$$
w _ { s _ { \theta } , I } ^ { \alpha _ { \theta } , \beta _ { \theta } } \hat { \psi } \in L ^ { p } ( \mathbb { R } ^ { 3 N } ) .
$$

The proof is deferred to the end of Section 5.

2.2. Sharpness of the main results. Hydrogen, helium, and lithium ground states prove uniform sharpness in the clamped-nuclei Coulomb class. The proofs are given in Section 6. Let $\Phi _ { N , Z }$ denote the positive scalar ground state of the N-electron atom with nuclear charge Z. Such a ground state exists for $N < Z + 1$ by Zhislin’s binding theorem [13]; see also [10, Corollary 11.10].

Proposition 2.7 (Atomic endpoint tests). Let $s , \alpha , \beta \geq 0$

(i) For $N = 1$ and $I = \{ 1 \}$ , the hydrogen ground state $\psi _ { \mathrm { H } } ( x ) = e ^ { - | x | }$ satisfies

$$
\begin{array} { r } { \psi _ { \mathrm { H } } \in \mathcal { B } _ { I ; \alpha , \beta } ^ { s } ( \mathbb { R } ^ { 3 } ) \quad \Longleftrightarrow \quad s + \alpha < 1 . } \end{array}
$$

(ii) For the helium scalar ground state and $I = \{ 1 \}$

$$
\Phi _ { 2 , 2 } \in \mathcal { B } _ { I ; \alpha , \beta } ^ { s } ( \mathbb { R } ^ { 6 } ) \quad \Longleftrightarrow \quad s + \alpha + \beta < 1 .
$$

(iii) For the lithium scalar ground state and $I = \{ 1 \}$

$$
\Phi _ { 3 , 3 } \in \mathcal { B } _ { I ; \alpha , \beta } ^ { s } ( \mathbb { R } ^ { 9 } ) \quad \Longleftrightarrow \quad s + \operatorname* { m a x } \{ \alpha + \beta , 2 \beta \} < 1 .
$$

Corollary 2.8 (Uniform sharpness of the index-set family). The three parameter regions in (2.2) are uniformly sharp in $N$ , the nonempty index set I, and the clamped-nuclei Coulomb Hamiltonian (1.1).

The same examples prove uniform sharpness of the spin-dependent conclusions. Multiplying the helium spatial ground state by the spin singlet gives a physical fermionic state with one electron in each of the two occupied spin blocks, so the case $| \mathcal { T } _ { \sigma } | = 2$ in (2.4) is sharp. Hydrogen makes $s + \alpha < 1$ sharp in the fully spin-polarized class when N is allowed to vary. If $q \geq 3$ , multiplying the lithium spatial ground state by a totally antisymmetric three-spin factor and taking a fixed-spin component with three distinct occupied spin states realizes both the $\alpha + \beta$ and 2β faces.

## 3. Weighted Coulomb operators

The Fourier transform of $| x | ^ { - 1 }$ in three dimensions is a positive multiple of $| \xi | ^ { - 2 }$ Nuclear terms therefore lead to a scalar Riesz potential, while an electron–electron term translates two momentum variables in opposite directions. We first derive sharp weighted $L ^ { 1 }$ bounds for the scalar and ordinary pair operators, and then exploit antisymmetric cancellation to improve the pair threshold. These estimates provide the convolution bounds used in Section 4 to obtain a positive resolvent gain.

3.1. Scalar and ordinary pair estimates. This subsection computes the exact weighted $L ^ { 1 }$ norms of the nuclear convolution and the ordinary electron–pair translation operator. Their endpoint failures determine the restrictions used for nuclear and non-antisymmetric pair terms in Section 4. For $f \in L ^ { 1 } ( \mathbb { R } ^ { 3 } )$ , define

$$
( I _ { 2 } f ) ( x ) \colon = \int _ { \mathbb { R } ^ { 3 } } { \frac { f ( y ) } { | x - y | ^ { 2 } } } \operatorname { d } y .
$$

Proposition 3.1 (Scalar Coulomb convolution). For $\tau ~ < ~ - 1$ , the exact norm of $I _ { 2 } \colon \bar { L } ^ { 1 } ( \mathbb { R } ^ { 3 } )  L ^ { 1 } ( \mathbb { R } ^ { 3 } , \langle x \rangle ^ { \tau } \mathrm { d } x )$ is

$$
2 \pi ^ { 3 / 2 } \frac { \Gamma ( ( - \tau - 1 ) / 2 ) } { \Gamma ( - \tau / 2 ) } .
$$

The mapping is unbounded when $\tau \geq - 1$

Proof. Set $w _ { \tau } ( x ) = \langle x \rangle ^ { \tau }$ and

$$
\Lambda _ { \tau } ( y ) = \int _ { \mathbb { R } ^ { 3 } } \frac { w _ { \tau } ( x ) } { | x - y | ^ { 2 } } \mathrm { ~ d } x .
$$

For $f \in L ^ { 1 } ( \mathbb { R } ^ { 3 } )$ , positivity of the kernel and Tonelli’s theorem give

$$
\begin{array} { r l } & { \displaystyle \| I _ { 2 } f \| _ { L ^ { 1 } ( w _ { \tau } \mathrm { d } x ) } \leq \int _ { \mathbb R ^ { 3 } } w _ { \tau } ( x ) \int _ { \mathbb R ^ { 3 } } \frac { | f ( y ) | } { | x - y | ^ { 2 } } \mathrm { d } y \mathrm { d } x } \\ & { \qquad = \int _ { \mathbb R ^ { 3 } } | f ( y ) | \Lambda _ { \tau } ( y ) \mathrm { d } y \leq ( \exp \Lambda _ { \tau } ( y ) ) \| f \| _ { L ^ { 1 } } . } \end{array}
$$

The functions $w _ { \tau }$ and $| \cdot | ^ { - 2 }$ are nonnegative, radial, radially nonincreasing, and vanish at infinity. The Brascamp–Lieb–Luttinger rearrangement inequality [1, Theorem 3.4, p. 233; see also Definition 3.3, p. 232] therefore gives, for every $\boldsymbol { y } \in \mathbb { R } ^ { 3 }$ 2

$$
\Lambda _ { \tau } ( y ) = \int _ { \mathbb { R } ^ { 3 } } \frac { w _ { \tau } ( x ) } { | x - y | ^ { 2 } } ~ \mathrm { d } x \leq \int _ { \mathbb { R } ^ { 3 } } \frac { w _ { \tau } ( x ) } { | x | ^ { 2 } } ~ \mathrm { d } x = \Lambda _ { \tau } ( 0 ) < \infty ,
$$

where finiteness follows from $\tau < - 1$ . The function $\Lambda _ { \tau }$ is continuous. Hence, for every $\varepsilon > 0$ , there is a ball $B _ { \varepsilon }$ centered at the origin on which $\Lambda _ { \tau } ( y ) > \Lambda _ { \tau } ( 0 ) - \varepsilon$ . Taking $f = | B _ { \varepsilon } | ^ { - 1 } \mathbf { 1 } _ { B _ { \varepsilon } }$ and using Tonelli once more gives

$$
\| I _ { 2 } f \| _ { L ^ { 1 } ( w _ { \tau } \mathrm { d } x ) } = \frac { 1 } { | B _ { \varepsilon } | } \int _ { B _ { \varepsilon } } \Lambda _ { \tau } ( y ) \mathrm { d } y > \Lambda _ { \tau } ( 0 ) - \varepsilon .
$$

Together with the upper bound, and then letting $\varepsilon \downarrow 0$ , this proves

$$
\| I _ { 2 } \| = \cos \operatorname* { s u p } _ { y \in \mathbb { R } ^ { 3 } } \Lambda _ { \tau } ( y ) = \Lambda _ { \tau } ( 0 ) .
$$

Using spherical coordinates, the substitution $v = r ^ { 2 }$ , and Euler’s beta integral, we obtain

$$
\Lambda _ { \tau } ( 0 ) = \int _ { \mathbb { R } ^ { 3 } } \frac { \langle x \rangle ^ { \tau } } { | x | ^ { 2 } } ~ \mathrm { d } x = 4 \pi \int _ { 0 } ^ { \infty } ( 1 + r ^ { 2 } ) ^ { \tau / 2 } ~ \mathrm { d } r = 2 \pi ^ { 3 / 2 } \frac { \Gamma ( ( - \tau - 1 ) / 2 ) } { \Gamma ( - \tau / 2 ) } .
$$

For $\tau \geq - 1$ , choose $R > 0$ and a nonnegative $f \in L ^ { 1 } ( \mathbb { R } ^ { 3 } )$ supported in $B _ { R }$ with $\begin{array} { r } { \int _ { \mathbb { R } ^ { 3 } } f ( y ) \ \mathrm { d } y = 1 } \end{array}$ . If $| x | \geq 2 R .$ , then $| x - y | \leq | x | + R \leq 3 | x | / 2$ for $y \in \operatorname { s u p p } { f } .$ and therefore

$$
( I _ { 2 } f ) ( x ) \geq { \frac { 4 } { 9 | x | ^ { 2 } } } .
$$

Consequently,

$$
\| I _ { 2 } f \| _ { L ^ { 1 } ( w _ { \tau } \mathrm { d } x ) } \geq \frac { 4 } { 9 } \int _ { | x | \geq 2 R } \frac { \langle x \rangle ^ { \tau } } { | x | ^ { 2 } } \ \mathrm { d } x = \frac { 1 6 \pi } { 9 } \int _ { 2 R } ^ { \infty } ( 1 + r ^ { 2 } ) ^ { \tau / 2 } \ \mathrm { d } r = \infty
$$

whenever $\tau \geq - 1$ , which proves the asserted unboundedness.

For integrable functions defined on $\mathbb { R } ^ { 3 } \times \mathbb { R } ^ { 3 }$ , define the pair translation operator

$$
( T f ) ( x , y ) \colon = \int _ { \mathbb { R } ^ { 3 } } { \frac { f ( x - k , y + k ) } { | k | ^ { 2 } } } \operatorname { d } k .
$$

Proposition 3.2 (Ordinary pair operator). For $t < - 1$ , the exact norm of

$$
T \colon L ^ { 1 } ( \mathbb { R } ^ { 6 } ) \longrightarrow L ^ { 1 } ( \mathbb { R } ^ { 6 } , \langle ( x , y ) \rangle ^ { t } \mathrm { d } x \mathrm { d } y )
$$

is $2 ^ { - 1 / 2 }$ times the scalar norm in Proposition 3.1, namely

$$
C _ { t } ^ { \mathrm { o } } = \sqrt { 2 } \pi ^ { 3 / 2 } \frac { \Gamma ( ( - t - 1 ) / 2 ) } { \Gamma ( - t / 2 ) } .
$$

The mapping is unbounded when $t \geq - 1$

Proof. For $t \in \mathbb { R }$ and $( u , v ) \in \mathbb { R } ^ { 6 }$ , set

$$
Q _ { t } ( u , v ) = \int _ { \mathbb { R } ^ { 3 } } \langle ( u + k , v - k ) \rangle ^ { t } \frac { \mathrm { d } k } { | k | ^ { 2 } } \in ( 0 , \infty ] .
$$

Suppose first that $t < - 1$ . Tonelli’s theorem gives

$$
\| T f \| _ { L ^ { 1 } ( \langle ( x , y ) \rangle ^ { t } \mathrm { d } x \mathrm { d } y ) } \leq \int _ { \mathbb { R } ^ { 6 } } \lvert f ( u , v ) \rvert Q _ { t } ( u , v ) \mathrm { d } u \mathrm { d } v .
$$

Equality holds here for every nonnegative $f .$ . Testing with normalized characteristic functions of finite-measure subsets of

$$
\{ ( u , v ) \mid Q _ { t } ( u , v ) > \mathrm { e s s } \operatorname* { s u p } Q _ { t } - \varepsilon \}
$$

and then letting $\varepsilon \downarrow 0$ proves

$$
\| T \| = \operatorname { \mathrm { e s s } \operatorname* { s u p } _ { ( u , v ) \in \mathbb { R } ^ { 6 } } Q _ { t } ( u , v ) . }
$$

Put $\begin{array} { r } { \lambda = 1 + \frac { 1 } { 2 } | u + v | ^ { 2 } } \end{array}$ and $b = ( u - v ) / 2$ . Since

$$
| { u + k } | ^ { 2 } + | { v - k } | ^ { 2 } = 2 | { k + b } | ^ { 2 } + \frac { 1 } { 2 } | { u + v } | ^ { 2 } ,
$$

the change of variables $q = \sqrt { 2 / \lambda } ( k + b )$ gives

$$
Q _ { t } ( u , v ) = { \frac { \lambda ^ { ( t + 1 ) / 2 } } { \sqrt { 2 } } } \Lambda _ { t } \Biggl ( \sqrt { \frac { 2 } { \lambda } } b \Biggr ) ,
$$

where $\Lambda _ { t }$ is the scalar convolution weight used in the proof of Proposition 3.1. That proof shows that $\Lambda _ { t }$ is continuous and $\Lambda _ { t } ( z ) \leq \Lambda _ { t } ( 0 )$ for every $z \in \mathbb { R } ^ { 3 }$ . Since $t < - 1$ and $\lambda \geq 1$

$$
Q _ { t } ( u , v ) \leq \frac { 1 } { \sqrt { 2 } } \Lambda _ { t } ( 0 ) = Q _ { t } ( 0 , 0 ) .
$$

The displayed representation makes $Q _ { t }$ continuous, so its pointwise maximum is also its essential supremum. Proposition 3.1 now gives

$$
\| T \| = \frac { 1 } { \sqrt { 2 } } \Lambda _ { t } ( 0 ) = \sqrt { 2 } \pi ^ { 3 / 2 } \frac { \Gamma ( ( - t - 1 ) / 2 ) } { \Gamma ( - t / 2 ) } = C _ { t } ^ { 0 } .
$$

Suppose $t \geq - 1$ . For each fixed $( u , v )$ , there are $R > 0$ and $c > 0$ such that

$$
\langle ( u + k , v - k ) \rangle ^ { t } \geq c | k | ^ { t } , \qquad | k | \geq R .
$$

Thus

$$
Q _ { t } ( u , v ) \geq 4 \pi c \int _ { R } ^ { \infty } r ^ { t } \ \mathrm { d } r = \infty .
$$

Applying Tonelli’s theorem to any nonzero nonnegative $f \in L ^ { 1 } ( \mathbb { R } ^ { 6 } )$ proves that $T$ is unbounded. □

3.2. The antisymmetric gain. Compared with Proposition 3.2, the next result extends the admissible target exponent from $t < - 1$ to $t < 0$ , thereby gaining one full order from antisymmetry. We first derive the weighted $L ^ { 1 }$ estimate from the diference-kernel representation of the pair operator and then record its fiberwise form for use in Section 4.

Proposition 3.3 (Antisymmetric pair operator). Let $s , a \geq 0$ and − $\ - 1 \leq t < 0$ . If

$$
s + a \geq t + 1 ,
$$

then, on functions satisfying $f ( x , y ) = - f ( y , x )$

$$
\int _ { \mathbb { R } ^ { 6 } } \langle ( x , y ) \rangle ^ { t } | T f ( x , y ) | \mathrm { ~ d } x \mathrm { d } y \leq C _ { t } ^ { \mathrm { a } } \int _ { \mathbb { R } ^ { 6 } } \langle ( x , y ) \rangle ^ { s } ( \langle x \rangle \langle y \rangle ) ^ { a } | f ( x , y ) | \mathrm { ~ d } x \mathrm { d } y ,\tag{3.1}
$$

where

$$
C _ { t } ^ { \mathrm { a } } = - { \frac { 2 ^ { t / 2 + 1 } \pi ^ { 2 } } { t + 2 } } \cot \left( { \frac { \pi t } { 4 } } \right) .
$$

Proof. It sufices first to consider antisymmetric $f \in C _ { c } ^ { \infty } ( \mathbb { R } ^ { 6 } )$ . The space $C _ { c } ^ { \infty } ( \mathbb { R } ^ { 6 } )$ is dense in the weighted input space, and the antisymmetrization

$$
( P _ { - } g ) ( x , y ) = \frac { 1 } { 2 } \big ( g ( x , y ) - g ( y , x ) \big )
$$

is contractive because the input weight is invariant under $x  y$ . The estimate below therefore extends to every antisymmetric input by density.

Put $d _ { x y } = x - y$ . Substituting $k \mapsto d _ { x y } - k$ and using antisymmetry give

$$
\begin{array} { l } { \displaystyle { T f ( x , y ) = - \int _ { \mathbb { R } ^ { 3 } } \frac { f ( x - k , y + k ) } { | k - d _ { x y } | ^ { 2 } } \mathrm { d } k } } \\ { \displaystyle { = \frac { 1 } { 2 } \int _ { \mathbb { R } ^ { 3 } } f ( x - k , y + k ) \left( \frac { 1 } { | k | ^ { 2 } } - \frac { 1 } { | k - d _ { x y } | ^ { 2 } } \right) \mathrm { d } k } . } \end{array}
$$

Set $u = x - k , v = y + k , X = u - v , Y = u + v$ , and $z = k + X / 2$ . Then

$$
x = z + { \frac { Y } { 2 } } , \qquad y = { \frac { Y } { 2 } } - z , \qquad \left. z + { \frac { Y } { 2 } } , { \frac { Y } { 2 } } - z \right. ^ { 2 } = 1 + 2 | z | ^ { 2 } + { \frac { 1 } { 2 } } | Y | ^ { 2 } .
$$

Since $t < 0$

$$
\left. z + \frac { Y } { 2 } , \frac { Y } { 2 } - z \right. ^ { t } \leq ( 1 + 2 | z | ^ { 2 } ) ^ { t / 2 } .
$$

Using this inequality and then applying Tonelli’s theorem to the nonnegative integrand gives

$$
\int _ { \mathbb { R } ^ { 6 } } \langle ( x , y ) \rangle ^ { t } | T f ( x , y ) | \mathrm { d } x \mathrm { d } y \leq \frac { 1 } { 2 } \int _ { \mathbb { R } ^ { 6 } } | f ( u , v ) | J _ { t } \bigg ( \frac { u - v } { 2 } \bigg ) \mathrm { d } u \mathrm { d } v ,
$$

where

$$
J _ { t } ( W ) = \int _ { \mathbb { R } ^ { 3 } } ( 1 + 2 | z | ^ { 2 } ) ^ { t / 2 } \left| \frac { 1 } { | z - W | ^ { 2 } } - \frac { 1 } { | z + W | ^ { 2 } } \right| \mathrm { d } z .
$$

Suppose $W \neq 0$ . With $e = W / | W |$ , substitute $z = | W | r \omega$ , where $r > 0$ and $\omega \in \mathbb { S } ^ { 2 }$ Since $t < 0$

$$
J _ { t } ( W ) \leq 2 ^ { t / 2 } | W | ^ { t + 1 } \int _ { 0 } ^ { \infty } r ^ { t + 2 } \int _ { \mathbb { S } ^ { 2 } } \left| \frac { 1 } { | r \omega - e | ^ { 2 } } - \frac { 1 } { | r \omega + e | ^ { 2 } } \right| \mathrm { d } \omega \mathrm { d } r .
$$

Direct integration in the polar variable $\omega \cdot e$ yields

$$
\int _ { \mathbb { S } ^ { 2 } } \left| \frac 1 { | r \omega - e | ^ { 2 } } - \frac 1 { | r \omega + e | ^ { 2 } } \right| \mathrm { d } \omega = \frac { 4 \pi } { r } \log \left| \frac { r ^ { 2 } + 1 } { r ^ { 2 } - 1 } \right| .
$$

Hence

$$
J _ { t } ( W ) \le 2 ^ { t / 2 + 2 } \pi \Theta _ { t } | W | ^ { t + 1 } ,
$$

with

$$
\Theta _ { t } = \int _ { 0 } ^ { \infty } r ^ { t + 1 } \log \left| \frac { r ^ { 2 } + 1 } { r ^ { 2 } - 1 } \right| \mathrm { d } r .
$$

For $- 1 \leq t < 0$ , the substitution $\varrho = r ^ { 2 } .$ inversion on $( 1 , \infty )$ , and the nonnegative expansion

$$
\log { \frac { 1 + \varrho } { 1 - \varrho } } = 2 \sum _ { m = 0 } ^ { \infty } \frac { \varrho ^ { 2 m + 1 } } { 2 m + 1 } , \qquad 0 < \varrho < 1 ,
$$

give

$$
\begin{array} { l } { { \Theta _ { t } = \displaystyle \sum _ { m = 0 } ^ { \infty } \frac { 1 } { 2 m + 1 } \left( \frac { 1 } { 2 m + t / 2 + 2 } + \frac { 1 } { 2 m - t / 2 } \right) } } \\ { { \displaystyle ~ = \frac { 1 } { t + 2 } \sum _ { m = 0 } ^ { \infty } \left( \frac { 1 } { m - t / 4 } - \frac { 1 } { m + 1 + t / 4 } \right) } } \\ { { \displaystyle ~ = - \frac { \pi } { t + 2 } \cot ( \frac { \pi t } { 4 } ) . } } \end{array}
$$

The last equality follows from

$$
\sum _ { m = 0 } ^ { \infty } \left( { \frac { 1 } { m + \theta } } - { \frac { 1 } { m + 1 - \theta } } \right) = \pi \cot ( \pi \theta ) , \qquad 0 < \theta < 1 ,
$$

evaluated at $\theta = - t / 4$ . Thus

$$
2 ^ { t / 2 + 2 } \pi \Theta _ { t } = - \frac { 2 ^ { t / 2 + 2 } \pi ^ { 2 } } { t + 2 } \cot \biggl ( \frac { \pi t } { 4 } \biggr ) = 2 C _ { t } ^ { \mathrm { a } } .
$$

If $W = 0 ,$ , then $J _ { t } ( W ) = 0$

Since

$$
\langle ( u , v ) \rangle ^ { 2 } = 1 + 2 | W | ^ { 2 } + \frac 1 2 | Y | ^ { 2 } , \qquad \langle u \rangle ^ { 2 } \langle v \rangle ^ { 2 } \geq \langle ( u , v ) \rangle ^ { 2 } ,
$$

we have, for $W \neq 0$

$$
\begin{array} { r } { | W | ^ { t + 1 } \leq ( 1 + 2 | W | ^ { 2 } ) ^ { ( t + 1 ) / 2 } \leq ( 1 + 2 | W | ^ { 2 } ) ^ { ( s + a ) / 2 } \leq \langle ( u , v ) \rangle ^ { s } ( \langle u \rangle \langle v \rangle ) ^ { a } . } \end{array}
$$

Together with $J _ { t } ( 0 ) = 0$ , this yields

$$
J _ { t } ( W ) \leq 2 C _ { t } ^ { \mathrm { a } } \langle ( u , v ) \rangle ^ { s } ( \langle u \rangle \langle v \rangle ) ^ { a } .
$$

Substitution into the preceding integral estimate gives (3.1). The density argument completes the proof. □

The estimates lift directly to additional spectator variables. We record the form used later.

Lemma 3.4 (Fiberwise lifting). Let S act only on $z \in \mathbb { R } ^ { n }$ and suppose that, for every g in its domain,

$$
\int _ { \mathbb { R } ^ { n } } W _ { \mathrm { o u t } } ( z ) | S g ( z ) | \mathrm { d } z \leq C \int _ { \mathbb { R } ^ { n } } W _ { \mathrm { i n } } ( z ) | g ( z ) | \mathrm { d } z .
$$

Let F be measurable, suppose that $F _ { \zeta } ( z ) = F ( z , \zeta )$ belongs to this domain for almost every $\zeta \in \mathbb { R } ^ { m }$ , and assume that $( z , \zeta ) \mapsto \check { S } F _ { \zeta } ( z )$ is measurable. $I f b \colon \mathbb { R } ^ { m }  [ 0 , \infty ]$ is measurable and the right-hand side below is finite, then

$$
\begin{array} { r l } {  { \int _ { \mathbb { R } ^ { m } } b ( \zeta ) \int _ { \mathbb { R } ^ { n } } W _ { \mathrm { o u t } } ( z ) | S F _ { \zeta } ( z ) | \mathrm { ~ d } z \mathrm { ~ d } \zeta } } \\ & { \leq C \int _ { \mathbb { R } ^ { m } } b ( \zeta ) \int _ { \mathbb { R } ^ { n } } W _ { \mathrm { i n } } ( z ) | F _ { \zeta } ( z ) | \mathrm { ~ d } z \mathrm { ~ d } \zeta . } \end{array}
$$

This applies to Propositions $\it 3 . 1 - 3 . 3 .$ In the antisymmetric case, assume that $F _ { \zeta } ( x , y )$ is antisymmetric in $( x , y )$ for almost every $\zeta .$ . The joint weights may then be used because

$$
\langle ( x , y , \zeta ) \rangle ^ { t } \leq \langle ( x , y ) \rangle ^ { t } , \qquad \langle ( x , y , \zeta ) \rangle ^ { s } \geq \langle ( x , y ) \rangle ^ { s }
$$

for $s \geq 0$ and $t < 0$

Proof. Apply the active-variable estimate to $F _ { \zeta }$ for almost every $\zeta ,$ multiply it by $b ( \zeta )$ and integrate in $\zeta .$ . Tonelli’s theorem justifies the order of integration and yields the stated estimate. □

## 4. A mixed-regularity gain for the Coulomb resolvent

This section combines the convolution bounds of Section 3 with the two-order decay of the free resolvent. We first establish density and the basic mixed Barron gain, then identify the Fourier-side extension with the $H ^ { \bar { 1 } }$ realization and incorporate all same-spin antisymmetries. These mapping results supply the high-frequency contraction used in Section 5.

Fix $\mu > 0$ and set

$$
\mathcal { R } _ { \mu } = \left( - \frac { 1 } { 2 } \Delta + \mu \right) ^ { - 1 } .
$$

Its Fourier multiplier $r _ { \mu } ( \xi ) = ( | \xi | ^ { 2 } / 2 + \mu ) ^ { - 1 }$ satisfies

$$
r _ { \mu } ( \xi ) \le \operatorname * { m a x } \{ 2 , \mu ^ { - 1 } \} \langle \xi \rangle ^ { - 2 } .\tag{4.1}
$$

Lemma 4.1 (Density). Let $I \subseteq \{ 1 , \ldots , N \}$ be nonempty and let $s , \alpha , \beta \geq 0$ . The Schwartz functions antisymmetric with respect to I are dense in the antisymmetric subspace of $B _ { I ; \alpha , \beta } ^ { s }$ . They are also dense in the antisymmetric subspace of

$$
H ^ { 1 } ( \mathbb { R } ^ { 3 N } ) \cap B _ { I ; \alpha , \beta } ^ { s } ,\tag{4.2}
$$

equipped with the sum norm.

Proof. Write $w = w _ { s , I } ^ { \alpha , \beta }$ and let $u \in B _ { I ; \alpha , \beta } ^ { s }$ be antisymmetric with respect to $I ,$ assuming also that $u \in H ^ { 1 }$ for the second assertion. Truncation on expanding balls, followed by mollification and a diagonal choice, gives $h _ { n } \in C _ { c } ^ { \infty } (  { \mathbb { R } } ^ { 3 N } )$ such that $h _ { n } $ ub in $L ^ { 1 } ( w \mathrm { d } \dot { \xi } )$ and, when $u \in H ^ { 1 }$ , in $L ^ { 2 } ( \langle \xi \rangle ^ { 2 } \mathrm { d } \xi )$ . Here truncation converges by dominated convergence, while on each fixed enlarged ball the weights are bounded and the standard $L ^ { 1 }$ and $L ^ { 2 }$ approximate-identity theorem applies [4, Theorem 8.14].

Let ${ \mathfrak { S } } _ { I }$ be the group of permutations of the indices in I, extended by the identity on $I ^ { c } ,$ , and set

$$
( { \mathcal { P } } _ { I } g ) ( \xi ) = { \frac { 1 } { | \mathfrak { S } _ { I } | } } \sum _ { \pi \in \mathfrak { S } _ { I } } \operatorname { s g n } ( \pi ) g ( \pi ^ { - 1 } \xi ) .
$$

Every coordinate permutation π satisfying $\pi ( I ) = I$ obeys

$$
w _ { s , I } ^ { \alpha , \beta } ( \pi \xi ) = w _ { s , I } ^ { \alpha , \beta } ( \xi ) , \qquad \langle \pi \xi \rangle = \langle \xi \rangle .\tag{4.3}
$$

By (4.3) and the change of variables $\eta = \pi ^ { - 1 } \xi$ , every such coordinate permutation is an isometry in both weighted spaces. Hence, by the triangle inequality,

$$
\| \mathcal { P } _ { I } g \| _ { L ^ { 1 } ( w \mathrm { ~ d } \xi ) } \leq \| g \| _ { L ^ { 1 } ( w \mathrm { ~ d } \xi ) } , \qquad \| \mathcal { P } _ { I } g \| _ { L ^ { 2 } ( \langle \xi \rangle ^ { 2 } \mathrm { ~ d } \xi ) } \leq \| g \| _ { L ^ { 2 } ( \langle \xi \rangle ^ { 2 } \mathrm { ~ d } \xi ) } .
$$

Since $\mathcal { P } _ { I } \widehat { u } = \widehat { u }$ , the functions defined by $\widehat { u } _ { n } = \mathcal { P } _ { I } h _ { n }$ belong to $C _ { c } ^ { \infty } ( \mathbb { R } ^ { 3 N } )$ , are antisymmetric with respect to I, and converge to ub in the required norm or sum norm. Their inverse Fourier transforms belong to $\bar { \mathcal { S } } ( \mathbb { R } ^ { 3 N } )$ , which proves both assertions. □

For the operator estimates, fix a nonempty set $I \subseteq \{ 1 , \ldots , N \}$ , let $s , \alpha , \beta \geq 0$ , and write

$$
a _ { i } = \left\{ \begin{array} { l l } { \alpha , } & { i \in I , } \\ { \beta , } & { i \notin I , } \end{array} \right. \quad \quad A _ { I } = \displaystyle \operatorname* { m a x } _ { 1 \leq i \leq N } a _ { i } .\tag{4.4}
$$

Define

$$
D _ { I } = \operatorname* { m a x } \Bigl ( \{ a _ { i } + a _ { j } \mid 1 \leq i < j \leq N , \ \{ i , j \} \not \subseteq I \} \cup \{ 0 \} \Bigr ) .
$$

Then

$$
D _ { I } = \left\{ \begin{array} { l l } { 0 , } & { I ^ { c } = \emptyset , } \\ { \alpha + \beta , } & { | I ^ { c } | = 1 , } \\ { \operatorname* { m a x } \{ \alpha + \beta , 2 \beta \} , } & { | I ^ { c } | \geq 2 . } \end{array} \right.\tag{4.5}
$$

Moreover, $A _ { I } = \alpha$ if $I ^ { c } = \emptyset$ and $A _ { I } = \operatorname* { m a x } \{ \alpha , \beta \}$ otherwise. In the latter case $D _ { I } \geq A _ { I }$ while in the former $D _ { I } = 0$ . Hence (2.2) is equivalent to

$$
s + A _ { I } < 1 , \qquad s + D _ { I } < 1 .\tag{4.6}
$$

Theorem 4.2 (Coulomb–resolvent gain). Let $I \neq \emptyset$ , and let $s , \alpha , \beta \geq 0$ satisfy (2.2). The operator ${ \mathcal { R } } _ { \mu } V$ , initially defined on Schwartz functions that are antisymmetric with respect to I, extends uniquely to a bounded map

$$
\mathcal { R } _ { \mu } V : \mathcal { B } _ { I ; \alpha , \beta } ^ { s } \cap \{ u \mid u \ i s \ a n t i s y m m e t r i c \ w i t h \ r e s p e c t \ t o \ I \} \longrightarrow \mathcal { B } _ { I ; \alpha , \beta } ^ { s + \delta }\tag{4.7}
$$

for some $\delta > 0$ . One may take any positive δ satisfying

$$
\delta < 1 - s - A _ { I } , \qquad \delta < 1 - s - D _ { I } .\tag{4.8}
$$

We prove that the composition ${ \mathcal { R } } _ { \mu } V$ gains a positive isotropic order δ in the mixed spectral Barron scale.

Proof. By (4.6), both upper bounds in (4.8) are positive, so such a δ exists. Since $I \neq \emptyset$ one has $\alpha \leq A _ { I }$ , and therefore

$$
\delta < 1 - s - A _ { I } \leq 1 - \alpha , \qquad \delta < 1 - s - A _ { I } < 2 - s - 2 \alpha .
$$

Here the final strict inequality follows because $( 1 - \alpha ) + ( A _ { I } - \alpha ) > 0$ . Thus the two bounds required below for pairs contained in I follow automatically from (4.8).

Take u $ { \varepsilon } { } ~ \in ~ { \mathcal { S } } ( \mathbb { R } ^ { 3 N } )$ with the prescribed antisymmetry. Under the unitary Fourier convention, set $\kappa _ { \mathrm { C } } = ( 2 \pi ^ { 2 } ) ^ { - 1 }$ . The Coulomb terms satisfy

$$
\bigl ( | x _ { i } \widehat { - R _ { \nu } | } ^ { - 1 } u \bigr ) ( \xi ) = \kappa _ { \mathrm { C } } \int _ { \mathbb { R } ^ { 3 } } \frac { e ^ { - i k \cdot R _ { \nu } } } { | k | ^ { 2 } } \widehat { u } ( \xi _ { 1 } , \dots , \xi _ { i } - k , \dots , \xi _ { N } ) \mathrm { d } k ,
$$

$$
\widehat { ( | x _ { i } - x _ { j } | ^ { - 1 } u ) } ( \xi ) = \kappa _ { \mathrm { C } } \int _ { \mathbb { R } ^ { 3 } } \frac { \widehat { u } ( \xi _ { 1 } , \ldots , \xi _ { i } - k , \ldots , \xi _ { j } + k , \ldots , \xi _ { N } ) } { | k | ^ { 2 } } \mathrm { d } k .
$$

The Coulomb coeficients and their signs are absorbed into the final operator norm. Define

$$
W _ { \mathrm { i n } } ( \xi ) = \langle \xi \rangle ^ { s } \prod _ { \ell = 1 } ^ { N } \langle \xi _ { \ell } \rangle ^ { a _ { \ell } } , \qquad W _ { \mathrm { o u t } } ( \xi ) = \langle \xi \rangle ^ { s + \delta } \prod _ { \ell = 1 } ^ { N } \langle \xi _ { \ell } \rangle ^ { a _ { \ell } } .
$$

Equation (4.1) gives

$$
W _ { \mathrm { o u t } } ( \xi ) r _ { \mu } ( \xi ) \leq \operatorname* { m a x } \{ 2 , \mu ^ { - 1 } \} \langle \xi \rangle ^ { s + \delta - 2 } \prod _ { \ell = 1 } ^ { N } \langle \xi _ { \ell } \rangle ^ { a _ { \ell } } .
$$

Consider a nuclear term acting on the ith coordinate. Write

$$
\zeta = ( \xi _ { \ell } ) _ { \ell \neq i } , \qquad b _ { i } ( \zeta ) = \prod _ { \ell \neq i } \langle \xi _ { \ell } \rangle ^ { a _ { \ell } } , \qquad \tau _ { i } = s + \delta - 2 + a _ { i } .
$$

Since $a _ { i } \leq A _ { I }$ , (4.8) gives

$$
\tau _ { i } \le s + \delta - 2 + A _ { I } < - 1 .
$$

Since $\langle \xi _ { i } \rangle ^ { a _ { i } } \leq \langle \xi \rangle ^ { a _ { i } }$ and $\tau _ { i } < 0$

$$
W _ { \mathrm { o u t } } ( \xi ) r _ { \mu } ( \xi ) \leq C b _ { i } ( \zeta ) \langle \xi \rangle ^ { \tau _ { i } } \leq C b _ { i } ( \zeta ) \langle \xi _ { i } \rangle ^ { \tau _ { i } } .
$$

Proposition 3.1, applied for almost every fixed $\zeta , \mathrm { y }$ ields

$$
\begin{array} { r l } & { \| \mathcal { R } _ { \mu } ( | x _ { i } - R _ { \nu } | ^ { - 1 } u ) \| _ { \mathcal { B } _ { I ; \alpha , \beta } ^ { s + \delta } } } \\ & { \quad \leq C \int _ {  { { \mathbb R } } ^ { 3 ( N - 1 ) } } b _ { i } ( \zeta ) \int _ {  { { \mathbb R } } ^ { 3 } } \langle \xi _ { i } \rangle ^ { \tau _ { i } } \int _ {  { { \mathbb R } } ^ { 3 } } \frac { | \widehat { u } ( \xi _ { i } - k , \zeta ) | } { | k | ^ { 2 } } \mathrm { d } k \mathrm { d } \xi _ { i } \mathrm { d } \zeta } \\ & { \quad \leq C \int _ {  { { \mathbb R } } ^ { 3 N } } b _ { i } ( \zeta ) | \widehat { u } ( \xi ) | \mathrm { d } \xi \leq C \| u \| _ { \mathcal { B } _ { I ; \alpha , \beta } ^ { s } } . } \end{array}\tag{4.9}
$$

The last inequality follows from $b _ { i } ( \zeta ) \leq W _ { \mathrm { i n } } ( \xi )$

Let $i < j$ and write

$$
\zeta = ( \xi _ { \ell } ) _ { \ell \not \in \{ i , j \} } , \qquad b _ { i j } ( \zeta ) = \prod _ { \ell \not \in \{ i , j \} } \langle \xi _ { \ell } \rangle ^ { a _ { \ell } } .
$$

For functions of all momentum variables, let

$$
( T _ { i j } f ) ( \xi ) = \int _ { \mathbb { R } ^ { 3 } } \frac { f ( \xi _ { 1 } , \dots , \xi _ { i } - k , \dots , \xi _ { j } + k , \dots , \xi _ { N } ) } { | k | ^ { 2 } } \mathrm { d } k .
$$

For $\{ i , j \} \not \subseteq I$ , set $t _ { i j } = s + \delta - 2 + a _ { i } + a _ { j }$ . The definition of $D _ { I }$ and (4.8) imply

$$
t _ { i j } \le s + \delta - 2 + D _ { I } < - 1 .
$$

Since $t _ { i j } < 0$

$$
W _ { \mathrm { o u t } } ( \xi ) r _ { \mu } ( \xi ) \leq C b _ { i j } ( \zeta ) \langle \xi \rangle ^ { t _ { i j } } \leq C b _ { i j } ( \zeta ) \langle ( \xi _ { i } , \xi _ { j } ) \rangle ^ { t _ { i j } } .
$$

Proposition 3.2, applied fiberwise, and the inequality $b _ { i j } ( \zeta ) \leq W _ { \mathrm { i n } } ( \xi )$ give

$$
\| \mathcal { R } _ { \mu } ( | x _ { i } - x _ { j } | ^ { - 1 } u ) \| _ { \mathcal { B } _ { I ; \alpha , \beta } ^ { s + \delta } } \leq C \| u \| _ { \mathcal { B } _ { I ; \alpha , \beta } ^ { s } } .\tag{4.10}
$$

Suppose next that $i , j \in I ,$ . Then $a _ { i } = a _ { j } = \alpha$ , and $\widehat { u }$ is antisymmetric under interchange of $\xi _ { i }$ and $\xi _ { j }$ . Put

$$
t _ { i j } = s + \delta - 2 + 2 \alpha .
$$

Since $\delta < 2 - s - 2 \alpha$ , one has $t _ { i j } < 0 . \mathrm { ~ I f ~ } t _ { i j } < - 1$ , Proposition 3.2 applies. $\mathrm { I f } - 1 \le t _ { i j } < 0$ then

$$
( s + \alpha ) - ( t _ { i j } + 1 ) = 1 - \alpha - \delta > 0 .
$$

Proposition 3.3 and Lemma 3.4 therefore imply

$$
\begin{array} { r l } & { \displaystyle \int _ { \mathbb { R } ^ { 3 ( N - 2 ) } } b _ { i j } ( \zeta ) \int _ { \mathbb { R } ^ { 6 } } \langle ( \xi _ { i } , \xi _ { j } , \zeta ) \rangle ^ { t _ { i j } } | T _ { i j } \widehat { u } ( \xi _ { i } , \xi _ { j } , \zeta ) | \mathrm { d } \xi _ { i } \mathrm { d } \xi _ { j } \mathrm { d } \zeta } \\ & { \quad \leq C _ { t _ { i j } } ^ { \mathrm { a } } \displaystyle \int _ { \mathbb { R } ^ { 3 N } } b _ { i j } ( \zeta ) \langle \xi \rangle ^ { s } \big ( \langle \xi _ { i } \rangle \langle \xi _ { j } \rangle \big ) ^ { \alpha } | \widehat { u } ( \xi ) | \mathrm { d } \xi } \\ & { \quad = C _ { t _ { i j } } ^ { \mathrm { a } } \| u \| _ { \mathcal { B } _ { I ; \alpha , \beta } ^ { s } } . } \end{array}
$$

Thus (4.10) also holds when $i , j \in I .$

Summing (4.9) and (4.10) over the finitely many Coulomb terms gives

$$
\begin{array} { r } { \| \mathcal { R } _ { \mu } V u \| _ { \mathcal { B } _ { I ; \alpha , \beta } ^ { s + \delta } } \leq C \| u \| _ { \mathcal { B } _ { I ; \alpha , \beta } ^ { s } } , } \end{array}
$$

where $C$ is independent of u. Lemma 4.1 extends this estimate uniquely to the domain in (4.7). □

Lemma 4.3 (Sobolev bounds for V and $\mathcal { R } _ { \mu } )$ . There is a constant $C _ { V } > 0$ such that

$$
\| V f \| _ { L ^ { 2 } } \leq C _ { V } \| f \| _ { H ^ { 1 } } , \qquad f \in H ^ { 1 } ( \mathbb { R } ^ { 3 N } ) .\tag{4.11}
$$

Moreover,

$$
\| \mathcal { R } _ { \mu } g \| _ { H ^ { 2 } } \leq \operatorname* { m a x } \{ 2 , \mu ^ { - 1 } \} \| g \| _ { L ^ { 2 } } , \qquad g \in L ^ { 2 } ( \mathbb { R } ^ { 3 N } ) .\tag{4.12}
$$

Proof. Applying the three-dimensional Hardy inequality fiberwise in each collision variable $x _ { i } - R _ { \nu }$ and, after a linear change of variables, in $x _ { i } - x _ { j }$ , and then summing the finitely many Coulomb terms gives (4.11); see also [12, proof of Lemma 4.1]. The multiplier bound (4.1) and Plancherel’s theorem give (4.12), which completes the proof. □

Lemma 4.4 (Consistency of the extension). The bounded Fourier-side extension of ${ \mathcal { R } } _ { \mu } V$ furnished by Theorem $4 . 2$ agrees on the antisymmetric subspace of (4.2) with the operator obtained from multiplication by V in position space followed by the $L ^ { 2 }$ resolvent.

Proof. Let E denote the Fourier-side extension in Theorem 4.2. Let $u \in H ^ { 1 } ( \mathbb { R } ^ { 3 N } ) \cap B _ { I ; \alpha , \beta } ^ { s }$ be antisymmetric with respect to $I ,$ and let $( u _ { n } ) \subset S ( \mathbb { R } ^ { 3 N } )$ be the approximating sequence furnished by Lemma 4.1. Equations (4.11) and (4.12) give

$$
\begin{array} { r } { \| \mathcal { R } _ { \mu } V ( u _ { n } - u ) \| _ { H ^ { 2 } } \leq \operatorname* { m a x } \{ 2 , \mu ^ { - 1 } \} C _ { V } \| u _ { n } - u \| _ { H ^ { 1 } } \longrightarrow 0 . } \end{array}
$$

Theorem 4.2 also gives

$$
\| \mathcal { E } ( u _ { n } - u ) \| _ { \mathcal { B } _ { I ; \alpha , \beta } ^ { s + \delta } } \leq C \| u _ { n } - u \| _ { \mathcal { B } _ { I ; \alpha , \beta } ^ { s } } \longrightarrow 0 .
$$

Since $\mathcal { E } u _ { n } = \mathcal { R } _ { \mu } V u _ { n }$ for every $n ,$ their limits in $S ^ { \prime } ( \mathbb { R } ^ { 3 N } )$ coincide. Hence $\mathcal { E } u = \mathcal { R } _ { \mu } V u$ , as asserted. □

Set

$$
\mathfrak { A } _ { \sigma } = \bigcap _ { J \in \mathcal { T } _ { \sigma } } \ \bigcap _ { i , j \in J \atop i < j } \ker ( U _ { i j } + \mathrm { I d } ) \subset \mathcal { S } ^ { \prime } ( \mathbb { R } ^ { 3 N } ) .
$$

For $I \in \mathcal { T } _ { \sigma }$ , use the exponents $a _ { i }$ from (4.4) and define

$$
D _ { I , \sigma } = \operatorname* { m a x } \Bigl ( \{ a _ { i } + a _ { j } \mid 1 \leq i < j \leq N , \sigma _ { i } \neq \sigma _ { j } \} \cup \{ 0 \} \Bigr ) .
$$

Since I is an occupied spin block,

$$
A _ { I } = \left\{ \begin{array} { l l } { \alpha , } & { | \mathcal { T } _ { \sigma } | = 1 , } \\ { \operatorname* { m a x } \{ \alpha , \beta \} , } & { | \mathcal { T } _ { \sigma } | \geq 2 , } \end{array} \right. \quad D _ { I , \sigma } = \left\{ \begin{array} { l l } { 0 , } & { | \mathcal { T } _ { \sigma } | = 1 , } \\ { \alpha + \beta , } & { | \mathcal { T } _ { \sigma } | = 2 , } \\ { \operatorname* { m a x } \{ \alpha + \beta , 2 \beta \} , } & { | \mathcal { T } _ { \sigma } | \geq 3 . } \end{array} \right.
$$

For $| \mathcal { T } _ { \sigma } | \geq 2$ , one has $D _ { I , \sigma } \geq A _ { I }$ . Thus (2.4) is equivalent, for every $I \in \mathcal { T } _ { \sigma }$ , to

$$
s + A _ { I } < 1 , \qquad s + D _ { I , \sigma } < 1 .\tag{4.13}
$$

Theorem 4.5 (Spin-partition Coulomb–resolvent gain). Fix $I \in \mathcal { T } _ { \sigma }$ , let $s , \alpha , \beta \geq 0$ , and assume (2.4). The operator ${ \mathcal { R } } _ { \mu } V$ , initially defined on ${ \mathcal { S } } ( \mathbb { R } ^ { 3 N } ) \cap \mathfrak { A } _ { \sigma }$ , extends uniquely to a bounded map

$$
\mathcal { R } _ { \mu } V \colon B _ { I ; \alpha , \beta } ^ { s } \cap \mathfrak { A } _ { \sigma } \longrightarrow B _ { I ; \alpha , \beta } ^ { s + \delta }
$$

for some $\delta > 0$ . One may take any positive δ satisfying

$$
\delta < 1 - s - A _ { I } ,
$$

$$
\begin{array} { r } { \delta < 1 - s - D _ { I , \sigma } . } \end{array}
$$

Proof. Equation (4.13) makes both upper bounds imposed on δ positive.

For a nuclear term, the argument leading to (4.9) uses only $\delta < 1 - s - A _ { I }$ and therefore applies unchanged.

Suppose that i and j belong to diferent spin blocks. Then

$$
a _ { i } + a _ { j } \leq D _ { I , \sigma } , \qquad t _ { i j } = s + \delta - 2 + a _ { i } + a _ { j } \leq s + \delta - 2 + D _ { I , \sigma } < - 1 .
$$

Thus (4.10) holds for every pair in distinct spin blocks.

Suppose instead that i and j lie in the same spin block. Their coordinate exponents have a common value $c \in \{ \alpha , \beta \}$ , and ub is antisymmetric in $( \xi _ { i } , \xi _ { j } )$ . Since $c \leq A _ { I }$ and $s + c \leq s + A _ { I } < 1$ 2

$$
\delta < 1 - s - A _ { I } \leq 1 - c < 2 - s - 2 c .
$$

With $a _ { i } = a _ { j } = c ,$ the antisymmetric-pair argument in the proof of Theorem 4.2 gives $t _ { i j } = s + \delta - 2 + 2 c < 0$ and $s + c - ( t _ { i j } + 1 ) = 1 - c - \delta > 0$ , so (4.10) holds. These two cases cover every electron pair, and summation proves the asserted boundedness.

For the density argument, define

$$
\mathcal { P } _ { \sigma } = \prod _ { J \in \mathcal { Z } _ { \sigma } } \left( \frac { 1 } { \vert \mathfrak { S } _ { J } \vert } \sum _ { \pi \in \mathfrak { S } _ { J } } \operatorname { s g n } ( \pi ) U _ { \pi } \right) ,
$$

where $U _ { \pi }$ permutes the variables in J and $\widehat { U _ { \pi } f } = U _ { \pi } \widehat { f } .$ . Each spin block lies entirely in I or in $I ^ { c }$ , so every $\pi \in { \mathfrak { S } } _ { J }$ satisfies $\pi ( I ) = I$ and (4.3) applies. Hence

$$
\| \mathcal { P } _ { \sigma } f \| _ { \mathcal { B } _ { I ; \alpha , \beta } ^ { s } } \leq \| f \| _ { \mathcal { B } _ { I ; \alpha , \beta } ^ { s } } , \qquad \| \mathcal { P } _ { \sigma } f \| _ { H ^ { 1 } } \leq \| f \| _ { H ^ { 1 } } .
$$

Given $u \in B _ { I ; \alpha , \beta } ^ { s } \cap \mathfrak { A } _ { \sigma }$ , take the sequence $( u _ { n } ) \subset S ( \mathbb { R } ^ { 3 N } )$ furnished by Lemma 4.1. Then $v _ { n } = \mathcal { P } _ { \sigma } u _ { n }$ satisfies $v _ { n }  \mathcal { P } _ { \sigma } u = u$ in the mixed Barron norm, and also in $H ^ { 1 }$ when $u \in H ^ { 1 }$ . This proves density in ${ \mathfrak { A } } _ { \sigma }$ , and the argument of Lemma 4.4 proves consistency of the extension. □

Remark 4.6. The last case in (4.5) is the ordinary-pair restriction obtained when only antisymmetry on I is used. When additional spin-block antisymmetries or some pair types are absent, Theorem 2.3 retains the sharper value $D _ { I , \sigma }$

## 5. Proof of the eigenfunction theorem

The gain in Theorem 4.2 makes the high-frequency part of the resolvent fixed-point map contractive, while Plancherel’s theorem controls the low-frequency part. We formulate the eigenvalue equation as a fixed point, prove contraction on the relevant $H ^ { 1 }$ and mixed Barron spaces, and identify the solution by a Neumann series. This adapts the argument of [12, Lemmas 4.5–4.6 and Theorem 4.7] to the mixed spectral Barron spaces and proves Theorems 2.2 and 2.3.

For a nonempty set $I \subseteq \{ 1 , \ldots , N \}$ , set

$$
\mathfrak { A } _ { I } = \{ u \in \mathcal { S } ^ { \prime } ( \mathbb { R } ^ { 3 N } ) \mid U _ { i j } u = - u \mathrm { ~ f o r ~ a l l ~ d i s t i n c t ~ } i , j \in I \} ,
$$

and choose $\delta > 0$ as in Theorem 4.2.

Fix $\mu > 0$ . The estimate (4.11) gives $V \psi \in L ^ { 2 } ( \mathbb { R } ^ { 3 N } )$ . The eigenvalue equation therefore implies

$$
\left( - { \frac { 1 } { 2 } } \Delta + \mu \right) \psi = \left( \left( E + \mu \right) \mathrm { I d } - V \right) \psi
$$

in $L ^ { 2 } ( \mathbb { R } ^ { 3 N } )$ . Applying ${ \mathcal { R } } _ { \mu }$ yields

$$
\psi = \mathcal { A } \psi , \qquad \mathcal { A } = \mathcal { R } _ { \mu } \big ( ( E + \mu ) \mathrm { I d } - V \big ) .\tag{5.1}
$$

The choice of $\delta$ implies $\delta < 1 - s - A _ { I } \leq 1$ . Since $w _ { s + \delta , I } ^ { \alpha , \beta } ( \xi ) = \langle \xi \rangle ^ { \delta } w _ { s , I } ^ { \alpha , \beta } ( \xi )$ and $\operatorname* { s u p } _ { \xi \in \mathbb { R } ^ { 3 N } } \langle \xi \rangle ^ { \delta } / ( | \xi | ^ { 2 } / 2 + \mu ) < \infty$ , the scalar part of A satisfies

$$
\| \mathcal { R } _ { \mu } ( E + \mu ) u \| _ { \mathcal { B } _ { I ; \alpha , \beta } ^ { s + \delta } } \leq C _ { \mu , \delta } \vert E + \mu \vert \| u \| _ { \mathcal { B } _ { I ; \alpha , \beta } ^ { s } } ,
$$

which together with Theorem 4.2 gives

$$
\begin{array} { r } { \| \boldsymbol { \mathcal { A } } \| _ { \mathcal { B } _ { I ; \alpha , \beta } ^ { s + \delta } } \leq C _ { \mathrm { B } } \| \boldsymbol { u } \| _ { \mathcal { B } _ { I ; \alpha , \beta } ^ { s } } , \qquad \boldsymbol { u } \in \mathcal { B } _ { I ; \alpha , \beta } ^ { s } \cap \mathfrak { A } _ { I } . } \end{array}\tag{5.2}
$$

The estimates (4.11) and (4.12) give

$$
\begin{array} { r } { \| A u \| _ { H ^ { 2 } } \leq \operatorname* { m a x } \lbrace 2 , \mu ^ { - 1 } \rbrace \big ( | E + \mu | \| u \| _ { L ^ { 2 } } + \| V u \| _ { L ^ { 2 } } \big ) \leq C _ { \mathrm { H } } \| u \| _ { H ^ { 1 } } . } \end{array}
$$

Lemma 4.4 also ensures that this position-space realization of A agrees on

$$
H ^ { 1 } ( \mathbb { R } ^ { 3 N } ) \cap \mathring { B } _ { I ; \alpha , \beta } ^ { s } ( \mathbb { R } ^ { 3 N } ) \cap \mathfrak { A } _ { I }
$$

with the Fourier-side extension used in (5.2).

For $K > 0$ , define

$$
\widehat { P _ { K } f } ( \xi ) = { \bf 1 } _ { \{ | \xi | \geq K \} } \widehat { f } ( \xi ) , \qquad \widehat { Q _ { K } f } ( \xi ) = { \bf 1 } _ { \{ | \xi | < K \} } \widehat { f } ( \xi ) .
$$

On $\{ | \xi | \geq K \}$ , one has $w _ { s , I } ^ { \alpha , \beta } ( \xi ) \leq \langle K \rangle ^ { - \delta } w _ { s + \delta , I } ^ { \alpha , \beta } ( \xi )$ . It follows from (5.2) that

$$
\begin{array} { r } { \| P _ { K } A u \| _ { \mathcal { B } _ { I ; \alpha , \beta } ^ { s } } \leq C _ { \mathrm { B } } \langle K \rangle ^ { - \delta } \| u \| _ { \mathcal { B } _ { I ; \alpha , \beta } ^ { s } } . } \end{array}\tag{5.3}
$$

Similarly,

$$
\| P _ { K } f \| _ { H ^ { 1 } } ^ { 2 } = \int _ { | \xi | \geq K } \langle \xi \rangle ^ { 2 } | { \widehat { f } } ( \xi ) | ^ { 2 } \mathrm { d } \xi \leq \langle K \rangle ^ { - 2 } \| f \| _ { H ^ { 2 } } ^ { 2 } ,
$$

and therefore

$$
\begin{array} { r } { \| P _ { K } A u \| _ { H ^ { 1 } } \leq C _ { \mathrm { H } } \langle K \rangle ^ { - 1 } \| u \| _ { H ^ { 1 } } . } \end{array}\tag{5.4}
$$

Set

$$
X _ { I } = H ^ { 1 } ( \mathbb { R } ^ { 3 N } ) \cap \mathcal { B } _ { I ; \alpha , \beta } ^ { s } ( \mathbb { R } ^ { 3 N } ) \cap \mathfrak { A } _ { I } , \qquad Y _ { I } = H ^ { 1 } ( \mathbb { R } ^ { 3 N } ) \cap \mathfrak { A } _ { I } ,
$$

and equip $X _ { I }$ with

$$
\| u \| _ { X _ { I } } = \| u \| _ { H ^ { 1 } } + \| u \| _ { \mathcal { B } _ { I ; \alpha , \beta } ^ { s } } .
$$

Each antisymmetry constraint is the kernel of the continuous map $u \mapsto U _ { i j } u + u$ on both ambient spaces, so $X _ { I }$ and $Y _ { I }$ are Banach spaces. The operators $V , \mathcal { R } _ { \mu } , P _ { K }$ , and $Q _ { K }$ commute with permutations of electron coordinates and hence preserve ${ \mathfrak { A } } _ { I }$ . Define

$$
{ \mathcal T } _ { \cal K } = P _ { \cal K } { \cal A } , \qquad { \gamma } _ { \cal K } = \operatorname* { m a x } \left\{ C _ { \mathrm { B } } \langle { \cal K } \rangle ^ { - \delta } , C _ { \mathrm { H } } \langle { \cal K } \rangle ^ { - 1 } \right\} .
$$

The estimates (5.3) and (5.4) imply

$$
\| T _ { K } u \| _ { X _ { I } } \le \gamma _ { K } \| u \| _ { X _ { I } } , \qquad \| T _ { K } u \| _ { H ^ { 1 } } \le \gamma _ { K } \| u \| _ { H ^ { 1 } } .
$$

Choosing $K$ suficiently large implies $\gamma _ { K } < 1$ . Then $\mathcal { T } _ { K }$ is a strict contraction on both $X _ { I }$ and $Y _ { I }$

The low-frequency part belongs to $X _ { I }$ . By Cauchy–Schwarz and Plancherel’s theorem,

$$
\begin{array} { r l r } {  { \| Q _ { K } \psi \| _ { \mathcal { B } _ { I ; \alpha , \beta } ^ { s } } = \int _ { | \xi | < K } w _ { s , I } ^ { \alpha , \beta } ( \xi ) | \widehat { \psi } ( \xi ) | \mathrm { ~ d } \xi } } \\ & { } & { \leq ( \int _ { | \xi | < K } ( w _ { s , I } ^ { \alpha , \beta } ( \xi ) ) ^ { 2 } \mathrm { ~ d } \xi ) ^ { 1 / 2 } \| \psi \| _ { L ^ { 2 } } = C _ { K , s , \alpha , \beta } \| \psi \| _ { L ^ { 2 } } . } \end{array}\tag{5.5}
$$

Moreover, $Q _ { K } \psi \in H ^ { 1 } \cap \mathfrak { A } _ { I }$ . By (5.2) and the $H ^ { 1 } { - } \mathrm { t o } { - } H ^ { 2 }$ estimate,

$$
\boldsymbol { \mathcal { A } } ( Q _ { K } \boldsymbol { \psi } ) \in H ^ { 2 } \cap \mathcal { B } _ { I ; \alpha , \beta } ^ { s + \delta } \cap \mathfrak { A } _ { I } \subset X _ { I } .
$$

Hence $g _ { K } = P _ { K } A ( Q _ { K } \psi ) \in X _ { I }$

Applying $P _ { K }$ to (5.1), we obtain

$$
P _ { K } \psi = \mathcal { T } _ { K } ( P _ { K } \psi ) + g _ { K } .
$$

Since $\| \mathcal { T } _ { K } \| _ { \mathcal { L } ( X _ { I } ) } < 1$ , the Neumann series

$$
v = ( \mathrm { I d } - { \mathcal { T } } _ { K } ) ^ { - 1 } g _ { K } = \sum _ { n = 0 } ^ { \infty } { \mathcal { T } } _ { K } ^ { n } g _ { K }
$$

converges in $X _ { I }$ and gives the unique solution of $v = \mathcal { T } _ { K } v + g _ { K }$ in that space. The function $P _ { K } \psi \in Y _ { I }$ satisfies the same equation. If $h _ { 1 } , h _ { 2 } \in Y _ { I }$ are two solutions, then

$$
\begin{array} { r } { \| h _ { 1 } - h _ { 2 } \| _ { H ^ { 1 } } = \| \mathcal { T } _ { K } ( h _ { 1 } - h _ { 2 } ) \| _ { H ^ { 1 } } \leq \gamma _ { K } \| h _ { 1 } - h _ { 2 } \| _ { H ^ { 1 } } . } \end{array}
$$

Since $\gamma _ { K } < 1$ , one has $h _ { 1 } = h _ { 2 }$ . Thus $P _ { K } \psi = v \in X _ { I }$ , which together with (5.5) proves Theorem 2.2.

Remark 5.1 (Redundancy of the same-block bounds). The resolvent estimate for pairs contained in I also requires $\delta < 1 - \alpha$ and $\delta < 2 - s - 2 \alpha$ . These inequalities impose no additional restriction on Theorem 2.2. Indeed, (2.2) implies $s + \alpha < 1$ , while Theorem 4.2 permits

$$
\delta < 1 - s - A _ { I } \leq 1 - s - \alpha \leq 1 - \alpha , \qquad 1 - s - \alpha < 2 - s - 2 \alpha .
$$

For Theorem 2.3, (2.3) gives $\psi \in { \mathfrak { A } } _ { \sigma }$ . Fix $I \in \mathcal { T } _ { \sigma }$ , replace Theorem 4.2 by Theorem 4.5, and use

$$
X _ { I , \sigma } = H ^ { 1 } ( \mathbb { R } ^ { 3 N } ) \cap \mathscr { B } _ { I ; \alpha , \beta } ^ { s } ( \mathbb { R } ^ { 3 N } ) \cap \mathfrak { A } _ { \sigma } , \qquad Y _ { \sigma } = H ^ { 1 } ( \mathbb { R } ^ { 3 N } ) \cap \mathfrak { A } _ { \sigma } .
$$

All operators used above preserve ${ \mathfrak { A } } _ { \sigma }$ . The estimates $( 5 . 2 ) \ – ( 5 . 4 )$ , the low-frequency estimate, and the two uniqueness arguments remain valid in these spaces, giving $\psi \in B _ { I ; \alpha , \beta } ^ { s } .$ Since (2.4) is independent of I, this holds for every $I \in \mathcal { T } _ { \sigma }$ . This proves (2.5).

Proof of Corollary 2.6. Theorem 2.3 gives $w _ { s _ { 0 } , I } ^ { \alpha _ { 0 } , \beta _ { 0 } } \hat { \psi } \in L ^ { 1 }$ . Under the conversion $\xi = 2 \pi \zeta$ from Meng’s Fourier-transform convention to ours, the mixed-Sobolev estimate [6, (2.3)– (2.7) and Corollary 2.4] gives $w _ { 1 , I } ^ { \alpha _ { 2 } , \beta _ { 2 } } \hat { \psi } \in L ^ { 2 } ( \mathbb { R } ^ { 3 N } )$ . Moreover,

$$
w _ { s _ { \theta } , I } ^ { \alpha _ { \theta } , \beta _ { \theta } } = \big ( w _ { s _ { 0 } , I } ^ { \alpha _ { 0 } , \beta _ { 0 } } \big ) ^ { 1 - \theta } \big ( w _ { 1 , I } ^ { \alpha _ { 2 } , \beta _ { 2 } } \big ) ^ { \theta } , \qquad \frac { 1 } { p } = ( 1 - \theta ) + \frac { \theta } { 2 } .
$$

Since $1 < p < 2$ , the exponents $( 2 - p ) ^ { - 1 }$ and $( p - 1 ) ^ { - 1 }$ are conjugate. We employ Hölder’s inequality to obtain

$$
\begin{array} { r } { \displaystyle \int _ {  { \mathbb { R } } ^ { 3 N } } \big ( w _ { s _ { \theta } , I } ^ { \alpha _ { \theta } , \beta _ { \theta } } \vert \widehat \psi \vert \big ) ^ { p } \mathrm { d } \xi \leq \left( \int _ {  { \mathbb { R } } ^ { 3 N } } w _ { s _ { 0 } , I } ^ { \alpha _ { 0 } , \beta _ { 0 } } \vert \widehat \psi \vert \mathrm { d } \xi \right) ^ { p ( 1 - \theta ) } } \\ { \times \left( \int _ {  { \mathbb { R } } ^ { 3 N } } \big ( w _ { 1 , I } ^ { \alpha _ { 2 } , \beta _ { 2 } } \vert \widehat \psi \vert \big ) ^ { 2 } \mathrm { d } \xi \right) ^ { p \theta / 2 } . } \end{array}
$$

Taking the pth root, we prove the assertion.

## 6. Proofs of the sharpness results

We first convert a nonvanishing Coulomb cusp into failure of the mixed Fourier $L ^ { 1 }$ norm at the boundary exponents and then apply this obstruction to the atomic states in Proposition 2.7 and Corollary 2.8.

Lemma 6.1 (Localized cusp obstruction). Let $I \subseteq \{ 1 , \ldots , N \}$ be nonempty, let $s , \alpha , \beta \geq 0$ and let $u \in S ^ { \prime } ( \mathbb { R } ^ { 3 N } )$ . Suppose that an open set $\mathcal { U } \bar { \mathbf { \Lambda } } \doteq \mathbb { R } ^ { 3 } \times \mathbb { R } ^ { { \hat { 3 } } N - 3 }$ meets $\{ r = 0 \}$ and that

$$
u ( r , y ) = u _ { 0 } ( r , y ) + | r | u _ { 1 } ( r , y ) , \qquad ( r , y ) \in \mathcal { U } ,
$$

where $u _ { 0 }$ and $u _ { 1 }$ are real analytic on U. If $u _ { 1 } ( 0 , \cdot ) \not \equiv 0$ on $\{ y \mid ( 0 , y ) \in \mathcal { U } \}$ , then the following assertions hold.

(i) $\begin{array} { r } { I f r = x _ { i } - R _ { \nu } } \end{array}$ is a nuclear collision variable, then u $\notin B _ { I ; \alpha , \beta } ^ { s }$ whenever $s + a _ { i } \geq 1$

(ii) $\textit { I f r } = x _ { i } - x _ { j }$ is an electron–electron collision variable, then u $\notin B _ { I ; \alpha , \beta } ^ { s }$ whenever $s + a _ { i } + a _ { j } \geq 1$

Proof. The inequality $\langle z + z ^ { \prime } \rangle \leq \sqrt { 2 } \langle z \rangle \langle z ^ { \prime } \rangle$ implies

$$
w _ { s , I } ^ { \alpha , \beta } ( \boldsymbol { \xi } ) \leq 2 ^ { ( s + \sum _ { i = 1 } ^ { N } a _ { i } ) / 2 } w _ { s , I } ^ { \alpha , \beta } ( \boldsymbol { \xi } - \boldsymbol { \zeta } ) w _ { s , I } ^ { \alpha , \beta } ( \boldsymbol { \zeta } ) .
$$

For $\varphi \in C _ { c } ^ { \infty } (  { \mathbb { R } } ^ { 3 N } )$ , the Fourier product formula and Tonelli’s theorem therefore yield

$$
\begin{array} { r } { \| \varphi u \| _ { \mathcal { B } _ { I ; \alpha , \beta } ^ { s } } \leq ( 2 \pi ) ^ { - 3 N / 2 } 2 ^ { ( s + \sum _ { i = 1 } ^ { N } a _ { i } ) / 2 } \| w _ { s , I } ^ { \alpha , \beta } \widehat \varphi \| _ { L ^ { 1 } } \| u \| _ { \mathcal { B } _ { I ; \alpha , \beta } ^ { s } } . } \end{array}\tag{6.1}
$$

Since $ { \widehat { \varphi } } \in { \mathcal { S } } ( \mathbb { R } ^ { 3 N } )$ , one has $\| w _ { s , I } ^ { \alpha , \beta } \widehat \varphi \| _ { L ^ { 1 } } < \infty$

Put $m = 3 N - 3$ and let $( \rho , \eta ) \in \mathbb { R } ^ { 3 } \times \mathbb { R } ^ { m }$ denote the Fourier variables dual to $( r , y )$ We write $A \Subset B$ if A is a compact subset of B. Assume first that $u _ { 1 } ( 0 , \cdot ) \not \equiv 0$ and choose $y _ { 0 } \in \mathbb { R } ^ { m }$ such that $( 0 , y _ { 0 } ) \in \mathcal { U }$ and $u _ { 1 } ( 0 , y _ { 0 } ) \neq 0$ . There are $0 < \varepsilon ^ { \prime } < \varepsilon$ and open sets $y _ { 0 } \in U _ { y } ^ { \prime } \Subset U _ { y } \subset \mathbb { R } ^ { m }$ such that $B _ { \varepsilon } ( 0 ) \times U _ { y } \Subset \mathcal { U }$ . Choose

$$
\begin{array} { r l } & { \chi _ { 0 } \in C _ { c } ^ { \infty } ( B _ { \varepsilon } ( 0 ) ) , \ : 0 \leq \chi _ { 0 } \leq 1 , \ : \ : \chi _ { 0 } \equiv 1 \mathrm { ~ o n ~ } B _ { \varepsilon ^ { \prime } } ( 0 ) , } \\ & { \chi _ { y } \in C _ { c } ^ { \infty } ( U _ { y } ) , \qquad 0 \leq \chi _ { y } \leq 1 , \ : \ : \chi _ { y } \equiv 1 \mathrm { ~ o n ~ } U _ { y } ^ { \prime } . } \end{array}
$$

Define $\chi ( r , y ) = \chi _ { 0 } ( r ) \chi _ { y } ( y )$ and $b _ { 0 } ( y ) = \chi _ { y } ( y ) u _ { 1 } ( 0 , y )$ . Then

$$
\chi \in C _ { c } ^ { \infty } ( \mathcal { U } ) , \qquad 0 \le \chi \le 1 , \qquad \chi \equiv 1 \mathrm { ~ o n ~ } B _ { \varepsilon ^ { \prime } } ( 0 ) \times U _ { y } ^ { \prime } , \qquad \operatorname { s u p p } \chi \in \mathcal { U } ,
$$

while $b _ { 0 } \in C _ { c } ^ { \infty } ( U _ { y } )$ and $b _ { 0 } ( y _ { 0 } ) = u _ { 1 } ( 0 , y _ { 0 } ) \neq 0 .$

Set $g ( r ) = \chi _ { 0 } ( r ) | r |$ . Since $\chi _ { 0 } \equiv 1$ near the origin, the distributional identity $\Delta _ { r } ^ { 2 } | r | =$ −8π ${ \bf \delta } \cdot \delta _ { 0 }$ gives

$$
G : = \Delta _ { r } ^ { 2 } g + 8 \pi \delta _ { 0 } \in C _ { c } ^ { \infty } ( \mathbb { R } ^ { 3 } ) .
$$

Under the unitary Fourier convention,

$$
| \rho | ^ { 4 } { \widehat g } ( \rho ) = - 8 \pi ( 2 \pi ) ^ { - 3 / 2 } + { \widehat G } ( \rho ) .
$$

Since $\widehat { G }$ is Schwartz, for every $M > 0$

$$
\widehat g ( \rho ) = - 2 \sqrt { \frac 2 \pi } | \rho | ^ { - 4 } + { \cal O } _ { M } ( | \rho | ^ { - M } ) \qquad ( | \rho | \to \infty ) ,
$$

and the same rapidly decaying remainder estimate holds after any number of ρ-derivatives. To justify the Taylor step, set

$$
b _ { k } ( y ) = \chi _ { y } ( y ) \partial _ { r _ { k } } u _ { 1 } ( 0 , y ) , \qquad C _ { k \ell } ( r , y ) = \chi _ { y } ( y ) \int _ { 0 } ^ { 1 } ( 1 - \lambda ) \partial _ { r _ { k } } \partial _ { r _ { \ell } } u _ { 1 } ( \lambda r , y ) \ \mathrm { d } \lambda .
$$

Since $B _ { \varepsilon } ( 0 )$ is star-shaped with respect to the origin, Taylor’s formula with integral remainder gives, on supp $\chi _ { 0 } \times \mathrm { s u p p } \chi _ { y }$

$$
\chi _ { y } ( y ) u _ { 1 } ( r , y ) = b _ { 0 } ( y ) + \sum _ { k = 1 } ^ { 3 } r _ { k } b _ { k } ( y ) + \sum _ { k , \ell = 1 } ^ { 3 } r _ { k } r _ { \ell } C _ { k \ell } ( r , y ) .
$$

Consequently,

$$
\chi u = \chi u _ { 0 } + g ( r ) b _ { 0 } ( y ) + \sum _ { k = 1 } ^ { 3 } r _ { k } g ( r ) b _ { k } ( y ) + g ( r ) \sum _ { k , \ell = 1 } ^ { 3 } r _ { k } r _ { \ell } C _ { k \ell } ( r , y ) .
$$

Since supp $\chi \Subset U$ , extension by zero gives $\chi u _ { 0 } \in C _ { c } ^ { \infty } ( \mathbb { R } ^ { 3 + m } )$ . Hence, for every $M _ { 0 } \geq 0$ there is a constant $C _ { M _ { 0 } }$ such that

$$
\vert \widehat { \chi u _ { 0 } } ( \rho , \eta ) \vert \leq C _ { M _ { 0 } } \langle ( \rho , \eta ) \rangle ^ { - M _ { 0 } } , \qquad ( \rho , \eta ) \in \mathbb { R } ^ { 3 } \times \mathbb { R } ^ { m } .
$$

Moreover,

$$
\widehat { r _ { k } g } ( \rho ) = i \partial _ { \rho _ { k } } \widehat { g } ( \rho ) = 8 i \sqrt { \frac { 2 } { \pi } } \rho _ { k } | \rho | ^ { - 6 } + { \cal O } _ { M } ( | \rho | ^ { - M } ) \qquad ( | \rho | \to \infty ) .
$$

For the last sum in the decomposition above, a further first-order Taylor formula gives

$$
C _ { k \ell } ( r , y ) = C _ { k \ell } ( 0 , y ) + \sum _ { j = 1 } ^ { 3 } r _ { j } E _ { k \ell j } ( r , y ) , \qquad E _ { k \ell j } ( r , y ) = \int _ { 0 } ^ { 1 } \partial _ { r _ { j } } C _ { k \ell } ( \lambda r , y ) \ \mathrm { d } \lambda .
$$

The terms containing $C _ { k \ell } ( 0 , y )$ satisfy

$$
\widehat { r _ { k } r _ { \ell } g } ( \rho ) = - \partial _ { \rho _ { k } } \partial _ { \rho _ { \ell } } \widehat { g } ( \rho ) = O ( | \rho | ^ { - 6 } ) \qquad ( | \rho | \to \infty ) .
$$

Define the remaining part by

$$
h ( r , y ) : = g ( r ) \sum _ { k , \ell , j = 1 } ^ { 3 } r _ { k } r _ { \ell } r _ { j } E _ { k \ell j } ( r , y ) .
$$

Since $g ( r ) = \chi _ { 0 } ( r ) | r |$ , this is a finite sum of terms $a _ { k \ell j } ( r , y ) | r | r _ { k } r _ { \ell } r _ { j }$ with $a _ { k \ell j } = \chi _ { 0 } E _ { k \ell j } \in$ $C _ { c } ^ { \infty } ( \mathbb { R } ^ { 3 + m } )$ . Homogeneity gives, for every multi-index γ with $| \gamma | \le 6$

$$
\big | \partial _ { r } ^ { \gamma } \big ( | r | r _ { k } r _ { \ell } r _ { j } \big ) \big | \le C _ { \gamma } | r | ^ { 4 - | \gamma | } , \qquad 0 < | r | \le 1 .
$$

At order six the right-hand side is $C _ { \gamma } | r | ^ { - 2 }$ , and

$$
\int _ { \{ | r | < 1 \} } | r | ^ { - 2 } ~ \mathrm { d } r = 4 \pi < \infty .
$$

Integration by parts on $\{ | r | > \varepsilon \}$ produces boundary terms of order at most $O ( \varepsilon )$ , which vanish as $\varepsilon \downarrow 0$ . Thus these locally integrable classical derivatives represent the weak derivatives. The Leibniz rule gives $\partial _ { r } ^ { \gamma } h \in L ^ { 1 } ( \mathbb { R } ^ { 3 + m } )$ for $| \gamma | \le 6$ . For each $\rho \neq 0$ , choose $k _ { 0 } \in \{ 1 , 2 , 3 \}$ such that $| \rho _ { k _ { 0 } } | \geq | \rho | / \sqrt { 3 }$ . Six integrations by parts in $r _ { k _ { 0 } }$ give

$$
| \widehat { h } ( \rho , \eta ) | \leq ( 2 \pi ) ^ { - ( 3 + m ) / 2 } | \rho _ { k _ { 0 } } | ^ { - 6 } \| \partial _ { r _ { k _ { 0 } } } ^ { 6 } h \| _ { L ^ { 1 } ( \mathbb { R } ^ { 3 + m } ) } \leq C | \rho | ^ { - 6 } ,
$$

uniformly in η. Thus $\widehat { h } ( \rho , \eta ) = O ( | \rho | ^ { - 6 } )$ uniformly in η.

Combining these estimates yields the more precise expansion

$$
\begin{array} { l } { { \displaystyle \widehat { \chi u } ( \rho , \eta ) = - 2 \sqrt { \frac 2 \pi } | \rho | ^ { - 4 } \widehat { b _ { 0 } } ( \eta ) } } \\ { { \displaystyle \qquad + 8 i \sqrt { \frac 2 \pi } | \rho | ^ { - 6 } \sum _ { k = 1 } ^ { 3 } \rho _ { k } \widehat { b _ { k } } ( \eta ) + { \cal O } ( | \rho | ^ { - 6 } ) \qquad ( | \rho | \to \infty ) , } } \end{array}\tag{6.2}
$$

uniformly in $\eta \in \mathbb { R } ^ { m }$ . In particular, the second line is $O ( | \rho | ^ { - 5 } )$

Fourier injectivity and $b _ { 0 } \not \equiv 0$ give a point $\eta _ { 0 } \in \mathbb { R } ^ { m }$ with $\widehat { b } _ { 0 } ( \eta _ { 0 } ) \neq 0$ . By continuity, there is $\varepsilon _ { \eta } > 0$ such that, with

$$
\mathcal { O } = B _ { \varepsilon _ { \eta } } ( \eta _ { 0 } ) , \qquad b _ { \ast } = \frac { 1 } { 2 } | \widehat { b } _ { 0 } ( \eta _ { 0 } ) | ,
$$

one has $| \widehat { b } _ { 0 } ( \eta ) | \geq b _ { * }$ for every $\eta \in \mathcal { O }$ . Denote m-dimensional Lebesgue measure by $\mathcal { L } ^ { m }$ The uniform remainder in (6.2) yields a $\rho _ { \mathcal { O } } \geq 1$ for which

$$
| \widehat { \chi u } ( \rho , \eta ) | \geq \sqrt { \frac { 2 } { \pi } } b _ { * } | \rho | ^ { - 4 } , \qquad \eta \in \mathcal { O } , \quad | \rho | \geq \rho \sigma .\tag{6.3}
$$

Consider case (ii). Write

$$
R = \frac { x _ { i } + x _ { j } } { 2 } , \qquad r = x _ { i } - x _ { j } , \qquad y = ( R , x _ { i , j } ) , \qquad x _ { i , j } = ( x _ { k } ) _ { k \not \in \{ i , j \} } .
$$

The corresponding Fourier variables are

$$
\eta = ( \eta _ { R } , \eta ^ { \prime } ) = ( \xi _ { i } + \xi _ { j } , \xi _ { \widehat { i , j } } ) , \qquad \rho = \frac { \xi _ { i } - \xi _ { j } } 2 , \qquad \xi _ { \widehat { i , j } } = ( \xi _ { k } ) _ { k \not \in \{ i , j \} } ,
$$

and the active-variable transformation has

$$
{ \binom { \eta _ { R } } { \rho } } = { \binom { 1 } { \frac { 1 } { 2 } } } \ { \binom { 1 } { \zeta _ { j } } } \int _ { \zeta _ { j } } ^ { \zeta _ { i } } \biggr ) , \qquad \biggl | \mathrm { d e t } \left( { \begin{array} { c c } { 1 } & { 1 } \\ { { \frac { 1 } { 2 } } } & { - { \frac { 1 } { 2 } } } \end{array} } \right) \biggr | = 1 .
$$

The same matrix acts in each spatial component. Hence

$$
\xi _ { i } = \rho + \frac { \eta _ { R } } { 2 } , \qquad \xi _ { j } = - \rho + \frac { \eta _ { R } } { 2 } , \qquad \xi _ { _ { i , j } } = \eta ^ { \prime } , \qquad \mathrm { d } \xi = \mathrm { d } \rho \mathrm { d } \eta .
$$

These relations define $\xi = \xi ( \rho , \eta )$ . Thus η is the Fourier variable dual to the coordinates tangent to the collision manifold $\{ r = 0 \}$ . Set

$$
\begin{array} { r l r } & { M _ { \mathcal { O } } = \displaystyle \operatorname* { s u p } _ { \eta \in \mathcal { O } } \left| \eta \right| , } & { \rho _ { 0 } = \operatorname* { m a x } \{ \rho _ { \mathcal { O } } , 2 M _ { \mathcal { O } } , 1 \} , } \\ & { \mathcal { C } _ { \mathcal { O } } = \{ ( \rho , \eta ) \in \mathbb { R } ^ { 3 } \times \mathbb { R } ^ { m } \ \left| \ \eta \in \mathcal { O } , \ \left| \ \rho \right| \ge \rho _ { 0 } \right\} . } & \end{array}
$$

For $( \rho , \eta ) \in \mathcal { C } _ { \mathcal { O } }$

$$
\frac 3 4 | \rho | \leq | \xi _ { i } | , | \xi _ { j } | \leq \frac 5 4 | \rho | , \qquad | \xi _ { i } | ^ { 2 } + | \xi _ { j } | ^ { 2 } = 2 | \rho | ^ { 2 } + \frac 1 2 | \eta _ { R } | ^ { 2 } , \qquad \langle \xi \rangle \geq | \rho | .
$$

Since all exponents in (2.1) are nonnegative, the full weight satisfies

$$
w _ { s , I } ^ { \alpha , \beta } ( \xi ( \rho , \eta ) ) \geq \langle \xi \rangle ^ { s } \langle \xi _ { i } \rangle ^ { a _ { i } } \langle \xi _ { j } \rangle ^ { a _ { j } } \geq \left( \frac 3 4 \right) ^ { a _ { i } + a _ { j } } | \rho | ^ { s + a _ { i } + a _ { j } }\tag{6.4}
$$

on $\mathcal { C } _ { \mathcal { O } }$ . Combining (6.3), (6.4), and spherical coordinates in ρ gives

$$
\begin{array} { r l } & { \| \chi u \| _ { \mathcal { B } _ { I ; \alpha , \beta } ^ { s } } \geq \displaystyle \int _ { \mathcal { O } } \int _ { | \rho | \geq \rho _ { 0 } } w _ { s , I } ^ { \alpha , \beta } ( \xi ( \rho , \eta ) ) | \widehat { \chi u } ( \rho , \eta ) | \mathrm { d } \rho \mathrm { d } \eta } \\ & { \qquad \geq 4 \pi \mathcal { L } ^ { m } ( \mathcal { O } ) \sqrt { \displaystyle \frac { 2 } { \pi } } b _ { * } \left( \frac { 3 } { 4 } \right) ^ { a _ { i } + a _ { j } } \displaystyle \int _ { \rho _ { 0 } } ^ { \infty } \varrho ^ { s + a _ { i } + a _ { j } - 2 } \mathrm { d } \varrho . } \end{array}
$$

The last integral diverges if and only if $s + a _ { i } + a _ { j } \geq 1$ , proving (ii).

In case (i), take $r = x _ { i } - R _ { \nu } , y = x _ { \widehat { i } } = ( x _ { k } ) _ { k \neq i } , \rho = \xi _ { i }$ , and $\eta = \xi _ { \widehat { i } } = ( \xi _ { k } ) _ { k \neq i }$ . The translation by $R _ { \nu }$ contributes only the unimodular factor $e ^ { - i R _ { \nu } \cdot \rho }$ , and $\mathrm { d } \xi = \mathrm { d } \rho \ \mathrm { d } \eta$ . On $\mathcal { O } \times \{ | \rho | \geq \rho _ { \mathcal { O } } \}$ , one has $w _ { s , I } ^ { \alpha , \beta } ( \xi ) \geq \langle \xi \rangle ^ { s } \langle \xi _ { i } \rangle ^ { a _ { i } } \geq | \rho | ^ { s + a _ { i } }$ . Hence

$$
\| \chi u \| _ { \mathcal { B } _ { I ; \alpha , \beta } ^ { s } } \geq 4 \pi \mathcal { L } ^ { m } ( \mathcal { O } ) \sqrt { \frac { 2 } { \pi } } b _ { * } \int _ { \rho _ { \mathcal { O } } } ^ { \infty } \varrho ^ { s + a _ { i } - 2 } \ \mathrm { d } \varrho ,
$$

which is infinite if and only if $s + a _ { i } \geq 1$ . This proves (i).

If u belonged to $B _ { I ; \alpha , \beta } ^ { s } .$ , the localization bound (6.1) would imply $\chi u \in B _ { I ; \alpha , \beta } ^ { s } .$ , contradicting the divergent lower bounds in (i)–(ii). This completes the proof. □

The localized obstruction yields the endpoint tests in Proposition 2.7. The ground state Φ $_ { N , Z }$ has a continuous representative by [9, Theorem C.1.1]; the Harnack inequality [9, Theorem C.1.3] makes this representative strictly positive.

Proof of Proposition 2.7. The Fourier transform of ψ is a nonzero multiple of $( 1 + | \xi | ^ { 2 } ) ^ { - 2 }$ see $\left[ 1 2 , \ : ( 1 . 5 ) \ – ( 1 . 6 ) \right]$ . Hence

$$
\begin{array} { r l r } {  { \| \psi _ { \mathrm { H } } \| _ { \mathcal { B } _ { \{ 1 \} ; \alpha , \beta } ^ { s } } = C \int _ { \mathbb { R } ^ { 3 } } \langle \xi \rangle ^ { s + \alpha - 4 } \mathrm { d } \xi } } \\ & { } & { = 4 \pi C \int _ { 0 } ^ { \infty } r ^ { 2 } ( 1 + r ^ { 2 } ) ^ { ( s + \alpha - 4 ) / 2 } \mathrm { d } r . } \end{array}
$$

The integrand is $\mathcal { O } ( r ^ { 2 } )$ as $r \downarrow 0$ and is comparable to $r ^ { s + \alpha - 2 }$ as $r  \infty$ . The integral is finite if and only if $s + \alpha < 1$ , proving (i).

Fix a point at which $x _ { i } = x _ { j }$ and no other electron–electron or electron–nucleus collision occurs. Set $r = x _ { i } - x _ { j }$ $R \stackrel { \cdot } { = } ( x _ { i } + x _ { j } ) / 2$ , and $y = ( R , x _ { i , j } )$ , where $x _ { \hat { i , j } } = ( x _ { k } ) _ { k \not \in \{ i , j \} }$ Restrict to a neighborhood meeting no other collision set. In this neighborhood, the local structure result [3, Theorem 1.4 and Remark 1.6] gives

$$
\Phi _ { N , Z } ( r , y ) = u _ { 0 } ( r , y ) + | r | u _ { 1 } ( r , y ) ,
$$

with analytic coeficients. In these coordinates,

$$
- \frac { 1 } { 2 } ( \Delta _ { x _ { i } } + \Delta _ { x _ { j } } ) = - \frac { 1 } { 4 } \Delta _ { R } - \Delta _ { r } .
$$

All potential terms other than the isolated pair potential are locally bounded in these coordinates. On every smaller compact set in the tangential variable $y ,$ as $r  0$ one has uniformly

$$
\begin{array} { c } { { \displaystyle - \Delta _ { r } \big ( | r | u _ { 1 } ( r , y ) \big ) = - \frac { 2 u _ { 1 } ( 0 , y ) } { | r | } + O ( 1 ) , } } \\ { { \displaystyle \frac { 1 } { | r | } \Phi _ { N , Z } ( r , y ) = \frac { u _ { 0 } ( 0 , y ) } { | r | } + O ( 1 ) , } } \end{array}
$$

All remaining terms in $( H - E ) \Phi _ { N , Z }$ are O(1), uniformly on such compact sets. Thus

$$
( H - E ) \Phi _ { N , Z } = \frac { - 2 u _ { 1 } ( 0 , y ) + u _ { 0 } ( 0 , y ) } { | r | } + O ( 1 ) ,
$$

so

$$
u _ { 1 } ( 0 , y ) = \frac { 1 } { 2 } u _ { 0 } ( 0 , y ) .
$$

The continuous representative of $\Phi _ { N , Z }$ is strictly positive, so $u _ { 0 } ( 0 , y ) = \Phi _ { N , Z } ( 0 , y ) > 0$ Lemma 6.1 applied to the pair (1, 2) proves the necessity of $s + \alpha + \beta < 1$ in (ii). For (iii), the pairs (1, 2) and (2, 3) give $s + \alpha + \beta < 1$ and $s + 2 \beta < 1$ , respectively. Conversely, antisymmetry with respect to the singleton $I = \{ 1 \}$ is a void condition, and Theorem 2.2 proves suficiency in all three cases. □

Proof of Corollary 2.8. Theorem 2.2 proves suficiency. Proposition 2.7 proves necessity for the three cases by taking, respectively, hydrogen with $( N , I ) = ( 1 , \{ 1 \} )$ , helium with $( N , I ) = ( 2 , \{ 1 \} )$ , and lithium with $( N , I ) = ( 3 , \{ 1 \} )$ . Hence no boundary face of the family can be enlarged uniformly. □

## 7. Conclusion

We established sharp mixed spectral Barron regularity for eigenfunctions of clampednuclei Coulomb Hamiltonians. Weighted Fourier $L ^ { \tilde { 1 } }$ estimates for the nuclear and electron– electron Coulomb operators, together with antisymmetric cancellation on same-spin blocks, yield index-set-dependent regularity and simultaneous product-moment bounds over all occupied spin blocks. Interpolation with known mixed-Sobolev estimates gives an intermediate Fourier–Lebesgue scale for $1 < p < 2$ , while the hydrogen, helium, and lithium ground states establish the uniform sharpness of every defining inequality of the corresponding parameter regions.

Lemma 2.5 shows that, when $\alpha > 0$ and $n _ { \sigma } > 1$ , the mixed product-weighted space $\mathcal { X } _ { \sigma } ^ { 0 , \alpha , 0 }$ is strictly contained in $B ^ { \alpha } ( \mathbb { R } ^ { 3 N } )$ and thus captures additional regularity that is invisible to the isotropic Barron scale. Another natural question is whether neural-network architectures adapted to this product structure can exploit the additional regularity to achieve sharper approximation rates or lower parameter complexity for a prescribed accuracy than those obtained from isotropic Barron regularity alone.

In the isotropic setting, recent results show that quotients obtained by extracting suitable cut-of Jastrow factors belong to $B ^ { s }$ for every $s < 2$ , and that this range is sharp [2, 8]. A natural question is whether combining cusp extraction with the product weights introduced here yields a corresponding sharp mixed spectral Barron theory.

## Appendix A. Space-comparison lemmas

This appendix contains the proofs of the two space-comparison results used in Section 2. We first prove Lemma 2.5, and then state and prove the two-spin result used after Theorem 2.3.

Proof of Lemma 2.5. For positive continuous weights U and $V ,$

$$
{ \mathcal { F } } L ^ { 1 } ( U ) \hookrightarrow { \mathcal { F } } L ^ { 1 } ( V ) \quad \iff \quad V ( \xi ) \le C U ( \xi ) \quad ( \xi \in \mathbb { R } ^ { 3 N } )\tag{A.1}
$$

for some $C > 0$ . The pointwise bound implies the embedding. Conversely, assume the embedding estimate, fix $\eta \in  { \mathbb { R } } ^ { 3 N }$ , and choose a nonnegative $\varphi \in C _ { c } ^ { \infty } (  { \mathbb { R } } ^ { 3 N } )$ with $\begin{array} { r } { \int _ {  { \mathbb { R } } ^ { 3 N } } \varphi ( \xi ) \ \mathrm { d } \xi = 1 } \end{array}$ . Set

$$
g _ { \varepsilon } ( \xi ) = \varepsilon ^ { - 3 N } \varphi \bigg ( \frac { \xi - \eta } { \varepsilon } \bigg ) , \qquad f _ { \varepsilon } = \mathcal { F } ^ { - 1 } g _ { \varepsilon } .
$$

Since $f _ { \varepsilon } \in { \mathcal { S } } ( \mathbb { R } ^ { 3 N } )$ , the embedding estimate gives

$$
\int _ { \mathbb { R } ^ { 3 N } } V ( \xi ) g _ { \varepsilon } ( \xi ) \ \mathrm { d } \xi \leq C \int _ { \mathbb { R } ^ { 3 N } } U ( \xi ) g _ { \varepsilon } ( \xi ) \ \mathrm { d } \xi .
$$

Letting $\varepsilon \downarrow 0$ and using continuity yields $V ( \eta ) \leq C U ( \eta )$ , proving (A.1).

Write

$$
W _ { \sigma , \alpha } ( \xi ) = \sum _ { I \in \mathcal { Z } _ { \sigma } } p _ { I } ( \xi ) ^ { \alpha } , \qquad p _ { I } ( \xi ) = \prod _ { i \in I } \langle \xi _ { i } \rangle .
$$

This is the weight defining the sum norm of $\mathcal { X } _ { \sigma } ^ { 0 , \alpha , 0 }$ . Since $\langle \xi _ { i } \rangle \leq \langle \xi \rangle$ ，

$$
W _ { \sigma , \alpha } ( \xi ) \leq \sum _ { I \in \mathbb { Z } _ { \sigma } } \langle \xi \rangle ^ { \alpha | I | } \leq | \mathbb { Z } _ { \sigma } | \langle \xi \rangle ^ { \alpha n _ { \sigma } } .
$$

Hence $B ^ { t } \hookrightarrow \mathcal { X } _ { \sigma } ^ { 0 , \alpha , 0 }$ whenever $t \geq \alpha n _ { \sigma }$

Since $\mathcal { T } _ { \sigma }$ partitions $\{ 1 , \ldots , N \}$ 2

$$
\begin{array} { l } { \displaystyle \langle \xi \rangle ^ { \alpha } = \left( 1 + \sum _ { I \in \mathcal { T } _ { \sigma } } \sum _ { i \in I } \lvert \xi _ { i } \rvert ^ { 2 } \right) ^ { \alpha / 2 } } \\ { \displaystyle \qquad \leq \left( \sum _ { I \in \mathcal { T } _ { \sigma } } p _ { I } ( \xi ) ^ { 2 } \right) ^ { \alpha / 2 } \leq \sum _ { I \in \mathcal { T } _ { \sigma } } p _ { I } ( \xi ) ^ { \alpha } = W _ { \sigma , \alpha } ( \xi ) , } \end{array}
$$

where the second inequality uses the subadditivity of $r \mapsto r ^ { \alpha / 2 }$ . For $\alpha = 0$ , the conclusion follows from $1 \leq | \mathcal { T } _ { \sigma } | = W _ { \sigma , 0 }$ . Thus $\mathcal { X } _ { \sigma } ^ { 0 , \alpha , 0 } \hookrightarrow B ^ { t }$ whenever $0 \leq t \leq \alpha$

We prove the necessity of both conditions. Choose $I _ { \ast } \in \mathcal { T } _ { \sigma }$ with $| I _ { * } | = n _ { \sigma }$ and a unit vector $e \in \mathbb { R } ^ { 3 }$ . For $R > 0$ , let $\xi _ { i } ^ { ( R ) } = R e \mathrm { i f } \ i \in I _ { * }$ and $\xi _ { i } ^ { ( R ) } = 0$ otherwise. Then

$$
W _ { \sigma , \alpha } ( \xi ^ { ( R ) } ) = ( 1 + R ^ { 2 } ) ^ { \alpha n _ { \sigma } / 2 } + | \mathcal { Z } _ { \sigma } | - 1 , \qquad \langle \xi ^ { ( R ) } \rangle ^ { t } = ( 1 + n _ { \sigma } R ^ { 2 } ) ^ { t / 2 } .
$$

If $B ^ { t } \hookrightarrow \mathcal { X } _ { \sigma } ^ { 0 , \alpha , 0 }$ , criterion $\mathrm { ( A . 1 ) }$ implies $W _ { \sigma , \alpha } \leq C \langle \cdot \rangle ^ { t }$ . The preceding ray then forces $t \geq \alpha n _ { \sigma }$

Next fix $k \in \{ 1 , \ldots , N \}$ and let $\xi _ { k } ^ { ( R ) } = R e \mathrm { ~ a n d ~ } \xi _ { i } ^ { ( R ) } = 0 \mathrm { ~ f o r ~ } i \neq k$ . Along this ray,

$$
W _ { \sigma , \alpha } ( \xi ^ { ( R ) } ) = ( 1 + R ^ { 2 } ) ^ { \alpha / 2 } + | \mathcal { Z } _ { \sigma } | - 1 , \qquad \langle \xi ^ { ( R ) } \rangle ^ { t } = ( 1 + R ^ { 2 } ) ^ { t / 2 } .
$$

If $\mathcal { X } _ { \sigma } ^ { 0 , \alpha , 0 } \hookrightarrow B ^ { t }$ , the criterion implies $\langle \xi \rangle ^ { t } \leq C W _ { \sigma , \alpha } ( \xi )$ and hence $t \leq \alpha$ . This proves both equivalences in (2.8).

If $\alpha > 0$ and $n _ { \sigma } > 1$ , the same two rays exclude the reverse of each endpoint embedding. All the weighted Fourier $L ^ { 1 }$ spaces involved are Banach spaces; equality as sets would therefore make the inverse identity continuous by the open mapping theorem. Both endpoint inclusions are consequently strict. If $\alpha = 0$ or $n _ { \sigma } = 1$ , the two endpoint bounds above give $W _ { \sigma , \alpha } \asymp \langle \xi \rangle ^ { \alpha }$ , and the corresponding spaces coincide with equivalent norms. This completes the proof. □

Lemma A.1 (Minimal mixed Barron space for two spin states). Suppose that $\sigma \in \{ 1 , 2 \} ^ { N }$ allowing either spin block to be empty.<sup>1</sup> For $s , \alpha , \beta \geq 0$ , set $\tau = s + \alpha + \beta$ . Then

$$
\begin{array} { r l r } { \chi _ { \sigma } ^ { 0 , \tau , 0 } \hookrightarrow \chi _ { \sigma } ^ { s , \alpha , \beta } , \qquad } & { { } \parallel f \parallel _ { \chi _ { \sigma } ^ { s , \alpha , \beta } } \le | \mathbb { Z } _ { \sigma } | N ^ { s / 2 } \parallel f \parallel _ { \chi _ { \sigma } ^ { 0 , \tau , 0 } } . } \end{array}\tag{A.2}
$$

Thus $\mathcal { X } _ { \sigma } ^ { 0 , \tau , 0 }$ is the least member, under continuous inclusion, of the family with fixed total order τ. If both spin blocks are occupied, then $\mathcal { X } _ { \sigma } ^ { 0 , \tau , 0 } = \mathcal { X } _ { \sigma } ^ { 0 , 0 , \tau }$ with equal norms.

Proof of Lemma A.1. Suppose first that only one spin block $I _ { 1 }$ is occupied, and set $\begin{array} { r } { p _ { 1 } ( \xi ) = \prod _ { i \in I _ { 1 } } \langle \xi _ { i } \rangle } \end{array}$ . Then $I _ { 1 } = \{ 1 , \ldots , N \}$ and $\langle \xi \rangle \overset { \cdot } { \leq } N ^ { 1 / 2 } p _ { 1 } ( \xi )$ . Since $s + \alpha \leq \tau$ and $p _ { 1 } ( \xi ) \ge 1$ ，

$$
\begin{array} { r } { \langle \xi \rangle ^ { s } p _ { 1 } ( \xi ) ^ { \alpha } \leq N ^ { s / 2 } p _ { 1 } ( \xi ) ^ { s + \alpha } \leq N ^ { s / 2 } p _ { 1 } ( \xi ) ^ { \tau } . } \end{array}
$$

Integration against $| \widehat { f } |$ proves (A.2) in this case.

Suppose next that both spin blocks $I _ { 1 } , I _ { 2 }$ are occupied, and set $\begin{array} { r } { p _ { \ell } ( \xi ) = \prod _ { i \in I _ { \ell } } \langle \xi _ { i } \rangle } \end{array}$ for $\ell = 1 , 2 .$ . Since the two blocks partition $\{ 1 , \ldots , N \}$

$$
\langle \xi \rangle \leq N ^ { 1 / 2 } \operatorname* { m a x } \{ p _ { 1 } ( \xi ) , p _ { 2 } ( \xi ) \} .
$$

Using $s + \alpha + \beta = \tau$ gives

$$
\langle \xi \rangle ^ { s } \big ( p _ { 1 } ( \xi ) ^ { \alpha } p _ { 2 } ( \xi ) ^ { \beta } + p _ { 2 } ( \xi ) ^ { \alpha } p _ { 1 } ( \xi ) ^ { \beta } \big ) \leq 2 N ^ { s / 2 } \operatorname* { m a x } \{ p _ { 1 } ( \xi ) , p _ { 2 } ( \xi ) \} ^ { \tau } \leq 2 N ^ { s / 2 } \sum _ { \ell = 1 } ^ { 2 } p _ { \ell } ( \xi ) ^ { \tau } .
$$

Integration proves (A.2). Since the triple $( 0 , \tau , 0 )$ belongs to the fixed-total-order family and its space embeds into every other member, it is the least member under continuous inclusion.

When both blocks are occupied, the weights of $\mathcal { X } _ { \sigma } ^ { 0 , \tau , 0 }$ and $\mathcal { X } _ { \sigma } ^ { 0 , 0 , \tau }$ are, respectively, $p _ { 1 } ( \xi ) ^ { \tau } + p _ { 2 } ( \xi ) ^ { \tau }$ and $p _ { 2 } ( \xi ) ^ { \tau } + p _ { 1 } ( \xi ) ^ { \tau }$ , so they coincide. When only one block is occupied, no complementary-coordinate weight occurs. This completes the proof. □

## References

[1] Brascamp, H.J., Lieb, E.H., Luttinger, J.M.: A general rearrangement inequality for multiple integrals. J. Funct. Anal. 17(2), 227–237 (1974)

[2] Ehrlacher, V.: Cut-of Jastrow factors and spectral Barron regularity of Coulombic electronic wave functions. arXiv:2607.02492 (2026)

[3] Fournais, S., Hofmann-Ostenhof, M., Hofmann-Ostenhof, T., Sørensen, T.Ø.: Analytic structure of many-body Coulombic wave functions. Commun. Math. Phys. 289(1), 291–310 (2009)

[4] Folland, G.B.: Real Analysis: Modern Techniques and Their Applications, 2nd edn. Wiley-Interscience, New York (1999)

[5] Kreusler, H.-C., Yserentant, H.: The mixed regularity of electronic wave functions in fractional order and weighted Sobolev spaces. Numer. Math. 121(4), 781–802 (2012)

[6] Meng, L.: On the mixed regularity of N-body Coulombic wavefunctions. ESAIM Math. Model. Numer. Anal. 57(4), 2257–2282 (2023)

[7] Ming, P., Yu, H.: Barron regularity of many particle Schrödinger eigenfunctions. arXiv:2508.17722 (2025)

[8] Ming, P., Yu, H.: Sharp Barron regularity results for Coulombic many-electron wave functions. arXiv:2608.22252 (2026)

[9] Simon, B.: Schrödinger semigroups. Bull. Am. Math. Soc. 7(3), 447–526 (1982)

[10] Simon, B.: Tosio Kato’s work on non-relativistic quantum mechanics, Part 2. Bull. Math. Sci. 9(1), 1950005 (2019)

[11] Yserentant, H.: The mixed regularity of electronic wave functions multiplied by explicit correlation factors. ESAIM Math. Model. Numer. Anal. 45(5), 803–824 (2011)

[12] Yserentant, H.: The regularity of electronic wave functions in Barron spaces. ESAIM Math. Model. Numer. Anal. 60(2), 689–699 (2026)

[13] Zhislin, G.M.: A study of the spectrum of the Schrödinger operator for a system of several particles. Trudy Moskovskogo Matematicheskogo Obshchestva 9, 81–120 (1960). In Russian

SKLMS, Institute of Computational Mathematics and Scientific/Engineering Computing, Academy of Mathematics and Systems Science<sub>,</sub> Chinese Academy of Sciences<sub>,</sub> Beijing 100190<sub>,</sub> China

School of Mathematical Sciences<sub>,</sub> University of Chinese Academy of Sciences<sub>,</sub> Beijing 100049<sub>,</sub> China