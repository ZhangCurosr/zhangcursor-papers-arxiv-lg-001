# Fractional is Better: Learnable Derivative Orders in Neural Operator Learning<sup>∗</sup>

Fares B. Mehouachi<sup>1,∗</sup> fares.mehouachi@nyu.edu

Saif Eddin Jabari<sup>1,2</sup> sej7@nyu.edu

<sup>1</sup>New York University Abu Dhabi, Abu Dhabi, UAE

<sup>2</sup>New York University, New York, NY, USA

## Abstract

Neural operators learn mappings between function spaces, enabling fast surrogate solutions to partial differential equations. Despite remarkable architectural diversity, these methods often share a common input representation: raw coordinate-value pairs $( x , u ( x ) )$ . We ask whether inputs aligned with PDE differential structure can improve learning. Through Picard iteration on mild solutions, we show that derivatives of the input appear explicitly in the solution operator, suggesting that providing derivative features should reduce the network’s implicit differentiation burden. We prove this intuition: providing derivative features improves approximation bounds from $O ( W ^ { - ( s - m ) / d } )$ to $O ( W ^ { - s / d } )$ , where W is parameter count, s is input regularity, m is the PDE order, and d is spatial dimension. Our central finding, however, is a surprise: the optimal derivative order $\beta ^ { * }$ is strictly less than the PDE order m. This gap arises from a bias-variance tradeoff in spectral space that we characterize in closed form. Learning β from data achieves automatic spectral regularization. We introduce ∂-NO (del-NO), for derivative-augmented neural operators, an architecture-agnostic augmentation that provides learnable fractional derivative features to any neural operator backbone. Across benchmark problems and architectures, ∂-NO consistently improves prediction accuracy, with learned orders that reflect a representation of known physics modulated by noise and finite-sample constraints.

## 1 Introduction

Neural operators have emerged as a powerful tool for scientific computing. By learning mappings between function spaces from data, they enable fast surrogate solutions to partial differential equations, with applications spanning climate modeling, materials science, and engineering design [1–5]. The field has witnessed rapid architectural innovation: spectral convolutions [1], attention mechanisms [6, 7], physics-aware designs [8], and latent representations [9, 10]. These advances have substantially improved accuracy and efficiency. Yet beneath this diversity lies a striking uniformity. Virtually all neural operators represent their inputs in the same way:

$$
{ \mathrm { 1 D : ~ } } ( x , u ( x ) ) , \quad { \mathrm { 2 D : ~ } } ( x , y , u ( x , y ) ) .\tag{1}
$$

This representation ignores the differential structure that defines the underlying physics. Partial differential equations are built from derivatives: $\partial _ { x } u , \partial _ { x x } u .$ , and their combinations. A neural operator receiving only function values must learn to differentiate implicitly from finite, noisy samples. This places an unnecessary burden on the network.

![](images/97ac505520e32157889f656a0a675cf1acb2e289dcfdd8adf3c06d51df5c3a03.jpg)  
Figure 1: ∂-NO architecture.<sup>1</sup> The input function u is augmented with learnable fractional derivatives ${ \mathfrak { D } } ^ { { \bar { \beta } } _ { k } } u ,$ where each order $\beta _ { k }$ and scale $s _ { k }$ is optimized jointly with backbone weights during training. This simple augmentation wraps any neural operator backbone without modification.

This paper asks a simple question: Can we provide inputs that align with PDE differential structure? And if so, what is the best way to do it?

Our answer proceeds in three steps. First, we analyze the structure of PDE solution operators through Picard iteration on mild solutions. This reveals that derivatives of the input function appear explicitly in the solution operator, so providing them as features reduces the implicit differentiation burden on the network (§3). Second, we prove that providing these features improves approximation bounds by a factor depending on the PDE order (§4). Third, and most unexpectedly, we uncover an operator-level bias-variance tradeoff: the optimal derivative order $\beta ^ { * }$ satisfies

$$
\boxed { \beta ^ { * } < m }\tag{2}
$$

for any practically realizable finite, noisy dataset (§5). Using the exact PDE order m introduces estimation variance that outweighs the approximation benefit. The gap $m - \beta ^ { * }$ shrinks as data grows, but remains positive at every practically realizable scale.

This finding has important practical implications. It means derivative features should not simply match known physics, but should balance physical fidelity against statistical efficiency. Since this optimal order depends on unknown quantities like noise level and sample size, we propose learning it directly from data.

The resulting architecture, which we call ∂-NO (del-NO) for derivative-augmented neural operators (Figure 1; §6), augments any backbone with fractional derivative features whose orders are jointly optimized during training. The method is simple to implement, adds modest overhead, and wraps existing architectures without modification.

Where does ∂-NO sit among existing uses of fractional calculus in deep learning? At least three established lines exist: fractional PINNs include fractional operators directly in the loss [11–14]; fractional gradient methods replace integer gradients with fractional ones during optimization [15– 17]; andfractional architectures embed fractional dynamics into the network itself [18–21]. ∂-NO occupies a distinct fourth role: learnable fractional derivative input features applied as preprocessing, with no modification to equation, optimizer, or architecture.

Across benchmark problems and backbone choices, ∂-NO consistently improves prediction accuracy (§7). The learned orders reflect a representation of known physics modulated by noise and finitesample constraints, providing both better performance and interpretable insight into each problem’s differential structure.

## 2 Background

## 2.1 Neural Operators

Let $\mathcal { G } : \mathcal { U }  \mathcal { V }$ denote the true solution operator between function spaces. A neural operator $\mathcal { G } _ { \theta }$ is its finite-dimensional approximation parameterized by $\theta ,$ learned from paired examples $\{ ( u _ { i } , v _ { i } ) \} _ { i = 1 } ^ { n }$ with $v _ { i } = \mathcal { G } [ u _ { i } ]$ by minimizing the empirical risk:

$$
\operatorname* { m i n } _ { \theta } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \| \mathcal { G } _ { \theta } [ u _ { i } ] - v _ { i } \| _ { L ^ { 2 } } ^ { 2 } .\tag{3}
$$

The target operator is typically a PDE solution map, taking initial or boundary conditions to the corresponding solution.

Different architectures approach this problem from different angles. The Fourier Neural Operator [1] performs convolutions efficiently in frequency space. DeepONet [2] factorizes the operator into branch and trunk networks. More recent work explores attention mechanisms [7, 8, 22] and implicit representations [23]. What these methods share is their input representation: they receive the input function sampled at spatial locations, either as coordinate-value pairs or on a regular grid.

## 2.2 A Concrete Example: Burgers’ Equation

To make the limitation concrete, consider the viscous Burgers’ equation:

$$
\partial _ { t } u + u \partial _ { x } u = \nu \partial _ { x x } u .\tag{4}
$$

The solution operator $\mathcal { G } : u ( \cdot , 0 ) \mapsto u ( \cdot , T )$ maps initial conditions to solutions at time $T .$ . This operator depends on the initial condition through two mechanisms: the diffusive term $\nu \partial _ { x x } u$ smooths the solution according to second derivatives, while the convective term $u \partial _ { x } u$ transports features at speeds determined by first derivatives. A standard neural operator must learn both differentiation operations from data.

Recent work has explored providing precomputed derivatives as additional inputs to neural operators [24]. This raises a natural question: which derivatives should we provide? When the PDE is known, one might supply $\partial _ { x } u$ and $\partial _ { x x } u$ to match the equation structure. But what if the governing physics is unknown or only partially specified? And even when known, is matching the PDE order necessarily optimal?

These considerations motivate a different approach: rather than prescribing derivative orders, let the network learn them from data. To enable this, we need derivatives defined for non-integer orders.

## 2.3 Fractional Derivatives

To enable optimization over derivative orders, we need a continuous parameterization. Fractional calculus provides exactly this.

Definition 2.1 (Fractional Derivative). For order $\beta \geq 0$ and function $u \in H ^ { s } ( \Omega )$ with $s > \beta$ , we define the Weylfractional derivative through the Fourier transform:

$$
D ^ { \beta } u : = \mathcal { F } ^ { - 1 } \left[ ( i \xi ) ^ { \beta } \hat { u } ( \xi ) \right] ,\tag{5}
$$

where $\hat { u } ( \xi ) = \mathcal { F } [ u ] ( \xi )$ denotes the Fourier transform and $( i \xi ) ^ { \beta } = | \xi | ^ { \beta } e ^ { i ( \pi / 2 ) \beta \mathrm { s g n } ( \xi ) }$

The Weyl derivative acts as a frequency filter weighting mode ξ by $( i \xi ) ^ { \beta }$ , preserving phase information that distinguishes odd and even derivatives. When $\bar { \beta } = 1$ , we recover $\partial _ { x } .$ ; when $\beta = 2$ , we get $\partial _ { x x }$ Intermediate values interpolate smoothly. For non-periodic domains or to avoid Fourier artifacts (Gibbs ringing), we use the Grunwald-Letnikov finite-difference formulation (Appendix¨ H).

Crucially, the definition is differentiable with respect to $\beta \colon$

$$
\frac { \partial } { \partial \beta } ( i \xi ) ^ { \beta } = ( i \xi ) ^ { \beta } \left( \log | \xi | + i \frac { \pi } { 2 } \mathrm { s g n } ( \xi ) \right) .\tag{6}
$$

This enables gradient-based learning of derivative orders.

## 3 Picard Iteration and Derivative Emergence

Why should derivative features help neural operators? The answer emerges from analyzing how PDE solutions depend on initial conditions through Picard iteration.

## 3.1 Mild Solutions and the Green’s Function

Consider a semilinear PDE of the form:

$$
\partial _ { t } u = \mathcal { L } u + \mathcal { N } [ u ] , \quad u ( 0 ) = u _ { 0 } ,\tag{7}
$$

where $\mathcal { L }$ is a linear differential operator of order m and $\mathcal { N }$ is a nonlinearity (polynomial or quasilinear). The mild solution satisfies the integral equation:

$$
u ( t ) = G _ { t } \ast u _ { 0 } + \int _ { 0 } ^ { t } G _ { t - s } \ast \mathcal { N } [ u ( s ) ] d s ,\tag{8}
$$

where $G _ { t }$ is the Green’s function (semigroup) generated by $\mathcal { L } .$ For constant-coefficient operators, this takes the explicit form $G _ { t } = e ^ { t \mathcal { L } }$

## 3.2 The Picard Mechanism

Picard iteration constructs solutions by successive approximation. Starting from $u ^ { ( 0 ) } ( t ) = G _ { t } * u _ { 0 }$ ， each iterate refines the approximation:

$$
u ^ { ( k + 1 ) } ( t ) = G _ { t } \ast u _ { 0 } + \int _ { 0 } ^ { t } G _ { t - s } \ast \mathcal { N } [ u ^ { ( k ) } ( s ) ] d s .\tag{9}
$$

The key observation is that the first iterate already reveals essential structure.

Expanding the semigroup for small times, $G _ { s } * u _ { 0 } \approx u _ { 0 } + s \mathcal { L } u _ { 0 } + O ( s ^ { 2 } )$ , and substituting into the nonlinearity:

$$
\begin{array} { r } { \mathcal { N } [ G _ { s } * u _ { 0 } ] \approx \mathcal { N } [ u _ { 0 } ] + s \cdot D \mathcal { N } [ u _ { 0 } ] [ \mathcal { L } u _ { 0 } ] + O ( s ^ { 2 } ) , } \end{array}\tag{10}
$$

where $D \mathcal { N } [ u _ { 0 } ] [ v ]$ is the Frechet derivative of ´ $\mathcal { N }$ at $u _ { 0 }$ applied to direction v. Evaluating the integral yields:

Theorem 3.1 (Derivative Emergence, Parabolic Case). Let $\mathcal { G } : u _ { 0 } \mapsto u ( T )$ be the solution operator for a semilinear parabolic PDE with polynomial nonlinearity N of degree N and linear operator L of differential order m. For $u _ { 0 } \in B _ { s } ( R )$ , there exists $T _ { * } \dot { ( } R ) = ( \check { L ( R ) } ) ^ { - 1 } ;$ , where $L ( \bar { R } )$ is the Lipschitz constant ofN on $B _ { s } ( R )$ , such thatfor all $T < T _ { * } ( \dot { R } )$

$$
\mathcal { G } [ u _ { 0 } ] = G _ { T } * u _ { 0 } + T \cdot \bar { G } _ { T } * \mathcal { N } [ u _ { 0 } ] + \mathcal { R } ( u _ { 0 } , T ) ,\tag{11}
$$

where $\begin{array} { r } { \begin{array} { r } { \bar { G } _ { T } \ = \ T ^ { - 1 } \int _ { 0 } ^ { T } G _ { T - s } } \end{array} } \end{array}$ ds is an averaged Green’s function, and the remainder satisfies $\Vert \mathcal { R } ( u _ { 0 } , T ) \Vert _ { L ^ { 2 } } \leq C ( R ) T ^ { 2 }$ uniformly over $u _ { 0 } \in B _ { s } ( R )$ . The nonlinearity $\mathcal { N } [ u _ { 0 } ]$ involves derivatives of u<sub>0</sub> up to order m and powers up to $u _ { 0 } ^ { N }$

The full proof appears in Appendix B. The essential point is that derivatives enter through two mechanisms: the linear operator $\mathcal { L }$ directly differentiates $u _ { 0 }$ , while the nonlinearity $\mathcal { N }$ combines powers and derivatives according to its structure.

Remark 3.2 (Beyond Short Times). The derivative dependence persists for arbitrary times via iterated Picard and extends to stationary PDEs (e.g., Darcy flow) and quasilinear systems (e.g., Navier-Stokes) where derivatives appear directly. See Appendix B.2.

Remark 3.3 (Beyond Parabolic Evolution). A complementary spectral resolvent argument $( \mathsf { A p } \cdot \mathsf { \Gamma }$ pendix B.3) extends the same derivative structure to general evolution equations $\partial _ { t } ^ { a } u = \mathcal { L } u$ with temporal order $a > 0$ , recovering parabolic $( a { = } 1 )$ , hyperbolic $\scriptstyle ( a = 2 )$ , and fractional-in-time (Mittag-Leffler) cases uniformly. The temporal order a determines the number of initial conditions ⌈a⌉ but not the derivative-feature structure, which is set by the spatial order m.

Example 3.4 (Burgers Features). For Burgers’ equation, $\mathcal { L } = \nu \partial _ { x x }$ and $\mathcal { N } [ u ] = - u \partial _ { x } u$ . The solution operator depends on:

$$
\{ u _ { 0 } , \partial _ { x } u _ { 0 } , \partial _ { x x } u _ { 0 } , u _ { 0 } \cdot \partial _ { x } u _ { 0 } \} .\tag{12}
$$

The first three are derivatives; the fourth emerges from the convective nonlinearity. In this work, we focus on the derivative terms $\{ u _ { 0 } , \partial _ { x } u _ { 0 } , \partial _ { x x } \overline { { u _ { 0 } } } \}$ , leaving the nonlinear product $u _ { 0 } \cdot \partial _ { x } u _ { 0 }$ to the network. This differs from equation-aware approaches that encode full nonlinear structure $[ 2 4 ] ;$ our method requires only the PDE order $m ,$ not equation-specific terms.

## 4 Approximation Theory: Why Features Help

We now prove that derivative features admit a better approximation bound. This section establishes that providing derivative features is beneficial; the optimal choice of derivative order is deferred to Section 5. The improvement quantifies how much easier the learning problem becomes when inputs align with PDE structure.

## 4.1 The Feature Map

Definition 4.1 (Feature Maps). We define two related feature maps. For a single derivative order $\beta \geq 0$ , the β-feature map $\Phi ^ { \bar { \beta } } : H ^ { s } ( \Omega ) \to H ^ { s } \times H ^ { s - \beta }$ is:

$$
\begin{array} { r } { \Phi ^ { \beta } [ u ] = ( u , D ^ { \beta } u ) . } \end{array}\tag{13}
$$

For the full set of integer derivatives up to order $m ,$ , the full feature map $\Phi ^ { [ m ] } : H ^ { s } ( \Omega )  \mathcal { X }$ is:

$$
\Phi ^ { [ m ] } [ u ] = ( u , D ^ { 1 } u , D ^ { 2 } u , \ldots , D ^ { m } u ) ,\tag{14}
$$

where $\mathcal { X } = H ^ { s } \times H ^ { s - 1 } \times \cdot \cdot \cdot \times H ^ { s - m }$ with the product norm. When context is clear, we write Φ for either map.

The feature map bundles the input function with its derivatives. Since Φ includes u as its first component, no information is lost.

Assumption 4.2 (Regularity). Inputs lie in a Sobolev ball: $u \in B _ { s } ( R ) : = \{ u \in H ^ { s } ( \Omega ) : \| u \| _ { H ^ { s } } \leq $ R} with $s > m + d / 2$

Proposition 4.3 (Stability). Under Assumption 4.2, the feature map $\Phi : B _ { s } ( R ) \to \mathcal { X }$ is bi-Lipschitz:

$$
\begin{array} { r } { \| u _ { 1 } - u _ { 2 } \| _ { H ^ { s } } \leq \| \Phi [ u _ { 1 } ] - \Phi [ u _ { 2 } ] \| _ { \mathcal { X } } \leq \sqrt { 1 + m } \cdot \| u _ { 1 } - u _ { 2 } \| _ { H ^ { s } } . } \end{array}\tag{15}
$$

The upper constant $\sqrt { 1 + m }$ is independent of the ball radius R.

The lower bound is immediate since Φ preserves u. The upper bound follows from the continuity of differentiation in Sobolev spaces. See Appendix C for the full proof.

## 4.2 Main Approximation Result

Theorem 4.4 (Approximation Rates). Let $\mathcal { G } : B _ { s } ( R )  L ^ { 2 } ( \Omega )$ be the solution operatorfor a PDE of differential order m. Denote by $\mathcal { N O } _ { W }$ the class ofneural operators with W parameters.

(i) Without features:

$$
\operatorname* { i n f } _ { \mathcal { G } _ { \theta } \in N \mathcal { O } _ { W } } \operatorname* { s u p } _ { u \in B _ { s } ( R ) } \| \mathcal { G } [ u ] - \mathcal { G } _ { \theta } ( u ) \| _ { L ^ { 2 } } = O \left( W ^ { - \frac { s - m } { d } } \right) .\tag{16}
$$

(ii) With full features $\Phi ^ { [ m ] } .$

$$
\operatorname* { i n f } _ { \mathscr { G } _ { \theta } \in N \mathscr { O } _ { W } } \operatorname* { s u p } _ { u \in B _ { s } ( R ) } \left\| \mathscr { G } [ u ] - \mathscr { G } _ { \theta } ( \Phi ^ { [ m ] } [ u ] ) \right\| _ { L ^ { 2 } } = O \left( W ^ { - \frac { s } { d } } \right) .\tag{17}
$$

Proof. Both bounds are achievability results (upper bounds on the minimax error). Full details appear in Appendix D, with the scope of the comparison between (i) and (ii) discussed in Remark D.2.

Part (i): The operator $\mathcal { G } : B _ { s } ( R ) \to L ^ { 2 }$ has modulus of continuity $\omega ( { \varepsilon } ) \leq C ( R ) \varepsilon ^ { ( s - m ) / s }$ when measured in $L ^ { 2 }$ . This follows from Sobolev interpolation: for $u _ { 1 } , u _ { 2 } \in B _ { s } ( R )$ with $\| u _ { 1 } - u _ { 2 } \| _ { L ^ { 2 } } \leq \varepsilon .$ we have $\| u _ { 1 } - u _ { 2 } \| _ { H ^ { m } } \lesssim \varepsilon ^ { ( s - m ) / s } R ^ { m / s }$ . The Holder exponent¨ $( s - m ) /$ s arises because inputs lie in $H ^ { s }$ but the operator depends on m-th order derivatives: the “missing” regularity m out of total s determines the continuity loss. The ϵ-covering number of $B _ { s } ( R )$ in $L ^ { 2 }$ satisfies log $\mathcal { N } ( \epsilon ) \asymp ( R / \epsilon ) ^ { d / s }$ [25]. Applying the approximation theory of Yarotsky [26] for Holder-continuous functions on sets ¨ with this metric entropy yields the rate $\dot { W } ^ { - ( s - m ) / d }$

Part (ii): Define the lifted operator $\tilde { \mathcal { G } } [ \Phi ^ { [ m ] } [ u ] ] : = \mathcal { G } [ u ]$ . On bounded balls, $\tilde { \mathcal { G } }$ is Lipschitz: since the features $\Phi ^ { [ m ] } [ u ] = ( u , D ^ { 1 } u , \dots , D ^ { \dot { m } } u )$ provide the derivatives explicitly, the operator no longer needs to “invert” differentiation, an unbounded operation that was the source of the Holder dis-¨ continuity in Part (i). By Theorem 3.1, the solution operator depends on u and its derivatives through polynomial combinations; polynomial functions are Lipschitz on bounded balls with constant $L ( R ) { \bf \bar { \Psi } } = { \bf \dot { \cal O } } { \bf \dot { ( } } R ^ { N - 1 } )$ for degree-N nonlinearity. By Proposition 4.3, the feature map is bi-Lipschitz, so metric entropy is preserved up to constants. Applying Yarotsky [26] to the Lipschitz function $\tilde { \mathcal { G } }$ achieves rate $W ^ { - s / d }$ □

The improvement in the bound, from $( s - m ) / d { \mathrm { t o } } s / d ,$ is substantial. For a second-order PDE in one dimension with $s = 3$ , the exponent improves from 1 to $^ { 3 , }$ meaning the same accuracy requires far fewer parameters.

## 5 The Statistical Twist: Why $\beta ^ { * } < m$

Theorem 4.4 suggests using the true PDE order m as the derivative order. We now prove this is suboptimal for noisy, finite samples. This finding is our central theoretical contribution.

Remark 5.1 (Complementary Theoretical Perspectives). Section 4 and this section answer different questions. Section 4 asks: “Do derivative features help?” and answers via approximation theory. This section asks: “Given that features help, what order is optimal?” and answers via spectral bias-variance analysis. These perspectives are complementary: one justifies using derivative features at all, the other determines the best choice.

## 5.1 The Core Tension

Consider learning an operator that depends on $D ^ { m } u$ using features $D ^ { \beta } u$ with potentially different order $\beta .$ . Two competing effects arise:

Bias: If $\beta < m ,$ the features miss high-frequency information relevant to the target. This creates approximation error that persists regardless of sample size.

Variance: Differentiation amplifies noise. The Fourier multiplier $| \xi | ^ { \beta }$ grows unboundedly with frequency, meaning high-frequency noise gets amplified by factor $| \xi | ^ { \beta }$ when we compute $D ^ { \beta } u$ Higher $\boldsymbol { \beta }$ means more amplification.

These effects respond oppositely to $\beta \colon$ bias decreases while variance increases. The optimal choice balances them.

## 5.2 Spectral Analysis

To make this tradeoff precise, we assume inputs have a typical spectral structure.

Assumption 5.2 (Spectral Decay). The input distribution $\mu$ on $B _ { s } ( R )$ has power spectral density $S ( \xi ) : = \mathbb { E } [ | \hat { u } ( \xi ) | ^ { 2 } ] = C _ { S } | \xi | ^ { - 2 s - d }$ for $1 \leq | \xi | \leq \xi _ { \mathrm { m a x } }$ , where $C _ { S } > 0$ depends on the distribution.

This rate is the borderline upper envelope for $H ^ { s }$ integrability: $\mathbb { E } \Vert u \Vert _ { H } ^ { 2 }$ <sub>s</sub> converges for any $S ( \xi )$ decaying strictly faster than $| \xi | ^ { - 2 s - d }$ and is logarithmically divergent at exactly this rate. We adopt this envelope as the worst-case spectral profile on $B _ { s } ( R )$ , yielding the worst-case bias integral in §5; Matern-´ s fields [27] sit at this edge. The coefficient $C _ { S }$ cancels in the optimal $\beta ^ { * }$ formula (Theorem 5.6), so only the exponent matters.

Proposition 5.3 (Bias-Variance Decomposition). Under Assumption 5.2, suppose the network has effective frequency cutoff $\xi _ { c }$ (depending on capacity W but not on $\beta )$ . Using features $D ^ { \beta }$ u with $\beta \leq m$ , the MSE decomposes as $\mathrm { \dot { M S E } } ( \bar { \beta } ) = \boldsymbol { \dot { B ( \beta ) } } + \boldsymbol { V } ( \beta )$ where:

$$
B i a s \colon B ( \beta ) = C _ { B } \int _ { | \xi | > \xi _ { c } } | \xi | ^ { 2 ( m - \beta ) } S ( \xi ) d \xi ,\tag{18}
$$

$$
V a r i a n c e : \quad V ( \beta ) = \frac { \sigma ^ { 2 } C _ { V } } { n } \int _ { | \xi | \leq \xi _ { \operatorname* { m a x } } } | \xi | ^ { 2 \beta } d \xi ,\tag{19}
$$

where $\xi _ { \operatorname* { m a x } } = \pi / h$ is the Nyquist frequency for grid spacing $h .$

The bias captures the spectral mismatch: providing $D ^ { \beta } u$ instead of $D ^ { m }$ u leaves a residual $| \xi | ^ { m - \beta }$ that the network must learn implicitly. The variance captures noise amplification: computing $\dot { D } ^ { \beta } u$ u amplifies noise atfrequency ξ byfactor $| \xi | ^ { \beta }$

Both integrals admit closedforms (see Appendix E). The bias is strictly decreasing in $\beta .$ . The variance is strictly increasing in $\beta$ provided $\xi _ { \mathrm { m a x } } > e ^ { 1 / ( 2 \hat { \beta } + d ) }$ , which holds for typical discretizations with $\xi _ { \operatorname* { m a x } } \gg 1$ . There exists a unique $\beta ^ { * } \in [ 0 , m ]$ minimizing the total MSE.

Remark 5.4 (Terminology). We use $\mathrm { \hbar \dot { \ b i a s } ^ { \prime \prime } }$ and “variance” by analogy with statistical learning theory. Here, $B ( \beta )$ represents approximation bias from using derivative order $\beta < m \colon$ high-frequency information beyond network capacity cannot be captured. Meanwhile, $V ( \beta )$ represents estimation variance from noise amplification: computing $D ^ { \beta } v$ u from noisy data amplifies observation noise by $| \xi | ^ { \beta }$ at each frequency. This mirrors the classical tradeoff where model complexity reduces bias at the cost of increased variance.

Remark 5.5 (Capacity Cutoff in FNO). For Fourier Neural Operators, the capacity cutoff is well approximated by $\xi _ { c } = 2 \pi k _ { \mathrm { m a x } } / L \mathrm { : }$ FNO truncates to $k _ { \mathrm { m a x } }$ Fourier modes by architectural design, imposing a hard frequency cutoff independent of input features. The preprocessing $D ^ { \beta } u$ changes the input’s spectral content but not the architecture’s representational capacity.

## 5.3 The Optimal Derivative Order

Balancing bias and variance yields our main theoretical result.

Theorem 5.6 (Optimal Derivative Order). Under Assumptions $4 . 2$ and 5.2, define $\tau : = 2 ( s - m ) > 0$ and the capacity constant $A _ { c } : = 2 s$ ln $\xi _ { c } + ( 2 m + d ) \ln \xi _ { \mathrm { m a x } }$ , where $\xi _ { c }$ is the network’sfrequency cutoff and $\xi _ { \operatorname* { m a x } } = \pi / h$ is the Nyquistfrequency. The MSE-optimal derivative order satisfies:

$$
\boxed { \beta ^ { * } = m - \delta ^ { * } , \quad \delta ^ { * } = \cfrac { A _ { c } - \ln ( n / \sigma ^ { 2 } ) } { 2 \ln ( \xi _ { c } \cdot \xi _ { \operatorname* { m a x } } ) } }\tag{20}
$$

up to $O ( 1 / \ln \xi _ { c } )$ corrections, with $\beta ^ { * } \in [ 0 , m ]$

The formula shows that $\delta ^ { * }$ depends on the signal-to-noise ratio $n / \sigma ^ { 2 } ;$ as data quality improves, the gap shrinks and $\beta ^ { * } \to m$ . The capacity constant $A _ { c }$ captures network architecture effects. The full derivation appears in Appendix F.

Remark 5.7 (Data Sufficiency for Exact Physics). To leading order in 1/ ln $\xi _ { c } , \delta ^ { * } > 0$ strictly whenever $n / \bar { \sigma } ^ { 2 } < e ^ { A _ { c } }$ , with exact-physics recovery requiring

$$
\begin{array} { r } { n ^ { \star } = \sigma ^ { 2 } \exp ( A _ { c } ) . } \end{array}\tag{21}
$$

The exponential dependence makes $n ^ { \star }$ astronomical in practice. For a representative FNO setting (unit-interval Burgers, $N { = } 1 0 2 4 , k _ { \mathrm { m a x } } { = } 1 6$ , giving $\xi _ { \operatorname* { m a x } } \approx 3 2 0 0 , \xi _ { c } \approx 1 0 0 , A _ { c } \approx 6 8 )$ , even at $\sigma = 1 0 ^ { - 3 }$ this implies $n ^ { \star } \approx 1 0 ^ { 2 3 }$ samples (a leading-order figure, shifted by the unidentified constant κ of App. F.4), far beyond any realistic training budget. Sub-physical regularization is therefore the operational regime at any practically achievable scale.

These results have three key implications:

1. Strict suboptimality of m: For any finite sample size n and noise level $\sigma > 0$ with $n / \sigma ^ { 2 } < e ^ { A _ { c } }$ the optimal order satisfies $\beta ^ { * } < m$ strictly. Below this threshold, the exact PDE order is never optimal.

2. Asymptotic recovery: $\operatorname { A s } n \to \infty \left( \operatorname { o r } n \geq n ^ { \star } \right)$ or $\sigma \to 0$ , the gap $\delta ^ { * } \to 0$ and $\beta ^ { * } \to m$ . Physics is recovered in the large-data limit.

3. Capacity-adaptive: Networks with higher frequency cutoff $\xi _ { c } \left( \mathrm { e . g . } \right.$ ., more Fourier modes in FNO) benefit from lower $\beta ^ { * }$ , automatically trading approximation power for regularization.

## 6 ∂-NO: Derivative-Augmented Neural Operators

Our theoretical analysis motivates a simple architecture: neural operators with learnable fractional derivative features (Figure 1).

Given a base neural operator $\mathcal { N O } _ { \theta }$ , the $\partial { \cdot } \mathbf { N O }$ augmentation replaces inputs $( x , u )$ with:

$$
\begin{array} { r l } { \partial \mathbf { - N O } : } & { { } \left( x , u , s _ { 1 } \mathfrak { D } ^ { \beta _ { 1 } } u , s _ { 2 } \mathfrak { D } ^ { \beta _ { 2 } } u , \dots , s _ { k } \mathfrak { D } ^ { \beta _ { k } } u \right) , } \end{array}\tag{22}
$$

where both the derivative orders $\beta _ { 1 } , \ldots , \beta _ { k }$ and the scales $s _ { 1 } , \ldots , s _ { k }$ are learnable parameters optimized jointly with network weights θ. The learnable scales normalize features across different derivative orders, since ${ \mathfrak { D } } ^ { \beta }$ u can vary dramatically in magnitude as $\beta$ changes.

While the Weyl fractional derivative (Definition 2.1) provides an elegant spectral formulation, it can introduce Fourier artifacts (Gibbs ringing) near discontinuities. We instead use the Grunwald-Letnikov¨ (GL) finite-difference formulation:

$$
{ \mathfrak { D } } ^ { \beta } u ( x ) = { \frac { 1 } { h ^ { \beta } } } \sum _ { j = 0 } ^ { K _ { \mathrm { G L } } } ( - 1 ) ^ { j } { \binom { \beta } { j } } u ( x - j h ) ,\tag{23}
$$

where h is the grid spacing and the generalized binomial coefficients $\binom { \beta } { j }$ extend naturally to noninteger $\beta$ via the Gamma function. This formulation is differentiable with respect to $\beta ,$ enabling gradient-based optimization.

In practice, we omit the factor $h ^ { - \beta }$ . For small h and moderate $\beta ,$ , this factor can be dangerously large and destabilize training. Since the learnable scales $s _ { k }$ and the network itself absorb any constant scaling, removing $h ^ { - \beta }$ preserves the functional structure of the derivative while ensuring training stability. We denote backbones augmented with learnable derivative features using the ∂-prefix: ∂-FNO, ∂-Transolver, etc.

## 7 Experiments

We validate ∂-NO across multiple PDEs and backbone architectures. Our experiments address two questions: Does ∂-augmentation improve accuracy? Does the theory correctly predict how $\beta ^ { * }$ varies with noise, sample size, and network capacity?

## 7.1 Setup

Datasets. We use benchmark datasets following Li et al. [1]: Burgers’ equation (smooth, $\nu = 0 . 1 $ near-shock, $\nu = 0 . 0 0 1 )$ ), Darcy flow (elliptic), and two-dimensional Navier-Stokes (incompressible turbulence at $\nu = 1 0 ^ { - 5 } )$ , plus a third-order KdV dataset (m=3) obtained by direct numerical integration from random initial conditions (App. I).

Backbones. Our primary goal is to validate the theoretical predictions from Section 5. Neural operator training is particularly sensitive to hyperparameters, often requiring very small batch sizes (values of 8, 4, or even 1–2 are common [8]) to achieve good performance. This highlights the delicate nature of training such models. We use FNO [1] as a canonical spectral method, its tensorized variant TFNO, the localized operator LocalNO [28], and Transolver [8] as a state-of-the-art attention-based architecture, employing the THUML library [29] with default hyperparameters.

Derivative features. For the second-order benchmarks (Burgers, Darcy, NS), $m = 2$ and we use two learnable derivative features. For 1D problems: $\mathfrak { D } _ { x } ^ { \beta _ { 1 } }$ u and $\mathfrak { D } _ { x } ^ { \beta _ { 2 } } u$ , initialized at $\beta _ { 1 } = 1 . 0$ and $\beta _ { 2 } = 2 . 0 ;$ for the third-order KdV (m = 3) we use a third feature $\mathfrak { D } _ { x } ^ { \beta _ { 3 } }$ u initialized at $\beta _ { 3 } = 3 . 0$ . For 2D problems, we include directional derivatives $\mathfrak { D } _ { x } ^ { \beta _ { x , 1 } } u , \mathfrak { D } _ { x } ^ { \beta _ { x , 2 } } u , \bar { \mathfrak { D } } _ { y } ^ { \beta _ { y , 1 } } u , \mathfrak { D } _ { y } ^ { \beta _ { y , 2 } } u$ and one mixed partial $\mathfrak { D } _ { y } ^ { \beta _ { y , c } } \mathfrak { D } _ { x } ^ { \beta _ { x , c } } u$ . Each order is learned independently. In particular, the mixed partial has two learnable orders $( \beta _ { x , c } , \beta _ { y , c } )$ and a single shared scaling factor.

Training. All models train with AdamW. The $\beta$ parameters use 10× higher learning rate $( 1 0 ^ { - 2 }$ vs $1 0 ^ { - 3 } )$ because they sit at the input level: gradients through them propagate back through the full backbone and attenuate accordingly. We report relative $\breve { L } ^ { 2 }$ errors (%). Full hyperparameters and implementation details are provided in Appendix I.

## 7.2 Validation of ∂-NO Improvements

Table 1 summarizes our main results: five PDE benchmarks, four backbones, and both clean $( \sigma { = } 0 )$ and noisy (σ=0.05) regimes. The ∂-augmentation improves every backbone/dataset/noise combination tested; the separate CNO benchmarks (§7.4, Table 2) show larger typical gains. A broader stress test under Gaussian and uniform additive noise with $\sigma \in [ 0 . 0 5 , 0 . 3 ]$ confirms this robustness across every tested setting (Appendix L.3), and a matched-parameter control isolates the gains to the derivative inductive bias rather than added capacity (Appendix L.2). Further robustness analyses (emergent directional isotropy on 2D/3D isotropic PDEs in Figs. $6 , 7 ;$ resolution-dependent gains in Fig. 4; complementarity with PINO physics-informed training in Table 9) are reported in the appendix.

Table 1: Relative $L ^ { 2 }$ error (%), averaged over 5 seeds, across 5 PDE benchmarks × 4 backbones × {clean, noisy}. Bold = best per (backbone, dataset, noise) cell. ∂-augmentation improves every cell tested.
<table><tr><td rowspan="2"></td><td colspan="2">Burgers  $\left( \nu \mathrm { = } 0 . 1 \right)$ </td><td colspan="2">Burgers  $\scriptstyle ( \nu = 0 . 0 0 1 )$ </td><td colspan="2">KdV</td><td colspan="2">Darcy (Elliptic)</td><td colspan="2">NS2D  $( \nu { = } 1 0 ^ { - 5 } )$ </td></tr><tr><td> $\sigma { = } 0$ </td><td> $\sigma { = } 0 . 0 5$ </td><td> $\sigma { = } 0$ </td><td> $\sigma { = } 0 . 0 5$ </td><td> $\sigma { = } 0$ </td><td> $\sigma { = } 0 . 0 5$ </td><td> $\sigma { = } 0$ </td><td> $\sigma { = } 0 . 0 5$ </td><td> $\sigma { = } 0$ </td><td> $\sigma { = } 0 . 0 5$ </td></tr><tr><td>FNO</td><td>0.065</td><td>0.250</td><td>1.138</td><td>1.632</td><td>1.598</td><td>2.255</td><td>5.643</td><td>5.590</td><td>3.479</td><td>3.575</td></tr><tr><td>∂-FNO</td><td>0.055</td><td>0.190</td><td>0.837</td><td>1.366</td><td>1.549</td><td>2.195</td><td>5.426</td><td>5.415</td><td>3.428</td><td>3.511</td></tr><tr><td>Improv.</td><td>15.1%</td><td>24.1%</td><td>26.5%</td><td>16.3%</td><td>3.0%</td><td>2.7%</td><td>3.8%</td><td>3.1%</td><td>1.5%</td><td>1.8%</td></tr><tr><td>TFNO</td><td>0.071</td><td>0.209</td><td>0.702</td><td>1.032</td><td>2.312</td><td>2.745</td><td>5.223</td><td>5.206</td><td>3.428</td><td>3.534</td></tr><tr><td>∂-TFNO</td><td>0.062</td><td>0.178</td><td>0.676</td><td>0.917</td><td>2.227</td><td>2.660</td><td>4.843</td><td>4.906</td><td>3.228</td><td>3.368</td></tr><tr><td>Improv.</td><td>11.9%</td><td>14.7%</td><td>3.7%</td><td>11.2%</td><td>3.7%</td><td>3.1%</td><td>7.3%</td><td>5.8%</td><td>5.8%</td><td>4.7%</td></tr><tr><td>LocalNO</td><td>0.063</td><td>0.236</td><td>0.768</td><td>1.272</td><td>2.587</td><td>3.625</td><td>5.511</td><td>5.534</td><td>1.994</td><td>2.620</td></tr><tr><td>∂-LocalNO</td><td>0.060</td><td>0.199</td><td>0.732</td><td>1.203</td><td>2.352</td><td>3.441</td><td>5.430</td><td>5.487</td><td>1.984</td><td>2.533</td></tr><tr><td>Improv.</td><td>6.1%</td><td>15.8%</td><td>4.7%</td><td>5.4%</td><td>9.1%</td><td>5.1%</td><td>1.5%</td><td>0.9%</td><td>0.5%</td><td>3.3%</td></tr><tr><td>Transolver</td><td>0.990</td><td>0.765</td><td>18.988</td><td>7.521</td><td>22.427</td><td>16.334</td><td>14.645</td><td>15.332</td><td>4.301</td><td>4.821</td></tr><tr><td>∂-Transolver</td><td>0.709</td><td>0.736</td><td>4.633</td><td>4.049</td><td>17.957</td><td>9.588</td><td>13.125</td><td>11.212</td><td>4.043</td><td>4.341</td></tr><tr><td>Improv.</td><td>28.4%</td><td>3.8%</td><td>75.6%</td><td>46.2%</td><td>19.9%</td><td>41.3%</td><td>10.4%</td><td>26.9%</td><td>6.0%</td><td>9.9%</td></tr></table>

## 7.3 Theory Validation

Theorem 5.6 makes testable predictions about how $\beta ^ { * }$ responds to noise σ, sample size $n ,$ and network capacity (Fourier modes k). Since $\xi _ { c } = 2 \pi k$ , the formula simplifies to show explicit dependence on the three experimental parameters:

$$
\delta ^ { * } \propto \frac { 2 s \ln k + 2 \ln \sigma - \ln n + C } { 2 \ln k + C ^ { \prime } }\tag{24}
$$

where $C , C ^ { \prime }$ are problem-dependent constants. This yields three monotonicities under Assumption $4 . 2 \colon$ $\sigma \uparrow \Rightarrow \beta ^ { * } \downarrow , n \uparrow \Rightarrow \beta ^ { * } \uparrow$ , and $k \uparrow \Rightarrow \beta ^ { * } \downarrow$ . Near the smoothness boundary $s \approx m + d / 2$ , the capacityeffect margin shrinks and the prediction loses sharpness, as detailed below. We test all three on Burgers (smooth and shock), using a single learnable $\beta$ and varying one factor at a time.

Figures 2 and 3 confirm both unconditional predictions on smooth Burgers, the noise prediction on shock Burgers (whose sample-size effect plateaus), and the capacity effect (a decrease of $\beta ^ { * }$ with k in both regimes, less cleanly so near the smoothness boundary):

Noise $( \sigma \uparrow \Rightarrow \beta ^ { * } \downarrow )$ . The +2 ln σ term in (24) predicts that increasing noise increases $\delta ^ { * }$ . Experimentally, learned $\beta ^ { * }$ decreases with σ on both smooth and shock Burgers (monotonically on shock, a noisier downward trend on smooth), as predicted. The network automatically regularizes by learning lower derivative orders when data is noisy.

Sample size $( n \uparrow \Rightarrow \beta ^ { \ast } \uparrow )$ . The − ln n term predicts that more data decreases $\delta ^ { * }$ and drives $\beta ^ { * } \to m$ Experimentally, on smooth Burgers, learned $\beta ^ { * }$ increases monotonically with $n ,$ , approaching but not reaching $m = 2$ at any practically achievable n. The persistent gap $\delta ^ { * } > 0$ is consistent with Remark $5 . 7 \colon$ the recovery threshold $\begin{array} { r } { \dot { n ^ { \star } } = \sigma ^ { 2 } \exp ( A _ { c } ) } \end{array}$ is far beyond the sample sizes accessible in this sweep.

Capacity $( k \uparrow \Rightarrow \beta ^ { * } \downarrow )$ . The $+ 2 s$ ln k term in the numerator competes with +2 ln k in the denominator. Under Assumption $4 . 2 \ : ( s > m + d / 2 )$ , the numerator dominates and $k \uparrow \Rightarrow \delta ^ { * } \uparrow ;$ ; near the smoothness boundary the margin shrinks and the prediction loses sharpness. This predicts qualitatively different behavior for smooth vs. shock Burgers. Experimentally, we observe two qualitatively consistent but differently scaled behaviors in Figures 2 and 3: smooth Burgers $( \tau \approx 2 . 2 )$ shows a clean monotonic decrease, while shock Burgers $( \tau \approx 0 . 8 )$ shows a noisier, non-monotone decrease over the same range. The consistency across smooth and shock regimes supports the bias-variance framework.

![](images/8541f6d521a8023117ab68720bcb2682b250b1ccfb41ebb0b977ff65a3223e41.jpg)

![](images/36e7ca0faa94ab4bdcfdae326e5e4cee86736077a87db09c2fec5257a8b96b0a.jpg)

![](images/2ac3c1279d91e690ad0b2eeeed59d224551acb893d5de2b6db233482cffb77e4.jpg)

![](images/afafeba1618bf2958ae3128ea0180809c3af72629f5b38cccc559e31af7441d4.jpg)

![](images/d67b732281fb36d7c5c9907b6177d13ffa2294a77ab827c4c2f41cce2fdc65a0.jpg)

![](images/dfe77f70f4e612cb77ee1e18299b576e5d09849739a7b267007f67ad450c8952.jpg)  
Figure 2: Burgers (smooth, $\nu { = } 0 . 1 ) \colon$ Theory vs. experiment. Top: relative $L ^ { 2 }$ error; bottom: learned $\beta ^ { * }$ . Theorem $5 . 6 \mathrm { { } \mathrm { { s } } }$ two unconditional predictions are supported: (left) $\sigma \uparrow \Rightarrow \beta ^ { * }$ ↓; (center) $n \uparrow \Rightarrow \beta ^ { * } \uparrow$ , with $\beta ^ { * }$ approaching but not reaching $m { = } 2 .$ , consistent with the unreachable recovery threshold (Remark 5.7). At smoothness $\tau \approx 2 . 2$ (comfortably above the Assumption 4.2 threshold $s > m + d / 2 )$ , the capacity formula predicts $k \uparrow \Rightarrow \beta ^ { * }$ ↓ (right), which holds monotonically.

![](images/a99a3e91192cfc1684dd1652e99efa8a0b1259dc21d8b3dd519215bd41a4dcb8.jpg)

![](images/e4ea477e0ddbc5bfa2f4e45121bf77bc12a0b853d3681861e1001501c44712b9.jpg)

![](images/d12d3f3719267d46c23869d8f22c55965fc3b705e4c33ba65f7a22a2ab8bed06.jpg)

![](images/4988492f04b335b12b5001944d507e3e57bbe35861ffd46e70d46f1eb10cf200.jpg)

![](images/b527e14290f6578288e85dc0c9faec90823de947d24805092e91c8a03e4add23.jpg)

![](images/dd189ec80b1d0a765ec3a0019f72d7d50438f113af583c7eef78475a51536313.jpg)  
Figure 3: Burgers (near-shock, $\scriptstyle \nu = 0 . 0 0 1 ) $ : Theory vs. experiment. Same panels as Figure 2. The noise prediction $( \sigma \uparrow \Rightarrow \beta ^ { * } \downarrow )$ holds clearly; the sample-size effect $( n \uparrow \Rightarrow \beta ^ { * } \uparrow )$ attenuates to a noisy plateau near the smoothness boundary $( \tau \approx 0 . 8 )$ , as expected when Assumption 4.2 is marginal. At rough-data smoothness $\tau \approx 0 . 8$ (near the Assumption 4.2 smoothness boundary), the capacity formula admits a weakened k-effect; the empirical k-sweep is correspondingly noisier and non-monotone rather than the clean monotone of the smooth case, consistent with this regime change.

## 7.4 Differential Inductive Bias: ∂-CNO

A natural question is whether ∂-NO’s gains come from an increased local processing capacity or from the specific inductive bias from fractional differentiation. Convolutional architectures provide a sharp test: a generic convolution can represent a finite-difference derivative, but it is not constrained to learn one. To validate the importance of the $\partial { \cdot } { \bf N O } ^ { \prime } { \bf s }$ inductive bias, we instantiate ∂-augmentation on the Convolutional Neural Operator [30] (CNO) and compare ∂-CNO against the vanilla CNO across seven PDE benchmarks from the benchmark suite of Raonic et al. [30].

Results are in Table 2. ∂-CNO improves CNO in all settings, with peak gains up to 55–65% on Poisson and Wave, and ∼ 39% on NS Shear. The pattern is consistent with the differential interpretation: gains are largest on PDEs whose solution operators most depend on global differential structure (elliptic Poisson, shear-dominated NS, wave propagation), and smaller on data-dominated settings (Darcy, Airfoil). Because CNO is built entirely from learnable convolutions, the persistent improvement from imposing the GL derivative structure on the input is direct evidence that derivative features carry information convolutional layers do not reliably acquire from data alone.

Table 2: ∂-CNO vs. CNO on seven benchmarks from the suite of Raonic et al. [30], evaluated under both relative $L ^ { 2 }$ error and median relative $L ^ { 1 }$ error (averaged over 5 seeds). Bold = best per (dataset, noise, metric) cell. ∂-CNO improves CNO in 14/14 settings under both metrics.
<table><tr><td></td><td></td><td colspan="2">Poisson</td><td colspan="2">NS Shear</td><td colspan="2">Wave</td><td colspan="2">Allen-Cahn</td><td colspan="2">Transp. Cont.</td><td colspan="2">Darcy</td><td colspan="2">Airfoil</td></tr><tr><td>σ</td><td></td><td>0</td><td>0.05</td><td>0</td><td>0.05</td><td>0</td><td>0.05</td><td>0</td><td>0.05</td><td>0</td><td>0.05</td><td>0</td><td>0.05</td><td>0</td><td>0.05</td></tr><tr><td> $\tilde { \Xi }$ </td><td>CNO ∂-CNO</td><td>1.788</td><td>3.664</td><td>9.686</td><td>12.218</td><td>0.590</td><td>1.990</td><td>2.618</td><td>1.980</td><td>0.416</td><td>0.413</td><td>0.763</td><td>0.724</td><td>1.184</td><td>1.130</td></tr><tr><td></td><td></td><td>1.053</td><td>1.293</td><td>7.104</td><td>7.746</td><td>0.520</td><td>0.900</td><td>1.632</td><td>1.575</td><td>0.353</td><td>0.381</td><td>0.752</td><td>0.718</td><td>1.137</td><td>1.111</td></tr><tr><td></td><td>Improv.</td><td>41.2%</td><td>64.7%</td><td>26.7%</td><td>36.6%</td><td>11.8%</td><td>54.8%</td><td>37.7%</td><td>20.4%</td><td>15.2%</td><td>7.7%</td><td>1.4%</td><td>0.8%</td><td>4.0%</td><td>1.7%</td></tr><tr><td></td><td>CNO</td><td>1.200</td><td>2.504</td><td>8.120</td><td>10.080</td><td>0.440</td><td>1.550</td><td>0.870</td><td>0.940</td><td>0.358</td><td>0.380</td><td>0.514</td><td>0.482</td><td>0.446</td><td>0.473</td></tr><tr><td>L</td><td>∂-CNO</td><td>0.660</td><td>0.884</td><td>5.910</td><td>6.110</td><td>0.439</td><td>0.692</td><td>0.646</td><td>0.857</td><td>0.341</td><td>0.352</td><td>0.501</td><td>0.457</td><td>0.443</td><td>0.410</td></tr><tr><td></td><td>Improv.</td><td>45.0%</td><td>64.7%</td><td>27.2%</td><td>39.4%</td><td>0.2%</td><td>55.4%</td><td>25.7%</td><td>8.8%</td><td>4.7%</td><td>7.5%</td><td>2.6%</td><td>5.1%</td><td>0.7%</td><td>13.3%</td></tr></table>

## 7.5 Robustness and Directionality

Beyond §7.3’s monotonicities, the bias-variance picture predicts three further effects: learnable orders should recover the geometric symmetries of the PDE without prescription, ∂-NO’s advantage should grow at coarser resolution, and input-side derivative features should complement loss-side PDE residuals. Each is observed.

Emergent isotropy. On isotropic 2D PDEs (Darcy, Navier-Stokes), learned $\beta _ { x }$ and $\beta _ { y }$ converge to within $\Delta \ : < \ : 0 . 0 3 5$ of each other without isotropy being prescribed (Appendix Fig. 6). On 3D compressible Navier-Stokes, axis-aligned channels settle near $\beta \approx 2$ (Laplacian) while the cross-derivative channel collapses to near-identity (Appendix Fig. 7): the network identifies which directional derivatives matter and matches their order to the observed physics.

Resolution-dependent gains. Empirically, ∂-FNO outperforms FNO at every resolution from N=64 to N=2048, with the noisy-data improvement decaying from 39% to 14% (Appendix Fig. 4). The advantage of input-side derivative features grows at coarser resolution, where the network’s Nyquist headroom is most constrained.

Complementarity with physics-informed training. PINO [31] encodes the PDE on the loss side; ∂-NO encodes it on the input side. Adding ∂-features improves PINO in all $8 / 8$ tested settings (Appendix Table 9): the two channels do not duplicate but complement each other.

## 8 Related Work

Neural operators. The field has grown rapidly since FNO [1] and DeepONet [2]. Recent work explores attention mechanisms [6–8], wavelets [32], and implicit representations [23]. Theoretical foundations include approximation bounds [3] and universal approximation results. Our ∂-augmentation is orthogonal to these advances: it wraps any backbone with learnable derivative features, requiring no architectural modification.

Local-feature operators. A parallel line of work injects locality into neural operators [33]: Conv-FNO [34] attaches local convolutions to FNO, LocalNO [28] introduces localized integral and differential kernels, and CNO [30] replaces spectral layers with structured convolutions throughout. ∂-NO is complementary: it provides a specific local operation (fractional differentiation) with a built-in inductive bias for PDE structure. Our ∂-CNO experiments (§7) directly isolate this bias from generic local processing.

Derivative and physics-aware features. Recent work has explored providing derivative information to neural operators. Equation-aware emulators [24] encode full PDE structure including nonlinear terms, achieving improvements but requiring the PDE form a priori. Physics-informed approaches like PINNs [35, 36] and the physics-informed neural operator [31] incorporate derivatives through loss terms via automatic differentiation. Our approach differs in three ways: (1) we learn derivative orders rather than specifying them from known physics; (2) we provide derivatives as input features, computed via FFT or GL; and (3) our bias-variance analysis reveals that optimal orders should be less than the PDE order, a phenomenon that equation-matching approaches do not address.

Spectral perspectives. FNO performs convolutions in Fourier space with a hard mode truncation at $k _ { \mathrm { m a x } }$ . Our analysis exploits this structure: the capacity cutoff $\xi _ { c }$ in Theorem 5.6 is sharply defined for FNO and other neural operators with a fixed number of Fourier modes, making the bias-variance tradeoff particularly sharp. More broadly, spectral bias in neural networks, where low frequencies are learned before high frequencies [37], suggests that derivative features may help by preprocessing high-frequency content into a form the network can access earlier in training.

Fractional calculus. Fractional derivatives model anomalous diffusion, memory effects, and nonlocal phenomena [38], and have found applications in financial machine learning for constructing stationary features that preserve memory [39]. We use them differently: as a continuous parameterization enabling gradient-based optimization over derivative order. This treats the fractional derivative as a learnable feature transformation, not as a physical model of sub-diffusion or long-range dependence. Connections between learned orders and anomalous diffusion remain an interesting direction for future work.

## 9 Discussion

We introduced ∂-NO, a simple augmentation providing learnable fractional derivative features to any neural operator backbone. Through Picard iteration on mild solutions, we showed how derivatives of the initial condition directly contribute to the solution operator, providing theoretical grounding for derivative features. Providing these features improves approximation bounds from $\bar { O ( } W ^ { - ( s - \bar { m } ) / d } )$ to $O ( W ^ { - s / d } )$ , a substantial gain in parameter efficiency.

Our central finding is that the optimal derivative order $\beta ^ { * }$ is less than the PDE order $m ,$ with the gap characterized by a spectral bias-variance tradeoff. The closed-form expression for $\delta ^ { * } = m - \bar { \beta } ^ { * }$ correctly predicts how the optimal order responds to noise, sample size, and capacity, as confirmed experimentally. For FNO and other operators with fixed Fourier modes, the capacity cutoff is sharply defined. Learning $\beta$ achieves automatic spectral regularization: the network selects $\lvert \beta ^ { * }$ in response to noise, sample size, and capacity, as predicted by Theorem 5.6 and confirmed empirically.

Limitations. Four scope boundaries: (i) the Grunwald-Letnikov stencil assumes a regular grid¨ (extensions via graph fractional Laplacians or RBF operators are open); (ii) Theorem 5.6’s sharp predictions require $s > m + d / 2$ and degrade to a directional guide near shocks, as the near-shock Burgers experiments empirically confirm; (iii) we preprocess only spatial inputs, leaving temporal fractional derivatives as an orthogonal direction; and (iv) the theorem characterizes the populationoptimal $\beta ^ { * }$ rather than the optimizer’s converged value. Empirically, $\beta$ stabilizes within 200–400 AdamW epochs across every backbone and PDE we tested, but formal trajectory guarantees remain open [40–42].

Stepping back from these scope considerations, the central finding $\beta ^ { * } < m$ illustrates a broader principle for physics-informed machine learning at finite scales: inductive biases derived from governing equations should be parameterized rather than prescribed, allowing the optimizer to locate its own compromise between approximation power and estimation variance. Under finite, noisy data, the statistically optimal physics is rarely the literal equation; learning the inductive bias is what turns physical knowledge into statistical efficiency.

## Impact Statement

This work advances methodology for scientific machine learning, with potential applications in a wide range of scientific fields. The finding that learned derivative orders should be sub-physical has implications for how we incorporate domain knowledge into machine learning systems: matching physics exactly may not always be optimal. We identify no specific ethical concerns.

## Code Availability

The code for ∂-NO is available at https://github.com/FaresBMehouachi/delNO. The authors declare no competing interests.

## Acknowledgments

This work was supported by the NYUAD Center for Interacting Urban Networks (CITIES), funded by Tamkeen under the NYUAD Research Institute Award CG001. The views expressed in this article are those of the authors and do not reflect the opinions of CITIES or their funding agencies.

## References

[1] Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Fourier neural operator for parametric partial differential equations. In International Conference on Learning Representations, 2021. arXiv:2010.08895.

[2] Lu Lu, Pengzhan Jin, Guofei Pang, Zhongqiang Zhang, and George Em Karniadakis. Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. Nature machine intelligence, 3(3):218–229, 2021.

[3] Nikola Kovachki, Zongyi Li, Burigede Liu, Kamyar Azizzadenesheli, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Neural operator: Learning maps between function spaces with applications to PDEs. Journal of Machine Learning Research, 24(89):1–97, 2023.

[4] Chuhan Yang, Fares B Mehouachi, Monica Menendez, and Saif Eddin Jabari. Urban traffic analysis and forecasting through shared Koopman eigenmodes. Nonlinear Dynamics, 113(23): 32307–32328, 2025.

[5] Bilal Thonnam Thodi, Sai Venkata Ramana Ambadipudi, and Saif Eddin Jabari. Fourier neural operator for learning solutions to macroscopic traffic flow models: Application to the forward and inverse problems. Transportation Research Part C: Emerging Technologies, 160:104500, 2024.

[6] Shuhao Cao. Choose a transformer: Fourier or Galerkin. Advances in Neural Information Processing Systems, 34:24924–24940, 2021.

[7] Zhongkai Hao, Zhengyi Wang, Hang Su, Chengyang Ying, Yinpeng Dong, Songming Liu, Ze Cheng, Jian Song, and Jun Zhu. GNOT: A general neural operator transformer for operator learning. In International Conference on Machine Learning, pages 12556–12569. PMLR, 2023.

[8] Haixu Wu, Huakun Luo, Haowen Wang, Jianmin Wang, and Mingsheng Long. Transolver: A fast transformer solver for PDEs on general geometries. In International Conference on Machine Learning, pages 53681–53705. PMLR, 2024.

[9] Haixu Wu, Tengge Hu, Huakun Luo, Jianmin Wang, and Mingsheng Long. Solving highdimensional PDEs with latent spectral models. In International Conference on Machine Learning, pages 37417–37438. PMLR, 2023.

[10] Fares B. Mehouachi and Saif Eddin Jabari. FlowMixer: A depth-agnostic neural architecture for interpretable spatiotemporal forecasting. Advances in Neural Information Processing Systems, 38:88811–88861, 2025.

[11] Guofei Pang, Lu Lu, and George Em Karniadakis. fPINNs: Fractional physics-informed neural networks. SIAM Journal on Scientific Computing, 41(4):A2603–A2626, 2019.

[12] Ling Guo, Hao Wu, Xiaochen Yu, and Tao Zhou. Monte Carlo fPINNs: Deep learning method for forward and inverse problems involving high dimensional fractional partial differential equations. Computer Methods in Applied Mechanics and Engineering, 400:115523, 2022.

[13] Hongpeng Ren, Xiangyun Meng, Rongrong Liu, Jian Hou, and Yongguang Yu. A class of improved fractional physics informed neural networks. Neurocomputing, 562:126890, 2023.

[14] Rana Javadi, Hamid Mesgarani, Omid Nikan, and Zakieh Avazzadeh. Solving fractional order differential equations by using fractional radial basis function neural network. Symmetry, 15(6): 1275, 2023.

[15] Shujaat Khan, Jawwad Ahmad, Imran Naseem, and Muhammad Moinuddin. A novel fractional gradient-based learning algorithm for recurrent neural networks. Circuits, Systems, and Signal Processing, 37(2):593–612, 2018.

[16] Yeonjong Shin, Jer´ ome Darbon, and George Em Karniadakis. Accelerating gradient descent ˆ and Adam via fractional gradients. Neural Networks, 161:185–201, 2023.

[17] Sroor M Elnady, Mohamed El-Beltagy, Ahmed G Radwan, and Mohammed E Fouda. A comprehensive survey of fractional gradient descent methods and their convergence analysis. Chaos, Solitons & Fractals, 194:116154, 2025.

[18] Yi-Fei Pu, Zhang Yi, and Ji-Liu Zhou. Fractional Hopfield neural networks: Fractional dynamic associative recurrent neural networks. IEEE Transactions on Neural Networks and Learning Systems, 28(10):2319–2333, 2017. doi: 10.1109/TNNLS.2016.2582512.

[19] Wenjun Cui, Qiyu Kang, Xuhao Li, Kai Zhao, Wee Peng Tay, Weihua Deng, and Yidong Li. Neural variable-order fractional differential equation networks. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 16109–16117, 2025.

[20] Qiyu Kang, Kai Zhao, Qinxu Ding, Feng Ji, Xuhao Li, Wenfei Liang, Yang Song, and Wee Peng Tay. Unleashing the potential of fractional calculus in graph neural networks with FROND. In International Conference on Learning Representations, 2024. arXiv:2404.17099.

[21] Cecilia Coelho, M Fernanda P Costa, and Luis L Ferras. Neural fractional differential equations: Optimising the order of the fractional derivative. Fractal and Fractional, 8(9):529, 2024.

[22] Huakun Luo, Haixu Wu, Hang Zhou, Lanxiang Xing, Yichen Di, Jianmin Wang, and Mingsheng Long. Transolver++: An accurate neural solver for PDEs on million-scale geometries. In International Conference on Machine Learning, pages 41432–41449. PMLR, 2025.

[23] Louis Serrano, Lise Le Boudec, Armand Kassa¨ı Koupa¨ı, Thomas X Wang, Yuan Yin, Jean-Noel Vittaut, and Patrick Gallinari. Operator learning with neural fields: Tackling PDEs on¨ general geometries. In Advances in Neural Information Processing Systems, volume 36, pages 70581–70611, 2023.

[24] Qian-Ze Zhu, Paul Raccuglia, and Michael P Brenner. Generalizing PDE emulation with equation-aware neural operators. arXiv preprint arXiv:2511.09729, 2025.

[25] Bernd Carl and Irmtraud Stephani. Entropy, Compactness and the Approximation of Operators, volume 98 of Cambridge Tracts in Mathematics. Cambridge University Press, Cambridge, 1990.

[26] Dmitry Yarotsky. Error bounds for approximations with deep ReLU networks. Neural Networks, 94:103–114, 2017. ISSN 0893-6080.

[27] Michael L. Stein. Interpolation of Spatial Data: Some Theory for Kriging. Springer Series in Statistics. Springer, New York, NY, 1999. ISBN 978-0-387-98629-6. doi: 10.1007/978-1-4612-1494-6.

[28] Miguel Liu-Schiaffini, Julius Berner, Boris Bonev, Thorsten Kurth, Kamyar Azizzadenesheli, and Anima Anandkumar. Neural operators with localized integral and differential kernels. In International Conference on Machine Learning, pages 32576–32594. PMLR, 2024.

[29] THUML. Neural-Solver-Library: A library for advanced neural PDE solvers. https:// github.com/thuml/Neural-Solver-Library, 2025. Tsinghua Machine Learning Group.

[30] Bogdan Raonic, Roberto Molinaro, Tim De Ryck, Tobias Rohner, Francesca Bartolucci, Rima Alaifari, Siddhartha Mishra, and Emmanuel De Bezenac. Convolutional neural operators for´ robust and accurate learning of PDEs. Advances in Neural Information Processing Systems, 36: 77187–77200, 2023.

[31] Zongyi Li, Hongkai Zheng, Nikola Kovachki, David Jin, Haoxuan Chen, Burigede Liu, Kamyar Azizzadenesheli, and Anima Anandkumar. Physics-informed neural operator for learning partial differential equations. ACM/IMS Journal of Data Science, 1(3):1–27, 2024.

[32] Tapas Tripura and Souvik Chakraborty. Wavelet neural operator for solving parametric partial differential equations in computational mechanics problems. Computer Methods in Applied Mechanics and Engineering, 404:115783, 2023.

[33] Prajwal Chauhan, Salah Eddine Choutri, and Saif Eddin Jabari. Efficient dilated squeeze and excitation neural operator for differential equations. Transactions on Machine Learning Research, 2026. arXiv:2601.17407.

[34] Chaoyu Liu, Davide Murari, Lihao Liu, Yangming Li, Chris Budd, and Carola-Bibiane Schonlieb. Enhancing Fourier neural operators with local spatial features. ¨ arXiv preprint arXiv:2503.17797, 2025.

[35] Maziar Raissi, Paris Perdikaris, and George E Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal ofComputational Physics, 378:686–707, 2019.

[36] Bilal Thonnam Thodi, Sai Venkata Ramana Ambadipudi, and Saif Eddin Jabari. Learning-based solutions to nonlinear hyperbolic PDEs: Empirical insights on generalization errors. In Fifth Workshop on Machine Learning and the Physical Sciences (NeurIPS 2022), 2022.

[37] Nasim Rahaman, Aristide Baratin, Devansh Arpit, Felix Draxler, Min Lin, Fred Hamprecht, Yoshua Bengio, and Aaron Courville. On the spectral bias of neural networks. In International Conference on Machine Learning, pages 5301–5310. PMLR, 2019.

[38] Ralf Metzler and Joseph Klafter. The random walk’s guide to anomalous diffusion: a fractional dynamics approach. Physics reports, 339(1):1–77, 2000.

[39] Marcos M. Lopez de Prado. ´ Advances in Financial Machine Learning. John Wiley & Sons, Hoboken, NJ, 1st edition, 2018. ISBN 978-1-119-48208-6.

[40] Fares B. Mehouachi and Saif Eddin Jabari. Catastrophic overfitting, entropy gap, and participation ratio: A noiseless ℓ<sup>p</sup> norm solution for fast adversarial training. arXiv preprint arXiv:2505.02360, 2025.

[41] Fares B. Mehouachi and Chaouki Kasmi. Exact stochastic second order deep learning. arXiv preprint arXiv:2104.03804, 2021.

[42] Fares B. Mehouachi, Qingjie Yang, and Chaouki Kasmi. Systems, methods, and computerreadable media for utilizing a Sifrian inversion to build a model to generate an image of a surveyed medium. US Patent 11,551,416, jan 2023. US Patent 11,551,416.

[43] Samuel Lanthaler, Siddhartha Mishra, and George E Karniadakis. Error estimates for Deep-ONets: A deep learning framework in infinite dimensions. Transactions ofMathematics and its Applications, 6(1), 2022.

[44] Igor Podlubny. Fractional differential equations: an introduction to fractional derivatives, fractional differential equations, to methods of their solution and some of their applications, volume 198. Academic Press, 1999.

[45] Makoto Takamoto, Timothy Praditia, Raphael Leiteritz, Daniel MacKinlay, Francesco Alesiani, Dirk Pfluger, and Mathias Niepert. PDEBench: An extensive benchmark for scientific machine¨ learning. In Advances in Neural Information Processing Systems Datasets and Benchmark Track, 2022.

[46] Fares B. Mehouachi, Juan Galvis, Santiago Morales, Milosch Meriac, Felix Vega, and Chaouki Kasmi. Detection and identification of UAVs based on spectrum monitoring and deep learning in negative SNR conditions. In XXXIV URSI General Assembly and Scientific Symposium (URSI GASS), 2021.

[47] Fares B. Mehouachi, Qingjie Yang, Juan Galvis, Santiago Morales, Milosch Meriac, Felix Vega, and Chaouki Kasmi. Detection of UAVs based on spectrum monitoring and deep learning in negative SNR conditions. URSI Radio Science Letters, 3, 2021. doi: 10.46620/21-0043.

## A Table of Notation

Table 3: Summary of notation used throughout the paper.
<table><tr><td>Notation</td><td>Meaning</td></tr><tr><td colspan="2">Spaces and functions</td></tr><tr><td> $\Omega \subset \mathbb { R } ^ { d }$ </td><td>Spatial domain</td></tr><tr><td> $d$ </td><td>Spatial dimension</td></tr><tr><td> $H ^ { s } ( \Omega )$ </td><td>Sobolev space of regularity s</td></tr><tr><td> $B _ { s } ( { \cal R } )$ </td><td>Sobolev ball  $\{ u \in \bar { H } ^ { s } : \| \dot { u } \| _ { H ^ { s } } \leq R \}$ </td></tr><tr><td> $u , u _ { 0 }$ </td><td>Input function / initial condition</td></tr><tr><td> $v$ </td><td>Solution function</td></tr><tr><td colspan="2">PDE structure</td></tr><tr><td></td><td>Differential order of the PDE</td></tr><tr><td> $m$ </td><td>Linear differential operator</td></tr><tr><td> $\mathcal { L }$ </td><td></td></tr><tr><td> $\mathcal { N }$ </td><td>Nonlinearity</td></tr><tr><td> $N$   $G _ { t }$ </td><td>Degree of polynomial nonlinearity Green&#x27;s function / semigroup  $e ^ { t { \mathcal { L } } }$ </td></tr><tr><td> $\nu$ </td><td>Viscosity coefficient</td></tr><tr><td colspan="2">Operators and features</td></tr><tr><td> ${ \mathcal { G } } , { \mathcal { G } } ^ { \dag }$ </td><td></td></tr><tr><td> $\mathcal { G } _ { \theta }$ </td><td>Solution operator (true / ground truth) Neural operator with parameters θ</td></tr><tr><td> $\Phi ^ { [ m ] }$ </td><td>Feature map  $( u , D ^ { 1 } u , \dots , D ^ { m } u )$ </td></tr><tr><td> $D ^ { \beta }$ </td><td>Fractional derivative of order  $\beta$ </td></tr><tr><td> ${ \mathfrak { D } } ^ { \beta }$ </td><td>GL approximation (Alternatively Weyl via FFT on periodic domains)</td></tr><tr><td colspan="2">Derivative orders</td></tr><tr><td> $\beta$ </td><td>Learnable derivative order</td></tr><tr><td> $\beta ^ { * }$ </td><td>Optimal derivative order (MSE-minimizing)</td></tr><tr><td> $\delta ^ { * } = m - \beta ^ { * }$ </td><td>Gap from PDE order</td></tr><tr><td> $s _ { k }$ </td><td>Learnable scale for k-th derivative feature</td></tr><tr><td colspan="2"></td></tr><tr><td>Spectral quantities  $\xi$ </td><td></td></tr><tr><td> $\overset { \cdot } { \hat { u } } ( \xi ) = \mathcal { F } [ u ]$ </td><td>Frequency variable Fourier transform of  $u$ </td></tr><tr><td></td><td>Power spectral density  $\mathbb { E } [ | \hat { u } ( \xi ) | ^ { 2 } ]$ </td></tr><tr><td> $S ( \xi )$ </td><td>Network capacity cutoff frequency</td></tr><tr><td> $\xi _ { c }$ </td><td></td></tr><tr><td> $\xi _ { \operatorname* { m a x } } = \pi / h$   $k , k _ { \mathrm { m a x } }$ </td><td>Nyquist frequency Fourier modes / maximum modes</td></tr><tr><td>Statistical quantities</td><td></td></tr><tr><td colspan="2"></td></tr><tr><td>n</td><td>Number of training samples</td></tr><tr><td> $\sigma ^ { 2 }$ </td><td>Observation noise variance</td></tr><tr><td>MSE,  $B , V$ </td><td>Mean squared error, bias, variance</td></tr><tr><td> $\tau = 2 ( s - m )$ </td><td>Regularity gap parameter</td></tr><tr><td colspan="2">Implementation</td></tr><tr><td> $W$ </td><td>Network width / number of parameters</td></tr><tr><td> $h$ </td><td>Grid spacing</td></tr><tr><td> $K / K _ { \mathrm { G L } }$ </td><td>Number of derivative channels / GL stencil length</td></tr><tr><td> $w _ { k } ^ { ( \beta ) }$ </td><td>GL weights for order  $\beta$ </td></tr><tr><td> $\psi = \Gamma ^ { \prime } / \Gamma$ </td><td>Digamma function</td></tr></table>

## B Proof of Theorem 3.1 (Derivative Emergence)

We prove that derivatives of the initial condition naturally enter the solution operator through Picard iteration.

Proof. Consider the semilinear PDE:

$$
\partial _ { t } u = \mathcal { L } u + \mathcal { N } [ u ] , \quad u ( 0 ) = u _ { 0 } ,\tag{25}
$$

where $\mathcal { L }$ is a linear differential operator of order m and $\mathcal { N }$ is a polynomial nonlinearity of degree $N$ Step 1: Mild solution formulation. The mild solution satisfies the integral equation:

$$
u ( t ) = G _ { t } \ast u _ { 0 } + \int _ { 0 } ^ { t } G _ { t - s } \ast \mathcal { N } [ u ( s ) ] d s ,\tag{26}
$$

where $G _ { t } = e ^ { t \mathcal { L } }$ is the semigroup generated by ${ \mathcal { L } } .$ . For the heat operator ${ \mathcal { L } } = \nu \Delta$ , this is the heat kernel. For general L, it is the Green’s function.

Step 2: Picard iteration. Define iterates:

$$
u ^ { ( 0 ) } ( t ) = G _ { t } * u _ { 0 } ,\tag{27}
$$

$$
u ^ { ( k + 1 ) } ( t ) = G _ { t } \ast u _ { 0 } + \int _ { 0 } ^ { t } G _ { t - s } \ast \mathcal { N } [ u ^ { ( k ) } ( s ) ] d s .\tag{28}
$$

Under standard assumptions (bounded $u _ { 0 } .$ , Lipschitz ${ \mathcal { N } } ,$ finite time horizon), the Banach fixed-point theorem guarantees convergence $u ^ { ( k ) } \to$ u in an appropriate function space.

Step 3: Expansion of the semigroup. The semigroup satisfies $\partial _ { t } G _ { t } \ = \ { \mathcal { L } } G _ { t }$ with $G _ { 0 } \ = \ I .$ For analytic semigroups (which include all parabolic operators with constant coefficients), Taylor expanding in s:

$$
G _ { s } * u _ { 0 } = u _ { 0 } + s \cdot \mathcal { L } u _ { 0 } + \frac { s ^ { 2 } } { 2 } \cdot \mathcal { L } ^ { 2 } u _ { 0 } + O ( s ^ { 3 } ) .\tag{29}
$$

The term $\mathcal { L } u _ { 0 }$ involves m-th order derivatives of $u _ { 0 }$ (e.g., $\mathcal { L } u _ { 0 } = \nu \Delta u _ { 0 }$ for diffusion). The $O ( s ^ { 3 } )$ remainder is uniform over $u _ { 0 } \in B _ { s } ( R )$ by standard semigroup estimates.

Step 4: Expansion of the nonlinearity. Since $\mathcal { N }$ is Frechet differentiable:´

$$
\mathcal { N } [ G _ { s } * u _ { 0 } ] = \mathcal { N } [ u _ { 0 } + s \mathcal { L } u _ { 0 } + O ( s ^ { 2 } ) ] = \mathcal { N } [ u _ { 0 } ] + s \cdot D \mathcal { N } [ u _ { 0 } ] [ \mathcal { L } u _ { 0 } ] + O ( s ^ { 2 } ) ,\tag{30}
$$

where $D \mathcal { N } [ u _ { 0 } ] [ v ]$ is the directional derivative of $\mathcal { N }$ at $u _ { 0 }$ in direction v.

For polynomial ${ \mathcal { N } } .$ this derivative has explicit form. For example, $\mathrm { i f } \mathcal { N } [ u ] = - u \partial _ { x } u$ , then:

$$
D \mathcal { N } [ u _ { 0 } ] [ v ] = - v \partial _ { x } u _ { 0 } - u _ { 0 } \partial _ { x } v .\tag{31}
$$

When $v = \mathcal { L } u _ { 0 } = \nu \partial _ { x x } u _ { 0 }$ , we get terms involving $\partial _ { x x } u _ { 0 } \cdot \partial _ { x } u _ { 0 }$ and $u _ { 0 } \cdot \partial _ { x x x } u _ { 0 }$

Step 5: First Picard iterate. The first iterate at final time $T$ is:

$$
u ^ { ( 1 ) } ( T ) = G _ { T } * u _ { 0 } + \int _ { 0 } ^ { T } G _ { T - s } * \mathcal { N } [ G _ { s } * u _ { 0 } ] d s .\tag{32}
$$

Substituting the expansions:

$$
\int _ { 0 } ^ { T } G _ { T - s } * \mathcal { N } [ G _ { s } * u _ { 0 } ] d s = \int _ { 0 } ^ { T } G _ { T - s } * \left( \mathcal { N } [ u _ { 0 } ] + s \cdot D \mathcal { N } [ u _ { 0 } ] [ \mathcal { L } u _ { 0 } ] + O ( s ^ { 2 } ) \right) d s\tag{33}
$$

$$
= \mathcal { N } [ u _ { 0 } ] * \int _ { 0 } ^ { T } G _ { T - s } ~ d s + D \mathcal { N } [ u _ { 0 } ] [ \mathcal { L } u _ { 0 } ] * \int _ { 0 } ^ { T } s \cdot G _ { T - s } ~ d s + O ( T ^ { 3 } ) ,\tag{34}
$$

where we exchanged the order of convolution and integration by Fubini’s theorem (valid since $G _ { T - s }$ is integrable on [0, T] for the semigroups considered).

Step 6: Averaged Green’s operators. Define:

$$
\bar { G } _ { T } : = \frac { 1 } { T } \int _ { 0 } ^ { T } G _ { T - s } d s ,\tag{35}
$$

$$
\tilde { G } _ { T } : = \frac { 2 } { T ^ { 2 } } \int _ { 0 } ^ { T } s \cdot G _ { T - s } d s .\tag{36}
$$

These are bounded linear operators for $T > 0$ (the semigroup is smoothing).

Step 7: Structure of the solution operator. Combining:

$$
u ^ { ( 1 ) } ( T ) = G _ { T } * u _ { 0 } + T \cdot \bar { G } _ { T } * \mathcal { N } [ u _ { 0 } ] + \frac { T ^ { 2 } } { 2 } \cdot \tilde { G } _ { T } * D \mathcal { N } [ u _ { 0 } ] [ \mathcal { L } u _ { 0 } ] + O ( T ^ { 3 } ) .\tag{37}
$$

The true solution $\begin{array} { r } { u ( T ) = \operatorname* { l i m } _ { k  \infty } u ^ { ( k ) } ( T ) } \end{array}$ differs from $u ^ { ( 1 ) } ( T )$ by higher-order Picard corrections. For fixed $T < T _ { * } ( \tilde { R } )$ , these corrections are bounded perturbations that preserve the dependence structure. Defining the remainder $\mathcal { R } ( u _ { 0 } , T ) : = \mathcal { G } [ u _ { 0 } ] - G _ { T } * u _ { 0 } - \hat { T } \cdot \bar { G } _ { T } * \mathcal { N } [ u _ { 0 } ]$ , we have $\Vert \mathcal { R } ( u _ { 0 } , T ) \Vert _ { L ^ { 2 } } \leq \bar { C } ( R ) T ^ { 2 }$ by standard Picard remainder estimates. The constant $\mathcal { \bar { C } } ( \mathcal { \bar { R } } )$ depends polynomially on R through the Lipschitz constant of $\mathcal { N }$ on $B _ { s } ( R )$ and the semigroup bounds.

Step 8: Identifying the features. The solution depends on $u _ { 0 }$ through:

• $u _ { 0 }$ (zeroth order): appears in $G _ { T } * u _ { 0 }$ and in $\mathcal { N } [ u _ { 0 } ]$

$\mathcal { L } u _ { 0 }$ (order m): appears in the expansion of $G _ { s } * u _ { 0 }$

$\partial ^ { j } u _ { 0 }$ for $j \le$ m: appear in $\mathcal { N } [ u _ { 0 } ] \mathrm { i f } \mathcal { N }$ involves derivatives

• $u _ { 0 } ^ { n }$ for $n \leq N :$ appear in $\mathcal { N } [ u _ { 0 } ] \mathrm { i f } \mathcal { N }$ is polynomial degree N

This completes the proof that the solution operator $\mathcal { G } [ u _ { 0 } ] = u ( T )$ depends on derivatives of $u _ { 0 }$ up to order m and powers up to $u _ { 0 } ^ { N }$ □

## B.1 Explicit Features for Specific PDEs

Burgers equation. $\partial _ { t } u + u \partial _ { x } u = \nu \partial _ { x x } u , \mathrm { s o } \ \mathcal { L } = \nu \partial _ { x x } \ \mathrm { a n d } \ \mathcal { N } [ u ] = - u \partial _ { x } u$

Features: $\left\{ u _ { 0 } , \partial _ { x } u _ { 0 } , \partial _ { x x } u _ { 0 } , u _ { 0 } \partial _ { x } u _ { 0 } \right\}$

Navier-Stokes (2D vorticity). $\partial _ { t } \omega + \mathbf { u } \cdot \nabla \omega = \nu \Delta \omega$ , where $\mathbf { u } = \nabla ^ { \perp } \psi$ and $\Delta \psi = - \omega$

Features: $\{ \omega _ { 0 } , \nabla \omega _ { 0 } , \Delta \omega _ { 0 } , \mathbf { u } _ { 0 } \cdot \nabla \omega _ { 0 } \}$

Darcy flow. $- \nabla \cdot ( a \nabla p ) = f$ . This is elliptic; the solution operator maps coefficient $a \mapsto$ pressure $p .$

Features: $\{ a , \nabla a , \nabla ^ { 2 } a \}$ (encoding coefficient variations).

KdV. $\partial _ { t } u \ : + \ : u \partial _ { x } u \ : + \ : \delta \partial _ { x x x } u \ : = \ : \ : 0$ with $\delta ~ = ~ 2 . 5 \times 1 0 ^ { - 3 } ~ ( \mathrm { A p p . ~ I ) , }$ , so $\begin{array} { r c l } { \mathcal { L } } & { = } & { - \delta \partial _ { x x x } } \end{array}$ $( m ~ = ~ 3 )$ and $\begin{array} { r } { \mathcal { N } [ u ] { \mathrm { ~ \scriptsize ~ = ~ \ } } - u \partial _ { x } u . } \end{array}$ The dependence on $\partial _ { x x x } u _ { 0 } ,$ and hence the feature set $\{ u _ { 0 } , \partial _ { x } u _ { 0 } , \partial _ { x x } u _ { 0 } , \partial _ { x x x } u _ { 0 } , u _ { 0 } \partial _ { x } u _ { 0 } \}$ , is independent of the value of δ.

## B.2 Extension to Arbitrary Times and Stationary Problems

Theorem 3.1 establishes derivative dependence for $T < T _ { * } ( R )$ . We now show this structure persists for arbitrary times and extends to stationary problems.

Iterated Picard for large T. The key observation is simple: once derivatives $\{ D ^ { j } u _ { 0 } \} _ { j \le m }$ appear in the first Picard iterate, they propagate through all subsequent iterations. For $T > T _ { * } ( \tilde { R ) }$ , the solution $u ( T )$ can be written as a (possibly complex) functional of $\{ u _ { 0 } , D u _ { 0 } , \ldots , D ^ { m } u _ { 0 } \}$ . The specific form of this functional (involving compositions of semigroups, integrals, and nonlinear operations) is precisely what the neural operator learns. Our contribution is showing that providing $\{ D ^ { j } u _ { 0 } \}$ as input features gives the network direct access to the quantities it needs, rather than requiring it to learn differentiation implicitly.

Stationary (elliptic) problems. For stationary PDEs like Darcy flow $- \nabla \cdot ( a \nabla p ) = f$ , there is no time evolution. The solution operator $\mathcal { G } : a \mapsto p$ depends directly on derivatives of the coefficient through the elliptic structure.

Expanding around a reference coefficient $\scriptstyle a _ { 0 } .$

$$
\begin{array} { r } { p [ a ] = p [ a _ { 0 } ] + \mathcal { L } ^ { - 1 } [ \nabla \cdot ( \delta a \nabla p [ a _ { 0 } ] ) ] + O ( \| \delta a \| ^ { 2 } ) , } \end{array}\tag{38}
$$

where ${ \mathcal { L } } = - { \boldsymbol { \nabla } } \cdot ( { \boldsymbol { a } } _ { 0 } { \boldsymbol { \nabla } } \cdot )$ and $\delta a = a - a _ { 0 }$ . The linear response involves $\nabla ( \delta a )$ , so the solution operator depends on $\{ a , \nabla a \}$ . Higher-order terms involve repeated first derivatives of δa rather than higher derivatives. We nevertheless supply $\nabla ^ { 2 } c$ a as a redundant feature, since the elliptic operator is second order, yielding the feature set $\{ \stackrel { . } { a } , \stackrel { . } { \nabla } a , \nabla ^ { 2 } a \}$ for this second-order operator.

The key insight is identical to the parabolic case: the PDE structure induces dependence on derivatives of the input, and providing these as features aids learning.

Quasilinear and non-polynomial nonlinearities. The analysis extends beyond strictly polynomial N . For quasilinear PDEs like Navier-Stokes, where the velocity $\mathbf { u } = \nabla ^ { \perp } \bar { \Delta } ^ { - 1 } \omega$ involves inverting the Laplacian, the key observation is that $\Delta ^ { - 1 }$ is a bounded operator on Sobolev spaces; it does not introduce new dependence on derivatives of $u _ { 0 }$ . What matters is that the PDE structure (through L and $\mathcal { N } )$ creates dependence on $\{ u _ { 0 } , D ^ { j } u _ { 0 } \} _ { j \le m } ;$ other operations (like $\Delta ^ { - 1 } )$ are learned by the neural operator. This justifies applying our framework to Navier-Stokes (with vorticity input ω) as a second-order problem.

## B.3 General Evolution Equations: A Spectral Argument

The Picard analysis of §B establishes derivative dependence under the parabolic hypotheses of Theorem 3.1 (semilinear evolution with semigroup-generating $\mathcal { L }$ and polynomial $\dot { \mathcal { N } } )$ . Here we present a complementary spectral argument that extends the same derivative structure to general evolution equations, including hyperbolic and fractional-in-time PDEs. This addresses the role of the temporal order in determining the derivative-feature structure.

Setup. Consider the abstract evolution problem

$$
\partial _ { t } ^ { a } u = \mathscr { L } u , \quad u ( \cdot , 0 ) = u _ { 0 } , \quad t \in [ 0 , T ] ,\tag{39}
$$

where $\partial _ { t } ^ { a }$ is the Caputo fractional derivative of order $a > 0$ and L is a linear operator whose Fourier symbol $\overset { \cdot } { P } ( \xi )$ satisfies $P ( \xi ) \asymp | \xi | ^ { m }$ as $| \xi | \to \infty$ . For $a > 1$ , the equation requires ⌈a⌉ initial conditions; we set the higher-order ones to zero for clarity (they enter linearly and admit the same analysis). Assume zero-mean initial data $\hat { u } _ { 0 } ( 0 ) = 0$ , standard for periodic domains and PDEs with vanishing constants.

Step 1: Fourier reduction. The spatial Fourier transform reduces (39) to a family of scalar fractional ODEs indexed by $\xi \colon$

$$
\partial _ { t } ^ { a } \hat { u } ( t , \xi ) = P ( \xi ) \hat { u } ( t , \xi ) , \quad \hat { u } ( 0 , \xi ) = \hat { u } _ { 0 } ( \xi ) .\tag{40}
$$

Step 2: Laplace transform and resolvent. Applying the Laplace transform in t and solving algebraically yields

$$
\hat { u } ( T , \xi ) = R _ { a } ( P ( \xi ) , T ) \cdot \hat { u } _ { 0 } ( \xi ) ,\tag{41}
$$

where the resolvent multiplier $R _ { a } ( z , T )$ is the inverse Laplace transform of $s ^ { a - 1 } / ( s ^ { a } - z )$ . Three canonical cases:

• Parabolic $\left( a = 1 \right) : R _ { 1 } ( z , T ) = e ^ { z T }$ (heat-type semigroup).

• Hyperbolic $( a = 2 ) \colon R _ { 2 } ( z , T ) = \cos ( \sqrt { - z } T )$ (wave-type oscillator).

• Fractional $( 0 < a < 2 , a \neq 1 ) \colon R _ { a } ( z , T ) = E _ { a } ( z T ^ { a } )$ , with $E _ { a }$ the Mittag-Leffler function.

Step 3: Factorization through $| \xi | ^ { - m }$ . The symbol satisfies $P ( \xi ) = \Theta ( | \xi | ^ { m } )$ , while $R _ { a } ( P ( \xi ) , T )$ is bounded uniformly in $\xi$ in each of the three cases above (the standard well-posedness condition

Re $P ( \xi ) \leq 0 )$ . The zero-mean condition $\hat { u } _ { 0 } ( 0 ) = 0$ ensures $| \xi | > 0$ on the support of $\hat { u } _ { 0 }$ , so we may factor

$$
R _ { a } ( P ( \xi ) , T ) \hat { u } _ { 0 } ( \xi ) = \underbrace { \frac { R _ { a } ( P ( \xi ) , T ) } { | \xi | ^ { m } } } _ { = : Q _ { a } ( \xi , T ) } \cdot \underbrace { | \xi | ^ { m } \hat { u } _ { 0 } ( \xi ) } _ { = \widehat { D ^ { m } u _ { 0 } } ( \xi ) } .\tag{42}
$$

The first factor $Q _ { a } ( \xi , T )$ is a Fourier multiplier of order $- m$ , bounded uniformly in ξ for fixed $T$ on the (discrete, zero-mean) periodic spectrum. The second factor is the Fourier transform of $D ^ { m } u _ { 0 }$ Inverting the transform gives

$$
\mathcal { G } [ u _ { 0 } ] ~ = ~ u ( \cdot , T ) ~ = ~ Q _ { a } ( T ) * D ^ { m } u _ { 0 } ,\tag{43}
$$

where $Q _ { a } ( T )$ is a bounded linear operator on $L ^ { 2 } ( \Omega )$

Step 4: Consequence for derivative features. Providing $D ^ { b } u _ { 0 }$ as an input feature for any $b \leq m$ reduces the residual differentiation burden from order m to order $m - b \colon$

$$
{ \mathcal G } [ u _ { 0 } ] = Q _ { a } ( T ) * D ^ { m - b } ( D ^ { b } u _ { 0 } ) .\tag{44}
$$

This is the spectral analogue of the Picard derivative structure (Theorem 3.1). It holds for every temporal order $a > 0$ for which $R _ { a }$ is well-defined, with the caveat that the operator norm $\lVert Q _ { a } ( T ) \rVert$ depends on a and T but not on the choice of input order b.

Nonlinear extension. For semilinear PDEs $\partial _ { t } ^ { a } u = \mathcal { L } u + \mathcal { N } [ u ]$ , linearizing around the linear flow $u ^ { \mathrm { l i n } } ( t ) = R _ { a } ( \mathcal { L } , t ) u _ { 0 }$ and applying Duhamel’s principle yields

$$
u ( T ) = R _ { a } ( \mathcal { L } , T ) u _ { 0 } + \int _ { 0 } ^ { T } R _ { a } ( \mathcal { L } , T - s ) \mathcal { N } [ u ^ { \mathrm { l i n } } ( s ) ] d s + \mathcal { R } ( u _ { 0 } , T ) ,\tag{45}
$$

with $\Vert \mathcal { R } \Vert _ { L ^ { 2 } } \leq C ( R ) T ^ { 2 a }$ uniformly on $B _ { s } ( R )$ under Assumption $4 . 2$ and the hypotheses of Theorem 3.1. The Frechet derivative of´ $\dot { \mathcal { G } }$ satisfies a linearized equation with the same resolvent structure, so the factorization through $| \xi | ^ { - m }$ persists term by term.

Role of the temporal order. The temporal order a enters through three quantities: (i) the number of initial conditions ⌈a⌉ (information at $t = 0 ) ; ( \mathrm { i i } )$ the specific functional form of $R _ { a }$ (heat for $a = 1$ wave for $a = 2$ , Mittag-Leffler for $0 < a < 2 )$ ; and (iii) the rate at which $\| Q _ { a } ( T ) \|$ decays in $T ,$ Crucially, a does not change the derivative-feature structure, which is determined entirely by the spatial order m via $P ( \xi ) \asymp | \xi | ^ { m }$ . This is why our framework, calibrated on parabolic $( a = 1 )$ Burgers and elliptic Darcy, transfers to other temporal regimes provided the spatial order m is matched.

## C Proof of Proposition 4.3: Bi-Lipschitz Stability

Proof. Let $\Phi ^ { [ m ] } [ u ] = ( u , D ^ { 1 } u , . . . , D ^ { m } u ) \operatorname* { m a p } H ^ { s } ( \Omega )  \mathcal { X } = H ^ { s } \times H ^ { s - 1 } \times \cdot \cdot \cdot \times H ^ { s - m } .$

Lower bound (information preservation). Since $\Phi ^ { [ m ] }$ includes u as its first component:

$$
\| \Phi ^ { [ m ] } [ u _ { 1 } ] - \Phi ^ { [ m ] } [ u _ { 2 } ] \| _ { \mathcal { X } } ^ { 2 } = \| u _ { 1 } - u _ { 2 } \| _ { H ^ { s } } ^ { 2 } + \sum _ { j = 1 } ^ { m } \| D ^ { j } ( u _ { 1 } - u _ { 2 } ) \| _ { H ^ { s - j } } ^ { 2 } \geq \| u _ { 1 } - u _ { 2 } \| _ { H ^ { s } } ^ { 2 } .\tag{46}
$$

Taking square roots gives the lower bound with constant 1.

Upper bound (Lipschitz continuity). By the definition of Sobolev norms:

$$
\| D ^ { j } u \| _ { H ^ { s - j } } ^ { 2 } = \int ( 1 + | \xi | ^ { 2 } ) ^ { s - j } | \xi | ^ { 2 j } | { \hat { u } } ( \xi ) | ^ { 2 } d \xi \leq \int ( 1 + | \xi | ^ { 2 } ) ^ { s } | { \hat { u } } ( \xi ) | ^ { 2 } d \xi = \| u \| _ { H ^ { s } } ^ { 2 } ,\tag{47}
$$

since $( 1 + | \xi | ^ { 2 } ) ^ { s - j } | \xi | ^ { 2 j } \leq ( 1 + | \xi | ^ { 2 } ) ^ { s }$ for $j \le m \le s$

Therefore:

$$
\begin{array} { r } { \| \Phi ^ { [ m ] } [ u _ { 1 } ] - \Phi ^ { [ m ] } [ u _ { 2 } ] \| _ { \mathcal { X } } ^ { 2 } \leq ( 1 + m ) \| u _ { 1 } - u _ { 2 } \| _ { H ^ { s } } ^ { 2 } . } \end{array}\tag{48}
$$

Taking square roots: $\begin{array} { r } { \| \Phi ^ { [ m ] } [ u _ { 1 } ] - \Phi ^ { [ m ] } [ u _ { 2 } ] \| _ { \mathcal { X } } \leq \sqrt { 1 + m } \| u _ { 1 } - u _ { 2 } \| _ { H ^ { s } } . } \end{array}$

## D Proof of Theorem 4.4: Approximation Rates

This section provides the approximation-theoretic analysis establishing that derivative features admit improved approximation bounds. This complements the bias-variance analysis in Section 5 and Appendix F: the approximation theory shows that features help (the established bound improves from $W ^ { - ( s - m ) / d } \mathrm { t } 0 W ^ { - s / d } )$ , while the bias-variance analysis determines which derivative order is optimal $( \beta ^ { * } < m )$ . The two frameworks answer different questions and use different tools; their qualitative agreement $( \beta ^ { * } < m$ is beneficial) provides evidence of robustness.

Proof. Part (i): Upper bound without features.

The operator $\mathcal { G }$ depends on derivatives up to order m. By Sobolev interpolation, for any $f \in H ^ { s }$ with $\| f \| _ { H ^ { s } } \leq R :$

$$
\left\| f \right\| _ { H ^ { m } } \leq C \| f \| _ { L ^ { 2 } } ^ { ( s - m ) / s } \| f \| _ { H ^ { s } } ^ { m / s } \leq C R ^ { m / s } \| f \| _ { L ^ { 2 } } ^ { ( s - m ) / s } .\tag{49}
$$

This interpolation inequality determines the modulus of continuity of G. Since G depends on $D ^ { m } u$ through a Lipschitz function on bounded sets (Theorem 3.1), the operator inherits Holder continuity¨ with exponent $( s - m ) / s \colon$

$$
\Vert \mathcal { G } [ u _ { 1 } ] - \mathcal { G } [ u _ { 2 } ] \Vert _ { L ^ { 2 } } \lesssim \Vert u _ { 1 } - u _ { 2 } \Vert _ { L ^ { 2 } } ^ { ( s - m ) / s } .\tag{50}
$$

The ϵ-covering number of $B _ { s } ( R )$ in $L ^ { 2 }$ satisfies log $\mathcal { N } ( \epsilon ) \asymp ( R / \epsilon ) ^ { d / s }$ . By the entropy lower bound of Carl and Stephani [25], approximating Holder- ¨ γ functions on sets with entropy exponent $\rho = d / s$ requires:

$$
\operatorname { e r r o r } \geq c \cdot W ^ { - \gamma s / d } = c \cdot W ^ { - ( s - m ) / d } .\tag{51}
$$

This is a worst-case statement over that function class, not a lower bound for G itself. Theorem 4.4 uses only the achievable side (Remark D.2). The achievable rate $W ^ { - ( s - m ) / d }$ follows from Yarotsky [26].

Part (ii): Upper bound with features.

Define $\tilde { \mathcal { G } } : \mathcal { X } \to L ^ { 2 }$ by $\tilde { \mathcal { G } } [ \Phi ^ { [ m ] } [ u ] ] : = \mathcal { G } [ u ]$

Step $\boldsymbol { { \mathit { 1 } } } \colon \tilde { \mathcal { G } }$ is Lipschitz on bounded balls.

Lemma D.1 (Lipschitz Property of Lifted Operator). On the image $\Phi ^ { [ m ] } [ B _ { s } ( R ) ]$ , the lifted operator $\tilde { \mathcal { G } }$ is Lipschitz:

$$
\| \tilde { \mathcal { G } } ( \Phi ^ { [ m ] } [ u _ { 1 } ] ) - \tilde { \mathcal { G } } ( \Phi ^ { [ m ] } [ u _ { 2 } ] ) \| _ { L ^ { 2 } } \leq L ( R ) \| \Phi ^ { [ m ] } [ u _ { 1 } ] - \Phi ^ { [ m ] } [ u _ { 2 } ] \| _ { \mathcal { X } } ,\tag{52}
$$

where $L ( R )$ depends polynomially on the ball radius R and the nonlinearity degree N.

Proof. The key insight is that providing derivative features removes the need to “invert” differentiation. By Theorem 3.1:

$$
\mathcal { G } [ u ] = G _ { T } * u + T \bar { G } _ { T } * \mathcal { N } [ u ] + \mathcal { R } ( u , T ) .\tag{53}
$$

The linear term $G _ { T } * u$ is bounded. The nonlinear term $\mathcal { N } [ u ]$ is a polynomial of degree $N$ in u and its derivatives up to order m. On bounded balls $B _ { s } ( R )$ , polynomial functions are Lipschitz: for any polynomial P of degree $N .$

$$
| P ( x ) - P ( y ) | \leq C _ { P } ( R ) | x - y | \quad { \mathrm { f o r ~ } } | x | , | y | \leq R ,\tag{54}
$$

where $C _ { P } ( R ) = O ( R ^ { N - 1 } )$ ). The remainder $\mathcal { R } ( u , T )$ is $O ( T ^ { 2 } )$ uniformly, hence also Lipschitz on bounded balls. Together, $\tilde { \mathcal { G } }$ is Lipschitz with constant $L ( R ) = O ( R ^ { N - 1 } )$ □

Step 2: Metric entropy. A Sobolev ball is not totally bounded in its own norm, so the covering step must use a weaker metric, and the choice matters: measuring inputs in $L ^ { 2 }$ gives log $\mathcal { N } ( B _ { s } ( R ) , \epsilon ; L ^ { 2 } ) \asymp ( R / \epsilon ) ^ { d / s }$ [25], while measuring the feature vector in the product- $L ^ { 2 }$ norm gives exponent $d / ( s - m )$ , driven by the roughest component $D ^ { m } u$ . Part (ii) uses the first reading, with Lipschitz continuity established in the feature-space metric (Step 1). Remark D.2 discusses the resulting scope. The bi-Lipschitz constant $\sqrt { 1 + m }$ (Proposition 4.3) is absorbed into the approximation constant.

Step 3: Approximation of Lipschitz functions. By Yarotsky [26] for finite-dimensional ReLU approximation and Kovachki et al. [3], Lanthaler et al. [43] for the operator-learning rate framework, Lipschitz functions on sets with metric entropy $( 1 / \epsilon ) ^ { d / s }$ can be approximated by deep ReLU networks with $W$ parameters to error:

$$
\operatorname { e r r o r } = O ( W ^ { - s / d } ) .\tag{55}
$$

This completes the proof.

□

Remark D.2 (Scope of the Comparison). Parts (i) and (ii) of Theorem 4.4 are achievability bounds under different input-side metrics ((i): L<sup>2</sup>; (ii): the product norm of X, Step 2). Under a common metric the two exponents coincide (the governing Holder-over-entropy ratio is ¨ $( s - m ) / d$ in both cases), so the bounds record what is established with and without derivative features, not a proven separation. Norm-constrained classes (realizing $D ^ { m }$ at spacing h costs $O ( h ^ { - m } )$ weights) and architectures with truncated internal spectral resolution (e.g., $\mathrm { \bar { F } N O ^ { \prime } s } k _ { \mathrm { m a x } } )$ remain routes to a genuine separation.

## E Proof of Proposition 5.3: Bias-Variance Decomposition

We provide the complete derivation with explicit constants.

## E.1 Assumptions

(A1) Bandlimited signals: Inputs satisfy supp $( \hat { u } ) \subseteq B _ { \xi _ { \operatorname* { m a x } } } ( 0 )$ , where $\xi _ { \operatorname* { m a x } } = \pi / h$ is the Nyquist frequency for grid spacing h.

(A2) Spectral decay: The input distribution has spectral density $S ( \xi ) = C _ { S } | \xi | ^ { - 2 s - d }$ for $\xi _ { 0 } \le$ $| \xi | \leq \xi _ { \mathrm { m a x } }$ , as required by Assumption 5.2. This is the borderline upper envelope for $H ^ { s }$ integrability.

(A3) Network capacity cutoff: We assume the network class $\mathcal { N O } _ { W }$ has fixed architecture independent of input features, hence fixed approximation capacity characterized by frequency cutoff $\xi _ { c } = C _ { N } W ^ { \gamma / d }$ , where $C _ { N } > 0$ is an architecture-dependent constant and $\gamma > 0$ is the capacity exponent. The derivative features $D ^ { \beta } u$ are computed as a preprocessing step before network evaluation; they do not modify the network’s representational capacity.

(A4) Input noise: Training inputs are observed as $\tilde { u } _ { i } = u _ { i } + \varepsilon _ { i }$ , with $\varepsilon _ { i }$ Gaussian white noise of spectral density $\sigma ^ { 2 }$ . Features are computed from $\tilde { u } _ { i }$

(A5) Sufficient resolution: We require $\xi _ { \operatorname* { m a x } } > e ^ { 1 / ( 2 \beta + d ) }$ , which for $\beta \leq 2$ and $d \leq 3$ is satisfied whenever $\xi _ { \mathrm { { m a x } } } > e ,$ , i.e., grid spacing $h < \pi / e \approx 1 . 1 6$ . This ensures variance monotonicity holds for all $\beta \geq 0$ . In practice, this is easily satisfied: standard PDE benchmarks use unit or 2π-periodic domains where $N \geq 1 6$ grid points already give $\xi _ { \operatorname* { m a x } } \ge 8 \gg e$

Remark E.1 (Justification of A3). The distinction between input features and network capacity is fundamental. The derivative features $D ^ { \beta } .$ u preprocess the input, exposing information that was latent in u. This changes what the operator “sees” (and hence which approximation problem it solves) but not what functions the network can represent.

Concretely, in the Fourier Neural Operator and related architectures, the network retains only a fixed number of Fourier modes $k _ { \mathrm { m a x } }$ (typically 12–16 for standard benchmarks). This architectural choice imposes a hard frequency cutoff $\xi _ { c } \propto k _ { \mathrm { m a x } }$ that is independent of the input features. Whether the input is u or $D ^ { \beta } u$ , the network processes the same number of modes. The preprocessing $D ^ { \beta } \iota$ amplifies high-frequency content in the input, but modes beyond $k _ { \mathrm { m a x } }$ are still truncated by the architecture. Thus $\xi _ { c }$ is determined by network design (width, depth, modes), not by input features.

## E.2 Bias Derivation

Proof. The target operator $\mathcal { G }$ depends on $D ^ { m } u$ through the PDE structure (Theorem 3.1). When we provide features $D ^ { \hat { \beta } } u$ with $\beta < m$ , there is a spectral mismatch: the network receives information about $| \xi | ^ { \beta } \hat { u } ( \xi )$ but needs information about $| \xi | ^ { \bar { m } } \hat { u } ( \xi )$

The residual operator (what the network must learn implicitly) maps $D ^ { \beta } u \mapsto D ^ { m } u .$ , which corresponds to multiplication by $| \xi | ^ { m - \beta }$ in frequency space. For frequencies $| \xi | > \xi _ { c }$ (beyond network

capacity), this cannot be learned, contributing to bias:

$$
B ( \beta ) = C _ { B } \int _ { | \xi | > \xi _ { c } } | \xi | ^ { 2 ( m - \beta ) } S ( \xi ) d \xi .\tag{56}
$$

Converting to spherical coordinates with $\omega _ { d } = 2 \pi ^ { d / 2 } / \Gamma ( d / 2 )$ and using $S ( \xi ) = C _ { S } | \xi | ^ { - 2 s - d }$ from Assumption 5.2:

$$
B ( \beta ) = C _ { B } C _ { S } \omega _ { d } \int _ { \xi _ { c } } ^ { \infty } r ^ { 2 ( m - \beta ) } \cdot r ^ { - 2 s - d } \cdot r ^ { d - 1 } d r = C _ { B } C _ { S } \omega _ { d } \int _ { \xi _ { c } } ^ { \infty } r ^ { - \tau ( \beta ) - 1 } d r ,\tag{57}
$$

where $\tau ( \beta ) : = 2 ( s - m + \beta )$ . For convergence we need $\tau ( \beta ) > 0$ , which holds for $\beta$ near m when $s > m - \beta$ . Then:

$$
\int _ { \xi _ { c } } ^ { \infty } r ^ { - \tau ( \beta ) - 1 } d r = \frac { 1 } { \tau ( \beta ) } \xi _ { c } ^ { - \tau ( \beta ) } .\tag{58}
$$

Therefore:

$$
\boxed { B ( \beta ) = \frac { C _ { B } C _ { S } \omega _ { d } } { \tau ( \beta ) } \cdot \xi _ { c } ^ { - \tau ( \beta ) } , \quad \tau ( \beta ) = 2 ( s - m + \beta ) . }\tag{59}
$$

(With band-limited $S ,$ the upper limit is $\xi _ { \mathrm { m a x } } .$ , and the omitted tail is subleading.)

$\operatorname { A s } \beta$ increases, $\tau ( \beta )$ increases, so $\xi _ { c } ^ { - \tau ( \beta ) }$ decreases (for $\xi _ { c } > 1 )$ . Thus $B ( \beta )$ is strictly decreasing in $\beta .$ □

## E.3 Variance Derivation

Proof. The variance measures noise amplification through fractional derivatives. When we compute $D ^ { \beta } \iota$ u from noisy data, observation noise ϵ with variance $\overline { { \sigma ^ { 2 } } }$ gets amplified: at frequency $\xi ,$ the noise power becomes $\sigma ^ { 2 } | \boldsymbol { \xi } | ^ { 2 \beta }$

The total variance contribution is:

$$
V ( \beta ) = { \frac { \sigma ^ { 2 } C _ { V } } { n } } \int _ { | \xi | \leq \xi _ { \operatorname* { m a x } } } | \xi | ^ { 2 \beta } d \xi .\tag{60}
$$

Converting to spherical coordinates, with $\xi _ { 0 } ~ = ~ 2 \pi / ( N h )$ the fundamental frequency (smallest non-zero wavenumber for $N$ grid points with spacing h):

$$
V ( \beta ) = \frac { \sigma ^ { 2 } C _ { V } \omega _ { d } } { n } \int _ { \xi _ { 0 } } ^ { \xi _ { \operatorname* { m a x } } } r ^ { 2 \beta + d - 1 } d r = \frac { \sigma ^ { 2 } C _ { V } \omega _ { d } } { n ( 2 \beta + d ) } \left( \xi _ { \operatorname* { m a x } } ^ { 2 \beta + d } - \xi _ { 0 } ^ { 2 \beta + d } \right) .\tag{61}
$$

For $\xi _ { \mathrm { m a x } } \gg \xi _ { 0 }$ (which holds for any reasonable discretization):

$$
\boxed { V ( \beta ) \approx \frac { \sigma ^ { 2 } C _ { V } \omega _ { d } } { n ( 2 \beta + d ) } \cdot \xi _ { \mathrm { m a x } } ^ { 2 \beta + d } . }\tag{62}
$$

Monotonicity: Taking the derivative with respect to $\beta \colon$

$$
\frac { \partial V } { \partial \beta } = \frac { \sigma ^ { 2 } C _ { V } \omega _ { d } } { n } \cdot \frac { \xi _ { \operatorname* { m a x } } ^ { 2 \beta + d } } { ( 2 \beta + d ) ^ { 2 } } \left[ 2 ( 2 \beta + d ) \ln \xi _ { \operatorname* { m a x } } - 2 \right] > 0\tag{63}
$$

for $\xi _ { \operatorname* { m a x } } > e ^ { 1 / ( 2 \beta + d ) }$ , which holds for typical discretizations where $\xi _ { \operatorname* { m a x } } \gg 1$

Thus $V ( \beta )$ is strictly increasing in $\beta \colon$ higher derivative orders amplify noise more.

## E.4 Monotonicity and Existence of Optimum

Proof. Monotonicity: From the closed forms:

$B ( \beta )$ is strictly decreasing: as $\beta$ increases, $\tau ( \beta ) = 2 ( s - m + \beta )$ increases, so $\xi _ { c } ^ { - \tau ( \beta ) }$ decreases.

$V ( \beta )$ is strictly increasing: as $\beta$ increases, $\xi _ { \operatorname* { m a x } } ^ { 2 \beta + d }$ increases faster than the prefactor $1 / ( 2 \beta +$ d) decreases.

## Boundary behavior:

• As β → 0: B(0) < ∞ (finite bias), V (0) = K<sub>V</sub>σ<sup>2</sup>ξ<sup>d</sup><sub>max</sub>/(nd) < ∞.

• As β → m: B(m) = K<sub>B</sub>ξ<sup>−2s</sup> (small bias), $V ( m ) = K _ { V } \sigma ^ { 2 } \xi _ { \mathrm { m a x } } ^ { 2 m + d } / ( n ( 2 m + d ) )$ (larger variance).

$$
\bullet \mathrm { ~ A s ~ } \beta \to \infty \colon B ( \beta ) \to 0 , V ( \beta ) \to \infty .
$$

Since $B ^ { \prime } ( \beta ) < 0$ and $V ^ { \prime } ( \beta ) > 0$ , we examine the total derivative MSE $' ( \beta ) = B ^ { \prime } ( \beta ) + V ^ { \prime } ( \beta )$ . In the intermediate data regime (neither clip of Theorem 5.6 binding): $\mathbf { A } \mathbf { t } \ \beta = 0 \mathbf { : }$ : the bias derivative dominates (large $| B ^ { \prime } ( 0 ) |$ |, small $V ^ { \prime } ( 0 ) )$ , so $\mathrm { M S E } ^ { \prime } ( 0 ) ~ < ~ 0$ . At $\beta \ : = \ : m \colon$ the variance derivative dominates, so $\bar { \mathrm { M S E } } ^ { \prime } ( \dot { m } ) \stackrel { . } { > } 0$ . By the Intermediate Value Theorem applied to the continuous function $\mathrm { M S E } ^ { \prime } ( \beta )$ , there exists $\beta ^ { * } \in ( 0 , m )$ where $\mathrm { M S E } ^ { \prime } ( \beta ^ { * } ) = 0$

Convexity and Uniqueness: We verify $B ^ { \prime \prime } ( \beta ) > 0$ and $V ^ { \prime \prime } ( \beta ) > 0 \mathrm { d i r e c t l y }$

For the bias $B ( \beta ) = K \xi _ { c } ^ { - \tau ( \beta ) } / \tau ( \beta )$ with $\tau ( \beta ) = 2 ( s - m + \beta )$

$$
B ^ { \prime } ( \beta ) = K \xi _ { c } ^ { - \tau } \left( \frac { - 2 \ln \xi _ { c } } { \tau } - \frac { 2 } { \tau ^ { 2 } } \right) = - \frac { 2 K \xi _ { c } ^ { - \tau } } { \tau ^ { 2 } } ( \tau \ln \xi _ { c } + 1 ) .\tag{64}
$$

Taking another derivative (and using $\tau ^ { \prime } = 2 )$ :

$$
B ^ { \prime \prime } ( \beta ) = \frac { 4 K \xi _ { c } ^ { - \tau } } { \tau ^ { 3 } } \left[ ( \tau \ln \xi _ { c } + 1 ) ^ { 2 } + 1 \right] .\tag{65}
$$

The dominant term $( \tau \ln \xi _ { c } + 1 ) ^ { 2 } > 0$ ensures $B ^ { \prime \prime } ( \beta ) > 0$ for $\xi _ { c } > 1$

For the variance $V ( \beta ) = K ^ { \prime } \xi _ { \mathrm { m a x } } ^ { 2 \beta + d } / ( 2 \beta + d )$

$$
V ^ { \prime } ( \beta ) = K ^ { \prime } \xi _ { \mathrm { m a x } } ^ { 2 \beta + d } \left( \frac { 2 \ln \xi _ { \mathrm { m a x } } } { 2 \beta + d } - \frac { 2 } { ( 2 \beta + d ) ^ { 2 } } \right) .\tag{66}
$$

The second derivative $V ^ { \prime \prime } ( \beta ) > 0$ follows similarly, with the $( \ln \xi _ { \mathrm { m a x } } ) ^ { 2 }$ term dominating.

Thus $\mathrm { M S E } ( \beta ) = B ( \beta ) + V ( \beta )$ is strictly convex, guaranteeing the critical point $\beta ^ { * }$ is unique and is the global minimum. □

## F Proof of Theorem 5.6: Optimal Derivative Order

We derive the optimal $\beta ^ { * }$ for fixed network capacity.

## F.1 Setup

From Proposition 5.3, write the MSE using explicit forms. Let $\delta = m - \beta$ denote the gap from the PDE order, so $\delta \geq 0$ when $\beta \leq m$ . The network capacity cutoff $\xi _ { c } = C _ { N } W ^ { \gamma / d }$ is independent of $\beta .$

For the bias (with $\tau ( \beta ) = 2 ( s - m + \beta ) = 2 ( s - \delta )$ where we use $\delta = m - \beta )$

$$
B ( \delta ) = \frac { C _ { B } C _ { S } \omega _ { d } } { 2 ( s - \delta ) } \cdot \xi _ { c } ^ { - 2 ( s - \delta ) } = K _ { B } \cdot W ^ { - 2 \gamma ( s - \delta ) / d } \cdot \frac { 1 } { s - \delta } ,\tag{67}
$$

where $K _ { B }$ collects constants. $\operatorname { A s } \delta$ increases $( \mathrm { i . e . , } \beta$ decreases), the exponent $- 2 ( s - \delta )$ becomes less negative, so $B ( \delta )$ increases.

For the variance, using $\beta = m - \delta \colon$

$$
V ( \delta ) = \frac { \sigma ^ { 2 } C _ { V } \omega _ { d } } { n ( 2 ( m - \delta ) + d ) } \cdot \xi _ { \mathrm { m a x } } ^ { 2 ( m - \delta ) + d } = K _ { V } \frac { \sigma ^ { 2 } } { n } \cdot \xi _ { \mathrm { m a x } } ^ { 2 m + d } \cdot \xi _ { \mathrm { m a x } } ^ { - 2 \delta } ,\tag{68}
$$

where $K _ { V }$ absorbs the slowly-varying prefactor $1 / ( 2 ( m - \delta ) + d )$ . As δ increases $( \mathrm { i . e . , } \beta$ decreases), the factor $\xi _ { \operatorname* { m a x } } ^ { - 2 \delta }$ decreases, so $V ( \delta )$ decreases.

## F.2 Derivation

Proof. With W fixed, denote $\tau _ { 0 } : = 2 ( s - m )$ (the bias exponent $\tau ( \beta ) = 2 ( s - m + \beta )$ evaluated at $\beta = 0 ;$ the exponent at $\beta = m \mathrm { i s } 2 s = \tau _ { 0 } + 2 m )$ ).

The bias at gap δ is:

$$
B ( \delta ) = K _ { B } \cdot \xi _ { c } ^ { - 2 ( s - \delta ) } = K _ { B } ^ { \prime } \xi _ { c } ^ { - \tau _ { 0 } } \cdot \xi _ { c } ^ { 2 \delta } ,\tag{69}
$$

where $K _ { B } ^ { \prime } = K _ { B } \xi _ { c } ^ { - 2 m }$ absorbs the factor relating $- 2 ( s - \delta ) \ : \mathrm { t o } - \tau _ { 0 } + 2 \delta = - 2 ( s - m ) + 2 \delta .$

The variance at gap $\delta$ is:

$$
V ( \delta ) = K _ { V } \frac { \sigma ^ { 2 } } { n } \cdot \xi _ { \mathrm { m a x } } ^ { 2 ( m - \delta ) + d } = K _ { V } \frac { \sigma ^ { 2 } } { n } \xi _ { \mathrm { m a x } } ^ { 2 m + d } \cdot \xi _ { \mathrm { m a x } } ^ { - 2 \delta } .\tag{70}
$$

Note: As δ increases (moving away from the PDE order $m )$ , bias $B ( \delta )$ increases (less accurate features) while variance $V ( \delta )$ decreases (less noise amplification). The optimal $\delta ^ { * }$ balances these.

Setting $\partial _ { \delta } ( B + V ) = 0 \colon$

$$
2 K _ { B } ^ { \prime } \xi _ { c } ^ { - \tau _ { 0 } } \xi _ { c } ^ { 2 \delta } \ln \xi _ { c } = 2 K _ { V } \frac { \sigma ^ { 2 } } { n } \xi _ { \mathrm { { m a x } } } ^ { 2 m + d } \xi _ { \mathrm { { m a x } } } ^ { - 2 \delta } \ln \xi _ { \mathrm { { m a x } } } .\tag{71}
$$

Taking logarithms and solving for $\delta \colon$

$$
2 \delta ( \ln \xi _ { c } + \ln \xi _ { \mathrm { m a x } } ) = \underbrace { \ln ( K _ { V } \xi _ { \mathrm { m a x } } ^ { 2 m + d } / K _ { B } ^ { \prime } \xi _ { c } ^ { - \tau _ { 0 } } ) + \ln ( \ln \xi _ { \mathrm { m a x } } / \ln \xi _ { c } ) } _ { = : C _ { 0 } } + \ln ( \sigma ^ { 2 } / n ) .\tag{72}
$$

Since $\ln ( \sigma ^ { 2 } / n ) = - \ln ( n / \sigma ^ { 2 } )$ , this gives:

$$
\boxed { \delta ^ { * } = \frac { C _ { 0 } - \ln ( n / \sigma ^ { 2 } ) } { 2 \ln ( \xi _ { c } \xi _ { \mathrm { m a x } } ) } \quad \mathrm { f o r } n / \sigma ^ { 2 } < e ^ { C _ { 0 } } . }\tag{73}
$$

Interpretation: The constant $C _ { 0 } = 2 s \ln \xi _ { c } + ( 2 m + d ) \ln \xi _ { \operatorname* { m a x } } + O ( 1 )$ (the ln $\xi _ { c }$ coefficient combines the $\tau _ { 0 } \ln \xi _ { c }$ from the boxed equation and the −2m ln $\xi _ { c }$ inside ln $K _ { B } ^ { \prime } { \mathrm { : } }$ , which together give $\tau _ { 0 } + 2 m = 2 s )$ is large and positive. When $n / \sigma ^ { 2 } < e ^ { C _ { 0 } }$ (finite data regime), we have $\delta ^ { * } > 0 .$ so $\beta ^ { * } = m - \delta ^ { * } < m$ . As $n \to \infty \mathrm { o r } \sigma \to 0 , \dot { \delta } ^ { * } \to 0$ and $\beta ^ { * }  m \colon$ physics is recovered. When $n / \sigma ^ { 2 } \geq e ^ { C _ { 0 } }$ , we set $\delta ^ { * } = 0$ (no regularization needed).

With $\xi _ { c } = C _ { N } W ^ { \gamma / d }$ , the gap $\delta ^ { * }$ decreases linearly in ln $\scriptstyle ( n / \sigma ^ { 2 } )$ to leading order. More data enables higher derivative orders. □

## F.3 Key Properties

1. $\beta ^ { * } < m$ strictly in the practical regime: Whenever $n / \sigma ^ { 2 } < e ^ { A _ { c } }$ (Remark $5 . 7 ) , \delta ^ { * } > 0$ and hence $\beta ^ { * } < m$ . Since the threshold $n ^ { \star } = \sigma ^ { 2 } \exp ( \dot { A } _ { c } )$ is unreachable in any realistic training budget, this is the operational regime.

2. Asymptotic recovery: $\operatorname { A s } n \to \infty \operatorname { o r } \sigma \to 0$ , we have $\delta ^ { * } \to 0$ , hence $\beta ^ { * } \to m$

3. Logarithmic closure: To leading order in 1/ ln $\xi _ { c } , \delta ^ { * }$ decreases linearly in $\ln ( n / \sigma ^ { 2 } )$ with slope $- 1 / [ 2 \ln ( \xi _ { c } \xi _ { \mathrm { m a x } } ) ]$ , approaching zero as $n / \sigma ^ { 2 }  e ^ { A _ { c } }$

4. Requirement $s > m { : }$ The condition $\tau > 0$ requires input smoothness s to exceed PDE order m. This is natural: inputs must be smooth enough that the operator is well-posed.

## F.4 Beyond Leading Order: The $O ( 1 / \ln \xi _ { c } )$ Correction

The leading-order $\delta ^ { * }$ derived in $\ S \mathrm { F }$ absorbs prefactor and ln ln contributions into an $O ( 1 / \ln \xi _ { c } )$ remainder. Carrying this remainder explicitly is useful both as a consistency check and to expose which terms depend on the (unknown) integral constants $C _ { B } , C _ { S } , C _ { V }$

Write $\begin{array} { r } { L _ { c } : = \ln \xi _ { c } , L _ { x } : = \ln \xi _ { \mathrm { m a x } } , a : = 2 ( s - \delta ) , q : = 2 ( m - \delta ) + d , \mathrm { s o } B = \frac { C _ { B } C _ { S } \omega _ { d } } { a } \xi _ { c } ^ { - a } } \end{array}$ and $\begin{array} { r } { V = \frac { \sigma ^ { 2 } C _ { V } \omega _ { d } } { n q } \xi _ { \mathrm { m a x } } ^ { q } } \end{array}$ , with $a ^ { \prime } = q ^ { \prime } = - 2$ under $\partial _ { \delta }$ . The exact derivatives are

$$
\begin{array} { r } { \partial _ { \delta } B = B \Big ( 2 L _ { c } + \frac { 2 } { a } \Big ) , \qquad \partial _ { \delta } V = V \Big ( - 2 L _ { x } + \frac { 2 } { q } \Big ) , } \end{array}\tag{74}
$$

so the stationarity condition $\partial _ { \delta } ( B + V ) = 0$ is equivalent to the exact implicit relation

$$
- a L _ { c } - q L _ { x } + \ln { \frac { n } { \sigma ^ { 2 } } } + \kappa + \ln { \frac { q } { a } } \ = \ \ln \Bigl ( L _ { x } - { \textstyle { \frac { 1 } { q } } } \Bigr ) - \ln \Bigl ( L _ { c } + { \textstyle { \frac { 1 } { a } } } \Bigr ) , \qquad \kappa : = \ln { \frac { C _ { B } C _ { S } } { C _ { V } } } .\tag{75}
$$

(The $\omega _ { d }$ factors cancel between bias and variance.) Equation (75) is transcendental; expanding the right-hand logs in $1 / L _ { c } , 1 / L _ { x }$ and isolating $\delta$ yields

$$
\delta ^ { * } \ = \ \delta _ { 0 } ^ { * } + \frac { \Delta _ { 1 } } { 2 \ln ( \xi _ { c } \xi _ { \mathrm { m a x } } ) } , \qquad \delta _ { 0 } ^ { * } = \frac { A _ { c } - \ln ( n / \sigma ^ { 2 } ) } { 2 \ln ( \xi _ { c } \xi _ { \mathrm { m a x } } ) } ,\tag{76}
$$

with $A _ { c } = 2 s \ln \xi _ { c } + ( 2 m + d ) \ln \xi _ { \operatorname* { m a x } }$ as in Theorem 5.6 and

$$
\Delta _ { 1 } = \ln \frac { L _ { x } } { L _ { c } } - \ln \frac { q _ { 0 } } { a _ { 0 } } - \kappa - \frac { 1 } { q _ { 0 } L _ { x } } - \frac { 1 } { a _ { 0 } L _ { c } } + { \cal O } \big ( ( \ln \xi ) ^ { - 2 } \big ) ,\tag{77}
$$

where $a _ { 0 } = 2 ( s - \delta _ { 0 } ^ { * } )$ and $q _ { 0 } = 2 ( m - \delta _ { 0 } ^ { * } ) + d$ evaluate the slowly varying exponents at the leadingorder optimum. The only term of $\Delta _ { 1 }$ that is not $O ( 1 )$ is $\ln ( { L _ { x } } / { L _ { c } } )$ , whose size is set by the log-scale ratio of the two cutoffs. Since the capacity cutoff cannot exceed Nyquist $( \xi _ { c } \le \xi _ { \mathrm { m a x } } )$ and both derive from the same discretization, this ratio is bounded (in our benchmarks ln $( L _ { x } / L _ { c } ) \in [ 0 . 2 , 0 . 6 ] )$ , so $\Delta _ { 1 } = O ( 1 )$ and the correction is $O ( 1 / \ln \xi _ { c } )$ , exactly as stated in Theorem $5 . 6 .$ Only in the non-physical limit ln $\xi _ { \operatorname* { m a x } } / \ln \xi _ { c } \to \infty$ does it relax to the conservative bound $O ( \ln \ln \xi / \ln \dot { \xi } )$

Three structural observations. First, every term of $\Delta _ { 1 }$ except κ depends only on the cutoffs $\left( \xi _ { c } , \xi _ { \operatorname* { m a x } } \right)$ and the exponents $( s , m , d )$ ; the integral prefactor constants enter only through $\kappa .$ This is the same property that lets $C _ { S }$ cancel in Theorem 5.6: stating the result at leading order is the principled choice, not merely the simplest one. Second, κ is independent of n and $\sigma ,$ so it cancels in the differential predictions of $\ S 7 . 3 \colon$ the $\sigma \mathrm { - }$ and n-sweep slopes sharpen by including $\Delta _ { 1 }$ without any unknown constant. The k-sweep retains $\kappa ,$ which need not agree across datasets with different spectra, so we read it qualitatively. Third, the expansion describes an interior optimum, valid when $0 < \delta _ { 0 } ^ { * } < \operatorname* { m i n } ( m , s )$ (so that $\beta ^ { * } > 0$ and $a _ { 0 } > 0 )$ . Outside this range the optimum is pinned at a boundary of the admissible set and the stationary expansion does not apply.

Numerically, where the interior expansion applies, $\Delta _ { 1 } / 2 \ln ( \xi _ { c } \xi _ { \mathrm { m a x } } )$ is at most a few hundredths in $\delta ^ { * }$ at benchmark-scale cutoffs, below the resolution of the learned-β measurements and comparable to the modeling idealizations already in place (hard-versus-soft capacity cutoff, borderline spectral envelope). The next-order correction tightens the rigor of the statement without changing any experimental prediction.

## G Spectral Regularization and Tikhonov Interpretation

Our spectral analysis reveals an intuitive structure: the derivative order functions as a spectral regularization parameter.

Remark G.1 (Tikhonov Interpretation). Using features $D ^ { \beta }$ u with $\beta < m$ corresponds to Tikhonov regularization with frequency-dependent penalty:

$$
\lambda ( \xi ) = \sigma ^ { 2 } \cdot \operatorname* { m a x } \left( | \xi | ^ { 2 ( m - \beta ) } - 1 , 0 \right) .\tag{78}
$$

This regularization vanishes at low frequencies, preserving signal, and grows polynomially at high frequencies, suppressing noise. Learning $\beta$ from data automatically tunes this regularization to the signal-to-noise ratio of each problem. The remainder of this appendix gives the derivation sketch.

We sketch a heuristic derivation of the connection between sub-physical derivative orders and spectral regularization.

Consider learning a linear operator $\mathcal { K } : H ^ { m } \to L ^ { 2 }$ from noisy data. The target is $\mathcal { \kappa } [ D ^ { m } u ]$ but we observe $D ^ { \beta } u + \epsilon$ where ϵ is noise.

Frequency-space formulation. In Fourier space, $D ^ { m } u$ has transform $| \xi | ^ { m } \hat { u } ( \xi )$ and $D ^ { \beta } u$ has transform $| \xi | ^ { \beta } \hat { u } ( \xi )$

The optimal linear estimator of $| \xi | ^ { m } \hat { u } ( \xi )$ from noisy observation $| \xi | ^ { \beta } \hat { u } ( \xi ) + \hat { \epsilon } ( \xi )$ is the Wiener filter:

$$
\hat { v } ( \xi ) = \frac { | \xi | ^ { 2 \beta } S ( \xi ) } { | \xi | ^ { 2 \beta } S ( \xi ) + \sigma ^ { 2 } / n } \cdot | \xi | ^ { m - \beta } \cdot \mathrm { ( o b s e r v a t i o n ) } .\tag{79}
$$

Effective regularization. The denominator introduces regularization. The effective Tikhonov penalty at frequency $\xi$ is:

$$
\lambda ( \xi ) = \frac { \sigma ^ { 2 } / n } { | \xi | ^ { 2 \beta } S ( \xi ) } .\tag{80}
$$

With $S ( \xi ) \sim | \xi | ^ { - \alpha }$ , this gives $\lambda ( \xi ) \propto | \xi | ^ { - 2 \beta + \alpha }$

Comparison to using order m. If we used $D ^ { m } u$ directly, the penalty would be $\lambda _ { m } ( \xi ) \propto | \xi | ^ { - 2 m + \alpha }$ The ratio $\lambda ( \xi ) / \lambda _ { m } ( \xi ) = | \xi | ^ { 2 ( m - \beta ) }$ grows with |ξ| for $\beta < m ,$ , meaning using $\beta < m$ applies more regularization at high frequencies.

Effective penalty formula. Normalizing to have $\lambda ( 1 ) = 0$ for unit frequency:

$$
\lambda ( \xi ) = \sigma ^ { 2 } \cdot \operatorname* { m a x } \left( | \xi | ^ { 2 ( m - \beta ) } - 1 , 0 \right) .\tag{81}
$$

This vanishes at low frequencies (preserving signal) and grows as $| \xi | ^ { 2 \delta }$ at high frequencies (suppressing noise), where $\delta = m - \beta$

## H Grunwald-Letnikov Implementation¨

For non-periodic domains, we use the Grunwald-Letnikov (GL) formulation, which computes¨ fractional derivatives via convolution [44, Chapters 2, 7]. The GL fractional derivative is:

$$
D ^ { \beta } u ( x ) = \operatorname* { l i m } _ { h  0 } \frac { 1 } { h ^ { \beta } } \sum _ { k = 0 } ^ { \infty } w _ { k } ^ { ( \beta ) } u ( x - k h ) , \quad \mathrm { w h e r e ~ } w _ { 0 } = 1 , \ w _ { k } = w _ { k - 1 } \cdot \frac { k - 1 - \beta } { k } .\tag{82}
$$

For integer $\beta = n ,$ this recovers the standard n-th derivative via finite differences. The one-sided GL derivative introduces a phase shift; we use the centered version $\begin{array} { r } { D _ { \mathrm { c e n } } ^ { \beta } u = \frac { 1 } { 2 } ( D _ { + } ^ { \beta } u + \cos ( \pi \beta ) D _ { - } ^ { \beta } u ) } \end{array}$ where the factor cos(πβ) ensures correct symmetry (antisymmetric for $\beta = 1$ , symmetric for $\beta = 2 )$

Differentiability. The weights depend smoothly on $\beta$ via $\partial w _ { k } / \partial \beta = w _ { k } [ \psi ( \beta + 1 ) - \psi ( \beta - k + 1 ) ]$ where $\psi = \Gamma ^ { \prime } / \Gamma$ is the digamma function. This enables gradient-based optimization of derivative orders.

Complexity and validation. For input length $N$ and stencil length $K _ { \mathrm { G L } }$ , the cost is $O ( N K _ { \mathrm { G L } } )$ via direct convolution (we use $K _ { \mathrm { G L } } = 1 6 ) $ . On periodic domains, GL and FFT (Weyl) implementations agree to relative $L ^ { 2 } \mathrm { e r r o r } < 1 \bar { 0 } ^ { - 3 }$ at integer orders. At non-integer $\beta ,$ , the truncated centered stencil realizes a bounded gain-and-phase modification of the Weyl symbol, absorbed by the learnable scales $s _ { k }$ . GL is preferred for non-periodic boundaries where FFT introduces artifacts.

## I Experimental Details

All experiments use $n _ { \mathrm { t r a i n } } = 1 0 0 0$ training samples, unless otherwise noted $( \mathrm { e . g . }$ ., the sample-size sweep in $\ S 7 )$ , and $n _ { \mathrm { t e s t } } = 2 0 0$ test samples. Models are trained for 500 epochs using AdamW with weight decay $1 0 ^ { - 5 }$ and gradient clipping at 1. All results are averaged over 5 random seeds.

Table 4: Architecture hyperparameters by backbone.
<table><tr><td>Backbone</td><td>Width</td><td>Modes</td><td>Layers</td><td>Heads</td><td>Slices</td></tr><tr><td>FNO</td><td>64-128</td><td>12-16</td><td>4</td><td>一</td><td></td></tr><tr><td>Transolver</td><td>128-256</td><td>一</td><td>8</td><td>8</td><td>32-64</td></tr></table>

## I.1 Architecture Hyperparameters

Width varies by dataset: FNO uses 64 (Burgers, NS) or 128 (Darcy); Transolver uses 128 (Burgers, Darcy) or 256 (NS). Transolver slice count is 64 (Burgers, Darcy) or 32 (NS). FNO uses 16 Fourier modes for Burgers and 12 for Darcy, NS, and KdV.

## I.2 Training Hyperparameters

Table 5: Training hyperparameters by dataset.
<table><tr><td>Dataset</td><td>Resolution</td><td>Batch</td><td>LR</td><td>Scheduler</td><td> $T _ { \mathbf { i n } } / T _ { \mathbf { o u t } }$ </td></tr><tr><td>Burgers</td><td>1024</td><td>8</td><td> $1 0 ^ { - 3 }$ </td><td> $_ \mathrm { O n e C y c l e }$ </td><td></td></tr><tr><td>KdV</td><td>1024</td><td>8</td><td> $1 0 ^ { - 3 }$ </td><td> $_ \mathrm { O n e C y c l e }$ </td><td></td></tr><tr><td>Darcy</td><td> $8 5 \times 8 5$ </td><td>4</td><td> $1 0 ^ { - 3 }$ </td><td>OneCycle</td><td></td></tr><tr><td>Navier-Stokes</td><td> $6 4 \times 6 4$ </td><td>2-20</td><td> $5 \times 1 0 ^ { - 4 }$ </td><td>Step</td><td>10/10</td></tr></table>

OneCycle uses pct start= 0.3. Step scheduler uses step size 100 and $\gamma = 0 . 5$ . NS batch size is 20 (FNO) or 2 (Transolver).

KdV dataset generation. The KdV data are produced by direct numerical integration of $\partial _ { t } u + u \partial _ { x } u + \delta \partial _ { x x x } u = 0$ with dispersion coefficient $\delta \stackrel { \cdot } { = } 2 . 5 \times 1 0 ^ { - 3 }$ , on the periodic domain $[ 0 , 2 \pi )$ discretized with $N { = } 1 0 2 4$ points. Initial conditions are smooth random fields $u _ { 0 } ( x ) =$ $\textstyle \sum _ { j = 1 } ^ { J } ( a _ { j } / j ) \sin ( j x + \phi _ { j } )$ with $J \sim \mathrm { U n i f } \{ 3 , \dots , 7 \} , a _ { j } \sim \mathrm { U n i f } ( - 1 , 1 )$ , and $\phi _ { j } \sim \mathrm { U n i f } ( 0 , 2 \pi )$ (fixed seed for reproducibility). Time integration uses a pseudo-spectral integrating-factor scheme: the dispersive term is advanced exactly in Fourier space via $e ^ { - \hat { \delta } ( i k ) ^ { 3 } \Delta t }$ , while the nonlinear flux $- \partial _ { x } ( u ^ { \bar { 2 } } / 2 )$ is advanced by explicit Euler, with step $\bar { \Delta } t = 1 0 ^ { - 4 }$ to final time $T { = } 1 ( 1 0 ^ { 4 }$ steps). The learned map is $u ( \cdot , 0 ) \mapsto \bar { u } ( \cdot , \bar { T } )$ , with 1000 training and 200 test trajectories.

## I.3 ∂-NO Augmentation

All ∂-NO variants use identical derivative feature settings: Grunwald-Letnikov stencil length¨ $K _ { \mathrm { { G L } } } =$ 16, maximum derivative order $m _ { \mathrm { m a x } } = 2 \left( m _ { \mathrm { m a x } } = 3 \right.$ for KdV), and β learning rate fixed at $1 0 ^ { - 2 }$ (10× the $1 0 ^ { - 3 }$ backbone LR for Burgers/Darcy, 20× the $5 \times 1 0 ^ { - 4 }$ backbone LR for NS). Derivative orders are initialized at $\beta = 1 . 0$ and $\beta = 2 . { \overset { \cdot } { 0 } }$ , plus $\beta _ { 3 } = 3 . 0$ for KdV. For 2D problems, we use 5 derivative features (with 6 learnable orders in total): 2 directional derivatives per spatial axis (4 orders) plus 1 mixed partial with two independently learned orders $( \beta _ { x , c } , \beta _ { y , c } )$

## I.4 Evaluation

We report relative $L ^ { 2 }$ error:

$$
\mathrm { E r r o r } = \frac { \| \mathcal { G } _ { \theta } [ u ] - \mathcal { G } ^ { \dag } [ u ] \| _ { L ^ { 2 } } } { \| \mathcal { G } ^ { \dag } [ u ] \| _ { L ^ { 2 } } } ,\tag{83}
$$

averaged over the test set. Confidence intervals are computed from 5 random seeds.

## I.5 Ablation: Learnable vs Fixed Derivative Orders

Table 6 compares learnable derivative orders against fixed integer orders $( \beta = 1 , 2 )$ . Learning β provides gains on problems with complex spectral structure (Navier-Stokes, Darcy); on the simpler Burgers benchmarks, no benefit over the fixed-{1, 2} baseline is observed.

Table 6: Ablation: learnable vs fixed derivative orders. Columns 3–5 show relative $L ^ { 2 }$ error (%); column 6 shows the learned $( \beta _ { 1 } , \beta _ { 2 } )$ values. Learning β improves over fixed orders on NS and Darcy. The FNO baseline here uses the original Li et al. [1] implementation (rather than the THUML default of Table 1), which accounts for the differing baseline values on NS. The Darcy rows further differ from Table 1 in the unit-Gaussian normalization of the data; absolute values are therefore not comparable across the two tables.
<table><tr><td>Model</td><td>Dataset</td><td>Baseline</td><td>Fixed {1, 2}</td><td>Learned  $\{ \beta _ { 1 } , \beta _ { 2 } \}$ </td><td>Learned Values</td></tr><tr><td>FNO</td><td>Burgers</td><td>0.098</td><td>0.078</td><td>0.079</td><td>1.36, 2.00</td></tr><tr><td>FNO</td><td>Navier-Stokes</td><td>9.64</td><td>9.46</td><td>9.18</td><td>1.05, 2.00</td></tr><tr><td>Transolver</td><td>Burgers</td><td>0.65</td><td>0.63</td><td>0.63</td><td>0.74, 1.83</td></tr><tr><td>Transolver</td><td>Darcy</td><td>1.10</td><td>0.98</td><td>0.95</td><td>0.67, 1.78</td></tr></table>

The lower learned orders $( \beta _ { 1 } )$ consistently fall well below the PDE order $m = 2 ,$ in the sub-physical regime predicted by Theorem $5 . 6 ;$ the higher orders $( \beta _ { 2 } )$ sit at or near the cap $m _ { \mathrm { m a x } } = 2$ (the 2.00 entries are boundary solutions under the constraint $\beta \leq m _ { \mathrm { m a x } } )$ . The largest gains from learning β occur on Navier-Stokes (9.18% vs 9.46%) and Darcy (0.95% vs 0.98%), where the spectral structure is more complex.

## J Verifying the Spectral Decay Assumption

Assumption 5.2 posits that inputs have power spectral density $S ( \xi ) = C _ { S } | \xi | ^ { - 2 s - d }$ . We verify this empirically for each benchmark.

## J.1 Method

For each dataset with training inputs $\{ u _ { i } \} _ { i = 1 } ^ { n } \colon$

1. Compute the empirical power spectrum:

$$
\hat { S } ( \xi ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } | \hat { u } _ { i } ( \xi ) | ^ { 2 }\tag{84}
$$

2. On log-log axes, fit the power law:

$$
\log \hat { S } ( \xi ) = - \alpha \log | \xi | + C\tag{85}
$$

for $| \xi | \in [ \xi _ { \operatorname* { m i n } } , \xi _ { \operatorname* { m a x } } ]$ (avoiding DC and Nyquist artifacts).

3. Extract the implied smoothness:

$$
s _ { \mathrm { e m p i r i c a l } } = { \frac { \alpha - d } { 2 } }\tag{86}
$$

4. Verify the regularity condition: s<sub>empirical</sub> $> m + d / 2$

## J.2 Results

Table 7: Verification of spectral decay assumption across benchmarks.
<table><tr><td>Dataset</td><td>d</td><td>m</td><td>Required s</td><td>Fitted  $\alpha$ </td><td> $s _ { \mathrm { e m p } }$ </td><td>Satisfied?</td></tr><tr><td>Burgers  $\nu = 0 . 1$ </td><td>1</td><td>2</td><td> $> 2 . 5$ </td><td> $\sim 7 . 2$ </td><td> $\sim 3 . 1$ </td><td> $\checkmark$ </td></tr><tr><td>Burgers  $\nu = 0 . 0 0 1$ </td><td>1</td><td>2</td><td> $> 2 . 5$ </td><td> $\sim 5 . 8$ </td><td> $\sim 2 . 4$ </td><td>marginal</td></tr><tr><td>Darcy</td><td>2</td><td>2</td><td> $> 3$ </td><td> $\sim 9 . 5$ </td><td> $\sim 3 . 7 5$ </td><td>√</td></tr><tr><td>Navier-Stokes</td><td>2</td><td>2</td><td> $> 3$ </td><td> $\sim 8 . 2$ </td><td> $\sim 3 . 1$ </td><td> $\checkmark$ </td></tr></table>

Interpretation. Most benchmarks satisfy the regularity assumption comfortably. The low-viscosity Burgers case $( \nu = 0 . 0 0 1 )$ is marginal: with $s _ { \mathrm { e m p } } \approx 2 . 4$ versus the required $s > 2 . 5$ , it violates

Assumption 4.2 by approximately 4%. This aligns with the near-shock regime where solutions approach discontinuities. Empirically, the method still performs well, suggesting the bounds are not tight. In this case, the theory provides qualitative guidance (use lower β) even if quantitative predictions are approximate.

## K Computational Overhead

Table 8 reports wall-clock training times comparing baseline backbones with their ∂-augmented versions. Note that our implementation is not optimized and runs in eager mode (no torch.compile), so these timings reflect upper bounds on the overhead. The overhead varies by backbone: for Transolver (which is dominated by attention computations), the relative overhead is 25–41%, while for FNO the overhead is higher (57–114%) since the baseline is already very fast. In absolute terms, the derivative feature computation adds modest time because: (1) the Grunwald-Letnikov stencil is ¨ local $( K _ { \mathrm { G L } } = 1 6 \mathrm { t e r m s } )$ ; and (2) derivative computation parallelizes trivially across batch and spatial dimensions.

Table 8: Training time per 100 epochs (seconds).
<table><tr><td>Model</td><td>Burgers (ν=0.1)</td><td>Burgers (ν=0.001)</td><td>Darcy</td><td>NS  $( \nu { = } 1 0 ^ { - 5 } )$ </td></tr><tr><td>FNO</td><td>95</td><td>78</td><td>313</td><td>63</td></tr><tr><td>∂-FNO</td><td>149</td><td>124</td><td>664</td><td>135</td></tr><tr><td>Transolver</td><td>412</td><td>382</td><td>1745</td><td>2241</td></tr><tr><td>∂-Transolver</td><td>517</td><td>480</td><td>2201</td><td>3158</td></tr></table>

For larger backbones (e.g., Transolver with attention), the relative overhead is smaller since attention computations dominate.

## L Additional Experiments and Ablations

This appendix collects the extended empirical evidence referenced from §7: a four-way physicsinformed comparison, a matched-parameter ablation isolating the role of the derivative inductive bias, stress tests under Gaussian and uniform noise, resolution sensitivity, directional isotropy on 2D isotropic PDEs, and a 3D compressible Navier-Stokes study.

## L.1 Physics-Informed Comparison: ∂-PINO

We compare four configurations on four PDE benchmarks under clean and noisy inputs: data-only FNO, ∂-FNO, the physics-informed neural operator PINO [31], and ∂-PINO obtained by adding learnable fractional derivative inputs to PINO. PINO uses a PDE-residual loss with $\lambda _ { \mathrm { p d e } } = 1 . 0$ and the same FNO backbone topology used in our main runs; baselines may differ numerically from Table 1 due to paired-training hyperparameters.

Table 9: Four-way physics-informed comparison. Relative $L ^ { 2 }$ error (%). $\Delta _ { \partial }$ is the relative gain from adding ∂-augmentation to each baseline. ∂-PINO improves PINO in 8/8 settings.
<table><tr><td></td><td></td><td colspan="3">DATA-ONLY</td><td colspan="3">PHYSICS-INFORMED  $( \lambda _ { \mathrm { p d e } } > 0 )$ </td></tr><tr><td>Dataset</td><td>σ</td><td>FNO</td><td>∂-FNO</td><td> $\Delta _ { \partial }$ </td><td>PINO</td><td>∂-PINO</td><td> $\Delta _ { \partial }$ </td></tr><tr><td>Burgers  $( \nu { = } 1 0 ^ { - 1 } )$ </td><td>0</td><td>0.07</td><td>0.07</td><td>7.1%</td><td>1.32</td><td>0.69</td><td>47.8%</td></tr><tr><td>Burgers (ν=10−1)</td><td>0.05</td><td>0.41</td><td>0.38</td><td>9.3%</td><td>1.44</td><td>0.67</td><td>53.2%</td></tr><tr><td>Burgers (ν=10−3)</td><td>0</td><td>0.61</td><td>0.56</td><td>6.9%</td><td>0.62</td><td>0.57</td><td>8.9%</td></tr><tr><td>Burgers (ν=10−3)</td><td>0.05</td><td>1.09</td><td>1.02</td><td>6.4%</td><td>1.05</td><td>1.01</td><td>3.6%</td></tr><tr><td>KdV</td><td>0</td><td>2.50</td><td>2.49</td><td>0.1%</td><td>6.69</td><td>6.19</td><td>7.5%</td></tr><tr><td>KdV</td><td>0.05</td><td>3.14</td><td>2.98</td><td>5.3%</td><td>7.48</td><td>6.73</td><td>10.0%</td></tr><tr><td>Darcy</td><td>0</td><td>5.72</td><td>5.48</td><td>4.2%</td><td>6.25</td><td>5.76</td><td>7.9%</td></tr><tr><td>Darcy</td><td>0.05</td><td>5.69</td><td>5.50</td><td>3.5%</td><td>6.02</td><td>5.70</td><td>5.4%</td></tr></table>

The two paradigms are complementary: ∂-augmentation improves PINO in all $8 / 8$ settings, with the largest gains (47.8%, 53.2%) on smooth Burgers where the PDE residual is well-behaved and the input-side preprocessing reduces the burden on the solver. ∂-FNO outperforms PINO across the benchmark despite using no PDE-form information at training time, indicating that learned fractional input features can substitute for, not just augment, loss-side physics regularization on standard benchmarks.

## L.2 Matched-Parameter Ablation

A natural concern is whether ∂-FNO’s gains come from the GL derivative structure or simply from the extra channels added at the input. We address this with a matched-parameter control: four models tuned to the same parameter count (∼292K) on Burgers (ν=0.1), evaluated across seven noise levels and five seeds (35 (model, σ, seed) cells per model). The variants are:

• FNO: vanilla baseline, no input augmentation.

• ∂-FNO: input augmented with two learnable GL fractional derivative channels.

• Random-FNO: same architecture as ∂-FNO, but the GL stencil weights are replaced by fixed random filters of the same length. Tests whether the gains come from any structured local channel.

• Wider-FNO: vanilla FNO with a widened first layer that absorbs the same extra parameters. Tests whether the gains come from added capacity at the input.

Table 10: Matched-parameter ablation on Burgers (ν=0.1). Mean relative $L ^ { 2 }$ error across 5 seeds; all four models at parameter count ≈ 292K. Bold = best per row. ∂-FNO wins 32/35 seeds vs FNO (91.4%), 34/35 vs Random-FNO (97.1%), and 27/35 vs Wider-FNO (77.1%).
<table><tr><td>σ</td><td>FNO</td><td>∂-FNO</td><td>Random-FNO</td><td>Wider-FNO</td></tr><tr><td>0</td><td>0.000597</td><td>0.000560</td><td>0.000638</td><td>0.000579</td></tr><tr><td>0.01</td><td>0.000824</td><td>0.000733</td><td>0.000843</td><td>0.000747</td></tr><tr><td>0.02</td><td>0.001344</td><td>0.001110</td><td>0.001299</td><td>0.001178</td></tr><tr><td>0.05</td><td>0.004210</td><td>0.003357</td><td>0.004037</td><td>0.003847</td></tr><tr><td>0.10</td><td>0.008356</td><td>0.006704</td><td>0.008811</td><td>0.009224</td></tr><tr><td>0.25</td><td>0.018227</td><td>0.013698</td><td>0.018047</td><td>0.018264</td></tr><tr><td>0.50</td><td>0.030202</td><td>0.025173</td><td>0.027412</td><td>0.029035</td></tr></table>

Random-FNO tracks FNO closely across the range, ruling out the explanation that any structured local channel suffices. Wider-FNO is comparable to FNO in the clean regime but degrades faster at moderate noise levels, ruling out the explanation that extra parameters at the input drive the gain. The improvement is specific to the GL derivative structure.

## L.3 Stress Tests under Gaussian and Uniform Noise

We stress-test ∂-augmentation across a wider noise range than the $\sigma { = } 0 . 0 5$ used in the main paper. For each of five PDE benchmarks (Burgers ν=0.1 and $\bar { \nu } { = } 1 0 ^ { - 3 }$ , KdV, Darcy, NS2D), each of two backbones (FNO, LocalNO), each of two noise distributions (Gaussian and uniform), and six noise levels $( \sigma \in \{ 0 . 0 5 , 0 . 1 0 , 0 . 1 5 , 0 . 2 0 , 0 . 2 5 , 0 . 3 0 \} )$ , we measure whether the ∂-augmented variant beats its baseline. This yields $5 \times 2 \times 2 \times 6 = 1 2 0$ settings in total.

∂-augmentation improves every tested setting (120/120 overall). Peak gains reach 48.5% (Gaussian) and 48.1% (uniform) on smooth Burgers at $\sigma { = } 0 . 3 0$ . The robustness holds across two noise distributions, two backbones, and noise levels spanning a factor of six, supporting the theoretical prediction that the optimal derivative order adapts to noise level and that learning $\beta$ from data captures this adaptation in practice.

## L.4 Resolution Sensitivity

We probe how the value of input-side derivative features varies with grid resolution. Coarser grids reduce the Nyquist cutoff $\dot { \xi _ { \operatorname* { m a x } } } = \pi / h$ and tighten the network’s high-frequency headroom, so we expect the marginal benefit of ∂-features to grow. We test this on Burgers (ν=0.1) at resolutions {64, 128, 256, 512, 1024, 2048} under clean (σ=0) and noisy (σ=0.02) inputs.

Table 11: Stress-test: relative $L ^ { 2 }$ error under Gaussian and uniform additive noise (σ from 0.05 to 0.30), two backbones (FNO, LocalNO). Bold = best per (row, noise type, backbone); $\Delta _ { \partial }$ in green. ∂-FNO improves the baseline in 60/60 settings; ∂-LocalNO improves in 60/60 settings; combined 120/120 wins. Peak improvement: 48.5% (Gaussian) and 48.1% (uniform) on smooth Burgers. n=1000 training samples. $\Delta _ { \partial } = ( \mathrm { \mathbf { B L } } - \partial \mathbf { - M o d e l } ) / \mathrm { \mathbf { B L } } \times 1 0 0 $
<table><tr><td rowspan="3" colspan="2"></td><td colspan="6">Gaussian</td><td colspan="6">Uniform</td></tr><tr><td colspan="3">FNO</td><td colspan="3">LocalNO</td><td colspan="3">FNO</td><td colspan="3">LocalNO</td></tr><tr><td>BL</td><td>∂</td><td> $\Delta _ { \partial }$ </td><td>BL</td><td>∂</td><td> $\Delta _ { \partial }$ </td><td>BL</td><td>∂</td><td> $\Delta _ { \partial }$ </td><td>BL</td><td>∂</td><td> $\Delta _ { \partial }$ </td></tr><tr><td></td><td>0.05</td><td>0.00259</td><td>0.00210</td><td>19.1%</td><td>0.00261</td><td>0.00215</td><td>17.7%</td><td>0.00138</td><td>0.00111</td><td>19.9%</td><td>0.00137</td><td>0.00113</td><td>17.4%</td></tr><tr><td>01</td><td>0.10</td><td>0.00621</td><td>0.00459</td><td>26.0%</td><td>0.00726</td><td>0.00518</td><td>28.7%</td><td>0.00281</td><td>0.00206</td><td>26.9%</td><td>0.00568</td><td>0.00348</td><td>38.8%</td></tr><tr><td></td><td>0.15</td><td>0.00944</td><td>0.00711</td><td>24.7%</td><td>0.01264</td><td>0.00995</td><td>21.3%</td><td>0.00530</td><td>0.00392</td><td>26.0%</td><td>0.00908</td><td>0.00539</td><td>40.6%</td></tr><tr><td></td><td>0.20</td><td>0.01270</td><td>0.01050</td><td>17.4%</td><td>0.01925</td><td>0.01584</td><td>17.7%</td><td>0.00770</td><td>0.00598</td><td>22.3%</td><td>0.01075</td><td>0.00757</td><td>29.6%</td></tr><tr><td>Burrg</td><td>0.25</td><td>0.01782</td><td>0.01236</td><td>30.6%</td><td>0.02295</td><td>0.01646</td><td>28.3%</td><td>0.00943</td><td>0.00905</td><td>4.0%</td><td>0.01540</td><td>0.00939</td><td>39.0%</td></tr><tr><td></td><td>0.30</td><td>0.02212</td><td>0.01493</td><td>32.5%</td><td>0.03055</td><td>0.01573</td><td>48.5%</td><td>0.01385</td><td>0.01188</td><td>14.2%</td><td>0.02409</td><td>0.01250</td><td>48.1%</td></tr><tr><td></td><td>0.05</td><td>0.02593</td><td>0.02448</td><td>5.6%</td><td>0.01697</td><td>0.01678</td><td>1.1%</td><td>0.01519</td><td>0.01294</td><td>14.8%</td><td>0.01034</td><td>0.01001</td><td>3.2%</td></tr><tr><td></td><td>0.10</td><td>0.03223</td><td>0.03150</td><td>2.2%</td><td>0.02809</td><td>0.02570</td><td>8.5%</td><td>0.01963</td><td>0.01757</td><td>10.5%</td><td>0.01859</td><td>0.01504</td><td>19.1%</td></tr><tr><td></td><td>0.15</td><td>0.03276</td><td>0.03024</td><td>7.7%</td><td>0.04088</td><td>0.03336</td><td>18.4%</td><td>0.02407</td><td>0.02221</td><td>7.7%</td><td>0.02646</td><td>0.02511</td><td>5.1%</td></tr><tr><td></td><td>0.20</td><td>0.03897</td><td>0.03630</td><td>6.9%</td><td>0.05948</td><td>0.04410</td><td>25.9%</td><td>0.02827</td><td>0.02683</td><td>5.1%</td><td>0.03645</td><td>0.03363</td><td>7.7%</td></tr><tr><td>BBu-g= 1)</td><td>0.25</td><td>0.05195</td><td>0.05026</td><td>3.3%</td><td>0.06839</td><td>0.05614</td><td>17.9%</td><td>0.03248</td><td>0.03043</td><td>6.3%</td><td>0.04087</td><td>0.03334</td><td>18.4%</td></tr><tr><td></td><td>0.30</td><td>0.04982</td><td>0.04644</td><td>6.8%</td><td>0.07652</td><td>0.05710</td><td>25.4%</td><td>0.03662</td><td>0.03444</td><td>5.9%</td><td>0.05561</td><td>0.04098</td><td>26.3%</td></tr><tr><td>ΛP</td><td>0.05</td><td>0.02640</td><td>0.02567</td><td>2.8%</td><td>0.04840</td><td>0.04597</td><td>5.0%</td><td>0.01997</td><td>0.01974</td><td>1.2%</td><td>0.03434</td><td>0.03238</td><td>5.7%</td></tr><tr><td></td><td>0.10</td><td>0.03575</td><td>0.03468</td><td>3.0%</td><td>0.05649</td><td>0.05474</td><td>3.1%</td><td>0.02447</td><td>0.02406</td><td>1.7%</td><td>0.03874</td><td>0.03672</td><td>5.2%</td></tr><tr><td></td><td>0.15</td><td>0.04332</td><td>0.04230</td><td>2.3%</td><td>0.05716</td><td>0.05524</td><td>3.4%</td><td>0.03064</td><td>0.03038</td><td>0.8%</td><td>0.04542</td><td>0.04184</td><td>7.9%</td></tr><tr><td></td><td>0.20</td><td>0.05590</td><td>0.05205</td><td>6.9%</td><td>0.07070</td><td>0.07012</td><td>0.8%</td><td>0.03722</td><td>0.03587</td><td>3.6%</td><td>0.05419</td><td>0.04733</td><td>12.7%</td></tr><tr><td></td><td>0.25</td><td>0.06838</td><td>0.06455</td><td>5.6%</td><td>0.08954</td><td>0.08492</td><td>5.2%</td><td>0.04369</td><td>0.04222</td><td>3.4%</td><td>0.06054</td><td>0.05383</td><td>11.1%</td></tr><tr><td></td><td>0.30</td><td>0.08012</td><td>0.06979</td><td>12.9%</td><td>0.09959</td><td>0.08592</td><td>13.7%</td><td>0.05126</td><td>0.04844</td><td>5.5%</td><td>0.07272</td><td>0.05854</td><td>19.5%</td></tr><tr><td></td><td>0.05</td><td>0.05660</td><td>0.05430</td><td>4.1%</td><td>0.05581</td><td>0.05529</td><td>0.9%</td><td>0.05540</td><td>0.05456</td><td>1.5%</td><td>0.05567</td><td>0.05306</td><td>4.7%</td></tr><tr><td>Darc</td><td>0.10</td><td>0.05644</td><td>0.05431</td><td>3.8%</td><td>0.05583</td><td>0.05483</td><td>1.8%</td><td>0.05643</td><td>0.05428</td><td>3.8%</td><td>0.05560</td><td>0.05371</td><td>3.4%</td></tr><tr><td></td><td>0.15</td><td>0.05636</td><td>0.05476</td><td>2.8%</td><td>0.05597</td><td>0.05483</td><td>2.0%</td><td>0.05646</td><td>0.05435</td><td>3.8%</td><td>0.05556</td><td>0.05371</td><td>3.3%</td></tr><tr><td></td><td>0.20</td><td>0.05646</td><td>0.05515</td><td>2.3%</td><td>0.05625</td><td>0.05587</td><td>0.7%</td><td>0.05666</td><td>0.05439</td><td>4.0%</td><td>0.05574</td><td>0.05374</td><td>3.6%</td></tr><tr><td></td><td>0.25</td><td>0.05682</td><td>0.05582</td><td>1.7%</td><td>0.05730</td><td>0.05652</td><td>1.4%</td><td>0.05614</td><td>0.05438</td><td>3.1%</td><td>0.05590</td><td>0.05409</td><td>3.2%</td></tr><tr><td></td><td>0.30</td><td>0.05744</td><td>0.05618</td><td>2.2%</td><td>0.05744</td><td>0.05629</td><td>2.0%</td><td>0.05667</td><td>0.05551</td><td>2.0%</td><td>0.05549</td><td>0.05441</td><td>1.9%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>5</td><td>0.05 0.10</td><td>0.03718</td><td>0.03631</td><td>2.3%</td><td>0.02667</td><td>0.02472</td><td>7.3%</td><td>0.03566</td><td>0.03513</td><td>1.5%</td><td>0.02438</td><td>0.02279</td><td>6.5%</td></tr><tr><td></td><td>0.15</td><td>0.03861 0.04284</td><td>0.03749</td><td>2.9%</td><td>0.03147</td><td>0.02919</td><td>7.2%</td><td>0.03659</td><td>0.03645</td><td>0.4%</td><td>0.02745</td><td>0.02547</td><td>7.2%</td></tr><tr><td>00</td><td></td><td>0.04435</td><td>0.04115 0.04353</td><td>4.0% 1.9%</td><td>0.03548 0.03879</td><td>0.03309 0.03639</td><td>6.7% 6.2%</td><td>0.03856 0.03925</td><td>0.03807 0.03890</td><td>1.3% 0.9%</td><td>0.02994 0.03286</td><td>0.02810 0.03046</td><td>6.2% 7.3%</td></tr><tr><td>SN</td><td>0.20 0.25</td></table>

Resolution sensitivity: FNO vs -FNO (Burgers =0.1)  
![](images/165cc0e528aaeacde2767f529c04ab6af4b77f5bd289dd7b5757c745eb929800.jpg)  
Figure 4: Resolution sensitivity: FNO vs. ∂-FNO on Burgers $( \nu { = } 0 . 1 )$ across grid resolutions 64 to 2048. Left: clean inputs. Right: noisy inputs $\scriptstyle ( \sigma = 0 . 0 2 )$ . Shaded area indicates the ∂-FNO improvement. Clean improvements are resolution-stable (6–8%); noisy improvements grow at coarser resolutions, from 14% at 2048 up to 39% at 64: input-side derivative features are most valuable when the network’s Nyquist headroom is tightest.

A practical implication is that downsampling, the most common strategy when GPU memory is limited, silently makes exact physics enforcement increasingly suboptimal. ∂-NO compensates automatically by learning lower orders at coarser resolutions, while fixed-order approaches cannot.

![](images/358c8c71956af32d07c219c6de54cb44cfff3f71c5a708ea300419924179f793.jpg)  
Figure 5: Learned $\beta$ clusters across resolutions (Burgers, $K { = } 8$ learnable channels). Two clusters consistently emerge near $m { = } 2$ (diffusion, top) and $m { = } 1$ (convection, bottom). The diffusion cluster is not resolution-invariant: it drifts below 2 at coarser resolutions, an empirical effect of finite stencil fidelity at large grid spacing rather than a prediction of Theorem $5 . 6 ,$ whose $\xi _ { \mathrm { m a x } }$ -dependence would, with all else fixed, imply the opposite drift. The sweep varies the data spectrum and stencil fidelity alongside $\xi _ { \mathrm { m a x } }$ and does not isolate this effect. Downsampling, the most common strategy when GPU memory is limited, silently makes exact physics enforcement increasingly suboptimal. ∂-NO compensates automatically by learning lower orders.

## L.5 Directional Isotropy on 2D Isotropic PDEs

On PDEs whose differential operator is isotropic in space (no preferred axis), we expect $\beta _ { x } \approx \beta _ { y }$ to emerge if ∂-NO captures the true differential structure rather than incidental data features. We test this on Darcy (isotropic elliptic) and 2D Navier-Stokes $( \nu { = } 1 0 ^ { - 5 } )$ with K=4 learnable channels.

![](images/51ddbf6b7527d41fc5e0b062909e3845d3f7737f93ed8ba21a2fb0ce09f1ea0f.jpg)  
Figure 6: Directional isotropy of learned orders on two 2D isotropic PDEs. Left: Darcy. Right: Navier-Stokes $( \nu { = } 1 0 ^ { - 5 } )$ . Across the $K { = } 4$ learnable channels, $\beta _ { x }$ (blue) and $\beta _ { y }$ (orange) track each other closely, with channel means $\bar { \beta } _ { x } = 1 . 0 4 , \bar { \beta } _ { y } = 1 . 0 7$ on Darcy $\left( \Delta { = } 0 . 0 3 2 \right)$ and $\bar { \beta } _ { x } = 1 . 0 2 , \bar { \beta } _ { y } = 1 . 0 2$ on NS $\scriptstyle ( \Delta = 0 . 0 0 2 )$ The four channels self-organize into a hierarchy spanning [0.15, 2.0], each capturing a different spectral scale. Isotropy is recovered without being prescribed. (NS channel 4 sits at the cap $m _ { \mathrm { m a x } } { = } 2 . )$

Across both benchmarks, the channel-averaged $\beta _ { x }$ and $\beta _ { y }$ agree within $\Delta < 0 . 0 3 5$ of each other without isotropy being prescribed. The channels self-organize into a hierarchy of spectral scales rather than collapsing to identical orders, indicating that the network uses the $K { = } 4$ budget productively to span low through high frequencies.

## L.6 Three-Dimensional Compressible Navier-Stokes

We test $\partial \mathrm { - }$ -augmentation on the 3D compressible Navier-Stokes equations from the PDEBench benchmark suite [45] $( 6 4 \times 6 4 \times 6 4$ spatial grid, T=10, five-variable input $V _ { x } , V _ { y } , V _ { z }$ , density, pressure), with $K _ { x } = K _ { y } = K _ { z } = 2$ axis-aligned channels and $K _ { x y } = K _ { x z } = K _ { y z } = K _ { x y z } = 1$ cross- and triple-derivative channels.

![](images/29a01db902c8295a38f98f8de181c6ace8aa311548673c3c07ecf9592ca92123.jpg)  
Figure 7: Learned fractional orders on 3D compressible Navier-Stokes, for three backbones (FNO3D, TFNO, Transolver) under clean $( \sigma { = } 0 ,$ , left) and noisy $( \sigma { = } 0 . 0 5 ,$ , right) conditions. Each axis $( x , y , z )$ retains a channel with order near 2 (Laplacian, the dominant differential operator), while crossderivative $( \beta _ { x y } , \beta _ { x z } , \beta _ { y z } )$ and triple-derivative $( \beta _ { x y z } )$ channels collapse to near-identity $( \beta \approx 0 )$ . The network selectively amplifies the physically dominant differential structure without it being prescribed.

Each axis retains a channel with order close to the Laplacian order $m { = } 2$ , while cross- and triplederivative channels saturate near identity. The directional structure is consistent across three backbones and persists under noise, supporting the claim that learned $\beta$ values reflect physically meaning ful differential structure rather than backbone-specific quirks.

## L.7 Three-Dimensional Compressible Navier-Stokes: Quantitative Results

Complementing the learned-orders breakdown in §L.6, we report the quantitative test error on the 3D compressible Navier-Stokes benchmark $( 6 4 \times 6 4 \times 6 4 , { \overline { { T } } } { = } 1 0$ , five-variable input). Three backbones (FNO3D, TFNO, Transolver) are evaluated under clean and noisy inputs; ∂-augmentation uses $K _ { x } = K _ { y } = K _ { z } = 2$ axis channels and $K _ { x y } = K _ { x z } = K _ { y z } = K _ { x y z } = 1$ cross- and triple-derivative channels.

Table 12: 3D Compressible Navier-Stokes: test relative $L ^ { 2 }$ error. ∂-augmentation improves every (backbone, noise) cell; gains widen under noise on Transolver.
<table><tr><td></td><td colspan="2">FNO3D</td><td colspan="2">TFNO</td><td colspan="2">Transolver</td></tr><tr><td></td><td>σ=0</td><td>σ=0.05</td><td> $\sigma { = } 0$ </td><td> $\sigma { = } 0 . 0 5$ </td><td> $\sigma { = } 0$ </td><td>σ=0.05</td></tr><tr><td>Model</td><td>0.2101</td><td>0.2168</td><td>0.2079</td><td>0.2225</td><td>0.1990</td><td>0.2039</td></tr><tr><td>∂-Model</td><td>0.1964</td><td>0.2115</td><td>0.1971</td><td>0.2033</td><td>0.1699</td><td>0.1695</td></tr><tr><td>Improv.</td><td>6.5%</td><td>2.4%</td><td>5.2%</td><td>8.6%</td><td>14.6%</td><td>16.9%</td></tr></table>

Transolver gains are largest both clean and noisy, consistent with its larger effective capacity and the bias-variance prediction that higher-capacity models benefit more from a regularization-aware input feature.

## L.8 Conv-FNO Standalone Comparison

We compare ∂-FNO against Conv-FNO [34] as direct alternatives: both preprocess the input, ∂-FNO with a GL derivative stencil and Conv-FNO with learnable convolution kernels. Both build on the same FNO backbone, allowing a clean head-to-head test of preprocessing strategies.

Parameter counts. On Burgers, ∂-FNO adds 4 learnable preprocessing parameters (two orders $\beta _ { k }$ two scales $s _ { k } ,$ , since the stencil taps are functions of $\beta _ { k }$ rather than free parameters); Conv-FNO adds ≈ 136 (16-wide kernels per channel). On Darcy/NS, ∂-FNO adds 11 (six orders, five scales) vs $\mathrm { C o n v  – F N O ^ { \ast } s \approx 2 0 5 6 }$ . The constrained inductive bias matches or beats the learned kernel at 34–187× lower parameter overhead.

Table 13: Conv-FNO vs. ∂-FNO on three benchmarks. Relative $L ^ { 2 }$ error (%); ∆ is the relative improvement over the FNO baseline. ∂-FNO matches or outperforms Conv-FNO with $\mathbf { 3 4 - 1 8 7 } \times$ fewer preprocessing parameters (see note below).
<table><tr><td></td><td></td><td></td><td></td><td></td><td colspan="2">Improv. over FNO</td></tr><tr><td>Dataset</td><td>σ</td><td>FNO</td><td>∂-FNO</td><td>Conv-FNO</td><td>∂-FNO</td><td>Conv-FNO</td></tr><tr><td>Burgers (ν=0.1)</td><td>0</td><td>0.0832</td><td>0.0738</td><td>0.0791</td><td>11.3%</td><td>4.9%</td></tr><tr><td>Burgers (ν=0.1)</td><td>0.02</td><td>0.1161</td><td>0.0899</td><td>0.1052</td><td>22.6%</td><td>9.4%</td></tr><tr><td>Darcy</td><td>0</td><td>5.736</td><td>5.474</td><td>5.687</td><td>4.6%</td><td>0.9%</td></tr><tr><td>Darcy</td><td>0.02</td><td>5.721</td><td>5.487</td><td>5.678</td><td>4.1%</td><td>0.8%</td></tr><tr><td>NS (ν=10−5)</td><td>0</td><td>3.506</td><td>3.456</td><td>4.219</td><td>1.5%</td><td>-20.3%</td></tr><tr><td> $\mathrm { N S } \stackrel { \cdot } { ( } \nu = 1 0 ^ { - 5 } )$ </td><td>0.02</td><td>3.614</td><td>3.575</td><td>4.208</td><td>1.1%</td><td>-16.4%</td></tr></table>

## L.9 Ablations on K and $K _ { \mathrm { G L } }$

Two architectural hyperparameters control the cost-quality tradeoff of ∂-augmentation: the number of learnable derivative channels K and the length of the Grunwald-Letnikov stencil¨ $K _ { \mathrm { G L } }$ . We ablate both on Burgers $( \nu { = } 0 . 1 )$ under clean $( \sigma { = } 0 )$ and noisy $_ { ( \sigma = 0 . 0 2 ) }$ conditions.

![](images/fc4725d3eb9cfe41bc58fa3914a7b84b509a18e5400b284551d66c3c6e994b21.jpg)  
Figure 8: Ablation on the number of derivative channels $K \in \{ 1 , 2 , 4 , 8 , 1 6 , 3 2 \}$ . Increasing K consistently improves performance with diminishing returns past $K { = } 8 .$ . The natural tradeoff is memory and compute cost. Under noise the curve flattens more slowly, indicating that additional channels help more when the regularization burden is larger.

![](images/ead4ab49cf0054e43b29ad6c74fa5c1e9028fb57b0c7ea0e52ec0190c69a1483.jpg)  
Figure 9: Sensitivity to GL stencil length $K _ { \mathrm { G L } } \in \{ 2 , 4 , 8 , 1 6 , 3 2 , 6 4 \}$ . Performance stabilizes around $K _ { \mathrm { G L } } { = } 1 6$ under both clean and noisy conditions. At $K _ { \mathrm { G L } } { = } 2$ , accuracy degrades because the stencil is too short to resolve non-integer orders. We use $K _ { \mathrm { G L } } { = } 1 6$ throughout the main paper as a balance between accuracy and compute.

Both ablations support the default choices used in the main paper $( K _ { \mathrm { G L } } { = } 1 6$ universally). Increasing K further would improve accuracy at proportional memory cost; reducing $K _ { \mathrm { G L } }$ below 16 would save compute at the cost of resolution-dependent degradation of fractional orders.

## L.10 Trajectory of Learned $\beta$ During Training

Theorem 5.6 characterizes the population-optimal $\beta ^ { * }$ , but in practice we recover $\beta$ via stochastic gradient descent. Figure 10 traces all $K { = } 8$ learnable orders across training epochs on Burgers $( \nu { = } 0 . 1 )$ , under five noise levels $\sigma \in \{ 0 , 0 . 0 1 , 0 . 0 2 , 0 . 0 5 , 0 . 1 \}$

![](images/6625812c2ffd566f3bf03c88ebd5660e93343ee7a7814cfc39c5e0c1c9f275b1.jpg)  
Figure 10: Trajectory of learned $\beta$ values during training (5 noise levels). All $K { = } 8$ orders stabilize within 200–400 epochs and converge to values consistent with Theorem $5 . 6 \colon$ higher $\beta ^ { * }$ under clean data, systematically lower $\beta ^ { * }$ as $\sigma$ grows. Initial values are $\beta _ { 1 } { = } 1 . 0$ and $\beta _ { 2 } { = } 2 . 0 ;$ the optimizer redistributes channels across the spectral hierarchy as training proceeds.

Three takeaways. First, all eight orders stabilize within 200–400 epochs, indicating optimization is reliable. Second, the final distribution is noise-dependent: at $\sigma { = } 0$ orders cluster near $\bar { \boldsymbol { \beta } } \in [ 0 . 5 , 2 . 0 ]$ while at $\sigma { = } 0 .$ 1 the cluster shifts toward [0.3, 1.8], exactly the direction Theorem 5.6 predicts. Third, $\beta$ values redistribute across the channel budget rather than collapsing to redundant orders, validating the design choice of multiple learnable channels per direction.

## L.11 Interaction Between Noise Level and Channel Count

The K-ablation (Fig. 8) and the stress test (Table 11) study the effects of channel count $K$ and noise level σ separately. Figure 11 shows their interaction: the relative improvement of ∂-FNO over FNO as a heatmap over the joint $( \sigma , K )$ plane on Burgers (ν=0.1). The super-additivity has

![](images/a900a25698abae04e65620b9574cc77b43729cdad3299c218198e1e2c1495380.jpg)  
Figure 11: Interaction between noise level σ and number of derivative channels $K .$ . Cell values are the relative improvement of ∂-FNO over FNO (% reduction in relative $L ^ { 2 } )$ on Burgers $( \nu { = } 0 . 1 )$ Gains broadly increase with both σ and K and concentrate in the high-noise, multi-channel corner, though the pattern is not monotone in either variable at these seed counts. The interaction is superadditive: the largest single gain (+36.4%) sits at $( \sigma { = } 0 . 1 , K { = } 8 )$ , in the high-noise regime near the diminishing-returns plateau of Figure 8, supporting the design choice of multiple channels precisely when data is noisy.

a clean theoretical reading: each channel $\beta _ { k }$ captures a different point on the bias-variance curve from Theorem $5 . 6 ,$ so when noise broadens the optimal $\beta$ distribution the diverse-channel budget pays off more than the sum of its parts. The learnable spectral parameterization may extend to other frequency-selective domains such as signal detection under adverse SNR conditions [46, 47].