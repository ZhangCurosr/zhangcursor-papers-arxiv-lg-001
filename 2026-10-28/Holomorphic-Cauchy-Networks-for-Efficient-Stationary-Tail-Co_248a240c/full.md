# Holomorphic Cauchy Networks for Efficient Stationary Tail Computation in High-Dimensional Reflected Brownian Motions

Kehan Wang

School of Data Science, The Chinese University of Hong Kong, Shenzhen, China

## Abstract

Stationary tail probabilities quantify severe congestion and workload risks in stochastic networks, yet their computation for high-dimensional reflected Brownian motions (RBMs) remains challenging. Building on the shared-coordinate, basic-adjoint-relationship (BAR) transform-learning framework of NN4MGF, we develop Holomorphic Cauchy Networks (HCNs) to move holomorphicity from the loss function into the network architecture. Rather than coupling real and imaginary output heads through a sampled Cauchy–Riemann (CR) penalty, HCN embeds rational Cauchy activations into native complexvalued subnetworks that are holomorphic by construction on pole-free domains, avoiding the inputderivative graphs required by the CR loss. Exact origin anchors and boundary-coordinate exclusion enforce normalization and boundary-independence identities algebraically, anchored interaction subnetworks extend the coordinate-wise representation to correlated systems, and closed-form real-axis derivatives with sequential branch backpropagation further control training memory. Under holomorphic-log assumptions, we establish simultaneous value and first-derivative approximation for separable targets with the sharedcoordinate representation and for general constrained holomorphic targets with the interaction extension, while deriving a mixed-derivative lower bound showing why coordinate-only representations incur an irreducible error under nonzero cross-covariance. In a 30-dimensional profile on the same NVIDIA H100 NVL, the tested HCN configuration achieves a 9.7-fold reduction in mean update time and approximately tenfold lower peak allocated memory than NN4MGF; extrapolating the measured update times to the respective scheduled update counts gives a projected full-schedule update-time ratio of 58.2. Using 64 million sampled base inputs per high-dimensional run, all HCN experiments can be completed on a single GPU and scale to 50 dimensions on a single laptop GPU with 6.367% mean relative error over thresholds 1 through 8. Across degree-matched comparisons over the primary evaluation ranges, HCN attains lower mean relative errors than NN4MGF, reducing error from 6.684% to 1.422% on the two-dimensional Harrison non-product benchmark. An exploratory ten-dimensional correlated RBM example further illustrates the interaction extension beyond two dimensions. Code is available at https://github.com/KehanWang126/holomorphic-cauchy-networks-rbm.

Keywords: reflected Brownian motion; stationary distributions; Laplace-transform inversion; holomorphic neural networks; scientific machine learning.

## 1 Introduction

Reflected Brownian motions (RBMs) serve as diffusion models of congestion and workload dynamics in stochastic networks. Their stationary tail probabilities quantify the risk of large workloads and therefore add important risk information beyond mean-based performance measures. Explicit stationary laws are available only for special model classes, including product-form systems characterized by structural skew-symmetry conditions on the model parameters (Harrison and Williams, 1987; Dai et al., 2014). For more general models, numerical approaches include methods based on the basic adjoint relationship (BAR) (Dai and Harrison, 1992) and high-dimensional steady-state simulation (Blanchet et al., 2021). Neural computation has also proved useful for high-dimensional RBM drift control (Ata et al., 2025), for learning convergence rates and stationary distributions of Markov chains (Qu et al., 2024, 2026), and more broadly in equation-residual scientific machine learning (Raissi et al., 2019). As dimension increases, obtaining accurate stationary tail estimates at a manageable computational cost remains a central challenge.

A recent approach by Dai and Zhang (2026), NN4MGF, learns stationary and boundary Laplace transforms from the BAR and recovers tail probabilities through numerical Laplace inversion. This formulation allows neural networks to be trained directly from the RBM parameters $( \mu , \Sigma , R )$ without requiring samples from the stationary distribution. As noted by Dai and Zhang (2026), however, GPU memory demand and training time remain key obstacles to further scaling: processing large training batches in a single update requires substantial GPU memory, whereas splitting them across smaller batches increases runtime. Our aim is to address these structural and computational bottlenecks—substantially reducing GPU memory demand, update time, and sampling requirements while maintaining or improving tail accuracy—so that BAR-based transform learning scales to higher dimensions on modest hardware and extends to correlated non-product systems.

The complex-analytic structure of Laplace transforms offers a natural way to relieve this computational bottleneck. Numerical Laplace inversion evaluates transforms at complex arguments, where the true stationary and boundary transforms are holomorphic throughout the interior of their convergence domains. Within standard real-valued deep learning frameworks, Dai and Zhang (2026) map the real and imaginary parts of each coordinate through Fourier features into two-headed real-valued networks $( g ^ { \mathrm { R e } } , g ^ { \mathrm { I m } } )$ to approximate the real and imaginary parts of the log transform. Because two real-valued outputs do not automatically form a complex-differentiable pair, NN4MGF couples them during training via a sampled Cauchy–Riemann (CR) penalty. Treating holomorphicity as a training-time differential constraint is effective, but it requires computing first-order input derivatives and retaining their computational graphs in GPU memory during parameter backpropagation, adding memory overhead as dimension increases.

A natural alternative is to move holomorphicity from the loss function into the network architecture itself. Recently, Li et al. (2025) introduced a rational Cauchy activation function motivated by Cauchy’s integral formula for real-function approximation, and Si et al. (2026) demonstrated its effectiveness in physics-informed neural networks for real-valued PDEs. Although both works operate on real-valued inputs, the rational form itself is naturally suited to the complex domain: when evaluated directly on complex affine transformations in C, the Cauchy activation $\sigma _ { r } ( z ) = ( a _ { r } z + b _ { r } ) / ( z ^ { 2 } + s _ { r } ^ { 2 } )$ is complex-differentiable wherever its denominator is nonzero. Building on this observation, we construct native complex-valued Cauchy subnetworks and embed them into NN4MGF’s shared-coordinate BAR framework to form Holomorphic Cauchy Networks (HCNs). Because HCN satisfies the Cauchy–Riemann equations identically by construction on pole-free domains, this design replaces separate real and imaginary output heads and Fourier feature encodings while removing the sampled CR penalty, thereby avoiding the input-derivative graphs required by the CR loss and preserving the strengths of the BAR-learning formulation.

Building on NN4MGF’s parameter-sharing mechanism across coordinates, HCN incorporates several structural and computational refinements for the interior and boundary transforms. Additive coordinate blocks provide an economical representation of separable log transforms, while anchored interaction subnetworks extend the architecture to dependent systems. Exact origin anchoring enforces the known interior normalization and boundary masses, and boundary-coordinate exclusion ensures that each boundary representation is independent of the coordinate normal to its supporting face. These identities hold algebraically by construction rather than through additional soft penalty terms, and evaluating isolated boundary residuals directly in transform space avoids taking complex logarithms of BAR coefficients. To further control memory usage during training, we employ closed-form input derivatives for the real-axis constraints, sequential backpropagation across the BAR, boundary, and real-axis loss branches, and dimension-dependent batch schedules.

We establish approximation properties for the resulting constrained architecture (Theorem 1). Under holomorphic-log assumptions, the shared-coordinate representation can simultaneously approximate function values and first derivatives of separable targets on compact polydiscs, while the anchored interaction extension approximates general holomorphic targets satisfying the prescribed origin and boundary-coordinate constraints. The construction preserves parameter sharing and the exact structural identities with a uniform denominator margin. Conversely, we derive a mixed-derivative lower bound showing that whenever stationary coordinates have nonzero cross-covariance, coordinate-only representations incur an irreducible uniform approximation error on the joint log transform that increasing width alone cannot remove.

Our numerical experiments evaluate the proposed framework across computational efficiency, hardware accessibility, and tail accuracy. In a 30-dimensional profile on the same NVIDIA H100 NVL, the tested HCN configuration achieves a 9.7-fold reduction in mean update time and approximately tenfold lower peak allocated memory compared with NN4MGF; extrapolating the measured update times to the respective scheduled update counts gives a projected full-schedule update-time ratio of 58.2. These ratios compare the complete computational configurations—including their architectures, batch sizes, derivative calculations, and graph schedules—rather than isolating the effect of CR penalty removal. Using 64 million sampled base inputs in each high-dimensional run, all HCN experiments can be completed on a single GPU and scale to 50 dimensions on a single laptop GPU, achieving 6.367% mean relative error over thresholds 1 through 8 at 50 dimensions with comparable accuracy across final-stage checkpoints. Across the degreematched comparisons, HCN attains lower reported mean relative errors than NN4MGF over the primary evaluation ranges, reducing error from 6.684% to 1.422% on the two-dimensional Harrison non-product benchmark, while an exploratory ten-dimensional correlated example illustrates the interaction extension in higher dimensions. Exact-transform inversion diagnostics and a complex-tanh control further motivate Degree 8 Fixed-Talbot inversion and the rational Cauchy activation.

Section 2 introduces the RBM transform problem, Section 3 develops HCN and its approximation theorem, and Section 4 details the memory-controlled training computations. Section 5 reports the numerical experiments, and Sections 6–7 discuss scope and conclusions. Proofs, implementation details, and additional diagnostics are provided in the appendices, and code is available at https://github.com/ KehanWang126/holomorphic-cauchy-networks-rbm.

## 2 RBM Stationary Tails and Transform Learning

## 2.1 The stationary tail problem

A d-dimensional RBM in $\mathbb { R } _ { + } ^ { d }$ satisfies

$$
Z ( t ) = Z ( 0 ) + \mu t + B ( t ) + R Y ( t ) , \qquad t \geq 0 ,\tag{1}
$$

where $\boldsymbol { \mu } \in \mathbb { R } ^ { d }$ is the drift vector, B is a d-dimensional zero-mean Brownian motion with covariance matrix $t \Sigma .$ , and the columns $R ^ { ( k ) }$ of the $d \times d$ matrix R specify the reflection directions on the boundary faces of the orthant. Each continuous regulator process $Y _ { k }$ starts at zero, is nondecreasing, and increases only when the state hits the kth boundary face $Z _ { k } = 0 .$ , satisfying $\begin{array} { r } { \int _ { 0 } ^ { T } Z _ { k } ( u ) d Y _ { k } ( u ) = 0 } \end{array}$ for all $T \geq 0$

We assume that Σ is symmetric positive definite, R is nonsingular with componentwise positive vector $R ^ { - 1 } \mu$ , and the RBM is well defined with a unique stationary distribution π (Harrison and Williams, 1987). Under stationary initialization $Z ( 0 ) \sim \pi$ , the state vector has finite first moments and the regulator process has finite expected increments over unit time intervals. Our primary performance metric is the stationary tail probability of the total system workload:

$$
S = \mathbf { 1 } ^ { \mathsf { T } } Z ( 0 ) , \qquad p ( t ) = \mathbb { P } _ { \pi } ( S > t ) , \qquad t \geq 0 .\tag{2}
$$

Our objective is to compute $p ( t )$ directly from the system primitives $( \mu , \Sigma , R )$ without generating sample paths from π.

## 2.2 Transforms and the basic adjoint relationship

For a complex vector $\ b \theta \in \mathbb { C } ^ { d }$ , define the interior and boundary Laplace transforms as

$$
\begin{array} { r } { \phi _ { 0 } \mathopen { } \mathclose \bgroup \left( \theta \aftergroup \egroup \right) = \mathbb { E } _ { \pi } \mathopen { } \mathclose \bgroup \left[ e ^ { - \theta ^ { \top } Z ( 0 ) } \aftergroup \egroup \right] , } \end{array}\tag{3}
$$

$$
\phi _ { k } ( \theta ) = \mathbb { E } _ { \boldsymbol { \pi } } \left[ \int _ { 0 } ^ { 1 } e ^ { - \theta ^ { \intercal } Z ( u ) } d Y _ { k } ( u ) \right] , \qquad k = 1 , \ldots , d ,\tag{4}
$$

where complex inner products use ordinary transposes without conjugation. These transforms are finite whenever ℜθ is componentwise nonnegative and are holomorphic throughout the interior of their common domain of absolute convergence. Each $\phi _ { k }$ represents the Laplace transform of a finite boundary measure supported on the face $\{ z \in \mathbb { R } _ { + } ^ { d } : z _ { k } = 0 \}$ , which renders $\phi _ { k } ( \theta )$ structurally independent of the kth coordinate $\theta _ { k }$

Stationarity of (1) directly determines the transform values at the origin:

$$
\begin{array} { r } { m = \mathbb { E } _ { \boldsymbol \pi } [ Y ( 1 ) ] = - R ^ { - 1 } \mu , \qquad \phi _ { 0 } ( 0 ) = 1 , \qquad \phi _ { k } ( 0 ) = m _ { k } , \qquad k = 1 , \ldots , d . } \end{array}\tag{5}
$$

Applying $\mathrm { I t } \hat { \mathrm { o } } ^ { \cdot } \mathrm { s }$ formula to the test function $e ^ { - \theta ^ { \mathsf { T } } z }$ and taking expectations under π yields the Laplace transform version of the basic adjoint relationship (BAR):

$$
\gamma _ { 0 } ( \theta ) \phi _ { 0 } ( \theta ) = \sum _ { k = 1 } ^ { d } \gamma _ { k } ( \theta ) \phi _ { k } ( \theta ) ,\tag{6}
$$

with known quadratic and linear polynomial coefficients

$$
\begin{array} { r } { \gamma _ { 0 } ( \theta ) = - \frac 1 2 \theta ^ { \mathsf { T } } \Sigma \theta + \mu ^ { \mathsf { T } } \theta , \qquad \gamma _ { k } ( \theta ) = - \theta ^ { \mathsf { T } } R ^ { ( k ) } , \qquad k = 1 , \ldots , d . } \end{array}\tag{7}
$$

The BAR identity (6) holds throughout the common transform domain and its analytic continuation (see Appendix A.1 for the complete derivation) and couples the d +1 unknown holomorphic functions $\big ( \phi _ { 0 } , \phi _ { 1 } , \ldots , \phi _ { d } \big )$ solely through the model parameters $( \mu , \Sigma , R )$

## 2.3 From learned transforms to tails

Once the interior transform $\phi _ { 0 } ( \theta )$ is learned, restricting it to the complex diagonal gives the Laplace transform of the total workload S, namely $\phi _ { S } ( s ) = \phi _ { 0 } ( s \mathbf { 1 } )$ . Integrating the survival function $p ( t ) = \mathbb { P } _ { \pi } ( S > t )$ yields the tail Laplace transform

$$
G _ { S } ( s ) = \int _ { 0 } ^ { \infty } e ^ { - s t } p ( t ) d t = { \frac { 1 - \phi _ { 0 } ( s { \bf 1 } ) } { s } } , \qquad \Re s > 0 .\tag{8}
$$

Following Dai and Zhang (2026), we parameterize the log transforms, train them to satisfy the BAR and structural constraints, and recover tail probabilities by applying numerical Laplace inversion (Abate and Whitt, 1995)—specifically the Fixed-Talbot algorithm (Abate and Valkó, 2004; Abate and Whitt, 2006)—to the learned tail transform

$$
\widehat { G } _ { S } ( s ) = \frac { 1 - \widehat { \phi } _ { 0 } ( s { \bf 1 } ) } { s } .\tag{9}
$$

Because Fixed-Talbot inversion evaluates ${ \widehat { G } } _ { S } ( s )$ at complex quadrature nodes along a deformed contour in the complex plane, accurate tail recovery requires a representation that maintains analyticity and numerical stability across complex arguments, paired with an inversion degree suited to the system dimension.

## 3 Holomorphic Cauchy Networks

Following the shared-coordinate log-transform formulation of Dai and Zhang (2026), we parameterize the interior and boundary Laplace transforms as

$$
\widehat { \phi } _ { k } ( \theta ) = \exp ( f _ { k } ( \theta ) ) , \qquad k = 0 , \ldots , d ,\tag{10}
$$

which stabilizes evaluations across regions where transform magnitudes vary exponentially. HCN constructs $f _ { k } ( \theta )$ using native complex Cauchy subnetworks, origin anchors, and boundary-coordinate exclusion. These constructions enforce the prescribed transform values at the origin and boundary-coordinate independence exactly, while ensuring holomorphicity on pole-free domains. The BAR and real-axis constraints are imposed through the training loss.

## 3.1 Complex Cauchy representation

For each hidden unit $r = 1 , \ldots , H$ , we extend the rational Cauchy activation of Li et al. (2025) and Si et al. (2026) to complex inputs and complex coefficients:

$$
\sigma _ { r } ( z ) = \frac { a _ { r } z + b _ { r } } { z ^ { 2 } + s _ { r } ^ { 2 } } , \qquad s _ { r } = \mathrm { s o f t p l u s } ( \rho _ { r } ) + s _ { \mathrm { m i n } } ,\tag{11}
$$

with trainable complex coefficients $a _ { r } , b _ { r } \in \mathbb { C }$ , trainable scale parameter $\rho _ { r } \in \mathbb { R }$ , and fixed floor $s _ { \operatorname* { m i n } } > 0$ Each scalar subnetwork on $\mathbb { C } ^ { p }$ takes the form

$$
q _ { \omega } ( x ) = w _ { 2 } ^ { \mathsf { T } } \sigma ( W _ { 1 } x + c _ { 1 } ) + c _ { 2 } , \qquad x \in \mathbb { C } ^ { p } ,\tag{12}
$$

where $W _ { 1 } \in \mathbb { C } ^ { H \times p } , c _ { 1 } , w _ { 2 } \in \mathbb { C } ^ { H }$ , and $c _ { 2 } \in \mathbb { C }$ are complex weights and biases, and $w _ { 2 } ^ { \mathsf { T } }$ uses the ordinary transpose without complex conjugation. Trainable complex embeddings identify coordinates and boundary faces and remain constant with respect to θ, while positive $s _ { r } \ge s _ { \mathrm { m i n } }$ regulates the locations of the simple poles at $z = \pm \mathrm { i } s _ { r }$

Proposition 1 (Structural holomorphicity). Fix network parameters for which all anchor evaluations are well defined, and let $U \subset \mathbb { C } ^ { d }$ be an open domain on which every Cauchy denominator appearing in $f _ { 0 } , \ldots , f _ { d }$ is nonzero. Then $f _ { 0 } , \ldots , f _ { d }$ and their exponentials $\widehat { \phi } _ { 0 } , \ldots , \widehat { \phi } _ { d }$ are holomorphic throughout U.

Proof. Complex affine maps and pole-free rational activations $\sigma _ { r } ( z )$ are holomorphic in z. Composition, finite summation, coordinate restriction, subtraction of well-defined anchors, and complex exponentiation preserve holomorphicity, so $f _ { k }$ and $\widehat { \phi _ { k } } = \exp ( f _ { k } )$ satisfy the Cauchy–Riemann identities on $U$ □

By Proposition 1, HCN satisfies the Cauchy–Riemann equations on every pole-free open set by construction, removing the sampled Cauchy–Riemann penalty $\mathcal { L } _ { \mathrm { C R } }$ and the computational graphs required to differentiate that penalty with respect to network parameters.

## 3.2 Shared coordinate representations and interactions

Let $e _ { j } \in \mathbb { C } ^ { p _ { e } }$ and $b _ { k } \in \mathbb { C } ^ { p _ { b } }$ denote trainable complex embedding vectors for coordinate $j \in \{ 1 , \dotsc , d \}$ and boundary face $k \in \{ 1 , \ldots , d \}$ , respectively. Separate shared subnetworks $q _ { \mathrm { i n t } }$ and $q _ { \mathrm { b d } }$ give the coordinate representations:

$$
f _ { 0 } ^ { \mathrm { c o o r d } } ( \theta ) = \sum _ { j = 1 } ^ { d } \left[ q _ { \mathrm { i n t } } ( \theta _ { j } , e _ { j } ) - q _ { \mathrm { i n t } } ( 0 , e _ { j } ) \right] ,\tag{13}
$$

$$
f _ { k } ^ { \mathrm { c o o r d } } ( \theta ) = \log { m _ { k } } + \sum _ { j \ne k } \bigl [ q _ { \mathrm { b d } } ( \theta _ { j } , e _ { j } , b _ { k } ) - q _ { \mathrm { b d } } ( 0 , e _ { j } , b _ { k } ) \bigr ] , \qquad k = 1 , \ldots , d ,\tag{14}
$$

where arguments denote concatenation and $m = - R ^ { - 1 } \mu$ is the known boundary-mass vector from (5). Anchoring at zero and excluding $\theta _ { k }$ from the kth boundary representation imply

$$
\widehat { \phi } _ { 0 } ( 0 ) = 1 , \qquad \widehat { \phi } _ { k } ( 0 ) = m _ { k } , \qquad \partial _ { \theta _ { k } } f _ { k } ( \theta ) = 0 , \qquad k = 1 , \ldots , d .\tag{15}
$$

The origin anchors enforce the prescribed interior and boundary transform masses, while exclusion of $\theta _ { k }$ enforces normal-coordinate independence for the kth boundary representation. These identities hold wherever the network evaluations are well defined and require no additional penalty terms.

The coordinate interior representation is separable: exp $\begin{array} { r } { \ L ) \big ( f _ { 0 } ^ { \mathrm { c o o r d } } ( \theta ) \big ) = \prod _ { j = 1 } ^ { d } \exp \bigl [ q _ { \mathrm { i n t } } \bigl ( \theta _ { j } , e _ { j } \bigr ) - q _ { \mathrm { i n t } } \bigl ( 0 , e _ { j } \bigr ) \bigr ] } \end{array}$ When this is a probability transform, it describes independent coordinates, and the boundary representations likewise impose separability within each face. They suit product-form benchmarks and have parameter count linear in d at fixed widths: $p _ { e } = p _ { b } = 1 6$ and width $H = 1 2 8$ give $1 5 , 1 0 8 + 6 4 d$ real scalar parameters (Appendix A.4).

For dependent transforms, we add anchored interaction subnetworks $g _ { \mathrm { i n t } }$ and g<sub>bd</sub>:

$$
f _ { 0 } ( \theta ) = f _ { 0 } ^ { \mathrm { c o o r d } } ( \theta ) + g _ { \mathrm { i n t } } ( \theta ) - g _ { \mathrm { i n t } } ( 0 ) ,\tag{16}
$$

$$
f _ { k } ( \pmb { \theta } ) = f _ { k } ^ { \mathrm { c o o r d } } ( \pmb { \theta } ) + g _ { \mathrm { b d } } ( \theta _ { - k } , b _ { k } ) - g _ { \mathrm { b d } } ( 0 , b _ { k } ) , \qquad k = 1 , \ldots , d ,\tag{17}
$$

where $\theta _ { - k } \in \mathbb { C } ^ { d - 1 }$ preserves the order of the remaining coordinates and the boundary zero lies in $\mathbb { C } ^ { d - 1 }$ Mixing coordinates before applying a Cauchy activation permits cross-coordinate interactions while retaining (15) and Proposition 1. The coordinate and interaction blocks are trained jointly, allowing the architecture to represent nonseparable log transforms while preserving exact origin normalization and boundary-coordinate exclusion. In two dimensions, each boundary face has only one free coordinate, so the 2D Harrison model uses an interior interaction block $g _ { \mathrm { i n t } }$ only; the 10D correlated model uses both interaction components (Appendix A.4).

## 3.3 Approximation with shared blocks and exact constraints

We next establish approximation properties for these architectures, allowing hidden widths to increase while keeping the embedding dimensions fixed. The target class is holomorphic, consistent with transform learning. While polynomial-based approximation is a standard tool for analytic neural networks (Park and Wojtowytsch, 2022), our construction treats the rational Cauchy activation in (11) directly and incorporates shared coordinate/boundary embeddings, origin anchors, and boundary-coordinate exclusion.

For $c \in \mathbb { C } ^ { d }$ and radii $r = ( r _ { 1 } , \ldots , r _ { d } ) \in ( 0 , \infty ) ^ { d }$ , write $D ( c , r ) = \{ \theta \in \mathbb { C } ^ { d } : | \theta _ { j } - c _ { j } | < r _ { j } , j = 1 , \dots , d \}$ . Fix nested polydiscs

$$
K = \overline { { { D ( c , r ) } } } , \qquad U = D ( c , r ^ { + } ) , \qquad | c _ { j } | < r _ { j } < r _ { j } ^ { + } < \infty , \qquad j = 1 , \ldots , d ,\tag{18}
$$

so that 0 lies in the interior of $K \Subset U$ . Define the simultaneous value and first-derivative norm

$$
\| F \| _ { 1 , K } = \operatorname* { m a x } \bigg \{ \operatorname* { s u p } _ { \theta \in K } \vert F ( \theta ) \vert , \ \operatorname* { m a x } _ { 1 \leq j \leq d } \operatorname* { s u p } _ { \theta \in K } \vert \partial _ { j } F ( \theta ) \vert \bigg \} .
$$

Let $F _ { 0 } , F _ { 1 } , \ldots , F _ { d }$ be holomorphic on $U _ { : }$ with $F _ { 0 } ( 0 ) = 0 , F _ { k } ( 0 ) = \log m _ { k }$ , and $F _ { k }$ independent of $\theta _ { k }$ for $k \geq 1$ Theorem 1 (Constrained HCN approximation). Fix $d \ge 2 , p _ { e } , p _ { b } \ge 1$ , and embeddings $e _ { j } \in \mathbb { C } ^ { p _ { e } }$ and $b _ { k } \in \mathbb { C } ^ { p _ { b } }$ that are pairwise distinct within eachfamily. For every $\eta > 0$ , thefollowing statements hold:

(i) Coordinate-separable targets. Ifthe targets are separable,

$$
F _ { 0 } ( \pmb \theta ) = \sum _ { j = 1 } ^ { d } h _ { j } ( \pmb \theta _ { j } ) , \qquad F _ { k } ( \pmb \theta ) = \log m _ { k } + \sum _ { j \neq k } h _ { k j } ( \pmb \theta _ { j } ) , \qquad k = 1 , \dots , d ,
$$

with univariate functions $h _ { j } , h _ { k j }$ holomorphic on the corresponding coordinate discs of U and zero at the origin, then the coordinate architecture (13)–(14) admitsfinite-width blocks satisfying

$$
\operatorname* { m a x } _ { 0 \leq k \leq d } \| f _ { k } - F _ { k } \| _ { 1 , K } < \eta .\tag{19}
$$

(ii) General dependent targets. For general targets satisfying the preceding holomorphicity, origin, and boundary-independence conditions, the interaction architecture (16)–(17) admits finite-width blocks satisfying (19). Its coordinate component may be fixed in advance, provided all its required denominators are nonzero on U.

In both cases, the identities (15) hold exactly. For any fixed $s _ { * } > s _ { \mathrm { m i n } } ,$ all constructed blocks can use scale $s _ { * }$ and have every raw denominator bounded below by ${ \frac { 3 } { 4 } } s _ { * } ^ { 2 }$ in modulus on the required inputsfrom $K ,$ including anchors.

The proof (Appendix A.2) first approximates multivariate polynomials by Cauchy blocks with a uniform denominator margin via roots-of-unity filtering, and then uses Lagrange interpolation over projections of the distinct embeddings $e _ { j }$ and $b _ { k }$ to realize all coordinate and boundary slices through shared blocks.

Part (i) establishes representation sufficiency for separable targets despite parameter sharing; part (ii) removes the separability restriction through anchored interactions. Conversely, every coordinate-only interior representation satisfies $\partial _ { i } \partial _ { j } f _ { 0 } ^ { \mathrm { c o o r d } } ( \theta ) = 0$ for $i \neq j .$ . Cauchy’s integral estimate therefore gives, whenever $F _ { 0 }$ is holomorphic near $K .$

$$
\operatorname* { s u p } _ { \theta \in K } \left| F _ { 0 } ( \theta ) - f _ { 0 } ^ { \mathrm { c o o r d } } ( \theta ) \right| \geq r _ { i } r _ { j } \big | \partial _ { i } \partial _ { j } F _ { 0 } ( c ) \big | , \qquad i \neq j .\tag{20}
$$

At $c = 0$ , for an analytic stationary log transform $F _ { 0 } = \log { \phi _ { 0 } }$ with finite second moments, the mixed derivative is $\partial _ { i } \partial _ { j } F _ { 0 } ( 0 ) = \mathrm { C o v } _ { \pi } ( Z _ { i } , Z _ { j } )$ . Nonzero cross-covariance thus creates a uniform approximation obstruction for the joint interior log transform on K that increasing coordinate-only width cannot remove, motivating the interaction architecture in (16)–(17).

## 3.4 Training constraints

The training objective follows the BAR learning framework of Dai and Zhang (2026). For complex interior samples $\theta ^ { ( n ) } , n = 1 , \ldots , N _ { \mathrm { B A R } }$ , a common real shift $\nu _ { n }$ stabilizes the exponential terms:

$$
\kappa _ { n k } = \gamma _ { k } ( \theta ^ { ( n ) } ) e ^ { f _ { k } ( \theta ^ { ( n ) } ) - \nu _ { n } } , \quad \quad A _ { n } = \kappa _ { n 0 } , \quad \quad B _ { n } = \sum _ { k = 1 } ^ { d } \kappa _ { n k } .\tag{21}
$$

We define the normalized relative residual and its weighted mean by

$$
\begin{array} { r } {  { \mathcal { E } } ( A , B ) = \frac { | A - B | ^ { 2 } } { ( | A | + | B | + \varepsilon ) ^ { 2 } } , \qquad { \mathcal { L } } _ { \mathrm { B A R } } = \frac { \sum _ { n = 1 } ^ { N _ { \mathrm { B A R } } } w _ { n }  { \mathcal { O } } ( A _ { n } , B _ { n } ) } { \sum _ { n = 1 } ^ { N _ { \mathrm { B A R } } } w _ { n } + \varepsilon _ { w } } , } \end{array}\tag{22}
$$

where $\varepsilon , \varepsilon _ { w } > 0$ are stabilizers and the detached weights $w _ { n } \in [ 0 . 5 , 2 ]$ emphasize larger residuals $( \mathsf { A p - }$ pendix B.1).

Additional BAR constraints are obtained by evaluating the transforms at auxiliary arguments that isolate a single boundary term. For sampled boundary-batch vectors $\boldsymbol { y } ^ { ( n ) } \in \mathbb { C } ^ { d } , n = 1 , \dots , N _ { \mathrm { b d } }$ , and standard basis vectors $u _ { k } \in \mathbb { R } ^ { d }$ , define

$$
\nu _ { k } = \frac { R ^ { - \top } u _ { k } } { ( R ^ { - \top } ) _ { k k } } , \qquad \widetilde { \theta } ^ { ( n , k ) } = y _ { k } ^ { ( n ) } \nu _ { k } , \qquad k = 1 , \ldots , d .\tag{23}
$$

Since $R ^ { \mathsf { T } } \nu _ { k } = u _ { k } / ( R ^ { - \mathsf { T } } ) _ { k k } ( \mathrm { w i t h } ( R ^ { - \mathsf { T } } ) _ { k k } = 1$ in all benchmarks considered here), all boundary coefficients except the kth vanish, reducing the BAR at $\widetilde { \theta } ^ { ( n , k ) }$ to the two-term relation $\gamma _ { 0 } \phi _ { 0 } = \gamma _ { k } \phi _ { k }$ . We evaluate the pairwise boundary loss directly via the shift-stabilized transform residual:

$$
\mathcal { L } _ { \mathrm { b d } } = \frac { 1 } { N _ { \mathrm { b d } } d } \sum _ { k = 1 } ^ { d } \sum _ { n = 1 } ^ { N _ { \mathrm { b d } } } \mathcal { O } \big ( \widetilde { \kappa } _ { 0 } ^ { ( n , k ) } , \widetilde { \kappa } _ { k } ^ { ( n , k ) } \big ) ,\tag{24}
$$

where each pair shares a common stabilizing shift, avoiding complex logarithms of the BAR coefficients.

At real arguments in their convergence domain, transforms of positive measures are positive and coordinatewise nonincreasing. For real samples $\boldsymbol { x } ^ { ( n ) } \in \mathbb { R } ^ { d } , n = 1 , \dots , N _ { \mathrm { r e a l } }$ , write $J _ { n k j } = \partial _ { j } f _ { k } ( x ^ { ( n ) } )$ . We encourage real, nonincreasing log transforms through

$$
\mathcal { L } _ { \mathrm { m o n o } } = \frac { \sum _ { n = 1 } ^ { N _ { \mathrm { r e a l } } } \sum _ { k = 0 } ^ { d } \sum _ { j = 1 } ^ { d } [ \operatorname* { m a x } \{ 0 , \Re J _ { n k j } \} ] ^ { 2 } } { N _ { \mathrm { r e a l } } ( d + 1 ) d } ,\tag{25}
$$

$$
\mathcal { L } _ { \mathrm { i m a g } } = \frac { \sum _ { n = 1 } ^ { N _ { \mathrm { r e a l } } } \sum _ { k = 0 } ^ { d } [ \Im f _ { k } ( x ^ { ( n ) } ) ] ^ { 2 } } { N _ { \mathrm { r e a l } } ( d + 1 ) } .\tag{26}
$$

The complete training loss is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { B A R } } + \mathcal { L } _ { \mathrm { b d } } + \mathcal { L } _ { \mathrm { m o n o } } + \mathcal { L } _ { \mathrm { i m a g } } , } \end{array}\tag{27}
$$

with all four weights equal to one in all reported experiments. Moreover, if the target transforms satisfy the BAR and real-axis monotonicity on a compact sampling support covered by Theorem 1, uniform log-transform approximation with $\eta \downarrow 0$ drives all four loss terms in (27) simultaneously to zero (Appendix A.3).

## 4 Training under GPU Memory Constraints

While structural holomorphicity completely eliminates the Cauchy–Riemann penalty and its derivative graph, evaluating the BAR residual at each sample still couples $d + 1$ transforms, and the real-axis monotonicity penalty requires first-order coordinate derivatives. To make high-dimensional training fast and memoryefficient on a single GPU, we combine closed-form complex input derivatives, sequential loss-branch backpropagation, and dimension-adapted batch schedules.

## 4.1 Explicit input derivatives

Because the rational Cauchy activation (11) and scalar subnetwork (12) are elementary holomorphic functions, their complex derivatives admit simple closed-form expressions:

$$
\sigma _ { r } ^ { \prime } ( z ) = \frac { a _ { r } ( s _ { r } ^ { 2 } - z ^ { 2 } ) - 2 b _ { r } z } { ( z ^ { 2 } + s _ { r } ^ { 2 } ) ^ { 2 } } ,\tag{28}
$$

$$
\begin{array} { r } { D _ { x } q _ { \omega } ( x ) = \big ( w _ { 2 } \odot \sigma ^ { \prime } ( W _ { 1 } x + c _ { 1 } ) \big ) ^ { \mathsf { T } } W _ { 1 } , } \end{array}\tag{29}
$$

where $\odot$ denotes the elementwise product and no complex conjugation enters the holomorphic chain rule. Coordinate blocks require only the scalar derivative with respect to their univariate transform input $\theta _ { j }$ whereas interaction blocks extract the corresponding columns of $W _ { 1 }$ . For the interior log transform, the jth partial derivative is simply

$$
\partial _ { j } f _ { 0 } ( \theta ) = \partial _ { 1 } q _ { \mathrm { i n t } } ( \theta _ { j } , e _ { j } ) + \partial _ { j } g _ { \mathrm { i n t } } ( \theta ) ,\tag{30}
$$

where the second term is omitted in the coordinate-separable model, and origin anchors vanish under $\partial _ { j }$ while retaining their dependence on trainable network parameters.

We evaluate input derivatives explicitly only for the real-axis monotonicity loss ${ \mathcal { L } } _ { \mathrm { m o n o } }$ ; the interior BAR loss $\mathcal { L } _ { \mathrm { B A R } }$ and isolated boundary loss ${ \mathcal { L } } _ { \mathrm { b d } }$ require forward function evaluations alone. Computing $J _ { n k j } =$ $\partial _ { j } f _ { k } ( x ^ { ( n ) } )$ directly via the closed-form expressions (28)–(30) replaces the inner automatic-differentiation pass used to construct input derivatives. The explicit Jacobian expressions remain differentiable with respect to the network parameters ω, and reverse-mode automatic differentiation computes the parameter gradients of the resulting losses.

## 4.2 Sequential backpropagation and the sampling budget

In a d-dimensional RBM, evaluating all d boundary log transforms across a batch of B arguments requires $B d ( d - 1 )$ coordinate–face pair evaluations in addition to origin anchors. Linear parameter growth therefore does not imply linear training-memory growth. To bound individual tensor allocations during the forward pass, we chunk coordinate–face pair evaluations and group boundary faces in the high-dimensional product-form implementation.

To prevent backward-pass intermediate activations from accumulating across all loss terms simultaneously, our 30D and 50D implementations further evaluate and backpropagate the interior BAR branch $( \mathcal { L } _ { \mathrm { B A R } } )$ the isolated boundary branch $( \mathcal { L } _ { \mathrm { b d } } )$ , and the combined real-axis branch $( \mathcal { L } _ { \mathrm { m o n o } } + \mathcal { L } _ { \mathrm { i m a g } } )$ in sequence. Each branch executes its backward pass, accumulates parameter gradients, and releases its computation graph before the next branch is constructed, with anchors recomputed in fresh graphs:

$$
\begin{array} { r } { \nabla _ { \omega } \mathcal { L } = \nabla _ { \omega } \mathcal { L } _ { \mathrm { B A R } } + \nabla _ { \omega } \mathcal { L } _ { \mathrm { b d } } + \nabla _ { \omega } ( \mathcal { L } _ { \mathrm { m o n o } } + \mathcal { L } _ { \mathrm { i m a g } } ) . } \end{array}\tag{31}
$$

For fixed parameters ω, samples, and detached BAR weights $w _ { n }$ , sequential accumulation yields the same summed-loss gradient as a combined backward pass up to floating-point arithmetic, while reducing the simultaneous storage of branch-specific backward intermediates. Global gradient-norm clipping and a single Adam update are then applied once all three branches have accumulated their gradients (Algorithm 1).

## Algorithm 1. One optimizer update with sequential loss backpropagation

1. Clear gradients and sample fresh BAR, boundary, and real-axis batches.

2. Evaluate the BAR loss ${ \mathcal { L } } _ { \mathrm { B A R } } ;$ ; backpropagate and release its computation graph.

3. Evaluate the boundary loss ${ \mathcal { L } } _ { \mathrm { b d } } ;$ accumulate its gradients and release its graph.

4. Evaluate real-axis values and explicit Jacobians (28)–(30); backpropagate $\mathcal { L } _ { \mathrm { m o n o } } + \mathcal { L } _ { \mathrm { i m a g } }$ and release the graph.

5. Check gradient finiteness, apply global gradient-norm clipping with threshold 1.0 to the accumulated gradient, and take one Adam update.

The 10D implementation additionally accumulates boundary faces separately within Step 3, retaining the $1 / d$ factor in (24). The 2D and 20D implementations use a combined backward call. These are graphscheduling choices rather than changes to the training objective.

We count freshly sampled transform-argument vectors as base inputs. With U optimizer updates, the cumulative sampling budget is

$$
N _ { \mathrm { t r a i n } } = U \big ( B _ { \mathrm { B A R } } + B _ { \mathrm { b d } } + B _ { \mathrm { r e a l } } \big ) ,\tag{32}
$$

where a boundary base input supplies d isolated arguments (23) and a real-axis base input supplies two losses; derived arguments and origin anchors are excluded from $N _ { \mathrm { t r a i n } }$ . As dimension increases, Table 1 reduces per-update batch sizes and increases the number of optimizer updates $U$ to maintain a per-run budget of 64 million sampled base inputs in the high-dimensional experiments.

Table 1: HCN batches and fixed training endpoints.
<table><tr><td>Benchmark</td><td>Updates (U)</td><td> $B _ { \mathrm { B A R } }$ </td><td> $B _ { \mathrm { b d } }$ </td><td> $B _ { \mathrm { r e a l } }$ </td><td>Base inputs  $( N _ { \mathrm { t r a i n } } )$ </td></tr><tr><td>2D, both models</td><td>15,000</td><td>3,416</td><td>427</td><td>427</td><td>64.05M</td></tr><tr><td>10D non-product</td><td>25,000</td><td>2,048</td><td>256</td><td>256</td><td>64M</td></tr><tr><td>20D product</td><td>25,000</td><td>2,048</td><td>256</td><td>256</td><td>64M</td></tr><tr><td>30D product</td><td>50,000</td><td>1,024</td><td>128</td><td>128</td><td>64M</td></tr><tr><td>50D product</td><td>100,000</td><td>512</td><td>64</td><td>64</td><td>64M</td></tr></table>

$\mathbf { M } = 1 0 ^ { 6 }$ base inputs per run. Updates count optimizer steps, rather than backward calls.

## 4.3 Sampling and optimization

Each update resamples arguments. The BAR batch mixes approximately 50% general complex inputs, 20% complex-diagonal inputs, 20% near-diagonal inputs, and a remainder on the nonnegative real diagonal. Diagonal coverage reflects the query $\phi _ { 0 } ( s \mathbf { 1 } )$ used for sum tails, while off-diagonal inputs constrain the multivariate BAR. The nested-box sampler and integer rounding are documented in Appendix B.

Complex samples are drawn from a bounded domain $\Theta = [ \underline { { \theta } } ^ { \mathrm { R e } } , \overline { { \theta } } ^ { \mathrm { R e } } ] ^ { d } \times [ - \overline { { \theta } } ^ { \mathrm { I m } } , \overline { { \theta } } ^ { \mathrm { I m } } ] ^ { d }$ following the twostage nested-box sampling scheme of Dai and Zhang (2026). Boundary base vectors are drawn from the same complex sampler and mapped to the isolated arguments in (23), while real-axis batches use the corresponding real-part projection of the nested-box sampler.

All HCN training runs use complex128 arithmetic and Adam with learning rate $5 \times 1 0 ^ { - 5 }$ for the first 80% of updates and $1 0 ^ { - 5 }$ thereafter, retaining optimizer state across the schedule transition. Gradient clipping follows gradient accumulation, and preflight denominator and finiteness checks verify evaluated inputs (Appendix B.3). All reported evaluations in Section 5 use the fixed endpoints in Table 1.

## 5 Numerical Experiments

We evaluate resource requirements, tail accuracy, and extension beyond product form across 2- to 50- dimensional benchmarks. We also examine numerical inversion degree, complex activation choice, and late-training stability in the 50D run. Throughout all experiments, networks are trained solely from the RBM parameters $( \mu , \Sigma , R )$ ; analytical and simulation references enter post-training evaluation only.

## 5.1 Benchmarks and evaluation protocol

The product-form benchmark family of Dai and Zhang (2026) uses $d \in \{ 2 , 2 0 , 3 0 , 5 0 \}$ , µ = −1, $R _ { i i } = 1$ $R _ { i , i - 1 } = - 1$ , and $\Sigma = R + R ^ { \top }$ , with unspecified entries zero. Under this skew-symmetric structure, the stationary coordinates are independent Exp(i) variables (Harrison and Williams, 1987), giving exact sumworkload transform and tail probability

$$
\phi _ { S } ( s ) = \prod _ { i = 1 } ^ { d } \frac { i } { i + s } , \qquad p _ { d } ( t ) = 1 - ( 1 - e ^ { - t } ) ^ { d } , \qquad t \geq 0 .\tag{33}
$$

The two-dimensional Harrison non-product benchmark (Dai and Harrison, 1992; Dai and Zhang, 2026) has parameters

$$
\mu = \binom { - 1 } { 0 } , \qquad \Sigma = I _ { 2 } , \qquad R = \binom { 1 } { - 1 } \quad 1 \biggr ) .\tag{34}
$$

Its reference tail probabilities are obtained by one-dimensional polar quadrature of the known stationary density (Appendix A.5). The exploratory ten-dimensional correlated non-product model is specified in Section 5.6.

We evaluate the fixed endpoints in Table 1 using a common random seed (12345), and include three seeds (12345, 22345, and 32345) at 20D to summarize initialization variability via the sample mean and standard deviation of per-run errors. No minimum-error checkpoint is substituted for a fixed endpoint. Fixed-Talbot inversion uses degree $M = 5$ in 2D and $M = 8$ in higher dimensions; at 30D, we also report $M = 5$ for the degree-matched comparison with published NN4MGF results. Changing the inversion degree does not retrain the network.

Following the probability-clipping convention in the released high-dimensional NN4MGF evaluation code, all finite real inversion outputs are projected onto [0, 1] via $\widehat { p } _ { M } ( t ) = \operatorname* { m i n } \{ 1 , \operatorname* { m a x } \{ 0 , T _ { M } [ \widehat { G } _ { S } ] ( t ) \} \}$ . Against positive reference probabilities, mean relative error (MRE, in percent) is

$$
\mathrm { M R E } _ { 1 : K } = \frac { 1 0 0 } { K } \sum _ { t = 1 } ^ { K } \frac { | \widehat { p } _ { M } ( t ) - p _ { \mathrm { r e f } } ( t ) | } { p _ { \mathrm { r e f } } ( t ) } .\tag{35}
$$

We use $\mathrm { M R E } _ { 1 : 8 }$ as the primary accuracy metric for the product-form benchmarks and the exploratory 10D example, and additionally report $\mathrm { M R E } _ { 1 : 5 }$ and $\mathrm { M R E } _ { 1 : 1 0 }$ for the product-form benchmarks to characterize shorter and extended tail ranges. The 2D Harrison benchmark uses $\mathrm { M R E } _ { 1 : 5 }$ to match the evaluation range of Dai and Zhang (2026). The 2D and 30D NN4MGF MREs are computed from the pointwise tail predictions released in the official code repository of Dai and Zhang (2026), while 20D uses archived Degree-5 and Degree-8 evaluations of the released NN4MGF checkpoint under their respective training budgets.

## 5.2 Computational performance and dimensional scaling

Both 30D configurations are profiled on the same NVIDIA H100 NVL GPU, with 10 warmup updates followed by 200 measured updates. CUDA synchronization barriers bracket update timing, and peak-memory statistics are reset after warmup. Timings include in-loop sampling, forward evaluation, and backpropagation, excluding model construction and final output generation.

Table 2: Thirty-dimensional training profiles on the same NVIDIA H100 NVL.
<table><tr><td>Quantity</td><td>HCN</td><td>NN4MGF</td></tr><tr><td>Real scalar parameters</td><td>17,028</td><td>43,080</td></tr><tr><td>Base inputs / update</td><td>1,280</td><td>32,768</td></tr><tr><td>Measured updates (after 10 warmup)</td><td>200</td><td>200</td></tr><tr><td>Time for 200 updates (s)</td><td>33.023</td><td>320.209</td></tr><tr><td>Mean time / update (s)</td><td>0.165113</td><td>1.601046</td></tr><tr><td>Peak allocated memory (GiB)</td><td>7.805</td><td>77.815</td></tr><tr><td>Peak reserved memory (GiB)</td><td>8.414</td><td>81.428</td></tr><tr><td>Updates in full schedule</td><td>50,000</td><td>300,000</td></tr><tr><td>Base inputs in full schedule</td><td>64M</td><td>9.8304B</td></tr><tr><td>Projected full-schedule update time (h)</td><td>2.293</td><td>133.420</td></tr></table>

Both use double precision: complex128 for HCN and float64 for NN4MGF. Times and memory peaks are measured over the profile; full-schedule times are projections. GiB denotes $2 ^ { 3 0 }$ bytes. The methods use their respective batches and architectures.

As shown in Table 2, mean per-update times are 0.165113 s for HCN and 1.601046 s for NN4MGF, giving a 9.7-fold reduction in mean update time (ratio 9.697). Peak allocated GPU memory is 7.805 GiB for HCN versus 77.815 GiB for NN4MGF, an approximately tenfold reduction (ratio 9.970). At 30D, the tested NN4MGF configuration reserves 81.428 GiB of H100 memory, reflecting the memory cost of retaining first-order derivative graphs across large batches for the derivative-based penalties. By removing $\mathcal { L } _ { \mathrm { C R } }$ structurally and combining explicit real-axis Jacobians with sequential branch backpropagation, HCN keeps peak reserved memory under 8.5 GiB.

Multiplying the measured mean update times by the respective scheduled update counts (50,000 for HCN and 300,000 for NN4MGF) gives projected full-schedule update times of 2.293 and 133.420 hours:

$$
\frac { \widehat { T } _ { \mathrm { N N 4 M G F } } } { \widehat { T } _ { \mathrm { H C N } } } = \frac { 1 . 6 0 1 0 4 5 5 2 } { 0 . 1 6 5 1 1 2 8 5 } \frac { 3 0 0 , 0 0 0 } { 5 0 , 0 0 0 } \approx 9 . 6 9 7 \times 6 = 5 8 . 1 8 .\tag{36}
$$

The 58.2-fold projected full-schedule update-time ratio combines the measured 9.7-fold per-update ratio with the sixfold difference in scheduled update counts. This projection extrapolates the short-profile measurements to the respective full training schedules rather than measuring completed training runs or time to a common accuracy target, and compares the complete tested configurations—including their architectures, batch sizes, derivative calculations, and graph schedules—rather than isolating the contribution of CR penalty removal.

The tested NN4MGF code path samples two batches of 16,384 per update, giving $3 0 0 , 0 0 0 \times 2 \times 1 6 , 3 8 4 =$ 9.8304 billion base inputs over its full schedule. HCN’s 64M cumulative budget represents 0.651% of that count (1.302% of the paper-reported $3 0 0 , 0 0 0 \times 1 6 , 3 8 4$ single-batch count; Appendix E.3). Separately, this reduced memory footprint allows all HCN training experiments in this study to be completed on a single GPU, scaling up to the 50D benchmark (100,000 updates, 18,308 real scalar parameters, and 64M base inputs) on a single NVIDIA GeForce RTX 4080 Laptop GPU.

## 5.3 Tail accuracy and the 50D extension

Table 3 compares tail errors at the fixed training endpoints. Across all degree-matched comparisons over the primary evaluation ranges, HCN attains lower reported MRE than NN4MGF. The error reductions are substantial on the two-dimensional benchmarks—especially on the non-product Harrison model—and modest on the 20D and 30D product-form benchmarks, where HCN achieves competitive accuracy using less than 1% of the baseline sampling budget while extending computation to 50 dimensions.

Table 3: Tail accuracy at fixed training endpoints (MRE, percent).
<table><tr><td>Benchmark</td><td>Method</td><td>M</td><td>1:5</td><td>1:8</td><td>1:10</td></tr><tr><td>2D product</td><td>HCN</td><td>5</td><td>2.147</td><td>2.213</td><td>9.737</td></tr><tr><td>2D product</td><td>NN4MGF (published)</td><td>5</td><td>2.448</td><td>8.050</td><td>13.915</td></tr><tr><td>2D Harrison</td><td>HCN</td><td>5</td><td>1.422</td><td></td><td></td></tr><tr><td>2D Harrison</td><td>NN4MGF (published)</td><td>5</td><td>6.684</td><td></td><td></td></tr><tr><td>20D product</td><td>HCN</td><td>8</td><td>1.441±1.322</td><td>7.632±1.565</td><td>12.264±3.678</td></tr><tr><td>20D product</td><td>NN4MGF (archived)</td><td>8</td><td>2.802</td><td>8.148</td><td>18.908</td></tr><tr><td>30D product</td><td>HCN (comparison)</td><td>5</td><td>3.429</td><td>10.107</td><td>19.084</td></tr><tr><td>30D product</td><td>NN4MGF (published)</td><td>5</td><td>8.353</td><td>10.476</td><td>21.507</td></tr><tr><td>30D product</td><td>HCN (primary)</td><td>8</td><td>1.683</td><td>6.546</td><td>8.872</td></tr><tr><td>50D product</td><td>HCN (extension)</td><td>8</td><td>1.495</td><td>6.367</td><td>8.011</td></tr></table>

The 20D HCN entries are the mean ± sample SD over three seeds; SD is in percentage points. Other entries are single runs or single baseline records. The same 30D HCN weights are used at both degrees. Dashes indicate unavailable reporting ranges.

At 2D (Figure 1), the product-form 1:8 errors are 2.213% for HCN and 8.050% for NN4MGF (9.737% versus 13.915% on 1:10). On the non-product 2D Harrison benchmark, HCN with the interior interaction branch $g _ { \mathrm { i n t } }$ reduces $\mathrm { M R E } _ { 1 : 5 }$ to 1.422% compared with 6.684% for NN4MGF. Here the HCN Harrison architecture matches the HCN 2D product-form model’s parameter budget (15,236 real scalar parameters and 64.05M inputs) by adjusting subnetwork widths to accommodate the interior interaction branch (Appendix A.4). This improvement illustrates the effectiveness of the anchored interaction representation on a non-product RBM with dependent coordinates.

![](images/6c5562848b4068510598f395b72048c9e0baa39067a145e7d588339619c12312.jpg)

![](images/f05355392a6fa54f4c5a8fe4243a5cfee93c00bd14151cb5ab4aaaa08604a6b1.jpg)  
Figure 1: 2D tails at Degree 5. HCN uses the fixed 15,000-update endpoints; NN4MGF uses published values from Dai and Zhang (2026). References are analytical in (a) and density-based in (b).

In the high-dimensional product-form benchmarks (Figure 2):

• 20D benchmark: At Degree 8, HCN achieves a three-seed mean $\mathrm { M R E } _ { 1 : 8 }$ of 7.632% (SD 1.565 percentage points; individual runs 9.003%, 7.966%, and 5.927%), compared with 8.148% for NN4MGF. On 1:5 and 1:10, HCN’s mean MREs are 1.441% ± 1.322% and 12.264% ± 3.678%, compared with 2.802% and 18.908% for NN4MGF.

• 30D benchmark: Under the degree-matched M = 5 rule, $\mathrm { M R E } _ { 1 : 8 }$ is 10.107% for HCN versus 10.476% for NN4MGF (3.429% versus 8.353% on 1:5, and 19.084% versus 21.507% on 1:10). Evaluating the same 30D HCN weights with the primary M = 8 rule reduces $\mathbf { M R E } _ { 1 : 5 } , \mathbf { M R E } _ { 1 : 8 }$ , and $\mathrm { M R E } _ { 1 : 1 0 }$ to 1.683%, 6.546%, and 8.872%, respectively.

• 50D extension: Completed on a single laptop GPU, the 50D HCN run achieves $\mathrm { M R E } _ { 1 : 5 }$ of 1.495%, $\mathrm { M R E } _ { 1 : 8 }$ of $6 . 3 6 7 \%$ , and $\mathrm { M R E } _ { 1 : 1 0 }$ of 8.011%, closely tracking the analytical survival curve across the evaluation range (Figure 2c).

## 5.4 Inversion and activation diagnostics

When initially evaluating our models under the baseline Degree M = 5 rule, we observed that HCN achieved lower error than NN4MGF in 2D and 30D, yet showed slightly higher error on segments of the 20D benchmark (for instance, on 1:10 under M = 5, HCN’s three-seed mean MRE was 19.347% versus 16.801% for NN4MGF). This discrepancy led us to ask whether the intrinsic truncation error of the Fixed-Talbot algorithm at $M = 5$ was canceling with NN4MGF’s approximation error or reinforcing HCN’s error. Writing $p _ { M } ^ { \star } ( t ) = T _ { M } [ G _ { S } ] ( t )$ for the output obtained by applying Degree-M Fixed-Talbot inversion to the exact transform, the prediction error decomposes into two signed components:

(a) 20D  
![](images/eb17efeb688f3e32139c5fea02fb234c0d714740abd3b706d4dc9551cec2eec4.jpg)

![](images/a3029d8dedd10e420fabd7e9f254829905d6ff0be7abc30f1711dc1e1f9aa4d1.jpg)

![](images/bc915bf75dfa25028305e4f569fa6ca587a3d6e711419686c4e5b2507ebae2b1.jpg)  
Figure 2: Product-form tails at Degree 8. Panel (a) shows the three 20D HCN runs and the archived NN4MGF checkpoint. Panels (b) and (c) show the fixed endpoints at 30D and 50D.

$$
\widehat { p } _ { M } ( t ) - p ( t ) = \underbrace { \bigl ( p _ { M } ^ { \star } ( t ) - p ( t ) \bigr ) } _ { e _ { \mathrm { i n v } , M } ( t ) } + \underbrace { \bigl ( \widehat { p } _ { M } ( t ) - p _ { M } ^ { \star } ( t ) \bigr ) } _ { e _ { \mathrm { n e t } , M } ( t ) } .\tag{37}
$$

These two signed components represent intrinsic inversion error and network approximation error, respectively, and can partially cancel or reinforce each other when summed.

Table 4: Fixed-Talbot inversion of the exact product-form transform: Degree 5 versus Degree 8 (MRE, percent).
<table><tr><td rowspan="2">Dimension</td><td colspan="2">t = 1:5</td><td colspan="2">t = 1:8</td><td colspan="2">t = 1:10</td></tr><tr><td> $M = 5$ </td><td> $M = 8$ </td><td> $M = 5$ </td><td> $M = 8$ </td><td> $M = 5$ </td><td> $M = 8$ </td></tr><tr><td>20D</td><td>3.391</td><td>0.073</td><td>4.044</td><td>0.122</td><td>9.430</td><td>0.332</td></tr><tr><td>30D</td><td>3.218</td><td>0.152</td><td>3.955</td><td>0.178</td><td>8.397</td><td>0.375</td></tr><tr><td>50D</td><td>34.866</td><td>0.507</td><td>23.898</td><td>0.414</td><td>23.352</td><td>0.563</td></tr></table>

Only the exact analytical transform is inverted; no neural network approximation enters this table. Bold entries indicate the lower error in each degree pair.

To test this hypothesis independently of neural approximation, we applied the Fixed-Talbot procedure directly to the exact product-form transforms in (33). As shown in Table 4, Degree M = 5 indeed incurs substantial intrinsic inversion error in higher dimensions (4.044% at 20D, 3.955% at 30D, and 23.898% at 50D on 1:8), and at 20D this $M = 5$ quadrature bias partially cancels NN4MGF’s transform error, making the Degree-5 metric favorable to NN4MGF (Appendix C.2). Increasing the Talbot degree to $M = 8$ reduces the exact-transform inversion error on 1:8 to 0.122% at 20D, 0.178% at 30D, and 0.414% at 50D, while keeping exact-transform MRE below 0.57% on 1:5 and 1:10 across all three dimensions. We therefore adopt Degree 8 as our primary high-dimensional evaluation metric. Evaluating the exact same 20D trained weights under Degree 8—where intrinsic inversion bias is largely removed—HCN achieves lower mean MRE than NN4MGF across all evaluation ranges (1.441% versus 2.802% on 1:5, 7.632% versus 8.148% on 1:8, and 12.264% versus 18.908% on 1:10; Appendix D.1). Beyond motivating our evaluation protocol, this diagnostic highlights an important practical lesson for BAR-based transform learning: in high dimensions, sufficient inversion degree is needed to prevent quadrature errors from masking or distorting network performance.

Table 5: Activation comparison at 20D and Degree 8: MRE on 1:8 (percent).
<table><tr><td>Seed</td><td>Cauchy MRE (%) Complex-tanh MRE (%)</td></tr><tr><td>12345</td><td>9.003</td></tr><tr><td>22345</td><td>32.990 7.966 19.823 13.502</td></tr><tr><td>32345</td><td>5.927</td></tr><tr><td> $\mathbf { M e a n } \pm \mathbf { S D }$ </td><td> $7 . 6 3 2 \pm 1 . 5 6 5$   $2 2 . 1 0 5 \pm 9 . 9 4 2$ </td></tr></table>

All three seeds use the identical coordinate architecture, 64M input budget, and fixed 25,000-update endpoint.

To examine activation choice within the complex architecture, the 20D complex-tanh control retains the exact same coordinate architecture, widths, seeds, 64M input budget, and 25,000-update endpoint. Both activations are holomorphic away from their poles and omit CR penalties, with Cauchy’s trainable activation parameters giving 16,388 real scalars versus 15,108 for complex-tanh. As shown in Table 5, Cauchy achieves lower $\mathrm { M R E } _ { 1 : 8 }$ on every seed at Degree 8, with a three-seed mean of $7 . 6 3 2 \% \pm 1 . 5 6 5 \%$ compared with $2 2 . 1 0 5 \% \pm 9 . 9 4 2 \%$ for complex-tanh.

## 5.5 Late-training stability in 50 dimensions

To examine whether the 50D tail accuracy is sustained across late-stage updates rather than isolated to a single checkpoint, we evaluate five archived checkpoints along the 50D trajectory from update 92,000 to 100,000 in increments of 2,000 updates under the fixed Degree-8 protocol.

Table 6: Late-training tail accuracy along the 50D trajectory.
<table><tr><td>Update 1:8 MRE (%) Role</td></tr><tr><td>92,000 6.650 Diagnostic checkpoint</td></tr><tr><td>94,000 5.299 Diagnostic checkpoint</td></tr><tr><td>96,000 6.845 Diagnostic checkpoint</td></tr><tr><td>98,000 7.155 Diagnostic checkpoint</td></tr><tr><td>100,000 6.367 Reported endpoint</td></tr></table>

All checkpoints are evaluated at Degree 8. The main accuracy comparison uses the fixed 100,000-update endpoint.

Table 6 shows $\mathrm { M R E } _ { 1 : 8 }$ values between 5.299% (at update 94,000) and 7.155% (at update 98,000), with 6.367% at the prescribed 100,000-update endpoint. Errors fluctuate nonmonotonically within a narrow single-digit range across all five checkpoints, providing evidence of late-training stability along this trajectory under the fixed evaluation rule (see Appendix D.2 for the corresponding 1:5 interval).

## 5.6 An exploratory 10D non-product example

Beyond two dimensions, non-product RBMs lack closed-form joint stationary densities or Laplace transforms, so evaluating tail accuracy requires steady-state simulation (Blanchet et al., 2021). With standard time-discretized trajectory simulation, however, generating reliable tail probability references becomes computationally demanding and numerically delicate as dimension grows: simultaneously controlling reflection discretization bias across d faces, burn-in to stationarity, and relative Monte Carlo variance in the tail requires substantial simulation effort, and within our computational budget we do not have a certified simulation ground truth in 30 or 50 dimensions. We therefore conduct our multidimensional non-product evaluation at $d = 1 0 ,$ , a scale that is large enough to test multi-coordinate coupling while still permitting a pilot steady-state simulation reference alongside an exact analytical first-moment cross-check.

Specifically, to introduce cross-coordinate dependence while retaining known stationary marginals, consider the 10-dimensional orthogonally reflected RBM

$$
d = 1 0 , \qquad R = I _ { 1 0 } , \qquad \mu = - { \bf 1 } , \qquad \Sigma _ { i j } = \frac { 2 \rho ^ { | i - j | } } { \sqrt { i j } } , \qquad \rho = 0 . 3 .\tag{38}
$$

Here $\Sigma$ is a positive diagonal rescaling of an $\mathbf { A R } ( 1 )$ correlation matrix. Because $R = I _ { 1 0 } .$ , each coordinate $Z _ { i }$ is marginally a one-dimensional RBM with drift −1 and variance rate $2 / i ,$ , so $Z _ { i } \sim \mathrm { E x p } ( i )$ in stationarity. However, nonzero off-diagonal covariances violate the product-form skew-symmetry condition for $R = I _ { 1 0 }$ (Harrison and Williams, 1987). Consequently,

$$
p _ { \mathrm { i n d } } ( t ) = 1 - ( 1 - e ^ { - t } ) ^ { 1 0 }\tag{39}
$$

serves as an independent-marginal comparator rather than the correlated system’s true sum-tail reference.

![](images/008590717471a69013acafd1b42f3894260d6f2a9491d2047b6209f3506206ad.jpg)  
Figure 3: Exploratory 10D non-product tails at Degree 8: HCN interaction model compared with the pilot simulation reference and independent-marginal tail.

We evaluate the HCN interaction model (incorporating both $g _ { \mathrm { i n t } }$ and $g _ { \mathrm { b d } } )$ at 25,000 updates and Degree 8. As an independent analytical diagnostic of the learned interior transform at the origin, evaluating ${ \widehat { G } } _ { S } ( 0 ) =$ $\textstyle - \sum _ { j = 1 } ^ { 1 0 } \partial _ { j } f _ { 0 } ( 0 )$ yields a learned mean total workload of 2.924299, which differs from the exact analytical first moment $\begin{array} { r } { \mathbb { E } _ { \pi } [ S ] = H _ { 1 0 } = \sum _ { i = 1 } ^ { 1 0 } 1 / j \approx 2 . 9 2 8 9 6 8 } \end{array}$ by only 0.159% (Appendix D.3). For the sum tail, the numerical reference is a pilot steady-state simulation estimate for which a complete discretization and sampling error bound has not yet been established; evaluated against this pilot reference, HCN achieves an $\mathrm { M R E } _ { 1 : 8 }$ of 3.684%, compared with 19.409% for the independent-marginal comparator $p _ { \mathrm { i n d } } ( t )$ $\mathbf { A } \mathbf { t } \ t = 8$ the pilot reference, HCN prediction, and independent-marginal probabilities are 0.005905, 0.005512, and 0.003350, respectively (Figure 3). Relative to the independent-marginal comparator, this result provides evidence that the anchored interaction representation captures cross-coordinate dependence effects on the sum tail in a multidimensional non-product setting.

## 6 Discussion

Structural holomorphy and memory-controlled computation. On pole-free domains, native complex Cauchy subnetworks are holomorphic by construction, eliminating sampled Cauchy–Riemann (CR) penalties and the first-order input-derivative graphs they require across complex collocation points. In a d-dimensional RBM, however, replacing the activation function alone is not sufficient to resolve the high-dimensional memory bottleneck: each parameter update still evaluates BAR and real-axis monotonicity constraints across one interior transform and d boundary transforms, so practical memory demand also depends on derivative implementation and graph scheduling. Pairing native complex Cauchy representations with closed-form real-axis derivatives, coordinate–face chunking, and sequential three-branch backpropagation yields the runtime and memory reductions observed in the 30D H100 profile. Because HCN and NN4MGF differ in their architectures, batch sizes, scheduled update counts, derivative calculations, and graph schedules, those 30D ratios characterize the complete configured pipelines jointly rather than isolating CR penalty removal alone. In practice, combining this memory-controlled implementation with dimension-dependent batch schedules and the 64M base-input budget allows all tested HCN benchmarks to run on a single GPU and enables the 50D product-form run on a single laptop GPU; across the five recorded checkpoints from 92,000 to 100,000 updates in Section 5.5, $\mathrm { M R E } _ { 1 : 8 }$ remains between 5.299% and 7.155%, with the fixed 100,000-update endpoint (6.367%) reported as the primary result.

Constrained approximation and non-product dependence. Theorem 1 characterizes fixed-dimension representation capacity under holomorphic-log assumptions: parameter-shared additive coordinate blocks simultaneously approximate function values and first derivatives of separable log transforms on compact polydiscs, while anchored interaction subnetworks approximate general holomorphic targets satisfying exact origin normalization and boundary-coordinate independence. For the coordinate-only architecture used in the product-form benchmarks, sharing subnetwork weights across coordinates and faces avoids replicating full networks for each coordinate or boundary face, so at fixed width only the coordinate and face embedding parameters grow linearly with dimension $( P ( d ) = 1 5 , 1 0 8 + 6 4 d$ in Appendix A.4). Conversely, the mixed-derivative lower bound (20) shows why coordinate-only representations are structurally limited on correlated systems: whenever stationary coordinates have nonzero cross-covariance, the true joint log transform has nonzero mixed partial derivatives at the origin, whereas any purely additive representation has identically zero mixed partials and therefore incurs an irreducible uniform approximation error on the joint log transform over compact polydiscs that increasing width alone cannot remove. Although (20) bounds uniform log-transform error rather than directly bounding sum-tail MRE after numerical inversion, the empirical improvement on the 2D Harrison benchmark (1.422% versus $6 . 6 8 4 \% \mathrm { M R E } _ { 1 : 5 } )$ and the results on the exploratory 10D correlated RBM are consistent with this structural distinction. In the 10D study, because the trajectory reference is a pilot estimate without a certified discretization bound, we treat the tail comparison as exploratory multi-dimensional evidence; meanwhile, the 0.159% relative discrepancy between the learned origin derivative sum and the analytical mean workload $H _ { 1 0 }$ provides an independent first-moment consistency check on the interior transform at the origin, even though matching the sum mean alone does not certify cross-coordinate covariance.

Separating transform approximation from numerical inversion. In BAR-based solvers, the neural network learns complex log-Laplace transforms, and tail probabilities are subsequently computed via Fixed Talbot inversion. Accordingly, (37) decomposes the final tail error ${ \widehat { p } } _ { M } - p$ into two signed components: the intrinsic inversion error $e _ { \mathrm { i n v } , M } = p _ { M } ^ { \star }$ −p incurred by applying Talbot quadrature to the exact transform, and the network approximation error $e _ { \mathrm { n e t } , M } = \widehat { p } _ { M } - p _ { M } ^ { \star }$ from replacing the exact transform with the learned network at the complex Talbot nodes. Because both components are signed, they can partially cancel or reinforce each other across thresholds. As a result, even when increasing the Talbot degree substantially reduces exact-transform inversion error—which motivates our choice of Degree 8 in 20 to 50 dimensions—changing the complex evaluation nodes and weights can occasionally expose or amplify network errors that were previously masked by sign cancellation (Appendix D.1). Tail accuracy should therefore be evaluated for a specific pair of learned transform and inversion rule: where analytical transforms are available, exacttransform diagnostics should be used first to select an appropriate inversion degree, after which learned models are compared at fixed training endpoints under explicit evaluation rules.

Scope and future directions. Several extensions remain open. First, Theorem 1 establishes the existence of pole-free approximations at finite but unrestricted width on compact polydiscs under a nonvanishingtransform assumption; it does not by itself guarantee optimization convergence at a fixed trained width, nor do structural holomorphicity and finitely many sampled BAR and real-axis constraints certify that a learned transform corresponds to a globally valid probability measure. Second, because the trainable Cauchy scale parameters $s _ { r }$ govern complex pole locations, relating learned pole distributions directly to the geometry of the Fixed-Talbot contour could strengthen the theoretical foundation for deep-tail inversion. Finally, developing steady-state simulation references with certified error bounds for larger correlated RBMs and conducting single-factor resource ablations will further clarify the scope of holomorphic neural solvers and the individual contributions of their design components.

## 7 Conclusion

This paper develops Holomorphic Cauchy Networks (HCNs) for stationary RBM tail computation within NN4MGF’s BAR-learning and shared-coordinate framework. By embedding rational Cauchy activations into native complex-valued subnetworks, HCN enforces holomorphicity on pole-free domains by construction without sampled Cauchy–Riemann penalties, while exact origin anchors and boundary-coordinate exclusion enforce normalization and face-independence identities algebraically. Under holomorphic-log assumptions, we establish simultaneous value and first-derivative approximation for separable targets with shared coordinate blocks and for general constrained holomorphic targets with anchored interaction blocks on compact polydiscs, and derive a mixed-derivative lower bound showing why coordinate-only representations cannot approximate joint log transforms arbitrarily well when stationary coordinates have nonzero cross-covariance. Combined with closed-form real-axis derivatives, sequential branch backpropagation, and dimension-dependent batch schedules, this design substantially reduces GPU memory demand and update time, allowing all tested benchmarks to run on a single GPU and scaling the 50-dimensional product-form benchmark to a single laptop GPU with stable accuracy across late-training checkpoints. Across degree-matched comparisons over the primary evaluation ranges, HCN attains lower reported mean relative errors than NN4MGF, including reducing error from 6.684% to 1.422% on the two-dimensional Harrison non-product benchmark, while an exploratory ten-dimensional correlated example illustrates the practical applicability of the interaction extension beyond two dimensions. Together with exact-transform and signed-error diagnostics, these findings support the joint design of holomorphic network architectures, memory-aware training schedules, and numerical Laplace inversion for BAR-based stationary-tail solvers.

## References

Joseph Abate and Péter P. Valkó. Multi-precision Laplace transform inversion. International Journalfor Numerical Methods in Engineering, 60(5):979–993, 2004. doi: 10.1002/nme.995.

Joseph Abate and Ward Whitt. Numerical inversion of Laplace transforms of probability distributions. ORSA Journal on Computing, 7(1):36–43, 1995. doi: 10.1287/ijoc.7.1.36.

Joseph Abate and Ward Whitt. A unified framework for numerically inverting Laplace transforms. INFORMS Journal on Computing, 18(4):408–421, 2006. doi: 10.1287/ijoc.1050.0137.

Baris Ata, J. Michael Harrison, and Nian Si. Drift control of high-dimensional reflected Brownian motion: A computational method based on neural networks. Stochastic Systems, 15(2):111–146, 2025. doi: 10.1287/stsy.2023. 0044.

Jose Blanchet, Xinyun Chen, Nian Si, and Peter W. Glynn. Efficient steady-state simulation of high-dimensional stochastic networks. Stochastic Systems, 11(2):174–192, 2021. doi: 10.1287/stsy.2021.0077.

J. G. Dai and J. Michael Harrison. Reflected Brownian motion in an orthant: Numerical methods for steady-state analysis. The Annals of Applied Probability, 2(1):65–86, 1992. URL https://people.orie.cornell. edu/jdai/publications/daiHarrison92.pdf.

J. G. Dai, Masakiyo Miyazawa, and Jian Wu. A multi-dimensional SRBM: Geometric views of its product form stationary distribution. Queueing Systems, 78(4):313–335, 2014. doi: 10.1007/s11134-014-9411-0.

Jim Dai and Zhanhao Zhang. Deep learning method for stationary distribution of reflected Brownian motion, 2026. URL https://arxiv.org/abs/2607.08091.

J. Michael Harrison and Ruth J. Williams. Multidimensional reflected Brownian motions having exponential stationary distributions. The Annals ofProbability, 15(1):115–137, 1987. doi: 10.1214/aop/1176992259.

Xin Li, Zhihong Xia, and Hongkun Zhang. Cauchy activation function and XNet. Neural Networks, 188:107375, 2025. doi: 10.1016/j.neunet.2025.107375.

Josiah Park and Stephan Wojtowytsch. Qualitative neural network approximation over R and C: Elementary proofs for analytic and polynomial activation, 2022. URL https://arxiv.org/abs/2203.13410.

Yanlin Qu, Jose Blanchet, and Peter W. Glynn. Deep learning for computing convergence rates of Markov chains. In Advances in Neural Information Processing Systems, volume 37, pages 84777–84798, 2024.

Yanlin Qu, Jose Blanchet, and Peter W. Glynn. Deep learning for Markov chains: Lyapunov functions, Poisson’s equation, and stationary distributions. Queueing Systems, 110(1):10, 2026. doi: 10.1007/s11134-025-09969-8.

Maziar Raissi, Paris Perdikaris, and George E. Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378:686–707, 2019. doi: 10.1016/j.jcp.2018.10.045.

Chenhao Si, Ming Yan, Xin Li, and Zhihong Xia. Complex physics-informed neural network. Journal of Computational Physics, 553:114713, 2026. doi: 10.1016/j.jcp.2026.114713.

## Appendices

The appendices provide mathematical details, constrained approximation proofs, and reference formulas (Appendix A), implementation settings (B), inversion diagnostics (C), supplementary results (D), and reproducibility information (E).

## A Mathematical and Architectural Details

## A.1 Stationary balance and boundary measures

For $h _ { \theta } ( z ) = e ^ { - \theta ^ { \mathsf { T } } z }$ , the interior generator and reflection derivatives are

$$
\frac { \boldsymbol { \mathcal { A } } h _ { \theta } ( \boldsymbol { z } ) } { h _ { \theta } ( \boldsymbol { z } ) } = \frac { 1 } { 2 } \theta ^ { \top } \Sigma \theta - \mu ^ { \top } \theta , \qquad \frac { \nabla h _ { \theta } ( \boldsymbol { z } ) ^ { \top } R ^ { ( k ) } } { h _ { \theta } ( \boldsymbol { z } ) } = - \theta ^ { \top } R ^ { ( k ) } .\tag{A.1}
$$

Under the integrability conditions in Section 2, stationarity and Itô’s formula imply

$$
0 = \Big ( { \textstyle \frac { 1 } { 2 } } \theta ^ { \mathsf { T } } { \boldsymbol { \Sigma } } \theta - \mu ^ { \mathsf { T } } \theta \Big ) \phi _ { 0 } \big ( \theta \big ) - \sum _ { k = 1 } ^ { d } \theta ^ { \mathsf { T } } R ^ { ( k ) } \phi _ { k } \big ( \theta \big ) .\tag{A.2}
$$

This gives (6) with the signs in (7). The same stationarity argument applied to the coordinates gives $R m = - \mu$ The boundary measure

$$
\nu _ { k } ( A ) = \mathbb { E } _ { \pi } \left[ \int _ { 0 } ^ { 1 } \mathbf { 1 } _ { \{ Z ( u ) \in A \} } d Y _ { k } ( u ) \right]
$$

has total mass $m _ { k }$ and support on $\{ z : z _ { k } = 0 \}$ . Thus $\phi _ { k }$ is a finite-measure Laplace transform with prescribed mass $m _ { k }$ at the origin. In the product-form family, $m _ { k } = k ;$ in the 2D Harrison and orthogonally reflected 10D models, every $m _ { k } = 1$

The boundary-isolation directions in (23) satisfy

$$
R ^ { \mathsf { T } } \nu _ { k } = \frac { u _ { k } } { ( R ^ { - \mathsf { T } } ) _ { k k } } ,\tag{A.3}
$$

which verifies that all boundary coefficients except the kth vanish at $y _ { k } \nu _ { k }$ . For $R = I ,$ , the isolated argument simplifies to $y _ { k } u _ { k }$

## A.2 Proofs of the constrained approximation results

We use the nested polydisc domains and first-derivative norm $\| \cdot \| _ { 1 , K }$ defined in Section 3.3. All derivatives in this subsection are complex derivatives, and the construction uses the exact rational Cauchy activation and scalar block parameterization in (11)–(12).

Lemma A.1 (Polynomial approximation by Cauchy blocks). Let P be a multivariate polynomial on $\mathbb { C } ^ { p } { } _ { ; }$ $L \subset \mathbb { C } ^ { p }$ a compact set, and $s _ { * } > s _ { \mathrm { m i n } } .$ . For every $\delta > 0 ,$ , there exists afinite-width Cauchy block q whose value andfirst-derivative errors relative to P are strictly less than $\delta$ on L. Moreover, every hidden scale parameter can be set to $s _ { r } = s _ { * } ,$ with every raw denominator satisfying $\begin{array} { r } { | z ^ { 2 } + s _ { * } ^ { 2 } | \geq \frac { 3 } { 4 } s _ { * } ^ { 2 } } \end{array}$ on a neighborhood ofL.

Proof. Set $a _ { r } = - \mathrm { i } s ,$ <sub>∗</sub> and $b _ { r } = s _ { * } ^ { 2 }$ in the rational Cauchy activation (11). For preactivation $z = \mathrm { i } s _ { \mathrm { * } } t$ with $| t | \leq 1 / 2$ , factoring the denominator gives

$$
\sigma ( \mathrm { i } s _ { \ast } t ) = \frac { 1 + t } { 1 - t ^ { 2 } } = \frac { 1 } { 1 - t } , \qquad | z ^ { 2 } + s _ { \ast } ^ { 2 } | = s _ { \ast } ^ { 2 } | 1 - t ^ { 2 } | \geq \frac { 3 } { 4 } s _ { \ast } ^ { 2 } .\tag{A.4}
$$

Both raw poles $t = \pm 1$ lie strictly outside $| t | \leq 1 / 2$ , so the implemented fraction is smooth and bounded away from zero throughout the disk, and $s _ { * }$ is realizable via $\rho _ { r } = \log ( \exp ( s _ { * } - s _ { \mathrm { m i n } } ) - 1 )$ since $s _ { * } > s _ { \mathrm { m i n } }$

For any integer degree $n \geq 1$ , choose $N > n ,$ step $h > 0 ;$ , and primitive root of unity $\omega = e ^ { 2 \pi \mathrm { i } / N }$ . By the geometric series and roots-of-unity orthogonality filter,

$$
\frac { 1 } { N h ^ { n } } \sum _ { \ell = 0 } ^ { N - 1 } \frac { \omega ^ { - \ell n } } { 1 - h \omega ^ { \ell } u } = \sum _ { q = 0 } ^ { \infty } h ^ { q N } u ^ { n + q N } = \frac { u ^ { n } } { 1 - ( h u ) ^ { N } } , \qquad | h u | < 1 .\tag{A.5}
$$

For $| u | \le A$ and $h A < 1$ , the uniform error relative to the monomial $u ^ { n }$ is bounded by

$$
A ^ { n } { \frac { ( h A ) ^ { N } } { 1 - ( h A ) ^ { N } } } \longrightarrow 0 \qquad { \mathrm { a s ~ } } h \neq 0 .\tag{A.6}
$$

Each summand in $( \mathsf { A } . 5 )$ is realized by a single Cauchy unit using preactivation is $\mathbf { \nabla } _ { \mathbf { \mathcal { A } } } \omega ^ { \ell } \boldsymbol { u }$ and output weight $\omega ^ { - \ell n } / ( N h ^ { n } )$ . Any affine linear form $u ( x )$ is realized by the complex input layer, and choosing $h A \leq 1 / 2$ guarantees the uniform denominator margin (A.4).

Finally, powers of linear forms span all multivariate polynomials via the polarization identity:

$$
x _ { 1 } \cdots x _ { n } = { \frac { 1 } { n ! } } \sum _ { S \subseteq \{ 1 , \ldots , n \} } ( - 1 ) ^ { n - | S | } { \biggl ( } \sum _ { j \in S } x _ { j } { \biggr ) } ^ { n } ,\tag{A.7}
$$

which remains valid when variables are repeated coordinates. Summing the finitely many resulting Cauchy units approximates P uniformly on any compact polydisc, with constant terms absorbed into the output bias $c _ { 2 } .$ . Enclosing L in the interior of a slightly larger compact polydisc, uniform value convergence on the outer polydisc implies simultaneous first-derivative convergence on L by Cauchy’s integral estimates, while choosing h sufficiently small so that $| h \omega ^ { \ell } u | \leq 1 / 2$ on the outer polydisc preserves the denominator margin ${ \frac { 3 } { 4 } } s _ { * } ^ { 2 }$ □

Interpolation of shared labels. For any finite family of pairwise distinct boundary embeddings $b _ { k } \in \mathbb { C } ^ { p _ { b } }$ choose a complex linear functional λ such that the scalar projections $\beta _ { k } = \lambda ( b _ { k } )$ are pairwise distinct (which exists since the finitely many excluded equalities define proper hyperplanes). The univariate Lagrange polynomials

$$
D _ { k } ( b ) = \prod _ { \ell \neq k } \frac { \lambda ( b ) - \beta _ { \ell } } { \beta _ { k } - \beta _ { \ell } }
$$

satisfy $D _ { k } ( b _ { \ell } ) = \mathbf { 1 } _ { k = \ell }$ . Consequently, for any family of polynomials $P _ { k } ( z )$ , the single joint polynomial $\begin{array} { r } { Q ( z , b ) = \sum _ { k } D _ { k } ( b ) P _ { k } ( z ) } \end{array}$ interpolates all slices simultaneously, and Lemma A.1 approximates Q and its z-derivatives using a single shared Cauchy block on any compact envelope of the labelled inputs. An identical construction provides interpolating polynomials $E _ { j } ( e )$ for the distinct coordinate embeddings $e _ { j } \in \mathbb { C } ^ { p _ { e } }$

ProofofTheorem 1(i). By Taylor’s theorem and Cauchy’s integral estimates on an intermediate polydisc inside $U .$ , univariate polynomials $P _ { j }$ and $P _ { k j }$ approximate each target component $h _ { j }$ and $h _ { k j }$ , together with its first derivative, on the corresponding closed coordinate disc of K. Forming the joint interpolating polynomials

$$
\displaystyle { \cal Q } _ { \mathrm { i n t } } ( z , e ) = \sum _ { j = 1 } ^ { d } E _ { j } ( e ) P _ { j } ( z ) , \qquad { \cal Q } _ { \mathrm { b d } } ( z , e , b ) = \sum _ { k = 1 } ^ { d } \sum _ { j \neq k } D _ { k } ( b ) E _ { j } ( e ) P _ { k j } ( z ) ,
$$

we apply Lemma A.1 on a compact envelope containing all required labelled inputs and origin anchors (since 0 lies in every coordinate disc of $K )$ so that each slice of $q _ { \mathrm { i n t } }$ and $q _ { \mathrm { b d } }$ has value and first-derivative error below $\delta > 0$ . For any slice with target satisfying $h ( 0 ) = 0$ , writing $E ( z ) = q ( z ) - h ( z )$ gives anchored error $E ( z ) - E ( 0 )$ , which is bounded by $2 \delta$ in value and $\delta$ in derivative. Summing at most d coordinate slices yields total value error below $2 d \delta$ and coordinate derivative error below $\delta ;$ choosing $\delta < \eta / ( 2 d )$ establishes (19) while preserving (15) and the uniform denominator margin. □

ProofofTheorem $I ( i i )$ . Let $B _ { 0 } , B _ { 1 } , \ldots , B _ { d }$ denote the fixed coordinate-backbone functions (including log $m _ { k } )$ which are holomorphic on U and satisfy (15). The residual targets $A _ { k } = F _ { k } - B _ { k }$ are holomorphic on $U$ and vanish at the origin, with $A _ { k }$ depending only on $\theta _ { - k } \in K _ { - k }$ for $k \geq 1$ . Choose multivariate Taylor polynomials $P _ { 0 } , P _ { 1 } , \ldots , P _ { d }$ approximating $A _ { 0 } , A _ { 1 } , \ldots , A _ { d }$ to within δ in $\Vert \cdot \Vert _ { 1 , K }$ on $K , K _ { - 1 } , \ldots , K _ { - d }$ . Applying Lemma A.1 approximates $P _ { 0 }$ by $g _ { \mathrm { i n t } }$ within $\delta ,$ , and interpolating $P _ { 1 } , \ldots , P _ { d }$ via $\begin{array} { r } { Q ( z , b ) = \sum _ { k = 1 } ^ { d } D _ { k } ( b ) P _ { k } ( z ) } \end{array}$ on an enclosing compact polydisc in $\mathbb { C } ^ { d - 1 } \times \mathbb { C } ^ { p _ { b } }$ approximates all boundary residuals by a single shared block $g _ { \mathrm { b d } }$ within $\delta .$ . On each slice, the unanchored error $E = g - A$ has value and first-derivative bounds below 2δ; since $A ( 0 ) = 0$ , origin anchoring yields error $E ( { \boldsymbol { \theta } } ) - E ( { \boldsymbol { 0 } } )$ bounded by 4δ in value and 2δ in derivative. Setting $\delta < \eta / 4$ proves (19). □

Coordinate-only obstruction. Applying the two-variable Cauchy integral formula to $F _ { 0 } - f _ { 0 } ^ { \mathrm { c o o r d } }$ along the ith and jth coordinate circles of K (holding other coordinates at c) and using $\partial _ { i } \partial _ { j } f _ { 0 } ^ { \mathrm { c o o r d } } \equiv 0$ for $i \neq j$ yields $| \partial _ { i } \partial _ { j } F _ { 0 } ( c ) | \le ( r _ { i } r _ { j } ) ^ { - 1 } \operatorname* { s u p } _ { K } | F _ { 0 } - f _ { 0 } ^ { \mathrm { c o o r d } } |$ , which rearranges to (20). Since $\phi _ { 0 } ( 0 ) = 1 , \partial _ { i } \phi _ { 0 } ( 0 ) = - \mathbb { E } _ { \pi } [ Z _ { i } ]$ ], and $\partial _ { i } \partial _ { j } \phi _ { 0 } ( 0 ) = \mathbb { E } _ { \pi } [ Z _ { i } Z _ { j } ]$ , differentiating $F _ { 0 } = \log$ φ<sub>0</sub> at c = 0 gives $\partial _ { i } \partial _ { j } \log \phi _ { 0 } ( 0 ) = \mathrm { C o v } _ { \pi } ( Z _ { i } , Z _ { j } )$

## A.3 Consequences for transforms and the training constraints

Let $\phi _ { k } = e ^ { F _ { k } }$ and $\widehat { \phi _ { k } } = e ^ { f _ { k } }$ , and suppose the log-transform approximation bound (19) holds with tolerance $\eta > 0$ . Writing $E _ { k } = f _ { k } - F _ { k }$ and using $| e ^ { E _ { k } } - 1 | \leq e ^ { | E _ { k } | } - 1$ on K yields

$$
| \widehat { \phi } _ { k } - \phi _ { k } | \leq | \phi _ { k } | ( e ^ { \eta } - 1 ) ,\tag{A.8}
$$

$$
| \partial _ { j } \widehat { \phi } _ { k } - \partial _ { j } \phi _ { k } | \leq | \phi _ { k } | \left[ ( e ^ { \eta } - 1 ) | \partial _ { j } F _ { k } | + e ^ { \eta } \eta \right] .\tag{A.9}
$$

When the exact transforms satisfy the BAR identity (6) on $K ,$ subtracting (6) gives

$$
\left| \gamma _ { 0 } { \widehat { \phi } } _ { 0 } - \sum _ { k = 1 } ^ { d } \gamma _ { k } { \widehat { \phi } } _ { k } \right| \leq ( e ^ { \eta } - 1 ) { \Bigg ( } | \gamma _ { 0 } \phi _ { 0 } | + \sum _ { k = 1 } ^ { d } | \gamma _ { k } \phi _ { k } | { \Bigg ) } ,\tag{A.10}
$$

whose right-hand multiplier is uniformly bounded on the compact set K. Under the common stabilizing shift $\begin{array} { r } { \nu _ { f } ( \theta ) = \mathrm { m a x } _ { 0 \le k \le d } \{ \mathrm { l o g } ( | \gamma _ { k } ( \theta ) | + \varepsilon _ { \gamma } ) + \Re f _ { k } ( \theta ) \} } \end{array}$ , uniform convergence $f _ { k } \to F _ { k }$ on K bounds $e ^ { - \nu _ { f } }$ uniformly above, while the denominator of E in (22) is bounded below by $\varepsilon ^ { \overset { \triangledown } { 2 } }$ and the weights $w _ { n } \in [ 0 . 5 , 2 ]$ are bounded. Consequently, (A.10) implies $\mathcal { L } _ { \mathrm { B A R } } \to 0$ and $\mathcal { L } _ { \mathrm { b d } } \to 0$ as $\eta \downarrow 0$ . Likewise, on any real sample support in K where $F _ { k }$ is real and coordinatewise nonincreasing, $| \Im f _ { k } | \le \eta$ and max $\{ 0 , \Re \partial _ { j } f _ { k } \} \le \eta , \mathrm { s o } \mathcal { L } _ { \mathrm { m o n o } } \le \eta ^ { 2 }$ and $\mathcal { L } _ { \mathrm { i m a g } } \leq \eta ^ { 2 }$ . Thus all four loss components in (27) tend to zero simultaneously as $\eta \downarrow 0$

## A.4 Parameter counts and interaction allocations

A scalar Cauchy block with p complex inputs and H hidden units contains $p H$ complex input weights, H complex hidden biases, H complex output weights, one complex output bias, two complex activation coefficients per unit, and one real scale parameter per unit. Counting each complex scalar as two real scalars gives

$$
P _ { \mathrm { C a u c h y } } ( p , H ) = 2 ( p H + H + H + 1 + 2 H ) + H = H ( 2 p + 9 ) + 2 .\tag{A.11}
$$

With $p _ { e } = p _ { b } = 1 6$ , the coordinate and boundary embeddings contribute $6 4 d$ real parameters, and the coordinate interior and boundary input dimensions are 17 and 33, respectively. Setting $H = 1 2 8$ yields $1 5 , 1 0 8 + 6 4 d$ real parameters. A complex-tanh block has $H ( 2 p + 4 ) + 2$ real parameters since its activation contains no trainable coefficients.

Table A.1: Hidden widths of the scalar subnetworks and real scalar parameter counts.
<table><tr><td>Model</td><td> $q _ { \mathrm { i n t } }$ </td><td>qbd</td><td> $g _ { \mathrm { i n t } }$ </td><td> $g _ { \mathrm { b d } }$  Parameters</td></tr><tr><td>2D product</td><td>128</td><td>128</td><td></td><td>15,236</td></tr><tr><td>2D Harrison</td><td>95</td><td>128</td><td>109</td><td>15,236</td></tr><tr><td>10D non-product</td><td>95</td><td>128</td><td>109 109</td><td>23,925</td></tr><tr><td>20D product, Cauchy</td><td>128</td><td>128</td><td></td><td>16,388</td></tr><tr><td>20D product, tanh</td><td>128</td><td>128</td><td></td><td>15,108</td></tr><tr><td>30D product</td><td>128</td><td>128</td><td></td><td>17,028</td></tr><tr><td>50D product</td><td>128</td><td>128</td><td></td><td>18,308</td></tr></table>

The four middle columns list hidden-unit counts for the coordinate interior, coordinate boundary, interior interaction, and boundary interaction blocks. Parameter totals include embeddings and count each complex parameter as two real scalars.

In the 2D Harrison model, the two embedding arrays contribute 128 real parameters, while $q _ { \mathrm { i n t } } , q _ { \mathrm { b d } } .$ , and $g _ { \mathrm { i n t } }$ contribute 4,087, 9,602, and 1,419 parameters, matching the 15,236-parameter budget of the 2D productform model. In the 10D correlated model, the corresponding contributions are $6 4 0 + 4 , 0 8 7 + 9 , 6 0 2 + 3 , 1 6 3 +$ $6 { , } 4 3 3 = 2 3 { , } 9 2 5$ , where the final term is the shared boundary interaction block $g _ { \mathrm { b d } }$ with input dimension $9 + 1 6 = 2 5$

## A.5 Analytical and density-based references

For the product-form family, $R ^ { - 1 } \mathbf { 1 } = ( 1 , . . . , d ) ^ { \mathsf { T } }$ and $\Sigma = R + R ^ { \top }$ , giving independent exponential stationary coordinates with rates $1 , \ldots , d$ (Harrison and Williams, 1987). Letting $F _ { d } ( t ) = ( 1 - e ^ { - t } ) ^ { d }$ for $t \geq 0$ and substituting $u = e ^ { - t }$ yields

$$
\int _ { 0 } ^ { \infty } e ^ { - s t } d F _ { d } ( t ) = d \int _ { 0 } ^ { 1 } u ^ { s } ( 1 - u ) ^ { d - 1 } d u = \frac { \Gamma ( d + 1 ) \Gamma ( s + 1 ) } { \Gamma ( s + d + 1 ) } = \prod _ { i = 1 } ^ { d } \frac { i } { i + s } , \quad \mathrm { R e } s > 0 ,\tag{A.12}
$$

confirming (33). In floating-point evaluation, computing $p _ { d } ( t ) = - \exp 1 \{ d \log 1 \} ( - e ^ { - t } ) \}$ avoids catastrophic cancellation at large thresholds.

For the 2D Harrison benchmark (34), the exact stationary density (Dai and Harrison, 1992; Dai and Zhang, 2026) in polar coordinates $( z _ { 1 } , z _ { 2 } ) = ( r \cos \alpha , r \sin \alpha )$ is

$$
\rho _ { \mathrm { H } } ( r \cos \alpha , r \sin \alpha ) = \frac { 2 ^ { 3 / 2 } } { \sqrt { \pi } } r ^ { - 1 / 2 } e ^ { - ( 1 + \cos \alpha ) r } \cos ( \alpha / 2 ) , \quad r > 0 , \quad 0 \leq \alpha \leq \pi / 2 .\tag{A.13}
$$

With area element dz<sub>1</sub> $d z _ { 2 } = r d r d \alpha .$ , the tail event $z _ { 1 } + z _ { 2 } > t$ imposes the radial lower limit $r _ { 0 } ( \alpha ) = t / ( \cos \alpha +$ sin α). Integrating out r analytically via the substitution $u = ( 1 + \cos \alpha ) \scriptscriptstyle { I }$ r yields the one-dimensional integral representation

$$
p _ { \mathrm { H } } ( t ) = \frac { 2 ^ { 3 / 2 } } { \sqrt { \pi } } \int _ { 0 } ^ { \pi / 2 } \frac { \cos ( \alpha / 2 ) } { ( 1 + \cos \alpha ) ^ { 3 / 2 } } \Gamma \biggl ( \frac { 3 } { 2 } , \frac { ( 1 + \cos \alpha ) t } { \cos \alpha + \sin \alpha } \biggr ) d \alpha ,\tag{A.14}
$$

where $\textstyle \Gamma ( a , x ) = \int _ { x } ^ { \infty } u ^ { a - 1 } e ^ { - u }$ du is the upper incomplete gamma function. High-precision quadrature of (A.14) supplies the exact reference probabilities at $t = 1 , \ldots , 5$ , agreeing with the rounded values in Dai and Zhang (2026) to within $6 \times 1 0 ^ { - 8 }$

## B Training and Implementation Details

## B.1 Stable residuals

For $\gamma _ { n k } = \gamma _ { k } ( \theta ^ { ( n ) } )$ and $f _ { n k } = f _ { k } ( \boldsymbol { \theta } ^ { ( n ) } )$ , the common real shift in (21) is

$$
\nu _ { n } = \operatorname* { m a x } _ { 0 \leq k \leq d } \{ \log ( | \gamma _ { n k } | + \varepsilon _ { \gamma } ) + \mathrm { R e } f _ { n k } \} , \qquad \varepsilon _ { \gamma } = 1 0 ^ { - 1 2 } .\tag{B.1}
$$

This normalization bounds each scaled term $\left| \kappa _ { n k } \right|$ by one in exact arithmetic while preserving the zeroresidual locus of the BAR equation. The relative residual in (22) uses $\varepsilon = 1 0 ^ { - 8 }$ . For $\ell _ { n } = \mathcal { E } ( A _ { n } , B _ { n } )$ and the stop-gradient operator sg, the adaptive sample weights are defined by

$$
u _ { n } = ( \mathrm { s g } ( \ell _ { n } ) + \varepsilon _ { w } ) ^ { 0 . 3 } , \qquad \varepsilon _ { w } = 1 0 ^ { - 1 2 } ,\tag{B.2}
$$

$$
w _ { n } = \mathrm { s g } \left[ \operatorname* { m i n } \left\{ 2 , \operatorname* { m a x } \left\{ 0 . 5 , \frac { u _ { n } } { N _ { \mathrm { B A R } } ^ { - 1 } \sum _ { m } u _ { m } + \varepsilon _ { w } } \right\} \right\} \right] .\tag{B.3}
$$

At isolated boundary arguments (23), the stabilizing shift takes the maximum over the two active terms log $\vert \gamma _ { j } \vert + \varepsilon _ { \gamma } ) + \mathrm { R e } f _ { j }$ for $j \in \{ 0 , k \}$ , and the pairwise boundary loss (24) takes the unweighted mean across samples and boundary faces.

## B.2 Sampling and numerical settings

Following Dai and Zhang (2026), the general complex sampler draws points from random nested boxes on a bounded domain $\Theta = [ \underline { { \theta } } ^ { \mathrm { R e } } , \overline { { \theta } } ^ { \mathrm { R e } } ] ^ { d } \times [ - \overline { { \theta } } ^ { \mathrm { I m } } , \overline { { \theta } } ^ { \mathrm { I m } } ] ^ { d }$ . For each base input, we independently draw $U , V \sim$ $\mathrm { U n i f } ( 0 , 1 )$ , set the box thresholds $A = \underline { { \theta } } ^ { \mathrm { R e } } + ( \overline { { \theta } } ^ { \mathrm { R e } } - \underline { { \theta } } ^ { \mathrm { R e } } ) U$ and $C = { \overline { { { \theta } } } } ^ { \mathrm { I m } } V$ , and draw coordinates conditionally independently as

$$
\operatorname { R e } \theta _ { j } \mid A \sim \operatorname { U n i f } ( \underline { { \theta } } ^ { \mathrm { R e } } , A ) , \qquad \operatorname { I m } \theta _ { j } \mid C \sim \operatorname { U n i f } ( - C , C ) , \qquad j = 1 , \ldots , d .\tag{B.4}
$$

Sharing $( A , C )$ across coordinates concentrates a higher fraction of samples in corner regimes where all coordinates simultaneously take small or large values. Complex-diagonal inputs repeat a single univariate complex draw across all d coordinates; near-diagonal inputs add independent real and imaginary perturbations from $\left[ - 0 . 0 5 , 0 . 0 5 \right]$ and project back into $\Theta ;$ and nonnegative real-diagonal inputs take $\overline { { \theta } } ^ { \mathrm { R e } } U ^ { 2 } { \bf 1 }$ . Each interior batch $B _ { \mathrm { B A R } }$ allocates integer parts of 50%, 20%, and 20% to the first three components and assigns the remainder to the real diagonal. Boundary base vectors use the general complex sampler before applying (23), and real-axis samples use the real-part projection of (B.4).

Complex affine layers are implemented via $\mathtt { t o r c h }$ .nn.Linear $( \dots ,$ dtype=complex128) with standard Kaiming fan-in initialization. During sequential branch accumulation (Algorithm 1), all loss branches evaluate the exact same parameter vector ω prior to global gradient clipping and the single Adam update.

## B.3 Preflight verification checks

Before training commences, automated preflight diagnostics verify numerical well-posedness and closed-form Jacobian accuracy at initialization. We monitor the minimum denominator modulus $\Delta _ { \mathrm { d e n } } = \operatorname* { m i n } | h _ { r } ( \theta ) ^ { 2 } + s _ { r } ^ { 2 } |$ across all hidden preactivations $h _ { r } ( \theta )$ and verify finiteness of all forward outputs, Jacobians, losses, and parameter gradients. In addition, the 2D and 10D routines verify the closed-form complex input derivatives (28)–(30) against centered finite differences:

$$
D _ { j } ^ { ( h ) } F ( x ) = \frac { F ( x + h u _ { j } ) - F ( x - h u _ { j } ) } { 2 h } , \qquad h = 1 0 ^ { - 6 } .\tag{B.5}
$$

Table B.1: Implementation hyperparameters and numerical settings.  
Setting Value   
Arithmetic complex128 network values and weights; float64 real quantities.   
Embeddings $p _ { e } = p _ { b } = 1 6$ complex components per coordinate and boundary face, initialized   
$\mathcal { N } ( 0 , 0 . 0 1 ^ { 2 } )$   
Cauchy initialization $a _ { r } = b _ { r } = 0 . 1 , s _ { r } = 0 . 5 , s _ { \mathrm { m i n } } = 1 0 ^ { - 3 } ; \rho _ { r } = \log ( \exp ( 0 . 5 - s _ { \mathrm { m i n } } ) - 1 ) .$   
Optimizer Adam, learning rate $5 \times 1 0 ^ { - 5 }$ for the first 80% of updates, $1 0 ^ { - 5 }$ thereafter.   
Loss and gradients Unit weights (1,1,1,1) in (27); global accumulated gradient norm clipped at 1.0.   
Pair chunk cap 32,768 scalar inputs in 2D and product-form models; 4,096 in 10D.   
Backward schedule Combined pass in 2D and 20D; sequential three-branch pass in 30D and 50D; facewise   
boundary accumulation in 10D.

As shown in Table B.2, the maximum discrepancy between our explicit analytical Jacobians and centered finite differences is on the order of $1 0 ^ { - 1 0 } \mathrm { t o } 1 0 ^ { - 9 }$ , well within double-precision finite-difference truncation tolerances. Origin-anchor identities $\widehat { \phi } _ { 0 } ( 0 ) = 1 , \widehat { \phi } _ { k } ( 0 ) = m _ { k }$ , and normal-coordinate independence $\partial _ { \theta _ { k } } f _ { k } = 0$ hold to machine precision (10<sup>−12</sup> to 10<sup>−10</sup>).

Table B.2: Finite-difference Jacobian verification checks at initialization.
<table><tr><td>Benchmark</td><td>Checked object</td><td>Tolerance</td><td>Maximum error</td></tr><tr><td>2D product</td><td>Coordinate block</td><td> $2 \times 1 0 ^ { - 7 }$ </td><td> $1 . 1 3 1 \times 1 0 ^ { - 9 }$ </td></tr><tr><td>2D Harrison</td><td>Coordinate block</td><td> $2 \times 1 0 ^ { - 7 }$ </td><td> $2 . 9 9 \times 1 0 ^ { - 9 }$ </td></tr><tr><td>2D Harrison</td><td>Interaction block</td><td> $2 \times 1 0 ^ { - 7 }$ </td><td> $3 . 5 7 7 \times 1 0 ^ { - 1 0 }$ </td></tr><tr><td>10D non-product</td><td>Full log-transform vector </td><td> $1 \times 1 0 ^ { - 5 }$ </td><td> $1 . 8 1 8 \times 1 0 ^ { - 1 0 }$ </td></tr></table>

All checks use centered finite differences with step $h = 1 0 ^ { - 6 }$ and report the maximum absolute discrepancy across all tested complex probes and derivative entries.

## C Evaluation and Inversion Diagnostics

## C.1 Evaluation implementation

Frozen endpoint weights supply ${ \widehat { G } } _ { S } ( s )$ in (9) to mpmath.invertlaplace with method="talbot" and an explicit degree $M \in \{ 5 , 8 \}$ . Although the caller initializes mpmath.mp.dps to 50 decimal digits in 2D and 10D and 100 decimal digits in 20D–50D, specifying an explicit degree M causes mpmath's FixedTalbot implementation to set its internal quadrature working precision from M automatically, while all neural network queries are evaluated in machine complex128.

At zero, exact origin normalization $\widehat { \phi _ { 0 } } ( 0 ) = 1$ gives the removable singularity limit

$$
\widehat { G } _ { S } ( 0 ) = - \sum _ { j = 1 } ^ { d } \partial _ { j } f _ { 0 } ( 0 ) .\tag{C.1}
$$

The exact product-form limit is $\begin{array} { r } { H _ { d } = \sum _ { j = 1 } ^ { d } 1 / j } \end{array}$

## C.2 Signed-error decomposition

Let $p = p _ { \mathrm { r e f } } ( t ) > 0$ be the reference tail probability, $p _ { M } ^ { \star } = T _ { M } [ G _ { S } ] ( t )$ the output obtained by inverting the exact transform, and $\widehat { p } _ { M }$ the corresponding learned-transform tail prediction. We define the signed relative

error components in percent:

$$
r _ { \mathrm { i n v } } = 1 0 0 \frac { p _ { M } ^ { \star } - p } { p } , \qquad r _ { \mathrm { n c t } } = 1 0 0 \frac { \widehat { p } _ { M } - p _ { M } ^ { \star } } { p } , \qquad r _ { \mathrm { t o t a l } } = r _ { \mathrm { i n v } } + r _ { \mathrm { n c t } } = 1 0 0 \frac { \widehat { p } _ { M } - p } { p } .\tag{C.2}
$$

Pointwise relative error is $| r _ { \mathrm { t o t a l } } |$ , and $\mathbf { M R E } _ { 1 : K }$ averages this quantity over $t \in \{ 1 , \ldots , K \}$ . Taking absolute values before summing the components would discard sign cancellation, where $r _ { \mathrm { n e t } }$ measures the net effect of substituting the learned transform into the inversion pipeline.

Table C.1: Signed relative-error decomposition at 20D (t = 8, percent).
<table><tr><td>Model</td><td>M</td><td>t</td><td> $r _ { \mathrm { i n v } }$ </td><td> $r _ { \mathrm { n e t } }$ </td><td>rtotal</td></tr><tr><td>NN4MGF</td><td>5</td><td>8</td><td>-2.516</td><td>+18.374</td><td>+15.858</td></tr><tr><td>HCN (32345)</td><td>5</td><td>8</td><td>-2.516</td><td>-16.412</td><td>-18.928</td></tr><tr><td>NN4MGF</td><td>8</td><td>8</td><td>-0.447</td><td>-28.582</td><td>-29.028</td></tr><tr><td>HCN (32345)</td><td>8</td><td>8</td><td>-0.447</td><td>-8.037</td><td>-8.484</td></tr></table>

Components are computed at full precision and rounded only for display, satisfying $r _ { \mathrm { t o t a l } } = r _ { \mathrm { i n v } } + r _ { \mathrm { n e t } }$

Table C.1 compares NN4MGF and HCN at the same tail threshold $t = 8$ in 20 dimensions under Degrees 5 and 8. Under Degree $5 \ ( M = 5 )$ , the intrinsic quadrature error $r _ { \mathrm { i n v } } = - 2 . 5 1 6 \%$ has the opposite sign to NN4MGF’s network error $( r _ { \mathrm { n e t } } = + 1 8 . 3 7 4 \% )$ , partially canceling to yield a smaller total error (+15.858%). By contrast, HCN (seed 32345) has a smaller network error magnitude $( | r _ { \mathrm { n e t } } | = 1 6 . 4 1 2 \% )$ , but because its negative sign matches $r _ { \mathrm { i n v } } = - 2 . 5 1 6 \%$ , the two components reinforce to give −18.928% (and similarly −23.220% for seed 22345). Switching to Degree 8 $( M = 8 )$ at the exact same threshold t = 8 reduces the intrinsic inversion error magnitude to $r _ { \mathrm { i n v } } = - 0 . 4 4 7 \%$ , removing the artificial Degree-5 cancellation and revealing HCN’s substantially lower total error (−8.484% for seed 32345 and −14.963% for seed 22345, compared with −29.028% for NN4MGF). For a linear quadrature rule $\begin{array} { r } { T _ { M } [ G ] = \mathbf { R e } \sum _ { j } w _ { j } G ( s _ { j } ) } \end{array}$ , exact arithmetic gives

$$
| T _ { M } [ \widehat { G } ] - T _ { M } [ G ] | \leq \sum _ { j } | w _ { j } | | \widehat { G } ( s _ { j } ) - G ( s _ { j } ) | ,\tag{C.3}
$$

showing how quadrature weights $\left| w _ { j } \right|$ and node locations $s _ { j }$ govern the propagation of transform approximation errors across degrees.

## D Supplementary Experimental Results

## D.1 Degree sensitivity at fixed endpoints

Table D.1 pairs Degrees 5 and 8 at the fixed training endpoints across the 20D, 30D, and 50D product-form benchmarks. Within each pair, only the inversion degree changes; the trained network weights are identical. Moving from Degree 5 to Degree 8—which removes the large high-dimensional quadrature bias documented in Table 4—reduces MRE across all three evaluation intervals (1:5, 1:8, and 1:10) for every HCN benchmark: the 20D HCN three-seed mean decreases from 2.983% to 1.441% on 1:5, from 7.730% to 7.632% on 1:8, and from 19.347% to 12.264% on 1:10; the 30D HCN errors decrease from 3.429%, 10.107%, and 19.084% to 1.683%, 6.546%, and 8.872%; and the 50D HCN errors decrease from 4.605%, 7.196%, and 13.182% to 1.495%, 6.367%, and 8.011%. By contrast, for the archived 20D NN4MGF checkpoint, removing the Degree-5 sign cancellation increases $\mathrm { M R E } _ { 1 : 8 }$ from 8.006% to 8.148% and increases $\mathrm { M R E } _ { 1 : 1 0 }$ from 16.801% to 18.908%.

Table D.1: Fixed-endpoint MREs under Degrees 5 and 8 (percent).
<table><tr><td>Model</td><td>M</td><td>1:5</td><td>1:8</td><td>1:10</td></tr><tr><td>20D HCN (3-seed mean)</td><td>5</td><td>2.983</td><td>7.730</td><td>19.347</td></tr><tr><td>20D HCN (3-seed mean)</td><td>8</td><td>1.441</td><td>7.632</td><td>12.264</td></tr><tr><td>20D NN4MGF</td><td>5</td><td>4.013</td><td>8.006</td><td>16.801</td></tr><tr><td>20D NN4MGF</td><td>8</td><td>2.802</td><td>8.148</td><td>18.908</td></tr><tr><td>30D HCN</td><td>5</td><td>3.429</td><td>10.107</td><td>19.084</td></tr><tr><td>30D HCN</td><td>8</td><td>1.683</td><td>6.546</td><td>8.872</td></tr><tr><td>50D HCN</td><td>5</td><td>4.605</td><td>7.196</td><td>13.182</td></tr><tr><td>50D HCN</td><td>8</td><td>1.495</td><td>6.367</td><td>8.011</td></tr></table>

Each degree pair evaluates the exact same trained network weights without retraining. The 20D HCN entries report the mean across the three trained seeds.

## D.2 Late-training diagnostics in 50 dimensions

Table D.2 supplements Section 5.5 by reporting $\mathrm { M R E } _ { 1 : 5 }$ and $\mathrm { M R E } _ { 1 : 8 }$ across the five late-training checkpoints (92,000 to 100,000 updates) of the 50D trajectory at Degree 8. Across these five checkpoints, $\mathrm { M R E } _ { 1 : 5 }$ remains between 1.318% and 1.611%, and $\mathrm { M R E } _ { 1 : 8 }$ remains between 5.299% and 7.155% (6.367% at the fixed 100,000-update endpoint).

Table D.2: Late-training checkpoints along the 50D trajectory at Degree 8 (MRE, percent).
<table><tr><td>Update</td><td>1:5</td><td>1:8 Role</td><td></td></tr><tr><td>92,000</td><td>1.524</td><td>6.650</td><td>Diagnostic checkpoint</td></tr><tr><td>94,000</td><td>1.611</td><td>5.299</td><td>Diagnostic checkpoint</td></tr><tr><td>96,000</td><td>1.318</td><td>6.845</td><td>Diagnostic checkpoint</td></tr><tr><td>98,000</td><td>1.473</td><td>7.155</td><td>Diagnostic checkpoint</td></tr><tr><td>100,000</td><td>1.495</td><td>6.367</td><td>Reported endpoint</td></tr></table>

All five checkpoints from update 92,000 to 100,000 are evaluated under Degree 8. The main accuracy comparison uses the fixed 100,000-update endpoint.

## D.3 The exploratory 10D non-product example

In the 10-dimensional orthogonally reflected RBM (38), orthogonal reflection $( R = I _ { 1 0 } )$ preserves the onedimensional stationary marginal exponential laws despite correlated Brownian input:

$$
\mathbb { P } _ { \pi } ( Z _ { i } > x ) = e ^ { - i x } , \quad \mathbb { E } _ { \pi } [ Z _ { i } ] = 1 / i , \quad \phi _ { 0 } ( s u _ { i } ) = \frac { i } { i + s } , \quad \mathbb { E } _ { \pi } [ S ] = H _ { 1 0 } = 2 . 9 2 8 9 6 8 2 5 4 0 \ldots .\tag{D.1}
$$

Evaluating the removable-limit derivative formula (C.1) for the frozen HCN network gives a learned mean total workload of 2.9242988034, a 0.159% discrepancy relative to the exact analytical mean $H _ { 1 0 }$ . This checks the learned network’s first moment at the origin separately from the sum-tail comparison in Table D.3, where pointwise relative errors for both HCN and the independent-marginal comparator are reported against the pilot simulation estimates.

Table D.3: Ten-dimensional correlated non-product RBM at Degree 8.
<table><tr><td>t</td><td>Simulation ref.</td><td>HCN interaction</td><td>Independence</td><td>HCN RE (%)</td><td>Indep. RE (%)</td></tr><tr><td>1</td><td>0.972040</td><td>1.000000</td><td>0.989814</td><td>2.88</td><td>1.83</td></tr><tr><td>2</td><td>0.720910</td><td>0.722273</td><td>0.766398</td><td>0.19</td><td>6.31</td></tr><tr><td>3</td><td>0.400300</td><td>0.398938</td><td>0.399920</td><td>0.34</td><td>0.10</td></tr><tr><td>4</td><td>0.191240</td><td>0.187112</td><td>0.168775</td><td>2.16</td><td>11.75</td></tr><tr><td>5</td><td>0.084430</td><td>0.086949</td><td>0.065373</td><td>2.98</td><td>22.57</td></tr><tr><td>6</td><td>0.035760</td><td>0.039650</td><td>0.024513</td><td>10.88</td><td>31.45</td></tr><tr><td>7</td><td>0.014645</td><td>0.015140</td><td>0.009081</td><td>3.38</td><td>37.99</td></tr><tr><td>8</td><td>0.005905</td><td>0.005512</td><td>0.003350</td><td>6.66</td><td>43.28</td></tr><tr><td>Mean (1:8)</td><td></td><td></td><td></td><td>3.68</td><td>19.41</td></tr></table>

HCN and independent-marginal comparator MREs relative to the pilot steady-state simulation reference are 3.684% and 19.409%, respectively. First-moment cross-validation against the exact analytical mean $\mathbb { E } [ S ] = H _ { 1 0 }$ agrees to within 0.159%.

## E Reproducibility and Computational Accounting

## E.1 Experimental records and provenance

In the companion reproducibility package linked below, archived evaluation records in data/ and audit/ store the complete pointwise predictions, hardware profiles, configurations, and SHA-256 checksums for every benchmark, seed, checkpoint, and Talbot degree reported in the paper. Baseline NN4MGF comparisons use the pointwise 2D and 30D tail predictions released in the official repository of Dai and Zhang (2026) (https: $/ \operatorname { \nabla } / { \mathfrak { g } } .$ ithub.com/zhangz73/NN4MGF) and full-precision evaluations of the released 20D NN4MGF checkpoint. In the 30D product-form experiment, training proceeds for 50,000 updates with learning-rate annealing at update 40,000 under the audited 64M base-input budget.

## E.2 Software environment

Training and evaluation runs on the workstation/laptop platform use Python 3.10.20, PyTorch 2.5.1+cu121, CUDA 12.1, and mpmath 1.3.0 on an NVIDIA GeForce RTX 4080 Laptop GPU, while the controlled 30-dimensional hardware resource profiles in Section 5.2 are executed on a common NVIDIA H100 NVL GPU. Complete environment metadata and preflight verification logs are archived in provenance/ implementation\_details.json and provenance/implementation\_checks/.

## E.3 Resource and sampling-budget accounting

In the 30D H100 profile (Table 2), given measured per-update times $\tau _ { j }$ over 200 post-warmup updates, the mean update time is $\begin{array} { r } { \overline { { \tau } } = 2 0 0 ^ { - 1 } \sum _ { j = 1 } ^ { 2 0 0 } \tau _ { j } } \end{array}$ (0.165113 s for HCN and 1.601046 s for NN4MGF, totaling 33.023 s and 320.209 s over 200 updates), and the projected full-schedule update time is $\widehat { T } = U \overline { { \tau } }$ . Peak allocated and peak reserved GPU memory (in $\mathrm { G i B } = 2 ^ { 3 0 }$ bytes) are recorded directly from CUDA memory allocator statistics reset immediately after the 10 warmup updates.

Under the base-input definition (32), HCN processes $1 , 0 2 4 + 1 2 8 + 1 2 8 = 1 , 2 8 0$ base inputs per update over U = 50,000 updates, whereas the released NN4MGF training code samples two batches of 16,384 inputs

per update over U = 300,000 updates:

$$
N _ { \mathrm { H C N } } = 5 0 , 0 0 0 \times \left( 1 , 0 2 4 + 1 2 8 + 1 2 8 \right) = 6 4 , 0 0 0 , 0 0 0 ,\tag{E.1}
$$

$$
N _ { \mathrm { N N 4 M G F } } = 3 0 0 , 0 0 0 \times ( 2 \times 1 6 , 3 8 4 ) = 9 , 8 3 0 , 4 0 0 , 0 0 0 .\tag{E.2}
$$

Thus HCN’s 64M cumulative base-input budget represents 0.651% of the implemented NN4MGF code-path budget, and 1.302% of the single-batch count $( 3 0 0 , 0 0 0 \times 1 6 , 3 8 4 = 4 . 9 1 5 2 8 )$ cited in the text of Dai and Zhang (2026).

## E.4 Code availability and reproduction scripts

The source code for Holomorphic Cauchy Networks—including the complex Cauchy network modules, exact origin-anchoring and boundary-exclusion layers, closed-form complex Jacobians, sequential backpropagation training loops, and Fixed-Talbot evaluation routines—is available at https://github.com/ KehanWang126/holomorphic-cauchy-networks-rbm. In addition, the self-contained reproduction scripts listed in Table E.1 verify all SHA-256 data checksums and regenerate every table, figure, and compiled PDF in the main text and appendices directly from the archived evaluation records.

Table E.1: Entry points for verifying records and rebuilding all manuscript tables, figures, and PDFs.
<table><tr><td>Task</td><td>Python entry point</td></tr><tr><td>Verify SHA-256 hashes and summaries</td><td>audit/audit_results.py</td></tr><tr><td>Rebuild all main and supplement ta-</td><td>build_results.py,then</td></tr><tr><td>bles/figures</td><td>scripts/build_appendix_tables.py</td></tr><tr><td>Compile main and supplement PDFs</td><td>scripts/build_pdfs.py</td></tr></table>