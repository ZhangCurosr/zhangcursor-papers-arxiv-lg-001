# FLASHKAN: B-SPLINE KANS VIA TRUNCATED POWER FORM

Naveen Mysore

nmysore.work@gmail.com

## ABSTRACT

Kolmogorov-Arnold Networks (KANs) place learnable B-spline activations on network edges rather than fixed activations on nodes. The standard Cox-de Boor recursion evaluates these activations through k sequential passes for degree-k splines, consuming over 90% of forward-pass time.

FlashKAN replaces this recursion with the truncated power form, a classical result from approximation theory that expresses each uniform cubic B-spline as five (x)<sup>3</sup> terms at shifted knot positions. This paper makes three contributions: (1) a torch.compile-fused implementation that collapses these operations into a single GPU kernel, eliminating all recursion, span lookup, and scatter-gather operations; (2) a bounded-coordinate stabilization that clamps the normalized input to [0, k+1], preventing the catastrophic cancellation that historically motivated the Cox-de Boor recursion; and (3) a production-ready, open-source package (pip install flashkan) that serves as a drop-in replacement for existing KAN layers.

## 1 INTRODUCTION

The Kolmogorov-Arnold representation theorem (Kolmogorov, 1957) guarantees that any multivariate continuous function decomposes into univariate functions and addition. KANs (Liu et al., 2024) operationalize this result by placing a learnable activation on every edge. Each activation is a weighted sum of B-spline basis functions, which can be inspected, plotted, and in favorable cases symbolically recovered.

This interpretability carries a cost. The Cox-de Boor recursion (Cox, 1972; de Boor, 1971) evaluates B-spline basis functions through k sequential passes for degree k. For cubic splines (k=3), three passes are needed, each depending on the output of the previous. Profiling shows that basi computation alone accounts for 91% of a KAN layer’s forward-pass time (Table 1).

The recursion itself was not always the standard. In the 1940s, Schoenberg introduced splines as piecewise polynomials while smoothing ballistics data at Aberdeen Proving Ground (Schoenberg, 1946). His representation used truncated power functions max(0, x)<sup>k</sup> as the foundational building block, and Curry and Schoenberg formalized B-splines through this basis (Curry & Schoenberg, 1947). The truncated power form is mathematically elegant: proofs of compact support, partition of unity, and smoothness follow directly. But early implementations on 1960s hardware suffered from numerical cancellation in the alternating sum of large powers, and the truncated power basis matrix is poorly conditioned (de Boor, 2001; Schumaker, 2007). Cox and de Boor independently developed the recursive algorithm (Cox, 1972; de Boor, 1971) as a numerically stable alternative, and de Boor’s textbook (de Boor, 2001) established the recursion as the standard for the next five decades.

Prior work has pursued three strategies to reduce the cost of B-spline evaluation in KANs. Restructuring the Cox-de Boor recursion (Blealtan, 2024) reduces constant factors but preserves the sequential dependency. Precomputing per-span matrices (Coffman & Chen, 2025) eliminates the recursion but introduces data-dependent gather operations that fragment GPU memory access. Gaussian radial basis functions (Li, 2024) replace B-splines entirely with a single exp() call, gaining speed but sacrificing compact support and automatic partition of unity (Gaussians are non-negative and C<sup>∞</sup>, so smoothness is not the trade-off).

Most recently, Southworth et al. (2026) established a formal algebraic equivalence between spline KAN layers and multichannel MLPs with power-ReLU activations through a change-of-basis matrix. For uniform knots, this matrix is Toeplitz with entries matching the classical truncated power coefficients. Their work exploits this equivalence primarily for multilevel training: the spline basis enables complementary relaxation dynamics across refinement levels, yielding orders-of-magnitude improvements in PINN accuracy. They note that the non-recursive formulation is faster by a factor equal to the spline degree, but do not pursue compiler-level optimization, numerical stabilization, or a deployable implementation.

FlashKAN bridges the gap between algebraic equivalence and practical deployment. The contributions are:

1. Compiler-fused evaluation. For uniform cubic B-splines, the truncated power form reduces to five $\operatorname* { m a x } ( 0 , \cdot ) ^ { 3 }$ terms with fixed coefficients. torch.compile fuses all elementwise operations into a single GPU kernel, eliminating recursion, span lookup, and data-dependent memory access.

2. Bounded-coordinate stabilization. The raw truncated power sum suffers from catastrophic cancellation when the normalized input u lies far outside the basis support $[ 0 , k { + } 1 ]$ Clamping u to this interval before evaluation bounds all intermediate terms and eliminates the cancellation, with no change to the mathematically correct output (since the B-spline is exactly zero outside its support). This addresses the numerical concern that historically motivated the Cox-de Boor recursion (de Boor, 2001).

3. Drop-in package. FlashKAN is available as an open-source Python package (pip install flashkan, MIT license) with identical API to standard KAN layers. Replacing Cox-de Boor evaluation requires changing one import.

## 2 FROM RECURSION TO CLOSED FORM

The derivation proceeds in four steps. Each transforms an algorithm into an expression, revealing structure that the algorithm conceals.

## 2.1 STEP 1: DE CASTELJAU TO BERNSTEIN

Let $P _ { 0 } , P _ { 1 } , \ldots , P _ { n } \in \mathbb { R } ^ { d }$ be a sequence of control points and let $t \in [ 0 , 1 ]$ be a scalar parameter. The linear interpolation (lerp) between two points is defined as le $\mathrm { r p } ( \boldsymbol { \dot { A } } , \boldsymbol { { \dot { B , t } } } ) = ( 1 - t \bar { ) } A + t B$ De Casteljau’s algorithm (Farin, 2002) evaluates a degree-n Bezier curve´ $Q ( t )$ by applying lerp recursively: at each level, adjacent points are interpolated pairwise, reducing the number of points by one until a single value remains.

For the cubic case (n=3), three levels of interpolation over $P _ { 0 } , P _ { 1 } , P _ { 2 } , P _ { 3 }$ produce the curve point Q(t). Expanding and collecting terms on each control point yields the Bernstein form:

$$
Q ( t ) = \sum _ { i = 0 } ^ { 3 } B _ { i , 3 } ( t ) P _ { i } , \quad B _ { i , 3 } ( t ) = { \binom { 3 } { i } } t ^ { i } ( 1 { - } t ) ^ { 3 - i }\tag{1}
$$

The coefficients $\binom { 3 } { i } = \{ 1 , 3 , 3 , 1 \}$ are row 3 of Pascal’s triangle. Each interpolation level multiplies by $[ ( 1 - t ) + t ] = 1$ . Three levels produce $( ( 1 - t ) + t ) ^ { 3 }$ , which the binomial theorem expands into the Bernstein basis. The partition of unity $\textstyle \sum _ { i = 0 } ^ { 3 } B _ { i , 3 } ( t ) = 1$ follows immediately.

## 2.2 STEP 2: MATRIX FORM

Expanding each Bernstein polynomial into powers of t separates the Bezier curve into three factors:´

$$
Q ( t ) = \underbrace { \left[ 1 \quad t \quad t ^ { 2 } \quad t ^ { 3 } \right] } _ { \mathbf { T } ( t ) } \underbrace { \left[ { \begin{array} { l l l l } { 1 } & { 0 } & { 0 } & { 0 } \\ { - 3 } & { 3 } & { 0 } & { 0 } \\ { 3 } & { - 6 } & { 3 } & { 0 } \\ { - 1 } & { 3 } & { - 3 } & { 1 } \end{array} } \right] } _ { \mathbf { M } _ { \mathrm { B e z i e r } } } \underbrace { \left[ P _ { 0 } \right] } _ { P _ { 1 } }\tag{2}
$$

Table 1: Forward-pass cost breakdown for a KAN layer (256×784→64, MPS GPU). Basis computation dominates.
<table><tr><td>Component</td><td>Time (ms)</td><td>Share</td></tr><tr><td>B-spline basis (Cox-de Boor, 3 passes)</td><td>1.44</td><td>91%</td></tr><tr><td>Spline einsum (bases × weights)</td><td>0.09</td><td>6%</td></tr><tr><td>Base function (SiLU + einsum)</td><td>0.05</td><td>3%</td></tr><tr><td>Total forward</td><td>1.58</td><td>100%</td></tr></table>

The power vector $\mathbf T ( t )$ depends on the input. The control points P are free parameters. The basis matrix $\mathbf { M } _ { \mathrm { B e z i e r } }$ is a constant determined entirely by the degree: it encodes the binomial expansion of the Bernstein polynomials. This factorization T(t) M P is general. Different spline types share the same structure but differ only in M. The input-dependent computation is always $\mathbf { T } ( t ) \stackrel {  } { = } [ 1 , t , t ^ { 2 } , t ^ { 3 } ]$ four values, trivially cheap.

## 2.3 STEP 3: BEZIER SEGMENTS TO´ B-SPLINES

A single cubic Bezier has four control points and one segment. It also has a fundamental limitation:´ global control. Moving any control point affects the entire curve. To model a function over a wide input range, multiple Bezier segments must be chained end to end, and at every joint three constraints´ must be enforced to ensure smoothness: $C ^ { 0 }$ continuity (the segments meet), $C ^ { 1 }$ continuity (their tangents match), and $C ^ { 2 }$ continuity (their curvatures agree). For two cubic segments with eight control points, these six constraints (three per joint, one joint) consume three degrees of freedom per segment, leaving only five free parameters out of eight. The constraint cost grows linearly with the number of segments: each new joint introduces three equations that lock three control points to their neighbors.

B-splines solve this problem by construction. Instead of stitching independent segments and imposing constraints after the fact, B-splines define basis functions that inherently satisfy smoothness across the full domain. Given a non-decreasing knot vector $\mathbf { t } ~ = ~ ( t _ { 0 } , t _ { 1 } , \ldots , t _ { m } )$ that partitions the parameter space into spans, the Cox-de Boor recursion (de Boor, 2001) defines basis functions $N _ { i , k } ( x )$ of degree k with automatic $C ^ { k - 1 }$ continuity at every interior knot:

$$
N _ { i , 0 } ( x ) = { \left\{ \begin{array} { l l } { 1 } & { { \mathrm { i f ~ } } t _ { i } \leq x < t _ { i + 1 } } \\ { 0 } & { { \mathrm { o t h e r w i s e } } } \end{array} \right. }\tag{3}
$$

$$
N _ { i , k } ( x ) = \frac { x - t _ { i } } { t _ { i + k } - t _ { i } } N _ { i , k - 1 } ( x ) + \frac { t _ { i + k + 1 } - x } { t _ { i + k + 1 } - t _ { i + 1 } } N _ { i + 1 , k - 1 } ( x )\tag{4}
$$

Each basis function has compact support: $N _ { i , k } ( x )$ is nonzero only over $k { + 1 }$ consecutive knot spans.   
Moving one control point changes the curve only within that local region. No constraints to manage.   
No degrees of freedom lost. Every control point is free.

For a uniform knot vector with spacing $h ,$ each B-spline segment admits the same matrix factorization as Bezier, but with a different basis matrix:´

$$
\mathbf { M } _ { \mathrm { B - s p l i n e } } = { \frac { 1 } { 6 } } \left[ { \begin{array} { c c c c } { 1 } & { 4 } & { 1 } & { 0 } \\ { - 3 } & { 0 } & { 3 } & { 0 } \\ { 3 } & { - 6 } & { 3 } & { 0 } \\ { - 1 } & { 3 } & { - 3 } & { 1 } \end{array} } \right]\tag{5}
$$

The structure $\mathbf T ( t )$ M P is identical; only the coefficients in M change, encoding the smoothness constraints that B-splines satisfy by construction.

The recursion mirrors De Casteljau: same linear interpolations, but over knot intervals rather than [0, 1]. For k=3, three sequential passes are needed. Each depends on the previous. Table 1 shows these passes consume 91% of a KAN layer’s forward time.

## 2.4 STEP 4: COX-DE BOOR TO TRUNCATED POWER FORM

The same algebraic move that transformed De Casteljau into Bernstein polynomials applies to Coxde Boor. For a uniform knot vector with constant spacing $h = t _ { i + 1 } - t _ { i }$ , expanding the recursion yields a finite difference of truncated power functions (Curry & Schoenberg, 1947; Schoenberg, 1946):

$$
N _ { i , k } ( x ) = { \frac { 1 } { k ! h ^ { k } } } \sum _ { j = 0 } ^ { k + 1 } ( - 1 ) ^ { j } { \binom { k + 1 } { j } } \operatorname* { m a x } ( 0 , x - t _ { i + j } ) ^ { k }\tag{6}
$$

For cubic splines (k=3), substituting $\boldsymbol { u } = ( x - g _ { i } ) / h$ produces:

$$
N _ { i } ( u ) = \frac { 1 } { 6 } \Big [ ( u ) _ { + } ^ { 3 } - 4 ( u - 1 ) _ { + } ^ { 3 } + 6 ( u - 2 ) _ { + } ^ { 3 } - 4 ( u - 3 ) _ { + } ^ { 3 } + ( u - 4 ) _ { + } ^ { 3 } \Big ]\tag{7}
$$

where $g _ { i }$ is the start of the i-th support interval and $( z ) _ { + } ~ = ~ \operatorname* { m a x } ( 0 , z )$ The coefficients $\{ 1 , - 4 , { \overset { - } { 6 } } , - 4 , 1 \}$ are the fourth row of Pascal’s triangle with alternating signs, divided by $3 ! = 6 $ This is the same coefficient stencil identified by Southworth et al. (2026) through their change-ofbasis analysis.

The parallel between the two derivations is exact. De Casteljau is to Bernstein as Cox-de Boor is to the truncated power form. In both cases, an algorithm built from nested interpolations admits a closed-form expression built from binomial coefficients. The algorithm is sequential. The expression is parallel.

Computational properties. Eq. 7 inherits the matrix form’s separation. The input-dependent part is $u = ( x - g _ { i } ) / h \colon$ one subtraction and one multiply. The basis matrix M is absorbed into the fixed coefficients $\{ 1 , - 4 , 6 , - 4 , 1 \} / 6$ . The control points are the learnable weights. No sequential dependency, no data-dependent memory access, only elementwise operations, all of which torch.compile fuses into a single kernel.

## 2.5 BOUNDED-COORDINATE STABILIZATION

The truncated power sum in Eq. 7 is exact in infinite precision. In finite-precision arithmetic, however, the alternating sum of five terms of size $\Theta ( u ^ { 3 } )$ suffers catastrophic cancellation when u lies far outside the support interval [0, 4]. The mathematically correct value is exactly zero for u $\geq$ 4 or $u \leq 0$ , but floating-point evaluation produces a residual that grows as $O ( \epsilon _ { \mathrm { m a c h } } \cdot u ^ { 3 } )$ . This concern was precisely what motivated Cox and de Boor to develop the recursive algorithm (Cox, 1972; de Boor, 1971; 2001).

For KAN layers, the risk is concrete. Hidden-layer activations can drift outside the nominal grid domain during training, producing large u values even when the original inputs are normalized. In mixed precision $( \mathrm { \ f \ o a t i 6 } ) , u ^ { 3 }$ overflows once u exceeds approximately 40.

The fix is simple. Before evaluating Eq. 7, clamp the normalized coordinate to the support:

$$
\bar { u } = \mathrm { c l a m p } ( u , 0 , k { + } 1 )\tag{8}
$$

and compute $N _ { i } ( \bar { u } )$ instead of $N _ { i } ( u )$ . This is algebraically correct because $N _ { i } ( u ) = 0$ for $u \leq 0$ and u $\geq k { + } 1$ , so clamping does not change any in-support value. But it bounds every intermediate term: the largest cubed quantity is at most $( k { + } 1 ) ^ { 3 } = 6 4$ for cubic splines, well within float16 range. The numerical error no longer grows with distance from the support.

The clamped evaluation adds one clamp operation per basis function, which torch.compile folds into the same fused kernel at no measurable cost.

## 2.6 INTEGRATION WITH KAN LAYERS

A KAN layer computes:

$$
y _ { o } = \sum _ { i } \left[ \sum _ { k } c _ { o , i , k } N _ { k } ( x _ { i } ) + w _ { o , i } \sigma ( x _ { i } ) \right]\tag{9}
$$

where $c _ { o , i , k }$ are learnable spline coefficients, $w _ { o , i }$ are residual weights, and $\sigma$ is SiLU. Replacing the Cox-de Boor evaluation of $N _ { k } ( x _ { i } )$ with the bounded truncated power form (Eq. 7 with Eq. 8) changes only the basis computation; weights, residual connections, and gradient flow remain identical.

## 3 RELATED WORK

KAN implementations. The original KAN (Liu et al., 2024) evaluates B-spline bases via the Coxde Boor recursion. Efficient-KAN (Blealtan, 2024) restructures the computation into dense tensor operations but preserves the recursive dependency. FastKAN (Li, 2024) replaces B-splines entirely with Gaussian radial basis functions $R _ { j } ( \boldsymbol { x } ) = \mathrm { e x p } [ - \gamma ( \boldsymbol { x } - \boldsymbol { c } _ { j } ) ^ { 2 } ]$ . Gaussians are non-negative and $C ^ { \infty }$ smooth, so the trade-off is not smoothness or sign but rather the loss of exact compact support (each Gaussian has global, though rapidly decaying, influence) and automatic partition of unity $\begin{array} { r } { ( \sum _ { j } R _ { j } ( x ) = 1 } \end{array}$ does not hold without explicit normalization).

Truncated power basis and change of basis. The truncated power representation of B-splines dates to Curry and Schoenberg (Curry & Schoenberg, 1947) and is presented in standard references (Schumaker, 2007; de Boor, 2001). Southworth et al. (2026) recently applied this equivalence to KANs, proving that spline KAN layers correspond to multichannel MLPs with power-ReLU activations through a linear change-of-basis matrix that, for uniform knots, is Toeplitz. Their primary contribution is a multilevel training framework that exploits the spline basis for complementary relaxation dynamics across refinement levels, yielding orders-of-magnitude improvements in PINN accuracy. FlashKAN builds on the same algebraic identity but pursues a different goal: compiler-level optimization for inference and training speed, numerical stabilization through bounded-coordinate evaluation (Section 2.5), and a production-ready package. The two contributions are complementary: FlashKAN could serve as the fast evaluation engine within a multilevel training loop.

Numerical stability of the truncated power form. De Boor (de Boor, 2001) demonstrated that the truncated power basis matrix is poorly conditioned relative to the B-spline basis, motivating the recursive algorithm. This conditioning concern applies when the alternating sum is evaluated far from the basis support, where terms of size $\Theta ( u ^ { k } )$ cancel to produce zero. The bounded-coordinate stabilization (Section 2.5) eliminates this regime entirely, confining all intermediate values to a bounded range regardless of the input.

## 4 DISCUSSION

Why bounded-coordinate evaluation matters. The numerical concerns that motivated de Boor apply whenever the normalized coordinate u takes large values. In a trained KAN, hidden-layer activations are not guaranteed to remain within the nominal grid domain. The bounded-coordinate variant (Section 2.5) is not merely defensive: it extends the truncated power form’s domain of reliable operation from inputs near the grid to arbitrary inputs, including out-of-distribution data and intermediate activations in deep networks. Without this stabilization, the truncated power form in herits the exact failure mode that led to its historical abandonment.

Comparison with Gaussian RBF. FastKAN (Li, 2024) introduced Gaussians to accelerate KANs by replacing B-splines with a simpler basis. Gaussians are non-negative and infinitely differentiable, so the sacrifice is not smoothness or sign. The properties lost are exact compact support (each Gaussian has global, though rapidly decaying, influence) and automatic partition of unity (the sum $\textstyle \sum _ { j } R _ { j } ( x )$ is not guaranteed to equal one without explicit normalization). The truncated power form achieves the same structural simplification (a single broadcastable expression) while preserving both properties.

## 5 LIMITATIONS

The closed form in Eq. 6 applies only to uniform knot vectors. Non-uniform grids, which allow adaptive refinement based on input distribution, require per-span polynomial coefficients and reintroduce data-dependent indexing. The multilevel training framework of Southworth et al. (2026), which progressively refines the knot vector, is complementary: FlashKAN could serve as the fast evaluation engine at each refinement level, provided the grid remains uniform within each level. Extension to non-uniform grids is left for future work.

## 6 CONCLUSION

FlashKAN is the first KAN implementation to evaluate B-spline basis functions via the truncated power form. The mathematical identity dates to Curry and Schoenberg (Curry & Schoenberg, 1947), and Southworth et al. (2026) recently proved that spline KAN layers are algebraically equivalent to power-ReLU MLPs through this basis. FlashKAN turns that equivalence into a deployable system: bounded-coordinate stabilization eliminates the numerical cancellation that historically discouraged the truncated power form, torch.compile fuses the evaluation into a single GPU kernel, and the result is packaged as a drop-in replacement. FlashKAN is open-source (pip install flashkan, MIT license) and includes 26 unit tests covering basis correctness, partition of unity, compact support, C<sup>2</sup> smoothness, and gradient flow.

## REFERENCES

Blealtan. An efficient implementation of Kolmogorov-Arnold network, 2024. URL https:// github.com/Blealtan/efficient-kan.

Cale Coffman and Lizhong Chen. MatrixKAN: Parallelized Kolmogorov-Arnold network. arXiv preprint arXiv:2502.07176, 2025.

Maurice G Cox. The numerical evaluation of B-splines. Journal ofthe Institute ofMathematics and its Applications, 10(2):134–149, 1972.

Haskell B Curry and Isaac J Schoenberg. On spline distributions and their limits: The Polya dis-´ tribution functions. Bulletin of the American Mathematical Society, 53:1114, 1947. Abstract 380t.

Carl de Boor. Subroutine package for calculating with B-splines. Technical Report LA-4728-MS, Los Alamos Scientific Laboratory, 1971.

Carl de Boor. A Practical Guide to Splines. Springer-Verlag, New York, revised edition, 2001.

Gerald Farin. Curves and Surfacesfor CAGD: A Practical Guide. Morgan Kaufmann, 5th edition, 2002.

Andrey N Kolmogorov. On the representation of continuous functions of many variables by superposition of continuous functions of one variable and addition. Doklady Akademii Nauk SSSR, 114 (5):953–956, 1957.

Ziyao Li. Kolmogorov-Arnold networks are radial basis function networks. arXiv preprint arXiv:2405.06721, 2024.

Ziming Liu, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljaciˇ c,´ Thomas Y Hou, and Max Tegmark. Kan: Kolmogorov-arnold networks. arXiv preprint arXiv:2404.19756, 2024.

Isaac J Schoenberg. Contributions to the problem of approximation of equidistant data by analytic functions, Parts A and B. Quarterly ofApplied Mathematics, 4:45–99, 112–141, 1946.

Larry L Schumaker. Spline Functions: Basic Theory. Cambridge University Press, 3rd edition, 2007.

Ben S Southworth, Jonas A Actor, Graham Harper, and Eric C Cyr. Multilevel training for Kolmogorov-Arnold networks. arXiv preprint arXiv:2603.04827, 2026.