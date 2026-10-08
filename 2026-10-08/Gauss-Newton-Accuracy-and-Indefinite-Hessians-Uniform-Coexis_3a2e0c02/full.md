# Gauss–Newton Accuracy and Indefinite Hessians: Uniform Coexistence in Low-Cost Sets

Kihun Rhee rheekh00@snu.ac.kr

Hanjoon Byun byunhanjoon@snu.ac.kr

## Abstract

We study the accuracy of Gauss–Newton curvature in ridge-regularized nonlinear least squares. Under local regularity and persistence of level-set curvature magnitude along an exact-fit section, we prove uniform coexistence of two curvature regimes. Global minimizers exist, and every global minimizer has relative Hessian error below (1 + 2)/8, while the same low-cost set contains a point with an indefinite Hessian and relative error at least 15/8. One positive ridge cap works for all independent center and label perturbations in fixed neighborhoods and every positive ridge weight up to the cap. These neighborhoods do not shrink as the ridge weight tends to zero. A pointwise certificate based on the current prediction level set controls the normal, mixed, and tangent parts of the Hessian correction. We prove a sharp relative-error bound over the stated pointwise class when the prediction map and ridge vary. Analytic examples describe the roles of output alignment, curvature orientation, and persistence. A separate structural result gives full Jacobian row rank throughout low-cost sets and exact interpolation near a rank-deficient reference.

## 1 Introduction

Gauss–Newton approximates the curvature of a least-squares objective using the prediction Jacobian. It is a standard tool in nonlinear least squares [1, 10] and in Hessian-free optimization of neural networks [13, Section 4.2]. A small objective value controls the residual. It is therefore natural to ask whether it also makes the Gauss–Newton approximation accurate throughout a low-cost set.

The obstruction lies in directions that leave predictions unchanged to first order. The prediction-Jacobian contribution vanishes in these directions, so ridge regularization supplies all of the Gauss– Newton curvature. The full Hessian also contains a residual-weighted second-derivative term. This classical correction can be large relative to the ridge even when the residual is small [1, 10, 11]. The geometry of the current prediction level set describes its action in these directions.

Let $G : \mathbb { R } ^ { p }  \mathbb { R } ^ { n }$ be a smooth prediction map. For labels Y , ridge center $d ,$ and weight $\lambda > 0$ write

$$
\begin{array} { r } { Q ( { \boldsymbol { \theta } } ) = \frac 1 2 \| { \boldsymbol { G } } ( { \boldsymbol { \theta } } ) - { \boldsymbol { Y } } \| ^ { 2 } + \frac { \lambda } { 2 } \| { \boldsymbol { \theta } } - { \boldsymbol { d } } \| ^ { 2 } , \qquad { \boldsymbol { P } } = D { \boldsymbol { G } } ( { \boldsymbol { \theta } } ) ^ { T } D { \boldsymbol { G } } ( { \boldsymbol { \theta } } ) + \lambda I . } \end{array}
$$

With $H = \nabla ^ { 2 } Q ( \theta )$ , define the relative curvature error

$$
\rho ( \theta ) = \| P ^ { - 1 / 2 } ( H - P ) P ^ { - 1 / 2 } \| _ { 2 } = \operatorname* { s u p } _ { h \neq 0 } \frac { | h ^ { T } ( H - P ) h | } { h ^ { T } P h } .\tag{1}
$$

Thus small $\rho$ means that P and H assign similar curvature to every direction at the point under consideration. We study this comparison on sets of the form $\{ Q \leq C \lambda \varepsilon ^ { 2 } \}$

1. Uniform coexistence. Our main result is Theorem 1. Assume common local $C ^ { 2 , 1 }$ bounds and persistence of curvature magnitude along an exact-fit section. We fix a positive cap W before varying the center and labels. For every allowed independent pair and every $0 < \lambda \leq W$ 2 global minimizers exist and all satisfy $\rho < c _ { * } = ( 1 + \sqrt { 2 } ) / 8$ . The same low-cost set contains a point with an indefinite Hessian and $\rho \ge 1 5 / 8$ . The center and label neighborhoods remain fixed as $\lambda \to 0$ . The minimizers and indefinite point may vary with the objective.

2. A pointwise accuracy certificate. Theorem 2 and Corollary 3 use cost, rank, derivative bounds, and current level-set curvature to control the Hessian correction. The proof treats its normal, mixed, and tangent blocks together. The resulting constant $c _ { * }$ is a sharp supremum over the stated pointwise class when the map and ridge vary. A fixed two-output example explains which alignment information is lost by aggregate bounds.

3. Rank and interpolation near a singular reference. Section 7 gives derivative conditions under which every low-cost point has full Jacobian row rank and every allowed refitting problem has a low-cost exact interpolator. This structural result is separate: applying the coexistence theorem still requires its common regularity and curvature-persistence assumptions.

Earlier results give Hessian bounds near solution manifolds and nearby nonconvexity [4, 6, 5, 11]. Our conclusion places accurate global minimizers and an indefinite point in the same low-cost set for each member of an entire family of objectives. The common ridge cap and fixed perturbation neighborhoods make this a uniform statement.

We state the coexistence theorem first, then develop the pointwise certificate used in its proof. The later sections give analytic comparisons and the structural result. Appendices A–D contain the detailed proofs and further examples.

## 2 Objective and level-set geometry

For $G : \mathbb { R } ^ { p } \to \mathbb { R } ^ { n } , p \geq n$ , define the raw Euclidean objective

$$
\begin{array} { r } { Q ( \theta ) = \frac { 1 } { 2 } \left\| G ( \theta ) - Y \right\| ^ { 2 } + \frac { \lambda } { 2 } \left\| \theta - d \right\| ^ { 2 } . } \end{array}\tag{2}
$$

Write $r = G ( \theta ) - Y , J = D G ( \theta )$ and $b = \nabla Q ( \theta )$ . Its complete Hessian is

$$
H = J ^ { T } J + E + \lambda I , \qquad E = \sum _ { i } r _ { i } D ^ { 2 } G _ { i } ( \theta ) ,\tag{3}
$$

$$
b = J ^ { T } r + \lambda ( \theta - d ) .
$$

This is the residual-inclusive formula in Eriksson et al. [1], Theorem 4.1, (26)–(27).

At a full-row-rank point, let T = ker J and $N = T ^ { \perp }$ . Define

$$
\begin{array} { r } { \mathrm { I I } ( v , w ) = - J ^ { \dagger } D ^ { 2 } G [ v , w ] \quad ( v , w \in T ) , } \\ { K = \left\| D ^ { 2 } G | _ { T \times T } \right\| , \quad B = \left\| \mathrm { I I } \right\| . } \end{array}\tag{4}
$$

These are bilinear operator norms; set $K = B = 0 { \mathrm { ~ i f ~ } } T = \{ 0 \}$ . The tensor II is the extrinsic second fundamental form of the regular prediction level set $G ^ { - 1 } ( G ( \theta ) )$ in the fixed raw Euclidean parameter metric [2, Lemma 2.2.1]. That current level set equals the target fiber $G ^ { - 1 } ( Y )$ only at an exact fit. The matrix P supplies a pointwise relative quadratic-form comparison; neither this geometry nor the ridge objective is asserted invariant under arbitrary reparameterization. The same ambient-to-intrinsic curvature correction appears in Absil et al. [3, Theorem 1, (10)] and Masiha et al. [4, (8)].

We call any point with $Q \leq C \lambda \varepsilon ^ { 2 }$ a low-cost point. It need not be stationary or optimal.

For later statements, define the admissible cost cap

$$
C _ { * } ( s , M ) = \operatorname* { m i n } \{ s ^ { 4 } / ( 8 M ^ { 2 } ) , ~ s ^ { 2 } / ( 1 2 8 M ^ { 2 } ) \} , \qquad s , M > 0 .
$$

The constants s and M respectively control the smallest Jacobian singular value and the second derivative.

## 3 Uniform coexistence under center and label changes

We fix one ridge interval before varying the center and labels independently. Every resulting objective has accurate global minimizers and an indefinite-Hessian point in the same low-cost set. The theorem constructs an exact-fit section as the labels vary. Persistence of curvature magnitude along this section is an assumption; the relevant tangent directions may rotate.

Theorem 1 (Uniform coexistence). Let $p > n \geq 1 , 0 < \varepsilon \leq 1$ , and $s , M , r > 0 , \ L \geq 0$ . Let $G : \mathbb { R } ^ { p }  \mathbb { R } ^ { n }$ be continuous and $C ^ { 2 , 1 }$ on a neighborhood of ${ \overline { { B } } } ( x _ { 0 } , R )$ , where $R \geq r \varepsilon , \| D ^ { 2 } G \| \leq M$ , and $D ^ { 2 } G$ has Lipschitz constant at most L on that ball. Suppose $J _ { 0 } = D G ( x _ { 0 } )$ is onto, $\sigma = \sigma _ { n } ( J _ { 0 } ) \geq s \varepsilon$ and $B _ { 0 } = \| \mathrm { I I } _ { x _ { 0 } } \| > 0$ . Set $s _ { * } = s / 2$ and $0 < C \le C _ { * } ( s _ { * } , M )$ . Define

$$
\begin{array} { l } { { \delta = \operatorname* { m i n } \{ \sqrt { C } / 8 , \ s / ( 1 6 M ) \} , \qquad \alpha = \delta / 2 , } } \\ { { \beta = \operatorname* { m i n } \{ s \delta / 4 , \ s ^ { 2 } / ( 3 2 M ) , \ r s / 8 , \ 1 / 4 \} , } } \end{array}
$$

and assume $R \geq ( \alpha + \sqrt { 2 C } ) \varepsilon$ . For each $\| v \| \leq \beta \varepsilon ^ { 2 }$ , let $y ( v ) = x _ { 0 } + h ( v )$ be the normal-section exact fit with

$$
\begin{array} { c } { { h ( v ) \in ( \ker J _ { 0 } ) ^ { \perp } , \qquad G ( y ( v ) ) = G ( x _ { 0 } ) + v , } } \\ { { \lVert h ( v ) \rVert \le 2 \lVert v \rVert / \sigma . } } \end{array}
$$

The stated bounds guarantee existence and uniqueness among $h \in ( \ker J _ { 0 } ) ^ { \perp }$ with $\left\| h \right\| \leq 2 \left\| v \right\| / \sigma$ Assume that its curvature magnitude persists:

$$
B _ { 0 } / 2 \le B _ { y ( v ) } \le 2 B _ { 0 } \qquad ( \| v \| \le \beta \varepsilon ^ { 2 } ) .\tag{5}
$$

Put $m _ { 0 } = \sigma / 2 , b _ { 0 } = B _ { 0 } / 2 , D _ { 0 } = L m _ { 0 } + 8 M ^ { 2 }$ , and

$$
\begin{array} { c } { { \Lambda = \displaystyle \operatorname* { m i n } \{ m _ { 0 } ^ { 3 } b _ { 0 } / ( 1 2 8 M ) , m _ { 0 } ^ { 4 } b _ { 0 } ^ { 2 } / ( 6 4 D _ { 0 } ) , } } \\ { { m _ { 0 } ^ { 2 } / 6 4 , ~ C m _ { 0 } ^ { 2 } b _ { 0 } ^ { 2 } \varepsilon ^ { 2 } / 1 6 , } } \\ { { \sqrt { C } \varepsilon m _ { 0 } ^ { 2 } b _ { 0 } / 1 6 , R m _ { 0 } ^ { 2 } b _ { 0 } / 1 6 \} > 0 , } } \\ { { W = \displaystyle \operatorname* { m i n } \{ \Lambda , \varepsilon ^ { 2 } \} . } } \end{array}\tag{6}
$$

For every independent pair $d = x _ { 0 } + u , \| u \| \leq \alpha \varepsilon .$ , and $Y = G ( x _ { 0 } ) + v , \| v \| \leq \beta \varepsilon ^ { 2 }$ , and every $0 < \lambda \leq W$ , use the same Q in (2) and set

$$
\begin{array} { r } { \begin{array} { r l } & { \mathcal { L } _ { \lambda } = \{ x : Q ( x ) \leq C \lambda \varepsilon ^ { 2 } \} , } \\ & { A ( x ) = B _ { x } \left\| \nabla Q ( x ) \right\| / \lambda , } \\ & { \quad P _ { x } = D G ( x ) ^ { T } D G ( x ) + \lambda I . } \end{array} } \end{array}
$$

Then the following hold for this one objective and low-cost set.

1. Every $x \in { \mathcal { L } } _ { \lambda }$ lies in $\overline { { B } } ( x _ { 0 } , R )$ , has $\sigma _ { n } ( D G ( x ) ) \ge 2 9 s \varepsilon / 3 2 > s _ { * } \varepsilon$ , and satisfies the local smoothness premises of Theorem 2.

2. A global minimizer exists. Every global minimizer $z _ { \lambda }$ has $Q ( z _ { \lambda } ) \leq C \lambda \varepsilon ^ { 2 } / 1 2 8$ and $A ( z _ { \lambda } ) = 0$ 2 so the test passes even when $Q ( d ) > 0$

3. At every $x \in { \mathcal { L } } _ { \lambda }$ with $A ( x ) \leq 1 / 8$

$$
\begin{array} { c } { { H _ { x } \succeq \lambda I / 2 , \qquad c _ { * } = ( 1 + \sqrt { 2 } ) / 8 , } } \\ { { ( 1 - c _ { * } ) P _ { x } \prec H _ { x } \prec ( 1 + c _ { * } ) P _ { x } , \qquad \rho ( x ) < c _ { * } . } } \end{array}
$$

4. The same set contains $x _ { \lambda } \in B ( x _ { 0 } , R / 2 )$ with strict low cost and indefinite complete Hessian. At this point

$$
{ \frac { 2 1 } { 1 6 } } \leq A ( x _ { \lambda } ) \leq { \frac { 4 5 } { 1 6 } } ,
$$

$$
\frac { 7 \lambda } { 8 B _ { 0 } } \leq \| \nabla Q ( x _ { \lambda } ) \| \leq \frac { 9 \lambda } { 2 B _ { 0 } } ,
$$

$$
\frac { 3 B _ { 0 } } { 8 } \leq B _ { x _ { \lambda } } \leq \frac { 5 B _ { 0 } } { 2 } .
$$

Moreover, $\rho ( x _ { \lambda } ) \ge 1 5 / 8$ . The witness $x _ { \lambda }$ is nonstationary.

For the last bound, the proof gives a unit $t \in$ ker $D G ( x _ { \lambda } )$ with $t ^ { T } H _ { x _ { \lambda } } t \leq - 7 \lambda / 8$ , while $t ^ { T } P _ { x _ { \lambda } } t = \lambda$ Its relative Rayleigh quotient in (1) is at least $1 5 / 8$

Under Theorem 1, one W works for every allowed independent center and label pair and every $0 < \lambda \leq W$ . For each such objective, global minimizers exist. Any global minimizer $z _ { \lambda }$ and some indefinite point $x _ { \lambda }$ lie in the same low-cost set and satisfy

$$
\begin{array} { r } { \boxed { \rho ( z _ { \lambda } ) < c _ { * } , \qquad \rho ( x _ { \lambda } ) \geq 1 5 / 8 } . } \end{array}
$$

The points may vary with these choices. Values between the two regimes remain unclassified.

For each allowed $d , Y , \lambda ,$ the exact-fit anchor $y ( v )$ has low cost for the same $Q .$ . Thus every global minimizer lies in $\mathcal { L } _ { \lambda }$ . The ridge term confines that entire set near $x _ { 0 }$ , where the derivative bound preserves full row rank. The tangent identity (9) bounds the certificate’s absolute tangent correction and supplies the witness’s signed curvature pairing.

At the anchor, the actual residual is zero. The witness proof in Appendix B.1 chooses a unit tangent and a separate residual. Their pairing against the anchor’s second derivative is $- 2 \lambda$ . That residual is realized nearby for the same $d , Y , \lambda$ . The common ridge cap controls residual cost, displacement, and changes in tangent direction and curvature. The bounds on $A ( x _ { \lambda } )$ use gradient and curvature estimates tied to the same moving anchor. Multiplying the separate $B _ { \mathrm { 0 ^ { - r e l a t i v e } } }$ intervals weakens them. These bounds give no exact threshold for A.

Appendix B.2 gives a near-flat example with fixed radii.

The persistence premise compares curvature magnitudes along the exact-fit section. The construction selects a new tangent and residual direction at each label; it does not need a fixed orientation. This comparison permits center and label radii independent of $B _ { 0 }$ , while the ridge cap in (6) still depends on $B _ { 0 }$ . Proposition 10 illustrates the changing direction. Proposition 11 gives a stronger boundary for the stated smoothness bounds. For suficiently small $\varepsilon$ and $0 < \lambda \leq \varepsilon ^ { 4 } / ( 8 C )$ after an arbitrarily small normalized label shift, its entire nonempty low-cost set lies where G is locally afine. There $E = 0$ , so $H = P \succ 0$ and $\rho = 0$ at every point of that set. This rules out a uniform label-radius conclusion from smoothness alone; it does not prove that the exact persistence condition above is necessary.

## 4 A pointwise curvature certificate

The certificate depends on where the residual correction acts. On $T , \ J ^ { T } J$ vanishes, so the gate bounds $\| E _ { T T } \|$ by $\lambda / 4$ . On $N , J ^ { T } J$ has a gap of at least $s ^ { 2 } \varepsilon ^ { 2 }$ . This gap can absorb a correction larger than λ. The mixed block reduces the tangent margin through the Schur complement. The two caps on C protect the normal gap and keep this loss at most $\lambda / 3 2$ . The tangent identity (9) explains the $\lambda / B$ gradient tolerance for $B > 0$ . The test uses second-order information about G.

Theorem 2 (Pointwise certificate). Let G be $C ^ { 2 }$ on a neighborhood of the point in question. Fix $s , M > 0$ and

$$
0 < C \leq C _ { * } ( s , M ) : = \operatorname* { m i n } \{ s ^ { 4 } / ( 8 M ^ { 2 } ) , s ^ { 2 } / ( 1 2 8 M ^ { 2 } ) \} .
$$

Let $0 < \varepsilon \le 1 , 0 < \lambda \le \varepsilon ^ { 2 }$ . At any point with $Q \leq C \lambda \varepsilon ^ { 2 } , \sigma _ { n } ( J ) \geq s \varepsilon$ , and $\| D ^ { 2 } G \| \leq M$ , the condition

$$
\operatorname* { m i n } \{ K \sqrt { 2 C \lambda } \varepsilon , B ( \| b \| + \lambda \sqrt { 2 C } \varepsilon ) \} \leq \lambda / 4\tag{7}
$$

implies $H \succeq \lambda I / 2$ . Suficient alternatives are $B = 0 ; \| b \| \leq \lambda / ( 8 B ) \ i f B > 0 ; \| b \| \leq s \lambda \varepsilon / ( 8 \bar { K } )$ if $K \leq \bar { K } \leq M , \bar { K } > 0 ,$ or the cost-only bound

$$
K \leq \frac { \sqrt { \lambda } } { 4 \sqrt { 2 C } \varepsilon } .\tag{8}
$$

A positive envelope $B \leq \bar { B } \leq M / ( s \varepsilon )$ may replace B in the gradient test. A zero envelope needs no gradient test.

Proof. Cost gives $\| r \| \leq \sqrt { 2 C \lambda } \varepsilon$ and $\| \theta - d \| \leq \sqrt { 2 C } \varepsilon$ . Put $\delta _ { E } = \| E \| \leq M \sqrt { 2 C \lambda } \varepsilon \leq s ^ { 2 } \varepsilon ^ { 2 } / 2$ . On tangent pairs the level-set identity gives

$$
\begin{array} { r } { E [ v , w ] = - \langle J ^ { T } r , \operatorname { I I } ( v , w ) \rangle = - \langle b - \lambda ( \theta - d ) , \operatorname { I I } ( v , w ) \rangle . } \end{array}\tag{9}
$$

Thus (7) bounds $\| E _ { T T } \|$ by $\lambda / 4$ . In orthonormal $N \oplus T$ coordinates, the normal block of $H - \lambda I / 2$ is at least $s ^ { 2 } \varepsilon ^ { 2 } I / 2 + \lambda I / 2 ;$ its tangent block is at least $\lambda I / 4 ;$ and its mixed block has norm at most $\delta _ { E }$ . The Schur loss is at most

$$
\frac { 2 \delta _ { E } ^ { 2 } } { s ^ { 2 } \varepsilon ^ { 2 } } \leq \frac { 4 C M ^ { 2 } } { s ^ { 2 } } \lambda \leq \lambda / 3 2 .
$$

Hence $H \succeq \lambda I / 2$ , including when T is empty. Finally $B \leq K / ( s \varepsilon ) \leq M / ( s \varepsilon )$ and $M \sqrt { 2 C } / s \leq 1 / 8$ The ridge-displacement term in the second branch is at most $\lambda / 8 ,$ proving the alternatives. The first branch gives (8). □

The same conditions also compare the full Hessian with the ridge Gauss–Newton matrix.

Corollary 3 (Pointwise relative certificate). Assume all the hypotheses of Theorem 2, including the gate (7), at the point under consideration. Set $P = J ^ { T } J + \lambda I , H = \nabla ^ { 2 } Q$ , and $c _ { * } = ( 1 + \sqrt { 2 } ) / 8$ Then

$$
\boxed { ( 1 - c _ { * } ) P \prec H \prec ( 1 + c _ { * } ) P } .\tag{10}
$$

Both inequalities are strict for every positive λ. In every dimension,

$$
\kappa _ { 2 } ( P ^ { - 1 / 2 } H P ^ { - 1 / 2 } ) < \frac { 1 + c _ { * } } { 1 - c _ { * } } < 1 . 8 6 5 .
$$

If $p = n$ , the stronger bound $\left\| P ^ { - 1 / 2 } ( H - P ) P ^ { - 1 / 2 } \right\| _ { \gamma } \leq 1 / 1 6$ holds. The constant $c _ { * }$ is an optimal supremum over the stated pointwise least-squares class when the map and ridge vary. This does not assert sharpness for one fixed map, a fixed-reference weak-jet family, or a whole tube.

The gate bounds the absolute size of the tangent correction, not only its negative part. This is what gives both sides of (10). A one-sided tangent lower bound alone does not give the same upper bound. The general envelope and the proof are in Appendix A. A separate bound on each normal, mixed, and tangent block already gives $\rho \leq 5 / 1 6$ . The smaller $c _ { * }$ uses their shared bound ∥E∥.

Equivalently, (1) gives $\rho < c _ { * }$ at the evaluated point. At a qualifying stationary point, the ordinary full-correction bound already gives $\rho \leq 1 / 8 $ indeed $\| J ^ { T } r \| = \lambda \| \theta - d \| , \| r \| \leq \lambda \| \theta - d \| / ( s \varepsilon )$ and the cost and cap imply $\| E \| / \lambda \le M \sqrt { 2 C } / s \le 1 / 8$ . The directional distinction is therefore clearest at nonstationary points.

Corollary 4 (Quality of the local quadratic model). At a point with $b \neq 0$ , let $m _ { H } ( h ) = b ^ { T } h +$ $h ^ { T } H h / 2 , h _ { P } = - P ^ { - 1 } b$ , and $h _ { H } = - H ^ { - 1 } b$ $I f \rho < 1$ , then

$$
- m _ { H } ( h _ { P } ) \geq ( 1 - \rho ^ { 2 } ) [ - m _ { H } ( h _ { H } ) ] > 0 .\tag{11}
$$

The proof is in Appendix A.3.

Both steps are evaluated in the same quadratic model $m _ { H }$ . The bound compares their predicted decreases. It does not claim that $Q ( \theta + h _ { P } ) $ decreases or that a Gauss–Newton iteration succeeds.

## 5 An output-alignment example

The next fixed map has two outputs. Aggregate derivative norms lose which output carries the residual. This makes several valid upper bounds inconclusive, while the directional gate passes.

Proposition 5 (A fixed two-output map). All unmarked norms are Euclidean vector and spectral matrix norms; $\| D ^ { 2 } G \|$ is the bilinear operator norm. Fix

$$
G ( u , v , z ) = ( u + z ^ { 2 } / 2 , v ^ { 2 } / 2 ) .
$$

For every $0 < \varepsilon \leq 1 / 1 2 8 , 1 \leq t \leq 2$ , and $1 / 3 2 \leq a \leq 1 / 1 6$ , take $x = d = ( 0 , \varepsilon , 0 ) , \lambda = t \varepsilon ^ { 4 }$ , and $Y = ( 0 , \varepsilon ^ { 2 } / 2 - a \varepsilon ^ { 3 } )$ . Use $s = M = 1$ and $C = 1 / 5 1 2$ in Theorem 2. Then $n = 2 < p = 3 , \sigma _ { 2 } ( J ) = \varepsilon$ the cost and ridge caps hold, and $b \neq 0$ . At $x , K = B = 1$ and

$$
B ( \| b \| + \lambda \sqrt { 2 C } \varepsilon ) \le 1 2 9 \lambda / 2 0 4 8 < \lambda / 8 .
$$

The cost branch of (7) is above $\lambda / 4$ , and $K \| r \| \ge 2 \lambda > \lambda / 4$

To compare whole-correction bounds, freeze P at x. Set $\Delta = x - d , a _ { 0 } = J ^ { T } r = b - \lambda \Delta$ $j = \sigma _ { 2 } ( J )$ , and

$$
\begin{array} { r l } & { M _ { P } = \underset { \Vert h \Vert _ { P } = \Vert k \Vert _ { P } = 1 } { \operatorname* { s u p } } \Vert P ^ { 2 } G [ h , k ] \Vert , } \\ & { \quad j _ { P } ^ { 2 } = \lambda _ { \operatorname* { m i n } } ( J P ^ { - 1 } J ^ { T } ) , } \\ & { \Vert h \Vert _ { P } ^ { 2 } = h ^ { T } P h , \quad \Vert v \Vert _ { P ^ { - 1 } } ^ { 2 } = v ^ { T } P ^ { - 1 } v . } \end{array}
$$

The four valid upper envelopes

$$
\frac { M \lVert r \rVert } { \lambda } , \quad \frac { M \lVert a _ { 0 } \rVert } { j \lambda } , \quad M _ { P } \lVert r \rVert , \quad \frac { M _ { P } \lVert a _ { 0 } \rVert _ { P ^ { - 1 } } } { j _ { P } }
$$

all equal $a / ( t \varepsilon ) \geq 2$ . They therefore do not certify $\rho < 1$ . Here $\Delta = 0$ , so replacing $a _ { 0 }$ by the usual triangle bound adds no loss. Yet $E _ { T T } = 0$ and

$$
\rho = \frac { a \varepsilon } { 1 + t \varepsilon ^ { 2 } } < 1 / 2 0 4 8 .\tag{12}
$$

The componentwise residual bound

$$
\sum _ { i = 1 } ^ { 2 } | r _ { i } | \| P ^ { - 1 / 2 } D ^ { 2 } G _ { i } P ^ { - 1 / 2 } \| _ { 2 } = \rho\tag{13}
$$

also passes. Moreover $E \succeq 0$ , so positivity is immediate from its sign. This example separates the named aggregate bounds, not all residual-aware bounds or methods.

At $x ,$ the first output supplies the fiber’s bending. Only the second output carries residual, so $E _ { T T } = 0$

The ordinary residual and raw-gradient envelopes are the squared-loss specialization of Semenov, Jaggi, and Doikov [11, Example 7, (13)–(14); Appendix H.2, Example 12, (80)–(81)]. The two P-weighted envelopes follow from the same argument after a fixed whitening at the evaluated point. The componentwise estimate uses the residual of each output separately. This pointwise comparison shows how output alignment afects the certificate. The proof is in Appendix A.2.

## 6 Relation to a distance-tube test

Masiha et al. [4, Lemma E.1] prove an ambient full-Hessian bound $\nabla ^ { 2 } g ( x ) + \gamma I \succeq \gamma I / 2$ inside a distance tube around a compact minimizer manifold S. Here $g = \left\| G - Y \right\| ^ { 2 } / 2$ . Their Proposition E.6 bounds the second fundamental form on S. Our $B ( x )$ describes the current output fiber at x. These are diferent curvature objects.

There is also a useful gradient route into their tube. Suppose a point x is already localized near S, the normal Hessian of g along each nearest-point segment is at least $c / 2$ , and y is a nearest point in S. Integrating along the normal segment, as in the local argument of Masiha et al. [4, Lemma E.7, (17)], gives

$$
\operatorname { d i s t } ( x , S ) \leq { \frac { 2 } { c } } \left\| \nabla g ( x ) \right\| \leq { \frac { 2 } { c } } { \big ( } \| \nabla Q ( x ) \| + \lambda \| x - d \| { \big ) } .\tag{14}
$$

Localization is a premise of this inequality. When the right side lies in the radius of Lemma E.1, that lemma yields the same full-Hessian margin. The proof of E.1 also permits the refined radius min $\{ \rho , \lambda / ( 2 L _ { g , 3 } ) \}$ } when $L _ { g , 3 } > 0 ;$ here $\rho$ is that source’s tube radius. Thus this prior route can give a $\lambda / B$ gradient scale when its normal-gap, third-derivative, localization, and curvature-comparison conditions allow it.

Proposition 6 (A negative residual outside two suficient domains). Let $0 < a \leq 1 / 6 4 , 0 < \varepsilon \leq 1 / 3 2$ ， and set

$$
\begin{array} { c } { { G _ { a } ( x , y ) = x ^ { 2 } + a y ^ { 2 } , } } \\ { { Y = \varepsilon ^ { 2 } , } } \end{array}
$$

$$
\begin{array} { l } { { d = ( \varepsilon , 0 ) , } } \\ { { \lambda = a ^ { 2 } \varepsilon ^ { 4 } . } } \end{array}
$$

Define $X = { \sqrt { \varepsilon ^ { 2 } - \lambda / ( 3 2 a ) } }$ and $z = ( X , 0 )$ . This point satisfies Theorem 2 with $C = 1 / 5 1 2 , s = 1$ and $M = 2$ . In particular,

$$
\begin{array} { c } { { Q ( z ) < C \lambda \varepsilon ^ { 2 } , \qquad H ( z ) \succeq \lambda I / 2 , } } \\ { { \| \nabla Q ( z ) \| < \displaystyle \frac { \lambda } { 8 B ( z ) } . } } \end{array}
$$

Its residual correction is negative definite, with

$$
\begin{array} { c l c r } { { \displaystyle E ( z ) = - \frac \lambda { 1 6 a } \mathrm { d i a g } ( 1 , a ) , } } \\ { { \displaystyle \frac { \| E ( z ) \| } \lambda = \frac 1 { 1 6 a } . } } \end{array}\tag{15}
$$

Let $S = \{ ( x , y ) : x ^ { 2 } + a y ^ { 2 } = \varepsilon ^ { 2 } \}$ and $g = \left\| G _ { a } - Y \right\| ^ { 2 } / 2$ . For any admissible positive tube radius $\rho$ and third-derivative bound $L _ { g , 3 } > 0$ in Masiha et al. $/ \mathscr { 4 } ;$ Lemma $E . { \cal { 1 } } \jmath ,$ write

$$
\begin{array} { r l } & { r _ { \mathrm { l i t e r a l } } = \operatorname* { m i n } \{ \rho , \lambda / ( 2 \operatorname* { m a x } \{ L _ { g , 3 } , 1 \} ) \} , } \\ & { r _ { \mathrm { r e f i n e d } } = \operatorname* { m i n } \{ \rho , \lambda / ( 2 L _ { g , 3 } ) \} . } \end{array}
$$

Then

$$
\begin{array} { c l c r } { { \mathrm { d i s t } ( z , S ) = \displaystyle \frac \lambda { 3 2 a ( \varepsilon + X ) } > r _ { \mathrm { l i t e r a l } } , } } \\ { { \displaystyle \frac { \mathrm { d i s t } ( z , S ) } { r _ { \mathrm { r e f i n e d } } } > \displaystyle \frac 3 { 8 a } . } } \end{array}\tag{16}
$$

The proof is in Appendix C.4. Thus the point passes our certificate while failing the absolute residual-norm margin $\| E \| < \lambda$ [9, Proposition 5] and these two isotropic tube tests. The refined radius follows from the proof of Lemma E.1, not its displayed statement. The comparison uses Assumptions 1 and 3 of Masiha et al. for each fixed positive $a , \varepsilon ;$ their global smoothness Assumption 4 fails here. The two ratios diverge as $a \downarrow 0$ at fixed $\varepsilon ,$ but the source constants are not uniform in that limit.

This example illustrates control of a large negative normal correction; its mixed block is zero. It does not show that gradient information is necessary: a directional test using the tighter observed cost also certifies this point. The separation concerns the named suficient conditions, not every cost-based test, every prior method, or the full region of positive curvature.

Appendix C.3 compares the specified suficient tests.

## 7 A structural singular-interpolation family

The pointwise certificate assumes a Jacobian lower bound. Under the derivative conditions below, the Jacobian has full row rank at every low-cost point. These weak-jet conditions concern a rankdeficient reference. For each allowed map, scale, center, label, and ridge weight, the construction also gives an exact interpolator whose objective value is strictly below the low-cost cap. Coexistence still needs the separate common $C ^ { 2 , 1 }$ bounds and curvature-persistence premise of Theorem 1.

Fix a globally $C ^ { 2 }$ reference $F$ , a point $^ { c , }$ and $A = D F ( c )$ of rank $g < n$ . Choose g independent physical output rows $R ,$ let D be the remaining rows, and define the fixed row matrix L by $A _ { D } = L A _ { R }$ . Let $Z$ have orthonormal columns spanning ker A. Empty retained blocks are omitted when $g = 0$ . Define the tensor field and quadratic map

$$
\begin{array} { r c l } { { } } & { { } } & { { \ K _ { F } = D ^ { 2 } F _ { D } - L D ^ { 2 } F _ { R } , } } \\ { { } } & { { } } & { { q ( z ) = - \frac { 1 } { 2 } \mathcal K _ { F } ( c ) [ Z z , Z z ] , } } \\ { { } } & { { } } & { { \chi ( \ell ) = \ell _ { D } - L \ell _ { R } . } } \end{array}
$$

Fix $z _ { 0 } , \ell _ { 0 }$ such that $q ( z _ { 0 } ) = \chi ( \ell _ { 0 } )$ and rank $D q ( z _ { 0 } ) = n - g$ , and put $u _ { 0 } = Z z _ { 0 }$ . Fix $r _ { \star } > 0$ and $M \geq 1 + \operatorname* { s u p } _ { \overline { { B } } ( c , r _ { \star } ) } \| D ^ { 2 } F \|$

Lemma 7 (Cost-only rank and flexible $\exp )$ . There are fixed $s > 0$ , finite weak/strong Jacobian bounds, and $\bar { C } > 0$ depending only on these data with the following property. For every $0 < C \le \bar { C } _ { \mathrm { { } } }$ one can choose $r _ { 0 } \le r _ { \star } , \kappa > 0 , R _ { c } , R _ { \ell } > 0$ and $0 < \varepsilon _ { 0 } \le 1$ , with the tube-radius parameter $\rho = R _ { c } + \sqrt { 2 C } < 1 / 2$ , so that all assertions below are uniform. For every $0 < \varepsilon < \varepsilon _ { 0 }$ , let G be globally $C ^ { 2 }$ with

$$
\begin{array} { r } { \| D G ( c ) - A \| \leq \kappa , \quad \sigma _ { g + 1 } ( D G ( c ) ) \leq \kappa \varepsilon , } \end{array}
$$

$$
\operatorname* { s u p } _ { \overline { { B } } ( c , r _ { 0 } ) } \left\| D ^ { 2 } G \right\| \leq M ,\tag{17}
$$

$$
\operatorname* { s u p } _ { \overline { { B } } ( c , r _ { 0 } ) } \| ( K _ { G } - K _ { F } ) [ Z \cdot , Z \cdot ] \| \leq \kappa .\tag{18}
$$

For all $u \in \overline { { B } } ( u _ { 0 } , R _ { c } ) , \ell \in \overline { { B } } ( \ell _ { 0 } , R _ { \ell } ) , 0 < \lambda \leq \varepsilon ^ { 2 }$ , set $d = c + \varepsilon u$ and $Y = G ( c ) - \varepsilon ^ { 2 } \ell$ . Every whole-space point with $Q \leq C \lambda \varepsilon ^ { 2 }$ lies in $\mathcal { T } _ { \varepsilon } = \overline { { B } } ( c + \varepsilon u _ { 0 } , \rho \varepsilon ) \subset B ( c , r _ { 0 } )$ and satisfies

$$
\left\{ \begin{array} { l l } { a _ { s } \le \sigma _ { j } ( J ) \le b _ { s } , } & { 1 \le j \le g , } \\ { s \varepsilon \le \sigma _ { j } ( J ) \le b _ { w } \varepsilon , } & { g < j \le n , } \end{array} \right.\tag{19}
$$

where $a _ { s } , b _ { s }$ are omitted if $g = 0$ . There is an exact interpolator with $Q < C \lambda \varepsilon ^ { 2 }$ for every tuple. The constants in (19) are fixed before the final cap C is chosen. Only the reference Hessian needs a continuity modulus; maps at diferent scales may be unrelated.

At $^ { c , }$ the reduced reference output $F _ { D } - L F _ { R }$ has zero first derivative. Its quadratic term along ker $A { \mathrm { ~ i s ~ } } { - q }$ . Thus a kernel move of size $O ( \varepsilon )$ changes that reference output by $O ( \varepsilon ^ { 2 } )$ , possibly by zero. For an allowed $G ,$ the weak first derivative need not vanish. The Jacobian bounds control this leakage in moving weak coordinates. The weak-jet bound controls the quadratic comparison.

Under all the lemma’s hypotheses, the proof in Appendix D.1 solves the retained equations, when present, before the reduced equations. The reduced solve uses $q ( z _ { 0 } ) = \chi ( \ell _ { 0 } )$ and surjective $D q ( z _ { 0 } )$ to produce an exact fit. Separately, cost places the weak coordinate of every low-cost point near $z _ { 0 }$ , where Dq $D q$ remains full row rank. After division by $\varepsilon ,$ the weak Jacobian Schur block stays close $\mathrm { t o } - D q$ there. Thus all $n - g$ weak singular values have order ε throughout the low-cost set. Right singular vectors for these nonzero singular values are normal to the current fiber. Tangent directions have zero Jacobian response. The interpolation tolerances are selected after the cap.

Corollary 8 (Indexed spectrum in the structural family). Under all hypotheses of Theorem ${ \mathcal { Q } } ,$ including its cap $C \le C _ { * } ( s , M )$ , local $C ^ { 2 }$ and second-derivative bound, ridge range, rank lower bound, low-cost bound, and gate (7), suppose in addition that the two-sided Jacobian bands (19) hold with fixed constants. Then there are fixed positive $c _ { H } , C _ { H }$ such that, for suficiently small $\varepsilon ,$ the increasingly ordered eigenvalues obey

$$
\begin{array} { l l } { { c _ { H } \lambda \leq \lambda _ { j } ( H ) \leq C _ { H } \lambda , } } & { { 1 \leq j \leq p - n , } } \\ { { c _ { H } \varepsilon ^ { 2 } \leq \lambda _ { j } ( H ) \leq C _ { H } \varepsilon ^ { 2 } , } } & { { p - n < j \leq p - g , } } \\ { { c _ { H } \leq \lambda _ { j } ( H ) \leq C _ { H } , } } & { { p - g < j \leq p . } } \end{array}\tag{20}
$$

Choose the cap of Lemma 7 below $C _ { * } ( s , M )$ . Then these conclusions hold at every low-cost point of its structural family satisfying the adaptive test. In particular $\| b \| \leq \gamma \lambda \varepsilon$ with $\gamma = s / ( 8 M )$ sufices. Every tuple has a certified stationary global minimizer strictly below the cap.

The proof is in Appendix D.3.

The indexed estimates do not assert asymptotic separation of the ridge and weak bands when $\lambda \asymp \varepsilon ^ { 2 }$ . Without the two-sided Jacobian bands, Theorem 2 still gives a tangent Rayleigh-quotient upper bound $5 \lambda / 4$ and a lower bound $s ^ { 2 } \varepsilon ^ { 2 } / 2$ on the normal compression. Min–max and interlacing turn these into indexed eigenvalue bounds. They do not identify invariant tangent eigenvectors. The complete classification needs (19). Appendix D.2 gives the quadratic calculation.

## 8 Related work and scope

The residual-inclusive Hessian [1, Theorem 4.1], level-set curvature [2, Lemma 2.2.1], and embeddedmanifold correction [3, Theorem 1] are classical foundations. Our conditional common interval certifies every global minimizer while retaining a low-cost indefinite point for each allowed objective.

Masiha et al. [4, (8), Lemma E.1, Proposition E.6] give related curvature and distance-tube results, including a full ambient positivity margin. Semenov, Jaggi, and Doikov [11, Example 7, (13)–(14); Appendix H.2, Example 12, (80)–(81)] give the ordinary residual and gradient bound for a nonlinear least-squares correction. Applied to $g = \| G - Y \| ^ { 2 } / 2$ , it gives $\| E \| \leq M \| r \| \leq ( M / j ) \| b - \lambda ( \theta - d ) \|$ Fixing P and changing coordinates gives a pointwise P-metric bound. Our gate uses additional information from the current prediction level set. The frozen-metric bound also certifies the scalar point in Proposition 6, where $\rho = 1 / 1 6$

Liu, Zhu, and Belkin [6, Section 3, (9)–(10), Proposition 2] and Rebjock and Boumal [5, Section 4.2] give qualitative nearby nonconvexity results for curved sets of interpolants or minima. For squared distance to a fixed manifold, Leobacher and Steinicke’s projection formula [7, Theorem C] also yields a λ/curvature scale by the calculation in Appendix C.1. That inference does not cover an arbitrary raw least-squares map. Bock et al. [8, Section 2, Example 2] reverse a residual Hessian term by mirroring measurement labels. Their construction uses a nonzero-residual stationary point and a full-column-rank Jacobian. Kharel et al. [9, Proposition 5] give a margin when their residual Hessian norm lies below the ridge.

Our common interval covers every allowed independent center and label pair for all $0 < \lambda \leq W$ The same neighborhoods work as the ridge tends to zero. The structural bridge builds on regular-zero theory [12, 14] and stability of inversion under small perturbations [15, 16]. These perturbation results apply once the full normalized map is close to a regular reference. Our weak-jet estimates establish this closeness and give uniform rank throughout the low-cost set. Appendix D.1 details the connection.

The pointwise result needs local $C ^ { 2 }$ bounds; coexistence separately requires common $C ^ { 2 , 1 }$ bounds and curvature-magnitude persistence (Sections B.1 and B.3). Proposition 10 permits changing orientation but shows failure of a fixed base residual; Proposition 11 shows smoothness alone cannot fix normalized label radii.

The results concern curvature in the fixed Euclidean parameter metric. They do not imply convergence of a Gauss–Newton iteration.

## 9 Conclusion

Low cost and accurate curvature at every global minimizer can coexist with an indefinite Hessian elsewhere in the same low-cost set. Under the stated regularity and curvature-persistence assumptions, this contrast survives independent center and label perturbations throughout one common positive ridge interval. The perturbation neighborhoods remain fixed as the ridge weight tends to zero.

The pointwise certificate measures the residual correction through current level-set geometry. Its relative bound is sharp over the stated pointwise class. The analytic examples show why aggregate bounds can lose output alignment and why local smoothness alone does not give the same uniform label neighborhood. Separately, the structural result supplies rank and interpolation guarantees near a singular reference. These results distinguish curvature accuracy at a point from accuracy throughout a low-cost region.

## Acknowledgments

Generative AI assisted proof development, checks of intermediate arguments, literature comparisons, and manuscript editing. The authors verified the claims and proofs and take responsibility for the contents.

## References

[1] J. Eriksson, P. A. Wedin, M. E. Gulliksson, and I. Söderkvist. Regularization methods for uniformly rank-deficient nonlinear least-squares problems. Journal of Optimization Theory and Applications, 127(1):1–26, 2005. doi:10.1007/s10957-005-6389-0.

[2] P. M. N. Feehan and T. G. Leness. Virtual Morse–Bott index, moduli spaces of pairs, and applications to the topology of smooth four-manifolds. arXiv:2010.15789v10, 2026.

[3] P.-A. Absil, R. Mahony, and J. Trumpf. An extrinsic look at the Riemannian Hessian. In Geometric Science of Information, 2013. doi:10.1007/978-3-642-40020-9\_39.

[4] S. Masiha, Z. Shen, N. Kiyavash, and N. He. Select-then-diferentiate: Solving bilevel optimization with manifold lower-level solution sets. arXiv:2605.09209v1, 2026.

[5] Q. Rebjock and N. Boumal. Fast convergence to non-isolated minima: four equivalent conditions for $C ^ { 2 }$ functions. Mathematical Programming, 213:151–199, 2025. doi:10.1007/s10107-024-02136- 6.

[6] C. Liu, L. Zhu, and M. Belkin. Loss landscapes and optimization in over-parameterized nonlinear systems and neural networks. Applied and Computational Harmonic Analysis, 59:85–116, 2022. doi:10.1016/j.acha.2021.12.009.

[7] G. Leobacher and A. Steinicke. Existence, uniqueness and regularity of the projection onto diferentiable manifolds. arXiv:1811.10578v4, 2020.

[8] H. G. Bock, J. Gutekunst, A. Potschka, and M. E. Suárez Garcés. A flow perspective on nonlinear least-squares problems. Vietnam Journal of Mathematics, 2020. doi:10.1007/s10013- 020-00441-z.

[9] A. Kharel, I. Kuzborskij, P. Rebeschini, and Y. Abbasi-Yadkori. Generalization in nonlinear least squares via learned feature geometry. arXiv:2606.08799v2, 2026.

[10] S. Gratton, A. S. Lawless, and N. K. Nichols. Approximate Gauss–Newton methods for nonlinear least squares problems. SIAM Journal on Optimization, 18(1):106–132, 2007. doi:10.1137/050624935. Author report (2004).

[11] A. Semenov, M. Jaggi, and N. Doikov. Gradient-normalized smoothness for optimization with approximate Hessians. In International Conference on Learning Representations, 2026. Oficial conference PDF.

[12] H. J. Sussmann. High-order open mapping theorems. In A. Rantzer and C. I. Byrnes (eds.), Directions in Mathematical Systems Theory and Optimization, Lecture Notes in Control and Information Sciences 286, pp. 293–316. Springer, 2003. doi:10.1007/3-540-36106-5\_22. Author preprint.

[13] J. Martens. Deep learning via Hessian-free optimization. In Proceedings of the 27th International Conference on Machine Learning, 2010. Oficial conference PDF.

[14] A. V. Arutyunov and D. Yu. Karamzin. Regular zeros of quadratic maps and their application. Sbornik: Mathematics, 202(6):783–806, 2011. doi:10.1070/SM2011v202n06ABEH004166.

[15] E. R. Avakov and G. G. Magaril-Il’yaev. General implicit function theorem for close mappings. Proceedings of the Steklov Institute of Mathematics, 315:1–12, 2021. doi:10.1134/S0081543821050011.

[16] A. V. Arutyunov and S. E. Zhukovskiy. On stability of smooth nonlinear mappings at a given point. Trudy Instituta Matematiki i Mekhaniki UrO RAN, 31(2):30–37, 2025 (in Russian). doi:10.21538/0134-4889-2025-31-2-30-37.

## A Relative accuracy and a pointwise example

## A.1 Relative spectral control

The adaptive gate also controls the Hessian relative to the ridge Gauss–Newton matrix. This comparison is pointwise. It does not require the two-sided Jacobian bands of Lemma 7.

Let $p \geq n \geq 1 , J \in \mathbb { R } ^ { n \times p }$ have full row rank and $\sigma _ { n } ( J ) \geq j > 0$ . Set $T =$ ker J, $\gamma = j ^ { 2 }$ $P = J ^ { T } J + \lambda I _ { p }$ with $\lambda > 0 .$ , and $H = P + E$ for symmetric E with $\| E \| _ { 2 } \leq \delta$ . A one-sided tangent envelope $0 \leq k \leq \delta$ means $t ^ { T } E t \geq - k \left. t \right. ^ { 2 }$ for $t \in T$ . An absolute envelope means $| t ^ { T } E t | \leq k \| t \| ^ { 2 }$

Theorem 9 (Relative envelope). $I f p > n$ , define

$$
\rho _ { - } = \frac { \gamma k + \sqrt { \gamma ^ { 2 } k ^ { 2 } + 4 \lambda ( \lambda + \gamma ) \delta ^ { 2 } } } { 2 \lambda ( \lambda + \gamma ) } .\tag{21}
$$

This is the nonnegative root of $\lambda ( \lambda + \gamma ) u ^ { 2 } - \gamma k u - \delta ^ { 2 } = 0$ . A one-sided tangent envelope gives

$$
( 1 - \rho _ { - } ) P \preceq H \preceq ( 1 + \delta / \lambda ) P .\tag{22}
$$

Both bounds are sharp in the displayed information class. If the envelope is absolute, the sharp symmetric bound is

$$
( 1 - \rho _ { - } ) P \preceq H \preceq ( 1 + \rho _ { - } ) P .\tag{23}
$$

Each endpoint is attained in dimension $p = 2 , n = 1$ . They are attained together in one $p = 4 , n = 2$ instance when $\rho _ { - } < 1$ . In that class, the upper bound $( 1 + \rho _ { - } ) / ( 1 - \rho _ { - } )$ on $\kappa _ { 2 } ( P ^ { - 1 / 2 } H P ^ { - 1 / 2 } )$ is sharp. These endpoint statements do not assert joint attainment in every fixed dimension. $I f p = n$ $T = \{ 0 \}$ and the sharp norm replacement is

$$
\left\| P ^ { - 1 / 2 } E P ^ { - 1 / 2 } \right\| _ { 2 } \leq \frac { \delta } { \lambda + \gamma } .\tag{24}
$$

The scalar case can attain this norm endpoint even when H is singular. $I f p = n = 1$ and $H \succ 0$ then $\kappa _ { 2 } ( P ^ { - 1 / 2 } H P ^ { - 1 / 2 } ) = 1$

The proof is in Appendix A.1.1. Its two-sided conclusion uses the absolute tangent bound. From a one-sided bound alone, the upper endpoint remains $\delta / \lambda$ . The formula is a rescaling corollary of a sharp additive envelope derived from block and Schur arguments; the elementary two-dimensional extremizer also appears in that additive result.

To prove Corollary 3, set $\mu = \| r \| , \delta = M \mu , j = s \varepsilon$ , and $\gamma = s ^ { 2 } \varepsilon ^ { 2 } . \mathrm { ~ I f ~ } p > n$ , the pointwise hypotheses give the absolute tangent envelope

$$
\begin{array} { r } { k = \operatorname* { m i n } \{ \delta , \ K \sqrt { 2 C \lambda } \varepsilon , \ } \\ { B ( \| b \| + \lambda \sqrt { 2 C } \varepsilon ) \} . } \end{array}\tag{25}
$$

It satisfies $k \leq \lambda / 4$ and $\delta ^ { 2 } \le \gamma \lambda / 6 4$ . Substituting these bounds in (21) gives $\rho _ { - } < c _ { * }$ for every positive ridge. For $p = n , ( 2 4 )$ is at most $1 / 1 6$ . Appendix A.1.1 proves these estimates and the varying-map supremum claim.

## A.1.1 Proof and sharpness

Write $N = T ^ { \perp } = \tan J ^ { T }$ and $\gamma = j ^ { 2 }$ . An additive block bound gives, for $p > n$ , a symmetric E with $\| E \| _ { 2 } \leq \delta$ and $E | _ { T } \succeq - k I _ { T }$ obeys

$$
J ^ { T } J + \lambda I + E \succeq ( \lambda - \chi ) I , \qquad \chi ( \gamma , k , \delta ) = \frac { \sqrt { \gamma ^ { 2 } + 4 \gamma k + 4 \delta ^ { 2 } } - \gamma } { 2 } .\tag{P1}
$$

This additive frontier follows from the block and Schur argument below. Its extremizer also gives the sharp relative endpoint. It holds for every positive ridge and every row-rank lower bound.

Put $a = \chi$ . Its defining identity is $a ( a + \gamma ) = \delta ^ { 2 } + \gamma k$ . It sufices to show $a I + E + J ^ { T } J \succeq 0$ Write $B _ { a } = a I + E$ in $T \oplus N$ blocks $\left\lceil \begin{array} { c c } { R } & { F } \\ { F ^ { T } } & { C } \end{array} \right\rceil$ . The norm bound gives $( a - \delta ) I \preceq B _ { a } \preceq ( a + \delta ) I$ , and $R \succeq ( a - k ) I . \mathrm { ~ I f ~ } a \geq \delta ,$ then $B _ { a } \succeq 0$ . The remaining case has $k < a < \delta$ . Indeed, $a < k$ contradicts the defining identity and $\delta \geq k ;$ equality $a = k$ requires $a = k = \delta$ and belongs to the first case.

Set $l = a { - } \delta < 0 , u = a { + } \delta > 0 \mathrm { a n d } t = a { - } k > 0 .$ . The Schur complement of R is $S = C { - } F ^ { T } R ^ { - 1 } F$ For a unit $y \in N$ , its minimizing tangent vector is $x = - R ^ { - 1 } F y$ . If $x \neq 0$ , compress $B _ { a }$ to the plane spanned by $x / \left. x \right.$ and y, obtaining $\left[ \begin{array} { l } { r _ { 0 } } \\ { f _ { 0 } } \end{array} \right.$ . Stationarity gives $f _ { 0 } = - \left\| x \right\| r _ { 0 }$ and hence $y ^ { T } S y = c _ { 0 } - f _ { 0 } ^ { 2 } / r _ { 0 } = \operatorname * { d e t } ( B _ { a } | _ { \mathrm { p l a n e } } ) / r _ { 0 }$ . Both eigenvalues of this compression lie in $[ l , u ]$ . Their product is at least lu. Since $r _ { 0 } \geq t$ and lu $< 0 , y ^ { T } S y \ge l u / t . \mathrm { ~ I f ~ } x = 0$ , then $y ^ { T } S y = y ^ { T } C y \geq \bar { l } \geq l u / t$ Thus $S \succeq ( l u / t ) I _ { N }$ . Finally $J ^ { T } J | _ { N } \succeq \gamma I _ { N }$ and $\gamma t \geq \delta ^ { 2 } - a ^ { 2 } = - l u$ by the defining identity. The Schur complement of R in $a I + E + J ^ { T } J$ is nonnegative. This proves (P1), including the case $a = 0$ by the first case.

Fix $u > 0$ and apply (P1) to $u P + E$ using $J _ { u } = { \sqrt { u } } J$ , ridge uλ, and $\gamma _ { u } = u \gamma$ . Thus $u P + E \succeq 0$ whenever $u \lambda \ge \chi ( u \gamma , k , \delta )$ . The function $a \mapsto a ( a + u \gamma )$ is increasing on $a \geq 0$ , and the defining identity for χ is $\chi ( \chi + u \gamma ) = \delta ^ { 2 } + u \gamma k$ . Therefore

$$
u \lambda \geq \chi ( u \gamma , k , \delta ) \quad \iff \quad \lambda ( \lambda + \gamma ) u ^ { 2 } - \gamma k u - \delta ^ { 2 } \geq 0 .\tag{P2}
$$

For $\delta > 0$ the polynomial is negative at zero, then crosses zero once on $u \geq 0$ , at $\rho .$ <sub>−</sub> in (21). Consequently $E \succeq - \rho _ { - } P$ . If $\delta = 0$ , then $k = 0 , E = 0$ , and the assertion holds with $\rho _ { - } = 0$

The universal norm bound $E \preceq \delta I \preceq ( \delta / \lambda ) P$ gives the upper side of (22). If the tangent premise is absolute, apply the lower argument $\mathrm { t o } - E$ . Both signs have norm at most δ and tangent lower envelope k. This gives (23). One-sided information alone leaves the upper endpoint at $\delta / \lambda$

For sharpness at fixed parameters, use $p = 2 , n = 1$ with coordinates $( t , y ) , J = ( 0 , j )$ , and

$$
P = \mathrm { d i a g } ( \lambda , \lambda + \gamma ) , \quad q = \sqrt { \delta ^ { 2 } - k ^ { 2 } } , \quad E _ { 0 } = \left( \begin{array} { c c } { { - k } } & { { q } } \\ { { q } } & { { k } } \end{array} \right) .\tag{P3}
$$

The matrix $E _ { 0 }$ has eigenvalues $\pm \delta$ and tangent value −k. For every $u \geq 0$

$$
\operatorname * { d e t } ( u { \cal P } + E _ { 0 } ) = ( u \lambda - k ) ( u ( \lambda + \gamma ) + k ) - q ^ { 2 } = \lambda ( \lambda + \gamma ) u ^ { 2 } - \gamma k u - \delta ^ { 2 } .\tag{P4}
$$

The determinant vanishes at $u = \rho .$ <sub>−</sub>, so $P ^ { - 1 / 2 } E _ { 0 } P ^ { - 1 / 2 }$ has eigenvalue $- \rho .$ <sub>−</sub>. The sign reversal $- E _ { 0 }$ has the same norm and absolute tangent bound and gives eigenvalue $+ \rho _ { - }$ . In the one-sided class, $E = \mathrm { d i a g } ( \delta , 0 )$ instead gives the positive relative endpoint $\delta / \lambda$ while satisfying $t ^ { T } E t \geq - k \| t \| ^ { 2 }$ for any $k \geq 0$ . Moreover

$$
\rho _ { - } \le \delta / \lambda ,
$$

because the polynomial in (P2) at $u = \delta / \lambda$ equals $\gamma \delta ( \delta - k ) / \lambda \geq 0 ;$ it is strictly positive when $k < \delta$ and $\delta > 0$ . Thus the two-sided optimum with one-sided tangent information is $\delta / \lambda$

To attain both endpoints together, take two independent $( t , y )$ blocks with J singular value $j$ on each normal coordinate and $E = \mathrm { d i a g } ( E _ { 0 } , - E _ { 0 } )$ . Its spectral norm remains $\delta ,$ its tangent restriction has norm $k ,$ and the relative correction has eigenvalues $- \rho _ { - }$ and $+ \rho _ { - }$ . If $\rho _ { - } < 1$ , H is positive definite, and the extremal preconditioned eigenvalues give condition number exactly $( 1 + \rho _ { - } ) / ( 1 - \rho _ { - } )$ . These witnesses range over an abstract pointwise envelope class; they do not assert extremality for a fixed architecture or whole-tube conditions.

When $p = n , T = \{ 0 \}$ and $J ^ { T } J \succeq \gamma I$ . Hence $P \succeq ( \lambda + \gamma ) I$ , and $- \delta I \preceq E \preceq \delta I$ gives (24). Taking $J = j I$ and $E = \pm \delta I$ attains either norm endpoint. The norm endpoint may have singular $H \colon$ for $p = n = 1$ , take $E = - ( \lambda + j ^ { 2 } )$ and $\delta = \lambda + j ^ { 2 }$ to get $H = 0$ . When $p = n = 1$ and $H \succ 0$ the scalar matrix $P ^ { - 1 / 2 } H P ^ { - 1 / \overset { . } { 2 } }$ has spectral condition number one.

The pointwise gate and its uniform constant. At the point $\theta ,$ the least-squares Hessian identity gives

$$
E ( v , w ) = \langle r , D ^ { 2 } G ( \theta ) [ v , w ] \rangle .\tag{P5}
$$

The cost inequality yields

$$
\mu = \| r \| \leq \sqrt { 2 C \lambda } \varepsilon , \qquad \| \theta - d \| \leq \sqrt { 2 C } \varepsilon .\tag{P6}
$$

Thus $\| E \| _ { 2 } \leq M \mu = \delta$ and, on $T$ , the absolute quadratic-form bound $\begin{array} { r } { | E ( t , t ) | \leq K \mu \| t \| ^ { 2 } } \end{array}$ holds. For the other branch, $J J ^ { \dagger } = I _ { n }$ and the definition of II imply $D ^ { 2 } G [ t , t ] = - J \mathrm { I I } ( t , t )$ on $T$ . Since $J ^ { T } r = b - \lambda ( \theta - d )$

$$
| E ( t , t ) | \leq B \| J ^ { T } r \| \| t \| ^ { 2 } \leq B \big ( \| b \| + \lambda \sqrt { 2 C } \varepsilon \big ) \| t \| ^ { 2 } .\tag{P7}
$$

The global bound also bounds this tangent form by $\delta .$ Taking the minimum gives the absolute envelope (25), and (7) gives $k \leq \lambda / 4$ . The second ceiling on $C$ gives

$$
\delta ^ { 2 } \leq 2 M ^ { 2 } C \lambda \varepsilon ^ { 2 } \leq s ^ { 2 } \varepsilon ^ { 2 } \lambda / 6 4 = \gamma \lambda / 6 4 .\tag{P8}
$$

The first ceiling on $C$ and $\lambda \le \varepsilon ^ { 2 }$ remain part of Theorem 2; (P8) and the relative envelope do not use their full strength.

Let $c _ { * } = ( 1 + \sqrt { 2 } ) / 8$ , the positive root of $c _ { * } ^ { 2 } - c _ { * } / 4 - 1 / 6 4 = 0 . \mathrm { ~ A t ~ } u = c _ { * }$ , the left side of the polynomial in (P2) satisfies

$$
\lambda ( \lambda + \gamma ) c _ { * } ^ { 2 } - \gamma k c _ { * } - \delta ^ { 2 } \geq \lambda ^ { 2 } c _ { * } ^ { 2 } + \gamma \lambda ( c _ { * } ^ { 2 } - c _ { * } / 4 - 1 / 6 4 ) = \lambda ^ { 2 } c _ { * } ^ { 2 } > 0 .\tag{P9}
$$

The nonnegative root therefore obeys $\rho _ { - } < c _ { * }$ . For $p = n$ , (P8) and the arithmetic-geometric mean inequality give

$$
\frac { \delta } { \lambda + \gamma } \leq \frac { \sqrt { \gamma \lambda } } { 8 ( \lambda + \gamma ) } \leq \frac { 1 } { 1 6 } .\tag{P10}
$$

Since $c _ { * } < 1$ , these bounds make H positive definite. The spectral condition-number ratio is at most $( 1 + \rho ) / ( 1 - \rho )$ for the respective dimension-specific $\rho ,$ and is strictly below $( 1 + c _ { * } ) / ( 1 - c _ { * } )$

The uniform constant cannot be decreased over the stated pointwise least-squares class when the map varies. Set $p = 4 , n = 2 , s = M = \varepsilon = 1 , C = 1 / 1 2 8 , 0 < \lambda \leq 1 / 4 , \mu = { \sqrt { \lambda } } / 8 , \delta = \mu , k = \lambda / 4$ and

$$
A = \left( \frac { - k } { \sqrt { \delta ^ { 2 } - k ^ { 2 } } } \begin{array} { c c } { { \sqrt { \delta ^ { 2 } - k ^ { 2 } } } } \\ { { k } } \end{array} \right) , \qquad { \mathcal { E } } = \mathrm { d i a g } ( A , - A ) .
$$

For $u = ( t _ { 1 } , y _ { 1 } , t _ { 2 } , y _ { 2 } )$ define

$$
\boldsymbol { F } ( \boldsymbol { u } ) = \left( \mu + y _ { 1 } + \frac { u ^ { T } \mathcal { E } \boldsymbol { u } } { 2 \mu } , y _ { 2 } \right) , \qquad \boldsymbol { \theta } = \boldsymbol { d } = 0 , \quad Y = 0 .
$$

At this point, $J = \left( \begin{array} { l l l } { 0 } & { 1 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 1 } \end{array} \right)$ has both singular values one, $r = ( \mu , 0 )$ , and $Q ( 0 ) = \mu ^ { 2 } / 2 = \lambda / 1 2 8 =$ $C \lambda \varepsilon ^ { 2 }$ . The scalar second derivative of $F _ { 1 }$ is $\mathcal { E } / \mu$ , whose bilinear norm is $\delta / \mu = 1 = M ;$ ; the second derivative of $F _ { 2 }$ vanishes. The restriction to $T = \operatorname { s p a n } \{ e _ { t _ { 1 } } , e _ { t _ { 2 } } \}$ has norm $K = k / \mu ,$ so the first branch of (7) is $K { \sqrt { 2 C \lambda } } = K \mu = k = \lambda / 4$ . All other displayed pointwise hypotheses hold for $0 < \lambda \leq 1 / 4 ;$ in particular $k \leq \delta$ . The Hessian correction is exactly $\mathcal { E } ,$ and the two relative blocks are opposites. Substitution into (21) gives

$$
\rho _ { \lambda } = \frac { 1 + \sqrt { 2 + \lambda } } { 8 ( 1 + \lambda ) } \xrightarrow [ \lambda \downarrow 0 ] { } \frac { 1 + \sqrt { 2 } } { 8 } .\tag{P11}
$$

One block has minimum $- \rho _ { \lambda }$ and the other maximum $+ \rho _ { \lambda }$ , proving both the uniform-norm supremum and the condition-number-ratio supremum. Each λ defines a diferent $F ;$ nothing here embeds them into one fixed model or one fixed-reference, weak-jet, or whole-tube family.

## A.2 Proof for the two-output example

Proof. Write $R = a \varepsilon ^ { 3 }$ . Direct diferentiation at x gives

$$
r = ( 0 , R ) , \quad J = \left( { \begin{array} { c c c } { 1 } & { 0 } & { 0 } \\ { 0 } & { \varepsilon } & { 0 } \end{array} } \right) , \quad D ^ { 2 } G [ h , k ] = ( h _ { z } k _ { z } , h _ { v } k _ { v } ) .
$$

For unit $h , k ,$ the squared norm of the last vector is at most $( h _ { z } ^ { 2 } + h _ { v } ^ { 2 } ) ( k _ { z } ^ { 2 } + k _ { v } ^ { 2 } ) \le 1$ . Equality is attained on $e _ { z }$ or $e _ { v } ,$ so $M = 1$ globally. The singular values of J are $1 , \varepsilon .$ . Also $\lambda = t \varepsilon ^ { 4 } \leq \varepsilon ^ { 2 }$ and $C = 1 / 5 1 2 \le C _ { * } ( 1 , 1 ) = 1 / 1 2 8$ . The cost is $Q ( x ) = a ^ { 2 } \varepsilon ^ { 6 } / 2 \leq t \varepsilon ^ { 6 } / 5 1 2 = C \lambda \varepsilon ^ { 2 }$ because $2 5 6 a ^ { 2 } \leq t$ The gradient is $b = J ^ { T } r = ( 0 , a \varepsilon ^ { 4 } , 0 ) \neq 0$

Here $T = \mathrm { s p a n } \{ e _ { z } \} , D ^ { 2 } G [ e _ { z } , e _ { z } ] = ( 1 , 0 )$ , and $\operatorname { I I } ( e _ { z } , e _ { z } ) = - e _ { u }$ . Thus $K = B = 1$ and $E _ { T T } = 0$ Since $\sqrt { 2 C } = 1 / 1 6$ , the gradient gate divided by λ is $a / t + \varepsilon / 1 6 \leq 1 2 9 / 2 0 4 8 < 1 / 8$ . The cost gate is $\sqrt { t } \varepsilon ^ { 3 } / 1 6$ , whose ratio to $\lambda / 4$ is $1 / ( 4 \sqrt { t } \varepsilon ) > 1$ . The observed tangent residual envelope obeys $K \| r \| / \lambda = a / ( t \varepsilon ) \geq 2$

In coordinate order $( u , v , z )$ , the matrices are

$$
E = \mathrm { d i a g } ( 0 , R , 0 ) , \quad P = \mathrm { d i a g } ( 1 + \lambda , \varepsilon ^ { 2 } + \lambda , \lambda ) , \quad H = P + E .
$$

This proves (12), since $R / ( \varepsilon ^ { 2 } + \lambda ) = a \varepsilon / ( 1 + t \varepsilon ^ { 2 } )$ . The inequality is strict even at $a = 1 / 1 6 , \varepsilon = 1 / 1 2 8$ because the denominator exceeds one.

For the frozen metric, the exact bilinear norm is $M _ { P } = \operatorname* { m a x } \{ 1 / \lambda , 1 / ( \varepsilon ^ { 2 } + \lambda ) \} = 1 / \lambda$ . Moreover

$$
\begin{array} { r } { J P ^ { - 1 } J ^ { T } = \operatorname { d i a g } \left( ( 1 + \lambda ) ^ { - 1 } , ( 1 + t \varepsilon ^ { 2 } ) ^ { - 1 } \right) , \quad j _ { P } = ( 1 + t \varepsilon ^ { 2 } ) ^ { - 1 / 2 } , } \end{array}
$$

and $\| a _ { 0 } \| _ { P ^ { - 1 } } = R / ( 1 + t \varepsilon ^ { 2 } ) ^ { 1 / 2 }$ . Since $\Delta = 0 , a _ { 0 } = b _ { \mathrm { { + } } }$ , and direct substitution makes all four named envelopes $a / ( t \varepsilon )$ . Finally, the whitened component Hessians are $\mathrm { d i a g } ( 0 , 0 , 1 / \lambda )$ and dia $\mathrm { g } ( 0 , 1 / ( \varepsilon ^ { 2 } +$ $\lambda ) , 0 )$ . Only the second has nonzero residual weight, proving (13). The displayed E is positive semidefinite. □

## A.3 Quality of the local quadratic model

We prove Corollary 4. Both steps are compared in the same quadratic model of the full Hessian.

Proof. Set $S ~ = ~ P ^ { - 1 / 2 } ( H - P ) P ^ { - 1 / 2 }$ and $q \ = \ P ^ { - 1 / 2 } b .$ . Then $- 2 m _ { H } ( h _ { P } ) \ = \ q ^ { T } ( I - S ) q$ and $- 2 m _ { H } ( h _ { H } ) = q ^ { T } ( I + S ) ^ { - 1 } q$ . Each eigenvalue s of S satisfies $( 1 - s ) ( 1 + s ) = 1 - s ^ { 2 } \geq 1 - \rho ^ { 2 }$ . The spectral theorem proves the claim. □

## B Common-domain coexistence and geometric limits

## B.1 Perturbed common-domain theorem

We use the constants and notation of Theorem 1. The proof separates the exact-fit anchor, the entire low-cost set, the passing global minimizer, and the recentered indefinite witness.

A normal exact-fit anchor for every label. Let $N _ { 0 } = ( \ker J _ { 0 } ) ^ { \perp }$ . For $h \in N _ { 0 }$ with $\| h \| \leq$ $H _ { v } : = 2 \lVert v \rVert / \sigma$ , write

$$
G ( x _ { 0 } + h ) - G ( x _ { 0 } ) = J _ { 0 } h + E _ { 0 } ( h ) , \qquad \| E _ { 0 } ( h ) \| \leq \frac { M } { 2 } \| h \| ^ { 2 } .
$$

The Hessian bound also makes $E _ { 0 } ~ M H _ { v } – \mathrm { L i p s c h i t z }$ on this ball. The fixed-point map $F _ { v } ( h ) =$ $J _ { 0 } ^ { \dag } ( v - E _ { 0 } ( h ) )$ has $\lVert F _ { v } ( 0 ) \rVert \le H _ { v } / 2$ and Lipschitz constant

$$
\frac { M H _ { v } } { \sigma } \leq \frac { 2 M \beta } { s ^ { 2 } } \leq \frac { 1 } { 1 6 } .
$$

Because $H _ { v } \leq ( 2 \beta / s ) \varepsilon \leq R / 4$ , this is a contraction of the closed $H _ { v } – \mathrm { b a l l }$ into itself, within the $C ^ { 2 , 1 }$ neighborhood. Its unique fixed point $h ( v )$ satisfies $G ( x _ { 0 } + h ( v ) ) = G ( x _ { 0 } ) + v$ . For $v = 0$ it is zero. The other radius bound in $\beta$ gives

$$
\| h ( v ) \| \leq H _ { v } \leq \alpha \varepsilon , \quad \| y ( v ) - d \| \leq ( \alpha + 2 \beta / s ) \varepsilon \leq \delta \varepsilon \leq \frac { \sqrt { C } } { 8 } \varepsilon .\tag{26}
$$

In addition, $\| D G ( y ( v ) ) - J _ { 0 } \| \le M H _ { v } \le \sigma / 1 6$ , so $m : = \sigma _ { n } ( D G ( y ( v ) ) ) \geq 1 5 \sigma / 1 6 \geq m _ { 0 }$ . The assumed persistence gives $B : = B _ { y ( v ) } \in [ b _ { 0 } , 2 B _ { 0 } ]$

The entire low-cost set is inside the certified domain. For any $x \in { \mathcal { L } } _ { \lambda }$ , its ridge term gives $\| x - d \| \leq \sqrt { 2 C } \varepsilon$ . Therefore

$$
\| x - x _ { 0 } \| \leq ( \alpha + \sqrt { 2 C } ) \varepsilon \leq R .
$$

The second term of $C _ { * } ( s _ { * } , M )$ implies $\sqrt { 2 C } \leq s / ( 1 6 M )$ , while $\alpha \leq s / ( 3 2 M )$ . Hence the Jacobian Lipschitz bound on the segment from $x _ { 0 }$ to x gives

$$
\sigma _ { n } ( D G ( x ) ) \geq \sigma - M \| x - x _ { 0 } \| \geq \left( 1 - \frac { 1 } { 3 2 } - \frac { 1 } { 1 6 } \right) s \varepsilon = \frac { 2 9 } { 3 2 } s \varepsilon > s _ { * } \varepsilon .\tag{27}
$$

All these points inherit $\| D ^ { 2 } G ( x ) \| \leq M$ . Since $\| D G ( x ) ^ { \dag } \| \leq 1 / ( s _ { * } \varepsilon )$ , the bilinear norm of the level-set curvature obeys

$$
B _ { x } \leq \frac { M } { s _ { * } \varepsilon } .\tag{28}
$$

This proves the first claim for all low-cost points, not just for points constructed in the proof.

A passing point exists for the perturbed objective. The anchor is an exact fit for the selected label, so by (26),

$$
Q ( y ( v ) ) = \frac { \lambda } { 2 } \| y ( v ) - d \| ^ { 2 } \leq \frac { C } { 1 2 8 } \lambda \varepsilon ^ { 2 } < C \lambda \varepsilon ^ { 2 } .\tag{29}
$$

The whole-space objective is continuous and $Q ( x ) \geq \lambda \| x - d \| ^ { 2 } / 2 \to \infty$ as $\| x \| \to \infty$ . It therefore attains a global minimum. Every global minimizer $z _ { \lambda }$ satisfies $Q ( z _ { \lambda } ) \leq Q ( y ( v ) )$ , so is strictly

low cost. By (27) it lies in the local $C ^ { 2 }$ neighborhood. Unconstrained first-order optimality gives $\nabla Q ( z _ { \lambda } ) = 0$ , and the statistic satisfies $A ( z _ { \lambda } ) = 0$ . This establishes passing-point existence even when $Q ( d ) > 0$

For any low-cost x satisfying $A ( x ) \leq 1 / 8$ , the second branch of the pointwise gate is, by (28),

$$
B _ { x } ( \| \nabla Q ( x ) \| + \lambda \sqrt { 2 C } \varepsilon ) \leq \frac { \lambda } { 8 } + \frac { M \sqrt { 2 C } } { s _ { * } } \lambda \leq \frac { \lambda } { 4 } .
$$

Here the last inequality is exactly the second restriction in $C _ { * } ( s _ { * } , M )$ . The other hypotheses of Theorem 2 hold by (27), the local Hessian bound, and $\lambda \le W \le \varepsilon ^ { 2 }$ . Its full-Hessian conclusion gives $H _ { x } \succeq \lambda I / 2$ . Its relative corollary gives $( 1 - c _ { * } ) P _ { x } \prec H _ { x } \prec ( 1 + c _ { * } ) P _ { x }$ with $c _ { * } = ( 1 + \sqrt { 2 } ) / 8$ For completeness, the relative proof uses $\lVert H _ { x } - P _ { x } \rVert ^ { 2 } \leq M ^ { 2 } ( 2 C \lambda \varepsilon ^ { 2 } ) \leq s _ { * } ^ { 2 } \varepsilon ^ { 2 } \lambda / 6 4$ and the absolute tangent correction bound $\lambda / 4 ;$ these are precisely the relative-envelope premises. In particular they apply to each global minimizer.

The indefinite point for the same center and label. We use the geometric-persistence construction with the selected $y = y ( v ) , m = \sigma _ { n } ( D G ( y ) )$ , and $B = B _ { y }$ to make clear that it uses the same $d , Y , \lambda$ . Since $p > n$ and $B > 0$ , choose a unit $t \in \ker D G ( y )$ with $b = \Pi _ { y } ( t , t )$ and $\| b \| = B$ The norm is attained on a diagonal tangent because a scalar symmetric bilinear form has the same bilinear and diagonal operator norms, after dualizing the vector output. Put

$$
w = \frac { 2 b } { B ^ { 2 } } , \qquad r _ { \ast } = \lambda ( D G ( y ) ^ { \dagger } ) ^ { T } w , \qquad q = \frac { 4 \lambda } { m ^ { 2 } B } , \qquad D = L m + 8 M ^ { 2 } .
$$

Then $D G ( y ) ^ { T } r _ { * } = \lambda w , \| r _ { * } \| \leq 2 \lambda / ( m B )$ , and the tangent identity $D ^ { 2 } G ( y ) [ t , t ] = - D G ( y ) b$ yields

$$
\langle r _ { * } , D ^ { 2 } G ( y ) [ t , t ] \rangle = - 2 \lambda .\tag{30}
$$

Because $m \ge m _ { 0 } , B \ge b _ { 0 }$ , and $m \mapsto m ^ { 4 } / ( L m + 8 M ^ { 2 } )$ is increasing, the six bounds in (6) imply

$$
\begin{array} { c c c } { { \displaystyle \frac { M q } { m } \le \frac { 1 } { 3 2 } , ~ } } & { { ~ \displaystyle q \le \frac { m ^ { 2 } B } { 1 6 D } , ~ } } & { { ~ B q \le \frac { 1 } { 1 6 } , } } \\ { { \displaystyle \frac { 2 \lambda } { m ^ { 2 } B ^ { 2 } } \le \frac { C \varepsilon ^ { 2 } } { 8 } , ~ } } & { { ~ \displaystyle q \le \frac { \sqrt { C } \varepsilon } { 4 } , ~ } } & { { ~ \displaystyle q \le \frac { R } { 4 } . } } \end{array}\tag{31}
$$

On the q-ball in (ker $D G ( y ) ) ^ { \perp }$ , the map

$$
h \longmapsto D G ( y ) ^ { \dagger } ( r _ { * } - G ( y + h ) + G ( y ) + D G ( y ) h )
$$

has center norm at most $q / 2$ and Lipschitz constant at most $M q / m \leq 1 / 3 2$ . It has a fixed point h with $\| h \| < q$ and $G ( y + h ) - G ( y ) = r _ { * }$ exactly. Set $x _ { \lambda } = y + h$ . Thus $G ( x _ { \lambda } ) - Y = r _ { * } ,$ , and $\| x _ { \lambda } - x _ { 0 } \| < R / 4 + R / 4 = R / 2$ . Also $\| D G ( x _ { \lambda } ) - D G ( y ) \| \le M q \le m / 3 2$ , so $\sigma _ { n } ( D G ( x _ { \lambda } ) ) \ge 3 1 m / 3 2$ The exact residual, (26), and (31) give

$$
\frac { Q ( x _ { \lambda } ) } { \lambda } \leq \frac { 2 \lambda } { m ^ { 2 } B ^ { 2 } } + \frac { 1 } { 2 } \big ( q + \| y - d \| \big ) ^ { 2 } \leq \left( \frac { 1 } { 8 } + \frac { 9 } { 1 2 8 } \right) C \varepsilon ^ { 2 } = \frac { 2 5 } { 1 2 8 } C \varepsilon ^ { 2 } < C \varepsilon ^ { 2 } .
$$

Hence $x _ { \lambda }$ belongs to the same low-cost set that contains the global minimizer.

For its gradient, subtract the exact anchor force:

$$
\nabla Q ( x _ { \lambda } ) - \lambda w = \left( D G ( x _ { \lambda } ) - D G ( y ) \right) ^ { T } r _ { * } + \lambda ( x _ { \lambda } - d ) .
$$

The elementary $B _ { 0 } \le M / \sigma$ and persistence give $B \| y - d \| \leq ( 2 M / ( s \varepsilon ) ) \delta \varepsilon \leq 1 / 8$ . Therefore

$$
\frac { B } { \lambda } \| \nabla Q ( x _ { \lambda } ) - \lambda w \| \leq \frac { 2 M q } { m } + B q + B \| y - d \| \leq \frac { 1 } { 1 6 } + \frac { 1 } { 1 6 } + \frac { 1 } { 8 } = \frac { 1 } { 4 } , \qquad \frac { 7 } { 4 } \leq \frac { B \| \nabla Q ( x _ { \lambda } ) \| } { \lambda } \leq \frac { 9 } { 4 } .\tag{32}
$$

To check the complete Hessian signs, let $t _ { x }$ be the normalized projection of t onto ker $D G ( x _ { \lambda } )$ The kernel perturbation bound yields $\lVert t _ { x } - t \rVert \leq 4 M q / m$ and hence

$$
\| D ^ { 2 } G ( x _ { \lambda } ) [ t _ { x } , t _ { x } ] - D ^ { 2 } G ( y ) [ t , t ] \| \leq ( L + 8 M ^ { 2 } / m ) q = D q / m .
$$

Equations (30)–(31) make its tangent quadratic form at most

$$
t _ { x } ^ { T } H _ { x _ { \lambda } } t _ { x } \leq - \lambda + \frac { 2 \lambda D q } { m ^ { 2 } B } \leq - \frac { 7 } { 8 } \lambda < 0 .
$$

For any unit $n _ { x } \in ( \ker D G ( x _ { \lambda } ) ) ^ { \perp } , \lVert D G ( x _ { \lambda } ) n _ { x } \rVert \ge 3 1 m / 3 2$ . The first cap term gives $M \| r _ { * } \| \leq m ^ { 2 } / 6 4$ whence

$$
n _ { x } ^ { T } H _ { x _ { \lambda } } n _ { x } \geq ( 3 1 m / 3 2 ) ^ { 2 } - m ^ { 2 } / 6 4 + \lambda > 0 .
$$

Both calculations include the residual-weighted second derivatives.

Finally, the full-row-rank pseudoinverse and kernel-projection perturbation bounds at Jacobian distance $M q \leq m / 3 2 { \mathrm { ~ g i v e ~ } } \| D G ( x _ { \lambda } ) ^ { \dagger } - D G ( y ) ^ { \dagger } \| \leq 3 M q / m ^ { 2 }$ and matched unit tangents within $4 M q / m$ . Applying these in both directions to the curvature norm, with $\| D G ( x _ { \lambda } ) ^ { \dagger } \| \le 2 / m$ , yields

$$
| B _ { x _ { \lambda } } - B | \leq \left( \frac { 2 L } { m } + \frac { 1 9 M ^ { 2 } } { m ^ { 2 } } \right) q \leq \frac { 2 L m + 1 9 M ^ { 2 } } { 1 6 ( L m + 8 M ^ { 2 } ) } B \leq \frac { 1 9 } { 1 2 8 } B < \frac { 1 } { 4 } B .
$$

The last ratio is maximized at $L m = 0$ . Thus $3 B / 4 \le B _ { x _ { \lambda } } \le 5 B / 4$ . Multiplying these anchorcoupled bounds by (32) gives $2 1 / 1 6 \le A ( x _ { \lambda } ) \le 4 5 / 1 6$ . Converting each estimate separately through $B _ { 0 } / 2 \le B \le 2 B _ { 0 }$ gives the two stated B<sub>0</sub>-relative intervals. This completes the common-domain assertions. □

## B.2 A near-flat family

The assumptions have a near-flat example. For $G _ { \kappa } ( x , y ) = x ^ { 2 } + \kappa y ^ { 2 }$ at $x _ { 0 } = ( \varepsilon , 0 )$ , take $0 < \kappa \leq 1 / 4$ $s = M = 2 , L = 0 , r = 1 / 2 , R = \varepsilon / 2$ , and $C = 1 / 5 1 2$ . Then $\alpha = \beta = 1 / ( 2 5 6 \sqrt { 2 } )$ for every κ, $\sigma B _ { 0 } = 2 \kappa  0$ , and $W = \kappa ^ { 2 } \varepsilon ^ { 2 } / 3 2 7 6 8$ . The curvature premise holds along $y ( v ) = ( \sqrt { \varepsilon ^ { 2 } + v } , 0 )$ Taking $u = 0$ and $v = \beta \varepsilon ^ { 2 } / 2$ gives $Q ( d ) > 0$ . Thus the passing minimizer in the theorem is not supplied by an exact center fit. The ridge window shrinks as this family becomes flatter.

A nonempty near-flat family and a nonexact center. For $G _ { \kappa } ( x , y ) = x ^ { 2 } + \kappa y ^ { 2 } \mathrm { ~ a t ~ } x _ { 0 } = ( \varepsilon , 0 )$ direct diferentiation gives $\sigma = 2 \varepsilon , B _ { 0 } = \kappa / \varepsilon , \| D ^ { 2 } G _ { \kappa } \| = 2$ , and $L = 0$ . With the constants in the claim, $s _ { * } = 1 , C _ { * } ( 1 , 2 ) = 1 / 5 1 2$ , and

$$
\delta = \frac { 1 } { 1 2 8 \sqrt { 2 } } , \qquad \alpha = \beta = \frac { 1 } { 2 5 6 \sqrt { 2 } } , \qquad \alpha + \sqrt { 2 C } < \frac { 1 } { 2 } = R / \varepsilon .
$$

The normal section is $y ( v ) = ( \sqrt { \varepsilon ^ { 2 } + v } , 0 )$ and $B _ { y ( v ) } / B _ { 0 } = ( 1 + v / \varepsilon ^ { 2 } ) ^ { - 1 / 2 } \in [ 1 / 2 , 2 ]$ for $| v | \leq \beta \varepsilon ^ { 2 }$ The six entries of $\Lambda / \varepsilon ^ { 2 }$ are, in order,

$$
{ \frac { \kappa } { 5 1 2 } } , \quad { \frac { \kappa ^ { 2 } } { 8 1 9 2 } } , \quad { \frac { 1 } { 6 4 } } , \quad { \frac { \kappa ^ { 2 } } { 3 2 7 6 8 } } , \quad { \frac { \kappa } { 3 2 { \sqrt { 5 1 2 } } } } , \quad { \frac { \kappa } { 6 4 } } .
$$

For $0 < \kappa \leq 1 / 4$ their minimum, also below one, is $\kappa ^ { 2 } / 3 2 7 6 8$ . Thus W has the claimed value, while σ $\cdot B _ { 0 } = 2 \kappa \downarrow 0$ . The admissible choice $u = 0 , v = \beta \varepsilon ^ { 2 } / 2$ has $Q ( d ) = v ^ { 2 } / 2 = \beta ^ { 2 } \varepsilon ^ { 4 } / 8 > 0$ □

## B.3 Two boundaries of curvature persistence

The next two constructions have separate scopes. The first shows why the residual direction may need to change with the label. The second shows why common smoothness bounds alone cannot give a fixed normalized label radius when curvature vanishes.

Proposition 10 (Rotating curvature). Fix $C , r > 0$ and a tube-radius parameter $\rho > 0$ . Set

$$
b = \operatorname* { m i n } \{ 1 / 1 6 , \sqrt { C } / 1 6 , r / 8 , \rho / 8 \} , \quad k = \pi / b , \quad a ( s ) = ( \cos k s , \sin k s ) .
$$

For $0 < \varepsilon \le 1$ let

$$
G _ { \varepsilon } ( t , z ) = \varepsilon z + \frac { \varepsilon ^ { 2 } t ^ { 2 } } { 2 } a ( z _ { 2 } / \varepsilon ) , \quad z \in \mathbb { R } ^ { 2 } , \quad x _ { 0 } = 0 , \quad R = r \varepsilon .
$$

These maps have common local $C ^ { 2 , 1 }$ bounds. On the exact-fit section $t = 0 , D G _ { \varepsilon } = ( 0 , \varepsilon I _ { 2 } )$ and $B = \varepsilon ~ f o r$ every z, although the curvature vector rotates by π between $z _ { 2 } = 0$ and $z _ { \mathrm { 2 } } = b \varepsilon$ . For every $\| u \| \leq b \varepsilon , \| v \| \leq b \varepsilon ^ { 2 }$ , and

$$
0 < \lambda \leq \operatorname* { m i n } \{ C \varepsilon ^ { 6 } / 1 6 , \varepsilon ^ { 4 } / ( 4 k ) \} ,
$$

the objective with $d = u$ and $Y = v$ has a strict low-cost witness in $B ( 0$ , min $\{ r , \rho \} \varepsilon )$ with

$$
\frac { 3 \lambda } { 2 B _ { 0 } } \leq \| \nabla Q \| \leq \frac { 5 \lambda } { 2 B _ { 0 } } , \quad B = B _ { 0 } = \varepsilon ,
$$

and complete Hessian eigenvalues $\{ - \lambda , \varepsilon ^ { 2 } + \lambda , \varepsilon ^ { 2 } + \lambda \}$ . At the allowed label $v = ( 0 , b \varepsilon ^ { 2 } )$ , however, the fixed base residual $r _ { 0 } = ( - 2 \lambda / \varepsilon ^ { 2 } , 0 )$ gives a strict low-cost section point with positive Hessian eigenvalues $\{ 3 \lambda , \varepsilon ^ { 2 } + \lambda , \varepsilon ^ { 2 } + \lambda \}$

Proof. Write $F ( s ) = s _ { t } ^ { 2 } a ( s _ { 2 } ) / 2$ . The nonlinear part of $G _ { \varepsilon }$ equals $\varepsilon ^ { 4 } F ( x / \varepsilon )$ , so its second and third derivatives on $B ( 0 , ( r + 1 ) \varepsilon )$ are bounded by $\varepsilon ^ { 2 } K _ { 2 }$ and $\varepsilon K _ { 3 }$ , where $K _ { j } = \operatorname* { s u p } _ { \| s \| \leq r + 1 } \left\| D ^ { j } F ( s ) \right\| < \infty$ Taking $M = \operatorname* { m a x } \{ 1 , K _ { 2 } \}$ and $L = \operatorname* { m a x } \{ 1 , K _ { 3 } \}$ proves the uniform local bounds. $\mathrm { A t } ~ t = 0$ the only nonzero second derivative is $D ^ { 2 } G [ e _ { t } , e _ { t } ] = \varepsilon ^ { 2 } a ( z _ { 2 } / \varepsilon )$ . Thus $\operatorname { I I } ( e _ { t } , e _ { t } ) = ( 0 , - \varepsilon a ( z _ { 2 } / \varepsilon ) )$ and $B = \varepsilon$

Put $q = 2 \lambda / \varepsilon ^ { 3 }$ and $y = v / \varepsilon$ . The map $T ( s ) = y _ { 2 } - q \sin ( k s / \varepsilon )$ is a contraction on R since $q k / \varepsilon \leq 1 / 2$ . Let $z _ { 2 }$ be its fixed point and set $z _ { 1 } = y _ { 1 } - q \cos ( k z _ { 2 } / \varepsilon )$ . Then $x = ( 0 , z )$ has the exact residual

$$
G _ { \varepsilon } ( x ) - v = - { \frac { 2 \lambda } { \varepsilon ^ { 2 } } } a ( z _ { 2 } / \varepsilon ) .
$$

The ridge cap gives $q \leq \varepsilon / ( 2 k ) < b \varepsilon$ , so $\| x \| < 2 b \varepsilon < \operatorname* { m i n } \{ r , \rho \} \varepsilon$ and $\| x - u \| < 3 b \varepsilon$ . Hence

$$
\frac { Q ( x ) } { \lambda \varepsilon ^ { 2 } } < \frac { C } { 8 } + \frac { 9 b ^ { 2 } } { 2 } \leq \left( \frac { 1 } { 8 } + \frac { 9 } { 5 1 2 } \right) C < C .
$$

At a section point the normal gradient term has norm $2 \lambda / \varepsilon ;$ the ridge-gradient error is below 3bλε. This gives the displayed gradient bounds. The residual-weighted Hessian correction acts only in the t direction and equals −2λ there. Adding the Gauss–Newton and ridge terms gives the three eigenvalues.

For the last assertion take $v = ( 0 , b \varepsilon ^ { 2 } )$ and $x _ { F } = ( 0 , - q , b \varepsilon )$ . Here $a ( b ) = ( - 1 , 0 ) , G _ { \varepsilon } ( x _ { F } ) - v = r _ { 0 }$ and the tangent residual correction is +2λ. The same cost estimate, with $u = 0 .$ , proves strict low cost, and the displayed positive eigenvalues follow. Thus constant curvature magnitude does not preserve the sign generated by a fixed base residual. □

Proposition 11 (A compact-bump boundary). Let $\chi ( q ) = ( 1 - q ^ { 2 } ) ^ { 4 } ~ f o r ~ | q | < 1$ and zero otherwise, and let $\phi ( t , z ) = t ^ { 2 } \chi ( t ) \chi ( z ) / 2$ . Fix $C , r > 0$ and a tube-radius parameter $\rho > 0$ . For $0 < \varepsilon \le 1$ define

$$
G _ { \varepsilon } ( t , z ) = \varepsilon z + \varepsilon ^ { 6 } \phi ( t / \varepsilon ^ { 2 } , z / \varepsilon ^ { 2 } ) , \quad x _ { 0 } = d = 0 , \quad R = r \varepsilon .
$$

These maps have fixed global $C ^ { 2 , 1 }$ bounds, $\sigma = \varepsilon$ and $B _ { 0 } = \varepsilon$ . For every $\beta > 0$ and all suficiently small $\varepsilon ,$ the label $Y = 3 \varepsilon ^ { 3 }$ lies in $| Y - G _ { \varepsilon } ( x _ { 0 } ) | \leq \beta \varepsilon ^ { 2 }$ . For every $0 < \lambda \leq \varepsilon ^ { 4 } / ( 8 C )$ , the low-cost set is nonempty, but its complete Hessian is positive definite at every point. Thus no fixed positive normalized label radius follows from the original common smoothness bounds alone, even $i f$ the positive ridge cap is decreased.

Proof. The cutof has three continuous derivatives at $\pm 1$ , so $\phi \in C ^ { 3 }$ with bounded second and third derivatives. The chain rule gives $D ^ { 2 } G _ { \varepsilon } = \varepsilon ^ { 2 } D ^ { 2 } \phi ( x / \varepsilon ^ { 2 } )$ and $D ^ { 3 } G _ { \varepsilon } = D ^ { 3 } \phi ( x / \varepsilon ^ { 2 } )$ . Thus one can take common positive bounds $M \geq \operatorname* { s u p } \left\| D ^ { 2 } \phi \right\|$ and $L \ge \operatorname* { s u p } \left\| D ^ { 3 } \phi \right\|$ . At the origin, $D G = ( 0 , \varepsilon )$ and $D ^ { 2 } G [ e _ { t } , e _ { t } ] = \varepsilon ^ { 2 }$ , giving $\sigma = B _ { 0 } = \varepsilon$

On the square $S _ { \varepsilon } = [ - \varepsilon ^ { 2 } , \varepsilon ^ { 2 } ] ^ { 2 }$ that contains the curvature support, $| G _ { \varepsilon } | \le \varepsilon ^ { 3 } + \varepsilon ^ { 6 } / 2 \le 3 \varepsilon ^ { 3 } / 2$ hence $| G _ { \varepsilon } - Y | \ge 3 \varepsilon ^ { 3 } / 2$ . Every low-cost point for the stated ridge range instead has

$$
| G _ { \varepsilon } - Y | \leq \sqrt { 2 C \lambda } \varepsilon \leq \varepsilon ^ { 3 } / 2 .
$$

The low-cost set avoids the whole square. Outside it, $G _ { \varepsilon } ( t , z ) = \varepsilon z$ locally, so its complete Hessian is $\mathrm { d i a g } ( \lambda , \varepsilon ^ { 2 } + \lambda ) \succ 0$ . The exact fit $( 0 , 3 \varepsilon ^ { 2 } )$ has $Q = 9 \lambda \varepsilon ^ { 4 } / 2 < C \lambda \varepsilon ^ { 2 }$ when $\varepsilon < \sqrt { C } / 3$ , proving nonemptiness. Also $3 \varepsilon ^ { 3 } \leq \beta \varepsilon ^ { 2 }$ when $\varepsilon \le \beta / 3$ . Every proposed positive ridge cap contains some $\lambda$ in the stated range, so shrinking it cannot restore an all-ridges indefinite-witness conclusion. □

## C Prior certificate comparisons

## C.1 Squared-distance curvature scale

This calculation follows from the nearest-projection formulas of Leobacher and Steinicke [7, Theorem C and the derivative of squared distance on p. 16]. Let $\mathcal { M }$ be a smooth manifold with a singlevalued diferentiable nearest-point map p in a normal neighborhood. For $g ( x ) = \mathrm { d i s t } ( x , { \mathcal M } ) ^ { 2 } / 2$ $\nabla g ( x ) = x - p ( x )$ and $\nabla ^ { 2 } g ( x ) = I - D p ( x )$ . At $x = y + t \nu$ with $y \in \mathcal { M }$ and a unit normal $\nu ,$ suppose a principal tangent direction has signed curvature $\kappa > 0$ under the source’s shape-operator convention. Theorem C gives the projection derivative eigenvalue $( 1 - t \kappa ) ^ { - 1 }$ on that direction. Thus

$$
\lambda _ { \mathrm { t a n g e n t } } ( \nabla ^ { 2 } g ( x ) ) = 1 - { \frac { 1 } { 1 - t \kappa } } = - { \frac { t \kappa } { 1 - t \kappa } } .
$$

If the ridge center is $y ,$ the objective $g ( x ) + \lambda \left\| x - y \right\| ^ { 2 } / 2$ has tangent eigenvalue $\lambda - t \kappa / ( 1 - t \kappa )$ and gradient magnitude $( 1 + \lambda ) t$ . For example, with $0 < \lambda < 1 / 4$ and $t = 2 \lambda / \kappa$ inside the projection neighborhood, the tangent eigenvalue is negative and the gradient has order $\lambda / \kappa .$ . This is an inference for squared distance to a fixed manifold. It does not establish a theorem for an arbitrary raw residual $\| G - Y \| ^ { 2 } / 2$ or for varying center and label.

## C.2 Localized distance-gradient comparison

Let y ∈ $y \in S$ be a nearest point to a localized $x ,$ and set $u = x - y$ . For a smooth embedded $S ,$ u is normal at $y .$ . Since $\nabla g ( y ) = 0$ , integration along the normal segment gives

$$
\langle \nabla g ( x ) , u \rangle = \int _ { 0 } ^ { 1 } u ^ { T } \nabla ^ { 2 } g ( y + t u ) u d t \geq ( c / 2 ) \left\| u \right\| ^ { 2 } .
$$

Table 1: Comparison of specified suficient conditions. Failure of a row’s test is not failure of every method in the cited work. The example also admits a directional cost-only check if its observed cost is used; it does not show that gradient information is necessary.
<table><tr><td>Sufficient test</td><td>Required information</td><td>Guarantee</td><td>At Proposition  $\mathrm { 6 3 }$  point</td></tr><tr><td>Full residual norm [9, Proposition 5]</td><td> $\| E \| < \lambda ,$  with ridge and residual correction in the same normalization. Hessian with</td><td>Positive full margin  $\lambda - \| E \|$ </td><td>Fails:  $\| E \| / \lambda =$   $1 / ( 1 6 a ) \geq 4 .$ </td></tr><tr><td>Distance tube [4, Lemma E.1]</td><td>Source Assumptions 1 and 3: a compact minimizer manifold, local PL regularity, negative-definite Hessians at exterior critical points, and a bounded third derivative on a tube. The point must lie within the</td><td> $H \succeq \lambda I / 2$  inside that sufficient tube.</td><td>Fails both radii: dist  $/ r _ { \mathrm { r e f i n e d } } >$   $3 / ( 8 a )$ </td></tr><tr><td>Pointwise directional test Theorem 2</td><td>stated or proof-refined radius. Cost, Jacobian rank, second-derivative bound, and the tangent curvature gate (7).</td><td> $H \succeq \lambda I / 2$  relative bound in Corollary 3.</td><td>and the Passes with  $C = 1 / 5 1 2 ,$   $s = 1 , M = 2 .$ </td></tr></table>

Cauchy–Schwarz and $\nabla g = \nabla Q - \lambda ( x - d )$ give (14); the case $u = 0$ is immediate. One localization route under the additional global growth premise of Masiha et al. [4, Lemma E.7, (16)] is $\sqrt { 2 C \lambda \varepsilon ^ { 2 } / \mu _ { Q G } } \leq r _ { \mathrm { l o c } }$ . If $d \in S$ , another is $\sqrt { 2 C } \varepsilon \leq r _ { \mathrm { l o c } }$ , because dist $( x , S ) \leq \| x - d \|$ . Neither route follows from the gradient inequality itself. Under the source’s tube hypotheses, (14) places x in Lemma E.1’s literal radius if

$$
\lVert \nabla Q ( x ) \rVert + \lambda \sqrt { 2 C } \varepsilon \leq \frac { c } { 2 } \operatorname* { m i n } \{ \rho , \lambda / ( 2 \operatorname* { m a x } \{ L _ { g , 3 } , 1 \} ) \} .
$$

For $L _ { g , 3 } > 0$ , the proof of that lemma compares Hessians by $L _ { g , 3 } \operatorname { d i s t } ( x , S )$ . The same proof therefore permits $L _ { g , 3 }$ in place of max $\{ L _ { g , 3 } , 1 \}$ in this condition. This refinement is derived from the proof, not part of its displayed lemma statement.

## C.3 Comparison of suficient certificates

Table 1 compares three suficient tests at the point in Proposition 6. All quantities in the table use the raw objective (2). In the averaged notation of Kharel et al. [9, Proposition 5], $\hat { \Delta } = E / n$ and $\lambda _ { \mathrm { m e a n } } = \lambda / n$ . Thus their margin condition is exactly $\| E \| < \lambda$ after rescaling. We compare this matrix condition, not their generalization theorem.

## C.4 Proof of the negative-residual comparison

Use the parameters of Proposition 6 and put

$$
t = { \frac { a \varepsilon ^ { 2 } } { 3 2 } } \leq 2 ^ { - 2 1 } , \qquad h = \varepsilon - X = { \frac { \lambda } { 3 2 a ( \varepsilon + X ) } } .
$$

Then $X = \varepsilon \sqrt { 1 - t } > \varepsilon / 2$ and $0 < h \leq a \varepsilon ^ { 3 } / 3 2$

Cost, rank, and the gradient test. Direct diferentiation at z gives

$$
\begin{array} { c c c c } { \displaystyle { r = - \frac { \lambda } { 3 2 a } , \qquad } } & { \qquad J = ( 2 X , 0 ) , } & { \qquad } & { \left\| D ^ { 2 } G _ { a } \right\| = 2 , } \\ { \displaystyle K = 2 a , \qquad } & { B = \frac { a } { X } , } & { \qquad } & { b = \left( - \lambda \left[ \frac { X } { 1 6 a } + h \right] , 0 \right) . } \end{array}
$$

The normalized cost satisfies

$$
\frac { Q ( z ) } { \lambda \varepsilon ^ { 2 } } = \frac { \varepsilon ^ { 2 } } { 2 0 4 8 } + \frac { h ^ { 2 } } { 2 \varepsilon ^ { 2 } } \leq \frac { \varepsilon ^ { 2 } ( 1 + a ^ { 2 } \varepsilon ^ { 2 } ) } { 2 0 4 8 } < \frac { 1 } { 5 1 2 } .
$$

Also $\sigma _ { 1 } ( J ) = 2 X > \varepsilon , \lambda \leq \varepsilon ^ { 2 }$ , and $C _ { * } ( 1 , 2 ) = 1 / 5 1 2$ . The gradient obeys

$$
\frac { B \left. b \right. } { \lambda } = \frac { 1 } { 1 6 } + \frac { a h } { X } \leq \frac { 1 } { 1 6 } + \frac { a ^ { 2 } \varepsilon ^ { 2 } } { 1 6 } < \frac { 1 } { 8 } .
$$

Thus the gradient alternative in Theorem 2 applies with the stated common constants. One can also check its full gate: since $\sqrt { 2 C } = 1 / 1 6$ and $\varepsilon / X < 2 $

$$
\frac { B ( \| b \| + \lambda \sqrt { 2 C } \varepsilon ) } { \lambda } = \frac { 1 } { 1 6 } + \frac { a h } { X } + \frac { a \varepsilon } { 1 6 X } < \frac { 1 } { 1 6 } + \frac { a ^ { 2 } \varepsilon ^ { 2 } } { 1 6 } + \frac { a } { 8 } < \frac { 1 } { 4 } .
$$

The complete Hessian. The residual correction and full Hessian are

$$
E = - { \frac { \lambda } { 1 6 a } } \mathrm { d i a g } ( 1 , a ) , \qquad H = \mathrm { d i a g } \left( 4 \varepsilon ^ { 2 } - { \frac { 3 \lambda } { 1 6 a } } + \lambda , { \frac { 1 5 \lambda } { 1 6 } } \right) .
$$

The first eigenvalue is greater than $3 \varepsilon ^ { 2 }$ , and the second is $1 5 \lambda / 1 6 .$ . Hence $H \succeq \lambda I / 2$ . Although $E \prec 0$ , its tangent entry is only $- \lambda / 1 6$ ; the larger negative entry lies in the normal direction. There is no mixed block in this example.

Distance to the ellipse. For any $( u , v ) \in S , u \in [ - \varepsilon , \varepsilon ]$ and $v ^ { 2 } = ( \varepsilon ^ { 2 } - u ^ { 2 } ) / a$ . Consequently

$$
\left\| ( X , 0 ) - ( u , v ) \right\| ^ { 2 } - h ^ { 2 } = 2 X ( \varepsilon - u ) + ( a ^ { - 1 } - 1 ) ( \varepsilon ^ { 2 } - u ^ { 2 } ) \geq 0 .
$$

Equality requires $u = \varepsilon$ and $v = 0$ . Thus d is the unique nearest point even though z lies inside the ellipse, and dist $( z , S ) = h$

The source hypotheses and the two tube radii. For each fixed positive $a , \varepsilon _ { \mathrm { { i } } }$ , the set $S$ is a compact, connected smooth ellipse. The only critical points of $g = ( x ^ { 2 } + a y ^ { 2 } - \varepsilon ^ { 2 } ) ^ { 2 } / 2$ are $S$ and the origin. At the origin, $\nabla ^ { 2 } g = - 2 \varepsilon ^ { 2 } \mathrm { d i a g } ( 1 , a ) \prec 0$ , so S is the only component of local minima. In the shell $G _ { a } \in [ \varepsilon ^ { 2 } / 2 , 3 \varepsilon ^ { 2 } / 2 ]$ ，

$$
\| \nabla G _ { a } \| ^ { 2 } = 4 ( x ^ { 2 } + a ^ { 2 } y ^ { 2 } ) \geq 4 a ( x ^ { 2 } + a y ^ { 2 } ) \geq 2 a \varepsilon ^ { 2 } , \qquad \| \nabla g \| ^ { 2 } = 2 g \| \nabla G _ { a } \| ^ { 2 } \geq 4 a \varepsilon ^ { 2 } g .
$$

These facts give the local Polyak–Lojasiewicz and minimizer-set conditions in Assumption 1 of Masiha et al. [4] for a singleton parameter set. Polynomial smoothness and compactness give finite third-derivative bounds on each fixed finite tube, as required by their Assumption 3. Their Lemma E.1 uses these two assumptions. The Hessian of the quartic g is unbounded globally, so their global smoothness Assumption 4 does not hold. We do not invoke their full Lemma E.7.

On S, the normal eigenvalue of $\nabla ^ { 2 } g$ is $\| \nabla G _ { a } \| ^ { 2 }$ . Its minimum is $4 a \varepsilon ^ { 2 }$ , attained at $( 0 , \pm \varepsilon / \sqrt { a } )$ Since $d \in S$ , every admissible third-derivative bound on a tube containing S satisfies

$$
L _ { g , 3 } \geq | \partial _ { x } ^ { 3 } g ( d ) | = 1 2 \varepsilon .
$$

The proof of Masiha et al. [4, Lemma E.1] bounds the Hessian change by $L _ { g , 3 } \operatorname { d i s t } ( z , S )$ . For $L _ { g , 3 } > 0$ , it therefore permits the refined radius stated above. For every admissible choice of the tube constants,

$$
r _ { \mathrm { r e f i n e d } } \leq \frac { \lambda } { 2 4 \varepsilon } , \qquad \frac { h } { r _ { \mathrm { r e f i n e d } } } \geq \frac { 3 \varepsilon } { 4 a ( \varepsilon + X ) } > \frac { 3 } { 8 a } .
$$

The literal radius is at most $\lambda / 2$ , and

$$
\frac { h } { r _ { \mathrm { l i t e r a l } } } \geq \frac { 1 } { 1 6 a ( \varepsilon + X ) } > \frac { 1 } { 3 2 a \varepsilon } > 1 .
$$

This proves the two distance claims. For completeness, the normal Hessian along the nearest-point segment also has a positive lower bound: for $x \in [ X , \varepsilon ]$

$$
\partial _ { x } ^ { 2 } g ( x , 0 ) = 6 x ^ { 2 } - 2 \varepsilon ^ { 2 } \geq 4 \varepsilon ^ { 2 } - \frac { 3 a \varepsilon ^ { 4 } } { 1 6 } \geq \frac { 1 } { 2 } ( 4 a \varepsilon ^ { 2 } ) .
$$

This direct check concerns that segment; it is not an application of the global conclusion of Lemma E.7.

What the comparison shows. The absolute residual-norm condition of Kharel et al. [9, Proposition 5] fails because $\| E \| / \lambda = 1 / ( 1 6 a ) \geq 4$ . Their matrices use sample averages: for the raw objective, $E = n { \widehat { \Delta } }$ and $\lambda = n \lambda _ { \mathrm { m e a n } }$ , so the condition becomes exactly $\| E \| < \lambda$ . Here $n = 1$ We compare this suficient matrix condition, not the fitted-point assumptions or generalization conclusion of that paper.

With the fixed cap $C = 1 / 5 1 2$ , the cost branch displayed in Theorem 2 fails:

$$
\frac { K \sqrt { 2 C \lambda } \varepsilon } { \lambda / 4 } = \frac { 1 } { 2 \varepsilon } \geq 1 6 .
$$

This does not make gradient information necessary. Set $C _ { z } = Q ( z ) / ( \lambda \varepsilon ^ { 2 } ) < C$ . A sharper directional test using this observed cost passes, since

$$
\left( { \frac { K { \sqrt { 2 Q ( z ) } } } { \lambda } } \right) ^ { 2 } = { \frac { 1 } { 2 5 6 } } + { \frac { 4 a ^ { 2 } h ^ { 2 } } { \lambda } } \leq { \frac { 1 + a ^ { 2 } \varepsilon ^ { 2 } } { 2 5 6 } } < { \frac { 1 } { 1 6 } } .
$$

All other pointwise assumptions are unchanged when C is replaced by $C _ { z }$ . In particular, this is not a separation from all cost-based certificates.

$a \downarrow 0$ at fixed ε, the ratios $\| E \| / \lambda$ and $h / r _ { \mathrm { r e f i n e d } }$ diverge, but both $\| E \|$ and h tend to zero. The ellipse’s long axis $\varepsilon / \sqrt { a }$ diverges and its minimum normal gap $4 a \varepsilon ^ { 2 }$ vanishes. The source comparison holds for each fixed positive pair; it does not assert a uniform compact family or a uniform positive gap through $a = 0$

An exact relative check. This example also permits a direct calculation with $P = J ^ { T } J + \lambda I \colon$

$$
P = \mathrm { d i a g } ( 4 X ^ { 2 } + \lambda , \lambda ) , ~ P ^ { - 1 / 2 } { \cal E } P ^ { - 1 / 2 } = \mathrm { d i a g } \left( - \zeta , - \frac { 1 } { 1 6 } \right) , ~ \zeta = \frac { \lambda } { 1 6 a ( 4 X ^ { 2 } + \lambda ) } < \frac { 1 } { 1 6 } .
$$

The last inequality follows from $\lambda < a ( 4 X ^ { 2 } + \lambda )$ under the stated parameter bounds. Hence the relative norm is exactly $1 / 1 6$ and the preconditioned condition number is $( 1 - \zeta ) / ( 1 5 / 1 6 ) < 1 6 / 1 5$ These are pointwise calculations for this example, with no iteration or runtime conclusion.

## D Structural bridge and analytic examples

## D.1 Proof of the structural bridge

All constants denoted by $L _ { * }$ below are finite and depend only on the fixed reference data and preliminary compact neighborhoods. They can increase between estimates, but never depend on $G , \varepsilon , u , \ell$ or λ. We establish rank from cost before applying the curvature test.

Fixed regular patch and moving weak coordinates. Write $k = n - g , m = p - g$ . Choose an injection $S : \mathbb { R } ^ { k }  \mathbb { R } ^ { m }$ with $D q ( z _ { 0 } ) S$ invertible and nested compact neighborhoods $W \Subset W _ { 1 }$ of $z _ { \mathrm { 0 } }$ on which $\sigma _ { k } ( D q ( w ) ) \geq 4 s _ { q } > 0$ . If $g > 0$ , set $X = A _ { R } ^ { T } ( A _ { R } A _ { R } ^ { T } ) ^ { - 1 }$ , so $A _ { R } X = I$ and $Z ^ { T } X = 0$ Let $J _ { 0 } = D G ( c )$ , and let $P _ { J }$ be the projector on its bottom $p - g$ right-singular subspace. For $y$ in this subspace, $\| A y \| \le \kappa ( 1 + \varepsilon ) \| y \|$ . Since A has a fixed nonzero singular gap on (ker $A ) ^ { \perp }$ , the equal-dimensional principal-angle estimate yields $\left\| P _ { J } - Z Z ^ { T } \right\| \leq L _ { * } \kappa$ . Put $Z _ { J } = P _ { J } Z$ . For small fixed $\kappa ,$

$$
\| Z _ { J } - Z \| \le L _ { * } \kappa , \qquad \| J _ { 0 } Z _ { J } \| \le \kappa \varepsilon .
$$

Both $[ X Z _ { J } ]$ and its inverse are uniformly bounded, and $J _ { 0 , R } X$ is uniformly invertible. For $g = 0$ set $Z _ { J } = Z$ and omit X and all retained equations. Define $h _ { G } = G _ { D } - L G _ { R }$ and $K _ { 0 } = D h _ { G } ( c )$ The base assumptions give

$$
\| K _ { 0 } Z _ { J } \| \le L _ { * } \kappa \varepsilon , \qquad \| K _ { 0 } X \| \le L _ { * } \kappa .\tag{33}
$$

Retained solve. For $w \in W _ { 1 }$ put $v = Z _ { J } w + \varepsilon X \beta$ and

$$
\Phi _ { R } ( \beta , w , \ell _ { R } ) = \varepsilon ^ { - 2 } [ G _ { R } ( c + \varepsilon v ) - G _ { R } ( c ) ] + \ell _ { R } .
$$

On preliminary fixed bounded label sets,

$$
\Phi _ { R } ( 0 , w , \ell _ { R } ) = \varepsilon ^ { - 1 } J _ { 0 , R } Z _ { J } w + \ell _ { R } + \int _ { 0 } ^ { 1 } ( 1 - t ) D ^ { 2 } G _ { R } ( c + t \varepsilon Z _ { J } w ) [ Z _ { J } w , Z _ { J } w ] d t
$$

is uniformly bounded. Let $\begin{array} { l l l } { { P _ { R } } } & { { = } } & { { J _ { 0 , R } X } } \end{array}$ On a fixed suficiently large $\beta$ ball, $D _ { \beta } \Phi _ { R } ~ =$ $D G _ { R } ( c + \varepsilon v ) X ~ = ~ P _ { R } + O ( M \varepsilon )$ . The map $\beta \mapsto \beta - P _ { R } ^ { - 1 } \Phi _ { R }$ equals the uniformly bounded center $- P _ { R } ^ { - 1 } \Phi _ { R } ( 0 , w , \ell _ { R } )$ plus $O ( M \varepsilon ( 1 + \| \beta \| ) )$ . It therefore maps one common closed ball into itself and is a contraction when $\varepsilon$ is small. It yields $\beta = \beta ( w , \ell _ { R } )$ with uniformly bounded norm. The ordinary implicit function theorem for each $G$ gives

$$
D _ { w } \beta = - ( D _ { \beta } \Phi _ { R } ) ^ { - 1 } \varepsilon ^ { - 1 } D G _ { R } ( c + \varepsilon v ) Z _ { J } , \qquad \| D _ { w } \beta \| \le L _ { * } .
$$

The last bound uses $J _ { 0 } Z _ { J } = { \cal O } ( \kappa \varepsilon )$ and the Hessian magnitude bound. It uses no modulus for $D ^ { 2 } G$ and no third derivative.

Reduced weak equation and interpolation. Substitute the retained solution into

$$
\Psi ( w , \ell ) = { \varepsilon } ^ { - 2 } [ h _ { G } ( c + { \varepsilon } v ( w ) ) - h _ { G } ( c ) ] + { \chi } ( \ell ) , \quad v ( w ) = { Z } _ { J } w + { \varepsilon } { X } \beta ( w , \ell _ { R } ) .
$$

Taylor’s integral formula, (33), the full Hessian bound and (18) compare this to $\Psi _ { 0 } ( w , \ell ) = \chi ( \ell ) - q ( w )$ In each Hessian integral first replace v by $Z _ { J } w$ , then $Z _ { J }$ by $Z _ { i }$ , then $\kappa _ { G }$ by $\ = \kappa _ { \ / F }$ on its two $Z$ arguments, and finally evaluate the reference tensor at $c .$ This bounds the function error by

$$
E _ { * } = L _ { * } \{ ( 1 + M ) \kappa + M \varepsilon + \omega _ { F } ( L _ { * } \varepsilon ) \} ,
$$

where $\omega _ { F }$ is the modulus of the fixed reference Hessian. The same error scale controls the derivative. For any direction $^ { a , }$

$$
D _ { w } \Psi [ a ] = \varepsilon ^ { - 1 } D h _ { G } ( c + \varepsilon v ) Z _ { J } a + D h _ { G } ( c + \varepsilon v ) X ( D _ { w } \beta [ a ] ) .
$$

The first term is $\varepsilon ^ { - 1 } K _ { 0 } Z _ { J } a + \int _ { 0 } ^ { 1 } \mathcal { K } _ { G } ( c + t \varepsilon v ) [ v , Z _ { J } a ] c$ dt and admits the same replacements. The second is bounded by $L _ { * } ( \kappa + M \varepsilon ) \left\| a \right\|$ using (33). Thus $\| \Psi - \Psi _ { 0 } \| _ { C ^ { 1 } ( W _ { 1 } ; w ) } \leq E _ { * }$ , without diferentiating any Hessian. This $C ^ { 1 }$ bound permits a fixed local contraction. Restricting to $w = z _ { 0 } + S t$ and using a fixed inverse of ${ - D q ( z _ { 0 } ) S }$ gives a contraction on a suficiently small fixed ball for small $E _ { * }$ and $\| \ell - \ell _ { 0 } \|$ . Its zero satisfies $\| t \| \leq L _ { * } ( E _ { * } + \| \ell - \ell _ { 0 } \| )$ . The resulting exact interpolator obeys

$$
\begin{array} { r } { \| ( \theta _ { \mathrm { i n t } } - c ) / \varepsilon - u _ { 0 } \| \le L _ { * } ( E _ { * } + R _ { \ell } + \kappa + \varepsilon ) . } \end{array}\tag{34}
$$

Every low-cost point, before using its gradient. The cost test gives

$$
\| \theta - ( c + \varepsilon u ) \| \le \sqrt { 2 C } \varepsilon , \qquad \| r \| \le \sqrt { 2 C } \varepsilon ^ { 2 } .
$$

Hence $v = ( \theta - c ) / \varepsilon$ lies within $R _ { c } + \sqrt { 2 C }$ of $u _ { 0 }$ . Decompose exactly $v = Z J w + X \alpha$ . The uniformly conditioned basis gives $\| w - z _ { 0 } \| \le L _ { * } ( R _ { c } + \sqrt { C } + \kappa )$ . Taylor expansion of the full output $G ( c + \varepsilon v ) - G ( c ) = - \varepsilon ^ { 2 } \ell + r$ gives

$$
\left\| J _ { 0 } v \right\| \leq \varepsilon \big ( \| \ell \| + \sqrt { 2 C } + ( M / 2 ) \| v \| ^ { 2 } \big ) \leq L _ { * } \varepsilon ,
$$

and hence $\| \alpha \| \leq L _ { * } \varepsilon$ . Write $\alpha = \varepsilon \beta ;$ then $\beta$ is uniformly bounded and $w \in W$ for suitably small preliminary choices. This does not use the retained exact solver at the point, nor any small-gradient condition.

Weak Schur block. Apply the fixed row operation $U = \left[ \begin{array} { c c } { { I } } & { { 0 } } \\ { { - L \ I } } & { { I } } \end{array} \right]$ and proof-coordinate basis $[ X Z _ { J } ]$ to $J { \boldsymbol { : } }$

$$
U J [ X ~ Z _ { J } ] = \biggl [ { \cal A } _ { s } \quad B _ { s } \biggr ] .
$$

Here $A _ { s } = J _ { 0 , R } X + O ( M \varepsilon )$ is uniformly invertible, $B _ { s } = O ( \varepsilon ) , C _ { w } = O ( \kappa + M \varepsilon )$ , and the same integral replacements give

$$
\left\| \varepsilon ^ { - 1 } D _ { w } + D q ( w ) \right\| \leq E _ { * } .
$$

Eliminating $C _ { w }$ replaces $D _ { w }$ by $S _ { w } = D _ { w } - C _ { w } A _ { s } ^ { - 1 } B _ { s }$ . Its scaled correction is $O ( \kappa + M \varepsilon )$ , so $\sigma _ { k } ( \varepsilon ^ { - 1 } S _ { w } ) \geq 2 s _ { q }$ for small preliminary parameters, with a fixed upper bound as well. The row, column and elimination factors and their inverses are uniformly bounded. Singular values are therefore comparable to those of the block diagonal matrix with an order-one retained block and an order-ε weak block. This proves (19), with exact counts $g , n - g .$ , including $g = 0$

Noncircular selection. First fix the regular patches, preliminary compact coordinate/label sets, and positive upper bounds on $C , R _ { c } , \kappa , \varepsilon _ { 0 }$ making the preceding estimates valid uniformly for all smaller choices. Fix the Jacobian constants $s , a _ { s } , b _ { s } , b _ { w }$ at this stage. Next choose $\bar { C }$ small enough for localization and $\sqrt { 2 \bar { C } } < 1 / 4$ . For any prescribed $C \in ( 0 , \bar { C } ]$ , subsequently shrink $\kappa ,$ choose positive $R _ { c } , R _ { \ell } .$ , and shrink $\varepsilon _ { \mathrm { 0 } } \ \mathrm { s o }$ that (34) plus $R _ { c }$ is strictly less than ${ \sqrt { 2 C } } / 2$ , while $R _ { c } + \sqrt { 2 C } < 1 / 2$ and all points and their segments to c lie strictly inside $B ( c , r _ { 0 } )$ . This gives $Q ( \theta _ { \mathrm { i n t } } ) < C \lambda \varepsilon ^ { 2 } / 4$ uniformly even on the closed center and label balls. Taking $C$ also below $C _ { * } ( s , M )$ is therefore legitimate before those final shrinkages. All universal tuple choices occur last. This completes the proof.

Connection to inverse-function theory. The conditions $q ( z _ { 0 } ) = \chi ( \ell _ { 0 } )$ and rank $D q ( z _ { 0 } ) = n - g$ make $( z _ { 0 } , 1 )$ a regular zero of $q ( z ) - t ^ { 2 } \chi ( \ell _ { 0 } )$ . For the augmented reference $\begin{array} { r } { \widetilde { F } ( h , t ) = F ( { c + h } ) + t ^ { 2 } \ell _ { 0 } , } \end{array}$ this is the regular-zero condition for second-order openness [12, Section 5, Theorems 22–24] and local inversion [14, Theorem 1]. Interpolation with fixed $t = \varepsilon$ uses the separate reduced solve above.

Avakov and Magaril-Il’yaev [15, Theorem 1] give uniform local solvability for continuous maps close to a regular reference. Arutyunov and Zhukovskiy [16, Theorem 1] give perturbation stability from a regular weighted polynomial truncation. Here the normalized reference $\Psi _ { 0 }$ is polynomial and is regular at $( z _ { 0 } , \ell _ { 0 } )$ on the selected slice $w = z _ { 0 } + S t$ . The full $C ^ { 1 }$ estimate above supplies the closeness needed for local inversion; we use a contraction to obtain the displayed bounds. The weak-jet and cost estimates also control rank at every low-cost point, including points that do not interpolate.

## D.2 A quadratic worked family

The following calculation places the certificate and obstruction in one model and makes their curvature scales explicit. The flexible tuple and exact-center tuple have diferent quantifiers.

Proposition 12 (Product quadratic). Fix $0 < C \leq 1 / 1 1 5 2 , 0 < a _ { 0 } \leq 1 / 4$ , and $R _ { c } = R _ { \ell } = \sqrt { C } / 8$ Let $\boldsymbol { u } _ { 0 } = ( 0 , 1 , 0 )$ and $\ell _ { 0 } = ( 0 , - 1 )$ . For $0 < \varepsilon \le 1 / 4 , 0 \le a \le a _ { 0 } , 0 < \lambda \le \varepsilon ^ { 2 } , \| u - u _ { 0 } \| \le R _ { c }$ , and $\| \ell - \ell _ { 0 } \| \leq R _ { \ell } .$ , set

$$
G _ { a } ( v , x , y ) = ( v , x ^ { 2 } + a y ^ { 2 } ) , \quad d = \varepsilon u , \quad Y = - \varepsilon ^ { 2 } \ell .
$$

Every whole-space low-cost point lies in $\overline { { B } } ( \varepsilon u _ { 0 } , h \varepsilon )$ , where $h = R _ { c } + \sqrt { 2 C } < 1 / 4$ . At such points the ordered singular values are 1 and $2 \sqrt { x ^ { 2 } + a ^ { 2 } y ^ { 2 } } \in [ \varepsilon , 3 \varepsilon ]$ . Every tuple has an exact interpolator with $Q < C \lambda \varepsilon ^ { 2 }$ and hence a low-cost global minimizer.

On this tube, $K \leq$ 4a and $B \leq 4 a / \varepsilon . \ I f a > 0$ , every low-cost point with $\| b \| \leq \lambda \varepsilon / ( 3 2 a )$ obeys $H \succeq \lambda I / 2 ; f o r ~ a = 0$ , cost alone sufices. At every such certified point, the increasing eigenvalues satisfy

$$
\begin{array} { c } { { \lambda / 2 \leq \lambda _ { 1 } ( H ) \leq 5 \lambda / 4 , } } \\ { { \varepsilon ^ { 2 } / 2 \leq \lambda _ { 2 } ( H ) \leq 1 1 \varepsilon ^ { 2 } , \quad \lambda _ { 3 } ( H ) = 1 + \lambda . } } \end{array}\tag{35}
$$

Every low-cost point is certified by cost alone $i f 5 1 2 C a ^ { 2 } \varepsilon ^ { 2 } \leq \lambda \leq \varepsilon ^ { 2 }$

Now take $a > 0 , u = u _ { 0 }$ , and $\ell = \ell _ { 0 }$ . For every $0 < \lambda \leq C a ^ { 2 } \varepsilon ^ { 2 }$ , a point $\theta _ { \lambda }$ exists with $Q ( \theta _ { \lambda } ) < C \lambda \varepsilon ^ { 2 } , \sqrt { 3 } \lambda \varepsilon / a \leq \| \nabla Q ( \theta _ { \lambda } ) \| \leq 2 \lambda \varepsilon / a ,$ , and indefinite full Hessian. Here $Q ( d ) = 0$ . The two displayed ridge intervals leave an unclassified factor-512 gap.

The family also meets the weak-jet premise with reference $F ( v , x , y ) = ( v , x ^ { 2 } )$ at $c = 0$ whenever $2 a \le \kappa$ . Its indexed ridge and weak bands in (35) need not be separated when λ is of order $\varepsilon ^ { 2 }$ . The calculations are in Appendix D.2.1.

## D.2.1 Proof for the product quadratic

Write $R = { \sqrt { C } } / 8$ and $h = R + \sqrt { 2 C } < 1 / 4$ . The cost cap gives $\| \theta - d \| \leq \sqrt { 2 C } \varepsilon$ . Since $\| d - \varepsilon u _ { 0 } \| \le$ $R \varepsilon$ , every low-cost point lies in the stated tube. There $x \geq ( 1 - h ) \varepsilon$ and $| y | \leq h \varepsilon$ . The Jacobian rows are orthogonal:

$$
{ \cal D } G _ { a } = \left( { 1 \atop 0 } \ : \ : { 0 \atop 2 x } \ : \ : { 0 \atop 2 a y } \right) .
$$

Thus its singular values are 1 and $2 { \sqrt { x ^ { 2 } + a ^ { 2 } y ^ { 2 } } }$ . Since $a \leq 1 / 4$ and $h < 1 / 4 , \varepsilon \leq 2 \sqrt { x ^ { 2 } + a ^ { 2 } y ^ { 2 } } \leq$ $3 \varepsilon < 1$

The point

$$
\theta _ { \mathrm { i n t } } = ( - \varepsilon ^ { 2 } \ell _ { 1 } , \varepsilon \sqrt { - \ell _ { 2 } } , 0 )
$$

matches Y exactly. The label ball ensures $- \ell _ { 2 } > 0$ . The three coordinate diferences from d are bounded by $( 5 R / 4 , 2 R , R ) \varepsilon$ . Hence $\| \theta _ { \mathrm { i n t } } - d \| < 3 R \varepsilon$ and $Q ( \theta _ { \mathrm { i n t } } ) < 9 C \lambda \varepsilon ^ { 2 } / 1 2 8$ . For $\lambda > 0 , Q$ is coercive, so a global minimizer exists and has a value below the cap.

A unit tangent has direction $( 0 , - a y , x )$ . Direct contraction gives

$$
K = { \frac { 2 a ( x ^ { 2 } + a y ^ { 2 } ) } { x ^ { 2 } + a ^ { 2 } y ^ { 2 } } } , \qquad B = { \frac { a ( x ^ { 2 } + a y ^ { 2 } ) } { ( x ^ { 2 } + a ^ { 2 } y ^ { 2 } ) ^ { 3 / 2 } } } .
$$

Because $| y / x | < 1$ on the tube, $K \leq 4 a$ and $B \leq 4 a / \varepsilon$ . For $a > 0$ and $\| b \| \leq \lambda \varepsilon / ( 3 2 a )$ ，

$$
B \left\| \boldsymbol { b } \right\| \leq \lambda / 8 , \qquad B \lambda \left\| \theta - d \right\| \leq 4 a \sqrt { 2 C } \lambda < \lambda / 8 .
$$

Theorem 2 applies with $s = 1 , M = 2 ; C \leq 1 / 1 1 5 2 < C _ { * } ( 1 , 2 ) = 1 / 5 1 2 . \mathrm { ~ I f ~ } a = 0$ , then $K = B = 0$ Also $K \sqrt { 2 C \lambda } \varepsilon \leq \lambda / 4$ whenever $\lambda \geq 5 1 2 C a ^ { 2 } \varepsilon ^ { 2 }$

The v coordinate is a Hessian block $1 + \lambda .$ . The $( x , y )$ block has $J _ { x y } ^ { T } J _ { x y }$ with nonzero eigenvalue between $\varepsilon ^ { 2 }$ and $9 \varepsilon ^ { 2 }$ . Since $\| E \| \le 2 \sqrt { 2 C \lambda } \varepsilon \le \varepsilon ^ { 2 } / 1 2$ and $\lambda \le \varepsilon ^ { 2 }$ , its largest eigenvalue is less than $1 1 \varepsilon ^ { 2 } < 1$ , while its larger eigenvalue is at least $\varepsilon ^ { 2 } - \| E \| > \varepsilon ^ { 2 } / 2$ . For a certified point, Theorem 2 gives the lower bound on its smaller eigenvalue. The tangent Rayleigh quotient gives its upper bound $\lambda + \lambda / 4$ . Thus (35) holds; in particular, the v eigenvalue is the largest. A global minimizer has $b = 0$ and is certified by the same gate.

For the exact-center tuple let $\tau = \lambda / ( a ^ { 2 } \varepsilon ^ { 2 } ) \le C$ and put

$$
\theta _ { \lambda } = ( 0 , \varepsilon \sqrt { 1 - a \tau } , 0 ) .
$$

Then $r = ( 0 , - \lambda / a )$ and

$$
\frac { Q ( \theta _ { \lambda } ) } { \lambda \varepsilon ^ { 2 } } = \frac { \tau } { 2 } + \frac { ( 1 - \sqrt { 1 - a \tau } ) ^ { 2 } } { 2 } < C .
$$

The strict inequality follows from $1 - { \sqrt { 1 - a \tau } } \leq a \tau$ and $a \leq 1 / 4$ . The displacement is less than $a \tau \varepsilon < \varepsilon / 8$ . The gradient and full Hessian are

$$
\| b \| = \lambda ( 2 x / a + \varepsilon - x ) , \qquad H = \mathrm { d i a g } ( 1 + \lambda , 4 \varepsilon ^ { 2 } - 6 \lambda / a + \lambda , - \lambda ) .
$$

Since ${ \sqrt { 3 } } \varepsilon / 2 \leq x \leq \varepsilon$ , the gradient lies in the claimed interval. Also $H _ { x x } \geq ( 4 - 6 a C ) \varepsilon ^ { 2 } \geq ( 5 / 2 ) \varepsilon ^ { 2 }$ This proves indefiniteness for every ridge in the stated interval.

For $F ( v , x , y ) = ( v , x ^ { 2 } )$ and $c = 0$ , retain the first output row. Then $L = 0 , q ( z _ { x } , z _ { y } ) = - z _ { x } ^ { 2 } $ $q ( 1 , 0 ) = \chi ( 0 , - 1 ) = - 1$ , and $D q ( 1 , 0 )$ is onto. The Hessian discrepancy on the kernel is 2a, so the bridge assumptions hold when its chosen tolerance obeys $2 a \le \kappa$

## D.3 Indexed Hessian spectrum

We prove Corollary 8 using the Jacobian bands and the pointwise certificate.

Proof. Min–max on the tangent space gives the upper bound $5 \lambda / 4$ on its $p - n$ indices. The normal compression is at least $s ^ { 2 } \varepsilon ^ { 2 } I / 2$ , so interlacing gives the lower weak bound. Weyl’s inequality gives the upper weak bound $( b _ { w } ^ { 2 } + 1 + M \sqrt { 2 C } ) \varepsilon ^ { 2 }$ , and the strong bounds follow from the fixed strong singular values and $\| E \| = O ( \varepsilon ^ { 2 } )$ . Empty bands are omitted. Lemma 7 provides the hypotheses from cost alone. The gradient implication uses the envelope $B \leq M / ( s \varepsilon )$ . Finally $Q \geq \lambda \left. \theta - d \right. ^ { 2 } / 2$ is coercive, so an exact strict-cap interpolator supplies a strict-cap global minimizer. Its gradient is zero and it passes the adaptive test. □