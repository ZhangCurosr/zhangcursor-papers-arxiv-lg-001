# Exact Fusion of Softmax Attention Heads Can Require Exponentially Many Atoms

HIROYASU USAMI<sup>1</sup>, KEISUKE HARA<sup>1</sup>, AYATO TSUBOI<sup>1</sup>, and NAOHIKO MATSUDA<sup>2</sup>

<sup>1</sup>Department of Computer Science, Graduate School of Engineering, Chubu University, Kasugai, Aichi 487-8501, Japan <sup>2</sup>Heat Transfer Research Department, Research & Innovation Center, Mitsubishi Heavy Industries, Ltd., Takasago, Hyogo 676-8686, Japan

Corresponding author: Hiroyasu Usami (e-mail: usami@fsc.chubu.ac.jp).

ABSTRACT Fusing independently normalized attention heads can increase the number of explicit key–value entries, or atoms, needed to preserve their output. We prove a sharp separation for $\begin{array} { r } { W _ { d } ( q ) = \sum _ { j = 1 } ^ { d } \sigma ( q _ { j } ) \colon } \end{array}$ its d two-atom heads use 2d head–atom terms, whereas its minimum single-head representation has exactly $2 ^ { d }$ atoms, even with arbitrary fixed replacement keys, scalar values, and positive masses. The same lower bound holds for every fixed uniform error below 1/2 on R<sup>d</sup>, and the threshold is sharp. The proof uses exposed-key normal cones; exact equality on any nonempty open query domain inherits the bound. The resulting atom bound for a fixed head count is tight when that count divides d. A normalized family retains its exponential exact minimum while approaching a constant on fixed-radius query balls. For pretrained-model caches, a classical affine-rank bound and a certified instance exclude exact vector preservation at a specified atom count. Learned-cache comparisons on two disjoint banks of 64 contexts show budget-dependent reversals and calibration effects.

INDEX TERMS Attention mechanisms, head fusion, key–value cache, minimum realization, softmax, exponential sums.

## I. INTRODUCTION

Fusing separately normalized attention heads into a single softmax head changes the normalization structure. Within one readout, an atom is one explicit key–value entry with a fixed positive weight. In the constructed family, 12 two-atom heads. use 24 head–atom terms, but any exactly equivalent singlehead readout needs at least 4,096 atoms. Multiplying the head denominators produces a cache indexed by tuples of atoms. The question is whether jointly reoptimizing the replacement keys, values, and masses can reduce this product expansion.

We resolve this minimum-realization question forX $W _ { d } ( q ) =$ $\begin{array} { r l } {  { \sum _ { j = 1 } ^ { d } \sigma ( q _ { j } ) } } \end{array}$ . Its d coordinate heads use two atoms each, while every single-head representation requires at least $2 ^ { d }$ atoms, allowing arbitrary fixed real keys, scalar values, and positive constant masses. The obstruction is geometric: although $W _ { d }$ has only d + 1 distinct scalar limits along generic rays, a convex unique-winner region cannot combine directions across a coordinate jump. This forces $2 ^ { d }$ exposed-key regions.

Keys, values, and masses are query-independent; queries range over $\mathbb { R } ^ { d }$

Contributions.

1) Sharp atom cost. We prove a matching exponential lower bound with a sharp global uniform-error threshold. A derived fixed-head budget is tight whenever the head count divides $d .$

2) Exactness and approximation. We distinguish exact preservation on open query domains from boundedquery approximation, and show that identical key geometry can have different exact atom requirements when values change.

3) Finite-query capacity and learned approximation. We apply a classical affine-rank bound to captured vector outputs and certify one native instance. We give separate and fused caches the same memory budget, fit both, and test how calibration-query placement changes their errors.

## II. RELATED WORK AND POSITIONING

Jordan and Jacobs [1] express a two-way softmax gate as a logistic ridge (their Eq. 2.7). We ask how small a flat, query-independent affine-logit readout can be for a specified function.

Haverbeck et al. [2] analyze joint key–value compression by approximating a context measure under a query distribution, allowing new atoms and positive masses. Streaming memory bounds [5] and head-count or rank separations [6], [7], [8] address other targets and resources. Table 1 contrasts their interfaces with the minimum-realization question studied here.

Multi-query attention (MQA) shares one key–value cache across query heads, while grouped-query attention (GQA) shares a cache within each group [3], [4]. Both preserve headspecific queries and softmax normalizations. The family $W _ { d }$ itself can use two shared scalar KV positions, with keys $( 0 , 1 )$ and values (0, 1), coordinate queries $q _ { j } ,$ and output summation. Thus KV sharing preserves the factored normalizers; fusion here replaces their separate distributions by one affinelogit softmax denominator.

Mixture-of-experts identifiability concerns equality of entire conditional distributions under irreducibility and kernelindependence conditions [9]. It does not directly determine the minimum number of constant scalar experts representing a prescribed mean. In particular, the stated binomial criterion requires $N \geq 2 m - 1$ trials for m experts, so its Bernoulli case $N = 1$ does not cover $m \geq 2 .$ . The cube construction also repeats expert values for $d \geq 2$ . Our lower bound allows arbitrary fixed replacement keys, scalar values, and positive masses without an expert-density identifiability assumption.

## III. READOUT MODEL AND FUSION IDENTITY

Let $\sigma ( t ) = ( 1 + e ^ { - t } ) ^ { - 1 }$ . For a positive integer M, consider

$$
B ( q ) = \frac { \sum _ { i = 1 } ^ { M } m _ { i } e ^ { \langle q , k _ { i } \rangle } \nu _ { i } } { \sum _ { i = 1 } ^ { M } m _ { i } e ^ { \langle q , k _ { i } \rangle } } , \qquad q \in \mathbb { R } ^ { d } ,\tag{1}
$$

where $k _ { i } \in \mathbb { R } ^ { d } , \nu _ { i } \in \mathbb { R }$ , and $0 < m _ { i } < \infty$ are fixed. An atom is one indexed key-value-mass term in this expression. Identical keys can be merged by summing their masses and taking the mass-weighted value average. A constant positive mass is equivalent to a logit bias. Thus a lower bound for (1) also covers unit-mass replacement caches. A fixed scalar linear output projection can be absorbed into the values.

For head h, fix $P _ { h } \in \mathbb { R } ^ { r _ { h } \times d }$ , keys $k _ { h i } \in \mathbb { R } ^ { r _ { h } }$ , scalar values $\nu _ { h i } .$ , and positive masses $m _ { h i }$ . With query $P _ { h } q$ , its output is

$$
A _ { h } ( q ) = \frac { \sum _ { i = 1 } ^ { n _ { h } } m _ { h i } e ^ { \left. P _ { h } q , k _ { h i } \right. } \nu _ { h i } } { \sum _ { i = 1 } ^ { n _ { h } } m _ { h i } e ^ { \left. P _ { h } q , k _ { h i } \right. } } .
$$

Proposition 1 (product expansion; background identity). The sum $\textstyle \sum _ { h = 1 } ^ { H } A _ { h } ( q )$ has a representation ofform (1) with at most $\prod _ { h } n _ { h }$ atoms. A tuple $I = ( i _ { 1 } , \dots , i _ { H } )$ has

$$
\begin{array} { l l } { { \displaystyle k _ { I } = \sum _ { h } { P } _ { h } ^ { \mathsf { T } } k _ { h i _ { h } } , \qquad m _ { I } = \prod _ { h } m _ { h i _ { h } } , } } \\ { { \displaystyle \nu _ { I } = \sum _ { h } \nu _ { h i _ { h } } . } } \end{array}\tag{2}
$$

Proof. Put the head outputs over the common denominator equal to the product of their denominators and expand. Each tuple contributes its product weight to the denominator. Its numerator coefficient is the same product weight multiplied by the sum of its selected values. All masses are positive. Coincident tuple keys may be merged, which only reduces the atom count. □

## IV. A SHARP EXPONENTIAL ATOM BOUND Define

$$
W _ { d } ( q ) = \sum _ { j = 1 } ^ { d } \sigma ( q _ { j } ) , \qquad d \geq 1 .\tag{3}
$$

Each summand is a two-atom, unit-mass head with scalar keys 0, 1, values 0, 1, and coordinate query $q _ { j }$

Theorem 2 (explicit-atom separation). The minimum number of atoms in (1) representing $W _ { d }$ on all of $\mathbb { R } ^ { d } \ i s \ 2 ^ { d }$ . More strongly, ifa representation (1) satisfies

$$
\operatorname* { s u p } _ { q \in \mathbb { R } ^ { d } } | B ( q ) - W _ { d } ( q ) | \leq \varepsilon \qquad 0 \leq \varepsilon < 1 / 2 ,\tag{4}
$$

then $M \geq 2 ^ { d }$ . In contrast, d coordinate heads represent $W _ { d }$ using 2d head-atom terms and a linear sum.

We use the following geometric observation; Figure 1(b) illustrates its convexity obstruction. Panels (a) and (c) show the attaining cube construction and the fixed-head budget of Corollary 6, respectively.

Lemma 3 (an open cone cannot hide a coordinate jump). Let $C ~ \subset ~ \mathbb { R } ^ { d }$ be nonempty, open, and convex. Suppose that a constant a satisfies

$$
| a - r ( u ) | \leq \varepsilon < 1 / 2 , \qquad r ( u ) = \# \{ j : u _ { j } > 0 \} ,
$$

for every $u \in C$ with no zero coordinate. Then C is contained in one open orthant.

Proof. If C crosses a coordinate hyperplane $u _ { j } = 0$ , convexity supplies an intersection and openness supplies a relatively open slice. Choose a slice point whose other coordinates are nonzero, and perturb it a sufficiently small distance in the two coordinate-normal directions. Both perturbed points remain in $C ,$ but their r-values differ by one. The assumed bounds imply $1 \leq 2 \varepsilon < 1$ , a contradiction. If C contains a point on a coordinate hyperplane, openness gives the same contradiction. The argument also covers $d = 1$ □

ProofofTheorem 2. For the upper bound, take all cube keys $a ~ \in ~ \{ 0 , 1 \} ^ { d }$ , masses one, and values $\begin{array} { r } { \nu _ { a } ~ = ~ \sum _ { j } a _ { j } } \end{array}$ . The denominator factors as $\textstyle \prod _ { j } ( 1 + e ^ { q _ { j } } )$ . Summing the marginal expectation of each bit gives exactly $W _ { d }$ . These are also the tuples in Proposition 1.

For the lower bound, merge duplicate keys and let K be the remaining finite key set. Each vertex k of conv(K) has the nonempty, ambient-open, convex strict normal cone

$$
C _ { k } = \bigcap _ { k ^ { \prime } \in K \setminus \{ k \} } \{ u : \langle u , k - k ^ { \prime } \rangle > 0 \} .\tag{5}
$$

This remains true if the key polytope has a lower-dimensional affine span. A singleton key set has $C _ { k } = \mathbb { R } ^ { d }$

For $u \in C _ { k }$ , the unique dominant exponential gives

$$
\operatorname* { l i m } _ { t \to + \infty } B ( t u ) = \nu _ { k } .
$$

For a direction with no zero coordinate,

$$
\operatorname* { l i m } _ { t  + \infty } W _ { d } ( t u ) = r ( u ) .\tag{6}
$$

TABLE 1. Closest results grouped by target, interface, and counted resource.
<table><tr><td>Study</td><td>Target and interface</td><td>Guarantee or counted resource</td></tr><tr><td>Jordan-Jacobs [1]</td><td>Hierarchical conditional expert mixtures</td><td>Logistic gates and hierarchical products; construction back- ground.</td></tr><tr><td>Haverbeck et al. [2]</td><td>Joint KV replacement, including new atoms and positive masses</td><td>Query-distribution approximation risk; broader than support-subset eviction.</td></tr><tr><td>Haris-Onak [5]</td><td>Worst-case autoregressive streaming</td><td>Memory bits with probabilistic guarantees. Head count, dimension, precision, and threshold-degree</td></tr><tr><td>Hsu-Xu [6]</td><td>Multi-task attention computation</td><td>tradeoffs.</td></tr><tr><td>Yu et al. [7]</td><td>attention networks</td><td>Varying sequences, trainable encoders and post- Transformer parameter approximation rates.</td></tr><tr><td>Amsel et al. [8]</td><td>Retrieval with varying contexts and queries</td><td>Rank, head-count, and weight-norm barriers.</td></tr><tr><td>Jiang-Tanner [9]</td><td>generacy</td><td>Full conditional expert densities under nonde- Parameter identifiability; not a scalar-mean minimum.</td></tr><tr><td>This work</td><td>Prescribed fixed scalar readout; arbitrary re- placement keys, values, and positive masses</td><td>Minimum explicit atoms: exact on an open domain, or a sharp global uniform tolerance.</td></tr></table>

![](images/d7ff75697019165a1bda37ea07f1ee671939cf1453280e00edd9f72674c7e50f.jpg)

![](images/507562382eaf81d664a2b1ad3243d5f9fdc37deb195865bdc7e849f8c8887ac9.jpg)  
FIGURE 1. Attainment, convexity obstruction, and fixed-head atom budgets. (a) Eight unit-mass cube atoms realize $W _ { 3 } ;$ three two-atom heads use six head–atom terms. Vertex labels are scalar values, taking four distinct values. (b) The strict-winner cone for $\pmb { k _ { 0 } } = \pmb { 0 }$ against keys $( - { \bf 3 } , - { \bf 2 } ) , ( - { \bf 2 } , - { \bf 3 } )$ contains $\pmb { u } = ( - 1 , 2 )$ and $\pmb { v } = ( \pmb { 2 } , - \pmb { 1 } )$ with r = 1, and their midpoint w = (1/2, 1/2) with r = 2. One winning key gives one constant ray limit throughout its cone, requiring $\varepsilon \geq 1 / 2 .$ Shading shows the clipped closure; dashed boundaries and the origin are excluded. (c) The curve is $H _ { 2 } \mathbf { 1 2 } / H _ { \mathrm { i } }$ filled markers attain it for H | 12. Counts are head–atom terms.

Condition (4) therefore implies $| \nu _ { k } - r ( u ) | \leq \varepsilon \mathrm { o n }$ the generic directions of $C _ { k }$ . Lemma 3 places this cone inside a single orthant.

Every open orthant contains a direction avoiding all pairwise key-tie hyperplanes. In that direction a unique key maximizes the inner product, and it is an exposed vertex. A vertex cone cannot serve two orthants, so at least $2 ^ { d }$ vertices, hence at least $2 ^ { d }$ atoms, are necessary. This counts regions even though the ray limit takes only $d + 1$ distinct values: equal values on disconnected orthants cannot be assigned to one open convex vertex cone. Exact equality is the case $\varepsilon = 0$ □

## V. SHARPNESS OF THE UNIFORM-ERROR THRESHOLD

Proposition 4 (boundary construction). There is a $2 ^ { d - 1 }$ -atom representation with uniform error $1 / 2$ from $W _ { d } . \ F o r \ d = 1$ interpret it as a constant one-atom representation.

Proof. Drop the last coordinate and use

$$
B _ { \mathrm { b d r y } } ( q ) = \sum _ { j = 1 } ^ { d - 1 } \sigma ( q _ { j } ) + \frac { 1 } { 2 } .\tag{7}
$$

The cube construction in $d - 1$ coordinates uses $2 ^ { d - 1 }$ unitmass atoms; its values are the retained bit count plus $1 / 2$ Its error is $| 1 / 2 - \sigma ( q _ { d } ) |$ , whose supremum is $1 / 2 .$ . Thus (4) cannot be extended to $\varepsilon = 1 / 2$

For every finite query the error in (7) is strictly less than $1 / 2 ,$ while its supremum is $1 / 2 .$ . Consequently pointwise strict error below $1 / 2$ does not imply the atom lower bound: a single fixed uniform tolerance strictly below $1 / 2$ is required. □

Corollary 5 (exact preservation on an open domain). Let $U ~ \subset ~ \mathbb { R } ^ { d }$ be nonempty and open. If a readout of form (1) equals $W _ { d }$ throughout U, it has at least $2 ^ { d }$ atoms. In particular, this applies to a bounded open box. It does not assert an approximation bound on a bounded domain.

Proof. For $a \in \{ 0 , 1 \} ^ { d }$ , write $| a | : = \textstyle \sum _ { j } a _ { j }$ . Put $D _ { W } ( q ) =$ $\textstyle \prod _ { j } ( 1 + e ^ { q _ { j } } )$ and $\begin{array} { r c l } { N _ { W } ( q ) } & { = } & { \sum _ { a \in \{ 0 , 1 \} ^ { d } } | a | e ^ { \langle q , a \rangle } } \end{array}$ . Crossmultiplying the positive denominators gives the finite exponential sum

$$
E ( q ) = \sum _ { i , a } m _ { i } ( \nu _ { i } - | a | ) e ^ { \langle q , k _ { i } + a \rangle } .
$$

It vanishes on U. Fix $q _ { 0 } \in \ U$ and any $q _ { * } ~ \in ~ \mathbb { R } ^ { d }$ . On the line $q _ { 0 } + t ( q _ { * } - q _ { 0 } )$ , group equal exponents to write $\begin{array} { r } { E = \sum _ { \ell = 1 } ^ { L } c _ { \ell } e ^ { \lambda _ { \ell } t } } \end{array}$ with distinct real $\lambda _ { \ell } .$ . The line remains in U for sufficiently small $| t | ,$ , so derivatives of orders $0 , \ldots , L - 1$ at zero vanish. The resulting Vandermonde system implies every $c _ { \ell } = 0$ . Hence $E ( q _ { * } ) = 0$ , and positive denominators imply equality at every $q _ { * } . \mathrm { A p p l y }$ Theorem 2. □

Corollary 6 (fixed-head lower bound; sharp when $H \ | \ d )$ Suppose a sum of $H _ { \mathrm { ~ \scriptsize ~ \geq ~ 1 ~ } }$ heads of the form preceding Proposition 1, with $n _ { h } \ge 1$ atoms in head h, equals $W _ { d }$ on a nonempty open subset $o f \mathbb { R } ^ { d }$ , or approximates $W _ { d }$ uniformly on all $o f \mathbb { R } ^ { d }$ withfixed error $\varepsilon < 1 / 2 .$ . Then

$$
\prod _ { h = 1 } ^ { H } n _ { h } \geq 2 ^ { d } , \qquad S : = \sum _ { h = 1 } ^ { H } n _ { h } \geq H 2 ^ { d / H } .
$$

If H divides $d ,$ the bound on S is attained with total query width d.

Proof. Proposition 1 produces a single-head representation with at most $\prod _ { h } n _ { h }$ atoms. Theorem 2 or Corollary 5 requires at least $2 ^ { d }$ . The arithmetic–geometric mean inequality gives the bound on S. For attainment, partition the coordinates into H blocks of size $d / H$ and apply the cube construction independently in each block. Every head has $2 ^ { d / H }$ atoms, and summing the block outputs gives $W _ { d }$ exactly. □

The bound allows arbitrary fixed linear query projections and constant heads, with a normalizer for each head and a linear sum of their outputs. It leaves the optimum over integer head sizes unspecified when H does not divide d.

TABLE 2. Closed-form theoretical comparison for $\begin{array} { r } { F _ { d , 1 / \sqrt { d } } \bullet \bullet \smash { \| \pmb { q } \| _ { 2 } } \leq 1 . } \end{array}$ All cube keys have norm at most one and values lie in $[ \dot { \bf 0 } , 1 ] . \dot { M } _ { \mathrm e x a c t }$ is the minimum single-head atom count; head terms count the coordinate construction. $\mathbf { \bar { \Sigma } } _ { \pmb { { \cal E } } _ { 1 } }$ is the optimal one-atom uniform error. Values follow the closed forms of Proposition 7.
<table><tr><td> $d$ </td><td> $M _ { \mathrm { e x a c t } }$ </td><td>Head terms</td><td> $E _ { 1 }$ </td></tr><tr><td>1</td><td>2</td><td>2</td><td>0.231059</td></tr><tr><td>2</td><td>4</td><td>4</td><td>0.122459</td></tr><tr><td>4</td><td>16</td><td>8</td><td>0.062177</td></tr><tr><td>8</td><td>256</td><td>16</td><td>0.031209</td></tr><tr><td>16</td><td>65536</td><td>32</td><td>0.015620</td></tr></table>

Proposition 7 (sharp one-atom comparison on a query ball). Let $d \geq 1 , \tau > 0 , R > 0 ,$ , and

$$
F _ { d , \tau } ( q ) = \frac { 1 } { d } \sum _ { j = 1 } ^ { d } \sigma ( \tau q _ { j } ) .
$$

On the origin-centered closed ball $\| q \| _ { 2 } \leq R ,$ the best oneatom approximation is the constant $1 / 2$ , with error

$$
\begin{array} { l } { { E _ { 1 } ( d , \tau , R ) : = \underset { c \in \mathbb { R } } { \operatorname* { i n f } } \ \underset { \| q \| _ { 2 } \leq R } { \operatorname* { s u p } } | F _ { d , \tau } ( q ) - c | } } \\ { { \ = \sigma ( \tau R / \sqrt { d } ) - \frac { 1 } { 2 } . } } \end{array}\tag{8}
$$

Exact preservation on this ball still requires $2 ^ { d }$ atoms.

The elementary proof is given in Appendix A.

For $\tau = 1 / \sqrt { d }$ , the cube representation has keys $a / \sqrt { d }$ of norm at most one and values $| a | / d ~ \in ~ [ 0 , 1 ]$ . Its exact minimum on the ball is $2 ^ { d }$ , while

$$
{ \cal E } _ { 1 } ( d , 1 / \sqrt { d } , R ) = \sigma ( R / d ) - { \frac { 1 } { 2 } } \leq { \frac { R } { 4 d } } .\tag{9}
$$

At fixed R, this normalized family becomes nearly constant on the ball while retaining its exponential exact minimum (Table 2).

## VI. RESOURCE ACCOUNTING AND NATIVE-CACHE ANALYSIS

The 2d quantity counts head–atom pairs, not sequence positions: several heads can read the same two positions. Rescaling the target to $W _ { d } / d$ puts cube values in [0, 1] without changing the exact atom bound, but changes the uniform-error threshold to $1 / ( 2 d )$ . Scaling cube keys by $1 / { \sqrt { d } }$ changes the fixed-query target to $W _ { d } ( q / \sqrt { d } )$ ; a compensating query or temperature change must be counted.

Keep the cube key table and unit masses fixed. Values $\nu _ { a } = a _ { 1 } \mathrm { g i v e } \sigma ( q _ { 1 } )$ after nuisance factors cancel, so the exact minimum is two atoms. Values $\nu _ { a } = | a | / d$ give $W _ { d } / d$ , which requires $2 ^ { d }$ atoms by Theorem 2. Both value sets lie in [0, 1], and the keys, rank, and key singular spectrum are identical. Thus key rank or key spectrum alone does not determine the minimum exact atom count.

For each context, normalized root mean squared error (NRMSE) is $\| { \widehat { Y } } - Y \| _ { F } / \| Y \| _ { F }$ with the uncentered target in the denominator; macro results give each context equal weight.

TABLE 3. Controlled FP64 evaluation at $\pmb { d } = 1 2 , \pmb { R } = 1 ,$ with 4,608 queries. KV counts explicit array storage. The first three methods attain the fixed-head atom bound with H = 1, 6, 12, respectively. The fixed mean is a prespecified two-atom approximation.
<table><tr><td>Method</td><td>Terms</td><td>KV (KiB)</td></tr><tr><td>Cube head</td><td>4096</td><td>416.000</td></tr><tr><td>Block size 2</td><td>24</td><td>0.562</td></tr><tr><td>Coordinate</td><td>24</td><td>0.375</td></tr><tr><td>Fixed mean</td><td>2</td><td>0.203</td></tr></table>

## A. CONTROLLED FIXED-CACHE READOUTS

We use $d ~ \in ~ \{ 4 , 8 , 1 2 \}$ and $R ~ \in ~ \{ 0 . 2 5 , 1 , 4 \}$ . For each $d ,$ a paired query bank contains 512 PCG64 uniform points in $[ - 1 , 1 ] ^ { d }$ (seed 20261101) and all $2 ^ { d }$ sign corners, scaled by R. Four methods use the same generic, batched doubleprecision (FP64) NumPy softmax kernel: the cube head, twocoordinate block heads, coordinate heads, and a fixed twoatom approximation $d \sigma ( d ^ { - 1 } \textstyle \sum _ { j } q _ { j } )$ . All K/V arrays are materialized; query reshaping and construction are outside kernel timing. Kernel timing details are retained in the supplement.

Across the 27 exact-method conditions, the maximum absolute error was $7 . 0 3 4 4 \times 1 0 ^ { - 1 3 }$ . Figure 2 shows the finitebank approximation error and explicit array storage. Table 3 reports the $d = 1 2 , R = 1$ condition. $\mathrm { ~ A t ~ } d \stackrel { } { = } 1 2$ , the cube $( H = 1 )$ , two-coordinate blocks $( H ~ = ~ 6 )$ , and coordinate heads $( H = 1 2 )$ attain Corollary 6 with 4,096, 24, and 24 terms, respectively. Conversely, the approximate two-atom form is accurate on the smallest bank radius; the unboundedquery theorem supplies no bounded-query approximation barrier.

## B. PRETRAINED-MODEL TARGETS AND NUMERICAL IDENTITY

We capture a pretrained model’s head outputs and ask how many atoms are needed to preserve them. We then fit compressed caches and measure how approximation changes with the storage budget and calibration queries.

We use the public CODI GPT-2 checkpoint zen-E/COD I-gpt2 [10]. We run ordinary text inference on 64 label-free public BoolQ validation texts [11], truncated to 96 GPT-2 tokens. We capture layer 0 and heads 0–3. Positions 0–15 supply a prefix cache, positions 16–47 calibrate compression, and positions 48 onward supply 2,932 held-out queries. The target renormalizes attention over that prefix. The vector analyses use concatenated head outputs before the model output projection.

Native 64-dimensional queries are divided by 8 before concatenation; the fused dot product applies no further scaling. In the initial two-head scalar-projection diagnostic, full Cartesian fusion agrees with the separate reference to $8 . 8 8 \times 1 0 ^ { - 1 6 }$ in FP64 and $4 . 7 7 \times 1 0 ^ { - 7 }$ in FP32. At fused subset sizes 32 and 64, calibration top-mass pruning has larger macro error than uniform pruning. The supplement contains this diagnostic, its array accounting, and warm-kernel timings. The learned comparison uses trainable keys, values, and log masses.

![](images/93791b9aae993b19dd13453a9cb559092b94e3e8b383e92e0610fa8d5bd844b9.jpg)  
FIGURE 2. Controlled readouts. Top: observed normalized maximum error for the fixed two-atom mean baseline, evaluated on paired finite query banks in boxes. Bottom: FP64 storage used by each method’s K/V arrays. Exact-method errors across all banks are at most $\mathbf { 7 . 0 4 \times 1 0 ^ { - 1 3 } }$

For $H \in \{ 2 , 4 \}$ , the target is the concatenation of the first H head-output vectors before the fixed output projection:

$$
Y _ { H } ( q ) = { \mathrm { c o n c a t } } _ { h = 0 } ^ { H - 1 } \left[ { \mathrm { s o f t m a x } } ( q _ { h } K _ { h } ^ { \top } ) V _ { h } \right] \in \mathbb { R } ^ { 6 4 H } .
$$

## C. NATIVE-CACHE CAPACITY AT FIXED KV PAYLOAD

An n-atom readout produces vectors in the affine hull of its n values. Let $\boldsymbol { Y } ~ \in \bar { \mathbb { R } ^ { N \times D } }$ stack target vectors from any fixed bank of N queries, and let $\boldsymbol { C } \ = \ I - \mathbf { 1 1 } ^ { \top } / N$ . An n-atom readout has prediction matrix $B = P V$ , where $P \mathbf { 1 } _ { n } = \mathbf { 1 } _ { N }$ , so rank $( C B ) \leq n - 1$ . Orthogonal centering and the low-rank approximation theorem [12] give

$$
\frac { \| Y - B \| _ { F } } { \| Y \| _ { F } } \geq L _ { n } ( Y ) : = \frac { \left( \sum _ { j = n } ^ { \operatorname* { m i n } ( N , D ) } \sigma _ { j } ( C Y ) ^ { 2 } \right) ^ { 1 / 2 } } { \| Y \| _ { F } } .\tag{10}
$$

We evaluate this floor numerically in FP64. It applies to the single-head class even with arbitrary replacement keys and values; softmax need not attain it.

A scalar target matrix has centered rank at most one, so the affine-rank floor vanishes for two or more atoms and cannot detect the exponential scalar minimum.

A second bank contains the next 64 texts in the same labelindependent ordering, disjoint from the first bank. The same checkpoint, layer, two heads, and 16-position prefix define its targets.

Each original two-head cache occupies 16 KiB of FP32 key/value arrays. A 16-atom single head with 128- dimensional keys and values has the same KV payload. We additionally grant its 64 bytes of log masses free, enlarging the comparison class. On the 32 early calibration queries of each new context, (10) gives a macro NRMSE floor of 0.02486, ranging from 0.01773 to 0.03270 across all 64 contexts. The original separate heads realize their target exactly. Thus these native targets exhibit a finite-query fusion cost at equal dense KV payload. For the interval certificate, the captured FP32 query, key, and value entries define exact real constants, and the certified output is the corresponding real-exponential attention map. For one fixed native instance, an outward-rounded interval computation certifies a nonzero $1 7 \times 1 7$ minor of the augmented output matrix [1, Y ]. Its rank is therefore at least 17, excluding exact preservation by any 16-atom single head on those 17 queries. Appendix B specifies the instance and verification.

## D. LEARNED APPROXIMATION UNDER AN EQUAL-BYTE BUDGET

On the first bank, we fit vector-valued records of form (1), with $\nu _ { i } \in \mathbb { R } ^ { 6 4 H }$ . The exported keys, values, and masses are fixed at evaluation.

The two-head condition is primary; the four-head condition is a prespecified nested comparison on the same capture. For $n \in \{ 4 , 8 \}$ , the separate heads receive Hn block-local atoms and the single head receives n full-dimensional atoms, at the same allocated memory budget (Table 4).

Each cache uses 32 calibration queries, four fixed kmeans++ initializations, and 512 Adam updates of keys and log masses, at learning rate 0.03. Values are solved conditionally by ridge regression at each update and once after the final update. We scale the target using its calibration RMS and denote the result by ${ \widetilde { Y } } .$ For $D = 6 4 H$ and $N _ { c } = 3 2$ , the objective shared by both architectures is

$$
J = \frac { \| A V - \widetilde { Y } \| _ { F } ^ { 2 } } { N _ { c } D } + \frac { 1 0 ^ { - 6 } \| V \| _ { F } ^ { 2 } } { D } .
$$

Separate-head residuals and penalties are summed with the same denominator D. FP64 Cholesky solves give $( A ^ { \top } A +$ $N _ { c } 1 0 ^ { - 6 } I ) V = A ^ { \top } \widetilde { Y }$ ; envelope gradients update the keys and log masses. Initial keys come from native-key clusters for separate heads and Cartesian-key clusters for the single head. The four-head initializer samples 512 fixed tuples; subsequent keys are unrestricted. The query scaling obtained only from calibration data is folded into the exported keys. We export the final caches in FP32. For each cache, the start with lowest exported calibration MSE is fixed before opening held-out assets.

## a: Results.

The primary two-head comparison does not meet the prespecified two-budget superiority criterion: at $H = 2 , n = 4$ the single-head macro error is lower (0.4467 versus 0.4948); at $n = 8$ , separate heads are lower (0.3363 versus 0.5269). Separate heads have lower errors on 32/64 and 62/64 contexts, respectively. The median single-minus-separate differences are −0.0011 and 0.1179. At $H = 4$ , separate heads are lower at both budgets, on 42/64 and 60/64 contexts.

Calibration error decreases with capacity in both classes, whereas the fitted single head’s held-out error increases in the primary condition (Table 5). Across starts, the largest NRMSE range within a context is 1.8158 for separate heads at $H = 2 , n = 4$ , and 1.1793 for the single head at $H = 2 , n =$ 8. Macro rank floors are 0.2539 and 0.1265 for two heads, and 0.2559 and 0.1495 for four heads. Every selected separate fit lies above its corresponding single-head floor, so this native bank supplies no separation certified by that bound.

## E. CALIBRATION-QUERY COVERAGE

We compare calibration-query placement on the second bank. Both policies retain 32 calibration queries and share the final 16 queries as their held-out set. The early policy uses positions 16–47. For a text of T tokens, the distributed policy uses positions $1 6 + | j ( T - 3 3 ) / 3 1 | , j = 0 , \dots , 3 1$ , before the held-out interval $[ T - 1 6 , T )$ . Here $6 7 \leq T \leq 9 6$ . Keys, values, fitting steps, starts, and retention budgets follow the same procedure; all eight selectors are fixed before evaluating either policy. Changing the calibration queries also changes the scaling and initialization used by the fitting procedure.

Distributed calibration lowers macro NRMSE in all four planned contrasts: by 0.0250 and 0.0098 for separate and single heads at 4.031 KiB, and by 0.0129 and 0.0918 at 8.062 KiB. At the smaller budget, the two errors are close (0.4717 and 0.4711); at the larger budget, separate heads are lower (0.3515 versus 0.4817). The worst per-context separate-head error at the smaller budget decreases from 1.5183 to 0.8585. The mean nearest-calibration cosine distance of held-out queries decreases from 0.4916 to 0.4240 in the fixed native query coordinates.

## VII. IMPLICATIONS FOR KV COMPRESSION

A factored evaluator computes $W _ { d }$ with $O ( d )$ sigmoid evaluations; a single normalized exponential sum of form (1) needs $2 ^ { d }$ atoms.

The scalar obstruction also transfers to vector targets through a fixed linear functional. If $F : \mathbb { R } ^ { d }  \mathbb { R } ^ { p }$ satisfies $\ell ( F ( q ) ) = W _ { d } ( q )$ for a fixed nonzero $\ell ,$ applying ℓ to vector values preserves the keys, positive masses, and atom count. Theorem 2 and Corollary 5 therefore give the same exact lower bound. A global uniform vector error $\varepsilon$ also gives $M \geq 2 ^ { d }$ whenever $\| \ell \| _ { * } \varepsilon < 1 / 2$

(a) Native-cache capacity  
TABLE 4. Resource allocation in the equal-byte learned comparison. Both architectures retain the same number of key/value coefficients and receive the same FP32 storage budget. Normalizer count, atom count, coordinate access, and weight sharing vary together, allowing different training arithmetic at the same memory budget.
<table><tr><td>Resource</td><td>Separate heads</td><td>Single head</td></tr><tr><td>Softmax normalizers</td><td>H</td><td>1</td></tr><tr><td>Atoms; log masses</td><td> $H n ; H n$ </td><td>n; n</td></tr><tr><td>Key and value width per atom</td><td>64 each</td><td>64H each</td></tr><tr><td>Query and output coordinates</td><td>Fixed 64D block per head</td><td>All 64H coordinates per atom</td></tr><tr><td>Mixture-weight sharing</td><td>Within each 64D output block</td><td>Across all 64H output coordinates</td></tr><tr><td>Key coefficients; value coefficients</td><td>64Hn; 64Hn</td><td>64Hn; 64Hn</td></tr><tr><td>FP32 array payload</td><td>4(129Hn) bytes</td><td>4(128Hn + n) bytes</td></tr><tr><td>Unused allocated padding</td><td>0</td><td>4n(H − 1) bytes</td></tr><tr><td>Total allocated retention</td><td>4(129Hn) bytes</td><td>4(129Hn) bytes</td></tr></table>

TABLE 5. Learned compression with equal FP32 storage budgets. Each result uses all 64 prompts. Cal.: macro calibration NRMSE from FP32 exports; held: macro held-out NRMSE from FP32 records selected by calibration error. Lower is better. S/F denote separate/single-head fits. Separate/single allocations retain Hn/n total atoms. Fits use 512 updates and four starts; Table 4 details the resource allocation.
<table><tr><td>H</td><td>n per head</td><td>KiB each</td><td>Cal.S</td><td>Cal. F</td><td>Held S</td><td>Held F</td><td>S lower</td></tr><tr><td>2</td><td>4</td><td>4.031</td><td>0.1901</td><td>0.2355</td><td>0.4948</td><td>0.4467</td><td>32/64</td></tr><tr><td>2</td><td>8</td><td>8.062</td><td>0.0573</td><td>0.1179</td><td>0.3363</td><td>0.5269</td><td>62/64</td></tr><tr><td>4</td><td>4</td><td>8.062</td><td>0.1550</td><td>0.2845</td><td>0.4855</td><td>0.5135</td><td>42/64</td></tr><tr><td>4</td><td>8</td><td>16.125</td><td>0.0463</td><td>0.2071</td><td>0.3657</td><td>0.4896</td><td>60/64</td></tr></table>

![](images/a9f6cebd07bddc179531076680d888913adca2a013759a8b0814078e6c9ad773.jpg)

(b) 4.031 KiB fitted cache  
![](images/dcc5861a2dad6422152bff737a09d3ef8ef59f442aa5bf82e7e19b19c489392a.jpg)

(c) 8.062 KiB fitted cache  
![](images/c7ab225f589a4be827877bb1b88672195ba1f9d7f65e150c0ef351e244a641ff.jpg)  
FIGURE 3. Native capacity and calibration on the second, disjoint 64-context bank. (a) The original two-head cache realizes its target with a 16 KiB KV payload. Gray points are classical affine-rank error floors for arbitrary 16-atom single heads with the same KV payload and free masses, evaluated on 32 early queries per context; the red bar is their mean. (b,c) Macro error on the common final-16-query bank when the 32 calibration queries are early or distributed. Both architectures improve at both fitted-cache budgets.

TABLE 6. Calibration-policy follow-up on 64 unused contexts. Held-out NRMSE is averaged equally across prompts on the common bank of final-16 queries. S/F: separate/single-head fits.
<table><tr><td></td><td colspan="4">Early</td></tr><tr><td>n</td><td>KiB</td><td>S</td><td>F</td><td>Distributed S</td><td>F</td></tr><tr><td>4</td><td>4.031</td><td>0.4967</td><td>0.4809</td><td>0.4717</td><td>0.4711</td></tr><tr><td>8</td><td>8.062</td><td>0.3644</td><td>0.5734</td><td>0.3515</td><td>0.4817</td></tr></table>

Key/value dimensions, precision, normalizers, and evaluation kernels determine how atoms translate into storage and time. Table 4 records the matched storage resources and the architectural differences in the learned comparison.

## VIII. CONCLUSION

Fusing softmax heads can require exponentially many atoms even after all replacement keys, scalar values, and masses are reoptimized. The minimum single-head representation of $W _ { d }$ has $2 ^ { d }$ atoms, with a sharp all-query uniform-error threshold of $1 / 2$ . For H separately normalized heads, the total atom count is at least ${ \bf \dot { \boldsymbol { H } } } 2 ^ { d / H }$ , attained when H divides d. The normalized family becomes nearly constant on a bounded query ball while retaining its exact minimum.

For one native vector target, the certificate rules out exact single-head preservation at the original dense KV payload. Learned-cache rankings vary with storage budget, while distributed calibration reduces macro error in both allocations.

## CODE AND DATA AVAILABILITY

The reproducibility materials are maintained at https://github.com/usamihiroyasu/attention-head-fusion. The same files are supplied as supplementary material. They include diagnostic and learned-cache implementations, frozen protocols, both query banks with upstream attribution, complete results and all-start diagnostics, and environment and asset identifiers. Model weights are retrieved from the cited pinned repository rather than redistributed.

## The checkpoint revision is

fd641b3d3edc59e4f534b55588e906588c9e36bb. All 251 checkpoint entries have the expected keys and shapes. Every loaded tensor equals its checkpoint value after conversion to FP32. The dataset Parquet file has SHA-256 prefix 52355d11524b4b87; the supplement records the full digest and all input hashes. All 36 method–bank conditions completed. All eight conditions completed for all 64 contexts.

## APPENDIX A PROOF OF PROPOSITION 7

Proof. Every one-atom readout is constant. Put $\begin{array} { r l } { f ( x ) } & { { } = } \end{array}$ $\sigma ( \tau x ) \mathrm { ~ - ~ } 1 / 2$ for $x ~ \geq ~ 0$ . This function is increasing and concave. Odd symmetry of $\sigma ( t ) - 1 / 2$ , Jensen’s inequality, and Cauchy–Schwarz give

$$
\begin{array} { l } { \displaystyle | F _ { d , \tau } ( q ) - 1 / 2 | \le \frac 1 d \sum _ { j } f ( | q _ { j } | ) } \\ { \le f \left( \frac { \| q \| _ { 1 } } { d } \right) \le f \big ( R / \sqrt { d } \big ) . } \end{array}
$$

The opposite points $q = \pm ( R / \sqrt { d } )$ 1 attain the two symmetric extrema. Their output separation is twice the displayed error, so no constant does better. Exact preservation follows from Corollary 5 after the invertible query scaling by τ and output scaling by d; the ball has nonempty interior. □

## APPENDIX B A CERTIFIED NATIVE-CACHE INSTANCE

Take context 0 of the second bank (BoolQ validation row 842), its first 17 early queries (positions 16–32), and the fixed prefix from positions 0–15. Treat the captured FP32 query, key, and value entries as exact real constants. Form a $1 7 \times 1 7$ matrix A with a constant-one column, output coordinates 0– 14 of head 0, and coordinate 0 of head 1.

The certificate encloses every dot product, exponential, positive normalization, and output using 80-digit decimal arithmetic. Correctly rounded endpoint exponentials [13] are expanded outwards using next\_minus and next\_plus in the same 80-digit context; the remaining interval operations use directed rounding. Interval elimination gives

$$
\begin{array} { r } { 3 . 3 1 1 0 3 3 7 4 6 8 1 5 8 2 0 2 \times 1 0 ^ { - 3 0 } < \mathrm { d e t } A } \\ { < 3 . 3 1 1 0 3 3 7 4 6 8 1 5 8 2 0 3 \times 1 0 ^ { - 3 0 } . } \end{array}
$$

An independent check uses a proposed numerical inverse R only as an exact rational preconditioner. Exact fraction arithmetic shows that every matrix in the stored entry enclosure has RA strictly row diagonally dominant, with all dominance margins exceeding 0.9999999993. Hence it is nonsingular. The input arrays, entry enclosures, preconditioner, and two verification programs accompany the supplement.

For an M-atom flat readout, $B \ : = \ : P V$ and $P \mathbf { 1 } _ { M } = \mathbf { 1 }$ , so $[ { \bf 1 } , B ] = P [ { \bf 1 } _ { M } , V ]$ has rank at most M. The certified rank 17 excludes $M \leq 1 6$ for this instance.

## REFERENCES

[1] M. I. Jordan and R. A. Jacobs. Hierarchical Mixtures of Experts and the EM Algorithm. Neural Computation, 6(2):181–214, 1994. https://doi.org/10.1162/neco.1994.6.2.181

[2] L. Haverbeck, C. Amo Alonso, A. F. Posada-Moreno, S. Trimpe, and M. Pavone. The risk of KV cache compression. arXiv:2607.01520v1, 2026. https://arxiv.org/abs/2607.01520v1

[3] N. Shazeer, ‘‘Fast Transformer Decoding: One Write-Head is All You Need,’’ arXiv:1911.02150v1, 2019. https://arxiv.org/abs/1911.02150v1

[4] J. Ainslie, J. Lee-Thorp, M. de Jong, Y. Zemlyanskiy, F. Lebrón, and S. Sanghai, ‘‘GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints,’’ in Proceedings ofEMNLP, pp. 4895–4901, 2023, doi: 10.18653/v1/2023.emnlp-main.298. https://aclanthology.org/2023.emnlp-main.298/

[5] T. Haris and K. Onak. Compression Barriers in Autoregressive Transformers. Proceedings of the 38th Conference on Learning Theory, PMLR 291, pp. 2757-2785, 2025. https://proceedings.mlr.press/v291/haris25a.html

[6] D. Hsu and M. Xu. Attention-based representations for multi-task computation. arXiv:2608.04243v1, 2026. https://arxiv.org/abs/2608.04243v1

[7] P. Yu, H. Jiang, Z. Bao, R. Yu, and Q. Li. The Effect of Attention Head Count on Transformer Approximation. arXiv:2510.06662v2, 2026. https://arxiv.org/abs/2510.06662v2

[8] N. Amsel, G. Yehudai, and J. Bruna. Quality over Quantity in Attention Layers: When Adding More Heads Hurts. International Conference on Learning Representations, 2025. ICLR 2025 proceedings

[9] W. Jiang and M. A. Tanner. On the identifiability of mixtures-of-experts. Neural Networks, 12(9):1253–1258, 1999. https://doi.org/10.1016/S0893-6080(99)00066-0

[10] Z. Shen, H. Yan, L. Zhang, Z. Hu, Y. Du, and Y. He. CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation. Proceedings of EMNLP, pp. 677–693, 2025. https://aclanthology.org/2025.emnlp-main.36/

[11] Google. BoolQ validation dataset, revision 35b264d03638db9f4ce671b711558bf7ff0f80d5. CC BY-SA 3.0. Pinned public dataset. Only public input texts are used in this diagnostic.

[12] C. Eckart and G. Young, ‘‘The approximation of one matrix by another of lower rank,’’ Psychometrika, vol. 1, no. 3, pp. 211–218, 1936, doi: 10.1007/BF02288367.

[13] Python Software Foundation, ‘‘decimal—Decimal fixed-point and floating-point arithmetic,’’ Python 3.12 Documentation, accessed Nov. 1, 2026. https://docs.python.org/3.12/library/decimal.html