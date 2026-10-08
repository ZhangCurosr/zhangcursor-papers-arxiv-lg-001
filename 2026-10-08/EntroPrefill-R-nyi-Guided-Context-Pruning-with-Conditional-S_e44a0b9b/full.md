# EntroPrefill: Rényi-Guided Context Pruning with Conditional Stability Guarantees for Retrieval-Augmented Generation

Inbasekaran S.<sup>1</sup>

<sup>1</sup>SRM Institute of Science and Technology, India

## Abstract

Mid-prefill pruning can reduce the sequence processed by deeper transformer layers, but attention concentration alone does not certify that discarded context is dispensable. We formulate EntroPrefill as a Rényi-guided proposal mechanism coupled to explicit constraints on discarded attention mass. Sink-isolated, regularized head pooling respects grouped-query attention while exposing a quantitative trade-of between specialization and worst-head coverage. We derive a mixture-to-head deletion envelope, a computable upper bound on feasible token removal, and a finite-sample observer guarantee that remains valid when the pruning layer is selected adaptively. We then establish a conditional transformer perturbation bound with explicit suficient Lipschitz constants and a first-token decision-margin corollary. A counterexample shows why shallow observations alone cannot imply an unconditional future-output guarantee. The systems analysis distinguishes query-head unions, physical page allocation, and KV-transfer payload, and gives an arithmetic break-even condition for pruning. These results define a theoretical design and its limits; they do not assert measured acceleration or preserved task accuracy. Experiments are reserved for subsequent validation of the assumptions, approximation tightness, and end-to-end resource trade-ofs.

## 1 Introduction & Related Work

Retrieval-augmented generation augments a query with document passages that difer in their relevance and computational value. A passage can contain a directly useful fact, an intermediate reasoning dependency, or a distractor. Processing every passage through every transformer layer incurs work even when some content is unnecessary for the response. Removing passages during prefill therefore ofers a natural optimization opportunity, provided the selection rule distinguishes a computational saving from an unsupported claim of semantic preservation.

Attention-based selection has substantial prior art. LLMLingua compresses prompts before target-model inference, while AttentionRAG uses query-focused attention for context pruning [1, 2]. LazyLLM dynamically selects prompt tokens and supports their reconsideration during generation; SlimInfer prunes intermediate hidden states; ASL adaptively selects a pruning layer from attention-rank behavior [3–5]. These works preclude claiming novelty for attention-guided compression, intermediate-layer pruning, or adaptive layer selection in isolation. EntroPrefill instead studies how an entropy-guided proposal can be coupled to explicit deletion constraints, adaptive statistical auditing, and conditional output bounds.

Decode-cache policies address a related resource dimension. H2O balances recent and highimportance tokens; SnapKV derives important positions from a prompt observation window;

Ada-KV studies adaptive cache-budget allocation and bounds attention-output loss [6–8]. The local relationship between discarded attention mass and output perturbation is thus not introduced here as an unprecedented principle. Our analysis develops its consequences for a global chunk-removal decision that changes the support processed by subsequent layers, where unobserved attention and representation drift become essential assumptions.

Grouped-query attention (GQA) shares keys and values among query heads [9]. Page-EntroKV proposes sink-isolated entropy weighting and page-level selection within these groups [10]. Attention sinks motivate conditioning entropy away from designated initial positions, but concentration after conditioning still does not identify factual correctness [11, 12]. At the execution level, FlashAttention-2, PagedAttention, DistServe, and Sarathi-Serve optimize diferent parts of the computation and resource schedule [13–16]. Their benefits do not transfer automatically to a newly combined pruning system.

The mathematical contribution is a unified conditional analysis. First, a mixture envelope converts pooled discarded mass into a bound for an individual observed attention row and quantifies when pooling loses useful coverage. Second, a constrained removal formulation admits a dual upper bound, separating feasible pruning from unattainable budget targets. Third, an independent observer audit controls selected-layer error without treating adaptive stopping as a fixed test. Fourth, a blockwise stability theorem states the additional assumptions needed to propagate deletion through a pre-normalized transformer. Elementary probability, convex-mixture identities, and norm inequalities are identified as established ingredients; the proposed contribution is their coupling to one fully specified pruning procedure.

## 2 Problem Formulation

## 2.1 Prompt support and GQA attention

Let the prompt contain protected tokens P and disjoint candidate document chunks $c _ { 1 } , \ldots , c _ { k }$ Instructions, the complete user query, required delimiters, and configured initial sink tokens are protected. If protection intersects a chunk, the entire chunk is protected before indexing the remaining candidates. Write $\begin{array} { r } { n _ { i } = | c _ { i } | , \mathcal { D } = \bigcup _ { i } c _ { i } , N = \sum _ { i } n _ { i } } \end{array}$ , and $T = | \mathcal { P } | + N$ . All candidate documents precede the user-query observer positions. If $k = 0$ , the method performs ordinary inference.

Consider a decoder-only transformer with L blocks, model width $d _ { m } , H _ { Q }$ query heads, G KV groups, and head dimension d. Let $r = H _ { Q } / G$ be an integer, and let ${ \mathcal { H } } _ { g }$ contain the query heads sharing group g. At layer l, query position u, and head $h ,$ , attention is

$$
A _ { u , j } ^ { l , h } = \frac { \exp { z _ { u , j } ^ { l , h } } } { \sum _ { v \le u } \exp { z _ { u , v } ^ { l , h } } } , \qquad z _ { u , j } ^ { l , h } = \frac { q _ { u } ^ { l , h } \cdot k _ { j } ^ { l , g ( h ) } } { \sqrt { d } } , \quad j \le u ,\tag{1}
$$

and is zero on masked positions. Original position IDs are used throughout. Every document token is visible to every user-query observer; no causal-visibility correction favoring later documents follows from this layout.

Choose a fixed collection $\mathcal { Q } _ { p }$ of $W _ { p } \geq 1$ proposal observers from the user query. They may be a deterministic sufix or a random sample drawn independently of the audit described in Section 5. Define

$$
a _ { j } ^ { l , h } = \frac { 1 } { W _ { p } } \sum _ { u \in \mathcal { Q } _ { p } } A _ { u , j } ^ { l , h } .\tag{2}
$$

The averaging is over query rows, not over a truncated key window. All causally visible key positions remain in the distribution.

## 2.2 Decisions, guarantees, and norms

A removal set $E \subseteq { \mathcal { D } }$ is a union of complete candidate chunks. The retained support is $R = [ T ] \setminus E$ , containing ${ \mathcal { P } } _ { : }$ , with length $T ^ { \prime } = T - | E |$ . At the selected layer $l ^ { * }$ , the method keeps $X ^ { l ^ { * } } | _ { R }$ and the shallow-layer KV entries indexed by R, then processes only R through the remaining layers. It does not restart the model on a shortened text prompt.

We distinguish an exact guarantee on specified proposal rows, a probabilistic guarantee for a specified observer distribution, and a conditional guarantee for the full-model output. These statements concern diferent objects. For a retained-state matrix $X$ , use the row-maximum norm $\left\| X \right\| _ { \operatorname* { m a x } , 2 } = \operatorname* { m a x } _ { u } \left\| X _ { u } \right\| _ { 2 }$ . Vector and matrix norms without another subscript are Euclidean and spectral, respectively. Attention measures use natural logarithms. The group-floor parameter is $\zeta \in ( 0 , 1 ]$ , the entropy temperature is $\tau > 0$ , and numerical residual thresholds are positive.

## 3 Sink-Isolated Rényi Pooling

## 3.1 Regularized head weighting

Let $\mathcal { S } \subseteq \mathcal { P }$ be the sink set. For each head, set $\begin{array} { r } { z _ { h } = \sum _ { j \notin S } a _ { j } ^ { l , h } } \end{array}$ . A head is entropy-eligible when $z _ { h } \ge z _ { \operatorname* { m i n } } > 0$ . For eligible heads, define the conditional distribution on $j \not \in { \mathcal { S } }$ by

$$
\widetilde { a } _ { j } ^ { h } = \frac { a _ { j } ^ { l , h } } { z _ { h } } , \qquad C _ { h } = \sum _ { j \notin { \cal S } } ( \widetilde { a } _ { j } ^ { h } ) ^ { 2 } , \qquad H _ { 2 , h } ^ { \mathrm { n s } } = - \ln C _ { h } .\tag{3}
$$

For a group with at least one eligible head, assign eligible heads

$$
v _ { h } = \frac { C _ { h } ^ { 1 / \tau } } { \sum _ { a \in \mathcal { H } _ { g } : z _ { a } \geq z _ { \operatorname* { m i n } } } C _ { a } ^ { 1 / \tau } } , \qquad w _ { h } = ( 1 - \zeta ) v _ { h } + \frac { \zeta } { r } ,\tag{4}
$$

with $v _ { h } = 0$ for ineligible heads. If all heads in a group are ineligible, set $v _ { h } = 1 / r$ . This fallback preserves normalization; if every group falls back, the algorithm bypasses pruning. Collision sums and normalizations are accumulated in FP32 or greater precision.

The weights are exponential in negative entropy, since $C _ { h } ^ { 1 / \tau } = \exp ( - H _ { 2 , h } ^ { \mathrm { n s } } / \tau )$ . They are not reciprocals of entropy. The positive floor prevents a numerically or statistically suppressed head from receiving zero mixture coverage. Pool the original attention distributions:

$$
\mu _ { j } ^ { l } = \frac { 1 } { G } \sum _ { g = 1 } ^ { G } \sum _ { h \in \mathcal { H } _ { g } } w _ { h } a _ { j } ^ { l , h } = \sum _ { h = 1 } ^ { H _ { Q } } \sum _ { u \in \mathcal { Q } _ { p } } \omega _ { h , u } A _ { u , j } ^ { l , h } , \qquad \omega _ { h , u } = \frac { w _ { h } } { G W _ { p } } .\tag{5}
$$

Thus $\mu ^ { l }$ is a probability distribution and $\omega _ { h , u } \geq \zeta / ( H _ { Q } W _ { p } )$ . Entropy controls the proposal;   
the floor explicitly limits how strongly it may suppress a head.

Proposition 3.1 (Entropy separation and its floor). Suppose a group contains an eligible head $h _ { * }$ with entropy H<sub>∗</sub> and $s - 1$ other eligible heads with entropy at least $H _ { * } + \Delta$ , where $s < r$ and $\Delta \geq 0$ . Then

$$
w _ { h _ { * } } \geq \frac { \zeta } { r } + \frac { 1 - \zeta } { 1 + ( s - 1 ) e ^ { - \Delta / \tau } } .\tag{6}
$$

If an eligible sink-dominated head has a uniform residual on n non-sink positions and an eligible sibling has collision at least $c _ { 0 } > 0 \AA$ , then

$$
w _ { \mathrm { s i n k } } \leq \frac { \zeta } { r } + ( 1 - \zeta ) \frac { n ^ { - 1 / \tau } } { n ^ { - 1 / \tau } + c _ { 0 } ^ { 1 / \tau } } .\tag{7}
$$

Proof. Divide the denominator of $v _ { h _ { * } }$ by $\exp ( - H _ { * } / \tau )$ . Each competing term is at most $\exp ( - \Delta / \tau )$ , yielding (6). A uniform residual has collision $1 / n$ , whereas the specified sibling contributes at least $c _ { 0 } ^ { 1 / \tau }$ to the denominator. Substitution into (4) gives (7). □

For fixed positive ζ, the sink weight approaches the floor $\zeta / r ,$ not zero. Without the floor, its unnormalized factor decays polynomially in n. If all siblings have identical difuse residuals, normalized weights remain uniform. If a sink head’s residual is itself sharply concentrated, the uniform-residual premise fails. These qualifications prevent the entropy statistic from being interpreted as a universal retrieval-head detector. The standard Rényi family and its monotonicity are discussed by van Erven and Harremoës [17]; order two is chosen here for its explicit second-moment representation, not for universal statistical optimality.

## 3.2 Null-calibrated chunk priorities

Let $Z _ { l } = \mu ^ { l } ( \mathcal { D } )$ . When $Z _ { l } < Z _ { \mathrm { m i n } }$ , bypass pruning. Otherwise, define document-conditioned probabilities $p _ { j } = \mu _ { j } ^ { l } / Z _ { l }$ , chunk mass $\begin{array} { r } { M _ { i } \ = \ \sum _ { j \in c _ { i } } p _ { j } } \end{array}$ , and null mass $\pi _ { i } = n _ { i } / N$ . For smoothing parameter $\varepsilon _ { s } > 0$ , set

$$
\widehat { p } _ { i , j } = \frac { p _ { j } + \varepsilon _ { s } \pi _ { i } / n _ { i } } { M _ { i } + \varepsilon _ { s } \pi _ { i } } , \qquad \ell _ { i } = \frac { M _ { i } + \varepsilon _ { s } \pi _ { i } } { ( 1 + \varepsilon _ { s } ) \pi _ { i } } .\tag{8}
$$

For $n _ { i } > 1$ , define

$$
\kappa _ { i } = \frac { n _ { i } \sum _ { j \in c _ { i } } \widehat { p } _ { i , j } ^ { 2 } - 1 } { n _ { i } - 1 } , \qquad S _ { i } = \ln \ell _ { i } + \lambda \kappa _ { i } , \quad \lambda \geq 0 ,\tag{9}
$$

and set $\kappa _ { i } = 0$ for singleton chunks. The score determines a proposal ordering; it does not replace discarded-mass constraints.

Proposition 3.2 (Calibration and bounded concentration preference). For every chunk, $0 \leq \kappa _ { i } \leq 1$ . Uniform attention on D gives $S _ { i } = 0$ for every chunk. For any two chunks,

$$
S _ { i } - S _ { j } \geq \ln ( \ell _ { i } / \ell _ { j } ) - \lambda .\tag{10}
$$

In particular, $\ell _ { i } / \ell _ { j } > e ^ { \lambda }$ implies $S _ { i } > S _ { j }$ , independently of their within-chunk concentration.

Proof. For a probability vector on n positions, Cauchy–Schwarz gives $\textstyle \sum p _ { j } ^ { 2 } \geq 1 / n$ , while $p _ { j } ^ { 2 } \leq p _ { j }$ gives $\textstyle \sum p _ { j } ^ { 2 } \leq 1$ . Equation (9) yields the range. Under the uniform null, $M _ { i } = \pi _ { i } ,$ $\widehat { p } _ { i , j } = 1 / n _ { i }$ , and $\dot { \ell _ { i } } = 1$ . Finally, $\kappa _ { i } - \kappa _ { j } \geq - 1$ proves (10). □

The last inequality is a precise, limited safeguard for difuse evidence: a suficiently large relative-mass advantage cannot be overturned by concentration alone. It does not assert that all difuse relevant chunks possess that advantage. Lowering the Rényi order would not supply this property, because $H _ { 1 } ( P ) \ge H _ { \alpha } ( P ) \ge H _ { 2 } ( P )$ for $1 \leq \alpha \leq 2$ implies the reverse ordering for negative entropies. No adaptive-order rescue claim is required by the present formulation.

## 4 Deletion Envelopes and Constrained Selection

## 4.1 From a pooled measure to individual attention rows

Write a for a proposal attention row and ω for its positive mixture coeficient in (5). For a removal set E, let $q = \mu ^ { l } ( E )$ and $\eta = a ( E )$ . The relevant divergence is

$$
D _ { \chi ^ { 2 } } ( a \| \mu ^ { l } ) = \sum _ { j : \mu _ { j } ^ { l } > 0 } \frac { ( a _ { j } - \mu _ { j } ^ { l } ) ^ { 2 } } { \mu _ { j } ^ { l } } .\tag{11}
$$

Mixture domination implies $a _ { j } = 0$ whenever $\mu _ { j } ^ { l } = 0$ , so the expression is well-defined.

Theorem 4.1 (Mixture-to-row deletion envelope). For every E and every constituent proposal row,

$$
\eta \leq \operatorname* { m i n } \left\{ 1 , \frac { q } { \omega } , q + \sqrt { D _ { \chi ^ { 2 } } ( a \| \mu ^ { l } ) q ( 1 - q ) } \right\} , \qquad D _ { \chi ^ { 2 } } ( a \| \mu ^ { l } ) \leq \omega ^ { - 1 } - 1 .\tag{12}
$$

Consequently, max $\begin{array} { r } { \ L _ { 1 \leq h \leq H _ { Q } } \operatorname* { m a x } _ { u \in \mathcal { Q } _ { p } } A _ { u } ^ { l , h } ( E ) \leq \operatorname* { m i n } \{ 1 , H _ { Q } W _ { p } q / \zeta \} } \end{array}$ . For head-averaged proposal rows $a ^ { l , h }$ , the corresponding bound is min $\{ 1 , H _ { Q } q / \zeta \}$

Proof. The pointwise inequality $\mu _ { j } ^ { l } ~ \ge ~ \omega a _ { j }$ gives $\eta \leq q / \omega$ . On the support of $\mu ^ { l } .$ , put $f _ { j } = a _ { j } / \mu _ { j } ^ { l } - 1$ . Then $\mathbb { E } _ { \mu ^ { l } } f = 0$ and

$$
\eta - q = \mathbb { E } _ { \mu ^ { l } } [ f ( \mathbf { 1 } _ { E } - q ) ] .
$$

Cauchy–Schwarz bounds its absolute value by $\sqrt { \mathbb { E } _ { \mu ^ { l } } f ^ { 2 } q ( 1 - q ) }$ , yielding the third term. Also,

$$
D _ { \chi ^ { 2 } } ( a \| \mu ^ { l } ) = \sum _ { j } \frac { a _ { j } ^ { 2 } } { \mu _ { j } ^ { l } } - 1 \le \omega ^ { - 1 } \sum _ { j } a _ { j } - 1 = \omega ^ { - 1 } - 1 .
$$

Substitute the row coeficient lower bound. Averaging observers first gives constituent coeficient $w _ { h } / G \geq \zeta / H _ { Q }$ , proving the last statement. □

The envelope can be sharp. Let a put all its mass on one position, let another distribution be supported of that position, and let $\mu = \omega a + ( 1 - \omega ) b$ . For $E = \operatorname { s u p p } ( a ) , q = \omega , \eta = 1$ and $D _ { \chi ^ { 2 } } = \omega ^ { - 1 } - 1 \mathrm { : }$ ; both nontrivial envelope terms equal one. Thus a small pooled mass does not uniformly imply a small head-specific loss when the corresponding mixture weight is small. The factors $H _ { Q }$ and $W _ { p }$ expose a real coverage cost, rather than an artifact to omit from the guarantee.

For a fixed query and value vectors, the envelope yields a direct output statement.

Lemma 4.2 (Sharp fixed-row deletion bound). Let $\eta = a ( E ) < 1 , \| v _ { j } \| \leq V$ , and

$$
o = \sum _ { j } a _ { j } v _ { j } , \qquad o _ { R } = \sum _ { j \not \in E } \frac { a _ { j } } { 1 - \eta } v _ { j } .\tag{13}
$$

Then $\| o - o _ { R } \| \le 2 V \eta$ . The factor two is attainable on the probability simplex.

Proof. For $\eta > 0$ , write $o = ( 1 - \eta ) o _ { R } + \eta o _ { E }$ . Both conditional averages have norm at most V , so $\left\| o - o _ { \boldsymbol { R } } \right\| = \eta \left\| o _ { E } - o _ { \boldsymbol { R } } \right\| \le 2 V \eta$ . The case $\eta = 0$ is immediate. Assign every discarded value Ve and every retained value $- V e$ for a unit vector e to attain equality. This proof requires fixed queries, retained keys, and values; it is not a theorem about later layers.

## 4.2 Budget feasibility and an optimality-gap certificate

Index proposal constraints by $a = \left( h , u \right)$ and let $\begin{array} { r } { m _ { a i } = \sum _ { j \in c _ { i } } A _ { u , j } ^ { l , h } } \end{array}$ . A removal decision $x _ { i } \in \{ 0 , 1 \}$ has token saving $\textstyle \sum _ { i } n _ { i } x _ { i }$ . For row tolerances $\epsilon _ { a } \in [ 0 , 1 )$ and a minimum retained chunk count $m _ { 0 } \le k$ , the exact screening problem is

$$
\begin{array} { l } { { \displaystyle D _ { l } ^ { * } = \operatorname* { m a x } _ { x \in \{ 0 , 1 \} ^ { k } } \sum _ { i } n _ { i } x _ { i } } \ ~ } \\ { { \mathrm { s u b j e c t ~ t o } \sum _ { i } m _ { a i } x _ { i } \le \epsilon _ { a } ~ \mathrm { f o r ~ e v e r y } ~ a , } } \\ { { \displaystyle \sum _ { i } x _ { i } \le k - m _ { 0 } . } } \end{array}\tag{14}
$$

Here $m _ { 0 } \in \{ 0 , \ldots , k \}$ , so the empty removal is always feasible. A document-token target $B _ { \mathrm { p r e f } } \in [ 0 , N ]$ requires a saving of at least $N - B _ { \mathrm { p r e f } }$ . Cardinality and token budgets are distinct, and retaining $m _ { 0 }$ chunks does not establish retention of a reasoning chain.

The reference proposer sorts by increasing $S _ { i }$ , breaks ties by decreasing $n _ { i }$ and then by original index, and scans once. It adds a chunk to E only if every row constraint and the count constraint remain satisfied. It stops when the token-saving target is reached or no candidate remains. Its sorting cost is $O ( k \log k )$ and its constraint-update cost is $O ( k H _ { Q } W _ { p } )$ once the chunk-mass table is available. This is a feasible heuristic, not an optimizer for (14).

Theorem 4.3 (Dual certificate for feasible token removal). For arbitrary nonnegative multipliers $y _ { a }$ and t, define

$$
U _ { l } ( y , t ) = \sum _ { a } y _ { a } \epsilon _ { a } + t ( k - m _ { 0 } ) + \sum _ { i } \left[ n _ { i } - \sum _ { a } y _ { a } m _ { a i } - t \right] _ { + } .\tag{15}
$$

Every feasible removal with saving $D _ { l }$ satisfies

$$
D _ { l } \leq D _ { l } ^ { * } \leq U _ { l } ( y , t ) .\tag{16}
$$

If $U _ { l } ( y , t ) \ < \ N - \ B _ { \mathrm { p r e f } }$ , the target is infeasible under the stated screening constraints.   
Otherwise the upper bound alone does not establish feasibility.

Proof. For a feasible x, subtract and add the constraint prices:

$$
\sum _ { i } n _ { i } x _ { i } = \sum _ { a } y _ { a } \sum _ { i } m _ { a i } x _ { i } + t \sum _ { i } x _ { i } + \sum _ { i } ( n _ { i } - \sum _ { a } y _ { a } m _ { a i } - t ) x _ { i } .
$$

The first two terms are bounded by their priced capacities. Since $0 \leq x _ { i } \leq 1$ , each final summand is at most the positive part of its coeficient. Taking the maximum over feasible binary x proves the result. □

The certificate requires no assumption of strong duality or integrality. Prices can be chosen directly or obtained from an optional relaxation solver, whose cost must be charged to the implementation. The diference $U _ { l } - D _ { l }$ is a computable upper bound on the proposal’s suboptimality in removed tokens. Failure of the greedy proposal to meet a target is not itself a proof that the target is impossible.

## 5 Independent Auditing and Adaptive Layer Selection

Let $\mathcal { L } \subseteq \{ 1 , \ldots , L - 1 \}$ be a nonempty, prespecified monitored-layer set with $J = | \mathcal { L } |$ . For each layer, define its candidate $E _ { l }$ using the unpruned forward trajectory and proposal information only. These counterfactual candidates need not all be executed after a decision is accepted, but their definitions may not depend on audit samples. In particular, audit results may stop execution but may not be used to change candidates, thresholds, or the observer distribution within the same guarantee.

Let ν be a fixed distribution on user-query positions, for example uniform on the complete query. Fix $\delta \in ( 0 , 1 ) , \epsilon _ { \mathrm { a u d i t } } \in [ 0 , 1 )$ , and an integer $m _ { a } \geq 1$ . Draw $m _ { a }$ audit observers independently with replacement from $\nu ,$ independently of the proposal mechanism. At layer $l ,$ define

$$
\eta _ { l , h } = \mathbb { E } _ { u \sim \nu } [ A _ { u } ^ { l , h } ( E _ { l } ) ] , \qquad \widehat { \eta } _ { l , h } = \frac { 1 } { m _ { a } } \sum _ { s = 1 } ^ { m _ { a } } A _ { U _ { s } } ^ { l , h } ( E _ { l } ) , \quad b = \sqrt { \frac { \ln ( J H _ { Q } / \delta ) } { 2 m _ { a } } } .\tag{17}
$$

The same audit positions may be reused across layers; independence between diferent layer–head estimates is not required. The samples within an estimate are independent draws from ν conditional on the prompt, model, and proposal randomness.

Theorem 5.1 (Simultaneous observer guarantee). With probability at least $1 - \delta$ over the audit sampling, every monitored layer and head satisfies

$$
\eta _ { l , h } \leq \operatorname* { m i n } \{ 1 , \widehat { \eta } _ { l , h } + b \} .\tag{18}
$$

Therefore any layer selected after examining these bounds has $\eta _ { l ^ { * } , h } \leq \epsilon _ { \mathrm { a u d i t } }$ for every head whenever its acceptance rule requires $\widehat { \eta } _ { l ^ { * } , h } + b \leq \epsilon _ { \mathrm { a u d i t } }$

Proof. Conditional on the prompt, model, and proposal mechanism, each audit observation is in [0, 1] with mean $\eta _ { l , h }$ . The one-sided Hoefding inequality gives

$$
\mathbb { P } ( \eta _ { l , h } > \widehat { \eta } _ { l , h } + b ) \le e ^ { - 2 m _ { a } b ^ { 2 } } = \frac { \delta } { J H _ { Q } } .
$$

A union bound over the $J H _ { Q }$ pairs establishes a simultaneous event [18]. Selection of a layer within that event cannot invalidate its already established inequality. Averaging the conditional probability statement over proposal randomness proves the unconditional claim. □

This controls expected discarded mass under $\nu ,$ not the worst observer, every document position, or future decode attention. A deterministic last-token observation does not satisfy the sampling theorem merely because it is called an audit. If b exceeds the configured tolerance, no layer passes; one must enlarge the audit, compute the finite observer population exactly, relax the tolerance with an explicit scope change, or bypass pruning. Using audit outcomes to retune the proposal requires a fresh independent audit or a diferent uniformconvergence argument.

For a valid candidate, define retained chunks $\mathcal { R } _ { l }$ and the set-change statistic

$$
d _ { l } = \frac { | \mathcal { R } _ { l } \triangle \mathcal { R } _ { l ^ { - } } | } { \operatorname* { m a x } \{ 1 , | \mathcal { R } _ { l } \cup \mathcal { R } _ { l ^ { - } } | \} } ,\tag{19}
$$

where $l ^ { - }$ is the preceding monitored layer. A streak of v valid transitions with $d _ { l } \leq \theta$ is an optional scheduling condition; it is not a correctness theorem. The first valid candidate initializes the predecessor. An invalid proposal resets the streak. The method accepts the earliest layer meeting the token target, the proposal constraints, the stability condition, and the audit threshold. If none passes, the complete prompt continues.

## Algorithm 1: EntroPrefill with independent observer audit

1. Fix protected support, monitored layers, all thresholds, proposal observers, audit sample size, and audit distribution before observing audit values.

2. Execute unpruned blocks sequentially; at each monitored layer, extract proposal attention and compute (3)–(9).

3. If residual or document mass is invalid, reset the predecessor and streak and continue without pruning.

4. Construct $E _ { l }$ with the feasible scan for (14); update the candidate-stability streak independently of audit values.

5. If the saving target or streak requirement fails, continue; otherwise evaluate the fixed independent audit and (18).

6. If every head passes, set $l ^ { * } = l .$ , gather hidden states and shallow KV on $R = [ T ] \ \backslash \ E _ { l }$ preserve original positions, and process all remaining blocks on $R .$

7. If the monitored window ends without acceptance, finish ordinary full-context prefill. Decode using the resulting cache and the configured page policy.

## 6 Conditional Transformer Output Stability

## 6.1 Why a shallow certificate is insuficient

Proposition 6.1 (No uniform output bound from shallow attention). Consider attention networks that can preserve a token-specific value coordinate through a shallow prefix and use an unconstrained later query–key map. There exist networks and prompt pairs with identical shallow attention and identical retained states for which the shallow attention mass on a removed position tends to zero, while at least one future-output deletion error remains bounded away from zero. Therefore no bound depending only on shallow discarded attention can vanish uniformly over this class.

Proof. Take two prompts difering only in a value coordinate at a position $j ^ { * }$ , and remove $E = \{ j ^ { * } \}$ . Set every shallow attention-output projection and feed-forward map to zero, so the shallow prefix is the identity. Choose shallow query and key maps that ignore the value coordinate and give mass $\varepsilon > 0$ to $j ^ { * }$ . The retained states and shallow scores agree between prompts, whereas the removed values are $+ V$ and $- V$ for $V > 0$ . A later head uses a separate position-specific key coordinate to give reference mass $1 - \gamma$ to $j ^ { * }$ ; its value map reads only the signed coordinate. Set other values to zero. The two reference outputs are $+ ( 1 - \gamma ) V$ and $- ( 1 - \gamma ) V$ , whereas reduced outputs agree because their inputs agree. The triangle inequality makes at least one error at least $( 1 - \gamma ) V$ . Taking $\varepsilon \to 0$ with fixed $\gamma < 1$ proves the claim; taking $\gamma \to 0$ makes the lower bound approach V . A two-logit readout can make the reference first-token decisions difer. This construction concerns attention-based evidence for deleting $E ;$ it does not assert that every conceivable pruning rule chooses E.

The proposition does not claim impossibility for a fixed model whose later weights and behavior are additionally constrained. It identifies the missing premise in an unconditional inference from shallow concentration. The following theorem states suficient premises for that restricted setting.

## 6.2 Block constants and deletion defects

Consider pre-normalized blocks

$$
B _ { l } ( X ) = X + O _ { l } \operatorname { A t t n } _ { l } ( N _ { l } ( X ) ) , \qquad F _ { l } ( X ) = B _ { l } ( X ) + \operatorname { M L P } _ { l } ( \overline { { N _ { l } } } ( B _ { l } ( X ) ) ) .\tag{20}
$$

Heads are concatenated before $O _ { l }$ . Let $c _ { l }$ and $\bar { c } _ { l }$ be rowwise Lipschitz constants for the two normalizers. Let $f _ { l }$ bound the MLP Lipschitz constant on a convex enclosure of its normalized input domain, and put $R _ { l } = 1 + f _ { l } \bar { c } _ { l }$ . Assume normalized-domain query, key, and value norms are bounded by $\overline { { Q } } _ { l , h } , \overline { { K } } _ { l , h }$ , and $V _ { l , h }$ . The matrices $W _ { l , h } ^ { Q } , W _ { l , h } ^ { K }$ , and $W _ { l , h } ^ { V }$ include the appropriate shared group projections. Biases are absorbed into the norm bounds. Fixed RoPE rotations preserve these norms.

Define suficient constants

$$
a _ { l , h } = \frac { c _ { l } } { \sqrt { d } } \left( \overline { { Q } } _ { l , h } \left. W _ { l , h } ^ { K } \right. + \overline { { K } } _ { l , h } \left. W _ { l , h } ^ { Q } \right. \right) ,\tag{21}
$$

$$
b _ { l , h } = c _ { l } \left\| \boldsymbol { W } _ { l , h } ^ { V } \right\| + 2 V _ { l , h } a _ { l , h } , \qquad K _ { l } = R _ { l } \left( 1 + \left\| \boldsymbol { O } _ { l } \right\| \sqrt { \sum _ { h } b _ { l , h } ^ { 2 } } \right) .\tag{22}
$$

For a fixed removal E and retained set R, let $X ^ { l }$ be the full reference trajectory and

$$
\eta _ { l , h } ^ { \mathrm { r e f } } = \operatorname* { m a x } _ { u \in { \cal R } } \sum _ { j \in { \cal E } } { \cal A } _ { u , j } ^ { l , h } ( X ^ { l - 1 } ) , \qquad e _ { l } = 2 R _ { l } \| O _ { l } \| \sqrt { \sum _ { h } V _ { l , h } ^ { 2 } ( \eta _ { l , h } ^ { \mathrm { r e f } } ) ^ { 2 } } .\tag{23}
$$

The maximum includes every retained query row, not only the user-query observers. Protected causal positions guarantee nonempty retained attention support.

Theorem 6.2 (Conditional first-token perturbation). Under the stated domain bounds, let the reduced trajectory start at $Y ^ { l ^ { * } } = X ^ { l ^ { * } } | _ { R }$ and apply the same blocks on support R with original position IDs. Then

$$
\Delta _ { L } : = \left\| X ^ { L } | _ { R } - Y ^ { L } \right\| _ { \operatorname* { m a x } , 2 } \leq \sum _ { j = l ^ { * } + 1 } ^ { L } e _ { j } \prod _ { s = j + 1 } ^ { L } K _ { s } .\tag{24}
$$

If the final query-row readout is Λ-Lipschitz into the logit $\ell _ { \infty }$ norm, then

$$
\| z - \widetilde { z } \| _ { \infty } \leq \Lambda \Delta _ { L } .\tag{25}
$$

Proof. Appendix A proves that the retained-support block $F _ { l , R }$ is $K _ { l ^ { - } } \mathrm { I }$ ipschitz. Apply the full and restricted block to the same reference retained states. For each retained query, its query and surviving keys and values agree; restricting softmax to R renormalizes the surviving probabilities. Lemma 4.2 bounds the diference for head h by $2 V _ { l , h } \eta _ { l , h } ^ { \mathrm { r e f } }$ . Concatenation, output projection, and the tokenwise post-attention map give

$$
\begin{array} { r } { \left\| F _ { l } ( X ^ { l - 1 } ) \vert _ { R } - F _ { l , R } ( X ^ { l - 1 } \vert _ { R } ) \right\| _ { \operatorname* { m a x } , 2 } \leq e _ { l } . } \end{array}
$$

Add and subtract $F _ { l , R } ( X ^ { l - 1 } | _ { R } )$ to obtain $\Delta _ { l } \le e _ { l } + K _ { l } \Delta _ { l - 1 }$ . The initial discrepancy is zero. Unrolling the recurrence proves (24); the readout Lipschitz inequality proves (25). □

Corollary 6.3 (Decision margin). Let the full-context first-token winner have logit margin $\mathfrak { m } = z _ { ( 1 ) } - z _ { ( 2 ) } > 0$ . If $\mathfrak { m } > 2 \Lambda \Delta _ { L }$ , the reduced and full computations have the same greedy first token.

Proof. The winning logit decreases by at most $\Lambda \Delta _ { L }$ , and any competing logit increases by at most that amount. The remaining margin is strictly positive. □

This is a first-token statement. Extending it to a generated sequence requires new bounds on every shared generation prefix and suficient margins at every step. The theorem supplies neither those future margins nor equality of sampled generations.

## 6.3 A suficient bridge from the probe to deeper layers

A precise support-transfer assumption can connect the probe measure to (23). Suppose that for each deeper layer and head, every retained reference row satisfies

$$
A _ { u } ^ { l , h } ( D ^ { \prime } ) \leq \Gamma _ { l , h } \mu ^ { l ^ { * } } ( D ^ { \prime } ) + \sigma _ { l , h } \quad \mathrm { f o r ~ e v e r y ~ } D ^ { \prime } \subseteq \mathcal { D } ,\tag{26}
$$

where $\Gamma _ { l , h } \geq 0$ and $\sigma _ { l , h } \geq 0$ are known bounds. Then

$$
\eta _ { l , h } ^ { \mathrm { r e f } } \leq \operatorname* { m i n } \{ 1 , \Gamma _ { l , h } q + \sigma _ { l , h } \} , \qquad q = \mu ^ { l ^ { \ast } } ( E ) .\tag{27}
$$

Substitution into (23) and Theorem 6.2 gives a probe-dependent output bound. A suficient construction is a density ratio $A _ { u , j } ^ { l , h } / \mu _ { j } ^ { l ^ { * } } \leq \Gamma _ { l , h }$ on documents, with $\sigma _ { l , h } = 0$ . More generally, the total positive excess $\begin{array} { r } { \sum _ { j \in \mathcal { D } } \left[ A _ { u , j } ^ { l , h } - \Gamma _ { l , h } \mu _ { j } ^ { l ^ { * } } \right] _ { - } } \end{array}$ can be bounded by $\sigma _ { l , h }$ +

Equation (26) is an assumption about later reference attention, not an output of the shallow audit. It can be loose or fail to provide a useful bound: tiny probe probabilities can require large density ratios, and products of $K _ { l }$ can amplify small defects. The implemented acceptance rule certifies its stated observer quantities; an end-to-end certificate additionally requires justified bounds of this stronger kind. In particular, merely measuring stable chunk ranks does not establish (26).

## 7 GQA Unions and Physical Page Budgets

If each sibling head independently selects s token positions and the group retains every selection, then

$$
\mathrm { U O R } _ { g } = \frac { | \bigcup _ { h \in \mathcal { H } _ { g } } { \cal K } _ { h } | } { s } , \qquad 1 \le \mathrm { U O R } _ { g } \le \operatorname* { m i n } \{ r , T / s \} .\tag{28}
$$

For two equal-cardinality sets with Jaccard disagreement D, inclusion–exclusion gives $\mathrm { U O R } = 2 / ( 2 - D )$ . These are set identities, not claims about all implementations of an existing cache policy. A policy that already shares one selection within each physical group does not incur this particular intra-group union.

For decode integration, let $p _ { l , g }$ be a normalized group importance distribution on resident tokens and let page capacity be b. Page-EntroKV motivates the page priority

$$
I _ { l , g } ( P ) = \operatorname* { m a x } _ { j \in P } p _ { l , g } ( j ) .\tag{29}
$$

Protect sink and recent pages, including the writable page, and fill a specified total quota with top-scoring remaining pages. Exact quota cardinality requires enough available pages and a quota at least as large as the protected set. Selecting one common page set for all sibling query heads makes the intra-group UOR one by construction. It does not prove that block-max is statistically preferable to page mass for all tasks.

Proposition 7.1 (Competitive page retention and allocation). If $K \geq 1$ nonprotected pages are selected by block-max and a token has pooled mass greater than $1 / ( K + 1 )$ , its page is retained. Separately, if G groups each choose K logical pages from $N _ { P }$ candidates, their union satisfies

$$
K \le U _ { l } : = | \bigcup _ { g } \mathcal { K } _ { l , g } | \le \operatorname* { m i n } \{ G K , N _ { P } \} .\tag{30}
$$

Proof. If the token’s page were omitted, K distinct competing pages would have maxima at least as large. Those pages and its own page would contribute more than $( K + 1 ) / ( K + 1 ) = 1$ total probability, a contradiction. For the union, any one selected set supplies the lower bound, while subadditivity and the finite candidate pool supply the upper bounds. □

If pages are independently allocated per group, their KV payload is 2edb $\Sigma _ { l , g } K _ { l , g }$ bytes for e bytes per element. If a physical layer block contains all G groups, payload under union materialization is instead $2 e d b G \sum _ { l } U _ { l }$ . If allocation also spans layers, its actual release unit must replace this layer-local accounting. Neither formula includes metadata, padding outside the nominal block, quantization scales, allocator reservation, or transport headers. Whole-page selection avoids token-level holes introduced by partial deletion, but it does not eliminate tail slack or irrelevant tokens co-resident with a retained fact. Group-level UOR equal to one is therefore not a zero-fragmentation theorem.

Decode weights must be initialized for each managed layer; shallow-layer weights cannot stand in for the entire cache. Recomputing scores from a compressed cache accesses resident tokens only. Recovering evicted tokens requires an explicit ofload or recomputation mechanism. Prefill pruning and decode eviction can share scoring concepts, but deleting positive attention mass and renormalizing it is not measure preservation.

## 8 Computational and Serving Consequences

## 8.1 Prefill break-even and score-extraction cost

Let the dominant full-layer floating-point cost be $F _ { l } ( T ) = a _ { l } T + b _ { l } T ^ { 2 }$ , with nonnegative coeficients and the linear term including projections and feed-forward computation. For accepted pruning,

$$
F _ { \mathrm { E P } } = \sum _ { l \le l ^ { * } } F _ { l } ( T ) + \sum _ { l > l ^ { * } } F _ { l } ( T ^ { \prime } ) + F _ { \mathrm { a u x } } .\tag{31}
$$

Arithmetic saving is positive precisely when the removed deep-layer arithmetic exceeds $F _ { \mathrm { a u x } } .$ For identical layers and $\xi = T ^ { \prime } / T$ , the ideal saving before auxiliary work is

$$
\left( 1 - \frac { l ^ { * } } { L } \right) \frac { a T ( 1 - \xi ) + b T ^ { 2 } ( 1 - \xi ^ { 2 } ) } { a T + b T ^ { 2 } } .\tag{32}
$$

For positive total cost it lies between $( 1 - l ^ { * } / L ) ( 1 - \xi )$ and $( 1 - l ^ { * } / L ) ( 1 - \xi ^ { 2 } )$ . With $L = 3 2$ $l ^ { * } = 4$ , and $\xi = 1 / 2$ , these endpoints are 43.75% and 65.625%. They are analytic examples, not measured reductions. A nonzero dense shallow window preserves quadratic asymptotic attention complexity in T.

Exact proposal-row extraction can require $O ( H _ { Q } W _ { p } T d )$ dot-product work per monitored layer. An explicit reference path stores the averaged head distributions in $O ( H _ { Q } T )$ space and the row–chunk mass table in $O ( H _ { Q } W _ { p } k )$ space. Entropy weights are available only after the head statistics are formed, so exact pooling generally needs a subsequent pass. Audit rows add their own capture or recomputation cost. A full $T \times T$ attention matrix is unnecessary, but zero intermediate HBM trafic is not implied.

For one attention row, stable online accumulators satisfy

$$
\begin{array} { l l } { { \ell ^ { \prime } = e ^ { m - m ^ { \prime } } \ell + \displaystyle \sum _ { j \in \mathrm { t i l e } } e ^ { z _ { j } - m ^ { \prime } } , } } \\ { { s ^ { \prime } = e ^ { 2 ( m - m ^ { \prime } ) } s + \displaystyle \sum _ { j \in \mathrm { t i l e } } e ^ { 2 ( z _ { j } - m ^ { \prime } ) } , \quad } } & { { \displaystyle \sum _ { j } A _ { j } ^ { 2 } = s / \ell ^ { 2 } , } } \end{array}\tag{33}
$$

where $m ^ { \prime }$ is the updated maximum. This computes collision for one row. It does not compute collision after observer or head averaging, since averaging before squaring introduces crossproducts. A fused kernel must implement the specified pooled statistic or explicitly introduce a separately analyzed approximation. Register pressure, shared-memory use, L2 displacement, kernel launches, and synchronization require measurement.

Compaction gathers retained states and shallow KV entries and preserves token order and original position IDs. The next generated position follows the original prompt range, not the compact storage length. Pure gathers contribute byte trafic rather than floatingpoint arithmetic; temporary allocations and copies must enter latency and peak-memory accounting. Retained states still encode earlier interaction with the full prompt, so this path is not equivalent to textual truncation followed by a fresh forward pass.

## 8.2 Disaggregation and queueing

DistServe motivates distinct prefill and decode resource pools, while Sarathi-Serve supplies a relevant colocated scheduling alternative [15, 16]. With a dense retained cache and no replication or quantization metadata, transfer payload is $2 e L G d T ^ { \prime }$ bytes. A page-based implementation must use allocated slots and its actual group/layer union instead. If decode selection occurs only after transfer, it does not reduce the bytes already sent.

Writing actual wire payload as $B _ { \mathrm { w i r e } }$ and achieved bandwidth as $\beta _ { \mathrm { e f f } }$ gives $t _ { \mathrm { x f e r } } \geq B _ { \mathrm { w i r e } } / \beta _ { \mathrm { e f f } }$ Queueing, layout conversion, and setup are additional costs. Emitting the first token on the prefill worker can move transfer delay into the first-to-second-token interval rather than removing it. Diferent requests also prune at diferent layers, requiring a defined batch-cohort or device-routing policy; padding retained lengths alone does not resolve depth divergence. Tensor-parallel grouping can introduce score reductions and index broadcasts. Thus a compute reduction, an allocation reduction, and a goodput improvement are separate propositions.

## 9 Scope and Planned Empirical Validation

This manuscript establishes a theoretical procedure and conditional guarantees. It reports no task benchmark, kernel speedup, throughput increase, or empirical noninferiority result.

Theoretical examples and implementation-level algebra checks must not be presented as model experiments. The next empirical phase should determine whether entropy priorities obtain better feasible removals than mass-only or uniform priorities, whether the audit is nonvacuous at practical sample sizes, and whether the conditional stability constants explain observed perturbations.

Evaluation should compare full-context inference, prompt compression, fixed-layer pruning, LazyLLM, SlimInfer, ASL, and implementation-appropriate cache baselines under matched token and physical-byte budgets. Single-hop retrieval, multi-hop bridges, difuse evidence, contradictory answer strings, and reordered documents should be analyzed separately. Key diagnostics are proposal saving $D _ { l } ,$ dual gap $U _ { l } - D _ { l }$ , audit coverage, actual future discarded masses, density-ratio excess, first-token logit margins, full support-chain retention, and bypass frequency. A held-out calibration protocol must fix all thresholds before the reported test evaluation; an input-level observer confidence statement is not a guarantee across future requests or model families.

Accuracy preservation should use a prespecified noninferiority margin and paired uncertainty estimates. Systems measurements should include synchronized TTFT, first-to-secondtoken delay, decode latency, request completion time, peak allocated and reserved memory, transfer bytes, and SLO goodput at equal total GPU budgets. Weight precision, checkpoint revision, workload distribution, compilation, warmup, and arrival process must be fixed or disclosed. An integrated EntroPrefill–Page-EntroKV–disaggregation comparison needs component ablations rather than a single combined speedup.

The primary limitations are structural. Concentrated attention can select a false fact. Bridge relevance can emerge only in deeper layers. An independent audit controls its observer distribution and can be too conservative for short queries. The full-model bound can be vacuous because of large Lipschitz products or poor support transfer. Irreversible deletion limits recovery during long generation and multi-turn reuse. Finally, kernel compatibility, page release semantics, and distributed scheduling remain implementation obligations. These limitations identify the conditions under which the proposed theory can become useful, rather than being exceptions hidden behind a universal preservation claim.

## 10 Conclusion

EntroPrefill couples a sink-isolated Rényi proposal with explicit deletion constraints and independent adaptive-layer auditing. The analysis characterizes the coverage lost through head pooling, bounds attainable token removal, and states suficient conditions for first-token output stability. A complementary counterexample prevents shallow attention from being interpreted as an unconditional certificate. Physical allocation and serving costs remain explicit in the composition with page eviction and disaggregation. Establishing practical advantage requires implementing this exact procedure and testing whether its guarantees are informative at useful pruning budgets.

## A Proof of the Transformer Block Constants

## A.1 Normalizer and softmax bounds

For regularized LayerNorm in dimension $d _ { m }$ , let $P = I - \mathbf { 1 } \mathbf { 1 } ^ { \top } / d _ { m } , \ z = P x$ , and $v =$ $\| z \| ^ { 2 } / \bar { d } _ { m } + \epsilon _ { \mathrm { L N } }$ , with $\epsilon _ { \mathrm { L N } } > 0$ . With learned scale $\gamma$ and bias $\beta$

$$
N ( x ) = \gamma \odot { \frac { z } { \sqrt { v } } } + \beta , \qquad J _ { N } ( x ) = \mathrm { d i a g } ( \gamma ) \left( { \frac { P } { \sqrt { v } } } - { \frac { z z ^ { \top } } { d _ { m } v ^ { 3 / 2 } } } \right) .\tag{34}
$$

On the mean-zero subspace, directions orthogonal to $z$ have eigenvalue $1 / \sqrt { v }$ before scaling, while the direction of $z$ has eigenvalue $\epsilon _ { \mathrm { L N } } / v ^ { 3 / 2 }$ . The constant direction has eigenvalue zero. Consequently,

$$
\| J _ { N } ( x ) \| \leq \frac { \| \gamma \| _ { \infty } } { \sqrt { \epsilon _ { \mathrm { L N } } } } , \qquad \| N ( x ) \| \leq \| \gamma \| _ { \infty } \sqrt { d _ { m } } + \| \beta \| .\tag{35}
$$

The first is a global Lipschitz bound; the second supplies a bounded normalized domain for query, key, and value projections. RMS normalization admits the corresponding argument with $P = I$ . The constants can be very large at small numerical regularization and should not be confused with tight empirical estimates.

For $p = \operatorname { s o f t m a x } ( z )$ and perturbation direction $v ,$ its Jacobian gives $( J v ) _ { i } = p _ { i } ( v _ { i } { - } \sum _ { j } p _ { j } v _ { j } )$ Therefore

$$
\left\| J v \right\| _ { 1 } \leq \sum _ { i } p _ { i } \left( \left| v _ { i } \right| + \left| \sum _ { j } p _ { j } v _ { j } \right| \right) \leq 2 \left\| v \right\| _ { \infty } .\tag{36}
$$

Integration along the segment between logits proves $\begin{array} { r l } { \Vert \mathrm { s o f t m a x } ( z ) - \mathrm { s o f t m a x } ( z ^ { \prime } ) \Vert _ { 1 } } & { { } \leq } \end{array}$ $2 \left\| z - z ^ { \prime } \right\| _ { \infty } .$ . This deliberately conservative cross-norm constant sufices here; no optimality claim is made for it. For causal attention, the calculation is performed on each row’s fixed allowed support.

## A.2 Attention and feed-forward composition

Let two retained-state matrices difer by at most $\Delta$ in row-maximum norm. Their normalized states difer by at most $c _ { l } \Delta$ . By adding and subtracting a mixed query–key product, every head logit changes by at most

$$
\frac { \overline { { { Q } } } _ { l , h } \left\| W _ { l , h } ^ { K } \right\| + \overline { { { K } } } _ { l , h } \left\| W _ { l , h } ^ { Q } \right\| } { \sqrt { d } } c _ { l } \Delta = a _ { l , h } \Delta .
$$

The corresponding value diference is at most $c _ { l } \left\| W _ { l , h } ^ { V } \right\| \Delta$ . Decompose the output diference into a value-change term and a probability-change term. Equation (36) bounds the latter by $2 V _ { l , h } a _ { l , h } \Delta$ , so the total is at most $b _ { l , h } \Delta$ . Concatenation across heads changes the row norm by at most $\sqrt { \sum _ { h } b _ { l , h } ^ { 2 } } \Delta$ . Multiplication by $O _ { l }$ and addition of the attention residual produce the factor $1 + \left. O _ { l } \right. \sqrt { \textstyle \sum _ { h } b _ { l , h } ^ { 2 } }$

The tokenwise map $x \mapsto x { + } \mathrm { M L P } _ { l } ( \overline { { N } } _ { l } ( x ) )$ has Lipschitz constant at most $R _ { l } = 1 + f _ { l } \bar { c } _ { l }$ , proving (22). For a two-matrix MLP $W _ { 2 } \phi ( W _ { 1 } x + b _ { 1 } ) + b _ { 2 }$ , one may take $f _ { l } \leq \| W _ { 2 } \| L _ { \phi } \| W _ { 1 } \|$ when $\phi$ is $L _ { \phi } \mathrm { - L i p s c h i t z }$ . For gated MLPs, a bound on their actual Jacobian over a convex bounded enclosure of the normalized domain is required instead; the two-matrix expression cannot be substituted unchanged. The final readout constant can be taken as $\Lambda = \| W _ { \mathrm { o u t } } \| _ { 2 \to \infty }$ c<sub>final</sub> when final normalization has Lipschitz constant $c _ { \mathrm { f i n a l } }$

## B Limits of Entropy-Only and Cross-Phase Guarantees

The mixture envelope depends on both head coverage and the distributional discrepancy. A low collision entropy alone controls neither the value directions nor the truth of the attended statement. The sharp construction in Lemma 4.2 shows that value information is necessary to improve a worst-case $2 V \eta$ guarantee. Conversely, equal values can make deletion harmless despite a large discarded mass. Thus the attention-only bound is conservative for some prompts and cannot be universally tightened using entropy without further assumptions.

Likewise, a common Rényi functional does not imply conservation across page partitions. Consider one chunk made of two equal pages of size $b ,$ with masses $1 - \alpha$ and $\alpha _ { \mathrm { { ; } } }$ , uniform within pages. Its negative collision entropy is

$$
\Phi _ { 2 } ( c ) = - \ln b + \ln ( ( 1 - \alpha ) ^ { 2 } + \alpha ^ { 2 } ) .\tag{37}
$$

If a page score is its conditional negative collision entropy plus log page mass, the arithmetic mean of the two page scores is

$$
{ \frac { \Psi ( P _ { 1 } ) + \Psi ( P _ { 2 } ) } { 2 } } = - \ln b + { \frac { 1 } { 2 } } \ln ( \alpha ( 1 - \alpha ) ) .\tag{38}
$$

The diference diverges as $\alpha  0 ,$ , despite perfect page alignment and identical distributions in the two phases. A boundary-only error term therefore cannot justify such a conservation identity. The admissible cross-phase statements in this paper are explicit resource counts and conditional output inequalities, not measure preservation.

## C Implementation Invariants and Reproducibility Contract

An implementation must reproduce observer averaging before entropy, the regularization floor, complete-chunk selection, deterministic ties, and the independence condition of Theorem 5.1. Row-wise collision accumulation followed by averaging is a diferent operator. The audit must not silently reuse proposal-dependent sample selection. Threshold changes triggered by audit failures require a separately justified protocol. Empty candidate sets, all-ineligible groups, nonfinite statistics, insuficient protected-page capacity, and failed token targets must have explicit fallbacks.

Retained positions must stay in original order, protected positions must survive, and logical sequence positions must remain distinct from compact storage ofsets. Query, key, value, normalization, and readout constants in Theorem 6.2 must correspond to the same checkpoint and numerical execution assumptions. A floating-point implementation may require additional rounding-error terms if a formal numerical certificate is claimed. Analytic operator bounds in real arithmetic are not automatically machine-verified guarantees.

The future artifact should release the exact scoring reference, optimized-kernel comparisons against it, the candidate and audit observer indices, constraint tables, dual prices when used, retained-index maps, and per-request resource traces. Experiments should retain failed and bypassed cases in the denominator. This contract makes the theoretical objects reproducible without treating implementation completion or empirical validation as already established.

## References

[1] H. Jiang et al. LLMLingua: Compressing prompts for accelerated inference of large language models. EMNLP, 2023. https://arxiv.org/abs/2310.05736.

[2] Y. Fang et al. AttentionRAG: Attention-guided context pruning in retrieval-augmented generation. 2025. https://arxiv.org/abs/2503.10720.

[3] Q. Fu et al. LazyLLM: Dynamic token pruning for eficient long context LLM inference. 2024. https://arxiv.org/abs/2407.14057.

[4] L. Long et al. SlimInfer: Accelerating long-context LLM inference via dynamic token pruning. 2025. https://arxiv.org/abs/2508.06447.

[5] R. Taniguchi et al. Adaptive layer selection for layer-wise token pruning in LLM inference. Findings of ACL, 2026. https://arxiv.org/abs/2601.07667.

[6] Z. Zhang et al. H2O: Heavy-hitter oracle for eficient generative inference of large language models. 2023. https://arxiv.org/abs/2306.14048.

[7] Y. Li et al. SnapKV: LLM knows what you are looking for before generation. 2024. https: //arxiv.org/abs/2404.14469.

[8] Y. Feng et al. Ada-KV: Optimizing KV cache eviction by adaptive budget allocation for eficient LLM inference. 2024; revised 2025. https://arxiv.org/abs/2407.11550.

[9] J. Ainslie et al. GQA: Training generalized multi-query transformer models from multi-head checkpoints. EMNLP, 2023. https://arxiv.org/abs/2305.13245.

[10] Inbasekaran S. Page-EntroKV: Hardware-aligned, entropy-weighted KV-cache eviction under grouped-query attention. Unpublished author manuscript supplied with this work, 2026.

[11] G. Xiao et al. Eficient streaming language models with attention sinks. ICLR, 2024. https: //arxiv.org/abs/2309.17453.

[12] W. Wu et al. Retrieval head mechanistically explains long-context factuality. 2024. https: //arxiv.org/abs/2404.15574.

[13] T. Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. 2023. https://arxiv.org/abs/2307.08691.

[14] W. Kwon et al. Eficient memory management for large language model serving with PagedAttention. SOSP, 2023. https://arxiv.org/abs/2309.06180.

[15] Y. Zhong et al. DistServe: Disaggregating prefill and decoding for goodput-optimized large language model serving. 2024. https://arxiv.org/abs/2401.09670.

[16] A. Agrawal et al. Taming throughput-latency tradeof in LLM inference with Sarathi-Serve. 2024. https://arxiv.org/abs/2403.02310.

[17] T. van Erven and P. Harremoës. Rényi divergence and Kullback–Leibler divergence. IEEE Transactions on Information Theory, 60(7):3797–3820, 2014. https://arxiv.org/abs/1206. 2459.

[18] W. Hoefding. Probability inequalities for sums of bounded random variables. Journal of the American Statistical Association, 58(301):13–30, 1963. https://doi.org/10.1080/01621459. 1963.10500830.