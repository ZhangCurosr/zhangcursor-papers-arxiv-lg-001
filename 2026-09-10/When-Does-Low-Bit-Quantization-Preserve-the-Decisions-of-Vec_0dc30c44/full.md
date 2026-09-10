# When Does Low-Bit Quantization Preserve the Decisions of Vector Search?

Wenxuan Xiao Astrmira Tech.

w.xiao@astrmira.com

Xu Cao Astrmira Tech.

x.cao@astrmira.com

## Abstract

The same two-bit coordinate code that recovers 95% of exact nearest neighbours on Cohere text embeddings recovers 2% on GIST image descriptors. Average distortion does not explain the gap. A graph-based search algorithm never consumes a distance estimate on its own; it consumes comparisons, and a comparison fails only when quantization noise crosses the specific decision boundary that the comparison sits on. We develop a theory of low-bit vector search at this level.

The first result is a distribution-free decomposition: the probability that a comparison flips is at most the probability mass of exact margins near zero plus the tail probability of the calibrated residual. Global fidelity metrics average over both quantities and therefore cannot separate them. The second result treats residual dependence induced by shared structure. In the inspected Cohere selected-pair population, residuals sharing a query have strong pooled correlation, and the measured variance of their diference is 7.4 times smaller than the sum of their marginal variances. The third result is a deterministic coupling theorem for the neighbour-selection call of Vamana: for a frozen candidate permutation, the approximate replay produces the same neighbour list exactly when every candidatelevel pruning action agrees with the exact one, and the first disagreement identifies the edge at which the outputs diverge.

We connect these decision-level results to representation geometry through an exact Gaussian oracle. Stein’s lemma gives a closed-form sign–linear covariance identity and shows how of-diagonal covariance enters the ranking signal; an aligned bilinear model yields a strict correlation gain from a deterministic magnitude bit; and a rare-contamination construction proves that marginal Gaussianity, thin-shell concentration, and regular spectra cannot by themselves imply the exponential tails the bounds require. For representations outside the analytical regime, a held-out block certificate bounds the selective failure risk of a frozen quantized rule from data alone.

On learned, classical, and synthetic embeddings, standardized exact margins predict held-out flip rates with Spearman correlation 0.97 for ranking and 0.99 for pruning, against 0.74 and 0.05 for global rank correlation. The coupling identity holds in all 960 sampled selection calls. Direct-ratio selective certificates at 512 blocks average 16.7% for ranking and 18.7% for pruning; no selected empirical validation risk exceeds its certificate. A random rotation raises Cohere’s global rank fidelity from 0.58 to 0.93 while leaving one held-out flip rate unchanged and raising another by half, so fidelity does not determine the sign of the change in decision risk. The framework covers coordinate binary codes, RaBitQ, Lucene BBQ, and product quantizers through a common decision interface.

Keywords: vector search, binary quantization, decision stability, concentration inequalities, contrastive representations

## 1 Introduction

Binary quantization compresses a 768-dimensional float vector to 96 bytes and replaces most floating-point distance arithmetic with packed integer operations. Two recent systems exploit low-bit codes with opposite strategies. QuIVer (Xiao et al., 2026) keeps the coordinate axes of the encoder and builds and searches a proximity graph directly on twobit scores; RaBitQ (Gao and Long, 2024) applies a random rotation before one-bit coding and corrects the estimate. Both reach competitive recall on their benchmarks. Yet the two-bit coordinate index of QuIVer reaches 95% recall at ten neighbours on Cohere-768 embeddings and 2% on GIST-960 descriptors (Xiao et al., 2026), and the classical analysis of sign codes does not predict either number: the SimHash collision law (Charikar, 2002) describes random projection directions, not the fixed coordinate axes of a trained encoder, and it says nothing about how the training objective shapes those axes.

The right level of analysis. Prior work measures score fidelity: how well approximate distances track exact ones, summarized by mean squared error, Spearman correlation, or end-to-end recall. A graph algorithm does not consume scores in this form. It executes a sequence of binary comparisons (sort these candidates, prune that edge, advance along this path), and a comparison has consequences only when it crosses its decision boundary. Two features of this setting are invisible to fidelity metrics. First, the comparisons that matter have small margins, because candidate sets are selected to be close to the query. Second, the two residuals entering a comparison can be dependent because the corresponding distances share a query or graph node, so the variance of their diference includes a covariance term.

Example 1 (Shared error cancels in a comparison) On Cohere embeddings with twobit codes, the calibrated errors of the two distances in a nearest-neighbour comparison have standard deviations 0.024 and 0.027 in cosine-distance units. Their measured diference has standard deviation 0.013, compared with 0.036 after omitting the empirical covariance term; the pooled residual correlation is 0.87. A metric that averages individual distance errors does not expose this diference residual.

This paper develops the theory at the level of individual comparisons: their exact margins, their correlated residuals, and the way they compose inside a graph algorithm.

Contributions. The results form four layers.

1. A decision-level risk decomposition (§3). For any comparison with exact value Ψ and calibrated residual R,

$$
\mathbb { P } ( \mathrm { H i p } ) \ \le \ \underbrace { \mathbb { P } ( 0 < | \Psi | \le \tau ) } _ { \mathrm { b o u n d a r y ~ m a s s } } + \underbrace { \mathbb { P } ( | R | \ge c \tau ) } _ { \mathrm { r e s i d u a l ~ t a i l } }
$$

for every $\tau > 0$ , with no distributional assumption. Instantiated for ranking and for Vamana pruning, the diference residual has an exact covariance-aware secondmoment identity and admits a tail bound under a joint MGF proxy. On the inspected Cohere selected pairs, omitting empirical covariance overstates the measured diference variance by a factor of 7.4.

2. Frozen-trace composition (§4). A Vamana selection call is a deterministic state machine: a candidate ordering, a sequence of pruning actions, and a neighbour list. For a frozen ordering, the approximate replay returns the exact list if and only if every candidate-level action evaluated on the frozen exact state agrees, and the first disagreement is the exact point of divergence. Local bounds therefore compose into trace-, edge-, and path-level certificates.

3. The representation bridge and its limits (§5). Under an exact Gaussian oracle, Stein’s lemma gives a sign–linear covariance identity, and an aligned bilinear model gives a strict correlation gain from a magnitude bit. A rare-contamination construction proves that fixed-dimensional Gaussianity, thin-shell concentration, and an isotropic spectrum are jointly insuficient for useful exponential tails; concentration needs one of three additional ingredients, which we supply.

4. Operational certificates and quantizer scope (§6, §7). A held-out block certificate bounds the selective risk of a frozen quantized rule with finite-sample validity and no analytical assumption. RaBitQ, Lucene BBQ, and product quantizers enter the same decision interface through family-specific residual mechanisms.

Scope. The theory covers fixed candidate sets and frozen execution traces. End-to-end recall additionally depends on candidate coverage, which requires separate graph-expansion arguments and is outside this paper.

Notation. Throughout, $d ( \cdot , \cdot )$ is an exact distance or dissimilarity and $\hat { d } _ { Q }$ its quantized approximation; Ψ is a signed decision functional whose sign selects an action; $\Gamma = | \Psi |$ is the exact margin; R is the calibrated residual; $c _ { Q } > 0$ is the calibration scale; and v is a residual-tail parameter. For $G \sim { \mathcal { N } } ( { \boldsymbol { \mu } } , { \boldsymbol { \Sigma } } )$ we write $\sigma _ { i } ^ { 2 } = \Sigma _ { i i } , \rho _ { i j } = \Sigma _ { i j } / ( \sigma _ { i } \sigma _ { j } )$ , and $\varphi$ for the standard normal density.

## 2 Related Work

Binary codes and random projections. SimHash (Charikar, 2002) shows that random hyperplane sign bits preserve angular similarity, with collision probability $1 - \operatorname { a r c c o s } \langle x , y \rangle / \pi$ for fixed vectors. The randomness lives in the projection directions; the result does not describe the fixed coordinate axes of a learned encoder, whose per-coordinate variances difer by an order of magnitude. RaBitQ (Gao and Long, 2024) gives a sharp pointwise error bound for a randomized ratio estimator of a single inner product. Product quantization (J´egou et al., 2011; Ge et al., 2014) and ScaNN (Guo et al., 2020) operate with learned codebooks at 4 to 64 bits. These methods characterize individual distance estimates; our decision analysis studies diferences of estimates with shared structure.

Graph-based nearest-neighbour search. HNSW (Malkov and Yashunin, 2020) and Vamana (Subramanya et al., 2019) build navigable graphs by iterated candidate sorting and diversity pruning, and QuIVer (Xiao et al., 2026) runs both operations on two-bit scores. Existing theory studies navigability under random-graph or geometric assumptions. We ask a diferent question, whether quantization preserves the local decisions made during construction and traversal, and answer it with a deterministic composition theorem for the selection state machine.

Contrastive representation geometry. Betser et al. (2026) prove asymptotic Gaussianity of fixed-dimensional projections under stated alignment and concentration assumptions, with a second regime that adds a vanishing regularizer. We use this result to motivate an exact Gaussian oracle; approximate Gaussian diagnostics alone do not transfer its identities or tails without error control.

Concentration and margin conditions. The boundary-plus-tail split is the low-noise condition of Tsybakov (2004) transported to algorithmic comparisons. The residual side is supplied by sub-Gaussian and sub-gamma tail bounds (Vershynin, 2018; Boucheron et al., 2013), by Efron–Stein replacement bounds (Efron and Stein, 1981), and by empirical Bernstein inequalities (Maurer and Pontil, 2009). The new element is the combination of rolespecific quantization residuals, covariance-aware scales, and frozen algorithmic traces in one framework.

## 3 Decisions, Boundaries, and Residual Tails

Every step of graph-based vector search reduces to the question “is Ψ positive or negative?”, where Ψ is a diference of distances (ranking), a scaled comparison (pruning), or a threshold test (stopping). Quantization replaces Ψ by an approximation $\widehat { \Psi } _ { Q }$ , and the step fails when the two disagree in sign. Table 1 summarizes what each result in the paper assumes and what it delivers; the rest of the paper fills in the rows.

Table 1: The results of the paper, the source of randomness each one conditions on, and the object each one controls.
<table><tr><td>Result</td><td>Randomness</td><td>Key assumption</td><td>Controls</td><td>Does not control</td></tr><tr><td>Boundary- residual split (Thm. 2)</td><td>any common probability space</td><td>positive calibration scale; exact ties handled separately</td><td>flip probability of one comparison</td><td>residual concentration</td></tr><tr><td>Covariance-aware tails (Props. 5, 6)</td><td>quantizer or data randomness with the compared</td><td>joint sub-Gaussian proxy and bias term</td><td>one-sided ranking and pruning flip risk</td><td>MGF control from sample covariance alone</td></tr><tr><td>Trace coupling (Thm. 8)</td><td>objects fixed none: deterministic state machine</td><td>shared permutation, append-only selection, one tie rule</td><td>equality of neighbour lists; first divergence</td><td>candidate generation, navigability, or recall</td></tr><tr><td>Gaussian oracle (Thm. 11, Prop. 12)</td><td>exact joint Gaussian representation</td><td>stated target, scorer, threshold</td><td>sign-linear covariance; aligned Pearson gain</td><td>arbitrary scorers or decision risk</td></tr><tr><td>Necessity (Thm. 13)</td><td>explicit construction</td><td>none</td><td>impossibility of tails from low-order</td><td>behavior of every real embedding</td></tr><tr><td>Selective certificate (Thm. 14)</td><td>i.i.d. blocks given positive coverage; a frozen fit</td><td>prespecified grid</td><td>diagnostics ratio of expected failure to expected coverage</td><td>distribution shift or end-to-end recall</td></tr></table>

## 3.1 Setup and the main decomposition

Definition 1 (Decision residual and margin) Let Ψ and $\widehat { \Psi } _ { Q }$ be the exact and approximate decision functionals on a common probability space. For a calibration constant $c _ { Q } > 0$ define

$$
R = \widehat \Psi _ { Q } - c _ { Q } \Psi , \Gamma = | \Psi | .\tag{1}
$$

The conservative flip event is $\mathcal { E } = \{ \Psi \widehat { \Psi } _ { Q } \leq 0 , \Psi \neq 0 \}$ : an approximate tie against a non-tied exact decision counts as a failure.

The constant $c _ { Q }$ absorbs multiplicative distortion and is fitted by least squares on an independent sample. A through-origin fit makes R orthogonal to Ψ in the sample, but it does not make R mean zero; the risk bounds below therefore carry an explicit bias term, and in practice we use afine calibration.

Theorem 2 (Boundary–residual decomposition) For every $\tau > 0$

$$
\begin{array} { r } { \mathbb { P } ( \mathcal { E } ) \ \leq \ \underbrace { \mathbb { P } ( 0 < \Gamma \leq \tau ) } _ { b o u n d a r y \ m a s s } + \underbrace { \mathbb { P } ( | R | \geq c _ { Q } \tau ) } _ { r e s i d u a l \ t a i l } . } \end{array}\tag{2}
$$

No independence between R and Ψ is required.

Proof On E the residual opposes the sign of Ψ with magnitude at least $c _ { Q } \Gamma > 0$ . Split according to whether $\Gamma \leq \tau$ or $\Gamma > \tau ;$ in the second case $| R | \geq c _ { Q } \Gamma > c _ { Q } \tau$ ■

The two terms are the two failure mechanisms that fidelity metrics merge. High average fidelity coexists with large boundary mass on a hard candidate set, and poor average fidelity is harmless when every margin of interest is large. Rank correlation and boundary crossings are diferent functionals of the same joint law, which is why a Spearman coeficient computed on random pairs does not determine the flip rate on selected pairs (§8 gives an explicit construction in which the two are decoupled).

Corollary 3 (Sub-Gaussian instantiation) $I f \mathbb { P } ( 0 < \Gamma \leq \tau ) \leq C \tau ^ { \beta } \ f o r \ 0 < \tau \leq \tau _ { 0 }$ and $\begin{array} { r } { \mathbb { P } ( | R | \ge z ) \le 2 \exp ( - z ^ { 2 } / 2 v ^ { 2 } ) } \end{array}$ , then

$$
\mathbb { P } ( \mathcal { E } ) \leq \operatorname* { i n f } _ { 0 < \tau \leq \tau _ { 0 } } \left[ C \tau ^ { \beta } + 2 \exp \left( - \frac { c _ { Q } ^ { 2 } \tau ^ { 2 } } { 2 v ^ { 2 } } \right) \right] ,\tag{3}
$$

and the minimizing $\tau ^ { \star }$ scales as $( v / c _ { Q } ) [ \log ( 1 / v ) ] ^ { 1 / 2 }$ . The resulting rate is of order

$$
( v / c _ { Q } ) ^ { \beta } [ \log ( c _ { Q } / v ) ] ^ { \beta / 2 }
$$

(Appendix A).

Remark 4 (The boundary exponent belongs to the embedding) Fitting the empirical margin distribution on logarithmic axes over the inspected quantile window gives $\beta =$ 1.526 on Cohere $( R ^ { 2 } ~ = ~ 0 . 9 9 4 )$ , 1.500 on MiniLM $\left( R ^ { 2 } \ = \ 0 . 9 7 9 \right)$ , and 1.595 on GIST $( R ^ { 2 } = 0 . 9 8 9 )$ , identical for one- and two-bit codes on the same data. The boundary-mass term can therefore be estimated from the embedding before any quantizer is chosen, and it tells a practitioner how much residual control a given decision population will demand.

## 3.2 Ranking: the shared-query correlation

For a query q and candidates x, y with $d ( q , x ) < d ( q , y )$ , the ranking functional and its residual are

$$
\Psi _ { \mathrm { r a n k } } = d ( q , y ) - d ( q , x ) > 0 , \qquad R _ { \mathrm { r a n k } } = \xi _ { y } - \xi _ { x } ,\tag{4}
$$

where $\xi _ { x } = \hat { d } _ { Q } ( q , x ) - c _ { Q } d ( q , x )$ is the calibrated error on the edge $( q , x )$ . Both errors involve the same quantized query, so they share a common component.

Proposition 5 (Covariance-aware ranking) Fix q, x, y before drawing the approximation randomness, and let $( \xi _ { x } , \xi _ { y } )$ have mean m and joint sub-Gaussian proxy matrix K. The ranking residual is sub-Gaussian with scale

$$
\nu _ { \mathrm { r a n k } } ^ { 2 } = K _ { x x } + K _ { y y } - 2 K _ { x y } ,\tag{5}
$$

and $f o r$ exact margin $\gamma > 0$ and bias $\mu _ { R } = \mathbb { E } ( \xi _ { y } - \xi _ { x } )$ 2

$$
\mathbb { P } _ { Q } ( r a n k \ : \ : \mathcal { \ : } \mathrm { \# } p ) \leq \exp \left[ - \frac { ( c _ { Q } \gamma + \mu _ { R } ) _ { + } ^ { 2 } } { 2 \nu _ { \mathrm { r a n k } } ^ { 2 } } \right] .\tag{6}
$$

The empirical cross term $( \widehat { \Sigma } _ { \xi } ) _ { x y }$ measures shared-query covariance in the selected-pair population, where $\widehat { \Sigma } _ { \xi }$ is the sample covariance of $( \xi _ { x } , \xi _ { y } )$ . It gives the exact second-moment identity

$$
\widehat { \mathrm { V a r } } ( \xi _ { y } - \xi _ { x } ) = ( \widehat { \Sigma } _ { \xi } ) _ { x x } + ( \widehat { \Sigma } _ { \xi } ) _ { y y } - 2 ( \widehat { \Sigma } _ { \xi } ) _ { x y } .\tag{7}
$$

This empirical covariance is distinct from the joint MGF proxy K in Proposition 5.

Table 2: Empirical residual correlation on fixed 32-neighbour candidate sets (top candidate against each competitor), pooled over queries and competitors. The independence ratio is $( \widehat { \mathrm { V a r } } ( \xi _ { x } ) + \widehat { \mathrm { V a r } } ( \xi _ { y } ) ) / \widehat { \mathrm { V a r } } ( \xi _ { y } - \xi _ { x } )$
<table><tr><td>Dataset</td><td>Quantizer</td><td>Residual corr.  $\rho _ { x y }$ </td><td>Independence ratio</td><td>Flip rate</td></tr><tr><td>Cohere-768</td><td>1-bit</td><td>0.794</td><td>4.71×</td><td>8.89%</td></tr><tr><td>Cohere-768</td><td>2-bit</td><td>0.870</td><td>7.42×</td><td>5.83%</td></tr><tr><td>Cohere-768</td><td>rotated 2-bit</td><td>0.435</td><td>1.77×</td><td>8.81%</td></tr><tr><td>MiniLM-384</td><td>1-bit</td><td>0.047</td><td>1.05×</td><td>11.21%</td></tr><tr><td>MiniLM-384</td><td>2-bit</td><td>0.093</td><td>1.10×</td><td>4.19%</td></tr><tr><td>GIST-960†</td><td>2-bit</td><td>0.832</td><td>5.70×</td><td>35.83%</td></tr></table>

<sup>†</sup>The eligibility gate of §6 rejects this configuration (calibration slope near zero, left tail 7.8× Gaussian); the row is included for comparison.

Two facts about this correlation matter. First, it belongs to the selected-pair population rather than the i.i.d. endpoint model. For i.i.d. endpoints and a single symmetric kernel, the Hoefding decomposition bounds shared-endpoint correlation by $1 / 2$ and the independence ratio by 2 (Appendix B). Second, substituting empirical covariance into the Gaussian-tail expression gives an informative plug-in diagnostic on the inspected eligible configurations:

it is below one and covers the observed flip rate in each case (0.55 against an observed 5.8% for Cohere two-bit codes), whereas omitting the cross term makes the expression equal one on every Cohere configuration. This substitution is not the analytical assumption itself; the concentration routes of $\ S 5$ provide tail control under their stated conditions, and the held-out route of §6 provides an independent alternative. A six-bin margin-conditional scale changes predictive correlation by at most 0.002, so we retain one global scale in the reported experiments.

## 3.3 Pruning: role-specific residuals

Vamana’s RobustPrune rule declares candidate c dominated by an already selected neighbour s of target t when

$$
\Psi _ { \alpha } ( t , c , s ) = d ( t , c ) - \alpha d ( s , c ) \geq 0 , \qquad \alpha \geq 1 .\tag{8}
$$

The residual combines two distances that share the node c rather than a query:

$$
Z _ { \alpha } = \xi _ { t c } - \alpha \xi _ { s c } , \qquad \nu _ { \alpha } ^ { 2 } = K _ { t c , t c } + \alpha ^ { 2 } K _ { s c , s c } - 2 \alpha K _ { t c , s c } .\tag{9}
$$

A shared additive intercept in the edge calibration cancels in the ranking diference but survives in $Z _ { \alpha }$ as $( 1 - \alpha ) b _ { \mathrm { : } }$ , so pruning needs its own bias term.

Proposition 6 (Fixed-triple pruning) Fix the triple $( \boldsymbol { t } , \boldsymbol { c } , s )$ before drawing the approximation randomness, and let $Z _ { \alpha }$ have joint sub-Gaussian scale $\nu _ { \alpha }$ and mean $\mu _ { Z }$ . For exact margin $\gamma = | \Psi _ { \alpha } | > 0$

• a false keep (missed prune, $\Psi _ { \alpha } > 0 )$ has probability at most $\exp [ - ( c _ { Q } \gamma + \mu _ { Z } ) _ { + } ^ { 2 } / ( 2 \nu _ { \alpha } ^ { 2 } ) ]$ ;

• a false prune $( \Psi _ { \alpha } < 0 )$ has the same form with efective margin $c _ { Q } \gamma - \mu _ { Z } ,$

• at a structural tie $( \Psi _ { \alpha } = 0 )$ disagreement is the event $Z _ { \alpha } < 0$ , whose probability must be carried explicitly; a no-atom condition on $Z _ { \alpha }$ does not control it.

The two orientations behave very diferently on real data: across all sampled configurations in §8.4, every observed pruning disagreement is a false keep.

## 3.4 Fixed top-K on a frozen candidate set

For a candidate set C frozen before approximate scoring, with exact top-K subset $A \subset C$

$$
\mathbb { P } _ { Q } ( \widehat { A } \neq A ) \leq \sum _ { x \in A } \sum _ { y \in C \backslash A } \mathbb { P } _ { Q } ( \mathrm { r a n k \ f i i p \ o n } \ ( x , y ) ) ,\tag{10}
$$

and any deterministic suficient comparison set may replace the full cross product.

## 4 From Local Decisions to Algorithmic Traces

A Vamana selection call makes dozens of comparisons in sequence, each depending on the outcome of earlier ones. This section shows that local bounds compose into a statement about the whole call once the exact execution state is frozen.

## 4.1 The selection call as a state machine

Fix a target t, a candidate permutation $\pi = ( c _ { ( 1 ) } , \ldots , c _ { ( N ) } )$ sorted by exact distance with deterministic tie-breaking, a degree budget M, and $\alpha \geq 1$ . Starting from $S = ( )$ , scan π until it is exhausted or $| S | = M$ . At candidate c form the candidate-level action

$$
D ( c ; S ) = \bigvee _ { s \in S } \mathbf { 1 } \{ d ( t , c ) - \alpha d ( s , c ) \geq 0 \} ,\tag{11}
$$

and append c exactly when $D ( c ; S ) = 0$ . This is the standard non-saturated RobustPrune call. Any deterministic refill applied afterwards is a separate post-processing map.

Definition 7 (Semantic trace) The semantic trace T records the shared permutation, each visited candidate, its selected prefix, its candidate-level action, and the termination state. It does not record implementation-dependent short-circuit order among witnesses.

## 4.2 The coupling theorem

Theorem 8 (Trace coupling) Let the approximate replay use the same candidate permutation, unique candidate labels, degree budget, and tie convention. For each exact visited candidate j, evaluate the approximate candidate-level action on the frozen exact prefix $S _ { j }$ and let $E _ { j }$ be the event that it difers from the exact action. Then, for the append-only non-saturated call,

$$
\{ \widehat { T } \neq T \} = \{ \widehat { S } _ { \mathrm { o u t } } \neq S _ { \mathrm { o u t } } \} = \bigcup _ { j \in \mathcal { V } } E _ { j } ,\tag{12}
$$

and consequently

$$
\mathbb { P } ( \widehat { S } _ { \mathrm { o u t } } \neq S _ { \mathrm { o u t } } ) \leq \sum _ { j \in \mathcal { V } } \sum _ { s \in S _ { j } } \mathbb { P } ( t r i p l e \ a c t i o n \ d i s a g r e e m e n t \ a t \ ( j , s ) ) .\tag{13}
$$

The second inequality is generally strict, because several witnesses enter one candidate-level OR.

Proof If no $E _ { j }$ occurs, induction from the empty prefix makes states, actions, and termination identical. Otherwise let $j _ { \star }$ be the first occurring event. Both executions have the same prefix and visit the same candidate before $j _ { \star }$ , so $E _ { j , \ast }$ is an actual action disagreement; that candidate is appended in exactly one run and, since selection is append-only with unique labels, the final sets difer.

The identity is deterministic for a shared permutation. When approximate scoring also changes the permutation, its disagreement event is added before applying the theorem conditionally on equal order. Probability enters only through the local action bounds; no independence across comparisons is used anywhere.

## 4.3 Dependency slicing, edge certificates, and what they show

Full-trace equality is often more than an application needs. For a single diversity-selected edge, a backward slice keeps only the sorting and pruning atoms suficient for that edge’s survival.

Corollary 9 (Edge and path certificates) For an exact output edge e selected at scan position $j _ { e }$ , agreement of all candidate actions in the exact prefix through $j _ { e }$ is suficient for e to survive. For a fixed path P composed of such edges,

$$
\mathbb { P } ( s o m e \ d g e \ o f \ P \ i s \ l o s t ) \le \sum _ { e \in P } \sum _ { j \le j _ { e } } \sum _ { s \in S _ { j } } \mathbb { P } ( t r i p l e \ a c t i o n \ d i s a g r e e m e n t \ a t \ ( j , s ) ) .\tag{14}
$$

The condition $M _ { \mathrm { m i n } } ^ { 2 } > 2 \log N _ { P }$ , with $N _ { P }$ the number of listed terms and $M _ { \mathrm { m i n } }$ the smallest standardized margin among them, indicates when the additive bound can stay below one.

In the non-saturated experiments of $\ S 8 . 4 .$ , an edge’s prefix certificate contains 21 to 25 triple comparisons on average, and the plug-in additive bound saturates at one on 89 to 92% of tested edges. The certificate nevertheless carries information in two ways. Its logical half is exact: in every tested case a passing certificate implied edge survival. Its numerical half remains a useful ranking: the plug-in risk sum orders edges by observed failure with Spearman correlation 0.81, 0.69, and 0.73 on Cohere, MiniLM, and GIST.

Remark 10 (Failure concentrates on few decisions) The union bound weights all decisions equally, but query-level failure is driven by a few of them. Ranking the $\it 3 1$ comparisons of a candidate set by individual risk and summing only the r largest improves the correlation with observed query failure up to $r \ : = \ : 3$ to $5$ (Cohere two-bit $0 . 4 7 1  0 . 4 8 0 _ { ; }$ ; MiniLM one-bit $0 . 4 6 0  0 . 5 0 0 ;$ MiniLM rotated two-bit reaches 0.589), after which further terms add noise. The full sum remains the valid upper bound; the top-r score is an empirical predictor, and we keep the two named separately.

## 5 From Representations to Residual Laws

Sections 3 and 4 are quantizer-agnostic: any source of residual control feeds them. This section develops the analytically richest instance, coordinate-preserving one- and two-bit codes on representations with Gaussian coordinate structure, and locates the boundary of that analysis.

## 5.1 The Gaussian model and its empirical support

Assumption 1 (Coordinate Gaussianity) Let $G \in \mathbb { R } ^ { D }$ be the encoder output before $L _ { 2 }$ normalization.

(a) For any fixed k coordinates I, d<sub>BL</sub> $( { \mathcal { L } } ( G _ { I } ) , { \mathcal { L } } ( Z _ { I } ) ) \leq \varepsilon _ { D }$ with $Z \sim { \mathcal { N } } ( \mu , \Sigma )$ .

(b) Thin shell: $\mathbb { E } { \lVert { G } \rVert } - r _ { D } { \lvert { \big / } \mathclose { r } _ { D } \leq \delta _ { D } }$ with $\delta _ { D }  0$

Betser et al. (2026) prove asymptotic Gaussianity of fixed-dimensional projections under alignment and concentration assumptions on the InfoNCE objective. Empirically, all eleven learned representations we inspected, produced by eight diferent encoders, have coordinatewise QQ-plot $R ^ { 2 } \geq 0 . 9 9 5 9$ and norm coeficient of variation at most 0.09; the classical GIST and SIFT descriptors fail both diagnostics. The joint law is a stronger statement than these marginal checks, and §8.7 measures how far each dataset satisfies it. The identities below are exact for the Gaussian model and are used as an oracle; the held-out route of §6 covers representations for which the model is not credible.

## 5.2 The Stein covariance identity

Let $\begin{array} { r } { G \sim \mathcal { N } ( \mu , \Sigma ) , S _ { 0 } = \sum _ { i } \mathrm { s i g n } ( G _ { i } ) , \mathrm { a n d } T _ { 0 } = \sum _ { j } d _ { j } G _ { j } } \end{array}$

Theorem 11 (Stein covariance identity)

$$
\operatorname { C o v } ( S _ { 0 } , T _ { 0 } ) = \sum _ { i , j } a _ { i } \Sigma _ { i j } d _ { j } , \qquad a _ { i } = \frac { 2 \varphi ( \mu _ { i } / \sigma _ { i } ) } { \sigma _ { i } } .\tag{15}
$$

Proof $\mathrm { B y }$ bilinearity $\begin{array} { r } { \operatorname { C o v } ( S _ { 0 } , T _ { 0 } ) = \sum _ { i , j } d _ { j } \operatorname { C o v } ( \operatorname { s i g n } ( G _ { i } ) , G _ { j } ) } \end{array}$ . For jointly Gaussian $( G _ { i } , G _ { j } )$ ,J the conditional mean is $\operatorname { \mathbb { E } } [ G _ { j } \mid G _ { i } ] = \mu _ { j } + ( \Sigma _ { i j } / \sigma _ { i } ^ { 2 } ) ( G _ { i } - \mu _ { i } )$ , whence

$$
\operatorname { C o v } ( \operatorname { s i g n } ( G _ { i } ) , G _ { j } ) = \frac { \sum _ { i j } } { \sigma _ { i } ^ { 2 } } \cdot 2 \sigma _ { i } \varphi ( \mu _ { i } / \sigma _ { i } ) = a _ { i } \Sigma _ { i j } .
$$

The full matrix Σ enters, not only its diagonal, and the of-diagonal contribution is measured by the Frobenius energy $\begin{array} { r } { \mathcal { T } _ { \mathrm { o f f } } = \sum _ { i \neq j } ( a _ { i } \Sigma _ { i j } ) ^ { 2 } } \end{array}$ . Table 3 shows that this term carries 31 to 38% of the predicted ranking signal even though individual coordinate correlations are small (mean $| \rho _ { i j } |$ between 0.04 and 0.11): with $| \rho _ { i j } | \asymp \kappa / \sqrt { D }$ the sum $\begin{array} { r } { \sum _ { i \neq j } \rho _ { i j } ^ { 2 } \asymp \kappa ^ { 2 } D } \end{array}$ is of constant order relative to the diagonal.

Table 3: Spearman ranking fidelity F of the one-bit code, measured and predicted from the Gaussian model with the full covariance and with its diagonal only. Gap explained is $( F _ { \mathrm { f u l l - } } - \mathrm { \Delta } F _ { \mathrm { d i a g } } ) / ( F _ { \mathrm { a c t u a l } } - F _ { \mathrm { d i a g } } )$
<table><tr><td>Dataset</td><td> $F _ { \mathrm { a c t u a l } }$ </td><td> $F _ { \mathrm { f u l l - } \Sigma }$ </td><td> $F _ { \mathrm { d i a g } }$ </td><td>Gap explained</td></tr><tr><td>Cohere</td><td>0.681</td><td>0.688</td><td>0.474</td><td>103%</td></tr><tr><td>Arxiv</td><td>0.897</td><td>0.886</td><td>0.546</td><td>97%</td></tr><tr><td>CodeSearch</td><td>0.823</td><td>0.837</td><td>0.542</td><td>105%</td></tr><tr><td>Random</td><td>0.907</td><td>0.899</td><td>0.560</td><td>98%</td></tr></table>

## 5.3 The magnitude bit under an aligned bilinear model

Let $G , H$ be independent with independent coordinates $G _ { i } , H _ { i } \sim \mathcal { N } ( 0 , \sigma _ { i } ^ { 2 } )$ , and for a deterministic threshold $\tau > 0$ define

$$
T = \sum _ { i } G _ { i } H _ { i } , \qquad S _ { 1 } = \sum _ { i } \mathrm { s i g n } ( G _ { i } ) \mathrm { s i g n } ( H _ { i } ) , \qquad S _ { 2 } = \sum _ { i } q _ { i } ( G ) q _ { i } ( H ) ,\tag{16}
$$

with $q _ { i } ( g ) = \mathrm { s i g n } ( g _ { i } ) ( 1 + { \bf 1 } \{ | g _ { i } | > \tau \} )$

Proposition 12 (Aligned magnitude-bit gain) For every $D \geq 1$ , every $\sigma _ { i } > 0$ , and every finite $\tau > 0$ , the Pearson correlations satisfy $\rho ( S _ { 2 } , T ) > \rho ( S _ { 1 } , T )$ .

The closed forms and the proof are in Appendix E. The alignment between target and scorer is essential: in a linear-target model whose scorer weights do not match the target weights, the same second bit lowers the correlation (by 0.163 in the eight-dimensional example of Appendix E). More code information improves the best readout, not every fixed readout. Empirically the second bit raises global ranking fidelity by +0.071 to +0.132 on the contrastive embeddings we inspected, and on Cohere it brings the 3σ residual survival ratio from 2.09 times Gaussian to 1.05 and lowers pairwise flip rates by 34.5%.

## 5.4 Rotation duality

By L´evy concentration and a union bound, a Haar-random orthogonal rotation equalizes the coordinate variances to

$$
\frac { \mathrm { t r } ( \Sigma ) } { D } \pm O \left( \| \Sigma \| _ { \mathrm { o p } } \sqrt { \frac { \log D } { D } } \right) .
$$

This single fact has two opposite design consequences. It destroys the coordinate variance and sign-entropy structure that a coordinate code exploits, so its efect on a fixed two-bit scorer depends on the representation. It also creates exactly the coordinate uniformity that RaBitQ’s randomized codebook and corrected estimator rely on. The two strategies are dual uses of the same representation. Across 12 datasets the sign-entropy gap $1 - H _ { \mathrm { s i g n } }$ predicts the global two-bit response to rotation with Spearman correlation 0.909, against 0.657 for coordinate-variance heterogeneity alone, and the interaction $( 1 - H _ { \mathrm { s i g n } } ) \times \mathrm { C V } ( \sigma )$ reaches 0.930. Section 8.6 shows what rotation does to local decisions, which is a diferent question with a diferent answer.

## 5.5 Oracle decomposition of the residual

For a fixed ranking or pruning role, let $R _ { a } ^ { \circ }$ be the oracle residual computed with the population magnitude threshold and radius. Its Hoefding decomposition has exact variance

$$
V = \kappa _ { 1 } \sum _ { u } d _ { u } ^ { 2 } + \kappa _ { 2 } \sum _ { p } A _ { p } ^ { 2 } ,\tag{17}
$$

where $d _ { u }$ are node incidences and $A _ { p }$ are unordered-pair edge coeficients (Appendix E). The replacement (Efron–Stein) proxy satisfies $V \leq V _ { \mathrm { E S } } \leq 2 V$ , with equality at two when the first-order Hoefding component vanishes. The actual residual adds a threshold remainder and a shell remainder,

$$
R _ { a } = R _ { a } ^ { \circ } + \Delta _ { \mathrm { t h r } } + \Delta _ { \mathrm { s h e l l } } ,\tag{18}
$$

and their sizes are measurable. On Cohere and MiniLM the threshold remainder is 1.3 to 4.7% of the role variance; on unrotated GIST it is essentially all of it, and it falls to 2.3% after rotation. The remainder is therefore itself a diagnostic of whether the oracle describes a representation.

## 5.6 Necessity: what weak Gaussianity cannot prove

Theorem 13 (Counterexample) There is a sequence of distributions $G _ { D }$ whose fixedk standardized marginals converge to Gaussian in total variation at rate $O _ { k } ( D ^ { - 1 } )$ , whose

relative shell mean-square error is $O ( D ^ { - 1 } )$ , whose covariance is isotropic, and whose average threshold-boundary occupancy vanishes, yet whose fixed-threshold oracle ranking residual satisfies

$$
V _ { D } = \mathrm { V a r } ( R _ { D } ^ { \circ } ) = \Theta ( D ^ { - 1 } ) , \qquad \kappa _ { 4 } ( R _ { D } ^ { \circ } ) = \Omega ( D ^ { 2 } ) .\tag{19}
$$

Consequently a cumulant condition of Bernstein type, or a two-sided sub-gamma bound with variance proxy $v _ { D } = O ( V _ { D } )$ , requires $b _ { D } / \sqrt { V _ { D } } = \Omega ( D ^ { 2 } )$

Proof [Construction] Take $G _ { D } = ( 1 - J _ { D } ) Z _ { D } + J _ { D } D ^ { 3 / 2 } S _ { D }$ with $J _ { D } \sim \mathrm { B e r n o u l l i } ( D ^ { - 4 } )$ , $Z _ { D } \sim \mathcal { N } ( 0 , I _ { D } )$ , and $S _ { D }$ Rademacher. The event that two of the three nodes in a ranking triple are contaminated and one is clean has probability $\Theta ( D ^ { - 8 } )$ ; after oracle normalization its bilinear term contributes $\Theta ( D ^ { 2 } )$ to the fourth moment. The bounded-code component has fourth moment $O ( D ^ { - 2 } )$ and cannot cancel this in $L ^ { 4 }$ , while the total variance stays $\Theta ( D ^ { - 1 } )$ . Appendix H verifies the diagnostic properties and derives the cumulant and MGF consequences. 7

The theorem identifies what must be added to reach exponential concentration. Any route must control one of three things: individual-coordinate influence (bounded codes), conditional cumulants (Doob increments), or the range after truncation.

## 5.7 Three routes to concentration

1. Conditional MGF. If the three Doob increments of $R _ { a } ^ { \circ }$ satisfy deterministic subgamma bounds with parameters $( v _ { j } , b _ { j } )$ , then

$$
\mathbb { P } ( | R _ { a } ^ { \circ } - \mathbb { E } R _ { a } ^ { \circ } | \geq z ) \leq 2 \exp \left[ - \frac { z ^ { 2 } } { 2 ( v _ { \star } + b _ { \star } z ) } \right] , \qquad v _ { \star } = \sum _ { j } v _ { j } , \ b _ { \star } = \operatorname* { m a x } _ { j } b _ { j } .\tag{20}
$$

2. Stable truncation. If a clipped functional $R _ { L }$ agrees with $R _ { a } ^ { \circ }$ outside an event of probability $\varepsilon _ { L } ,$ has bias $| \mathbb { E } R _ { a } ^ { \circ } - \mathbb { E } R _ { L } | \leq d _ { L }$ , and is sub-gamma with $( v _ { L } , b _ { L } )$ , then

$$
\mathbb { P } ( | R _ { a } ^ { \circ } - \mathbb { E } R _ { a } ^ { \circ } | \geq z + d _ { L } ) \leq 2 \exp \left[ - \frac { z ^ { 2 } } { 2 ( v _ { L } + b _ { L } z ) } \right] + \varepsilon _ { L } .\tag{21}
$$

3. Held-out calibration (§6), which needs no analytical tail at all.

Section 8.8 measures the parameters of the first two routes on real embeddings.

## 6 Selective Certificates from Held-Out Blocks

When the analytical route is not available, because Gaussian diagnostics fail, residual tails are heavy, or calibration is degenerate, the decision framework still applies; only the source of the residual tail changes. This section bounds decision risk from held-out data alone. The natural form is a selective guarantee: retain the decisions whose standardized margin exceeds a cutof, route the rest to exact verification, and bound the failure rate on the retained set. This is how a deployed system would use the theory.

Theorem 14 (Direct selective block certificate) Condition on an independent fit split, so that nuisance parameters, selectors, thresholds, and tie rules are fixed. For i.i.d. blocks $Z _ { 1 } , \ldots , Z _ { n }$ let $C _ { i , s }$ be the fraction of decisions in block i accepted by selector s, $U _ { i , s , \tau }$ the accepted fraction that lies in the boundary-or-residual union event at threshold $\tau _ { \mathrm { { ; } } }$ and $F _ { i , s }$ the accepted fraction that actually fails, so that

$$
0 \leq F _ { i , s } \leq U _ { i , s , \tau } \leq C _ { i , s } \leq 1 .\tag{22}
$$

For a prespecified grid G of $( s , \tau )$ pairs set $\epsilon _ { n } = \sqrt { \log ( | \mathcal { G } | / \delta ) / ( 2 n ) }$ . With probability at least $1 - \delta$ , simultaneously for every grid pair with $\widehat { C } _ { s } > 0$

$$
p _ { s } : = \frac { \mathbb { E } F _ { i , s } } { \mathbb { E } C _ { i , s } } \ \leq \ q _ { s , \tau } : = \frac { \mathbb { E } U _ { i , s , \tau } } { \mathbb { E } C _ { i , s } } \ \leq \ \operatorname* { m i n } \left\{ 1 , \frac { \widehat { U } _ { s , \tau } + \epsilon _ { n } } { \widehat { C } _ { s } } \right\} .\tag{23}
$$

The ratios are defined when $\mathbb { E } C _ { i , s } > 0 ,$ zero empirical coverage yields no certificate.

Proof Fix a grid pair and write $q = q _ { s , \tau }$ . Since $0 \leq U _ { i } \leq C _ { i } \leq 1$ , the variable $W _ { i } = U _ { i } - q C _ { i }$ has mean zero and lies in $[ - q , 1 - q ]$ , an interval of length one. One-sided Hoefding gives $\widehat { U } - q \widehat { C } \geq - \epsilon _ { n }$ except with probability $\delta / | { \mathcal { G } } | ;$ ; a union bound over the grid and division by $\widehat C > 0$ finish the proof. Dependence among decisions inside a block is arbitrary. ■

Corollary 15 (Direct accepted-failure certificate) Replacing $U _ { i }$ by $F _ { i }$ and paying only for the selector family S gives, simultaneously over $s \in { \mathcal { S } }$ 2

$$
p _ { s } \leq \operatorname* { m i n } \Biggl \{ 1 , \frac { \widehat { F } _ { s } + \sqrt { \log ( | S | / \delta ) / ( 2 n ) } } { \widehat { C } _ { s } } \Biggr \} .\tag{24}
$$

The two certificates bound the same quantity $p _ { s }$ but through diferent events. The union certificate keeps the boundary-plus-residual mechanism of Theorem 2 visible and is what one reports when the mechanism is the object of interest; the direct certificate targets the observed failure of the frozen rule and is tighter. A conservative alternative that controls numerator and denominator separately costs $\sqrt { \log ( 2 | \mathcal { G } | / \delta ) / ( 2 n ) }$ in both and is pointwise looser; Appendix F.1 also gives a variance-adaptive empirical-Bernstein version obtained by inverting a single-crossing function.

Protocol. We condition on an independent coeficient-fit split, then use prespecified prefixes of 128, 256, 512, and 1,024 i.i.d. certification blocks and 1,024 independent validation blocks; each block is one query or target with 32 sampled candidates. Every method and sample size is fixed in advance, so each reported point carries its own marginal $1 - \delta$ guarantee. Results are in §8.5.

## 7 Quantizer Instantiations

The decision shell is quantizer-agnostic; each family enters through its own residual mechanism.

Coordinate binary codes. The analytical instance of $\ S 5 \colon$ covariance structure, threshold efects, magnitude-bit gain, rotation duality, and the conditional-MGF route are all specific to this family.

RaBitQ. A randomized ratio estimator (Gao and Long, 2024) with a per-rotation error radius. For a fixed query q and data directions $x , y$ sharing one random rotation P,

$$
\begin{array} { r } { \mathbb { P } \{ | \varepsilon _ { R } ( x , q ) - \varepsilon _ { R } ( y , q ) | > \rho _ { x } + \rho _ { y } \} \le 4 \exp ( - c _ { 0 } \epsilon _ { 0 } ^ { 2 } ) , } \end{array}\tag{25}
$$

where $\rho _ { x } , \rho _ { y }$ are the random radii. The shared rotation couples the two edges, so ranking and pruning statements $_ \mathrm { g o }$ through joint events and the triangle inequality rather than independence. For pruning on raw squared distances the distance rule squares to an $\alpha ^ { 2 }$ rule with edge-specific radial factors (Appendix G); a rule stated directly in squared distance or cosine dissimilarity keeps its own parameter.

Lucene BBQ. A role-asymmetric design (Trent, 2024): stored vectors are one-bit while query-role vectors are int4. During HNSW construction Lucene keeps temporary int4 queryrole vectors, scores candidates against existing one-bit vectors, and uses the int4 representations for diversity and reverse-link scoring; the temporary file is removed afterwards. Each scorer role therefore needs its own afine calibration,

$$
\widehat { d } _ { A  B } = b _ { A  B } + a _ { A  B } d + \xi _ { A  B } .\tag{26}
$$

Our reference scorer reproduces the role semantics, not Lucene’s metric-specific corrections.

Block estimators. Product quantization (J´egou et al., 2011; Ge et al., 2014) decomposes the residual over codebook blocks and enters through bounded-block concentration or heldout survival laws; the coordinate-sign Stein identity does not transfer.

Table 4: Cross-quantizer validation on the decision interface, averaged over Cohere, MiniLM, SIFT, and GIST with 60,000 held-out ranking and 60,000 pruning triples per dataset. “Hard” restricts to the 20% smallest exact margins. All calibration slopes are positive. The PQ reference stores 64 index bits per vector.
<table><tr><td></td><td></td><td colspan="2">Rank flip</td><td colspan="2">Prune flip</td></tr><tr><td>Quantizer family</td><td>Edge ρ</td><td>all</td><td>hard</td><td>all</td><td>hard</td></tr><tr><td>Sign (1-bit)</td><td>.373</td><td>35.9%</td><td>47.1%</td><td>20.1%</td><td>36.5%</td></tr><tr><td>Shared 2-bit</td><td>.789</td><td>18.5%</td><td>41.5%</td><td>12.8%</td><td>33.9%</td></tr><tr><td>RaBitQ reference</td><td>.939</td><td>11.8%</td><td>37.2%</td><td>6.5%</td><td>27.1%</td></tr><tr><td>BBQ-like</td><td>.923</td><td>12.8%</td><td>37.3%</td><td>9.0%</td><td>30.7%</td></tr><tr><td>Scalar int4</td><td>.997</td><td>2.1%</td><td>10.1%</td><td>1.2%</td><td>5.8%</td></tr><tr><td>PQ (16 blocks × 4 bits)</td><td>.801</td><td>22.4%</td><td>43.3%</td><td>11.0%</td><td>33.1%</td></tr></table>

Table 4 makes one point that no bit budget removes. Scalar int4 has the best interface metrics of the inspected references, with edge correlation 0.997 and 2.1% ranking flips, yet on the hardest fifth of margins it still flips 10.1% of ranking decisions. A larger budget changes the residual law and shrinks the boundary crossings; it does not move the boundary.

## 8 Experiments

The experiments test each link of the argument: boundary mass, residual covariance and its origin, trace composition, held-out certification, the analytical regime and its edge, and the decoupling of fidelity from decision risk. All protocols use disjoint data splits, and every experiment is a numbered, non-interactive Python module with fixed seeds.

## 8.1 Datasets and protocols

Table 5: Representation regimes. “Learned” means contrastive or self-supervised training; “classical” means hand-crafted or dimensionality-reduced features.
<table><tr><td>Group</td><td>Datasets</td><td>D</td><td>Role</td></tr><tr><td>Contrastive text</td><td>Cohere, MiniLM, BGE-M3, Jina, MSMARCO</td><td>384-1024</td><td>primary regimes</td></tr><tr><td>Vision</td><td>Wolt-CLIP, Landmark-DINO</td><td>512-768</td><td>modality transfer</td></tr><tr><td>Classical</td><td>SIFT, GIST, GloVe</td><td>100-960</td><td>negative controls</td></tr><tr><td>Synthetic</td><td>Gaussian, sphere, random, contaminated</td><td>64-2048</td><td>mechanism tests</td></tr></table>

Each dataset contributes a fixed random sample of vectors (seed 42), L<sub>2</sub>-normalized. Local decision experiments use fixed 32-neighbour candidate sets with 80 calibration and 160 held-out queries per dataset; held-out certificate experiments use the block protocol of §6; the cross-quantizer study uses 1,200 fit and 2,000 held-out vectors per dataset. Pruning uses the standard non-saturated RobustPrune rule with $\alpha = 1 . 2$ and $M = 3 2$ unless stated otherwise.

## 8.2 Standardized margins predict local failures

On Cohere two-bit decisions the flip rate falls monotonically with the standardized margin $M = c _ { Q } \Gamma / v$ over more than two orders of magnitude (Table 6), and no flips are observed above $M = 5$

Table 6: Flip rate by standardized-margin bin on Cohere-768 two-bit codes, $n = 4 { , } 9 6 0$ heldout decisions.
<table><tr><td>Margin bin</td><td> $M < 0 . 2 5$ </td><td>0.25-1</td><td>1-2</td><td>2-3</td><td>3-5</td><td> $M > 5$ </td></tr><tr><td>Flip rate</td><td>48.6%</td><td>25.7%</td><td>9.50%</td><td>2.54%</td><td>0.14%</td><td>0.00%</td></tr><tr><td>Decisions</td><td>70</td><td>381</td><td>1210</td><td>1577</td><td>1413</td><td>309</td></tr></table>

Across 24 held-out dataset–quantizer configurations (12 datasets, two-bit codes with and without rotation), the calibrated boundary-plus-tail predictor tracks unseen ranking flip rates with Spearman correlation 0.970 and unseen pruning flip rates with 0.992. Using each configuration’s global distance Spearman correlation as the predictor instead gives 0.739 for ranking and 0.053 for pruning; using its distance mean squared error gives 0.184 and 0.447. Fidelity is a weak predictor of ranking risk and no predictor of pruning risk; the standardized margin predicts both.

## 8.3 Where the shared-query covariance comes from

Table 2 reports correlations far above the 1/2 bound that holds for i.i.d. endpoints and a symmetric kernel. To locate the source we keep one globally fitted afine edge scorer fixed and vary only how pairs are generated (Table 7).

Table 7: Pooled shared-query residual correlation under diferent pair-generation regimes with one fixed edge scorer. “Anchor” pairs the exact top candidate with each remaining top-32 candidate.
<table><tr><td>Dataset</td><td>Scorer</td><td>i.i.d.</td><td>Random cand.</td><td>Top-256</td><td>Top-32 / anchor</td></tr><tr><td>Cohere</td><td>1-bit</td><td>.361</td><td>.312</td><td>.524</td><td>.525 / .544</td></tr><tr><td>Cohere</td><td>2-bit</td><td>.391</td><td>.346</td><td>.571</td><td>.599 / .613</td></tr><tr><td>Cohere</td><td>rotated 2-bit</td><td>.259</td><td>.212</td><td>.324</td><td>.340 / .364</td></tr><tr><td>MiniLM</td><td>2-bit</td><td>.010</td><td>.020</td><td>.067</td><td>.111 / .118</td></tr><tr><td>GIST</td><td>1-bit</td><td>.386</td><td>.370</td><td>.925</td><td>.934 / .870</td></tr><tr><td>GIST</td><td>2-bit</td><td>.405</td><td>.422</td><td>.576</td><td>.640 / .616</td></tr></table>

Local selection, not identity collisions, drives the increase: an i.i.d. control with replacement and a distinct-triple control difer by at most 0.003, while moving from random candidates to the top-32 roughly doubles the correlation on Cohere and more than doubles it on GIST. The exact weighted within/between identity of Appendix B holds to floatingpoint precision in every regime, and most pooled covariance in this experiment comes from variation in query-specific residual means, while weighted within-query covariance is small on average. This describes the selected-pair population with a random query; conditioning on a fixed query removes the between-query term. Moreover, this experiment uses a globa afine edge calibration, whereas Table 2 uses ranking-margin calibration. It therefore identifies query-level heterogeneity under this protocol without quantitatively attributing the earlier independence ratios.

## 8.4 Trace coupling and its consequences

All 960 sampled selection calls (three datasets, two quantizers, 160 targets each) satisfy the identity of Theorem 8: the approximate output difers from the exact output exactly when some candidate-level action difers, and never otherwise. Table 8 shows what the identity implies for output agreement.

Output disagreement is large even where local disagreement is modest, because each candidate action is an OR over its whole selected prefix, and the first action diference persists in an append-only output. Every one of the sampled triple disagreements is a false keep; no false prune occurs at $\alpha = 1 . 2$ When a saturating refill step is added after the standard call, it contributes 34.1% of the final edges on Cohere, none on MiniLM, and 28.8% on GIST; these shares belong to the refill variant, not to RobustPrune itself, and the local pruning analysis is unafected by them.

Table 8: Standard non-saturated RobustPrune replay with exact candidate pools, two-bit codes, $\alpha = 1 . 2 ,$ $M = 3 2$ . Fixed-order replay reuses the exact permutation; free replay also re-sorts candidates by quantized distance.
<table><tr><td>Dataset</td><td>Local prune flip</td><td>Fixed-order Jaccard</td><td>Free replay Jaccard</td></tr><tr><td>Cohere-768</td><td>13.06%</td><td>0.322</td><td>0.395</td></tr><tr><td>MiniLM-384</td><td>1.92%</td><td>0.654</td><td>0.529</td></tr><tr><td>GIST-960</td><td>7.91%</td><td>0.458</td><td>0.260</td></tr></table>

## 8.5 Held-out selective certificates

Table 9 reports the certificates of §6 at 512 and 1,024 blocks, averaged over 24 dataset– quantizer configurations per role. Every method issued a certificate on every configuration, and on every one the empirical risk of the selected policy on 1,024 independent validation blocks was below its certificate.

Table 9: Selective block certificates $\left( \delta = 0 . 0 5 \right)$ , averaged over 24 configurations per role. Coverage is the fraction of validation decisions retained by the selected policy and risk is their observed flip rate; each method optimizes its own policy, so coverages difer. Ranking / pruning.
<table><tr><td>Blocks</td><td>Method</td><td>Certificate</td><td>Validation coverage</td><td>Validation risk</td></tr><tr><td>512</td><td>two-event Hoeffding</td><td>20.59% 22.95%</td><td>67.0% 67.6%</td><td>0.42% 0.86%</td></tr><tr><td>512</td><td>direct union</td><td>16.74% 18.65%</td><td>64.6% 63.5%</td><td>0.33% 0.75%</td></tr><tr><td>512</td><td>empirical-Bernstein union</td><td>9.49% 10.91%</td><td>61.2% 56.8%</td><td>0.26% 0.72%</td></tr><tr><td>512</td><td>direct failure</td><td>9.07% 10.45%</td><td>87.8% 80.2%</td><td>1.34%/ 2.01%</td></tr><tr><td>1,024</td><td>two-event Hoeffding</td><td>15.13% 17.36%</td><td>63.7% 61.0%</td><td>0.32% /0.75%</td></tr><tr><td>1,024</td><td>direct union</td><td>12.74% 14.65%</td><td>61.2% 61.0%</td><td>0.26%/ 0.75%</td></tr><tr><td>1,024</td><td>empirical-Bernstein union</td><td>6.13% / 7.27%</td><td>57.7% / 51.1%</td><td>0.22% / 0.44%</td></tr><tr><td>1,024</td><td>direct failure</td><td>6.75% / 7.99%</td><td>81.9% / 75.2%</td><td>0.94% / 1.44%</td></tr></table>

For a fixed selector at the same confidence level, the direct ratio removes the denominator slack of the two-event bound without changing the accepted set. The policies in Table 9 are optimized separately by method, so their retained fractions difer. The empirical-Bernstein union certificate is tighter with lower coverage, while the direct-failure certificate retains about four fifths of decisions because it does not carry the structural gap between the union event and actual failure. Certificate values must therefore be read together with retained coverage.

A cutof targeting the highest-margin fifth is frozen on calibration data and then applied to a separate validation pool. Across the 24 configurations, the mean validation retention is 19.9% for ranking and 20.9% for pruning; mean empirical flip rates fall from 8.58% to 5.27% and from 3.84% to 0.12%, respectively. This is a held-out empirical reduction, while the population guarantee is provided separately by the block certificates above.

## 8.6 Rotation: global fidelity does not determine decision risk

A Haar-random rotation is the cleanest intervention on a coordinate code: it changes the representation’s coordinate structure and nothing else. On Cohere it raises the Spearman correlation between quantized and exact distances from 0.576 to 0.926 and lowers the distance mean squared error by a factor of four. What happens to local decisions depends on which decisions are asked.

Table 10: Rotation on Cohere codes under two decision protocols. The distance Spearman correlation is measured on the calibration pairs of the block protocol.
<table><tr><td>Protocol</td><td>Quantity</td><td>Unrotated</td><td>Rotated</td></tr><tr><td rowspan="4">32-neighbour anchor pairs</td><td>one-bit flip rate</td><td>8.89%</td><td>15.58%</td></tr><tr><td>two-bit flip rate</td><td>5.83%</td><td>8.81%</td></tr><tr><td>shared-query residual correlation</td><td>0.870</td><td>0.435</td></tr><tr><td>difference residual s.d. (cosine units)</td><td>0.0134</td><td>0.0148</td></tr><tr><td rowspan="3">48-candidate held-out blocks</td><td>distance Spearman</td><td>0.576</td><td>0.926</td></tr><tr><td>ranking flip rate</td><td>4.43%</td><td>4.48%</td></tr><tr><td>pruning flip rate</td><td>10.25%</td><td>6.93%</td></tr></table>

On the anchor protocol, which compares the exact nearest neighbour against each of its 31 competitors, rotation raises the two-bit flip rate by half and nearly doubles the onebit rate. The mechanism is visible in the residual structure. Rotation shrinks each edge’s error variance by a factor of three to four, which is why global fidelity improves, but it also cuts the shared-query correlation from 0.870 to 0.435. At fixed marginal variances a smaller positive correlation raises the variance of a diference, and here the two efects net out to an 11% larger standard deviation for the comparison residual together with a more negative bias relative to the margin (from −0.26 to −0.36 standard deviations). On the block protocol, whose candidate sets are larger and whose pairs are not anchored on the top candidate, the ranking flip rate is unchanged and the pruning flip rate falls. A fidelity statistic that moves by +0.35 cannot predict an efect whose sign depends on the decision population; quantizer choices for graph search must be evaluated on the decisions the graph actually makes.

## 8.7 The analytical regime and its edge

Coordinate-wise diagnostics are broadly compatible with Gaussian marginals in the inspected learned representations. The joint law is where datasets separate. Testing k = 8 coordinate subsets with a Kolmogorov–Smirnov test on the Mahalanobis radius at the 1% level (24 random subsets per dataset), BGE-M3 and GloVe reject none, Landmark-DINO rejects 8%, MiniLM 13%, Cohere 29%, and Wolt-CLIP 58%; SIFT and GIST reject every subset. The exact Gaussian oracle is therefore partially supported on the primary dataset, which is the reason the analytical route is paired with the model-free one rather than ofered alone.

Where the oracle applies it is accurate. The incidence-covariance identity (17) reproduces the measured ranking and pruning residual variances within 4.6% across twelve role configurations, with median error 1.2%. The variance route is also tight in a way the worstcase constant does not reveal. Over 28 dataset–coordinate–role configurations, a componentwise Cauchy–Schwarz proxy exceeds the measured variance by a median factor of 40.8, while the replacement proxy exceeds it by a median factor of 1.67 (range 1.21 to 2.04), consistent with the analytical factor of at most two. The reason the combined kernel behaves so much better than the sum of its parts is that in all 28 configurations the bounded-code and bilinear components have negative covariance, with a median cancellation ratio of 0.947.

The GIST descriptors show the same machinery diagnosing and repairing a failure. Unrotated GIST has calibration slope near zero, zero margin correlation, residual kurtosis 26, a left tail 7.7 times Gaussian, and a threshold remainder equal to the whole role variance; every one of these is observable without a search run. A random rotation raises the margin correlation to 0.510 and 0.669 for one- and two-bit codes, brings the kurtosis to 3.5 and 2.8, the left tail to 1.88 and 0.76 times Gaussian, and the threshold remainder to 2.3%, and the flip rates fall from 100% to 14.80% (one-bit) and from 35.83% to 9.33% (two-bit).

## 8.8 Conditional-MGF and truncation parameters

Across 24 role configurations (six datasets, two quantizers, two roles) the untruncated increment parameter b /sd has minimum 0.133, median 0.158, and maximum 0.324; the Doob variance sum exceeds the total variance by 5 to 10%; and the calibration-to-validation variance transfer ratio lies between 0.945 and 1.075. Clipping at the 1% exceptional-mass cutof brings b<sub>⋆</sub>/sd to a median of 0.147 and a maximum of 0.258. Every configuration admits a clipped-functional bound below one, and no held-out tail exceeded its reported bound. These are the measured inputs to the two analytical routes of §5; a uniform conditional-MGF assumption over the population is a hypothesis these measurements are consistent with, not one they establish.

## 8.9 Necessity and falsifiability

Two constructions show that the framework’s ingredients are necessary. At exact margin zero, residuals of arbitrarily small norm select opposite decisions, so the boundary-mass term cannot be dropped. And two perturbations of the same exact scores, one light-tailed and one coupled to the decision boundary, have Spearman correlation 0.9860282 with the exact scores to seven digits, yet flip 5.09% against 10.00% of all decisions and 23.91% against 50.00% of the hardest fifth. Mean squared error does not rescue the comparison: it is lower for the boundary-coupled perturbation (0.0170 against 0.0251), so a practitioner selecting by either global metric would pick the quantizer with twice the failure rate. Standardized margins separate the two immediately.

The framework also abstains where it should: on nonpositive calibration, heavy tails, failed Gaussian diagnostics, and saturated certificates it reports an uncertified decision rather than a number.

## 9 Discussion: Practical Guidance

A per-decision reliability score. Given an embedding and a quantizer, fit the afine calibration and residual scale on independent blocks and evaluate $M = c _ { Q } \Gamma / v$ wherever the exact margin is available. In Table 6, $M > 5$ has no observed flips in 309 decisions and $M \ : < 1$ flips more than a quarter of the time. Because Γ is the exact margin, this is an ofline audit; an online router additionally needs an observable lower bound on the margin or a residual envelope.

Select quantizers on decisions, not on fidelity. Section 8.6 shows a rotation that raises global fidelity by +0.35 while raising one local flip rate by half and leaving another unchanged, and §8.9 shows two quantizers with identical rank fidelity, opposite ordering by mean squared error, and a factor of two in failure rate. Pilot the candidate quantizers on the decision population of the intended index, measure standardized margins and flip rates there, and choose on those.

Two diagnostics, not one. Sign entropy is the natural single routing statistic and it is insuficient. Cohere $( H _ { \mathrm { s i g n } } = 0 . 7 4 7 )$ and SIFT (0.746) are indistinguishable by it, and their global rotation responses are nearly identical (+0.170 and +0.200), yet coordinate binary quantization is usable on Cohere and useless on SIFT. The entropy gap $1 - H _ { \mathrm { s i g n } }$ predicts how much rotation changes a code $( \rho = 0 . 9 0 9$ over 12 datasets). Whether the unrotated code is usable at all is answered by the eligibility gate (calibration slope, margin correlation, residual tail), which passes Cohere and rejects SIFT. Run the gate first; if it passes, rotation is an optimization to be evaluated on decision metrics, and if it fails, rotation is a repair whose size the entropy gap predicts.

Two levels of assurance. Union-event certificates keep the boundary-plus-residual mechanism visible; direct-failure certificates bound the frozen rule’s observed failure and are tighter at higher coverage. Both are population statements for the block distribution they were calibrated on, and neither transfers across a distribution shift without recalibration.

Open directions. Whether the trace-length saturation boundary can be characterized from representation geometry alone; whether adaptive candidate sets (beam search) admit a martingale extension of the coupling argument; and whether a pilot selector choosing rotation, bit width, and verification cutof from covariance statistics alone recovers exactscore topology.

## 10 Conclusion

Low-bit vector search cannot be understood through a single fidelity number. The decisions that flip are concentrated at the boundary, and their noise depends on shared structure that global metrics erase. A distribution-free boundary–residual decomposition localizes the risk, covariance-aware role analysis captures the shared structure, and a deterministic frozen-trace coupling connects local decisions to the neighbour lists a graph algorithm produces. Under an exact Gaussian model of contrastive representations the residual covariance is explicit and a magnitude bit provably helps an aligned scorer; a rare-contamination construction shows exactly why low-order Gaussian diagnostics cannot by themselves deliver exponential tails; and held-out block certificates bound selective risk for any quantizer whose analytical description is out of reach.

Limitations. The trace certificates saturate on long paths and say nothing about candidate generation or graph navigability. The Gaussian identities are oracle results, and approximate Gaussian diagnostics do not come with a transfer theorem. The held-out certificates require representative independent blocks, use exact margins in the selector, and do not transfer under distribution shift. End-to-end recall additionally depends on candidate coverage, which is outside this analysis.

## Reproducibility

All experiments are implemented as numbered, non-interactive Python modules with fixed seeds, disjoint data pools, and machine-readable JSON or NPZ outputs. Negative and vacuous results are preserved in the result files. Appendix I lists the protocol of each experiment.

## Acknowledgments and Disclosure of Funding

No external funding was received for this work.

## Appendix A. The Decision Shell: Proofs and Rates

## A.1 Conservative ties and event inclusion

Write $\widehat { \Psi } _ { Q } = c _ { Q } \Psi + R _ { \Psi , Q }$ with $c _ { Q } > 0$ . If $\Psi > 0$ and the conservative failure event occurs, then $\widehat \Psi _ { Q } \leq 0$ and

$$
R _ { \Psi , Q } \leq - c _ { Q } \Psi = - c _ { Q } \Gamma _ { \Psi } .\tag{27}
$$

If $\Psi < 0$ , failure gives $R \Psi , Q \geq c _ { Q } \Gamma \Psi$ . Hence

$$
\mathcal { E } _ { \Psi } \subseteq \{ \Gamma _ { \Psi } > 0 , | R _ { \Psi , Q } | \geq c _ { Q } \Gamma _ { \Psi } \} ,\tag{28}
$$

and splitting the right-hand side according to $0 < \Gamma _ { \Psi } \leq \tau$ or $\Gamma _ { \Psi } > \tau$ proves Theorem 2. If only strict sign reversal counts as failure, the residual event may use a strict inequality. If exact ties are part of the target risk, one adds $\mathbb { P } ( \Psi = 0 )$ together with the exact and approximate tie actions.

## A.2 Explicit small-ball rate

Under Corollary 3 write $r = v / c _ { Q }$ and $A = C r ^ { \beta }$ . When $0 < A < 2 \quad$ , let $L = \log ( 2 / A )$ and $\tau _ { r } = r \sqrt { 2 L }$ . If $\tau _ { r } \leq \tau _ { 0 }$ , substitution into (3) gives

$$
\begin{array} { r } { \mathbb { P } ( \mathcal { E } _ { \Psi } ) \le C r ^ { \beta } ( 2 L ) ^ { \beta / 2 } + C r ^ { \beta } , } \end{array}\tag{29}
$$

so the worst-case consequence of marginal small-ball and residual-tail assumptions is

$$
O \Big ( r ^ { \beta } [ \log ( 1 / r ) ] ^ { \beta / 2 } \Big ) .\tag{30}
$$

The logarithm is a feature of the worst case over the assumed class, not a lower bound for each fixed distribution: a margin-conditional residual tail removes it and yields $O ( r ^ { \beta } )$ by direct integration.

## A.3 Necessity at the boundary

$\mathrm { A t } \ \Psi = 0$ , the approximations $\widehat \Psi _ { + } = \epsilon$ and $\widehat \Psi _ { - } = - \epsilon$ have residual magnitudes tending to zero with ϵ yet select opposite actions. Every orientation theorem therefore needs a positive margin, a boundary-mass term, or an explicit tie policy.

## Appendix B. Covariance-Aware Ranking

## B.1 Diference residual and bias

Let $\boldsymbol { X } = ( \xi _ { x } , \xi _ { y } ) ^ { \top } , m = \mathbb { E } \boldsymbol { X }$ , and $\boldsymbol { u } = ( - 1 , 1 ) ^ { \top }$ . The joint MGF proxy gives

$$
\begin{array} { r } { \mathbb { E } \exp \{ \lambda u ^ { \top } ( X - m ) \} \le \exp \{ \lambda ^ { 2 } u ^ { \top } K _ { Q } u / 2 \} , } \end{array}\tag{31}
$$

so $R _ { \mathrm { r a n k } } - \mu _ { R }$ is sub-Gaussian with

$$
\mu _ { R } = u ^ { \top } m , \qquad \nu _ { \mathrm { r a n k } } ^ { 2 } = u ^ { \top } K _ { Q } u = K _ { x x } + K _ { y y } - 2 K _ { x y } .\tag{32}
$$

For a fixed positive exact margin $\gamma$ the failure event is $R _ { \mathrm { r a n k } } ~ \leq ~ - c _ { Q } \gamma$ , and Chernof’s method gives Proposition 5. For the negative orientation the relevant right tail has efective margin $c _ { Q } \gamma - \mu _ { R }$ . If $\nu _ { \mathrm { r a n k } } = 0$ , Jensen’s inequality and the MGF bound give $R _ { \mathrm { r a n k } } = \mu _ { R }$ almost surely and the decision is deterministic.

## B.2 Shared-endpoint covariance for i.i.d. and selected populations

Let $X , Y , Z$ be i.i.d. and $h \in L ^ { 2 } ( P \otimes P )$ symmetric, with Hoefding decomposition

$$
h ( x , y ) - \mu = h _ { 1 } ( x ) + h _ { 1 } ( y ) + h _ { 2 } ( x , y ) ,\tag{33}
$$

where $h _ { 2 }$ is degenerate in each argument. Put $a = \operatorname { V a r } ( h _ { 1 } ( X ) )$ and $b = \operatorname { V a r } ( h _ { 2 } ( X , Y ) )$ . Orthogonality gives

$$
\mathrm { V a r } ( h ( X , Y ) ) = 2 a + b , \qquad \mathrm { C o v } ( h ( X , Y ) , h ( X , Z ) ) = a ,\tag{34}
$$

so that, when the marginal variance is positive,

$$
0 \leq \operatorname { C o r r } ( h ( X , Y ) , h ( X , Z ) ) = { \frac { a } { 2 a + b } } \leq { \frac { 1 } { 2 } } , \qquad 1 \leq { \frac { \operatorname { V a r } ( R _ { 1 } ) + \operatorname { V a r } ( R _ { 2 } ) } { \operatorname { V a r } ( R _ { 1 } - R _ { 2 } ) } } = { \frac { 2 a + b } { a + b } } \leq 2 .\tag{35}
$$

The lower bounds are attained at $a = 0 , b > 0$ and the upper bounds at $b = 0 , a > 0 ;$ if $a = b = 0$ both ratios are undefined.

For query-conditioned roles let $m _ { a } ( Q ) = \mathbb { E } [ R _ { a } \mid Q ] , v _ { a } ( Q ) = \operatorname { V a r } ( R _ { a } \mid Q )$ , and $c ( Q ) =$ $\operatorname { C o v } ( R _ { L } , R _ { R } \mid Q )$ . The laws of total covariance and total variance give

$$
\operatorname { C o v } ( R _ { L } , R _ { R } ) = \operatorname { \mathbb { E } } c ( Q ) + \operatorname { C o v } ( m _ { L } ( Q ) , m _ { R } ( Q ) ) ,\tag{36}
$$

$$
\operatorname { V a r } ( R _ { L } - R _ { R } ) = \mathbb { E } [ v _ { L } ( Q ) + v _ { R } ( Q ) - 2 c ( Q ) ] + \operatorname { V a r } ( m _ { L } ( Q ) - m _ { R } ( Q ) ) .\tag{37}
$$

When candidates are conditionally independent draws from a frozen selection kernel $K _ { Q }$ $c ( Q ) = 0$ , and the pooled correlation is driven entirely by the second term; it can approach one when query-specific means dominate the conditional variance. Selection changes the candidate law, hence conditional means and variances; it changes the query mixture weights only if query sampling, retention, or row weighting also changes. For uniform sampling without replacement from a pool of size M,

$$
\mathrm { C o v } ( f _ { I } , g _ { J } \mid Q ) = - \frac { c _ { f g } ( Q ) } { M - 1 } .\tag{38}
$$

A fixed anchor has zero conditional covariance because its residual is constant given the frozen state, so its within-query correlation is undefined. These identities describe second moments; the MGF proxy K is a separate input.

## B.3 Exact empirical within/between identity

For query group g with $n _ { g }$ observed pairs $( \ell _ { g i } , r _ { g i } )$ , let $N = \textstyle \sum _ { g } n _ { g }$ , let $\ell _ { g } , \bar { r } _ { g }$ be group means, and ${ \bar { \ell } } , { \bar { r } }$ row-weighted grand means. Expanding each centered product gives the deterministic identity

$$
\sum _ { g , i } ( \ell _ { g i } - \bar { \ell } ) ( r _ { g i } - \bar { r } ) = \sum _ { g , i } ( \ell _ { g i } - \bar { \ell } _ { g } ) ( r _ { g i } - \bar { r } _ { g } ) + \sum _ { g } n _ { g } ( \bar { \ell } _ { g } - \bar { \ell } ) ( \bar { r } _ { g } - \bar { r } ) ,\tag{39}
$$

so that, with the pooled N − 1 denominator,

$$
s _ { \mathrm { p o o l } } = \sum _ { g : n _ { g } \geq 2 } \frac { n _ { g } - 1 } { N - 1 } s _ { g } + \frac { 1 } { N - 1 } \sum _ { g } n _ { g } ( \bar { \ell } _ { g } - \bar { \ell } ) ( \bar { r } _ { g } - \bar { r } ) .\tag{40}
$$

The same matrix identity decomposes both marginal variances and therefore Var $( R _ { L } \mathrm { ~ - ~ }$ $R _ { R } )$ . It holds for any row dependence; reading its two pieces as unbiased estimators of the population terms in (37) requires a sampling model.

## B.4 Fixed top-K

Let C be fixed before the approximation randomness, let A be the exact top-K subset, and suppose no exact tie crosses the boundary. Exact top-K preservation follows if every $x \in A$ stays ahead of every $y \in C \setminus A$ , so

$$
\mathbb { P } _ { Q } ( \widehat { A } \neq A ) \leq \sum _ { x \in A } \sum _ { y \notin A } \mathbb { P } _ { Q } \{ \widehat { d } _ { Q } ( q , y ) \leq \widehat { d } _ { Q } ( q , x ) \} ,\tag{41}
$$

and any deterministic suficient comparison set may replace the full cross product.

## Appendix C. Standard Vamana Pruning

The RobustPrune rule declares candidate c dominated by selected neighbour s when $\alpha d ( s , c ) \leq$ $d ( t , c )$ with $\alpha \geq 1$ . Define

$$
\Psi _ { \alpha } = d ( t , c ) - \alpha d ( s , c ) , \qquad \widehat { \Psi } _ { \alpha , Q } = c _ { Q } \Psi _ { \alpha } + Z _ { \alpha , Q } , \qquad Z _ { \alpha , Q } = \xi _ { t c } - \alpha \xi _ { s c } ,\tag{42}
$$

with action $D = \mathbf { 1 } \{ \Psi _ { \alpha } \geq 0 \}$ . For nonzero margins the false-keep and false-prune events are

$$
\{ \Psi _ { \alpha } > 0 , \ \widehat { \Psi } _ { \alpha , Q } < 0 \} , \qquad \{ \Psi _ { \alpha } < 0 , \ \widehat { \Psi } _ { \alpha , Q } \geq 0 \} ,\tag{43}
$$

and at an exact structural tie disagreement is $\{ \Psi _ { \alpha } = 0 , \ Z _ { \alpha , Q } < 0 \}$ . A no-atom property of $Z _ { \alpha , Q }$ rules out exact approximate ties but says nothing about this directional probability, which must be carried as its own term.

With $w = ( 1 , - \alpha ) ^ { \top }$ a joint MGF proxy gives

$$
\nu _ { \alpha , Q } ^ { 2 } = w ^ { \top } K _ { Q } ^ { ( 3 ) } w = K _ { t c , t c } + \alpha ^ { 2 } K _ { s c , s c } - 2 \alpha K _ { t c , s c } .\tag{44}
$$

Writing $\mu _ { Z } = \mathbb { E } Z _ { \alpha , Q }$ , the two orientations have efective one-sided margins $c _ { Q } \gamma + \mu _ { Z }$ and $c _ { Q } \gamma - \mu _ { Z }$ . A through-origin calibration does not make $\mu _ { Z }$ vanish, and a shared edge intercept b enters $Z _ { \alpha , Q }$ as $( 1 - \alpha ) b$ . If both edge residual magnitudes are deterministically at most $\epsilon ,$ then $| Z _ { \alpha , Q } | \leq ( 1 + \alpha ) \epsilon$

## Appendix D. Frozen Semantic Trace Coupling

## D.1 Candidate-level state machine

Fix unique candidate labels in an exact-distance-sorted permutation $\pi = ( c _ { ( 1 ) } , \ldots , c _ { ( N ) } )$ with deterministic tie-breaking, a target t, a degree budget M, and $\alpha \geq 1$ . Starting from $S _ { 1 } = ( )$ , define at each visited position

$$
D _ { i } = \bigvee _ { s \in S _ { i } } \mathbf { 1 } \{ d ( t , c _ { ( i ) } ) - \alpha d ( s , c _ { ( i ) } ) \geq 0 \} .\tag{45}
$$

Append $c _ { ( i ) }$ exactly when $D _ { i } = 0$ and stop when the permutation is exhausted or the degree cap is reached. The call is append-only and has no refill; for a fixed sorted permutation it is equivalent to repeatedly selecting the nearest remaining candidate and deleting the candidates dominated by a selected point.

For each exact visited state evaluate the approximate counterfactual action $\widehat { D } _ { i } ^ { * }$ on the exact prefix $S _ { i }$ and write $E _ { i } = \{ \widehat { D } _ { i } ^ { * } \neq D _ { i } \}$ . A first-divergence induction gives the deterministic identity

$$
\{ \widehat { T } _ { \mathrm { s e m } } \neq T _ { \mathrm { s e m } } \} = \{ \widehat { S } _ { \mathrm { o u t } } \neq S _ { \mathrm { o u t } } \} = \bigcup _ { i \in \mathcal { V } } E _ { i } .\tag{46}
$$

The implication from output diference back to some $E _ { i }$ uses unique labels, append-only selection, and the absence of refill: the first action-disagreement candidate belongs to exactly one final set.

## D.2 Witness and conservative-event inclusions

For $s \in S _ { i }$ let

$$
\begin{array} { r } { A _ { i , s } = \big \{ \mathbf { 1 } \{ \Psi _ { i , s } \geq 0 \} \neq \mathbf { 1 } \{ \widehat { \Psi } _ { i , s } \geq 0 \} \big \} . } \end{array}\tag{47}
$$

Then $E _ { i } \subseteq \cup _ { s \in S _ { i } } A _ { i , s }$ , and the inclusion can be strict because another witness may preserve the OR. If ${ \widehat { \Psi } } = \gamma \Psi + R$ with $\gamma > 0$ , then for nonzero margins

$$
A \cap \{ \Psi \neq 0 \} \subseteq \{ \Psi \neq 0 , \ \Psi \widehat { \Psi } \leq 0 \} \subseteq \{ \Psi \neq 0 , \ | R | \geq \gamma | \Psi | \} ,\tag{48}
$$

where the first inclusion can be strict when $\Psi > 0$ and $\widehat \Psi = 0$ . For exact ties, actual disagreement is $R < 0$ and is added separately. Consequently, after fixing the data and the exact trace,

$$
\begin{array} { r l r } {  { \mathbb { P } ( \widehat { S } _ { \mathrm { o u t } } \neq S _ { \mathrm { o u t } } ) \le \sum _ { i \in \mathcal { V } } \sum _ { s \in S _ { i } } \big [ { \mathbf 1 } \{ { \Psi } _ { i , s } > 0 \} { \mathbb { P } } ( - R _ { i , s } \ge \gamma \Psi _ { i , s } ) } } \\ & { } & { + { \mathbf 1 } \{ { \Psi } _ { i , s } < 0 \} { \mathbb { P } } ( R _ { i , s } \ge \gamma | \Psi _ { i , s } | ) } \\ & { } & { + { \mathbf 1 } \{ { \Psi } _ { i , s } = 0 \} { \mathbb { P } } ( R _ { i , s } < 0 ) \big ] , } \end{array}\tag{49}
$$

with no independence between comparisons.

## D.3 Edge, path, sorting, and saturation

If the exact output edge $( t , c )$ is selected at position $j _ { c } ,$ agreement of all candidate actions in the exact prefix through $j _ { c }$ is suficient for its survival. Source-scoped prefix certificates for a fixed path may be union-bounded without cross-source independence; the resulting statement is about retention of the listed edges.

If approximate scoring changes the permutation, the ordering event is added:

$$
\mathbb { P } ( \mathrm { o u t p u t ~ d i v e r g e n c e } ) \le \mathbb { P } ( \widehat { \pi } \ne \pi ) + \sum _ { i \in \mathcal { V } } \mathbb { P } ( E _ { i } ) .\tag{50}
$$

A deterministic saturation map that appends unselected candidates after RobustPrune satisfies

$$
\{ { \mathrm { s a t u r a t e d - o u t p u t ~ d i v e r g e n c e } } \} \subseteq \{ { \mathrm { s t a n d a r d - t r a c e ~ d i v e r g e n c e } } \} ,\tag{51}
$$

and the reverse inclusion fails in general, because distinct diversity traces can saturate to the same list. Saturation is therefore analysed as a post-processing map applied to the standard call.

## Appendix E. Gaussian Residual Transfer

## E.1 Oracle decomposition

Let $G _ { u } \overset { \mathrm { i . i . d . } } { \sim } \mathcal { N } ( \mu _ { D } , \Sigma _ { D } )$ and $X _ { u } = G _ { u } / \lVert G _ { u } \rVert$ . Define

$$
T ( g ) = D ^ { - 1 } \sum _ { i } | g _ { i } | , \qquad q _ { i } ( g ) = \mathrm { s i g n } ( g _ { i } ) [ 1 + { \bf 1 } \{ | g _ { i } | > T ( g ) \} ] .\tag{52}
$$

Positive scale invariance gives $q _ { i } ( X ) = q _ { i } ( G )$ . Replacing $T ( G )$ by $\overline { { T } } _ { D } = \mathbb { E } T ( G )$ and ∥G∥ by $r _ { D } = \mathbb { E } \| G \|$ defines the oracle kernel, and for every fixed edge combination

$$
R _ { a } = R _ { a } ^ { \circ } + \Delta _ { a , \mathrm { t h r } } + \Delta _ { a , \mathrm { s h e l l } } .\tag{53}
$$

The threshold remainder is supported on threshold deviation or coordinate boundary occupancy; the shell remainder is the diference between exact cosine normalization and the fixed-radius bilinear surrogate.

## E.2 Aligned magnitude-bit theorem

Let G, H be independent with independent coordinates $G _ { i } , H _ { i } \sim \mathcal { N } ( 0 , \sigma _ { i } ^ { 2 } )$ . Put $m = \sqrt { 2 / \pi }$ $z _ { i } = \tau / \sigma _ { i } , p _ { i } = 2 [ 1 - \Phi ( z _ { i } ) ] , k _ { i } = m + 2 \varphi ( z _ { i } )$ , and $v _ { i } = 1 + 3 p _ { i }$ . Coordinate independence gives

$$
\operatorname { V a r } ( T ) = \sum _ { i } \sigma _ { i } ^ { 4 } , \qquad \operatorname { C o v } ( S _ { 1 } , T ) = m ^ { 2 } \sum _ { i } \sigma _ { i } ^ { 2 } , \qquad \operatorname { V a r } ( S _ { 1 } ) = D ,\tag{54}
$$

$$
\operatorname { C o v } ( S _ { 2 } , T ) = \sum _ { i } \sigma _ { i } ^ { 2 } k _ { i } ^ { 2 } , \qquad \operatorname { V a r } ( S _ { 2 } ) = \sum _ { i } v _ { i } ^ { 2 } .\tag{55}
$$

For every finite $z > 0$

$$
k ( z ) ^ { 2 } > m ^ { 2 } v ( z ) .\tag{56}
$$

To see this write $a = e ^ { - z ^ { 2 } / 2 }$ and $h ( z ) = k ( z ) ^ { 2 } / m ^ { 2 } - v ( z ) = 2 a + a ^ { 2 } - 6 [ 1 - \Phi ( z ) ]$ . Then $h ( 0 ) = \mathrm { l i m } _ { z  \infty } h ( z ) = 0$ and $h ^ { \prime } ( z ) = a \{ 3 m - 2 z ( 1 + a ) \}$ ; since $2 z ( 1 + a )$ is strictly increasing, h rises and then falls and stays strictly positive on $( 0 , \infty )$

Now set $t _ { i } = \sigma _ { i } ^ { 2 }$ and regard $v _ { i } = v ( t _ { i } )$ . Both $v ( t )$ and $t / v ( t )$ increase because $t v ^ { \prime } ( t ) =$ $3 z \varphi ( z ) < 1 \leq v ( t )$ . Pairwise expansion and Cauchy–Schwarz give

$$
{ \frac { \sum _ { i } t _ { i } v _ { i } } { \sum _ { i } t _ { i } } } \geq { \frac { \sum _ { i } v _ { i } ^ { 2 } } { \sum _ { i } v _ { i } } } \geq { \sqrt { { \frac { 1 } { D } } \sum _ { i } v _ { i } ^ { 2 } } } ,\tag{57}
$$

and combining this with (56) proves $\rho ( S _ { 2 } , T ) > \rho ( S _ { 1 } , T )$

The alignment between target and scorer is what makes the inequality hold. In the linear-target model $\begin{array} { r } { T = \sum _ { i } w _ { i } G _ { i } } \end{array}$ with a fixed equally weighted scorer, take $D = 8 , w =$ $( 1 , 1 , 1 , 1 , 0 , 0 , 0 , 0 )$ , coordinate standard deviations $( 1 , 1 , 1 , 1 , L , L , L , L )$ , and $\tau = m ( 1 +$ $L ) / 2$ . For $L = 8$ the two-bit minus one-bit Pearson correlation $\mathrm { i s \ - 0 . 1 6 2 6 8 7 \ldots }$ the second bit amplifies four coordinates the target ignores, and a fixed readout loses. For equalvariance independent coordinates and the random threshold $D ^ { - 1 } \sum _ { i } \left| { G _ { i } } \right|$ , the strong law and dominated convergence recover the deterministic-threshold correlation as $D \to \infty$

## E.3 Oracle covariance

For the symmetric oracle kernel $H _ { D }$ let

$$
h _ { 1 , D } ( u ) = \mathbb { E } [ H _ { D } ( u , V ) ] - \theta _ { D } , \qquad h _ { 2 , D } ( u , v ) = H _ { D } ( u , v ) - \theta _ { D } - h _ { 1 , D } ( u ) - h _ { 1 , D } ( v ) ,\tag{58}
$$

and $\kappa _ { j } = \mathbb { E } h _ { j } ^ { 2 }$ . Aggregate directed or repeated edges into unordered-pair coeficients $A _ { p }$ and node incidences $\begin{array} { r } { d _ { u } = \sum _ { p \ni u } A _ { p } } \end{array}$ . Hoefding orthogonality gives

$$
\mathrm { V a r } ( R _ { a } ^ { \circ } ) = \kappa _ { 1 } \sum _ { u } d _ { u } ^ { 2 } + \kappa _ { 2 } \sum _ { p } A _ { p } ^ { 2 } .\tag{59}
$$

Disjoint edges have zero covariance, edges sharing exactly one node have covariance $\kappa _ { 1 } .$ and identical or reversed edges have variance $2 \kappa _ { 1 } + \kappa _ { 2 }$

## E.4 Three analytical routes

The coarse route combines a Gaussian quadratic-form MGF with a bounded score range; it needs no coordinate independence and is loose by four orders of magnitude on real embeddings (Appendix I). The exact-covariance route keeps (59) and adds a cumulant or conditional-MGF condition. The empirical route calibrates the fitted role residual on independent blocks. The three routes bound the same residual tail with diferent inputs; the decision shell of Theorem 2 accepts any of them.

## Appendix F. Held-Out Certificates

Condition on nuisance parameters fitted on an independent split. Let $Z _ { 1 } , \ldots , Z _ { n }$ be independent blocks with arbitrary dependence among decisions inside a block. For threshold τ let $u ( Z , \tau )$ be the block-average indicator that a decision lies in the boundary event or the residual event, so that pointwise $0 \leq f ( Z ) \leq u ( Z , \tau ) \leq 1$ with $f ( Z )$ the block-average failure. For fixed τ, one-sided Hoefding gives

$$
p _ { F } ^ { \mathrm { b l k } } \leq \widehat { U } _ { n } ( \tau ) + \sqrt { \frac { \log ( 1 / \delta ) } { 2 n } }\tag{60}
$$

with probability at least $1 - \delta .$ . For a prespecified grid $\tau$ of size M, a union bound replaces δ by $\delta / M$ and licenses any calibration-measurable minimizer $\widehat { \tau } \in \mathcal { T }$

## F.1 Selective ratio certificates

Condition on the fit split. For selector s and threshold τ let $C _ { i , s } , U _ { i , s , \tau }$ , and $F _ { i , s }$ be the block fractions accepted, accepted and in the union event, and accepted and failed; empty blocks contribute zero, and pointwise $0 \leq F _ { i , s } \leq U _ { i , s , \tau } \leq C _ { i , s } \leq 1$ . The target is the ratio of expectations

$$
p _ { s } = \frac { \mathbb { E } F _ { i , s } } { \mathbb { E } C _ { i , s } } , \qquad q _ { s , \tau } = \frac { \mathbb { E } U _ { i , s , \tau } } { \mathbb { E } C _ { i , s } } ,\tag{61}
$$

which is the block-uniform selective risk, not the average of within-block ratios; it requires positive population coverage.

The proof of Theorem 14 fixes $q = q _ { s , \tau }$ and uses $W _ { i } ( q ) = U _ { i , s , \tau } - q C _ { i , s } \in [ - q , 1 - q ]$ with $\mathbb { E } W _ { i } ( q ) = 0$ . The interval has length one, so a one-sided Hoefding bound and a union bound over $G = | { \mathcal { G } } |$ pairs give

$$
q _ { s , \tau } \leq \frac { \widehat { U } _ { s , \tau } + \sqrt { \log ( G / \delta ) / ( 2 n ) } } { \widehat { C } _ { s } }\tag{62}
$$

simultaneously whenever $\widehat { C } _ { s } > 0$ . Replacing U by F and paying only for the selector family proves Corollary 15. No comparison-level independence is assumed.

The two-event baseline controls EU from above and EC from below separately. With $\epsilon _ { 2 } = \sqrt { \log ( 2 G / \delta ) / ( 2 n ) }$ it yields

$$
q _ { s , \tau } \leq \frac { \widehat { U } _ { s , \tau } + \epsilon _ { 2 } } { \widehat { C } _ { s } - \epsilon _ { 2 } }\tag{63}
$$

when the denominator is positive; if only $S _ { 0 }$ distinct selectors occur, 2G may be replaced by $G + S _ { 0 }$ . Using log $\left( G / \delta \right)$ for both separate events would only establish failure probability 2δ. The direct-ratio event dominates: whenever its denominator is positive, the two-event expression is a looser consequence of (23).

## F.2 Empirical-Bernstein ratio inversion

Let $n \geq 2$ , let $\widehat { V } _ { W } ( t )$ be the unbiased sample variance of $W _ { i } ( t ) = U _ { i } - t C _ { i }$ , and define

$$
L = \log ( 2 G / \delta ) , \qquad a = { \sqrt { 2 L / n } } , \qquad b = { \frac { 7 L } { 3 ( n - 1 ) } } ,\tag{64}
$$

$$
H ( t ) = \widehat { U } - t \widehat { C } + a \sqrt { \widehat { V } _ { W } ( t ) } + b , \qquad t \in [ 0 , 1 ] .\tag{65}
$$

Set

$$
\begin{array} { r } { B _ { \mathrm { E B } } = \left\{ \begin{array} { l l } { \operatorname* { i n f } \{ t \in [ 0 , 1 ] : H ( t ) \leq 0 \} , } & { \mathrm { i f ~ t h e ~ s e t ~ i s ~ n o n e m p t y } , } \\ { 1 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{66}
$$

Then $q _ { s , \tau } \leq B _ { \mathrm { E B } , s , \tau }$ <sub>τ</sub> simultaneously over the grid with probability at least $1 - \delta$

At the population truth q, the range-one empirical Bernstein inequality gives $\mathbb { P } \{ H ( q ) <$ $0 \} \leq \delta / G$ . The function H need not be monotone, but it has a single-crossing property. Write $s _ { U } = \sqrt { \widehat { V } _ { U } }$ and $\eta = b + \widehat { U } - a s _ { U }$ . Since $0 \leq U _ { i } \leq 1$

$$
s _ { U } ^ { 2 } \leq \frac { n } { n - 1 } \widehat { U } , \qquad a s _ { U } - \widehat { U } \leq \frac { L } { 2 ( n - 1 ) } ,\tag{67}
$$

so $\eta \geq 1 1 L / [ 6 ( n - 1 ) ] > 0$ . For $0 < r < t \leq 1$ put $\lambda = t / r$ . From $W ( t ) = \lambda W ( r ) - ( \lambda - 1 ) U$ and the triangle inequality for sample standard deviations,

$$
H ( t ) \leq \lambda H ( r ) - ( \lambda - 1 ) \eta ,\tag{68}
$$

so $H ( r ) \leq 0$ implies $H ( t ) < 0$ for every $t > r$ . Since $H ( 0 ) > 0$ and H is continuous, the first crossing in (66) is a valid upper confidence endpoint and no grid over t is needed. Numerically, if $H ( 1 ) \geq 0$ the implementation returns one; otherwise bisection returns the upper bracket endpoint.

## F.3 Independent selection and multiplicity

If the block budget is split before observation into selection and certification blocks, a grid pair and certificate family may be chosen on the first part and, conditionally on that choice, certified on the second part with the single-policy width $\sqrt { \log ( 1 / \delta ) / ( 2 n _ { \mathrm { c e r t } } ) }$ and no grid factor. Choosing again after viewing several certification results requires a common simultaneous guarantee. Likewise each prespecified method and sample size carries a marginal guarantee, and selecting the smallest certificate across a curve requires a confidence allocation across the points eligible for selection.

## Appendix G. Cross-Quantizer Instantiations

## G.1 RaBitQ

For a fixed unit data direction o and query direction q, RaBitQ uses

$$
{ \widehat s } _ { R } ( o , q ) = \frac { \langle \bar { o } ( { \cal P } ) , q \rangle } { \langle \bar { o } ( { \cal P } ) , o \rangle } ,\tag{69}
$$

whose denominator is positive for every orthogonal P because $\langle \bar { o } ( P ) , o \rangle = \| P ^ { \top } o \| _ { 1 } / \sqrt { D } \geq$ $1 / \sqrt { D }$ . The random-radius event is measurable under a fixed tie rule. A shared rotation couples the edges, so ranking and pruning statements go through the triangle inequality and a union bound over the joint event rather than through independence, and observing the realized radius does not create conditional coverage. For raw squared distance,

$$
\xi _ { R } ^ { d ^ { 2 } } ( u _ { r } , v _ { r } ) = - 2 r _ { u } r _ { v } \varepsilon _ { R } ( u , v ) ,\tag{70}
$$

and a pruning rule stated in Euclidean distance becomes an $\alpha ^ { 2 }$ rule after squaring, with the two edges keeping distinct radial factors. A rule stated directly in squared distance or cosine dissimilarity keeps its own parameter.

## G.2 BBQ and block estimators

Lucene BBQ defines asymmetric scorer roles that determine graph topology: stored vectors are one-bit, temporary int4 query-role vectors support graph construction, and candidate collection as well as diversity and reverse-link scoring each need role-specific residual calibration. Our reference scorer implements the role semantics. Product quantizers and related block estimators enter through block residual sums, whose survival laws come from bounded blocks, block covariance, or held-out calibration. Native binary embeddings without a float teacher metric fall outside the latent-float residual formulation.

## Appendix H. Necessity and Replacement Concentration

## H.1 Rare-contamination counterexample

Let $\varepsilon _ { D } = D ^ { - 4 } , A _ { D } = D ^ { 3 / 2 }$ , and

$$
G _ { D } = ( 1 - J _ { D } ) Z _ { D } + J _ { D } A _ { D } S _ { D } ,\tag{71}
$$

where $J _ { D } \sim \mathrm { B e r n o u l l i } ( \varepsilon _ { D } ) , Z _ { D } \sim \mathcal { N } ( 0 , I _ { D } )$ , and $S _ { D }$ has independent Rademacher coordinates; independent nodes use independent copies. Then $\mathrm { C o v } ( G _ { D } ) = ( 1 + D ^ { - 1 } - D ^ { - 4 } ) I _ { D }$ For every fixed coordinate block of size $k ,$ , mixture decomposition and Gaussian scale comparison give total-variation distance $O _ { k } ( D ^ { - 1 } )$ from the standardized Gaussian law. With $r _ { D } = \mathbb { E } \| G _ { D } \| \sim \sqrt { D }$

$$
\mathbb { E } \left( \frac { \| G _ { D } \| } { r _ { D } } - 1 \right) ^ { 2 } = O ( D ^ { - 1 } ) ,\tag{72}
$$

a relative mean-square statement whose $L ^ { 2 }$ norm is $O ( D ^ { - 1 / 2 } )$ . At the population magnitude threshold the expected fraction of coordinates in a shrinking boundary band tends to zero.

Fix $0 < \eta \leq 1 / 8 , \beta _ { D } = \eta / D$ , and define the fixed-threshold oracle residual

$$
R _ { D } = Q _ { D } + B _ { D } ,
$$

$$
Q _ { D } = { \frac { G _ { 1 } ^ { \top } ( G _ { 3 } - G _ { 2 } ) } { r _ { D } ^ { 2 } } } ,\tag{73}
$$

(74)

$$
B _ { D } = - \frac { \eta } { D } \sum _ { i } q _ { i } ^ { \circ } ( G _ { 1 } ) \{ q _ { i } ^ { \circ } ( G _ { 3 } ) - q _ { i } ^ { \circ } ( G _ { 2 } ) \} .\tag{75}
$$

Conditioning on the three contamination indicators and combining the two shared-endpoint edges coordinatewise gives independent centred coordinate summands, whence

$$
\begin{array} { r } { \mathbb { E } B _ { D } ^ { 2 } \leq 3 2 \eta ^ { 2 } / D , \qquad \mathbb { E } B _ { D } ^ { 4 } \leq 3 0 7 2 \eta ^ { 4 } / D ^ { 2 } , } \end{array}\tag{76}
$$

while $\mathrm { V a r } ( Q _ { D } ) ~ = ~ 2 D \sigma _ { D } ^ { 4 } / r _ { D } ^ { 4 } ~ \sim ~ 2 / D$ . The $L ^ { 2 }$ reverse triangle inequality gives $V _ { D } \ =$ $\mathrm { V a r } ( R _ { D } ) = \Theta ( D ^ { - 1 } )$

On $\mathcal { A } _ { D } = \{ J _ { 1 } = J _ { 2 } = 1 , J _ { 3 } = 0 \}$ , whose probability is $\varepsilon _ { D } ^ { 2 } ( 1 - \varepsilon _ { D } ) = \Theta ( D ^ { - 8 } )$

$$
Q _ { D } = { \frac { A _ { D } S _ { 1 } ^ { \top } Z _ { 3 } - A _ { D } ^ { 2 } S _ { 1 } ^ { \top } S _ { 2 } } { r _ { D } ^ { 2 } } } .\tag{77}
$$

Since $\mathbb { E } ( S _ { 1 } ^ { \top } S _ { 2 } ) ^ { 4 } = 3 D ^ { 2 } - 2 D$

$$
\mathbb { E } [ Q _ { D } ^ { 4 } ; \mathcal { A } _ { D } ] \ge \frac { \varepsilon _ { D } ^ { 2 } ( 1 - \varepsilon _ { D } ) A _ { D } ^ { 8 } ( 3 D ^ { 2 } - 2 D ) } { r _ { D } ^ { 8 } } = ( 3 + o ( 1 ) ) D ^ { 2 } .\tag{78}
$$

The $L ^ { 4 }$ reverse triangle inequality and the bound on $B _ { D }$ give $\mathbb { E } R _ { D } ^ { 4 } = \Omega ( D ^ { 2 } )$ , hence

$$
\kappa _ { 4 } ( R _ { D } ) = \Omega ( D ^ { 2 } ) , \qquad \kappa _ { 4 } ( R _ { D } ) / V _ { D } ^ { 2 } = \Omega ( D ^ { 4 } ) .\tag{79}
$$

For a cumulant condition $| \kappa _ { m } ( \boldsymbol { R _ { D } } ) | \leq ( m ! / 2 ) V _ { D } b _ { D } ^ { m - 2 }$ , the case $m = 4$ forces $b _ { D } / \sqrt { V _ { D } } =$ $\Omega ( D ^ { 2 } )$ . For a two-sided MGF bound

$$
\log \mathbb { E } e ^ { \lambda X } \leq \frac { \lambda ^ { 2 } v } { 2 ( 1 - b | \lambda | ) } , \qquad | \lambda | < 1 / b ,\tag{80}
$$

evaluating both signs at $\lambda = [ 2 \operatorname* { m a x } \{ \sqrt { v } , b \} ] ^ { - 1 }$ and using cosh $y - 1 \geq y ^ { 4 } / 2 4$ gives $\mathbb { E } X ^ { 4 } \leq$ 192v max $\{ v , b ^ { 2 } \}$ , so $v _ { D } = O ( V _ { D } )$ again forces $b _ { D } / \sqrt { V _ { D } } = \Omega ( D ^ { 2 } )$ . The construction uses a fixed threshold, fixed normalization, and the oracle slope; it is a logical obstruction, and §8.7 measures how far real embeddings sit from it.

## H.2 Replacement variance

For the Hoefding decomposition of $F _ { a }$ , independent replacement of node u gives

$$
\begin{array} { r } { \frac 1 2 \mathbb { E } ( F _ { a } - F _ { a } ^ { ( u ) } ) ^ { 2 } = \mathbb { E } \operatorname { V a r } ( F _ { a } \mid G _ { - u } ) . } \end{array}\tag{81}
$$

Summing over nodes counts every first-order component once and every degenerate pair component twice,

$$
V _ { \mathrm { E S } } = \kappa _ { 1 } \sum _ { u } d _ { u } ^ { 2 } + 2 \kappa _ { 2 } \sum _ { p } A _ { p } ^ { 2 } ,\tag{82}
$$

hence $V \leq V _ { \mathrm { E S } } \leq 2 V$ with equality at two when $h _ { 1 } = 0$ and $h _ { 2 } \neq 0$

Table 11: Representation regimes. Each result file records the subset of datasets it uses.
<table><tr><td>Group</td><td>Datasets</td><td>D</td><td>Role</td></tr><tr><td>Contrastive text</td><td>Cohere, MiniLM, BGE-M3, Jina, MSMARCO</td><td>384-1024</td><td>primary regimes</td></tr><tr><td rowspan="3">Vision Classical Synthetic</td><td>Wolt-CLIP, Landmark-DINO</td><td>512-768</td><td>modality transfer</td></tr><tr><td>GloVe, SIFT, GIST</td><td>100-960</td><td>negative controls</td></tr><tr><td>Gaussian, sphere, random, contaminated</td><td>64-2048</td><td>mechanism and stress</td></tr></table>

## H.3 Conditional MGF and stable truncation

Let $D _ { j }$ be the three Doob increments. If deterministic $v _ { j } , b _ { j }$ satisfy

$$
\log \mathbb { E } [ e ^ { \lambda D _ { j } } \mid \mathcal { F } _ { j - 1 } ] \leq \frac { \lambda ^ { 2 } v _ { j } } { 2 ( 1 - b _ { j } | \lambda | ) } ,\tag{83}
$$

iterated conditioning gives the tail (20) with $\begin{array} { r } { v _ { \star } = \sum _ { j } v _ { j } } \end{array}$ and $b _ { \star } = \operatorname* { m a x } _ { j } b _ { j }$ . The expectation of a replacement proxy alone cannot be inserted into Freedman’s inequality; the conditional bounds must hold almost surely.

A truncation route constructs a functional $F _ { L }$ on the full input space with

$$
\mathbb { P } ( F _ { L } \neq F _ { a } ) \leq \varepsilon _ { L } , \qquad | \mathbb { E } F _ { a } - \mathbb { E } F _ { L } | \leq d _ { L } .\tag{84}
$$

If $F _ { L }$ has a sub-gamma tail with $( v _ { L } , b _ { L } )$ , then

$$
\mathbb { P } ( | F _ { a } - \mathbb { E } F _ { a } | \ge z + d _ { L } ) \le 2 \exp \left[ - \frac { z ^ { 2 } } { 2 ( v _ { L } + b _ { L } z ) } \right] + \varepsilon _ { L } .\tag{85}
$$

Bounding replacements only on a good event sufices only when that event is stable under every single-node replacement, which is what the construction of $F _ { L }$ on the full space provides.

## H.4 Negative component covariance

For zero-mean Gaussian nodes of arbitrary covariance define $M _ { i j } = \mathbb { E } [ q _ { i } ^ { \circ } ( G ) G _ { j } ]$ . Endpoint independence gives

$$
\mathrm { C o v } ( B _ { a } , Q _ { a } ) = - \frac { \beta _ { 1 } } { r _ { D } ^ { 2 } } \Bigl ( \sum _ { p } A _ { p } ^ { 2 } \Bigr ) \| M \| _ { F } ^ { 2 } \le 0 ,\tag{86}
$$

with no symmetry or positive-semidefiniteness of M required. At nonzero mean, first-order projections reappear and the sign depends on separate cross-Hoefding conditions.

## Appendix I. Datasets, Protocols, and Experiment Inventory

Every experiment draws a bounded sample from a memory-mapped source artifact rather than loading a full million-vector file. Calibration, validation, reference-integration, and stress pools are disjoint whenever the estimand requires it. Table 12 maps each experiment in the paper to its script and result file; script numbers refer to the numbered modules of the code release, and result files are named by experiment number with a sufix identifying the standard non-saturated RobustPrune rule where it is involved.

Table 12: Experiment inventory.
<table><tr><td>Section</td><td>Object</td><td>Scripts</td><td>Results</td></tr><tr><td>§8.2</td><td>margin bins, boundary exponent, held-out predictor</td><td>07, 08, 16</td><td>24, 25, 33</td></tr><tr><td>§8.3</td><td>shared-query covariance, selection regimes</td><td>09, 29</td><td>26, 46</td></tr><tr><td>§8.4</td><td>pruning replay, trace, edge, path, refill variant</td><td>10-14</td><td>27-31</td></tr><tr><td>§8.5</td><td>selective certificates, sample-size curves</td><td>17, 28</td><td>34, 45</td></tr><tr><td>§8.7</td><td>joint Gaussianity, oracle remainders, variance proxies</td><td>15, 19, 21, 22</td><td>32, 36, 38, 39</td></tr><tr><td>§8.8</td><td>conditional-MGF and truncation parameters</td><td>23</td><td>40</td></tr><tr><td>§8.9</td><td>margin-zero and matched-Spearman</td><td>18</td><td>35</td></tr><tr><td>§7 §5</td><td>constructions cross-quantizer interface representation diagnostics,</td><td>20 01-06</td><td>37 1-23</td></tr></table>

Table 13: Geometry diagnostics. $d \textmu _ { 9 0 }$ is the PCA dimension explaining 90% of the variance; $\Delta \theta _ { \mathrm { N N } }$ is a representative nearest-neighbour angular gap.
<table><tr><td>Dataset</td><td>D</td><td> $d _ { 9 0 } / D$ </td><td>Mean angle</td><td>NN angle</td><td> $\Delta \theta _ { \mathrm { N N } }$ </td></tr><tr><td>Cohere</td><td>768</td><td>.309</td><td> $4 5 . 8 ^ { \circ }$ </td><td> $3 5 . 4 ^ { \circ }$ </td><td> $. 1 2 5 ^ { \circ }$ </td></tr><tr><td>BGE-M3</td><td>1024</td><td>.091</td><td> $5 5 . 4 ^ { \circ }$ </td><td>34.5°</td><td> $. 2 4 8 ^ { \circ }$ </td></tr><tr><td>MiniLM</td><td>384</td><td>.497</td><td>88.8°</td><td>65.6°</td><td>.277°</td></tr><tr><td>GIST</td><td>960</td><td>.181</td><td> $4 0 . 7 ^ { \circ }$ </td><td> $2 8 . 9 ^ { \circ }$ </td><td> $. 0 9 1 ^ { \circ }$ </td></tr><tr><td>Random</td><td>768</td><td>.268</td><td> $8 7 . 6 ^ { \circ }$ </td><td> $6 7 . 9 ^ { \circ }$ </td><td> $. 1 1 7 ^ { \circ }$ </td></tr></table>

## Appendix J. Representation Diagnostics

The diagnostics in this section motivated the Gaussian model of $\ S 5$ and the eligibility gate.

Intrinsic dimension and average angle do not separate the regimes: GIST has the most concentrated geometry and the smallest local angular gap yet is the worst coordinatesign regime, while random vectors have regular high-dimensional geometry and no usable neighbour structure. The final theory conditions on decision margins and residual placement instead.

The GIST row is the key negative control: almost all coordinate signs agree regardless of angular relation, whereas random hyperplanes keep an angular collision law. This is the origin of both the rotation mechanism and the eligibility gate.

In the same-dimensional survey of nine 768-dimensional regimes, eight had $R ^ { 2 } \geq . 9 9 6 5 ;$ the near-isotropic RoBERTa regime had low $R ^ { 2 }$ because its entropy variance is close to zero, with MAE .00166. These are coordinate-wise checks; the joint Gaussianity audit of §8.7 is the stronger test.

Table 14: Coordinate-sign versus random-hyperplane diagnostics. GW is the angular collision prediction and $H _ { \mathrm { s i g n } }$ the normalized sign entropy.
<table><tr><td>Dataset</td><td>Coord. sign</td><td>Random HP</td><td>GW</td><td>KL(coord||HP)</td><td> $H _ { \mathrm { s i g n } }$ </td></tr><tr><td>Cohere</td><td>.650</td><td>.744</td><td>.746</td><td>.0229</td><td>.747</td></tr><tr><td>BGE-M3</td><td>.677</td><td>.701</td><td>.692</td><td>.0022</td><td>.700</td></tr><tr><td>MiniLM</td><td>.508</td><td>.506</td><td>.506</td><td>.0021</td><td>.987</td></tr><tr><td>GIST</td><td>.9999</td><td>.768</td><td>.774</td><td>.2648</td><td>.0004</td></tr><tr><td>Random</td><td>.513</td><td>.513</td><td>.513</td><td>.0010</td><td>.981</td></tr></table>

Table 15: Anisotropic Gaussian sign-entropy model. Sign entropy is predicted from coordinate SNR; $R ^ { 2 }$ is unstable when entropy has almost no variance across coordinates, so MAE is reported alongside.
<table><tr><td>Dataset</td><td>D</td><td> $R ^ { 2 }$ </td><td>MAE</td><td>Mean |SNR|</td><td>Measured / predicted entropy</td></tr><tr><td>Cohere</td><td>768</td><td>.9996</td><td>.004</td><td>.884</td><td>.747 / .748</td></tr><tr><td>BGE-M3</td><td>1024</td><td>.9989</td><td>.007</td><td>.908</td><td>.700 / .699</td></tr><tr><td>MiniLM</td><td>384</td><td>.9222</td><td>.002</td><td>.127</td><td>.987 / .988</td></tr><tr><td>Random</td><td>768</td><td>.9967</td><td>.001</td><td>.164</td><td>.981 / .982</td></tr><tr><td>GIST</td><td>960</td><td>failure</td><td>.240</td><td>1.763</td><td>.000 / .240</td></tr><tr><td>MSMARCO-Cohere</td><td>1024</td><td>.9995</td><td>.0019</td><td>.370</td><td>.910 / .910</td></tr></table>

## Appendix K. Representation Mechanisms

Tables 16 to 19 collect the representation-level measurements behind §5: the fidelity and recall gain of the magnitude bit, a controlled intervention on coordinate heterogeneity, the rotation response of the two-bit code, and the predictors of that response.

In Table 19, excluding the one non-Gaussian outlier (Landmark-Nomic) raises the entropy-gap correlation to .964 and the heterogeneity correlation to .746; the response is independent of dimension. In the intervention study of Table 17, the correlation between heterogeneity and magnitude gain was +1.0 in four of five datasets.

Rotation response and eligibility are diferent questions. Cohere $( H _ { \mathrm { s i g n } } = . 7 4 7$ $\Delta F _ { \mathrm { r o t } } = + . 1 7 0 )$ and SIFT $( H _ { \mathrm { s i g n } } = . 7 4 6 , \Delta F _ { \mathrm { r o t } } = + . 2 0 0 )$ are nearly identical on both statistics, yet coordinate binary quantization is usable on Cohere and not on SIFT. The entropy gap predicts how much rotation changes a code; the eligibility gate (calibration slope, margin correlation, residual tail) decides whether the unrotated code was usable at all. This pair of datasets is why the paper reports two diagnostics rather than one routing statistic.

Table 16: Magnitude-bit gain in global ranking fidelity and simulated recall across representations.
<table><tr><td>Dataset</td><td> $\operatorname { C V } ( \sigma )$ </td><td> $\Delta F$  ∆ Recall</td></tr><tr><td>MiniLM</td><td>.118</td><td>+.132 +.174</td></tr><tr><td>Cohere</td><td>.182</td><td>+.091 +.144</td></tr><tr><td>CodeSearch</td><td>.098 +.088</td><td>+.143</td></tr><tr><td>Landmark</td><td>.110 +.088</td><td>+.110</td></tr><tr><td>Arxiv</td><td>.108 +.071</td><td>+.159</td></tr><tr><td>Random</td><td>.070 +.049</td><td>+.210</td></tr></table>

Table 17: Controlled coordinate-heterogeneity intervention on Cohere. Absolute fidelity and the magnitude-bit gain move in opposite directions.
<table><tr><td>Intervention</td><td> $\operatorname { C V } ( \sigma )$ </td><td> $F _ { \mathrm { 1 b i t } }$ </td><td> $F _ { \mathrm { 2 b i t } }$ </td><td> $\Delta F _ { \mathrm { m a g } }$ </td></tr><tr><td>Amplify  $\gamma = 2$ </td><td>.316</td><td>.682</td><td>.719</td><td>.036</td></tr><tr><td>Amplify  $\gamma = 1 . 5$ </td><td>.252</td><td>.694</td><td>.728</td><td>.034</td></tr><tr><td>Original</td><td>.182</td><td>.706</td><td>.738</td><td>.032</td></tr><tr><td>Whiten .50</td><td>.091</td><td>.720</td><td>.746</td><td>.026</td></tr><tr><td>Flatten</td><td>.002</td><td>.736</td><td>.757</td><td>.022</td></tr></table>

## Appendix L. Local Ranking and Boundary Diagnostics

The last two rows are the operational summary: on eligible two-bit configurations, two thirds to three quarters of decisions lie in the low-risk region $M \geq 2$ and 7 to 9% fall below $M = 1$

## Appendix M. Pruning and Trace Composition

Every observed triple disagreement is a false keep. Fixed-order output divergence occurs on 99.4 to 100% of sampled calls although local triple disagreement rates are 1.9 to 13.1%, as the first-divergence theorem permits: one candidate-level action disagreement changes an append-only output. In this experiment rotation leaves the fixed-order pruning statistics nearly unchanged, while the free-order replay also reflects candidate re-sorting.

The implication “certificate $\mathrm { p a s s e s } \Rightarrow \mathrm { e d g e }$ survives” held in every tested case. The numerical saturation reflects the conservativeness of witness-level additive probabilities, while the risk sum still orders edges by their observed failure.

Path certificates saturate within two hops, and the observed path failure rate grows with length as the union of its edge failures. The certificate suficiency implication held on every path.

Table 18: Rotation responses of the two-bit code. Recall changes follow the original experiment reports.
<table><tr><td>Dataset</td><td>Sign entropy before → after</td><td>Recall response</td><td>Regime</td></tr><tr><td>GIST</td><td> $. 0 0 0  . 5 1 1$ </td><td>+307%</td><td>degenerate signs</td></tr><tr><td>Wolt-CLIP</td><td>.836 → .616</td><td>+3.2 pp</td><td>over-spread</td></tr><tr><td>Cohere</td><td> $. 7 4 7  . 5 6 3$ </td><td>−0.5 pp</td><td>coordinate signal</td></tr><tr><td>MiniLM</td><td>approximately unchanged approximately zero</td><td></td><td>near-isotropic</td></tr></table>

Table 19: Predictors of the global two-bit rotation response $\Delta F _ { \mathrm { r o t } }$ over 12 datasets.
<table><tr><td>Predictor</td><td>Spearman ρ</td><td>p</td></tr><tr><td> $\operatorname { C V } ( \sigma )$ </td><td>+.657</td><td>.020</td></tr><tr><td> $H _ { \mathrm { s i g n } }$ </td><td>-.909</td><td>&lt; .001</td></tr><tr><td> $1 - H _ { \mathrm { s i g n } }$ </td><td>+.909</td><td>&lt; .001</td></tr><tr><td> $( 1 - \bar { H _ { \mathrm { s i g n } } } ) \times \mathrm { C V } ( \sigma )$ </td><td> $\mathbf { + . 9 3 0 } \quad < . 0 0 1$ </td><td></td></tr><tr><td>log D</td><td>-.080</td><td>.805</td></tr></table>

## Appendix N. Analytical Eligibility and Variance Proxies

The MiniLM and GIST source artifacts are stored unit-normalized, so their pre-normalization shell is not identifiable; Cohere retains radial information, with norm coeficient of variation 4.4% and 2.03% of vectors beyond a 10% relative shell.

All 28 component cross-covariances are negative, with median cancellation ratio .947. In dimension stress tests from $D = 6 4$ to 2048, clean, equicorrelated, and block-correlated Gaussian regimes keep the oracle $ { b _ { \star } } / \mathrm { s d }$ near .19 to .25; a dense global shock raises it towards .88 and is detected at the same time by the Gaussian, shell, and spectrum diagnostics. The rare-contamination construction of Theorem 13 is designed to pass those diagnostics, which is what makes it the relevant obstruction.

## Appendix O. Held-Out Certificates

This protocol uses 128 coeficient-fit, 512 risk-calibration, 512 validation, 256 stress-cutof, and 512 stress-evaluation blocks per configuration. The prespecified grid has five selector cutofs and seven thresholds plus the accept-all case, so $| \mathcal { G } | = 4 0 ;$ with $n = 5 1 2$ the width is $\epsilon _ { n } = . 0 8 0 8$ . The certificate is the two-event form $( \widehat { U } + \epsilon _ { n } ) / ( \widehat { C } - \epsilon _ { n } )$ , which Appendix F.1 shows to be a looser consequence of the direct-ratio event whenever its denominator is positive. Ranking certificates range from 11.23 to 58.25% and pruning certificates from 11.17 to 46.46%. The stress columns evaluate the frozen cutof on the hardest fifth of margins from a separate pool.

All 48 role configurations satisfy the block-level invariant $0 \leq F _ { i } \leq U _ { i } \leq C _ { i } \leq 1$ , and every prespecified method and sample size showed validation risk below its certificate on all configurations. Certificate values decrease as the number of certification blocks grows; validation risks change little, while each method may select a diferent policy at each sam-

Table 20: Fixed-local-set decision study on 32-neighbour candidate sets. The eligibility column is the empirical gate of §6.
<table><tr><td>Dataset</td><td>Quantizer</td><td>Eligible</td><td>Flip</td><td>Query failure</td><td>Boundary exponent</td></tr><tr><td>Cohere</td><td>1-bit</td><td>yes</td><td>8.89%</td><td>63.12%</td><td>1.526</td></tr><tr><td>Cohere</td><td>2-bit</td><td>yes</td><td>5.83%</td><td>47.50%</td><td>1.526</td></tr><tr><td>MiniLM</td><td>1-bit</td><td>yes</td><td>11.21%</td><td>59.38%</td><td>1.500</td></tr><tr><td>MiniLM</td><td>2-bit</td><td>yes</td><td>4.19%</td><td>43.75%</td><td>1.500</td></tr><tr><td>GIST</td><td>1-bit</td><td>no</td><td>100.00%</td><td>100.00%</td><td>1.595</td></tr><tr><td>GIST</td><td>2-bit</td><td>no</td><td>35.83%</td><td>91.88%</td><td>1.595</td></tr></table>

Table 21: Standardized-margin bins, n = 4,960 decisions per configuration, with counts in parentheses; Table 6 pools these into six bins. On the ineligible GIST configuration the flip rate is flat in the margin, so no cutof separates safe from unsafe decisions.
<table><tr><td>Bin</td><td>Cohere 1-bit</td><td>Cohere 2-bit</td><td>MiniLM 1-bit</td><td>MiniLM 2-bit</td><td>GIST 2-bit</td></tr><tr><td>[0, .25) [.25, .5) [.5, .75)</td><td>33.3% (93) 31.3% (150)</td><td>48.6% (70) 35.3% (102)</td><td>48.1% (104)</td><td>50.0% (72)</td><td>41.1% (158)</td></tr><tr><td></td><td>30.2% (182)</td><td>24.8% (121)</td><td>36.0% (125) 37.4% (227)</td><td>39.6% (53) 29.5% (78)</td><td>37.5% (301) 39.2% (551)</td></tr><tr><td>[.75, 1)</td><td>21.3% (286)</td><td>20.3% (158)</td><td>24.7% (312)</td><td>23.9% (138)</td><td>35.7% (762)</td></tr><tr><td>[1, 1.5)</td><td></td><td>11.9% (520)</td><td></td><td></td><td></td></tr><tr><td>[1.5, 2)</td><td>13.2% (832)</td><td>7.68% (690)</td><td>15.8% (991)</td><td>11.5% (355)</td><td>28.2% (1389)</td></tr><tr><td>[2, 3)</td><td>8.49% (1083)</td><td>2.54% (1577)</td><td>10.2% (1064)</td><td>5.95% (571)</td><td>34.2% (803)</td></tr><tr><td></td><td>3.02% (1458)</td><td></td><td>2.53% (1227)</td><td>1.29% (1393)</td><td>40.8% (622)</td></tr><tr><td>[3,5)</td><td>0.12% (804)</td><td>0.14% (1413)</td><td>0.28% (718)</td><td>0.13% (1531)</td><td>42.9% (231)</td></tr><tr><td>[5,∞)</td><td>0.00% (72)</td><td>0.00% (309)</td><td>0.00% (192)</td><td>0.00% (769)</td><td>63.6% (143)</td></tr><tr><td>Share  $M \geq 2$  Share  $M < 1$ </td><td>47.1% 14.3%</td><td>66.5% 9.1%</td><td>43.1% 15.5%</td><td>74.5%</td><td>20.1%</td></tr></table>

ple size. The ranking failure label uses the same lexicographic tie-breaking as candidate ordering, and pruning uses the nonnegative-is-dominated convention.

## Appendix P. Candidate-Selection Covariance

This experiment holds one global afine edge scorer fixed and varies the pair-generation law, using 6,000 sampled vectors, 60,000 independent fit pairs, 160 query blocks, and 128 candidate pairs per query. The i.i.d. control samples all three endpoint identities independently with replacement; a distinct-triple control rejects identity collisions.

In every regime the weighted within/between identity (39) reproduces the pooled covariance and the diference variance to floating-point precision. Query-block bootstrap intervals accompany the grouped regimes, and the 5th, 50th, and 95th percentiles of query-specific within covariance are retained because a small weighted average need not describe every query. The anchor’s left residual is constant within a query, so its within-query correlation is reported as undefined.

Table 22: Shared-query covariance audit on 32-neighbour anchor pairs. “Ind./true” is the independence variance divided by the observed diference variance.
<table><tr><td>Dataset</td><td>Quantizer</td><td>Flip</td><td>Residual corr.</td><td>Ind./true</td><td>Tail gate</td></tr><tr><td>Cohere</td><td>1-bit</td><td>8.89%</td><td>.794</td><td>4.71</td><td>pass</td></tr><tr><td>Cohere</td><td>2-bit</td><td>5.83%</td><td>.870</td><td>7.42</td><td>pass</td></tr><tr><td>Cohere</td><td>rotated 1-bit</td><td>15.58%</td><td>.465</td><td>1.87</td><td>pass</td></tr><tr><td>Cohere</td><td>rotated 2-bit</td><td>8.81%</td><td>.435</td><td>1.77</td><td>pass</td></tr><tr><td>MiniLM</td><td>1-bit</td><td>11.21%</td><td>.047</td><td>1.05</td><td>pass</td></tr><tr><td>MiniLM</td><td>2-bit</td><td>4.19%</td><td>.093</td><td>1.10</td><td>pass</td></tr><tr><td>GIST</td><td>1-bit</td><td>100.00%</td><td>.902</td><td>8.89</td><td>abstain</td></tr><tr><td>GIST</td><td>2-bit</td><td>35.83%</td><td>.832</td><td>5.70</td><td>abstain</td></tr><tr><td>GIST</td><td>rotated 1-bit</td><td>14.80%</td><td>.585</td><td>2.34</td><td>pass</td></tr><tr><td>GIST</td><td>rotated 2-bit</td><td>9.33%</td><td>.529</td><td>2.09</td><td>pass</td></tr></table>

Table 23: Standard non-saturated RobustPrune replay with exact candidate pools (α = 1.2, M = 32, 64 candidates, 160 targets). Free-order divergence includes the sorting change.
<table><tr><td>Dataset</td><td>Quantizer</td><td>Local flip</td><td>Fixed-order divergence</td><td>Fixed Jaccard</td><td>Free Jaccard</td></tr><tr><td>Cohere</td><td>2-bit</td><td>13.06%</td><td>100.0%</td><td>.322</td><td>.395</td></tr><tr><td>Cohere</td><td>rotated 2-bit</td><td>13.07%</td><td>100.0%</td><td>.322</td><td>.289</td></tr><tr><td>MiniLM</td><td>2-bit</td><td>1.92%</td><td>100.0%</td><td>.654</td><td>.529</td></tr><tr><td>MiniLM</td><td>rotated 2-bit</td><td>1.92%</td><td>100.0%</td><td>.656</td><td>.529</td></tr><tr><td>GIST</td><td>2-bit</td><td>7.91%</td><td>99.4%</td><td>.458</td><td>.260</td></tr><tr><td>GIST</td><td>rotated 2-bit</td><td>7.90%</td><td>99.4%</td><td>.458</td><td>.359</td></tr></table>

## Appendix Q. Cross-Quantizer and Conditional-MGF Audits

Table 4 averages over Cohere, MiniLM, SIFT, and GIST. Each family uses 1,200 fit vectors and 2,000 held-out vectors per dataset, generating 60,000 ranking and 60,000 pruning triples. The RaBitQ reference uses a seeded signed-DCT orthogonal transform without query scalar quantization; the BBQ-like reference implements centroid-centred one-bit storage with an int4 query role and omits Lucene’s metric-specific corrections; the PQ reference uses 16 subspaces with 16 centroids each, for 64 index bits per vector excluding codebooks.

The four disjoint pools contain 900 fit, 1,200 reference, 1,200 calibration, and 1,200 validation vectors. Nested integration uses 160 first-node groups, 16 second nodes, 12 observed third nodes, and separate reference completions; clipping cutofs are the .95, .975, and .99 calibration quantiles. No held-out tail exceeded its hybrid bound.

## Appendix R. Negative Controls

The framework’s failure modes are distinct and each is observable: nonpositive calibration, heavy residual tails, high boundary mass, a saturated trace or path certificate, a postprocessing map that merges distinct traces, an unidentifiable shell, and deployment blocks

Table 24: Exact-prefix edge certificates under standard RobustPrune (two-bit codes). “Saturation” is the fraction of plug-in additive bounds at least one; the last column is the Spearman correlation between the plug-in risk sum and observed edge failure.
<table><tr><td>Dataset</td><td>Edges</td><td>Mean prefix terms</td><td>Fixed-order failure</td><td>Saturation</td><td>Risk-failure ρ</td></tr><tr><td>Cohere</td><td>3,054</td><td>24.7</td><td>34.3%</td><td>89.0%</td><td>.81</td></tr><tr><td>MiniLM</td><td>5,118</td><td>21.0</td><td>21.5%</td><td>89.4%</td><td>.69</td></tr><tr><td>GIST</td><td>3,701</td><td>21.7</td><td>24.7%</td><td>91.6%</td><td>.73</td></tr></table>

Table 25: Saturating refill applied after the standard call (α = 1.2, M = 32). The refill is a post-processing variant and not part of RobustPrune.
<table><tr><td>Dataset</td><td>Diversity-selected</td><td>Refill-added</td><td>Refill share</td></tr><tr><td>Cohere</td><td>21.07</td><td>10.93</td><td>34.14%</td></tr><tr><td>MiniLM</td><td>32.00</td><td>0.00</td><td>0.00%</td></tr><tr><td>GIST</td><td>22.80</td><td>9.20</td><td>28.75%</td></tr></table>

that difer from calibration blocks. Each produces either an empirically high risk or an uncertified decision.

## Appendix S. Code Release

The release consists of numbered, non-interactive Python modules with fixed seeds and machine-readable outputs: modules 01–06 compute the representation diagnostics of Appendices J and K; 07–09 the decision-stability and covariance experiments; 10–14 the pruning replay, trace, edge, refill, and path experiments; 15, 19, 21, and 22 the Gaussian eligibility audits; 16, 17, and 28 the held-out certificates; 18 the necessity constructions; 20 the cross-quantizer interface; 23 the conditional-MGF parameters; 24–27 the semantic and numerical audits of the pruning rule, calibration, Stein identity, and contamination formulas; and 29 the candidate-selection covariance experiment. Samples are drawn without loading complete artifacts, no validation outcome is used to retune a frozen certificate, and negative results, ineligible configurations, and saturated bounds remain in the result files.

## References

Roy Betser, Eyal Gofer, Meir Yossef Levi, and Guy Gilboa. Infonce induces gaussian distribution. In International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=BlSH7gNQSq. Oral presentation.

St´ephane Boucheron, G´abor Lugosi, and Pascal Massart. Concentration Inequalities: A Nonasymptotic Theory of Independence. Oxford University Press, 2013.

Moses S Charikar. Similarity estimation techniques from rounding algorithms. In Proceedings of the 34th Annual ACM Symposium on Theory of Computing, pages 380–388,

Table 26: Fixed-order diversity-path composition for the unrotated two-bit scorer, 600 paths per length. “Fail” is the observed path failure rate and “sat.” the fraction of saturated plug-in certificates, both in percent.
<table><tr><td rowspan="2">Dataset</td><td colspan="2">1 hop</td><td colspan="2">2 hops</td><td colspan="2">3 hops</td><td colspan="2">4 hops</td></tr><tr><td>fail</td><td>sat.</td><td>fail</td><td>sat.</td><td>fail</td><td>sat.</td><td>fail</td><td>sat.</td></tr><tr><td>Cohere</td><td>33.8</td><td>88.2</td><td>50.0</td><td>98.8</td><td>68.2</td><td>99.8</td><td>78.7</td><td>100.0</td></tr><tr><td>MiniLM</td><td>17.7</td><td>89.5</td><td>35.5</td><td>98.3</td><td>46.3</td><td>99.8</td><td>58.7</td><td>100.0</td></tr><tr><td>GIST</td><td>24.0</td><td>84.7</td><td>46.0</td><td>98.7</td><td>60.8</td><td>99.8</td><td>72.3</td><td>100.0</td></tr></table>

Table 27: Oracle remainder and covariance audit (two-bit codes). “Oracle error” is the relative error of the incidence-covariance prediction against the measured role variance for ranking / pruning; coarse looseness is the coarse proxy divided by the empirical oracle role variance.
<table><tr><td>Dataset</td><td>Coordinates</td><td>Mag. flips</td><td>Remainder share</td><td>Oracle error (R / P)</td><td>Coarse loose.</td></tr><tr><td>Cohere</td><td>original</td><td>1.834%</td><td>4.66%</td><td>3.1% / 4.6%</td><td>11,420</td></tr><tr><td>Cohere</td><td>rotated</td><td>.313%</td><td>2.09%</td><td>0.7% / 2.1%</td><td>51,130</td></tr><tr><td>MiniLM</td><td>original</td><td>.509%</td><td>1.31%</td><td>0.3% / 1.0%</td><td>13,876</td></tr><tr><td>MiniLM</td><td>rotated</td><td>.516%</td><td>1.37%</td><td>0.9% / 1.5%</td><td>13,969</td></tr><tr><td>GIST</td><td>original</td><td>3.446%</td><td>100.73%</td><td>0.0% / 0.7%</td><td>19,016</td></tr><tr><td>GIST</td><td>rotated</td><td>.267%</td><td>2.30%</td><td>2.3% / 1.4%</td><td>77,414</td></tr></table>

2002.

Bradley Efron and Charles Stein. The jackknife estimate of variance. The Annals of Statistics, 9(3):586–596, 1981.

Jianyang Gao and Cheng Long. Rabitq: Quantizing high-dimensional vectors with a theoretical error bound for approximate nearest neighbor search. Proceedings of the ACM on Management of Data, 2(3):1–27, 2024. doi: 10.1145/3654970.

Tiezheng Ge, Kaiming He, Qifa Ke, and Jian Sun. Optimized product quantization. IEEE Transactions on Pattern Analysis and Machine Intelligence, 36(4):744–755, 2014.

Ruiqi Guo, Philip Sun, Erik Lindgren, Quan Geng, David Simcha, Felix Chern, and Sanjiv Kumar. Accelerating large-scale inference with anisotropic vector quantization. In International Conference on Machine Learning, pages 3887–3896, 2020.

Herv´e J´egou, Matthijs Douze, and Cordelia Schmid. Product quantization for nearest neighbor search. IEEE Transactions on Pattern Analysis and Machine Intelligence, 33 (1):117–128, 2011.

Yu A Malkov and D A Yashunin. Eficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. IEEE Transactions on Pattern Analysis and Machine Intelligence, 42(4):824–836, 2020.

Table 28: Looseness of the variance proxies over 28 dataset–coordinate–role configurations (seven datasets, two coordinate systems, two roles).
<table><tr><td>Proxy</td><td>Minimum</td><td>Median</td><td>Mean</td><td>Maximum</td></tr><tr><td>Coarse range</td><td>403</td><td>22,044</td><td>27,109</td><td>73,572</td></tr><tr><td>Componentwise Cauchy-Schwarz</td><td>5.91</td><td>40.84</td><td>47.59</td><td>93.13</td></tr><tr><td>Replacement (Efron-Stein)</td><td>1.21</td><td>1.67</td><td>1.72</td><td>2.04</td></tr></table>

Table 29: Five-pool selective certificate protocol at 512 certification blocks $\left( \delta = . 0 5 \right)$ . “No exceedance” counts configurations whose validation risk stayed below the frozen certificate.
<table><tr><td>Role</td><td>Issued</td><td>No exceedance</td><td>Retained</td><td>Validation risk</td><td>Certificate</td><td>Stress risk</td></tr><tr><td>Ranking</td><td>24</td><td>24</td><td>66.54%</td><td>.370%</td><td>19.76%</td><td>13.02%</td></tr><tr><td>Pruning</td><td>24</td><td>24</td><td>67.26%</td><td>.848%</td><td>22.75%</td><td>21.47%</td></tr></table>

Andreas Maurer and Massimiliano Pontil. Empirical bernstein bounds and sample variance penalization. In Conference on Learning Theory, pages 115–124, 2009.

Suhas Jayaram Subramanya, Fnu Devvrit, Harsha Vardhan Simhadri, Ravishankar Krishnaswamy, and Rohan Kadekodi. Diskann: Fast accurate billion-point nearest neighbor search on a single node. In Advances in Neural Information Processing Systems, volume 32, 2019.

Benjamin Trent. Better binary quantization (BBQ) in Lucene and Elasticsearch. Elastic Search Labs, 2024. URL https://www.elastic.co/search-labs/blog/ better-binary-quantization-lucene-elasticsearch. Accessed 2026-09-08.

Alexandre B. Tsybakov. Optimal aggregation of classifiers in statistical learning. The Annals of Statistics, 32(1):135–166, 2004.

Roman Vershynin. High-Dimensional Probability: An Introduction with Applications in Data Science. Cambridge University Press, 2018.

Wenxuan Xiao, Zhiyou Wang, and Chengcheng Li. Quiver: Rethinking ann graph topology via training-free binary quantization, 2026. URL https://arxiv.org/abs/2605.02171.

Table 30: Selective certificate comparison over prespecified sample sizes $\left( \delta = . 0 5 \right)$ . Each entry averages 24 dataset–quantizer configurations; each method optimizes its own policy. Validation coverage and risk are measured on 1,024 independent blocks. Entries are ranking / pruning.
<table><tr><td>Blocks</td><td>Method</td><td>Certificate</td><td>Validation coverage</td><td>Validation risk</td></tr><tr><td>128</td><td>two-event Hoeffding</td><td>40.22% 43.96%</td><td>76.8% 73.4%</td><td>0.99% 1.91%</td></tr><tr><td>128</td><td>direct union</td><td>29.08% 31.32%</td><td>70.4% 68.4%</td><td>0.45% 0.94%</td></tr><tr><td>128</td><td>empirical-Bernstein union</td><td>28.60% 29.52%</td><td>67.9% 68.4%</td><td>0.37% 0.94%</td></tr><tr><td>128</td><td>direct failure</td><td>16.29% 18.62%</td><td>95.1% 88.5%</td><td>2.18% 3.31%</td></tr><tr><td>256</td><td>two-event Hoeffding</td><td>28.58% 31.39%</td><td>70.4% 70.1%</td><td>0.45% 1.26%</td></tr><tr><td>256</td><td>direct union</td><td>22.26% 24.07%</td><td>66.2% 67.6%</td><td>0.35% 0.86%</td></tr><tr><td>256</td><td>empirical-Bernstein union</td><td>16.44% 17.64%</td><td>62.0% 61.0%</td><td>0.26% 0.75%</td></tr><tr><td>256</td><td>direct failure</td><td>12.18% 13.96%</td><td>93.5% 82.7%</td><td>1.95% 2.32%</td></tr><tr><td>512</td><td>two-event Hoeffding</td><td>20.59% 22.95%</td><td>67.0% 67.6%</td><td>0.42% 0.86%</td></tr><tr><td>512</td><td>direct union</td><td>16.74% 18.65%</td><td>64.6% 63.5%</td><td>0.33% 0.75%</td></tr><tr><td>512</td><td>empirical-Bernstein union</td><td>9.49% 10.91%</td><td>61.2% 56.8%</td><td>0.26% 0.72%</td></tr><tr><td>512</td><td>direct failure</td><td>9.07% 10.45%</td><td>87.8% 80.2%</td><td>1.34% 2.01%</td></tr><tr><td>1,024</td><td>two-event Hoeffding</td><td>15.13% 17.36%</td><td>63.7% 61.0%</td><td>0.32% 0.75%</td></tr><tr><td>1,024</td><td>direct union</td><td>12.74% 14.65%</td><td>61.2% 61.0%</td><td>0.26% 0.75%</td></tr><tr><td>1,024</td><td>empirical-Bernstein union</td><td>6.13% / 7.27%</td><td>57.7% 51.1%</td><td>0.22%/ 0.44%</td></tr><tr><td>1,024</td><td>direct failure</td><td>6.75% /7.99%</td><td>81.9% 75.2%</td><td>0.94% / 1.44%</td></tr></table>

Table 31: Pooled residual correlation across candidate-selection regimes.
<table><tr><td>Dataset</td><td>Scorer</td><td>i.i.d.</td><td>Distinct</td><td>Random</td><td>Top-256</td><td>Top-32</td><td>Anchor</td></tr><tr><td>Cohere</td><td>1-bit</td><td>.361</td><td>.364</td><td>.312</td><td>.524</td><td>.525</td><td>.544</td></tr><tr><td>Cohere</td><td>2-bit</td><td>.391</td><td>.393</td><td>.346</td><td>.571</td><td>.599</td><td>.613</td></tr><tr><td>Cohere</td><td>rotated 2-bit</td><td>.259</td><td>.259</td><td>.212</td><td>.324</td><td>.340</td><td>.364</td></tr><tr><td>MiniLM</td><td>2-bit</td><td>.010</td><td>.010</td><td>.020</td><td>.067</td><td>.111</td><td>.118</td></tr><tr><td>GIST</td><td>1-bit</td><td>.386</td><td>.387</td><td>.370</td><td>.925</td><td>.934</td><td>.870</td></tr><tr><td>GIST</td><td>2-bit</td><td>.405</td><td>.405</td><td>.422</td><td>.576</td><td>.640</td><td>.616</td></tr></table>

Table 32: Conditional-MGF and stable-clipping parameters over 24 role configurations (six datasets, two quantizers, two roles).
<table><tr><td>Quantity</td><td>Minimum</td><td>Median</td><td>Maximum</td></tr><tr><td>Untruncated  $ { b _ { \star } } / \mathrm { s d }$ </td><td>.133</td><td>.158</td><td>.324</td></tr><tr><td>Doob variance sum / total variance</td><td>1.047</td><td>1.086</td><td>1.102</td></tr><tr><td>Validation / calibration variance</td><td>.945</td><td>.994</td><td>1.075</td></tr><tr><td>Clipped  $ { b _ { \star } } / \mathrm { s d }$  (1% exceptional mass)</td><td>.130</td><td>.147</td><td>.258</td></tr></table>

Table 33: Matched-Spearman construction $( n = 5 0 0 , 0 0 0 )$ . The two perturbations share the exact scores and the same rank fidelity; mean squared error orders them the wrong way.
<table><tr><td>Perturbation</td><td>Spearman</td><td>MSE</td><td>Global flip</td><td>Hardest-20% flip</td></tr><tr><td>Light-tail Gaussian</td><td>.9860281863</td><td>.02514</td><td>5.09%</td><td>23.91%</td></tr><tr><td>Boundary-coupled</td><td>.9860281863</td><td>.01700</td><td>10.00%</td><td>50.00%</td></tr></table>