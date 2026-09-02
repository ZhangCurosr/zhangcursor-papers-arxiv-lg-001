# Scaled Idempotence in Transformer Attention: Paired OV Geometry and Shared-Value Algebras

Jiming Feng   
College of Computer Science, Beijing University of Technology   
Beijing, China

jming2011@emails.bjut.edu.cn

Junliang Li

School of Mathematics, Statistics and Mechanics, Beijing University of Technology Beijing, China

## Abstract

We identify a recurrent algebraic regularity in Transformer attention: a sparse subset of efective OV operators $T = O V ^ { \top }$ nearly closes under composition, $T ^ { 2 } \approx \alpha T . ^ { 1 }$ Across six pretrained endpoints spanning 2.8B–235B parameters, 3.98–8.00% of heads reach squared closure alignment $\mathcal { P } \geq 0 . 9 .$ , whereas no matched within-layer $\mathrm { O } / \mathrm { V }$ mismatch reaches this threshold. The tail is not numerically square-zero: over all 881 strong heads, the return gain $\| T ^ { 2 } \| _ { F } / \| T \| _ { F }$ has model medians 0.032–2.51 and a pooled minimum of 0.014. An exact factorization in principal coordinates,

$$
T = Q _ { \cal O } K Q _ { V } ^ { \top } , \qquad T ^ { 2 } = Q _ { \cal O } ( K D K ) Q _ { V } ^ { \top } ,
$$

separates within-support transport K from read–write return geometry D. A population-wide intervention over all 7,304 heads in nine MHA/GQA models scrambles only the orientation of K while preserving its singular spectrum, norm, read/write factor spans, and every principal angle; all 14,608 participating O/V factors have full numerical column rank. Median closure falls from 0.336 to $1 . 0 4 \times 1 0 ^ { - 4 }$ ; the trained orientation yields higher closure for 98.64% of heads and in all 268 layers. A complementary search, with heads sampled independently of closure, finds an explicit feasible construction above 0.8 among the sampled heads in every layer, including layers without a strong head, although most sampled high-capacity heads do not attain strong closure. Retrospective trajectories in three independently trained lineages further separate broadly available feasible geometry from the orientations attained by the final strong population. Under exact value sharing, headwise closure extends to a fixed-gain right-action algebra: $T _ { i } T _ { j } = \alpha _ { j } T _ { i }$ for every sibling i. Seven-model experiments verify the approximate law and reveal distinct oblique projections with a shared value-defined kernel. Together, these results characterize scaled idempotence as a sparse trained orientation within a broadly available geometric capacity and show that value sharing extends the headwise relation to a local operator algebra.

## 1 Introduction

For a generic low-rank map, applying it twice need not preserve the direction of applying it once. Yet a nontrivial minority of trained Transformer attention heads exhibit

$$
T ^ { 2 } \approx \alpha T , \qquad T = O V ^ { \top } ,\tag{1}
$$

where T is the head’s efective residual-stream OV operator. This scaled idempotence is visually and algebraically suggestive: the operator nearly closes under composition. It also poses a concrete mechanistic question. Is Equation 1 a chance consequence of low rank, an approximate projection or copying rule, or the visible endpoint of a relation learned between the separately parameterized value and output factors?

To distinguish these possibilities, we compare each canonical O/V pair with cyclic within-layer wrong-V pairings that preserve the two factor collections while breaking their trained correspondence. Strong scaled idempotence disappears under this control. We then decompose each operator into principal-angle geometry and a transport core, allowing the contribution of their head-specific pairing to be tested directly.

Our results attribute Equation 1 to a specific relation between separately parameterized O and V factors and establish an exact compositional consequence under value sharing. The relation does not require T to be an orthogonal projection, the read and write subspaces to coincide, or the head to copy tokens. We derive a small-matrix factorization that separates subspace support from within-subspace transport. Let the principal-angle cosines between the read and write spaces form a diagonal matrix $D ,$ and let K express the OV core in the corresponding principal coordinates. Then

$$
T = Q _ { \cal O } K Q _ { V } ^ { \top } , \qquad T ^ { 2 } = Q _ { \cal O } ( K D K ) Q _ { V } ^ { \top } .\tag{2}
$$

When $D \approx c I .$ , closure of $T$ is governed by the more general latent law $K ^ { 2 } \approx \beta K$ ; the transport core need not be close to a scalar matrix. More importantly, the decomposition makes the mechanism experimentally separable: support geometry and transport orientation can be intervened on while their coarse invariants remain fixed.

We study this phenomenon across model scale, attention topology, training stage, and matched geometric controls. Our contributions are:

• We identify a recurrent scaled-idempotent upper tail across pretrained Transformer endpoints from 2.8B to 235B parameters, show that it disappears under marginal-preserving $\mathrm { O } / \mathrm { V }$ mismatch, and derive its exact KDK return geometry.

• We isolate transport orientation without selecting heads by closure. Across all heads and layers in nine MHA/GQA models, closure collapses when the trained orientation is scrambled while spectrum, read/write spans, and principal angles remain fixed.

• We separate geometric capacity from trained attainment. High closure is constructively feasible in every surveyed layer, while retrospective trajectories associate the sparse final tail primarily with attained orientation rather than with scarce capacity.

• We prove that shared values turn exact headwise closure into a fixed-gain cross-head operator algebra. Seven-model tests confirm the approximate law, and an afine normal form characterizes distinct oblique projections with a common value-defined kernel.

Scaled idempotence is therefore a sparse signature of a paired $\mathrm { O } / \mathrm { V }$ relation, not a property visible in either factor alone.

## 2 Related Work

Mechanistic interpretability commonly separates QK routing from OV content transformation (Elhage et al., 2021); its V-composition scores already ask whether one component writes into directions read by another. Subsequent work has identified task-local OV circuits for induction, indirect-object identification, and algorithmic computation (Olsson et al., 2022; Wang et al., 2023; Nanda et al., 2023). The recent Communication Map extends operator-product analysis into a task-free census of component-to-component communication strength (Wang, 2026). Our object is diferent: whether one learned OV operator closes directionally under reapplication, and whether value sharing turns that self-relation into a fixed-gain cross-head law.

Our analysis uses principal angles and Grassmannian subspace similarity (Björck & Golub, 1973; Hamm & Lee, 2008). Recent work compares attention-head weight subspaces in $\mathrm { G P T - 2 }$ (Yamagiwa et al., 2026), while Chen et al. (2026) find projection-like OV routing in a one-layer modular-multiplication transformer. Principal-angle overlap discards the signed transport orientation retained by K. Our matched intervention holds the complete angle spectrum fixed while varying this additional degree of freedom.

Idempotence has also been imposed explicitly on neural networks (Jensen & Vicary, 2025), and scaledidempotent projection bases appear in theoretical analyses of in-context learning (Bu et al., 2024). In contrast, the relation studied here emerges without an idempotence objective in pretrained language-model OV weights, permits signed $\alpha ,$ and is measured by direct self-composition. GQA provides the architectura premise of shared key–value heads (Ainslie et al., 2023); we derive and test the resulting right-action law. The present work connects a recurrent scaled-idempotent OV relation to its KDK paired-return geometry and its algebraic extension under value sharing.

## 3 Scaled Idempotence and Its Paired-Return Decomposition

## 3.1 Efective OV operator

Ignoring token-to-token attention mixing, write the residual-stream map of head h as

$$
T _ { h } = O _ { h } V _ { g ( h ) } ^ { \top } , \qquad O _ { h } , V _ { g ( h ) } \in \mathbb { R } ^ { d \times r } ,\tag{3}
$$

where d is the model width, r the head dimension, and $g ( h )$ selects the value head. In MHA, each query head has its own value head; in $\mathrm { G Q A }$ , several query/output heads share one value factor. The factorization is gauge-dependent, but $T _ { h }$ is not.

The small cross-factor matrix

$$
M _ { h } = V _ { g ( h ) } ^ { \top } O _ { h }\tag{4}
$$

is the return core of the pairing: $T _ { h } ^ { 2 } = O _ { h } M _ { h } V _ { g ( h ) } ^ { \top }$ . It records how directions written through $O _ { h }$ are read back through the value factor. Our primary object is the relation represented by $M _ { h }$ and its basis-invariant support geometry, rather than either factor in isolation.

## 3.2 Scaled idempotence as a composite endpoint

We measure scaled idempotence as

$$
\mathcal { P } ( T ) = 1 - \frac { \operatorname* { m i n } _ { \alpha } \| T ^ { 2 } - \alpha T \| _ { F } ^ { 2 } } { \| T ^ { 2 } \| _ { F } ^ { 2 } }\tag{5}
$$

$$
= { \frac { \langle T ^ { 2 } , T \rangle _ { F } ^ { 2 } } { \| T ^ { 2 } \| _ { F } ^ { 2 } \| T \| _ { F } ^ { 2 } } } , \qquad \alpha ^ { * } = { \frac { \langle T ^ { 2 } , T \rangle _ { F } } { \| T \| _ { F } ^ { 2 } } } .\tag{6}
$$

Thus $\mathcal { P }$ is a squared cosine in operator space. It lies in $[ 0 , 1 ]$ and is invariant to uniform rescaling of T. All computations use $r \times r$ Gram matrices; no d × d operator need be materialized. We define the score only when both $T$ and $T ^ { 2 }$ are nonzero; every empirical operator in the reported analyses satisfies this condition. Because the square removes the sign of proportionality, $\mathcal { P }$ measures projective closure and permits either positive or negative fitted α.

## 3.3 Principal coordinates and the KDK identity

Take reduced QR factorizations $O = Q _ { O } R _ { O }$ and $V = Q _ { V } R _ { V }$ . Let

$$
Q _ { O } ^ { \top } Q _ { V } = A D B ^ { \top } , \qquad D = \operatorname { d i a g } ( \cos \theta _ { 1 } , \dots , \cos \theta _ { r } )\tag{7}
$$

be an SVD. Define $\widetilde { Q } _ { O } = Q _ { O } A , \widetilde { Q } _ { V } = Q _ { V } B$ , and

$$
K = A ^ { \top } R _ { O } R _ { V } ^ { \top } B .\tag{8}
$$

Then $\widetilde { Q } _ { V } ^ { \top } \widetilde { Q } _ { O } = D$ , giving the exact identities

$$
T = \widetilde Q _ { \cal O } K \widetilde Q _ { V } ^ { \top } , \qquad T ^ { 2 } = \widetilde Q _ { \cal O } ( K D K ) \widetilde Q _ { V } ^ { \top } .\tag{9}
$$

Computing Equation 6 from $( K , D )$ exactly reconstructs the direct low-rank score for all analyzed heads. We therefore write the same quantity in principal coordinates as

$$
\mathcal { P } _ { D } ( K ) = \frac { \langle K D K , K \rangle _ { F } ^ { 2 } } { \| K D K \| _ { F } ^ { 2 } \| K \| _ { F } ^ { 2 } } = \mathcal { P } ( T ) .\tag{10}
$$

Two support summaries will be useful. Their total overlap is

$$
G = \frac { \| D \| _ { F } ^ { 2 } } { r } = \frac { 1 } { r } \sum _ { i } \cos ^ { 2 } \theta _ { i } ,\tag{11}
$$

and their scale-free isotropy is

$$
I _ { D } = \frac { r \bar { c } ^ { 2 } } { \sum _ { i } c _ { i } ^ { 2 } } , \qquad \bar { c } = \frac { 1 } { r } \sum _ { i } c _ { i } .\tag{12}
$$

For independent random r-dimensional subspaces in $\mathbb { R } ^ { d } , \mathbb { E } [ G ] = r / d \colon$ neither G nor $I _ { D }$ records the signed orientation carried by K.

The latent transport law is measured directly by

$$
\beta _ { K } ^ { * } = \frac { \langle K ^ { 2 } , K \rangle _ { F } } { \| K \| _ { F } ^ { 2 } } , \qquad \mathcal { P } _ { I } ( K ) = \frac { \langle K ^ { 2 } , K \rangle _ { F } ^ { 2 } } { \| K ^ { 2 } \| _ { F } ^ { 2 } \| K \| _ { F } ^ { 2 } } .\tag{13}
$$

If $D = c I ,$ then $\mathcal { P } ( T ) = \mathcal { P } _ { I } ( K )$ and the optimal outer coeficient is $\alpha _ { T } ^ { * } = c \beta _ { K } ^ { * }$ . Thus isoclinic support exposes the general scaled-idempotent law $\begin{array} { r } { K ^ { 2 } \approx \beta K ; } \end{array}$ it does not require $K$ to resemble a scalar matrix. The coeficient c may be far below one, so a map between distinct isoclinic subspaces can close exactly without being an identity or orthogonal projection.

Non-isoclinic support admits an exact perturbation description. Assume K, $K ^ { 2 }$ , and $K D K$ are nonzero. Set $c = \mathrm { t r } ( D ) / r > 0$ , write $D = c I + E$ , and define $X = K ^ { 2 } , Y = K$ , and $F = K E K$ . Let $x = X / \| X \| _ { F }$ , set $\sigma = \mathrm { s i g n } \langle X , Y \rangle _ { F }$ (with $\sigma = 1$ when the inner product is zero), and define $y = \sigma Y / \| Y \| _ { F }$ and $s = \langle x , y \rangle _ { F } =$ $\sqrt { \mathcal { P } _ { I } ( K ) }$ . For $s < 1$ , let $u = ( y - s x ) / \sqrt { 1 - s ^ { 2 } }$ and decompose

$$
{ \frac { F } { c \| X \| _ { F } } } = a x + b u + w , \qquad w \perp x , u .\tag{14}
$$

Direct substitution gives

$$
\mathcal { P } _ { D } ( K ) = \frac { \left[ s ( 1 + a ) + \sqrt { 1 - s ^ { 2 } } b \right] ^ { 2 } } { ( 1 + a ) ^ { 2 } + b ^ { 2 } + \| w \| _ { F } ^ { 2 } } .\tag{15}
$$

The radial term a cancels when acting alone, b rotates within the closure-relevant plane, and w enters only the denominator as pure closure leakage. When $s = 1$ , the corresponding one-dimensional decomposition gives $( 1 + a ) ^ { 2 } / [ ( 1 + a ) ^ { 2 } + \| w \| _ { F } ^ { 2 } ]$ whenever KDK $\neq 0$ . Bounds and numerical certificates derived from this identity are reported in the supplement.

The same law also has a direct factor characterization. With $M = V ^ { \top } O$

$$
T ^ { 2 } - \alpha T = O ( M - \alpha I ) V ^ { \top } , \qquad T ^ { 2 } = \alpha T \Longleftrightarrow V ^ { \top } O = \alpha I ,\tag{16}
$$

where the equivalence holds for full-column-rank factors. Under the gauge $O \mapsto O G , V \mapsto V G ^ { - \top }$ , M changes by similarity, so the scalar-identity condition is gauge invariant.

Assuming full-column-rank $O , V$ , the approximate statement is exact in the factor-induced metric. Let $A = O ^ { \top } \bar { O } , B = V ^ { \top } V$ , and $\langle X , Y \rangle _ { A , B } = \mathrm { t r } ( X ^ { \top } A Y B )$ . Then

$$
\langle O X V ^ { \top } , O Y V ^ { \top } \rangle _ { F } = \langle X , Y \rangle _ { A , B } , \qquad \mathcal { P } ( T ) = \frac { \langle M , I \rangle _ { A , B } ^ { 2 } } { \| M \| _ { A , B } ^ { 2 } \| I \| _ { A , B } ^ { 2 } } .\tag{17}
$$

The fitted coeficient is $\alpha ^ { * } = \langle M , I \rangle _ { A , B } / \| I \| _ { A , B } ^ { 2 }$ . Thus $T ^ { 2 }$ ≈ αT is equivalent to weighted projective scalarity of the paired return core $M$ . Ordinary unweighted Frobenius scalarity additionally depends on factor conditioning and is not implied uniformly.

Proposition 1 (Shared-value operator algebra). Let $V \in \mathbb { R } ^ { d \times r }$ have full column rank and define $C _ { V } =$ $V ( \bar { V ^ { \top } } V ) ^ { - 1 }$ . The family

$$
\begin{array} { r } { \mathcal { A } _ { V } = \left\{ X _ { a , N } = ( a C _ { V } + N ) V ^ { \top } : a \in \mathbb { R } , V ^ { \top } N = 0 \right\} } \end{array}\tag{18}
$$

is a linear operator algebra satisfying

$$
X _ { a , N } X _ { b , M } = b X _ { a , N } .\tag{19}
$$

Consequently $X _ { a , N } ^ { 2 } = a X _ { a , N }$ . Conversely, if $T = O V ^ { \top }$ with O, V full column rank, then $T ^ { 2 } = a T$ if and only if $T = X _ { a , N }$ for some N with $V ^ { \top } N = 0$

The product follows immediately from $V ^ { \top } C _ { V } = I$ and $V ^ { \top } N = 0$ . For $a = 1$ , the normalized slice consists of oblique projections $P _ { N } = ( C _ { V } + N ) V ^ { \top }$ with common kernel ker $( V ^ { \top } )$ and $P _ { N } P _ { M } = P _ { N }$ . When $r < d , A _ { V }$ is nonunital: value sharing defines a local algebra rather than a copy of the full residual-stream algebra.

The exact law also controls approximation. For shared-V heads $T _ { i } = O _ { i } V ^ { \top }$ and $T _ { j } = O _ { j } V ^ { \top }$ , let $A _ { i } = O _ { i } ^ { \top } O _ { i }$ $A _ { j } = O _ { j } ^ { \top } O _ { j }$ , and $\gamma _ { i j } = \lambda _ { \operatorname* { m a x } } ( A _ { j } ^ { - 1 / 2 } A _ { i } A _ { j } ^ { - 1 / 2 } )$ . If $O _ { j }$ has full column rank, then for every $\alpha .$

$$
\begin{array} { r } { \| T _ { i } T _ { j } - \alpha T _ { i } \| _ { F } ^ { 2 } \leq \gamma _ { i j } \| T _ { j } ^ { 2 } - \alpha T _ { j } \| _ { F } ^ { 2 } . } \end{array}\tag{20}
$$

Thus exact closure of the right head propagates its coeficient to every sibling, while the approximate law can be amplified by relative output-factor conditioning and must be measured empirically. A proof of Equation 20 is given in the supplement.

## 4 Experimental Design

Each experiment addresses a distinct question and therefore uses a dedicated model population, summarized in Table 1. The six-endpoint survey provides broad scale coverage; the nine-model analysis uses complete factors for matched interventions; and the remaining experiments examine capacity, endpoint geometry, training trajectories, and the algebraic consequences of value sharing.

Table 1: Study map. Each population has one primary inferential role.
<table><tr><td>Question</td><td>Population</td><td>Primary comparison</td></tr><tr><td>Does closure recur?</td><td>6 endpoints; 15,936 all heads</td><td>canonical versus wrong-V</td></tr><tr><td>What degree of freedom carries it?</td><td>9 models; 7,304 all heads</td><td>fixed-invariant orientation scramble</td></tr><tr><td>Why is the upper tail sparse?</td><td>268 layers plus 3 trajectories</td><td>feasible capacity versus attained orientation</td></tr><tr><td>What geometry survives normalization?</td><td>12 endpoints; 1,180 strong heads</td><td>latent core versus trained support</td></tr><tr><td>What does value sharing imply?</td><td>7 GQA models; 1,458 products</td><td>fixed-gain sibling composition</td></tr></table>

## 4.1 Cross-model survey without head selection

We scan every attention head in six fully trained endpoints: Pythia-2.8B-deduped (Biderman et al., 2023), Qwen3-4B Base (Yang et al., 2025), Mistral-7B-v0.3 (Jiang et al., 2023), OLMo-2-13B (Team OLMo et al., 2025), Qwen2.5-72B Base (Qwen Team, 2024), and Qwen3-235B-A22B. This covers 15,936 heads, MHA and GQA, dense and MoE architectures. For each head, a cyclic wrong-V control preserves the layerwise factor collections while breaking trained identity.

All quantities are evaluated through exact $r \times r$ identities in float64 and checked against direct calculations. We use $\mathcal { P } \geq 0 . 9$ as a fixed descriptive convention and sweep thresholds from 0.75 to 0.975 over all six populations. Because $1 - \mathcal { P }$ is the minimum normalized squared residual after fitting $T ^ { 2 }$ ≈ αT, the threshold corresponds to a relative residual of at most $\sqrt { 0 . 1 } \approx 0 . 3 1 6$

## 4.2 Factorial test of the KDK mechanism

The factorial experiment uses every head in six independent families: Pythia-2.8B-deduped and OLMo-1B (Groeneveld et al., 2024) for MHA; TinyLlama-1.1B (Zhang et al., 2024), SmolLM3-3B (Bakouch et al., 2025), Qwen3-4B Base, and Mistral-7B for GQA. Within each layer, we exhaustively reassign K and D separately. GQA donors move by whole KV groups, changing the value factor while holding the query slot fixed. The primary endpoint is the layer mean of

$$
I _ { D K } = \mathcal { P } _ { D _ { h } } ( K _ { h } ) - \mathcal { P } _ { D _ { h } } ( K _ { h ^ { \prime } } ) - \mathcal { P } _ { D _ { h ^ { \prime \prime } } } ( K _ { h } ) + \mathcal { P } _ { D _ { h ^ { \prime \prime } } } ( K _ { h ^ { \prime } } ) .\tag{21}
$$

Layers are the inferential units. As a matched control, every $O _ { h }$ receives a wrong V from another head or KV group; $K , D$ are recomputed before repeating the factorial test. The complete correction and admission protocol is given in the supplement.

## 4.3 Spectrum- and support-preserving intervention

The factorial test changes the donor core and therefore does not hold its singular spectrum fixed. We construct a stricter counterfactual for every head. In principal coordinates, replace

$$
K \longmapsto K ^ { \prime } = P _ { L } K P _ { R } ^ { \top } ,\tag{22}
$$

where $P _ { L } , P _ { R }$ are independently sampled signed permutation matrices. Because they are orthogonal, $K ^ { \prime }$ has exactly the same singular values as K. Keeping $Q _ { O } , Q _ { V }$ , and D fixed also preserves rank, every unitarily invariant norm, the read/write factor spans, and their complete principal-angle spectrum. An all-head audit confirms full numerical column rank for all 14,608 participating $\mathrm { O } / \mathrm { V }$ factors under the tolerance max $\iota ( m , n ) \epsilon _ { 6 4 } s _ { \mathrm { { m a x } } } ;$ the smallest observed $s _ { \operatorname* { m i n } } / s _ { \operatorname* { m a x } }$ is $2 . 6 3 \times 1 0 ^ { - 4 }$ . For GQA, we refactor $Q _ { O } K ^ { \prime } Q _ { V } ^ { \top } = O ^ { \prime } V ^ { \top }$ using the original shared value factor, so group topology is unchanged. Eight deterministic interventions are applied to all 7,304 heads across seven GQA and two MHA families.

Repeated or nearly repeated principal angles make individual principal vectors non-unique. We therefore add an audit whose null distribution is basis invariant: $P _ { L }$ and $P _ { R }$ are replaced by independent Haar orthogonal matrices. Haar measure is unchanged by any orthogonal reparameterization within a degenerate principal-angle block. Two deterministic draws are evaluated per head, while the complete singular spectrum and support geometry remain fixed.

To separate geometric capacity from trained attainment, we hold the same invariants fixed and consider

$$
\mathrm { C a p } ( \sigma , D ) = \operatorname* { m a x } _ { U _ { L } , U _ { R } \in \mathsf { O } ( r ) } \mathcal { P } _ { D } \bigl ( U _ { L } \mathrm { d i a g } ( \sigma ) U _ { R } ^ { \top } \bigr ) .\tag{23}
$$

Here σ is the trained singular spectrum. Projected ascent with exact SVD retraction provides a constructive lower bound ${ \widehat { \mathrm { C a p } } } \leq \mathrm { C a p }$ . We evaluate all 510 strong heads and deterministic layer-matched ordinary controls. A complementary sample selects two heads per layer by a fixed hash rule, independently of closure, across all 268 layers, including those without a strong head. Model- and layer-level conclusions resample models and complete layers; numerical-route and rank-one boundary audits are reported in the supplement.

## 4.4 Supporting natural-activation diagnostic

To test whether the weight-space law survives the anisotropic distribution of real hidden states, we use four deterministic 256-token WikiText segments in OLMo-1B, TinyLlama-1.1B, SmolLM3-3B, and Qwen3-4B-Instruct. The high-closure decile is paired with an equal number of low-closure heads from the same eligible layers. Writing activations as columns, if $y$ is a head’s natural post-O output, the endpoint is the squared directional alignment of y with its diagnostic return $\boldsymbol { T } \boldsymbol { y } = \boldsymbol { O } \boldsymbol { V } ^ { \intercal } \boldsymbol { y }$ . We also record the RMS return magnitude $\begin{array} { r } { g _ { y } = ( \sum \lVert T y \rVert _ { 2 } ^ { \bar { 2 } } / \sum \lVert y \rVert _ { 2 } ^ { 2 } ) ^ { \bar { 1 } / 2 } } \end{array}$ . A cyclic wrong-V pairing holds y and O fixed while breaking the trained return factor. The fixed population contains 271 high/low pairs (542 heads) across 86 eligible layers; this is a data-weighted consistency check, not a forward-circuit intervention.

## 4.5 Latent-core and support decomposition

Since $\mathcal { P } _ { \lambda D } ( K ) = \mathcal { P } _ { D } ( K )$ , total support magnitude cannot directly change closure. We replace trained support by $D _ { \mathrm { i s o } } = ( \| D \| _ { F } / \sqrt { r } ) I$ , preserving K and total overlap while exposing the latent score ${ \mathcal { P } } _ { I } ( K )$ . We apply this normalization to 1,180 strong heads across twelve endpoints and test specificity against all 7,304 local heads plus layer-matched ordinary heads from the three large endpoints.

Conditional on latent closure, $I _ { D }$ tests whether support shape preserves the core law. Equation 15 then separates radial response, in-plane rotation, and of-plane leakage. The main text treats this analysis as a decomposition of endpoint geometry rather than a causal account of formation; threshold-sweep and clusteredinference summaries, axis-assignment controls, and perturbation bounds are reported in the supplement.

## 4.6 Capacity and attainment across training

We track fixed final strong/ordinary populations through independently pretrained OLMo-1B, TinyLlama-1.1B, and Pythia-2.8B-deduped trajectories. At four representative checkpoints per lineage, we recompute feasible capacity under the same spectrum/support constraints. If R is the median matched-scramble closure, the constructed-range attainment index is

$$
E = \frac { \mathcal { P } - R } { \widehat { \mathrm { C a p } } - R } .\tag{24}
$$

Values are not clipped. Because the denominator uses a constructive lower bound rather than a certified global optimum, E is a diagnostic index, not a certified fraction of total capacity. This retrospective comparison asks whether training creates a scarce feasible set or increasingly realizes a compatible orientation within an already capable set; it does not identify the examples or gradients that select final membership.

## 4.7 Shared-value composition and gain-inheritance tests

Proposition 1 predicts that an exactly closed right head acts with its own coeficient on every sibling sharing $V _ { g }$ We test the approximate law in seven independently trained GQA families: TinyLlama-1.1B, SmolLM3-3B, Qwen3-4B Base, Mistral-7B, Granite-3.3-2B Base (IBM Granite Team, 2025), Falcon3-3B Base (Technology Innovation Institute, 2024), and Yi-1.5-6B Base (01.AI, 2024). Right heads satisfy the common threshold $\mathcal { P } ( T _ { j } ) \ge 0 . 9$ , and every distinct sibling $i \neq j$ is evaluated. The matched control uses the same query slot in the next KV group, preserving layer and architecture while breaking shared-V identity.

Direction alone permits a diferent coeficient for every product, so we also test whether the right head transmits its own gain. For each pair, define

$$
F _ { i j } ( \alpha _ { j } ) = 1 - \frac { \| T _ { i } T _ { j } - \alpha _ { j } T _ { i } \| _ { F } ^ { 2 } } { \| T _ { i } T _ { j } \| _ { F } ^ { 2 } } , \quad \beta _ { i j } = \frac { \langle T _ { i } T _ { j } , T _ { i } \rangle _ { F } } { \| T _ { i } \| _ { F } ^ { 2 } } , \quad \rho _ { i j } = \frac { \beta _ { i j } } { \alpha _ { j } } .\tag{25}
$$

Here $\alpha _ { j }$ is fitted once from $T _ { j } ^ { 2 } \approx \alpha _ { j } T _ { j }$ and is not refitted to a sibling. After fixing the coeficient and control definitions on the development families, we evaluate Yi-1.5-6B as an additional-model confirmation. We also test whether normalized strong siblings $P _ { h } = T _ { h } / \alpha _ { h }$ are distinct operators satisfying $P _ { i } P _ { j } \approx P _ { i }$ rather than duplicate maps.

For the afine-normal-form prediction, we compare each normalized strong head with its canonical valuedefined anchor and a cyclic wrong-V anchor, then rank every compatible value factor in the same layer for seven GQA and two MHA models with complete candidate sets. OLMo-2-13B, Qwen2.5-72B Base, and Qwen3-235B-A22B provide an additional-model extension to larger scales.

## 5 Results

## 5.1 Scaled idempotence recurs from 2.8B to 235B parameters

Under a common closure statistic, threshold, and marginal-preserving wrong-V control, all six fully trained endpoints contain a continuous upper tail with $\mathcal { P } \geq 0 . 9$ (Table 2). The tail contains 3.98–8.00% of heads,

or 67–338 heads per model. No wrong-V control reaches the threshold, and control medians range from $5 . 7 \times 1 0 ^ { - 5 } \ \mathrm { t o } \ 1 . 3 \times 1 0 ^ { - 4 }$ . Because both canonical and mismatched maps have rank at most r, low rank alone cannot account for the separation.

The tail persists throughout the fixed 0.75–0.975 threshold sweep, whereas no mismatch reaches even the lowest tested threshold. The marginal-preserving wrong-V control supplies the pairing-specific comparison used uniformly across all six endpoints.

Across the 13B–235B endpoints, 69–76% of high-closure heads have negative fitted α. The recurring structure is therefore a signed return relation rather than only positive “copy and amplify.”

Table 2: Unified scaled-idempotence survey. All endpoints use the same score P and strong-head threshold $\mathcal { P } \geq 0 . 9$ . “Wrong median” uses the same cyclic within-layer wrong-V control for every endpoint.
<table><tr><td>Model</td><td>Topology</td><td>Heads</td><td>Median  $\mathcal { P }$ </td><td> ${ \mathrm { q } } 9 0$ </td><td>Max P</td><td> $\geq 0 . 9$ </td><td>Wrong median</td></tr><tr><td>Pythia-2.8B-deduped</td><td>MHA</td><td>1024</td><td>.4869</td><td>.8827</td><td>.9887</td><td>72 (7.03%)</td><td> $1 . 3 7 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Qwen3-4B Base</td><td>GQA</td><td>1152</td><td>.2752</td><td>.8425</td><td>.9799</td><td>67 (5.82%)</td><td> $6 . 5 3 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Mistral-7B-v0.3</td><td>GQA</td><td>1024</td><td>.3236</td><td>.8633</td><td>.9855</td><td>72 (7.03%)</td><td> $4 . 7 7 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>OLMo-2-13B, 5T</td><td>MHA</td><td>1600</td><td>.5251</td><td>.8827</td><td>.9854</td><td>128 (8.00%)</td><td> $5 . 7 0 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Qwen2.5-72B Base</td><td>GQA</td><td>5120</td><td>.2677</td><td>.8036</td><td>.9825</td><td>204 (3.98%)</td><td> $7 . 1 4 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Qwen3-235B-A22B</td><td>GQA</td><td>6016</td><td>.2794</td><td>.8255</td><td>.9894</td><td>338 (5.62%)</td><td> $1 . 0 8 \times 1 0 ^ { - 4 }$ </td></tr></table>

Because P is scale-free, we assess return magnitude separately. For $\mathcal { P } ( T ) > 0 ;$ , define

$$
g _ { T } = \frac { \Vert T ^ { 2 } \Vert _ { F } } { \Vert T \Vert _ { F } } = \frac { \vert \alpha ^ { * } \vert } { \sqrt { \mathcal { P } ( T ) } } .\tag{26}
$$

Across the 881 strong heads in Table 2, model-median g<sub>T</sub> ranges from 0.0317 to 2.512, its pooled minimum is 0.0144, and no strong head has $g _ { T } < 0 . 0 1$ . Moreover, 880/881 strong heads exceed the model-specific all-head median return gain. The high-closure tail is therefore not driven by numerically square-zero maps; a modelwise quantile summary is reported in the supplement.

## 5.2 Closure requires the head-specific pairing of D and K

Exhaustive reassignment gives a positive head-specific D×K interaction in all six families and 95.5–100% of their layers. When K, D are recomputed after wrong-V pairing, the interaction collapses to approximately zero and the real-minus-control contrast is positive in 96.9–100% of layers. This establishes pair specificity; the stricter intervention below isolates the responsible degree of freedom while fixing spectrum and support.

## 5.3 Closure depends on within-support transport orientation

Across all 7,304 heads, pooled median closure is 0.336 in the trained orientation and $1 . 0 4 \times 1 0 ^ { - 4 }$ after scrambling. The trained orientation yields higher closure for 7,205/7,304 heads and a positive median diference in all 268 layers. The efect persists after excluding the 510 heads with $\mathcal { P } \geq 0 . 9$ : 98.54% of the remaining 6,794 heads still favor the trained orientation. None of the 58,432 matched counterfactuals reaches 0.85. Thus rank, energy, singular spectrum, read/write spans, and principal angles do not jointly determine closure. The largest counterfactual occurs in a near-rank-one head, consistent with the algebraic boundary that every non-nilpotent rank-one map closes automatically.

The basis-invariant Haar null gives the same population-level conclusion. The pooled Haar median is $1 . 6 7 \times 1 0 ^ { - 4 }$ , and trained closure exceeds the median of two Haar reorientations for 98.38% of heads. The median paired diference is positive in all nine models and all 268 layers. Singular values are preserved to relative error below $1 0 ^ { - 1 2 }$ . The orientation efect is therefore not an artifact of choosing a particular basis inside the principal subspaces.

![](images/9ef746963ae3ae5b342229231459cd86f1d1be037915a5104093da408f201d48.jpg)

![](images/831aae825e65c2bc9c5542bf2002a3fa672c79877270010cf7b70b81671ae9f0.jpg)  
Figure 1: A population-wide, spectrum- and support-preserving intervention removes closure throughout the head population. (A) Model medians and interquartile ranges for trained heads and the per-head median of eight matched transport-core scrambles. (B) Trained and fixed-first-scramble tail counts over all 7,304 heads; none of the 58,432 counterfactuals reaches 0.85.

Constrained reorientation gives the complementary result. Across all nine models, the constructed feasible median is 0.988–0.994 for strong heads and 0.880–0.926 for layer-matched ordinary heads. The achieved lower-bound gap is only 11.5–20.8% as large as the observed closure gap. Strong heads therefore realize a substantially larger fraction of the explicitly constructed feasible range; numerical-route and invariantpreservation audits support the calculation.

A sample chosen without reference to closure yields model-median feasible capacity of 0.884–0.927. All 268 layer medians exceed 0.8, including the 82 layers that contain no strong head. Among 291 sampled heads with feasible capacity of at least 0.9, 87.3% nevertheless remain below the strong-head threshold. High capacity is therefore widespread but insuficient to explain which heads attain closure.

The retrospective training trajectories show the same constructed-capacity–attainment separation (Table 3). OLMo and Pythia begin at initialization with nearly identical constructed capacity in the two endpoint-defined groups and no attainment-index gap; TinyLlama’s first public checkpoint occurs after 10B training tokens. At the final checkpoint, the attainment-index gap is positive in all three models and all 54 eligible layers, with a hierarchical-bootstrap 95% interval of 0.468–0.643. In every lineage, the final achieved lower-bound gap is less than half the observed closure gap, and the attainment gap grows more than that lower-bound gap. These trajectories establish a robust retrospective separation, not that the final labels or their causal origin were known at early checkpoints.

Table 3: Constructed capacity and attainment across three training trajectories. Early columns give median feasible lower bounds for future strong (S) and matched ordinary (O) heads. Final columns are betweenpopulation median gaps. TinyLlama begins at 10B tokens; the other two begin at initialization.
<table><tr><td>Model</td><td>Early  ${ \widehat { \mathrm { C a p } } } _ { S }$ </td><td>Early  $\widehat { \mathrm { C a p } _ { O } }$ </td><td>Final  $\Delta \mathcal { P }$ </td><td>Final  $\Delta \widehat { \mathrm { C a p } }$ </td><td>Final  $\Delta E$ </td></tr><tr><td>OLMo-1B</td><td>0.839</td><td>0.839</td><td>0.412</td><td>0.083</td><td>0.353</td></tr><tr><td>TinyLlama-1.1B</td><td>0.894</td><td>0.899</td><td>0.536</td><td>0.111</td><td>0.488</td></tr><tr><td>Pythia-2.8B-deduped</td><td>0.806</td><td>0.805</td><td>0.368</td><td>0.069</td><td>0.321</td></tr></table>

![](images/c9f19bf2f75b70785b1b0e336683920bb668d4231445351cf50e6f2032396c8e.jpg)  
Figure 2: Complete saved-checkpoint trajectories for the endpoint-defined populations. Each curve is the future-strong minus layer-matched-ordinary median gap in observed closure $\mathcal { P } _ { \cdot }$ , constructive capacity $\widehat { \mathrm { C a p } }$ , or constructed-range attainment index E. Horizontal spacing indexes the four saved checkpoints rather than elapsed training time; TinyLlama begins at its first public 10B-token checkpoint. The lower-bound gap remains smaller while the attained-orientation gap carries most of the later separation.

These are constructive capacity–attainment separations, not claims that the numerical search finds the global optimum.

## 5.4 Supporting check on natural activations

Across OLMo-1B, TinyLlama-1.1B, SmolLM3-3B, and Qwen3-4B-Instruct, the paired return preserves the direction of natural outputs for high-closure heads (model medians 0.842–0.886), whereas wrong-V returns are below $1 0 ^ { - 3 }$ and layer-matched low-closure heads are near 0.012. Both contrasts are positive in all 86 eligible layers. The corresponding median RMS return magnitudes are 0.088, 2.071, 0.412, and 1.239, compared with 0.038, 0.867, 0.146, and 0.432 for layer-matched low-closure heads. Thus the directional result is not produced by vanishing returns on the sampled activations. This remains a data-weighted consistency check rather than evidence that the forward pass executes or behaviorally requires a second application.

## 5.5 Scale-free endpoint geometry of the strong tail

Isoclinic normalization exposes the general core law from Equation 13. Across twelve endpoints, 1,088/1,180 strong heads remain above 0.9 after replacing trained support by $D _ { \mathrm { i s o } } ;$ ; the pooled latent score is 0.957. This is not a generic low-rank efect: over all 7,304 local heads, latent closure separates the trained strong tail from ordinary heads with pooled AUROC 0.984, and a layer-matched 13B–235B additional-model extension gives the same pooled AUROC.

Support shape explains much of the remaining variation. Conditional on latent closure, $I _ { D }$ distinguishes cores whose closure is preserved from those whose closure is reduced, with pooled AUROC 0.945 in the local models and 0.963 in the large-endpoint extension. Both separations remain strong throughout the common 0.80–0.95 threshold sweep and under model/layer-clustered inference. This yields an endpoint decomposition in which most strong heads contain a latent $K ^ { 2 } \approx \beta K$ core, while trained support determines how faithfully that relation is preserved in KDK.

Equation 15 identifies the corresponding mechanism. Cores with reduced closure have larger nonradial KEK responses in every local family; of-plane leakage accounts for most of the reduction and can only decrease closure. For a smaller subset, closure also depends on how the same principal-angle spectrum is assigned to transport axes. The perturbation bound and axis-permutation controls are reported in the supplement.

## 5.6 Shared values extend self-closure into fixed-gain right action

Proposition 1 predicts that the headwise closure law extends to siblings sharing the same value factor. We evaluate this prediction for 406 strong right heads and 1,458 sibling compositions (Table 4). Without refitting the coeficient, the right head’s own $\alpha _ { j }$ explains 0.913–0.936 of median sibling-product energy. Freely fitted sibling gains satisfy $\rho _ { i j } = \beta _ { i j } / \alpha _ { j }$ with model medians of 0.959–0.974, whereas matched cross-value controls retain only 0.013–0.053 of that gain.

The additional Yi-1.5-6B evaluation reproduces the separation after the coeficient and control definitions were fixed: all 20 informative layers favor the shared-value law, with median fixed-gain explanation of 0.922 and a gain ratio of 0.963. Thus both direction and signed magnitude are inherited through the shared value channel.

Table 4: Gain-preserving shared-V algebra. $F ( \alpha _ { j } )$ fixes the coeficient to the right head’s self-fitted gain; $\rho = \beta _ { i j } / \alpha _ { j }$ reports the freely fitted coeficient relative to that gain. Control uses the same query slot in the next KV group. <sup>∗</sup>Yi-1.5-6B was evaluated after the coeficient and control definitions were fixed.
<table><tr><td>Model</td><td>Right heads</td><td>Pairs</td><td>Layers</td><td> $F ( \alpha _ { j } )$ </td><td>Same  $\rho$ </td><td>Control  $\rho$ </td></tr><tr><td>TinyLlama-1.1B</td><td>32</td><td>224</td><td>14</td><td>.932</td><td>.974</td><td>.013</td></tr><tr><td>SmolLM3-3B</td><td>35</td><td>105</td><td>18</td><td>.915</td><td>.966</td><td>.019</td></tr><tr><td>Qwen3-4B Base</td><td>67</td><td>201</td><td>21</td><td>.928</td><td>.964</td><td>.027</td></tr><tr><td>Mistral-7B</td><td>72</td><td>216</td><td>24</td><td>.934</td><td>.970</td><td>.014</td></tr><tr><td>Granite-3.3-2B</td><td>142</td><td>426</td><td>34</td><td>.936</td><td>.966</td><td>.017</td></tr><tr><td>Falcon3-3B</td><td>24</td><td>48</td><td>15</td><td>.913</td><td>.959</td><td>.053</td></tr><tr><td>Yi-1.5-6B*</td><td>34</td><td>238</td><td>20</td><td>.922</td><td>.963</td><td>.022</td></tr></table>

The operators remain distinct despite obeying a common product law. For normalized strong heads $P _ { h } = T _ { h } / \alpha _ { h }$ , 157 unordered sibling pairs (314 directed products) satisfy the approximate left-zero-band relation $P _ { i } P _ { j } \approx P _ { i }$ and $P _ { j } P _ { i } \approx P _ { j }$ . In the five families with at least ten such pairs, median operator squared cosine is 0.315–0.486, whereas median fixed-scale product explanation is 0.935–0.952. At least 98.6% of pairs remain below 0.8 squared cosine.

Equation 18 explains this nonduplication. For each strong GQA head, we compare $P _ { h }$ with the canonical anchor $P _ { V } = { V } ( V ^ { \top } V ) ^ { - 1 } V ^ { \top }$ . Across all seven families, the model-median squared cosine is 0.449–0.639, while 0.373–0.553 of operator energy lies in the free oblique component. Every strong head favors its trained anchor over wrong-V, and exhaustive same-layer retrieval ranks the trained V first for all 510 strong heads in the nine locally complete models. The empirical band is therefore the approximate counterpart of $\mathbf { \mathcal { A } } _ { V }$ : a common value-defined kernel coexists with distinct write-side degrees of freedom.

The additional-model scale extension confirms the anchor geometry for all 670 strong heads in OLMo-2-13B, Qwen2.5-72B Base, and Qwen3-235B-A22B. Together with the seven-family analysis, the result covers 1,076 strong heads across ten endpoints from 1.1B to 235B parameters. MHA confirmation shows that the anchor is intrinsic to paired O/V organization, while GQA sharing supplies the cross-head product law.

## 6 Discussion

## 6.1 What $T ^ { 2 }$ ≈ αT encodes

The KDK decomposition and matched orientation intervention show that scaled idempotence is carried by a specific relation between separately parameterized but jointly trained read and write factors. Spectrum, rank, support, and principal angles can all remain fixed while closure disappears. The resulting object is an oriented return geometry rather than a generic consequence of low rank or a token-copying map. The negative fitted coeficient found in 69–76% of large-endpoint strong heads further supports this signed interpretation, while the return-gain analysis excludes a numerically square-zero explanation.

This geometry is widely available but sparsely attained. Heads sampled independently of closure can be reoriented toward high closure in every surveyed layer, including layers in which the trained endpoint contains no strong head. In the three retrospective trajectories, the endpoint-defined groups separate primarily in attained orientation rather than feasible capacity. The sparse upper tail is therefore associated with the orientation present at the trained endpoint inside a large feasible set. Our experiments characterize that separation geometrically; identifying the data, gradients, and downstream objectives that produce it remains an open question.

The latent-core decomposition provides a complementary view. Most strong heads satisfy $K ^ { 2 } \approx \beta K$ after isoclinic normalization. Trained support then determines how this core appears in KDK: radial response preserves projective closure, whereas nonradial response rotates the operator or leaks energy away from its closure line. This decomposition describes the endpoint geometry of the observed weights but does not by itself identify the training process that produced them.

Value sharing turns the headwise relation into an algebraic one. Proposition 1 shows that an exactly closed right head transfers its coeficient to every sibling with the same value factor. The algebra permits distinct oblique projections with a common value-defined kernel, so the heads need not be duplicates. Seven-model fixed-gain tests and the ten-endpoint anchor analysis place pretrained weights near this exact solution family. In $\mathrm { G Q A }$ , shared values therefore define the domain on which a headwise self-relation becomes a cross-head product law.

## 6.2 Limitations and future work

The present conclusions are structural. The natural-activation analysis shows that the diagnostic return survives sampled hidden-state covariance, but does not establish that the forward pass explicitly composes these operators or that language-model predictions depend on closure. The three trajectories use endpointconditioned groups and describe capacity and attainment rather than the causal dynamics of optimization. The cross-head product law requires exact value sharing and does not directly extend to MHA. Establishing functional relevance will require interventions in the full context-conditioned path that control first-pass activation changes, QK feedback, normalization, and surrounding computation.

## Broader Impact Statement

Understanding recurrent internal transport geometry may support model auditing, debugging, and controlled modification. The same insight could enable more targeted manipulation of learned behavior; applications should therefore distinguish structural diagnostics from claims about semantics or capability.

## 7 Conclusion

Transformer attention repeatedly contains OV operators that nearly satisfy $T ^ { 2 } \approx \alpha T$ . Their closure is explained by an exact KDK return geometry and depends on trained transport orientation, not on low rank, spectrum, support, principal angles, or a near-zero return alone. High closure is constructively feasible throughout the network but attained only by a sparse population, separating geometric capacity from the trained endpoint. Under shared values, the same self-relation extends to the fixed-gain cross-head law $T _ { i } T _ { j } \approx \alpha _ { j } T _ { i }$ , producing distinct oblique operators with a common value-defined kernel. These results identify scaled idempotence as a recurrent geometry in trained weights and show how it induces a local operator algebra in Transformer attention.

## Author Contributions

Jiming Feng led the conception, experimental design, implementation, analysis, and manuscript preparation.   
Junliang Li contributed through research discussions and experimental development.

## Reproducibility Statement

A reproducibility package has been prepared for this work. All figures are generated from fixed result artifacts; the package contains archival analysis entry points, recorded model identifiers and revisions, selection rules, seeds, controls, per-head outputs, and automated checks of the retained numerical claims. No model weights are redistributed.

## References

01.AI. Yi-1.5-6B. https://huggingface.co/01-ai/Yi-1.5-6B, 2024.

Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. GQA: Training generalized multi-query transformer models from multi-head checkpoints. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 2023.

Elie Bakouch, Loubna Ben Allal, Anton Lozhkov, et al. SmolLM3: smol, multilingual, long-context reasoner. https://huggingface.co/blog/smollm3, 2025.

Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, et al. Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning, 2023.

Åke Björck and Gene H. Golub. Numerical methods for computing angles between linear subspaces. Mathematics of Computation, 27(123):579–594, 1973.

Dake Bu, Wei Huang, Andi Han, Atsushi Nitanda, Taiji Suzuki, Qingfu Zhang, and Hau-San Wong. Provably transformers harness multi-concept word semantics for eficient in-context learning. In Advances in Neural Information Processing Systems, 2024.

Zitong Andrew Chen, Junaid Hasan, Akhil Srinivasan, Hemkesh Bandi, and Jarod Alper. Multiplication beyond groups: Stratified fourier mechanisms in transformer circuits. arXiv preprint arXiv:2607.07066, 2026.

Nelson Elhage, Neel Nanda, Catherine Olsson, et al. A mathematical framework for transformer circuits. Transformer Circuits Thread, 2021. URL https://transformer-circuits.pub/2021/framework/index.html.

Dirk Groeneveld, Iz Beltagy, Pete Walsh, et al. OLMo: Accelerating the science of language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics, 2024.

Jihun Hamm and Daniel D. Lee. Grassmann discriminant maps. In International Conference on Machine Learning, 2008.

IBM Granite Team. Granite-3.3-2B-Base. https://huggingface.co/ibm-granite/granite-3.3-2b-base, 2025.

Skyler Bryn Jensen and Jamie Vicary. Enforcing idempotency in neural networks. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 27070–27090. PMLR, 2025.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.

Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, and Jacob Steinhardt. Progress measures for grokking via mechanistic interpretability. In International Conference on Learning Representations, 2023.

Catherine Olsson, Nelson Elhage, Neel Nanda, et al. In-context learning and induction heads. arXiv preprint arXiv:2209.11895, 2022.

Qwen Team. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.

Team OLMo, Pete Walsh, Luca Soldaini, et al. 2 OLMo 2 furious. arXiv preprint arXiv:2501.00656, 2025.

Technology Innovation Institute. Falcon 3: A Family of Open Language Models. https://huggingface.co/blog/ falcon3, 2024.

Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. Interpretability in the wild: a circuit for indirect object identification in GPT-2 small. In International Conference on Learning Representations, 2023.

Richard Zhe Wang. The communication map of a transformer. arXiv preprint arXiv:2608.22007, 2026.

Hiroaki Yamagiwa, Yusuke Takase, and Hidetoshi Shimodaira. Measuring afinity between attention-head weight subspaces via the projection kernel. arXiv preprint arXiv:2601.10266, 2026.

An Yang et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Peiyuan Zhang, Guangtao Zeng, Tianda Wang, and Wei Lu. TinyLlama: An open-source small language model. arXiv preprint arXiv:2401.02385, 2024.