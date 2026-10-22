# A Sharp Dynamic Haussler Lemma

Shihao Shao<sup>1</sup> and Dacheng Tao<sup>1</sup>

<sup>1</sup>College of Computing and Data Science, Nanyang Technological University

{shihao.shao, dacheng.tao}@ntu.edu.sg

## Abstract

Haussler’s covering lemma (JCTA 1995) established the sharp $( C / \delta ) ^ { d }$ metric-entropy bound for binary classes of VC dimension d under normalized $\ell _ { 1 }$ distance. We establish a sharp global dynamic counterpart for bounded real-valued classes in every normalized $\ell _ { p }$ metric, $1 \leq p < \infty$ . For every such $p ,$ every integer $d \geq 1$ , and every $0 < \varepsilon < 1 / 4$ , we prove the matching bounds

$$
\left( \frac { c } { \varepsilon } \right) ^ { p d } \leq \operatorname* { s u p } _ { \mathcal { F } , T : \atop \operatorname { s f a t } _ { \varepsilon / 3 2 } ( \mathcal { F } ) \leq d } N _ { p } ^ { \mathrm { d y n } } ( \mathcal { F } , \varepsilon , T ) \leq \left( \frac { C } { \varepsilon } \right) ^ { p d } ,
$$

where $c , C > 0$ are universal constants, independent also of $p .$ The upper bound holds for each class and each horizon, and extends to all $0 < \varepsilon < 1 / 2$ The cover is a single finite family of deterministic causal experts, fixed independently of the instance sequence, that approximates every target on every sequence to normalized $\ell _ { p }$ error at most ε. The lower bound already holds for binary classes on a single fixed sequence, even with real-valued covering experts.

For $p \ = \ 2$ , our theorem resolves the horizon-independent sequential $\ell _ { 2 }$ covering problem of Rakhlin, Sridharan, and Tewari (PTRF 2015) in a stronger global form. For $p = 1$ , it yields the sharp $( C / \delta ) ^ { \mathrm { L d i m } ( \mathcal { H } ) }$ global dynamic covering bound for binary classes. This resolves the global approximate covering problem posed by Alon, Ben-Eliezer, Dagan, Moran, Naor, and Yogev (STOC 2021), as well as the dynamic Haussler problem posed by Malliaris and Moran (Trans. AMS 2024). Our proof uses SOA-style predictions with scheduled rank-decreasing version-space updates to control large pointwise errors. Rather than keeping every resulting prediction stream as a standalone expert, we use accurate prefixes that can be extended by exact predictions after identifying a target. Each activated target has many eligible prefixes but needs only one continuation. Shared-parent routing distributes these continuation requests across eligible prefixes, avoiding the worst-case concentration of demand on individual prefixes and eliminating the logarithmic loss.

## Contents

1 Introduction 1   
1.1 Our Contributions 2   
1.2 Historical Context and Related Work . 3   
2 Technical Overview 5   
2.1 Outlier Reduction and Correction Schedules 5   
2.2 From One Pool to Many Short Pools 6   
2.3 Shared-Parent Routing and Sharp Counting 7   
2.4 Simultaneous Accuracy and Global Determinization 9   
2.5 The Matching Lower Bound 10   
3 Definitions and Main Results 11   
3.1 Sequential Fat-Shattering Dimension 11   
3.2 Global Dynamic Covering Numbers 12   
3.3 Sharp Upper and Lower Bounds 12   
3.4 Binary Classes and Sequential Covering 13   
4 Correction Schedules and Adaptive Sampling 15   
4.1 Separation and Rank-Decreasing Centers . 15   
4.2 Correction Schedules and Pointwise Covers 17   
4.3 Partial Binary Completion and Adaptive Sampling 19   
5 Master Histories and Multi-Pool Accuracy 23   
5.1 The Analytical Master and Uniform Accuracy 24   
5.2 Tolerant Replay and Bufered Corrections 26   
5.3 Simultaneous Accuracy Across Pools 28   
6 Robust Identification and Eligible Parents 31   
6.1 Robust Identification and Local Counting 31   
6.2 Parent and Child Experts 33   
6.3 Eligibility and Many Compatible Parents 34   
7 Shared-Parent Routing and Load Balancing 36   
7.1 The Causal Routing Protocol 36   
7.2 Routing Probabilities and Load Bounds 37   
8 Expert Counting and Global Determinization 39   
8.1 Expert Counting and Simultaneous Coverage 39   
8.2 Game Determinization and Main Upper Bound 40   
9 Matching Lower Bounds 43   
9.1 Finite-Field Hyperplanes and Their Dimensions . 43   
9.2 Fixed-Sequence Packing and Lower Bound . 44

## 1 Introduction

How many predictors are needed to approximate an entire function class? In the classical setting, Haussler’s covering lemma [Hau95] gives a sharp answer for binary classes. For every class ${ \mathcal { H } } \subseteq \{ 0 , 1 \} ^ { \chi }$ of VC dimension $d ,$ every finite instance sequence $x _ { 1 } , \ldots , x _ { T }$ , and every $0 < \delta < 1$ there is a set of at most $( C / \delta ) ^ { d }$ representatives such that each target in H disagrees with some representative on at most $\delta T$ positions. The bound is independent of the sequence length, and the exponent is exactly the combinatorial dimension. VC dimension originated in uniform-convergence theory [VC71] and became a central parameter in distribution-free PAC learnability [Blu+89]. The connection between combinatorial dimensions and metric entropy extends to real-valued function classes [MV03] and plays a central role in uniform convergence and generalization [VW96; BM02]. More broadly, metric entropy underlies chaining bounds for stochastic processes [Dud67; Tal05].

The classical cover may be chosen after the entire instance sequence is known. Such hindsight dependence is unsuitable for online prediction, where future instances are unavailable. To eliminate this dependence, one may instead maintain a bounded number of candidate prediction streams, updating their predictions as instances arrive. This leads to the notion of a causal expert, whose prediction at time t may depend on the observed prefix $x _ { 1 } , \ldots , x _ { t }$ . We ask whether a single finite family of such experts, specified in advance, can approximate every target on every possible input sequence. This global guarantee requires consistency across diferent sequences: whenever two sequences share a prefix, each expert must make identical predictions along that prefix.

We study this question for bounded real-valued classes, measuring approximation in normalized $\ell _ { p }$ distance for any $1 \leq p < \infty$ . Let ${ \mathcal { F } } \subseteq [ 0 , 1 ] ^ { \mathcal { X } }$ be nonempty, and fix a horizon $T \geq 1$ . A deterministic causal expert is a tuple $g = ( g _ { t } ) _ { t = 1 } ^ { T }$ with $g _ { t } : \mathcal { X } ^ { t }  [ 0 , 1 ]$ . A finite family $\mathcal { G }$ is a global dynamic $\ell _ { p }$ cover at accuracy ε if

$$
\forall x _ { 1 : T } \in \mathcal { X } ^ { T } \forall f \in \mathcal { F } \exists g \in \mathcal { G } : \quad \left( \frac { 1 } { T } \sum _ { t = 1 } ^ { T } | g _ { t } ( x _ { 1 : t } ) - f ( x _ { t } ) | ^ { p } \right) ^ { 1 / p } \leq \varepsilon .\tag{1}
$$

We write $N _ { p } ^ { \mathrm { d y n } } ( \mathcal { F } , \varepsilon , T )$ for the smallest cardinality of such a family. The family may depend on ${ \mathcal { F } } , p , \varepsilon$ , and $T$ , but is fixed before the instances are revealed. Its experts receive no target values, and the expert witnessing the approximation may depend on both the target and the entire sequence. Thus global refers to a single family that works on all instance sequences; integral refers to an actual finite family of experts.

Sequential covering has a substantial history. The Littlestone dimension characterizes the optimal deterministic mistake bound in realizable online learning [Lit88]. Extending this characterization to agnostic online learning, Ben-David, P´al, and Shalev-Shwartz [BPS09] constructed a global exact cover for every binary class of Littlestone dimension $d ,$ consisting of at most $\scriptstyle \sum _ { i = 0 } ^ { \operatorname* { m i n } \{ d , T \} } { \binom { T } { i } }$ deterministic causal experts. Allowing a positive error rate suggests that the dependence on $T$ might disappear. In the real-valued setting, Rakhlin, Sridharan, and Tewari [RST10; RST15a; RST15b] developed sequential covering numbers and asked for an $\ell _ { 2 }$ bound independent of the horizon. Related sequential complexity methods also apply to objectives beyond external regret, including calibration and approachability [RST11]. Sequential covering numbers also have direct applications to online prediction. Rakhlin and Sridharan [RS14] established optimal regret rates for online nonparametric regression in terms of sequential entropy. Gaillard and Gerchinovitz [GG15] subsequently developed chaining-based forecasters with entropy-dependent regret guarantees. The global approximate covering question for binary classes was posed in [Alo+21], and Malliaris and Moran [MM24] explicitly asked for a dynamic analogue of Haussler’s lemma. Horizon-independent fractional covering bounds were established for global binary covering by Alon, Ben-Eliezer, Dagan,

Moran, Naor, and Yogev [Alo+21] and for tree-dependent real-valued sequential covering by Block, Dagan, and Rakhlin [BDR21]. The former provides one distribution over causal experts, whereas the latter allows the distribution to depend on the prescribed instance tree. Haussler’s lemma, however, guarantees an actual finite family of representatives. A sharp, horizon-independent integra dynamic analogue, together with its extension to real-valued classes, has remained open.

## 1.1 Our Contributions

We establish a sharp global dynamic covering theorem for bounded real-valued classes, uniformly over all finite $p \geq 1$ . The controlling parameter is the sequential fat-shattering dimension sfa $\mathfrak { j } _ { \eta } ( \mathcal { F } )$ the scale-sensitive sequential analogue of VC dimension. Informally, it is the maximum depth of a complete binary instance tree on which every path can be realized by a function in ${ \mathcal { F } } _ { : }$ , with the two alternatives at each node separated by a gap η around a node-dependent threshold. For binary classes, this dimension equals Littlestone dimension at every scale $0 < \eta \leq 1$ . The precise definition is given in the preliminaries.

Theorem 1 (Sharp global dynamic $\ell _ { p }$ covering). There are universal constants $c > 0$ and $C \geq 1$ such that, for every $1 \leq p < \infty$ , every integer $d \geq 0$ , and every $0 < \varepsilon < 1 / 4$

$$
\left( \frac { c } { \varepsilon } \right) ^ { p d } \leq \operatorname* { s u p } _ { \substack { \chi , \ T \geq 1 , \ \varnothing \neq \mathcal { F } \subseteq [ 0 , 1 ] ^ { \chi } \colon } } N _ { p } ^ { \mathrm { d y n } } ( \mathcal { F } , \varepsilon , T ) \leq \left( \frac { C } { \varepsilon } \right) ^ { p d } .\tag{2}
$$

More precisely, for every $1 \leq p < \infty$ , every $T \geq 1$ , and every $0 < \varepsilon < 1 / 2$ , each nonempty class ${ \mathcal { F } } \subseteq [ 0 , 1 ] ^ { \chi }$ with sfa $\bar { \upsilon } _ { \varepsilon / 3 2 } ( \mathcal { F } ) < \infty$ satisfies

$$
N _ { p } ^ { \mathrm { d y n } } ( { \mathcal { F } } , \varepsilon , T ) \leq \left( { \frac { C } { \varepsilon ^ { p } } } \right) ^ { \mathrm { s f a t } _ { \varepsilon / 3 2 } ( { \mathcal { F } } ) } .\tag{3}
$$

The constants are independent of $p , d , T$ , and ε. The lower bound is witnessed by binary classes on a single fixed instance sequence, even when the covering experts are allowed real-valued predictions. If the displayed dimension is zero, one expert sufices.

The theorem identifies the exact worst-case accuracy exponent pd, with universal constants in the base. The more precise upper bound (3) retains d as the outer exponent and has a base constant independent also of $p .$ Thus the bounds for $p = 1$ and $p = 2$ are $( C / \varepsilon ) ^ { d }$ and $( C / \varepsilon ^ { 2 } ) ^ { d }$ , respectively. In every case, the dimension is measured at scale $\varepsilon / 3 2$ , corresponding to pointwise accuracy, while the mean pth-power error budget is $\varepsilon ^ { p }$

The lower bound is given by indicators of afine hyperplanes over a finite field. On one fixed instance sequence, their prediction vectors form a normalized $\ell _ { p }$ packing with separation greater than 2ε. Since the pth power of the normalized $\ell _ { p }$ distance between binary vectors equals their normalized Hamming distance, the same construction explains the exponent pd for every finite $p .$ The triangle inequality shows that even a real-valued expert can cover at most one vector in this packing.

Sequential covering and the $p \ = \ 2$ problem. Restricting every global expert to a fixed instance tree gives a sequential cover on that tree, with the same cardinality. Theorem 1 therefore yields horizon-independent integral sequential $\ell _ { p }$ covers for every $1 \leq p < \infty$ . In particular, its $p = 2$ case resolves the sequential $\ell _ { 2 }$ covering question of Rakhlin, Sridharan, and Tewari [RST15b] in a stronger global form: a single family of causal experts, fixed independently of the instance sequence, sufices to approximate every target on every possible sequence. It supplies an integral covering theorem alongside the fractional result of Block, Dagan, and Rakhlin [BDR21].

Binary classes and the $p = 1$ case. For a binary class H of Littlestone dimension $d ,$ the $p = 1$ case yields a family of at most $( C ^ { \prime } / \delta ) ^ { d }$ binary causal experts such that

$$
\forall x _ { 1 : T } \in \mathcal { X } ^ { T } \forall h \in \mathcal { H } \exists g \in \mathcal { G } : \quad \sum _ { t = 1 } ^ { T } \mathbf { 1 } \{ g _ { t } ( x _ { 1 : t } ) \neq h ( x _ { t } ) \} \leq \delta T ,\tag{4}
$$

for every $0 < \delta < 1 / 4$ . Indeed, apply (3) with $p = 1$ and $\varepsilon = \delta / 2$ , then round each prediction at $1 / 2$ . Each binary mistake contributes at least $1 / 2$ to the original absolute error, so the number of mistakes is at most $\delta T$ . The finite-field construction gives a matching worst-case lower bound $( c ^ { \prime } / \delta ) ^ { d }$ . This settles the global dynamic covering question posed in $[ \mathrm { A l o } + 2 1 ]$ and the dynamic Haussler problem of Malliaris and Moran [MM24]. A global cover also supplies a finite reference family for prediction-with-expert-advice methods [CL06], although our theorem does not address eficient construction of the experts.

Proof ideas. Our proof first establishes an outlier cover of size $( C _ { 0 } R ) ^ { d }$ for a uniformly separated class, with pointwise error at least 8η on fewer than $T / ( 8 R )$ rounds. Setting $\eta = \varepsilon / 3 2$ , passing to a maximal uniformly η-separated subclass of ${ \mathcal { F } } _ { : }$ , and choosing $R = 1 6 / \varepsilon ^ { p }$ then yields mean pth-power error at most $( 9 \varepsilon / 3 2 ) ^ { p } + \varepsilon ^ { p } / 1 2 8 < \varepsilon ^ { p }$ for the original class. This gives the bound $( 1 6 C _ { 0 } / \varepsilon ^ { p } ) ^ { d }$ for every finite $p .$

To obtain the outlier guarantee, SOA-style predictions and correction schedules implement rankdecreasing version-space updates without target labels. A common master and tolerant replays across short sampling pools give each target many accurate histories. A pool identifies a function when a robust certificate isolates it in a singleton version space; the function activates once identified by at least half the pools. Bounds on the number of functions each pool can certify within a lowrank version space, combined with this activation threshold, limit how many activated targets can be eligible for any given parent.

Rather than retaining every full schedule as an expert, we use parent schedules to cover unactivated targets or provide accurate prefixes for activated ones. Each activated target has many eligible parents but needs only one exact continuation. Shared-parent routing spreads these requests, avoiding worst-case concentration on individual parents and reducing their leading continuation capacity by a factor proportional to the number of pools. This eliminates the logarithmic overhead. A finite-horizon game argument then yields a deterministic global cover.

## 1.2 Historical Context and Related Work

Two distinctions clarify the relation between our theorem and earlier covering results: whether the cover may depend on a prescribed instance sequence or tree, and whether covering is witnessed by finitely many representatives or by a probability measure assigning uniformly positive mass to successful approximations.

Static, sequential, and global covers. In Haussler’s classical theorem [Hau95], the representatives may be chosen after the sample is specified. The sequential covers of Rakhlin, Sridharan, and Tewari [RST15b] are instead defined on a fixed binary instance tree x. A cover is a finite family $V _ { \mathbf { x } }$ of real-valued prediction trees such that, for every target $f$ and every root-to-leaf path $\sigma ,$ some $\mathbf { v } \in V _ { \mathbf { x } }$ approximates the target trace along that path. The family may depend on the entire instance tree, including nodes outside the realized path. The distinction is captured by the

following quantifier orders

$$
\begin{array} { r l } { \mathrm { t r e e - d e p e n d e n t : } } & { \forall \mathbf { x } \exists V _ { \mathbf { x } } \forall f \in \mathcal { F } \forall \sigma \in \{ - 1 , + 1 \} ^ { T } \exists \mathbf { v } \in V _ { \mathbf { x } } , } \\ { \mathrm { g l o b a l : } } & { \exists \mathcal { G } \forall x _ { 1 : T } \in \mathcal { X } ^ { T } \forall f \in \mathcal { F } \exists g \in \mathcal { G } . } \end{array}\tag{5}
$$

In the first line, accuracy is measured on the path $\sigma$ of x; the second line has the accuracy requirement in (1). In both cases the witnessing member may depend on the target and the complete path or sequence. The global formulation additionally specifies consistent predictions on every shared instance prefix, across all input sequences. Restricting a global family to a tree gives a sequential cover of the same size. Taking a supremum of sequential covering numbers over trees does not, by itself, interchange the first two quantifiers.

Globality is already present in the exact binary expert construction of Ben-David, P´al, and Shalev-Shwartz [BPS09], whose size grows with T, and in the fractional dynamic covers of [Alo+21]. The global approximate covering questions in [Alo+21; MM24] ask for a horizon-independent finite family. Our theorem supplies such a family with the sharp dimension and accuracy dependence, and extends the guarantee to bounded real-valued classes.

These questions also connect online learning to model-theoretic stability. Malliaris and Shelah [MS14] established regularity lemmas for stable graphs, while Malliaris and Moran [MM24] subsequently studied excellent sets through agnostic online learning.

Real-valued metric entropy. Scale-sensitive dimensions have long been used to control static covering numbers for real-valued function classes, both on finite samples and under distributiondependent metrics [Alo+97; BKP97; MV03]. Empirical entropy also yields oracle inequalities and risk bounds for nonparametric regression [RST17]. Recently, Aiyer, Mansour, Moran, Shao, and Waknine $[ \mathrm { A i y + 2 6 } ]$ obtained sharp results for empirical $\ell _ { \infty }$ covering numbers at optimal scales. In these settings, covering families may depend on the sample or probability measure defining the metric. Our theorem instead provides a single finite family of causal experts, with sharp global dynamic bounds in every finite $\ell _ { p }$ metric.

Geometric refinements. Haussler’s static packing bound has also been refined for shallow geometric set systems [DEG16]. These refinements exploit additional structure of static set systems, whereas our focus is global causal covering.

Fractional and integral covering. For a target–sequence pair $( f , x _ { 1 : T } )$ , let $B _ { \varepsilon } ( f , x _ { 1 : T } )$ be the set of causal experts satisfying the error bound in (1) for that pair. A global fractional cover of size $\Gamma \geq 1$ is a probability measure $\mu ,$ fixed independently of the input sequence, such that

$$
\forall x _ { 1 : T } \in \mathcal { X } ^ { T } \forall f \in \mathcal { F } : \qquad \mu \big ( B _ { \varepsilon } ( f , x _ { 1 : T } ) \big ) \geq \Gamma ^ { - 1 } .\tag{6}
$$

This is the probability-mass formulation used for binary dynamic covers in $[ \mathrm { A l o } + 2 1 ]$ . A treedependent fractional cover instead chooses a measure $\mu _ { \mathbf { x } }$ on prediction trees for each fixed $\mathbf { x } ,$ and imposes the same mass condition for every target and path on that tree, as in Block, Dagan, and Rakhlin [BDR21].

An integral cover requires an actual finite family $\mathcal { G }$ intersecting every set $B _ { \varepsilon } ( f , x _ { 1 : T } )$ . The uniform distribution on a family of size K satisfies (6) with $\Gamma = K$ . The converse is not immediate: Γ controls the reciprocal of the mass of each successful set, rather than the cardinality of the support of $\mu .$ In particular, even a finitely supported fractional cover need not have support of size at most Γ. Here integral concerns the selection of representatives; the representatives themselves may make real-valued predictions.

Horizon-independent fractional covering bounds are already known. For binary classes, $[ \mathrm { A l o } + 2 1 ]$ establishes a global fractional bound of $( C / \delta ) ^ { d }$ at normalized Hamming accuracy $\delta ,$ while Block, Dagan, and Rakhlin [BDR21] obtain real-valued sequential $\ell _ { 2 }$ fractional bounds controlled by sequential fat-shattering dimension. These results, however, do not directly yield finite global integral covers. The dificulty is simultaneous coverage. Sampling m independent experts from $\mu$ misses a fixed target–sequence pair with probability at most $( 1 - 1 / \Gamma ) ^ { m } \leq e ^ { - m / \Gamma }$ . For M prescribed pairs, a union bound gives simultaneous coverage with probability at least $1 - \zeta$ using $m = \lceil \Gamma \log ( M / \zeta ) \rceil$ samples, where $0 < \zeta < 1$ . Thus, for any prescribed finite collection of M target–sequence pairs, there exists a family of m experts covering all of them. However, this argument introduces a logarithmic factor and does not directly handle the possibly infinite collection of all target–sequence pairs. On a fixed static sample, the classical packing argument converts fractional mass into a finite cover after doubling the radius. In the sequential setting, however, maximal packings on individual paths need not yield a common covering family, while packing uniformly across paths imposes a stronger requirement that is not controlled by the fractional covering bound. Block, Dagan, and Rakhlin [BDR21] discuss this failure of the classical packing–covering argument.

Our proof instead uses shared-parent routing to share continuation capacity across compatible sampled histories. This yields a sharp bound on the global integral covering number in terms of sequential fat-shattering dimension, without relying on a general fractional-to-integral conversion principle.

## 2 Technical Overview

We follow the standard optimal algorithm (SOA) principle: predict a center of the current version space, the functions surviving earlier restrictions. A fixed correction schedule prescribes when and in which direction to restrict this space; the candidate streams need no observed target labels. We restrict updates to short randomly sampled lists of rounds, called pools. Multiple pools give each target many accurate histories, but an activated target needs only one accurate past followed by exact predictions. Our shared-parent routing distributes these one-per-target continuations among eligible histories, reducing the leading capacity required per history by a factor of the number of pools. This cancels the multiplicative cost of maintaining multiple pools, removing the remaining logarithmic loss. A final game argument makes the family deterministic and global.

## 2.1 Outlier Reduction and Correction Schedules

We first control the number of rounds with large pointwise error. Call such a round an outlier, and write $\begin{array} { r } { \| h - h ^ { \prime } \| _ { \infty } = \operatorname* { s u p } _ { x \in \mathcal { X } } | h ( x ) - h ^ { \prime } ( x ) | } \end{array}$

Proposition 2 (Global outlier cover). There is a universal constant $C _ { 0 } \geq 1$ with the following property. Let $d \geq 0$ be an integer, $0 < \eta < 1 , R > 3 2 , T \geq 1$ , and let $\emptyset \neq { \mathcal { H } } \subseteq [ 0 , 1 ] ^ { \chi }$ satisfy s $\mathrm { f a t } _ { \eta } ( \mathcal { H } ) \leq d < \infty$ and $\| h - h ^ { \prime } \| _ { \infty } > \eta$ for distinct $h , h ^ { \prime } \in \mathcal { H }$ . There is a family G of at most $( C _ { 0 } R ) ^ { d }$ deterministic causal experts such that, for every instance sequence $x _ { 1 : T }$ and every $h \in \mathcal H$ , some $g \in { \mathcal { G } }$ satisfies

$$
\# \{ t \in [ T ] : | g _ { t } ( x _ { 1 : t } ) - h ( x _ { t } ) | \geq 8 \eta \} < \frac { T } { 8 R } .\tag{7}
$$

The family is fixed independently of the instance sequence.

To obtain the $\ell _ { p }$ theorem, set $\eta = \varepsilon / 3 2$ and $d = \mathrm { s f a t } _ { \eta } ( \mathcal { F } )$ . Choose a maximal uniformly greaterthan-η separated subclass $\mathcal { H } \subseteq \mathcal { F }$ : distinct members have uniform distance greater than $\eta ,$ and

maximality makes every $f \in { \mathcal { F } }$ lie within uniform distance $\eta$ of some $h \in \mathcal H$ . Its dimension is at most d. Applying Proposition 2 with $R = 1 6 / \varepsilon ^ { p }$ gives

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } | g _ { t } ( x _ { 1 : t } ) - f ( x _ { t } ) | ^ { p } \leq ( 9 \eta ) ^ { p } + \frac { 1 } { 8 R } = \varepsilon ^ { p } \left[ \left( \frac { 9 } { 3 2 } \right) ^ { p } + \frac { 1 } { 1 2 8 } \right] < \varepsilon ^ { p } .\tag{8}
$$

Here the error is below 9η outside the outliers and at most one everywhere. The family has size at most $( 1 6 C _ { 0 } / \varepsilon ^ { p } ) ^ { d }$ . Thus the $\ell _ { p }$ upper bound reduces to Proposition 2, with $p$ entering only through the outlier budget.

Corrections that decrease dimension. For a version space $V \subseteq { \mathcal { H } }$ consisting of functions remaining after restrictions, define its rank as $\operatorname { s f a t } _ { \eta } ( V )$ (rank −1 if empty). Distinct functions in the separated class would shatter a depth-one tree; thus rank zero means V is a singleton, which predicts exactly. Assume $d \geq 1$ and $| \mathscr { H } | \geq 2$ henceforth.

For a nonempty V of rank r and current instance x, choose a center $a ( V , x ) \in [ 0 , 1 ]$ such that both sign-update classes

$$
V _ { x } ^ { - } : = \{ f \in V : f ( x ) \leq a ( V , x ) - \eta \} , \qquad V _ { x } ^ { + } : = \{ f \in V : f ( x ) \geq a ( V , x ) + \eta \}\tag{9}
$$

have rank at most $r - 1$ . Such a center exists because two rank-r subclasses separated by a gap η at x would form a shattered tree of depth $r + 1$ ; Lemma 12 gives the precise choice. For a target $h ,$ a reference procedure starts from $V = \mathcal { H }$ . On each pool-selected round, while $| a ( V , x ) - h ( x ) | > \eta ;$ it restricts V to the sign class containing h and recomputes the center. Each update keeps h and decreases rank, so at most d updates occur; after them, selected-round error is at most η. Unselected rounds cause no update.

Encoding updates without target values. Experts receive no target values, so we enumerate all correction lists. On a pool of at most m selected rounds, a schedule is

$$
J = { \big ( } ( s _ { 1 } , \sigma _ { 1 } ) , \ldots , ( s _ { i } , \sigma _ { i } ) { \big ) } , \qquad 1 \leq s _ { 1 } \leq \cdots \leq s _ { i } \leq m , \quad \sigma _ { \ell } \in \{ - , + \} , \quad i \leq d .
$$

The depth i counts corrections; an ordinal $s _ { \ell }$ is a position on the selected list, not in [T]. Repeated ordinals allow successive corrections at one instance. The empty schedule is allowed; an invalid update produces zero thereafter. A schedule updates before predicting. For each target and realized input-and-selection history, recording the correction times and signs of the target-dependent reference procedure yields a witnessing schedule in this fixed family. Only ordinals and signs are encoded: no expert reads target values or discretizes them.

There are $n _ { i } ( m ) = 2 ^ { i } \binom { m + i - 1 } { i }$ depth-i schedules for $i \geq 1$ , and $n _ { 0 } ( m ) = 1$ . Their total through depth r satisfies

$$
Q _ { r } ( m ) : = \sum _ { i = 0 } ^ { r } n _ { i } ( m ) \leq \left[ 2 e \left( 1 + { \frac { m } { r } } \right) \right] ^ { r } \quad ( r \geq 1 ) .\tag{10}
$$

For a fixed sampling outcome, these streams use only the observed instance-and-selection history. Selecting every round yields a deterministic pointwise cover of size $Q _ { d } ( T )$ , suficient for $T = O ( d R )$

## 2.2 From One Pool to Many Short Pools

Put $b = T / R _ { \ast }$ ; we permit fewer than $b / 8$ outliers. A Bernoulli-γ pool selects each round independently with probability γ, after $x _ { t }$ is chosen. Inputs may depend on past random choices, but not on the current sampling bits.

The cost of a direct single-pool analysis. A pool with $\gamma b = \Theta ( d )$ and cap $m = \left\lceil 2 \gamma T \right\rceil =$ $\Theta ( d R )$ admits only $Q _ { d } ( m ) \leq ( C R ) ^ { d }$ schedules. But accuracy on selected rounds does not control errors on unselected rounds. We need a single sampling outcome on which, simultaneously for every target, some schedule has few outliers over the entire horizon.

Lemma 17 bounds the probability that many errors occur but few are selected, uniformly over targets. Applying it to each fixed schedule and taking a union bound gives, for $\gamma \le 1 / 2$ and $\gamma b$ a suficiently large multiple of $d ,$ a principal term of the form

$$
\exp \left\{ O \left( d \log \left( 1 + \frac { \gamma T } { d } \right) \right) - c \gamma b \right\} ,\tag{11}
$$

plus exponentially small terms accounting for excessive sample lengths. With $\gamma b = \Theta ( d )$ , the displayed bound exceeds 1 as R grows and thus fails to establish simultaneous accuracy. Taking $\gamma b = A d \log R$ for a suficiently large constant A makes it useful, but increases the schedule count to $( C R \log R ) ^ { d }$ . Selecting every round $( m : = T )$ handles the range where the required rate exceeds $1 / 2$ . This direct analysis thus leaves an extra factor of $( \log R ) ^ { d }$

Several short pools and a common reference. Fix a suficiently large universal α. The all-round cover handles $b \leq 3 2 \alpha d ;$ otherwise set

$$
k = 4 { \lceil \log _ { 2 } R \rceil } , \qquad \beta = \frac { \alpha d } { b } < \frac { 1 } { 3 2 } , \qquad m = { \lceil 2 \beta T \rceil } = \Theta ( d R ) .\tag{12}
$$

Use independent Bernoulli-β bits $\xi _ { t , j }$ across rounds $t \in [ T ]$ and pools $j \in [ k ]$ . Each pool retains only its first $m$ selections. We call the uncapped lists raw pools; their independent bits are used in the probability estimates. Let Cap be the event that no raw list exceeds m; on this event, raw and capped simulations agree.

To exploit all pools without lengthening their codes, for each h run a target-dependent reference procedure, its master, on the union of raw selections if $k \beta \leq 1 / 8$ , and on every round otherwise. This larger sample lets us control the master’s errors uniformly over targets. Every pool selection is also a master selection, so the master’s final value $y _ { t } ^ { h }$ is within η of $h ( x _ { t } )$ on those rounds. For each target $h ,$ the analytical master provides a common reference prediction history for controlling the errors of its pool replays.

A replay in pool j maintains its own version space. On that pool’s selected rounds it repeatedly corrects toward $y _ { t } ^ { h }$ until its center is within $4 \eta$ , using the same sign updates. For example, $y _ { t } ^ { h } >$ $a + 4 \eta$ implies $h ( x _ { t } ) > a + 3 \eta$ , whereas the retained constraint is only $f ( x _ { t } ) \geq a + \eta$ . Thus h lies more than $2 \eta$ beyond each retained inequality’s boundary; this distance is the constraint’s slack. There are at most $d$ corrections, and selected-round error is at most $5 \eta$ . Replays are analytical witnesses, not additional experts. On Cap, each is reproduced by a formal schedule on its capped pool.

Proposition 21 shows that, with probability above $1 5 / 1 6$ , Cap holds and every target has at least $3 k / 4$ replays with fewer than $b / 8$ outliers of size at least $8 \eta$ over the whole horizon; see Section 2.4. Retaining every pool’s full schedules would cost $K _ { \mathrm { n a i v e } } = k Q _ { d } ( m ) \leq ( C R ) ^ { d } \log R$ experts. To eliminate this remaining factor of k, we exploit the redundancy of accurate histories across pools.

## 2.3 Shared-Parent Routing and Sharp Counting

Identifying a function permits exact continuation. Once a schedule’s version space is $\{ h \}$ the function h is known and future predictions can equal $h ( x _ { t } )$ exactly; this is a continuation. It cannot retroactively repair earlier predictions, so it must extend a stream with an accurate past.

A schedule is born once all its prescribed updates have occurred; its version space is fixed thereafter. A robust certificate for h is a valid born schedule whose version space is $\{ h \}$ and whose update constraints hold for h with slack at least $2 \eta .$ , as in the replay inequality above. This margin will make diferent functions certified in one pool difer by more than $2 \eta$ on some selected instance, allowing us to bound how many functions the pool can identify. A pool identifies h when it has such a certificate; h activates once at least $k / 2$ pools identify it. Identification concerns a candidate function, not the unknown true target: its past values can be checked without labels.

Parents preserve the past; children supply continuations. For every pool and every schedule of depth at most $d - 1$ , retain its prediction stream as a parent. There are $P = k Q _ { d - 1 } ( m )$ parents. After a depth-i parent v is born valid, its version space $V _ { v }$ is fixed and has rank at most $d - i .$ Preallocate $c _ { i }$ additional streams, called children, to every depth-i parent. Each child copies its parent’s outputs until assigned to an activated function $h ,$ then predicts h exactly from the assignment round onward. All children exist from the start, so assignment never changes their past outputs.

At the activation of h on round t, a parent is eligible if it is valid and born by $t - 1$ , and

$$
h \in V _ { v } , \qquad | V _ { v } | \geq 2 , \qquad \sum _ { s < t } \mathbf { 1 } \{ | p _ { s } ^ { v } - h ( x _ { s } ) | \geq 8 \eta \} < b / 8 ,
$$

where $p _ { s } ^ { v }$ is its actual output. Retain one eligible parent per pool in a fixed order, and call the resulting set $\mathcal { E } ( h , t )$ . Eligibility is testable at activation using only known past outputs and the now-identified function, not future accuracy.

On the simultaneous-accuracy event, at least $3 k / 4$ replays for h are accurate, while fewer than $k / 2$ pools identified it by $t - 1$ . Thus more than $k / 4$ accurate replay prefixes still have nonsingleton version spaces: a singleton replay prefix would already be a robust certificate. Such a space has rank at least one, so its prefix uses at most $d - 1$ corrections and coincides with a preallocated parent. Hence $| \mathcal { E } ( h , t ) | \geq k / 4 ;$ any of these parents has an accurate past that a child can extend by predicting h exactly. If h never activates, an accurate unidentified pool supplies a full replay of depth at most $d - 1$ , already among the parents. Depth-d schedules serve identification but are not retained.

Robust certificates bound the demand at each parent. Two distinct functions identified in one pool must difer by more than $2 \eta$ on a selected instance: one violates a constraint of the other’s singleton certificate, whereas the certified function satisfies that constraint with slack at least $2 \eta$ . For any rank-r subclass $V ,$ , the pointwise radius-η schedule cover on that selected list has size $Q _ { r } ( m )$ . No covering vector can approximate two such identified functions, so the pool identifies at most $Q _ { r } ( m )$ members of $V$

Let A be the set of activated functions. Each is identified in at least $k / 2$ pools, and hence

$$
\frac { k } 2 | A \cap V | \leq k Q _ { r } ( m ) , \qquad | A \cap V | \leq 2 Q _ { r } ( m ) .\tag{13}
$$

A depth-i parent can therefore be eligible for at most $2 Q _ { d - i } ( m )$ activations. This bound holds on every outcome, even for version spaces produced during the run, and ensures finitely many activations when H is infinite.

Routing balances the continuation requests. Without routing, a depth-i parent might need $2 Q _ { d - i } ( m )$ children: all eligible activations could choose it. Instead, each activation with at least $k / 4$ eligible parents chooses one uniformly using fresh randomness; an activation with fewer is left unassigned. Any given parent is then chosen with conditional probability at most $4 / k$ per request, and its expected load is at most $8 Q _ { d - i } ( m ) / k$ . Full parents remain eligible to preserve the $4 / k$ selection bound, while a separate no-overload event controls capacity violations. An exponentia tail bound, followed by a union bound over all P parents, justifies capacities

$$
c _ { i } = \left\lceil \frac { 1 6 Q _ { d - i } ( m ) } { k } + \log ( 4 P ) \right\rceil , \qquad 0 \leq i \leq d - 1 .\tag{14}
$$

The logarithm covers simultaneous load fluctuations, and the probability of any overload is at most $1 / 4$ even with adaptive inputs, without conditioning on the accuracy event.

There are $k n _ { i } ( m )$ depth-i parents. Counting all parents and children, including invalid schedules and unused streams, gives

$$
\begin{array} { r l } {  { K = P + \sum _ { i = 0 } ^ { d - 1 } k n _ { i } ( m ) c _ { i } } } \\ & { \leq 1 6 \sum _ { i = 0 } ^ { d - 1 } n _ { i } ( m ) Q _ { d - i } ( m ) + P \big ( \log ( 4 P ) + 2 \big ) } \\ & { \leq 1 6 Q _ { d } ( 2 m ) + P \big ( \log ( 4 P ) + 2 \big ) . } \end{array}\tag{15}
$$

Thus the k parents-per-schedule factor cancels with the $1 / k$ capacity per parent. The final inequality follows by distributing at most d signed corrections between two lists of length m (Lemma 14). No child is assigned to two targets; the saving comes from choosing among many eligible parents. Since $m = O ( d R )$ , the first term is at most $( C R ) ^ { d }$ . Parents have depth at most $d - 1$ , so their overhead is at most Cd(log $R ) ^ { 2 } ( C R ) ^ { d - 1 }$ . The saved power of R absorbs the logarithmic factors, since $( \log R ) ^ { 2 } \leq R ;$ the factor $d \leq 2 ^ { d }$ is absorbed into the universal base constant. Thus the overhead is at most $( C ^ { \prime } R ) ^ { d }$ . Overall, the counts improve as

$$
\underbrace { ( C R \log R ) ^ { d } } _ { \mathrm { o n e ~ l o n g ~ p o o l } } \longrightarrow \underbrace { ( C R ) ^ { d } \log R } _ { \mathrm { m a n y ~ s h o r t ~ p o o l s } } \longrightarrow \underbrace { ( C R ) ^ { d } } _ { \mathrm { s h a r e d - p a r e n t ~ r o u t i n g } } .
$$

## 2.4 Simultaneous Accuracy and Global Determinization

We control four events: Cap bounds pool lengths; Master bounds each target’s large master errors; Joint excludes too many replays with frequent master–replay disagreements but few sampled disagreements; and NoOverload prevents routing beyond child capacities. All four hold together with positive probability.

Cap: bounded sampling. A Chernof bound makes Cap highly probable. On this event, every target’s master code lies in a fixed family of $Q _ { d } ( 4 k m )$ codes. We use the independent raw bits without conditioning on Cap.

Master: uniform master accuracy. For fixed master code $J ,$ let $a _ { t } ^ { J }$ denote its provisional prediction, computed before the current pool bits are revealed. Lemma 17 bounds, uniformly over targets, the probability of many master errors of size at least $2 \eta$ but few sampled errors exceeding $\eta ,$ in either direction. In the union regime, $u = 1 - ( 1 - \beta ) ^ { k } = \Theta ( k \beta )$ and $u b = \Theta ( \alpha k d )$ make this bound small enough for a union bound over fixed codes and both directions. Outside these union-bound failures, the d-correction bound implies Master on Cap: every target has fewer than $b / 3 2$ such provisional errors. In the all-round regime, each such error forces a correction, so Master holds directly.

Joint: simultaneous replay accuracy. Let $a _ { t , j } ^ { J }$ denote pool $j ^ { \prime } \mathrm { s }$ provisional replay prediction. Set $\begin{array} { r } { D _ { j } ^ { J } : = \sum _ { t } \mathbf { 1 } \{ | a _ { t } ^ { J } - a _ { t , j } ^ { J } | > 6 \eta \} } \end{array}$ and $\begin{array} { r } { H _ { j } ^ { J } : = \sum _ { t } \tilde { \xi _ { t , j } } \mathbf { 1 } \{ | a _ { t } ^ { J } - a _ { t , j } ^ { J } | > 6 \eta \} } \end{array}$ , counting all and sampled disagreements. Their indicators are fixed before current independent pool bits. For a fixed J and any $k / 4$ pools, a joint exponential supermartingale bounds the probability that all have $D _ { j } ^ { J } \geq b / 1 6$ and $H _ { j } ^ { J } \leq 2 d \mathrm { ~ b y ~ } e ^ { - \alpha k d / 2 5 6 }$ . A union bound over $Q _ { d } ( 4 k m ) \binom { k } { k / 4 }$ code–subset pairs yields Joint with high probability; the replay histories need not be independent.

For a target’s realized code $J ^ { h }$ on ${ \mathsf { C a p } } _ { \mathsf { i } }$ every sampled disagreement forces a master or replay correction, so $H _ { j } ^ { J ^ { h } } \leq 2 d$ . Hence Joint gives at least $3 k / 4$ pools with $D _ { j } ^ { J ^ { h } } < b / 1 6$ . Together with Master, each has fewer than $b / 3 2 + b / 1 6 < b / 8$ provisional errors of size at least 8η: each is a large master error or disagreement. Final predictions obey the same bound, since selected-round errors are at most 5η and unselected predictions are unchanged.

NoOverload: successful routing. The load bound gives $\operatorname* { P r } ( \mathsf { N o O v e r l o a d ^ { c } } ) \leq 1 / 4$ . For suficiently large $\alpha _ { \mathrm { { i } } }$ , the first three events hold with probability above $1 5 / 1 6 .$ , providing eligible parents for activated targets and covering parents for unactivated ones. On the intersection of these three events with NoOverload, each activation receives an unused child. A union bound then gives simultaneous coverage with probability greater than $1 - 1 / 1 6 - 1 / 4 = 1 1 / 1 6$ , even against adaptive input rules observing earlier sampling and routing choices.

A global strategy, not a seed for each sequence. Positive success probability against each adaptive input rule does not give one seed for all sequences. Instead, consider a finite-horizon perfect-information game: the adversary chooses $x _ { t } ,$ then the constructor chooses sampling and routing actions, both observing past moves. The constructor wins when its K streams cover all targets. If no deterministic constructor strategy always won, backward induction would give an adversary forcing failure against every constructor strategy, contradicting the probability bound. Fixing a winning strategy makes each stream’s predictions depend only on $x _ { 1 : t }$ , yielding K deter ministic causal experts. This proves Proposition 2; (8) gives the general-p upper bound.

## 2.5 The Matching Lower Bound

For a prime power $q ,$ let $\mathbb { F } _ { q }$ be a field with q elements. Take the hyperplane indicators $h _ { a , c } ( x ) =$ $\mathbf { 1 } \{ a \cdot x = c \}$ on $\mathbb { F } _ { q } ^ { d } ,$ , where $a \doteq \mathbb { F } _ { q } ^ { d } \backslash \{ 0 \}$ and $c \in \mathbb { F } _ { q }$ . The resulting class contains $q ( q ^ { d } - 1 ) / ( q - 1 ) \geq q ^ { d }$ distinct functions and has Littlestone dimension exactly d. For the upper bound on dimension, predict one precisely on the afine span of previously observed positive instances. This span stays inside the target hyperplane. Each mistake adds a positive point outside the span and increases its afine dimension by one, from −1 for the initially empty span to at most $d - 1$ . Thus at most d mistakes occur. For the reverse bound, the d standard basis vectors are shattered.

List all $T = q ^ { d }$ domain points once. The normalized Hamming distance between distinct hyperplane indicators is at least $2 ( q - 1 ) / q ^ { 2 }$ . For binary vectors, this Hamming distance is the pth power of the normalized $\ell _ { p }$ distance. For $0 < \varepsilon < 1 / 4$ , choose $q$ as the largest power of two not exceeding $( 2 \varepsilon ) ^ { - p }$ . The traces are then pairwise more than $2 \varepsilon$ apart, and

$$
q ^ { d } > \left( \frac { 1 } { 2 ^ { p + 1 } \varepsilon ^ { p } } \right) ^ { d } \geq \left( \frac { 1 } { 4 \varepsilon } \right) ^ { p d } .
$$

The triangle inequality lets even an arbitrary real-valued covering vector approximate at most one trace. Since binary sequential fat-shattering dimension equals Littlestone dimension at every scale in (0, 1], this proves the matching exponent pd on a single fixed sequence, without using causality or adaptive inputs.

## 3 Definitions and Main Results

We now formalize the notions used in the preceding overview and state the precise covering theorems and their consequences.

Throughout, X is a nonempty instance domain, and function classes are nonempty unless stated otherwise. For an integer $T \geq 1$ , write $[ T ] = \{ 1 , \dots , T \}$ and $\boldsymbol { x } _ { 1 : t } = \left( x _ { 1 } , \ldots , x _ { t } \right)$ . For a sign sequence $\sigma _ { : }$ , write $\sigma _ { < t } = \left( \sigma _ { 1 } , \ldots , \sigma _ { t - 1 } \right)$ ; the prefix at t = 1 is empty. The exponent p always lies in [1, ∞). All prediction rules are arbitrary set-theoretic maps; no computability or measurability assumption is imposed.

## 3.1 Sequential Fat-Shattering Dimension

Fat-shattering extends VC dimension to real-valued function classes [KS94; BLW96]. Its sequential analogue is a central complexity measure in online learning [RST10; RST15a]. An X-valued binary tree of depth $r$ is a collection $\mathbf { x } = ( \mathbf { x } _ { t } ) _ { t = 1 } ^ { r }$ of maps $\mathbf { x } _ { t } : \{ - 1 , + 1 \} ^ { t - 1 }  \mathcal { X }$ . A real-valued tree $\mathbf { s } = ( \mathbf { s } _ { t } ) _ { t = 1 } ^ { r }$ is defined in the same way, with range R. Thus the instance and threshold at level t may depend on the preceding signs, but not on the sign selected at that level.

Definition 3 (Sequential fat-shattering dimension). Let $\eta > 0$ and ${ \mathcal { F } } \subseteq [ 0 , 1 ] ^ { \mathcal { X } }$ . The class $\mathcal { F }$ sequentially fat-shatters an instance tree x of depth r at scale η if there is a real-valued threshold tree s such that

$$
\forall \sigma \in \{ - 1 , + 1 \} ^ { r } \exists f \in { \mathcal { F } } \forall t \in [ r ] : \qquad \sigma _ { t } { \big ( } f ( \mathbf { x } _ { t } ( \sigma _ { < t } ) ) - \mathbf { s } _ { t } ( \sigma _ { < t } ) { \big ) } \geq { \frac { \eta } { 2 } } .\tag{16}
$$

The sequential fat-shattering dimension $\mathrm { s f a t } _ { \eta } ( \mathcal { F } )$ is the largest such depth, and is ∞ if these depths are unbounded. A nonempty class has dimension at least zero; we set $\mathrm { s f a t } _ { \eta } ( \mathcal { O } ) = - 1$

Our convention uses a margin $\eta / 2$ on each side of a threshold, so the two branches are separated by a gap η. Instances may repeat, including along a single path. The dimension is monotone under inclusion of classes and nonincreasing in the scale: if ${ \mathcal { F } } _ { 0 } \subseteq { \mathcal { F } }$ and $0 < \eta \le \eta ^ { \prime }$ , then

$$
\mathrm { s f a t } _ { \eta ^ { \prime } } ( { \mathcal F } _ { 0 } ) \leq \mathrm { s f a t } _ { \eta } ( { \mathcal F } ) .
$$

For a binary class $\mathcal { H } \subseteq \{ 0 , 1 \} ^ { \mathcal { X } }$ , the Littlestone dimension Ldim(H) is the largest depth of an instance tree on which every sign path $\sigma$ has a witness $h \in \mathcal H$ satisfying $h ( \mathbf { x } _ { t } ( \sigma _ { < t } ) ) = ( 1 + \sigma _ { t } ) / 2$ at every level. We use the same conventions for infinite dimension and the empty class. For binary classes, sequential fat-shattering dimension coincides with Littlestone dimension:

$$
\mathrm { s f a t } _ { \eta } ( \mathcal { H } ) = \mathrm { L d i m } ( \mathcal { H } ) , 0 < \eta \leq 1 .
$$

Indeed, a Littlestone-shattered tree is η-fat-shattered with thresholds $1 / 2$ , while fat-shattering of a binary class at any positive scale forces the two branches at each node to have labels 0 and 1, respectively.

## 3.2 Global Dynamic Covering Numbers

For $u , v \in \mathbb { R } ^ { T }$ , define the normalized $\ell _ { p }$ distance by

$$
\rho _ { p , T } ( u , v ) : = \left( \frac { 1 } { T } \sum _ { t = 1 } ^ { T } | u _ { t } - v _ { t } | ^ { p } \right) ^ { 1 / p } .\tag{17}
$$

A deterministic causal expert at horizon $T$ is a tuple $\textit { g } = \left( g _ { t } \right) _ { t = 1 } ^ { T }$ with $g _ { t } : \mathcal { X } ^ { t }  [ 0 , 1 ]$ . The prediction $g _ { t } { \big ( } x _ { 1 : t } { \big ) }$ may use the current instance $x _ { t }$ and all preceding instances. The expert receives no target values. For a fixed sequence, write

$$
g [ x _ { 1 : T } ] : = ( g _ { t } ( x _ { 1 : t } ) ) _ { t = 1 } ^ { T } , \qquad f [ x _ { 1 : T } ] : = ( f ( x _ { t } ) ) _ { t = 1 } ^ { T } .
$$

Definition 4 (Global dynamic $\ell _ { p }$ cover). Let $\mathcal { F } \subseteq [ 0 , 1 ] ^ { \mathcal { X } } , T \geq 1$ , and $\varepsilon > 0$ . A finite family G of deterministic causal experts is a global dynamic $\ell _ { p }$ cover of $\mathcal { F }$ at accuracy ε and horizon T if

$$
\forall x _ { 1 : T } \in \mathcal { X } ^ { T } \forall f \in \mathcal { F } \exists g \in \mathcal { G } : \qquad \rho _ { p , T } \big ( g [ x _ { 1 : T } ] , f [ x _ { 1 : T } ] \big ) \leq \varepsilon .\tag{18}
$$

The global dynamic covering number $N _ { p } ^ { \mathrm { d y n } } ( \mathcal { F } , \varepsilon , T )$ is the least cardinality of such a finite family, or ∞ $i f$ none exists.

The entire family, including its predictions on every possible prefix, is fixed before the sequence is chosen. The witnessing expert may depend on both the target and the complete sequence, but it is a single member of the family used throughout that sequence. There is no requirement that this member be identifiable online. Experts need not arise from functions in ${ \mathcal F } .$ . In particular, integral covering means selecting finitely many experts; it imposes no integrality constraint on their predictions.

The family may depend on ${ \mathcal { F } } , p , \varepsilon$ , and T. A bound independent of the horizon therefore asserts a uniform bound on its size for all $T ,$ while allowing a diferent family at each horizon.

## 3.3 Sharp Upper and Lower Bounds

Theorem 5 (Global dynamic $\ell _ { p }$ upper bound). There is a universal constant $C \geq 1$ such that, for every $1 \leq p < \infty ,$ every integer $T \geq 1$ , every $0 < \varepsilon < 1 / 2$ , and every nonempty $\mathcal { F } \subseteq [ 0 , 1 ] ^ { \mathcal { X } }$ with $d = \mathrm { s f a t } _ { \varepsilon / 3 2 } ( \mathcal { F } ) < \infty$

$$
N _ { p } ^ { \mathrm { d y n } } ( { \mathcal { F } } , \varepsilon , T ) \leq \left( { \frac { C } { \varepsilon ^ { p } } } \right) ^ { d } .\tag{19}
$$

The constant C is independent $o f p , d , T , \varepsilon $ , and the class. $I f d = 0$ , the covering number is exactly one.

The zero-dimensional case follows directly from the definition. If $\mathrm { s f a t } _ { \eta } ( \mathcal { F } ) = 0$ , no two values at any one instance can difer by at least $\eta ,$ since they would shatter a depth-one tree. Hence

$$
\operatorname* { s u p } _ { f \in { \mathcal { F } } } f ( x ) - \operatorname* { i n f } _ { f \in { \mathcal { F } } } f ( x ) \leq \eta \qquad ( x \in { \mathcal { X } } ) .
$$

The expert that predicts the midpoint of this interval at the current instance has pointwise error at most $\eta / 2$ against every target. Taking $\eta = \varepsilon / 3 2$ proves the assertion.

Theorem 6 (Matching $\ell _ { p }$ lower bound). For every $1 \leq p < \infty$ , every integer $d \geq 1$ , and every $0 < \varepsilon < 2 ^ { - 1 - 1 / p }$ , there are a finite domain $x ,$ a binary class $\mathcal { H } \subseteq \{ 0 , 1 \} ^ { \mathcal { X } }$ with $\mathrm { s f a t } _ { \varepsilon / 3 2 } ( \mathcal { H } ) = d ;$ and a horizon $T \geq 1$ such that

$$
N _ { p } ^ { \mathrm { d y n } } ( \mathcal { H } , \varepsilon , T ) > \left( \frac { 1 } { 2 ^ { p + 1 } \varepsilon ^ { p } } \right) ^ { d } .\tag{20}
$$

The lower bound holds even when covering is required only on one fixed instance sequence and the covering predictions are arbitrary vectors in $[ 0 , 1 ] ^ { T }$

The construction uses indicators of afine hyperplanes in $\mathbb { F } _ { q } ^ { d }$ and lists the points of this domain once. For a suitable power of two $q ,$ the class has Littlestone dimension exactly d and at least $q ^ { d }$ members, whose traces are pairwise more than 2ε apart in $\rho _ { p , T }$ . Each covering vector can therefore approximate at most one trace. The class and the sequence may depend on $p , d ,$ and $\varepsilon ;$ the lower bound is a worst-case assertion, not a lower bound for every class of dimension $d .$

For a nonnegative integer d, define the worst-case quantity

$$
\mathfrak M _ { d , p } ( \varepsilon ) : = \operatorname* { s u p } _ { T \geq 1 } \operatorname* { s u p } _ { \begin{array} { c } { \mathcal { X } , \varnothing \neq \mathcal { F } \subseteq [ 0 , 1 ] ^ { \mathcal { X } } \colon } \\ { \operatorname { s f a t } _ { \varepsilon / 3 2 } ( \mathcal { F } ) \leq d } \end{array} } N _ { p } ^ { \mathrm { d y n } } ( \mathcal { F } , \varepsilon , T ) . \hfill\tag{21}
$$

Theorems 5 and 6 give, for $d \geq 1$ and $0 < \varepsilon < 2 ^ { - 1 - 1 / p }$ 2

$$
\left( { \frac { 1 } { 2 ^ { p + 1 } \varepsilon ^ { p } } } \right) ^ { d } < { \mathfrak { M } } _ { d , p } ( \varepsilon ) \leq \left( { \frac { C } { \varepsilon ^ { p } } } \right) ^ { d } .\tag{22}
$$

In particular, since $2 ^ { 1 + 1 / p } \leq 4$ and $C ^ { 1 / p } \leq C$ , for every $1 \leq p < \infty , d \geq 1$ , and $0 < \varepsilon < 1 / 4$

$$
\left( { \frac { 1 } { 4 \varepsilon } } \right) ^ { p d } < { \mathfrak { M } } _ { d , p } ( \varepsilon ) \leq \left( { \frac { C } { \varepsilon } } \right) ^ { p d } .\tag{23}
$$

For $d = 0 , \mathfrak { M } _ { 0 , p } ( \varepsilon ) = 1$ on the same range. Thus (23), with a non-strict lower inequality, also includes $d = 0$ and yields Theorem 1 with $c = 1 / 4$

## 3.4 Binary Classes and Sequential Covering

We record two consequences with their proofs. The first gives binary-valued experts for binary classes; the second restricts the global family to a prescribed instance tree.

Binary Hamming covers. For $u , v \in \{ 0 , 1 \} ^ { T }$ , let

$$
d _ { \mathrm { H } , T } ( u , v ) : = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbf { 1 } \{ u _ { t } \neq v _ { t } \}
$$

be normalized Hamming distance. For $\mathcal { H } \subseteq \{ 0 , 1 \} ^ { \mathcal { X } }$ , define $N _ { \mathrm { b i n } } ^ { \mathrm { d y n } } ( \mathcal { H } , \delta , T )$ as in Definition 4, with binary-valued experts $g _ { t } \ : \mathcal { X } ^ { t } \  \ \{ 0 , 1 \}$ and with $d _ { \mathrm { H } , T } \big ( g [ x _ { 1 : T } ] , h [ x _ { 1 : T } ] \big ) \ \leq \ \delta$ in place of the $\ell _ { p }$ requirement. Requiring binary outputs distinguishes this number from $N _ { 1 } ^ { \mathrm { d y n } }$ , which permits real outputs.

Corollary 7 (Sharp binary dynamic covering). There is a universal constant $C _ { \mathrm { b i n } } \geq 1$ such that, for every binary class H with $d = \operatorname { L d i m } ( \mathcal { H } ) < \infty$ , every $T \geq 1$ , and every $0 < \delta < 1$

$$
N _ { \mathrm { b i n } } ^ { \mathrm { d y n } } ( \mathcal { H } , \delta , T ) \leq \left( \frac { C _ { \mathrm { b i n } } } { \delta } \right) ^ { d } .\tag{24}
$$

Conversely, for every integer $d \geq 1$ and $0 < \delta < 1 / 4$ , some binary class of Littlestone dimension d and some horizon satisfy

$$
N _ { \mathrm { b i n } } ^ { \mathrm { d y n } } ( \mathcal { H } , \delta , T ) > \left( \frac { 1 } { 4 \delta } \right) ^ { d } .\tag{25}
$$

Proof. Apply Theorem 5 with $p = 1$ and $\varepsilon = \delta / 2$ . The sequential fat-shattering dimension at scale $\delta / 6 4$ equals $d ,$ so the resulting real-valued family has size at most $( 2 C / \delta ) ^ { d }$ . Round each prediction to $\bar { g } _ { t } ( x _ { 1 : t } ) = \mathbf { 1 } \{ g _ { t } ( x _ { 1 : t } ) \geq 1 / 2 \}$ . For a binary target,

$$
\mathbf { 1 } \{ \bar { g } _ { t } ( x _ { 1 : t } ) \neq h ( x _ { t } ) \} \subseteq 2 | g _ { t } ( x _ { 1 : t } ) - h ( x _ { t } ) | .
$$

The rounded family is causal and has normalized Hamming error at most $\delta ,$ proving (24) with $C _ { \mathrm { b i n } } = 2 C$ . For the lower bound, apply Theorem 6 with $p = 1$ and $\varepsilon = \delta$ , and use $N _ { \mathrm { b i n } } ^ { \mathrm { d y n } } \geq N _ { 1 } ^ { \mathrm { d y n } }$

In particular, Corollary 7 resolves the global approximate covering question posed by Alon, Ben-Eliezer, Dagan, Moran, Naor, and Yogev $[ \mathrm { A l o } + 2 1 ]$ and the dynamic Haussler problem of Malliaris and Moran [MM24]. For binary vectors, $\rho _ { p , T } ( u , v ) ^ { p } = d _ { \mathrm { H } , T } ( u , v )$ . For $0 < \varepsilon < 1$ , taking $\delta = \varepsilon ^ { p }$ in (24) therefore also gives a global cover by binary experts with normalized $\ell _ { p }$ accuracy ε and size at most $( C _ { \mathrm { { b i n } } } / \varepsilon ^ { p } ) ^ { d }$

Covers on a fixed instance tree. For a tree $\mathbf { x } = ( \mathbf { x } _ { t } ) _ { t = 1 } ^ { T }$ and a sign path $\sigma \in \{ - 1 , + 1 \} ^ { T }$ , write $\mathbf { x } ( \sigma ) = ( \mathbf { x } _ { t } ( \sigma _ { < t } ) ) _ { t = 1 } ^ { T }$ . If x is an instance tree, $f ( \mathbf { x } ( \sigma ) )$ ) denotes the vector obtained by applying f to each coordinate. The normalized sequential $\ell _ { p }$ covering number $N _ { p } ^ { \mathrm { s e q } } ( \mathcal { F } , \varepsilon , \mathbf { x } )$ is the least cardinality of a finite family V of $[ 0 , 1 ]$ -valued prediction trees of depth T such that

$$
\forall f \in { \mathcal { F } } \forall \sigma \in \{ - 1 , + 1 \} ^ { T } \exists \mathbf { v } \in V : \qquad \rho _ { p , T } \big ( \mathbf { v } ( \sigma ) , f ( \mathbf { x } ( \sigma ) ) \big ) \leq \varepsilon .\tag{26}
$$

As before, the covering number is ∞ if no finite family exists. Here V may depend on the full tree x, as in the sequential covering definition of Rakhlin, Sridharan, and Tewari [RST15b]. Set

$$
N _ { p } ^ { \mathrm { s e q } } ( \mathcal { F } , \varepsilon , T ) : = \operatorname* { s u p } _ { \bf x } N _ { p } ^ { \mathrm { s e q } } ( \mathcal { F } , \varepsilon , { \bf x } ) ,
$$

where the supremum ranges over all instance trees of depth T.

Corollary 8 (Sequential $\ell _ { p }$ covering). For every $1 \leq p < \infty , T \geq 1$ , and $\varepsilon > 0$

$$
N _ { p } ^ { \mathrm { s e q } } ( \mathcal { F } , \varepsilon , T ) \leq N _ { p } ^ { \mathrm { d y n } } ( \mathcal { F } , \varepsilon , T ) .\tag{27}
$$

Consequently, $i f 0 < \varepsilon < 1 / 2$ and $d = \mathrm { s f a t } _ { \varepsilon / 3 2 } ( \mathcal { F } ) < \infty ,$ then

$$
N _ { p } ^ { \mathrm { s e q } } ( { \mathcal { F } } , \varepsilon , T ) \leq \left( { \frac { C } { \varepsilon ^ { p } } } \right) ^ { d } .\tag{28}
$$

The worst-case accuracy exponent pd is sharp for these sequential covering numbers as well.

Proof. Fix an instance tree x and a global family $\mathcal { G }$ . For each $g \in { \mathcal { G } }$ , define a prediction tree by restriction:

$$
\mathbf { v } _ { t } ^ { g } ( \sigma _ { < t } ) : = g _ { t } \big ( \mathbf { x } _ { 1 } ( \emptyset ) , \mathbf { x } _ { 2 } ( \sigma _ { 1 } ) , \dots , \mathbf { x } _ { t } ( \sigma _ { < t } ) \big ) .
$$

For every target $f$ and path $\sigma _ { \mathrm { : } }$ , the global covering property applied to the instance sequence $\mathbf { x } ( \sigma )$ supplies a member of this family satisfying (26). Its cardinality is at most |G|. Taking the minimum over global covers and then the supremum over instance trees proves (27); if no finite global cover exists, the inequality is immediate. Theorem 5 gives (28).

For sharpness, take the fixed sequence from Theorem 6 and set $\mathbf { x } _ { t } ( \sigma _ { < t } ) = x _ { t }$ at every node of level t. Fixing any one sign path turns each prediction tree into one covering vector for that sequence. The fixed-sequence lower bound therefore gives

$$
N _ { p } ^ { \mathrm { s e q } } ( \mathcal { H } , \varepsilon , T ) > \left( \frac { 1 } { 2 ^ { p + 1 } \varepsilon ^ { p } } \right) ^ { d }
$$

for the class in that theorem and its stated parameter range.

In particular, the case $p = 2$ resolves the horizon-independent sequential $\ell _ { 2 }$ covering question posed by Rakhlin, Sridharan, and Tewari [RST15b].

## 4 Correction Schedules and Adaptive Sampling

We establish the deterministic and probabilistic ingredients used to prove Proposition 2. We first justify the reduction to a separated class and construct finite families of rank-decreasing correction schedules. We then encode prediction errors by partial binary functions and prove a sampling estimate that is uniform over such functions, even when the inputs depend on past sampling choices. The choice of sampling parameters and the construction of the master and replay histories are deferred to Section 5.

## 4.1 Separation and Rank-Decreasing Centers

The reduction in Section 2.1 uses a uniformly separated subclass, chosen independently of the instance sequence. We first give the existence argument, including the case of an infinite function class.

Lemma 9 (Uniformly separated reduction). For every nonempty ${ \mathcal { F } } \subseteq [ 0 , 1 ] ^ { \mathcal { X } }$ and every $\eta > 0$ there is a nonempty subclass $\mathcal { H } \subseteq \mathcal { F }$ such that

$$
\| h - h ^ { \prime } \| _ { \infty } > \eta \quad ( h \neq h ^ { \prime } , \ h , h ^ { \prime } \in \mathcal { H } ) , \qquad \forall f \in \mathcal { F } \exists h \in \mathcal { H } : \| f - h \| _ { \infty } \le \eta .\tag{29}
$$

Moreover, sfa $\mathrm { t } _ { \eta } ( \mathcal { H } ) \leq \mathrm { s f a t } _ { \eta } ( \mathcal { F } )$

Proof. Order the uniformly greater-than-η separated subclasses of $\mathcal { F }$ by inclusion. The union of a chain is still separated: any two members of the union belong together to one class in the chain. Zorn’s lemma therefore supplies a maximal separated subclass $\mathcal { H } .$ . It is nonempty, since a singleton could otherwise be added. If some $f \in { \mathcal { F } }$ had distance greater than η from every member of ${ \mathcal { H } } ,$ adjoining f would contradict maximality. This proves the approximation property. The dimension inequality follows from monotonicity under inclusion. ■

Together with the calculation in (8), Lemma 9 reduces the main upper bound to Proposition 2. For the proof of that proposition, fix a nonempty separated class H and parameters $\eta , R , T ,$ d satisfying its hypotheses. The rank of a class V means $\operatorname { s f a t } _ { \eta } ( V )$ . Separation is needed for the following consequence of rank zero, rather than for the schedule construction itself.

Lemma 10 (Rank-zero classes). $I f \varnothing \neq V \subseteq [ 0 , 1 ] ^ { \chi }$ has rank zero, then

$$
\operatorname* { s u p } _ { f \in V } f ( x ) - \operatorname* { i n f } _ { f \in V } f ( x ) \leq \eta \qquad ( x \in \mathcal { X } ) .\tag{30}
$$

If, in addition, V is a subclass of a uniformly greater-than-η separated class, then V is a singleton.

Proof. Two functions whose values at an instance difer by at least η would shatter a depth-one tree, using the midpoint of their values as the threshold. Thus every pairwise diference at a fixed instance is less than η. Taking the supremum of these diferences gives (30); the supremum need not be attained. If distinct h, $h ^ { \prime } \in V$ had $\| h - h ^ { \prime } \| _ { \infty } > \eta$ , some instance would satisfy $| h ( x ) - h ^ { \prime } ( x ) | > \eta .$ contradicting the same argument. ■

In particular, a separated class of rank zero is covered exactly by its unique member. This proves the $d = 0$ case of Proposition 2. The remaining local lemmas apply to arbitrary finite-rank classes, so they may also be used for version spaces arising later in the construction.

Rank-decreasing centers and sign updates. We need a prediction rule determined by the current version space and instance, with the property that a suficiently separated correction strictly lowers rank. The elementary operation behind this rule is to attach two shattered trees below a common root.

Lemma 11 (Gluing shattered trees). Let $V \subseteq [ 0 , 1 ] ^ { \mathcal { X } } , \ x \in \mathcal { X }$ , and $z _ { 1 } - z _ { 0 } \ge \eta$ . If both classes

$$
\{ f \in V : f ( x ) \leq z _ { 0 } \} , \qquad \{ f \in V : f ( x ) \geq z _ { 1 } \}
$$

have rank at least an integer $r \geq 0$ , then sfa $\mathrm { t } _ { \eta } ( V ) \geq r + 1$

Proof. Attach depth-r shattered trees of the two subclasses below a new root at x with threshold $( z _ { 0 } + z _ { 1 } ) / 2 $ the assumption $z _ { 1 } - z _ { 0 } \ge \eta$ guarantees the required margin $\eta / 2$ at the root. ■

For a nonempty class V of finite rank $r = \mathrm { s f a t } _ { \eta } ( V )$ , define the two critical thresholds and their center by

$$
\begin{array} { r l } & { \ell ( V , x ) : = \operatorname* { i n f } \big \{ z \in [ 0 , 1 ] : \mathrm { s f a t } _ { \eta } ( \{ f \in V : f ( x ) \le z \} ) = r \big \} , } \\ & { u ( V , x ) : = \operatorname* { s u p } \big \{ z \in [ 0 , 1 ] : \mathrm { s f a t } _ { \eta } ( \{ f \in V : f ( x ) \ge z \} ) = r \big \} , } \\ & { a ( V , x ) : = \frac 1 2 \big ( \ell ( V , x ) + u ( V , x ) \big ) . } \end{array}\tag{31}
$$

Both defining sets are nonempty: the lower fiber equals V at $z = 1$ , and the upper fiber equals V at $z = 0$ . Hence the thresholds and the center belong to [0, 1]. We do not assume that the infimum or supremum is attained.

Lemma 12 (Rank-decreasing center). For every such V and x, the two sign-update classes

$$
V _ { x } ^ { - } : = \{ f \in V : f ( x ) \leq a ( V , x ) - \eta \} , \qquad V _ { x } ^ { + } : = \{ f \in V : f ( x ) \geq a ( V , x ) + \eta \}\tag{32}
$$

have rank at most $r - 1$ , with the convention sf $\mathrm { a t } _ { \eta } ( \mathcal { O } ) = - 1$

Proof. Abbreviate the thresholds by $\ell ,$ u and the center by a. Suppose first that $u - \ell > \eta .$ Choose $\ell < z _ { 0 } < z _ { 1 } < u$ with $z _ { 1 } - z _ { 0 } \geq \eta .$ By the definition of $\ell ,$ some admissible lower threshold is smaller than $z _ { 0 } ;$ monotonicity therefore makes the lower fiber at $z _ { 0 }$ have rank r. The corresponding argument for u makes the upper fiber at $z _ { 1 }$ have rank $r .$ Lemma 11 would give rank at least $r + 1$ for $V ,$ a contradiction. Consequently $u - \ell \leq \eta$ , and

$$
a - \eta \le \ell - \eta / 2 < \ell , \qquad a + \eta \ge u + \eta / 2 > u .
$$

If $a - \eta < 0$ , the lower sign-update class is empty. Otherwise, its threshold is strictly below $\ell ,$ so its rank cannot be $r .$ Similarly, the upper class is either empty because $a + \eta > 1$ , or has rank smaller than r because its threshold exceeds u. Ranks are integers, which proves both bounds. ■

A sign update replaces the current nonempty version space V by $V _ { x } ^ { - }$ or $V _ { x } ^ { + }$ . A nonempty update is called valid. Starting from a class of rank at most $^ { r , }$ any sequence of i valid updates leaves a version space of rank at most $r - i ;$ in particular, there can be at most r valid updates. The update is specified by its sign and uses the fixed center rule in (31), not a target value or a discretized approximation to that value.

## 4.2 Correction Schedules and Pointwise Covers

We next turn prescribed sign updates into label-free causal simulations. Fix an initial nonempty version space $V _ { 0 }$ of rank at most r and a positive integer m. A selection history specifies which rounds are selected. A selected round has an ordinal $s ,$ meaning that it is the sth selected round, regardless of its position in the original instance sequence.

A correction schedule of depth i is a list

$$
J = ( ( s _ { 1 } , \sigma _ { 1 } ) , \ldots , ( s _ { i } , \sigma _ { i } ) ) , \qquad 1 \leq s _ { 1 } \leq \cdots \leq s _ { i } \leq m , \qquad \sigma _ { j } \in \{ - , + \} .\tag{33}
$$

The empty list is the depth-zero schedule. Repeated ordinals are allowed because more than one rank-decreasing correction may be needed at a single instance. The classical Sauer–Shelah– Perles bound [Sau72; She72] has dynamic analogues [BPS09; MM24]. Our encoding similarly uses dimension-decreasing updates, but accommodates signed corrections and repeated ordinals for real-valued prediction.

Simulation and prediction timing. Every simulation starts at $V _ { 0 }$ . On round $t ,$ after observing $x _ { t }$ but before the current selection is revealed, its provisional prediction is $a ( V , x _ { t } )$ , using the version space left by the preceding round. If this round has selected ordinal $s ,$ the simulation executes all entries of $J$ with $s _ { j } ~ = ~ s$ , in their listed order. Each entry applies the indicated sign update, and the center is recomputed from the new version space before the next entry is executed. The simulation emits its final prediction only after all these updates. On an unselected round it makes no update, and its final prediction equals its provisional prediction. We denote the provisional and final predictions by $a _ { t } ^ { J }$ and $y _ { t } ^ { J }$ , respectively.

An update to an empty fiber makes the simulation invalid. From that update onward, it remains in an absorbing invalid state and emits zero, including on the current round. If the selection history ends before a prescribed ordinal is reached, the corresponding update is not executed. The same rule is also defined on selection histories longer than m: there are simply no prescribed updates at ordinals larger than $m$ . These conventions ensure that every formal schedule has an output on every outcome, not only on outcomes where the intended sampling cap is respected.

The simulation uses only the instance history, the selection history, and its fixed schedule. If the selection policy is a deterministic causal function of the instances, the simulation defines a deterministic causal expert. Under random selection, its outputs additionally depend on the selection history, while remaining causal. We will first analyze these simulations and later obtain deterministic experts by the argument in Section 8.

A valid schedule is born once all its prescribed corrections have occurred. The empty schedule is born at time zero. After birth, its version space remains fixed, although its predictions may still vary with the current instance. A born depth-i schedule has residual rank at most $r \mathrm { ~ - ~ } i$ Moreover, let $q _ { t }$ be the number of selected rounds up to round t. Removing from a schedule J all entries with selection ordinals greater than $q _ { t }$ leaves its version spaces and predictions unchanged on rounds $1 , \ldots , t .$ The resulting prefix schedule has completed all its prescribed corrections by round $t ,$ provided they are valid.

The formal schedule family. The number of weakly increasing ordinal lists of length i in $[ m ]$ is $\binom { m + i - 1 } { i }$ , and every list admits $2 ^ { i }$ sign assignments. Thus, writing $n _ { 0 } ( m ) = 1$ , the exact counts are

$$
n _ { i } ( m ) : = 2 ^ { i } { \binom { m + i - 1 } { i } } , \qquad Q _ { r } ( m ) : = \sum _ { i = 0 } ^ { r } n _ { i } ( m ) .\tag{34}
$$

These counts include invalid schedules, schedules that never become born, and schedules whose outputs coincide. Consequently $Q _ { r } ( m )$ is an upper bound on the number of distinct prediction rules, with no need to recognize or remove such duplications.

Lemma 13 (Pointwise schedule cover). Let $\emptyset \neq V _ { 0 } \subseteq [ 0 , 1 ] ^ { \chi }$ satisfy sfa $t _ { \eta } ( V _ { 0 } ) \leq r$ . With every round selected, the schedules of depth at most r form one family of at most $Q _ { r } ( m )$ deterministic causal experts such that, on every instance sequence of length at most $m ,$ , each target in $V _ { 0 }$ is approximated by one expert with pointwise error at most η. More generally, on any selection history with at most m selected rounds, a schedule in the same formal family attains error at most η on every selected round.

Proof. Fix a target $f \in V _ { 0 }$ and an instance-and-selection history. Consider the following reference procedure, which is used only to find a witnessing schedule. It starts from $V _ { 0 }$ and does not update on unselected rounds. At a selected instance $x _ { i }$ while $| f ( x ) - a ( V , x ) | > \eta$ , it updates toward $f ( x )$ it uses the plus fiber if $f ( x ) > a ( V , x ) + \eta$ and the minus fiber otherwise. It recomputes the center and repeats at the same instance until the error is at most η.

Every update retains $f ,$ and Lemma 12 decreases the rank by at least one. Since the version space stays nonempty, at most $r$ updates occur over the entire history. Thus, the correction loop at each selected round terminates with prediction error at most $\eta .$ Record the selected ordinal and sign of each update, retaining their order when an ordinal occurs repeatedly. The resulting list is a schedule of depth at most $r$ in (33). Its ordinary simulation has exactly the reference procedure’s states and final predictions, by induction on rounds and on updates within a round. It therefore has error at most η at every selected round.

When all rounds are selected, this is the asserted pointwise cover. The formal family is fixed independently of both the target and the instance sequence. Only the choice of its witnessing member used the target values; no expert in the family receives those values. ■

We record both the size estimate and a convolution identity for these counts. The latter will be used when counting the continuation capacity in Section 8.

Lemma 14 (Schedule-count estimates). For integers $m , r \geq 1$ 2

$$
Q _ { r } ( m ) \leq 2 ^ { r } { \binom { m + r } { r } } \leq \left[ 2 e \left( 1 + { \frac { m } { r } } \right) \right] ^ { r } .\tag{35}
$$

For every integer $\ell \geq 0$

$$
\sum _ { i = 0 } ^ { \ell } n _ { i } ( m ) n _ { \ell - i } ( m ) = n _ { \ell } ( 2 m ) .\tag{36}
$$

Consequently, for every integer $d \geq 1$ ，

$$
\sum _ { i = 0 } ^ { d - 1 } n _ { i } ( m ) Q _ { d - i } ( m ) \leq Q _ { d } ( 2 m ) .\tag{37}
$$

Proof. The hockey-stick identity gives

$$
Q _ { r } ( m ) \leq 2 ^ { r } \sum _ { i = 0 } ^ { r } { \binom { m + i - 1 } { i } } = 2 ^ { r } { \binom { m + r } { r } } .
$$

Using $\binom { N } { r } \leq ( e N / r ) ^ { r }$ proves (35). For the convolution, each of the m selection ordinals can carry any number $j \geq 0$ of corrections, with $2 ^ { j }$ possible sign assignments. Thus each ordinal contributes the generating function

$$
\sum _ { j = 0 } ^ { \infty } 2 ^ { j } z ^ { j } = ( 1 - 2 z ) ^ { - 1 } ,
$$

and multiplying over all m ordinals gives

$$
\sum _ { i = 0 } ^ { \infty } n _ { i } ( m ) z ^ { i } = ( 1 - 2 z ) ^ { - m } .
$$

Squaring this identity and comparing the coeficient of $z ^ { \ell }$ gives (36). Summing it over $\ell \leq d$ yields

$$
\sum _ { i , j \geq 0 \atop i + j \leq d } n _ { i } ( m ) n _ { j } ( m ) = Q _ { d } ( 2 m ) .
$$

The sum in (37) contains all of these nonnegative terms except the one indexed by $( i , j ) = ( d , 0 )$ ■

In particular, the all-round cover has size $Q _ { d } ( T ) \leq [ 2 e ( 1 + T / d ) ] ^ { d }$ when $d \geq 1$ . This will handle the short-horizon regime directly. For larger horizons, the same schedule family will be used on shorter sampled lists. In either case, the count depends on selected ordinals and signs, without an additional factor for discretizing the target values at scale η.

## 4.3 Partial Binary Completion and Adaptive Sampling

The sampling argument encodes one-sided prediction errors using partial binary labels: discrepancies at most η receive label 0, those at least $2 \eta$ receive label 1, and intermediate values receive the undefined symbol ∗. This gap lets us control the resulting partial binary classes by sequential fat-shattering dimension at scale η. We first define these classes, then construct the finite families of binary rules needed to analyze their sampled values.

Let $\mathcal { Z }$ be a domain. A partial binary class is a subset $\mathcal { A } \subseteq \{ 0 , 1 , * \} ^ { \mathcal { Z } }$ . Such a class shatters a binary instance tree $z _ { t } : \{ 0 , 1 \} ^ { t - 1 }  \mathcal { Z } , t \in [ s ] .$ , if

$$
\forall b \in \{ 0 , 1 \} ^ { s } \exists \phi \in { \mathcal { A } } \forall t \in [ s ] : \quad \phi ( z _ { t } ( b _ { < t } ) ) = b _ { t } .
$$

Thus ∗ realizes neither edge of the tree. Define Ldim(A) to be the largest shattered depth, or infinity when these depths are unbounded. As before, a nonempty class has dimension at least zero, and the empty class has dimension −1. On total binary classes this is ordinary Littlestone dimension.

Partial error classes. For $f \in \mathcal H$ and $( x , a ) \in \mathcal { X } \times [ 0 , 1 ]$ , define

$$
\phi _ { f } ^ { + } ( x , a ) : = \left\{ \begin{array} { l l } { 0 , } & { f ( x ) - a \leq \eta , } \\ { 1 , } & { f ( x ) - a \geq 2 \eta , } \\ { * , } & { \eta < f ( x ) - a < 2 \eta , } \end{array} \right. \quad \phi _ { f } ^ { - } ( x , a ) : = \left\{ \begin{array} { l l } { 0 , } & { a - f ( x ) \leq \eta , } \\ { 1 , } & { a - f ( x ) \geq 2 \eta , } \\ { * , } & { \eta < a - f ( x ) < 2 \eta . } \end{array} \right.\tag{38}
$$

Write $\mathcal { A } ^ { \pm } : = \{ \phi _ { f } ^ { \pm } : f \in \mathcal { H } \}$ . A definite one records a one-sided error of size at least $2 \eta ;$ a nonzero value, including ∗, records an error strictly greater than $\eta$ in that direction.

Lemma 15 (Dimension of the partial error classes). Both partial classes satisfy

$$
\operatorname { L d i m } ( A ^ { + } ) , \quad \operatorname { L d i m } ( A ^ { - } ) \leq \operatorname { s f a t } _ { \eta } ( { \mathcal { H } } ) .\tag{39}
$$

Proof. Suppose a depth-s tree of pairs $( x , a )$ is shattered by $\mathcal { A } ^ { + }$ . Replacing each node $( x , a )$ with instance x and threshold $a + 3 \eta / 2$ yields an η-fat-shattered tree, since labels 0 and 1 imply $f ( x ) \leq a + \eta$ and $f ( x ) \geq a + 2 \eta$ , respectively. The argument for A<sup>−</sup> is identical, using threshold $a - 3 \eta / 2$ and interchanging the two branches at each node. ■

Completion by finitely many causal rules. For a partial class $W$ and $b \in \{ 0 , 1 \}$ , write $W _ { z } ^ { b } : = \{ \phi \in W : \phi ( z ) = b \}$ . For nonempty W of finite Littlestone dimension, the standard optimal algorithm $\mathrm { ( S O A ) }$ [Lit88] prediction soa $( W , z )$ is the label maximizing Ldim $( W _ { z } ^ { b } )$ , with ties broken in favor of zero. As in Lemma 11, the two defined-label fibers cannot both have the rank of $W$ Since SOA chooses a maximum-rank fiber,

$$
\operatorname { L d i m } \left( W _ { z } ^ { 1 - \operatorname { s o a } ( W , z ) } \right) \leq \operatorname { L d i m } ( W ) - 1 .\tag{40}
$$

Undefined values belong to neither fiber.

Lemma 16 (Partial SOA completion). Let $A \subseteq \{ 0 , 1 , * \} ^ { \mathcal { Z } }$ have Littlestone dimension at most $r \geq 0$ , and let $n \geq 1$ be an integer. There is a family of at most

$$
S _ { r } ( n ) : = \sum _ { i = 0 } ^ { \operatorname* { m i n } \{ r , n \} } { \binom { n } { i } }
$$

deterministic causal prediction rules $\boldsymbol { v } = ( v _ { t } ) _ { t = 1 } ^ { n }$ , with $v _ { t } : \mathcal { Z } ^ { t }  \{ 0 , 1 \}$ , such that, on every sequence of length at most $n ,$ each $\phi \in { \mathcal { A } }$ is matched at all its defined positions by one rule.

Proof. If A is empty, a constant rule sufices. Otherwise, for each $J \subseteq [ n ]$ with $| J | \le r$ , define a rule starting from $W = A$ . Outside $^ { J , }$ it outputs soa $( W , z )$ and makes no update. At a round in $^ { J , }$ it outputs the opposite label and restricts W to that opposite-label fiber. An empty restriction makes the rule invalid, with output zero from that round onward. Thus every $J$ specifies a total binary causal rule without access to partial target values. There are $S _ { r } ( n )$ such sets $J .$

To find a rule for a particular target $\phi$ and sequence, use a reference procedure that restricts its version space only when $\phi ( z )$ is defined and difers from the current SOA prediction. It records each such round and restricts to $W _ { z } ^ { \phi ( z ) }$ . It leaves W unchanged when the prediction is correct or when $\phi ( z ) = *$ . Every restriction retains $\phi$ and, by (40), strictly decreases rank. The recorded set $J$ therefore has size at most r. Induction on rounds shows that the rule indexed by this $J$ has exactly the reference procedure’s states and matches every defined target value. ■

The lemma constructs causal completions along input histories; it does not choose one total static completion of each partial target. Its codes record at most one binary reversal per round and require no sign, which explains the count $S _ { r } ( n )$ rather than $Q _ { r } ( n )$ . We will also use the elementary estimate

$$
S _ { r } ( n ) \leq \left( { \frac { e n } { r } } \right) ^ { r } \qquad ( 1 \leq r \leq n ) .\tag{41}
$$

Indeed, with $x = r / n \leq 1$ ，

$$
x ^ { r } S _ { r } ( n ) \leq \sum _ { i = 0 } ^ { r } { \binom { n } { i } } x ^ { i } \leq ( 1 + x ) ^ { n } \leq e ^ { n x } = e ^ { r } .
$$

For $r = 0$ the count is one, and for $r \geq n$ it is $2 ^ { n }$

Adaptive low-hit sampling. Sampling against inputs that react to earlier sample selections has been studied in [BY20; Alo+21]. We now bound the probability that a partial target has many definite ones but few sampled nonzero values. A completion rule must match all defined labels but may output either label at an undefined position. Consequently, the number of sampled ones predicted by the completion is at most the number of sampled nonzero values of the partial target.

Order of information. Let $\xi _ { 1 } , \dots , \xi _ { T }$ be independent Bernoulli-γ bits. The input $z _ { t }$ is chosen before $\xi _ { t }$ is revealed and may depend on $\xi _ { < t }$ and an independent finite-support auxiliary seed ω. Equivalently, $z _ { t }$ is a deterministic function of $\left( \xi _ { < t } , \omega \right)$ . We call this a predictable input stream for the Bernoulli sample. Fixing ω leaves the product law of the sampling bits unchanged.

Lemma 17 (Adaptive low-hit sampling). Let $\mathcal { A } \subseteq \{ 0 , 1 , * \} ^ { \mathcal { Z } }$ have Littlestone dimension at most an integer $r \geq 0$ . Let $z _ { 1 } , \dots , z _ { T }$ be a predictable input stream for independent Bernoulli-γ bits $\xi _ { 1 } , \dots , \xi _ { T }$ , where $0 < \gamma \le 1 / 2$ . For each $\phi \in { \mathcal { A } }$ , define

$$
L _ { \phi } : = \sum _ { t = 1 } ^ { T } \mathbf { 1 } \{ \phi ( z _ { t } ) = 1 \} , \qquad H _ { \phi } : = \sum _ { t = 1 } ^ { T } \xi _ { t } \mathbf { 1 } \{ \phi ( z _ { t } ) \neq 0 \} .\tag{42}
$$

Let $q \geq 0$ and $B \geq 2 q$ satisfy $\gamma B \geq 1 6 \log 2$ . With $M = \lceil 4 \gamma T \rceil$ and $c _ { 0 } = \log ( 3 / 2 ) / 4$ , one has

$$
\begin{array} { r } { \operatorname* { P r } \{ \exists \phi \in \mathcal { A } : L _ { \phi } \geq B , \ H _ { \phi } \leq q \} \leq 2 \left( S _ { r } ( M ) 2 ^ { q } e ^ { - c _ { 0 } \gamma B } + e ^ { - 2 \gamma T / 3 } \right) . } \end{array}\tag{43}
$$

Proof. Fix a value of the auxiliary seed $\omega$ in its support. The sampling bits remain independent Bernoulli-γ variables, and each $z _ { t }$ is now a deterministic function of $\xi _ { < t }$ . All probabilities below are conditional on this fixed seed until the final averaging step. Call a round red when $\xi _ { t } = 1$ , and let E denote the event on the left of (43).

A second sample for comparison. We introduce a second sample to detect definite ones outside the red sample. After the red bits have been generated, independently color each nonred round blue with probability $\gamma / ( 1 - \gamma )$ . This choice makes the marginal probability of blue equal to $\gamma ,$ the same as that of red.

For each red-bit outcome in $E ,$ choose a witnessing $\phi .$ There are only finitely many red-bit outcomes, so this choice raises no measurability issue even when $\mathcal { A }$ is infinite. Conditional on that red outcome, both the input sequence and this witness are fixed. The witness has at least

$B - q \ge B / 2$ nonred definite-one positions, since its red definite ones are included among its at most q red nonzero values. Its number of blue definite ones is therefore binomial with mean

$$
\nu \ge \frac { \gamma } { 1 - \gamma } ( B - q ) \ge \gamma B / 2 .
$$

Since $\gamma B / 4 \le \nu / 2$ , a standard Chernof bound gives, conditional on the red outcome,

$$
\operatorname* { P r } \{ \mathrm { a t ~ l e a s t ~ } \gamma B / 4 \mathrm { ~ b l u e ~ d e f i n i t e ~ o n e s } \} \geq 1 - e ^ { - \nu / 8 } \geq 1 - e ^ { - \gamma B / 1 6 } \geq \frac { 1 } { 2 } .
$$

Let $E ^ { \prime }$ be the event that some partial target has at most $q$ red nonzero values and at least $\gamma B / 4$ blue definite ones. Thus $\operatorname* { P r } ( E ^ { \prime } \mid \xi _ { 1 : T } ) \ge 1 / 2$ on $E ,$ , and hence

$$
\operatorname* { P r } ( E ) \leq 2 \operatorname* { P r } ( E ^ { \prime } ) .\tag{44}
$$

Conditioning on the sampled union. The three possible colors—red, blue, and neither— have probabilities $\gamma , \gamma , 1 - 2 \gamma$ , independently across rounds. Equivalently, we may first generate independent union bits $U _ { t } \sim \mathrm { B e r n o u l l i } ( 2 \gamma )$ and then give each included round an independent fair red-or-blue color. Setting $\xi _ { t } = 1$ exactly on red rounds and reconstructing $z _ { t }$ from $\xi _ { < t }$ gives the original joint distribution of the red bits and inputs.

Write $\textstyle N = \sum _ { t = 1 } ^ { T } U _ { t }$ . Since $\mathbb { E } N = 2 \gamma T$ and $M \geq 4 \gamma T$ , another standard Chernof bound gives

$$
\operatorname* { P r } \{ N > M \} \leq e ^ { - 2 \gamma T / 3 } .\tag{45}
$$

Fix a union pattern with $N \leq M$ , and enumerate its included rounds as $\tau _ { 1 } < \cdots < \tau _ { N }$ . Conditional on this pattern, the colors at these rounds remain independent and fair. Since $z _ { t }$ depends only on the preceding sampling bits $\xi _ { < t }$ , fixing the union pattern makes $z _ { \tau _ { j } }$ a function of the colors at $\tau _ { 1 } , \ldots , \tau _ { j - 1 }$ . Indeed, $\xi _ { t } = 1$ exactly when round t is included and colored red. Thus the compressed stream $z _ { \tau _ { 1 } } , \dots , z _ { \tau _ { N } }$ is predictable with respect to the independent fair colors. Conditioning on the union has not fixed this input stream; it may still depend on the preceding colors.

Use the fixed family V of at most $S _ { r } ( M )$ deterministic binary causal rules supplied by Lemma 16 at horizon M. On $E ^ { \prime }$ , some rule in V matches all defined labels of a witnessing target on the compressed stream. It has at least $\gamma B / 4$ blue ones, because it matches every blue definite one. It has at most q red ones, because it must output zero at every red position where the target is zero; all other red positions, including undefined ones, are included in the target’s red nonzero count.

Bounding the red–blue discrepancy. Still conditioning on the union pattern, fix a rule $v \in \mathcal V$ Let $A _ { j }$ and $C _ { j }$ count its red and blue ones, respectively, among the first $j$ included rounds. The rule’s prediction at the next included round is determined before that round’s color is revealed. Therefore

$$
Z _ { j } : = 2 ^ { - A _ { j } } ( 3 / 2 ) ^ { C _ { j } } , \qquad 0 \le j \le N ,
$$

is a nonnegative martingale with $Z _ { 0 } = 1$ . When the prediction is zero, its multiplier is one; when the prediction is one, the conditional expected multiplier is

$$
\begin{array} { r } { \frac { 1 } { 2 } \cdot \frac { 1 } { 2 } + \frac { 1 } { 2 } \cdot \frac { 3 } { 2 } = 1 . } \end{array}
$$

On $\{ A _ { N } \leq q , ~ C _ { N } \geq \gamma B / 4 \}$ , one has $Z _ { N } \ge 2 ^ { - q } ( 3 / 2 ) ^ { \gamma B / 4 }$ . Markov’s inequality gives

$$
\mathrm { P r } \{ A _ { N } \leq q , \ C _ { N } \geq \gamma B / 4 \mid U _ { 1 : T } \} \leq 2 ^ { q } ( 3 / 2 ) ^ { - \gamma B / 4 } = 2 ^ { q } e ^ { - c _ { 0 } \gamma B } .
$$

Take a union bound over V, average over union patterns with $N \leq M$ , and add (45). We obtain

$$
\mathrm { P r } ( E ^ { \prime } ) \leq S _ { r } ( M ) 2 ^ { q } e ^ { - c _ { 0 } \gamma B } + e ^ { - 2 \gamma T / 3 } .
$$

Combining this with (44) proves the asserted bound conditional on each value of ω. Averaging over ω completes the proof. ■

For the application to prediction errors, fix a formal schedule J and set $\boldsymbol { z } _ { t } = ( x _ { t } , a _ { t } ^ { J } )$ , where $a _ { t } ^ { J }$ is its provisional prediction. Under the stated information order, this stream is predictable. For either partial error class in (38), a sampled nonzero value records a one-sided provisional error exceeding η. When J is the schedule produced for a target h by the reference procedure in Lemma 13, each such round requires at least one correction, so $H _ { \phi _ { h } ^ { \pm } } \leq d .$

Section 5 will apply Lemma 17 to each fixed schedule and take a union bound over all schedules, so the resulting guarantee also holds for the witnessing schedule of every target.

## 5 Master Histories and Multi-Pool Accuracy

The preceding section gives sign schedules that are accurate on selected rounds. We now introduce multiple sampling pools to obtain a simultaneous guarantee: for every target, a fixed positive fraction of the pools provide schedules with few large prediction errors over the entire horizon, including unselected rounds. We will also ensure that the target satisfies every constraint recorded by these schedules with a positive margin.

Parameters and capped sampling pools. Fix $\mathcal { H } , \eta , R , T , d$ as in Proposition 2. Recall that the rank of a version space is its sequential fat-shattering dimension at scale $\eta .$ The case $d = 0$ was settled in Section 4.1, so assume $d \geq 1$ and put $b = T / R$ . Fix a universal constant $\alpha \geq 1 2 8 ;$ we will specify suficient conditions on its size at the end of this section.

If $b \leq 3 2 \alpha d .$ , select every round. Lemmas 13 and 14 give a global pointwise radius-η cover of size

$$
Q _ { d } ( T ) \leq \left[ 2 e \left( 1 + \frac { T } { d } \right) \right] ^ { d } \leq ( 6 6 e \alpha R ) ^ { d } .\tag{46}
$$

Its witnessing experts have no errors of size at least $8 \eta .$ . This proves Proposition 2 in this regime, after increasing its universal constant if necessary.

For the rest of the section assume b > 32αd and set

$$
k = 4 \lceil \log _ { 2 } R \rceil , \qquad \beta = \frac { \alpha d } { b } < \frac { 1 } { 3 2 } , \qquad m = \lceil 2 \beta T \rceil = \lceil 2 \alpha d R \rceil .\tag{47}
$$

In particular, $k \geq 2 4$ and k is divisible by four. All logarithms below are natural unless a base is indicated.

For each $t \in [ T ]$ and $j \in [ k ]$ , let $\xi _ { t , j } \sim$ Bernoulli(β), independently over all rounds and pools. The raw selection list of pool $j$ contains the rounds with $\xi _ { t , j } = 1$ . Its capped list retains only the first m such rounds, with indicators

$$
\widehat { \xi } _ { t , j } : = \xi _ { t , j } { \bf 1 } \Bigg \{ \sum _ { s < t } \xi _ { s , j } < m \Bigg \} .
$$

The capped list therefore has length at most m on every outcome. The retained indicators are not themselves independent Bernoulli variables; all independence arguments below use the raw bits.

Define

$$
{ \mathsf { C a p } } : = \left\{ \sum _ { t = 1 } ^ { T } \xi _ { t , j } \leq m { \mathrm { ~ f o r ~ e v e r y ~ } } j \in [ k ] \right\} .\tag{48}
$$

Each raw list has expected length $\beta T = \alpha d R$ . Since $m \geq 2 \beta T$ , a Chernof bound and a union bound give

$$
\operatorname* { P r } ( \mathsf { C a p } ^ { \mathsf { c } } ) \leq k e ^ { - \alpha d R / 3 } .\tag{49}
$$

On Cap, the raw and capped selection indicators agree at every round. Thus, running the same schedule with raw or capped selections on the same realized input history produces identical states and predictions. We prove the sampling bounds under the original product law of the raw bits, without conditioning on Cap, and use this agreement to transfer the guarantees to the capped simulations on Cap.

Adaptive inputs. As in Lemma 17, we allow adaptive input sequences. Write $\pmb { \xi } _ { t } = ( \xi _ { t , 1 } , \dots , \xi _ { t , k } )$ and $\pmb { \xi } _ { < t } = ( \pmb { \xi } _ { 1 } , \dots , \pmb { \xi } _ { t - 1 } )$ . More precisely, we allow

$$
x _ { t } = F _ { t } ( \pmb { \xi } _ { < t } , \omega _ { 0 } ) ,\tag{50}
$$

where $F _ { t }$ are arbitrary deterministic maps and $\omega _ { 0 }$ is an independent finite-support auxiliary random seed.

Conditional on $\omega _ { 0 }$ and $\pmb { \xi } _ { < t }$ , the instance $x _ { t }$ and the provisional predictions of all fixed schedule simulations are determined, while the coordinates of $\xi _ { t }$ remain independent Bernoulli-β variables.

All probability bounds in this section hold uniformly over input rules of the form (50).

## 5.1 The Analytical Master and Uniform Accuracy

We first specify a common selection process for the reference histories, containing every round selected by any pool. The choice depends on $k \beta = \alpha k d R / T$ . When $k \beta \leq 1 / 8$ , we use the union of the pool selections, whose sampling rate is comparable to $k \beta .$ . When $k \beta > 1 / 8$ , we have $T < 8 \alpha k d R .$ so selecting every round is afordable. Accordingly, define

$$
\zeta _ { t } : = \left\{ \begin{array} { l l } { \mathbf { 1 } \{ \pmb { \xi } _ { t } \neq \mathbf { 0 } \} , } & { k \beta \le 1 / 8 , } \\ { 1 , } & { k \beta > 1 / 8 . } \end{array} \right.\tag{51}
$$

We call these cases the union regime and the all-round regime, respectively. In either case, $\xi _ { t , j } = 1$ implies $\zeta _ { t } = 1$

In the union regime, the $\zeta _ { t }$ are independent Bernoulli variables with rate

$$
u = 1 - ( 1 - \beta ) ^ { k } , \qquad { \frac { k \beta } { 2 } } \leq u \leq k \beta \leq { \frac { 1 } { 8 } } .\tag{52}
$$

The upper bound follows from a union bound, while the lower bound follows from $( 1 - \beta ) ^ { k } \leq e ^ { - k \beta }$ and $1 - e ^ { - z } \ge z / 2$ for $0 \leq z \leq 1$

The analytical master. For each target $h \in \mathcal H$ , define its analytical master by the following target-dependent procedure. Initialize $V = \mathcal { H }$ . At round t, before revealing $\xi _ { t } ,$ form the provisional prediction $a _ { t } ^ { h } \ = \ a ( V , x _ { t } )$ . If $\zeta _ { t } ~ = ~ 0$ , leave V unchanged and set $y _ { t } ^ { h } \ = \ a _ { t } ^ { h }$ . If $\zeta _ { t } ~ = ~ 1$ , while $| a ( V , x _ { t } ) - h ( x _ { t } ) | > \eta$ , apply the sign update toward $h ( x _ { t } )$ as in Lemma 13, recomputing the center after each update. Set $y _ { t } ^ { h } = a ( V , x _ { t } )$ after the correction loop.

Every update retains $h$ and decreases the sequential fat rank. Hence the procedure makes at most d corrections over the entire horizon, and

$$
\zeta _ { t } = 1 \quad \Longrightarrow \quad | y _ { t } ^ { h } - h ( x _ { t } ) | \leq \eta .\tag{53}
$$

The procedure uses target values only to find a witnessing schedule; it is not a label-free prediction rule. Record the selected ordinal and sign of each correction, and denote the resulting code by $J ^ { h }$ This code is defined on every raw outcome and always has depth at most $d ,$ although its ordinals need not satisfy a fixed cap.

A fixed family of master codes. Set

$$
M _ { * } : = 4 k m ,\tag{54}
$$

and let $\mathcal { I } _ { * }$ be the family of all formal sign schedules of depth at most $d$ on ordinals $1 , \ldots , M _ { * }$ . It is fixed before the experiment, and $| \mathcal { I } _ { * } | = Q _ { d } ( M _ { * } )$ . In the union regime, Cap bounds the number of master selections by km. In the all-round regime,

$$
k \beta > \frac { 1 } { 8 } \quad \Longrightarrow \quad T < 8 \alpha k d R \leq 4 k m = M _ { * } .
$$

Thus, on $\mathsf { C a p } , J ^ { h } \in \mathcal { I } _ { * }$ for every target. There is no truncation of the master: $M _ { * }$ is a bound used to specify the finite code family.

For every fixed $J \in \mathcal { I } _ { * }$ , simulate its prescribed updates on the master selection list, starting from $\mathcal { H } ,$ and denote its provisional and final predictions by $a _ { t } ^ { J }$ and $y _ { t } ^ { J }$ . Use the conventions of Section 4.2: an empty-fiber update makes the simulation invalid with subsequent predictions zero, and no updates are prescribed at ordinals greater than $M _ { * }$ . Every fixed-code simulation is therefore defined on every raw outcome, including outcomes outside Cap. All simulations use the same realized inputs and selection lists; varying J does not change the input rule. On an outcome with $J ^ { h } \in { \mathcal { I } } _ { * }$ , the simulation of $J ^ { h }$ has exactly the target-driven master’s states and predictions, by induction on rounds and on corrections within each round.

Accuracy before the current selection. The selected-round guarantee (53) concerns final predictions. We also need control of the provisional predictions on all rounds. Define

$$
\mathsf { M a s t e r } : = \left\{ \mathrm { f o r ~ e v e r y ~ } h \in \mathcal { H } , \quad \# \{ t \in [ T ] : | a _ { t } ^ { h } - h ( x _ { t } ) | \geq 2 \eta \} < \frac { b } { 3 2 } \right\} .\tag{55}
$$

Lemma 18 (Uniform master accuracy). In the all-round regime, Master holds on every outcome. In the union regime, put $M = \lceil 4 u T \rceil$ and let $c _ { 0 } = \log ( 3 / 2 ) / 4$ be the constant in Lemma 17. Then

$$
\begin{array} { r } { \operatorname* { P r } ( C { \mathsf { a p } } \cap \mathsf { M a s t e r } ^ { \mathsf { c } } ) \leq 4 Q _ { d } ( M _ { * } ) \left( S _ { d } ( M ) 2 ^ { d } e ^ { - c _ { 0 } u b / 6 4 } + e ^ { - 2 u T / 3 } \right) . } \end{array}\tag{56}
$$

Proof. In the all-round regime, every provisional error of size at least $2 \eta$ forces a correction. There are at most d such rounds, and $d < b / 3 2$ because $b > 3 2 \alpha d$ . This proves the first claim.

Now consider the union regime and fix $J \in \mathcal { I } _ { * }$ . We will apply Lemma 17 to the stream $( x _ { t } , a _ { t } ^ { J } )$ and the Bernoulli-u master selections. The lemma requires the stream to be determined by the past sampling bits and an auxiliary seed independent of those bits. Although $\zeta _ { t }$ are independent Bernoulli-u variables, the inputs may depend on the full preceding pool vectors $\pmb { \xi } _ { < t }$ , which are not determined by $\zeta _ { < t }$ and $\omega _ { 0 }$ alone.

An equivalent way to generate the raw vectors is to draw independent Bernoulli-u bits $\zeta _ { t }$ and independent nonzero vectors $W _ { t } ,$ where each $W _ { t }$ has the Bernoulli- $\boldsymbol { \cdot } \beta$ product law conditioned on being nonzero, and to set

$$
\pmb { \xi } _ { t } = \zeta _ { t } W _ { t } .
$$

The entire sequence $( W _ { t } ) _ { t = 1 } ^ { T }$ is independent of the union bits and of $\omega _ { 0 }$ . This generates exactly the original joint law of the raw pool vectors. Fixing the finite-support auxiliary seed $\omega = ( \omega _ { 0 } , W _ { 1 }$ $\dots , W _ { T } )$ leaves the union bits independent Bernoulli-u variables. Once this auxiliary seed is fixed, each preceding pool vector is determined by the corresponding union bit through ${ \pmb { \xi } } _ { s } = \zeta _ { s } W _ { s }$ , for $s < t$ . Consequently, the stream $( x _ { t } , a _ { t } ^ { J } )$ is determined by $\zeta _ { < t }$ and $\omega ,$ as required by Lemma 17. For each fixed $\omega ,$ Lemma 17 applies and gives the same probability bound, independently of $\omega .$ Averaging this conditional bound over $\omega$ yields the corresponding unconditional bound. Since the reconstructed pool vectors have the original joint distribution, the bound holds for the original sampling process.

For either partial error class $\mathcal { A } ^ { + }$ or $A ^ { - }$ from (38), apply Lemma 17 with

$$
\gamma = u , \qquad r = q = d , \qquad B = b / 6 4 .
$$

The dimension hypothesis follows from Lemma 15. The numerical hypotheses follow from $u \leq 1 / 8$ and

$$
\frac { b } { 6 4 } > \frac { \alpha d } { 2 } \geq 2 d , \qquad \frac { u b } { 6 4 } \geq \frac { \alpha k d } { 1 2 8 } \geq 1 6 \log 2 ,
$$

using $k \geq 2 4$ and $\alpha \geq 1 2 8$ . For this fixed code $J \in \mathcal { I } _ { * }$ and either sign, the probability that some partial target has at least $b / 6 4$ definite ones but at most d sampled nonzero values is at most

$$
2 \left( S _ { d } ( M ) 2 ^ { d } e ^ { - c _ { 0 } u b / 6 4 } + e ^ { - 2 u T / 3 } \right) .
$$

Take a union bound over the two signs and all $Q _ { d } ( M _ { * } )$ fixed codes.

Outside this union of bad events, consider an outcome in ${ \mathsf { C a p } }$ and any target $h .$ . Its actual code $J ^ { h }$ belongs to $\mathcal { I } _ { * }$ . A sampled nonzero value of either $\phi _ { h } ^ { + } ( x _ { t } , a _ { t } ^ { J ^ { h } } )$ or $\phi _ { h } ^ { - } ( x _ { t } , a _ { t } ^ { J ^ { h } } )$ means that the corresponding one-sided provisional error exceeds $\eta .$ . It therefore forces a correction in the target-driven master. For each sign, there are at most d such sampled rounds, so the one-sided definite-error count is less than $b / 6 4$ . The sum of these two counts is exactly # $\{ t : | a _ { t } ^ { h } - h ( x _ { t } ) | \geq 2 \eta \}$ which is consequently less than $b / 3 2$ . This proves (56). The estimates were applied to fixed codes before selecting any $J ^ { h }$ ; no conditioning on a target’s realized code was used. ■

## 5.2 Tolerant Replay and Bufered Corrections

We next use the master’s final predictions as reference values for version space updates in each individual raw pool. Each pool maintains its own version space and applies the same center rule and sign-update rule as the master, but to its own version space. Unlike the master, which corrects when its center difers from $h ( x _ { t } )$ by more than $\eta ,$ a pool replay corrects when its center difers from the master’s final prediction by more than 4η.

The replay rule. For each fixed master code $J \in \mathcal { I } _ { * }$ and each pool $j$ , start a separate version space at H. Its provisional prediction at round t is the center of the version space left by round $t - 1$ at $x _ { t }$ . After revealing $\xi _ { t } ,$ , first compute the master’s final value $y _ { t } ^ { J }$ . If $\xi _ { t , j } = 0$ , leave the replay version space unchanged. If $\xi _ { t , j } ~ = ~ 1$ , repeatedly update its current center a toward $y _ { t } ^ { J }$ while $\begin{array} { r } { | a - y _ { t } ^ { J } | > 4 \eta : } \end{array}$ use the plus fiber in (32) when $y _ { t } ^ { J } > a + 4 \eta$ , and the minus fiber otherwise.

Recompute the center after each update and emit the final prediction when the loop stops. We call this procedure the tolerant replay of J in pool $j ,$ and denote its provisional and final predictions by $a _ { t , j } ^ { J }$ and $y _ { t , j } ^ { J }$

For an arbitrary fixed formal code $J \in \mathcal { I } _ { * }$ , the replay need not retain any particular target, since the master’s final predictions $y _ { t } ^ { J }$ need not approximate any target on that outcome. An update may therefore lead to an empty fiber, in which case the replay enters an absorbing invalid state and outputs zero from that round onward. Every nonempty update strictly decreases rank, so at most d such updates can occur. Hence each correction loop terminates either when the current center is within 4η of the reference value or when an update produces an empty fiber. In either case, the replay has a well-defined final prediction on every round of every raw outcome.

For a target $h ,$ , apply exactly the same replay rule to its target-driven master values $y _ { t } ^ { h }$ , and write $a _ { t , j } ^ { h }$ and $y _ { t , j } ^ { h }$ for the resulting predictions. These target-driven replays are defined directly from the master for $h ,$ not by conditioning on any formal code. On an outcome with $J ^ { h } \in \mathcal { I } _ { * }$ , they coincide with the replays of $J ^ { h }$

Lemma 19 (Bufered replay). For every target $h \in \mathcal H$ and pool $j$ , the target-driven replay always retains h and makes at most d corrections. More precisely, whenever a correction is performed at center $^ { a , }$ we have

$$
\left\{ \begin{array} { l l } { h ( x _ { t } ) > a + 3 \eta , } & { f o r \ a \ p l u s \ u p d a t e , } \\ { h ( x _ { t } ) < a - 3 \eta , } & { f o r \ a \ m i n u s \ u p d a t e . } \end{array} \right.\tag{57}
$$

Moreover, on every round selected by pool $j ,$

$$
| y _ { t , j } ^ { h } - h ( x _ { t } ) | \leq 5 \eta .\tag{58}
$$

Proof. Every pool-selected round is also selected by the master, so $| y _ { t } ^ { h } - h ( x _ { t } ) | \leq \eta$ on such rounds. If the replay performs a plus update at center $a ,$ , then $y _ { t } ^ { h } > a + 4 \eta$ , and hence

$$
h ( x _ { t } ) \geq y _ { t } ^ { h } - \eta > a + 3 \eta .
$$

The minus case is symmetric. Since the corresponding sign-update fibers are defined by the thresholds $a + \eta$ and $a - \eta$ , respectively, each correction retains h. Every correction strictly decreases rank, so at most d corrections occur.

On a selected round, the replay therefore terminates with $| y _ { t , j } ^ { h } - y _ { t } ^ { h } | \leq 4 \eta$ . The triangle inequality gives the claimed error bound. ■

On Cap, the target-driven replay in pool $j$ makes at most d corrections, all at selected ordinals in [m]. Recording their ordinals and signs gives a formal schedule $J _ { j } ^ { h }$ of depth at most d. Simulating $J _ { j } ^ { h }$ on the same instance sequence and pool $j ^ { \prime } { : }$ s capped selection history reproduces exactly the replay’s version spaces and predictions.

From disagreements to prediction errors. For each target and pool, define

$$
\begin{array} { l } { { \displaystyle { D _ { j } ^ { h } : = \sum _ { t = 1 } ^ { T } \mathbf { 1 } \{ | a _ { t } ^ { h } - a _ { t , j } ^ { h } | > 6 \eta \} } , } } \\ { { \displaystyle { H _ { j } ^ { h } : = \sum _ { t = 1 } ^ { T } \xi _ { t , j } \mathbf { 1 } \{ | a _ { t } ^ { h } - a _ { t , j } ^ { h } | > 6 \eta \} } . } } \end{array}\tag{59}
$$

Thus $D _ { j } ^ { h }$ counts disagreements between the provisional predictions, and $H _ { j } ^ { h }$ counts those disagree ments selected by pool j. On a round selected by pool j with $| a _ { t } ^ { h } - a _ { t , j } ^ { h } | > 6 \dot { \eta }$ , if the master makes at least one correction, charge the round to the master. Otherwise, $y _ { t } ^ { h } = a _ { t } ^ { h }$ , so $| y _ { t } ^ { h } - a _ { t , j } ^ { h } | > 6 \eta > 4 \eta$ and the replay must make at least one correction; charge the round to the replay. Since each procedure makes at most d corrections in total, at most 2d distinct rounds can be charged. Hence

$$
H _ { j } ^ { h } \leq 2 d \qquad ( h \in { \mathcal { H } } , ~ j \in [ k ] ) .\tag{60}
$$

If $| a _ { t , j } ^ { h } - h ( x _ { t } ) | \geq 8 \eta$ and $| a _ { t } ^ { h } - h ( x _ { t } ) | < 2 \eta$ , then $| a _ { t } ^ { h } - a _ { t , j } ^ { h } | > 6 \eta$ by the triangle inequality. Consequently, on Master,

$$
\# \{ t : | a _ { t , j } ^ { h } - h ( x _ { t } ) | \geq 8 \eta \} < \frac { b } { 3 2 } + D _ { j } ^ { h } .\tag{61}
$$

By (58), a final replay error of size at least 8η can occur only on an unselected round. There the provisional and final predictions agree, so final outliers are a subset of provisional outliers. In particular, on Master, the condition $D _ { j } ^ { h } < b / 1 6$ makes both outlier counts less than $3 b / 3 2 < b / 8$

## 5.3 Simultaneous Accuracy Across Pools

Although the replay histories need not be independent, the current disagreement indicators are determined by the past, while the fresh pool selection bits remain independent Bernoulli-β variables even after conditioning on the past. We use this fact to establish a joint bound.

A joint bound for fixed master codes. For a fixed $J \in \mathcal { I } _ { * }$ and its replays defined above, set

$$
\Delta _ { t , j } ^ { J } : = \mathbf { 1 } \{ | a _ { t } ^ { J } - a _ { t , j } ^ { J } | > 6 \eta \} , \qquad D _ { j } ^ { J } : = \sum _ { t = 1 } ^ { T } \Delta _ { t , j } ^ { J } , \qquad H _ { j } ^ { J } : = \sum _ { t = 1 } ^ { T } \xi _ { t , j } \Delta _ { t , j } ^ { J } .\tag{62}
$$

Every current indicator $\Delta _ { t , j } ^ { J }$ is determined before $\xi _ { t }$ is revealed. We do not assert $H _ { j } ^ { J } \leq 2 d$ for arbitrary formal codes; that bound is needed only for target-driven histories, as in (60).

Lemma 20 (Joint disagreement bound). For every fixed $J \in \mathcal { I } _ { * }$ and $I \subseteq [ k ]$

$$
\operatorname* { P r } \left\{ D _ { j } ^ { J } \geq \frac { b } { 1 6 } , \ H _ { j } ^ { J } \leq 2 d \ f o r \ e v e r y \ j \in I \right\} \leq e ^ { - \alpha | I | d / 6 4 } .\tag{63}
$$

Proof. Work under the original joint law of the raw pool bits. Let $D _ { j , t } ^ { J }$ and $H _ { j , t } ^ { J }$ denote the corresponding counts through round $t ,$ and define

$$
Z _ { t } : = \exp \left( \frac { \beta } { 2 } \sum _ { j \in I } D _ { j , t } ^ { J } - ( \log 2 ) \sum _ { j \in I } H _ { j , t } ^ { J } \right) , \qquad Z _ { 0 } = 1 .
$$

Conditional on $\omega _ { 0 }$ and $\pmb { \xi } _ { < t }$ , all current disagreement indicators are fixed and the current pool bits are independent Bernoulli-β variables. Hence

$$
\mathbb { E } \left[ \frac { Z _ { t } } { Z _ { t - 1 } } \Bigg | \omega _ { 0 } , \pmb { \xi } _ { < t } \right] = \prod _ { j \in I : \Delta _ { t , j } ^ { J } = 1 } e ^ { \beta / 2 } ( 1 - \beta / 2 ) \le 1 ,
$$

where the last inequality uses $1 - z \leq e ^ { - z }$ . Thus $\left( Z _ { t } \right)$ is a nonnegative supermartingale for the filtration generated by $\omega _ { 0 } , \pmb { \xi } _ { 1 } , \ldots , \pmb { \xi } _ { t }$ , and $\mathbb { E } Z _ { T } \leq 1$ . On the event in (63),

$$
Z _ { T } \geq \exp \left[ | I | d \left( { \frac { \alpha } { 3 2 } } - 2 \log 2 \right) \right] ,
$$

since $\beta b = \alpha d .$ Markov’s inequality gives the upper bound ex $\left. \mathrm { p } [ - | I | d ( \alpha / 3 2 - 2 \log 2 ) ] \right.$ ]. Finally, $\alpha \geq 1 2 8$ implies $\alpha / 3 2 - 2 \log 2 \geq \alpha / 6 4$ ■

Define Joint to be the event that there are no $J \in \mathcal { T } _ { * }$ and $I \subseteq [ k ]$ with $| I | = k / 4$ such that $D _ { j } ^ { J } \geq b / 1 6$ and $H _ { j } ^ { J } \leq 2 d$ for every $j \in I .$ . A union bound in Lemma 20 gives

$$
\mathrm { P r } ( \mathsf { J o i n t ^ { c } } ) \leq Q _ { d } ( M _ { * } ) \binom { k } { k / 4 } e ^ { - \alpha k d / 2 5 6 } .\tag{64}
$$

This estimate does not condition on Cap or Master, and it assumes no independence between replays.

Proposition 21 (Simultaneous replay accuracy). For a suficiently large universal choice of α, in the regime $b > 3 2 \alpha d$ , the event

$$
\mathsf { G o o d } : = \mathsf { C a p } \cap \mathsf { M a s t e r } \cap \mathsf { J o i n t }
$$

has probability greater than 15/16 for every input rule (50). On this event, for every $h \in \mathcal H$ , at least 3k/4 pools j satisfy

$$
\operatorname* { m a x } \Bigl \{ \# \{ t : | a _ { t , j } ^ { h } - h ( x _ { t } ) | \geq 8 \eta \} , \# \{ t : | y _ { t , j } ^ { h } - h ( x _ { t } ) | \geq 8 \eta \} \Bigr \} < \frac { b } { 8 } .\tag{65}
$$

Proof. We first prove the accuracy assertion on Good. Fix an outcome in this event and a target $h \in \mathcal H$ . Since Cap holds, the master’s correction code $J ^ { h }$ belongs to $\mathcal { I } _ { * }$ . On this realized history, the simulations associated with $J ^ { h }$ reproduce exactly the target-driven master and its pool replays. Hence

$$
D _ { j } ^ { J ^ { h } } = D _ { j } ^ { h } , \qquad H _ { j } ^ { J ^ { h } } = H _ { j } ^ { h } \qquad ( j \in [ k ] ) .
$$

By (60), $H _ { j } ^ { h } \leq 2 d$ for every pool j. Since Joint holds, fewer than $k / 4$ pools can satisfy $D _ { j } ^ { h } \geq b / 1 6$ Consequently, the set

$$
I _ { h } : = \{ j \in [ k ] : D _ { j } ^ { h } < b / 1 6 \}
$$

has cardinality at least $3 k / 4$

For every $j \in I _ { h }$ , the error transfer (61) and Master give

$$
\# \{ t : | a _ { t , j } ^ { h } - h ( x _ { t } ) | \geq 8 \eta \} < \frac { b } { 3 2 } + D _ { j } ^ { h } < \frac { 3 b } { 3 2 } < \frac { b } { 8 } .
$$

Every final replay outlier is also a provisional outlier, as established above. Thus (65) holds for every $j \in I _ { h }$ . Since the outcome and h were arbitrary, this conclusion holds for all targets on every outcome in Good.

It remains to show that $\operatorname* { P r } ( \mathsf { G o o d } ) > 1 5 / 1 6$ . We verify that a suficiently large universal α makes each of the three failure probabilities small enough.

Recall that $R > 3 2 , d \geq 1$ , and $k = 4 \lceil \log _ { 2 } R \rceil$ . The parameter definitions give

$$
2 4 \leq k \leq R , \qquad \log R \leq \frac { k \log 2 } { 4 } , \qquad m \leq 3 \alpha d R , \qquad M _ { * } \leq 1 2 \alpha k d R .\tag{66}
$$

In the union regime, the bounds $k \beta / 2 \le u \le k \beta$ further imply

$$
d \leq M = \lceil 4 u T \rceil \leq 5 \alpha k d R , \qquad u b \geq { \frac { \alpha k d } { 2 } } , \qquad u T \geq { \frac { \alpha k d R } { 2 } } .
$$

Lemma 14 and (41) therefore yield

$$
Q _ { d } ( M _ { \ast } ) \leq ( 2 6 e \alpha k R ) ^ { d } , \qquad S _ { d } ( M ) \leq ( 5 e \alpha k R ) ^ { d } ,\tag{67}
$$

where the second bound is needed only in the union regime.

Set $A _ { \alpha } : = \log ( 1 0 0 e \alpha )$ . Since $k \leq R$ and log $R \leq k \log 2 / 4$ , we have

$$
\log ( k R ) \leq 2 \log R \leq { \frac { k \log 2 } { 2 } } .
$$

Together with (67), $d \geq 1$ , and ${ \binom { k } { k / 4 } } \leq 2 ^ { k }$ , this gives

$$
\begin{array} { r } { \log \bigl ( 4 Q _ { d } ( M _ { * } ) S _ { d } ( M ) 2 ^ { d } \bigr ) \leq ( 2 A _ { \alpha } + 3 ) k d , } \\ { \log \bigl ( 4 Q _ { d } ( M _ { * } ) \bigr ) \leq ( A _ { \alpha } + 2 ) k d , } \\ { \log \biggl ( Q _ { d } ( M _ { * } ) \binom { k } { k / 4 } \biggr ) \leq ( A _ { \alpha } + 2 ) k d . } \end{array}
$$

Choose $\alpha \geq 1 2 8$ suficiently large that

$$
{ \frac { c _ { 0 } \alpha } { 1 2 8 } } \geq 2 A _ { \alpha } + 4 , \qquad { \frac { \alpha } { 2 5 6 } } \geq A _ { \alpha } + 3 ,\tag{68}
$$

where $c _ { 0 } = \log ( 3 / 2 ) / 4$ . Such a universal choice exists because $A _ { \alpha } = O ( \log \alpha )$

In the union regime, (56) and the preceding estimates imply

$$
\operatorname* { P r } ( \mathsf { C a p } \cap \mathsf { M a s t e r } ^ { \mathsf { c } } ) \leq 2 e ^ { - k d } .
$$

Indeed, the first term in (56) is at most $e ^ { - k d }$ because

$$
\frac { c _ { 0 } u b } { 6 4 } \geq \frac { c _ { 0 } \alpha k d } { 1 2 8 } \geq ( 2 A _ { \alpha } + 4 ) k d ,
$$

while the second is at most $e ^ { - k d }$ because

$$
\frac { 2 u T } { 3 } \geq \frac { \alpha k d R } { 3 } \geq \frac { \alpha k d } { 2 5 6 } \geq ( A _ { \alpha } + 3 ) k d .
$$

In the all-round regime, Master holds on every outcome, so the same probability bound holds in both regimes.

Similarly, (64) and (68) give

$$
\mathrm { P r } ( \mathsf { J o i n t } ^ { \mathrm { c } } ) \leq Q _ { d } ( M _ { * } ) { \binom { k } { k / 4 } } e ^ { - \alpha k d / 2 5 6 } \leq e ^ { - k d } .
$$

Finally, by (49), $k \leq R$ , and $\alpha \geq 1 2 8$ ，

$$
\operatorname* { P r } ( \mathsf { C a p } ^ { \mathsf { c } } ) \leq k e ^ { - \alpha d R / 3 } \leq e ^ { - R } .
$$

Combining these bounds,

$$
\begin{array} { r l } & { \operatorname* { P r } ( \mathsf { G o o d } ^ { \mathrm { c } } ) \leq \operatorname* { P r } ( \mathsf { C a p } ^ { \mathrm { c } } ) + \operatorname* { P r } ( \mathsf { C a p } \cap \mathsf { M a s t e r } ^ { \mathrm { c } } ) + \operatorname* { P r } ( \mathsf { J o i n t } ^ { \mathrm { c } } ) } \\ & { \qquad \leq e ^ { - R } + 3 e ^ { - k d } < \displaystyle \frac { 1 } { 1 6 } , } \end{array}
$$

where the last inequality uses $R > 3 2$ and $k d \geq 2 4$ . The choice of α and all probability bounds are independent of the input rule and the law of its independent auxiliary seed. ■

By Lemma 19, on Cap each target-driven replay can be reproduced by a sign schedule of depth at most d on the corresponding pool’s capped selection history. Thus Proposition 21 provides at least $3 k / 4$ accurate schedules for every target on the same event Good.

## 6 Robust Identification and Eligible Parents

We use the parameters fixed in Section 5, in the regime $b > 3 2 \alpha d$ . A singleton class is covered exactly by its unique member, so assume $| \mathcal { H } | \geq 2$ . All simulations in this section use the capped pools. Each selected list therefore has length at most $m ,$ including on outcomes outside Cap.

The preceding section provides accurate replay histories for every target. We now consider sign schedules whose version spaces reduce to singletons. To obtain useful counting bounds, we additionally require the surviving function to satisfy every correction inequality with a prescribed margin.

## 6.1 Robust Identification and Local Counting

For each integer $0 \leq r \leq d ,$ , let $\mathcal { J } _ { r } ( m )$ denote the formal sign schedules of depth at most r with selected ordinals in $[ m ]$ . Thus $| \mathcal { I } _ { r } ( m ) | = Q _ { r } ( m )$ . In each capped pool $j ,$ simulate every $L \in \mathcal { I } _ { d } ( m )$ from the initial version space $\mathcal { H } .$ using the rules of Section 4.2. Write $V _ { t } ^ { \bar { j } , L }$ for the version space after round t and $p _ { t } ^ { j , L }$ for the final prediction on that round. An invalid simulation has $V _ { t } ^ { j , L } = \emptyset$ and $p _ { t } ^ { j , L } = 0$ from the invalidating update onward. These are simulations of prescribed sign schedules: they do not use master reference values or target values.

The inequalities imposed by a schedule. Consider a schedule $L = \left( \left( s _ { 1 } , \sigma _ { 1 } \right) , \ldots , \left( s _ { i } , \sigma _ { i } \right) \right)$ that is valid and born by round t. Recall that this means that all its prescribed updates have occurred without producing an empty version space. For each update $\ell \in [ i ]$ , let $t _ { \ell } \leq t$ be its actual round and let $a \ell$ be the center immediately before that update. Several $t _ { \ell }$ may coincide; their centers are recomputed after each update. By the definition of the sign-update fibers,

$$
\begin{array}{c} V _ { t } ^ { j , L } = \left\{ f \in \mathcal { H } : \begin{array} { l l } { f ( x _ { t _ { \ell } } ) \geq a _ { \ell } + \eta } & { \mathrm { f o r ~ e v e r y ~ } \ell \mathrm { ~ w i t h ~ } \sigma _ { \ell } = + , } \\ { f ( x _ { t _ { \ell } } ) \leq a _ { \ell } - \eta } & { \mathrm { f o r ~ e v e r y ~ } \ell \mathrm { ~ w i t h ~ } \sigma _ { \ell } = - } \end{array} \right\} .  \end{array}\tag{69}
$$

Here $f$ ranges over candidate functions in H. We call these inequalities the update constraints. Their thresholds are computed from the instance history, the pool selections, and $L ;$ they are not additional entries in the code.

Definition 22 (Robust certificate). The simulation of L in pool j is a robust certificate for $h \in \mathcal H$ by round t if it is valid and born by t, its version space is $V _ { t } ^ { \bar { j } , L } = \{ h \}$ , and every update satisfies

$$
\begin{array}{c} \begin{array} { r } { \left\{ h ( x _ { t _ { \ell } } ) \geq a _ { \ell } + 3 \eta , \quad i f \sigma _ { \ell } = + , \right.} \\ { h ( x _ { t _ { \ell } } ) \leq a _ { \ell } - 3 \eta , \quad i f \sigma _ { \ell } = - . } \end{array}   \end{array}\tag{70}
$$

Thus, for a plus update, $h ( x _ { t _ { \ell } } )$ exceeds the retained boundary $a _ { \ell } + \eta$ by at least $2 \eta ;$ for a minus update it lies at least $2 \eta$ below $a \ell - \eta$ . The test first examines the simulation’s version space. If this space is a singleton, its unique member specifies the function h whose values are used in (70). No externally supplied target labels are needed. These conditions are defined set-theoretically; we do not require them to be eficiently computable.

Identification and activation. Say that pool j identifies h by round t if one of its simulations is a robust certificate for h by that round. Let

$$
\mathcal { T } _ { j } ( t ) : = \{ h \in \mathcal { H } : \mathrm { p o o l } \ j \ \mathrm { i d e n t i f i e s } \ h \mathrm { b y ~ r o u n d } \ t \} .\tag{71}
$$

Identification is tested after the schedule updates on round t. Once a schedule is born, its version space and update constraints no longer change. Its certificate therefore persists. In particular, ${ \mathcal { T } } _ { j } ( t ) \subseteq { \mathcal { T } } _ { j } ( t + 1 )$ ), and each schedule can certify at most one function over the entire run. Hence $| \mathcal { T } _ { j } ( t ) | \le Q _ { d } ( m )$ . At time zero only the empty schedule is born, and $| \mathcal { H } | \geq 2 ,$ , so $\mathcal { T } _ { j } ( 0 ) = \emptyset$

A function h activates on the first round when at least $k / 2$ pools have identified it. Formally, define

$$
\tau ( h ) : = \operatorname* { m i n } \left\{ t \in [ T ] : \# \{ j \in [ k ] : h \in \mathbb { Z } _ { j } ( t ) \} \geq k / 2 \right\} ,\tag{72}
$$

with $\tau ( h ) = \infty$ if the set is empty. Write $\mathcal { A } _ { t } : = \{ h \in \mathcal { H } : \tau ( h ) \leq t \}$ and $\mathcal { A } : = \mathcal { A } _ { T }$ . These sets are finite, since $\textstyle \mathcal { A } _ { t } \subseteq \bigcup _ { j = 1 } ^ { k } \mathcal { T } _ { j } ( t )$

Singleton replay states. Write $U _ { t , j } ^ { h }$ for the version space of the target-driven replay for h in raw pool $j$ after round t. On Cap, record only its corrections through round t, obtaining a prefix schedule in $\mathcal { I } _ { d } ( m )$ . In capped pool $j ,$ this prefix schedule is valid and born by t and reproduces the replay through that round. Lemma 19 gives the strict versions of (70) for all its updates. Consequently,

$$
\mathrm { o n ~ C a p } , \qquad U _ { t , j } ^ { h } = \{ h \} \quad \Longrightarrow \quad h \in \mathbb { Z } _ { j } ( t ) .\tag{73}
$$

This implication uses the bufered update inequalities, not the accuracy of the replay on unselected rounds.

Local identification bounds. The separation in (70) limits how many functions a pool can identify within a subclass of small $\operatorname { s f a t } _ { \eta } ( V )$ . The bound holds for every realized capped history and does not require Good.

Lemma 23 (Local identification bound). On every outcome, for every $t \in \{ 0 , \ldots , T \}$ , pool $j \in [ k ]$ and nonempty subclass $V \subseteq \mathcal { H }$ with $\mathrm { s f a t } _ { \eta } ( V ) \leq r$ , where $r \in \{ 0 , \ldots , d \}$ ,

$$
| { \cal T } _ { j } ( t ) \cap V | \leq Q _ { r } ( m ) .\tag{74}
$$

Proof. Fix the outcome, $t , j$ , and $V .$ We first show that any two distinct functions $h , h ^ { \prime } \in \mathbb { Z } _ { j } ( t )$ difer by more than $2 \eta$ at some round selected by capped pool $j$ by time t. Choose a certificate for $h .$ Since it is born by round t and its version space is $\{ h \}$ , the function $h ^ { \prime }$ must violate an update constraint imposed at some round $s \leq t$ selected by capped pool $j$

If this is a plus update at center $^ { a , }$ then

$$
h ^ { \prime } ( x _ { s } ) < a + \eta , \qquad h ( x _ { s } ) \geq a + 3 \eta ,
$$

so $h ( x _ { s } ) - h ^ { \prime } ( x _ { s } ) > 2 \eta$ . The minus case is symmetric.

Let $z _ { 1 } , \ldots , z _ { n }$ be the instances in that list through round t, in order, so $n \leq m$ . If $n = 0$ , no pool schedule other than the empty schedule is born, and $\mathcal { T } _ { j } ( t ) = \mathcal { O }$ . Otherwise, apply Lemma 13 to the initial class $V$ and the sequence $z _ { 1 } , \ldots , z _ { n }$ , selecting every position. It gives at most $Q _ { r } ( m )$ prediction vectors such that every $f \in V$ is within η of one vector at every position. Two distinct functions in $\mathbb { Z } _ { j } ( t ) \cap V$ cannot use the same vector: the triangle inequality would make their values difer by at most 2η at every selected position, contrary to the separation just proved. This proves (74). ■

Corollary 24 (Local activation bound). Under the same hypotheses on $V$ and $^ { r , }$ on every outcome and for every $t \in \{ 0 , \ldots , T \}$ ，

$$
\left| { \mathcal { A } } _ { t } \cap V \right| \leq 2 Q _ { r } ( m ) .\tag{75}
$$

In particular, $| \mathcal { A } | \le 2 Q _ { d } ( m )$

Proof. Every $h \in { \mathcal { A } } _ { t }$ has been identified by at least $k / 2$ pools by time $t ,$ since identification persists after activation. Counting the pairs $( h , j )$ with $h \in \mathcal { A } _ { t } \cap V$ and $h \in \mathbb { Z } _ { j } ( t )$ gives

$$
\frac { k } { 2 } | \mathcal { A } _ { t } \cap V | \leq \sum _ { j = 1 } ^ { k } | \mathbb { Z } _ { j } ( t ) \cap V | \leq k Q _ { r } ( m ) .
$$

The first assertion follows by division, and the second by taking $t = T , V = \mathcal { H }$ , and $r = d .$

In particular, $V$ may be a version space produced during the run; it need not be fixed before the experiment.

## 6.2 Parent and Child Experts

We preallocate two types of prediction streams: parents, which follow fixed sign schedules, and children, which may later be assigned to activated functions. For every pool and every schedule of depth at most $d - 1$ , retain its prediction stream as a parent. Thus the set of parent indices is

$$
\mathscr { P } : = [ k ] \times \mathscr { J } _ { d - 1 } ( m ) , \qquad \mathscr { P } : = | \mathscr { P } | = k Q _ { d - 1 } ( m ) .\tag{76}
$$

For each $v = ( j , L ) \in \mathcal { P }$ , let $p _ { t } ^ { v }$ denote the prediction of schedule $L$ in capped pool $j$ at round $t ,$ and call the depth of L the depth of $v .$ . The parent always follows this schedule. It outputs zero if the schedule becomes invalid.

If a depth-i parent becomes valid and born, let $V _ { v }$ be its version space after birth. This space is fixed for the rest of the run, and the rank-drop property gives

$$
\mathrm { s f a t } _ { \eta } ( V _ { v } ) \leq d - i .\tag{77}
$$

Identification considers all schedules in $\mathcal { I } _ { d } ( m )$ , whereas parent indices are allocated only to schedules in $\mathcal { I } _ { d - 1 } ( m )$

Children and their outputs. For each $i \in \{ 0 , \ldots , d - 1 \}$ , fix a nonnegative integer $c _ { i }$ before the experiment. Attach $c _ { i }$ additional prediction indices to every depth-i parent $v ,$ and call them its children, indexed by $( v , \ell )$ for $\ell \in \{ 1 , \ldots , c _ { i } \}$ . Each child initially copies its parent’s outputs. On a round when a function h activates, at most one unused child may be assigned to $h .$ . From that round onward it predicts $h ( x _ { s } )$ on every subsequent instance $x _ { s } .$ , and it is never reassigned.

Assignments on round t are made after the capped schedule updates and identification tests, but before emitting the child predictions on that round. Thus

$$
g _ { t } ^ { v , \ell } : = { \left\{ \begin{array} { l l } { h ( x _ { t } ) , } & { { \mathrm { i f ~ t h i s ~ c h i l d ~ h a s ~ b e e n ~ a s s i g n e d ~ t o ~ } } h { \mathrm { ~ b y ~ r o u n d ~ } } t , } \\ { p _ { t } ^ { v } , } & { { \mathrm { i f ~ t h i s ~ c h i l d ~ i s ~ s t i l l ~ u n a s s i g n e d . } } } \end{array} \right. }\tag{78}
$$

In particular, assignment does not change any past prediction.

The capacities $c _ { i }$ and the rule choosing the assignments will be specified in the next section. Here we define which choices are permitted and prove their deterministic properties. For any such capacities, the number of parent and child indices is

$$
K = P + \sum _ { i = 0 } ^ { d - 1 } k n _ { i } ( m ) c _ { i } .\tag{79}
$$

This counts invalid schedules and unused children as well. All these indices are present from the start. The other schedule simulations are internal calculations, not additional members of the covering family.

## 6.3 Eligibility and Many Compatible Parents

An unused child can cover a newly activated function h if its parent has already made few errors against h: after assignment, the child predicts h exactly. We therefore test the parent’s actua predictions before the activation round. For $v \in \mathcal { P } , h \in \mathcal { H }$ , and $t \in [ T ]$ , write

$$
E _ { t - 1 } ( v , h ) : = \sum _ { s = 1 } ^ { t - 1 } \mathbf { 1 } \{ | p _ { s } ^ { v } - h ( x _ { s } ) | \geq 8 \eta \} .\tag{80}
$$

The sum is zero at $t = 1$

Definition 25 (Eligible parent). Suppose h activates on round t. A parent v is eligible for this activation if its schedule is valid and born by round $t - 1$ , and

$$
h \in V _ { v } , \qquad | V _ { v } | \geq 2 , \qquad E _ { t - 1 } ( v , h ) < b / 8 .\tag{81}
$$

Let $\mathcal { E } _ { j } ( h , t )$ be the set of eligible parents in pool j. Fix an arbitrary ordering of the sign schedules in $\mathcal { I } _ { d - 1 } ( m )$ . For each pool j with $\mathcal { E } _ { j } ( h , t ) \neq \emptyset$ , choose the eligible parent whose schedule comes first in this ordering. Let $\mathcal { E } ( h , t )$ be the set of these chosen parents.

We permit an activation of h on round t to use only a parent from $\mathcal { E } ( h , t )$ , and only an unused child of that parent.

The sets in this definition can be determined on round t. The function h is known from its certificates at activation; its values on earlier instances can therefore be evaluated. The parent states and all predictions used in (80) are already available. Neither the test nor the choice of a representative from each pool uses future instances or future accuracy. In particular, Good is not part of the eligibility test. Eligibility also does not depend on whether a parent’s children have already been used; a parent with no unused child still belongs to the same eligible sets.

If an unused child of $v \in \mathcal { E } ( h , t )$ is assigned to h on round t, then (78) gives

$$
\sum _ { s = 1 } ^ { T } \mathbf { 1 } \{ | g _ { s } ^ { v , \ell } - h ( x _ { s } ) | \geq 8 \eta \} = E _ { t - 1 } ( v , h ) < b / 8 .\tag{82}
$$

Indeed, before t the child copied its parent, and from t onward it predicts h exactly. Requiring assignments to use only the information just described makes these prediction rules causal in the instances and the revealed random choices.

There is also a uniform bound on how many distinct activations can have a given parent among their choices. Fix a valid, born parent v of depth i. Every function h that activates and has $v \in \mathcal { E } ( h , \tau ( h ) )$ belongs to $A \cap V _ { v }$ , since eligibility requires $h \in V _ { v }$ . By the local activation bound, the number of such functions is at most $2 Q _ { d - i } ( m )$ . Hence (77) and Corollary 24 give

$$
\# \{ h \in { \cal A } : v \in \mathcal { E } ( h , \tau ( h ) ) \} \leq | { \cal A } \cap V _ { v } | \leq 2 Q _ { d - i } ( m ) .\tag{83}
$$

A parent that never becomes valid and born is never eligible. This bound holds on every outcome, not only on Good, and allows the later input instances to depend on earlier assignment decisions.

Many eligible parents at activation. We now use the simultaneous replay accuracy from Proposition 21 to show that the eligible sets are large on Good.

Lemma 26 (Many eligible parents). On Good, every function h that activates on a round $t \in [ T ]$ satisfies

$$
| { \mathcal { E } } ( h , t ) | \geq k / 4 .\tag{84}
$$

Proof. Fix an outcome in Good and a function h activating on round t. Let $G _ { h } \subseteq [ k ]$ be the pools whose replays satisfy (65). By Proposition 21, $\vert G _ { h } \vert \ge 3 k / 4$ . Before round $t ,$ fewer than $k / 2$ pools had identified $h ;$ otherwise it would already have activated. Therefore

$$
\# \{ j \in G _ { h } : h \notin \mathcal { T } _ { j } ( t - 1 ) \} \geq | G _ { h } | - \# \{ j : h \in \mathbb { Z } _ { j } ( t - 1 ) \} > k / 4 .
$$

Fix a pool $j$ in this set, and let L consist of all corrections made by its target-driven replay through round $t - 1$ . On Cap these have ordinals in $[ m ]$ , and the capped-pool simulation of L is valid, born by t − 1, and has version space $U _ { t - 1 , j } ^ { h }$ . This space contains h. It cannot equal {h}, since (73) would imply $h \in \mathbb { Z } _ { j } ( t - 1 )$

Since H is uniformly greater-than-η separated, every nonsingleton subclass of H has rank at least one. If L has depth $i ,$ the rank-drop property consequently gives

$$
\begin{array} { r } { 1 \leq \mathrm { s f a t } _ { \eta } ( U _ { t - 1 , j } ^ { h } ) \leq d - i , \qquad \mathrm { s o ~ } i \leq d - 1 . } \end{array}
$$

Thus $v = ( j , L )$ is one of the preallocated parents. Its fixed version space is $V _ { v } = U _ { t - 1 , j } ^ { h }$ , which contains h and is not a singleton. By the prefix property in Section 4.2,

$$
p _ { s } ^ { v } = y _ { s , j } ^ { h } \qquad ( 1 \leq s < t ) .
$$

Since $j \in G _ { h }$ , it follows that

$$
E _ { t - 1 } ( v , h ) = \sum _ { s < t } \mathbf { 1 } \{ | y _ { s , j } ^ { h } - h ( x _ { s } ) | \geq 8 \eta \} \leq \sum _ { s = 1 } ^ { T } \mathbf { 1 } \{ | y _ { s , j } ^ { h } - h ( x _ { s } ) | \geq 8 \eta \} < b / 8 .
$$

All eligibility conditions hold, so $\mathcal { E } _ { j } ( h , t )$ is nonempty. This applies to more than $k / 4$ distinct pools. Since $\mathcal { E } ( h , t )$ contains one eligible parent from each such pool, we obtain (84). ■

The parent used in this proof is the separate schedule containing only the replay’s corrections before round t. Its version space remains fixed, even if the full replay makes further corrections or identifies h on round t. Only their predictions before t must agree: a child assigned on round t switches to h before emitting its prediction on that round.

Corollary 27 (Targets that do not activate). On Good, every $h \in { \mathcal { H } } \setminus A$ has a parent $v \in \mathcal { P }$ with

$$
\sum _ { t = 1 } ^ { T } \mathbf { 1 } \{ | p _ { t } ^ { v } - h ( x _ { t } ) | \geq 8 \eta \} < b / 8 .\tag{85}
$$

Proof. Fewer than $k / 2$ pools identify h by time T, while at least $3 k / 4$ replays satisfy (65). Choose a pool j with an accurate replay that has not identified h. Its final replay version space is not a singleton, by (73). The same rank argument as above shows that its entire replay code $J _ { j } ^ { h }$ has depth at most d − 1. On Cap this code has ordinals in [m], so $v = ( j , J _ { j } ^ { h } )$ is a parent and reproduces the replay on all $T$ rounds. Its final-prediction accuracy gives (85). ■

Thus a target that never activates is covered by a permanent parent. A target that activates has at least $k / 4$ eligible parents on Good, and assigning an unused child of any one of them gives (82). It remains to choose the capacities and distribute the activations among these parents so that, with positive probability jointly with Good, every activation receives an unused child and the count in (79) is at most $( C R ) ^ { d }$

## 7 Shared-Parent Routing and Load Balancing

We now specify how an activating function receives a child of one of its eligible parents. The rule chooses one parent uniformly from the retained set $\mathcal { E } ( h , t )$ when this set has at least $k / 4$ members. We bound how often a parent is chosen and use this bound to set the child capacities $c _ { i }$

## 7.1 The Causal Routing Protocol

For now, let the capacities $c _ { 0 } , \ldots , c _ { d } .$ <sub>−1</sub> be any fixed nonnegative integers, as in Section 6.2. By Corollary 24, the total number of functions that activate during a run is at most

$$
A _ { * } : = 2 Q _ { d } ( m ) .\tag{86}
$$

We allow $A _ { * }$ routing slots on each round, where one slot processes one newly activated function; slots with no function do nothing. Fix an arbitrary total order on $\mathcal { H } .$ . In each round, process the newly activated functions in this order.

For $t \in [ T ]$ and $a \in [ A _ { * } ]$ , let $\pi _ { t , a }$ be a uniformly random permutation of the finite parent set $\mathcal { P } . \mathrm { ~ \tiny ~ \textnormal ~ { ~ A l l } ~ }$ these permutations are mutually independent and independent of the raw pool bits. A permutation specifies an ordering of all parents. For an activation with $| \mathcal { E } ( h , t ) | \geq k / 4$ , conditional on the preceding history, each member of $\mathcal { E } ( h , t )$ is selected with probability $1 / | \mathcal { E } ( h , t ) |$ . The permutation $\pi _ { t , a }$ is revealed only in slot a of round t, including when that slot has no function to process.

Write $\pmb { \pi } _ { t } = ( \pi _ { t , 1 } , \ldots , \pi _ { t , A _ { * } } )$ and $\pmb { \pi } _ { < t } = ( \pmb { \pi } _ { 1 } , \dots , \pmb { \pi } _ { t - 1 } )$ . We allow inputs of the form

$$
x _ { t } = \widetilde { F } _ { t } ( \pmb { \xi } _ { < t } , \pmb { \pi } _ { < t } , \omega _ { \mathrm { i n } } ) ,\tag{87}
$$

where the maps $\widetilde { F } _ { t }$ are deterministic and $\omega _ { \mathrm { i n } }$ is a finite-support input seed independent of all pool bits and routing permutations. Thus $x _ { t }$ may depend on all earlier selections, assignments, and predictions, but is chosen before the current pool vector and current routing permutations are revealed.

Initialize a counter $L _ { v } = 0$ for each parent $v \in \mathcal { P }$ . A parent remains eligible after all its children have been assigned. If routing selects such a parent, its load counter increases and an overload is declared, but no child is assigned. At round t, reveal $\xi _ { t } ,$ perform the capped schedule updates, and determine the newly activated functions $\mathbf { \mathcal { A } } _ { t } \setminus \bigcup _ { t = 1 }$ . List them in the fixed order as $h _ { t , 1 } , \ldots , h _ { t , n _ { t } } ,$ where $n _ { t } \le A _ { * }$ . The eligible set $\mathcal { E } ( h _ { t , a } , t )$ does not depend on the current routing permutation $\pi _ { t , a }$ In each slot $a \in [ A _ { * } ]$ , reveal $\pi _ { t , a }$ and proceed as follows. If $a > n _ { t } .$ , make no change. Otherwise put $h = h _ { t , a \cdot } \ \mathrm { I f } \ | { \mathcal E } ( h , t ) | < k / 4 .$ , leave h unassigned. I $: | \mathcal { E } ( h , t ) | \ge k / 4$ , choose the first parent v in $\mathcal { E } ( h , t )$ according to $\pi _ { t , a }$ and increase $L _ { v }$ by one. Write ℓ for its updated value and i for the depth of v. When $\ell \leq c _ { i }$ , assign child $( v , \ell )$ to h. When $\ell > c _ { i }$ , make no child assignment; this is an overload at v.

The first $c _ { i }$ choices of a depth-i parent use its children in order, so every assigned child is unused at assignment. Each function is processed only on its activation round and only once. After all slots of round t, emit the parent predictions and the child predictions specified in (78).

We continue the selections and counters after an overload, and also after an activation is left unassigned. Thus $L _ { v }$ counts routing choices, not just successful child assignments. On Good, the case $| \mathcal { E } ( h , t ) | < k / 4$ cannot occur, by Lemma 26.

The sampling guarantee from the preceding sections applies to (87). Indeed, the auxiliary seed

$$
\omega _ { 0 } : = \left( \omega _ { \mathrm { i n } } , ( \pi _ { t , a } ) _ { t \in [ T ] , a \in [ A _ { * } ] } \right)
$$

has finite support and is independent of all raw pool bits. Given this seed, every earlier assignment is determined by the past raw vectors. Hence the input rule has the form (50), and Proposition 21 gives $\operatorname* { P r } ( \mathsf { G o o d } ) > 1 5 / 1 6$ for the routing protocol just defined.

## 7.2 Routing Probabilities and Load Bounds

Fix a parent v. Consider a routing step (t, a) that processes a newly activated function $h _ { t , a }$ . Define

$$
B _ { t , a } ^ { v } : = \mathbf { 1 } \big \{ | \mathcal { E } ( h _ { t , a } , t ) | \geq k / 4 \mathrm { ~ a n d ~ } v \in \mathcal { E } ( h _ { t , a } , t ) \big \} ,\tag{88}
$$

and let $X _ { t , a } ^ { v }$ be the indicator that the routing rule chooses v in this slot, whether or not a child is assigned. For an empty slot, set both indicators to zero. Thus

$$
N _ { v } : = \sum _ { t = 1 } ^ { T } \sum _ { a = 1 } ^ { A _ { * } } B _ { t , a } ^ { v } \qquad \mathrm { a n d } \qquad L _ { v } = \sum _ { t = 1 } ^ { T } \sum _ { a = 1 } ^ { A _ { * } } X _ { t , a } ^ { v }
$$

count, respectively, the activations at which v could be chosen and the activations at which it is chosen.

If v has depth i, then on every outcome

$$
N _ { v } \leq \# \{ h \in \mathcal { A } : v \in \mathcal { E } ( h , \tau ( h ) ) \} \leq 2 Q _ { d - i } ( m ) .\tag{89}
$$

The first inequality holds because each function activates only once; the size test in (88) can only remove activations from the count. The second inequality is (83). A parent that never becomes valid and born has $N _ { v } = L _ { v } = 0$

To describe the information available at a routing choice, let

$$
\begin{array} { r l } & { \mathcal { F } _ { t , 0 } : = \sigma \big ( \omega _ { \mathrm { i n } } , \xi _ { 1 } , \ldots , \xi _ { t } , \pi _ { 1 } , \ldots , \pi _ { t - 1 } \big ) , } \\ & { \mathcal { F } _ { t , a } : = \sigma \big ( \mathcal { F } _ { t , 0 } , \pi _ { t , 1 } , \ldots , \pi _ { t , a } \big ) \qquad ( 1 \leq a \leq A _ { * } ) . } \end{array}
$$

Also put $\mathcal { F } _ { 0 , A _ { * } } = \sigma ( \omega _ { \mathrm { i n } } )$ . These sigma-fields reveal the raw vector first on each round and then the routing permutations one at a time. The functions $h _ { t , a }$ and their eligible sets are already determined by $\mathcal { F } _ { t , 0 }$ . In particular, $B _ { t , a } ^ { v }$ is known given $\mathcal { F } _ { t , a - 1 }$ , while $\pi _ { t , a }$ remains an independent uniform permutation. Consequently,

$$
\operatorname* { P r } \{ X _ { t , a } ^ { v } = 1 \mid \mathcal { F } _ { t , a - 1 } \} = \left\{ \begin{array} { l l } { | \mathcal { E } ( h _ { t , a } , t ) | ^ { - 1 } , } & { B _ { t , a } ^ { v } = 1 , } \\ { 0 , } & { B _ { t , a } ^ { v } = 0 } \end{array} \right. \leq \frac { 4 } { k } B _ { t , a } ^ { v } .\tag{90}
$$

For the first case, each member of the eligible set is equally likely to appear first in the fresh permutation. This remains true when v has no unused children, since routing still chooses from the same set and increments $L _ { v }$

Load supermartingale and child capacities. We combine the bound on how often v can be considered with the conditional probability that it is chosen. The routing indicators $X _ { t , a } ^ { v }$ need not be independent across rounds, since later inputs and eligible sets may depend on earlier routing choices.

Lemma 28 (Exponential moment of a parent load). For every fixed parent v of depth i, every fixed choice of capacities, and every input rule (87),

$$
\mathbb { E } e ^ { L _ { v } } \leq \exp \left( \frac { 8 ( e - 1 ) Q _ { d - i } ( m ) } { k } \right) .\tag{91}
$$

Proof. Let $N _ { v } ( t , a )$ and $L _ { v } ( t , a )$ be the two counts through slot a of round $t ,$ including all earlier rounds. Both counts are initially zero. Here $a = 0$ denotes the start of round t, before any routing steps, and $a = A _ { * }$ <sub>∗</sub> denotes the end of that round, after all routing steps. Thus, for $t \geq 2$

$$
N _ { v } ( t , 0 ) = N _ { v } ( t - 1 , A _ { * } ) , \qquad L _ { v } ( t , 0 ) = L _ { v } ( t - 1 , A _ { * } ) .
$$

Initially, $N _ { v } ( 1 , 0 ) = L _ { v } ( 1 , 0 ) = 0$ . Set $\lambda : = 4 ( e - 1 ) / k$ and define

$$
Z _ { t , a } : = \exp \bigl ( L _ { v } ( t , a ) - \lambda N _ { v } ( t , a ) \bigr ) .\tag{92}
$$

Its initial value is one. At a raw-vector reveal it is unchanged. At a routing slot, the increments of $L _ { v }$ and $N _ { v }$ are $X _ { t , a } ^ { v }$ and $B _ { t , a } ^ { v }$ , respectively, so

$$
\frac { Z _ { t , a } } { Z _ { t , a - 1 } } = \exp \bigl ( X _ { t , a } ^ { v } - \lambda B _ { t , a } ^ { v } \bigr ) .
$$

Since $X _ { t , a } ^ { v } \in \{ 0 , 1 \}$ and $B _ { t , a } ^ { v }$ is $\mathcal { F } _ { t , a }$ <sub>−1</sub>-measurable, (90) gives

$$
\begin{array} { r l } & { \mathbb { E } \left[ e ^ { X _ { t , a } ^ { v } } \ \big \vert \ \mathcal { F } _ { t , a - 1 } \right] = 1 + \left( e - 1 \right) \operatorname* { P r } \{ X _ { t , a } ^ { v } = 1 \ \vert \ \mathcal { F } _ { t , a - 1 } \} } \\ & { \qquad \leq 1 + \lambda B _ { t , a } ^ { v } \leq e ^ { \lambda B _ { t , a } ^ { v } } . } \end{array}
$$

It follows that

$$
\mathbb { E } \left[ \frac { Z _ { t , a } } { Z _ { t , a - 1 } } \bigg | \mathcal { F } _ { t , a - 1 } \right] \leq 1 .
$$

Thus $Z$ is a nonnegative supermartingale along the finite sequence of raw-vector reveals and routing slots, with respect to the sigma-fields just defined. In particular, $\mathbb { E } Z _ { T , A _ { \ast } } \leq 1$

By (89), $N _ { v } \le 2 Q _ { d - i } ( m )$ on every outcome. Therefore

$$
\begin{array} { r l } { \mathbb { E } e ^ { L _ { v } } = \mathbb { E } \big [ Z _ { T , A _ { * } } e ^ { \lambda N _ { v } } \big ] } & { } \\ { \ } & { \leq e ^ { 2 \lambda Q _ { d - i } ( m ) } \mathbb { E } Z _ { T , A _ { * } } } \\ { \ } & { \leq \exp \biggl ( \frac { 8 ( e - 1 ) Q _ { d - i } ( m ) } { k } \biggr ) . } \end{array}
$$

We can now choose the number of children for each parent. The exponential moment bound turns an additive $\log ( 4 P )$ in the capacity into a failure probability of at most $1 / ( 4 P )$ for each parent. Since $e - 1 < 2$ , set

$$
c _ { i } : = \left\lceil \frac { 1 6 Q _ { d - i } ( m ) } { k } + \log ( 4 P ) \right\rceil , \qquad 0 \leq i \leq d - 1 .\tag{93}
$$

These values depend only on the parameters fixed before the experiment.

Proposition 29 (No overload). With capacities (93), define

$$
\mathsf { N o O v e r l o a d : } = \{ L _ { v } \leq c _ { i } \ f o r \ e v e r y \ d e p t h - i \ p a r e n t \ v \in \mathcal { P } , \ 0 \leq i \leq d - 1 \} .\tag{94}
$$

For every input rule (87),

$$
\operatorname* { P r } ( \mathsf { N o O v e r l o a d } ^ { \mathsf { c } } ) \leq \frac { 1 } { 4 } .\tag{95}
$$

Proof. Fix a parent v of depth i. Markov’s inequality and Lemma 28 imply

$$
\begin{array} { r l } & { \operatorname* { P r } \{ L _ { v } > c _ { i } \} \leq e ^ { - c _ { i } } \mathbb { E } e ^ { L _ { v } } } \\ & { \qquad \leq \exp \biggl ( - c _ { i } + \frac { 8 ( e - 1 ) Q _ { d - i } ( m ) } { k } \biggr ) } \\ & { \qquad \leq \exp \biggl ( - c _ { i } + \frac { 1 6 Q _ { d - i } ( m ) } { k } \biggr ) \leq \frac { 1 } { 4 P } . } \end{array}
$$

Taking a union bound over the P parent indices proves (95).

The overload bound is under the full joint law of sampling and routing, without conditioning on Cap or Good. On Good ∩ NoOverload, every activating function receives an unused child on its activation round: its eligible set has at least $k / 4$ members, and the selected parent’s counter never exceeds its capacity.

## 8 Expert Counting and Global Determinization

For the counting and probability arguments, retain the separated class H and the regime $d \geq 1$ $| \mathcal { H } | \geq 2$ , and $b > 3 2 \alpha d$ from the preceding section. The child capacities are now fixed by (93).

## 8.1 Expert Counting and Simultaneous Coverage

Lemma 30 (Expert count). There is a universal constant $C _ { \mathrm { c n t } } \geq 1$ such that the number K of preallocated parent and child indices satisfies

$$
K \leq ( C _ { \mathrm { c n t } } R ) ^ { d } .\tag{96}
$$

Proof. There are $k n _ { i } ( m )$ parents of depth $i ,$ each with $c _ { i }$ children. Using (79), (93), and $\lceil z \rceil \leq z + 1$ 2 we obtain

$$
\begin{array} { r l } {  { K = P + \sum _ { i = 0 } ^ { d - 1 } k n _ { i } ( m ) c _ { i } } } \\ & { \leq 1 6 \sum _ { i = 0 } ^ { d - 1 } n _ { i } ( m ) Q _ { d - i } ( m ) + P \big ( \log ( 4 P ) + 2 \big ) } \\ & { \leq 1 6 Q _ { d } ( 2 m ) + P \big ( \log ( 4 P ) + 2 \big ) . } \end{array}\tag{97}
$$

The factor k from the $k n _ { i } ( m )$ depth-i parents cancels with the $1 / k$ in the leading term of $c _ { i }$ . The term $P ( \log ( 4 P ) + 2 )$ accounts for the permanent parents, the logarithmic capacity overhead, and the rounding overhead. The last inequality is (37).

Put $C _ { 1 } : = 1 4 e \alpha$ . Since $m \leq 3 \alpha d R$ , (35) gives

$$
{ Q } _ { d } ( 2 m ) \leq \left[ 2 e ( 1 + 6 \alpha R ) \right] ^ { d } \leq ( C _ { 1 } R ) ^ { d } .
$$

For $d \geq 2$ , the inequality $d / ( d - 1 ) \leq 2$ similarly gives

$$
Q _ { d - 1 } ( m ) \leq \left[ 2 e \left( 1 + \frac { 3 \alpha d R } { d - 1 } \right) \right] ^ { d - 1 } \leq ( C _ { 1 } R ) ^ { d - 1 } .
$$

For $d = 1$ , the same final bound holds because $Q _ { 0 } ( m ) = 1$ . Thus, in all cases,

$$
P = k Q _ { d - 1 } ( m ) \leq k ( C _ { 1 } R ) ^ { d - 1 } .\tag{98}
$$

We next bound the second term in (97). The definition $k = 4 \lceil \log _ { 2 } R \rceil$ and $R > 3 2$ imply $k \leq 8 \log R _ { \mathrm { : } }$ and (66) gives $k \leq R .$ . Consequently,

$$
\log ( 4 P ) + 2 \leq \log ( 4 k ) + 2 + ( d - 1 ) \log ( C _ { 1 } R ) \leq C _ { 2 } d \log R ,
$$

where we may take the universal constant $C _ { 2 } : = 1 + ( \log ( 4 C _ { 1 } ) + 2 ) /$ log 32, using log $k \leq \log R$ and log $R > \log 3 2$ . It follows that

$$
\begin{array} { r l } & { P \big ( \log ( 4 P ) + 2 \big ) \leq 8 C _ { 2 } d ( \log R ) ^ { 2 } ( C _ { 1 } R ) ^ { d - 1 } } \\ & { \qquad \leq 8 C _ { 2 } d R ( C _ { 1 } R ) ^ { d - 1 } } \\ & { \qquad \leq ( C _ { 3 } R ) ^ { d } } \end{array}\tag{99}
$$

for a universal $C _ { 3 }$ . The second inequality uses (log $R ) ^ { 2 } \leq R$ for $R > 3 2 ;$ the last follows from $d \leq 2 ^ { d }$ , for example with $C _ { 3 } = 2 \operatorname* { m a x } \{ C _ { 1 } , 8 C _ { 2 } \}$ . Combining these bounds in (97), we may take $C _ { \mathrm { c n t } } = 1 6 C _ { 1 } + C _ { 3 }$ . All these constants are universal because α was fixed universally in (68). ■

Simultaneous coverage. Number the K preallocated parent and child indices by $\nu \in [ K ]$ , and let $w _ { t } ^ { \nu }$ denote the prediction of index ν at round t under the randomized sampling and routing protocol.

Lemma 31 (Simultaneous coverage). For every input rule (87), with probability greater than $1 / 2$ the outputs satisfy

$$
\forall h \in { \mathcal { H } } \exists \nu \in [ K ] : \qquad \sum _ { t = 1 } ^ { T } \mathbf { 1 } \{ | w _ { t } ^ { \nu } - h ( x _ { t } ) | \geq 8 \eta \} < b / 8 .\tag{100}
$$

Proof. The application of Proposition 21 in Section 7.1 and Proposition 29 give

$$
\begin{array} { r } { \mathrm { P r } ( \mathsf { G o o d } \cap \mathsf { N o O v e r l o a d } ) \geq 1 - \mathrm { P r } ( \mathsf { G o o d } ^ { \mathsf { c } } ) - \mathrm { P r } ( \mathsf { N o O v e r l o a d } ^ { \mathsf { c } } ) } \\ { > 1 - \displaystyle \frac { 1 } { 1 6 } - \frac { 1 } { 4 } = \displaystyle \frac { 1 1 } { 1 6 } > \frac { 1 } { 2 } . \qquad } \end{array}\tag{101}
$$

Fix an outcome in this intersection and $h \in \mathcal H$ . If h activates, the routing protocol assigns it an unused child of a parent in $\mathcal { E } ( h , \tau ( h ) )$ , as proved at the end of Section 7. By (82), this child’s index supplies the required witness for h. If h does not activate, Corollary 27 supplies a parent with the same error bound. In either case, the witnessing index belongs to the preallocated family. Since h was arbitrary, the one event in (101) gives the stated coverage for every target. ■

## 8.2 Game Determinization and Main Upper Bound

Lemma 31 gives successful random choices for each fixed input rule. Choosing one successful realization separately for each instance sequence, however, need not give the same predictions on shared prefixes. To obtain a global family, we need one deterministic rule for choosing the pool selections and routing permutations after each observed input prefix. The guarantee against adaptive input rules allows us to obtain such a rule by the following game argument.

Lemma 32 (Finite-horizon determinization). Consider a game of T rounds between an adversary and a constructor. In each round the adversary first chooses an input from a nonempty set, and the constructor then chooses an action from a nonempty set. The legal choices may depend on the preceding moves. Both players observe all earlier moves, and the constructor also observes the current input. Each terminal history is designated winning or losing for the constructor.

If one randomized constructor strategy wins with positive probability against every deterministic adversary strategy, then there is a deterministic constructor strategy that wins against every adversary strategy.

Proof. Label each terminal history according to the prescribed winning condition. Working backwards through nonterminal histories, label a constructor decision node as winning if some legal action leads to a winning successor. Label a history at which the adversary is to move as winning if every legal input leads to a winning history. All other histories are losing. The recursion is well-defined because the game has finite depth, even if its branching is infinite.

Suppose that the initial history were losing. At each losing history where the adversary moves, there is an input leading to a losing history. Choose one such input at each of these histories, and choose arbitrary legal inputs at the adversary’s other histories. At a losing history where the constructor moves, every legal action leads to a losing history. The chosen adversary strategy therefore forces a losing terminal history for every realization of the constructor’s random choices. Its winning probability against this deterministic adversary would be zero, a contradiction.

The initial history is consequently winning. At each winning history where the constructor moves, fix an action leading to a winning history; specify arbitrary legal actions at its other histories. At every winning history where the adversary moves, all legal inputs preserve the winning status. These choices define a deterministic constructor strategy that wins against all adversary moves. The choices over arbitrary collections of histories may use the axiom of choice. ■

Remark 33 (Role of the information structure). The determinization argument relies on perfect information: before choosing the next input, the adversary observes the constructor’s realized actions, not merely their distributions. This difers from distribution-observing protocols in randomized online learning with set-valued or partial feedback [RST24; Sha+26]. Without access to the realized actions, the conclusion of Lemma 32 can fail, even in a two-round game. For example, suppose the constructor secretly chooses a fair bit and wins if it equals the adversary’s second-round input. The constructor then wins with probability 1/2 against every deterministic adversary, but no deterministic constructor strategy can guarantee a win.

Proof of Proposition 2. The case $d = 0$ was proved in Section 4.1; a singleton H is covered exactly by its unique member. For $d \geq 1$ and $b \leq 3 2 \alpha d .$ , the deterministic family in (46) already proves the proposition. We therefore consider the remaining regime and use the randomized construction counted in Lemma 30.

Let ${ \mathfrak { S } } ( { \mathcal { P } } )$ be the finite set of permutations of P. Define a game in which the adversary chooses $x _ { t } \in \mathcal { X }$ on round t, and the constructor then chooses

$$
( \pmb { \xi } _ { t } , \pmb { \pi } _ { t } ) \in \{ 0 , 1 \} ^ { k } \times \mathfrak { S } ( \mathcal { P } ) ^ { A _ { * } } .
$$

Apply the capped schedule updates and routing protocol to these choices, processing the routing slots in their prescribed order, and emit the K predictions before the next adversary move. All these operations are determined by the inputs and chosen actions. Designate a terminal history as winning precisely when (100) holds.

Every deterministic adversary strategy in this game has the form (87) with a constant input seed: its earlier inputs, and hence its current choice, can be recovered recursively from the preceding pool vectors and routing permutations. The randomized protocol therefore wins with probability greater than $1 / 2$ against every such strategy, by Lemma 31. Lemma 32 supplies a deterministic winning constructor strategy. Fix this strategy.

For any input prefix $x _ { 1 : t } .$ , run the fixed strategy through round t. By induction on rounds, all pool vectors, routing permutations, assignments, and predictions through that round are determined by $x _ { 1 : t }$ . For each $\nu \in [ K ]$ , define

g<sub>ν,t</sub>(x<sub>1:t</sub>) := the output of index ν on round t in this run.

Thus $g _ { \nu , t } : \mathcal { X } ^ { t } \to [ 0 , 1 ]$ is a well-defined deterministic causal map. The family

$$
\mathcal { G } : = \{ ( g _ { \nu , t } ) _ { t = 1 } ^ { T } : \nu \in [ K ] \}
$$

is fixed independently of the instance sequence. Since the strategy wins against every sequence of adversary inputs,

$$
\forall x _ { 1 : T } \in \mathcal { X } ^ { T } \forall h \in \mathcal { H } \exists \nu \in [ K ] : \qquad \sum _ { t = 1 } ^ { T } \mathbf { 1 } \{ | g _ { \nu , t } ( x _ { 1 : t } ) - h ( x _ { t } ) | \geq 8 \eta \} < \frac { T } { 8 R } .\tag{102}
$$

Finally, $| \mathcal { G } | \le K \le { ( C _ { \mathrm { c n t } } R ) ^ { d } }$ . Choosing one universal $C _ { 0 } \geq \operatorname* { m a x } \{ 1 , C _ { \mathrm { c n t } } , 6 6 e \alpha \}$ also includes the short-horizon bound (46), and proves the proposition in all regimes. ■

Proof of Theorem 5. Fix $1 \leq p < \infty , T \geq 1$ , and $0 < \varepsilon < 1 / 2$ , and put $\eta = \varepsilon / 3 2$ and $d = \mathrm { s f a t } _ { \eta } ( \mathcal { F } )$ If $d = 0$ , the midpoint rule

$$
g _ { t } ( x _ { 1 : t } ) : = \frac { 1 } { 2 } \left( \operatorname* { i n f } _ { f \in \mathcal { F } } f ( x _ { t } ) + \operatorname* { s u p } _ { f \in \mathcal { F } } f ( x _ { t } ) \right)
$$

has pointwise error at most $\eta / 2$ against every $f \in { \mathcal { F } }$ , by Lemma 10. Its normalized $\ell _ { p }$ error is therefore at most $\eta / 2 < \varepsilon .$ , and one expert sufices.

Suppose $d \geq 1$ . Lemma 9 gives a separated subclass $\mathcal { H } \subseteq \mathcal { F }$ of rank at most d such that

$$
\forall f \in { \mathcal { F } } \exists h \in { \mathcal { H } } : \quad \| f - h \| _ { \infty } \leq \eta .
$$

Apply Proposition 2 with

$$
R : = \frac { 1 6 } { \varepsilon ^ { p } } > 3 2 .\tag{103}
$$

The resulting global family has size at most $( 1 6 C _ { 0 } / \varepsilon ^ { p } ) ^ { d }$

Fix an instance sequence $x _ { 1 : T }$ and a target $f \in { \mathcal { F } }$ . Choose $h \in \mathcal H$ with $\| f - h \| _ { \infty } \leq \eta$ and a witnessing expert g for h from Proposition 2. Writing

$$
B : = \{ t \in [ T ] : | g _ { t } ( x _ { 1 : t } ) - h ( x _ { t } ) | \geq 8 \eta \} ,
$$

we have $| B | / T < 1 / ( 8 R )$ . For $t \notin B$

$$
| g _ { t } ( x _ { 1 : t } ) - f ( x _ { t } ) | \leq | g _ { t } ( x _ { 1 : t } ) - h ( x _ { t } ) | + | h ( x _ { t } ) - f ( x _ { t } ) | < 9 \eta .
$$

For $t \in B$ , the error is at most one because both values lie in [0, 1]. Consequently,

$$
\begin{array} { l } { \displaystyle \frac { 1 } { T } \sum _ { t = 1 } ^ { T } | g _ { t } ( x _ { 1 : t } ) - f ( x _ { t } ) | ^ { p } \leq ( 9 \eta ) ^ { p } + \displaystyle \frac { | B | } { T } } \\ { \displaystyle < \varepsilon ^ { p } \left[ \left( \frac { 9 } { 3 2 } \right) ^ { p } + \frac { 1 } { 1 2 8 } \right] } \\ { \displaystyle \leq \frac { 3 7 } { 1 2 8 } \varepsilon ^ { p } < \varepsilon ^ { p } . } \end{array}\tag{104}
$$

Here $( 9 / 3 2 ) ^ { p } \leq 9 / 3 2$ because $p \geq 1$ . Taking the pth root gives the required normalized $\ell _ { p }$ error bound. The separated subclass and the expert family were fixed independently of $f$ and $x _ { 1 : T } ,$ so the same family satisfies Definition 4. Taking $C = 1 6 C _ { 0 }$ proves (19), with $C$ independent of $p , d ,$ $T _ { i }$ , and ε.

The upper-bound assertions of Corollaries 7 and 8 now follow from the rounding and treerestriction arguments in Section 3.4. Moreover, $C \geq 1$ and $p \geq 1$ give

$$
\left( \frac { C } { { \varepsilon } ^ { p } } \right) ^ { d } \leq \left( \frac { C } { \varepsilon } \right) ^ { p d } ,
$$

which also proves the upper-bound part of Theorem 1.

## 9 Matching Lower Bounds

We prove Theorem 6 using indicators of afine hyperplanes over a finite field. The instance sequence lists every point of the domain once. Related finite-field hyperplane classes have been studied through their static VC dimension $[ \mathrm { A s c + 2 5 } ] ;$ here we require the Littlestone dimension of the full afine-hyperplane class and a fixed-sequence $\ell _ { p }$ packing.

## 9.1 Finite-Field Hyperplanes and Their Dimensions

Fix an integer $d \geq 1$ and a prime power $q \geq 2$ , and let $\mathbb { F } _ { q }$ be a field with q elements. On the domain $\mathcal { X } _ { q } : = \mathbb { F } _ { q } ^ { d }$ , define

$$
H _ { a , b } : = \{ x \in \mathbb { F } _ { q } ^ { d } : a \cdot x = b \} , \qquad h _ { a , b } ( x ) : = \mathbf { 1 } \{ x \in H _ { a , b } \} ,\tag{105}
$$

where $a \in \mathbb { F } _ { q } ^ { d } \setminus \{ 0 \} , b \in \mathbb { F } _ { q } ,$ and $\textstyle { a \cdot x } = \sum _ { j = 1 } ^ { d } a _ { j } x _ { j }$ is computed in $\mathbb { F } _ { q }$ . Let

$$
\mathcal { H } _ { q } : = \{ h _ { a , b } : a \in \mathbb { F } _ { q } ^ { d } \setminus \{ 0 \} , \ b \in \mathbb { F } _ { q } \} .
$$

Each $H _ { a , b }$ is an afine subspace of dimension $d - 1$ and has $q ^ { d - 1 }$ points: if $a _ { j } \neq 0$ , choosing the other $d - 1$ coordinates freely determines $x _ { j }$ uniquely.

Two parameter pairs $( a , b )$ and $( a ^ { \prime } , b ^ { \prime } )$ define the same hyperplane exactly when $( a ^ { \prime } , b ^ { \prime } ) = ( \lambda a , \lambda b )$ for some $\lambda \in \mathbb { F } _ { q } \setminus \{ 0 \}$ . Indeed, equal hyperplanes have the same direction subspace, so the nonzero linear forms a · x and $a ^ { \prime } \cdot x$ have the same kernel and are proportional. Evaluating them at any point of the hyperplane then gives $b ^ { \prime } = \lambda b$ . The converse follows by multiplying the defining equation by λ. Thus each distinct hyperplane has exactly $q - 1$ parameter representations, and

$$
| \mathcal { H } _ { q } | = \frac { q ( q ^ { d } - 1 ) } { q - 1 } = q \sum _ { j = 0 } ^ { d - 1 } q ^ { j } \ge q ^ { d } .\tag{106}
$$

Lemma 34 (Dimension of the hyperplane class). For every prime power $q \geq 2$ and integer $d \geq 1$

$$
\mathrm { L d i m } ( \mathcal { H } _ { q } ) = \mathrm { s f a t } _ { \gamma } ( \mathcal { H } _ { q } ) = d \qquad ( 0 < \gamma \leq 1 ) .\tag{107}
$$

Proof. For the upper bound on Littlestone dimension, consider a binary predictor that observes the true label after each prediction. It maintains the afine span S of all previously observed instances with label one, initially $S = \emptyset$ , and predicts one exactly when the current instance belongs to S. After a positive label it replaces S by the afine span of $S$ and the current instance; after a negative label it leaves $S$ unchanged.

On any sequence labeled by the indicator of a fixed hyperplane $H ,$ , we always have $S \subseteq H$ . The predictor therefore never predicts one at a point labeled zero. Every mistake is a positive point outside $S _ { i }$ , whose addition increases the afine dimension by one. Assigning dimension −1 to the initially empty span, and using dim $S <$ dim $H = d - 1$ , we obtain at most d mistakes.

If $\mathcal { H } _ { q }$ shattered a Littlestone tree of depth $d { + 1 }$ , we could follow, at each node, the edge opposite the predictor’s current prediction and provide that edge’s label before proceeding to the next node. Some member of $\mathcal { H } _ { q }$ realizes the resulting path. The predictor would then make $d + 1$ mistakes on a sequence labeled by that member, contradicting the bound just proved. Hence Ldim $( { \mathcal { H } } _ { q } ) \leq d .$

For the lower bound, let $e _ { 1 } , \ldots , e _ { d }$ be the standard basis vectors in $\mathbb { F } _ { q } ^ { d } .$ Every labeling of these points is realized by a hyperplane indicator. To see this, let $S \subseteq [ d ]$ be the indices to receive label one. If $S \neq [ d ]$ , take $b = 0$ and

$$
a _ { j } = { \Bigg \{ } { 0 , } \quad j \in S ,
$$

Then $a \neq 0$ and $h _ { a , 0 } ( e _ { j } ) = 1$ exactly for $j \in S$ . If $S = [ d ]$ , take $a = ( 1 , \ldots , 1 )$ and $b = 1 ;$ every $e _ { j }$ belongs to this hyperplane. The complete binary instance tree that queries $e _ { j }$ at every node of level $j$ is therefore shattered. This gives Ldim $( { \mathcal { H } } _ { q } ) \geq d$

Finally, $\mathcal { H } _ { q }$ is binary, so the equality of sequential fat-shattering and Littlestone dimensions at scales $0 < \gamma \leq 1$ , proved in Section 3.1, gives (107). ■

## 9.2 Fixed-Sequence Packing and Lower Bound

List all points of $\mathcal { X } _ { q }$ once, in any fixed order, as $x _ { 1 } , \ldots , x _ { T }$ , where $T = q ^ { d }$ . For distinct hyperplanes $H , H ^ { \prime }$ and their indicators $h , h ^ { \prime }$ , the normalized metric from (17) satisfies

$$
\rho _ { p , T } \big ( h [ x _ { 1 : T } ] , h ^ { \prime } [ x _ { 1 : T } ] \big ) ^ { p } = \frac { | H \triangle H ^ { \prime } | } { q ^ { d } } \qquad ( 1 \leq p < \infty ) ,\tag{108}
$$

where $H \triangle H ^ { \prime }$ is the set of points belonging to exactly one of the two hyperplanes. This identity holds because the absolute diference of two binary values, raised to the power $p ,$ is one precisely when they difer.

If the normal vectors of H and $H ^ { \prime }$ are proportional, the two distinct hyperplanes are parallel and disjoint. Each has $q ^ { d - 1 }$ points, so the right-hand side of (108) is $2 / q$ . Otherwise the two normal vectors are linearly independent, which can occur only when $d \geq 2$ . Their two defining equations have $q ^ { d - 2 }$ common solutions, and hence

$$
\frac { | H \triangle H ^ { \prime } | } { q ^ { d } } = \frac { 2 q ^ { d - 1 } - 2 q ^ { d - 2 } } { q ^ { d } } = \frac { 2 ( q - 1 ) } { q ^ { 2 } } .
$$

For $d = 1$ , hyperplanes are singleton sets, so the value $2 / q$ applies to every distinct pair. Thus, for all $d \geq 1$

$$
\rho _ { p , T } \big ( h [ x _ { 1 : T } ] , h ^ { \prime } [ x _ { 1 : T } ] \big ) ^ { p } \geq \frac { 2 ( q - 1 ) } { q ^ { 2 } } \qquad ( h \not = h ^ { \prime } , \ h , h ^ { \prime } \in \mathcal { H } _ { q } ) .\tag{109}
$$

Choice of the field size. To obtain pairwise $\rho _ { p , 1 }$ <sub>T</sub>-distance greater than $2 \varepsilon$ , fix $1 \leq p < \infty$ and $0 < \varepsilon < 2 ^ { - 1 - 1 / p }$ , and choose

$$
q : = 2 ^ { \left\lfloor \log _ { 2 } \left( ( 2 \varepsilon ) ^ { - p } \right) \right\rfloor } .\tag{110}
$$

Since $( 2 \varepsilon ) ^ { - p } > 2$ , this is a power of two with $q \geq 2$ . It is the largest power of two not exceeding $( 2 \varepsilon ) ^ { - p }$ , so $q \leq ( 2 \varepsilon ) ^ { - p } < 2 q$ . Equivalently,

$$
\frac { 1 } { 2 ^ { p + 1 } \varepsilon ^ { p } } < q \leq \frac 1 { 2 ^ { p } \varepsilon ^ { p } } .\tag{111}
$$

The separation is strict in both possible cases:

$$
\begin{array} { c c } { { q > 2 : } } & { { \displaystyle \frac { 2 ( q - 1 ) } { q ^ { 2 } } > \frac { 1 } { q } \geq ( 2 \varepsilon ) ^ { p } , } } \\ { { q = 2 : } } & { { \displaystyle \frac { 2 ( q - 1 ) } { q ^ { 2 } } = \frac { 1 } { 2 } > ( 2 \varepsilon ) ^ { p } . } } \end{array}
$$

The first line uses (111), and the second uses $\varepsilon < 2 ^ { - 1 - 1 / p }$ . By (109), we conclude that

$$
\rho _ { p , T } \big ( h [ x _ { 1 : T } ] , h ^ { \prime } [ x _ { 1 : T } ] \big ) > 2 \varepsilon \qquad ( h \neq h ^ { \prime } , ~ h , h ^ { \prime } \in \mathcal { H } _ { q } ) .\tag{112}
$$

Proof of Theorem 6. Take $q$ as in (110), the class $\mathcal { H } _ { q }$ , and the fixed sequence of all $T = q ^ { d }$ domain points. Lemma 34 gives $\mathrm { s f a t } _ { \varepsilon / 3 2 } ( \mathcal { H } _ { q } ) = d .$

An arbitrary real-valued prediction vector $w \in [ 0 , 1 ] ^ { T }$ cannot approximate two distinct members $h , h ^ { \prime } \in \mathcal { H } _ { q }$ to normalized $\ell _ { p }$ error at most ε. Otherwise, the triangle inequality would give

$$
\begin{array} { r } { \rho _ { p , T } \big ( h [ x _ { 1 : T } ] , h ^ { \prime } [ x _ { 1 : T } ] \big ) \leq \rho _ { p , T } \big ( h [ x _ { 1 : T } ] , w \big ) + \rho _ { p , T } \big ( w , h ^ { \prime } [ x _ { 1 : T } ] \big ) \leq 2 \varepsilon , } \end{array}
$$

contrary to (112). Thus any family of vectors covering all target vectors on this fixed sequence has at least $| \mathcal { H } _ { q } |$ members.

Every deterministic causal expert g produces one such vector $g [ x _ { 1 : T } ]$ on the fixed sequence. Consequently, (106) and (111) yield

$$
N _ { p } ^ { \mathrm { d y n } } ( \mathcal { H } _ { q } , \varepsilon , q ^ { d } ) \ge | \mathcal { H } _ { q } | \ge q ^ { d } > \left( \frac { 1 } { 2 ^ { p + 1 } \varepsilon ^ { p } } \right) ^ { d } .\tag{113}
$$

The argument already allowed arbitrary vectors in $[ 0 , 1 ] ^ { T }$ ; hence it also proves the theorem’s fixedsequence assertion. ■

Taking the supremum in (21) gives the lower bound in (22). The comparisons in Sections 3.3 and 3.4 then complete Theorem 1 and the lower-bound assertions of Corollaries 7 and 8.

## AI Disclosure

Several technical ideas underlying this work, particularly the detailed design of the routing mechanism and the tuning of its parameters, were developed through extensive discussions and iterative refinement involving the authors and ChatGPT 6 Pro. The authors also used ChatGPT 6 Pro to discuss the organization of the manuscript and to improve its clarity and presentation. The authors take full responsibility for all content of the paper, including the correctness of its mathematical results and proofs.

## References

[Aiy+26] Shashaank Aiyer, Yishay Mansour, Shay Moran, Han Shao, and Tom Waknine. Scale-Sensitive Shattering: Learnability and Evaluability at Optimal Scale. 2026. url: https: //arxiv.org/abs/2605.13684.

[Alo+21] Noga Alon, Omri Ben-Eliezer, Yuval Dagan, Shay Moran, Moni Naor, and Eylon Yogev. “Adversarial laws of large numbers and optimal regret in online classification”. In: Proceedings of the 53rd annual ACM SIGACT symposium on theory of computing. 2021, pp. 447–455.

[Alo+97] Noga Alon, Shai Ben-David, Nicolo Cesa-Bianchi, and David Haussler. “Scale-sensitive dimensions, uniform convergence, and learnability”. In: Journal of the ACM 44.4 (1997), pp. 615–631.

[Asc+25] Ruben Ascoli, Livia Betti, Justin Cheigh, Alex Iosevich, Ryan Jeong, Xuyan Liu, Brian McDonald, Wyatt Milgrim, Steven J. Miller, Francisco Romero Acosta, and Santiago Velazquez Iannuzzelli. “VC-Dimension of Hyperplanes Over Finite Fields”. In: Graphs and Combinatorics 41.2 (2025).

[BDR21] Adam Block, Yuval Dagan, and Alexander Rakhlin. “Majorizing Measures, Sequential Complexities, and Online Learning”. In: Proceedings of the 34th Annual Conference on Learning Theory. 2021, pp. 587–590.

[BKP97] Peter L Bartlett, Sanjeev R Kulkarni, and S Eli Posner. “Covering numbers for realvalued function classes”. In: IEEE transactions on information theory 43.5 (1997), pp. 1721–1724.

[Blu+89] Anselm Blumer, A. Ehrenfeucht, David Haussler, and Manfred K. Warmuth. “Learnability and the Vapnik-Chervonenkis dimension”. In: Journal of the ACM 36.4 (1989), pp. 929–965.

[BLW96] Peter L. Bartlett, Philip M. Long, and Robert C. Williamson. “Fat-Shattering and the Learnability of Real-Valued Functions”. In: Journal of Computer and System Sciences 52.3 (1996), pp. 434–452.

[BM02] Peter L Bartlett and Shahar Mendelson. “Rademacher and gaussian complexities: Risk bounds and structural results”. In: Journal of machine learning research 3 (2002), pp. 463–482.

[BPS09] Shai Ben-David, D´avid P´al, and Shai Shalev-Shwartz. “Agnostic Online Learning”. In: Proceedings of the 22nd Annual Conference on Learning Theory. 2009.

[BY20] Omri Ben-Eliezer and Eylon Yogev. “The Adversarial Robustness of Sampling”. In: Proceedings of the 39th ACM SIGMOD-SIGACT-SIGAI Symposium on Principles of Database Systems. 2020, pp. 49–62.

[CL06] Nicolo Cesa-Bianchi and Gabor Lugosi. Prediction, Learning, and Games. Cambridge University Press, 2006.

[DEG16] Kunal Dutta, Esther Ezra, and Arijit Ghosh. “Two Proofs for Shallow Packings”. In: Discrete & Computational Geometry 56.4 (2016), pp. 910–939. issn: 0179-5376.

[Dud67] Richard M Dudley. “The sizes of compact subsets of Hilbert space and continuity of Gaussian processes”. In: Journal of Functional Analysis 1.3 (1967), pp. 290–330.

[GG15] Pierre Gaillard and S´ebastien Gerchinovitz. “A Chaining Algorithm for Online Nonparametric Regression”. In: Proceedings of The 28th Conference on Learning Theory. Ed. by Peter Gr¨unwald, Elad Hazan, and Satyen Kale. 2015, pp. 764–796.

[Hau95] David Haussler. “Sphere packing numbers for subsets of the Boolean n-cube with bounded Vapnik-Chervonenkis dimension”. In: Journal ofCombinatorial Theory, Series A 69.2 (1995), pp. 217–232.

[KS94] Michael J. Kearns and Robert E. Schapire. “Eficient distribution-free learning of probabilistic concepts”. In: Journal of Computer and System Sciences 48.3 (1994), pp. 464– 497. issn: 0022-0000.

[Lit88] Nick Littlestone. “Learning Quickly When Irrelevant Attributes Abound: A New Linear-Threshold Algorithm”. In: Machine Learning 2.4 (1988), pp. 285–318.

[MM24] Maryanthe Malliaris and Shay Moran. “Model theory and agnostic online learning via excellent sets”. In: Transactions of the American Mathematical Society 377.11 (2024), pp. 7753–7776.

[MS14] Maryanthe Malliaris and Saharon Shelah. “Regularity lemmas for stable graphs”. In: Transactions of the American Mathematical Society 366.3 (2014), pp. 1551–1585.

[MV03] Shahar Mendelson and Roman Vershynin. “Entropy and the combinatorial dimension”. In: Inventiones mathematicae 152.1 (2003), pp. 37–55.

[RS14] Alexander Rakhlin and Karthik Sridharan. “Online Non-Parametric Regression”. In: The 27th Conference on Learning Theory. 2014, pp. 1232–1264.

[RST10] Alexander Rakhlin, Karthik Sridharan, and Ambuj Tewari. “Online Learning: Random Averages, Combinatorial Parameters, and Learnability”. In: Advances in Neural Information Processing Systems. 2010, pp. 1984–1992.

[RST11] Alexander Rakhlin, Karthik Sridharan, and Ambuj Tewari. “Online Learning: Beyond Regret”. In: Proceedings of the 24th Annual Conference on Learning Theory. Vol. 19. 2011, pp. 559–594.

[RST15a] Alexander Rakhlin, Karthik Sridharan, and Ambuj Tewari. “Online Learning via Sequential Complexities”. In: Journal of Machine Learning Research 16.6 (2015), pp. 155– 186.

[RST15b] Alexander Rakhlin, Karthik Sridharan, and Ambuj Tewari. “Sequential complexities and uniform martingale laws of large numbers”. In: Probability theory and related fields 161.1 (2015), pp. 111–153.

[RST17] Alexander Rakhlin, Karthik Sridharan, and Alexandre B. Tsybakov. “Empirical entropy, minimax regret and minimax risk”. In: Bernoulli 23.2 (2017), pp. 789–824.

[RST24] Vinod Raman, Unique Subedi, and Ambuj Tewari. “Online Learning with Set-valued Feedback”. In: Proceedings of 37th Conference on Learning Theory. Ed. by Shipra Agrawal and Aaron Roth. 2024, pp. 4381–4412.

[Sau72] Norbert Sauer. “On the density of families of sets”. In: Journal of Combinatorial Theory, Series A 13.1 (1972), pp. 145–147.

[Sha+26] Shihao Shao, Cong Fang, Zhouchen Lin, and Dacheng Tao. Partial Feedback Online Learning. 2026. url: https://arxiv.org/abs/2601.21462.

[She72] Saharon Shelah. “A combinatorial problem; stability and order for models and theories in infinitary languages”. In: Pacific Journal of Mathematics 41.1 (1972), pp. 247–261.

[Tal05] Michel Talagrand. The generic chaining: upper and lower bounds of stochastic processes. Springer, 2005.

[VC71] V. N. Vapnik and A. Ya. Chervonenkis. “On the Uniform Convergence of Relative Frequencies of Events to Their Probabilities”. In: Theory of Probability & Its Applications 16.2 (1971), pp. 264–280.

[VW96] A. van der Vaart and J.A. Wellner. Weak Convergence and Empirical Processes: With Applications to Statistics. Springer, 1996.