# Characterizing Language Generation in the Limit: Finite Witnesses and a Separation-Width Hierarchy

Xiaoyu Li<sup>1</sup> Andi Han<sup>2</sup> Jiaojiao Jiang<sup>1</sup> Junbin Gao<sup>2</sup>

<sup>1</sup>University of New South Wales {xiaoyu.li2,jiaojiao.jiang}@unsw.edu.au <sup>2</sup>University of Sydney {andi.han,junbin.gao}@sydney.edu.au

September 10, 2026

## Abstract

Language generation in the limit asks for valid unseen elements from every exhaustive positive presentation of an unknown infinite language. We characterize this task for arbitrary families over a countable universe. Generation is possible exactly when each target can be assigned a finite positive witness so that the targets activated by any finite sample have an infinite common intersection. The necessary direction follows from a universal normalization: a search through unconfirmed histories converts any successful generator into one depending only on the observed set. We then ask how large compatible witnesses must be. Positive separation width records the smallest uniform size bound, with two further levels for unbounded finite witnesses and the absence of any compatible finite-witness assignment. Every level occurs. Countable families admit singleton witnesses, explicit families realize every finite width, and a union of two families with infinite common cores requires unbounded finite witnesses. Finally, countable-support and finiteprofile obstructions explain why local combinatorial data cannot determine generation in the limit. The characterization and full width hierarchy are checked in Lean, including the simplified normalization and a direct diagonal capture lemma. The accompanying Lean development is maintained at https://github.com/xiaoyulics/language-generation-characterization.

## 1 Introduction

A generator observes positive examples from an unknown language and must produce another valid example. In generation in the limit, the observations arrive in an arbitrary order and eventually enumerate the entire infinite target. Success means that, after some finite time, every output belongs to the target and has not appeared among the inputs. There is no feedback about whether an output is correct.

Kleinberg and Mullainathan (2024) showed that every countable family of infinite languages admits such a generator, even when identifying the target is impossible. For arbitrary families over a countable universe, Raman et al. (2025) characterized uniform generation by finite closure dimension and non-uniform generation by increasing countable covers of finite closure dimension. For ordinary generation, whose convergence time may depend on the entire presentation, they gave suficient conditions and asked for a complete characterization.

Our first result answers this question by identifying the finite positive evidence that makes generation possible. Our second result studies the size of that evidence and shows why pointwise finiteness cannot be replaced by any uniform finite bound.

Finite witnesses: when generation is possible. Assign each target � a finite set $T ( L ) \subseteq L$ . At a finite sample $S ,$ call � active if $T ( L ) \subseteq S \subseteq L$ : the sample is consistent with � and has already revealed its witness. Theorem 2.3 states that generation in the limit is possible exactly when one assignment makes the full intersection of every nonempty active family infinite.

The suficient direction explains the condition. Every target eventually becomes active on each of its texts. From then on, any fresh point common to all active targets is a valid output. Several targets may remain indistinguishable indefinitely; generation needs only a common source of new elements.

The necessary direction must handle a generator’s dependence on order and repetitions. Theorem 3.1 gives a universal transformation from any sequence-input generator � to a set-input generator �, preserving every infinite target on which � succeeds. The new generator searches histories whose outputs have not yet been observed. Finitely many positive confirmations force this search to follow a canonical finite chain of genuine errors and then pass beyond it. This produces witnesses satisfying

$$
T ( L ) \subseteq S \subseteq L , \qquad S { \mathrm { ~ f i n i t e ~ } } \quad \Longrightarrow \quad g ( S ) \in L \setminus S .
$$

The construction uses no knowledge of the target and no correctness oracle.

Witness size: a complete separation-width hierarchy. Call a nonempty subfamily bad when its full common intersection is finite. The witness condition is equivalent to requiring that each bad subfamily contain $L , K$ with $T ( L ) \not \subseteq K$ . We call this positive separation.

Positive separation width records the least uniform finite bound on the sizes of separating witnesses, when one exists. Otherwise it is � if a pointwise finite assignment exists, and $\omega + 1$ if none exists. Here � is the first infinite ordinal and $\omega + 1$ its successor; Section 5 explains the scale. The core characterization has the immediate numerical form

$$
\mathcal { H } \mathrm { \ g e n e r a t a b l e \ i n t h e \ l i m i t } \quad \Longleftrightarrow \quad \mathfrak { s } ( \mathcal { H } ) \leq \omega .
$$

The additional content is the hierarchy in Theorem 6.1. Every countable family admits distinct singleton witnesses. Nevertheless, explicit families have arbitrarily large finite width, with matching upper and lower bounds. A family of languages containing either of two fixed disjoint infinite blocks has width �. Thus even a union of two classes with infinite common cores can require unbounded finite witnesses. These are witness-size statements; an arbitrary text may delay a designated witness for as long as it wishes.

The finite-level lower bounds must allow witnesses to depend on entire targets, including their infinite tails. A direct diagonal capture lemma handles this dependence before a finite incidence count finishes the proof. The same lemma yields forced witness growth along an explicit chain of targets in the width-� example.

Why local dimensions miss the distinction. The compatibility of witnesses is global. Every countable family is generatable, but the family of all infinite subsets is not. Consequently, a dimension whose infinity is always supported by a countable subclass cannot characterize generation in the limit by finiteness. Even the complete finite trace and positive-closure profiles fail to distinguish generatable cofinite targets from the class of all infinite targets. Section 7 makes these limitations precise without excluding invariants that retain additional global structure.

## 1.1 Relation to earlier work

The positive-data setting originates in language identification in the limit (Gold, 1967). Finite tell-tales are central to Angluin (1980)’s characterization for indexed recursive-language families. Our witnesses have a diferent compatibility requirement: several active targets may remain indistinguishable if their intersection supplies infinitely many outputs. A witness need not identify a target or exclude all its proper sublanguages. For a broader account of language generation and its variants, see Mehrotra (2026, Part II).

Set-drivenness and locking normal forms have also been studied in language identification (Kötzing et al., 2017). Our normalization concerns the single-element generation criterion and uses the test of whether a simulated output has appeared among the observations. Sorting the inputs before querying an arbitrary successful generator does not sufice; Section A gives a counterexample.

The two-core class is a union of two classes with infinite fixed common subsets, so its generatability also follows from the finite-union suficient condition of Raman et al. (2025). Its role here is the necessity of unbounded finite witnesses. Our characterization also recovers the increasing-EUC-cover suficient condition by a direct assignment (Appendix C). Arbitrary unions require care: Hanneke et al. (2025) and Bai et al. (2026) give a uniformly generatable class and a non-uniformly generatable class whose union is not generatable. Assignments that work separately need not stay valid when targets from both classes become active at one sample.

Other characterizations place requirements on the range of generated outputs. Kalavasis et al. (2026) characterize generation with several notions of breadth, while Kleinberg and Wei (2025) quantify coverage using density. These questions concern how much of a target is generated. Separation width instead measures the size of positive evidence compatible across targets; the infinite-intersection condition guarantees a supply of fresh elements without prescribing their density.

The observation model also matters. Kleinberg and Wei (2026) study partial enumeration, where only an infinite subset of the target is revealed. Li et al. (2026a) study unordered pairs with opposite labels and characterize uniform contrastive generation by a contrastive closure dimension. In the nested model of Li et al. (2026b), a verifier recognizes an ambient formal language while the examples reveal part of an unknown valuable sublanguage; validity and valuable coverage then become separate requirements. Our finite witnesses use exhaustive positive observations of individual target elements. Applying the same idea to these other observation models would require a corresponding notion of witness compatibility.

We define the model and state the characterization in Section 2, prove normalization and the characterization in Sections 3 and 4, and develop separation width and its hierarchy in Sections 5 and 6. Sections 7 and 8 explain the structural and computational boundaries. The appendices give the sorting counterexample, stronger witness divergence, the recovery of the increasing-EUC-cover condition, the padding obstruction, and formalization details.

## 2 The model and the characterization

Let � be a countably infinite set. Write $[ X ] ^ { < \omega }$ for its finite subsets, including the empty set, and $[ X ] ^ { \omega }$ for its infinite subsets. A language family is any $\mathcal { H } \subseteq [ X ] ^ { \omega }$ . No efectiveness, cardinality, or descriptive-set-theoretic assumption is imposed on ℋ. The set of finite words over � is $\chi ^ { < \omega }$ , its empty word is $\varepsilon ,$ and $\mathsf { c } ( \sigma )$ is the content of a word �. We write $\sigma \prec \tau$ for strict prefix and |�| for word length. Our conventions are $\mathbb { N } = \{ 1 , 2 , \ldots \}$ and $ { \mathbb { N } } _ { 0 } =  { \mathbb { N } } \cup \lbrace 0 \rbrace$

A text for � is an infinite sequence $\boldsymbol { x } = \left( \boldsymbol { x } _ { t } \right) _ { t \geq 1 }$ with $\{ x _ { t } : t \geq 1 \} = L$ . Repetitions and arbitrarily long delays are permitted. Let $\boldsymbol { x } _ { 1 : t } = \left( x _ { 1 } , \ldots , x _ { t } \right)$ and $S _ { t } = \mathsf { c } ( x _ { 1 : t } )$

Definition 2.1 (Generation in the limit). A total function $G : X ^ { < \omega }  X$ generates � in the limit if for every text � for � there is a time $t _ { 0 } = t _ { 0 } ( L , x )$ such that

$$
G ( x _ { 1 : t } ) \in L \ \backslash \ S _ { t } \qquad { \mathrm { f o r ~ e v e r y ~ } } t \geq t _ { 0 } .\tag{2.1}
$$

The family ℋ is generatable in the limit if one total � has this property for every $L \in { \mathcal { H } }$ . A generator is set-driven if it has the form $G ( \sigma ) = g ( \operatorname { c } ( \sigma ) )$ for some $g : [ X ] ^ { < \omega }  X$

The output is a single element. Freshness refers to the observed inputs, not to earlier outputs. The generator has no target-membership oracle and receives no correctness feedback. We work with arbitrary total functions; computability is discussed separately in Corollary 3.2. Requiring freshness on every finite input, rather than only eventually on each text, gives the same existence notion by the repair in Equation (3.2).

Definition 2.2 (Finite-witness condition). A witness assignment for ℋ is a function $T : \mathcal { H }  [ X ] ^ { < \omega }$ with $T ( L ) \subseteq L$ for every $L \in { \mathcal { H } }$ . At a finite set $S \subseteq X$ , define

$$
{ \mathcal { A } } _ { T } ( S ) = \{ L \in { \mathcal { H } } : T ( L ) \subseteq S \subseteq L \} .\tag{2.2}
$$

The assignment satisfies the finite-witness condition if

$$
\mathcal { A } _ { T } ( S ) \neq \emptyset \quad \Longrightarrow \quad \left| \bigcap _ { L \in \mathcal { A } _ { T } ( S ) } L \right| = \infty \qquad \mathrm { f o r ~ e v e r y ~ } S \in [ X ] ^ { < \omega } .\tag{2.3}
$$

Theorem 2.3 (Finite-witness characterization). For every countably infinite � and every $\mathcal { H } \subseteq [ X ] ^ { \omega }$ , the following are equivalent:

(i) ℋ is generatable in the limit.

(ii) ℋ is generatable in the limit by a set-driven generator.

(iii) ℋ admits a witness assignment satisfying (2.3).

The intersection in (2.3) is over all active languages simultaneously. Pairwise infinite intersections do not express the stated condition. The empty active family imposes no requirement, so no convention for its intersection is needed.

A finite witness concerns which examples have appeared. Even an arbitrarily large finite subset of a target may omit a specified witness element. Thus the theorem does not replace convergence in the limit by a target-dependent bound on the number of distinct examples. Nor does the existence of an assignment supply a procedure to compute it or to test the intersection condition.

Algorithm 1 Set-driven normalization by unconfirmed extensions   
Require: $S \in [ X ] ^ { < \omega } ;$ � from (3.2); fixed enumerations   
1: � ← �   
2: for $k = 1 , \dots , | S |$ do   
3: Choose the least-code $\tau \in S ^ { < \omega }$ satisfying   
$\sigma < \tau , E _ { k } ( S ) \subseteq \mathbf { c } ( \tau ) ,$ , and �(�) ∉ �   
� ← �   
5: end for   
6: return �(�)

## 3 Set-driven normalization

The obstacle is that a generator may use the order and repetitions of its input. We remove this dependence by simulating histories over the observed set. An output outside that set is unconfirmed: it may be an error, or a valid point that has not yet appeared. Positive observations eventually eliminate the latter possibility for any fixed finite collection of histories.

Theorem 3.1 (Universal normalization). For every total $G : X ^ { < \omega }  X .$ , there is a total $g : [ X ] ^ { < \omega }  X$ such that every infinite language � generated by � has a finite $T _ { L } \subseteq L$ satisfying

$$
T _ { L } \subseteq S \subseteq L , \qquad S f i n i t e \quad \Longrightarrow \quad g ( S ) \in L \setminus S .\tag{3.1}
$$

The same � worksfor all such � and uses onlyfinitely many evaluations of � on each input �.

## 3.1 The construction

Fix an enumeration of $\chi ,$ and let $E _ { k } ( A )$ be the first min $\{ k , | A | \}$ points of � in this order, with $E _ { k } ( A )$ containing exactly � points when � is infinite. Fix also an injective coding code : $X ^ { < \omega }  \mathbb { N } _ { 0 }$ . Each word then has only finitely many predecessors in code order.

Repair repeated outputs by setting

$$
F ( \sigma ) = \left\{ { \cal G } ( \sigma ) , \begin{array} { l l } { G ( \sigma ) \notin \mathrm { c } ( \sigma ) , } \\ { \operatorname* { m i n } ( X \setminus \mathrm { c } ( \sigma ) ) , } & { G ( \sigma ) \in \mathrm { c } ( \sigma ) . } \end{array} \right.\tag{3.2}
$$

Thus � is fresh on every history and eventually agrees with � on each text where � succeeds. Any fixed fresh fallback can replace the minimum.

Given �, start at the empty history. At round �, take the least-code strict extension over � that contains $E _ { k } ( S )$ and whose output remains unconfirmed by �. Perform exactly |�| rounds.

Every round has a candidate: append a listing of all of � to the current history. Its content is exactly �, so its �-output lies outside �. The listing is nonempty whenever a round is performed. Searching by code therefore terminates after finitely many evaluations. The output is fresh against the whole sample; for $S = \emptyset$ , the loop is empty and returns $F ( \varepsilon )$

## 3.2 Why finite confirmations sufice

Proof of Theorem 3.1. Fix an infinite target � generated by $G ,$ hence also by �. For the proof only, construct the genuine-error chain: $\sigma _ { 0 } = \varepsilon ,$ , and at stage $j \geq 1$ choose the least-code word satisfying

$$
\sigma _ { j - 1 } \prec \sigma _ { j } , \qquad \mathsf { c } ( \sigma _ { j } ) \subseteq L , \qquad E _ { j } ( L ) \subseteq \mathsf { c } ( \sigma _ { j } ) , \qquad F ( \sigma _ { j } ) \notin L .\tag{3.3}
$$

Stop when no such word exists. An infinite chain would give an exhaustive text for � with errors at the strictly increasing times $| \sigma _ { j } | ,$ , contradicting success. Let � be its length. The stopping condition says

$$
\sigma _ { m } \prec \tau , \quad \mathsf { c } ( \tau ) \subseteq L , \quad E _ { m + 1 } ( L ) \subseteq \mathsf { c } ( \tau ) \quad \Longrightarrow \quad F ( \tau ) \in L .\tag{3.4}
$$

Choose the finite positive witness

$$
T _ { L } = E _ { m + 1 } ( L ) \cup \mathsf { c } ( \sigma _ { m } ) \cup \bigcup _ { j = 1 } ^ { m } \{ F ( \tau ) : \mathsf { c o d e } ( \tau ) < \mathsf { c o d e } ( \sigma _ { j } ) , F ( \tau ) \in L \} .\tag{3.5}
$$

Each code has finitely many predecessors, so this union is finite. Its first component already ensures $| T _ { L } | \ge m + 1$

Fix any finite $T _ { L } \subseteq S \subseteq L$ . Then $E _ { j } ( S ) = E _ { j } ( L )$ for $j \leq m + 1$ . We claim that the first � rounds of Algorithm 1 select exactly $\sigma _ { 1 } , \ldots , \sigma _ { m }$ . Inductively, $\sigma _ { j }$ is an eligible extension of $\sigma _ { j - 1 }   .$ its content lies in $\mathsf { c } ( \sigma _ { m } ) \subseteq S$ , its checkpoint agrees, and its output lies outside �, hence outside �. Any earlier-code candidate � with $F ( \tau ) \in L$ has its output included in $T _ { L }$ by (3.5), contradicting $F ( \tau ) \not \in S$ . Any earlier candidate with $F ( \tau ) \notin L$ would contradict the least-code choice in (3.3). This proves the claim for every such �.

Since $| S | \geq m + 1$ , the algorithm performs another round. Its final history $\rho$ therefore strictly extends $\sigma _ { m , \mathbf { \ell } }$ , lies over $L ,$ and contains $E _ { m + 1 } ( L )$ . Equation (3.4) gives $F ( \rho ) \in L ,$ while the final selection gives $F ( \rho ) \not \in S$ . This proves (3.1), including $m = 0$ . The algorithm itself used only $G$ and $S ;$ the target and its genuine errors entered only the proof. □

Corollary 3.2 (Computability preservation). With an efective coding of �, a total computable � has a total computable normalization � as in Theorem 3.1.

Proof. Choose efective point and word enumerations. Each candidate test is decidable from � and a call to $G ,$ and each code search terminates by the append-all argument. There are only |�| rounds. □

The proof isolates the mechanism: finitely many positive confirmations force the simulation to follow all genuine errors and then pass beyond them. Later simulated histories need not stabilize.

## 4 From normalization to the characterization

Normalization supplies witnesses with a uniform guarantee over all finite extensions inside a target.   
This directly gives the necessary compatibility between witnesses of diferent targets.

Proof of Theorem 2.3. For $( \mathrm { i } ) { \Longrightarrow } ( \mathrm { i i i } )$ , apply Theorem 3.1 to a generator for $\mathcal { H } ,$ obtaining one $g$ and witnesses $T ( L )$ satisfying (3.1). Fix a finite $S$ with $\mathscr { A } _ { T } ( S ) \neq \emptyset ,$ , and put

$$
C = \bigcap _ { L \in { \mathcal { A } } _ { T } ( S ) } L .
$$

If � were finite, every active � would satisfy $T ( L ) \subseteq S \subseteq C \subseteq L . \operatorname { A p p l y i n g }$ (3.1) at the same input � gives $g ( C ) \in L \setminus C$ for all these $L ,$ forcing $g ( C ) \in C \setminus C$ . Thus � is infinite. This argument uses the full active family, regardless of its cardinality.

For $( \mathrm { i i i } ) { \Longrightarrow } ( \mathrm { i i } )$ , define

$$
g ( S ) = \left\{ \operatorname* { m i n } \left( \left( \bigcap _ { L \in { \mathcal { A } } _ { T } ( S ) } L \right) \backslash S \right) , \quad { \mathcal { A } } _ { T } ( S ) \neq \emptyset , \right.\tag{4.1}
$$

The assumed infinitude makes this a total function. On any text for $L ,$ its finite witness $T ( L )$ eventually appears. Thereafter � remains active, so every output lies in $L \backslash S$ . Finally, $( \mathrm { i i } ) { \Longrightarrow } ( \mathrm { i } )$ is immediate by setting $G ( \sigma ) = g ( { \mathsf { c } } ( \sigma ) )$ □

The same-input argument explains why the common intersection must be infinite: a finite common core would itself be a legal input beyond all active witnesses, with no common fresh output left.

## 5 Positive separation and its width

The active-family condition has a generator-free reformulation in terms of separating languages inside subfamilies with small common cores. This reformulation also lets us measure the cardinality of the positive witnesses. For nonempty ${ \mathcal { F } } \subseteq { \mathcal { H } } ,$ write $\begin{array} { r } { \mathrm { c o r e } ( \mathcal { F } ) = \bigcap _ { L \in \mathcal { F } } L } \end{array}$ and call $\mathcal { F }$ bad if this intersection is finite.

Definition 5.1 (Positive separation). A positive assignment $P ,$ with $P ( L ) \subseteq L$ for every $L \in { \mathcal { H } }$ separates ℋ if

$$
\forall \mathrm { n o n e m p t y b a d } \mathcal { F } \subseteq \mathcal { H } , \quad \exists L , K \in \mathcal { F } : P ( L ) \not \subseteq K .\tag{5.1}
$$

The values of � are allowed to be infinite unless stated otherwise.

Thus a bad subfamily must contain a witness point, assigned to one of its members, that another member omits. One assignment must meet all these constraints simultaneously.

Proposition 5.2 (Separation equivalence). A finite positive assignment � satisfies (2.3) if and only if it separates $\mathcal { H } .$ For any fixed positive assignment, it is equivalent in (5.1) to test only countable bad subfamilies.

Proof. If separation fails for a nonempty bad $\mathcal { F } _ { \mathbf { \lambda } }$ , then $T ( L ) \subseteq K$ for every $L , K \in { \mathcal { F } }$ . Consequently $T ( L ) \subseteq \mathrm { c o r e } ( \mathcal { F } )$ for every $L \in { \mathcal { F } }$ . At the finite sample $S = \mathrm { c o r e } ( { \mathcal { F } } )$ all these languages are active. Their full active intersection is contained in $S ,$ violating (2.3).

Conversely, a nonempty active family with finite intersection is itself a bad subfamily. Every one of its witnesses lies in $S ,$ and every one of its languages contains �. Hence it is not separated.

For the countable reduction, fix a nonempty $\mathcal { F }$ and one member $L _ { * } \in \mathcal { F }$ . For each � ∉ core $( { \mathcal { F } } )$ choose $K _ { x } \in \mathcal { F }$ omitting �. The at-most-countable family $\{ L _ { * } \} \cup \{ K _ { x } : x \notin \mathrm { c o r e } ( \mathcal { F } ) \}$ has exactly the same intersection. If the original family is unseparated by a fixed assignment, so is this subfamily. □

Countable reduction does not exchange the quantifiers $\exists T \forall { \mathcal { F } }$ and $\forall \mathcal { F } \exists T _ { \mathcal { F } }$ . Assignments valid on separate countable subfamilies need not fit together on the whole class.

Theorem 5.3 (Singleton witnesses for countable families). Every countable family of infinite languages admits a separating singleton assignment $T ( L ) = \{ p _ { L } \}$ with all points $p _ { L }$ distinct.

Proof. List the distinct targets as $L _ { 1 } , L _ { 2 } , . . . ,$ using a finite list if needed. At step $i ,$ take the union $B _ { i }$ of the finite intersections of all nonempty subfamilies of $\left\{ L _ { 1 } , \ldots , L _ { i } \right\}$ . There are finitely many such subfamilies, so $B _ { i }$ is finite. Choose

$$
p _ { i } \in L _ { i } \setminus \left( B _ { i } \cup \{ p _ { 1 } , . . . , p _ { i - 1 } \} \right) .
$$

If a bad subfamily is finite, let � be its largest index. Its core is contained in $B _ { i }$ and therefore omits $p _ { i } ;$ some member of the subfamily omits this assigned point. If a bad subfamily is infinite, its infinitely many distinct assigned points cannot all lie in its finite core. It too is separated. The empty class is vacuous. □

This strengthens the witness construction for the countable-family theorem of Kleinberg and Mullainathan (2024). It concerns the size of a certificate, not the time needed to observe that certificate: a text may postpone its single designated point arbitrarily long.

To measure witness size, we use the ordered scale

$$
0 < 1 < 2 < \cdots < \omega < \omega + 1 .
$$

Here $\omega$ is the first infinite ordinal, the order type of $0 < 1 < 2 < \cdots ;$ its successor $\omega + 1$ appends a greatest element after that sequence. Both have countably infinite underlying sets. The distinction concerns their order types. In particular, $\operatorname* { s u p } _ { n \in \mathbb { N } _ { 0 } } n = \omega$

Definition 5.4 (Positive separation width). The positive separation width ${ \mathfrak { s } } ( { \mathcal { H } } )$ is defined by three cases:

(i) If a separating assignment has all its witness sizes bounded by one finite integer, the width is the least such bound.

(ii) If a pointwise finite separating assignment exists, but no separating assignment has a uniform finite bound, the width is �.

(iii) If no pointwise finite separating assignment exists, the width is $\omega + 1$

Thus width $\omega$ allows each target a finite witness while requiring unbounded sizes across targets. Width $\omega + 1$ means that every separating assignment has at least one infinite value. The empty class has width zero.

Corollary 5.5 (Generation threshold). For every $\mathcal { H } \subseteq [ X ] ^ { \omega }$

$$
\mathcal { H } \ i s \ g e n e r a t a b l e \ i n \ t h e \ l i m i t \quad \Longleftrightarrow \quad \mathfrak { s } ( \mathcal { H } ) \leq \omega .\tag{5.2}
$$

Proof. By definition, width at most � means that a finite separating assignment exists. Apply Proposition 5.2 and Theorem 2.3. □

At width $\omega ,$ each target’s finite witness still appears by some finite time on every text for that target. Unbounded sizes across targets are therefore compatible with convergence on each text.

Proposition 5.6 (Basic properties). The width is invariant under bĳective relabeling of � and monotone under passing to subfamilies. It is zero exactly when the full common intersection is infinite, with $\mathrm { c o r e } ( \emptyset ) = \mathcal { X }$

Proof. Restriction preserves separation and witness-size bounds; bĳections preserve inclusions and cardinalities. Width zero means the empty assignment separates. This holds exactly when there is no nonempty bad subfamily, equivalently when the full common intersection is infinite. □

Remark 5.7 (Equivalent optimization formula). Set $c ( W ) = | W |$ for finite $W _ { \ell }$ , and $c ( W ) = \omega + 1$ for infinite �. Then

$$
\mathfrak { s } ( \mathcal { H } ) = \operatorname* { m i n } _ { P \ : \mathrm { s e p a r a t e s } \mathcal { H } } \ : \mathfrak { s u p } _ { L \in \mathcal { H } } \ : c ( P ( L ) ) ,\tag{5.3}
$$

with empty supremum zero. A feasible assignment always exists: $P ( L ) = L$ separates every bad subfamily. Each assignment has cost in $\mathbb { N } _ { 0 } \cup \{ \omega , \omega + 1 \}$ , so the minimum is attained, and its three possible regimes are exactly those in Definition 5.4. The charge $\omega + 1$ is a cost convention, not the cardinality of an infinite witness. Charging $\omega$ would merge the cost of an infinite witness with the supremum of unbounded finite witness sizes.

## 6 The complete separation-width hierarchy

The characterization guarantees finite witnesses for individual targets. Their required sizes have a richer structure: every finite level occurs, and some generatable classes require unbounded finite witnesses.

Theorem 6.1 (Separation-width hierarchy). On every countably infinite universe $\chi _ { \mathrm { i } }$

(i) Every countable language family has width at most one.

(ii) For every � $\in \mathbb { N } _ { 0 } \cup \{ \omega \}$ , some generatable family has width exactly �.

(iii) The family $[ X ] ^ { \omega }$ of all infinite subsets has width $\omega + 1$

Thus every value of the width occurs.

The finite levels come from an incidence-counting construction. A class containing these examples for every finite size then realizes �. The only auxiliary selection fact needed for the lower bounds is the following diagonal lemma.

## 6.1 A bounded-capture lemma

The lower bound must handle assignments that depend arbitrarily on the entire target language.   
The following lemma supplies that step.

Lemma 6.2 (Bounded capture with core avoidance). Let $( U _ { n } ) _ { n \geq 1 }$ be finite subsets of a set � with $\mathsf { s u p } _ { n } \left| U _ { n } \right| < \infty$ , and let � be an at-most-countable family of infinite subsets of �. There is $D \subseteq Y$ such that $U _ { n } \subseteq D$ for infinitely many �, but $C \nsubseteq D$ for every $C \in C$

Proof. Choose � with $| U _ { n } | \leq d$ . If � is empty, take $D = \textstyle \bigcup _ { n } U _ { n }$ . Otherwise list its members as $C _ { 1 } , C _ { 2 } , . . . ,$ repeating a finite list if necessary.

Maintain an infinite set $I _ { j - 1 }$ of available indices, starting with $I _ { 0 } = \mathbb { N }$ . At step $j ,$ choose $d + 1$ points of $C _ { j } \setminus \bigcup _ { i < j } U _ { n _ { i } } ;$ this is possible because the removed union is finite. Each $U _ { n }$ contains at most $d$ of these points. By the infinite pigeonhole principle, some chosen point $x _ { j }$ is omitted by infinitely many $U _ { n }$ with $n \in I _ { j - 1 }$ . Let

$$
I _ { j } = \{ n \in I _ { j - 1 } : x _ { j } \notin U _ { n } \} ,
$$

and choose $n _ { j } \in I _ { j }$ larger than every previously selected index.

Set $D = \cup _ { j } U _ { n _ { j } }$ . The indices $n _ { j }$ are distinct, so � captures infinitely many terms. For each $j ,$ the choice of $x _ { j }$ excludes it from every earlier selected set; the nested pools exclude it from the current and every later selected set. Hence $x _ { j } \in C _ { j } \setminus D$ for every �. □

## 6.2 Every finite level

Fix $k \geq 1$ . Take disjoint sets of anchors $A _ { 0 } = \{ a _ { 0 } , \ldots , a _ { k - 1 } \}$ and $B _ { 0 } = \{ b _ { 0 } , \ldots , b _ { k - 1 } \}$ and disjoint countably infinite tails $A _ { * } , B _ { * } ,$ all four sets pairwise disjoint. Put $A = A _ { 0 } \cup A _ { * } , B = B _ { 0 } \cup B ,$ ∗ and $\chi = A \sqcup B$ . Define the anchored class

$$
L _ { i } ( D ) = A \cup ( B _ { 0 } \setminus \{ b _ { i } \} ) \cup D ,
$$

$$
i < k , \quad D \subseteq B _ { * } ,\tag{6.1}
$$

$$
K _ { j } ( E ) = B \cup ( A _ { 0 } \setminus \{ a _ { j } \} ) \cup E ,
$$

$$
j < k , \quad E \subseteq A _ { * } ,\tag{6.2}
$$

$$
\mathscr { H } ^ { ( k ) } = \{ L _ { i } ( D ) : i < k , D \subseteq B _ { * } \} \cup \{ K _ { j } ( E ) : j < k , E \subseteq A _ { * } \} .
$$

Each side has an infinite common block. Opposite-side types omit diferent specified anchors, while their tails vary without restriction.

Proposition 6.3 (Exact finite levels). For every $k \geq 1$

$$
{ \mathfrak { s } } ( { \mathcal { H } } ^ { ( k ) } ) = \left\lceil { \frac { k } { 2 } } \right\rceil .
$$

Consequently every positive integer is attained by a class generatable in the limit.

Proof. Upper bound. Put $d = \lceil k / 2 \rceil$ . For each row $i ,$ let $J _ { i }$ be the � cyclic columns $i , i + 1 , \dots , i + d - 1$ modulo �. Set

$$
T ( L _ { i } ( D ) ) = \{ a _ { j } : j \in J _ { i } \} , \qquad T ( K _ { j } ( E ) ) = \{ b _ { i } : j \notin J _ { i } \} .
$$

These positive witnesses have sizes � and $k - d \leq d ,$ , respectively, because each column occurs in exactly � of the sets $J _ { i } .$ . Every pair $( i , j )$ is separated: if $j \in J _ { i . }$ , the first witness contains $a _ { j }$ omitted by $K _ { j } ( E ) _ { \ j }$ ; otherwise the second contains $b _ { i }$ omitted by $L _ { i } ( D )$ . Hence no opposite-side pair can be simultaneously active. A nonempty active family shares all � or all $B ,$ so its intersection is infinite.

Lower bound. Fix $q \in  { \mathbb { N } } _ { 0 }$ and an arbitrary valid assignment � with $| T ( L ) | \leq q$ for every target. Enumerate $A ,$ ∗ and let $E _ { n }$ be its first � points. For $j < k ,$ , put $K _ { j , n } = K _ { j } ( E _ { n } )$ . Pass to increasing indices � on which all � anchor sets

$$
R _ { j } = T ( K _ { j , n } ) \cap B _ { 0 }
$$

are fixed simultaneously. This is possible because their joint pattern has only finitely many values. On those indices set

$$
U _ { n } = \bigcup _ { j < k } \bigl ( T ( K _ { j , n } ) \cap B _ { * } \bigr ) , \qquad | U _ { n } | \leq k q .
$$

For every $i < k$ and finite sample $S ,$ form

$$
\begin{array} { r l } & { { \mathscr F } _ { i } ( S ) = \{ L _ { i } ( D ^ { \prime } ) : T ( L _ { i } ( D ^ { \prime } ) ) \subseteq S \subseteq L _ { i } ( D ^ { \prime } ) \} , } \\ & { \quad C _ { i , S } = \displaystyle \bigcap \{ D ^ { \prime } \subseteq B _ { * } : L _ { i } ( D ^ { \prime } ) \in { \mathscr F } _ { i } ( S ) \} , } \end{array}
$$

with empty intersection $B _ { * }$ . There are only countably many pairs $( i , S )$ . Apply Lemma $6 . 2$ to the infinite members of this collection of cores and to $( U _ { n } )$ . Obtain $D \subseteq B ,$ ∗ capturing infinitely many $U _ { n }$ and containing none of those infinite cores.

Fix the � targets $L _ { i } ( D )$ . We claim that every pair $( i , j )$ obeys

$$
a _ { j } \in T ( L _ { i } ( D ) ) \quad { \mathrm { o r } } \quad b _ { i } \in R _ { j } .\tag{6.3}
$$

If both fail, choose a captured � large enough that $T ( L _ { i } ( D ) ) \cap A _ { * } \subseteq E _ { n }$ . The target $K _ { j , n }$ contains all � and every �-anchor except $a _ { j } ,$ so $T ( L _ { i } ( D ) ) \subseteq K _ { j , n }$ . Conversely $L _ { i } ( D )$ contains all $A ;$ the �-anchor part $R _ { j }$ omits $b _ { i } ,$ and the �-tail part of $T ( K _ { j , n } )$ lies in $U _ { n } \subseteq D$ . Thus $T ( K _ { j , n } ) \subseteq L _ { i } ( D )$

Both targets are active at $S = T ( L _ { i } ( D ) ) \cup T ( K _ { j , n } )$ . The full active intersection � is infinite. Since $K _ { j , n }$ has only finitely many $A { \mathrm { - } } \mathsf { p o i n t s }$ and $B _ { 0 }$ is finite, $J \cap B _ { * }$ is infinite. Every target in ${ \mathcal { F } } _ { i } ( S )$ contains $J ,$ so $J \cap B _ { * } \subseteq C _ { i , S }$ . This core is infinite, yet $C _ { i , S } \subseteq D$ because $L _ { i } ( D )$ itself is active. This contradicts the choice of � and proves (6.3).

The $k ^ { 2 }$ pairs are therefore covered by at most

$$
\sum _ { i < k } | T ( L _ { i } ( D ) ) \cap A _ { 0 } | + \sum _ { j < k } | R _ { j } | \leq 2 k q
$$

anchor incidences. It follows that $q \geq \lceil k / 2 \rceil$ . The counting step applies to arbitrary tail-dependent assignments because the capture argument first established (6.3); it does not assume witnesses use anchors only. The definition of width and Proposition 5.2 give the exact width. Taking $k = 2 r$ realizes each positive integer �. □

The upper bound assigns each edge of a complete bipartite graph to one of its endpoints, with at most $\lceil k / 2 \rceil$ edges assigned to any endpoint. The lower bound shows that arbitrary use of tail points cannot beat the corresponding incidence budget. The infinite target class is essential: a finite choice of representatives would have singleton witnesses by Theorem 5.3.

## 6.3 The unbounded-finite level

Let $\chi = A \sqcup B$ with both blocks countably infinite, and define

$$
{ \mathcal { H } } _ { \mathrm { t w o } } = \{ A \cup D : D \subseteq B \} \cup \{ B \cup E : E \subseteq A \} .\tag{6.4}
$$

Proposition 6.4 (The � level). The class $\mathcal { H } _ { \mathrm { t w o } }$ is generatable in the limit and $\mathfrak { s } ( \mathcal { H } _ { \mathrm { t w o } } ) = \omega$

Proof. Enumerate the blocks as $( a _ { i } ) _ { i \geq 0 }$ and $( b _ { i } ) _ { i \geq 0 }$ . For a proper target $A \cup D$ , put $i = \operatorname* { m i n } \{ r : b _ { r } \notin D \}$ and assign $\{ a _ { 0 } , \ldots , a _ { i } \}$ . For a proper target $B \cup E , \mathrm { p u t } j = \operatorname* { m i n } \{ r : a _ { r } \notin E \}$ and assign $\{ b _ { 0 } , \ldots , b _ { j } \}$ Assign the empty witness to $\chi ,$ the only target on both sides. Opposite proper targets cannot be active together: containment of the first witness in the second target forces $j > i ,$ , whereas containment of the second witness in the first forces $i > j$ . Every nonempty active family therefore shares an infinite block, or consists only of �. This gives width at most $\omega$

For any $k ,$ use the first � points in each block as anchors. The corresponding $\mathcal { H } ^ { ( k ) }$ is a subfamily of $\mathcal { H } _ { \mathrm { t w o } }$ . Monotonicity and Proposition 6.3 give width at least $\lceil k / 2 \rceil$ for every �. It cannot be finite and must equal $\omega$ □

Thus the pointwise finiteness in Theorem 2.3 cannot be strengthened to a uniform finite cardinality bound. A stronger fact holds: along the explicit chain $B \cup \{ a _ { 0 } , \ldots , a _ { n - 1 } \}$ , the number of �-points in every valid assignment’s witness tends to infinity. Proposition B.1 proves this assertion.

<table><tr><td>Class</td><td>Width</td><td>Reason</td></tr><tr><td>Infinite common core</td><td>0</td><td>Empty witnesses</td></tr><tr><td>Any countable class</td><td> $\leq 1$ </td><td>Distinct singleton witnesses</td></tr><tr><td>Anchored class  $\mathcal { H } ^ { ( k ) }$ </td><td> $\lceil k / 2 \rceil$ </td><td>Matching incidence bounds</td></tr><tr><td>Two-core class  $\mathcal { H } _ { \mathrm { t w o } }$ </td><td>ω</td><td>Unbounded finite witnesses</td></tr><tr><td>All infinite subsets</td><td> $\omega + 1$ </td><td>No finite witness assignment</td></tr></table>

Table 1: The separation-width hierarchy. The two infinite levels distinguish unbounded finite certificates from the absence of any finite-certificate assignment. These widths are not convergence-time bounds.

## 6.4 The endpoints and full range

Proposition 6.5. The family of all cofinite subsets of N has width 1, whereas the family $[ \mathbb { N } ] ^ { \omega }$ of all infinite subsets has width $\omega + 1$

Proof. The cofinite family is countable and has empty common intersection, so Theorem 5.3 and Proposition 5.6 give width exactly one. For all infinite targets, fix an arbitrary total generator. Maintain finite disjoint sets of revealed and permanently forbidden points. At each round reveal a new point outside their union. If the output is already revealed, freshness fails; otherwise permanently forbid it. There is always another point available. The resulting input sequence has infinite range � and is exhaustive for that fixed resulting target. Every output either repeats an input or is absent from �. Thus the generator fails on �. Since the generator was arbitrary, Corollary 5.5 gives width $\omega + 1$ □

Completion of Theorem 6.1. The countable bound is Theorem 5.3. A single infinite target has width zero. Taking $k = 2 d$ in Proposition 6.3 realizes every positive integer $d ,$ and Proposition 6.4 realizes �. These classes are generatable by Corollary 5.5. The all-infinite family has width $\omega + 1$ by Proposition 6.5. Bĳective relabeling gives the claims on any countably infinite �. □

## 7 Why local dimensions cannot sufice

The preceding width is global: its assignment must handle every bad subfamily at once. The following obstruction explains why a dimension built only from locally realized configurations cannot simply replace it.

Theorem 7.1 (Countable-support obstruction). There is no invariant $D ( \mathcal { H } ) \in \mathbb { N } _ { 0 } \cup \{ \infty \}$ satisfying both (i) ℋ is generatable in the limit if and only $i f D ( \mathcal { H } ) < \infty ;$

(ii) whenever $D ( \mathcal { H } ) = \infty .$ , some countable $\mathcal { H } _ { 0 } \subseteq \mathcal { H }$ also has $D ( \mathcal { H } _ { 0 } ) = \infty$

No monotonicity hypothesis is needed for this impossibility.

Proof. All infinite subsets of N form a nongeneratable class by Proposition 6.5, so (i) gives infinite dimension. Condition (ii) then supplies a countable subfamily of infinite dimension. That subfamily is generatable by Theorems 2.3 and 5.3, contradicting (i). □

For a dimension defined by configurations of arbitrarily large finite depth, condition (ii) follows whenever each depth has at most countably many language realizers and collecting those realizers preserves the configuration. Choose one witness family at each integer depth and take their countable union. Finite VC shattering and finite-depth Littlestone shattering have this property. A finite positive-closure obstruction also has countable support: select, for each point outside its finite core, a consistent language omitting that point. Adding more consistent languages can only shrink the core.

There is also a finite-level failure of countable determination. For every integer $d \ge 1$ , the class $\mathcal { H } ^ { ( 2 d + 1 ) }$ has width $d + 1$ , while each of its countable subfamilies has width at most one. Thus even the obstruction ${ \mathfrak { s } } ( { \mathcal { H } } ) > d$ need not be witnessed by a countable subclass.

This argument does not cover every infinitary rank. A tree condition requiring each complete infinite branch to be realized by an actual target, or a global compatibility condition such as (5.1), need not be supported on a countable subfamily. The theorem’s explicit support assumption is essential.

Proposition 7.2 (Identical finite profiles). The cofinite class $\mathcal { H } _ { \mathrm { c f } }$ and the class $\mathcal { H } _ { \mathrm { i n f } } = [ \mathbb { N } ] ^ { \omega }$ have identical traces on every finite domain and identical positive closures at every finite sample. Their generatability in the limit difers.

Proof. For finite $F \subseteq \mathbb { N }$ and $E \subseteq F ,$ the cofinite target N $\backslash ( F \backslash E )$ has trace � on �. Thus both classes realize all finite traces. At a finite positive sample �, every consistent target contains �, whereas for each $x \notin S$ the cofinite target N \ {�} is consistent and omits �. Both positive closures are exactly �. Their difering status follows from Proposition 6.5 and Corollary 5.5. □

Consequently, even the complete collection of finite trace sets or of finite-sample positive closures cannot determine generation in the limit. This includes invariants determined solely by these profiles and their finite iterations; it does not include enriched data that retain additional relations among the realizing languages.

The width measures the sizes of the witnesses themselves. An alternative based on the size of bad active samples collapses to zero or infinity; Appendix D gives the short padding argument.

## 8 Scope and computational interpretation

The characterization separates two existence questions. A successful generator yields a finite witness for every target, and compatible witnessed targets must have infinitely many common elements. Conversely, an assignment with this intersection property specifies a generator. This is a structural existence statement about the family. The assignment and the intersections in (4.1) need not be computable.

The normalization theorem has a separate efective content: given a total computable successful generator, Algorithm 1 yields a total computable set-driven replacement. It requires neither efective membership in the unknown target nor an efective procedure for extracting $T _ { L }$ . The search may examine many words and is not accompanied by an eficiency guarantee. Set-drivenness removes dependence on order and multiplicity; it does not bound the memory required to retain the observed set. Memory constraints define a separate restriction on generation, studied by Kleinberg et al. (2026).

The assumptions also have distinct roles in the proof. Countability supplies an exhaustive ordering of target elements and a finite-predecessor ordering of words. Infinitude supplies fresh outputs and gives checkpoints of every finite size. Exhaustiveness both rules out an infinite marked error sequence and ensures that the final finite witness is eventually observed. The theorem covers precisely these positive-text, single-element generation requirements; it imposes no identification or output-diversity requirement.

Combinatorial meaning and remaining questions. Positive separation width measures the number of positive incidences that each target must contribute to separate every subfamily with finite core. The finite anchored examples give matching lower and upper bounds, and the � example separates pointwise finiteness from every uniform finite budget. The definition still optimizes a global assignment; the present results do not provide an intrinsic tree-rank formula or a method to compute the width from a presentation of the class. The countable-support obstruction identifies a constraint on such a formula rather than prohibiting all infinitary alternatives.

It remains useful to ask which additional regularity assumptions permit canonical witness choices, which operations preserve finite width, and what extra information would relate witness size to observation or mistake complexity. Exhaustive texts alone allow any finite certificate to be delayed, so such quantitative conclusions require separate hypotheses. Eficient witness discovery and active-intersection selection remain separate algorithmic questions.

Formal verification. The characterization and full width hierarchy are checked in Lean. The simplified normalization has a separately checked implementation, and the diagonal capture argument has its own checked proof. Appendix E describes the correspondence, implementation variants, and trusted dependencies.

AI Disclosure. OpenAI Codex assisted with mathematical exploration, proof development, Lean formalization, and manuscript preparation. Responsibility for the paper’s claims and interpretations rests with the authors.

## References

Dana Angluin. Inductive inference of formal languages from positive data. Information and Control, 45(2):117–135, 1980. doi: 10.1016/S0019-9958(80)90285-5.

Yannan Bai, Debmalya Panigrahi, and Ian Zhang. Language generation in the limit: Noise, loss, and feedback. In Proceedings of the 2026 Annual ACM-SIAM Symposium on Discrete Algorithms, pages 794–816. Society for Industrial and Applied Mathematics, 2026. doi: 10.1137/1.9781611978971.31.

E. Mark Gold. Language identification in the limit. Information and Control, 10:447–474, 1967. doi: 10.1016/S0019-9958(67)91165-5.

Steve Hanneke, Amin Karbasi, Anay Mehrotra, and Grigoris Velegkas. On union-closedness of language generation. In Advances in Neural Information Processing Systems, volume 38, 2025. doi: 10.52202/085713-3326.

Alkis Kalavasis, Anay Mehrotra, and Grigoris Velegkas. Characterizations of language generation with breadth. In Proceedings of the International Conference on Algorithmic Learning Theory, 2026. URL https://arxiv.org/abs/2412.18530. arXiv:2412.18530.

Jon Kleinberg and Sendhil Mullainathan. Language generation in the limit. In Advances in Neural Information Processing Systems, volume 37, 2024. doi: 10.52202/079017-2111.

Jon Kleinberg and Fan Wei. Density measures for language generation. In Proceedings of the 66th Annual IEEE Symposium on Foundations of Computer Science, pages 620–658. IEEE, 2025. doi: 10.1109/FOCS63196.2025.00034. URL https://arxiv.org/abs/2504.14370.

Jon Kleinberg and Fan Wei. Language generation and identification from partial enumeration: Tight density bounds and topological characterizations. In Proceedings of the 58th Annual ACM Symposium on Theory of Computing, 2026. URL https://arxiv.org/abs/2511.05295. arXiv:2511.05295.

Jon Kleinberg, Anay Mehrotra, Amin Saberi, and Grigoris Velegkas. On language generation in the limit with bounded memory, 2026. URL https://arxiv.org/abs/2605.30324. arXiv:2605.30324.

Timo Kötzing, Martin Schirneck, and Karen Seidel. Normal forms in semantic language identification. In Proceedings of the 28th International Conference on Algorithmic Learning Theory, volume 76 of Proceedings of Machine Learning Research, pages 493–516. PMLR, 2017. URL https://proceedings. mlr.press/v76/k%C3%B6tzing17a.html.

Xiaoyu Li, Andi Han, Jiaojiao Jiang, and Junbin Gao. Contrastive identification and generation in the limit, 2026a. URL https://arxiv.org/abs/2605.06211. arXiv:2605.06211.

Xiaoyu Li, Andi Han, Dai Shi, Zheng Gao, Jiaojiao Jiang, and Junbin Gao. Flood and harvest: The provable necessity of trivia for generating valuable mathematics via the lens of language generation in the limit, 2026b. URL https://arxiv.org/abs/2606.14688. arXiv:2606.14688.

Anay Mehrotra. Learning Theory in the Wild: Foundations of Missing Data and Language Generation. PhD thesis, Yale University, May 2026. URL https://anaymehrotra.com/static/anay\_mehrotra\_ phd\_thesis\_2026.pdf.

Vinod Raman, Jiaxun Li, and Ambuj Tewari. Generation through the lens of learning theory. In Proceedings of the Thirty Eighth Conference on Learning Theory, volume 291 of Proceedings ofMachine Learning Research, pages 4740–4776. PMLR, 2025. URL https://proceedings.mlr.press/v291/ raman25a.html.

## A Why sorting the observed set does not sufice

The normalization theorem constructs a new generator by finite simulation. The following example shows why simply sorting the input to an existing successful generator is not a valid replacement.

Let $\chi = \mathbb { N } _ { 0 }$ and let the sole target be $L = \mathbb { N } .$ . Set $G ( \varepsilon ) = 1$ . For a nonempty finite word $\sigma ,$ define

$$
G ( \sigma ) = \left\{ \begin{array} { l l } { 0 , } & { \sigma \mathrm { ~ i s ~ s t r i c t l y ~ i n c r e a s i n g ~ a n d ~ } \mathrm { c } ( \sigma ) \neq \{ 1 , \dots , \operatorname* { m a x } \mathrm { c } ( \sigma ) \} , } \\ { \operatorname* { m a x } \mathrm { c } ( \sigma ) + 1 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

On every text for $L ,$ this generator is eventually correct and fresh. If the entire text is strictly increasing, exhaustiveness forces it to be $1 , 2 , 3 , . . . ,$ and every output is correct. If it is not strictly increasing, there is a finite prefix that witnesses this fact. All later prefixes retain that witness, and their outputs are the maximum observed value plus one, hence valid and fresh.

Now define $h ( S ) = G ( \mathrm { s o r t } ( S ) )$ , where $\mathrm { \ s o r t } ( S )$ lists the distinct elements of � in increasing order. On the text

$$
1 , 3 , 2 , 5 , 4 , 7 , 6 , \dots ,
$$

every prefix ending at a newly introduced odd number $2 k + 1$ , for $k \geq 1$ , omits 2�. Sorting that prefix therefore triggers the output 0. The generator ℎ makes infinitely many errors.

The example separates these two transformations of a particular generator; it does not separate sequence-input and set-input generatability. The singleton family plainly has a successful set-driven generator.

## B Witness divergence along an explicit chain

The width-� example has a stronger property than the absence of a uniform bound. Its witnesses must become large along one specified chain, regardless of how the assignment depends on the targets.

Proposition B.1. Let � be any valid finite-witness assignment for $\mathcal { H } _ { \mathrm { t w o } }$ in (6.4). Enumerate � as $( a _ { i } ) _ { i \geq 0 }$ and put

$$
E _ { n } = \{ a _ { 0 } , \ldots , a _ { n - 1 } \} , \qquad K _ { n } = B \cup E _ { n } .
$$

Then $\left| T ( K _ { n } ) \cap B \right| \to \infty a s n \to \infty$ . The symmetric assertion holds with � and � interchanged.

Proof. Suppose some infinite subsequence of $Q _ { n } = T ( K _ { n } ) \cap B$ has bounded cardinality. For each finite sample �, let ${ \mathcal { F } } _ { A } ( S )$ consist of its active targets of the form $A \cup D ^ { \prime }$ , and set

$$
C _ { S } = \bigcap \{ D ^ { \prime } \subseteq B : A \cup D ^ { \prime } \in { \mathcal { F } } _ { A } ( S ) \} ,
$$

with empty intersection �. There are only countably many such cores. Apply Lemma 6.2 to the bounded subsequence and to the infinite members of this core collection. It supplies $D \subseteq B$ capturing infinitely many $Q _ { n }$ and containing no infinite $C _ { S }$

For the target $L = A \cup D$ , the finite set $T ( L ) \cap A$ lies in $E _ { n }$ for all suficiently large �. Choose such a captured index. Positivity gives $T ( L ) \subseteq K _ { n }$ and $T ( K _ { n } ) \subseteq L ,$ , so both targets are active at $S \ : = \ : T ( L ) \cup T ( K _ { n } )$ . Their full active intersection � is infinite. Since $J \subseteq B \cup E _ { n }$ and $E _ { n }$ is finite, $J \cap B$ is infinite. It is contained in $C _ { S } ,$ while the active target � gives $C _ { S } \subseteq D$ . This contradicts the construction of �. Thus no bounded-cardinality infinite subsequence exists, which is exactly the asserted divergence. The symmetric proof interchanges the blocks. □

This argument gives no universal rate of divergence and no generation-time bound. It concerns the cardinalities of the certificates assigned to a specific increasing sequence of languages.

## C Increasing covers with eventually unbounded closure

The finite-witness criterion also recovers an existing suficient condition by a direct assignment.

For a language family � and a finite set $S$ contained in at least one member of $\mathcal { K } ,$ , define its positive closure by

$$
\mathrm { c l } _ { { \mathcal { K } } } ( S ) = \bigcap \{ K \in { \mathcal { K } } : S \subseteq K \} .
$$

If $F \subseteq S$ and the latter version space is nonempty, then $\mathrm { c l } _ { \mathcal { K } } ( F ) \subseteq \mathrm { c l } _ { \mathcal { K } } ( S )$ . We use the following finite-prefix formulation of the eventually unbounded closure property introduced by Raman et al. (2025).

Definition C.1 (Eventually unbounded closure). A family � has eventually unbounded closure (EUC) if for every $L \in \mathcal { K }$ there is a finite $F \subseteq L$ such that ${ \mathrm { c l } } _ { \mathcal { K } } ( F )$ is infinite.

This formulation is equivalent to requiring an infinite closure eventually along every text of every member. One direction follows by taking a prefix of any such text. Conversely, every text eventually contains a fixed finite $F ,$ after which closure monotonicity preserves infinitude. This equivalence uses only texts that enumerate a member of $\mathcal { K }$

Corollary C.2. Suppose $\textstyle { \mathcal { H } } = \bigcup _ { n \geq 1 } { \mathcal { H } } _ { n }$ , where $\mathcal { H } _ { 1 } \subseteq \mathcal { H } _ { 2 } \subseteq \cdots$ and every $\mathcal { H } _ { n }$ has EUC. Then ℋ satisfies thefinite-witness condition.

Proof. For each $L \in \mathcal H ,$ , choose $n ( L )$ with $L \in \mathcal { H } _ { n ( L ) }$ . By EUC, choose a finite $F ( L ) \subseteq L$ for which $\mathrm { c l } _ { \mathcal { H } _ { n ( L ) } } ( F ( L ) )$ is infinite. Extend $F ( L )$ to a finite $T ( L ) \subseteq L$ with $| T ( L ) | \geq n ( L )$

Fix � with a nonempty active family. For each active � we have $n ( L ) \leq | T ( L ) | \leq | S | ,$ , so the active layer indices have a maximum �. Choose an active $L _ { * }$ with $n ( L _ { * } ) = m$ . Because $\begin{array} { r } { F ( L _ { * } ) \subseteq S \subseteq L _ { * , } } \end{array}$ , the version space of $\mathcal { H } _ { m }$ at � is nonempty, and

$$
\mathrm { c l } _ { \mathcal { H } _ { m } } ( F ( L _ { * } ) ) \subseteq \mathrm { c l } _ { \mathcal { H } _ { m } } ( S ) .
$$

The right-hand side is therefore infinite. Increasingness of the cover puts every active language in $\mathcal { H } _ { m } ,$ , whence

$$
{ \mathrm { c l } } _ { { \mathcal { H } } _ { m } } ( S ) \subseteq \bigcap _ { L \in { \mathcal { A } } _ { T } ( S ) } L .
$$

The active intersection is infinite, as required.

The corollary gives a witness-based proof of the increasing-EUC-cover suficient condition. It does not assert that such a cover is necessary. The full criterion in Theorem 2.3 concerns the joint compatibility of the assigned witnesses at each finite sample.

## D Why optimizing bad-sample size collapses

A diferent shortcut fails for a quantitative reason. One might assign finite witnesses first, measure the largest bad active sample, and then optimize that measurement. The optimization removes every finite defect.

Proposition D.1 (Padding collapse). For afinite positive assignment �, define

$$
\begin{array} { r } { b _ { T } ( \mathcal { H } ) = \operatorname* { s u p } \left\{ \left| S \right| + 1 : \mathcal { H } _ { T } ( S ) \neq \emptyset , \left| \operatorname { c o r e } ( \mathcal { A } _ { T } ( S ) ) \right| < \infty \right\} , } \end{array}
$$

with empty supremum zero and unbounded supremum ∞. Then

$$
\operatorname* { i n f } _ { T } b _ { T } ( \mathcal { H } ) = \left\{ \begin{array} { l l } { 0 , } & { \mathcal { H } g e n e r a t a b l e , } \\ { \infty , } & { \mathcal { H } n o n g e n e r a t a b l e . } \end{array} \right.
$$

Proof. Suppose $b _ { T } ( \mathcal { H } ) = d < \infty$ . Enlarge each $T ( L )$ within its infinite target to a finite $T ^ { \prime } ( L )$ of size at least $d ,$ retaining $T ( L )$ . If a sample � activates a target under $T ^ { \prime } { _ { - } }$ , then $| S | \geq d$ and the original active family is nonempty. Its core cannot be finite, since that would imply $| S | + 1 \leq d$ . The new active family is smaller, so its core contains the old infinite core. Thus $T ^ { \prime }$ is valid and $b _ { T ^ { \prime } } ( \mathcal { H } ) = 0$ . A finite infimum over $\mathbb { N } _ { 0 } \cup \lbrace \infty \rbrace$ entails some finite-valued assignment and hence, by this argument, a zero-valued one. Conversely a valid assignment has value zero. Apply Theorem 2.3. □

Positive separation width retains the size cost of padding and therefore escapes this collapse. Propositions 6.3 and 6.4 show that this cost has every finite level as well as a necessary unboundedfinite level.

## E Formalization and implementation details

We implemented the information-theoretic characterization, universal locking normalization, positive-separation equivalence, and the full separation-width hierarchy in Lean 4.24.0. Our develop ment is maintained at https://github.com/xiaoyulics/language-generation-characterization It extends the public library generation-in-the-limit-lib developed and maintained by Shuangping Li and Peng Zhang, at commit de0d70c7e4645bface1d19bded9c8a5ade080fa8. We retain their core definitions and Apache-2.0 license; the FiniteWitness modules are additions for this paper. The repository’s credits and provenance manifest distinguish the inherited files from our extensions. The checked definition of generation in the limit is the library’s sequence-input, exhaustive-positive-text notion. The checked theorem axiom closures contain only propext, Classical.choice, and Quot.sound. The extension also checks singleton witnesses, the width endpoints and restriction law, chain divergence, the dimension and local-profile obstructions, padding collapse, EUC consequences, and the sorting counterexample. The current normalization is formalized separately in the Simplified modules. First-� checkpoints are defined and verified in the chosen point order; their cardinality, exhaustion, and agreement properties are proved. The cutof-free search is proved to satisfy the universal locking conclusion, with the generator fixed before the target. The diagonal capture proof uses nested infinite pools of indices and preserves the original lemma’s arbitrary-universe statement.

On the natural-number universe, an executable implementation realizes the new least-code search. At each round, the history obtained by appending the sorted sample is a known candidate. Its code bounds a finite scan that finds the globally least candidate. Equality with the mathematical search and the locking theorem are checked in Lean. This construction supplies the computability preservation argument in Corollary 3.2 without requiring a target-membership oracle.

The repository also retains the earlier bounded implementation and its separate guarantee that only words of length at most 2|�| are queried. That bound concerns the earlier algorithm; the current least-code search uses the adaptive code bound just described. Both satisfy universal locking. The earlier formal implementation uses ambient-code checkpoints, while the current one uses the first-� checkpoints stated in the paper.

These are executable Lean definitions with correctness theorems. The packet does not include a separate Mathlib Computable/Partrec or formal Turing-machine compiler theorem. Formal acceptance establishes the encoded statements under the declared trust base; novelty and eficiency require separate arguments.