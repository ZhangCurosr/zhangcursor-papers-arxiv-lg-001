# How Much Does One Wrong Answer Cost? Error Propagation in Constraint Pruning with Noisy LLM Oracles

Zhengtao Gui

Weining Shen

University of California, Irvine

## Abstract

Constraint pruning saves queries to a large language model (LLM) or another noisy oracle by skipping each yes/no question implied by answers accepted earlier, for example through transitivity. Entity-resolution systems and LLM-based rankers use this idea. However, their analyses typically assume a correct oracle. We argue that pruning should be judged by its risk when the oracle errs, since an accepted wrong answer spreads to every question it implies. When each answer is wrong independently with small probability ε, the first-order risk is set by how many facts one wrong answer corrupts on average. We consider two kinds of constraints. For implication rules, as in taxonomies, pruning in a random order adds no first-order risk: a wrong answer that would change k facts is accepted with probability 1/k, so it costs one fact on average. Rules with s premises break this balance: a wrong answer at a premise shared by r rules can cost order $r ^ { 1 - 1 \bar { / } s }$ facts. For entity resolution with m records of one entity, the first-order risk equals ε times the Wiener index (the total pairwise distance) of the spanning tree built by the query order. Pivot orders, which first compare one record with all others, amplify errors by less than 2 and are optimal; random orders by order $m ^ { 1 / 3 }$ ; the common most-similar-first order is the worst in one dimension and, for large m, worse than random in two. With several entities, the cost of wrongly merging two does not depend on how many others exist. Simulations agree with every predicted constant within sampling error.

## 1 INTRODUCTION

Constraint pruning reduces the cost of collecting related yes/no facts from an oracle, such as an LLM or a human annotator, whose answers may be wrong. In entity resolution, a system asks whether two records refer to the same entity and skips pairs implied by transitivity (Wang et al., 2013; Vesdapunt et al., 2014). For example, accepting that A and B are the same and that B and C are the same determines the answer for A and C without another query. Similar propagation occurs when enforcing entailment and contradiction constraints between LLM beliefs (Mitchell et al., 2022), skipping comparisons in LLM ranking (Agrawal et al., 2026), or propagating orientation hints in causal discovery (Meek, 1995; Long et al., 2023; Lyu et al., 2026).

However, an accepted error can also propagate to unqueried facts. If A, B and C belong to one entity, a correct “same” answer for A, B followed by a wrong “diferent” answer for B, C implies a wrong “diferent” answer for A, C. In the synthetic citation records of Figure 1, one oracle error thus produces two final wrong decisions and saves one query, and the output is still logically consistent. Query counts alone cannot capture this cost. Prior work studies recovery probabilities, assumes deterministic oracles or examines the robustness of query structures (Long et al., 2023; Hiremath et al., 2026; Agrawal et al., 2026; Galhotra et al., 2018).

We study greedy closure decoding: process questions in an order, skip those implied by accepted answers, and otherwise query and accept the oracle’s answer, without revisiting earlier decisions. We measure risk by the expected number of final wrong facts, includ ing inferred ones. The raw baseline asks every question and uses its answer unchanged, without enforcing consistency. Both procedures use the same potential answers: the answers the oracle would give if each question were asked.

In single-error analysis, we make one potential answer wrong and keep all others correct. We count the final wrong facts caused by each error location, averaging over query orders when they are random. In Figure 1, damage is 2; it is 0 if earlier correct answers imply and skip the wrong question. Under independent errors of small probability ε (Long et al., 2023; Hiremath et al., 2026), events with two or more errors have probability $O ( \varepsilon ^ { 2 } )$ for a fixed finite instance. A reduction lemma expresses risk as ε times the sum of single-error damages, plus O(ε<sup>2</sup>). The term linear in ε is the first-order risk; its ratio to the raw baseline’s nε is the error amplification, where n counts questions. Random-order ranking and Meek propagation are not analyzed.

![](images/09f76ceb40c5fa1cdd5782292ad2fb7b0a4c89e867d94503074e7681ede7770d.jpg)

![](images/dd5c08080134ec18370fc21c790d05ba3b582cddad9c08011ed8936b76c24962.jpg)

![](images/908d4fd003e60f8b11e4a19b2814c1610ee23ff236d23ac95edb2c253832594b.jpg)  
Figure 1: Propagation in three synthetic citation records of one publication: one wrong answer produces two wrong pair decisions and saves one query.

Using single-error analysis, we derive the following first-order risk laws:

• Risk conservation for pairwise constraints. For every satisfiable 2-CNF theory, including implications and pairwise contradictions, uniformly random orders give mean single-error damage 1 for unfixed facts and 0 for facts fixed by the constraints. Pruning therefore does not increase firstorder risk over raw decoding (Section 4).

• Error amplification through a shared premise. For k Horn rules with $s \geq 2$ premises each and one shared false premise, we derive the exact mean damage of a single wrong answer there, growing as $\Theta \bar { ( k ^ { 1 - 1 / s } ) }$ under uniformly random orders, for fixed s (Theorem 6).

• Relating query order to tree structure. For m records of one entity, independent errors at a common rate give first-order risk equal to that rate times the noiseless query tree’s Wiener index (total pairwise tree distance). Pivot stars minimize this risk among greedy orders, with amplification below 2; random orders give $\Theta ( m ^ { 1 / 3 } )$ , and similarity-first can be worse under Theorem 11’s embedding assumptions. We also give bounds for multiple errors at any error rate (Section 5.1).

• Locality of a single false merge. With only one wrong answer, false-merge damage depends only on the two afected entities. This yields separate within-entity and cross-entity first-order risks. We give bounds, exact small cases and the pivot pipeline’s exact first-order formula; the large-cluster growth exponent remains a conjecture (Section 5.2).

We test the theoretical predictions and compare query orders using simulations and LLM answers on the Cora citation data (Section 6).

## 2 RELATED WORK

Noisy recovery and tree structure. Work on probabilistic unclean databases relates integrity constraints to minimum-repair and maximum-likelihood decoding (De Sa et al., 2019; Gilad et al., 2023). Other studies consider noisy graph-label recovery (Globerson et al., 2015; Foster et al., 2018; Heidari et al., 2020; Lee et al., 2022; Jin et al., 2026), correlation clustering (Makarychev et al., 2015), same-cluster queries (Mazumdar and Saha, 2017a) and noisy sorting (Feige et al., 1994; Braverman and Mossel, 2008). Their main goals are query eficiency or global recovery guarantees, usually at constant noise (Banerjee and Shen, 2022). Our tree-based analysis uses Kruskal’s construction (Kruskal, 1956), extremal properties of the Wiener index (Dobrynin et al., 2001), and results on random and spatial spanning trees (Addario-Berry et al., 2009, 2017; Aizenman et al., 1999).

Entity resolution and LLM comparisons. Crowdsourced entity resolution uses transitivity (Wang et al., 2013; Vesdapunt et al., 2014). Analyses of query ordering typically assume a correct oracle (Firmani et al., 2016; Mazumdar and Saha, 2017b); robust methods add verification queries (Galhotra et al., 2018) or weaken transitivity (Gruenheid et al., 2015). Recent LLM work studies prompt sensitivity (Gui et al., 2026) and matching errors (Peeters et al., 2025), compares matching strategies (Wang et al., 2025), filters candidates using matching feedback (Chen et al., 2026), and balances quality and cost through operator and model selection (Guo et al., 2026).

Ranking work addresses non-transitive preferences and baseline-sensitive rankings (Xu et al., 2025), presentation-order and transitivity inconsistencies (Zeng et al., 2026), and risks from long inference chains (Agrawal et al., 2026). These studies improve matching, ranking or query allocation. We analyze how an accepted wrong answer causes errors in inferred decisions.

Logical consistency and causal hints. Beliefcorrection methods enforce entailment and contradiction relations (Mitchell et al., 2022). Logic.py translates natural-language puzzles into formal problems for a constraint solver (Kesseli et al., 2025). Causal discovery methods elicit and propagate hints about edge orientations (Long et al., 2023; Lyu et al., 2026; Kıcı- man et al., 2024), while Guess2Graph uses unreliable expert guesses to order conditional-independence tests (Hiremath et al., 2026). These methods correct answers or select hints. We study a sequential decoding process that accepts oracle answers in order and keeps accepted answers fixed during subsequent inference.

## 3 SETTING AND REDUCTION LEMMA

Questions, constraints and noisy answers. There are n binary questions. Complete vectors satisfying the constraints form the code ${ \mathcal { C } } \subseteq \{ 0 , 1 \} ^ { n } ;$ ; each is a feasible codeword. They are the models (satisfying assignments) of a propositional theory Φ. The decoder knows the constraints but not the fixed truth $x ^ { \star } \in \mathcal { C }$ We study implication, 2-CNF and definite Horn theories (Section 4), partitions under “same entity” transitivity (Section 5), and total orders under “before” transitivity (Section A.4).

The oracle’s potential answer y<sub>i</sub> to question i is observed only if queried. Thus $y = x ^ { \star } \oplus Z$ , where $Z _ { i }$ are independent Bern(ε) variables: $Z _ { i } ~ = ~ 1$ flips the true answer and $Z _ { i } ~ = ~ 0$ leaves it unchanged. Section 5.2 allows two rates. Hamming loss $d _ { \mathrm { H } } ( \hat { x } , x ^ { \star } )$ counts wrong final answers, including inferred ones; risk $R ( \varepsilon ) = \mathbb { E } d _ { \mathrm { H } } ( { \hat { x } } , x ^ { \star } )$ is its expectation. The raw decoder asks all questions and returns $y ,$ with risk

$R _ { \mathrm { r a w } } = \varepsilon n ;$ the output may violate constraints.

The greedy decoder $G _ { \pi }$ follows order π, storing accepted question–answer pairs in an initially empty set $L .$ . Among codewords agreeing with L, if all give the same answer at $i ,$ use it and skip the query. Otherwise both answers are feasible: query the oracle and add $( i , y _ { i } )$ to $L ,$ keeping this answer fixed. A wrong answer can thus be accepted while preserving feasibility. After all questions, one codeword remains (Lemma 16), also obtained by the accept-if-consistent rule. Unless stated otherwise, π is uniform and independent of Z. Implication tests take polynomial time without enumerating $\mathcal { C } \colon$ satisfiability tests for 2-CNF and Horn theories, union-find for partitions and transitive closure for orders. The maximum-likelihood (ML) decoder uses all answers jointly, returning a codeword closest to $y$ in Hamming distance, with uniform tiebreaking.

Single-error damage. For a decoder xˆ whose output depends on y and on an independent seed π, the single-error damage at question a is the expected loss when only the answer to a is wrong,

$$
D ( a ) = \mathbb { E } _ { \pi } d _ { \mathrm { H } } \big ( \hat { x } ( x ^ { \star } \oplus e _ { a } , \pi ) , x ^ { \star } \big ) ,
$$

where $e _ { a }$ is the a-th unit vector. For the greedy decoder we write $D _ { \pi } ( a ) = d _ { \mathrm { H } } ( G _ { \pi } ( x ^ { \star } \oplus e _ { a } ) , x ^ { \star } )$ for a fixed order, so that $D ( a ) = \mathbb { E } _ { \pi } D _ { \pi } ( a )$ . The first-order constant is $\kappa = \mathrm { l i m } _ { \varepsilon  0 } R ( \varepsilon ) / \varepsilon$ , and the amplification of a decoder is $\kappa / \kappa _ { \mathrm { r a w } } = \kappa / n$ , the mean damage of one wrong answer relative to the raw decoder.

For $x \in { \mathcal { C } }$ let $m _ { a } ( x )$ be the number of codewords of the form $x \oplus e _ { a } \oplus e _ { b } , b \neq a ,$ when $x \oplus e _ { a } \not \in { \mathcal { C } }$ (and 0 otherwise). In that case they are the codewords other than x that tie with x as nearest to $x \oplus e _ { a }$

Lemma 1 (Reduction lemma). Let xˆ be any decoder that returns $x ^ { \star }$ when there is no wrong answer. For every $\varepsilon \in [ 0 , { \frac { 1 } { 2 } } ]$

$$
\begin{array} { r } { \displaystyle \Big | R _ { \hat { x } } ( \varepsilon ) - \varepsilon \sum _ { a = 1 } ^ { n } D ( a ) \Big | \le \frac { 3 } { 2 } \varepsilon ^ { 2 } n ^ { 2 } ( n - 1 ) . } \end{array}
$$

Moreover: (i) ML has $D _ { \mathrm { M L } } ( a ) = 1 ~ i f ~ x ^ { \star } \oplus e _ { a } \in { \mathcal { C } }$ and $D _ { \mathrm { M L } } ( a ) = 2 m _ { a } / ( 1 + m _ { a } )$ otherwise. (ii) The greedy decoder has first-order constant $\begin{array} { r } { \sum _ { a } \mathbb { E } _ { \pi } D _ { \pi } ( a ) } \end{array}$ , so its amplification is the mean damage of one wrong answer.

The remainder grows with $n ,$ so the expansion describes a fixed instance as $\varepsilon \to 0$

## 4 IMPLICATION THEORIES AND HORN RULES

We apply the single-error analysis of Section 3 to logical rules to identify when constraint propagation amplifies errors. We first show that, under uniformly random query orders, implication rules and, more generally, 2-CNF constraints do not increase first-order risk over raw decoding; a shared-premise Horn family then shows why this conclusion does not hold in general.

Table 1: Mean single-error damage; skipped errors count as zero. Logical rows use uniform orders; partition rows average over all within- or cross-cluster pairs, giving first-order amplification over raw risk. Optimality is among greedy orders at first order. Conj.: conjectured growth.
<table><tr><td>Setting</td><td>Mean damage (wrong final facts)</td><td>Reference</td></tr><tr><td>2-CNF, implication rules</td><td>1 for unfixed facts, 0 for fixed facts (backbone);  ${ \mathrm { M L } } \colon \leq 1$ </td><td>Cor. 4; Prop. 5</td></tr><tr><td>Horn hub: k rules,  $s \geq 2$  premises</td><td> $\sim \Gamma ( 1 + { \textstyle { \frac { 1 } { s } } } ) k ^ { 1 - 1 / s }$  at the shared premise;  ${ \mathrm { M L } } \colon 0 \ ( k \geq 2 )$ </td><td>Thm. 6; Rem. 18</td></tr><tr><td>One entity, m records</td><td>pivot  $2 ( \bar { m } - 1 ) / m$  (optimal); random  $\Theta ( m ^ { 1 / 3 } ) ;$  similarity-first under Thm. 11:  $\gtrsim ( m / \log m ) ^ { 1 / d }$ </td><td>Props. 9, 10; Thm. 11</td></tr><tr><td>Cross pair, s records per entity</td><td>random  $\begin{array} { r } { g ( s ) \in [ \frac { s - 1 } { 2 s - 1 } } \end{array}$  , 13s], conj.  $\Theta ( s ^ { 2 / 3 } )$  pivot  $( 2 s - 1 ) / s$ </td><td>Props. 13, 14; Conj. 15</td></tr><tr><td>Total orders</td><td>randomized quicksort → 2</td><td>Prop. 24</td></tr></table>

Let Φ be a satisfiable theory over atoms $V , | V | = n$ each representing a binary fact. Write the truth $x ^ { \star }$ as a satisfying assignment $\sigma ^ { \star }$ here. A literal is an atom or its negation. Let $t _ { x } = x$ when $\sigma ^ { \star } ( x ) = 1$ and $t _ { x } = \lnot x$ otherwise, with $\bar { t } _ { x }$ the opposite literal. The backbone $\boldsymbol { B } = \{ \boldsymbol { x } : \boldsymbol { \Phi } \ v { x } \ v { x } \ v { x } \}$ contains facts fixed by the rules (|= denotes logical entailment). For $a \notin B$ , the flip set

$$
K _ { a } = \{ x \in V : \ \Phi \land \bar { t } _ { a } \in \bar { t } _ { x } \}
$$

contains a and every atom whose value is forced to change by a wrong value at a. We say that a has the single-flip property if flipping exactly the atoms of $K _ { a }$ in $\sigma ^ { \star }$ , while leaving all others unchanged, gives another model of Φ.

Theorem 2 (Damage of one wrong answer). Suppose that the only wrong answer is at atom a. If $a \in B ,$ the greedy decoder returns the truth for every order. If $a \notin B$ has the single-flip property, then for every order π

$D _ { \pi } ( a ) = | K _ { a } | \cdot { \bf 1 }$ {a comes first in π among $K _ { a } \}$

so $\mathbb { E } _ { \pi } D _ { \pi } ( a ) = 1$ for a uniformly random (or any exchangeable) order.

In the second case, a correct value on another atom of $K _ { a }$ implies a’s true value and skips its query. Only if a comes first is the error accepted, causing $| K _ { a } |$ wrong facts. Its probability $1 / | K _ { a } |$ yields mean damage $1 ;$ individual runs have damage 0 or $| K _ { a } |$

Proposition 3. If every clause of Φ has at most two literals, every atom outside the backbone has the singleflip property.

Proposition 3 holds because, if a flip makes one literal of a two-literal clause false, $\Phi \wedge \bar { t } _ { a }$ forces the other to be true, preserving the clause.

Corollary 4 (Conservation law). Let Φ be a satisfiable 2-CNF theory with backbone B. For every $\varepsilon \in [ 0 , { \frac { 1 } { 2 } } ]$

$$
\begin{array} { r } { - \varepsilon ^ { 2 } n ( n - 1 ) \ \le \ R _ { \mathrm { g r } } ( \varepsilon ) - \varepsilon ( n - | B | ) \ \le \ \frac 1 2 \varepsilon ^ { 2 } n ^ { 2 } ( n - 1 ) . } \end{array}
$$

In particular, for every implication graph $G = ( V , A )$ with $\Phi _ { G } ~ = ~ \bigwedge _ { ( u , v ) \in A } ( \neg u \vee v )$ , greedy decoding has $R _ { \mathrm { g r } } ( \varepsilon ) = \varepsilon n + { \dot { O } } ( \varepsilon ^ { 2 } ) = R _ { \mathrm { r a w } } ( \varepsilon ) + O ( \varepsilon ^ { 2 } )$ , for falsepositive and false-negative errors alike and for every truth.

Thus the first-order risk is $\varepsilon ( n - | B | )$ ; implication graphs have no fixed facts and match raw at first order. Multiple errors can still cause diferences. If accepted, a false positive (false to true) changes false descendants, and a false negative (true to false) changes true ancestors. The order matters: on $v _ { 1 } \ \to \ \cdots \ \to \ v _ { n }$ queried as $v _ { 1 } , \ldots , v _ { n } .$ , a single false positive is accepted and changes every later false fact.

Proposition 5 (ML on 2-CNF). On a satisfiable 2- CNF theory, the ML single-error damage at a /∈ B is $\textit { 1 } i f \left| K _ { a } \right| \leq 2$ and $\textit { 0 i f } | K _ { a } | \geq 3$ . Hence $R _ { \mathrm { M L } } ( \varepsilon ) =$ $R _ { \mathrm { g r } } ( \varepsilon ) - \varepsilon \# \{ a \not \in B : | K _ { a } | \geq 3 \} + O ( \varepsilon ^ { 2 } )$

Horn rules. The Horn hub $H _ { k , s }$ has k rules $b _ { i , 1 } \wedge$ $\cdot \cdot \cdot \wedge b _ { i , s - 1 } \wedge a \to c _ { i } ( i = 1 , \dots , k )$ , sharing premise a. Each rule’s other premises and conclusion occur in no other rule. All $b _ { i , j }$ are true and $a , c _ { i }$ are false, so $n = 1 + k s$ . For $s = 2$ this is the pattern isa $( x , y ) \wedge$ has $( y , z ) \to \mathrm { h a s } ( x , z )$ with a shared premise.

Theorem 6 (Horn hub). For a single false positive at the shared premise a of $H _ { k , s } , s \geq 2$ , and a uniformly random order,

$$
\begin{array} { l } { \displaystyle \mathbb { E } _ { \pi } D _ { \pi } ( a ) = ( k + 1 ) \int _ { 0 } ^ { 1 } ( 1 - u ^ { s } ) ^ { k } d u } \\ { \displaystyle = \frac { ( k + 1 ) \Gamma ( 1 + \frac { 1 } { s } ) \Gamma ( k + 1 ) } { \Gamma ( k + 1 + \frac { 1 } { s } ) } , } \end{array}
$$

which equals $\Gamma ( 1 + { \textstyle \frac { 1 } { s } } ) k ^ { 1 - 1 / s } ( 1 + { \cal O } ( 1 / k ) )$ ). Every other single error on $H _ { k , s }$ has damage exactly 1. Hence $R _ { \mathrm { g r } } ( \varepsilon ) = R _ { \mathrm { r a w } } ( \varepsilon ) + \varepsilon \left[ \mathbb { E } _ { \pi } D _ { \pi } ( a ) - 1 \right] + O ( \varepsilon ^ { 2 } )$

Here Γ is the Gamma function. The mechanism difers from Theorem 2: a wrong a alone implies no other values. It is skipped only if one rule’s other s atoms were processed earlier: their true premises and false conclusion force a false. If accepted, each rule’s last atom is inferred wrongly and skipped; it can be a premise or the conclusion. Including $^ { a , }$ the total damage is $k + 1$

Appendix A.2 gives the ML proof and the arrival-time derivation of the Horn formula, including its $s \ = \ 2$ special case.

For Horn theories with at most s premises per rule, we conjecture that mean false-positive damage at a is bounded by $C _ { s } c ^ { 1 - 1 / s }$ , where c counts newly true atoms after adding a to the truth, including a (Conjecture 19). The hub shows this exponent cannot be lowered.

Under uniformly random query orders, acceptance probability and propagation damage balance for 2- CNF constraints, so pruning does not increase firstorder risk over raw decoding. The shared-premise Horn family shows that this balance can fail, with the mean cost of one wrong answer growing with the number of rules.

## 5 ENTITY RESOLUTION WITH TRANSITIVITY PRUNING

We now apply the greedy-decoding framework to entity resolution, where each pairwise answer states whether two records refer to the same entity. Transitivity lets “same” on A–B and B–C imply “same” on $A { \mathrm { - } } C _ { \mathrm { - } }$ saving a query but potentially spreading an error. This involves three pairwise facts, so the 2-CNF conservation law in Section 4 does not directly apply. We first study false splits within one entity, then false merges between entities.

## 5.1 One Cluster

The m records in [m] all refer to one entity, so every pair’s true answer is “same”. The algorithm does not know this: its feasible outputs are all partitions of [m], each specifying which records belong together. There are $\binom m 2$ pairwise questions. We write the error rate as $\delta \colon$ each answer is independently flipped to “different” with probability δ. The loss $L _ { \pi }$ counts pairs incorrectly separated in the final output.

The union-find pipeline $U _ { \pi }$ (Wang et al., 2013) maintains current groups of records, called blocks, and accepted $" \ne "$ constraints between them. A block need not coincide with a true entity. It scans pairs in order π. For $\{ u , v \}$ , membership in one block already implies “same”, while a $\neq$ constraint between their blocks implies “diferent”; either case skips the query. Otherwise it asks the oracle: “same” merges the blocks, and “diferent” records a constraint. Existing constraints are carried over when blocks merge. Lemma 20 shows that this is the greedy decoder on the partition code.

Construct the Kruskal tree $T _ { \pi }$ from the order alone, without using the noisy answers. Scan pairs in order π, retaining a pair only when it connects two components of the edges retained so far. With all answers correct, these $m - 1$ tree edges are exactly the queried pairs. For a tree T, let $d _ { T } ( u , v )$ count edges on the unique path from u to v. Its Wiener index (Wiener, 1947) is

$$
W ( T ) = \sum _ { \{ u , v \} } d _ { T } ( u , v ) = \sum _ { e \in T } n _ { e } ( m - n _ { e } )
$$

where deleting edge e splits the tree into parts of sizes $n _ { e }$ and $m - n _ { e }$ . Exactly $n _ { e } ( m - n _ { e } )$ pairwise paths use that edge, which gives the second equality.

Lemma 7 (Sandwich). Fix any order π and any pattern of wrong answers. For a pair $\{ u , v \}$ , let $k ( u , v )$ be the number of edges of $T _ { \pi }$ whose answer is wrong on the tree path from u to v. $I f k ( u , v ) = 0$ , then u and v end in the same block; $i f k ( u , v ) = 1$ , they end in diferent blocks. Hence $N _ { 1 } \leq L _ { \pi } \leq N _ { \geq 1 }$ , where $N _ { j }$ $( N _ { \geq j } )$ is the number of pairs with $k = j \ ( k \geq j )$

Every tree edge is queried even when errors occur, and each pipeline block stays inside a component of the forest built so far by the Kruskal scan. This yields Lemma 7. Paths with two or more wrong edges need not lead to separation; with a single wrong answer, the two bounds coincide and give the exact loss.

Theorem 8 (Wiener law). For every order π and every $\delta \in [ 0 , 1 ]$

$$
\delta W ( T _ { \pi } ) - \delta ^ { 2 } W _ { 2 } ( T _ { \pi } ) \ \leq \ \mathbb { E } L _ { \pi } \ \leq \ \delta W ( T _ { \pi } ) ,
$$

where $\begin{array} { r } { W _ { 2 } ( T ) = \sum _ { \{ u , v \} } d _ { T } ( u , v ) ^ { 2 } } \end{array}$ . A single wrong answer on a tree edge e costs exactly $n _ { e } ( m - n _ { e } )$ pairs, and a single wrong answer on any other pair costs nothing.

Theorem 8 makes the efect of the query tree explicit: for fixed m, $\mathbb { E } L _ { \pi } = \delta W ( T _ { \pi } ) + O ( \delta ^ { 2 } )$ as $\delta  0 ,$ with the displayed bounds valid for every $\delta \in [ 0 , 1 ]$ . Raw decoding has risk $\delta \left( { m \atop 2 } \right)$ , so first-order amplification is $W ( T _ { \pi } ) / \binom { m } { 2 }$ , the average tree distance. The singleerror cut identity and Lemma 7 hold for any error placement; the expectation uses independent errors with a common rate. Galhotra et al. (2018) previously identified the vulnerability of tree-based inference to errors; here the bounds quantify it for this decoder.

Proposition 9 (Pivot orders are optimal). (i) For every spanning tree $T$ of $K _ { m } , m \ge 3 , W ( T ) \ge ( m -$

$1 ) ^ { 2 }$ , with equality if and only if T is a star (a classical property of the Wiener index; Dobrynin et $a l . , ~ 2 0 0 1 )$ So the first-order risk $\delta W ( T _ { \pi } )$ is minimized exactly by the orders whose Kruskal tree is a star, such as asking every pair with a fixed pivot first (Vesdapunt et al., ${ \it 2 0 1 4 } ;$ their amplification is $2 ( m - 1 ) / m < 2 . ( i i )$ Among decoders that observe the answers on a fixed spanning tree T only (non-adaptive designs with $m -$ 1 queries, each observed answer flipped independently with probability δ), if $\delta \ \leq \ 1 / m$ the worst-case risk over the one-cluster truth and the $m - 1$ two-block partitions obtained by cutting one edge of T is at least $\bar { \frac { 1 } { 2 } } \delta ( 1 - \delta ) ^ { m - 2 } ( m - \bar { 1 } ) ^ { 2 }$ So, to first order in $\delta ,$ the worst-case risk of every such design is at least half of the pivot order’s risk $\delta ( m - 1 ) ^ { 2 }$ at the one-cluster truth.

Proposition 10 (Random order). For a uniformly random order, $T _ { \pi }$ is the minimum spanning tree MST<sub>m</sub> of $K _ { m }$ with $i . i . d .$ weights, and EL = $\delta \mathbb { E } W ( \mathrm { M S T } _ { m } ) + O ( \delta ^ { 2 } )$ with E $W ( \mathrm { M S T } _ { m } ) = \Theta ( m ^ { 7 / 3 } )$ an amplification $\Theta \dot { ( m ^ { 1 / 3 } ) }$ $4 t \ \delta = x m ^ { - 1 / 3 }$ with $x > 0$ fixed, the expected fraction of lost pairs stays bounded away from 0 and 1 as $m \to \infty$

Independent uniform edge weights generate a uniform pair order; known tree-distance estimates (Addario-Berry et al., 2009, 2017) then yield Proposition 10 through Lemma 7.

Theorem 11 (Most-similar-first order). Let items carry i.i.d. embeddings on $[ 0 , 1 ] ^ { d }$ with a density bounded below by $f _ { 0 } > 0$ , let $\mu _ { f }$ be the mean distance between two items, and let pairs be asked in increasing order of distance. Then $T _ { \pi }$ is the Euclidean minimum spanning tree, and for every m $\geq 2$ with $f _ { 0 } m \ge 2 \log m$ (natural logarithm),

$$
\frac { \mathbb { E } W ( \mathrm { E M S T } _ { m } ) } { \binom { m } { 2 } } \geq \frac { \mu _ { f } - \sqrt { d } / m } { 4 \sqrt { d } } \Big ( \frac { f _ { 0 } m } { 2 \log m } \Big ) ^ { 1 / d } .
$$

For $d = 1$ the amplification is exactly $( m + 1 ) / 3$

Under these embedding assumptions, similarity-first has larger asymptotic amplification than random order in one and two dimensions. Nearby connections can create long tree paths, giving a single error more opportunities to separate a pair. In one dimension, the tree is a path, which maximizes the Wiener index (Dobrynin et al., 2001). Appendix A.3 derives the tree estimates and quantitative growth-rate comparisons. With unequal error rates, semantic similaritybased querying (Wang et al., 2013) also depends on which pairs are answered wrongly (Section 6).

## 5.2 Several Clusters

Now records belong to K true clusters of sizes $s _ { 1 } , \ldots , s _ { K }$ , with $n \ = \ \textstyle \sum _ { i } s _ { i }$ records (here n counts records, not pairwise facts). All $\binom { n } { 2 }$ pairs are scanned in a uniformly random order. Answers are flipped independently: a within-cluster pair is wrongly answered “diferent” with probability $\varepsilon _ { w } ,$ and a cross-cluster pair “same” with probability $\varepsilon _ { x }$ . Write $R _ { U }$ for the expected number of wrong final pair decisions, counting both false merges and false splits.

To isolate a false merge, consider only two clusters of sizes $s _ { A } , s _ { B }$ . Fix a cross pair and make its potential answer “same”, with all other answers correct. Let $g ( s _ { A } , s _ { B } )$ be its mean damage over random orders, counting a skipped wrong answer as zero, and write $g ( s ) = g ( s , s )$ . Accepting the error can also block later correct within-cluster merges. Thus g includes both wrong cross-cluster merges and possible splits within these two clusters.

Theorem 12 (Exact first-order decomposition). With $\varepsilon = \operatorname* { m a x } ( \varepsilon _ { w } , \varepsilon _ { x } )$

$$
\begin{array} { l } { { \displaystyle { \cal R } _ { U } = \varepsilon _ { w } \sum _ { i = 1 } ^ { K } \mathbb { E } W ( \mathrm { M S T } _ { s _ { i } } ) } } \\ { { \displaystyle ~ + \varepsilon _ { x } \sum _ { i < j } s _ { i } s _ { j } g ( s _ { i } , s _ { j } ) + { \cal O } ( \varepsilon ^ { 2 } ) } . } \end{array}
$$

In particular the damage of a false merge does not depend on K or on the other clusters.

The first term sums false-split risks within the true clusters, applying the Wiener law to each. The second sums false-merge risks: clusters $i , j$ have s<sub>i</sub>s<sub>j</sub> cross pairs, each contributing mean damage $g ( s _ { i } , s _ { j } )$ Correct cross-cluster answers confine a single error to its incident clusters. Simultaneous errors can involve more clusters, so this locality underlies the first-order decomposition.

The following bounds on $g ( s )$ count skipped errors as zero.

Proposition 13 (Size of one false merge). Let q be the event that the false pair is asked. For $s \ \geq \ 2 ,$ $\begin{array} { r } { \frac { 1 } { 2 s - 1 } \leq \mathbb { P } ( q ) \leq \frac { 1 2 . 4 } { s + 1 } + \frac { 1 } { 2 } \big ( \frac { 9 } { 1 6 } \big ) ^ { s - 1 } } \end{array}$ and $\begin{array} { r } { \frac { s - 1 } { 2 s - 1 } \le g ( s ) \le } \end{array}$ 13 s. Under standard local-limit facts for Erdős–Rényi graphs, $s \mathbb { P } ( q ) \to C _ { q } \approx 1 . 4 2 8 6$ (Proposition 23).

Exact computation gives $g ( 2 ) \ = \ 2 9 / 1 8$ and $g ( 3 ) =$ 313913/150150 ≈ 2.091 (Table 5). For comparison, the pivot pipeline processes records one at a time. It compares each new record with the first record of each existing block, in block-creation order, until “same” lets it join that block; if all answers are “diferent”, it creates a new block (Vesdapunt et al., 2014). Its first-order risk is given by Proposition 14.

Proposition 14 (Pivot pipeline). For K clusters of size s, with items arriving in a uniformly random or-

der,

$$
R _ { \mathrm { p i v } } = \varepsilon _ { w } K ( s - 1 ) ^ { 2 } + \varepsilon _ { x } { \binom { K } { 2 } } s ( 2 s - 1 ) + O ( \varepsilon ^ { 2 } ) .
$$

The coeficient of $\varepsilon _ { x }$ in Proposition 14, divided by the ${ \binom { K } { 2 } } s ^ { 2 }$ cross pairs, gives mean damage $( 2 s - 1 ) / s$ per potential wrong cross-pair answer. The random-order mean is $g ( s )$ , giving a loss ratio $g ( s ) s / ( 2 s - 1 )$

Conjecture 15 (Contamination law). $\begin{array} { r l } { g ( s ) } & { { } = } \end{array}$ $\Theta ( s ^ { 2 \bar { / } 3 } )$ . Consequently, for $K \ge 2$ clusters of size $s , R _ { U } = \Theta ( \varepsilon _ { w } K s ^ { 7 / 3 } ) + \Theta ( \varepsilon _ { x } K ^ { 2 } s ^ { 8 / 3 } )$ at first order.

Simulation supports $s ^ { 2 / 3 }$ growth but does not determine the asymptotic constant (Figure 2, lower panel $\mathrm { C } ;$ Table 6). A proof would need a scaling limit describing the constrained merging process in the critical window of the within-cluster random graphs.

In entity resolution, the noiseless query tree determines one-cluster first-order risk; with independent errors at a common rate, pivot stars minimize this risk among greedy query orders. With several entities, a single error remains confined to the one or two entities involved, yielding separate first-order contributions from within-entity and cross-entity errors.

## 6 EXPERIMENTAL VALIDATION

We check the formulas in simulation, compare query orders at higher error rates allowing multiple wrong answers, and evaluate final losses using LLM answers on Cora. Loss counts wrong final decisions, including inferred ones. In record matching, false splits separate records of the same entity, while false merges group records of diferent entities. Amplification is the final wrong-pair count divided by the raw baseline’s wronganswer count. All single-error averages count skipped errors as zero.

Figure 2 connects the propagation examples with numerical checks of the preceding laws. (A) contrasts 2-CNF conservation with shared-premise Horn amplification. In (B), splitting eight records into $1 + 7$ or 4+4 produces 7 or 16 wrong pairs. Its lower panel compares first-order amplification, equal to mean noiselesstree distance under independent, equal-rate errors. In (C), one false merge can cause both false merges and false splits within the two involved entities; the lower panel compares $g ( s )$ with pivot’s $( 2 s - 1 ) / s$ The $2 . 3 8 m ^ { 1 / 3 }$ and $0 . 9 s ^ { 2 / 3 }$ curves are guides; the finite-size slope ≈ 0.43 in (B) does not determine the asymptotic exponent, and (C) does not establish its asymptotic constant.

Checking the analytical predictions. Enumeration and direct decoding test conservation, Horn damage and false-merge locality independently of the formulas. The conservation estimates are within 2.66 Monte Carlo standard errors (MCSE) of their predictions; the other tabulated estimates are within 1.62 MCSE. $\mathrm { A t } ~ \varepsilon = 1 0 ^ { - 3 }$ , dividing the full decoder’s mean false-merge (FP) and false-split (FN) counts by ε also gives estimates close to the first-order coeficients. Appendix B gives the full comparison (Table 4), simulation settings and statistical details.

Table 2: Fraction of wrongly separated pairs on one cluster of $m = 1 0 0$ records when each answer is flipped to $\mathrm { \mathrm { \Omega ^ { 6 6 } d i f f e r e n t { \vec { \tau } } } }$ with probability δ. Means ±1 Monte Carlo standard error over 200 runs; Random/Pivot is the ratio of the mean losses.
<table><tr><td>δ</td><td>Pivot</td><td>Random</td><td>Random/Pivot</td></tr><tr><td>0.01</td><td> $0 . 0 1 9 \pm 0 . 0 0 2$ </td><td> $0 . 0 8 2 \pm 0 . 0 0 8$ </td><td>4.2</td></tr><tr><td>0.02</td><td> $0 . 0 3 9 \pm 0 . 0 0 2$ </td><td> $0 . 1 7 0 \pm 0 . 0 1 2$ </td><td>4.3</td></tr><tr><td>0.05</td><td> $0 . 0 9 4 \pm 0 . 0 0 3$ </td><td> $0 . 3 2 0 \pm 0 . 0 1 0$ </td><td>3.4</td></tr><tr><td>0.10</td><td> $0 . 1 7 4 \pm 0 . 0 0 4$ </td><td> $0 . 4 5 8 \pm 0 . 0 0 8$ </td><td>2.6</td></tr><tr><td>0.20</td><td> $0 . 3 3 5 \pm 0 . 0 0 4$ </td><td> $0 . 6 0 4 \pm 0 . 0 0 6$ </td><td>1.8</td></tr></table>

Table 3: Amplification on Qwen2.5-1.5B answers for Cora: final wrong pairs per wrong answer. Wrong: answer-error rate. Pivot/random use 200 orders in rows 1 and 4; rows 2–3 use 30 matched orders before and after uniformly redistributing errors within each entity, preserving its error count.
<table><tr><td>Setting</td><td>Wrong</td><td></td><td>Pivot Random</td><td>Sim.-first</td></tr><tr><td>One entity</td><td>6.6%</td><td>1.23</td><td>2.31</td><td>0.85</td></tr><tr><td>One entity, 30 orders</td><td>6.6%</td><td>1.37</td><td>2.19</td><td>0.85</td></tr><tr><td>Shuffled, 30 orders</td><td>6.6%</td><td>1.73</td><td>3.55</td><td>3.48</td></tr><tr><td>Multi-entity blocks</td><td>6.3%</td><td>1.14</td><td>1.46</td><td>1.20</td></tr></table>

Higher error rates. We also test the full decoder where multiple errors may interact. One entity has $m \ = \ 1 0 0$ records, so every pair’s correct answer is “same”. Each answer is independently flipped to “different” at rate $\delta = 1 \% { - 2 0 \% }$ , with 200 runs per configuration. Table 2 reports the fraction of pairs wrongly separated: at a 10% answer-error rate, this fraction averages 17.4% for pivot and 45.8% for random order. Across the tested rates, random-order mean loss is 1.8– 4.3 times the pivot value; at $m = 2 0 0$ the ratios are 1.9–5.4 (Table 7). Pivot’s advantage therefore persists in these settings. At higher rates, the first-order sum overestimates random-order loss: several errors may afect the same pairs, which are counted only once.

Real LLM answers. In Cora (Hasso Plattner Institute, n.d.), each entity comprises citation records of one paper. We use 62,423 within-entity pairs from 23 entities with at least 20 records. Qwen2.5-1.5B (base) (Qwen Team, 2024) answers 6.6% wrongly. All orders

(A) Implication vs Horn

## A Logical constraints

![](images/8958402857e7298cbc13acd8f43eceaceac293c66fb490418f7253c35c4af3aa.jpg)

## B Query order

(C) Several clusters  
![](images/c722cfc593ede961642af4f13472b2367eeb6b2c983066169465cc6b0ab469f9.jpg)

## C Several entities

![](images/d3957b1ce0bf538b4f8539e5d75ec534a9b64dc73fb846230d92c17078ac907f.jpg)

![](images/90d525be3495ccb03f28929a141264a15fd3acd003351a50fcde744f6193578e.jpg)

![](images/9b9ca78c23ae132beb368836c6ffcbd4a274ad0ea2c396685b2b02dcd4cbf3b3.jpg)  
Shared-premise rules, k  
Exact formula

![](images/d3a68a35db3bf0ed045826245b83dc5ff4cb1db7ad2206d0a3770e0bdd0a5f9b.jpg)  
Monte Carlo ±1 SE  
Asymptote / conjecture

![](images/5ff4f20854b3c38c5d02e640300e87d6310c74eb9cb53abddf8f49d2415dc437.jpg)  
Exact enumeration

Figure 2: Single-answer error propagation (top) and numerical checks (bottom): (A) logical constraints, (B) within-entity query order, (C) cross-entity false merges. Horn example: k = s = 2, a, c<sub>i</sub> false, b<sub>i</sub> true. Shading marks afected true clusters, not the final partition. Thick solid: exact formulas; thin connectors: visual guides; dashed: asymptotes (A), conjecture (C); dotted: fitted guide (B). Open markers: simulations ±1 standard error; diamonds: exact enumeration.

reuse cached answers; similarity-first follows decreasing textual similarity.

Pivot loses less than random order, while similarityfirst has the lowest point estimate (Table 3). Pooled answer-error rates on noiseless query-tree edges are 2.2% for similarity-first, 8.4% for pivot and 8.6% for random order, linking loss to both tree shape and edge accuracy. Pivot beats random order at all tested thresholds (approximately 1%–20% answer-error rates; Table 8).

Redistributing each entity’s errors uniformly, with error counts fixed and query orders matched, raises all three losses and removes similarity-first’s advantage over pivot (Table 3, rows 2–3). Thus error locations matter.

In six blocks of three or four similar entities, both false merges and false splits can occur; pivot has the lowest pooled loss. Appendix C gives the prompting protocol, threshold sweep and detailed results for these settings.

## 7 DISCUSSION

We analyzed error propagation in greedy, constraintbased decoding. Under uniformly random query orders, 2-CNF constraints do not increase first-order risk over raw answers, while shared-premise Horn rules can amplify errors. In record matching, one-entity first-order risk is determined by pairwise distances on the noiseless query tree. Under independent, equalrate errors, pivot minimizes this risk among greedy orders. With one wrong answer, false merges afect only two entities. Simulations support the formulas and show pivot’s advantage over random order at the tested higher error rates. The Cora results show that tree shape and error locations both matter: uniformly redistributing errors removes similarity-first’s advantage over pivot. Future work could study interacting errors, especially false merges, across more models and datasets. The general Horn false-positive damage bound and false-merge growth law remain conjectures.

## AI Use Statement

The research ideas, mathematical claims, experimental design, and interpretation of the results were developed by the authors. During the preparation of this paper, ChatGPT and Codex assisted with editing the prose, restructuring and improving portions of the code, and organizing data and relevant literature. We also consulted these tools to identify proof arguments that could benefit from clearer explanations. All AI assisted work was independently reviewed and verified by the authors, including the proofs, code, data, references, and changes to the manuscript. The authors accept full responsibility for the final paper.

## References

Louigi Addario-Berry, Nicolas Broutin, and Bruce Reed. Critical random graphs and the structure of a minimum spanning tree. Random Structures & Algorithms, 35(3):323–347, 2009. doi: 10.1002/rsa. 20241.

Louigi Addario-Berry, Nicolas Broutin, Christina Goldschmidt, and Grégory Miermont. The scaling limit of the minimum spanning tree of the complete graph. The Annals of Probability, 45(5):3075–3144, 2017. doi: 10.1214/16-AOP1132.

Sheshansh Agrawal, Thien Hang Nguyen, and Douwe Kiela. BlitzRank: Principled zero-shot ranking agents with tournament graphs. In Forty-third International Conference on Machine Learning (ICML 2026), 2026. arXiv preprint arXiv:2602.05448.

Michael Aizenman, Almut Burchard, Charles M. Newman, and David B. Wilson. Scaling limits for minimal and random spanning trees in two dimensions. Random Structures & Algorithms, 15(3–4):319–367, 1999. doi: 10.1002/(SICI)1098-2418(199910/12)15: 3/4<319::AID-RSA8>3.0.CO;2-G.

Laurent Alonso, Philippe Chassaing, Florent Gillet, Svante Janson, Edward M. Reingold, and René Schott. Quicksort with unreliable comparisons: A probabilistic analysis. Combinatorics, Probability and Computing, 13(4–5):419–449, 2004. doi: 10. 1017/S0963548304006297.

Sayantan Banerjee and Weining Shen. Graph signal denoising using t-shrinkage priors. Journal of Statistical Planning and Inference, 219:279–305, 2022. doi: 10.1016/j.jspi.2021.12.012.

Beijing Academy of Artificial Intelligence. bgesmall-en-v1.5: Model card, 2023. URL https:// huggingface.co/BAAI/bge-small-en-v1.5. MIT license; accessed October 1, 2026.

Mark Braverman and Elchanan Mossel. Noisy sorting without resampling. In Proceedings of the Nine-

teenth Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), pages 268–276, 2008.

Nicolas Broutin and Jean-François Marckert. Convex minorant trees associated with brownian paths and the continuum limit of the minimum spanning tree. arXiv preprint arXiv:2307.12260, 2023. URL https: //arxiv.org/abs/2307.12260v1.

Xingguang Chen, Yimin Shi, and Xiaokui Xiao. Generalized entity matching with adaptivity via large language models. Proceedings of the ACM on Management of Data, 4(3):189:1–189:25, 2026. doi: 10.1145/3802066. URL https://dl.acm.org/doi/ 10.1145/3802066.

Christopher De Sa, Ihab F. Ilyas, Benny Kimelfeld, Christopher Ré, and Theodoros Rekatsinas. A formal framework for probabilistic unclean databases. In 22nd International Conference on Database Theory (ICDT 2019), volume 127 of Leibniz International Proceedings in Informatics (LIPIcs), pages 6:1–6:18. Schloss Dagstuhl–Leibniz-Zentrum für Informatik, 2019. doi: 10.4230/LIPIcs.ICDT.2019.6.

Andrey A. Dobrynin, Roger Entringer, and Ivan Gutman. Wiener index of trees: Theory and applications. Acta Applicandae Mathematicae, 66(3):211– 249, 2001. doi: 10.1023/A:1010767517079.

Uriel Feige, Prabhakar Raghavan, David Peleg, and Eli Upfal. Computing with noisy information. SIAM Journal on Computing, 23(5):1001–1018, 1994. doi: 10.1137/S0097539791195877.

Donatella Firmani, Barna Saha, and Divesh Srivastava. Online entity resolution using an oracle. Proceedings of the VLDB Endowment, 9(5):384–395, 2016. doi: 10.14778/2876473.2876474.

Dylan J. Foster, Daniel Reichman, and Karthik Sridharan. Inference in sparse graphs with pairwise measurements and side information. In Proceedings of the 21st International Conference on Artificial Intelligence and Statistics, volume 84 of Proceedings of Machine Learning Research, pages 1810–1818. PMLR, 2018.

Sainyam Galhotra, Donatella Firmani, Barna Saha, and Divesh Srivastava. Robust entity resolution using random graphs. In Proceedings of the 2018 International Conference on Management of Data (SIGMOD), pages 3–18. ACM, 2018. doi: 10.1145/ 3183713.3183755.

Amir Gilad, Aviram Imber, and Benny Kimelfeld. The consistency of probabilistic databases with independent cells. In 26th International Conference on Database Theory (ICDT 2023), volume 255 of Leibniz International Proceedings in Informatics (LIPIcs), pages 22:1–22:19. Schloss Dagstuhl–

Leibniz-Zentrum für Informatik, 2023. doi: 10.4230/ LIPIcs.ICDT.2023.22.

Amir Globerson, Tim Roughgarden, David Sontag, and Cafer Yildirim. How hard is inference for structured prediction? In Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pages 2181–2190. PMLR, 2015.

Anja Gruenheid, Besmira Nushi, Tim Kraska, Wolfgang Gatterbauer, and Donald Kossmann. Faulttolerant entity resolution with the crowd. arXiv preprint arXiv:1512.00537, 2015.

Zhengtao Gui, Sheng Yang, and Zhouxing Shi. Evaluating the robustness of proof autoformalization in Lean 4. arXiv preprint arXiv:2606.14867, 2026. doi: 10.48550/arXiv.2606.14867. URL https://arxiv. org/abs/2606.14867.

Chaohui Guo, Michel C. A. Klein, and Zhisheng Huang. CaRL-EM: Cost-aware reinforcement learning for entity matching with LLMs. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 27281–27293. Association for Computational Linguistics, 2026. doi: 10.18653/v1/2026. acl-long.1258. URL https://aclanthology.org/ 2026.acl-long.1258/.

Hasso Plattner Institute. CORA dataset: Bibliographic records and duplicate pairs. Repeatability datasets, n.d. URL https://hpi.de/en/ naumann/projects/repeatability/datasets/ cora-dataset.html. Accessed October 1, 2026.

Alireza Heidari, Ihab F. Ilyas, and Theodoros Rekatsinas. Approximate inference in structured instances with noisy categorical observations. In Proceedings of The 35th Uncertainty in Artificial Intelligence Conference, volume 115 of Proceedings of Machine Learning Research, pages 412–421. PMLR, 2020. URL https://proceedings.mlr.press/ v115/heidari20a.html.

Sujai Hiremath, Dominik Janzing, Philipp Michael Faller, Patrick Blöbaum, Elke Kirschbaum, Shiva Prasad Kasiviswanathan, and Kyra Gan. From Guess2Graph: When and how can unreliable experts safely boost causal discovery in finite samples? In Proceedings of The 29th International Conference on Artificial Intelligence and Statistics, volume 300 of Proceedings of Machine Learning Research, pages 1459–1467. PMLR, 2026. URL https://proceedings.mlr.press/v300/ hiremath26a.html.

Omer Israeli and Yuval Peled. Noise sensitivity of the minimum spanning tree of the complete graph.

Combinatorics, Probability and Computing, 33(6): 708–723, 2024. doi: 10.1017/S0963548324000129.

Shuyue Jin, Zhengtao Gui, Jie Hu, and Yu Chen. Community detection and network reconstruction with dependent connectivity from rich but noisy network data. Australian & New Zealand Journal of Statistics, 68(1):e70026, 2026. doi: 10.1111/anzs.70026.

Pascal Kesseli, Peter O’Hearn, and Ricardo Silveira Cabral. Logic.py: Bridging the gap between LLMs and constraint solvers. In Advances in Neural Information Processing Systems, volume 38, pages 62505–62525. Curran Associates, Inc., 2025. doi: 10.52202/085713-2093.

Emre Kıcıman, Robert Ness, Amit Sharma, and Chenhao Tan. Causal reasoning and large language models: Opening a new frontier for causality. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum? id=mqoxLkX210.

Joseph B. Kruskal, Jr. On the shortest spanning subtree of a graph and the traveling salesman problem. Proceedings of the American Mathematical Society, 7(1):48–50, 1956. doi: 10.1090/ S0002-9939-1956-0078686-7.

Hanbyul Lee, Kevin Bello, and Jean Honorio. On the fundamental limits of exact inference in structured prediction. In 2022 IEEE International Symposium on Information Theory (ISIT), pages 3174–3179. IEEE, 2022. doi: 10.1109/ISIT50566.2022.9834614.

Stephanie Long, Alexandre Piché, Valentina Zantedeschi, Tibor Schuster, and Alexandre Drouin. Causal discovery with language models as imperfect experts. In ICML 2023 Workshop on Structured Probabilistic Inference & Generative Modeling, 2023. arXiv preprint arXiv:2307.02390.

Ruiqi Lyu, Alistair Turcan, Martin Jinye Zhang, and Bryan Wilder. Improving constraint-based discovery with robust propagation and LLM priors. In Conference on Neural Information Processing Systems (NeurIPS 2026), 2026. URL https:// neurips.cc/virtual/2026/poster/151902. Accepted; proceedings forthcoming.

Konstantin Makarychev, Yury Makarychev, and Aravindan Vijayaraghavan. Correlation clustering with noisy partial information. In Proceedings of The 28th Conference on Learning Theory, volume 40 of Proceedings of Machine Learning Research, pages 1321–1342. PMLR, 2015.

Arya Mazumdar and Barna Saha. Clustering with noisy queries. In Advances in Neural Information Processing Systems, volume 30, 2017a.

Arya Mazumdar and Barna Saha. A theoretical analysis of first heuristics of crowdsourced entity res-

olution. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 31, pages 970–976, 2017b. doi: 10.1609/aaai.v31i1.10636.

Christopher Meek. Causal inference and causal explanation with background knowledge. In Proceedings of the Eleventh Conference on Uncertainty in Artificial Intelligence, pages 403–410, 1995.

Eric Mitchell, Joseph Noh, Siyan Li, Will Armstrong, Ananth Agarwal, Patrick Liu, Chelsea Finn, and Christopher D. Manning. Enhancing selfconsistency and performance of pre-trained language models through natural language inference. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 1754–1768. Association for Computational Linguistics, 2022. doi: 10.18653/v1/2022.emnlp-main.115.

Ralph Peeters, Aaron Steiner, and Christian Bizer. Entity matching using large language models. In Proceedings of the 28th International Conference on Extending Database Technology, pages 529–541. OpenProceedings.org, 2025. doi: 10.48786/edbt. 2025.42. URL https://www.openproceedings. org/2025/conf/edbt/paper-81.pdf.

Qwen Team. Qwen2.5-1.5B: Base model card, 2024. URL https://huggingface.co/Qwen/Qwen2.5-1. 5B. Apache 2.0; model revision recorded in the experimental metadata.

Igal Sason and Shlomo Shamai. Performance analysis of linear codes under maximum-likelihood decoding: A tutorial. Foundations and Trends in Communications and Information Theory, 3(1–2):1–222, 2006. doi: 10.1561/0100000009.

Remco van der Hofstad. Random Graphs and Complex Networks. Volume 1. Number 43 in Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, Cambridge, 2017. doi: 10.1017/9781316779422.

Norases Vesdapunt, Kedar Bellare, and Nilesh Dalvi. Crowdsourcing algorithms for entity resolution. Proceedings of the VLDB Endowment, 7(12):1071– 1082, 2014. doi: 10.14778/2732977.2732982.

Jiannan Wang, Guoliang Li, Tim Kraska, Michael J. Franklin, and Jianhua Feng. Leveraging transitive relations for crowdsourced joins. In Proceedings of the 2013 ACM SIGMOD International Conference on Management of Data, pages 229–240. ACM, 2013. doi: 10.1145/2463676.2465280.

Tianshu Wang, Xiaoyang Chen, Hongyu Lin, Xuanang Chen, Xianpei Han, Le Sun, Hao Wang, and Zhenyu Zeng. Match, compare, or select? an investigation of large language models for entity matching. In Proceedings of the 31st International Conference on Computational Linguistics,

pages 96–109. Association for Computational Linguistics, 2025. URL https://aclanthology.org/ 2025.coling-main.8/.

Harry Wiener. Structural determination of parafin boiling points. Journal of the American Chemical Society, 69(1):17–20, 1947. doi: 10.1021/ ja01193a005.

Yi Xu, Laura Ruis, Tim Rocktäschel, and Robert Kirk. Investigating non-transitivity in LLM-as-a-judge. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 69583–69612. PMLR, 2025. URL https://proceedings.mlr. press/v267/xu25w.html.

Yifan Zeng, Ojas Tendolkar, Raymond Baartmans, Qingyun Wu, Lizhong Chen, and Huazheng Wang. LLM-RankFusion: Mitigating intrinsic inconsistency in LLM-based ranking. Transactions on Machine Learning Research, 2026. URL https:// openreview.net/forum?id=VUY0j74Yes.

# How Much Does One Wrong Answer Cost? Supplementary Materials

## A PROOFS

## A.1 Setting and reduction

We first express small-noise risk in terms of single-error damage, then prove the results for logical constraints, entity resolution and sorting. Numerical checks are in Section B.

A code is a finite nonempty set ${ \mathcal { C } } \subseteq \{ 0 , 1 \} ^ { n }$ . Its elements are codewords, and its n positions are coordinates (the binary queries). The truth is a codeword $x ^ { \star } \in \mathcal { C }$ . The answers are $y = x ^ { \star } \oplus Z ,$ , where $Z _ { 1 } , \ldots , Z _ { n }$ are independent and $Z _ { i } \sim \operatorname { B e r n } ( \varepsilon )$ with $\varepsilon \in [ 0 , \frac { 1 } { 2 } ]$ . We write $e _ { a }$ for the a-th unit vector of $\{ 0 , 1 \} ^ { n }$ and $d _ { \mathrm { H } }$ for the Hamming distance. A decoder maps the answers y to an estimate, possibly using a random seed π independent of $Z$ to choose the query order or break ties; its output is xˆ. Its risk is $R ( \varepsilon ) = \mathbb { E } d _ { \mathrm { H } } ( \hat { x } , x ^ { \star } )$ , where the expectation is over Z and π. We use three decoders.

• The raw decoder outputs ${ \hat { x } } = y$ . Hence $R _ { \mathrm { r a w } } ( \varepsilon ) = \varepsilon n \ \mathrm { e x a c t l y }$

• The ML decoder outputs a codeword at minimum Hamming distance from y, with ties broken uniformly at random. For $\begin{array} { r } { \varepsilon < { \frac { 1 } { 2 } } } \end{array}$ this is maximum-likelihood decoding.

• The greedy decoder $G _ { \pi }$ processes the coordinates in an order π. It starts from the empty set $L _ { 0 } = \emptyset$ of accepted answers. At step t, with $i = \pi ( t )$ , it skips i if all codewords consistent with $L _ { t - 1 }$ have the same value at i (the query is implied); otherwise it accepts the answer and sets $L _ { t } = L _ { t - 1 } \cup \{ ( i , y _ { i } ) \}$ . It outputs the unique codeword consistent with $L _ { n }$ . Lemma 16 in Section A.2 shows that this codeword exists and is unique. Unless stated otherwise, π is uniform over the n! orders and independent of Z.

For a codeword x and a coordinate a we define

$$
B _ { a } ( x ) = \{ b \neq a : \ x \oplus e _ { a } \notin \mathcal C , \ x \oplus e _ { a } \oplus e _ { b } \in \mathcal C \} , \qquad m _ { a } ( x ) = | B _ { a } ( x ) | .
$$

If $m _ { a } ( x ) \geq 1$ , an error at a leaves the observed word at distance 1 from x and from $m _ { a } ( x )$ alternative codewords. Each alternative difers from the original codeword in two positions and is therefore at distance 2 from x. For a decoder with seed π, the single-error damage at a is the expected loss when the only wrong answer is at a:

$$
D ( a ) = \mathbb { E } _ { \pi } d _ { \mathrm { H } } \big ( \hat { x } ( x ^ { \star } \oplus e _ { a } , \pi ) , x ^ { \star } \big ) .
$$

Probabilistic unclean databases (De Sa et al., 2019) formalize a noisy-channel model of database cleaning and seek the most likely intended database. Their Theorems 7 and 8 relate this optimum to weighted minimum repairs under suficiently high constraint weights and specified prior/channel assumptions; the update-repair result also requires attribute independence. Union-bound analyses of ML decoding are surveyed by Sason and Shamai (2006); part (i) of Lemma 1 gives an elementary first-order expansion for our finite code setting. Part (ii) applies the same reduction to greedy decoding.

Proof of Lemma $^ { 1 , }$ remainder bound. We show that every decoder that is exact on the noiseless input, that is, $\hat { x } ( x ^ { \star } , \pi ) = x ^ { \star }$ for every seed, satisfies for all $\varepsilon \in [ 0 , { \frac { 1 } { 2 } } ]$

$$
\Big | R ( \varepsilon ) - \varepsilon \sum _ { a = 1 } ^ { n } D ( a ) \Big | \ \le \ \varepsilon ^ { 2 } n ( n - 1 ) \Big ( \operatorname* { m a x } _ { a } D ( a ) + \frac { n } { 2 } \Big ) \ \le \ \frac { 3 } { 2 } \varepsilon ^ { 2 } n ^ { 2 } ( n - 1 ) .\tag{1}
$$

Condition on the error set $S = \{ i : ~ Z _ { i } = 1 \}$ . If $S = \emptyset .$ , then $y = x ^ { \star }$ and the loss is 0. The event $S = \{ a \}$ has probability $\varepsilon ( 1 - \varepsilon ) ^ { n - 1 }$ . On this event the conditional expected loss is $D ( a )$ , because π is independent of Z. If $| S | \ge 2$ , the loss lies in [0, n], and a union bound over pairs of coordinates gives $\mathbb { P } ( | S | \ge 2 ) \le \binom { \bar { n } } { 2 } \varepsilon ^ { 2 }$ . Therefore

$$
R ( \varepsilon ) - \varepsilon \sum _ { a } D ( a ) \ = \ - \varepsilon \left[ 1 - ( 1 - \varepsilon ) ^ { n - 1 } \right] \sum _ { a } D ( a ) \ + \ r , \qquad 0 \leq r \leq \frac { 1 } { 2 } n ^ { 2 } ( n - 1 ) \varepsilon ^ { 2 } .
$$

Bernoulli’s inequality gives $0 \leq 1 - ( 1 - \varepsilon ) ^ { n - 1 } \leq ( n - 1 ) \varepsilon$ . So the first term lies in $[ - \varepsilon ^ { 2 } n ( n - 1 ) \operatorname* { m a x } _ { a } D ( a ) , 0 ]$ and the first inequality in (1) follows. The second inequality follows from $D ( a ) \leq n$ . For ML, part (i) below gives $D ( a ) < 2 .$ , which yields the tighter bound $\varepsilon ^ { 2 } n ( n - 1 ) ( 2 + n / 2 )$

The same argument applies when coordinate i has its own error rate $\varepsilon _ { i } \leq \varepsilon$ . Then $\begin{array} { r } { \mathbb { P } ( S = \{ a \} ) = \varepsilon _ { a } \prod _ { b \neq a } ( 1 - \varepsilon _ { b } ) } \end{array}$ $\begin{array} { r } { 0 \leq 1 - \prod _ { b \neq a } ( 1 - \varepsilon _ { b } ) \leq ( n - 1 ) \varepsilon } \end{array}$ , and $\mathbb { P } ( | S | \ge 2 ) \le \binom { n } { 2 } \varepsilon ^ { 2 }$ . Hence (1) holds with $\varepsilon \sum _ { a } D ( a )$ replaced by $\textstyle \sum _ { a } \varepsilon _ { a } D ( a )$ □

Proof of Lemma $1 ( i )$ . We show that the ML decoder has $D _ { \mathrm { M L } } ( a ) = 1 \mathrm { i f } x ^ { \star } \oplus e _ { a } \in \mathcal { C }$ and $D _ { \mathrm { M L } } ( a ) = 2 m _ { a } ( x ^ { \star } ) / ( 1 +$ $m _ { a } ( x ^ { \star } ) )$ otherwise, so that

$$
R _ { \mathrm { M L } } ( \varepsilon ) = \varepsilon \Big [ N _ { 1 } ( x ^ { \star } ) + \sum _ { a : x ^ { \star } \oplus e _ { a } \notin \mathcal { C } } \frac { 2 m _ { a } ( x ^ { \star } ) } { 1 + m _ { a } ( x ^ { \star } ) } \Big ] + O ( \varepsilon ^ { 2 } ) .
$$

Here $N _ { 1 } ( x ^ { \star } ) = | \{ a : \ x ^ { \star } \oplus e _ { a } \in { \mathcal { C } } \} |$ counts the codewords at Hamming distance one from the truth. The ML decoder is exact on the noiseless input, because $x ^ { \star }$ is the only codeword at distance 0 from $y = x ^ { \star }$ . Let $\boldsymbol { y } = \boldsymbol { x } ^ { \star } \oplus \boldsymbol { e } _ { a }$ $\operatorname { I f } y \in { \mathcal { C } }$ , the decoder returns y, which is at distance 0, and the loss is $d _ { \mathrm { H } } ( y , x ^ { \star } ) = 1$ . Otherwise $d _ { \mathrm { H } } ( y , \mathcal { C } ) = 1$ . The codewords at distance 1 from y are $x ^ { \star } = y \oplus e _ { a }$ and the codewords of the form y ⊕ $\boldsymbol { e } _ { b } = \boldsymbol { x } ^ { \star } \oplus \boldsymbol { e } _ { a } \oplus \boldsymbol { e } _ { b }$ with $b \neq a$ The latter are exactly the $m _ { a } ( x ^ { \star } )$ words indexed by $B _ { a } ( x ^ { \star } )$ , and each of them is at distance 2 from $x ^ { \star }$ . Uniform tie-breaking among these $1 + m _ { a } ( x ^ { \star } )$ words gives the expected loss $2 m _ { a } ( x ^ { \star } ) / ( 1 + m _ { a } ( x ^ { \star } ) )$ . The displayed form of $R _ { \mathrm { M L } }$ follows from (1). If $m _ { a } ( x ^ { \star } ) \leq 1$ for every $^ { a , }$ as on 2-CNF theories (see the proof of Proposition 5), the bracket equals $N _ { 1 } ( x ^ { \star } ) + \# \{ a : \ m _ { a } ( x ^ { \star } ) = 1 \}$ }. In words, each codeword at distance 1 counts 1, and each two-way tie counts $2 \times { \frac { 1 } { 2 } }$ □

Proof of Lemma $1 ( i i )$ . We show that $\begin{array} { r } { R _ { \mathrm { g r } } ( \varepsilon ) = \varepsilon \sum _ { a } \mathbb { E } _ { \pi } D _ { \pi } ( a ) + O ( \varepsilon ^ { 2 } ) } \end{array}$ , with the remainder bound (1), where $D _ { \pi } ( a ) = d _ { \mathrm { H } } ( G _ { \pi } ( x ^ { \star } \oplus e _ { a } ) , x ^ { \star } )$ is the damage of one wrong answer at a, whether this answer is accepted or not. The output of $G _ { \pi }$ is a function of y and of the order π, and π is independent of $Z .$ . The decoder is exact on the noiseless input. Indeed, if $y = x ^ { \star }$ , every accepted pair $( i , y _ { i } )$ agrees with $x ^ { \star } ;$ so $x ^ { \star }$ is consistent with $L _ { n }$ , and it is therefore the unique codeword consistent with $L _ { n }$ (Lemma 16). So (1) applies with $D ( a ) = \mathbb { E } _ { \pi } D _ { \pi } ( a )$ . Let $\begin{array} { r } { \bar { D } = \frac { 1 } { n } \sum _ { a } \mathbb { E } _ { \pi } D _ { \pi } ( a ) } \end{array}$ be the mean single-error damage over positions and orders. Then $R _ { \mathrm { g r } } ( \varepsilon ) = \varepsilon n \bar { D } + O ( \varepsilon ^ { 2 } )$ Since $R _ { \mathrm { r a w } } ( \varepsilon ) = \varepsilon n$ , the amplification of the greedy decoder over the raw decoder is $\bar { D }$ □

## A.2 Logical constraints

Let Φ be a satisfiable propositional formula, that is, a conjunction of clauses, over a finite set V of atoms with $| V | = n$ . The code is the set ${ \mathcal { C } } = \operatorname { M o d } ( \Phi ) \subseteq \{ 0 , 1 \} ^ { V }$ of models (satisfying assignments) of Φ. Every nonempty code ${ \mathcal { C } } \subseteq \{ 0 , 1 \} ^ { n }$ has this form: take for Φ the conjunction, over the words w $\notin { \mathcal { C } } ,$ of the clause that is false exactly at w. The truth is a model $\sigma ^ { \star } \in \mathcal { C }$ . For an atom x, the true literal $t _ { x }$ is x if $\sigma ^ { \star } ( x ) = 1$ and ¬x otherwise; the false literal $\bar { t } _ { x }$ is the other literal on x. The answer on x is $y _ { x } = \sigma ^ { \star } ( x ) \oplus Z _ { x } ,$ where the $Z _ { x }$ are i.i.d. Bern(ε). The claim literal of x is x if $y _ { x } = 1$ and ¬x if $y _ { x } = 0$ . A set L of literals records accepted answers; requiring all its members to hold is equivalent to taking their conjunction. We write $L \Vdash _ { \Phi } m$ if Φ $\land L \models m$ , that is, if every model of Φ that satisfies all literals of L also satisfies the literal m.

For an order π of V, the greedy decoder $G _ { \pi }$ (skip if implied, accept otherwise) starts with $L _ { 0 } = \emptyset$ . At step $t ,$ with $x = \pi ( t )$ , it sets $L _ { t } ~ = ~ L _ { t - 1 }$ if $L _ { t - 1 } \ \models _ { \Phi } \ x$ or $L _ { t - 1 } \ \models _ { \Phi } \ \lnot x$ (the atom is skipped); otherwise it sets $L _ { t } = L _ { t - 1 } \cup$ {claim literal of x} (the claim is accepted). Its output is the unique model of $\Phi \wedge L _ { n }$ . On the code Mod(Φ) this is the greedy decoder of Section A.1, because “all models of Φ that satisfy L have the same value at $x '$ means ${ } ^ { \mathfrak { s } } L \ \ v { \Sigma } { \ v { \Sigma } } | = _ { \Phi } \ v$ or ${ \cal L } | = _ { \Phi }  x ^ { \prime \prime } $ whenever Φ $\wedge L$ is satisfiable. The loss is the Hamming distance, and $R _ { \mathrm { g r } } ( \varepsilon ) = \mathbb { E } _ { Z , \pi } d _ { \mathrm { H } } ( G _ { \pi } ( y ) , \sigma ^ { \star } )$ with π uniform and independent of $Z .$ . The raw decoder outputs y and has $R _ { \mathrm { r a w } } ( \varepsilon ) = \varepsilon n$ exactly.

Lemma 16 (Well-definedness; equivalence with accept-if-consistent). Let Φ be a satisfiable formula over $V ,$ , let π be an order of V, and let $y \in \{ 0 , 1 \} ^ { V }$ be arbitrary answers.

(i) $\Phi \wedge L _ { t }$ is satisfiable for every t.

(ii) $\Phi \wedge L _ { n }$ has exactly one model, so the output $G _ { \pi } ( y )$ is well defined.

(iii) $G _ { \pi } ( y )$ equals the output of the accept-if-consistent decoder. This decoder processes the atoms in the same order, starts with $L _ { 0 } ^ { \prime } = \emptyset$ , adds the claim literal ℓ of the current atom to $L ^ { \prime } \textit { \textbf { \ i f } }$ and only if $\Phi \wedge L ^ { \prime } \wedge \ell$ is satisfiable, and outputs the unique model of $\Phi \wedge L _ { n } ^ { \prime }$

Proof. (i) We use induction on t. The formula Φ $\wedge L _ { 0 } = \Phi$ is satisfiable by assumption. A literal m is added at step t only if $L _ { t - 1 } \not \in _ { \Phi } \neg m$ , that is, only $\mathrm { i f } \ \Phi \wedge L _ { t - 1 } \wedge$ m is satisfiable.

(ii) At its own step t, each atom x either has its value entailed by $\Phi \land L _ { t - 1 }$ or receives a literal in $L _ { t }$ . Entailment is monotone in L, and $L _ { t - 1 } \subseteq L _ { n }$ . So $\Phi \wedge L _ { n }$ entails a value for every atom. By (i) it has a model, and this model is unique because every atom’s value is entailed.

(iii) We show by induction on t that $\Phi \wedge L _ { t }$ and $\Phi \wedge L _ { t } ^ { \prime }$ have the same set of models. This holds for $t = 0$ Suppose it holds for $t - 1 ;$ then both runs entail the same literals before step t. Let ℓ be the claim literal of the atom processed at step t. If neither value of the atom is entailed, then ¬ℓ is not entailed, so ℓ is consistent, and both decoders add ℓ. If ℓ is entailed, the greedy decoder skips the atom and the accept-if-consistent decoder adds $\ell ;$ adding an entailed literal does not change the set of models. If ¬ℓ is entailed, ℓ is inconsistent, and both decoders leave their sets unchanged. In all cases the two sets of models remain equal. ${ \mathrm { A t ~ } } t = n$ this set is a single model by (ii), so the two outputs coincide. □

Since every nonempty code is Mod(Φ) for some satisfiable Φ, Lemma 16 also shows that the greedy decoder of Section A.1 is well defined on every code.

The backbone is $B = \{ x \in V : \ \Phi \left. = t _ { x } \right\}$ , the set of atoms that take the same value in every model. For $, \notin B$ we put

$$
K _ { a } = \{ x \in V : \ \Phi \wedge \bar { t } _ { a } \vert = \bar { t } _ { x } \} , \qquad \sigma ^ { ( a ) } = \sigma ^ { \star } \oplus { \bf 1 } _ { K _ { a } } ,
$$

and for $a \in B$ we put $K _ { a } = \emptyset$ . When a $\notin B$ , the set $K _ { a }$ contains $a _ { \mathrm { : } }$ , and we call it the flip set of a. In this case $\Phi \wedge \bar { t } _ { a }$ is satisfiable, so it cannot entail both $t _ { x }$ and ${ \bar { t } } _ { x } ,$ and $\sigma ^ { ( a ) }$ makes $\bar { t } _ { x }$ true for every $x \in K _ { a }$ . We say that $( \Phi , \sigma ^ { \star } )$ has the single-flip property at $^ { a , }$ written $( \mathrm { P } _ { a } ) , \mathrm { i f } \ \sigma ^ { ( a ) }$ is a model of Φ. This means that flipping the forced atoms is enough to accommodate the wrong answer, without changing any other atom. Under $( { \bar { \operatorname { P } } } _ { a } ) , \sigma ^ { ( a ) }$ is the unique model of $\Phi \wedge \bar { t } _ { a }$ that agrees with $\sigma ^ { \star }$ outside $K _ { a }$ , because every model of Φ $\cdot \wedge \bar { t } _ { a }$ satisfies $\bar { t } _ { x }$ for all $x \in K _ { a }$ The single-error damage of the greedy decoder at a is $D _ { \pi } ( a ) = d _ { \mathrm { H } } ( G _ { \pi } ( y ) , \sigma ^ { \star } )$ for the answers $y = \sigma ^ { \star } \oplus e _ { a } ,$ in which only the answer on a is wrong.

Proof of Theorem 2. Suppose that exactly one answer is wrong, at the atom a. Then $D _ { \pi } ( a ) = 0$ for every order π when $a \in B$ . If a $\notin B$ and $( \mathrm { P } _ { a } )$ holds, then for every order π

$$
D _ { \pi } ( a ) = | K _ { a } | \cdot { \bf 1 } \{ a \mathrm { ~ p r e c e d e s ~ e v e r y ~ e l e m e n t ~ o f } { } K _ { a } \setminus \{ a \} \mathrm { ~ i n } \pi \} ,
$$

so that $\mathbb { E } _ { \pi } D _ { \pi } ( a ) = 1$ for the uniform order, and indeed for every exchangeable random order.

Case $a \in B$ . Here $\emptyset \models _ { \Phi } t _ { a }$ , so the atom a is always skipped and its wrong claim $\bar { t } _ { a }$ is never accepted. All other claims are true literals. Hence all accepted literals are true, $\sigma ^ { \star }$ is a model of $\Phi \wedge L _ { n }$ , and by Lemma $1 6 ( \mathrm { i i } )$ the output is $\sigma ^ { \star }$

Case a $\notin B$ under $( P _ { a } )$ . Let $\tau$ be the step at which a is processed. For $t < \tau$ , every accepted literal is the claim of an atom with a correct answer, so $L _ { t }$ consists of true literals.

Step 1 (when $t _ { a }$ is entailed). For every set S of true literals, $S \Vdash _ { \Phi } t _ { a }$ if and only if S contains a literal on some atom of $K _ { a }$ . Indeed, $S \Vdash _ { \Phi } t _ { a }$ if and only if $\Phi \wedge S \wedge \bar { t } _ { a }$ is unsatisfiable. If S contains no literal on an atom of $K _ { a }$ , then $\sigma ^ { ( a ) }$ satisfies $S ,$ because it agrees with $\sigma ^ { \star }$ outside $K _ { a } ;$ it also satisfies $\bar { t } _ { a } .$ , and it satisfies Φ by $( \mathrm { P } _ { a } )$ . So Φ $\wedge S \wedge \bar { t } _ { a }$ is satisfiable, and $S \not \vdash _ { \Phi } t _ { a }$ . If S contains $t _ { x }$ with $x \in K _ { a }$ , then $\Phi \wedge \bar { t } _ { a } \left| = \bar { t } _ { x } \right.$ contradicts $t _ { x } ,$ so Φ $\wedge S \wedge \bar { t } _ { a }$ is unsatisfiable.

Step 2 (when the wrong claim is accepted). A literal enters L only at the step of its own atom. So if $L _ { \tau - 1 }$ contains a literal on an atom of $K _ { a } .$ , this atom lies in $K _ { a } \ \backslash \ \{ a \}$ and precedes a. Conversely, suppose that some $x \in K _ { a } \setminus \{ a \}$ is processed at a step $t < \tau$ . Either $t _ { x }$ was added, and then $L _ { \tau - 1 }$ contains a literal on $K _ { a } ;$ or x was skipped. In the second case $L _ { t - 1 }$ entails $t _ { x }$ or $\bar { t } _ { x } ;$ it cannot entail ${ \bar { t } } _ { x } .$ , because $\sigma ^ { \star }$ is a model of Φ $\wedge L _ { t - 1 } ;$ so $L _ { t - 1 } \mid = _ { \Phi } t _ { x }$ . Then $\Phi \wedge L _ { t - 1 } \wedge \bar { t } _ { a }$ entails both $t _ { x }$ and $\bar { t } _ { x }$ , so it is unsatisfiable, that is, $L _ { t - 1 } \mid = _ { \Phi } t _ { a }$ . By Step 1, $L _ { t - 1 } \subseteq L _ { \tau - 1 }$ contains a literal on $K _ { a }$ . Together with Step 1 this shows: at step $\tau , L _ { \tau - 1 } \Vdash _ { \Phi } t _ { a }$ if and only if some element of $K _ { a } \setminus \{ a \}$ precedes a. The wrong literal $\bar { t } _ { a }$ is never entailed at step τ, because $\sigma ^ { \star }$ is a model of $\Phi \wedge L _ { \tau - 1 }$ . Hence $\bar { t } _ { a }$ is accepted if and only if a is the first atom of $K _ { a }$ in $\pi ;$ otherwise a is skipped.

Step 3 (rejected case). If $\bar { t } _ { a }$ is not accepted, every literal ever added is true, so the output is $\sigma ^ { \star }$ and $D _ { \pi } ( a ) = 0$

Step 4 (accepted case). Now $L _ { n } = \{ \bar { t } _ { a } \} \cup S$ , where S consists of true literals of atoms other than a. No atom of $K _ { a } \ \backslash \ \{ a \}$ contributes to S. Indeed, such an atom x comes after $a ;$ at its step L contains $t _ { a } ,$ and $\{ \bar { t } _ { a } \} \ : \lvert = _ { \Phi } \ : \bar { t } _ { x }$ by the definition of $K _ { a } ;$ so x is skipped. Hence $\sigma ^ { ( a ) }$ satisfies S (it agrees with $\sigma ^ { \star }$ outside $K _ { a } )$ , satisfies $t _ { a }$ , and satisfies Φ by $( \mathrm { P } _ { a } )$ . By Lemma 16(ii) the output is $\sigma ^ { ( a ) }$ , and $D _ { \pi } ( a ) = | K _ { a } |$

Expectation. For the uniform order, and for every exchangeable random order, each of the $| K _ { a } |$ atoms of $K _ { a }$ is equally likely to come first among them. So a comes first with probability $1 / | K _ { a } |$ , and $\begin{array} { r } { \mathbb { E } _ { \pi } D _ { \pi } ( a ) = | K _ { a } | \cdot | K _ { a } | ^ { - 1 } = } \end{array}$ 1. □

Proof of Proposition 3. We show that if every clause of Φ has at most two literals, then $( \mathrm { P } _ { a } )$ holds for every truth $\sigma ^ { \star } \in \mathcal { C }$ and every atom $a \notin B$ . Let E be the set of literals entailed by $\Phi \wedge \bar { t } _ { a }$ . Since $\Phi \wedge \bar { t } _ { a }$ is satisfiable, E contains no complementary pair. By definition, for $x \in K _ { a }$ the literal $\bar { t } _ { x }$ lies in $E ,$ and $\sigma ^ { ( a ) }$ makes it true. Take a clause $p \vee q$ of Φ and suppose that $\sigma ^ { ( a ) }$ falsifies both p and q. If neither atom lies in $K _ { a } ,$ then $\sigma ^ { ( a ) }$ agrees with $\sigma ^ { \star }$ on both atoms, and $\sigma ^ { \star }$ satisfies the clause; this is a contradiction. So we may assume that atom $( p ) \in K _ { a }$ , where atom(p) denotes the atom of the literal p. Then ¬p is true under $\sigma ^ { ( a ) }$ , and it equals $\bar { t } _ { \mathrm { a t o m } ( p ) }$ , which lies in $E .$ From $\Phi \models p \lor q$ we get $q \in E$ . If $\mathrm { a t o m } ( q ) \in K _ { a }$ , the unique literal of E on this atom is $\bar { t } _ { \mathrm { a t o m } ( q ) }$ , which $\sigma ^ { ( a ) }$ makes true; so q is true, a contradiction. If atom $\mathfrak { i } ( q ) \not \in K _ { a }$ , then $\bar { t } _ { \mathrm { a t o m } ( q ) } \notin E$ , so $q = t _ { \mathrm { a t o m } ( q ) } ;$ ; this literal is true under $\sigma ^ { \star }$ and hence under $\sigma ^ { ( a ) }$ , again a contradiction. Finally, take a unit clause $( p )$ . Then $p \in E$ , and $p = t _ { \mathrm { a t o m } ( p ) }$ because $\sigma ^ { \star } \vdash p .$ . If atom(p) were in $K _ { a }$ , then E would contain both p and $\bar { t } _ { \mathrm { a t o m } ( p ) } = \lnot p ;$ so $\mathrm { a t o m } ( p ) \notin K _ { a }$ , and $\sigma ^ { ( a ) }$ satisfies p. □

Implication theories. For a directed graph $G = ( V , A )$ let $\Phi _ { G } = \textstyle \bigwedge _ { ( u , v ) \in A } ( \neg u \lor v )$ . Identify an assignment with its set of true atoms. The models of $\Phi _ { G }$ are then the closed sets: the sets M such that $u \in M$ and $( u , v ) \in A$ imply $v \in M$ (the up-closed sets of the preorder generated by G). For $X \subseteq V .$ , let cl(X) be the smallest closed set that contains X. Since ∅ and V are models, the backbone is empty. Let $M ^ { \star }$ be the true set, let reach(a) be the set of vertices reachable from a (including a), and let $\operatorname { a n c } ( a )$ be the set of vertices from which a is reachable (including a). Then:

• for a false positive $( a ~ \notin ~ { \cal M } ^ { \star } ) , ~ { \cal K } _ { a } = \mathrm { r e a c h } ( a ) \setminus { \cal M } ^ { \star }$ and $\sigma ^ { ( a ) } = M ^ { \star } \cup$ reach(a); hence $\left| K _ { a } \right| = c _ { a } : =$ $\left| \operatorname { c l } ( M ^ { \star } \cup \{ a \} ) \setminus M ^ { \star } \right|$ , the cone of a;

• for a false negative $( a \in M ^ { \star } ) , K _ { a } = \operatorname { a n c } ( a ) \cap M ^ { \star } { \mathrm { ~ a n d ~ } } \sigma ^ { ( a ) } = M ^ { \star } \setminus \operatorname { a n c } ( a )$

Indeed, the models of $\Phi _ { G } \wedge a$ are the closed sets that contain reach(a); the smallest is reach(a) and the largest is V. So the literals entailed by $\Phi _ { G } \wedge a$ are exactly the positive literals on reach(a). For a $\notin \ : M ^ { \star }$ we have $\bar { t } _ { a } = a$ , and $\hat { t } _ { x }$ is the positive literal x exactly when $x \notin M ^ { \star }$ ; this gives $K _ { a } = \operatorname { r e a c h } ( a ) \setminus M ^ { \star }$ . Since $M ^ { \star }$ is closed, M<sup>⋆</sup> ∪ reach(a) is closed and equals cl $( M ^ { \star } \cup \{ a \} )$ . Dually, the models of $\Phi _ { G } \wedge \neg a$ are the closed sets contained in $V \setminus \operatorname { a n c } ( a )$ , which is itself closed; the smallest is ∅ and the largest is $V \backslash$ anc(a). So the literals entailed by $\Phi _ { G } \wedge \neg a$ are exactly the negative literals on anc(a). For $a \in M ^ { \star }$ we have $\bar { t } _ { a } = \neg a$ , and $\bar { t } _ { x } = \neg x$ exactly when $x \in M ^ { \star }$ ; this gives $K _ { a } = \operatorname { a n c } ( a ) \cap M ^ { \star }$

Proof of Corollary 4. We show that for a satisfiable 2-CNF Φ with backbone B, every truth $\sigma ^ { \star } \in \mathcal { C }$ and every $\varepsilon \in [ 0 , { \frac { 1 } { 2 } } ]$

$$
\begin{array} { r } { - \varepsilon ^ { 2 } n ( n - 1 ) \ \le \ R _ { \mathrm { g r } } ( \varepsilon ) - \varepsilon ( n - | B | ) \ \le \ \frac 1 2 \varepsilon ^ { 2 } n ^ { 2 } ( n - 1 ) . } \end{array}
$$

We split on the error set $S = \{ x : ~ Z _ { x } = 1 \}$ . With no error the output is $\sigma ^ { \star }$ (Lemma $1 6 ( \mathrm { i i } ) )$ , and the loss is 0. With exactly one error, at a (probability $\varepsilon ( 1 - \varepsilon ) ^ { n - 1 } )$ , Theorem 2 and Proposition 3 give ${ \mathbb E } _ { \pi } D _ { \pi } ( a ) = { \bf 1 } \{ a \notin B \}$ Summed over $^ { a , }$ these events contribute $\varepsilon ( 1 - \varepsilon ) ^ { n - 1 } ( n - | B | )$ . With two or more errors the loss lies in $[ 0 , n ]$ and the probability is at most ${ \binom { n } { 2 } } \varepsilon ^ { 2 } ;$ ; these events contribute a term $r \in [ 0 , \frac { 1 } { 2 } n ^ { 2 } ( n - 1 ) \varepsilon ^ { 2 } ]$ . Hence $\begin{array} { r l } { R _ { \mathrm { g r } } ( \varepsilon ) - \varepsilon ( n - | B | ) = } & { { } } \end{array}$ $- \varepsilon ( n - | B | ) \bigl [ 1 - ( 1 - \varepsilon ) ^ { n - 1 } \bigr ] + r$ . Finally, $0 \leq \varepsilon ( n - | B | ) [ 1 - ( 1 - \varepsilon ) ^ { { n - 1 } } ] \leq \varepsilon ^ { 2 } ( n - 1 ) n$ by Bernoulli’s inequality, which gives both bounds.

For an implication theory $\Phi _ { G }$ the backbone is empty, so $R _ { \mathrm { g r } } ( \varepsilon ) = \varepsilon n + O ( \varepsilon ^ { 2 } ) = R _ { \mathrm { r a w } } ( \varepsilon ) + O ( \varepsilon ^ { 2 } )$ for every truth. The per-atom statement $\mathbb { E } _ { \pi } D _ { \pi } ( a ) = 1$ holds for each atom separately, so the conclusion holds for false-positive and false-negative errors alike. □

Remark 17 (The random order is essential). For a fixed order the single-error damage takes only the values 0 and $| K _ { a } |$ . Under a random order, the wrong claim is accepted with probability $1 / | K _ { a } |$ , and then it damages exactly $| K _ { a } |$ atoms. A fixed order can make every error of one type maximally damaging. Consider the chain $v _ { 1 } \to v _ { 2 } \to \cdots \to v _ { n }$ (clauses $\neg v _ { i } \lor v _ { i + 1 } )$ , whose models are the up-sets $\{ v _ { j } , \ldots , v _ { n } \}$ and ∅. For a false positive at $^ { a , }$ the flip set $K _ { a } = \operatorname { r e a c h } ( a ) \setminus M ^ { \star }$ consists of a and the false atoms after it on the chain, and the wrong claim is accepted if and only if a is asked before every other atom of $K _ { a }$ . So the topological order $v _ { 1 } , v _ { 2 } , \ldots , v _ { n }$ accepts every false positive and lets it corrupt every false atom reachable from it, while the reverse order $v _ { n } , \ldots , v _ { 1 }$ rejects every false positive with $| K _ { a } | \ge 2$ . For example, with the chain $a  b ,$ , the truth “both false” and the wrong answer $a = 1$ , the order $( a , b )$ accepts a, then finds b implied true, and has damage 2; the order $( b , a )$ accepts $b = 0$ , then finds a implied false, and has damage 0. For false negatives the roles of the two orders are exchanged: $K _ { a } = \operatorname { a n c } ( a ) \cap M ^ { \star }$ consists of a and the true atoms before ${ \mathrm { i t } } ,$ and the reverse topological order is the damaging one.

Proof of Proposition 5. We show that on a satisfiable 2-CNF with truth $\sigma ^ { \star }$ , a single wrong answer at an atom a $\rangle \notin B$ costs the ML decoder exactly 1 in expectation if $| K _ { a } | \le 2$ and 0 if $| K _ { a } | \ge 3$ , and that a single wrong answer at an atom $a \in B$ costs 0. Consequently $R _ { \mathrm { M L } } ( \varepsilon ) = \varepsilon \left| \{ a \not \in B : \ | K _ { a } | \leq 2 \} \right| + O ( \varepsilon ^ { 2 } )$

We use Lemma 1(i). Let $\boldsymbol { y } = \boldsymbol { \sigma } ^ { \star } \oplus \boldsymbol { e } _ { a } ,$ , and let $m _ { a }$ be the number of codewords of the form $y \oplus e _ { b }$ with $b \neq a$ when $y \not \in { \mathcal { C } }$ . Then $D _ { \mathrm { M L } } ( a ) = 1$ if $y \in { \mathcal { C } }$ and $D _ { \mathrm { M L } } ( a ) = 2 m _ { a } / ( 1 + m _ { a } )$ otherwise. Every word that difers from $\sigma ^ { \star }$ at a satisfies $\bar { t } _ { a } .$

Case $a \in B$ . No model satisfies ${ \bar { t } } _ { a } .$ . So $y \not \in { \mathcal { C } } ,$ , no word $y \oplus e _ { b }$ is a codeword, $m _ { a } = 0$ and $D _ { \mathrm { M L } } ( a ) = 0$

Case a /∈ B. By Proposition $\mathrm { 3 , ~ } ( \mathrm { P } _ { a } )$ holds. Every model that satisfies $\bar { t } _ { a }$ satisfies $\bar { t } _ { x }$ for all $x \in K _ { a } ,$ , so it difers from $\sigma ^ { \star }$ on all of $K _ { a }$ . If $y \in { \mathcal { C } }$ , then y difers from $\sigma ^ { \star }$ only at $^ { a , }$ so $K _ { a } = \{ a \}$ . Conversely, if $K _ { a } = \{ a \}$ , then $y = \sigma ^ { ( a ) }$ is a model by $( \mathrm { P } _ { a } )$ . So $y \in { \mathcal { C } }$ if and only $\mathrm { i f } \ | K _ { a } | = 1$ , and in this case $D _ { \mathrm { M L } } ( a ) = 1$ . Now let $| K _ { a } | \geq 2 $ so that $y \not \in { \mathcal { C } }$ . A codeword $\boldsymbol { y } \oplus \boldsymbol { e } _ { b } = \sigma ^ { \star } \oplus e _ { a } \oplus \boldsymbol { e } _ { b }$ with $b \neq a$ difers from $\sigma ^ { \star }$ exactly on $\{ a , b \}$ and satisfies $\bar { t } _ { a } ;$ hence $K _ { a } \subseteq \{ a , b \}$ , that is, $K _ { a } = \{ a , b \}$ . Conversely, if $K _ { a } = \{ a , b \}$ , then $\sigma ^ { \star }$ ⊕ $e _ { a } \oplus e _ { b } = \sigma ^ { ( a ) }$ is a model by $( \mathrm { P } _ { a } )$ So $m _ { a } = 1 \mathrm { i f } | K _ { a } | = 2$ , and $m _ { a } = 0 \mathrm { i f } \left| K _ { a } \right| \geq 3$ . With uniform tie-breaking this gives $D _ { \mathrm { M L } } ( a ) = 2 \cdot \textstyle { \frac { 1 } { 2 } } = 1$ and $D _ { \mathrm { M L } } ( a ) = 0 .$ , respectively.

Summing over the atoms and applying (1) gives the formula for $R _ { \mathrm { M L } }$ . In particular $m _ { a } \le 1$ for every atom, as used in the proof of Lemma 1(i). Comparing with Corollary $4 , R _ { \mathrm { g r } } ( \varepsilon ) - R _ { \mathrm { M L } } ( \varepsilon ) = \varepsilon | \{ a \not \in B : | K _ { a } | \geq 3 \} | + O ( \varepsilon ^ { 2 } )$ On an implication theory $( B = \varnothing )$ , the greedy and raw decoders therefore have the same first-order risk, while the greedy risk exceeds the ML risk by $\varepsilon \left| \{ a : \ \vert K _ { a } \vert \geq 3 \} \right|$ at first order. □

Horn hubs. For integers $k \geq 1$ and $s \geq 2$ , the Horn hub $H _ { k , s }$ has the atom a and, for each $i = 1 , \ldots , k .$ the atoms $b _ { i , 1 } , \ldots , b _ { i , s - 1 } , c _ { i } ;$ so $n = 1 + k s$ . Its rules are $b _ { i , 1 } \wedge \cdot \cdot \cdot \wedge b _ { i , s - 1 } \wedge a  c _ { i }$ for $i = 1 , \ldots , k .$ , each with a body of s atoms. The corresponding clauses are $C _ { i } = \neg b _ { i , 1 } \lor \cdots \lor \neg b _ { i , s - 1 } \lor \neg a \lor c _ { i }$ The truth is $M ^ { \star } = \{ b _ { i , j } : \ i \leq k , \ j \leq s - 1 \}$ , the assignment in which exactly the atoms $b _ { i , j }$ are true; it is a model because a is false. Block i is $S _ { i } = \{ b _ { i , 1 } , \ldots , b _ { i , s - 1 } , c _ { i } \}$ , and $T _ { i } = \{ b _ { i , 1 } , \dots , b _ { i , s - 1 } , \neg c _ { i } \}$ is the set of its true literals. The cone of a is $\begin{array} { r } { c _ { a } = k + 1 } \end{array}$ : the closure of $M ^ { \star } \cup \{ a \}$ adds a and $c _ { 1 } , \ldots , c _ { k }$ . For $s = 2$ the rules have the form of the rule isa $( x , y ) \land \operatorname { h a s } ( y , z )  \operatorname { h a s } ( x , z )$ with a shared premise.

Proof of Theorem 6. For uniform π, a single false positive on a (all other answers correct) has

$$
\mathbb { E } _ { \pi } D _ { \pi } ( a ) = ( k + 1 ) \int _ { 0 } ^ { 1 } ( 1 - u ^ { s } ) ^ { k } d u = ( k + 1 ) \frac { \Gamma ( 1 + 1 / s ) \Gamma ( k + 1 ) } { \Gamma ( k + 1 + 1 / s ) } = \Gamma \big ( 1 + \frac { 1 } { s } \big ) k ^ { 1 - 1 / s } \big ( 1 + O ( 1 / k ) \big ) ,
$$

where Γ is the Gamma function. For $s = 2$ this equals $\begin{array} { r } { ( k + 1 ) 4 ^ { k } ( k ! ) ^ { 2 } / ( 2 k + 1 ) ! \sim \frac { \sqrt { \pi } } { 2 } \sqrt { k } . } \end{array}$ . Every other single error has damage exactly 1 for every π. Hence

$$
R _ { \mathrm { g r } } ( \varepsilon ) = \varepsilon \left[ k s + \mathbb { E } _ { \pi } D _ { \pi } ( a ) \right] + O ( \varepsilon ^ { 2 } ) = R _ { \mathrm { r a w } } ( \varepsilon ) + \varepsilon \left[ \mathbb { E } _ { \pi } D _ { \pi } ( a ) - 1 \right] + O ( \varepsilon ^ { 2 } ) .
$$

Satisfiability criterion. Let $L ^ { \prime }$ be a set of literals without complementary pairs. We claim that $\Phi \wedge L ^ { \prime }$ is unsatisfiable if and only if $a \in L ^ { \prime }$ and $T _ { i } \subseteq L ^ { \prime }$ for some i, that is, $L ^ { \prime }$ falsifies every literal of some clause $C _ { i }$ other than $\neg a$ . The $ { \mathbf { \hat { \theta } } } _ { 1 }  { \mathbf { f } } ^ { \dagger }$ direction is clear. For the “only $\mathrm { i f } ^ { \dag }$ direction, suppose first that $a \notin L ^ { \prime }$ . Setting a false and every other atom as $L ^ { \prime }$ prescribes (free atoms arbitrarily) satisfies every clause, since each clause contains ¬a. Suppose next that $a \in L ^ { \prime }$ but $T _ { i } \ \not \subseteq { \cal L ^ { \prime } }$ for every i. Then each clause $C _ { i }$ has a literal on $S _ { i }$ that is either true under $L ^ { \prime }$ or lies on an atom that $L ^ { \prime }$ leaves free. Set each free atom of $S _ { i }$ so as to satisfy $C _ { i }$ . The atoms of $S _ { i }$ occur only in $C _ { i } .$ , so these choices do not interact, and $\Phi \wedge L ^ { \prime }$ is satisfiable.

The run with the wrong claim a. All other claims are the true literals, the members of the sets $T _ { i }$ . Let $\tau$ be the step of a.

• Before τ, the atom a has no literal in $L .$ For an atom x 6= a processed before $\tau ,$ neither $L \cup \{ x \}$ nor $L \cup \{ \lnot x \}$ contains $^ { a , }$ so both are consistent with Φ by the criterion. Hence nothing is entailed and every claim is accepted. So $L _ { \tau - 1 }$ is the set of true literals of the atoms processed before a.

$\mathrm { A t } \ \tau ,$ the literal a is not entailed, because $L _ { \tau - 1 } \cup \{ \lnot a \}$ does not contain a and is consistent. The literal ¬a is entailed if and only $\mathrm { i f } \Phi \wedge L _ { \tau - 1 } \wedge a$ is unsatisfiable, that is, if and only if $T _ { i } \subseteq L _ { \tau - 1 }$ for some i. This means that some block lies entirely before a in $\pi .$ . In that case the wrong claim is skipped, all accepted literals are true, and the output is $\sigma ^ { \star }$ (damage 0).

• Otherwise the wrong claim a is accepted. Consider block i and an atom $x \in S _ { i }$ processed after $^ { a , }$ and let L be the set of accepted literals at the step of x. Its true literal $t _ { x }$ is not entailed. Indeed, unsatisfiability of $\Phi \wedge L \wedge \bar { t } _ { x }$ would require $T _ { j } \subseteq L \cup \{ \bar { t } _ { x } \}$ for some j. The case $j = i$ is impossible because $t _ { x } \in T _ { i }$ but $t _ { x } \notin L \cup \{ \bar { t } _ { x } \}$ . The case $j \neq i$ would give $T _ { j } \subseteq L$ , which contradicts the satisfiability of $\Phi \wedge L$ (Lemma 16(i)). $\mathrm { B y }$ the same argument, the false literal $\bar { t } _ { x }$ is entailed if and only if $\Phi \wedge L \wedge t _ { x }$ is unsatisfiable, that is, if and only if $T _ { i } \setminus \{ t _ { x } \} \subseteq L$

Now follow block i along the order π. Every atom of $S _ { i }$ other than the last one in $\pi$ is accepted with its true literal. For atoms before a this follows from the first bullet. For an atom x after a that is not the last atom of $S _ { i }$ , the literal of the last atom is not yet in $L ,$ so $T _ { i } \setminus \{ t _ { x } \} \subset L ,$ and neither literal of x is entailed. The last atom of $S _ { i }$ comes after $^ { a , }$ , because $S _ { i }$ does not lie entirely before a. At its step the other $s - 1$ literals of $T _ { i }$ are in $L ,$ so its false literal is entailed, and the output gives it its false value. So each block contributes damage exactly 1, and together with a the damage is $k + 1$

Probability of acceptance. Attach independent $\mathrm { U n i f } [ 0 , 1 ]$ times to all atoms; sorting the times gives a uniform order. Conditionally on the time u of $^ { a , }$ block i lies entirely before a with probability $u ^ { s }$ , independently over the blocks. Hence the wrong claim is accepted with probability $\textstyle \int _ { 0 } ^ { 1 } ( 1 - u ^ { s } ) ^ { k } d u$ , and $\begin{array} { r } { \mathbb { E } _ { \pi } D _ { \pi } ( a ) = ( k + 1 ) \int _ { 0 } ^ { 1 } ( 1 - } \end{array}$ $u ^ { s } ) ^ { k } d u$ . The substitution $t = u ^ { s }$ turns the integral into $\textstyle { \frac { 1 } { s } } \operatorname { B } ( { \frac { 1 } { s } } , k + 1 )$ , where $\begin{array} { r } { \mathrm { B } ( \alpha , \beta ) = \int _ { 0 } ^ { 1 } t ^ { \alpha - 1 } ( 1 - t ) ^ { \beta - 1 } d t = } \end{array}$ $\Gamma ( \alpha ) \Gamma ( \beta ) / \Gamma ( \alpha + \beta )$ is the Beta function. Since $\textstyle { \frac { 1 } { s } } \Gamma ( { \frac { 1 } { s } } ) = \Gamma ( { \bar { 1 } } + { \frac { 1 } { s } } )$ , the integral equals $\Gamma ( 1 + { \textstyle \frac { 1 } { s } } ) \bar { \Gamma } ( k + 1 ) / \Gamma ( k + 1 + { \textstyle \frac { 1 } { s } } )$ . For $s = 2$ , the identity $\begin{array} { r } { \Gamma ( k + \frac { 3 } { 2 } ) = \frac { ( 2 k + 2 ) ! } { 4 ^ { k + 1 } ( k + 1 ) ! } \sqrt { \pi } } \end{array}$ and $\textstyle \Gamma ( { \frac { 3 } { 2 } } ) = { \frac { \sqrt { \pi } } { 2 } }$ give

$$
{ \frac { \Gamma ( { \frac { 3 } { 2 } } ) k ! } { \Gamma ( k + { \frac { 3 } { 2 } } ) } } = { \frac { 2 \cdot 4 ^ { k } k ! ( k + 1 ) ! } { ( 2 k + 2 ) ! } } = { \frac { 4 ^ { k } ( k ! ) ^ { 2 } } { ( 2 k + 1 ) ! } } .
$$

The asymptotic form follows from $\Gamma ( k + 1 ) / \Gamma ( k + 1 + \alpha ) = k ^ { - \alpha } ( 1 + O ( 1 / k ) )$ for fixed $\alpha \in ( 0 , 1 ]$ , together with $k + 1 = k ( 1 + 1 / k )$ . For $s = 2$ it reads $\begin{array} { r } { \mathbb { E } _ { \pi } D _ { \pi } ( a ) \sim \Gamma ( \frac { 3 } { 2 } ) \sqrt { k } = \frac { \sqrt { \pi } } { 2 } \sqrt { k } . } \end{array}$

Other single errors. The atom a is false in $M ^ { \star }$ , so its only possible error is the false positive treated above. Consider a false negative on some $b _ { i , j }$ or a false positive on some $c _ { i }$ , with all other answers correct. The answer vector y then has a false, so it satisfies every clause and is itself a model. Hence every claim is consistent with every set of accepted claims, since all of them are literals of the model $y .$ . So the negation of a claim is never entailed, and every atom either has its claim accepted or has its claim entailed. Thus y is a model of $\Phi \wedge L _ { n }$ and by Lemma $1 6 ( \mathrm { i i } )$ the output is y, with damage $d _ { \mathrm { H } } ( y , \sigma ^ { \star } ) = 1$ , for every π.

Summation. There are ks atoms other than a, each with single-error damage 1, and $n = k s + 1$ . By Lemma 1(ii), $R _ { \mathrm { g r } } ( \varepsilon ) = \varepsilon [ k s + \mathbb { E } _ { \pi } D _ { \pi } ( a ) ] + O ( \varepsilon ^ { 2 } ) = \varepsilon n + \varepsilon [ \mathbb { E } _ { \pi } D _ { \pi } ( a ) - 1 ] + O ( \varepsilon ^ { 2 } )$ . Since max $: \mathbb { E } _ { \pi } D _ { \pi } ( x ) \le k + 1$ , the remainder bound (1) gives $| R _ { \mathrm { g r } } ( \varepsilon ) - \varepsilon [ k s + \mathbb { E } _ { \pi } D _ { \pi } ( a ) ] | \le \varepsilon ^ { 2 } n ( n - 1 ) ( k + 1 + \frac { n } { 2 } )$ . The excess $\mathbb { E } _ { \pi } D _ { \pi } ( a ) - 1 { \mathrm { ~ i s ~ } } \Theta ( k ^ { 1 - 1 / s } )$ , so the greedy risk exceeds the raw risk by $\varepsilon \Theta \big ( k ^ { 1 - 1 / s } \big )$ at first order. □

Remark 18 (Where the single-flip property fails). In $H _ { k , s }$ the formula Φ $\wedge a$ entails nothing new. Indeed, by the satisfiability criterion $\{ a , \ell \}$ is consistent for every literal ℓ on another atom, because each $T _ { i }$ has $s \geq 2$ literals. So $K _ { a } = \{ a \}$ and $\sigma ^ { ( a ) } \overset { \cdot } { = } \overset { \cdot } { M ^ { \star } } \cup \{ a \}$ , which violates every clause $C _ { i }$ and is not a model. The wrong literal alone does not imply additional incorrect values. These values are implied later, when the wrong literal is combined with correct literals. Acceptance is determined by the s-element blocks rather than by the flip set. The single-flip property can also fail for false negatives. Change the signs of the literals on a and on $c _ { i }$ in every clause; this gives the rules $b _ { i , 1 } \wedge \cdot \cdot \cdot \wedge b _ { i , s - 1 } \wedge c _ { i } \to a \ ( \mathrm { f o r } \ s = 2 )$ , the rules $b _ { i } \wedge c _ { i } \to a )$ . Take the truth in which all atoms are true, and a false negative on a. The unsatisfiability criterion becomes: $\neg a \in L ^ { \prime }$ and some block has all its true literals in $L ^ { \prime }$ . The proof above, with the literals of each block exchanged, gives the same damage law. For comparison with ${ \mathrm { M L } } ,$ when $k \geq 2$ the word $M ^ { \star } \cup \{ a \}$ has $M ^ { \star }$ as its unique nearest codeword, because flipping a single atom of a block repairs only one of the k violated clauses. So the ML single-error damage at a is 0.

General Horn theories. A definite Horn theory is a conjunction of rules $b _ { 1 } \wedge \cdot \cdot \cdot \wedge b _ { r } \to h$ over atoms, where $\{ b _ { 1 } , \ldots , b _ { r } \}$ is the body and h the head of the rule. Identify a model with its set of true atoms. The models are then the closed sets, that ${ \mathrm { i s } } ,$ the sets that contain the head of every rule whose body they contain. For $X \subseteq V .$ $\operatorname { c l } ( X )$ is the smallest closed set that contains X, obtained by repeatedly adding conclusions whose premises are already present. For a closed truth $M ^ { \star }$ and an atom a $\notin M ^ { \star }$ , the cone of a is $c _ { a } = | \operatorname { c l } ( M ^ { \star } \cup \{ a \} ) \setminus M ^ { \star } |$ ; this extends the cone of implication theories.

Conjecture 19 (Body-size law). For every $s \geq 2$ there is a constant $C _ { s } < \infty$ with the following property. Let Φ be a definite Horn theory whose rule bodies have at most s atoms, let $M ^ { \star }$ be a closed set, and let $a \notin M ^ { \star }$ . Then a single false positive at a (all other answers correct) has, for uniform π,

$$
\begin{array} { r } { \mathbb { E } _ { \pi } D _ { \pi } ( a ) ~ \le ~ C _ { s } c _ { a } ^ { 1 - 1 / s } . } \end{array}
$$

By Theorem 6 the exponent cannot be lowered: the hub $H _ { k , s }$ has $\textstyle c _ { a } = k + 1$ and $\mathbb { E } _ { \pi } D _ { \pi } ( a ) \sim \Gamma ( 1 + \frac { 1 } { s } ) k ^ { 1 - 1 / s }$ . For $s = 1$ , that is, for implication theories, the statement holds with $C _ { 1 } = 1$ and with equality: there $| K _ { a } | = c _ { a }$ for a false positive, and $\mathbb { E } _ { \pi } D _ { \pi } ( a ) = 1$ by Theorem 2 and Proposition 3. A corresponding bound for false negatives remains open; it requires a way to measure the atoms afected when a true premise is removed. In the simulations of Section B, which also test Conjecture 19, false-negative damages reach 1.48.

## A.3 Entity resolution

The items are $[ m ] = \{ 1 , \dots , m \}$ . The code is the set of partitions of $[ m ]$ , written as vectors of $\binom m 2$ pair indicators $x _ { u v } = { \bf 1 } \{ u \sim v \}$ , where u ∼ v means that u and v lie in the same block. The constraints are transitivity, $x _ { u v } \wedge x _ { v w } \to x _ { u w }$ . Each pair e carries an independent answer $y _ { e } = x _ { e } ^ { \star } \oplus \xi _ { e }$ with $\xi _ { e } \sim \mathrm { B e r n } ( \delta )$ . In this section the true partition $x ^ { \star }$ has a single cluster, so every flip is a false “diferent”, and δ is the false-split rate. The loss L is the number of pairs on which the output partition disagrees with the truth; here it is the number of pairs that the output separates.

The pipeline $U _ { \pi }$ is the transitivity pruning of Wang et al. (2013). It maintains a partition of [m] into blocks and a set of $\neq$ constraints between blocks. It processes the pairs in the order π. For a pair $e = \{ u , v \}$ , it skips e if u and v lie in the same block or if their blocks carry $\mathrm { ~ a ~ } \neq$ constraint. Otherwise it queries the answer $y _ { e }$ . On $^ { 6 6 } \mathrm { s a m e } ^ { 5 5 }$ it merges the two blocks, and the merged block inherits all $\neq$ constraints of both; on “diferent” it adds a $\neq$ constraint between the two blocks. Its output is the final partition into blocks.

For an order π of the pairs, $T _ { \pi }$ is the spanning tree obtained by scanning π and keeping a pair if and only if it joins two diferent components of the pairs kept so far. This is Kruskal’s algorithm (Kruskal, 1956) for the edge weights “position in $\pi ^ { \mathfrak { n } }$ , so $T _ { \pi }$ is the minimum spanning tree for these weights. It depends on π only, not on the answers. With all answers correct, its edges are exactly the queried pairs; we therefore use it as a reference tree even in noisy runs. For a spanning tree T of the complete graph $K _ { m } , d _ { T } ( u , v )$ is the number of edges on the path from u to v in T. For an edge $e \in T , n _ { e }$ and $m - n _ { e }$ are the numbers of vertices of the two components of $T - e$ . The Wiener index of T is

$$
W ( T ) = \sum _ { \{ u , v \} } d _ { T } ( u , v ) = \sum _ { e \in T } n _ { e } ( m - n _ { e } ) ,
$$

where the second form counts, for each edge e, the $n _ { e } ( m - n _ { e } )$ pairs whose path uses e. We also write $W _ { 2 } ( T ) =$ $\sum _ { \{ u , v \} } d _ { T } ( u , v ) ^ { 2 }$

Lemma 20 (Union-find is the greedy decoder). For every order π of the pairs and all answers, the union-find pipeline $U _ { \pi }$ equals the greedy decoder on the partition code. That is, it skips exactly the pairs whose relation is entailed by the answers accepted so far, it accepts the answer on every other pair, and its output is the unique partition that satisfies all accepted answers.

Proof. Let L be a set of accepted literals, each an equality $u \sim v$ or an inequality u $\nsim v .$ , and assume that L is satisfiable, that is, that some partition satisfies all of them. Call the connected components of the graph of accepted equalities the equality-components. We claim: (a) L entails $u \sim v$ if and only if u and v lie in the same equality-component; (b) L entails u 6∼ v if and only if some accepted inequality joins the equality-component of u to that of v.

(a) If u and v are joined by a path of accepted equalities, transitivity gives u ∼ v in every model. Conversely, consider the partition into equality-components. It satisfies every accepted equality. It also satisfies every accepted inequality: in any model every equality-component lies in one block, so a satisfiable L contains no inequality inside an equality-component. This partition separates any two points of diferent equality-components, so for such points L does not entail $u \sim v$

(b) If an accepted inequality joins a point x of the component of u to a point y of the component of v, then every model has $u \sim x , x \not \sim y$ and $y \sim v$ , hence u $\nsim$ v by transitivity. Conversely, suppose that no accepted inequality joins the two components. If the two components coincide, L entails $u \sim v$ and, being satisfiable, does not entail u $\nsim$ . Otherwise, merge these two components and keep all other equality-components as blocks. The resulting partition satisfies every accepted equality and every accepted inequality, and in it $u \sim v ;$ so L does not entail u $\nsim$

We now compare the two decoders step by step. Suppose that before some step both have accepted the same satisfiable set $L ;$ this holds at the start, with $L = \emptyset$ . The blocks of the pipeline are the equality-components of $L ,$ , because a merge happens exactly when an equality is accepted. Two blocks carry a $\neq$ constraint if and only if an accepted inequality joins them, because constraints are created by accepted inequalities and inherited through merges. By (a) and (b), the pipeline skips the current pair exactly when L entails its relation, which is the rule of the greedy decoder; otherwise both decoders accept the answer. An accepted answer is consistent with L, because its negation is not entailed, so the new set is again satisfiable. By induction, the two decoders accept the same literals at every step. At the end, the final blocks form the partition into equality-components, which satisfies $L _ { n }$ by the proof of (a); by Lemma 16(ii) it is the unique model of $L _ { n }$ , that is, the output of the greedy decoder. □

The full flip vector ξ assigns a potential error to every pair, but the pipeline reads $\xi _ { e }$ only when it queries that pair. The decision to query e uses only earlier answers, so its unobserved error is still independent of that decision. This gives the same distribution as independently flipping each queried answer.

Proof of Lemma 7. Let the truth be one cluster, and fix an order π and a flip vector ξ. Let $F = \{ e \in T _ { \pi } : \xi _ { e } = 1 \}$ be the set of flipped tree edges. For a pair $\{ u , v \}$ , let $k ( u , v )$ be the number of edges of F on the path from u to v in $T _ { \pi }$ . We show that $k ( u , v ) = 0$ implies that u and v end in the same block, and that $k ( u , v ) = 1$ implies that they end in diferent blocks. Consequently $N _ { 1 } \leq L _ { \pi } \leq N _ { \geq 1 }$ , where $N _ { j }$ (respectively $N _ { \geq j } )$ is the number of pairs with $k ( u , v ) = j$ (respectively $k ( u , v ) \geq j )$

Let $\kappa _ { t }$ be the partition of [m] into the components of the edges of $T _ { \pi }$ among the first t pairs of $\pi .$ . It equals the partition into the components of all of the first t pairs, because Kruskal’s algorithm keeps a pair exactly when it joins two components. Let $P _ { t }$ be the partition of the pipeline into blocks after t pairs.

(A) $P _ { t }$ refines $\kappa _ { t } \colon$ every pipeline block lies inside one κ<sub>t</sub>-component. We use induction on $t ;$ blocks change only by merges. A merge at step t happens on a queried pair $e = \{ u , v \}$ . After step t, u and v lie in the same κ<sub>t</sub>-component: either e joins two $\kappa _ { t - 1 }$ -components, and then $e \in T _ { \pi }$ , or u and v were already in one component. The merged block is the union of two blocks, each contained in a κ <sub>−</sub> -component by the induction hypothesis, and both of these components lie inside the $\kappa _ { t }$ -component of $u .$

(B) At every time $t ,$ every $\neq$ constraint joins two blocks that lie in the same $\kappa _ { t } \mathrm { - c o m p o n e n t }$ . A constraint is created at some step on a queried pair $\{ u , v \}$ , and from then on it joins the current block of u and the current block of $v ,$ because constraints are inherited through merges. As in $\mathrm { ( A ) }$ , u and v lie in one κ-component from that step on, since the partitions $\kappa _ { t }$ only become coarser. By (A), the current blocks of u and v lie inside this component.

(C) Every edge $e = \{ u , v \} \in T _ { \pi }$ is queried. At its step t, the points u and v lie in diferent $\kappa _ { t - 1 } { \mathrm { - c o m p o n e n t s . } }$ , by the definition of $T _ { \pi } . \mathrm { \ B y \ ( A ) }$ they lie in diferent blocks. By (B) no $\neq$ constraint joins these two blocks. So e is queried. Since the truth is one cluster, the query merges the two blocks if $\xi _ { e } = 0$ , and it creates $\mathrm { ~ a ~ } \neq$ constraint between them if $\xi _ { e } = 1$

Blocks never split. By (C), the two ends of every edge of $T _ { \pi } \ : \backslash F$ end in the same block. So every component of $T _ { \pi } \setminus F$ (a fragment) lies inside one final block. This is the first claim, because $k ( u , v ) = 0$ means that u and v lie in the same fragment. For the second claim, let $f = \{ x , y \}$ be the unique flipped edge on the path from u to $v ,$ with u on the side of x. By (C), at the step of $f \mathrm { ~ a ~ } \neq$ constraint is placed between the block of $x$ and the block of $y .$ Two blocks joined by a $\neq$ constraint are never merged, because a merge follows only a query, and a query is made only between unconstrained blocks. The constraint is inherited by every later block that contains x or $y .$ . So x and $y$ end in diferent blocks. The paths from u to x and from $y$ to v contain no flipped edge, so by the first claim u ends with x and v ends with $y .$ Hence u and v end in diferent blocks. Since the loss counts the separated pairs, the pairs with $k = 1$ are counted and the pairs with $k = 0$ are not, which gives $N _ { 1 } \leq L _ { \pi } \leq N _ { \geq 1 }$ □

Proof of Theorem 8. We show that for every order π and every $\delta \in [ 0 , 1 ]$ , with $d = d _ { T _ { \pi } } ( u , v )$ in each summand,

$$
\delta W ( T _ { \pi } ) - \delta ^ { 2 } W _ { 2 } ( T _ { \pi } ) \ \leq \ \sum _ { \{ u , v \} } d \delta ( 1 - \delta ) ^ { d - 1 } \ \leq \ \mathbb { E } _ { \xi } L _ { \pi } \ \leq \ \sum _ { \{ u , v \} } \left( 1 - ( 1 - \delta ) ^ { d } \right) \ \leq \ \delta W ( T _ { \pi } ) .
$$

In particular $\mathbb { E } _ { \xi } L _ { \pi } = \delta W ( T _ { \pi } ) + O ( \delta ^ { 2 } )$ , with remainder at most $\delta ^ { 2 } W _ { 2 } ( T _ { \pi } ) \leq \delta ^ { 2 } \binom { m } { 2 } ( m - 1 ) ^ { 2 }$ . Moreover, a single flip on a tree edge e costs exactly $n _ { e } ( m - n _ { e } )$ , and a single flip on a pair outside the tree costs nothing.

Take expectations over $\xi$ in Lemma 7. The tree $T _ { \pi }$ does not depend on $\xi ,$ so the flips on the d edges of the path from u to v are i.i.d. Bern(δ), and $k ( u , v )$ is binomial with parameters d and δ. Hence $\mathbb { P } ( k ( u , v ) = 1 ) =$ $\bar { d } \delta ( 1 - \delta ) ^ { d - 1 }$ and $\mathbb { P } ( k ( u , v ) \geq 1 ) = 1 - ( 1 - \delta ) ^ { d }$ , which gives the two middle inequalities. For the outer ones, $d \delta ( 1 - \delta ) ^ { d - 1 } \geq d \delta ( 1 - ( d - 1 ) \delta ) \geq d \delta - d ^ { 2 } \delta ^ { 2 }$ by Bernoulli’s inequality, and $1 - ( 1 - \delta ) ^ { d } \leq d \delta$ . Summing over pairs gives the bounds, and $\begin{array} { r } { W _ { 2 } ( T _ { \pi } ) \leq { \binom { m } { 2 } } ( m - 1 ) ^ { 2 } } \end{array}$ because $d \leq m - 1$ . With exactly one $\mathrm { f i p }$ , on the pair $e ,$ we have $k ( u , v ) \in \{ 0 , 1 \}$ for all pairs. I $\textit { a e } \in T _ { \pi }$ , then $L _ { \pi } = N _ { 1 }$ is the number of pairs whose tree path uses $e ,$ that is, $n _ { e } ( m - n _ { e } )$ . If $e \notin T _ { \pi }$ , then $F = \emptyset$ and $L _ { \pi } = 0 ;$ ; indeed, the pair e is never queried when all earlier answers are correct. □

With no wrong answer, the pipeline makes exactly $m - 1$ queries, namely the edges of $T _ { \pi }$ . By (C) in the proof of Lemma $7$ every tree edge is queried. $\mathrm { A }$ pair outside the tree arrives when its two ends already lie in one κ-component; with all answers correct, every tree edge so far has merged its blocks, so the blocks coincide with the κ-components and the pair is skipped.

Known results on the random minimum spanning tree. For uniform $\pi ,$ the tree $T _ { \pi }$ is distributed as the minimum spanning tree $\mathrm { M S T } _ { m }$ of $K _ { m }$ with i.i.d. continuous edge weights. We use two known results on $\mathrm { M S T } _ { m }$

(F1) (Addario-Berry et al., 2009) E diam $( \mathrm { M S T } _ { m } ) = \Theta ( m ^ { 1 / 3 } )$ , where diam is the largest tree distance between two vertices.

(F2) (Addario-Berry et al., 2017, Theorem 1.1) The tree ${ \mathrm { M S T } } _ { m } ,$ with distances multiplied by $m ^ { - 1 / 3 }$ and with the uniform probability measure on its vertices, converges in distribution, in the Gromov–Hausdorf–Prokhorov topology, to a random compact measured real tree $( \mathcal { M } , d , \mu )$ . Let $D _ { m }$ be the tree distance between two independent uniform vertices of $\mathrm { M S T } _ { m }$ . By the standard Gromov–Prokhorov consequence of this convergence, $D _ { m } / m ^ { 1 / 3 }$ converges in distribution to $D : = d ( \zeta _ { 1 } , \zeta _ { 2 } )$ , where, given $( { \mathcal { M } } , d , \mu )$ , the points $\zeta _ { 1 }$ and $\zeta _ { 2 }$ are independent with law $\mu .$ Broutin and Marckert (2023, Theorem 1.1 and Proposition $6 . 5 )$ identify the same measured limit and show that, for a point $\zeta$ sampled from $\mu ,$ almost surely $\mu ( B ( \zeta , r ) ) \le r ^ { 5 / 2 }$ for all suficiently small $r > 0$ . Thus $\mu ( \{ \zeta \} ) = 0$ almost surely. Conditional independence then gives $\mathbb { P } ( D = 0 ) = \mathbb { E } [ \mu ( \{ \zeta _ { 1 } \} ) ] = 0$ , so $\mathbb { P } ( D > 0 ) = 1$

The scale $m ^ { - 1 / 3 }$ also appears in this tree’s edge-weight-resampling noise threshold (Israeli and Peled, 2024).

Proof of Proposition 10. First, $\mathbb { E } W ( \mathrm { M S T } _ { m } ) = \Theta ( m ^ { 7 / 3 } )$ , so that $\mathbb { E } L = \delta \mathbb { E } W ( \mathrm { M S T } _ { m } ) + O ( \delta ^ { 2 } ) = \Theta ( \delta m ^ { 7 / 3 } ) +$ $O ( \delta ^ { 2 } )$ , an amplification $\mathbb { E } W ( \mathrm { M S T } _ { m } ) / { \binom { m } { 2 } } = \Theta ( m ^ { 1 / 3 } )$ over the raw rate. Second, let $\delta = x m ^ { - 1 / 3 }$ with $x > 0$ fixed, and let $\varphi _ { m } ( x ) = \mathbb { E } L / { \binom { m } { 2 } }$ be the expected fraction of separated pairs. Then

$$
\mathbb { E } \big [ x D e ^ { - x D } \big ] \ \le \ \operatorname* { l i m i n f } _ { m \to \infty } \varphi _ { m } ( x ) \ \le \ \operatorname* { l i m s u p } _ { m \to \infty } \varphi _ { m } ( x ) \ \le \ 1 - \mathbb { E } e ^ { - x D } \ \le \ x \mathbb { E } D , \qquad 0 < \mathbb { E } D < \infty .
$$

The $m ^ { 7 / 3 }$ law. For the upper bound, $W ( T ) \leq { \binom { m } { 2 } }$ diam(T) for every tree $T ,$ and (F1) gives $\mathbb { E } W ( \mathrm { M S T } _ { m } ) =$ $O ( m ^ { 7 / 3 } )$ . For the lower bound, let $D _ { m } ^ { \prime }$ be the tree distance between a uniformly random pair of distinct vertices. Then E $W ( \mathrm { M S T } _ { m } ) = \binom { m } { 2 } \mathbb { E } D _ { m } ^ { \prime }$ , and E $D _ { m } ^ { \prime } = \mathbb { E } [ D _ { m }$ | the two vertices difer] $\geq \mathbb { E } D _ { m }$ because $D _ { m } = 0$ when the two vertices coincide. By (F2), $D _ { m } / m ^ { 1 / 3 }$ converges in distribution to $D _ { \ast }$ . For nonnegative random variables, convergence in distribution implies lim inf $\mathbb { E } [ D _ { m } / m ^ { 1 / 3 } ] \geq \mathbb { E } D$ (Fatou’s lemma, or the portmanteau theorem applied to the nonnegative continuous map $t \mapsto t )$ . Since $\mathbb { P } ( D > 0 ) > 0 , \mathbb { E } D > 0$ , so $\mathbb { E } \bar { W } ( \mathrm { M S T } _ { m } ) = \Omega ( m ^ { 7 / 3 } )$ The first-order statement then follows from Theorem 8 by averaging over π: $\delta \mathbb { E } W ( \mathrm { M S T } _ { m } ) - \delta ^ { 2 } \mathbb { E } W _ { 2 } ( \mathrm { M S T } _ { m } ) \leq$ $\mathbb { E } L \leq \delta \mathbb { E } W ( \mathrm { M S T } _ { m } )$ .

The window. Here the error rate shrinks on the reciprocal scale of typical tree distances, so a typical path encounters a constant-order number of errors. By Lemma 7 and the independence of the flips from $T _ { \pi }$ , as in the proof of Theorem $8 , \varphi _ { m } ( x )$ lies between E $\left[ D _ { m } ^ { \prime } \delta ( \stackrel { \cdot } { 1 } - \delta ) ^ { D _ { m } ^ { \prime } - 1 } \right]$ and $\mathbb { E } \big [ 1 - ( 1 - \delta ) ^ { D _ { m } ^ { \prime } } \big ]$ . The laws of $D _ { m } ^ { \prime }$ and $D _ { m }$ difer only by an event of probability $1 / m$ (the two uniform vertices coincide), on which $D _ { m } = 0$ . So $D _ { m } ^ { \prime } / m ^ { 1 / 3 }$ also converges in distribution to D. Write $\left( 1 - \delta \right) ^ { D _ { m } ^ { \prime } } = \exp \left( c _ { m } D _ { m } ^ { \prime } / m ^ { 1 / 3 } \right)$ with $c _ { m } = m ^ { 1 / 3 } \ln ( 1 - x m ^ { - 1 / 3 } )  - x$ . By the continuous mapping theorem, $( 1 - \delta ) ^ { D _ { m } ^ { \prime } }$ converges in distribution to $e ^ { - x D }$ , and $D _ { m } ^ { \prime } \delta ( 1 - \delta ) ^ { D _ { m } ^ { \prime } - 1 }$ converges in distribution to $x D e ^ { - x D }$ . Both variables are bounded: the first by 1, and the second by $\mathrm { s u p } _ { y > 0 } y e ^ { - y } / ( 1 - \delta )$ because $d \delta ( 1 - \delta ) ^ { d - 1 } \leq y e ^ { - y } / ( 1 - \delta )$ with $y = - d \ln ( 1 - \delta ) \geq d \delta$ . So their expectations converge to $\mathbf { \mathbb { E } } [ x D e ^ { - x D } ]$ and $1 - \mathbb { E } e ^ { - x D }$ , which gives the two middle inequalities. The last inequality follows from $1 - e ^ { - t } \leq t .$ . Finally, $D _ { m } \leq$ diam $\left( \mathrm { M S T } _ { m } \right)$ , so Fatou’s lemma and (F1) give $\mathbb { E } D \leq$ lim in $\mathrm { f } _ { m } \mathbb { E } D _ { m } / m ^ { \mathrm { i } / 3 } \leq$ lim inf E diam $( \mathrm { M S T } _ { m } ) / m ^ { 1 / 3 } < \infty ;$ and $\mathbb { E } D > 0$ as shown above.

Interpretation. As $x  0$ both bounds equal $x \mathbb { E } D \left( 1 + o ( 1 ) \right)$ ; for the lower bound this follows by monotone convergence of $D e ^ { - x D } \uparrow D ,$ and for the upper bound by monotone convergence of $( 1 - e ^ { - x D } ) / x \mathrm { ~ \stackrel { . } { \scriptscriptstyle } ~ } D$ . Thus lim $\operatorname { i n f } _ { m } \varphi _ { m } ( x )$ and lim $\operatorname { s u p } _ { m } \varphi _ { m } ( x )$ both behave as $x \mathbb { E } D$ as $x  0 .$ . Inside the window, for each fixed $x ,$ the loss fraction is of constant order, bounded away from 0 (because $\mathbb { E } D > 0 )$ and from 1 (because $D < \infty$ almost surely). □

Whether $\varphi _ { m } ( x )$ has a limit $\Lambda ( x )$ , and whether $\Lambda ( x )  1$ or stays below 1 as $x \to \infty$ , depends on how the pipeline repairs pairs with at least two flipped edges on their path; this question is open. For a fixed query order, the loss need not increase when additional answers are flipped. Suppose that removing two tree edges $f _ { 1 } , f _ { 2 }$ from $T _ { \pi }$ leaves three fragments $P _ { 1 } , P _ { 2 } , P _ { 3 }$ , with $f _ { 1 }$ between $P _ { 1 }$ and $P _ { 2 }$ and $f _ { 2 }$ between $P _ { 2 }$ and $P _ { 3 }$ . If only $f _ { 1 }$ is flipped, Lemma 7 gives the final partition $\{ P _ { 1 } , P _ { 2 } \cup P _ { 3 } \}$ , and similarly for $f _ { 2 }$ . If both are flipped, the run can end in the partition $\{ P _ { 1 } \cup P _ { 3 } , P _ { 2 } \}$ , for instance when a pair between $P _ { 1 }$ and $P _ { 3 }$ arrives after both flipped edges; it is then queried and answered “same”. When $P _ { 2 }$ has fewer points than each of $P _ { 1 }$ and $P _ { 3 }$ , this separates fewer pairs than flipping either edge alone. So, for a fixed order, flipping more answers can decrease the loss.

Proof of Proposition 9. (i) Every spanning tree $T$ of $K _ { m }$ with $m \geq 3$ has $W ( T ) \geq ( m - 1 ) ^ { 2 }$ , with equality if and only if T is a star; hence, among all orders, the first-order risk $\delta W ( T _ { \pi } )$ of $U _ { \pi }$ is minimized exactly by the orders whose Kruskal tree is a star (for example the pivot orders). (ii) Let a decoder observe only the answers on the $m - 1$ pairs of a fixed spanning tree $T$ and output any partition, possibly at random. Let $\theta _ { 0 }$ be the one-cluster partition, and for $e \in T$ let $\theta _ { e }$ be the partition into the two components of $T - e$ . If $\delta \leq 1 / m$ , then

$$
\operatorname* { m a x } _ { \theta \in \{ \theta _ { 0 } \} \cup \{ \theta _ { e } \} _ { e \in T } } \mathbb { E } _ { \theta } L ~ \ge ~ \frac { 1 } { 2 } \delta ( 1 - \delta ) ^ { m - 2 } W ( T ) ~ \ge ~ \frac { 1 } { 2 } \delta ( 1 - \delta ) ^ { m - 2 } ( m - 1 ) ^ { 2 } ,
$$

where $\mathbb { E } _ { \theta } L$ is the expected number of pairs on which the output disagrees with θ when the truth is $\theta .$

(i) For $1 \leq n _ { e } \leq m - 1$ we have $n _ { e } ( m - n _ { e } ) \geq m - 1$ , with equality if and only if $n _ { e } \in \{ 1 , m - 1 \}$ . So $\begin{array} { r } { W ( T ) = \sum _ { e \in T } n _ { e } ( m - n _ { e } ) \geq ( m - 1 ) ^ { 2 } } \end{array}$ . Equality holds if and only if every edge has a leaf as an endpoint, which for a tree on $m \geq 3$ vertices means that $T$ is a star. An order that lists the $m - 1$ pairs $\{ 1 , j \}$ first, as in the node-ordering (pivot) pipeline of Vesdapunt et al. (2014), has the star centered at 1 as its Kruskal tree. Its first-order risk is therefore exactly $\delta ( m - 1 ) ^ { 2 }$ , an amplification $( m - 1 ) ^ { 2 } / { \binom { m } { 2 } } = 2 ( m - 1 ) / m < 2$ over the raw rate. With no wrong answer, both the pivot order and a random order make exactly $m - 1$ queries (the remark after the proof of Theorem 8).

(ii) Let $z _ { e }$ be the noiseless observation on T under $\theta _ { e } ,$ , namely “diferent” on e and “same” on the other edges of $T ;$ these observations are distinct for distinct e. Let <sup>ˆ</sup>θ be the decoder, and let $\ell ( \cdot , \cdot )$ denote the number of pairs on which two partitions disagree. Put prior weight $\frac { 1 } { 2 }$ on $\theta _ { 0 }$ and weight $\frac { 1 } { 2 ( m - 1 ) }$ on each $\theta _ { e }$ . Keeping only the observations $z _ { e } ,$ , the Bayes risk of $\hat { \theta }$ is at least

$$
\sum _ { e \in T } \Big [ \frac { 1 } { 2 } P _ { \theta _ { 0 } } ( z _ { e } ) \mathbb { E } \ell ( \hat { \theta } ( z _ { e } ) , \theta _ { 0 } ) + \frac { 1 } { 2 ( m - 1 ) } P _ { \theta _ { e } } ( z _ { e } ) \mathbb { E } \ell ( \hat { \theta } ( z _ { e } ) , \theta _ { e } ) \Big ] ,
$$

where the expectations are over the internal randomness of the decoder. Here $P _ { \theta _ { 0 } } ( z _ { e } ) = \delta ( 1 - \delta ) ^ { m - 2 }$ and $P _ { \theta _ { e } } ( z _ { e } ) = ( 1 - \delta ) ^ { m - 1 }$ . For $\delta \leq 1 / m$ we have $1 - \delta \geq ( m - 1 ) \delta$ , hence $\begin{array} { r } { \frac { ( 1 - \delta ) ^ { m - 1 } } { m - 1 } \geq \delta ( 1 - \delta ) ^ { m - 2 } } \end{array}$ . The distance ℓ is a Hamming distance on pair indicators, so $\ell ( \hat { \theta } , \theta _ { 0 } ) + \ell ( \hat { \theta } , \theta _ { e } ) \geq \ell ( \theta _ { 0 } , \theta _ { e } ) = n _ { e } ( m - n _ { e } )$ . So the Bayes risk is at least $\begin{array} { r } { \frac 1 2 \delta ( 1 - \delta ) ^ { m - 2 } \sum _ { e \in T } { n _ { e } ( m - n _ { e } ) } = \frac 1 2 \delta ( 1 - \delta ) ^ { m - 2 } W ( T ) } \end{array}$ . The maximum risk is at least the Bayes risk, and (i) gives the second inequality. Part (ii) assumes that the observed tree is fixed in advance. The pipeline $U _ { \pi }$ can change its queries in response to answers, even though its noiseless query set is $T _ { \pi }$ , so the bound does not directly apply to it. □

Similarity order. The items carry embeddings $X _ { 1 } , \ldots , X _ { m } ,$ i.i.d. with a density $f$ on $[ 0 , 1 ] ^ { d }$ that is bounded below by $f _ { 0 } > 0$ . The pairs are processed in increasing order of the Euclidean distance $| X _ { u } - X _ { v } |$ (most similar first), as in the order by decreasing match probability of Wang et al. (2013). Then $T _ { \pi }$ is the Euclidean minimum spanning tree $\mathrm { E M S T } _ { m }$ of the points, which is almost surely unique. The order is independent of the answers, so Theorem 8 gives the first-order risk $\delta W ( \mathrm { E M S T } _ { m } )$ conditionally on the embeddings. Throughout, log is the natural logarithm.

Proof of Theorem 11. Let $\mu _ { f } = \mathbb { E } | X _ { 1 } - X _ { 2 } | > 0$ . We show that for every $m \geq 2$ with $f _ { 0 } m \ge 2$ log m,

$$
\frac { \mathbb { E } W ( \mathrm { E M S T } _ { m } ) } { { \binom { m } { 2 } } } \geq \frac { \mu _ { f } - \sqrt { d } / m } { 4 \sqrt { d } } \Big ( \frac { f _ { 0 } m } { 2 \log m } \Big ) ^ { 1 / d } = c _ { d , f } \Big ( \frac { m } { \log m } \Big ) ^ { 1 / d } \big ( 1 - o ( 1 ) \big ) , \qquad c _ { d , f } = \frac { \mu _ { f } } { 4 \sqrt { d } } \Big ( \frac { f _ { 0 } } { 2 } \Big ) ^ { 1 / d } ,
$$

and that for $d = 1$ and any continuous distribution $W ( { \mathrm { E M S T } } _ { m } ) / { \binom { m } { 2 } } = ( m + 1 ) / 3$ exactly.

The proof bounds the length of the longest tree edge, then uses it to show that paths between distant points must contain many edges.

(1) Bottleneck property. For $r > 0$ let $G ( X ; r )$ be the geometric graph on the points with an edge between any two points at distance at most r. The longest edge $\ell _ { \mathrm { m a x } }$ of $\mathrm { E M S T } _ { m }$ is at most any r for which $G ( X ; r )$ is connected. Indeed, suppose that $G ( X ; r )$ is connected and that an edge e of $\mathrm { E M S T } _ { m }$ is longer than r. Removing e splits the points into two sides. Some edge of $G ( X ; r )$ crosses this cut, and exchanging it for e gives a lighter spanning tree, which is a contradiction.

(2) Tiling. Let $x = ( f _ { 0 } m / ( 2 \log m ) ) ^ { 1 / d }$ , which is at least 1 by assumption, and let $N = \lfloor x \rfloor \geq x / 2$ . Tile $[ 0 , 1 ] ^ { d }$ by $N ^ { d }$ cubes of side $1 / N$ . Each cube receives a given point with probability at least $f _ { 0 } N ^ { - d } \geq f _ { 0 } x ^ { - d } = 2 \log$ g m $, / m$ So the probability that some cube is empty is at most $N ^ { d } ( 1 - 2 \log m / m ) ^ { m } \leq x ^ { d } m ^ { - 2 } = f _ { 0 } / ( 2 m \log m ) \leq 1 / m ;$ the last step uses $f _ { 0 } \leq 1 \leq 2 \log m$ , where $f _ { 0 } \leq 1$ because $f$ integrates to 1 on the unit cube. Suppose that no cube is empty. Two points in the same cube are within distance $\sqrt { d } / N$ , and two points in face-adjacent cubes are within distance $\sqrt { d + 3 } / N \le 2 \sqrt { d } / N$ . The face-adjacency graph of the cubes is connected, and every point lies in a cube. So $G ( X ; 2 { \sqrt { d } } / N )$ is connected, and by (1), $\ell _ { \mathrm { m a x } } \leq 2 \sqrt { d } / N \leq 4 \sqrt { d } / x$ . Call this event $A ;$ we have shown $\mathbb { P } ( A ^ { c } ) \le 1 / m$

(3) Paths are long. On A, the tree path from u to v consists of $d _ { T } ( u , v )$ edges of length at most $\ell _ { \mathrm { m a x } } .$ and it covers the Euclidean distance $| X _ { u } - X _ { v } |$ . So $d _ { T } ( u , v ) \geq | X _ { u } - X _ { v } | / \ell _ { \operatorname* { m a x } } \geq x | \bar { X } _ { u } - X _ { v } | / ( 4 \sqrt { d } )$ , where $T = \mathrm { E M S T } _ { m }$ Hence

$$
\mathbb { E } W ( \mathrm { E M S T } _ { m } ) \geq \frac { x } { 4 \sqrt { d } } \mathbb { E } \Big [ \mathbf { 1 } _ { A } \sum _ { \{ u , v \} } \big | X _ { u } - X _ { v } \big | \Big ] \geq \frac { x } { 4 \sqrt { d } } \binom { m } { 2 } \Big ( \mu _ { f } - \frac { \sqrt { d } } { m } \Big ) ,
$$

because $| X _ { u } - X _ { v } | \leq \sqrt { d }$ and $\mathbb { P } ( A ^ { c } ) \le 1 / m$ . Dividing by <sup>m</sup><sub>2</sub> gives the bound.

(4) The case $d = 1$ . For distinct points on a line, which occur almost surely under a continuous distribution, the minimum spanning tree is the path through the points in sorted order. The path $P _ { m }$ has $W ( P _ { m } ) =$ ${ \begin{array} { r l } { } & { \sum _ { i = 1 } ^ { m - 1 } i ( m - i ) = { \binom { m + 1 } { 3 } } } \end{array} }$ , so $\begin{array} { r } { W ( P _ { m } ) / { \binom { m } { 2 } } = ( m + 1 ) / 3 } \end{array}$

Relative to Proposition 10, the similarity-first amplification is larger by order $m ^ { 2 / 3 }$ in $d = 1$ and at least order $m ^ { 1 / 6 } / \sqrt { \log m }$ in $d = 2$ . Thus, under these embedding assumptions and for all suficiently large m, similarity ordering within a cluster has higher first-order risk than random ordering and pivot ordering. □

For $d = 2$ , sharper amplification exponents would require finer control of planar MST path geometry; existing planar scaling results give bounds on path dimension (Aizenman et al., 1999). For $d \geq 3$ the bound $( m / \log m ) ^ { 1 / d }$ is below $m ^ { 1 / \breve { 3 } }$ and gives no comparison with random order.

Several clusters. There are K latent clusters $C _ { 1 } , \ldots , C _ { K }$ of sizes $s _ { 1 } , \ldots , s _ { K }$ , and $n = \textstyle \sum _ { i } s _ { i }$ items. All <sup>n</sup><sub>2</sub> pairs are processed in a uniformly random order π by the union-find pipeline $U _ { \pi }$ of Section A.3. A pair inside a cluster is a within pair, and a pair between two clusters is a cross pair. The answer on a within pair is flipped to a false “diferent” with probability $\varepsilon _ { w } ,$ and the answer on a cross pair is flipped to a false “same” with probability $\varepsilon _ { x } ;$ all flips are independent. The loss is the number of false-positive (FP) pairs, which the output wrongly puts together, plus the number of false-negative (FN) pairs, which the output wrongly separates. We generate the uniform order by attaching independent Unif[0, 1] arrival times to the pairs. For $N \geq 1$ and $p \in [ 0 , 1 ] , G ( N , p )$ is the Erdős–Rényi random graph on N vertices, in which each pair of vertices is an edge independently with probability p. As in Section A.3, MST<sub>s</sub> is the minimum spanning tree of $K _ { s }$ with i.i.d. continuous edge weights, and W is the Wiener index.

For sizes $s _ { A }$ and $s _ { B } .$ , consider only two clusters A and B, with all pairs of $A \cup B$ in uniformly random order. All answers are correct except one cross pair $e = \{ u , v \}$ , with $u \in A$ and $v \in B$ , which is answered $^ { 6 } \mathrm { s a m e } ^ { 9 } .$ . Let $D _ { e }$ be the loss, and let $g ( s _ { A } , s _ { B } ) = \mathbb { E } _ { \pi } D _ { e } ;$ by symmetry it does not depend on the choice of $e .$ We split $g = g _ { \mathrm { F P } } + g _ { \mathrm { F N } }$ into the expected numbers of FP and FN pairs, and we write $g ( s ) = g ( s , s )$

Proof of Theorem 12. We show that, with $\varepsilon = \operatorname* { m a x } ( \varepsilon _ { w } , \varepsilon _ { x } )$ and $N = { \binom { n } { 2 } }$ ，

$$
R _ { U } ( \varepsilon _ { w } , \varepsilon _ { x } ) = \varepsilon _ { w } \sum _ { i = 1 } ^ { K } \mathbb { E } W ( \mathrm { M S T } _ { s _ { i } } ) + \varepsilon _ { x } \sum _ { i < j } s _ { i } s _ { j } g ( s _ { i } , s _ { j } ) + O ( \varepsilon ^ { 2 } ) ,
$$

with a remainder of absolute value at most $\textstyle { \frac { 3 } { 2 } } \varepsilon ^ { 2 } N ^ { 2 } ( N - 1 )$ . For equal sizes s this reads $R _ { U } = \varepsilon _ { w } K \mathbb { E } W ( \mathrm { M S T } _ { s } ) +$ $\varepsilon _ { x } { \binom { K } { 2 } } s ^ { 2 } g ( s ) + O ( \varepsilon ^ { 2 } )$

The pipeline is exact on the noiseless input, and its output is a function of the answers and of the order, which is independent of the answers. By Lemma 1 with per-coordinate rates (end of its proof), it sufices to compute the single-error damages. We treat within-cluster and cross-cluster errors separately, since they create false splits and false merges by diferent mechanisms.

Within flip on a pair $e \subseteq C _ { i }$ . All cross answers are correct (“diferent”), so no block ever contains items of two clusters, since a merge needs $\mathrm { a \ ^ { 6 6 } s a m e ^ { 9 5 } }$ answer. $\mathrm { ~ A ~ } \neq$ constraint between two blocks of the same cluster is created only by a query between them answered “diferent”, that is, only by e. Constraints between blocks of diferent clusters never afect a decision on a within pair. Hence cluster $C _ { i }$ evolves exactly as the single-cluster pipeline under the order $\pi$ restricted to its pairs, which is uniform, with one flip; and the other clusters end intact. By Theorem 8 the damage is $n _ { e } ( s _ { i } - n _ { e } ) { \bf 1 } \{ e \in T _ { \pi | C _ { i } } \}$ , where $T _ { \pi | C _ { i } }$ is the Kruskal tree of the restricted order. The sum of these damages over the within pairs of $C _ { i }$ is $W ( T _ { \pi | \boldsymbol { C } _ { i } } )$ , whose expectation is $\mathbb { E } W ( \mathrm { M S T } _ { s _ { i } } )$

Cross flip on a pair e between $C _ { i }$ and $C _ { j }$ . Let $S = C _ { i } \cup C _ { j }$ . We use induction over time. (a) Every block is contained in S or disjoint from S, and every block disjoint from S lies inside one cluster. This holds because a merge of blocks from diferent clusters needs a “same” answer on a cross pair, and only $e \subseteq S$ has one. (b) A $\neq$ constraint between two blocks inside $S$ is created only by a query of a pair inside $S _ { ; }$ , and it is inherited only through merges inside S. Pairs with an endpoint outside $S$ only create constraints that involve a block outside $S ,$ or merge two blocks of one cluster outside $S ;$ neither changes a decision on a pair inside S. Therefore the state restricted to S (its blocks and the constraints among them) changes only at the arrivals of pairs inside $S ,$ as a deterministic function of their relative order, which is uniform. The clusters outside S end intact, as in the within case without a flip, and they are separated from everything else. So the expected loss is exactly $g ( s _ { i } , s _ { j } )$ Summing over the $s _ { i } s _ { j }$ cross pairs between $C _ { i }$ and $C _ { j }$ gives the formula.

Lemma 21 (Anatomy of one false merge). In the two-cluster setting of $^ { g , }$ let $\tau$ be the arrival time of e. Let $P$ be the component of u in the graph formed by the pairs inside A that arrive before τ, and let $Q$ be the component of v in the graph formed by the pairs inside B that arrive before τ . Let q denote the event that e is queried.

1. (Query event.) The pair e is queried if and only if no pair of $P \times Q$ other than e arrives before τ . Hence $\tilde { \mathbb { P } } ( q \mid \tau , P , Q ) = ( 1 - \bar { \tau } ) ^ { | P | | Q | - 1 }$ . If e is not queried, then $D _ { e } = 0$

2. (Final form.) If e is queried, the final partition of $A \cup B$ is $\{ X , \ A \setminus X _ { A } , \ B \setminus X _ { B } \}$ , with empty sets dropped. Here X is the final block of u, which contains u and v, and $X _ { A } = X \cap A , X _ { B } = X \cap B . W i t h \ a = | X _ { A } |$ and $b = | X _ { B } |$

$$
D _ { e } = \underbrace { a b } _ { \mathrm { F P } } + \underbrace { a ( s _ { A } - a ) + b ( s _ { B } - b ) } _ { \mathrm { F N } } .
$$

3. (Mechanism.) After τ, when a pair arrives between the current block R 6= X inside A and the block $X ,$ , and the two blocks carry no constraint, the pair is queried. If it lies inside $A ,$ it is answered “same” and R joins X. If it is a cross pair between R and $X _ { B } ,$ , it is answered $^ { \iint } d i f f e r e n t ^ { \prime \prime } \left( c o r r e c t l y \right)$ , and $R ,$ together with every block that later merges with $R ,$ stays outside X, because the 6= constraint is inherited. The same holds with A and B exchanged. $A \neq$ constraint inherited from before $\tau ,$ created by a cross pair between R and $Q$ that arrived before τ, excludes R from the start.

4. (Bounds on the damage.) On the query event, $\operatorname* { m a x } ( s _ { A } , s _ { B } ) - 1 \leq D _ { e } \leq \binom { s _ { A } + s _ { B } } { 2 }$ . For $s _ { A } = s _ { B } = s ,$ $s - 1 \leq D _ { e } \leq s ^ { 2 }$

Proof. 1. Before τ all answers are correct. Two blocks of the same cluster never carry a constraint, because all within answers are “same”. So the blocks are the components of the within-cluster graphs of the pairs that have arrived. By induction over time, a block $P ^ { \prime }$ inside A and a block $Q ^ { \prime }$ inside B carry a constraint if and only if some pair of $P ^ { \prime } \times Q ^ { \prime }$ has arrived. Indeed, when such a pair arrives, the blocks of its two endpoints (which lie inside $P ^ { \prime }$ and $Q ^ { \prime } )$ either already carry a constraint, or the pair is queried and answered “diferent”, which creates one; and constraints persist under merges. Conversely, a constraint is created only by a query on an arrived pair between the two blocks involved. At time $\tau$ the blocks of u and v are P and $Q ,$ , and e is queried if and only if they carry no constraint. Given τ and the within-cluster graphs, the $| P | | Q | - 1$ cross pairs of $P \times Q$ other than e arrive before τ independently, each with probability τ . If e is skipped, every accepted answer is correct, and the output is the truth.

2. After $\tau ,$ the only block that contains items of both clusters is X, because no other cross answer is “same”. Two blocks inside A and outside X never carry a constraint against each other: such a constraint needs a query between them answered “diferent”, but all answers on pairs inside A are “same”. So all items of A outside X end in one block, since the first arriving pair between two such blocks merges them; the same holds for $B ;$ and X never splits. The loss formula counts the pairs of $X _ { A } \times X _ { B }$ (wrongly together) and the pairs of $X _ { A } \times ( A \backslash X _ { A } )$ and $X _ { B } \times ( B \setminus X _ { B } )$ (wrongly apart).

3. This is the rule of the pipeline applied to the blocks R and X, together with the inheritance of constraints.

4. We have $a , b \geq 1$ . If $a < s _ { A }$ , then $a ( s _ { A } - a ) \geq s _ { A } - 1 ; { \mathrm { ~ i f ~ } } a = s _ { A }$ , then ab ≥ s . In both cases $D _ { e } \geq s _ { A } - 1$ and the same argument gives $D _ { e } \geq s _ { B } - 1$ . The upper bound $\binom { s _ { A } + s _ { B } } { 2 }$ is the number of pairs. For equal sizes, $a b + a ( s - a ) + b ( s - b ) = s ( a + b ) - a ^ { 2 } - b ^ { 2 } + a b \leq s ^ { 2 }$ , since on $[ 0 , s ] ^ { 2 }$ this concave function is maximized at $a = b = s$ □

Lemma 22 (Disconnection tail). For two fixed distinct vertices u and w of $G ( N , p )$ , let $h _ { N } ( p )$ be the probability that u and w lie in diferent components. Then $h _ { N } ( p ) \leq e ^ { 5 } e ^ { - N p / 2 }$ for all $N \geq 2$ and $p \in [ 0 , 1 ]$

Proof. Two vertices in diferent components cannot both lie in components with more than $N / 2$ vertices. So $h _ { N } ( p ) \leq 2 \mathbb { P } ( | C ( u ) | \leq N / 2 )$ , where $C ( u )$ is the component of u. $\mathrm { I f ~ } | C ( u ) | = k \le N / 2 .$ then some k-set $S \ni$ u contains a spanning tree of present edges and has no present edge to its complement; there are $k ( N - k ) \geq k N / 2$ pairs between S and its complement. For a fixed set S and a fixed tree on S, these two events concern disjoint sets of pairs, so they are independent. There are $\binom { N - 1 } { k - 1 }$ choices of S and, by Cayley’s formula, $k ^ { k - 2 }$ trees on S. A union bound gives

$$
\begin{array} { r } { \mathbb { P } ( | C ( u ) | = k ) ~ \le ~ \binom { N - 1 } { k - 1 } k ^ { k - 2 } p ^ { k - 1 } ( 1 - p ) ^ { k N / 2 } ~ \le ~ e ^ { k } ( N p ) ^ { k - 1 } e ^ { - k N p / 2 } , } \end{array}
$$

where we used $\binom { N - 1 } { k - 1 } k ^ { k - 2 } \leq N ^ { k - 1 } k ^ { k - 1 } / k ! \leq N ^ { k - 1 } e ^ { k }$ and $1 - p \leq e ^ { - p }$ . The k-th term equals $e e ^ { - N p / 2 } x ^ { k - 1 }$ with $x = e N p e ^ { - N p / 2 }$ . Suppose that $N p \geq 1 0$ . The map $y \mapsto y e ^ { - y / 2 }$ decreases for $y \geq 2 ,$ so $x \leq 1 0 e \cdot e ^ { - 5 } \leq 0 . 1 9 .$ and $\textstyle \sum _ { k > 1 } x ^ { k - 1 } \leq 1 / 0 . 8 1 \leq 1 . 3$ . Hence $\mathbb { P } ( | C ( u ) | \le N / 2 ) \le 1 . 3 e e ^ { - N p / 2 }$ and $h _ { N } ( p ) \leq 2 . 6 e e ^ { - N p / 2 } \leq e ^ { 5 } e ^ { - N p / 2 }$ If $N p < 1 0$ , the right-hand side $e ^ { 5 } e ^ { - N p / 2 }$ exceeds 1, and the bound is trivial. □

Proof of Proposition 13. For $s _ { A } = s _ { B } = s \ge 2$ , we bound the chance that the wrong answer is queried and then combine it with the damage conditional on that event. This gives

$$
{ \frac { 1 } { 2 s - 1 } } \leq \mathbb { P } ( q ) \leq { \frac { 1 2 . 4 } { s + 1 } } + { \frac { 1 } { 2 } } { \Big ( } { \frac { 9 } { 1 6 } } { \Big ) } ^ { s - 1 } , \qquad { \frac { s - 1 } { 2 s - 1 } } \leq g ( s ) \leq 1 3 s ,
$$

and that consequently, for equal sizes, the cross-error term $\varepsilon _ { x } \binom { K } { 2 } s ^ { 2 } g ( s )$ of Theorem 12 lies between $\textstyle { \frac { 1 } { 3 } } \varepsilon _ { x } { \binom { K } { 2 } } s ^ { 2 }$ and $1 3 \varepsilon _ { x } { \binom { K } { 2 } } s ^ { 3 }$

Lower bounds. If u is isolated in the graph of the pairs inside A that arrived before $\tau ,$ and v is isolated in the corresponding graph inside B, then $P \times Q = \{ e \}$ , and e is queried by Lemma $2 1 ( 1 )$ . Given $\tau ,$ this has probability $( 1 - \tau ) ^ { 2 ( s - 1 ) }$ , and $\begin{array} { r } { \int _ { 0 } ^ { 1 } ( 1 - \tau ) ^ { 2 ( s - 1 ) } d \tau = 1 / ( 2 s - 1 ) } \end{array}$ . By Lemma 21(1) and $( 4 ) , D _ { e } = 0$ of the query event and $D _ { e } \geq s - 1$ on it. So $g ( s ) \geq ( s - 1 ) \mathbb { P } ( q ) \geq ( s - 1 ) / ( 2 s - 1 )$

Reduction to one component. Given τ, the graphs of the pairs inside A and inside B that arrived before τ are independent copies of $G ( s , \tau )$ . So P and Q are independent, and each is distributed as the component $C _ { \tau }$ of a fixed vertex of $G ( s , \tau )$ . Since $( | P | - 1 ) ( | Q | - 1 ) \geq 0$ , we have $| P | | Q | - 1 \geq ( | P | - 1 ) + ( | Q | - 1 )$ . With Lemma 21(1) this gives

$$
\begin{array} { r } { \mathbb { P } ( q \mid \tau ) = \mathbb { E } \left( 1 - \tau \right) ^ { | P | | Q | - 1 } \ \leq \ \Big ( \mathbb { E } \left( 1 - \tau \right) ^ { | C _ { \tau } | - 1 } \Big ) ^ { 2 } . } \end{array}
$$

Extra-vertex identity. Add a vertex w to $G ( s , \tau )$ and join it to each of the s vertices independently with probability τ; the result is a copy of $G ( s + 1 , \tau )$ . The vertices u and w are connected in it if and only if w has a neighbor in $C _ { \tau } ( u )$ , the component of u in $G ( s , \tau )$ . On a simple path from u to w, the vertex w occurs only at the end; its predecessor is connected to u without w and therefore lies in $C _ { \tau } ( u )$ . Given $C _ { \tau } ( u )$ , the vertex w has no neighbor in it with probability $( 1 - \tau ) ^ { | C _ { \tau } ( u ) | }$ <sup>|</sup>. Therefore

$$
\mathbb { E } \left( 1 - \tau \right) ^ { | C _ { \tau } | } = h _ { s + 1 } ( \tau ) , \qquad \mathrm { a n d ~ s o } \qquad \mathbb { E } \left( 1 - \tau \right) ^ { | C _ { \tau } | - 1 } = \frac { h _ { s + 1 } ( \tau ) } { 1 - \tau } \quad ( \tau < 1 ) ,\tag{2}
$$

with $h _ { s + 1 }$ as in Lemma 22.

Upper bound on $\mathbb { P } ( q )$ . For $\tau \leq \frac { 1 } { 2 } , ( 2 )$ and Lemma 22 give $\begin{array} { r } { \mathbb { E } ( 1 - \tau ) ^ { | C _ { \tau } | - 1 } \le 2 e ^ { 5 } e ^ { - ( s + 1 ) \tau / 2 } } \end{array}$ . Hence $\mathbb { P } ( q \mid \tau ) \le$ min $\{ 1 , 4 e ^ { 1 0 } e ^ { - ( s + 1 ) \tau } \}$ . For $\alpha \geq 1$ and $\lambda > 0$ we have $\textstyle \int _ { 0 } ^ { \infty } \operatorname* { m i n } \{ 1 , \alpha e ^ { - \lambda \tau } \} d \tau = ( 1 + \ln \alpha ) / \lambda$ . So $\begin{array} { r } { \int _ { 0 } ^ { 1 / 2 } \mathbb { P } ( q \mid \tau ) d \tau \leq } \end{array}$ $( 1 + \ln ( 4 e ^ { 1 0 } ) ) / ( s + 1 ) = ( 1 1 + \ln 4 ) / ( s + 1 ) \leq 1 2 . 4 / ( \bar { s } + 1 )$ . For $\tau > { \frac { 1 } { 2 } }$ we use $h _ { s + 1 } ( \tau ) \leq ( 1 - \tau ) ( 1 - \tau ^ { 2 } ) ^ { s - 1 }$

This holds because, for u and w to be disconnected, the edge uw must be absent and, for each of the other $s - 1$ vertices x, the path $u ^ { - x - }$ –w must be absent; these s events are independent. So $\mathbb { E } ( 1 - \tau ) ^ { | C _ { \tau } | - 1 } \leq ( 1 - \tau ^ { 2 } ) ^ { s - 1 }$ , and $\begin{array} { r } { \mathbb { P } ( q \mid \tau ) \le ( 1 \bar { - } \tau ^ { 2 } ) ^ { 2 ( s - 1 ) } \le ( 3 / 4 ) ^ { 2 ( s - 1 ) } = ( 9 / 1 6 ) ^ { s - 1 } } \end{array}$ . Integrating over $\left( { \frac { 1 } { 2 } } , 1 \right]$ gives at most $\frac { 1 } { 2 } ( 9 / 1 6 ) ^ { s - 1 }$ . Adding the two parts proves the upper bound on $\mathbb { P } ( q )$

Upper bound on g. $\mathrm { B y }$ Lemma $2 1 ( 4 ) , D _ { e } \leq s ^ { 2 }$ on the query event, so $g ( s ) \leq s ^ { 2 } \mathbb { P } ( q ) \leq s ^ { 2 } \bigl \lceil 1 2 . 4 / ( s + 1 ) + { \frac { 1 } { 2 } } ( 9 / 1 6 ) ^ { s - 1 } \bigr \rceil$ This is at most 13s for all $s \geq 2$ . Indeed, $\begin{array} { r } { s \big [ 1 2 . 4 / ( s + 1 ) + \frac { 1 } { 2 } ( 9 / 1 6 ) ^ { s - 1 } \big ] } \end{array}$ equals about 8.83, 9.77 and 10.28 at $s = 2 , 3 , 4$ . For $s \geq 4$ it is at most $1 2 . 4 + 0 . 3 6 \mathrm { { ^ { - } < 1 3 } }$ , because $\bar { s ( 9 / 1 6 ) } ^ { s - 1 }$ decreases in s and equals about 0.71 at $s = 4$

Consequence. For $s \geq 2$ we have $\begin{array} { r } { ( s - 1 ) / ( 2 s - 1 ) \geq \frac { 1 } { 3 } } \end{array}$ , and $s ^ { 2 } g ( s ) \leq 1 3 s ^ { 3 }$ ; this gives the bounds on the cross-error term. □

Limit of the query probability. For $c > 0$ let $\theta _ { c }$ be the largest solution in $[ 0 , 1 )$ of $\theta = 1 - e ^ { - c \theta }$ ; thus $\theta _ { c } = 0$ for $c \leq 1$ and $\theta _ { c } > 0$ for $c > 1$ . For $\mu \in ( 0 , 1 )$ , the Borel(µ) law is the law of the total progeny of a Galton–Watson tree with Poisson(µ) ofspring, $\mathbb { P } ( K = k ) = e ^ { - \mu k } ( \mu k ) ^ { k - 1 } / k !$ for $k \geq 1$ . Fix $c > 0$ and let $C _ { s }$ be the component of a fixed vertex in $G ( s , c / s )$ . The next proposition uses three standard facts about the Erdős–Rényi graph (van der Hofstad, 2017, Chs. 3–4).

(E1) As $s  \infty , | C _ { s } |$ converges in distribution to the total progeny $T _ { c } \in \{ 1 , 2 , \dots \} \cup \{ \infty \}$ of a Galton–Watson tree with Poisson(c) ofspring, and $\mathbb { P } ( T _ { c } = \infty ) = \theta _ { c }$

(E2) For $c > 1$ , every $M _ { s }  \infty$ with $M _ { s } = o ( s )$ and every $\eta > 0 , \mathbb { P } \big ( | C _ { s } | > M _ { s }$ and $| | C _ { s } | - \theta _ { c } s | > \eta s )   0 :$ a component that is not small is the giant.

(E3) For $c > 1$ , the law of $T _ { c }$ conditioned on $T _ { c } < \infty$ is the $\mathrm { B o r e l } ( \mu _ { c } )$ law with the dual parameter $\mu _ { c } = c ( 1 - \theta _ { c } )$

Proposition 23 (Limit of the query probability). Assume $\left( E 1 \right) - \left( E 3 \right)$ . In the two-cluster setting with $s _ { A } = s _ { B } =$ $s ,$

$$
\operatorname* { l i m } _ { s \to \infty } s \mathbb { P } ( q ) = C _ { q } : = \int _ { 0 } ^ { \infty } f ( c ) d c , \qquad f ( c ) = \left\{ \begin{array} { l l } { 1 , } & { c \leq 1 , } \\ { ( 1 - \theta _ { c } ) ^ { 2 } + 2 \theta _ { c } ( 1 - \theta _ { c } ) \mathbb { E } e ^ { - c \theta _ { c } K } , } & { K \sim \mathrm { B o r e l } ( \mu _ { c } ) , } & { c > 1 . } \end{array} \right.
$$

Numerically, $C _ { q } = 1 . 4 2 8 6 4 ~ ( S e c t i o n ~ B )$

Proof. By Lemma 21(1), $\begin{array} { r } { \mathbb { P } ( q ) = \int _ { 0 } ^ { 1 } \mathbb { P } ( q \mid \tau ) } \end{array}$ dτ with $\mathbb { P } ( q \mid \tau ) = \mathbb { E } ( 1 - \tau ) ^ { | P | | Q | - 1 }$ , where P and Q are independent copies of the component of a fixed vertex in $G ( s , \tau )$ . The substitution $\tau = c / s$ gives $\begin{array} { r } { s \mathbb { P } ( q ) = \int _ { 0 } ^ { s } \mathbb { P } ( q \mid \tau = c / s ) } \end{array}$ dc.

Pointwise limit. Fix c. We separate the cases in which the two endpoint components remain finite, one grows to the giant component, or both do; these cases determine whether an earlier cross pair blocks the erroneous query. Fix M and split on $\{ | P | \leq M \}$ and $\{ | P | > M \}$ , and likewise for $Q ;$ we let s → ∞ first and then $M \to \infty . \mathrm { ~ B y ~ }$ (E1), $\mathbb { P } ( | P | \le M ) \to \mathbb { P } ( T _ { c } \le { \mathrm { \# } } )$ , which tends to $1 - \theta _ { c }$ as $M \to \infty ;$ by (E2), on $\{ | P | > M \}$ the component is the giant, of size $( \theta _ { c } + o _ { P } ( 1 ) ) s$ , up to an event whose probability vanishes in this double limit. The same holds for $Q ,$ , independently of P. If both are finite in the limit (always the case when $c \leq 1 )$ , then $| P | | Q |$ is bounded in probability and $( 1 - c / s ) ^ { | P | | Q | - 1 } \to 1$ . If exactly one is the giant and the other has limit size $K$ , then $( 1 - c / s ) ^ { | P | | \mathsf { \bar { Q } } | - 1 } \to e ^ { - c \theta _ { c } K }$ , and by (E3) K has the $\mathrm { B o r e l } ( \mu _ { c } )$ law. If both are giants, then $| P | | Q |$ is of order $s ^ { 2 }$ and $( { \mathrm { i } - c } / s ) ^ { | P | | Q | - 1 } \to 0 .$ . The three cases have limit probabilities $( 1 - \theta _ { c } ) ^ { 2 } , 2 \theta _ { c } ( 1 - \theta _ { c } )$ and $\theta _ { c } ^ { 2 }$ . The variables $( 1 - \dot { c } / s ) ^ { | \dot { P } | | \dot { Q } | - 1 }$ lie in [0, 1], so their expectations converge, and $\mathbb { P } ( q \mid \tau = c / s ) \to f ( c )$

Domination. For $c \leq s / 2$ , that is, for $\tau \leq \frac { 1 } { 2 }$ , the proof of Proposition 13 gives

$$
\mathbb { P } ( q | \tau = c / s ) \le \operatorname* { m i n } \{ 1 , 4 e ^ { 1 0 } e ^ { - ( s + 1 ) c / s } \} \le \operatorname* { m i n } \{ 1 , 4 e ^ { 1 0 } e ^ { - c } \} ,
$$

and the right-hand side is integrable on $[ 0 , \infty )$ . The same proof also gives $\begin{array} { r } { \int _ { s / 2 } ^ { s } \mathbb { P } ( q \mid \tau = c / s ) d c = s \int _ { 1 / 2 } ^ { 1 } \mathbb { P } ( q \mid } \end{array}$ $\tau ) d \tau \leq \frac { s } { 2 } ( 9 / 1 6 ) ^ { s - 1 } \to 0$ . Dominated convergence on $[ 0 , s / 2 ]$ finishes the proof. □

Exact small cases are in Table $5 ,$ and Monte Carlo estimates for Conjecture 15 are in Table 6.

Proof of Proposition $1 \not \angle .$ . We show that the node-ordering (pivot) pipeline of Vesdapunt et al. (2014), with K clusters of size $s ,$ has

$$
R _ { \mathrm { p i v } } = \varepsilon _ { w } K ( s - 1 ) ^ { 2 } + \varepsilon _ { x } { \binom { K } { 2 } } s ( 2 s - 1 ) + O ( \varepsilon ^ { 2 } ) .
$$

In this pipeline the items arrive in uniformly random order. A new item is compared with each existing block’s representative (its first item), following the order in which the blocks were created. It joins the first block that answers $\mathrm { { ^ { * } s a m e ^ { * } } } ;$ if all answers are “diferent”, it starts a new block.

Greedy form. Each comparison of an item with a representative settles the relation of the item to the whole block. So the pipeline is a greedy decoder on the partition code with an adaptive order. Its output is a function of the answers and of the arrival order, which is independent of the answers, and with correct answers it returns the truth. So Lemma 1, with per-coordinate rates, applies. With only one wrong answer, the run agrees with the noiseless run until that pair is compared. It therefore has the same probability of being queried as in the noiseless run.

Numbers of comparisons. Clusters are created in a uniformly random order. In the noiseless run, an item of the cluster created $\rho \mathrm { - t h }$ is compared with exactly the $\rho - 1$ earlier representatives before it reaches its own representative (or before it creates its cluster). Since $\rho$ is uniform on $\{ 1 , \ldots , K \}$ , the expected number of cros comparisons is $n ( K - 1 ) / 2$ . Every item except the K founders makes exactly one within comparison, so there are $n - K$ within comparisons.

Damage of a false merge. Suppose that the only wrong answer is a false $^ { 6 6 } \mathrm { s a m e } ^ { 9 5 }$ between an item $i \in A$ and the representative of the block of another cluster $B .$ Then i joins the block of $B$ and remains there for the rest of the run. Later items of A are compared with representatives, never with i, and the block of $B$ collects all of $B .$ So the damage is s FP pairs (i with the items of $B )$ plus $s - 1$ FN pairs (i with the other items of $A )$ , that is, $2 s - 1$ . This also holds when i is the first item of A: the next item of A then creates the block of A.

Damage of a false split. Suppose that the only wrong answer is a false “diferent” between an item i and the representative of its own block. Then i continues with the later representatives, all of which answer “diferent” correctly, and it creates a singleton block. Later items of its cluster meet the older representative first, receive $^ { 6 6 } \mathrm { s a m e } ^ { 5 5 }$ , and never reach the block of i. So the damage is s − 1 FN pairs.

Summation. By Lemma 1, $\begin{array} { r } { R _ { \mathrm { p i v } } = \varepsilon _ { x } \cdot \frac { n ( K - 1 ) } { 2 } \cdot ( 2 s - 1 ) + \varepsilon _ { w } ( n - K ) ( s - 1 ) + O ( \varepsilon ^ { 2 } ) } \end{array}$ with $n = K s$ . Since $\begin{array} { r } { \frac { n ( K - 1 ) } { 2 } = { \binom { K } { 2 } } s \mathrm { ~ a n d ~ } ( n - K ) ( s - 1 ) = K ( s - \bar { 1 } ) ^ { 2 } } \end{array}$ , this is the claim. □

## A.4 Pivot-based sorting

There are n keys, which we identify with their true ranks $1 , \ldots , n$ . The order code has one coordinate for each of the <sup>n</sup> pairs of keys, and its codewords are the n! total orders. The answer on a pair is its orientation, that is, which of the two keys is smaller. Each answer is wrong with probability $p ,$ independently over the pairs. The loss is the number of pairs that the output order inverts; this is the Hamming distance on the order code. The raw decoder has risk $p { \binom { n } { 2 } }$ . Randomized quicksort with noisy comparisons chooses a pivot q uniformly at random from the current list and compares it with every other key of the list, reading the answer on each such pair. It sends each key to the left or to the right sublist according to the answer, sorts the two sublists recursively, and outputs the left sublist, the pivot and the right sublist, in this order. We write $\textstyle H _ { n } = \sum _ { k = 1 } ^ { n } 1 / k$ for the n-th harmonic number.

Alonso et al. (2004) established $\mathbb { E } I ( n , p ) \sim n ^ { 2 } p$ for quicksort with independent comparison errors in the regime $n  \infty , p  0$ and $n p \to \infty$ (their Theorem 2.2 and Remark 2.8). Below, we derive the exact fixed-n first-order constant in Proposition 24 by counting single-error damage and using the classical comparison probabilities.

Proposition 24 (Quicksort). With comparisons wrong independently with probability $p ,$ the expected number of inversions of randomized quicksort is $p \left( n ^ { 2 } + 3 n - 2 ( n + 1 ) H _ { n } \right) + O ( p ^ { 2 } )$ , where $H _ { n }$ is the $n { - } t h$ harmonic number; the amplification tends to ${ \it 2 . }$

Proof of Proposition ${ \it 2 4 } .$ . We first identify randomized quicksort as a greedy decoder with an adaptive, pivot-first order. We then show that a single wrong comparison of key i with pivot $q$ costs $| i - q |$ inversions. Weighting

these costs by their comparison probabilities gives

$$
R _ { \mathrm { Q S } } ( p ) = p ( n ^ { 2 } + 3 n - 2 ( n + 1 ) H _ { n } ) + O ( p ^ { 2 } ) , \qquad { \frac { n ^ { 2 } + 3 n - 2 ( n + 1 ) H _ { n } } { { \binom { n } { 2 } } } } \longrightarrow 2 \quad ( n  \infty ) ,
$$

so that the amplification over the raw decoder tends to 2.

Quicksort as a greedy decoder. The accepted answers compare each chosen pivot with the keys in its current sublist. They arrange the keys into a sequence of current sublists separated by pivots. Every total order that keeps this left-to-right arrangement, and orders each current sublist arbitrarily, is consistent with all answers read so far. Hence the relation between two keys of the same current sublist is not implied, and quicksort queries such a pair if one of the two keys becomes the pivot while both are still in that sublist. After a pivot q has been compared with every key of its sublist, every pair $( l , r )$ with l and r on opposite sides of $q$ is implied by transitivity through $q ,$ and it is skipped. More generally, a pair that quicksort never compares is split by some pivot that was compared with both of its keys, so its relation is implied by transitivity. Pairs inside a sublist are processed later, in the recursion. So quicksort accepts the non-implied pairs and skips the implied ones, in an adaptive order: first the pivot against the keys of its sublist, then the pairs across the two new sublists, then recursively inside each sublist. This is the greedy decoder with an adaptive pivot-first order, and its output is the unique total order consistent with all accepted answers. Pivots are selected by i.i.d. uniform priorities on the keys (the pivot of a sublist is its key of smallest priority), which are independent of the answers; the output is a function of the answers and the priorities, and with correct answers quicksort returns the true order. So Lemma 1 applies.

Damage of a single wrong comparison. Suppose that the only wrong answer is on the comparison of key i with pivot q, say with $i < q ,$ so that i is sent to the right sublist. All other comparisons are correct. So the right sublist, which contains the keys larger than $q$ together with i, is sorted correctly, and i becomes its minimum, placed immediately after $q .$ The left sublist is also sorted correctly. The inverted pairs are $( i , q )$ and $( i , k )$ for $i < k < q$ , that is, $q - i$ pairs. The case $i > q$ is symmetric, with $i - q$ inverted pairs. An answer on a pair that quicksort does not compare is never read, and then the damage is 0.

Probability of a comparison. In the noiseless run, the keys of ranks $i < j$ are compared if and only if one of them is the first pivot chosen among the ranks $i , \ldots , j ;$ before that, all these keys stay in one sublist. By symmetry each of these $j - i + 1$ keys is equally likely to be chosen first, so the probability is $2 / ( j - i + 1 )$ . With a single wrong answer on the pair $\{ i , j \}$ , the run coincides with the noiseless run until this answer is read, so the pair is compared with the same probability.

First-order constant. By Lemma 1, the first-order constant is

$$
\sum _ { i < j } \frac { 2 } { j - i + 1 } \left( j - i \right) = n ( n - 1 ) - \sum _ { i < j } \frac { 2 } { j - i + 1 } = n ( n - 1 ) - \left[ 2 ( n + 1 ) H _ { n } - 4 n \right] = n ^ { 2 } + 3 n - 2 ( n + 1 ) H _ { n } .
$$

Here $\begin{array} { r } { \sum _ { i < j } \frac { 2 } { j - i + 1 } = \sum _ { k = 2 } ^ { n } \frac { 2 ( n + 1 - k ) } { k } = 2 ( n + 1 ) ( H _ { n } - 1 ) - 2 ( n - 1 ) = 2 ( n + 1 ) H _ { n } - 4 n ( 1 + 1 ) + 4 ( n + 1 ) H _ { n } - 1 } \end{array}$ , where $k = j - i + 1$ and there are $n + 1 - k$ pairs with $j - i + 1 = k$ . Dividing by ${ \binom { n } { 2 } } = n ( n - 1 ) / 2$ and using $H _ { n } = O ( \log n )$ shows that the amplification tends to 2. □

## B NUMERICAL VALIDATION

This appendix gives the simulation methods and results for Tables 2, 4 and 7 and Figure 2. Query orders are uniformly random unless stated otherwise. Reported uncertainties are Monte Carlo standard errors (MCSE): the sample standard deviation divided by the square root of the number of replicates. The code supplement provides the scripts, outputs and commands needed to reproduce each result; random runs use fixed seeds. Simulation run on one CPU core.

Conservation and Horn hubs. To check the conservation law, we implement the greedy decoder by enumerating all models that satisfy the theory. After each accepted answer, it retains the consistent models and skips an atom if they all agree on its value. We generate 6 implication theories (16 implications on $n = 1 2$ atoms) and 6 2-CNF theories (14 two-literal clauses, with a unit clause in every second theory). The 11 satisfiable theories contain 132 atoms, including 18 backbone atoms. We make one potential answer wrong at a time and run 600 random orders per atom. Every estimate of $\mathbb { E } _ { \pi } D _ { \pi } ( a )$ is within 2.66 standard errors of the predicted ${ \bf 1 } \{ a \notin B \}$ Here $| z | = | \mathrm { m e a n - r e f e r e n c e } | / \mathrm { M C S E } .$ with $z = 0$ when MCSE is zero.

Table 4: Analytical references and simulated damage. Rows 1–4 use exact values (rounded); rows 5–6 use first-order coeficients. Uncertainty is ±1 Monte Carlo standard error (MCSE); |z| is the standardized deviation.
<table><tr><td>Quantity</td><td>Reference</td><td>Simulated</td></tr><tr><td>Conservation, ED(a)</td><td> ${ \bf 1 } \{ a \notin B \}$ </td><td> $\operatorname* { m a x } | z | = 2 . 6 6$ </td></tr><tr><td>Horn  ${ \mathrm { h u b } } , s = 2 , k = 4$ </td><td>2.032</td><td> $2 . 0 9 5 \pm 0 . 0 3 9$ </td></tr><tr><td>Horn  ${ \mathrm { h u b } } , s = 3 , k = 3$ </td><td>2.314</td><td> $2 . 2 9 1 \pm 0 . 0 3 1$ </td></tr><tr><td> $g ( 3 ) , K = 2 , 3 , 5$ </td><td>2.0907</td><td> $2 . 0 9 9 , 2 . 1 0 6 , 2 . 0 8 3$ </td></tr><tr><td> $\bar { \mathbb { E } } \mathrm { \dot { F } P } / \varepsilon , K = 4 , s = 3$ </td><td>56.87</td><td> $5 8 . 1 \pm 1 . 1$ </td></tr><tr><td> $\mathbb { E } \mathrm { F N } / \varepsilon , K = 4 , s = 3$ </td><td>72.03</td><td> $7 1 . 9 \pm 1 . 0$ </td></tr></table>

For the Horn hub $H _ { k , s } ,$ we run 4000 orders per setting and compare the mean loss at the shared premise with Theorem 6. For $s = 2$ and $k = 1 , 2 , 3 , 4 .$ , the estimates are $1 . 3 4 8 \pm 0 . 0 1 5 , 1 . 6 2 4 \pm 0 . 0 2 4 , 1 . 8 3 0 \pm 0 . 0 3 2$ and $2 . 0 9 5 { \pm } 0 . 0 3 9 .$ , against exact values 1.333, 1.600, 1.829 and 2.032. For $s = 3$ and $k = 1 , 2 , 3$ , they are $1 . 5 2 2 \pm 0 . 0 1 4 .$ $1 . 8 9 7 \pm 0 . 0 2 3$ and $2 . 2 9 1 \pm 0 . 0 3 1$ , against 1.500, 1.929 and 2.314. We also compare the exact formula with its large-k approximation $\Gamma ( 1 + 1 / s ) k ^ { 1 - 1 / s } :$ at $k = 1 0 2 4$ , the two values are 28.377 and 28.359 for $s = 2 ,$ , and 90.790 and 90.721 for $s = 3 .$

General Horn theories. We explore Conjecture 19 beyond the hub using 12 random Horn theories with $n = 1 4 { \mathrm { ~ a t o m s } } ,$ 14 to 24 rules and at most $s = 2 \mathrm { ~ o r ~ } s = 3$ premises per rule. For each atom, we run 300 orders. The cone size $c _ { a }$ counts the newly true atoms, including a, when a false atom a is incorrectly set to true and the rules are applied. In these theories, mean damage from such a false positive never exceeds $\dot { c } _ { a } ^ { 1 - 1 / s }$ , with equality only at cone size 1. False-negative damages reach 1.48.

We also test two-level trees with rules $a \wedge b _ { i } \to c _ { i }$ and $c _ { i } \wedge d _ { i j } \to e _ { i j }$ . For $k = 2 , 4 , 6 ,$ , the mean damage of a wrong answer at a is 0.77, 0.61 and 0.57 times $\sqrt { c _ { a } }$ . In three larger random theories with at most two premises per rule $( n = 6 0 ,$ , 60 and 80), the largest ratio to $\sqrt { c _ { a } }$ is 1.36. The tested families are consistent with the square-root growth suggested by the $s = 2$ hub.

One cluster. We compute first-order amplification from the query tree as $\mathbb { E } W ( \mathrm { M S T } _ { m } ) / ( \mathbf { \Sigma } _ { 2 } ^ { m } )$ , its mean path length over all record pairs. For random orders, we generate random weight matrices and find their minimum spanning trees. At m = 50, 100, 200, 400, 800, 1600 and 3200, using 200, 100, 50, 20, 40, 40 and 40 matrices respectively, the estimates are 6.965 ± 0.053, 9.729 ± 0.104, 13.393 ± 0.248, 17.526 ± 0.423, 23.662 ± 0.428, 32.033 ± 0.645 and $4 1 . 8 8 7 \pm 0 . 7 5 1$ . The ratios to $m ^ { 1 / 3 }$ rise from 1.89 to 2.84, so these sizes do not determine the asymptotic exponent. The reference line $c m ^ { 1 / 3 }$ in Figure 2(B) uses $c = 2 . 3 7 7$ , the geometric mean of the seven ratios.

For similarity-first, we draw m uniform points in $[ 0 , 1 ] ^ { d }$ and compute the Euclidean minimum spanning tree, which is the noiseless query tree for this order. In one dimension, amplification is $( m + 1 ) / 3$ . In two dimensions, it is $1 6 . 5 2 \pm 0 . 3 0 , 4 1 . 5 0 \pm 1 . 0 6 , 9 8 . 5 2 \pm 1 . 3 9$ and $2 3 5 . 2 6 \pm 4 . 4 2$ at m = 100, 400, 1600 and 6400 (40, 20, 40 and 40 point sets). These values are 3.56, 5.63, 8.42 and 12.67 times $m ^ { 1 / 3 }$ . The largest three random-orde sizes and largest two 2D sizes each use 40 independent replicates, with seed sequences (20261006, 0, m) and (20261006, 2, m) respectively.

Two clusters. We compute the exact values of $g ( s _ { A } , s _ { B } )$ in Table 5 by a recursive calculation with rational arithmetic, caching states already visited. A state records the current blocks and the constraints between them. Under a random query order, the next pair that can change this state is uniform among the unconstrained pairs between distinct blocks.

For the Monte Carlo estimates of $g ( s )$ in Table 6, a run stops when the block containing the wrongly merged pair can no longer change; Lemma 21 ensures that the final damage is then determined. $\mathrm { A t } \ s = 2 , 3 , 4$ , the estimates are $\mathbf { 1 . 6 0 0 1 { \pm } 0 . 0 0 7 9 , 2 . 1 0 6 5 { \pm } 0 . 0 1 4 5 }$ and $2 . 4 8 6 7 { \scriptstyle \pm 0 . 0 2 1 2 }$ , close to the exact values. The column s $\mathbb { P } ( q )$ measures the chance that the potential wrong pair is actually queried, rescaled by cluster size. We evaluate Lemma 21(1) using a dynamic program for the component sizes of $G ( s , \tau )$ and Gauss–Legendre quadrature. The values approach the limiting constant $C _ { q } \approx 1 . 4 2 8 6$ in Proposition 23, computed by numerical integration. At $s = 1 0$ , 40000 orders give $g ( 1 0 ) = 4 . 4 3$ . The random-to-pivot false-merge loss ratio $g ( s ) s / ( 2 s - 1 )$ equals $2 9 / 2 7 \approx 1 . 0 7 4$ , 1.254 and 1.430 for $s = 2 , 3 , 4$ (exact) and about 2.3, 4.5 and 11.5 at $s = 1 0$ , 32 and 128 (simulation).

The estimates in Table 6 give $g ( s ) / s ^ { 2 / 3 } = 0 . 9 0 \pm 0 . 0 2$ , 0.87 ± 0.03, $0 . 8 9 \pm 0 . 0 6$ and $0 . 9 0 \pm 0 . 1 5$ at $s = 1 6 ,$ , 32, 64 and 128. They support $s ^ { 2 \mathord { \left/ { \vphantom { \left( 3 \right) 3 } } \right. \kern - delimiterspace } 3 }$ growth but do not determine the asymptotic constant.

Table 5: Exact mean damage of one false merge between two clusters of sizes $s _ { A }$ and $s _ { B } ,$ split into wrongly merged pairs $\left( g _ { \mathrm { F P } } \right)$ and wrongly separated pairs (g ), with $g = g _ { \mathrm { F P } } + g _ { \mathrm { F N } }$ . The mean is over random query orders, with zero damage when the false pair is skipped. Decimals are rounded to six places.
<table><tr><td> $\left( s _ { A } , s _ { B } \right)$ </td><td>gFP</td><td>gFN</td><td>g</td><td></td></tr><tr><td>(1,1)</td><td>1</td><td>0</td><td>1</td><td></td></tr><tr><td>(1, 2)</td><td>1</td><td> $1 / 3$ </td><td></td><td> $4 / 3$ </td></tr><tr><td>(2, 2)</td><td> $4 6 / 4 5$ </td><td> $5 3 / 9 0$ </td><td> $2 9 / 1 8 = 1 . 6 1 1 1 1 1$ </td><td></td></tr><tr><td>(1, 3)</td><td>1</td><td> $2 8 / 4 5$ </td><td> $7 3 / 4 5 = 1 . 6 2 2 2 2 2$ </td><td></td></tr><tr><td>(2, 3)</td><td>1.032959</td><td>0.830710</td><td></td><td> $1 0 5 6 7 / 5 6 7 0 = 1 . 8 6 3 6 6 8$ </td></tr><tr><td>(3, 3)</td><td>1.053091</td><td>1.037571</td><td></td><td> $3 1 3 9 1 3 / 1 5 0 1 5 0 = 2 . 0 9 0 6 6 3$ </td></tr><tr><td>(2,4)</td><td>1.037609</td><td>1.058191</td><td></td><td>2.095800</td></tr><tr><td>(3, 4)</td><td>1.064236</td><td>1.239428</td><td></td><td>2.303664</td></tr><tr><td>(4, 4)</td><td>1.081504</td><td>1.420525</td><td></td><td>2.502029</td></tr></table>

Table 6: Two clusters of size $s \mathrm { : }$ Monte Carlo estimates of $g ( s )$ (mean ±1 Monte Carlo standard error over the given number of random orders), the numerically evaluated rescaled probability $s \mathbb { P } ( q )$ that the false pair is queried, and $g / s ^ { 2 / 3 }$
<table><tr><td>S</td><td>Orders</td><td> $g ( s )$ </td><td> $s \mathbb { P } ( q )$ </td><td> $g / s ^ { 2 / 3 }$ </td></tr><tr><td>8</td><td>40000</td><td> $3 . 8 7 9 \pm 0 . 0 5 0$ </td><td>1.176</td><td>0.970</td></tr><tr><td>16</td><td>40000</td><td> $5 . 7 1 9 \pm 0 . 1 1 2$ </td><td>1.244</td><td>0.901</td></tr><tr><td>32</td><td>30000</td><td> $8 . 7 9 0 \pm 0 . 2 9 5$ </td><td>1.298</td><td>0.872</td></tr><tr><td>64</td><td>16000</td><td> $1 4 . 2 0 \pm 0 . 9 3$ </td><td>1.340</td><td>0.888</td></tr><tr><td>128</td><td>5000</td><td> $2 2 . 9 \pm 3 . 9$ </td><td>1.369</td><td>0.901</td></tr></table>

Several clusters. To test false-merge locality, we plant one false merge between two of $K = 2 , 3 , 5$ clusters of size 3 and run 20000 orders for each K. The mean damages are $\mathrm { 2 . 0 9 9 \pm 0 . 0 2 1 , 2 . 1 0 6 \pm 0 . 0 2 1 }$ and $2 . 0 8 3 \pm 0 . 0 2 1$ compared with the common prediction $g ( 3 ) = 2 . 0 9 0 7$ from Theorem 12. We then run the full pipeline on $K = 4$ clusters of size 3, with independent answer errors at rate $\varepsilon = 1 0 ^ { - 3 }$ . Over 200000 runs, E $\mathrm { \Delta } \mathrm { \cdot } \mathrm { F P } / \varepsilon = 5 8 . 1 \pm 1 . 1$ and E $\mathrm { F N } / \varepsilon = 7 1 . 9 \pm 1 . 0$ . Dividing by the error rate makes these estimates comparable to the first-order coeficients: 56.87 and 72.03, computed from the exact $g _ { \mathrm { F P } } ( 3 ) , g _ { \mathrm { F N } } ( 3 )$ and $\mathbb { E } W ( \mathrm { { M S T } _ { 3 } } ) = 4$ in Theorem 12. Because these runs allow multiple errors, the normalized expectations can difer from the first-order coeficients by $O ( \varepsilon )$

Finite noise. To test the orders when several answers may be wrong in one run, Tables 2 and $7$ use one cluster with $m = 1 0 0$ or $m = 2 0 0$ records. Every answer is independently flipped to “diferent” with probability $\delta \in \{ 0 . 0 1 , 0 . 0 2 , 0 . 0 5 , 0 . 1 , 0 . 2 \}$ . Pivot first asks all pairs $\{ 0 , j \}$ involving a fixed record and then processes the remaining pairs in random order. The random order uniformly permutes all pairs. Each cell uses 200 runs, with seeds 20260929 and 20261006 for $m = 1 0 0$ and $m = 2 0 0$ , respectively. All 2000 runs at $m = 2 0 0$ satisfy the loss bounds of Lemma 7.

Table 7: Scale check for finite noise: fraction of wrongly separated pairs in one cluster of $m = 2 0 0$ records. Mean ±1 Monte Carlo standard error over 200 runs per cell; Random/Pivot is the ratio of mean losses.
<table><tr><td>δ</td><td>Pivot</td><td>Random</td><td>Random/Pivot</td></tr><tr><td>0.01</td><td> $0 . 0 2 0 \pm 0 . 0 0 1$ </td><td> $0 . 1 0 6 \pm 0 . 0 0 9$ </td><td>5.4</td></tr><tr><td>0.02</td><td> $0 . 0 3 9 \pm 0 . 0 0 1$ </td><td> $0 . 2 0 0 \pm 0 . 0 1 2$ </td><td>5.1</td></tr><tr><td>0.05</td><td> $0 . 0 9 7 \pm 0 . 0 0 2$ </td><td> $0 . 3 6 1 \pm 0 . 0 1 0$ </td><td>3.7</td></tr><tr><td>0.10</td><td> $0 . 1 7 9 \pm 0 . 0 0 3$ </td><td> $0 . 5 0 2 \pm 0 . 0 0 7$ </td><td>2.8</td></tr><tr><td>0.20</td><td> $0 . 3 3 7 \pm 0 . 0 0 3$ </td><td> $0 . 6 4 6 \pm 0 . 0 0 5$ </td><td>1.9</td></tr></table>

## C REAL-LLM EXPERIMENT

Data and model answers. We use HPI’s 1,879-record Cora file and duplicate-pair list (Hasso Plattner Institute, n.d.). Duplicate-pair connected components define 182 entities, one per paper. Records use “field: value” pairs in the fixed order authors, title, journal, booktitle, institution, publisher, volume, pages, date and year. Qwen2.5-1.5B (base) (Qwen Team, 2024) receives six hand-written citation-pair examples (three Yes, three No; none from Cora), then the test pair with the smaller record identifier first. It answers “same” when logit( Yes) > logit( No). These next-token scores are used without sampling or fine-tuning. Similarity-firs uses cosine similarity of normalized CLS embeddings from bge-small-en-v1.5 (Beijing Academy of Artificial Intelligence, 2023).

Experimental settings. The one-entity experiment uses the 23 entities with at least 20 records: 1,266 records, sizes 21 to 238, and all 62,423 within-entity pairs. For the multi-entity experiment, we group these entities into six blocks of three or four with nearby embedding centroids, mimicking similarity blocking. This includes comparisons between records from similar but distinct entities. Each entity contributes at most 30 records; blocks contain 87 to 120 records and 36,151 pairs in total. Pivot uses the multi-entity pipeline of Proposition 14.

All query orders reuse the same saved model answers. Primary pivot and random comparisons average over 200 orders per instance. The natural/shufled error-placement control uses 30 matched order draws. Shufled errors are uniformly redistributed within each entity. Similarity-first randomly breaks cosine ties: once per entity in the primary comparison and 200 times per multi-entity block. Amplification divides final wrong-pair counts by wrong answers across all candidate pairs, including skipped pairs. The 95% bootstrap intervals resample entities (or blocks) with model answers fixed.

One-entity results. At threshold 0, final wrong-pair fractions are 8.1% for pivot, 15.3% for random order and 5.7% for similarity-first. Pivot reduces this fraction by 7.2 percentage points relative to random order (pivot-minus-random interval [−18.7, −1.4]). Similarity-first’s lower point estimate difers from pivot by −2.5 points, with interval [−9.6, 6.0] including zero. Table 8 varies the Yes–No score-diference threshold to change the answer-error rate. At every tested rate, the pivot-minus-random interval lies below zero. In the error-placement control of Table 3, the 30 matched draws give natural-placement amplifications of 1.37 (pivot), 2.19 (random) and 0.85 (similarity-first).

Table 8: Amplification in the one-entity setting as the decision threshold changes. Wrong: wrong answers as a percentage of the 62,423 pairs; 6.6% is the threshold 0 of Table 3.
<table><tr><td>Wrong</td><td>Pivot</td><td>Random</td><td>Similarity-first</td></tr><tr><td>1.1%</td><td>1.48</td><td>4.85</td><td>1.18</td></tr><tr><td>2.1%</td><td>1.40</td><td>3.75</td><td>1.05</td></tr><tr><td>5.1%</td><td>1.26</td><td>2.64</td><td>0.71</td></tr><tr><td>6.6%</td><td>1.23</td><td>2.31</td><td>0.85</td></tr><tr><td>10.0%</td><td>1.22</td><td>1.99</td><td>1.21</td></tr><tr><td>20.1%</td><td>1.20</td><td>1.61</td><td>1.00</td></tr></table>

Multi-entity results. Across the six blocks, the model gives 1,444 wrong “same” answers and 846 wrong “different” answers. The final wrong-pair counts are 2,603 for pivot, 3,347 for random order and 2,759 for similarityfirst. The pivot-minus-random loss-fraction diference is −2.1 percentage points, with interval [−3.4, −0.7]. Similarity-first produces the most false merges: 1,727 wrongly merged pairs, compared with 1,414 for pivot. Its priority for similar pairs also brings similar cross-entity pairs forward.

Exploratory runs with SmolLM2-1.7B-Instruct gave almost exclusively “diferent” answers and the same loss for all orders. DBLP–ACM entities contain only two records, leaving no within-entity propagation to compare. These runs are not used in the reported comparisons.

Assets. Qwen2.5 is released under the Apache 2.0 license and bge-small-en-v1.5 under the MIT license; the HPI page provides the Cora data with attribution. The supplementary bundle contains the cached model scores and embeddings, the analysis code and the per-instance results.