# An Exponential Deterministic–Randomized Gap in ERM-Oracle Complexity for Thresholds on an Unknown Order

Xuan Li

University of New South Wales, Sydney, Australia winny.li@unsw.edu.au ORCID: 0009-0002-0213-6991

## Abstract

Attias, Hanneke and Ramaswami (NeurIPS 2025) asked whether randomization provably reduces the oracle calls needed for online learning when the class is accessible only through an oracle. We study the instance they singled out: transductive online learning of thresholds on an unknown total order of T instances, with a consistency-type ERM oracle that returns a full concept consistent with a queried labeled set (or reports non-realizability). Our main result is a separation for a fixed natural oracle. When the oracle is the minimal-prefix rule (or the maximal-prefix rule), every deterministic learner makes M mistakes and Q calls with � + � ≥ � − � on some instance (� ∈ {0, 1}, according to whether the empty prefix is a concept), and the constant is exact; hence �(log �) mistakes cost $T - \varepsilon - O ( \log T )$ calls, whereas that paper’s randomized learner achieves �(log �) expected calls and mistakes under the same rule. The randomized order is optimal: on an explicit hard distribution under the minimal-prefix rule, every learner has expected mistakes at least ((� + 1 − �) 128<sup>−E[�]</sup> − 1)/2, so Ω(log �) expected calls are necessary for polylogarithmic mistakes. The separation is governed by the oracle’s selection rule, not by the class alone: for a legal feasible-median ERM rule a deterministic learner achieves �(log �) calls and mistakes, while a global-median rule again forces linear total cost. The same linear bound holds when the oracle’s answers are chosen adversarially and then frozen into a memoryless oracle. We add partial tradeof results for fixed query budgets (the middle regime is open) and an interface contrast: with only a weak consistency oracle, returning a realizability bit, both deterministic and randomized learners need Θ(�) calls.

## 1 Introduction

In oracle-based online learning the learner does not see the concept class directly. It interacts with the class only through an oracle, and two resources are then distinguished: the number of prediction mistakes, which measures statistical dificulty, and the number of oracle calls, which measures how much of the class the learner must actually inspect. A class of small Littlestone dimension may be learnable with few mistakes and yet require many oracle calls, and the calls themselves can be the expensive part. Which oracle interface is ofered, and how the oracle chooses among many legal answers, then becomes part of the problem specification.

Attias, Hanneke and Ramaswami [1] (henceforth AHR25) study online and transductive online learning with two such interfaces: a consistency-type ERM oracle, which given a labeled finite set � returns some concept of the class consistent with � (as a full label vector) or reports that � is not realizable, and a weak consistency oracle, which returns only the realizability bit. For the family of threshold classes on an unknown total order of the instances they prove (their Theorem 4.5) that a deterministic learner can achieve �(log �) mistakes with �(�) ERM calls, and that a randomized learner can achieve �(log �) mistakes with �(log �) expected ERM calls. They write that the latter “represents an exponential improvement over deterministic algorithms” (AHR25, §1.1); no lower bound for deterministic learners on this family is proved there, and their concluding section asks:

“Are randomized algorithms provably more powerful than deterministic ones for online learning with oracles?”

(The abstract of AHR25 states that “�(log �) ERM queries sufice” for thresholds on an unknown ordering; the algorithm in question is randomized and the bound is in expectation.)

A separation for a fixed natural oracle. Our main result answers this question for the instance the authors singled out, and it does so for a single oracle that is announced before the learner. Let the ERM oracle be the minimal-prefix rule: on a realizable sample it returns the smallest consistent prefix of the unknown order. This is arguably the most natural consistency-type ERM oracle for thresholds, it is memoryless, and it is public. Theorem 3.1 states that for every deterministic learner there are an order and a target on which the number of mistakes plus the number of calls is at least $T - \varepsilon .$ , where $\varepsilon \in \{ 0 , 1 \}$ records whether the empty prefix counts as a concept; the constant is exact, and the same holds for the maximal-prefix rule and for any sample-dependent choice between the two. Consequently a deterministic learner with ${ \cal O } ( \log T )$ mistakes must make $T - \varepsilon - O ( \log T )$ calls, whereas the randomized learner of AHR25, which works for every legal oracle, makes ${ \cal O } ( \log T )$ expected calls with ${ \cal O } ( \log T )$ mistakes under the very same rule. In the Littlestone dimension $d = \Theta ( \log T )$ of the class this is a gap of $O ( d + 1 )$ versus $2 ^ { \Omega ( d ) }$

Why a full concept does not help a deterministic learner. The oracle returns a complete label vector, so one call can reveal the relative order of many instances at once, and the sorting-based deterministic algorithm of AHR25 does exploit this. The lower bound nevertheless charges each call for at most one point. The reason is a one-pin common-value property of the extremal rules: the adversary maintains a block $L \prec F \prec R$ of instances with � freely permutable, and for every query it can move at most one point of � to an endpoint of the block so that the minimal-prefix answer is the same full vector on every remaining order. Every prediction on a still-free point is then forced to be wrong, and since only calls remove free points without a mistake, mistakes plus calls add up to the number of free points. Because the rule is fixed in advance, the final instance is simply one of the retained orders and the declared oracle needs no record of past answers.

Randomized calls are optimal in order. Theorem 3.2 shows that the randomized ${ \cal O } ( \log T )$ of AHR25 cannot be improved in order: on an explicit distribution over orders and targets, under the minimal-prefix rule, every randomized learner satisfies $\mathbb { E } [ M ] \geq ( ( T + 1 - \varepsilon ) 1 2 8 ^ { - \mathbb { E } [ Q ] } - 1 ) / 2 ,$ , so polylogarithmic expected mistakes require $\Omega ( \log T )$ expected calls. Here a single call may legitimately reveal a linear number of labels, so a one-point charging argument is unavailable; instead a logarithmic potential (“paid predictions plus remaining hidden $\mathrm { \ p o i n t s ^ { \prime \prime } ) }$ decreases by at most 7 ln 2 per call in expectation. The natural uniformrandom-order distribution provably cannot yield such a bound.

The oracle’s selection rule is a complexity axis. The linear deterministic bound is not a property of the threshold class alone. Theorem 3.4 exhibits a legal, memoryless “feasible-median” ERM rule under which a deterministic learner achieves $\lceil \log _ { 2 } T \rceil$ calls and $\lceil \log _ { 2 } T \rceil$ mistakes (for $T \geq 2 )$ , while a “global-median” rule again forces linear total cost. The exponential advantage of randomization is therefore an advantage against unfavourable selection (extremal or adversarial), and randomization substitutes for it: the randomized learner is indiferent to how the oracle selects, whereas a deterministic learner is at the mercy of the rule. We regard this as the conceptual content of the paper: for oracle-based online learning, the selection behaviour of the oracle is a complexity parameter alongside the interface and the class.

Contributions. Throughout, � is the number of instances, � the number of mistakes, $Q$ the number of oracle calls, and $\varepsilon = 0$ under convention E (the empty prefix on the instances is a concept) and $\varepsilon = 1$ under convention N (it is not); see Section 2.

1. Fixed-natural-oracle separation (Theorem 3.1). Under the pre-declared minimal-prefix rule (or the maximal-prefix rule, or any sample-dependent choice between the two), every deterministic learner has $M + Q \geq T - \varepsilon$ on some instance, and inf $\mathcal { \pi } \operatorname* { s u p } ( M + Q ) = T - \varepsilon$ exactly. Only the order and the target are chosen adversarially; the oracle precedes the learner, matching the literal reading of “an ERM oracle for $C ^ { \mathfrak { p } }$

2. Matching randomized lower bound (Theorem 3.2). On an explicit finite distribution under the minimalprefix rule, every randomized learner satisfies E $[ M ] \ge \left( ( T + 1 - \varepsilon ) 1 2 8 ^ { - \mathbb { E } [ Q ] } - 1 \right) / 2$ . With AHR25, the randomized ERM query complexity of ${ \cal O } ( \log T )$ mistakes is $\Theta ( \log T )$

3. Selection-rule dependence (Theorem 3.4). The linear bound does not extend to every legal predeclared rule: the feasible-median rule admits deterministic $\lceil \log _ { 2 } T \rceil$ calls and mistakes for $T \geq 2$ whereas the global-median rule forces $M + Q \geq T - \lfloor ( \varepsilon + T ) / 2 \rfloor$ . Every linear deterministic ERM lower bound in this paper is a statement about the extremal rules of Theorem 3.1 and about worst-case legal oracles; it is not uniform over all legal pre-declared selection rules.

Three further results support and refine the picture. Theorem 4.1 shows that the bound $M + Q \geq T - \varepsilon$ also holds when the oracle’s answers are chosen adversarially after the learner and then frozen into a memoryless oracle, which is the operational form of the lower bounds in AHR25. Theorem 7.1 gives deterministic lower and upper bounds for fixed query budgets, with exact values in the small- and large-budget regimes and finite counterexamples showing that the naive exact formula fails in between. Theorem 7.2 contrasts interfaces: with only a weak consistency oracle, both deterministic and randomized learners need $\Theta ( T )$ calls for �(log �) mistakes, so the exponential advantage of randomization is tied to the oracle returning an evaluable concept.

Techniques and what is portable. Two lower-bound templates carry the paper. The deterministic template (Section 4) is a common-value argument: a partial commitment on the order, a lemma showing that one endpoint pin per query stabilizes the oracle’s full answer, an additive charging of free points, and a fixation step that turns the adaptive construction into one fixed instance. Only the one-pin lemma uses the threshold structure and the extremal rule; the commitment, charging and fixation steps apply verbatim to any pre-declared rule with the one-pin common-value property (Appendix A, §6), and, with an answer cache, to adversarial oracles (Appendix B). The randomized template (Section 5) is a hard-distribution potential argument: a hidden-center interval-tree distribution, an auxiliary oracle that reveals strictly more than the real one, a posterior-invariance lemma, and a bound of $O ( 1 )$ on the expected decrease of a logarithmic po tential per call. The interval-tree geometry is threshold-specific; the device of paying for revealed labels with already-made predictions is not. Neither template uses the general lower-bound theorem of AHR25 (their Theorem 4.4); Remark 4.3 explains once why it does not apply here.

Organization. Section 2 fixes the model, the default convention and the quantifier forms. Section 3 states the three headline results. Sections 4 and 5 sketch the deterministic and randomized lower bounds, and Section 4 also states the adversarial-then-frozen theorem. Section 6 discusses selection-rule dependence. Section 7 states the fixed-budget tradeof bounds and the weak-consistency interface contrast. Section 8 lists open problems. Complete proofs are in Appendices A–E.

## 1.1 Related work

Mistake bounds and the transductive protocol. The mistake-bound model and the Littlestone dimension are due to Littlestone [9]; the transductive $( ^ { 6 6 } \mathrm { o f f - l i n e ^ { 3 3 } } )$ variant, in which the sequence of instances is announced in advance, was introduced by Ben-David, Kushilevitz and Mansour [10], and Hanneke, Moran and Shafer [11] show that the optimal transductive mistake bound is always Θ(1), Θ(log �) or �. In those models the learner knows the class; the number of mistakes is the only resource. For thresholds on a known order of � points the optimal mistake bound is $\lfloor \log _ { 2 } ( T + 1 - \varepsilon ) \rfloor$ , and this remains the information-theoretic benchmark here; the question studied in this paper is what it costs, in oracle calls, to reach it when the order is unknown and the class is accessible only through an oracle.

Oracle-eficient online learning. Assos et al. [4] learn online with an ERM oracle that returns a concept minimizing empirical error on a submitted sample, and Kozachinskiy and Steifer [5] with a consistent oracle that returns some concept agreeing with the examples seen so far; both obtain mistake bounds depending only on the Littlestone dimension. In those works the learner submits a sample, the oracle returns a concept, and the resource optimized is the number of mistakes; the number of calls is not the object of study. AHR25 made the number of calls itself the resource, treated the weak consistency interface alongside the ERM interface, and proved the upper bounds recalled above; earlier oracle-eficient results are surveyed there. Daskalakis and Golowich [6] study the weak consistency oracle (a realizability bit) in the PAC setting and show that eficient PAC learning is possible with it; their setting is distributional rather than online, and their oracle answers about a fixed sample rather than an adaptively grown one. Syrgkanis, Krishnamurthy and Schapire [7] obtain oracle-eficient adversarial contextual learning from an optimization oracle; the interaction between an oracle’s tie-breaking and the learner appears there too, in a diferent model.

Regret–oracle tradeofs and learning orders. In the agnostic, non-transductive setting against adaptive adversaries, Attias, Hanneke and Ramaswami [2] prove query-budget regret lower bounds (their The orem 5.1); for Littlestone dimension � = 1 and $1 \leq Q = o ( { \sqrt { T } } )$ these imply $\Omega ( T / Q )$ (their general form $\Omega ( d T / Q )$ needs $d \leq Q + 1$ and $Q = o ( { \sqrt { d T } } )$ , and $Q = 0$ is covered only by the original formula). The setting, the class and the cost measure difer from ours. Alon, Moran and Moran [3] learn an unknown linear order from counterexamples: the learner proposes a complete order and receives either confirmation or a wrongly ordered pair (with up to � untruthful answers), and the query complexity is $\Theta ( n \log n + n k )$ . Their goal is the order itself; ours is to predict threshold labels through an ERM or realizability oracle for the class, and the cost is mistakes plus calls. In the classical query models of Angluin [12] the oracle answers questions about the target (membership, equivalence); the oracles here answer questions about the class (consistency), and the target is revealed only through the online labels. Our results say that a randomized learner need not learn the order at all (�(log �) calls), whereas a deterministic one must pay a linear price against worst-case legal oracles and under the extremal rules of Theorem 3.1 (Theorems 3.1 and 4.1; not under every legal rule, Theorem 3.4). We do not claim any equivalence between these models; they are cited to place the selection behaviour of an oracle, as a design parameter of the feedback, in a broader landscape.

## 2 Model and conventions

Instances $x _ { 1 } , \ldots , x _ { T }$ are distinct and announced in advance; the adversary fixes a total order ⪯ on a finite domain � and a target concept $c _ { z } ( x ) = 1 [ x \preceq z ]$ with $z \in X ;$ the threshold class is $C _ { \preceq } = \{ c _ { z } : z \in X \}$ and the family is $\mathcal { F } = \{ C _ { \preceq } \}$ . Labels arrive one per round; the learner predicts $\hat { y } _ { t }$ before seeing $y _ { t } = c _ { z } ( x _ { t } )$ Restricted to the instances, every concept is the indicator of a ⪯-prefix.

Default convention and its correction. By default we use convention E: the domain is $X = \{ s \} \cup \{ x _ { 1 } , \ldots , x _ { T } \}$ where the sentinel � precedes all instances in every order considered, so the empty prefix on the instances is a concept; there are $T + 1$ concepts and the Littlestone dimension is $\lfloor \log _ { 2 } ( T + 1 ) \rfloor$ . Convention N, in which the domain is exactly the instance set $X = \{ x _ { 1 } , \ldots , x _ { T } \}$ (there are $T$ concepts; AHR25 write $\lfloor \log _ { 2 } T \rfloor )$ , differs from E by the bookkeeping constant �: the lower bounds of Theorems 3.1, 3.2 and 4.1 are stated with $\varepsilon \in \{ 0 , 1 \}$ and hold under both conventions, while Theorems 3.4, 7.1 and 7.2 state the two conventions explicitly. All lower bounds are proved on these finite domains, and all upper bounds hold on them. The numbers of concepts difer by one between the two conventions, while their Littlestone dimensions difer by at most one (for $T = 5$ both equal 2); we write $d = \lfloor \log _ { 2 } ( T + 1 - \varepsilon ) \rfloor$ for the Littlestone dimension, so that $T - \varepsilon \geq 2 ^ { d } - 1$ . The parallel treatment of convention N, including the sentinel bookkeeping, is confined to the appendices.

Oracles. A consistency-type ERM oracle takes any finite $S \subseteq X \times \{ 0 , 1 \}$ and returns some $c \in C _ { \preceq }$ consistent with $S ,$ observable as a full label vector, or ⊥ if none exists (AHR25, Def. 2.1, second variant). A weak consistency oracle returns only whether such a � exists (AHR25, Def. 2.2). Every call costs one, regardless of |�|; calls may be made at any time. The learner knows F, �, � and the sequence, but $\mathbf { n o t } \preceq \mathbf { o r } z .$ . A selection rule is a function $\mathcal { R } ( S , \preceq )$ that is a legal consistency-type ERM oracle for every order; the minimal-prefix rule $O _ { \operatorname* { m i n } } ( S , \preceq )$ returns the smallest consistent prefix and the maximal-prefix rule $O _ { \mathrm { m a x } }$ the largest. Degenerate queries: $S = \emptyset$ is realizable; � assigning both labels to one point is not; under E the sentinel � has label 1 in every concept.

Who chooses the oracle’s answer. The lower bounds in this paper come in three quantifier forms, which we fix here once and refer to throughout.

Fixed rule (Theorems 3.1, 3.4): the selection rule R is announced first. For every deterministic learner there   
exist an order and a target: ∀A $\exists ( \preceq , z )$ , with oracle $S \mapsto { \mathcal { R } } ( S , \preceq )$   
Adversarial then frozen (Theorem 4.1): for every deterministic learner there exist an order, a target and a   
memoryless legal oracle $O ^ { \star }$ , a function of the sample alone, on which the learner is replayed: ∀A $\exists ( \preceq , z , O ^ { \star } )$   
The oracle is produced after the learner. (AHR25 note that for deterministic learners an oblivious adversary is   
as powerful as an adaptive one; the freezing step is where this is made precise.)   
Randomized learners (Theorem 3.2): the instance (order, target, sequence, selection rule) is fixed before the   
learner’s private randomness (oblivious adversary), and costs are worst-instance expectations over that random   
ness.

The fixed-rule form is the literal reading of “an ERM oracle for $C ^ { \mathfrak { s } }$ , in which the oracle precedes the learner; the adversarial-then-frozen form is the operational standard of the lower bounds in AHR25. The randomized learner of AHR25 works for every legal oracle, so the fixed-rule form is the one that yields a separation with the oracle fixed first.

## 3 Main results

Theorem 3.1 (Deterministic lower bound; pre-declared extremal rule). Let $T \geq 1$ and fix either convention. Let R be $O _ { \operatorname* { m i n } } , o r O _ { \operatorname* { m a x } }$ (the largest consistent prefix), or any rule $\mathcal { R } _ { \eta } ( S , \preceq )$ that applies $O _ { \operatorname* { m i n } { } } o r O _ { \operatorname* { m a x } { } }$ according to afunction $\eta ( S )$ ofthe sample alone; R is announced before the learner and the instance.

(a) For every deterministic learner A and every instance sequence there exist $\preceq ^ { \star }$ and $z ^ { \star }$ such that, with oracle $S \mapsto { \mathcal { R } } ( S , \preceq ^ { \star } ) , M ( { \mathcal { A } } ) + Q ( { \mathcal { A } } ) \geq T - \varepsilon$

(b) Consequently inf ${ \bf { \dot { \mathcal { A } } } } \operatorname* { s u p } _ { \preceq , z } \left( M + Q \right) = T - \varepsilon$ (the upper bound needs no calls: under E any fixed prediction rule, under N always predicting 1).

Consequently, if a deterministic learner guarantees at most �(�) mistakes on every instance under one of these rules, its worst-case number of calls is at least $T - \varepsilon - m ( T )$ ; an ${ \cal O } ( \log T )$ -mistake guarantee costs $T { - } \varepsilon { - } O ( \log T )$ calls. The randomized learner of AHR25 (Thm. 4.5(4)) achieves ${ \cal O } ( \log T )$ expected calls with ${ \cal O } ( \log T )$ mistakes against every legal oracle, hence under the same fixed rule. In terms of $d = \lfloor \log _ { 2 } ( T + 1 -$ $\varepsilon ) \rfloor$ , the deterministic worst-case total cost is at least $T - \varepsilon \geq 2 ^ { d } - 1$ while the randomized expected total cost is $O ( d + 1 )$ . The statement concerns only the instance singled out in AHR25 (thresholds on an unknown order, consistency-type ERM oracle with full label vectors, transductive protocol, the finite domains of Section 2); it does not claim a uniform improvement for arbitrary classes or for the non-transductive setting.

Theorem 3.2 (Randomized lower bound). For every $T \geq 1$ and either convention there exist a finite distribution over (total order, target), a fixed instance sequence, and the memoryless minimal-prefix oracle $O _ { \operatorname* { m i n } } ( \cdot , \preceq )$ , such that every randomized learner satisfies

$$
\mathbb { E } [ M ] \ge \frac { ( T + 1 - \varepsilon ) \thinspace 1 2 8 ^ { - \mathbb { E } [ Q ] } - 1 } { 2 } ,
$$

the expectations being over the distribution and the learner’s randomness. In particular, $i f \operatorname { \mathbb { E } } [ Q ] \leq q$ (or $Q \leq q$ almost surely) then $\mathbb { E } [ M ] \ge \operatorname* { m a x } \{ 0 , ( ( T + 1 - \varepsilon ) 1 2 8 ^ { - q } - 1 ) / 2 \}$ , and there is afixed instance on which this holds for any learner with a uniform expected-query guarantee �.

Corollary 3.3. If a randomized learner has worst-instance expected mistakes �<sub>�</sub> and expected calls $q _ { T }$ , then $q _ { T } \geq \left( \ln ( T + 1 - \varepsilon ) - \ln ( 2 m _ { T } + 1 ) \right) / ( 7 \ln 2 ) ; f o r m _ { T } = O ( ( \log T ) ^ { d } )$ this is Ω(log �). WithAHR25 (Thm. 4.5(4)) the randomized ERM query complexity of �(log �) mistakes is Θ(log �), and with Theorems 3.1 and 4.1 and the deterministic � (�)-call algorithm of AHR25 (Thm. 4.5(3)) the deterministic one is $\Theta ( T )$ under the extremal rules of Theorem 3.1 and against worst-case legal oracles.

Theorem 3.4 (Selection-rule dependence). The statement of Theorem 3.1 does not extend to every predeclared memoryless rule. Let $\mathcal { R } _ { \mathrm { f e a s } }$ return the prefix whose cutofis the midpoint $\lfloor ( \ell + u ) / 2 \rfloor$ of the interval [ℓ, �] offeasible cutofs (and ⊥ if the interval is empty). $\mathcal { R } _ { \mathrm { f e a s } }$ is a legal memoryless consistency-type ERM oracle, andfor $T \geq 2$ there is a deterministic learner with $Q \leq \lceil \log _ { 2 } T \rceil$ and $M \leq \lceil \log _ { 2 } T \rceil$ on every instance; for � = 1 no query is needed and the learner makes at most one mistake under E and none under N (under E, $T = 1$ has two legal targets, so zero mistakes cannot be guaranteed by any learner). For the rule $\mathcal { R } _ { \mathrm { g l o b a l } }$ that returns ⊥ on unrealizable samples and otherwise projects a fixed global median cutof $m = \lfloor ( k _ { 0 } + T ) / 2 \rfloor$ onto the feasible cutof interval $[ \ell , u ]$ , i.e. returns cutof min $\{ u , \operatorname* { m a x } \{ \ell , m \} \} \ ( k _ { 0 } = \varepsilon )$ , one query and $\lceil T / 2 \rceil$ mistakes sufice, while every deterministic learner has $M + Q \ge T - m$ on some instance; so no deterministic learner achieves �(log �) mistakes and ${ \cal O } ( \log T )$ calls under $\mathcal { R } _ { \mathrm { g l o b a l } }$ . These are upper and lower bounds, not the exact minimax values ofthese rules.

Table 1 summarizes the query complexity of ${ \cal O } ( \log T )$ mistakes at the two interfaces, with the oracle quantifier made explicit.

## 4 The deterministic separation

We sketch the proof of Theorem 3.1 for $O _ { \mathrm { m i n } } ;$ the complete argument, with the reduction for $O _ { \mathrm { m a x } }$ and $\mathcal { R } _ { \eta } .$ is Appendix A.

<table><tr><td>Oracle interface</td><td>Deterministic</td><td>Randomized (expected)</td></tr><tr><td>Consistency-type ERM (returns a full concept)</td><td>Θ(T): Thm. 3.1, 4.1; AHR25 Thm. 4.5(3)</td><td>Θ(log T): AHR25 Thm. 4.5(4); Thm. 3.2</td></tr><tr><td>Weak consistency (returns a realizability bit)</td><td>Θ(T): Thm. 7.2; AHR25 Thm. 4.3</td><td>Θ(T): AHR25 Thm. 4.5(2), 4.3; Lemma E.4</td></tr></table>

Table 1: Query complexity of achieving �(log �) mistakes for thresholds on an unknown order. The deterministic ERM entry refers to the pre-declared extremal rules of Theorem 3.1 and to worst-case (adversarial) legal oracles; it is not uniform over all legal pre-declared rules: Theorem 3.4 gives a legal ERM rule under which the entry drops to �(log �).

Anchored states. Fix an anchor $\sigma$ that is constrained to be the order-minimum and hence always has label 1: under E, $\sigma = s ;$ under N, $\sigma = x _ { 1 }$ . Let $V = X \setminus \{ \sigma \}$ and $n = | V | = T - \varepsilon$ . The adversary maintains an ordered list $L ,$ an unordered set � and an ordered list � partitioning �; the retained orders are $\sigma \cdot L \cdot \pi ( F ) \cdot R $ for all permutations � of $F ,$ , and the retained targets cut anywhere through $F \colon$ every point of � is committed to label 1, every point of � to label 0, and the target may include any initial segment of the free permutation. Initially $F = V .$ . The only state changes are endpoint pins: a left pin moves $p \in F$ to the end of $L ,$ a right pin moves it to the front of $R .$ . Pins shrink the retained set (Appendix A, (A.5)).

The one-pin common-value lemma (Lemma A.2). For every state and every query $S ,$ at most one endpoint pin makes $O _ { \operatorname* { m i n } } ( S , \preceq )$ the same full vector on all retained orders. Write � for the positive and � for the negative sample points (after rejecting contradictory samples and $( \sigma , 0 )$ , whose answer is ⊥). By the prefix criterion (Lemma A.1), � is realizable in an order if no negative precedes a positive, and then the minimal prefix ends at $\operatorname* { m a x } _ { \preceq } A$ . If some $b \in B$ precedes some $a \in A$ in every retained order, answer $\perp$ without a pin. If � and � both meet $F ,$ left-pin a free negative �: every retained order now has � before every free positive, so the answer is ⊥. Otherwise the sample is realizable in every retained order and only the selected prefix must be stabilized: if � meets �, the last positive of � is max � in every retained order and its prefix $\{ \sigma \} \cup L \cup F \cup R _ { \leq p }$ is fixed (no pin); if � meets � but not �, then $B \subseteq R$ , and right-pinning a free positive $p$ makes $p = \operatorname* { m a x } A$ with prefix $\{ \sigma \} \cup L \cup F _ { \mathrm { o l d } }$ (one pin); if $A \subseteq L$ , the prefix through the last positive of � is fixed (no pin). The six cases are exhaustive. The lemma is a statement about a fixed natural rule on genuine total-order thresholds; it does not say that the returned vector carries one bit of information.

Predictions and charging (Lemmas A.3–A.4). When the learner predicts �ˆ on an instance �: if $x \in$ $\{ \sigma \} \cup L$ reveal 1, if $x \in R$ reveal 0, and if $x \in F$ reveal $1 - \hat { y }$ and pin � to the corresponding side. The transcript invariant is that every recorded answer equals $O _ { \operatorname* { m i n } } ( S , \preceq )$ on every currently retained order, that all revealed labels agree with every retained target, and that no free point has been predicted. The potential $\Phi = | \boldsymbol { F } |$ starts at �, decreases by at most one per call and decreases at a prediction only through a mistake; when all � predictions are made, Φ = 0, so $M + Q \geq n = T - \varepsilon$

Fixation without a cache (Lemma A.5). Simulate the learner until all predictions are made or until the �-th call (this covers learners that would call infinitely often). Order the residual � by arrival index and take ${ \preceq ^ { \star } } \mathbf { = } ~ \sigma \cdot L \cdot \pi _ { 0 } ( F )$ · � and $z ^ { \star }$ the last point of � (or � if $L \ = \ \varnothing ) ;$ this is a retained pair. Run the learner on $( \preceq ^ { \star } , z ^ { \star } )$ with the declared oracle $S \mapsto O _ { \operatorname* { m i n } } ( S , \preceq ^ { \star } )$ . By the invariant every recorded answer equals $O _ { \operatorname* { m i n } } ( S , \preceq ^ { \star } )$ and every recorded label equals $c _ { z ^ { \star } }$ , so by determinism the interaction replays exactly, and the bound follows. The oracle is the originally declared function: legal on every sample, memoryless, and consistent on repeated queries by definition. The upper bound in (b) needs no calls: under E any fixed predictions, under N always predicting 1. For $O _ { \mathrm { m a x } }$ an anchored reversal-complement duality (Lemma A.6) maps $O _ { \mathrm { m a x } }$ to $O _ { \operatorname* { m i n } }$ preserving mistakes and calls; for $\mathcal { R } _ { \eta } , \eta ( S )$ is fixed independently of the order, so the minimum or maximum procedure is applied at each query. More generally the argument applies to any fixed legal rule with the one-pin common-value property: on every anchored state and every sample, some refinement by zero or one endpoint pin makes the rule’s full answer constant on all retained orders. This is a suficient condition for the linear bound, not a characterization (Section 6).

## 4.1 Robustness: adversarial then frozen oracles

The bound of Theorem 3.1 is not an artefact of the particular rule: it persists when the adversary chooses the answers.

Theorem 4.1 (Deterministic lower bound; adversarial then frozen oracle). Let $T \geq 1$ andfix either convention. For every deterministic learner A there exist a total order $\preceq ^ { \star }$ on �, a point $z ^ { \star } \in X$ , and a function $O ^ { \star } : 2 ^ { X \times \{ 0 , 1 \} } \to C _ { \preceq ^ { \star } } \cup \{ \bot \}$ which is a legal consistency-type ERM oracle on every sample and depends on the sample only, such that on thisfixed instance

$$
M ( \mathcal { A } ) + Q ( \mathcal { A } ) \geq T - \varepsilon .
$$

The constant cannot be improved uniformly over $T \geq 1$

Corollary 4.2. If a deterministic learner guarantees at most �(�) mistakes on all instances and legal memoryless ERM oracles, its worst-case number ofcalls is at least $T - \varepsilon - m ( T )$ . In particular an �(log �)-mistake guarantee requires $T - \varepsilon - O ( \log T )$ calls, whereas the randomized learner ofAHR25 (Thm. 4.5(4)) achieves � (log �) expected calls with � (log �) mistakes. In terms of $d \ : = \ : \lfloor \log _ { 2 } ( T \ : + \ : 1 \ : - \ : \varepsilon ) \rfloor$ : the deterministic worst-case total cost is at least $T - \varepsilon \geq 2 ^ { d } - 1$ while the randomized expected total cost is $O ( d + 1 )$ .

The proof (Appendix B) is the same one-point charging argument with an adaptive answering rule in place of the fixed one: on a realizable query with no mixed-label free block it returns the prefix of the atom list $( L , [ F ] , R )$ through the rightmost positive atom. Its delicate part is the freezing step. Because this adaptive rule may legally return two diferent vectors to the same sample at diferent states (Appendix B exhibits such a sample), the adversary records the first answer to each sample in a cache, and the frozen oracle $O ^ { \star }$ returns cached answers on recorded samples and the minimal consistent prefix of the final order otherwise. The fixed-rule proof of Theorem 3.1 needs no cache. Theorem 4.1 has the quantifier form ∀A $\exists ( \preceq , z , O )$ of the box in Section 2: the oracle is produced after the learner and then frozen. It does not by itself yield a statement in which one oracle is fixed before all learners; that form is Theorem 3.1.

Remark 4.3 (Why Theorem 4.4 of AHR25 is not invoked). AHR25 prove a general lower bound (their Theorem 4.4) for families of all classes of a given Littlestone dimension $d _ { \mathrm { L D } }$ by an equivalence-class construction in which “each query provides information about at most one point”. Neither Theorem 3.1 nor Theorem 4.1 invokes it. Its hypothesis $T \geq 2 ^ { d _ { \mathrm { L D } } + 1 }$ fails for the threshold family, whose dimension is $d = \lfloor \log _ { 2 } ( T + 1 - \varepsilon ) \rfloor$ ; its construction uses thresholds on equivalence classes rather than on distinct totally ordered points; and its one-point property is not literally true for full-vector answers, which may encode many comparisons among already committed points. Our arguments charge at most one free point per query, which is a statement about the residual block, not about the information content of the answer. The verbatim statement and the comparison are given once, in Appendix B; the theorem is not used anywhere in this paper.

## 5 The randomized lower bound

We sketch the proof of Theorem 3.2; details are in Appendix C. Logarithms are natural and $n = T - \varepsilon$ is the number of hard points; under E the sentinel $a = s$ is the minimum of the order, and under N the remaining instance � (which arrives last) is the minimum of the order and always has label 1.

The hard distribution. Identify the hard points with $x _ { 1 } , \ldots , x _ { n }$ in arrival order and build the balanced binary interval tree on [�] (an interval of size � $\geq 2$ splits into its first $\lfloor m / 2 \rfloor$ and remaining $\lceil m / 2 \rceil$ indices). Draw a hidden center � uniformly from [�] and independent fair labels $Y _ { 1 } , \dots , Y _ { n }$ . The priority order $\pi _ { J }$ lists, along the root-to-� path, each sibling interval not containing � (in increasing index order) and finally $J ;$ thus every sibling met earlier has outer priority to everything in the current interval around �. The total order is $a \prec$ (points with $Y _ { i } = 1$ in $\pi _ { J }$ order) ≺ (points with $Y _ { i } = 0$ in reverse $\pi _ { J }$ order), and the target ends at the last $Y _ { i } = 1$ point (or at �). Then $c _ { z } ( x _ { i } ) = Y _ { i }$ : the labels are fair bits, and the geometry of the order is what the oracle can reveal. The oracle is the memoryless minimal-prefix rule; all random choices precede the interaction. This is a genuine total-order threshold class, not a class on equivalence classes.

A stronger auxiliary oracle. A minimal-prefix call is simulated by one call to a prefix-max oracle that returns the prefix ending at max of the positive sample points (Lemma C.1). We replace it by an even stronger revelation oracle that maintains an interval $I \ni J$ (initially [�]) and, on a query with positive set �, repeatedly reveals which child of � contains � together with all labels in the other child, stopping as soon as $A \backslash I$ contains a target-negative point, or $A \cap I = \emptyset$ , or � is a singleton (which it then reveals); all of this costs one call. At every stopping point the requested prefix is determined by the revealed information (Lemma C.2), so any learner for the real oracle is simulated with the same number of calls and the same predictions. A single call may thus reveal a linear number of labels; this is permitted and is exactly why a one-point charging argument is unavailable.

Posterior invariance and the potential. Conditional on the entire auxiliary history, � is uniform in the current �, the labels of unarrived points in � are independent fair bits, and they are independent of � (Lemma C.3); this replaces, rather than assumes, exchangeability after ⊥ answers, and it holds for randomized learners after conditioning on the seed. Call a prediction unrevealed if its point is still in � when it is made; such a prediction faces an independent fair bit and costs $1 / 2$ in expectation, so $\begin{array} { r } { \mathbb { E } [ M ] \geq \frac { 1 } { 2 } \mathbb { E } [ H _ { \mathrm { f i n a l } } ] } \end{array}$ where � counts unrevealed predictions made so far. With � the number of not-yet-predicted points in �, the potential is

$$
\Phi = \ln ( H + r + 1 ) .
$$

Prediction steps leave Φ unchanged (an unrevealed prediction moves one unit from � to �), and one binary refinement of � decreases Φ by at most ln 2, because already-predicted points inside � have been paid for in � (Lemma C.4). Revealing the label of an already-predicted point costs nothing in Φ, because that point has been paid for in �; only unpredicted points inside the retained interval count in �.

One call contracts the potential by �(1) (Lemma C.5). Conditional on any history before a call, the expected decrease of Φ during that call is at most 7 ln 2, whatever the query set. Call a state early if fewer than $\lfloor m / 2 \rfloor$ of the � points of � have arrived. In an early state every point of the right child is unarrived, so its labels are fresh fair bits. If the query avoids the right child, the call stops when � lies there, which has probability at least $1 / 2 ;$ if it meets the right child, then with probability at least ${ \frac { 1 } { 3 } } \cdot { \frac { 1 } { 2 } }$ the center is in the left child and some queried point of the right child is target-negative, which also stops the call. Hence each early refinement continues with probability at most ${ 5 / 6 } ,$ the expected number of early refinements is at most 6, and from the first late state on the remaining decrease is at most ln 2 since $r \leq H + 1$ there. Summing over calls, $\ln ( n + 1 ) - \mathbb { E } [ \ln ( H _ { \mathrm { f i n a l } } + 1 ) ] \leq 7 \ln 2$ · E[�] for an adaptive random number of calls, and Jensen’s inequality gives $\mathbb { E } [ H _ { \mathrm { f i n a l } } ] \geq ( n + 1 ) 1 2 8 ^ { - \mathbb { E } [ Q ] } - 1$ , hence Theorem 3.2. The fixed-instance clause follows because the distribution has finite support.

Why the uniform distribution cannot work. Under a uniformly random order and cutof (convention E), predicting the first � labels by posterior majority and then making one call on all � observed labels achieves $\mathbb { E } [ M ] \le F ( m ) + 2 ( T - m ) / ( m + 2 )$ for every consistent returned threshold, where $F ( m ) \leq m / 2$ is the exact zero-query optimum (Propositions $C . 6 – C . 7 )$ ; with $m = \lceil { \sqrt { T } } \rceil$ this is $O ( { \sqrt { T } } )$ after one call, so no bound of the form $T \rho ^ { Q }$ with $Q = 1$ can hold there. Moreover no history-uniform constant contraction bound holds on that distribution even with +1 regularization (Proposition C.8). The hidden-center distribution above is designed so that each call decreases the paid potential by �(1) in expectation, however many levels it descends.

## 6 Oracle selection matters

Theorem 3.4 is proved in Appendix A, §8; we describe the two rules and draw the interpretation.

The feasible-median rule admits deterministic logarithmic learning. Index concepts by the number � of positive instances; for every sample the feasible cutofs form an integer interval $[ \ell , u ]$ (or are empty), and $\mathcal { R } _ { \mathrm { f e a s } }$ returns the prefix of cutof $\lfloor ( \ell + u ) / 2 \rfloor$ . It depends only on $( S , \preceq )$ and knows nothing about the learner. The learner (Proposition A.7) maintains $H _ { 1 } \prec U \prec H _ { 0 }$ , where $H _ { 1 }$ is known to be positive, $H _ { 0 }$ known to be negative and � the uncertain interval, initially all instances. While $\left| U \right| \geq 2$ it queries $( H _ { 1 } \times \{ 1 \} ) \cup ( H _ { 0 } \times \{ 0 \} )$ the returned concept ℎ splits � into $U _ { 1 } = \{ h = 1 \}$ and $U _ { 0 } = \{ h = 0 \}$ with $1 \leq | U _ { 0 } | , | U _ { 1 } | \leq \lceil | U | / 2 \rceil$ , because the feasible interval is exactly $[ | H _ { 1 } | , | H _ { 1 } | + | U | ]$ (under N with $H _ { 1 } = \varnothing$ it is $[ 1 , | U | ]$ ). It predicts with ℎ until an error. An error with $h ( x ) = 0 , y = 1$ certifies every point of $U _ { 1 }$ positive and replaces � by $U _ { 0 } ;$ an error with $h ( x ) = 1 , y = 0$ certifies $U _ { 0 }$ negative and replaces � by $U _ { 1 }$ . Each error halves � (rounded up), so there are at most $\lceil \log _ { 2 } T \rceil$ errors and at most as many calls. Past labels inside � need not be included in the query: the interface allows arbitrary samples. $\mathbf { A } \mathbf { t } T = 8$ this learner has $M + Q \leq 6 < T - \varepsilon$ on every instance, which refutes the universal extension of Theorem 3.1 with the required quantifiers.

The global-median rule does not. Let $m = \lfloor ( k _ { 0 } + T ) / 2 \rfloor$ with $k _ { 0 } = \varepsilon$ and let $\mathcal { R } _ { \mathrm { g l o b a l } }$ return the feasible cutof closest to � (formally min $\{ u , \operatorname* { m a x } \{ \ell , m \} \}$ , and ⊥ on unrealizable samples). One empty query returns a prefix with � positive instances; predicting with it until the first error and arbitrarily thereafter on the uncertain side gives $Q = 1$ and $M \leq \lceil T / 2 \rceil$ , so this rule also violates the full bound $T - \varepsilon$ . But it admits no deterministic �(log �)-call, �(log �)-mistake learner: fix an ordered block � of � instances, revealed to the learner and labeled positive, before a block � of $T - m$ instances in unknown order. On such instances every call to $\mathcal { R } _ { \mathrm { g l o b a l } }$ can be simulated by at most one anchored $\bar { O } _ { \operatorname* { m i n } }$ call on $U$ (whenever the sample has a positive in � and no negative in �, every feasible cutof exceeds � and the rule returns the minimal one), so Theorem 3.1 on � gives $M + Q \geq T - m \geq \lfloor T / 2 \rfloor$ on some instance.

Interpretation. Three legal, memoryless, pre-declared rules for the same class and the same interface have qualitatively diferent deterministic complexities: the extremal rules have exact worst-case total cost $T - \varepsilon ;$ the global-median rule has linear total cost but not the full constant; the feasible-median rule admits ${ \cal O } ( \log T )$ calls and mistakes. The randomized learner of AHR25 is oblivious to the rule. In this precise sense randomization substitutes for a favourable selection rule, and the selection rule is a complexity axis of oracle-based online learning, alongside the interface (Section 7) and the class. We do not define a numerical measure of “helpfulness” of a rule; the one-pin common-value property of Section 4 is a suficient condition for the linear bound, and Section 8 records what a characterization would have to explain. In particular, whether a rule returns an endpoint of the feasible interval does not decide the matter: neither median rule always returns an endpoint, yet their complexities difer.

## 7 Consequences: fixed budgets and the weak consistency interface

Deterministic tradeofs at a fixed query budget. Against adversarial oracles, one may ask for the exact minimax number of mistakes with at most � calls. We obtain bounds, exact values in the small- and largebudget regimes, and finite counterexamples to the natural conjecture that the lower bound is exact; the middle regime is open.

Theorem 7.1 (Deterministic mistake–query bounds). Let $M _ { E } ^ { * } ( T , Q )$ (resp. $M _ { N } ^ { * } )$ be the minimax number of mistakes of deterministic learners making at most � ERM calls, against an adversarial oracle as in Theorem 4.1.

(i) max $\{ T - Q , \lfloor \log _ { 2 } ( T + 1 ) \rfloor \} \le M _ { E } ^ { * } ( T , Q ) \le$ max $\{ T - Q , \lceil T / 2 \rceil \}$ ; under N, $M _ { N } ^ { * } ( T , 0 ) = M _ { N } ^ { * } ( T , 1 ) = T { - } 1$ E and for $Q \geq 1$ , max $\{ T - Q , \lfloor \log _ { 2 } T \rfloor \} \le M _ { N } ^ { * } ( T , Q ) \le$ min $\{ T - 1$ , max $\{ T - Q , \lceil T / 2 \rceil \} \}$ }. (The bound $T - Q$ under N is a budget statement $f o r \ Q \ \ge \ 1$ , proved by a first-action recurrence $M _ { N } ^ { * } ( n , q ) \ \geq$ min $\{ M _ { E } ^ { * } ( n - 1 , q - 1 ) , \ : 1 + M _ { E } ^ { * } ( n - 1 , q ) , \ : 1 + M _ { N } ^ { * } ( n - 1 , q ) \} f o r \ : n \geq 2 , \ : q \geq 1$ and induction; it is not a pathwise statement.)

(ii) Fo $\begin{array} { r } { \cdot Q \leq \lfloor T / 2 \rfloor , M _ { E } ^ { * } ( T , Q ) = T - Q ; f o r Q \geq 2 ( T - 1 ) , M _ { E } ^ { * } ( T , Q ) = \lfloor \log _ { 2 } ( T + 1 ) \rfloor a n d M _ { N } ^ { * } ( T , Q ) = } \end{array}$ $\lfloor \log _ { 2 } T \rfloor$ (this budget is suficient, not necessary).

(iii) The lower bound in (i) is not the exact value: $M _ { E } ^ { * } ( 5 , 3 ) = 3 a n d M _ { N } ^ { * } ( 6 , 4 ) = 3 .$

The lower bounds combine Theorem 4.1 with the known-order lower bound $\left\lfloor \log _ { 2 } ( T + 1 - \varepsilon ) \right\rfloor$ (Lemma D.10); the upper bounds come from a protection algorithm whose accounting is corrected in Appendix D, and from sorting with $2 ( T - 1 )$ calls followed by majority prediction. The counterexamples are established analytically: a five-point game analysis under E, and a first-action recurrence reducing the N case to it (Appendix D).

Interface contrast. The exponential advantage of randomization at the ERM interface depends on a successful call returning an evaluable concept. When the oracle returns only the realizability bit, the advantage disappears at the level of query order: both deterministic and randomized learners need $\Theta ( T )$ calls for ${ \cal O } ( \log T )$ mistakes.

Theorem 7.2 (Weak consistency oracle). With only the weak consistency oracle, there is a deterministic learner that, on every instance, makes at most 66� calls and $\lfloor \log _ { 2 } ( T + 1 ) \rfloor$ mistakes under $E ,$ and at most $6 7 ( T - 1 )$ calls and $\lfloor \log _ { 2 } T \rfloor$ mistakes under N. Conversely, for $T - \varepsilon \geq 1 0 0$ : any learner—deterministic or randomized—that makes at most $( T - \varepsilon ) / 2 0$ calls on every run (a hard cap) incurs at least $( T - \varepsilon ) / 2 0$ expected mistakes on some instance (AHR25, Thm. 4.3, applied with efective horizon $T - \varepsilon ) _ { : }$ , and any learner with worst-instance expected mistakes � and worst-instance expected calls � satisfies $m + 2 0 q \geq ( T - \varepsilon ) / 2 0$ (Lemma E.4 in Appendix E). Hence, at this interface, Θ(�) calls are necessary and suficient for �(log �) mistakesfor both deterministic and randomized learners.

Remark 7.3. The randomized half of Theorem 7.2 restates AHR25 (Thm. 4.5(2) and 4.3); the deterministic upper bound is a modest improvement over the �(� log �) of AHR25 (Thm. 4.5(1)) whose linear order can also be obtained by using deterministic linear-time selection [8] in place of a random pivot (this is a statement about this specific algorithm, not a claim that arbitrary randomized oracle algorithms can be derandomized for free), and its content is the explicit constants and the exact integer mistake bound; the selection procedure, the constants and the convention correction are in Appendix E. Lemma E.4 (a truncation lemma for expected budgets) is ours. The constant 128 in Theorem 3.2 is not optimized. The finite counterexamples in Theorem 7.1 are established analytically.

## 8 Discussion and open problems

The results answer the question of AHR25 for thresholds on an unknown order, in the transductive protocol and with the consistency-type ERM interface, in the following sense: for a fixed natural oracle the deterministic query complexity of � (log �) mistakes is at least $T - \varepsilon - O ( \log T )$ , hence Θ(�), and the randomized one is Θ(log �); the same linear bound holds against adversarial memoryless oracles; and the gap is governed by the oracle’s selection rule. The general question, for arbitrary classes and in the non-transductive setting, remains open, as do the following.

• Characterize favourable selection rules. Determine which pre-declared ERM rules permit deterministic logarithmic-query, logarithmic-mistake learning. The one-pin common-value property (Section 4) is a suficient condition for a linear lower bound, satisfied by the extremal rules; endpoint versus non-endpoint selection alone does not characterize the complexity, since neither the feasible-median rule nor the global-median rule always returns an endpoint of the feasible interval, yet the former admits deterministic �(log �) calls and mistakes while the latter forces linear total cost (Theorem 3.4). A parameter measuring how much a rule can shrink the feasible set in one call, related to the deterministic mistake–query complexity even one-sidedly, would turn the three examples into a theorem.

• The middle regime. Determine $M _ { E } ^ { * } ( T , Q )$ for $\lfloor T / 2 \rfloor < Q < 2 ( T - 1 )$ ; the value is not $\operatorname* { m a x } ( T -$ �, $\lfloor \log _ { 2 } ( T + 1 ) \rfloor )$ (Theorem 7.1(iii)).

• Constants. Improve the constant 128 in Theorem 3.2.

• Beyond thresholds. Extend the separation to arbitrary Littlestone classes and to the non-transductive setting, which is the general form of the open question of AHR25. The charging and fixation steps of Section 4 and the paid-potential device of Section 5 are not threshold-specific; the one-pin lemma and the interval-tree distribution are.

## References

[1] I. Attias, S. Hanneke, A. Ramaswami. Tradeofs between mistakes and ERM oracle calls in online and transductive online learning. NeurIPS 2025; arXiv:2506.00135.

[2] I. Attias, S. Hanneke, A. Ramaswami. Regret–oracle complexity tradeofs in agnostic online learning. arXiv:2605.07155, 2026.

[3] Noga Alon, Shay Moran, Shlomo Moran. Sorting from counterexamples. arXiv:2608.21579, 2026.

[4] A. Assos, I. Attias, Y. Dagan, C. Daskalakis, M. K. Fishelson. Online learning and solving infinite games with an ERM oracle. COLT 2023, PMLR 195:274–324.

[5] A. Kozachinskiy, T. Steifer. Simple online learning with consistent oracle. COLT 2024, PMLR 247:3241–3256.

[6] C. Daskalakis, N. Golowich. Is eficient PAC learning possible with an oracle that responds “yes” or “no”? COLT 2024, PMLR 247:1263–1307.

[7] V. Syrgkanis, A. Krishnamurthy, R. E. Schapire. Eficient algorithms for adversarial contextual learning. ICML 2016, PMLR 48:2159–2168.

[8] M. Blum, R. W. Floyd, V. Pratt, R. L. Rivest, R. E. Tarjan. Time bounds for selection. J. Comput. Syst. Sci. 7:448–461, 1973.

[9] N. Littlestone. Learning quickly when irrelevant attributes abound: a new linear-threshold algorithm. Machine Learning 2:285–318, 1988.

[10] S. Ben-David, E. Kushilevitz, Y. Mansour. Online learning versus ofline learning. Machine Learning 29:45–63, 1997.

[11] S. Hanneke, S. Moran, J. Shafer. A trichotomy for transductive online learning. NeurIPS 2023; arXiv:2311.06428.

[12] D. Angluin. Queries and concept learning. Machine Learning 2:319–342, 1988.

## A Proof of Theorem 3.1 and Theorem 3.4

## 1. Statement and conventions

All lemmas below are proved here. We use the consistency-ERM interface of Section 2, including full-vector returns and unrestricted finite query samples. No weak-consistency interface is used.

For a finite ordered domain, write:

$O _ { \mathrm { m i n } } \mathrm { . }$ return the smallest consistent prefix, $\mathrm { { o r } \perp }$ if none exists;

$O _ { \mathrm { m a x } } ;$ : return the largest consistent prefix, or ⊥ if none exists.

Under convention E, the domain is

$$
X = \{ s , x _ { 1 } , . . . , x _ { T } \} , \qquad s \prec x \quad ( x \neq s ) .
$$

Thus every concept labels � by 1, and the smallest prefix is $\{ s \}$ . This is the rule fixed in Theorem 3.1.

Under convention N, the domain is $X = \{ x _ { 1 } , \ldots , x _ { T } \}$ , and prefixes must be nonempty. In particular, when a query has no positive labels, $O _ { \operatorname* { m i n } }$ returns the singleton containing the order-minimum if that singleton is consistent, and otherwise returns ⊥. This is the appropriate nonempty-prefix version of the same rule.

## Statement of Theorem 3.1 (fixed extremal ERM rules)

Fix either $\mathcal { R } = O _ { \operatorname* { m i n } } \ : \mathrm { o r } \ : \mathcal { R } = O _ { \operatorname* { m a x } }$ , publicly and before choosing the learner or instance. For every deterministic learner ${ \mathcal { A } } .$ , and every prescribed sequence of distinct instances $x _ { 1 } , \ldots , x _ { T }$ , there exist a total order $\preceq ^ { \star }$ and an endpoint $z ^ { \star } \in X$ such that

$$
M \big ( \mathcal { A } , \mathcal { R } ( \cdot , \preceq ^ { \star } ) , x _ { 1 : T } , c _ { z ^ { \star } } \big ) + Q \big ( \mathcal { A } , \mathcal { R } ( \cdot , \preceq ^ { \star } ) , x _ { 1 : T } , c _ { z ^ { \star } } \big ) \geq T - \varepsilon ,
$$

where $\varepsilon = 0$ under E and $\varepsilon = 1$ under N.

The oracle is the originally declared function ${ \mathcal { R } } ;$ no return-value table is selected after constructing the instance.

The statement also holds for every predeclared sample-dependent extremal selector

$$
\mathcal { R } _ { \eta } ( S , \preceq ) = \left\{ \begin{array} { l l } { O _ { \operatorname* { m i n } } ( S , \preceq ) , } & { \eta ( S ) = 0 , } \\ { O _ { \operatorname* { m a x } } ( S , \preceq ) , } & { \eta ( S ) = 1 , } \end{array} \right.\tag{A.1}
$$

where � depends on �, not on the unknown order or interaction history.

We first prove the theorem for $O _ { \operatorname* { m i n } }$

## 2. Anchored states

Use a distinguished anchor �, which is constrained to be the order-minimum and therefore always has target label 1.

• Under E, take $\sigma = s$ . It is not an online instance.

• Under N, take $\sigma = x _ { 1 }$ . It is an online instance whose label is fixed to 1.

Let

$$
V = X \setminus \{ \sigma \} , \qquad n = | V | = T - \varepsilon .
$$

The construction maintains ordered lists �, � and an unordered set �, partitioning �. Its retained orders are

$$
\Omega ( L , F , R ) = \left\{ \sigma \cdot L \cdot \pi ( F ) \cdot R : \pi ( F ) \mathrm { i s  a n y p e r m u t a t i o n } \mathrm { o f } F \right\} .\tag{A.2}
$$

The retained order–target pairs are

$$
\Gamma ( L , F , R ) = \left\{ \big ( \sigma \cdot L \cdot \pi ( F ) \cdot R , { \bf 1 } _ { \{ \sigma \} \cup L \cup \{ \pi _ { 1 } , \dots , \pi _ { k } \} } \big ) : 0 \leq k \leq | F | \right\} .\tag{A.3}
$$

Thus:

• every point of � is committed to target label 1;

• every point of � is committed to target label 0;

• the target may cut anywhere through the freely permutable block �.

Commitment does not mean that the label has already been revealed. In particular, query-pinned points can be assigned their eventual target labels immediately. Hence no separate blocks of pinned points are needed: pinned points are absorbed directly into � or �.

Initially,

$$
L = R = \emptyset , \qquad F = V .
$$

## Endpoint refinements

There are two permitted one-point refinements:

$$
{ \begin{array} { r l } & { { \mathrm { l e f t ~ p i n ~ o f ~ } } p \in F : \quad ( L , F , R ) \longmapsto ( L \cdot p , F \setminus \{ p \} , R ) , } \\ & { { \mathrm { r i g h t ~ p i n ~ o f ~ } } p \in F : \quad ( L , F , R ) \longmapsto ( L , F \setminus \{ p \} , p \cdot R ) . } \end{array} }\tag{A.4}
$$

Both preserve nonemptiness and satisfy

$$
\Gamma _ { \mathrm { n e w } } \subseteq \Gamma _ { \mathrm { o l d } } , \qquad \Omega _ { \mathrm { n e w } } \subseteq \Omega _ { \mathrm { o l d } } .\tag{A.5}
$$

Indeed, a left pin restricts the old free permutation to $p \cdot \pi ( F \setminus \{ p \} )$ and restricts the target to include �. A right pin restricts it to $ \pi ( F \setminus \{ p \} ) \cdot p$ and restricts the target to exclude $p .$ These were already allowed choices in (A.3).

No relative order among the remaining free points is imposed.

## 3. Prefix realizability and the common-value query lemma

## Lemma A.1 — Prefix criterion

Suppose $\sigma$ is the minimum. Reject immediately if a sample � is contradictory or contains $( \sigma , 0 )$ . Otherwise define

$$
A = \{ x \neq \sigma : ( x , 1 ) \in S \} , \qquad B = \{ x : ( x , 0 ) \in S \} .
$$

Then � is realizable if and only if there are no $b \in B , a \in A$ with $b \prec a$ . When realizable,

$$
O _ { \operatorname* { m i n } } ( S , \preceq ) = { \left\{ \begin{array} { l l } { \mathbf { 1 } _ { \{ \sigma \} } , } & { A = \varnothing , } \\ { \mathbf { 1 } _ { \{ x : x \preceq \operatorname* { m a x } _ { \preceq } A \} } , } & { A \neq \varnothing . } \end{array} \right. }\tag{A.6}
$$

Proof. A prefix containing � must contain every $b \prec a ,$ , proving necessity. Conversely, if $A \ne \emptyset$ , the prefix through max � contains all positives and no negatives. If $A = \varnothing$ , the singleton anchor is consistent. These are the smallest possible consistent prefixes. □

For a current state, write

$$
b \triangleleft \longleftrightarrow b \prec a \mathrm { i n e v e r y o r d e r i n } \Omega ( L , F , R ) .\tag{A.7}
$$

This relation is determined by the block order and the fixed internal orders of $L , R .$ Two distinct free points are incomparable under ⊳.

## Lemma A.2 — One-pin common-value query lemma

For every state and every query �, the construction can make at most one endpoint pin so that

$$
O _ { \operatorname* { m i n } } ( S , \preceq )
$$

has exactly the same full-vector value for every order retained after the pin.

Proof.

Contradictory samples and samples containing $( \sigma , 0 )$ have the common answer ⊥, with no refinement. Otherwise use �, � from Lemma A.1.

Case I: an inversion is already forced If some $b \in B , a \in A$ satisfy $b \mathsf { \pmb { q } } a$ , answer ⊥ without changing the state.

Lemma A.1 makes this the required answer for every retained order.

Case II: both label sides contain free points Suppose Case I does not apply and

$$
A \cap F \neq \emptyset , \qquad B \cap F \neq \emptyset .
$$

Choose $b \in B \cap F ,$ , using the smallest arrival index to make the construction explicit, and left-pin �.

Because the sample is not contradictory, any $a \in A \cap F$ is distinct from �. Every new retained order has $b \prec a$ . Therefore its prescribed oracle answer is ⊥.

Exactly one free point was pinned.

Remaining cases Now suppose neither Case I nor Case II applies.

Every positive–negative pair has at least one endpoint outside $F ,$ so its relative order is already fixed. Absence of a forced inversion therefore implies that the sample is realizable in every current retained order. It remains to stabilize the selected smallest prefix.

There are three exhaustive possibilities.

Case III-R: $A \cap R \neq \emptyset .$

Let $p$ be the last point of $A \cap R$ in the fixed list $R .$ In every retained order,

$$
p = \operatorname* { m a x } _ { \preceq } A .
$$

Its down-set is the fixed set

$$
D = \{ \sigma \} \cup L \cup F \cup R _ { \leq p } ,\tag{A.8}
$$

where $R _ { \leq p }$ is the initial segment of � ending at $p$

Answer ${ \bf 1 } _ { D }$ , without a pin.

This case includes samples having positives in both � and �.

Case III-F: $A \cap R = \emptyset$ and $A \cap F \neq \emptyset ,$

Case II’s exclusion gives $B \cap F = \emptyset$ . Case I’s exclusion gives $B \cap L = \emptyset$ , since every left point precedes every free positive. Thus

$$
B \subseteq R .\tag{A.9}
$$

Choose $p \in A \cap F$ by smallest arrival index and right-pin it:

$$
R _ { \mathrm { n e w } } = p \cdot R _ { \mathrm { o l d } } .
$$

The point $p$ is after every remaining free point but before every old right point. Since there are no positives in the old right block,

$$
p = \operatorname* { m a x } _ { \preceq } A \quad { \mathrm { f o r ~ e v e r y ~ n e w ~ r e t a i n e d ~ o r d e r } } .
$$

All negatives are in the old right block and hence follow $p .$ . The required answer is therefore

$$
\mathbf { 1 } _ { D } , \qquad D = \{ \sigma \} \cup L \cup F _ { \mathrm { o l d } } .\tag{A.10}
$$

This is one common full vector, obtained with one pin.

Case III-L: $A \subseteq L$

If $A = \varnothing ,$ , answer ${ \bf 1 } _ { \{ \sigma \} }$ . Otherwise let $p$ be the last point of � in $L ,$ , and answer

$$
\mathbf { 1 } _ { \{ \sigma \} \cup L _ { \leq p } . }\tag{A.11}
$$

No pin is needed.

These cases exhaust all queries and establish the assertion.

## 4. Labels, invariants, and charging

Lemma A.3 — Strong transcript invariant

The construction can maintain all of the following:

1. Γ(�, �, �) is nonempty.

2. Every retained order–target pair agrees with every label already committed by the construction.

3. Every previous oracle answer is exactly the value of the predeclared $O _ { \mathrm { m i n } }$ on every order in the current $\Omega ( L , F , R )$

4. Every point still in � has not yet been predicted.

At a prediction on a free point, the construction can force an error while preserving these properties.   
Proof.

The assertions hold initially. Queries are handled by Lemma A.2. Their refinements preserve nonemptiness and nesting by (A.5). Consequently all earlier common-value assertions remain true, and the new answer has the same property.

At a prediction event:

• if � = � or $x \in L ,$ commit � = 1;

• if $x \in R ,$ , commit $y = 0 ;$

• if $x \in F$ , observe the deterministic prediction ${ \widehat { y } } ,$ set

$$
y = 1 - { \widehat { y } } ,
$$

and left-pin � when $y = 1$ , or right-pin � when $y = 0 .$

The last operation is an endpoint refinement. It leaves a nonempty subset of previously retained order– target pairs, all having the newly committed label. It also incurs an error and removes the predicted point from �.

A label may be committed in this ofline simulation at the prediction event. Lemma A.5 below turns the result into an actual oblivious instance, so the final interaction obeys the label-before-prediction requirement of the transductive protocol. □

## Lemma A.4 — Additive charging

If all � predictions are completed, then

$$
M + Q \geq n = T - \varepsilon .\tag{A.12}
$$

Proof.

Let $p$ be the number of points removed from � by queries. Lemma A.2 gives

$$
p \leq Q .
$$

Every other initially free point is removed at its prediction event and incurs an error. Since every point in � eventually appears,

$$
M \geq n - p .
$$

Adding the inequalities gives (A.12).

Equivalently, the initial potential is $\Phi _ { 0 } = | F | = n ;$ a query decreases it by at most one, and a prediction decreases it only when an error occurs. At completion, $\Phi = 0$ □

## 5. Oblivious fixation and memorylessness

## Lemma A.5 — Finite fixation and replay

The construction yields a single fixed order and target with the claimed cost, using the originally declared memoryless oracle.

Proof.

If $n = 0 .$ , the assertion is immediate. Otherwise simulate the deterministic learner, stopping at the first of:

1. completion of all � predictions; or

2. completion of the �-th query.

This uses at most � query events and � prediction events. In particular, a learner that would make infinitely many queries is covered by the second stopping condition.

Let the final retained state be (�, �, �). Order the remaining � by arrival index, obtaining $\pi _ { 0 } ( F )$ , and define

$$
{ \preceq } ^ { \star } { = } \sigma \cdot L \cdot \pi _ { 0 } ( F ) \cdot R ,\tag{A.13}
$$

and

$$
z ^ { \star } = \left\{ { \begin{array} { l l } { { \mathrm { l a s t ~ p o i n t ~ o f ~ } } L , } & { L \neq \emptyset , } \\ { \sigma , } & { L = \emptyset . } \end{array} } \right.\tag{A.14}
$$

This is a retained order–target pair: it chooses the cut immediately after �. Hence its target agrees with all committed labels.

Now run the learner from the beginning on this fixed order, target, and the fixed oracle

$$
S \longmapsto O _ { \operatorname* { m i n } } ( S , \preceq ^ { \star } ) .
$$

Induct over the recorded events. Determinism implies that the next query or prediction is unchanged whenever the preceding transcript is unchanged. For every recorded query �, Lemma A.3 gives

$$
\mathrm { r e c o r d e d \ a n s w e r } = O _ { \mathrm { m i n } } ( S , \preceq ^ { \star } ) ,
$$

including the entire returned vector. Recorded labels also agree with $c _ { z ^ { \star } }$ . Thus the interaction replays exactly through the stopping event.

If all predictions were completed, Lemma A.4 applies. Otherwise the replay already contains � queries, so $Q \geq n .$

The fixed oracle is legal on every sample by its definition and Lemma A.1, including samples outside the recorded transcript. No answer cache or oracle-extension table is needed.

Repeated queries are automatically consistent: every later retained family is a nonempty subset of the earlier one, and a deterministic function cannot have two diferent values on the same sample and the same retained order.

Finally, (A.13)–(A.14) are selected before the actual replay. The actual adversary is therefore oblivious as required in Section 2. □

Lemmas A.1–A.5 prove Theorem 3.1 for $\bar { O } _ { \operatorname* { m i n } }$

## 6. The maximal-prefix rule and other extremal selectors

## Lemma A.6 — Anchored reversal-complement duality

Keep the anchor fixed. Reverse the order of �, complement every label on �, and leave the anchor label unchanged.

Denote this transformation by �. For a concept,

$$
( D h ) ( \sigma ) = 1 , \qquad ( D h ) ( x ) = 1 - h ( x ) \quad ( x \in V ) .
$$

For samples, labels on � are complemented, while labels on � are unchanged; let $D \perp = \perp$

Then

$$
D ( O _ { \operatorname* { m a x } } ( S , \preceq ) ) = O _ { \operatorname* { m i n } } ( D S , D \preceq ) .\tag{A.15}
$$

## Proof.

An anchored prefix containing � points of � becomes an anchored prefix containing $n - k$ points in the reversed order. This is a consistency-preserving bijection between the two concept classes, reversing inclusion. Consequently, the largest consistent prefix becomes the smallest consistent prefix. Unrealizable samples remain unrealizable. □

Given a deterministic learner for $O _ { \mathrm { m a x } }$ , simulate it using $O _ { \mathrm { m i n } }$ through (A.15), complementing predictions and revealed labels on $V ,$ and leaving the anchor unchanged. Queries and errors are preserved exactly. Applying the already proved anchored minimum-rule theorem proves Theorem 3.1 for the maximum rule.

This also handles N correctly: the actual minimum instance is kept as an always-positive anchor. Ordinary reversal and complementation of all points would incorrectly introduce an empty concept.

For completeness, duality maps a state to

$$
D ( L , F , R ) = ( \operatorname { r e v e r s e } R , \ F , \ \operatorname { r e v e r s e } L ) .\tag{A.16}
$$

Thus the maximum rule has the same one-pin common-value property. For the selector (A.1), �(�) is fixed independently of the compatible order, so one may apply the minimum or maximum one-pin procedure separately at each query. The remainder of the proof is unchanged.

More generally, the proof applies to any fixed legal rule satisfying the following suficient condition:

One-pin common-value property: On every anchored block state and every sample, some refinement by zero or one endpoint pin makes the rule’s full answer constant on all retained orders.

This is a suficient condition, not a claimed characterization of every possible rule.

## 7. Sharpness and relation to AHR25

The total-cost lower bound is exact:

$$
\operatorname* { i n f } _ { \mathcal { A } \mathrm { \ d e t e r m i n i s t i c } } \operatorname* { s u p } _ { \preceq , z } \left( M _ { \mathcal { A } } + Q _ { \mathcal { A } } \right) = T - \varepsilon\tag{A.17}
$$

for either extremal rule.

The matching upper bounds require no oracle calls:

• under E, any fixed predictions make at most � errors;

• under N, always predicting 1 makes at most $T - 1$ errors, since every nonempty prefix labels at least one instance by 1.

Thus (A.17) includes both an upper-bound algorithm and the lower bound; it is not an assertion about an unproved exact fixed-� tradeof.

Our common-value lemma establishes a one-point property for genuine total-order thresholds and a fixed natural ERM selection rule; it does not assert that the returned vector contains only one bit. Theorem 4.4 of AHR25 is not invoked here; its statement, and why its hypothesis fails for the threshold family, are discussed once in Appendix B.

By Theorem 3.1, a deterministic learner with ${ \cal O } ( \log T )$ worst-case errors under $O _ { \operatorname* { m i n } }$ must make

$$
T - \varepsilon - O ( \log T )
$$

queries on some fixed instance. AHR25 (Thm. 4.5(4)) supplies the randomized �(log �)-error, ${ \cal O } ( \log T )$ expected-query upper bound under the same fixed rule, since that algorithm works for arbitrary legal ERM returns. This establishes the intended separation without adversarial tie-breaking.

## 8. Arbitrary predeclared rules: a counterexample

The universal extension

“Theorem 3.1 holds for every fixed function of $( S , \preceq ) ^ { \ast }$

is false.

## 8.1 Median of the feasible prefixes

Index concepts by the number � of positive online-domain points:

• E: $k \in \{ 0 , \ldots , T \}$ , with the sentinel always positive;

$\mathsf { N } : k \in \{ 1 , \ldots , T \}$

For any sample, its feasible cutof indices form an integer interval

$$
I ( S , \preceq ) = \{ \ell , \ell + 1 , \ldots , u \} ,
$$

or are empty. Define the predeclared rule

$$
{ \mathcal R } _ { \mathrm { f e a s } } ( S , \preceq ) = \left\{ \begin{array} { l l } { \perp , } & { I ( S , \preceq ) = \emptyset , } \\ { \mathrm { p r e f i x ~ o f ~ c u t o f f ~ } \left\lfloor \frac { \ell + u } { 2 } \right\rfloor , } & { \mathrm { o t h e r w i s e . } } \end{array} \right.\tag{A.18}
$$

This rule depends only on � and the order. In particular, it does not know a learner’s state or free block.

## Proposition A.7 — Deterministic logarithmic learning under (A.18)

Under either convention, for $T \geq 2$ , there is a deterministic learner satisfying, for every order and target,

$$
Q \leq \lceil \log _ { 2 } T \rceil , \qquad M \leq \lceil \log _ { 2 } T \rceil .\tag{A.19}
$$

Proof.

Maintain a decomposition of the instance set into

$$
H _ { 1 } \prec U \prec H _ { 0 } ,
$$

where $H _ { 1 }$ is an order-prefix known to have target label 1, $H _ { 0 }$ is an order-sufix known to have target label 0, and � is the intervening interval. Initially � is the entire instance set.

For $\left| U \right| \geq 2 ,$ , query

$$
S = ( H _ { 1 } \times \{ 1 \} ) \cup ( H _ { 0 } \times \{ 0 \} ) .\tag{A.20}
$$

This sample is consistent. Let ℎ be the returned concept and set

$$
U _ { 1 } = \{ x \in U : h ( x ) = 1 \} , \qquad U _ { 0 } = U \setminus U _ { 1 } .
$$

Write $a = | H _ { 1 } | , m = | U |$ . Under $\mathrm { E , }$ the feasible cutof interval is exactly $[ a , a + m ]$ , so $| U _ { 1 } | = \lfloor m / 2 \rfloor$ Under N the same holds when $a > 0 ;$ when $a = 0$ , the feasible interval is [1, �], so $| U _ { 1 } | = \lceil m / 2 \rceil$ . Therefore

$$
1 \leq | U _ { 0 } | , | U _ { 1 } | \leq \lceil m / 2 \rceil \qquad ( m \geq 2 ) .\tag{A.21}
$$

Predict with ℎ until an error occurs.

• If $h ( x ) = 0$ but $y = 1$ , then every point in $U _ { 1 }$ precedes � and must have target label 1. Replace

$$
H _ { 1 }  H _ { 1 } \cup U _ { 1 } , \qquad U  U _ { 0 } .
$$

$\mathrm { I f } h ( x ) = 1$ but $y = 0 ,$ , then every point in $U _ { 0 }$ follows � and must have target label 0. Replace

$$
H _ { 0 } \gets U _ { 0 } \cup H _ { 0 } , \qquad U \gets U _ { 1 } .
$$

Thus each error reduces the active interval to at most half, rounded up. If the new interval is a singleton, it consists of the just-mistaken point, whose label is now known; all target labels are then determined. Otherwise make the next query before the next prediction.

After � such errors,

$$
\left| U \right| \leq \left\lceil { \frac { T } { 2 ^ { j } } } \right\rceil .
$$

Consequently there are at most $\lceil \log _ { 2 } T \rceil$ errors and at most that many queries. For $T = 1$ , no queries and at most one error sufice under E; under N, predict 1 without error.

Past labels inside � need not be included in (A.20). The interface of Section 2 allows arbitrary samples; a queried hypothesis is not required to fit observations omitted from that sample. □

At $T = 8 ,$ , this gives one explicit deterministic learner with

$$
M + Q \leq 6
$$

for every order and target, under both conventions. Since $6 < 8$ under E and $6 < 7$ under N, the all-rules extension is refuted with the required quantifiers.

These are rule-specific upper bounds. No balance is assumed of an arbitrary consistency-ERM return.

## 8.2 The diferent “closest to the global median” rule

A diferent natural rule chooses the feasible cutof closest to a fixed global median. This is diferent from (A.18).

Let

$$
k _ { 0 } = \left\{ 0 , E , \frac { } { } \quad \quad m = \left\lfloor \frac { k _ { 0 } + T } { 2 } \right\rfloor . \right.
$$

Define $\mathcal { R } _ { \mathrm { g l o b a l } } ( S , \preceq ) = \bot$ when $I ( S , \preceq ) \ = \ \emptyset$ (this covers contradictory samples and, under E, a negative sentinel); otherwise, when the feasible interval is $[ \ell , u ]$ , let it return the prefix of cutof

$$
\mathrm { p r o j } _ { [ \ell , u ] } ( m ) = \operatorname* { m i n } \{ u , \operatorname* { m a x } \{ \ell , m \} \} .\tag{A.22}
$$

This rule also fails the full inequality of Theorem 3.1(a), but does not admit simultaneous � (log �) mistakes and queries.

Upper bound suficient to refute the bound of Theorem 3.1(a) One empty-sample query returns a prefix having � positive instances. Predict with that concept until the first error.

If an error occurs, the target uncertainty is confined to the side containing the mistaken point: the opposite side has a forced target label. Thereafter, arbitrary predictions inside the uncertain side incur at most one error per remaining point.

Hence

$$
Q = 1 , \qquad M \leq \operatorname* { m a x } ( m , T - m ) = \lceil T / 2 \rceil .\tag{A.23}
$$

This argument is valid for any returned threshold, with its actual side sizes. Balance enters only through the declared rule’s empty-query output.

For � = 8, (A.23) gives $M + Q \leq 5 \quad \quad$ , refuting the bound of Theorem 3.1(a) under both conventions.

Linear lower bound for the global-median rule For every deterministic learner, some instance satisfies

$$
M + Q \geq T - m .\tag{A.24}
$$

To prove this, let $r = T - m$ . If � = 0 the claim is $M + Q \geq 0$ and holds trivially, so assume $r \geq 1$ . If some instance in the family constructed below makes the original learner issue at least � queries (possibly infinitely many), then (A.24) already holds on that instance; otherwise the original learner makes at most � − 1 queries on every instance of the family, and the per-query simulation below yields a derived learner that completes the protocol. Fix an ordered block � of � instances before a block � of $T - m$ instances whose internal order is unknown. Restrict targets to label all of � positively, with an arbitrary, possibly empty, prefix target on �. The order of � can even be revealed to the learner.

An interaction with $\mathcal { R } _ { \mathrm { g l o b a l } }$ can be simulated using an anchored $\bar { O } _ { \operatorname* { m i n } }$ oracle on �, with at most one minimum-oracle call per original query:

1. Contradictions and an illegal negative sentinel are rejected directly.

2. If the sample has a negative in � and a positive in �, it is unrealizable.

3. If it has a negative in � but no positive in �, any feasible cutof lies strictly before �. Its largest feasible cutof, or infeasibility, is computable entirely inside the known ordered block �.

4. If it has neither a negative in � nor a positive in �, cutof � is feasible and is returned.

5. Otherwise it has a positive in � and no negative in �. Every feasible cutof exceeds �, so (A.22) selects the minimum feasible cutof. One minimum-oracle query on the restriction of the sample to � produces exactly that answer; extend it by assigning 1 to all of �.

Simulate predictions on � using their known labels, and use the learner’s predictions on � as those of the derived minimum-oracle learner. The derived learner has no more queries and no more mistakes than the original learner.

Applying the already proved E-convention minimum-rule theorem to �, with a virtual anchor, proves (A.24). Under N, the nonempty block � supplies the anchor when embedding the resulting target back into the original instance.

Since $T - m \ge \lfloor T / 2 \rfloor$ , a deterministic learner cannot simultaneously achieve ${ \cal O } ( \log T )$ mistakes and �(log �) queries under (A.22).

Thus the precise conclusions are:

<table><tr><td>Predeclared rule</td><td>Deterministic conclusion</td></tr><tr><td>Minimum / maximum / sample-dependent extremal selectors</td><td>Exact worst-case minimax total cost  $\overline { { T - \varepsilon } }$ </td></tr><tr><td>Closest feasible cutoff to a fixed global median</td><td>Linear total-cost lower bound; full T -ε bound is false</td></tr><tr><td>Median of the feasible cutoffs</td><td>Deterministic O(log T) queries and mistakes</td></tr></table>

The general-rule counterexamples do not refute Theorem 4.1 or Theorem 3.1: they change the publicly declared oracle rule.

## B Proof of Theorem 4.1

Theorem 4.1: an additive deterministic lower bound with a fixed memoryless ERM oracle

Conventions and quantifiers. Let $T \geq 1$ , and announce the distinct instances

$$
I _ { T } = \{ x _ { 1 } , \ldots , x _ { T } \}
$$

in the fixed arrival sequence $( x _ { 1 } , \ldots , x _ { T } )$ . Consider either of the following conventions (Section 2):

1. Empty-prefix convention: $X = I _ { T } \cup \{ s \}$ , where � is placed before every instance. Put $\varepsilon = 0$

2. Nonempty-prefix convention: $X = I _ { T }$ . Put $\varepsilon = 1$

For every deterministic learner ${ \mathcal { A } } .$ , there exist a total order $\preceq ^ { * }$ on �, a point $z ^ { * } \in X$ , and a function

$$
O ^ { * } : 2 ^ { X \times \{ 0 , 1 \} } \longrightarrow C _ { \preceq ^ { * } } \cup \{ \bot \}
$$

such that:

• $O ^ { * }$ is a legal consistency-type ERM oracle on every sample �;

$O ^ { * } ( S )$ depends only on �, not on the time, query history, or target;

• the order, target, instance sequence, and oracle function are all fixed before the actual interaction; and

• on this one fixed instance,

$$
\boxed { M ( \mathcal { A } ) + Q ( \mathcal { A } ) \geq T - \varepsilon . }
$$

Here the learner may submit arbitrary samples, including contradictory samples, samples involving previously labeled or future instances, and samples involving the non-instance �. It may query at arbitrary times and receives complete concept label vectors, as stipulated in the model of Section 2.

Thus the additive constant is 0 under the empty-prefix convention and 1 under the nonempty-prefix convention.

The proof is self-contained below.

## 1. A common anchored formulation

For both conventions, introduce an anchor � and a set � of initially free instances:

<table><tr><td>Convention</td><td> $a$ </td><td> $V$ </td><td> $\frac { n : = | V | } { T }$ </td></tr><tr><td>Empty prefix</td><td> $s$ </td><td> $I _ { T }$ </td><td></td></tr><tr><td>Nonempty prefix</td><td> $x _ { 1 }$ </td><td> $I _ { T } \backslash \{ x _ { 1 } \}$ </td><td> $T - 1$ </td></tr></table>

We commit to placing � first in the total order. Every concept consequently labels � by 1.

In the nonempty-prefix convention, choosing $x _ { 1 }$ as the least element is simply an allowed adversarial choice of order. The argument would remain valid even if this information were given to the learner for free.

A state consists of:

• an ordered list �;

• an unordered set �;

• an ordered list �;

partitioning �. Its compatible orders are

$$
\mathcal { E } ( L , F , R ) = \left\{ a \cdot L \cdot \pi ( F ) \cdot R : \pi ( F ) \mathrm { i s  a p e r m u t a t i o n } \mathrm { o f } F \right\} .\tag{B.1}
$$

Points in � are committed to target label 1, and points in � to target label 0. They need not already have arrived. Points in � have not arrived and remain free.

We use a list also for its underlying set when taking unions. For a nonempty $H \subseteq X$ , write $h _ { H } = \mathbf { 1 } _ { H }$

All lemmas below apply to both conventions, with $n = T - \varepsilon$

## Lemma B.1 — Prefix consistency criterion

Suppose a total order � has � as its least element. For a sample �, define

$$
{ \cal P } ( S ) = \{ x : ( x , 1 ) \in S \} , \qquad { \cal N } ( S ) = \{ x : ( x , 0 ) \in S \} .
$$

If $P ( S ) \cap N ( S ) \neq \emptyset$ , or if $a \in N ( S )$ , then � is not realizable by $C _ { \tau }$

Otherwise put

$$
A = P ( S ) \setminus \{ a \} , \qquad B = N ( S ) .
$$

Then � is realizable if and only if there is no pair $b \in B , a ^ { \prime } \in A$ with

$$
b \prec _ { \tau } a ^ { \prime } .\tag{B.2}
$$

Proof. Contradictory labels cannot be realized (Section 2). Every nonempty prefix contains the least point �, so a negative label on � is also impossible.

An inversion as in (B.2) prevents any prefix from containing $a ^ { \prime }$ while excluding �. Conversely, if there is no inversion and $A \neq \emptyset$ , the prefix ending at max<sub>�</sub> � contains every positive sample point and no negative sample point. If $A = \emptyset$ , the singleton prefix {�} is consistent. □

Notice that the oracle tests realizability by the class, not agreement with the target. For example, a positive singleton query remains realizable even when that point has already received target label 0.

## 2. The one-pin query rule

The adversary keeps an initially empty table Ans, keyed by the entire sample set �.

The partial order $\prec _ { K }$ of a state $K = ( L , F , R )$ fixes the internal orders of �, �, puts � before � before �, and leaves distinct points of � incomparable.

A useful representation is the ordered list of atoms

$$
W _ { K } = ( \ell _ { 1 } , \ldots , \ell _ { p } , [ F ] , r _ { 1 } , \ldots , r _ { q } ) ,\tag{B.3}
$$

omitting [�] when $F = \varnothing$ . The entire free set is one atom.

## Lemma B.2 — Exhaustive query handling with at most one pin

For every state � and sample �, the following rule either returns an answer already cached, or constructs a new answer legal for every order in the resulting state. A fresh query removes at most one point from �.

Rule and proof.

Cached sample. If $S \in \mathrm { d o m } ( \mathsf { A n s } )$ , return its stored answer without changing the state. Preservation of this answer is proved in Lemma B.3.

Why this rule needs the cache. Let initially $\boldsymbol { F } = \{ u , \nu \}$ and let the first query be $\{ ( u , 1 ) \}$ ; case (iii) below may answer with the vector that is positive on both � and �. Suppose � then arrives and is labelled 1, so that the state becomes $L = [ u ] , F = \{ \nu \}$ . If the same sample $\{ ( u , 1 ) \}$ were now treated as a fresh query, the rule may legally return the prefix [�] alone. Both answers are legal, but the same sample would receive two diferent full vectors, so without the first-answer cache the transcript could not be frozen into a function of the sample alone (Lemma B.5). This is a property of the particular adaptive answering rule used here; the fixed-rule proof in Appendix A needs no answer cache.

For a fresh sample, first handle contradictions or a negative label on � by returning ⊥, without changing the state. Otherwise strip the redundant positive label on �, obtaining disjoint �, $B \subseteq V$

There are exactly three remaining cases.

## (i) A forced inversion already exists

If some $b \in B , a ^ { \prime } \in A$ satisfy

$$
b \prec _ { K } a ^ { \prime } ,
$$

return ⊥, with no state change.

Every order in $\mathcal { E } ( K )$ contains this inversion, so Lemma B.1 proves the answer legal.

## (ii) Both labels occur in the free block

Suppose there is no forced inversion, but

$$
A \cap F \neq \emptyset , \qquad B \cap F \neq \emptyset .
$$

Choose one $b \in B \cap F$ , breaking ties by the public arrival indices, and update

$$
L ^ { \prime } = L \cdot b , \qquad F ^ { \prime } = F \setminus \{ b \} , \qquad R ^ { \prime } = R .\tag{B.4}
$$

Return ⊥.

Because �, � are disjoint, some $a ^ { \prime } \in A \cap F$ is diferent from �. Every new compatible order places � before this $a ^ { \prime }$ . Hence the sample is unrealizable in every new compatible order.

Moreover,

$$
{ \mathcal E } ( L ^ { \prime } , F ^ { \prime } , R ^ { \prime } ) \subseteq { \mathcal E } ( L , F , R ) :
$$

the new orders are exactly old-compatible orders whose free permutation begins with �. Exactly one free point is removed.

Committing the target label of � to 1, although � asks for label 0 there, is legitimate: the oracle is answering that � is not realizable by the chosen class. It is not required to make the target consistent with �.

## (iii) No forced inversion and no mixed-label free block

Leave the state unchanged and return a prefix of the atom list (B.3):

• if $A = \varnothing$ , return $h _ { \{ a \} }$

• otherwise, let � be the rightmost atom containing a point of �, and return the indicator of

$$
H = \{ a \} \cup \{ { \mathrm { a l l p o i n t s i n a t o m s u p t o a n d i n c l u d i n g ~ } } j \} .\tag{B.5}
$$

This contains every positive sample point. It contains no negative sample point: a negative point in an earlier atom would give a forced inversion, while a negative point in the same atom as a positive point could only occur inside �, which is excluded in this case.

The set � contains either all or none of �. Consequently it is a nonempty prefix of every order in $\mathcal { E } ( K )$ Finally, store the new answer as $\mathsf { A n s } [ S ]$

Exhaustiveness. After contradictions are removed, two distinct sample points are incomparable precisely when both lie in �. Thus, if neither (i) nor (ii) applies, every positive–negative pair is comparable and correctly ordered, or one side is empty. This is exactly case (iii). □

The claim is about preserving a completely free residual block—not about the number of bits in a returned concept. A reply may encode many comparisons concerning already committed points.

## 3. Prediction rule and the full invariant

The following is an ofline transcript-construction device. Its conversion into a legal oblivious instance, including compliance with the label-before-prediction convention of the transductive protocol (Section 2), is proved in Lemma B.5.

When the deterministic learner produces prediction $\widehat { y }$ on the next instance �:

${ \mathrm { i f ~ } } x = a { \mathrm { ~ o r ~ } } x \in L$ , set $y = 1$ , with no state change;

${ \mathrm { i f ~ } } x \in R , { \mathrm { s e t ~ } } y = 0$ , with no state change;

• if $x \in F .$ , set

$$
y = 1 - { \widehat { y } } .
$$

Remove � from �, and:

$$
\{ \begin{array} { l l } { L  L \cdot x , } & { y = 1 , } \\ { R  x \cdot R , } & { y = 0 . } \end{array}\tag{B.6}
$$

## Lemma B.3 — Strong invariant, including all recorded answers

Starting from $L = R = \emptyset , F = V .$ , the construction maintains all of the following after every event.

1. Ordered partition and revealed labels. The lists �, � and set � partition �. No point of � has arrived. Every revealed label is 1 on $L \cup \{ a \}$ and 0 on �.

1. Universal validity of afirmative records. For every recorded afirmative answer $h _ { H } .$ , the same complete vector $h _ { H }$ is a concept in $C _ { \tau }$ consistent with its query, for every $\tau \in \mathcal { E } ( L , F , R )$

1. Universal validity of negative records. For every query recorded as ⊥, that query is unrealizable by $C _ { \tau }$ , for every $\tau \in \mathcal { E } ( L , F , R )$

1. An entire interval of feasible targets. For every permutation $\pi ( F )$ and every $k \in \{ 0 , \ldots , | F | \}$ , the prefix with positive set

$$
\{ a \} \cup L \cup \{ \pi _ { 1 } , . . . , \pi _ { k } \}\tag{B.7}
$$

agrees with every revealed label.

In particular, a consistent order and target always exist.

Proof. Initially there are no records or revealed labels, and every prefix in (B.7) is a valid nonempty prefix.

Every state-changing operation refines the compatible-order set:

• a query pin or a free prediction labeled 1 restricts the old free permutation to begin with the extracted point;

• a free prediction labeled 0 restricts it to end with that point.

Thus previously valid afirmative and negative records remain valid in every new compatible order. New query records are valid by Lemma B.2.

For completeness, target feasibility also persists as a joint order–target statement, not merely as an order statement. If a point � is appended to �, a new free permutation $\pi ^ { \prime }$ and cut � correspond to the old free permutation

$$
( u , \pi ^ { \prime } )
$$

and old cut $k + 1$ . If � is prepended to �, they correspond to the old permutation

$$
( \pi ^ { \prime } , u )
$$

and old cut �. Hence every new pair in (B.7) was an admissible old pair. An extracted free point has not previously received a label, so the new commitment does not contradict history.

For prediction events, all new targets give the just-revealed label required by (B.6). An arriving point already in �, �, or the anchor has its committed label, so no change is needed.

A repeated query changes neither state nor table. Its stored answer is legal by items 2 or 3. This also proves the cache branch of Lemma B.2.

Finally, $ { \mathcal { E } } ( L , F , R )$ is nonempty because every finite set �, including the empty set, has a permutation; (B.7) always contains the anchor and therefore defines a valid concept. □

A useful strengthening. Every binary labeling of the current � remains feasible: order its desired 1- points before its desired 0-points and choose the corresponding cut in (B.7). Thus no hidden target-label restriction on the residual free points has been silently accumulated.

## 4. Accounting

## Lemma B.4 — Query pins pay for every possibly avoided free-point mistake

Under either convention, let � be the number of points removed from � by queries, and let � be the number removed when they arrive. After all � predictions,

$$
n = p + f , \qquad p \leq Q , \qquad f \leq M .\tag{B.8}
$$

Consequently,

$$
M + Q \geq n = T - \varepsilon .\tag{B.9}
$$

Proof. The potential

$$
\Phi = | F |
$$

starts at � and ends at 0, because every point of � appears exactly once in the transductive protocol.

A query decreases Φ by at most one. Cached queries, contradictory queries, existing-inversion queries, and afirmative replies do not decrease it. Thus $p \leq Q$

A prediction decreases Φ only if its own instance is still free. Such a prediction is wrong by (B.6). Hence $f \leq M$

These are the only ways points leave �, proving (B.8) and (B.9). The anchor’s possible mistake in the nonempty-prefix convention is simply an additional, uncharged mistake. □

## 5. Freezing into an oblivious instance

## Lemma B.5 — Memoryless freezing and exact deterministic replay

For either convention, every finite constructed transcript prefix can be frozen into a fixed order, target, and globally legal memoryless ERM function reproducing that prefix.

For a completed �-round construction, write the final state as $( L , \emptyset , R )$ . An explicit freezing is

$$
\tau ^ { * } = a \cdot L \cdot R , \qquad z ^ { * } = \mathrm { l a s t } ( a \cdot L ) .\tag{B.10}
$$

The fixed target therefore has positive set $\{ a \} \cup L$

Let $\tau ^ { * } = \left( u _ { 1 } , \ldots , u _ { n + 1 } \right)$ , and let $H _ { j } = \{ u _ { 1 } , \dotsc . . . , u _ { j } \}$ . Define

$$
\begin{array} { r } { O ^ { * } ( S ) = \left\{ \begin{array} { l l } { \mathsf { A n s } [ S ] , } & { S \in \mathrm { d o m } ( \mathsf { A n s } ) , } \\ { h _ { H _ { j ( S ) } } , } & { S \notin \mathrm { d o m } ( \mathsf { A n s } ) \mathrm { ~ a n d ~ a ~ c o n s i s t e n t ~ p r e f i x ~ e x i s t s } , } \\ { \bot , } & { \mathrm { ~ o t h e r w i s e } , } \end{array} \right. } \end{array}\tag{B.11}
$$

where

$$
j ( S ) = \operatorname* { m i n } \{ j \in \{ 1 , \dots , n + 1 \} : h _ { H _ { j } } { \mathrm { ~ i s ~ c o n s i s t e n t ~ w i t h ~ } } S \} .\tag{B.12}
$$

The table in (B.11) is an immutable parameter of the function, not mutable memory of calls during the actual interaction.

Proof.

## Legality on every sample

By Lemma B.3, every recorded afirmative vector is a prefix concept of the final order and is consistent with its recorded sample. Every recorded ⊥ sample is genuinely unrealizable by that same class.

For an unrecorded sample, (B.12) searches exactly all concepts of $C _ { \tau ^ { * } }$ . It returns a consistent one precisely when one exists. This includes:

$S = \emptyset$ , for which the search succeeds;

• contradictory samples, for which it fails;

• samples containing a negative label on the anchor, for which it fails.

Thus (B.11) is a legal consistency ERM oracle on its whole domain. Its value depends only on �.

The returned objects are the stipulated complete label vectors. A vector recorded earlier may have had diferent possible endpoint names in diferent unfinished orders; after freezing it is exactly $c _ { \operatorname* { m a x } _ { \tau ^ { * } } H }$ . No endpoint identity was additionally promised by the interface.

## Target legality

By Lemma B.3, the target in (B.10) agrees with every revealed label. It is a genuine concept, including when $L = \varnothing \colon$ then $z ^ { * } = a$

## Replay

Run the same deterministic learner on the fixed domain, announced sequence, target $c _ { z ^ { * } }$ , and oracle $O ^ { * }$

Induct on successive interaction events. The initial input is identical. If the histories agree so far, determinism implies that the learner’s next operation, query set, or prediction agrees with the constructed one.

• At a query event, (B.11) returns the recorded answer. This includes every repetition of a previously queried set.

• At a label-revelation event, the fixed target returns exactly the constructed label.

The histories therefore remain identical. Complete returned vectors agree on every point of �, so arbitrary evaluations of returned concepts also agree.

For a finite, nonterminal transcript prefix, choose any $\tau ^ { * } \in \mathcal { E } ( L , F , R )$ and any target cut from (B.7), then use the same definition (B.11). The same argument reproduces that prefix.

All adaptive choices have thus been compiled into one fixed triple. In its actual execution, each label $c _ { z ^ { * } } ( x _ { t } )$ is fixed before the learner predicts, as required by the transductive protocol. □

Finiteness detail. It is enough to construct a transcript until all � predictions occur or until � oracle calls have occurred. In the latter case, freeze the current finite prefix using Lemma B.5; its replay already has $Q \geq n$ . Otherwise the completed construction is finite. Thus an infinite-query behavior cannot obstruct the existential construction.

## Completion of Theorem 4.1 and sharpness of the constants

Lemma B.4 gives the additive bound for the constructed transcript, and Lemma B.5 transfers it to one fixed oblivious instance.

The constants cannot be improved uniformly over $T \geq 1$

• Under the main convention, at $T = 1$ a zero-query fixed-label predictor always has $M + Q \leq 1$ . Thus no smaller universal additive constant is possible.

• Under the nonempty-prefix convention, at $T = 1$ the only target labels the sole instance by 1. Predicting 1 without querying gives $M + Q = 0$ , ruling out a universal additive constant smaller than 1.

These witnesses certify only the additive constants; they are not an asserted general error–query upper tradeof. □

## 6. Separation and relation to AHR25

## Corollary — Deterministic low-mistake learning needs nearly � queries

Under the empty-prefix convention, suppose a deterministic learner guarantees at most $m ( T )$ mistakes uniformly over the allowed fixed instances and legal memoryless ERM oracles. Then

$$
Q _ { \mathrm { w o r s t } } ( \mathcal { A } , T ) \geq T - m ( T ) .\tag{B.14}
$$

Under the nonempty-prefix convention, the corresponding bound is

$$
Q _ { \mathrm { w o r s t } } ( \mathcal { A } , T ) \geq T - 1 - m ( T ) .\tag{B.15}
$$

Proof. Apply Theorem 4.1. On the instance it provides, $M \leq m ( T ) , \operatorname { s o } Q \geq T - \varepsilon - m ( T )$ In particular, a deterministic ${ \cal O } ( \log T )$ -mistake guarantee requires

$$
Q _ { \mathrm { w o r s t } } \geq T - \varepsilon - O ( \log T ) .\tag{B.16}
$$

By the randomized theorem of AHR25 (Thm. 4.5(4)), both expected mistakes and expected calls can instead be ${ \cal O } ( \log T )$ . Consequently:

• randomized expected total cost $M + Q$ is ${ \cal O } ( \log T )$ ;

• every deterministic learner has worst-case total cost at least $T - \varepsilon .$

Equivalently, using the dimension conventions of Section 2, the gap is exponential when parameterized by $d = \Theta ( \log T )$ : logarithmic-in-� cost versus linear-in-� cost, or $O ( d + 1 )$ versus $2 ^ { \Omega ( d ) }$

This establishes a positive answer to the open question of AHR25 about the power of randomization within the threshold-family, consistency-ERM, oblivious-instance setting of this paper.

## Why Theorem 4.4 of AHR25 is not the proof of this result

The statement of Theorem 4.4 of AHR25 is:

“Let F be the family of all classes with Littlestone dimension $d _ { \mathrm { L D } }$ . For $T \geq 2 ^ { d _ { \mathrm { L D } } + 1 }$ , when having access to the ERM oracle, if fewer than $T / 2$ queries are made, at least $2 ^ { d _ { \mathrm { L D } } } - 1$ mistakes will be made.”

Its proof contains the sentence:

“An essential property is that each query provides information about at most one point.”

The present proof implements a related one-point charging principle, but that theorem is not invoked as a lower-bound lemma. Its construction uses threshold classes on equivalence classes, rather than the present threshold classes on distinct totally ordered points. Moreover, for either dimension convention of Section 2,

$$
d = \lfloor \log _ { 2 } ( T + 1 ) \rfloor \quad { \mathrm { o r } } \quad d = \lfloor \log _ { 2 } T \rfloor ,
$$

the required condition $T \geq 2 ^ { d + 1 }$ fails.

Here, forced-realizable queries are explicitly answered by proper full concepts, and all replies are then frozen into one memoryless oracle.

## Why the construction does not contradict Theorem 4.5(4) of AHR25

For a randomized learner, performing the construction separately for each random tape would generally produce

$$
( \tau _ { \omega } , z _ { \omega } , O _ { \omega } )
$$

depending on that tape. In particular, which negative sample point is pinned may depend on the learner’s random query.

This gives an instance after seeing randomness, not one fixed oblivious instance. The invalid quantifier exchange would be

$$
\forall \omega \exists I _ { \omega } \quad \implies \quad \exists I \mathrm { w i t h t h e s a m e } \mathrm { e x p e c t e d l o w e r } \mathrm { b o u n d o v e r } \omega .
$$

Determinism permits the replay argument for one learner execution; it does not justify this exchange.

## C Proof of Theorem 3.2

## 1. Theorem 3.2

All logarithms in the proof are natural.

Let

$$
\epsilon = \left\{ { \begin{array} { l l } { 0 , } & { { \mathrm { u n d e r ~ c o n v e n t i o n ~ ( E ) } } , } \\ { 1 , } & { { \mathrm { u n d e r ~ c o n v e n t i o n ~ ( N ) } } , } \end{array} } \right. \qquad n = T - \epsilon .
$$

Under (E), there are $n \ : = \ : T$ prediction points and an additional sentinel $\textit { a } = \textit { s }$ . Under (N), there are $n = T - 1$ hard prediction points and one additional prediction point �. The point � will be the minimum of the total order and will always have target label 1.

## Theorem 3.2 — expected-query and worst-case-query lower bounds

For every $T \geq 1$ , under either convention above, there exist:

1. a finite distribution $\mathcal { D } _ { T } ^ { \epsilon }$ over total orders and target thresholds;

2. a fixed transductive sequence of � distinct points; and

3. an explicit deterministic, memoryless consistency-type ERM oracle

$$
O ( S , \preceq ) ,
$$

depending only on the sample and the total order,

such that every randomized learner satisfies

$$
\boxed { \mathbb { E } [ M ] \ \geq \ \frac { \left( T + 1 - \epsilon \right) 1 2 8 ^ { - \mathbb { E } [ N _ { \mathrm { q r y } } ] } - 1 } { 2 } . }\tag{C.1}
$$

Here both expectations are over $\mathcal { D } _ { T } ^ { \epsilon }$ and the learner’s private randomness, and the assertion is substantive when the expected number of queries is finite.

Consequently:

• Expected-query budget. $\mathrm { I f } \mathbb { E } [ N _ { \mathrm { q r y } } ] \leq Q$ , then

$$
\mathbb { E } [ M ] \geq \operatorname* { m a x } \left\{ 0 , \frac { ( T + 1 - \epsilon ) 1 2 8 ^ { - Q } - 1 } { 2 } \right\} .\tag{C.2}
$$

• Worst-case-query budget. If $N _ { \mathrm { q r y } } \leq Q$ almost surely, the same bound holds.

In particular, for both conventions,

$$
\boxed { \mathbb { E } [ M ] \geq \frac { 1 } { 2 } T \left( \frac { 1 } { 1 2 8 } \right) ^ { Q } - \frac { 1 } { 2 } . }\tag{C.3}
$$

The expected-query hypothesis in (C.2) may be imposed merely on the average over the displayed hard distribution. In particular, the uniform, per-instance expected-query guarantee in the cost accounting of Section 2 implies it.

The construction and proof below are self-contained.

## 2. The hard distribution and the actual oracle

## 2.1 A balanced interval tree

Identify the hard points with $x _ { 1 } , \ldots , x _ { n }$ , in their arrival order.

Build a binary tree on the interval [�]. An interval of size $m \ge 2$ is split into its first $\lfloor m / 2 \rfloor$ indices and its remaining $\lceil m / 2 \rceil$ indices. Singletons are leaves.

For $n \geq 1$ , draw

$$
J \sim { \mathrm { U n i f } } ( [ n ] ) , \qquad Y _ { 1 } , \dots , Y _ { n } \stackrel { \mathrm { i i d } } { \sim } { \mathrm { B e r n o u l l i } } ( 1 / 2 ) ,
$$

independently.

Define a priority permutation $\pi _ { J }$ as follows:

• traverse the path from the root to leaf $J ;$

• at every internal node, append all indices in the sibling not containing �, in increasing numerical order;

• finally append �.

Thus, every sibling encountered earlier has higher, or “outer,” priority than every point in the current interval containing �.

## 2.2 From priorities and labels to a genuine total order

The total order is

$$
a \ \prec \ \left( x _ { i } : Y _ { i } = 1 , \ \mathrm { i n } \ \pi _ { J } \ \mathrm { o r d e r } \right) \ \prec \ \left( x _ { i } : Y _ { i } = 0 , \ \mathrm { i n } \ \mathrm { r e v e r s e } \ \pi _ { J } \ \mathrm { o r d e r } \right) .\tag{C.4}
$$

The target threshold ends at the last $Y _ { i } = 1$ point in the first list, or at � if that list is empty. Therefore

$$
c _ { z } ( a ) = 1 , \qquad c _ { z } ( x _ { i } ) = Y _ { i } .\tag{C.5}
$$

This is a total-order threshold class, not a threshold class on equivalence classes: every point occupies its own distinct position, and every point-prefix belongs to the class.

The transductive sequence is:

• under $( \mathrm { E } ) \colon x _ { 1 } , \ldots , x _ { n }$ , with $a = s$ a non-instance sentinel;

• under $( \mathrm { N } ) \colon x _ { 1 } , \ldots , x _ { n } , a .$

If $n = 0$ , which occurs only for $T = 1 \mathrm { u n d e r } \left( \mathrm { N } \right)$ , use the singleton order consisting of �, with target label 1.

All random choices are made before interaction. This defines $\mathcal { D } _ { T } ^ { \epsilon }$ , independently of the learner.

## 2.3 The actual, memoryless oracle

Use the minimum consistent prefix oracle. For completeness, the following definition is valid on any nonempty finite totally ordered domain.

Given $S \subseteq X \times \{ 0 , 1 \}$ :

1. If � assigns both labels to some point, return ⊥.

2. Let

$$
P = \{ x : ( x , 1 ) \in S \} , \qquad B = \{ x : ( x , 0 ) \in S \} .
$$

1. Set

$$
p = \left\{ { \begin{array} { l l } { \operatorname* { m a x } _ { \preceq } P , } & { P \neq \emptyset , } \\ { \operatorname* { m i n } _ { \substack { \preceq } } X , } & { P = \emptyset . } \end{array} } \right.
$$

1. If some $b \in B$ satisfies $b \preceq p .$ , return ⊥. Otherwise return the full vector $c _ { p } .$

This rule depends only on $( S , \preceq )$ , not on �, the target, the history, or the learner’s private randomness.

It is a legal oracle: every threshold containing � must contain the prefix ending at $p ;$ thus an element of � in that prefix makes consistency impossible. Otherwise $c _ { p }$ itself is consistent. This also covers empty and one-sided samples. On our support, every concept contains $^ { a , }$ so a query containing $( a , 0 )$ is correctly rejected.

## 3. Removing failure answers by a stronger oracle

The next reduction is important: it handles arbitrary �, including all information carried by ⊥ answers.

## Lemma C.1 — prefix-max domination

On the constructed instances, every call to � can be simulated using one call to a stronger oracle which, on any set $A \subseteq [ n ]$ , returns the full prefix ending at

$$
\operatorname* { m a x } _ { \preceq } \{ x _ { i } : i \in A \} ,
$$

or at � when $A = \varnothing$

Proof Take � to be the hard points assigned label 1 by the sample. The stronger response is precisely the candidate minimum prefix in the definition of �. Checking the sample’s 0-labeled points against that full vector determines whether $o$ returns this prefix or ⊥. Contradictory samples and $( a , 0 )$ may be rejected immediately.

Thus one stronger call determines the original answer for every sample, regardless of its size or the arrival status of its points. □

We now strengthen this prefix-max oracle further.

## The auxiliary revelation oracle

The auxiliary oracle maintains an interval � containing �, initially $I = [ n ]$ . It reveals to the learner:

• every target label outside $I ;$

• the priority order of all those outside points;

• the interval � itself.

Such information may include future labels. This is deliberately additional information.

For a query $A \subseteq [ n ]$ , it proceeds as follows.

1. If � \ � contains a point with target label 0, stop.

2. If � ∩ � = ∅, stop.

3. If � is a singleton, reveal its label, set � = ∅, and stop.

4. Otherwise reveal which child of � contains �, reveal all target labels in the other child, replace � by the child containing �, and repeat.

At termination it returns the requested full prefix as well as the auxiliary information.

The entire procedure costs one query, however many internal revelations occur.

This is only an auxiliary, stronger information source. It is not asserted to be the actual ERM oracle. Its memory and its access to target labels do not enter the definition of �(�, ⪯).

## Lemma C.2 — the auxiliary oracle simulates the full prefix

At every stopping point above, the requested prefix is determined by the revealed information. In particular, an original learner can be simulated by a learner using the auxiliary oracle, with the same number of queries and the same prediction distribution.

Proof All points outside the current � precede all points inside � in priority order.

Suppose �\� contains a target-negative point. The maximum of � in the total order is then the outermost target-negative point of �: the one appearing earliest in �<sub>�</sub>. It lies outside �, and its identity is known.

Its prefix contains:

• every point of �;

• every outside target-positive point; and

• exactly those outside target-negative points whose priority is no earlier than its own.

The entire vector is therefore known.

If there is no outside target-negative point and � ∩ � = ∅, all points of � are revealed target-positive points. Their total-order maximum is their latest point in priority order. Its prefix contains no point of �, and its values outside � are known. The empty-� case returns the prefix {�}.

The singleton revelation makes all remaining information available. The procedure terminates because every nonterminal refinement reduces the interval size.

Combining this computation with Lemma C.1 reconstructs every original response, including ⊥. An original learner may simply ignore the additional information. □

## 4. The posterior invariant

## Lemma C.3 — conditional independence inside the active interval

At every point of the auxiliary interaction, including between internal revelations of a query, conditional on the complete auxiliary history:

1. � is uniform in the current nonempty interval �;

2. all labels of unarrived points in � are independent fair bits;

3. these unarrived labels are independent of �.

The statement remains valid for randomized learners.

## Proof Initially this is the definition of the distribution.

Suppose the statement holds for �. The next child containing � has probability equal to its size divided by | �|. Conditional on that child, � is uniform within it.

The labels revealed in the sibling are independent of � within the retained child and independent of all unarrived labels in that child. The priority ordering of the revealed sibling is fixed by its numerical indices and does not depend on the remaining location of �.

The decision to stop or continue uses only:

• the chosen query;

• the retained child’s identity;

• the revealed labels outside it; and

• whether the query has any point in the retained child.

It does not depend on the location of � within that child or on its unrevealed future labels. Hence conditioning on that decision preserves the asserted law. The final full-prefix response is determined by the revealed information, by Lemma C.2, and creates no additional conditioning.

An arriving label in � is a fair bit independent of � and the remaining unarrived labels. Revealing it therefore also preserves the invariant.

For randomized learners, include the learner’s private seed in the conditioning, or apply the same induction after fixing that seed. Its independence from the hard instance supplies the base case. □

This explicitly replaces, rather than assumes, exchangeability after failure answers.

## 5. Charging free labels to predictions

Call a prediction unrevealed if its point is still in the active interval immediately before that prediction. Let � be the number of such hard-point predictions made so far; � is a cumulative count over the whole interaction and never decreases when the active interval shrinks (it must not be confused with the number � of already-predicted points inside the current interval, for which $H \geq k )$ .

Queries between a prediction and its label feedback can be treated as post-feedback queries in a stronger protocol: provide the label immediately after the prediction, and let a simulator ignore it until the original learner would have received it. The prediction loss is unchanged. Thus it sufices to analyze queries between prediction-feedback steps.

## Lemma C.4 — mistake charge and the potential

Let the active interval have size �. Let � of its points have already been predicted, and let $r = m - k$ . Then:

1. $H \geq k ;$

2. $\begin{array} { r } { \mathbb { E } [ M ] \geq \frac { 1 } { 2 } \mathbb { E } [ H _ { \mathrm { f i n a l } } ] } \end{array}$

3. the potential

$$
\Phi = \ln ( H + r + 1 )\tag{C.6}
$$

is unchanged by prediction-feedback steps;

1. one binary interval refinement decreases Φ by at most ln 2.

Proof The active intervals are nested. Every already-predicted point still belonging to the current interval also belonged to the active interval when it was predicted. Hence $H \geq k$

By Lemma C.3, an unrevealed prediction faces an independent fair target bit. Its conditional expected loss is exactly 1/2, regardless of the prediction distribution. Summing these conditional expectations proves assertion 2.

On an unrevealed prediction, � increases by one and � decreases by one. Otherwise neither changes. This proves assertion 3.

For assertion 4, write $m ^ { \prime } , k ^ { \prime }$ for the retained child’s size and number of already-predicted points. We have

$$
m ^ { \prime } \geq \lfloor m / 2 \rfloor , \qquad k ^ { \prime } \leq k , \qquad H \geq k .
$$

Consequently,

$$
H + m - k + 1 \leq 2 ( H + m ^ { \prime } - k ^ { \prime } + 1 ) .\tag{C.7}
$$

Indeed, with $d = H - k \geq 0$ , the left side is $d + m + 1$ , whereas the new quantity is at least $d + m ^ { \prime } + 1$ , and

$$
d + m + 1 \leq 2 ( d + m ^ { \prime } + 1 ) .
$$

Taking logarithms proves the assertion.

The +1 is essential for handling empty residual sets and the last leaf without undefined logarithms.

## 6. The single-query information lemma

## Lemma C.5 — bounded expected logarithmic decrease per query

Conditional on any auxiliary history immediately before a query,

$$
\begin{array} { r } { \boxed { \mathbb { E } [ \Phi _ { \mathrm { b e f o r e } } - \Phi _ { \mathrm { a f t e r } } \mid \mathrm { h i s t o r y } ] \le 7 \ln 2 . } } \end{array}\tag{C.8}
$$

This holds for every query set �, including sets chosen using arbitrarily many previously observed labels.

Proof During one query, � is fixed.

Call a state early if

$$
k < \lfloor m / 2 \rfloor ,
$$

and late otherwise.

Late states

In a late state,

$$
r = m - k \leq k + 1 \leq H + 1 .
$$

Therefore

$$
H + r + 1 \leq 2 ( H + 1 ) .\tag{C.9}
$$

The potential at the end of the query is at least ln(� + 1). Thus, from the first late state onward, the entire remaining potential decrease is at most ln 2, regardless of how many further revelations occur.

This also covers a singleton interval.

## Early states

Consider an early state in which the query needs another refinement. Let $I _ { L } , I _ { R }$ be the left and right children.

Because arrival order is numerical order and fewer than $| I _ { L } |$ points of � have arrived, every point of $I _ { R }$ is unarrived. Its labels are therefore independent fair bits by Lemma C.3.

There are two exhaustive cases.

• If $A \cap I _ { R } = \emptyset$ , then when $J \in I _ { R }$ , the new active interval contains no point of �, so the query stops. This event has probability

$$
{ \frac { | I _ { R } | } { m } } \geq { \frac { 1 } { 2 } } .
$$

• If $A \cap I _ { R } \neq \emptyset$ , consider the event that $J \in I _ { L }$ and at least one point of $A \cap I _ { R }$ has target label 0. The right child is then revealed, and the query stops because an outside target-negative point of � has been found. Its conditional probability is

$$
\frac { | I _ { L } | } { m } \left( 1 - 2 ^ { - | A \cap I _ { R } | } \right) \geq \frac { 1 } { 3 } \cdot \frac { 1 } { 2 } = \frac { 1 } { 6 } .\tag{C.10}
$$

Thus, after each early-state refinement, the conditional probability of continuing to another early-state refinement is at most $5 / 6$

Let � be the number of early-state refinements before the query stops or first reaches a late state. No independence between refinements is required:

$$
\Pr ( D \geq j ) \leq ( 5 / 6 ) ^ { j - 1 } , \qquad \mathbb { E } [ D ] \leq 6 .\tag{C.11}
$$

Each refinement costs at most ln 2 in potential, by Lemma C.4. The remaining late-state part costs at most another ln 2. Hence

$$
\mathbb { E } [ \Phi _ { \mathrm { b e f o r e } } - \Phi _ { \mathrm { a f t e r } } ] \le 6 \ln 2 + \ln 2 .
$$

This proves (C.8).

The two mechanisms in this lemma are complementary:

• querying genuinely future points encounters an independent wrong-side point with constant probability;

• exploiting many already known labels is permitted, but their predictions have already contributed to �.

## 7. Proof of Theorem 3.2

Initially,

$$
H = 0 , \qquad r = n , \qquad \Phi _ { \mathrm { i n i t i a l } } = \ln ( n + 1 ) .
$$

After all hard points have been predicted,

$$
r = 0 , \Phi _ { \mathrm { f i n a l } } = \ln ( H _ { \mathrm { f i n a l } } + 1 ) .
$$

Prediction-feedback steps leave the potential unchanged. Let $\Delta _ { j } ~ \geq ~ 0$ be its decrease during the �-th query, setting $\Delta _ { j } = 0$ when that query does not occur.

Lemma C.5 gives

$$
\mathbb { E } [ \Delta _ { j } ] \le 7 \ln 2 \ \operatorname* { P r } ( N _ { \mathrm { q r y } } \ge j ) .
$$

Summing nonnegative quantities,

$$
\begin{array} { r l r } {  { \ln ( n + 1 ) - \mathbb { E } [ \ln ( H _ { \mathrm { f i n a l } } + 1 ) ] = \mathbb { E } \Bigg [ \sum _ { j } \Delta _ { j } \Bigg ] } } \\ & { } & { \leq 7 \ln 2 \sum _ { j } \operatorname* { P r } ( N _ { \mathrm { q r y } } \geq j ) } \\ & { } & { = 7 \ln 2 \mathbb { E } [ N _ { \mathrm { q r y } } ] . } \end{array}\tag{C.12}
$$

This argument directly permits an adaptive, random number of queries. It does not condition the hard distribution on a low-query event.

By concavity of ln,

$$
\ln ( \mathbb { E } [ H _ { \mathrm { f i n a l } } ] + 1 ) \geq \mathbb { E } [ \ln ( H _ { \mathrm { f i n a l } } + 1 ) ] .
$$

Combining this with (C.12),

$$
\mathbb { E } [ H _ { \mathrm { f i n a l } } ] \ge \left( n + 1 \right) \exp \bigl ( - 7 \ln 2 \mathbb { E } [ N _ { \mathrm { q r y } } ] \bigr ) - 1 .\tag{C.13}
$$

Lemma C.4 now yields

$$
\mathbb { E } [ M ] \ge \frac { ( n + 1 ) ! 2 8 ^ { - \mathbb { E } [ N _ { \mathrm { q r y } } ] } - 1 } { 2 } .
$$

This lower bound holds even for learners receiving the auxiliary information. By Lemmas C.1 and C.2, it therefore holds for every learner using the actual memoryless oracle.

Substituting $n = T - \epsilon \mathrm { p r o v e s } \left( \mathbf { C } . 1 \right)$ . Monotonicity in the query budget proves both versions of (C.2).

For $n = 0$ , the instance has one known positive point and the claimed lower bound is nonpositive; it is immediate. Thus all boundary cases are included. □

## Oblivious fixed-instance consequence

Suppose a learner satisfies the uniform expected-query guarantee of the cost accounting of Section 2:

$$
\operatorname* { s u p } _ { \leq , z } \mathbb { E } _ { \mathcal { A } } [ N _ { \mathrm { q r y } } ] \leq Q .
$$

Its average query count under $\mathcal { D } _ { T } ^ { \epsilon }$ is at most $Q .$ Since the distribution has finite support, some fixed support instance satisfies

$$
\mathbb { E } _ { \mathcal { A } } [ M ] \geq \frac { ( T + 1 - \epsilon ) 1 2 8 ^ { - Q } - 1 } { 2 } .
$$

The query guarantee also holds on that instance by hypothesis.

The order, target, sequence, and oracle are already fixed before the learner’s private randomness. No adaptive-adversary solidification step is needed.

## 8. Consequence for logarithmic query complexity

## Corollary C.9

Under either convention, suppose a randomized learner has worst-case expected mistakes at most $m _ { T }$ and worst-case expected queries at most $q _ { T }$ , in the sense of the cost accounting of Section 2. Then

$$
q _ { T } \geq \frac { \ln ( T + 1 - \epsilon ) - \ln ( 2 m _ { T } + 1 ) } { 7 \ln 2 } .\tag{C.14}
$$

In particular, if $m _ { T } = O ( ( \ln T ) ^ { d } )$ for a fixed $d ,$ then

$$
q _ { T } \geq \frac { \ln T - d \ln \ln T - O ( 1 ) } { 7 \ln 2 } = \Omega ( \ln T ) .\tag{C.15}
$$

Proof Apply Theorem 3.2 to the learner and rearrange.

Together with AHR25, Thm. 4.5(4), this proves that ${ \cal O } ( \log T )$ expected oracle calls are optimal in order for achieving polylogarithmic expected mistakes.

Together with Theorem 4.1 and AHR25, Thm. 4.5(3), it also closes the stated comparison at ${ \cal O } ( \log T )$ mistakes:

• randomized query complexity: Θ(log �), in expectation;

• deterministic query complexity: Θ(�) against worst-case legal oracles and under the extremal rules of Theorem 3.1 (not under every legal pre-declared rule, Theorem 3.4).

This does not claim sharp constants or a complete pointwise tight characterization for every possible mistake/query budget.

## Relation to the one-point property

The “one point per query” property used in the proof of Theorem 4.4 of AHR25 is not used here and generally fails for the present full-vector ERM interface: a query in our analysis may reveal a whole sibling interval, containing a linear number of labels. The replacement property is Lemma C.5: one query decreases a paid logarithmic potential by only a constant in expectation. Theorem 4.4 of AHR25 is not invoked here; its statement, and why its hypothesis fails for the threshold family, are discussed once in Appendix B.

## 9. Exact zero-query analysis for the uniform distribution

The following results use the uniform distribution under convention (E), not the diferent distribution used in Theorem 3.2.

## Proposition C.6 — exact zero-query optimum

Under the uniform distribution, the minimum expected number of mistakes of any zero-query learner, including randomized learners, is

$$
\boxed { F ( T ) = \sum _ { t = 1 } ^ { T } \frac { \big \lfloor ( t + 1 ) ^ { 2 } / 4 \big \rfloor } { t ( t + 1 ) } . }\tag{C.16}
$$

In particular,

$$
F ( T ) \geq T / 4 , \qquad F ( T ) = T / 4 + { \textstyle { \frac { 1 } { 4 } } } \ln T + O ( 1 ) .\tag{C.17}
$$

After observing � positive and � negative labels, the next label has posterior probability

$$
\operatorname* { P r } ( Y _ { \mathrm { n e x t } } = 1 \mid \mathrm { h i s t o r y } ) = \frac { k + 1 } { k + m + 2 } .\tag{C.18}
$$

Proof An equivalent representation of the uniform distribution is to draw independent

$$
P , U _ { 1 } , \dots , U _ { T } \sim \mathrm { U n i f } [ 0 , 1 ]
$$

and set $Y _ { i } = \mathbf { 1 } [ U _ { i } \leq P ]$ . Ordering the $U _ { i } { ^ \mathrm { { \tiny ~ s } } }$ gives a uniform permutation, and the position of $P$ among them is uniform on $\{ 0 , \ldots , T \}$ . The actual threshold point remains an element of $X \colon$ it is the last instance below $P ,$ or the sentinel.

For a history with � ones and � zeros, the posterior density of � is proportional to $p ^ { k } ( 1 - p ) ^ { m }$ . Taking the ratio of the elementary integrals for its first moment proves (C.18).

After � observations, the number $K _ { r }$ of ones is uniform on $\{ 0 , \ldots , r \}$ . Therefore the expected Bayes error on round $r + 1$ is

$$
\frac { 1 } { ( r + 1 ) ( r + 2 ) } \sum _ { k = 0 } ^ { r } \operatorname* { m i n } ( k + 1 , r - k + 1 ) = \frac { \lfloor ( r + 2 ) ^ { 2 } / 4 \rfloor } { ( r + 1 ) ( r + 2 ) } .
$$

Summing proves (C.16).

A randomized prediction cannot improve on the posterior-majority decision, since its conditional loss is afine in the probability of predicting 1. Posterior majority attains every summand, proving exact optimality.

Finally, the round-� summand equals

$$
{ \frac { 1 } { 4 } } + { \left\{ \begin{array} { l l } { { \frac { 1 } { 4 t } } , } & { t { \mathrm { ~ o d d } } , } \\ { \qquad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ { { \frac { 1 } { 4 ( t + 1 ) } } , } & { t { \mathrm { ~ e v e n } } . } \end{array} \right. }
$$

This proves (C.17).

Convention (N) correction. If the literal no-empty-prefix counterpart draws a uniform permutation and a uniform cut in $\{ 1 , \ldots , T \}$ , its label distribution is the (E) distribution conditioned on a nonempty positive

set. The total variation distance is $1 / ( T + 1 )$ . Since total loss lies in $[ 0 , T ]$ , its optimal zero-query value $F _ { N } ( T )$ satisfies

$$
| F _ { N } ( T ) - F ( T ) | \leq \frac { T } { T + 1 } < 1 .\tag{C.19}
$$

In particular, $F _ { N } ( T ) \ge T / 4 - 1$

Notice that the exact correction in (C.17) is logarithmic and positive; the exact optimum is not $T / 4 \substack { + } O ( 1 )$ .

## 10. Why the uniform distribution cannot prove Theorem 3.2

## Proposition C.7 — a one-query learner for the uniform distribution

Fix $1 \leq m \leq T$ . Predict the first � labels by posterior majority. Then query all � observed labels once and predict the returned hypothesis on all remaining points.

For the minimum-prefix rule on the uniform distribution, this algorithm has exactly

$$
\boxed { \mathbb { E } [ M ] = F ( m ) + \frac { T - m } { m + 2 } . }\tag{C.20}
$$

For every consistent returned threshold, it satisfies

$$
\mathbb { E } [ M ] \leq F ( m ) + \frac { 2 ( T - m ) } { m + 2 } .\tag{C.21}
$$

Thus one query sufices for $O ( { \sqrt { T } } )$ expected mistakes on the uniform distribution.

Proof Use the representation in Proposition C.6. Let

$$
L = \operatorname* { m a x } \{ U _ { i } : i \leq m , Y _ { i } = 1 \} ,
$$

with $L = 0$ if there is no observed positive point.

The minimum-prefix hypothesis makes future mistakes exactly on $( L , P ]$ . Conditional on $P = p$

$$
\operatorname { \mathbb { E } } [ p - L \mid P = p ] = \int _ { 0 } ^ { p } ( 1 - \nu ) ^ { m } d \nu .
$$

Averaging over $p ,$

$$
\mathbb { E } [ P - L ] = \frac { 1 } { m + 2 } .\tag{C.22}
$$

Each future point is independent of the first � points and $P ,$ proving (C.20).

For arbitrary consistent tie-breaking, let � be the smallest observed negative coordinate, with $U = 1$ if none exists. Every consistent threshold agrees with the target outside $( L , U )$ , regardless of how its choice depends on the full order. By symmetry,

$$
\mathbb { E } [ U - P ] = \frac { 1 } { m + 2 } .
$$

Bounding mistakes by membership in $( L , U )$ proves (C.21).

For example, with $m = \lceil { \sqrt { T } } \rceil$ , using $F ( m ) \leq m / 2$ , (C.21) gives the explicit bound

$$
\mathbb { E } [ M ] \leq \frac { 5 } { 2 } \sqrt { T } + \frac { 1 } { 2 } .\tag{C.23}
$$

This is a distribution-specific upper bound, not a worst-case upper bound over all orders. It does not contradict Theorem 4.1. It does prove that no positive-constant $T \rho ^ { Q }$ lower bound with $Q = 1$ can hold on the uniform distribution, even if the oracle’s consistent tie-breaking is chosen adversarially.

## 11. A counterexample to a naive per-history contraction bound

## Proposition C.8 — no history-uniform constant contraction bound for the uniform distribution

Even after regularizing block sizes by +1, a single-query logarithmic contraction bound on the uniform distribution cannot hold uniformly over all positive-probability histories.

Proof Choose integers $k \geq 1 , u \geq 1$ , and $T = k + u + 1$ . Put $b = x _ { k + 1 }$

Before prediction, query the singleton positive sample $\{ ( b , 1 ) \}$ . Consider the history in which:

1. the answer is the all-one vector, so � is the maximum instance in the unknown order;

2. the first � arriving labels are all 1;

3. the label of � is 0.

Under the uniform distribution this history has probability

$$
{ \frac { 1 } { T } } \cdot { \frac { T } { T + 1 } } \cdot { \frac { 1 } { k + 1 } } = { \frac { 1 } { ( T + 1 ) ( k + 1 ) } } > 0 .\tag{C.24}
$$

Before the next query, all � unarrived points have undetermined labels. Query the � known positive points. Let $U ^ { \prime }$ be the set of unarrived points above their maximum.

Removing the known maximum �, the conditional model is again a uniform random order and uniform cut, now conditioned on � specified positive labels. The expected coordinate of their maximum is $k / ( k + 2 )$ . Hence

$$
\mathbb { E } [ | U ^ { \prime } | \mid \mathrm { h i s t o r y } ] = \frac { 2 u } { k + 2 } .\tag{C.25}
$$

All points removed from the uncertainty set are certified positive; the points in $U ^ { \prime }$ remain label-undetermined.

By Jensen’s inequality,

$$
\mathbb { E } [ \ln { \frac { u + 1 } { | U ^ { \prime } | + 1 } } | { \mathrm { h i s t o r y } } ] \geq \ln { \frac { u + 1 } { 2 u / ( k + 2 ) + 1 } } .\tag{C.26}
$$

Taking $u = ( k + 2 ) ^ { 2 }$ , the right side becomes

$$
\ln { \frac { ( k + 2 ) ^ { 2 } + 1 } { 2 k + 5 } } ,
$$

which diverges with �.

All parameters and probabilities here are finite and rational. For one explicit member,

$$
k = 2 5 5 , \quad u = 6 6 0 4 9 , \quad T = 6 6 3 0 5 ,
$$

the history probability is $1 / 1 6 9 7 4 3 3 6 ,$ , and

$$
\mathbb { E } [ | U ^ { \prime } | ] = 5 1 4 , \qquad \frac { \mathbb { E } [ | U ^ { \prime } | + 1 ] } { u + 1 } = \frac { 1 0 3 } { 1 3 2 1 0 } .
$$

Thus the expected logarithmic decrease is at least

$$
\ln ( 1 3 2 1 0 / 1 0 3 ) > \ln 1 2 8 .
$$

The parametrized family rules out every universal constant, not merely 7 ln 2.

The failure is not caused by taking a logarithm of zero. It persists with +1 regularization. The posterior on the threshold within the remaining label-undetermined set is not the fresh uniform posterior assumed by a naive potential argument.

## D Proof of Theorem 7.1

## 1. Conventions and the counterexamples

Budgets below are nonnegative integers. For a real-valued budget, replace � by ⌊�⌋.

Write $M _ { E } ^ { * } ( T , Q )$ and $M _ { N } ^ { * } ( T , Q )$ for the deterministic minimax mistake counts under conventions E and N, respectively. Lower bounds use the finite domains fixed in Section 2:

$\operatorname { E } \colon X = \{ s \} \cup V .$ , where � is the least point and � consists of the � instances;

$\mathrm { N } \colon X = V .$

All oracle answers are complete label vectors. No threshold endpoint name or additional representation information is assumed.

## Theorem D.1 — The conjectured formula is false (conventions E and N)

In the model of Section 2,

$$
M _ { E } ^ { * } ( 5 , 3 ) = 3 , \qquad M _ { N } ^ { * } ( 6 , 4 ) = 3 .
$$

In particular, the proposed E formula

$$
M _ { E } ^ { * } ( T , Q ) = \operatorname* { m a x } \{ T - Q , \lfloor \log _ { 2 } ( T + 1 ) \rfloor \}
$$

is false. Consequently no formula that assigns the value 2 to $M _ { N } ^ { * } ( 6 , 4 )$ can be correct either.

The proof is analytic.

## 2. What is correct in the protection algorithm

## Lemma D.2 — Computable knowledge and sound protection

Maintain a partial order $P _ { t }$ on the instance set �. Its relations comprise:

1. relations learned from previous oracle answers;

2. every relation $a \prec b$ implied by an observed positive � and an observed negative $b ;$

3. their reflexive/transitive closure.

Let $A _ { t } , B _ { t }$ be the observed positive and negative sets. Define the retained version space

$$
\mathcal { I } _ { t } ^ { E } = \{ I \subseteq V : I \mathrm { { i s ~ a n ~ i d e a l ~ o f ~ } } P _ { t } , \ A _ { t } \subseteq I , \ I \cap B _ { t } = \emptyset \} ,
$$

and

$$
\begin{array} { r } { \mathcal { I } _ { t } ^ { N } = \mathcal { I } _ { t } ^ { E } \setminus \{ \emptyset \} . } \end{array}
$$

An ideal is a downward-closed subset. Every ideal is a prefix of some linear extension of $P _ { t }$ : first topologically sort the ideal, then its complement. Thus these version spaces describe precisely the target vectors compatible with the information deliberately retained by the algorithm.

Define

$$
D _ { 1 } = \bigcap _ { I \in { \cal I } _ { t } } I , \qquad D _ { 0 } = V \setminus \bigcup _ { I \in { \cal I } _ { t } } I , \qquad U = V \setminus ( D _ { 0 } \cup D _ { 1 } ) .
$$

These sets are computable without knowing the true order. Realizability guarantees $\mathcal { T } _ { t } \neq \emptyset$

Under E, they have the simpler reachability descriptions

$$
D _ { 1 } = \{ \nu : \exists w \in A _ { t } , \ \nu \preceq _ { P _ { t } } \ w \} , \qquad D _ { 0 } = \{ \nu : \exists w \in B _ { t } , \ w \preceq _ { P _ { t } } \ \nu \} .
$$

Indeed, the smallest feasible ideal is ↓ $A _ { t } \colon \mathrm { i f } \ \nu \not \in D _ { 0 } .$ , then $\downarrow ( A _ { t } \cup \{ \nu \} )$ avoids $B _ { t }$ . Under N, enumeration of nonempty ideals also incorporates the nonemptiness inference that the reachability tests alone can miss.

Every previously observed point is determined. Consequently, if the current point � and another point � are both undetermined, then � has not arrived.

Query

$$
S = \{ ( x , 1 ) , ( u , 0 ) \} .
$$

• Answer ⊥. This happens exactly when � ≺ �. If � ≺ �, the threshold $c _ { x }$ realizes �; the reverse ordering makes � impossible. Predict 0. If this prediction is wrong, then $x = 1$ , and $u \prec x$ forces $u = 1$

• Answer ℎ. Put $H = \{ \nu \in V : h ( \nu ) = 1 \}$ . Then $x \in H , u \notin H$ , and

$$
H \prec V \setminus H .
$$

Predict 1. If this prediction is wrong, then $x = 0$ . Every point outside � follows �, so every such point is 0, including �.

The latter inference uses the fact that � is a threshold prefix, not merely that it contains �. Both branches work for every legal consistent return value. □

## Lemma D.3 — Correct accounting, including early termination

Run the protection algorithm with an internal query cap �: query at every undetermined arrival whenever budget remains and another undetermined point exists. Let $q \leq k$ be the number of calls actually made. Then

$$
\boxed { M \leq T - q . }
$$

If $q < k$ , the valid additional conclusion is

$$
\boxed { M \leq q + 1 , }
$$

not $M \leq 1$ . Therefore

$$
\begin{array} { r } { \boxed { M \le \operatorname* { m a x } \{ T - k , k \} . } } \end{array}\tag{D.1}
$$

Proof. Let � be the set of arrivals that were undetermined immediately before their prediction/query processing. Let � be the number of mistaken query rounds.

Each mistaken query protects a previously undetermined future point. Once protected, that point remains determined. Hence the protected points from the � mistaken queries are distinct and do not belong to �. Thus

$$
| D | + b \leq T .
$$

At most $| D | - q$ mistakes occur on nonquery rounds, giving

$$
M \leq b + | D | - q \leq T - q .
$$

If $q < k$ , budget was never exhausted. Any undetermined arrival at which no query was made must therefore have been the sole remaining undetermined point. After its label is observed, no future undetermined arrival is possible. Thus at most one member of $D$ is a nonquery round:

$$
| D | \leq q + 1 , \qquad M \leq q + 1 \leq k .
$$

Combining this with the $q = k$ case proves (D.1).

In particular, choosing

$$
k = \operatorname* { m i n } \{ Q , \lfloor T / 2 \rfloor \}
$$

gives the valid general upper bound

$$
M \leq \operatorname* { m a x } \{ T - Q , \lceil T / 2 \rceil \} .\tag{D.2}
$$

For Theorem D.1:

$\mathrm { E } , T = 5 , Q = 3 \mathrm { : }$ take � = 2, obtaining $M \leq 3 ;$

$\ N , T = 6 , Q = 4 ;$ take $k = 3$ , obtaining $M \leq 3 .$

The remaining work is to prove matching lower bounds for every deterministic learner.

## 3. An analytic five-point lower bound under E

Throughout this section there are five instances, an E sentinel, and no observed labels unless explicitly stated. A poset state means that every linear extension of that poset remains available to the adversary. Providing the poset to the learner for free only strengthens the learner.

## Lemma D.4 — Antichain cube

Suppose no queries remain and the available orders include all linear extensions of a poset � containing an antichain � of size �. Then the adversary can force � mistakes in the prescribed transductive sequence.

Proof. Let

$$
D = \{ \nu \notin W : \nu < _ { P } w { \mathrm { ~ f o r ~ s o m e ~ } } w \in W \} .
$$

For every $J \subseteq W$ , the set $D \cup J$ is an ideal. Label � positively and $V \setminus ( D \cup W )$ negatively. The labels on � are then independently selectable: every choice is a prefix in some linear extension of $P .$

At each arrival in �, choose the opposite of the deterministic prediction. Finally choose a linear extension in which the resulting positive ideal comes first, and take its corresponding threshold. □

## Lemma D.5 — Extreme-point padding

Under E, suppose one of five points, �, can be fixed as either

• the least instance, with target label 1; or

• the greatest instance, with target label 0,

while the other four instances have arbitrary relative order and arbitrary E-threshold target. With at most one further query, at least three mistakes can be forced.

Proof. This embeds the four-point E problem, to which Theorem 4.1 gives

$$
M \geq 4 - 1 = 3 .
$$

It remains to check that queries involving the padding point do not invalidate the embedding.

If $p$ is least:

• a query requiring $p = 0$ and some ordinary instance to be 1 is impossible;

• if it requires $p = 0$ but no ordinary positive, return the empty-instance prefix;

• otherwise, make at most one base ERM call and extend its vector by $p = 1$

If $p$ is greatest:

• a query requiring $p = 1$ and some ordinary instance to be 0 is impossible;

• if it requires $p = 1$ but no ordinary zero, return the all-one concept;

• otherwise, make at most one base call and extend its vector by $p = 0$

Conflicting labels and a negative sentinel are handled directly by ⊥. Previously returned vectors that are valid for every order in the embedded subclass can be cached and returned without a base call.

These are legal, deterministic, memoryless simulations on the full domains.

The same greatest-point simulation also works when the base class uses convention $\mathrm { N } ;$ this will be used later.

## Lemma D.6 — Predicting before resolving a matching costs three mistakes

Suppose the retained relations are

$$
a \prec b , \qquad c \prec d ,
$$

with a fifth point � otherwise unrelated. If the learner predicts the first label now, the adversary can force that prediction to be wrong and then force two further mistakes, even if the full order is subsequently disclosed.

The conclusion also holds with zero or one retained comparison, since such a state can be strengthened to the displayed matching.

Proof. For any current point � and either desired label �, construct three future points � such that:

1. all four prefix labelings on � are possible while $\nu = y ;$

2. the earliest-arriving point of � is its median.

The constructions below sufice; exchanging the two pairs handles $\nu = c , d .$

<table><tr><td>Current point</td><td>Desired label</td><td>Choose H</td><td>Order of the groups</td></tr><tr><td>a</td><td>0</td><td> $\overline { { \{ c , d , e \} } }$ </td><td> $\overline { { H < a < b } }$ </td></tr><tr><td>a</td><td>1</td><td> $\{ c , d , e \}$ </td><td> $a < H < b$ </td></tr><tr><td>b</td><td>0</td><td> $\{ c , d , e \}$ </td><td> $a < H < b$ </td></tr><tr><td>b</td><td>1</td><td>{c, d, e}</td><td> $a < b < H$ </td></tr><tr><td>e</td><td>0</td><td> $\{ b , c , d \}$ </td><td> $a < H < e$ </td></tr><tr><td>e</td><td>1</td><td> $\{ b , c , d \}$ </td><td> $a < e < H$ </td></tr></table>

Each triple has just one required comparison, $c < d .$ Its earliest point can always be made the median. For the triple $\{ c , d , w \}$ , use respectively

$$
w < c < d , \qquad c < d < w , \qquad c < w < d
$$

when the earliest point is $c , d ,$ , �.

Choose � opposite to the current prediction. On the triple, make the median prediction wrong. Its label leaves two possible thresholds difering on one later extreme; make that prediction wrong too. The fourth future point has a fixed label common to all four triple thresholds.

An ERM oracle reveals the class, not the target. Even full knowledge of this order cannot distinguish these remaining target thresholds before their labels arrive. □

## Lemma D.7 — The second-query alternatives

Suppose the only retained comparison is $a < b$ . Any query can be answered so that the remaining orders contain either:

1. a subclass covered by Lemma D.5; or

2. all extensions of two disjoint comparisons.

Proof. Remove the sentinel from the query after checking its label, and handle conflicts, empty queries, and single-sided queries directly. Such a query can receive a no-information answer, after which fixing � as the least instance gives case 1.

For a genuine mixed query, let $A \neq \emptyset$ be its positive set and $B \neq \emptyset$ its negative set.

If there are $p \in A , n \in B$ such that adding $n < p$ is consistent with $a < b$ and creates no three-element chain, answer ⊥ and retain this relation. The resulting poset is one of:

• one comparison;

• a two-edge fork;

• two disjoint comparisons.

For a fork, put its common source globally first, or its common sink globally last. The other four points are unrestricted, giving case 1. Disjoint comparisons give case 2. If the added relation is already $a < b$ , fix � as the least instance and set its target label to 1; the other four instances retain arbitrary relative order and arbitrary E-prefix targets, which satisfies the pre-existing comparison and makes the present ⊥ universally legal, so this is case 1 as well.

Otherwise, $a \notin B$ and � ∉ �: either membership would provide a permitted reversed pair. Moreover, if both $A \setminus \{ a \}$ and $B \setminus \{ b \}$ were nonempty, a pair between these sets would be disjoint from $a < b ,$ , again permitted. Hence

$$
A = \{ a \} \quad { \mathrm { o r } } \quad B = \{ b \} .
$$

In the first case return the prefix $\{ a \}$ ; in the second return $V \backslash \{ b \}$ . These are respectively consistent with making � least or � greatest, leaving the other four points unrestricted. □

## Lemma D.8 — One query cannot eliminate a three-point antichain from the matching

From the matching state

$$
a < b , \qquad c < d ,
$$

plus isolated �, every query has a legal answer leaving a poset with a three-point antichain.

Proof. Put

$$
L = \{ a , c \} , \qquad R = \{ b , d \} .
$$

Trivial queries can receive a no-information answer. For a mixed query with positive set � and negative set $B \colon$

• If $B \cap L \neq \emptyset$ , choose $\ell \in B \cap L$ and $p \in A$ . Answer ⊥, retaining $\ell < p$ . The set $R \cup \{ e \}$ remains an antichain.

• Otherwise, if $A \cap R \neq \emptyset$ , choose $r \in A \cap R$ and $n \in B$ . Answer ⊥, retaining $n < r .$ . The set $L \cup \{ e \}$ remains an antichain.

• Otherwise,

$$
A \subseteq L \cup \{ e \} , \qquad B \subseteq R \cup \{ e \} .
$$

Return the prefix

$$
H = { \left\{ \begin{array} { l l } { L , } & { e \in B , } \\ { L \cup \{ e \} , } & { e \notin B . } \end{array} \right. }
$$

This respects the matching and the query. Its three-point side is an antichain.

In each ⊥ case, a requested negative has been placed before a requested positive, making the query genuinely unrealizable. □

## Proposition $\mathbf { D } . 9 - M _ { E } ^ { * } ( 5 , 3 ) \geq 3$

Proof. Fix any deterministic learner making at most three calls.

1. Before the first call. If it predicts, Lemma D.6 forces three mistakes. Otherwise, answer its first mixed query $\boldsymbol { \mathrm { b y \perp } }$ , retaining one reversed requested pair $a < b .$ . A trivial query can receive a no-information answer, and one comparison may then be supplied for free.

1. After the first call. If it predicts, Lemma D.6 again applies. Otherwise apply Lemma D.7 to its second call. An extreme-point/fork outcome invokes Lemma D.5, with at most one call left. The only remaining case is two disjoint comparisons.

1. In the matching case. Predicting before the last call loses by Lemma D.6. If the learner makes the last call, Lemma D.8 leaves a three-point antichain. No queries remain, so Lemma D.4 forces three mistakes.

This covers algorithms that use fewer than three calls as well.

Oblivious solidification and memorylessness. Cache every queried set and its answer; repeated queries receive their cached answer. Such repetitions consume budget without imposing additional restrictions.

Every returned nontrivial vector was retained as a prefix cut. Every nontrivial ⊥ answer has a retained reversed pair witnessing impossibility. At the end, choose a total-order extension and a target prefix realizing the constructed labels. Define

$$
\begin{array} { r } { O ^ { * } ( S ) = \left\{ \begin{array} { l l } { \mathrm { t h e ~ c a c h e d ~ a n s w e r , ~ } } & { S \mathrm { ~ w a s ~ q u e r i e d , } } \\ { \mathrm { t h e ~ l e a s t ~ c o n s i s t e n t ~ t h r e s h o l d ~ i n ~ t h e ~ f i n a l ~ o r d e r , ~ } } & { S \mathrm { ~ i s ~ r e a l i z a b l e ~ a n d ~ u n q u e r i e d , } } \\ { \perp , } & { \mathrm { o t h e r w i s e . } } \end{array} \right. } \end{array}
$$

All cached answers are legal in the final order. In the padding branch, use the fixed oracle from Theorem 4.1 through the explicit simulation of Lemma D.5.

Thus $O ^ { * }$ is a legal function of � alone. Determinism makes the replay identical. The resulting order, target, sequence, and oracle are all fixed before that replay, as required by the oblivious-instance requirement.

All choices can be tie-broken by arrival index. Final order extensions can be obtained by lexicographic topological sorting, with the target ideal first. This is a finite constructive solidification, not a compactness argument. □

Lemma D.3 supplies the matching upper bound, proving

$$
\boxed { M _ { E } ^ { * } ( 5 , 3 ) = 3 } .
$$

## 4. Embedding the known-class lower bound in the transductive model

## Lemma D.10 — Known-order lower bounds

For every query budget,

$$
M _ { E } ^ { * } ( T , Q ) \geq \lfloor \log _ { 2 } ( T + 1 ) \rfloor , \qquad M _ { N } ^ { * } ( T , Q ) \geq \lfloor \log _ { 2 } T \rfloor .
$$

Proof under E. Put $d = \lfloor \log _ { 2 } ( T + 1 ) \rfloor$ and choose $2 ^ { d } - 1$ ordered instances. Restrict the target to their $2 ^ { d }$ prefixes, including the empty prefix. Present these instances in breadth-first order of the perfectly balanced binary search tree; present any extra, always-negative instances afterwards.

The entire sequence is fixed in advance. At each depth, the node on the currently surviving search path splits the possible target cuts equally. Choose its label opposite to the deterministic prediction. Other nodes encountered outside the surviving interval have labels common to all surviving targets. This forces one mistake at each of the � depths.

The full order may be disclosed for free. Fix, for example, the least-consistent-threshold ERM rule. Given the order, all oracle answers are simulatable and contain no information about which target cut was chosen.

The adaptive choice of target cuts is solidified by selecting the final surviving target and replaying the deterministic interaction.

Under N, reserve the least instance as an always-positive point. Use $2 ^ { d } - 1$ further instances for the same construction, where $d = \lfloor \log _ { 2 } T \rfloor$ . The reserved minimum supplies the “empty” prefix on the hard instances while the target on the full domain remains nonempty. □

This is a worst-sequence statement. It does not assert that every known-order instance permutation has this mistake complexity.

## 5. The N convention also has a strict counterexample

## Lemma D.11 — First-action lower recurrence under N

For $n \geq 2$ and $q \geq 1$

$$
M _ { N } ^ { * } ( n , q ) \geq \operatorname* { m i n } \left\{ M _ { E } ^ { * } ( n - 1 , q - 1 ) , \ 1 + M _ { E } ^ { * } ( n - 1 , q ) , \ 1 + M _ { N } ^ { * } ( n - 1 , q ) \right\} .\tag{D.3}
$$

Proof.

If the first action is a query, it can always be answered while leaving some point � globally least and all other relative orders unrestricted:

• Mixed positive/negative query: choose � among its negatives and answer ⊥. A globally least point cannot be negative under N.

• Negative-only query on a proper subset: choose � outside that subset and return the singleton prefix $\{ p \}$

• Negative-only query on the whole domain: answer ⊥.

• Positive-only or empty query: return the all-one concept.

• Conflicting labels: answer ⊥.

Disclose $p$ as the minimum for free. Its target label is necessarily 1; the other $n - 1$ points form an E problem with at most $q - 1$ calls remaining. Cached first answers are universally legal in this subclass.

If the first action is prediction 0, give label 1, make that point least, and retain an E problem on the remaining $n - 1$ points.

If the first action is prediction 1, give label 0, make that point greatest, and retain an N problem on the remaining � − 1 points. The additional all-one concept introduced by the greatest point is handled by the simulation in Lemma D.5.

These three cases prove (D.3).

Apply (D.3) with $n = 6 , q = 4$ . Proposition D.9 and Lemma D.10 give

$$
\begin{array} { c } { { M _ { E } ^ { * } ( 5 , 3 ) = 3 , } } \\ { { 1 + M _ { E } ^ { * } ( 5 , 4 ) \geq 1 + \lfloor \log _ { 2 } 6 \rfloor = 3 , } } \\ { { 1 + M _ { N } ^ { * } ( 5 , 4 ) \geq 1 + \lfloor \log _ { 2 } 5 \rfloor = 3 . } } \end{array}
$$

Therefore $M _ { N } ^ { * } ( 6 , 4 ) \ge 3$ . Lemma D.3, with internal cap $k = 3$ , gives the reverse inequality:

$$
\boxed { M _ { N } ^ { * } ( 6 , 4 ) = 3 } .
$$

Consequently, any proposed formula assigning the value 2 to $( T , Q ) = ( 6 , 4 )$ is false.

## Further N conclusions

Equation (D.3), Theorem 4.1 under E, and induction on � yield the strengthened budget lower bound (this is a budget statement for $Q \geq 1$ , obtained by the first-action recurrence and induction rather than directly from the pathwise bound of Theorem 4.1 under N: for $Q \geq 1$ the three branches of (D.3) give $M _ { E } ^ { * } ( T - 1 , Q - 1 ) \geq T - Q$ and $1 + M _ { E } ^ { * } ( T - 1 , Q ) \geq T - Q$ by Theorem 4.1 under E, and $1 + M _ { N } ^ { * } ( T - 1 , Q ) \geq T - Q$ by the induction hypothesis, while the base case $T = 1$ has a nonpositive right-hand side)

$$
M _ { N } ^ { * } ( T , Q ) \geq T - Q \qquad ( Q \geq 1 ) .
$$

Together with Lemma D.10,

$$
M _ { N } ^ { * } ( T , Q ) \geq \left\{ \begin{array} { l l } { T - 1 , } & { Q = 0 , } \\ { \operatorname* { m a x } \{ T - Q , \lfloor \log _ { 2 } T \rfloor \} , } & { Q \geq 1 . } \end{array} \right.\tag{D.4}
$$

Predicting 1 throughout makes at most $T - 1$ mistakes under N. Consequently,

$$
\boxed { M _ { N } ^ { * } ( T , 0 ) = M _ { N } ^ { * } ( T , 1 ) = T - 1 . }\tag{D.5}
$$

This is a budget statement; it is not a new pathwise assertion $M + q \geq T$ when the actual query count might be zero.

## 6. Valid bounds after rejecting the claimed exact curve

The results established above imply

$$
\operatorname* { m a x } \{ T - Q , \lfloor \log _ { 2 } ( T + 1 ) \rfloor \} \le M _ { E } ^ { * } ( T , Q ) \le \operatorname* { m a x } \{ T - Q , \lceil T / 2 \rceil \} ,\tag{D.6}
$$

and the N lower bound (D.4), together with

$$
M _ { N } ^ { * } ( T , Q ) \leq \operatorname* { m i n } \left\{ T - 1 , \operatorname* { m a x } \{ T - Q , \lceil T / 2 \rceil \} \right\} .\tag{D.7}
$$

There is also exact large-budget saturation:

$$
\begin{array} { r } { Q \geq 2 ( T - 1 ) \quad \Longrightarrow \quad M _ { E } ^ { * } ( T , Q ) = \lfloor \log _ { 2 } ( T + 1 ) \rfloor , \quad M _ { N } ^ { * } ( T , Q ) = \lfloor \log _ { 2 } T \rfloor . } \end{array}
$$

For completeness, the explicit count follows from the sorting method of AHR25 (Thm. 4.5(3)): query a pair in one orientation, and if necessary in the reverse orientation. A successful return splits the current set into two nonempty ordered parts. There are $T - 1$ internal splits, each costing at most two calls. Once sorted, maintain the set $V _ { t }$ of prefix concepts consistent with the labels seen so far and predict its majority label; every mistake at least halves $\left| V _ { t } \right|$ , while the true target is never removed, so $1 \leq | V _ { 0 } | 2 ^ { - M }$ and $M \leq \lfloor \log _ { 2 } \left. V _ { 0 } \right. \rfloor$ , with $\left| V _ { 0 } \right| = T + 1$ under E and $| V _ { 0 } | = T$ under N. (The suficient budget $2 ( T - 1 )$ is not claimed to be minimal; for instance $M _ { E } ^ { * } ( 2 , 1 ) = 1$ : one pair query fixes the order of two points and halving then makes one mistake, matching the lower bound.)

Equations (D.6)–(D.7) are bounds, not an exact characterization. The strict finite counterexamples show why the conjectured equality cannot be assembled from Theorem 4.1 and the known-class Halving bound alone.

Theorem 4.4 of AHR25 is not invoked here; its statement, and why its hypothesis fails for the threshold family, are discussed once in Appendix B. (For the E counterexample, $T = 5$ and $d _ { \mathrm { L D } } = 2$ , so its hypothesis $T \geq 2 ^ { d _ { \mathrm { L D } } + 1 } = 8$ fails.) Theorem 4.1 is invoked as a proved theorem; the deterministic/randomized separation is unafected.

## E Proof of Theorem 7.2

## Theorem 7.2: deterministic linear-query learning with a weak consistency oracle

Interface and conventions. Throughout this section, the learner accesses the class only through the Boolean oracle of Definition 2.2 of AHR25. It never receives or evaluates a returned concept. The concept-returning interface of Section 2 is used only in the later comparison table.

Let the $T \geq 1$ instances be distinct and revealed in advance, as stipulated in the transductive protocol. Labels are generated by an arbitrary target

$$
c _ { z } ( x ) = 1 [ x \preceq z ]
$$

under an arbitrary unknown total order ⪯.

Write $\epsilon = 0$ under convention E and $\epsilon = 1$ under convention N:

• E: the domain contains a point preceding all � instances, so the empty prefix on the instances is available.

• N: the domain consists exactly of the � instances, so the empty prefix is unavailable.

## Theorem 7.2 — explicit bounds

There are deterministic algorithms using only weak consistency queries such that, for every admissible domain, total order, target, and transductive sequence,

<table><tr><td>Convention</td><td>Number of queries</td><td>Number of mistakes</td></tr><tr><td>E</td><td> $Q \leq 6 6 T$ </td><td> $\overline { { M \leq \lfloor \log _ { 2 } ( T + 1 ) \rfloor } }$ </td></tr><tr><td>N</td><td> $Q \leq 6 7 ( T - 1 )$ </td><td> $M \leq \lfloor \log _ { 2 } T \rfloor .$ </td></tr></table>

Every oracle query made by these algorithms has exactly the form

$$
\{ ( u , 1 ) , ( \nu , 0 ) \} , \qquad u \neq \nu .
$$

Consequently, in either convention,

$$
Q \leq 6 7 T , \qquad M \leq \log _ { 2 } T + 1 .
$$

These are pathwise bounds: there is no randomization and no averaging over oracle answers.

## Lemma E.1 — comparison interface and adaptive consistency

For distinct $u , \nu \in X$

$$
\mathrm { W C } \big ( \{ ( u , 1 ) , ( \nu , 0 ) \} \big ) = \mathrm { r e a l i z a b l e } \quad \iff \quad u \prec \nu .
$$

This holds under both E and N.

Proof. Realizability means that some $z \in X$ satisfies

$$
u \preceq z \prec \nu ,
$$

which implies $u \prec \nu$ . Conversely, if $u \prec \nu .$ , the concept $c _ { u }$ realizes the query. No empty-prefix assumption is needed. □

The query concerns the existence of some consistent concept, not consistency with the actual target. In particular, observed target labels must not be appended to this comparison query.

The comparison implementation is also valid against adaptively supplied comparison answers, provided the answers remain consistent with a total order. Indeed, fix any completed total order consistent with a finite comparison transcript. A deterministic comparison algorithm, run against that fixed order, follows exactly the same transcript. Thus every correctness and comparison-count guarantee proved for arbitrary fixed total orders holds for every consistent adaptive transcript.

For a finite instance set, a consistent partial order has a total-order extension: repeatedly remove a minimal remaining element. This is the only extension fact needed here.

## Lemma E.2 — exact selection in at most 32� comparisons

There is a deterministic comparison algorithm

$$
\operatorname { S E L E C T } ( A , k ) , \qquad 1 \leq k \leq n = | A | ,
$$

which returns the element of rank � using at most 32� comparisons.

Therefore, an exact median of rank

$$
k = \left\lceil { \frac { n } { 2 } } \right\rceil
$$

can be found using at most 32� weak consistency queries, and partitioning the other elements relative to it costs at most $n - 1$ additional queries.

## Algorithm.

• If $\dot { n } \leq 6 4$ , insertion-sort the elements and return the element of rank �.

• Otherwise, let $m = \lfloor n / 5 \rfloor$ . Form � complete groups of five; leave at most four elements outside these groups.

• Insertion-sort each complete group and collect its median.

• Recursively select the median � of those � medians, using rank $a = \lceil m / 2 \rceil$

• Compare every other original element with $q .$

• Return �, or recurse into the appropriate side with the appropriately adjusted rank.

All grouping and tie-independent choices use the original arrival-index order, making the procedure deterministic.

Correctness. The partition determines the exact rank of $q .$ . The desired element is therefore either � itself or the desired adjusted rank in exactly one of the two sides. Induction proves correctness.

Comparison count. Let $C ( n )$ be the worst-case number of comparisons, with $C ( 0 ) = 0$ . For $n \leq 6 4$ insertion sort uses at most

$$
\frac { n ( n - 1 ) } { 2 } \leq \frac { 6 3 } { 2 } n \leq 3 2 n .
$$

Now suppose $n \geq 6 5$ . Sorting the complete groups costs at most 10� comparisons, and partitioning around � costs at most $n - 1$

There are at least

$$
3 ( a - 1 ) + 2 = 3 a - 1
$$

elements strictly below �. There are at least

$$
3 ( m - a ) + 2 \geq 3 a - 1
$$

elements strictly above $q ;$ the last inequality follows directly for both parities of $m .$

Thus the side used by the second recursive call has size � satisfying

$$
s \leq n - 3 a .
$$

Consequently,

$$
m + s \leq n + m - 3 \left\lceil { \frac { m } { 2 } } \right\rceil \leq n - { \frac { m } { 2 } } \leq { \frac { 9 n } { 1 0 } } + { \frac { 2 } { 5 } } ,
$$

where the last step uses $m \geq ( n - 4 ) / 5$

Induction now gives

$$
\begin{array} { c l } { { } } & { { C ( n ) \leq C ( m ) + C ( s ) + 1 0 m + n - 1 } } \\ { { } } & { { \leq 3 2 ( m + s ) + 3 n - 1 } } \\ { { } } & { { \leq 3 2 \left( \displaystyle \frac { 9 n } { 1 0 } + \displaystyle \frac { 2 } { 5 } \right) + 3 n - 1 } } \\ { { } } & { { = \displaystyle \frac { 1 5 9 n + 5 9 } { 5 } } } \\ { { } } & { { \leq 3 2 n , } } \end{array}
$$

because $n \geq 6 5 > 5 9$ . This proves the stated explicit constant.

For $n \geq 2$ , the selected rank satisfies

$$
{ \frac { 3 n } { 1 0 } } \leq \left\lceil { \frac { n } { 2 } } \right\rceil \leq { \frac { 7 n } { 1 0 } } .
$$

For $n = 2$ this is immediate; for $n \geq 3$ , use $\lceil n / 2 \rceil \leq ( n + 1 ) / 2 \leq 7 n / 1 0$ . The literal middle-rank interval contains no integer when $n = 1 ;$ a singleton instead requires no comparisons.

## Lemma E.3 — a phase contains at most one mistake

The learner maintains a partial map � of certified labels. Every certification will be proved correct. Let

$$
U = \{ x _ { 1 } , . . . , x _ { T } \} \setminus \mathrm { d o m } ( K ) .
$$

Thus every point in $U$ is unobserved and uncertified. Certification may be conservative: � need not contain exactly the logically undetermined points, and it need not be an interval in the unknown order.

At the start of a phase, freeze

$$
V = U , \qquad n = | V | \geq 1 .
$$

Using Lemma E.2, find the element $p$ of rank $k = \lceil n / 2 \rceil$ in $V .$ , and partition

$$
L = \{ u \in V : u \prec p \} , \qquad R = \{ u \in V : p \prec u \} .
$$

Throughout this phase:

• Predict the certified label on points already in dom(�).

• On an uncertified point in $L \cup \{ p \}$ , predict 1.

• On an uncertified point in �, predict 0.

• Certify each observed label.

• Following a mistake:

• if the prediction was 1, certify every point of $\{ p \} \cup R$ as 0;

• if the prediction was 0, certify every point of $L \cup \{ p \}$ as 1.

End the phase, before another prediction, whenever

$$
| U | \leq \rho ( n ) , \qquad \rho ( n ) : = \left\lfloor { \frac { n - 1 } { 2 } } \right\rfloor .
$$

Then every certification is correct, and each phase contains at most one mistake.

Proof. Certified points cannot cause mistakes, assuming the certification invariant.

Consider a mistake on an uncertified point �. The following table lists all possibilities.

<table><tr><td>Mistake</td><td>Valid threshold implication</td><td>Active set after certification</td></tr><tr><td> $\overline { { x \in L } }$  prediction 1, label 0</td><td>Every point in  $\overline { { \{ p \} \cup R \mathrm { i s } 0 } }$ </td><td> ${ \overline { { U ^ { \prime } \subseteq L \setminus \{ x \} } } }$ </td></tr><tr><td> $x = p ,$  prediction 1, label 0</td><td>Every point in  $\{ p \} \cup R \mathrm { i s } 0$ </td><td> $U ^ { \prime } \subseteq L$ </td></tr><tr><td> $x \in R ,$  prediction 0, label 1</td><td>Every point in  $L \cup \{ p \}$  is 1</td><td> $U ^ { \prime } \subseteq R \backslash \{ x \}$ </td></tr></table>

For example, in the first case,

$$
z \prec x \prec p ,
$$

so � and every point above it are negative. In the third case,

$$
p \prec x \preceq z ,
$$

so � and every point below it are positive. These implications also show that new certifications never conflict with previous correct certifications.

Since $| L | = k - 1$ and $\left| R \right| = n - k$ , the corresponding size bounds are

$$
k - 2 , \qquad k - 1 , \qquad n - k - 1 .
$$

Cases involving an empty side simply cannot occur. In every possible case,

$$
| U ^ { \prime } | \leq \operatorname* { m a x } \{ k - 1 , n - k - 1 \} = \left\lfloor { \frac { n - 1 } { 2 } } \right\rfloor = \rho ( n ) .
$$

Therefore, the first mistake immediately ends the phase. There can be no second mistake in that phase.

Importantly, these are containments in the original phase sets �, �. They remain valid even if many points have already been removed by correct predictions. No lower bound on the number of newly removed points is required. □

## Proof of Theorem 7.2

## Convention E

Initially, � is empty and the active set has size �. Repeatedly execute the phases of Lemma E.3. If� becomes empty, all remaining predictions use certified labels.

Let the positive phase-start sizes be

$$
n _ { 1 } , n _ { 2 } , \ldots , n _ { J } .
$$

Then $n _ { 1 } = T$ , and the phase-ending rule gives

$$
n _ { i + 1 } \leq \left\lfloor { \frac { n _ { i } - 1 } { 2 } } \right\rfloor , \qquad { \mathrm { h e n c e } } \qquad n _ { i + 1 } + 1 \leq { \frac { n _ { i } + 1 } { 2 } } .
$$

Since $n _ { J } \geq 1$

$$
2 \leq n _ { J } + 1 \leq \frac { T + 1 } { 2 ^ { J - 1 } } ,
$$

which implies

$$
J \leq \lfloor \log _ { 2 } ( T + 1 ) \rfloor .
$$

Lemma E.3 therefore yields

$$
M \leq J \leq \lfloor \log _ { 2 } ( T + 1 ) \rfloor .
$$

The query cost of a phase of size $n _ { i }$ is at most

$$
3 2 n _ { i } + \left( n _ { i } - 1 \right) \leq 3 3 n _ { i } .
$$

Furthermore, $n _ { i + 1 } \leq n _ { i } / 2 ,$ , so

$$
\sum _ { i = 1 } ^ { J } n _ { i } \leq 2 T .
$$

Thus

$$
Q \leq 3 3 \sum _ { i } n _ { i } \leq 6 6 T .
$$

## Convention N

Here � consists exactly of the � instances. First find its minimum � by a sequential minimum scan, using exactly $T - 1$ comparisons.

Every target satisfies $a \preceq z , \operatorname { s o } c _ { z } ( a ) = 1$ . Certify � as positive before online prediction begins.

Run the same phase algorithm on the remaining $T - 1$ active points. The preceding proof, with initial active size $T - 1$ , gives

$$
M \leq \lfloor \log _ { 2 } T \rfloor
$$

and

$$
Q \leq ( T - 1 ) + 6 6 ( T - 1 ) = 6 7 ( T - 1 ) .
$$

For $T = 1$ , this means zero queries and zero mistakes under N. Under $\mathrm { E , }$ the singleton phase uses zero queries and makes at most one mistake.

Consequently, under either convention, $Q \leq 6 7 T$ and $M \leq \lfloor \log _ { 2 } ( T + 1 ) \rfloor \leq \log _ { 2 } T + 1 ;$ ; the sharper convention-specific bounds are the ones stated above. □

Convention correction. The E algorithm itself also works unchanged under N, with the same $6 6 T$ query bound and at most $\lfloor \log _ { 2 } ( T + 1 ) \rfloor$ mistakes. The minimum-scan variant obtains the sharper N bound $\lfloor \log _ { 2 } T \rfloor$ Thus the integer mistake correction is exactly the replacement of $T + 1$ by $T ,$ a diference of at most one, matching the two Littlestone dimensions of Section 2.

## Weak consistency: matching linear lower bounds

## The quoted lower bound

The statement of Theorem 4.3 of AHR25 is:

Consider any family $\mathcal { F }$ of concept classes of the form $C \subset \{ 0 , 1 \} ^ { X }$ , where the family $\mathcal { F }$ has the property that for every labeling function $f : \{ x _ { 1 } , x _ { 2 } , \ldots , x _ { T } \}  \{ 0 , 1 \}$ , there exists some concept class $C \in { \mathcal { F } }$ and some concept $c \in C$ such that $c ( x _ { t } ) = f ( x _ { t } )$ for all $t \in [ T ]$ (i.e., all $2 ^ { T }$ possible binary labelings of the sequence � are captured by the family F). For $T \geq 1 0 0$ any (possibly randomized) algorithm that makes at most $T / 2 0$ queries to the weak consistency oracle will incur an expected mistake bound of at least $T / 2 0$ . In particular, this theorem holds for general classes like Littlestone classes, and also for special families of classes, including thresholds, �-intervals, and �-Hamming balls (see the next section).

## Applicability under E

Use the finite domain

$$
X = \{ s , x _ { 1 } , \ldots , x _ { T } \} ,
$$

with � preceding all instances. For an arbitrary labeling $f ,$ order the points as

$s \prec$ {positive instances, in arrival-index order} ≺ {negative instances, in arrival-index order}.

If there is a positive instance, choose the last positive instance as $z ;$ otherwise choose $z = s .$

This realizes every one of the $2 ^ { T }$ labelings. Therefore, for $T \geq 1 0 0$ , a hard cap of $T / 2 0$ weak consistency queries entails a worst-case expected mistake bound of at least $T / 2 0$ . Deterministic algorithms are included as a special case.

## Applicability under N: the necessary one-point correction

The full �-point N family does not realize the all-zero labeling, so the hypothesis of Theorem 4.3 of AHR25 should not be applied to those � points without adjustment.

Instead, designate the first actual instance � as the minimum and restrict to orders with that property. Its label is always 1. On the remaining $T - 1$ instances, every binary labeling is realizable by arranging positives before negatives and choosing $z = s$ when all remaining labels are zero.

Given a learner for the full N problem, simulate it on

$$
( s , x _ { 2 } , \ldots , x _ { T } ) ,
$$

supply the known first label 1, and count its mistakes on the remaining sequence. Queries are unchanged, and the remaining mistakes are no greater than its total mistakes.

Theorem 4.3 of AHR25 therefore applies with efective horizon

$$
h = T - 1 .
$$

For $T \geq 1 0 1$ , a hard cap of $( T - 1 ) / 2 0$ queries entails at least $( T - 1 ) / 2 0$ expected mistakes in the worst case.

The designated minimum is an actual instance of the full N problem, not an additional unavailable query point.

## The oracle is fixed and memoryless

There is no adversarial choice of a Boolean answer once the order is fixed. For completeness, on either finite lower-bound domain, let $r ( x ) \in \{ 1 , \ldots , d \}$ be the rank of �. Define

$$
a ( S ) = \operatorname* { m a x } \big ( \{ 1 \} \cup \{ r ( x ) : ( x , 1 ) \in S \} \big ) ,
$$

$$
b ( S ) = \operatorname* { m i n } \big ( \{ d \} \cup \{ r ( x ) - 1 : ( x , 0 ) \in S \} \big ) .
$$

The weak oracle is the deterministic function

$$
\boxed { \mathbf { W C } ( S , \preceq ) = \mathbf { 1 } [ a ( S ) \leq b ( S ) ] . }
$$

This handles arbitrary query sizes, empty queries, one-label queries, contradictory labels, and every available domain point.

The lower-bound invocation is Theorem 4.3 of AHR25 in its oblivious-adversary model. No adaptive oracle construction is being substituted for an oblivious instance.

## Lemma E.4 — expected-query budgets also require linear queries

Let

$$
h = T - \epsilon \geq 1 0 0 .
$$

Suppose a possibly randomized learner has worst-instance expected mistake bound � and worst-instance expected query bound �. Then

$$
\boxed { m + 2 0 q \geq \frac { h } { 2 0 } . }
$$

In particular,

$$
q \geq \frac { h } { 4 0 0 } - \frac { m } { 2 0 } .
$$

Proof. Work on the efective ℎ-round problem above, including the N simulation when necessary.

$$
b = \left\lfloor { \frac { h } { 2 0 } } \right\rfloor .
$$

Construct a truncated learner that follows the original learner, using the same private random bits, until it would make query $b + 1$ . At that point it makes no further queries and uses an arbitrary fixed prediction for all remaining rounds.

The truncated learner has a hard query cap �. On any fixed instance, let � be the event that truncation occurs. Coupling the two runs gives

$$
M _ { \mathrm { t r u n c } } \leq M _ { \mathrm { o r i g i n a l } } + h \mathbf { 1 } _ { A } .
$$

Also,

$$
\mathrm { P r } ( A ) \leq \frac { \mathbb { E } [ Q _ { \mathrm { o r i g i n a l } } ] } { b + 1 } ,
$$

because � implies $Q _ { \mathrm { o r i g i n a l } } \geq b + 1$

Theorem 4.3 of AHR25 supplies a fixed instance on which

$$
\mathbb { E } [ M _ { \mathrm { t r u n c } } ] \geq \frac { h } { 2 0 } .
$$

On that same instance,

$$
\frac { h } { 2 0 } \leq m + \frac { h } { b + 1 } q \leq m + 2 0 q .
$$

This proves the claim.

□

Thus a logarithmic expected mistake guarantee requires

$$
q \ge \frac { T - \epsilon } { 4 0 0 } - O ( \log T ) = \Omega ( T ) .
$$

For example, the hard-cap statement alone implies that

$$
q \leq { \frac { h } { 8 0 0 } } \quad \Longrightarrow \quad m \geq { \frac { h } { 4 0 } }
$$

under an expected-query budget.

If the unqualified query budget of Theorem 4.3 of AHR25 is read directly as the expected quantity � defined in Section 2, its quoted $T / 2 0$ implication applies directly to that budget as well. The argument above avoids needing that interpretation: the required expected-query $\Omega ( T )$ lower bound follows even from the hard-cap reading alone.

## Corollary E.5 — tight weak-consistency query order

For either E or N, the query complexity needed to guarantee ${ \cal O } ( \log T )$ mistakes is

$$
\boxed { \Theta ( T ) }
$$

for both deterministic and randomized learners.

For randomized learners this remains true when query and mistake budgets are worst-instance expectations. The upper bound may be taken from Theorem 4.5(2) of AHR25, or, more strongly regarding query tails, from the deterministic algorithm of Theorem 7.2.

## Interface comparison and the precise separation statement

The table concerns ${ \cal O } ( \log T )$ mistakes. Randomized query bounds use worst-instance expectation, as in the cost accounting.

<table><tr><td>Oracle interface</td><td>Deterministic query complexity</td><td>Randomized query complexity</td></tr><tr><td>Consistency-type ERM: returns a full concept or ⊥</td><td>Θ(T) for worst-case legal oracles and for the extremal rules of The- orem 3.1 (not uniform over all pre- declared rules; see Theorem 3.4) — Theorems 3.1 and 4.1 give Q ≥ T – € – O(log T); the sorting algorithm of AHR25 gives O(T)</td><td>Θ(log T) — the O(log T) upper bound is AHR25, Thm. 4.5(4); the matching lower bound is Theo- rem 3.2</td></tr><tr><td>Weak consistency: returns only re- alizable / not realizable</td><td>Θ(T) — Theorem 7.2 together with Theorem 4.3 of AHR25</td><td>Θ(T) — Theorem 4.5(2) of AHR25 (or Theorem 7.2) together with Theorem 4.3 of AHR25 and Lemma E.4</td></tr></table>

## Interpretive proposition — scope of the interface separation

In the model of Section 2 and for logarithmic mistake guarantees on unknown-order thresholds:

1. Deterministic learners require linear-order query complexity under either interface: at the weak consistency interface unconditionally, and at the ERM interface against worst-case legal oracles and against the pre-declared extremal rules of Theorem 3.1. The ERM statement is not uniform over all legal predeclared selection rules: the feasible-median rule admits deterministic �(log �) queries and mistakes (Theorem 3.4).

2. Randomized learners can use �(log �) expected queries when a successful query returns an evaluable full concept, by AHR25, Thm. 4.5(4).

3. Randomized learners still require Ω(�) expected queries when only the realizability bit is returned.

Thus the established linear-versus-logarithmic improvement from randomization is available with the concept-returning ERM interface, but disappears at the level of query order with the weak consistency interface.

This conclusion does not require Theorem 3.2. Theorem 3.2 is needed only to upgrade the randomized ERM upper bound to a matching Θ(log �) characterization. The separation is exponential when expressed in the parameter � = Θ(log �).

The statement is about these two interfaces, this family, and this mistake regime. It does not assert that randomization cannot improve constants or other weak-oracle tradeofs.

Theorem 4.4 of AHR25 is not invoked here; its statement, and why its hypothesis fails for the threshold family, are discussed once in Appendix B. The weak-oracle lower bound above comes from Theorem 4.3 of AHR25, and the deterministic ERM lower bound from Theorems 3.1 and 4.1.