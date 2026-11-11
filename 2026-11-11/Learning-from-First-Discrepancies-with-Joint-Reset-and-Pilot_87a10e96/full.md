# Learning from First Discrepancies with Joint Reset and Pilot Guarantees

Michal Certik The World bank Group michal.certik@gmail.com

November 2026

## Abstract

A run that stops at its first wrong decision shows the queries and their correct labels up to and including that decision, and nothing after it. We study how many such runs are needed for learning. Each run tests a hypothesis, the pilot, chosen beforehand from a fixed binary class of VC dimension at most d, on a fresh record of at most D queries drawn from one unknown distribution that does not depend on the pilot. The terminal information of a record is shown only if the pilot makes no discrepancy, and a bounded loss, fixed in advance, is attached to the position where a hypothesis stops. If the discrepancy probabilities of members of the class have infimum at most f and their risks have infimum at most r, one learner uses $\widetilde { O } ( ( d + 1 ) [ D ( f + \varepsilon ) + r ] / \varepsilon ^ { 2 } )$ runs and $\widetilde { O } ( D ( 1 + f / \varepsilon ) )$ distinct pilots on every history. It returns a member of the class and an interval of width at most ε containing, with probability at least 1 − δ, both the output’s risk and the optimal risk. We also treat unknown ceilings and expected costs. Lower bounds for finite classes show that each order is necessary in the worst case, possibly on diferent instances, at positive dimension, small accuracy and fixed confidence. The learner enumerates label patterns of the class on displayed queries, possibly in exponential time; the results, under standard measurability conditions, concern statistical resources only. The proof combines a gain computable from stopped displays, exponential weights, a sparse depth-robust graph and a harmonic-mean bound.

## 1 Introduction

Many checking procedures go through a sequence of items and stop at the first item that is handled incorrectly. When a candidate decision rule is run on such a sequence, the run reveals the items and their correct labels up to and including the first item on which the rule is wrong; the later items, and any outcome attached to the end of the sequence, remain hidden. This paper studies how to learn a good rule when this is the only feedback. The learner must choose which rule to run before it sees the sequence, and an early mistake hides everything that follows. Even when every revealed label is correct, minimizing the empirical error on the prefixes observed so far need not produce a good rule, because the behaviour that matters for the loss of a candidate may lie beyond those prefixes, which were determined by the rules that happened to be run. Learning therefore requires deliberate exploration.

We formalize this as follows (Section 2 gives the precise model). A fresh run uses a record of at most D queries, drawn from one unknown distribution, independently of the rule that is run on it. The rule, called the pilot, is a member of a fixed binary class H of VC dimension at most d; it is selected before the record is drawn and does not change during the run. The run displays the queries and their true labels through the pilot’s first discrepancy, inclusive; if the pilot makes no discrepancy, it also displays a terminal descriptor of the record. A bounded loss, fixed before learning, is attached to the position at which a hypothesis stops: a hypothesis that errs at query $j$ pays a loss that depends on the first $j$ queries and labels, and a hypothesis that survives pays a loss that may depend on the whole record and its descriptor. We count the number N of runs, also called resets, and the number V of distinct members of H that are actually used as pilots during training. The second count measures exploration: every rule played while a cover is being acquired counts, not only the members of a final mixture.

Two optima. Two infima govern the problem:

$$
F ^ { * } = \operatorname* { i n f } _ { h \in \mathcal { H } } \operatorname* { P r } ( h \ \mathrm { e r r s ~ o n ~ t h e ~ r e c o r d } ) , \qquad R ^ { * } = \operatorname* { i n f } _ { h \in \mathcal { H } } \mathbb { E } [ \mathrm { l o s s ~ o f ~ } h \ \mathrm { w h e r e ~ i t ~ s t o p s } ] .
$$

Neither needs to be attained, and they may be approached by diferent hypotheses. The first controls how much of a record a good pilot can reveal; the second controls the variance of the final estimate. Given numbers $f , r$ with $F ^ { * } \leq f$ and $R ^ { * } \leq r ,$ our main result (Theorem 3.1) is a learner whose numbers of resets and distinct pilots are at most

$$
N = \widetilde { O } \left( \left( d + 1 \right) \frac { D \left( f + \varepsilon \right) + r } { \varepsilon ^ { 2 } } \right) , \qquad V = \widetilde { O } \left( D \left( 1 + \frac { f } { \varepsilon } \right) \right)\tag{1.1}
$$

on every history. Here $\widetilde O$ hides universal constants and polylogarithmic factors in $D , d + 1 , 1 / \varepsilon$ and $1 / \delta$ . On every history the learner returns a member $\widehat { h } \in \mathcal { H }$ and an interval of width at most $\varepsilon ,$ computed from the observations; with probability at least $1 - \delta$ the interval contains both $R ^ { * }$ and the risk $R ( \widehat { h } )$ of the output, so in particular $R ( \widehat { h } ) \leq R ^ { * } + \varepsilon$

When no ceilings are supplied, a second learner (Theorem 3.2) attains the same orders with $F ^ { * }$ and $R ^ { * }$ in place of $f$ and $r ,$ on the same event as the coverage guarantee; on every history it satisfies the coarser caps $\widetilde { O } ( ( d + 1 ) D / \varepsilon ^ { 2 } )$ and $\tilde { O } ( D / \varepsilon )$ , and with a smaller internal failure level the local orders also hold in expectation. For the natural discrepancy loss, which charges one unit for a discrepancy and nothing for survival, $R ^ { * } = F ^ { * }$ , so (1.1) holds with $f = r$ (Corollary 3.4); in particular the reset scale is $\widetilde { O } ( ( d + 1 ) D / \varepsilon )$ when the natural infimum is zero, even if it is not attained.

Both orders in (1.1) are necessary in the worst case (Theorem 3.5): for every $d , D \ge 1$ , accuracy $\varepsilon \le 1 / 1 6$ and success probability $4 / 5$ , there are finite classes of VC dimension exactly $d ,$ record laws and fixed losses satisfying the two ceilings on which every successful learner has expected reset count $\Omega ( d [ D ( f + \varepsilon ) + r ] / \varepsilon ^ { 2 } )$ in some instance, and others on which it has expected distinct-pilot count $\Omega ( D ( 1 + f / \varepsilon ) )$ in some instance. These are coordinatewise ceiling-minimax statements at fixed confidence, not a trade-of between N and V and not a lower bound for every class. Natural discrepancy has a separate statement under its single ceiling $F ^ { * } = R ^ { * } \leq r$ (Theorem 3.6), with an exception at $D = 1$ , where every query and its label are displayed whatever pilot is used and one fixed training pilot sufices.

## 1.1 How the learner works

The construction has six steps; the arguments specific to stopped feedback are in Steps 2–6.

1. A coarse anchor. The learner first finds a member a of $\mathcal { H }$ and a number $w = O ( f + \varepsilon )$ such that, if $f$ is a valid ceiling, $F ( a ) \leq w$ on a success event. It uses an ownership argument: the loss of a candidate on a record is assigned to the first pilot, in order of use, that reaches the candidate’s stopping position. When the numerical inputs of an ownership call are valid, a bounded potential limits the number of failed validations on a concentration event; on every history a cap on the number of candidates bounds the cost (Section 4, Appendix B).

2. A legal gain at truncated ranks. Fix a frozen mixture L of already available members, with weight at least 1/2 on a; we call it a logger. We define, for every candidate h, a score that can be computed from one display of a pilot drawn from L. Its conditional mean is a potential evaluated at the candidate’s stopping rank, truncated one position after the last rank that L reaches with probability at least a threshold ζ. The key point is that a displayed pair at position j is used to estimate an increment of the potential at the virtual position j + 1. This is legal because the discrepancy pair itself is displayed (Lemma 5.1).

3. An observed dictionary. The class may be infinite. The learner plays L on fresh resets and calls the pattern oracle on the queries that were actually displayed. A VC missed-set argument shows that every member of H has a representative in the resulting finite dictionary whose truncated rank difers from its own with small probability.

4. Exponential weights and one sparsification. Classical exponential weights run numerically on the dictionary, while fresh data still come from the frozen logger L. A single random sparsification turns the averaged weights into a small lottery p, and a single transfer from the dictionary to the whole class yields a bound on the potential of p at the truncated rank of every member of H (Lemma 5.2).

5. Propagation along a robust graph. The four steps above learn a lottery that is good only up to the logger’s visibility frontier. To reach the true stopping ranks without paying a polynomial factor in D, we run S = 4D acquisition stages arranged along a sparse depth-robust directed graph of polylogarithmic indegree. The logger of a stage mixes the lotteries of its graph predecessors. A successful predecessor moves the frontier one position further. Deleting any set of unsuccessful stages leaves a long path, so the argument allows the failures of diferent stages to be arbitrarily correlated. On most records, a constant fraction of the stages see the true stopping rank of a given candidate (Lemma 6.3).

6. Harmonic recovery and a final estimate. The final lottery puts weight 1/2 on a and spreads the rest uniformly over the S stages. A naive bound would lose the factor S. The harmonic– arithmetic mean inequality over the visible stages recovers it (Lemma 6.4). Inverse-propensity estimates on fresh runs of this frozen lottery, a uniform relative deviation bound, proper empirical minimization over the patterns on the displayed queries, and explicit clipping then give the output and its interval (Proposition 6.7).

The depth-robust graph, the exponential-weights optimizer, inverse-propensity estimation, the rank-tail inequality and the VC arguments are established tools, and we do not claim them as new. What the proof establishes is that they can be combined using only the observations available in this experiment, while counting every reset and every played function. Complete latent records appear only in the proofs; the learner sees a whole record only when its pilot survives, and has no other source of complete records.

## 1.2 Relation to the closest work

Two earlier results are especially close. Li and Zhang [18] give a fast trajectory-wise learner with first-step mixtures for a supplied finite evaluable class in the attained-realizable setting; an absorbingstate reduction turns it into a legal stopped-feedback baseline with a single proper output. Balcan, Blum, Li and Sharma [6] already give proper agnostic learning of verifiers with coeficient one, and its extension to ordinary VC classes, when complete annotated traces are supplied. Neither properness, agnostic competition nor finite VC dimension is therefore the distinguishing feature here. What we add is the acquisition of the missing trace information when the pilot’s own earlier discrepancy censors the record, including its terminal information, under original-proper frozen pilots and with both training resources controlled. Section 8 compares the exact interfaces, including an earlier proper policy search with resets under a mistake-bound assumption [19] and a legal empirical specialization of the planning engine of Amortila, Foster and Krishnamurthy [4].

## 1.3 Scope and organization

The results concern statistical resources under a strict observation interface: one common record law that does not depend on the pilot, exact and fixed labels, a pilot frozen for the whole run with no prefix resets and no continuations after an error, and an oracle that returns representatives of the patterns of H on displayed queries and may need exponential time. Sections 2 and 3 give the model and the results, Sections 4 to 6 prove the main upper bound, Section 7 explains the lower-bound constructions, Section 8 discusses related work and Section 9 the assumptions and constants; the appendices contain the remaining proofs.

## 2 The experiment

## 2.1 Records, pilots and displays

Definition 2.1 (Records). Let Z be a measurable query space and $y : \mathcal { Z } \to \{ 0 , 1 \}$ a measurable truth function. Fix a known integer $D \geq 1$ . A record is

$$
T = { \bigl ( } ( z _ { 1 } , y _ { 1 } ) , \dotsc , ( z _ { M } , y _ { M } ) , A { \bigr ) } , \qquad 0 \leq M \leq D ,
$$

where $z _ { j } \in \mathcal { Z } , y _ { j } = y ( z _ { j } )$ , and A is a terminal descriptor taking values in a measurable space. A world consists of the truth function y and a probability distribution P on records with $y _ { j } = y ( z _ { j } )$ almost surely. Both are unknown to the learner.

Queries in a record may repeat, may depend on the earlier labels, and the length M and the descriptor A may depend arbitrarily on the queries. The truth is exact: the label of a query is a fixed function of the query. The descriptor carries the information that is available only at the end of a record, for instance the outcome of a completed task.

Let H be a nonempty class of functions ${ \mathcal { Z } }  \{ 0 , 1 \}$ whose VC dimension is at most a known integer $d \geq 0$ . For a function $g : { \mathcal { Z } }  \{ 0 , 1 \}$ and a record T define the stopping rank

$$
\tau _ { g } ( T ) = \operatorname* { m i n } \{ j \leq M : \ g ( z _ { j } ) \neq y _ { j } \} ,
$$

with $\tau _ { g } ( T ) = M + 1$ if the set is empty. We say that g errs at $\tau _ { g } \operatorname { i f } \tau _ { g } \leq M$ , and that g survives if $\tau _ { g } = M + 1$

Definition 2.2 (Display). The display of g on T consists of the pairs $( z _ { j } , y _ { j } )$ for $j \le \operatorname* { m i n } ( \tau _ { g } ( T ) , M )$ together with an end tag and the descriptor A if and only if $\tau _ { g } ( T ) = M + 1$ . We write $n _ { g } = \operatorname* { m i n } ( \tau _ { g } , M )$ for the number of displayed pairs.

Two cases deserve attention. A discrepancy at the last query, $\tau _ { g } = M$ , displays the pair $( z _ { M } , y _ { M } )$ but neither the end tag nor A; in general the display does not reveal that the record has ended (it does when $M = D ,$ , since D is known). On an empty record $( M = 0 )$ every function survives with $\tau _ { g } = 1$ , and the display consists of the end tag and A.

Definition 2.3 (Interaction). A learner proceeds in resets $t = 1 , 2 , \ldots$ . Before reset t it selects a pilot $g _ { t } \in \mathcal { H }$ as a measurable function of the earlier displays and of its private randomness. (The selection may consist in sampling from a finite distribution on $\mathcal { H } ;$ the sampled function is fixed before the record is drawn.) Then a fresh record $T _ { t } \sim \mathcal { P }$ is drawn, independently of everything before, and the learner observes the display of $g _ { t }$ on T<sub>t</sub>. After any reset the learner may stop; N denotes the number of resets. At the end it returns $\dot { h } \in \mathcal { H }$ and an interval, both computed from its observations and private randomness.

No other access is available. In particular there is no access to the hidden sufix of a record, no continuation after a discrepancy, no reset to an arbitrary prefix, no change of the pilot within a run, no oracle for ${ \mathcal P } ,$ and no deployment oracle. The record law does not depend on the pilot or on the private coins. This common law is what allows the analysis to evaluate the stopping ranks of several hypotheses on the same fresh reference record; such a reference record is an object of the analysis, not a source of observations.

## 2.2 Losses and risks

Definition 2.4 (Prefix-observable loss). A loss is a known measurable function λ assigning to a record $T$ and a rank $j \in \{ 1 , \dots , M + 1 \}$ a value $\lambda ( T , j ) \in [ 0 , 1 ]$ , such that for $j \le M$ the value $\lambda ( T , j )$ is a function of the first j pairs $( z _ { 1 } , y _ { 1 } ) , \dotsc , ( z _ { j } , y _ { j } )$ , while $\lambda ( T , M + 1 )$ may depend on the whole record including A. The loss is fixed before learning.

For $h \in \mathcal H$ write

$$
\begin{array} { r l } & { \ell _ { h } ( T ) = \lambda ( T , \tau _ { h } ( T ) ) , \qquad R ( h ) = \operatorname { \mathbb { E } } \ell _ { h } ( T ) , \qquad F ( h ) = \operatorname* { P r } ( \tau _ { h } ( T ) \leq M ) , } \\ & { \qquad R ^ { * } = \displaystyle \operatorname* { i n f } _ { h \in \mathcal { H } } R ( h ) , \qquad F ^ { * } = \displaystyle \operatorname* { i n f } _ { h \in \mathcal { H } } F ( h ) . } \end{array}
$$

The dependence on the candidate enters only through its stopping rank. The natural discrepancy loss is $\lambda _ { \mathrm { n a t } } ( T , j ) = \mathbf { 1 } \{ j \leq M \}$ ; for it $R = F$ . The learner may use the natural loss as an auxiliary objective, for instance to find an anchor, even when the target loss λ is diferent.

The following observation is used many times. Let g be a pilot and h any function whose values at the displayed queries are known.

• If h disagrees with some displayed pair, then its first such disagreement is $\tau _ { h } \leq n _ { g }$ , and $\ell _ { h } ( T ) = \lambda ( T , \tau _ { h } )$ is determined by the displayed pairs.

• If $g$ survives, the whole record is displayed, so $\tau _ { h }$ and $\ell _ { h } ( T )$ are determined.

• If g errs and h agrees with every displayed pair, then h is correct at $\tau _ { g } .$ , so $\tau _ { h } > \tau _ { g } ;$ in general $\ell _ { h } ( T )$ is not determined.

## 2.3 The pattern oracle and regularity

Assumption 2.5 (Observed-pattern oracle). For every finite list $\left( z _ { 1 } , \ldots , z _ { q } \right)$ of queries that have actually been displayed, the learner can obtain a finite set $\mathcal { G } \subseteq \mathcal { H }$ containing, for every $h \in \mathcal H$ exactly one $h ^ { \prime } \in \mathcal { G }$ with $h ^ { \prime } ( z _ { i } ) = h ( z _ { i } )$ for $1 \leq i \leq q$ . For $q = 0$ the set is a single member of H. The returned members are evaluable at every query, and their selection may depend on the list. A call draws no record and plays no function.

The oracle is stronger than an empirical risk minimizer: it lists one representative for each label pattern that H realizes on the displayed queries. It supplies nothing on queries that have not been displayed, and it may require exponential time.

Assumption 2.6 (Measurability). The evaluation maps $( h , z ) \mapsto h ( z )$ of the functions used, including the evaluation of functions selected from data at later queries, the loss, the oracle’s selections and finite tie-breaks, and the uniform events and witness selections in all VC arguments of this paper (Appendices A to C, E and G) are measurable, so that all events used have ordinary probabilities.

Assumption 2.6 is the usual measurable-VC convention. It does not follow from finite VC dimension and pointwise measurability alone.

## 2.4 Resources and the required output

Definition 2.7 (Resources). N is the total number of resets, over all phases of the learner: anchor acquisition, validation, design acquisition and final estimation. V is the number of distinct functions in $\{ g _ { 1 } , \dotsc , g _ { N } \}$ , that is, the size of the union of all functions actually played as pilots during training.

Computing with a finite distribution over H, enumerating a dictionary and privately drawing the identity of a function play nothing by themselves. A function counts towards V when it is first actually used as a pilot, whatever other roles it has; it may, for instance, also belong to a dictionary. The returned $\widehat { h }$ need not have been played. The number of functions from which a procedure can draw its pilots, for instance the anchor together with all slots of all designs in a precision stage, bounds the number of distinct functions it plays; the support of the final lottery alone need not.

Definition 2.8 (Joint certificate). For $\varepsilon \in ( 0 , 1 ]$ , an ε-certificate is a pair $( \widehat { h } , \widehat { I } )$ where $\widehat { h } \in \mathcal { H }$ and $\widehat { I } = [ \widehat { l } , \widehat { u } ]$ with $0 \leq \widehat { l } \leq \widehat { u } \leq 1$ and $\widehat { u } - \widehat { l } \le \varepsilon$ , both computed from the observations. It covers if $R ^ { * } \in \widehat { I }$ and $R ( \widehat { h } ) \in \widehat { I }$

A covering ε-certificate gives $R ( \widehat { h } ) \leq R ^ { * } + \varepsilon$ , and it also locates $R ^ { * }$ within ε. Our learners return an ε-certificate on every history; only the covering property is probabilistic.

Example 2.9 (A censored terminal). Let every record consist of a gate query, a second query and a descriptor. A pilot that errs at the gate displays the gate and its label and nothing else. A pilot that errs at the second query displays both pairs, but not the descriptor. A candidate that survives both queries can have a loss that depends on the descriptor. Thus, already at horizon two, knowing every observed label does not give an unbiased sample of the loss of every candidate.

Conventions. A lottery is a finitely supported probability distribution on $\mathcal { H } ,$ , and $\delta _ { h }$ is the point mass at h. Logarithms are natural. A statement holds on every history if it holds for every realization of the records and of the private randomness. We write ${ \widetilde { O } } ( \cdot )$ for an upper bound up to a universal constant and a factor that is polynomial in $\log ( e D )$ ， $\log ( e ( d + 1 ) ) , \log ( e / \varepsilon )$ and $\log ( e / \delta )$ it never hides a polynomial factor in d or D. Every statement concerns the supplied horizon $D ;$ we do not assert one confidence event that holds simultaneously for all possible horizons.

## 3 Results

## 3.1 Upper bounds

Theorem 3.1 (Joint guarantees with supplied ceilings). Let $0 < \varepsilon \le 1 , 0 < \delta < 1$ and $f , r \in [ 0 , 1 ]$ There is a learner, receiving d, $D , \varepsilon , \delta , f , r$ , with the following properties.

(a) On every history it returns an ε-certificate $( \widehat { h } , \widehat { I } )$ and uses

$$
N = \widetilde { O } \bigg ( ( d + 1 ) \frac { D ( f + \varepsilon ) + r } { \varepsilon ^ { 2 } } \bigg ) r e s e t s a n d V = \widetilde { O } \bigg ( D \bigg ( 1 + \frac { f } { \varepsilon } \bigg ) \bigg ) d i s t i n c t p i l o t s .
$$

(b) If $F ^ { * } \leq f$ and $R ^ { * } \leq r ,$ then with probability at least $1 - \delta$ the certificate covers: $R ^ { * } \in \widehat { I }$ and $R ( \widehat { h } ) \in \widehat { I }$ . In particular $R ( \widehat { h } ) \leq R ^ { * } + \varepsilon$

The counts include every reset and every function played in all phases, including anchor acquisition, exploration, all intermediate loggers and final estimation.

The bounds in (a) hold on every history: the sample sizes of the final precision stage are fixed numerical functions of $( d , D , \varepsilon , \delta , f , r )$ , and those of the anchor, which depend on observed endpoints, are bounded by such functions. The proof does not use that f and r are valid ceilings; validity is used only for the covering property (b).

Theorem 3.2 (Unknown ceilings and expected costs). Let $0 < \varepsilon \le 1$ and $0 < \delta < 1$ . There is a learner, receiving d, $D , \varepsilon , \delta$ , with the following properties.

(a) On every history it returns an ε-certificate and uses at most $\widetilde { O } ( ( d + 1 ) D / \varepsilon ^ { 2 } )$ resets and $\tilde { O } ( D / \varepsilon )$ distinct pilots.

(b) With probability at least $1 - \delta$ , the certificate covers and, on the same event,

$$
N = \widetilde O \bigg ( ( d + 1 ) \frac { D \big ( F ^ { * } + \varepsilon \big ) + R ^ { * } } { \varepsilon ^ { 2 } } \bigg ) , \qquad V = \widetilde O \bigg ( D \bigg ( 1 + \frac { F ^ { * } } { \varepsilon } \bigg ) \bigg ) .
$$

(c) If the learner is run with internal failure level $\delta ^ { \prime } = \operatorname* { m i n } ( \delta , \varepsilon ^ { 2 } )$ , with all logarithms recalculated at that level, then $( a )$ and (b) hold and in addition EN and EV obey the two local orders displayed in (b).

The bounds in (b) hold on the success event, not on every history. Part (c) is a separate expectation argument based on the caps in (a); it is not a relabelling of (b).

Proposition 3.3 (Auxiliary accounting). The learners of Theorems 3.1 and 3.2 have the following properties on every history.

(i) At most DN true labels are displayed.

(ii) The number of oracle calls is $\tilde { O } ( D ( 1 + \log ( 1 / \varepsilon ) ) )$ , including the initial call that produces a first member of H.

(iii) Within one precision stage (Procedure 2), the $S = 4 D$ design dictionaries contain at most $\textstyle \sum _ { s = 1 } ^ { S } ( D m _ { s } + 1 ) ^ { d }$ representatives in total, where $m _ { s }$ is the dictionary sample size of design s. The final selection uses a separate enumeration with at most $( D n _ { F } + 1 ) ^ { d }$ representatives, where n is the final estimation sample size. These numbers are not a bound on the total memory including the anchor work.

(iv) Within one precision stage, every pilot is the anchor or a sparse slot of one of its designs. Enumerating a dictionary and computing numerical weights play nothing by themselves; a dictionary member can still be played, for instance when it coincides with the anchor or with a slot, and every actual use is counted in V.

(v) $I f \varepsilon = 1$ , the learners return the representative obtained from one empty-list oracle call together with $\widehat { I } = [ 0 , 1 ]$ , using zero resets and zero training pilots.

No bound on computation time or on the encoding of real numbers is asserted.

Corollary 3.4 (Natural discrepancy and unit-error losses). Suppose the loss satisfies $F ( h ) \leq R ( h )$ for every $h \in \mathcal H$ . This holds for the natural discrepancy loss, and for every loss that charges one unit at each discrepancy. If $R ^ { * } \leq r$ , the learner of Theorem 3.1 with $f = r$ uses on every history

$$
N = \widetilde O \left( ( d + 1 ) \frac { D ( r + \varepsilon ) } { \varepsilon ^ { 2 } } \right) , \qquad V = \widetilde O \left( D \left( 1 + \frac { r } { \varepsilon } \right) \right) ,
$$

and covers with probability at least $1 - \delta$ . The learner of Theorem 3.2 satisfies these bounds with r replaced by $R ^ { * }$ , in the sense of parts (b) and (c) of that theorem.

Proof. Taking infima in $F ( h ) \ \leq \ R ( h )$ gives $F ^ { * } \leq R ^ { * } \leq r .$ , so $f = r$ is a valid ceiling for $F ^ { * }$ ; no common minimizer of F and R is needed. Substituting $\boldsymbol { f } \ : = \ : \boldsymbol { r }$ in Theorem 3.1 and using $D ( r + \varepsilon ) + r \le 2 D ( r + \varepsilon )$ gives the bounds. For Theorem 3.2, use $F ^ { * } \leq R ^ { * }$ in the same way.

## 3.2 Lower bounds

In the lower bounds a learner is a procedure as in Definition 2.3; it may depend on the known class and loss, and the pilots and the output must belong to the class. An admissible instance for parameters $( d , D , f , r )$ consists of a finite class of VC dimension exactly $d ,$ a world whose records have length at most D, and a fixed known prefix-observable loss, such that $F ^ { * } \leq f$ and $R ^ { * } \leq r$ The constructions below come in finite families of instances that share a class and a loss rule and difer in the world.

Theorem 3.5 (Two-ceiling lower bounds). Let d, $D \geq 1$ be integers, $0 < \varepsilon \le 1 / 1 6$ and $f , r \in [ 0 , 1 ]$ Let a learner return, on every admissible instance, a member $\widehat { h }$ of the class with $\operatorname* { P r } \{ R ( \widehat { h } ) \leq R ^ { * } + \varepsilon \} \geq$ $4 / 5$ . Then

$$
\operatorname* { s u p } \mathbb { E } N \geq c d \frac { D ( f + \varepsilon ) + r } { \varepsilon ^ { 2 } } , \qquad \operatorname* { s u p } \mathbb { E } V \geq c D \left( 1 + \frac { f } { \varepsilon } \right) ,
$$

where $c > 0$ is a universal constant and each supremum is over admissible instances. For the second inequality we assume, as usual, that the learner terminates almost surely on each instance of the finite families used in the proof; every learner with deterministic resource caps does. The suprema may be attained on diferent families for diferent summands and for the two resources. Each family uses one loss rule, known and common to all its worlds.

Theorem 3.6 (Natural discrepancy under its single ceiling). Let d, $D \geq 1$ be integers, $0 < \varepsilon \le 1 _ { \scriptscriptstyle { \mathscr { S } } }$ /128 and $r \in [ 0 , 1 ]$ . Consider the natural discrepancy loss and finite classes of VC dimension exactly d, with the single ceiling $F ^ { * } = R ^ { * } \leq r$ . Let a learner satisfy $\operatorname* { P r } \{ F ( { \widehat { h } } ) \leq F ^ { * } + \varepsilon \} \geq 4 / 5$ on every such instance with records of length at most D. Then:

(a) sup $\mathbb { E } N \geq c d D ( r + \varepsilon ) / \varepsilon ^ { 2 } ,$

(b) if $D \geq 2 .$ , then sup $\mathbb { E } V \geq c D ( 1 + r / \varepsilon )$ , under the almost-sure termination convention of Theorem 3.5;

(c) if $D = 1$ , then for every class satisfying Assumptions 2.5 and 2.6 (in particular for the finite classes above), every $\varepsilon \in ( 0 , 1 ]$ and every $\delta \in ( 0 , 1 )$ , there is a learner for the natural discrepancy loss that plays one fixed member of the class on all of its resets and returns an ε-certificate that covers with probability at least $1 - \delta$

Here $c > 0$ is a universal constant, and the suprema may again be attained on diferent families.

For the natural loss, two simultaneous ceilings $F ^ { * } \leq f$ and $R ^ { * } \leq r$ amount to the single ceiling min $( f , r )$ ; Theorem 3.6 must be applied with that ceiling, not with r. In particular, if $f = 0 < r .$ , the natural discrepancy infimum is zero (in an infinite class it need not be attained), and Theorem 3.1 applied with both ceilings equal to min $( f , r ) = 0$ uses $\widetilde { O } ( ( d + 1 ) D / \varepsilon )$ resets; no lower bound of order $r / \varepsilon ^ { 2 }$ holds there.

Remark 3.7 (How to read the lower bounds). The lower bounds use only the excess-risk requirement. An ε-certificate that covers implies it, so they apply to the certificates of Theorems 3.1 and 3.2. Since deterministic caps bound expectations, Theorems 3.1 and 3.5 show that the learner of Theorem 3.1 is coordinatewise ceiling-minimax optimal up to logarithmic factors, at fixed confidence and in the stated range of $\varepsilon ,$ at positive dimension; part (c) of Theorem 3.2 gives the same comparison for expected costs without supplied ceilings. The lower bounds assert neither a trade-of between N and $V$ on a common instance, nor a lower bound for every class or every loss, nor sharp dependence on $\delta ,$ nor anything for $d = 0$ . A nonempty class of VC dimension zero consists of a single function (two functions that difer at some query shatter that query), so excess-risk learning needs no data, and a certificate only requires estimating one bounded mean using at most one pilot; the upper bounds remain valid for $d = 0$ through the factor $d + 1$ , but are not sharp there. At $D = 1$ the general-loss pilot lower bound $\Omega ( 1 + f / \varepsilon )$ remains valid, because a terminal descriptor is still hidden after a discrepancy, whereas for natural discrepancy at $D = 1$ one training pilot sufices.

Appendix F proves Theorem 3.2 and Proposition 3.3; Section 6 proves Theorem 3.1; Appendix G proves Theorems 3.5 and 3.6.

## 4 Propensities, potentials and the anchor

This section collects the objects used by the learner and the elementary facts about them. Throughout, $T$ denotes a record with length $M$

## 4.1 Propensities

Definition 4.1 (Propensity). For a lottery π and a rank $l \in \{ 1 , \ldots , M + 1 \}$ let

$$
Q _ { \pi } ( T , l ) = \pi \big ( \{ g \in \mathcal { H } : \tau _ { g } ( T ) \geq l \} \big ) .
$$

Lemma 4.2 (Properties of propensities). For every lottery $\pi$ and record $T .$

(i) $Q _ { \pi } ( T , 1 ) = 1$ , and $l \mapsto Q _ { \pi } ( T , l )$ is nonincreasing on $\{ 1 , \ldots , M + 1 \}$ ;

(ii) $Q _ { \pi } ( T , l )$ depends on T only through the pairs $( z _ { j } , y _ { j } )$ with $j < l$ , so it can be computed from every display that contains these pairs;

(iii) $\pi \mapsto Q _ { \pi } ( T , l )$ is linear under mixtures of lotteries;

(iv) if a pilot g is drawn from π independently of T, then $\operatorname* { P r } ( \tau _ { g } \geq l \mid T ) = Q _ { \pi } ( T , l )$ , and for $l \leq M$ this is the conditional probability that the pair at position l is displayed.

Proof. We have $\tau _ { g } ( T ) \geq l$ if and only if $g ( z _ { j } ) = y _ { j }$ for all $j < l$ . This gives (i) and (ii). Part (iii) holds because $Q _ { \pi } ( T , l )$ is the π-measure of a fixed set. For $( \mathrm { i v } )$ , independence gives $\operatorname* { P r } ( \tau _ { g } \geq l \mid T ) =$ $\pi ( \{ g : \tau _ { g } ( T ) \geq l \} )$ , and for $l \leq M$ the pair at position l is displayed exactly when $l \le \operatorname* { m i n } ( \tau _ { g } , M )$ 2 that is, when $\tau _ { g } \geq l$ □

The algorithm uses lotteries in two ways. Some are used to draw pilots. Others are purely numerical: they enter only through propensities in computations and are never used to draw a pilot, although their members may be played in other roles.

## 4.2 The potential

Fix an anchor $a \in \mathcal H$ and a regularizer $b \in ( 0 , 1 / 2 ]$ , and write $\tau _ { a } = \tau _ { a } ( T )$ . For a lottery $\pi$ define

$$
\Phi _ { \pi } ( T , l ) = { \frac { { \bf 1 } \{ l > \tau _ { a } \} } { Q _ { \pi } ( T , l ) + b } } \qquad ( 1 \leq l \leq M + 1 ) , \qquad \Lambda _ { b } = \log { \frac { 1 + b } { b } } .
$$

Since $\tau _ { a } \geq 1$ , the potential vanishes for $l \leq \tau _ { a } .$ , in particular at $l = 1$ . For $\tau _ { a } < l \leq M + 1$ it is nondecreasing in l by Lemma 4.2(i). It takes values in $[ 0 , 1 / b ]$ . If the anchor survives, $\tau _ { a } = M + 1$ then $\Phi _ { \pi } ( T , l ) = 0$ for all $l \leq M + 1$ . Finally, for fixed T and l, the map $\pi \mapsto \Phi _ { \pi } ( T , l )$ is convex, because $x \mapsto 1 / ( x + b )$ is convex and $\pi \mapsto Q _ { \pi } ( T , l )$ is linear.

## 4.3 Loggers, frontiers and truncated ranks

Definition 4.3 (Logger, frontier, truncated rank). A logger is a lottery L with $L ( a ) \ge 1 / 2$ that is fixed before the resets on which it is used. Fix a threshold $\zeta \in ( 0 , 1 / 2 ]$ . The frontier of L on $T$ and the truncated rank of $h \in \mathcal H$ are

$$
J _ { L } ( T ) = \operatorname* { m a x } \{ l \in \{ 1 , \ldots , M + 1 \} : \ Q _ { L } ( T , l ) \geq \ \zeta \} , \qquad \rho _ { L } ( h ) = \rho _ { L } ( h , T ) = \operatorname* { m i n } ( \tau _ { h } ( T ) , J _ { L } ( T ) + 1 ) .\tag{4.1}
$$

By Lemma $4 . 2 ( \mathrm { i } ) , \{ l : Q _ { L } ( T , l ) \geq \zeta \} = \{ 1 , \ldots , J _ { L } \}$ . Since the anchor survives up to its own stopping rank, $Q _ { L } ( T , \tau _ { a } ) \geq L ( a ) \geq 1 / 2 \geq \zeta$ , hence $J _ { L } \ge \tau _ { a }$ . Also $\rho _ { L } ( h ) \leq \tau _ { h } \leq M + 1$ . The frontier and the truncated rank are used only in the analysis; the algorithm never needs their values, only the local tests $Q _ { L } ( T , j ) \ge \zeta$ at displayed positions $j$

Lemma 4.4 (Rank tail). For every record $T _ { i }$ , every lottery π and every logger $L _ { i }$

$$
\sum _ { h } \pi ( h ) \Phi _ { \pi } ( T , \rho _ { L } ( h ) ) \ \leq \ 1 \{ \tau _ { a } \leq M  \} \Lambda _ { b } ,
$$

where the sum runs over the support of π.

Proof. If $\tau _ { a } = M + 1$ , every potential vanishes. Assume $\tau _ { a } \leq M$ and put $\bar { l } = \operatorname* { m i n } ( J _ { L } { + } 1 , M { + } 1 )$ . Under $h \sim \pi$ the truncated rank takes values in $\{ 1 , \ldots , \bar { l } \}$ . Let $\nu _ { l } = \pi ( \rho _ { L } ( h ) = l )$ and $q _ { l } = \pi ( \rho _ { L } ( h ) \geq l )$ for $1 \leq l \leq \bar { l } .$ and $q _ { \bar { l } + 1 } = 0$ . For $l \leq \bar { l }$ we have $l \le J _ { L } + 1$ , so $\rho _ { L } ( h ) \geq l$ holds exactly when $\tau _ { h } \geq l ;$ hence $q _ { l } = Q _ { \pi } ( T , l )$ . (The convention $q _ { \bar { l } + 1 } = 0$ concerns the truncated rank only; it says nothing about $Q _ { \pi } ( T , \bar { l } + 1 ) . \rangle$ ) Dropping the indicator in Φ and using $\nu _ { l } = q _ { l } - q _ { l + 1 }$ 2

$$
\sum _ { h } \pi ( h ) \Phi _ { \pi } ( T , \rho _ { L } ( h ) ) \leq \sum _ { l = 1 } ^ { \bar { l } } \frac { \nu _ { l } } { q _ { l } + b } \leq \sum _ { l = 1 } ^ { \bar { l } } \int _ { q _ { l + 1 } } ^ { q _ { l } } \frac { d x } { x + b } \leq \int _ { 0 } ^ { 1 } \frac { d x } { x + b } = \Lambda _ { b } .
$$

The second inequality holds because $1 / ( x + b ) \geq 1 / ( q _ { l } + b ) \mathrm { ~ o n ~ } [ q _ { l + 1 } , q _ { l } ]$ , and the third because these intervals have disjoint interiors and lie in $[ 0 , 1 ] = [ 0 , q _ { 1 } ]$ □

The rank-tail inequality is elementary and is related to rank-cover arguments in the literature; we use it as a tool and do not claim it as new.

## 4.4 The anchor

The first phase of the learner finds, on a success event, an anchor with small natural discrepancy probability. The anchor makes most records easy to observe, independently of the target loss. It is obtained by an ownership procedure, proved in full in Appendix B. On one record, the loss of a candidate is assigned to the first pilot, in the order of use, whose run reaches the candidate’s stopping rank. These assignments are computable from displays, and their predictable means add up to at most the candidate’s risk. Consider a call whose numerical input r is a valid risk ceiling, $R ^ { * } \leq r$ . On the joint concentration event of Appendix B, a candidate whose validation fails then stops later than all earlier pilots with probability more than a fixed amount, so it increases the potential E max $z _ { g } \tau _ { g }$ by that amount. This potential exceeds its value for the initial pilot a of the procedure by at most $D F ( a ) ;$ so if moreover $F ( a ) \leq w _ { : }$ , not all allowed candidates fail and a candidate is accepted. On every history the number of candidates is capped, and if all allowed tests are exhausted without acceptance, the procedure returns its initial pilot with the interval [0, 0].

Lemma 4.5 (Anchor with a supplied ceiling; Appendix B). Let $0 < \varepsilon \leq 1 , 0 < \alpha < 1 , f \in [ 0 , 1 ]$ and $\bar { \varepsilon } = \operatorname* { m i n } ( 1 , f + \varepsilon )$ . There is a legal procedure that returns a member $a \in { \mathcal { H } }$ and the number

$$
w = \operatorname* { m a x } \big ( \varepsilon , \operatorname* { m i n } ( 1 , f + \bar { \varepsilon } ) \big ) , \qquad w h i c h \ s a t i s f i e s \ \qquad \varepsilon \leq w \leq 1 , \quad w \leq 2 ( f + \varepsilon ) .
$$

If $F ^ { * } \leq f$ , then $\operatorname* { P r } \{ F ( a ) \leq w \} \geq 1 - \alpha$ . On every history the procedure uses $\tilde { O } ( ( d + 1 ) D / \bar { \varepsilon } ) =$ $\tilde { O } ( ( d + 1 ) D / ( f + \varepsilon ) )$ resets and ${ \cal \tilde { O } } ( D )$ distinct pilots and oracle calls, where the hidden factors are polylogarithmic in $d + 1 , D , 1 / \varepsilon$ and 1/α. $I f \bar { \varepsilon } = 1$ it uses no reset: it returns an empty-list representative and $w = 1$

Lemma 4.6 (Anchor without a supplied ceiling; Appendix B). Let $0 < \varepsilon \le 1$ and $0 < \alpha < 1$ . There is a legal procedure that returns a member $a \in \mathcal H$ and an observable number $w _ { 0 } \in [ \varepsilon , 1 ]$ with

$$
\operatorname* { P r } \{ F ( a ) \leq w _ { 0 } \leq 2 ( F ^ { * } + \varepsilon ) \} \geq 1 - \alpha .
$$

On every history it uses $\widetilde { O } ( ( d + 1 ) D / \varepsilon )$ resets and ${ \widetilde { O } } ( D )$ distinct pilots and oracle calls. If its last scale is $t _ { c } ,$ it uses $\widetilde { O } ( ( d + 1 ) D / t _ { c } )$ resets, and on the success event $t _ { c } \ge ( F ^ { * } + \varepsilon ) / 5$ , so that the reset count is $\tilde { O } ( ( d + 1 ) D / ( F ^ { * } + \varepsilon ) )$ .

When an ownership schedule is executed, its initial pilot and every tested candidate, including an accepted output, are played and counted. The bypasses for $\bar { \varepsilon } = 1$ in Lemma 4.5 and for $\varepsilon = 1$ in Lemma 4.6 make no reset and return an unplayed representative, which is counted only when it is first used as a pilot (Remark B.2).

## 5 A legal gain and one acquisition stage

Fix an anchor a, a logger L, thresholds $\zeta , b \in ( 0 , 1 / 2 ]$ , and a lottery π, all determined before a fresh reset. The lottery π is numerical: it enters only through the potential $\Phi _ { \pi }$ and is never used to draw a pilot. The pilot of the reset is drawn from L.

## 5.1 The displayed gain

Lemma 5.1 (Displayed gain). Draw $g \sim L$ before the record T, and for $h \in \mathcal H$ let

$$
\widehat { G } ( h ) = \sum _ { j = 1 } ^ { n _ { g } } \mathbf { 1 } \{ Q _ { L } ( T , j ) \geq \zeta , ~ h ( z _ { i } ) = y _ { i } ~ f o r ~ a l l ~ i \leq j \} ~ \frac { \Phi _ { \pi } ( T , j + 1 ) - \Phi _ { \pi } ( T , j ) } { Q _ { L } ( T , j ) } ,\tag{5.1}
$$

Table 1: Principal notation of Sections 4 to 6.
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $\tau _ { h } , n _ { g }$   $\ell _ { h } , R , F$   $Q _ { \pi } ( T , l )$   $a , w$   $b , \Phi _ { \pi } , \Lambda _ { b }$   $L , \zeta$ </td><td>stopping rank of  $h ;$  number min  $( \tau _ { g } , M )$  of pairs displayed by pilot g loss of h at its stopping rank; risk; natural discrepancy probability propensity: π-mass of functions that are correct at positions  $1 , \ldots , l - 1$  anchor and the numerical bound  $F ( a ) \leq w$  used by the learner regularizer; potential  $\mathbf { 1 } \{ l > \tau _ { a } \} / ( Q _ { \pi } ( T , l ) + b ) ; \log ( ( 1 + b ) / b )$  logger (a frozen lottery with  $L ( a ) \geq 1 / 2 ) ;$  visibility threshold frontier of  $L ;$  truncated rank min  $( \tau _ { h } , J _ { L } + 1 )$  (analysis only) displayed gain of  $h ;$  observable disagreement event of  $h , h ^ { \prime }$  observed dictionary; sparse lottery returned by an acquisition stage  $( 2 + ( e - 1 ) e ) / ( 1 - e ^ { - 1 } ) ; 2 C _ { \mathrm { E G } } \Lambda _ { b } + 2$  number 4D of stages; predecessor set of stage  $s ;$  deterministic indegree cap  $\textstyle { \frac { 1 } { 2 } } { \bar { \delta } } _ { a } + { \frac { 1 } { 2 S } } \sum _ { s } p _ { s }$ </td></tr></table>

where each summand is defined to be zero unless its indicator equals one, and the division is carried out only in that case. Then:

(a) ${ \widehat { G } } ( h )$ is a function of the display of g, of the fixed objects $a , L , \pi , \zeta , b$ , and of the values of h at the displayed queries;

(b) $0 \leq \widehat { G } ( h ) \leq 2 / ( \zeta b )$ );

(c) $\mathbb { E } [ \widehat { G } ( h ) \mid T ] = \Phi _ { \pi } ( T , \rho _ { L } ( h ) ) ;$

(d) for every pair h, $h ^ { \prime } \in \mathcal { H }$ there is an event $O _ { h h ^ { \prime } }$ , decided by the display of g and the values of h and $h ^ { \prime }$ at the displayed queries, such that

$$
\operatorname* { P r } ( O _ { h h ^ { \prime } } \mid T ) \ \geq \ \zeta \Im \{ \rho _ { L } ( h ) \neq \rho _ { L } ( h ^ { \prime } ) \} .\tag{5.2}
$$

Proof. (a) Let $j \leq n _ { g }$ . The pairs at positions $1 , \ldots , j$ are displayed. By Lemma $4 . 2 ( \mathrm { i i } ) , Q _ { L } ( T , j )$ depends on the pairs before $j$ , while $Q _ { \pi } ( T , j )$ and $Q _ { \pi } ( T , j + 1 )$ depend on the pairs up to j. The indicators $\mathbf { 1 } \{ j > \tau _ { a } \}$ and $1 \{ j + 1 > \tau _ { a } \}$ in the potentials depend only on whether a agrees with the pairs up to $j ,$ , and the test on h uses the pairs up to $j .$ . Hence every summand is computable. The position $j + 1$ is virtual: only the predictions at positions up to $j$ enter $\Phi _ { \pi } ( T , j + 1 )$ , so no query at position $j + 1$ and no end-of-record information is needed. In particular, if g errs at the last actual query, then $j = n _ { g } = M$ is processed in exactly the same way, and the learner never needs to know that the record has ended. Summands with $j > n _ { g }$ are not included, and the descriptor is never used.

(c) Fix T. For $1 \leq j \leq M$ , the jth summand is present exactly when $j \leq n _ { g }$ , that is, when $\tau _ { g } \geq j$ , which by Lemma $4 . 2 ( \mathrm { i v } )$ has conditional probability $Q _ { L } ( T , j )$ . Its two tests say $j \le J _ { L }$ (by monotonicity of $Q _ { L } )$ and $\tau _ { h } \geq j + 1$ , that is, together, $j + 1 \le$ min $( J _ { L } + 1 , \tau _ { h } ) = \rho _ { L } ( h )$ . On this branch $Q _ { L } ( T , j ) \ge \zeta > 0$ . Since $\rho _ { L } ( h ) \leq M + 1$ , every $j$ with $j + 1 \le \rho _ { L } ( h )$ satisfies $j \leq M$ Therefore

$$
\mathbb { E } [ \widehat { G } ( h ) \mid T ] = \sum _ { l = 2 } ^ { \rho _ { L } ( h ) } \left( \Phi _ { \pi } ( T , l ) - \Phi _ { \pi } ( T , l - 1 ) \right) = \Phi _ { \pi } ( T , \rho _ { L } ( h ) ) - \Phi _ { \pi } ( T , 1 ) = \Phi _ { \pi } ( T , \rho _ { L } ( h ) ) .
$$

Procedure 1 Acquire $\prime L , a , w , \zeta , b , \alpha ) \colon$ one acquisition stage   
Input: logger L with $L ( a ) \ge 1 / 2 ;$ numbers $w \in ( 0 , 1 ] , \zeta , b \in ( 0 , 1 / 2 ] , \alpha \in ( 0 , 1 )$   
1: $W _ { 0 } \gets 1 / ( \zeta w b )$ $W  2 / ( \zeta b w \Lambda _ { b } )$ $C _ { \mathrm { E G } } \gets ( 2 + ( e - 1 ) e ) / ( 1 - e ^ { - 1 } )$   
2: m ← least integer $m \geq 8 W _ { 0 }$ with $2 ( 2 D m + 1 ) ^ { 2 d } 2 ^ { - m / ( 2 W _ { 0 } ) } \leq \alpha / 4$   
3: for $i = 1 , \ldots , m$ do draw $g \sim L ,$ play it on a fresh reset, store the display   
4: call the oracle on all queries displayed in these m resets; let $\mathcal { G }$ be the result; $K _ { 0 } \gets ( D m + 1 ) ^ { d }$   
5: $\iota  \log ( 4 K _ { 0 } / \alpha )$ , n $ \lceil W \iota \rceil$ $\pi _ { 1 } $ uniform on $\mathcal { G }$   
6: for $t = 1 , \ldots , n$ do   
7: draw $g _ { t } \sim L$ and play it on a fresh reset   
8: for every $h \in \mathcal { G } \colon \gamma _ { t , h } \gets \widehat { G } _ { t } ( h ) / ( w \Lambda _ { b } )$ , with $\widehat { G } _ { t }$ given by (5.1) for $\pi = \pi _ { t }$   
9: $\pi _ { t + 1 } ( h ) \gets \pi _ { t } ( h ) e ^ { \gamma _ { t , h } / W } / \sum _ { u \in \mathcal { G } } \pi _ { t } ( u ) e ^ { \gamma _ { t , u } / W }$ for $h \in { \mathcal { G } }$   
10: π¯ $\textstyle  { \frac { 1 } { n } } \sum _ { t = 1 } ^ { n } \pi _ { t }$ ▷ average of the pre-update weights   
11: $\begin{array} { r } { k \gets \left\lceil \frac { 8 } { b } \log \frac { 4 ( D + 1 ) } { \alpha b } \right\rceil ; } \end{array}$ ; draw k slots independently from $\bar { \pi }$ with private coins   
12: return the uniform lottery $p$ on the k slots (with multiplicity)

(b) The increments $\Phi _ { \pi } ( T , j + 1 ) - \Phi _ { \pi } ( T , j )$ are nonnegative: both potentials vanish when $j + 1 \le \tau _ { a } ,$ the increment at $j = \tau _ { a }$ equals $\Phi _ { \pi } ( T , \tau _ { a } + 1 ) \geq 0 $ , and for $j > \tau _ { a }$ the potential is nondecreasing. Every present summand has $j + 1 \le \rho _ { L } ( h )$ . The summand with $j = \tau _ { a } .$ if present, has divisor $Q _ { L } ( T , \tau _ { a } ) \geq 1 / 2$ and is at most $2 \Phi _ { \pi } ( T , \tau _ { a } + 1 ) \leq 2 / b$ . The summands with $j > \tau _ { a }$ have divisors at least $\zeta ,$ and their increments add up to at most $\Phi _ { \pi } ( T , \rho _ { L } ( h ) ) \leq 1 / b ,$ so they contribute at most $1 / ( \zeta b )$ . Summands with $j < \tau _ { a }$ vanish. Since $\zeta \leq 1 / 2$ , the total is at most $2 / b + 1 / ( \zeta b ) \leq 2 / ( \zeta b )$

(d) Let $O _ { h h ^ { \prime } }$ be the event that some displayed position $j \leq n _ { g }$ has $Q _ { L } ( T , j ) \ge \zeta$ , both h and $h ^ { \prime }$ agree with the pairs at positions $1 , \ldots , j - 1$ , and exactly one of them agrees with the pair at position j. It is decided by the display and by the values of $h , h ^ { \prime }$ at displayed queries. Suppose $\rho _ { L } ( h ) < \rho _ { L } ( h ^ { \prime } )$ (the other case is symmetric). Then $\rho _ { L } ( h ) \leq J _ { L }$ , so $\rho _ { L } ( h ) = \tau _ { h } \leq J _ { L }$ , and $\tau _ { h } < \rho _ { L } ( h ^ { \prime } ) \leq M + 1$ so $\tau _ { h } \leq M$ . Put $j ^ { * } = \tau _ { h }$ . Then h is correct before $j ^ { * }$ and errs at $j ^ { * }$ , while $\tau _ { h ^ { \prime } } \geq \rho _ { L } ( h ^ { \prime } ) > j ^ { * }$ , so $h ^ { \prime }$ is correct at $1 , \ldots , j ^ { * }$ . Since $j ^ { * } \le J _ { L }$ , we have $Q _ { L } ( T , j ^ { * } ) \geq \zeta .$ , and with conditional probability $\operatorname* { P r } ( \tau _ { g } \geq j ^ { * } \mid T ) = Q _ { L } ( T , j ^ { * } ) \geq \zeta$ the pair at $j ^ { * }$ is displayed; on this event $O _ { h h ^ { \prime } }$ occurs with $j = j ^ { * }$ This includes the case in which g itself errs at $j ^ { * }$ , because discrepancies are displayed. □

The legal content of (5.1) is the one-position shift: a pair displayed at position $j$ is used to estimate the increment of the potential at the virtual position $j + 1$ . The shift is possible because the pilot’s discrepancy pair is displayed, and it requires dividing by the propensity $Q _ { L } ( T , j )$ of position $j ,$ , not of position $j + 1$

## 5.2 One acquisition stage

The numerical optimization in a stage is the classical exponential-weights update [15, Eq. (3.3)] with the multiplicative-potential analysis of Freund and Schapire [13]. Once the legal bounded gain of Lemma 5.1 and a finite observed dictionary are available, the reset and support orders of one stage follow from a standard multiplicative-potential bound, bounded martingale concentration and one randomized sparsification. We prove this conversion for the displayed gain in Appendix $\mathrm { C } ;$ the optimizer itself is not a new primitive. The logger stays fixed while the numerical weights change, so the gains form an adapted sequence and their concentration needs martingale arguments rather than independence.

Lemma 5.2 (One acquisition stage). Condition on a history that fixes $a , L , w , \zeta , b ,$ α with $L ( a ) \geq 1 / 2$ $0 ~ < ~ w ~ \leq ~ 1 , ~ 0 ~ < ~ \zeta , b ~ \leq ~ 1 / 2$ and $0 ~ < ~ \alpha ~ < ~ 1$ , and let p be the output of Procedure 1. Put $C ( b ) = 2 C _ { \mathrm { E G } } \Lambda _ { b } + 2$

(a) On every history the procedure uses exactly m + n resets, makes one oracle call and returns a lottery with at most k slots. The pilot of every reset is drawn from L. Enumerating G, updating the numerical weights and drawing the slots play nothing by themselves; a member of G is played only if it is drawn from L.

(b) If $F ( a ) \leq w$ , then with conditional probability at least $1 - \alpha _ { : }$ , simultaneously for every $h \in \mathcal H$

$$
\begin{array} { r } { \mathbb { E } _ { T } \Phi _ { p } ( T , \rho _ { L } ( h ) ) \le C ( b ) w , } \end{array}\tag{5.3}
$$

where $\mathbb { E } _ { T }$ integrates a fresh reference record with $a , L , p$ held fixed.

(c) The numbers m, n, k are deterministic functions of $( d , D , w , \zeta , b , \alpha )$ , and

$$
m + n = \widetilde O \left( \frac { d + 1 + \log ( 1 / \alpha ) } { \zeta w b } \right) , \qquad k = O \left( \frac 1 b \log \frac { D } { \alpha b } \right) ,
$$

where the hidden factors are polylogarithmic in $D , d + 1 , 1 / ( \zeta w b )$ and $1 / \alpha$

The proof is in Appendix C. Two transfers are worth pointing out. First, the dictionary sample controls, uniformly over all pairs, the observable disagreement events of Lemma 5.1(d) before any representative of a particular comparator is chosen. Together with (5.2) this gives $\mathrm { P r } ( \rho _ { L } ( h ) \neq \rho _ { L } ( h ^ { \prime } ) ) \leq$ wb for every h and its representative $h ^ { \prime } ,$ at the price of exactly the visibility factor ζ. Second, after the numerical pass, one random sparse bad set of records controls all ranks of a record on which the anchor fails, so the dense and sparse potentials can be compared there; on records where the anchor survives both potentials vanish and no comparison is needed. The transfer from the dictionary to the whole class is paid once, after the sparsification, and costs at most w. The returned slots are available functions that have not necessarily been played. A later logger may play them, and each such use is counted in V.

## 6 From truncated visibility to a joint learner

Lemma 5.2 gives a certificate only at ranks truncated at the logger’s frontier. This section moves the frontier forward through a sequence of stages, recovers the weight lost by mixing the stages, and turns the resulting lottery into an output and an interval.

## 6.1 The stage graph

The stages are arranged along a sparse directed acyclic graph whose long paths survive the deletion of any set of vertices. Such graphs are classical. The multiscale connector argument is due to Erdős, Graham and Szemerédi [12]; the local-expander formulation that we follow is that of Alwen, Blocki and Pietrzak [3, Section 4, Lemmas 4.2–4.4]. Appendix D gives a self-contained proof with the constants we need.

Lemma 6.1 (Robust forward graph; Appendix D). Let $S \geq 2$ and $0 < \beta < 1$ , and put

$$
k _ { G } = \Big \lceil 4 0 \Big ( 2 \log ( 1 0 e ) + \log \frac { 2 S ^ { 2 } } { \beta } \Big ) \Big \rceil , \qquad k _ { A } = \operatorname* { m i n } \{ S - 1 , \ 1 9 + k _ { G } \big ( \lfloor \log _ { 2 } ( 2 S ) \rfloor + 1 \big ) \} .\tag{6.1}
$$

```latex
Procedure 2 PrecisionStage $( a , t , w , v , \beta , \Gamma )$
Input: anchor $a \in \mathcal H ;$ numbers $0 < t \leq w \leq 1 , v \in [ 0 , 1 ] , \beta \in ( 0 , 1 )$ ; a stage graph Γ on $[ S ] , S = 4 D .$
with predecessor sets $A _ { s }$ and indegree cap $k _ { A }$
1: $b $ the largest $b \in ( 0 , 1 / 2 ]$ with $b C ( b ) \leq t / ( 1 0 2 4 w )$ , where $C ( b ) = 2 C _ { \mathrm { E G } } \Lambda _ { b } + 2 ; ~ \zeta  b / ( 2 k _ { A } )$
2: for $s = 1 , \ldots , S$ do
3: $L _ { s } \gets \delta _ { a } \mathrm { ~ i f ~ } s = 1$ , and $\begin{array} { r } { L _ { s } \gets \frac { 1 } { 2 } \delta _ { a } + \frac { 1 } { 2 | A _ { s } | } \sum _ { r \in A _ { s } } p _ { r } \mathrm { ~ i f ~ } s \geq 2 } \end{array}$
4: $p _ { s } \gets \mathrm { A C Q U I R E } ( L _ { s } , a , w , \zeta , b , \beta / S )$
5: $\begin{array} { r } { \Pi  \frac { 1 } { 2 } \delta _ { a } + \frac { 1 } { 2 S } \sum _ { s = 1 } ^ { S } p _ { s } , \ b _ { F }  b / 2 , \ C _ { * }  6 4 C ( b ) } \end{array}$
6: $n _ { F } $ the sample size (6.11) for $W _ { Y } = 2 / b , c _ { 1 } = 2 , c _ { 2 } = 2 C _ { * } w .$ , accuracy t, ceiling v and failure
$\beta$
7: play $n _ { F }$ pilots drawn independently from Π on fresh resets and store their displays
8: call the oracle on all queries displayed in these resets; for each returned member h compute
$\begin{array} { r } { \widehat { R } ( h ) = \frac { 1 } { n _ { F } } \sum _ { i = 1 } ^ { n _ { F } } Y _ { h , i } } \end{array}$ with $Y _ { h }$ from (6.7); let $\widehat { h }$ be a returned member minimizing $\widehat { R }$ (fixed
tie-break)
9: $q  \widehat { R } ( \widehat { h } ) , \ l  ( q - t / 4 ) / ( 1 + \theta ) , \ u  ( q + t / 4 ) / ( 1 - \theta )$ $\triangleright \theta$ from (6.9)
10: clip l and u separately to $[ 0 , 1 ] ;$ if the result is unordered or wider than $t ,$ replace it by [0, 0]
11: return $\widehat { h }$ and the interval
```

Using private coins only, one can draw a directed graph Γ on $[ S ] = \{ 1 , \ldots , S \}$ , with every edge $( x , y )$ satisfying $x < y$ , such that on every draw each $s \geq 2$ has a predecessor set $A _ { s } \subseteq [ s - 1 ]$ with $s - 1 \in A _ { s }$ and $| A _ { s } | \le k _ { A }$ . With probability at least $1 - \beta$ the graph has the path property: for every set $B \subseteq [ S ]$ there is a directed path in Γ that avoids B and has at least $S - 4 | B |$ vertices (a vacuous statement when $S - 4 | B | \leq 0 )$

The set B in the path property is arbitrary; it may depend on the graph and on all data collected later. The cap $k _ { A }$ is a deterministic number, and every parameter below uses $k _ { A }$ rather than the realized indegrees. Note that $k _ { A } = O ( ( 1 + \log S ) \log ( S / \beta ) )$ ).

More generally, we call a stage graph any directed graph Γ on [S] with edges $( x , y ) , x < y$ together with a number $k _ { A }$ fixed before Γ is drawn, such that on every draw $s - 1 \in A _ { s }$ and $| A _ { s } | \le k _ { A }$ for $s \geq 2$ . Besides the graph of Lemma 6.1, the complete graph with all edges $( x , y )$ 2 $x < y .$ , and $k _ { A } = S - 1$ is a stage graph; it has the path property on every draw, since $[ S ] \setminus B$ listed in increasing order is a path with $S - | B |$ vertices.

## 6.2 One precision stage

Procedure 2 describes one precision stage. It receives a proper anchor a, a precision t, a numerical bound w intended to satisfy $F ( a ) \leq w$ , a numerical bound v intended to satisfy $R ^ { * } \leq v .$ , a failure allowance $\beta _ { i }$ , and a stage graph. The procedure and all its sample sizes are defined whether or not $F ( a ) \leq w$ and $R ^ { * } \leq v$ hold; these inequalities are needed only for the coverage guarantee.

Every logger $L _ { s }$ consists of the anchor and of the sparse designs $p _ { r }$ of earlier stages, so it uses only functions that are available when it is formed. The dense dictionaries of the designs are never loggers. Averaging earlier exploration distributions is an established device; for example, PC-PG averages the visitation distributions of earlier policies [2, Section 3, Eq. (3)]. What has to be shown here is that the average Π has a good certificate at the true stopping ranks, using only legal acquisition.

Lemma 6.2 (Choice of b). Let $0 < t \leq w \leq 1$ . The function $b \mapsto b C ( b )$ is continuous and strictly increasing on $( 0 , 1 / 2 ]$ , tends to 0 as $b  0$ , and exceeds $t / ( 1 0 2 4 w )$ at $b = 1 / 2$ . Hence the b of

Procedure 2 is the unique root of $b C ( b ) = t / ( 1 0 2 4 w )$ in $( 0 , 1 / 2 )$ . It satisfies

$$
\frac { t } { 6 5 5 3 6 w \log ( 1 0 2 4 w / t ) } ~ \leq ~ b ~ \leq ~ \frac { t } { 2 0 4 8 w } \leq \frac { 1 } { 2 0 4 8 } ,\tag{6.2}
$$

so that $1 / b = O \big ( ( w / t ) \log ( e w / t ) \big ) , \ : \Lambda _ { b } = O ( \log ( e w / t ) ) , \ : C ( b ) = O ( \log ( e w / t ) )$ , and $\Lambda _ { b } \geq \log 2 0 4 9 > 2$

Proof. The derivative of $b C ( b ) = b ( 2 C _ { \mathrm { E G } } \log ( 1 + 1 / b ) + 2 )$ is $2 C _ { \mathrm { E G } } [ \log ( 1 + 1 / b ) - 1 / ( 1 + b ) ] + 2 .$ which is at least 2 because $\log ( 1 + x ) \geq x / ( 1 + x )$ with $x = 1 / b$ . Since $b \log ( 1 + 1 / b ) \to 0 , b C ( b ) \to 0$ as $b \to 0 . \mathrm { A t } \ b = 1 / 2$ we have $b C ( b ) = C _ { \mathrm { E G } } \log 3 + 1 > 1 > t / ( 1 0 2 4 w )$ . This proves the first claims. $\operatorname { A s } C ( b ) \geq 2$ , the root satisfies $2 b \leq t / ( 1 0 2 4 w )$ , which is the upper bound in (6.2); then $\Lambda _ { b } = \log ( 1 + 1 / b ) \geq \log 2 0 4 9$

For the lower bound put $x = 1 0 2 4 w / t \geq 1 0 2 4$ and $b ^ { \prime } = 1 / ( 6 4 x \log x ) \leq 1 / 2$ . Since 65 log $x \leq x ^ { 2 }$ for $x \geq 1 0 2 4$ , we have $1 + 1 / b ^ { \prime } = 1 +$ 64x log x $\leq x ^ { 3 }$ , hence $\log ( 1 + 1 / b ^ { \prime } ) \leq 3 \log x$ and

$$
b ^ { \prime } C ( b ^ { \prime } ) \leq \frac { 6 C _ { \mathrm { E G } } \log x + 2 } { 6 4 x \log x } = \frac { 6 C _ { \mathrm { E G } } + 2 / \log x } { 6 4 x } \leq \frac { 1 } { x } ,
$$

because $C _ { \mathrm { E G } } = 1 0 . 5 5 3 0 . .$ . and log x ≥ log 1024 give $6 C _ { \mathrm { E G } } + 2 / \log x < 6 3 . 6 1 < 6 4$ . Thus $b ^ { \prime } C ( b ^ { \prime } ) \leq$ $t / ( 1 0 2 4 w )$ , and monotonicity gives $b \geq b ^ { \prime } = t / ( 6 5 5 3 6 w \log ( 1 0 2 4 w / t ) )$ . The order statements follow from $\log ( 1 0 2 4 w / t ) \leq \log 1 0 2 4 + \log ( w / t )$ and $\Lambda _ { b } \le \log ( 1 + 1 / b ^ { \prime } ) \le 3 \log x$ □

## 6.3 Recordwise propagation with correlated failures

Freeze the graph and the designs $p _ { 1 } , \ldots , p _ { S }$ . Fix $h \in \mathcal H$ and a reference record T with $\tau _ { h } > \tau _ { a }$ . For each stage s let

$$
\rho _ { s } ( h ) = \operatorname* { m i n } ( \tau _ { h } , J _ { L s } + 1 ) , \quad \quad \Delta = \tau _ { h } - \tau _ { a } , \quad \quad x _ { s } = \rho _ { s } ( h ) - \tau _ { a } ,
$$

where $J _ { L _ { s } }$ is the frontier of $L _ { s }$ with threshold ζ. Call stage s successful for $( h , T )$ if $Q _ { p _ { s } } ( T , \rho _ { s } ( h ) ) \geq b .$ and let Bad $( h , T ) \subseteq [ S ]$ be the set of unsuccessful stages. Call s visible for $( h , T ) \mathrm { i f } \rho _ { s } ( h ) = \tau _ { h }$ , and let $\operatorname { V i s } ( h , T )$ be the set of visible stages.

Lemma 6.3 (Propagation and visibility). In the situation above:

(i) $1 \leq x _ { s } \leq \Delta \leq D$ for every s;

(ii) $i f r \in A _ { s } \setminus \operatorname { B a d } ( h , T )$ , then $x _ { s } \ge \operatorname* { m i n } ( \Delta , x _ { r } + 1 )$ ;

(iii) if Γ has the path property, then $| \mathrm { V i s } ( h , T ) | \geq S - 4 | \mathrm { B a d } ( h , T ) | - \Delta + 1$ whenever the right side is positive.

If moreover the certificate (5.3) holds for every stage s (with logger $L _ { s } )$ , then for every $h \in \mathcal H$ and every s,

$$
\operatorname* { P r } _ { T } ( \tau _ { h } > \tau _ { a } , \ s \in \mathrm { B a d } ( h , T ) ) \le 2 b C ( b ) \ w ,\tag{6.3}
$$

and if in addition Γ has the path property, then

$$
\operatorname* { P r } _ { T } ( \tau _ { h } > \tau _ { a } , \ | \mathrm { V i s } ( h , T ) | < S / 4 ) \leq 1 6 b C ( b ) w .\tag{6.4}
$$

Proof. (i) Since $J _ { L _ { s } } \geq \tau _ { a }$ and $\tau _ { h } > \tau _ { a } .$ we have $\rho _ { s } ( h ) \geq \tau _ { a } + 1$ , so $x _ { s } \geq 1$ ; and $\rho _ { s } ( h ) \leq \tau _ { h }$ gives $x _ { s } \le \Delta$ . Finally $\Delta \leq ( M + 1 ) - 1 \leq D$ because $\tau _ { a } \geq 1$

(ii) Let $r \in A _ { s } \ \backslash$ Bad $. ( h , T )$ . Since $L _ { s }$ gives weight $1 / ( 2 | A _ { s } | )$ to $p _ { r }$ and propensities are linear in the lottery,

$$
Q _ { L s } ( T , \rho _ { r } ( h ) ) ~ \geq ~ \frac { Q _ { p _ { r } } ( T , \rho _ { r } ( h ) ) } { 2 | A _ { s } | } ~ \geq ~ \frac { b } { 2 k _ { A } } = \zeta .
$$

As $\rho _ { r } ( h ) \leq M + 1$ , the definition of the frontier gives $J _ { L _ { s } } \geq \rho _ { r } ( h )$ , hence $\rho _ { s } ( h ) \geq \operatorname* { m i n } ( \tau _ { h } , \rho _ { r } ( h ) + 1 )$ which is the claim.

(iii) Let $s _ { 1 } < \cdots < s _ { \ell }$ be the vertices of a path avoiding Bad(h, T) with $\ell \geq S - 4 | \mathrm { B a d } ( h , T ) |$ 2 so $s _ { i } \in A _ { s _ { i + 1 } } \setminus \operatorname { B a d } ( h , T )$ . By (i) and $( \mathrm { i i } ) , x _ { s _ { i } } \geq \operatorname* { m i n } ( \Delta , i )$ by induction on i. Hence $s _ { \Delta } , \ldots , s _ { \ell }$ are visible, which gives $\ell - \Delta + 1$ visible stages when this number is positive. If the claimed bound is nonpositive there is nothing to prove.

For (6.3), note that on $\{ \tau _ { h } > \tau _ { a } \}$ we have $\rho _ { s } ( h ) > \tau _ { a } , \mathrm { ~ s o ~ } \Phi _ { p _ { s } } ( T , \rho _ { s } ( h ) ) = 1 / ( Q _ { p _ { s } } ( T , \rho _ { s } ( h ) ) + b )$ For $q \geq 0 , 1 \{ q < b \} \leq 2 b / ( q + b )$ . Hence

$$
\operatorname* { P r } _ { T } ( \tau _ { h } > \tau _ { a } , ~ s \in \mathrm { B a d } ( h , T ) ) \le 2 b \mathbb { E } _ { T } \Phi _ { p _ { s } } ( T , \rho _ { s } ( h ) ) \le 2 b C ( b ) w
$$

by (5.3). For (6.4), suppose $\tau _ { h } > \tau _ { a }$ and $| \mathrm { V i s } ( h , T ) | < S / 4 = D . \mathrm { I f } \ S - 4 | \mathrm { B a d } | - \Delta + 1 > 0$ then (iii) gives $D > 4 D - 4 | { \mathrm { B a d } } | - \Delta + 1 \geq 3 D - 4 | { \mathrm { B a d } } | + 1 , \mathrm { ~ s o ~ } | { \mathrm { B a d } } | > D / 2$ . Otherwise $4 | \mathrm { B a d } | \geq 4 D - \Delta + 1 \geq 3 D + 1$ , and again $| \mathrm { B a d } | > D / 2$ . Markov’s inequality and linearity of expectation give

$$
\begin{array} { r l r } & { } & { { \displaystyle \operatorname* { P r } _ { T } ( \tau _ { h } > \tau _ { a } , | \mathrm { B a d } ( h , T ) | > D / 2 ) \le \frac { 2 } { D } \sum _ { s = 1 } ^ { S } \operatorname* { P r } ( \tau _ { h } > \tau _ { a } , s \in \mathrm { B a d } ( h , T ) ) } } \\ & { } & { { \displaystyle \le \frac { 2 } { D } \cdot 4 D \cdot 2 b C ( b ) w = 1 6 b C ( b ) w } . } \end{array}
$$

No independence between the failures of diferent stages is used: Bad(h, T) may be any set, and it may depend on the graph, the designs, h and T. □

## 6.4 Harmonic recovery of the mixture weight

Lemma 6.4 (Final certificate). Suppose Γ has the path property and the certificate (5.3) holds for every stage. Then, simultaneously for every $h \in \mathcal H$ 2

$$
\mathbb { E } _ { T } \frac { { \bf 1 } \{ \tau _ { h } > \tau _ { a } \} } { Q _ { \Pi } ( T , \tau _ { h } ) + b _ { F } } \ \le \ C _ { * } w , \qquad C _ { * } = 6 4 C ( b ) , \ b _ { F } = b / 2 .\tag{6.5}
$$

Proof. Fix h and a record with $\tau _ { h } > \tau _ { a }$ and $| \mathrm { V i s } ( h , T ) | \geq S / 4$ . Then $Q _ { \delta _ { a } } ( T , \tau _ { h } ) = 0$ , so with $q _ { s } = Q _ { p _ { s } } ( T , \tau _ { h } )$

$$
Q _ { \Pi } ( T , \tau _ { h } ) + \frac { b } { 2 } = \frac { 1 } { 2 S } \Bigl ( \sum _ { s = 1 } ^ { S } q _ { s } + S b \Bigr ) \ \ge \ \frac { 1 } { 2 S } \sum _ { s \in \mathrm { V i s } ( h , T ) } ( q _ { s } + b ) .
$$

For positive numbers $y _ { s }$ , the Cauchy–Schwarz inequality gives $\begin{array} { r } { \left( \sum _ { s \in \mathrm { V i s } } y _ { s } \right) ^ { - 1 } \leq | \mathrm { V i s } | ^ { - 2 } \sum _ { s \in \mathrm { V i s } } y _ { s } ^ { - 1 } } \end{array}$ With $y _ { s } = q _ { s } + b$ and $| \mathrm { V i s } | \ge S / 4$

$$
\frac { 1 } { Q _ { \mathrm { I I } } ( T , \tau _ { h } ) + b / 2 } \le \frac { 2 S \cdot 1 6 } { S ^ { 2 } } \sum _ { s \in \mathrm { V i s } ( h , T ) } \frac { 1 } { q _ { s } + b } = \frac { 3 2 } { S } \sum _ { s \in \mathrm { V i s } ( h , T ) } \Phi _ { p _ { s } } ( T , \rho _ { s } ( h ) ) ,\tag{6.6}
$$

because $\rho _ { s } ( h ) = \tau _ { h } > \tau _ { a }$ for visible s. Integrate (6.6) over the records with $\tau _ { h } > \tau _ { a }$ and $| \mathrm { V i s } | \geq S / 4 .$ enlarge the nonnegative sum to all $s \in [ S ]$ , and apply (5.3) to each stage; this part contributes at most $( 3 2 / S ) \cdot S \cdot C ( b ) w = 3 2 C ( b ) w$ . On the records with $\tau _ { h } > \tau _ { a }$ and $| \mathrm { V i s } | < S / 4$ , bound the integrand by $1 / b _ { F } = 2 / b$ and use (6.4); this contributes at most $( 2 / b ) \cdot 1 6 b C ( b ) w = 3 2 C ( b ) w$ . The sum is $C _ { * } w$ □

The factor 1/(2S) with which each design enters Π is recovered because a constant fraction of the stages is visible; bounding the contribution of a single stage would lose a further factor of order D. The certificate holds simultaneously for all h, but its exceptional set of records with low propensity depends on h. We never take a union of these sets over the class.

## 6.5 The final estimate and the observable interval

For the final estimate, pilots are drawn from the frozen lottery Π before fresh records. For $h \in \mathcal H$ write $Q _ { h } = Q _ { \Pi } ( T , \tau _ { h } )$ and define the score

$$
Y _ { h } = \frac { \lambda ( T , \tau _ { h } ) } { Q _ { h } } \mathrm { i f } \tau _ { g } \geq \tau _ { h } \mathrm { a n d } Q _ { h } \geq b _ { F } , \qquad Y _ { h } = 0 \mathrm { o t h e r w i s e } ,\tag{6.7}
$$

where the test is carried out first and the division only on the first branch. This is inverse-propensity estimation under a frozen lottery; inverse-action-probability exploration is classical, see for instance Dudík et al. [11, Section 2.2, Lemma 1 and Algorithm 1].

Lemma 6.5 (The final score). $( a ) \ Y _ { h }$ is a function of the display of g and of the values of h at the displayed queries, and it depends on h only through $\tau _ { h } ( T )$

(b) Conditionally on $T , \mathbb { E } [ Y _ { h } \mid T ] = \ell _ { h } ( T ) { \bf 1 } \{ Q _ { h } \geq b _ { F } \} , 0 \leq Y _ { h } \leq 2 / b$ and

$$
\mathbb { E } [ Y _ { h } ^ { 2 } \mid T ] = { \left\{ \begin{array} { l l } { \ell _ { h } ( T ) ^ { 2 } / Q _ { h } } & { i f Q _ { h } \geq b _ { F } , } \\ { 0 } & { i f Q _ { h } < b _ { F } . } \end{array} \right. }
$$

(c) If (6.5) holds and b is chosen as in Procedure ${ \mathcal { Q } } ,$ then for every $h \in \mathcal H$

$$
0 \leq R ( h ) - \mathbb { E } Y _ { h } \leq 2 b _ { F } C _ { * } w = 6 4 b C ( b ) w \leq \frac { t } { 1 6 } , \qquad \mathbb { E } Y _ { h } ^ { 2 } \leq 2 R ( h ) + 2 C _ { * } w .\tag{6.8}
$$

Proof. (a) If h disagrees with some displayed pair, its first disagreement is $\tau _ { h } \leq n _ { g } \leq \tau _ { g } ;$ then $Q _ { h }$ depends on the pairs before $\tau _ { h } , \lambda ( T , \tau _ { h } )$ on the pairs up to $\tau _ { h }$ , and all are displayed. If $g$ survives, the whole record and its descriptor are displayed, so $\tau _ { h } , \ Q _ { h }$ and $\lambda ( T , \tau _ { h } )$ are known, including on an empty record, where $\tau _ { h } = 1$ and $Q _ { h } = 1$ . If g errs and h agrees with every displayed pair, then $\tau _ { h } > \tau _ { g } ,$ , the test fails and $Y _ { h } = 0$ ; this case is recognized from the display (no end tag, no disagreement), and the later loss of h is never evaluated. In all cases $Y _ { h }$ depends on h only through $\tau _ { h }$

(b) Given T, the rank $\tau _ { h }$ and the number $Q _ { h }$ are fixed. If $Q _ { h } < b _ { F }$ , then $Y _ { h } = 0$ . If $Q _ { h } \geq b _ { F }$ then $Y _ { h } = \lambda ( T , \tau _ { h } ) / Q _ { h }$ on the event $\tau _ { g } \geq \tau _ { h }$ , which has conditional probability $Q _ { h }$ by Lemma $4 . 2 ( \mathrm { i v } )$ and $Y _ { h } = 0$ otherwise. This gives both moments, and on the nonzero branch $Y _ { h } \le 1 / b _ { F } = 2 / b$

(c) By (b), $R ( h ) - \mathbb { E } Y _ { h } = \mathbb { E } [ \ell _ { h } { \mathbf { 1 } } \{ Q _ { h } < b _ { F } \} ] \in [ 0 , \operatorname* { P r } ( Q _ { h } < b _ { F } ) ] . \mathrm { ~ I f ~ } \tau _ { h } \leq \tau _ { a } ,$ the anchor survives up to $\tau _ { h }$ , so $\begin{array} { r } { Q _ { h } \geq \frac { 1 } { 2 } \geq b _ { F } } \end{array}$ . If $\tau _ { h } > \tau _ { a }$ , use ${ \bf 1 } \{ Q _ { h } < b _ { F } \} \le { 2 b _ { F } } / ( Q _ { h } + b _ { F } )$ and (6.5). Thus the bias is at most $2 b _ { F } C _ { * } w ^ { - } = 6 4 b C ( b ) w \leq 6 4 t / 1 0 2 4 = t / 1 6$ by the choice of b. For the second moment, on $\{ \tau _ { h } \ \leq \tau _ { a } \}$ we have $\ell _ { h } ^ { 2 } / Q _ { h } \leq 2 \ell _ { h }$ . On $\{ \tau _ { h } > \tau _ { a } , Q _ { h } \geq b _ { F } \}$ we have $Q _ { h } + b _ { F } \leq 2 Q _ { h }$ , hence $\ell _ { h } ^ { 2 } / Q _ { h } \leq 2 / ( Q _ { h } + b _ { F } )$ , and (6.5) bounds the integral of this part by $2 C _ { * } w$ □

None of these bounds assumes $R ( h ) \leq v$ , neither for all candidates nor for the selected output.

Estimation parameters. Let $W _ { Y } = 2 / b , c _ { 1 } = 2 , c _ { 2 } = 2 C _ { * } w$ , and

$$
s _ { E } = \operatorname* { m i n } \Bigl \{ \frac { 1 } { W _ { Y } } , \frac { t } { 1 6 c _ { 1 } ( v + t ) } , \frac { t } { 1 6 \operatorname* { m a x } ( c _ { 2 } , t ) } \Bigr \} , \qquad \theta = s _ { E } c _ { 1 } .\tag{6.9}
$$

For an integer $n \geq 1$ define the law-free numbers

$$
\begin{array} { r l } & { \nu _ { n } = \operatorname* { m i n } \bigl \{ v ^ { \prime } \in \mathbb { N } : \ v ^ { \prime } \geq 8 n , \ 2 ( 2 v ^ { \prime } D + 1 ) ^ { 2 d } e ^ { - v ^ { \prime } / ( 4 n ) } < 1 \bigr \} , \qquad H _ { n } = ( \nu _ { n } D + 1 ) ^ { d } , } \\ & { \iota _ { n } = \log \frac { 4 H _ { n } } { \beta } , \qquad \kappa _ { n } = \operatorname* { m a x } \Bigl \{ 4 , \ 1 8 \log \frac { 4 ( 2 n D + 1 ) ^ { 2 d } } { \beta } \Bigr \} , } \end{array}\tag{6.10}
$$

and let $n _ { F } \geq 2$ be the least integer n with

$$
\frac { 1 6 \iota _ { n } } { s _ { E } n } \leq t \qquad \mathrm { a n d } \qquad \frac { 1 6 \left( W _ { Y } \kappa _ { n } + 2 \right) } { n } \leq t .\tag{6.11}
$$

The quantities $\nu _ { n }$ and $H _ { n }$ are the size parameters of a latent-record net (Lemma A.3 with mass $1 / n )$ , and $\kappa _ { n }$ is a rare-count threshold (Lemma A.4). The net is used only in the analysis; the learner never constructs it.

Lemma 6.6 (Uniform relative estimate and interval; Appendix E). Condition on the history before the final estimate, and suppose that (6.5) holds. Then with conditional probability at least $1 - \beta$

$$
| \widehat { R } ( h ) - R ( h ) | \leq \theta R ( h ) + \frac { t } { 4 } \qquad f o r \ e v e r y \ h \in \mathcal { H } .\tag{6.12}
$$

$O n$ this event the raw endpoints of Procedure 2 satisfy $l \leq R ^ { * } \leq R ( \widehat { h } ) \leq u$ . If moreover $R ^ { * } \leq v ,$ then $u - l \leq 2 t / 3$ , so the clipped interval is ordered, has width at most t, and contains $R ^ { * }$ and $R ( \widehat { h } )$ On every history the returned interval is ordered, lies in [0, 1] and has width at most t, and $\widehat { h } \in \mathcal { H }$

The selection step enumerates the patterns of H on the queries displayed in the final $n _ { F }$ resets. Since $\widehat { R } ( h )$ depends on h only through its values on these queries, a minimizer over the returned representatives is a minimizer over all of $\mathcal { H } ;$ no population minimizer is assumed. This enumeration is separate from the design dictionaries.

## 6.6 The precision stage and the proof of Theorem 3.1

Proposition 6.7 (One precision stage). Let Γ be a stage graph with cap $k _ { A }$ , for instance the graph of Lemma 6.1 drawn with any allowance. Run Procedure 2 with $0 < t \leq w \leq 1 , v \in [ 0 , 1 ]$ and $\beta \in ( 0 , 1 )$

(a) On every history the output is a member of H and an ordered interval in [0, 1] of width at most t. The number of resets is

$$
\sum _ { s = 1 } ^ { S } ( m _ { s } + n _ { s } ) + n _ { F } = \widetilde O \bigg ( \big ( d + 1 + \log ( D / \beta ) \big ) \frac { D k _ { A } w + v + w + t } { t ^ { 2 } } \bigg ) ,\tag{6.13}
$$

where the hidden factors are polylogarithmic in d $+ 1 , D , k _ { A } , 1 / t$ and $1 / \beta$ . Every function played is the anchor or a slot of one of the S designs, so the stage plays at most

$$
1 + S \Big \lceil \frac { 8 } { b } \log \frac { 4 ( D + 1 ) S } { \beta b } \Big \rceil = \tilde { O } \Big ( \frac { D w } { t } \Big )
$$

distinct functions. The stage makes $S$ design oracle calls and one selection call.

Procedure 3 Learner of Theorem 3.1 (supplied ceilings $f , r )$   
Input: $\overline { { d , D , \varepsilon , \delta , f , r } }$   
1: $\mathbf { i f } \ \varepsilon = 1$ then return an empty-list representative and $[ 0 , 1 ]$   
2: run the procedure of Lemma 4.5 with failure $\alpha = \delta / 4 ;$ obtain a and w   
3: draw Γ by Lemma 6.1 on $[ 4 D ]$ with allowance $\delta / 1 6$   
4: return PrecisionStage $( a , \varepsilon , w , r , \delta / 1 6 , \Gamma )$

(b) Suppose $F ( a ) \leq w$ and $R ^ { * } \leq v$ . Conditionally on the history before the stage and on any realization $o f \Gamma$ with the path property, the interval covers $R ^ { * }$ and $R ( \widehat { h } )$ with probability at least $1 - 2 \beta$ . Consequently, if Γ is drawn with allowance $\beta$ just before the stage, the interval covers with conditional probability at least $1 - 3 \beta$

All sample sizes are numerical functions of $( d , D , t , w , v , \beta , k _ { A } )$

Proof. (b) Each call of Procedure 1 uses a logger with $L _ { s } ( a ) \ge 1 / 2$ and fresh resets. Since $F ( a ) \leq w _ { : }$ Lemma 5.2(b) gives, conditionally on every history before that call, failure probability at most $\beta / S .$ A union bound over the S calls shows that all certificates hold with probability at least $1 - \beta$ On this event and on the path property, Lemma 6.4 gives (6.5). Conditionally on the designs, Lemma 6.6 then gives coverage with probability at least $1 - \beta$ . The two failure probabilities add up to 2β; drawing the graph adds $\beta .$

$$
1 / b \ =
$$

$$
O ( ( w / t ) \log ( e w / t ) )
$$

$$
W _ { 0 } = 1 / ( \zeta w b ) = 2 k _ { A } / ( w b ^ { 2 } )
$$

$$
W = 2 W _ { 0 } / \Lambda _ { b } \leq W _ { 0 }
$$

$$
{ \Lambda } _ { b } > 2
$$

$$
\beta / S _ { ; }
$$

$$
m _ { s } + n _ { s } = \widetilde { O } \left( \left( d + 1 + \log ( S / \beta ) \right) \frac { k _ { A } } { w b ^ { 2 } } \right) = \widetilde { O } \left( \left( d + 1 + \log ( D / \beta ) \right) \frac { k _ { A } w } { t ^ { 2 } } \right) ,
$$

and summing over $S = 4 D$ designs gives the first term of (6.13). By Lemma E.2 in Appendix $\mathrm { E } ,$

$$
n _ { F } = \tilde { O } \left( ( d + 1 + \log ( 1 / \beta ) ) \Big [ \frac { W _ { Y } } { t } + \frac { c _ { 1 } ( v + t ) } { t ^ { 2 } } + \frac { c _ { 2 } + t } { t ^ { 2 } } \Big ] \right) ,
$$

with $W _ { Y } / t = 2 / ( b t ) = \tilde { O } ( w / t ^ { 2 } )$ and $c _ { 2 } = 1 2 8 C ( b ) w = \widetilde { O } ( w )$ ; this is the remaining part of (6.13). By Lemma 5.2(a), every pilot of the stage is drawn from a logger $L _ { s }$ or from Π, and these lotteries are built from the anchor and the slots of $p _ { 1 } , \ldots , p _ { S } ;$ so every function played in the stage is the anchor or a slot. With failure $\beta / S$ each design has $\lceil ( 8 / b ) \log ( 4 ( D + 1 ) S / ( \beta b ) ) \rceil$ slots, which gives the bound on played functions; by Lemma 6.2 it is ${ \cal \tilde { O } } ( D w / t )$ . Each design makes one oracle call and the selection makes one more. □

Proof of Theorem 3.1. If $\varepsilon = 1$ , Procedure 3 uses one oracle call, no reset and no pilot, and the interval [0, 1] covers every pair of numbers in [0, 1]. Let $\varepsilon < 1$ . By Lemma $4 . 5 , \varepsilon \le w \le 1$ and $w \leq 2 ( f + \varepsilon )$ , so $t = \varepsilon$ satisfies $t \leq w$ . If $F ^ { * } \leq f$ and $R ^ { * } \leq r ,$ the anchor satisfies $F ( a ) \leq w$ except with probability $\delta / 4$ , and then Proposition $6 . 7 ( \mathrm { b } )$ with $v = r$ and $\beta = \delta / 1 6$ gives coverage except with probability $3 \delta / 1 6 .$ The total failure probability is at most $\delta / 4 + 3 \delta / 1 6 < \delta$

For the resources, the anchor uses $\widetilde { O } ( ( d + 1 ) D / \bar { \varepsilon } )$ resets with $\bar { \varepsilon } = \operatorname* { m i n } ( 1 , f + \varepsilon ) \geq \varepsilon$ , and

$$
\frac { D } { \bar { \varepsilon } } \leq \frac { D } { \varepsilon } \leq \frac { D ( f + \varepsilon ) } { \varepsilon ^ { 2 } } .
$$

In (6.13), $k _ { A } = O ( ( 1 + \log D ) \log ( e D / \delta ) )$ is polylogarithmic, $w \leq 2 ( f + \varepsilon )$ $v \ = \ r$ and $t = \varepsilon .$ so the stage uses $\widetilde { \cal O } ( ( d + 1 ) [ D ( f + \varepsilon ) + r ] / \varepsilon ^ { 2 } )$ resets; the term $\log ( D / \beta )$ is absorbed into the polylogarithmic factor multiplying $d + 1$ . The anchor plays ${ \cal \tilde { O } } ( D )$ functions and the stage plays $\widetilde { O } ( D w / \varepsilon ) = \widetilde { O } ( D ( f + \varepsilon ) / \varepsilon )$ functions, which gives $V = \widetilde { \cal O } ( D ( 1 + f / \varepsilon ) )$ . The sample sizes and slot counts of the precision stage are numerical functions of $( d , D , \varepsilon , \delta , f , r )$ , since w is a function of $( \varepsilon , f )$ . The batch sizes of the anchor depend on observed endpoints, but by the proof of Lemma 4.5 in Appendix B they are bounded on every history by numbers that depend only on $( d , D , \varepsilon , \delta , f )$ So every phase has a deterministic numerical upper bound, and the caps hold on every history, whether or not the supplied numbers are valid ceilings and whether or not the confidence events occur. Appendix F proves Proposition 3.3, including the complete accounting of oracle calls.

## 7 The lower-bound constructions

Appendix G proves Theorems 3.5 and 3.6 with finite families of instances: a learner that succeeds with probability at least $4 / 5$ on every instance of a family must have a large expected cost on some instance. Each family uses one known loss rule, and its hidden parameters are carried by the truth function and the record law. Diferent terms use diferent families, and a maximum of separate lower bounds gives their sum up to a constant. We describe the mechanisms because they explain why the two ceilings and the two resources enter diferently.

Three obstructions for resets. The term $d D / \varepsilon$ comes from rare realizable threshold paths. With probability proportional to ε, a record selects one of d independent threshold blocks and follows the binary-search path of length D towards the hidden threshold of that block. The address of each query depends only on the earlier branch decisions, so it reveals no future bit. A frozen pilot that has not yet seen a bit predicts it correctly with probability $1 / 2 ,$ , so it reveals at most two new bits in expectation before its first error. Identifying most of the d thresholds requires $\Omega ( d D )$ bits, hence $\Omega ( d D / \varepsilon )$ resets. These are familiar fresh-bit and symmetric-counterexample mechanisms; related decision-tree constructions appear in Braverman et al. [10, Example 2]. We prove the exact stopped-record version rather than importing an equivalence-query theorem whose interface does not match.

For the term $d D f / \varepsilon ^ { 2 }$ , a hard record occurs with probability f. It selects a block and one of K arms, asks the gate query of that arm, and then follows a threshold path on the arm. A member of the class opens exactly one gate per block. A discrepancy at a wrong gate costs nothing under the target loss, while a later discrepancy or a bad terminal arm costs one. Hence $R ^ { * } = 0$ and $F ^ { * } \leq f$ but finding enough good arms requires both matching the gate and paying the fresh-bit price on the path. Taking K of order $f / \varepsilon$ gives the term. Properness matters here: a pilot outside the class that opens every gate, or an output outside the class that rejects every gate, could avoid the obstruction. Contextual combination locks of a related kind appear in Jia et al. [17]; here both the observations and the target loss are those of the stopped-record experiment.

For the term $d r / \varepsilon ^ { 2 }$ , records have length one and carry hidden Bernoulli terminal descriptors. A good pilot can have zero discrepancy probability, but the terminal means of the candidates still have to be estimated to accuracy of order $\varepsilon / r$ on records of probability r. Giving the learner complete records only makes it stronger, and a stopped-transcript relative-entropy bound gives the lower bound even then. This witness shows why the r term carries no extra factor of the horizon.

Distinct functions rather than resets. The pilot floor $\Omega ( D )$ comes from the threshold paths again: an unchanged pilot that meets the same deterministic record repeats the same stopped prefix and exposes nothing new, so only the first encounter of each pilot with each block can reveal new bits. The term $\Omega ( D f / \varepsilon )$ uses the free-gate family and counts completed arms in a comparison experiment in which every arm of one block is bad. The final output is allowed to be one additional unplayed guess. For expected costs, the proof truncates the number of distinct pilots and imposes a common finite reset cap, which exists because runs on the finite family terminate almost surely; no budget assumption is placed on the comparison experiment, which is not a promised instance. A lower bound on resets, or on the support of the final lottery, would not by itself bound the number of distinct functions used in training.

The natural-loss boundary. For natural discrepancy at $D \ \geq 3 .$ , a diluted family of gated threshold trees, in which some blocks have a winning arm and others have none, gives both the reset term and the pilot term. At $D = 2$ , a separate hidden-arm family gives the pilot term. In that family the class predicts one at every second query; adding arbitrary further functions to the class could remove the hardness, which is why it is a statement about a specific class. At $D = 1$ every query and its label are displayed whatever pilot is used, so one fixed pilot already yields an ordinary sample of the natural losses of all candidates, and the natural pilot lower bound fails. A general loss at $D = 1$ can still hide a terminal descriptor after a discrepancy, and the general pilot lower bound remains valid there. This is why the two pilot lower bounds have diferent horizon ranges.

## 8 Related work

We compare exact interfaces: who chooses what before a record is drawn, what is displayed, what is counted, and what the output must satisfy. Several close results become legal special cases of our experiment after a reduction. These reductions are our derivations, not theorems stated in the cited papers.

Interactive imitation with first-step mixtures. Li and Zhang [18, Algorithm 2 and Theorem 4] study finite-class realizable interactive imitation learning with trajectory-wise annotation, using first-step mixtures: a base policy is drawn at the start of an episode and kept fixed. Their Lemma 12 partitions trajectories by the first disagreement. Their learner takes the finite policy class itself as input and runs an online learner over the probability simplex on it. An absorbing-state specialization turns this into a legal stopped-feedback baseline for a supplied finite evaluable class in the attained-realizable setting, or under equivalent access to updates and sampling over the whole class. (Finiteness alone does not give this access under Assumption 2.5, whose empty-list call returns a single representative.) Represent a truthful record as a trajectory, move to an uninformative absorbing state with a known action after the first mismatch, and let the trajectory annotation reveal the same informative prefix as our display; the tail after absorption is known padding, not an actual query after an error. A first-step draw commits to a member of the class before the trajectory, which is exactly a legal proper pilot. Sampling components of the returned mixture and validating each on independent resets in which it is its own pilot yields a single proper output with small discrepancy probability. Since the optimum is then known to be zero, the interval $[ 0 , \varepsilon ]$ is a joint certificate for this output in the natural-loss specialization. What this baseline does not provide is the subject of Theorem 3.1: classes of finite VC dimension accessed only through Assumption 2.5, a general fixed loss with possibly unattained and nonzero infima, a certificate for an unknown optimum, and counting of all distinct functions played during acquisition.

Learning verifiers from annotated traces. Balcan et al. [6, Definition C.1, Theorem C.2 and the VC extension that follows it] already give proper agnostic learning of verifiers with coeficient one from complete annotated traces, together with an extension to classes of finite VC dimension; in the published version these are Definition F.1 and Theorem F.2 [5]. Properness, agnostic competition and finite VC dimension are therefore not distinguishing features of the present work. Our additional obligation is to acquire enough information when the chosen pilot’s earlier discrepancy censors the rest of the trace, including its terminal information. An empirical minimizer over complete traces does not implement this acquisition under Assumption 2.5, because the traces are not supplied.

Proper policy search with resets. Mansour [19, Sections 2.1–2.2, Theorem 4.2 and Figure 2] gives near-best policy search over a restricted class of strategies, in a partially observable environment that is accessed only by actions and resets to the initial state, assuming a proper (in his terms, realizable) mistake-bounded online learner for the class. Every generated trajectory follows one fixed strategy of the class, and the output competes with the best expected bounded return in the class. Making the first discrepancy absorbing, paying the reward $1 - \lambda ( T , \tau _ { h } )$ once at the observed stopping outcome (a discrepancy, or survival, including an empty record) and padding the rest with known steps of reward zero, we obtain from it a legal stopped-record special case with frozen proper pilots and agnostic bounded-loss competition: the expected return of h is $1 - R ( h )$ , and neither a hidden sufix nor a terminal descriptor after a discrepancy is used. The mistake-bound assumption is stronger than finite VC dimension: thresholds on the real line have VC dimension one, but an adversary that chooses each new query between the remaining candidate thresholds forces a mistake in every round. The restriction operations of the online learner are not supplied by Assumption 2.5, and the running time stated there is exponential in the mistake bound. That result does not give the full statement of Theorem 3.1: classes of finite VC dimension accessed only through Assumption 2.5, with the simultaneous localized reset and distinct-pilot orders together with the joint certificate.

Stopping-rank covers. Goel et al. [14, Lemma 3.1, Algorithms 1 and 2, Proposition 3.3] give rank and exchangeability covers, a minimax formulation and a no-regret game algorithm for selective imitation under dynamics shift; their rank argument in Lemma B.2 is an important antecedent for reasoning about stopping positions. Their game is trained and evaluated on available complete trajectories. The no-regret subroutine (their Algorithm 2) returns a distribution over validator sets of bounded size, and the final selector (their Algorithm 1) uses the union of logarithmically many sets sampled from it. These per-set and final-selector size bounds are not bounds on all hypotheses used while a stopped-data cover is acquired. Our acquisition and propagation arguments address that diference; we do not claim to have invented rank covers or the no-regret construction.

Sparse policy covers and a legal empirical specialization. Amortila et al. [4, Algorithm 1, Eq. (7), Proposition 4.1, Theorem 4.1 and Appendix H.1] provide a sparse policy-cover planning engine. Their planner can be applied to a known empirical one-step model built from legally displayed observations and observable nonnegative weights: the sample indices become the states, the two correctness outcomes become the actions, and the pattern oracle performs the policy optimizations. This reduction, followed by uniform relative concentration, recovers the rate powers of horizon one and of suitable horizon-two cases that are supplied with an evaluable finite list of proper pilots with pointwise first-query coverage, for example a list that realizes both labels at every possible first query. That the population record law is unknown is no obstacle, because the empirical model is known. This specialization is our derivation from their planning theorem; it does not by itself supply the truncated-rank acquisition and the growing-horizon composition used here. Inverse-propensity certificates, policy covers, and discovery followed by evaluation have a substantial literature. PC-PG averages the visitation distributions of earlier policies [2], but it uses rollout and continuation access and a critic transfer assumption that difer from our frozen original-proper pilots. POPLER and its extension to infinite policy classes [17, Appendix G.4] also precede discover-then-evaluate strategies; their sunflower assumptions and the switching between reacher and core policies within an episode are not available here.

Proper online learning with unknown base measure. Representatives of historical patterns and proper learning without knowledge of the law are also known. R-Cover is a proper algorithm for classes of finite VC dimension that commits to a function before each new query and does not know the base measure [7, Theorem 4 and Algorithm 1]; the later version, with a new title, states the classification result as Theorem 2 [8]. Its feedback is revealed query by query, whereas here one pilot is frozen for an entire stopped record. Our law-dependent nets are used only in the analysis, while the dictionaries used by the learner are acquired on displayed queries.

Classical tools. Given the legal gain and the observed dictionary, the reset and support orders of one acquisition stage follow from the standard exponential-weights potential, bounded martingale concentration and one sparsification [13, 15]; we credit this whole conversion, not only the form of the update. The propensities and mixtures are related to exploration in contextual bandits [11]. Agarwal et al. [1, Section 3.6, with Sections 2.4 and 3.4] also bound the total number of policies that ever receive positive weight in their warm-start algorithm, a form of accounting over the whole run rather than only the final support. That count difers from the resource V counted here: positive weight is not the same as actual use as a pilot, their action distributions add uniform mass on individual actions, which a frozen proper pilot need not be able to implement, and their objective is regret. The stage graph is a classical depth-robust graph [3, 9, 12]. The relevant lemmas of Alwen, Blocki and Pietrzak are already in their 2017 preprint, and the explicit construction of Blocki, Cinkoske, Lee and Son was public in 2021, before its 2022 proceedings version. Theorem 9 of the latter paper is credited there to the argument of Erdős, Graham and Szemerédi, applied to the explicit bipartite expanders constructed in that paper. Apple tasting [16], DAgger [20] and expert intervention learning [21], which includes Boolean stop signals as feedback, show that learning from selective feedback and from interventions is not a new paradigm; their action choices, annotation rules and timing within a run must be checked before a theorem is transferred to the present experiment.

Summary. The comparisons above concern the stated observation and oracle interfaces; none of the works discussed, and none of the reductions to them described here, gives the complete statement of Theorem 3.1. The contribution is not a new optimization, concentration or graph primitive, but the joint stopped-data theorem: legal acquisition at truncated ranks, transfer to the whole class, propagation of visibility under correlated failures, harmonic recovery of the mixture weight, and the two simultaneous resource caps together with the joint certificate.

## 9 Limitations and open questions

Computation and constants. The learner calls an oracle that lists one representative for every label pattern of H on the displayed queries. For a general class this may require exponential time, and we prove no bound on running time, memory or the precision of the real-number computations. The suficient constants are large. For example, Procedure 2 sizes every design with the predetermined cap $k _ { A }$ of (6.1), not with the realized indegrees, and this cap equals the complete-graph value $S - 1$ at all horizons below a few thousand. Evaluating (6.1) with $S = 4 D$ , the first strict crossing of this predetermined budget, that is, the first horizon at which $k _ { A } < S - 1$ , is $D = 5 0 8 6$ when $\beta = 1 0 ^ { - 2 }$ and $D = 7 8 0 2$ when $\beta = 1 0 ^ { - 9 }$ ; for every $\beta \in [ 1 0 ^ { - 9 } , 1 0 ^ { - 2 } ]$ the first crossing lies between these two values, because at each horizon $k _ { A }$ is nonincreasing in β. The strict inequality need not persist: for $\beta = 1 0 ^ { - 9 }$ the cap equals $S - 1$ again for $8 1 9 2 \leq D \leq 8 3 0 9$ , where the factor $\lfloor \log _ { 2 } ( 2 S ) \rfloor + 1$ increases. Wherever the cap equals $S - 1$ , in particular below the first crossing, the design budget in (6.13) grows like $D ^ { 2 }$ up to logarithmic factors, so the linear dependence on D in Theorem 3.1 is asymptotic. These statements concern the budget charged by the procedure; a cap equal to $S - 1$ does not mean that the random graph of Lemma 6.1 is complete. Using the complete graph for every D gives a deterministic stage graph, the reset bound $\widetilde { \cal O } ( ( d + 1 ) [ D ^ { 2 } ( f + \varepsilon ) + r ] / \varepsilon ^ { 2 } )$ and the same pilot bound (Remark D.4).

The model. Three assumptions are substantive. First, all runs use one common record law, which does not depend on the pilot. If the pilot could change the distribution of the rest of a record, the identities that evaluate several hypotheses on one reference record, for instance $\operatorname* { P r } ( \tau _ { g } \geq l \mid T ) = Q _ { L } ( T , l )$ , can fail. Second, the true labels are exact and fixed. A noisy checker that stops a run at a correct decision can invalidate the conditional-mean identity of the displayed gain and the moment and bias bounds of the final score. Third, the pilot is frozen for the whole run, and there are no continuations after an error and no resets to an arbitrary prefix. Extending the results beyond these assumptions requires new assumptions and new proofs. Nothing here is a statement about deployed systems or empirical performance.

## Open questions.

(1) Are there computationally eficient learners with the same two resource orders for structured classes, for instance thresholds or other classes with eficient consistency oracles?

(2) Which weakening of the exact-truth and common-law assumptions still permits joint guarantees of the same type?

(3) The lower bounds are coordinatewise at fixed confidence (Remark 3.7). What is the optimal dependence of the two resources on δ, and what is the full set of achievable pairs (N, V) on a common instance? The learners display at most DN labels; is this upper bound on the number of displayed labels optimal?

(4) Is there an explicit stage graph with small constants, so that the linear dependence on D appears already at moderate horizons?

## Declarations

Contributions and use of AI: AI tools assisted with mathematical exploration, proof development, computational checks and manuscript preparation. The author takes responsibility for the final content.

## References

[1] Alekh Agarwal, Daniel Hsu, Satyen Kale, John Langford, Lihong Li, and Robert E. Schapire. Taming the monster: A fast and simple algorithm for contextual bandits. In Proceedings of the 31st International Conference on Machine Learning (ICML 2014), volume 32 of Proceedings of Machine Learning Research, pages 1638–1646, 2014. URL https://arxiv.org/pdf/1402. 0555v2. Cited locators refer to arXiv:1402.0555v2, 14 October 2014.

[2] Alekh Agarwal, Mikael Henaf, Sham Kakade, and Wen Sun. PC-PG: Policy cover directed exploration for provable policy gradient learning. arXiv:2007.08459v1, 16 July 2020. URL https://arxiv.org/pdf/2007.08459v1.

[3] Joël Alwen, Jeremiah Blocki, and Krzysztof Pietrzak. Sustained space complexity. arXiv:1705.05313v2, 7 July 2017. URL https://arxiv.org/pdf/1705.05313v2. Version 1: 15 May 2017.

[4] Philip Amortila, Dylan J. Foster, and Akshay Krishnamurthy. Scalable online exploration via coverability. arXiv:2403.06571v2, 4 June 2024. URL https://arxiv.org/pdf/2403.06571v2. Version 1: 11 March 2024.

[5] Maria-Florina Balcan, Avrim Blum, Zhiyuan Li, and Dravyansh Sharma. On learning verifiers and implications to chain-of-thought reasoning. In 39th Conference on Neural Information Processing Systems (NeurIPS 2025), 2025. URL https://proceedings.neurips.cc/paper\_ files/paper/2025/file/e33b9cc87d3833f6769279e03011cdc1-Paper-Conference.pdf.

[6] Maria-Florina Balcan, Avrim Blum, Zhiyuan Li, and Dravyansh Sharma. On learning verifiers for chain-of-thought reasoning. arXiv:2505.22650v1, 28 May 2025. URL https://arxiv.org/ pdf/2505.22650v1.

[7] Moïse Blanchard. Agnostic smoothed online learning. arXiv:2410.05124v1, 7 October 2024. URL https://arxiv.org/pdf/2410.05124v1.

[8] Moïse Blanchard. Agnostic smoothed online learning without knowledge of the base measure. arXiv:2410.05124v3, 12 June 2025. URL https://arxiv.org/pdf/2410.05124v3.

[9] Jeremiah Blocki, Mike Cinkoske, Seunghoon Lee, and Jin Young Son. On explicit constructions of extremely depth robust graphs. In 39th International Symposium on Theoretical Aspects of Computer Science (STACS 2022), volume 219 of LIPIcs, pages 14:1–14:11, 2022. URL https://drops.dagstuhl.de/storage/00lipics/lipics-vol219-stacs2022/ LIPIcs.STACS.2022.14/LIPIcs.STACS.2022.14.pdf. Preprint arXiv:2110.04190v1, 8 October 2021.

[10] Mark Braverman, Roi Livni, Yishay Mansour, Shay Moran, and Kobbi Nissim. Learning from equivalence queries, revisited. In Proceedings of the 39th Annual Conference on Learning Theory, volume 336 of Proceedings of Machine Learning Research, 2026. URL https://proceedings. mlr.press/v336/braverman26a.html.

[11] Miroslav Dudík, Daniel Hsu, Satyen Kale, Nikos Karampatziakis, John Langford, Lev Reyzin, and Tong Zhang. Eficient optimal learning for contextual bandits. arXiv:1106.2369v1, 13 June 2011. URL https://arxiv.org/pdf/1106.2369v1.

[12] Paul Erdős, Ronald L. Graham, and Endre Szemerédi. On sparse graphs with dense long paths. Computers & Mathematics with Applications, 1:365–369, 1975. URL https://static.renyi. hu/\~p\_erdos/1976-26.pdf. Received April 1975.

[13] Yoav Freund and Robert E. Schapire. A decision-theoretic generalization of on-line learning and an application to boosting. Journal of Computer and System Sciences, 55(1):119–139, 1997. URL https://www.schapire.net/papers/FreundSc95.pdf. Cited locators refer to the author version dated 19 December 1996.

[14] Surbhi Goel, Jonathan Pei, and James Wang. Learning when to stop: Selective imitation learning under arbitrary dynamics shift. arXiv:2605.09183v2, 16 May 2026. URL https: //arxiv.org/pdf/2605.09183v2.

[15] David P. Helmbold, Robert E. Schapire, Yoram Singer, and Manfred K. Warmuth. On-line portfolio selection using multiplicative updates. Mathematical Finance, 8(4):325–347, 1998. URL https://www.schapire.net/papers/HelmboldScSiWa98.pdf.

[16] David P. Helmbold, Nicholas Littlestone, and Philip M. Long. Apple tasting. Information and Computation, 161:85–139, 2000. URL https://users.soe.ucsc.edu/\~dph/mypubs/ AppleTasting.pdf. Conference version at FOCS 1992.

[17] Zeyu Jia, Gene Li, Alexander Rakhlin, Ayush Sekhari, and Nathan Srebro. When is agnostic reinforcement learning statistically tractable? arXiv:2310.06113v1, 9 October 2023. URL https://arxiv.org/pdf/2310.06113v1.

[18] Yichen Li and Chicheng Zhang. A note on sample complexity of interactive imitation learning with log loss. arXiv:2412.07057v1, 9 December 2024. URL https://arxiv.org/pdf/2412. 07057v1.

[19] Yishay Mansour. Reinforcement learning and mistake bounded algorithms. In Proceedings of the 12th Annual Conference on Computational Learning Theory (COLT 1999), pages 183–192, 1999. Cited locators refer to the author manuscript dated 1 March 1999.

[20] Stéphane Ross, Geofrey J. Gordon, and J. Andrew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the 14th International Conference on Artificial Intelligence and Statistics (AISTATS 2011), volume 15 of Proceedings of Machine Learning Research, pages 627–635, 2011. URL https://proceedings.mlr.press/ v15/ross11a/ross11a.pdf.

[21] Jonathan Spencer, Sanjiban Choudhury, Matthew Barnes, Matthew Schmittle, Mung Chiang, Peter Ramadge, and Sidd Srinivasa. Expert intervention learning: An online framework for robot learning from explicit and implicit human feedback. Autonomous Robots, 2022. URL https://jspencer.org/data/spencer2022expert.pdf. Published online 19 October 2021; doi:10.1007/s10514-021-10006-9.

## A VC tools for latent records

In this appendix a record $T = ( ( z _ { j } , y _ { j } ) _ { j = 1 } ^ { M } , A ) , 0 \leq M \leq D _ { 1 }$ , is a complete probability-space object; the learner sees only its displays. Write $\mathcal { P }$ for the record law. For $h , h ^ { \prime } \in { \mathcal { H } }$ let

$$
E ( h , h ^ { \prime } ) = \{ T : ~ h ( z _ { j } ) \neq h ^ { \prime } ( z _ { j } ) ~ \mathrm { f o r ~ s o m e } ~ j \leq M \}
$$

be the event that h and $h ^ { \prime }$ disagree somewhere on the record. Assumption 2.6 makes all events below measurable.

## A.1 Trace counts

Lemma A.1 (Trace counts). Let H have VC dimension at most $d \geq 0$

(i) On any list of q queries, H realizes at most $( q + 1 ) ^ { d }$ label patterns.

(ii) Consider a Boolean test of a pair $( h , h ^ { \prime } )$ on an observation which is determined by the values of h and $h ^ { \prime }$ at the at most D queries of that observation, together with fixed data of the observation that do not depend on $( h , h ^ { \prime } )$ . On n observations, the family of such tests over all pairs realizes at most $( n D + 1 ) ^ { 2 d }$ distinct indicator vectors. This applies to the events $E ( h , h ^ { \prime } )$ on records, and to the events $O _ { h h ^ { \prime } }$ of Lemma $5 . 1 ( d )$ on displays.

Proof. (i) Let P be the set of patterns of H on distinct points $x _ { 1 } , \ldots , x _ { q }$ (repeated queries only reduce the count), and write $\begin{array} { r } { \phi ( \boldsymbol { q } , \boldsymbol { d } ) = \sum _ { j = 0 } ^ { \operatorname* { m i n } ( d , \boldsymbol { q } ) } \binom { q } { j } } \end{array}$ . We show $| \mathfrak { P } | \le \phi ( q , d )$ for every set $\mathfrak { P }$ of patterns that shatters no set of size $d + 1$ , by induction on $q + d .$ . If $q = 0$ there is at most one pattern. If $d = 0$ , no point is shattered, so all patterns agree at every point and there is at most one pattern. Otherwise let $\mathfrak { P ^ { \prime } }$ be the set of restrictions to $x _ { 1 } , \ldots , x _ { q - 1 }$ , and $\mathfrak { P } ^ { \prime \prime } \subseteq \mathfrak { P } ^ { \prime }$ the set of restrictions that have both extensions at $x _ { q }$ in $\mathfrak { P }$ . Then $| \mathfrak { P } | = | \mathfrak { P } ^ { \prime } | + | \mathfrak { P } ^ { \prime \prime } |$ . The set $\mathfrak { P ^ { \prime } }$ shatters no set of size $d + 1$ . If $\mathfrak { P } ^ { \prime \prime }$ shattered a set $Y .$ , then P would shatter $Y \cup \{ x _ { q } \}$ , so $\mathfrak { P } ^ { \prime \prime }$ shatters no set of size d. By induction $| \mathfrak { P } | \le \phi ( q - 1 , d ) + \phi ( q - 1 , d - 1 ) = \phi ( q , d )$ . Finally, for $d \geq 1$ the map sending a subset $\{ i _ { 1 } < \cdot \cdot \cdot < i _ { k } \} \subseteq \{ 1 , \cdot \cdot \cdot , q \}$ with $k \leq d$ to $( i _ { 1 } , \dots , i _ { k } , 0 , \dots , 0 ) \in \{ 0 , \dots , q \} ^ { d }$ is injective, so $\begin{array} { r } { \phi ( q , d ) \leq ( q + 1 ) ^ { d } ; } \end{array}$ for $d = 0$ both sides equal one.

(ii) The n observations contain at most $n D$ query positions. The indicator vector of the test for $( h , h ^ { \prime } )$ is determined by the pattern of h and the pattern of $h ^ { \prime }$ on the list of these queries, and by (i) there are at most $( ( n D + 1 ) ^ { d } ) ^ { 2 }$ such pairs of patterns. □

## A.2 Missed sets and a latent net

Lemma A.2 (Uniform missed-set bound). Let A be a class of events on an observation space such that on any m observations it realizes at most $G ( m )$ indicator vectors. Let $X _ { 1 } , \ldots , X _ { n }$ be independent with common law $\mu ,$ and let $0 < \chi \leq 1$ with $n \chi \geq 8$ . Then

$$
\operatorname* { P r } \{ \exists A \in { \mathcal A } : \ \mu ( A ) > \chi , \ X _ { i } \not \in A \ ( 1 \leq i \leq n ) \} \leq 2 G ( 2 n ) 2 ^ { - n \chi / 2 } \leq 2 G ( 2 n ) e ^ { - n \chi / 4 } .\tag{A.1}
$$

The same bound holds conditionally on a previously frozen history, if the fresh observations are conditionally independent with a common conditional law and the bound G is deterministic.

Proof. Let E be the event in $( \mathrm { A . 1 } )$ . On E select a witness $A ^ { * } \in { \mathcal { A } }$ with $\mu ( A ^ { * } ) > \chi$ and no $X _ { i }$ in $A ^ { * }$ . Let $X _ { 1 } ^ { \prime } , \ldots , X _ { n } ^ { \prime }$ be an independent copy of the sample. Conditionally on $X$ , the number $Z ^ { \prime }$ of i with $X _ { i } ^ { \prime } \in A ^ { * }$ is binomial with mean np, $p = \mu ( A ^ { * } ) > \chi$ , and variance at most $n p$ . By Chebyshev’s inequality,

$$
\operatorname* { P r } ( Z ^ { \prime } < n p / 2 \mid X ) \leq { \frac { n p } { ( n p / 2 ) ^ { 2 } } } = { \frac { 4 } { n p } } \leq { \frac { 4 } { n \chi } } \leq { \frac { 1 } { 2 } } .
$$

Hence $\mathrm { P r } ( \mathsf { E } ) \leq 2 \mathrm { P r } ( \mathsf { E } ^ { \prime } )$ , where $\mathsf { E } ^ { \prime }$ is the event that some $A \in { \mathcal { A } }$ contains none of the $X _ { i }$ and at least $n \chi / 2$ of the $X _ { i } ^ { \prime } .$ Now exchange $X _ { i }$ and $X _ { i } ^ { \prime }$ independently for each i with probability $1 / 2 \colon$ this does not change the joint law. Condition on the unordered pairs $\{ X _ { i } , X _ { i } ^ { \prime } \}$ . There are at most $G ( 2 n )$ indicator vectors of events of A on the 2n pooled observations. Fix one with at least $n \chi / 2$ hits. If some pair contains two hits, the first sample cannot avoid the event. Otherwise the hits lie in distinct pairs, and all of them land in the second sample with probability at most $2 ^ { - n \chi / 2 }$ A union bound over indicator vectors gives $\operatorname* { P r } ( \mathrm { E } ^ { \prime } \mid \mathrm { p a i r s } ) \leq G ( 2 n ) 2 ^ { - n \chi / 2 }$ , and the first inequality follows. The second uses $( \log 2 ) / 2 \geq 1 / 4$ . The argument applies verbatim conditionally on a frozen history. □

For $0 < \chi \leq 1$ let $\nu ( \chi )$ be the least positive integer v with

$$
v \chi \geq 8 \qquad \mathrm { a n d } \qquad 2 ( 2 v D + 1 ) ^ { 2 d } e ^ { - v \chi / 4 } < 1 ,\tag{A.2}
$$

and put $H _ { \chi } = ( \nu ( \chi ) D + 1 ) ^ { d }$ . With $\chi = 1 _ { / }$ n these are the numbers $\nu _ { n }$ and $H _ { n }$ of (6.10).

Lemma A.3 (A latent-record net). For every $0 < \chi \leq 1$ there is a finite set $\mathcal { N } _ { \chi } \subseteq \mathcal { H }$ , possibly depending on the unknown law, with $| \mathcal { N } _ { \chi } | \leq H _ { \chi }$ , such that every $h \in \mathcal H$ has some $h ^ { \prime } \in \mathcal { N } _ { \chi }$ with $\mathcal { P } ( E ( h , h ^ { \prime } ) ) \leq \chi$ . Moreover

$$
\nu ( \chi ) \leq \Big \lceil \frac { 8 } { \chi } \operatorname* { m a x } \Big \{ 1 , \ \log 4 + 2 d \log \frac { 4 8 d D } { \chi } \Big \} \Big \rceil \quad ( d \geq 1 ) , \qquad \nu ( \chi ) = \lceil 8 / \chi \rceil \quad ( d = 0 ) ,
$$

so that $\nu ( \chi ) = O ( ( d + 1 ) \chi ^ { - 1 } \log ( e ( d + 1 ) D / \chi ) )$ and log $H _ { \chi } = O ( d \log ( e ( d + 1 ) D / \chi ) )$ .

Proof. Let $v = \nu ( \chi )$ and apply Lemma A.2 to the events $E ( h , h ^ { \prime } )$ on v independent records, with $G ( m ) = ( m D + 1 ) ^ { 2 d }$ from Lemma A.1(ii). Since the right side of $( \mathrm { A . 1 } )$ is less than one, there is a realization $T _ { 1 } , \ldots , T _ { v }$ such that every pair $( h , h ^ { \prime } )$ that agrees at all queries of these records has $\mathcal { P } ( E ( h , h ^ { \prime } ) ) \leq \chi$ . Fix such a realization. Let $\mathcal { N } _ { \chi }$ contain one member of H for each pattern of H on the at most vD queries of $T _ { 1 } , \dots , T _ { v } ;$ by Lemma A.1(i), $| \mathcal { N } _ { \chi } | \leq ( v D + 1 ) ^ { d }$ . Every h shares its pattern with some $h ^ { \prime } \in \mathcal { N } _ { \chi }$ , and this pair agrees at all sampled queries, so $\mathcal { P } ( E ( h , h ^ { \prime } ) ) \leq \chi$ . This is an existence argument used only in proofs; the learner never constructs $\mathcal { N } _ { \chi }$ and makes no oracle call on latent queries.

For the bound on $\nu ( \chi )$ , let $d \geq 1$ . Since $2 v D + 1 \leq 3 v D$ , the second condition in (A.2) holds if $v \chi / 4 \geq \log 4 + 2 d \log ( 3 D v )$ , that is, if $v \geq ( 4 / \chi ) ( \log 4 + 2 d \log ( 3 D v ) )$ . By Lemma A.6 with $A = 4 / \chi .$ $B = \log 4 , k = 2 d$ and $c = 3 D$ , this holds for every $v \geq ( 8 / \chi ) ( \log 4 + 2 d \log ( 4 8 d D / \chi ) )$ . Together with $v \geq 8 / \chi$ this gives the bound. For $d = 0$ the second condition reads $v \chi > 4 \log 2$ , which follows from $v \chi \geq 8 .$ The order statements follow by substitution. □

## A.3 Rare counts

Lemma A.4 (Uniform rare-event counts). Under the trace assumption of Lemma A.2, let $X _ { 1 } , \ldots , X _ { N }$ be independent with law $\mu ,$ , let $\chi \geq 0$ and $\kappa \ge \operatorname* { m a x } \{ 4 , 4 N \chi \}$ . Then

$$
\operatorname* { P r } \Bigl \{ \exists A \in A : \ \mu ( A ) \le \chi , \ \sum _ { i = 1 } ^ { N } \mathbf { 1 } _ { A } ( X _ { i } ) > \kappa \Bigr \} \le 2 G ( 2 N ) e ^ { - \kappa / 1 8 } .
$$

In particular, for the events $E ( h , h ^ { \prime } )$ of mass at most $1 / N$ on N independent records, all counts are at most

$$
\kappa _ { N } = \operatorname* { m a x } \Bigl \{ 4 , ~ 1 8 \log \frac { 4 ( 2 N D + 1 ) ^ { 2 d } } { \beta } \Bigr \}
$$

except on an event of probability at most $\beta / 2$

Proof. Let E be the event in question and on E select a witness $A ^ { * }$ with $\mu ( A ^ { * } ) \leq \chi$ and count $O > \kappa$ in the sample. For an independent copy of the sample, the count $O ^ { \prime }$ of $A ^ { * }$ has conditional mean at most $N \chi \leq \kappa / 4$ , so by Markov’s inequality $O ^ { \prime } \le \kappa / 2$ with conditional probability at least $1 / 2$ Hence $\operatorname* { P r } ( \mathsf { E } )$ is at most twice the probability that some $A \in { \mathcal { A } }$ has $O _ { A } > \kappa$ and $O _ { A } ^ { \prime } \le \kappa / 2$ . On this event put ${ \cal L } _ { A } = { \cal O } _ { A } + { \cal O } _ { A } ^ { \prime } ;$ then $L _ { A } > \kappa$ and, since $O _ { A } ^ { \prime } < O _ { A } / 2$ , also $O _ { A } - O _ { A } ^ { \prime } > L _ { A } / 3$ . Exchange the members of each pair independently as before, and condition on the unordered pairs. For a fixed indicator vector on the pooled sample, $L _ { A }$ is fixed and $\begin{array} { r } { O _ { A } - O _ { A } ^ { \prime } = \sum _ { i \in I } \epsilon _ { i } } \end{array}$ , where I is the set of pairs with exactly one hit, $\vert I \vert \le L _ { A }$ , and the $\epsilon _ { i }$ are independent uniform signs. Since cosh $x \leq e ^ { x ^ { 2 } / 2 }$ the exponential Markov inequality with parameter $s / | I |$ gives $\mathrm { P r } ( \sum _ { i \in I } \epsilon _ { i } > s ) \le e ^ { - s ^ { 2 } / ( 2 | I | ) }$ , so with $s = L _ { A } / 3$ the conditional probability is at most $e ^ { - L _ { A } / 1 8 } \leq e ^ { - \kappa / 1 8 }$ . (If I is empty the sum is zero and the probability vanishes.) A union bound over the at most G(2N) indicator vectors and the factor two prove the claim. For the last assertion take $\chi = 1 / N$ , so that $4 N \chi = 4 \AA$ , and note $2 ( 2 N D + 1 ) ^ { 2 d } e ^ { - \kappa _ { N } / 1 8 } \leq \beta / 2$ □

Remark A.5 (Transfer and adaptively censored data). If h and h<sup>′</sup> agree at every query of a record, their stopping ranks and losses on it agree, including the terminal loss. Every score computed from the record, from frozen auxiliary functions and lotteries, and from the candidate only through its stopping rank, also agrees. Hence a score with range [0, W] difers between h and $h ^ { \prime }$ by at most $W \mathbf { 1 } _ { E ( h , h ^ { \prime } ) }$

Fresh records remain independent with law P when pilots are chosen from past observations, because the record law does not depend on the pilot. One may therefore draw each complete record independently, reveal only the pilot’s display, and pad a stopped procedure to its deterministic cap by arbitrary predictable pilots. Lemma A.4 then controls the number of score diferences on this independent array of latent records even when the displays are adaptive. It does not make adaptive scores independent; their concentration uses predictable conditional means, as in Appendices B and C.

## A.4 Two elementary tools

Lemma A.6 (Inverting logarithmic conditions). Let $A , k , c > 0$ and $B \geq 0$ . Every real m $> 0$ with $m \geq 2 A { \big ( } B + k \log ( 2 A c k ) { \big ) } s a t i s f i e s m \geq A { \big ( } B + k \log ( c m ) { \big ) }$

Proof. Since log $y \le y - 1$

$$
\log ( c m ) = \log ( 2 A c k ) + \log { \frac { m } { 2 A k } } \leq \log ( 2 A c k ) + { \frac { m } { 2 A k } } - 1 .
$$

Hence $A ( B + k \log ( c m ) ) \leq A B + A k \log ( 2 A c k ) + m / 2 - A k < A B + A k \log ( 2 A c k ) + m / 2 \leq m ,$ where the last step is the hypothesis divided by two. □

Lemma A.7 (Exponential bounds for adapted bounded variables). Let $( \mathcal { F } _ { t } ) _ { t \geq 0 }$ be a filtration and $Y _ { t } \in [ 0 , 1 ]$ be $\mathcal { F } _ { t }$ -measurable with $\mu _ { t } = \operatorname { \mathbb { E } } [ Y _ { t } \mid { \mathcal { F } } _ { t - 1 } ]$ . Let $\vartheta > 0$

$$
( a ) ~ \mathbb { E } [ \exp ( \vartheta Y _ { t } ) ~ \big | ~ \mathcal { F } _ { t - 1 } ] \leq \exp ( ( e ^ { \vartheta } - 1 ) \mu _ { t } ) ~ a n d ~ \mathbb { E } [ \exp ( - \vartheta Y _ { t } ) ~ \big | ~ \mathcal { F } _ { t - 1 } ] \leq \exp ( - ( 1 - e ^ { - \vartheta } ) \mu _ { t } ) .
$$

(b) For every $u \geq 0$ and every integer $n \geq 1$

$$
\operatorname* { P r } \Bigl \{ \exists s \leq n : \ \sum _ { t = 1 } ^ { s } \bigl ( \vartheta Y _ { t } - ( e ^ { \vartheta } - 1 ) \mu _ { t } \bigr ) \geq u \Bigr \} \leq e ^ { - u } ,
$$

and the same holds for $\begin{array} { r } { \sum _ { t < s } ( - \vartheta Y _ { t } + ( 1 - e ^ { - \vartheta } ) \mu _ { t } ) } \end{array}$

Proof. (a) By convexity, $e ^ { \vartheta y } \leq 1 + ( e ^ { \vartheta } - 1 ) y$ and $e ^ { - \vartheta y } \leq 1 - ( 1 - e ^ { - \vartheta } ) y$ for $y \in [ 0 , 1 ]$ . Take conditional expectations and use $\begin{array} { r } { 1 + x \leq e ^ { x } . \mathrm { ( b ) } \mathrm { ~ B y ~ } ( \mathrm { a } ) , M _ { s } = \exp { \sum _ { t < s } ( \vartheta Y _ { t } - ( e ^ { \vartheta } - 1 ) \mu _ { t } ) } } \end{array}$ is a nonnegative supermartingale with $M _ { 0 } = 1$ . Let σ be the first $s \leq n$ with $M _ { s } \geq e ^ { u }$ , and $\sigma = n$ if there is none. Optional stopping at the bounded time σ gives $\mathbb { E } M _ { \sigma } \le 1$ , and $M _ { \sigma } \geq e ^ { u }$ on the event in question, so its probability is at most $e ^ { - u }$ . The second sum is treated in the same way. □

## B Ownership and the coarse anchors

This appendix proves Lemmas 4.5 and 4.6. The basic tool is an ownership procedure for a general fixed loss $\lambda ;$ the anchors use it with the natural loss. Throughout, $\ell _ { h } = \lambda ( T , \tau _ { h } ) , R ( h ) = \mathbb { E } \ell _ { h }$ $R ^ { * } = \operatorname* { i n f } _ { \mathcal { H } } R$ and $F ( h ) = \operatorname* { P r } ( \tau _ { h } \leq M )$ . All oracle calls are made on displayed queries only.

## B.1 Anchored ownership

Let $a \in \mathcal H$ be fixed before the procedure, and let $r , w \in [ 0 , 1 ] , 0 < t < 1$ and $0 < \alpha < 1$ be numbers. Put

$$
\eta = \frac { t } { 3 2 } , \quad K = \operatorname* { m a x } \Bigl \{ 1 , \Bigl \lceil \frac { D w } { \eta } \Bigr \rceil \Bigr \} , \quad J = K + 1 , \quad \theta = \frac { \eta } { 8 ( r + \eta ) } \le \frac { 1 } { 8 } , \quad \chi = \frac { \eta } { 3 2 J } ,
$$

let $H _ { \chi }$ be as in Lemma $\mathrm { { A . 3 , } }$ and let B be the least positive integer with

$$
B \eta \geq 1 6 ,
$$

$$
\frac { \log ( 8 H _ { \chi } / \alpha ) } { B \theta } \leq \frac { \eta } { 4 } ,\tag{B.1}
$$

$$
2 ( 2 J B D + 1 ) ^ { 2 d } e ^ { - B \eta / 7 2 } \leq \frac { \alpha } { 4 } .\tag{B.2}
$$

These conditions are law-free; the net $\mathcal { N } _ { \chi }$ itself is not an input. For pilots $g _ { 1 } , g _ { 2 } , . . .$ . and a record T define the ownership score of h in batch j,

$$
Y _ { j , h } ( T ) = \ell _ { h } ( T ) \mathbf { 1 } \Bigl \{ \operatorname* { m a x } _ { i < j } \tau _ { g _ { i } } ( T ) < \tau _ { h } ( T ) \leq \tau _ { g _ { j } } ( T ) \Bigr \} ,\tag{B.3}
$$

where the maximum of an empty list is 0. If $T _ { j , 1 } , \dotsc , T _ { j , B }$ are the records of batch $j ,$ , put $S _ { m } ( h ) =$ $\begin{array} { r } { B ^ { - 1 } \sum _ { j \leq m } \sum _ { k \leq B } Y _ { j , h } ( T _ { j , k } ) } \end{array}$

Lemma B.1 (Anchored ownership). Procedure 4 returns, on every history, a member of H and an ordered interval in [0, 1] of width at most t. It uses at most J batches of B resets, at most K oracle calls and at most J distinct pilots, and

$$
B = { \cal O } \left( \frac { r + \eta } { \eta ^ { 2 } } \Big [ ( d + 1 ) \log \frac { e ( d + 1 ) D } { \eta } + \log \frac { e } { \alpha } \Big ] \right) .\tag{B.4}
$$

If $R ^ { * } \leq r$ and $F ( a ) \leq w$ , then with probability at least $1 - \alpha$ the interval contains $R ^ { * }$ and the risk of the returned function. The same holds conditionally on any earlier history that fixes $a , r ,$ w and satisfies these two inequalities.

Proof. Sample size. Condition (i) of (B.1) is $B \geq 1 6 / \eta$ , (ii) is $B \geq 3 2 ( r + \eta ) \log ( 8 H _ { \chi } / \alpha ) / \eta ^ { 2 }$ , and (iii) is $B \geq 8 ( r + \eta ) \log ( 8 K / \alpha ) / \eta ^ { 2 }$ . Since $2 J B D + 1 \leq 3 J D B$ , condition (B.2) holds if $B \eta / 7 2 \geq$ $\log ( 8 / \alpha ) + 2 d \log ( 3 J D B )$ , and by Lemma A.6 (with $A = 7 2 / \eta , B _ { 0 } = \log ( 8 / \alpha ) , k = 2 d , c = 3 J D )$ this holds for every $B \geq ( 1 4 4 / \eta ) ( \log ( 8 / \alpha ) + 2 d \log ( 8 6 4 d J D / \eta ) )$ ; for $d = 0$ the threshold is $( 7 2 / \eta ) \log ( 8 / \alpha )$ Hence B is at most one more than the largest of these four thresholds. Since $w \leq 1$ and $D / \eta \geq 3 2$ we have $J \le D / \eta + 2 \le 3 D / \eta$ and $K \leq J$ , so Lemma A.3 gives log $H _ { \chi } = O ( d \log ( e ( d + 1 ) D / \eta ) )$ Using $1 / \eta \le ( r + \eta ) / \eta ^ { 2 }$ , we obtain (B.4).

Procedure 4 Ownership $( a , r , w , t , \alpha )$   
1: compute $\overline { { \eta , K , J , \theta , \chi , B } }$ as above; $g _ { 1 }  a ;$ play $g _ { 1 }$ on a batch of B fresh resets   
2: for $m = 1 , \ldots , K$ do   
3: call the oracle on all queries displayed in batches $1 , \ldots , m ;$ ; let $h _ { m }$ be a returned member   
minimizing $S _ { m }$   
4: $g _ { m + 1 } \gets h _ { m } ;$ play $g _ { m + 1 }$ on a batch of B fresh resets; $\begin{array} { r } { \widetilde { R } _ { m } \gets B ^ { - 1 } \sum _ { k \leq B } \ell _ { h _ { m } } ( T _ { m + 1 , k } ) } \end{array}$   
5: $L _ { m } \gets ( S _ { m } ( h _ { m } ) - \eta ) / ( 1 + \theta ) , U _ { m } \gets ( \widetilde { R } _ { m } + \eta ) / ( 1 - \theta )$ ; clip each to [0, 1] separately   
6: if the clipped pair is ordered and has width at most t then return $h _ { m }$ and the clipped   
interval   
7: ▷ otherwise batch $m + 1$ is kept as the next ownership batch   
8: return a and [0, 0] ▷ cap exhausted

Observability. Consider a record T of batch $j ,$ , displayed by $g _ { j }$ . If h disagrees with some displayed pair, its first disagreement q equals $\tau _ { h } .$ and $\ell _ { h }$ is determined by the pairs up to $q .$ The condition $\operatorname* { m a x } _ { i < j } \tau _ { g _ { i } } < q$ says that each earlier pilot disagrees with some pair strictly before $q ,$ which is decidable from the same display, and $\tau _ { h } = q \le n _ { g _ { j } } \le \tau _ { g _ { j } }$ . If h agrees with all displayed pairs and $g _ { j }$ errs, then $\tau _ { h } > \tau _ { g _ { j } }$ and the score is zero; nothing hidden is evaluated. If $g _ { j }$ survives, the whole record and its descriptor are displayed and the score is computable. On an empty record all functions survive at rank one, so only the first batch owns its loss; a discrepancy at the last actual query is treated as a discrepancy, not as survival. Consequently $S _ { m } ( h )$ depends on h only through its values at the queries displayed in batches $1 , \ldots , m$ , and a minimizer over the representatives returned by the oracle is an exact empirical minimizer over $\mathcal { H } ;$ at most $( m B D + 1 ) ^ { d }$ representatives are needed.

Totalization. If the procedure returns in the loop, the interval is ordered and has width at most t by the test. Otherwise it returns a and [0, 0]. In both cases the output is a member of H. An invalid endpoint pair is never accepted; the fallback [0, 0] is a convention for histories on which no certificate was found and carries no coverage claim.

Predictable means. Let $\mu _ { j } ( h )$ be the conditional mean of one score of batch j given everything before that batch. The pilots $g _ { 1 } , \dotsc , g _ { j }$ are then fixed and the record is fresh, so

$$
\mu _ { j } ( h ) = \mathbb { E } _ { T } \Bigl [ \ell _ { h } \mathbf { 1 } \Bigl \{ \operatorname* { m a x } _ { i < j } \tau _ { g _ { i } } < \tau _ { h } \leq \tau _ { g _ { j } } \Bigr \} \Bigr ] ,
$$

where $\mathbb { E } _ { T }$ integrates a separate reference record with the realized pilots held fixed. The same value is the conditional mean of every row of batch j given the earlier rows of that batch. Let $\bar { \tau } _ { m } ( T ) = \mathrm { m a x } _ { j \leq m } \tau _ { g _ { j } } ( T )$ and $\begin{array} { r } { C _ { m } ( h ) = \sum _ { j \leq m } \mu _ { j } ( h ) } \end{array}$ . On one reference record, the events in the indicators are disjoint for diferent $j ,$ and their union is $\{ \tau _ { h } \le \bar { \tau } _ { m } \}$ (the first $j$ with $\tau _ { g _ { j } } \geq \tau _ { h }$ owns the loss). Hence

$$
C _ { m } ( h ) = \mathbb { E } _ { T } \bigl [ \ell _ { h } { \mathbf { 1 } } \{ \tau _ { h } \leq \bar { \tau } _ { m } \} \bigr ] \leq R ( h ) .\tag{B.5}
$$

This is an identity of predictable compensators with the realized functions held fixed; it is not a statement about the conditional law of old records given the final list of pilots.

Uniform martingale bound. For the analysis, continue the procedure after it returns by playing arbitrary predictable pilots until J batches have been used; this does not change any event concerning the actual run. Fix $h ^ { \prime } \in \mathcal { N } _ { \chi }$ . The rows $Y _ { j , h ^ { \prime } } ( T _ { j , k } )$ , taken in order, lie in [0, 1] and have conditional means $\mu _ { j } ( h ^ { \prime } )$ . Apply Lemma $\mathrm { A . 7 ( b ) }$ with $\vartheta = \theta , u = \log ( 8 H _ { \chi } / \alpha )$ , to both signs. Since $( e ^ { \theta } - 1 ) / \theta \leq$

$1 + \theta$ and $( 1 - e ^ { - \theta } ) / \theta \geq 1 - \theta$ for $0 < \theta \leq 1$ , at every batch boundary $m \leq J ,$

$$
| S _ { m } ( h ^ { \prime } ) - C _ { m } ( h ^ { \prime } ) | \leq \theta C _ { m } ( h ^ { \prime } ) + \frac { \log ( 8 H _ { \chi } / \alpha ) } { B \theta } \leq \theta C _ { m } ( h ^ { \prime } ) + \frac { \eta } { 4 } ,\tag{B.6}
$$

except with probability $2 e ^ { - u }$ . A union bound over $h ^ { \prime } \in \mathcal { N } _ { \chi }$ gives failure at most $2 H _ { \chi } e ^ { - u } = \alpha / 4$ There is no factor for the number of batches.

Transfer to the class. Apply Lemma A.4 to the $N = J B$ independent latent records of the padded run, with mass χ and threshold $\kappa = B \eta / 4$ . Here $\kappa \geq 4$ and $4 N \chi = B \eta / 8 \le \kappa$ , and by (B.2), outside probability $\alpha / 4 ,$ every event $E ( h , h ^ { \prime } )$ of mass at most $\chi$ contains at most κ of these records. For $h \in \mathcal H$ choose $h ^ { \prime } \in \mathcal { N } _ { \chi }$ with $\mathcal { P } ( E ( h , h ^ { \prime } ) ) \leq \chi$ . By Remark A.5, the two ownership scores can difer only on records in $E ( h , h ^ { \prime } )$ , whatever the pilots, and each diference is at most one, so $| S _ { m } ( h ) - S _ { m } ( h ^ { \prime } ) | \leq \kappa / B = \eta / 4$ . For the compensators, the integrands in (B.5) for h and $h ^ { \prime }$ agree outside $E ( h , h ^ { \prime } )$ and lie in [0, 1], so $| C _ { m } ( h ) - C _ { m } ( h ^ { \prime } ) | \leq \chi .$ , not $m \chi$ . Combining with (B.6) and $( 1 + \theta ) \chi \leq \eta / 2$

$$
| S _ { m } ( h ) - C _ { m } ( h ) | \leq \theta C _ { m } ( h ) + \eta \qquad \mathrm { f o r ~ e v e r y ~ } m \leq J \mathrm { ~ a n d ~ e v e r y ~ } h \in \mathcal { H } .\tag{B.7}
$$

Validation. Each candidate $h _ { m }$ is fixed before batch $m + 1$ , whose B losses $\ell _ { h _ { m } } ( T _ { m + 1 , k } )$ are then independent with mean $R ( h _ { m } )$ . Lemma $\mathrm { A . 7 ( b ) }$ at the final time with $\vartheta = \theta$ and $u = B \theta \eta$ , for both signs, gives

$$
| \widetilde { R } _ { m } - R ( h _ { m } ) | \leq \theta R ( h _ { m } ) + \eta\tag{B.8}
$$

except with conditional probability $2 e ^ { - B \theta \eta } \leq \alpha / ( 4 K )$ , by the last condition in (B.1). A union bound over the at most K tested candidates costs $\alpha / 4$ . Thus (B.7) and (B.8) hold together with probability at least $1 - 3 \alpha / 4$ . Neither event refers to which validations failed, so reusing a failed validation batch as the next ownership batch is legitimate.

Coverage. Work on this event. For the empirical minimizer $h _ { m } , ~ ( \mathrm { B } . 7 )$ and (B.5) give, for every $h , S _ { m } ( h _ { m } ) \leq S _ { m } ( h ) \leq ( 1 + \theta ) R ( h ) + \eta ;$ taking an infimizing sequence, $S _ { m } ( h _ { m } ) \leq ( 1 + \theta ) R ^ { * } + \eta$ so $L _ { m } \leq R ^ { * }$ . No population minimizer is needed. By (B.8), $U _ { m } \geq R ( h _ { m } )$ . Hence $L _ { m } \leq R ^ { * } \leq$ $R ( h _ { m } ) \leq U _ { m }$ . Separate clipping to [0, 1] preserves this chain and does not increase the width, so every accepted interval covers.

Acceptance within the cap. Assume now $R ^ { * } \leq r$ and write $C = C _ { m } ( h _ { m } ) , R = R ( h _ { m } )$ . The lower half of (B.7) and the bound on $S _ { m } ( h _ { m } )$ give $( 1 - \theta ) C - \eta \leq ( 1 + \theta ) r + \eta ,$ , so

$$
C \leq \frac { ( 1 + \theta ) r + 2 \eta } { 1 - \theta } \leq r + 3 \eta ,
$$

where the last step is equivalent to $\theta ( 2 r + 3 \eta ) \leq \eta$ , which holds because $\theta = \eta / ( 8 ( r + \eta ) )$ . Suppose $R - C \leq \eta$ . Then $R \leq r + 4 \eta$ , and from (B.7), (B.8),

$$
C - L _ { m } \leq \frac { 2 \theta C + 2 \eta } { 1 + \theta } , \qquad U _ { m } - R \leq \frac { 2 \theta R + 2 \eta } { 1 - \theta } .
$$

Since $\theta C \le \theta ( r + 3 \eta ) \le 3 \eta / 8$ and $\theta R \leq \theta ( r + 4 \eta ) \leq \eta / 2$

$$
U _ { m } - L _ { m } = ( U _ { m } - R ) + ( R - C ) + ( C - L _ { m } ) \le \frac { 2 4 \eta } { 7 } + \eta + \frac { 1 1 \eta } { 4 } = \frac { 2 0 1 \eta } { 2 8 } < 8 \eta < t ,
$$

so the candidate is accepted. Consequently, on the event above, a rejected candidate satisfies $R ( h _ { m } ) - C _ { m } ( h _ { m } ) > \eta ,$

For a nonempty finite list G of functions put $\Psi ( G ) = \mathbb { E } _ { T } \operatorname* { m a x } _ { g \in G } \tau _ { g } .$ . Let $G _ { m } = \{ g _ { 1 } , \dots , g _ { m } \}$ Since ranks are integers and $\ell \leq 1$ , (B.5) gives for a rejected candidate

$$
\Psi ( G _ { m } \cup \{ h _ { m } \} ) - \Psi ( G _ { m } ) \geq \operatorname* { P r } _ { T } ( \tau _ { h _ { m } } > \bar { \tau } _ { m } ) \geq R ( h _ { m } ) - C _ { m } ( h _ { m } ) > \eta .\tag{B.9}
$$

Every list contains a. On a record on which a survives, max $\ O _ { G } \tau _ { g } = M + 1 = \tau _ { a } ,$ so

$$
0 \leq \Psi ( G ) - \Psi ( \{ a \} ) = \mathbb { E } _ { T } \Big [ \big ( \operatorname* { m a x } _ { g \in G } \tau _ { g } - \tau _ { a } \big ) \mathbf { 1 } \{ \tau _ { a } \leq M \} \Big ] \leq D F ( a ) \leq D w .\tag{B.10}
$$

If all K candidates were rejected, the potential would increase by more than $K \eta \ge D w$ , which is impossible. $( \operatorname { I f } w = 0$ , even one rejection is impossible.) Hence on the event above a candidate is accepted within the cap.

Accounting. The first batch, every rejected candidate’s batch and the accepted validation batch are actual resets and are all counted; there are at most J batches and J pilots. There are at most K oracle calls. Every argument holds after conditioning on an earlier history fixing $a , r , w$ , which gives the conditional statement. □

## B.2 The anchor with a supplied ceiling

Proof of Lemma $4 . 5 .$ Let $\bar { \varepsilon } = \operatorname* { m i n } ( 1 , f + \varepsilon )$ . If $\bar { \varepsilon } = 1$ , call the oracle on the empty list, return its member and $w = 1 ;$ no reset is used and $\varepsilon \le w = 1 \le 2 ( f + \varepsilon )$

Let $\bar { \varepsilon } < 1 , \mathrm { ~ s o ~ } \bar { \varepsilon } = f + \varepsilon .$ . Put $I = \lceil \log _ { 2 } ( 1 / \bar { \varepsilon } ) \rceil , t _ { 0 } = 1$ and $t _ { i } = \operatorname* { m a x } ( \bar { \varepsilon } , 2 ^ { - i } )$ for $1 \leq i \leq I$ Then $t _ { I } = \bar { \varepsilon } , t _ { i - 1 } \leq 2 t _ { i }$ , and $\begin{array} { r } { \sum _ { i = 1 } ^ { I } t _ { i } ^ { - 1 } \le 3 / \bar { \varepsilon } , } \end{array}$ , because $t _ { i } = 2 ^ { - i }$ for $i < I$ and $\begin{array} { r } { \sum _ { i < I } 2 ^ { i } < 2 ^ { I } < } \end{array}$ $2 / \bar { \varepsilon }$ . Start with a member $a _ { 0 }$ from an empty-list call and $U _ { 0 } = 1$ . At stage $i = 1 , \dots , I$ , run Ownership $( a _ { i - 1 } , U _ { i - 1 } , U _ { i - 1 } , t _ { i } , \alpha / I )$ for the natural loss. Let $a _ { i }$ and $[ L _ { i } , \widetilde { U } _ { i } ]$ be its output and put

$$
U _ { i } = \operatorname* { m i n } ( \widetilde { U } _ { i } , \ f + t _ { i } ) \in [ 0 , 1 ] .
$$

Return $a = a _ { I }$ and $w = \operatorname* { m a x } ( \varepsilon , \operatorname* { m i n } ( 1 , f + \bar { \varepsilon } ) )$

Costs on every history. At the first stage $U _ { 0 } = 1 \leq 2 t _ { 1 }$ . At a later stage, $U _ { i - 1 } \leq f + t _ { i - 1 } \leq$ $\bar { \varepsilon } + 2 t _ { i } \le 3 t _ { i }$ , because $f \leq \bar { \varepsilon } \leq t _ { i }$ . So the call at stage i has inputs $r = w = U _ { i - 1 } \leq \bar { u } _ { i } : = \operatorname* { m i n } ( 1 , 3 t _ { i } )$ These inputs depend on the history, but $K \leq \lceil D \bar { u } _ { i } / \eta \rceil \leq 9 6 D$ , so the call has at most $9 6 D + 1$ batches and pilots. By the proof of Lemma B.1, its batch size is at most one more than the largest of four thresholds. These thresholds are nondecreasing in r and in $w \colon \theta$ decreases in $r , K$ and J increase in $w ,$ and $\nu ( \chi )$ , hence $H _ { \chi }$ , increases as $\chi = \eta / ( 3 2 J )$ decreases. Evaluating them at $r = w = \bar { u } _ { i }$ gives a bound on the batch size that depends only on d, $D , t _ { i }$ and $\alpha / I ,$ , and that is $\widetilde { O } ( ( d + 1 ) / t _ { i } )$ by (B.4). Summing over the stages gives $\widetilde { O } ( ( d + 1 ) D / \bar { \varepsilon } )$ resets and $O ( D I ) = \tilde { O } ( D )$ pilots and oracle calls, plus one initial call. These deterministic bounds hold on every history, including histories on which a call fails or $f$ is not a valid ceiling.

Validity. Suppose $F ^ { * } \leq f .$ If the call at stage i succeeds and its inputs are valid, both $F ^ { * }$ and $F ( a _ { i } )$ lie in $[ L _ { i } , \tilde { U } _ { i } ]$ , an interval of width at most $t _ { i }$ , so

$$
F ^ { * } \leq F ( a _ { i } ) \leq \widetilde { U } _ { i } \leq L _ { i } + t _ { i } \leq F ^ { * } + t _ { i } \leq f + t _ { i } ;
$$

in particular the clipping at $f + t _ { i }$ is inactive, $F ( a _ { i } ) \leq U _ { i }$ and $F ^ { * } \leq U _ { i }$ , so the inputs of the next call are valid. The first call has valid inputs because $U _ { 0 } = 1$ . Summing the conditional failure probabilities of the calls, each at most $\alpha / I$ given a history with valid inputs, all calls succeed with probability at least $1 - \alpha$ . On this event $F ( a ) \leq U _ { I } \leq f + \bar { \varepsilon }$ and $F ( a ) \leq 1$ , so $F ( a ) \leq w$ . Finally $\varepsilon \leq 2 ( f + \varepsilon )$ and $f + \bar { \varepsilon } = 2 f + \varepsilon \leq 2 ( f + \varepsilon ) , \mathrm { ~ s o ~ } \varepsilon \leq w \leq \operatorname* { m i n } ( 1 , 2 ( f + \varepsilon ) )$ □

## B.3 The anchor without a supplied ceiling

Proof of Lemma $\ 4 . 6 . \ \mathrm { H f } \ \varepsilon = 1$ , return an empty-list member and $w _ { 0 } = 1$ . Let $\varepsilon < 1 , I = \lceil \log _ { 2 } ( 1 / \varepsilon ) \rceil$ and $t _ { i } = \operatorname* { m a x } ( \varepsilon , 2 ^ { - i } )$ for $1 \leq i \leq I$ . Start with an empty-list member $a _ { 0 }$ and $U _ { 0 } = 1$ . At stage i run Ownership $( a _ { i - 1 } , U _ { i - 1 } , U _ { i - 1 } , t _ { i } , \alpha / I )$ for the natural loss, and let $a _ { i }$ and $[ L _ { i } , U _ { i } ]$ be its (clipped and totalized) output. Stop at the first i with $t _ { i } = \varepsilon \ \mathrm { o r } \ U _ { i } > 2 t _ { i } ;$ call it $^ { c , }$ and return $a = a _ { c }$ and $w _ { 0 } = \operatorname* { m a x } ( \varepsilon , U _ { c } ) \in [ \varepsilon , 1 ]$

Costs on every history. The first call has bounds $1 \le 2 t _ { 1 }$ . A later stage i is reached only if stage $i - 1$ did not stop, so $U _ { i - 1 } \leq 2 t _ { i - 1 } \leq 4 t _ { i }$ . As in the supplied case, the call at stage i therefore has at most $1 2 8 D + 1$ batches and pilots, and its batch size is at most a number that depends only on $d ,$ $D , t _ { i }$ and $\alpha / I$ and is $\widetilde { O } ( ( d + 1 ) / t _ { i } )$ . Since $\textstyle \sum _ { i \leq c } t _ { i } ^ { - 1 } \leq 3 / t _ { c }$ and $t _ { c } \geq \varepsilon$ , the claimed caps follow. All numerical inputs stay in [0, 1] even after an unsuccessful call; the clipping and the proper fallback with [0, 0] are what make this all-history statement possible.

Validity. As in the supplied case, with probability at least $1 - \alpha$ all executed calls succeed with valid inputs, and then $F ^ { * } \leq F ( a _ { i } ) \leq U _ { i } \leq F ^ { * } + t _ { i }$ at each executed stage. If $t _ { c } > \varepsilon$ , the stopping rule gives $U _ { c } > 2 t _ { c } .$ , so $F ^ { * } \geq U _ { c } - t _ { c } > t _ { c }$ and $v _ { 0 } = U _ { c } \le F ^ { * } + t _ { c } < 2 F ^ { * }$ . If $t _ { c } = \varepsilon$ , then $w _ { 0 } \leq \operatorname* { m a x } ( \varepsilon , F ^ { * } + \varepsilon ) \leq F ^ { * } + \varepsilon$ . In both cases $F ( a ) \leq U _ { c } \leq w _ { 0 } \leq 2 ( F ^ { * } + \varepsilon )$

Local reset count. If $c = 1$ , then $t _ { c } \ge 1 / 2$ and ${ F } ^ { * } + \varepsilon \le 2 \le 4 { t } _ { c }$ . If $c > 1$ , success at stage $c - 1$ and the fact that it did not stop give $F ^ { * } \le U _ { c - 1 } \le 2 t _ { c - 1 } \le 4 t _ { c }$ , and also $\varepsilon \leq t _ { c } .$ . In all cases $t _ { c } \ge ( F ^ { * } + \varepsilon ) / 5$ on the success event. □

Remark B.2 (Deployment and failure accounting). An empty-list representative is obtained without being played. When the coarse schedule runs, its initial pilot and every candidate, including the final validated anchor, are played and counted. When the schedule is bypassed because $\bar { \varepsilon } = 1$ , the anchor may be unplayed; its first use by a logger is then a counted training pilot. Enumerating a dictionary plays nothing by itself. The bounds above control the union of actual deployments by the total number of pilot slots over all calls. Coverage and the local bounds are used only while all earlier calls have succeeded; the numerical caps, properness and the ordered intervals of the stated widths hold on every history. No conditioning on the final adaptive list, no free exploration batch and no evaluation of a hidden terminal is used.

## C Proof of the acquisition lemma

We prove Lemma 5.2. All probabilities are conditional on the history before the stage, which fixes $a , L , w , \zeta , b , \alpha ;$ assume $F ( a ) \leq w$ for part (b). The stopped-data content of the proof is the legal gain of Lemma 5.1 and the uniform transfer through observed patterns. Given these, the rest is the standard conversion through a multiplicative potential, bounded martingale concentration and one randomized sparsification.

Part (a) and the sample sizes. The procedure plays a pilot drawn from L on each of its $m + n$ resets, makes one oracle call on the m dictionary displays, and draws k slot identities with private coins; the oracle call and the draws play nothing. Since the dictionary displays contain at most Dm queries, Lemma A.1(i) gives $K : = | \mathcal { G } | \leq K _ { 0 } = ( D m + 1 ) ^ { d }$ . For part (c): since $2 D m + 1 \le 3 D m$ , the defining condition on m holds if (m log $2 ) / ( 2 W _ { 0 } ) \geq \log ( 8 / \alpha ) + 2 d \log ( 3 D m )$ , and by Lemma A.6 (with $A = 2 W _ { 0 } / \log 2 , B = \log ( 8 / \alpha ) , k = 2 d , c = 3 D )$ this holds for every

$$
m \ \ge \ \frac { 4 W _ { 0 } } { \log 2 } \Big ( \log \frac { 8 } { \alpha } + 2 d \log \frac { 2 4 d D W _ { 0 } } { \log 2 } \Big ) \qquad ( d \ge 1 ) ,
$$

and for every $m \ge ( 2 W _ { 0 } / \log 2 ) \log ( 8 / \alpha )$ if $d = 0$ . Together with $m \ge 8 W _ { 0 }$ this shows $m =$ $O ( W _ { 0 } [ \log ( e / \alpha ) + d \log ( e ( d + 1 ) D W _ { 0 } ) ] )$ . Next $n = \lceil W \iota \rceil$ with $\iota = \log ( 4 / \alpha ) + d \log ( D m + 1 )$ , and $W = 2 W _ { 0 } / \Lambda _ { b } \leq 2 W _ { 0 } / \log 3$ . Hence $n + n = \widetilde O ( ( d + 1 + \log ( 1 / \alpha ) ) W _ { 0 } )$ with $W _ { 0 } = 1 / ( \zeta w b )$ , where the hidden factor is polylogarithmic in $D , d + 1 , W _ { 0 }$ and $1 / \alpha$ . The bound on k is immediate.

The observed dictionary. For $h , h ^ { \prime } \in { \mathcal { H } }$ consider the event $O _ { h h ^ { \prime } }$ of Lemma $5 . 1 ( \mathrm { d } )$ on one display of a pilot drawn from L. It depends on the pair only through the values of $h , h ^ { \prime }$ at the at most D displayed queries, together with data of the display, so by Lemma $\mathrm { A . 1 } ( \mathrm { i i } )$ these events realize at most $( 2 D m + 1 ) ^ { 2 d }$ indicator vectors on 2m displays. The m dictionary displays are independent with a common law. By Lemma $\mathrm { A . 2 }$ with $\chi = 1 / W _ { 0 }$ (note $m \chi \geq 8 )$ , except with probability $2 ( 2 D m + 1 ) ^ { 2 d } 2 ^ { - m / ( 2 W _ { 0 } ) } \leq \alpha / 4$ , every pair $( h , h ^ { \prime } )$ such that no dictionary display lies in $O _ { h h ^ { \prime } }$ satisfies $\operatorname* { P r } ( O _ { h h ^ { \prime } } ) \leq 1 / W _ { 0 }$ . This uniform statement is established before any representative is chosen.

Every $h \in \mathcal H$ agrees with its representative $h ^ { \prime } \in \mathcal { G }$ at all displayed dictionary queries. Then no dictionary display lies in $O _ { h h ^ { \prime } }$ , because $O _ { h h ^ { \prime } }$ requires exactly one of the two to agree with the label at some displayed query. Integrating (5.2) over $T$ therefore gives, on the event above, simultaneously for all $h ,$

$$
\operatorname* { P r } _ { T } \bigl ( \rho _ { L } ( h ) \neq \rho _ { L } ( h ^ { \prime } ) \bigr ) \leq \frac { \operatorname* { P r } ( O _ { h h ^ { \prime } } ) } { \zeta } \leq \frac { 1 } { \zeta W _ { 0 } } = w b .\tag{C.1}
$$

The representatives are algorithmic objects obtained from displayed queries; no latent net is used here.

The exponential-weights pass. From now on condition also on the dictionary, which is fixed during the pass. Let $\mathcal { F } _ { t - 1 }$ contain the dictionary and the first $t - 1$ rounds of the pass, so that $\pi _ { t }$ is $\mathcal { F } _ { t - 1 }$ -measurable while the pilot and the record of round t are fresh. Put

$$
G _ { t , h } = \mathbb { E } [ \gamma _ { t , h } \mid { \mathcal { F } } _ { t - 1 } ] , \qquad { \bar { \gamma } } _ { t } = \sum _ { h \in { \mathcal { G } } } \pi _ { t } ( h ) \gamma _ { t , h } .
$$

By Lemma 5.1(b),(c) and Lemma 4.4,

$$
G _ { t , h } = \frac { \mathbb { E } _ { T } \Phi _ { \pi _ { t } } ( T , \rho _ { L } ( h ) ) } { w \Lambda _ { b } } , \qquad 0 \le \gamma _ { t , h } \le \frac { 2 } { \zeta b w \Lambda _ { b } } = W , \qquad \mathbb { E } [ \bar { \gamma } _ { t } \mid \mathcal { F } _ { t - 1 } ] \le \frac { F ( a ) \Lambda _ { b } } { w \Lambda _ { b } } \le 1 .
$$

In the last bound the same numerical lottery $\pi _ { t }$ appears as weight and in the potential, which is exactly the setting of Lemma 4.4.

Pathwise bound. Recall $K = | \mathcal { G } | \leq K _ { 0 }$ , and let $\begin{array} { r } { Z _ { t } = \sum _ { h \in \mathcal { G } } K ^ { - 1 } \exp ( \sum _ { s < t } \gamma _ { s , h } / W ) } \end{array}$ , so that $Z _ { 1 } = 1$ and $\pi _ { t } ( h ) = K ^ { - 1 } \exp ( \sum _ { s < t } \gamma _ { s , h } / W ) / Z _ { t }$ . Since $e ^ { x } \leq 1 + ( e - 1 ) x$ for $x \in [ 0 , 1 ]$ ，

$$
\frac { Z _ { t + 1 } } { Z _ { t } } = \sum _ { h \in \mathcal { C } } \pi _ { t } ( h ) e ^ { \gamma _ { t , h } / W } \leq 1 + ( e - 1 ) \frac { \bar { \gamma } _ { t } } { W } \leq \exp \Bigl ( ( e - 1 ) \frac { \bar { \gamma } _ { t } } { W } \Bigr ) .
$$

As $\begin{array} { r } { K ^ { - 1 } \exp ( \sum _ { t } \gamma _ { t , h } / W ) \leq Z _ { n + 1 } } \end{array}$ for every $h \in { \mathcal { G } }$ , taking logarithms gives

$$
\sum _ { t = 1 } ^ { n } \gamma _ { t , h } \leq W \log K + ( e - 1 ) \sum _ { t = 1 } ^ { n } { \bar { \gamma } } _ { t } \qquad ( h \in { \mathcal { G } } ) .\tag{C.2}
$$

Two supermartingales. Apply Lemma $\mathrm { A . 7 ( a ) }$ with $\vartheta = 1$ to the $[ 0 , 1 ]$ -valued variables $\bar { \gamma } _ { t } / W$ and $\gamma _ { t , h } / W \colon$

$$
\mathbb { E } \big [ e ^ { \widehat { \mathsf { r } } _ { t } / W } \big \vert \mathcal { F } _ { t - 1 } \big ] \leq \exp \Big ( ( e - 1 ) \frac { \mathbb { E } \big [ \widehat { \gamma } _ { t } \big \vert \mathcal { F } _ { t - 1 } \big ] } { W } \Big ) , \qquad \mathbb { E } \big [ e ^ { - \gamma _ { t , h } / W } \big \vert \mathcal { F } _ { t - 1 } \big ] \leq \exp \Big ( - ( 1 - e ^ { - 1 } ) \frac { G _ { t , h } } { W } \Big ) .
$$

Multiplying over the rounds gives two nonnegative supermartingales starting at one. Markov’s inequality at the final time, with threshold $e ^ { \iota }$ , and a union over the $K \leq K _ { 0 }$ coordinates in the second case, show that outside an event of probability at most $e ^ { - \iota } + K _ { 0 } e ^ { - \iota } \leq \alpha / 2$

$$
\sum _ { t } \bar { \gamma } _ { t } \leq ( e - 1 ) \sum _ { t } \mathbb { E } [ \bar { \gamma } _ { t } \mid \mathcal { F } _ { t - 1 } ] + W \iota \leq ( e - 1 ) n + W \iota , \qquad ( 1 - e ^ { - 1 } ) \sum _ { t } G _ { t , h } \leq \sum _ { t } \gamma _ { t , h } + W \iota \quad ( h \in \mathcal { G } ) .
$$

Since W log $K \leq W \iota \leq n$ , combining with (C.2) yields

$$
( 1 - e ^ { - 1 } ) \sum _ { t } G _ { t , h } \leq W \iota + ( e - 1 ) ( ( e - 1 ) n + W \iota ) + W \iota \leq ( 2 + ( e - 1 ) e ) n ,
$$

that is, $n ^ { - 1 } \textstyle \sum _ { t } G _ { t , h } \leq C _ { \mathrm { E G } }$ for every $h \in { \mathcal { G } }$ , with $C _ { \mathrm { E G } } = ( 2 + ( e - 1 ) e ) / ( 1 - e ^ { - 1 } ) = 1 0 . 5 5 3 0 \dots . . .$

Averaging. For fixed T and $l , \pi \mapsto \Phi _ { \pi } ( T , l )$ is convex, and neither the indicator $\mathbf { 1 } \{ l > \tau _ { a } \}$ nor the rank $\rho _ { L } ( h )$ depends on the numerical lottery. Jensen’s inequality for the average π¯ of the pre-update vectors gives, pointwise, $\begin{array} { r } { \Phi _ { \bar { \pi } } ( T , \rho _ { L } ( h ) ) \le n ^ { - 1 } \sum _ { t } \Phi _ { \pi _ { t } } ( T , \rho _ { L } ( h ) ) } \end{array}$ ), hence

$$
\mathbb { E } _ { T } \Phi _ { \bar { \pi } } ( T , \rho _ { L } ( h ) ) \leq \frac { 1 } { n } \sum _ { t } w \Lambda _ { b } G _ { t , h } \leq C _ { \mathrm { E G } } w \Lambda _ { b } \qquad ( h \in \mathcal { G } ) .\tag{C.3}
$$

One sparsification. Condition on the whole pass, and let p be the uniform lottery on k independent draws from $\bar { \pi } .$ . Define the bad set

$$
B _ { \mathrm { s p } } = \{ T \colon \tau _ { a } \leq M \mathrm { ~ a n d ~ } \exists l \leq M + 1 \mathrm { ~ w i t h ~ } Q _ { \bar { \tau } } ( T , l ) \geq b \mathrm { ~ a n d ~ } Q _ { p } ( T , l ) < Q _ { \bar { \pi } } ( T , l ) / 2 \} .
$$

For a fixed record and rank, $Q _ { p } ( T , l )$ is the average of k independent indicators with mean $Q _ { \bar { \pi } } ( T , l )$ so by the multiplicative Chernof bound $\mathrm { P r } ( Q _ { p } < \bar { Q } _ { \bar { \pi } } / 2 ) \leq e ^ { - k Q _ { \bar { \pi } } / 8 } \leq e ^ { - k b / 8 }$ when $Q _ { \bar { \pi } } \geq b .$ . A union over the at most $D + 1$ ranks of the record and integration over a fresh record with $\tau _ { a } \leq M$ give

$$
\mathbb { E } _ { \mathrm { c o i n s } } \operatorname* { P r } _ { T } ( B _ { \mathrm { s p } } ) \leq F ( a ) ( D + 1 ) e ^ { - k b / 8 } \leq w ( D + 1 ) e ^ { - k b / 8 } \leq \frac { \alpha w b } { 4 }
$$

by the choice of k. By Markov’s inequality, $\mathrm { P r } _ { T } ( B _ { \mathrm { s p } } ) \leq$ wb except with probability $\alpha / 4$ over the coins. This bad set does not depend on the comparator.

On records with $\tau _ { a } \leq M$ outside $B _ { \mathrm { s p } }$ we have $\begin{array} { r } { Q _ { p } ( T , l ) + b \geq \frac { 1 } { \gamma } ( Q _ { \bar { \pi } } ( T , l ) + b ) } \end{array}$ for every l: if $Q _ { \bar { \pi } } ( T , l ) \geq b$ this follows from $Q _ { p } \geq Q _ { \bar { \pi } } / 2$ , and otherwise from $b \ge ( Q _ { \bar { \pi } } + b ) / 2$ . Hence $\Phi _ { p } \le 2 \Phi _ { \bar { \pi } }$ there. No such comparison is claimed on records on which the anchor survives, and none is needed, because there all potentials at ranks at most $M + 1$ vanish. On $B _ { \mathrm { s p } }$ the sparse potential is at most $1 / b$ . With (C.3),

$$
\mathbb { E } _ { T } \Phi _ { p } ( T , \rho _ { L } ( h ) ) \le 2 C _ { \mathrm { E G } } w \Lambda _ { b } + \frac { \mathrm { P r } _ { T } ( B _ { \mathrm { s p } } ) } { b } \le ( 2 C _ { \mathrm { E G } } \Lambda _ { b } + 1 ) w \qquad ( h \in \mathcal { G } ) .
$$

Transfer to the class. Let $h \in \mathcal H$ and let $h ^ { \prime } \in \mathcal { G }$ be its representative. Both potentials lie in $[ 0 , 1 / b ]$ and coincide when the truncated ranks coincide, so

$$
\vert \Phi _ { p } ( T , \rho _ { L } ( h ) ) - \Phi _ { p } ( T , \rho _ { L } ( h ^ { \prime } ) ) \vert \leq \frac { 1 } { b } \mathbf { 1 } \{ \rho _ { L } ( h ) \neq \rho _ { L } ( h ^ { \prime } ) \} ,
$$

and (C.1) adds at most $w .$ This proves (5.3) with $C ( b ) = 2 C _ { \mathrm { E G } } \Lambda _ { b } + 2$ , simultaneously for all $h \in \mathcal H$ The three failure probabilities add up to $\alpha / 4 + \alpha / 2 + \alpha / 4 = \alpha$ . The transfer is paid once, after the sparsification. No function is played because of its numerical weight or because it was enumerated; only pilots drawn from the frozen logger are played during the procedure.

## D A sparse graph with the required path property

We prove Lemma 6.1. The argument is a reconstruction of the sparse-connector, good-vertex and joining method of Erdős, Graham and Szemerédi [12], written in the local-expander language of Alwen, Blocki and Pietrzak [3, Section 4]. Those graph lemmas were already public in the May 2017 version of the preprint; the cited July 2017 version supplies the locators. The proof below fixes constants and boundary cases and is not a new graph-theoretic result.

Let $m _ { 0 } \geq 5$ be an integer, $\xi = 1 / m _ { 0 }$ and $\gamma \in [ 5 \xi , 1 )$ . A directed graph on [S] with edges $( x , y )$ $x < y .$ , is a ξ-local expander if it contains every edge $( x , y )$ with $1 \leq y - x \leq 2 m _ { 0 } - 1$ , and, for every two adjacent intervals

$$
I = [ v , v + r - 1 ] , \qquad J = [ v + r , v + 2 r - 1 ] \subseteq [ S ] ,
$$

all $X \subseteq I$ and $Y \subseteq J$ with $| X | , | Y | \ge \xi r$ are joined by an edge from X to Y. For $r \leq m _ { 0 }$ the short edges already give this, since then X and $Y$ are nonempty and every $x \in X , y \in Y$ satisfy $1 \leq y - x \leq 2 r - 1$

Lemma D.1 (Random local expansion). Include all edges $( x , y )$ with $1 \leq y - x \leq 2 m _ { 0 } - 1$ . For each $y \in \{ 2 , \ldots , S \}$ and each integer i with $m _ { 0 } \leq 2 ^ { i } \leq 2 S$ , independently draw

$$
k _ { G } = \Bigl \lceil 4 m _ { 0 } \Bigl ( 2 \log ( e m _ { 0 } ) + \log \frac { 2 S ^ { 2 } } { \beta } \Bigr ) \Bigr \rceil
$$

uniform points from the window $\left[ \operatorname* { m a x } ( 1 , y - 2 ^ { i + 1 } ) , y - 1 \right]$ , which is nonempty because $y \geq 2$ , and add the edges from these points to y. With probability at least $1 - \beta$ the graph is a ξ-local expander, and on every draw every indegree is at most min $\{ S - 1 , \ 2 m _ { 0 } - 1 + k _ { G } ( \lfloor \log _ { 2 } ( 2 S ) \rfloor + 1 ) \}$

Proof. The indegree bound is immediate: there are at most $2 m _ { 0 } - 1$ short edges into $y ,$ at most $\lfloor \log _ { 2 } ( 2 S ) \rfloor + 1$ scales, at most $k _ { G }$ draws per scale (duplicates only reduce the count), and at most $y - 1 \le S - 1$ possible predecessors. Vertex 1 receives no random edge and has no predecessors.

Fix $r > m _ { 0 }$ and adjacent intervals $I , J$ as above, and put $s = \lceil \xi r \rceil \geq 2$ . It sufices to consider $| X | = | Y | = s .$ . Let $i = \left\lceil \log _ { 2 } r \right\rceil$ , so that $m _ { 0 } < r \le 2 ^ { i } < 2 r \le S ;$ hence scale i is drawn. Every $y \in Y$ satisfies $y \ge v + r \ge 2$ , so y receives random edges; and $y \le v + 2 r - 1$ with $2 ^ { i + 1 } \geq 2 r$ gives $y - 2 ^ { i + 1 } < v _ { : }$ , so the window of y at scale i contains $I \supseteq X$ and has fewer than 4r points. One draw hits X with probability at least $s / ( 4 r ) \geq \xi / 4$ . The draws are independent over $y \in Y$ and over repetitions, so

$$
\operatorname* { P r } ( \mathrm { n o ~ e d g e ~ f r o m ~ } X \mathrm { ~ t o ~ } Y ) \le ( 1 - \xi / 4 ) ^ { k _ { G } s } \le e ^ { - k _ { G } s / ( 4 m _ { 0 } ) } .
$$

There are at most $\textstyle { \binom { r } { s } } ^ { 2 } \leq ( e r / s ) ^ { 2 s } \leq ( e m _ { 0 } ) ^ { 2 s }$ choices of $( X , Y )$ , so the failure probability for this pair of intervals is at most

$$
\exp \Bigl [ - s \Bigl ( \frac { k _ { G } } { 4 m _ { 0 } } - 2 \log ( e m _ { 0 } ) \Bigr ) \Bigr ] \le \exp \Bigl [ - s \log \frac { 2 S ^ { 2 } } { \beta } \Bigr ] \le \frac { \beta } { 2 S ^ { 2 } } .
$$

A union bound over at most $S ^ { 2 } / 2$ pairs of adjacent intervals gives failure at most $\beta / 4 < \beta$ . The cases $r \leq m _ { 0 }$ are covered by the short edges. □

Given a deletion set $B \subseteq [ S ]$ , call $x \in [ S ] \ \prime$ γ-good if every interval in [S] with endpoint x has at most a $( 1 - \gamma ) – \mathrm { f r a c t i o n }$ of deleted vertices:

$$
\begin{array} { r } { | B \cap [ x , x + \varrho - 1 ] | \leq ( 1 - \gamma ) \varrho \quad \mathrm { a n d } \quad | B \cap [ x - \varrho + 1 , x ] | \leq ( 1 - \gamma ) \varrho } \end{array}
$$

for every $\varrho \ge 1$ for which the interval lies in [S]. Taking $\varrho = 1$ shows that a good vertex is not deleted.

Lemma D.2 (Few bad vertices). At most $2 | B | / ( 1 - \gamma )$ vertices are not γ-good.

Proof. If B is empty, every vertex is good. Otherwise call x forward-bad if some interval $[ x , x + \varrho - 1 ] \subseteq$ [S] has more than $( 1 - \gamma ) \varrho$ deleted vertices. Choose the least forward-bad vertex and such a witness interval; then the least forward-bad vertex beyond this interval and a witness interval; and so on. The chosen intervals are disjoint and cover all forward-bad vertices, and each of them contains more than a $( 1 - \gamma )$ -fraction of deleted vertices, so their total length is less than $| B | / ( 1 - \gamma )$ . Backward-bad vertices are counted in the same way with the order reversed. The sum of the two bounds proves the lemma. □

Lemma D.3 (Joining good vertices). In a ξ-local expander with $\gamma \geq 5 \xi$ , any two γ-good vertices $x < y$ are joined by a directed path whose vertices lie in $[ x , y ] \setminus B$

Proof. Let $n = y - x + 1$ . If $y - x \leq 2 m _ { 0 } - 1$ , the short edge $( x , y )$ works, as $x , y \notin B .$ . Otherwise $n > 2 m _ { 0 }$ . Let $R _ { f }$ be the set of vertices of $[ x , y ] \backslash B$ reachable from x by a path inside $[ x , y ] \backslash B$ , and $R _ { b }$ the set of vertices of $[ x , y ] \setminus B$ from which y is reachable inside $[ x , y ] \setminus B$

Expansion step. If $2 \varrho \le n$ and $| R _ { f } \cap [ x , x + \varrho - 1 ] | \geq \xi \varrho _ { ! }$ , then fewer than $\xi \varrho$ undeleted vertices of $[ x + \varrho , x + 2 \varrho - 1 ]$ lie outside $R _ { f }$ . Otherwise local expansion would give an edge from a vertex of $R _ { f }$ to an undeleted vertex of $[ x + \varrho , x + 2 \varrho - 1 ]$ outside $R _ { f }$ , and that vertex would be in $R _ { f }$

Doubling. We show by induction that for every $i \geq 0$ with $2 ^ { i } \leq n$ , fewer than $\xi 2 ^ { i }$ undeleted vertices of $[ x , x + 2 ^ { i } - 1 ]$ lie outside $R _ { f } .$ , and $| R _ { f } \cap [ x , x + 2 ^ { i } - 1 ] | \geq ( \gamma - \xi ) 2 ^ { i }$ . For $i = 0$ the interval is $\{ x \}$ and $\boldsymbol { x } \in R _ { f }$ . If the claim holds for i and $2 ^ { i + 1 } \leq n ,$ , then $| R _ { f } \cap [ x , x + 2 ^ { i }$ $- 2 ^ { i } - 1 ] | \geq ( \gamma - \xi ) 2 ^ { i } \geq \xi 2 ^ { i }$ because $\gamma \geq 2 \xi$ , so the expansion step with $\varrho = 2 ^ { i }$ shows that fewer than $\xi 2 ^ { i }$ undeleted vertices of the second half lie outside $R _ { f } ;$ adding the two halves gives fewer than $\xi 2 ^ { i + 1 }$ . Goodness of x gives at least $\gamma 2 ^ { i + 1 }$ undeleted vertices in $[ x , x + 2 ^ { i + 1 } - 1 ]$ , so at least $( \gamma - \xi ) 2 ^ { i + 1 }$ of them are in $R _ { f }$ . The same argument with the order reversed, using backward goodness of y, gives $| R _ { b } \cap [ y - 2 ^ { i } + 1 , y ] | \geq ( \gamma - \xi ) 2 ^ { i }$ for $2 ^ { i } \leq n$

Connection. Put $r = \lfloor n / 2 \rfloor \geq m _ { 0 }$ and let p be the largest power of two with $p \leq r ,$ , so $p > r / 2$ Let $I _ { 1 } = [ x , x + r - 1 ]$ and $I _ { 2 } = [ x + r , x + 2 r - 1 ] \subseteq [ x , y ]$ . Then $| R _ { f } \cap I _ { 1 } | \geq ( \gamma - \xi ) p > ( \gamma - \xi ) r / 2 \geq \xi r ,$ since $\gamma \geq 3 \xi$ . Moreover $[ y - p + 1 , y - 1 ] \subseteq I _ { 2 }$ , because $r + p \leq 2 r \leq \mathrm { ~ ( ~ }$ n and $2 r \geq n - 1$ . Hence

$$
| R _ { b } \cap I _ { 2 } | \geq ( \gamma - \xi ) p - 1 > \frac { ( \gamma - \xi ) r } { 2 } - 1 \geq \xi r ,
$$

where the last step uses $r \ge m _ { 0 } = 1 / \xi$ and $\gamma \geq 5 \xi$ . Local expansion gives an edge from $R _ { f } \cap I _ { 1 }$ to $R _ { b } \cap I _ { 2 }$ . Concatenating a path from x to its tail, the edge, and a path from its head to y gives the required path. □

Proof of Lemma 6.1. Use Lemma D.1 with $m _ { 0 } = 1 0$ , so that $\xi = 1 / 1 0 .$ , 2m $_ { 0 } - 1 = 1 9$ and $k _ { G }$ is as in (6.1), and take $\gamma = 1 / 2 = 5 \xi$ . The short edge $( s - 1 , s )$ gives $s - 1 \in A _ { s }$ for $s \geq 2$ , and the indegree bound is $k _ { A }$ . On the local-expansion event, fix any $B \subseteq [ S ]$ and list the good vertices in increasing order. By Lemma D.2 there are at least $S - 4 | B |$ of them. Join consecutive good vertices by Lemma D.3; the joining paths lie in intervals with disjoint interiors, so their concatenation is a directed path avoiding B through all good vertices, with at least $S - 4 | B |$ vertices. A single good vertex is a path with one vertex, and if the bound $S - 4 | B |$ is nonpositive there is nothing to prove. Local expansion holds simultaneously for all pairs of intervals, so the conclusion holds for every deletion set at once, including sets that depend on the graph and on later data. □

Remark D.4 (Variants of the stage graph). (i) Since Lemma D.1 shows that a suitable graph exists, a deterministic exhaustive search over graphs with the same short edges and indegree cap also finds one; testing local expansion is a finite computation and uses no reset.

(ii) Blocki, Cinkoske, Lee and Son [9, Section 4, Theorem 9] give explicit graphs $G ( \delta , N )$ on $N = 2 ^ { n }$ vertices with maximum indegree O(log N) that are δ-local expanders in the sense that every two adjacent intervals of equal length with subsets of size at least δ times the length are joined by an edge; they credit Theorem 9 to the argument of Erdős, Graham and Szemerédi applied to their explicit bipartite expanders. Take the prefix [S] of their graph on the next power of two, with $\delta = 1 / 1 0$ , and add every edge of length at most 19. Every test of adjacent intervals inside [S] keeps its edge, and the added edges give the short-edge condition. Lemmas D.2 and D.3 then give the path property on every draw, with indegree O(log S) and no failure allowance. This uses the published graph property, not an implementation of the pseudocode. The construction was public in 2021 and published at STACS in 2022.

(iii) The complete graph on [S], with $k _ { A } = S - 1$ , is a stage graph with the path property on every draw (Section 6). Running Procedure 3 with the complete graph instead of the random one gives a learner whose resets and distinct pilots are at most $\widetilde { \cal O } ( ( d + 1 ) [ D ^ { 2 } ( f + \varepsilon ) + r ] / \varepsilon ^ { 2 } )$ and $\widetilde { O } ( D ( 1 + f / \varepsilon ) )$ on every history, and which covers with probability at least $1 - \delta$ when $F ^ { * } \leq f$ and $R ^ { * } \leq r$ . Indeed Proposition 6.7 applies with $k _ { A } = 4 D - 1$ and no graph event, so $\left( 6 . 1 3 \right)$ has $D k _ { A } w = O ( D ^ { 2 } ( f + \varepsilon ) )$ , while the pilot bound in Proposition $6 . 7 ( \mathrm { a } )$ does not depend on $k _ { A } ;$ the failure probability is at most $\delta / 4 + 2 \delta / 1 6 < \delta$ . We do not use this weaker variant in the main theorem.

## E Uniform relative estimation and the joint interval

This appendix proves Lemma 6.6. We first state a generic estimate for a frozen lottery, then bound its sample size, then construct the interval. All randomness before the frozen lottery is conditioned on. The latent-record nets and rare counts of Appendix A are used only in the analysis.

Lemma E.1 (Frozen-lottery relative estimate). Condition on a history that fixes a lottery Π and a measurable function $\psi \geq 0$ , the scoring rule. In each of n rounds a pilot g<sub>i</sub> is played on a fresh record $T _ { i }$ , and, conditionally on the history, the pairs $( g _ { 1 } , T _ { 1 } ) , \ldots , ( g _ { n } , T _ { n } )$ are independent, with $g _ { i } \sim \Pi$ and $T _ { i } \sim \mathcal { P }$ independent of $g _ { i }$ . The score of $h \in \mathcal H$ in round i is $Y _ { h , i } = \psi ( g _ { i } , T _ { i } , \tau _ { h } ( T _ { i } ) )$ . For a generic pair $( g , T )$ with this law write $Y _ { h } = \psi ( g , T , \tau _ { h } ( T ) )$ , and assume:

(i) $Y _ { h }$ is a function of the display of g on T and of the values of h at the displayed queries;

(ii) $0 \leq Y _ { h } \leq W _ { Y }$ for a number $W _ { Y } \geq 1$ ;

(iii) $\mathbb { E } [ Y _ { h } \mid T ] \in [ 0 , 1 ]$ and $R ( h ) - \beta _ { b } \leq \mathbb { E } Y _ { h } \leq R ( h )$ for a number $\beta _ { b } \geq 0$ ;

(iv) $\mathbb { E } Y _ { h } ^ { 2 } \le c _ { 1 } R ( h ) + c _ { 2 }$ for numbers $c _ { 1 } > 0 , c _ { 2 } \geq 0$

Let $v \in [ 0 , 1 ] , 0 < t \leq 1$ and $0 < \beta < 1$ , define $s _ { E }$ and θ by (6.9) and $\nu _ { n } , H _ { n } , \iota _ { n } , \kappa _ { n }$ by (6.10), and suppose $n \geq 2$ satisfies the two conditions of (6.11). Let $\hat { R } ( h )$ be the average of the n scores of h. Then, with probability at least $1 - \beta$

$$
| \widehat { R } ( h ) - R ( h ) | \leq \theta R ( h ) + \frac { 3 t } { 1 6 } + \beta _ { b } \qquad f o r \ e v e r y \ h \in \mathcal { H } .\tag{E.1}
$$

Proof. Let $\mathcal { N } = \mathcal { N } _ { 1 / n }$ be the net of Lemma $\mathrm { A . 3 }$ with mass $1 / n ;$ it has at most $H _ { n }$ members, and every h has a neighbor $h ^ { \prime } \in \mathcal { N }$ with $\mathcal { P } ( E ( h , h ^ { \prime } ) ) \leq 1 / n$ . No member of the net is given to the learner.

Concentration on the net. For a fixed $h ^ { \prime } \in \mathcal { N }$ the n scores $Y _ { h ^ { \prime } , i } = \psi ( g _ { i } , T _ { i } , \tau _ { h ^ { \prime } } ( T _ { i } ) )$ are values of one fixed function at the independent pairs $( g _ { i } , T _ { i } )$ , so they are independent and identically

distributed. Since $0 \le s _ { E } Y \le 1$ and $e ^ { x } \leq 1 + x + x ^ { 2 }$ for $x \leq 1$ , for either sign

$$
\begin{array} { r } { \mathbb E e ^ { \pm s _ { E } ( Y - \mathbb E Y ) } \le e ^ { \mp s _ { E } \mathbb E Y } \bigl ( 1 \pm s _ { E } \mathbb E Y + s _ { E } ^ { 2 } \mathbb E Y ^ { 2 } \bigr ) \le e ^ { s _ { E } ^ { 2 } \mathbb E Y ^ { 2 } } . } \end{array}
$$

Hence, by the exponential Markov inequality, for every $x > 0$

$$
\operatorname* { P r } \Bigl ( \pm \bigl ( \widehat { R } ( h ^ { \prime } ) - \mathbb { E } Y _ { h ^ { \prime } } \bigr ) \geq s _ { E } \mathbb { E } Y _ { h ^ { \prime } } ^ { 2 } + \frac { x } { s _ { E } n } \Bigr ) \leq e ^ { - x } .
$$

A union bound over both signs and over ${ \mathcal { N } } ,$ with $x = \iota _ { n } = \log ( 4 H _ { n } / \beta )$ , costs at most $\beta / 2$

Rare disagreements. By Lemma A.4, outside an event of probability at most $\beta / 2 ,$ , every event $E ( h , h ^ { \prime } )$ of mass at most $1 / n$ contains at most $\kappa _ { n }$ of the n latent records. The two exceptional events may be dependent; the union bound sufices.

Transfer. Fix h and its neighbor $h ^ { \prime } .$ . Outside $E ( h , h ^ { \prime } )$ the two functions agree at every query of the record, so $\tau _ { h } ( T ) = \tau _ { h ^ { \prime } } ( T )$ and their scores agree for every pilot draw. Hence

$$
| \widehat { R } ( h ) - \widehat { R } ( h ^ { \prime } ) | \leq \frac { W _ { Y } \kappa _ { n } } { n } , \qquad | \mathbb { E } Y _ { h } - \mathbb { E } Y _ { h ^ { \prime } } | \leq \frac { 1 } { n } , \qquad R ( h ^ { \prime } ) \leq R ( h ) + \frac { 1 } { n } ,
$$

where the middle bound uses that the conditional means lie in [0, 1] and agree outside $E ( h , h ^ { \prime } )$ , and the last uses $\ell \in [ 0 , 1 ]$ . Combining these with the concentration bound and with (iv),

$$
| \widehat { R } ( h ) - \mathbb { E } Y _ { h } | \leq s _ { E } c _ { 1 } R ( h ) + s _ { E } c _ { 2 } + \frac { \iota _ { n } } { s _ { E } n } + \frac { W _ { Y } \kappa _ { n } + 1 + s _ { E } c _ { 1 } } { n } .
$$

Now $s _ { E } c _ { 2 } \leq t / 1 6 , \iota _ { n } / ( s _ { E } n ) \leq t / 1 6$ by (6.11), and $s _ { E } c _ { 1 } \leq t / ( 1 6 ( v + t ) ) \leq 1 / 1 6$ , so the last term is at most $( W _ { Y } \kappa _ { n } + 2 ) / n \leq t / 1 6$ . Adding the bias bound from (iii) gives (E.1). □

Lemma E.2 (Sample size of the final estimate). In Lemma E.1, the least integer $n \geq 2$ satisfying (6.11) exists, and it is at most the largest of 2 and the two numbers

$$
\frac { 3 2 } { s _ { E } t } \Big ( \log \frac { 4 } { \beta } + 2 d \log \frac { 3 0 7 2 d ^ { 2 } D } { s _ { E } t } \Big ) , \qquad \frac { 3 2 } { t } \Big ( W _ { Y } \Big ( 4 + 1 8 \log \frac { 4 } { \beta } \Big ) + 2 + 3 6 W _ { Y } d \log \frac { 3 4 5 6 d D W _ { Y } } { t } \Big )
$$

$( f o r d = 0$ the logarithmic terms with factor d are absent), plus one. Consequently

$$
n _ { F } = \tilde { O } \bigg ( \big ( d + 1 + \log ( 1 / \beta ) \big ) \Big [ \frac { W _ { Y } } { t } + \frac { c _ { 1 } \big ( v + t \big ) } { t ^ { 2 } } + \frac { c _ { 2 } + t } { t ^ { 2 } } \Big ] \bigg ) ,
$$

where the hidden factor is polylogarithmic in $d + 1 , D , W _ { Y } , 1 / s _ { E }$ and $1 / t$

Proof. Let $d \ge 1$ . By Lemma A.3 with $\chi = 1 / n ,$ , and since log $4 \leq d \log ( 4 8 d D n )$ , we have $\nu _ { n } \leq 2 4 d n \log ( 4 8 d D n ) + 1$ With $x = 4 8 d D n$ this gives $\nu _ { n } D + 1 \leq ( x / 2 ) \log x + D + 1 \leq x ^ { 2 }$ so $\log ( \nu _ { n } D + 1 ) \ \leq \ 2$ log x and $\iota _ { n } \leq \log ( 4 / \beta ) + 2 d \log ( 4 8 d D n )$ . The first condition of (6.11) therefore holds i $\mathrm { ~ f ~ } n ~ \geq ~ ( 1 6 / ( s _ { E } t ) ) ( \log ( 4 / \beta ) + 2 d \log ( 4 8 d D n ) )$ , and by Lemma A.6 (with A = $1 6 / ( s _ { E } t ) , k = 2 d , c = 4 8 d D )$ this holds for every n above the first displayed number. Next, $\kappa _ { n } \leq 4 + 1 8 \log ( 4 / \beta ) + 3 6 d \log ( 3 D n )$ because 2n $D + 1 \le 3 D n$ , so the second condition holds if $n \ge ( 1 6 / t ) ( W _ { Y } ( 4 + 1 8 \log ( 4 / \beta ) ) + 2 + 3 6 W _ { Y } d \log ( 3 D n ) )$ , and Lemma A.6 (with $A = 1 6 / t$ $k = 3 6 W _ { Y } d , c = 3 D )$ shows that this holds above the second displayed number. For $d = 0$ we have $H _ { n } = 1 , \iota _ { n } = \log ( 4 / \beta )$ and $\kappa _ { n } = \operatorname* { m a x } \{ 4 , 1 8 \log ( 4 / \beta ) \}$ , and the conditions are explicit. Every integer above both thresholds satisfies (6.11), which proves the first claim. For the order, use $1 / s _ { E } \leq W _ { Y } + 1 6 c _ { 1 } ( v + t ) / t + 1 6 ( c _ { 2 } + t ) / t$ □

Lemma E.3 (Proper minimization and the totalized interval). Suppose that (E.1) holds with $\beta _ { b } \leq t / 1 6$ , so that $| \widehat { R } ( h ) - R ( h ) | \leq \theta R ( h ) + t / 4$ for all h. Enumerate the patterns of H on all queries displayed in the n rounds, let $\widehat { h }$ be a returned member minimizing ${ \widehat { R } } ,$ let $q = \widehat { R } ( \widehat { h } )$ , and put

$$
l = \frac { q - t / 4 } { 1 + \theta } , \qquad u = \frac { q + t / 4 } { 1 - \theta } .
$$

Then $l \leq R ^ { * } \leq R ( \widehat { h } ) \leq u$ . If moreover $R ^ { * } \leq v$ , then $u - l \leq 2 t / 3$ . Clip l and u separately $t o \ [ 0 , 1 ]$ and replace the result by [0, 0] if it is unordered or wider than t. On every history the result is an ordered interval in [0, 1] of width at most t and $\widehat { h } \in \mathcal { H }$ ; under the two conditions above it contains $R ^ { * }$ and $R ( \widehat { h } )$

Proof. The empirical score of h depends only on its values at the displayed queries, so its minimum over the returned representatives is its minimum over $\mathcal { H } .$ For every $h , q \leq \widehat { R } ( h ) \leq ( 1 + \theta ) R ( h ) + t / 4$ hence $l \leq R ( h )$ for every h and so $l \leq R ^ { * }$ . Taking an infimizing sequence also gives $q \leq ( 1 + \theta ) R ^ { * } { + } t / 4$ no population minimizer is needed. The lower half of the estimate at $\widehat { h }$ gives $q \geq ( 1 - \theta ) R ( \widehat { h } ) - t / 4$ so $R ( \widehat { h } ) \leq u$ . Since $R ^ { * } \leq R ( \widehat { h } )$ , both numbers lie in $[ l , u ]$

Now let $R ^ { * } \leq v$ . Since $\theta = c _ { 1 } s _ { E } \leq t / ( 1 6 ( v + t ) ) \leq 1 / 1 6$ by (6.9),

$$
u - l = \frac { 2 \theta q + t / 2 } { 1 - \theta ^ { 2 } } \leq \frac { 2 \theta ( 1 + \theta ) v + \theta t / 2 + t / 2 } { 1 - \theta ^ { 2 } } \leq \frac { ( 1 7 / 1 2 8 + 4 / 1 2 8 + 6 4 / 1 2 8 ) t } { 2 5 5 / 2 5 6 } = \frac { 2 t } { 3 } ,
$$

where we used $2 \theta ( 1 + \theta ) v \leq ( 1 7 / 8 ) t v / ( 1 6 ( v + t ) ) \leq 1 7 t / 1 2 8$ and $\theta t / 2 \leq t / 3 2$ . Clipping preserves membership of numbers in [0, 1] and does not increase the width of an ordered pair, so on this event the clipped pair is ordered, has width at most $2 t / 3 ,$ contains both numbers, and the fallback is not used. On every history the fallback guarantees an ordered interval of width at most $t ;$ of the event no coverage is claimed. Every selected representative and the fallback output are members of H.

Proof of Lemma 6.6. Procedure 2 plays $n _ { F }$ pilots drawn independently from Π on fresh resets, so given the history before the final estimate the pairs $( g _ { i } , T _ { i } )$ are independent with $g _ { i } \sim \Pi$ and $T _ { i } \sim \mathcal { P }$ and the scoring rule (6.7) is fixed by that history. The scores satisfy $( \mathrm { i } ) { - } ( \mathrm { i } \mathrm { v } )$ of Lemma E.1 with $W _ { Y } = 2 / b \geq 1 , c _ { 1 } = 2 , c _ { 2 } = 2 C _ { * } w$ and $\beta _ { b } = 6 4 b C ( b ) w \leq t / 1 6$ , by Lemma 6.5, when (6.5) holds. Lemma E.1 then gives (6.12) with probability at least $1 - \beta$ , since $3 t / 1 6 + \beta _ { b } \le t / 4 .$ , and Lemma E.3 gives the remaining claims. The sample size $n _ { F }$ of Procedure 2 is the one of Lemma E.2. No trace bound in this argument uses more than $D$ queries per latent record, so the argument is valid for every horizon. □

## F Unknown ceilings, expected costs and accounting

This appendix proves Theorem 3.2 and Proposition 3.3.

Procedure 5 reuses one anchor and one graph at all precisions, and uses fresh dictionaries, fresh numerical passes, fresh sparse slots and fresh final-estimation data at every precision. The scales satisfy $t _ { I } = \varepsilon , t _ { i } = 2 ^ { - i }$ for $i < I , t _ { i - 1 } \leq 2 t _ { i }$ (with $t _ { 0 } = 1 )$ , and

$$
\sum _ { i = 1 } ^ { I } \frac { 1 } { t _ { i } } \leq \frac { 3 } { \varepsilon } , \qquad \sum _ { i = 1 } ^ { I } \frac { 1 } { t _ { i } ^ { 2 } } \leq \frac { 3 } { \varepsilon ^ { 2 } } ,\tag{F.1}
$$

because $2 ^ { I - 1 } < 1 / \varepsilon$ gives $\begin{array} { r } { \sum _ { i < I } 2 ^ { i } < 2 / \varepsilon } \end{array}$ and $\textstyle \sum _ { i < I } 4 ^ { i } < 4 ^ { I } / 3 < 4 / ( 3 \varepsilon ^ { 2 } )$ . On every history $t _ { i } \le w _ { i } \le 1$ and $0 \leq v _ { i } \leq 1$ , since $w _ { 0 } \in [ \varepsilon , 1 ]$ and the endpoints of $\widehat { I } _ { i - 1 }$ lie in [0, 1].

Procedure 5 Learner of Theorem 3.2 (no supplied ceilings)   
Input: $\overline { { d , D , \varepsilon , \delta } }$   
1: $\mathbf { i f } \ \varepsilon = 1$ then return an empty-list representative and $[ 0 , 1 ]$   
$2 \colon I  \lceil \log _ { 2 } ( 1 / \varepsilon ) \rceil ; t _ { i }  \operatorname* { m a x } ( \varepsilon , 2 ^ { - i } ) \mathrm { f o r } 1 \leq i \leq I ; \beta  \delta / ( 3 2 ( I + 1 ) )$   
3: run the procedure of Lemma 4.6 with failure $\alpha = \delta / 4 ;$ obtain a and w<sub>0</sub>   
4: draw one graph Γ by Lemma 6.1 on [4D] with allowance $\delta / 8$   
5: for $i = 1 , \dots , I$ do   
6: w ← max(t , w ); v ← 1 if i = 1, and $v _ { i } $ the upper endpoint of $\widehat { I } _ { i - 1 } \mathrm { ~ i f ~ } i > 1$   
7: $( \widehat { h } _ { i } , \widehat { I } _ { i } ) \gets \mathrm { P R E C I S I O N S T A G E } ( a , t _ { i } , w _ { i } , v _ { i } , \beta , \Gamma )$ ▷ fresh data at every stage   
8: return $( \widehat { h } _ { I } , \widehat { I } _ { I } )$

Proof of Theorem 3.2. If $\varepsilon = 1$ the claims are trivial, so let $\varepsilon < 1$

(a) Every history. Each output of Procedure 2 is a member of H with an ordered interval in $[ 0 , 1 ]$ of width at most its precision (Proposition $6 . 7 ( \mathrm { a } ) )$ ; the last precision is ε. The graph is drawn with allowance $\delta / 8 ,$ so its cap is $k _ { A } = O ( ( 1 + \log D ) \log ( e D / \delta ) )$ . Replacing $w _ { i }$ and $v _ { i }$ by their bound 1 in (6.13), stage i uses $\widetilde { O } ( ( d + 1 ) ( D k _ { A } + 3 ) / t _ { i } ^ { 2 } )$ resets and plays at most ${ \cal \tilde { O } } ( D / t _ { i } )$ functions. By (F.1) the sums are $\widetilde { O } ( ( d + 1 ) D / \varepsilon ^ { 2 } )$ and $\bar { O } ( D / \varepsilon )$ . The anchor adds $\widetilde { O } ( ( d + 1 ) D / \varepsilon )$ resets and ${ \cal \tilde { O } } ( D )$ pilots by Lemma 4.6. Every function played in a stage is the anchor or one of its slots, so the union of the functions played over all stages is at most one plus the total number of slots; these are the global caps. They are not instancewise bounds.

(b) The success event. Let $E _ { 0 }$ be the event that the anchor procedure succeeds, $E _ { \Gamma }$ the event that Γ has the path property, and $E _ { i }$ the event that all designs and the final estimate of stage i satisfy their guarantees. Then $\operatorname* { P r } ( E _ { 0 } ^ { c } ) \leq \delta / 4$ and $\operatorname* { P r } ( E _ { \Gamma } ^ { c } ) \leq \delta / 8$ . On $E _ { 0 }$ we have $F ( a ) \leq w _ { 0 } \leq 2 ( F ^ { * } + \varepsilon )$ hence $F ( a ) \leq w _ { i }$ and

$$
w _ { i } = \operatorname* { m a x } ( t _ { i } , w _ { 0 } ) \leq 2 F ^ { * } + 3 t _ { i } .
$$

We claim that on $E _ { 0 } \cap E _ { \Gamma } \cap E _ { 1 } \cap \cdot \cdot \cdot \cap E _ { i - 1 }$ we have $R ^ { * } \leq v _ { i } \leq R ^ { * } + 2 t _ { i }$ . For $i = 1$ this holds because $t _ { 1 } \geq 1 / 2$ and $v _ { 1 } = 1$ . For $i > 1$ , stage $i - 1$ had valid inputs and succeeded, so $\widehat { I } _ { i - 1 }$ contains $R ^ { * }$ and has width at most $t _ { i - 1 } \leq 2 t _ { i } ;$ ; its upper endpoint $v _ { i }$ therefore lies in $[ R ^ { * } , R ^ { * } + 2 t _ { i } ]$ . Thus on this event stage i has valid inputs, and Proposition $6 . 7 ( \mathrm { b } )$ , applied conditionally on the history before stage i and on the realized graph, gives $\operatorname* { P r } ( E _ { i } ^ { c } \cap E _ { 0 } \cap E _ { \Gamma } \cap E _ { 1 } \cap \cdots \cap E _ { i - 1 } ) \leq 2 \beta$ . No fresh graph event is charged at later precisions; the single graph is charged once. The total failure probability is at most

$$
\frac \delta 4 + \frac \delta 8 + 2 I \beta < \frac \delta 4 + \frac \delta 8 + \frac \delta { 1 6 } < \delta .
$$

On the joint event the last interval covers. Moreover (6.13) with $w _ { i } \leq 2 F ^ { * } + 3 t _ { i }$ and $v _ { i } \leq R ^ { * } + 2 t _ { i }$ gives

$$
N _ { i } = \widetilde O \left( ( d + 1 ) \frac { D \big ( F ^ { * } + t _ { i } \big ) + R ^ { * } + t _ { i } } { t _ { i } ^ { 2 } } \right) , \qquad V _ { i } = \widetilde O \left( D \frac { F ^ { * } + t _ { i } } { t _ { i } } \right)
$$

for the resets and new pilots of stage i. Summing with (F.1) gives $\widetilde O ( ( d + 1 ) [ D ( F ^ { * } + \varepsilon ) + R ^ { * } ] / \varepsilon ^ { 2 } )$ resets, since $\begin{array} { r } { D \sum _ { i } t _ { i } ^ { - 1 } \leq 3 D / \varepsilon \leq 3 D \varepsilon / \varepsilon ^ { 2 } , } \end{array}$ and $\widetilde O ( D ( F ^ { * } / \varepsilon + I ) ) = \widetilde O ( D ( 1 + F ^ { * } / \varepsilon ) )$ pilots, since $I = O ( \log ( e / \varepsilon ) )$ ). On $E _ { 0 }$ the anchor uses $\widetilde O ( ( d + 1 ) D / ( F ^ { * } + \varepsilon ) ) \le \widetilde O ( ( d + 1 ) D ( F ^ { * } + \varepsilon ) / \varepsilon ^ { 2 } )$ resets and ${ \cal \tilde { O } } ( D )$ pilots. All new sparse slots and all anchor pilots are counted; reuse can only make the union smaller.

(c) Expected costs. Run the same learner with internal failure $\delta ^ { \prime } = \operatorname* { m i n } ( \delta , \varepsilon ^ { 2 } )$ , recomputing all allocations and logarithms; then (a) and (b) hold with $\delta ^ { \prime } { } .$ and $\log ( 1 / \delta ^ { \prime } ) \leq \log ( 1 / \delta ) + 2 \log ( 1 / \varepsilon )$ is still polylogarithmic. On the success event the costs obey the local bounds of (b); on its complement, which has probability at most $\delta ^ { \prime }$ , they obey the global caps of (a). Hence

$$
\mathbb { E } N \le \tilde { O } \left( ( d + 1 ) \frac { D \left( F ^ { * } + \varepsilon \right) + R ^ { * } } { \varepsilon ^ { 2 } } \right) + \varepsilon ^ { 2 } \tilde { O } \left( \frac { ( d + 1 ) D } { \varepsilon ^ { 2 } } \right) , \qquad \mathbb { E } V \le \tilde { O } \left( D \left( 1 + \frac { F ^ { * } } { \varepsilon } \right) \right) + \varepsilon ^ { 2 } \tilde { O } \left( \frac { D } { \varepsilon } \right) .
$$

The second terms are $\widetilde { O } ( ( d + 1 ) D )$ and $\tilde { O } ( D \varepsilon )$ , and they are absorbed because $D \leq D ( F ^ { * } + \varepsilon ) / \varepsilon ^ { 2 }$ and $D \varepsilon \le D ( 1 + F ^ { \ast } / \varepsilon )$ . This is an expectation argument based on the global caps, not a relabelling of the high-probability statement. □

Proof of Proposition 3.3. (i) Every reset displays at most $M \leq D$ pairs.

(ii) For Procedure 3: the anchor procedure of Lemma 4.5 makes $O ( D \log ( e / \varepsilon ) )$ ownership minimizations and one initial call, and the precision stage makes $S = 4 D$ design calls and one selection call. For Procedure 5: the anchor procedure of Lemma 4.6 makes ${ \widetilde O } ( D )$ calls including initialization, and the I precision stages make $I ( 4 D + 1 ) = O ( D \log ( e / \varepsilon ) )$ calls. In both cases the total is $\tilde { O } ( D ( 1 + \log ( 1 / \varepsilon ) ) )$ ; the term 1 covers the endpoint $\varepsilon = 1$ , where exactly one empty-list call is made although no reset is used.

(iii) In design s the oracle is called on the queries of $m _ { s }$ displays, at most $D m _ { s }$ query occurrences, so by Lemma $\mathrm { { A . 1 } ( i ) }$ it returns at most $( D m _ { s } + 1 ) ^ { d }$ representatives. The final selection call sees at most $D n _ { F }$ queries and returns at most $( D n _ { F } + 1 ) ^ { d }$ representatives. These counts do not bound the memory used by the anchor procedures, the running time of the oracle, or the bit complexity of exact arithmetic.

(iv) This is the last part of Proposition $6 . 7 ( \mathrm { a } )$ and its proof: every pilot of a stage is the anchor or a slot, and enumeration plays nothing by itself; every actual use of a function, in whatever role, is counted in V.

(v) This is the first line of Procedures 3 and 5: width $1 \leq \varepsilon ,$ and $[ 0 , 1 ]$ contains $R ^ { * }$ and $R ( \widehat { h } )$

For $d = 0$ all trace counts equal one and every argument remains valid with the factor $d + 1 ;$ no lower-bound sharpness at $d = 0$ is claimed. □

## G Proofs of the lower bounds

This appendix proves Theorems 3.5 and 3.6. Throughout, d and D are positive integers and the ceilings f and r lie in $[ 0 , 1 ] ;$ individual lemmas add further cutofs. Pilots, displays and outputs are as in Section 2; in particular the output belongs to the class but need not have been played. N is the number of resets and V the number of distinct functions actually played as pilots: computing a mixture, enumerating functions or returning the output plays nothing by itself, and every function actually used as a pilot is counted. For the bounds on $\mathbb { E } V$ we assume, as stated in the theorems, that runs terminate almost surely on every instance of the finite families below; this holds in particular for every learner with deterministic caps.

The hard families below are finite: their classes, query sets and sets of worlds are finite. (The proof of Theorem $3 . 6 ( \mathrm { c } )$ at the end of the appendix is an upper bound and holds for every class satisfying Assumptions 2.5 and 2.6.) Each family uses one known loss, fixed before learning, and its hidden parameters are carried by the truth function and by the record law. In every family an empty record has a fixed constant descriptor. We write $\mathbb { E } ^ { 0 }$ and $\mathrm { P r } ^ { 0 }$ for expectations and probabilities in a comparison experiment.

## G.1 Threshold paths and two information tools

The fresh-bit calculation for threshold paths is standard; related symmetric-counterexample constructions appear in Braverman et al. [10, Example 2]. A block of depth $L \geq 1$ has ordered queries $1 , \ldots , 2 ^ { L } - 1$ and the threshold functions $h _ { \varsigma } ( j ) = \mathbf { 1 } \{ j \geq \varsigma \} , 1 \leq \varsigma \leq 2 ^ { L }$ . A hidden leaf is written

$$
{ \mathfrak { b } } = 1 + \sum _ { s = 1 } ^ { L } B _ { s } 2 ^ { L - s } , \qquad B _ { s } \in \{ 0 , 1 \} ,
$$

and the truthful labels are those of $h _ { \mathfrak { b } }$ . At level s put

$$
o _ { s } = 1 + \sum _ { q < s } B _ { q } 2 ^ { L - q } , \qquad j _ { s } = o _ { s } + 2 ^ { L - s } - 1 .
$$

The record of the block is the binary-search path $j _ { 1 } , \ldots , j _ { L }$ , and the label of $j _ { s }$ is $1 - B _ { s }$ (if $B _ { s } = 0$ then ${ \mathfrak { b } } \leq j _ { s } ,$ and if $B _ { s } = 1$ then ${ \mathfrak { b } } \geq j _ { s } + 1 )$ . The address $j _ { s }$ depends only on $B _ { 1 } , \ldots , B _ { s - 1 } . \ \mathrm { ~ A ~ }$ threshold $h _ { \varsigma }$ agrees with the whole path if and only if $\varsigma = { \mathfrak { b } }$ , because the path separates all $2 ^ { L }$ leaves. By convention, $L = 0$ means that there are no path queries and a single trivial threshold.

The families of Lemmas G.4, G.5 and G.7, and the comparison experiments used in their proofs, have the following structure, which we call an arm family. There is a finite set $\boldsymbol { A } _ { \mathrm { a l l } }$ of arms; each arm a is a block of depth $L _ { a } \ge 0$ on its own queries, with its own hidden leaf. A world consists of the vector B of all leaves and of further parameters $\varphi$ (winners, activities). Under the prior, the leaf bits are independent and uniform and B is independent of $\varphi .$ Each record is generated in two steps. First a type $\iota \in { \mathcal { A } } _ { \mathrm { a l l } } \cup \{ \emptyset \}$ is drawn from a fixed law that is the same in all worlds; the type $\iota = a$ names the arm a, so each arm has exactly one type. Then the record is a deterministic function of $\iota ,$ of $\varphi$ and, if $\iota \neq \varnothing$ , of the leaf of the arm $\iota ,$ of the following form. If $\iota = \varnothing$ , the record is a function of $\varphi . { \mathrm { ~ H ~ } } \iota = a$ , the record consists of an initial part whose queries and labels are functions of $( a , \varphi )$ (for instance a gate query; the initial part may be empty), followed by the binary-search path of the leaf of $^ { a , }$ followed by a final part (further queries and the descriptor) that is a function of $a , \varphi$ and that leaf. Thus in each world every arm has one fixed record. We say that the s-th bit of the leaf of a is displayed when the pair at level s of the path of a is displayed. For the analysis we realize the types by a sequence $\iota _ { 1 } , \iota _ { 2 } , \ldots$ . of independent draws from the type law, made before learning: the n-th reset, if it takes place, has type $\iota _ { n }$ , and the other draws are not used. Let $\Xi$ consist of $\varphi _ { : }$ the whole sequence $( \iota _ { n } ) _ { n \geq 1 }$ and the learner’s private randomness. Under the prior, B, φ, the sequence $( \iota _ { n } ) _ { n \geq 1 }$ and the private randomness are independent; in particular B is independent of $\Xi .$ This realization changes neither the law of the experiment nor what the learner observes.

Lemma G.1 (Fresh bits and charging to distinct pilots). In an arm family:

(a) conditionally $o n \Xi$ together with the displays of the first n resets, for any $n ,$ or together with the complete transcript of a run that terminates, the leaf bits that have not been displayed are independent and uniform;

(b) at a reset of type $a \neq \emptyset$ , the number of bits of a displayed for the first time has conditional expectation at most 2 given $\Xi$ and the earlier displays, and it is 0 if the pilot errs in the initial part of the record of a;

(c) a fixed function displays the same prefix on all resets of the same type $^ { a , }$ so only its first such reset can display new bits of a;

(d) for a set A of arms let $U _ { A }$ be the number of pairs $( a , s )$ with $a \in { \mathcal { A } }$ and $1 \leq s \leq L _ { a }$ such that the s-th bit of a is displayed on at least one reset (each such pair counts once, however often its bit is displayed), and let $X _ { \mathcal { A } }$ be the number of resets of a type $a \in { \mathcal { A } }$ whose pilot g passes the initial part of the record of a and was not the pilot of an earlier reset of type a. Then $\mathbb { E } U _ { \mathcal { A } } \leq 2 \mathbb { E } X _ { \mathcal { A } }$ . In particular $\mathbb { E } U _ { A }$ is at most twice the expected number of resets of a type in $\mathcal { A }$ whose pilot passes the initial part of its record, and at most twice the expected number of pairs $( g , a ) , a \in { \mathcal { A } }$ , such that the played function g passes the initial part of a.

Proof. $\operatorname { F i x } \Xi$ The transcript is then a deterministic function of B: the pilot of each reset is a function of the private randomness and of the earlier displays, the type of the n-th reset is the coordinate $\iota _ { n }$ of $\Xi$ , and the record is a function of the type, of $\varphi$ and of the leaf of the arm named by the $\mathrm { t y p e }$ . We show by induction on $n$ that for every history $h _ { n }$ of the first n displays, the set of leaf vectors that produce $h _ { n }$ is a cylinder: there are numbers $s _ { a } \geq 0$ and values of the first $s _ { a }$ bits of every arm $^ { a , }$ determined by $( \Xi , h _ { n } )$ , such that B produces $h _ { n }$ if and only if its bits take these values. For $n = 0$ take all $s _ { a } = 0$ . Let the claim hold for $n ,$ and let $g$ and ι be the pilot and the type of reset $n + 1$ ; both are determined by $( \Xi , h _ { n } )$ . If $\iota = \varnothing$ , the display is determined by $( \Xi , h _ { n } )$ , so the set for $h _ { n + 1 }$ is the cylinder for $h _ { n }$ or empty. Let $\iota = a$ be an arm. The display of the initial part is determined by $( \Xi , h _ { n } )$ , and if $g$ errs there nothing else is displayed. Otherwise the display continues along the path of $a .$ . At level s it shows the address $j _ { s }$ , which is determined by $B _ { 1 } , \ldots , B _ { s - 1 }$ of $a ,$ , and the label $1 - B _ { s }$ , and it stops at the first level at which $g$ errs. Hence, given $( \Xi , h _ { n } )$ , the displayed path and the first $q$ bits of a determine each other, where $q$ is the last displayed level $( q = L _ { a }$ if g agrees with the whole path). If $g$ agrees with the whole path, the final part is a function of $\Xi$ and of the leaf of $a ,$ , which the displayed path already determines. So the set for $h _ { n + 1 }$ is the cylinder for $h _ { n }$ intersected with the condition that the first $q$ bits of a take the displayed values, again a cylinder or empty. Since B is uniform and independent of $\Xi .$ its conditional law given $\Xi$ and $h _ { n }$ is uniform on the cylinder. The decision to stop after n resets is determined by $( \Xi , h _ { n } )$ , so the same holds given $\Xi$ and a complete transcript with n resets. This proves (a).

For (b), let s bits of a be displayed before reset $n + 1$ . To display $j \geq 1$ new bits, g must pass the initial part and agree with the path at levels $s + 1 , \ldots , s + j - 1$ Given $\Xi , h _ { n }$ and $B _ { s + 1 } , \ldots , B _ { s + i - 1 }$ , the address $j _ { s + i }$ and the prediction of $g$ there are fixed, while $B _ { s + i }$ is uniform by (a); so $g$ agrees at level $s + i$ with conditional probability $1 / 2$ . Hence the number G of new bits satisfies $\operatorname* { P r } ( G \geq j \mid \Xi , h _ { n } ) \leq 2 ^ { - ( j - 1 ) }$ , and $\begin{array} { r } { \mathbb { E } [ G \mid \Xi , h _ { n } ] \le \sum _ { i > 1 } 2 ^ { - ( j - 1 ) } = 2 } \end{array}$

For (c), in a world all resets of type a have the same record, consisting of the fixed initial part, path and final part of $a ,$ , and a fixed function stops at the same position whenever it meets the same record. So if $g$ was the pilot of an earlier reset of type $^ { a , }$ then either it erred in the initial part then and errs there again, or it displays the same prefix again.

For (d), $U _ { A }$ is the sum over resets of the number of bits of arms in A displayed for the first time at that reset. Whether reset $n + 1$ counts in $X _ { A }$ is determined by $( \Xi , h _ { n } )$ , and so is the event that reset $n + 1$ takes place. By (b) and (c), the number of bits of arms in A displayed for the first time at reset $n + 1$ is zero unless the reset counts in $X _ { A }$ , and its conditional expectation given $( \Xi , h _ { n } )$ is at most twice the indicator that it counts. Summing over n by Tonelli’s theorem, which needs no bound on N, gives $\mathbb { E } U _ { \mathcal { A } } \leq 2 \mathbb { E } X _ { \mathcal { A } }$ . The two further bounds follow because $X _ { A }$ is at most each of the two counts. □

For laws $P , Q$ let $\mathrm { K L } ( P \| Q )$ be the relative entropy and $\mathrm { T V } ( P , Q ) = \operatorname* { s u p } _ { A } \left| P ( A ) - Q ( A ) \right|$

Lemma G.2 (Testing with padded transcripts). Suppose two worlds share the learner’s action and stopping rules and the law of its private randomness, and that before stopping the conditional relative entropy of the next observation is at most κ. $I f \mathbb { E } _ { P } N < \infty$ , the complete stopped transcripts, including the output, satisfy

$$
\begin{array} { r } { \mathrm { K L } ( P _ { \mathrm { t r } } \| Q _ { \mathrm { t r } } ) \leq \kappa \mathbb { E } _ { P } N , \qquad \mathrm { T V } ( P _ { \mathrm { t r } } , Q _ { \mathrm { t r } } ) \leq \sqrt { \kappa \mathbb { E } _ { P } N / 2 } . } \end{array}
$$

Moreover, if a hidden vector is uniform on $\{ - 1 , 1 \} ^ { d }$ and the transcript laws of every two vectors that difer in one coordinate are within total variation $\tau _ { 0 }$ , then every decoder has prior-average fraction of incorrect coordinates at least $( 1 - \tau _ { 0 } ) / 2$

Proof. After stopping, fill every later position of the transcript with a fixed symbol. Given a common finite past, the private randomness and the chosen actions have the same conditional kernels in both worlds and contribute no relative entropy; padded positions contribute none either. The chain rule therefore gives, for the first m positions,

$$
\mathrm { K L } ( P _ { [ m ] } \| Q _ { [ m ] } ) \leq \kappa \sum _ { t = 1 } ^ { m } \operatorname* { P r } _ { P } ( N \geq t ) = \kappa \mathbb { E } _ { P } \operatorname* { m i n } ( N , m ) .
$$

Relative entropy on an increasing sequence of $\sigma { \mathrm { - f i e l d s } }$ converges to the relative entropy on the generated σ-field, as follows from its variational formula by approximating bounded measurable functions with their conditional expectations given the finite prefixes. Letting $m  \infty$ gives the first bound. Appending the output is a common randomized kernel of the stopped transcript and cannot increase relative entropy; equivalently, the output randomness can be included in the initial private seed.

For the second bound we recall Pinsker’s inequality. For an event $A ,$ data processing follows from the log-sum inequality applied on A and on its complement. For Bernoulli parameters $p , q ,$ the binary relative entropy has second derivative $1 / ( p ( 1 - p ) ) \geq 4$ in $p$ and vanishes together with its first derivative at $p = q .$ , so it is at least $2 ( p - q ) ^ { 2 }$ . Taking events that approach the total variation gives $\mathrm { K L } \geq 2 \mathrm { T V } ^ { 2 }$ . Finally, for two vectors that difer only in coordinate $i ,$ the probabilities of decoding that coordinate incorrectly under the two laws sum to at least $1 - \tau _ { 0 }$ . Pair the $2 ^ { d }$ vectors along coordinate i and average over i. □

Remark G.3 (Exact dimension and padding). A product of d threshold blocks on disjoint query sets has VC dimension exactly $d \colon$ one query per block is shattered, and two queries of one block cannot be shattered by thresholds. The same holds for a block in which a function selects one arm, accepts that arm’s gate, and is positive on a final segment of that arm’s path queries: within one arm the positive sets are nested, and points of diferent arms cannot both be positive. The product of $d$ such blocks again has dimension exactly d.

If a construction has a class of VC dimension one, add $d - 1$ further queries on which the class takes all $2 ^ { d - 1 }$ combinations of values independently of the rest. The result has VC dimension exactly d: a shattered set restricts to a shattered set on the original domain, so it has at most one original point and at most $d - 1$ added points, and one shattered original point together with all added points is shattered. Fix arbitrary labels on the added queries and never sample them. They reveal no information, and the variants of a function that difer only there count as distinct functions, which can only increase $V ;$ on the supported queries every variant behaves like the original function.

## G.2 A realizable floor for both resources

Lemma G.4 (Rare threshold paths). For all d, $D \geq 1$ and $0 < \varepsilon \le 1 / 1 6$ there is a finite family with the natural loss, a class of VC dimension exactly d and $F ^ { * } = R ^ { * } = 0$ , on which every learner that succeeds with probability at least $4 / 5$ in every world has

$$
\operatorname* { s u p } \mathbb { E } N \geq { \frac { d D } { 4 0 \varepsilon } } , \qquad \operatorname* { s u p } \mathbb { E } V \geq { \frac { D } { 5 } } .
$$

Proof. Take d blocks of depth D on disjoint query sets, with leaves $\mathfrak { f } _ { 1 } , \ldots , \mathfrak { b } _ { d } .$ , and let the class be the product of thresholds. With probability $\theta = 8 \varepsilon \leq 1 / 2$ a record chooses a uniform block and is its path, of length $D ;$ otherwise it is empty. Descriptors are constant. The product of the true thresholds is perfect, so $F ^ { * } = R ^ { * } = 0$ , and a member of the class has risk exactly θ times the fraction of blocks whose threshold is wrong. A successful output therefore has the correct threshold in at least $7 / 8$ of the blocks.

Give the leaves independent uniform priors. This is an arm family whose arms are the d blocks and whose types are the blocks and $\mathcal { O } ,$ with empty initial parts and no further parameters. Runs terminate almost surely: for the reset bound we may assume $\mathbb { E } N < \infty$ , and for the pilot bound this is the convention of Theorem 3.5. Let U be the number of distinct leaf bits displayed, that is, $U _ { A }$ of Lemma G.1(d) for the set $\mathcal { A }$ of all blocks, and J the number of blocks whose path is completely displayed; then $J \le U / D$ . The output is a function of $\Xi$ and of the transcript, so by Lemma G.1(a) the threshold it chooses for an incompletely displayed block is correct with conditional probability at most $1 / 2$ . Hence the expected fraction $C$ of correct blocks satisfies

$$
{ \frac { 7 } { 1 0 } } = { \frac { 4 } { 5 } } \cdot { \frac { 7 } { 8 } } \leq \mathbb { E } C \leq { \frac { 1 } { 2 } } + { \frac { \mathbb { E } J } { 2 d } } \leq { \frac { 1 } { 2 } } + { \frac { \mathbb { E } U } { 2 d D } } ,
$$

so $\mathbb { E } U \ge 2 d D / 5$ . By Lemma G.1(d) applied to all arms, $\mathbb { E } U \le 2 \theta \mathbb { E } N$ , since the type of each reset names an arm with probability $\theta ,$ independently of the event that the reset takes place; and $\mathbb { E } U \le 2 d \mathbb { E } V$ , since a played function and an arm form at most $d V$ pairs. Substituting gives the bounds for the prior means, hence for some world; if an expectation is infinite there is nothing to prove. The argument includes $D = 1$ □

## G.3 Free gates: the natural-ceiling term at zero general optimum

Comparisons between a hidden winner and a null environment are standard information arguments; related contextual combination locks appear in Jia et al. [17]. The proof below checks the precise proper, frozen-pilot interface and its resource accounting.

Lemma G.5 (A diluted free-gate family). Let $0 < \varepsilon \leq 1 / 1 6 , f \geq 1 6 \varepsilon , d , D \geq 1$ , and put

$$
K = \Big \lfloor \frac { f } { 4 \varepsilon } \Big \rfloor \geq 4 , \qquad L = D - 1 , \qquad L _ { + } = \operatorname* { m a x } ( 1 , L ) .
$$

There is a family with a class of VC dimension exactly d, one fixed binary loss, $F ^ { * } = f ( 1 - 1 / K ) \leq f$ and $R ^ { * } = 0$ , on which every learner that succeeds with probability at least $4 / 5$ in every world has

$$
\operatorname* { s u p } \mathbb { E } N \geq \frac { 1 1 d K ^ { 2 } L _ { + } } { 8 0 0 f } , \qquad \operatorname* { s u p } \mathbb { E } V \geq \frac { 1 9 K L _ { + } } { 1 6 0 0 } .
$$

Proof. Class and worlds. Each of d blocks has K arms. Arm $( i , c )$ has a gate query $e _ { i c }$ with label one and a threshold block of depth L with its own hidden leaf ${ \mathfrak { b } } _ { i c } .$ . A member of the class chooses in each block one arm $a _ { i }$ and one threshold, predicts one at the gate of $a _ { i }$ , uses the chosen threshold on that arm, and predicts zero at every query of the other arms of the block. (For $L = 0$ it just chooses a gate.) By Remark G.3 the VC dimension is exactly $d .$

A world specifies one good arm $C _ { i }$ in each block and the leaves of all arms. A record is empty, with the constant descriptor 0, with probability $1 - f ;$ otherwise it chooses $( i , c )$ uniformly from $[ d ] \times [ K ]$ , asks the gate $e _ { i c } ,$ follows the path of arm $( i , c )$ , and has descriptor $\mathbf { 1 } \{ c \neq C _ { i } \}$ . All labels are fixed functions of the queries within a world. The loss is

$$
\lambda = \left\{ \begin{array} { l l } { 0 } & { \mathrm { a t ~ a ~ d i s c r e p a n c y ~ a t ~ t h e ~ g a t e } , } \\ { 1 } & { \mathrm { a t ~ a ~ l a t e r ~ d i s c r e p a n c y } , } \\ { \mathrm { t h e ~ d e s c r i p t o r } } & { \mathrm { a t ~ a ~ s u r v i v a l } , } \end{array} \right.
$$

and zero on the empty record. A gate discrepancy displays no path or descriptor, and a path discrepancy displays its label but not the descriptor. The loss is bounded, prefix-observable and the same for all worlds.

A direct computation gives, for a member h of the class,

$$
R ( h ) = { \frac { f } { d K } } \# \{ i : \ a _ { i } \neq C _ { i } { \mathrm { ~ o r ~ t h e ~ t h r e s h o l d ~ o n ~ } } a _ { i } { \mathrm { ~ i s ~ w r o n g } } \} ,\tag{G.1}
$$

with the threshold condition omitted if $L = 0$ . Hence $R ^ { * } = 0$ . The natural loss is minimized by using a correct threshold on the chosen arm of every block, while the other arms fail at their gates, so $F ^ { * } = f ( 1 - 1 / K )$ . Since $K \leq f / ( 4 \varepsilon )$ , a successful output has the good arm in at least $3 / 4$ of the blocks.

Give the good arms independent uniform priors and the leaf bits independent uniform priors. The family is an arm family whose arms are the pairs $( i , c )$ , whose types are these pairs and $\mathcal { O } ,$ and whose further parameters are the good arms; the initial part of a record of arm $( i , c )$ is its gate, and its final part is its descriptor. For a fixed block $i ,$ the null experiment is the experiment in which every arm of block i has descriptor one, while all other parameters keep their priors; it is again an arm family, with the same arms and types. The null experiment need not satisfy $R ^ { * } = 0$ , so no assumption on the learner’s costs is made there; instead the procedure is capped before it is run in it.

The null coupling. Let a procedure have deterministic caps. Run it in the null experiment for block i with its usual private randomness and record types, draw $C _ { i }$ uniformly and independently, and couple with the original experiment using the same leaves, types and randomness. Let $A _ { i }$ be the number of distinct arms of block i whose whole path and descriptor were displayed in the null run. The two runs coincide until the first completed visit to arm $C _ { i }$ . If $C _ { i }$ is not among the completed arms of the null run, the runs and their outputs coincide. Writing $\widehat { a } _ { i }$ for the arm chosen in block i by the output,

$$
\operatorname* { P r } ( \widehat { a } _ { i } = C _ { i } ) \leq \frac { \mathbb { E } ^ { 0 } A _ { i } + 1 } { K } .\tag{G.2}
$$

The first term bounds the probability that the independent uniform arm $C _ { i }$ is completed in the null run, and the second the probability that the null output names it. We do not claim that $C _ { i }$ is uniform given a miss. The term +1 allows an unplayed final answer.

Expected resets. Let $n = \operatorname* { s u p } \mathbb { E } N$ over the family, and assume it is finite. Stop the procedure after $N _ { 0 } = \lfloor 1 0 n \rfloor$ resets, returning any member of the class on truncation. In every world $\mathrm { P r } ( N >$ $N _ { 0 } ) \le \mathbb { E } N / ( N _ { 0 } + 1 ) \le 1 / 1 0$ , so the truncated procedure succeeds with probability at least $7 / 1 0$ and its prior-mean fraction of correct arms is at least $2 1 / 4 0$ . The cap $N _ { 0 }$ also holds in every null experiment. A member of the class passes the gate of exactly one arm of block $i ,$ and the type of each reset names that arm with probability $f / ( d K )$ , independently of the event that the reset takes place. If $L \geq 1$ , a completed arm needs all its L bits displayed, so Lemma $\mathrm { G . 1 ( d ) }$ , applied to the arms of block i, gives $\mathbb { E } ^ { 0 } A _ { i } \le 2 N _ { 0 } f / ( d K L )$ ; if $L = 0$ , a completed arm needs a matching gate, so $\mathbb { E } ^ { 0 } A _ { i } \le N _ { 0 } f / ( d K )$ . In both cases

$$
\mathrm { P r } ( \widehat { a } _ { i } = C _ { i } ) \leq \frac { 1 } { K } + \frac { 2 N _ { 0 } f } { d K ^ { 2 } L _ { + } } .
$$

Averaging over i and using $2 1 / 4 0 - 1 / K \geq 1 1 / 4 0$ gives $1 0 n \geq N _ { 0 } \geq 1 1 d K ^ { 2 } L _ { + } / ( 8 0 f )$ , which is the reset bound. The floor and the denominator $N _ { 0 } + 1$ avoid an additive rounding loss.

Expected distinct pilots. Let $v = \operatorname* { s u p } \mathbb { E } V$ over the family, and assume it is finite. Stop the procedure immediately before it would play an $( m + 1 )$ -st distinct function, where $m = \lfloor 1 0 v \rfloor$ ; in every world this happens with probability at most $v / ( m + 1 ) \leq 1 / 1 0$ . Impose in addition a common finite reset cap $N _ { 1 }$ with $\mathrm { s u p } _ { \omega } \operatorname* { P r } _ { \omega } ( N > N _ { 1 } ) \leq 1 / 2 0$ over the worlds ω of the family; such an $N _ { 1 }$ exists because the family is finite and every run terminates almost surely. No finite expected reset count is needed. The doubly capped procedure succeeds with probability at least $4 / 5 - 1 / 1 0 - 1 / 2 0 = 1 3 / 2 0$ in every world, so its prior-mean fraction of correct arms is at least $3 9 / 8 0$

In the null experiment for block i, a played function passes the gate of exactly one arm of block $i ,$ and at most m functions are played. Lemma $\mathrm { G . 1 ( d ) }$ , applied to the arms of block $i ,$ therefore gives $\mathbb { E } ^ { 0 } A _ { i } \ \leq \ 2 m / L$ if $L \geq 1 ;$ if $L = 0$ , each distinct pilot can complete at most one arm, so $A _ { i } \le m$ . Hence $\left( \mathrm { G . 2 } \right)$ is at most $1 / K + 2 m / ( K L _ { + } )$ . Averaging and using $3 9 / 8 0 - 1 / K \geq 1 9 / 8 0$ gives $1 0 v \ge m \ge 1 9 K L _ { + } / 1 6 0$ , which is the pilot bound. Dilution changes how long one waits for a matching gate, but not the information obtained per distinct pilot. □

## G.4 A hidden terminal mean with zero natural optimum

Lemma G.6 (The general-risk term). Let $0 < \varepsilon \leq 1 / 1 6 , r \geq$ 16ε and d, $D \geq 1$ . There is a family with a class of VC dimension exactly d, $F ^ { * } = 0$ and $R ^ { * } \leq r ,$ on which every learner that succeeds with probability at least $4 / 5$ in every world has sup $\mathbb { E } N \geq d r / ( 1 6 3 8 4 \varepsilon ^ { 2 } )$

Proof. Let $\Delta = 4 \varepsilon / r \le 1 / 4$ . There are d queries, all with label zero, and the class consists of all $2 ^ { d }$ functions on them. A record is empty with probability $1 - r ;$ otherwise it chooses i uniformly and asks query i. Predicting one there is a discrepancy with loss $1 / 2$ . Predicting zero survives and displays a terminal bit $Y \sim \mathrm { B e r } ( 1 / 2 + \sigma _ { i } \Delta )$ , which is the loss; $\sigma \in \{ - 1 , 1 \} ^ { d }$ is the hidden vector of the world. The empty record has loss zero. The loss rule is fixed: the unknown signs afect the record law, not the loss rule or the labels. The records have length at most one, so the family is admissible for every $D \geq 1$

The all-zero function has no discrepancy, so $F ^ { * } = 0$ . If $m _ { - }$ signs are negative, the optimum predicts one exactly at the positive signs, and

$$
R ^ { * } = \frac { r } { 2 } - \frac { r \Delta m _ { - } } { d } \leq r , \qquad R ( h ) - R ^ { * } = r \Delta \frac { \# \{ i : \ h \operatorname* { d e c i d e s } \operatorname { c o o r d i n a t e } i \operatorname { w r o n g l y } \} } { d } .
$$

Success therefore requires at most $1 / 4$ of the coordinates to be wrong.

Give the learner the complete record, including $Y _ { i }$ , on every reset; this stronger experiment can simulate the stopped one by discarding information. For two sign vectors that difer in one coordinate, the relative entropy of one reset is at most $\begin{array} { r } { ( r / d ) \cdot 2 \Delta \log \frac { 1 + 2 \Delta } { 1 - 2 \Delta } } \end{array}$ in either direction, and

$$
2 \Delta \log \frac { 1 + 2 \Delta } { 1 - 2 \Delta } \leq 1 6 \Delta ^ { 2 } ,
$$

because log $\begin{array} { r } { \frac { 1 + x } { 1 - x } = \int _ { 0 } ^ { x } \frac { 2 d u } { 1 - u ^ { 2 } } \leq \frac { 8 x } { 3 } } \end{array}$ for $x = 2 \Delta \le 1 / 2$ . Let n be the largest expected number of resets over the family. If $n < d / ( 1 0 2 4 r \Delta ^ { 2 } )$ , Lemma G.2 gives total variation less than $\sqrt { 1 / 1 2 8 } < 1 / 8$ for every neighboring pair, and the prior-average fraction of wrong coordinates is at least $7 / 1 6$ . Uniform success bounds that fraction by $\textstyle { \frac { 4 } { 5 } } \cdot { \frac { 1 } { 4 } } + { \frac { 1 } { 5 } } = { \frac { 2 } { 5 } } < { \frac { 7 } { 1 6 } }$ , a contradiction. Hence $n \geq d / ( 1 0 2 4 r \Delta ^ { 2 } ) =$ $d r / ( 1 6 3 8 4 \varepsilon ^ { 2 } )$ . Data-dependent stopping, output randomization and private coins are covered by Lemma G.2. □

Proof of Theorem 3.5. Lemma G.4 applies for every pair of ceilings, because its family has $F ^ { * } =$ $R ^ { * } = 0$ , and gives $\Omega ( d D / \varepsilon )$ resets and $\Omega ( D )$ pilots. If $f \geq 1 6 \varepsilon$ , Lemma G.5 satisfies both ceilings (also when $r = 0 )$ , and since $K \geq f / ( 8 \varepsilon )$ and $L _ { + } \ge D / 2$ it gives $\Omega ( d D f / \varepsilon ^ { 2 } )$ resets and $\Omega ( D f / \varepsilon )$ pilots. If $f < 1 6 \varepsilon$ , these terms are at most sixteen times the realizable ones. If $r \geq 1 6 \varepsilon$ , Lemma G.6 satisfies both ceilings (it has $F ^ { * } = 0 )$ and gives $\Omega ( d r / \varepsilon ^ { 2 } )$ resets. If $r < 1 6 \varepsilon$ , this term is at most $1 6 d D / \varepsilon$ . For each resource take the largest of the applicable lower bounds; the maximum of three (or two) nonnegative numbers is at least a third (or a half) of their sum. This proves Theorem 3.5 with a universal constant. A covering ε-certificate implies $R ( \widehat { h } ) \leq R ^ { * } + \varepsilon ,$ , so the stronger output requirement does not avoid these bounds, and per-world success $1 - \delta$ with $\delta \leq 1 / 5$ implies their hypothesis. They concern ceilings on two separate infima, not prescribed exact optima, and do not supply an optimizing hypothesis (see also Remark 3.7). □

## G.5 Natural discrepancy

For the natural loss $R = F$ , so the two ceilings amount to the single ceiling min $( f , r )$ , which this subsection denotes by r; the free-gate family of Lemma G.5 does not apply to this loss.

Lemma G.7 (Diluted active and null blocks). Let $D \geq 3 , 0 < \varepsilon \leq 1 / 1 2 8$ and $r \geq 1 2 8 \varepsilon$ , and put $k = D - 2 \ge 1$ and $K = \lfloor r / ( 1 6 \varepsilon ) \rfloor \ge 8$ . There is a finite natural-loss family with a class of VC dimension exactly d and $F ^ { * } \leq r$ on which every learner that succeeds with probability at least $4 / 5$ in every world satisfies, for the prior described below,

$$
{ \mathbb E N } \ge \frac { d k K ^ { 2 } } { 1 6 r } , \qquad { \mathbb E V } \ge \frac { k K } { 1 6 } .
$$

Proof. Take d blocks with K gated arms each, every arm having its own threshold block of depth k, and the class of Lemma G.5. Add two queries $t _ { 0 } , t _ { 1 }$ with fixed labels zero and one, at which every member of the class predicts one; these constant coordinates do not change the VC dimension. Each block is either null or active, and an active block has one winning arm $W _ { i }$

A record is empty with probability $1 - r .$ Otherwise it chooses a uniform block-arm pair, asks its gate (label one), follows its path of k queries, and then asks $t _ { 1 }$ if the arm is the winner of an active block, and $t _ { 0 }$ otherwise. The descriptor is constant, and hard records have exactly $k + 2 = D$ queries. All labels are fixed; which final query occurs is part of the record law. The final query is hidden after any earlier discrepancy, including one at the last level of the path. A member of the class survives a hard record exactly when the record’s arm is its chosen arm, its threshold is correct, and the arm is the winner of an active block. Hence, if A blocks are active,

$$
F ^ { * } = r { \bigl ( } 1 - { \frac { A } { d K } } { \bigr ) } , \qquad F ( h ) - F ^ { * } = { \frac { r } { d K } } \# \{ i { \mathrm { ~ a c t i v e : ~ } } h { \mathrm { ~ m i s s e s ~ t h e ~ w i n n e r ~ o r ~ i t s ~ l e a f ~ i n ~ b l o c k ~ } } i \} ,
$$

and in particular $F ^ { * } \in [ r ( 1 - 1 / K ) , r ]$

(G.3)

Give the activities independent Ber $( 1 / 2 )$ priors, the winners of active blocks independent uniform priors, and the leaves independent uniform priors. Call an arm probed when its final query is displayed, which requires surviving its whole path, and let $S _ { i }$ be the set of probed arms of block i. For a fixed $i ,$ couple a run in which block i is forced to be null with a run in which it is forced to be active, with the winner drawn independently of the null run. Before termination the runs can difer only in the final query of the winning arm. Hence, writing $C _ { i }$ for the event that the output has the winner and the correct leaf in block $i ,$

$$
\operatorname* { P r } ( C _ { i } \mid i { \mathrm { ~ a c t i v e } } ) \leq { \frac { \mathbb { E } ^ { 0 } | S _ { i } | + 1 } { K } } ,\tag{G.4}
$$

where $\mathbb { E } ^ { 0 }$ refers to the run with block i forced null. Both versions belong to the original family, so no out-of-promise budget assumption is involved; the coupling is valid for almost surely finite runs, and the +1 allows an unplayed final answer as in (G.2).

Let M be the number of active blocks whose winner or leaf is missed by the output. On the success event (G.3) gives $\mathsf { M } \leq \varepsilon d K / r \leq d / 1 6$ , and always $\mathsf { M } \leq d ,$ so $\mathbb { E } \mathsf { M } \leq d / 1 6 + d / 5 = 2 1 d / 8 0$ Since $\mathbb { E } A = d / 2$ , the expected number of active blocks identified correctly is at least $1 9 d / 8 0$ , that is, $\textstyle \sum _ { i } \operatorname* { P r } ( i$ active, $C _ { i } ) \geq 1 9 d / 8 0$ . Multiplying (G.4) by $K / 2$ and summing over $i ,$

$$
\mathbb { E } \sum _ { i : i \mathrm { ~ i s ~ n u l l } } | S _ { i } | = \frac { 1 } { 2 } \sum _ { i } \mathbb { E } ^ { 0 } | S _ { i } | \geq K \sum _ { i } \operatorname* { P r } ( i \mathrm { ~ a c t i v e } , C _ { i } ) - \frac { d } { 2 } \geq d \Big ( \frac { 1 9 K } { 8 0 } - \frac { 1 } { 2 } \Big ) \geq \frac { d K } { 8 } ,
$$

using $K \geq 8$ . Every probed arm has displayed all k of its bits, so the number U of distinct leaf bits displayed $( U _ { A }$ of Lemma G.1(d) for the set A of all arms) satisfies $\mathbb { E } U \geq d k K / 8$ . On the other hand, the family is an arm family whose arms are the block-arm pairs, whose types are these pairs and $\mathcal { O } _ { : }$ and whose further parameters are the activities and the winners, with the gates as initial parts and the final queries in the final parts; the forced-null and forced-active versions have the same arms and types. A member of the class passes the gate of exactly one arm in each block, and the type of each reset names one of these d arms with probability d $\cdot r / ( d K ) = r / K$ , independently of the event that the reset takes place. Lemma G.1(d) therefore gives $\mathbb { E } U \le ( 2 r / K ) \mathbb { E } N$ and $\mathbb { E } U \le 2 d \mathbb { E } V$ These inequalities prove both bounds; if an expectation is infinite there is nothing to prove. □

Lemma G.8 (Natural resets at horizons one and two). Let $D \in \{ 1 , 2 \} , 0 < \varepsilon \leq 1 / 1 6$ and $r \geq 1 6 \varepsilon$ There is a natural-loss family with a class of VC dimension exactly d and $F ^ { * } \leq r$ , on which every learner that succeeds with probability at least $4 / 5$ in every world has sup $\mathbb { E } N > d r / ( 4 0 9 6 \varepsilon ^ { 2 } )$

Proof. Let $\gamma = 4 \varepsilon / r \le 1 / 4$ . Block i has two queries $x _ { i 0 } , x _ { i 1 }$ with labels zero and one, fixed and known in all worlds. A member of the class predicts one common bit on both queries of each block, independently across blocks; the VC dimension is exactly d. A world has a hidden vector $\varpi \in \{ 0 , 1 \} ^ { d }$ . With probability r a record selects a uniform block i and then the query $x _ { i \varpi _ { i } }$ with probability $1 / 2 + \gamma$ , and the other query otherwise; with the remaining probability the record is empty. Records have length at most one, and every pilot sees the query and its label, even when it errs.

The best prediction in block i is $\varpi _ { i } .$ , and

$$
F ^ { * } = r { \Bigl ( } { \frac { 1 } { 2 } } - \gamma { \Bigr ) } \leq r , \qquad F ( h ) - F ^ { * } = 2 r \gamma { \frac { { \mathcal { H } } \{ i : \ h ( x _ { i 0 } ) \neq \varpi _ { i } \} } { d } } .
$$

A uniformly successful learner therefore has expected wrong fraction at most $\varepsilon / ( 2 r \gamma ) + 1 / 5 = 1 3 / 4 0$ Neighboring worlds have one-reset relative entropy at most $1 6 r \gamma ^ { 2 } / d ,$ by the Bernoulli computation in Lemma G.6. If every world had $\mathbb { E } N \le d / ( 2 5 6 r \gamma ^ { 2 } )$ , Lemma G.2 would give neighboring total variation at most $\sqrt { 1 / 3 2 } < 1 / 4$ , and the prior-average wrong fraction would exceed $3 / 8 > 1 3 / 4 0$ , a contradiction. Hence some world has $\mathbb { E } N > d / ( 2 5 6 r \gamma ^ { 2 } ) = d r / ( 4 0 9 6 \varepsilon ^ { 2 } )$ . Properness is essential: a function outside the class could predict the known label on both queries. □

Lemma G.9 (Natural hidden arms at horizon two). Let $d \geq 1 , 0 < \varepsilon \leq 1 / 1 6$ and $r \geq 8 \varepsilon$ . There is a finite natural-loss family of horizon two with a class of VC dimension exactly d and $F ^ { * } \leq r$ , on which every learner that succeeds with probability at least 4/5 in every world has sup $\mathbb { E } V \geq r / ( 1 6 0 \varepsilon )$

Proof. Let $K = \lfloor r / ( 2 \varepsilon ) \rfloor \ge 4$ . The supported class consists of $h _ { 1 } , \ldots , h _ { K }$ , where $h _ { i }$ predicts one at $x _ { i } ,$ , zero at $x _ { j }$ for $j \neq i$ , and one at every second query $z _ { j }$ . In world $a \in [ K ]$ all $x _ { j }$ have label one and $z _ { j }$ has label ${ \bf 1 } \{ j = a \}$ . A record is empty with probability $1 - r ;$ otherwise it chooses a uniform $j$ and asks $x _ { j }$ and then $z _ { j }$ . Descriptors are constant. The supported class has VC dimension one: a single first query is shattered, two first queries cannot both be positive, and the second queries are constant on the class. Apply the padding of Remark G.3 to obtain dimension exactly d.

Only $h _ { j }$ and its padded variants pass the first query of arm $j ,$ so $F ( h _ { a } ) = r ( 1 - 1 / K )$ and $F ( h _ { i } ) = r$ for $i \neq a .$ . Since $r / K \geq 2 \varepsilon$ , success requires an output whose supported part is $h _ { a } .$

Grant the learner, at the first selection of each arm, the answer to the question whether this arm is $^ { a , }$ whether or not a matching record occurs. This dominates legal play: record types can be simulated independently, and the only world-dependent displayed label is that of the second query of a selected arm. Let $Q$ be the number of distinct arms selected; then $Q \leq V$ , also with padded variants. Fix all learner and simulation coins. A procedure that probes at most $u < K$ arms succeeds in at most $u + 1$ of the $K$ worlds: on its all-no path it probes at most u arms, and every world whose winner is not among them follows the all-no path and receives the same output, which names at most one further arm. Almost-sure termination on the finite family makes this valid outside a common null set of coins. Averaging over the coins and truncating immediately before the $( u + 1 )$ -st new arm, which changes the run only if $Q > u$ , gives under the uniform prior on the winner

$$
\operatorname* { P r } ( { \mathrm { s u c c e s s } } ) \leq { \frac { u + 1 } { K } } + \operatorname* { P r } ( Q > u ) \leq { \frac { u + 1 } { K } } + \operatorname* { P r } ( V > u ) .
$$

Take $u \ = \ \lfloor K / 2 \rfloor$ . For $K \geq 4$ we have $( u + 1 ) / K \le 3 / 4$ , so success probability $4 / 5$ forces $\operatorname* { P r } ( V > u ) \geq 1 / 2 0$ and $\mathbb { E } V \geq ( u + 1 ) / 2 0 \geq K / 4 0 \geq r / ( 1 6 0 \varepsilon )$ , where the last step uses $K \geq r / ( 4 \varepsilon )$ The final output is not charged as a training pilot. □

Proof of Theorem 3.6. (a) For $D \geq 3$ and $r \geq 1 2 8 \varepsilon$ , Lemma G.7 with $K \geq r / ( 3 2 \varepsilon )$ and $k = D - 2 \geq$ $D / 3$ gives $\Omega ( d D r / \varepsilon ^ { 2 } )$ resets. For $D \in \{ 1 , 2 \}$ and $r \geq 1 6 \varepsilon$ , Lemma G.8 gives $\Omega ( d r / \varepsilon ^ { 2 } ) = \Omega ( d D r / \varepsilon ^ { 2 } )$ resets. In the remaining cases $r + \varepsilon \le 1 2 9 \varepsilon$ , and Lemma G.4, whose family has $F ^ { * } = 0$ , gives $d D / ( 4 0 \varepsilon ) \geq d D ( r + \varepsilon ) / ( 5 1 6 0 \varepsilon ^ { 2 } )$ . Taking the maximum with the realizable floor gives the sum $d D ( r + \varepsilon ) / \varepsilon ^ { 2 }$ up to a constant. All lemmas used are valid in the common range $\varepsilon \le 1 / 1 2 8 ;$ some of them hold in larger ranges.

(b) For $D \geq 3$ and $r \geq 1 2 8 \varepsilon$ , Lemma G.7 gives $\mathbb { E } V \ge k K / 1 6 = \Omega ( D r / \varepsilon )$ . For $D = 2$ and $r \geq 8 \varepsilon$ Lemma G.9 gives $\Omega ( r / \varepsilon ) = \Omega ( D r / \varepsilon )$ . In the remaining cases $1 + r / \varepsilon \le 1 2 9$ , and the realizable floor $D / 5$ sufices. Taking the maximum with $D / 5$ gives the claim.

(c) Let $D = 1$ . Every nonempty record displays its single query and its label whatever the pilot, and empty records have loss zero, so on every reset the natural loss of every candidate is observed. Play one fixed member of the class on all resets, for instance an empty-list representative. The observations form an independent sample, and the natural losses of all candidates are functions of it. Let ${ \widehat { F } } _ { n }$ be the empirical natural loss. For $n t ^ { 2 } \geq 2$ 2

$$
\operatorname* { P r } \Bigl ( \operatorname* { s u p } _ { h \in \mathcal { H } } | \widehat { F } _ { n } ( h ) - F ( h ) | > t \Bigr ) \leq 4 ( 2 n + 1 ) ^ { d } e ^ { - n t ^ { 2 } / 8 } .\tag{G.5}
$$

Indeed, on the event in (G.5) select a witness of the deviation, measurably by Assumption $2 . 6 ;$ an independent copy of the sample has empirical mean within $t / 2$ of the witness’s mean with probability at least $1 / 2 { \mathrm { . } }$ , by Chebyshev’s inequality and variance at most $1 / ( 4 n )$ . So the probability in (G.5) is at most twice the probability that the two empirical means difer by more than $t / 2$ for some candidate. Conditionally on the n pairs of observations, exchange the two members of each pair independently. For one pattern of losses, Hoefding’s inequality for the resulting independent signed diferences bounds this event by $2 e ^ { - n t ^ { 2 } / 8 }$ . By Lemma $\mathrm { A . 1 ( i ) }$ there are at most $( 2 n + 1 ) ^ { d }$ patterns of the class on the pooled queries (empty records add none), and a union bound proves (G.5).

Take $t = \varepsilon / 2$ and n finite with $n \geq 8 / \varepsilon ^ { 2 }$ and $4 ( 2 n + 1 ) ^ { d } e ^ { - n \varepsilon ^ { 2 } / 3 2 } \leq \delta$ . Let $q = \operatorname* { m i n } _ { h } \widehat { F } _ { n } ( h )$ attained by a member returned by the oracle on the displayed queries. On the complementary event $\begin{array} { r } { \operatorname* { s u p } _ { h \in \mathcal { H } } | \widehat { F } _ { n } ( h ) - F ( h ) | \leq t } \end{array}$ , which has probability at least $1 - \delta .$ , both $F ^ { * }$ and the risk of the minimizer lie in $[ \operatorname* { m a x } ( 0 , q - t ) , \operatorname* { m i n } ( 1 , q + t ) ]$ ; for $F ^ { * }$ use a sequence of functions whose risks approach

$F ^ { * }$ . This interval has width at most $2 t = \varepsilon$ on every history, and the learner uses one training pilot. A nonzero constant pilot floor is still necessary in the worst case, by Lemma G.4 with $D = 1 . \quad \bigsqcup$

Remark G.10 (Boundary cases). For general losses the f term survives at $r = 0 \mathrm { : }$ the free-gate family has an attained zero general optimum and a positive natural optimum. At $f = r = 0$ the remaining scales are $d D / \varepsilon$ resets and D pilots. For the natural loss, $r = 0$ forces a zero natural infimum, which is attained in the finite classes used here, so there is no agnostic $f$ term to import. No lower-bound assertion is made for $d = 0$ (see Remark 3.7), for $D = 0$ or for accuracies outside the stated lower-bound ranges, and the reset lower bounds are not lower bounds on the number of displayed labels. The constructions establish the stated proper, frozen-pilot, stopped-record lower bounds; we do not claim novelty of their individual information-theoretic ingredients.