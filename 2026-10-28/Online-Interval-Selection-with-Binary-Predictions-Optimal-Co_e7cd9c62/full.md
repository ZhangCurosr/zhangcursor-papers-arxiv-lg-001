# Online Interval Selection with Binary Predictions: Optimal Consistency and a Continuous Robustness Tradeoff

Zhong Han <sup>1</sup>

## Abstract

We study online interval selection with binary predictions of membership in a final optimal solution. Requests arrive in arbitrary order; accepted requests may be revoked, but rejected or revoked requests cannot be recovered. With rewards proportional to interval length, the best consistency under finite strict robustness is $d = ( 3 + { \sqrt { 5 } } ) / 2$ when both guarantees hold almost surely on each fixed input, including for randomized algorithms. A continuous lower bound forces consistency strictly above d at every robustness budget below $4 \varphi + 3 \approx 9 . 4 7 2 .$ , whereas our deterministic algorithm attains d with exact robustness $\varphi ^ { 5 } \approx 1 1 . 0 9 0$ , where $\varphi = ( 1 + { \sqrt { 5 } } ) / 2$ . Each held interval inherits a ledger recording positive prediction mass lost through irrevocable decisions. A tunable replacement cost gives a continuous family with exact consistency throughout and exact robustness on two parameter intervals, including $( C , R ) \ : = \ : ( 2 5 / 9 , 1 0 )$ and (4, 7). Randomization permits expected consistency approaching one with finite expected robustness, whereas almost-sure robustness forces expected consistency strictly above two at every finite budget.

## 1. Introduction

An online selection algorithm can revoke accepted requests but cannot recover rejected alternatives. Even perfect predictions leave a dilemma: a short request may belong to the final optimum because compatible requests arrive later, yet accepting it can be disastrous if the input stops. How much can perfect predictions help while preserving finite robustness?

In the binary optimum-membership model of Karavasilis (2025), optimal consistency under two almost-sure guarantees is $d = \varphi ^ { 2 } ,$ , below the limit 3 for every 1-increasing policy with finite robustness (Karavasilis, 2025, Definition 4.8 and Lemma 4.9). Such a policy never accepts an interval shorter than its longest conflicting holder. Shaped-BANK attains d with robustness $\varphi ^ { 5 }$ , whereas every budget below $4 \varphi + 3$ forces consistency strictly above d.

Why per-execution protection? Consider provisional reservations for one resource: offers specify future slots, decisions precede service, and declined offers expire. When accepted reservations can still be cancelled and hourly values are equal, duration is the selection objective. A guarantee for each completed batch protects against a poor random outcome as well as incorrect advice. Appendix E treats proportional cancellation costs.

Our lower bounds cover randomized algorithms with both ratios almost surely on each fixed input; deterministic Shaped-BANK attains optimal consistency. Mixing a prediction follower with a robust algorithm improves expected ratios but can return zero reward with positive probability. Almost-sure robustness with expected consistency has different necessary bounds and an open optimum (Section 7).

Remembering missed predictions. When a positive request is rejected, BANK distributes its length among the ledgers of conflicting holders. A replacement inherits those ledgers, so the same held interval can require different future decisions depending on what has already been missed. Bounding this inherited mass permits shorter replacements while protecting both guarantees. Varying the replacement cost and both special thresholds gives a continuous tradeoff; Section 6 identifies its exact segments.

## 1.1. Related work

Consistency–robustness tradeoffs quantify the benefit of accurate advice together with worst-case protection when it is wrong (Mitzenmacher & Vassilvitskii, 2020; Lykouris & Vassilvitskii, 2018; Purohit et al., 2018; Angelopoulos et al., 2020; Wei & Zhang, 2020). Here even exact optimum-membership advice cannot recover rejected or revoked alternatives. Optimal consistency with a matching lower bound, an attainable budget curve and budget lower bounds quantify this irreversible decision constraint. In our model, Karavasilis (2025, Observation 4.6) excludes consistency one by early stopping; Definition 4.8 and Lemma 4.9 give $C \ \geq \ 3$ for every finitely robust 1- increasing policy, including Revoke-Proportional (RP), and Theorem 4.10 certifies $C ( \mathrm { R P } _ { \lambda } )  \leq 3 \lambda / ( \lambda - 1 )$ . Revoke-Prop-Half permits shorter positive replacements without a matching guarantee. BANK makes such replacements finitely robust below three and matches a full-class consistency lower bound.

Replacement charging appears in any-order interval selection (Borodin & Karavasilis, 2023). Karavasilis identified dependence between direct and transferred charges as possible slack. Our joint geometric bound determines exact robustness for two BANK parameter intervals and unchanged RP; source measures control mass lost through rejections.

Antoniadis et al. (2026) also use binary optimum advice and length rewards, but require release-time order and irrevocable acceptance. TrustGreedy (Boyar et al., 2026) maximizes cardinality, receives a predicted full input set before arrivals, and cannot revoke accepted intervals; its replacements update the unaccepted part of a plan. Other models permit fractional allocation, resumption, recall or unit profits; Appendix R.1 distinguishes these scopes.

## 2. Model and Main Guarantees

A request $I = [ s ( I ) , f ( I ) )$ has a unique identity, length $w ( I ) ~ = ~ f ( I ) ~ - ~ s ( I ) ~ > ~ 0 ,$ , and a fixed prediction $p ( I ) \ \in \ \{ 0 , 1 \}$ revealed on arrival. Positive-length overlap means conflict; endpoint contact is compatible. A feasible set contains pairwise compatible requests, with $\begin{array} { r } { w ( X ) = \sum _ { I \in X } w ( I ) } \end{array}$ . Distinct identities may have coincident intervals.

Requests arrive in arbitrary order. Each arrival is immediately accepted or permanently rejected, maintaining a feasible held set $S _ { t }$ . Held requests may be revoked; rejected and revoked identities cannot be recalled, but their information may be retained. Future arrivals, their total number and termination time are unknown; any finite prefix may be complete.

Definition 1 (Strict consistency and robustness). For a nonempty finite labelled input σ, let P be its positive requests, ${ \mathrm { O P T } } = \operatorname* { m a x } _ { X { \mathrm { ~ f e a s i b l e } } } w ( X )$ and $\mathrm { A L G } = w ( S _ { \mathrm { f i n a l } } )$ Predictions are perfect when P equals one final optimum, allowing ties. For a possibly randomized algorithm and inputs fixed independently of its coins, define

$$
\begin{array} { r l } & { R _ { \mathrm { a s } } ( A ) = \underset { \sigma } { \operatorname* { s u p } } \exp \mathrm { e s s } \operatorname* { s u p } \frac { \operatorname { O P T } ( \sigma ) } { \operatorname { A L G } _ { A } ( \sigma ) } , } \\ & { C _ { \mathrm { a s } } ( A ) = \underset { \sigma : P \operatorname { o p t i m a l } } { \operatorname* { s u p } } \exp \frac { \operatorname { O P T } ( \sigma ) } { \operatorname { A L G } _ { A } ( \sigma ) } . } \end{array}\tag{1}
$$

The essential supremum is over the coins. A zero denominator gives +∞; there is no additive term. For deterministic algorithms write R, C instead of $R _ { \mathrm { a s } } , C _ { \mathrm { a s } }$

Thus a finite budget B gives, on every fixed labelled prefix $\tau ,$

$$
\operatorname* { P r } \{ \mathrm { O P T } ( \tau ) \le B \mathrm { A L G } _ { A } ( \tau ) \} = 1 .\tag{2}
$$

The null set may depend on τ . Predictions may be infeasible.

Write $\alpha = 1 / \varphi , d = \varphi ^ { 2 }$ , and $M = \varphi ^ { 3 } ,$ . Define

$$
\begin{array} { r } { C ^ { * } ( B ) = \underset { A \mathrm { d e t e r m i n i s t i c : } } { \operatorname* { i n f } } C ( A ) \leq B ^ { C } ( A ) , } \\ { B _ { \operatorname* { m i n } } ( d ) = \underset { A \mathrm { d e t e r m i n i s t i c : } } { \operatorname* { i n f } } R ( A ) = d } \end{array}
$$

with inf $\varnothing ~ = ~ + \infty$ Define $C _ { \mathrm { a s } } ^ { * } ( B )$ and $B _ { \mathrm { m i n } } ^ { \mathrm { a s } } ( d )$ analogously using possibly randomized algorithms and $C _ { \mathrm { a s } } , R _ { \mathrm { a s } } .$ The class defining $C ^ { * } ( B )$ is empty below M: all-zero labels transfer the classical prediction-free lower bound (Tomkins, 1995). The LR rule of Garay et al. (1997), in the interval formulation of Karavasilis (2025, Lemma 4.5), has $C = R = M$ because it ignores predictions.

Theorem 1 (Optimal consistency under almost-sure guarantees). Allowing randomized algorithms,

$$
\operatorname* { m i n } _ { A : R _ { \mathrm { a s } } ( A ) < \infty } C _ { \mathrm { a s } } ( A ) = d .\tag{3}
$$

The deterministic Shaped-BANK T (Section 4) attains the minimum, also under the stronger requirement that both guarantees hold on every execution.

Corollary 1 (Bounds on its robustness cost). The universal budget and the Shaped-BANK budget satisfy

$$
4 \varphi + 3 \leq B _ { \operatorname* { m i n } } ^ { \mathrm { a s } } ( d ) \leq B _ { \operatorname* { m i n } } ( d ) \leq R ( T _ { * } ) = B _ { * } : = \varphi ^ { 5 } .\tag{4}
$$

The remaining factor is $\varphi ^ { 5 } / ( 4 \varphi + 3 ) \approx 1 . 1 7 1$ for either class.

Table 1 and Figure 1 distinguish universal bounds from attainments. Theorems 2, 5, and 6 prove Theorem 1; Theorem 4 gives the corollary’s lower endpoint. Naive accepts a positive request exactly when it conflicts with no previously accepted positive, and never revokes; it rejects all negative requests.

## 3. Universal Lower Bounds

Theorem 2. Every deterministic algorithm with finite robustness has $C ( A ) \geq d .$ For each such algorithm and every $\epsilon > 0 .$ , a finite perfectly labelled input with a unique optimum gives ratio greater than $d - \epsilon .$

Proof. Perfect labels concern the final input; a stopping prefix may have incorrect labels. Fix a finite robustness bound R. Let $t = \alpha$ and choose $0 < \delta < \operatorname* { m i n } \{ t / 2 , t \epsilon / 2 \}$ Set $s = 1 - 2 \delta$ , take an integer $m > R s ,$ , and partition [δ, 1 − δ) into m positive tiles of length $\ell = s / m < 1 / R .$

![](images/3e6cd5695789d1c1eb0ca99e2de01dc1d5abb68c699b1441a5fb3cca1af536bd.jpg)

a  
Figure 1. Known performance bounds. (a) The universal lower envelope is $\operatorname* { m a x } \{ d , G ( B ) , H ( B ) \}$ ; the attainable envelope combines LR with $T _ { a } .$ . Solid ${ \bar { \mathbf { \Gamma } } } _ { T _ { a } }$ segments have exact robustness, dotted segments only a certificate; the shaded gap is unresolved. Near $B = 6 . 6 5 .$ where $G$ meets $d + 1 / 2 0$ , the lower envelope switches to H, which is proved only for $\gamma \leq 1 / 2 0 .$ . The dashed LR/RP/Shaped-BANK envelope excludes interior $T _ { a }$ parameters; its jump at $B = \varphi ^ { 5 }$ is Shaped-BANK attaining d. (b) At $C = d ,$ the three algorithm budgets are exact, while the universal minimum remains in the marked interval. Both panels concern almost-sure guarantees.  
Table 1. Principal guarantees. First block: also valid for randomized algorithms with both guarantees almost surely, using $C _ { \mathrm { a s } } , R _ { \mathrm { a s } }$ . Second: deterministic attainments. LR is due to Garay et al. (1997); RP’s consistency bound to Karavasilis (2025). The last row instead uses expected reward on fixed inputs.
<table><tr><td>Scope</td><td>Guarantee</td></tr><tr><td>Finite almost-sure robustness</td><td> $C _ { \mathrm { a s } } \geq d$ </td></tr><tr><td>Budget  $R _ { \mathrm { a s } } \leq B$  Optimal  $C _ { \mathrm { a s } } = d$ </td><td> $C _ { \mathrm { a s } } \stackrel { - } { \geq } \operatorname* { m a x } \{ d , G ( B ) , H ( B ) \}$   $R _ { \mathrm { a s } } \geq 4 \varphi + 3$ </td></tr><tr><td>LR (prediction-free)</td><td> $C = R = M$ </td></tr><tr><td>SUM-BANK at  $\lambda = M$ </td><td> $C = d , R = 7 \varphi + 4$ </td></tr><tr><td>Shaped-BANK</td><td> $C = d , R = \varphi ^ { 5 }$ </td></tr><tr><td>Parametric  $T _ { a } , \alpha \leq a \leq 1$ </td><td> $C ~ = ~ ( 1 + a ) ^ { 2 } , ~ R ~ \leq ~ K _ { a } ;$ </td></tr><tr><td></td><td>equality outside  $( a _ { - } , a _ { + } )$  (The- orem 7)</td></tr><tr><td> $\mathrm { R P } , \lambda \geq 2$ </td><td> $C \le 3 \lambda / ( \lambda - 1 ) ; R = 2 \lambda + 2$ </td></tr><tr><td>Initial Naive/LR mixture,  $1 < c \le M$ </td><td> $C _ { \mathrm { e x p } } = c , R _ { \mathrm { e x p } } = 2 \varphi c / ( c - 1 )$ </td></tr></table>

First offer the negative unit interval $J = [ 0 , 1 )$ , then the tiles. Robustness forces acceptance of J. At the first abandonment of $J ,$ only the arriving tile can remain: earlier tiles have been rejected and cannot be recalled. Stopping there gives $\mathrm { O P T } \geq 1$ and $\mathrm { A L G } \leq \ell < 1 / R ,$ , a contradiction. Thus every tile is rejected and J remains. This argument uses robustness, since the positive tiles alone are not yet

optimal.

Offer the positive interval ${ \cal L } = [ \delta - t , \delta )$ . If it is accepted, stop. The positives are compatible and have weight $s + t >$ 1; every positive conflicts with the sole negative J. They therefore form the unique optimum. The algorithm holds at most L, giving ratio at least

$$
( s + t ) / t = d - 2 \delta / t > d - \epsilon .
$$

If L is rejected, offer the positive unit interval $U = [ 1 -$ $\delta , 2 - \delta )$ and stop. The full positives again form the unique optimum, now of weight $s { + } t { + } 1$ . The algorithm can retain only J or U, so its value is at most one and its ratio is at least $d - 2 \delta > d - \epsilon$ . Emptying the held set only worsens either bound.

Determinism fixes the decision on L from the shared prefix, so the appropriate finite input can be chosen for A without changing any label. Taking the supremum proves the result.

The same argument applies to two almost-sure guarantees: fix the finite tree of stopping and perfect terminal inputs, and intersect their probability-one guarantee events. A tape in that intersection would survive every branch, contradicting the deterministic argument. All lower bounds below have such finite trees and therefore also hold with $C _ { \mathrm { a s } } , R _ { \mathrm { a s } } , C _ { \mathrm { a s } } ^ { * }$ in place of $C , R , C ^ { * }$ (Appendix L.4). Tile rejection relies on unknown termination, strict ratios, and no recall. Figure 2 illustrates the two terminal branches. $\mathsf { A p - }$ pendix A gives bounded-size rational witnesses for fixed R.

Theorem 3 (A budget-dependent lower curve). For every deterministic algorithm with $R ( A ) \leq B < \infty$ and $B \geq$ M,

$$
C ( A ) \geq \operatorname* { m a x } \{ d , G ( B ) \} , \qquad G ( B ) = \frac { B } { ( \sqrt { B } - 1 ) ^ { 2 } } .\tag{5}
$$

A finite binary-reserve tree amplifies missed positive mass under either response to a negative reserve, until a perfect terminal violates consistency (Appendix B). For example, $C ^ { * } ( 5 ) > 3 . 2 7$ and $C ^ { * } ( 6 ) > 2 . 8 5$

The robustness cost remains separated from d even when consistency only approaches its optimum.

Theorem 4 (A continuous cost barrier). Put $\beta = 4 \varphi +$ 3. Every deterministic algorithm with finite robustness and $C ( A ) \leq d$ has $R ( A ) \geq \beta .$ . More generally,

$$
C ^ { * } ( B ) \geq d + \operatorname* { m i n } \left\{ { \frac { 1 } { 1 0 0 } } , { \frac { \beta - B } { 8 3 \varphi + 5 3 } } \right\} > d \quad ( B < \beta ) .\tag{6}
$$

Proof sketch: missed sources and a longer reserve. Figure 3 shows four forced decisions: retain the negative anchor $K ,$ , reject the right reserve $R _ { 0 } .$ , switch to a positive unit $J _ { + }$ (called J in the figure), then reject $I , L .$ Tiny positive tiles force retention of K, of length near $\varphi .$ Accepting $R _ { 0 }$ , of length near $\varphi ^ { 2 }$ , permits a perfect completion violating consistency. To force $J _ { + }$ , a remote anchor and boundary requests repair full-history optimality after either response; analogous completions exclude $I , L$ . The realized history contains a compatible packing of limiting weight $\varphi ^ { 3 } + \stackrel { \cdot } { 1 } + \varphi + \varphi ^ { 2 } = \beta$ while only $J _ { + }$ survives. Hypothetical completion requests are not counted.

For $C \leq d + \gamma$ , suppose $R ( A ) < D _ { 1 } ( \gamma )$ and choose r strictly between them. A finite remote anchor of length h consumes consistency slack $\gamma h$ . Choosing $\begin{array} { r } { \boldsymbol { h } \ = \ \left( r \ - \right. } \end{array}$ $1 ) a + \varepsilon$ after a hypothetical switch to held value at most a, and reducing the initial anchor, preserves the four forcing steps. Appendix C supplies finite parameters and strict full-history optimality certificates. Its implicit bound $D _ { 1 } ( \gamma )$ satisfies $D _ { 1 } ( 0 ) ~ = ~ \beta$ and $D _ { 1 } ( \gamma ) ~ \geq ~ \beta ~ - ~ ( 8 3 \varphi + 5 3 ) \gamma$ on $[ 0 , 1 / 1 0 0 ]$ ; approaching equality budgets from below proves (6).

The stronger implicit bound gives $D _ { 1 } ( . 0 0 5 ) \approx 8 . 6 8 0$ and $D _ { 1 } ( . 0 1 ) \approx 8 . 0 8 8$ . A shorter reserve gives $D _ { \mathrm { s } }$ on $0 \leq \gamma \leq$ $1 / 2 0$ , useful farther from d: $D _ { \mathrm { s } } ( . 0 2 ) \ > \ 7 . 4 1$ . Figure 1 combines them as $H ( B ) = \operatorname* { m a x } \{ H _ { \mathrm { s } } ( B ) , H _ { \mathrm { l } } ( B ) \}$ , where

$$
H _ { j } ( B ) = d + \operatorname* { s u p } \bigl ( \{ 0 \} \cup \{ \gamma \in [ 0 , \gamma _ { j } ] : B < D _ { j } ( \gamma ) \} \bigr ) ,
$$

with $\gamma _ { \mathrm { s } } = 1 / 2 0$ and $\gamma _ { 1 } = 1 / 1 0 0$

## 4. The BANK Mechanism

Each held request has a nonnegative ledger $b ( J )$ . A positive arrival adds its length once: acceptance inherits revoked ledgers, while rejection distributes the length among conflicts. Negative rejection adds nothing. Ledgers preserve prediction mass lost through irrevocable decisions.

For conflicts $D ,$ , write $W = w ( D )$ and $\begin{array} { r } { b _ { D } = \sum _ { J \in D } \bar { b } _ { J } } \end{array}$ where $\bar { b } _ { J }$ is the snapshot before this arrival. The ordinary gate $\mathcal { Q } ( D )$ is a sum of replacement costs. SUM-BANK uses cost $\lambda w ( J ) \left( \lambda = M \right.$ unless stated) for every conflict. Scalar Credit-BANK uses min $\{ M v , d v + c b \}$ for a negative conflict, with a fixed coefficient $c \geq 0 .$ . Shaped-BANK retains cost Mv for a positive conflict of length $v ,$ but uses

$$
q _ { * } ( v , b ) = \operatorname* { m i n } \{ 2 \varphi v , \operatorname* { m a x } \{ d v , \varphi v + \alpha b \} \}\tag{7}
$$

for a negative conflict. Thus $\begin{array} { r } { \mathcal { Q } ( D ) = \sum _ { J \in D } \tau ( J ) } \end{array}$ , with $\tau ( J ) = M w ( J )$ or $q _ { * } ( w ( J ) , b ( J ) )$ according to its label. The negative cost stays at dv up to $b = \varphi v$ , increases linearly up to $b = d v .$ , and then remains at $2 \varphi v$

After ordinary acceptance fails, a positive request can still replace all its conflicts if they are negative and

$$
w \ge \alpha W , \qquad w \ge b _ { D } / \varphi .
$$

The first test permits replacing longer conflicts, unlike 1- increasing policies. More generally, an a-increasing rule requires $w \geq a \operatorname* { m a x } _ { J \in D } w ( J )$ when accepting with $D \neq \varnothing ;$ Shaped-BANK satisfies this for $a \ = \ \alpha .$ . The second prevents the same missed positive mass from being discarded repeatedly. Ordinary acceptance has priority; all thresholds accept equality.

For a rejected positive, put $Z = [ \forall J \in D : p ( J ) = 0 ]$ . The simultaneous fees are

$$
g _ { J } = \left\{ \begin{array} { l l } { w \ : w ( J ) / W , } & { Z , \ : w < \alpha W , } \\ { w \ : \bar { b } _ { J } / b _ { D } , } & { Z , \ : w \geq \alpha W , \ : w < b _ { D } / \varphi , } \\ { w \ : w ( J ) / W , } & { \lnot Z . } \end{array} \right.\tag{8}
$$

These cases are exhaustive on rejection; $\neg Z$ means that a positive conflict exists. The compatible-positive filter keeps a positive label only if its request is compatible with all earlier kept positives (Appendix K). For fixed parameters, this implementation uses $O ( n ^ { 2 } )$ exact arithmetic operations/comparisons and $O ( n )$ stored scalars or interval records, including filtering (Appendix O.4).

Lemma 1 (Conservation). For arbitrary labels and each time t, $\textstyle \sum _ { J \in { \mathcal { S } } _ { t } } b ( J ) = w ( P _ { t } )$ , where $P _ { t }$ is the positive set revealed by time t. All ledgers arefinite and nonnegative.

Proof. Acceptance removes the sum $b _ { D }$ and adds $b _ { D } + p w$ A rejected negative adds zero; a rejected positive adds fees summing to w. On rejection $W ~ > ~ 0$ , and a ledger-test failure has $b _ { D } > 0$ . Induction proves both assertions.

![](images/7f65ded552220e65115617f5f81b87422bc7386b9afa7504e7e6d2b02f7e0606.jpg)  
Figure 2. Universal lower-bound construction. Filled left and open right endpoints indicate half-open intervals. The common prefix forces retention of J and permanent rejection of every H<sub>i</sub>; the decision on $\hat { L }$ selects one of two complete inputs. Both have perfec predictions and a unique optimum. The geometry uses $\ s = 1 - 2 \delta , t = \varphi ^ { - 1 }$ , and $\delta > 0 ;$ six central pieces are drawn schematically.

```latex
Algorithm 1 Shared ledger policy with ordinary gate Q
1: Initialize $s  \varnothing .$
2: On arrival of $( I , p )$ , with $w = w ( I ) ;$
3: $D  \{ J \in { \mathcal { S } } : J$ conflicts with $I \} ; W  w ( D )$
4: Snapshot $\bar { b } _ { J }  b ( J )$ for every $J \in D .$
5: $\begin{array} { r } { \boldsymbol { b } _ { D } \dot {  } \sum _ { J \in D } \bar { \boldsymbol { b } } _ { J } ; Z  [ \forall J \in D : p ( J ) = 0 ] . } \end{array}$
6: Test ordinary $w \geq \mathcal { Q } ( D )$ first (true if $D = { \mathcal { O } } )$
7: If it fails, test special $p = 1 , Z , w \ge \alpha W , w \ge b _ { D } / \varphi .$
8: if either test accepts then
9: Accept I; permanently revoke all of D.
10: $\begin{array} { r } { S \gets ( S \setminus D ) \cup \{ I \} ; b ( I ) \gets b _ { D } + p w . } \end{array}$
11: else
12: Permanently reject I.
13: i $\mathbf { f } p = 1$ then
14: Compute every g<sub>J</sub> by (8) from snapshots.
15: Simultaneously set $b ( J )  \bar { b } _ { J } + g _ { J }$ for $J \in D .$
16: end if
17: end if
18: All other existing ledgers are unchanged; output S at
termination.
```

Why length alone does not determine a replacement. For $0 < \varepsilon < 1 / 8$ , start with negative [0, 1) and reveal the compatible positives $[ \varepsilon , 1 / 2 ) , [ 1 / 2 , 1 - \overset { \cdot } { \varepsilon } )$ and $[ - \alpha + 2 \varepsilon , \varepsilon )$ All three are shorter than α and fail both acceptance routes. The holder remains [0, 1), but its ledger grows from zero to $\varphi - 3 \varepsilon > 1$ . Now positive $[ 1 - \varepsilon , 1 - \varepsilon + \alpha )$ passes the length test at equality. Its ledger test fails: $b _ { D } > 1 = \varphi w$ . Removing that test would leave reward α and a perfect-label ratio approaching $\varphi + 2 > d .$ The complete five-request calculation is in Appendix M.

The price of a special replacement. A negative lengthφ anchor has zero-ledger cost M. A unit positive replaces it at the special threshold and acquires cost M. Rejections around these successive holders yield a compatible packing approaching $\varphi ^ { 5 }$ while only the unit survives (Proposition D.5).

## 5. Why the Ledger Gives Optimal Consistency

Theorem 5 (A sufficient consistency interface). Use positive cost Mv, the summed ordinary gate, and the shared BANK rules. Suppose the negative cost is finite and satisfies q(v, b) ≥ dv for all $b \geq 0 ,$ , and

$$
v + \operatorname* { m a x } \{ \alpha v , \alpha ^ { 2 } b \} \leq \alpha q ( v , b ) \quad ( 0 \leq b \leq d v ) .\tag{9}
$$

Ifthe complete positive set P isfeasible, every held request satisfies $b ( J ) \leq d w ( J )$ . Consequently $w ( P ) \leq d \mathrm { A L G }$ Shaped-BANK and SUM-BANK with cost Mv satisfy this interface; theirfinite robustness and Theorem 2 imply C = d.

Proof sketch: full source shares and directional cuts. Fix a finite execution with feasible complete positive set P. At snapshot n, conserved fractions $a _ { X , n } ( I )$ assign every arrived positive source I among held identities X. The

measure

$$
\mu _ { X , n } ( A ) = \sum _ { I \in P } a _ { X , n } ( I ) | I \cap A |\tag{10}
$$

has mass $b _ { n } ( X )$ and density at most one; unarrived sources have zero coefficients. Its support is not clipped to $X \colon$ replacing positive $[ 0 , 1 )$ by negative [.9, 6) inherits mass one, not .1. Let $E _ { L } , E _ { R }$ denote left and right exterior mass and $\begin{array} { r } { L _ { n , X } ( s ) = \sum _ { I \in P : f ( I ) \leq s } a _ { X , n } ( I ) w ( I ) } \end{array}$ the full left shares.

Lemma 2 (Side and directional-cut invariant). Under the cost and BANK-rule hypotheses of Theorem 5, fix a finite execution with feasible complete positive set $P .$ . Use the conserved source measure and full shares defined above. Write $\nu _ { X }$ for birth and $\delta _ { X }$ for revocation, or one beyond the final arrival if X survives. For every negative $V =$ $[ x , x + v )$ and live snapshot $\nu _ { V } ~ \leq ~ n ~ < ~ \delta _ { V } ,$ , each side $S \in \{ L , R \}$ satisfies

$$
\begin{array} { r l } & { E _ { S , \nu _ { V } } ( V ) \leq \alpha v , \qquad b _ { n } ( V ) \leq d v , } \\ & { E _ { S , n } ( V ) \leq \operatorname* { m a x } \{ \alpha v , \alpha ^ { 2 } b _ { n } ( V ) \} . } \end{array}
$$

For each actual positive $T = [ s , t ) \in P$ arriving after $\nu _ { V } ,$ $i f s < z \leq t , z \leq$ x and $L _ { \nu _ { V } , V } ( s ) > 0 ,$ , then

$$
L _ { \nu _ { V } , V } ( s ) + z - s \leq \alpha v .
$$

At each live snapshot n, a positive Q of length u has $b _ { n } ( Q ) \leq$ du and, for every actual positive $T = [ s , t )$ arriving after its live snapshot n,

$$
L _ { n , Q } ( s ) + z - s \leq d u \quad ( s < z \leq t \leq s ( Q ) , L _ { n , Q } ( s ) > 0 ) .
$$

Both cuts include their reflections under $y \mapsto - y .$ The complete proof is in Appendix D.1.

Why the negative capacity follows. Induct on accepted birth order, establishing every live snapshot of each earlier identity before using it at a parent’s birth. Earlier children’s cuts and the cost interface pay the new negative birth bounds. The positive cut uses the three cases below. Independently, replacement-tree geometry bounds subtree length by 2v at a negative root and Mv at a positive root (Appendix D). Density bounds the mass inside V by v. A side with nonzero exterior mass at birth stays bounded by αv: an existing crossing source excludes another, while wholly exterior birth sources and a later crossing source share the birth-cut budget. Each initially empty side has at most one later crossing source. Its fee obeys $g \ : < \alpha v$ after length failure, or $g \leq \alpha b _ { \mathrm { p r e } }$ after ledger failure, which implies $g \leq \alpha ^ { 2 } ( b _ { \mathrm { p r e } } + g )$ . These are event-time bounds.

With two occupied birth sides, $b \leq ( 1 + 2 \alpha ) v < d \iota$ . With one, the other side’s first crossing has $b _ { \mathrm { p r e } } \leq ( 1 + \alpha ) v$ , so its fee is at most $\alpha ( 1 + \alpha ) v = v$ . Without that crossing, density already suffices. Thus

$$
b \leq \underbrace { v } _ { \mathrm { i n s i d e } } + \underbrace { \alpha v } _ { \mathrm { p r o t e c t e d s i d e } } + \underbrace { v } _ { \mathrm { o t h e r s i d e } } = d v .
$$

With no occupied birth side, the first crossing has $b _ { \mathrm { p r e } } \leq v$ and fee at most αv, reducing to the preceding case. A source crossing both endpoints pays one fee and excludes every later crossing. Thus event-time bounds close capacity.

At a positive birth, let $B _ { Q } , W _ { Q }$ be its negative children’s total ledger and length. Ordinary acceptance gives $B _ { Q } \le$ $u , W _ { Q } \leq u / d ;$ special acceptance gives $B _ { Q } , W _ { Q } \leq \varphi u .$ Adding $Q$ gives capacity, and compatibility excludes later positive fees. For the cut, set $e = z - s , G = s ( Q ) - s \geq$ $e .$ Trace a left source share backward to the first support starting below s. There are three cases.

(i) An intermediate positive H occurs before that crossing. Shrinking paths give $w ( H ) \leq u / d$ and $e < d w ( H ) \leq u ;$ the cut is at most $B _ { Q } + u \leq d u .$

(ii) Otherwise the negative path starts with child $V ,$ of length $v .$ For ordinary $Q , G < \varphi v \leq \alpha u$ . For special $Q ,$ , a nonleftmost $V$ gives $G < \alpha v \leq u ;$ if another child K carries left shares, protrusion gives $\varphi G < v + w ( K ) \leq \varphi u$ Again $B _ { Q } + u \ p \mathrm { a y s }$ the cut.

(iii) Only the leftmost child V carries shares and $Q$ is special. If $s ( V ) \geq s ,$ , its inherited birth cut plus the remaining distance gives a bound below $\varphi v$ . Otherwise density gives

$$
L _ { n , Q } ( s ) + e < v + E _ { L } ( V ) \leq \operatorname* { m a x } \{ \varphi v , v + \alpha ^ { 2 } b ( V ) \} \leq d u .
$$

Here the length gate pays for $\ v \ \leq \ \varphi u$ and the ledger gate for $b ( V ) \leq \varphi u ;$ neither budget alone suffices for this bound. Reflection and source conservation finish w $\mathop { \prime } ( P ) =$ $\sum _ { X } b ( X ) \leq d \mathrm { A L G }$ . Appendix D proves the geometric and boundary-case details.

At the golden endpoint, $q _ { * }$ is the smallest cost nondecreasing in b that satisfies this sufficient interface (Proposition D.2). The interface itself is not necessary: setting $q ( v , 0 ) \ = \ \varphi v$ and keeping $q _ { * }$ for $b ~ > ~ 0$ also preserves d-consistency, by pruning source-free children (Appendix E.1). The relaxed policy still satisfies the nonshrinking hypothesis below, so its robustness is at least $\varphi ^ { 5 }$

## 6. Exact Robustness and the Remaining Gap

Theorem 6 (Exact Shaped-BANK robustness). For arbitraryfinite labelled inputs, $R ( T _ { * } ) = \varphi ^ { 5 }$

Proof sketch: joint endpoint charges. In the replacement forest, revoked requests are children of their replacements. Give each accepted identity its ordinary cost just before revocation or final termination. Costs do not decrease during a lifetime. An ordinary parent of length h pays at most h in child costs. At a special parent, its negative children have total length and ledger $W , b _ { D } \leq \varphi h$ , and

$$
\begin{array} { l } { { q _ { * } ( v , b ) \leq d v + \alpha ^ { 3 } b , } } \\ { { \displaystyle \sum _ { i } q _ { i } \leq d W + \alpha ^ { 3 } b _ { D } \leq A h , \quad A = \varphi + 3 . } } \end{array}
$$

For analysis, split each rejected optimal request among its current conflicts in proportion to their costs. These receivers form an ancestry antichain: no receiver is an ancestor of another. The comparator is a fixed offline optimum. Passing the charges to final roots preserves their sum. At most two disjoint optimal requests cross the boundary of a root’s connected accepted hull; the others fit between their cuts. Bounding the two exterior charges jointly with the space remaining between them avoids counting the same interval twice.

Let $H _ { J }$ be the connected accepted hull of a root J. For a cut x, define $L _ { J } ( x )$ as the largest sum of terminal costs over an ancestry antichain in this subtree whose intervals start strictly left of $x ;$ define $U _ { J } ( y )$ by ends strictly right of $y .$ The cuts $x , y$ are the inward endpoints of the crossing comparator intervals, using the hull endpoints when a crossing is absent. Unless one comparator covers the entire hull, feasibility gives $x \leq y$ and the root charge is bounded by

$$
F _ { J } ( x , y ) = | H _ { J } \cap [ x , y ] | + L _ { J } ( x ) + U _ { J } ( y ) .
$$

The two maxima need not use the same antichain. A comparator covering the entire hull is treated separately: it is the only contributor there and is charged at most $A h$

At an ordinary node, its own cost dominates every descendant antichain’s total terminal cost, so the node itself maximizes an endpoint whenever eligible. At a special node $\boldsymbol { J } ~ = ~ [ a , b )$ , each endpoint maximum is either the root’s singleton cost or the sum of child maxima. Orient a single singleton to the left. If $y < b ,$ the gap lies inside $J ;$ if $y \geq b ,$ at most one compatible child extends beyond y. With both maxima supplied by children, at most one child bridges both cuts. These alternatives exhaust the geometry, including coincident cuts and equality at an interval endpoint. The coefficients are in Table 2 (Appendix D.3).

The largest case is a special root with one endpoint charged at the root and the other at an outward child. Write $\rho =$ $A / M$ . Monotonicity and the fixed-ledger slope inequality $q _ { * } ( W , b _ { D } ) - q _ { * } ( v , b _ { D } ) \geq \varphi ( W - v )$ absorb that child’s cost and excess length. Since $\rho - 1 < \varphi$ , the total charge is at most

$$
\begin{array} { l } { { ( M + 1 ) h + W + q _ { * } ( W , b _ { D } ) } } \\ { { \qquad \leq ( M + 1 + \varphi ) h + q _ { * } ( \varphi h , \varphi h ) = \varphi ^ { 5 } h . } } \end{array}\tag{11}
$$

The other endpoint choices give smaller coefficients; $\mathsf { A p - }$ pendix D lists and proves them. Summing over final roots proves the upper bound. The five-request construction in Proposition D.5 approaches $\varphi ^ { 5 }$ , proving equality. □

The endpoint cannot be improved merely by reshaping nonshrinking negative costs $\left( q _ { - } \left( v , b \right) \ \geq \ v \right)$ within the same special tests: d-consistency then forces $R \geq \varphi ^ { 5 }$ , even without cost continuity or monotonicity (Appendix E.1). To trade consistency for a smaller budget, we change both special thresholds below.

## 6.1. A continuous attainable tradeoff

The optimal-consistency endpoint is part of a tunable family. Fix $a \in [ \alpha , 1 ]$ and put

$$
m _ { a } = \operatorname* { m a x } \{ ( 1 + a ) / a ^ { 2 } , 1 + a + a ^ { - 1 } \} .
$$

Policy $T _ { a }$ uses positive cost $m _ { a } v$ , negative cost

$$
\begin{array} { c } { q _ { a } ( v , b ) = \operatorname* { m i n } \bigr \{ ( 1 + a + a ^ { - 1 } ) v , } \\ { \operatorname* { m a x } \bigr \{ ( 1 + a ^ { - 1 } ) v , v / a + b / ( 1 + a ) \bigr \} \bigl \} , } \end{array}\tag{12}
$$

and special tests w $\ge a W , w \ge a b _ { D }$ . In all fee-case guards, replace αW by aW and $b _ { D } / \varphi$ by $a b _ { D } ;$ the proportional fees, priority and inheritance are unchanged.

Theorem 7 (Continuous attainable guarantees). For every $a \ \in \ [ \alpha , 1 ] , \ C ( T _ { a } ) \ = \ ( 1 + a ) ^ { 2 }$ and $R ( T _ { a } ) ~ \leq ~ K _ { a } ~ =$ max $\{ O _ { a } , N _ { a } \}$ , where the outward and no-bridge coefficients are defined in $( \mathrm { D } . 2 3 ) \colon O _ { a }$ comes from a special root with one outward child; $N _ { a }$ from child charges with no child bridging both cuts. The bound is exact on $[ \alpha , a _ { - } ] \cup$ $[ a _ { + } , 1 ]$ , where $a _ { - } ~ \approx ~ 0 . 6 7 7 8 2 1$ and $a _ { + } \approx 0 . 8 1 3 8 2 4$ are the unique roots specified by (D.29) in Proposition D.4. In particular,

$$
\begin{array} { r } { \frac { a } { C ( T _ { a } ) } \left| \begin{array} { c c c } { \alpha } & { 2 / 3 } & { 1 } \\ { d } & { 2 5 / 9 } & { 4 } \\ { \varphi ^ { 5 } } & { 1 0 } & { 7 } \end{array} \right. } \end{array}
$$

All three pairs are exact, and the guarantees hold after the compatible-positive filter.

Proof sketch: mechanism of the tradeoff. The source induction separates negative capacity $( 1 + a ) ^ { 2 } \ i$ from positive capacity $a m _ { a } v .$ . The exterior-mass bounds become av at birth and max $\{ a v , a b / ( 1 + a ) \}$ during a negative lifetime. Their weighted cut is paid by $q _ { a } .$ In the positive cut’s multichild case, the accepted-history bounds change branch at $a ^ { 3 } + a ^ { 2 } = 1$ ; both branches pay the same directional gap. The joint-cut analysis reduces to the two continuous coefficients $O _ { a } , N _ { a }$ . Five-request families match the consistency throughout the range and the robustness on the two stated intervals. Appendix D gives the induction, coefficient comparison and matching constructions. At budget 7, $T _ { 1 }$ improves $\operatorname { L R } ^ { \prime } \mathrm { { s } } \ M \approx 4 . 2 3 6$ and $\mathrm { R P s }$ certificate 5; continuity extends this improvement to an interval. At budget 10, $T _ { 2 / 3 }$ has exact consistency $2 5 / 9 < 3$ . For $a _ { - } < a < a _ { + }$ , only $O _ { a } \leq R ( T _ { a } ) \leq N _ { a }$ is known.

The envelope combines LR’s $C = R = M , { \tt R P s } C \leq$ $3 ( B - 2 ) / ( B - 4 )$ for $B \geq 6$ , and $T _ { a }$ . RP has exact robustness $2 \lambda + 2$ for $\lambda \geq 2 \ ( \mathbf { A } _ { } ]$ ppendix J); its consistency certificate beats LR above $4 + 3 \varphi \approx 8 . 8 5 4$ . Appendix G compares SUM and scalar Credit. For $M \leq B < 7 .$ , LR remains the best guarantee among these policies.

## 7. Guarantees and Learning Interfaces

## 7.1. Randomization and the guarantee being protected

For fixed inputs, measure $\mathrm { O P T } / \mathbb { E } [ \mathrm { A L G } ]$ . An initial coin selects NAIVE with probability $\pi ,$ otherwise LR. Its exact tradeoff is

$$
\overline { { R } } = \frac { 2 \varphi c } { c - 1 } , \quad 1 < c \leq M .\tag{13}
$$

Here $\pi = ( M / c - 1 ) / ( M - 1 )$ . At $c = d , \overline { { R } } = 2 d ;$ at $\overline { { R } } = \varphi ^ { 5 } , c \approx 1 . 4 1 2$ . On all-negative inputs the Naive branch gains zero. The finite-expected-robustness consistency infimum is one, unattained; the full frontier remains open (Appendix L.2).

With only robustness almost surely and consistency in expectation, every finite budget $B \ > \ 1$ instead requires (Proposition L.4)

$$
C _ { \mathrm { e x p } } \geq 2 + \frac { 1 } { B - 1 } > 2 .
$$

Robustness on a fixed shorter input forces rejection on the shared prefix; a fixed longer perfect input then gives the bound.

A half-length special gate with ordinary factor four allows pathwise robustness at most 18, a sharp uniform bound over this action family, but every fixed independent acceptance probability has $C _ { \mathrm { e x p } } \geq d ( \mathrm { A }$ ppendix L.5).

For $0 < \pi < 1$ , adapting inputs to observed actions makes the mixture’s expected robustness infinite (Appendix L.3; Ben-David et al., 1994). This further motivates protection of individual executions rather than relying solely on a mixture’s fixed-input expected reward.

## 7.2. Workload and learned-label diagnostics

NASA-iPSC and CTC-SP2 give 18,066 and 77,205 intervals. Twenty shared reveal/label repetitions flip members (FN) or nonmembers (FP) of one optimum; FP 1 is allpositive, FN 1 all-negative.

Filtering preserves both exact guarantees and adds $\mathrm { O P T \leq }$ d $\mathrm { A L G + } \eta _ { K }$ . Here $\eta _ { K }$ is missed optimal weight plus each false positive’s nonnegative excess of conflicting optimal weight over its own length (Appendix K). Zero conflict error need not make positives feasible, so raw SG lacks this guarantee.

Table 5 (Appendix N) reports means, intervals and individual-run fifth percentiles $q . 0 5$ . Intervals use 10,000 shared whole-input bootstrap resamples (Appendix N.1).

LR leads FP .1 and 1 in mean and $q . 0 5$ . In the two low-FP rows of Table 5, $\mathrm { S G } ^ { F }$ gains .245 and .532 points over LR (paired intervals [.076, .401] and [.365, .693]), and raw SG gains a further .527 and .916 (Appendix N), so filtering is not the source. Both rows use the smallest rate of the prespecified low-FP grid (Appendix O). Perfect-label Naive gets 100% but lacks finite robustness. Filtered SG exceeds filtered RP in all 20 cell means; its differences from best scalar Credit range from about −.001 to .04 points (Appendix N).

Chronological logistic advice on 7 NASA and 31 CTC instances gives $\mathrm { S G } ^ { F }$ 94.58/89.66%, below LR’s 96.86/91.25%. Reward selection on reused blocks mainly chooses all-negative advice; the nonconstant CTC cost score remains below LR (Appendices P.1–P.2).

## 7.3. What should a learned advice bit predict?

The same rejection can leave the same held interval but different ledgers, changing a later replacement; thus the useful prediction can concern continuation reward rather than immediate reward. Appendix Q lets the bit estimate its rollout effect on final reward. Its transfer bound uses the reference policy’s membership error and deployment-state rollout error; the latter is not estimated here. An independent cohort favors lookahead but does not confirm additional learning gains.

## 8. Discussion

The universal budget for attaining d remains in $[ 4 \varphi + 3 , \varphi ^ { 5 } ]$ The optimal universal frontier and the tradeoff with almostsure robustness but expected consistency are open; within $T _ { a }$ , only $( a _ { - } , a _ { + } )$ lacks exact robustness.

Improving the budget upper bound $\varphi ^ { 5 }$ requires leaving policies that always reject negative arrivals shorter than d times a sole negative holder, even with arbitrary historydependent ledgers and positive rules (Appendix F).

A filter-free policy with $C \ = \ d ,$ finite robustness and a smooth conflict-error guarantee remains open; weighted precision and recall need not control this error (Appendix K.5).

## Impact Statement

The results concern retained length under unrestricted revocation. Proportional fees account for one cancellation cost, but not reliability obligations or unequal consequences across users. Training labels inherit their offline objective’s biases; neither optimal membership nor these competitive ratios ensure fair allocation. The public job traces are used only to construct selection instances, without inferring individual behavior.

## References

Ameli, A. J., Sanità, L., and Venzin, M. Learningaugmented online covering problems. In Proceedings ofthe 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pp. 2307–2331. PMLR, 2026.

Angelopoulos, S., Dürr, C., Jin, S., Kamali, S., and Renault, M. Online computation with untrusted advice. In 11th Innovations in Theoretical Computer Science Conference, volume 151 of Leibniz International Proceedings in Informatics, pp. 52:1–52:15, 2020. doi: 10.4230/LIPIcs.ITCS.2020.52.

Antoniadis, A., Coester, C., Elias, M., Polak, A., and Simon, B. Online metric algorithms with untrusted predictions. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings ofMachine Learning Research, pp. 345–355, 2020.

Antoniadis, A., Shahheidar, A., Shahkarami, G., and Soltani, A. A switching framework for online interval scheduling with predictions. Proceedings of the AAAI Conference on Artificial Intelligence, 40(43): 36129–36136, 2026. doi: 10.1609/aaai.v40i43.40930.

Babaioff, M., Hartline, J. D., and Kleinberg, R. D. Selling ad campaigns: Online algorithms with cancellations. In Proceedings of the Tenth ACM Conference on Electronic Commerce, 2009. URL https://www.eecs.northwestern. edu/\~hartline/papers/buyback-EC-09.pdf.

Ben-David, S., Borodin, A., Karp, R., Tardos, G., and Wigderson, A. On the power of randomization in online algorithms. Algorithmica, 11(1):2–14, 1994. doi: 10.1007/BF01294260.

Benomar, Z. and Perchet, V. Non-clairvoyant scheduling with partial predictions. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 3506–3538. PMLR, 2024.

Berg, M. Comparing the hardness of online minimization and maximization problems with predictions. In

Frontiers of Algorithmics: IJTCS-FAW 2025, volume 15828 of Lecture Notes in Computer Science, pp. 33–48. Springer, 2025. doi: 10.1007/978-981-96-8312-3\_3.

Borodin, A. and Karavasilis, C. Any-order online interval selection. In Byrka, J. and Wiese, A. (eds.), Approximation and Online Algorithms: 21st International Workshop, WAOA 2023, volume 14297 of Lecture Notes in Computer Science, pp. 175–189. Springer, 2023. doi: 10.1007/978-3-031-49815-2\_13.

Boyar, J., Favrholdt, L. M., Kamali, S., and Larsen, K. S. Online interval scheduling with predictions. Journal of Computer and System Sciences, 160:103795, 2026. doi: 10.1016/j.jcss.2026.103795.

Chen, P., Zhao, H., Tang, X., Wang, Y., and Deng, S. Towards optimal robustness in learning-augmented paging. In Proceedings ofthe 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pp. 18492–18516. PMLR, 2026.

Dallot, J., Emek, Y., Gil, Y., Pacut, M., and Schmid, S. Online algorithms with unreliable guidance. arXiv:2602.20706v2, 2026.

Daneshvaramoli, M., Karisani, H., Lechowicz, A., Sun, B., Musco, C. N., and Hajiesmaili, M. Nearoptimal consistency-robustness trade-offs for learningaugmented online knapsack problems. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 12459–12489. PMLR, 2025.

Elmachtoub, A. N. and Grigas, P. Smart “predict, then optimize”. arXiv preprint arXiv:1710.08005v5, 2020. Author preprint version 5, revised 19 November 2020.

Feitelson, D. G., Tsafrir, D., and Krakov, D. Experience with using the parallel workloads archive. Journal of Parallel and Distributed Computing, 74(10):2967–2982, 2014. doi: 10.1016/j.jpdc.2014.06.013.

Fung, S. P. Y., Poon, C. K., and Zheng, F. Improved randomized online scheduling of intervals and jobs. arXiv preprint arXiv:1204.2933, 2012.

Garay, J. A., Gopal, I. S., Kutten, S., Mansour, Y., and Yung, M. Efficient on-line call control algorithms. Journal ofAlgorithms, 23(1):180–194, 1997. doi: 10.1006/ jagm.1996.0821.

Grigorescu, E., Lin, Y.-S., and Song, M. Learningaugmented algorithms for online concave packing and convex covering problems. In Proceedings of the 28th International Conference on Artificial Intelligence and Statistics, volume 258 of Proceedings ofMachine Learning Research, pp. 1072–1080. PMLR, 2025.

Jin, B. and Ma, W. Online bipartite matching with advice: Tight robustness-consistency tradeoffs for the two-stage model. In Advances in Neural Information Processing Systems, volume 35, 2022. URL https://proceedings.ne urips.cc/paper\_files/paper/2022/file/5d68a3f05ee2aae6 a0fb2d94959082a0-Paper-Conference.pdf.

Karavasilis, C. Interval selection with binary predictions. In Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, pp. 8527–8535. International Joint Conferences on Artificial Intelligence Organization, 2025. doi: 10.24963/ijcai.2025/948.

Khodak, M., Balcan, M.-F., Talwalkar, A., and Vassilvitskii, S. Learning predictions for algorithms with predictions. In Advances in Neural Information Processing Systems, volume 35, pp. 3542–3555, 2022.

Li, Y., Lavasani, A. M., and Pankratov, D. Online interval selection on a simple chain. Theoretical Computer Science, 1085:116195, 2026. doi: 10.1016/j.tcs.2026.116 195.

Lipton, R. J. and Tomkins, A. Online interval scheduling. In Proceedings of the Fifth Annual ACM-SIAM Symposium on Discrete Algorithms, pp. 302–311, 1994.

Lykouris, T. and Vassilvitskii, S. Competitive caching with machine learned advice. In Proceedings ofthe 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pp. 3296– 3305. PMLR, 2018.

Mitzenmacher, M. and Vassilvitskii, S. Algorithms with predictions. arXiv:2006.09123, 2020.

Nair, V., Bartunov, S., Gimeno, F., von Glehn, I., Lichocki, P., Lobov, I., O’Donoghue, B., Sonnerat, N., Tjandraatmadja, C., Wang, P., Addanki, R., Hapuarachchi, T., Keck, T., Keeling, J., Kohli, P., Ktena, I., Li, Y., Vinyals, O., and Zwols, Y. Solving mixed integer programs using neural networks. arXiv preprint arXiv:2012.13349v3, 2021.

Parallel Workloads Archive. The Cornell Theory Center (CTC) IBM SP2 log. Dataset documentation, n.d.a. URL https://www.cs.huji.ac.il/labs/parallel/workload/l\_ctc\_s p2/index.html.

Parallel Workloads Archive. The NASA Ames iPSC/860 log. Dataset documentation, n.d.b. URL https://www.cs .huji.ac.il/labs/parallel/workload/l\_nasa\_ipsc/index.htm l.

Parallel Workloads Archive. Standard Workload Format, version 2.2. Format documentation, n.d.c. URL https: //www.cs.huji.ac.il/labs/parallel/workload/swf.html.

Purohit, M., Svitkina, Z., and Kumar, R. Improving online algorithms via ML predictions. In Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018. URL https://papers.neurips.cc/pape r\_files/paper/2018/hash/73a427badebe0e32caa2e1fc753 0b7f3-Abstract.html.

Ross, S. and Bagnell, J. A. Reinforcement and imitation learning via interactive no-regret learning. arXiv preprint arXiv:1406.5979v1, 2014.

Ryabchenko, A., Attias, I., and Roy, D. M. Capacityconstrained online learning with delays: Scheduling frameworks and regret trade-offs. In Proceedings of Thirty Eighth Conference on Learning Theory, volume 291 of Proceedings of Machine Learning Research, pp. 4973–5014, 2025.

SchedMD. Slurm workload manager: Advanced resource reservation guide, n.d. URL https://slurm.schedmd.co m/reservations.html. Accessed 18 October 2026; live documentation.

Seiden, S. S. Randomized online interval scheduling. Operations Research Letters, 22(4–5):171–177, 1998. doi: 10.1016/S0167-6377(98)00019-4.

Tesauro, G. and Galperin, G. R. On-line policy improvement using monte-carlo search. In Advances in Neural Information Processing Systems, volume 9, 1996. URL https://proceedings.neurips.cc/paper\_files/paper/1996/ hash/996009f2374006606f4c0b0fda878af1-Abstract.ht ml.

Tomkins, A. Lower bounds for two call control problems. Information Processing Letters, 56(3):173–178, 1995. doi: 10.1016/0020-0190(95)00147-5.

Wang, Z., Zhang, Y., Wei, Z., and Li, S. Bandit learning for online scheduling with immediate decision. In Proceedings of the 42nd Conference on Uncertainty in Artificial Intelligence, volume 337 of Proceedings of Machine Learning Research, pp. 7168–7199. PMLR, 2026.

Wei, A. and Zhang, F. Optimal robustness-consistency trade-offs for learning-augmented online algorithms. In Advances in Neural Information Processing Systems, volume 33, 2020. URL https://proceedings.neurips.cc/pap er/2020/file/5bd844f11fa520d54fa5edec06ea2507-Pap er.pdf.

Woeginger, G. J. On-line scheduling of jobs with fixed start and end times. Theoretical Computer Science, 130(1):5– 16, 1994. doi: 10.1016/0304-3975(94)90150-3.

## Technical Appendix

## Guide to the appendix

<table><tr><td>Question</td><td>Reading route</td></tr><tr><td>Universal consistency and budget lower bounds</td><td>Appendices A-C</td></tr><tr><td>Parametric guarantees, exact intervals and cost limits</td><td>Appendix D, especially Proposition D.4; penalties in  $\mathsf { A p - }$ </td></tr><tr><td>Credit, SUM and RP guarantees</td><td>pendix E and history-dependent restriction in Appendix F Overview in Appendix G; proofs in Appendices H–J</td></tr><tr><td>Filtering and randomized guarantees</td><td>Appendix  $\mathrm { K } ;$  randomized guarantees in Appendix L, includ- ing the sharp safe action family in Proposition L.5</td></tr><tr><td>Mechanism experiments and protocol</td><td>Results in Appendices M-N and  $\mathrm { P } ;$  shared protocol in  $\mathsf { A p - }$ </td></tr><tr><td>Learned labels and rollout advice</td><td>pendix O Appendix P.2; rollout and its expected reference certificate in</td></tr><tr><td>Model and literature comparisons</td><td>Appendix Q Appendix R</td></tr></table>

## Notation guide

The algorithm stores only requests, labels and ledgers. Source measures, cuts, capacities and replacement histories are proof quantities.

<table><tr><td>Notation</td><td>Meaning</td></tr><tr><td colspan="2">Core policies and error</td></tr><tr><td> $\varphi , \alpha , d , M$ </td><td>Golden ratio,  $\alpha = 1 / \varphi _ { \mathrm { : } }$  optimal consistency  $d = \varphi ^ { 2 }$  , and positive cost coefficient  $M = \varphi ^ { 3 }$ </td></tr><tr><td> $T _ { * } , q _ { * } ( v , b )$ </td><td>Shaped-BANK and its negative cost for length v and ledger b, defined in (D.18).</td></tr><tr><td> $T _ { a } , q _ { a } , C _ { a } , K _ { a }$ </td><td>Parametric policy, negative cost, exact consistency  $( 1 + a ) ^ { 2 }$  , and robustness certificate for  $a \in [ \alpha , 1 ] .$ </td></tr><tr><td> $O _ { a } , N _ { a } , a _ { - } , a _ { + }$ </td><td>Outward and no-bridge coefficients, with  $K _ { a } = \operatorname* { m a x } \{ O _ { a } , N _ { a } \}$  ; exact robustness holds on  $[ \alpha , a _ { - } ] \cup [ a _ { + } , 1 ]$ </td></tr><tr><td> $\tau ( J ) , q _ { J }$ </td><td>A held request&#x27;s current ordinary cost and its terminal cost at revocation or final termination; subscripts specify the policy.</td></tr><tr><td> $W , b _ { D } , b ( J )$ </td><td>Total conflicting length, total conflicting ledger, and the ledger of a current request; some appendix proofs denote the total ledger by B.</td></tr><tr><td> $\eta _ { K } ( P , O )$ </td><td>Conflict-sensitive error of original labels relative to a fixed optimum; subscript K names the metric.</td></tr><tr><td colspan="2">Lower bounds and attained budgets</td></tr><tr><td> $B _ { * } , B _ { C * } , B _ { S }$   $G ( \boldsymbol { B } ) , D _ { \mathrm { s } } , D _ { 1 } , H$ </td><td>Exact Shaped-BANK, best scalar Credit and SUM endpoints  $\varphi ^ { 5 } , 4 \varphi + 5$  and  $7 \varphi + 4$  Binary-reserve lower curve, short/long-reserve necessary budgets, and  $H = \operatorname* { m a x } \{ H _ { \mathrm { s } } , H _ { \mathrm { l } } \}$ </td></tr><tr><td> $\beta$ </td><td>from their inverse consistency bounds. Local notation:  $4 \varphi + 3$  in Appendix  $\operatorname { C } ; \varphi$  in Appendix I; a free transfer parameter in</td></tr><tr><td></td><td>Appendix J.</td></tr><tr><td>π, rLR Appendix variants and proof quantities</td><td>Initial-mixture probability of Naive and the LR ratio  $r _ { \mathrm { L R } } = M .$ </td></tr><tr><td colspan="2"> $\beta _ { a } , q _ { 0 , a } , Q _ { a } , m _ { a } , k _ { a }$ </td></tr><tr><td></td><td>Appendix D&#x27;s parameter-dependent cost and capacity coefficients; •  $Q _ { a }$  is distinct from a request Q.</td></tr><tr><td> $h _ { 0 , a } , h _ { 1 , a } , \theta _ { a } , c _ { a } , A _ { a } , \rho _ { a }$ </td><td>Parameter-dependent history, source-share and terminal-cost coefficients in (D.1) and  $( \mathrm { D } . 1 9 ) ; a _ { c } ^ { 3 } + a _ { c } ^ { 2 } = 1$  marks the history branch change.</td></tr><tr><td> $A _ { \lambda }$ </td><td>Descriptive MAX-BANK comparator with  $\alpha = 1 / \varphi , \beta = \varphi$  and ordinary maximum thresh- old λ.</td></tr><tr><td> $S _ { \lambda }$ </td><td>SUM-BANK with the same special tests and ledgers, and ordinary total-conflict threshold λ.</td></tr><tr><td> $T _ { c } , c _ { * } = \alpha ^ { 3 }$ </td><td>Scalar Credit-BANK and its best consistent coefficient;  $T = T _ { \varphi / 2 }$  denotes the fixed  $c =$   $\varphi / 2$  comparison.</td></tr><tr><td> $a _ { X , n } ( I ) , \mu _ { X , n } , E _ { L } , E _ { R }$ </td><td>Conserved share of positive source I in X, its source measure, and the mass exterior to X on each side.</td></tr><tr><td> $P , P ^ { \prime }$ </td><td>Original positive set and permanent feasible positive history produced by the filter.</td></tr></table>

## A. Finite universal witnesses

Theorem 2 gives the complete stopping-prefix proof. For its bounded-size refinement, fix a finite robustness bound $R \geq 1$ with

$$
\mathrm { O P T } ( \tau ) \leq { R \mathrm { A L G } _ { { \cal A } } } ( \tau ) \quad \mathrm { o n e v e r y ~ n o n e m p t y ~ l a b e l l e d ~ p r e f i x ~ } \tau .\tag{A.1}
$$

Write $J = [ 0 , 1 )$ , s = 1 − 2δ and $\ell = s / m < 1 / R .$ . The positive requests are

$$
H _ { k } = [ \delta + ( k - 1 ) \ell , \delta + k \ell ) , \quad k = 1 , \ldots , m ,\tag{A.2}
$$

$$
L = [ \delta - t , \delta ) ,\tag{A.3}
$$

$$
U = [ 1 - \delta , 2 - \delta ) .\tag{A.4}
$$

The negative anchor J is followed by the $H _ { k }$ and $L ; U$ arrives only if L is rejected. Thus $\sigma ^ { - } = ( J , H _ { 1 } , \ldots , H _ { m } , L )$ and $\sigma ^ { + } = ( J , H _ { 1 } , \ldots , H _ { m } , L , U )$ . Figure 2 shows both branches.

Corollary A.1 (Bounded-size witnesses with a prescribed ratio). Fix a deterministic algorithm A in the model in the main text and any validfinite robustness bound $R \geq 1$ satisfying (A.1). For every real number $\kappa \in ( 2 , d )$ , there is a finite input $\sigma _ { \kappa }$ with at most $\lceil R \rceil + 3$ requests such that its positive requestsform its unique optimal solution and

$$
{ \frac { \mathrm { O P T } ( \sigma _ { \kappa } ) } { \mathrm { A L G } _ { A } ( \sigma _ { \kappa } ) } } = \kappa .
$$

If κ is rational, all interval endpoints can be chosen rational. In particular, for this same A and R, every $\varepsilon > 0$ admits a rational-endpoint input with at most $\lceil R \rceil + 3$ requests, unique optimum, perfect predictions, and ratio greater than $d - \varepsilon .$

Proof. Fix $\kappa \in ( 2 , d )$ , and set

$$
t = \frac { \kappa - 1 } { \kappa } , \qquad s = \frac { ( \kappa - 1 ) ^ { 2 } } { \kappa } , \qquad \delta = \frac { 1 - s } { 2 } = \frac { 3 - \kappa - \kappa ^ { - 1 } } { 2 } , \qquad m = \lceil R \rceil , \qquad \ell = \frac { s } { m } .\tag{A.5}
$$

Since $2 < \kappa < d ,$ we have $\kappa ^ { 2 } - 3 \kappa + 1 < 0$ , and hence $0 < s < 1$ and $\delta > 0$ . Moreover,

$$
0 < t < 1 , \qquad t - 2 \delta = \kappa - 2 > 0 , \qquad s + t = \kappa - 1 > 1 .\tag{A.6}
$$

In particular, $0 < \delta < t / 2 < 1 / 2$ . Also,

$$
m \ge R > R s , \qquad 0 < \ell = \frac { s } { m } \le \frac { s } { R } < \frac { 1 } { R } .\tag{A.7}
$$

Thus the same $m ,$ chosen solely from R, works for every κ in the stated range.

Use the requests $J , H _ { 1 } , \dots , H _ { m } , L , U$ defined in $( \mathrm { A } . 2 ) \mathrm { - } ( \mathrm { A } . 4 )$ and the preceding definition of $^ { J , }$ now with the parameters (A.5). The prefix argument of Theorem 2 applies because $\ell < 1 / R \colon$ the algorithm must retain J and permanently reject every $H _ { k }$ . Explicitly, a first abandonment at $H _ { k }$ would leave value at most $\ell ,$ while stopping at that prefix leaves the offline request J of weight one, contradicting (A.1).

The geometry is unchanged: the positive requests are mutually compatible, each conflicts with the sole negative request J, and the positive total weights in the two complete inputs are

$$
\mathrm { O P T } ( \sigma ^ { - } ) = s + t = \kappa - 1 > 1 , \qquad \mathrm { O P T } ( \sigma ^ { + } ) = s + t + 1 = \kappa > 1 .
$$

They are therefore the unique optima, so both inputs have perfect predictions.

If A accepts $L ,$ choose $\sigma ^ { - }$ . Its only possible nonempty final set is $\{ L \}$ , and (A.1) excludes an empty final set. Hence its value is exactly t, giving

$$
\frac { \mathrm { O P T } ( \sigma ^ { - } ) } { \mathrm { A L G } _ { A } ( \sigma ^ { - } ) } = \frac { \kappa - 1 } { ( \kappa - 1 ) / \kappa } = \kappa .
$$

If A rejects $L ,$ choose $\sigma ^ { + }$ . Its only possible nonempty final sets are $\{ J \}$ and $\{ U \}$ , each of weight one. Again (A.1) excludes the empty set, so the ratio is exactly κ. As in Theorem 2, determinism and unknown termination allow the appropriate complete input to be fixed for $A ,$ with no change to its labels.

The two branches use $m + 2$ and $m + 3$ requests, respectively. If κ is rational, then $t , s , \delta ,$ , and $\ell = s / m$ are rational, so every endpoint in the construction is rational. Finally, for any $\varepsilon > 0$ , choose a rational

$$
\kappa \in \big ( \operatorname* { m a x } \{ 2 , d - \varepsilon \} , d \big ) .
$$

The construction gives the asserted rational witness with the same request bound.

Request count and the limiting boundary. The request bound $\lceil R \rceil + 3$ is uniform in κ and ε for fixed $R ,$ but not over all algorithms. Rational endpoints give finite descriptions without a uniform encoding-length bound as $\kappa  d$ . The gap $\delta > 0$ is essential: at $\delta = 0 , L = [ - t , 0 )$ and $U = [ 1 , 2 )$ are compatible with $J = [ 0 , 1 )$ . The two branch bounds $\mathrm { A L G } \le t$ and $\mathrm { A L G } \leq 1$ no longer follow, and replacing the $H _ { k }$ by J gives a second optimum of equal weight. The witnesses therefore keep $\delta > 0$ and only approach this limit.

## B. An all-algorithm tradeoff by binary reserve amplification

Put $\varphi = ( 1 + { \sqrt { 5 } } ) / 2 , d = \varphi ^ { 2 } ;$ , and

$$
G ( t ) = \frac { t } { ( \sqrt { t } - 1 ) ^ { 2 } } , \qquad t > 1 .\tag{B.1}
$$

The following bound applies to every deterministic algorithm in the original model.

Theorem B.1 (Universal consistency–robustness tradeoff). For everyfinite $B \geq \varphi ^ { 3 } ;$ , an algorithm with $R ( A ) \leq B$ satisfies

$$
C ( A ) \geq \operatorname* { m a x } \{ d , G ( B ) \} .\tag{B.2}
$$

In particular,

$$
R ( A ) \leq 5 \quad \Longrightarrow \quad C ( A ) \geq { \frac { 1 5 + 5 { \sqrt { 5 } } } { 8 } } > 3 ,\tag{B.3}
$$

and $R ( A ) \leq 6$ implies $C ( A ) \geq ( 4 2 + 1 2 { \sqrt { 6 } } ) / 2 5$

Corollary B.1 (Endpoint of the binary-reserve bound). An algorithm with $C ( A ) = d$ has

$$
R ( A ) \geq \varphi ^ { 4 } = 3 \varphi + 2 .\tag{B.4}
$$

Theorem 4 strengthens this endpoint to $4 \varphi + 3 ;$ the continuous bound above remains useful away from $C = d .$

Proof of Theorem B.1 and its corollary. We prove the theorem by constructing a finite adaptive input for every strict pair

$$
\varphi ^ { 3 } \leq r < \varphi ^ { 4 } , \qquad d \leq c < G ( r ) .\tag{B.5}
$$

All requests have positive length and all stated conflicts have positive overlap. Labels are fixed at arrival. At every prefix at which consistency is invoked, the complete positive set is a feasible unique optimum.

Exchange certificates. For a feasible positive set $P$ and a set $S$ of negative requests, let P(S) $P ( S )$ be the positive requests intersecting at least one member of S. If

$$
w ( P ( S ) ) > w ( S ) { \mathrm { ~ f o r ~ e v e r y ~ n o n e m p t y ~ f e a s i b l e ~ n e g a t i v e ~ s e t ~ } } S ,\tag{B.6}
$$

then $P$ is the unique optimum. Indeed, any feasible solution using exactly the negatives $S$ has weight at most $w ( S ) + w ( P \backslash$ $P ( S ) ) < w ( P )$ . A proper subset of $P$ also has smaller weight. In the induction below, the certificate for old negatives uses old positives and that for newly added negatives uses a disjoint group of new positives. Thus their weights can be added even when a single new positive intersects several new negatives.

The two ideal transitions. Write

$$
k = c - 2 , \quad \quad A = ( r - 1 ) ( c - 1 ) - 1 ,
$$

and define

$$
F ( q ) = \frac { c } { A - r q } , \qquad T ( q ) = \frac { 1 + q } { k + 1 - q } .\tag{B.7}
$$

Since $r \geq \varphi ^ { 3 } >$ 4 and $G$ decreases with $G ( 4 ) = 4 ,$ we have $0 < k < 2$ . Also $G ( r ) < r$ for $r > 4 ,$ so $r - c > 0 .$ . On $[ 0 , k ] ,$

$$
A - r q \geq A - r k = r - c > 0 , \qquad k + 1 - q \geq 1 .
$$

The sign of $F ( q ) - q$ is the sign of $r q ^ { 2 } - A q + c$ . Its discriminant,

$$
\Delta = A ^ { 2 } - 4 r c = \left( ( r - 1 ) c - r \right) ^ { 2 } - 4 r c ,
$$

has, as a quadratic in $c ,$ roots $r / ( \sqrt { r } + 1 ) ^ { 2 } < 1$ and $G ( r )$ . Hence (B.5) gives $\Delta < 0$ and $F ( q ) > q$ . Furthermore,

$$
T ( q ) - q = \frac { q ^ { 2 } - k q + 1 } { k + 1 - q } > 0 , \qquad q ^ { 2 } - k q + 1 = ( q - k / 2 ) ^ { 2 } + 1 - k ^ { 2 } / 4 > 0 .
$$

Consequently

$$
\mu = \operatorname* { m i n } _ { 0 \leq q \leq k } \left( \operatorname* { m i n } \{ F ( q ) , T ( q ) \} - q \right) > 0 .\tag{B.8}
$$

Starting at $q = 0$ , every ideal sequence of the two transitions exceeds k within $\lfloor k / \mu \rfloor + 1 \mathrm { s t e p s }$ . This gives a uniformly finite binary response tree.

A strict initial state. Choose a sufficiently small $\delta _ { 0 } > 0$ . Present negative $J = [ 0 , 1 )$ , which robustness forces the algorithm to accept. Present positive pieces partitioning $[ \delta _ { 0 } , 1 + 2 \delta _ { 0 } )$ : prescribe the final piece $[ 1 - \delta _ { 0 } , 1 + 2 \delta _ { 0 } )$ and partition $\big [ \delta _ { 0 } , 1 - \delta _ { 0 } \big )$ into pieces shorter than $1 / r$ . Take $\delta _ { 0 } < 1 / 2$ and $3 \delta _ { 0 } < 1 / r$ . Every piece strictly intersects J. At the first departure from $J ,$ only the incoming piece could be held, contradicting robustness because historical $\mathrm { O P T } \geq 1$ . Thus all pieces are missed. Their total weight is $1 + \delta _ { 0 } > 1$ , and each conflicts with the sole negative $J ,$ so they form a unique optimum. The algorithm holds J alone.

The inductive state and terminal tail. At a live state the algorithm holds a sole negative anchor $Q = [ b , b + x )$ . Every positive has been missed; their feasible set $P$ is the unique optimum and has weight

$$
p = x + w , \qquad q = w / x > 0 .
$$

The leftmost positive endpoint is $b + \delta ,$ , where $0 < \delta < x$ , and every positive starts at or to the right of that point. Initially $( x , w , \delta ) = ( 1 , \delta _ { 0 } , \delta _ { 0 } )$

If $q > k$ , append a positive tail of length x ending at $b + \delta$ . It is compatible with $P$ and overlaps $Q$ by δ. The old exchange certificates persist after adding a compatible positive, so the full positive set remains uniquely optimal. Either nonempty response has gain $x ,$ but its optimum has weight $p + x = 2 x + w > c x$ . The empty response violates robustness. This terminates the branch.

A reserve and the accepting branch. Suppose $q \leq k .$ . Set

$$
z = ( c - 1 ) x - w , \qquad \ell = z + \delta , \qquad Z = [ b + \delta / 4 - z , b + \delta / 4 ) .\tag{B.9}
$$

Then $z \geq x > 0$ . Present $Z$ as a negative request. It overlaps $Q$ by $\delta / 4$ and is separated from P by a gap of $3 \delta / 4$ .

If Z is accepted, put $\zeta = \delta / 1 6$ and present a positive partition of

$$
[ \mathrm { l e f t } ( Z ) + \zeta , \mathrm { r i g h t } ( Z ) + 2 \zeta ) .
$$

Prescribe the final crossing piece $\mathrm { [ r i g h t } ( Z ) - \zeta , \mathrm { r i g h t } ( Z ) + 2 \zeta )$ and partition the preceding interval into pieces shorter than $x / r$ . We choose the initial perturbation so that $3 \zeta < x / r$ and $z > 2 \zeta$ at every such state. Every piece strictly intersects $Z .$

Historical $Q$ has length x, so robustness forces every piece to be missed. All new positives end at or before $b + 3 \delta / 8 < b + \delta .$ hence are compatible with $P .$

The new positive group has weight $z + \zeta > z$ and every member intersects $Z .$ For any feasible subset of historical negatives, use its old conflict certificate inside $P ;$ if it also contains $Z ,$ add this new group. The groups contain distinct positive requests, so (B.6) holds for every subset and the augmented positive set is uniquely optimal. The new state has

$$
x ^ { \prime } = z , \qquad w ^ { \prime } = x + w + \delta / 1 6 , \qquad \delta ^ { \prime } = \delta / 1 6 .\tag{B.10}
$$

Its density tends to $T ( q )$ as $\delta / x  0$ . The small right-crossing piece repairs the reserve while leaving a positive free margin at its left endpoint.

The rejecting branch and a forced reset. If Z is rejected, present positive

$$
L = [ b - z , b + \delta ) , \qquad w ( L ) = \ell .
$$

It contains $Z$ with margins $\delta / 4$ and $3 \delta / 4 ,$ , is compatible with $P ,$ and overlaps $Q$ by δ. Old $P$ certifies every old negative subset, while L alone strictly dominates Z. Thus $P + L$ is uniquely optimal. Its weight is

$$
\begin{array} { r } { p + \ell = c x + \delta , } \end{array}
$$

so retaining $Q$ violates consistency. A surviving algorithm accepts $L .$

Present negative X, ending at left(Z), of length

$$
x ^ { \prime } = r \ell - p - z + \delta = ( A - r q ) x + ( r + 1 ) \delta \geq ( r - c ) x + ( r + 1 ) \delta > 0 .\tag{B.11}
$$

It overlaps L by $\delta / 4$ and is compatible with Z and every member of $P .$ The historical feasible set $P + Z + X$ has weight $r \ell + \delta .$ , forcing replacement of L by X through robustness.

Fill $\operatorname { \mathbb { I } e f t } ( X ) + \delta / 4 , \operatorname { l e f t } ( L ) )$ with positive pieces shorter than $x ^ { \prime } / r$ . Its length is $x ^ { \prime } - \delta / 2 > 0$ by (B.11). Each piece intersects $X ,$ , and historical X forces every piece to be missed. The new positive group, consisting of L and this fill, has weight

$$
\ell + x ^ { \prime } - \delta / 2 = x ^ { \prime } + z + \delta / 2 .
$$

For the newly added negatives, the full certificates are

<table><tr><td>negative subset</td><td>positive conflict group</td><td>strict surplus</td></tr><tr><td>{Z}</td><td>L</td><td>δ</td></tr><tr><td>{x}</td><td>L + fill</td><td> $z + \delta / 2$ </td></tr><tr><td>{z, X}</td><td>L + fill</td><td> $\delta / 2 .$ </td></tr></table>

In the joint row the positive group is counted once. Combining a row with the old certificate inside P establishes (B.6) for every feasible historical negative subset. Hence the full positive set is uniquely optimal. All positives have been missed, X is the sole held request, and the new state is

$$
x ^ { \prime } = A x - r w + ( r + 1 ) \delta , \qquad w ^ { \prime } = c x + \delta / 2 , \qquad \delta ^ { \prime } = \delta / 4 .\tag{B.12}
$$

Its limiting density is $F ( q )$

One finite perturbation for the complete tree. The ideal binary tree from (B.8) has finitely many nodes. All its nonterminal anchor lengths are positive, since the transition denominators are bounded below by $r \mathrm { ~ - ~ } c$ and 1; every terminal density strictly exceeds k. For each branch, the initial state and the exact updates (B.10), (B.12) depend continuously on $\delta _ { 0 }$ near zero and converge to the corresponding ideal states.

Choose one sufficiently small $\delta _ { 0 } > 0$ simultaneously for all nodes. Then all ideal terminal excesses remain positive, al anchor lengths remain positive, and every required ratio $\delta / x$ is small enough to ensure $\delta < x , z > 2 \zeta .$ , and $3 \zeta < x / r$ as well as the initial seed inequalities. The margin is divided by 16 on acceptance and by 4 on rejection. Anchor lengths need not increase: continuity at the finitely many positive ideal lengths controls every ratio separately. If a perturbed state crosses $k$ earlier, terminate there. Otherwise it reaches a terminal level of the finite ideal tree with its strict terminal excess preserved.

All overlaps and forcing margins that vanish in the ideal limit have explicit positive finite identities. These are the overlaps $\delta / 4 , \delta , \delta / 4$ at the reserve, positive repair, and reset; the accepting repair surplus $\delta / 1 6 ;$ the two forcing surpluses $\delta ;$ and the joint reset certificate surplus $\delta / 2$ . The seed has positive surplus $\delta _ { 0 }$ and a crossing piece of length $3 \delta _ { 0 }$

Causality, all responses, and termination. Each partition has finitely many pieces. Along a surviving history the held set is a singleton and the incoming request intersects it. The legal responses are therefore to retain that singleton, replace it by the incoming request, or hold nothing. Empty responses violate robustness. Both nonempty reserve responses are continued explicitly. At all other steps the alternative response has a strict violation at the current prefix, where the adversary may stop. Voluntary revocation creates no additional available request, and rejected or revoked requests cannot be recalled. Thus the construction covers every deterministic decision and uses no assumption about the termination time. All consistency invocations have the complete positive optimum certified above. The finite response tree proves the contradiction for every strict pair (B.5).

For $\varphi ^ { 3 } \leq B < \varphi ^ { 4 }$ and every $d \leq c < G ( B )$ , the construction gives $C ( A ) > c .$ Therefore $C ( A ) \geq G ( B )$ by taking the supremum over those c. Since G is decreasing and $G ( \varphi ^ { 4 } ) = d ,$ the established universal lower bound $C ( A ) \geq d$ proves Theorem B.1 for all stated budgets. The corollary follows at the crossing. More generally, $G ( G ( t ) ) = t ,$ so for $d \leq c < G ( \varphi ^ { 3 } )$ the theorem equivalently requires $R ( A ) \geq G ( c )$ whenever $C ( A ) \leq c .$ □

Scope of the lower bound. The proof provides no matching algorithm. $\mathrm { A t } \mathrm { \Delta } c = d$ and $r = \varphi ^ { 4 }$ , the reset map $F$ has a double fixed point at $q = k ;$ its ideal orbit from zero approaches k from below. For larger r its smaller fixed point is below $k .$ Always rejecting the reserve therefore prevents this ideal construction from crossing k. This explains the limit of the present mechanism.

Corollary B.2 (An explicit finite request bound). For a strict pair (r, c) in (B.5), put

$$
k = c - 2 , \quad A = ( r - 1 ) ( c - 1 ) - 1 , \quad \Delta = A ^ { 2 } - 4 r c , \quad \quad m _ { 0 } = \operatorname* { m i n } \left\{ \frac { - \Delta } { 4 r A } , \frac { 1 - k ^ { 2 } / 4 } { k + 1 } \right\} > 0 .
$$

Set $N = \lfloor k / m _ { 0 } \rfloor + 1 a n d Q = \operatorname* { m a x } \{ \lfloor r ( c - 1 ) \rfloor + 3 , \lfloor r \rfloor + 4 \}$ . The binary-reserve adversary can use at most

$$
\lfloor r \rfloor + 4 + N Q
$$

requests on every branch. The bound depends on the strict target ratios, not on the algorithm or the choice of sufficiently small perturbation and allowed subdivisions.

Proof. On $[ 0 , k ] ,$ , completing the square gives $r q ^ { 2 } - A q + c \geq - { \Delta } / { ( 4 r ) }$ , while $0 < A - r q \leq A$ . Thus $F ( q ) - q \geq$ $- \Delta / ( 4 r A )$ . Likewise $q ^ { 2 } - k q + 1 \geq 1 - k ^ { 2 } / 4$ and $0 < k + 1 - q \leq k + 1 , \mathrm { s o } T ( q ) - q \geq ( 1 - k ^ { 2 } / 4 ) / ( k + 1 )$ . Every ideal branch therefore terminates in at most N transitions. The common finite-tree perturbation in the proof preserves this depth; a perturbed branch reaching $q > k$ sooner terminates sooner.

Partition a segment of length L into $\lfloor { L } / { u } \rfloor + 1$ equal pieces whenever every piece must be strictly shorter than u. The seed uses the anchor, at most $\lfloor r \rfloor + 1$ interior pieces, and one prescribed crossing piece. On an accepting-reserve transition, $z / x = c - 1 - q \leq c - 1$ : the reserve, at most $\lfloor r ( c - 1 ) \rfloor + 1$ interior pieces, and the crossing piece use at most $\lfloor r ( c - 1 ) \rfloor + 3$ requests. On a rejecting-reserve transition, the reserve, positive repair and negative reset are three requests. The fill has length less than $x ^ { \prime }$ , with pieces shorter than $x ^ { \prime } / r$ , so at most $\lfloor r \rfloor + 1$ pieces suffice. Finally one terminal tail is enough. Adding these counts gives the displayed bound. Early violations or voluntary deletion only shorten a branch. □

## C. Source-first budget bounds with two reserve lengths

Put $c = d + \gamma , k = d + \gamma r$ , and $\beta = 4 \varphi + 3$ . We use two constructions, distinguished by the length of a rejected right reserve. Their limiting packing values are

$$
\begin{array} { l l } { \displaystyle { F _ { \mathrm { s } } ( \gamma , r ) = 2 V _ { \mathrm { s } } + \left( 1 + V _ { \mathrm { s } } \right) \left( \frac { 1 } { k } + \frac { 1 } { k - 2 } \right) , \qquad } } & { \displaystyle { V _ { \mathrm { s } } = \frac { 1 } { c - 2 } , } } \\ { \displaystyle { F _ { \mathrm { l } } ( \gamma , r ) = V _ { \mathrm { l } } \left( 1 + \frac { 1 } { c - 2 } \right) + \left( 1 + V _ { \mathrm { l } } \right) \left( \frac { 1 } { k } + \frac { 1 } { k - 2 } \right) , \qquad } } & { \displaystyle { V _ { \mathrm { l } } = \frac { 1 } { k - 2 } . } } \end{array}\tag{C.1}
$$

The short-reserve construction allows $0 \leq \gamma \leq 1 / 2 0 $ ; the long-reserve construction allows $0 \leq \gamma \leq 1 / 1 0 0$ . Let $D _ { \mathrm { s } } ( \gamma )$ and $D _ { 1 } ( \gamma )$ solve

$$
D _ { j } ( \gamma ) = F _ { j } ( \gamma , D _ { j } ( \gamma ) ) , \qquad j \in \{ \mathrm { s } , \mathrm { l } \} .\tag{C.2}
$$

Theorem C.1 (Necessary robustness budgets). The roots above are unique in $[ 6 , 2 \varphi ^ { 3 } ]$ and $[ 8 , \beta ] ;$ , respectively. Each is continuous and strictly decreasing on its stated domain, with $D _ { \mathrm { s } } ( 0 ) = 2 \varphi ^ { 3 }$ and $D _ { 1 } ( 0 ) = \beta$ . In the model of Section 2, every deterministic algorithm withfinite robustness satisfies

$$
C ( A ) \leq d + \gamma \quad \Longrightarrow \quad R ( A ) \geq D _ { j } ( \gamma )
$$

on the corresponding domain. Every perfect terminal used below has a unique full-history optimum equal to the complete positivefamily.

For fixed $\gamma ,$ each $F _ { j }$ is nonincreasing in $r ,$ so $F _ { j } - r$ is strictly decreasing. Each $F _ { j }$ strictly decreases with γ at fixed positive $r _ { \ast }$ At the upper root endpoints its value is at most that endpoint, with equality when $\gamma = 0$ . For the short construction, $1 3 / 5 < d < 2 1 / 8$ gives

$$
F _ { \mathrm { s } } ( \gamma , 6 ) > \frac { 8 0 } { 2 7 } + \frac { 6 7 } { 2 7 } \left( \frac { 4 0 } { 1 1 7 } + \frac { 4 0 } { 3 7 } \right) > 6 .
$$

For the long construction, $d < 1 3 1 / 5 0$ gives $k < 2 7 / 1 0 , c < 2 6 3 / 1 0 0 \mathrm { a t } r = 8 ,$ , and hence

$$
F _ { 1 } ( \gamma , 8 ) > \frac { 1 0 } { 7 } \left( 1 + \frac { 1 0 0 } { 6 3 } \right) + \frac { 1 7 } { 7 } \left( \frac { 1 0 } { 2 7 } + \frac { 1 0 } { 7 } \right) = \frac { 1 0 6 7 0 } { 1 3 2 3 } > 8 .
$$

Continuity proves existence and uniqueness. The strict parameter decrease and uniqueness on a compact interval prove the root properties.

## C.1. A finite completion

For a feasible positive family $\mathcal { P } _ { \cdot }$ , let $\Gamma _ { \mathcal { P } } ( S )$ be the positive identities conflicting with a negative family S. The condition

$$
w ( \Gamma _ { \mathcal P } ( \mathcal S ) ) > w ( \mathcal S ) \quad \mathrm { f o r ~ e v e r y ~ n o n e m p t y ~ f e a s i b l e ~ n e g a t i v e ~ f a m i l y ~ } \mathcal S\tag{C.3}
$$

makes $\mathcal { P }$ the unique full-history optimum: a packing using $s$ has weight at most $w ( \mathcal { P } ) - w ( \Gamma _ { \mathcal { P } } ( S ) ) + w ( S )$ , and a proper subset of P also has smaller weight.

Lemma C.1 (Completion with finite consistency slack). Let $c = d + \gamma , \gamma \geq 0 ,$ , and $r > 1$ . Suppose the complete positive family P is feasible, ends at or before p, and has weight $P _ { 0 }$ . The negatives are $\mathcal { N }$ and an unavailable $R _ { 0 } = [ p , p + z )$ $z > 0$ . Suppose (C.3) holds for every nonempty feasible subset ofN. The held weight is at most $a > 0$ , and a historical feasiblefamily has weight at least a. Iffinite $h , \zeta > 0$ satisfy

$$
h > ( r - 1 ) a ,
$$

$$
h > 2 \zeta ,
$$

$$
\alpha h > z + 2 \zeta ,
$$

$$
p + \alpha h - \zeta > e \nu e r y o l d r i g h t e n d p o i n t ,
$$

$$
P _ { 0 } - c a - \gamma h - 2 \zeta > 0 ,\tag{C.4}
$$

there is a finite continuation causing an r-robustness violation or a c-consistency violation at a unique-perfect terminal.

Proof. Set $b = p + \alpha h - \zeta$ and offer negative $H = [ b , b + h )$ , disjoint from all old requests. Omitting it leaves gain at most $a < ( a + h ) / r$ , so it must be retained. Offer a finite positive partition $\mathcal { F }$ of $[ b + \zeta , b + h - \zeta )$ , each tile shorter than $( a + h ) / r - a . \mathrm { ~ A ~ }$ first departure from H leaves at most a plus the incoming tile, so robustness forces every tile to be missed.

Offer positive $U = [ p , p + \alpha h )$ . If retained, stop. The gain is at most $a + \alpha h$ , while the complete positives weigh $P _ { 0 } + \varphi h - 2 \zeta$ . Their consistency excess is $P _ { 0 } - c a - \alpha \gamma h - 2 \zeta > 0$ . If U is rejected and H discarded, stop for a robustness violation. Otherwise offer positive $V _ { 1 } = [ b + h - \zeta , b + 2 h - \zeta )$ and stop. Every response has gain at most $a + h$ , whereas the positives weigh $P _ { 0 } + d h - 2 \zeta .$ with excess $P _ { 0 } - c a - \gamma h - 2 \zeta > 0 .$

The old positive group strictly pays every feasible subset of ${ \mathcal { N } } .$ . The disjoint new group $\{ U \} \cup { \mathcal { F } }$ strictly pays $R _ { 0 } ,$ since αh $> z ;$ it pays H or $\{ R _ { 0 } , H \}$ , since its weight $\varphi h - 2 \zeta$ exceeds $h + z$ . This covers every mixed negative family without double counting. Adding $V _ { 1 }$ preserves the certificate. All partitions are finite, no old request is recalled, and voluntary deletion only reduces the gain bounds. □

## C.2. One construction with short or long reserves

Figure 3 shows the surviving long-reserve branch and the compatible packing used at its terminal state.

Proof of Theorem C.1. Choose a construction $j$ in its stated domain and suppose $C ( A ) \leq c$ and $R ( A ) < D _ { j } ( \gamma )$ . Choose $r > R ( A )$ strictly below the root, with $r \geq 6$ for j = s or r ≥ 8 for j = l. Then $F _ { j } ( \gamma , r ) > r$ . Write $V = V _ { j } , P = 1 + V$ and set

$$
\begin{array} { c } { { \Xi _ { \mathrm { s } } = \displaystyle \frac { 1 0 } { k - 2 } + \frac { 6 } { k } + 1 2 , \qquad } } \\ { { 0 < \varepsilon < \displaystyle \operatorname* { m i n } \left\{ \frac { 1 } { 1 0 0 0 0 } , \frac { F _ { j } ( \gamma , r ) - r } { \Xi _ { j } } \right\} . \qquad } } \end{array} \ U _ { \mathrm { } } = \frac { 1 0 } { k - 2 } + \frac { 6 } { k } + \frac { 4 } { c - 2 } + 1 7 ,\tag{C.5}
$$

Use common parameters

$$
\begin{array} { l } { { v = V - 4 \varepsilon , \qquad p = 1 + v - 2 \varepsilon = P - 6 \varepsilon , } } \\ { { \begin{array} { l } { { t = p / k - \varepsilon , \qquad } } \end{array} } } \\ { { \begin{array} { l } { { \ell = ( p - 4 \varepsilon ) / ( k - 2 ) , } } \end{array} } } \end{array}\tag{C.6}
$$

and reserve length

$$
z = \left\{ { v - \varepsilon } , \ \begin{array} { l l } { v - \varepsilon , } & { j = \mathrm { s } , } \\ { { v / ( c - 2 ) - 1 0 \varepsilon } , } & { j = \mathrm { l } . } \end{array} \right.
$$

Both cases satisfy $v > 1 , p > 2 , 4 / 5 < t < 1 - 2 \varepsilon , \ell > 2$ , and $\varepsilon < v / ( 4 r )$ . In the short case $z < v < \varphi ;$ in the long case v $< z < d .$ . To check the uniform bounds, in the short case $V > 4 0 / 2 7 , p > 6 7 / 2 7 - 6 / 1 0 0 0 0$ and $k < 6 1 / 2 0$ imply $t > 4 / 5 ;$ in the long case $V > 2 5 / 1 8 , p > 2 . 3 8$ and $k < 6 8 / 2 5$ do so. In both cases $p \leq d - 6 \varepsilon$ , so $t \leq 1 - ( 6 / k + 1 )$ ε and $p - v - t \geq ( 6 / k - 1 ) \varepsilon > 0$

Missed sources and a rejected right reserve. Offer negative $K = [ 1 - \varepsilon , 1 - \varepsilon + v )$ and a positive partition $\mathcal { A } _ { \mathrm { 0 } }$ of $[ 1 , p )$ each tile shorter than $v / r$ . Finite robustness first forces $K$ to be retained; at a first departure from it, only the incoming tile could remain. Historical K would then give a ratio exceeding r. Thus every tile is missed and K remains alone.

Offer negative $R _ { 0 } = [ p , p + z )$ , overlapping K by ε. If retained, offer positive $S = [ 1 - \varepsilon , 1 )$ , then a positive partition $\mathcal { F }$ of $[ p , p + z - \varepsilon )$ whose first tile is $[ p , p + 2 \varepsilon )$ and all tiles are shorter than $v / ( 2 r )$ . Discarding $R _ { 0 }$ on S leaves at most $\varepsilon < v / r ;$ a first departure on a tile leaves at most ε plus that tile, also below $v / r$ . Hence $R _ { 0 }$ is retained throughout. Offer positive $T _ { R } = [ p + z - \varepsilon , p + 2 z - \varepsilon )$ and stop. Every response has gain at most $z + \varepsilon .$ and the complete positive weight is $v + 2 z - 2 \varepsilon$ . Its excess over c times the gain bound is

$$
\left\{ \begin{array} { l l } { ( 3 - c ) v - 4 \varepsilon > 0 , } & { j = \mathrm { s } , } \\ { ( 9 c - 2 2 ) \varepsilon > 0 , } & { j = \mathrm { l } . } \end{array} \right.
$$

The positives are compatible. The group $S , A _ { 0 }$ and the first new tile conflicts with K and weighs $v + \varepsilon > v$ . The group $\mathcal { F } , T _ { R }$ conflicts with $R _ { 0 }$ and weighs $2 z - \varepsilon > z .$ . Since $K , R _ { 0 }$ conflict, these checks cover every feasible negative family. The terminal is unique-perfect, contradicting consistency. Consequently $R _ { 0 }$ is rejected and K retained. The hypothetical positive continuation is absent from this rejection history.

Forcing a positive switch. Offer positive $J = [ 0 , 1 )$ and write $\mathcal { A } = \{ J \} \cup \mathcal { A } _ { 0 }$ , of weight p. Suppose J is rejected. Offer positive $T = [ p , p + v )$ and put $E = p + v$ . The only possible local holders are K and T, which conflict and each weigh v. For the short reserve, stop: $T$ strictly pays $R _ { 0 }$ because $v > z ,$ , and A pays K. The positive set tiles $[ 0 , E )$ , is uniquely optimal, and has consistency excess

$$
\begin{array} { r } { E - c v = \big ( 4 ( c - 2 ) - 2 \big ) \varepsilon > 0 . } \end{array}
$$

For the long reserve, T alone does not pay $R _ { 0 }$ . Put $\chi = 2 \alpha - 1 > 0 , \zeta = \chi \varepsilon / 4$ and $h = ( r - 1 ) v + \varepsilon$ . With $b = E + \alpha h - \zeta .$ offer negative $H = [ b , b + h )$ , a positive partition $\mathcal { F } _ { H }$ of $[ b + \zeta , b + h - \zeta )$ with tiles shorter than $( v + h ) / r - v .$ , and then positive $U = [ E , E + \alpha h )$ . Here $\alpha h > 7 \alpha > d + 2 \zeta > z + 2 \zeta$ , so H is beyond all old requests. Historical $\{ K , H \}$ forces H to be retained while its tiles are missed. If U is retained, stop; otherwise H must remain, and we offer $V _ { 1 } = [ b + h - \zeta , b + 2 h - \zeta )$ and stop. Gains are at most $v + \alpha h$ and $v + h ,$ respectively. The second consistency excess, which is no larger than the first, is

$$
E - c v - \gamma h - 2 \zeta = \left( 4 ( k - 2 ) - 2 - \gamma - \chi / 2 \right) \varepsilon > 0 ,
$$

since its coefficient is at least $3 \chi / 2 - 1 / 1 0 0 > 0 .$

For full-history optimality, the negative families are subsets of $\{ K , H \} \ \mathrm { o r } \ \{ R _ { 0 } , H \}$ . The following payment groups have positive surplus:

<table><tr><td></td><td>Negative family Conflicting positive group Surplus</td><td></td></tr><tr><td>K</td><td>A</td><td> $p - v$ </td></tr><tr><td> $R _ { 0 }$ </td><td> $T , U$ </td><td> $v + \alpha h - z$ </td></tr><tr><td>H</td><td> $U , { \mathcal { F } } _ { H }$ </td><td> $\alpha h - 2 \zeta$ </td></tr><tr><td> $R _ { 0 } , H$ </td><td> $T , U , { \mathcal { F } } _ { H }$ </td><td> $v + \alpha h - z - 2 \zeta$ </td></tr></table>

The groups for K and H are disjoint, covering the remaining joint family. The shared $U$ is counted only once for $R _ { 0 } , H .$ Adding $V _ { 1 }$ preserves all certificates. Thus both terminals are unique-perfect. In either reserve construction, J must be accepted and retained alone.

Forcing rejection of the interior and left negatives. Offer negative $I = [ \varepsilon , \varepsilon + t )$ , disjoint from $K , R _ { 0 }$ and conflicting with J. If accepted, apply Lemma C.1 with positives A, held bound $a = t ,$ local negatives $\{ K , I \}$ , boundary reserve $R _ { 0 }$ and historical witness I. The positive J strictly pays I because $t < 1$ , and A pays K and $\{ K , I \}$ because $p - v - t > 0$ Take $h \ = \ ( r - 1 ) t + \varepsilon$ and $\zeta = \varepsilon / 4$ . In the short case αh $> ( 3 / 5 ) 5 ( 4 / 5 ) = 1 2 / 5 > z + 2 \zeta ;$ in the long case $\alpha h > ( 3 / 5 ) 7 ( 4 / 5 ) = 8 4 / 2 5 > z + 2 \zeta$ . The anchor therefore lies beyond all old endpoints, and the deficit is

$$
p - c t - \gamma h - 2 \zeta = ( k - \gamma - 1 / 2 ) \varepsilon > 0 .
$$

The lemma rules out acceptance of I.

Now offer negative $L = [ s , \varepsilon )$ , touching I, disjoint from $K , R _ { 0 }$ , and conflicting with J. If accepted, set $\eta = \varepsilon / 4$ , offer a positive partition of $[ s + \eta , 0 )$ with tiles shorter than $\ell / r ,$ , and then positive $B = [ s + \eta - \ell , s + \eta )$ . Historical L forces the tiles to be missed. Their union with B is a feasible positive family $\mathcal { Z }$ of weight $2 \ell - \varepsilon - \eta > \ell ,$ all conflicting with L. Every response to B leaves gain at most ℓ.

Apply the lemma with positives ${ \mathcal { A } } \cup { \mathcal { Z } } , a = \ell ,$ local negatives $\{ K , I , L \}$ , boundary reserve $R _ { 0 }$ and witness L. For {I}, the conflicting group is $\Gamma ( I ) = \{ J \}$ and $w ( J ) = 1 > t$ . For $\{ K \}$ and $\{ K , I \}$ , A pays with surplus at least $p - v - t > 0$ The disjoint group $\mathcal { Z }$ pays L. Combining these certificates covers $\{ I , L \}$ and every other nonempty subset of the three compatible local negatives without double counting. Take

$$
h = ( r - 1 ) \ell + \varepsilon , \qquad \zeta = \varepsilon / 4 .\tag{C.7}
$$

Since $\ell > 2$ , the geometric requirements are stronger than those just checked for I. The deficit is

$$
p + 2 \ell - \varepsilon - \eta - c \ell - \gamma h - 2 \zeta = ( 9 / 4 - \gamma ) \varepsilon > 0 .
$$

The lemma rules out acceptance of $L ,$ including the joint $R _ { 0 } , H$ payment without reusing a boundary positive.

![](images/5a27a20813dfa1c52b7886e2e9734782c4135cdf6b630fd07d419f002fbb92e5.jpg)  
Total → 4φ + 3; finite witness has weight > 9.4

b  
![](images/9cd2f357e4e88481bd257dc04095ecb2c15964f3ab2b3c0519ba8e248b41e034.jpg)  
Algorithm retains only J: ALG = 1

Figure 3. Surviving branch with a long right reserve, at $\gamma = 0 , r = 9 .$ 4 and $\varepsilon = 1 / 2 0 0 0 0$ . Endpoint gaps are exaggerated: the schematic is not to scale, and hatching marks positive-length overlaps. Positive tiles arrive before the reserve. The forced responses retain K, reject $R _ { 0 } ,$ switch to $J ,$ and reject $I , L .$ . The actual packing $\bar { L , I , A _ { 0 } , R _ { 0 } }$ approaches $4 \varphi + 3 ,$ , while the algorithm retains value one. If J is rejected, the additional positive T and remote completion pay the entire longer reserve; this branch is separate from the displayed history. Packing labels give the limiting lengths as $\varepsilon \downarrow 0$ , not the finite instance’s exact lengths.

The actual robustness terminal. Stop after L is rejected. The algorithm holds only $^ { J , }$ of value one. The actual requests $L , I , { \mathcal { A } } _ { 0 } , R _ { 0 }$ form a compatible packing of weight

$$
\ell + t + ( p - 1 ) + z = F _ { j } ( \gamma , r ) - \Xi _ { j } \varepsilon > r .
$$

This packing is uniquely optimal. Without $K$ , choosing L, I beats J and permits every member of $\mathcal { A } _ { \mathrm { 0 } }$ and $R _ { 0 } ;$ a packing containing $K$ has weight at most $\ell + t + v$ , which is smaller because $( p - 1 ) + z > v$ . This violates robustness. Al partitions are finite; each branch retains its original labels and complete negative history. Voluntary deletion either reduces the displayed gain or violates robustness at an earlier prefix. Determinism fixes which finite branch is taken. □

## C.3. Consequences for the consistency infimum

For $j \in \{ \mathrm { s } , \mathrm { l } \} \mathrm { p u t } \gamma _ { \mathrm { s } } = 1 / 2 0 , \gamma _ { \mathrm { l } } = 1 / 1 0 0$ , and define

$$
H _ { j } ( B ) = \left\{ \begin{array} { l l } { d + \gamma _ { j } , } & { B \leq D _ { j } ( \gamma _ { j } ) , } \\ { d + D _ { j } ^ { - 1 } ( B ) , } & { D _ { j } ( \gamma _ { j } ) < B < D _ { j } ( 0 ) , } \\ { d , } & { B \geq D _ { j } ( 0 ) . } \end{array} \right.
$$

Taking parameters approaching equality from below in the theorem gives

$$
C ^ { * } ( B ) \geq \operatorname* { m a x } \{ d , G ( B ) , H ( B ) \} , \qquad H ( B ) = \operatorname* { m a x } \{ H _ { \mathrm { s } } ( B ) , H _ { \mathrm { l } } ( B ) \} .\tag{C.8}
$$

The G term is used on its domain $B \geq M$ . Since $D _ { \mathrm { l } }$ is continuous at zero, every $B < \beta$ has $C ^ { * } ( B ) > d$ An explicit weaker version follows by fixing $r = \beta$ and writing

$$
F _ { 1 } ( \gamma , \beta ) = 2 V + V ^ { 2 } + V / q + 1 / k + V / k , \quad V = ( d + \gamma \beta - 2 ) ^ { - 1 } , \quad q = d + \gamma - 2 .
$$

Every term is a positive product of reciprocals of increasing affine functions. Its negative derivative is nonincreasing, and at zero their sum is

$$
- \partial _ { \gamma } F _ { 1 } ( 0 , \beta ) = \beta ( 7 \varphi + 8 ) + \varphi ^ { 3 } = 8 3 \varphi + 5 3 .
$$

As $D _ { 1 } ( \gamma ) \le \beta , D _ { 1 } ( \gamma ) \ge F _ { 1 } ( \gamma , \beta ) \ge \beta - ( 8 3 \varphi + 5 3 ) \gamma$ . Thus, by taking smaller parameters and then their supremum,

$$
C ^ { * } ( B ) \geq d + \operatorname* { m i n } \left\{ { \frac { 1 } { 1 0 0 } } , { \frac { \beta - B } { 8 3 \varphi + 5 3 } } \right\} > d \qquad ( B < \beta ) .
$$

The short-reserve curve remains useful farther from d (Section 3). Its corresponding explicit bound $D _ { \mathrm { s } } ( \gamma ) \geq 2 \varphi ^ { 3 } - ( 3 2 \varphi +$ $2 0 ) \gamma$ follows by the same derivative argument, and is retained by (C.8).

$\mathbf { A } \mathbf { t } \ \gamma = 0$ the long construction permits $v  \varphi , z  \varphi ^ { 2 } , t  1$ and $\ell \to \varphi ^ { 3 }$ . The larger reserve is possible because a separate completion pays it when J is rejected. These four parameter inequalities give the limiting value $\beta .$ Further improvement requires additional compatible missed mass on the continuing history or different forcing decisions. A larger counterfactual completion cannot itself supply that mass.

## D. Source measures and shaped replacement costs

Write $\alpha = 1 / \varphi , d = \varphi ^ { 2 }$ , and $M = \varphi ^ { 3 }$ . We prove a common capacity theorem for $a \in [ \alpha , 1 ] ; a = \alpha$ recovers the interface used by Shaped-BANK, SUM-BANK, and Credit-BANK. Set

$$
\begin{array} { c } { { \beta _ { a } = a ^ { - 1 } , \quad C _ { a } = ( 1 + a ) ^ { 2 } , \quad q _ { 0 , a } = 1 + \beta _ { a } , \quad Q _ { a } = 1 + a + \beta _ { a } , } } \\ { { m _ { a } = \operatorname* { m a x } \{ ( 1 + a ) / a ^ { 2 } , Q _ { a } \} , \quad \quad k _ { a } = a m _ { a } = \operatorname* { m a x } \{ 1 + \beta _ { a } , 1 + a + a ^ { 2 } \} , } } \\ { { \theta _ { a } = \displaystyle \frac { a } { 1 + a } , \quad \quad h _ { 0 , a } = \displaystyle \frac { m _ { a } + 1 } { m _ { a } - \beta _ { a } } , \quad \quad h _ { 1 , a } = 1 + \beta _ { a } h _ { 0 , a } . } } \end{array}\tag{D.1}
$$

These constants are fixed for an entire execution. The ordinary gate adds the actual costs of all current conflicts and has priority. The special gate accepts a positive arrival only if every conflict is negative and $w \ge a W , w \ge a b _ { D }$ . The simultaneous fees are those in $( 8 )$ , with α replaced by a and the ledger-test condition replaced by $w < a b _ { D }$ . Full inheritance and equality acceptance are unchanged.

## D.1. A common consistency interface

Theorem D.1 (Parametric feasible-positive capacity). Let the fixed finite nonnegative ordinary costs satisfy

$$
\begin{array} { l l } { { q _ { + } ( v , b ) \geq m _ { a } v } } & { { \qquad ( v > 0 , b \geq 0 ) , } } \\ { { q _ { - } ( v , b ) \geq q _ { 0 , a } v } } & { { \qquad ( v > 0 , b \geq 0 ) , } } \\ { { q _ { - } ( v , b ) \geq \operatorname* { m a x } \{ q _ { 0 , a } v , v / a + b / ( 1 + a ) \} } } & { { \qquad ( v > 0 , 0 \leq b \leq C _ { a } v ) . } } \end{array}\tag{D.2}
$$

(D.3)

For every finite execution whose complete positive set $P$ is feasible, each negative held identity satisfies $b ( J ) \leq C _ { a } w ( J )$ and each positive one satisfies $b ( J ) \leq k _ { a } w ( J )$ throughout its lifetime. Consequently $w ( P ) \leq C _ { a } \mathrm { A L G }$

The complete set includes future positive requests and previously rejected or revoked identities. The theorem assumes no monotonicity or upper bound on the costs; the robustness proof imposes these on the capped cost.

Event times and conserved sources. Fix one finite execution with arrivals numbered $1 , \ldots , N$ and feasible complete positive set $P .$ . Write $S _ { n }$ for the held identities immediately after arrival n. An accepted identity X has birth index $\nu _ { X }$ and first revocation index $\delta _ { X } ;$ ; set $\delta _ { X } = N + 1$ if it survives. Its live snapshots are precisely $\nu _ { X } \leq n < \delta _ { X }$ . A rejection fee at arrival n uses the old state $S _ { n - 1 }$ ; an accepted parent’s inherited ledger uses this same state, before adding its own positive source.

For an arrived positive source $T ,$ , let $a _ { X , n } ( T )$ be its conserved coefficient in the ledger of held identity X. Acceptance merges coefficients and adds a unit coefficient for a positive arrival; rejection distributes a unit coefficient by its fees. Thus

$$
a _ { X , n } ( T ) \geq 0 , \qquad \sum _ { X \in { \cal S } _ { n } } a _ { X , n } ( T ) = 1 , \qquad b _ { n } ( X ) = \sum _ { T \in { \cal P } } a _ { X , n } ( T ) w ( T ) .
$$

The unit-sum identity is for sources arriving by $n ;$ coefficients of future sources are zero. Define

$$
\mu _ { X , n } ( E ) = \sum _ { T \in P } a _ { X , n } ( T ) | T \cap E | .\tag{D.4}
$$

Compatibility of the sources and coefficients at most one imply that this measure has density at most one. For $X = [ x , x + v )$ put $E _ { L , n } ( X ) = \mu _ { X , n } ( ( - \infty , x ) )$ and $E _ { R , n } ( X ) = \mu _ { X , n } ( [ x + v , \infty ) )$ ). For a cut $s ,$ the whole-source mass is

$$
L _ { n , X } ( s ) = \sum _ { T \in P : f ( T ) \leq s } a _ { X , n } ( T ) w ( T ) .
$$

It counts complete shares of sources ending by $s ,$ whereas the measure of $( - \infty , s )$ may include a partial share of a source crossing s. In particular, the former is at most the latter. A subscript 0 below denotes the birth snapshot $\nu _ { X } ;$ omitted time subscripts refer to the single live snapshot being considered. Every left-side assertion also has its reflected right-side form.

Exterior mass and one-event fees. Internal mass is at most the holder’s length. A fixed endpoint is crossed by at most one positive source; only a post-birth source crossing that endpoint can increase exterior mass on that side. Feasibility excludes positive held conflicts of a positive arrival. A negative receiver of length v receives $g \textless \textit { a v }$ when the special length test fails, or $g \leq a b _ { \mathrm { p r e } }$ when the ledger test fails. The latter gives

$$
g \leq \theta _ { a } ( b _ { \mathrm { p r e } } + g ) .\tag{D.5}
$$

The exterior contribution is bounded both by its fee and by the length of the source’s exterior segment.

Geometry independent of capacity. Let $a _ { c } \in \left( \alpha , 1 \right)$ solve $a _ { c } ^ { 3 } + a _ { c } ^ { 2 } = 1$ . Directly from (D.1),

$$
\begin{array} { c } { { \frac { a \leq a _ { c } } { m _ { a } } \frac { a \geq a _ { c } } { ( 1 + a ) / a ^ { 2 } } \frac { a \geq a _ { c } } { Q _ { a } } } } \\ { { k _ { a } } } \\ { { h _ { 0 , a } } } \\ { { h _ { 1 , a } } } \end{array}
$$

The branches agree at $a _ { c }$ . Throughout the range,

$$
\begin{array} { c } { { 1 + \beta _ { a } \le k _ { a } \le C _ { a } , ~ m _ { a } \ge Q _ { a } > q _ { 0 , a } \ge 2 , ~ h _ { 0 , a } \ge 1 + a , } } \\ { { \displaystyle \frac { h _ { 1 , a } } { m _ { a } } = h _ { 0 , a } - 1 , ~ \displaystyle \frac { h _ { 0 , a } } { q _ { 0 , a } } \le h _ { 0 , a } - 1 , ~ \displaystyle \frac { \beta _ { a } } { m _ { a } } < 1 , ~ \displaystyle \frac { 1 + a } { a ^ { 2 } m _ { a } } \le 1 . } } \end{array}\tag{D.6}
$$

For $k _ { a } \leq C _ { a } .$ , the first branch is equivalent to $a ^ { 3 } + 2 a ^ { 2 } - 1 \geq 0$ , which vanishes at α and increases thereafter; the other branch has slack a.

Let $H ( X )$ be the sum of lengths of all accepted identities in the replacement subtree rooted at X. For arbitrary labels,

$$
H ( X ) \leq h _ { 0 , a } w ( X ) \quad \mathrm { a t ~ a n ~ o r d i n a r y ~ b i r t h } , \qquad H ( X ) \leq h _ { 1 , a } w ( X ) \quad \mathrm { a t ~ a ~ s p e c i a l ~ b i r t h } .\tag{D.7}
$$

Indeed each child’s history divided by its actual cost is at most max $\left\{ h _ { 0 , a } / q _ { 0 , a } , h _ { 1 , a } / m _ { a } \right\} = h _ { 0 , a } - 1$ . An ordinary parent pays the sum of child costs, at most its own length; a special parent has only negative children, of total length at most $\beta _ { a }$ times its own. These recurrences prove (D.7) from empty births, without a ledger-capacity assumption. Negative nodes are always ordinary; an ordinary positive also satisfies the larger $h _ { 1 , a }$ bound. The accepted union is connected, and a negative subtree protrudes beyond either root endpoint by at most $\left( h _ { 0 , a } - 1 \right)$ times the root length.

Negative-to-negative downward edges shrink length by at least $q _ { 0 , a } ;$ a positive child of a negative node costs at least $m _ { a }$ times its length. An edge from a positive parent to a negative child expands by at most $\beta _ { a }$ . Thus successive positive nodes on a downward path have length ratio at most $\beta _ { a } / m _ { a } < 1$ . Before the first node starting below a fixed cut, a negative tail from length v displaces by less than $v / ( q _ { 0 , a } - 1 ) = a v . \mathrm { ~ A ~ }$ tail from a positive root of length $u ,$ with only negative intermediate nodes, displaces by less than $\beta _ { a } q _ { 0 , a } u / ( q _ { 0 , a } - 1 ) = ( 1 + a ) u / a $ . Each nonzero source has an accepted entry support overlapping it. At the golden endpoint these estimates also give the original accepted-history bounds

$$
H ( X ) \leq 2 w ( X ) \ ( p ( X ) = 0 ) , \qquad H ( X ) \leq M w ( X ) \ ( p ( X ) = 1 ) , \qquad H ( X ) \leq \tau ( X ) \quad ( a = \alpha ) ,\tag{D.8}
$$

where $\tau ( X )$ is the actual ordinary cost; $2 \leq \cdot$ d and the positive cost lower bound is M.

The birth-order induction. The induction order is the arrival index of accepted identities. Each identity’s claim covers all its live snapshots, including the last snapshot before a later parent’s birth. For a negative V of length v, prove capacity $b _ { n } ( V ) \leq C _ { a } v$ and

$$
E _ { L , \nu _ { V } } ( V ) , E _ { R , \nu _ { V } } ( V ) \leq a v ,\tag{D.9}
$$

$$
E _ { L , n } ( V ) , E _ { R , n } ( V ) \leq \operatorname* { m a x } \{ a v , \theta _ { a } b _ { n } ( V ) \} ,\tag{D.10}
$$

$$
L _ { \nu _ { V } , V } ( s ) + z - s \leq a v \quad { \mathrm { i f ~ } } s < z \leq t , \ z \leq s ( V ) , \ L _ { \nu _ { V } , V } ( s ) > 0 .\tag{D.11}
$$

In the last assertion $T _ { 0 } = [ s , t )$ ranges over actual positive requests arriving after $\nu _ { V }$ . It need not overlap V . For a positive Q of length u, prove capacity $b _ { n } ( Q ) \leq k _ { a } u$ and

$$
L _ { n , Q } ( s ) + z - s \leq k _ { a } u \quad { \mathrm { i f ~ } } s < z \leq t \leq s ( Q ) , L _ { n , Q } ( s ) > 0 ,\tag{D.12}
$$

where $T _ { 0 } = [ s , t )$ is an actual positive request arriving after the live snapshot n. Positive ledgers are constant during a lifetime. All cut assertions include their reflections, counting full shares of sources starting beyond the right cut.

Earlier capacities and cuts first establish a new negative node’s birth bounds. Its birth bounds and subsequent one-event fees then establish its entire-lifetime bounds. A positive node uses earlier negative assertions and the independent geometry. In particular, a new node’s capacity is never used to establish its own birth cuts.

Lemma D.1 (Weighted directional cut at a live snapshot). Let a negative $X = [ x , x + v )$ be live at snapshot n and satisfy its capacity, birth exterior, lifetime exterior, and birth-cut assertions. For an actual positive $T _ { 0 } = [ s , t )$ arriving after n, if $s < z \leq t , z < f ( X )$ , and $L _ { n , X } ( s ) > 0$ , then

$$
L _ { n , X } ( s ) + z - s \leq a q _ { - } ( v , b _ { n } ( X ) ) .\tag{D.13}
$$

The reflected assertion holds. No later parent’s capacity or cut is an assumption.

Proof. At this snapshot the cost hypothesis gives

$$
v + \operatorname* { m a x } \{ a v , \theta _ { a } b \} \leq a q _ { - } ( v , b ) .\tag{D.14}
$$

If $x < s ,$ density gives $L _ { X } ( s ) \leq E _ { L } ( X ) + s - x ;$ adding $z - s$ leaves a quantity smaller than $E _ { L } ( X ) + v$ , paid by the lifetime bound and $_ { ( \mathrm { D } . 1 4 ) }$ . ${ \mathrm { ~ f ~ } } x \geq s ,$ a source ending by s cannot newly overlap X during its lifetime, so $L _ { X } ( s ) = L _ { 0 , X } ( s )$ Set $z _ { 0 } = \operatorname* { m i n } \{ z , x \}$ . For $z _ { 0 } \ > \ s$ use the birth cut; for $z _ { 0 } = s$ necessarily $x = s ,$ and use the birth exterior bound. Thus $L _ { X } ( s ) + z _ { 0 } - s \leq a v$ . The remaining distance is zero or less than v, giving $( 1 + a ) v = a q _ { 0 , a } v \leq a q _ { - } ( v , b )$ □

Lemma D.2 (Negative birth bounds). Assume thefull induction assertionsfor every earlier accepted identity at all its live snapshots. A new negative identity satisfies the birth exterior bounds and birth cuts above.

Proof. Let the ordinary parent be V of length ℓ. Its children have old-state costs summing to at most ℓ. A negative child $X = [ x , x + v )$ contributes mass to the left of $s ( V )$ at most $E _ { L } ( X ) + \operatorname* { m a x } \{ s ( V ) - x , 0 \} \leq E _ { L } ( X ) + v \leq a q _ { - } ( v , b ( X ) )$ A positive child’s entire ledger is at most $k _ { a } v = a m _ { a } v \leq a q _ { + } ( v , b )$ . Sum to obtain the left birth exterior bound, and reflect for the right.

For the left birth cut, every child with a nonzero left share overlaps V and ends strictly beyond z. A negative child obeys Lemma D.1. A positive child is compatible with $T _ { 0 }$ and overlaps V to its right, forcing its start at least t; its earlier positive cut pays its share and the gap with $k _ { a } w ( X ) \leq a \tau ( X )$ . Sum over all children with nonzero shares. There is at least one, so the gap is paid at least once and $\begin{array} { r } { L _ { 0 , V } ( s ) + z - s \le a \sum _ { X } \tau ( X ) \le a \ell . } \end{array}$ . Zero-share children need no whole-ledger estimate. Reflection and empty births finish the proof. □

Lemma D.3 (Negative lifetime and capacity). The negative birth assertions and the subsequent rejection fees imply the lifetime exterior bounds and $b _ { n } ( V ) \leq C _ { a } w ( V )$ at every live snapshot, without assuming this capacity in advance.

Proof. Write $V = \left[ x , x + \ell \right)$ . A side with positive exterior mass at birth stays bounded by aℓ. A birth source crossing that endpoint excludes any later positive crosser. Otherwise all birth sources contributing to that side lie wholly beyond the endpoint. If a later source $T = [ s , t )$ crosses the left endpoint, those sources end by s, so $E _ { L , 0 } ( V ) = L _ { 0 , V } ( s )$ . The birth cut bounds $E _ { L , 0 } ( V ) + x - s \leq a \ell$ , and the new exterior contribution is at most $x - s .$ . No second boundary source can cross that side. Reflect for the right.

A side with zero birth exterior receives exterior mass only from its unique later boundary source. Its fee is bounded by aℓ after length failure, or by $\theta _ { a } b ( V )$ after ledger failure, using (D.5). Ledger monotonicity preserves this bound thereafter. This proves the lifetime exterior assertion.

For capacity, two birth-nonzero sides give $b \leq ( 1 + 2 a ) \ell \leq C _ { a } \ell$ by density. With exactly one such side, before a first boundary on the other side $b _ { \mathrm { p r e } } \leq ( 1 + a ) \ell .$ , hence its fee is at most $a ( 1 + a ) \ell .$ . Subsequently $b \leq \ell + a \ell + a ( 1 + a ) \ell = C _ { a } \ell .$ With neither side nonzero at birth, the first boundary has $b _ { \mathrm { p r e } } \leq \ell$ and fee at most aℓ. One side is then protected from any later crosser, and the preceding argument applies to the other side. If that first source crosses both endpoints, no later boundary source is possible; with no boundary, density suffices. These are event-time fee bounds. □

Lemma D.4 (Positive birth and cut). Assume thefull induction assertionsfor every earlier identity. A positive accepted Q satisfies capacity $b _ { n } ( Q ) \leq k _ { a } w ( Q )$ and the positive cuts throughout its live snapshots.

Proof. Let $u = w ( Q )$ and evaluate all child quantities immediately before its birth. All children are negative. Ordinary acceptance gives $W _ { Q } \ \leq \ u / q _ { 0 , a }$ and $B _ { Q } \ \leq \ C _ { a } W _ { Q } \ \leq \ a ( 1 + a ) u ;$ special acceptance gives $B _ { Q } , W _ { Q } ~ \le ~ \beta _ { a } u$ . Thus $b ( Q ) = u + B _ { Q } \leq k _ { a } u$ . No later positive request overlaps Q, so its ledger and shares remain constant.

Fix a live snapshot and an actual later positive $T _ { 0 } = [ s , t )$ with $s < z \leq t \leq s ( Q )$ and $L _ { Q } ( s ) > 0$ . Set $e = z - s$ and $G = s ( Q ) - s , \ s _ { 0 } \ e \leq G$ . The source $Q$ is not counted in $L _ { Q } ( s )$ , which therefore equals $\begin{array} { r } { \sum _ { X } L _ { X } ( s ) \le B _ { Q } } \end{array}$ . Trace one nonzero share of a source ending by s down the replacement tree to its accepted entry support. Whether the source was accepted or rejected there, that support overlaps it and starts strictly below s. Stop at the first such start. Every positive node before this crossing starts at least at $t ,$ by compatibility with $T _ { 0 }$

Case 1: a positive node occurs strictly below Q before the crossing. Let H be the last one. The independent path estimates give $w ( H ) \leq \beta _ { a } u / m _ { a }$ . Its subsequent tail has only negative intermediate nodes, hence

$$
e \leq s ( H ) - s < \frac { 1 + a } { a } w ( H ) \leq \frac { 1 + a } { a ^ { 2 } m _ { a } } u \leq u .
$$

This also permits a positive crossing node, whose cost is at least $q _ { 0 , a }$ times its length. Thus $L _ { Q } ( s ) + e \le B _ { Q } + u \le k _ { a } u$

Case 2: no such positive node occurs, and either $Q$ is ordinary or more than one child is relevant. Let V be the first child on the selected path and $v = w ( V )$ . The negative-tail displacement gives $s ( V ) - s <$ av if $s ( V ) \geq s ; \mathrm { i f } s ( V ) < s$ the inequality is immediate. Since V overlaps $Q , G < ( 1 + a ) v$ . For ordinary $Q , v \leq u / q _ { 0 , a }$ yields $e < a u ,$ , and

$$
B _ { Q } + e < ( a ^ { 2 } + 2 a ) u \leq ( 1 + a + a ^ { 2 } ) u \leq k _ { a } u ,
$$

where $a \leq 1$ . For special Q, if V is not leftmost, a compatible child to its left ends after $s ( Q )$ ; then $s ( V ) > s ( Q )$ and $G < a v \leq u$ . If V is leftmost but another child K has $L _ { K } ( s ) > 0$ , that child starts to the right of $s ( Q )$ . Its entry support lies below $s ,$ so the independent protrusion bound gives $G < ( h _ { 0 , a } - 1 ) w ( K )$ . Now

$$
\frac { 1 } { 1 + a } + \frac { 1 } { h _ { 0 , a } - 1 } = \left\{ \begin{array} { l l } { \beta _ { a } , } & { a \leq a _ { c } , } \\ { ( 1 + a ) ^ { - 1 } + a \geq \beta _ { a } , } & { a \geq a _ { c } , } \end{array} \right.
$$

with the last inequality equivalent to $a ^ { 3 } + a ^ { 2 } \geq 1$ . Consequently $\beta _ { a } G < v + w ( K ) \le W _ { Q } \le \beta _ { a } u$ . In either special subcase, $e \leq G <$ u and $B _ { Q } + e < ( 1 + \beta _ { a } ) u \leq k _ { a } u$

Case 3: Q is special and only its leftmost child V carries a left share. Then $L _ { Q } ( s ) = L _ { V } ( s )$ and $z \leq t \leq s ( Q ) < f ( V )$ If $s ( V ) \ \geq \ s ,$ , no source ending by s can charge V after birth, so $L _ { V } ( s ) ~ = ~ L _ { 0 , V } ( s )$ . Put $z _ { 0 } = \operatorname* { m i n } \{ z , s ( V ) \}$ . For $z _ { \mathrm { 0 } } \ > \ s ,$ , use the earlier birth cut. For $z _ { \mathrm { 0 } } = s _ { \mathrm { : } }$ , necessarily $s ( V ) = s ,$ , and use $L _ { 0 , V } ( s ) \leq E _ { L , 0 } ( V ) \leq a v$ instead. Thus $L _ { V } ( s ) + z _ { 0 } - s \le a v$ , while $0 \leq z - z _ { 0 } < v .$ , giving $L _ { Q } ( s ) + e < ( 1 + a ) v \leq ( 1 + \beta _ { a } ) u \leq k _ { a } u . \mathrm { ~ I f ~ } s ( V ) < s ,$ density and the earlier lifetime exterior bound give

$$
\begin{array} { r l } & { L _ { Q } ( s ) + e < v + E _ { L } ( V ) } \\ & { \qquad \leq \operatorname* { m a x } \{ ( 1 + a ) v , v + \theta _ { a } b ( V ) \} } \\ & { \qquad \leq \operatorname* { m a x } \{ 1 + \beta _ { a } , \beta _ { a } + ( 1 + a ) ^ { - 1 } \} u = ( 1 + \beta _ { a } ) u \leq k _ { a } u . } \end{array}
$$

The last inequality uses the two separate special budgets $v \le W _ { Q } \le \beta _ { a } u$ and $b ( V ) \leq B _ { Q } \leq \beta _ { a } u$ . These three cases exhaust the path and child-share alternatives. Reflection proves the right cut. □

Completion ofTheorem D.1. A new negative identity obtains its birth assertions from earlier children, then its lifetime assertions from its own actual fees. Its weighted cut is used only for later births. A new positive identity uses earlier negative assertions and independent geometry. This closes the birth induction. Source conservation and $k _ { a } \leq C _ { a }$ give $\begin{array} { r } { w ( P ) = \sum _ { X } b ( X ) \leq C _ { a } \sum _ { X } w ( X ) } \end{array}$ □

Theorem D.2 (Golden endpoint interface). For the special tests and fees at $a = \alpha ,$ , suppose $q _ { + } ( v , b ) \geq M v$ and

$$
\begin{array} { l l } { { q _ { - } ( v , b ) \geq d v } } & { { \qquad ( v > 0 , \ b \geq 0 ) , } } \\ { { q _ { - } ( v , b ) \geq \operatorname* { m a x } \{ d v , \varphi v + \alpha b \} } } & { { \qquad ( v > 0 , \ 0 \leq b \leq d v ) . } } \end{array}\tag{D.15}
$$

(D.16)

For every feasible complete positive set, each held identity satisfies $b ( J ) \leq d w ( J )$ throughout its lifetime and $w ( P ) \leq$ $d \operatorname { A L G }$

Proof. This is the $a = \alpha$ specialization of Theorem D.1: $C _ { a } = k _ { a } = q _ { 0 , a } = d , m _ { a } = M$ , and $v / a + b / ( 1 + a ) =$ $\varphi v + \alpha b$ □

## D.2. The capped family and exact consistency

Define $T _ { a }$ by the preceding a-special tests and fees, positive cost $m _ { a } v$ , and negative cost

$$
q _ { a } ( v , b ) = \operatorname* { m i n } \{ Q _ { a } v , \operatorname* { m a x } \{ q _ { 0 , a } v , v / a + b / ( 1 + a ) \} \} .\tag{D.17}
$$

Its filtered form is $T _ { a } ^ { F }$ . The cost is $q _ { 0 , a } v$ up to $b = ( 1 + a ) v$ , then linear up to $b = C _ { a } v$ , and then equal to $Q _ { a } v$ . The cap is inactive on the capacity range, so Theorem D.1 gives consistency at most $C _ { a }$

Proposition D.1 (Exact consistency of the family). For every $a \in [ \alpha , 1 ]$ , both $T _ { a }$ and $T _ { a } ^ { F }$ have $C ( T _ { a } ) = C ( T _ { a } ^ { F } ) =$ $( 1 + a ) ^ { 2 }$

Proof. Choose $0 < \epsilon < \operatorname* { m i n } \{ 1 / 1 0 0 , a ^ { 2 } / [ 4 ( 3 a + 1 ) ] \}$ and offer

$$
\begin{array} { c c l } { J } & { 0 } & { [ 0 , 1 ) , } \\ { H _ { 1 } } & { 1 } & { [ \epsilon , 1 / 2 ) , } \\ { H _ { 2 } } & { 1 } & { [ 1 / 2 , 1 - \epsilon ) , } \\ { L } & { 1 } & { [ 2 \epsilon - a , \epsilon ) , } \\ { R } & { 1 } & { [ 1 - \epsilon , 1 - \epsilon + z ) , } \end{array} \qquad z = a ( 1 + a - 3 \epsilon ) - \epsilon .
$$

The two seeds and L are shorter than a and fail ordinary acceptance, since the negative cost is at least $q _ { 0 , a } \geq 2$ . They fail the special length test and leave ledger $b _ { J } = 1 + a - 3 \epsilon$ . This is below the first breakpoint, so $q _ { a } ( 1 , b _ { J } ) = q _ { 0 , a }$ . The final request has $z > a$ but $z = a b _ { J } - \epsilon$ , hence passes the length test and fails the ledger test. It also fails ordinary acceptance because

$$
z < a + a ^ { 2 } \leq 1 + 1 / a = q _ { 0 , a } ,
$$

where the last inequality is $( a - 1 ) ( a + 1 ) ^ { 2 } \leq 0$ after multiplication by a. Thus $\mathrm { A L G } = 1$ . The positives are consecutive in spatial order $L , H _ { 1 } , H _ { 2 } , R$ . Their union contains the anchor and every offered request; they therefore form an optimum of weight $( 1 + a ) ^ { 2 } - ( 3 a + 4 ) \epsilon$ . Any feasible set containing the anchor contains none of these positives, so this optimum is unique. Letting ϵ decrease to zero proves the lower bound. The capacity theorem proves the upper bound. Filtering leaves these perfect labels unchanged and inherits the upper bound. □

At $a = \alpha$ , the family recovers Shaped-BANK T<sub>∗</sub> with

$$
\begin{array} { r } { q _ { * } ( v , b ) = \operatorname* { m i n } \{ 2 \varphi v , \operatorname* { m a x } \{ d v , \varphi v + \alpha b \} \} = \left\{ \begin{array} { l l } { d v , } & { b \leq \varphi v , } \\ { \varphi v + \alpha b , } & { \varphi v \leq b \leq d v , } \\ { 2 \varphi v , } & { b \geq d v . } \end{array} \right. } \end{array}\tag{D.18}
$$

Proposition D.2 (Smallest monotone sufficient cost). Among costs satisfying the interface of Theorem 5 and nondecreasing in bforfixed v, q is pointwise smallest.

Proof. For $b \leq d v$ , the two interface inequalities require $q ( v , b ) \geq \operatorname* { m a x } \{ d v , \varphi v + \alpha b \}$ , exactly q<sub>∗</sub> on this range. $\mathbf { A } \mathbf { t } \ b = d v$ this value is $2 \varphi v$ , so monotonicity requires at least $2 \varphi v$ for larger b. The cost $q _ { * }$ meets all requirements. □

The branches of (D.18) agree at both boundaries. Minimality is relative to this sufficient interface, which is not necessary (Appendix E.1).

## D.3. Joint-cut robustness of the family

The following common transfer lemma is also used in Appendices H and J.

Lemma D.5 (Comparator-to-root transfer). Consider a finite execution on positive-length intervals, starting from an empty held set. Acceptance revokes precisely the current conflicts and retains all nonconflicts; rejection leaves the held set unchanged, and there are no other revocations. Give each accepted identity V a weight $\omega _ { V } ~ \ge ~ w ( V )$ . Suppose every rejected I has a nonempty current conflict set $D ( I )$ and pieces $z _ { I , V }$ satisfying

$$
0 \leq z _ { I , V } \leq \omega _ { V } \quad ( V \in D ( I ) ) , \qquad \sum _ { V \in D ( I ) } z _ { I , V } = w ( I ) .
$$

Link each revoked identity to its replacement. In this forest, let $H _ { J }$ be the closure of the union in J’s subtree, including J. An ancestry antichain contains no node and its strict ancestor; let $\Omega _ { J }$ be its maximum weight in this subtree. Let $L _ { J } ( x )$ and $U _ { J } ( y )$ be the same maximum restricted respectively to starts below x andfinishes above $y ,$ both strictly; empty antichains are allowed. Set $F _ { J } ( x , y ) = | H _ { J } \cap [ x , y ] | + L _ { J } ( x ) + U _ { J } ( y )$ . If each final root satisfies

$$
\Omega _ { J } \leq K w ( J ) , \qquad F _ { J } ( x , y ) \leq K w ( J ) \quad f o r a l l r e a l x \leq y ,
$$

then every feasible subset O of the input requests satisfies $w ( O ) \leq K { \mathrm { A L G } } .$

Proof. An accepted member of O charges its length to its own identity; a rejected member uses the stipulated pieces. Pass each piece to its receiver’s final root. Its charge there is at most its own length and is positive only if it intersects that root’s hull in positive length. Each hull is connected along replacement edges.

Fix $H _ { J } = [ A , B ]$ . If one comparator covers its whole interior, allowing endpoint equality, it is the sole positive-charge contributor. Its receivers form an ancestry antichain: they were simultaneously held, or consist of its own accepted identity. Thus its charge is at most $\Omega _ { J }$ . Otherwise feasibility permits at most one comparator strictly crossing each hull endpoint. Let x be the left crosser’s finish (or A if absent), and y the right crosser’s start $( \mathrm { o r } ~ B )$ . Then $A \leq x \leq y \leq B$ . The left crosser’s receivers form an ancestry antichain with starts below x, giving charge at most $L _ { J } ( x ) ;$ the right side gives $U _ { J } ( y )$ . An absent side has zero maximum at its hull endpoint. All remaining contributors are contained in $[ x , y ]$ , and their disjoint lengths bound their total charge by $\vert H _ { J } \cap [ x , y ] \vert$ . The root charge is therefore at most $F _ { J } ( x , y )$ . Summing over roots preserves every initial piece, even when their hulls overlap, and proves the claim. □

Cost shape and terminal budgets. Let

$$
c _ { a } = \frac { a } { ( 1 + a ) ^ { 2 } } , A _ { a } = \beta _ { a } + \beta _ { a } ^ { 2 } + ( 1 + a ) ^ { - 2 } , \rho _ { a } = \operatorname* { m a x } \{ 1 , A _ { a } / m _ { a } \} .\tag{D.19}
$$

The three branches of (D.17) give

$$
q _ { 0 , a } v \leq q _ { a } ( v , b ) \leq Q _ { a } v \leq m _ { a } v ,
$$

$$
q _ { a } ( v , b ) \leq q _ { 0 , a } v + c _ { a } b ,\tag{D.20}
$$

$$
q _ { a } ( W , B ) - q _ { a } ( v , B ) \geq \beta _ { a } ( W - v ) \quad ( 0 \leq v \leq W ) .\tag{D.21}
$$

For the affine majorant the middle-branch slack is $v - b / C _ { a } \geq 0 ;$ the cap branch uses $b \geq C _ { a } v$ and $c _ { a } C _ { a } = a = Q _ { a } - q _ { 0 , a } .$ For fixed $B ,$ the successive slopes in v are $Q _ { a } , \beta _ { a } , q _ { 0 , a } ;$ the branches are continuous and $q _ { a } ( 0 , B ) = 0$ . The cost is nondecreasing in both variables.

Give each accepted identity X its ordinary cost $q _ { X }$ at the last live snapshot, immediately before revocation or at termination. These costs are nondecreasing during each lifetime. An ordinary parent of length h has $\textstyle \sum _ { i } q _ { i } \leq h$ . A special parent has only negative children and

$$
W , B \leq \beta _ { a } h , \qquad \sum _ { i } q _ { i } \leq q _ { 0 , a } W + c _ { a } B \leq A _ { a } h .\tag{D.22}
$$

Since $m _ { a } \ge q _ { 0 , a } \beta _ { a } , c _ { a } \le 1 / 4$ , and $q _ { 0 , a } \geq 2$

$$
\rho _ { a } \leq 1 + c _ { a } / q _ { 0 , a } \leq 9 / 8 < q _ { 0 , a } , \qquad \rho _ { a } - 1 < \beta _ { a } .
$$

An ancestry antichain has total weight at most $q { \mathrm { } } J$ in an ordinary-root subtree and at most max $\{ m _ { a } , A _ { a } \} w ( J ) = \rho _ { a } m _ { a } w ( J )$ in a special-root subtree. Indeed, omitting an ordinary root gives at most $\rho _ { a } \sum _ { i } q _ { i } \le \rho _ { a } h < q _ { J } . \mathrm { ~ A ~ }$ special root’s negative children are ordinary and their weights sum to at most $A _ { a } h ;$ the singleton costs $m _ { a } h$ . This simultaneous induction uses no feasible-positive capacity. It also shows that an eligible ordinary root attains its endpoint maximum, while an ineligible side has weight at most $\rho _ { a } h$

Theorem D.3 (Parametric robustness certificate). Let $K _ { a }$ be the maximum ofthe general coefficients in Table 2. For every finite arbitrarily labelled input, both $T _ { a }$ and $T _ { a } ^ { F }$ satisfy $\mathrm { O P T } \leq K _ { a } \mathrm { A L G }$ . In fact,

$$
\begin{array} { c } { { O _ { a } = m _ { a } + 1 + 2 \beta _ { a } + \beta _ { a } ^ { 2 } , \qquad N _ { a } = h _ { 1 , a } + A _ { a } + \rho _ { a } \beta _ { a } , } } \\ { { K _ { a } = \operatorname* { m a x } \{ O _ { a } , N _ { a } \} . } } \end{array}\tag{D.23}
$$

The map $a \mapsto K _ { a }$ is continuous on $[ \alpha , 1 ]$

Proof. Use Lemma D.5 with terminal cost weights $\omega _ { X } = q _ { X }$ . The independent history bounds give hull length at most $h _ { 0 , a } h$ at an ordinary root and $h _ { 1 , a } h$ at a special root. For $x \leq y$ , write $F _ { J } ( x , y ) = | H _ { J } \cap [ x , y ] | + L _ { J } ( x ) + U _ { J } ( y )$ as in that lemma. The two endpoint maxima need not coexist.

An antichain either contains the root singleton or omits it and decomposes over child subtrees. Thus

$$
L _ { J } ( x ) = \operatorname* { m a x } \left\{ q _ { J } \mathbf { 1 } _ { \left\{ s ( J ) < x \right\} } , \sum _ { i } L _ { C _ { i } } ( x ) \right\} ,
$$

with the analogous right recurrence. A zero or ineligible root term is assigned to the child choice. Ties of positive eligible values can use either attaining choice. There is no mixed root and descendant choice, since they are comparable in ancestry. The descendants of an antichain need not be spatially compatible; only immediate children are known to have been held together.

At an ordinary root, two eligible endpoints put the gap in the root and give $1 + 2 m _ { a }$ . One eligible endpoint gives $h _ { 0 , a } +$ $m _ { a } + \rho _ { a }$ , and none gives $h _ { 0 , a } + 2 \rho _ { a }$ , using the full hull.

At a special root $J = [ l , r )$ , the two endpoints choose exactly two, one, or zero singletons. Two singletons give $1 + 2 m _ { a }$ For exactly one singleton, reflect to put it on the left, so $l < x . \mathrm { H f } y < r .$ , the gap lies in J and the coefficient is $1 + m _ { a } + A _ { a }$ Otherwise $r \le y$ . Every child starts strictly before r; compatibility permits at most one child V ending beyond y. If this outward child exists, the gap is contained in $J \cup V$ , of length at most $h + v$ . Its right maximum is $q _ { V }$ , while all other child roots are right-ineligible. Hence

$$
\begin{array} { r l } & { F _ { J } \leq ( m _ { a } + 1 ) h + ( 1 - \rho _ { a } ) v + q _ { a } ( v , b _ { V } ) + \rho _ { a } W } \\ & { \quad \leq ( m _ { a } + 1 ) h + ( 1 - \rho _ { a } ) v + q _ { a } ( v , B ) + \rho _ { a } W } \\ & { \quad \leq ( m _ { a } + 1 ) h + W + q _ { a } ( W , B ) } \\ & { \quad \leq ( m _ { a } + 1 + \beta _ { a } + q _ { 0 , a } \beta _ { a } ) h . } \end{array}\tag{D.24}
$$

The third line follows from the fixed-ledger slope $\beta _ { a } > \rho _ { a } - 1$ ; the last uses monotonicity, both special budgets, and $q _ { a } ( \beta _ { a } h , \beta _ { a } h ) = q _ { 0 , a } \beta _ { a } h$ . If there is no outward child, all child roots are right-ineligible. The full hull then gives $h _ { 1 , a } +$ $m _ { a } + \rho _ { a } \beta _ { a }$

If both endpoint choices use children, call a child V a bridge when $s ( V ) < x \leq y < f ( V )$ . There is at most one compatible bridge, including at $x = y$ . Its gap and endpoint weights contribute at most $v + 2 q _ { V }$ . Every other child has

Table 2. Complete joint-cut case list. The special root is $[ l , r )$ and mixed choices are oriented with the singleton on the left. $K _ { a }$ is the maximum of the general coefficients, with the repeated $1 + 2 m _ { a }$ counted once. The endpoint column preserves the specialized Shaped-BANK calculation; the largest entry there is $5 \varphi + 3$
<table><tr><td>Root and endpoint choices</td><td>General coefficient</td><td> $\mathrm { A t } a = \alpha$ </td></tr><tr><td>Ordinary: two eligible Ordinary: one eligible Ordinary: none eligible</td><td> $1 + 2 m _ { a }$   $h _ { 0 , a } + m _ { a } + \rho _ { a }$   $h _ { 0 , a } + 2 \rho _ { a }$ </td><td> $4 \varphi + 3$   $7 \varphi - 4$   $1 0 \varphi - 1 2$ </td></tr><tr><td>Special: two singletons Special: one,  $y < r$ </td><td> $1 + 2 m _ { a }$   $1 + m _ { a } + A _ { a }$ </td><td> $4 \varphi + 3$   $3 \varphi + 5$ </td></tr><tr><td>Special: one, outward child</td><td> $m _ { a } + 1 + \beta _ { a } + q _ { 0 , a } \beta _ { a }$ </td><td> $5 \varphi + 3$ </td></tr><tr><td>Special: one, no outward child Special: children, bridge</td><td> $h _ { 1 , a } + m _ { a } + \rho _ { a } \beta _ { a }$   $\beta _ { a } + 2 A _ { a }$ </td><td> $2 \varphi + 7$   $3 \varphi + 6$ </td></tr></table>

at most one eligible endpoint, so its endpoint sum is at most $q _ { i } + \rho _ { a } v _ { i }$ ; this also covers two ineligible endpoints since $q _ { i } \geq q _ { 0 , a } v _ { i } \geq \rho _ { a } v _ { i }$ . Consequently

$$
F _ { J } \leq v + 2 q _ { V } + \sum _ { i \neq V } ( q _ { i } + \rho _ { a } v _ { i } ) \leq W + 2 \sum _ { i } q _ { i } \leq ( \beta _ { a } + 2 A _ { a } ) h ,
$$

where the middle inequality uses $( \rho _ { a } - 1 ) v _ { i } \leq q _ { i }$ . With no bridge, use the full hull and all endpoint sums to obtain $h _ { 1 , a } + A _ { a } + \rho _ { a } \beta _ { a }$ . These cases exhaust the table: root versus children at each endpoint, the unique outward child in the mixed case, and the presence or absence of a bridge in the child-only case. Strict eligibility assigns endpoint equality to ineligibility.

The full antichain bound max $\{ m _ { a } , A _ { a } \} h$ is at most $( 1 + m _ { a } + A _ { a } ) h \leq K _ { a } h .$ . A rejected request I has a nonempty conflict set and fails ordinary acceptance, so assign analytical pieces ${ \boldsymbol { v } } ( I ) { \boldsymbol { \tau } } ( { \boldsymbol { X } } ) / \sum _ { D } { \boldsymbol { \tau } }$ to its current conflicts. They sum to $w ( I )$ and are each below its receiver’s current cost, hence below $q _ { X }$ . All weights exceed the corresponding interval lengths. These analytical pieces are not ledger fees. Every hypothesis of Lemma D.5 is now verified, giving the robustness bound. The filter produces another causal labelled execution and is covered by the same bound.

It remains to reduce the table to (D.23). Put $e _ { a } = ( 1 + a ) ^ { - 2 }$ and $m _ { 0 , a } = \beta _ { a } + \beta _ { a } ^ { 2 }$ , so ${ \cal A } _ { a } = m _ { 0 , a } + e _ { a }$ and $m _ { a } \geq m _ { 0 , a } .$ The branch formulas give $h _ { 0 , a } \leq 1 + \beta _ { a } ;$ ; also $e _ { a } \le \alpha ^ { 2 } < 1 / 2$ and $1 \le \rho _ { a } \le 9 / 8 < m _ { a }$ . The gap $O _ { a } - ( 1 + 2 m _ { a } )$ is $\beta _ { a }$ for $a \leq a _ { c }$ and $\beta _ { a } + \beta _ { a } ^ { 2 } - 1 - a \ge 0$ for $a \geq a _ { c } .$ Further,

$$
\begin{array} { r l } & { O _ { a } - \left( h _ { 0 , a } + m _ { a } + \rho _ { a } \right) \geq \beta _ { a } + \beta _ { a } ^ { 2 } - \rho _ { a } \geq 7 / 8 , } \\ & { O _ { a } - \left( 1 + m _ { a } + A _ { a } \right) = \beta _ { a } - e _ { a } > 0 , } \\ & { \quad \quad O _ { a } - \left( \beta _ { a } + 2 A _ { a } \right) = m _ { a } - m _ { 0 , a } + 1 - 2 e _ { a } > 0 . } \end{array}
$$

The ordinary no-endpoint row is smaller than its one-endpoint row. The no-outward row $h _ { 1 , a } + m _ { a } + \rho _ { a } \beta _ { a }$ is at most $N _ { a }$ when $A _ { a } \geq m _ { a }$ . Otherwise $a > a _ { c } ,$ , since $A _ { a } = m _ { a } + e _ { a }$ on the first branch; then $h _ { 1 , a } = 1 + \beta _ { a } + \beta _ { a } ^ { 2 }$ and $\rho _ { a } = 1$ , making this row equal to $O _ { a }$ . Since both $O _ { a } , N _ { a }$ occur in the table, the reduction follows. Continuity follows from the defining functions and maxima. □

The golden robustness endpoint. ${ \mathrm { A t } } a = \alpha , c _ { a } = \alpha ^ { 3 } , h _ { 0 , a } = 2 , h _ { 1 , a } = m _ { a } = M , A _ { a } = \varphi + 3 , { \mathrm { a n d } } \rho _ { a } = A _ { a } / M$ . The cost identities specialize to

$$
d v \leq q _ { * } ( v , b ) \leq 2 \varphi v < M v , \qquad q _ { * } ( v , b ) \leq d v + \alpha ^ { 3 } b ,\tag{D.25}
$$

$$
q _ { * } ( W , B ) - q _ { * } ( v , B ) \geq \varphi ( W - v ) ,\tag{D.26}
$$

and the special terminal budgets become

$$
W , B \leq \varphi h , \qquad \sum _ { i } q _ { i } \leq d W + \alpha ^ { 3 } B \leq ( \varphi + 3 ) h .\tag{D.27}
$$

In particular, the critical outward case is

$$
F _ { J } \leq ( M + 1 ) h + W + q _ { * } ( W , B ) \leq ( M + 1 + \varphi ) h + q _ { * } ( \varphi h , \varphi h ) = ( 5 \varphi + 3 ) h .\tag{D.28}
$$

All other endpoint coefficients in the table are smaller: their positive slacks follow directly from $3 / 2 < \varphi < 2$ . This gives $K _ { \alpha } = \varphi ^ { 5 }$

Theorem D.4 (Exact shaped-cost guarantees). Both $T _ { * }$ and $T _ { * } ^ { F }$ have consistency d and exact strict robustness $\varphi ^ { 5 } = 5 \varphi + 3$

Proof. The capacity theorem and Proposition D.1 give exact consistency. The specialized joint-cut calculation gives the robustness upper bound; Proposition D.5 below supplies the matching family. □

Proposition D.3 (A matching outward family). For every $a \in [ \alpha , 1 ]$ , raw and filtered $T _ { a }$ satisfy $R \geq O _ { a \cdot } A t a = 1$ the exact pair is $( C , R ) = ( 4 , 7 )$

Proof. For $0 < \epsilon < 1 / 1 0$ offer

$$
\begin{array} { c c l } { { K } } & { { 0 } } & { { [ 1 - \epsilon , 1 + \beta _ { a } - \epsilon ) , } } \\ { { I } } & { { 0 } } & { { [ \epsilon , 1 + \beta _ { a } - 2 \epsilon ) , } } \\ { { R } } & { { 0 } } & { { [ 1 + \beta _ { a } - 2 \epsilon , 1 + \beta _ { a } - 2 \epsilon + q _ { 0 , a } \beta _ { a } - \epsilon ) , } } \\ { { J } } & { { 1 } } & { { [ 0 , 1 ) , } } \\ { { L } } & { { 0 } } & { { [ 2 \epsilon - m _ { a } , \epsilon ) . } } \end{array}
$$

The anchor K has length $\beta _ { a } ,$ zero ledger, and cost $q _ { 0 , a } \beta _ { a }$ . Both I and R reject; their lengths are $q _ { 0 , a } - 3 \epsilon < q _ { 0 , a } \beta _ { a }$ and $q _ { 0 , a } \beta _ { a } - \epsilon$ . The unit positive J fails ordinary acceptance and passes the special length test at equality with zero ledger. After it replaces K, the final negative L rejects at length $m _ { a } - \epsilon < m _ { a }$ . Thus $\mathrm { A L G } = 1$ , while the compatible $L , I , R$ tile the entire offered hull and give $\mathrm { O P T } = m _ { a } + 1 + \beta _ { a } + q _ { 0 , a } \beta _ { a } - 5 \epsilon$ ϵ. There is only one positive, so filtering changes nothing. Let ϵ decrease to zero. $\mathrm { A t } a = 1$ the constants are $q _ { 0 , a } = 2 , m _ { a } = Q _ { a } = 3 , A _ { a } = 9 / 4 , \rho _ { a } = 1 , h _ { 0 , a } = 2 , h _ { 1 , a } = 3 ;$ the table’s maximum is 7, matching this lower bound, and the exact consistency proposition gives 4. □

Proposition D.4 (Exact robustness on two parameter intervals). Let $a _ { - } \in \left( \alpha , a _ { c } \right)$ and $a _ { + } \in \left( a _ { c } , 1 \right)$ be the unique zeros in these respective intervals of

$$
\begin{array} { r l } & { p _ { - } ( x ) = 1 + 3 x + x ^ { 2 } - 5 x ^ { 3 } } \\ & { \qquad - 6 x ^ { 4 } - 4 x ^ { 5 } - x ^ { 6 } , } \\ & { p _ { + } ( x ) = 1 + 2 x + x ^ { 2 } - 3 x ^ { 3 } - 3 x ^ { 4 } - x ^ { 5 } . } \end{array}\tag{D.29}
$$

For both $T _ { a }$ and $T _ { a } ^ { F }$ , the exact robustness is

$$
R = \left\{ \begin{array} { l l } { 1 + 3 / a + 2 / a ^ { 2 } , } & { \alpha \leq a \leq a _ { - } , } \\ { a + 2 + 3 / a + 1 / a ^ { 2 } , } & { a _ { + } \leq a \leq 1 . } \end{array} \right.
$$

On $( a _ { - } , a _ { + } )$ , the bounds are $O _ { a } \ \leq \ R \ \leq \ N _ { a }$ with $O _ { a } \ < \ N _ { a }$ . In particular, $a = 2 / 3$ gives the exact pair $( C , R ) =$ $( 2 5 / 9 , 1 0 )$

Proof. On $[ \alpha , a _ { c } ]$ , substitution gives

$$
\begin{array} { c } { { \Delta _ { a } : = { \cal O } _ { a } - N _ { a } = \displaystyle \frac { 1 } { a ^ { 2 } } - 1 - a - \displaystyle \frac { 1 + 2 a } { ( 1 + a ) ^ { 3 } } , } } \\ { { a ^ { 2 } ( 1 + a ) ^ { 3 } \Delta _ { a } = p _ { - } ( a ) . } } \end{array}
$$

Here $d \Delta _ { a } / d a = - 2 / a ^ { 3 } - 1 + ( 1 + 4 a ) / ( 1 + a ) ^ { 4 } < 0 .$ , since the last fraction is below one. The signs $\Delta _ { \alpha } = 2 \sqrt { 5 } - 4 > 0$ and $\Delta _ { a _ { c } } = - ( 1 + 2 a _ { c } ) / ( 1 + a _ { c } ) ^ { 3 } < 0$ give the unique crossing $a _ { -- } \mathrm { O n } \left[ a _ { c } , 1 \right]$ , put

$$
\begin{array} { c } { { g _ { a } : = A _ { a } - m _ { a } = \displaystyle \frac { 1 } { a ^ { 2 } } + \displaystyle \frac { 1 } { ( 1 + a ) ^ { 2 } } - 1 - a , } } \\ { { a ^ { 2 } ( 1 + a ) ^ { 2 } g _ { a } = p _ { + } ( a ) . } } \end{array}
$$

Now $d g _ { a } / d a = - 2 / a ^ { 3 } - 2 / ( 1 + a ) ^ { 3 } - 1 < 0 , g _ { a _ { c } } = ( 1 + a _ { c } ) ^ { - 2 } > 0 ,$ , and $g _ { 1 } = - 3 / 4$ . Thus its unique zero is $a _ { + }$ , where $\rho _ { a }$ reaches one, and

$$
O _ { a } - N _ { a } = \left\{ \begin{array} { l l } { { ( m _ { a } - A _ { a } ) ( 1 + \beta _ { a } / m _ { a } ) , } } & { { a _ { c } \leq a \leq a _ { + } , } } \\ { { m _ { a } - A _ { a } , } } & { { a _ { + } \leq a \leq 1 . } } \end{array} \right.
$$

The signs and Proposition D.3 prove the two exact intervals and the strict middle certificate gap; substituting the two branches of $m _ { a }$ gives the displayed R. Finally, $\Delta _ { 2 / 3 } = 1 1 9 / 1 5 0 0 > 0$ and $2 / 3 < a _ { c }$ , with $O _ { 2 / 3 } = 1 0$ . Proposition D.1 gives $C = 2 5 / 9$ □

Numerically, $a _ { - } \approx 0 . 6 7 7 8 2 1 , a _ { c } \approx 0 . 7 5 4 8 7 8 , \mathrm { a n d } a _ { + } \approx 0 . 8 1 3 8 2 4 .$

## D.4. A matching family and the architectural limitation

Proposition D.5 (A floor for fixed special rules). Fix $a = \alpha .$ . Keep positive ordinary cost $M v ,$ the shared special tests, ordinary priority,fees, and ledger inheritance. Ifthe negative cost isfinite and satisfies $q ( v , 0 ) \geq d v ,$ , the resulting raw and filtered policies have strict robustness at least $\varphi ^ { 5 }$

Proof. For $0 < \epsilon < 1 /$ 100 offer the following five requests in order:

$$
\begin{array} { c c l } { K } & { 0 } & { [ 1 - \epsilon , d - \epsilon ) , } \\ { I } & { 0 } & { [ \epsilon , d - 2 \epsilon ) , } \\ { R } & { 0 } & { [ d - 2 \epsilon , d - 2 \epsilon + M - \epsilon ) , } \\ { J } & { 1 } & { [ 0 , 1 ) , } \\ { L } & { 0 } & { [ 2 \epsilon - M , \epsilon ) . } \end{array}
$$

K is accepted empty, has length $\varphi ,$ and carries zero ledger. Both I and R overlap it and have length below $q ( \varphi , 0 ) \geq M$ so they are rejected. The positive J fails ordinary acceptance, passes the special length test at equality and the zero-ledger test, and replaces K. Finally L overlaps J and has length $M - \epsilon < M .$ , so it is rejected. Thus $\mathrm { A L G } = 1$ . The compatible intervals L, I, R tile the entire offered hull; all five requests lie in that hull. Therefore

$$
\mathrm { O P T } = 2 ( M - \epsilon ) + ( d - 3 \epsilon ) = 5 \varphi + 3 - 5 \epsilon .
$$

There is only one positive, so filtering changes nothing. Taking $\epsilon \downarrow 0$ proves the floor and the matching lower bound in Theorem D.4. □

With positive cost $M v$ , these results make Shaped-BANK robustness-optimal within the sufficient-cost class of Theorem D.2. The zero-ledger relaxation and the nonshrinking-cost floor are analyzed next.

## E. Cancellation penalties and replacement-cost limits

The buyback objective subtracts a proportional payment for each revoked request from the value finally retained (Babaioff et al., 2009). For the unchanged Shaped-BANK policy, its accepted-history bound also controls this payment. The follow ing result applies with or without the permanent-history filter and for arbitrary raw predictions.

Proposition E.1 (Sharp cumulative revocation factor). Let T be the total length ofaccepted identities subsequently revoked, each counted once, and let $N , P$ be thefinal lengths held with effective labels zero and one. Then

$$
T \leq N + 2 \varphi P \leq 2 \varphi { \mathrm { A L G } } .
$$

The supremum of T /ALG over finite nonempty inputs is $2 \varphi ,$ also for the filtered policy.

Proof. The replacement forest partitions all accepted identities among the final held roots. Equation (D.8) bounds the accepted history of a negative root by twice its length and that of a positive root by M times its length. Subtracting the final roots gives $T \le N + ( M - 1 ) P = N + 2 \varphi P$

For sharpness, put $n _ { k } = d ^ { k }$ and $a _ { k } = \alpha ^ { 2 } ( d ^ { k } - 1 )$ . For $k = 0 , \ldots , K$ , present in order

$$
N _ { k } = [ 0 , n _ { k } ) , p ( N _ { k } ) = 0 , \qquad P _ { k } = [ a _ { k } , a _ { k + 1 } ) , p ( P _ { k } ) = 1 .
$$

The positives are mutually compatible, have lengths $\alpha n _ { k }$ , and lie inside their respective $N _ { k }$ , since $a _ { k + 1 } = n _ { k } - \alpha ^ { 2 }$ . Thus the permanent filter preserves their labels. Before $P _ { k }$ , the negative holder has ledger $\begin{array} { r } { \sum _ { j < k } \alpha n _ { j } = a _ { k } \le n _ { k } } \end{array}$ . The special

length test accepts $P _ { k }$ at equality, and its ledger test passes. Then $N _ { k + 1 }$ passes the positive holder’s ordinary test at equality, since $n _ { k + 1 } = M \alpha n _ { k }$ . The final reward is $\boldsymbol { \alpha } \bar { d } ^ { K }$ , and

$$
T = \sum _ { k = 0 } ^ { K } d ^ { k } + \alpha \sum _ { k = 0 } ^ { K - 1 } d ^ { k } = 2 \varphi \mathrm { A L G } - 1 .
$$

Taking $K  \infty$ proves the supremum.

Charge $c w ( I )$ for revoking $I ,$ in the same units as retained length, and define $\mathrm { A L G } _ { c } = \mathrm { A L G } - c T$ . Finite net robustness means a positive constant $R _ { c }$ satisfying $\mathrm { O P T } \leq R _ { c } \mathrm { A L G } _ { c }$ on every input. An offline algorithm can select its final optimum without cancellations, so its net optimum is still OPT. For $0 \leq c < 1 / ( 2 \varphi )$ , Proposition E.1 gives

$$
\mathrm { A L G } _ { c } \geq ( 1 - 2 \varphi c ) \mathrm { A L G } , \qquad R _ { c } \leq { \frac { \varphi ^ { 5 } } { 1 - 2 \varphi c } } .\tag{E.1}
$$

Under perfect membership predictions the corresponding sufficient consistency bound is $d / ( 1 - 2 \varphi c )$ . For arbitrary predictions with the permanent filter, Corollary K.1 instead gives

$$
\mathrm { A L G } _ { c } \geq ( 1 - 2 \varphi c ) \operatorname* { m a x } \left\{ { \frac { \mathrm { O P T } } { \varphi ^ { 5 } } } , { \frac { \mathrm { O P T } - \eta _ { K } } { d } } \right\} .
$$

These ratio bounds are sufficient.

The fee range for finite robustness is exact for this unchanged algorithm. In the sharpness family all intervals lie in $N _ { K }$ itself an offered interval, so ${ \mathrm { O P T } } = d ^ { K }$ and

$$
\mathrm { A L G } _ { c } = \alpha d ^ { K } ( 1 - 2 \varphi c ) + c .
$$

At $c = 1 / ( 2 \varphi )$ the net reward is constant while the optimum diverges; above that threshold it is eventually negative. This threshold concerns the unmodified policy; a monetary refund rate must first be converted to retained-duration units to give $c .$

## E.1. Which replacement-cost assumptions are necessary?

The zero-ledger condition in the sufficient interface of Theorem D.2 is not necessary. What matters in that proof is the cost paid along paths that actually carry positive source mass.

Theorem E.1 (Consistency after relaxing the zero-ledger cost). In the shared ledger policy, suppose positive costs are at least $M v .$ For negative costs require

$$
q ( v , b ) \geq d v \quad ( b > 0 ) , \qquad q ( v , b ) \geq \varphi v + \alpha b \quad ( 0 < b \leq d v ) .
$$

The value $\boldsymbol { q } ( v , 0 )$ may be any finite nonnegative number. For every finite execution with feasible complete positive set $P ,$ every held interval satisfies $b ( X ) \leq d w ( X )$ and w $, ( P ) \leq d \mathrm { A L G }$

Proof. Use the source coefficients and birth-order induction of Theorem D.2. A zero-ledger child has zero coefficient for every positive source, so it contributes nothing to an inherited measure, exterior mass, or whole-source cut. It can be omitted from every such sum; the omitted ordinary costs are nonnegative.

We justify the geometric estimates separately, since they need not hold for source-free accepted history. Follow any nonzero source share from its accepted entry support to its current holder. At each replacement edge on this path, the child has positive ledger immediately before revocation. A negative child therefore costs at least $d v ,$ , and a positive child at leas $M v$ . The original edge contractions apply: an ordinary parent has a negative child of length at most $1 / d$ times its own, or a positive child of length at most $1 / M$ times its own; a special positive parent has negative children of total length at most $\varphi$ times its own. Consecutive positive nodes on a source path contract by at least d. The negative-tail displacement bounds remain αv from a negative root and du from a positive root.

Equivalently, retain the root and only replacement edges whose child had positive ledger at revocation. Every nonzero source has its entry support in this retained tree. Its interval union is connected. Induction gives retained history at most

2v at a negative root and Mv at a positive root: an ordinary parent pays all retained child costs, and a special parent has only negative children of total length at most $\varphi$ times its own. Thus the negative-subtree protrusion bound also holds for all source-bearing paths used by the proof.

The weighted cut has a nonzero left share, hence positive ledger, so its cost inequalities are unchanged. At a negative birth, omit the zero-ledger children and apply the original weighted or positive cuts to the others. Their costs sum to at most the actual ordinary gate. Birth exterior and cut bounds follow, with the zero-birth-ledger case immediate. The lifetime argument and its three capacity cases use no cost restriction at zero ledger.

At an ordinary positive birth of length u, every child satisfies $b _ { i } \leq q ( v _ { i } , b _ { i } )$ , also when $b _ { i } = 0 .$ , so the inherited ledger is at most u. One must not infer that the total child length is at most $u / d \colon$ zero-ledger children can violate that inequality. The positive cut proof only needs this length bound for the selected child carrying a nonzero share, for which its individual cost gives $v _ { i } \leq u / d .$ . At a special birth the actual tests still give both child length and inherited ledger at most $\varphi u$ . All remaining path and sibling arguments carrying nonzero shares use the geometry established above. The positive cuts and the birth induction therefore close as before. Finally source conservation gives the claimed bound for $w ( P )$ □

For example, define

$$
q _ { 0 } ( v , b ) = \left\{ \begin{array} { l l } { \varphi v , } & { b = 0 , } \\ { q _ { * } ( v , b ) , } & { b > 0 . } \end{array} \right.
$$

With positive cost $M v .$ , this policy is d-consistent and has finite robustness, although $q _ { 0 } ( v , 0 ) < d v$ . It is homogeneous and nondecreasing in ledger, with an explicit jump at zero. A finite robustness certificate follows. Total accepted subtree length is at most dv at a negative root and $( M + 1 ) v = 2 d v$ at a positive root. At an ordinary parent, child history divided by child cost is at most $\varphi ;$ at a special parent all children are negative and their total length is at most $\varphi v$ . This proves both bounds simultaneously. Thus total accepted history H is at most 2d ALG.

Distribute each rejected offline-comparator interval’s length among its then-held conflicts proportionally to their lengths. Ordinary rejection and costs at most $M v$ bound a receiver’s charge by $M v .$ . For a fixed accepted identity, at most two compatible comparator intervals extend past its endpoints; those contained inside it have total length at most its own. Its total rejected-comparator charge is therefore at most $( 2 M + 1 ) v$ . Accepted comparator identities contribute at most H. Consequently $\mathrm { O P T } \leq ( 2 M + 2 ) H \leq 8 d ^ { 2 } \mathrm { A L G }$ . The universal lower bound then makes its consistency exactly d.

Theorem E.2 (A floor under nonshrinking negative costs). Keep the ordinary-first sum gate, both special tests, fees, and full ledger inheritance. Allow arbitraryfixedfinite nonnegative costfunctions $\boldsymbol { q } _ { - } ( \boldsymbol { v } , \boldsymbol { b } )$ and $\boldsymbol { q } _ { + } ( \boldsymbol { v } , \boldsymbol { b } )$ , but assume

$$
q _ { - } ( v , b ) \geq v \qquad ( v > 0 , b \geq 0 ) .\tag{E.2}
$$

If the raw or permanently filtered policy is d-consistent, then

$$
q _ { - } ( v , v ) \geq d v , \qquad q _ { + } ( v , d v ) \geq M v \qquad ( v > 0 ) ,
$$

and its strict robustness is at least $\varphi ^ { 5 }$ . No monotonicity, continuity, or homogeneity of either cost is assumed.

Proof. First fix $v > 0$ and suppose $c = q _ { - } ( v , v ) / v < d ;$ by $( \operatorname { E } . 2 ) , c \geq 1$ . All following coordinates are multiplied by this fixed v, without rescaling the cost function. Put $t = \alpha - \epsilon$ and $x = 1 - t .$ , with sufficiently small $\epsilon > 0$ . Offer negative $V = \left[ 0 , 1 \right)$ , three positive tiles partitioning $[ 1 - \epsilon - x , 1 - \epsilon )$ , then positive $T _ { 0 } = [ 1 - \epsilon , 1 - \epsilon + t )$ . All four positives fai the special length test and are shorter than the ordinary cost, which is at least one. Their rejections give ledger $x + t = 1$ exactly. Negative $W = [ \epsilon - c , \epsilon )$ now replaces $V$ at ordinary equality. Offer four positive tiles partitioning $[ 2 \epsilon - c , 0 )$ ), and then $T _ { 1 } = [ 0 , t - \epsilon )$ . Their lengths are below both αc and c, so all reject, leaving ledger $B = c + 1 + t - 3 \epsilon$

Finally offer a positive L ending at $2 \epsilon - c$ with length $\ell = \alpha B - \epsilon$ . It fails the special ledger test. If it is accepted ordinarily, its ratio is $1 + B / \ell > d$ . If it is rejected, the ratio is $( \varphi B - \epsilon ) / c ,$ , and

$$
\varphi B - \epsilon - d c = d - c - ( 4 \varphi + 1 ) \epsilon > 0 .
$$

The positives are consecutive and tile the full offered hull; both negatives lie inside it. Hence their labels are perfect in either branch. This contradiction proves $q _ { - } ( v , v ) \geq d v$ and uses no positive cost.

Next fix $u > 0$ and suppose $q _ { + } ( u , d u ) < M u$ . Choose $0 < c < M$ with $c u \ge q _ { + } ( u , d u )$ , and multiply all following coordinates by u. Offer negative $K = [ 1 - \epsilon , d - \epsilon )$ and three positive tiles partitioning $[ 1 , d )$ . They reject, giving length and ledger both $\varphi .$ The first part makes its ordinary cost at least M. Positive $Q = [ 0 , 1 )$ is special at equality in both tests and has ledger d. Negative $W = [ \epsilon - c , \epsilon )$ replaces it ordinarily. Four positive tiles partitioning $[ 2 \epsilon - c , 0 )$ reject, leaving ledger $B = d + c - 2 \epsilon$ . A final positive L ending at $2 \epsilon - c ,$ of length $\ell = \alpha B - \epsilon$ , again gives ratio $1 + B / \ell > d$ if accepted, whereas rejection gives

$$
\varphi B - \epsilon - d c = M - c - ( 2 \varphi + 1 ) \epsilon > 0 .
$$

The positives again partition the complete hull containing both negatives. Thus $q _ { + } ( u , d u ) \geq M u$

For the robustness witness, use unit scale and negative $K = [ 1 - \epsilon , d - \epsilon )$ followed by three positive tiles partitioning $[ 1 , d )$ These rejections give ledger exactly $\varphi ,$ so $q _ { - } ( \varphi , \varphi ) \geq M$ . Offer

$$
\begin{array} { r l } { I } & { { } 0 \quad [ \epsilon , d - 2 \epsilon ) , } \\ { R } & { { } 0 \quad [ d - 2 \epsilon , d - 2 \epsilon + M - \epsilon ) , } \\ { J } & { { } 1 \quad [ 0 , 1 ) , } \\ { L } & { { } 0 \quad [ 2 \epsilon - M , \epsilon ) . } \end{array}
$$

The first two requests reject. Request J is special at equality in both tests, with ledger d, and its positive cost is at least $M$ Thus L also rejects and the final gain is one. The compatible $L , I , R$ tile the entire hull and give $\mathrm { O P T } = \varphi ^ { 5 } - 5 \epsilon$ . The four positive predictions are mutually compatible, so the filter changes no label in this witness or in either perfect-label construction above. Letting $\epsilon \downarrow 0$ proves the bound. □

Unlike Proposition D.5, which fixes positive cost Mv and assumes $q ( v , 0 ) \geq d v$ but not $C = d ,$ this floor permits arbitrary fixed positive costs and assumes the global condition (E.2) together with $C = d .$ In particular, changing the fixed positive coefficient to $\kappa < M$ , including $\kappa = d ,$ cannot preserve d-consistency under these hypotheses. The unrestricted claim for arbitrary negative costs remains open.

Discounting positive-ledger states can fail. The lower-cost rule

$$
q _ { \mathrm { l o w } } ( v , b ) = \operatorname* { m i n } \{ 2 \varphi v , \operatorname* { m a x } \{ \operatorname* { m i n } \{ \varphi ( v + b ) , d v \} , \varphi v + \alpha b \} \}
$$

is not d-consistent, even with the original positive cost $M v$ . Here is a 14-request family with $0 < \epsilon \leq 1 0 ^ { - 3 }$ . Put $v = \varphi ^ { 5 }$ $u = \varphi ^ { 4 } , w = \varphi ^ { 7 }$ and offer

$$
\begin{array} { r l } { V _ { 0 } \quad } & { 0 \quad [ 0 , 1 ) , } \\ { S _ { 0 } , S _ { 1 } \quad 1 \quad [ 0 , 1 / 2 ) , [ 1 / 2 , 1 ) , } \\ { V _ { 1 } \quad } & { 0 \quad [ 1 - \epsilon , \varphi + 2 - \epsilon ) , } \\ { V _ { 2 } \quad } & { 0 \quad [ \varphi + 2 - 2 \epsilon , 4 \varphi + 3 - 2 \epsilon ) , } \\ { V _ { 3 } \quad } & { 0 \quad [ 4 \varphi + 3 - 3 \epsilon , 9 \varphi + 6 - 3 \epsilon ) , } \\ { Q \quad } & { 1 \quad [ 9 \varphi + 6 - 4 \epsilon , 1 2 \varphi + 8 - 4 \epsilon ) , } \\ { W \quad } & { 0 \quad [ 9 \varphi + 6 - 5 \epsilon , 2 2 \varphi + 1 4 - 5 \epsilon ) . } \end{array}
$$

Next offer four positive tiles partitioning $[ 1 2 \varphi + 8 - 4 \epsilon , 2 2 \varphi + 1 4 - 6 \epsilon )$ , followed by positive $L = [ d - 3 \epsilon , 9 \varphi + 6 - 4 \epsilon )$ and positive R starting at $2 2 \varphi + 1 4 - 6 \epsilon$ with length $\alpha ( 1 + w + \varphi ^ { 6 } - 3 \epsilon ) - \epsilon$

The first two positives reject. At ledger one, the successive negative costs are d, $3 \varphi + 1 , 5 \varphi + 3 .$ , so $V _ { 1 } , V _ { 2 } , V _ { 3 }$ are ordinary replacements at equality. Request $Q$ is special at length equality and W is ordinary at equality $w = M u$ . All remaining requests reject: the four tiles and L fail the length test, and R fails the ledger test. They also fail ordinary acceptance. Thus

$$
\mathrm { A L G } = w , \qquad w ( P ) = d w + \varphi - ( 3 \varphi + 1 ) \epsilon > d w .
$$

To check optimality, the first two positives pay for $V _ { 0 }$ at equality. Request $L$ is longer than every compatible subset of $\left\{ V _ { 1 } , V _ { 2 } , V _ { 3 } \right\}$ , whose largest weight is $w ( V _ { 1 } ) + w ( V _ { 3 } ) = 6 \varphi + 4 < \varphi ^ { 6 } - \epsilon$ . The positive $Q .$ the four tiles, and R all conflic with W and have total weight $w - 2 \epsilon + w ( R ) > w$ . These three assigned positive groups are disjoint and compatible, and each belongs to the positive conflict set of its assigned negative group. Replacing any feasible negative subset by its conflicting positives therefore never loses weight. Hence $P$ is an optimum; it need not be unique because $V _ { 0 }$ ties $S _ { 0 } , S _ { 1 }$ The ratio tends to $d + \varphi ^ { - 6 } \approx 2 . 6 7 4 > d .$ . Unlike the zero-ledger relaxation, this policy transports nonzero source mass through several discounted negative replacements.

## F. A barrier for singleton-negative replacement

The five-request floor in Proposition D.5 fixes the positive cost and special predicates. The following result permits arbitrary ledgers, positive costs, and history-dependent rules. It requires only one restriction on negative arrivals, evaluated before their decisions.

Theorem F.1 (Singleton-negative barrier). Consider a deterministic policy A in the model of Section 2 that satisfies the following condition at every reachable prearrival state:

(N) If the sole held request is a negative K of length v, an incoming negative that conflicts with K and has length less than dv is rejected.

$I f C ( A ) \leq d$ and $R ( A ) < \infty ,$ then $R ( A ) \geq \varphi ^ { 5 }$ . The result allows voluntary revocation and arbitrary dependence on the entire rejected history. It also holds after the permanent feasibility filter.

Proof. Fix a finite robustness bound $r > \operatorname* { m a x } \{ 1 , R ( A ) \}$ and $0 < \epsilon < 1 / 1 0 0 0$ . Put

$$
v = \varphi ( 1 - 4 \epsilon ) , \qquad p = 1 + v - 2 \epsilon , \qquad z = d v - \epsilon .
$$

Offer the negative requests

$$
K = [ 1 - \epsilon , 1 - \epsilon + v ) , \qquad I = [ \epsilon , p ) , \qquad R = [ p , p + z )
$$

in this order. Finite robustness forces K to be retained. Both subsequent requests conflict with it and lie strictly below the threshold $d v ,$ , with margins $\varphi v - 1 + 3 \epsilon > 0$ and ϵ, respectively. Condition (N) rejects both; discarding K would leave zero reward.

Offer a finite positive partition of $[ 1 , p )$ , each tile shorter than $v / r$ . At the first departure from K, only the incoming tile could remain, whereas historical K gives $\mathrm { O P T } \geq v .$ . Robustness therefore forces every tile to be missed and K retained. Now offer positive $J = [ 0 , 1 )$ . Let A consist of J and these tiles; it tiles $[ 0 , p )$ and has weight p. This positive family is feasible but is not an optimum of the current history, since R is compatible with all of it.

A perfect completion forces retention of J. Suppose J is rejected. Then K remains, since deleting it gives zero reward. Offer positive $T = [ p , p + v )$ and put $E = p + v .$ . After any response, the local held weight is at most $v ,$ whereas

$$
w ( A \cup \{ T \} ) = E , \qquad E - d v = 2 \epsilon .
$$

Set $\eta = \epsilon / 2$ , choose finite $h > \operatorname* { m a x } \{ d v , ( r - 1 ) v , 1 \}$ , and put

$$
\begin{array} { r } { u = \alpha h , \quad b = E + u - \eta , \quad H = [ b , b + h ) , \quad U = [ E , b + \eta ) . } \end{array}
$$

Offer negative H. It is disjoint from all earlier requests; in particular,

$$
b - ( p + z ) = \alpha h - \varphi v + \epsilon - \eta > 0 .
$$

The historical packing $\{ K , H \}$ has weight $v + h > r v$ , so H must be retained. Offer a finite positive partition $\mathcal { F }$ of $\left[ b + \eta , b + h - \eta \right)$ , with each tile shorter than $( v + h ) / r - v > 0$ . At the first departure from H, the held weight is at most v plus the incoming tile. The same historical packing forces every tile to be missed and H retained, even if the local holder is voluntarily revoked.

Offer positive U. If retained, stop, with held weight at most $v + \alpha h$ . If U is not retained, robustness forces H to remain. In that case offer positive $V = [ b + h - \eta , b + 2 h - \eta )$ and stop; every response has held weight at most $v + h$ . All positives are compatible. Their totals in the two branches are $E + \varphi h - 2 \eta$ and $E + d h - 2 \eta .$ , so

$$
\begin{array} { r } { E + \varphi h - 2 \eta - d ( v + \alpha h ) = \epsilon , } \\ { E + d h - 2 \eta - d ( v + h ) = \epsilon . } \end{array}
$$

To apply consistency, we check the entire history. At the first terminal the negatives are $K , I , R , H$ . Every feasible negative family is a subset of $\{ K , H \}$ or $\{ I , R , H \}$ . The following positive groups conflict with the indicated negative families and have the stated strict surpluses:

Optimal Consistency for Interval Selection
<table><tr><td>Negative family</td><td>Positive group Surplus</td><td></td></tr><tr><td>{K}</td><td> $A \cup \{ T \}$ </td><td> $E - v$ </td></tr><tr><td>{I}</td><td>A</td><td>€</td></tr><tr><td>{R}</td><td> $\{ T , U \}$ </td><td> $\alpha h - \varphi v + \epsilon$ </td></tr><tr><td>{H}</td><td> $\{ U \} \cup { \mathcal { F } }$ </td><td> $\alpha h - 2 \eta$ </td></tr><tr><td> $\{ R , H \}$ </td><td> $\{ T , U \} \cup { \mathcal { F } }$ </td><td> $\alpha h - \varphi v + \epsilon - 2 \eta$ </td></tr></table>

All margins are positive because αh $> \varphi v$ and $2 \eta = \epsilon .$ . The groups for K and H are disjoint. The group A for I is disjoint from each group for R, H, or $\{ R , H \}$ . Adding the corresponding margins therefore covers all nine nonempty feasible negative subsets, counting the shared U only once. Adding V preserves these certificates. The strict exchange argument in (C.3) proves that the complete positives are the unique optimum at both terminals. Either terminal contradicts $C ( A ) \leq d .$ Thus J must be accepted and retained on the common prefix; accepting and then discarding it would leave zero reward.

A second perfect completion forces rejection of $L .$ Define

$$
\ell = \varphi p - 2 \epsilon , \qquad s = \epsilon - \ell , \qquad L = [ s , \epsilon ) ,
$$

and offer negative L. It overlaps the sole holder J by ϵ. Suppose it is accepted; finite robustness forces its retention. Put $\chi = 2 \alpha - 1 = \alpha ^ { 3 } > 0$ and $\eta = \chi \epsilon / 4$ . Offer a positive partition of $[ s + \eta , 0 )$ with each tile shorter than $\ell / r$ . Historical L forces every tile to be missed by the same first-departure argument. Next offer positive $B = [ s + \eta - \ell , s + \eta )$ , of length ℓ. It overlaps L by $\eta$ and is compatible with every earlier positive. After either response the local held weight is at most ℓ. Let $\mathcal { Z }$ consist of $B$ and this left partition. The full local positive family $\mathcal { P } = \mathcal { Z } \cup \mathcal { A }$ tiles $\left[ s + \eta - \ell , p \right)$ , with weight

$$
P _ { 0 } = p + 2 \ell - \epsilon - \eta , \qquad P _ { 0 } - d \ell = \chi \epsilon - \eta = 3 \chi \epsilon / 4 .\tag{F.1}
$$

The negatives are $\mathcal { N } = \{ K , I , L \}$ and the unavailable $R = [ p , p + z )$ . Among $K , I , L ,$ only K and I conflict. All five nonempty feasible negative subsets have strict exchange certificates:

<table><tr><td>Negative family Positive group Surplus</td><td></td><td></td></tr><tr><td> $\{ K \}$ </td><td>A</td><td> $p - v = 1 - 2 \epsilon$ </td></tr><tr><td>{I}</td><td>A</td><td>€</td></tr><tr><td>{L}</td><td> ${ \mathcal { Z } } \cup \{ J \}$ </td><td> $\ell + 1 - \epsilon - \eta$ </td></tr><tr><td> $\{ K , L \}$ </td><td> $\mathcal { Z } \cup \mathcal { A }$ </td><td> $p + \ell - v - \epsilon - \eta$ </td></tr><tr><td> $\{ I , L \}$ </td><td> $\mathcal { Z } \cup \mathcal { A }$ </td><td> $\ell - \eta$ </td></tr></table>

Every margin is positive for the stated ϵ range. Apply Lemma C.1 with $c = d , \gamma = 0 , a = \ell ,$ , and $\zeta = \chi \epsilon / 8$ . Historical L has weight a. Choose finite $h > \operatorname* { m a x } \{ ( r - 1 ) \ell , 1 \}$ with αh $> z + 2 \zeta$ . Then $h > 2 \zeta$ and $p + \alpha h - \zeta > p + z$ , the largest old right endpoint, while

$$
P _ { 0 } - d \ell - 2 \zeta = \chi \epsilon / 2 > 0 .
$$

Thus every hypothesis of the lemma holds. Its satellite construction repairs R as well as the new negative H; for their joint subset the new conflict group has surplus $\alpha h - z - 2 \zeta > 0$ and counts its shared boundary positive once. This group is disjoint from every local group above, so mixed negative subsets also satisfy strict exchange. The resulting robustness or actual-perfect consistency violation rules out acceptance of L.

The finite robustness witness. Stop on the common prefix immediately after $L ,$ which must be rejected. J remains the sole holder, so $\mathrm { A L G } = 1$ . The compatible negatives $L , I .$ , R tile the entire offered hull, containing every request on this short input. Length reward therefore gives the exact optimum

$$
\begin{array} { r l } & { \mathrm { O P T } = \ell + ( p - \epsilon ) + ( d v - \epsilon ) } \\ & { \qquad = d + 2 d v - ( 2 \varphi + 6 ) \epsilon } \\ & { \qquad = \varphi ^ { 5 } - ( 1 8 \varphi + 1 4 ) \epsilon . } \end{array}
$$

Table 3. Deterministic guarantees. $d = \varphi ^ { 2 } , c _ { * } = \alpha ^ { 3 } , B _ { * } = \varphi ^ { 5 } , B _ { C * } = 4 \varphi + 5 , B _ { S } = 7 \varphi + 4$ . In paired rows the error guarantee applies only to the filtered policy (superscript F); η uses original labels. LR is due to Garay et al. (1997); its interval formulation, Naive and the RP consistency bound are given by Karavasilis (2025).
<table><tr><td>Algorithm</td><td>Consistency</td><td>Robustness</td><td>Error guarantee</td></tr><tr><td>LR</td><td>M</td><td>M (exact)</td><td></td></tr><tr><td> $\mathsf { R P } _ { \lambda } , \lambda \ge 2$ </td><td> $\leq \frac { 3 \lambda } { \lambda - 1 }$ </td><td> $2 \lambda + 2 ( \mathrm { e x a c t } )$ </td><td></td></tr><tr><td>Naive</td><td>1</td><td>∞</td><td> $\mathrm { O P T } \le \mathrm { A L G } + \eta _ { K }$ </td></tr><tr><td>Shaped  $T _ { * } , T _ { * } ^ { F }$ </td><td>d</td><td> $B _ { * } \ ( \mathrm { e x a c t } )$ </td><td> $\mathrm { O P T } \leq d \mathrm { A L G } + \eta _ { K }$ </td></tr><tr><td> $\mathrm { C r e d i t } \ : T _ { c * } , T _ { c * } ^ { F }$ </td><td>d</td><td> $B _ { C } { \mathrm { : } }$  (exact)</td><td> $\mathrm { O P T } \leq d \mathrm { A L G } + \eta _ { K }$ </td></tr><tr><td> $\mathrm { S U M } \ : S _ { \varphi ^ { 3 } } , S _ { \varphi ^ { 3 } } ^ { F }$ </td><td>d</td><td> $B _ { S } \ ( \mathrm { e x a c t } )$ </td><td> $\mathrm { O P T } \leq d \mathrm { A L G } + \eta _ { K }$ </td></tr></table>

Every partition above is finite: a positive mass can be split into finitely many tiles below any positive threshold. The two longer completions are causal witnesses that force decisions on their shared prefixes; neither changes an earlier label or assumes prefix optimality. Taking the supremum over positive ϵ proves $R ( A ) \geq \varphi ^ { 5 }$ . All constructed positive families are compatible, so permanent filtering changes no bit and the same witnesses apply. □

In particular, a source-cost policy whose sole negative holder of length v costs at least $d v ,$ and whose negatives have no acceptance route outside that cost gate, satisfies (N). Its positive costs, source fees, inheritance, and history-dependent special rules do not enter the proof. Any deterministic policy with $C \leq d$ and finite $R < \varphi ^ { 5 }$ must instead accept some conflicting negative shorter than dv at a reachable state whose sole holder is a negative of length $v .$

The zero-ledger policy q<sub>0</sub> in Appendix E.1 gives a concrete escape from this early negative prefix. With the same $v =$ $\varphi ( 1 - 4 \epsilon )$ and $p = 1 + v - 2 \epsilon$ , offer K followed immediately by $I = [ \epsilon , p )$ , before any positive source arrives. The ledger is zero and

$$
w ( I ) - q _ { 0 } ( v , 0 ) = 1 + v - 3 \epsilon - \varphi v = \epsilon > 0 .
$$

Thus $q _ { 0 }$ replaces K by I, breaking the intended singleton-anchor branch. This policy is nevertheless d-consistent and finitely robust. It shows why an early-negative branch cannot simply be presumed to reject. However, $q _ { 0 } ( v , b ) \geq v$ at every negative state, so Theorem E.2 still gives $R ( q _ { 0 } ) \geq \varphi ^ { 5 }$

## G. Additional ledger policies and guarantees

Credit varies the negative ledger coefficient; SUM uses total conflicting length. MAX uses maximum conflicting length and is retained as a descriptive experimental comparator. Credit-BANK $T _ { c }$ has negative cost min $\{ M v , d v + c b \} , c \geq 0$ The constant-cost gates are

$$
\begin{array} { r } { G _ { \lambda } ( D ) = \left\{ \begin{array} { l l } { \lambda \sum _ { J \in D } w ( J ) , } & { \mathrm { S U M } , } \\ { \lambda \operatorname* { m a x } _ { J \in D } w ( J ) , } & { \mathrm { M A X } . } \end{array} \right. } \end{array}\tag{G.1}
$$

Definition G.1 (Ledger policies). Shaped-BANK $T _ { * }$ uses (7) and positive cost $M v ;$ Credit-BANK $T _ { c }$ uses (H.1). SUM-BANK $S _ { \lambda }$ , for $\lambda > d ,$ , and MAX-BANK $A _ { \lambda } ,$ for $\lambda \geq d ,$ use Algorithm 1 with the respective gate in (G.1). All other rules are shared: ordinary acceptance has priority, all thresholds accept equality, and special acceptance requires a positive prediction, only negative conflicts, and both length and ledger tests.

The guarantees below concern Shaped, Credit, SUM and RP.

Table 3 collects the endpoint comparisons. The full scalar envelope, its exact consistency threshold $c _ { * } = \alpha ^ { 3 }$ , and its global minimum $4 \varphi + 5$ are in Theorem H.1. The fixed comparison $T = T _ { \varphi / 2 }$ has exact robustness $6 \varphi + 3$ . For SUM, Corollary I.1 proves the policy-specific threshold $\lambda \geq \varphi ^ { 3 }$ , and Proposition J.1 gives exact robustness $\varphi ( 2 \lambda + 1 )$ on that range, with the upper bound $K ( \varphi , \lambda )$ from (J.2) for $\lambda > d .$ . For $c \in [ c _ { * } , 1 ]$ and $\lambda \geq \varphi ^ { 3 }$ , respectively, both families satisfy $w ( P ) \leq d \mathrm { A L G }$ for feasible complete positives.

For RP, Corollary J.1 proves exact robustness $2 \lambda + 2$ when $\lambda \geq 2 .$ , and $\lambda + 2 \lambda / ( \lambda - 1 )$ when $1 < \lambda < 2$ . Its published consistency bound $3 \lambda / ( \lambda - 1 )$ need not be exact.

Theorem F.1 treats the shared singleton-negative obstruction. Tables 4 and 6 compare the policies empirically; Appendices M–O specify the matched repeats and differing robustness budgets.

## H. Credit-dependent replacement costs

Put $\alpha = 1 / \varphi , d = \varphi ^ { 2 }$ , and $M = \varphi ^ { 3 }$ . For any fixed real $c \geq 0 ,$ , Credit-BANK $T _ { c }$ uses the shared ledger updates and special tests of Algorithm 1, with ordinary gate $\sum { } _ { J \in D } \tau _ { c } ( J )$ , where

$$
\tau _ { c } ( J ) = \left\{ { \cal M } w ( J ) , \atop { \operatorname* { m i n } \{ { \cal M } w ( J ) , d w ( J ) + c b ( J ) \} , }  \right. ~ p ( J ) = 1 ,\tag{H.1}
$$

Write $q _ { c } ( v , b ) = \operatorname* { m i n } \{ M v , d v + c b \}$ for the negative cost. The coefficient changes only this ordinary cost and is fixed for an entire execution. Ordinary acceptance has priority, thresholds accept equality, and rejection fees use simultaneous old-state snapshots with length-failure priority. Special acceptance still requires a positive arrival, only negative conflicts, $W , B \leq \varphi w ( I )$ , and full ledger inheritance. The permanent-filter composition is denoted $T _ { c } ^ { F }$

Theorem H.1 (Exact scalar envelope and consistency threshold). For $0 \leq c \leq 1$ , both raw andfiltered policies have exact robustness

$$
\begin{array} { c } { R ( T _ { c } ) = R ( T _ { c } ^ { F } ) = K ( c ) : = \operatorname* { m a x } \{ K _ { \mathrm { o u t } } ( c ) , K _ { \mathrm { b r } } ( c ) \} , } \\ { K _ { \mathrm { o u t } } ( c ) = 5 \varphi + 3 + \varphi c , } \\ { K _ { \mathrm { b r } } ( c ) = 5 \varphi + 2 + 2 \varphi c . } \end{array}\tag{H.2}
$$

Within this range, their consistency equals d $i f$ and only $i f c \ge c _ { * } : = \alpha ^ { 3 } = 2 \varphi - 3 .$ . For $c \in [ c _ { * } , 1 ]$ , every execution with feasible complete positive set $P$ satisfies

$$
b ( J ) \leq d w ( J ) t h r o u g h o u t e \nu e r y h e l d l i f e t i m e , \qquad w ( P ) \leq d \mathrm { A L G } .\tag{H.3}
$$

Among all $c \geq 0$ whose raw policy has consistency d, the minimum robustness is $4 \varphi + 5 ,$ attained only at $c = c _ { * }$ . The same statement holdsfor thefiltered policies.

The branches in (H.2) agree at $c = \alpha ;$ the outward branch dominates below it and the bridge branch above it. In particular, the comparison policy $T = T _ { \varphi / 2 }$ retains $C ( T ) = d$ and $R ( T ) = 6 \varphi + 3$

## H.1. Consistency from the common source-measure interface

Theorem D.2 applies precisely when the scalar cost satisfies its two negative-cost inequalities. Globally $q _ { c } ( v , b ) \geq d v$ . For $0 \leq b \leq d v$ , the cap branch obeys

$$
\varphi v + \alpha b \leq ( \varphi + \alpha d ) v = 2 \varphi v < M v .
$$

The affine branch dominates the same expression exactly when

$$
v + ( c - \alpha ) b \geq 0 \qquad ( 0 \leq b \leq d v ) .\tag{H.4}
$$

For $c \geq \alpha$ this is immediate. For $0 \leq c < \alpha$ , its minimum over this interval is $[ 1 + d ( c - \alpha ) ] v ,$ , so the necessary and sufficient condition for the interface is

$$
c \geq \alpha - 1 / d = \alpha - \alpha ^ { 2 } = \alpha ^ { 3 } = c _ { * } .
$$

At $c = c _ { * } .$ , equality occurs at $b = d v$ , where the cost is $2 \varphi v < M v$ . Thus equality is included without invoking the cap. The source-measure induction proves (H.3), in fact for every $c \geq c _ { * }$ . On $[ c _ { * } , 1 ]$ the finite robustness proved below and Theorem 2 give exact consistency d. The following finite construction proves necessity.

## H.2. Eleven requests proving the consistency threshold

Fix $0 \leq c < c _ { * }$ and put

$$
k = 3 \varphi + 1 , \qquad \Delta = 2 \varphi - d ( 1 + c ) > 0 , \qquad D = ( \varphi - c ) k + 1 > 0 .
$$

Choose

$$
0 < \epsilon < \operatorname* { m i n } \{ 1 / 1 0 0 , \Delta / ( 2 D ) \} .\tag{H.5}
$$

First offer the negative unit interval $X = [ 0 , 1 )$ and then the positive seed intervals $[ \epsilon , 1 / 2 )$ and $[ 1 / 2 , 1 - \epsilon )$ . Their lengths $1 / 2 - \epsilon < \alpha$ fail the special length test and are below the ordinary cost, which is at least d. Both reject and deposit their complete lengths into the sole held conflict X, giving ledger $1 - 2 \epsilon$ . The positive interval

$$
A = [ 2 \epsilon - \alpha , \epsilon )
$$

has length $\alpha - \epsilon$ and rejects for the same reasons. The ledger is now $b _ { 1 } = \varphi - 3 \epsilon$

Next offer positive B starting at $1 - \epsilon$ with length

$$
\ell _ { B } = \alpha b _ { 1 } - \epsilon = 1 - ( 3 \alpha + 1 ) \epsilon .
$$

Since $\epsilon < 1 / 1 0 0$ and $\alpha < 3 1 / 5 0$ , we have $\ell _ { B } > 0 . 9 6 > \alpha$ . Thus B passes the special length test but fails the ledger test: $\varphi \ell _ { B } = b _ { 1 } - \varphi \epsilon < b _ { 1 }$ . It also fails the ordinary test because $\ell _ { B } < 1 < d .$ Its complete length enters $X \mathbf { \hat { s } }$ ledger, which becomes

$$
b _ { X } = d - k \epsilon .
$$

Let $g = d + c b _ { X }$ and offer negative $J = [ \epsilon - g , \epsilon )$ . The cap on X is inactive, since

$$
g \leq d ( 1 + c ) < d ( 1 + c _ { * } ) = 2 \varphi < M .
$$

Therefore J has exactly the old ordinary cost of X as its length. It overlaps X in $[ 0 , \epsilon )$ , accepts at equality, and inherits all of $b _ { X }$ . We have $d \leq g < 2 \varphi$

Partition $[ 2 \epsilon - g , 2 \epsilon - \alpha )$ into four equal positive tiles. Each has positive length $( g - \alpha ) / 4 < \alpha g$ because 4α $> 1$ , and its length is below $d g ,$ a lower bound on $J { \boldsymbol { \mathbf { \mathit { s } } } }$ ordinary cost. Each tile overlaps $^ { J , }$ fails both the ordinary test and the specia length test, and deposits its full length. After the four tiles, the ledger of J is

$$
B _ { \mathrm { p r e } } = b _ { X } + g - \alpha = g + 2 - k \epsilon .
$$

Finally offer positive C ending at $2 \epsilon - g$ with length

$$
z = \alpha B _ { \mathrm { p r e } } - \epsilon .
$$

It passes the special length test because $\varphi z - g = 2 - ( k + \varphi ) \epsilon > 0 ,$ , using $k + \varphi = 4 \varphi + 1 < 8$ . It fails the ledger test strictly, as $\varphi z = B _ { \mathrm { p r e } } - \varphi \epsilon < B _ { \mathrm { p r e } }$ . It also fails ordinary acceptance, since

$$
z < \alpha ( g + 2 ) < d g ;
$$

the last inequality follows from $d - \alpha = 2$ and $g > \alpha$ . Its overlap with J is positive: $z > \alpha g > \epsilon .$ . Thus every fee in the construction is the full arriving positive length, and the final held set is $\{ J \}$ with reward g.

In spatial order the positive requests are $C ,$ the four new tiles, A, the two seed tiles, and B. They are consecutive and tile one interval. Its left end lies left of J because $z > \epsilon ,$ , and its right end lies right of X because $\ell _ { B } > \epsilon ;$ hence it contains both historical negatives. The positive tiling attains the length of the entire offered hull and is optimal. Moreover it is the unique optimum. Covering the two extreme segments requires C and B, respectively. They exclude J and X, leaving only the positive tiles, all of which are needed to cover their positive-length segments. The labels are therefore perfect and the filter changes none of them.

The exact optimum is $\mathrm { O P T } = B _ { \mathrm { p r e } } + z = \varphi ( g + 2 - k \epsilon ) - \epsilon ,$ so

$$
\begin{array} { r l } & { \mathrm { O P T } - d \mathrm { A L G } = 2 \varphi - g - ( \varphi k + 1 ) \epsilon } \\ & { \qquad = \Delta - [ ( \varphi - c ) k + 1 ] \epsilon = \Delta - D \epsilon > \Delta / 2 > 0 . } \end{array}\tag{H.6}
$$

This is a strict finite violation for every $c < c _ { * }$ , for both raw and filtered policies. $\mathrm { A t } ~ c = c _ { * }$ , all gates and the uniqueness argument still hold for $0 < \epsilon < 1 / 1 0 0 ;$ ; only the strict violation disappears. Its limit is

$$
\operatorname* { l i m } _ { \epsilon \downarrow 0 } \frac { \mathrm { O P T } } { \mathrm { A L G } } = \varphi + \frac { 2 \alpha } { 1 + c _ { * } } = d .
$$

For positive ϵ at this endpoint, $\mathrm { O P T } - d g = - D \epsilon < 0 .$ , consistent with the upper bound.

## H.3. Terminal costs and the scalar robustness envelope

We use the replacement-forest and connected-hull framework of Lemma D.5 and the proof of Theorem D.3. The scalar specific estimates below verify its hypotheses without any feasible-label assumption. Throughout this subsection $0 \leq c \leq 1$ For each accepted identity $X$ , let $q _ { X }$ be its ordinary cost immediately before revocation, or at termination if retained. Ledgers grow by nonnegative fees during a held lifetime, so costs are nondecreasing and d $v ( X ) \leq q _ { X } \leq M w ( X )$ . An ordinary parent of length h satisfies $\sum _ { i } q _ { i } \leq h . \mathrm { ~ A ~ }$ special parent is positive and has negative, ordinarily accepted children. Its special tests imply

$$
W , B \leq \varphi h , \qquad \sum _ { i } q _ { i } \leq d W + c B \leq A h , \qquad A = M + \varphi c , \quad \rho = A / M = 1 + c / d .\tag{H.7}
$$

In this range $1 \leq \rho \leq 1 + 1 / d < d . \quad$

The terminal-cost induction from the shaped proof applies with these $A , \rho \colon$ an ordinary root’s omitted-root bound is $\rho \sum _ { i } q _ { i } \le \rho h < q _ { J } ;$ a special root’s ordinary children sum to at most $A h ,$ as does its singleton Mh. Thus the full antichain bound is $q _ { J }$ or $A w ( J )$ respectively. The history bound (D.8) also applies, since it uses only the same positive cost $M h .$ , negative lower bound $d h$ , and special length gate. An ordinary birth gives history at most $\begin{array} { r } { h + \sum _ { i } q _ { i } \le 2 h ; } \end{array}$ a special birth gives $h + 2 W \leq M h$ . The connected hulls obey these same bounds.

Use Lemma D.5 with $\omega _ { X } = q _ { X } \colon$

$$
F _ { J } ( x , y ) = | H _ { J } \cap [ x , y ] | + L _ { J } ( x ) + U _ { J } ( y ) .\tag{H.8}
$$

At an ordinary root, an eligible root singleton attains the side maximum $q { \mathrm { , } } r { \mathrm { ; } }$ an ineligible side has weight at most $\rho h$ . With two eligible endpoints the gap lies inside the root and has length at most $h ;$ otherwise use its full history bound 2h. Thus the coefficients for two, one or no eligible endpoints are

$$
1 + 2 M , \qquad 2 + M + \rho , \qquad 2 + 2 \rho .\tag{H.9}
$$

At a special root $\boldsymbol { J } = [ a , b )$ , each side chooses the eligible root singleton of cost Mh or the sum of the children’s side maxima. A zero maximum may choose children. Two singleton choices give $( 1 + 2 M ) h$ . With exactly one singleton, reflect so it is the left. If $b > y$ , the gap is inside J, giving $( 1 + M + A ) h$ . If $b \leq y$ , every child starts below b, so at most one compatible child k ends above $y .$ If such a child exists, the gap has length at most $h + h _ { k }$ and

$$
\begin{array} { r l } & { F _ { J } \leq ( M + 1 ) h + ( 1 - \rho ) h _ { k } + q _ { k } + \rho W } \\ & { \qquad \leq ( M + 1 ) h + ( 1 + d - \rho ) h _ { k } + c b _ { k } + \rho W } \\ & { \qquad \leq ( M + 1 ) h + ( 1 + d ) W + c B } \\ & { \qquad \leq [ M + 1 + \varphi ( 1 + d + c ) ] h = K _ { \mathrm { o u t } } ( c ) h . } \end{array}\tag{H.10}
$$

Here $q _ { k } \le d h _ { k } + c b _ { k } , 1 + d - \rho > 0 , h _ { k } \le W$ , and $b _ { k } \leq B$ . Without an outward child, all right-side maxima are ineligible, and the full-hull bound gives $( 2 M + \rho \varphi ) h$

If both endpoint choices use children, at most one child bridges both cuts, meaning $s ( C _ { k } ) < x \leq y < f ( C _ { k } )$ . This child contains the gap and contributes at most $h _ { k } + 2 q _ { k }$ . Every other child has at most one eligible endpoint and contributes at most $q _ { i } + \rho h _ { i }$ . This also bounds a child with two ineligible endpoints because $q _ { i } \geq d h _ { i } > \rho h _ { i }$ . Consequently

$$
\begin{array} { l } { { F _ { J } \le h _ { k } + 2 q _ { k } + \displaystyle \sum _ { i \ne k } ( q _ { i } + \rho h _ { i } ) } } \\ { { \le W + 2 \displaystyle \sum _ { i } q _ { i } \le ( 1 + 2 d ) W + 2 c B \le K _ { \mathrm { b r } } ( c ) h . } } \end{array}\tag{H.11}
$$

The second line uses $( \rho - 1 ) h _ { i } \leq q _ { i }$ <sub>i</sub>. If no child bridges, each contributes at most $q _ { i } + \rho h _ { i } ;$ together with the full-hull bound, this gives $( M + A + \rho \varphi ) h$

All remaining coefficients lie below the two-term envelope. For the special-root cases,

$$
\begin{array} { c } { { K _ { \mathrm { o u t } } - ( 1 + 2 M ) = \varphi + \varphi c , } } \\ { { \ } } \\ { { K _ { \mathrm { o u t } } - ( 1 + M + A ) = \varphi , } } \\ { { \ } } \\ { { K _ { \mathrm { o u t } } - ( 2 M + \rho \varphi ) = 1 + c , } } \\ { { \ } } \\ { { K _ { \mathrm { b r } } - ( M + A + \rho \varphi ) = c . } } \end{array}
$$

For the remaining ordinary cases,

$$
\begin{array} { c } { { K _ { \mathrm { o u t } } - ( 2 + M + \rho ) = 3 \varphi - 1 + 2 \alpha c > 0 , } } \\ { { K _ { \mathrm { o u t } } - ( 2 + 2 \rho ) = 5 \varphi - 1 + ( 3 \varphi - 4 ) c > 0 . } } \end{array}
$$

At $c = 0$ one non-bridge comparison is equality, which is sufficient. Strict endpoint eligibility assigns contact to the ineligible case; bridge uniqueness holds also when $x = y .$ . We have proved $F _ { J } ( x , y ) \leq K ( c ) w ( J )$ in every case.

For the charge hypothesis, assign each rejected I pieces $w ( I ) \tau _ { c } ( X ) / \sum _ { Y \in D } \tau _ { c } ( Y )$ to its current conflicts. Strict ordinary rejection and lifetime monotonicity bound each piece by $q _ { X } \geq w ( X )$ . These charges are used only in the analysis. The full antichain bound is at most $A h ,$ and $K _ { \mathrm { o u t } } - A = 3 \varphi + 2 > 0$ . Lemma D.5 therefore yields $\mathrm { O P T } \leq K ( c ) \mathrm { A L G }$ . The same upper bound holds on effective labels after filtering.

## H.4. Two finite families matching the envelope

Eight requests for the outward coefficient. Fix $0 < \epsilon < 1 / 1 0 0$ . Offer negative $K = [ 1 - \epsilon , d - \epsilon )$ , of length $\varphi ,$ then three positive tiles partitioning $[ 1 + \epsilon , d - \epsilon )$ equally. Each tile has length $( \varphi - 2 \epsilon ) / 3 < 1 = \alpha w ( K )$ , so it fails the special length test. Its length is also below dw $( K ) = M$ , so it rejects ordinarily. Each pays its full length to $K ,$ giving ledger $b _ { K } = \varphi - 2 \epsilon$ . For $0 \leq c \leq 1$ , put $g = M + c b _ { K }$ . The cost is exactly $^ { g , }$ since $g \leq M + \varphi < M \varphi$ . Next offer, in this order,

$$
\begin{array} { r l } { I } & { 0 \quad [ \epsilon , d - 2 \epsilon ) , } \\ { R } & { 0 \quad [ d - 2 \epsilon , d - 2 \epsilon + g - \epsilon ) , } \\ { J } & { 1 \quad [ 0 , 1 ) , } \\ { L } & { 0 \quad [ 2 \epsilon - M , \epsilon ) . } \end{array}
$$

Both I and R overlap the held K and have lengths $d - 3 \epsilon < g$ and $g - \epsilon < g$ , respectively; being negative, they reject without changing the ledger. The positive J fails ordinary acceptance and passes both special tests: $\varphi w ( J ) = w ( K )$ and $b _ { K } < \varphi$ . It replaces $K$ . Finally L overlaps J in positive length and has length $M - \epsilon < M$ , so it rejects against the positive ordinary cost. The algorithm ends with reward one.

The four positive intervals are compatible, so filtering leaves all labels unchanged. The intervals $L , I , R$ are consecutive, compatible, and tile the entire offered hull; all other requests are contained in that hull. Therefore their total is exactly

$$
\begin{array} { l } { { \mathrm { O P T } = ( M - \epsilon ) + ( d - 3 \epsilon ) + ( g - \epsilon ) } } \\ { { = K _ { \mathrm { o u t } } ( c ) - ( 5 + 2 c ) \epsilon . } } \end{array}\tag{H.12}
$$

Taking positive $\epsilon \downarrow 0$ gives the outward lower supremum for both raw and filtered policies.

Seven requests for the bridge coefficient. Fix $0 < \epsilon < 1 / 1 0 0$ , put $p = ( 1 - 2 \epsilon ) / 3 , b _ { 0 } = 1 - 2 \epsilon , g = d + c b _ { 0 }$ , and $a = g - \epsilon .$ . Offer negative $J = [ 0 , 1 )$ ), then three positive tiles partitioning $[ \epsilon , 1 - \epsilon )$ into lengths $p ,$ then negative

$$
L = [ \epsilon - a , \epsilon ) , \qquad R = [ 1 - \epsilon , 1 - \epsilon + a ) ,
$$

and finally positive $Q = [ 1 - \epsilon , 1 - \epsilon + \alpha )$ . Every tile fails ordinary acceptance and the special length test because $p < \alpha$ pays exactly p to the sole conflict $^ { J , }$ and leaves it held. The ledger becomes $b _ { 0 }$ . Its ordinary cost is exactly $g < M$ , since $d + c b _ { 0 } < d + 1 < M$ . Both exterior negatives overlap J in length ϵ and have length $a < g .$ , so they reject. The final $Q$ fails the ordinary gate but satisfies both special tests: $\varphi w ( Q ) = 1$ and $b _ { 0 } < 1$ . Thus $\mathrm { A L G } = \alpha$

The two exterior negatives and the three tiles form a feasible tiling of the entire input hull. All offered requests lie in it, including $Q \subset R$ because $a > \alpha$ . Thus

$$
\frac { \mathrm { O P T } } { \mathrm { A L G } } = \frac { 2 a + b _ { 0 } } { \alpha } = K _ { \mathrm { b r } } ( c ) - 4 \varphi ( c + 1 ) \epsilon .\tag{H.13}
$$

All positives are compatible, so the same execution occurs after filtering. Their feasible positive mass has ratio $d - 2 \varphi \epsilon < d ;$ the robustness witness has imperfect labels. Taking positive $\epsilon \downarrow 0$ gives the bridge lower supremum. Since $K _ { \mathrm { o u t } } - K _ { \mathrm { b r } } =$ $1 - \varphi c$ , the two constructions match the upper envelope, including the switch at $c = \alpha$ and both endpoints $c = 0 , 1$

## H.5. The global scalar minimum

It remains to exclude larger coefficients. For any fixed $c \geq 1$ , use the eight-request outward construction with the actual capped cost

$$
g = \mathrm { m i n } \{ M \varphi , M + c ( \varphi - 2 \epsilon ) \} .
$$

The tile rejections still follow from the lower cost M on K. The negative requests I, R have lengths $d - 3 \epsilon < g$ and $g - \epsilon < g$ , so reject. The same unit positive J fails the ordinary gate and passes both special tests; the held negative’s ledger is still $\varphi - 2 \epsilon$ . The final L rejects against M. The compatible $L , I , R$ still tile the entire input hull; hence $\mathrm { A L G } = 1$ and $\mathrm { O P T } = M + d + g - 5 \epsilon$ . Continuity of the minimum as $\epsilon \downarrow 0$ gives

$$
\begin{array} { c l c r } { { R ( T _ { c } ) , R ( T _ { c } ^ { F } ) \geq M + d + \operatorname* { m i n } \{ M \varphi , M + c \varphi \} } } \\ { { } } & { { } } \\ { { } } & { { = \varphi ^ { 5 } + \varphi \operatorname* { m i n } \{ c , \varphi \} } } \\ { { } } & { { \geq \varphi ^ { 5 } + \varphi > 4 \varphi + 5 . } } \end{array}\tag{H.14}
$$

This remains valid at $c = \varphi ,$ even if the cap is only reached in the limit. The filter changes no positive label.

On [0, 1] both affine robustness branches have strictly positive slope, so $K ( c )$ is strictly increasing. Since $c _ { * } < \alpha$

$$
K ( c _ { * } ) = \varphi ^ { 5 } + \varphi \alpha ^ { 3 } = \varphi ^ { 5 } + \alpha ^ { 2 } = 4 \varphi + 5 .
$$

Every $c < c _ { * }$ fails consistency d by the eleven-request construction. Every $c \in ( c _ { * } , 1 ]$ has strictly larger exact robustness, and (H.14) excludes every $c \geq 1$ regardless of its consistency. This proves the global minimum and uniqueness in Theorem H.1. Shaped-BANK’s exact robustness $\varphi ^ { 5 }$ improves this scalar minimum by $\alpha ^ { 2 }$

## I. The exact consistency threshold of SUM-BANK

SUM-BANK uses the constant cost λv for either prediction. Its optimal-consistency range follows from the source-measure theorem, while a separate finite construction proves necessity.

Theorem I.1 (SUM feasible-positive capacity). For every $\lambda \geq M = \varphi ^ { 3 }$ and feasible complete positive set P, each held interval satisfies

$$
b ( J ) \leq d w ( J ) , \qquad w ( P ) \leq d \mathrm { A L G } .\tag{I.1}
$$

Consequently $C ( S _ { \lambda } ) = d .$

Proof. Apply Theorem D.2 with $q _ { + } ( v , b ) = q ( v , b ) = \lambda v$ . Globally q ≥ dv and $q _ { + } \geq M v$ . For $0 \leq b \leq d v$

$$
\varphi v + \alpha b \leq 2 \varphi v < M v \leq \lambda v .
$$

Thus every hypothesis holds with the exact shared fees and inheritance. Conservation gives the feasible-positive bound. The finite robustness in Proposition J.1 and Theorem 2 give equality for consistency. □

Corollary I.1 (Exact threshold). For $\lambda > d , C ( S _ { \lambda } ) = d i f a n d o n l y i f \lambda \geq \varphi ^ { 3 } .$

Proof. The preceding theorem proves sufficiency. For $\smash { d < \lambda < \varphi ^ { 3 } }$ , Proposition I.1 gives

$$
C ( S _ { \lambda } ) \geq \varphi \left( 1 + { \frac { \varphi } { \lambda } } + { \frac { \varphi ^ { 3 } } { \lambda ^ { 2 } } } \right) > d .
$$

The expression decreases strictly in λ and equals d at $\varphi ^ { 3 }$

## I.1. A two-source lower bound with eleven requests

The next construction gives the converse for the sharp golden-line threshold. It concerns SUM-BANK on that parameter line.

Proposition I.1. For every real $\lambda > d = \varphi ^ { 2 }$

$$
C ( S _ { \lambda } ) \geq \varphi \left( 1 + { \frac { \varphi } { \lambda } } + { \frac { \varphi ^ { 3 } } { \lambda ^ { 2 } } } \right) .\tag{I.2}
$$

Eleven-request inputs with a unique optimum and perfect predictions approach this bound. In particular it exceeds d for $d < \lambda < \varphi ^ { 3 }$ and equals d at the endpoint in the limit.

Proof. Write $\beta = \varphi .$ , so $d = \beta + 1$ and $\beta ^ { 3 } = 2 \beta + 1 .$ . Fix $\lambda >$ d and choose

$$
\begin{array} { r } { 0 < q , \eta , \zeta < \frac { 1 } { 4 } , \qquad 0 < \varepsilon < \operatorname* { m i n } \{ \frac { 1 } { 1 0 0 } , \frac { \beta } { 4 \lambda } \} . } \end{array}
$$

Set

$$
\begin{array} { c } { { x = q - \beta + \varepsilon , \quad y = x + \varepsilon - \beta / \lambda , \quad m = ( q + 1 + \lambda - \eta ) / 2 , \nonumber } } \\ { { \nonumber } } \\ { { B = \lambda + \beta + \beta ^ { 3 } / \lambda - q - \eta - 3 \varepsilon , \qquad u = ( 1 - \zeta ) B / \beta . } } \end{array}
$$

Reveal the following distinct requests in the displayed order.

<table><tr><td>Request</td><td>Interval</td><td>Prediction</td></tr><tr><td> $Z$ </td><td> $[ y - d / \lambda + \varepsilon , y + \varepsilon )$ </td><td>0</td></tr><tr><td> $E _ { 1 }$ </td><td> $[ y - d / \lambda , y - d / ( 2 \lambda ) )$ </td><td>1</td></tr><tr><td> $E _ { 2 }$ </td><td> $[ y - d / ( 2 \lambda ) , y )$ </td><td>1</td></tr><tr><td>Y</td><td> $[ y , y + \beta / \lambda )$ </td><td>1</td></tr><tr><td>X</td><td> $[ x , x + \beta )$ </td><td>0</td></tr><tr><td>Q</td><td> $[ q , q + 1 )$ </td><td>1</td></tr><tr><td>J</td><td>[0, λ)</td><td>0</td></tr><tr><td> $T$ </td><td> $[ x + 2 \varepsilon , q )$ </td><td>1</td></tr><tr><td> $H _ { 1 }$ </td><td> $[ q + 1 , m )$ </td><td>1</td></tr><tr><td> $H _ { 2 }$ </td><td>[m, λ − η)</td><td>1</td></tr><tr><td> $U$ </td><td>[λ − η, λ − η + u)</td><td>1</td></tr></table>

All lengths are positive. In particular, $w ( T ) = \beta - 3 \varepsilon > 0$ and $w ( H _ { i } ) = ( \lambda - 1 - q - \eta ) / 2 > 0$ . Also $B > \lambda - 5 3 / 1 0 0 >$ $1 4 7 / 1 0 0$ and $u > ( 3 / 4 ) ( 1 4 7 / 1 0 0 ) / 2 = 4 4 1 / 8 0 0 > \eta ,$ , so U crosses the right endpoint of J. The positive requests in spatial order are $E _ { 1 } , E _ { 2 } , Y , T , Q , H _ { 1 } , H _ { 2 } , U$ . They are compatible: Y ends at $x + \varepsilon .$ , before T starts at $x + 2 \varepsilon$ , and all other consecutive contacts are at endpoints. Among negative requests, Z is compatible with X and J, whereas X overlaps J. Indeed, Z ends at $x + 2 \varepsilon - \beta / \lambda < x < 0$ , while X ends at $q + \varepsilon > 0$

We now follow the actual acceptance rules, with ordinary acceptance taking priority and equality accepted. Initially $Z$ is accepted. Each $E _ { i }$ has length $d / ( 2 \lambda ) < \beta / \lambda = w ( Z ) / \beta$ . Both tests fail, and each assigns its full length to its sole conflict Z. Hence $b ( Z ) = d / \lambda$ . Request Y overlaps Z by ε and fails the ordinary test. Both special thresholds hold at equality:

$$
w ( Y ) = \beta / \lambda = w ( Z ) / \beta = b ( Z ) / \beta .
$$

Thus Y replaces Z and has ledger $( d + \beta ) / \lambda = \beta ^ { 3 } / \lambda$ . Request X overlaps Y by ε and replaces it ordinarily at equality, $w ( X ) = \lambda w ( Y ) = \beta .$ , inheriting that ledger. Next Q fails the ordinary test, meets the special length test at equality, and satisfies its ledger test since $b ( X ) / \beta = d / \lambda < 1 = w ( Q )$ . Therefore $b ( Q ) = 1 + \beta ^ { 3 } / \lambda$ . Request J replaces $Q$ ordinarily at equality, with birth ledge $b _ { 0 } ( J ) = 1 + \beta ^ { 3 } / \lambda$

Each of $T , H _ { 1 } , H _ { 2 }$ has only J as a current conflict and length less than $\lambda / \beta \colon$ for T, use $\lambda > d = \beta ^ { 2 } ;$ ; for $H _ { i }$ , use $w ( H _ { i } ) < \lambda / 2 < \lambda / \beta$ . Each fails both ordinary acceptance and the special length test. Their full lengths are assigned to $J ,$ leaving ledger B just before $U$ . The special ledger test for U strictly fails because $u = ( 1 - \zeta ) B / \beta < B / \beta$ . The ordinary test also fails: $B < \lambda + 2 \beta$ implies $u < \lambda / \beta + 2 < \lambda ^ { 2 }$ . For the last inequality, $\lambda ^ { 2 } - \lambda / \beta - 2$ is increasing for $\lambda > d$ and equals $2 \beta > 0 \mathrm { a t } \lambda = d .$ . Whether or not the special length test fails, the sole-conflict allocation assigns all of u to $J .$ . The terminal solution is {J} and $\mathrm { A L G } = \lambda$ , while ledger conservation gives $w ( P ) = B + u$

To check perfect predictions, fix a feasible subset of negative requests and extend it by all compatible positives. This is the best extension since all weights are positive. Besides the empty subset, the only feasible choices and their deficits relative to $P$ are

$$
\begin{array} { r l } & { \left\{ Z \right\} } \\ & { \left\{ X \right\} } \\ & { \left\{ J \right\} } \\ & { \left\{ Z , X \right\} } \\ & { \left\{ Z , J \right\} } \end{array} \left| \begin{array} { l } { \beta / \lambda } \\ { 1 + \beta / \lambda - 3 \varepsilon } \\ { \beta - q - \eta - 3 \varepsilon + u } \\ { 1 + \beta / \lambda - 3 \varepsilon } \\ { \beta / \lambda + \beta - q - \eta - 3 \varepsilon + u . } \end{array} \right.
$$

All are strictly positive under the parameter restrictions. A proper subset of $P$ also loses positive weight. Thus $P$ is the unique optimum. Consequently the exact finite ratio is

$$
\frac { \mathrm { O P T } } { \mathrm { A L G } } = \left( 1 + \frac { 1 - \zeta } { \beta } \right) \left( 1 + \frac { \beta } { \lambda } + \frac { \beta ^ { 3 } } { \lambda ^ { 2 } } - \frac { q + \eta + 3 \varepsilon } { \lambda } \right) .
$$

Let all four positive perturbations tend to zero through their permitted ranges, keeping eleven requests. Since $1 + 1 / \beta = \beta _ { \mathrm { { \scriptsize { + } } } }$ this proves the supremum lower bound (I.2). Subtracting $\beta ( 1 + d / \lambda )$ gives $\beta ( \beta ^ { 3 } - \lambda ) / \lambda ^ { 2 }$ . The lower bound exceeds d exactly when $\lambda ^ { 2 } - d \lambda - d ^ { 2 } < 0$ , whose positive root is $\beta ^ { 3 }$ □

For an explicit finite witness, take $\lambda = 3 , q = 1 / 1 0 0$ , and $\varepsilon = \eta = \zeta = 1 / 1 0 0 0$ . The ratio is

$$
{ \frac { 2 5 0 2 4 7 9 + 7 4 7 4 0 2 1 \varphi } { 4 5 0 0 0 0 0 } } \approx 3 . 2 4 3 4 8 8 6 6 9 > d .
$$

## J. Joint geometric cuts for aggregate replacement

Proportional charging of rejected requests is used in Karavasilis (2025, Theorems 4.10–4.11). Here we bound the charge arriving at a final replacement tree. Its two exterior charges and the space left between them must be bounded together. The proof distinguishes an ancestry antichain from a feasible set of intervals: only the former is required for the intermediate maxima.

Theorem J.1 (Aggregate-gate transfer). Let $\beta \geq 1$ and $\lambda > \beta .$ . A causal policy accepts a request by revoking precisely its current conflicts, retains all nonconflicts, leaves its held set unchanged on rejection, and makes no other revocations. Label each acceptance M $o r { \mathsf { S } } ,$ , with empty-conflict acceptances labelled M. Assume that:

1. An M node’s children have total length at most its length divided by λ.

2. An ${ \sf S }$ node has only M children, with total length at most $\beta$ times its length.

3. Every rejected request I has a nonempty conflict set D and satisfies $w ( I ) < \lambda w ( D )$

Put

$$
q _ { M } = \frac { \lambda + 1 } { \lambda - \beta } , q _ { S } = \frac { \lambda ( \beta + 1 ) } { \lambda - \beta } .\tag{J.1}
$$

$$
K ( \beta , \lambda ) = \operatorname * { m a x } \{ \beta ( 2 \lambda + 1 ) , ( \beta + 1 ) \lambda + q _ { S } , \beta \lambda + \beta ^ { 2 } + q _ { S } \} .\tag{J.2}
$$

Every feasible request set O then satisfies $w ( O ) \le K ( \beta , \lambda ) \mathrm { A L G }$ . No correctness or feasibility of the predictions is assumed.

Proof. We first bound each replacement tree and then distribute the optimal requests over the final trees.

Replacement trees and restricted antichains. Each accepted identity is a node, made a child of its unique replacement if revoked. The roots are the final held requests. Child roots are compatible, and each overlaps its parent in positive length. An ancestry antichain contains no node and its strict ancestor. Subtree induction bounds its mass by h at an M root of length h, and by βh at an S root. If the root is included, the antichain is its singleton. Otherwise an M root gives mass at most $\beta h / \lambda < h$ , while an S root sums the M child bounds to at most $\beta h$

Let $H _ { J }$ be the closure of the union of the accepted intervals in $J { \boldsymbol { \mathbf { \mathit { s } } } }$ subtree. This union is connected along replacement edges. The identities $q _ { M } = 1 + q _ { S } / \lambda$ and $q _ { S } = 1 + \beta q _ { M }$ bound the sum of all node lengths, and hence $| H _ { J } |$ , by $q _ { M } w ( J )$ or $q _ { S } w ( J )$ according to the root’s type. Also $q _ { S } \geq q _ { M }$ . Different final hulls need not be disjoint.

For arbitrary real $x \leq y ,$ let $L _ { J } ( x )$ be the maximum mass of an ancestry antichain restricted to nodes V with $s ( V ) < x$ Define $U _ { J } ( y )$ using $f ( V ) > y$ . Empty antichains are allowed. Set

$$
F _ { J } ( x , y ) = | H _ { J } \cap [ x , y ] | + \lambda \{ L _ { J } ( x ) + U _ { J } ( y ) \} .\tag{J.3}
$$

We prove $F _ { J } ( x , y ) \leq K ( \beta , \lambda ) w ( J )$ for every subtree. These maxima may come from different times.

An ordinary root. Write $\boldsymbol { J } = [ a , b )$ and $h = b { - } a$ . Root-omitting antichains have mass at most $\beta h / \lambda < h$ . Consequently

$$
L _ { J } ( x ) = h \quad ( a < x ) , \qquad L _ { J } ( x ) \leq \beta h / \lambda \quad ( a \geq x ) ,
$$

and the analogous right statements hold with $b > y$ and $b \leq y .$ If both root endpoint tests hold, the gap lies inside $^ { J , }$ and $F _ { J } \leq ( 2 \lambda + 1 ) h .$ . If exactly one holds, the coefficient is at most $q _ { M } + \lambda + \beta$ . It does not exceed $( \beta + 1 ) \lambda + q _ { S }$ , since the difference is $\beta ( \lambda - 1 ) + ( q _ { S } - q _ { M } ) \ge 0$ . If neither holds, the coefficient $q _ { M } + 2 \beta$ is smaller still because $\lambda > \beta$

A special root. Its ordinary children $C _ { i }$ have lengths $h _ { i }$ and total $\begin{array} { r } { H = \sum _ { i } h _ { i } \le \beta h } \end{array}$ . The exact recursions are

$$
L _ { J } ( x ) = \operatorname* { m a x } \{ h \mathbf { 1 } _ { a < x } , \sum _ { i } L _ { C _ { i } } ( x ) \} , \qquad U _ { J } ( y ) = \operatorname* { m a x } \{ h \mathbf { 1 } _ { b > y } , \sum _ { i } U _ { C _ { i } } ( y ) \} .
$$

Select maximizing branches, using children when the maximum is zero. If both selections are the singleton root, the gap lies in J, giving $( 2 \lambda + 1 ) h$ . If one selects the root and the other selects children, the coefficient is at most $q _ { S } + ( \beta + 1 ) \lambda$

Suppose both select children. Call $C _ { k }$ a bridge when $s ( C _ { k } ) < x \leq y < f ( C _ { k } )$ . At most one bridge exists, including when $x = y$ , because two would overlap in positive length. If it exists, the entire gap lies in $C _ { k }$ and both its maxima equal $h _ { k }$ Every other child has at most one eligible root endpoint, so its two maxima sum to at most $( 1 + \beta / \lambda ) h _ { i }$ . This also covers two ineligible endpoints, since $2 \beta / \lambda < 1 + \beta / \lambda$ . Therefore

$$
F _ { J } \le ( 2 \lambda + 1 ) h _ { k } + ( \lambda + \beta ) ( H - h _ { k } ) \le ( 2 \lambda + 1 ) H \le \beta ( 2 \lambda + 1 ) h .
$$

If there is no bridge, the hull bound instead gives $F _ { J } \le q _ { S } h + ( \lambda + \beta ) H \le ( q _ { S } + \beta \lambda + \beta ^ { 2 } ) h$ . These cases prove the asserted joint-cut bound. Equality at an endpoint is consistently assigned to the ineligible case.

From cuts to actual optimal requests. Apply Lemma D.5 with $\omega _ { V } = \lambda w ( V ) \geq w ( V )$ . Its endpoint maxima are λ times the length maxima used here, so its $F _ { J }$ is exactly (J.3). A rejected I sends $w ( I ) w ( V ) / w ( D ) < \lambda w ( V )$ to each current conflict V. The full weighted antichain bound is at most $\lambda \beta h$ , below $K ( \beta , \lambda ) h$ since $K \ge \beta ( 2 \lambda + 1 )$ . The established joint-cut bound therefore proves the theorem. The empty input and empty O are immediate. □

Lemma J.1 (A single-conflict witness for SUM). For every $\lambda > d ,$ the following family gives SUM-BANK ratio approaching $( 2 \lambda + 1 ) \varphi$ . Thefamily has one positive prediction, so permanentfiltering leaves it unchanged.

Proof. Put $\alpha = 1 / \varphi$ and choose $0 < \tau < \operatorname* { m i n } \{ ( 1 - \alpha ) / 2 , ( \lambda - 1 ) / 2 \}$ . Reveal

$$
[ 0 , 1 ) _ { 0 } , \quad [ - \lambda + 2 \tau , \tau ) _ { 0 } , \quad [ \tau , 1 - \tau ) _ { 0 } , \quad [ 1 - \tau , 1 + \lambda - 2 \tau ) _ { 0 } , \quad [ \tau , \tau + \alpha ) _ { 1 } ,
$$

where subscripts are prediction labels. The first request is accepted. Each later request has exactly one maintained conflict. The three subsequent negative requests have length below λ, are rejected, and add no ledger. The last request passes the special length test at equality with zero conflicting ledger, leaving $\mathrm { A L G } = \alpha$ . The middle three requests are pairwise compatible and form the unique optimum: the central one strictly contains and outweighs the last positive request, while the initial unit interval conflicts with every other request and has smaller weight than their packing. Their total is $2 \lambda { + } 1 { - } 4 \tau$ Letting $\tau \downarrow 0$ proves the ratio through finite inputs with positive overlaps. □

Proposition J.1 (SUM robustness). For every $\lambda > d ,$

$$
\varphi ( 2 \lambda + 1 ) \le R ( S _ { \lambda } ) \le \overline { { R } } _ { S } ( \lambda ) : = K ( \varphi , \lambda ) .\tag{J.4}
$$

For $\lambda \geq \varphi ^ { 3 }$ , these bounds match:

$$
R ( S _ { \lambda } ) = \varphi ( 2 \lambda + 1 ) , \qquad R ( S _ { \varphi ^ { 3 } } ) = 7 \varphi + 4 .\tag{J.5}
$$

For $3 \le \lambda \le \varphi ^ { 3 }$ , the upper coefficient is $d \lambda + \lambda d / ( \lambda - \varphi )$

Proof. Ordinary acceptance gives the aggregate child bound. Special acceptance has only negative children, each of type M, and its length gate gives total child length at most $\varphi w ( I )$ . Every rejection fails the ordinary gate. Thus all transfer hypotheses hold with $\beta = \varphi$ , independently of the ledgers and labels.

The first term of K dominates when $q _ { S } \le ( \varphi - 1 ) \lambda + \varphi$ and $q _ { S } \le \varphi \lambda - 1 . ~ \mathrm { A t } ~ \lambda = \varphi ^ { 3 }$ the first is an equality and the second has positive slack; $q _ { S }$ decreases while both right sides increase. Below that threshold, down to 3, the mixed term dominates the first; it exceeds the third by $\lambda - d > 0$ . This proves the stated upper coefficients.

The lower bound is supplied by Lemma J.1.

Corollary J.1 (Exact robustness of Revoke-Proportional). For every $\lambda > 1$ , the unchanged Revoke-Proportional algorithm of Karavasilis (2025, Algorithm 4) satisfies

$$
R ( \mathrm { R P } _ { \lambda } ) = \operatorname* { m a x } \left\{ 2 \lambda + 2 , \lambda + \frac { 2 \lambda } { \lambda - 1 } \right\} = \left\{ \lambda + 2 \lambda / ( \lambda - 1 ) , ~ 1 < \lambda \leq 2 , \right.\tag{J.6}
$$

The sameformula holds with the permanent predictionfilter ofSection K. Thus 2 is the exact cutofffor theformula $2 \lambda + 2 .$ The published consistency bound $3 \lambda / ( \lambda - 1 )$ is unchanged.

Proof. RP has the ordinary aggregate gate. Its special branch requires a positive label, only negative conflicts, and new length at least their total. Thus the replacement-tree and charge-conversion definitions of Theorem J.1 apply with $\beta = 1$ Every subtree ancestry antichain has mass at most its root length. In particular, an eligible singleton root attains the corresponding restricted maximum $L _ { J }$ or $U _ { J }$

Coupled one-sided tails. For $\boldsymbol { J } = [ a , b )$ of length h and $y \geq a$ , set

$$
T _ { J } ( y ) = \left| H _ { J } \cap [ a , y ] \right| + \lambda U _ { J } ( y ) , \qquad t _ { M } = \operatorname* { m a x } \left\{ \lambda + 1 , \frac { \lambda + 1 } { \lambda - 1 } \right\} , \quad t _ { S } = t _ { M } + 1 .
$$

We claim $T _ { J } ( y ) \leq t _ { M } h$ at an ordinary root and $T _ { J } ( y ) \leq t _ { S } h$ at a special root. The reflected statement also holds. If $a \leq y < b ,$ , the root attains $U _ { J } ( y ) = h$ , and the bound is $( \lambda + 1 ) h . \mathrm { H } y \geq b ,$ the root is ineligible and $\begin{array} { r } { U _ { J } ( y ) = \sum _ { i } U _ { C _ { i } } ( y ) } \end{array}$ The closed hull $H _ { J }$ is the union of $[ a , b ]$ and its child hulls. Each child starts at $a _ { i } < b \leq y $ , so

$$
\left| H _ { J } \cap [ b , y ] \right| \leq \sum _ { i } \left| H _ { C _ { i } } \cap [ b , y ] \right| \leq \sum _ { i } \left| H _ { C _ { i } } \cap [ a _ { i } , y ] \right| .
$$

Consequently $\begin{array} { r } { T _ { J } ( y ) \leq h + \sum _ { i } T _ { C _ { i } } ( y ) } \end{array}$ . At an ordinary root the child lengths sum to at most $h / \lambda$ , giving $( 1 + t _ { S } / \lambda ) h \le t _ { M } h$ because $( \lambda - 1 ) t _ { M } \geq \lambda + 1$ . At a special root all children are ordinary and their lengths sum to at most $h ,$ giving $( 1 + t _ { M } ) h = t _ { S } h .$ . This proves the tail bounds by induction, including endpoint equality.

Joint cuts. Set $c _ { M } = \lambda + t _ { M }$ and $c _ { S } = c _ { M } + 1$ . We show that the functional $F _ { J } ( x , y )$ from (J.3) is at most $c _ { M } h$ or $c _ { S } h$ according to the root type. If both root endpoints are eligible, the gap is within J and $F _ { J } \leq ( 2 \lambda + 1 ) h \leq c _ { M } h$ . If exactly one is eligible, say $a < x$ and $b \leq y .$ , then $L _ { J } ( x ) = h$ and

$$
{ \cal F } _ { J } ( x , y ) \leq \lambda h + | H _ { J } \cap [ a , y ] | + \lambda { \cal U } _ { J } ( y ) = \lambda h + T _ { J } ( y ) .
$$

The tail bound gives the desired coefficient for either type. Reflection handles the other orientation.

If neither endpoint is eligible, both restricted maxima decompose over children, and the hull union yields $F _ { J } ( x , y ) \ \leq$ $\begin{array} { r } { h + \sum _ { i } F _ { C _ { i } } ( x , y ) } \end{array}$ . At a special root this is at most $( 1 + c _ { M } ) h = c _ { S } h$ . At an ordinary root it is at most $( 1 + c _ { S } / \lambda ) h \le c _ { M } h$ since $( \lambda - 1 ) c _ { M } \geq \lambda + 1$ . Equalities in endpoint tests are ineligible cases; an eligible root may always be selected in an antichain tie. These cases cover all rea $x \leq y$

The charge conversion in Theorem J.1 now gives $R \leq c _ { S } ;$ its whole-hull covering case has charge at most $\lambda h \le c _ { S } h$ Algebraically, $c _ { S } = \operatorname* { m a x } \{ 2 \lambda + 2 , \lambda + 2 \lambda / ( \lambda - 1 ) \}$

Lower bound for $\lambda \geq 2 .$ . For $0 < \varepsilon < 1 / 8$ , reveal

$$
\begin{array} { c } { { K } } \\ { { G } } \\ { { U } } \\ { { J } } \\ { { L } } \end{array} \left| \begin{array} { c } { { \left[ 1 - \varepsilon , 2 - \varepsilon \right) } } \\ { { \left[ \varepsilon , 2 - 2 \varepsilon \right) } } \\ { { \left[ 2 - 2 \varepsilon , 2 - 2 \varepsilon + \lambda - \varepsilon \right) } } \\ { { \left[ 0 , 1 \right) } } \\ { { \left[ - \lambda + 2 \varepsilon , \varepsilon \right) } } \end{array} \right| \begin{array} { c } { { 0 } } \\ { { 0 } } \\ { { 0 } } \\ { { 1 } } \\ { { 0 . } } \end{array}
$$

The trace is ordinary acceptance, two rejections, special acceptance, and rejection. In particular $w ( G ) = 2 - 3 \varepsilon < \lambda$ , also at $\lambda = 2$ . The final value is one, and the compatible comparison set $\{ L , G , U \}$ has value $2 \lambda + 2 - 5 \varepsilon . \operatorname { L e t } \varepsilon \downarrow 0$

Alternating-chain lower bound. For any $\lambda > 1$ , integer $n \geq 0$ , and $0 < \varepsilon < 1 / 2$ , define candidate lengths

$$
\ell _ { 2 k } = \ell _ { 2 k + 1 } = \lambda ^ { k - n } \quad ( 0 \leq k \leq n ) , \qquad a _ { 0 } = 0 , \quad a _ { i + 1 } = a _ { i } + ( 1 - \varepsilon ) \ell _ { i } .
$$

Reveal $A _ { i } = [ a _ { i } , a _ { i } + \ell _ { i } )$ in increasing i, labelling even indices negative and odd indices positive. Immediately after $A _ { i }$ reveal the negative filler

$$
F _ { i } = [ a _ { i } + \varepsilon \ell _ { i } , a _ { i } + ( 1 - \varepsilon ) \ell _ { i } ) .
$$

Each even candidate after the first ordinarily replaces its sole predecessor at growth equality. Each odd candidate specially replaces its negative predecessor at length equality. Consecutive candidates overlap in positive length, every filler is rejected, and the final held set is the singleton $A _ { 2 n + 1 }$ of length one. Writing $b = f ( A _ { 2 n + 1 } )$ , finally reveal the negative request $U = \left[ b - \varepsilon , b - \varepsilon + \lambda - \varepsilon \right)$ . It overlaps the held interval and is rejected.

The fillers are compatible: $F _ { i }$ ends at $a _ { i + 1 }$ and the next filler starts strictly to its right. The final filler ends where $U$ starts. They therefore give the feasible comparison value

$$
( 1 - 2 \varepsilon ) 2 \sum _ { k = 0 } ^ { n } \lambda ^ { k - n } + \lambda - \varepsilon .
$$

Letting first $\varepsilon \downarrow 0$ and then $n \to \infty$ gives $R \geq \lambda { + } 2 \lambda / ( \lambda { - } 1 )$ . All executions in the family are finite and use positive-length overlaps. For $1 < \lambda < 2$ this limit exceeds $2 \lambda + 2$ b $( 2 - \lambda ) ( \lambda + 1 ) / ( \lambda - 1 ) > 0 ;$ ; for $\lambda \geq 2$ the preceding five-request family gives the larger branch.

Finally, consecutive positive carriers in the chain, of lengths a and $\lambda a ,$ have gap $[ \lambda - \varepsilon ( 1 + \lambda ) ] a > 0 .$ . Thus all positive predictions are feasible and the permanent filter changes no label. The five-request family has just one positive. The upper bound allows arbitrary effective labels, proving the filtered formula as well. □

## K. Prediction filtering and zero-error inputs

Fix an optimum O, the original positive set $P ,$ and the set $N _ { O } ( F )$ of optimum requests conflicting with F. The error of Karavasilis (2025) is

$$
\eta _ { K } ( P , O ) = w ( O \setminus P ) + \sum _ { F \in P \setminus O } [ w ( N _ { O } ( F ) ) - w ( F ) ] .\tag{K.1}
$$

Optimality makes each summand nonnegative; a false positive of equal weight to its optimal conflicts contributes zero.

Proposition K.1 (Zero-error limitation). No finite $k \geq 0$ makes $\mathrm { O P T } \leq a \mathrm { A L G } + k \eta _ { K } ( P , O )$ valid for all inputs of raw Shaped-BANK when $a < 4 \varphi + 1$ , or raw scalar Credit with fixed $c > 0$ when $a < 4 \varphi + 3 .$ . For SUM-BANK at $\lambda > d ,$ the obstruction holds for $a < 2 \lambda + 1$ . Finite witnesses with pairwise distinct interval geometries have $\mathrm { A L G } = 1 , \mathrm { O P T } > a$ and zero error for every optimum.

Distinct overlapping equal-length alternatives inflate the ledger while contributing zero error; their complete positive set is infeasible.

The filter maintains a permanent feasible history P<sup>′</sup>. An originally positive request receives effective label 1 precisely when it is compatible with all of $P ^ { \prime } ;$ insert it immediately, before running the selected policy. Every other effective label is 0. Every request reaches the algorithm, and the effective label is fixed for all decisions and ledger updates. Requests remain in $P ^ { \prime }$ after rejection or revocation. This is the Naive selection rule of Karavasilis (2025), used as a filter. Denote the compositions by $\bar { T } _ { * } ^ { F } , T _ { c } ^ { F }$ , and $S _ { \lambda } ^ { F }$

Corollary K.1 (Simultaneous guarantees). For original labels and fixed optimum O, Shaped-BANK satisfies

$$
\mathrm { A L G } ( T _ { * } ^ { F } ) \geq \operatorname* { m a x } \left\{ \frac { \mathrm { O P T } } { \varphi ^ { 5 } } , \frac { \mathrm { O P T } - \eta _ { K } ( P , O ) } { d } \right\} ,\tag{K.2}
$$

with $C = d$ and exact $R = \varphi ^ { 5 } .$ . The same simultaneous formula holds for scalar Credit $T _ { c } ^ { F } , c \in [ \alpha ^ { 3 } , 1 ] ,$ , with robustness $K ( c )$ from Theorem H.1; its minimum is $4 \varphi + 5 .$ . For SUM at $\lambda \geq \varphi ^ { 3 }$ , replace the robustness denominator b $( 2 \lambda + 1 ) \varphi .$ These policies also have $C = d$ and exact stated robustness.

## K.1. The permanent-history filter and its quality

We analyze the permanent history $P ^ { \prime }$ defined above, produced by the NAIVE rule (Karavasilis, 2025, Algorithm 1). Original labels determine $\eta _ { K } ( P , O )$ ; effective labels determine every policy decision and ledger increment.

Lemma K.1 (Quality of the known NAIVE filter). For everyfinite request stream and everyfixed optimum O, the permanent history produced by thefilter in Section K isfeasible, is a subset ofP, and satisfies

$$
w ( P ^ { \prime } ) \geq \mathrm { O P T } - \eta _ { K } ( P , O ) .\tag{K.3}
$$

The inequality is the NAIVE guarantee of Karavasilis (2025, Theorem 3.5).

Proof. Each insertion preserves compatibility with every earlier insertion, so the entire set $P ^ { \prime }$ is feasible. Only originally positive requests are inserted, giving $P ^ { \prime } \subseteq P .$ For $F \in P \setminus O .$ , set

$$
e _ { F } = w ( N _ { O } ( F ) ) - w ( F ) .
$$

The set $( O \setminus N _ { O } ( F ) ) \cup \{ F \}$ is feasible. Optimality of O therefore gives

$$
w ( O ) \geq w ( O ) - w ( N _ { O } ( F ) ) + w ( F ) ,
$$

and hence $e _ { F } \geq 0$ for every such $F .$

Partition the positive predictions and their filtered subset as

$$
A = P \cap O , \quad A ^ { \prime } = P ^ { \prime } \cap O , \qquad B = P \setminus O , \quad B ^ { \prime } = P ^ { \prime } \setminus O .
$$

If $I \in A \backslash A ^ { \prime } .$ it was blocked at its arrival by some earlier request $F \in P ^ { \prime }$ . Since $I \in O$ and O is feasible, that blocker cannot belong to $O ;$ thus $F \in B ^ { \prime }$ . Consequently,

$$
A \setminus A ^ { \prime } \subseteq \bigcup _ { F \in B ^ { \prime } } N _ { O } ( F ) , \qquad w ( A \setminus A ^ { \prime } ) \leq \sum _ { F \in B ^ { \prime } } w ( N _ { O } ( F ) ) .\tag{K.4}
$$

Overlapping neighborhoods may count a request more than once, which only enlarges the right-hand side. Using $B ^ { \prime } \subseteq B$

and the nonnegativity of every $e _ { F } ,$ , we obtain

$$
\begin{array} { r l } { w ( P ^ { \prime } ) = w ( A ) - w ( A \backslash A ^ { \prime } ) + w ( B ^ { \prime } ) } \\ { \ } & { \geq w ( A ) - \displaystyle \sum _ { F \in \mathcal { B } ^ { \prime } } w ( N o ( F ) ) + \displaystyle \sum _ { F \in \mathcal { B } ^ { \prime } } w ( F ) } \\ { \ } & { = w ( A ) - \displaystyle \sum _ { F \in \mathcal { B } ^ { \prime } } e _ { F } } \\ { \ } & { \geq w ( A ) - \displaystyle \sum _ { F \in \mathcal { B } } e _ { F } } \\ { \ } & { = w ( O ) - w ( O \backslash P ) - \displaystyle \sum _ { F \in P \backslash O } e _ { F } } \\ { \ } & { = \mathrm { O P T } - \eta _ { K } ( P , O ) . } \end{array}
$$

This also accounts for false positives that the filter never inserts.

## K.2. Composition with the feasible-set guarantee

ProofofCorollary K.1. Let A be a policy with a feasible-positive guarantee w $( P ^ { \prime } ) \leq d \mathrm { A L G } _ { A }$ and an arbitrary-label robustness bound $\mathrm { O P T } \le R _ { A } \mathrm { A L G } _ { A }$ . The filter passes every request to $A$ with an effective label fixed on arrival. Its complete effective positive set is the feasible set $P ^ { \prime }$ , while the offline optimum is unchanged. Lemma K.1 therefore gives

$$
\mathrm { A L G } _ { A ^ { F } } \geq \operatorname* { m a x } \left\{ \frac { \mathrm { O P T } } { R _ { A } } , \frac { \mathrm { O P T } - \eta _ { K } ( P , O ) } { d } \right\} .
$$

This remains valid when the second term is negative. The required policy-specific bounds are as follows.

<table><tr><td>Policy and range</td><td> $R _ { A }$ </td><td>Feasible-positive bound</td><td>Robustness bound</td></tr><tr><td>Shaped-BANK</td><td> $\varphi ^ { 5 }$ </td><td>Theorem D.2</td><td>Theorem D.4</td></tr><tr><td>Credit,  $c \in [ \alpha ^ { 3 } , 1 ]$ </td><td> $K ( c )$ </td><td>Theorem H.1</td><td>Theorem H.1</td></tr><tr><td>SUM,  $\lambda \geq \dot { \varphi } ^ { 3 }$ </td><td> $( 2 \lambda + 1 ) \varphi$ </td><td>Theorem I.1</td><td>Proposition J.1</td></tr></table>

In particular, the Credit specialization is

$$
\mathrm { A L G } ( T _ { c } ^ { F } ) \geq \operatorname* { m a x } \left\{ \frac { \mathrm { O P T } } { K ( c ) } , \frac { \mathrm { O P T } - \eta _ { K } ( P , O ) } { d } \right\} .\tag{K.5}
$$

For perfect original predictions, every positive belongs to a feasible optimum, so filtering changes no label. Thus the filtered and raw consistency suprema agree and equal $d .$ To retain exact robustness, one must also check the matching witnesses. Shaped-BANK’s family in Proposition D.5 has one positive request. Both Credit families in Theorem H.1 have compatible positive sets. The SUM family in Lemma J.1 has one positive request. All therefore pass unchanged, proving equality with the listed $R _ { A }$ . For Credit this includes $4 \varphi + 5$ at $c = \alpha ^ { 3 }$ and $6 \varphi + 3$ at $c = \varphi / 2$ □

Standalone Naive returns $P ^ { \prime }$ and has the stronger coefficient-one error bound but unbounded robustness: a single negative labelled request gives it zero reward. At the optimal-consistency configurations, the compositions add finite robustness while attaining the smallest deterministic consistency compatible with it.

## K.3. A zero-error obstruction with distinct intervals

Proposition K.2 (Uniform zero-error obstruction). For SUM-BANK at every $\lambda > d ,$ with $\alpha = 1 / \varphi$ and $\beta = \varphi ,$ and every $c < 2 \lambda + 1$ , there is a finite input with pairwise distinct interval geometries, $\mathrm { A L G } = 1$ $\mathrm { O P T } > c ,$ and $\eta _ { K } ( P , { \cal O } ) = 0$ for every optimum O. Therefore no finite $k \geq 0$ makes $\mathrm { O P T } \leq c \mathrm { A L G } + k \eta _ { K } ( P , O )$ validfor all inputs ofthat unfiltered configuration.

Proof. Every arrival below has at most one maintained conflict. Choose $\delta > 0$ with $\delta < 1 / 4 , \lambda - \delta > \operatorname* { m a x } \{ \alpha , \delta \}$ , and $2 \lambda + 1 - 5 \delta > \operatorname* { m a x } \{ c , 1 \}$ . Choose an integer $n > 1 / \alpha .$ , set $h = ( 1 - 3 \delta ) / n$ , and choose an integer m large enough that

$$
B = m n h = m ( 1 - 3 \delta ) > \beta ( \lambda - \delta ) .
$$

For $j = 1 , \dotsc , n$ let $a _ { j } = \delta + ( j - 1 ) ( h + \delta / n )$ and define m positive requests in group j by

$$
I _ { j , k } = \left[ a _ { j } + \frac { k \delta } { m n } , a _ { j } + \frac { k \delta } { m n } + h \right) , \qquad k = 0 , \ldots , m - 1 .
$$

Within a group, all starts are different and their total span is less than $\delta / n ~ < ~ h .$ , so all intervals overlap in positive length. Consecutive groups are separated by a gap $\delta / { \left( m n \right) } > 0 .$ . The first group starts at δ and the last endpoint is $1 - \delta - \delta / ( m n ) < 1 - \delta$ . Thus all central intervals have distinct geometries and lie in $[ \delta , 1 - \delta )$

Offer the negative anchor $J = [ 0 , 1 )$ , then all the $I _ { j , k }$ in lexicographic $( j , k )$ order, and finally the positive requests

$$
L = [ 2 \delta - \lambda , \delta ) , \qquad U = [ 1 - \delta , 1 + \lambda - 2 \delta ) .
$$

Each central request has length $h < \alpha < \lambda ,$ , so it fails both ordinary acceptance and the special length test. Its sole conflict is $^ { J , }$ which receives its full length. After all central requests the ledger is B. Each boundary request has length $\lambda - \delta < \lambda$ and fails the special ledger test because the current ledger is at least $B > \beta ( \lambda - \delta )$ . Both reject and the final reward is one. The strict overlaps with J have length δ, and no requests in this construction are geometrically identical.

Every other request conflicts with J. A feasible set avoiding J contains at most one interval of each central group, and can contain $L , U$ together with any such selection. All weights are positive, and

$$
\mathrm { O P T } = 2 ( \lambda - \delta ) + n h = 2 \lambda + 1 - 5 \delta > \operatorname* { m a x } \{ c , 1 \} .
$$

Consequently the optima are exactly $L , U$ and one representative per group. Every optimum is contained in $P .$ An unselected positive conflicts with exactly one of its members, the chosen representative of its own group, which has the same length h. Every term of (K.1) is therefore zero, for every optimum. Substitution proves the assertion for all finite $k .$ □

The construction has multiple optima but no coincident intervals. The complete positive set contains overlapping alternatives, so the predictions are not perfect. $\mathrm { A t } \lambda = \varphi ^ { 3 }$ it excludes every error-bound coefficient below $4 \varphi + 3$

Corollary K.2 (Credit-BANK has the same zero-error limitation). For every fixed $c > 0$ , every $a < 4 \varphi + 3$ , and every finite $k \geq 0$ , raw $T _ { c } f a i l s \mathrm { O P T } \leq a \mathrm { A L G } + k \eta _ { K }$ on somefinite input with distinct interval geometries.

Proof. Use the construction above with $\lambda = M = \varphi ^ { 3 }$ and choose $\delta$ so that $2 M + 1 - 5 \delta > a$ . Increase m until

$$
B = m ( 1 - 3 \delta ) > \operatorname* { m a x } \{ \varphi ( M - \delta ) , \varphi / c \} .
$$

Initially the unit negative anchor costs $d ,$ and every central length satisfies $h < \alpha < d ,$ so all central requests reject. The inequality $c B > \varphi = M - d$ brings the anchor cost to its cap M before $L , U$ arrive. Each boundary request then fails the ordinary and ledger tests, and subsequent deposits preserve the cap. The reward is one and the same optimum calculation gives ${ \mathrm { O P T } } = 2 M + 1 - 5 \delta > a$ and zero error for every optimum. □

Corollary K.3 (Shaped-BANK’s raw zero-error limitation). For every $a < 4 \varphi + 1$ and every finite $k \geq 0 ,$ , raw Shaped-BANKfails $\mathrm { O P T } \leq a \mathrm { A L G } + k \eta _ { K } ( P , O )$ on afinite input with pairwise distinct interval geometries, $\mathrm { A L G } = 1 , \mathrm { O P T } > a ,$ and $\eta _ { K } ( P , O ) = 0 f o r$ every optimum O.

Proof. Use Proposition ${ \mathrm { K } } { . } 2 { : } \mathrm { s }$ overlapping-group construction with $\lambda = 2 \varphi$ . Choose $\delta > 0$ small enough that $4 \varphi + 1 - 5 \delta >$ $a .$ The initial negative unit anchor costs $d ,$ and all central requests have length $h < \alpha < d .$ They therefore reject ordinarily and by the special length test, each depositing its whole length. The finite multiplicity m gives

$$
B = m ( 1 - 3 \delta ) > \varphi ( 2 \varphi - \delta ) > d .
$$

The shaped cost has reached its cap $2 \varphi$ before the two boundary arrivals. Each boundary length is $2 \varphi - \delta$ , below both its ordinary threshold and the ledger threshold. Both reject, their deposits preserve the cap, and the reward is one. The optimal sets are the two boundary requests and one equal-length representative of each central group. They have weight $4 \varphi + 1 - 5 \delta > a ,$ , and every unchosen positive has exactly one equal-weight conflict in each optimum. Thus $\eta _ { K } ( P , { \cal O } ) = 0$ for every optimum, as required. □

The two policies’ different caps give different zero-error coefficients. For raw Shaped-BANK, $4 \varphi + 1 > d$ already excludes the optimal-consistency coefficient in such a bound.

## K.4. Exact implementation of the special tests

For rational $w > 0$ and $X \geq 0 ,$

$$
w \geq X / \varphi \quad \iff \quad w ^ { 2 } + w X - X ^ { 2 } \geq 0 .
$$

For $X > 0 ,$ , divide by $X ^ { 2 }$ and use that $t ^ { 2 } + t - 1$ is strictly increasing for $t \geq 0$ with root $1 / \varphi ;$ for $X = 0$ , both conditions hold. Taking $X = W$ or $X = B$ implements the two special tests without rounding the golden ratio.

## K.5. Precision, recall, and confidence scores

Classification summaries and conflict-weighted error answer different questions. Fix an optimum O of positive weight and write $T = P \cap O , { \mathcal { F } } = P \setminus O$ . The length-weighted recall is $r _ { w } = w ( T ) / w ( O )$ ; when $w ( P ) > 0$ , length-weighted precision is $p _ { w } = w ( T ) / w ( P )$ . For each false-positive request $F \in { \mathcal { F } }$ define $\kappa _ { \ / F } = w ( N _ { \cal O } ( { \cal F } ) ) / w ( { \cal F } ) \geq 1$ . The error definition gives exactly

$$
\frac { \eta _ { K } ( P , O ) } { \mathrm { O P T } } = 1 - r _ { w } + \sum _ { F \in P \setminus O } \frac { w ( F ) } { \mathrm { O P T } } ( \kappa _ { F } - 1 ) .
$$

If $p _ { w } > 0$ and every false positive has $\kappa _ { F } \leq \kappa ,$ , this implies the conditional bound

$$
\frac { \eta _ { K } } { \mathrm { O P T } } \leq 1 - r _ { w } + ( \kappa - 1 ) r _ { w } \frac { 1 - p _ { w } } { p _ { w } } .
$$

There is no universal bound on κ in our model: a short false-positive interval can conflict with a much longer optima interval.

Indeed, fix $0 < \epsilon < 1 / 1 0$ and an integer $m \geq 2 .$ . Let O contain [0, 1) and $m ^ { 2 } - 1$ mutually compatible intervals to its right with total length ϵ. Add m distinct intervals of length $\epsilon / m ^ { 2 }$ inside $[ 0 , 1 )$ , and label every request positive. These added intervals have combined length at most $\epsilon / m < 1$ , so O is the unique optimum. Recall is one, count precision is $m ^ { 2 } / ( m ^ { 2 } + m )  1$ , and weighted precision is $( 1 + \epsilon ) / ( 1 + \epsilon + \epsilon / m )  1$ . Nevertheless

$$
\frac { \eta _ { K } } { \mathrm { O P T } } = \frac { m - \epsilon / m } { 1 + \epsilon } \longrightarrow \infty .
$$

Thus even simultaneously high count and weighted precision and perfect recall do not alone make this error guarantee informative. The observed conflict costs, not only the confusion matrix, must be reported.

A predictor may output a score $s ( I ) \in [ 0 , 1 ]$ and fix its bit at arrival using $p ( I ) = \mathbf { 1 } \{ s ( I ) \geq \tau \}$ . For every threshold fixed without future test outcomes, the deterministic robustness theorem still applies to the resulting bits, and the filtered error theorem applies to their actual $\eta _ { K }$ . Calibration or threshold selection alone does not bound the conflict multipliers above.

## L. Randomized guarantees and pathwise protection

For the expected-gain comparison, allow randomized algorithms with the same online information and feasible, irrevocable rejection and revocation rules as in Section 2. Fix each complete nonempty finite labelled input σ independently of the algorithm’s random coins. The input includes the reveal order, all intervals, all labels, and its termination point; its length and future requests are not disclosed to the algorithm. Define

$$
R _ { \mathrm { e x p } } ( A ) = \operatorname* { s u p } _ { \sigma } { \frac { \mathrm { O P T } ( \sigma ) } { \mathbb { E } [ \mathrm { A L G } _ { A } ( \sigma ) ] } } , \qquad C _ { \mathrm { e x p } } ( A ) = \operatorname* { s u p } _ { \sigma : P { \mathrm { i s o p t i m a l } } } { \frac { \mathrm { O P T } ( \sigma ) } { \mathbb { E } [ \mathrm { A L G } _ { A } ( \sigma ) ] } } .
$$

A zero denominator gives ratio $+ \infty ,$ and no additive term is allowed. For deterministic algorithms, $R _ { \mathrm { e x p } }$ and $C _ { \mathrm { e x p } }$ coincide with the ratios in Definition 1.

## L.1. The exact initial-mixture curve

Proposition L.1 (An initial mixture of two existing algorithms). $L e t r _ { \mathrm { L R } } = 2 \varphi + 1$ and $0 \leq \pi < 1$ . At initialization, $M _ { \pi }$ selects NAIVE with probability π and LR with probability $1 - \pi ,$ , and then follows the selected algorithm for the whole input. Put

$$
\overline { { C } } _ { \pi } = \frac { 1 } { \pi + ( 1 - \pi ) / r _ { \mathrm { L R } } } , \qquad \overline { { R } } _ { \pi } = \frac { r _ { \mathrm { L R } } } { 1 - \pi } .
$$

Then

$$
C _ { \mathrm { e x p } } ( M _ { \pi } ) = \overline { { C } } _ { \pi } , \qquad R _ { \mathrm { e x p } } ( M _ { \pi } ) = \overline { { R } } _ { \pi } < \infty .
$$

Eliminating π gives $R = 2 \varphi C / ( C - 1 ) f o r 1 < C \le \varphi ^ { 3 }$ . This is the exact curve of this mixture family. For arbitrary original labels and any one fixed optimum O,

$$
\begin{array} { l l } { { \displaystyle { \mathbb E } [ \mathrm { A L G } _ { M _ { \pi } } ] \geq \pi \operatorname* { m a x } \{ 0 , \mathrm { O P T } - \eta _ { K } ( P , O ) \} + \frac { 1 - \pi } { r _ { \mathrm { L R } } } \mathrm { O P T } } } \\ { ~ } \\ { { \displaystyle ~ = \operatorname* { m a x } \left\{ \frac { \mathrm { O P T } } { \overline { { { R } } } _ { \pi } } , \frac { \mathrm { O P T } } { \overline { { { C } } } _ { \pi } } - \pi \eta _ { K } ( P , O ) \right\} } . }  \end{array}\tag{L.1}
$$

Here P is the original positive set and $\eta _ { K } ( P , O )$ is the error in (K.1).

Proof. The known guarantees used here are $\mathrm { A L G _ { N a i v e } } \geq \operatorname* { m a x } \{ 0 , \mathrm { O P T } - \eta _ { K } ( P , O ) \}$ and $\mathrm { A L G } _ { \mathrm { L R } } \geq \mathrm { O P T } / r _ { \mathrm { L R } }$ (Karavasilis, 2025, Theorem 3.5 and Section 4). The first also follows from Lemma K.1, since standalone NAIVE returns its permanent feasible history. For each fixed input, linearity of expectation gives

$$
\mathbb { E } [ \mathrm { A L G } _ { M _ { \pi } } ] = \pi \mathrm { A L G } _ { \mathrm { N a i v e } } + ( 1 - \pi ) \mathrm { A L G } _ { \mathrm { L R } } .
$$

Substituting the two guarantees proves the first line of (L.1). Its second line is the identity $a + \pi \operatorname* { m a x } \{ 0 , x \} = \operatorname* { m a x } \{ a , a +$ πx} for $\pi \geq 0$ , with $a = ( 1 - \pi ) \mathrm { O P T } / r _ { \mathrm { L R } }$ and $x = \mathrm { O P T } - \eta _ { K } ( P , O )$ . In particular, it remains valid for $\pi = 0$ and for $\eta _ { K } ( P , O ) > \mathrm { O P T }$ . Under perfect predictions, $P = O$ for some optimum, and NAIVE accepts every member of P and no other request. It therefore gains OPT. This yields the stated uniform consistency bound; the LR term yields the uniform robustness bound. For matching lower bounds, take finite unlabelled inputs whose LR ratios tend to its exact worst-case ratio $r _ { \mathrm { L R } }$ . Labelling one optimum positive gives Naive reward OPT without changing LR, and hence mixture ratios tending to $\overline { { C } } _ { \pi }$ . Labelling every request negative instead gives Naive reward zero and ratios tending to $\overline { { R } } _ { \pi }$ □

## L.2. A fixed-input lower bound for randomized algorithms

A different, elementary bound shows a uniform separation from consistency one at every fixed robustness budget. Define

$$
L _ { \exp } ( B ) = { \frac { 1 } { 2 \sqrt { 1 - B ^ { - 1 } } - ( 1 - B ^ { - 1 } ) } } , \qquad B \geq 4 / 3 .
$$

Proposition L.2 (A two-input randomized lower bound). Every randomized algorithm in the fixed-input expected-gain model has $R _ { \mathrm { e x p } } \geq 4 / 3 . \ I f R _ { \mathrm { e x p } } \leq B < \infty$ , then

$$
C _ { \mathrm { e x p } } \geq L _ { \mathrm { e x p } } ( B ) > 1 .
$$

Both inequalities use inputs with only three requests. In particular this lower bound is valid for an oblivious adversary.

Proof. Fix $v \geq 1$ . The following two inputs have the same labelled prefix:

$$
A = [ 0 , 1 ) , p ( A ) = 1 ; \qquad N = [ 1 / 2 , 1 / 2 + v ) , p ( N ) = 0 .
$$

The first input ends with $D = [ 1 , 1 + v )$ labelled positive. The second ends with $L = [ - 1 / 2 , 1 / 2 )$ labelled negative. All conflicts have positive overlap. Both optima have weight v + 1: the first is uniquely $\{ A , D \}$ and has perfect predictions; the second is uniquely $\{ N , L \}$ and does not. These inputs and their stopping points are fixed before the algorithm draws its coins.

Let p be the probability that A is still held after the common prefix. Coupling the two executions on the same random tape gives the same value of $p .$ If A is held, the respective final rewards are at most $v + 1$ and 1. If it is not held, no recall permits it to return, and the rewards are at most v and $v + 1$ . These bounds also cover an empty held set and voluntary revocation. Consequently

$$
\begin{array} { r } { \mathbb { E } [ \mathrm { A L G } _ { + } ] \leq v + p , \qquad \mathbb { E } [ \mathrm { A L G } _ { - } ] \leq v + 1 - v p . } \end{array}
$$

For $v = 1$ , their sum is at most 3, whereas both optima are $2 ;$ therefore robustness is at least $4 / 3$

Write $c = C _ { \mathrm { e x p } }$ and assume $R _ { \mathrm { e x p } } \leq B$ . The consistency guarantee on the first input and robustness guarantee on the second imply

$$
{ \frac { v + 1 } { c } } - v \leq p \leq { \frac { ( v + 1 ) ( 1 - B ^ { - 1 } ) } { v } } .
$$

Thus, for every $v \geq 1$

$$
\frac { 1 } { c } \leq \frac { v } { v + 1 } + \frac { 1 - B ^ { - 1 } } { v } .\tag{L.2}
$$

Set $t = \sqrt { 1 - B ^ { - 1 } } \in [ 1 / 2 , 1 )$ and $v = t / ( 1 - t ) \geq 1$ . The right side becomes $2 t - t ^ { 2 }$ , proving the claimed bound. Since $t < 1$ , this number is strictly less than one. □

The bound is tight for this two-input family, including its stopping prefixes. At the displayed $v = t / ( 1 - t )$ , accept A, retain it at N with probability t, and otherwise replace it by N. At D, take $A + D$ when A survived and obtain v otherwise. At L, retain A when it survived and take $N + L$ otherwise. The two final expected rewards are $v + t$ and $v + 1 - v t$ , giving exactly $( C _ { \mathrm { e x p } } ^ { \mathcal { F } } , R _ { \mathrm { e x p } } ^ { \mathcal { F } } ) = ( L _ { \mathrm { e x p } } ( B ) , B )$ for the restricted family $\mathcal { F }$ of these two inputs and their two distinct nonempty proper prefixes. The prefix after N has expected reward 2t and ratio $v / ( 2 t ) = 1 / ( 2 ( 1 - t ) ) \leq B \colon$ ; the first prefix has ratio one.

Corollary L.1 (An unattained randomized consistency infimum). Among randomized algorithms with finite $R _ { \mathrm { e x p } } ,$ , the infimum of $C _ { \mathrm { e x p } }$ is one and is not attained.

Proof. Every feasible reward is at most OPT. The initial mixtures have finite robustness and consistency tending to one as $\pi \uparrow 1$ , while Proposition L.2 excludes equality at every finite budget. □

## L.3. What changes under adaptive or runwise guarantees

An adaptive-offline adversary chooses each next labelled request and the stopping point from the observed past requests and retained sets, without seeing future random bits, and compares the result with the offline optimum of the realized history. Its expected robustness criterion is

$$
\mathbb { E } [ \mathrm { O P T } ( \sigma ) ] \le B \mathbb { E } [ \mathrm { A L G } ( \sigma ) ]
$$

for every such finite adversary. Here σ is random.

Proposition L.3 (The initial mixture is not adaptively robust). For every $0 < \pi < 1$ , the initial mixture $M _ { \pi }$ has infinite expected robustness against adaptive-offline adversaries.

Proof. Offer one negative interval [0, 1). Naive rejects it and LR accepts it, so the observed retained set reveals the chosen algorithm. If it was accepted, stop. Otherwise offer one more negative interval $[ 2 , 2 + H )$ and stop. Naive rejects that request as well. Thus

$$
\begin{array} { r } { \mathbb { E } [ \mathrm { A L G } ] = 1 - \pi , \qquad \mathbb { E } [ \mathrm { O P T } ] = 1 + \pi H . } \end{array}
$$

The ratio is unbounded as finite H increases. Each adversary uses at most two requests and no future coin information.

Even on a fixed one-request negative input, the mixture has zero reward with probability π. Hence its expected-gain bound cannot be interpreted as a finite per-run ratio with failure probability less than π.

## L.4. Finite-tree transfer of the lower bounds

The deterministic lower bounds do extend to randomized algorithms required to satisfy both stated ratios almost surely on every fixed input. To see this, take a finite deterministic adversary tree proving that a pair $( c , B )$ is impossible. It has finitely many terminal input sequences. If both ratios held almost surely on each fixed sequence, the intersection of those finitely many probability-one events would still have probability one. A tape in that intersection defines a deterministic causal rule that survives the entire tree, a contradiction. This argument does not extend to ratios of expected gains: distinct leaves can offset one another in expectation.

Ben-David et al.’s adaptive-offline derandomization theorem addresses a single comparison criterion. With two criteria, perfect and imperfect terminal histories carry different target factors. Adaptive robustness covers every adversary, whereas adaptive consistency covers adversaries whose every terminal history has perfect predictions. These separate requirements do not supply the mixed-terminal comparison needed to transfer the deterministic frontier to expected adaptive guarantees.

## L.5. Almost-sure robustness with expected consistency

Fix the complete input, including termination, independently of the coins. The hybrid requirements are

$$
\mathrm { P r \{ O P T ( } \sigma \mathrm { ) } \le B \mathrm { A L G } ( \sigma ) \} = 1 \quad \mathrm { f o r e v e r y ~ f i x e d ~ f i n i t e ~ l a b e l l e d ~ } \sigma ,
$$

and $\mathrm { O P T } \leq C _ { \mathrm { e x p } } \mathbb { E } [ \mathrm { A L G } ]$ on perfectly labelled inputs, without an additive term. The finite-tree transfer in Appendix L.4 requires both guarantees almost surely.

Proposition L.4 (A necessary hybrid budget bound). Every almost-surely B-robust algorithm, for finite $B > 1 ,$ , satisfies

$$
C _ { \mathrm { e x p } } \geq 2 + \frac { 1 } { B - 1 } .
$$

Proof. Put $s = 1 - 2 \delta$ . Present a negative unit interval $J = [ 0 , 1 )$ , then positive tiles partitioning $[ \delta , 1 - \delta )$ , each of length less than $1 / B$ , then a positive interval ${ \cal L } = [ \delta - t , \delta )$ . Choose $t > \delta , s + t > 1$ , and $t < s / ( B - 1 )$ . The input either ends there or continues with the positive unit interval $U = [ 1 - \delta , 2 - \delta )$ . Both complete positive sets are optimal, of weights $s + t { \mathrm { ~ a n d ~ } } s + t + 1$ respectively.

At the first tile on which the algorithm abandons $^ { J , }$ its available reward is at most that tile’s length, whereas the prefix optimum is at least one. Almost-sure robustness on each of these finitely many prefixes therefore forces it to keep J and reject all tiles. On the fixed input ending at $L ,$ accepting L would give $( s + t ) / t > B$ , so this action has probability zero. By causality the same holds on the fixed longer input. There its final reward is at most one, since all tiles and $L$ are irrecoverable and $J$ conflicts with $U$ . Hence $C _ { \mathrm { e x p } } \geq s + t + 1$ . Let $\delta \downarrow 0$ and $t \uparrow s / ( B - 1 )$ , retaining the strict inequalities. □

The bound is exact for the original single-anchor/two-terminal family when $B \geq 3$ . For $0 < t \leq 1$ , accepting L with probability p gives maximal expected rewards $1 - p + p t \mathrm { a n d } 1 + p t$ . When both actions are legal, $p = 1 / ( 1 + s )$ equalizes the two ratios at $1 + s$ . Below $t = s / ( B - 1 )$ only rejection is legal, giving $1 + s + t ;$ for $t \geq 1$ , accepting always gives a ratio at most $1 + s .$ These responses protect every prefix of this family for $B \geq 3 .$ . Taking suprema gives $2 + 1 / ( B - 1 )$ . At the original $t = \alpha$ , the balanced ratios tend to two.

Limits of mixing the stated deterministic policies. Let $\alpha = 1 / \varphi$ and $d = \varphi ^ { 2 }$ . There is a common perfect-input obstruction to initially mixing SG, LR, and the stated SUM/Credit policies. Take $0 < t < \alpha$ and $0 < \delta < \operatorname* { m i n } \{ t , 1 / 4 \}$ Start with a negative unit anchor $[ 0 , 1 )$ , then positive tiles partitioning $[ \delta , 1 - \delta )$ , each shorter than α. Next issue ${ \cal L } = [ \delta - t , \delta )$ and $U = [ 1 - \delta , 2 - \delta )$ . All these policies retain the anchor until $U ;$ after $U$ their gain is at most one. The positives tile the whole request hull, of length $2 - 2 \delta + t $ d as $t \uparrow \alpha$ and $\delta \downarrow 0$ . The same input therefore precludes consistency below $d$ for every initial mixture of these policies. The strict inequality $t <$ < α removes the complementarity present at $t = \alpha$

Proposition L.5 (Sharp guarantee for a safe action family). Consider a policy that accepts empty-conflict arrivals, accepts whenever $w ( I ) \geq 4 w ( D )$ , and otherwise may accept only when I is positive, all conflicts are negative, and $w ( I ) \geq$ $w ( D ) / 2$ . An acceptance replaces exactly $D ;$ a rejection changes nothing. Its choice within this last action set may be randomized and depend on the history. Every execution satisfies $\mathrm { O P T } \leq 1 8 \mathrm { A L G }$ . The coefficient 18 is sharp uniformly over this actionfamily, also after permanentfeasibilityfiltering.

Proof. Label ordinary and empty-conflict acceptances M, and other acceptances S. Every M node has child mass at most one fourth of its length. Every S node has only negative, hence M, children, with mass at most twice its length. Every rejection has nonempty $D$ and $w ( I ) < 4 w ( D )$ . Theorem J.1 applies with $( \beta , \lambda ) = ( 2 , 4 ) \colon q _ { S } = 6$ and all three terms defining K equal 18.

For sharpness, fix $0 < \epsilon < 1 / 8$ and offer

$$
\begin{array} { r l } { J } & { 0 \quad [ 0 , 1 ) , } \\ { X \quad 0 \quad [ 2 \epsilon - 4 , \epsilon ) , } \\ { Y \quad 0 \quad [ 1 - \epsilon , 5 - 2 \epsilon ) , } \\ { Z \quad 0 \quad [ \epsilon , 1 / 2 - \epsilon ) , } \\ { P \quad 1 \quad [ 1 / 2 - \epsilon , 1 - \epsilon ) . } \end{array}
$$

The rules force retention of J through $Z \colon$ each subsequent negative request overlaps $J$ and has length below four. The final request $P$ has length $1 / 2$ and is eligible for special acceptance at equality. On the accepting execution, $\mathrm { A L G } = 1 / 2$ The compatible requests $X , Z , P , Y$ tile the entire request hull, so $\mathrm { O P T } = 9 \mathrm { ~ - ~ }$ 4ϵ and the ratio is $1 8 - 8 \epsilon$ . The policy that accepts every special-eligible arrival realizes this execution for every $\epsilon ,$ proving sharpness as $\epsilon \downarrow 0$ . There is only one positive request, so the filter leaves the construction unchanged. □

Let $H _ { p }$ use an independent $\operatorname { B e r n o u l l i } ( p )$ coin at each eligible nonordinary positive arrival in this family. For every fixed $p >$ $0 ,$ the preceding construction accepts $\dot { P }$ with probability $p ,$ so $H _ { p }$ and its filtered form have exact almost-sure robustness 18. This uniform sharpness does not impose robustness 18 on every policy in the action family or determine its best hybrid budget. On this input, $\mathrm { O P T } / \mathbb { E } [ \mathrm { A L G } ]$ tends to $9 / ( 1 - p / 2 )$ , which is below 18 when $p < 1$

Proposition L.6 (Fixed acceptance probabilities do not beat $d ) .$ . For every constant $p \in [ 0 , 1 ]$ , the expected consistency of $H _ { p }$ is at least d, despite its pathwise robustness bound of18.

Proof. Write $q = 1 - p . \mathrm { F o r } n \geq 2 , 1 \leq r < 4$ , and small $\varepsilon > 0$ , first issue disjoint negative unit anchors

$$
J _ { i } = [ a _ { i } , a _ { i } + 1 ) , \qquad a _ { i } = i ( 1 + r - 2 \varepsilon ) , \quad 0 \leq i < n .
$$

Then issue positive tiles of length below $1 / 2$ partitioning each $[ a _ { i } + \varepsilon , a _ { i } + 1 - \varepsilon )$ , and positive intervals of length $t < 1 / 2$ crossing the two exterior anchor endpoints by $\varepsilon ,$ where $t > \varepsilon . \mathrm { A l l }$ are rejected. Finally issue, from left to right, the positive bridges

$$
I _ { i } = [ a _ { i } + 1 - \varepsilon , a _ { i + 1 } + \varepsilon ) , \qquad 0 \leq i < n - 1 .
$$

Each has length $^ r \cdot$ The positives tile the entire request hull, so their optimal value is $n + ( n - 1 ) r - 2 n \varepsilon + 2 t . \mathrm { A t } I _ { i } ,$ the untouched right anchor is still held, and the left anchor may also be held. Thus $w ( D ) \in \{ 1 , 2 \}$ , and every bridge passes the special gate and strictly fails the ordinary gate. Its acceptance is exactly its independent coin outcome. Consequently

$$
\mathbb { E } [ \mathrm { A L G } ] = ( n - 1 ) p r + 2 q + ( n - 2 ) q ^ { 2 } .
$$

Taking $\varepsilon \downarrow 0 , t \uparrow 1 / 2$ , and then $n \to \infty$ gives the consistency lower bound

$$
H _ { r } ( p ) = \frac { 1 + r } { ( 1 - p ) ^ { 2 } + p r } .
$$

Set $p _ { * } = 1 / d = ( 3 - \sqrt { 5 } ) / 2 . \mathrm { I f } p < p _ { * }$ , let $r \uparrow 4 ; $ then $H _ { r } ( p ) \to 5 / ( 1 + p ) ^ { 2 } > d .$ since $( 1 + p _ { * } ) ^ { 2 } = 5 / d . \mathrm { I f } p _ { * } \leq p \leq 1 / 2$ use $r = 1 ; 1 - p + p ^ { 2 } \leq 1 - p _ { * } + p _ { * } ^ { 2 } = 2 / d _ { * }$ , hence $H _ { 1 } ( p ) \geq d . { \mathrm { ~ A t ~ } } p = p _ { * }$ this is a supremum statement, sufficient to exclude every uniform consistency constant below d.

For $p \geq 1 / 2$ , use a different perfect input with $0 < \varepsilon < 1 / 4$ . Start with [0, 1) negative, and finish with $K = [ 1 / 4 , 3 / 4 )$ positive. Before K, issue positive tiles partitioning $[ \varepsilon , 1 / 4 )$ and $[ 3 / 4 , 1 \ - \varepsilon )$ , and the positive intervals $[ \varepsilon - t , \varepsilon )$ and $[ 1 - \varepsilon , 1 - \varepsilon + t )$ , where $\varepsilon < t < 1 / 2$ and all tiles are shorter than $1 / 2$ . These early positives are rejected. The positives together tile a hull of length $1 - 2 \varepsilon + 2 t \to 2$ . Only K invokes a coin, yielding expected gain $1 - p / 2$ . The limiting ratio is $2 / ( 1 - p / 2 ) \geq 8 / 3 > d .$ □

State-dependent probabilities and correlated decisions, which remain pathwise protected within Proposition L.5’s action set, are outside this obstruction.

## M. Same-threshold ledger ablation

The ablation removes only SUM-BANK’s special ledger acceptance test, retaining ordinary priority, length thresholds, all-negative conflicts, ledger updates and irrecoverability. It measures the gate’s total effect on the resulting trajectory.

## M.1. A five-request explanation of the ledger test

Fix $\lambda \geq \varphi ^ { 3 }$ and $0 < \varepsilon < 1 / 8$ . Reveal the following intervals in order; the subscript is the prediction:

$$
[ 0 , 1 ) _ { 0 } , \quad [ \varepsilon , 1 / 2 ) _ { 1 } , \quad [ 1 / 2 , 1 - \varepsilon ) _ { 1 } , \quad [ - 1 / \varphi + 2 \varepsilon , \varepsilon ) _ { 1 } , \quad [ 1 - \varepsilon , 1 - \varepsilon + 1 / \varphi ) _ { 1 } .
$$

The four positives tile the full input hull, giving the unique optimum $1 + 2 / \varphi - 3 \varepsilon$ . The initial negative conflicts with each of them. Both policies reject the next three requests because their lengths are below $1 / \varphi ;$ none passes the ordinary gate. Immediately before the last arrival, the held length is one and the ledger is $B = \varphi - 3 \varepsilon > 1$ . The last positive passes the special length test at equality. Without the ledger test it replaces the negative, leaving reward $1 / \varphi$ and ratio

$$
\varphi + 2 - 3 \varphi \varepsilon \longrightarrow \varphi + 2 > d .
$$

With the test, it is rejected because $B > 1 = \varphi w ( I )$ , and SUM retains reward one. Permanent filtering changes no label because the positives are compatible. Thus the ablation loses the proved consistency guarantee even though its mean effect on the workloads below is small. This construction also applies to Shaped-BANK and every Credit-BANK $T _ { c }$ in the stated family when only its special ledger test is disabled and ledger updates are retained: throughout the prefix the sole held negative has ordinary cost at least d, whereas every positive length is at most $1 / \varphi$ . Hence all ordinary tests still fail, and the same length and ledger decisions follow. The conclusion concerns only these policies with the test removed.

## M.2. Paired workload intervention

Each of the eight conditions has 100 repetitions; within a repetition, enabled and disabled SUM gates share inputs, optima, permutations and labels. Differences are percentage points of $\mathrm { A L G / O P T }$ , using the separate eight-comparison bootstrap family in Appendix O. This intervention does not cover Shaped-BANK or the low-FP rates in Table 5.

Table 4. Ledger ablation, 100 repetitions per condition. Win/tie/loss counts compare exact paired rewards; positive favors the enabled gate.
<table><tr><td>Log</td><td>Condition</td><td>Mean difference</td><td>Adjusted 99.375% CI</td><td>Win/tie/loss</td></tr><tr><td>NASA</td><td>Perfect</td><td>0.00006</td><td>[0.00000, 0.00019]</td><td>2/98/0</td></tr><tr><td>NASA</td><td>FN.1</td><td>0.00000</td><td>[0.00000, 0.00000]</td><td>0/100/0</td></tr><tr><td>NASA</td><td>FP .1</td><td>0.01227</td><td>[0.00238, 0.02129]</td><td>72/0/28</td></tr><tr><td>NASA</td><td>FP 1</td><td>0.00000</td><td>[0.00000, 0.00000]</td><td>0/100/0</td></tr><tr><td>CTC</td><td>Perfect</td><td>0.00776</td><td>[0.00297, 0.01375]</td><td>17/81/2</td></tr><tr><td>CTC</td><td>FN .1</td><td>0.00406</td><td>[0.00064, 0.00847]</td><td>10/89/1</td></tr><tr><td>CTC</td><td>FP .1</td><td>0.54094</td><td>[0.48731, 0.59718]</td><td>99/0/1</td></tr><tr><td>CTC</td><td>FP1</td><td>0.00000</td><td>[0.00000, 0.00000]</td><td>0/100/0</td></tr></table>

All-positive inputs make the special branch inactive, producing identical complete trajectories. On NASA FN .1, no disabled-gate veto occurs in these streams, also giving equality. The remaining comparisons include losses in individua repetitions despite nonnegative means.

## N. Reward and error-bound activation

Table 5 gives the reward, uncertainty and tail summaries discussed in Section 7.2. Figure 4 compares eight policies on twenty shared joint repetitions. Appendix O defines the policies, inputs and conditional uncertainty estimates.

Reward comparisons. Filtered SG loses to filtered SUM on both perfect-label cells, and to raw SG on all six nonzero low-FP cells. $\mathbf { M A X } ( \varphi ^ { 3 } )$ beats SG at FP .1, .5 and 1 on both logs; LR is strongest among the displayed methods at those rates and at FN 1. Table 8 contains a low-FP exception to this ranking. The filtered-RP means favor SG in all 20 cells, bu at CTC FP .1 the raw-RP paired interval includes zero.

Table 5. Random-order reward, percent of optimum (20 inputs per condition). Each cell is mean $^ { + u } _ { - \ell } / q . 0 5 \colon$ the pointwise 95% bootstrap interval is $[ \mathrm { m e a n - } \ell , \mathrm { m e a n } + u ] ; q . 0 5$ is the empirical individual-run fifth percentile: the sample minimum for each deterministic method, and the weighted quantile of 40 branch outcomes for each mixture. Mix<sub>C</sub>, Mix<sub>B</sub> use $\pi = { \alpha } ^ { 2 } / 2 , 1 / \varphi$ in Section 7 (expected consistency d, resp. robustness $\varphi ^ { 5 } ) ;$ means integrate the coin. Act. counts inputs with $\eta _ { K } < ( 4 - 2 \varphi ) \mathrm { O P T }$ (Appendix O). SG denotes Shaped $\mathbf { B A N K } ; \mathbf { S G } ^ { F }$ is filtered SG. These are exploratory fixed-log summaries.
<table><tr><td>Log</td><td>Labels</td><td>Act.</td><td></td><td>LR</td><td></td><td>Mixc</td><td></td><td>MixB</td><td></td></tr><tr><td>NASA</td><td>Perfect</td><td>20/20</td><td> $9 6 . 8 3 _ { - . 1 3 } ^ { + . 1 4 }$ </td><td>/96.25</td><td> $9 7 . 4 3 _ { - . 1 0 } ^ { + . 1 2 }$ </td><td>/96.47</td><td> $9 8 . 7 9 _ { - . 0 5 } ^ { + . 0 5 }$ </td><td> $/ 9 6 . 5 1 $ </td><td> $9 9 . 1 0 _ { - . 0 9 } ^ { + . 0 9 }$  /98.76</td></tr><tr><td>NASA</td><td>FP 0.01923</td><td>20/20</td><td> $9 6 . 8 3 _ { - . 1 3 } ^ { + . 1 4 }$ </td><td>/96.25</td><td> $9 4 . 9 5 _ { - . 1 5 } ^ { + . 1 4 }$ </td><td>/85.85</td><td> $9 0 . 7 5 _ { - . 3 7 } ^ { + . 3 6 }$  1</td><td> $/ 8 5 . 4 1 $   $9 7 . 0 7 _ { - . 1 0 } ^ { + . 1 1 }$ </td><td>/96.61</td></tr><tr><td>NASA</td><td>FP 0.1</td><td>0/20</td><td> $9 6 . 8 3 _ { - . 1 3 } ^ { + . 1 4 }$ </td><td>/96.25</td><td> $9 1 . 6 6 _ { - \ 1 8 } ^ { + . 2 0 }$ </td><td> $^ { \prime } 6 9 . 2 8 $ </td><td> $8 0 . 1 1 _ { - . 4 4 } ^ { + . 4 4 }$  1</td><td> $^ \prime 6 7 . 5 5$   $9 4 . 3 6 _ { - . 2 0 } ^ { + . 2 0 }$ </td><td>/93.60</td></tr><tr><td>NASA</td><td>FP1</td><td>0/20</td><td> $9 6 . 8 3 _ { - . 1 3 } ^ { + . 1 4 }$ </td><td>/96.25</td><td> $9 0 . 2 1 _ { - . 1 1 } ^ { + . 1 2 } / 6 1 . 3 6$ </td><td></td><td> $7 5 . 4 2 _ { - . 2 4 } ^ { + . \dot { 2 } \bar { 4 } }$  24</td><td> $/ 6 1 . 0 0$   $8 9 . 1 2 _ { - . 2 8 } ^ { + . 2 7 }$ </td><td>/87.84</td></tr><tr><td>CTC</td><td>Perfect</td><td>20/20</td><td> $9 3 . 2 7 _ { - . 1 1 } ^ { + . 1 0 }$ </td><td>/92.77</td><td> $9 4 . 5 5 _ { - . 0 8 } ^ { + : \bar { 0 } \bar { 8 } } / 9 2 . 8 6$ </td><td></td><td> $9 7 . 4 3 _ { \frac { - } { . } . 0 4 } ^ { + . 0 4 }$ </td><td> $/ 9 2 . 9 9 $   $9 5 . 5 8 _ { - . 1 4 } ^ { + . 1 4 }$ </td><td>/94.83</td></tr><tr><td>CTC</td><td>FP 0.0034</td><td>20/20</td><td> $9 3 . 2 7 _ { - . 1 1 } ^ { + . 1 0 } \ : / \ : 9 2 . 7 7$ </td><td></td><td> $9 1 . 7 4 _ { - . 1 5 } ^ { + . 1 5 } / 8 4 . 2 6$ </td><td></td><td> $8 8 . 3 3 _ { - . 3 9 } ^ { + . 4 2 } / 8 3 . 2 2$ </td><td></td><td> $9 3 . 8 0 _ { - . 1 7 } ^ { + . 1 6 } / 9 2 . 9 5$ </td></tr><tr><td>CTC</td><td>FP 0.1</td><td>0/20</td><td> $9 3 . 2 7 _ { - . 1 1 } ^ { + . 1 0 } / 9 2 . 7 7$ </td><td></td><td> $8 4 . 8 1 _ { - . 1 2 } ^ { + . 1 2 } / 4 8 . 4 9$ </td><td></td><td> $6 5 . 8 9 _ { - . 2 8 } ^ { + . 2 8 } / 4 7 . 5 8$ </td><td> $8 9 . 2 5 _ { - . 1 4 } ^ { + . 1 4 }$ </td><td>/88.72</td></tr><tr><td>CTC</td><td>FP1</td><td>0/20</td><td> $9 3 . 2 7 _ { - . 1 1 } ^ { + . 1 0 } / 9 2 . 7 7$ </td><td></td><td> $8 7 . 7 2 _ { - . 1 0 } ^ { + . 1 1 } / 6 3 . 8 7$ </td><td></td><td> $7 5 . 3 2 _ { - . 2 2 } ^ { + . 2 2 } / 6 3 . 4 3$ </td><td></td><td> $8 6 . 3 9 _ { - . 1 6 } ^ { + . 1 6 } / 8 5 . 6 8$ </td></tr></table>

Table 6. Mean retained reward (% of OPT; twenty shared repetitions per cell). Shaped-BANK (SG) and RP at $\lambda = ( 5 \varphi + 1 ) / 2$ share exact robustness $5 \varphi + 3 .$ . Superscript F denotes permanent filtering. The best scalar Credit uses $c _ { * } = 2 \varphi - 3$ and budget $4 \varphi + 5 ;$ SUM has budget $\mathsf { 7 } \varphi + 4 ; \mathbf { M A X } ( \varphi ^ { 3 } )$ is a descriptive comparator. Credit at $c = \varphi / 2$ is included in the full comparison tables.
<table><tr><td>Log</td><td>Labels</td><td> $\mathrm { S G } ^ { F }$ </td><td>SG</td><td> $\mathrm { S c a l a r } ^ { F }$ </td><td> $\mathrm { S U M } ^ { F }$ </td><td> $\mathbf { M A X } ( \varphi ^ { 3 } )$ </td><td>LR</td><td>Half(4)</td><td> ${ \mathrm { R P } } ^ { F }$ </td></tr><tr><td>NASA</td><td>Perfect</td><td>99.10</td><td>99.10</td><td>99.10</td><td>99.17</td><td>98.93</td><td>96.83</td><td>99.26</td><td>98.67</td></tr><tr><td>NASA</td><td>FP .1</td><td>94.36</td><td>94.92</td><td>94.35</td><td>92.26</td><td>96.20</td><td>96.83</td><td>93.15</td><td>91.76</td></tr><tr><td>NASA</td><td>FP1</td><td>89.12</td><td>88.63</td><td>89.08</td><td>88.56</td><td>93.07</td><td>96.83</td><td>89.01</td><td>88.17</td></tr><tr><td>CTC</td><td>Perfect</td><td>95.58</td><td>95.58</td><td>95.58</td><td>95.91</td><td>95.51</td><td>93.27</td><td>97.22</td><td>92.80</td></tr><tr><td>CTC</td><td>FP .1</td><td>89.25</td><td>89.68</td><td>89.24</td><td>87.34</td><td>90.63</td><td>93.27</td><td>86.84</td><td>87.24</td></tr><tr><td>CTC</td><td>FP 1</td><td>86.39</td><td>85.72</td><td>86.37</td><td>85.73</td><td>89.15</td><td>93.27</td><td>86.12</td><td>85.37</td></tr></table>

Figure 5 shows activation counts 20,7,0 on NASA and 20,5,0 on CTC at the three nonzero low-FP rates; both perfect cells activate 20/20. In contrast, the main FP grid has nonpositive error branches.

Scalar Credit at $c = \varphi / 2$ has a different low-FP threshold and middle-rate counts 9/20 and 11/20. SG’s larger guaranteed robust floor can reduce the error-branch activation count while strengthening the worst-case guarantee.

Best scalar control. Filtered SG minus filtered best scalar Credit (c<sub>∗</sub>; Appendix O.5) has fifteen positive means, four exact zero means and one negative mean; the range is $\left[ - 0 . 0 0 0 8 5 , 0 . 0 3 9 2 7 \right]$ percentage points. At NASA FP.05769, SG minus scalar is −0.00085 with pointwise interval [−0.00946, 0.00592] and 2/17/1 wins/ties/losses. Raw differences have fourteen positive and six zero means, with maximum 0.16959 points at CTC FP.5 (interval [0.13983, 0.20048]). The workload-reward differences are small despite the strict improvement in the worst-case guarantee.

## N.1. Uncertainty and individual-run rewards

Table 5 uses the twenty matched inputs and pointwise bootstrap protocol of Appendix O. Each input fixes the requests, order, labels and termination; mixture coins introduce no additional input replications.

The mixture mean integrates the initial policy coin. Its empirical fifth percentile instead uses the forty branch outcomes with masses π/20 and $( 1 - \pi ) / 2 0$ . For a fixed input, both branch probabilities exceed .05, so the conditional fifth percentile is min $\{ N _ { r } , L _ { r } \}$ . Averaging these conditional quantities is not the pooled quantile in Table 5. The latter can differ between $\operatorname { M i x } _ { C }$ and $\mathbf { M i x } _ { B }$ even though their conditional minima coincide.

Explicitly, with rewards $N _ { r } , L _ { r }$ as percentages of optimum,

$$
\widehat { F } _ { \pi } ( t ) = \frac { 1 } { 2 0 } \sum _ { r = 1 } ^ { 2 0 } [ \pi \mathbf { 1 } _ { N _ { r } \leq t } + ( 1 - \pi ) \mathbf { 1 } _ { L _ { r } \leq t } ] , \qquad q _ { . 0 5 } = \operatorname* { i n f } \{ t : \widehat { F } _ { \pi } ( t ) \geq . 0 5 \} .
$$

![](images/102424dba23aa1c58c56666dd732fcf6a3b93145f173a86d5658c9532fc70767.jpg)  
Figure 4. Main-regime rewards: NASA/CTC rows and FN/FP columns. The eight methods are raw/filtered SG, filtered best scalar Credit, filtered SUM, $\mathbf { M A X } ( \varphi ^ { 3 } )$ , LR, Half(4), and filtered RP at the SG robustness budget. Points are twenty-repetition means with pointwise 95% bootstrap intervals (10,000 shared resamples). Perfect labels appear in both columns; FN and FP flip different classes. The FP perfect-to-0.1 segment is omitted because the low-FP regime lies between them. Methods have different guarantee budgets; MAX is descriptive. Raw/filtered SG coincide in the FN panels, while filtered SG and scalar Credit nearly overlap at this scale.

![](images/571d834338cdc1c83111d5022628760ddde191626f114e7a435f6379c7c6a933.jpg)  
Figure 5. Bounded low-FP study. Top: mean original-label $\eta _ { K } / \mathrm { O P T }$ with pointwise bootstrap intervals; dashed line: $\mathbf { S } \mathbf { G } ^ { \prime } \mathbf { s }$ exact activation threshold $4 - 2 \varphi$ . Bottom: actual activation counts out of twenty with pointwise Wilson intervals. Dotted vertical references mark the analytic expected-error crossings, .03700 and .00655; the measured grid was fixed around the Credit crossings. Rates share nested labels and reveal orders. Mean-error crossing and realization-level activation are distinct quantities.

## O. Shared experimental protocol and provenance

## O.1. Instances, policies and comparison budgets

Raw/filtered SG and raw/filtered RP at $\lambda = ( 5 \varphi + 1 )$ /2 share exact robustness $5 \varphi + 3 .$ Each uses two fixed logs, seven main conditions (perfect labels; FN or FP at $. 1 , . 5 , 1 )$ , and three additional low-FP rates per log, with twenty joint repetitions, giving 400 cases. The low-FP rates are .01923, .03846, .05769 on NASA and .0034, .0068, .0102 on CTC.

Thirteen main-regime controls share those repetitions: raw/filtered Credit $( c = \varphi / 2 )$ , raw/filtered SUM $( \lambda = \varphi ^ { 3 } )$ , MAX at $\varphi ^ { 3 }$ and Λ, LR, Half(4), and five RP settings. These RP settings are $\varphi ^ { 3 } , ( 7 \varphi + 2 ) / 2 , \lambda _ { 0 } = ( B _ { 0 } + \sqrt { B _ { 0 } ( B _ { 0 } - 8 ) } ) / 4$ with $\begin{array} { r } { B _ { 0 } = 8 \varphi + 5 . } \end{array}$ , and raw/filtered $( 6 \varphi + 1 ) / 2$ . The last pair matches Credit’s budget $6 \varphi + 3 ; ( 7 \varphi + 2 ) / 2$ matches SUM’s $7 \varphi + 4 .$ RP at $\lambda _ { 0 }$ has exact robustness $2 \lambda _ { 0 } + 2 .$ , not $B _ { 0 }$ . Here $\Lambda = ( 5 \varphi + 2 + \sqrt { 3 3 \varphi + 2 1 } ) / 2 \simeq 9 . 3 5 7 7 1 5 ;$ MAX is a descriptive comparator with no claimed guarantee.

The separate SUM ledger ablation (Appendix M) uses 100 repetitions; its paired differences are computed within that study.

The logs are NASA-iPSC-1993-3.1-cln (Parallel Workloads Archive, n.d.b) and CTC-SP2-1996-3.1-cln (Parallel Workloads Archive, n.d.a) from the Parallel Workloads Archive (Feitelson et al., 2014). Mirrored inputs match later officia downloads; earlier download logs and original cleaning source are unavailable. Raw SWF files are not redistributed because unrestricted redistribution is not established. Mapping follows SWF (Parallel Workloads Archive, n.d.c).

Each retained record gives $[ s , s + t )$ with weight t > 0 seconds. NASA uses SWF column 2 as a start-time surrogate, without adding its unknown wait −1; CTC uses $s = { \mathrm { s u b m i t } } + { \mathrm { w a i t } }$ with known nonnegative wait. Cleaning requires eighteen columns, valid required integer fields, nonnegative start surrogate/submission and positive runtime; neither status nor processor count is a filter. All exclusions are zero-runtime records. Stable IDs are physical SWF line numbers; coincident intervals remain distinct.

Table 7. Fixed workload-derived instances. Exact objectives are seconds.
<table><tr><td>Quantity</td><td>NASA</td><td>CTC</td></tr><tr><td>Data records</td><td>18,239</td><td>77,222</td></tr><tr><td>Excluded zero-runtime records</td><td>173</td><td>17</td></tr><tr><td>Retained intervals</td><td>18,066</td><td>77,205</td></tr><tr><td>Exact optimum V</td><td>5,816,181</td><td>28,353,920</td></tr><tr><td>Fixed optimal-set size (FN eligible)</td><td>3,635</td><td>1,464</td></tr><tr><td>Complement size (FP eligible)</td><td>14,431</td><td>75,741</td></tr></table>

Integer weighted-interval dynamic programming sorts by (finish, start, id) and compute $V _ { j } = \operatorname* { m a x } \{ V _ { j - 1 } , w _ { j } + V _ { k ( j ) } \}$ where $k ( j )$ counts earlier requests finishing no later than request $j$ starts. A tie excludes the current request, fixing an optimum O independently of reveal order and labels. Endpoint contact is compatible. Only offline optimization and label construction use the full instance. Online methods see immutable labels one request at a time, without termination knowledge, recall or geometric expiration. This single-resource, random-reveal reduction does not reproduce queues or processor allocation.

## O.2. Randomization and uncertainty

For each log and repetition $r = 0 , \ldots , 1 9$ , PCG64 streams supply one reveal permutation and one uniform per stable ID. FN flips members of O to zero; FP flips its complement to one. An eligible ID flips when its uniform is below the binary64 conversion of the stored rational rate q. Rates therefore share nested labels; methods and error types share streams, so cells are not independent samples. Perfect FN/FP labels coincide. FN positives are feasible; FP positives need not be.

Rewards are 100 ALG/V. The bootstrap draws 10,000 resamples of the twenty whole repetitions per log, shared across methods and conditions. Pointwise 95% percentile intervals use linear interpolation at .025 and .975; paired intervals resample within-repetition differences in percentage points. The 60 primary contrasts compare filtered SG with raw SG and raw/filtered matched RP in twenty cells. The 34 secondary contrasts compare filtered SG with filtered $c = \varphi / 2$ Credit in twenty cells and filtered SUM in fourteen main cells. Forty further contrasts compare raw/filtered SG with their best scalar counterparts. The ancillary tables (see anc/README.md) give full summaries for all twenty cells and all 134 contrasts, including adverse and zero results. These bootstrap intervals and the Wilson activation intervals are pointwise

and conditional on the two fixed logs.

The SUM ablation uses $1 0 , 0 0 0 \times 1 0 0$ whole-repetition resamples and quantiles .003125/.996875 for its eight-comparison family, separately from the pointwise SG comparisons. Without public preregistration, the ablation conditions, low-FP rates and c were fixed before their outcomes; the eight-policy display before scalar outcomes; and the models, thresholds and independent-cohort comparisons in Appendix Q before test rewards.

## O.3. Original-label error and activation

For original positives P and fixed optimum $O ,$ compute exactly

$$
\eta _ { K } = w ( O \setminus P ) + \sum _ { F \in P \setminus O } \left[ w ( N _ { O } ( F ) ) - w ( F ) \right] .
$$

Original labels define this statistic; immutable effective labels govern filtered decisions and fees. SG’s error branch exceeds its robust floor precisely when

$$
\eta _ { K } / V < 1 - d / ( 5 \varphi + 3 ) = 4 - 2 \varphi \simeq . 7 6 3 9 3 2 .
$$

For marginal FP probability $q , \mathbb { E } [ \eta _ { K } / V ] \ = \ q S ,$ with $S ~ = ~ 1 2 0 0 7 5 2 8 0 / 5 8 1 6 1 8 1 ~ \simeq ~ 2 0 . 6 4 5 0$ on NASA and $S ~ = ~ 3 3 0 9 2 8 3 1 7 2 / 2 8 3 5 3 9 2 0 ~ \simeq ~ 1 1 6 . 7 1 3 4 ~ \mathrm { o n } ~ \mathrm { C T C }$ . Credit’s threshold $( 4 - \varphi ) / 3 \simeq . 7 9 3 9 8 9$ gives mean crossings .038459/.006803; SG gives .0370031776/.0065453654. The added grid surrounds both crossings below the original minimum nonzero rate .1.

## O.4. Policy definitions and implementation

With $d = \varphi + 1 , \alpha = \varphi - 1 , M = 2 \varphi + 1 , \mathrm { S G }$ uses negative cost min $\{ 2 \varphi v , \operatorname* { m a x } \{ d v , \varphi v + \alpha b \} \}$ and positive cost $M v$ capped per holder before summation. Breakpoints $b = \varphi v$ , dv and rational/golden-field comparisons are exact. Ordinary acceptance has priority; non-strict boundaries accept equality. Rejection fees use old values, and replacement inherits whole ledgers. Permanent positive history records a compatible effective positive even if subsequently rejected or revoked.

Credit at $c = \varphi / 2$ uses $v + ( v + \operatorname* { m i n } \{ v , b / 2 \} ) \varphi$ for a negative holder. SUM’s ordinary test uses total conflicting length; MAX changes only this ordinary quantity to maximum conflicting length, retaining SUM’s special gates and ledger updates. LR follows Algorithm 3 of Karavasilis (2025): empty acceptance or strict $x > \varphi y$ , where y is maximum conflicting length. RP follows Algorithm 4: after ordinary acceptance fails, a positive arrival with all-negative conflicts must satisfy $w ( I ) \geq W$ Half(4) changes this last test to $2 w ( I ) \geq W$ and uses ordinary parameter 4. It is an author-chosen empirical control; no RP certificate or optimal-parameter claim transfers to it. These controls use the definitions above, rather than reproducing published workload curves.

An independent full-scan implementation agrees on all SG decisions and rewards; labels, errors and bootstrap summaries were separately reconstructed.

Arithmetic resources. For fixed parameters of the displayed SUM, Shaped and $T _ { a }$ costs, a full-scan implementation uses $O ( n ^ { 2 } )$ exact arithmetic operations and comparisons, and $O ( n )$ stored scalars and interval records over n arrivals. At time t, scan at most $t - 1$ holders to form conflicts, sum lengths, costs and ledgers, then update fees or replace records. Each displayed cost takes constant arithmetic work; fee denominators are positive on their branches. The permanent positive filter adds one scan and at most t records. Source measures and replacement forests are proof objects, not stored policy state. Parameters are exact arithmetic constants; rational inputs and rational a use rational operations, and the golden endpoint uses a fixed quadratic field. This bounds scalar counts, not their bit sizes or floating-point error; OPT computation and retained diagnostic traces are excluded.

## O.5. Strongest-scalar control

The scalar $c _ { * } = \alpha ^ { 3 } = 2 \varphi - 3$ is the best consistent scalar endpoint, with exact robustness $4 \varphi + 5$ and cost min $\{ M v , d v +$ $c _ { * } b \}$ . Raw/filtered scalar Credit uses the same 400 cases and bootstrap resamples as SG. Appendix N summarizes its forty contrasts; $c = \varphi / 2$ remains in the full tables.

## P. Experimental diagnostics: randomization, learned labels, and application limits

A randomized control with its coin integrated exactly. Naive accepts a positively predicted request only when it is compatible with every previously accepted request, and never revokes one. The initial mixture selects Naive with probability π and LR otherwise, once before the input is processed. For each fixed input and prediction vector we report

$$
1 0 0 \{ \pi \mathrm { N a i v e } + ( 1 - \pi ) \mathrm { L R } \} / \mathrm { O P T } .
$$

We use $\pi _ { C } = \alpha ^ { 2 } / 2$ and $\pi _ { B } = 1 / \varphi _ { ; }$ , with expected guarantees $( \bar { C } , \bar { R } ) \leq ( d , 2 d )$ and $( \varphi ^ { 5 } / ( 3 \varphi + 3 ) , \varphi ^ { 5 } )$ , respectively. Inputs are fixed independently of the coin; these are expected-reward guarantees, not finite per-execution robustness bounds.

In the perfect/FP .1/FP 1 rows of Table 5, the underlying Naive means are 100.00/69.78/62.18% on NASA and 100.00/48.97/64.23% on CTC. The budget-matched mixture beats filtered SG on perfect CTC labels, but loses at FP .1 on both logs; the consistency-matched mixture beats filtered SG at FP 1 on both logs.

Finite binary64 decision check. With integer geometry/rewards and binary64 thresholds/ledgers, literal min/max evaluation agrees with exact breakpoint evaluation on branches, labels, final IDs and rewards for 800 SG streams (400 cases; 38,108,400 arrivals). This finite check gives no guarantee for adversarial near-boundary inputs or accumulated ledger error.

A CTC submission-order diagnostic. We keep CTC’s interval geometry [submit + wait, submit + wait + runtime), fixed optimum and uniform arrays, but reveal requests in ascending submission time, breaking ties by physical-line ID. Perfect labels give one deterministic case; FP probabilities .0068 and .1 each use the twenty label streams. Both SG variants use exact arithmetic. This changes reveal order alone; the application interpretation is examined in Appendix P.2.

Table 8. CTC submission-order sensitivity. Mean percentages of the fixed optimum; $n = 1$ for perfect labels and $n = 2 0$ for each nonzero FP rate. The initial coin is integrated for both mixture columns.
<table><tr><td>Condition</td><td>Naive</td><td>LR</td><td>Mixc</td><td> $\mathrm { M i x } _ { B }$ </td><td>SG</td><td> $\mathrm { S G } ^ { F }$ </td></tr><tr><td>Perfect</td><td>100.00</td><td>94.98</td><td>95.94</td><td>98.08</td><td>95.61</td><td>95.61</td></tr><tr><td>FP .0068</td><td>95.05</td><td>94.98</td><td>94.99</td><td>95.02</td><td>95.30</td><td>95.28</td></tr><tr><td>FP .1</td><td>78.11</td><td>94.98</td><td>91.76</td><td>84.55</td><td>94.10</td><td>93.57</td></tr></table>

At FP .1, submission order raises filtered SG’s mean from 89.25% to 93.57%, and the budget-matched mixture’s from 65.89% to 84.55%. LR remains better than filtered SG. At FP .0068, filtering costs only 0.020 percentage points, compared with 1.276 under the random-reveal protocol. The perfect-label mixture still exceeds SG. The magnitude of these comparisons thus depends strongly on reveal order.

## P.1. Learned labels, structured errors, and a planted obstruction

To test errors beyond independent flips, we construct separate instances from the logs. Requests are ordered by $( t _ { i } , \mathrm { I D } _ { i } )$ where $t _ { i }$ is SWF column 2 and ID is the physical-line identifier. The 60/20/20 split starts at b.6nc, b.8nc and advances each cut past equal timestamps. Training/validation/test counts are 10,839/3,613/3,614 for NASA and 46,323/15,441/15,441 for CTC. Consecutive blocks of 512 requests form separate instances; a final block shorter than 32 is appended to its predecessor. Each block has its own canonical offline optimum, so conflicts across blocks are omitted. Its optimum uses the tie-breaking DP of Appendix O. Block boundaries, unlike split boundaries, may separate equal timestamps. There are 7 NASA and 31 CTC test blocks. These blocks are not assumed to be independent samples from a workload population.

A concrete prediction pipeline. For each log, an ℓ -regularized logistic model learns membership in the training blocks’ canonical optima. Its seven causal features are log-duration, log-lead-time, log-gap since the preceding reveal, log-count and log-total-length of previously revealed conflicts, and two time-of-day coordinates. Logarithms use log(1 + x); lead time is $s _ { i } - t _ { i } .$ . History starts empty in every block, including at split boundaries; the first gap is zero. Earlier equal-time requests enter in ID order. Conflict features include all earlier overlapping reveals in the block, regardless of labels or acceptance, and their full lengths are summed. The periodic features are sine and cosine of $2 \pi ( t _ { i }$ mod 86400)/86400, with no calendar offset. Training-only unweighted means and population standard deviations scale all splits, with zero deviations replaced by one. The constructed interval inputs reveal runtime at arrival; NASA’s column 2 is a start-time surrogate. See Appendix P.2 for the application boundary.

Table 9. Held-out learned-label checks. Precision/recall pool request counts; weighted recall and $\eta _ { K } / \mathrm { O P T }$ pool lengths. Reward rows are means of block percentages. The selected threshold maximizes validation F1, not the online objective.
<table><tr><td>Quantity</td><td>NASA</td><td>CTC</td></tr><tr><td>Precision / recall</td><td>.851/.660</td><td>.428/.426</td></tr><tr><td>Weighted recall  $\eta _ { K } \mathrm { / O P T }$ </td><td>.610 .577</td><td>.364 1.574</td></tr><tr><td>Active error branch</td><td>6/7</td><td>0/31</td></tr><tr><td>Shaped / filtered</td><td>94.58/94.58</td><td>89.81/89.66</td></tr><tr><td>SUM / MAX at  $\varphi ^ { 3 }$ </td><td>95.17/95.17</td><td>89.43/89.95</td></tr><tr><td>LR / Naive</td><td>96.86/61.43</td><td>91.25/62.68</td></tr><tr><td>Consistency-matched mixture</td><td>90.09</td><td>85.79</td></tr><tr><td>Budget-matched mixture</td><td>74.96</td><td>73.59</td></tr><tr><td>Filtered Shaped, all-negative</td><td>96.96</td><td>90.30</td></tr><tr><td>Naive, all-positive</td><td>90.45</td><td></td></tr><tr><td></td><td></td><td>83.51</td></tr></table>

Writing $h _ { i } = \beta _ { 0 } { + } \beta ^ { \top } z _ { i }$ for standardized features, training minimizes the sum of logistic losses $\begin{array} { r } { \sum _ { i } [ \log ( 1 + \exp h _ { i } ) - y _ { i } h _ { i } ] + } \end{array}$ $\| \beta \| _ { 2 } ^ { 2 } / 2 ;$ the intercept is unpenalized. A deterministic Newton solver fits the model. Labels are $1 \{ p _ { i } \geq \tau \}$ . Validation pooled F1 chooses $\tau \in \{ . 0 5 , . 1 0 , . . . , . 9 5 \}$ , breaking ties toward the largest threshold and giving .35 and .20; neither test data nor online reward is used for tuning. Fixed .5, explicit all-negative/all-positive and perfect-label controls are also reported.

Both constant controls outperform their corresponding learned-label policies in Table 9. Fixed threshold .5 gives 94.88/89.90% for filtered Shaped, and perfect labels give 97.61/96.58%. Validation F1 need not select useful advice for the online objective or determine conflict-weighted error.

Matched counts with different error structure. On each test block we create FP errors at target .01 or FN errors at target .1, rounding the eligible count to the nearest integer. Uniform placement, length-proportional sampling without replacement, and a cyclic consecutive burst in eligible reveal order have exactly the same error count within a block. They do not match error weight or per-request marginals. Each block and placement uses one fixed seed; the comparisons are descriptive. With 85 total NASA false negatives, uniform versus length-weighted placement changes mean $\eta _ { K } / \mathrm { O P T }$ from .081 to .526, Naive’s reward from 91.85 to 47.41%, and Shaped’s from 97.92 to 96.71%. For CTC the corresponding numbers are .108 versus .272, 89.24 versus 72.80%, and 94.96 versus 92.86%. Bursts and FP placements give different rankings; the ancillary tables (see anc/README.md) list all 228 cases, including unfavorable comparisons.

Embedding the zero-error construction. We instantiate the distinct-geometry family with $\delta = 1 / 1 0 0$ , three central groups, eight alternatives per group, and rational boundary parameter $1 6 / 5 < 2 \varphi .$ . Its 27 requests have $\mathrm { O P T } = 1 4 7 / 2 0$ raw Shaped reward 1, and $\eta _ { K } = 0$ . A spatially disjoint scaled copy is embedded into the first test block of each log, whose background labels are perfect. The witness supplies a fraction $a \in \{ 0 , 1 / 4 , 1 / 2 , 9 / 1 0 , 1 \}$ of total optimum; its requests retain their adversarial order. At $a = 1 / 2 ,$ , raw/filtered Shaped give 56.46/93.05% on NASA and 55.12/91.72% on CTC. At $a = 1$ , raw Shaped, SUM and MAX give $2 0 / 1 4 7$ , filtered Shaped and LR give 638/735, and Naive is optimal. The error remains zero, although the positive set is infeasible. This explains the filter’s role on this proved construction.

Decisions and ledgers use exact rational/golden arithmetic; mixture coins are integrated analytically.

## P.2. Decision-aligned labels and the application boundary

We keep the split, seven features and linear-logistic capacity of Appendix P.1, but align training and threshold selection with decision costs. This secondary design was chosen after test-block inspection. It uses weighted logistic training, motivated by prediction versus decision loss (Elmachtoub & Grigas, 2020), rather than SPO+ or its guarantee.

A cost target tied to the error. For a training block with optimum O of value V , assign cost

$$
c _ { i } = \left\{ \begin{array} { l l } { w _ { i } / V , ~ } & { i \in O , } \\ { \left( w ( N _ { O } ( i ) ) - w _ { i } \right) / V , } & { i \notin O . } \end{array} \right.
$$

Table 10. Secondary objective-alignment diagnostic. Entries are mean block percentages for filtered SG on the same 7 NASA and 31 CTC test blocks. Reward selection uses only validation blocks. Threshold 1 gives all-negative labels on these scores.
<table><tr><td>Condition</td><td>NASA</td><td>CTC</td></tr><tr><td>Ordinary score, F1-selected</td><td>94.58</td><td>89.66</td></tr><tr><td>Ordinary score, error-selected</td><td>93.71</td><td>90.30</td></tr><tr><td>Ordinary score, reward-selected</td><td>96.96</td><td>90.30</td></tr><tr><td>Cost score, F1-selected</td><td>93.13</td><td>90.21</td></tr><tr><td>Cost score, error-selected</td><td>94.91</td><td>90.30</td></tr><tr><td>Cost score, reward-selected</td><td>96.96</td><td>90.48</td></tr><tr><td>All-negative</td><td>96.96</td><td>90.30</td></tr><tr><td>Perfect labels</td><td>97.61</td><td>96.58</td></tr><tr><td>LR (ignores labels)</td><td>96.86</td><td>91.25</td></tr></table>

Then $\textstyle \sum _ { i } c _ { i } \mathbf { 1 } \{ b _ { i } \neq \mathbf { 1 } _ { i \in O } \} = \eta _ { K } ( b , O ) / V$ . In addition to the ordinary-logistic score, we fit a score using $c _ { i } .$ -weighted binary log loss, globally rescaling training costs to mean one and retaining the same L2 coefficient 1 on the summed loss. Thus the loss weights are $m c _ { i } / \sum _ { j } c _ { j }$ for m training requests. Zero-cost negatives have zero loss weight but remain in the unweighted training standardization. Only the cost model is fitted anew, using the same solver and an additional final gradient infinity-norm check below $1 0 ^ { - 5 }$ . Such a cost score is not an ordinary calibrated membership probability. Neither does minimizing η maximize retained reward.

Each score uses the 21 validation thresholds $0 , . 0 5 , \ldots , 1$ ; the endpoints give all-positive and all-negative labels on the evaluated inputs. Explicit constant controls are retained. Selection maximizes pooled F1, minimizes mean block $\eta _ { K } / V$ or maximizes mean block filtered-SG reward; ties choose the largest threshold. Together with fixed .5 thresholds and negative/positive/ perfect controls, this gives 11 conditions for each of the eight policies.

Reward selection chooses all-negative advice for both ordinary scores and the NASA cost score. Their improvement over F1 therefore cannot be credited to learning. The CTC cost score selects .30 and gains .1802 percentage points over the constant: 3 blocks improve, 27 tie and 1 worsens. The leave-one-block-out mean range is [.1018, .1991] points, a sensitivity range, not a confidence interval. Its mean remains .7690 points below LR. The ancillary tables (see anc/README.md) list all twelve paired contrasts, including adverse results.

The error-oriented NASA score reduces pooled $\eta _ { K } / V$ from .5767 to .4838 and makes the error branch active in all seven blocks rather than six, yet gives lower reward than the all-negative control, whose error branch is inactive. Error informativeness and empirical utility are thus distinct diagnostics. The CTC reward-selected cost score reduces pooled error from 1.5735 to .9501, activates the error branch in only 4 of 31 blocks and produces four special admissions. These results support objective sensitivity, not a claim that objective mismatch is the sole reason for the ordinary predictor’s failure.

Feasibility and boundary context. Permanently projecting each positive through the existing causal Naive filter gives feasible immutable labels. Raw SG on those labels exactly equals filtered SG on the original labels in every evaluated case, verifying the wrapper identity. Feasibility does not resolve future membership ambiguity: the first offer [0, 1) has identical observed features whether the input stops or later reveals [0, 2), although its optimal membership differs. An immutable feasible positive history cannot retain both overlapping offers.

We also concatenate each test tail, recomputing its optimum and causal features without internal resets while keeping all learned coefficients, standardization and validation thresholds fixed. History still starts empty, with first gap zero, at the test boundary. NASA’s sum of block optima is 1,054,070, compared with continuous optimum 1,032,493; CTC’s corresponding values are 16,376,035 and 6,390,734. There are 357/152,107 cross-block overlap pairs and 64/1,041 canonical label changes. Hence the blocks are substantively different constructed problems, particularly for CTC. Continuous-tail reward-selected cost SG gives 97.16/94.03%, versus 97.26/94.95% for LR; the CTC margin over constant-negative SG is only .0182 points. This jointly changes feature context, labels, geometry and normalization, so it does not isolate a boundary effect.

What these inputs mean for an application. Prefix-invariant features are causal only given known interval endpoints. SWF distinguishes realized runtime from requested time (Parallel Workloads Archive, n.d.c). NASA substitutes start times for unavailable submissions (Parallel Workloads Archive, n.d.b); CTC’s actual waits and runtimes are retrospective observations (Parallel Workloads Archive, n.d.a). Under the chronological interpretation, ordinary-F1 filtered SG revokes 58/58 previously started NASA intervals and 20/263 previously started CTC intervals. These are simulated revocations in historical traces. A real reservation interface can specify start and duration while disallowing deletion during running jobs (SchedMD, n.d.). Thus a sufficient model interpretation is a provisional planning batch completed before any service starts; interleaved execution needs additional constraints and analysis. The current data do not validate that application.

A valid application test needs complete offers, original reveal times, then-available features and permission for the algorithm’s actions. Completed-service records and these adjacent-block comparisons establish neither that match nor improvement over LR.

## Q. Forecasting the effect of an advice bit

An alternative to optimal-membership prediction is to estimate how the current advice bit affects final retained length. This interface specializes fixed-reference rollout (Tesauro & Galperin, 1996) to BANK. It preserves arbitrary-bit robustness, with accuracy measured by conditional reward contrasts.

The conditional target. Consider a finite horizon with possibly earlier random termination and integrable terminal rewards. The exogenous offer process is independent of the algorithm’s choices. A state s contains the observed history, time, held set, ledgers, permanent filter history, and any reference memory. Fix a deterministic pure causal bit rule A, defined at every reachable counterfactual state. Let $Q ^ { A } ( s , b )$ be expected final length after using bit b now and following A thereafter, under the true conditional future-offer and stopping law. Define

$$
\Delta _ { A } ( s ) = Q ^ { A } ( s , 1 ) - Q ^ { A } ( s , 0 ) .
$$

A paired estimate replays the same hypothetical suffix under both current bits, with each branch continuing A from its own state. The deployed controller ν recomputes this estimate at each arrival and chooses its sign; ties may use A(s). Hypothetical suffixes must not be the test episode’s realized future. Historical samples alone also do not certify that their conditional law matches deployment.

Why immediate retained value is insufficient. An advice bit can change a ledger even when both current actions reject the request. One reachable example begins with a negative [0, 1), followed by positive intervals $[ - . 6 , . 0 1 ) , [ . 0 1 , . 5 )$ and [.5, .99). All three are rejected, leaving the unit holder with ledger 1.59. At $I = [ . 9 9 , 1 . 6 )$ both bits again reject; bit one raises its ledger to 2.20. Now offer the negative [0, 2.8). The smaller ledger permits replacement, while the larger one prevents it. A final negative $[ 1 , 4 )$ gives branch rewards 2.8 and 4; replacing that final offer by [3, 6) gives 5.8 and 4. The bit contrast is therefore $6 / 5 \mathrm { o r } - 9 / 5 $ despite identical immediate reward and identical suffix lengths. The positive intervals are compatible, so filtering does not erase the example. It diagnoses information that the interface can use; it is not a performance comparison of policies started from empty state.

Proposition Q.1 (Expected transfer of a reference certificate). Let $Y _ { A } , Y _ { \nu }$ be final rewards of the reference and deployed controllers, both using filtered Shaped-BANK from empty state. For an estimated contrast $\widehat { \Delta } _ { A }$ , suppose the quantities below are integrable and put

$$
\varepsilon _ { \mathrm { r o l l } } = \mathbb { E } _ { \nu } \sum _ { t } | \widehat { \Delta } _ { A } ( s _ { t } ) - \Delta _ { A } ( s _ { t } ) | .
$$

Let $\eta _ { A }$ be the error in (K.1) of the raw labels produced by A on its own trajectory. Then

$$
\mathbb { E } Y _ { \nu } \geq \operatorname* { m a x } \left\{ \frac { \mathbb { E } \mathrm { O P T } } { \varphi ^ { 5 } } , \frac { \mathbb { E } \mathrm { O P T } - \mathbb { E } \eta _ { A } } { d } - \varepsilon _ { \mathrm { r o l l } } \right\} .\tag{Q.1}
$$

Proof. The finite-horizon performance-difference identity (Ross & Bagnell, 2014, Lemma 4.3) gives

$$
\mathbb { E } Y _ { \nu } - \mathbb { E } Y _ { A } = \mathbb { E } _ { \nu } \sum _ { t } ( \nu ( s _ { t } ) - A ( s _ { t } ) ) \Delta _ { A } ( s _ { t } ) .
$$

The sign rule satisfies $( \nu - A ) \widehat { \Delta } _ { A } \geq 0$ and $| \nu - A | \leq 1$ , so each summand is at least $- | \widehat { \Delta } _ { A } - \Delta _ { A } |$ . Apply Corollary K.1 b bto A’s own trajectory. Independently, arbitrary-bit robustness applies pathwise to ν, giving the other branch of (Q.1).

Table 11. Fixed all-negative-reference attribution cohort: six Context trials and 384 test episodes. No-MC columns use certified finite expectations on the same models and episodes as the 128-suffix column. Retained and revoked lengths are in the original interval units; the primary paired outcome is retained length.
<table><tr><td>Method</td><td>128 suffixes % OPT</td><td>No MC % OPT</td><td>No MC Retained</td><td>No MC Revoked</td></tr><tr><td>Conditional</td><td>94.334</td><td>94.817</td><td>13.76302</td><td>4.86719</td></tr><tr><td>Pooled</td><td>94.474</td><td>94.458</td><td>13.68750</td><td>4.91406</td></tr><tr><td>Uniform</td><td>94.143</td><td>94.423</td><td>13.68229</td><td>4.66406</td></tr><tr><td>Known law</td><td>94.660</td><td>94.665</td><td>13.74740</td><td>4.89063</td></tr><tr><td>Myopic</td><td></td><td>90.652</td><td>12.91667</td><td>2.45313</td></tr><tr><td>All negative</td><td></td><td>87.297</td><td>12.45573</td><td>.83333</td></tr></table>

The error is measured on $\nu \mathrm { { s } }$ visited states, not on a fixed archive of reference states; it includes scenario-model and sampling error. The member-label error $\eta _ { A }$ belongs to A, not to ν. Exact conditional contrasts remove $\varepsilon _ { \mathrm { r o l l } }$ , but do not turn rollout bits into perfect membership predictions or supply a per-execution consistency-d guarantee. With an all-negative reference, $\eta _ { A } = \mathrm { O P T }$ and the second branch is vacuous. Thus a good causal reference is a substantive part of this interface.

## Q.1. Complete-policy diagnostics

We use two fixed synthetic offer laws, with all policies starting empty. The 16 interval types combine starts {0, 3, 6, 9} and lengths {1, 3, 6, 12}. Initial types are uniform; episodes stop with probability .1 after each offer and are capped at 24 offers. IID transitions are uniform. In the Context law, an observed binary context is fixed for each episode. Given current type i, next-type weights are ${ \bf 1 } + 4 { \bf 1 } \{ j$ compatible with $i \} + 2 1 \{ w _ { j } \leq 3 \}$ in context zero and ${ 1 + 4 \mathbf { 1 } \{ j } $ contains $i \rbrace + 2 \mathbf { 1 } \lbrace w _ { j } \geq 6 \rbrace$ in context one. Both laws have full support. Laplace-smoothed counts fit conditional or pooled transitions; fitted stopping accounts for censoring at the cap.

Learning versus lookahead. The original attribution study used six independent Context training trials and two IID controls, each with 2,048 training and 64 fresh test episodes. Every method continues with the all-negative reference after the current advice bit. Conditional, pooled and uniform models differ only in their future laws; uniform learns no transitions and knows the specified stopping rate. A known-law control knows the generator but never the realized future. The original Monte Carlo comparison used 16/128 common-random-number suffixes; the finite-expectation analysis below reuses its fitted models and test episodes.

Finite-state recursion evaluates the 512 test episodes (384 Context, 128 IID) through the remaining horizon. Outward bounds certify nonzero contrast signs under the sampler’s 53-bit law, and exact rational recursion certifies the remaining 26 zeros.

On the 384 Context episodes, conditional minus uniform gains .08073 in mean retained length and .3943 percentage points in mean $\mathrm { A L G } / \mathrm { O P T }$ (13 wins, 365 ties, 6 losses). Four of six trial length differences are positive and two negative. Uniform minus myopic gains .76563 length and 3.7711 points. Conditional minus pooled gains .07552 length and .3596 points, compared with −.1409 points using 128 suffixes. Sampling precision therefore affects this cohort’s comparison. In all 128 IID episodes, the four no-Monte-Carlo laws have identical terminal values, averaging 95.4432% of OPT. These reused-test results are descriptive; Table 12 reports an independent cohort.

Independent cohort. The independent cohort used 24 Context and eight IID training trials, each with 2,048 training and 64 test episodes under the same generator, fitting rules, all-negative reference and finite-expectation evaluation. All seven policies started empty; a swapped control exchanged the fitted context rows. The primary contrast was conditional minus uniform length, averaged within training trial (Table 12). Secondary intervals are descriptive and unadjusted for multiplicity.

The primary trial means split 11 positive, three zero and ten negative (22 episode wins, 1,496 ties, 18 losses). Its interval confirms neither a positive mean gain nor a zero effect; the earlier six-trial positive cohort remains separate. Uniform minus myopic gains 2.0836 percentage points in mean $\mathrm { A L G / O P T }$ (descriptive 95% interval [1.5439, 2.6234]). Conditional/pooled/uniform mean test negative log-likelihoods per opportunity, averaging trials equally, are 2.630468/2.772968/2.818286: better future-law fit does not establish added terminal reward. In the eight IID trials, conditional minus uniform is −.007813 length, with 95% interval $\left[ - . 0 4 9 7 0 7 , . 0 3 4 0 8 2 \right]$

Table 12. Independent Context cohort: 24 training trials and 1,536 test episodes. Differences are in retained-length units; Student-t intervals use 23 degrees of freedom. Episodes sharing a fitted model are not independent training replications.
<table><tr><td>Predeclared comparison</td><td>Mean difference</td><td>Trial-level 95% interval</td></tr><tr><td>Conditional — uniform (primary)</td><td>+.003255</td><td> $\left[ - . 0 4 1 7 0 4 , + . 0 4 8 2 1 4 \right]$ </td></tr><tr><td>Conditional — pooled</td><td>-.010417</td><td> $\left[ - . 0 5 3 0 7 2 , + . 0 3 2 2 3 9 \right]$ </td></tr><tr><td>Conditional — swapped</td><td>+.024740</td><td> $\left[ - . 0 3 3 2 6 8 , + . 0 8 2 7 4 7 \right]$ </td></tr><tr><td>Uniform — myopic</td><td>+.462891</td><td> $\left[ + . 3 7 5 4 0 7 , + . 5 5 0 3 7 4 \right]$ </td></tr><tr><td>Known law – uniform</td><td>+.019531</td><td> $\left[ - . 0 1 9 7 8 0 , + . 0 5 8 8 4 3 \right]$ </td></tr></table>

Earlier cohorts and interpretation. Earlier exploratory cohorts used four fitted training trials per law and separate test episodes. They do not provide uniform learning gains: with an all-negative reference, fitted Context rollout averaged 92.83% versus uniform’s 94.36%, and increasing suffixes from 16 to 128 lowered another fitted Context mean from 95.44% to 94.98%. A learned-reference diagnostic instead fit A from context, type and index bin with Laplace smoothing and threshold $1 / 2$ . Its error branch was active in 134/256 episodes; rollout improved every trial mean over A, but fitted-minusuniform averaged $- . 2 5 / + . 1 3$ points in IID/Context. The earlier cohorts’ shared fitted models and different references preclude pooling with the cohorts above.

The known-law controller is not optimal continuation: it follows its fixed reference in hypothetical suffixes and replans at the next actual arrival. All tables average episode ratios, not $\mathbb { E } \mathrm { A L G } / \mathbb { E } \mathrm { O P T }$ in the certificate. Cancellation is material: in the attribution cohort, mean revoked length is 2.4531 for myopic, 4.6641 for uniform and 4.8672 for conditional rollout; in the learned-reference cohort it rises from 1.70/1.85 for A to 4.39/4.16 with fitted-law rollout. On the original attribution cohort’s fixed Context trajectories, subtracting a proportional fee c per revoked length gives conditional minus uniform mean net reward $( 3 1 - 7 8 c ) / 3 8 4$ . It stays positive throughout the unchanged algorithm’s fee domain $0 \leq c < 1 / ( 2 \varphi )$ the six trial differences remain four positive and two negative. This does not hold across all environments: the IID rollout minus-myopic mean is $( 6 3 - 2 4 5 c ) / 1 2 8 .$ , which turns negative above $c = 9 / 3 5$ within that domain. This is retrospective accounting in original length units, with no new samples or fee-aware reoptimization.

On the new fixed Context trajectories, charging c per revoked length gives conditional minus uniform mean net reward $( 5 - 1 9 0 c ) / 1 5 3 6$ and uniform minus myopic $( 7 1 1 - 2 8 8 7 c ) / 1 5 3 6$ . Their sign changes at $1 / 3 8$ and $7 1 1 / 2 8 8 7$ both lie inside $0 \leq c < 1 / ( 2 \varphi )$ . Thus the original cohort’s positive conditional-minus-uniform net difference throughout that domain does not extend to this cohort.

These studies do not estimate deployment-state error in (Q.1), and matched suffix counts do not equalize inference costs.

## R. Further model comparisons

## R.1. Further related models and source scope

Guarantees in the same model. The extended version of Karavasilis (2025), Theorem 3.5, proves proportional-weight Naive; Appendix K.1 gives our self-contained proof. The proceedings, Section 4, Algorithm 3 and Lemma 4.5, give LR’s maximum-conflict rule at parameter $\varphi$ with competitive ratio at most $2 \varphi + 1$ , attributed there to Garay et al. (1997). Section 1 cites the stopping-prefix antecedent and finitely robust 1-increasing barrier. The latter uses $\alpha = 1 { : }$ for $\alpha < 1$ , its construction need not force rejection, since the internal request has length $v - 2 \epsilon > \alpha v$ when $\epsilon < ( 1 - \alpha ) v / 2$

The matching prediction-free lower bound is Theorem 1 in Section 2 of the October 28, 1995 author manuscript of Tomkins (1995). It allows preemption of infinite-duration calls on a line, with length rewards paid only to calls retained at termina tion. Represent a route from i to j by [i, j): shared-edge conflict, endpoint compatibility, length reward, and permanent rejection or preemption agree with our model. Fix every prediction to zero. Any deterministic algorithm here then induces a prediction-free algorithm on this subclass, so $R \geq 2 \varphi + 1$ . No request expires, and arbitrary stopping prefixes are allowed. This is a supremum lower bound, not a claim that a finite witness attains equality.

Relation to earlier charging arguments. For the unweighted 2k guarantee, Borodin & Karavasilis (2023, Section 3, Propositions 1–2) charge optimal intervals to holders and transfer charges through replacements, separately bounding two direct and at most 2k − 2 transferred charges. We retain this device, adding cost-weighted ancestry antichains and the twocut functional $F _ { J } ( x , y )$ to bound the remaining geometric gap and both exterior charges jointly under special, potentially length-decreasing replacements and nonconstant ledger costs.

Prediction and decision interfaces. Our bits refer to one fixed final optimum, with deterministic robustness for every fixed label sequence. In contrast, Dallot et al. (2026, Section 2) use answer-dependent guidance whose replacement by adversarial guidance is selected independently at each step. TrustGreedy of Boyar et al. (2026) maximizes cardinality using an upfront prediction of the full input set; it revises a planned solution while preserving irrevocably accepted requests. L et al. (2026) study unit-length, unit-profit simple chains under random and adversarial orders, including complexity with a trustworthy advice tape.

Several adjacent guarantees allow different decisions. Daneshvaramoli et al. (2025, Theorems 4.1 and 5.2) mix fractional knapsack decisions from critical-value predictions; their integral conversion assumes bounded unit values and sufficiently small item weights. It addresses a capacity constraint, not interval conflicts. Grigorescu et al. (2025) use real packing decisions, advice interpolation and a quantified feasibility relaxation. Scheduling with partial size predictions permits rate sharing and resumption (Benomar & Perchet, 2024); paging permits reloads and a consumable, resettable relative prediction budget (Chen et al., 2026). These mechanisms differ from BANK’s inherited positive mass and permanent rejection. The request-prediction framework of Ameli et al. (2026, Section 1, Theorems 1.1–1.2) uses irrevocable covering decisions, feasibility under unions, and subadditive costs. Feasible interval selections need not remain feasible under unions, so that covering guarantee does not directly transfer.

Weights, order, and source versions. The technical version of Berg (2025), Corollary 36, treats unit profits, irrevocable decisions and bounded overlap t, with $1 \leq \alpha \leq t$ and decision bit 0 denoting acceptance. Splitting length-weighted intervals into unit requests can change feasibility, prediction identities and available online decisions, so it does not transfer that guarantee. Antoniadis et al. (2026) use membership bits and length weights, but require release-time order and irre vocable selection; their guarantees involve additive terms, length ratios or randomization. Our results instead use arbitrary order and strict ratios. Berg and Boyar are compared through their public technical versions; the simple-chain source is arXiv:2608.10376v1, and the paging and concave-packing sources are arXiv:2606.01342v3 and arXiv:2411.08332v1.

Advice, learning, and randomization. The Pareto viewpoint for trusted and untrusted advice precedes this work (An gelopoulos et al., 2020); Mitzenmacher & Vassilvitskii (2020) survey prediction-based algorithms. Khodak et al. (2022) learn predictors through algorithm-dependent quality measures, without directly supplying a guarantee for our membership interface. Theorem 2.1 of Ben-David et al. (1994) derandomizes against adaptive offline adversaries in a request-answer model. Our mixture instead fixes the complete labelled input before randomization, with consistency and robustness suprema over different input classes; that theorem alone does not yield an adaptive two-criterion extension.

Immediate service and learning. Capacity-constrained online learning loses delayed feedback when a tracked round is abandoned (Ryabchenko et al., 2025); its regret objective depends on which observations survive. Our duration guarantee does not imply a learning-regret bound. Wang et al. (2026) study no-buffer scheduling with abandonment and censored reward feedback. Their general objective uses task rewards, not occupied duration. With one machine, arrival times $b _ { i }$ known lengths $\ell _ { i } ,$ rewards equal to $\ell _ { i } .$ , and time to finish the final accepted jobs, the intervals $[ b _ { i } , b _ { i } + \ell _ { i } )$ form a start-ordered subclass of our model. Completed jobs can remain in the held-set record because later arrivals cannot conflict with them. Our upper guarantees therefore apply to this restricted case; our full-class lower bounds and their bandit regret bounds do not transfer between the two models.

Earlier scheduling and nearby tradeoffs. Early interval scheduling includes Lipton & Tomkins (1994) and Woeginger (1994); the latter rewards completion of fixed-start/end jobs started at arrival and lost if aborted, rather than terminal heldset rewards. Seiden (1998) improves the randomized guarantee in this start-ordered setting, distinguished from any-order selection by Borodin & Karavasilis (2023, Section 1); its guarantees do not replace LR for arbitrary order.

Fung et al. (2012) randomize initially between two deterministic schedulers; arrival order and completion rewards prevent transferring their constants. Jin & Ma (2022) prove a tight two-stage matching tradeoff where naive coin interpolation can be suboptimal for vertex weights. Its advice/hindsight ratios and irrevocable batches do not give an interval lower bound. Antoniadis et al. (2020) robustify metrical-task and line-matching predictions through different states and interfaces. Nair et al. (2021) learn MIP partial assignments and branching decisions.