# Explicit Asymptotic Bounds for Sequential Calibration Beyond $T ^ { 2 / 3 }$

Eric Dai

Edison Academy Magnet School

eric.dai.kj@gmail.com

Maxwell Fishelson

Institute for Advanced Study

maxfish@ias.edu

October 7, 2026

## Abstract

Probability forecasts are calibrated when predicted probabilities match empirical outcome frequencies: among events assigned a probability p, we’d hope that the fraction of positive outcomes is close to p. We study the problem of sequential forecasting of binary outcomes. The classical $O ( T ^ { 2 / 3 } )$ bound on expected cumulative ℓ -calibration error established by Foster and Vohra [FV98] stood for over two decades until Dagan et al. $[ \mathrm { D D F ^ { + } 2 5 } ]$ reduced the exponent 2/3 by an unspecified constant. We establish a new two-phase recursive labeling strategy for the sign-preservation-with-reuse game that yields the bound $O ( n ^ { \alpha } t ^ { \beta } )$ for all choices of space and time. We then sharpen the reduction from upper bounds on sign preservation to calibration by modifying the equivalence of Dagan et al. to use only O(log T) instances of the sign-preservationwith-reuse game. This lets us establish an explicit bound of $O ( T ^ { 0 . 6 6 2 9 4 2 2 8 8 } )$ as the first explicit exponent below 2/3 for sequential calibration by combining both improvements and choosing explicit feasible parameters.

## 1 Introduction

Calibration is a fundamental criterion in statistical theory with well-known theoretical and empirical foundations [Daw82]. A forecaster is calibrated if its probability forecasts agree with the frequencies of observed outcomes. For example, on the days when a calibrated weather forecaster reports a 30% chance of rain, it should rain on approximately 30% of those days. Therefore, calibration is a natural desideratum for probability forecasts.

The notion of calibration has found a wide variety of applications in modern machine learning. Calibration has been used to assess the confidence of neural networks in image classification and other tasks [GPSW17, KLM19]. Calibration can also improve uncertainty estimates for numerical predictions [KFE18] and appears in algorithmic fairness criteria [PRW<sup>+</sup>17]. Calibration has led to the idea of multicalibration, which requires calibration across a specified collection of subpopulations [HJKRR18].

Sequential calibration. We study the classical sequential calibration problem. The problem describes a two-player game between a forecaster and an adversary who interact for T time steps. At each time step $t \in [ T ]$ , the forecaster chooses a prediction $p _ { t } \in [ 0 , 1 ]$ and the adversary simul taneously chooses an outcome $y _ { t } \in \{ 0 , 1 \}$ , and then both choices are revealed. Each player may randomize and use the history of past predictions and outcomes, but the adversary cannot observe the current realized prediction before choosing $y _ { t }$ . The prediction $p _ { t }$ represents the forecaster’s probability for the outcome $y _ { t } = 1$

Let $n _ { T } ( p )$ be the number of time steps on which the prediction is $p ,$ and let $m _ { T } ( p )$ be the number of those time steps with outcome 1. The cumulative $\ell _ { 1 }$ -calibration error is

$$
\operatorname { c a l e r r } ( T ) : = \sum _ { p \in [ 0 , 1 ] } | m _ { T } ( p ) - p \cdot n _ { T } ( p ) | .
$$

Foster and Vohra [FV98] were the first to provide the classical upper bound $O ( T ^ { 2 / 3 } )$ on expected cumulative calibration error. Following their work, the question of whether the exponent $2 / 3$ could be improved remained unresolved for more than two decades. Recently, Dagan et al. [DDF<sup>+</sup>25] broke the $2 / 3$ barrier by proving a bound of $O ( T ^ { 2 / 3 - \eta } )$ for some absolute constant $\eta > 0$ . However, their proof does not state a specific number for $\eta ,$ and thus does not explicitly state an exponent below $2 / 3$ for calibration error. This motivates the quantitative question: how much can the exponent $2 / 3$ be reduced in the sequential calibration problem?

## 1.1 Our contributions

We obtain the following explicit upper bound on expected cumulative calibration error. The exponent is an improvement of more than 0.00372 over the classical upper bound of $2 / 3$

Theorem 1.1 (Main result, informal). For every horizon $T \geq 1$ , there exists a randomized forecaster whose expected cumulative calibration error against any adversary is

$$
\mathbb { E } [ \mathrm { c a l e r r } ( T ) ] = O \left( T ^ { 0 . 6 6 2 9 4 2 2 8 8 } \right) .
$$

Related work and approach. The key technical idea is to reduce calibration to a combinatorial game. Qiao and Valiant [QV21] introduced the sign-preservation game to prove lower bounds on calibration error. Dagan et al. $[ \mathrm { D D F ^ { + } 2 5 } ]$ introduced the sign-preservation-with-reuse $( S P R )$ game and established its connection to calibration upper bounds. SPR takes place on an ordered board of cells. In each round, one player, the pointer, selects an empty cell, and a second player, the labeler, places a plus or minus sign there. The labeler may remove minus signs to the left of the selected cell and plus signs to its right. The labeler and the pointer seek to minimize and maximize the number of signs remaining on the board, respectively. Dagan et al. show that given a labeling strategy that preserves $O ( n ^ { 1 - \varepsilon } )$ signs after at most n rounds on n cells, there exists a forecaster with expected calibration error $O ( T ^ { 2 / 3 - \varepsilon / 1 8 } )$

This reduction from calibration to a labeling strategy in SPR uses the minimax approach of Hart [Har25]. His proof establishes the classical $O ( T ^ { 2 / 3 } )$ bound of Foster and Vohra [FV98] by first allowing the forecaster to know the adversary’s conditional probability of the next outcome, and then applying minimax to obtain a strategy without this information. Dagan et al. use SPR to improve the bound in this argument and break the $2 / 3$ barrier. One limitation of this approach is that the minimax step does not give an eficient forecasting algorithm. Zhang [Zha26] gave a polynomial-time randomized forecaster with expected calibration error $O ( T ^ { 2 / 3 - \eta } )$ for some constant $\eta > 0$ , combining the SPR forecasting procedure with a correction based on Blackwell’s approachability theorem that controls the error from using surrogate conditional means. Our contribution focuses on improving the explicit upper bound on the calibration exponent.

An improved labeling strategy. We give a two-phase recursive labeling strategy, building on the recursive bias ideas of Dagan et al. [DDF<sup>+</sup>25]. The strategy first runs on the two halves of an interval until their request counts satisfy a balance condition. When the condition is satisfied, the strategy continues with an adjusted bias toward one sign type. The analysis allows us to establish separate bounds for newly placed signs and signs that survive from earlier epochs. This labeling strategy yields bounds that depend separately on the number of cells and the number of rounds.

A sharper passage to calibration. We modify the forecasting construction in the reduction of Dagan et al. $[ \mathrm { D D F ^ { + } 2 5 } ]$ . We analyze this construction using the labeling bound $O ( n ^ { \alpha } t ^ { \beta } )$ at each partition scale. Our construction uses only ${ \cal O } ( \log T )$ SPR instances, compared with the $O ( ( \log T ) ^ { 2 } )$ instances used by Dagan et al. $[ \mathrm { D D F ^ { + } 2 5 } ]$ . For $\alpha \in ( 0 , 1 ) , \beta \in ( 0 , 1 )$ and $\varepsilon = 1 - \alpha - \beta > 0$ , our construction obtains an expected calibration error of

$$
O \Big ( T ^ { \frac { 2 + \varepsilon } { 3 + 2 \varepsilon } } \Big ) = O \Big ( T ^ { \frac { 2 } { 3 } - \frac { \varepsilon } { 9 + 6 \varepsilon } } \Big ) .
$$

The analysis of this construction sums the conditional-means contributions of the SPR instances across diferent space and bias scales, and then balances their total against the fluctuations of outcomes around their conditional means.

Explicit parameters. Our final calibration exponent depends on the parameters chosen for the labeling strategy. We prove that the parameters we choose satisfy certain inequalities that are suficient to establish our upper bound for SPR. The resulting game bound is

$$
O \left( n ^ { 0 . 0 0 8 2 5 4 0 8 9 2 } t ^ { 0 . 9 5 7 4 6 0 3 4 6 7 } \right) .
$$

Applying the reduction with these exponents proves Theorem 1.1.

## 2 Preliminaries

In this section, we formally introduce and define sequential calibration and the sign-preservationwith-reuse game. For an integer $n \geq 1$ , write $[ n ] : = \{ 1 , \dots , n \}$ . Unless otherwise specified, horizons, board sizes, and round counts are integers.

## 2.1 Sequential calibration

Sequential calibration is a theoretical framework where the task of making reliable predictions over time is modeled as a repeated online game between a forecaster and an adversary. Fix a horizon $T \geq 1$ and a finite prediction set $P \subset [ 0 , 1 ]$ which the forecaster chooses. At each time step $t \in [ T ]$ ， the forecaster predicts $p _ { t } \in P$ and the adversary simultaneously chooses an outcome $y _ { t } \in \{ 0 , 1 \}$ meaning the adversary cannot observe the current prediction $p _ { t }$ before choosing $y _ { t }$ . Both players may randomize and use the history of the game $H _ { t } = ( ( p _ { i } , y _ { i } ) ) _ { i = 1 } ^ { t - 1 }$ , but the forecaster does not know the adversary’s conditional probability of choosing $y _ { t } = 1$ given that history.

For $p \in P$ , let $\begin{array} { r } { n _ { t } ( p ) = \sum _ { i = 1 } ^ { t } \mathbf { 1 } \{ p _ { i } = p \} } \end{array}$ be the number of predictions at $p ,$ and let $m _ { t } ( p ) =$ $\textstyle \sum _ { i = 1 } ^ { t } y _ { i } \mathbf { 1 } \{ p _ { i } = p \}$ count their positive outcomes. The cumulative $\ell _ { 1 } { - } c a l i b r a t i o n$ error is

$$
\operatorname { c a l e r r } ( t ) = \sum _ { p \in P } { \big | } m _ { t } ( p ) - n _ { t } ( p ) \cdot p { \big | } .
$$

We wish to obtain a bound on E[calerr(T)] against every adversary, where the expectation is over both players’ randomness.

## 2.2 Sign-preservation-with-reuse game

The sign-preservation game, introduced by Qiao and Valiant [QV21], is a two-player combinatorial game used to model a simplified version of sequential calibration. In this paper, we consider the generalized version of the sign-preservation game, which Dagan et al. [DDF<sup>+</sup>25] introduce as the sign-preservation-with-reuse game. Fix integers $n \geq 1$ and $t \geq 0$ . There are initially n empty cells, numbered $1 , \ldots , n$ , and two players: Player-P (the pointer ) and Player-L (the labeler ). Each of at most t rounds proceeds as follows:

1. The pointer may terminate the game. Otherwise, it selects an empty cell $j \in [ n ]$

2. The labeler may remove any subset of the minus signs to the left of $j$ and the plus signs to the right of $j$

3. The labeler places either a plus or a minus in cell $j .$

Sign-preservation-with-reuse difers from sign-preservation by allowing the pointer to select cells that have been emptied before. The game ends when the pointer stops, all t rounds have occurred, or no empty cell remains. The pointer and labeler seek to maximize and minimize the number of signs remaining, respectively. Both players may observe the full history of the game and randomize in their strategies.

Let opt(n, t) denote the number of signs remaining at the end of a game of SPR with n cells and t rounds under optimal play. Qiao and Valiant [QV21] described strategies for the pointer in order to prove calibration lower bounds. Conversely, we and Dagan et al. [DDF<sup>+</sup>25] describe strategies for the labeler, yielding upper bounds on opt(n, t) and calibration error.

## 3 Overview of Theorem 1.1

In this section, we outline the proof of Theorem 1.1, which is broken down into three main parts. Section 4 describes the new labeler strategy for the sign-preservation-with-reuse game. Section 5 describes the improved reduction from calibration to the sign-preservation-with-reuse game. Section 6 provides the specific parameters used to obtain the final calibration exponent.

## 3.1 Overview of Section 4

We seek the upper bound on $\mathrm { o p t } ( n , t )$ of Theorem 4.1,

$$
\mathrm { o p t } ( n , t ) = O ( n ^ { \alpha } t ^ { \beta } )\tag{1}
$$

with $\varepsilon : = 1 - \alpha - \beta > 0$ , by choosing the parameters from an admissible tuple as in Theorem 4.2. For now, assume n is a power of two. Note that simply maintaining independent labelers on the two halves does not sufice: if each labeler receives $t / 2$ requests, summing their bounds gives

$$
2 ( n / 2 ) ^ { \alpha } ( t / 2 ) ^ { \beta } = 2 ^ { 1 - \alpha - \beta } n ^ { \alpha } t ^ { \beta } .\tag{2}
$$

Since $2 ^ { 1 - \alpha - \beta } > 1$ , (2) fails to close the induction. Instead, we must account for deletions between the two halves to obtain stronger bounds.

Our labeling strategy deletes every minus to the left of a request and every plus to its right, which afects the balance of positive and negative signs within the game. Following Dagan et al.

$[ \mathrm { D D F ^ { + } 2 5 } ]$ , we use recursive bias to exploit this asymmetry: we wish to show an instance with integer bias parameter b leaves at most

$$
C \lambda ^ { b \sigma } n ^ { \alpha } t ^ { \beta }\tag{3}
$$

surviving signs of each type $\sigma \in \{ - 1 , + 1 \}$ among those it placed, where C is a constant and $\lambda > 1$ We can see that by changing b by one, we trade of the bounds in (3) on the two sign types. For example, decreasing b by one decreases the plus bound by a factor of λ while increasing the minus bound by a factor of λ. We will show that (3) holds by induction on space and strongly on time at each fixed space.

Each epoch first runs instances in phase 1 on the two halves. Write $t _ { h } \ge t _ { l }$ for their request counts, denoting the number of times the heavy half and light half were requested, respectively, and τ for the requests in earlier epochs. By the inductive hypothesis, the two children’s contributions are bounded by

$$
C \lambda ^ { b \sigma } ( n / 2 ) ^ { \alpha } t _ { h } ^ { \beta } \quad \mathrm { a n d } \quad C \lambda ^ { b \sigma } ( n / 2 ) ^ { \alpha } t _ { l } ^ { \beta } .\tag{4}
$$

Since some signs survive between epochs, treating each epoch separately gives too weak a bound. Instead, we maintain a bound for all remaining signs and a stronger bound for persistent signs. Persistent signs are pluses in the left half of an interval and minuses in the right half. Oppositehalf requests cannot delete persistent signs, and once both halves have been visited in the current epoch, only persistent signs from earlier epochs can remain. At each restart, for each sign type σ, we bound persistent and total signs, respectively, by

$$
C \lambda ^ { b \sigma } ( n / 2 ) ^ { \alpha } K _ { p } \tau ^ { \beta } \quad \mathrm { a n d } \quad C \lambda ^ { b \sigma } ( n / 2 ) ^ { \alpha } K _ { t } \tau ^ { \beta } ,\tag{5}
$$

where $0 < K _ { p } < 1$ and $K _ { t } > K _ { p }$ . We distinguish three cases according to the state of the current epoch.

Case 1: phase 1, only one half has been visited. Here $t _ { l } = 0$ and $t = \tau + t _ { h }$ . Both persistent and non-persistent signs from earlier epochs may remain, so we use the total-sign bound in (5). Adding the visited child’s contribution from (4) gives

$$
C \lambda ^ { b \sigma } ( n / 2 ) ^ { \alpha } ( K _ { t } \tau ^ { \beta } + t _ { h } ^ { \beta } ) \leq C \lambda ^ { b \sigma } ( n / 2 ) ^ { \alpha } D t ^ { \beta } .\tag{6}
$$

Thus, the admissibility condition $D < 2 ^ { \alpha }$ makes (6) smaller than the desired bound (3).

Case 2: phase 1, both halves have been visited. Only persistent signs from earlier epochs remain, so the coeficient $K _ { t }$ in (6) is replaced by $K _ { p }$ , while both contributions in (4) must be counted. Since $0 < \beta < 1$ , for a fixed sum $t _ { h } + t _ { l }$ , the expression $t _ { h } ^ { \beta } + t _ { l } ^ { \beta }$ is largest when the counts are equal. Thus, while $t _ { l } < \rho t _ { h }$ , the imbalance improves the bound, and we can continue using the two half-interval instances. If $\delta t _ { l } < \tau$ , the smaller coeficient $K _ { p }$ in (5) supplies the saving even when the two half-counts are balanced. These two sources of saving explain why phase 1 continues until both $t _ { l } \ge \rho t _ { h }$ and $\delta t _ { l } \ge \tau$ . While either phase 1 condition holds, admissibility gives, with $t = \tau + t _ { h } + t _ { l }$

$$
C \lambda ^ { b \sigma } ( n / 2 ) ^ { \alpha } ( K _ { p } \tau ^ { \beta } + t _ { h } ^ { \beta } + t _ { l } ^ { \beta } ) \le C \lambda ^ { b \sigma } ( n / 2 ) ^ { \alpha } M t ^ { \beta } .\tag{7}
$$

Thus, $M \leq D < 2 ^ { \alpha }$ makes Equation (7) smaller than the desired bound (3).

Case 3: phase 2. When the counts become suficiently balanced and both are large relative to the count from earlier epochs, we trigger phase 2. Suppose the trigger lies in the right half, whose count is $t _ { l }$ . It deletes every minus in the left half, removing the larger child’s contribution to the minus bound. The strategy then uses a full-interval continuation with bias $b - 1$ for subsequent requests. After $t _ { c }$ continuation requests, its contributions to the plus and minus bounds are, respectively,

$$
C \lambda ^ { b - 1 } n ^ { \alpha } t _ { c } ^ { \beta } \quad \mathrm { a n d } \quad C \lambda ^ { - ( b - 1 ) } n ^ { \alpha } t _ { c } ^ { \beta } .\tag{8}
$$

The deleted signs provide room for the larger minus allowance in (8), while the plus bound benefits from the smaller allowance. This continuation lasts for $\lfloor \kappa t _ { h } \rfloor$ requests in order to limit the time for which the instance has changed bias. A left-half trigger exchanges the roles of the signs.

At an epoch’s end, the deletion and bias change ensure that the persistent and total survivors satisfy (5) with the updated request count. This allows the argument to continue across restarts. The proof must also bound executions that stop before an epoch ends. Combining (6), (7), and the phase 2 contributions in (8) with the inherited bounds (5) gives, for suficiently large t, the bound

$$
C \lambda ^ { b \sigma } ( n / 2 ) ^ { \alpha } ( D t ^ { \beta } + 1 ) .\tag{9}
$$

The admissibility condition $D < 2 ^ { \alpha }$ absorbs the additive term in (9) for large t, while executions with small t are handled by the constant $C ,$ proving the desired bound of (3). Summing over the two sign types at bias zero and padding the board to a power of two then gives the bound (1) of Theorem 4.1 for arbitrary board sizes.

This simpler two-phase strategy replaces the four-phase strategy of Dagan et al. $[ \mathrm { D D F ^ { + } 2 5 } ]$ Section 6 gives an admissible parameter tuple with $\varepsilon > 0$ ε > , completing the claimed bound of (1).

## 3.2 Overview of Section 5

Theorem 5.1 converts the bound on the number of remaining signs in SPR into calibration error

$$
O ( T ^ { \frac { 2 + \varepsilon } { 3 + 2 \varepsilon } } ) ,\tag{10}
$$

where $\varepsilon = 1 - \alpha - \beta > 0$ is derived from an admissible tuple as defined in Theorem 4.2. We modify the forecasting construction of Dagan et al. $[ \mathrm { D D F ^ { + } 2 5 } ]$ and analyze it using the separate space and time bounds for SPR, rather than only using the diagonal regime $t = n$ . As in their minimax approach, we first allow the forecaster to observe the conditional mean $e _ { t }$ of the next outcome.

For $T = 2 ^ { \tau }$ , the forecaster uses h partition levels starting at i<sub>0</sub>. The coarsest level, $i = i _ { 0 }$ , uses several bias thresholds, while each finer level, $i \in \{ i _ { 0 } { + } 1 , \ldots , i _ { 0 } { + } h { - } 1 \}$ , uses just one bias threshold. At each level and bias threshold pair, we establish an even and odd instance of SPR, so that cells in one game are separated. This construction uses only ${ \cal O } ( \log T )$ SPR instances, compared to the $O ( ( \log T ) ^ { 2 } )$ instances used by Dagan et al. $[ \mathrm { D D F ^ { + } 2 5 } ]$ . At level i, each cell represents an interval of length $2 ^ { - ( i + 1 ) }$ , and a threshold index $j$ specifies the bias threshold $2 ^ { j - i }$ . A plus selects the left endpoint, and a minus selects the right endpoint. Each cell in each game keeps a separate bias record at each endpoint, adding $e _ { t } - p _ { t }$ whenever that endpoint is selected for the cell. The record corresponding to the cell’s current sign is its active bias, taken to be zero for an empty cell.

Bias removal. The forecaster first tries to reduce an active bias, in a process called bias removal. If the active bias is positive and $e _ { t }$ lies to the left of its endpoint, predicting that endpoint gives $e _ { t } - p _ { t } < 0$ and reduces the bias. Similarly, if the active bias is negative and $e _ { t }$ lies to the right of its endpoint, predicting that endpoint gives $e _ { t } - p _ { t } > 0$ and reduces its magnitude. Bias removal is performed only when the active bias magnitude exceeds one, so an update of magnitude at most one cannot reverse its sign.

The importance of bias removal is twofold. If bias removal is performed, then the total sum of magnitudes of bias across all cells in all games cannot increase. If bias removal is not performed, every sign that the placement request can legally delete has active bias of magnitude at most one.

Bias placement. If no removal is available, the forecaster uses the first eligible game, in increasing order of i and then $j ,$ , to place bias by predicting the endpoint specified by the cell’s sign. A cell is eligible when its interval contains $e _ { t }$ and its active bias magnitude is below its bias threshold.

If the cell is empty, we will simulate a round of the SPR instance, where the pointer player points towards this cell in the corresponding game, and the labeler will return its sign. Let $\sigma$ be the sign in this cell. Then, when we predict the endpoint corresponding to the chosen sign $\sigma ,$ the bias increment satisfies

$$
0 \leq \sigma ( e _ { t } - p _ { t } ) \leq 2 ^ { - ( i + 1 ) } .\tag{11}
$$

Thus placement adds a small nonnegative amount of bias in the direction chosen by the labeler. If no game is eligible, the forecaster rounds $e _ { t }$ to the nearest prediction point on a fixed grid.

Since $\mathrm { o p t } ( n , t )$ depends on time t, it is important to bound the number of recorded rounds in a game. To do this, we consider a reduced transcript, undoing moves whose signs are deleted by the next remaining move. Except at the smallest coarsest-level threshold, between successive queries in this transcript to the same cell c in the same instance $G ,$ bias in another instance must be rebuilt to its threshold, which takes time. The increment bound (11) therefore limits these transcripts’ lengths, allowing the SPR guarantee to bound the surviving signs.

From bias to calibration. To relate calibration error to bias, we use the decomposition

$$
y _ { t } - p _ { t } = ( e _ { t } - p _ { t } ) + ( y _ { t } - e _ { t } ) .\tag{12}
$$

After summing over rounds at each prediction and applying the triangle inequality, (12) bounds calibration error by a bias term from $e _ { t } - p _ { t }$ and a variance term from $y _ { t } - e _ { t }$

To bound the bias term, note that the bias thresholds ensure an active bias has magnitude at most $2 ^ { j - i } + 1$ , while an inactive bias has magnitude at most one. Thus, if $S _ { G }$ signs remain in $G$ and $N _ { G }$ cells have received placements, the total absolute stored bias in that game is at most

$$
2 ^ { j - i } S _ { G } + 2 N _ { G } .\tag{13}
$$

The first term in (13) is bounded by multiplying the number of remaining signs in each SPR instance by its bias threshold. For the second term, calling a cell in a game with fine intervals requires its parent to reach its bias threshold first. By (11), this requires many placements, limiting both the number of used cells and therefore the number of distinct predictions. The contribution to bias from the fixed grid rounding fallback is $O ( 2 ^ { \tau - i _ { 0 } - h } )$

For the variance term, the increments $y _ { t } - e _ { t }$ have conditional mean zero. The bound on used cells gives at most $L = O ( 2 ^ { i _ { 0 } } + 2 ^ { \tau - i _ { 0 } - h } )$ distinct predictions, including those made by rounding. A martingale second-moment bound and Cauchy–Schwarz bound the expected variance term by $\sqrt { L T }$

Combining these bounds and summing the threshold-weighted SPR guarantees gives the calibration estimate. Setting $h = \tau - 2 i _ { 0 }$ therefore gives

$$
\mathbb { E } [ \mathrm { c a l e r r } ( T ) ] = O \left( 2 ^ { \tau - ( 1 + \varepsilon ) i _ { 0 } } + h 2 ^ { i _ { 0 } } + 2 ^ { ( \tau + i _ { 0 } ) / 2 } \right) .\tag{14}
$$

By setting $i _ { 0 } = \lfloor \tau / ( 3 + 2 \varepsilon ) \rfloor$ , we balance the first and third terms of (14) up to constant factors, which is suficient because the second is smaller. The minimax step removes the forecaster’s access to conditional means, giving (10) and completing the proof of Theorem 5.1. Its exponent depends only on ε, so we choose admissible parameters to maximize $1 - \alpha - \beta$ The explicit choice in Section 6 then gives the claimed calibration exponent.

## 4 An improved labeling strategy for the sign-preservation-withreuse game

In this section, we prove the following theorem, giving a strategy for Player-L whose guarantee depends separately on the numbers of cells and rounds.

Theorem 4.1. For every admissible parameter tuple $( \alpha , \beta , \lambda , \rho , \kappa , \delta )$ in the sense of Theorem $4 . 2 ,$ there is a constant $K > 0$ such that, for all integers $n \geq 1$ and $t \geq 0$

$$
\operatorname { o p t } ( n , t ) \leq K n ^ { \alpha } t ^ { \beta } .
$$

We will prove Theorem 4.1 by establishing a per-sign bound for the recursive strategy in Theorem 4.3, then applying it to the root instance and padding to arbitrary board sizes.

## 4.1 Description of the strategy

Notation. The strategy for Player-L is given by two mutually recursive algorithms, A and B. The calls A.initialize $( l , r , b )$ and B.initialize $\cdot ( l , r , b , \tau )$ create instances with the specified parameters and initialize their local state. Subsequent calls to A.label(s) or B.label(s) process a request at cell s and return its label, except that B.label(s) returns ⊥ when its epoch has ended.

An instance initialized on [l, r] covers $n : = r - l + 1$ cells and has bias $b \in \mathbb { Z } .$ , where $l \in \mathbb N$ $r \in \mathbb N$ , and $l \leq r$ . For an instance I of either A or B, let executionSteps(I) denote the number of calls to I.label that have returned a sign in {−1, 1}; calls returning ⊥ are not counted. An epoch of A is the sequence of requests answered by one of its instances of B. The epoch’s parameter τ counts the number of requests answered in all earlier epochs, and signs placed in earlier epochs that still remain are called inherited signs. Note that executionSteps(A) includes requests from earlier epochs, while executionSteps(B) counts only those in that epoch.

For $\sigma \in \{ - 1 , 1 \}$ , let remainingSigns(I, σ) denote the number of remaining signs of type σ placed in its covered cells in response to requests made to $\mathcal { T } ,$ including recursive calls. For an instance covering at least two cells, a sign is persistent if it is a plus in the left half or a minus in the right half. Notice that persistent signs survive requests in the opposite half but can still be deleted by a request in their own half. We write persistentSigns $( \mathcal { T } , \sigma )$ for the number of such signs among those counted by remainingSigns(I, σ).

We use the “.” notation for the local variables and routines of an instance, as in B.phase and A.label(s). We use the convention sign $( b ) = + 1$ for $b \geq 0$ and $\mathrm { s i g n } ( b ) = - 1$ for $b < 0$ , so a zero bias also returns a legal sign, and positive bias favors plus signs.

Overview of the labeling routines. For now, to simplify the recursion, let n be a power of two. We initialize a root instance $\mathsf { A } _ { 0 }$ with $l = 1 , r = n , b = 0$ . At each time step, after Player-P selects, or requests, an empty cell $s \in [ n ]$ , we delete every minus to the left of s and every plus to its right, and then place the sign returned by $\mathsf { A } _ { 0 } . | \mathsf { a b e l } ( s )$ . Each instance of A covering more than one cell passes the request to its current instance currentB of B.

Algorithm 1 Labeling strategy across epochs (A)   
Require: Parameters $l \in \mathbb { N } , r \in \mathbb { N } , l \leq r , b \in \mathbb { Z } .$ , with $r - l + 1$ a power of two. $( [ l , r ]$ is the covered   
interval and b is the bias.)   
1: function A.initialize $\left( l , r , b \right)$   
2: count $ 0 .$   
3: if $l < r$ then   
4: currentB ← B.initialize $( l , r , b , 0 )$   
5: function A.label(s) $\triangleright \ s \in [ l , r ]$   
6: if $l = r$ then   
7: $\sigma \gets \mathrm { s i g n } ( b )$   
8: else   
9: σ ← currentB.label(s).   
10: if $\sigma = \bot$ then   
11: currentB ← B.initialize $( l , r , b ,$ count).   
12: σ ← currentB.label(s).   
13: count ← count + 1.   
14: return $\sigma .$

In phase 1, this instance passes the request to one of its two children A[0], A[1], according to whether s belongs to the left or right half of its interval. In phase 2, it instead passes the request to a new instance shiftedA of A, which covers the same interval and has bias one larger or one smaller than that of A. The recursive calls eventually reach an instance covering one cell, which returns the sign of its bias. This answer is then returned through the preceding calls to $\mathsf { A } _ { 0 }$ . After a prescribed number of continuation requests, B returns ⊥ on the next request, and A starts a new epoch to answer that request.

Algorithm parameters. The algorithms in Algorithms 1 and 2 depend on three parameters $\rho \in ( 0 , 1 ] , \kappa \in ( 0 , 1 ]$ and $\delta > 0$ . The parameter $\rho$ determines how balanced the two counts must be before B.phase changes to 2, and κ determines the number of requests answered in phase 2. The parameter δ scales the smaller count in the comparison with τ .

Summary of A (Algorithm 1). On input $( l , r , b )$ , A.initialize initializes an instance of B with parameters $( l , r , b , 0 ) { \mathrm { i f } } l < r$ , and sets a counter for the number of rounds, count = executionSteps(A), to zero. $\operatorname { I f } l = r ,$ then A does not initialize a corresponding B instance and A.label(s) chooses sign(b). Otherwise, it calls currentB.label(s) to obtain a sign. If this call returns ⊥, meaning that the current instance of B has terminated, then A initializes a new instance with parameters $( l , r , b ,$ count) and passes the same request to it. After obtaining a sign, A increments count and returns the sign. Thus, A maintains a sequence of instances of B, each of which answers requests until it returns ⊥ or the game ends. Note that at the end of every round involving a non-leaf instance,

$$
\mathsf { e x e c u t i o n S t e p s ( A ) } = \tau + \mathsf { e x e c u t i o n S t e p s ( A . c u r r e n t B ) } .\tag{15}
$$

The variable count is the total number of requests processed so far across all epochs. At a restart, count is exactly the number of requests answered in earlier epochs, so the new epoch receives this number as τ.

Algorithm 2 Two-phase labeling epoch (B)   
Require: Parameters $\overline { { l \in \mathbb { N } , r \in \mathbb { N } , l < r , b \in \mathbb { Z } , \tau \geq 0 } }$ an integer, with $r - l + 1$ a power of two.   
(τ is the total number of requests answered in all earlier epochs of the parent A instance.)   
1: function B.initialize $( l , r , b , \tau )$   
2: $m \gets \left\lfloor { \frac { l + r } { 2 } } \right\rfloor$   
3: $\mathsf { A } [ 0 ] \gets \bar { \mathsf { A } }$ .initialize $( l , m , b )$   
4: $\mathsf { A } [ 1 ] \gets \mathsf { A }$ .initialize $( m + 1 , r , b )$   
5: countHalf[0] ← 0, countHalf[1] $ 0 ,$ phase $ 1 .$   
6: function B.label(s)   
7: if phase = 1 then   
8: half ← 0 if s ≤ m else half $ 1$   
9: countHalf[half] ← countHalf[half] $+ 1 .$   
10: σ ← A[half].label(s).   
11: $t _ { h } \gets \operatorname* { m a x } ^ { . }$ {countHalf[0], countHalf[1]}.   
12: $t _ { l } \gets$ min{countHalf[0], countHalf[1]}.   
13: if $\delta t _ { l } \ge \tau$ and $\rho t _ { h } \le t _ { l }$ then   
14: $L \gets \lfloor \kappa t _ { h } \rfloor , t _ { c } \gets 0 .$   
15: if $L > 0$ then   
16: if half = 0 then   
17: shiftedA ← A.initialize $( l , r , b + 1 )$   
18: else   
19: shiftedA ← A.initialize $( l , r , b - 1 )$   
20: phase $ 2 .$   
21: return σ.   
22: else if phase = 2 then   
23: if $t _ { c } = L$ then ▷ The continuation is complete.   
24: return ⊥.   
25: σ ← shiftedA.label(s).   
26: $t _ { c } \gets t _ { c } + 1$   
27: return σ.

Summary of B (Algorithm 2). On input $( l , r , b , \tau )$ , B.initialize initializes two instances of $\mathsf { A } ,$ denoted by $\mathsf { A } [ 0 ] , \mathsf { A } [ 1 ]$ , covering the left and right halves of $[ l , r ]$ , respectively, both with bias b. It also maintains the two phase 1 request counts and denotes their minimum by $t _ { l }$ and their maximum by $t _ { h }$ . In phase 1, each request is answered by the child covering its cell. After obtaining this answer, B checks whether the smaller count $t _ { l }$ is at least $\tau / \delta$ and at least $\rho t _ { h }$ , where $t _ { h }$ is the larger count. When both conditions hold, a trigger occurs. It sets $L = \lfloor \kappa t _ { h } \rfloor$ and enters phase 2 for L requests. If $L > 0 .$ , after obtaining the triggering request’s answer from the old child, it initializes shiftedA on the full interval $[ l , r ]$ , with bias $b + 1$ if the trigger was in the left half, and $b - 1$ otherwise. The next L requests, which are the continuation of B, are answered by shiftedA, after which B returns ⊥ on the following request. If $L = 0$ , it returns ⊥ on the first request after entering phase 2. Note that the game may end during either phase, in which case the instance need never return ⊥.

## 4.2 Parameters and the labeling lemma

Analysis parameters. The exponents $\alpha > 0$ and $\beta \in ( 0 , 1 )$ describe how the number of signs remaining depends on space and time, respectively. The constant $\lambda \in ( 1 , 2 ]$ controls how changing bias afects the bounds, while $0 < K _ { p } < 1$ and $K _ { t } > K _ { p }$ control, respectively, the numbers of persistent and total inherited signs. The constant $C > 0$ in the bound will be chosen in the proof. We write $t = \mathsf { e x e c u t i o n S t e p s } ( \mathsf { A } )$ and take n to be a power of two, so that each split gives two intervals of equal size. Our goal is to choose the parameters with $\alpha + \beta$ as small as possible.

For an instance of A with bias b covering $n \geq 2$ cells, the coeficient in the bound for signs of type $\sigma \in \{ - 1 , 1 \}$ for each of the two children of a B instance is $C \lambda ^ { b \sigma } \left( \frac { n } { 2 } \right) ^ { \alpha }$ When phase 2 begins, the new instance of A covers n cells and has bias $b + 1$ or $b - 1$ . Relative to $C \bar { \lambda ^ { b \sigma } } \left( \frac { n } { 2 } \right) ^ { \alpha }$ for the original children, its contribution is multiplied by one of $\lambda ^ { - 1 } 2 ^ { \alpha }$ or $\lambda 2 ^ { \alpha }$ . We collect all the inequalities needed by the analysis in the following definition, and only later give values satisfying these inequalities in Theorem 6.1.

Definition 4.2. A tuple $( \alpha , \beta , \lambda , \rho , \kappa , \delta )$ is admissible if

$$
\alpha > 0 , \qquad 0 < \beta < 1 , \qquad 1 < \lambda \le 2 , \qquad 0 < \rho \le 1 , \qquad 0 < \kappa \le 1 , \qquad \delta > 0 , \qquad\tag{16}
$$

and there exist real constants $F , K _ { p } , K _ { t } , M , D , P , Q$ satisfying all of the following conditions.

Here, F is the maximum number of rounds handled by the base case for the induction on time. $K _ { p } , K _ { t }$ are coeficients that bound inherited persistent signs and total signs, respectively. M bounds epochs in phase 1, D bounds epochs in all phases, and $P , Q$ control persistent and total signs at the end of epochs, respectively. The tuple is admissible if:

1. The constants lie in the ranges

$$
\begin{array} { r } { F \geq 1 , \qquad 0 < K _ { p } < 1 , \qquad K _ { t } > K _ { p } , } \\ { K _ { p } \leq M \leq D , \qquad P > 0 , \qquad Q > 0 . } \end{array}\tag{17}
$$

2. To close the induction for short executions, persistent signs, and total signs, we have the bounds

$$
\begin{array} { c } { { P ( 1 + F ^ { - 1 } ) + F ^ { - \beta } < K _ { p } , ~ Q ( 1 + F ^ { - 1 } ) + F ^ { - \beta } < K _ { t } , } } \\ { { { } } } \\ { { D + F ^ { - \beta } < 2 ^ { \alpha } . } } \end{array}\tag{18}
$$

3. If an execution stops during phase 1, the inherited signs and the two half-interval instances are bounded as follows for all positive x and $y \colon$

$$
K _ { t } x ^ { \beta } + y ^ { \beta } \leq D ( x + y ) ^ { \beta } .\tag{19}
$$

If it stops during the continuation in phase 2, the inherited signs and the continuing instance are bounded as follows for all positive x and $y \colon$

$$
M x ^ { \beta } + \lambda ^ { - 1 } 2 ^ { \alpha } y ^ { \beta } \leq D ( x + y ) ^ { \beta } .\tag{20}
$$

4. If an instance is in phase 1, the inherited persistent signs and the signs from both halves are bounded. Specifically, for all nonnegative integers τ , $t _ { h } , t _ { l }$ with $t _ { l } \le t _ { h }$ , if $\delta t _ { l } < \tau$ or $t _ { l } < \rho t _ { h }$ 2 then

$$
K _ { p } \tau ^ { \beta } + t _ { h } ^ { \beta } + t _ { l } ^ { \beta } \leq M ( \tau + t _ { h } + t _ { l } ) ^ { \beta } .\tag{21}
$$

If an instance has just switched into phase 2, the same bounds hold with an additional 1. Specifically, if instead $1 \leq t _ { l } \leq t _ { h } , \delta t _ { l } \geq \tau , t _ { l } \geq \rho t _ { h }$ , and either $\delta ( t _ { l } - 1 ) < \tau { \mathrm { ~ o r ~ } } t _ { l } - 1 < \rho t _ { h }$ 2 then

$$
K _ { p } \tau ^ { \beta } + t _ { h } ^ { \beta } + t _ { l } ^ { \beta } \leq M ( \tau + t _ { h } + t _ { l } ) ^ { \beta } + 1 .\tag{22}
$$

5. If an instance is triggered into phase 2, then $( z , r ) \in \mathcal { T }$ , where $\begin{array} { r } { z = \frac { \tau } { t _ { h } } } \end{array}$ and $\begin{array} { r } { r = \frac { \operatorname* { m a x } \{ \tau / \delta , \rho t _ { h } \} } { t _ { h } } } \end{array}$ for integers $1 \leq t _ { l } \leq t _ { h }$ , and

$$
\mathcal T : = \{ ( z , r ) : 0 \leq z \leq \delta r , \ \rho \leq r \leq 1 , \ r = \rho \ \mathrm { o r } \ z = \delta r \} .\tag{23}
$$

Then the numbers of persistent pluses, persistent minuses, and total signs, respectively, are bounded by

$$
K _ { p } z ^ { \beta } + 1 + \lambda ^ { - 1 } 2 ^ { \alpha } \kappa ^ { \beta } \leq P ( z + 1 + r + \kappa ) ^ { \beta } ,\tag{24}
$$

$$
K _ { p } z ^ { \beta } + r ^ { \beta } + \lambda 2 ^ { \alpha } \kappa ^ { \beta } \leq P ( z + 1 + r + \kappa ) ^ { \beta } ,\tag{25}
$$

$$
K _ { \cal P } z ^ { \beta } + 1 + r ^ { \beta } + \lambda ^ { - 1 } 2 ^ { \alpha } \kappa ^ { \beta } \leq Q ( z + 1 + r + \kappa ) ^ { \beta } .\tag{26}
$$

6. When a continuation finishes and an epoch ends, the numbers of persistent minuses, persistent pluses, and total signs are bounded. Specifically, for all nonnegative integers $\tau , t _ { h } , t _ { l }$ satisfying the count conditions of (22), set $L = \lfloor \kappa t _ { h } \rfloor$ and $t = \tau + t _ { l } + t _ { h } + L$ . Whenever $t \geq F$ 2

$$
K _ { p } \tau ^ { \beta } + t _ { l } ^ { \beta } + \lambda 2 ^ { \alpha } L ^ { \beta } < K _ { p } t ^ { \beta } ,\tag{27}
$$

$$
K _ { p } \tau ^ { \beta } + t _ { h } ^ { \beta } + \lambda ^ { - 1 } 2 ^ { \alpha } L ^ { \beta } < K _ { p } t ^ { \beta } ,\tag{28}
$$

$$
K _ { p } \tau ^ { \beta } + t _ { h } ^ { \beta } + t _ { l } ^ { \beta } + \lambda ^ { - 1 } 2 ^ { \alpha } L ^ { \beta } < K _ { t } t ^ { \beta } .\tag{29}
$$

At any point during the continuation, the persistent count of the lesser sign remains within its induction bound, so for every real $0 \leq t _ { c } \leq L$ 2

$$
\begin{array} { r } { K _ { p } \tau ^ { \beta } + t _ { l } ^ { \beta } + \lambda 2 ^ { \alpha } t _ { c } ^ { \beta } \leq K _ { p } ( \tau + t _ { l } + t _ { h } + t _ { c } ) ^ { \beta } . } \end{array}\tag{30}
$$

Lemma 4.3. Fix an admissible tuple $( \alpha , \beta , \lambda , \rho , \kappa , \delta )$ as in Theorem 4.2. There exists a constant $C > 0$ , depending only on this tuple, such that every instance A with bias $b \in \mathbb { Z }$ , covering a power-of-two number n of cells, satisfies the following bound. For $t : =$ executionSteps(A) and each $\sigma \in \{ - 1 , 1 \}$ ,

$$
r e m a i n i n g S i g n s ( A , \sigma ) \leq C \lambda ^ { b \sigma } n ^ { \alpha } t ^ { \beta } .\tag{31}
$$

Proof overview of Theorem 4.3. Fix any admissible tuple $( \alpha , \beta , \lambda , \rho , \kappa , \delta )$ and constants $F , K _ { p } ,$ $K _ { t } , M , D , P , Q$ from Theorem 4.2. Take $C \geq F ^ { 2 } / K _ { p }$ . We prove the bound (31) in Theorem 4.3 by an outer induction on cells $n ,$ where n is a power of two, and an inner strong induction on requests t. For an instance $\mathcal { T }$ of A with bias $b ^ { \prime } ,$ , covering $n ^ { \prime }$ cells and having answered $t ^ { \prime }$ requests, the space hypothesis is, for $n ^ { \prime } < n$   
，

$$
\mathsf { r e m a i n i n g S i g n s } ( \mathbb { Z } , \sigma ) \leq C \lambda ^ { b ^ { \prime } \sigma } ( n ^ { \prime } ) ^ { \alpha } ( t ^ { \prime } ) ^ { \beta } .\tag{32}
$$

At the fixed space $n ,$ the time hypothesis is, for $t ^ { \prime } < t .$

$$
\mathsf { r e m a i n i n g S i g n s } ( \mathbb { Z } , \sigma ) \le C \lambda ^ { b ^ { \prime } \sigma } n ^ { \alpha } ( t ^ { \prime } ) ^ { \beta } .\tag{33}
$$

Both hypotheses use the same constant $C$ and hold for all biases $b ^ { \prime } \in \mathbb { Z }$ and sign types $\sigma \in \{ - 1 , + 1 \}$ Assuming both hypotheses, we fix a parent instance A with bias b on $n \geq 2$ cells and t requests, noting that the base cases $t = 0$ and $n = 1$ are immediate.

• If $t \leq F$ , we show that $C$ is large enough to establish the inductive claim by Theorem 4.5. Thus, it sufices to only consider $t > F$

• If only one half has been visited in the current epoch, the epoch is still in phase 1, and both persistent and non-persistent inherited signs may remain. We bound these with the total sign coeficient $K _ { t }$ by Theorem 4.6, giving $\ddot { K _ { t } \tau ^ { \beta } } + t _ { h } ^ { \bar { \beta } } \leq D t ^ { \beta }$

• Once both halves have been visited in the current epoch, the only inherited signs that remain are persistent, so we can sharpen the bound on inherited signs to the smaller persistent sign coeficient $K _ { p }$ by Theorem 4.6. If the epoch is still in phase 1, then $t _ { l } < \rho t _ { h }$ or $\delta t _ { l } < \tau$ , so admissibility gives $K _ { p } \tau ^ { \beta } + t _ { h } ^ { \beta } + t _ { l } ^ { \beta } \leq M t ^ { \beta }$

• If the epoch has entered phase 2, we account for the triggering deletion and combine the contributions of both the child instances and the shifted bias instance from the continuation, as in Theorem 4.4. Using the inherited bounds from Theorem 4.6 gives coeficients $D t ^ { \beta } + 1$ and $K _ { p } t ^ { \beta }$ for pluses and minuses, respectively, when the trigger is on the right; a left-half trigger exchanges the sign types.

• Since $K _ { p } \leq M \leq D$ , all three cases give $C \lambda ^ { b \sigma } ( n / 2 ) ^ { \alpha } ( D t ^ { \beta } + 1 )$ , which is suficient since the given inequality $D + F ^ { - \beta } < 2 ^ { \alpha }$ closes the induction.

## 4.3 Lemmas in the proof of Theorem 4.3

We begin by bounding the number of total signs and persistent signs in a continuation instance.

Lemma 4.4. Under the inductive setup, suppose an epoch of A, with instance B and parameter $\tau ,$ starts with at most $C \lambda ^ { b \sigma } ( n / 2 ) ^ { \alpha } K _ { p } \tau ^ { \beta }$ inherited persistent signs of each type, where $K _ { p }$ is the inherited persistent-sign coeficient from Theorem 4.2. Suppose B.phase = 2 is triggered by the right half. Let $t _ { h } , \ t _ { l }$ be the larger and smaller counts at the trigger, and set $L = \lfloor \kappa t _ { h } \rfloor$ . During phase 2, after $0 \leq t _ { c } \leq L$ continuation requests, we have

$$
r e m a i n i n g S i g n s ( A , - 1 ) \leq C \lambda ^ { - b } \left( \frac { n } { 2 } \right) ^ { \alpha } \left( K _ { p } \tau ^ { \beta } + t _ { l } ^ { \beta } + \lambda 2 ^ { \alpha } t _ { c } ^ { \beta } \right) ,
$$

$$
p e r s i s t e n t S i g n s ( A , + 1 ) \leq C \lambda ^ { b } \left( \frac { n } { 2 } \right) ^ { \alpha } \left( K _ { p } \tau ^ { \beta } + t _ { h } ^ { \beta } + \lambda ^ { - 1 } 2 ^ { \alpha } t _ { c } ^ { \beta } \right) ,
$$

$$
r e m a i n i n g S i g n s ( A , + 1 ) \leq C \lambda ^ { b } \left( \frac { n } { 2 } \right) ^ { \alpha } \left( K _ { p } \tau ^ { \beta } + t _ { h } ^ { \beta } + t _ { l } ^ { \beta } + \lambda ^ { - 1 } 2 ^ { \alpha } t _ { c } ^ { \beta } \right) .
$$

For a left-half trigger, swap the two sign types and exchange the factors $\lambda ^ { b }$ and $\lambda ^ { - b }$

Proof. B.phase = 2 implies that both halves have been visited. In particular, the first visit to the left half deletes all right-half pluses from earlier epochs, and the first visit to the right half deletes all earlier left-half minuses. Thus, only inherited persistent signs can remain from earlier epochs.

Notice that the right count cannot be larger than the left count immediately after the phase changes. Otherwise, it was already at least the left count before this request, so the smaller count was unchanged and the larger count could only increase. Neither condition for changing the phase could therefore become true for the first time. Thus, $t _ { h }$ denotes the count on the left and $t _ { l }$ denotes the count on the right.

By the inductive hypothesis on space, the signs of type σ placed by $\mathsf { B } . \mathsf { A } [ 0 ]$ and $\mathsf { B } . \mathsf { A } [ 1 ]$ that remain are at most $C \lambda ^ { b \sigma } \left( \frac { n } { 2 } \right) ^ { \alpha } t _ { h } ^ { \beta }$ and $C \lambda ^ { b \sigma } \left( \frac { n } { 2 } \right) ^ { \alpha } t _ { l } ^ { \beta }$ , respectively, because both these instances cover ${ \frac { n } { 2 } } < n$ cells. After B.phase changes to $^ { 2 , }$ neither instance receives further requests, so these contributions cannot increase.

If $t _ { c } > 0$ , then $L > 0$ and B.shiftedA has been initialized. Since the triggering request is on the right, this instance has bias $b - 1$ and covers n cells. The parent A has processed at least $t _ { h } + t _ { l } + t _ { c } > t _ { c }$ requests, so the inductive hypothesis on time applies to B.shiftedA. Its plus and minus counts are therefore at most

$$
\begin{array} { c } { { C \lambda ^ { b - 1 } n ^ { \alpha } t _ { c } ^ { \beta } = C \lambda ^ { b } \left( \displaystyle \frac n 2 \right) ^ { \alpha } \lambda ^ { - 1 } 2 ^ { \alpha } t _ { c } ^ { \beta } , } } \\ { { C \lambda ^ { - ( b - 1 ) } n ^ { \alpha } t _ { c } ^ { \beta } = C \lambda ^ { - b } \left( \displaystyle \frac n 2 \right) ^ { \alpha } \lambda 2 ^ { \alpha } t _ { c } ^ { \beta } . } } \end{array}
$$

For $t _ { c } = 0$ , both contributions are zero, whether or not the continuation has been initialized.

To bound the number of minuses, note that the right request on which the phase changes deletes every minus placed by B.A[0]. It sufices to consider the minuses placed by $\bar { \mathsf { B } . A } [ 1 ] ~ ( C \lambda ^ { - \overline { { b } } } \left( \frac { n } { 2 } \right) ^ { \alpha } t _ { l } ^ { \beta } )$ the inherited persistent minuses $( C \lambda ^ { - b } \left( \frac { n } { 2 } \right) ^ { \alpha } K _ { p } \tau ^ { \beta } )$ , and the minuses placed by B.shiftedA $( C \lambda ^ { - b }$ $( \frac { n } { 2 } ) ^ { \alpha } \lambda 2 ^ { \alpha } t _ { c } ^ { \beta } )$ . Summing these bounds gives the desired inequality on remainingSigns $( \mathsf { A } , - 1 )$

To bound the number of pluses, we first bound the number of persistent pluses. It sufices to consider the pluses placed by $\begin{array} { r } { \dot { \mathsf { B } } . \mathsf { A } [ 0 ] ( C \lambda ^ { b } \left( \frac { n } { 2 } \right) ^ { \alpha } t _ { h } ^ { \beta } ) } \end{array}$ , the inherited persistent pluses $\begin{array} { r l r } {  { ( C \lambda ^ { b } ( \frac { n } { 2 } ) ^ { \alpha } K _ { p } \tau ^ { \beta } ) } } \end{array}$ and the pluses placed by B.shiftedA $( C \bar { \lambda ^ { b } } \left( \frac { n } { 2 } \right) ^ { \alpha } \lambda ^ { - 1 } 2 ^ { \alpha } t _ { c } ^ { \beta } )$ . Counting all pluses from the last instance also bounds its persistent pluses. Summing these bounds gives the desired inequality on persistentSigns $( \mathsf { A } , + 1 )$ ). To obtain the inequality on remainingSigns(A, +1), we also include the pluses placed by $\mathsf { B . A [ 1 ] } \ ( C \lambda ^ { b } \left( \frac { n } { 2 } \right) ^ { \alpha } t _ { l } ^ { \beta } )$ □

To induct on time, we first handle instances which have processed at most F requests. The cutof F and the choice $C \geq F ^ { 2 } / K _ { p }$ come from the inductive setup.

Lemma 4.5. Take the cutof F from Theorem $4 . 2$ and $C \geq F ^ { 2 } / K _ { p }$ as in the inductive setup. Let A be an instance with bias b covering a power-of-two number n of cells, and let $t = e x e c u t i o n S t e p s ( A ) \leq$ $F$ . Then, for each $\sigma \in \{ - 1 , 1 \}$ ,

$$
r e m a i n i n g S i g n s ( A , \sigma ) \leq C \lambda ^ { b \sigma } n ^ { \alpha } t ^ { \beta } .
$$

Proof. Since $K _ { p } < 1$ , our choice of C gives $C \geq F ^ { 2 } . { \mathrm { ~ I f ~ } } t = 0$ or the instance has returned no sign of type $\sigma _ { \mathrm { : } }$ , the desired bound is immediate. Otherwise, we first show that

$$
C \lambda ^ { b \sigma } \geq \frac { F ^ { 2 } } { t } .
$$

Fix a request on which the given instance returns $\sigma .$ . Write $\mathsf { A } _ { 0 } , \mathsf { A } _ { 1 } , \mathsf { \Omega } \ldots , \mathsf { A } _ { k }$ for the successive instances whose answers are passed back to produce this $\sigma _ { \mathrm { { ; } } }$ , where $\mathsf { A } _ { 0 }$ is the given instance and $\mathsf { A } _ { k }$ covers one cell. For each $i < k$ , let $\mathsf { B } _ { i }$ be the instance called by $\mathsf { A } _ { i }$ that returns $\sigma$ . Thus, $\mathsf { B } _ { i }$ obtains its answer from $\mathsf { A } _ { i + 1 }$ . Evaluate each instance’s number of processed requests just after this request.

If $\mathsf { B } _ { i } . \mathrm { p h a s e } = 1$ when it calls $\mathsf { A } _ { i + 1 }$ , then $\mathsf { A } _ { i + 1 }$ is one of $\mathsf { B } _ { i } . \mathsf { A } [ 0 ]$ and $\mathsf { B } _ { i } . \mathsf { A } [ 1 ]$ ]. It has the same bias as $\mathsf { A } _ { i }$ and has processed at most as many requests. Otherwise, $\mathsf { B } _ { i } . \mathrm { p h a s e } = 2$ and $\mathsf { A } _ { i + 1 } = \mathsf { B } _ { i }$ .shiftedA, which was initialized with bias difering by one from that of $\mathsf { A } _ { i }$ when $\mathsf { B } _ { i } . \mathrm { p h a s e }$ changed from 1 to 2. Let $t _ { h } , t _ { l }$ be the counts at that change, and let $t _ { c }$ be the number of requests processed by $\mathsf { A } _ { i + 1 }$ Then $t _ { c } \le L = \lfloor \kappa t _ { h } \rfloor \le t _ { h }$ , whereas $\mathsf { A } _ { i }$ has processed at least $t _ { h } + t _ { l } + t _ { c } \geq 2 t _ { c }$ requests. Thus, each bias change reduces the number of processed requests by at least a factor of two. If there are j bias changes among these instances, $\mathsf { A } _ { k }$ has processed at least one request and at most $t 2 ^ { - j }$ requests, so $j \leq \log _ { 2 } t$

Let b<sup>′</sup> be the bias of $\mathsf { A } _ { k }$ . Since each bias change is by one, $| b ^ { \prime } - b | \leq j$ . The instance $\mathsf { A } _ { k }$ returns $\sigma = \mathrm { s i g n } ( b ^ { \prime } )$ , so $b ^ { \prime } \sigma \geq 0$ . Consequently, bσ $\geq - j \geq - \log _ { 2 } t$ , and

$$
\lambda ^ { b \sigma } \geq \lambda ^ { - \log _ { 2 } t } \geq 2 ^ { - \log _ { 2 } t } = \frac { 1 } { t } .
$$

The second inequality uses $\lambda < 2$ . Multiplying by $C \geq F ^ { 2 }$ gives the claimed inequality.

Now, since $n ^ { \alpha } \ge 1 , t ^ { \beta } \ge 1$ and $\begin{array} { r } { \frac { F ^ { 2 } } { t } \geq t \mathrm { f o r } t \leq F } \end{array}$ , we have $C \lambda ^ { b \sigma } n ^ { \alpha } t ^ { \beta } \geq t$ . This proves the desired bound, as the total number of signs preserved cannot be more than the total number of requests processed by the instance. □

Lemma 4.6. Under the inductive setup, let $K _ { p }$ and $K _ { t }$ be the inherited persistent- and total-sign coeficients. At the start of every epoch with parameter $\tau ,$ the following inequalities hold for each $\sigma \in \{ - 1 , 1 \}$

$$
p e r s i s t e n t S i g n s ( A , \sigma ) \leq C \lambda ^ { b \sigma } \left( \frac { n } { 2 } \right) ^ { \alpha } K _ { p } \tau ^ { \beta } ,
$$

$$
r e m a i n i n g S i g n s ( A , \sigma ) \leq C \lambda ^ { b \sigma } \left( \frac { n } { 2 } \right) ^ { \alpha } K _ { t } \tau ^ { \beta } .
$$

Proof. At the start of the first epoch, A has placed no signs, so both bounds hold with $\tau = 0$ Suppose the bounds hold at the start of an epoch with parameter $\tau ,$ and that the next epoch begins. Let $t _ { h } , \ t _ { l }$ be the completed epoch’s counts when B.phase changes from 1 to 2, and set $L = \lfloor \kappa t _ { h } \rfloor$ . The epoch answers $t _ { h } + t _ { l } + L$ requests before returning ⊥ on the next request. Thus, the next epoch has parameter $\boldsymbol { u } = \boldsymbol { \tau } + \boldsymbol { t } _ { h } + \boldsymbol { t } _ { l } + \boldsymbol { L } .$ , by (15), and A has answered exactly u requests before that next epoch begins. Any deletions on the request that starts the new epoch can only reduce the inherited counts. Each continuation used to bound this completed epoch has fewer than $u \leq t$ requests, so the fixed time hypothesis applies.

Suppose first that $u \leq F$ . If A has returned no sign of type $\sigma ,$ its count is zero. Otherwise, the argument in the proof of Theorem 4.5 gives $\lambda ^ { b \sigma } \geq \frac { 1 } { u }$ . Since $n \geq 2$ and $\begin{array} { r } { C \geq \frac { F ^ { 2 } } { K _ { p } } } \end{array}$ , we have

$$
C \lambda ^ { b \sigma } \left( \frac n 2 \right) ^ { \alpha } K _ { p } u ^ { \beta } \geq \frac { C K _ { p } } u \geq \frac { F ^ { 2 } } u \geq u .
$$

The total number of signs of type σ placed by A that remain is at most u. This proves both bounds for the next epoch, since persistent signs are a subset of all remaining signs and $K _ { p } < K _ { t }$

Suppose instead that $u > F$ . First consider the case where the request on which B.phase changes from 1 to 2 is on the right. Applying Theorem 4.4 at $t _ { c } = L$ and (27), (28), and (29) gives

$$
\begin{array} { r l r } {  { \mathsf { r e m a i n i n g S i g n s } ( \mathsf { A } , - 1 ) \le C \lambda ^ { - b } ( \displaystyle \frac { n } { 2 } ) ^ { \alpha } ( K _ { p } \tau ^ { \beta } + t _ { l } ^ { \beta } + \lambda 2 ^ { \alpha } L ^ { \beta } ) } } \\ & { } & \\ & { } & { \le C \lambda ^ { - b } ( \displaystyle \frac { n } { 2 } ) ^ { \alpha } K _ { p } u ^ { \beta } , } \end{array}
$$

$$
\begin{array} { r l r } {  { \mathsf { p e r s i s t e n t S i g n s } ( \mathsf { A } , + 1 ) \le C \lambda ^ { b } ( \frac n 2 ) ^ { \alpha } ( K _ { p } \tau ^ { \beta } + t _ { h } ^ { \beta } + \lambda ^ { - 1 } 2 ^ { \alpha } L ^ { \beta } ) } } \\ & { } & \\ & { } & { \le C \lambda ^ { b } ( \frac n 2 ) ^ { \alpha } K _ { p } u ^ { \beta } , } \end{array}
$$

$$
\begin{array} { r l r } {  { \mathsf { r e m a i n i n g S i g n s } ( \mathsf { A } , + 1 ) \le C \lambda ^ { b } ( \frac { n } { 2 } ) ^ { \alpha } ( K _ { p } \tau ^ { \beta } + t _ { h } ^ { \beta } + t _ { l } ^ { \beta } + \lambda ^ { - 1 } 2 ^ { \alpha } L ^ { \beta } ) } } \\ & { } & { \le C \lambda ^ { b } ( \frac { n } { 2 } ) ^ { \alpha } K _ { t } u ^ { \beta } . } \end{array}
$$

Since $K _ { p } < K _ { t }$ and persistent $\mathtt { S i g n s ( A , - 1 ) } \le \mathtt { r e m a i n i n g S i g n s ( A , - 1 ) }$ , the required bounds for both sign types hold. If the trigger is on the left half instead, exchange the two sign types and the same arguments apply.

## 4.4 Completing the labeling induction

It remains to combine the bounds above for an instance whose current epoch need not have ended.

Proof of Theorem 4.3. Fix an admissible parameter tuple and take F and C as chosen in the inductive setup. We induct on space (the power of two n), and strongly on time (t) for each fixed $n ,$ uniformly over all biases and histories allowed in the lemma.

For $t = 0$ , the count is zero. For $n = 1$ , at most one sign remains, and any returned type $\sigma$ satisfies bσ $\geq 0$ , so $C \lambda ^ { b \sigma } t ^ { \beta } \ge 1$ when $t \geq 1$

Now suppose $n \geq 2 .$ Theorem 4.5 handles $t \leq F$ , so assume $t > F$ . The inductive hypothesis on space applies to both $\mathsf { B } . \mathsf { A } [ 0 ]$ and $\mathsf { B } . \mathsf { A } [ 1 ]$ , since they cover fewer cells, and the inductive hypothesis on time applies to each B.shiftedA, since it processes fewer requests. Thus, Theorems 4.4 and 4.6 apply.

Let B = A.currentB, with $t _ { c } = 0$ in phase 1. By (15), we have $t = \tau + t _ { h } + t _ { l } + t _ { c }$ . There are three cases.

Case 1: only one half of B has been visited. Then B.phase = 1, and only the instance corresponding to the visited half has placed signs in this epoch. Both persistent and non-persistent signs from earlier epochs may remain. Here $t _ { l } = 0$ and $t = \tau + t _ { h }$ . We have

$$
\begin{array} { r c l } { { \mathsf { r e m a i n i n g S i g n s } ( \mathsf { A } , \sigma ) \leq C \lambda ^ { b \sigma } \left( \displaystyle \frac n 2 \right) ^ { \alpha } \left( K _ { t } \tau ^ { \beta } + t _ { h } ^ { \beta } \right) } } \\ { { \leq C \lambda ^ { b \sigma } \left( \displaystyle \frac n 2 \right) ^ { \alpha } D t ^ { \beta } . } } \end{array}
$$

The first inequality follows from Theorem 4.6 and the inductive hypothesis on space; the second follows from (19).

Case 2: both halves of B have been visited, and B.phase = 1. Only persistent signs from earlier epochs remain. Since phase 1 has not ended, $\delta t _ { l } < \tau$ or $t _ { l } < \rho t _ { h }$ . With $t = \tau + t _ { h } + t _ { l }$ , we have

$$
\begin{array} { c } { \mathsf { r e m a i n i n g S i g n s } ( \mathsf { A } , \sigma ) \leq C \lambda ^ { b \sigma } \left( \displaystyle \frac n 2 \right) ^ { \alpha } \left( K _ { p } \tau ^ { \beta } + t _ { h } ^ { \beta } + t _ { l } ^ { \beta } \right) } \\ { \leq C \lambda ^ { b \sigma } \left( \displaystyle \frac n 2 \right) ^ { \alpha } M t ^ { \beta } . } \end{array}
$$

The first inequality follows from Theorem 4.6 and the inductive hypothesis on space; the second follows from (21).

Case 3: B.phase = 2. Let $t _ { h } .$ , t<sub>l</sub> be the counts when the phase changes from 1 to 2, and let $t _ { c }$ be the continuation count, with $t _ { c } = 0$ if $L = 0$ . Then $t = \tau + t _ { h } + t _ { l } + t _ { c }$ . Since $t _ { c } \leq L$

$$
\tau + t _ { h } + t _ { l } + L \geq t > F ,
$$

so (30) applies.

Suppose the request on which the phase changes is on the right. For the plus signs, we have

$$
\begin{array} { l } { \mathsf { r e m a i n i n g S i g n s } ( \mathsf { A } , + 1 ) \leq C \displaystyle \lambda ^ { b } \left( \frac { n } { 2 } \right) ^ { \alpha } \left( K _ { p } \tau ^ { \beta } + t _ { h } ^ { \beta } + t _ { l } ^ { \beta } + \lambda ^ { - 1 } 2 ^ { \alpha } t _ { c } ^ { \beta } \right) } \\ { \leq C \displaystyle \lambda ^ { b } \left( \frac { n } { 2 } \right) ^ { \alpha } \left( M ( \tau + t _ { h } + t _ { l } ) ^ { \beta } + \lambda ^ { - 1 } 2 ^ { \alpha } t _ { c } ^ { \beta } + 1 \right) } \\ { \leq C \displaystyle \lambda ^ { b } \left( \frac { n } { 2 } \right) ^ { \alpha } \left( D t ^ { \beta } + 1 \right) . } \end{array}
$$

The first inequality follows from Theorems 4.4 and 4.6, the second from (22), and the third from (20). For the minus signs, we have

$$
\begin{array} { r l } & { \mathsf { r e m a i n i n g S i g n s } ( \mathsf { A } , - 1 ) \leq C \lambda ^ { - b } \left( \displaystyle \frac { n } { 2 } \right) ^ { \alpha } \left( K _ { p } \tau ^ { \beta } + t _ { l } ^ { \beta } + \lambda 2 ^ { \alpha } t _ { c } ^ { \beta } \right) } \\ & { \qquad \leq C \lambda ^ { - b } \left( \displaystyle \frac { n } { 2 } \right) ^ { \alpha } K _ { p } t ^ { \beta } . } \end{array}
$$

The first inequality follows from Theorems 4.4 and 4.6; the second follows from (30). A left-half trigger exchanges the roles of the two sign types.

To finish the proof, use $D \geq M \geq K _ { p }$ from (17). Combining the three cases gives

$$
\mathsf { r e m a i n i n g S i g n s } ( \mathsf { A } , \sigma ) \leq C \lambda ^ { b \sigma } \left( \frac { n } { 2 } \right) ^ { \alpha } \left( D t ^ { \beta } + 1 \right) .
$$

Since $t > F$ and $0 < \beta < 1 , D t ^ { \beta } + 1 = \left( D + t ^ { - \beta } \right) t ^ { \beta } \leq ( D + F ^ { - \beta } ) t ^ { \beta }$ . Therefore,

$$
\begin{array} { l } { \mathsf { r e m a i n i n g S i g n s } ( \mathsf { A } , \sigma ) \leq C \lambda ^ { b \sigma } \left( \displaystyle \frac n 2 \right) ^ { \alpha } \left( D t ^ { \beta } + 1 \right) } \\ { \qquad \leq C \lambda ^ { b \sigma } n ^ { \alpha } t ^ { \beta } \displaystyle \frac { D + F ^ { - \beta } } { 2 ^ { \alpha } } } \\ { \qquad \leq C \lambda ^ { b \sigma } n ^ { \alpha } t ^ { \beta } , } \end{array}
$$

where the last inequality uses (18). This completes the inner induction on time and hence the outer induction on space. □

We now deduce Theorem 4.1 from Theorem 4.3 by summing the bounds for each sign type at the root and padding the board to handle arbitrary sizes.

Proof of Theorem 4.1. For a power-of-two board size, summing the bounds for each sign type in Theorem 4.3 at the root, whose bias is $b = 0$ , gives

$$
\begin{array} { r } { \mathrm { o p t } ( n , t ) \le C ( \lambda ^ { b } + \lambda ^ { - b } ) n ^ { \alpha } t ^ { \beta } = 2 C n ^ { \alpha } t ^ { \beta } . } \end{array}
$$

For arbitrary $n ,$ pad the board to a power of two $N \in [ n , 2 n )$ and request only the original n cells. Thus, we may take $K = 2 ^ { 1 + \alpha } C$ □

## 5 From sign-preservation bounds to calibration

In this section, we prove the following theorem, converting separate space and time bounds for SPR into a calibration guarantee.

Theorem 5.1. Let $\alpha \in ( 0 , 1 ) , \beta \in ( 0 , 1 )$ and $\varepsilon : = 1 - \alpha - \beta > 0$ . Suppose $\mathrm { o p t } ( n , t ) = O ( n ^ { \alpha } t ^ { \beta } )$ for all integers $n \geq 1$ and $t \geq 0$ . Then, for every horizon $T \geq 1$ , there exists a randomized forecaster whose expected calibration error against any adversary satisfies

$$
\mathbb { E } [ \mathrm { c a l e r r } ( T ) ] = O \left( T ^ { \frac { 2 + \varepsilon } { 3 + 2 \varepsilon } } \right) = O \left( T ^ { \frac { 2 } { 3 } - \frac { \varepsilon } { 9 + 6 \varepsilon } } \right) .
$$

First, we establish the scale-sum estimate in Theorem 5.2. Then, we prove Theorem 5.1 by applying the SPR bound from Section 4.

## 5.1 Description of the strategy

Conditional means. Our strategy uses the conditional mean approach developed by Hart [Har25] and Dagan et al. [DDF<sup>+</sup>25]. Fix any adversary and let $e _ { t } : = \mathbb { E } [ y _ { t } \mid H _ { t } ]$ , where $H _ { t }$ is the history before round t. For the construction, the forecaster observes $e _ { t }$ before predicting. We use deterministic labeling strategies and fixed tie-breaking rules, so the forecaster’s prediction is a function of $H _ { t }$ . Specifically, all loops run in increasing order, ties between grid points are resolved by choosing the smallest, and ties between cells are resolved by choosing the leftmost. After predicting, it observes $y _ { t }$ and proceeds to the next round.

Notation. The forecasting strategy maintains several instances of the SPR game. Each cell represents an interval of possible conditional means, and the sign in that cell specifies an endpoint to predict: a plus selects the left endpoint, and a minus selects the right endpoint. The instances use intervals of diferent lengths and diferent thresholds for the bias accumulated at their endpoints.

For a calibration horizon $T = 2 ^ { \tau }$ , choose integers $i _ { 0 } , h$ with $1 \leq i _ { 0 } < i _ { 0 } + h \leq \tau$ . The parameter $i _ { 0 }$ specifies the coarsest partition, and $h$ is the number of partition levels. We index these levels by

$$
{ \mathcal { T } } : = \{ i _ { 0 } , \ldots , i _ { 0 } + h - 1 \} .
$$

At each level $i ,$ a second index $j$ specifies the bias threshold $2 ^ { j - i }$ . The coarsest level $i = i _ { 0 }$ uses h thresholds, while each finer level $i > i _ { 0 }$ uses just one. Specifically, define

$$
\begin{array} { r } { \mathcal { I } _ { i } : = \left\{ \begin{array} { l l } { \{ i _ { 0 } + 1 , \ldots , i _ { 0 } + h \} , } & { i = i _ { 0 } , } \\ { \{ i _ { 0 } + h \} , } & { i > i _ { 0 } , } \end{array} \right. } \end{array}
$$

to represent the set of possible $j$ for each i. At level i, partition [0, 1] into $2 ^ { i + 1 }$ equal intervals, numbered from zero. For each $j \in \mathcal { I } _ { i }$ , group the intervals by the parity of their indices and assign one instance to each parity. The even and odd SPR instances contain only cells representing the evennumbered and odd-numbered intervals, respectively. Thus any two intervals represented in the same game are separated by an interval of the opposite parity. We write $G _ { i , j , \ell }$ for the instance with indices $i , j$ and parity $\ell \in \{ 0 , 1 \}$ . The total number of SPR instances is $2 h + 2 ( h - 1 ) = 4 h - 2 = O ( \log T )$ compared with the ${ \dot { O } } ( ( \log T ) ^ { 2 } )$ instances used in the construction of Dagan et al. $[ \mathrm { D D F ^ { + } 2 5 } ]$

Note that an instance $G _ { i , j , \ell }$ has $2 ^ { i }$ cells. Then for a cell $c \in [ 2 ^ { i } ]$ , write interva $| ( c , G )$ for its interval and $\operatorname { p r o b } ( c , \sigma , G )$ for the prediction corresponding to the sign $\sigma \in \{ - 1 , 1 \}$ . Setting $m = 2 ( c - 1 ) + \ell ;$ these are

$$
\begin{array} { r l } & { \mathrm { i n t e r v a l } ( c , G ) : = [ m 2 ^ { - ( i + 1 ) } , ( m + 1 ) 2 ^ { - ( i + 1 ) } ) , } \\ & { \mathrm { p r o b } ( c , \sigma , G ) : = \left\{ \begin{array} { l l } { m 2 ^ { - ( i + 1 ) } , } & { \sigma = + 1 , } \\ { ( m + 1 ) 2 ^ { - ( i + 1 ) } , } & { \sigma = - 1 . } \end{array} \right. } \end{array}
$$

The final interval also includes 1. Note that all predictions lie in the fixed grid

$$
\mathcal { P } : = \{ k 2 ^ { - ( i _ { 0 } + h + 1 ) } : k = 0 , \ldots , 2 ^ { i _ { 0 } + h + 1 } \} .
$$

Given a level $i ,$ a threshold index $j ,$ , and a conditional mean $e \in [ 0 , 1 ]$ , the routine cel ${ \mathrm { l } } ( i , j , e )$ locates the cell to which that mean belongs. It returns the unique pair $( c , G )$ whose interval contains e. Specifically, $m = \operatorname* { m i n } \{ \lfloor 2 ^ { i + 1 } e \rfloor , 2 ^ { i + 1 } - 1 \} , \ell = m$ mod $2 ,$ , and $c = ( m - \ell ) / 2 + 1$

For each cell, the forecaster keeps a separate record of the bias accumulated at each of its two endpoints. We denote these records by $\operatorname { b i a s } ( c , \sigma , G )$ for $\sigma \in \{ - 1 , 1 \}$ , which both start at zero. When a round is assigned to an endpoint, its bias is updated by adding $e _ { t } - p _ { t }$ . The value corresponding to the cell’s current sign is its active $b i a s ,$ denoted bias $( c , G )$ , and the other value is inactive. When the cell is empty, set bias $( c , G ) : = 0 ;$ ; both stored values are then inactive.

```latex
Algorithm 3 Calibration strategy using sign preservation
Require: Parameters $\overline { { T = 2 ^ { \tau } , i _ { 0 } \in \mathbb { N } , h \in \mathbb { N } } }$ with $\begin{array} { r } { 1 \leq i _ { 0 } < i _ { 0 } + h \leq \tau , } \end{array}$ and the deterministic
Player-L strategies specified below.
1: ${ \mathcal { T } } \gets \{ i _ { 0 } , \dots , i _ { 0 } + h - 1 \} ; { \mathcal { T } } _ { i _ { 0 } } \gets \{ i _ { 0 } + 1 , \dots , i _ { 0 } + h \}$ and $\mathcal { T } _ { i } \gets \{ i _ { 0 } + h \}$ for $i > i _ { 0 } .$
2: Set $\mathcal { P } \gets \{ k 2 ^ { - ( i _ { 0 } + h + 1 ) } : k = 0 , \dots , 2 ^ { i _ { 0 } + h + 1 } \}$
3: Initialize each board $B _ { G }$ to be empty.
4: Initialize $\operatorname { b i a s } ( c , - 1 , G ) = \operatorname { b i a s } ( c , 1 , G ) = 0$ for all instances $G$ and all cells c.
5: for $t \in [ T ]$ do
6: Observe $e _ { t } .$
7: for $i \in \mathcal { Z }$ do ▷ bias removal loop
8: for $j \in \mathcal { I } _ { i }$ do
9: for $\ell \in \{ 0 , 1 \}$ do
10: $G  G _ { i , j , \ell } .$
11: if ∃c $\in [ 2 ^ { i } ] : \left( \operatorname { b i a s } ( c , G ) > 1 \right)$ and $e _ { t } \leq \mathrm { p r o b } ( c , + 1 , G ) )$ or $( \operatorname { b i a s } ( c , G ) <$
−1 and $e _ { t } \geq \mathrm { p r o b } ( c , - 1 , G ) \}$ then
12: Choose the least such c.
13: $\sigma \gets \mathrm { s i g n } ( \mathrm { b i a s } ( c , G ) )$
14: Predict $p _ { t } \gets \mathrm { p r o b } ( c , \sigma , G )$ ▷ predict to perform bias removal
15: $\operatorname { b i a s } ( c , \sigma , G )  \operatorname { b i a s } ( c , \sigma , G ) + e _ { t } - p _ { t } .$
16: Observe $y _ { t }$ and end round t.
17: for $i \in \mathcal { Z }$ do ▷ bias placement loop
18: for $j \in \mathcal { I } _ { i }$ do
19: $( c , G )  \mathrm { c e l l } ( i , j , e _ { t } ) .$
20: if $| \operatorname { b i a s } ( c , G ) | < 2 ^ { j - i }$ then
21: if c is empty in $B _ { G }$ then
22: simulateGame $( c , G )$ ▷ place/delete signs in SPR
23: $\sigma \gets \mathrm { s i g n }$ in cell $c$ of $B _ { G }$
24: Predict $p _ { t } \gets \mathrm { p r o b } ( c , \sigma , G )$ ▷ predict to perform bias placement
25: bias $( c , \sigma , G ) \gets \mathrm { b i a s } ( c , \sigma , G ) + e _ { t } - p _ { t }$
26: Observe $y _ { t }$ and end round t.
27: Predict a nearest point $p _ { t } \in \mathcal P$ to $e _ { t }$ ▷ rounding fallback
28: Observe $y _ { t }$ and end round t.
```

For every instance $G ,$ initialize the board of the game to be empty. Fix a deterministic optimal base labeler for each instance $G _ { i , j , \ell }$ with space $2 ^ { i }$ and time $r _ { i , j , \ell } ,$ as defined in Theorem 5.5. The routine simulateGame $( c , G )$ supplies a sign for cell c by simulating the SPR game for instance G. When a new move is needed, it treats c as the pointer’s chosen cell and applies $G \mathrm { { } s }$ labeling strategy. The forecasting loop treats that strategy as a black box. In the analysis below, we choose it by wrapping the base labeler with a reduced transcript.

Summary of the forecaster (Algorithm 3). Initially, all SPR boards are empty and all stored endpoint biases are zero. At round t, after observing the conditional mean $\textstyle e _ { t } ,$ the forecaster first checks every interval in the bias removal loop to see if bias can be removed. Specifically, it checks whether an endpoint prediction can reduce an active bias of magnitude greater than one. Its test at $( c , G )$ is

$$
\begin{array} { r l } & { \mathrm { b i a s } ( c , G ) > 1 \quad \mathrm { a n d } \quad e _ { t } \leq \mathrm { p r o b } ( c , + 1 , G ) , \qquad \mathrm { o r } } \\ & { \mathrm { b i a s } ( c , G ) < - 1 \quad \mathrm { a n d } \quad e _ { t } \geq \mathrm { p r o b } ( c , - 1 , G ) . } \end{array}
$$

If the test succeeds, the forecaster predicts the corresponding endpoint and is able to remove some of the active bias from that cell.

If the test fails, the algorithm finds a cell for placement in the bias placement loop by looping through partition levels $i \in \mathcal { T }$ , starting from the coarsest level $i = i _ { 0 }$ and ending at the finest level $i = i _ { 0 } + h - 1$ . For each i, it loops through all thresholds in $\mathcal { I } _ { i }$ . For each $( i , j )$ , it locates the cell c whose interval contains $e _ { t }$ and selects the first one with active bias magnitude below $2 ^ { j - i }$ . If c is empty, we also call the routine simulateGame $( c , G )$ and record the sign it returns. If simulateGame $( c , G )$ returns a plus or minus sign, we predict the left or right endpoint of the interval of cell $c ,$ respectively, and update the bias of the cell. We call a prediction and bias update at a cell selected by this loop a placement at that cell, whether or not it was empty. If no such cell is selected by the bias removal or bias placement loop, we fall back to simply predicting the nearest point of $\mathcal { P }$ (Line 27). Every prediction is followed by observing $y _ { t }$ and ending the round.

## 5.2 The scale-sum lemma

The following lemma bounds the calibration error of the conditional-mean construction in terms of two sums of SPR guarantees. The sums correspond to the coarsest level and the finer levels, respectively. We will apply the bounds on SPR to these sums to prove Theorem 5.1.

Lemma 5.2. Define

$$
\begin{array} { l } { { \displaystyle S _ { 0 } : = \sum _ { j = i _ { 0 } + 2 } ^ { i _ { 0 } + h } 2 ^ { j - i _ { 0 } } \mathrm { o p t } ( 2 ^ { i _ { 0 } } , 2 ^ { \tau - j + 1 } ) , } } \\ { { \displaystyle S _ { 1 } : = \sum _ { i = i _ { 0 } + 1 } ^ { i _ { 0 } + h - 1 } 2 ^ { i _ { 0 } + h - i } \mathrm { o p t } ( 2 ^ { i } , 2 ^ { \tau - i _ { 0 } - h } ) . } } \end{array}
$$

By the end of round $T = 2 ^ { \tau }$ , the forecaster in Algorithm 3 satisfies

$$
\mathbb { E } [ \mathrm { c a l e r r } ( T ) ] = O \left( h 2 ^ { i _ { 0 } } + 2 ^ { \tau - i _ { 0 } - h } + S _ { 0 } + S _ { 1 } + \sqrt { 2 ^ { \tau } ( 2 ^ { i _ { 0 } } + 2 ^ { \tau - i _ { 0 } - h } ) } \right) .\tag{34}
$$

Proof overview of Theorem 5.2. We bound calibration error by conditional-mean bias and the fluctuations $y _ { t } - e _ { t }$ , then bound each contribution.

• We first bound the conditional-mean bias in terms of the signs remaining in each game. An active endpoint has bias magnitude at most $2 ^ { j - i } { + 1 }$ , while an inactive endpoint has magnitude at most one by Theorem 5.3. Thus a game with $S _ { G }$ signs and $N _ { G }$ used cells contributes at most $2 ^ { j - i } S _ { G } + 2 N _ { G }$ , and rounding adds $O ( 2 ^ { \tau - i _ { 0 } - h } )$

• To bound $S _ { G }$ , we undo wasted moves using the reduced transcript in Theorem 5.4. Except at the smallest coarsest-level threshold, repeated recorded queries to a cell require bias to be rebuilt elsewhere. The resulting transcript bounds give $S _ { G } \leq \mathrm { o p t } ( 2 ^ { i } , r _ { i , j , \ell } )$ with the round budgets from Theorem 5.5. Multiplying by the thresholds and summing over both parities bounds the bias from signs by $2 S _ { 0 } + 2 S _ { 1 } + O ( 2 ^ { i _ { 0 } } )$ , where the last term accounts for the smallest coarsest-level threshold.

• We next account for the $N _ { G }$ terms and the number of distinct predictions. Before a fine cell can be used, its parent must receive enough placements to reach its threshold. This shows that at most $O ( 2 ^ { \tau - i _ { 0 } - h } )$ fine cells are used, and there are at most $L = O ( 2 ^ { i _ { 0 } } + 2 ^ { \tau - i _ { 0 } - h } )$ distinct predictions, including rounding, by Theorem 5.6. The coarsest-level cells contribute $O ( h 2 ^ { i _ { 0 } } )$

• Finally, the increments $y _ { t } - e _ { t }$ have conditional mean zero, so the prediction-wise sums have total second moment at most T. Cauchy–Schwarz and the bound on L give an expected contribution of at most LT. Adding this to the bias bound proves (34).

To deduce Theorem 5.1, we apply the $\mathrm { S P R }$ power bound to $S _ { 0 } + S _ { 1 }$ in Theorem 5.2, set $h = \tau - 2 i _ { 0 }$ and choose $i _ { 0 } = \lfloor \tau / ( 3 + 2 \varepsilon ) \rfloor$ to balance the leading bias and variance terms up to constant factors. We extend to arbitrary horizons and apply minimax to remove access to the conditional means.

## 5.3 Lemmas for the proof of Theorem 5.2

We first record the bias bounds and accounting identity, adapting Lemma A.1 of Dagan et al.   
$[ \mathrm { D D F ^ { + } 2 5 } ]$ to our endpoint predictions and simulation with restored signs.

Lemma 5.3. Let $T = 2 ^ { \tau }$ . Throughout Algorithm ${ \mathcal { B } } ,$ the following hold.

1. For every cell c in G, bias $( c , + 1 , G ) \ge 0$ and bias $( c , - 1 , G ) \leq 0$

2. For every cell c in $G _ { i , j , \ell }$ and sign type $\sigma \in \{ - 1 , 1 \} , | \operatorname { b i a s } ( c , \sigma , G ) | \leq 2 ^ { j - i } + 1$ if that value is active, and $| \operatorname { b i a s } ( c , \sigma , G ) | \leq 1$ if that value is inactive. The magnitude of either positive or negative bias for a cell can only increase by at most $2 ^ { - ( i + 1 ) }$ per round.

3. At the end of round $T _ { i }$

$$
\sum _ { p \in \mathcal { P } } \left| \sum _ { t = 1 } ^ { T } \mathbf { 1 } \{ p _ { t } = p \} ( e _ { t } - p ) \right| \leq \sum _ { G , c , \sigma } | \operatorname { b i a s } _ { T } ( c , \sigma , G ) | + O ( 2 ^ { \tau - i _ { 0 } - h } ) .\tag{35}
$$

Proof. We use the induction underlying [DDF<sup>+</sup>25, Lemma A.1], keeping track of the two endpoint biases separately. Initially both are zero. A removal update moves an active value of magnitude greater than one toward zero by at most one, so it preserves its sign and cannot increase its magnitude.

Before a call to simulateGame $( c , G )$ , every sign that may be deleted by c has bias of magnitude at most one, since a larger bias would have resulted in bias removal for that cell instead. Every sign deleted by this call therefore leaves behind a stored value of magnitude at most one. Since c was empty before the call, both its biases have magnitude at most one. The returned sign therefore has active magnitude at most one, and this call preserves the inactive bound.

At level i, a chosen cell’s active bias has magnitude below $2 ^ { j - i }$ . Notice that by predicting at the endpoint corresponding to the chosen sign $\sigma ,$ we have

$$
0 \leq \sigma ( e _ { t } - p _ { t } ) \leq 2 ^ { - ( i + 1 ) } .
$$

Thus a placement at a cell preserves the sign of the stored value and increases its magnitude by at most $2 ^ { - ( i + 1 ) }$ , leaving it below $2 ^ { j - i } + 1$ . These observations complete the induction for items 1–2.

For item 3, each stored value is the sum of the increments $e _ { t } - p _ { t }$ assigned to its fixed endpoint, since it is never reset. Grouping these increments by prediction and applying the triangle inequality gives a bound of $\begin{array} { r } { \sum _ { G , c , \sigma } | \operatorname { b i a s } _ { T } ( c , \sigma , G ) | } \end{array}$ . Each rounding round adds at most half the grid spacing, so their total contribution is at most $T 2 ^ { - ( i _ { 0 } + h + 2 ) } = O ( 2 ^ { \tau - i _ { 0 } - h } )$ □

Following Dagan et al. $\mathrm { [ D D F ^ { + } 2 5 }$ , Appendix $\mathrm { A . 2 } ]$ , we modify each instance’s labeler to remove wasted moves. If a plus is followed by a query to its left, or a minus by a query to its right, that sign is deleted immediately, so we can skip the move that placed it. We formalize this intuition in the definition of a reduced transcript.

Definition 5.4 (Reduced transcript). Given a game G, call the forecaster’s board the real board and the base labeler’s internal board the restored board. Save the base labeler’s full state before each move, including the restored board state. Suppose for a game $G ,$ the reduced transcript shows the pointer previously selected cells $\{ c _ { 1 } , c _ { 2 } , \ldots , c _ { k - 1 } \}$ , from earliest to latest, and simulateGame $( c _ { k } , G )$ is called in the current round. First, simulate all sign deletions that are legally allowed by the call to cell $c _ { k }$ in the real board. If the transcript is nonempty and the sign placed in cell $c _ { k - 1 }$ can legally be deleted by this call, delete the previous call to $c _ { k - 1 }$ and restore the saved base-labeler state, including the restored board, from before that call.

While the transcript is nonempty, continue to undo the last remaining call to $c _ { k ^ { \prime } }$ in reverse chronological order, from latest to earliest, while the call $c _ { k }$ could legally delete the sign in cell $c _ { k ^ { \prime } }$ After these moves have been undone, if cell $c _ { k }$ is occupied on the restored board, reuse its sign without querying the labeler or recording a new move; otherwise, query the labeler for a sign at $c _ { k }$ and record the move. Then place the sign at cell $c _ { k }$ on the real board. The recorded calls that are not deleted and remain after this process form the reduced transcript of G.

Note that a reduced transcript only afects the signs present on the board, but not any of the bias values associated with a cell. Rewinding restores older signs only on the restored board, so that board contains every actual sign, with the same sign in every actually occupied cell, throughout the call. Restoring the full labeler state makes the reduced transcript a legal play of the fixed base labeler. The sign count coming from the reduced transcript thus bounds the actual sign count after each completed call.

The next lemma bounds the length of the reduced transcript in Theorem 5.4; undone moves do not contribute to its length.

Lemma 5.5. Through round $T = 2 ^ { \tau }$ , the number of recorded moves in the reduced transcript of $G = G _ { i , j , \ell }$ is at most ${ r _ { i , j , \ell } } ,$ , where

$$
r _ { i , j , \ell } : = \left\{ \begin{array} { l l } { 2 ^ { \tau } , } & { i = i _ { 0 } , \ j = i _ { 0 } + 1 , } \\ { 2 ^ { \tau - j + 1 } , } & { i = i _ { 0 } , \ j \geq i _ { 0 } + 2 , } \\ { 2 ^ { \tau - i _ { 0 } - h } , } & { i > i _ { 0 } , \ j = i _ { 0 } + h . } \end{array} \right.
$$

Thus, the number of signs on the actual board of $G _ { i , j , \ell }$ is at most opt $( 2 ^ { i } , r _ { i , j , \ell } )$

Proof. Fix an instance and its reduced transcript through round T. Consider two successive recorded queries of a cell c. By definition of a reduced transcript, the move immediately after the first query must not delete the sign, and therefore lies on its preserving side, while the move that actually deletes that sign lies on its deleting side. These moves lie on opposite sides of c and occur between the two queries of c. We consider two cases depending on the value of i.

Case 1: $i = i _ { 0 } . \mathrm { ~ H ~ } j = i _ { 0 } + 1$ , the reduced transcript has length at most $\textit { T } = \textit { r } _ { i , j , \ell }$ because there are at most $T$ rounds played. Now consider $j \ge i _ { 0 } + 2$ . Whenever a move at c is recorded in $G _ { i _ { 0 } , j , \ell } , \vert \mathrm { b i a s } ( c , G _ { i _ { 0 } , j - 1 , \ell } ) \vert \ge 2 ^ { j - 1 - i _ { 0 } }$ <sup>0</sup>. Otherwise, the bias placement loop in Algorithm 3 would have stopped at the earlier instance because the condition on Line 20 would be true. We now show that at some round between successive queries of $c ,$ we have $| \operatorname { b i a s } ( c , G _ { i _ { 0 } , j - 1 , \ell } ) | ~ \leq ~ 1$ . If the sign at $c$ in $G _ { i _ { 0 } , j - 1 , \ell }$ changes between these queries, the conclusion is immediate because the cell must first become empty, when its active bias is zero. Otherwise, this follows because interva $\ l ( c , G _ { i _ { 0 } , j , \ell } ) =$ interval $( c , G _ { i _ { 0 } , j - 1 , \ell } )$ , and the two moves to the left and right of c are only possible when | bias $( c , G _ { i _ { 0 } , j - 1 , \ell } ) | \le 1 ;$ ; otherwise, the condition on Line 11 of Algorithm 3 would be true and bias removal would be performed on c in $G _ { i _ { 0 } , j - 1 , \ell } .$

By Theorem 5.3, both stored values have magnitude at most one at that round. Thus, between successive queries of $c ,$ , the stored value active at the next query must increase in magnitude from at most 1 to $2 ^ { j - 1 - i _ { 0 } }$ . Each round assigned to its endpoint can increase its magnitude by at most $2 ^ { - ( i _ { 0 } + 1 ) }$ , so this takes at least

$$
2 ^ { i _ { 0 } + 1 } ( 2 ^ { j - 1 - i _ { 0 } } - 1 ) = 2 ^ { j } - 2 ^ { i _ { 0 } + 1 } \geq 2 ^ { j - 1 }
$$

rounds. The first query needs as many rounds because the biases start at zero.

Case 2: $i > i _ { 0 }$ . Let $c ^ { \prime }$ denote the parent of c in $G _ { i - 1 , j , \ell ^ { \prime } }$ . Similarly, whenever a move at c is recorded in $G _ { i , j , \ell } , \ | \operatorname { b i a s } ( c ^ { \prime } , G _ { i - 1 , j , \ell ^ { \prime } } ) | \ \geq \ 2 ^ { i _ { 0 } + h - i + 1 }$ . Otherwise, the bias placement loop in Algorithm 3 would have stopped at the earlier instance because the condition on Line 20 would be true. Similarly, we show that at some round between successive queries of $c ,$ we have $| \operatorname { b i a s } ( c ^ { \prime } , G _ { i - 1 , j , \ell ^ { \prime } } ) | \leq$ 1. If the sign at $c ^ { \prime }$ in $G _ { i - 1 , j , \ell ^ { \prime } }$ changes between these queries, the conclusion is immediate because the cell must first become empty, when its active bias is zero. Otherwise, this follows because the only cell in $G _ { i , j , \ell }$ that lies in interva $( c ^ { \prime } , G _ { i - 1 , j , \ell ^ { \prime } } )$ is $c .$ The other cell belongs to a game of diferent parity. Therefore, the conditional means for the two moves lie to the left and right of the parent interval, respectively. This implies $| \operatorname { b i a s } ( c ^ { \prime } , G _ { i - 1 , j , \ell ^ { \prime } } ) | \le 1$ ; otherwise, the condition on Line 11 of Algorithm 3 would be true and bias removal would be performed on $c ^ { \prime }$ in $G _ { i - 1 , j , \ell ^ { \prime } }$

Again, both stored values have magnitude at most one at that round. Thus, between successive queries of $c ,$ the stored value active at the next query must increase in magnitude from at most 1 to $2 ^ { i _ { 0 } + h - i + \dot { 1 } }$ . Each round assigned to its endpoint can increase its magnitude by at most $2 ^ { - i }$ , so this takes at least

$$
2 ^ { i } ( 2 ^ { i _ { 0 } + h - i + 1 } - 1 ) = 2 ^ { i _ { 0 } + h + 1 } - 2 ^ { i } \geq 2 ^ { i _ { 0 } + h }
$$

rounds. The first query needs as many rounds because the biases start at zero.

Note that between successive queries of one cell and across distinct cells, the required rounds that build bias are disjoint: cells have distinct corresponding cells in Case 1 or distinct parents in Case 2, and each round updates at most one of them. Thus, since $T = 2 ^ { \tau }$ , the above cases give the claimed bounds on $r _ { i , j , \ell } .$ □

The next lemma bounds the number of predictions used.

Lemma 5.6. Through round $T = 2 ^ { \tau }$ , the forecaster uses at most $2 ^ { i _ { 0 } + 1 } + 1 + 2 ^ { \tau - i _ { 0 } - h - 1 }$ distinct prediction values.

Proof. Let R count the distinct parent intervals at levels $i _ { 0 } , \ldots , i _ { 0 } + h - 1$ for which either the forecaster selects one of their children and predicts its endpoint, or the conditional mean lies in a child interval at level $i _ { 0 } + h$ on a final rounding round. Fix such a parent at level $i - 1$ , where $i > i _ { 0 }$ . The first prediction in a child cannot be a removal round, since this would require an earlier placement at that child. When one of its children first contributes to $R ,$ the forecaster has passed over the parent in the bias placement loop. Thus the test on Line 20 fails for the parent’s cell in its instance with index $i _ { 0 } + h .$ , and its active stored bias has magnitude at least $2 ^ { i _ { 0 } + h - i + 1 }$

An endpoint’s stored bias starts at zero, and each prediction assigned to this parent’s endpoint increases its magnitude by at most $2 ^ { - i }$ . Hence reaching the threshold requires at least $2 ^ { i _ { 0 } + h + 1 }$ rounds. Each such round updates only one cell of one instance, so we can conclude

$$
R \leq T / 2 ^ { i _ { 0 } + h + 1 } = 2 ^ { \tau - i _ { 0 } - h - 1 } .
$$

Every parent counted by R at a level greater than $i _ { 0 }$ must first have been used to predict an endpoint to build its bias, so its own parent is also counted. Starting from the coarsest partition, each counted parent adds at most one new prediction value, the common endpoint of its two children. The coarsest partition has $2 ^ { i _ { 0 } + 1 } + 1$ endpoints, so there are at most $2 ^ { i _ { 0 } + 1 } + 1 + R \le 2 ^ { i _ { 0 } + 1 } + 1 + 2 ^ { \tau - i _ { 0 } - h - 1 }$ distinct predictions. □

## 5.4 Proof of Theorem 5.2

We bound calibration error by considering two separate terms. These are the stored conditionalmean bias, or the bias term, and the martingale fluctuations, or the variance term. The preceding lemmas control the bias by establishing the length of each reduced transcript $r _ { i , j , \ell }$ and control the fluctuations by bounding the number of distinct predictions.

Proof of Theorem 5.2. For each prediction $p ,$ the identity $y _ { t } - p = ( e _ { t } - p ) - ( e _ { t } - y _ { t } )$ splits its cumulative calibration error into two sums. The first, from $e _ { t } - p .$ is the bias term bounded by (35) in Theorem $5 . 3 ;$ the second, from the centered diferences $e _ { t } - y _ { t }$ , is the variance term. The triangle inequality gives

$$
\mathrm { c a l e r r } ( T ) \leq \sum _ { p \in \mathcal { P } } \left| \sum _ { t = 1 } ^ { T } \mathbf { 1 } \{ p _ { t } = p \} ( e _ { t } - p ) \right| + \sum _ { p \in \mathcal { P } } \left| \sum _ { t = 1 } ^ { T } \mathbf { 1 } \{ p _ { t } = p \} ( e _ { t } - y _ { t } ) \right| .
$$

Bounding the bias. Let $N _ { G }$ be the number of cells of G that received placements through round $T _ { i }$ , and let $S _ { G }$ be its number of signs at that time. By Theorem 5.3, an occupied cell contributes at most $2 ^ { j - i } + 2$ , and an empty used cell contributes at most two. Consequently,

$$
\sum _ { c , \sigma } | \operatorname { b i a s } _ { T } ( c , \sigma , G ) | \leq 2 ^ { j - i } S _ { G } + 2 N _ { G } .
$$

At level $i = i _ { 0 }$ , each parity instance has $2 ^ { i _ { 0 } }$ cells. For $j = i _ { 0 } + 1$ , the bias threshold is two and $S _ { G } \leq N _ { G }$ , so its two instances contribute $O ( 2 ^ { i _ { 0 } } )$ bias. For each $j = i _ { 0 } + 2 , \ldots , i _ { 0 } + h$ , the board inclusion above and Theorem 5.5 give $S _ { G } \leq \mathrm { o p t } ( 2 ^ { i _ { 0 } } , 2 ^ { \tau - j + 1 } )$ in each parity instance. The $2 ^ { j - i _ { 0 } } S _ { G }$ terms from these instances sum to at most $2 S _ { 0 } ;$ their $2 N _ { G }$ terms contribute $O ( h 2 ^ { i _ { 0 } } )$

For $i = i _ { 0 } + 1 , \ldots , i _ { 0 } + h - 1$ , the threshold index is $j = i _ { 0 } + h$ . The same inclusion and roundcount bound give $S _ { G } \leq \mathrm { o p t } ( 2 ^ { i } , 2 ^ { \tau - i _ { 0 } - h } )$ in each parity instance. Multiplying by the threshold $2 ^ { i _ { 0 } + h - i }$ and summing over these levels gives at most $2 S _ { 1 }$ . The remaining $2 N _ { G }$ terms count used intervals at levels $i > i _ { 0 }$ . Each such interval is a child of a parent counted in Theorem 5.6, and each parent has at most two children. Thus the total number of these intervals is $O ( 2 ^ { \tau - i _ { 0 } - h } )$ . Hence, all contributions from cells that have received placements are $O ( h 2 ^ { i _ { 0 } } + 2 ^ { \tau - i _ { 0 } - h } )$ . This bounds the bias by $O ( h 2 ^ { i _ { 0 } } + 2 ^ { \tau - i _ { 0 } - h } + S _ { 0 } + S _ { 1 } )$ .

Bounding the variance. Let

$$
M _ { p } : = \sum _ { t = 1 } ^ { T } \mathbf { 1 } \{ p _ { t } = p \} ( e _ { t } - y _ { t } ) , \qquad p \in \mathcal { P } .
$$

Note that the variables $p _ { t } , \ e _ { t }$ are measurable with respect to the history before $y _ { t }$ is drawn, and $\mathbb { E } [ y _ { t } \mid H _ { t } ] = e _ { t }$ . Thus, for each $p ,$ the increments of $M _ { p }$ have conditional mean zero, and increments at diferent times have zero expected product. Consequently,

$$
\begin{array} { r l } {  { \mathbb { E } \sum _ { p \in \mathcal { P } } M _ { p } ^ { 2 } = \sum _ { t = 1 } ^ { T } \sum _ { p \in \mathcal { P } } \mathbb { E } \big [ \mathbf { 1 } \{ p _ { t } = p \} ( y _ { t } - e _ { t } ) ^ { 2 } \big ] } \quad } & { } \\ & { = \sum _ { t = 1 } ^ { T } \mathbb { E } [ e _ { t } ( 1 - e _ { t } ) ] \leq T . } \end{array}
$$

By Theorem $5 . 6 ,$ at most $L : = 2 ^ { i _ { 0 } + 1 } + 1 + 2 ^ { \tau - i _ { 0 } - h - 1 }$ distinct prediction values are used on every run. Cauchy–Schwarz on each run, followed by Jensen’s inequality, gives

$$
\mathbb { E } \sum _ { p \in \mathcal { P } } \left. M _ { p } \right. \leq \sqrt { L \mathbb { E } \sum _ { p \in \mathcal { P } } M _ { p } ^ { 2 } } \leq \sqrt { L T } = O \left( \sqrt { 2 ^ { \tau } \big ( 2 ^ { i _ { 0 } } + 2 ^ { \tau - i _ { 0 } - h } \big ) } \right) .
$$

Combining the two terms proves the lemma.

## 5.5 Proof of Theorem 5.1

We apply the separate space and time bound to the two scale sums, then set the value of the coarsest level $i _ { 0 }$ to balance the bias and variance terms. We conclude the proof by removing the construction’s access to conditional means.

Proof of Theorem 5.1. First let the construction’s horizon be $T = 2 ^ { \tau }$ . Choose

$$
i _ { 0 } = \left\lfloor { \frac { \tau } { 3 + 2 \varepsilon } } \right\rfloor , \qquad h = \tau - 2 i _ { 0 } .
$$

For suficiently large $\tau ,$ these parameters satisfy $1 \leq i _ { 0 } < i _ { 0 } + h \leq \tau$ . Each SPR board at level $i = i _ { 0 }$ has $2 ^ { i _ { 0 } }$ cells. Each instance at a level $i > i _ { 0 }$ has a round budget of $2 ^ { i _ { 0 } }$ , since $2 ^ { \tau - i _ { 0 } - h } = 2 ^ { i _ { 0 } }$ Recall the estimate from Theorem 5.2:

$$
\mathbb { E } [ \mathrm { c a l e r r } ( T ) ] = O \left( h 2 ^ { i _ { 0 } } + 2 ^ { \tau - i _ { 0 } - h } + S _ { 0 } + S _ { 1 } + \sqrt { 2 ^ { \tau } ( 2 ^ { i _ { 0 } } + 2 ^ { \tau - i _ { 0 } - h } ) } \right) .
$$

We now apply the hypothesis $\mathrm { o p t } ( n , t ) = O ( n ^ { \alpha } t ^ { \beta } )$ to express the two sums $S _ { 0 }$ and $S _ { 1 }$ in terms of the parameters. Write $\tau = 2 i _ { 0 } + h$ and $\alpha + \beta = 1 - \varepsilon$ . The exponents in the two sums can then be rewritten as

$$
\begin{array} { r l } & { \qquad j - i _ { 0 } + \alpha i _ { 0 } + \beta ( \tau - j + 1 ) = \tau - ( 1 + \varepsilon ) i _ { 0 } + \beta - ( 1 - \beta ) ( i _ { 0 } + h - j ) , } \\ & { \qquad i _ { 0 } + h - i + \alpha i + \beta ( \tau - i _ { 0 } - h ) = \tau - ( 1 + \varepsilon ) i _ { 0 } - ( 1 - \alpha ) ( i - i _ { 0 } ) . } \end{array}
$$

The common leading term produces the factor $2 ^ { \tau - ( 1 + \varepsilon ) i _ { 0 } }$ in both bounds. The extra $2 ^ { \beta }$ in the first identity is a constant. Thus, for the terms at level $i = i _ { 0 }$

$$
\begin{array} { c } { { 2 ^ { j - i _ { 0 } } \mathrm { o p t } ( 2 ^ { i _ { 0 } } , 2 ^ { \tau - j + 1 } ) = { \cal O } \Big ( 2 ^ { j - i _ { 0 } + \alpha i _ { 0 } + \beta ( \tau - j + 1 ) } \Big ) } } \\ { { = { \cal O } \Big ( 2 ^ { \tau - ( 1 + \varepsilon ) i _ { 0 } } \cdot 2 ^ { - ( 1 - \beta ) ( i _ { 0 } + h - j ) } \Big ) . } } \end{array}
$$

The exponent $i _ { 0 } + h - j$ is nonnegative for every j in this sum. Similarly, for the terms at levels $i > i _ { 0 }$ 2

$$
\begin{array} { r l } & { 2 ^ { i _ { 0 } + h - i } \operatorname { o p t } ( 2 ^ { i } , 2 ^ { \tau - i _ { 0 } - h } ) = O \Big ( 2 ^ { i _ { 0 } + h - i + \alpha i + \beta ( \tau - i _ { 0 } - h ) } \Big ) } \\ & { \qquad = O \Big ( 2 ^ { \tau - ( 1 + \varepsilon ) i _ { 0 } } \cdot 2 ^ { - ( 1 - \alpha ) ( i - i _ { 0 } ) } \Big ) . } \end{array}
$$

Here $i - i _ { 0 } \ge 1$ . Expanding the definitions of $S _ { 0 }$ and $S _ { 1 }$ and applying these bounds term by term yields

$$
\begin{array} { r l } & { S _ { 0 } + S _ { 1 } = \displaystyle \sum _ { j = i _ { 0 } + 2 } ^ { i _ { 0 } \lceil h } 2 ^ { j - i _ { 0 } } \mathrm { o p t } ( 2 ^ { i _ { 0 } } , 2 ^ { \tau - j + 1 } ) + \displaystyle \sum _ { i = i _ { 0 } + 1 } ^ { i _ { 0 } \lfloor h - 1 } 2 ^ { i _ { 0 } + h - i _ { 0 } } \mathrm { o p t } ( 2 ^ { i } , 2 ^ { \tau - i _ { 0 } - h } ) } \\ & { \quad \quad \quad = O \left( 2 ^ { \tau - ( 1 + \varepsilon ) i _ { 0 } } \cdot \displaystyle \left( \sum _ { j = i _ { 0 } + 2 } ^ { i _ { 0 } + h } 2 ^ { - ( 1 - \beta ) ( i _ { 0 } + h - j ) } + \displaystyle \sum _ { i = i _ { 0 } + 1 } ^ { i _ { 0 } + h - 1 } 2 ^ { - ( 1 - \alpha ) ( i - i _ { 0 } ) } \right) \right) } \\ & { \quad \quad \quad = O \left( 2 ^ { \tau - ( 1 + \varepsilon ) i _ { 0 } } \cdot \displaystyle \left( \sum _ { q = 0 } ^ { h - 2 } 2 ^ { - ( 1 - \beta ) q } + \displaystyle \sum _ { q = 1 } ^ { h - 1 } 2 ^ { - ( 1 - \alpha ) q } \right) \right) } \\ & { \quad \quad \quad = O \left( 2 ^ { \tau - ( 1 + \varepsilon ) i _ { 0 } } \cdot \displaystyle \left( \sum _ { q = 0 } ^ { i _ { 0 } + h } 2 ^ { - ( 1 - \beta ) q } + \displaystyle \sum _ { q = 1 } ^ { i _ { 0 } + h - 1 } 2 ^ { - ( 1 - \alpha ) q } \right) \right) } \\ & { \quad \quad \quad = O \left( 2 ^ { \tau - ( 1 + \varepsilon ) i _ { 0 } } \right) . } \end{array}
$$

In the penultimate line, set $q = i _ { 0 } + h - j$ in the first sum and $q = i - i _ { 0 }$ in the second. Because $\alpha < 1$ and $\beta < 1$ , both ratios $2 ^ { - ( 1 - \beta ) }$ and $2 ^ { - ( 1 - \alpha ) }$ are strictly below 1; the finite sums are therefore bounded by $1 / ( 1 - 2 ^ { - ( 1 - \beta ) } )$ and $1 / ( 1 - 2 ^ { - ( 1 - \alpha ) } )$ , respectively, independently of $h .$ . Empty sums are zero. Substituting $S _ { 0 } + S _ { 1 } = O ( 2 ^ { \tau - ( 1 + \varepsilon ) i _ { 0 } } )$ and $2 ^ { \tau - i _ { 0 } - h } = 2 ^ { i _ { 0 } }$ into this estimate, and using $h \geq 1$ gives

$$
\mathbb { E } [ \mathrm { c a l e r r } ( T ) ] = O \left( 2 ^ { \tau - ( 1 + \varepsilon ) i _ { 0 } } + h 2 ^ { i _ { 0 } } + 2 ^ { ( \tau + i _ { 0 } ) / 2 } \right)
$$

for the forecaster that uses $e _ { t } = \mathbb { E } [ y _ { t } \mid H _ { t } ]$ before predicting. Our choice of $i _ { 0 }$ balances the first and third terms $\mathrm { u p }$ to constant factors, since

$$
1 - \frac { 1 + \varepsilon } { 3 + 2 \varepsilon } = \frac { 1 } { 2 } + \frac { 1 } { 2 ( 3 + 2 \varepsilon ) } = \frac { 2 + \varepsilon } { 3 + 2 \varepsilon } .
$$

The remaining term is smaller: $h \leq \tau ,$ and

$$
\frac { 2 + \varepsilon } { 3 + 2 \varepsilon } - \frac { 1 } { 3 + 2 \varepsilon } = \frac { 1 + \varepsilon } { 3 + 2 \varepsilon } > 0 .
$$

Rounding $i _ { 0 }$ changes only constant factors. Thus

$$
\mathbb { E } [ \mathrm { c a l e r r } ( T ) ] = O \left( T ^ { \frac { 2 + \varepsilon } { 3 + 2 \varepsilon } } \right) = O \left( T ^ { \frac { 2 } { 3 } - \frac { \varepsilon } { 9 + 6 \varepsilon } } \right) .
$$

To generalize to an arbitrary suficiently large horizon $T ,$ take $\tau = \lceil \log _ { 2 } T \rceil$ and run the construction with budget $2 ^ { \tau }$ for its first $T$ rounds. The bias and variance arguments use only $2 ^ { \tau }$ as an upper bound on the number of elapsed rounds, so they apply to this prefix. Since $2 ^ { \tau } < 2 T$ , this changes only the implied constant in the bound. For the remaining small horizons, enlarge the implied constant and use calerr $( T ) \leq T$

We must now address the fact that so far the forecaster has used $e _ { t }$ before predicting, but this is not allowed in the original game setting. It remains to remove the forecaster’s access to $e _ { t } ,$ , which is unavailable in the original game. For each fixed randomized adversary, suppose that the forecaster knows the adversary’s strategy. The forecaster can compute $e _ { t } = \operatorname* { P r } ( y _ { t } = 1 \mid H _ { t } )$ from the observed history and run the construction above, achieving the stated bound against that adversary. Thus, every randomized adversary admits a forecaster achieving the same bound. Since the fixed prediction grid P is finite, this allows us to apply finite-game minimax, as in Hart [Har25] and Dagan et al. $[ \mathrm { D D F ^ { + } 2 5 } ]$ , to obtain one randomized forecaster with the same bound against any adversary, without knowing the adversary’s strategy and without access to $e _ { t }$ □

## 6 Explicit parameters and the main theorem

The following parameter choice completes the proof of our calibration bound. We defer the verification that these parameters satisfy the necessary inequalities of Theorem 4.2 to Section A.

Proposition 6.1. The following six-parameter tuple is admissible:

$$
\begin{array} { r l } & { ~ ( \alpha , \beta , \lambda , \rho , \kappa , \delta ) } \\ & { = \bigl ( 0 . 0 0 8 2 5 4 0 8 9 2 , ~ 0 . 9 5 7 4 6 0 3 4 6 7 , ~ 1 . 9 4 4 4 8 5 8 1 5 6 , } \\ & { ~ 0 . 0 1 6 4 2 3 7 2 4 3 , ~ 0 . 4 1 4 8 7 2 3 3 3 9 , ~ 9 . 4 8 8 0 9 2 8 7 3 3 \bigr ) . } \end{array}\tag{36}
$$

We may also choose the constants

$$
{ \boldsymbol { F } } = 1 0 ^ { 1 0 } , \qquad K _ { p } = 0 . 8 8 1 2 1 5 5 6 7 8 , \qquad K _ { t } = 0 . 9 2 0 8 5 5 3 4 2 4 , \qquad { \boldsymbol { C } } = F ^ { 2 } / K _ { p }\tag{37}
$$

and define $M , D , P , Q$ as in Section A.1. These values satisfy (18). Moreover, $0 < \alpha < \beta < 1$ and

$$
\varepsilon : = 1 - \alpha - \beta = 0 . 0 3 4 2 8 5 5 6 4 1 > 0 .
$$

Proof of Theorem 1.1. Use the parameters and constant C in Theorem 6.1. By Theorem 4.1, the labeling strategy guarantees uniformly over integers $n \geq 1$ and $t \geq 0$ that for $K = 2 ^ { 1 + \alpha } C$

$$
\mathrm { o p t } ( n , t ) \leq K n ^ { 0 . 0 0 8 2 5 4 0 8 9 2 } t ^ { 0 . 9 5 7 4 6 0 3 4 6 7 } .
$$

Since $0 < \alpha < \beta < 1$ and $\varepsilon > 0$ , Theorem 5.1 shows that there exists a forecaster whose expected calibration error against any adversary is

$$
\mathbb { E } [ \mathrm { c a l e r r } ( T ) ] = O \left( T ^ { \theta } \right) , \qquad \theta = \frac { 2 } { 3 } - \frac { \varepsilon } { 9 + 6 \varepsilon } .
$$

$\theta < 0 . 6 6 2 9 4 2 2 8 8$ by Theorem A.5, which yields the desired upper bound on calibration error.

Remark 6.2. The parameters presented in Theorem 6.1 are close to the best values found in our numerical search. We minimize the calibration exponent by maximizing $\varepsilon = 1 - \alpha - \beta$ subject to the parameter inequalities. In Python, SciPy’s scipy.optimize.differential\_evolution first searches broadly for candidate parameters, penalizing parameters if they violate inequalities. We then refine the candidate and additional random starting points with scipy.optimize.minimize, using sequential least squares programming to impose the inequalities as explicit constraints. The smallest calibration exponent found is approximately 0.66294228152554, improving on our choice by only $6 . 3 \cdot 1 0 ^ { - 9 }$

## References

[Daw82] A. P. Dawid. The well-calibrated Bayesian. Journal of the American Statistical Association, 77(379):605–610, 1982. 1

[DDF<sup>+</sup>25] Yuval Dagan, Constantinos Daskalakis, Maxwell Fishelson, Noah Golowich, Robert Kleinberg, and Princewill Okoroafor. Breaking the $T ^ { 2 / 3 }$ barrier for sequential calibration. In Proceedings of the 57th Annual ACM Symposium on Theory of Computing (STOC 2025), pages 2007–2018, New York, NY, USA, 2025. ACM. 1, 2, 3, 4, 5, 6, 18, 21, 22, 27

[FV98] Dean P. Foster and Rakesh V. Vohra. Asymptotic calibration. Biometrika, 85(2):379– 390, 1998. 1, 2

[GPSW17] Chuan Guo, Geof Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1321–1330. PMLR, 2017. 1

[Har25] Sergiu Hart. Calibrated forecasts: The minimax proof. In M. Ali Khan, Nobusumi Sagara, and Alexander J. Zaslavski, editors, Matching, Dynamics and Games for the Allocation of Resources, volume 7 of Monographs in Mathematical Economics, pages 153–159. Springer, Singapore, 2025. 2, 18, 27

[HJKRR18] Ursula H´ebert-Johnson, Michael P. Kim, Omer Reingold, and Guy N. Rothblum. Multicalibration: Calibration for the (computationally-identifiable) masses. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 1939–1948. PMLR, 2018. 1

[KFE18] Volodymyr Kuleshov, Nathan Fenner, and Stefano Ermon. Accurate uncertainties for deep learning using calibrated regression. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 2796–2804. PMLR, 2018. 1

[KLM19] Ananya Kumar, Percy Liang, and Tengyu Ma. Verified uncertainty calibration. In Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019. 1

[PRW<sup>+</sup>17] Geof Pleiss, Manish Raghavan, Felix Wu, Jon Kleinberg, and Kilian Q. Weinberger. On fairness and calibration. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. 1

[QV21] Mingda Qiao and Gregory Valiant. Stronger calibration lower bounds via sidestepping. In Proceedings of the 53rd Annual ACM SIGACT Symposium on Theory of Computing (STOC 2021), pages 456–466, New York, NY, USA, 2021. ACM. 2, 4

[Zha26] Zihan Zhang. Eficient sequential calibration with $O ( T ^ { 2 / 3 - \varepsilon } )$ error bound. arXiv preprint arXiv:2607.12928, 2026. 2

## A Verification of the explicit parameters

This appendix proves Theorem 6.1 and verifies the resulting calibration exponent. We give explicit choices of M, D, P, Q, reduce admissibility to three strict inequalities, and check them using Python with the mpmath library.

## A.1 Verification of the explicit bounds

Fix $\alpha , \beta , \lambda , \rho , \kappa , \delta$ satisfying (16), but we do not yet assume admissibility for this tuple in the proof. Choose constants $F \geq 1 , 0 < K _ { p } < 1$ , and $K _ { t } > K _ { p }$ . Set $x _ { * } : = \operatorname* { m i n } \{ \delta ^ { - 1 } , K _ { p } ^ { - 1 / ( 1 - \beta ) } \}$ and define the phase 1 coeficient

$$
\begin{array} { c } { { \displaystyle M : = \operatorname* { m a x } \Biggl \{ \Biggl [ \left( \frac { K _ { p } + x _ { * } ^ { \beta } } { ( 1 + x _ { * } ) ^ { \beta } } \right) ^ { \frac { 1 } { 1 - \beta } } + 1 \Biggl ] ^ { 1 - \beta } , } } \\ { { \Biggl [ K _ { p } ^ { \frac { 1 } { 1 - \beta } } + \left( \frac { 1 + \rho ^ { \beta } } { ( 1 + \rho ) ^ { \beta } } \right) ^ { \frac { 1 } { 1 - \beta } } \Biggr ] ^ { 1 - \beta } \Biggl \} . } } \end{array}\tag{38}
$$

The coeficients $P , Q$ control persistent and total signs, respectively, at the end of phase 2 before rounding, with $\tau$ as in (23):

$$
P : = \operatorname* { s u p } _ { ( z , r ) \in \mathcal { T } } \operatorname* { m a x } \left\{ \frac { K _ { p } z ^ { \beta } + r ^ { \beta } + \lambda 2 ^ { \alpha } \kappa ^ { \beta } } { ( z + 1 + r + \kappa ) ^ { \beta } } , \frac { K _ { p } z ^ { \beta } + 1 + \lambda ^ { - 1 } 2 ^ { \alpha } \kappa ^ { \beta } } { ( z + 1 + r + \kappa ) ^ { \beta } } \right\} ,\tag{39}
$$

$$
Q : = \operatorname* { s u p } _ { ( z , r ) \in \mathcal { T } } \frac { K _ { p } z ^ { \beta } + 1 + r ^ { \beta } + \lambda ^ { - 1 } 2 ^ { \alpha } \kappa ^ { \beta } } { ( z + 1 + r + \kappa ) ^ { \beta } } .\tag{40}
$$

Finally, D combines the bounds for an execution that may stop in either phase:

$$
D : = \operatorname* { m a x } \left\{ \left( M ^ { \frac { 1 } { 1 - \beta } } + \left( \lambda ^ { - 1 } 2 ^ { \alpha } \right) ^ { \frac { 1 } { 1 - \beta } } \right) ^ { 1 - \beta } , \left( K _ { t } ^ { \frac { 1 } { 1 - \beta } } + 1 \right) ^ { 1 - \beta } \right\} .
$$

Ranges and bounds for unfinished epochs. We begin by establishing a weighted concavity inequality.

Lemma A.1. For every $\beta \in ( 0 , 1 )$ and all $A \geq 0 , B \geq 0 , x \geq 0 , y \geq 0 $

$$
A x ^ { \beta } + B y ^ { \beta } \leq \left( A ^ { \frac { 1 } { 1 - \beta } } + B ^ { \frac { 1 } { 1 - \beta } } \right) ^ { 1 - \beta } ( x + y ) ^ { \beta } .
$$

Proof. If $A = 0 { \mathrm { ~ o r ~ } } B = 0$ , the inequality follows from the monotonicity of $u \mapsto u ^ { \beta } \mathrm { ~ o n ~ } [ 0 , \infty )$ Otherwise, set

$$
S : = A ^ { \frac { 1 } { 1 - \beta } } + B ^ { \frac { 1 } { 1 - \beta } } , \qquad p : = \frac { A ^ { \frac { 1 } { 1 - \beta } } } { S } \in ( 0 , 1 ) .
$$

Since $u \mapsto u ^ { \beta }$ is concave on $[ 0 , \infty )$

$$
\begin{array} { c l l } { \displaystyle { A x ^ { \beta } + B y ^ { \beta } = S ^ { 1 - \beta } \left[ p \left( \frac { x } { p } \right) ^ { \beta } + ( 1 - p ) \left( \frac { y } { 1 - p } \right) ^ { \beta } \right] } } \\ { \displaystyle { \leq S ^ { 1 - \beta } ( x + y ) ^ { \beta } . } } \end{array}
$$

The second term defining M is greater than $K _ { p } .$ The definition of D gives $D \geq M$ . The functions defining $P , Q$ are positive and continuous on the nonempty compact domain $\tau ,$ so their suprema are finite and positive. This verifies all the remaining ranges in (17). For $x \ge 0 , y \ge 0$ the weighted concavity inequality Theorem A.1 gives

$$
K _ { t } x ^ { \beta } + y ^ { \beta } \leq \left( K _ { t } ^ { \frac { 1 } { 1 - \beta } } + 1 \right) ^ { 1 - \beta } ( x + y ) ^ { \beta } \leq D ( x + y ) ^ { \beta } ,
$$

$$
M x ^ { \beta } + \lambda ^ { - 1 } 2 ^ { \alpha } y ^ { \beta } \leq \left( M ^ { \frac { 1 } { 1 - \beta } } + ( \lambda ^ { - 1 } 2 ^ { \alpha } ) ^ { \frac { 1 } { 1 - \beta } } \right) ^ { 1 - \beta } ( x + y ) ^ { \beta } \leq D ( x + y ) ^ { \beta } .
$$

Thus (19) and (20) hold.

Phase 1 inequalities. The following lemma verifies (21) and (22) for the explicit choice of $M .$

Lemma A.2. For the phase 1 coeficient M in (38), (21) and (22) hold under their respective count conditions. In particular, they apply when $B . p h a s e = 1$ at the end of a round and when a request changes the phase from 1 to ${ \mathcal { Q } } ,$ respectively.

Proof. Suppose first that $\delta t _ { l } < \tau$ or $t _ { l } < \rho t _ { h }$ , as required in (21).

If $\delta t _ { l } ~ < ~ \tau .$ , then $\tau > 0$ and $t _ { l } / \tau < \delta ^ { - 1 }$ . For $x > 0$ , the derivative of $\frac { K _ { p } + x ^ { \beta } } { ( 1 + x ) ^ { \beta } }$ has the sign of $x ^ { \beta - 1 } - K _ { p }$ , which is nonnegative exactly when $x \le K _ { p } ^ { - 1 / ( 1 - \beta ) }$ . By continuity, the ratio is maximized on $[ 0 , \delta ^ { - 1 } ]$ at $x _ { * }$ . Thus, $\begin{array} { r } { K _ { p } \tau ^ { \beta } + t _ { l } ^ { \beta } \leq \frac { K _ { p } + x _ { * } ^ { \beta } } { ( 1 + x _ { * } ) ^ { \beta } } ( \tau + t _ { l } ) ^ { \beta } } \end{array}$ , and by Theorem $\mathrm { A . 1 }$ 2

$$
K _ { p } \tau ^ { \beta } + t _ { l } ^ { \beta } + t _ { h } ^ { \beta } \leq \left[ \left( \frac { K _ { p } + x _ { * } ^ { \beta } } { ( 1 + x _ { * } ) ^ { \beta } } \right) ^ { \frac { 1 } { 1 - \beta } } + 1 \right] ^ { 1 - \beta } ( \tau + t _ { l } + t _ { h } ) ^ { \beta } .
$$

Instead, if $\rho t _ { h } > t _ { l }$ , then $t _ { h } > 0$ and $\begin{array} { r } { \frac { t _ { l } } { t _ { h } } < \rho . } \end{array}$ . For $x \in ( 0 , 1 ]$ , the derivative of $\frac { 1 + x ^ { \beta } } { ( 1 + x ) ^ { \beta } }$ has the sign of $x ^ { \beta - 1 } - 1$ , which is nonnegative. By continuity, this ratio is also nondecreasing on [0, 1]. Thus, $\begin{array} { r } { t _ { l } ^ { \beta } + t _ { h } ^ { \beta } \le \frac { 1 + \rho ^ { \beta } } { ( 1 + \rho ) ^ { \beta } } ( t _ { l } + t _ { h } ) ^ { \beta } } \end{array}$ , and by Theorem $\mathrm { A . 1 }$

$$
K _ { p } \tau ^ { \beta } + t _ { l } ^ { \beta } + t _ { h } ^ { \beta } \leq \left[ K _ { p } ^ { \frac { 1 } { 1 - \beta } } + \left( \frac { 1 + \rho ^ { \beta } } { ( 1 + \rho ) ^ { \beta } } \right) ^ { \frac { 1 } { 1 - \beta } } \right] ^ { 1 - \beta } ( \tau + t _ { l } + t _ { h } ) ^ { \beta } .
$$

These two cases establish (21). Under the conditions of (22), (21) applies to $( \tau , t _ { h } , t _ { l } - 1 )$ . Since $( x + 1 ) ^ { \beta } - x ^ { \beta } \leq 1$ for all $x \geq 0$ and $0 < \beta < 1$ ，

$$
\begin{array} { r } { K _ { p } \tau ^ { \beta } + t _ { h } ^ { \beta } + t _ { l } ^ { \beta } \leq M ( \tau + t _ { h } + t _ { l } - 1 ) ^ { \beta } + 1 \leq M ( \tau + t _ { h } + t _ { l } ) ^ { \beta } + 1 . } \end{array}
$$

For an actual epoch, a round ending in phase 1 fails the trigger test in line 13 of Algorithm 2, so the first condition holds. At the first trigger, the requested half has a smaller count just before the request: increasing a strictly larger count cannot make either part of the trigger test become true, and increasing one of two equal counts cannot do so either. Thus the sorted counts change from $( t _ { h } , t _ { l } - 1 )$ to $( t _ { h } , t _ { l } )$ , which gives the second condition. □

Phase 2 inequalities. Each expression in the maximum defining P is at most $P ,$ and the expression defining Q is at most Q. Multiplying by the positive denominator $( z + 1 + r + \kappa ) ^ { \beta }$ proves (24), (25), and (26), respectively. The next lemma verifies (27), (28), (29), and (30), conditional on the first two inequalities in (18).

Lemma A.3. Take the endpoint coeficients P, Q from (39)–(40) and suppose $P ( 1 + F ^ { - 1 } ) + F ^ { - \beta } <$ $K _ { p }$ and $Q ( 1 + F ^ { - 1 } ) + F ^ { - \beta } < K _ { t }$ . Then (27), (28), (29), and (30) hold for all counts in the domain specified in Theorem $4 . 2 .$

Proof. Fix counts satisfying the conditions in Theorem 4.2, and set $L = \lfloor \kappa t _ { h } \rfloor$ and $t = \tau + t _ { l } +$ $t _ { h } + L \ge F$ . Set $\widetilde { t _ { l } } : = \operatorname* { m a x } \{ \tau / \delta , \rho t _ { h } \} , z = \tau / t _ { h }$ , and $r = \widetilde { t } _ { l } / t _ { h }$ . The specified count conditions give $t _ { l } - 1 < \widetilde { t } _ { l } \leq t _ { l }$ and $( z , r ) \in \mathcal { T }$ . Thus,

$$
\begin{array} { r l } & { K _ { p } \tau ^ { \beta } + t _ { l } ^ { \beta } + \lambda 2 ^ { \alpha } L ^ { \beta } \leq K _ { p } \tau ^ { \beta } + \tilde { t } _ { l } ^ { \beta } + \lambda 2 ^ { \alpha } ( \kappa t _ { h } ) ^ { \beta } + 1 } \\ & { \qquad = t _ { h } ^ { \beta } ( K _ { p } z ^ { \beta } + r ^ { \beta } + \lambda 2 ^ { \alpha } \kappa ^ { \beta } ) + 1 } \\ & { \qquad \leq P ( \tau + \tilde { t } _ { l } + t _ { h } + \kappa t _ { h } ) ^ { \beta } + 1 } \\ & { \qquad \leq P ( t + 1 ) ^ { \beta } + 1 } \\ & { \qquad = P { t } ^ { \beta } ( 1 + { t } ^ { - 1 } ) ^ { \beta } + 1 } \\ & { \qquad \leq \big ( P ( 1 + { r } ^ { - 1 } ) + { r } ^ { - \beta } \big ) { t } ^ { \beta } } \\ & { \qquad < K _ { p } { t } ^ { \beta } . } \end{array}
$$

The second inequality follows from the definition of $P ,$ the third inequality follows because $\kappa t _ { h } <$ $L + 1$ implies $\tau + \widetilde { t } _ { l } + t _ { h } + \kappa t _ { h } < t + 1$ , and the last inequality follows from the hypothesis of the lemma. The second expression defining P similarly gives

$$
K _ { p } \tau ^ { \beta } + t _ { h } ^ { \beta } + \lambda ^ { - 1 } 2 ^ { \alpha } L ^ { \beta } \leq P ( \tau + \widetilde t _ { l } + t _ { h } + \kappa t _ { h } ) ^ { \beta } < K _ { p } t ^ { \beta } .
$$

Finally, the expression defining $Q$ similarly gives

$$
\begin{array} { r l } & { K _ { p } \tau ^ { \beta } + t _ { h } ^ { \beta } + t _ { l } ^ { \beta } + \lambda ^ { - 1 } 2 ^ { \alpha } L ^ { \beta } \leq K _ { p } \tau ^ { \beta } + t _ { h } ^ { \beta } + \widehat { t } _ { l } ^ { \beta } + \lambda ^ { - 1 } 2 ^ { \alpha } ( \kappa t _ { h } ) ^ { \beta } + 1 } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ &  \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \ \end{array}
$$

For the intermediate bound, define $f ( t _ { c } ) = K _ { p } \tau ^ { \beta } + t _ { l } ^ { \beta } + \lambda 2 ^ { \alpha } t _ { c } ^ { \beta } - K _ { p } ( \tau + t _ { l } + t _ { h } + t _ { c } ) ^ { \beta }$ . For $t _ { c } > 0$

$$
f ^ { \prime } ( t _ { c } ) = \beta \left( \lambda 2 ^ { \alpha } t _ { c } ^ { \beta - 1 } - K _ { p } ( \tau + t _ { l } + t _ { h } + t _ { c } ) ^ { \beta - 1 } \right) \ge 0
$$

since $\lambda 2 ^ { \alpha } > 1 > K _ { p }$ and $\beta < 1$ . By continuity at zero, f is nondecreasing on $[ 0 , L ]$ . The first inequality proved above gives $f ( L ) < 0$ , so $f ( t _ { c } ) \leq f ( L ) < 0$ for all $t _ { c } \in [ 0 , L ]$ , including the case $L = 0 .$ □

The arguments above reduce all the conditions in Theorem 4.2 to the three inequalities in (18). These depend on the remaining choices of $F , K _ { p } , K _ { t }$ as well as the six parameters. The next lemma verifies these inequalities for the explicit values in Theorem 6.1.

Lemma A.4. The tuple in (36), with the auxiliary choices in (37), satisfies

$$
\begin{array} { c } { { P ( 1 + F ^ { - 1 } ) + F ^ { - \beta } < K _ { p } , ~ Q ( 1 + F ^ { - 1 } ) + F ^ { - \beta } < K _ { t } , } } \\ { { { } } } \\ { { D + F ^ { - \beta } < 2 ^ { \alpha } . } } \end{array}
$$

Proof. We first identify the points at which the suprema defining $P$ and $Q$ are attained. Recall that the trigger domain is the union of two segments:

$$
\mathcal T = \{ ( z , \rho ) : 0 \leq z \leq \delta \rho \} \cup \{ ( \delta r , r ) : \rho \leq r \leq 1 \} .
$$

Each expression in (39) and (40) has the form

$$
\frac { K _ { p } z ^ { \beta } + a r ^ { \beta } + b } { ( z + 1 + r + \kappa ) ^ { \beta } } ,
$$

where the coeficients $a , b$ are chosen as follows:

<table><tr><td></td><td>a b</td></tr><tr><td>First expression for P</td><td>1 λ2ακβ</td></tr><tr><td>Second expression for P</td><td>0  $1 + \lambda ^ { - 1 } 2 ^ { \alpha } \kappa ^ { \beta }$ </td></tr><tr><td>Expression for  $Q$ </td><td>1  $1 + \lambda ^ { - 1 } 2 ^ { \alpha } \kappa ^ { \beta }$ </td></tr></table>

Consider one row of the table. On the first segment, substituting $r = \rho$ yields, for $z \in [ 0 , \delta \rho ]$

$$
f ( z ) = \frac { K _ { p } z ^ { \beta } + a \rho ^ { \beta } + b } { ( z + 1 + \rho + \kappa ) ^ { \beta } } .
$$

On the second segment, substituting $z = \delta r$ yields, for $r \in [ \rho , 1 ]$ ，

$$
g ( r ) = \frac { ( K _ { p } \delta ^ { \beta } + a ) r ^ { \beta } + b } { ( ( 1 + \delta ) r + 1 + \kappa ) ^ { \beta } } .
$$

Both functions are continuous on their closed intervals and diferentiable in their interiors, since their denominators are positive. Each therefore attains its maximum at an endpoint or an interior critical point. Diferentiation gives

$$
f ^ { \prime } ( z ) = \frac { \beta \left[ K _ { p } ( 1 + \rho + \kappa ) z ^ { \beta - 1 } - ( a \rho ^ { \beta } + b ) \right] } { ( z + 1 + \rho + \kappa ) ^ { \beta + 1 } }
$$

and

$$
g ^ { \prime } ( r ) = \frac { \beta \left[ ( K _ { p } \delta ^ { \beta } + a ) ( 1 + \kappa ) r ^ { \beta - 1 } - ( 1 + \delta ) b \right] } { ( ( 1 + \delta ) r + 1 + \kappa ) ^ { \beta + 1 } } .
$$

Because $0 < \beta < 1$ , each numerator is strictly decreasing and has a unique positive zero. If the critical point lies in the interval, the maximum occurs at the critical point; otherwise, it occurs at the nearest endpoint. The maximizing points are therefore

$$
z _ { * } = \operatorname* { m i n } \left\{ \delta \rho , \left( \frac { K _ { p } ( 1 + \rho + \kappa ) } { a \rho ^ { \beta } + b } \right) ^ { 1 / ( 1 - \beta ) } \right\}
$$

and

$$
r _ { * } = \operatorname* { m i n } \left\{ 1 , \operatorname* { m a x } \left\{ \rho , \left( \frac { ( K _ { p } \delta ^ { \beta } + a ) ( 1 + \kappa ) } { ( 1 + \delta ) b } \right) ^ { 1 / ( 1 - \beta ) } \right\} \right\} .
$$

For each row of the table, the supremum over $\tau$ must be max $\{ f ( z _ { * } ) , g ( r _ { * } ) \}$ . This means P is the maximum of the four values corresponding to the first two rows, and $Q$ is the maximum of the two values corresponding to the third row.

It remains to certify

$$
P < \frac { K _ { p } - F ^ { - \beta } } { 1 + F ^ { - 1 } } , \qquad Q < \frac { K _ { t } - F ^ { - \beta } } { 1 + F ^ { - 1 } } , \qquad D + F ^ { - \beta } < 2 ^ { \alpha } .
$$

The $\mathrm { P y }$ thon program in Listing 1 verifies these bounds using mpmath 1.3.0 interval arithmetic at 80 decimal digits of precision. The code uses outward upper bounds for M, D, P, Q; the strict inequalities verified for these bounds also hold for the exact coeficients defined above. □

Proof of Theorem 6.1. The parameter ranges and the identity $\varepsilon = 1 - \alpha - \beta = 0 . 0 3 4 2 8 5 5 6 4 1$ follow directly from the exact decimals in (36) and (37). The explicit construction above verifies the remaining auxiliary ranges, the two bounds for unfinished epochs, and the three normalized endpoint bounds. Theorem A.2 verifies the two phase 1 inequalities. Theorem A.4 establishes all three inequalities in (18); the first two then allow Theorem A.3 to verify the four phase 2 inequalities. Thus every condition in Theorem 4.2 holds, proving admissibility of the six-parameter tuple. Finally, $C = 1 0 ^ { 2 0 } / K _ { p } = F ^ { 2 } / K _ { p }$ is a suficient choice for the induction. □

It remains to verify the resulting calibration exponent, which can be checked using rational arithmetic alone.

Lemma A.5. For the parameters in (36) and $\varepsilon = 1 - \alpha - \beta _ { ; }$

$$
{ \frac { 2 } { 3 } } - { \frac { \varepsilon } { 9 + 6 \varepsilon } } = { \frac { 2 0 3 4 2 8 5 5 6 4 1 } { 3 0 6 8 5 7 1 1 2 8 2 } } < 0 . 6 6 2 9 4 2 2 8 8 .
$$

Proof. Substituting the exact decimals into $( 2 + \varepsilon ) / ( 3 + 2 \varepsilon )$ gives the displayed fraction. The strict inequality follows from

$$
0 . 6 6 2 9 4 2 2 8 8 - { \frac { 2 0 3 4 2 8 5 5 6 4 1 } { 3 0 6 8 5 7 1 1 2 8 2 } } = { \frac { 1 6 2 3 9 0 4 1 3 } { 9 5 8 9 2 8 4 7 7 5 6 2 5 0 0 0 0 0 } } > 0 .
$$

## A.2 Python code

Listing 1 checks that the stated parameters satisfy all inequalities in Theorem 4.2 and computes the resulting calibration exponent.

```python
Listing 1: Independent verification of the explicit parameters.
1 # Run with Python and mpmath 1.3.0: python3 verify_parameters.py.
2
3 from fractions import Fraction
4 from mpmath import iv
5
6 iv.dps = 80
7 parameters = (
8 ’0.0082540892’, ’0.9574603467’, ’1.9444858156’,
9 ’0.0164237243’, ’0.4148723339’,
10 ’0.8812155678’, ’0.9208553424’, ’9.4880928733’,
11 )
12 alpha, beta, lam, rho, kappa, K_p, K_t, delta = map(iv.mpf, parameters)
```

```julia
13 F = iv.mpf(’1e10’)
14 p = 1/(1-beta)
15 s = 1+kappa
16 v = 2**alpha
17 rounding = F**(-beta)
18 inflation = 1+1/F
19
20 def clip(x, lower, upper):
21 """Enclose min(upper, max(lower, x)) using monotonicity."""
22 return iv.mpf([
23 min(upper.a, max(lower.a, x.a)),
24 min(upper.b, max(lower.b, x.b)),
25 ])
26
27
28 def norm(x, y):
29 """Coefficient in the weighted concavity inequality."""
30 return (x**p+y**p)**(1/p)
31
32
33 def endpoint_upper(a, b):
34 """Upper bound from the two maximizing points in Lemma A.1."""
35 z = clip((K_p*(rho+s)/(a*rho**beta+b))**p,
36 iv.mpf(0), delta*rho)
37 r = clip(((K_p*delta**beta+a)*s/((1+delta)*b))**p,
38 rho, iv.mpf(1))
39 f = (K_p*z**beta+a*rho**beta+b)/(z+rho+s)**beta
40 g = ((K_p*delta**beta+a)*r**beta+b)/((1+delta)*r+s)**beta
41 return max(f.b, g.b)
42
43
44 # Phase 1: the two cases in the waiting-counts lemma.
45 x_star = clip(K_p**(-p), iv.mpf(0), 1/delta)
46 M_tau = norm((K_p+x_star**beta)/(1+x_star)**beta, 1)
47 M_rho = norm(K_p, (1+rho**beta)/(1+rho)**beta)
48
49 # Upper endpoints are exact witnesses, so non-strict checks are rigorous.
50 M = max(M_tau.b, M_rho.b)
51 D_one_half = norm(K_t, 1)
52 D_continuation = norm(M, v/lam)
53 D = max(D_one_half.b, D_continuation.b)
54
55 # The three normalized endpoint coefficients.
56 P_plus = endpoint_upper(0, 1+v*kappa**beta/lam)
57 P_minus = endpoint_upper(1, lam*v*kappa**beta)
58 Q_total = endpoint_upper(1, 1+v*kappa**beta/lam)
59 P = max(P_plus, P_minus)
60 Q = Q_total
61
62 # Bounds after integer rounding, from the terminal-bounds lemma.
63 P_rounded = P*inflation+rounding
64 Q_rounded = Q*inflation+rounding
65 D_rounded = D+rounding
66 terminal_minus = P_minus*inflation+rounding
67 terminal_plus = P_plus*inflation+rounding
68 terminal_total = Q_total*inflation+rounding
69
70 epsilon = 1-alpha-beta
71 epsilon_exact = 1-Fraction(parameters[0])-Fraction(parameters[1])
```

72 exponent = (2+epsilon)/(3+2\*epsilon)   
73   
74   
75 # Parameter ranges, in the order of the admissibility definition.   
76 assert alpha > 0   
77 assert 0 < beta < 1   
78 assert 1 < lam <= 2   
79 assert 0 < rho <= 1   
80 assert 0 < kappa <= 1   
81 assert delta > 0   
82   
83 # 1. Auxiliary ranges.   
84 assert F >= 1   
85 assert 0 < K\_p < 1   
86 assert K\_t > K\_p   
87 assert K\_p <= M   
88 assert M <= D   
89 assert P > 0   
90 assert Q > 0   
91   
92 # 2. Induction slack.   
93 assert P\_rounded < K\_p   
94 assert Q\_rounded < K\_t   
95 assert D\_rounded < v   
96   
97 # 3. Unfinished epochs: weighted concavity covers every x,y >= 0.   
98 assert D\_one\_half.b <= D   
99 assert D\_continuation.b <= D   
100   
101 # 4. Phase 1: these two cases certify the waiting inequality.   
102 # The same bounds certify the first-trigger inequality with its +1,   
103 # since (x+1)\*\*beta-x\*\*beta <= 1 for every x >= 0 and 0 < beta < 1.   
104 assert M\_tau.b <= M   
105 assert M\_rho.b <= M   
106   
107 # 5. Normalized endpoints: the three inequalities over all of T.   
108 assert P\_plus <= P   
109 assert P\_minus <= P   
110 assert Q\_total <= Q   
111   
112 # 6. Phase 2: the three terminal inequalities, in their stated order.   
113 assert terminal\_minus < K\_p   
114 assert terminal\_plus < K\_p   
115 assert terminal\_total < K\_t   
116 # The intermediate inequality follows from the terminal-minus bound   
117 # and monotonicity of its deficit on 0 <= t\_c <= L.   
118 assert lam\*v >= K\_p   
119   
120 # Reduction to calibration and the exact value of epsilon.   
121 assert alpha < beta   
122 assert epsilon\_exact == Fraction(’0.0342855641’)   
123 assert epsilon > 0   
124 assert exponent < iv.mpf(’0.662942288’)   
125 print(’PASS: all admissibility conditions and the calibration exponent.’)