# Marginal Log-Likelihood Increments under Dirichlet-Smoothed Markov Estimation

Exact trace valuation, attainable gain, and budgeted selection

Levin David Schwab

21 September 2026

## Abstract

For a Dirichlet-smoothed transition model, the efect of adding one workflow trace to the training archive is an exact change in reference-weighted log likelihood. We derive that change and show that it is a weighted reduction of Kullback–Leibler divergence between the reference conditionals and the model. From this form we obtain an upper bound on the gain available to any acquisition, which expresses a millinat diference as a share of what is attainable, an exact covariance identity for the efect of the reference weighting, and a sign criterion for the interaction between two candidates, from which the batch objective is neither submodular nor supermodular. A case study on the BPI Challenge 2012 loan-application log measures all three and finds a positive selection result in one of the four combinations of reference weighting and budget unit. There, of two regressors fitted to identical descriptors and identical labels, the one that predicts individual increments far more accurately, median R<sup>2</sup> 0.87 against 0.62, realizes the smaller share of the attainable gain, 61 against 69 per cent, so ranking accuracy for individual traces is neither necessary nor suficient for batch quality.

## 1 Introduction

A workflow trace records the activities of one recorded case, such as a loan application. Suppose an archive already trains a model to predict the next recorded activity. An additional trace is useful if adding it reduces prediction loss on the workload against which the model is evaluated.

The practical question is whether inexpensive descriptions of a trace can identify useful additions under a limited acquisition budget. A useful answer must distinguish three claims. First, a descriptor may predict a trace’s marginal contribution for a fixed archive and evaluation population. Second, that prediction may remain useful after the evaluation period changes. Third, selecting a batch by individual predictions may improve the model. A priori none of these claims implies the next.

The learner studied here retains only the current activity and is estimated by counting transitions. The simplicity is deliberate: it makes every marginal contribution exactly calculable and removes valuation noise as an explanation for failure. It also limits the conclusions, since a first-order learner represents no long-range dependence.

## 2 Background

Event logs. An event log is a table produced as a by-product of a business information system. Each row records that a named activity occurred in a named case at a given time. Grouping the rows by case and ordering them by time turns the log into a finite set of finite sequences over a finite alphabet of activities, and one such sequence is a trace. The log used here has 164,506 retained events in 13,087 cases over an alphabet of 23 activities. A log is an observed record rather than a sample from a postulated process, and the only structure used below is the ordering of activities within a case. Selecting a subset of such a log for training has been studied under other objectives [FSVP<sup>+</sup>23].

The learner. The prediction task is the next recorded activity given the current one, one of the tasks collected under predictive process monitoring [TDLRM19]. A first-order Markov model assigns to each activity x a distribution over its successors, so the parameter is a row-stochastic $K \times K$ matrix consisting of one categorical distribution per row, and the rows are estimated independently. For multinomial observations the maximum-likelihood estimate of a row is the vector of relative frequencies $N ( x , y ) / N _ { x }$

Smoothing. Relative frequencies are unusable under a logarithmic loss. A successor never observed after x receives probability zero, and a single occurrence of that transition in the evaluation data makes the loss infinite; rows supported on few observations are unstable for the same reason. Adding a constant $\alpha > 0$ to every cell removes both defects and has an exact Bayesian reading. If each row carries an independent symmetric Dirichlet $( \alpha , \ldots , \alpha )$ prior and the successors of x are multinomial, the posterior for row x is Dirichlet $( N ( x , \cdot ) + \alpha )$ , and its posterior predictive distribution is the estimator (3.2) below. We take $\alpha = 1 / 2$ , the Jefreys prior for the multinomial, which is also the Krichevsky–Trofimov estimator of universal coding [KT81].

The evaluation. Performance is the expected log probability that the model assigns to transitions drawn from a reference distribution q on pairs $( x , y )$ . As a loss this is the cross entropy of q relative to the model, reported in nats, the unit of information belonging to base e, one nat being $1 / \log 2 \approx 1 . 4 4 3$ bits, which is the natural scale here because every increment below is a diference of logarithms of count ratios and so carries no conversion constant. Two properties of this convention matter here. The reference q is a choice and not a property of the data, since the same held-out cases admit several reference distributions, and Section 5 shows that the choice reorders the candidates. Log loss is also the only common evaluation under which an assigned probability of zero is inadmissible rather than merely poor, which is what makes the smoothing constant part of the model.

The quantity. Data valuation asks what a single training record is worth to a specified learner and task. The established answers are coalitional: Data Shapley averages a record’s marginal contribution over subsets of the training set [GZ19], and influence functions approximate the efect of removing it [KL17]. Both require retraining or approximation, so the target of any prediction is itself known only up to noise. The quantity studied here is the forward diference instead. Hold the archive fixed, add the transition counts of one trace, and record the change in the reference-weighted log likelihood. Exact computability is the reason for choosing a first-order learner. It removes valuation error as an explanation for whatever the selection experiments show, at the cost of a model that represents no long-range dependence.

The experiment in outline. Four disjoint groups of cases are used and should not be confused. The archive is the set of cases the learner has already counted; it is held fixed throughout. The candidate pool is the set of cases that might be acquired, of which an acquisition rule may take a limited number. A calibration reference is a set of held-out cases used to compute the increments that a scoring rule is allowed to see, and a test reference is a disjoint set used only to report results. Cases are assigned to these roles at random, and the assignment is repeated five times; we call one such assignment a role allocation and report the spread across the five, since a single allocation reveals nothing about stability. A budget caps what an acquisition rule may take, either in whole cases or in recorded transitions, and the two units are not interchangeable.

## 3 Objects and valuation target

## 3.1 From traces to transition counts

Let $d = ( a _ { 1 } , \dotsc , a _ { L } )$ be a trace with $L \geq 2$ . Its count matrix is

$$
C _ { d } ( x , y ) = \sum _ { t = 1 } ^ { L - 1 } \mathbf { 1 } \{ a _ { t } = x , ~ a _ { t + 1 } = y \} , \qquad T _ { d } = L - 1 = \sum _ { x , y } C _ { d } ( x , y ) .\tag{3.1}
$$

For an archive $B ,$ write $\begin{array} { r } { N = \sum _ { d \in B } C _ { d } } \end{array}$ and $\begin{array} { r } { N _ { x } = \sum _ { y } N ( x , y ) } \end{array}$ . The row for x records which activities followed x. With K possible output labels and a smoothing constant $\alpha > 0$ , the predictor is

$$
p _ { N } ( y \mid x ) = \frac { N ( x , y ) + \alpha } { N _ { x } + K \alpha } .\tag{3.2}
$$

As described in Section 2, the α pseudo-counts keep every transition at positive probability and make (3.2) the posterior predictive distribution of a row-wise Dirichlet prior. We use the estimator without assuming that the underlying workflow is a stationary Markov process.

## 3.2 Reference distributions

A reference distribution $q ( x , y ) \geq 0$ with $\begin{array} { r } { \sum _ { x , y } q ( x , y ) = 1 } \end{array}$ describes the transitions on which the model is evaluated. Define

$$
U _ { q } ( N ) = \sum _ { x , y } q ( x , y ) \log p _ { N } ( y \mid x ) , \qquad \ell _ { q } ( N ) = - U _ { q } ( N ) .\tag{3.3}
$$

Higher $U _ { q }$ means lower log loss. Logarithms are natural, so the unit is a nat; one millinat is $1 0 ^ { - 3 }$ nats. For a nonempty finite reference set $R ,$ two weightings are natural:

$$
q _ { \mathrm { c } } ( x , y ) = \frac { 1 } { | R | } \sum _ { e \in R } \frac { C _ { e } ( x , y ) } { T _ { e } } ,\tag{3.4}
$$

$$
q _ { \mathrm { t } } ( x , y ) = \frac { \sum _ { e \in R } C _ { e } ( x , y ) } { \sum _ { e \in R } T _ { e } } .\tag{3.5}
$$

Equation (3.4) samples a case uniformly and then a transition within it. Equation (3.5) samples uniformly from all recorded transitions. If R contains one two-transition case and one ten-transition case, the two carry weights $1 / 2$ and $1 / 2$ under $q _ { \mathrm { c } }$ but $1 / 6$ and $5 / 6$ under $q _ { \mathrm { t } }$ . Neither objective is intrinsically preferable. The choice records whether errors matter per case or per recorded step, and Section 5 quantifies its efect.

## 4 An exact expression for individual value

Hold $N , q ,$ and α fixed. The marginal increment is

$$
v _ { q } ( d \mid N ) = U _ { q } ( N + C _ { d } ) - U _ { q } ( N ) .\tag{4.1}
$$

A positive value means that adding d reduces the reference log loss. This is a fixed-archive increment, not an average over coalitions of training examples as in Data Shapley [GZ19].

Write $\begin{array} { r } { A _ { x y } = N ( x , y ) + \alpha , A _ { x } = N _ { x } + K \alpha , c _ { x y } = C _ { d } ( x , y ) , c _ { x } = \sum _ { y } c _ { x y } } \end{array}$ , and $\begin{array} { r } { q _ { x } = \sum _ { y } q ( x , y ) } \end{array}$ The ratio of updated to original predictions is

$$
\frac { p _ { N + C _ { d } } ( y \mid x ) } { p _ { N } ( y \mid x ) } = \frac { A _ { x y } + c _ { x y } } { A _ { x y } } \cdot \frac { A _ { x } } { A _ { x } + c _ { x } } = \frac { 1 + c _ { x y } / A _ { x y } } { 1 + c _ { x } / A _ { x } } .\tag{4.2}
$$

Taking logarithms and averaging under q gives the exact increment

$$
v _ { q } ( d \mid N ) = \sum _ { x , y } q ( x , y ) \log \left( 1 + \frac { c _ { x y } } { A _ { x y } } \right) - \sum _ { x } q _ { x } \log \left( 1 + \frac { c _ { x } } { A _ { x } } \right) .\tag{4.3}
$$

The first term rewards counts added to particular transitions and is large when a transition matters under $q$ but is rare in the archive. The second accounts for row normalization: counts added to one outcome reduce the probability assigned to the others in that row. The two terms have opposite signs and their balance determines the sign of the increment. Additional data can increase the loss when the new counts move predictions away from the reference.

## 4.1 Divergence form and the attainable gain

Equation (4.3) is convenient for computation but uninformative about magnitude. A second form serves that purpose. For every row x with $q _ { x } > 0$ write $\rho _ { x } ( y ) = q ( x , y ) / q _ { x }$ for the reference conditional, $H ( \rho _ { x } )$ for its entropy, and $\begin{array} { r } { \bar { H } _ { q } = \sum _ { x } q _ { x } H ( \rho _ { x } ) } \end{array}$

Throughout, $v _ { q } ( C \mid N ) = U _ { q } ( N + C ) - U _ { q } ( N )$ also denotes the increment of an arbitrary nonnegative count matrix $C _ { i }$ , of which (4.1) is the case $C = C _ { d }$ . Sums over rows are understood to run over $\{ x : q _ { x } > 0 \}$ , on which $\rho _ { x }$ is defined.

Proposition 4.1 (Divergence form). For every nonnegative archive N and every reference q,

$$
\ell _ { q } ( N ) = \bar { H } _ { q } + \sum _ { x } q _ { x } D ( \rho _ { x } \| p _ { N } ( \cdot \mid x ) ) ,\tag{4.4}
$$

and for every nonnegative count matrix $C _ { i }$

$$
v _ { q } ( C \mid N ) = \sum _ { x } q _ { x } \Bigl [ D \bigl ( \rho _ { x } \parallel p _ { N } ( \cdot \mid x ) \bigr ) - D \bigl ( \rho _ { x } \parallel p _ { N + C } ( \cdot \mid x ) \bigr ) \Bigr ] .\tag{4.5}
$$

Proof. Group the sum in (3.3) by rows: $\begin{array} { r } { \ell _ { q } ( N ) = - \sum _ { x } q _ { x } \sum _ { y } \rho _ { x } ( y ) \log p _ { N } ( y \mid x ) } \end{array}$ . Adding and subtracting $\textstyle \sum _ { y } \rho _ { x } ( y )$ log $\rho _ { x } ( y )$ inside each row gives (4.4). Equation (4.5) is the diference of two instances of (4.4), in which the entropy term cancels because it does not depend on $N$ □

A trace therefore helps exactly to the extent that it moves the row predictors toward the reference conditionals in the q-weighted average. Since divergences are nonnegative, the decomposition also limits what acquisition can achieve.

Corollary 4.2 (Attainable gain). Let $\begin{array} { r } { \Lambda _ { q } ( N ) = \sum _ { x } q _ { x } D ( \rho _ { x } \| p _ { N } ( \cdot \mid x ) ) } \end{array}$ . Then $v _ { q } ( C \mid N ) \leq$ $\Lambda _ { q } ( N )$ for every nonnegative $C ,$ and ${ \mathrm { s u p } } _ { C } v _ { q } ( C \mid N ) = \Lambda _ { q } ( N )$ , the supremum being over nonnegative integer count matrices. It is attained only in the exceptional case that some admissible C makes $p _ { N + C } ( \cdot \mid x ) = \rho _ { x }$ for every row with $q _ { x } > 0$

Proof. The inequality follows from (4.5) and $D \geq 0$ , with equality only when the second divergence vanishes in every row of positive weight. For the supremum, take ${ \cal C } _ { M } ( x , y ) =$ $\lfloor M \rho _ { x } ( y ) \rfloor$ on rows with $q _ { x } > 0$ and zero elsewhere. Its row sums satisfy $M - K < c _ { x } \leq M$ , so $p _ { N + C _ { M } } ( y \mid x ) \to \rho _ { x } ( y )$ as $M \to \infty$ . Because smoothing keeps every $p _ { N + C _ { M } } ( y \mid x )$ bounded away from zero for fixed M, and the limit is approached uniformly on the finite label set, $D ( \rho _ { x } \| p _ { N + C _ { M } } ( \cdot \mid x ) ) \to 0 ;$ hence $v _ { q } ( C _ { M } \mid N ) \to \Lambda _ { q } ( N )$ □

Write $\eta _ { q } ( S \mid N ) = J _ { q } ( S \mid N ) / \Lambda _ { q } ( N )$ for the acquisition eficiency of a batch $S ,$ with $J _ { q }$ the joint gain defined in (7.1). The eficiency expresses an improvement relative to what was available, which a raw millinat diference does not. The supremum in Corollary 4.2 is taken over arbitrary nonnegative count matrices, not over batches that a budget admits or that any set of traces can realize, so $\Lambda _ { q } ( N )$ is an upper bound and not the value of an attainable optimum. The ceiling $\Lambda _ { q } ( N )$ is also computed from an empirical reference, so it bounds the stated objective on that reference set rather than on the underlying population. Section 9.2 reports both quantities for BPI 2012, where ${ \bar { H } } _ { q }$ accounts for three quarters of the initial loss.

## 4.2 A two-outcome example

Fix the two-label alphabet $\{ b , c \}$ , the active row $x = b , \alpha = 1 / 2$ , and counts $N ( x , b ) = 0$ $N ( x , c ) = 2$ . The initial probabilities under the Dirichlet-smoothed estimator (3.2) are $( 1 / 6 , 5 / 6 )$ . A one-transition trace $( x , b )$ changes them to $( 3 / 8 , 5 / 8 )$ . With $q ( x , b ) = r$ and $q ( x , c ) = 1 - r .$

$$
v ( r ) = r \log ( 9 / 4 ) + ( 1 - r ) \log ( 3 / 4 ) = r \log 3 + \log ( 3 / 4 ) .\tag{4.6}
$$

The trace helps precisely when $r > \log ( 4 / 3 ) / \log 3 \approx 0 . 2 6 2$ . At $r = 1 / 2$ its value is 0.2616 nats; at $r = 1 / 1 0$ it is $- 0 . 1 7 7 8$ nats. The same trace and the same archive $\mathrm { g i }$ ve opposite answers because the workload difers.

When updates are small relative to the smoothed counts, the first-order Taylor approximation $\log ( 1 + z )$ ≈ z turns (4.3) into $\textstyle \sum _ { x , y } q ( x , y ) c _ { x y } / A _ { x y } - \sum _ { x } q _ { x } c _ { x } / A _ { x }$ , a weighted count added per smoothed count minus its row penalty. The experiments use the exact formula throughout, including the sparse cells where this approximation is poor.

## 5 The efect of the reference weighting

The two weightings of Section 3.2 difer only in how the reference cases are pooled. Because $v _ { q }$ is linear in q, the gap between the two increments of the same candidate has a closed form.

Proposition 5.1 (Weighting identity). Let R be a finite reference set of cases with counts $C _ { e }$ and totals $T _ { e } \ge 1$ , and let $q _ { \mathrm { c } } , q _ { \mathrm { t } }$ be as in (3.4) and (3.5). For a nonnegative count matrix $C$ put $L _ { C } ( x , y ) = \log p _ { N + C } ( y \mid x )$ − log $p _ { N } ( y \mid x )$ and define the per-case benefit

$$
w _ { C } ( e ) = \sum _ { x , y } \frac { C _ { e } ( x , y ) } { T _ { e } } L _ { C } ( x , y ) .\tag{5.1}
$$

Then, with empirical mean and covariance taken over R under the uniform distribution,

$$
v _ { q _ { \mathrm { t } } } ( C \mid N ) - v _ { q _ { \mathrm { c } } } ( C \mid N ) = { \frac { \operatorname { C o v } _ { R } ( T , w _ { C } ) } { \mathbb { E } _ { R } [ T ] } } .\tag{5.2}
$$

Proof. Write $\widehat { C } _ { e } = C _ { e } / T _ { e }$ , so that $\begin{array} { r } { q _ { \mathrm { c } } = | \boldsymbol { R } | ^ { - 1 } \sum _ { e } \widehat { C } _ { e } } \end{array}$ and $\begin{array} { r } { q _ { \mathrm { t } } = \sum _ { e } ( T _ { e } / \sum _ { f } T _ { f } ) \widehat { C } _ { e } } \end{array}$ . Since $\begin{array} { r } { v _ { q } ( C \mid N ) = \sum _ { x , y } q ( x , y ) L _ { C } ( x , y ) } \end{array}$ is linear in q, and $\begin{array} { r } { \sum _ { x , y } \widehat { C } _ { e } ( x , y ) L _ { C } ( x , y ) = w _ { C } ( e ) } \end{array}$

$$
v _ { q _ { \mathrm { t } } } - v _ { q _ { \mathrm { c } } } = \sum _ { e \in R } \left( \frac { T _ { e } } { \sum _ { f } T _ { f } } - \frac { 1 } { | R | } \right) w _ { C } ( e ) = \frac { 1 } { \sum _ { f } T _ { f } } \sum _ { e \in R } \left( T _ { e } - \mathbb { E } _ { R } [ T ] \right) w _ { C } ( e ) ,\tag{5.3}
$$

and dividing numerator and denominator by $| R |$ gives (5.2).

The two objectives therefore disagree about a candidate only through the covariance between how long a reference case is and how much the candidate helps it, and coincide exactly where length and benefit are uncorrelated across the reference set. The identity applies to a batch as well, since $J _ { q }$ is also linear in q. The gap can have mean near zero across candidates and still spread further than the increments themselves, which is what the data show: in the earlier period the two increments of the same candidate have Spearman correlation −0.315, the covariance term has standard deviation 2.23 millinats against 1.59 for the case-weighted increment, and 33.1 per cent of candidates change sign. The Spearman coeficient is the Pearson correlation computed on the ranks of the two scores rather than on the scores themselves, so it measures agreement of the two orderings and is invariant under any increasing transformation of either. The weighting therefore leaves the average value of a trace almost unchanged and alters which traces are valuable.

## 6 Descriptor information and its limits

## 6.1 Prediction from restricted information

For a trace of length L with m distinct activities, $p _ { a }$ the relative frequency of activity a, r the number of adjacent repeated activities, and e the number of distinct directed transition pairs, we define the descriptor vector $F ( d ) \in \mathbb { R } ^ { 7 }$ as

$$
F ( d ) : = \left( \log ( 1 + L ) , \ \log ( 1 + m ) , \ - \sum _ { a } p _ { a } \log p _ { a } , \ { \frac { r } { L - 1 } } , \ { \frac { L - m } { L } } , \ { \frac { e } { m } } , \ { \frac { e } { m } } , \ \operatorname * { m a x } p _ { a } \right) .\tag{6.1}
$$

Its entries are log length, log distinct count, activity-frequency entropy, immediate-repeat fraction, revisit fraction, distinct edges per distinct activity, and maximum frequency share; no activity identities are retained. Repeated cases remain separate acquisition units even when their sequences coincide.

A regressor $f$ predicts $v _ { q } ( d \mid N )$ from $F ( d )$ . The archive and the reference are held fixed when its training labels are created, so their influence enters through the labels although they are not inputs to $f .$ Learning such value predictors has precedents in amortized attribution $\mathrm { [ C K L ^ { + } 2 4 ] } ;$ here the target labels are exact for a specified empirical reference.

To state the information limit, take a random candidate $D ,$ set $V = v _ { q } ( D \mid N )$ , and assume $V \in L ^ { 2 }$ . The best squared-error predictor is the conditional expectation $\mathsf { \bar { g } } ( F ) = \mathbb { E } [ V \mid F ]$ . For any square-integrable $f ( F )$ , expanding $V - f ( F ) = ( V - g ( F ) ) + ( g ( F ) - f ( F ) )$ gives

$$
\operatorname { \mathbb { E } } [ ( V - f ( F ) ) ^ { 2 } ] = \operatorname { \mathbb { E } } [ \operatorname { V a r } ( V \mid F ) ] + \operatorname { \mathbb { E } } [ ( g ( F ) - f ( F ) ) ^ { 2 } ] ,\tag{6.2}
$$

the cross term vanishing because $\operatorname { \mathbb { E } } [ V - g ( F ) \mid F ] = 0$ . The first term is information destroyed by the descriptors, the second estimation error above that limit; $( a , b )$ and $( a , c )$ share descriptors but touch diferent cells, so no deterministic function of $F$ predicts both exactly whenever their increments difer. This standard $L ^ { 2 }$ identity is useful here because its first term is estimable: grouping candidates with identical descriptor signatures gives a median estimated bound on the achievable $R ^ { 2 }$ of 0.951 in the earlier period, against 0.870 reached by the fitted forest. The estimate is optimistic in two ways. It resolves only exact descriptor collisions, so candidates with close but unequal signatures contribute nothing to the estimated conditional variance, and the collision groups are small, which biases a within-group variance downward. It is also unstable across role allocations, ranging from 0.891 to 0.965 while the fitted $R ^ { 2 }$ ranges from 0.444 to 0.885. In every allocation the gap between the fit and the estimated bound exceeds the gap between the bound and one, so the larger share of the prediction error is estimation error rather than information lost by the descriptors, but the margin is not uniform.

Prediction accuracy and selection quality are not the same criterion, and Section 9.3 shows them ordered oppositely for two regressors on this descriptor vector.

## 7 From singleton values to batches

For a set S of candidate cases the actual batch gain is

$$
J _ { q } ( S \mid N ) = U _ { q } \left( N + \sum _ { d \in S } C _ { d } \right) - U _ { q } ( N ) ,\tag{7.1}
$$

which is generally not $\textstyle \sum _ { d \in S } v _ { q } ( d \mid N )$ , because each new trace changes the archive against which the others contribute. In the calculation of (4.6) with $r = 1 / 2$ , two copies of $( x , b )$ give final probabilities $( 1 / 2 , 1 / 2 )$ and a joint gain of $\textstyle { \frac { 1 } { 2 } } \log ( 9 / 5 ) = 0 . 2 9 3 9$ nats against 0.5232 for twice the singleton value. The opposite sign occurs in the same archive and under the same reference. With $r = 1 / 2$ , the candidates $i = ( x , b )$ and $j = ( x , c )$ have singleton values 0.2616 and −0.1194 nats, which sum to 0.1422, while their joint gain is 0.2067 nats. A trace that is harmful on its own improves the batch, because it repairs the row normalization that the other distorts.

Proposition 7.1 (Sign of the pairwise interaction). Extend $U _ { q }$ to nonnegative real count matrices. For candidates with count matrices $C _ { i } , C _ { j }$ , row sums $\begin{array} { r } { c _ { i x } = \sum _ { y } C _ { i } ( x , y ) } \end{array}$ , and any nonnegative M,

$$
D ^ { 2 } U _ { q } ( M ) [ C _ { i } , C _ { j } ] = - \sum _ { x , y } \frac { q ( x , y ) C _ { i } ( x , y ) C _ { j } ( x , y ) } { ( M ( x , y ) + \alpha ) ^ { 2 } } + \sum _ { x } \frac { q _ { x } c _ { i x } c _ { j x } } { ( M _ { x } + K \alpha ) ^ { 2 } } .\tag{7.2}
$$

(a) $I f C _ { i }$ and $C _ { j }$ share no cell but share a row x with $q _ { x } > 0$ , then $D ^ { 2 } U _ { q } ( M ) [ C _ { i } , C _ { j } ] > 0$ : the candidates are complements and the joint gain exceeds the sum of the individual increments.

(b) If both add counts to a single common cell $( x , y )$ with $q _ { x } \ > \ 0$ and to no other cell, then the mixed derivative is nonpositive, so the candidates are substitutes, if and only if $\rho _ { x } ( y ) \geq p _ { M } ( y \mid x ) ^ { 2 }$

Consequently $S \mapsto J _ { q } ( S \mid N )$ is in general neither submodular nor supermodular; that $i s ,$ the gain of adding a candidate to a larger set is neither always smaller nor always larger than the gain of adding it to a smaller one.

Proof. Diferentiating $\begin{array} { r } { U _ { q } ( M ) = \sum _ { x , u } q ( x , y ) [ \log ( M ( x , y ) + \alpha ) - \log ( M _ { x } + K \alpha ) ] } \end{array}$ twice in the directions $C _ { i } , C _ { j }$ gives (7.2). For (a) the first sum is empty and the second is strictly positive. For (b) write $c _ { i } = C _ { i } ( x , y ) , c _ { j } = C _ { j } ( x , y ) , A _ { x y } = M ( x , y ) + \alpha$ and $A _ { x } = M _ { x } + K \alpha _ { : }$ ; then (7.2) equals $c _ { i } c _ { j } q _ { x } [ A _ { x } ^ { - 2 } - \rho _ { x } ( y ) A _ { x y } ^ { - 2 } ]$ , which is nonpositive exactly when $\rho _ { x } ( y ) \geq ( A _ { x y } / A _ { x } ) ^ { 2 } =$ $p _ { M } ( y \mid x ) ^ { 2 }$ . The two worked examples above realize both signs for one archive and one reference, so no single modularity direction can hold in general. □

Part (b) states that two traces reinforcing the same transition are substitutes when the reference conditional mass on that transition is at least the square of the probability the model currently assigns to it. Since $p \in [ 0 , 1 ]$ and therefore $p ^ { 2 } \leq p _ { ; }$ , the condition is weak, and substitution is the typical case for the transitions on which a selection rule concentrates. Part (a) accounts for the opposite outcome, in which a batch is worth more than the sum of its parts. Neither efect is visible in a ranking of singleton scores, and the failure of submodularity matters because the standard approximation guarantee for greedy maximization assumes it [NWF78]. Accumulated over a batch, the second-order terms are bounded.

Proposition 7.2 (A bound for nonadditivity). Fix a nonnegative archive N, a probability distribution q, and $\alpha > 0$ . For nonnegative candidate count matrices $C _ { i }$ with row sums $c _ { i x }$ define

$$
b _ { i j } = \sum _ { x , y } \frac { q ( x , y ) C _ { i } ( x , y ) C _ { j } ( x , y ) } { ( N ( x , y ) + \alpha ) ^ { 2 } } + \sum _ { x } \frac { q _ { x } c _ { i x } c _ { j x } } { ( N _ { x } + K \alpha ) ^ { 2 } } .\tag{7.3}
$$

Then for every finite candidate set $S _ { i }$

$$
\left| J _ { q } ( S \mid N ) - \sum _ { i \in S } v _ { q } ( i \mid N ) \right| \leq \sum _ { i < j , \ i , j \in S } b _ { i j } .\tag{7.4}
$$

Proof. Whenever $M \ \geq \ N$ entrywise, the triangle inequality applied to (7.2) bounds $| D ^ { 2 } U _ { q } ( M ) [ C _ { i } , C _ { j } ] |$ by $b _ { i j }$ . Order S arbitrarily. The joint gain telescopes into successive marginal increments. For the jth addition, compare its increment at $N + \textstyle \sum _ { i < j } C _ { i }$ with its increment at N and integrate the mixed derivative over the unit square, with one direction $\textstyle \sum _ { i < j } C _ { i }$ and the other $C _ { j }$ . Bilinearity and the derivative bound give an absolute diference of at most $\textstyle \sum _ { i < j } b _ { i j }$ . Summing over additions proves the claim □

The bound holds for a fixed batch under either budget rule. It is not an optimality or regret guarantee for ranking by individual scores under a transition constraint, and small smoothed counts make it loose; Section 9.4 reports how loose. The general gap between attribution and subset selection is discussed in $[ \mathrm { W Y Z ^ { + } 2 4 } ]$

## 8 Data and experimental design

## 8.1 Sources, roles, and evaluation

BPI Challenge 2012 records loan-application activity at a Dutch financial institution [vD12]. The released log has 13,087 cases and 262,200 events. We retain the 164,506 events marked COMPLETE, spanning 23 activity names. Events with equal timestamps keep their source order. Cases crossing the 60th percentile of case start times are removed: 7,097 cases remain in the earlier period, 5,235 in the later period, and 755 are purged.

Whole cases receive disjoint roles. Of the earlier cases, 1,419 form a base pool, 2,129 train the value regressor, 2,129 are acquisition candidates, and 710 each supply calibration and test references. Later cases split into 2,617 calibration and 2,618 test cases. The vocabulary is taken from the whole base pool with one extra unknown label, so $K = 2 4 ;$ ; the predictor’s counts use only the first 128 base cases. The smoothing constant is $\alpha = 1 / 2$

Labels for the value regressor use the earlier calibration reference. Two regressors are fitted to the same seven descriptors and the same labels: a random forest, an average of 100 regression trees fitted to bootstrap resamples with maximum depth 8 and minimum leaf size 5, and a ridge regression with penalty 1 on standardized inputs. Both are frozen after fitting and reused for both periods. They difer only in functional form, so a diference between them is a property of the fitted regressor and not of the information the descriptors carry. Test references supply the reported losses and the true increments, never training labels. Each reference weighting gets its own calibration labels and its own pair of regressors, so the weighting comparison evaluates corresponding pipelines rather than one ranking under two metrics.

## 8.2 Acquisition budgets

At a case budget of 20 per cent, a ranking selects its first ⌊0.2 · 2129⌋ = 425 cases. At a transition threshold of 20 per cent, it selects the shortest prefix whose cumulative transition count reaches 20 per cent of the candidate pool’s transitions; the final whole case may exceed the threshold. Random selection uses uniform subsets for case budgets and uniform random permutations with the same stopping rule for transition thresholds.

The distinction changes the comparison. At the primary allocation under case weighting, the transition rule selects 1,162 forest-ranked cases containing 4,381 transitions against 425.8 cases and 4,389.6 transitions for the random rule: matched transition volume, very diferent case counts. At a fixed 425-case budget, forest selection instead receives 1,355 transitions against 4,357.9. The two units encode diferent cost assumptions, and neither is a correction of the other.

<table><tr><td>Reference weight Budget unit</td><td></td><td>Earlier period</td><td>Later period</td></tr><tr><td>Cases</td><td>Cases</td><td> $+ 9 . 0 \ [ - 6 . 9 , + 1 4 . 3 ]$ </td><td>-22.5 [-44.2, -13.8]</td></tr><tr><td>Cases</td><td>Transitions</td><td>+39.6 [-16.4, +65.2]</td><td>+4.7 [-26.8, +36.5]</td></tr><tr><td>Transitions</td><td>Cases</td><td>-84.5 [-126.2, -63.1]</td><td>-56.1 [-82.4, -29.4]</td></tr><tr><td>Transitions</td><td>Transitions</td><td>-64.0 [-84.0, -58.0]</td><td>-41.9 [-48.0, -30.8]</td></tr></table>

Table 1: Forest gain over random in millinats: median [minimum, maximum] over five role allocations at the 20 per cent acquisition setting.

The budget restricts additions to the count model and excludes the reference data, the training of the regressor, and the inspection of candidate descriptors. The main comparison uses 199 random batches at the primary allocation and 39 at each of four further allocations, all reusing the same log and temporal boundary, so their ranges measure sensitivity to role assignment rather than uncertainty across organizations. Shortest-first and fewest-distinctactivities rankings serve as inexpensive baselines.

Analysis status. Case weighting with a case budget was specified in a locally dated analysis plan; it was not externally registered. The transition-budget and transition-weighted analyses were added after seeing the initial results, as were the ceiling, interaction, and sequential-rule analyses reported in Sections 9.2 to 9.4. The present focus on BPI is likewise retrospective. Both regressors were fitted in the same run and neither was dropped, but the decision to report the ridge comparison prominently was made after its results were seen, so the contrast of Section 9.3 is an observation to be replicated under a regressor fixed in advance, not a test.

## 9 Results

## 9.1 Selection under the two weightings and budgets

Under case weighting and the transition threshold, the forest batch at the primary allocation improves log loss over the mean random batch by 65.2 millinats in the earlier period and 26.0 in the later period; the gains over the unchanged archive are 171.6 and 106.4 millinats for the forest and for random selection in the earlier period. The comparison is less decisive against simple rules. Shortest-first gains 50.4 millinats over random in the earlier period and 13.8 in the later period, fewest-distinct 43.2 and 0.7, and across five allocations shortest-first has an earlier-period median advantage of 39.2 millinats against 39.6 for the forest. The favorable example does not establish that the forest beats an inexpensive length rule.

Two controls bound the alternative reading that the transition threshold rewards any rule preferring short cases. Ranking by descending length selects 115 cases and loses 216.0 millinats to random selection in the earlier period; a forest fitted to randomly permuted labels selects 222 cases and loses 244.9. Conversely, more cases do not imply a larger gain: at a matched transition volume shortest-first takes the most cases of any rule considered, a median of 1,328 against 1,138 for the forest and 1,266 for the ridge regression, and gains the least of the three. The case count is therefore not a suficient statistic for the outcome.

Figure 1 shows the spread across allocations. Under transition weighting both descriptor regressors lose to random selection in all five allocations at both budget units. Section 9.2 shows how much was available to be won in each configuration, and Section 9.3 separates the contribution of the fitted score from that of the selection principle.

<table><tr><td rowspan="2">Weight</td><td rowspan="2">Period</td><td colspan="3">Decomposition of the initial loss</td><td colspan="6">Share of the ceiling realized (%)</td></tr><tr><td> $\ell _ { q } ( N _ { 0 } )$ </td><td> $\textstyle { \bar { H } } _ { q }$ </td><td> $\Lambda _ { q }$ </td><td>Random</td><td>Forest Ridge Seq.</td><td></td><td></td><td>Oracle</td><td>All</td></tr><tr><td>Cases</td><td>Earlier</td><td>0.928</td><td>0.698</td><td>230</td><td>43</td><td>61</td><td>69</td><td>82</td><td>86</td><td>55</td></tr><tr><td>Cases</td><td>Later</td><td>0.957</td><td>0.753</td><td>204</td><td>48</td><td>54</td><td>65</td><td>81</td><td></td><td>61</td></tr><tr><td>Transitions</td><td>Earlier</td><td>1.156</td><td>0.991</td><td>165</td><td>70</td><td>30</td><td>30</td><td>70</td><td>79</td><td>89</td></tr><tr><td>Transitions</td><td>Later</td><td>1.168</td><td>1.002</td><td>166</td><td>71</td><td>44</td><td>47</td><td>71</td><td>79</td><td>90</td></tr></table>

Table 2: Loss decomposition (4.4) and acquisition eficiency at the 20 per cent transition budget, medians over five role allocations; $\ell _ { q } ( N _ { 0 } )$ and $\textstyle { \bar { H } } _ { q }$ in nats, $\Lambda _ { q }$ in millinats. The last six columns give $\eta _ { q }$ in per cent for random selection, the two descriptor regressors, the sequential calibration rule, the sequential test-score oracle, and the whole candidate pool added without any budget. $\Lambda _ { q }$ bounds arbitrary count additions, not budgeted ones, so these shares are not eficiencies against an attainable optimum.

## 9.2 Attainable gain and acquisition eficiency

Corollary 4.2 makes the magnitude of these diferences measurable. At the primary allocation under case weighting, the initial loss on the earlier test reference is $\ell _ { q } ( N _ { 0 } ) = 0 . 9 4 4$ nats, of which the reference’s own conditional entropy $\bar { H } _ { q } = 0 . 7 0 4$ nats cannot be removed by any amount of data. The attainable gain is $\Lambda _ { q } ( N _ { 0 } ) = 0 . 2 4 0$ nats, a quarter of the loss. Measured against that ceiling, the 65.2 millinat advantage of the forest over random selection is 27 per cent. These three figures are for the primary allocation; Table 2 reports medians over all five, where the ceiling is 230 millinats.

Table 2 reports the decomposition for all four configurations. Two observations follow.

The first concerns the room available to any selection rule. Conditional entropy accounts for 75 per cent of the loss under case weighting and 86 per cent under transition weighting, which leaves median ceilings of 230 and 165 millinats. Under case weighting random selection realizes 43 per cent of that ceiling, leaving most of it open to a rule; under transition weighting it already realizes 70 per cent of a smaller ceiling, leaving little. This is a property of the objective rather than of any estimator.

The second concerns unrestricted acquisition. Under case weighting, adding the whole candidate pool of 21,936 transitions without any budget realizes 55 per cent of the ceiling in the earlier period, less than the 61 per cent that the forest reaches with a fifth of them. Under transition weighting the ordering reverses and the unrestricted pool realizes 89 per cent. Equation (4.5) accounts for this. Under case weighting a median of 35 per cent of candidates have a negative increment and move the row predictors away from the reference conditionals, against 4 per cent under transition weighting. The pool figure also shows that $\Lambda _ { q }$ is not approached by the data available here: no budget-free addition of the entire pool comes within 40 per cent of it under case weighting.

## 9.3 A comparison of selection rules

The descriptor forest is one point on a scale from random selection to a rule that sees the evaluation reference itself. Table 3 places six further rules on that scale, at the same budget and against the same random comparators. Static rules rank candidates once; sequential rules recompute the exact increment of every remaining candidate after each addition and take the largest increment per transition, which costs about 2.4 million exact marginal evaluations for one weighting, scorer, and allocation.

Under case weighting, the instability of Table 1 belongs to the fitted forest and not to the descriptors. A ridge regression on the same seven descriptors, the same labels, and the same training cases is positive in all ten allocation-period cells and exceeds shortest-first at the median in both periods, neither of which the forest does. The forest collapses at one allocation, where its fitted $R ^ { 2 }$ falls to 0.444 against a median of 0.870; the ridge fit has no comparable outlier. Replacing one estimator of the same target by another therefore changes the qualitative conclusion, which is a reason to fix the estimator in advance rather than evidence that either form is preferable.

![](images/ebf0d0089d681893859eec4c09fe30fe96eb3091d27d8e7935f2dd6cf720bd5d.jpg)

<table><tr><td></td><td colspan="5">Case weighting</td><td colspan="4">Transition weighting</td></tr><tr><td>Selection rule</td><td colspan="4">Earlier</td><td colspan="2">Later</td><td colspan="2">Earlier</td><td colspan="2">Later</td></tr><tr><td>Descriptor forest</td><td></td><td>+40 [-16, +65]</td><td></td><td></td><td>+5 [-27, +36]</td><td></td><td>-64 [-84, -58]</td><td></td><td>-42 [-48, -31]</td></tr><tr><td>Descriptor ridge</td><td></td><td>+60 [+42, +79]</td><td></td><td></td><td>+30 [+19, +51]</td><td></td><td>-65 [-75, -52]</td><td></td><td>-38 [-45, -31]</td></tr><tr><td>Shortest first</td><td></td><td>+39 [+7, +50]</td><td></td><td></td><td>+10 [-25, +19]</td><td></td><td>-160 [-216, -147]</td><td></td><td>-169 [-229, -163]</td></tr><tr><td>Calibration score, static</td><td></td><td>+57 [+7, +72]</td><td></td><td></td><td>+28 [-10, +41]</td><td></td><td>-64 [-96, -57]</td><td></td><td>-41 [-61, -40]</td></tr><tr><td>Calibration score, sequential</td><td></td><td>+93 [+72, +95]</td><td></td><td></td><td>+63 [+55, +68]</td><td></td><td>+1 [-4, +2]</td><td></td><td>+3 [-2, +4]</td></tr><tr><td>Test-score oracle, sequential</td><td></td><td>+102 [+84, +105]</td><td></td><td></td><td>+72 [+64, +76]</td><td></td><td>+14 [+13, +17]</td><td></td><td>+15 [+13, +16]</td></tr><tr><td>Whole pool, no budget</td><td></td><td>+27 [+20, +30] +27 [+20, +30]</td><td></td><td></td><td></td><td></td><td>+32 [+29, +33]</td><td></td><td>+32 [+31, +32]</td></tr></table>

Table 3: Gain over the mean random batch in whole millinats at the 20 per cent transition budget: median [minimum, maximum] over five role allocations. Calibration-scored rules use earlier-period calibration data only; the oracle uses the test reference of the period it is evaluated on and is not implementable. The last row respects no budget.  
Figure 1: BPI 2012 at the 20 per cent transition budget. Each point is one role allocation; positive values favor the rule over the corresponding mean random batch. The descriptor forest crosses zero under case weighting and is uniformly negative under transition weighting. The ridge regression on the same descriptors and the sequential calibration rule stay positive in every allocation, the latter within 13 millinats of the oracle.

Accuracy on individual increments does not order the rules by the quality of the batches they select. The forest predicts single-trace values far better than the ridge regression in both periods, median $R ^ { 2 } 0 . 8 7 0$ against 0.623 and median Spearman correlation 0.927 against 0.726 in the earlier period, yet realizes the smaller share of the ceiling in Table 2. The two batches behave alike at the primary allocation, concentrating comparable mass in their five largest cells and realizing comparable fractions of the sum of their singleton scores. What difers is the variance of the fitted score across allocations.

Exact increments close most of the remaining gap. The sequential calibration rule is positive in all five allocations and both periods, and the oracle adds a further 9 millinats at the median and at most 13 in any allocation, so the implementable rule reaches 82 per cent of the ceiling against 86 for the oracle.

Under transition weighting, the objective leaves little room for any rule. The oracle gains only 14.4 and 14.9 millinats in the two periods, the sequential calibration rule is indistinguishable from random selection, and both descriptor regressors are negative, so the finding does not depend on the choice between them. The negative entries of Table 1 show descriptor scores misranking candidates for an objective whose weighting difers from the one their labels were computed under, as quantified by Proposition 5.1, while little is available to win even with perfect information.

The sequential rules require the candidate’s own count matrix and therefore assume that a candidate can be inspected before it is acquired, which is the case the descriptor route is intended to avoid; Table 3 compares selection quality, not total cost.

## 9.4 Batch interaction

Proposition 7.1 predicts that concentrated batches lose value to substitution. The measurement confirms this and gives its magnitude. At the primary allocation under case weighting and the transition budget, the forest batch of 1,162 cases has a joint gain of 171.6 millinats against a sum of individual increments of 2,516.4 millinats, which is 6.8 per cent of the sum of the singleton scores. One random comparator batch of 400 cases and comparable transition volume realizes 29.1 per cent, or 90.9 against 312.5 millinats; the 106.4 millinats quoted in Section 9.1 is the mean over the 199 random batches, not this one. The medians over five allocations are 5.6 and 27.6 per cent. The ridge batch behaves like the forest batch, at 7.1 per cent.

The mechanism is visible in the counts. The forest batch spreads its 4,381 added transitions over 85 distinct cells with 73.8 per cent of the mass in the five largest, while the random batch covers 117 cells with 36.6 per cent in the five largest. Both descriptor rules select many short, similar traces, and by Proposition 7.1(b) these are substitutes wherever the reference conditional mass exceeds the squared current probability, which holds for the cells they concentrate on. Selection by singleton score is therefore systematically optimistic, and the two descriptor rules are equally afected despite their diferent ranking accuracy, which is consistent with the interaction being a property of the concentration of the batch rather than of the score that produced it. The bound of Proposition 7.2 is valid but uninformative at this scale, since it permits 80.0 nats of nonadditivity against the 2.34 observed. Its content is qualitative. The departure from additivity is governed by pairwise products of counts weighted by inverse squared smoothed counts, which is why a batch concentrated on few sparse cells departs furthest.

## 9.5 Stability across evaluation periods

A ranking computed on one reference is then used on another. Since $v _ { q } ( C ~ \mid ~ N ) ~ =$ $\begin{array} { r } { \sum _ { x , y } q ( x , y ) L _ { C } ( x , y ) } \end{array}$ is linear in the reference and $q - q ^ { \prime }$ sums to zero, subtracting the midrange of $L _ { C }$ and applying Hölder’s inequality gives $| v _ { q } ( C \mid N ) - v _ { q ^ { \prime } } ( C \mid N ) | \leq \mathrm { T V } ( q , q ^ { \prime } )$ osc $( L _ { C } )$ where osc is the range of $L _ { C }$ over cells; the same argument applied to $L _ { C _ { i } } - L _ { C _ { i } }$ shows that a pair keeps its order whenever its margin exceeds $\mathrm { T V } ( q , q ^ { \prime } )$ osc $\left( L _ { C _ { i } } - L _ { C _ { j } } \right)$ . At the total variation of 0.0807 between the two test references under case weighting, the bound certifies none of the 2,128 adjacent pairs of the earlier-period ranking, because sparse cells make osc large. The realized instability is much smaller than the bound admits, in that 400 adjacent pairs, or 18.8 per cent, actually reverse. In this setting the transfer across periods has to be measured rather than bounded.

## 10 Interpretation and limitations

Under a case-weighted objective and a transition constraint, selection by trace value is worth roughly a quarter of the attainable gain over random selection, and most of that is reachable with exact increments computed on calibration data. The result does not extend to descriptor scores as a class. The fitted forest is unstable across role allocations and not clearly better than shortest-first, while a ridge regression on the same descriptors is positive in all ten allocation-period cells and beats shortest-first by about 20 millinats at the median, despite predicting individual increments considerably less accurately. The forest result therefore does not refute the descriptor route, and neither descriptor result should be relied on, because the choice between two estimators fitted in the same run changes the sign of the conclusion and was made after the outcomes were known. A replication should fix one regressor in advance and compare four arms, namely random selection, shortest-first, that regressor, and exact sequential scoring.

At the primary allocation none of the B = 199 random transition-budget batches matches the forest batch in either period, so the Monte Carlo tail probability $p _ { \mathrm { M C } } = ( 1 + \# \{ b : J _ { q } ( S _ { b } $ $N ) \geq J _ { q } ( S _ { \mathrm { f o r e s t } } \mid N ) \} ) / ( B + 1 )$ equals 0.005, the smallest attainable value at this simulation size. In this randomization test the null distribution is generated by the random-selection procedure rather than assumed, and the added unit in numerator and denominator keeps the test valid at finite B, so the value describes an extreme result relative to that procedure, conditional on the split, the candidate pool, and the test reference. Holm adjustment [Hol79], a step-down correction that controls the probability of any false rejection across a family of tests, raises it to 0.020 within the family of four dataset-period comparisons that the analysis plan specified. That correction does not account for the subsequent choice of a favorable weighting, budget unit, or case study, and none of the four originally planned case-budget comparisons across both logs passed the planned threshold. The configuration is therefore treated as exploratory.

The learner uses only the immediately preceding activity, so it models no long-range dependence, predicts no loan decision, and optimizes no interactive agent; agent-trajectory curation studies diferent objectives and models [ZYW<sup>+</sup>26]. The transition count is one acquisition cost among several, since privacy review, storage, and labeling carry case-level costs, and the sequential rules carry a scoring cost that the budget does not charge them.

Three sources of error act at once. An estimated score can difer from the exact increment, a singleton score computed exactly on a calibration reference can difer from the increment on a test reference, and even exact test-reference scores need not select the best joint batch, because of the interactions of Section 9.4. Moving from static to sequential exact scoring in Table 3 isolates the last of these, worth 36 millinats in the earlier period.

The five allocations overlap, use diferent numbers of random comparator batches, and come from one historical process, so they bound sensitivity to role assignment and nothing wider. The ceiling inherits the sampling error of the empirical reference it is computed from, and no interval is reported for it. A replication should also account for scoring costs and assess the final-case overshoot of the transition rule.

## 11 Conclusion

The exact increment separates the trace, the archive, and the reference distribution, and its divergence form bounds what any acquisition can achieve. On BPI 2012 that bound is a quarter of the initial loss. Random selection realizes just under half of it, and a sequential rule using exact calibration increments realizes most of it while remaining positive across role allocations. Descriptor ranking captures part of the same gain, and how much depends on which regressor is fitted to the same descriptors and labels, with the more accurate predictor of individual increments selecting the worse batches. Individual-score correlation is therefore insuficient evidence in either direction. A batch must be judged under the intended objective and acquisition rule, and against the gain that was available.

## References

[CKL<sup>+</sup>24] Ian Covert, Chanwoo Kim, Su-In Lee, James Zou, and Tatsunori Hashimoto. Stochastic amortization: A unified approach to accelerate feature and data attribution. In Advances in Neural Information Processing Systems, volume 37, pages 4374–4423, 2024. doi:10.52202/079017-0143.

[FSVP<sup>+</sup>23] Mohammadreza Fani Sani, Mozhgan Vazifehdoostirani, Gyunam Park, Marco Pegoraro, Sebastiaan J. van Zelst, and Wil M. P. van der Aalst. Performance-preserving event log sampling for predictive monitoring. Journal of Intelligent Information Systems, 61:53–82, 2023. doi:10.1007/s10844-022-00775-9.

[GZ19] Amirata Ghorbani and James Zou. Data Shapley: Equitable valuation of data for machine learning. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 2242–2251. PMLR, 2019. URL: https: //proceedings.mlr.press/v97/ghorbani19c.html.

[Hol79] Sture Holm. A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6(2):65–70, 1979. URL: https://www.jstor.org/stable/4615733.

[KL17] Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1885–1894. PMLR, 2017. URL: https://proceedings.mlr. press/v70/koh17a.html.

[KT81] Raphail E. Krichevsky and Victor K. Trofimov. The performance of universal encoding. IEEE Transactions on Information Theory, 27(2):199–207, 1981. doi:10.1109/TIT.1981.1056331.

[NWF78] George L. Nemhauser, Laurence A. Wolsey, and Marshall L. Fisher. An analysis of approximations for maximizing submodular set functions—I. Mathematical Programming, 14(1):265–294, 1978. doi:10.1007/BF01588971.

[TDLRM19] Irene Teinemaa, Marlon Dumas, Marcello La Rosa, and Fabrizio Maria Maggi. Outcomeoriented predictive process monitoring: Review and benchmark. ACM Transactions on Knowledge Discovery from Data, 13(2):1–57, 2019. doi:10.1145/3301300.

[vD12] Boudewijn van Dongen. BPI challenge 2012. 4TU.ResearchData, dataset, version 1, 2012. doi:10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f.

[WYZ<sup>+</sup>24] Jiachen T. Wang, Tianji Yang, James Zou, Yongchan Kwon, and Ruoxi Jia. Rethinking data Shapley for data selection tasks: Misleads and merits. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 52033–52063. PMLR, 2024. URL: https://proceedings.mlr.press/v235/wang24cg.html.

[ZYW<sup>+</sup>26] Dewu Zheng, Ruizhe Ye, Yanlin Wang, Yang Ye, Hongyu Zhang, Ensheng Shi, Xilin Liu, Yuchi Ma, Jianxing Yu, and Zibin Zheng. SWE-Prime: Fewer trajectories, better performance. arXiv:2608.27449, 2026. Preprint. doi:10.48550/arXiv.2608.27449.