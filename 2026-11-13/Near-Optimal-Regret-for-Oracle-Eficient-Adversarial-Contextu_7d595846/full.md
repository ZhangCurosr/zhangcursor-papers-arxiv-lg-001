# Near-Optimal Regret for Oracle-Eficient Adversarial Contextual Bandits

Kiarash Banihashem University of Maryland, College Park kiarash@umd.edu

MohammadTaghi Hajiaghayi University of Maryland, College Park hajiagha@umd.edu

## Abstract

We study contextual bandits with K actions, a class of N policies, and i.i.d. contexts from a distribution that the learner can sample. For any sequence of loss functions fixed before play, we give an oracle-eficient algorithm with expected regret $O ( \sqrt { K T \log ( 2 N ) } \log ( 2 K T ) )$ against the best policy in hindsight on the realized contexts. This improves the $T ^ { 2 / 3 }$ dependence on the horizon in prior oracle-eficient bounds and achieves the optimal regret rate in this setting up to logarithmic factors.

Our algorithm perturbs policy scores using a Poisson sample of auxiliary contexts. At each sampled context, the perturbation assigned to an action is a Laplace noise draw multiplied by a scale specific to that context and action. We show that choosing these scales to minimize a convex objective determined by past observations yields the stated regret bound. We further approximate the minimizing scales by projected stochastic gradient descent using only an exact ofline cost-sensitive optimization oracle that returns the optimum value. Since the table of scales can be large, the algorithm represents the SGD iterates implicitly by storing the random draws that determine each update and evaluates the scales only at queried contexts. Both the oracle complexity and the remaining resource bounds are polynomial in T, K, and log(2N).

## 1 Introduction

In a contextual bandit problem, the learner observes a context on each of T rounds, chooses one of K actions, and observes only the loss of that action. Regret is measured against the best policy in hindsight from a class of N policies. The classical Exp4 algorithm achieves expected regret $O ( { \sqrt { K T \log N } } )$ by maintaining a weight for each policy [ACBFS02]. When the policy class is large, a natural question is whether one can achieve this rate using only an ofline optimization oracle for the class. When contexts and loss vectors are drawn jointly i.i.d., Langford and Zhang [LZ07] obtained a regret bound with $T ^ { 2 / 3 }$ dependence on the horizon, and Agarwal et al. $[ \mathrm { A H K ^ { + } 1 4 } ]$ achieved the optimal square-root rate using an ofline optimization oracle.

Achieving square-root regret with an ofline optimization oracle becomes much more challenging however when losses are adversarial. In this setting, Rakhlin and Sridharan [RS16] used auxiliary context samples to obtain expected regret $O ( T ^ { 3 / 4 } K ^ { 1 / 2 } ( \log N ) ^ { 1 / 4 } )$ . Syrgkanis et al. [SLKS16] improved the dependence on the horizon to $T ^ { 2 / 3 }$ , with regret $O ( ( K T ) ^ { 2 / 3 } ( \log N ) ^ { 1 / 3 } )$ Banihashem et al. [BHSS23] then improved the dependence on the number of actions, obtaining $O ( T ^ { 2 / 3 } ( K \log N ) ^ { 1 / 3 } )$ regret. More recently, Buzaglo and Hazan [BH26] obtained ${ \widetilde O } ( T ^ { 2 / 3 } )$ dependence on the horizon without requiring auxiliary context samples. Despite these advances, achieving the optimal $\sqrt { T }$ dependence on the horizon with an oracle-eficient algorithm has remained an outstanding open problem.

In this paper, we resolve this problem up to logarithmic factors. We give an oracle-eficient algorithm with expected regret

$$
O \Big ( \sqrt { K T \log ( 2 N ) } \log ( 2 K T ) \Big ) .
$$

Following prior work, we consider a hybrid setting in which contexts are drawn i.i.d. and the adversary fixes an arbitrary sequence of loss functions before play. The learner can draw independent auxiliary samples from the context distribution. The learner has access to independent samples from the context distribution and to an exact ofline oracle that returns the minimum total cost over the policy class. Both the number and the size of the oracle queries are polynomial in T, K, and log(2N) (Theorem 1).

## 2 Model and Main Result

Let $T \geq 1 , K \geq 2$ , and let Π be a nonempty finite class of N measurable policies $\pi : \mathcal { X }  [ K ]$ Before play the adversary fixes measurable losses $c _ { t } : \mathcal { X } \to [ 0 , 1 ] ^ { K }$ . The actual contexts $x _ { t }$ are independent draws from D. After observing $x _ { t } ,$ the learner chooses $q _ { t } ,$ , draws $A _ { t } \sim q _ { t }$ with fresh randomness, and observes only $c _ { t } ( x _ { t } , A _ { t } )$ . Independent auxiliary samples from D are available. Given a list of contexts and real cost vectors, the value oracle returns

$$
\mathrm { V a l } ( ( y _ { i } , v _ { i } ) _ { i } ) = \operatorname* { m i n } _ { \pi \in \Pi } \sum _ { i } v _ { i } ( \pi ( y _ { i } ) ) .
$$

We count value-oracle calls, auxiliary context samples, draws from the specified elementary distributions, and exact real-arithmetic operations. The learner is given $L = \log ( 2 N )$ or an upper bound $L _ { 0 } \ge \log ( 2 N )$ . Each logarithm and square root used during initialization counts as one exact real-arithmetic operation.

Theorem 1. Let $L = \log ( 2 N )$ . Algorithm 2, tuned with L, satisfies

$$
\mathbb { E } \mathrm { R e g } _ { T } : = \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } c _ { t } ( x _ { t } , A _ { t } ) - \operatorname* { m i n } _ { \pi \in \Pi } \sum _ { t = 1 } ^ { T } c _ { t } ( x _ { t } , \pi ( x _ { t } ) ) \right] \leq C \sqrt { K T L } \log ( 2 K T )
$$

for a universal constant C. Its number of oracle calls, query sizes, auxiliary context samples, storage and real-arithmetic operations are polynomial in $T , K , L$ . A supplied upper bound $L _ { 0 } \ge L$ may replace L in both the tuning and the bounds.

Section 3 reduces regret to a one-round inequality for the ideal action distribution. Section 4 proves its stability estimate. Section 5 implements the distribution to suficient accuracy to add at most one to the regret bound.

## 3 Ideal Strategy and Regret Reduction

Set

$$
\gamma = ( T + 1 ) ^ { - 1 } , \quad B = K / \gamma , \quad \ell = \log ( e K / \gamma ) , \quad n = \ell ^ { - 1 } \sqrt { T L / K } , \quad \lambda = 4 n \ell .\tag{1}
$$

At the start of round t the score of $\pi$ is $\begin{array} { r } { S ( \pi ) = S _ { t - 1 } ( \pi ) = - \sum _ { i < t } \hat { c } _ { i } ( \pi ( x _ { i } ) ) } \end{array}$ , where $S _ { 0 } = 0$ and

$$
\hat { c } _ { t } = e _ { A _ { t } } c _ { t } ( x _ { t } , A _ { t } ) / q _ { t } ( A _ { t } ) .\tag{2}
$$

Algorithm 1 Poisson playout with exact action probabilities   
1. Set $\overline { { S _ { 0 } ( \pi ) = 0 } }$ for every $\pi \in \Pi$ . Use the parameters in (1).   
2. For $t = 1 , \dots , T \colon$   
(a) Choose $\sigma \in$ arg min $\mathrm { \ i } _ { \sigma } J _ { S _ { t - 1 } } ( \sigma )$   
(b) Receive $x = x _ { t }$ . For each $a \in [ K ]$ , compute   
$p _ { a } ( x ) = \operatorname* { P r } \biggl [ \biggl ( \underset { \pi \in \Pi } { \arg \operatorname* { m a x } } \big \{ S _ { t - 1 } ( \pi ) + P _ { \pi } ( \sigma ) + \sigma _ { \pi ( x ) } ( x ) Z _ { \pi ( x ) } \big \} \biggr ) ( x ) = a \biggr ]$   
The probability averages over the auxiliary pool defining $P _ { \pi } ( \sigma )$ and the independent   
vector $Z \sim \mu ,$ holding $S _ { t - 1 } , \sigma , x$ fixed.   
(c) Set $q _ { t } ( a ) = ( 1 - \gamma ) p _ { a } ( x ) + \gamma / K$ for each a. Draw $A _ { t } \sim q _ { t }$ and observe $c _ { t } ( x , A _ { t } )$   
(d) Update the policy scores by   
$S _ { t } ( \pi ) = S _ { t - 1 } ( \pi ) - { \frac { c _ { t } ( x , A _ { t } ) } { q _ { t } ( A _ { t } ) } } { \bf 1 } \{ \pi ( x ) = A _ { t } \} , \qquad \pi \in \Pi .$   
Define for any distribution $q$ with full support, with $c _ { a } = c _ { t } ( x , a )$ and $f _ { a } ( \pi ) = \mathbf { 1 } \{ \pi ( x ) = a \}$   
$\Phi _ { S , x } ( q ) = \sum _ { a } q _ { a } \left[ c _ { a } + R ( S - ( c _ { a } / q _ { a } ) f _ { a } ) - R ( S ) \right] .$   
This is the conditional mean of the played loss plus the change in $R .$

A pool consists of $M \sim \operatorname { P o i s } ( n )$ contexts $X _ { i } \sim D$ and vectors $Z _ { i } \sim \mu = \mathrm { L a p } ( 1 ) ^ { \otimes K }$ , all independent conditional on M. The one-dimensional density is $e ^ { - | z | } / 2$ . For a measurable scale $\sigma : \mathcal { X } \to [ 1 , B ] ^ { K }$ define

$$
\begin{array} { r l } & { \displaystyle P _ { \pi } ( \sigma ) = \sum _ { i = 1 } ^ { M } \sigma _ { \pi ( X _ { i } ) } ( X _ { i } ) Z _ { i , \pi ( X _ { i } ) } , } \\ & { \displaystyle F ( S , \sigma ) = \mathbb { E } \operatorname* { m a x } _ { \pi \in \Pi } \big [ S ( \pi ) + P _ { \pi } ( \sigma ) \big ] , } \\ & { \displaystyle J _ { S } ( \sigma ) = F ( S , \sigma ) + \lambda \mathbb { E } _ { X \sim D } \sum _ { a = 1 } ^ { K } \log \frac { B } { \sigma _ { a } ( X ) } , \qquad R ( S ) = \operatorname* { i n f } _ { \sigma } J _ { S } ( \sigma ) . } \end{array}
$$

Choose a minimizing scale σ before observing $x = x _ { t }$ (Lemma 6). With an independent $Z \sim \mu ,$ let

$$
W _ { x } = \mathop { \arg \operatorname* { m a x } } _ { \pi } \{ S ( \pi ) + P _ { \pi } ( \sigma ) + \sigma _ { \pi ( x ) } ( x ) Z _ { \pi ( x ) } \} .\tag{3}
$$

A fixed order breaks ties. Conditional on $S , x$ define

$$
p _ { a } ( x ) = \mathrm { P r } ( W _ { x } ( x ) = a ) , \qquad q _ { a } ^ { * } ( x ) = ( 1 - \gamma ) p _ { a } ( x ) + \gamma / K .\tag{4}
$$

Algorithm 1 gives the ideal strategy. Section 5 implements its scale minimization and action probabilities using a value oracle.

Theorem 2 (Ideal one-round bound). At every score $S$ and for every fixed measurable current loss function,

$$
\begin{array} { r } { \mathbb { E } _ { x \sim D } \Phi _ { S , x } ( q ^ { * } ( x ) ) \le u : = \gamma + 6 \log ( N ) / n + 3 K / B . } \end{array}
$$

The only stability estimate needed for this theorem is Lemma 3. For $q = q ^ { * } ( x )$ set $b _ { a } = c _ { a } / q _ { a }$ and define

$$
W _ { a } ^ { - } = \arg \operatorname* { m a x } _ { \pi } [ S ( \pi ) + P _ { \pi } ( \sigma ) - b _ { a } f _ { a } ( \pi ) ] , \qquad p _ { a } ^ { - } = \mathrm { P r } ( W _ { a } ^ { - } ( x ) = a ) .
$$

Lemma 3 (One-round stability). Fix a round t and an incoming score $S = S _ { t - 1 }$ . Let σ be the minimizing scale chosen by the ideal strategy at S. There exist nonnegative functions $D _ { a } : \mathcal { X } $ $[ 0 , \infty ) , a \in [ K ]$ , depending on $S$ and $\sigma ,$ , such that

$$
\mathbb { E } _ { x \sim D } \left[ \sum _ { a = 1 } ^ { K } D _ { a } ( x ) \right] \leq \frac { \log N } { n } .
$$

For every fixed measurable loss function $c _ { t } : \mathcal { X } \to [ 0 , 1 ] ^ { K }$ , let $p _ { a } ^ { - } ( x )$ be defined as above with $b _ { a } = c _ { t } ( x , a ) / q _ { a } ^ { * } ( x )$ . Then, for D-almost every x, all $a \in [ K ]$ satisfy

$$
p _ { a } ( x ) - p _ { a } ^ { - } ( x ) \leq 6 D _ { a } ( x ) + \frac { 3 } { B } .
$$

Proof of Theorem $\mathcal { Q }$ given Lemma 3. If $A _ { t } = a$ , then $S _ { t } = S - b _ { a } f _ { a }$ . At the realized ${ \cal W } _ { a } ^ { - }$

$$
\begin{array} { r l } & { \underset { \pi } { \operatorname* { m a x } } [ S ( \pi ) + P _ { \pi } ( \sigma ) ] \geq S ( W _ { a } ^ { - } ) + P _ { W _ { a } ^ { - } } ( \sigma ) } \\ & { \qquad = \underset { \pi } { \operatorname* { m a x } } [ S ( \pi ) + P _ { \pi } ( \sigma ) - b _ { a } f _ { a } ( \pi ) ] + b _ { a } f _ { a } ( W _ { a } ^ { - } ) . } \end{array}
$$

Here ${ \cal W } _ { a } ^ { - }$ and both maxima use the same pool realization. The same scale is feasible after the update. Its penalty cancels on subtraction, giving

$$
\begin{array} { r l } & { R ( S - b _ { a } f _ { a } ) - R ( S ) \leq J _ { S - b _ { a } f _ { a } } ( \sigma ) - J _ { S } ( \sigma ) } \\ & { \phantom { x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x } = F ( S - b _ { a } f _ { a } , \sigma ) - F ( S , \sigma ) } \\ & { \phantom { x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x } \leq - b _ { a } \mathbb { E } f _ { a } ( W _ { a } ^ { - } ) = - b _ { a } p _ { a } ^ { - } . } \end{array}
$$

Use $q _ { a } b _ { a } = c _ { a }$ and then $c _ { a } \in [ 0 , 1 ]$ to obtain

$$
\begin{array} { l } { \Phi _ { S , x } ( q ) \le \displaystyle \sum _ { a } q _ { a } ( c _ { a } - b _ { a } p _ { a } ^ { - } ) = ( q - p ) \cdot c + \displaystyle \sum _ { a } c _ { a } ( p _ { a } - p _ { a } ^ { - } ) } \\ { \le \gamma \left( K ^ { - 1 } \displaystyle \sum _ { a } c _ { a } - p \cdot c \right) + 6 \displaystyle \sum _ { a } D _ { a } ( x ) + 3 K / B . } \end{array}
$$

The first term is at most γ. Averaging over x and applying Lemma 3 gives

$$
\mathbb { E } _ { x } \Phi _ { S , x } ( q ^ { * } ( x ) ) \le \gamma + 6 \mathbb { E } _ { x } \sum _ { a } D _ { a } ( x ) + 3 K / B \le \gamma + 6 \log ( N ) / n + 3 K / B .
$$

Lemma 4 (Accumulation of one-round error). Suppose the actually played vector $q _ { t }$ satisfies $\mathbb { E } [ \Phi _ { S _ { t - 1 } , x _ { t } } ( q _ { t } ) \mid \mathcal { H } _ { t - 1 } ] \leq u + \xi$ at every round. Here $\mathcal { H } _ { t - 1 }$ is the past before observing $x _ { t }$ . Then

$$
\mathbb { E } \mathrm { R e g } _ { T } \leq K n + K \lambda \log B + T ( u + \xi ) .
$$

Proof. At scale $\begin{array} { r } { \sigma \equiv 1 , P _ { \pi } ( 1 ) \leq \sum _ { i , a } | Z _ { i , a } | } \end{array}$ for every $\pi ,$ so $R ( 0 ) \leq K n + K \lambda$ log B. At any score $S _ { i }$ the penalty is nonnegative, and $\mathbb { E } P _ { \pi } ( \sigma ) = 0$ at fixed $\pi , \sigma$ . Hence $R ( S ) \geq \operatorname* { m a x } _ { \pi } S ( \pi )$ . Since the loss functions are fixed before play and the action uses fresh randomness,

$$
\begin{array} { r } { \mathbb { E } \big [ \hat { c } _ { t } ( a ) \mid x _ { 1 : T } , \mathcal { H } _ { t - 1 } , q _ { t } \big ] = q _ { t } ( a ) c _ { t } ( x _ { t } , a ) / q _ { t } ( a ) = c _ { t } ( x _ { t } , a ) . } \end{array}
$$

Consequently,

$$
\mathbb { E } [ R ( S _ { T } ) \mid x _ { 1 : T } ] \ge \operatorname* { m a x } _ { \pi } \mathbb { E } [ S _ { T } ( \pi ) \mid x _ { 1 : T } ] = - \operatorname* { m i n } _ { \pi } \sum _ { t } c _ { t } ( x _ { t } , \pi ( x _ { t } ) ) .
$$

Telescoping the potential along the realized scores gives

$$
\begin{array} { r l r } {  { \mathbb { E } \mathrm { R e } \mathbf { g } _ { T } \le \mathbb { E } [ \sum _ { t } c _ { t } ( x _ { t } , A _ { t } ) + R ( S _ { T } ) ] } } \\ & { } & { = R ( 0 ) + \sum _ { t } \mathbb { E } \Phi _ { S _ { t - 1 } , x _ { t } } ( q _ { t } ) \le R ( 0 ) + T ( u + \xi ) . } \end{array}
$$

Put $A = \sqrt { K T L }$ . Under (1), $K n = A / \ell , K \lambda = 4 A , T L / n = A \ell ,$ , and $T \gamma = T K / B \le 1$ . Thus the lemma gives $O ( A \log ( 2 K T ) ) + T \xi$ . Theorem 2 gives $\xi = 0$ for ideal play; Theorem 11 below gives $\xi = 1 / T$ for oracle play.

## 4 Proof of One-round Stability

Lemma 3 requires two estimates. First, we define $D _ { a } ( x )$ and prove

$$
\mathbb { E } _ { x \sim D } \sum _ { a = 1 } ^ { K } D _ { a } ( x ) \leq \frac { \log N } { n } .
$$

We then use the optimality of the scale $\sigma$ to prove that, for D-almost every x and each $a \in [ K ]$

$$
p _ { a } ( x ) - p _ { a } ^ { - } ( x ) \leq 6 D _ { a } ( x ) + \frac { 3 } { B } .
$$

## 4.1 Information in the inserted context

Lemma 5 (Information carried by one inserted vector). Fix the score $S$ and the scale σ. For each context x, let $C = ( M , X _ { 1 : M } )$ be the auxiliary pool locations, let $Z \sim \mu$ be the independent Laplace vector inserted at x, and let $W _ { x }$ be the winning policy in (3).

For each $\pi \in \Pi$ , write $w _ { \pi } = \operatorname* { P r } ( W _ { x } = \pi \mid C )$ . When $w _ { \pi } > 0$ , define the conditional distribution ν<sub>π</sub> by

$$
\nu _ { \pi } ( A ) = \operatorname* { P r } ( Z \in A \mid C , W _ { x } = \pi ) \qquad f o r e v e r y m e a s u r a b l e A \subseteq \mathbb { R } ^ { K }
$$

For each action $a \in [ K ]$ , set

$$
D _ { a } ( x ) = \mathbb { E } _ { C } \left[ \sum _ { \pi \in \Pi : \pi ( x ) = a } w _ { \pi } \operatorname { K L } ( \nu _ { \pi } \| \mu ) \right] .
$$

Then

$$
\mathbb { E } _ { x \sim D } \left[ \sum _ { a = 1 } ^ { K } D _ { a } ( x ) \right] \leq \frac { \log N } { n } .
$$

Proof. For each $m \geq 1$ , let $( X _ { 1 } , Z _ { 1 } ) , \dots , ( X _ { m } , Z _ { m } )$ be independent pairs with law $D \otimes \mu ,$ , and define

$$
W _ { m } = \underset { \pi \in \Pi } { \arg \operatorname* { m a x } } \left\{ S ( \pi ) + \sum _ { i = 1 } ^ { m } \sigma _ { \pi ( X _ { i } ) } ( X _ { i } ) Z _ { i , \pi ( X _ { i } ) } \right\} ,
$$

using the same tie-breaking rule as in (3). Write $\rho _ { m } = \mathrm { P r } ( M = m ) = e ^ { - n } n ^ { m } / m !$

$$
n \mathbb { E } _ { x \sim D } \left[ \sum _ { a = 1 } ^ { K } D _ { a } ( x ) \right] = \sum _ { m \geq 1 } \rho _ { m } \sum _ { i = 1 } ^ { m } I ( Z _ { i } ; W _ { m } \mid X _ { 1 : m } ) .
$$

Proof of Claim 5.1. Fix x and the pool locations C. Write $I _ { x , C } ( Z ; W _ { x } )$ for the mutual information under these fixed locations. The inserted vector Z has law $\mu .$ Conditional on $W _ { x } = \pi$ , it has law $\nu _ { \pi }$ . Thus

$$
I _ { x , C } ( Z ; W _ { x } ) = \sum _ { \pi : w _ { \pi } > 0 } w _ { \pi } \operatorname { K L } ( \nu _ { \pi } \| \mu ) .
$$

Grouping the policies by their action at x and averaging over C gives

$$
\mathbb { E } _ { C } [ I _ { x , C } ( Z ; W _ { x } ) ] = \sum _ { a = 1 } ^ { K } D _ { a } ( x ) .
$$

Now average over an independent $x \sim D$ and condition on $M = k$ . Since $C = ( k , X _ { 1 : k } )$ on this event,

$$
\mathbb { E } _ { \boldsymbol { x } \sim D } \left[ \sum _ { a = 1 } ^ { K } D _ { a } ( \boldsymbol { x } ) \right] = \sum _ { k \ge 0 } \rho _ { k } \mathbb { E } _ { \boldsymbol { x } , X _ { 1 : k } } \left[ I _ { \boldsymbol { x } , ( k , X _ { 1 : k } ) } ( Z ; W _ { x } ) \right] .\tag{5}
$$

Fix k. Label the inserted pair $( X _ { k + 1 } , Z _ { k + 1 } ) = ( x , Z )$ . The resulting $k + 1$ pairs are independent with law $D \otimes \mu$ . For every policy π, its score in the inserted experiment is

$$
\begin{array} { l } { { \displaystyle S ( \pi ) + \sum _ { i = 1 } ^ { k } \sigma _ { \pi ( X _ { i } ) } ( X _ { i } ) Z _ { i , \pi ( X _ { i } ) } + \sigma _ { \pi ( x ) } ( x ) Z _ { \pi ( x ) } } } \\ { { \displaystyle \quad \quad = S ( \pi ) + \sum _ { i = 1 } ^ { k + 1 } \sigma _ { \pi ( X _ { i } ) } ( X _ { i } ) Z _ { i , \pi ( X _ { i } ) } . } } \end{array}
$$

Hence $W _ { x } = W _ { k + 1 }$ under this labeling. In particular, at fixed locations $u _ { 1 } , \ldots , u _ { k + 1 }$

$$
I _ { u _ { k + 1 } , ( k , u _ { 1 : k } ) } ( Z ; W _ { u _ { k + 1 } } ) = I ( Z _ { k + 1 } ; W _ { k + 1 } \mid X _ { 1 : k + 1 } = u _ { 1 : k + 1 } ) .
$$

By definition, averaging the right-hand side over the locations gives conditional mutual information. Therefore

$$
\begin{array} { l } { \mathbb { E } _ { x , X _ { 1 : k } } \left[ I _ { x , ( k , X _ { 1 : k } ) } ( Z ; W _ { x } ) \right] = \displaystyle \int I ( Z _ { k + 1 } ; W _ { k + 1 } \mid X _ { 1 : k + 1 } = u _ { 1 : k + 1 } ) D ^ { \otimes ( k + 1 ) } ( d u _ { 1 : k + 1 } ) } \\ { = I ( Z _ { k + 1 } ; W _ { k + 1 } \mid X _ { 1 : k + 1 } ) . } \end{array}
$$

Substitute into (5), multiply by $n ,$ and set $m = k + 1$ . The Poisson probabilities satisfy $n \rho _ { m - 1 } = m \rho _ { m } ,$ , so

$$
\begin{array} { r l } { n \mathbb { E } _ { x \sim D } \Bigg [ \underset { a = 1 } { \overset { K } { \sum } } D _ { a } ( x ) \Bigg ] = \underset { k \geq 0 } { \sum } n \rho _ { k } I ( Z _ { k + 1 } ; W _ { k + 1 } \mid X _ { 1 : k + 1 } ) } & { } \\ { = \underset { m \geq 1 } { \sum } n \rho _ { m - 1 } I ( Z _ { m } ; W _ { m } \mid X _ { 1 : m } ) } & { } \\ { = \underset { m \geq 1 } { \sum } \rho _ { m } m I ( Z _ { m } ; W _ { m } \mid X _ { 1 : m } ) . } \end{array}
$$

Finally, fix m and $i \in [ m ]$ . Swapping pairs i and m preserves their joint law. It also preserves $W _ { m }$ , since it only reorders the summands in every policy’s score. The swap exchanges $Z _ { i }$ and $Z _ { m }$ and reorders the conditioning vector $X _ { 1 : m }$ . This reordering is invertible, so it preserves the conditioning information. Consequently,

$$
I ( Z _ { m } ; W _ { m } \mid X _ { 1 : m } ) = I ( Z _ { i } ; W _ { m } \mid X _ { 1 : m } ) , \qquad i \in [ m ] .
$$

Summing over i gives

$$
m I ( Z _ { m } ; W _ { m } \mid X _ { 1 : m } ) = \sum _ { i = 1 } ^ { m } I ( Z _ { i } ; W _ { m } \mid X _ { 1 : m } ) ,
$$

which proves the claim.

It remains to bound the inner sum in Claim 5.1. Fix $m \geq 1$ , write ${ \bf X } = X _ { 1 : m } ,$ and let $Z _ { < i } = ( Z _ { 1 } , \ldots , Z _ { i - 1 } )$ . The noise vectors are independent conditional on X, so $I ( Z _ { i } ; Z _ { < i } \mid \mathbf { X } ) = 0$ For each i, the chain rule gives

$$
\begin{array} { r l } & { I ( Z _ { i } ; W _ { m } \mid \mathbf { X } ) \leq I ( Z _ { i } ; Z _ { < i } , W _ { m } \mid \mathbf { X } ) } \\ & { \phantom { x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x } } \\ & { \phantom { x x x x x x x x x x x x x x x x x x x x x x x x x x } = I ( Z _ { i } ; W _ { m } \mid \mathbf { X } , Z _ { < i } ) . } \end{array}
$$

Summing and applying the chain rule once more yields

$$
\begin{array} { r l r } {  { \sum _ { i = 1 } ^ { m } I ( Z _ { i } ; W _ { m } \mid \mathbf { X } ) \le \sum _ { i = 1 } ^ { m } I ( Z _ { i } ; W _ { m } \mid \mathbf { X } , Z _ { < i } ) } } \\ & { } & { = I ( Z _ { 1 : m } ; W _ { m } \mid \mathbf { X } ) } \\ & { } & { \le H ( W _ { m } \mid \mathbf { X } ) \le \log N , } \end{array}
$$

since $W _ { m }$ takes values in the N-element class Π. Substituting this bound into Claim 5.1 gives

$$
\mathbb { E } _ { x \sim D } \left[ \sum _ { a = 1 } ^ { K } D _ { a } ( x ) \right] \le \frac { 1 } { n } \sum _ { m \ge 1 } \rho _ { m } \log N \le \frac { \log N } { n } .
$$

## 4.2 The minimizing scale

For each context $x ,$ let $g ( x ) = ( \pi ( x ) ) _ { \pi \in \Pi }$ denote its prediction signature. Write $s = \left( s _ { g , a } \right)$ for a table with one row per prediction signature g and one column per action $a \in [ K ]$ . Each entry

$s _ { g , a } \in [ 1 , B ]$ specifies the scale for action a at every context with signature $g .$ Thus the scale defined by s is

$$
\sigma _ { a } ( x ) = s _ { g ( x ) , a } .
$$

For each prediction signature $^ { g , }$ let $\omega _ { g }$ be its probability under the context distribution $D _ { : }$ , and let $\Gamma _ { + }$ be the set of signatures with positive probability:

$$
\omega _ { g } = \operatorname* { P r } _ { X \sim D } ( g ( X ) = g ) , \qquad \Gamma _ { + } = \{ g : \omega _ { g } > 0 \} .
$$

Lemma 6 (Scale minimization on prediction signatures). There is a minimizing scale of the form $\sigma _ { a } ( x ) = s _ { g ( x ) , a } ^ { * } { \mathrm { : } }$ , where $g ( x ) = ( \pi ( x ) ) _ { \pi \in \Pi }$ . The table can be chosen measurably in S.

Proof. Fix a feasible scale σ. Define

$$
s _ { g , a } = \left\{ \begin{array} { l l } { \mathbb { E } [ \sigma _ { a } ( X ) \mid g ( X ) = g ] , } & { g \in \Gamma _ { + } , } \\ { 1 , } & { g \not \in \Gamma _ { + } , } \end{array} \right. \quad \quad \bar { \sigma } _ { a } ( x ) = s _ { g ( x ) , a } .
$$

Condition on $\mathcal { G } = ( M , ( g ( X _ { i } ) , Z _ { i } ) _ { i = 1 } ^ { M } )$ . The signature $g ( X _ { i } )$ determines $\pi ( X _ { i } )$ , and the noise is independent of the contexts. Hence

$$
\mathbb { E } [ P _ { \pi } ( \sigma ) \mid { \mathcal G } ] = \sum _ { i = 1 } ^ { M } s _ { g ( X _ { i } ) , \pi ( X _ { i } ) } Z _ { i , \pi ( X _ { i } ) } = P _ { \pi } ( \bar { \sigma } ) .
$$

Conditional Jensen for the maximum and for $u \mapsto \log ( B / u )$ gives

$$
\begin{array} { r l } { \underset { \pi } { \operatorname* { m a x } } [ S ( \pi ) + P _ { \pi } ( \bar { \sigma } ) ] \leq \mathbb { E } \left[ \underset { \pi } { \operatorname* { m a x } } [ S ( \pi ) + P _ { \pi } ( \sigma ) ] \Big | \mathscr { G } \right] , } & { } \\ { \log ( B / s _ { g , a } ) \leq \mathbb { E } [ \log ( B / \sigma _ { a } ( X ) ) \ | \ g ( X ) = g ] \quad } & { ( g \in \Gamma _ { + } ) . } \end{array}
$$

Take expectations in the first line. In the second, multiply by $\lambda \omega _ { g }$ and sum over $g , a .$ Adding yields $J _ { S } ( \bar { \sigma } ) \leq J _ { S } ( \sigma )$

Conversely, every table $s \in [ 1 , B ] ^ { K | \Gamma _ { + } | }$ defines a feasible scale by setting its entries on zero-mass signatures to one. Thus

$$
\operatorname* { i n f } _ { \sigma } J _ { S } ( \sigma ) = \operatorname* { i n f } _ { s \in [ 1 , B ] ^ { K | \Gamma _ { + } | } } J _ { S } ( s ) , \qquad J _ { S } ( s ) = F ( S , s ) + \lambda \sum _ { g \in \Gamma _ { + } } \omega _ { g } \sum _ { a = 1 } ^ { K } \log ( B / s _ { g , a } ) .
$$

The pool term $F ( S , s )$ is convex in s as an expected maximum of afine functions. It is jointly continuous in $( S , s )$ since

$$
| F ( S , s ) - F ( S ^ { \prime } , s ^ { \prime } ) | \leq \| S - S ^ { \prime } \| _ { \infty } + \| s - s ^ { \prime } \| _ { \infty } \mathbb { E } \sum _ { i = 1 } ^ { M } \sum _ { a = 1 } ^ { K } | Z _ { i , a } | = \| S - S ^ { \prime } \| _ { \infty } + K n \| s - s ^ { \prime } \| _ { \infty } .
$$

The penalty is strictly convex on the table because each coordinate has second derivative $\lambda \omega _ { g } / s _ { g , a } ^ { 2 } >$ 0. Hence $J _ { S }$ attains a unique minimum $s ^ { * } ( S )$ on the compact box.

Finally, if $S _ { j }  S$ and a subsequence $s ^ { * } ( S _ { j _ { k } } ) \to t$ , then for every feasible table $u ,$ joint continuity gives

$$
J _ { S } ( t ) = \operatorname* { l i m } _ { k } J _ { S _ { j _ { k } } } ( s ^ { * } ( S _ { j _ { k } } ) ) \leq \operatorname* { l i m } _ { k } J _ { S _ { j _ { k } } } ( u ) = J _ { S } ( u ) .
$$

Uniqueness gives $t = s ^ { * } ( S )$ . By compactness, $s ^ { * } ( S _ { j } ) \to s ^ { * } ( S )$ , so the minimizing table is continuous, hence measurable, in $S .$

Use the scale from Lemma 6 in the ideal strategy. For any feasible table $s ,$ let $W _ { x } ( s )$ denote the argmax in (3) evaluated at $s ,$ and write

$$
\alpha _ { a } ( x ; s ) = \mathbb { E } [ Z _ { a } { \bf 1 } \{ W _ { x } ( s ) ( x ) = a \} ] .
$$

For $g ( x ) = g ( y )$ , every policy has $\pi ( x ) = \pi ( y )$ and the added score is identical at $x , y$ for the same noise. Thus $\alpha _ { a } ( x ; s ) = \alpha _ { a } ( y ; s )$

Lemma 7 (Derivative of one scale entry). For $g ( x ) = g \in \Gamma _ { + }$ , the derivative in entry $( g , a )$ equals

$$
\frac { \partial J _ { S } ( s ) } { \partial s _ { g , a } } = \omega _ { g } \left( n \alpha _ { a } ( x ; s ) - \lambda / s _ { g , a } \right) .
$$

At the boundary use the derivative of the same formula for $J _ { S }$ extended to positive scales. In particular at $s = s ^ { * }$ , writing $\alpha _ { a } ( x ) = \alpha _ { a } ( x ; s ^ { * } )$ ，

$$
s _ { g , a } ^ { * } < B \implies \alpha _ { a } ( x ) \geq \lambda / ( n s _ { g , a } ^ { * } ) .\tag{6}
$$

Proof. Let $s ^ { \prime }$ be the table obtained by increasing $s _ { g , a }$ by ϵ. In the sum defining $P _ { \pi } ( s )$ , the ith term changes only if $g ( X _ { i } ) = g$ and $\pi ( X _ { i } ) = a$ . In that case it changes from $s _ { g , a } Z _ { i , a }$ to $( s _ { g , a } + \epsilon ) Z _ { i , a }$ Hence

$$
[ S ( \pi ) + P _ { \pi } ( s ^ { \prime } ) ] - [ S ( \pi ) + P _ { \pi } ( s ) ] = \epsilon \sum _ { i = 1 } ^ { M } { \bf 1 } \{ g ( X _ { i } ) = g , \pi ( X _ { i } ) = a \} Z _ { i , a } .
$$

Let $W _ { m }$ be the policy maximizing $S ( \pi ) + P _ { \pi } ( s )$ for a pool of m samples, with the fixed tie-breaking rule. We claim that

$$
\frac { \partial F ( S , s ) } { \partial s _ { g , a } } = \mathbb { E } \left[ \sum _ { i = 1 } ^ { M } \mathbf { 1 } \{ g ( X _ { i } ) = g , \ W _ { M } ( X _ { i } ) = a \} Z _ { i , a } \right] .\tag{7}
$$

To see this, fix the pool. Each policy’s score is afine in $s _ { g , a } ,$ with slope

$$
\sum _ { i = 1 } ^ { M } \mathbf { 1 } \{ g ( X _ { i } ) = g , \pi ( X _ { i } ) = a \} Z _ { i , a } .
$$

If two policies difer at $X _ { i } .$ , condition on the locations and all noise coordinates except $Z _ { i , \pi ( X _ { i } ) }$ Their score diference is a constant plus a nonzero multiple of this continuous variable, so they tie with probability zero. Since Π is finite, almost surely all tied maximizers agree at every sampled location and therefore have the same slope. The derivative of the maximum is thus the slope displayed above with $\pi = W _ { M }$

Every policy’s absolute slope is at most $\textstyle \sum _ { i = 1 } ^ { M } | Z _ { i , a } |$ . The same bound holds for the absolute diference quotient of the maximum, and its expectation is n. Dominated convergence therefore gives (7).

Expanding the right-hand side of (7) by conditioning on $M = m$ , all m summands have the same expectation by symmetry. Writing $\rho _ { m } = \operatorname* { P r } ( M = m )$ , we obtain

$$
\begin{array} { r l } & { \displaystyle \frac { \partial F ( S , s ) } { \partial s _ { g , a } } = \sum _ { m \ge 1 } m \rho _ { m } \mathbb { E } [ \mathbf { 1 } \{ g ( X _ { m } ) = g , \ : W _ { m } ( X _ { m } ) = a \} Z _ { m , a } ] } \\ & { \quad \quad \quad = n \displaystyle \sum _ { k \ge 0 } \rho _ { k } \mathbb { E } [ \mathbf { 1 } \{ g ( X _ { k + 1 } ) = g , \ : W _ { k + 1 } ( X _ { k + 1 } ) = a \} Z _ { k + 1 , a } ] } \\ & { \quad \quad \quad = n \mathbb { E } _ { X \sim D } [ \mathbf { 1 } \{ g ( X ) = g \} \alpha _ { a } ( X ; s ) ] = n \omega _ { g } \alpha _ { a } ( x ; s ) . } \end{array}
$$

The second line uses $m = k + 1$ and $m \rho _ { m } = n \rho _ { m - 1 }$ . In the third line, the first k pairs form the pool and the last pair is the independent inserted pair $( X , Z )$ . Averaging over $k \sim \mathrm { P o i s } ( n )$ and the noise at fixed X gives $\alpha _ { a } ( X ; s ) = \mathbb { E } [ Z _ { a } { \bf 1 } \{ W _ { X } ( s ) ( X ) = a \} \mid X ]$ . The last equality uses $\operatorname* { P r } ( g ( X ) = g ) = \omega _ { g }$ and $\alpha _ { a } ( X ; s ) = \alpha _ { a } ( x ; s )$ whenever $g ( X ) = g ( x ) = g$ . Only the term $\lambda \omega _ { g } \log ( B / s _ { g , a } )$ in the penalty depends on $s _ { g , a } .$ . Adding its derivative to the pool derivative gives

$$
\frac { \partial J _ { S } ( s ) } { \partial s _ { g , a } } = n \omega _ { g } \alpha _ { a } ( x ; s ) - \frac { \lambda \omega _ { g } } { s _ { g , a } } .
$$

If $s _ { g , a } ^ { * } < B$ , increasing this entry by a suficiently small amount is feasible. Since $s ^ { * }$ minimizes $J _ { S }$ the right derivative is nonnegative. Thus, writing $\alpha _ { a } ( x ) = \alpha _ { a } ( x ; s ^ { * } )$ ，

$$
0 \leq \mathit { n } \omega _ { g } \alpha _ { a } ( x ) - \frac { \lambda \omega _ { g } } { s _ { g , a } ^ { * } } \quad \Longrightarrow \quad \alpha _ { a } ( x ) \geq \frac { \lambda } { \mathit { n } s _ { g , a } ^ { * } } ,
$$

where we used $n \omega _ { g } > 0$

## 4.3 Comparing the two action probabilities

For the next two lemmas, write $r _ { \pi } ( z )$ for the winning probability when the inserted vector is fixed at z. Explicitly, at $C = ( m , x _ { 1 : m } )$ set

$$
\begin{array} { l } { { \displaystyle B _ { \pi } ( u , z ) = S ( \pi ) + \sum _ { i = 1 } ^ { m } \sigma _ { \pi ( x _ { i } ) } ( x _ { i } ) u _ { i , \pi ( x _ { i } ) } + \sigma _ { \pi ( x ) } ( x ) z _ { \pi ( x ) } , } } \\ { { \displaystyle r _ { \pi } ( z ) = \int \mathbf { 1 } \{ \pi = \arg \operatorname* { m a x } B _ { \pi ^ { \prime } } ( u , z ) \} \mu ^ { m } ( d u ) . } } \end{array}
$$

Here $\mu ( d z ) = 2 ^ { - K } e ^ { - \| z \| _ { 1 } } d z$ . The law defined in Lemma 5 satisfies $w _ { \pi } \nu _ { \pi } ( d z ) = r _ { \pi } ( z ) \mu ( d z )$ , since both sides equal $\operatorname* { P r } ( Z \in d z , W _ { x } = \pi \mid C )$

Lemma 8 (Midpoint inequality for a winning policy). Fix $x , C , \pi$ . For every pair $z , z ^ { \prime } \in \mathbb { R } ^ { K }$ ，

$$
r _ { \pi } ( ( z + z ^ { \prime } ) / 2 ) \geq \sqrt { r _ { \pi } ( z ) r _ { \pi } ( z ^ { \prime } ) } .
$$

Proof. Fix the context $x ,$ the pool locations $C \ = \ ( m , x _ { 1 : m } )$ , and the policy $\pi .$ . Write $u \ =$ $( u _ { i , b } ) _ { i \in [ m ] , b \in [ K ] }$ for the noise at the pool locations and $z \in \mathbb { R } ^ { K }$ for the noise at x. For each policy $\tau ,$ its score is

$$
B _ { \tau } ( u , z ) = S ( \tau ) + \sum _ { i = 1 } ^ { m } \sigma _ { \tau ( x _ { i } ) } ( x _ { i } ) u _ { i , \tau ( x _ { i } ) } + \sigma _ { \tau ( x ) } ( x ) z _ { \tau ( x ) } .
$$

Let $A _ { \pi }$ be the set of pairs $( u , z )$ for which π wins. For each competitor $\tau _ { \mathrm { { ; } } }$ membership in $A _ { \pi }$ requires

$$
\left\{ B _ { \pi } ( u , z ) > B _ { \tau } ( u , z ) , \mathrm { i f } \tau \mathrm { h a s ~ p r i o r i t y ~ o v e r } \pi \mathrm { i n ~ a ~ t i e } , \right.
$$

Each score diference $B _ { \pi } - B _ { \tau }$ is afine in $( u , z )$ . Thus each comparison defines a half-space, and their intersection $A _ { \pi }$ is convex.

Now hold z fixed. The only randomness left is the pool noise u, whose density is $2 ^ { - m K } e ^ { - \| u \| _ { 1 } }$ Hence

$$
r _ { \pi } ( z ) = \int _ { \mathbb { R } ^ { m K } } f ( u , z ) d u , \qquad f ( u , z ) = { \bf 1 } \{ ( u , z ) \in A _ { \pi } \} \ : 2 ^ { - m K } e ^ { - \| u \| _ { 1 } } .
$$

We verify that f is jointly log-concave. Fix $0 < t < 1$ and set

$$
u _ { t } = ( 1 - t ) u _ { 0 } + t u _ { 1 } , \qquad z _ { t } = ( 1 - t ) z _ { 0 } + t z _ { 1 } .
$$

If both $( u _ { 0 } , z _ { 0 } )$ and $( u _ { 1 } , z _ { 1 } )$ belong to $A _ { \pi } .$ , convexity gives $( u _ { t } , z _ { t } ) \in A _ { \pi }$ . Since $\| u _ { t } \| _ { 1 } \leq ( 1 - t ) \| u _ { 0 } \| _ { 1 } +$ $t \Vert u _ { 1 } \Vert _ { 1 }$ , we obtain

$$
f ( u _ { t } , z _ { t } ) = 2 ^ { - m K } e ^ { - \| u _ { t } \| _ { 1 } } \geq f ( u _ { 0 } , z _ { 0 } ) ^ { 1 - t } f ( u _ { 1 } , z _ { 1 } ) ^ { t } .
$$

If either endpoint lies outside $A _ { \pi }$ , the right-hand side is zero, so the same inequality holds.

Pr´ekopa’s marginalization theorem [Pr´e73] now gives log-concavity of $\begin{array} { r } { r _ { \pi } ( z ) = \int f ( u , z ) } \end{array}$ du in z. Taking $t = 1 / 2 ~ \mathrm { g i v e s }$

$$
r _ { \pi } \biggl ( { \frac { z _ { 0 } + z _ { 1 } } { 2 } } \biggr ) \geq \sqrt { r _ { \pi } ( z _ { 0 } ) r _ { \pi } ( z _ { 1 } ) } .
$$

Lemma 9 (Comparison at a fixed penalty). Fix x, a and put $v = b _ { a } / \sigma _ { a } ( x )$ . Then

$$
\begin{array} { r } { p _ { a } - p _ { a } ^ { - } \le 3 D _ { a } ( x ) + \frac { 3 } { 2 } p _ { a } \operatorname* { m i n } \{ v , 1 \} ^ { 2 } . } \end{array}\tag{8}
$$

Proof. For the ath standard basis vector $\textstyle e _ { a } .$

$$
\sigma _ { \pi ( x ) } ( x ) ( - v e _ { a } ) _ { \pi ( x ) } = - v \sigma _ { a } ( x ) { \bf 1 } \{ \pi ( x ) = a \} = - b _ { a } { \bf 1 } \{ \pi ( x ) = a \} .
$$

Consequently the two probabilities being compared are

$$
\begin{array} { r l } & { p _ { a } = \mathbb { E } _ { C } \displaystyle \sum _ { \pi \left( x \right) = a } \int r _ { \pi } ( z ) \mu ( d z ) = \mathbb { E } _ { C } \displaystyle \sum _ { \pi \left( x \right) = a } w _ { \pi } \mathrm { , } } \\ & { p _ { a } ^ { - } = \mathrm { P r } \Bigg \{ \Bigg ( \arg \operatorname* { m a x } _ { \pi } \left[ S ( \pi ) + P _ { \pi } ( \sigma ) - b _ { a } \mathbf { 1 } \left\{ \pi ( x ) = a \right\} \right] \Bigg ) ( x ) = a \Bigg \} = \mathbb { E } _ { C } \displaystyle \sum _ { \pi \left( x \right) = a } r _ { \pi } ( - v e _ { a } ) . } \end{array}
$$

Write $r = r _ { \pi } , w = w _ { \pi } , \nu = \nu _ { \pi }$ , and $J ( z ) = - 2 v e _ { a } - z$ . The midpoint of $z , J ( z ) { \mathrm { ~ i s ~ } } - v e _ { a } ,$ so Lemma 8 gives $r ( - v e _ { a } ) \geq \sqrt { r ( z ) r ( J ( z ) ) }$ . For any density Q, the reflected density is $( J Q ) ( z ) = Q ( J ( z ) )$ since $J ^ { - 1 } = J$ and | det $D J | = 1$ . Put $\begin{array} { r } { h ( P , Q ) ^ { 2 } = \int ( \sqrt { P } - \sqrt { Q } ) ^ { 2 } = 2 - 2 \int \sqrt { P Q } } \end{array}$ . For $w > 0$ , use $\nu ( z ) = r ( z ) \mu ( z ) / w \colon$

$$
\begin{array} { r l } { \displaystyle \int \sqrt { \mu ( z ) \mu ( J z ) } d z \le \left( \int \mu ( z ) d z \int \mu ( J z ) d z \right) ^ { 1 / 2 } = 1 , } & { } \\ { \displaystyle r ( - v e _ { a } ) \ge r ( - v e _ { a } ) \int \sqrt { \mu ( z ) \mu ( J z ) } d z } & { } \\ { \displaystyle \ge \int \sqrt { r ( z ) r ( J z ) \mu ( z ) \mu ( J z ) } d z } & { } \\ { \displaystyle } & { = w \int \sqrt { \nu ( z ) ( J \nu ) ( z ) } d z = w \left( 1 - \frac { 1 } { 2 } h ( \nu , J \nu ) ^ { 2 } \right) . } \end{array}
$$

When $w = 0 , w - r ( - v e _ { a } ) \leq 0$ and this policy can be dropped from the upper bound. For $w > 0$ the triangle inequality along $\nu , \mu , J \mu$ , Jν and a change of variables give

$$
h ( \nu , J \nu ) \leq h ( \nu , \mu ) + h ( \mu , J \mu ) + h ( J \mu , J \nu ) , \qquad h ( J \mu , J \nu ) = h ( \mu , \nu ) .
$$

Also

$$
h ( \nu , \mu ) ^ { 2 } = 2 \int _ { \{ \nu > 0 \} } \nu \left( 1 - \sqrt { \mu / \nu } \right) \leq 2 \int _ { \{ \nu > 0 \} } \nu \left( - \log \sqrt { \mu / \nu } \right) = \mathrm { K L } ( \nu \| \mu ) .
$$

Let $\varphi ( t ) ~ = ~ { \textstyle { \frac { 1 } { 2 } } } e ^ { - | t | }$ denote the standard Laplace density. Since $\begin{array} { r } { \mu ( z ) = \prod _ { b = 1 } ^ { K } \varphi ( z _ { b } ) , J ( z ) = } \end{array}$ $- 2 v e _ { a } - z$ , and $\varphi$ is even, we have

$$
( J \mu ) ( z ) = \mu ( J ( z ) ) = \varphi ( z _ { a } + 2 v ) \prod _ { b \neq a } \varphi ( z _ { b } ) .
$$

Put $g _ { 0 } ( t ) = \sqrt { \varphi ( t ) }$ . Integrating out the other $K - 1$ coordinates and using Cauchy–Schwarz gives

$$
\begin{array} { l } { \displaystyle h ( \mu , J \mu ) ^ { 2 } = \int [ g _ { 0 } ( t + 2 v ) - g _ { 0 } ( t ) ] ^ { 2 } d t } \\ { \displaystyle \le 2 v \int _ { 0 } ^ { 2 v } \int | g _ { 0 } ^ { \prime } ( t + s ) | ^ { 2 } d t d s = ( 2 v ) ^ { 2 } / 4 = v ^ { 2 } , } \end{array}
$$

since $| g _ { 0 } ^ { \prime } | = g _ { 0 } / 2$ almost everywhere and $\textstyle \int g _ { 0 } ^ { 2 } = 1$ . Using $( A + B + C ) ^ { 2 } \leq 3 ( A ^ { 2 } + B ^ { 2 } + C ^ { 2 } )$ now gives

$$
\begin{array} { r l r } {  { w - r _ { \pi } ( - v e _ { a } ) \le \frac 3 2 w [ h ( \nu , \mu ) ^ { 2 } + h ( \mu , J \mu ) ^ { 2 } + h ( J \mu , J \nu ) ^ { 2 } ] } } \\ & { } & { \le 3 w \mathrm { K L } ( \nu \| \mu ) + \frac 3 2 w v ^ { 2 } , } \\ & { } & { p _ { a } - p _ { a } ^ { - } = \mathbb { E } _ { C } \sum _ { \pi ( x ) = a } ( w _ { \pi } - r _ { \pi } ( - v e _ { a } ) ) \le 3 D _ { a } ( x ) + \frac 3 2 p _ { a } v ^ { 2 } . } \end{array}
$$

For $v > 1 , p _ { a } - p _ { a } ^ { - } \leq p _ { a } \leq 3 D _ { a } ( x ) + \frac { 3 } { 2 } p _ { a }$ . The two ranges therefore give the square of min $\{ v , 1 \}$ in (8).

Lemma 10 (Cost of the shift). At a minimizing scale, for $g ( x ) \in \Gamma _ { + }$ and $d = \operatorname* { m i n } \{ b _ { a } / \sigma _ { a } ( x ) , 1 \}$ },

$$
\begin{array} { r } { p _ { a } d ^ { 2 } \leq \left\{ \begin{array} { l l } { 2 / B , } & { \sigma _ { a } ( x ) = B , } \\ { \frac { 3 } { 2 } D _ { a } ( x ) , } & { \sigma _ { a } ( x ) < B . } \end{array} \right. } \end{array}
$$

Proof. Since $\gamma \leq 1 / 2 , ( 4 )$ gives $p _ { a } \leq 2 q _ { a }$ . Write $v = b _ { a } / \sigma _ { a } ( x )$ , so $q _ { a } v = c _ { a } / \sigma _ { a } ( x )$ . When $\sigma _ { a } ( x ) = B$ we have $p _ { a } d ^ { 2 } \leq 2 q _ { a } v \leq 2 / B$ . Suppose $\sigma _ { a } ( x ) < B$ . By Lemma $7$ and the parameter choice $\lambda = 4 n \ell$ with $\ell \geq 1$ , we have

$$
\alpha _ { a } ( x ) \geq { \frac { \lambda } { n \sigma _ { a } ( x ) } } \geq { \frac { 4 } { \sigma _ { a } ( x ) } } .
$$

Since $q _ { a } v = c _ { a } / \sigma _ { a } ( x )$ and $c _ { a } \leq 1$ , it follows that

$$
\alpha _ { a } ( x ) \geq { \frac { 4 } { \sigma _ { a } ( x ) } } \geq 4 q _ { a } v \geq 2 p _ { a } d ,
$$

where the last inequality uses $p _ { a } \leq 2 q _ { a }$ and $d = \operatorname* { m i n } \{ v , 1 \} \leq v$

It remains to turn the lower bound on $\alpha _ { a } ( x )$ into a lower bound on $D _ { a } ( x )$ . Set $\theta = d / 2$ . We will show that

$$
D _ { a } ( x ) \geq \theta \alpha _ { a } ( x ) + p _ { a } \log ( 1 - \theta ^ { 2 } ) .
$$

Fix the pool locations $C$ and a policy π with $w _ { \pi } > 0$ . Recall that $\mu$ is the law of the inserted noise vector $Z ,$ and $\nu _ { \pi }$ is its conditional law given C and $W _ { x } = \pi$ . To relate $\mathrm { K L } ( \nu _ { \pi } \| \mu )$ to $\mathbb { E } _ { \nu _ { \pi } } [ Z _ { a } ]$ define an auxiliary law $\mu _ { \theta }$ for the same vector by giving each z weight $e ^ { \theta z _ { a } }$ relative to $\mu .$ Dividing by the total weight gives

$$
\mu _ { \theta } ( d z ) = \frac { e ^ { \theta z _ { a } } } { M _ { \theta } } \mu ( d z ) , \qquad M _ { \theta } = \int e ^ { \theta z _ { a } } \mu ( d z ) .
$$

Since $Z _ { a }$ has density $e ^ { - | t | } / 2$ and $0 \leq \theta \leq 1 / 2$ ，

$$
M _ { \theta } = \frac { 1 } { 2 } \int _ { 0 } ^ { \infty } e ^ { - ( 1 - \theta ) t } d t + \frac { 1 } { 2 } \int _ { 0 } ^ { \infty } e ^ { - ( 1 + \theta ) t } d t = \frac { 1 } { 1 - \theta ^ { 2 } } .
$$

By construction, $\log ( d \mu _ { \theta } / d \mu ) ( z ) = \theta z _ { a } - \log M _ { \theta }$ . Hence

$$
\begin{array} { r l } & { \mathrm { K L } ( \nu _ { \pi } \| \mu ) = \mathrm { K L } ( \nu _ { \pi } \| \mu _ { \theta } ) + \mathbb { E } _ { \nu _ { \pi } } \left[ \log \frac { d \mu _ { \theta } } { d \mu } ( Z ) \right] } \\ & { \quad \quad \quad \geq \theta \mathbb { E } _ { \nu _ { \pi } } [ Z _ { a } ] - \log M _ { \theta } . } \end{array}
$$

Multiply by $w _ { \pi }$ , sum over policies with $\pi ( x ) = a$ and $w _ { \pi } > 0$ , and average over C. By the definition of $\nu _ { \pi }$

$$
\operatorname { \mathbb { E } } _ { C } \sum _ { \pi ( x ) = a \atop w _ { \pi } > 0 } w _ { \pi } \mathbb { E } _ { \nu _ { \pi } } [ Z _ { a } ] = \mathbb { E } [ Z _ { a } \mathbf { 1 } \{ W _ { x } ( x ) = a \} ] = \alpha _ { a } ( x ) ,
$$

$$
\mathbb { E } _ { C } \sum _ { \pi ( x ) = a } w _ { \pi } = \operatorname* { P r } ( W _ { x } ( x ) = a ) = p _ { a } .
$$

The weighted sum of the KL terms is $D _ { a } ( x )$ , so

$$
D _ { a } ( x ) \geq \theta \alpha _ { a } ( x ) - p _ { a } \log M _ { \theta } = \theta \alpha _ { a } ( x ) + p _ { a } \log ( 1 - \theta ^ { 2 } ) .
$$

Finally, $\theta = d / 2$ gives $\theta ^ { 2 } = d ^ { 2 } / 4 \leq 1 / 4$ , so

$$
- \log ( 1 - \theta ^ { 2 } ) = \int _ { 0 } ^ { \theta ^ { 2 } } \frac { d t } { 1 - t } \leq \frac { \theta ^ { 2 } } { 1 - \theta ^ { 2 } } \leq \frac { d ^ { 2 } } { 3 } .
$$

Using $\alpha _ { a } ( x ) \geq 2 p _ { a } d$ from above, we conclude that

$$
D _ { a } ( x ) \geq { \frac { d } { 2 } } \alpha _ { a } ( x ) - { \frac { 1 } { 3 } } p _ { a } d ^ { 2 } \geq { \frac { d } { 2 } } ( 2 p _ { a } d ) - { \frac { 1 } { 3 } } p _ { a } d ^ { 2 } = { \frac { 2 } { 3 } } p _ { a } d ^ { 2 } .
$$

Proof of Lemma 3. Lemma 5 gives the expectation bound. For a positive-mass signature, put $d = \operatorname* { m i n } \{ b _ { a } / \sigma _ { a } ( x ) , 1 \}$ and apply Lemmas 9 and 10 to obtain

$$
\begin{array} { r } { p _ { a } - p _ { a } ^ { - } \le 3 D _ { a } ( x ) + \frac { 3 } { 2 } p _ { a } d ^ { 2 } \le \left\{ \begin{array} { l l } { ( 3 + \frac { 3 } { 2 } \cdot \frac { 3 } { 2 } ) D _ { a } ( x ) = \frac { 2 1 } { 4 } D _ { a } ( x ) , } & { \sigma _ { a } ( x ) < B , } \\ { 3 D _ { a } ( x ) + \frac { 3 } { 2 } \cdot 2 / B = 3 D _ { a } ( x ) + 3 / B , } & { \sigma _ { a } ( x ) = B . } \end{array} \right. } \end{array}
$$

Both bounds are at most $6 D _ { a } ( x ) + 3 / B$ . This proves Lemma 3, since the positive-mass signatures have total probability one. □

## 5 Oracle Implementation

Fix the past $H = ( ( x _ { i } , { \hat { c } } _ { i } ) ) _ { i < t }$ and its score S. The algorithm uses projected stochastic gradient descent to approximate the minimizer of $J _ { S }$ . Its random iterates are functions of a context. A stored list of draws lets us evaluate those functions only where they are needed. After $x _ { t }$ arrives, independent playouts estimate the winning probabilities at that context.

Theorem 11 (Oracle round). On each round Algorithm 2 uses polynomial resources in $T , K , L$ and satisfies

$$
\begin{array} { r } { \mathbb { E } [ \Phi _ { S , x _ { t } } ( q _ { t } ) \mid \mathcal { H } _ { t - 1 } ] \leq \mathbb { E } _ { x \sim D } \Phi _ { S , x } ( q ^ { * } ( x ) ) + 1 / T . } \end{array}
$$

Together with Theorem 2 and Lemma 4, this proves Theorem 1. We give the algorithm first and then prove its accuracy and cost.

## 5.1 Sampling and evaluation

Use $n , \lambda , B , \gamma$ from (1). The following choices fix the precision, iteration count, number of probability samples, and pool cap:

$$
\begin{array} { r l r l r l } & { \delta = \displaystyle \frac { 1 } { T ( 1 + K / \gamma ) } , } & & { \varepsilon = \frac { \lambda \delta ^ { 2 } } { 3 2 B ^ { 2 } ( n + 1 ) } , } & & { m = \lceil 1 6 K / \delta ^ { 2 } \rceil , } \\ & { D _ { 0 } = B \sqrt { K } , } & & { G ^ { 2 } = K ( 4 n ^ { 2 } + 2 \lambda ^ { 2 } ) , } & & { J = \operatorname* { m a x } \{ 1 , \lceil D _ { 0 } ^ { 2 } G ^ { 2 } / \varepsilon ^ { 2 } \rceil \} , } \\ & { \eta = D _ { 0 } / ( G \sqrt { J } ) , } & & { M _ { \mathrm { m a x } } = \lceil ( e - 1 ) n + \log ( 4 ( J + m ) / \delta ) \rceil . } \end{array}
$$

Here $D _ { 0 }$ bounds the diameter of the feasible scales, and $G ^ { 2 }$ bounds the expected squared norm of one stochastic gradient (Lemma 14). A pool U consists of pairs $( Y , Z )$ with $Y \sim D , Z \sim \mu .$ Products $d \odot Z$ are coordinatewise. All fresh draws in the algorithms are independent. Write 1 for the all-ones vector of the required dimension.

Algorithm 2 One bandit round given history H   
1. Draw $J + m$ capped Poisson counts. If any exceeds $M _ { \mathrm { m a x } }$ , receive $x _ { t } ,$ , set $q = \mathbf { 1 } / K$ , and continue at line   
5. Otherwise use these counts in order.   
2. Start with an empty list of training records. For $j = 0 , \ldots , J - 1 \colon$   
(a) Draw a pool $U _ { j }$ of the specified size. For every $( Y , Z ) \in U _ { j } .$ , call Algorithm 4 at $Y$ on the first j   
records, obtaining $( d , u )$ . Store $( Y , d \odot Z )$ in the list $P _ { j } .$   
(b) Draw one $Z ^ { j } \sim \mu$ and append $( P _ { j } , Z ^ { j } )$ to the records.   
3. Receive $x = x _ { t } .$ Call Algorithm 4 at x on all J records and set $\bar { s } ( x ) = u / J$ from its second output.   
4. For $k = 1 , \ldots , m .$ , draw a fresh pool. At each of its locations Y, evaluate $\bar { s } ( Y ) = u / J$ in the same way.   
Form the list $P _ { k } ^ { \prime } = ( ( Y , \bar { s } ( Y ) \odot \bar { Z } ) : ( Y , Z )$ in this pool). Draw $Z _ { k } ^ { \prime } \sim \mu$ and set   
$A _ { k } ^ { \prime } = \mathrm { W I N N E R } ( H , P _ { k } ^ { \prime } , x , \bar { s } ( x ) \odot Z _ { k } ^ { \prime } ) .$   
After all m draws, compute for each $a \in [ K ]$   
$\hat { p } _ { a } = \frac { 1 } { m } \sum _ { k = 1 } ^ { m } { { \bf 1 } \{ A _ { k } ^ { \prime } = a \} } , \qquad q _ { a } = ( 1 - \gamma ) \hat { p } _ { a } + \gamma / K .$   
5. Draw $A _ { t } \sim q ,$ , observe $c _ { t } ( x _ { t } , A _ { t } )$ , and append $( x _ { t } , e _ { A _ { t } } c _ { t } ( x _ { t } , A _ { t } ) / q _ { A _ { t } } )$ to H.

Algorithm 3 Winner $( H , P , y , v )$   
1. For each $\overline { { a \in [ K ] } }$ obtain $V _ { a }$ by querying Val on the concatenation H, $\overline { { ( ( Y , - u ) : ( Y , u ) \in P ) , ( y , - v - e _ { a } ) } }$   
2. Return the smallest a attaining min<sub>b</sub> $V _ { b } .$

Algorithm 4 Evaluate a scale at context y using records $( P _ { i } , Z ^ { i } ) _ { 0 \leq i < j }$ and history H   
1. Initialize $d = \mathbf { 1 }$ and $u = 0$ in $\mathbb { R } ^ { K }$   
2. For $i = 0 , \ldots , j - 1 ;$   
(a) Set $u  u + d$ and $A \gets \mathrm { W I N N E R } ( H , P _ { i } , y , d \odot Z ^ { i } )$   
(b) For every $a \in [ K ]$ set   
$d _ { a } \gets \mathrm { c l i p } _ { [ 1 , B ] } \left( d _ { a } - \eta [ n Z _ { a } ^ { i } \mathbf { 1 } \{ A = a \} - \lambda / d _ { a } ] \right) .$   
3. Return $( d , u )$ , the current iterate $s ^ { j } ( y )$ and the sum $\textstyle \sum _ { i < j } s ^ { i } ( y )$

Lemma 12 (Replay computes the explicit SGD iterates). Fix the history H and the training draws $( U _ { j } , Z ^ { j } ) _ { 0 \leq j < J }$ . Consider explicit projected SGD starting from $s ^ { 0 } ( y ) = \mathbf { 1 }$ and applying the update in Algorithm 4 at every context y in every step. For each $0 \le j \le J$ and every y, Algorithm 4 on the first j records of Algorithm 2 returns

$$
( d , u ) = \left( s ^ { j } ( y ) , \sum _ { i = 0 } ^ { j - 1 } s ^ { i } ( y ) \right) .
$$

Proof. We induct on $j .$ With no records, replay returns $( \mathbf { 1 } , 0 ) = ( s ^ { 0 } ( y ) , 0 )$ for every y.

Suppose the claim holds after $j$ records. To build record j, Algorithm 2 calls replay at each location Y in the new pool $U _ { j }$ . By induction, the returned scale is $s ^ { j } ( Y )$ . Thus the stored pool is exactly the one explicit SGD constructs:

$$
P _ { j } = \left[ ( Y , s ^ { j } ( Y ) \odot Z ) : ( Y , Z ) \in U _ { j } \right] .
$$

Now fix any query context y. After the first $j$ records, replay has $d = s ^ { j } ( y )$ . On record $j ,$ replay and explicit SGD therefore compute the same action

$$
A _ { j } ( y ) = \mathrm { W I N N E R } ( H , P _ { j } , y , s ^ { j } ( y ) \odot Z ^ { j } ) .
$$

For each $a \in [ K ]$ , both update the scale to

$$
s _ { a } ^ { j + 1 } ( y ) = \mathrm { c l i p } _ { [ 1 , B ] } \left( s _ { a } ^ { j } ( y ) - \eta \left[ n Z _ { a } ^ { j } { \bf 1 } \{ A _ { j } ( y ) = a \} - \frac { \lambda } { s _ { a } ^ { j } ( y ) } \right] \right) .
$$

Replay also adds the pre-update value $s ^ { j } ( y )$ to u, giving $\textstyle u = \sum _ { i = 0 } ^ { j } s ^ { i } ( y )$ . This proves the claim for $j + 1$ □

Lemma 13 (Oracle access and cost). Algorithm 3 returns the winning action in each queried playout almost surely. Per round, Algorithm 2 makes at most

$$
O \big ( K [ M _ { \mathrm { m a x } } J ^ { 2 } + m ( M _ { \mathrm { m a x } } + 1 ) ( J + 1 ) ] \big )
$$

value calls, each on at most $T + M _ { \mathrm { m a x } } + 1$ contexts.

Proof. Write $\begin{array} { r } { H _ { \pi } = S ( \pi ) + \sum _ { ( Y , u ) \in P } u _ { \pi ( Y ) } + v _ { \pi ( y ) } } \end{array}$ . Then

$$
- V _ { a } = \operatorname* { m a x } _ { \pi } [ { \cal H } _ { \pi } + { \bf 1 } \{ \pi ( y ) = a \} ] , \quad - V _ { a } = 1 + \operatorname* { m a x } _ { \pi } { \cal H } _ { \pi } \iff \exists \pi \in \arg \operatorname* { m a x } _ { \bf H } : \pi ( y ) = a .
$$

Conditional on the other draws, the highest scores of distinct nonempty action groups are constants plus independent continuous shifts $s _ { b } ( y ) Z _ { b }$ . They tie with probability zero. Within a training step this holds jointly for every prediction signature by a finite union: inductively the input scales depend on y only through its signature. Thus reusing that step at later query locations preserves its winning action almost surely.

Building record j takes at most $j M _ { \mathrm { m a x } }$ winner calls. Evaluating $\bar { s } ( x )$ costs J calls, and each probability trial needs at most $J M _ { \mathrm { m a x } } + 1$ . Multiplying by K proves the stated bound. The storage of scaled training records is $O ( K J ( M _ { \operatorname* { m a x } } + 1 ) )$ in addition to the history and working lists. At most $( J + m ) M _ { \mathrm { m a x } }$ auxiliary contexts are drawn. Each list construction and coordinate update costs polynomially many operations in the dimensions shown. A capped count can be sampled with at most $M _ { \mathrm { m a x } } + 1$ independent exponential waiting times of mean one: stop when their sum exceeds n or $M _ { \mathrm { m a x } } + 1$ arrivals occur by n. For $k \leq M _ { \operatorname* { m a x } }$ the probability of k arrivals is the integral of $e ^ { - n }$ over $0 < t _ { 1 } < \cdot \cdot \cdot < t _ { k } < n$ , namely $e ^ { - n } n ^ { k } / k !$ . All displayed parameters and the reciprocals of $\gamma , \lambda , \delta , \varepsilon$ are polynomially bounded in $T , K , L$ since $L \geq \log 2$ □

## 5.2 Accuracy of the played distribution

For the next two lemmas allow full Poisson counts. Write $\begin{array} { r c l } { \bar { s } } & { = } & { J ^ { - 1 } \sum _ { j < J } s ^ { j } } \end{array}$ and $\| h \| _ { D } ^ { 2 } =$ $\begin{array} { r } { \mathbb E _ { Y \sim D } \sum _ { a } h _ { a } ( Y ) ^ { 2 } } \end{array}$ . At each y the stochastic gradient used in Algorithm 4 is

$$
\begin{array} { r } { v _ { a } ^ { j } ( y ) = n Z _ { a } ^ { j } \mathbf { 1 } \{ \mathrm { W I N N E R } ( H , P _ { j } , y , s ^ { j } ( y ) \odot Z ^ { j } ) = a \} - \lambda / s _ { a } ^ { j } ( y ) . } \end{array}\tag{9}
$$

Lemma 14 (Optimization error). With the parameters of Algorithm 2, $i f s ^ { * }$ is the scale minimizer at S, then

$$
\begin{array} { r } { \mathbb { E } [ J _ { S } ( \bar { s } ) - R ( S ) ] \le \varepsilon , \qquad \mathbb { E } \| \bar { s } - s ^ { * } \| _ { D } ^ { 2 } \le 2 B ^ { 2 } \varepsilon / \lambda . } \end{array}\tag{10}
$$

Proof. Fix the incoming score S and use uncapped Poisson pools. By Lemma 12, replay evaluates the explicit projected SGD iterates for the same training draws. Thus the scale used by the algorithm is $\begin{array} { r } { { \bar { s } } = { J ^ { - 1 } } \sum _ { i = 0 } ^ { J - 1 } { s ^ { j } } } \end{array}$

We first identify the conditional mean of one SGD update. Let $\mathcal { H } _ { j }$ be the history of the first $j$ training steps, and write $\mathbb { E } _ { j } [ \cdot ] = \mathbb { E } [ \cdot \mid \mathcal { H } _ { j } ]$ . Conditional on $\mathcal { H } _ { j }$ , the current scale $s ^ { j }$ is fixed. At a fixed context $y ,$ the action used in (9) is

$$
A _ { j } ( y ) = \mathrm { W I N N E R } ( H , P _ { j } , y , s ^ { j } ( y ) \odot Z ^ { j } ) .
$$

The pool $P _ { j }$ is a fresh ordinary pool scaled by $s ^ { j }$ , and $Z ^ { j }$ is the independent noise vector inserted at $y .$ Thus $A _ { j } ( y )$ is the action of the inserted winner in the experiment defining $\alpha _ { a } ( y ; s ^ { j } )$ . In particular,

$$
\mathbb { E } _ { j } \left[ Z _ { a } ^ { j } \mathbf { 1 } \{ A _ { j } ( y ) = a \} \right] = \alpha _ { a } ( y ; s ^ { j } ) .
$$

For each $g \in \Gamma _ { + }$ , choose a context $y _ { g }$ with $g ( y _ { g } ) = g$ . Substituting the preceding identity into (9) and applying Lemma $7$ gives

$$
\omega _ { g } \mathbb { E } _ { j } [ v _ { a } ^ { j } ( y _ { g } ) ] = \omega _ { g } \left( n \alpha _ { a } ( y _ { g } ; s ^ { j } ) - \frac { \lambda } { s _ { g , a } ^ { j } } \right) = \frac { \partial J _ { S } ( s ^ { j } ) } { \partial s _ { g , a } } .
$$

Consequently, with $\begin{array} { r } { \langle h , k \rangle _ { D } = \mathbb { E } _ { Y \sim D } \sum _ { a } h _ { a } ( Y ) k _ { a } ( Y ) } \end{array}$ , convexity of $J _ { S }$ gives

$$
\begin{array} { r l } & { J _ { S } ( s ^ { j } ) - R ( S ) = J _ { S } ( s ^ { j } ) - J _ { S } ( s ^ { * } ) } \\ & { \qquad \le \displaystyle \sum _ { g \in \Gamma _ { + } } \displaystyle \sum _ { a = 1 } ^ { K } \frac { \partial J _ { S } ( s ^ { j } ) } { \partial s _ { g , a } } ( s _ { g , a } ^ { j } - s _ { g , a } ^ { * } ) } \\ & { \qquad = \displaystyle \sum _ { g \in \Gamma _ { + } } \omega _ { g } \displaystyle \sum _ { a = 1 } ^ { K } \mathbb { E } _ { j } [ v _ { a } ^ { j } ( y _ { g } ) ] ( s _ { g , a } ^ { j } - s _ { g , a } ^ { * } ) = \langle \mathbb { E } _ { j } [ v ^ { j } ] , s ^ { j } - s ^ { * } \rangle _ { D } . } \end{array}
$$

To bound the squared size of the update, use $s _ { a } ^ { j } ( y ) \geq 1$ and $\mathbb { E } _ { j } [ ( Z _ { a } ^ { j } ) ^ { 2 } ] = 2$ . For every $y , a ,$

$$
\begin{array} { l l l } { { \mathbb { E } _ { j } [ ( v _ { a } ^ { j } ( y ) ) ^ { 2 } ] \leq 2 n ^ { 2 } \mathbb { E } _ { j } \left[ ( Z _ { a } ^ { j } ) ^ { 2 } { \bf 1 } \{ A _ { j } ( y ) = a \} \right] + \frac { 2 \lambda ^ { 2 } } { ( s _ { a } ^ { j } ( y ) ) ^ { 2 } } } } \\ { { \quad \quad \quad \quad \leq 4 n ^ { 2 } + 2 \lambda ^ { 2 } . } } \end{array}
$$

Hence

$$
\mathbb { E } _ { j } [ \| v ^ { j } \| _ { D } ^ { 2 } ] = \mathbb { E } _ { Y \sim D } \sum _ { a = 1 } ^ { K } \mathbb { E } _ { j } [ ( v _ { a } ^ { j } ( Y ) ) ^ { 2 } ] \le K ( 4 n ^ { 2 } + 2 \lambda ^ { 2 } ) = G ^ { 2 } ,
$$

where $Y$ is an independent draw from $D .$

Since every coordinate of $s ^ { * }$ lies in $[ 1 , B ]$ , clipping the update can only decrease its distance to $s ^ { * }$ . Expanding the square gives

$$
\begin{array} { r l } & { \| s ^ { j + 1 } - s ^ { * } \| _ { D } ^ { 2 } \leq \| s ^ { j } - \eta v ^ { j } - s ^ { * } \| _ { D } ^ { 2 } } \\ & { \qquad = \| s ^ { j } - s ^ { * } \| _ { D } ^ { 2 } - 2 \eta \langle v ^ { j } , s ^ { j } - s ^ { * } \rangle _ { D } + \eta ^ { 2 } \| v ^ { j } \| _ { D } ^ { 2 } . } \end{array}
$$

Taking $\mathbb { E } _ { j }$ and using the two bounds above yields

$$
\begin{array} { r } { 2 \eta \big ( J _ { S } ( s ^ { j } ) - R ( S ) \big ) \leq \| s ^ { j } - s ^ { * } \| _ { D } ^ { 2 } - \mathbb { E } _ { j } [ \| s ^ { j + 1 } - s ^ { * } \| _ { D } ^ { 2 } ] + \eta ^ { 2 } G ^ { 2 } . } \end{array}
$$

Taking expectations and summing from $j = 0$ to $J - 1$ now gives

$$
\begin{array} { r l } {  { 2 \eta \sum _ { j = 0 } ^ { J - 1 } \mathbb { E } [ J _ { S } ( s ^ { j } ) - R ( S ) ] \le \sum _ { j = 0 } ^ { J - 1 } \big ( \mathbb { E } \| s ^ { j } - s ^ { * } \| _ { D } ^ { 2 } - \mathbb { E } \| s ^ { j + 1 } - s ^ { * } \| _ { D } ^ { 2 } \big ) + J \eta ^ { 2 } G ^ { 2 } } } \\ & { = \| s ^ { 0 } - s ^ { * } \| _ { D } ^ { 2 } - \mathbb { E } \| s ^ { J } - s ^ { * } \| _ { D } ^ { 2 } + J \eta ^ { 2 } G ^ { 2 } } \\ & { \le D _ { 0 } ^ { 2 } + J \eta ^ { 2 } G ^ { 2 } . } \end{array}
$$

By convexity at ¯s and the choices $\eta = D _ { 0 } / ( G \sqrt { J } )$ and $J \ge D _ { 0 } ^ { 2 } G ^ { 2 } / \varepsilon ^ { 2 }$

$$
\begin{array} { r l } & { \displaystyle \mathbb { E } \big [ J _ { S } ( \bar { s } ) - R ( S ) \big ] \leq \frac { 1 } { J } \sum _ { j = 0 } ^ { J - 1 } \mathbb { E } \big [ J _ { S } ( s ^ { j } ) - R ( S ) \big ] } \\ & { \quad \quad \leq \frac { D _ { 0 } ^ { 2 } + J \eta ^ { 2 } G ^ { 2 } } { 2 \eta J } = \frac { D _ { 0 } G } { 2 \sqrt { J } } + \frac { D _ { 0 } G } { 2 \sqrt { J } } = \frac { D _ { 0 } G } { \sqrt { J } } \leq \varepsilon . } \end{array}
$$

It remains to bound the distance from ¯s to $s ^ { * }$ . For $f ( z ) = \lambda \log ( B / z )$ and $1 \le z \le B$ , we have $f ^ { \prime \prime } ( z ) = \lambda / z ^ { 2 } \ge \lambda / B ^ { 2 }$ . Thus, for $u , v \in [ 1 , B ]$ and $0 < t < 1$ 9

$$
f ( ( 1 - t ) u + t v ) \leq ( 1 - t ) f ( u ) + t f ( v ) - \frac { \lambda } { 2 B ^ { 2 } } t ( 1 - t ) ( u - v ) ^ { 2 } .
$$

Apply this inequality to $u = s _ { a } ^ { * } ( Y )$ and $v = s _ { a } ( Y )$ for any feasible scale s. Summing over a and averaging over Y gives the squared distance $\begin{array} { r } { \mathbb { E } _ { Y } \sum _ { a } ( s _ { a } ( Y ) - s _ { a } ^ { * } ( Y ) ) ^ { 2 } = \| s - s ^ { * } \| _ { D } ^ { 2 } } \end{array}$ . Adding the convex pool term and using the optimality of $s ^ { * }$ gives

$$
R ( S ) \leq J _ { S } ( ( 1 - t ) s ^ { * } + t s ) \leq ( 1 - t ) R ( S ) + t J _ { S } ( s ) - { \frac { \lambda } { 2 B ^ { 2 } } } t ( 1 - t ) \| s - s ^ { * } \| _ { D } ^ { 2 } .
$$

In particular,

$$
\begin{array} { c } { 0 \displaystyle \le t \big ( J _ { S } ( s ) - R ( S ) \big ) - \displaystyle \frac { \lambda } { 2 B ^ { 2 } } t ( 1 - t ) \| s - s ^ { * } \| _ { D } ^ { 2 } , } \\ { J _ { S } ( s ) - R ( S ) \ge \displaystyle \frac { \lambda } { 2 B ^ { 2 } } ( 1 - t ) \| s - s ^ { * } \| _ { D } ^ { 2 } . } \end{array}
$$

Letting $t \downarrow 0$ and applying the result to $s = \bar { s }$ yields

$$
\mathbb { E } \| \bar { s } - s ^ { * } \| _ { D } ^ { 2 } \leq \frac { 2 B ^ { 2 } } { \lambda } \mathbb { E } [ J _ { S } ( \bar { s } ) - R ( S ) ] \leq \frac { 2 B ^ { 2 } \varepsilon } { \lambda } .
$$

Fix the score vector $S$ at the start of the round. For a scale s and a context $x ,$ let $p _ { a } ^ { s } ( x )$ be the probability that the winning policy predicts action a at x:

$$
p _ { a } ^ { s } ( x ) = \operatorname* { P r } ( W _ { x } ( s ) ( x ) = a ) .
$$

Here $S , s$ , and $x$ are fixed. The probability is over a fresh Poisson pool and an independent noise vector inserted at x. The scale s determines the noise multipliers at every pool location and at x.

Let $s ^ { * }$ minimize $J _ { S }$ . The ideal strategy plays action a at x with probability

$$
q _ { a } ^ { * } ( x ) = ( 1 - \gamma ) p _ { a } ^ { s ^ { * } } ( x ) + \frac { \gamma } { K } .
$$

The algorithm instead uses the scale ¯s computed by SGD. After receiving $x ,$ it holds ¯s and $x$ fixed and runs m independent trials, each with a fresh pool and a fresh noise vector at x. If $A _ { k } ^ { \prime }$ is the winning policy’s action at x in trial $k ,$ then the fraction of trials won by action a and its resulting play probability are

$$
\widehat { p } _ { a } ( x ) = \frac { 1 } { m } \sum _ { k = 1 } ^ { m } \mathbf { 1 } \{ A _ { k } ^ { \prime } = a \} , \qquad \widehat { q } _ { a } ( x ) = ( 1 - \gamma ) \widehat { p } _ { a } ( x ) + \frac { \gamma } { K } .
$$

Lemma 15 (Accuracy of the action probabilities before capping). Suppose the algorithm runs with the pool-size cap disabled. For a fresh context $x \sim D$ , its play probabilities satisfy

$$
\mathbb { E } \left[ \sum _ { a = 1 } ^ { K } \left. \widehat { q } _ { a } ( x ) - q _ { a } ^ { * } ( x ) \right. \right] \leq \underbrace { 2 B \sqrt { \frac { ( n + 1 ) \varepsilon } { 2 \lambda } } } _ { e r r o r f r o m \ u s i n g \bar { s } } + \underbrace { \sqrt { \frac { K } { m } } } _ { e r r o r f r o m \ e s t i m a t i n g \ w i n b i l i t i e s } .\tag{11}
$$

The expectation is over the draws used by SGD to compute ${ \bar { s } } ,$ , the independent context x, and the m trials at that context.

Proof. Fix the incoming score S and use uncapped Poisson pools.

Claim 15.1. For any two fixed feasible scales $s , t ,$

$$
\begin{array} { r } { \mathbb { E } _ { x \sim D } \| p ^ { s } ( x ) - p ^ { t } ( x ) \| _ { 1 } \leq \sqrt { n + 1 } \| s - t \| _ { D } . } \end{array}
$$

Proof. Write $X _ { 0 } = x$ for the inserted context and $X _ { 1 } , \ldots , X _ { M }$ for the ordinary pool contexts. Let $Q _ { s }$ be the joint law of M, these locations, and the scaled noise coeficients $V _ { i , a } = s _ { a } ( X _ { i } ) Z _ { i , a }$ for $0 \leq i \leq M$ and $a \in [ K ]$ . Define $Q _ { t }$ analogously.

We first compare one coeficient. If $V \sim \mathrm { L a p } ( u )$ , then $\mathbb { E } | V | = u ,$ , so

$$
\mathrm { K L } ( \mathrm { L a p } ( u ) | | \mathrm { L a p } ( v ) ) = \mathbb { E } \left[ \log { \frac { v } { u } } + | V | \left( { \frac { 1 } { v } } - { \frac { 1 } { u } } \right) \right] = \log ( v / u ) + u / v - 1 \leq { \frac { ( u - v ) ^ { 2 } } { 2 } } .
$$

For the inequality, the expression as a function of u has value and derivative zero at $u = v ,$ , and second derivative $1 / u ^ { 2 } \leq 1$ on the feasible interval $[ 1 , B ]$

Under $Q _ { s }$ and $Q _ { t }$ , the count and locations have the same law. Conditional on them, the coeficients are independent. The chain rule for relative entropy therefore gives

$$
\begin{array} { r l } & { \mathrm { K L } ( Q _ { s } \| Q _ { t } ) = \mathbb { E } \displaystyle \sum _ { i = 0 } ^ { M } \displaystyle \sum _ { a = 1 } ^ { K } \mathrm { K L } \big ( \mathrm { L a p } ( s _ { a } ( X _ { i } ) ) \| \mathrm { L a p } ( t _ { a } ( X _ { i } ) ) \big ) } \\ & { \qquad \leq \frac { 1 } { 2 } \mathbb { E } \displaystyle \sum _ { i = 0 } ^ { M } \displaystyle \sum _ { a = 1 } ^ { K } ( s _ { a } ( X _ { i } ) - t _ { a } ( X _ { i } ) ) ^ { 2 } } \\ & { \qquad = \displaystyle \frac { 1 } { 2 } \displaystyle \sum _ { k > 0 } \mathrm { P r } ( M = k ) ( k + 1 ) \| s - t \| _ { D } ^ { 2 } = \frac { n + 1 } { 2 } \| s - t \| _ { D } ^ { 2 } . } \end{array}
$$

The factor $k + 1$ counts the inserted context and the k ordinary contexts, each drawn from D.

Given the locations and coeficients, the same rule under either scale returns $( x , A )$ : it maximizes $\begin{array} { r } { S ( \pi ) + \sum _ { i = 0 } ^ { M } V _ { i , \pi ( X _ { i } ) } } \end{array}$ with the fixed tie-breaking order and returns the winner’s action at $x = X _ { 0 }$ The context marginal is D under both scales. Thus, writing $Q _ { s } ^ { x , A }$ and $Q _ { t } ^ { x , A }$ for the two resulting laws, the chain rule and data processing give

$$
\begin{array} { r } { \mathbb { E } _ { x } \mathrm { K L } ( p ^ { s } ( x ) \| p ^ { t } ( x ) ) = \mathrm { K L } ( Q _ { s } ^ { x , A } \| Q _ { t } ^ { x , A } ) \leq \mathrm { K L } ( Q _ { s } \| Q _ { t } ) . } \end{array}
$$

Applying Pinsker’s inequality at each x yields

$$
\begin{array} { r } { \mathbb { E } _ { x } \| p ^ { s } ( x ) - p ^ { t } ( x ) \| _ { 1 } \leq \sqrt { \mathbb { E } _ { x } \| p ^ { s } ( x ) - p ^ { t } ( x ) \| _ { 1 } ^ { 2 } } \leq \sqrt { 2 \mathbb { E } _ { x } \mathrm { K L } ( p ^ { s } ( x ) \| p ^ { t } ( x ) ) } \leq \sqrt { n + 1 } \| s - t \| _ { D } . } \end{array}
$$

## Claim 15.2. Conditional on the learned scale s¯ and the context x, the m fresh trials satisfy

$$
\mathbb { E } \big [ \| \widehat { p } ( x ) - p ^ { \bar { s } } ( x ) \| _ { 1 } \big | \bar { s } , x \big ] \leq \sqrt { K / m } .
$$

Proof. Write $p _ { a } = p _ { a } ^ { \bar { s } } ( x )$ . For each $^ { a , }$ the m indicators $\mathbf { 1 } \{ A _ { k } ^ { \prime } = a \}$ are conditionally independent Bernoulli variables with mean $p _ { a }$ . Their average ${ \widehat { p } } _ { a } ( x )$ therefore satisfies

$$
\mathbb { E } \big [ ( \widehat { p } _ { a } ( x ) - p _ { a } ) ^ { 2 } \mid \bar { s } , x \big ] = \frac { p _ { a } ( 1 - p _ { a } ) } { m } .
$$

By Cauchy–Schwarz and Jensen’s inequality,

$$
\begin{array} { r } { \mathbb { E } [ \| \widehat { p } ( x ) - p \| _ { 1 } \mid \bar { s } , x ] \leq \sqrt { K \displaystyle \sum _ { a = 1 } ^ { K } \mathbb { E } [ ( \widehat { p } _ { a } ( x ) - p _ { a } ) ^ { 2 } \mid \bar { s } , x ] } } \\ { = \sqrt { \displaystyle \frac { K } { m } \left( 1 - \displaystyle \sum _ { a = 1 } ^ { K } p _ { a } ^ { 2 } \right) } \leq \sqrt { K / m } . } \end{array}
$$

Since $x \sim D$ is independent of the training draws, we may apply Claim 15.1 conditional on ¯s. Averaging over ¯s and using (10) gives

$$
\mathbb { E } \| p ^ { \bar { s } } ( x ) - p ^ { s ^ { * } } ( x ) \| _ { 1 } \leq \sqrt { ( n + 1 ) \mathbb { E } \| \bar { s } - s ^ { * } \| _ { D } ^ { 2 } } \leq B \sqrt { \frac { 2 ( n + 1 ) \varepsilon } { \lambda } } .
$$

Finally, $\widehat { q } ( x ) - q ^ { * } ( x ) = ( 1 - \gamma ) ( \widehat { p } ( x ) - p ^ { s ^ { * } } ( x ) )$ . Inserting $p ^ { \bar { s } } ( x )$ and applying Claim 15.2 yields

$$
\begin{array} { r l } & { \displaystyle \mathbb { E } \| \widehat { q } ( x ) - q ^ { * } ( x ) \| _ { 1 } \leq ( 1 - \gamma ) \left( \mathbb { E } \| \widehat { p } ( x ) - p ^ { \bar { s } } ( x ) \| _ { 1 } + \mathbb { E } \| p ^ { \bar { s } } ( x ) - p ^ { s ^ { * } } ( x ) \| _ { 1 } \right) } \\ & { \quad \quad \quad \leq \sqrt { \displaystyle \frac { K } { m } } + 2 B \sqrt { \displaystyle \frac { ( n + 1 ) \varepsilon } { 2 \lambda } } . } \end{array}
$$

## 5.3 Regret at the actual score

Lemma 16 (Sensitivity to the played probabilities). If $q _ { a } , r _ { a } \ge \gamma / K$ for all $^ { a , }$ then $| \Phi _ { S , x } ( q ) -$ $\Phi _ { S , x } ( r ) \vert \le ( 1 + K / \gamma ) \vert \vert q - r \vert \vert _ { 1 }$

Proof. Fix S and x. For each action a, write $f _ { a } ( \pi ) = \mathbf { 1 } \{ \pi ( x ) = a \}$ and define

$$
g _ { a } ( b ) = R ( S ) - R ( S - b f _ { a } ) , \qquad b \geq 0 .
$$

Thus $g _ { a } ( b )$ is the decrease in the potential when every policy predicting a at x receives a score penalty of b. We will bound how much the term $q _ { a } g _ { a } ( c _ { a } / q _ { a } )$ changes when $q _ { a }$ is replaced by $r _ { a }$

We first verify the properties of $g _ { a }$ used in this bound. Let $s _ { 0 } , s _ { 1 }$ be minimizing tables at scores $S _ { 0 } , S _ { 1 }$ , respectively. For $0 \leq t \leq 1$ , put $S _ { t } = ( 1 - t ) S _ { 0 } + t S _ { 1 }$ and $s _ { t } = ( 1 - t ) s _ { 0 } + t s _ { 1 }$ . The table $s _ { t }$ is feasible. For every fixed pool, linearity of $P _ { \pi }$ in the table gives

$$
\begin{array} { r l } & { \underset { \pi } { \operatorname* { m a x } } [ S _ { t } ( \pi ) + P _ { \pi } ( s _ { t } ) ] = \underset { \pi } { \operatorname* { m a x } } \Big ( ( 1 - t ) [ S _ { 0 } ( \pi ) + P _ { \pi } ( s _ { 0 } ) ] + t [ S _ { 1 } ( \pi ) + P _ { \pi } ( s _ { 1 } ) ] \Big ) } \\ & { \qquad \leq ( 1 - t ) \underset { \pi } { \operatorname* { m a x } } [ S _ { 0 } ( \pi ) + P _ { \pi } ( s _ { 0 } ) ] + t \underset { \pi } { \operatorname* { m a x } } [ S _ { 1 } ( \pi ) + P _ { \pi } ( s _ { 1 } ) ] . } \end{array}
$$

Taking expectations and using convexity of the log penalty gives

$$
R ( S _ { t } ) \leq J _ { S _ { t } } ( s _ { t } ) \leq ( 1 - t ) J _ { S _ { 0 } } ( s _ { 0 } ) + t J _ { S _ { 1 } } ( s _ { 1 } ) = ( 1 - t ) R ( S _ { 0 } ) + t R ( S _ { 1 } ) .
$$

Thus R is convex. Applying this inequality to the scores $S - b _ { 0 } f _ { a }$ and $S - b _ { 1 } f _ { a }$ gives

$$
\begin{array} { r l } & { g _ { a } ( ( 1 - t ) b _ { 0 } + t b _ { 1 } ) = R ( S ) - R \big ( ( 1 - t ) ( S - b _ { 0 } f _ { a } ) + t ( S - b _ { 1 } f _ { a } ) \big ) } \\ & { \qquad \geq ( 1 - t ) g _ { a } ( b _ { 0 } ) + t g _ { a } ( b _ { 1 } ) . } \end{array}
$$

The potential is also nondecreasing in each score entry, since increasing a score can only increase the maximum at each fixed table. Adding the same constant to every score shifts every maximum by that constant, so $R ( S - b \mathbf { 1 } ) = R ( S ) - b$ . Since $0 \leq f _ { a } \leq 1$ ，

$$
R ( S ) - b = R ( S - b \mathbf { 1 } ) \leq R ( S - b f _ { a } ) \leq R ( S ) .
$$

Hence $0 \leq g _ { a } ( b ) \leq b$ and $g _ { a } ( 0 ) = 0$ . Moreover, if $b _ { 0 } \leq b _ { 1 }$ , then $R ( S - b _ { 1 } f _ { a } ) \leq R ( S - b _ { 0 } f _ { a } )$ , so $g _ { a } ( b _ { 0 } ) \leq g _ { a } ( b _ { 1 } )$

Now fix one action a and two possible probabilities $\gamma / K \leq s \leq t$ . Since

$$
\frac { c _ { a } } { t } = \frac { s } { t } \frac { c _ { a } } { s } + \left( 1 - \frac { s } { t } \right) 0 ,
$$

concavity and $g _ { a } ( 0 ) = 0$ give $t g _ { a } ( c _ { a } / t ) \ge s g _ { a } ( c _ { a } / s )$ . For the reverse bound on their diference, use $c _ { a } / t \leq c _ { a } / s$ and monotonicity of $g _ { a } { \mathrm { : } }$

$$
\begin{array} { c l l } { 0 \leq t g _ { a } ( c _ { a } / t ) - s g _ { a } ( c _ { a } / s ) = ( t - s ) g _ { a } ( c _ { a } / t ) + s \big ( g _ { a } ( c _ { a } / t ) - g _ { a } ( c _ { a } / s ) \big ) } \\ { } & { } & { \leq ( t - s ) g _ { a } ( c _ { a } / t ) } \\ { } & { } & { \leq ( t - s ) \displaystyle \frac { c _ { a } } { t } \leq \displaystyle \frac { K } { \gamma } ( t - s ) . } \end{array}
$$

The last inequality uses $c _ { a } \leq 1$ and $t \geq \gamma / K$ . Taking $s = \operatorname* { m i n } \{ q _ { a } , r _ { a } \}$ and $t = \operatorname* { m a x } \{ q _ { a } , r _ { a } \}$ therefore yields, for every $a \in [ K ]$

$$
| q _ { a } g _ { a } ( c _ { a } / q _ { a } ) - r _ { a } g _ { a } ( c _ { a } / r _ { a } ) | \leq \frac { K } { \gamma } | q _ { a } - r _ { a } | .
$$

Finally, expand the diference of the two one-round quantities. From $\Phi _ { S , x } ( q ) ~ = ~ q \cdot c -$ $\textstyle \sum _ { a } q _ { a } g _ { a } ( c _ { a } / q _ { a } )$ , we obtain

$$
\begin{array} { l } { \displaystyle | \Phi _ { S , x } ( q ) - \Phi _ { S , x } ( r ) | = \left| \displaystyle \sum _ { a = 1 } ^ { K } ( q _ { a } - r _ { a } ) c _ { a } - \displaystyle \sum _ { a = 1 } ^ { K } \left( q _ { a } g _ { a } ( c _ { a } / q _ { a } ) - r _ { a } g _ { a } ( c _ { a } / r _ { a } ) \right) \right| } \\ { \displaystyle \qquad \leq \displaystyle \sum _ { a = 1 } ^ { K } | q _ { a } - r _ { a } | c _ { a } + \displaystyle \sum _ { a = 1 } ^ { K } | q _ { a } g _ { a } ( c _ { a } / q _ { a } ) - r _ { a } g _ { a } ( c _ { a } / r _ { a } ) | } \\ { \displaystyle \qquad \leq \displaystyle \sum _ { a = 1 } ^ { K } | q _ { a } - r _ { a } | + \displaystyle \frac { K } { \gamma } \displaystyle \sum _ { a = 1 } ^ { K } | q _ { a } - r _ { a } | } \\ { \displaystyle \qquad = \left( 1 + \displaystyle \frac { K } { \gamma } \right) \| q - r \| _ { 1 } . } \end{array}
$$

Proof of Theorem 11. Condition on the actual history $\mathcal { H } _ { t - 1 }$ , so the incoming score $S$ is fixed. Write $\mathrm { P r } _ { t }$ and $\mathbb { E } _ { t }$ for probability and expectation conditional on this history. Run the capped and uncapped algorithms with the same context $x _ { t } ,$ pool counts, pool samples, and inserted noise vectors. Let $q _ { t }$ and $\widehat { q _ { t } }$ be their respective play probabilities at $x _ { t }$

Let $M _ { 1 } , \dots , M _ { J + m }$ be the uncapped pool counts and let $E = \{ \operatorname* { m a x } _ { i } M _ { i } > M _ { \operatorname* { m a x } } \}$ be the event that the cap is exceeded. For each $M _ { i } \sim \operatorname { P o i s s o n } ( n )$

$$
\mathbb { E } _ { t } [ e ^ { M _ { i } } ] = e ^ { - n } \sum _ { k = 0 } ^ { \infty } \frac { ( e n ) ^ { k } } { k ! } = \exp ( ( e - 1 ) n ) .
$$

By the union bound and Markov’s inequality,

$$
\begin{array} { l } { \displaystyle \operatorname* { P r } _ { t } ( E ) \leq \sum _ { i = 1 } ^ { J + m } \operatorname* { P r } _ { t } ( e ^ { M _ { i } } > e ^ { M _ { \operatorname* { m a x } } } ) } \\ { \leq ( J + m ) \exp ( ( e - 1 ) n - M _ { \operatorname* { m a x } } ) } \\ { \leq ( J + m ) \exp \left( - \log \frac { 4 ( J + m ) } { \delta } \right) = \frac { \delta } { 4 } . } \end{array}
$$

On $E ^ { c }$ , the two algorithms use exactly the same draws and hence $q _ { t } = \widehat { q _ { t } }$ . On $E _ { \mathrm { { i } } }$ both outputs are probability vectors, so

$$
\| q _ { t } - { \widehat { q } } _ { t } \| _ { 1 } \leq \sum _ { a = 1 } ^ { K } ( q _ { t , a } + { \widehat { q } } _ { t , a } ) = 2 .
$$

Thus the expected error introduced by the cap is

$$
\mathbb { E } _ { t } \Vert q _ { t } - \widehat { q } _ { t } \Vert _ { 1 } = \mathbb { E } _ { t } [ { \mathbf { 1 } \{ E \} \Vert q _ { t } - \widehat { q } _ { t } \Vert _ { 1 } } ] \leq 2 \operatorname* { P r } _ { t } ( E ) \leq \frac { \delta } { 2 } .
$$

For the uncapped output $\widehat { q _ { t } }$ , Lemma 15 applies under the full Poisson law, including draws on which E occurs. Using $\varepsilon = \lambda \delta ^ { 2 } / [ 3 2 B ^ { 2 } ( n + 1 ) ]$ and $m \ge 1 6 K / \delta ^ { 2 }$ , we obtain

$$
\begin{array} { r l } & { \displaystyle \mathbb { E } _ { t } \| \widehat { q } _ { t } - q ^ { * } ( x _ { t } ) \| _ { 1 } \leq 2 B \sqrt { \frac { ( n + 1 ) \varepsilon } { 2 \lambda } } + \sqrt { \frac { K } { m } } } \\ & { \qquad \leq 2 B \sqrt { \frac { \delta ^ { 2 } } { 6 4 B ^ { 2 } } } + \sqrt { \frac { \delta ^ { 2 } } { 1 6 } } = \displaystyle \frac { \delta } { 2 } . } \end{array}
$$

Inserting $\widehat { q _ { t } }$ between the actual and ideal play probabilities therefore gives

$$
\mathbb { E } _ { t } \Vert q _ { t } - q ^ { * } ( x _ { t } ) \Vert _ { 1 } \leq \mathbb { E } _ { t } \Vert q _ { t } - \widehat { q _ { t } } \Vert _ { 1 } + \mathbb { E } _ { t } \Vert \widehat { q _ { t } } - q ^ { * } ( x _ { t } ) \Vert _ { 1 } \leq \frac { \delta } { 2 } + \frac { \delta } { 2 } = \delta .
$$

Every coordinate of $q _ { t }$ and $q ^ { * } ( x _ { t } )$ is at least $\gamma / K$ , including on overflow, when $q _ { t , a } = 1 / K$ Applying Lemma 16 at the same context $x _ { t }$ gives

$$
\begin{array} { r l } & { \displaystyle \mathbb { E } _ { t } [ \Phi _ { S , x _ { t } } ( q _ { t } ) ] \leq \mathbb { E } _ { t } [ \Phi _ { S , x _ { t } } ( q ^ { * } ( x _ { t } ) ) ] + \left( 1 + \frac { K } { \gamma } \right) \mathbb { E } _ { t } \| q _ { t } - q ^ { * } ( x _ { t } ) \| _ { 1 } } \\ & { \quad \quad \quad \quad \quad \quad \leq \mathbb { E } _ { x \sim D } [ \Phi _ { S , x } ( q ^ { * } ( x ) ) ] + \left( 1 + \frac { K } { \gamma } \right) \delta } \\ & { \quad \quad \quad \quad = \mathbb { E } _ { x \sim D } [ \Phi _ { S , x } ( q ^ { * } ( x ) ) ] + \displaystyle \frac { 1 } { T } . } \end{array}
$$

The second line uses that $x _ { t }$ has law D conditional on $\mathcal { H } _ { t - 1 }$ , and the last line uses $\delta = 1 / [ T ( 1 { + } K / \gamma ) ]$ Lemma 13 gives the claimed per-round resource bound. □

## Acknowledgments and AI Usage

This work resulted from extensive interaction between the authors and OpenAI Astra, with the authors providing suggestions and critically checking Astra’s responses through iterative exchanges. OpenAI Astra was also used in writing and revising the manuscript through detailed prompts from the authors, with Gemini Argon assisting in some parts. The authors directed the process and evaluated the proposed ideas and arguments. Lean files formalizing the main theorem are available on https://github.com/kbanihashem/bandit-oracle-lean.

The authors thank Suho Shin and Max Springer for their collaboration on earlier work on this problem, which led to the publication of [BHSS23] and Alex Slivkins for originally suggesting the problem.

## References

[ACBFS02] Peter Auer, Nicol\`o Cesa-Bianchi, Yoav Freund, and Robert E. Schapire. The nonstochastic multiarmed bandit problem. SIAM Journal on Computing, 32(1):48–77, 2002.

[AHK<sup>+</sup>14] Alekh Agarwal, Daniel Hsu, Satyen Kale, John Langford, Lihong Li, and Robert E. Schapire. Taming the monster: A fast and simple algorithm for contextual bandits. In Proceedings of the 31st International Conference on Machine Learning, volume 32 of Proceedings of Machine Learning Research, pages 1638–1646, 2014.

[BH26] Gon Buzaglo and Elad Hazan. Oracle-eficient online classification with stochastic inputs and adversarial outputs. arXiv:2609.33760, 2026.

[BHSS23] Kiarash Banihashem, MohammadTaghi Hajiaghayi, Suho Shin, and Max Springer. An improved relaxation for oracle-eficient adversarial contextual bandits. In Advances in Neural Information Processing Systems, volume 36, 2023.

[LZ07] John Langford and Tong Zhang. The Epoch-Greedy algorithm for multi-armed bandits with side information. In Advances in Neural Information Processing Systems, volume 20, 2007.

[Pr´e73] Andr´as Pr´ekopa. On logarithmic concave measures and functions. Acta Scientiarum Mathematicarum (Szeged), 34:335–343, 1973.

[RS16] Alexander Rakhlin and Karthik Sridharan. BISTRO: An eficient relaxation-based method for contextual bandits. In Proceedings of the 33rd International Conference on Machine Learning, volume 48 of Proceedings of Machine Learning Research, pages 1977–1985, 2016.

[SLKS16] Vasilis Syrgkanis, Haipeng Luo, Akshay Krishnamurthy, and Robert E. Schapire. Improved regret bounds for oracle-based adversarial contextual bandits. In Advances in Neural Information Processing Systems, volume 29, 2016.