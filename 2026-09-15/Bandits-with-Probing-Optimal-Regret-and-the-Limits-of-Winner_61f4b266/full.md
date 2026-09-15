# Bandits with Probing: Optimal Regret and the Limits of Winner Feedback

Yongjie Guan

Zhejiang University of Technology

## Abstract

A learner probes at most k of n arms each round, receives the maximum of their rewards in [0, 1], and competes with the best fixed arm. When does the probing advantage pay for learning? We determine two minimax laws. Under independent stochastic rewards with winner feedback—the maximum and a winning label—or on arbitrary fixed sequences given a single signed contrast between block maxima, the minimax regret has order

$$
\Phi _ { n , k } ( T ) = \operatorname* { m i n } \left\{ { \frac { n - k } { n } } T , { \frac { n - k } { k } } \right\} , \qquad 2 \leq k < n .
$$

Under winner feedback, both arbitrary joint i.i.d. rewards and fixed sequences have minimax regret of order

$$
\mathcal { R } _ { n , k } ( T ) = \frac { n - k } { n } \operatorname* { m i n } \left\{ T , \frac { n + T } { k } , \sqrt { \frac { n T } { k } } \right\} .
$$

Both laws have universal constants and anytime upper bounds. The first reduces regret to a pure coverage cost: same-round contrasts absorb the stability cost, and independence permits exact resampling whose gains fund sample advancement. The second adds a learning cost that becomes comparable to coverage at horizon $n ;$ beyond nk, numerical maxima improve over labels alone. The lower bound allows every adaptive action size.

Keywords: probing, winner feedback, minimax regret, Tsallis mirror descent

## 1. Introduction

Taking the better of two rewards gives a learner an advantage over either arm alone. Can this advantage pay for learning which arms to probe? Each round, a learner selects at most k of n arms and receives their maximum. BestProbe reveals this value and a winning label; AllProbe reveals all selected values. Relative to any selected anchor, the maximum supplies a nonnegative payof surplus. We ask whether this surplus covers the learning cost, leaving bounded regret against the best fixed arm.

We answer this with two optimal budget laws. Under independent stochastic rewards, BestProbe achieves the coverage cost $\Theta ( \Phi _ { n , k } ( T ) )$ . The same law holds for arbitrary fixed sequences under AllProbe; a single signed contrast sufices. The horizon-uniform cost ranges from $\Theta ( n )$ with two probes to $\Theta ( 1 / ( n - 1 ) )$ with $n - 1$ probes. Allowing within-round dependence under BestProbe changes the law to $\Theta ( \mathcal { R } _ { n , k } ( T ) )$ , adding a learning cost; joint i.i.d. rewards already attain this worst case. All bounds are uniform in $n , k , T$ , with anytime upper bounds.

Relation to prior work. Bhaskara et al. (2023) introduced these probe models, obtaining $O ( n ^ { 2 } \log T )$ regret for two-probe BestProbe and $O ( n ^ { 2 } )$ for three-probe AllProbe under independent stochastic rewards. They asked whether two probes sufice for bounded regret and whether the dependence on n can be linear. We resolve both questions with optimal $O ( n )$ horizon-independent regret using two probes under the weaker BestProbe interface, and obtain the complete budget dependence and a bounded contrast policy for arbitrary fixed sequences. The winner-feedback law determines exactly what independence buys.

Making surplus pay for learning. For observed contrasts, a centered estimator links the stability cost of the mirror update to the same dispersion that generates the payof surplus; a fixed step size cancels their cumulative terms. Winner feedback hides the contrast. Under independence, we recover exact anchor samples by rejection resampling and use the associated gains to fund switches. The dificulty is that a rarely refreshed estimate can be reused many times, so recovery alone is not enough: sample advancement must also be funded. Confining the resulting inverse-exploration factor to a single optimal arm’s stream keeps the total cost linear.

Grouping lifts both two-probe policies to larger budgets. Near full coverage, a direct reduction would lose the vanishing factor $( n - k ) / n$ . A gain-funded gate preserves it: the baseline pays for activating a fresh learner, and a comparator already covered by the baseline contributes nonpositive regret.

The cost that winner feedback leaves behind. The second law decomposes into the same coverage cost plus a learning cost. Winner probabilities control positive regret directly; numerical maxima supply the square-root tail. For the matching lower bound, a correlated threshold construction renders winner labels uninformative while common noise masks numerical levels. Smaller probe sets yield more informative numerical observations but incur higher omission costs; a stopped omission argument links these quantities and retains the full factor $( n - k ) / n$ . Section 2 states both laws; Section 6 explains their feedback time scales.

Tsallis regularization and decoupled exploration are established tools (Zimmert and Seldin, 2021; Rouyer and Seldin, 2020); the contrast policy couples their stability cost to the surplus of the same probes. Bacchiocchi et al. (2026) study best-action queries under a horizon-wide query budget, including an obstruction from correlated rewards; here the subset budget renews every round. The comparator also matters: in max value-index bandits, the best fixed set absorbs the surplus available against our single-arm benchmark (Wang et al., 2024). In costly-probe bandits, probing incurs a fee and the benchmark is the optimal probe-or-pull action (Elumar et al., 2025).

## 2. Model and main results

Write $[ n ] = \{ 1 , \dots , n \} , \ n \geq 2$ , and $( x ) _ { + } = \operatorname* { m a x } \{ x , 0 \}$ . Before seeing round-t rewards $X _ { i , t } \in [ 0 , 1 ]$ , the learner chooses a nonempty $S t \subseteq [ n ] , | S _ { t } | \leq k$ , and earns $W _ { t } = \operatorname* { m a x } _ { i \in S _ { t } } X _ { i , t }$ A known label order breaks ties. Each selected set is probed in one physical round; every additional probing action consumes another round. An anytime policy does not use the horizon.

Feedback. BestProbe reveals $( W _ { t } , J _ { t } )$ , where $J _ { t }$ is the winning label; AllProbe reveals all selected rewards. Under Contrast, an action is one nonempty block A or an ordered pair (A, B) of disjoint nonempty blocks, with at most k labels in total. Two blocks reveal only

$$
D _ { t } = \operatorname* { m a x } _ { i \in B } X _ { i , t } - \operatorname* { m a x } _ { i \in A } X _ { i , t } ;
$$

a single block returns zero. The payof remains the maximum on the union, but is not separately observed. Write $\mathsf { F } _ { 1 } \preceq \mathsf { F } _ { 2 }$ for simulation by a distribution-free action-conditional kernel; minimax regret decreases with information (Blackwell, 1953). AllProbe can simulate both BestProbe and Contrast, which are incomparable. For two singleton blocks, rewards (1, 0) and (1, 0.9) give the same winner feedback but diferent contrasts; rewards (0.8, 0.3) and (0.6, 0.1) give the same contrast but diferent maxima. Degenerate laws on these vectors rule out simulation in either direction.

Rewards and regret. For temporally i.i.d. vectors with law $D _ { : }$ , write $\mu _ { i } ~ = ~ \mathbb { E } X _ { i , t }$ $v _ { i } = \operatorname { V a r } ( X _ { i , t } ) , \mu ^ { \star } = \operatorname* { m a x } _ { i } \mu _ { i } ,$ and $\Delta _ { i } = \boldsymbol { \mu } ^ { \star } - \boldsymbol { \mu } _ { i }$ . For stochastic rewards and for an array $x \in [ 0 , 1 ] ^ { n \times T }$ fixed before policy randomization, respectively, define

$$
\begin{array} { r l } & { R _ { T } ^ { \pi } ( D ) = T \mu ^ { \star } - \mathbb { E } _ { \pi } \displaystyle \sum _ { t \leq T } W _ { t } , } \\ & { R _ { T } ^ { \pi , \operatorname { s e q } } ( x ) = \displaystyle \operatorname* { m a x } _ { j } \displaystyle \sum _ { t \leq T } x _ { j , t } - \mathbb { E } _ { \pi } \displaystyle \sum _ { t \leq T } \displaystyle \operatorname* { m a x } _ { i \in S _ { t } } x _ { i , t } . } \end{array}
$$

Both regrets may be negative. The minimax value $\begin{array} { r } { \mathcal { V } _ { n , T } ^ { \mathsf { F } , ( k ) } = \operatorname* { i n f } _ { \pi _ { T } } \operatorname* { s u p } _ { D } R _ { T } ^ { \pi _ { T } } ( D ) } \end{array}$ uses product laws $D = \otimes _ { i } D _ { i }$ . Superscripts joint and seq replace this class by arbitrary joint laws and fixed arrays, respectively. Policies in the infimum may know T.

Put $d = n - k , \delta = d / n$ , and define

$$
\Phi _ { n , k } ( T ) = \delta \operatorname* { m i n } \{ T , n / k \} , \qquad \mathcal { R } _ { n , k } ( T ) = \delta \operatorname* { m i n } \left\{ T , \frac { n + T } { k } , \sqrt { \frac { n T } { k } } \right\} .
$$

For integers $n > k \geq 2$ and $T \geq 1$ , we prove the following minimax laws (Table 1).

Theorem 1 (Bounded regret at the coverage scale) For independent stochastic rewards,

$$
\begin{array} { r } { \frac { 1 } { 2 } \Phi _ { n , k } ( T ) \leq \mathcal { V } _ { n , T } ^ { \mathrm { A l l P r o b e } , ( k ) } \leq \mathcal { V } _ { n , T } ^ { \mathrm { B e s t P r o b e } , ( k ) } < 1 4 0 0 \Phi _ { n , k } ( T ) . } \end{array}\tag{2.1}
$$

For arbitrary fixed arrays,

$$
\begin{array} { r } { \frac { 1 } { 2 } \Phi _ { n , k } ( T ) \leq \gamma _ { n , T } ^ { \mathsf { A l l P r o b e , s e q } , ( k ) } \leq \gamma _ { n , T } ^ { \mathsf { C o n t r a s t , s e q } , ( k ) } \leq 2 5 6 \Phi _ { n , k } ( T ) . } \end{array}\tag{2.2}
$$

Each upper bound is attained by one explicit anytime policy.

Theorem 2 (The complete winner-feedback law) For each $\mathsf { M } \in \{ \mathrm { j o i n t } , \mathrm { s e q } \}$ ，

$$
\frac { 1 } { 1 0 2 4 } \mathscr { R } _ { n , k } ( T ) \leq \mathscr { V } _ { n , T } ^ { \mathrm { B e s t P r o b e } , \mathsf { M } , ( k ) } \leq 6 4 \mathscr { R } _ { n , k } ( T ) .\tag{2.3}
$$

One explicit anytime policy attains both upper bounds. The lower bounds allow arbitrary adaptive action sizes from one to k.

The stochastic bound in Theorem 1 holds for every channel between BestProbe and AllProbe, and requires only that each set of at most k coordinates has the product law of its marginals (Appendix C.3). The fixed-array bound also gives the same minimax order for AllProbe and Contrast under arbitrary joint i.i.d. rewards, since $T$ max<sub>j</sub> $\begin{array} { r } { \mu _ { j } \leq \mathbb { E } \operatorname* { m a x } _ { j } \sum _ { t \leq T } X _ { j , t } } \end{array}$ and the policy has an array-wise guarantee.

Coverage and learning. The laws share a common structure:

$$
\mathcal { R } _ { n , k } ( T ) \asymp \underbrace { \Phi _ { n , k } ( T ) } _ { \mathrm { c o v e r a g e ~ c o s t } } + \underbrace { \frac { \delta } { k } \operatorname* { m i n } \{ T , \sqrt { n k T } \} } _ { \mathrm { r e m a i n i n g ~ l e a r n i n g ~ c o s t } } .\tag{2.4}
$$

The sum lies between $\mathcal { R } _ { n , k } ( T )$ and $2 \mathcal { R } _ { n , k } ( T )$ (Appendix G.4). Thus contrasts on fixed arrays, and winner feedback under independence, eliminate the cumulative learning term. Under general winner feedback that term survives, even without temporal dependence.

The unavoidable coverage cost. Hide one deterministic unit-reward arm at a uniformly random label; all others are zero. Along the all-zero transcript, the first t actions cover at most kt labels. An undiscovered arm outside that union costs one in round $t ,$ so the prior-average regret is at least

$$
C _ { n , k } ( T ) : = \sum _ { t = 1 } ^ { T } ( 1 - k t / n ) _ { + } \geq \frac { 1 } { 2 } \Phi _ { n , k } ( T ) .\tag{2.5}
$$

These instances are both product laws and fixed arrays, and discovery creates no negative regret. Appendix A verifies the integer rounding, the exact one-round value, and the endpoints.

Proof architecture. Sections 3 and 4 construct the two $O ( n )$ learning engines: surpluspaid stability from contrasts, and gain-funded sample advancement from winner feedback. Section 5 lifts both engines to the coverage law, using activation to retain the vanishing cost near full coverage. Section 6 proves the winner-feedback law by matching winner-only control and numerical bandit learning to a stopped omission lower bound. The appendices give the detailed certificates, followed by extensions to partial numerical feedback and constrained actions.

Table 1: Prior upper bounds and our minimax rates against the best fixed single arm.
<table><tr><td>Reward model</td><td>Feedback</td><td>k</td><td>Expected regret</td><td>Result</td></tr><tr><td colspan="5">Bhaskara et al. (2023): upper bounds</td></tr><tr><td>Independent i.i.d.</td><td>BestProbe / AllProbe</td><td>2</td><td> $O ( n ^ { 2 } \log T )$ </td><td>Thm.8</td></tr><tr><td>Independent i.i.d.</td><td>AllProbe</td><td>3</td><td> $O ( n ^ { 2 } )$ </td><td>Thm. 11</td></tr><tr><td>Joint i.i.d.</td><td>AllProbe</td><td>4</td><td> ${ \widetilde O } ( n ^ { 8 / 3 } T ^ { 1 / 3 } )$ </td><td>Thm. 13</td></tr><tr><td colspan="5">This work: minimax rates for every  $2 \leq k < n$ </td></tr><tr><td>Independent i.i.d.</td><td>BestProbe / AllProbe</td><td>Any</td><td> $\Theta ( \delta \operatorname* { m i n } \{ T , n / k \} )$ </td><td>Thm. 1</td></tr><tr><td>Joint i.i.d. / fixed arrays</td><td>AllProbe / Contrast</td><td>Any</td><td> $\Theta ( \delta \operatorname* { m i n } \{ T , n / k \} )$ </td><td>Thm. 1</td></tr><tr><td>Joint i.i.d. / fixed arrays</td><td>BestProbe</td><td>Any</td><td> $\begin{array} { r } { \Theta \Big ( \delta \operatorname* { m i n } \Big \{ T , \frac { n + T } { k } , \sqrt { \frac { n T } { k } } \Big \} \Big ) } \end{array}$ </td><td>Thm. 2</td></tr><tr><td>Joint i.i.d. / fixed arrays</td><td>Winner label only</td><td>Any</td><td> $\Theta \big ( \dot { \delta } \operatorname* { m i n } \big \{ \dot { T } , \frac { n + T } { k } \big \} \big )$ </td><td>Cor. 11</td></tr></table>

Here $\delta = ( n - k ) / n$ , and $\widetilde O$ suppresses logarithmic factors. All our rates have universal constants and anytime upper bounds. Our first two rates are Θ(min $\{ T , n \} )$ ) at $k = 2$ and saturate at $\Theta ( ( n - k ) / k )$ for general k.

## 3. Same-round contrasts pay for stability

Work on $m \geq 2$ arms and a fixed loss array $\ell _ { i , t } = 1 - x _ { i , t }$ . Probing an anchor $A _ { t }$ and an exploratory arm $B _ { t }$ incurs min $\{ \ell _ { A _ { t } , t } , \ell _ { B _ { t } , t } \}$ . Centering removes losses common to all arms: on a constant loss vector, both the estimate and the probing surplus vanish.

Start from $p _ { 1 , i } = 1 / m$ . With $\Delta _ { m }$ the probability simplex, use

$$
Z _ { t } = \sum _ { i } { \sqrt { p _ { t , i } } } , \quad q _ { t , i } = { \textstyle { \frac { 1 } { 2 } } } p _ { t , i } + { \frac { \sqrt { p _ { t , i } } } { 2 Z _ { t } } } , \quad \Psi ( p ) = - 2 \sum _ { i } { \sqrt { p _ { i } } } , \quad \eta = { \frac { 1 } { 1 6 { \sqrt { m } } } } .\tag{3.1}
$$

Independently draw $A _ { t } \sim p _ { t }$ and $B _ { t } \sim q _ { t }$ , probe their distinct labels, and set

$$
g _ { t , i } = \frac { \mathbf { 1 } \{ B _ { t } = i \} } { q _ { t , i } } ( \ell _ { i , t } - \ell _ { A _ { t } , t } ) , \qquad p _ { t + 1 } = \arg \operatorname* { m i n } _ { p \in \Delta _ { m } } \{ \eta \langle g _ { t } , p \rangle + D _ { \Psi } ( p , p _ { t } ) \} .\tag{3.2}
$$

Here $D _ { \Psi } ( u , p ) = \Psi ( u ) - \Psi ( p ) - \langle \nabla \Psi ( p ) , u - p \rangle$ . Repeated labels require one probe and give $g _ { t } = 0$ . The update has positive coordinates and is computable by a scalar normalization root (Appendix B).

Theorem 3 (Two-probe contrast policy) For every fixed array and horizon, the policy (3.1)–(3.2) satisfies $R _ { T } ^ { \mathrm { s e q } } \leq 3 2 ( m - \sqrt { m } )$

Proof Write $\mathbb { E } _ { t }$ for expectation conditional on the pre-round history, and suppress t. Define the learner-induced dispersion and the improvement over the anchor by

$$
\bar { \ell } = \langle p , \ell \rangle , \qquad \mathsf { v } = \sum _ { i } p _ { i } ( \ell _ { i } - \bar { \ell } ) ^ { 2 } , \qquad \gamma = \mathbb { E } _ { t } ( \ell _ { A } - \ell _ { B } ) _ { + } .
$$

The vector ℓ is fixed; v is not an environmental variance. Independence of the draws gives $\mathbb { E } _ { t } g _ { i } = \ell _ { i } - \bar { \ell } ,$ a common shift that preserves regret. The two components of $q$ yield

$$
V : = \mathbb { E } _ { t } \sum _ { i } p _ { i } ^ { 3 / 2 } g _ { i } ^ { 2 } \leq 2 Z \sum _ { i , a } p _ { i } p _ { a } ( \ell _ { i } - \ell _ { a } ) ^ { 2 } = 4 Z \mathsf { v } ,\tag{3.3}
$$

$$
\begin{array} { r } { \gamma \geq \frac { 1 } { 4 } \mathbb { E } _ { A , B \sim p } | \ell _ { A } - \ell _ { B } | \geq \frac { 1 } { 4 } \mathbb { E } _ { A , B \sim p } ( \ell _ { A } - \ell _ { B } ) ^ { 2 } = \frac { 1 } { 2 } \mathsf { v } . } \end{array}\tag{3.4}
$$

The square-root component controls inverse probabilities; the $p$ component supplies the matching improvement.

The estimate can be negative, but $\eta \sqrt { p _ { i } } | g _ { i } | \le 2 \eta Z \le 1 / 8$ . The signed mirror inequality in Appendix B.1 therefore gives

$$
\langle p - u , g \rangle \leq \frac { D _ { \Psi } ( u , p ) - D _ { \Psi } ( u , p ^ { + } ) } { \eta } + 2 \eta \sum _ { i } p _ { i } ^ { 3 / 2 } g _ { i } ^ { 2 } .
$$

The actual regret against a fixed arm $j$ equals the anchor regret minus the probing improvement. Telescoping with $u = e _ { j }$ and using $D _ { \Psi } ( e _ { j } , p _ { 1 } ) = 2 ( \sqrt { m } - 1 )$ gives

$$
\begin{array} { r l r } {  { R _ { T } ( j ) = \mathbb E \sum _ { t \leq T } \langle p _ { t } - e _ { j } , \ell _ { t } \rangle - \sum _ { t \leq T } \mathbb E \gamma _ { t } } } \\ & { } & { \leq \frac { 2 ( \sqrt { m } - 1 ) } { \eta } + \sum _ { t \leq T } \mathbb E [ 8 \eta \sqrt { m } \mathsf v _ { t } - \gamma _ { t } ] \leq 3 2 ( m - \sqrt { m } ) . } \end{array}\tag{3.5}
$$

The array’s best arm is fixed before randomization, so we may choose it as $j$

## 4. Recovering and paying for winner samples

Winner feedback hides the losing value needed by the contrast update. For $m \geq 3$ independent arms, we recover exact samples and use their associated gains to fund sample advancement. The obstacle is reuse: a stale estimate can accumulate cost before its stream advances.

The scores have summable one-sided errors over fixed sample prefixes. Matching charge weights to advancement probabilities preserves this summability under adaptive reuse. Only one fixed optimal arm pays an inverse-exploration factor of order m:

$$
\underbrace { O ( m ) \mathrm { ~ s t r e a m s } \times O ( 1 ) } _ { \mathrm { o r d i n a r y ~ s c o r e ~ e r r o r s } } + \underbrace { 1 \mathrm { ~ o p t i m a l ~ s t r e a m } \times O ( m ) } _ { \mathrm { r e s i d u a l ~ o p t i m a l ~ e r r o r } } = O ( m ) .
$$

A gain bank funds changes of anchor; runner and scout roles supply the advancement probabilities in Table 2. Without a gain bank, unnecessary switches incur persistent perround cost even when every pair probe includes the optimal arm. Without scouts, an arm with low initial samples can remain permanently stale (Appendix J).

## 4.1. Exact samples from winner feedback

Fix a set $S$ and an anchor $a \in S$ before fresh rewards. Let $M _ { - a } = \operatorname* { m a x } _ { i \in S \backslash \{ a \} } X _ { i }$ , with max $\begin{array} { r } { \varnothing \ = \ 0 . } \end{array}$ , and $W = \operatorname * { m a x } \{ X _ { a } , M _ { - a } \}$ . If a wins, return $( \widetilde { X } _ { a } , G ) = ( W , 0 )$ . Otherwise let $j$ be the winner. The anchor’s losing set is $A = [ 0 , W ] ~ { \mathrm { i f } } ~ j$ wins a tie with $^ { a , }$ and $A = [ 0 , W )$ otherwise. Probe a alone until its first draw in A, denoted $X _ { a } ^ { \dagger }$ , and return $( \widetilde { X } _ { a } , \bar { G } ) = ( X _ { a } ^ { \dagger } , W - X _ { a } ^ { \dagger } )$ . Only the returned anchor sample enters its persistent stream. Every rejected and accepted singleton counts toward physical time.

Lemma 4 (Predictable winner decomposition) For independent arms and a prereward choice of (S, a),

$$
\begin{array} { r } { ( \widetilde { X } _ { a } , G ) \overset { d } { = } ( X _ { a } , ( M _ { - a } - X _ { a } ) _ { + } ) , \qquad W = \widetilde { X } _ { a } + G . } \end{array}\tag{4.1}
$$

Conditional on the pre-call history, the procedure terminates almost surely, uses at most one extra singleton in expectation, and has expected regret at most $2 \Delta _ { a } - \Gamma _ { a } ^ { S }$ , where $\Gamma _ { a } ^ { S } =$ E ${ \mathrm { : } \operatorname* { m a x } } _ { i \in S } X _ { i } - \mu _ { a }$ . Under predictable adaptive calls, returned samples are revealed prefixes of independent i.i.d. arm arrays. The statement extends to independent marked outcomes with a known tie order (Appendix C).

Proof Conditional on nonanchor outcomes, the losing anchor and accepted replacement have the same restricted law. Mixing with the winning branch proves (4.1). A losing set of mass $b > 0$ is entered with probability b and needs $1 / b$ extra draws; zero-mass branches never occur. The initial probe costs $\Delta _ { a } - \Gamma _ { a } ^ { S }$ , and each extra singleton costs $\Delta _ { a }$ in expectation. Iterating this conditional kernel proves the adaptive and marked versions (Appendix C).

Lemma 5 (Variance–surplus bound) For independent X, $Y \in [ 0 , 1 ]$

$$
\mathbb { E } ( Y - X ) _ { + } \geq \operatorname* { m a x } \{ \operatorname { V a r } X , \operatorname { V a r } Y \} - ( \mathbb { E } X - \mathbb { E } Y ) _ { + } .
$$

Proof For $Z \in [ 0 , 1 ]$ , the identities Var $Z = \mathbb { E } [ ( Z - \mathbb { E } Z ) Z ] = \mathbb { E } [ ( \mathbb { E } Z - Z ) ( 1 - Z ) ]$ show that both $\mathbb { E } ( Z - \mathbb { E } Z ) _ { + }$ and $\mathbb { E } ( \mathbb { E } Z - Z ) _ { + }$ dominate Var Z. Conditional Jensen gives $\mathbb { E } ( Y - X ) _ { + } \geq$ $\mathbb { E } ( \mathbb { E } Y - X ) _ { - }$ <sub>+</sub> and $\mathbb { E } ( Y - X ) _ { + } \geq \mathbb { E } ( Y - \mathbb { E } X ) _ { + }$ . Apply $( u - c ) _ { + } \geq u _ { + } - c _ { + }$ to obtain the claim.

For distinct arms, write $\gamma _ { i j } = \mathbb { E } ( X _ { j } - X _ { i } ) _ { + }$ . Directed gains satisfy $\gamma _ { i j } - \gamma _ { j i } = \mu _ { j } - \mu _ { i }$ . If q is optimal and $a \neq q$ , then

$$
\gamma _ { a q } \geq \Delta _ { a } , \qquad \gamma _ { a q } \geq \operatorname* { m a x } \{ v _ { a } , v _ { q } \} .\tag{4.2}
$$

One surplus controls both the mean gap and the estimation variances.

A block comprises a pair probe and its recovery singletons. Blocks are indexed by $b ,$ physical rounds by t.

Theorem 6 (Two-probe guarantee) Algorithm 1 uses only pair and singleton probes and has regret less than 236m − 360 at every deterministic physical horizon. It is anytime and uses no unknown distributional parameters.

## 4.2. Scores and policy

Index arm i’s returned exact samples by j, including its two initialization samples. For a prefix $X _ { i , 1 } , \ldots , X _ { i , s }$ of length $s \geq 2$ , use

$$
\begin{array} { r l r } { \overline { { \boldsymbol X } } _ { i , s } = \displaystyle \frac { 1 } { s } \sum _ { j = 1 } ^ { s } \boldsymbol X _ { i , j } , } & { \widehat v _ { i , s } = \displaystyle \frac { 1 } { s - 1 } \sum _ { j = 1 } ^ { s } ( \boldsymbol X _ { i , j } - \overline { { \boldsymbol X } } _ { i , s } ) ^ { 2 } , } & \\ { \widehat u _ { i } = \overline { { \boldsymbol X } } _ { i , s } + \kappa \widehat v _ { i , s } , } & { \widehat \ell _ { i } = \overline { { \boldsymbol X } } _ { i , s } - \kappa \widehat v _ { i , s } , } & \end{array}\tag{4.3}
$$

where $\kappa = 1 / 1 2$ . Both scores lie in [0, 1] and update after every returned sample. By (4.2), the surplus absorbs variance-scale quantities, so the fixed ofset $\kappa \widehat { v }$ captures the right scale: one-sided errors beyond it are summable without a horizon-dependent radius.

The upper-score form and the use of the top two arms with separate exploration follow Bhaskara et al. (2023, Section 5, Eq. (6)), whose three-probe AllProbe policy uses a roundrobin exploration probe to estimate means and variances. Their empirical variance has divisor s; ours uses $s - 1$ . Adapting this approach to two-probe BestProbe requires exact sample recovery and gain-funded stream advancement, with the optimal-arm residual’s inverse-exploration cost confined to one stream.

The arm with largest upper score is the default anchor $a ;$ the second is the runner r. The runner receives most comparisons, while a uniformly chosen scout explores the remaining arms. For any potential target $i \neq a ,$ let $x _ { i } = ( \widehat { \ell } _ { a } - \widehat { u } _ { i } ) _ { + }$ estimate the cost of switching to it. Each target has separate runner and scout banks. Each bank starts at $B = 0$ and uses its target’s estimate x to set

$$
\alpha = \left\{ \begin{array} { l l } { { 1 / 3 , } } & { { x = 0 , } } \\ { { \operatorname* { m i n } \{ 1 / 3 , B / x \} , } } & { { x > 0 , } } \end{array} \right. \quad B ^ { \prime } = { \operatorname* { m i n } } \left\{ \frac { 1 } { 3 } , B - \alpha x + \frac { ( 1 - I ) G } { 3 } \right\} .\tag{4.4}
$$

Draw I ∼ Bernoulli(α) before rewards: retain a and credit its gain if $I = 0 ;$ otherwise anchor on the target and discard its gain. The probed pair $\{ a , j \}$ is the same regardless of $I ;$ the switch determines which arm’s sample stream advances. Since recovery singletons probe the chosen anchor, switching to a lower-mean arm incurs regret through these additional rounds. The debit αx reserves the estimated switching cost. The switch cap keeps $a \mathrm { { : } } \mathrm { { s } }$ advancement probability at least $2 / 3 ;$ the balance cap bounds terminal credit. Separate role banks keep each role’s telescoping multiplier fixed despite diferent selection probabilities.

Each scout has selection probability $\omega = 1 / [ 3 ( m - 2 ) ]$ . A count, mean, centered second moment, and two banks per arm require $O ( m )$ scalar state.

Algorithm 1: Pair-probe policy on m ≥ 3 arms   
1 Obtain two singleton samples per arm; initialize all banks to zero.   
2 while the policy is running do   
3 Form $\widehat { u } _ { i } , \widehat { \ell } _ { i }$ from all returned samples using (4.3).   
4 Let $a , r$ be the top two arms by $\widehat { u } _ { i } .$ , with ties broken by label.   
5 Choose $( j , z ) = ( r , { \mathsf { R } } )$ with probability $p = 2 / 3 ;$ otherwise choose $j$ uniformly from   
$[ m ] \setminus \{ a , r \}$ and set $z = 5 .$   
6 Set $x _ { j } = ( \widehat { \ell } _ { a } - \widehat { u } _ { j } ) _ { + }$ and compute α from $B _ { j } ^ { z }$ by (4.4).   
7 Draw I ∼ Bernoulli(α), then probe $\{ a , j \}$   
8 Complete winner decomposition with anchor a if $I = 0 ,$ and $j$ if I = 1.   
9 Append only the returned anchor sample and update only $B _ { j } ^ { z }$ by (4.4).   
10 end

## 4.3. Surplus pays the switching cost

The block regret splits into a switching cost, paid by the gain bank, and a residual controlled through a one-step certificate. Both steps use four one-sided score errors:

$$
\begin{array} { r l } & { U _ { i } ^ { \uparrow } = ( \widehat { u } _ { i } - \mu _ { i } - v _ { i } / 6 ) _ { + } , \qquad U _ { i } ^ { \downarrow } = ( \mu _ { i } - \widehat { u } _ { i } ) _ { + } , } \\ & { L _ { i } ^ { \uparrow } = ( \widehat { \ell } _ { i } - \mu _ { i } ) _ { + } , \qquad L _ { i } ^ { \downarrow } = ( \mu _ { i } - \widehat { \ell } _ { i } - v _ { i } / 6 ) _ { + } . } \end{array}\tag{4.5}
$$

Let $\mathcal { G } _ { b }$ be the history before block $b \mathrm { ^ { \prime } s }$ target draw, including scores and banks. Let $\mathcal { F } _ { b } ^ { - }$ additionally reveal the target and its role, but not the switch coin or rewards. Thus $x _ { b } , \alpha _ { b }$ are $\mathcal { F } _ { b } ^ { - }$ -measurable. Write $\Delta B = B ^ { \prime } - B$ . If the retained-anchor gain has conditional mean γ, then $\mathbb { E } [ ( 1 - I ) G \mid { \mathcal { F } } ^ { - } ] = ( 1 - \alpha ) \gamma$ . The bank supplies both a cumulative payment rule and a one-step residual certificate:

Lemma $\mathbf { 7 }$ (Gain bank) Starting at zero, every finite update prefix satisfies

$$
\sum _ { b } \alpha _ { b } x _ { b } \leq \frac { 1 } { 3 } \sum _ { b } ( 1 - I _ { b } ) G _ { b } \quad p a t h w i s e .\tag{4.6}
$$

For ${ \mathcal { F } }$ <sup>−</sup>-measurable $e _ { \mathrm { a } } , e _ { \mathrm { t } } \geq 0$ and $Q \in \mathbb { R }$ , if $x \leq e _ { \mathrm { a } } + e _ { \mathrm { t } } - Q / 2$ and $Q \leq 2 \gamma$ , then

$$
Q \leq 6 \mathbb { E } [ I e _ { \mathrm { t } } + \alpha e _ { \mathrm { a } } + \Delta B \mid \mathcal { F } ^ { - } ] .\tag{4.7}
$$

Both inequalities follow from the debit and capped credit, including signed Q (Appendix E). Property (4.7) will be applied in the next subsection to the specific residual arising from the optimal arm.

Let $\mathcal { R } _ { b }$ be the realized regret of a completed block, $\Gamma _ { b } = \gamma _ { a _ { b } j _ { b } }$ , and $c _ { b } = ( \mu _ { a _ { b } } - \mu _ { j _ { b } } ) _ { + }$ Lemma 4, including all recovery rounds, gives

$$
\mathbb { E } [ \mathcal { R } _ { b } \ | \ \mathcal { F } _ { b } ^ { - } ] \le 2 \Delta _ { a _ { b } } - \Gamma _ { b } + \alpha _ { b } c _ { b } .\tag{4.8}
$$

Suppressing the block index, the score errors and Lemma 5 convert the switching cost into an estimated debit:

$$
\alpha c \leq \frac { 3 } { 2 } \alpha x _ { j } + \frac { 3 } { 2 } \alpha ( L _ { a } ^ { \downarrow } + U _ { j } ^ { \uparrow } ) + \frac { 1 } { 2 } \alpha \Gamma .\tag{4.9}
$$

For nonoptimal targets, (4.6) pays the debit from half the retained gain; adding α $\Gamma / 2$ spends half the total gain. Optimal targets have zero switching cost. Summing (4.8) gives

$$
\mathbb { E } \sum _ { b = 1 } ^ { N } \mathcal { R } _ { b } \leq \mathbb { E } \sum _ { b = 1 } ^ { N } ( C _ { b } + K _ { b } ) ,\tag{4.10}
$$

where

$$
\begin{array} { l } { { C _ { b } = \displaystyle \frac { 3 } { 2 } \alpha _ { b } ( L _ { a _ { b } } ^ { \downarrow } + U _ { j _ { b } } ^ { \uparrow } ) \mathbf { 1 } \{ \Delta _ { j _ { b } } > 0 \} , } } \\ { { { } } } \\ { { K _ { b } = \displaystyle \left\{ 2 \Delta _ { a _ { b } } - \Gamma _ { b } , \qquad \Delta _ { j _ { b } } = 0 , \right. } } \\ { { \left. 2 \Delta _ { a _ { b } } - \Gamma _ { b } / 2 , \quad \Delta _ { j _ { b } } > 0 . \right. } } \end{array}
$$

Here $C _ { b }$ collects the score-error residual from the switching-cost payment, and $K _ { b }$ is the remaining per-block regret handled by the residual certificate below. Appendix E.2 proves the conversion and cumulative payment, also for predictably included block prefixes.

## 4.4. A single residual certificate

Fix any optimal arm q and condition on $\mathcal { G } _ { b }$ . If $q = a$ , then $K _ { b } \le 0$ . Otherwise, averaging over the target gives $\mathbb { E } [ K _ { b } \mid \mathcal { G } _ { b } ] \le Q$ , where

$$
Q = \left\{ \begin{array} { l l } { 2 \Delta _ { a } - \frac { 2 } { 3 } \gamma _ { a q } , } & { q = r , } \\ { 2 \Delta _ { a } - \frac { 1 } { 3 } \gamma _ { a r } - \omega \gamma _ { a q } , } & { q \notin \{ a , r \} . } \end{array} \right.
$$

Omitted gain contributions are nonpositive. When $x _ { q } = 0$ , we use a frequently updated witness—the anchor or runner—to absorb the selected-arm error, and apply the optimal arm’s bank certificate to the remainder. When $x _ { q } > 0$ , the score separation itself implies large enough errors to route all of $Q$ through $q \mathrm { ^ { \prime } s }$ bank.

Lemma 8 (Residual certificate) Assume a $\neq q$ . If $q = r$ , set $w = a ;$ otherwise let w be the lower-mean member of $\{ a , r \}$ , breaking a mean tie by label. Define

$$
\begin{array} { r } { E _ { \mathrm { s e l } } = U _ { w } ^ { \uparrow } , \quad E _ { \mathrm { a n c } } = L _ { a } ^ { \uparrow } , \quad E _ { \mathrm { o p t } } = U _ { q } ^ { \downarrow } , \quad Q ^ { \circ } = Q - 2 E _ { \mathrm { s e l } } { \bf 1 } \{ x _ { q } = 0 \} . } \end{array}
$$

Then

$$
Q \leq 2 ( E _ { \mathrm { s e l } } + E _ { \mathrm { o p t } } ) , \qquad x _ { q } \leq E _ { \mathrm { a n c } } + E _ { \mathrm { o p t } } - Q ^ { \circ } / 2 , \qquad Q ^ { \circ } \leq 2 \gamma _ { a q } .
$$

When $x _ { q } = 0$ , the witness’s returned-sample stream advances with conditional probability at least $2 / 9$ given $\mathcal { G } _ { b }$

Proof Choosing a witness. If $q = r , \ ( 4 . 2 )$ gives $Q \leq 2 \Delta _ { w } - v _ { w } / 3$ . Otherwise, $\gamma _ { a r } \geq$ $v _ { w } - \left( \Delta _ { w } - \Delta _ { a } \right)$ and $\Delta _ { a } \leq \Delta _ { w }$ give the same bound. Since both top scores dominate $\widehat { u } _ { q } .$

$$
\begin{array} { r } { E _ { \mathrm { s e l } } + E _ { \mathrm { o p t } } \geq \Delta _ { w } - v _ { w } / 6 \geq Q / 2 . } \end{array}
$$

Choosing the account. If $x _ { q } = 0$ , subtracting $2 E _ { \mathrm { s e l } }$ leaves $Q ^ { \circ } \leq 2 { \cal E } _ { \mathrm { o p t } }$ , which proves the switching-cost inequality. If $x _ { q } > 0$ , no witness term is charged; instead,

$$
E _ { \mathrm { a n c } } + E _ { \mathrm { o p t } } \geq ( { \widehat { \ell } } _ { a } - \mu _ { a } ) + ( \mu _ { q } - { \widehat { u } } _ { q } ) = x _ { q } + \Delta _ { a } \geq x _ { q } + Q / 2 .
$$

In both cases $Q ^ { \circ } \leq Q \leq 2 \Delta _ { a } \leq 2 \gamma _ { a q } .$ , so the optimal arm’s bank certificate applies.

Advancing a charged witness. The witness is charged only when $x _ { q } = 0$ . If $w = a$ , its stream advances with probability at least $2 / 3$ . If $w = r$ , then $x _ { r } \leq x _ { q } = 0$ , so its switch probability is $1 / 3$ whenever selected. Runner selection therefore advances its stream with probability $p / 3 = 2 / 9$

The witness is used only in the analysis; the policy need not know its mean.

When $q \neq a , \mathrm { l e t } ~ z \in \{ \mathsf { R } , \mathsf { S } \}$ be its role and $\boldsymbol { w } _ { \mathsf { R } } = \boldsymbol { p } , \boldsymbol { w } _ { \mathsf { S } } = \boldsymbol { \omega }$ . The switch probability $\alpha _ { q } ^ { z }$ that its bank would produce is $\mathcal { G } _ { b } .$ -measurable. Applying the bank inequality to Lemma 8 when $j = q$ gives

$$
\begin{array} { r l r } {  { \mathbb { E } [ K _ { b } \ | \ \mathcal { G } _ { b } ] \le 2 U _ { w } ^ { \uparrow } { \bf 1 } \{ x _ { q } = 0 \} } } \\ & { } & { + \ \frac { 6 } { w _ { z } } \mathbb { E } \Big [ { \bf 1 } \{ j = q \} \big ( I U _ { q } ^ { \downarrow } + \alpha _ { q } ^ { z } L _ { a } ^ { \uparrow } + \Delta B _ { q } ^ { z } \big ) \ | \ \mathcal { G } _ { b } \Big ] . } \end{array}\tag{4.11}
$$

After target averaging, each error inside the expectation has weight $6 \alpha _ { q } ^ { z } .$ The bank increment remains signed; every update, including those from blocks with negative $Q _ { i }$ , will enter the telescope.

## 4.5. Charging errors in physical time

The next lemma controls adaptive reuse through summable prefix errors: a charge is afordable whenever its weight is bounded by a constant multiple of the stream’s advancement probability.

Lemma 9 (Prefix charging) Fix an arm and one error type in (4.5). In block b, let $e _ { b }$ be its current error and $D _ { b }$ indicate advancement of its returned-sample stream. Relative to a history $\mathcal { H } _ { b }$ before this advancement, suppose $e _ { b } , \lambda _ { b } \geq 0$ are measurable and $h _ { b } = \mathbb { E } [ D _ { b } \mid \mathcal { H } _ { b } ]$ $I f \lambda _ { b } \leq H h _ { b }$ for a constant $H \geq 0$ , then

$$
\mathbb { E } \sum _ { b \leq \tau } \lambda _ { b } e _ { b } \leq \frac { 1 1 3 } { 1 6 } { \cal H }
$$

for every bounded block prefix whose inclusion indicators are $\mathcal { H } _ { b }$ -measurable.

For a fixed returned-sample prefix of length $s ,$ let $e _ { s }$ be one of the four errors. Appendix D proves $\textstyle \sum _ { s \geq 2 } \mathbb { E } e _ { s } \leq 1 1 3 / 1 6$ in two steps. Pairing independent samples gives scores whose variance and mean ofset scale with the same arm variance; Spitzer’s identity and a quadratic drift bound make their errors summable (Spitzer, 1956; Lindley, 1952). Averaging over a uniform maximum matching recovers the actual all-sample score, so conditional Jensen transfers this bound without discarding observations.

For adaptive reuse, weight each current error by its probability of advancement. A fixed prefix advances at most once, so its total expected charge is at most $H \mathbb { E } e _ { s }$ . Summing over prefixes proves the lemma. Table 2 records all five ratios. Only the residual-optimal row carries $1 / \omega$ , and every such charge goes to the same fixed arm $q ;$ this realizes the linear accounting above.

Table 2: Error weights and returned-sample advancement probabilities, on blocks where the charge is active. The first two rows condition on $\mathcal { F } _ { b } ^ { - }$ ; the others on $\mathcal { G } _ { b }$ . The witness row requires $x _ { q } = 0$ . Each ratio bounds weight divided by advancement probability.
<table><tr><td>Charge</td><td></td><td>Weight Advancement</td><td>Ratio</td><td>Streams</td></tr><tr><td>Switching anchor  $L _ { a } ^ { \downarrow }$ </td><td> $3 \alpha / 2$ </td><td> $1 - \alpha$ </td><td>3/4</td><td>m</td></tr><tr><td>Switching target  $U _ { i } ^ { \uparrow }$ </td><td> $3 \alpha / 2$ </td><td>α</td><td>3/2</td><td> $m - 1$ </td></tr><tr><td>Selected witness  $U _ { w } ^ { \dagger }$ </td><td> $2$ </td><td> $\geq 2 / 9$ </td><td>9</td><td> $m - 1$ </td></tr><tr><td>Residual anchor  $L _ { a } ^ { \uparrow }$ </td><td> $6 \alpha _ { q } ^ { z }$ </td><td> $\geq 2 / 3$ </td><td>3</td><td> $m - 1$ </td></tr><tr><td>Residual optimal  $U _ { q } ^ { \downarrow }$ </td><td> $6 \alpha _ { q } ^ { z }$ </td><td> $w _ { z } \alpha _ { q } ^ { z }$ </td><td> $1 8 ( m - 2 )$ </td><td>1</td></tr></table>

Proof of Theorem 6. Let $\sigma _ { b }$ be block $b \mathrm { { ^ { \circ } s } }$ first physical round. Its inclusion $\chi _ { b } = 1 \{ \sigma _ { b } \leq T \}$ is known before the target and switch draws. Complete only the recovery procedure straddling $T .$ The added singletons have nonnegative expected regret, and $\begin{array} { r } { \mathbb { E } \sum _ { b } \chi _ { b } A _ { b } \leq T } \end{array}$ for the full extra counts $A _ { b }$ . Thus completion is integrable and preserves every charging ratio and bank telescope (Appendix C.2). The optimal arm’s two role banks contribute at most $6 m - 9 ;$ initialization costs at most $2 ( m - 1 )$ , even if interrupted. Summing Table 2 gives $R _ { T } < 2 3 6 m - 3 6 0$ (Appendix E.3). 厂

## 5. From two probes to every budget

A disjoint partition defines virtual rewards $Y _ { g , t } = \operatorname* { m a x } _ { i \in G _ { g } } X _ { i , t }$ . On fixed arrays, Contrast reveals their signed diference; AllProbe reveals both maxima. Under product rewards, disjoint groups are independent marked arms: the mark is the winning physical label, and the original tie order makes every virtual comparison exact, including recovery singletons—each probing one physical group (Appendix C). In either case, the group containing a fixed physical comparator dominates it. A virtual pair is feasible when its union has size at most k.

## 5.1. Sparse budgets: direct grouping

For $3 k < 2 n$ , use $m = \lceil n / s \rceil$ groups of size at most $s = \lfloor k / 2 \rfloor$ . Apply the corresponding pair policy. The ceiling calculation in Appendix F.1 gives

$$
R _ { T } ^ { \mathrm { s p a r s e } } < 1 2 3 0 { \frac { n - k } { k } } \quad ( \mathrm { B e s t P r o b e } ) , \qquad R _ { T } ^ { \mathrm { s e q , s p a r s e } } < 2 5 6 { \frac { n - k } { k } } \quad ( \mathrm { C o n t r a s t } ) .\tag{5.1}
$$

## 5.2. Dense budgets: gain-funded activation

Assume $3 k \geq 2 n$ and put $d = n - k$ . Choose a uniform baseline $G _ { 0 }$ of size $n - 2 d .$ , and split the remaining labels into $G _ { 1 } , G _ { 2 }$ , each of size d. Every virtual pair is feasible. A direct three-arm reduction costs $O ( 1 )$ even when $d / k$ vanishes. We instead activate with probability proportional to gain, so the expected accumulated gain pays the startup cost.

Directly observed gains. Condition on the partition. For fixed arrays, choose $g \in \{ 1 , 2 \}$ uniformly each gate round, probe $G _ { 0 } \cup G _ { g } ,$ , and activate a fresh three-arm contrast policy next round with probability $( Y _ { g , t } - Y _ { 0 , t } ) _ { + } / L$ . Theorem 3 permits $L = 4 1$ , since $3 2 ( 3 - { \sqrt { 3 } } ) < 4 1$ Let $\chi _ { t }$ indicate that the gate is running and $\begin{array} { r } { \gamma _ { t } = \frac { 1 } { 2 } \sum _ { g } ( Y _ { g , t } - Y _ { 0 , t } ) _ { + } } \end{array}$ . At most one trigger occurs, so

$$
\mathbb { E } \sum _ { t \leq T } \chi _ { t } \gamma _ { t } = L \mathbb { P } ( \mathrm { t r i g g e r ~ b y ~ } T ) \leq L .\tag{5.2}
$$

Conditional on its start time and history, the continuation sees a fixed sufix and uses fresh randomness, so its regret against any fixed comparator is at most L. For such a comparator $j ,$ , set $c _ { t } = x _ { j , t } - Y _ { 0 , t }$ , and let $s T$ be the probability that the continuation starts by $T$ . Since $\begin{array} { r } { L s _ { T } \leq \mathbb { E } \sum _ { t } \chi _ { t } \gamma _ { t } } \end{array}$

$$
R _ { T } ( j ) \leq \mathbb { E } \sum _ { t \leq T } \chi _ { t } ( c _ { t } - \gamma _ { t } ) + L s _ { T } \leq \mathbb { E } \sum _ { t \leq T } \chi _ { t } c _ { t } .
$$

The bound is nonpositive for $j \in G _ { 0 } ;$ otherwise $( c _ { t } ) _ { + } \leq 2 \gamma _ { t }$ gives 2L. A best fixed arm misses $G _ { 0 }$ with probability $2 d / n$ , so averaging over partitions gives

$$
R _ { T } ^ { \mathrm { s e q , d e n s e } } \leq 4 L { \frac { d } { n } } = 1 6 4 { \frac { d } { n } } .\tag{5.3}
$$

Gains recovered from winner feedback. Take $L = 3 4 8$ from Theorem 6 on three virtual arms. Each gate attempt draws $g$ uniformly and $\xi \sim$ Bernoulli $( 1 / L )$ before rewards, and probes $G _ { 0 } \cup G _ { g }$ . The thinning coin $\xi$ limits recovery to a $1 / L$ fraction of attempts, keeping the expected recovery cost bounded. If $\xi = 1$ , recover an exact $G _ { 0 }$ sample and activate a fresh pair policy with probability equal to the returned gain G. If activation does not occur, begin the next attempt. The fresh policy discards gate samples. Write

$$
\gamma = \textstyle { \frac { 1 } { 2 } } \sum _ { g } \mathbb { E } ( Y _ { g } - Y _ { 0 } ) _ { + } , \qquad c _ { 0 } = \mu ^ { \star } - \mathbb { E } Y _ { 0 } .
$$

Lemma 10 (Recovered-gain activation) Conditioned on the partition, regret is nonpositive $i f c _ { 0 } \le 0$ and at most $2 L + 2$ otherwise, at every physical horizon.

Indeed, started attempts have trigger probability $\gamma / L$ and expected recovery count at most $1 / L$ . Gain pays continuation, leaving at most $( c _ { 0 } + ( c _ { 0 } ) _ { + } / L ) \mathbb { E } N _ { T }$ , where $N _ { T }$ counts started attempts and $\gamma \mathbb { E } N _ { T } \leq L$ . If $c _ { 0 } > 0$ , a residual group contains an optimal arm, so $c _ { 0 } \leq 2 \gamma$ Appendix F.2 proves these identities at interrupted horizons using only actual probes for regret. Averaging over $G _ { 0 }$ gives

$$
R _ { T } ^ { \mathrm { d e n s e } } \leq \frac { 2 d } { n } ( 2 L + 2 ) \leq 1 3 9 6 \frac { d } { k } .\tag{5.4}
$$

Implementation under Contrast. Order the anchor group first and the exploratory group second. The pair update uses $g _ { t , B _ { t } } = - D _ { t } / q _ { t , B _ { t } }$ , and gate activation uses $( D _ { t } ) _ { + } / L$ with the baseline first. Repeated group draws give a single-block action and a zero update. These operations determine the entire fixed-array policy: full selected values, the two block maxima, and Contrast produce the same physical actions under the same internal randomization.

Proof of Theorem 1. Use the appropriate sparse or dense policy from round one. The bounds above settle $T \geq n / k$ . Before that time, sparsity gives $R _ { T } \leq T < 3 \delta T$ . In the dense regime only $T = 1$ remains; the initial gate includes each arm with probability $k / n$ , so $R _ { 1 } \leq \delta$ . Combine these bounds with (2.5) and the feedback simulations.

## 6. The learning cost of winner feedback

Theorem 2 describes the cost that remains when within-round independence is unavailable. Its upper bound uses labels for short horizons and numerical maxima for long ones. A single correlated family supplies the matching learning lower bound, while (2.5) supplies coverage.

## 6.1. Winner labels control positive regret

For $k < n / 2$ , draw k labels independently from the mixed distribution $q _ { t }$ in (3.1) on n arms and probe their distinct labels. A comparator that often beats all probes must have suficient probability of winning when sampled. Conditional on the pre-round history,

$$
\mathbb { E } _ { t } ( x _ { j , t } - W _ { t } ) _ { + } \leq \frac { \mathbb { P } _ { t } ( J _ { t } = j ) } { k q _ { t , j } } , \qquad g _ { t , i } = - \frac { \mathbf { 1 } \{ J _ { t } = i \} } { q _ { t , i } } .
$$

Use the same Tsallis mirror update with step $1 / ( 4 \sqrt { n } )$ . This estimate controls positive regret directly; it need not estimate a reward or a loss.

For $k \geq n / 2$ , use batches of $\lfloor k / 2 \rfloor$ rounds. Retain every label that has won in the current batch and fill the remaining probe slots uniformly from the other labels. Once a comparator wins, it is protected for the rest of the batch. Appendix G.1 shows that these two policies satisfy

$$
R _ { T } ( j ) : = \mathbb { E } \sum _ { t \leq T } ( x _ { j , t } - W _ { t } ) \leq 1 6 \delta \operatorname* { m i n } \biggl \{ T , \frac { n + T } { k } \biggr \}\tag{6.1}
$$

for every fixed comparator and array, using only $J _ { t }$ . The same bound holds for the sum of positive instantaneous regrets.

## 6.2. Numerical maxima supply the square-root tail

For $k < n / 2$ , partition the arms into $\lceil n / k \rceil$ groups of size at most k and treat their maxima as bandit rewards. For $k \geq n / 2$ , choose a uniform baseline of size $2 k - n$ , split the remaining labels into two $( n - k )$ -sets, and treat the two augmented baselines as actions. A comparator in the baseline costs no regret; its probability of missing the baseline is 2δ. The anytime Tsallis-INF guarantee (Zimmert and Seldin, 2021, Theorem 1, IW estimators) then gives $1 6 \delta \sqrt { n T / k }$ regret in either case, using only $W _ { t }$

Run the winner-only policy through round $n k ,$ then start the numerical policy with fresh randomization. This one switch attains the minimum of the three terms in $\mathcal { R } _ { n , k } ( T )$ , up to a universal constant. Appendix G.2 verifies the finite-time bound against a comparator fixed before both phases, proving the upper bound of Theorem 2.

## 6.3. A lower bound that pays for informative probes

Hide a label q and take $0 < \varepsilon \le 1 / 6$ and $0 < \tau \leq 1$ . Each round draw independent $U _ { 1 } , \dots , U _ { n } \sim \mathrm { U n i f o r m } [ 0 , 1 ]$ and $Z \sim { \mathcal { N } } ( 0 , 1 )$ , and set

$$
Y _ { i } = \varepsilon U _ { i } + ( 1 - \varepsilon ) { \bf 1 } \{ U _ { i } \geq U _ { q } \} , \qquad X _ { i } = F _ { N } ( Z + \tau Y _ { i } ) ,\tag{6.2}
$$

where $F _ { N }$ is the standard normal CDF. The rank-preserving threshold makes every selected winner label uniform, hiding $q \mathrm { ^ { \prime } s }$ identity. Common Gaussian noise masks the numerical levels, giving divergence $O ( \tau ^ { 2 } / ( s + 1 ) ^ { 2 } )$ for an s-set. The small $\varepsilon$ limits the surplus that could ofset the omission cost. The vectors are i.i.d., and $q$ is uniquely mean-optimal.

An action of size s omitting q contributes a positive omission term of order $\tau / ( s + 1 )$ while every action’s expected positive surplus over $q$ is at most $\tau \varepsilon / 3$ . Small sets reveal more per label but incur higher omission costs; the proof stops when $\textstyle \sum _ { t } 1 / ( \left| S _ { t } \right| + 1 )$ first reaches

$T / ( k + 1 )$ , placing all adaptive action sizes on a common scale. A relative change-of-measure bound retains a constant fraction of the stopped omission cost without losing an extra factor of δ.

The positive omission cost is retained through this stopping time, and possible negative regret is charged over all $T$ rounds. With $\varepsilon = \delta / [ 1 0 0 ( k + 1 ) ]$ , this subtraction is at most $\tau \delta T / [ 3 0 0 ( k + 1 ) ]$ . Appendix G.3 obtains

$$
\mathcal { V } _ { n , T } ^ { \mathsf { B e s t P r o b e } , \mathsf { M } , ( k ) } \geq \frac { \delta } { 5 1 2 } \operatorname* { m i n } \left\{ \frac { T } { k } , \sqrt { \frac { n T } { k } } \right\} , \qquad \mathsf { M } \in \{ \mathrm { j o i n t } , \mathrm { s e q } \} .
$$

Combining this with coverage proves Theorem 2. The fixed-array lower bound follows by averaging arrays drawn from the same joint i.i.d. family.

## 6.4. Two feedback time scales

Corollary 11 (Labels alone) Suppose only $J _ { t }$ is observed: the payof remains $W _ { t }$ but is not separately observed. For fixed arrays and arbitrary joint i.i.d. laws, the minimax rate is

$$
\Theta \bigg ( \delta \operatorname* { m i n } \bigg \{ T , \frac { n + T } { k } \bigg \} \bigg ) = \Theta \bigg ( \Phi _ { n , k } ( T ) + \frac { \delta T } { k } \bigg ) ,
$$

with universal constants and an anytime upper bound, for every $n > k \geq 2$ and $T \geq 1$

The upper bound is (6.1). For the lower bound, set $\tau = 1$ in (6.2): the complete label transcript is independent of the hidden arm, so a linear omission cost remains. Appendix G.4 gives the constants.

The two laws expose diferent roles for feedback. At $T \asymp n$ , the learning term in (2.4) becomes comparable to coverage; beyond this scale, general winner feedback separates from the bounded contrast law. $\operatorname { A t } T \asymp n k$ , numerical maxima start to improve the rate over labels alone. In particular,

$$
\mathcal { R } _ { n , k } ( T ) \asymp \delta \left\{ \begin{array} { l l } { T , } & { T \leq n / k , } \\ { n / k , } & { n / k \leq T \leq n , } \\ { T / k , } & { n \leq T \leq n k , } \\ { \sqrt { n T / k } , } & { T \geq n k . } \end{array} \right.\tag{6.3}
$$

Thus labels alone attain the BestProbe minimax order through $T = n k$ , and values improve it by a factor of $\Theta ( \sqrt { T / ( n k ) } )$ ) when $T / ( n k ) \to \infty$

Beyond the basic interfaces. Appendix H quantifies sparse numerical revelation under independent rewards; Appendix I extends the coverage law to heterogeneous quotas. Contrasts make surplus usable on every fixed array; independence enables the same bounded cost from winner feedback. With arbitrary within-round dependence, the remaining learning cost is exactly the second term of (2.4), up to universal constants.

## References

Francesco Bacchiocchi, Matteo Castiglioni, Alberto Marchesi, and Francesco Emanuele Stradi. Multi-armed bandits with best-action queries. arXiv preprint arXiv:2605.08287, 2026. doi: 10.48550/arXiv.2605.08287.

Aditya Bhaskara, Sreenivas Gollapudi, Sungjin Im, Kostas Kollias, and Kamesh Munagala. Online learning and bandits with queried hints. In 14th Innovations in Theoretical Computer Science Conference (ITCS 2023), volume 251 of Leibniz International Proceedings in Informatics, pages 16:1–16:24. Schloss Dagstuhl – Leibniz-Zentrum für Informatik, 2023. doi: 10.4230/LIPIcs.ITCS.2023.16.

David Blackwell. Equivalent comparisons of experiments. The Annals of Mathematical Statistics, 24(2):265–272, 1953. doi: 10.1214/aoms/1177729032.

Bangrui Chen and Peter I. Frazier. Dueling bandits with weak regret. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 731–739. PMLR, 2017.

Jessica Dai, Bailey Flanigan, Nika Haghtalab, Meena Jagadeesan, and Chara Podimata. Can probabilistic feedback drive user impacts in online platforms? In Proceedings of the 27th International Conference on Artificial Intelligence and Statistics, volume 238 of Proceedings of Machine Learning Research, pages 2512–2520. PMLR, 2024.

Eray Can Elumar, Cem Tekin, and Osman Yağan. Multi-armed bandits with costly probes. IEEE Transactions on Information Theory, 71(1):618–643, January 2025. doi: 10.1109/ TIT.2024.3506866.

Hédi Joulak and Bernhard Beckermann. On Gautschi’s conjecture for generalized Gauss– Radau and Gauss–Lobatto formulae. Journal of Computational and Applied Mathematics, 233(3):768–774, 2009. doi: 10.1016/j.cam.2009.02.083.

Tor Lattimore and Csaba Szepesvári. Bandit Algorithms. Cambridge University Press, 2020. doi: 10.1017/9781108571401.

Dennis V. Lindley. The theory of queues with a single server. Mathematical Proceedings of the Cambridge Philosophical Society, 48(2):277–289, 1952. doi: 10.1017/S0305004100027638.

Chloé Rouyer and Yevgeny Seldin. Tsallis-INF for decoupled exploration and exploitation in multi-armed bandits. In Proceedings of Thirty Third Conference on Learning Theory, volume 125 of Proceedings of Machine Learning Research, pages 3227–3249. PMLR, 2020.

Frank Spitzer. A combinatorial lemma and its application to probability theory. Transactions of the American Mathematical Society, 82(2):323–339, 1956. doi: 10.1090/ S0002-9947-1956-0079851-X.

Haiyong Wang, Daan Huybrechs, and Stefan Vandewalle. Explicit barycentric weights for polynomial interpolation in the roots or extrema of classical orthogonal polynomials. Mathematics of Computation, 83(290):2893–2914, 2014. doi: 10.1090/S0025-5718-2014-02821-4.

Yiliu Wang, Wei Chen, and Milan Vojnović. Combinatorial bandits for maximum value reward function under value-index feedback. In The Twelfth International Conference on Learning Representations, 2024.

Julian Zimmert and Yevgeny Seldin. Tsallis-INF: An optimal algorithm for stochastic and adversarial bandits. Journal of Machine Learning Research, 22(28):1–49, 2021.

## Use of generative-AI tools

The author used ChatGPT to assist with proof exploration, reference discovery, and manuscript refinement. The author takes responsibility for the final arguments and presentation.

## Appendix A. Coverage lower bound and endpoints

Place a deterministic unit-reward arm at a uniformly random location and let every other arm be zero. Before discovery, even the complete AllProbe transcript agrees with the all-zero transcript. Fix the policy’s auxiliary randomness. Along that transcript, the first t probe sets cover at most kt locations. If the hidden arm lies outside their union, the learner incurs regret one in round t. Hence, for every policy,

$$
\mathcal { V } _ { n , T } ^ { \sf A l l P r o b e } , ( k ) \geq C _ { n , k } ( T ) : = \sum _ { t = 1 } ^ { T } \left( 1 - \frac { k t } { n } \right) _ { + } .\tag{A.1}
$$

Write $x = n / k > 1$ . If $T \leq x$ , then

$$
C _ { n , k } ( T ) = T \left( 1 - \frac { T + 1 } { 2 x } \right) \geq \frac { T ( x - 1 ) } { 2 x } = \frac { 1 } { 2 } \Phi _ { n , k } ( T ) .
$$

If $T \geq x ,$ set $q = \lfloor x \rfloor$ and $f = x - q$ . Then

$$
C _ { n , k } ( T ) = q - \frac { q ( q + 1 ) } { 2 x } = \frac { x - 1 } { 2 } + \frac { f ( 1 - f ) } { 2 x } \geq \frac { 1 } { 2 } \Phi _ { n , k } ( T ) .
$$

This proves the finite-horizon lower bound. At $T = 1$ , it gives $( n - k ) / n$ . Conversely, a uniformly random k-subset includes a fixed optimal arm with probability $k / n$ , so $\mathbb { E } W _ { 1 } \geq$ $( k / n ) \mu ^ { \star }$ . Thus $R _ { 1 } \leq ( n - k ) / n$ , proving the exact one-round value for every intermediate feedback channel. The same argument applies to a fixed array, whose best arm at $T = 1$ has value at most one. The all-zero array gives the matching zero endpoint when $k = n$

For $k = 1$ , the ordinary stochastic-bandit lower bound diverges with $T$ (Lattimore and Szepesvári, 2020, Exercise 15.4); averaging its reward arrays gives the same conclusion for fixed sequences.

## Appendix B. Signed stability and sampling

## B.1. The local mirror inequality

Let $\begin{array} { r } { \Psi ( p ) = - 2 \sum _ { i } \sqrt { p _ { i } } } \end{array}$ , let p have positive coordinates, and let $p ^ { + }$ minimize $\eta \langle g , u \rangle + D _ { \Psi } ( u , p )$ over $u \in \Delta _ { m }$ . For any signed vector g satisfying $\eta \sqrt { p _ { i } } g _ { i } \ge - 1 / 2$ , we claim

$$
\langle p - u , g \rangle \leq \frac { D _ { \Psi } ( u , p ) - D _ { \Psi } ( u , p ^ { + } ) } { \eta } + 2 \eta \sum _ { i } p _ { i } ^ { 3 / 2 } g _ { i } ^ { 2 } , \qquad u \in \Delta _ { m } .\tag{B.1}
$$

The three-point inequality reduces this to bounding $\eta \langle g , p - p ^ { + } \rangle - D _ { \Psi } ( p ^ { + } , p )$ . For $f ( x ) =$ $- 2 \sqrt { x }$ on $x \geq 0$ , its conjugate is $f ^ { * } ( \theta ) = - 1 / \theta$ for $\theta < 0$ . For $p _ { i } > 0$ and $s _ { i } = \eta \sqrt { p _ { i } } g _ { i } > - 1$ direct maximization gives

$$
\operatorname* { s u p } _ { z \geq 0 } \{ \eta g _ { i } ( p _ { i } - z ) - D _ { f } ( z , p _ { i } ) \} = \frac { \eta ^ { 2 } p _ { i } ^ { 3 / 2 } g _ { i } ^ { 2 } } { 1 + s _ { i } } , \qquad z ^ { \ast } = \frac { p _ { i } } { ( 1 + s _ { i } ) ^ { 2 } } .\tag{B.2}
$$

Relaxing the simplex to the nonnegative orthant and summing (B.2) proves (B.1). Boundary comparators such as $e _ { j }$ are allowed: $D _ { \Psi } ( e _ { j } , p )$ is finite and the gradient is evaluated only at positive $p .$

For the contrast estimator, $q _ { i } \geq \sqrt { p _ { i } } / ( 2 Z )$ and $| \ell _ { i } - \ell _ { A } | \leq 1 { \mathrm { ~ g i v e ~ } } | s _ { i } | \leq 2 \eta Z \leq 1 / 8$ , verifying the signed condition in every round.

## B.2. Well-defined update and sampling roles

Strict convexity and the infinite negative derivative at zero place the minimizer in the simplex interior. Lagrange optimality gives

$$
p _ { i } ^ { + } = \big ( p _ { i } ^ { - 1 / 2 } + \eta g _ { i } + \lambda \big ) ^ { - 2 } , \qquad \sum _ { i } \bigl ( p _ { i } ^ { - 1 / 2 } + \eta g _ { i } + \lambda \bigr ) ^ { - 2 } = 1 .
$$

The sum is continuous and strictly decreasing from infinity to zero on $\lambda > - \operatorname* { m i n } _ { i } ( p _ { i } ^ { - 1 / 2 } + \eta g _ { i } )$ 2 so there is a unique root.

The mixture in (3.1) controls two diferent quantities. Removing its p component need not preserve $V = { \cal { O } } ( \sqrt { m } \gamma )$ . For $q _ { i } = \sqrt { p _ { i } } / Z$ and losses $\ell _ { 1 } = 0 , \ell _ { i } = 1$ for $i > 1$ , direct calculation gives $V / \gamma = 2 Z ^ { 2 } \sqrt { p _ { 1 } }$ . Taking $p _ { 1 } = p _ { 2 } = 1 / 4$ and spreading the other half uniformly gives $V / \gamma = Z ^ { 2 } = \Theta ( m )$ . Removing the square-root component also breaks this cancellation: for $q = p ,$ two arms, losses $( 0 , 1 )$ , and $p = ( 1 - \varepsilon , \varepsilon )$ , one obtains

$$
V / \gamma = \frac { 1 } { \sqrt { 1 - \varepsilon } } + \frac { 1 } { \sqrt { \varepsilon } } .
$$

These calculations isolate the two sampling components needed by the stability–surplus cancellation.

## Appendix C. Adaptive exact streams and physical stopping

## C.1. Conditional kernels and marked outcomes

Let $\mathcal { P } _ { \ell }$ be the history before procedure $\ell \mathrm { { s } }$ fresh rewards, including its set and anchor. For every Borel set E, Lemma 4 gives

$$
\mathbb { P } ( \boldsymbol { \widetilde { X } _ { \ell } } \in E \mid \mathcal { P } _ { \ell } ) = D _ { a _ { \ell } } ( E ) .
$$

Indeed, conditional on the nonanchor outcomes, their maximum and winning label determine a measurable losing set A. The original anchor on the losing branch and the accepted replacement both have law $D _ { a } ( \cdot \mid A )$ . On the winning branch, the observed anchor has the complementary restriction. The branch probabilities $D _ { a } ( A )$ and $1 - D _ { a } ( A )$ recover the full law, including atoms.

Pre-generate an independent infinite i.i.d. array for each arm. When an arm is chosen as anchor, use its next unused coordinate as the returned sample, and generate all remaining block variables, including feedback and length, from their conditional law given that coordinate. Iteration reproduces the full adaptive experiment. The underlying arrays are independent; their revealed prefix lengths are adaptive. Only the returned anchor sample enters a persistent stream.

For marked arms, apply the same construction to the losing set in the marked outcome space. Disjoint physical groups have independent marked maxima, and a physical union returns their greatest outcome under the shared reward-and-label order. The comparison is exact even when the labels of diferent groups are interlaced. For a random partition, condition on the realized groups. This justifies every use of the two-probe policy on virtual arms.

## C.2. Deterministic physical horizons

Recovery can have a heavy tail. For two independent Uniform[0, 1] arms and a fixed anchor, the extra singleton count A satisfies

$$
\mathbb { P } ( A \geq r ) = \int _ { 0 } ^ { 1 } y ( 1 - y ) ^ { r - 1 } d y = { \frac { 1 } { r ( r + 1 ) } } , \qquad r = 1 , 2 , \ldots .
$$

Thus $\mathbb { E } A = 1$ and $\mathbb { E } A ^ { 2 } = \infty ;$ the following argument uses only the first moment.

Let $\sigma _ { b }$ be the physical round of block $b \mathrm { ^ { \prime } s }$ initial pair probe, after initialization. For a fixed horizon $T _ { i }$ , define

$$
\chi _ { b } = { \bf 1 } \{ \sigma _ { b } \leq T \} , \qquad \tau _ { T } = \sum _ { b \geq 1 } \chi _ { b } \leq T .
$$

The preceding block lengths determine $\chi _ { b }$ before block $b \mathrm { { s } }$ target and switch draws. Complete only the sampling procedure, if any, that straddles T. If $A _ { b }$ denotes block b’s full additional singleton count, Lemma 4 gives $\mathbb { E } [ A _ { b } \mid { \mathcal { G } } _ { b } ] \leq 1$ . Thus

$$
\mathbb { E } \sum _ { b \leq \tau _ { T } } A _ { b } = \sum _ { b = 1 } ^ { T } \mathbb { E } [ \chi _ { b } A _ { b } ] \leq \mathbb { E } \tau _ { T } \leq T .
$$

The completed prefix is integrable. Every added singleton is selected before its fresh reward and has conditional expected regret equal to the gap of its anchor, which is nonnegative. Summing these predictable inclusion indicators therefore gives a nonnegative expected completion cost. If T falls during initialization, its cost is bounded by the full initialization cost in the same way.

It follows that

$$
R _ { T } \leq 2 \sum _ { i } \Delta _ { i } + \mathbb { E } \sum _ { b \leq \tau _ { T } } \mathcal { R } _ { b } .
$$

Multiply each block inequality by $\chi _ { b }$ and apply the debit telescope to the realized finite prefix to obtain

$$
\mathbb { E } \sum _ { b \leq \tau _ { T } } \mathcal { R } _ { b } \leq \mathbb { E } \sum _ { b \leq \tau _ { T } } ( C _ { b } + K _ { b } ) .
$$

For any advancement indicator $D _ { b }$ and its pre-advancement history $\mathcal { H } _ { b }$ ,

$$
\mathbb { E } [ \chi _ { b } D _ { b } \mid \mathcal { H } _ { b } ] = \chi _ { b } \mathbb { E } [ D _ { b } \mid \mathcal { H } _ { b } ] .
$$

Thus every charging ratio in Table 2 is preserved, as are the bank telescopes. For grouped arms, this argument uses the best virtual-arm mean, so singleton gaps remain nonnegative. The dense gate is accounted for separately through its actual physical costs in Appendix F.2.

## C.3. Action-local independence

Suppose reward vectors are i.i.d. across rounds and every set of at most k coordinates has the product law of its one-dimensional marginals. Conditional on the past and the next chosen action, its selected reward vector then has exactly the same law as under fully independent arms with those marginals. The reward and feedback are measurable functions of that vector and independent feedback randomization. Induction over physical rounds gives equality of the entire visible experiment, for every adaptive k-probe policy. The marginal means and comparator are unchanged. The stochastic part of Theorem 1 and Theorem 12 therefore hold under this action-local assumption. In particular, the pair policy requires only pairwise independence.

## Appendix D. All-sample scores and predictable reuse

## D.1. A drift bound for empirical prefixes

Let $Z _ { j }$ be i.i.d. bounded variables with mean ν and variance at most $c _ { \mathrm { v a r } } v .$ , and let $\overline { { Z } } _ { r } =$ $r ^ { - 1 } \dot { \sum _ { j = 1 } ^ { r } } Z _ { j }$ . For $\eta > 0$ , set

$$
Y _ { j } = \pm ( Z _ { j } - \nu ) , \qquad S _ { r } = \sum _ { j = 1 } ^ { r } Y _ { j } , \qquad c = \eta v .
$$

When $v = 0$ all deviations vanish. For $v > 0$ , the first-moment form of Spitzer’s identity (Spitzer, 1956) gives

$$
\sum _ { r = 1 } ^ { N } \frac { 1 } { r } \mathbb { E } ( S _ { r } - r c ) _ { + } = \mathbb { E } \operatorname* { m a x } _ { 0 \leq r \leq N } ( S _ { r } - r c ) .\tag{D.1}
$$

The finite-horizon identity extends to bounded real increments by continuity under lattice approximation. Consider Lindley’s recursion (Lindley, 1952),

$$
\mathcal { W } _ { 0 } = 0 , \qquad \mathcal { W } _ { r + 1 } = ( \mathcal { W } _ { r } + Y _ { r + 1 } - c ) _ { + } .
$$

Reversing the first r i.i.d. increments shows that W has the law of $\mathrm { m a x } _ { 0 \le j \le r } ( S _ { j } - j c )$ Hence $\mathbb { E } { \boldsymbol { \mathcal { W } } _ { r } }$ is nondecreasing. Independence and $( x _ { + } ) ^ { 2 } \leq x ^ { 2 }$ give

$$
\mathbb { E } \mathcal { W } _ { r + 1 } ^ { 2 } - \mathbb { E } \mathcal { W } _ { r } ^ { 2 } \leq - 2 c \mathbb { E } \mathcal { W } _ { r } + c _ { \mathrm { v a r } } \boldsymbol { v } + c ^ { 2 } .
$$

Summing and dropping the nonnegative terminal second moment yields

$$
\frac { 1 } { N } \sum _ { r = 0 } ^ { N - 1 } \mathbb { E } \mathcal { W } _ { r } \leq \frac { c _ { \mathrm { v a r } } v + c ^ { 2 } } { 2 c } .
$$

A nondecreasing sequence with bounded Cesàro averages has the same bound on its supremum. Letting $N \to \infty$ in (D.1) proves

$$
\sum _ { r \ge 1 } \mathbb { E } \big ( \pm ( \overline { { Z } } _ { r } - \nu ) - \eta v \big ) _ { + } \le \frac { c _ { \mathrm { v a r } } } { 2 \eta } + \frac { \eta v } { 2 } .\tag{D.2}
$$

## D.2. Maximum matching recovers the all-sample score

For two independent samples $X , X ^ { \prime }$ from one arm, put

$$
M = \frac { X + X ^ { \prime } } 2 , \qquad V = \frac { ( X - X ^ { \prime } ) ^ { 2 } } 2 , \qquad Z ^ { \pm } = M \pm \kappa V .
$$

Then $V \leq \operatorname* { m i n } \{ M , 1 - M \} , \mathbb { E } V = v , \mathrm { V a r } M = v / 2$ , and Var $V \leq \mathbb { E } V ^ { 2 } \leq v / 2$ . Consequently

$$
0 \leq Z ^ { \pm } \leq 1 , \qquad \mathbb { E } Z ^ { \pm } = \mu \pm \kappa v , \qquad \mathrm { V a r } ( Z ^ { \pm } ) \leq c _ { \mathrm { v a r } , \kappa } v , \quad c _ { \mathrm { v a r } , \kappa } = \frac { ( 1 + \kappa ) ^ { 2 } } { 2 } .
$$

Now fix $s \geq 2$ samples and write $r = \lfloor s / 2 \rfloor$ . Independently choose a uniform maximum matching M of their indices, leaving one uniformly chosen index unmatched when s is odd. Define its average paired score by

$$
A _ { \mathcal { M } } ^ { \pm } = \frac { 1 } { r } \sum _ { \{ i , j \} \in \mathcal { M } } \left( \frac { X _ { i } + X _ { j } } { 2 } \pm \frac { \kappa } { 2 } ( X _ { i } - X _ { j } ) ^ { 2 } \right) .
$$

An index is included with probability $2 r / s$ and an unordered pair is matched with probability $2 r / [ s ( s - 1 ) ]$ . Since $\begin{array} { r } { \sum _ { i < j } ( X _ { i } - X _ { j } ) ^ { 2 } = { \overset { \cdot } { s } } \sum _ { i } ( X _ { i } - { \overline { { X } } } _ { s } ) ^ { 2 } } \end{array}$ ，

$$
\mathbb { E } _ { \mathcal { M } } [ A _ { \mathcal { M } } ^ { \pm } \ | \ X _ { 1 } , \dots , X _ { s } ] = \overline { { X } } _ { s } \pm \kappa \widehat { v } _ { s } .\tag{D.3}
$$

For a fixed matching the r paired scores are i.i.d. Each error in (4.5) is a convex positive-part function of one score. Conditional Jensen therefore bounds $\mathbb { E } e _ { s }$ by the expected paired-prefix error at length r. Applying (D.2) with $c _ { \mathrm { v a r } } = c _ { \mathrm { v a r , } \kappa }$ and $\eta = \kappa$ , each r occurring twice as s ranges over $2 , 3 , . . . ,$ gives

$$
\sum _ { s \ge 2 } \mathbb { E } e _ { s } \le 2 \left( \frac { c _ { \mathrm { v a r } , \kappa } } { 2 \kappa } + \frac { \kappa v } { 2 } \right) \le \frac { c _ { \mathrm { v a r } , \kappa } } { \kappa } + \frac { \kappa } { 4 } = \frac { 1 1 3 } { 1 6 } .\tag{D.4}
$$

The representation also proves that both all-sample scores lie in [0, 1]. The matching is solely an analysis device.

## D.3. Proof of the prefix-charging lemma

Fix an arm, an error type, and a returned-sample prefix of length $s \geq 2$ , and let $e _ { s } \geq 0$ be its error on the full arm array. Let $I _ { b , s }$ indicate that this prefix is current at the pre-advancement history $\mathcal { H } _ { b }$ . As in Lemma 9, let $D _ { b }$ indicate advancement of the arm’s returned-sample stream and $h _ { b } = \mathbb { E } [ D _ { b } \mid \mathcal { H } _ { b } ]$ . Although an unreached prefix need not be observed, $I _ { b , s } e _ { s }$ is H<sub>b</sub>-measurable. Thus

$$
\mathbb { E } \sum _ { b } I _ { b , s } \lambda _ { b } e _ { s } \leq H \mathbb { E } \sum _ { b } I _ { b , s } h _ { b } e _ { s } = H \mathbb { E } \sum _ { b } I _ { b , s } D _ { b } e _ { s } \leq H \mathbb { E } e _ { s } .
$$

The last inequality uses $\begin{array} { r } { \sum _ { b } I _ { b , s } D _ { b } \leq 1 \colon } \end{array}$ a returned-sample prefix advances at most once. Summing over $s \geq 2$ by Tonelli and applying (D.4) proves Lemma 9. A pre-advancement measurable inclusion indicator multiplies both charge and advancement and preserves the conditional identity. No independence across diferent prefix lengths is required.

## Appendix E. Bank certificates and the five charging ratios

## E.1. The bank certificate

Write an update as $B ^ { \prime } = B - \alpha x + ( 1 - I ) G / 3 - D$ , where $D \geq 0$ is discarded credit at the cap. Starting at zero and summing gives

$$
\sum _ { b = 1 } ^ { N } \alpha _ { b } x _ { b } = - B _ { N + 1 } + \frac { 1 } { 3 } \sum _ { b = 1 } ^ { N } ( 1 - I _ { b } ) G _ { b } - \sum _ { b = 1 } ^ { N } D _ { b } \leq \frac { 1 } { 3 } \sum _ { b = 1 } ^ { N } ( 1 - I _ { b } ) G _ { b } .
$$

For the one-step certificate, condition throughout on $\mathcal { F } ^ { - } , \mathrm { ~ H ~ } \alpha = 1 / 3$ , then $\Delta B \ge$ −αx, so

$$
\begin{array} { r } { \mathbb { E } [ I e _ { \mathrm { t } } + \alpha e _ { \mathrm { a } } + \Delta B ] \ge ( e _ { \mathrm { a } } + e _ { \mathrm { t } } - x ) / 3 \ge Q / 6 . } \end{array}
$$

Otherwise $x > 0 , \ B = \alpha x$ , and the credit $G / 3 \ \leq \ 1 / 3$ is untruncated. Writing $D _ { * } ~ =$ $\mathbb { E } [ I e _ { \mathrm { t } } + \alpha e _ { \mathrm { a } } + \Delta B ]$

$$
D _ { * } \geq \frac { \alpha } { 2 } Q + \frac { 1 - \alpha } { 3 } \gamma , \qquad D _ { * } - \frac { Q } { 6 } \geq \frac { 3 \alpha - 1 } { 6 } Q + \frac { 1 - \alpha } { 3 } \gamma \geq \frac { 2 \alpha } { 3 } \gamma \geq 0 .
$$

The final comparison uses $Q \leq 2 \gamma$ and 3α $- 1 \leq 0$ , and therefore holds for either sign of $Q .$

## E.2. Paying for nonoptimal targets

We prove the score conversion (4.9) and its cumulative payment. Suppress the block index. If $c = ( \mu _ { a } - \mu _ { j } ) _ { + } > 0$ , then the score errors imply

$$
c - x _ { j } \leq L _ { a } ^ { \downarrow } + U _ { j } ^ { \uparrow } + \frac { v _ { a } + v _ { j } } { 6 } .
$$

Lemma 5 gives $v _ { a } , v _ { j } \leq \Gamma + c$ , where $\Gamma = \gamma _ { a j }$ . Rearranging proves (4.9); the inequality is also immediate for $c = 0$ . Nonoptimal targets form fixed bank coordinates, so the debit telescope can be summed over exactly those coordinates:

$$
\mathbb { E } \sum _ { b : \Delta _ { j _ { b } } > 0 } \frac { 3 } { 2 } \alpha _ { b } x _ { j _ { b } } \le \frac { 1 } { 2 } \mathbb { E } \sum _ { b : \Delta _ { j _ { b } } > 0 } ( 1 - \alpha _ { b } ) \Gamma _ { b } .
$$

Together with the $\alpha _ { b } \Gamma _ { b } / 2$ term, this uses half of their total gain. Optimal targets have zero mean switching cost and retain their entire gain. Summing (4.8) therefore gives (4.10). The same argument holds after multiplying by indicators of a predictably included block prefix, because every selected bank is still telescoped over its actual prefix of updates.

## E.3. The charging ratios and terminal balances

Table 2 gives the ratios for the switching errors, witness, and residual anchor directly from their weights and advancement probabilities. The optimal-stream ratio requires target averaging in (4.11). Conditional on $\mathcal { G } _ { b } .$ , if $q \neq a$ has role z, it is selected with probability $w _ { z }$ and, conditional on selection, switched to with probability $\alpha _ { q } ^ { z }$ . Its error weight is $6 \alpha _ { q } ^ { z } ,$ and its advancement probability is $w _ { z } \alpha _ { q } ^ { z }$ , giving ratio at most $6 / \omega \stackrel { \cdot } { = } 1 8 ( m - 2 )$ . Every such charge belongs to the same fixed arm $q ,$ so this inverse-exploration factor occurs only once.

When $q = a$ , neither target bank of $q$ is updated. Otherwise each bank changes only when $q$ is selected in that role. Including every such update, their fixed multipliers give

$$
\sum _ { b } \frac { 6 } { p } \Delta B _ { q , b } ^ { \mathrm { R } } + \sum _ { b } \frac { 6 } { \omega } \Delta B _ { q , b } ^ { \mathrm { S } } = \frac { 6 } { p } B _ { q , N + 1 } ^ { \mathrm { R } } + \frac { 6 } { \omega } B _ { q , N + 1 } ^ { \mathrm { S } } \leq \frac { 2 } { p } + \frac { 2 } { \omega } = 6 m - 9 .
$$

The initial balances are zero. Each role keeps its fixed multiplier even as $q$ changes roles, and every update enters the corresponding telescope, including blocks with negative $Q$ . The five error-family coeficients sum to

$$
{ \frac { 3 } { 4 } } m + { \frac { 3 } { 2 } } ( m - 1 ) + 9 ( m - 1 ) + 3 ( m - 1 ) + 1 8 ( m - 2 ) = { \frac { 1 2 9 m - 1 9 8 } { 4 } } .
$$

Initialization and the terminal balances contribute $2 ( m - 1 ) + 6 m - 9 = 8 m - 1 1$ . Hence

$$
R _ { T } \leq \frac { 1 1 3 } { 1 6 } \frac { 1 2 9 m - 1 9 8 } { 4 } + 8 m - 1 1 = \frac { 1 5 0 8 9 m - 2 3 0 7 8 } { 6 4 } < 2 3 6 m - 3 6 0 ,
$$

proving Theorem 6 for every $m \geq 3 .$ , without requiring a unique optimal arm.

## Appendix F. Budget lifting and activation

## F.1. Sparse-budget ceiling efects

Assume $3 k < 2 n$ , set $s = \lfloor k / 2 \rfloor$ , and let $m = \lceil n / s \rceil \geq 4$ . Put $d = n - k$ . If $m = 4 .$ then $k / d < 2$ , so Theorem 6 gives regret below $2 ( 2 3 6 \cdot 4 - 3 6 0 ) d / k = 1 1 6 8 d / k$ . If $m \geq 5$ , then $n \geq ( m - 1 ) s + 1 , k \leq 2 s + 1$ , and $k \leq 3 s$ . Hence $d \geq ( m - 3 ) \ O { : }$ s and

$$
{ \frac { ( 2 3 6 m - 3 6 0 ) k } { d } } \leq { \frac { 3 ( 2 3 6 m - 3 6 0 ) } { m - 3 } } = 7 0 8 + { \frac { 1 0 4 4 } { m - 3 } } \leq 1 2 3 0 .
$$

For the fixed-array policy, use 32m in place of $2 3 6 m - 3 6 0$ . If $m = 4$ , its regret is below $2 5 6 d / k$ because $k / d < 2 $ . If $m \geq 5$ , the same ceiling inequalities yield

$$
{ \frac { 3 2 m k } { d } } \leq { \frac { 9 6 m } { m - 3 } } \leq 2 4 0 < 2 5 6 .
$$

Together these give both sparse bounds in (5.1).

## F.2. Dense activation at an interrupted horizon

Fix the partition. For each reached attempt $s ,$ couple winner decomposition with a potential gain $G _ { s }$ and a full additional singleton count $A _ { s }$ , independent of the pre-reward thinning coin $\xi _ { s }$ . When $\xi _ { s } = 0$ , these variables remain unobserved. Let $U _ { s }$ be an independent uniform variable on $[ 0 , 1 ]$ and define the potential trigger

$$
F _ { s } = \{ \xi _ { s } = 1 , ~ U _ { s } \leq G _ { s } \} .
$$

Conditional on the pre-attempt history, $\mathbb { E } G _ { s } = \gamma , \mathbb { E } A _ { s } \leq 1$ , and $\mathbb { P } ( F _ { s } ) = \gamma / L$ . The algorithm performs additional probes only when $\xi _ { s } = 1$

Let $M _ { T }$ count the gate’s recovery group-singleton probes actually performed through $T ,$ and let $\mathsf { A c t } _ { T }$ be the event that the continuation performs its first physical probe by T. Let $\chi _ { s }$ indicate that the initial probe of attempt s occurs by $T .$ Then $\begin{array} { r } { N _ { T } = \sum _ { s } \chi _ { s } \leq T } \end{array}$ , and $\chi _ { s }$ is determined before that attempt’s target, coin, and rewards. At most one potential trigger occurs among reached attempts. If the last attempt finishes after $T ,$ its trigger is included in this count, but it cannot be followed by another reached attempt. Thus $\begin{array} { r } { ( \gamma / L ) \mathbb { E } N _ { T } = \mathbb { E } \sum _ { s } \chi _ { s } \mathbf { 1 } \{ F _ { s } \} \leq 1 } \end{array}$ , and $\mathsf { A c t } _ { T }$ requires such a trigger. Actual recovery probes satisfy $\begin{array} { r } { \mathbb E M _ { T } \le \mathbb E \sum _ { s } \chi _ { s } \xi _ { s } A _ { s } \le \mathbb E N _ { T } / L } \end{array}$ . Together,

$$
\gamma \mathbb { E } N _ { T } \le L , \qquad L \mathbb { P } ( \mathsf { A c t } _ { T } ) \le \gamma \mathbb { E } N _ { T } , \qquad \mathbb { E } M _ { T } \le \mathbb { E } N _ { T } / L .\tag{F.1}
$$

Every initial gate probe is selected before fresh rewards, so its total expected cost is $( c _ { 0 } -$ $\gamma ) \mathbb { E } N _ { T }$ . Likewise, the actual singletons cost $c _ { 0 } \mathbb { E } M _ { T }$ , with either sign of $c _ { 0 }$ . Conditional on the continuation’s first probe time and the preceding history, its initialization uses fresh independent rewards. Its uniform guarantee applies to the remaining deterministic horizon, for a total cost at most $L \mathbb { P } ( \mathsf { A c t } _ { T } )$ . Summing gives

$$
R _ { T } \leq ( c _ { 0 } - \gamma ) \mathbb { E } N _ { T } + c _ { 0 } \mathbb { E } M _ { T } + L \mathbb { P } ( \mathsf { A c t } _ { T } ) \leq \left( c _ { 0 } + \frac { ( c _ { 0 } ) _ { + } } { L } \right) \mathbb { E } N _ { T } .\tag{F.2}
$$

Only the trigger accounting, not the physical regret, uses the completed attempt.

## Appendix G. The complete BestProbe budget law

We prove Theorem 2. Throughout, $n > k \geq 2 , d = n - k$ , and $\delta = d / n$ . Upper bounds hold for each comparator fixed before policy randomization. Lower-bound instances are joint reward laws sampled i.i.d. across rounds; their parameters may depend on the minimax horizon.

## G.1. Winner-only upper bounds

We prove (6.1) for every fixed comparator $j$ and reward array. Both policies below observe only $J _ { t } ;$ their bounds also control the expected sum of positive instantaneous regrets.

## G.1.1. Sparse budgets: a one-coordinate Tsallis update

Use the signed mirror update and regularizer of Section 3; Appendix B proves the local inequality and well-definedness. Initialize $p _ { 1 , i } = 1 / n$ . At each round put

$$
Z _ { t } = \sum _ { i } { \sqrt { p _ { t , i } } } , \qquad q _ { t , i } = { \frac { 1 } { 2 } } p _ { t , i } + { \frac { \sqrt { p _ { t , i } } } { 2 Z _ { t } } } , \qquad \eta = { \frac { 1 } { 4 { \sqrt { n } } } } .
$$

Draw k labels independently from $q _ { t }$ , probe their distinct labels, and update with

$$
g _ { t , i } = - { \frac { \mathbf { 1 } \{ J _ { t } = i \} } { q _ { t , i } } } .\tag{G.1}
$$

Repeated draws use only one physical probe. No numerical reward is used.

Condition on the pre-round history and suppress t. Let $r _ { i } = \mathbb { P } ( J = i )$ and $a _ { i } = r _ { i } / q _ { i }$ . For a fixed comparator $j ,$ write

$$
u _ { j } = \sum _ { i : x _ { i } < x _ { j } } q _ { i } .
$$

Positive regret requires every draw to have reward below $x _ { j }$ , so

$$
\mathbb { E } ( x _ { j } - W ) _ { + } \leq u _ { j } ^ { k } .
$$

The disjoint events in which exactly one draw is $j$ and all other draws are strictly worse give

$$
r _ { j } \geq k q _ { j } u _ { j } ^ { k - 1 } , \qquad \mathbb { E } ( x _ { j } - W ) _ { + } \leq \frac { a _ { j } } { k } .
$$

These inequalities accommodate every fixed tie-breaking order.

The mixed sampling distribution gives

$$
\langle p , a \rangle = \sum _ { i } \frac { p _ { i } r _ { i } } { q _ { i } } \leq 2 , \qquad \mathbb { E } \sum _ { i } p _ { i } ^ { 3 / 2 } g _ { i } ^ { 2 } = \sum _ { i } \frac { p _ { i } ^ { 3 / 2 } r _ { i } } { q _ { i } ^ { 2 } } \leq 4 Z \leq 4 \sqrt { n } .
$$

The second inequality uses $q _ { i } ^ { 2 } \ge p _ { i } ^ { 3 / 2 } / ( 4 Z )$ . Also $\eta \sqrt { p _ { i } } | g _ { i } | \le 2 \eta Z \le 1 / 2$ , so (B.1) applies. Since $\mathbb { E } g = - a$ , telescoping against $e _ { j }$ yields

$$
\mathbb { E } \sum _ { t < T } a _ { j , t } \le 2 T + \frac { 2 ( \sqrt { n } - 1 ) } { \eta } + 8 \eta \sqrt { n } T = 8 ( n - \sqrt { n } ) + 4 T .
$$

Consequently, for every fixed array,

$$
\mathbb { E } \sum _ { t \leq T } ( x _ { j , t } - W _ { t } ) _ { + } \leq { \frac { 8 ( n - { \sqrt { n } } ) + 4 T } { k } } .\tag{G.2}
$$

For $k < n / 2$ , combine this with $R _ { T } ( j ) \leq T$ and $\delta > 1 / 2$ to obtain (6.1).

## G.1.2. Dense budgets: protect the winners of each batch

Assume $k \geq n / 2$ . Use batches of $H = \lfloor k / 2 \rfloor$ rounds. At the start of a batch let $P = \varnothing$ . In each round, sample a uniform $\left( k - | P | \right)$ -subset of $[ n ] \setminus P$ , probe its union with $P _ { \mathrm { : } }$ and add the observed winning label to $P$ . Reset $P$ only at the next batch.

Fix an unprotected comparator $j$ . Set $\begin{array} { r } { b = | P | , N = n - b , } \end{array}$ and $s = k - b ;$ then $N - s = d .$ If $P \neq \emptyset$ and its maximum reward is at least $x _ { j } .$ , positive regret is zero. Otherwise, let $a$ be the number of unprotected labels strictly below $x _ { j }$ . Conditional on the history,

$$
\mathbb { P } ( W < x _ { j } ) = \frac { { \binom { a } { s } } } { { \binom { N } { s } } } , \qquad \mathbb { P } ( J = j ) \geq \frac { { \binom { a } { s - 1 } } } { { \binom { N } { s } } } .
$$

Using the convention ${ \binom { a } { b } } = 0$ when $b > a .$ , these imply

$$
\mathbb { E } ( x _ { j } - W ) _ { + } \leq \frac { d } { s } \mathbb { P } ( J = j ) \leq \frac { 2 d } { k } \mathbb { P } ( J = j ) .\tag{G.3}
$$

Indeed, when $a \geq s$ , the binomial ratio is $( a - s + 1 ) / s \leq d /$ s because $a \leq N - 1$ ; otherwise its numerator is zero. Here $b \leq H - 1$ , hence $s \ge k - H + 1 \ge k / 2$ . After the first win by j in a batch, it is protected and cannot incur positive regret. Therefore each full or partial batch costs at most $2 d / k$ , giving

$$
R _ { T } ( j ) \leq \frac { 2 d } { k } \left\lceil \frac { T } { H } \right\rceil \leq \frac { 2 d } { k } + \frac { 6 d T } { k ^ { 2 } } .
$$

We used $H \geq k / 3$ , also for $k = 2 , 3$ . Furthermore, an unprotected comparator is omitted with probability $d / ( n - b ) \leq 2 \delta$ , so $R _ { T } ( j ) \leq 2 \delta T$ . Since $n / k \le 2$ , these bounds imply (6.1).

## G.2. Numerical feedback and an anytime combination

Tsallis-INF with symmetric regularization, $\alpha = 1 / 2$ , importance-weighted loss estimates, and learning rate $2 / \sqrt { t }$ has regret at most

$$
B _ { m } ( T ) = 4 \sqrt { m T } + 1
$$

on any fixed reward array with m actions (Zimmert and Seldin, 2021, Theorem 1, IW estimators). Explicitly, with cumulative estimated losses $\widehat { L } _ { 0 } = 0$ , its round-t distribution is

$$
w _ { t } = \arg \operatorname* { m i n } _ { w \in \Delta _ { m } } \left\{ \langle w , \widehat { L } _ { t - 1 } \rangle - 2 \sqrt { t } \sum _ { i } \sqrt { w _ { i } } \right\} .
$$

Draw $I _ { t } \sim w _ { t }$ and add $( 1 - W _ { t } ) \mathbf { 1 } \{ I _ { t } = i \} / w _ { t , i }$ to coordinate i of $\widehat { L } _ { t - 1 }$ . The local round count restarts when this policy starts; no horizon is required.

For $k < n / 2$ , partition [n] into $m = \lceil n / k \rceil$ groups of size at most k. Treat each group maximum as an action reward and run this bandit policy. The group containing a fixed comparator dominates it in every round. Since $m \leq 3 n / ( 2 k )$ and $\delta > 1 / 2$ , the regret is at most $1 6 \delta \sqrt { n T / k }$

For $k \geq n / 2$ , choose once, independently of the array, a uniform baseline B of size $n - 2 d$ and split the other labels into $G _ { 1 } , G _ { 2 }$ , each of size d. The baseline may be empty when $k = n / 2$ . Conditional on the partition, run a two-action bandit on $B \cup G _ { 1 }$ and $B \cup G _ { 2 }$ . If the comparator lies in B, regret is nonpositive; otherwise one action dominates it. Its probability of missing B is 2δ. Thus the expected regret is at most $2 \delta B _ { 2 } ( T ) \le 1 6 \delta \sqrt { n T / k }$ . These two policies use only $W _ { t }$

For one anytime policy, use the appropriate winner-only policy through round $t _ { 0 } = n k$ then start the appropriate numerical policy with fresh randomization. Put

$$
G ( T ) = \operatorname * { m i n } \{ T , ( n + T ) / k \} , \qquad S ( T ) = \sqrt { n T / k } .
$$

For $T \leq n k$ , one has $\begin{array} { r } { G ( T ) \leq \frac { 3 } { 2 } \operatorname* { m i n } \{ G ( T ) , S ( T ) \} } \end{array}$ . To see this, the claim is immediate for $T \leq n / k ;$ otherwise

$$
\frac { ( n + T ) / k } { S ( T ) } = \sqrt { \frac { n } { k T } } + \sqrt { \frac { T } { n k } } \leq 1 + \frac { 1 } { k } \leq \frac { 3 } { 2 } .
$$

The last inequality follows by setting $z = { \sqrt { T / ( n k ) } } \in [ 1 / k , 1 ]$ : the convex function $z + 1 / ( k z )$ is at most its endpoint value $1 + 1 / k$ . The winner-only guarantee is therefore at most $2 4 \mathcal { R } _ { n , k } ( T )$ . Its cost at $t _ { 0 }$ is at most 24δn. For $T > n k$ , the total cost is at most

$$
2 4 \delta n + 1 6 \delta \sqrt { \frac { n ( T - n k ) } { k } } \leq 4 0 \delta \sqrt { \frac { n T } { k } } = 4 0 \mathcal { R } _ { n , k } ( T ) .
$$

A comparator maximizing the full fixed array is fixed before both phases’ randomization, so this addition is valid. At $T = n k$ , only the winner-only phase is used. This proves the upper bound of Theorem 2. For a joint i.i.d. law, average the fixed-array guarantee and use $T \operatorname* { m a x } _ { j } \mu _ { j } \le \mathbb { E } \operatorname* { m a x } _ { j } \sum _ { t } X _ { j , t }$ to obtain the pseudo-regret upper bound.

## G.3. A matching lower bound for every adaptive action size

We prove

$$
{ \mathcal V } _ { n , T } ^ { \mathrm { B e s t P r o b e } , \mathsf { M } , ( k ) } \ge \frac { \delta } { 5 1 2 } \operatorname* { m i n } \left\{ \frac { T } { k } , \sqrt { \frac { n T } { k } } \right\} .\tag{G.4}
$$

Here and below $\mathsf { M } \in \{ \mathrm { j o i n t } , \mathrm { s e q } \}$ . The proof permits arbitrary adaptive singleton and smaller-set actions.

## G.3.1. A rank-preserving correlated threshold family

For the family (6.2), take $0 < \varepsilon \le 1 / 6$ and $0 < \tau \leq 1$ . The vectors are i.i.d. across rounds and rewards lie in (0, 1). The transformation preserves the strict order of the uniforms. Every selected winner label is uniform on the selected set.

For independent standard normals $Z , Z ^ { \prime }$ , let $h ( s ) = \mathbb { E } F _ { N } ( Z + s ) = \mathbb { P } ( Z ^ { \prime } - Z \le s ) =$ $F _ { N } ( s / \sqrt { 2 } )$ . Its derivative is $h ^ { \prime } ( s ) = e ^ { - s ^ { 2 } / 4 } / ( 2 \sqrt { \pi } )$ , so on $[ 0 , 1 ] , 1 / 5 \leq h ^ { \prime } ( s ) \leq 1 / 3$ . If an s-set $S$ omits $q ,$ the event $U _ { q } > \operatorname* { m a x } _ { i \in S } U _ { i }$ has probability $1 / ( s + 1 )$ . On this event the latent advantage of $q$ is at least $1 - \varepsilon ;$ on its complement the surplus over q is at most ε. When $q$ is selected, that same surplus bound holds. Therefore every action satisfies

$$
r _ { q } ( S ) : = \mu _ { q } - \mathbb { E } _ { q } \operatorname* { m a x } _ { i \in S } X _ { i } \geq { \frac { \tau } { 6 ( s + 1 ) } } \mathbf { 1 } \{ q \not \in S \} - { \frac { \tau \varepsilon } { 3 } } .\tag{G.5}
$$

Applying this bound to singletons gives $\mu _ { q } - \mu _ { i } \geq \tau ( 1 / 1 2 - \varepsilon / 3 ) \geq \tau / 3 6 > 0$ for $i \neq q$ , so q is uniquely mean-optimal.

## G.3.2. The two observation kernels

Let $R = \operatorname* { m a x } _ { i \in S } U _ { i } \sim \mathrm { B e t a } ( s , 1 )$ . If $q \in S$ , conditional on $R = r$

$$
F _ { N } ^ { - 1 } ( W ) \sim \mathsf { H } ( r ) : = \mathcal { N } \big ( \tau ( 1 - \varepsilon + \varepsilon r ) , 1 \big ) .
$$

If $q \not \in S$ , its conditional law is

$$
M ( r ) = r \mathsf { H } ( r ) + ( 1 - r ) \mathsf { L } ( r ) , \qquad \mathsf { L } ( r ) : = \mathcal { N } ( \tau \varepsilon r , 1 ) .
$$

In both cases the winning label is independent of the numerical observation and uniform on S. Let $\mathsf { H } _ { s } , M _ { s }$ denote the numerical laws after integrating over R.

We claim

$$
\operatorname* { m a x } \{ \mathrm { K L } ( \mathsf { H } _ { s } , M _ { s } ) , \mathrm { K L } ( M _ { s } , \mathsf { H } _ { s } ) \} \leq \frac { 1 2 \tau ^ { 2 } } { ( s + 1 ) ^ { 2 } } , \qquad s \geq 1 .\tag{G.6}
$$

For equal-variance normal densities with mean diference $a = \tau ( 1 - \varepsilon )$ , their chi-square divergence is $e ^ { a ^ { 2 } } - 1 \leq 2 \tau ^ { 2 }$ . Since $M ( r ) \geq r \mathsf { H } ( r )$ , for $s \geq 2$

$$
\mathrm { K L } ( M ( r ) , \mathsf { H } ( r ) ) \leq 2 \tau ^ { 2 } ( 1 - r ) ^ { 2 } , \qquad \mathrm { K L } ( \mathsf { H } ( r ) , M ( r ) ) \leq 2 \tau ^ { 2 } { \frac { ( 1 - r ) ^ { 2 } } { r } } .
$$

Integrating uses

$$
\mathbb { E } ( 1 - R ) ^ { 2 } = \frac { 2 } { ( s + 1 ) ( s + 2 ) } , \qquad \mathbb { E } \frac { ( 1 - R ) ^ { 2 } } { R } = \frac { 2 } { ( s - 1 ) ( s + 1 ) } .
$$

Marginalization cannot increase $\mathrm { K L }$ , and $( s + 1 ) / ( s - 1 ) \leq 3$ . For $s = 1$ , mixture convexity in either direction bounds the conditional divergence by $( 1 - r ) a ^ { 2 } / 2$ , whose expectation is $a ^ { 2 } / 4$ . This proves (G.6). The CDF transformation preserves divergence.

## G.3.3. A stopped omission cost

Fix any policy, including one that knows the family and horizon. Define a reference experiment $Q ,$ , independent of the hidden label: for an action of size $s ,$ return kernel $M _ { s }$ when $s \leq n / 2$ and $\mathsf { H } _ { s }$ otherwise, together with an independent uniform selected label. This is a comparison law on transcripts; it need not be an admissible reward environment.

Give q the uniform prior, independently of the reference experiment. Let $D _ { q }$ be the hidden-$q$ reward law and $P _ { q }$ its transcript law. Under $Q ,$ the reference transcript is independent of $q .$ Let

$$
a _ { t } = \frac { 1 } { | S _ { t } | + 1 } , \qquad B _ { 0 } = \frac { T } { k + 1 } , \qquad L = B _ { 0 } + \frac { 1 } { 2 } .
$$

Stop at the first round $\sigma$ for which $\textstyle \sum _ { t < \sigma } a _ { t } \geq B _ { 0 }$ . Because $a _ { t } \geq 1 / ( k + 1 ) , \sigma \leq T$ . Because $a _ { t } \leq 1 / 2$ , the stopped sum is at most L. Let $\bar { Q } ^ { \sigma }$ and $\bar { P } ^ { \sigma }$ be the joint laws of $q$ and the stopped transcript, including actions and observations through $\sigma _ { \mathrm { { ; } } }$ , in the reference and true experiments, respectively. Define

$$
C _ { q } = \sum _ { t \leq \sigma } { \frac { { \bf 1 } \{ q \notin S _ { t } \} } { | S _ { t } | + 1 } } , \qquad 0 \leq C _ { q } \leq L .
$$

Independence under the reference experiment gives

$$
\mathbb { E } _ { \bar { Q } ^ { \sigma } } C _ { q } = \mathbb { E } _ { Q } \sum _ { t \leq \sigma } \frac { n - | S _ { t } | } { n ( | S _ { t } | + 1 ) } \geq \delta B _ { 0 } .\tag{G.7}
$$

The reference kernel difers from the true one for exactly min $\{ s , n - s \}$ of the n hidden labels. The stopped adaptive KL chain rule, followed by (G.6), gives

$$
\begin{array} { r l } { \displaystyle \mathrm { K L } ( \bar { Q } ^ { \sigma } , \bar { P } ^ { \sigma } ) \leq 1 2 \tau ^ { 2 } \mathbb { E } _ { Q } \sum _ { t \leq \sigma } \frac { \operatorname* { m i n } \{ | S _ { t } | , n - | S _ { t } | \} } { n ( | S _ { t } | + 1 ) ^ { 2 } } } & { } \\ { \displaystyle } & { \leq \frac { 2 4 \tau ^ { 2 } } { n } \mathbb { E } _ { \bar { Q } ^ { \sigma } } C _ { q } . } \end{array}\tag{G.8}
$$

The last inequality follows, for each $1 \leq s < n$ , from

$$
{ \frac { \operatorname* { m i n } \{ s , n - s \} } { ( s + 1 ) ( n - s ) } } \leq { \frac { 2 } { n } } .
$$

All stopping inclusion indicators are known before the current observation, so the chain rule applies to these stopped transcripts.

For completeness, if $0 \leq C \leq L , K = \mathrm { K L } ( Q , P ) , p = \mathbb { E } _ { Q } C / L ,$ , and $u = \mathbb { E } _ { P } C / L ,$ randomizing $C / L$ into a Bernoulli variable gives $\operatorname { k l } ( p , u ) \leq K$ . If $u \leq p$ , then

$$
\operatorname { k l } ( p , u ) = \int _ { u } ^ { p } { \frac { p - v } { v ( 1 - v ) } } d v \geq { \frac { ( p - u ) ^ { 2 } } { 2 p } } .
$$

If $K \leq a \mathbb { E } _ { Q } C$ , it follows in either case that

$$
\mathbb { E } _ { P } C \geq ( 1 - \sqrt { 2 a L } ) \mathbb { E } _ { Q } C .\tag{G.9}
$$

The case $\mathbb { E } _ { Q } C = 0$ is immediate. This relative expectation bound avoids losing a second factor of δ.

Choose

$$
\tau = \operatorname * { m i n } \left\{ 1 , \sqrt { \frac { n } { 1 9 2 L } } \right\} , \qquad \varepsilon = \frac { \delta } { 1 0 0 ( k + 1 ) } .
$$

Equations $( \mathrm { G } . 8 ) \mathrm { - } ( \mathrm { G } . 9 )$ imply $\begin{array} { r } { \mathbb { E } _ { \bar { P } ^ { \sigma } } C _ { q } \ge \frac { 1 } { 2 } \mathbb { E } _ { \bar { Q } ^ { \sigma } } C _ { q } \ge \delta B _ { 0 } / 2 } \end{array}$ . The expectation of $C _ { q }$ under the stopped law equals its expectation in the full experiment. Retain the positive omission cost through σ, but charge possible negative regret in all T rounds. Equation (G.5) then gives

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { n } \sum _ { q } R _ { T } ^ { \pi } ( D _ { q } ) \geq \frac { \tau } { 6 } \mathbb { E } _ { { \bar { P } } ^ { \sigma } } C _ { q } - \frac { \tau \varepsilon T } { 3 } } \\ & { \qquad \quad \geq \left( \displaystyle \frac { 1 } { 1 2 } - \frac { 1 } { 3 0 0 } \right) \tau \delta B _ { 0 } = \frac { 2 } { 2 5 } \tau \delta B _ { 0 } . } \end{array}
$$

To check constants uniformly, if $B _ { 0 } \geq 1 / 2$ , then $L \leq 2 B _ { 0 }$ and

$$
\tau B _ { 0 } \geq \frac { 1 } { 2 0 } \operatorname* { m i n } \{ B _ { 0 } , \sqrt { n B _ { 0 } } \} .
$$

If $B _ { 0 } < 1 / 2$ , then $L \ : < \ : 1$ and $n \geq 3$ give $\tau \geq 1 / 8$ , so the same bound holds. Finally $B _ { 0 } \geq 2 T / ( 3 k )$ , proving (G.4).

For the fixed-array transfer, average the arrays drawn from this family. Their hindsight comparator satisfies

$$
\mathbb { E } \operatorname* { m a x } _ { j } \sum _ { t } X _ { j , t } \geq \mathbb { E } \sum _ { t } X _ { q , t } = T \mu _ { q } .
$$

Thus the expected fixed-array regret is at least the prior-average pseudo-regret. For every policy, some fixed array attains this lower bound.

## G.4. Matching the costs and the labels-only law

The common coverage bound (2.5), proved in Appendix A, applies in both models and gives

$$
\mathcal { V } _ { n , T } ^ { \mathsf { B e s t P r o b e } , \mathsf { M } , ( k ) } \geq \frac { \delta } { 2 } \operatorname* { m i n } \{ T , n / k \} .
$$

Put $A = \operatorname* { m i n } \{ T , n / k \} , B = \operatorname* { m i n } \{ T / k , \sqrt { n T / k } \}$ , and $F = \mathcal { R } _ { n , k } ( T ) / \delta$ . Each of A and B is at most F. Splitting at $n / k$ and nk also gives $F \leq A + B$ , hence

$$
F \leq A + B \leq 2 F .
$$

Since $\delta A = \Phi _ { n , k } ( T )$ and $\delta B = ( \delta / k ) \operatorname* { m i n } \{ T , { \sqrt { n k T } } \}$ , this proves the factor-two equivalence in (2.4). Combining (G.4) and (2.5),

$$
\operatorname* { m a x } \left\{ \frac { \delta } { 2 } A , \frac { \delta } { 5 1 2 } B \right\} \geq \frac { \delta } { 1 0 2 4 } ( A + B ) \geq \frac { 1 } { 1 0 2 4 } \mathcal { R } _ { n , k } ( T ) .
$$

Together with Appendix G.2, this proves Theorem 2.

Proof of Corollary 11. The upper bound is (6.1). For the lower bound, use (6.2) with $\tau = 1$ and $\varepsilon = \delta / [ 1 0 0 ( k + 1 ) ]$ ]. Every label kernel is independent of $q ,$ so the entire visible transcript is independent of the uniform hidden label. Equation (G.5) gives

$$
\frac { 1 } { n } \sum _ { q } R _ { T } ^ { \pi } ( D _ { q } ) \geq \frac { 4 9 \delta T } { 3 0 0 ( k + 1 ) } \geq \frac { \delta T } { 1 0 k } .
$$

Combining with (2.5) gives a lower bound of δ min $\{ T , ( n { + } T ) / k \} / 2 0$ : use min $\left\{ T , ( n { + } T ) / k \right\} \leq$ min $\{ T , n / k \} + T / k$ . The reverse comparison $\Phi _ { n , k } ( T ) + \delta T / k \le 2 \delta \operatorname* { m i n } \{ T , ( n + T ) / k \}$ proves the equivalent expression for the rate. Transfer to fixed arrays as above. ■

## Appendix H. Partial numerical feedback under independence

## H.1. The cost of numerical revelation

Assume product rewards and put $b _ { k } = \lfloor ( k + 1 ) ^ { 2 } / 4 \rfloor$ . Channel $\mathsf { F } _ { \rho }$ always reveals $J _ { t }$ and reveals $W _ { t }$ when an independent $C _ { t } \sim$ Bernoulli(ρ) equals one. The coins are i.i.d. and independent of rewards and policy randomization; the same rule applies to singletons. The payof and comparator do not change. Let $\Phi _ { n , k , \rho } ( T ) = \operatorname* { m i n } \{ ( n - k ) T / n , ( n - k ) / ( k \rho ) \}$ , with the second term infinite at $\rho = 0$

Theorem 12 (Cost of numerical feedback) For $n > k \geq 2 , T \geq 1$ , and $\rho \in [ 0 , 1 ]$

$$
\frac { 1 } { 1 6 b _ { k } } \Phi _ { n , k , \rho } ( T ) \leq \mathcal { V } _ { n , T } ^ { \mathsf { F } _ { \rho } , ( k ) } \leq 1 4 0 0 \Phi _ { n , k , \rho } ( T ) .\tag{H.1}
$$

One anytime policy, knowing neither T nor $\rho ,$ attains the upper bound. At $\rho = 0$ , one fixed family gives the lower bound in prior average at every horizon.

Hold each action of Theorem 1’s BestProbe policy $\pi$ until its value is revealed, ignoring intervening labels (Dai et al., 2024, Algorithm 2). The revelation count $N _ { T } \sim$ Binomial $( T , \rho )$ is independent of the complete virtual trajectory, giving $\rho R _ { T } ^ { \mathrm { h o l d } } = \mathbb { E } R _ { N _ { T } } ^ { \pi }$ for $\rho > 0 ;$ concavity gives the upper bound. In the latent rank family, only revealed values distinguish $q ,$ and post-discovery surplus is $O ( \varepsilon )$ . The rank and binomial coverage arguments below yield the lower bound while retaining the dense-budget scale.

For each fixed $k ,$ the rate is $\Theta _ { k } ( \Phi _ { n , k , \rho } ( T ) )$ ; for growing $k ,$ the displayed bounds difer by $O ( k ^ { 2 } )$ . Ignoring numerical revelations and using (6.1) also gives

$$
\mathcal { V } _ { n , T } ^ { \mathsf { F } _ { \rho , ( k ) } } \leq 1 6 \frac { n - k } { n } \operatorname* { m i n } \biggl \{ T , \frac { n + T } { k } \biggr \} \qquad ( 0 \leq \rho \leq 1 ) .
$$

At $\rho = 0$ and $T \geq n$ , this reduces the gap between the available product-law bounds to $O ( k )$ The matching labels-only law of Corollary 11 concerns joint laws and fixed arrays; the budget dependence under product laws remains between these bounds. At fixed $n > k$ , labels alone give linear minimax regret, whereas every fixed $\rho > 0$ permits bounded regret.

## H.2. The independent-clock reduction

Fix an independent-arm instance and the anytime BestProbe policy $\pi$ of Theorem 1. The wrapper repeats each proposed virtual action until the common numerical-revelation coin succeeds. It passes only the complete value-index observation to $\pi$ and does not pass the waiting length or unrevealed labels. In particular, singleton steps within rejection resampling are also repeated until their values are revealed.

For $\rho > 0$ , generate the entire virtual experiment from i.i.d. reward arrays and the policy’s internal randomness, independently of an i.i.d. sequence of geometric waiting times. Unrevealed physical rewards can be filled in independently between successive virtual observations. Since revelation is independent of rewards, this construction has exactly the wrapper’s law. The number $N _ { T }$ of completed virtual rounds has law Binomial $( T , \rho )$ and is independent of the virtual experiment.

The current coin $C _ { t }$ is independent of the current action and reward. Even though instantaneous regret may be negative,

$$
\rho R _ { T } ^ { \mathrm { h o l d } } = \mathbb { E } \sum _ { t = 1 } ^ { T } C _ { t } ( \mu ^ { \star } - W _ { t } ) = \sum _ { s = 0 } ^ { T } \mathbb { P } ( N _ { T } = s ) R _ { s } ^ { \pi } , \qquad R _ { 0 } ^ { \pi } = 0 .\tag{H.2}
$$

The identity includes the unfinished final waiting period. The function $x \mapsto$ min $\{ ( n -$ $k ) x / n , ( n - k ) / k \}$ is concave on $[ 0 , \infty )$ , so

$$
R _ { T } ^ { \mathrm { h o l d } } \leq \frac { 1 4 0 0 } { \rho } \mathbb { E } \Phi _ { n , k } ( N _ { T } ) \leq 1 4 0 0 \operatorname* { m i n } \biggl \{ \frac { n - k } { n } T , \frac { n - k } { k \rho } \biggr \} .
$$

For $\rho = 0$ , the wrapper never advances and ignores every observed label. Therefore $R _ { T } ^ { \mathrm { h o l d } } =$ $T R _ { 1 } ^ { \pi } \leq 1 4 0 0 ( n - k ) T / n$ . Thus one policy attains the stated upper bound for every $\rho$ without knowing it. At $\rho = 0$ , repeating a uniformly random k-set improves the numerical constant to one.

## H.3. Rank moments for every budget

For a hidden arm $q$ and independent nonhidden arms with a common continuous CDF $F _ { 0 } .$ define the rank variable $Z = F _ { 0 } ( X _ { q } )$ . In an ℓ-set containing the hidden arm $q ,$ its winning probability equals $\mathbb { E } Z ^ { \ell - 1 }$ . Requiring every legal action to produce a uniform winner label therefore demands that the first $k - 1$ moments of $Z$ match those of Uniform[0, 1].

Lemma 13 (A rank-matching construction) For every $k \geq 2$ , there is a finitely supported probability law $\nu _ { k }$ on [0, 1] with

$$
\int z ^ { j } \nu _ { k } ( d z ) = { \frac { 1 } { j + 1 } } \quad ( 0 \leq j \leq k - 1 ) , \quad \quad \nu _ { k } ( \{ 1 \} ) = p _ { k } : = { \frac { 1 } { b _ { k } } } , \quad b _ { k } = \left\lfloor { \frac { ( k + 1 ) ^ { 2 } } { 4 } } \right\rfloor .\tag{H.3}
$$

Proof For $k = 2 r .$ , use the $( r + 1 )$ -point Gauss–Lobatto rule on [0, 1], exact through degree $2 r - 1$ , with endpoint masses $1 / [ r ( r + 1 ) ]$ . For $k = 2 r + 1$ , use the $( r + 1 )$ -point right Gauss–Radau rule, exact through degree $2 r .$ , with mass $1 / ( r + 1 ) ^ { 2 }$ at one. These rules have positive weights; exactness for constants makes them probability laws. See Joulak and Beckermann (2009) for exactness and positivity, and Wang et al. (2014, Sections 3.2–3.3) for the endpoint weights, taking Jacobi parameters zero, reflecting the Radau rule when needed, and rescaling to [0, 1]. The case $k = 2$ is simply $( \delta _ { 0 } + \delta _ { 1 } ) / 2$ 7

Hidden-arm construction. Let $0 < \varepsilon \leq 1 / 6 4 , a = 1 / 2 - \varepsilon , b = 1 / 2 + \varepsilon$ , and set $D _ { 0 } = { \mathrm { U n i f o r m } } [ a , b ]$ , with CDF $F _ { 0 }$ . At a uniformly hidden location $q ,$ independently draw $Z \sim \nu _ { k }$ each round and set

$$
X _ { q } = \left\{ \begin{array} { l l } { { a + 2 \varepsilon Z , } } & { { Z < 1 , } } \\ { { 1 , } } & { { Z = 1 . } } \end{array} \right. \quad \mu _ { q } = { \frac { 1 } { 2 } } + p _ { k } \left( { \frac { 1 } { 2 } } - \varepsilon \right) .\tag{H.4}
$$

All other arms have law $D _ { 0 }$ , independently across arms and rounds. Since $F _ { 0 } ( X _ { q } ) = Z$ , in any ℓ-set containing $q$ its winning probability is E $Z ^ { \ell - 1 } = 1 / \ell$ . The bad arms are exchangeable, so every selected label has probability $1 / \ell .$ If q is absent, the same follows from i.i.d. bad rewards. Ties have probability zero because at most one selected arm is discrete; singleton labels are deterministic. Thus every legal winner-label kernel is independent of $q .$

For this instance, let $r _ { q } ( S ) = \mu _ { q } - \mathbb { E } _ { q } \operatorname* { m a x } _ { i \in S } X _ { i }$ . If an ℓ-set omits $q ,$ its maximum has mean $1 / 2 + \varepsilon ( \ell - 1 ) / ( \ell + \bar { 1 } )$ , so its regret is $p _ { k } / 2 - \varepsilon [ p _ { k } + ( \ell - 1 ) / ( \ell + 1 ) ]$ . If it contains $q ,$ the surplus over $X _ { q }$ is pointwise at most $2 \varepsilon$ . Therefore every legal action, before or after identifying $q ,$ satisfies

$$
r _ { q } ( S ) \geq { \frac { p _ { k } } { 2 } } { \mathbf { 1 } } \{ q \not \in S \} - 2 \varepsilon .\tag{H.5}
$$

## H.4. Adaptive numerical coverage

Give the policy a genie: at the end of the first round that both reveals a value and selects $q ,$ it learns $q$ exactly. Let τ be this round. Run the policy on a counterfactual all-bad trajectory with the same revelation-coin law and no genie trigger, extending it arbitrarily to otherwise impossible histories. Write $S _ { t } ^ { 0 }$ for its action and $U _ { t - 1 }$ for the union of its earlier revealed probe sets. Choose $q$ uniformly and independently of this trajectory. Before the trigger, histories couple exactly: unrevealed labels have the same kernel, and revealed actions avoiding q contain only bad arms. Hence

$$
\mathbb { P } ( \tau \geq t , \ q \notin S _ { t } ) = \mathbb { E } \frac { n - | U _ { t - 1 } \cup S _ { t } ^ { 0 } | } { n } \geq \mathbb { E } \left( 1 - \frac { k ( N _ { t - 1 } + 1 ) } { n } \right) _ { + } ,\tag{H.6}
$$

where $N _ { t - 1 } \sim$ Binomial $( t - 1 , \rho )$ and $| U _ { t - 1 } \cup S _ { t } ^ { 0 } | \le k ( N _ { t - 1 } + 1 )$ . The argument permits arbitrary adaptive policies.

Lemma 14 (Binomial coverage) For every $n > k , T \geq 1$ , and $\rho \in [ 0 , 1 ]$

$$
\widetilde { K } _ { n , k , \rho } ( T ) : = \sum _ { t = 1 } ^ { T } \mathbb { E } \left( 1 - \frac { k ( N _ { t - 1 } + 1 ) } { n } \right) _ { + } \geq \frac { 1 } { 4 } \Phi _ { n , k , \rho } ( T ) .\tag{H.7}
$$

Proof At $\rho = 0 , \tilde { K } _ { n , k , 0 } ( T ) = \Phi _ { n , k , 0 } ( T )$ . For $\rho > 0$ , retain the random count rather than replacing it by its mean. With $d = n - k , x = n / k > 1$ , and $C _ { n , k } ( 0 ) = 0$ , the deterministic inequality (2.5) gives

$$
C _ { n , k } ( s ) \geq { \frac { d } { 2 n } } \operatorname* { m i n } \{ s , x \} .
$$

Thinning by the independent current revelation coin enumerates the successful rounds exactly once, so

$$
\rho \widetilde { K } _ { n , k , \rho } ( T ) = \mathbb { E } \sum _ { t = 1 } ^ { T } C _ { t } \left( 1 - \frac { k ( N _ { t - 1 } + 1 ) } { n } \right) _ { + } = \mathbb { E } C _ { n , k } ( N _ { T } ) .
$$

For integer $s \geq 0$ , min $\{ s , x \} \geq x [ 1 - ( 1 - 1 / x ) ^ { s } ]$ . Taking the binomial expectation and using $1 - e ^ { - y } \geq$ min $\{ y , 1 \} / 2$ yields

$$
\widetilde { K } _ { n , k , \rho } ( T ) \geq \frac { d } { 2 k \rho } \left[ 1 - ( 1 - k \rho / n ) ^ { T } \right] \geq \frac { d } { 4 k \rho } \operatorname* { m i n } \{ k \rho T / n , 1 \} = \frac { 1 } { 4 } \Phi _ { n , k , \rho } ( T ) .
$$

Completing the lower bound. Let $D ^ { ( q ) }$ denote the instance with good arm $q .$ Equations (H.5)–(H.7) give, for every policy π, the prior-average bound

$$
\begin{array} { l l } { { \displaystyle \overline { { { R } } } _ { T } ^ { \pi } : = \frac { 1 } { n } \sum _ { q = 1 } ^ { n } R _ { T } ^ { \pi } ( D ^ { ( q ) } ) , } } \\ { { \displaystyle \overline { { { R } } } _ { T } ^ { \pi } \geq \frac { p _ { k } } { 2 } \widetilde { K } _ { n , k , \rho } ( T ) - 2 \varepsilon T \geq \frac { p _ { k } } { 8 } \Phi _ { n , k , \rho } ( T ) - 2 \varepsilon T . } } \end{array}
$$

Choose $\varepsilon = p _ { k } \Phi _ { n , k , \rho } ( T ) / ( 3 2 T )$ , which lies in $( 0 , 1 / 6 4 )$ . This proves the lower bound in (H.1). The prior bound holds for every policy, even one knowing the instance family and all its parameters, and accounts for regret after the genie trigger. At $\rho = 0 , \varepsilon = p _ { k } ( n - k ) / ( 3 2 n )$ is independent of $T ,$ , so the same family works at every horizon. For $\rho > 0 ;$ , the fixed-horizon minimax definition permits a horizon-dependent choice of stationary instance.

## Appendix I. Action constraints and coverage

We first characterize bounded regret for downward-closed action families containing all singletons, then extend the budget law to class quotas.

## I.1. The pair-feasibility threshold

Proposition 15 (Feasible pairs) Let ${ \mathcal { F } } \subseteq 2 ^ { [ n ] }$ be downward closed and contain every singleton. A distribution-uniform horizon-independent regret bound against the best fixed single arm exists under product rewards and BestProbe if and only $i f$ every pair belongs to $\mathcal { F }$ The same equivalence holds for AllProbe on fixed arrays.

Proof When every pair is feasible, use Theorem 6 for $n \geq 3$ or Theorem 3, respectively; if $n = 2$ simply probe both. If $\{ i , j \} \not \in { \mathcal { F } }$ , downward closure prohibits selecting both in any legal action. Put an ordinary two-arm hard stochastic instance at these positions and zeros elsewhere. Every action selects and observes at most one nonzero arm, even under AllProbe, and earns at most that arm’s reward. The two-arm bandit lower bound gives $\Omega ( { \sqrt { T } } )$ (Lattimore and Szepesvári, 2020, Exercise 15.4). Averaging its reward arrays transfers the same obstruction to the fixed-array comparator.

Thus connectivity of the feasible-pair graph is insuficient. For a knapsack constraint $\textstyle \sum _ { i \in S } c _ { i } \leq B$ with $0 \leq c _ { i } \leq B$ , the threshold is that the two largest costs sum to at most B.

## I.2. A partition-quota budget law

Partition the labels into classes $C _ { c }$ of sizes $n _ { c }$ , with integer quotas $1 \leq k _ { c } \leq n _ { c }$ . An action must satisfy $| S \cap C _ { c } | \le k _ { c }$ for every class. Assume each nonfull class has $k _ { c } \ge 2$ , and let

$$
\beta = \operatorname* { m i n } _ { c } \frac { k _ { c } } { n _ { c } } , \qquad \Phi _ { \beta } ( T ) = \operatorname* { m i n } \{ ( 1 - \beta ) T , ( 1 - \beta ) / \beta \} .
$$

Proposition 16 (Heterogeneous quotas) For $\beta < 1$ , the minimax regret is $\Theta ( \Phi _ { \beta } ( T ) )$ , with universal constants independent of the number of classes, for product rewards with BestProbe and for fixed arrays with AllProbe. One anytime policy attains each upper bound. For $\beta = 1$ the value is zero.

Proof Put the deterministic hidden arm in a class attaining $\beta$ and set all other rewards to zero. Its per-round quota gives lower bound $\Phi _ { \beta } ( T ) / 2$ by (2.5).

If $\beta < 2 / 3$ , set

$$
m = \operatorname* { m a x } _ { c : k _ { c } < n _ { c } } \left\lceil \frac { n _ { c } } { \lfloor k _ { c } / 2 \rfloor } \right\rceil < 4 / \beta .
$$

For each nonfull class, distribute its labels across m disjoint global groups with at most $\lfloor k _ { c } / 2 \rfloor$ labels in each. A class attaining m supplies a nonempty piece to every group; assign fully available classes to any one group. Every pair of groups is feasible. The product-reward pair theorem costs less than 236m $< 9 4 4 / \beta < 2 8 3 2 ( 1 - \beta ) / \beta$ , and the fixed-array theorem costs less than $3 2 m < 3 8 4 ( 1 - \beta ) / \beta$ . Before $T = 1 / \beta , R _ { T } \leq T < 3 ( 1 - \beta ) T$

If $\beta \geq 2 / 3$ , set $d _ { c } = n _ { c } - k _ { c }$ . In each class choose a uniform baseline of size $n _ { c } - 2 d _ { c }$ and split the remainder into two $d _ { c } \mathrm { - s e t s }$ . Combining corresponding pieces produces three disjoint global groups. They are nonempty when $\beta < 1$ and every pair is feasible. An arm in class c misses the baseline with probability $2 d _ { c } / n _ { c } \leq 2 ( 1 - \beta )$ . The recovered-gain gate gives at most $1 3 9 6 ( 1 - \beta )$ regret; the directly observed gate gives at most $1 6 4 ( 1 - \beta )$ . At the only integer horizon below $1 / \beta$ , namely $T = 1$ , an arm of class c is included with probability $k _ { c } / n _ { c } \ge \beta$ . This proves the full finite-horizon law. Product grouping preserves independence; the fixed-array policy needs none. For $\beta = 1$ , selecting every arm gives the zero minimax value. ■

The global grouping is important: running separate learners and summing their regrets would introduce a class-count factor that is absent here. In the fixed-array case the same construction uses Contrast, since all global groups are disjoint.

## Appendix J. The roles of credit, scouts, and payof surplus

The first two examples show that two specific simplifications of Algorithm 1 incur linear regret: unfunded switching and runner-only exploration. The third isolates the payof surplus by removing it from the regret objective while revealing every reward.

## J.1. Unfunded switches incur persistent cost

On deterministic rewards $( 1 , 0 . 9 , 0 )$ , the scores are exact after initialization and the default anchor is optimal. Retaining it yields zero gain, so the banks stay empty and positive switching-cost estimates prevent switches. Every later pair probe earns one.

Replace the bank rule by a fixed switch probability $1 / 3 .$ . Every pair probe still earns one, but each switch adds a recovery singleton with expected gap $( 2 / 3 ) ( 1 / 1 0 ) + ( 1 / 3 ) \cdot 1 = 2 / 5$ . If $p _ { s }$ is the probability that physical round s after initialization is such a singleton, then $p _ { 1 } = 0$ and $p _ { s + 1 } = ( 1 - p _ { s } ) / 3$ . Thus $p _ { s }  1 / 4$ , so recovery singletons occupy a positive fraction of physical rounds and $R _ { T } ^ { \mathrm { n o \ b a n k } } = \Omega ( T )$ . The bank prevents this persistent cost of unnecessary sample advancement.

## J.2. Without scouts, an unseen good arm can remain unseen

Let arm $m \geq 3$ be Bernoulli(ε), $0 < \varepsilon < 1$ , and all other arms be zero, with smaller labels winning ties. If the policy always chooses the runner, then with probability $( 1 - \varepsilon ) ^ { 2 }$ both initial samples of arm m are zero. All scores are then zero, and only labels 1, 2 are revisited. On this event every subsequent physical round costs ε. Every action has nonnegative expected regret on this instance, so

$$
\operatorname* { l i m } _ { T \to \infty } \operatorname* { l n f } _ { T } \frac { R _ { T } ^ { \mathrm { n o \ s c o u t } } } { T } \geq ( 1 - \varepsilon ) ^ { 2 } \varepsilon > 0 .
$$

Scouting gives every remaining arm a positive chance of advancing while comparisons concentrate on the runner.

## J.3. Removing the payof surplus

For product rewards, define mean-based weak regret and the surplus of a selected set by

$$
R _ { T } ^ { \mathrm { w e a k } , \mu } = \mathbb { E } \sum _ { t = 1 } ^ { T } \left( \mu ^ { \star } - \operatorname* { m a x } _ { i \in S _ { t } } \mu _ { i } \right) , \qquad \sigma ( S ) = \mathbb { E } \operatorname* { m a x } _ { i \in S } X _ { i } - \operatorname* { m a x } _ { i \in S } \mu _ { i } .
$$

Then $\begin{array} { r } { R _ { T } = R _ { T } ^ { \mathrm { w e a k } , \mu } - \mathbb { E } \sum _ { t < T } \sigma ( S _ { t } ) } \end{array}$ . This objective extends the utility-based weak regret of Chen and Frazier (2017, Section 4.4), with utilities $\mu _ { i }$

Consider $n = 3 , k = 2$ with every round ending in observation of all three rewards. Even with this full feedback, the minimax mean-based weak regret over independent [0, 1] arm distributions is at least $\sqrt { T } / 4 8$

Lower bound. Let $\varepsilon = 1 / ( 8 \sqrt { T } )$ . For a fixed policy, let $P _ { 0 }$ be the law of its full observable trajectory when all three arms are Bernoull $( 1 / 2 )$ , and let $P _ { i }$ be the corresponding law when only arm i is changed to Bernoulli $( 1 / 2 + \varepsilon )$ . Let $O _ { i }$ count rounds omitting arm $i ;$ then $\begin{array} { r } { \sum _ { i } \mathbb { E } _ { 0 } O _ { i } \geq T } \end{array}$ . The common policy kernels contribute no divergence, so

$$
\mathrm { K L } ( P _ { 0 } , P _ { i } ) = T \mathrm { k l } ( 1 / 2 , 1 / 2 + \varepsilon ) = - { \frac { T } { 2 } } \log ( 1 - 4 \varepsilon ^ { 2 } ) \leq { \frac { 8 } { 3 } } T \varepsilon ^ { 2 } = { \frac { 1 } { 2 4 } } .
$$

Pinsker’s inequality and $0 \leq O _ { i } \leq T$ give

$$
\frac 1 3 \sum _ { i } \mathbb { E } _ { i } O _ { i } \geq \frac { T } 3 - \frac { T } { 4 \sqrt { 3 } } \geq \frac { T } 6 .
$$

The prior-average weak regret is therefore at least $\varepsilon T / 6 = \sqrt T / 4 8$

The change-of-measure inequalities used here are given in Lattimore and Szepesvári (2020). The lower bound isolates the role of the payof: observing all rewards does not yield bounded regret once the maximum’s surplus is removed.