# A POSITIVE RESOLUTION OF THE GAP-ENTROPY CONJECTURE

P. M. ARONOW, NATHAN KALLUS, AND PATRICK LOPATTO

Abstract. We prove the gap-entropy conjecture for fixed-confidence best-arm identification with independent unit-variance Gaussian arms, means in [0, 1], and a unique optimal arm. For each suboptimal arm i, let $\Delta _ { i } = \mu _ { * } - \mu _ { i }$ be its gap from the optimal mean, and write $\begin{array} { r } { H = \sum _ { i \neq * } \Delta _ { i } ^ { - 2 } } \end{array}$ Let $p _ { r }$ be the fraction of H contributed by arms with $2 ^ { - ( r + 1 ) } < \Delta _ { i } \leq 2 ^ { - r }$ , and let $\operatorname { E n t } ( I ) =$ $\sum _ { r : p _ { r } > 0 } p _ { r } \log ( 1 / p _ { r } )$ . Among all algorithms that identify the optimal arm with probability at least $\mathrm { ~ i ~ - ~ } \delta$ on every Gaussian instance, the optimal expected number of samples on a given instance, averaged over all permutations of the arm labels, is within absolute constant factors of $H ( \log ( 1 / \delta ) + \operatorname { E n t } ( I ) )$ ). Moreover, there is an algorithm, independent of the instance, whose expected number of samples is bounded by a constant multiple of this quantity plus $g ^ { - 2 } \log \log ( e ^ { e } / g )$ , where $g = \mathrm { m i n } _ { i \neq * } \Delta _ { i }$ is the gap to the closest competitor.

## 1. Introduction

Experiments often compare several alternatives in order to choose the one with the best mean reward. Adaptive sampling, wherein at each step an algorithm pulls one of $K \ge 2$ arms and immediately receives a sample from its unknown reward distribution to inform future pulls, can significantly improve performance. In regret minimization, avoiding apparently inferior arms in future sampling can help attain higher cumulative rewards over the experiment, and in best-arm identification, concentrating future sampling on apparently close competitors for best can find the best faster. Fixed-confidence best-arm identification formalizes this problem by seeking to identify the best arm with as few samples as possible while controlling the probability of returning the wrong arm for all admissible reward distributions [5, 10].

The dificulty depends on the gaps between the mean rewards $( \mu _ { 1 } , \dots , \mu _ { K } ) \in [ 0 , 1 ] ^ { K }$ , where $\mu _ { * } = \operatorname* { m a x } _ { i } \mu _ { i }$ is uniquely attained. In the unit-variance Gaussian model, a suboptimal arm $i \neq *$ with gap $\Delta _ { i } = \mu _ { * } - \mu _ { i }$ from the optimal arm requires $\Omega ( \Delta _ { i } ^ { - 2 } \log ( 1 / \delta ) )$ samples to distinguish it from the optimal arm at error level δ. Summing these costs gives the lower bound $\Omega ( H \log ( 1 / \delta ) )$ , where $\begin{array} { r } { H = \sum _ { i \neq * } \Delta _ { i } ^ { - 2 } } \end{array}$ is the sum of the inverse squared gaps, capturing the baseline cost of distinguishing the best arm if we know up to orders of magnitude how well separated it is from each other arm [10, 12]. This does not, however, reveal the cost of discovering this scale of the gaps, which matters even when the error level δ is fixed.

Exponential-Gap Elimination [9] and lil’UCB [8] achieved upper bounds with iterated-log dependence on gaps of the form $\begin{array} { r } { O ( H \log ( 1 / \delta ) + \sum _ { i \neq * } \bar { \Delta _ { i } ^ { - 2 } } \log \log ( e ^ { e } / \Delta _ { i } ) ) } \end{array}$ . Chen and Li [1] subsequently improved this bound, capping each arm’s iterated-logarithmic factor at log log K, apart from a $g ^ { - \bar { 2 } } \log \log ( 1 / g )$ term where $g = \mathrm { m i n } _ { i \neq * } \Delta _ { i }$ . They also proved that for suficiently small δ and large K, any δ-correct algorithm has a Gaussian instance requiring Ω(H log log K) samples. This raised the question of how this unavoidable adaptation cost depends on the particular configuration of gaps.

Chen and Li [2] proposed that the additional instance-dependent cost is described by gap entropy. For an instance I, peel the suboptimal arms into geometrically spaced shells according to their gaps, and assign to each shell the fraction of H contributed by its arms. The entropy of the resulting probability vector, Ent(I), measures how widely the required sampling efort is distributed across scales.

Chen and Li conjectured (their conjecture 3.5) that, among δ-correct algorithms, the smallest expected sample size achievable on a given instance I (averaged over permutations of its arm labels) is of order $H ( \log ( 1 / \delta ) + \operatorname { E n t } ( I ) )$ . (A diferent algorithm may attain the benchmark for each $I ,$ provided it remains δ-correct on all instances.) They further conjectured (their conjecture 3.2) that a single algorithm can attain this instance-wise benchmark simultaneously on all instances, up to an additive term of order $g ^ { - 2 } \log \log ( 1 / g )$ . Such an adaptation cost is unavoidable already with two arms: adapting to an unknown gap that may be arbitrarily small incurs an iterated-logarithmic cost along some sequences of instances, even though this cost is not required at every fixed instance [2, 6].

The conjectures seek nonasymptotic bounds that hold uniformly in the instance and δ. By contrast, Track-and-Stop [7] already achieves asymptotic optimality as $\delta \downarrow 0$ with the instance fixed. In that limit, the gap entropy of the fixed instance is eventually dominated by $\log ( 1 / \delta )$ . An asymptotically optimal leading constant therefore does not determine the additional cost when the gap configuration and the confidence parameter vary together. Simchowitz, Jamieson, and Recht [13] studied the complexity distinctions between fixed δ and $\delta \downarrow 0$ asymptotics, including the cost of learning a sampling allocation in structured models. Cho and Kallus [4] relax exact error control to asymptotic validity in the $\delta \downarrow 0$ regime, enabling the use of strong invariance principles to match the optimal exact-error Gaussian benchmark with known variances even under nonparametric reward distributions.

Chen, Li, and Qiao [3] made progress toward the conjectures, obtaining the upper bound $O ( H ( \log ( 1 / \delta ) + \operatorname { E n t } ( I ) ) ) + \overleftarrow { g } ^ { - 2 } \log \log ( 1 / g ) \ \mathrm { p o l y l o g } ( K , 1 / \delta ) )$ (their theorem 1.11) and, for Gaussian instances whose gaps are powers of two, a lower bound of $\Omega ( H ( \log ( 1 / \delta ) + \operatorname { E n t } ( I ) ) )$ for algorithms whose expected sample cost does not increase when suboptimal arms are deleted (their theorem 1.12 and corollary 1.13).

In this paper, we answer both conjectures in the positive. We remove both the extra polylogarithmic factor from the upper bound (which applies to arbitrary 1-sub-Gaussian reward distributions) and the restrictions on instances and algorithms in the lower bound (for the Gaussian model).

1.1. Main results. For $K \geq 2$ , a K-armed bandit instance $I = \left( \nu _ { 1 } , \dots , \nu _ { K } \right)$ has reward distributions $\nu _ { i }$ with means $\mu _ { i } \in [ 0 , 1 ]$ . The index of the largest mean, denoted by ∗, is assumed unique. Each arm has an sequence of independent samples with law $\nu _ { i } .$ and the sequences are independent across arms. Pulling an arm reveals its next sample.

Let $\scriptstyle { S _ { K } }$ be the class of such unique-optimum instances with 1-sub-Gaussian rewards, meaning that

$$
\begin{array} { r } { \mathbb { E } _ { X \sim \nu _ { i } } \exp \bigl ( \lambda ( X - \mu _ { i } ) \bigr ) \leq \exp ( \lambda ^ { 2 } / 2 ) \qquad ( \lambda \in \mathbb { R } , \ i \in [ K ] ) . } \end{array}
$$

The algorithm knows that the rewards are 1-sub-Gaussian, but not their distributions or means. Let $\mathcal G _ { K } \subset S _ { K }$ be the Gaussian subclass with $\nu _ { i } = N ( \mu _ { i } , 1 )$

For either class, write

$$
m = \mu _ { * } , \qquad \Delta _ { i } = m - \mu _ { i } ( i \neq * ) , \qquad H = \sum _ { i \neq * } \Delta _ { i } ^ { - 2 } , \quad g = \operatorname * { m i n } _ { i \neq * } \Delta _ { i } , \quad D = g ^ { - 2 } .
$$

For $r = 0 , 1 , . . . ,$ , let

$$
G _ { r } = \{ i \ne * : 2 ^ { - ( r + 1 ) } < \Delta _ { i } \leq 2 ^ { - r } \} , \qquad H _ { r } = \sum _ { i \in G _ { r } } \Delta _ { i } ^ { - 2 } , \qquad p _ { r } = H _ { r } / H .
$$

For a probability vector $q \in [ 0 , 1 ] ^ { \mathbb { N } _ { 0 } }$ , write $\begin{array} { r } { \operatorname { E n t } ( q ) = \sum _ { r : q _ { r } > 0 } q _ { r } \log ( 1 / q _ { r } ) } \end{array}$ for its Shannon entropy. The gap entropy of instance I is defined as

$$
\operatorname { E n t } ( I ) = \operatorname { E n t } ( p ) .\tag{1.1}
$$

It is zero when all gaps lie on the same geometric scale. If d groups contribute equally to H, it equals log d. More generally, writing $\mathcal { U } = \{ r : H _ { r } > 0 \}$ for the finite set of indices of the nonempty gap groups, we have $0 \leq \mathrm { E n t } ( I ) \leq \log | \mathcal { U } |$

All logarithms are natural unless otherwise indicated. Constants denoted by c or $C$ are positive and absolute and may change between occurrences. Constants used as algorithm parameters are fixed within each construction. We write $[ K ] = \{ 1 , \ldots , K \}$

An algorithm is δ-correct on a model class if, on every instance in that class, it stops almost surely and returns the optimal arm with probability at least 1 − δ. Write $T _ { A }$ for its total number of samples. For a permutation $\pi \in S _ { K }$ of the labels, let πI be the relabeled instance. For $I \in { \mathcal { G } } _ { K }$ following [2, definition 3.1], define

$$
\mathcal { L } ( I , \delta ) = \operatorname* { i n f } _ { A : \delta \mathrm { - c o r r e c t ~ o n } \mathcal { G } _ { K } } \frac { 1 } { K ! } \sum _ { \pi \in S _ { K } } \mathbb { E } _ { \pi I } T _ { A } .\tag{1.2}
$$

The averaging removes any advantage from a favorable labeling. The infimum is taken separately for each Gaussian instance I, but every algorithm in it must be δ-correct on the Gaussian class $\mathcal { G } _ { K }$ Appendix A compares these conventions with those of [2] and explains why our results imply their original conjectures.

We now present our main results.

Theorem 1.1. There exist absolute constants $c , C > 0$ such that, for every $I \in { \mathcal { G } } _ { K }$ and $0 < \delta < 0 . 1$ 2

$$
c H ( \log ( 1 / \delta ) + \operatorname { E n t } ( I ) ) \leq { \mathcal { L } } ( I , \delta ) \leq C H ( \log ( 1 / \delta ) + \operatorname { E n t } ( I ) ) .
$$

The lower and upper bounds are proved in Sections 2 and 3, respectively. The lower bound is proved directly for the original Gaussian instance, after averaging over permutations of the arm labels, without first passing to a subinstance obtained by deleting arms. The matching upper bound uses an algorithm selected for that target instance. More generally, Proposition 3.1 constructs, for each instance in $\scriptstyle { S _ { K } }$ , an algorithm that is δ-correct on $\scriptstyle { S _ { K } }$ and attains the same bound on every relabeling of the instance. The next theorem gives one algorithm for the whole sub-Gaussian class.

Theorem 1.2. There exists one algorithm, given only K and $\delta ,$ that is δ-correct on $\scriptstyle { S _ { K } }$ and satisfies, for every $I \in { \cal S } _ { K }$ and $0 < \delta < 0 . 1$ ，

$$
\begin{array} { r } { \mathbb { E } _ { I } T _ { A } \le C \left( H ( \log ( 1 / \delta ) + \mathrm { E n t } ( I ) ) + D \log \log ( e ^ { e } / g ) \right) . } \end{array}
$$

The constant C is absolute.

The algorithm is defined in Section 4; its correctness and bounds on the expected sample size are proved in Sections 5 and 6, respectively. The additive term depends only on the smallest gap, with no further factor involving K or δ. Specializing the uniform algorithm to $\mathcal { G } _ { K }$ and applying Theorem 1.1 gives the near-optimality on every individual instance conjectured in [2, conjecture 3.2].

Corollary 1.3. For every $I \in { \mathcal { G } } _ { K }$ and $0 < \delta < 0 . 1$ , the algorithm in Theorem 1.2 satisfies

$$
\mathbb { E } _ { I } T _ { A } \leq C ( \mathcal { L } ( I , \delta ) + D \log \log ( e ^ { e } / g ) ) .
$$

The proof is given at the end of Section 6. Theorem 1.1 proves $[ 2 ,$ conjecture 3.5], and together with Corollary 1.3 proves [3, conjecture 1.9]; see Appendix A for details.

1.2. Ideas of the proof. A short allocation calculation explains why entropy appears. Suppose the group weights $H _ { r }$ were known and group $G _ { r }$ were assigned error budget $\delta _ { r }$ . The corresponding sampling cost would have the form $\begin{array} { r l r } { \sum _ { r } H _ { r } \log ( 1 / \delta _ { r } ) } & { { } } & { } \end{array}$ . Under the constraint $\textstyle \sum _ { r } \delta _ { r } \leq \delta$ , this expression is minimized by $\delta _ { r } = \delta p _ { r }$ , giving

$$
\begin{array} { r } { \underset { \delta _ { r } > 0 } { \operatorname* { i n f } } ( r \varepsilon u ) \sum _ { r \in \mathcal { U } } H _ { r } \log ( 1 / \delta _ { r } ) = H ( \log ( 1 / \delta ) + \mathrm { E n t } ( I ) ) . } \end{array}\tag{1.3}
$$

In particular, if d groups contribute equally to $H .$ , then $\delta _ { r } = \delta / d$ and the additional cost is H log d. This calculation, used to motivate the conjecture in $[ 2 ,$ section $4 ] ,$ , does not by itself prove a lower bound for an arbitrary algorithm or give an algorithm when the weights are unknown. We first show that every algorithm correct on the Gaussian class must pay the entropy cost. We then attain this cost using an algorithm selected for a fixed target instance. Finally, we construct an algorithm that does not depend on the instance and bound the additional cost of adaptation.

1.2.1. The lower bound. The term $H \log ( 1 / \delta )$ follows from the standard bandit change-of-measure argument; see [10, lemma 1]. The remaining task is to establish the additional entropy cost for an arbitrary correct algorithm. For each gap group $G _ { r }$ , we choose one of its arms uniformly and examine how many times the algorithm samples that arm. Since arms in $G _ { r }$ have gaps of order $2 ^ { - r }$ their sampling complexity is on the scale $4 ^ { r }$ , and we associate $G _ { r }$ with an event in which this sample count lies in a suitable range around that scale. Widely separated scales give disjoint ranges, and the main dificulty is to compare the probabilities of these events under one common reference law.

Fix a Gaussian instance and an arbitrary algorithm A that is δ-correct on the Gaussian class. Define a new algorithm by first applying a uniformly random permutation to the arm labels, running A on the relabeled instance, and then applying the inverse permutation to $A \mathrm { { } i \mathrm { { s } } }$ output. The new algorithm is still $\delta \mathrm { - }$ correct, treats all labelings symmetrically, and has expected sample cost on every labeling equal to the average cost of $A$ over all labelings of the original instance. Let $u _ { i }$ be its expected number of samples from arm i of the target instance, and let

$$
\alpha _ { r } = \frac { \sum _ { i \in G _ { r } } u _ { i } } { | G _ { r } | 4 ^ { r } } , \qquad r \in \mathcal { U } .
$$

An arm in $G _ { r }$ has inverse squared gap comparable to $4 ^ { r }$ . Thus $\alpha _ { r }$ is the average sample count in the group divided by this sampling scale.

Fix $a \ge 1$ . For each group with $\alpha _ { r } \leq a .$ , let $P _ { r }$ be the law of the algorithm’s run when the optimal arm has label 1, a uniformly chosen arm from $G _ { r }$ has label 2, and the remaining arms are uniformly relabeled. Let $N _ { 2 }$ count the samples from label 2, and let $E _ { r } ( a )$ be the event that the algorithm returns label 1 with $c _ { 0 } 4 ^ { r } \le N _ { 2 } \le 4 a 4 ^ { r }$ , for a suficiently small absolute constant $c _ { 0 }$ . Since $\mathbb { E } _ { P _ { r } } N _ { 2 } = \alpha _ { r } 4 ^ { r } \le a 4 ^ { r }$ , Markov’s inequality gives $P _ { r } ( N _ { 2 } > 4 a 4 ^ { r } ) \leq 1 / 4$

For the lower tail, raise label 2’s mean to m, producing a tied reference law $Q _ { r }$ . Correctness on nearby instances in which label 2 is uniquely optimal forces the probability of returning label 1 under $Q _ { r }$ to be at most δ. Before label 2 has been sampled $c _ { 0 } 4 ^ { r }$ times, the KL divergence between the corresponding history distributions is bounded by an absolute constant times $c _ { 0 }$ . A finite-horizon change-of-measure argument therefore shows that returning label 1 this early also has small probability under $P _ { r }$ . Combining the two tail bounds with correctness gives $P _ { r } ( E _ { r } ( a ) ) \geq 1 / 2$ the details are in Section 2.

For suficiently separated indices $r , r ^ { \prime } ,$ the ranges $[ c _ { 0 } 4 ^ { r } , 4 a 4 ^ { r } ]$ and $[ c _ { 0 } 4 ^ { r ^ { \prime } } , 4 a 4 ^ { r ^ { \prime } } ]$ are disjoint. But their event probabilities have been bounded under diferent laws $P _ { r } ,$ and the tied reference laws $Q _ { r }$ also depend on r. To compare diferent groups, we construct a reference law that can be used for all of them. Fix a nonempty group $G _ { s } .$ . For $r > s$ , let i denote the arm placed at label 2, choose $j$ uniformly from $G _ { s }$ , independently of the labeling, and change the two means by

$$
( \mu _ { i } , \mu _ { j } ) \longmapsto ( m , \mu _ { i } ) .
$$

The second change restores the suboptimal mean removed by the first. Labels 1 and 2 are now tied, and the other labels carry exactly the original suboptimal means with $\mu _ { j }$ omitted. After averaging over the random choices, those remaining means are uniformly arranged on labels $3 , \ldots , K$ . The resulting law, denoted by $Q _ { s }$ , therefore depends on s but not on r. For $r = s$ , raising the selected arm to m alone gives the same $Q _ { s }$ . Note that each of these resulting instances is inadmissible because labels 1 and 2 are tied for best. While δ-correctness or almost-sure stopping are therefore not required for A on these, the law of the algorithm’s run is still well-defined and its finite-time answer probabilities are controlled by its required δ-correct behavior on nearby admissible instances where label 1’s mean is slightly reduced, making label 2 uniquely optimal.

A Gaussian change-of-measure calculation bounds the KL divergence from $P _ { r }$ to $Q _ { s } ,$ , at any finite horizon, by $( \alpha _ { r } + \alpha _ { s } ) / 2$ . If any group satisfies $\alpha _ { r } \leq a ,$ choose s = min $\{ r \in \mathcal { U } : \alpha _ { r } \leq a \}$ The divergence bound is then at most a for every such group, and $P _ { r } ( E _ { r } ( a ) ) \ \geq \ 1 / 2$ implies $Q _ { s } ( E _ { r } ( a ) ) \geq e ^ { - 2 a } / 4$ . Under this one law, events from suficiently separated groups are disjoint. Since each has probability at least $e ^ { - 2 a } / 4$ , any such collection contains at most $4 e ^ { 2 a }$ groups. Partitioning the indices into $O ( a )$ separated collections gives the required counting bound for $a \geq 1$ The standard per-arm lower bound gives $\alpha _ { r } > 1$ for every $^ { r , }$ so integrating this counting bound, as carried out in Section 2, yields

$$
\sum _ { r } e ^ { - 8 \alpha _ { r } } \leq 1 .
$$

Since the weights $e ^ { - 8 \alpha _ { \tau } }$ have total mass at most one, normalizing them and applying Gibbs’s inequality gives

$$
H \operatorname { E n t } ( I ) \leq 8 \sum _ { r } H _ { r } \alpha _ { r } \leq 3 2 \sum _ { i \neq * } u _ { i } .
$$

Since $\textstyle \sum _ { i \neq * } u _ { i }$ is the expected number of samples from the suboptimal arms, it is at most the expected total number of samples. Hence, writing $T$ for the total number of samples

$$
\mathbb { E } T \geq { \frac { 1 } { 3 2 } } H \operatorname { E n t } ( I ) .
$$

Together with the standard lower bound $\mathbb { E } T \geq c H \log ( 1 / \delta )$ , this gives

$$
\mathbb { E } T \geq c ^ { \prime } H ( \log ( 1 / \delta ) + \mathrm { E n t } ( I ) )
$$

for an absolute constant $c ^ { \prime } > 0$ , proving the lower half of Theorem 1.1.

1.2.2. The instance-wise upper bound. For each fixed target instance $I \in { \cal S } _ { K }$ , we may choose an algorithm specifically for I, provided it remains δ-correct on all of $\scriptstyle { S _ { K } }$ . The algorithm’s definition includes the indices and sizes of the target’s nonempty gap groups and a threshold $\tau \in ( m - g / 2 , m - g / 3 )$ . Thus $\tau$ lies strictly between the largest mean $m$ and the second-largest mean $m - g$ , at distance comparable to $g$ from both. On the target, the optimal arm is the unique arm with mean above $\tau .$ Thus it sufices to eliminate all arms whose means lie below $\tau .$ . We do this through a sequence of independent trials. Each trial may return an arm or return without an answer.

Choose probability weights $q _ { r }$ proportional to $| G _ { r } | 4 ^ { r }$ , so that $q _ { r }$ is comparable to $p _ { r }$ . Given a parameter $\eta$ that will serve as the trial’s error allowance, the trial starts with all arms active and cycles through the active arms, taking one sample from each in turn. For each arm, we combine one exponential test statistic for each supplied gap scale, using mixture weight $q _ { r }$ for scale $r .$ An arm is eliminated when this combined statistic reaches its rejection threshold. For a suboptimal target arm $i \in G _ { r }$ , let $U _ { i }$ be the number of samples from that arm before it is eliminated. We show that

$$
\mathbb { E } U _ { i } \le C \Delta _ { i } ^ { - 2 } \big [ \log ( 1 / \eta ) + \log ( 1 / q _ { r } ) \big ] .
$$

Thus the mixture weight $q _ { r }$ enters the sample bound through the additive term $\log ( 1 / q _ { r } )$ . Since $q _ { r }$ is comparable to $p _ { r } .$ , summing over the suboptimal arms gives

$$
\sum _ { i \neq * } \mathbb { E } U _ { i } \leq C H ( \log ( 1 / \eta ) + \operatorname { E n t } ( I ) ) .
$$

Because the procedure takes one sample from every active arm in each cycle through the active set, the optimal arm is sampled at most $\operatorname* { m a x } _ { i \neq * } U _ { i }$ times before at most one arm remains. Hence the total number of samples used during elimination is at most

$$
\sum _ { i \neq * } U _ { i } + \operatorname* { m a x } _ { i \neq * } U _ { i } \leq 2 \sum _ { i \neq * } U _ { i } .
$$

Although the algorithm is chosen using information from the target instance, it must remain δ-correct when run on every instance in $\boldsymbol { \mathcal { S } } _ { K }$ , including instances whose means and gap groups difer from those of the target. There are two cases. If the input has an arm with mean at least $\tau ,$ then the optimal arm also has mean at least τ. An incorrect output is therefore possible only if the tests eliminate the optimal arm, which happens with probability at most $\eta .$ If instead every input mean is below τ, then before returning the sole remaining arm the trial takes fresh samples from it and requires its empirical mean to exceed a level slightly above τ. We choose the number of confirmation samples so that, by the sub-Gaussian tail bound, this event has probability at most η. Thus in this case the trial returns any arm with probability at most $\eta ,$ and in particular returns an incorrect arm with probability at most $\eta .$

We impose a deterministic sample limit on each trial and repeat independent trials with geometrically decreasing error allowances whose sum is at most $\delta / 2$ . On the target instance, each trial returns the optimal arm with probability at least $3 / 4$ , so the probability of reaching later trials decreases geometrically while their sample limits grow only linearly with the trial index. The repeated-trial procedure need not stop on every other instance. To guarantee almost-sure stopping on the full class, we interleave it with an independent $\delta / 2 \cdot$ -correct confidence-interval procedure and return the first answer. This at most doubles the target-instance cost, up to one sample, while the two error guarantees together give δ-correctness.

1.2.3. The uniform upper bound. The preceding instance-wise construction uses the instance’s gap distribution to choose its testing scales and error allocation. We now construct one algorithm, given only K and $\delta ,$ that must discover this information from the data. It runs independent stages with geometrically increasing budgets $B _ { t } = 1 0 0 ^ { t }$ . Each stage begins with all K arms and either returns an arm or rejects and passes control to the next stage. A deterministic sample limit of order $B _ { t }$ bounds the cost of the stage on every sample path.

Within a stage, round r works at accuracy $\varepsilon _ { r } = 2 ^ { - r }$ . A Median call selects an arm whose mean is within $\varepsilon _ { r } / 8$ of the largest active mean, with probability at least 0.99. The algorithm estimates this arm’s mean and uses a Fraction call to determine whether suficiently many active arms lie substantially below it. If so, an Eliminate call removes arms with low estimated means; otherwise the active set is unchanged. The stage returns as soon as only one arm remains. It rejects if either of its two budget tests is triggered, its final permitted round is completed without a singleton, or its sample limit is reached.

For an active set $S _ { r }$ , the sampling-cost scale at accuracy $\varepsilon _ { r }$ is

$$
w _ { r } = | S _ { r } | \varepsilon _ { r } ^ { - 2 } = | S _ { r } | 4 ^ { r } .
$$

This observable quantity replaces the unknown gap-group weights used by the instance-wise algorithm. Set $\eta = c \delta$ , where $c > 0$ is a suficiently small absolute constant. An Eliminate call in round r receives an error allowance proportional to $w _ { r } / B _ { t }$ . The stage controls both the sum of these allowances and the cumulative sum of

$$
z _ { r } = w _ { r } \log \frac { B _ { t } } { \eta w _ { r } } .
$$

Up to an absolute constant, $z _ { r }$ bounds the expected cost of the Median and Eliminate calls in that round.

The main cost estimate controls how long each suboptimal arm remains active. In the comparison process of Section $5 ,$ if Median succeeds in round r and every Eliminate call through that round is valid, the next active set has size at most an absolute constant times

$$
1 + | \{ i \neq * : \Delta _ { i } \leq 2 \varepsilon _ { r } \} | .
$$

If Median fails for $j$ additional rounds, the probability of that delay is at most $0 . 0 1 ^ { j }$ , whereas the sampling scale grows by $4 ^ { j }$ . The series

$$
\sum _ { j \geq 0 } ( 4 \cdot 0 . 0 1 ) ^ { j }
$$

is finite, so these delays change the expected cost by only an absolute factor. Consequently, arms with gaps of order $2 ^ { - r }$ contribute mainly near round r. For $B _ { t } \geq H$ , sum $z _ { r }$ over executed rounds up to and including the first invalid Eliminate call, or until an earlier stage termination. The active-set and entropy estimates in Lemma 5.1 give

$$
\mathbb { E } \sum _ { r } z _ { r } \leq C H \left[ \log ( 1 / \delta ) + \mathrm { E n t } ( I ) + 1 + \log ( B _ { t } / H ) \right] .
$$

The outer Mean and Fraction calls require a separate summation. The separation between the two fraction thresholds is indexed by the number of rounds remaining. Reindexing from the final round turns the resulting polynomial factor into a summable correction to the geometric round costs; see Lemma 6.1. The resulting sample costs also grow logarithmically with the stage index. The calculation in Lemma 6.2 accounts for this dependence through the term $C H ( \log ( 1 / \delta ) +$ Ent(I)) together with

$$
C D \log \log ( e ^ { e } / g ) .
$$

Correctness across stages requires more than a direct union bound, because each stage receives the same total budget for elimination errors. We again use the comparison process. For a budget B, call the optimal arm and every suboptimal arm satisfying $\Delta _ { i } ^ { - 2 } > B$ hard; call the remaining suboptimal arms easy. A wrong output requires the optimal arm and all but possibly one of the hard suboptimal arms to be removed. Every removal of a hard suboptimal arm occurs before its gap scale is reached. The product guarantee for Eliminate controls these joint premature removals. When there are few hard arms, a localized estimate instead uses

$$
B ^ { - 1 } \sum _ { i : i \mathrm { \scriptsize ~ e a s y } } \Delta _ { i } ^ { - 2 } .
$$

When this ratio is large, an exponential-moment argument shows that acceptance itself is unlikely. Together with the bound on disagreement between the actual and comparison processes, these estimates control the total error across stages; see Section 5.

Finally, Lemma 6.2 shows that a stage accepts with probability at least 199/200 once

$$
B _ { t } \geq C \left\{ H ( \log ( 1 / \delta ) + \operatorname { E n t } ( I ) ) + D \log \log ( e ^ { e } / g ) \right\} .
$$

Beyond this point, the probability of reaching the next stage decreases by a factor at most $1 / 2 0 0$ while its sample limit increases by a factor of 100. The successive expected costs therefore decrease by a factor at most $1 / 2$ . This proves both almost-sure stopping and the claimed bound on the expected sample size. The construction modifies [3, algorithms 4–5] and is specified in Section 4; its correctness and cost are proved in Sections 5 and 6.

## 2. The instance-wise lower bound

Throughout this section the instance belongs to $\mathcal { G } _ { K }$ , and δ-correctness is required on $\mathcal { G } _ { K }$ . We prove the lower bound in Theorem 1.1. The term $H \log ( 1 / \delta )$ follows from the usual Gaussian change of measure. For the entropy term, we compare events defined by the number of samples from a selected arm across diferent gap groups, using a common law with two tied optimal arms.

The conversion of these sample-count bounds into an entropy lower bound parallels the entropy lower-bound argument of Chen, Li and Qiao $[ 3 ,$ lemma 4.1 and appendix D]. The additional issue here is to construct a common reference law while evaluating all expected sample counts on permutations of the original instance.

The proof of the following proposition appears at the end of this section.

Proposition 2.1. For every $I \in { \mathcal { G } } _ { K }$ and $0 < \delta < 0 . 1$ , write $b _ { \delta } = \operatorname { k l } ( 1 - \delta , \delta )$ . Then

$$
\begin{array} { r } { \mathcal { L } ( I , \delta ) \ge \operatorname* { m a x } \{ 2 b _ { \delta } H , \ H \operatorname { E n t } ( I ) / 3 2 \} , } \end{array}\tag{2.1}
$$

where $\mathrm { k l } ( p , q ) = p \log ( p / q ) + ( 1 - p ) \log ( ( 1 - p ) / ( 1 - q ) )$

Since $b _ { \delta } = ( 1 - 2 \delta ) \log ( ( 1 - \delta ) / \delta ) \asymp \log ( 1 / \delta )$ uniformly over the stated confidence range, the lower half of Theorem 1.1 follows.

2.1. Symmetry and changes of measure. Fix $I \in { \mathcal { G } } _ { K }$ and let A be an arbitrary algorithm that is δ-correct on $\mathcal { G } _ { K }$ . Randomly permuting labels before running A, and undoing the permutation on its answer, gives a permutation-equivariant algorithm B such that

$$
\mathbb { E } _ { \pi I } T _ { B } = \frac { 1 } { K ! } \sum _ { \rho \in S _ { K } } \mathbb { E } _ { \rho I } T _ { A } \quad \mathrm { f o r ~ e v e r y } ~ \pi .\tag{2.2}
$$

The algorithm B is still δ-correct. It is useful to distinguish an arm’s identity from its label: the identities refer to the fixed instance, whereas the labels specify where the arms are presented to the algorithm. Distinct identities are retained even when their means coincide. Let $u _ { i }$ be the expected number of samples from identity i under B. By permutation equivariance, the expected number of samples allocated to a given arm identity does not depend on the label at which that identity is placed. Thus, for every permutation $\pi .$

$$
\mathbb { E } _ { \pi I } N _ { \pi ( i ) } = u _ { i } .
$$

Consequently, averaging over any of the random labelings used below does not alter these expectations.

If the right side of (2.2) is infinite there is nothing to prove. Henceforth assume $\mathbb { E } _ { I } T _ { B } < \infty$ , and suppress the subscript B.

We use the standard finite-horizon likelihood-ratio calculation underlying the bandit change-ofmeasure inequality; see [10, lemma 1 and appendix A.1]. For a deterministic integer M, write $\bar { P } _ { \mu } ^ { ( M ) }$ for the distribution of the recorded history up to $T \wedge M$ under mean vector $\mu .$ . For two fixed mean vectors $\mu , \lambda$ , the chain rule gives

$$
\mathrm { K L } ( P _ { \mu } ^ { ( M ) } \| P _ { \lambda } ^ { ( M ) } ) = \frac { 1 } { 2 } \sum _ { i } ( \mu _ { i } - \lambda _ { i } ) ^ { 2 } { \mathbb E } _ { \mu } N _ { i } ( T \wedge M ) .\tag{2.3}
$$

Here $N _ { i } ( t )$ counts samples from the arm at label i by time t. Write $N _ { i } = N _ { i } ( T )$ for its total number of samples, and let $\widehat { i }$ denote the returned label. On $\{ T = \infty \}$ , set $\hat { i } = 0$ and interpret $N _ { i } ( T )$ as $\scriptstyle \operatorname* { l i m } _ { t \to \infty } N _ { i } ( t )$ . The history at time $T \wedge M$ records whether the algorithm has stopped and, if so, its output, as well as whether truncation occurred. Since the algorithm uses the same sampling, randomization, and stopping rules under both models, these decisions do not contribute additional KL divergence.

We will also use (2.3) after truncation at bounded stopping times and after introducing auxiliary random variables having the same law under both models; these follow from the same chain-rule calculation and data processing. Whenever we use a tied reference law, we first work at a fixed finite horizon and only then pass to the limit. Thus we never need to assume that the algorithm stops when the two arms are tied.

We first derive the standard lower bound for the expected number of samples from a single suboptimal arm; see [10, lemma 1]. We claim that

$$
u _ { i } \geq 2 b _ { \delta } \Delta _ { i } ^ { - 2 } , \qquad \mathbb { E } _ { I } T \geq 2 b _ { \delta } H .\tag{2.4}
$$

For an alternative instance with a diferent optimal arm, apply data processing to the event that the algorithm returns the original optimal arm to obtain

$$
{ \frac { 1 } { 2 } } \sum _ { i } u _ { i } ( \mu _ { i } - \lambda _ { i } ) ^ { 2 } \geq b _ { \delta } .\tag{2.5}
$$

(To justify (2.5) from (2.3), first use the event $\{ T \leq M , \widehat { i } = * \}$ and then let $M \to \infty . )$ If $m < 1$ raise suboptimal arm i to $m + \varepsilon$ and send $\varepsilon \downarrow 0$ in (2.5) to get (2.4). If $m = 1$ , instead raise arm i to 1 and lower the original optimal arm to $1 - \varepsilon ,$ , with $0 < \varepsilon < g _ { : }$ , and take $\varepsilon \to 0$ in (2.5), using $u _ { * } < \infty .$

For each nonempty gap group $r \in \mathcal { U }$ , put

$$
n _ { r } = | G _ { r } | , \qquad t _ { r } = \sum _ { i \in G _ { r } } u _ { i } , \qquad \alpha _ { r } = \frac { t _ { r } } { n _ { r } 4 ^ { r } } .\tag{2.6}
$$

Since $\Delta _ { i } ^ { - 2 } \geq 4 ^ { r }$ in $G _ { r } , ~ ( 2 . 4 )$ implies

$$
\alpha _ { r } \geq 2 b _ { \delta } > 1 .\tag{2.7}
$$

2.2. A common reference law. For $r \in \mathcal { U } .$ , let $P _ { r }$ be the law of the algorithm’s run under the following random relabeling of the original instance I, with all arm distributions left unchanged: the optimal arm has label 1; an identity i chosen uniformly from $G _ { r }$ has label $2 ;$ the other identities are placed uniformly on labels $3 , \ldots , K$ . Then

$$
P _ { r } ( \widehat { i } = 1 ) \geq 1 - \delta , \qquad \mathbb { E } _ { P _ { r } } N _ { 2 } = \alpha _ { r } 4 ^ { r } .\tag{2.8}
$$

The expectation identity follows from permutation equivariance, since the algorithm is not told which instance identities occupy labels 1 and 2.

Under $P _ { r }$ , raising only the mean at label 2 to m would produce a tied law whose remaining suboptimal arms omit the chosen identity from $G _ { r } ;$ this law would therefore still depend on r. To obtain a common reference law, fix $s \in \mathcal { U }$ and define $Q _ { s }$ as follows. Give labels 1 and 2 mean m, choose an identity $j$ uniformly from $G _ { s } ,$ and assign all original suboptimal identities other than $j$ uniformly to labels $3 , \ldots , K$ . Thus $Q _ { s }$ has exactly two tied optimal arms and does not depend on r. Although this tied law lies outside the unique-optimum model class, its finite-time answer probabilities are controlled by perturbations with a unique optimum, as the next lemma shows. Let $\mathbf { \bar { \mathbf { \Gamma } } } _ { P _ { r } ^ { ( M ) } }$ and $Q _ { s } ^ { ( M ) }$ denote the corresponding laws of the recorded histories up to $T \wedge M$

## Lemma 2.2. For every $s \in \mathcal { U } , Q _ { s } ( T < \infty , \widehat { i } = 1 ) \leq \delta$

Proof. Since the optimal arm is unique and all means are nonnegative, $m > 0$ . For $0 < \varepsilon < m$ , let $Q _ { s , \varepsilon }$ be obtained from $Q _ { s }$ by lowering the mean at label 1 from m to $m - \varepsilon$ . Every component of $Q _ { s , \varepsilon }$ has label 2 as its unique optimal arm. Therefore, for every fixed M,

$$
Q _ { s , \varepsilon } ( T \leq M , \widehat { i } = 1 ) \leq \delta .
$$

The KL divergence between the laws under $Q _ { s , \varepsilon }$ and $Q _ { s }$ of the history up to $T \wedge M$ is at most $M \varepsilon ^ { 2 } / 2$ . Pinsker’s inequality consequently gives

$$
Q _ { s } ( T \leq M , \widehat { i } = 1 ) \leq \delta + \frac { \varepsilon \sqrt { M } } { 2 } .
$$

First let $\varepsilon \downarrow 0$ with M fixed, and then let $M \to \infty$ to obtain the claim.

Lemma 2.3. For $r , s \in \mathcal { U }$ with $s \leq r$ and every finite horizon $M$

$$
\mathrm { K L } ( P _ { r } ^ { ( M ) } \| Q _ { s } ^ { ( M ) } ) \leq \frac { \alpha _ { r } + \alpha _ { s } } { 2 } .\tag{2.9}
$$

When $r = s$ , the sharper bound $\alpha _ { s } / 2$ holds.

Proof. Suppose first that $s < r$ . Start from the random labeling defining $P _ { r } \colon$ choose $i \in G _ { r }$ uniformly, place it at label 2, and arrange the remaining identities uniformly on the other labels. Independently choose $j \in G _ { s }$ uniformly. Modify two means: raise the mean at label 2 from $\mu _ { i }$ to $m ,$ and change the mean at the label occupied by $j$ from $\mu _ { j }$ to $\mu _ { i }$ . The first change makes labels 1 and 2 tied for the largest mean, while the second replaces the omitted identity i among labels $3 , \ldots , K$ by the omitted identity $j$

We verify that the modified model has law $Q _ { s }$ . Conditional on i and $j ,$ , the identities on labels $3 , \ldots , K$ are initially a uniform arrangement of all suboptimal identities except i. Replacing $j$ by i gives a bijection from these arrangements to the arrangements of all suboptimal identities except $j$ The latter arrangement is therefore uniform and independent of i and r. Averaging over the uniform choice of $j \in G _ { s }$ gives $Q _ { s }$ . This identity-level argument also applies when distinct arms have equal means.

Adjoin the choices of $i , j$ and the random labeling to the histories under both models. Conditional on these variables, the two models difer only at label 2 and at the label occupied by $j$ . Equation (2.3) therefore bounds their conditional KL divergence by

$$
\frac { 1 } { 2 } \mathbb { E } _ { P _ { r } } \left[ \Delta _ { i } ^ { 2 } N _ { 2 } ( T \wedge M ) + ( \Delta _ { j } - \Delta _ { i } ) ^ { 2 } N _ { \mathrm { l a b e l } ( j ) } ( T \wedge M ) \right] .
$$

Forgetting the auxiliary variables can only decrease KL divergence. Since $s < r$ , the definitions of $G _ { r }$ and $G _ { s }$ give

$$
0 < \Delta _ { i } < \Delta _ { j } , \qquad \Delta _ { i } ^ { 2 } \leq 4 ^ { - r } , \qquad ( \Delta _ { j } - \Delta _ { i } ) ^ { 2 } \leq 4 ^ { - s } .
$$

Moreover, equivariance and uniform choice within the two groups give expected total sample counts $t _ { r } / n _ { r }$ for i and $t _ { s } / n _ { s }$ for $j$ . Hence

$$
\mathrm { K L } ( P _ { r } ^ { ( M ) } \| Q _ { s } ^ { ( M ) } ) \leq \frac { 1 } { 2 } \left( 4 ^ { - r } \frac { t _ { r } } { n _ { r } } + 4 ^ { - s } \frac { t _ { s } } { n _ { s } } \right) = \frac { \alpha _ { r } + \alpha _ { s } } { 2 } .
$$

When $r = s ,$ , raise only the mean at label 2 from $\mu _ { i }$ to $m .$ . The resulting marginal law is $Q _ { s }$ , and

$$
\mathrm { K L } ( P _ { s } ^ { ( M ) } \| Q _ { s } ^ { ( M ) } ) \leq \frac { 1 } { 2 } 4 ^ { - s } \mathbb { E } _ { P _ { s } } N _ { 2 } = \frac { \alpha _ { s } } { 2 } .
$$

2.3. Sample counts and entropy. Recall that under $P _ { r }$ , label 2 contains an arm chosen uniformly from $G _ { r } ,$ and

$$
\mathbb { E } _ { P _ { r } } N _ { 2 } = \alpha _ { r } 4 ^ { r } .
$$

Set $c _ { 0 } = 1 / 2 5 6$ . For $a \ge 1$ and $r \in \mathcal { U }$ with $\alpha _ { r } \leq a ,$ , define

$$
E _ { r } ( a ) = \{ T < \infty , { \widehat { i } } = 1 , c _ { 0 } 4 ^ { r } \leq N _ { 2 } \leq 4 a 4 ^ { r } \} .\tag{2.10}
$$

Lemma 2.4. If $\alpha _ { r } \leq a _ { \mathrm { ~ 2 ~ } }$ , then $P _ { r } ( E _ { r } ( a ) ) \geq 1 / 2$

Proof. Put $L = c _ { 0 } 4 ^ { r }$ and $n = \lceil L \rceil - 1$ . Since $N _ { 2 }$ is integer,

$$
N _ { 2 } < L \quad \Longleftrightarrow \quad N _ { 2 } \leq n .
$$

Fix a total horizon M. Starting from the history under $P _ { r }$ , stop recording at the first of the following times: when the algorithm stops, when it has taken M samples in total, or immediately after it selects label 2 for the $( n + 1 ) \mathrm { s t }$ time and before the corresponding reward is revealed. Record which

of these three stopping conditions occurred. The resulting history contains at most n rewards from label 2 and determines the event

$$
A _ { M } = \{ T \leq M , { \ } { \widehat { i } } = 1 , \ N _ { 2 } \leq n \} .
$$

Compare this censored history under $P _ { r }$ with the corresponding history under $Q _ { r }$ . Conditional on the selected identity $i \in G _ { r }$ and the random labeling, the two models difer only in the mean at label $2 ,$ which changes from $\mu _ { i }$ to $m$ . Each observed reward from that label contributes Gaussian KL divergence $\Delta _ { i } ^ { 2 } / 2$ , and at most n such rewards are observed. Since $\Delta _ { i } ^ { 2 } \leq 4 ^ { - r }$

$$
\mathrm { K L } ( \overline { { { P } } } _ { r , M , n } \Vert \overline { { { Q } } } _ { r , M , n } ) \leq \frac { 1 } { 2 } n 4 ^ { - r } \leq \frac { c _ { 0 } } { 2 } ,
$$

where $\overline { { P } } _ { r , M , n }$ and $\overline { { Q } } _ { r , M , n }$ denote the laws of the censored histories. The same bound holds after averaging over the selected identity and labeling, since forgetting these auxiliary variables can only decrease KL divergence.

By Lemma 2.2,

$$
Q _ { r } ( A _ { M } ) \leq Q _ { r } ( T < \infty , \widehat { i } = 1 ) \leq \delta .
$$

Pinsker’s inequality therefore gives

$$
P _ { r } ( A _ { M } ) \leq Q _ { r } ( A _ { M } ) + \sqrt { \frac { 1 } { 2 } \mathrm { K L } ( \overline { { P } } _ { r , M , n } \Vert \overline { { Q } } _ { r , M , n } ) } \leq \delta + \frac { \sqrt { c _ { 0 } } } { 2 } .
$$

Letting $M \to \infty$ and using $c _ { 0 } = 1 / 2 5 6$ , we obtain

$$
P _ { r } ( T < \infty , \stackrel {  } { i } = 1 , N _ { 2 } < c _ { 0 } 4 ^ { r } ) \le \delta + \frac { 1 } { 3 2 } .
$$

For the upper tail, (2.8) and $\alpha _ { r } \leq a$ give

$$
\mathbb { E } _ { P _ { r } } N _ { 2 } = \alpha _ { r } 4 ^ { r } \le a 4 ^ { r } .
$$

Hence Markov’s inequality yields

$$
P _ { r } ( N _ { 2 } > 4 a 4 ^ { r } ) \leq \frac { 1 } { 4 } .
$$

Finally, $P _ { r } ( T < \infty , \stackrel { \frown } { i } = 1 ) \geq 1 - \delta$ . Removing from this event the lower- and upper-tail events bounded above gives

$$
P _ { r } ( E _ { r } ( a ) ) \geq 1 - \delta - \left( \delta + \frac { 1 } { 3 2 } \right) - \frac { 1 } { 4 } = 1 - 2 \delta - \frac { 1 } { 3 2 } - \frac { 1 } { 4 } > \frac { 1 } { 2 } ,
$$

because $\delta < 0 . 1$

Lemma 2.5. For $a \geq 0$ , let

$$
N ( a ) = | \{ r \in \mathcal { U } : \alpha _ { r } \leq a \} | .
$$

Then

$$
N ( a ) \leq 3 2 a e ^ { 2 a } \qquad ( a \geq 1 ) , \qquad \sum _ { r \in \mathcal { U } } e ^ { - 8 \alpha _ { r } } \leq 1 .\tag{2.11}
$$

Proof. Fix $a \ge 1$ . If $N ( a ) = 0$ , the first bound is immediate. Otherwise let

$$
s = \operatorname* { m i n } \{ r \in \mathcal { U } : \alpha _ { r } \leq a \} .
$$

Every index counted by $N ( a )$ satisfies $r > s$ . Since both $\alpha _ { r }$ and $\alpha _ { s }$ are at most $^ { a , }$ Lemma 2.3 gives

$$
\mathrm { K L } ( P _ { r } ^ { ( M ) } \| Q _ { s } ^ { ( M ) } ) \leq a
$$

for every finite horizon M.

We use this inequality to transfer the event $E _ { r } ( a )$ from $P _ { r }$ to the common law $Q _ { s }$ . Set

$$
B _ { r , M } = E _ { r } ( a ) \cap \{ T \leq M \} , \qquad p _ { M } = P _ { r } ( B _ { r , M } ) , \qquad q _ { M } = Q _ { s } ( B _ { r , M } ) .
$$

The event $B _ { r , M }$ is determined by the history up to $T \wedge M .$ , so data processing and the binarydivergence bound

$$
\operatorname { k l } ( p , q ) \geq p \log ( 1 / q ) - \log 2
$$

give

$$
a \ge \mathrm { k l } ( p _ { M } , q _ { M } ) \ge p _ { M } \log ( 1 / q _ { M } ) - \log 2 .
$$

As $M \to \infty , p _ { M }$ increases to $P _ { r } ( E _ { r } ( a ) )$ , which satisfies

$$
P _ { r } ( E _ { r } ( a ) ) \geq { \frac { 1 } { 2 } }
$$

by Lemma 2.4, while $q _ { M }$ increases to $Q _ { s } ( E _ { r } ( a ) )$ . Therefore

$$
\log \frac { 1 } { Q _ { s } ( E _ { r } ( a ) ) } \leq 2 a + 2 \log 2 ,
$$

and hence

$$
Q _ { s } ( E _ { r } ( a ) ) \geq { \frac { 1 } { 4 } } e ^ { - 2 a } .\tag{2.12}
$$

It remains to use the fact that suficiently separated indices impose disjoint conditions on the same random variable $N _ { 2 }$ . Let

$$
d ( a ) = \lceil \log _ { 4 } ( 1 0 2 4 a ) \rceil + 1 .
$$

If $r ^ { \prime } > r$ and $r ^ { \prime } \equiv r$ (mod d(a)), then $r ^ { \prime } - r \geq d ( a )$ and

$$
c _ { 0 } 4 ^ { r ^ { \prime } } \geq c _ { 0 } 4 ^ { r + d ( a ) } > 4 a 4 ^ { r } ,
$$

because $c _ { 0 } = 1 / 2 5 6$ and $4 ^ { d ( a ) } > 1 0 2 4 a$ . Thus the intervals

$$
[ c _ { 0 } 4 ^ { r } , 4 a 4 ^ { r } ]
$$

are disjoint within each residue class modulo $d ( a )$ . The corresponding events $E _ { r } ( a )$ are therefore disjoint under the common law $Q _ { s }$ . Since each has probability at least ${ \frac { 1 } { 4 } } e ^ { - 2 a }$ , each residue class contains at most $4 e ^ { 2 a }$ indices counted by $N ( a )$ . Consequently,

$$
N ( a ) \leq 4 d ( a ) e ^ { 2 a } .
$$

Finally,

$$
d ( a ) \leq \log _ { 4 } a + 7 \leq 8 a \qquad ( a \geq 1 ) ,
$$

which proves

$$
N ( a ) \leq 3 2 a e ^ { 2 a } .
$$

By (2.7), every $\alpha _ { r }$ is greater than 1, so $N ( a ) = 0$ for $a \leq 1$ . Using

$$
e ^ { - 8 \alpha _ { r } } = \int _ { \alpha _ { r } } ^ { \infty } 8 e ^ { - 8 a } \mathrm { d } a
$$

and summing over r gives

$$
\sum _ { r \in \mathcal { U } } e ^ { - 8 \alpha _ { r } } = \int _ { 1 } ^ { \infty } 8 e ^ { - 8 a } N ( a ) \mathrm { d } a \le 2 5 6 \int _ { 1 } ^ { \infty } a e ^ { - 6 a } \mathrm { d } a = \frac { 4 4 8 } { 9 } e ^ { - 6 } < 1 .
$$

Proof of Proposition 2.1. Recall that T denotes the stopping time of the symmetrized algorithm. Set

$$
z = \sum _ { r \in \mathcal { U } } e ^ { - 8 \alpha _ { r } } .
$$

By Lemma 2.5, $0 < z \le 1$ , so

$$
q _ { r } = \frac { e ^ { - 8 \alpha _ { r } } } { z } , \qquad r \in \mathcal { U } ,
$$

defines a probability vector. Gibbs’s inequality gives

$$
0 \leq \sum _ { r \in \mathcal { U } } p _ { r } \log \frac { p _ { r } } { q _ { r } } = - \operatorname { E n t } ( I ) + 8 \sum _ { r \in \mathcal { U } } p _ { r } \alpha _ { r } + \log z .
$$

Since log $z \leq 0$ , it follows that

$$
\operatorname { E n t } ( I ) \leq 8 \sum _ { r \in \mathcal { U } } p _ { r } \alpha _ { r } .\tag{2.13}
$$

We now convert the right side into an expected sample count. Using $p _ { r } = H _ { r } / H$ and $\alpha _ { r } = t _ { r } / ( n _ { r } 4 ^ { r } )$ gives

$$
H \sum _ { r \in \mathcal { U } } p _ { r } \alpha _ { r } = \sum _ { r \in \mathcal { U } } \frac { H _ { r } } { n _ { r } 4 ^ { r } } t _ { r } .
$$

For every $i \in G _ { r }$ , the inequality $\Delta _ { i } > 2 ^ { - ( r + 1 ) }$ implies $\Delta _ { i } ^ { - 2 } < 4 ^ { r + 1 }$ . Hence

$$
H _ { r } = \sum _ { i \in G _ { r } } \Delta _ { i } ^ { - 2 } \leq 4 n _ { r } 4 ^ { r } ,
$$

and therefore

$$
H \sum _ { r \in \mathcal { U } } p _ { r } \alpha _ { r } \leq 4 \sum _ { r \in \mathcal { U } } t _ { r } = 4 \sum _ { i \neq \ast } u _ { i } \leq 4 \mathbb { E } _ { I } T .
$$

Combining this estimate with (2.13) yields

$$
\mathbb { E } _ { I } T \geq \frac { H \operatorname { E n t } ( I ) } { 3 2 } .
$$

The per-arm change-of-measure bound (2.4) also gives

$$
\mathbb { E } _ { I } T \geq 2 b _ { \delta } H .
$$

Finally, (2.2) identifies $\mathbb { E } _ { I } T$ with the average expected cost of the original algorithm over all relabelings of I. Since that algorithm was arbitrary, taking the infimum in (1.2) proves (2.1). □

## 3. The instance-wise upper bound

The following sub-Gaussian proposition implies the upper bound in Theorem 1.1 when restricted to $I \in { \mathcal { G } } _ { K }$

Proposition 3.1. For every $I \in { \cal S } _ { K }$ and $0 < \delta < 0 . 1$ , there exists an algorithm A that is δ-correct on $\boldsymbol { \mathcal { S } } _ { K }$ and satisfies

$$
\mathbb { E } _ { \pi I } T _ { A } \leq C H ( \log ( 1 / \delta ) + \operatorname { E n t } ( I ) ) \qquad ( \pi \in S _ { K } ) .
$$

The constant C is absolute.

The remainder of this section proves Proposition 3.1. We first describe the target-instancedependent parameters and a single trial, then prove its correctness and expected-cost bounds, and finally repeat the trials and add an auxiliary procedure to ensure almost-sure stopping on every instance in $\scriptstyle { S _ { K } }$

It sufices to prove the upper bounds for $0 < \delta < 0 . 0 1$ . For $0 . 0 1 \leq \delta < 0 . 1$ , run the corresponding algorithm with error parameter 0.005. It remains δ-correct, while

$$
\log 2 0 0 \leq \frac { \log 2 0 0 } { \log 1 0 } \log ( 1 / \delta ) ,
$$

so the sample bound changes only by an absolute constant factor.

3.1. Instance-specific parameters and a trial. Fix a target instance $I \in { \cal S } _ { K }$ and assume $0 < \delta < 0 . 0 1$ . The algorithm constructed in this section may depend on the target’s unlabeled collection of means, but not on the label of its optimal arm. For $r \in \mathcal { U } .$ , set

$$
n _ { r } = | G _ { r } | , \qquad w _ { r } = n _ { r } 4 ^ { r } , \qquad W = \sum _ { r \in \mathcal { U } } w _ { r } , \qquad q _ { r } = \frac { w _ { r } } { W } ,
$$

and let

$$
R = \operatorname* { m a x } \mathcal { U } , \qquad h = 2 ^ { - R } .
$$

For $i \in G _ { r } .$

$$
4 ^ { r } \leq \Delta _ { i } ^ { - 2 } < 4 ^ { r + 1 } ,
$$

so summing over $G _ { r }$ gives $w _ { r } \leq H _ { r } < 4 w _ { r }$ . Consequently,

$$
w _ { r } \leq H _ { r } < 4 w _ { r } , \qquad W \leq H < 4 W , \qquad g \leq h < 2 g , \qquad h ^ { - 2 } \leq W .\tag{3.1}
$$

Here the bounds involving h follow because an arm with gap g belongs to $G _ { R } ,$ while $G _ { R }$ is nonempty and therefore $W \geq w _ { R } \geq 4 ^ { R } = h ^ { - 2 }$

We will allocate the mixture weights according to $q = \left( q _ { r } \right)$ . To compare the entropy of these weights with Ent(I), define, for a finite nonnegative vector v,

$$
\Phi ( v ) = \left( \sum _ { r } v _ { r } \right) \log \Big ( \sum _ { r } v _ { r } \Big ) - \sum _ { r } v _ { r } \log v _ { r } , \qquad 0 \log 0 = 0 .\tag{3.2}
$$

If $\begin{array} { r } { V = \sum _ { r } v _ { r } > 0 } \end{array}$ , then

$$
\Phi ( v ) = V \operatorname { E n t } ( v / V ) .
$$

Moreover, $\Phi$ is homogeneous of degree one and coordinatewise nondecreasing: at a positive coordinate,

$$
\frac { \partial \Phi } { \partial v _ { r } } = \log \frac { \sum _ { s } v _ { s } } { v _ { r } } \geq 0 ,
$$

and the same monotonicity at zero follows by continuity. Applying these properties to the coordinatewise inequalities $w _ { r } \leq H _ { r } \leq 4 w _ { r }$ gives

$$
W \operatorname { E n t } ( q ) = \Phi ( w ) \leq \Phi ( ( H _ { r } ) _ { r } ) = H \operatorname { E n t } ( I ) \leq \Phi ( 4 w ) = 4 W \operatorname { E n t } ( q ) .\tag{3.3}
$$

Choose a rational threshold

$$
\tau \in ( m - g / 2 , m - g / 3 ) \subset ( 0 , 1 ) .\tag{3.4}
$$

On the target instance, the optimal mean m lies above $\tau ,$ while every suboptimal mean is at most $m - g < \tau$ . The algorithm stores the finite set $u ,$ the integers $( n _ { r } ) _ { r \in \mathcal { U } }$ , and the rational number $\tau .$ These parameters have a finite description and do not identify the optimal arm’s label.

Fix an error allowance $0 < \eta < 0 . 0 1$ . A trial attempts to eliminate arms whose means are below τ while retaining any arm whose mean is at least $\tau .$ For each arm $i ,$ after observing t samples $X _ { i , 1 } , \ldots , X _ { i , t }$ , define

$$
M _ { i } ( t ) = \sum _ { r \in \mathcal { U } } q _ { r } \exp \left( - \lambda _ { r } \sum _ { \ell = 1 } ^ { t } ( X _ { i , \ell } - \tau ) - \frac { \lambda _ { r } ^ { 2 } t } { 2 } \right) , \qquad \lambda _ { r } = 2 ^ { - r - 2 } .\tag{3.5}
$$

The component indexed by $r$ is tuned to departures below τ on the scale $2 ^ { - r }$ . Under the hypothesis $\mu _ { i } \geq \tau .$ , each component, and hence their mixture, is a nonnegative supermartingale.

Initialize all arms as active. Repeatedly perform a complete sweep, drawing one new sample from each arm active at the start of the sweep. At the end of the sweep, simultaneously discard every arm i whose updated statistic satisfies $M _ { i } ( t ) \geq 1 / \eta$ . If no arm remains, end the trial without an answer. If exactly one arm c remains, take

$$
n _ { \eta } = \left\lceil 2 8 8 h ^ { - 2 } \log ( 1 / \eta ) \right\rceil
$$

fresh samples from it. Output c if their average exceeds $\tau + h / 1 2 ;$ otherwise end the trial without an answer. If every input mean lies below τ , this confirmation step bounds the probability that the trial returns an arm by η.

Finally, impose the deterministic sample limit

$$
B _ { \eta } = \left\lceil 1 6 0 0 0 W \left[ \log ( 1 / \eta ) + \mathrm { E n t } ( q ) + 1 \right] \right\rceil .\tag{3.6}
$$

The limit includes both elimination and confirmation samples. If the next sample would make the total exceed $B _ { \eta } .$ , end the trial without an answer.

3.2. Correctness and cost of a trial. We first show that the trial has error probability at most η on every input instance, even though its parameters were chosen from the fixed target instance. Let an input arm have mean $\mu \geq \tau$ . For each $r \in \mathcal { U }$ and a fresh sample X from this arm, the sub-Gaussian assumption gives

$$
\begin{array} { r } { \mathbb { E } e ^ { - \lambda _ { r } ( X - \tau ) - \lambda _ { r } ^ { 2 } / 2 } = e ^ { - \lambda _ { r } ( \mu - \tau ) } \mathbb { E } e ^ { - \lambda _ { r } ( X - \mu ) - \lambda _ { r } ^ { 2 } / 2 } \le e ^ { - \lambda _ { r } ( \mu - \tau ) } \le 1 . } \end{array}
$$

The same inequality holds conditionally on all preceding samples. Hence each exponential process in (3.5) is a nonnegative supermartingale. Their weighted sum $M _ { i }$ is therefore also a nonnegative supermartingale, with $\begin{array} { r } { M _ { i } ( 0 ) = \sum _ { r } q _ { r } = 1 } \end{array}$ . The maximal inequality for nonnegative supermartingales gives

$$
\mathbb { P } \left( \operatorname* { s u p } _ { t \geq 0 } M _ { i } ( t ) \geq \frac { 1 } { \eta } \right) \leq \eta ;
$$

see [11, theorem 3.9]. Thus an arm with mean at least $\tau$ is eliminated with probability at most $\eta .$ Now consider an arbitrary input instance. If its optimal mean is at least $\tau _ { : }$ , the trial can return a suboptimal arm only if it first eliminates the optimal arm. The preceding bound shows that this event has probability at most $\eta .$ If every arm has mean below $\tau ,$ condition on the history up to the confirmation step and on the remaining arm c. The confirmation samples are fresh, and $\mu _ { c } < \tau ,$ , so (B.1) gives

$$
\mathbb { P } \left( \overline { { X } } _ { c } > \tau + \frac { h } { 1 2 } \bigg | \mathrm { p r e c e d i n g ~ h i s t o r y } \right) \leq \exp \left( - \frac { n _ { \eta } h ^ { 2 } } { 2 8 8 } \right) \leq \eta .
$$

The deterministic sample limit can only end the trial without an answer. Therefore, on every instance in $\scriptstyle { S _ { K } }$ , the probability that the trial returns an incorrect arm is at most $\eta .$

We now return to the target instance I and bound the expected cost. Temporarily remove the deterministic sample limit, retaining all other parts of the trial. For each arm, generate an infinite independent sample sequence and define

$$
U _ { i } = \operatorname* { i n f } \left\{ t \geq 1 : M _ { i } ( t ) \geq \frac { 1 } { \eta } \right\} ,
$$

with $U _ { i } = \infty$ if the threshold is never reached. This definition allows us to bound the cost even on paths where the optimal arm is erroneously eliminated.

Fix a suboptimal target arm $i \in G _ { r }$ and put

$$
d _ { i } = \tau - \mu _ { i } .
$$

Since $m - \tau < g / 2 \le \Delta _ { i } / 2$

$$
d _ { i } = \Delta _ { i } - ( m - \tau ) > \frac { \Delta _ { i } } { 2 } .
$$

The definition of $G _ { r }$ and the choice $\lambda _ { r } = 2 ^ { - r - 2 }$ also give

$$
\frac { \Delta _ { i } } { 4 } \leq \lambda _ { r } < \frac { \Delta _ { i } } { 2 } < d _ { i } .
$$

The logarithm of the unweighted rth exponential component after t samples is

$$
a _ { i } t - \lambda _ { r } \sum _ { \ell = 1 } ^ { t } ( X _ { i , \ell } - \mu _ { i } ) , \qquad a _ { i } = \lambda _ { r } d _ { i } - \frac { \lambda _ { r } ^ { 2 } } { 2 } .
$$

The preceding inequalities imply

$$
a _ { i } = \lambda _ { r } \left( d _ { i } - \frac { \lambda _ { r } } { 2 } \right) \geq \frac { \Delta _ { i } ^ { 2 } } { 1 6 } , \qquad \frac { a _ { i } } { \lambda _ { r } } = d _ { i } - \frac { \lambda _ { r } } { 2 } \geq \frac { \Delta _ { i } } { 4 } .
$$

Set

$$
b _ { r } = \log { \frac { 1 } { \eta q _ { r } } } .
$$

If $U _ { i } > t ,$ then $M _ { i } ( t ) < 1 / \eta .$ so its rth component satisfies

$$
\exp \left( a _ { i } t - \lambda _ { r } \sum _ { \ell = 1 } ^ { t } ( X _ { i , \ell } - \mu _ { i } ) \right) < \frac { 1 } { \eta q _ { r } } .
$$

Consequently, for every integer $t \geq 2 b _ { r } / a _ { i }$

$$
\begin{array} { r l r } {  { \mathbb { P } ( U _ { i } > t ) \le \mathbb { P } ( \lambda _ { r } \displaystyle \sum _ { \ell = 1 } ^ { t } ( X _ { i , \ell } - \mu _ { i } ) > a _ { i } t - b _ { r } ) } } \\ & { } & \\ & { } & { \leq \exp ( - \frac { ( a _ { i } t - b _ { r } ) ^ { 2 } } { 2 \lambda _ { r } ^ { 2 } t } ) } \\ & { } & { \leq \exp ( - \frac { a _ { i } ^ { 2 } t } { 8 \lambda _ { r } ^ { 2 } } ) \leq \exp ( - \frac { \Delta _ { i } ^ { 2 } t } { 1 2 8 } ) . } \end{array}
$$

Here the penultimate inequality uses $a _ { i } t - b _ { r } \geq a _ { i } t / 2$ , and the last uses $a _ { i } / \lambda _ { r } \geq \Delta _ { i } / 4$

Let $t _ { 0 } = \lceil 2 b _ { r } / a _ { i } \rceil$ . Summing the tail probabilities gives

$$
\mathbb { E } U _ { i } \leq t _ { 0 } + \sum _ { t = t _ { 0 } } ^ { \infty } e ^ { - \Delta _ { i } ^ { 2 } t / 1 2 8 } \leq 1 3 0 \Delta _ { i } ^ { - 2 } ( b _ { r } + 1 ) .
$$

Indeed, $a _ { i } \geq \Delta _ { i } ^ { 2 } / 1 6$ gives $t _ { 0 } \leq 3 2 b _ { r } \Delta _ { i } ^ { - 2 } + 1$ , while $\Delta _ { i } \leq 1$ gives

$$
\sum _ { t = t _ { 0 } } ^ { \infty } e ^ { - \Delta _ { i } ^ { 2 } t / 1 2 8 } \leq \frac { 1 } { 1 - e ^ { - \Delta _ { i } ^ { 2 } / 1 2 8 } } \leq 1 2 9 \Delta _ { i } ^ { - 2 } .
$$

Summing over the suboptimal arms and using $b _ { r } = \log ( 1 / \eta ) + \log ( 1 / q _ { r } )$ yields

$$
\begin{array} { r l r } {  { \sum _ { i \not = * } \mathbb { E } U _ { i } \le 1 3 0 \sum _ { r \in \mathcal { U } } H _ { r } [ \log ( 1 / \eta ) + \log ( 1 / q _ { r } ) + 1 ] } } \\ & { } & { \le 5 2 0 \sum _ { r \in \mathcal { U } } w _ { r } [ \log ( 1 / \eta ) + \log ( 1 / q _ { r } ) + 1 ] } \\ & { } & { = 5 2 0 W [ \log ( 1 / \eta ) + \mathrm { E n t } ( q ) + 1 ] . } \end{array}
$$

The second inequality uses $H _ { r } \leq 4 w _ { r }$ , while the last equality uses $q _ { r } = w _ { r } / W .$

It remains to account for samples from the optimal arm and for the confirmation step. During the elimination phase, suboptimal arm i receives at most $U _ { i }$ samples. If the optimal arm receives t elimination samples, then in its final sweep at least one suboptimal arm is also active and receives its tth sample. Hence $U _ { i } \geq t$ for that arm, so the optimal arm receives at most ma $\mathrm { x } _ { i \neq * } U _ { i }$ samples before the elimination phase ends. This remains true if the optimal arm is itself eliminated. Hence the elimination cost is bounded pathwise by

$$
\sum _ { i \neq * } U _ { i } + \operatorname* { m a x } _ { i \neq * } U _ { i } \leq 2 \sum _ { i \neq * } U _ { i } .
$$

The confirmation step uses at most $n _ { \eta }$ further samples. Since $h ^ { - 2 } \leq W$ , the expected cost of the uncapped trial is therefore at most

$$
2 0 0 0 W \left[ \log ( 1 / \eta ) + \operatorname { E n t } ( q ) + 1 \right] .
$$

Comparing this with (3.6), Markov’s inequality shows that the probability of reaching the deterministic sample limit is at most $1 / 8$

Finally, on the target instance,

$$
m - \tau > { \frac { g } { 3 } } > { \frac { h } { 6 } } .
$$

Thus the confirmation threshold $\tau + h / 1 2$ lies at least $h / 1 2$ below $m .$ , and (B.1) gives

$$
\mathbb { P } \left( \overline { { X } } _ { * } \leq \tau + \frac { h } { 1 2 } \right) \leq e ^ { - n _ { \eta } h ^ { 2 } / 2 8 8 } \leq \eta .
$$

The bounds on $\mathbb { E } U _ { i }$ imply that $U _ { i } < \infty$ almost surely for every suboptimal arm i. If the optimal arm is never eliminated, the uncapped elimination phase therefore ends with that arm as the sole survivor. Therefore, unless the optimal arm is eliminated, the sample limit is reached, or the confirmation fails, the trial returns the optimal arm. Its success probability is at least

$$
1 - \eta - \frac { 1 } { 8 } - \eta = 1 - \frac { 1 } { 8 } - 2 \eta > \frac { 3 } { 4 } .
$$

All estimates are uniform over relabelings of the target instance.

3.3. Repetition and almost-sure stopping. Run independent trials with error levels

$$
\eta _ { j } = \delta 2 ^ { - j - 2 } , \qquad j \geq 0 ,
$$

and stop at the first trial that returns an arm. On every input instance, the probability that some trial returns an incorrect arm is at most

$$
\sum _ { j \geq 0 } \eta _ { j } = \frac { \delta } { 2 } .
$$

For any relabeling of the target instance, each trial returns the optimal arm with probability at least $3 / 4$ . Independence therefore implies that the probability of reaching trial $j$ is at most $4 ^ { - j }$ Moreover,

$$
B _ { \eta _ { j } } \le C W \bigl [ \log ( 1 / \delta ) + \mathrm { E n t } ( q ) + j + 1 \bigr ] .
$$

Consequently, the expected number of samples used by the sequence of trials is at most

$$
\sum _ { j \geq 0 } 4 ^ { - j } B _ { \eta _ { j } } \leq C W \big [ \log ( 1 / \delta ) + \mathrm { E n t } ( q ) + 1 \big ] .
$$

The sequence of trials need not stop almost surely when its fixed parameters do not describe the input instance. To ensure almost-sure stopping on every instance, run it in parallel with an independent auxiliary procedure that is $\delta / 2 \mathrm { - c o r r e c t }$ on $\scriptstyle { S _ { K } }$ . Alternate between the next sample requested by the trial sequence and the next sample requested by the auxiliary procedure, and return the first arm produced by either procedure. Before the trial sequence returns an arm, the combined algorithm uses at most twice as many samples, up to one additional sample. Thus its expected cost on any relabeling of the target instance remains bounded by

$$
C W [ \log ( 1 / \delta ) + \operatorname { E n t } ( q ) + 1 ] .
$$

We construct the auxiliary procedure using simultaneous confidence intervals. Put $\beta = \delta / 2$ . At stage $k \geq 0$ , draw $2 ^ { k }$ fresh samples from each arm and let $\overline { { \boldsymbol X } } _ { i , \boldsymbol k }$ be their average. Form the intervals

$$
\left[ \overline { { X } } _ { i , k } - c _ { k } , \overline { { X } } _ { i , k } + c _ { k } \right] , \qquad c _ { k } = \sqrt { 2 ^ { 1 - k } \log \frac { 8 K ( k + 1 ) ^ { 2 } } { \beta } } .
$$

Stop and return arm i if its lower endpoint exceeds the upper endpoint of every other arm. For each arm and stage, the sub-Gaussian tail bound gives

$$
\mathbb { P } \big ( | \overline { { X } } _ { i , k } - \mu _ { i } | > c _ { k } \big ) \le \frac { \beta } { 4 K ( k + 1 ) ^ { 2 } } .
$$

A union bound over all arms and stages shows that the probability of any confidence-interval failure is at most

$$
\frac { \beta } { 4 } \sum _ { k \geq 0 } \frac { 1 } { ( k + 1 ) ^ { 2 } } = \frac { \beta \pi ^ { 2 } } { 2 4 } < \beta .
$$

Whenever all intervals contain their respective means, any arm returned by the procedure is optimal.   
The auxiliary procedure is therefore $\beta \mathrm { . }$ -correct.

It remains to verify almost-sure stopping. For every fixed $e > 0$

$$
\begin{array} { r } { \mathbb { P } \big ( | \overline { { X } } _ { i , k } - \mu _ { i } | > e \big ) \leq 2 \exp ( - 2 ^ { k - 1 } e ^ { 2 } ) , } \end{array}
$$

and the right side is summable in k. Applying Borel–Cantelli for each arm and for $e = 1 , { \frac { 1 } { 2 } } , { \frac { 1 } { 3 } } , \ldots$ gives

$$
\overline { { { X } } } i , k \longrightarrow \mu _ { i } \qquad \mathrm { a l m o s t ~ s u r e l y ~ f o r ~ e v e r y ~ a r m ~ } i .
$$

Since $c _ { k } \to 0$ and the optimal arm is unique, its lower endpoint eventually exceeds every suboptima arm’s upper endpoint. Hence the auxiliary procedure, and therefore the combined algorithm, stops almost surely on every instance in $\scriptstyle { S _ { K } }$

The two component procedures have error probabilities at most $\delta / 2$ , so the combined algorithm is δ-correct. Finally, (3.1) and (3.3) give

$$
W \leq H , \qquad W \operatorname { E n t } ( q ) \leq H \operatorname { E n t } ( I ) .
$$

Since log $( 1 / \delta ) >$ log 100, the additive $W$ term can be absorbed into $H \log ( 1 / \delta )$ . Thus, uniformly over all relabelings of the target instance,

$$
\begin{array} { r } { \mathbb { E } _ { I } T \le C H \bigl [ \log ( 1 / \delta ) + \mathrm { E n t } ( I ) \bigr ] , } \end{array}
$$

which proves Proposition 3.1.

## 4. The uniform algorithm

We construct the algorithm in Theorem 1.2 for $\boldsymbol { \mathcal { S } } _ { K }$ by modifying the Entropy-Elimination and Complexity-Guessing procedures of Chen, Li and Qiao [3, algorithms 4–5]. We retain their geometrically increasing stage budgets, two cumulative budget tests, and elimination thresholds. We adjust the maximum number of rounds and fraction-threshold increments to account for all sampling costs, and impose a deterministic sample limit at each stage. The analysis uses the comparison process in Section 5. Each stage starts with all K arms. One cumulative sum limits the total elimination error budget, and the other limits its expected sampling cost.

4.1. Sampling subroutines. By the reduction in Section 3, it sufices to consider $0 < \delta < 0 . 0 1$ Each subroutine call uses samples independent of the history before the call. Consequently, the guarantees below hold conditionally on that history and on the possibly random inputs to the call. They apply to 1-sub-Gaussian reward laws and use only the sample-mean bound (B.1) and independence between sample blocks.

We refer to Mean and Fraction calls made directly by the main algorithm as outer tests, to distinguish them from calls made internally by other subroutines.

Median and Fraction require nonempty finite input sets, whereas Eliminate also accepts the empty set. Throughout, accuracy parameters and diferences between mean thresholds lie in (0, 1], error levels lie in (0, 0.1), and diferences between fraction thresholds lie in (0, 0.1]. These ranges cover every call made by the algorithm. The implementations and proofs are given in Appendix B. Here and below, C denotes an absolute constant.

(i) Mean estimation. The call $\mathrm { M e a n } ( i , e , \alpha )$ uses ${ \cal O } ( e ^ { - 2 } \log ( 1 / \alpha ) )$ ) samples deterministically and returns an estimate of $\mu _ { i }$ whose error is at most $e ,$ except with probability α.

(ii) Median elimination. The call Median $( S , e )$ uses $O ( | S | e ^ { - 2 } )$ samples deterministically and, with conditional probability at least 0.99, returns an arm $i \in S$ satisfying

$$
\mu _ { i } \geq \operatorname* { m a x } _ { j \in S } \mu _ { j } - e .
$$

(iii) Fraction testing. For $l < u$ and $\theta _ { - } < \theta _ { + }$ , the call Fraction $( S , l , u , \theta _ { - } , \theta _ { + } , \alpha )$ has deterministic cost at most

$$
C ( u - l ) ^ { - 2 } \log ( 1 / \alpha ) ( \theta _ { + } - \theta _ { - } ) ^ { - 2 } \log \frac { 1 } { \theta _ { + } - \theta _ { - } } .
$$

Except with probability α, its output obeys the following implications:

$$
{ \mathrm { T r u e } } \implies | \{ i \in S : \mu _ { i } < u \} | > \theta _ { - } | S | ,
$$

$$
{ \mathrm { F a l s e ~ } } \Longrightarrow | \{ i \in S : \mu _ { i } < l \} | < \theta _ { + } | S | .
$$

Fraction is not required to estimate either proportion. It need only return an answer whose corresponding implication is valid; on some inputs, both answers are valid.

(iv) Elimination. The call Eliminat $\mathrm { e } ( S , l , u , \alpha )$ returns a subset $S ^ { \prime } \subseteq S$ and terminates almost surely. If $S = \emptyset$ , it returns immediately. Its conditional expected cost is at most

$$
C | S | ( u - l ) ^ { - 2 } \log ( 1 / \alpha ) .
$$

If $S \neq \emptyset$ , fix, before the call, a designated arm $i ^ { \dag } \in S$ attaining $\operatorname* { m a x } _ { i \in S } \mu _ { i }$ . Except with probability α, the returned set simultaneously satisfies

$$
| \{ i \in S ^ { \prime } : \mu _ { i } < l \} | \le 0 . 1 | S ^ { \prime } | ,
$$

and, if $\mu _ { i ^ { \dagger } } \geq u$ , it contains $i ^ { \dagger }$ . When $S = \emptyset$ , the displayed inequality holds deterministically. In addition, for any k arms designated before the call whose means are all at least $u ,$ the probability that all k are removed is at most $\alpha ^ { k }$

We use a product bound to control the joint loss of several good arms across stages. Conditional on the inputs and the history before an Eliminate call, generate independent estimation blocks in advance for every arm and every internal stage. If an arm has mean at least $u ,$ then it can be removed only if one of its own estimates is inaccurate. The error budgets for that arm sum to at most $\alpha / 1 0$ , so this event has probability at most $\alpha / 1 0$ . Because the estimation blocks are independent across arms, the probability that k fixed arms of mean at least u are all removed is at most

$$
\left( { \frac { \alpha } { 1 0 } } \right) ^ { k } \leq \alpha ^ { k } .
$$

This conclusion does not require the Fraction calls to be correct: an incorrect Fraction answer may cause an unnecessary deletion step, but no such step removes an arm of mean at least u when that arm’s estimate is accurate. This is the argument of [3, lemma B.9]. The full construction, including its behavior on the empty set, is given in Appendix B.4.

4.2. Stages and stopping rules. Let J and M be the absolute constants specified in Lemmas 6.1 and 6.2, respectively. After fixing them, choose a suficiently small absolute constant $c > 0$ and set

$$
\eta = c \delta , \qquad \rho = \eta ^ { 2 } , \qquad a = \log ( 1 / \eta ) , \qquad B _ { t } = 1 0 0 ^ { t } , \qquad L _ { t } = a + \log ( t + 1 ) , \qquad t = 1 , 2 , \ldots .
$$

Each stage starts with all K arms and uses samples and internal randomness independent of every other stage. We say that a stage accepts if it returns an arm and rejects if it ends without doing so.

For stage $t ,$ set

$$
R _ { t } = \operatorname* { m a x } \{ 0 , \lfloor \log _ { 4 } ( B _ { t } / L _ { t } ) \rfloor - J \} .
$$

If $R _ { t } = 0$ , reject immediately. Otherwise, write $B = B _ { t }$ and $R = R _ { t }$ , and initialize

$$
S _ { 1 } = [ K ] , \qquad U _ { 1 } = V _ { 1 } = 0 .
$$

The variables $U _ { r }$ and $V _ { r }$ are running totals used by the two budget checks. At the start of round $r ,$

$$
U _ { r } = 4 \sum _ { \substack { \mathrm { E l i m i n a t e ~ w a s ~ c a l l e d ~ i n ~ r o u n d ~ } k } } w _ { k } , \qquad V _ { r } = \sum _ { \substack { \mathrm { r o u n d ~ } k \mathrm { ~ p a s s e d ~ b o t h ~ b u d g e t ~ c h e c k s ~ } } } z _ { k } .
$$

They do not record the number of samples used, since that quantity is controlled by the separate deterministic limit involving $M .$

Define the fraction thresholds by

$$
\theta _ { 0 } = 0 . 3 , \qquad \theta _ { r } = \theta _ { r - 1 } + { \frac { 1 } { 1 0 ( R + 1 - r ) ^ { 2 } } } , \qquad 1 \leq r \leq R .
$$

They are increasing and satisfy

$$
\theta _ { R } = 0 . 3 + \frac { 1 } { 1 0 } \sum _ { k = 1 } ^ { R } \frac { 1 } { k ^ { 2 } } < 0 . 3 + \frac { \pi ^ { 2 } } { 6 0 } < 0 . 5 .
$$

Thus $\theta _ { r } \in [ 0 . 3 , 0 . 5 )$ , and $v < r$ implies $\theta _ { v } \leq \theta _ { r - 1 }$

At the beginning of round $r ,$ return the unique arm in $S _ { r } \mathrm { ~ i f ~ } | S _ { r } | = 1$ . Otherwise, reject if $S _ { r } = \varnothing$ or $r > R$ . If the stage continues, set

$$
\varepsilon _ { r } = 2 ^ { - r } , \qquad w _ { r } = | S _ { r } | 4 ^ { r } .
$$

The first budget check rejects if

$$
\begin{array} { r } { U _ { r } + 4 w _ { r } \ge B . } \end{array}
$$

If it passes, then $w _ { r } < B / 4$ , and we may define the positive quantity

$$
z _ { r } = w _ { r } \log { \frac { B } { \eta w _ { r } } } .
$$

The second budget check rejects if

$$
V _ { r } + z _ { r } \ge 1 0 0 B .
$$

If both checks pass, perform the following operations:

(1) Update the second budget by setting

$$
V _ { r + 1 } = V _ { r } + z _ { r } , ~ \alpha _ { t , r } = \frac { \eta } { 5 0 t ^ { 2 } r ^ { 2 } } .
$$

(2) Call

$$
\widehat { i } _ { r } = \mathrm { M e d i a n } ( S _ { r } , \varepsilon _ { r } / 8 ) , \qquad \widehat { \mu } _ { r } = \mathrm { M e a n } ( \widehat { i } _ { r } , \varepsilon _ { r } / 8 , \alpha _ { t , r } ) .
$$

(3) Let $F _ { r }$ be the output of

$$
\begin{array} { r } { \mathrm { F r a c t i o n } \left( S _ { r } , \widehat { \mu } _ { r } - \frac { 7 } { 4 } \varepsilon _ { r } , \widehat { \mu } _ { r } - \frac { 9 } { 8 } \varepsilon _ { r } , \theta _ { r - 1 } , \theta _ { r } , \alpha _ { t , r } \right) . } \end{array}
$$

(4) If $F _ { r } =$ True, set

$$
\beta _ { r } = \frac { 4 w _ { r } \rho } { B }
$$

and call

$$
\begin{array} { r } { S _ { r + 1 } = \mathrm { E l i m i n a t e } \left( S _ { r } , \widehat { \mu } _ { r } - \frac { 3 } { 4 } \varepsilon _ { r } , \widehat { \mu } _ { r } - \frac { 5 } { 8 } \varepsilon _ { r } , \beta _ { r } \right) . } \end{array}
$$

$$
S _ { r + 1 } = S _ { r } , \qquad U _ { r + 1 } = U _ { r } .
$$

Then set $U _ { r + 1 } = U _ { r } + 4 w _ { r } . \mathrm { ~ I f ~ } F _ { r } = \mathrm { F a l s e }$ , make no elimination call and set

Maintain a counter for the samples used within the stage. Before taking any sample, including one requested by a subroutine, reject the stage if that sample would make the counter exceed $\lceil M B \rceil$ After a rejection, begin stage $t + 1$ with fresh samples and randomness. If the stage returns an arm, terminate the algorithm and output that arm.

The singleton check precedes the condition $r > R$ . Consequently, if round R produces a singleton $S _ { R + 1 }$ , the algorithm returns that arm at the beginning of round $R + 1$ ; otherwise, it rejects at tha point.

## 5. Progress and error control

Fix an instance $I \in S _ { K }$ , with optimal mean m. We analyze the algorithm through a comparison process in which the outer Mean and Fraction outputs are made valid by construction. Median and Eliminate are not corrected. Successful Median calls will control the sizes of the active sets, while invalid Eliminate calls will be handled separately. In particular, we never condition on all future calls being valid.

5.1. A comparison process. For an outer Mean call on arm i with accuracy $e ,$ replace its output by its projection onto

$$
[ \mu _ { i } - e , \mu _ { i } + e ] .
$$

Thus the corrected estimate always satisfies the advertised accuracy guarantee and agrees with the original output whenever that output is accurate.

For an outer Fraction call with parameters $S , l , u , \theta _ { - } , \theta _ { + }$ , call True valid if

$$
| \{ i \in S : \mu _ { i } < u \} | > \theta _ { - } | S |
$$

and False valid if

$$
| \{ i \in S : \mu _ { i } < l \} | < \theta _ { + } | S | .
$$

At least one answer is valid. Indeed, if True is invalid, then

$$
| \{ i \in S : \mu _ { i } < l \} | \leq | \{ i \in S : \mu _ { i } < u \} | \leq \theta _ { - } | S | < \theta _ { + } | S | ,
$$

so False is valid. Replace an invalid Fraction output by the other answer, leaving a valid output unchanged.

These corrections are measurable functions of the current inputs, outputs, and true means, and do not use future samples. They are introduced only for the analysis. They do not alter the samples already drawn or the cost of the corrected call, and the outputs of Median and Eliminate are never corrected.

Couple the actual and comparison processes using the same randomness until their first disagreement. Such a disagreement can occur only when an outer Mean or Fraction output is invalid. There are at most two such calls in each round, each with conditional failure probability $\alpha _ { t , r } = \eta / ( 5 0 t ^ { 2 } r ^ { 2 } )$ Hence

$$
\mathbb { P } ( \mathrm { t h e ~ t w o ~ p r o c e s s e s ~ e v e r ~ d i s a g r e e } ) \le 2 \sum _ { t , r \ge 1 } \frac { \eta } { 5 0 t ^ { 2 } r ^ { 2 } } \le C \eta .\tag{5.1}
$$

For a fixed stage t, the same argument gives a bound of $C \eta / t ^ { 2 }$ . We will not condition on agreement of the two processes. Because every Median call uses fresh samples, its conditional failure probability in the comparison process remains at most

$$
\kappa = 0 . 0 1 .
$$

For a call Eliminate ${ \bf \nabla } \cdot ( S , l , u , \alpha )$ , designate before the call one arm attaining $\operatorname* { m a x } _ { i \in S } \mu _ { i }$ . Call the returned set $S ^ { \prime }$ valid if

$$
| \{ i \in S ^ { \prime } : \mu _ { i } < l \} | \leq 0 . 1 | S ^ { \prime } |
$$

and, whenever the designated arm has mean at least $u ,$ that arm belongs to $S ^ { \prime }$

We first record why the true optimal arm remains active before the first invalid Eliminate call. The corrected estimate in round r satisfies

$$
\widehat { \mu } _ { r } \leq \mu _ { \widehat { i } _ { r } } + \frac { \varepsilon _ { r } } { 8 } \leq m + \frac { \varepsilon _ { r } } { 8 } .
$$

Consequently, the upper thresholds used by Fraction and Eliminate satisfy

$$
\begin{array} { r } { \widehat { \mu } _ { r } - \frac { 9 } { 8 } \varepsilon _ { r } \leq m - \varepsilon _ { r } , \qquad \widehat { \mu } _ { r } - \frac { 5 } { 8 } \varepsilon _ { r } \leq m - \frac { 1 } { 2 } \varepsilon _ { r } . } \end{array}
$$

Thus, whenever Eliminate is called while the optimal arm is active, its upper threshold is below m.   
A valid call therefore retains the optimal arm. Induction over the rounds proves the claim.

Call the Median output in round r successful if

$$
\mu _ { \widehat { i } _ { r } } \geq \operatorname* { m a x } _ { i \in S _ { r } } \mu _ { i } - \frac { \varepsilon _ { r } } { 8 } .
$$

Before the first invalid Eliminate call, the optimal arm lies in $S _ { r }$ , so the maximum on the right is m. If Median is successful, then the corrected Mean output satisfies

$$
\widehat { \mu } _ { r } \geq m - \frac { \varepsilon _ { r } } { 4 } .
$$

It follows that the lower thresholds used by Fraction and Eliminate are bounded below by

$$
\begin{array} { r } { \widehat { \mu } _ { r } - \frac { 7 } { 4 } \varepsilon _ { r } \geq m - 2 \varepsilon _ { r } , \qquad \widehat { \mu } _ { r } - \frac { 3 } { 4 } \varepsilon _ { r } \geq m - \varepsilon _ { r } . } \end{array}
$$

Suppose that Median is successful and that every Eliminate call through the current round is valid. If Fraction returns False, then

$$
| \{ i \in S _ { r } : \mu _ { i } < \widehat { \mu } _ { r } - \frac { 7 } { 4 } \varepsilon _ { r } \} | < \theta _ { r } | S _ { r } | < \frac { | S _ { r } | } { 2 } .
$$

More than half of $S _ { r }$ therefore consists of the optimal arm and suboptimal arms with $\Delta _ { i } \leq 2 \varepsilon _ { r }$ Since a False answer leaves the active set unchanged,

$$
| S _ { r + 1 } | \leq 2 \bigl ( 1 + | \{ i \neq * : \Delta _ { i } \leq 2 \varepsilon _ { r } \} | \bigr ) .
$$

If Fraction returns True, validity of Eliminate implies that at least $0 . 9 | S _ { r + 1 } |$ returned arms have mean at least $m - \varepsilon _ { r }$ . All such arms are included among the optimal arm and the suboptimal arms with $\Delta _ { i } \le \varepsilon _ { r }$ . Hence the same, slightly weaker bound holds:

$$
| S _ { r + 1 } | \leq 2 \bigl ( 1 + | \{ i \neq * : \Delta _ { i } \leq 2 \varepsilon _ { r } \} | \bigr ) .\tag{5.2}
$$

There is a sharper conclusion once the round scale is below the smallest gap. Let

$$
Q = \lfloor \log _ { 4 } D \rfloor , \qquad D = g ^ { - 2 } .
$$

Then

$$
2 ^ { - ( Q + 1 ) } < g \leq 2 ^ { - Q } .
$$

For $r \geq Q + 2 .$

$$
2 \varepsilon _ { r } = 2 ^ { 1 - r } \leq 2 ^ { - ( Q + 1 ) } < g .
$$

Thus every suboptimal arm lies below both lower thresholds displayed above. $\mathrm { I f } \ | S _ { r } | \geq 2$ , at least half of $S _ { r }$ then lies below the Fraction lower threshold, so False is not a valid answer and the comparison process must return True. A valid Eliminate call retains the optimal arm. If n suboptimal arms remain afterward, its validity also gives

$$
n \leq 0 . 1 ( n + 1 ) ,
$$

which forces $n = 0$ . Therefore every successful Median round $r \geq Q + 2$ , before any invalid Eliminate call, leaves $S _ { r + 1 }$ equal to the singleton containing the optimal arm.

5.2. Active-set bounds. We now use (5.2) to control the two cumulative quantities appearing in the budget tests: the linear sum of the round weights and its logarithmically weighted analogue. The next lemma is a stopped-process version of [3, lemmas B.4–B.5].

Recall Φ from (3.2). For a summable nonnegative sequence $v = \left( v _ { r } \right)$ with $V = \textstyle \sum _ { r } v _ { r }$ , we use the extension

$$
\Phi ( v ) = \sum _ { r : v _ { r } > 0 } v _ { r } \log \frac { V } { v _ { r } } ,
$$

with $\Phi ( 0 ) = 0 .$

We first remove the deterministic sample limit from the stage. The resulting bounds continue to hold when the limit is restored, because the limit can only terminate the stage earlier. A round is entered when the first budget test is reached and executed when both budget tests are passed. We adopt the convention

$$
c _ { r } \log \left( e + \frac { B } { \eta c _ { r } } \right) = 0 \qquad \mathrm { w h e n ~ } c _ { r } = 0 .
$$

Lemma 5.1. Consider one stage of the comparison process without its deterministic sample limit. Stop the analysis at the first invalid Eliminate call, at the first budget rejection, or after the last permitted round, whichever occurs first. The round containing an invalid Eliminate call and a round rejected by a budget test are both included.

For every entered round with at least two active arms, set

$$
c _ { r } = 4 ^ { r } | S _ { r } | ,
$$

and set $c _ { r } = 0$ for all subsequent rounds. Thus $c _ { r } = w _ { r }$ on executed rounds; for a round rejected immediately by a budget test, $c _ { r }$ records the corresponding round weight. There is a deterministic nonnegative sequence $b = ( b _ { r } ) _ { r \geq 1 }$ such that

$$
\mathbb { E } c _ { r } \leq b _ { r } , \qquad \sum _ { r } b _ { r } \leq C H , \qquad \Phi ( b ) \leq C H ( \operatorname { E n t } ( I ) + 1 ) .
$$

Consequently, whenever $B \geq H$

$$
\mathbb { E } \sum _ { r } c _ { r } \leq C H , \qquad \mathbb { E } \sum _ { r } c _ { r } \log \left( e + \frac { B } { \eta c _ { r } } \right) \leq C H \big [ a + \mathrm { E n t } ( I ) + 1 + \log ( B / H ) \big ] .\tag{5.3}
$$

We also have the following localized version. Fix $2 \leq s \leq K$ , an integer $q \geq 0$ , and a deterministic cutof $r _ { 0 } \in \mathbb { N } \cup \{ \infty \}$ . Designate $s - 1$ arms, including the optimal arm, and suppose that each designated arm has mean at least the upper threshold of every Eliminate call made in a round $r < r _ { 0 }$ Assume that every nondesignated arm belongs to a gap group with index at most q, and set

$$
H _ { \mathrm { t a i l } } = \sum _ { \mathrm { \it { i \ne * } } } \Delta _ { i } ^ { - 2 } .
$$

For this localized estimate only, assign all $s - 1$ designated arms to group q, regardless of their actual gaps.

Let R contain the entered rounds $r < r _ { 0 }$ , up to and including the first round $r \geq q + 2$ in which Median succeeds. If no such round occurs, let R contain every entered round before the cutof or an earlier termination. A round rejected immediately by a budget test is included in R. Then

$$
\mathbb { E } \sum _ { r \in \mathcal { R } } c _ { r } \leq C \big [ H _ { \mathrm { t a i l } } + ( s - 1 ) 4 ^ { q } \big ] .
$$

Proof. Step 1: A deterministic bound for the round weights. For the global estimate, treat the optimal arm as if it had gap g, and hence as if it belonged to group $Q = \lfloor \log _ { 4 } D \rfloor$ . Define the

augmented group sizes and weights by

$$
n _ { j } ^ { \prime } = | G _ { j } | + { \bf 1 } _ { \{ j = Q \} } , \qquad v _ { j } = n _ { j } ^ { \prime } 4 ^ { j } , \qquad A _ { \ell } = \sum _ { j \ge \ell } n _ { j } ^ { \prime } .
$$

Thus $\begin{array} { r } { \sum _ { j } n _ { j } ^ { \prime } = K } \end{array}$

Suppose that Median succeeds in round k. If $k \leq Q + 1$ , then (5.2) implies that any subsequently entered round has at most $2 A _ { k - 1 }$ active arms. If $k \geq Q + 2$ , the successful round leaves a singleton, so no later round is entered.

To bound $\mathbb { E } c _ { r }$ , decompose according to the last successful Median call before round $^ { r } \cdot$ The probability that all preceding $r - 1$ Median calls fail is at most $\kappa ^ { r - 1 }$ . If the last success occurs in round $k < r ,$ the subsequent $r - k - 1$ Median calls must all fail, an event of conditional probability at most $\kappa ^ { r - k - 1 }$ . Therefore

$$
\mathbb { E } c _ { r } \le 4 ^ { r } \left[ K \kappa ^ { r - 1 } + 2 \sum _ { k = 1 } ^ { r - 1 } \kappa ^ { r - k - 1 } A _ { k - 1 } \right] .
$$

An earlier invalid Eliminate call, budget rejection, or other termination can only set some later $c _ { r } \mathrm { { ^ { * _ { s } } } }$ to zero, so the same bound holds for the stopped process.

For a fixed augmented arm in group $j ,$ its contribution to the expression in brackets is at most

$$
\kappa ^ { r - 1 } + 2 \sum _ { k = 1 } ^ { \operatorname* { m i n } \{ r - 1 , j + 1 \} } \kappa ^ { r - k - 1 } \leq 4 \kappa ^ { \operatorname* { m a x } \{ 0 , r - j - 2 \} } .
$$

Define the two-sided kernel

$$
k _ { d } = \left\{ { 4 ^ { d } } , \ \begin{array} { l l } { { d \leq 2 , } } \\ { { 4 ^ { d } \kappa ^ { d - 2 } , } } \end{array} \right.
$$

Then

$$
\mathbb { E } c _ { r } \le b _ { r } , \qquad b _ { r } = 4 \sum _ { j \ge 0 } v _ { j } k _ { r - j } , \qquad r \ge 1 .
$$

Because $\kappa = 0 . 0 1$

$$
\sum _ { d \in \mathbb { Z } } k _ { d } = \sum _ { d \le 2 } { 4 ^ { d } } + \sum _ { d \ge 3 } { 4 ^ { d } } \kappa ^ { d - 2 } = { \frac { 6 4 } { 3 } } + { \frac { 6 4 \kappa } { 1 - 4 \kappa } } = 2 2 .
$$

In particular,

$$
\sum _ { r \geq 1 } b _ { r } \leq 8 8 \sum _ { j } v _ { j } .
$$

Step 2: Entropy calculation. The probability vector $( k _ { d } / 2 2 ) _ { d \in \mathbb { Z } }$ has finite entropy because both tails decay geometrically. Put

$$
V = \sum _ { j } v _ { j } ,
$$

let X have law $( v _ { j } / V ) _ { j }$ , and let Y be independent of X with law $( k _ { d } / 2 2 ) _ { d }$ . Extend the convolution to all $r \in \mathbb { Z }$ by setting

$$
\widetilde { b } _ { r } = 4 \sum _ { j } v _ { j } k _ { r - j } .
$$

Its total mass is 88V, and its normalized distribution is the law of $X + Y$ . Since $X + Y$ is a function of $( X , Y )$ ，

$$
\operatorname { E n t } ( X + Y ) \leq \operatorname { E n t } ( X , Y ) = \operatorname { E n t } ( X ) + \operatorname { E n t } ( Y ) .
$$

Consequently,

$$
\Phi ( { \tilde { b } } ) = 8 8 V \operatorname { E n t } ( X + Y ) \leq 8 8 \Phi ( v ) + 8 8 V \operatorname { E n t } ( Y ) .
$$

The sequence $b$ is obtained by restricting $\widetilde { b }$ to $r \geq 1$ . Coordinatewise monotonicity of $\Phi$ , first for finite truncations and then by passage to the limit, gives

$$
\Phi ( b ) \leq 8 8 \Phi ( v ) + 8 8 V \operatorname { E n t } ( Y ) .
$$

The closest competitor belongs to $G _ { Q } , \mathrm { s o } \ | G _ { Q } | \geq 1$ . Adding the artificial optimal arm therefore at most doubles the weight of that group. Since

$$
| G _ { j } | 4 ^ { j } \leq H _ { j } ,
$$

we have $v _ { j } \leq 2 H _ { j }$ for every j. The monotonicity and homogeneity of $\Phi$ now give

$$
V \leq 2 H , \qquad \Phi ( v ) \leq \Phi ( ( 2 H _ { j } ) _ { j } ) = 2 H \operatorname { E n t } ( I ) .
$$

It follows that

$$
\sum _ { r } b _ { r } \leq C H , \qquad \Phi ( b ) \leq C H ( \operatorname { E n t } ( I ) + 1 ) ,
$$

which proves the first set of assertions.

Step 3: Logarithmically weighted cost. Set

$$
f ( z ) = z \log \left( e + \frac { B } { \eta z } \right) , \qquad f ( 0 ) = 0 .
$$

This function is increasing and concave. Hence Jensen’s inequality, followed by $\mathbb { E } c _ { r } \le b _ { r }$ , yields

$$
\mathbb { E } f ( c _ { r } ) \leq f ( \mathbb { E } c _ { r } ) \leq f ( b _ { r } ) .
$$

Let $N = \textstyle \sum _ { r } b _ { r }$ . For every r with $b _ { r } > 0$

$$
e + \frac { B } { \eta b _ { r } } \leq \frac { N } { b _ { r } } \left( e + \frac { B } { \eta N } \right) .
$$

After taking logarithms, multiplying by $b _ { r }$ , and summing,

$$
\sum _ { r } f ( b _ { r } ) \leq N \log \left( e + { \frac { B } { \eta N } } \right) + \Phi ( b ) .
$$

Since $N \leq C H , B \geq H$ , and $a = \log ( 1 / \eta )$ , monotonicity of $f$ gives

$$
N \log \left( e + \frac { B } { \eta N } \right) \leq C H [ a + 1 + \log ( B / H ) ] .
$$

Together with the entropy bound for $b ,$ this proves (5.3).

Step $\it 4 \mathrm { : }$ Localized estimate. Let D be the set of $s - 1$ designated arms and put

$$
W _ { \mathrm { l o c } } = H _ { \mathrm { t a i l } } + ( s - 1 ) 4 ^ { q } .
$$

For this step, set $c _ { r } = 0$ outside the rounds in $\mathcal { R }$ . Thus the cutof $r _ { 0 } .$ , any earlier termination, and the first successful Median call at or after round $q + 2$ are all incorporated into the definition of $c _ { r }$

Assign the designated arms artificially to group q and define

$$
n _ { j } ^ { \mathrm { l o c } } = | \{ i \in G _ { j } : i \notin \mathcal { D } \} | + ( s - 1 ) \mathbf { 1 } _ { \{ j = q \} } , \qquad A _ { \ell } ^ { \mathrm { l o c } } = \sum _ { j \geq \ell } n _ { j } ^ { \mathrm { l o c } } .
$$

By the assumptions on the nondesignated arms,

$$
\sum _ { j = 0 } ^ { q } n _ { j } ^ { \mathrm { l o c } } = K , \qquad \sum _ { j = 0 } ^ { q } n _ { j } ^ { \mathrm { l o c } } 4 ^ { j } \le H _ { \mathrm { t a i l } } + ( s - 1 ) 4 ^ { q } = W _ { \mathrm { l o c } } .
$$

For $k \leq q + 1$ , every designated arm is counted in $A _ { k - 1 } ^ { \mathrm { l o c } }$ . Moreover, every nondesignated arm with $\Delta _ { i } \le 2 \varepsilon _ { k } = 2 ^ { 1 - k }$ belongs to a group of index at least $k - 1$ . Therefore, if a successful Median round $k \leq q + 1$ is followed by another entered round, (5.2) gives

$$
| S _ { k + 1 } | \leq 2 A _ { k - 1 } ^ { \mathrm { l o c } } .
$$

The same decomposition according to the last successful Median call now yields, for $1 \leq r \leq q + 2$

$$
\mathbb { E } c _ { r } \le 4 ^ { r } \left[ K \kappa ^ { r - 1 } + 2 \sum _ { k = 1 } ^ { r - 1 } \kappa ^ { r - k - 1 } A _ { k - 1 } ^ { \mathrm { l o c } } \right] .
$$

Applying the kernel estimate from Step 1 and summing over these rounds gives

$$
\mathbb { E } \sum _ { r = 1 } ^ { q + 2 } c _ { r } \le 4 \sum _ { j = 0 } ^ { q } n _ { j } ^ { \mathrm { l o c } } 4 ^ { j } \sum _ { r = 1 } ^ { q + 2 } k _ { r - j } \le 8 8 W _ { \mathrm { l o c } } .
$$

It remains to control rounds after $q + 2 .$ . The active set never grows. For $j \geq 1$ , in order for $c _ { q + 2 + j }$ to contribute to the localized sum, the Median calls in rounds $q + 2 , \ldots , q + 1 + j$ must all fail; otherwise the sum stops after the first successful call. Conditionally on the history upon entering round $q + 2$ , these j failures have probability at most $\kappa ^ { j }$ . Hence

$$
\mathbb { E } c _ { q + 2 + j } \le ( 4 \kappa ) ^ { j } \mathbb { E } c _ { q + 2 } \le C W _ { \mathrm { l o c } } ( 4 \kappa ) ^ { j } , \qquad j \ge 1 .
$$

Since $4 \kappa < 1$ , summing this geometric tail and combining it with the estimate for the earlier rounds gives

$$
\mathbb { E } \sum _ { r \in \mathcal { R } } c _ { r } \leq C W _ { \mathrm { l o c } } = C \big [ H _ { \mathrm { t a i l } } + ( s - 1 ) 4 ^ { q } \big ] .
$$

5.3. Joint and localized errors. Until the end of this section, all probabilities refer to the comparison process. We first control the joint probability that several arms are eliminated before the rounds at which their gaps become distinguishable.

Fix one stage, and let $\mathcal { E }$ be the set of rounds in which Eliminate is called. Whenever such a call is made in round r, the first budget test has ensured ${ U _ { r } } + 4 w _ { r } < B$ , and the subsequent update is $U _ { r + 1 } = U _ { r } + 4 w _ { r }$ . Hence, pathwise,

$$
\sum _ { r \in { \mathcal E } } \beta _ { r } = \frac { \rho } { B } \sum _ { r \in { \mathcal E } } 4 w _ { r } = \frac { \rho } { B } \sum _ { r \in { \mathcal E } } 4 | S _ { r } | \ 4 ^ { r } < \rho .\tag{5.4}
$$

For a suboptimal arm $i \in G _ { q } ,$ define

$$
M _ { i } = \mathbf { 1 } \{ i { \mathrm { ~ i s ~ r e m o v e d ~ b y ~ a n ~ E l i m i n a t e ~ c a l l ~ i n ~ s o m e ~ r o u n d ~ } } r < q \} .
$$

Thus $M _ { i } = 1$ means that i is removed before the algorithm reaches the scale associated with its gap. For the optimal arm, define

$$
M _ { * } = \mathbf { 1 } \{ * { \mathrm { ~ i s ~ r e m o v e d ~ b y ~ a n ~ E l i m i n a t e ~ c a l l ~ i n ~ a n y ~ r o u n d } } \} .
$$

We refer to either event as a premature elimination, corresponding to [3, definition B.8].

The next lemma combines (5.4) with the product guarantee for Eliminate. Its backward-induction argument adapts [3, lemma B.9] to the comparison process.

Lemma 5.2. For a fixed collection of k arms in one stage,

$$
\mathbb { P } ( M _ { i } = 1 \ f o r \ e v e r y \ a r m \ i n \ t h e \ c o l l e c t i o n ) \le \rho ^ { k } .\tag{5.5}
$$

Proof. The corrected mean estimate is at most $m + \varepsilon _ { r } / 8$ , even if the true optimal arm was removed earlier. Thus the upper elimination threshold in round r is at most $m - \varepsilon _ { r } / 2$ . For an arm in gap group $q > r , \Delta _ { i } \leq 2 ^ { - q } \leq \varepsilon _ { r } / 2$ , so its mean is at least the upper elimination threshold. The optimal arm also lies above this threshold. Hence every arm whose removal in round r would be premature satisfies the hypothesis of the simultaneous-removal bound. Conditional on the past, a specified set of j eligible arms is therefore eliminated in that call with probability at most $\beta _ { r } ^ { j }$

Induct backward over the finitely many remaining rounds. If a specified arm has already been removed at or after its gap-group index, the joint event is impossible. Immediately before a call in round $r ,$ , let

$$
u = \rho - \sum _ { v \in \mathscr { E } \atop v < r } \beta _ { v } .
$$

This quantity is determined by the past. By (5.4), the current and future calls together receive total error allowance at most u on every continuation. Suppose that k specified arms still have to be eliminated prematurely. If the current round has reached the gap-group index of any specified suboptimal arm, the joint event is impossible. Otherwise all k specified arms are eligible for premature elimination in the current call. The induction claim is that their joint probability is at most $u ^ { k }$ . Condition on the history immediately before the next Eliminate call, including the preceding Mean and Fraction outputs. Its error budget $\beta$ is then fixed, and the total budget available to later calls is at most $u - \beta$ on every continuation. The conditional probability of any particular exact subset of $j$ premature removals is at most $\beta ^ { j }$ , since it is bounded by the probability that those $j$ arms are all removed. Conditional on the resulting history, the probability that the remaining $k - j$ arms are all eliminated prematurely is at most $( u - \beta ) ^ { k - j }$ . Summing over subsets gives

$$
\sum _ { j = 0 } ^ { k } \binom { k } { j } \beta ^ { j } ( u - \beta ) ^ { k - j } = u ^ { k } .
$$

If no further call is made and $k > 0$ specified arms still require premature elimination, the conditional probability is zero. If $k = 0$ , the event is already satisfied and its conditional probability is one. These are the terminal cases of the induction. Initially the available budget is $\rho ,$ so the induction proves (5.5). □

When only a few arms are close to optimal, we use a sharper bound involving the sum of the inverse squared gaps of the remaining, easier arms. The next lemma, analogous to [3, lemma B.6], controls the probability of an invalid Eliminate call before the algorithm reaches the scale needed to distinguish the dificult arms.

Order the arms so that

$$
\mu _ { [ 1 ] } = m > \mu _ { [ 2 ] } \geq \cdots \geq \mu _ { [ K ] } , \qquad \Delta _ { [ i ] } = m - \mu _ { [ i ] } .
$$

Ties among the suboptimal arms may be ordered arbitrarily.

Lemma 5.3. Fix $2 \leq s \leq K$ $I f s = 2 ,$ , set $r _ { * } = \infty$ and impose no restriction on $B$ . If $s \geq 3$ assume

$$
B < \Delta _ { [ s - 1 ] } ^ { - 2 }
$$

and set

$$
r _ { * } = \left\lfloor \log _ { 2 } \frac { 1 } { \Delta _ { [ s - 1 ] } } \right\rfloor .
$$

Then

P(an invalid Eliminate call occurs in some round $r < r _ { * } ) \leq C s \rho \frac { \sum _ { i = s } ^ { K } \Delta _ { [ i ] } ^ { - 2 } } { B }$

(5.6)

Proof. Stop the analysis after the first invalid Eliminate call, including the round in which that call occurs. Set

$$
q = \left\lfloor \log _ { 2 } \frac { 1 } { \Delta _ { [ s ] } } \right\rfloor , \qquad H _ { \mathrm { t a i l } } = \sum _ { i = s } ^ { K } \Delta _ { [ i ] } ^ { - 2 } .
$$

Call the arms of ranks $1 , \ldots , s - 1$ the dificult arms and those of ranks $s , \ldots , K$ the tail arms.

Step 1: Separate the dificult and tail arms. For $r < r _ { * }$ 2

$$
\Delta _ { [ s - 1 ] } \leq \frac { \varepsilon _ { r } } { 2 } .
$$

When $s = 2$ , this remains true because $\Delta _ { [ 1 ] } = 0$ . The corrected Mean output satisfies

$$
\widehat { \mu } _ { r } \leq m + \frac { \varepsilon _ { r } } { 8 } .
$$

Consequently, the upper mean thresholds in Fraction and Eliminate are at most

$$
\widehat { \mu } _ { r } - \frac { 9 } { 8 } \varepsilon _ { r } \leq m - \varepsilon _ { r } , \qquad \widehat { \mu } _ { r } - \frac { 5 } { 8 } \varepsilon _ { r } \leq m - \frac { \varepsilon _ { r } } { 2 } .
$$

Every active dificult arm therefore lies above both upper thresholds.

The definition of $q$ gives

$$
\Delta _ { [ s ] } > 2 ^ { - ( q + 1 ) } .
$$

Hence, if $r \geq q + 2$ , then

$$
\Delta _ { [ s ] } > 2 \varepsilon _ { r } .
$$

When Median succeeds in such a round, the corrected Mean output satisfies

$$
\widehat { \mu } _ { r } \geq m - \frac { \varepsilon _ { r } } { 4 } .
$$

The lower mean thresholds in Fraction and Eliminate are therefore at least

$$
m - 2 \varepsilon _ { r } \qquad \mathrm { a n d } \qquad m - \varepsilon _ { r } ,
$$

respectively. Every tail arm lies below both lower thresholds.

Step 2: Control the calls after a successful Median round. Suppose that the first successful Median call with index at least $q + 2$ occurs in a round $r < r _ { * }$ . Let $\gamma$ be the fraction of the current active set consisting of tail arms.

If Fraction returns False, its validity gives

$$
\gamma < \theta _ { r } ,
$$

because every tail arm is below the lower Fraction threshold. The active set is unchanged. In any later round $v < r _ { * }$ , every arm below the upper Fraction threshold is a tail arm, whereas

$$
\theta _ { v - 1 } \geq \theta _ { r } .
$$

It follows inductively that True is invalid in every such round. All subsequent Fraction outputs before $r _ { * }$ are therefore False.

If Fraction returns True and the resulting Eliminate call is valid, every surviving tail arm is below the lower elimination threshold. The fraction of tail arms in the returned set is therefore at most 0.1. Since $\theta _ { v - 1 } \geq 0 . 3$ in every later round $v ,$ the same induction shows that all subsequent Fraction outputs before $r _ { * }$ are False.

Thus, after the first successful Median call in a round $r \geq q + 2 $ , either the Eliminate call in that round is invalid or no further Eliminate call occurs before $r _ { * }$

Step 3: Bound the total weight of the relevant rounds. Let R be the collection of entered rounds before $r _ { * }$ , stopped after and including the first successful Median round with index at least $q + 2 .$ , if such a round occurs. Step 2 shows that every possible first invalid Eliminate call before $r _ { * }$ occurs in a round belonging to $\mathcal { R }$

Apply the localized part of Lemma 5.1 with cutof $r _ { 0 } = r _ { * }$ , designating the top $s - 1$ arms and assigning them artificially to group $q .$ Step 1 verifies the required upper-threshold condition. Every tail arm has gap at least $\Delta _ { [ s ] }$ , so its gap-group index is at most $q .$ The lemma therefore gives

$$
\mathbb { E } \sum _ { r \in \mathcal { R } } | S _ { r } | \mathbb { I } ^ { r } \leq C \big [ H _ { \mathrm { t a i l } } + ( s - 1 ) 4 ^ { q } \big ] .
$$

Since

$$
4 ^ { q } \leq \Delta _ { [ s ] } ^ { - 2 } \leq H _ { \mathrm { t a i l } } ,
$$

we obtain

$$
\mathbb { E } \sum _ { r \in \mathcal { R } } | S _ { r } | \ r \leq C s H _ { \mathrm { t a i l } } .
$$

Step $\it 4 .$ Sum the probabilities of the first invalid call. Before the first invalid Eliminate call, the optimal arm remains active. Conditional on the past, the probability that the call in round r is invalid is therefore at most

$$
\beta _ { r } = \frac { 4 \rho | S _ { r } | 4 ^ { r } } { B } .
$$

Summing over the possible locations of the first invalid call and using the preceding estimate,

P(an invalid Eliminate call before round $r _ { * } ) \leq \frac { 4 \rho } { B } \mathbb { E } \sum _ { r \in \mathcal { R } } | S _ { r } | 4 ^ { r } \leq \frac { C s \rho H _ { \mathrm { t a i l } } } { B } .$

Substituting the definition of $H _ { \mathrm { t a i l } }$ proves (5.6).

5.4. Summing errors across budget scales. For each suboptimal arm, write

$$
w _ { i } = \Delta _ { i } ^ { - 2 } .
$$

At a stage with budget B, define

$$
h = 1 + | \{ i \neq * : w _ { i } > B \} | , \qquad x = \frac { 1 } { B } \sum _ { \stackrel { i \neq * } { w _ { i } \leq B } } w _ { i } .
$$

Call the optimal arm and the $h - 1$ suboptimal arms with $w _ { i } > B$ hard; call the remaining suboptimal arms easy. Thus Bx is the sum of the inverse squared gaps of the easy arms.

Step 1: A joint bound for the hard arms. Suppose that a hard suboptimal arm $i \in G _ { q }$ is removed in an executed round r. Its weight satisfies

$$
B < w _ { i } < 4 ^ { q + 1 } .
$$

Moreover, the first budget test and $\left| S _ { r } \right| \geq 2$ give

$$
8 \cdot 4 ^ { r } \leq 4 | S _ { r } | 4 ^ { r } < B < w _ { i } < 4 ^ { q + 1 } .
$$

It follows that $r < q ,$ so every removal of a hard suboptimal arm is premature. Every removal of the optimal arm is premature by definition.

If a stage returns an incorrect arm, then the optimal arm and at least $h - 2$ of the $h - 1$ hard suboptimal arms must have been removed. For $h \geq 2$ , take a union bound over the hard suboptimal arm that might remain. Applying (5.5) to each resulting collection of $h - 1$ arms gives

$$
J _ { 1 } = \rho , \qquad J _ { h } = ( h - 1 ) \rho ^ { h - 1 } \quad ( h \ge 2 )\tag{5.7}
$$

as upper bounds for the stage’s wrong-output probability.

Step 2: A bound using the easy arms. Suppose $h < K$ , so at least one easy arm is present. The hard suboptimal arms have ranks $2 , \ldots , h$ , while the easy arms have ranks $h + 1 , \ldots , K$ . Apply (5.6) with $s = h + 1 ;$ when $h = 1$ , use its $s = 2$ case.

For $h \geq 2 .$ , the cutof $r _ { * }$ in that lemma is the gap-group index of the hard suboptimal arm with the largest gap. The calculation in Step 1 shows that every executed Eliminate call occurs before this cutof. Since a wrong output requires the optimal arm to be removed, it requires an invalid Eliminate call before $r _ { * }$ . The same conclusion holds for $h = 1$ , for which $r _ { * } = \infty$ . Because

$$
\sum _ { i = h + 1 } ^ { K } \Delta _ { [ i ] } ^ { - 2 } = B x ,
$$

the local bound gives

$$
\mathbb { P } ( \mathrm { w r o n g ~ o u t p u t ~ i n ~ t h e ~ s t a g e } ) \le C ( h + 1 ) \rho x .\tag{5.8}
$$

If $h = K$ , every suboptimal arm satisfies $\Delta _ { i } < B ^ { - 1 / 2 }$ . In any executed round,

$$
8 \cdot 4 ^ { r } < B ,
$$

and hence $\varepsilon _ { r } = 2 ^ { - r } > B ^ { - 1 / 2 }$ . The upper Fraction threshold is at most $m - \varepsilon _ { r }$ , which is below every arm mean. Thus True is not valid, every corrected Fraction output is False, and the stage cannot accept. Its wrong-output probability is therefore zero.

Step 3: A necessary condition for acceptance. Assign truncated weights

$$
v _ { * } = 1 , \qquad v _ { i } = \operatorname* { m i n } \{ w _ { i } / B , 1 \} \quad ( i \neq * ) ,
$$

and write

$$
v _ { \mathrm { t o t } } = \sum _ { i } v _ { i } = h + x .
$$

Every accepted stage satisfies

$$
\sum _ { i } v _ { i } ( 1 - M _ { i } ) \leq 2 .\tag{5.9}
$$

To verify this, consider a suboptimal arm $i \in G _ { q }$ removed in a round $r \geq q$ . Then

$$
B v _ { i } \leq w _ { i } < 4 ^ { q + 1 } \leq 4 ^ { r + 1 } .
$$

The total $B v _ { i }$ over all such removals in round r is therefore at most

$$
| S _ { r } | 4 ^ { r + 1 } = 4 | S _ { r } | 4 ^ { r } ,
$$

the increment of U in that round. The first budget test ensures that the sum of these increments over all Eliminate calls is less than B. Hence the total weight of suboptimal arms removed at or after their gap-group indices is less than one.

The single returned arm contributes at most one more. If the output is wrong, then $M _ { * } = 1$ so the original optimal arm contributes zero to the left side of (5.9). If the output is correct, the optimal arm is the returned arm. This proves (5.9).

Step 4: An exponential-moment bound for acceptance. Set

$$
u = \log ( 1 / \rho ) .
$$

By choosing the absolute constant c suficiently small, we may assume $u \geq 4$ . Since $M _ { i } \in \{ 0 , 1 \}$ 2

$$
\exp \left( u \sum _ { i } v _ { i } M _ { i } \right) = \prod _ { i } \left[ 1 + ( e ^ { u v _ { i } } - 1 ) M _ { i } \right] .
$$

All coeficients in this product expansion are nonnegative. Applying (5.5) to each product of indicators gives

$$
\mathbb { E } \exp \left( u \sum _ { i } v _ { i } M _ { i } \right) \leq \prod _ { i } \left[ 1 + \rho ( e ^ { u v _ { i } } - 1 ) \right] .
$$

For $0 \leq v \leq 1$ , convexity gives

$$
e ^ { u v } - 1 \leq v ( e ^ { u } - 1 ) .
$$

Therefore

$$
\mathbb { E } \exp \left( u \sum _ { i } { v _ { i } M _ { i } } \right) \le \exp \left( \rho ( e ^ { u } - 1 ) \sum _ { i } { v _ { i } } \right)
$$

because $\rho ( e ^ { u } - 1 ) = 1 - \rho \leq 1$

On acceptance, (5.9) implies

$$
\sum _ { i } v _ { i } M _ { i } \geq v _ { \mathrm { t o t } } - 2 .
$$

If $v _ { \mathrm { t o t } } \geq 4 ,$ , Markov’s inequality consequently gives

$$
\begin{array} { r } { \mathbb { P } ( \mathrm { a c c e p t } ) \le e ^ { v _ { \mathrm { t o t } } } \rho ^ { v _ { \mathrm { t o t } } - 2 } } \\ { \le \rho ^ { v _ { \mathrm { t o t } } / 4 } . } \end{array}\tag{5.10}
$$

The final inequality uses $u = \log ( 1 / \rho ) \geq 4$ . Since every incorrect output is an acceptance, (5.10) also bounds the wrong-output probability.

The preceding exponential-moment argument follows [3, lemmas B.11–B.12 and section B.5], with the truncated weights above.

Step $5 { : }$ Sum over the geometric budget sequence. For a fixed value $h < K$ , the corresponding stages form an interval in the sequence $B _ { t } = 1 0 0 ^ { t }$ . Throughout this interval the easy-arm set is fixed, and each successive stage replaces x by $x / 1 0 0$ . We divide the stages into three ranges.

If $x < 1$ and $h = 1$ , summing (5.8) over the geometric sequence of $x _ { \mathrm { ~ S ~ } } ^ { \prime }$ gives a contribution of at most $C \rho$ . If $x < 1$ and $h \geq 2$ , then

$$
\operatorname* { m i n } \{ J _ { h } , C ( h + 1 ) \rho x \} \leq \sqrt { J _ { h } C ( h + 1 ) \rho x } \leq C h \rho ^ { h / 2 } \sqrt { x } .
$$

For fixed $h ,$ the sum of $\sqrt { x }$ over these stages is at most $1 0 / 9$ . Their total contribution is therefore at most $C h \rho ^ { h / 2 }$ , and summing over $h \geq 2$ gives $O ( \rho )$

${ \mathrm { I f ~ } } 1 \leq x < 1 6$ , there is at most one stage for each $h ,$ because successive positive values of x difer by a factor of 100. Hence (5.7) gives a total contribution bounded by

$$
J _ { 1 } + \sum _ { h \ge 2 } J _ { h } = O ( { \boldsymbol \rho } ) .
$$

Finally, suppose $x \geq 1 6$ . For each fixed $h ,$ enumerate these stages backward from the last one in this range. At the jth preceding stage,

$$
x \geq 1 6 \cdot 1 0 0 ^ { j } .
$$

Using $v _ { \mathrm { t o t } } = h + x$ in (5.10), the total contribution of this range is at most

$$
\sum _ { h \ge 1 } \rho ^ { h / 4 } \sum _ { j \ge 0 } \rho ^ { 4 \cdot 1 0 0 ^ { j } } = O ( \rho ) .
$$

Stages with $h = K$ contribute zero by Step 2. Generate the independent data for every stage in advance, so that the output of each stage is defined even if that stage is never reached. The event that the comparison process returns an incorrect arm is contained in the union of these stagewise wrong-output events. The preceding bounds and the union bound therefore give probability at most $C \rho .$ . Combining this estimate with (5.1), the error probability of the actual algorithm is at most

$$
C \eta + C \rho = C \eta + C \eta ^ { 2 } .
$$

Since $\eta = c \delta$ , choosing the absolute constant $c > 0$ suficiently small makes this quantity at most δ.

## 6. The expected sample complexity

It remains to bound the unconditional expected stopping time ET. We first show that the outer Mean and Fraction calls use only a fixed fraction of each stage budget. We then show that, once the budget is suficiently large, a stage accepts with probability at least $1 9 9 / 2 0 0$ . Together with the deterministic sample limit and the geometric growth of the stage budgets, this will control the expected cost of all restarts.

6.1. The cost of the outer tests. The increment $\theta _ { r } - \theta _ { r - 1 }$ is indexed by $j = R _ { t } + 1 - r$ , the number of permitted rounds from r through $R _ { t }$ . After factoring out $4 ^ { R _ { t } }$ , its polynomial growth is dominated by the resulting geometric decay. The series is summable independently of the active sets.

Lemma 6.1. There is an absolute integer J such that, in every stage t and on every sample path, the total number of samples used by the outer Mean and Fraction calls is at most $B _ { t } / 1 0 0$

Proof. For $1 \leq r \leq R _ { t }$ , we have

$$
R _ { t } \leq \log _ { 4 } B _ { t } = ( \log _ { 4 } 1 0 0 ) t .
$$

Since

$$
\alpha _ { t , r } = \frac { \eta } { 5 0 t ^ { 2 } r ^ { 2 } } ,
$$

it follows that

$$
\log \frac { 1 } { \alpha _ { t , r } } = a + \log 5 0 + 2 \log t + 2 \log r \le C \big [ a + \log ( t + 1 ) \big ] = C L _ { t } .
$$

Fix a permitted round r and put

$$
j = R _ { t } + 1 - r .
$$

The Mean call has accuracy $\varepsilon _ { r } / 8$ , so its cost is at most $C 4 ^ { r } L _ { t }$ . The two mean thresholds in the Fraction call difer by $\frac { 5 } { 8 } \varepsilon _ { r }$ , while its fraction thresholds difer by

$$
\theta _ { r } - \theta _ { r - 1 } = { \frac { 1 } { 1 0 j ^ { 2 } } } .
$$

The Fraction cost is therefore at most $C 4 ^ { r } L _ { t } j ^ { 4 } \log ( 1 0 j ^ { 2 } )$ . Since $j \geq 1$ , the combined cost of the two outer calls in round r is at most

$$
C 4 ^ { r } L _ { t } j ^ { 6 } .
$$

Summing over all permitted rounds, including rounds the stage may not reach, gives

$$
\begin{array} { r } { \mathrm { t o t a l ~ o u t e r \mathrm { - } t e s t ~ c o s t } \le C L _ { t } \displaystyle \sum _ { r = 1 } ^ { R _ { t } } 4 ^ { r } ( R _ { t } + 1 - r ) ^ { 6 } } \\ { = C L _ { t } 4 ^ { R _ { t } } \displaystyle \sum _ { j = 0 } ^ { R _ { t } - 1 } 4 ^ { - j } ( j + 1 ) ^ { 6 } } \\ { \le C L _ { t } 4 ^ { R _ { t } } \displaystyle \sum _ { j = 0 } ^ { \infty } 4 ^ { - j } ( j + 1 ) ^ { 6 } . } \end{array}
$$

The last series is finite and yields an absolute constant.

Whenever $R _ { t } \geq 1$ , its definition gives

$$
L _ { t } 4 ^ { R _ { t } } \leq 4 ^ { - J } B _ { t } .
$$

Choose the absolute integer J suficiently large that the resulting constant multiple of $4 ^ { - J } B _ { t }$ is at most $B _ { t } / 1 0 0$ . If $R _ { t } = 0$ , the stage rejects before making any outer call, so its outer-test cost is zero. □

6.2. Acceptance at a suficient budget. The next lemma fixes the absolute multiplier M in the deterministic sample limit. It also introduces an absolute constant A used only in the analysis.

Lemma 6.2. Define

$$
F = H ( a + \mathrm { E n t } ( I ) ) + D \ell _ { g } , \qquad \ell _ { g } = \log \log ( e ^ { e } / g ) .
$$

There are absolute constants M and A such that every stage with $B _ { t } \geq A F$ accepts with probability at least 199/200.

Proof. Write $B = B _ { t }$ and $R = R _ { t }$

Step 1: Control rejection at the sample limit. Temporarily remove the deterministic sample limit, retaining both budget tests and the restriction to R rounds. On every executed round, $w _ { r } < B / 4$ If Eliminate is called, then

$$
\beta _ { r } = \frac { 4 \eta ^ { 2 } w _ { r } } { B } ,
$$

and

$$
\log \frac { 1 } { \beta _ { r } } = \log \frac { B } { 4 \eta ^ { 2 } w _ { r } } \le 2 \log \frac { B } { \eta w _ { r } } .
$$

The conditional expected Eliminate cost in round r is therefore at most $C z _ { r }$ . The Median cost is at most $C w _ { r } \le C z _ { r }$ . Since

$$
\sum _ { \mathrm { \tiny ~ e x e c u t e d } \ r } z _ { r } < 1 0 0 B ,
$$

the tower property bounds the expected total cost of all Median and Eliminate calls by CB. By Lemma 6.1, the outer Mean and Fraction calls use at most B/100 additional samples. Thus the expected cost of the uncapped stage is at most $C _ { 0 } B$ on every instance, including paths on which subroutine errors occur.

Choose the absolute constant M suficiently large. Markov’s inequality then shows that the uncapped stage uses more than MB samples with probability at most 1/2000. Coupling the capped and uncapped stages until the cap is reached shows that imposing the deterministic limit causes rejection with probability at most 1/2000. The same estimate holds for the comparison process, since the preceding cost bound applies equally after correcting the outer outputs.

Step 2: Control rejection at the budget tests. In the comparison process, an Eliminate call in round r is invalid with conditional probability at most $\beta _ { r }$ . Consequently, (5.4) and the tower property give

$$
\mathbb { P } ( \mathrm { s o m e ~ E l i m i n a t e ~ c a l l ~ i s ~ i n v a l i d } ) \le \rho .
$$

Stop the comparison process at its first invalid Eliminate call, but include the round containing that call and any round rejected by a budget test. If the U-test rejects, then

$$
\sum _ { r } c _ { r } \geq { \frac { B } { 4 } } .
$$

If the V -test rejects, then

$$
\sum _ { r } c _ { r } \log \left( e + \frac { B } { \eta c _ { r } } \right) \geq 1 0 0 B .
$$

Indeed, the corresponding sum without the e is $V _ { r } + z _ { r }$ , which is at least 100B.

For $B \geq H$ , Markov’s inequality and (5.3) therefore give

P(a budget test rejects before the first invalid Eliminate call $) \leq { \frac { C H } { B } } \left[ a + \operatorname { E n t } ( I ) + 1 + \log { \frac { B } { H } } \right]$

(6.1)

To bound the right side, put

$$
\sigma = a + \mathrm { E n t } ( I ) , \qquad y = \frac { B } { H } .
$$

Since $F \geq H \sigma$ , the condition $B \geq A F$ gives $y \geq A \sigma$ . The function

$$
y \longmapsto { \frac { \sigma + 1 + \log y } { y } }
$$

is decreasing for $y \geq 1$ . Hence the right side of (6.1) is at most

$$
C { \frac { \sigma + 1 + \log ( A \sigma ) } { A \sigma } } \leq C { \frac { 1 + \log A } { A } } .
$$

It can therefore be made arbitrarily small by choosing the absolute constant A suficiently large. Step 3: Ensure suficiently many rounds. Set

$$
f = { \frac { F } { D } } .
$$

Since $H \geq D .$

$$
f = { \frac { H } { D } } [ a + \operatorname { E n t } ( I ) ] + \ell _ { g } \geq a + \ell _ { g } .
$$

Let $t _ { 0 }$ be the first stage index for which $B _ { t _ { 0 } } \geq A F$ . By minimality,

$$
B _ { t _ { 0 } } < 1 0 0 A F = 1 0 0 A D f ,
$$

and therefore

$$
t _ { 0 } + 1 \le 2 + \frac { \log ( 1 0 0 A ) + \log D + \log f } { \log 1 0 0 } .
$$

The definition of $\ell _ { g }$ gives

$$
\log D = 2 \log ( 1 / g ) = 2 ( e ^ { \ell _ { g } } - e ) \leq 2 e ^ { f } .
$$

Since $f \geq 1$ , we also have log $f \leq e ^ { f }$ . It follows that

$$
t _ { 0 } + 1 \le C [ 1 + \log A + e ^ { f } ] .
$$

Using $a \leq f ,$ we obtain

$$
L _ { t _ { 0 } } = a + \log ( t _ { 0 } + 1 ) \leq 2 f + C + \log ( 1 + \log A ) \leq C _ { A } f ,
$$

where

$$
C _ { A } = O ( 1 + \log \log ( A + e ) ) .
$$

Consequently,

$$
\frac { B _ { t _ { 0 } } } { L _ { t _ { 0 } } } \geq \frac { A } { C _ { A } } D .
$$

The ratio $B _ { t } / L _ { t }$ is increasing in $t ,$ so the same bound holds at every later stage.

Fix an integer $b \geq 1$ . Because $A / C _ { A }  \infty$ as $A \to \infty$ , we may choose A large enough that every stage with $B _ { t } \geq A F$ satisfies

$$
R _ { t } \geq Q + 2 + b .
$$

Consider the comparison process without the deterministic sample limit. Suppose that no budget test rejects and no Eliminate call is invalid. If the active set has not become a singleton by the last permitted round, then every Median call in rounds

$$
Q + 2 , Q + 3 , \ldots , R _ { t }
$$

must fail. Each call uses fresh samples and has conditional failure probability at most $\kappa .$ Since the list contains at least $b + 1$ calls, successive conditioning gives

$$
\mathbb { P } \left( \begin{array} { c } { \mathrm { n o ~ s i n g l e t o n ~ b y ~ t h e ~ l a s t ~ p e r m i t t e d ~ r o u n d } , } \\ { \mathrm { n o ~ b u d g e t ~ t e s t ~ r e j e c t s } , \mathrm { a n d } } \\ { \mathrm { n o ~ E l i m i n a t e ~ c a l l ~ i s ~ i n v a l i d } } \end{array} \right) \leq \kappa ^ { b + 1 } .
$$

Step 4: Combine the rejection probabilities. First choose b so that $\kappa ^ { b + 1 }$ is suficiently small. Then choose A large enough both to obtain the required number of rounds and to make (6.1) suficiently small. These choices can ensure that, in the comparison process without the deterministic sample limit, the probability of rejection before the first invalid Eliminate call, either at a budget test or at the final round, is at most $1 / 2 0 0 0$

The probability of an invalid Eliminate call is at most $\rho .$ By (5.1), the within-stage probability that the actual and comparison processes disagree is at most $C \eta / t ^ { 2 }$ . Choose the absolute constant c suficiently small that

$$
\rho \leq \frac { 1 } { 2 0 0 0 } , \qquad \frac { C \eta } { t ^ { 2 } } \leq \frac { 1 } { 2 0 0 0 }
$$

for every $t \geq 1$ . Step 1 bounds the probability of rejection at the sample limit by $1 / 2 0 0 0 .$ . Summing these four contributions gives a rejection probability below $1 / 2 0 0$ . Thus every stage with $B _ { t } \geq A F$ accepts with probability at least 199/200. □

## 6.3. Summing the stage costs. Let t be the first index such that

$$
B _ { t _ { * } } \geq A F .
$$

By minimality,

$$
B _ { t _ { * } } < 1 0 0 A F .
$$

Every stage uses fresh data and begins with all K arms. Hence Lemma 6.2, applied conditionally after each rejection, gives

$$
\mathbb { P } ( \mathrm { t h e ~ a l g o r i t h m ~ r e a c h e s ~ s t a g e ~ } t _ { * } + k ) \le 2 0 0 ^ { - k } , \qquad k \ge 0 .
$$

Since stage t uses at most $\lceil M B _ { t } \rceil$ samples,

$$
\begin{array} { r l r } {  { \mathbb { E } T \le \sum _ { t < t _ { * } } \lceil M B _ { t } \rceil + \sum _ { k \ge 0 } 2 0 0 ^ { - k } \lceil M 1 0 0 ^ { k } B _ { t _ { * } } \rceil } } \\ & { } & { \le C B _ { t _ { * } } \le C F . } \end{array}
$$

The reach probabilities tend to zero, so the algorithm also stops almost surely.

$$
a = \log ( 1 / \eta ) = \log ( 1 / \delta ) + \log ( 1 / c ) .
$$

Because $c$ is absolute and $\delta < 0 . 0 1$ , a is bounded by an absolute constant multiple of $\log ( 1 / \delta )$ Therefore

$$
\mathbb { E } T \le C \left\{ H \big ( \log ( 1 / \delta ) + \mathrm { E n t } ( I ) \big ) + D \log \log ( e ^ { e } / g ) \right\} ,
$$

Together with the correctness bound from the preceding section, this proves Theorem 1.2.

Proof of Corollary 1.3. Apply the lower bound in Theorem 1.1 to $H ( \log ( 1 / \delta ) + \operatorname { E n t } ( I ) )$ in Theorem 1.2. □

## Appendix A. Conventions in the original conjectures

Our formulation follows that of [2, section 1 and definition 3.1] with minor convention diferences. We record these below and explain how Theorems 1.1 and 1.2 still establish [2, conjectures 3.2 and 3.5] as formulated therein.

A.1. Gap-group boundary conventions. Choose a suboptimal arm i at random with probability $\Delta _ { i } ^ { - 2 } / H$ . Let R be the index of its gap group under our convention and S the corresponding index under the convention $[ 2 ^ { - s } , 2 ^ { - s + 1 } )$ in [2], including the group containing the endpoint 1. Every cell of either partition meets at most two cells of the other. Hence

$$
\operatorname { E n t } ( S \mid R ) \leq \log 2 , \qquad \operatorname { E n t } ( R \mid S ) \leq \log 2 .
$$

The entropy chain rule gives

$$
| \operatorname { E n t } ( R ) - \operatorname { E n t } ( S ) | \leq \log 2 .
$$

Thus the contributions H Ent(I) difer by an additive $O ( H )$ , absorbed by $H \log ( 1 / \delta )$ . The definition of the gap groups in [3, definition 1.7] already agrees with ours.

## A.2. Iterated logarithm. For $x = \log ( 1 / g ) \geq 0$

$$
0 \leq \log ( e + x ) - \log \operatorname* { m a x } \{ e , x \} \leq \log 2 .
$$

Our shifted iterated logarithm and the clipped form used in [2] therefore difer by at most a constant. Since $D \leq H$ , the resulting $O ( H )$ change is absorbed by $H \log ( 1 / \delta )$ . The unregularized logarithm is undefined at $g = 1$ and can be negative for large gaps; clipping makes the sample-complexity convention explicit.

A.3. Stopping conventions. The correctness definition in [2] only requires a correct output with probability at least $1 - \delta .$ , not that the algorithm also almost surely stops. The algorithms proving the upper bounds stop almost surely on every instance in $\boldsymbol { \mathcal { S } } _ { K }$ , and therefore satisfy either correctness definition. The lower-bound proof does not require the algorithm to stop on the Gaussian alternative instances, as it uses only their correct-output probabilities at finite horizons. If the expected count on the target instance is infinite, its lower bound is immediate. Otherwise the algorithm stops almost surely on that instance, permitting the limits in the lower-bound proof. Thus the bounds also cover the original correctness convention.

These comparisons preserve all bounds up to absolute constants. Together with Corollary 1.3, they give the stated formulations of the original conjectures.

## Appendix B. Sampling subroutines

We prove the subroutine guarantees used in Section 4 for 1-sub-Gaussian reward laws. Input sets may have tied optimal arms. Diferent calls use fresh samples. Within Fraction, arm identities are sampled independently across draws; conditional on these identities, sample blocks are independent. Fraction samples are also independent of the estimation blocks used to delete arms. All sample sizes are rounded up to integers. Accuracy parameters and diferences between mean thresholds are at most one, so rounding is absorbed in the cost bounds.

B.1. Mean estimation. For independent samples $X _ { 1 } , \ldots , X _ { n }$ from a 1-sub-Gaussian arm with mean $\mu ,$ independence gives

$$
\mathbb { E } \exp \left( \lambda \sum _ { s = 1 } ^ { n } ( X _ { s } - \mu ) \right) \leq e ^ { n \lambda ^ { 2 } / 2 } .
$$

Writing $\begin{array} { r } { \overline { { X } } _ { n } = n ^ { - 1 } \sum _ { s = 1 } ^ { n } X _ { s } } \end{array}$ and applying the Chernof bound with $\lambda = x$ and $\lambda = - x$ yields, for $x > 0$

$$
\begin{array} { r } { \mathbb { P } ( \overline { { X } } _ { n } - \mu \geq x ) \leq e ^ { - n x ^ { 2 } / 2 } , \qquad \mathbb { P } ( \overline { { X } } _ { n } - \mu \leq - x ) \leq e ^ { - n x ^ { 2 } / 2 } . } \end{array}\tag{B.1}
$$

This is the standard sub-Gaussian sample-mean bound; see [11, corollary 5.5]. For accuracy e and error level α, take

$$
n ( e , \alpha ) = \left\lceil 2 e ^ { - 2 } \log ( 2 / \alpha ) \right\rceil
$$

samples and return their average. Equation (B.1) gives

$$
\mathbb { P } ( | \widehat { \mu } - \mu | > e ) \leq 2 e ^ { - n e ^ { 2 } / 2 } \leq \alpha .
$$

The count is deterministic and of order $e ^ { - 2 } \log ( 1 / \alpha )$ on the stated parameter range. Fresh samples give the same bound conditional on the history before a call.

B.2. Median elimination. We use the median-elimination construction of [5], in the form stated in [3, fact 5.2]. Start with $S _ { 1 } = S$ . While $| S _ { j } | \geq 2$ , put

$$
e _ { j } = \frac { e } { 4 } \left( \frac { 3 } { 4 } \right) ^ { j - 1 } , \qquad a _ { j } = 0 . 0 1 \cdot 2 ^ { - j } .
$$

Estimate each current mean to accuracy $e _ { j } / 2$ with individual failure probability at most $a _ { j } / 3 _ { \mathrm { : } }$ , and retain the $\lceil | S _ { j } | / 2 \rceil$ arms with the largest estimates, using fixed tie-breaking. Return the remaining arm.

Conditional on the current set, fix one optimal arm. Its estimate is inaccurate with probability at most $a _ { j } / 3$ . If it is accurate and no arm whose mean is within $e _ { j }$ of its mean survives, then at least $\lceil \lvert S _ { j } \rvert / 2 \ ' \rceil$ retained arms have inaccurate estimates. The expected number of inaccurate estimates is at most $| S _ { j } | a _ { j } / 3$ , so Markov’s inequality bounds the probability of this event by $2 a _ { j } / 3$ . Thus the maximum true mean drops by more than $e _ { j }$ with conditional probability at most $a _ { j }$ . As

$$
\sum _ { j \geq 1 } e _ { j } = e , \qquad \sum _ { j \geq 1 } a _ { j } = 0 . 0 1 ,
$$

the returned arm has mean within e of the original maximum with probability at least 0.99.

Until termination, $| S _ { j } | \le 2 | S | 2 ^ { - ( j - 1 ) }$ . The deterministic total cost is bounded by

$$
C | S | e ^ { - 2 } \sum _ { j \geq 1 } \left( { \frac { 8 } { 9 } } \right) ^ { j - 1 } ( j + 1 ) = O ( | S | e ^ { - 2 } ) .
$$

Fresh samples give the same guarantee conditional on an arbitrary history before the call.

B.3. Fraction testing. Write $\varepsilon = u - l$ and $d = \theta _ { + } - \theta _ { - }$ . Draw

$$
m _ { F } = \left\lceil 3 6 d ^ { - 2 } \log ( 2 / \alpha ) \right\rceil
$$

independent uniform arm identities from S, with replacement. For each draw, obtain a fresh mean estimate of accuracy $\varepsilon / 2$ and error level $d / 6$ . Let $Y _ { j }$ indicate that this estimate is below $( l + u ) / 2$ Return True precisely when

$$
m _ { F } ^ { - 1 } \sum _ { j = 1 } ^ { m _ { F } } Y _ { j } > ( \theta _ { - } + \theta _ { + } ) / 2 .
$$

The variables $Y _ { j }$ are independent and identically distributed. If the fraction of arms with mean below l is at least $\theta _ { + }$ , then

$$
\mathbb { E } Y _ { j } \geq \theta _ { + } - d / 6 .
$$

If the fraction with mean below u is at most $\theta _ { - }$ , then

$$
\mathbb { E } Y _ { j } \leq \theta _ { - } + d / 6 .
$$

In the first case, False is an invalid answer; in the second, True is invalid. Either invalid answer requires a deviation of at least $d / 3$ from the Bernoulli mean. Hoefding’s inequality bounds its probability by

$$
\exp ( - 2 m _ { F } d ^ { 2 } / 9 ) \leq \alpha / 2 .
$$

These implications prove the stated True/False guarantees. The deterministic number of samples is

$$
O \left( \varepsilon ^ { - 2 } \log ( 1 / \alpha ) d ^ { - 2 } \log ( 1 / d ) \right)
$$

for $0 < \varepsilon \leq 1 , 0 < d \leq 0 . 1$ , and $0 < \alpha < 0 . 1$ . This is Algorithm 2 and Fact 5.3 of [3] with explicit rounding.

B.4. Elimination. We use the elimination procedure in [3, algorithm 3 and fact 5.4], returning immediately if the active set is empty and retaining arms whose estimates equal the deletion threshold. Fraction testing and deletion use separate samples. Put

$$
\varepsilon = u - l , \qquad v = ( l + u ) / 2 , \qquad q _ { j } = \frac { \alpha } { 1 0 \cdot 2 ^ { j } } \quad ( j \geq 1 ) .
$$

Start with $S _ { 1 } = S$ . At internal stage $j \colon$

(1) If $S _ { j }$ is empty, return it. Otherwise call Fraction with mean thresholds $l , v ,$ , fraction thresholds $0 . 0 5 , 0 . 1$ , and error level $q _ { j }$

(2) If False, return $S _ { j }$

(3) If True, estimate each current arm mean independently to accuracy $\varepsilon / 4$ with error level $q _ { j } .$ and retain precisely the arms whose estimates are at least $v + \varepsilon / 4$ . Continue with the retained set.

For the cost calculation only, set the active-set size to zero after the procedure returns. Conditional on a current set, a valid True answer implies that more than $0 . 0 5 | S _ { j } |$ arms have mean below v. Each such arm survives the independent estimation step with probability at most $q _ { j }$ . A False answer makes the number of active arms in the next internal stage zero. On an invalid Fraction answer, the number of active arms in the next internal stage is still at most $| S _ { j } |$ . Averaging over this event, whose probability is at most $q _ { j }$ , gives

$$
\mathbb { E } [ | S _ { j + 1 } | \ | \ S _ { j } ] \le [ 1 - 0 . 0 5 ( 1 - q _ { j } ) ^ { 2 } ] | S _ { j } | \le 0 . 9 6 | S _ { j } | .
$$

The contraction holds without conditioning on correctness. Iterating it gives

$$
\mathbb { E } | S _ { j } | \le 0 . 9 6 ^ { j - 1 } | S | .
$$

The conditional stage cost is at most

$$
C | S _ { j } | \varepsilon ^ { - 2 } [ \log ( 1 / \alpha ) + j ] .
$$

Here the Fraction cost is absorbed because $| S _ { j } | \geq 1$ while active. Summing the expected internalstage costs bounds the unconditional expected total number of samples by

$$
C | S | \varepsilon ^ { - 2 } \sum _ { j \ge 1 } 0 . 9 6 ^ { j - 1 } [ \log ( 1 / \alpha ) + j ] = O ( | S | \varepsilon ^ { - 2 } \log ( 1 / \alpha ) ) .
$$

The same estimate gives

$$
\mathbb { P } ( \mathrm { s t i l l ~ a c t i v e ~ a t ~ i n t e r n a l ~ s t a g e ~ } j ) \le \mathbb { E } | S _ { j } | \le 0 . 9 6 ^ { j - 1 } | S | .
$$

This tends to zero as $j \to \infty$ , so the subroutine terminates almost surely.

We next verify the fraction guarantee and bound the probability that specified arms of mean at least u are all removed. The probability that any internal Fraction answer is invalid is at most $\textstyle \sum _ { i } q _ { j } = \alpha / 1 0$ . If every such answer is valid, a nonempty set is returned only after a False answer and therefore has low-mean fraction less than 0.1. The empty set satisfies the non-strict inequality automatically. For each input arm, generate independent estimation blocks in advance for all internal stages, separately from the Fraction samples. If an arm has mean at least $u ,$ every accurate estimate is at least

$$
u - \varepsilon / 4 = v + \varepsilon / 4 ,
$$

so the arm cannot be removed unless one of its own estimates is inaccurate. For each arm, the event that at least one estimate is inaccurate has probability at most $\alpha / 1 0$ . These events are independent across arms. Consequently the probability that all k specified arms with mean at least u are removed is at most $( \alpha / 1 0 ) ^ { k } \le \alpha ^ { k }$ . This argument based on independent estimation errors follows [3, lemma B.9] and permits adaptive use of the blocks.

For one designated optimal arm of mean at least $u ,$ the probability of losing it or violating the bound on the number of returned arms with mean below l is at most $\alpha / 5 < \alpha$ . Fresh samples give these guarantees conditional on any history before the call.

## References

[1] Lijie Chen and Jian Li. On the optimal sample complexity for best arm identification, 2015. arXiv:1511.03774.

[2] Lijie Chen and Jian Li. Open problem: Best arm identification: Almost instance-wise optimality and the gap entropy conjecture. In Vitaly Feldman, Alexander Rakhlin, and Ohad Shamir, editors, Proceedings of the 29th Annual Conference on Learning Theory, volume 49 of Proceedings of Machine Learning Research, pages 1643–1646. PMLR, 2016.

[3] Lijie Chen, Jian Li, and Mingda Qiao. Towards instance optimal bounds for best arm identification. In Satyen Kale and Ohad Shamir, editors, Proceedings of the 2017 Conference on Learning Theory, volume 65 of Proceedings of Machine Learning Research, pages 535–592. PMLR, 2017.

[4] Brian M. Cho and Nathan Kallus. Exploration in the limit, 2026. arXiv:2601.00084.

[5] Eyal Even-Dar, Shie Mannor, and Yishay Mansour. Action elimination and stopping conditions for the multiarmed bandit and reinforcement learning problems. Journal of Machine Learning Research, 7(39):1079–1105, 2006.

[6] R. H. Farrell. Asymptotic behavior of expected sample size in certain one sided tests. The Annals of Mathematical Statistics, 35(1):36–72, 1964.

[7] Aurélien Garivier and Emilie Kaufmann. Optimal best arm identification with fixed confidence. In Vitaly Feldman, Alexander Rakhlin, and Ohad Shamir, editors, Proceedings of the 29th Annual Conference on Learning Theory, volume 49 of Proceedings of Machine Learning Research, pages 998–1027. PMLR, 2016.

[8] Kevin Jamieson, Matthew Malloy, Robert Nowak, and Sébastien Bubeck. lil’ UCB: An optimal exploration algorithm for multi-armed bandits. In Maria Florina Balcan, Vitaly Feldman, and Csaba Szepesvári, editors, Proceedings of the 27th Conference on Learning Theory, volume 35 of Proceedings of Machine Learning Research, pages 423–439. PMLR, 2014.

[9] Zohar Karnin, Tomer Koren, and Oren Somekh. Almost optimal exploration in multi-armed bandits. In Sanjoy Dasgupta and David McAllester, editors, Proceedings of the 30th International Conference on Machine Learning, volume 28(3) of Proceedings of Machine Learning Research, pages 1238–1246. PMLR, 2013.

[10] Emilie Kaufmann, Olivier Cappé, and Aurélien Garivier. On the complexity of best-arm identification in multi-armed bandit models. Journal of Machine Learning Research, 17(1):1–42, 2016.

[11] Tor Lattimore and Csaba Szepesvári. Bandit Algorithms. Cambridge University Press, 2020.

[12] Shie Mannor and John N. Tsitsiklis. The sample complexity of exploration in the multi-armed bandit problem. Journal of Machine Learning Research, 5:623–648, 2004.

[13] Max Simchowitz, Kevin Jamieson, and Benjamin Recht. The simulator: Understanding adaptive sampling in the moderate-confidence regime. In Satyen Kale and Ohad Shamir, editors, Proceedings of the 2017 Conference on Learning Theory, volume 65 of Proceedings of Machine Learning Research, pages 1794–1834. PMLR, 2017.