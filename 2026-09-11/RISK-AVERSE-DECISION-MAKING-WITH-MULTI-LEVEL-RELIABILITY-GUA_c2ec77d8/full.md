# RISK-AVERSE DECISION MAKING WITH MULTI-LEVEL RELIABILITY GUARANTEES

Amirmohammad Farzaneh and Osvaldo Simeone

Institute for Intelligent Networked Systems (INSI), Northeastern University London, London, UK a.farzaneh, o.simeone @nulondon.ac.uk

## ABSTRACT

Many applications in engineering, including wireless broadcasting, require designs that provide performance certificates at different target outage levels. This paper studies the problem of maximizing the weighted average of such certificates in the presence of uncertainty about the true system state. The problem is shown to be equivalent to an optimization over nested prediction sets, connecting to the literature on conformal prediction and extending prior art on single-level risk-averse decision making. Furthermore, we derive a dual formulation that decouples optimization across input values. Numerical experiments on a diversity-based wireless transmission system illustrate the cost of enforcing multi-level certificates with a single shared policy and trace the Pareto trade-off between multiple reliability levels.

Index Terms— Risk-averse decision making, conformal prediction, value at risk, graceful degradation, Lagrangian duality.

## 1. INTRODUCTION

Decisions under uncertainty require not only good nominal performance, but also predictable behavior as conditions deteriorate. For example, a wireless broadcast system may need to serve users whose connecting conditions range from line-of-sight to blocked propagation [1]; a control system may face unexpected load spikes [2]; and a learned classifier may encounter distributionally shifted inputs [3]. The standard formulation for optimal decision making includes an agent that observes features X, selects an action $a ( X )$ without knowing the true state Y , and receives utility $u ( a ( X ) , Y )$ A risk-neutral policy maximizes expected utility, whereas a riskaverse policy also controls unfavorable outcomes, certifying a single utility level that is attained with probability at least 1 − α [4]. However, a certificate at a single outage level does not control the shape of the tail of the utility distribution. For example, as illustrated in Fig. 1, two policies that are indistinguishable at level α may behave very differently at a more stringent level, $\mathrm { e . g . , } \alpha / 1 0$

Taking inspiration from differentiated quality of service in telecommunications [5, 6], we study a single policy that supports K utility certificates $\nu _ { 1 } ( X ) \ \geq \ \cdot \cdot \cdot \geq \ \nu _ { K } ( X )$ at outage levels $\alpha _ { 1 } \geq \cdots \geq \alpha _ { K }$ . Each pair $( \nu _ { k } , \alpha _ { k } )$ guarantees that utility $\nu _ { k } ( X )$ is attained in all but an $\alpha _ { k }$ -fraction of conditions. The ordering pairs stronger utility guarantees $\nu _ { k }$ with more permissive outage levels $\alpha _ { k }$ and weaker guarantees $\nu _ { k }$ with more stringent reliability requirements $\alpha _ { k }$ . Together, these certificates can enforce a form of graceful degradation in utility levels: the policy moves through progressively relaxed, but still certified, service levels as conditions worsen.

![](images/ed32c9eb099c52a8baaec1e367472e6970afa6b023bb5f766058051ff0e76065.jpg)  
Fig. 1. Two utility distributions with matching utility certificates at outage level $\alpha _ { 1 } = 0 . 2$ but different utility certificates at the more stringent level $\alpha _ { 2 } = \alpha _ { 1 } / 1 0 = 0 . 0 2$

The single-level instance, i.e., $K = 1$ , of the multi-level design problem described above was studied in [4], where value-at-risk optimization is shown to lead naturally to prediction sets and max–min decision rules. The role of prediction sets in the problem provides a formal justification for the use of conformal prediction as an uncertainty quantification strategy [7,8], as already proposed for a number of engineering applications [9, 10]. The framework in [4], however, does not extend directly to a setting with $K > 1$ levels, as applying it independently at each of the K outage levels would generally produce K different actions $\{ a _ { k } ( X ) \} _ { k = 1 } ^ { K }$ , and therefore would not define one deployable policy $a ( X )$ with graded guarantees.

In Section 2, we formulate the multi-level risk-averse decision problem, and Section 3 derives a dual characterization decoupling optimization across input values. In Section 4, we establish an equivalent formulation based on K nested prediction sets, extending the single-level equivalence of [4] and providing a formal motivation for the use of nested conformal prediction [11–13]. Finally, Section 5 uses a diversity-based wireless transmission example to illustrate the compromise between certificates at different service levels.

## 2. PROBLEM FORMULATION

Let $( X , Y ) \sim P _ { X Y }$ denote the pair of agent’s observation $X \in { \mathcal { X } }$ and unobserved state $Y \in \mathcal { D }$ . After observing $X = x .$ , the agent chooses an action $a ( x ) \in { \mathcal { A } }$ and receives utility $u ( a ( x ) , Y ) \ \in$ $[ 0 , u _ { \mathrm { m a x } } ]$ with $u _ { \mathrm { m a x } } < \infty$ . Assuming K service levels, fix maximum tolerated outage probabilities $\{ \alpha _ { k } \} _ { k = } ^ { K }$ satisfying the inequal-

ities

$$
1 > \alpha _ { 1 } \geq \cdot \cdot \cdot \geq \alpha _ { K } > 0 ,\tag{1}
$$

so that the required reliability $1 - \alpha _ { k }$ increases with the service level $k .$ For each service level $k ,$ we wish to identify a utility certificate $\nu _ { k } : \mathcal { X }  [ 0 , u _ { \mathrm { m a x } } ]$ meeting the target reliability $1 - \alpha _ { k } , \mathrm { i . e . }$

$$
\operatorname* { P r } [ u ( a ( X ) , Y ) \geq \nu _ { k } ( X ) ] \geq 1 - \alpha _ { k } ,\tag{2}
$$

where the certificates naturally obey the pointwise ordering

$$
\nu _ { 1 } ( x ) \geq \nu _ { 2 } ( x ) \geq \cdots \geq \nu _ { K } ( x ) \quad { \mathrm { f o r ~ a l l ~ } } x \in { \mathcal { X } }\tag{3}
$$

By condition (3), larger utility certificates $\nu _ { k } ( x )$ are assigned to more permissive outage requirements $\alpha _ { k }$

Definition 1 $\scriptstyle ( \mathrm { R A - D P O } ( \alpha ) )$ ). For a vector oftarget outage probabilities ${ \pmb { \alpha } } = ( \alpha _ { 1 } , \dots , \alpha _ { K } )$ satisfying the inequalities (1), given a vector of non-negative weights $\{ w _ { k } \} _ { k = 1 } ^ { K }$ with $w _ { k } \geq 0$ and $\textstyle \sum _ { k = 1 } ^ { K } w _ { k } =$ 1, the K-level risk-averse decision policy optimization (RA-DPO) problem is defined as

$$
\begin{array} { r l } { \displaystyle \operatorname* { m a x } _ { \substack { \boldsymbol { x } = 1 , \boldsymbol { \nu } _ { 1 } ( \cdot ) , \ldots , \boldsymbol { \nu } _ { K } ( \cdot ) } } } & { \displaystyle \sum _ { k = 1 } ^ { K } w _ { k } \mathbb { E } _ { \boldsymbol { X } } [ \nu _ { k } ( \boldsymbol { X } ) ] } \\ { \displaystyle \boldsymbol { a } ( \cdot ) , \boldsymbol { \nu } _ { 1 } ( \cdot ) , \ldots , \boldsymbol { \nu } _ { K } ( \cdot ) } & { \displaystyle \mathrm { P r } [ \boldsymbol { u } ( \boldsymbol { a } ( \boldsymbol { X } ) , \boldsymbol { Y } ) \geq \nu _ { k } ( \boldsymbol { X } ) ] \geq 1 - \alpha _ { k } , } \\ { \displaystyle \qquad \quad } & { \displaystyle k = 1 , \ldots , K , } \\ { \displaystyle \qquad \nu _ { 1 } ( \boldsymbol { x } ) \geq \nu _ { 2 } ( \boldsymbol { x } ) \geq \cdot \cdot \cdot \geq \nu _ { K } ( \boldsymbol { x } ) , } \\ { \displaystyle \boldsymbol { x } \in \mathcal { X } , } \end{array}\tag{4}
$$

and its optimal value is denoted as $\mathrm { O P T } ( \alpha )$

When $K = 1$ and $w _ { 1 } = 1$ , problem (4) corresponds to the single-level RA-DPO problem studied in [4]. For $K > 1$ , the certificates $\{ \nu _ { k } ( x ) \} _ { k = } ^ { K }$ <sub>1</sub> cannot generally be optimized in isolation, since they all depend on the same action $a ( x )$ that needs to cater to all service levels. The next basic result connects problem (4) to its singlelevel counterpart, whose optimal value at outage level α is denoted as $\mathrm { O P T } ( \alpha )$

Lemma 1. The optimal value $\mathrm { O P T } ( \alpha )$ for $R A { \cdot } D P O ( \alpha )$ can be bounded as

$$
\mathrm { O P T } ( \alpha _ { K } ) \leq \mathrm { O P T } ( \alpha ) \leq \sum _ { k = 1 } ^ { K } w _ { k } \mathrm { O P T } ( \alpha _ { k } ) ,\tag{5}
$$

where $\mathrm { O P T } ( \alpha )$ is the optimal value of $R A { - } D P O ( \alpha )$ with $K = 1$

The lower bound in (5) is the optimal value for the policy designed for the strictest reliability level $\alpha _ { K }$ only. In contrast, the upper bound allows every service level k to select its own action policy $a _ { k } ( X )$ and is therefore generally unattainable when one common policy $a ( X )$ must serve all levels. The next sections elaborate on the optimization of problem (4).

## 3. DUAL FORMULATION

Problem (8) jointly optimizes the $K + 1$ functions $a ( \cdot ) , t _ { 1 } ( \cdot ) , . . . ,$ $t _ { K } ( \cdot )$ , whose values at different inputs are coupled by the expected coverage constraints. In this section, we recast (4) as a joint optimization over an action and a vector of local reliability allocations.

We then show that this problem can be addressed via a dual formulation that decouples across values of x. To start, define

$$
t _ { k } ( x ) = \operatorname* { P r } [ u ( a ( x ) , Y ) \geq \nu _ { k } ( x ) \mid X = x ]\tag{6}
$$

as the reliability at service level k and input x, so that the constraint in (2) can be written as $\mathbb { E } _ { X } [ t _ { k } ( X ) ] \geq 1 - \alpha _ { k }$ , and the constraint (3) enforces the pointwise ordering $t _ { 1 } ( x ) \leq \dots \leq t _ { K } ( x )$ . For a given allocation $t _ { k } ( x )$ of outage levels, the largest utility that can be certified for action a is the conditional quantile

$$
\begin{array} { r l r } {  { Q _ { t _ { k } ( x ) } ( x ; a ) = \operatorname* { s u p } \bigl \{ v \in [ 0 , u _ { \operatorname* { m a x } } ] : } } \\ & { } & { \operatorname* { P r } [ u ( a , Y ) \geq v \mid X = x ] \geq t _ { k } ( x ) \bigr \} . } \end{array}\tag{7}
$$

Substituting these expressions in the objective of (4), RA-DPO(α) can be equivalently stated as

$$
\begin{array} { r l } { \displaystyle \operatorname* { m a x } _ { \alpha ( \cdot ) , t _ { 1 } ( \cdot ) , \dots , t _ { K } ( \cdot ) } } & { \displaystyle \sum _ { k = 1 } ^ { K } w _ { k } \mathbb { E } _ { X } \left[ Q _ { t _ { k } ( X ) } \big ( X ; a \big ( X \big ) \big ) \right] } \\ { \mathrm { ~ s . t . ~ } } & { \mathbb { E } _ { X } \big [ t _ { k } \big ( X \big ) \big ] \geq 1 - \alpha _ { k } , } \\ & { \qquad k = 1 , \dots , K , } \\ & { \qquad 0 \leq t _ { 1 } ( x ) \leq \cdot \cdot \cdot \leq t _ { K } ( x ) \leq 1 , } \\ & { \qquad x \in \mathcal { X } . } \end{array}\tag{8}
$$

Dualizing the constraints in (8), we now demonstrate that the maximization in (8) can be solved independently at each input x. To see this, let

$$
\mathcal { T } = \{ \mathbf { t } ( x ) \in [ 0 , 1 ] ^ { K } : t _ { 1 } ( x ) \leq \cdots \leq t _ { K } ( x ) \} ,
$$

so that the dual problem [14] for (8) can be written as

$$
\operatorname* { m i n } _ { \beta \geq 0 } (  \begin{array} { c } { { } } \\ { { d ( \beta ) = \mathbb E _ { X } [ \displaystyle \operatorname* { m a x } _ { a ( X ) \in { \cal A } } \{ \displaystyle \sum _ { k = 1 } ^ { K } w _ { k } Q _ { t _ { k } ( X ) } \big ( X ; a ( X ) \big ) } \\ { { } } \\ { { \tan ( { { X } \atop { t } ( X ) \in { \cal T } } \ } ( \begin{array} { c } { { K } } \\ { { k - 1 } } \\ { { + \displaystyle \sum _ { k = 1 } ^ { K } \beta _ { k } t _ { k } ( X ) } } \end{array} ) ] \} } } \\ { { } \\ { { } - \displaystyle \sum _ { k = 1 } ^ { K } \beta _ { k } ( 1 - \alpha _ { k } ) } } \end{array} \} ) )  { ~ . ~ }\tag{9}
$$

Generalizing [4, Theorem 3.2], solving problem (9) is shown next to yield a solution also for problem (8).

Theorem 1. Under the stated regularity conditions detailed in Appendix B, the dual problem (9) admits an optimal solution ${ \boldsymbol { \beta } } ^ { * } \geq \mathbf { 0 } ,$ and an optimal solutionfor problem (8) can befound separatelyfor each input $x \in \mathcal { X }$ as

$$
\begin{array} { l }  { \displaystyle ( a ^ { * } ( x ) , { \bf t ^ { * } } ( x ) ) \in \underset { a ( x ) \in { \cal T } } { \mathrm { a r g } \operatorname* { m a x } } \biggl \{ \displaystyle \sum _ { k = 1 } ^ { K } w _ { k } Q _ { t _ { k } ( x ) } ( x ; a ( x ) ) } \end{array}\tag{10}
$$

## 4. PREDICTION-SET FORMULATION

In this section, we show that problem (4) admits an equivalent formulation expressed in terms of K nested prediction sets $C _ { k } ( x ) \subseteq { \mathcal { V } }$ with

$$
C _ { 1 } ( x ) \subseteq \ldots \subseteq C _ { K } ( x )\tag{11}
$$

in the space of states Y. This view connects RA-DPO to nested conformal prediction [11–13], and gives the optimal decision rule produced by RA-DPO a worst-case, robust-optimization interpretation. Specifically, extending [4, Theorem 2.3], we have the following result.

Proposition 1. RA- $. D P O ( \alpha )$ is equivalent to the problem

$$
\begin{array} { r l } { \underset { C _ { 1 } ( \cdot ) \subseteq \cdots \subseteq C _ { K } ( \cdot ) } { \operatorname* { m a x } } } & { \mathbb { E } _ { X } \left[ \underset { a \in \mathcal { A } } { \operatorname* { m a x } } \underset { k = 1 } { \overset { K } { \sum } } w _ { k } \underset { y \in C _ { k } ( X ) } { \operatorname* { i n f } } u ( a , y ) \right] } \\ { s . t . } & { \mathrm { P r } [ Y \in C _ { k } ( X ) ] \geq 1 - \alpha _ { k } , \quad k = 1 , \dots , K } \end{array}\tag{12}
$$

in the sense that problems (4) and (12) have the same optimal value and optimal solutions for one problem yield optimal solutions for the other problem. Specifically, $i f ( a ^ { * } ( x ) , \nu _ { 1 } ^ { * } ( x ) , \ldots , \nu _ { K } ^ { * } ( x ) )$ solves $R A { \bf - } D P O ( \alpha )$ , then

$$
C _ { k } ^ { * } ( x ) = \{ y \in \mathcal { V } : u ( a ^ { * } ( x ) , y ) \geq \nu _ { k } ^ { * } ( x ) \}\tag{13}
$$

is optimal for (12). Conversely, $i f ( C _ { 1 } ^ { * } ( x ) , \dots , C _ { K } ^ { * } ( x ) )$ solves (12), then

$$
a ^ { * } ( x ) \in \mathop { \arg \operatorname* { m a x } } _ { a \in \mathcal { A } } \sum _ { k = 1 } ^ { K } w _ { k } \operatorname* { i n f } _ { y \in C _ { k } ^ { * } ( x ) } u ( a , y ) ,\tag{14}
$$

$$
\nu _ { k } ^ { * } ( x ) = \operatorname* { i n f } _ { y \in C _ { k } ^ { * } ( x ) } u ( a ^ { * } ( x ) , y )\tag{15}
$$

solve $R A { \cdot } D P O ( \alpha )$

By (13), each optimal prediction set is a utility superlevel set, and the ordering of certificates makes these sets nested. Moreover, the optimal policy (14) selects an action that maximizes the weighted sum of its worst-case utilities over all K sets, with (15) providing the corresponding utility certificates. The proof is provided in $\mathsf { A p - }$ pendix A.

## 5. NUMERICAL EXAMPLE

We illustrate the impact of graceful-degradation requirements on a wireless communication problem consisting of a two-channel diversity transmission system [15, 16]. The first channel is characterized by a power gain $G _ { 1 } \ \sim \ \mathrm { E x p } ( 1 )$ , corresponding to Rayleigh fading [16], while the second channel has power gain $G _ { 2 } \sim$ $\mathrm { G a m m a } ( 6 , 1 / 6 )$ , corresponding to Nakagami-6 fading [16] and is occasionally blocked. The blockage indicator Z is partially known and distributed as $Z \ \mid \ X = \ x \sim$ Bern $( q ( x ) )$ with $q ( x ) = 0 . 8 5 + 0 . 0 5 x$ , where $X \sim \mathrm { U n i f } [ - 1 , 1 ]$ is side information about link availability. The variables $X , G _ { 1 }$ , and $G _ { 2 }$ are mutually independent, and the unobserved state is $Y = ( Z , G _ { 1 } , G _ { 2 } )$ . The action $a ( x ) ~ \in ~ [ 0 , 1 ]$ is the fraction of the power budget assigned to the second channel. Upon maximum ratio combining [15], the combined channel gain is thus

$$
H ( a , Y ) = ( 1 - a ) G _ { 1 } + \kappa a Z G _ { 2 } ,\tag{16}
$$

where $\kappa = 1 1$ dB reflects the higher nominal gain of the second channel, and the utility is given by the transmission rate

$$
u ( a , Y ) = \operatorname* { m i n } \big \{ \log _ { 2 } ( 1 + \rho H ( a , Y ) ) , u _ { \operatorname* { m a x } } \big \} ,\tag{17}
$$

with $\rho = 1 0 ~ \mathrm { d B }$ and $u _ { \mathrm { m a x } } = 8$ bit/s/Hz. Overall, this setting models an always-available lower-gain channel, supplemented by a faster but blockage-prone channel.

Conditional on the blockage Z, the channel gain $H ( a , Y )$ is either a scaled exponential $( Z = 0 )$ or the sum of an exponential and an integer-shape Gamma variable $( Z = 1 )$ . Based on this, the conditional quantile $Q _ { t } ( x ; a )$ can be obtained by monotone numerical inversion for any fixed pair $( x , a )$ . To solve the dual problem (9), we approximate the continuous domains of $x , a ,$ and t using uniform grids containing 2001, 401, and 201 points, respectively, and pre-compute $Q _ { t } ( x ; a )$ on the resulting grid. For each multiplier vector $( \beta _ { 1 } , \beta _ { 2 } )$ , the pointwise maximization in (9) is solved exactly on these discrete grids. The ordering $t _ { 1 } ~ \le ~ t _ { 2 }$ is handled using a cumulative maximization over the admissible values of $t _ { 1 } .$ , avoiding explicit enumeration of every pair $( t _ { 1 } , t _ { 2 } )$ . The multipliers are obtained by coordinate bisection over finite intervals whose upper endpoints were verified to satisfy $\mathbb { E } _ { X } [ t _ { k } ^ { * } ( X ) ] \geq 1 - \alpha _ { k }$

We fix the first outage probability $\alpha _ { 1 } ~ = ~ 0 . 1 5$ , corresponding to 85% coverage, and vary the second outage probability $\alpha _ { 2 } \in$ $\{ 0 . 1 5 , 0 . 1 0 , 0 . 0 5 , 0 . 0 3 , 0 . 0 1 \}$ , corresponding to second-level coverages from 85% to 99%. We first set equal weights $w _ { 1 } ~ = ~ w _ { 2 }$ to isolate the effect of tightening the second-level reliability requirement. As shown in Fig. 2, lowering the outage probability $\alpha _ { 2 }$ tightens this requirement and therefore decreases the expected certificates $\mathbb { E } _ { X } [ \nu _ { 1 } ( X ) ]$ and $\mathbb { E } _ { X } [ \nu _ { 2 } ( X ) ]$ ]. We also show the independently optimized values $\mathrm { O P T } ( \alpha _ { 1 } )$ and $\mathrm { O P T } ( \alpha _ { 2 } )$ , quantifying the cost of using one action to support both reliability levels, rather than optimizing either certificate in isolation (see Lemma 1).

Varying the weights w and w traces the trade-off between the two expected certificates $\mathbb { E } _ { X } [ \nu _ { 1 } ( X ) ]$ and $\mathbb { E } _ { X } [ \nu _ { 2 } ( X ) ]$ ]. In ${ \mathrm { F i g . ~ } } 3 ,$ we fix the first outage probability $\alpha _ { 1 } = 0 . 2 \ :$ corresponding to 80% coverage, and fix different values for the second outage probability α<sub>2</sub> $\in \{ 0 . 2 , 0 . 1 5 , 0 . 1 0 , 0 . 0 5 , 0 . 0 3 , 0 . 0 1 \}$ . The figure highlights the price paid in terms of certificate $\mathbb { E } _ { X } [ \nu _ { 1 } ( X ) ]$ in order to increase the second level certificate $\mathbb { E } _ { X } [ \nu _ { 2 } ( X ) ]$ . Moreover, decreasing the outage probability $\alpha _ { 2 }$ shifts each frontier downward, since the second certificate must hold at higher reliability.

## 6. CONCLUSION

In this work, we have introduced a risk-averse decision framework in which a single action policy supports an ordered hierarchy of utility certificates. The formulation is shown to be equivalent to an optimization over nested prediction sets and admits a dual characterization with one scalar multiplier per service level. The resulting optimal policy lies between the strictest single-level solution and the weighted collection of independently optimized single-level solutions. A numerical example involving a diversity-based communication system illustrates how the shared action couples the two expected certificates. Developing more efficient methods for solving the pointwise problems, particularly for higher-dimensional applications, and constructing a distribution-free finite-sample calibration procedure for the full hierarchy are natural directions for future work.

![](images/41d220b15c41a7c5455a412af9bcbf46eab61fdd4ba2381bf2d113ed40a7beb5.jpg)  
Fig. 2. Expected certificates $\mathbb { E } _ { X } [ \nu _ { 1 } ( X ) ]$ and $\mathbb { E } _ { X } [ \nu _ { 2 } ( X ) ]$ at equal weights $w _ { 1 } = w _ { 2 }$ for fixed outage probability $\alpha _ { 1 } = 0 . 1 5$ and varying $\alpha _ { 2 }$ . The logarithmic horizontal axis increases from the most stringent second-level requirement $\alpha _ { 2 } ~ = ~ 0 . 0 1$ on the left to the equal-outage case $\alpha _ { 2 } = \alpha _ { 1 } = 0 . 1 5$ on the right. The dotted gray curve is the independently optimized single-level value $\mathrm { O P T } ( \alpha _ { 2 } )$

![](images/9b05e782fcc5413e587d7a0cab9330cad5498654c59585297c4504925d65aa5d.jpg)  
Fig. 3. Supported Pareto frontiers of the expected certificates for fixed outage probability $\alpha _ { 1 } = 0 . 2 0$ and selected values of $\alpha _ { 2 }$ . Along each frontier, the weight w<sub>1</sub> increases from left to right.

## A. PROOF OF PROPOSITION 1

We prove equality of the optimal values in both directions. First, take a feasible RA-DPO solution $( a , \nu _ { 1 } , \ldots , \nu _ { K } )$ and define $C _ { k }$ as in (13). The certificate ordering makes these sets nested, and

$$
\{ Y \in C _ { k } ( X ) \} = \{ u ( a ( X ) , Y ) \geq \nu _ { k } ( X ) \} ,\tag{18}
$$

so every set satisfies the required marginal coverage. If $C _ { k } ( x )$ is nonempty, then

$$
\operatorname* { i n f } _ { y \in C _ { k } ( x ) } u ( a ( x ) , y ) \geq \nu _ { k } ( x ) .\tag{19}
$$

The same inequality holds for an empty set under our convention, since $\nu _ { k } ( x ) \leq u _ { \mathrm { m a x } }$ . Evaluating the objective in (12) at the original action $a ( x )$ therefore gives a value at least as large as the RA-DPO value. It follows that the RA-CPO optimum is no smaller than the RA-DPO optimum.

Conversely, take any feasible nested family $\big ( C _ { 1 } , \dots , C _ { K } \big )$ and define $a ^ { * }$ and $\nu _ { k } ^ { * }$ by (14)–(15). Since $C _ { k } ( x ) \subseteq C _ { k + 1 } ( x )$ , taking the infimum over the larger set cannot increase the result; hence the induced certificates are ordered. Furthermore,

$$
\{ Y \in C _ { k } ( X ) \} \subseteq \{ u ( a ^ { * } ( X ) , Y ) \geq \nu _ { k } ^ { * } ( X ) \} .\tag{20}
$$

Thus the induced policy and certificates are feasible for RA-DPO. Their RA-DPO objective is exactly the RA-CPO objective of the original sets. The RA-DPO optimum is therefore no smaller than the RA-CPO optimum. Combining the two inequalities proves equality, and applying the constructions to optimal solutions gives the stated correspondences.

## B. PROOF OF THEOREM 1

We proceed under the following regularity conditions: $( \mathrm { i } ) ~ \mathcal { X }$ is a standard Borel space and $P _ { X }$ is non-atomic; (ii) $\mathcal { A } \subset \mathbb { R } ^ { d }$ is compact; and (iii) for $P _ { X }$ -almost every $x ,$ the mapping $( a , \mathbf { t } ) \mapsto$ $\begin{array} { r } { \sum _ { k } w _ { k } Q _ { t _ { k } ( x ) } ( x ; a ) } \end{array}$ is upper semicontinuous on $\mathcal { A } \times T$ and jointly measurable in $( x , a , \mathbf { t } )$ . The utility $u \ : \ A \times \ y \  \ [ 0 , u _ { \mathrm { m a x } } ]$ is bounded as in Section 2.

The argument follows the convexify–dualize–recover proof of [4, Theorem 3.2 and Appendix $\mathrm { A } . 4 ] ,$ applied jointly to the ordered allocation vector. Write $\begin{array} { r } { G ( x , a ( x ) , \mathbf { t } ( x ) ) = \sum _ { k } w _ { k } Q _ { t _ { k } ( x ) } \big ( x ; a ( x ) \big ) } \end{array}$ and introduce the hypograph correspondence

$$
\begin{array} { r l r } & { \Gamma ( x ) = \bigcup _ { a ( x ) \in \mathcal { A } } \big \{ ( { \mathbf t } ( x ) , r ) : { \mathbf t } ( x ) \in T , } & \\ & { } & { \qquad 0 \leq r \leq G ( x , a ( x ) , { \mathbf t } ( x ) ) \big \} . } \end{array}
$$

Joint upper semicontinuity and compactness of $\mathcal { A } \times T$ make $\Gamma ( x )$ compact-valued, while the stated measurability conditions ensure measurable selections [17, Theorem 14.37]. Its Aumann integral $\begin{array} { r } { S = \int \Gamma ( x ) d P _ { X } ( x ) } \end{array}$ is compact and, because $P _ { X }$ is non-atomic, convex [18, Theorems 1, 3, and 4]. The reduced problem (8) is therefore equivalent to maximizing r over $( \mathbf { m } , r ) \in \mathcal { S }$ subject to $\textbf { m } \geq \textbf { 1 } - \alpha$ . Every policy gives such a point, and a measurable action realizing the upper boundary can be selected in the reverse direction. Thus including the hypograph only adds dominated objective values and does not change the optimum.

For any $\varepsilon ~ \in ~ ( 0$ , min<sub>k</sub> $\alpha _ { k } )$ , the ordered constant allocation $t _ { k } ( x ) = 1 - \alpha _ { k } + \varepsilon$ is strictly feasible. Strong duality for this finitedimensional convex problem therefore yields multipliers ${ \boldsymbol { \beta } } ^ { * } \geq \mathbf { 0 }$ primal feasibility $\mathbb { E } _ { X } [ t _ { k } ^ { * } ( X ) ] \ge 1 - \alpha _ { k }$ , and the complementaryslackness conditions $\beta _ { k } ^ { * } \bigl ( \mathbb { E } _ { X } [ t _ { k } ^ { * } ( X ) ] - ( 1 - \alpha _ { k } ) \bigr ) \ = \ 0$ for $k \ =$ $1 , \ldots , K$ . Maximizing the corresponding Lagrangian over $s$ is equivalent to maximizing its integrand pointwise, and a measurable maximizing selection exists by the same regularity conditions. Selecting an action that attains the upper boundary of $\Gamma ( x )$ gives exactly (10). Finally, setting $\nu _ { k } ( x ) = Q _ { t _ { k } ^ { * } ( x ) } ( x ; a ^ { * } ( x ) )$ recovers the certificates that solve $\mathrm { R A - D P O } ( \alpha )$

## Acknowledgments

This work was supported by the European Research Council (ERC) under the European Union’s Horizon Europe program (grant agreement No. 101198347). The work of O. Simeone was also supported by an EPSRC Open Fellowship (EP/W024101/1) and by the EPSRC project EP/X011852/1.

## C. REFERENCES

[1] Roy Karasik, Osvaldo Simeone, and Shlomo Shamai Shitz, “Learning to broadcast with layered division multiplexing,” in 2022 IEEE International Symposium on Information Theory (ISIT). IEEE, 2022, pp. 2696–2701.

[2] Karl Johan Astr ˚ om and Richard M. Murray, ¨ Feedback Systems: An Introduction for Scientists and Engineers, Princeton University Press, Princeton, NJ, 2008.

[3] Joaquin Quinonero-Candela, Masashi Sugiyama, Anton˜ Schwaighofer, and Neil D. Lawrence, Eds., Dataset Shift in Machine Learning, The MIT Press, 12 2008.

[4] Shayan Kiyani, George J. Pappas, Aaron Roth, and Hamed Hassani, “Decision theoretic foundations for conformal prediction: Optimal uncertainty quantification for risk-averse agents,” in Forty-second International Conference on Machine Learning, 2025.

[5] Dapeng Wu and Rohit Negi, “Effective capacity: a wireless link model for support of quality of service,” IEEE Transactions on wireless communications, vol. 2, no. 4, pp. 630–643, 2003.

[6] Petar Popovski, Jimmy J Nielsen, Cedomir Stefanovic, Elisabeth De Carvalho, Erik Strom, Kasper F Trillingsgaard, Alexandru-Sabin Bana, Dong Min Kim, Radoslaw Kotaba, Jihong Park, et al., “Wireless access for ultra-reliable lowlatency communication: Principles and building blocks,” IEEE Network, vol. 32, no. 2, pp. 16–23, 2018.

[7] Vladimir Vovk, Alexander Gammerman, and Glenn Shafer, Algorithmic learning in a random world, Springer, 2005.

[8] Anastasios N Angelopoulos and Stephen Bates, “Conformal prediction: A gentle introduction,” Foundations and Trends in Machine Learning, vol. 16, no. 4, pp. 494–591, 2023.

[9] Osvaldo Simeone, Sangwoo Park, and Matteo Zecchin, “Conformal calibration: Ensuring the reliability of black-box ai in wireless systems,” IEEE Communications Magazine, 2026.

[10] Lars Lindemann, Yiqi Zhao, Xinyi Yu, George J Pappas, and Jyotirmoy V Deshmukh, “Formal verification and control with conformal prediction: Practical safety guarantees for autonomous systems,” IEEE Control Systems, vol. 45, no. 6, pp. 72–122, 2025.

[11] Arun K Kuchibhotla and Richard A Berk, “Nested conformal prediction sets for classification with applications to probation data,” The Annals ofApplied Statistics, vol. 17, no. 1, pp. 761– 785, 2023.

[12] Eduardo Ochoa Rivera and Ambuj Tewari, “Online conformal prediction: Enforcing monotonicity via online optimization,” arXiv preprint arXiv:2605.12668, 2026.

[13] Tiffany Ding, Isaac Gibbs, and Ryan J Tibshirani, “Calibrated multi-level quantile forecasting,” arXiv preprint arXiv:2512.23671, 2025.

[14] Lieven Vandenberghe and Stephen Boyd, Convex optimization, vol. 1, Cambridge university press Cambridge, 2004.

[15] David Tse and Pramod Viswanath, Fundamentals of Wireless Communication, Cambridge University Press, Cambridge, UK, 2005.

[16] Andrea Goldsmith, Wireless Communications, Cambridge University Press, Cambridge, UK, 2005.

[17] R Tyrrell Rockafellar and Roger JB Wets, Variational analysis, Springer, 1998.

[18] Robert J Aumann, “Integrals of set-valued functions,” Journal of mathematical analysis and applications, vol. 12, no. 1, pp. 1–12, 1965.