# Nearly Minimax-Optimal Regret for Linear Contextual Bandits with Arbitrary Adaptive Action Sets

Tianyuan Jin

Data Science and Analytics Thrust

The Hong Kong University of Science and Technology (Guangzhou)

## Abstract

We study stochastic linear contextual bandits with arbitrary action menus that may depend on the interaction history. We establish matching upper and lower bounds, up to logarithmic factors. Let d be the dimension, K be the menu size, and T the time horizon. For $2 \leq K \leq d ,$ , we prove an upper bound $\widetilde { \cal O } ( K ^ { 1 / 4 } \sqrt { d T } )$ . When $T \geq d ^ { 2 }$ , we further prove a lower bound $\Omega ( K ^ { 1 / 4 } \sqrt { d T } )$ Thus, for $T \geq d ^ { 2 }$ and $2 \leq K \leq d ,$ the upper and lower bounds match up to logarithmic factors, and the polynomial dependence on K is optimal. Compared with the $\widetilde { { \cal O } ( \sqrt { d K T } ) }$ bound of Foster and Rakhlin [7], our upper bound improves the dependence on K by a factor of $K ^ { 1 / 4 }$

For $K \geq d .$ , we prove an upper bound $\widetilde { O } _ { d , T } \left( \sqrt { d T } \operatorname* { m i n } \{ \sqrt { d } , ( d \log K ) ^ { 1 / 4 } \} \right)$ and a lower bound $\begin{array} { r } { \Omega \left( \sqrt { d T } \operatorname* { m i n } \left\{ \sqrt { d } , \left( \frac { d \log K } { \log ( 2 d ) } \right) ^ { 1 / 4 } \right\} \right) } \end{array}$ . Here, $\widetilde { O } _ { d , T }$ omits logarithmic factors only in $d , T .$ In particular, for polynomially large $K \geq d ,$ the upper and lower bounds both scale as $d ^ { 3 / 4 } { \sqrt { T } }$ up to logarithmic factors, improving the standard $\widetilde { O } ( d \sqrt { T } )$ bound [1] by a factor of $d ^ { 1 / 4 }$ . As K grows further, the regret smoothly recovers the $d \sqrt { T }$ scale once log K reaches order d.

The authors developed the main ideas and guided the overall direction of the work. GPT-5.6 Sol Pro assisted in developing some of the proofs based on the ideas provided by the authors.

## 1 Introduction

Contextual bandits provide a fundamental framework for sequential decision-making under partial feedback, with broad applications in adaptive routing, personalized recommendation, and mobile health interventions. In this paper, we study linear contextual bandits. The problem proceeds in $T$ rounds. A parameter $\theta \in B _ { 2 } ^ { d }$ is fixed before the interaction. At each round $t \in [ T ]$ , the learner observes a menu of actions, represented by feature vectors $\mathcal { A } _ { t } = ( x _ { t , 1 } , \ldots , x _ { t , K } ) , \| x _ { t , i } \| _ { 2 } \leq 1$ Here, we write the menu as containing exactly K actions. Based on the current menu and past observations, the learner selects an action $A _ { t } \in [ K ]$ and then observes the reward $Y _ { t } = x _ { t , A _ { t } } ^ { \top } \theta + \varepsilon _ { t }$ $\varepsilon _ { t } \sim \mathcal { N } ( 0 , 1 )$ , where the current noise is independent of everything fixed before it is drawn. In linear bandits, the learner’s goal is to minimize the diference between the total reward obtained by selecting an optimal action $i _ { t } ^ { * } \in \arg \operatorname* { m a x } _ { i \in [ K ] } x _ { t , i } ^ { \top } \theta$ at each round t and the total reward obtained by the learner.

There are two settings depending on how the menus $\{ \mathcal { A } _ { t } \} _ { t \in [ T ] }$ are chosen. The first is the oblivious setting, where the entire sequence $\{ \mathcal { A } _ { t } \} _ { t \in [ T ] }$ is chosen by the adversary at time 0. SupLinUCB-style methods achieve regret of order $\tilde { O } \left( \sqrt { d T } \right) \ [ 5 , 6 , 8 ]$ . On the lower-bound side, Chu et al. [6] show a lower bound of $\Omega ( { \sqrt { d T } } )$ and, for $K \leq 2 ^ { d / 2 }$ , Li et al. [8] further improve it to $\Omega ( \sqrt { d T \log K \log T } )$ .

The second is the adaptive setting, where the menu $\boldsymbol { A } _ { t }$ may depend on the fixed parameter θ and the complete past history $H _ { t - 1 } = ( \boldsymbol { A } _ { 1 } , \boldsymbol { A } _ { 1 } , Y _ { 1 } , \ldots , \boldsymbol { A } _ { t - 1 } , \boldsymbol { A } _ { t - 1 } , Y _ { t - 1 } )$ and θ, that is $\mathcal { A } _ { t } = \mathcal { A } _ { t } ( \theta , H _ { t - 1 } )$ In this setting, self-normalized confidence methods obtain $\widetilde { O } ( d \sqrt { T } )$ regret [1]. SquareCB reduces contextual bandits to online regression and, for a d-dimensional linear class, yields $\widetilde { \cal O } ( \sqrt { d K T } )$ regret [7]. A natural open question is:

Table 1: Comparison of regret bounds. Our lower bounds require $T \geq d ^ { 2 }$
<table><tr><td>Method</td><td>Menus</td><td>Regret Bound</td></tr><tr><td>LinRel [5] and SupLinUCB [6]</td><td>Oblivious</td><td> $O ( \sqrt { d T \log ^ { 3 } ( K T ) } )$ </td></tr><tr><td>VCL-SupLinUCB [8]</td><td>Oblivious</td><td> $O ( \sqrt { d T \log T \log K } ) \mathrm { p o l y } ( \log \log ( K T ) )$ </td></tr><tr><td>Lin-TS [3, 4]</td><td>Adaptive</td><td> $O ( d \log T \sqrt { T \log K } )$ </td></tr><tr><td>Lin-UCB/OFUL [1]</td><td>Adaptive</td><td> $O ( d \sqrt { T } \log T )$ </td></tr><tr><td>SquareCB [7]</td><td>Adaptive</td><td> $O ( \sqrt { d K T \log T } )$ </td></tr><tr><td>Feel-Good TS [12]</td><td>Adaptive</td><td> $O ( \operatorname* { m i n } \{ \sqrt { d K T \log T } , d \sqrt { T \log T } \} )$ </td></tr><tr><td>Theorem 3  $( 2 \leq K \leq d )$ </td><td>Adaptive</td><td> $O ( K ^ { 1 / 4 } \sqrt { d T } \log ^ { 3 / 4 } T )$ </td></tr><tr><td>Lower Bound: Theorem  $1 ( 2 \leq K \leq d )$ </td><td>Adaptive</td><td> $\Omega ( K ^ { 1 / 4 } \sqrt { d T } )$ </td></tr><tr><td>Theorem 4  $( K \geq d )$ </td><td>Adaptive</td><td> $O ( d ^ { 3 / 4 } \sqrt { T } \left[ \operatorname* { m i n } \{ \log K , d \} \right] ^ { 1 / 4 } \log T )$ </td></tr><tr><td>Lower Bound: Theorem 1  $( K \geq d )$ </td><td>Adaptive</td><td> $\Omega ( d ^ { 3 / 4 } \sqrt { T } \left[ \operatorname* { m i n } \{ \log K / \log ( 2 d ) , d \} \right] ^ { 1 / 4 } )$ </td></tr></table>

Is $\widetilde { \cal O } ( \operatorname* { m i n } \{ \sqrt { d K T } , d \sqrt { T } \} )$ worst-case optimal (within logarithm factors) when the menus are chosen by an adaptive adversary? If not, what is the optimal regret bound?

Our answer to the first question is no, and our answer to the second has two regimes.

1. Small menus (Theorems 1 and 3). For $2 \leq K \leq d \leq T$ , we prove an upper bound $\widetilde { O } ( K ^ { 1 / 4 } \sqrt { d T } )$ . For $2 \leq K \leq d$ and $T \geq d ^ { 2 }$ , we also prove a lower bound $\Omega ( K ^ { 1 / 4 } \sqrt { d T } )$ , showing that the polynomial dependence on K in our upper bound cannot be improved. Compared with the previous $\widetilde { \cal O } ( \sqrt { d K T } )$ bound of Foster and Rakhlin [7], our upper bound improves the dependence on K by a factor of $K ^ { 1 / 4 }$

2. Large menus (Theorems 1 and 4). For $K \geq d$ and $T \geq d ,$ we prove an upper bound $\widetilde { O } _ { d , T } \left( \sqrt { d T } \operatorname* { m i n } \{ \sqrt { d } , ( d \log K ) ^ { 1 / 4 } \} \right)$ . For $K \geq d$ and $T \geq d ^ { 2 }$ , we also prove a lower bound $\begin{array} { r } { \Omega \left( \dot { \sqrt { d T } } \operatorname* { m i n } \left\{ \sqrt { d } , \left( \frac { d \log K } { \log ( 2 d ) } \right) ^ { 1 / 4 } \right\} \right) } \end{array}$ . Thus, in the long-horizon regime, the upper and lower bounds have the same polynomial dependence on $d , T ,$ , and log $K ,$ up to logarithmic factors. Compared with the standard $\widetilde { O } ( d \sqrt { T } )$ bound of Abbasi-Yadkori et al. [1], our upper bound improves the dependence on d by a factor of $( d / \log K ) ^ { 1 / 4 }$ when log $K \leq d .$ . In particular, for polynomially large $K \geq d ,$ the upper bound is $d ^ { 3 / 4 } { \sqrt { T } }$ up to logarithmic factors, yielding a $d ^ { 1 / 4 }$ improvement over the standard $d \sqrt { T }$ rate. As K increases further, the bound recovers the usual $d \sqrt { T }$ scale once log K is of order d.

Table 1 compares our results with prior work.

## 2 Problem Setting

In our problem, a learner and the environment interact over T rounds. In each round $t = 1 , \dots , T$

1. The environment chooses a menu $\mathcal { A } _ { t } = ( x _ { t , 1 } , \dots , x _ { t , K } )$ of K actions, where $x _ { t , i } \in \mathbb { R } ^ { d }$ and $\| x _ { t , i } \| _ { 2 } \leq 1$ . In particular, $\boldsymbol { A } _ { t }$ may depend on the fixed parameter θ and the past history $H _ { t - 1 } : = ( A _ { 1 } , A _ { 1 } , Y _ { 1 } , \ldots , A _ { t - 1 } , A _ { t - 1 } , Y _ { t - 1 } )$

2. After observing $\boldsymbol { \mathcal { A } } _ { t }$ , the learner chooses an action $A _ { t } \in [ K ]$

3. The learner observes $Y _ { t } = x _ { t , A _ { t } } ^ { \top } \theta + \varepsilon _ { t } .$ , where $\varepsilon _ { t } \sim \mathcal { N } ( 0 , 1 )$ is independent of everything fixed before it is drawn, where $\theta \in B _ { 2 } ^ { d }$ is a fixed and unknown target vector.

Write $x _ { t } : = x _ { t , A _ { t } }$ for the action actually played and $\mu _ { t , i } : = x _ { t , i } ^ { \top } \theta$ for the mean reward of action i. Let $i _ { t } ^ { \star } \in \arg \operatorname* { m a x } _ { i \in [ K ] } \mu _ { t , i }$ . The cumulative pseudo-regret is defined as

$$
R _ { T } : = \sum _ { t = 1 } ^ { T } \left( \mu _ { t , i _ { t } ^ { \star } } - \mu _ { t , A _ { t } } \right) .
$$

Our goal is to minimize $\mathbb { E } [ R _ { T } ]$ , where the expectation is over the learner’s randomization, the environment’s internal randomization, and the reward noise.

## 3 High Level Ideas

The upper and lower bounds are driven by the same observation. An adaptive adversary can exploit directions in which the learner is currently uncertain, but such attacks are not free. If the adversary repeatedly uses completely new directions, it quickly runs out of dimensions. If it reuses old directions, then the resulting observations reveal the directions in which the learner’s predictor is inaccurate. The lower bound shows how much of this information can be hidden, while the upper bound shows how to exploit the information that necessarily becomes visible.

## 3.1 Lower Bounds

Our basic idea is to divide the horizon into consecutive blocks, each exploiting a diferent collection of directions in which the learner’s estimate may still be inaccurate. In a d-dimensional problem, such prediction errors can persist along many diferent directions and might be changed over time. A block uses some of these directions to construct a hard menu; after the observations in that block reveal information about them, later blocks can use other directions in which substantial uncertainty remains. Within each block, we establish two properties. First, we exploit the learner’s uncertainty in the selected directions to construct a menu containing a hidden good action, while ensuring that the menu itself reveals only a small amount of information about θ. This leaves enough uncertainty in other directions to repeat the construction in later blocks. Second, we show that the rewards within the block reveal the hidden action only slowly. Consequently, the learner misses the hidden action on a constant fraction of the rounds and the block incurs substantial regret.

At the beginning of a block, the learner still has uncertainty about the fixed parameter θ. Let h and Σ be its current posterior mean and covariance, i.e., conditional on history H, $\theta \sim \mathcal { N } ( h , \Sigma )$ We choose $D = \lfloor d / 2 \rfloor - 1$ directions in which the posterior variance is of order $\sigma ^ { 2 }$ . Let $U \in \mathbb { R } ^ { d \times D }$ contain these directions and write $\boldsymbol { S } : = \boldsymbol { U } ^ { \top } \Sigma \boldsymbol { U }$ . The normalized estimation error in this subspace is $z : = S ^ { - 1 / 2 } U ^ { \top } ( \theta - h )$ , so $z \sim \mathcal { N } ( 0 , I _ { D } )$ , where $I _ { D }$ denotes the D-dimensional identity matrix. We now create K candidate actions and hide one special index $J \sim \operatorname { U n i f } ( [ K ] )$ . Draw independent $G 1 , \dots , G _ { K } \sim { \mathcal { N } } ( 0 , I _ { D } )$ . For $i \neq J$ , the action is generated from $G _ { i } .$ , while for the hidden index we use

$$
\rho z + \sqrt { 1 - \rho ^ { 2 } } G _ { J } ,
$$

where $\rho$ controls the level of correlation with the unknown parameter. The hidden action looks exactly like every other action. Since both z and $G _ { J }$ are standard Gaussian, $\rho z + \sqrt { 1 - \rho ^ { 2 } } G _ { J }$ is also standard Gaussian. Hence the distribution of the whole menu is the same for every value of $J ,$ and observing the menu does not reveal which index is hidden.

Nevertheless, the hidden action is better because it is correlated with the estimation error $z .$ This creates a reward gap of order $\Delta \asymp \sigma \rho \sqrt { D }$ . Thus a larger $\rho$ creates a larger reward gap, but also makes the hidden index J easier to identify from the observed rewards.

Rewards reveal the hidden action slowly. Whenever the learner fails to pull $^ { J , }$ it incurs regret of order $\Delta$ . The question is how quickly the learner can identify J from the observed rewards.

The key point is that the average KL divergence to a common reference experiment contributed by one reward is only $O ( \Delta ^ { 2 } / K )$ . To explain this, introduce a common reference experiment in which the planted shift associated with the hidden index is removed, while the menu, learner policy, residual uncertainty, and reward noise are kept unchanged. Conditional on the menu and $^ { J , }$ let $\mu _ { J }$ denote the planted shift in the conditional mean of $U ^ { \top } \theta$ . If the learner pulls action $i ,$ its reward mean under hidden index $J$ difers from that in the reference experiment by $( U ^ { \top } x _ { i } ) ^ { \top } \mu _ { J }$ . Since the reward noise is Gaussian, the corresponding KL divergence is proportional to $\left( ( U ^ { \top } x _ { i } ) ^ { \top } \mu _ { J } \right) ^ { 2 }$ . The random-menu geometry ensures that, when $J$ is uniform, $\mathbb { E } _ { J } \Big [ \big ( ( U ^ { \top } x _ { i } ) ^ { \top } \mu _ { J } \big ) ^ { 2 } \Big ] = O ( \Delta ^ { 2 } / K )$

Hence after n rounds the average KL divergence to the reference experiment is only $O ( n \Delta ^ { 2 } / K )$ ). $\mathrm { A s }$ long as $n \Delta ^ { 2 } / K = { \cal O } ( 1 )$ , the learner cannot reliably identify the hidden action and therefore misses it on a constant fraction of the rounds. The block consequently incurs $\Omega ( n \Delta )$ regret and can remain hard for $n \times K / \Delta ^ { 2 } \times K / ( \sigma ^ { 2 } D \rho ^ { 2 } )$ rounds.

Controlling information across blocks. We next determine how many hard blocks can be generated. Fix the variance scale $\sigma ^ { 2 }$ of the initial Gaussian prior; this quantity remains fixed throughout the construction. What changes from block to block is the posterior covariance. Writing $\Sigma _ { b }$ for the covariance at the beginning of block b, we have $\Sigma _ { 1 } = \sigma ^ { 2 } I _ { d }$ and $\Sigma _ { b + 1 } \preceq \Sigma _ { b }$ . Our block construction remains valid as long as $\mathrm { t r } ( \Sigma _ { b } ^ { - 1 } ) < 2 d / \sigma ^ { 2 }$

Each block consumes this posterior-precision budget in two ways. First, the menu itself is correlated with the unknown parameter. Using correlation strength $\rho$ in a D-dimensional uncertain subspace increases the posterior precision by $O ( D \rho ^ { 2 } / \sigma ^ { 2 } )$ per block. Second, the rewards observed within a block also reveal information about θ: a block of length n contributes at most $\begin{array} { r } { \sum _ { s } \| x _ { s } \| _ { 2 } ^ { 2 } \le n } \end{array}$ additional trace precision. Across the whole horizon, the latter contribution is therefore at most $T$ With the suficiently small constant in our choice $\sigma ^ { 2 } \asymp d / T$ , this uses only a constant fraction of the available budget $d / \sigma ^ { 2 }$ with $\sigma ^ { 2 } \asymp d / T$ . Hence the number of reusable hard blocks is essentially determined by the information leaked through the menus.

Thus, after B blocks, the menu contribution is $O ( B D \rho ^ { 2 } / \sigma ^ { 2 } )$ . Keeping the posterior uncertainty large enough for another hard block requires $B D \rho ^ { 2 } / \sigma ^ { 2 } \lesssim d / \sigma ^ { 2 }$ , and therefore allows about $B \asymp$ $d / ( D \rho ^ { 2 } )$ blocks. By the reward-information calculation above, each block can remain hard for $n \asymp$ $\dot { K } / ( \sigma ^ { 2 } \dot { D } \rho ^ { 2 } )$ rounds. Hence the total number of hard rounds is of order $d K / ( \sigma ^ { 2 } D ^ { 2 } \rho ^ { 4 } )$ . Substituting $\sigma ^ { 2 } \asymp d / T$ , this becomes $T K / ( D ^ { 2 } \rho ^ { 4 } )$ . We therefore choose $D \rho ^ { 2 } \asymp \sqrt { K }$ , so that the hard blocks occupy a constant fraction of the horizon. With this choice, the reward gap satisfies $\Delta \asymp \sigma \rho \sqrt { D } \asymp$ $K ^ { 1 / 4 } \sqrt { d / T }$ . Since the learner incurs regret of order $\Delta$ on a constant fraction of the $T$ rounds, we obtain $R _ { T } = \Omega ( K ^ { 1 / 4 } \sqrt { d T } )$

Large menus. The small-menu construction hides one good action among K candidate actions.   
To obtain a lower bound that continues to improve when $K \gg d ,$ we use a product construction.

We divide the available directions into m orthogonal groups. In each group $\ell ,$ we construct k candidate vectors $x _ { \ell , 1 } , \ldots , x _ { \ell , k }$ , with one hidden good candidate. An action is obtained by choosing one candidate from each group and summing the m selected vectors, scaled by $1 / \sqrt { m }$ . Thus each index tuple $( i _ { 1 } , \ldots , i _ { m } ) \in [ k ] ^ { m }$ defines one action, giving $k ^ { m }$ distinct actions, with the construction requiring $d \gtrsim m k$ . Choosing m and k so that mk $\lesssim d$ and $k ^ { m } \leq K$ allows the construction to exploit large menus.

The $1 / \sqrt { m }$ scaling keeps every action inside the unit ball. It also reduces the reward contribution of each group by $1 / \sqrt { m }$ . Consequently, the information about the hidden index in any one group is reduced by a factor $1 / m$ , while the regret contributions from the m groups add. Together, these efects yield an overall $\sqrt { m }$ amplification in the regret. Optimizing m and k under mk $\lesssim d$ and $k ^ { m } \leq K$ gives the large-menu lower bound, with the dependence on K entering through log K.

## 3.2 Upper Bound Part

The algorithm maintains $\begin{array} { r } { V _ { t } = I + \sum _ { s < t } x _ { s } x _ { s } ^ { \top } } \end{array}$ and $w _ { t , i } = \sqrt { x _ { t , i } ^ { \top } V _ { t } ^ { - 1 } x _ { t , i } }$ . For small menus, we use EXP4-IX [9], which competes with the best of N experts at cost $\widetilde { O } ( \sqrt { K T \log N } )$ . We construct a finite expert family with small logarithmic size that contains a good predictor.

To define the comparator expert, consider an auxiliary repair process that knows θ. Whenever some action has prediction error larger than $\alpha w _ { t , i }$ , the auxiliary process repairs its predictor in the corresponding direction. Each repair yields at least $\alpha ^ { 2 }$ progress, while the total linear information is controlled, as in OFUL [1], by log det $V _ { T + 1 } \leq d \log ( 1 + T / d )$ . Hence, with high probability, the auxiliary process makes at most $M = \widetilde { \cal O } ( d / \alpha ^ { 2 } )$ repairs. Each repair is described by a triple $( t , i , s ) \in [ T ] \times [ K ] \times \{ - 1 , 1 \}$ , so considering all repair histories of length at most M gives $N \leq ( 2 K T + 1 ) ^ { M }$ and therefore log $N = \widetilde { O } ( d / \alpha ^ { 2 } )$ . One of these experts exactly follows the oracle and predicts every displayed action within $\alpha w _ { t , i }$

The remaining dificulty is that EXP4-IX does not necessarily play the action recommended by the good expert. A direct comparison would therefore leave an uncertainty term for an action that may never be sampled. We use a discounted loss to move this uncertainty to the action actually played. For Gaussian rewards, if action i is played, then $Y _ { t } \sim \mathcal { N } ( \mu _ { t , i } , 1 )$ , and hence $\operatorname* { P r } ( Y _ { t } \leq 0 \mid A _ { t } = i ) = \Phi ( - \mu _ { t , i } )$ , where Φ denote the cdf of $N ( 0 , 1 )$ . We therefore use the bounded loss $\ell _ { t } ( i ) : = \Phi ( - \mu _ { t , i } )$ , which admits the unbiased bounded observation ${ \bf 1 } _ { \{ Y _ { t } \leq 0 \} }$ when action i is played. Let $\beta _ { t , i } = \operatorname* { m i n } \{ 1 , \alpha w _ { t , i } \}$ and define $\begin{array} { r } { \bar { \ell } _ { t } ( i ) = \frac { \ell _ { t } ( i ) + 1 - \beta _ { t , i } } { 2 } } \end{array}$ . Let $a _ { t } ^ { \star }$ be the action recommended by the good expert and let $i _ { t } ^ { \star }$ be an optimal action. Since the good expert is accurate and optimistic, its reward gap is at most $2 \alpha w _ { t , a _ { t } ^ { \star } }$ , which implies $\ell _ { t } ( a _ { t } ^ { \star } ) - \ell _ { t } ( i _ { t } ^ { \star } ) \leq \beta _ { t , a _ { t } ^ { \star } }$ . For the action $A _ { t }$ actually played by the learner, a direct rearrangement gives $\ell _ { t } ( A _ { t } ) - \ell _ { t } ( i _ { t } ^ { \star } ) \leq 2 \left( \bar { \ell } _ { t } ( A _ { t } ) - \bar { \ell } _ { t } ( a _ { t } ^ { \star } ) \right) + \beta _ { t , A _ { t } }$ . The first term is controlled by EXP4-IX, while the second depends only on the uncertainty of the action actually played and can therefore be summed using the standard elliptical-potential bound.

Since log $N = \widetilde { O } ( d / \alpha ^ { 2 } )$ , the master cost is $\widetilde { \cal O } ( \sqrt { d K T } / \alpha )$ , while the elliptical-potential bound gives $\begin{array} { r } { \sum _ { t } \beta _ { t , A _ { t } } = \tilde { O } ( \alpha \sqrt { d T } ) } \end{array}$ . Hence $\mathbb { E } [ R _ { T } ] = \widetilde { O } ( \sqrt { d T } [ \sqrt { K } / \alpha + \alpha ] )$ , and choosing $\alpha \asymp K ^ { 1 / 4 }$ gives $\mathbb { E } [ R _ { T } ] = \widetilde { O } ( K ^ { 1 / 4 } \sqrt { d T } )$

For large menus, we keep the repair construction and the same transfer of uncertainty to played actions, but replace EXP4-IX by a geometric master with a linear surrogate reward. Geometric exploration replaces the master’s explicit K dependence by min $\{ K , d + 1 \}$ }, while the expert-family size still depends logarithmically on $K$

## 4 Lower Bounds

Let $\mathcal { X } _ { K } : = ( B _ { 2 } ^ { d } ) ^ { K }$ be the set of all menus containing K unit-ball actions. Let Q be the class of all randomized nonanticipating menu mechanisms $Q = ( Q _ { t } ) _ { t = 1 } ^ { T }$ , where $Q _ { t } ( \cdot | \theta , H _ { t - 1 } )$ is a probability distribution over $\mathcal { X } _ { K }$ . Thus the current menu may depend on the fixed parameter θ, the past history, and the environment’s internal randomness, but not on the learner’s current action or the current reward noise. For any policy π and a menu mechanism $Q ,$ , define the minimax optimal regret as follows.

$$
{ \mathfrak { R } } _ { T } ( d , K ) = \operatorname* { i n f } _ { \pi } \operatorname* { s u p } _ { \theta \in B _ { 2 } ^ { d } } \operatorname* { s u p } _ { Q } \mathbb { E } _ { \theta , Q , \pi } R _ { T } .\tag{4.1}
$$

Theorem 1 (Minimax lower bounds). Suppose d, $K \geq 2$ and $T \geq d ^ { 2 }$ . There is a universal constant $c > 0$ such that

$$
\Re _ { T } ( d , K ) \geq c K ^ { 1 / 4 } \sqrt { d T } , \qquad 2 \leq K \leq d ,\tag{4.2}
$$

$$
\Re _ { T } ( d , K ) \geq c { \sqrt { d T } } \operatorname* { m i n } \left\{ { \sqrt { d } } , \left( { \frac { d \log K } { \log ( 2 d ) } } \right) ^ { 1 / 4 } \right\} , \qquad K \geq d .\tag{4.3}
$$

Corollary 2 (Oblivious and adaptive menus). For $K = d$ and $T \geq d ^ { 2 }$ , adaptive menus have minimax regret $\Omega ( d ^ { 3 / 4 } \sqrt { T } )$ , whereas oblivious menus admit $\widetilde { O } ( \sqrt { d T } )$ regret. Thus adaptivity can increase the minimax regret by a factor of $d ^ { 1 / 4 }$ , up to logarithmic factors.

The detailed proof of Theorem 1 is given in Section B, with the supporting lemmas proved in Appendix C. Beyond the separation in Corollary 2, Theorem 1 gives a finer dependence on the menu size. For $K \leq d ,$ the lower bound scales as $\Omega ( K ^ { 1 / 4 } \sqrt { d T } )$ . For $K \geq d ,$ it grows with $( \log K ) ^ { 1 / 4 }$ until reaching the $d \sqrt { T }$ scale. In particular, for polynomially large $K \geq d ,$ the lower bound is $d ^ { 3 / 4 } { \sqrt { T } }$ up to logarithmic factors.

## 5 Algorithms

Both algorithms use a shared ridge regression together with a finite family of repair sequences. Each repair sequence modifies the shared estimate through its own ofset. The two algorithms difer in the master used to combine the expert recommendations and the feedback used to update their weights. We first present the small-menu algorithm for $K \leq d ,$ and then replace its master with a geometric one for large menus.

## 5.1 The Algorithm for Small Menus

This subsection presents Repair-IX, whose pseudo-code is given in Algorithm 1. Both algorithms enumerate the full repair family and are not polynomial-time. For $\alpha > 0$ , set $M : = \lceil ( 1 + d \log ( 1 +$ $T / d ) + 4 \log T ) / \alpha ^ { 2 } ]$ and define

$$
\mathcal { E } _ { M } : = \left. e = ( ( \tau _ { j } , i _ { j } , s _ { j } ) ) _ { j = 1 } ^ { r } : 0 \leq r \leq M , \ 1 \leq \tau _ { 1 } \leq \cdot \cdot \cdot \leq \tau _ { r } \leq T , \ i _ { j } \in [ K ] , \ s _ { j } \in \{ - 1 , 1 \} \right.\tag{5.1}
$$

Thus each $e \in { \mathcal { E } } _ { M }$ is a repair sequence containing at most M repairs, and $| \mathcal { E } _ { M } | \leq ( 2 K T + 1 ) ^ { M }$ All repair sequences share the ridge statistics $\begin{array} { r } { V _ { t } : = I _ { d } + \sum _ { s < t } x _ { s } x _ { s } ^ { \top } } \end{array}$ and $\begin{array} { r } { b _ { t } : = \sum _ { s < t } x _ { s } Y _ { s } } \end{array}$ . For each displayed action, let $w _ { t , i } : = \sqrt { x _ { t , i } ^ { \top } V _ { t } ^ { - 1 } x _ { t , i } }$ and $\beta _ { t , i } : = \operatorname* { m i n } \{ 1 , \alpha w _ { t , i } \}$ . Each repair sequence $e \in { \mathcal { E } } _ { M }$ maintains an ofset $c _ { e }$ , initialized at zero. This ofset records that expert’s repairs to the shared estimate; it does not use a separate data set. If e contains a repair $( t , i , s )$ , it applies

$$
c _ { e }  c _ { e } + \frac { \alpha s } { w _ { t , i } } x _ { t , i } \quad \mathrm { i f } \ w _ { t , i } > 0 .\tag{5.2}
$$

If $w _ { t , i } = 0$ , the repair leaves $c _ { e }$ unchanged. After applying all repairs specified by e at round t, it forms $v _ { t , e } : = V _ { t } ^ { - 1 } ( b _ { t } + c _ { e } )$ and recommends

$$
a _ { t , e } \in \arg \operatorname* { m a x } _ { i \in [ K ] } \left\{ x _ { t , i } ^ { \top } v _ { t , e } + \alpha w _ { t , i } \right\} .\tag{5.3}
$$

Each repair sequence is treated as an expert with weight $W _ { e } .$ initialized at one. The weights define

$$
p _ { t , i } : = \frac { \sum _ { e \in \mathcal { E } _ { M } } W _ { e } \mathbf { 1 } _ { \{ a _ { t , e } = i \} } } { \sum _ { e \in \mathcal { E } _ { M } } W _ { e } } .\tag{5.4}
$$

After sampling $A _ { t } \sim p _ { t }$ and observing $Y _ { t } ,$ define $Z _ { t } : = ( \mathbf { 1 } _ { \{ Y _ { t } \leq 0 \} } + 1 - \beta _ { t , A _ { t } } ) / 2$ and use

$$
\widehat { \ell } _ { t , e } : = \frac { Z _ { t } { \mathbf { 1 } } _ { \{ a _ { t , e } = A _ { t } \} } } { p _ { t , A _ { t } } + \eta } , \qquad W _ { e } \gets W _ { e } \exp ( - \eta \widehat { \ell } _ { t , e } ) .\tag{5.5}
$$

Algorithm 1 Repair-IX: shared ridge regression with repair sequences   
Parameters: $\alpha > 0$ and $\eta > 0$   
1: Let $\mathcal { E } _ { M }$ be the family of schedules defined in (5.1)   
2: Initialize $V _ { 1 }  I _ { d } , b _ { 1 }  0 ,$ and $c _ { e } \gets 0 , W _ { e } \gets 1$ for every $e \in { \mathcal { E } } _ { M }$   
3: for $t = 1 , \dots , T$ do   
4: Observe $x _ { t , 1 } , \ldots , x _ { t , K }$ and compute $w _ { t , i }$ and $\beta _ { t , i }$ for every $i \in [ K ]$   
5: For every $e \in { \mathcal { E } } _ { M } .$ , apply its repairs at round t using (5.2)   
6: For every $e \in { \mathcal { E } } _ { M }$ , compute its recommendation using (5.3)   
7: Compute $p _ { t }$ using (5.4) and sample $A _ { t } \sim p _ { t }$   
8: Observe $Y _ { t } ,$ , compute $Z _ { t } .$ , and update every $W _ { e }$ using (5.5)   
9: $V _ { t + 1 }  V _ { t } + x _ { t , A _ { t } } x _ { t , A _ { t } } ^ { \top } , b _ { t + 1 }  b _ { t } + x _ { t , A _ { t } } Y _ { t }$   
10: end for

Theorem 3 (Upper bound for small menus). For $2 \leq K \leq d \leq T$ , run Algorithm 1 with $\alpha = [ K \log ( 2 K { \dot { T } } + 1 ) ] ^ { 1 / 4 }$ and $\eta = \sqrt { M \log ( 2 K T + 1 ) / ( K T ) }$ . Then, for a universal constant $C > 0$

$$
\begin{array} { r } { \mathbb { E } [ R _ { T } ] \leq C \left[ K \log ( 2 K T + 1 ) \right] ^ { 1 / 4 } \sqrt { d T \log ( 1 + T / d ) } . } \end{array}\tag{5.6}
$$

The proof of Theorem 3 is in Appendix D. The supporting lemmas are proved in Appendix E. Comparison with previous work: For $2 \leq K \leq d ,$ the best previous bound for adaptive menus is $\widetilde { \cal O } ( \sqrt { d K T } )$ , obtained by SquareCB [7]. Theorem 3 improves this bound by a factor of $K ^ { 1 / 4 }$ Together with Theorem 1, this gives matching upper and lower bounds up to logarithmic factors when $T \geq d ^ { 2 }$

## 5.2 Algorithm for Large Menus

```latex
Algorithm 2 Repair-Geo: repair sequences with a geometric master
Parameters: $\alpha > 0$ and $0 < \eta \leq 1 / ( 1 6 \operatorname* { m i n } \{ K , d + 1 \} )$
1: Let $\mathcal { E } _ { M }$ be defined as in (5.1)
2: Set $\gamma  8$ min $\{ K , d + 1 \} \eta$
3: Initialize $V _ { 1 }  I _ { d } , b _ { 1 }  0 .$ , and $c _ { e } \gets 0 , W _ { e } \gets 1$ for every $e \in { \mathcal { E } } _ { M }$
4: for $t = 1 , \dots , T$ do
5: Observe $x _ { t , 1 } , \ldots , x _ { t , K }$ and compute $w _ { t , i }$ and $\beta _ { t , i }$ for every $i \in [ K ]$
6: For every $e \in { \mathcal { E } } _ { M }$ , apply all repairs of e scheduled at round t using (5.2), and compute $\boldsymbol { a } _ { t , e }$
using (5.3)
7: Form the lifted features $z _ { t , 1 } , \dots , z _ { t , K }$
8: Compute $p _ { t } ^ { 0 }$ and compute $\nu _ { t }$ by Lemma 15
9: Set $p _ { t } \gets ( 1 - \gamma ) p _ { t } ^ { 0 } + \gamma \nu _ { t }$ and $\begin{array} { r } { \Gamma _ { t } \gets \sum _ { i } p _ { t , i } z _ { t , i } z _ { t , i } ^ { \top } } \end{array}$
10: Sample $A _ { t } \sim p _ { t }$
11: Observe $Y _ { t }$ and set $Z _ { t } \gets ( Y _ { t } + 2 \beta _ { t , A _ { t } } - 1 ) / 2$
12: For every $e \in { \mathcal { E } } _ { M }$ , compute $\widehat { g } _ { t , e }$ and $h _ { t , e } ,$ and update $W _ { e } \gets W _ { e } \exp \{ \eta ( \widehat { g } _ { t , e } + 2 \eta h _ { t , e } ) \}$
13: $V _ { t + 1 } \gets V _ { t } + x _ { t , A _ { t } } x _ { t , A _ { t } } ^ { \top }$ and $b _ { t + 1 }  b _ { t } + x _ { t , A _ { t } } Y _ { t }$
14: end for
```

Repair-Geo keeps the repair family $\mathcal { E } _ { M }$ , ridge statistics, repairs, and recommendations $\boldsymbol { a } _ { t , e }$ from Repair-IX. Its master uses the shared linear structure of the rewards, rather than treating the K actions as unrelated arms. The geometric estimator and leverage correction follow the approach of Zimmert and Lattimore [13], adapted here to changing action features and repair experts chosen for comparison after the history is known.

The master. Choose $0 < \eta \leq 1 / ( 1 6$ min $\{ K , d + 1 \} )$ ) and set $\gamma : = 8$ min $\{ K , d + 1 \} \eta$ . The sign-based loss used by Repair-IX is not linear in the action features, so here we use a linear surrogate reward. For action $i ,$ define $z _ { t , i } : = ( x _ { t , i } ^ { \top } , 2 \beta _ { t , i } - 1 ) ^ { \top }$ ; if $A _ { t }$ is played, set $Z _ { t } : = ( Y _ { t } + 2 \beta _ { t , A _ { t } } - 1 ) / 2$ . With $\begin{array} { r } { \vartheta : = \frac { 1 } { 2 } ( \theta ^ { \top } , 1 ) ^ { \top } } \end{array}$ , its conditional mean is

$$
g _ { t } ( i ) : = \mathbb { E } _ { t } [ Z _ { t } \mid A _ { t } = i ] = z _ { t , i } ^ { \top } \vartheta = \frac { \mu _ { t , i } + 2 \beta _ { t , i } - 1 } { 2 } .
$$

The term $2 \beta _ { t , i }$ compensates for the possible reward gap of the good repair expert, so that the remaining uncertainty can be charged to the actions actually played.

The expert weights first define

$$
p _ { t , i } ^ { 0 } : = \frac { \sum _ { e \in { \mathcal { E } } _ { M } } W _ { e } \mathbf { 1 } _ { \{ a _ { t , e } = i \} } } { \sum _ { e \in { \mathcal { E } } _ { M } } W _ { e } } .
$$

This distribution may concentrate on only a few directions of the lifted feature space. We therefore mix it with an approximate exploration design. Let $\nu _ { t } \in \Delta _ { K }$ be the approximate design computed by the finite procedure in Lemma 15, where $\Delta _ { K } : = \{ \nu \in \mathbb { R } _ { + } ^ { K } : \sum _ { i } \nu _ { i } = 1 \}$ . Its largest leverage is at most twice the dimension of the current feature span. This span is nonzero: if $x _ { t , i } = 0$ , then $\beta _ { t , i } = 0$ and the last coordinate of $z _ { t , i }$ is −1. We mix the design with the expert distribution:

$$
p _ { t } : = \underbrace { ( 1 - \gamma ) p _ { t } ^ { 0 } } _ { \mathrm { f o l l o w ~ t h e ~ r e p a i r ~ e x p e r t s } } + \underbrace { \gamma \nu _ { t } } _ { \mathrm { g e o m e t r i c ~ e x p l o r a t i o n } } , \qquad \Gamma _ { t } : = \sum _ { i = 1 } ^ { K } p _ { t , i } z _ { t , i } z _ { t , i } ^ { \top } .\tag{5.7}
$$

The learner samples $A _ { t } \sim p _ { t }$ . The matrix $\Gamma _ { t }$ records how well the resulting sampling distribution covers the diferent directions in the current lifted feature space.

After observing $Z _ { t }$ , Repair-Geo uses the linear structure to evaluate all repair experts from this single observation. Write $\Gamma _ { t } ^ { \dagger }$ for the inverse of $\Gamma _ { t }$ on span $\{ z _ { t , 1 } , \ldots , z _ { t , K } \}$ , extended by zero on its orthogonal complement. For expert e, define $\widehat { g } _ { t , e } : = z _ { t , a _ { t , e } } ^ { \top } \Gamma _ { t } ^ { \dagger } z _ { t , A _ { t } } Z _ { t }$ . For each fixed expert $e ,$ this estimate is conditionally unbiased: $\mathbb { E } _ { t } [ \widehat { g } _ { t , e } ] = g _ { t } ( a _ { t , e } )$ . It uses the single observed reward to score even experts whose recommended actions were not played. The factor $\Gamma _ { t } ^ { \dagger }$ plays the same role as an importance-weighting correction, but uses the geometry shared by all actions rather than treating their sampling probabilities independently.

We also define $h _ { t , e } : = z _ { t , a _ { t , e } } ^ { \top } \Gamma _ { t } ^ { \dag } z _ { t , a _ { t , e } }$ . This leverage score measures how well the direction recommended by expert e is covered by the current sampling distribution. A poorly explored direction has a larger leverage score, which also controls the second moment of its geometric reward estimate. The expert weights are updated by

$$
W _ { e } \gets W _ { e } \exp \eta ( \widehat { g } _ { t , e } + 2 \eta h _ { t , e } ) .\tag{5.8}
$$

The positive correction $2 \eta h _ { t , e }$ controls cumulative underestimation of the repair expert chosen for comparison after the trajectory is observed. We use Repair-Geo when log $K \leq d$ and $M \log ( 2 K T +$ $1 ) \leq T / ( 1 6 \operatorname* { m i n } \{ K , d + 1 \} )$ . Otherwise we use OFUL with the parameters of Lemma 14 and its rule (D.9). The following theorem bounds this combined rule.

Theorem 4 (Upper bound for large menus). For $K \geq d \geq 2$ and $T \geq d _ { i }$ let $\alpha = [ \operatorname* { m i n } \{ K , d +$ $1 \} \log ( 2 K T + 1 ) ] ^ { 1 / 4 }$ . If log $K \leq d$ and M log $( 2 K T + 1 ) \leq \frac { T } { 1 6 \operatorname* { m i n } \{ K , d + 1 \} }$ , run Algorithm 2 with $\begin{array} { r } { \eta = \sqrt { \frac { M \log ( 2 K T + 1 ) } { 1 6 \operatorname* { m i n } \{ K , d + 1 \} T } } } \end{array}$ . Otherwise, run OFUL with $\lambda = 1$ and $\delta = T ^ { - 2 }$ . Then, there exists a universal constant $C > 0$

$$
\begin{array} { r } { \mathbb { E } [ R _ { T } ] \le C \log ( 2 d T ) \sqrt { d T } \operatorname* { m i n } \left\{ \sqrt { d } , ( d \log K ) ^ { 1 / 4 } \right\} . } \end{array}\tag{5.9}
$$

The proof of Theorem 4 is given in Appendix D.4. Comparison with previous work: For $K \geq d ,$ the best previous general bound for adaptive menus is the standard $\widetilde { O } ( d \sqrt { T } )$ confidence-based bound [1]. Theorem 4 improves this to ${ \widetilde O } ( d ^ { 3 / 4 } { \sqrt T } ( \log K ) ^ { 1 / 4 } )$ , corresponding to an improvement by a factor of $( d / \log K ) ^ { 1 / 4 }$ . In particular, for polynomially large $K$ , the improvement is $d ^ { 1 / 4 }$ up to logarithmic factors. Together with the lower bound $\Omega \left( \sqrt { d T } ( d \log K / \log ( 2 d ) ) ^ { 1 / 4 } \right)$ from Theorem 1, our upper bound is optimal up to logarithmic factors in d and $T$

## 6 Conclusion and Open Problems

We studied linear contextual bandits with adaptively chosen action menus and established matching upper and lower bounds on the worst-case regret up to logarithmic factors. For $2 \leq K \leq d$ , the optimal dependence is $K ^ { 1 / 4 } { \sqrt { d T } }$ up to logarithmic factors. For $d \leq K$ , the dependence on the menu size grows only as $( \log K ) ^ { 1 / 4 }$ ; in particular, for polynomially large $K$ , the optimal regret is $d ^ { 3 / 4 } { \sqrt { T } }$ up to logarithmic factors.

The main open problem is computational eficiency. Both Repair-IX and Repair-Geo maintain an exponentially large family of repair sequences. It remains open whether one can obtain the same regret guarantees with a polynomial-time algorithm, while retaining good empirical performance on standard linear-bandit instances.

## References

[1] Yasin Abbasi-Yadkori, Dávid Pál, and Csaba Szepesvári. Improved algorithms for linear stochastic bandits. In Advances in Neural Information Processing Systems, volume 24, pages 2312–2320, 2011.

[2] Alekh Agarwal, Daniel Hsu, Satyen Kale, John Langford, Lihong Li, and Robert E. Schapire. Taming the monster: A fast and simple algorithm for contextual bandits. In Proceedings of the 31st International Conference on Machine Learning, volume 32 of Proceedings of Machine Learning Research, pages 1638–1646. PMLR, 2014. Part 2.

[3] Shipra Agrawal and Navin Goyal. Thompson sampling for contextual bandits with linear payofs. In Proceedings of the 30th International Conference on Machine Learning, volume 28 of Proceedings of Machine Learning Research, pages 127–135. PMLR, 2013. Part 3.

[4] Shipra Agrawal and Navin Goyal. Thompson sampling for contextual bandits with linear payofs. arXiv:1209.3352v4, 2014. Revised version, February 2014.

[5] Peter Auer. Using confidence bounds for exploitation-exploration trade-ofs. Journal of Machine Learning Research, 3:397–422, 2002.

[6] Wei Chu, Lihong Li, Lev Reyzin, and Robert E. Schapire. Contextual bandits with linear payof functions. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, volume 15 of Proceedings of Machine Learning Research, pages 208–214. PMLR, 2011.

[7] Dylan J. Foster and Alexander Rakhlin. Beyond UCB: Optimal and eficient contextual bandits with regression oracles. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 3199–3210. PMLR, 2020.

[8] Yingkai Li, Yining Wang, and Yuan Zhou. Nearly minimax-optimal regret for linearly parameterized bandits. In Proceedings of the Thirty-Second Conference on Learning Theory, volume 99 of Proceedings of Machine Learning Research, pages 2173–2174. PMLR, 2019.

[9] Gergely Neu. Explore no more: Improved high-probability regret bounds for non-stochastic bandits. In Advances in Neural Information Processing Systems, volume 28, pages 3168–3176, 2015.

[10] David Simchi-Levi and Yunzong Xu. Bypassing the monster: A faster and simpler optimal algorithm for contextual bandits under realizability. Mathematics of Operations Research, 47 (3):1904–1931, 2022. doi: 10.1287/moor.2021.1193.

[11] Michael J. Todd and E. Alper Yıldırım. On Khachiyan’s algorithm for the computation of minimum-volume enclosing ellipsoids. Discrete Applied Mathematics, 155(13):1731–1744, 2007. doi: 10.1016/j.dam.2007.02.013.

[12] Tong Zhang. Feel-good thompson sampling for contextual bandits and reinforcement learning. SIAM Journal on Mathematics of Data Science, 4(2):834–857, 2022. doi: 10.1137/21M140924X.

[13] Julian Zimmert and Tor Lattimore. Return of the bias: Almost minimax optimal high probability bounds for adversarial linear bandits. In Proceedings of the Thirty Fifth Conference

on Learning Theory, volume 178 of Proceedings of Machine Learning Research, pages 3285–3312. PMLR, 2022.

## A Related Work

Linear bandits with adaptive features. Early work on linear contextual bandits mainly considered action features that are fixed before the interaction. Chu et al. [6] gave SupLinUCB with regret $O ( \sqrt { d T \log ^ { 3 } ( K T \log T ) } )$ and proved a matching $\Omega ( { \sqrt { d T } } )$ lower bound up to logarithmic factors. For $K \leq 2 ^ { d / 2 }$ , Li et al. [8] sharpened this picture by proving a lower bound of $\Omega ( \sqrt { d T \log T \log K } )$ and a VCL-SupLinUCB upper bound matching it up to iterated logarithmic factors. These results rely on the action features being chosen independently of the past reward noise.

When the features may depend on past observations, general confidence-based methods such as OFUL [1] give $\widetilde { O } ( d \sqrt { T } )$ regret. Linear Thompson sampling also allows adaptive contexts; the revised analysis of Agrawal and Goyal [3, 4] gives $O ( d \log T \sqrt { T \log K } )$ regret. Foster and Rakhlin [7, Section 2.3] highlight the gap between fixed and adaptive features and ask whether the additional dependence on K is necessary. Our results answer this question by giving the minimax dependence on the menu size K up to logarithmic factors.

Reductions to supervised learning. Another line of work reduces contextual bandits to supervised learning. ILOVETOCONBANDITS [2] considers a finite policy class of size N under i.i.d. context–reward pairs and achieves the optimal $O ( \sqrt { K T \log N } )$ regret up to logarithmic factors using a cost-sensitive classification oracle. Under realizability, Simchi-Levi and Xu [10] give an optimal reduction to ofline regression using only ${ \cal O } ( \log T )$ oracle calls, or $O ( \log \log T )$ calls when the horizon is known in advance. Both approaches rely on independently drawn contexts.

SquareCB instead reduces contextual bandits to online regression and allows the contexts to depend on the past [7]. If the online regression oracle has regret $B _ { T }$ , SquareCB gives regret of order $\widetilde { O } ( \sqrt { K T B _ { T } } )$ ; for a d-dimensional linear class this becomes $\widetilde { \cal O } ( \sqrt { d K T } )$ . This makes SquareCB directly applicable to our adaptive-menu setting, but leaves a $\sqrt { K }$ dependence. Our bounds use the linear structure more directly and improve this dependence to $K ^ { 1 / 4 }$ for small menus and to $( \log K ) ^ { 1 / 4 }$ for large menus, up to logarithmic factors.

Combining experts. Our upper bounds also build on existing expert and adversarial-bandit methods. Repair-IX uses the implicit-exploration estimator of Neu [9]; in particular, EXP4-IX achieves $O ( { \sqrt { K T \log N } } )$ regret against N experts up to lower-order and high-probability terms, without explicit uniform exploration. Repair-Geo uses a geometric estimator and leverage correction following the approach of Zimmert and Lattimore [13]. For a fixed finite action set, their exponentialweights method achieves $O ( { \sqrt { d T \log ( K T ) } } )$ ) regret against an adaptive adversary.

## B Proof of the Lower Bounds

We first focus on small K. Our proofs require the following lemmas.

## B.1 Key Technical Lemmas

For the proof we use a slightly easier auxiliary experiment. The unnormalized menu vectors are revealed at the beginning of each block, and its hidden labels are revealed after its last reward. The independent Gaussian seeds used to form those vectors are not revealed. Write H for the resulting augmented history before the next block. Conditioning on $\mathcal { H }$ preserves a Gaussian posterior. This history is used only in the lower-bound analysis; it is not the learner’s visible history $H _ { t - 1 }$ in the original model. Remark 9 explains how to remove the disclosures.

Lemma 5. Let $m , n ,$ k be positive integers and let $\sigma > 0$ , with $k \geq 2 ^ { 4 0 }$ 2 $1 6 m k \leq d ,$ and $k ^ { m } \leq K$ . In the auxiliary experiment, the unnormalized menu vectors are revealed before a block and its hidden labels after the block. Suppose that, at its start, the augmented history H satisfies $\theta \mid \mathcal { H } \sim \mathcal { N } ( h , \Sigma )$ with

$$
0 \prec \Sigma \preceq \sigma ^ { 2 } I _ { d } , \qquad \mathrm { t r } ( \Sigma ^ { - 1 } ) < \frac { 2 d } { \sigma ^ { 2 } } .\tag{B.1}
$$

If $n \sigma ^ { 2 } / ( m \sqrt { k } ) \le 1 / 6 4$ , then there exists an n-round block construction using the same menu of exactly K distinct unit-ball actions throughout the block such that any learner satisfies

$$
\mathbb { E } [ R _ { \mathrm { b l o c k } } \mid \mathcal { H } ] \ge \frac { n \sigma \sqrt { m } k ^ { 1 / 4 } } { 2 5 6 } .\tag{B.2}
$$

After the n rewards and the delayed hidden labels are revealed, the posterior is again Gaussian, with covariance $\Sigma _ { + } \preceq \Sigma$ satisfying

$$
\mathrm { t r } ( \Sigma _ { + } ^ { - 1 } ) \leq \mathrm { t r } ( \Sigma ^ { - 1 } ) + \frac { 8 m \sqrt { k } } { \sigma ^ { 2 } } + \sum _ { \textit { s i n t h i s b l o c k } } \| x _ { s } \| _ { 2 } ^ { 2 } .\tag{B.3}
$$

For $m = 1$ the block contains one hidden choice among k actions. For general m, it combines m such choices into $k ^ { m }$ product actions. The factor $\sqrt { m }$ in (B.2) comes from this product construction. The proof is in Appendix C.

Lemma 6 (Gaussian truncation). Let $T \geq d ^ { 2 } , \sigma ^ { 2 } = d / ( 4 0 9 6 T )$ , and $\theta \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I _ { d } )$ . Then,

$$
\mathbb { E } [ R _ { T } \mathbf { 1 } _ { \left\{ \| \theta \| _ { 2 } > 1 \right\} } ] \leq \frac { T \sigma } { 1 0 2 4 } .\tag{B.4}
$$

The Gaussian prior may place a small amount of mass outside the unit ball. Lemma 6 shows that the regret contributed by this event is negligible. We may therefore condition the prior on $\{ \| \theta \| _ { 2 } \leq 1 \}$ and lose only the small term in (B.4). The details of the proof are in Appendix C.2.

Lemma 7. $I f k \geq 2 ^ { 4 0 }$ , 16m $k \leq d , k ^ { m } \leq K$ , and $T \geq d ^ { 2 }$ , then

$$
\Re _ { T } ( d , K ) \geq 2 ^ { - 1 6 } { \sqrt { m } } k ^ { 1 / 4 } { \sqrt { d T } } .\tag{B.5}
$$

Lemma 8 (Oblivious baseline; cf. Chu et al. [6]). For d, $K \geq 2$ and $T \geq d _ { \ast }$ there is an oblivious menu construction with K distinct actions per round such that

$$
\Re _ { T } ( d , K ) \geq c _ { 0 } \sqrt { d T }\tag{B.6}
$$

for a universal constant $c _ { 0 } > 0$

## B.2 Proof of Theorem 1

Proof. Small menus. Suppose $2 \leq K \leq d .$ . If $K \geq 2 ^ { 4 5 }$ , take $m = 1$ and $k = \operatorname* { m i n } \{ K , \lfloor d / 1 6 \rfloor \}$ . Then $k \geq K / 3 2 \geq 2 ^ { 4 0 } , 1 6 k \leq d ,$ and $k \leq K$ . Lemma 7 gives

$$
\Re _ { T } ( d , K ) \geq 2 ^ { - 1 6 } k ^ { 1 / 4 } \sqrt { d T } \geq 2 ^ { - 1 8 } K ^ { 1 / 4 } \sqrt { d T } .
$$

If $K < 2 ^ { 4 5 }$ , Lemma 8 gives $\Re _ { T } ( d , K ) \geq c _ { 0 } \sqrt { d T }$ for a universal constant $c _ { 0 } > 0$ . Since $K ^ { 1 / 4 } \leq 2 ^ { 4 5 / 4 }$ this also gives $\Re _ { T } ( d , K ) \geq c K ^ { 1 / 4 } \sqrt { d T }$ for a universal constant $c > 0$

Large menus. Suppose first that $K \geq d \geq 2 ^ { 4 5 }$ and choose

$$
m = \left\lfloor \operatorname* { m i n } \left\{ { \frac { \log K } { \log d } } , { \frac { d } { 2 ^ { 4 5 } } } \right\} \right\rfloor , \qquad k = \left\lfloor { \frac { d } { 1 6 m } } \right\rfloor .\tag{B.7}
$$

Then $m \geq 1 , k \geq 2 ^ { 4 0 }$ , 16m $k \leq d ,$ , and $k ^ { m } \leq d ^ { m } \leq K$ . Also $k \geq d / ( 3 2 m )$ , and m is at least half the minimum in (B.7). Lemma 7 yields

$$
\Re _ { T } ( d , K ) \ge 2 ^ { - 1 6 } \sqrt { d T } \left( \frac { d m } { 3 2 } \right) ^ { 1 / 4 } \ge 2 ^ { - 3 0 } \sqrt { d T } \operatorname* { m i n } \left\{ \sqrt { d } , \left( \frac { d \log K } { \log ( 2 d ) } \right) ^ { 1 / 4 } \right\} .\tag{B.8}
$$

If $2 \leq d < 2 ^ { 4 5 }$ , Lemma 8 gives $\Re _ { T } ( d , K ) \geq c _ { 0 } \sqrt { d T }$ . Since the minimum in (B.8) is at most ${ \sqrt { d } } \leq 2 ^ { 4 5 / 2 }$ , the same lower bound holds with a smaller universal constant $c > 0$

Decreasing c if necessary proves both claims.

Remark 9 (About the auxiliary disclosures). The auxiliary disclosures are used only in the analysis and can be removed from the actual interaction. Given any learner for the original experiment, define a learner for the auxiliary experiment that simply ignores the extra disclosures. This is a valid learner in the auxiliary experiment, so the lower bound still applies to it.

For each fixed $\theta ,$ the adversary in the original model can generate the same auxiliary variables and keep them in its private state. Using these variables together with the observed history, it generates the same menus as in the auxiliary construction. This mechanism uses neither future reward noise nor the learner’s current random choice, and is therefore nonanticipating. Hence the learner sees the same distribution of menus, actions, and rewards in the two experiments and has the same expected regret.

Finally, after conditioning the prior on $\{ \| \theta \| _ { 2 } \leq 1 \}$ , its support lies in $B _ { 2 } ^ { d }$ . Its average regret is therefore no larger than the supremum over fixed $\theta \in B _ { 2 } ^ { d }$ and valid menu mechanisms in (4.1). Thus the lower bound also holds in the original model.

## C Proofs of the Lower Bound Lemmas

We repeatedly use the following standard probability and information-theoretic facts. If $Z \sim { \mathcal { N } } ( 0 , I _ { r } )$ then $\mathbb { E } e ^ { \lambda \| Z \| _ { 2 } ^ { 2 } } = ( 1 - 2 \lambda ) ^ { - r / 2 }$ for $\lambda < 1 / 2$ , and hence

$$
\begin{array} { r } { { \mathbb { P } } \{ \| Z \| _ { 2 } ^ { 2 } \notin [ r / 2 , 2 r ] \} \le 2 e ^ { - r / 1 6 } , \qquad { \mathbb { P } } \{ \| Z \| _ { 2 } ^ { 2 } \ge 4 r \} \le e ^ { - ( 3 / 2 - \log 2 ) r } . } \end{array}\tag{C.1}
$$

For a centered Gaussian random variable of variance at most $v , \mathbb { P } \{ | Z | > u \} \le 2 e ^ { - u ^ { 2 } / ( 2 v ) }$

For two probability measures P and $Q ,$ , write $\mathrm { T V } ( P , Q ) : = \operatorname* { s u p } _ { A } \left| P ( A ) - Q ( A ) \right|$ for their total variation distance, and $\mathrm { K L } ( P | | Q ) : = \mathbb { E } _ { P } [ \log ( d P / d Q ) ]$ for their Kullback–Leibler divergence when $P$ is absolutely continuous with respect to $Q ,$ , and set $\operatorname { K L } ( P \| Q ) = + \infty$ otherwise. Pinsker’s inequality gives $\mathrm { T V } ( P , Q ) \leq { \sqrt { \mathrm { K L } ( P \| Q ) / 2 } }$ . We also use that, for any random variable $W \in [ 0 , n ]$ $| \mathbb { E } _ { P } W - \mathbb { E } _ { Q } W | \le n \mathrm { T V } ( P , Q )$

## C.1 Proof of Lemma 5

We first consider the case $m = 1$ . Fix a block and the history H satisfying (B.1). Set $D = \lfloor d / 2 \rfloor - 1$ $\rho ^ { 2 } = \sqrt { k } / D$ , and $\Delta = \sigma k ^ { 1 / 4 } / 4$ . Then $D \geq 4 k , \rho ^ { 2 } \leq 1 / 2$ , and $\Delta ^ { 2 } = \sigma ^ { 2 } \rho ^ { 2 } D / 1 6$ . The trace condition in (B.1) implies that at least $\lceil d / 2 \rceil$ eigenvalues of $\Sigma$ are at least $\sigma ^ { 2 } / 4$ . Let $\mathcal { E }$ be the span of the corresponding eigenvectors. Since intersecting with $h ^ { \perp }$ loses at most one dimension, we may choose a D-dimensional subspace $\mathcal { U } \subseteq \mathcal { E } \cap h ^ { \bot }$ , where $D = \lfloor d / 2 \rfloor - 1$ . Let $U \in \mathbb { R } ^ { d \times D }$ contain an orthonormal basis of U. Then $\begin{array} { r } { U ^ { \top } h = 0 , \frac { \sigma ^ { 2 } } { 4 } I _ { D } \preceq U ^ { \top } \Sigma U \preceq \sigma ^ { 2 } I _ { D } } \end{array}$ . To make the choice of U well defined and measurable, we fix the following deterministic rule. Order the eigenvalues of $\Sigma ,$ and within each eigenspace apply Gram–Schmidt to the projections of the coordinate vectors $e _ { 1 } , \ldots , e _ { d }$ , skipping zero projections and fixing the sign of each resulting vector by its first nonzero coordinate. After intersecting the selected high-variance subspace with $h ^ { \perp }$ , apply the same rule again to obtain an orthonormal basis U. This makes U a Borel-measurable function of $( h , \Sigma )$ , and hence of the block-start history only; in particular, the construction uses no future information. Set $\boldsymbol { S } : = \boldsymbol { U } ^ { \top } \Sigma \boldsymbol { U }$ Draw J uniformly from [k] and independent $G _ { 1 } , \dots , G _ { k } \sim { \mathcal { N } } ( 0 , I _ { D } )$ . Define

$$
q _ { i } = \left\{ \begin{array} { l l } { \displaystyle \frac { 1 } { 4 \sqrt { D } } \left( \rho S ^ { - 1 / 2 } U ^ { \top } ( \theta - h ) + \sqrt { 1 - \rho ^ { 2 } } G _ { i } \right) , } & { i = J , } \\ { \displaystyle \frac { 1 } { 4 \sqrt { D } } G _ { i } , } & { i \neq J . } \end{array} \right.\tag{C.2}
$$

Let $Q = ( q _ { 1 } , \ldots , q _ { k } )$ and define the actions

$$
x _ { i } : = \frac { U q _ { i } } { \operatorname* { m a x } \{ 1 , \| q _ { i } \| _ { 2 } \} } .\tag{C.3}
$$

The same menu is used in every round of the block.

To keep the posterior Gaussian from one block to the next, we analyze a stronger auxiliary experiment in which the learner receives additional information. The unnormalized menu vectors $Q = ( q _ { 1 } , \ldots , q _ { k } )$ , but not the seeds $G _ { 1 } , \ldots , G _ { k }$ , are revealed at the beginning of the block, while the hidden index J is revealed only after the block ends, so it cannot help the learner identify the good action within the block. These disclosures preserve the Gaussian form of the posterior at block boundaries and only make the learner stronger. Remark 9 explains how to remove them.

Since $S ^ { - 1 / 2 } U ^ { \top } ( \theta - h ) \sim \mathcal { N } ( 0 , I _ { D } )$ , conditional on any fixed $J = j$ , the hidden column $q _ { j }$ has the same distribution $\mathcal { N } ( 0 , I _ { D } / ( 1 6 D ) )$ as every non-hidden column. Moreover, the columns are independent. Hence, for every $j \in [ k ]$

$$
\mathcal { L } ( Q \mid \mathcal { H } , J = j ) = \bigotimes _ { i = 1 } ^ { k } \mathcal { N } ( 0 , I _ { D } / ( 1 6 D ) ) , \qquad \mathbb { P } ( J = j \mid \mathcal { H } , Q ) = \frac { 1 } { k } .\tag{C.4}
$$

In particular, observing Q does not reveal which index is hidden.

To compute the conditional distribution of the parameter, note that, conditional on $J = j$ $4 \sqrt { D } q _ { j } = \rho S ^ { - 1 / 2 } U ^ { \top } ( \theta - h ) + \sqrt { 1 - \rho ^ { 2 } } G _ { j } .$ . Thus $S ^ { - 1 / 2 } U ^ { \top } ( \theta - h )$ and $4 \sqrt { D } q _ { j }$ are jointly Gaussian, with $\mathrm { C o v } ( S ^ { - 1 / 2 } U ^ { \top } ( \theta - h ) ) = I _ { D } , \mathrm { C o v } ( 4 \sqrt { D } q _ { j } ) = I _ { D }$ , and $\mathrm { C o v } ( S ^ { - 1 / 2 } U ^ { \top } ( \theta - h ) , 4 \sqrt { D } q _ { j } ) = \rho I _ { D }$ . The Gaussian conditioning formula therefore gives $S ^ { - 1 / 2 } U ^ { \top } ( \theta - h ) \mid Q , J = j \sim \mathcal { N } ( 4 \rho \sqrt { D } q _ { j } , ( 1 - \rho ^ { 2 } ) I _ { D } )$ Conditioning on the remaining columns of $Q$ does not change this distribution because, given $J = j$ they are independent of $( \theta , q _ { j } )$ . Finally, since $U ^ { \top } h = 0$ , multiplying by $S ^ { 1 / 2 }$ yields

$$
U ^ { \top } \theta \mid Q , J = j \sim { \mathcal { N } } ( \mu _ { j } , ( 1 - \rho ^ { 2 } ) S ) , \qquad { \mathrm { w h e r e ~ } } \mu _ { j } = 4 \rho { \sqrt { D } } S ^ { 1 / 2 } q _ { j } .\tag{C.5}
$$

Conditional on the realized within-block history, the learner’s action-selection probabilities are fixed functions of that history and therefore contribute no additional term to the posterior likelihood. Thus each observed reward contributes $x _ { s } x _ { s } ^ { \top }$ to the posterior precision. Together with the information revealed by the menu, this gives

$$
\Sigma _ { + } ^ { - 1 } = \Sigma ^ { - 1 } + \frac { \rho ^ { 2 } } { 1 - \rho ^ { 2 } } { U S ^ { - 1 } U } ^ { \top } + \sum _ { s = 1 } ^ { n } x _ { s } x _ { s } ^ { \top } .\tag{C.6}
$$

Taking traces in (C.6) and using $S \succeq ( \sigma ^ { 2 } / 4 ) I _ { D }$ and $\rho ^ { 2 } D = { \sqrt { k } }$ gives

$$
\mathrm { t r } ( \Sigma _ { + } ^ { - 1 } ) - \mathrm { t r } ( \Sigma ^ { - 1 } ) \leq \frac { 8 \sqrt { k } } { \sigma ^ { 2 } } + \sum _ { s = 1 } ^ { n } \| x _ { s } \| _ { 2 } ^ { 2 } .\tag{C.7}
$$

We next show that the random menu is regular, in the sense that it satisfies several useful geometric properties that simplify the later gap and information calculations, with constant probability. Define

$$
\mathsf { G } : = \left\{ \frac { 1 } { 3 2 } \leq \| q _ { i } \| _ { 2 } ^ { 2 } \leq \frac { 1 } { 8 } \ \forall i \in [ k ] , \ \| Q \| _ { \mathrm { o p } } \leq \frac { 3 } { 4 } , \ | q _ { i } ^ { \top } S ^ { 1 / 2 } q _ { j } | \leq \frac { \sigma } { 2 5 6 } \ \forall i \neq j \right\} .\tag{C.8}
$$

We now verify that G occurs with constant probability. By (C.4), conditional on $\mathcal { H }$ , the columns $q _ { 1 } , \ldots , q _ { k }$ are independent and each has distribution $\mathcal { N } ( 0 , I _ { D } / ( 1 6 D ) )$ . We use this common Gaussian law to verify the three conditions defining G. For every i, we have $1 6 D \| q _ { i } \| _ { 2 } ^ { 2 } \sim \chi _ { D } ^ { 2 }$ . Therefore, by (C.1), $\mathbb { P } \left( \lVert q _ { i } \rVert _ { 2 } ^ { 2 } \notin [ 1 / 3 2 , 1 / 8 ] \ | \ \mathcal { H } \right) \le 2 e ^ { - D / 1 6 }$ . A union bound over $i \in [ k ]$ gives a total contribution at most $2 k e ^ { - D / 1 6 }$

Next consider $\| Q \| _ { \mathrm { o p } } .$ . Let N be a 1/4-net of the unit sphere in $\mathbb { R } ^ { k }$ with $| { \mathcal { N } } | \leq 9 ^ { k }$ . The standard net bound gives $\begin{array} { r } { \| Q \| _ { \mathrm { o p } } \leq ( 4 / 3 ) \operatorname* { m a x } _ { v \in \mathcal { N } } \| Q v \| _ { 2 } } \end{array}$ . Hence $\| Q \| _ { \mathrm { o p } } > 3 / 4$ implies $\| Q v \| _ { 2 } > 9 / 1 6$ for some $v \in \mathcal N$ . For every fixed unit vector $v , Q v \sim \mathcal { N } ( 0 , I _ { D } / ( 1 6 D ) )$ , so $4 \sqrt { D } Q v \sim \mathcal { N } ( 0 , I _ { D } )$ . Therefore $\begin{array} { r } { \mathbb { P } \left( \| Q v \| _ { 2 } > \frac { 9 } { 1 6 } \mid \mathcal { H } \right) \le e ^ { - ( 3 / 2 } } \end{array}$ <sup>2−log</sup> <sup>2)D</sup>, where we used the second inequality in (C.1). A union bound over N gives $\begin{array} { r } { \mathbb { P } \left( \| Q \| _ { \mathrm { o p } } > \frac { 3 } { 4 } \mid \mathcal { H } \right) \le 9 ^ { k } e ^ { - ( 3 / 2 - \log { 2 } ) D } \le e ^ { - D / 1 0 } } \end{array}$ , where the last inequality follows from $D \geq 4 k$

It remains to control the cross terms $q _ { i } ^ { \intercal } S ^ { 1 / 2 } q _ { j }$ for $i \neq j$ . Fix $i \neq j$ . The vectors $q _ { i }$ and $q _ { j }$ are independent. Conditional on $q _ { i } , q _ { i } ^ { \top } S ^ { 1 / 2 } q _ { j }$ is therefore a centered Gaussian with variance $q _ { i } ^ { \top } S q _ { i } / ( 1 6 D )$ . On the event $\| q _ { i } \| _ { 2 } ^ { 2 } \leq 1 / 8$ , since $S \preceq \sigma ^ { 2 } I _ { D }$ , this variance is at most $\sigma ^ { 2 } / ( 1 2 8 D )$ . The scalar Gaussian tail bound therefore gives $\begin{array} { r } { \mathbb { P } \left( | q _ { i } ^ { \top } S ^ { 1 / 2 } q _ { j } | > \frac { \sigma } { 2 5 6 } , ~ \| q _ { i } \| _ { 2 } ^ { 2 } \leq \frac { 1 } { 8 } ~ | ~ \mathcal { H } \right) \leq 2 e ^ { - D / 1 0 2 4 } } \end{array}$ . Taking a union bound over all ordered pairs $i \neq j$ gives a contribution at most $2 k ^ { 2 } e ^ { - D / 1 0 2 4 }$

Combining the three bounds,

$$
\mathbb { P } ( \mathsf { G } ^ { c } \mid \mathcal { H } ) \le 2 k e ^ { - D / 1 6 } + e ^ { - D / 1 0 } + 2 k ^ { 2 } e ^ { - D / 1 0 2 4 } \le \frac { 1 } { 4 } .\tag{C.9}
$$

The last inequality uses $D \geq 4 k$ and $k \geq 2 ^ { 4 0 }$ . On $\mathsf { G } , \parallel q _ { i } \parallel _ { 2 } < 1$ for every i. Hence the normalization in $x _ { i } = U q _ { i } / \operatorname* { m a x } \{ 1 , \| q _ { i } \| _ { 2 } \}$ is inactive, so $x _ { i } = U q _ { i }$ and therefore $U ^ { \top } x _ { i } = q _ { i }$

One reward reveals little about the hidden index. Fix $Q \in { \mathsf { G } }$ . For each $j \in [ k ]$ , let $P _ { j }$ denote the law of the within-block interaction conditional on $J = j . \mathrm { \ B y \ ( C . 5 ) }$ , under $P _ { j }$ we have $U ^ { \top } \theta = \mu _ { j } + \zeta$ , where $\zeta \sim \mathcal { N } ( 0 , ( 1 - \rho ^ { 2 } ) S )$ is drawn once and remains fixed throughout the block. To measure how much the rewards reveal about the hidden index, define a reference law $P _ { 0 }$ using the same menu, the same learner policy, and the same residual ζ, but removing the planted shift $\mu _ { j } { \mathrm { ; } }$ that is, under $P _ { 0 }$ we set $U ^ { \top } \theta = \zeta$ . Thus $P _ { j }$ and $P _ { 0 }$ difer only through the shift $\mu _ { j }$ . Divergences below concern the marginal interaction laws. Probabilities of residual events refer to the same experiments with their latent residuals retained.

Conditional on $\zeta$ and the past, when the learner selects action $x _ { A _ { s } }$ , the reward distributions under $P _ { j }$ and $P _ { 0 }$ difer only in their means, by $( U ^ { \top } x _ { A _ { s } } ) ^ { \top } \mu _ { j }$ . Applying the Gaussian KL, the KL chain rule, and then data processing after removing ζ gives

$$
\mathrm { K L } ( P _ { 0 } \| P _ { j } ) \leq \frac { 1 } { 2 } \mathbb { E } _ { 0 } \sum _ { s = 1 } ^ { n } \left( ( \boldsymbol { U } ^ { \top } \boldsymbol { x } _ { A _ { s } } ) ^ { \top } \boldsymbol { \mu } _ { j } \right) ^ { 2 } .\tag{C.10}
$$

The key geometric fact is that one selected action cannot be strongly aligned with all k possible hidden shifts. On G, we have $U ^ { \top } x _ { i } = q _ { i }$ and $\mu _ { j } = 4 \rho \sqrt { D } S ^ { 1 / 2 } q _ { j }$ . Therefore

$$
\begin{array} { r l } {  { \sum _ { j = 1 } ^ { k } ( ( U ^ { \top } x _ { i } ) ^ { \top } \mu _ { j } ) ^ { 2 } = 1 6 \rho ^ { 2 } D \| Q ^ { \top } S ^ { 1 / 2 } q _ { i } \| _ { 2 } ^ { 2 } } } \\ & { \leq 1 6 \rho ^ { 2 } D \| Q \| _ { \mathrm { o p } } ^ { 2 } \| S ^ { 1 / 2 } \| _ { \mathrm { o p } } ^ { 2 } \| q _ { i } \| _ { 2 } ^ { 2 } } \\ & { \leq \frac { 9 } { 8 } \sigma ^ { 2 } \rho ^ { 2 } D } \\ & { = 1 8 \Delta ^ { 2 } . } \end{array}\tag{C.11}
$$

where the last equality uses $\Delta ^ { 2 } = \sigma ^ { 2 } \rho ^ { 2 } D / 1 6$ . Averaging (C.10) over the uniform hidden index gives

$$
{ \frac { 1 } { k } } \sum _ { j = 1 } ^ { k } \mathrm { K L } ( P _ { 0 } \| P _ { j } ) \leq { \frac { 9 n \Delta ^ { 2 } } { k } } .\tag{C.12}
$$

Since $\Delta ^ { 2 } = \sigma ^ { 2 } \sqrt { k } / 1 6$ , the assumption $n \sigma ^ { 2 } / \sqrt { k } \le 1 / 6 4$ implies $n \Delta ^ { 2 } / k \leq 1 / 1 0 2 4$

Let $N _ { j }$ be the number of pulls of action $j$ in the block. Since $0 \leq N _ { j } \leq n$ , the expectation–TV inequality gives $\mathbb { E } _ { j } [ N _ { j } ] \le \mathbb { E } _ { 0 } [ N _ { j } ] + n \mathrm { ~ T V } ( P _ { 0 } , P _ { j } )$ . Averaging over $j ,$ using $\textstyle \sum _ { j } N _ { j } \leq n$ , Pinsker’s inequality, Cauchy–Schwarz, and (C.12),

$$
\begin{array} { r l } { \displaystyle \frac 1 k \sum _ { j = 1 } ^ { k } \mathbb E _ { j } [ N _ { j } ] \leq \displaystyle \frac { n } { k } + \frac { n } { k } \sum _ { j = 1 } ^ { k } \mathrm { T V } ( P _ { 0 } , P _ { j } ) } & { } \\ { \leq \displaystyle \frac { n } { k } + n \sqrt { \frac { 1 } { 2 k } \sum _ { j = 1 } ^ { k } \mathrm { K L } ( P _ { 0 } \| P _ { j } ) } } & { } \\ { \leq \frac { 5 n } { 8 } . } \end{array}\tag{C.13}
$$

The hidden action has a definite gap. We show that the hidden action has a gap of at least $\Delta / 8 .$ . Recall that $\mu _ { j } = 4 \rho \sqrt { D } S ^ { 1 / 2 } q _ { j }$ . Since $S \succeq ( \sigma ^ { 2 } / 4 ) I _ { D }$ , we have $S ^ { 1 / 2 } \succeq ( \sigma / 2 ) I _ { D }$ . Hence, on $\begin{array} { r } { \mathsf { G } , q _ { j } ^ { \top } \mu _ { j } = 4 \rho \sqrt { D } q _ { j } ^ { \top } S ^ { 1 / 2 } q _ { j } \ge 4 \rho \sqrt { D } \frac { \sigma } { 2 } \| q _ { j } \| _ { 2 } ^ { 2 } \ge \frac { \sigma \rho \sqrt { D } } { 1 6 } = \frac { \Delta } { 4 } } \end{array}$ . For $i \neq j$ , the condition in (C.8) gives $\vert q _ { i } ^ { \top } \mu _ { j } \vert = 4 \rho \sqrt { D } \vert q _ { i } ^ { \top } S ^ { 1 / 2 } q _ { j } \vert \le 4 \rho \sqrt { D } \frac { \sigma } { 2 5 6 } = \frac { \Delta } { 1 6 }$ . Therefore

$$
q _ { j } ^ { \top } \mu _ { j } \geq \frac { \Delta } { 4 } , \qquad | q _ { i } ^ { \top } \mu _ { j } | \leq \frac { \Delta } { 1 6 } \quad ( i \neq j ) .\tag{C.14}
$$

Now consider the residual $\zeta \sim \mathcal { N } ( 0 , ( 1 - \rho ^ { 2 } ) S )$ . For every i, $\begin{array} { r } { \mathrm { V a r } ( q _ { i } ^ { \top } \zeta ) = ( 1 - \rho ^ { 2 } ) q _ { i } ^ { \top } S q _ { i } \leq \sigma ^ { 2 } \| q _ { i } \| _ { 2 } ^ { 2 } \leq \frac { \sigma ^ { 2 } } { 8 } } \end{array}$ Thus, by the Gaussian tail bound and a union bound,

$$
P _ { j } \left( \operatorname* { m a x } _ { i \leq k } | q _ { i } ^ { \top } \zeta | > \frac { \Delta } { 3 2 } \right) \leq 2 k \exp \left( - \frac { \Delta ^ { 2 } } { 2 5 6 \sigma ^ { 2 } } \right) = 2 k \exp \left( - \frac { \sqrt { k } } { 4 0 9 6 } \right) \leq \frac { 1 } { 1 6 } .\tag{C.15}
$$

Hence, with probability at least $1 5 / 1 6 , | q _ { i } ^ { \top } \zeta | \leq \Delta / 3 2$ simultaneously for all i. On this event, $\begin{array} { r } { x _ { j } ^ { \top } \theta = q _ { j } ^ { \top } ( \mu _ { j } + \zeta ) \geq \frac { \Delta } { 4 } - \frac { \Delta } { 3 2 } = \frac { 7 \Delta } { 3 2 } } \end{array}$ , while for every $\begin{array} { r } { i \neq j , x _ { i } ^ { \top } \theta = q _ { i } ^ { \top } ( \mu _ { j } + \zeta ) \leq \frac { \Delta } { 1 6 } + \frac { \Delta } { 3 2 } = \frac { 3 \Delta } { 3 2 } } \end{array}$ Therefore action $j$ is optimal and its gap to every other principal action is at least $\Delta / 8 .$

Let $\begin{array} { r } { \mathsf { E } _ { j } = \{ \operatorname* { m a x } _ { i \leq k } | q _ { i } ^ { \top } \zeta | \leq \Delta / 3 2 \} } \end{array}$ . Since $0 \leq n - N _ { j } \leq n$ , we do not need any independence between $N _ { j }$ and $\mathsf { E } _ { j }$ :

$$
\begin{array} { r l } { \displaystyle \mathbb { E } _ { j } [ R _ { \mathrm { b l o c k } } ] \geq \frac { \Delta } { 8 } \mathbb { E } _ { j } \left[ ( n - N _ { j } ) \mathbf { 1 } _ { \mathsf { E } _ { j } } \right] } & { } \\ { \displaystyle \geq \frac { \Delta } { 8 } \left( n - \mathbb { E } _ { j } N _ { j } - n P _ { j } ( \mathsf { E } _ { j } ^ { c } ) \right) . } \end{array}\tag{C.16}
$$

Averaging (C.16) over $j$ and using (C.13) and $P _ { j } ( \mathsf { E } _ { j } ^ { c } ) \leq 1 / 1 6$ gives

$$
\frac { 1 } { k } \sum _ { j = 1 } ^ { k } \mathbb { E } _ { j } [ R _ { \mathrm { b l o c k } } ] \geq \frac { \Delta } { 8 } \left( n - \frac { 5 n } { 8 } - \frac { n } { 1 6 } \right) = \frac { 5 n \Delta } { 1 2 8 } .\tag{C.17}
$$

The bound (C.17) holds for every $Q \in { \mathsf { G } }$ . Since $\mathbb { P } ( J = j \mid \mathcal { H } , Q ) = 1 / k$ , averaging over the hidden index is exactly the conditional expected regret given H and $Q .$ . Finally, $\mathbb { P } ( \mathsf { G } \mid \mathcal { H } ) \geq 3 / 4$ and regret is nonnegative, so

$$
\mathbb { E } [ R _ { \mathrm { b l o c k } } \mid \mathcal { H } ] \ge \frac { 3 } { 4 } \cdot \frac { 5 n \Delta } { 1 2 8 } \ge \frac { n \Delta } { 6 4 } = \frac { n \sigma k ^ { 1 / 4 } } { 2 5 6 } .\tag{C.18}
$$

Equation (C.18) establishes the one-group regret bound for the k principal actions. The padding needed when $K > k$ is handled in the general construction below, which also applies when $m = 1$

General m. We now combine m independent hidden choices. Set $D = \lfloor d / ( 2 m ) \rfloor - 1 , \rho ^ { 2 } = \sqrt { k } / D$ 2 and $\Delta = \sigma k ^ { 1 / 4 } / 4$ . The condition 16mk $\leq d$ implies $D \geq 4 k$ . Choose covariance-orthogonal frames $U _ { 1 } , \dots , U _ { m } \in \mathbb { R } ^ { d \times D }$ with

$$
U _ { \ell } ^ { \top } U _ { r } = 0 , \qquad U _ { \ell } ^ { \top } \Sigma U _ { r } = 0 \quad ( \ell \neq r ) , \qquad \frac { \sigma ^ { 2 } } { 4 } I _ { D } \preceq U _ { \ell } ^ { \top } \Sigma U _ { \ell } \preceq \sigma ^ { 2 } I _ { D } ,\tag{C.19}
$$

and $U _ { \ell } ^ { \top } h = 0$

Such frames can be constructed as follows. By (B.1), the high-variance eigenspace of $\Sigma$ has dimension at least $\lceil d / 2 \rceil$ . Using the same measurable eigenbasis selection as above, choose m mutually orthogonal subspaces $\mathcal { V } _ { 1 } , \ldots , \mathcal { V } _ { m }$ , each spanned by $D + 1 = \lfloor d / ( 2 m ) \rfloor _ { - }$ ⌋ of these eigenvectors. This is possible because $m ( D + 1 ) \leq d / 2$ . For each $\ell ,$ the intersection $\mathcal { V } _ { \ell } \cap h ^ { \perp }$ has dimension at least $D ;$ choose $U _ { \ell }$ as an orthonormal basis of a D-dimensional subspace of this intersection. Since the $\nu _ { \ell }$ are spanned by disjoint sets of eigenvectors, the resulting frames satisfy $U _ { \ell } ^ { \top } U _ { r } = 0$ and $U _ { \ell } ^ { \top } \Sigma U _ { r } = 0$ for $\ell \neq r$ , as well as the remaining properties in (C.19).

In group $\ell ,$ draw an independent hidden index $J _ { \ell } \in [ k ]$ and generate $q _ { \ell , 1 } , \ldots , q _ { \ell , k }$ by the one-group channel (C.2), with $U , S , J$ replaced by $U _ { \ell } , U _ { \ell } ^ { \top } \Sigma U _ { \ell } , J _ { \ell }$ . For $i = ( i _ { 1 } , \dots , i _ { m } ) \in [ k ] ^ { m }$ , define

$$
x _ { i } = \frac { 1 } { \sqrt { m } } \sum _ { \ell = 1 } ^ { m } U _ { \ell } \bar { q } _ { \ell , i _ { \ell } } , \qquad \bar { q } _ { \ell , i } = \frac { q _ { \ell , i } } { \operatorname* { m a x } \{ 1 , \| q _ { \ell , i } \| _ { 2 } \} } .\tag{C.20}
$$

The orthogonality of the frames and $\| \bar { q } _ { \ell , i } \| _ { 2 } \leq 1$ in (C.20) give $\begin{array} { r } { \| \boldsymbol { x } _ { i } \| _ { 2 } ^ { 2 } = m ^ { - 1 } \sum _ { \ell } \| \bar { q } _ { \ell , i _ { \ell } } \| _ { 2 } ^ { 2 } \leq 1 } \end{array}$ These $k ^ { m }$ principal actions are distinct almost surely. If $K > k ^ { m }$ , add distinct points on segments between principal actions. One deterministic rule is to enumerate rational coeficients on the segment joining the first two principal actions and skip points already in the menu. For a principal action, set $w _ { \ell , j } ( x _ { i } ) : = \mathbf { 1 } _ { \{ i _ { \ell } = j \} }$ . For a padded action $a = \lambda x _ { i } + ( 1 - \lambda ) x _ { i ^ { \prime } }$ , use the same interpolation for $w _ { \ell , j } ( a )$ Then $0 \ \leq \ w _ { \ell , j } ( a ) \leq 1$ and $\textstyle \sum _ { j } w _ { \ell , j } ( a ) = 1$ . Write $\begin{array} { r } { v _ { \ell } ( a ) : = \sum _ { j } w _ { \ell , j } ( a ) \bar { q } _ { \ell , j } } \end{array}$ , so that $a = m ^ { - 1 / 2 } \Sigma _ { \ell } U _ { \ell } v _ { \ell } ( a )$ for every action in the menu.

Let $Q _ { \ell } = ( q _ { \ell , 1 } , \dots , q _ { \ell , k } )$ and $S _ { \ell } : = U _ { \ell } ^ { \top } \Sigma U _ { \ell }$ . The standardized residuals in the m subspaces are jointly Gaussian and uncorrelated, hence independent. The one-group calculation therefore shows that the hidden labels remain independent and uniform after all the raw menus $Q _ { 1 } , \ldots , Q _ { m }$ are observed. Conditional on those menus and labels, the projections have the form $U _ { \ell } ^ { \top } \theta = \mu _ { \ell , J _ { \ell } } + \zeta _ { \ell } ,$ where $\mu _ { \ell , j } : = 4 \rho \sqrt { D } S _ { \ell } ^ { 1 / 2 } q _ { \ell , j }$ and the residuals $\zeta _ { \ell } \sim \mathcal { N } ( 0 , ( 1 - \rho ^ { 2 } ) S _ { \ell } )$ are independent across groups. The full posterior is Gaussian, and after the rewards its precision is

$$
\Sigma _ { + } ^ { - 1 } = \Sigma ^ { - 1 } + \frac { \rho ^ { 2 } } { 1 - \rho ^ { 2 } } \sum _ { \ell = 1 } ^ { m } U _ { \ell } S _ { \ell } ^ { - 1 } U _ { \ell } ^ { \top } + \sum _ { s = 1 } ^ { n } x _ { A _ { s } } x _ { A _ { s } } ^ { \top } .\tag{C.21}
$$

Taking traces in (C.21), as in (C.7), gives

$$
\mathrm { t r } ( \Sigma _ { + } ^ { - 1 } ) \leq \mathrm { t r } ( \Sigma ^ { - 1 } ) + { \frac { 8 m { \sqrt { k } } } { \sigma ^ { 2 } } } + \sum _ { s = 1 } ^ { n } \| x _ { A _ { s } } \| _ { 2 } ^ { 2 } ,\tag{C.22}
$$

which proves (B.3).

Testing one group. Fix ℓ. Let $\mathsf { G } _ { \ell }$ be the event in (C.8) with $( Q , S )$ replaced by $( Q _ { \ell } , S _ { \ell } )$ . Then $\mathbb { P } ( \mathsf { G } _ { \ell } \mid \mathcal { H } ) \ge 3 / 4 \mathrm { b y } \ ( \mathrm { C . 9 } )$ . Condition on all raw menus with $Q _ { \ell } \in \mathsf { G } _ { \ell }$ and on the other labels $( J _ { r } ) _ { r \neq \ell } . ^ { 1 }$ Under this conditioning, $J _ { \ell }$ is still uniform. Let $P _ { j }$ be the law of the within-block interaction when $J _ { \ell } = j$ . Let $P _ { 0 }$ use the same policy, the same residual vector $\left( \zeta _ { 1 } , \ldots , \zeta _ { m } \right)$ , and the same mean shifts in all other groups, but replace $U _ { \ell } ^ { \top } \theta$ by $\zeta _ { \ell }$ . These are conditional laws used for analysis; the learner is not given $J _ { \ell }$ during the block.

For action a, the diference between the two reward means is $m ^ { - 1 / 2 } v _ { \ell } ( a ) ^ { \top } \mu _ { \ell , j }$ . On $\mathsf { G } _ { \ell } ,$ normalization in group ℓ is inactive. The bound (C.11), followed by convexity for padded actions, gives $\begin{array} { r } { \sum _ { j } ( v _ { \ell } ( a ) ^ { \top } \mu _ { \ell , j } ) ^ { 2 } \le 1 8 \Delta ^ { 2 } } \end{array}$ for every available action a. Applying the Gaussian KL chain rule conditional on the common residual vector, and then removing that vector by data processing, yields

$$
{ \frac { 1 } { k } } \sum _ { j = 1 } ^ { k } \operatorname { K L } ( P _ { 0 } \| P _ { j } ) \leq { \frac { 1 } { 2 m k } } \mathbb { E } _ { 0 } \sum _ { s = 1 } ^ { n } \sum _ { j = 1 } ^ { k } { \big ( } v _ { \ell } ( x _ { A _ { s } } ) ^ { \top } \mu _ { \ell , j } { \big ) } ^ { 2 } \leq { \frac { 9 n \Delta ^ { 2 } } { m k } } .\tag{C.23}
$$

Since $n \sigma ^ { 2 } / ( m \sqrt { k } ) \le 1 / 6 4$ and $\Delta ^ { 2 } = \sigma ^ { 2 } \sqrt { k } / 1 6$ , we have $n \Delta ^ { 2 } / ( m k ) \leq 1 / 1 0 2 4$ . The random counts $\textstyle \sum _ { s } w _ { \ell , j } ( x _ { A _ { s } } )$ lie in $[ 0 , n ]$ and sum to n over $j$ . The expectation–TV inequality, together with (C.23), gives

$$
{ \frac { 1 } { k } } \sum _ { j = 1 } ^ { k } \mathbb { E } _ { j } \sum _ { s = 1 } ^ { n } w _ { \ell , j } ( x _ { A _ { s } } ) \leq { \frac { n } { k } } + n { \sqrt { \frac { 1 } { 2 k } } } \sum _ { j = 1 } ^ { k } \mathrm { K L } ( P _ { 0 } \| P _ { j } ) \leq { \frac { 5 n } { 8 } } .\tag{C.24}
$$

Adding the group regrets. For any action $^ { a , }$ define its unscaled regret in group ℓ by $D _ { \ell } ( a ) : =$ $\begin{array} { r } { \operatorname* { m a x } _ { j \in [ k ] } \bar { q } _ { \ell , j } ^ { \top } U _ { \ell } ^ { \top } \theta - v _ { \ell } ( a ) ^ { \top } U _ { \ell } ^ { \top } \theta } \end{array}$ . It is nonnegative, including for padded actions. Maximization over the product menu separates across groups, and padding cannot increase the maximum of a linear function. Consequently,

$$
R _ { \mathrm { b l o c k } } = { \frac { 1 } { \sqrt { m } } } \sum _ { \ell = 1 } ^ { m } \sum _ { s = 1 } ^ { n } D _ { \ell } ( x _ { A _ { s } } ) .\tag{C.25}
$$

Under the conditioning above, let $\begin{array} { r } { \mathsf { E } _ { \ell , j } : = \{ \operatorname* { m a x } _ { i } | q _ { \ell , i } ^ { \top } \zeta _ { \ell } | \le \Delta / 3 2 \} } \end{array}$ . Applying (C.15) in group ℓ gives $P _ { j } ( \mathsf { E } _ { \ell , j } ^ { c } ) \leq 1 / 1 6$ . On $\mathsf { E } _ { \ell , j }$ , the within-group gap calculation in (C.14) gives $D _ { \ell } ( a ) \geq \Delta ( 1 - w _ { \ell , j } ( a ) ) / 8$ ,J

when $J _ { \ell } = j$ . Since these regrets are nonnegative outside the event, we may subtract the failure probability without assuming independence from the chosen actions. Using (C.24) gives

$$
\begin{array} { r l } { \displaystyle \frac { 1 } { k } \sum _ { j = 1 } ^ { k } \mathbb { E } _ { j } \sum _ { s = 1 } ^ { n } D _ { \ell } ( x _ { A _ { s } } ) \geq \frac { \Delta } { 8 } \left( n - \frac { 1 } { k } \sum _ { j = 1 } ^ { k } \mathbb { E } _ { j } \sum _ { s = 1 } ^ { n } w _ { \ell , j } ( x _ { A _ { s } } ) - \frac { n } { 1 6 } \right) } & { } \\ { \geq \frac { 5 n \Delta } { 1 2 8 } . } \end{array}\tag{C.26}
$$

The bound (C.26) holds for every choice of the other raw menus and labels, provided $Q _ { \ell } \in \mathsf { G } _ { \ell } .$ Averaging over them and using $\mathbb { P } ( \mathsf { G } _ { \ell } \mid \mathcal { H } ) \geq 3 / 4$ gives $\mathbb { E } [ \sum _ { s } D _ { \ell } ( x _ { A _ { s } } ) \mid \mathcal { H } ] \ge n \Delta / 6 4$ . Linearity of expectation in (C.25) now gives

$$
\mathbb { E } [ R _ { \mathrm { b l o c k } } \mid \mathcal { H } ] \ge \frac { 1 } { \sqrt { m } } \sum _ { \ell = 1 } ^ { m } \frac { n \Delta } { 6 4 } = \frac { n \sigma \sqrt { m } k ^ { 1 / 4 } } { 2 5 6 } .\tag{C.27}
$$

Equations (C.22) and (C.27) prove Lemma 5.

## C.2 Proof of Lemma 6

Proof of Lemma 6. Since all actions lie in the unit ball, $0 \leq R _ { T } \leq 2 T \| \theta \| _ { 2 }$ . Write $\theta \ : = \ : \sigma Z$ with $Z \sim \mathcal { N } ( 0 , I _ { d } )$ . By Cauchy–Schwarz and the Gaussian tail bound, $\mathbb { E } [ \| Z \| _ { 2 } { \mathbf { 1 } } _ { \{ \| Z \| _ { 2 } > 1 / \sigma \} } ] \ \le$ $\sqrt { d } 2 ^ { d / 4 } e ^ { - 1 / ( 8 \sigma ^ { 2 } ) } < 2 ^ { - 1 1 }$ , where the last inequality uses $T \geq d ^ { 2 }$ and $\sigma ^ { 2 } = d / ( 4 0 9 6 T )$ . Therefore $\mathbb { E } [ R _ { T } \mathbf { 1 } _ { \left\{ \| \theta \| _ { 2 } > 1 \right\} } ] \le 2 T \sigma 2 ^ { - 1 1 } = T \sigma / 1 0 2 4$ □

## C.3 Proof of Lemma 7

Proof of Lemma 7. Set $\sigma ^ { 2 } = d / ( 4 0 9 6 T ) , B = \lfloor d / ( 3 2 m \sqrt { k } ) \rfloor$ , and $n \ = \ \lfloor T / B \rfloor$ , and draw $\theta \sim$ $\mathcal { N } ( 0 , \sigma ^ { 2 } I _ { d } )$ at the start. This gives $d / ( 6 4 m \sqrt { k } ) \leq B \leq d / ( 3 2 m \sqrt { k } ) , B n \geq T / 2 .$ , and $n \leq 6 4 T m \sqrt { k } / d$ Hence $n \sigma ^ { 2 } / ( m \sqrt { k } ) \leq 1 / 6 4$ , so every block has the length required by Lemma 5. In the last $T - B n$ rounds, repeat the final menu. These rounds add nonnegative regret and no extra menu changes. It remains to check that the posterior uncertainty is never exhausted. Initially tr $( \Sigma ^ { - 1 } ) = d / \sigma ^ { 2 }$ . After at most B blocks, (B.3) and $\| x _ { t } \| _ { 2 } \leq 1$ give

$$
\begin{array} { l } { \displaystyle \mathrm { t r } ( \Sigma ^ { - 1 } ) \leq \frac { d } { \sigma ^ { 2 } } + \frac { 8 B m \sqrt { k } } { \sigma ^ { 2 } } + T } \\ { \leq \left( 1 + \frac { 1 } { 4 } + \frac { 1 } { 4 0 9 6 } \right) \frac { d } { \sigma ^ { 2 } } < \frac { 2 d } { \sigma ^ { 2 } } . } \end{array}\tag{C.28}
$$

By (C.28), the hypothesis of Lemma 5 holds at every block boundary, so the construction can be repeated using the same fixed parameter. Summing (B.2) over the blocks and using $B n \geq T / 2$ 2

$$
\mathbb { E } [ R _ { T } ] \geq \frac { B n \sigma \sqrt { m } k ^ { 1 / 4 } } { 2 5 6 } \geq \frac { T \sigma \sqrt { m } k ^ { 1 / 4 } } { 5 1 2 } .\tag{C.29}
$$

Subtract the tail bound in Lemma 6 from (C.29). The remaining expected regret is nonnegative. Dividing it by $\mathbb { P } ( \| \theta \| _ { 2 } \le 1 ) \le 1$ can only increase it, so

$$
\mathbb { E } [ R _ { T } \ | \ \| \theta \| _ { 2 } \leq 1 ] \geq \frac { T \sigma \sqrt { m } k ^ { 1 / 4 } } { 5 1 2 } - \frac { T \sigma } { 1 0 2 4 } \geq 2 ^ { - 1 6 } \sqrt { m } k ^ { 1 / 4 } \sqrt { d T } .\tag{C.30}
$$

Remark 9 removes the auxiliary disclosures. The conditioned prior in (C.30) is supported in the unit ball, so its expected regret is at most the supremum over fixed parameters and menu mechanisms in (4.1). This proves (B.5).

## D Proof of the Upper Bounds

The proof uses three ingredients: the repair family contains one good repair sequence, the master can compete with this repair expert after the trajectory is known, and the remaining uncertainty can be charged only to the actions actually played. The technical proofs are deferred to Appendix E, except for the standard OFUL guarantee, which is invoked directly from Abbasi-Yadkori et al. [1].

## D.1 Key Technical Lemmas for Small Menus

We first show that the repair family $\mathcal { E } _ { M }$ contains a good repair sequence that remains accurate on every displayed action throughout the horizon. The following lemma formalizes this property.

Lemma 10. With probability at least $1 - T ^ { - 2 }$ , there exists $e ^ { \star } \in { \mathcal { E } } _ { M }$ such that

$$
| x _ { t , i } ^ { \top } ( \theta - v _ { t , e ^ { \star } } ) | \leq \alpha w _ { t , i } \quad f o r \ e v e r y \ t \in [ T ] \ a n d \ i \in [ K ] .\tag{D.1}
$$

The proof of Lemma 10 is given in Appendix E.1. For the small-menu algorithm, recall $\ell _ { t } ( i ) : = \Phi ( - \mu _ { t , i } )$ and $\bar { \ell } _ { t } ( i ) : = ( \ell _ { t } ( i ) + 1 - \beta _ { t , i } ) / 2$ . We use the following bound from the implicitexploration analysis of Neu [9].

Lemma 11. Algorithm 1 satisfies

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \bar { \ell } _ { t } ( A _ { t } ) - \operatorname* { m i n } _ { e \in \mathcal { E } _ { M } } \sum _ { t = 1 } ^ { T } \bar { \ell } _ { t } ( a _ { t , e } ) \right] \leq \frac { 2 \log | \mathcal { E } _ { M } | } { \eta } + \frac { 3 } { 2 } \eta K T .\tag{D.2}
$$

Importantly, the minimum is inside the expectation. Thus the comparator may depend on the realized trajectory, which allows us to use the random good repair sequence $e ^ { \star }$ from Lemma 10. Let $a _ { t } ^ { \star } : = a _ { t , e ^ { \star } }$ denote the action recommended by the good repair expert, and let $i _ { t } ^ { \star }$ denote an optimal action in round t.

Lemma 12. Suppose a repair expert $e ^ { \star }$ satisfies $| x _ { t , i } ^ { \top } ( \theta - v _ { t , e ^ { \star } } ) | \leq \alpha w _ { t , i }$ for every $t \in [ T ]$ and $i \in [ K ]$ and let $a _ { t } ^ { \star } : = a _ { t , e ^ { \star } }$ . Then

$$
\bar { \ell } _ { t } ( a _ { t } ^ { \star } ) \leq \frac { \ell _ { t } ( i _ { t } ^ { \star } ) + 1 } { 2 } \quad f o r \ e v e r y \ t \in [ T ] .\tag{D.3}
$$

For every realized action sequence,

$$
\sum _ { t = 1 } ^ { T } \beta _ { t , A _ { t } } \leq \alpha \sqrt { 2 T d \log ( 1 + T / d ) } .\tag{D.4}
$$

For Repair-Geo, the same repair expert also satisfies

$$
g _ { t } ( a _ { t } ^ { \star } ) \geq \frac { \mu _ { t , i _ { t } ^ { \star } } - 1 } { 2 } \quad f o r e v e r y t \in [ T ] .\tag{D.5}
$$

Indeed, (D.1) and optimism imply $\mu _ { t , i _ { t } ^ { \star } } - \mu _ { t , a _ { t } ^ { \star } } \leq 2 \alpha w _ { t , a _ { t } ^ { \star } }$ . The discounted loss absorbs this uncertainty and gives (D.3). The bound (D.4) follows from the usual elliptical-potential inequality applied only to the actions actually played.

## D.2 Proof of Theorem 3

Proof. For every round, $\ell _ { t } ( A _ { t } ) - \ell _ { t } ( i _ { t } ^ { \star } ) = 2 ( \bar { \ell } _ { t } ( A _ { t } ) - ( \ell _ { t } ( i _ { t } ^ { \star } ) + 1 ) / 2 ) + \beta _ { t , A _ { t } }$ . Summing over time and inserting the best repair expert gives

$$
\begin{array} { l } { \displaystyle \sum _ { t = 1 } ^ { T } ( \ell _ { t } ( A _ { t } ) - \ell _ { t } ( i _ { t } ^ { \star } ) ) = 2 \left[ \sum _ { t = 1 } ^ { T } \bar { \ell } _ { t } ( A _ { t } ) - \operatorname* { m i n } _ { e \in \mathcal { E } _ { M } } \sum _ { t = 1 } ^ { T } \bar { \ell } _ { t } ( a _ { t , e } ) \right] } \\ { \displaystyle \qquad + 2 \left[ \operatorname* { m i n } _ { e \in \mathcal { E } _ { M } } \sum _ { t = 1 } ^ { T } \bar { \ell } _ { t } ( a _ { t , e } ) - \sum _ { t = 1 } ^ { T } \frac { \ell _ { t } ( i _ { t } ^ { \star } ) + 1 } { 2 } \right] + \sum _ { t = 1 } ^ { T } \beta _ { t , A _ { t } } . } \end{array}\tag{D.6}
$$

Lemma 11 controls the first term in (D.6). On the event in Lemma 10, the second term is nonpositive by (D.3). On the failure event it is at most $2 T$ , so its expected contribution is at most $2 / T$ . Finally, (D.4) controls the last term. Therefore

$$
\mathbb { E } \sum _ { t = 1 } ^ { T } ( \ell _ { t } ( A _ { t } ) - \ell _ { t } ( i _ { t } ^ { \star } ) ) \leq \frac { 4 \log | \mathcal { E } _ { M } | } { \eta } + 3 \eta K T + \alpha \sqrt { 2 T d \log ( 1 + T / d ) } + \frac { 2 } { T } .\tag{D.7}
$$

Since $| \mathcal { E } _ { M } | \le ( 2 K T + 1 ) ^ { M }$ , the choices of α and η in Theorem 3 imply $\begin{array} { r } { \frac { 4 \log | \mathcal { E } _ { M } | } { \eta } + 3 \eta K T \ \leq } \end{array}$ $7 \sqrt { K T M \log ( 2 K T + 1 ) }$ . Moreover, all means lie in $[ - 1 , 1 ]$ , so $\ell _ { t } ( A _ { t } ) - \ell _ { t } ( i _ { t } ^ { \star } ) \geq \phi ( 1 ) ( \mu _ { t , i _ { t } ^ { \star } } - \mu _ { t , A _ { t } } ) , \ )$ wher ϕ denote the density of $N ( 0 , 1 )$ . Substituting $\alpha = [ K \log ( 2 K T + 1 ) ] ^ { 1 / 4 }$ and the definition of M into (D.7) yields $\mathbb { E } [ R _ { T } ] \le 1 3 0 \left[ K \log ( 2 K T + 1 ) \right] ^ { 1 / 4 } \sqrt { d T \log ( 1 + T / d ) }$ . The constant calculation is given in Appendix E.5. □

## D.3 Key Technical Lemmas for Large Menus

For large menus, the repair argument is unchanged. We only need a master whose variance depends on the rank of the lifted features rather than directly on $K$

Lemma 13. Suppose the current lifted features $z _ { t , i }$ span a nonzero subspace of dimension at most min $\{ K , d + 1 \}$ . For some fixed vector ϑ, let their mean rewards satisfy $g _ { t } ( i ) = z _ { t , i } ^ { \top } \vartheta \in [ - 1 , 1 ]$ , and suppose the observation has mean $g _ { t } ( i )$ and conditionally 1-subGaussian centered noise after action i is chosen. $J f 0 < \eta \leq 1 / ( 1 6 \operatorname* { m i n } \{ K , d + 1 \} ) , \gamma = 8 \operatorname* { m i n } \{ K , d + 1 \} \eta _ { 1 }$ , and the master uses (5.7) and (5.8) with the designs from Lemma 15, then

$$
\mathbb { E } \left[ \operatorname* { m a x } _ { e \in \mathcal { E } _ { M } } \sum _ { t = 1 } ^ { T } g _ { t } ( a _ { t , e } ) - \sum _ { t = 1 } ^ { T } g _ { t } ( A _ { t } ) \right] \leq \frac { 2 \log | \mathcal { E } _ { M } | } { \eta } + 3 2 \eta \operatorname* { m i n } \{ K , d + 1 \} T .\tag{D.8}
$$

With $\eta = \sqrt { M \log ( 2 K T + 1 ) / ( 1 6 \operatorname* { m i n } \{ K , d + 1 \} T ) }$ and $| \mathcal { E } _ { M } | \le ( 2 K T + 1 ) ^ { M }$ , the right-hand side of (D.8) is at most $1 6 \sqrt { \operatorname* { m i n } \{ K , d + 1 \} T M \log ( 2 K T + 1 ) }$ whenever $M \log ( 2 K T + 1 ) \ \leq$ $T / ( 1 6 \operatorname* { m i n } \{ K , d + 1 \} )$ . The following lemma is the standard OFUL guarantee of Abbasi-Yadkori et al. [1, Theorems 2 and 3], specialized to unit parameter-norm, action-norm, and noise bounds. The expected-regret form below includes the failure-event contribution 2T · $T ^ { - 2 } = 2 / T$

Lemma 14 (OFUL [1]). For d, $K \geq 2$ and $T \geq d$ , run OFUL with regularization $\lambda = 1$ and failure probability $\delta = T ^ { - 2 }$ . Initialize $V _ { 1 } = I _ { d }$ and $b _ { 1 } = 0$ . At round t, set $\widehat { \theta _ { t } } = V _ { t } ^ { - 1 } b _ { t }$ and choose

$$
A _ { t } \in \underset { i \in [ K ] } { \arg \operatorname* { m a x } } \left\{ x _ { t , i } ^ { \top } \widehat { \theta } _ { t } + \left( 1 + \sqrt { \log \operatorname* { d e t } V _ { t } + 4 \log T } \right) w _ { t , i } \right\} .\tag{D.9}
$$

After observing $Y _ { t }$ , update $V _ { t + 1 } = V _ { t } + x _ { t } x _ { t } ^ { \top }$ and $b _ { t + 1 } = b _ { t } + x _ { t } Y _ { t }$ . Under $\| \theta \| _ { 2 } \leq 1 , \| x _ { t , i } \| _ { 2 } \leq 1$ , and conditionally 1-subGaussian centered noise, this algorithm satisfies

$$
\mathbb { E } [ R _ { T } ] \le 4 \sqrt { T d \log ( 1 + T / d ) } \left( 1 + \sqrt { d \log ( 1 + T / d ) + 4 \log T } \right) + \frac { 2 } { T } .\tag{D.10}
$$

In particular, $\mathbb { E } [ R _ { T } ] \le 1 6 d \log ( 1 + T / d ) \sqrt { T }$

The proof of Lemma 13 is given in Appendix E.4. Lemma 14 is quoted from Abbasi-Yadkori et al. [1]; the elementary simplification of its constant is computed in Appendix E.5.

## D.4 Proof of Theorem 4

Proof. Recall $\alpha = [ \operatorname* { m i n } \{ K , d + 1 \} \log ( 2 K T + 1 ) ] ^ { 1 / 4 }$ . We distinguish two cases.

Case 1: log $K \leq d$ and $M \log ( 2 K T + 1 ) \leq T / ( 1 6 \operatorname* { m i n } \{ K , d + 1 \} )$ . Run Algorithm 2 with $\eta =$ $\sqrt { M \log ( 2 K T + 1 ) / ( 1 6 \operatorname* { m i n } \{ K , d + 1 \} T ) }$ . The second inequality implies $\eta \le 1 / ( 1 6 \operatorname* { m i n } \{ K , d + 1 \} )$ The lifted features $z _ { t , 1 } , \dots , z _ { t , K }$ have rank at most $\operatorname* { m i n } \{ K , d + 1 \}$ , and the lifted observation $Z _ { t }$ satisfies the assumptions of Lemma 13.

The reward regret admits the decomposition

$$
\begin{array} { l } { \displaystyle R _ { T } = 2 \left[ \underset { e \in \mathcal { E } _ { M } } { \operatorname* { m a x } } \sum _ { t = 1 } ^ { T } { g _ { t } ( a _ { t , e } ) - \sum _ { t = 1 } ^ { T } { g _ { t } ( A _ { t } ) } } \right] } \\ { \displaystyle \qquad + \left[ \sum _ { t = 1 } ^ { T } ( \mu _ { t , i _ { t } ^ { \star } } - 1 ) - 2 \underset { e \in \mathcal { E } _ { M } } { \operatorname* { m a x } } \sum _ { t = 1 } ^ { T } { g _ { t } ( a _ { t , e } ) } \right] + 2 \sum _ { t = 1 } ^ { T } { \beta _ { t , A _ { t } } } . } \end{array}\tag{D.11}
$$

For the geometric branch, Lemma 13 and $| \mathcal { E } _ { M } | \leq ( 2 K T + 1 ) ^ { M }$ bound the expected first term in (D.11) by $3 2 \sqrt { \operatorname* { m i n } \{ K , d + 1 \} T M \log ( 2 K T + 1 ) }$ . On the event of Lemma 10, the second term is nonpositive by (D.5); its failure event contributes at most $2 / T$ . Finally, (D.4) bounds the last term. Thus

$$
\mathbb { E } [ R _ { T } ] \le 3 2 \sqrt { \operatorname* { m i n } \{ K , d + 1 \} T M \log ( 2 K T + 1 ) } + 2 \alpha \sqrt { 2 T d \log ( 1 + T / d ) } + \frac { 2 } { T } .\tag{D.12}
$$

When $K \geq d .$ we have min $\{ K , d + 1 \} ~ \le ~ 3 d / 2$ . Since log $K \ \leq \ d ,$ substituting the definition of M into (D.12) and using the elementary bounds in Appendix E.5 gives $\mathbb { E } [ R _ { T } ] ~ \leq$ $1 6 7 \log ( 2 d T ) d ^ { 3 / 4 } { \sqrt { T } } ( \log K ) ^ { 1 / 4 }$

Case 2: log $K > d$ or $M \log ( 2 K T + 1 ) > T / ( 1 6 \operatorname* { m i n } \{ K , d + 1 \} )$ ). Run OFUL with $\lambda = 1$ and $\delta = T ^ { - 2 }$ . If log $K > d ,$ the standard OFUL guarantee in Lemma 14 gives $\mathbb { E } [ R _ { T } ] \le 1 6 \log ( 2 d T ) d \sqrt { T }$ If instead log $K \ \leq \ d ,$ the case assumption forces $M \log ( 2 K T + 1 ) \ > \ T / ( 1 6 \operatorname* { m i n } \{ K , d + 1 \} )$ . The deterministic bound $R _ { T } ~ \leq ~ 2 T$ , together with this case condition, gives $\mathbb { E } [ R _ { T } ] ~ \le ~ 2 T ~ <$ $\begin{array} { r } { 8 \sqrt { \operatorname* { m i n } \{ K , d + 1 \} T M \log ( 2 K T + 1 ) } \leq 3 2 \sqrt { \operatorname* { m i n } \{ K , d + 1 \} T M \log ( 2 K T + 1 ) } } \end{array}$ . Thus the right-hand side of (D.12) also bounds this case, without applying Lemma 13 outside its learning-rate range. The same substitution as in Case 1 yields $\mathbb { E } [ R _ { T } ] \le 1 6 7 \log ( 2 d T ) d ^ { 3 / 4 } \sqrt { T } ( \log K ) ^ { 1 / 4 }$ . Combining the two cases gives $\mathbb { E } [ R _ { T } ] \le 1 6 7 \log ( 2 d T ) \sqrt { d T } \operatorname* { m i n } \left\{ \sqrt { d } , ( d \log K ) ^ { 1 / 4 } \right\}$ , which proves (5.9) with $C = 1 6 7$ The constant calculation is given in Appendix E.5. □

## E Proofs of the Upper Bound Lemmas

This appendix proves the repair, master, and surrogate lemmas used in Section D. The parameter and the menu mechanism are fixed throughout. At every reward update, conditional on the current

menu, the learner’s action, the adversary’s private state, and the current repair sequences, the new noise remains independent $\mathcal { N } ( 0 , 1 )$ . In particular, no argument requires the current menu to be independent of past rewards.

## E.1 Proof of Lemma 10

The proof shares several ingredients with the standard OFUL analysis [1], in particular the selfnormalized prediction-error potential and the control of information growth through log det $V _ { t }$ . The main diference is that our predictor is additionally modified by repair updates interleaved with the usual ridge-regression updates, and we must control the total number of such repairs.

Proof of Lemma 10. For the proof, suppose that the true parameter θ is known, and use it only to construct one repair sequence. We allow this auxiliary sequence to grow without imposing the budget $M$ , and then bound its length. Starting from the empty sequence, at round t let $v = V _ { t } ^ { - 1 } ( b _ { t } + c _ { e } )$ be the predictor associated with the repairs constructed so far. While there exists an action $i \in [ K ]$ such that $| x _ { t , i } ^ { \top } ( \theta - v ) | > \alpha w _ { t , i } .$ , set i to the smallest violating index and $s = \mathrm { s i g n } \big ( x _ { t , i } ^ { \top } ( \theta - v ) \big )$ , and append $( t , i , s )$ to the sequence. Such an index has $w _ { t , i } > 0$ , so the update is well defined. Applying this repair changes the predictor to $\begin{array} { r } { \boldsymbol { v } ^ { \prime } = \boldsymbol { v } + \frac { \alpha s } { w _ { t , i } } \boldsymbol { V } _ { t } ^ { - 1 } \boldsymbol { x } _ { t , i } } \end{array}$ . We continue repairing until no displayed action violates the desired bound. Thus, if $v _ { t } ^ { + }$ denotes the predictor after all repairs in round $t ,$ then

$$
| x _ { t , i } ^ { \top } ( \theta - v _ { t } ^ { + } ) | \leq \alpha w _ { t , i } \qquad \mathrm { f o r ~ e v e r y ~ } i \in [ K ] .\tag{E.1}
$$

We first show that each repair makes definite progress. Consider a repair on $x _ { t , i }$ and write $\delta = x _ { t , i } ^ { \top } ( \theta - v )$ . The repair is made only when $\left. \delta \right. > \alpha w _ { t , i }$ , and $s = \mathrm { s i g n } ( \delta )$ . Since $w _ { t , i } ^ { 2 } = x _ { t , i } ^ { \top } V _ { t } ^ { - 1 } x _ { t , i }$

$$
\lVert \theta - v ^ { \prime } \rVert _ { V _ { t } } ^ { 2 } = \lVert \theta - v \rVert _ { V _ { t } } ^ { 2 } - \frac { 2 \alpha } { w _ { t , i } } \lvert \delta \rvert + \alpha ^ { 2 } < \lVert \theta - v \rVert _ { V _ { t } } ^ { 2 } - \alpha ^ { 2 } .\tag{E.2}
$$

By (E.2), every repair decreases the squared V<sub>t</sub>-error by more than $\alpha ^ { 2 }$ . In particular, since $V _ { t }$ is fixed during the repair loop and the squared error is nonnegative, only finitely many repairs can occur in each round. We now bound the total number of repairs over all rounds. Let ${ \boldsymbol v } _ { t } ^ { - }$ and $v _ { t } ^ { + }$ denote the predictors immediately before and after the repairs in round $t ,$ respectively, and let $C _ { t }$ be the total number of repairs through round t, with $C _ { 0 } = 0$ . Define $U _ { t } ^ { - } : = \lVert \theta - v _ { t } ^ { - } \rVert _ { V _ { t } } ^ { 2 } , \qquad U _ { t } ^ { + } : = \lVert \theta - v _ { t } ^ { + } \rVert _ { V _ { t } } ^ { 2 }$ There are $C _ { t } - C _ { t - 1 }$ repairs in round $t ,$ and each decreases the squared error by more than $\alpha ^ { 2 }$ Therefore

$$
U _ { t } ^ { + } \le U _ { t } ^ { - } - \alpha ^ { 2 } ( C _ { t } - C _ { t - 1 } ) ,\tag{E.3}
$$

or equivalently, $U _ { t } ^ { + } + \alpha ^ { 2 } C _ { t } \le U _ { t } ^ { - } + \alpha ^ { 2 } C _ { t - 1 }$ . It remains to control how much the noisy reward update can increase this potential. After the repairs, the learner chooses $x _ { t } = x _ { t , A _ { t } }$ and observes $Y _ { t } = x _ { t } ^ { \top } \theta + \varepsilon _ { t }$

The oracle ofset does not change during the reward update. Since $V _ { t } v _ { t } ^ { + } = b _ { t } + c _ { e }$ , we have $V _ { t + 1 } v _ { t } ^ { + } + x _ { t } ( Y _ { t } - x _ { t } ^ { \top } v _ { t } ^ { + } ) = b _ { t + 1 } + c _ { e }$ . Multiplication by $V _ { t + 1 } ^ { - 1 }$ gives

$$
\begin{array} { r } { v _ { t + 1 } ^ { - } = v _ { t } ^ { + } + V _ { t + 1 } ^ { - 1 } x _ { t } ( Y _ { t } - x _ { t } ^ { \top } v _ { t } ^ { + } ) . } \end{array}\tag{E.4}
$$

Set $u : = x _ { t } ^ { \top } ( \theta - v _ { t } ^ { + } )$ and $r : = x _ { t } ^ { \top } V _ { t } ^ { - 1 } x _ { t }$ . Then $Y _ { t } - x _ { t } ^ { \top } v _ { t } ^ { + } = u + \varepsilon _ { t }$ . The Sherman–Morrison identity gives $V _ { t + 1 } ^ { - 1 } x _ { t } = V _ { t } ^ { - 1 } x _ { t } / ( 1 + r )$ and $x _ { t } ^ { \top } V _ { t + 1 } ^ { - 1 } x _ { t } = r / ( 1 + r )$ . Using these identities in (E.4), with $q : = \theta - v _ { t } ^ { + }$ , yields

$$
\begin{array} { c } { { U _ { t + 1 } ^ { - } = q ^ { \top } V _ { t + 1 } q - 2 ( u + \varepsilon _ { t } ) q ^ { \top } x _ { t } + ( u + \varepsilon _ { t } ) ^ { 2 } \displaystyle \frac { r } { 1 + r } } } \\ { { = U _ { t } ^ { + } + \displaystyle \frac { - u ^ { 2 } - 2 u \varepsilon _ { t } + r \varepsilon _ { t } ^ { 2 } } { 1 + r } . } } \end{array}\tag{E.5}
$$

Conditional on the history and the chosen action, $u , r$ are fixed and $\varepsilon _ { t } \sim \mathcal { N } ( 0 , 1 )$ . Completing the square in the Gaussian density shows that

$$
\mathbb { E } _ { t } \left[ \exp \left\{ \left. \frac { - u ^ { 2 } - 2 u \varepsilon _ { t } + r \varepsilon _ { t } ^ { 2 } } { 2 ( 1 + r ) } \right\} \right| A _ { t } \right] = \frac { 1 } { \sqrt { 2 \pi } } \int _ { \mathbb { R } } \exp \left\{ - \frac { ( z + u ) ^ { 2 } } { 2 ( 1 + r ) } \right\} d z\tag{E.6}
$$

Averaging this conditional identity over $A _ { t }$ gives the next supermartingale inequality.

Define $S _ { 0 } : = \exp ( \| \theta \| _ { 2 } ^ { 2 } / 2 )$ and $S _ { t } : = \exp ( ( U _ { t + 1 } ^ { - } + \alpha ^ { 2 } C _ { t } ) / 2 ) / \mathrm { { / } } \mathrm { e t } V _ { t + 1 }$ for $t \in [ T ]$ . The determinant identity is det $V _ { t + 1 } = \operatorname* { d e t } V _ { t } ( 1 + r )$ . Combining (E.3), (E.5), and (E.6), conditional first on $A _ { t }$ , gives

$$
\mathbb { E } _ { t } [ S _ { t } \mid A _ { t } ] \leq \frac { \exp ( ( U _ { t } ^ { - } + \alpha ^ { 2 } C _ { t - 1 } ) / 2 ) } { \sqrt { \operatorname* { d e t } V _ { t } } } .\tag{E.7}
$$

For $t \geq 2$ , the right-hand side equals $S _ { t - 1 } ;$ for $t = 1$ , it equals $S _ { 0 }$ because $V _ { 1 } = I _ { d } , v _ { 1 } ^ { - } = 0$ , and $C _ { 0 } = 0$ . It does not depend on the sampled action. Averaging (E.7) over that action and then taking conditional expectations before the current menu is chosen proves that $( S _ { t } ) _ { t = 0 } ^ { T }$ is a nonnegative supermartingale. In particular, $\mathbb { E } S _ { T } \le S _ { 0 } \le e ^ { 1 / 2 }$

Since tr $( V _ { T + 1 } ) = d + \textstyle \sum _ { t = 1 } ^ { T } \| x _ { t } \| _ { 2 } ^ { 2 } \leq d + T$ , the arithmetic–geometric mean inequality for its eigenvalues gives det $V _ { T + 1 } \leq ( 1 + T / d ) ^ { d }$ . Together with $U _ { T + 1 } ^ { - } \geq 0$ , this implies $S _ { T } \geq \exp ( \left( \alpha ^ { 2 } C _ { T } - \right.$ $d \log ( 1 + T / d ) ) / 2 )$ . Hence $\alpha ^ { 2 } C _ { T } > 1 + d \log ( 1 + T / d ) + 4$ log T implies $S _ { T } > e ^ { 1 / 2 } T ^ { 2 }$ . Markov’s inequality yields

$$
\mathbb { P } \{ \alpha ^ { 2 } C _ { T } > 1 + d \log ( 1 + T / d ) + 4 \log T \} \le \frac { \mathbb { E } S _ { T } } { e ^ { 1 / 2 } T ^ { 2 } } \le T ^ { - 2 } .\tag{E.8}
$$

By (E.8), with probability at least $1 - T ^ { - 2 }$ we have $C _ { T } \leq ( 1 + d \log ( 1 + T / d ) + 4 \log T ) / \alpha ^ { 2 } \leq M$ On this event, the repairs constructed above, recorded in execution order, form a sequence $e ^ { \star } \in { \mathcal { E } } _ { M }$ By construction, after all repairs in round t its predictor is exactly $v _ { t , e ^ { \star } } = v _ { t } ^ { + }$ . Therefore (E.1) gives $| x _ { t , i } ^ { \top } ( \theta - v _ { t , e ^ { \star } } ) | \leq \alpha w _ { t , i }$ for every $t \in [ T ]$ and $i \in [ K ]$ □

## E.2 Proof of Lemma 11

The proof of this lemma is a direct adaptation of the standard EXP4-IX analysis of Neu [9].

Proof. Condition on the history, current menu, and all repair-expert recommendations before sampling $A _ { t }$ , and denote the corresponding conditional expectation by $\mathbb { E } _ { t }$ . For this proof only, write $q _ { t , e } : = W _ { e } / \sum _ { f \in \mathcal { E } _ { M } } W _ { f }$ and define $\widehat { \ell } _ { t } ( i ) : = Z _ { t } \mathbf { 1 } _ { \{ A _ { t } = i \} } / ( p _ { t , i } + \eta )$ . Then $\widehat { \ell } _ { t , e } = \widehat { \ell } _ { t } ( a _ { t , e } )$ and $\begin{array} { r } { p _ { t , i } = \sum _ { e \in \mathcal { E } _ { M } } q _ { t , e } \mathbf { 1 } _ { \{ a _ { t , e } = i \} } . } \end{array}$ . Using $e ^ { - u } \leq 1 - u + u ^ { 2 } / 2$ for $u \geq 0$ and $\log ( 1 + u ) \leq u$ , the exponentialweights update gives log $\begin{array} { r } { \frac { \sum _ { e } W _ { t + 1 , e } } { \sum _ { e } W _ { t , e } } \leq - \eta \sum _ { e } q _ { t , e } \widehat { \ell } _ { t , e } + \frac { \eta ^ { 2 } } { 2 } \sum _ { e } q _ { t , e } \widehat { \ell } _ { t , e } ^ { 2 } } \end{array}$ . Summing over time and comparing with any fixed repair expert yields

$$
\sum _ { t } \langle p _ { t } , \widehat { \ell } _ { t } \rangle - \sum _ { t } \widehat { \ell } _ { t } ( a _ { t , e } ) \leq \frac { \log | \mathcal { E } _ { M } | } { \eta } + \frac { \eta } { 2 } \sum _ { t } \sum _ { i } p _ { t , i } \widehat { \ell } _ { t } ( i ) ^ { 2 } .\tag{E.9}
$$

Because $\begin{array} { r } { 0 \leq Z _ { t } \leq 1 , \mathbb { E } _ { t } \sum _ { i } p _ { t , i } \widehat \ell _ { t } ( i ) ^ { 2 } = \sum _ { i } \frac { p _ { t , i } ^ { 2 } \mathbb { E } _ { t } [ Z _ { t } ^ { 2 } | A _ { t } = i ] } { ( p _ { t , i } + \eta ) ^ { 2 } } \leq K } \end{array}$ . Moreover, since $\mathbb { E } _ { t } [ Z _ { t } \mid A _ { t } = i ] =$ $\begin{array} { r } { \overline { { \ell } } _ { t } ( i ) , \langle p _ { t } , \overline { { \ell } } _ { t } \rangle - \mathbb { E } _ { t } \langle p _ { t } , \widehat { \ell } _ { t } \rangle = \sum _ { i } \frac { \eta p _ { t , i } \overline { { \ell } } _ { t } ( i ) } { p _ { t , i } + \eta } \leq \eta K } \end{array}$ . It remains to allow the comparator to be selected after the trajectory is known. Fix $e \in { \mathcal { E } } _ { M }$ . Since $0 \leq \eta \widehat { \ell } _ { t } ( a _ { t , e } ) \leq 1 , e ^ { u } \leq 1 + u + u ^ { 2 }$ on [0, 1], and $Z _ { t } ^ { 2 } \leq Z _ { t }$

$$
\mathbb { E } _ { t } e ^ { \eta \widehat { \ell _ { t } } ( a _ { t , e } ) } \leq 1 + \frac { p _ { t , a _ { t , e } } \eta \bar { \ell } _ { t } ( a _ { t , e } ) } { p _ { t , a _ { t , e } } + \eta } + \frac { p _ { t , a _ { t , e } } \eta ^ { 2 } \bar { \ell } _ { t } ( a _ { t , e } ) } { ( p _ { t , a _ { t , e } } + \eta ) ^ { 2 } } \leq e ^ { \eta \bar { \ell } _ { t } ( a _ { t , e } ) } .\tag{E.10}
$$

Iterating (E.10) gives E exp $\left\{ \eta \sum _ { t } ( \widehat { \ell } _ { t } ( a _ { t , e } ) - \bar { \ell } _ { t } ( a _ { t , e } ) ) \right\} \leq 1$ . Applying log-sum-exp and Jensen’s inequality over all $e \in { \mathcal { E } } _ { M }$ yields

$$
\mathbb { E } \operatorname* { m a x } _ { e \in \mathcal { E } _ { M } } \sum _ { t } \bigl ( \widehat { \ell } _ { t } ( a _ { t , e } ) - \bar { \ell } _ { t } ( a _ { t , e } ) \bigr ) \leq \frac { \log | \mathcal { E } _ { M } | } { \eta } .\tag{E.11}
$$

Choosing the repair expert minimizing $\textstyle \sum _ { t } { \bar { \ell } } _ { t } ( a _ { t , e } )$ on the realized trajectory and combining (E.9), (E.11), and the moment and bias bounds gives $\begin{array} { r } { \mathbb { E } \left[ \sum _ { t } \bar { \ell } _ { t } ( A _ { t } ) - \operatorname* { m i n } _ { e \in \mathcal { E } _ { M } } \sum _ { t } \bar { \ell } _ { t } ( a _ { t , e } ) \right] \ \leq \ \frac { 2 \log | \mathcal { E } _ { M } | } { \eta } \ + } \end{array}$ $\overset { 3 } { 2 } \eta K T$ , which proves (D.2). □

## E.3 Proof of Lemma 12

Proof. Work on the event of Lemma 10 and write $a : = a _ { t } ^ { \star }$ . By $( \mathrm { D . 1 } ) , \mu _ { t , i _ { t } ^ { \star } } \leq x _ { t , i _ { t } ^ { \star } } ^ { \top } v _ { t , e ^ { \star } } + \alpha w _ { t , i _ { t } ^ { \star } }$ . Since a maximizes the optimistic score in (5.3), $x _ { t , i _ { t } ^ { \star } } ^ { \top } v _ { t , e ^ { \star } } + \alpha w _ { t , i _ { t } ^ { \star } } \leq x _ { t , a } ^ { \top } v _ { t , e ^ { \star } } + \alpha w _ { t , a }$ . Applying (D.1) once more gives $\mu _ { t , i _ { t } ^ { \star } } - \mu _ { t , a } \leq 2 \alpha w _ { t , a }$ . For Gaussian rewards, $\mathbb { E } \big [ \mathbf { 1 } _ { \{ Y _ { t } \leq 0 \} } \mid H _ { t - 1 } , \mathcal { X } _ { t } , A _ { t } = i \big ] = \Phi ( - \mu _ { t , i } ) =$ $\ell _ { t } ( i )$ . Thus the discounted observation used by Algorithm 1 has conditional mean $\bar { \ell } _ { t } ( i )$ . Since the derivative of Φ is bounded by $\phi ( 0 ) < 1 / 2 , 0 \le \ell _ { t } ( a ) - \ell _ { t } ( i _ { t } ^ { \star } ) \le \phi ( 0 ) \bigl ( \mu _ { t , i _ { t } ^ { \star } } - \mu _ { t , a } \bigr ) \le \alpha w _ { t , a }$ . This diference is also at most one, so it is at most $\beta _ { t , a }$ . Therefore $\begin{array} { r } { \bar { \ell } _ { t } ( a ) = \frac { \ell _ { t } ( a ) + 1 - \beta _ { t , a } } { 2 } \leq \frac { \ell _ { t } ( i _ { t } ^ { \star } ) + 1 } { 2 } } \end{array}$ , which proves (D.3). For the geometric master, the reward gap is bounded by both $2 \alpha w _ { t , a }$ and 2, and hence by $2 \beta _ { t , a }$ . Thus $\begin{array} { r } { g _ { t } ( a ) = \frac { \mu _ { t , a } + 2 \beta _ { t , a } - 1 } { 2 } \geq \frac { \mu _ { t , i _ { t } ^ { \star } } - 1 } { 2 } } \end{array}$ , which proves (D.5). Finally, let $r _ { t } : = w _ { t , A _ { t } } ^ { 2 } = x _ { t } ^ { \top } V _ { t } ^ { - 1 } x _ { t }$ Since $0 \leq r _ { t } \leq 1$ and $r \leq 2 \log ( 1 + r )$ on $\begin{array} { r } { [ 0 , 1 ] , \sum _ { t = 1 } ^ { T } w _ { t , A _ { t } } ^ { 2 } \leq 2 \sum _ { t = 1 } ^ { T } \log ( 1 + r _ { t } ) = 2 \log \operatorname* { d e t } V _ { T + 1 } \leq } \end{array}$ $2 d \log ( 1 + T / d )$ . Since $\beta _ { t , A _ { t } } \le \alpha w _ { t , A _ { t } }$ , Cauchy–Schwarz gives $\begin{array} { r } { \sum _ { t = 1 } ^ { T } \beta _ { t , A _ { t } } \leq \alpha \sqrt { 2 T d \log ( 1 + T / d ) } } \end{array}$ which proves (D.4). □

## E.4 Proof of Lemma 13

We first establish the finite-design property used by the geometric master.

## E.4.1 A design on the current feature span

The design condition below is a constant-factor relaxation of the usual D-optimal design condition. We use a vertex-direction method for the log determinant objective, as in algorithms for enclosing ellipsoids [11]; the fixed-step version needed here has a short proof.

Definition 1 (Approximate exploration design). Let $z _ { 1 } , \dots , z _ { K }$ span a subspace of dimension s $\geq 1$ A distribution $\nu \in \Delta _ { K }$ is an approximate exploration design if $\begin{array} { r } { J : = \sum _ { i } \nu _ { i } z _ { i } z _ { i } ^ { \top } } \end{array}$ is positive definite on this span and max $\dot { \cdot } _ { i } z _ { i } ^ { \top } J ^ { \dagger } z _ { i } \le 2 s$

Lemma 15 (Finite-design leverage bound). For any such vectors, the following procedure returns an approximate exploration design. Start from $\nu _ { i } = 1 / K$ . At each step, form $\begin{array} { r } { J = \sum _ { i } \nu _ { i } z _ { i } z _ { i } ^ { \top } } \end{array}$ and let j be the smallest index maximizing $z _ { j } ^ { \top } J ^ { \dagger } z _ { j }$ . Stop if this maximum is at most 2s; otherwise set $\nu  ( 1 - 1 / ( 2 s ) ) \nu + ( 1 / ( 2 s ) ) e _ { j }$ , where $e _ { j }$ is the $j t h$ coordinate vector in $\mathbb { R } ^ { K }$ . The procedure stops after at most ⌈20s log $K ]$ updates and gives

$$
z _ { i } ^ { \top } J ^ { \dag } z _ { i } \le 2 s \quad f o r \ e v e r y \ i \in [ K ] .\tag{E.12}
$$

Its output is a Borel function of the ordered vectors.

Proof. Work in orthonormal coordinates on the span. The uniform second-moment matrix $J _ { 0 }$ is positive definite, and every update retains this property. For every distribution $\nu , J ( \nu ) \preceq K J _ { 0 }$ , so log det $J ( \nu ) -$ log det $J _ { 0 } \le s$ log $K$ . If an update is made, let $h : = z _ { j } ^ { \top } J ^ { - 1 } z _ { j } > 2 s$ and $\tau : = 1 / ( 2 s )$ The determinant lemma gives

$$
\frac { \operatorname * { d e t } ( ( 1 - \tau ) J + \tau z _ { j } z _ { j } ^ { \top } ) } { \operatorname * { d e t } J } = ( 1 - \tau ) ^ { s - 1 } ( 1 - \tau + \tau h ) > ( 1 - 1 / ( 2 s ) ) ^ { s - 1 } ( 2 - 1 / ( 2 s ) ) .\tag{E.13}
$$

For $s = 1$ , the last term in (E.13) is $3 / 2$ . For $s \geq 2$ , its logarithm is at least $- 1 / 2 + \log ( 7 / 4 ) > 1 / 2 0$ since l $\mathrm { o g } ( 1 - u ) \geq - u / ( 1 - u )$ for $0 < u < 1$ . Thus each update increases the log determinant by more than $1 / 2 0$ . The total increase is at most s log $K$ , so the procedure must stop within the stated number of updates. Its stopping condition is (E.12).

Rank and the Moore–Penrose inverse are Borel functions of a matrix. Choosing the smallest maximizing index and performing each update are also Borel operations. The bounded number of updates therefore makes the returned distribution Borel measurable. □

An orthonormal basis of the span and the reduced coordinates can be formed in $O ( K ( d + 1 ) s )$ arithmetic operations by incremental orthogonalization. In rank-s coordinates, each step evaluates the K quadratic forms and updates the inverse by a rank-one formula, using $O ( K s ^ { 2 } )$ operations. Thus the total cost is $O ( K ( d + 1 ) s + K s ^ { 3 } \log K )$ real-arithmetic operations for an explicitly listed menu.

## E.4.2 Conditional moments and the master comparison

Proof of Lemma 13. Fix a round and suppress the time index. Condition before sampling the current action, so the lifted features, repair-expert recommendations, and weights are fixed. Let their span have dimension $1 \leq s \leq \operatorname* { m i n } \{ K , d + 1 \}$

For this proof only, let $q _ { e } : = W _ { e } / \sum _ { f } W _ { f }$ and $\begin{array} { r } { J : = \sum _ { i } \nu _ { i } z _ { i } z _ { i } ^ { \top } } \end{array}$ . Since $\Gamma \succeq \gamma J$ , Lemma 15 gives $\begin{array} { r } { h _ { e } = z _ { a _ { e } } ^ { \top } \Gamma ^ { \dag } z _ { a _ { e } } \leq \frac { 2 \operatorname* { m i n } \{ K , d + 1 \} } { \gamma } } \end{array}$ . Also, $\begin{array} { r } { \Gamma \succeq ( 1 - \gamma ) \sum _ { i } p _ { i } ^ { 0 } z _ { i } z _ { i } ^ { \top } } \end{array}$ , and therefore

$$
\sum _ { e } q _ { e } h _ { e } = \operatorname { t r } \left( \Gamma ^ { \dagger } \sum _ { i } p _ { i } ^ { 0 } z _ { i } z _ { i } ^ { \top } \right) \leq \frac { s } { 1 - \gamma } \leq 2 \operatorname* { m i n } \{ K , d + 1 \} .\tag{E.14}
$$

For a fixed repair expert, define $a _ { i } : = z _ { a _ { e } } ^ { \top } \Gamma ^ { \dag } z _ { i }$ . The inverse-metric Cauchy–Schwarz inequality and $\begin{array} { r } { \Gamma = \sum _ { i } p _ { i } z _ { i } z _ { i } ^ { \top } } \end{array}$ give

$$
| a _ { i } | \leq \frac { 2 \operatorname* { m i n } \{ K , d + 1 \} } { \gamma } , \qquad \sum _ { i } p _ { i } a _ { i } ^ { 2 } = h _ { e } , \qquad \sum _ { i } p _ { i } a _ { i } g ( i ) = g ( a _ { e } ) .\tag{E.15}
$$

Conditionally on $A = i .$ , write $Z = g ( i ) + \xi _ { i }$ , where $\xi _ { i }$ is centered and conditionally 1-subGaussian. For $2 | \lambda |$ min $\{ K , d + 1 \} / \gamma \leq 1 / 4$ , the geometric estimate $\widehat { g } _ { e } = a _ { A } Z$ satisfies

$$
\mathbb { E } _ { t } e ^ { \lambda \widehat { g } _ { e } } \le 1 + \lambda g ( a _ { e } ) + 2 \lambda ^ { 2 } h _ { e } \le e ^ { \lambda g ( a _ { e } ) + 2 \lambda ^ { 2 } h _ { e } } .\tag{E.16}
$$

Indeed, with $u = \lambda a _ { i } , \mathbb { E } _ { t } [ e ^ { u Z } \mid A = i ] \leq e ^ { u g ( i ) + u ^ { 2 } / 2 }$ . Under the stated range of $\lambda ,$ , the exponent has absolute value at most $9 / 3 2$ , so $e ^ { u g ( i ) + u ^ { 2 } / 2 } \leq 1 + u g ( i ) + 2 u ^ { 2 }$ . Averaging over A and using (E.15) gives (E.16).

Return to the time-indexed notation and define locally $\begin{array} { r } { \widetilde { g } _ { t , e } ~ : = ~ \widehat { g } _ { t , e } + 2 \eta h _ { t , e } } \end{array}$ . Since $\gamma =$ $8 \operatorname* { m i n } \{ K , d + 1 \} \eta$ and $\eta ~ \leq ~ 1 / ( 1 6 \operatorname* { m i n } \{ K , d + 1 \} )$ ), the negative-moment side of (E.16) gives $\mathbb { E } _ { t }$ exp $\{ \eta ( g _ { t } ( a _ { t , e } ) - \widetilde { g } _ { t , e } ) \} \le 1$ . Iterated conditioning and log-sum-exp therefore imply

$$
\mathbb { E } \operatorname* { m a x } _ { e \in { \mathcal { E } _ { M } } } \sum _ { t } \left( g _ { t } ( a _ { t , e } ) - \widetilde { g } _ { t , e } \right) \leq \frac { \log | { \mathcal { E } _ { M } } | } { \eta } .\tag{E.17}
$$

For the weight potential, let $b _ { e } : = 2 \eta ^ { 2 } h _ { t , e }$ for this calculation. The leverage bound gives $0 \leq b _ { e } \leq 1 / 1 6$ , and $| \eta g _ { t } ( a _ { t , e } ) | \leq 1 / 8$ . The positive-moment part of (E.16) and $e ^ { b _ { e } } \leq 1 + 2 b _ { e }$ imply

$$
\begin{array} { r l } & { \mathbb { E } _ { t } e ^ { \eta \widehat { g } _ { t , e } + b _ { e } } \leq e ^ { b _ { e } } \big ( 1 + \eta g _ { t } ( a _ { t , e } ) + b _ { e } \big ) } \\ & { \qquad \leq 1 + \eta g _ { t } ( a _ { t , e } ) + 4 b _ { e } = 1 + \eta g _ { t } ( a _ { t , e } ) + 8 \eta ^ { 2 } h _ { t , e } . } \end{array}\tag{E.18}
$$

Here the second inequality follows by expanding the product with $1 + 2 b _ { e }$ and using the two stated bounds. Apply Jensen’s inequality to the logarithm, average (E.18) with weights $q _ { t , e }$ , and use log $( 1 + x ) \leq x$ and (E.14). This gives

$$
\mathbb { E } _ { t } \log \frac { \sum _ { e } W _ { t + 1 , e } } { \sum _ { e } W _ { t , e } } \leq \eta \sum _ { e } q _ { t , e } g _ { t } ( a _ { t , e } ) + 1 6 \eta ^ { 2 } \operatorname* { m i n } \{ K , d + 1 \} .\tag{E.19}
$$

On the other hand,

$$
\log \frac { \sum _ { e } W _ { T + 1 , e } } { \vert \mathcal { E } _ { M } \vert } \geq \eta \operatorname* { m a x } _ { e \in \mathcal { E } _ { M } } \sum _ { t } \widetilde { g } _ { t , e } - \log \vert \mathcal { E } _ { M } \vert .\tag{E.20}
$$

Combining $\left( \mathrm { E . 1 7 } \right) , \left( \mathrm { E . 1 9 } \right)$ , and (E.20) gives comparison with the repair-expert mixture of at most $\frac { 2 \log | \mathcal { E } _ { M } | } { n } +$ 16η min $\{ K , d + 1 \} T$ . Finally, mixing the repair-expert distribution with $\nu _ { t }$ changes a mean in $[ - 1 , 1 ]$ by at most $2 \gamma$ per round. Since $\gamma = 8 \operatorname* { m i n } \{ K , d + 1 \} \eta$ , the comparison with the learner’s actual actions is at most $\frac { 2 \log | \mathcal { E } _ { M } | } { \eta } + 3 2 \eta$ min $\{ K , d + 1 \} T$ , which proves (D.8). □

## E.5 Elementary Bounds Used in the Theorem Proofs

The calculations below give $C = 1 3 0$ in Theorem 3 and $C = 1 6 7$ in Theorem 4. For $T \geq d \geq$ 2, Bernoulli’s inequality gives $( 1 + T / d ) ^ { d } \geq 1 + T$ . Hence $d \log ( 1 + T / d ) \geq \log ( 1 + T ) > 1$ $d \log ( 1 + T / d ) \geq d \log 2$ , and $1 + d \log ( 1 + T / d ) + 4$ log $T \leq 6 d \log ( 1 + T / d )$ . When $K \leq d \leq T$ $2 K T + 1 \le ( 1 + T ) ^ { 3 }$ , so $\log ( 2 K T + 1 ) \leq 3 d \log ( 1 + T / d )$ . Together with $K \leq d \leq d \log ( 1 + T / d ) / \log 2$ this gives $\begin{array} { r } { K \log ( 2 K T + 1 ) \leq \frac { 3 [ d \log ( 1 + T / d ) ] ^ { 2 } } { \log 2 } } \end{array}$ . Since $M \leq 6 d \log ( 1 + T / d ) / \alpha ^ { 2 } + 1$ and $| \mathcal { E } _ { M } | \leq ( 2 K T +$ $1 ) ^ { M }$ , the choice $\eta = \sqrt { M \log ( 2 K T + 1 ) / ( K T ) }$ gives $\begin{array} { r } { \frac { 4 \log | \mathcal { E } _ { M } | } { \eta } + 3 \eta K T \leq 7 \sqrt { K T M \log ( 2 K T + 1 ) } } \end{array}$ With $\alpha = [ K \log ( 2 K T + 1 ) ] ^ { 1 / 4 }$ , the right-hand side is at most $7 \left\lceil \sqrt { 6 } + ( 3 / \log 2 ) ^ { 1 / 4 } \right\rceil [ K \log ( 2 K T +$ $1 ) ] ^ { 1 / 4 } \sqrt { T d \log ( 1 + T / d ) }$ . Adding the played-uncertainty and failure terms and using $\phi ( 1 ) > 0 . 2 4 1$ gives a coeficient below $1 2 4 < 1 3 0$

For the geometric branch, take $\alpha = [ \operatorname* { m i n } \{ K , d + 1 \} \log ( 2 K T + 1 ) ] ^ { 1 / 4 }$ . The general geometric bound gives

$$
\begin{array} { r l r } {  { \mathbb { E } [ R _ { T } ] \le 8 4 [ \operatorname* { m i n } \{ K , d + 1 \} \log ( 2 K T + 1 ) ] ^ { 1 / 4 } \sqrt { T d \log ( 1 + T / d ) } } } \\ & { } & { + ~ 3 2 \sqrt { \operatorname* { m i n } \{ K , d + 1 \} T \log ( 2 K T + 1 ) } . } \end{array}\tag{E.21}
$$

When $K \geq d \geq 2$ and $T \geq d ,$ we have min $\{ K , d + 1 \} \le 3 d / 2 , d \log ( 1 + T / d ) \le d \log ( 2 d T )$ , and $\log ( 2 K T + 1 ) \leq$ log $K + \log ( 2 d T ) \leq 2$ log $K \log ( 2 d T )$ . If log $K \leq d$ , the first term in (E.21) is at most 111 $\log ( 2 d T ) d ^ { 3 / 4 } { \sqrt { T } } ( \log K ) ^ { 1 / 4 }$ . The second satisfies the same bound with coeficient 56. Their sum gives the coeficient 167. For OFUL, using log $T \leq d \log ( 1 + T / d ) , d \log ( 1 + T / d ) > 1$ , and $2 / T \leq d \log ( 1 + T / d ) \sqrt { T }$ in (D.10) gives $\mathbb { E } [ R _ { T } ] \le ( 5 + 4 \sqrt { 5 } ) d \log ( 1 + T / d ) \sqrt { T } < 1 6 d \log ( 1 + T / d ) \sqrt { T }$

In particular, if log $K > d ,$ , then $\mathbb { E } [ R _ { T } ] \le 1 6 \log ( 2 d T ) d \sqrt { T }$ . Together with the deterministic bound $R _ { T } \leq 2 T$ used in Case 2 of the proof of Theorem 4, these estimates cover both branches and give (5.9) with $C = \operatorname* { m a x } \{ 1 6 7 , 1 6 \} = 1 6 7$

## F Noise Extensions

The repair potential and the geometric master require only conditional subGaussian moments. The small-menu algorithm is diferent because its bounded loss observation ${ \bf 1 } _ { \{ Y _ { t } \leq 0 \} }$ uses the Gaussian identity $\mathbb { E } _ { t } [ \mathbf { 1 } _ { \{ Y _ { t } < 0 \} } \mid A _ { t } = i ] = \Phi ( - \mu _ { t , i } )$ . We therefore treat the general subGaussian extension and the bounded-reward small-menu extension separately.

Lemma 16. Condition on the history, current menu, and chosen action. Suppose the centered noise satisfies $\mathbb { E } e ^ { \lambda \varepsilon } \le e ^ { \lambda ^ { 2 } / 2 }$ for every $\lambda \in \mathbb { R }$ . Then, for every $u \in \mathbb { R }$ and $r \geq 0$

$$
\mathbb { E } \exp \left\{ \frac { - u ^ { 2 } - 2 u \varepsilon + r \varepsilon ^ { 2 } } { 2 ( 1 + r ) } \right\} \leq \sqrt { 1 + r } .\tag{F.1}
$$

Consequently, the good-repair guarantee of Lemma 10 remains valid with the same constants.

Proof. All expectations may be read conditionally. Introduce an independent $G \sim \mathcal { N } ( 0 , 1 )$ . For $q \in [ 0 , 1 )$ and $a \in \mathbb { R }$ , Tonelli’s theorem and the subGaussian moment bound give

$$
\begin{array} { r l } & { \mathbb { E } _ { \varepsilon } e ^ { q \varepsilon ^ { 2 } / 2 - a \varepsilon } = \mathbb { E } _ { G } \mathbb { E } _ { \varepsilon } e ^ { ( \sqrt { q } G - a ) \varepsilon } \leq \mathbb { E } _ { G } e ^ { ( \sqrt { q } G - a ) ^ { 2 } / 2 } } \\ & { \quad \quad = ( 1 - q ) ^ { - 1 / 2 } \exp \left\{ \frac { a ^ { 2 } } { 2 ( 1 - q ) } \right\} . } \end{array}\tag{F.2}
$$

In (F.2), set $q = r / ( 1 + r )$ and $a = { u } / { ( 1 + r ) }$ and multiply by $e ^ { - u ^ { 2 } / ( 2 ( 1 + r ) ) }$ . The terms involving u cancel and give (F.1). Replacing (E.6) by this inequality leaves the supermartingale argument unchanged. □

Corollary 17 (subGaussian upper bounds). Suppose Y<sub>t</sub> has conditional mean $x _ { t } ^ { \top } \theta$ and conditionally 1-subGaussian centered noise after the current menu and action are fixed. Then the choice between Algorithm 2 and OFUL with $\lambda = 1$ and $\delta = T ^ { - 2 }$ , as specified in Theorem $^ { 4 , }$ satisfies the same bound. Moreover, for $2 \leq K \leq d \leq T ,$ , min $\{ K , d + 1 \} = K$ and $\alpha = [ K \log ( 2 K T + 1 ) ] ^ { 1 / 4 }$ Run Repair-Geo with $\eta ~ = ~ \sqrt { M \log ( 2 K T + 1 ) / ( 1 6 K T ) }$ when M lo $\ r ( 2 K T + 1 )  T / ( 1 6 K )$ , and otherwise run OFUL with $\lambda \ : = \ : 1$ and $\delta \ = \ T ^ { - 2 }$ as in Lemma $1 \not \angle \cdot$ . This choice gives $\mathbb { E } R _ { T } \ \leq$ $1 3 0 [ K \log ( 2 K T + 1 ) ] ^ { 1 / 4 } { \sqrt { d T \log ( 1 + T / d ) } }$

Proof. Lemma 16 gives the same good repair sequence as before. The lifted observation has centered noise $\varepsilon _ { t } / 2$ , so it satisfies the assumption of Lemma 13. Its feature rank is at most min $\{ K , d + 1 \}$ Hence the geometric-branch proof of Theorem 4 applies without change, while the OFUL guarantee of Lemma 14 already holds for conditionally 1-subGaussian centered noise.

For $K \leq d , \operatorname* { m i n } \{ K , d + 1 \} = K$ . When $M \log ( 2 K T + 1 ) \leq T / ( 1 6 K )$ , the geometric bound (D.12) applies. Otherwise, $R _ { T } ~ \leq ~ 2 T$ and M $\log ( 2 K T + 1 ) > T / ( 1 6 K )$ give $\mathbb { E } [ R _ { T } ] ~ \le ~ 2 T ~ <$ $8 \sqrt { K T M \log ( 2 K T + 1 ) } \leq 3 2 \sqrt { K T M \log ( 2 K T + 1 ) }$ . Thus the right-hand side of (D.12) still applies. For the constant, apply $M \leq 6 d \log ( 1 + T / d ) / \alpha ^ { 2 } + 1$ directly in (D.12), before rounding its coeficients. After division by $\alpha { \sqrt { T d \log ( 1 + T / d ) } }$ , the resulting bound is at most $3 2 { \sqrt { 6 } } + 2 { \sqrt { 2 } } +$ $3 2 ( 3 / \log 2 ) ^ { 1 / 4 } + 1 < 1 3 0$ . Here the third term uses $K \log ( 2 K T + 1 ) \leq 3 [ d \log ( 1 + T / d ) ] ^ { 2 } / \log 2$ , and the last term bounds $2 / T$ . This proves the stated constant.

Corollary 18 (Bounded rewards and the IX implementation). Suppose instead that $Y _ { t } \in [ - 1 , 1 ]$ almost surely with conditional mean $x _ { t } ^ { \top } \theta$ . Modify Algorithm 1 only by replacing its loss observation with

$$
Z _ { t } : = \frac { ( 1 - Y _ { t } ) / 2 + 1 - \beta _ { t , A _ { t } } } { 2 } .\tag{F.3}
$$

Then, for every $\alpha > 0$ and $\eta > 0$

$$
\mathbb { E } R _ { T } \leq 2 \left( \frac { 4 \log \left| \mathcal { E } _ { M } \right| } { \eta } + 3 \eta K T + \alpha \sqrt { 2 T d \log ( 1 + T / d ) } + \frac { 2 } { T } \right) .\tag{F.4}
$$

In particular, the tuning in Theorem 3 gives the same bound (5.6). No bounded-reward lower bound is asserted.

Proof. The centered reward noise has support in an interval of length two and is therefore 1- subGaussian. Hence Lemma 16 gives the same good repair sequence.

For this setting, use the bounded loss $\ell _ { t } ( i ) : = ( 1 - \mu _ { t , i } ) / 2$ and $\bar { \ell } _ { t } ( i ) : = ( \ell _ { t } ( i ) + 1 - \beta _ { t , i } ) / 2$ . The statistic in (F.3) lies in [0, 1] and has conditional mean $\bar { \ell } _ { t } ( i )$ when action i is played.

On the event of Lemma 10, the good repair expert satisfies $\mu _ { t , i _ { t } ^ { \star } } - \mu _ { t , a _ { t } ^ { \star } } \leq 2 \alpha w _ { t , a _ { t } ^ { \star } }$ , and therefore $\ell _ { t } \big ( a _ { t } ^ { \star } \big ) - \ell _ { t } \big ( i _ { t } ^ { \star } \big ) \leq \alpha w _ { t , a _ { t } ^ { \star } }$ . The loss gap is also at most one, so it is at most min $\{ 1 , \overset { \cdot } { \alpha } w _ { t , a _ { t } ^ { \star } } \} = \beta _ { t , a _ { t } ^ { \star } }$ Thus the same discounted-loss argument gives $\bar { \ell } _ { t } ( a _ { t } ^ { \star } ) \leq ( \ell _ { t } ( i _ { t } ^ { \star } ) + 1 ) / 2$

Applying exactly the same decomposition as in the proof of Theorem 3, Lemma 11 controls the repair-expert comparison and (D.4) controls the remaining played uncertainty. Since reward regret is exactly twice the surrogate-loss regret, we obtain (F.4). Substituting the parameters from Theorem 3 gives (5.6). □