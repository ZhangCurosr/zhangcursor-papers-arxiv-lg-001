# Optimal Alternating Regret for Online Learning and Games<sup>∗</sup>

Yixin Tao<sup>†</sup> Weiqiang Zheng<sup>‡</sup>

## Abstract

For OLO over the probability simplex $\Delta _ { d } ,$ we give an algorithm with $O ( \log d )$ alternating regret that remains a constant for any time horizon T, and a matching lower bound. Our constant regret bound significantly improves previous results with $O ( \log ^ { 2 / 3 } d \cdot T ^ { 1 / 3 } )$ regret [Cevher, Cutkosky, Kavis, Piliouras, Skoulakis, Viano, NeurIPS 2023, Hait, Li, Luo, Zhang, COLT 2025]. As a result, we obtain alternating learning dynamics with $O ( \log d / T )$ convergence to Nash equilibria in two-player zero-sum games and $O ( \log d / T )$ convergence to coarse correlated equilibria in two-player general-sum games. This is the first uncoupled learning dynamics with $O ( 1 / T )$ convergence to CCE in two-player general-sum games, while all prior works sufer additional log T factors.

For general OCO over a d-dimensional compact convex set, we give an algorithm with $O ( d \log ( 1 + T / d ) )$ alternating regret, improving the previous best of ${ \widetilde O } ( d ^ { 2 / 3 } T ^ { 1 / 3 } )$ . We also prove a matching lower bound of $\Omega ( d \log ( 1 + T / d ) )$ , showing that the $\Omega ( \log T )$ factor is unavoidable.

## Contents

1 Introduction 2   
1.1 Our results 3   
1.1.1 Optimal alternating regret . 4   
1.1.2 Fast alternating learning dynamics in games 4   
1.2 Further related works 6   
2 Constant alternating regret for OLO over the simplex 6   
2.1 The alternation-aware Hedge algorithm 6   
2.2 A potential proof of O(log d) alternating regret 7   
2.3 An Ω(log d) lower bound . 8   
3 Alternating regret for online convex optimization 9   
3.1 The continuous alternation-aware Hedge algorithm 9   
3.2 Alternating regret bounds . 10   
4 Matching lower bounds for OCO 11   
4.1 An $\Omega ( \log T )$ lower bound for OCO over 2-dimensional ball . 11   
4.2 An $\Omega ( d \log ( 1 + T / d ) )$ lower bound for OCO 14   
5 Conclusion 16   
A Proofs for Online Convex Optimization 19   
A.1 Existence and feasibility of the update . 19   
A.2 A potential proof of logarithmic alternating regret 19   
A.3 Proof of Theorem 7 . 20

## 1 Introduction

We consider the standard online convex optimization (OCO) problem [Zinkevich, 2003, Hazan, 2016, Orabona, 2019]. In each iteration $t \in [ T ]$ , the learner first chooses an action $\mathbf { \Phi } _ { \mathbf { \pmb { x } } _ { t } } ~ \in ~ \mathcal { X }$ from a d-dimensional convex set $x ,$ then the adversary chooses a bounded convex loss function $f _ { t } : \mathcal { X } \to [ - 1 , 1 ]$ and the learner sufers loss $f _ { t } ( \pmb { x } _ { t } )$

Regret and learning in games. The classic goal is to minimize the (external) regret defined as

$$
\mathrm { R e g } ^ { T } : = \operatorname* { m a x } _ { \boldsymbol { x } \in \mathcal { X } } \left\{ \mathrm { R e g } ^ { T } ( \boldsymbol { x } ) = \sum _ { t = 1 } ^ { T } f _ { t } ( \boldsymbol { x } _ { t } ) - \sum _ { t = 1 } ^ { T } f _ { t } ( \boldsymbol { x } ) \right\} .
$$

(standard regret)

If the loss functions $\{ f _ { t } ( \pmb { x } ) = \pmb { c } _ { t } ^ { \top } \pmb { x } \}$ are linear, the problem is known as the online linear optimization (OLO) problem. If additionally the action set $\mathcal { X } = \Delta _ { d }$ is the probability simplex, the problem reduces to the classic adversarial d-expert problem. The general OCO algorithm has regret $O ( \sqrt { T } )$ 2 which is tight by the lower bound of $\Omega ( { \sqrt { T } } )$ even for the 2-expert problem.

Algorithms for $\mathrm { O C O / O L O }$ have applications for learning in large-scale games and computing approximate equilibria. It is well-known that if players simultaneously employ OLO algorithms to choose their strategies, then their time-averaged strategy converges to approximate Nash equilibria (NE) in two-player zero-sum games; their empirical distribution of play converges to approximate coarse correlated equilibria (CCE) in general-sum games [Foster and Vohra, 1997, Freund and Schapire, 1999, Hart and Mas-Colell, 2000, Cesa-Bianchi and Lugosi, 2006]. The approximation depends on the average regret of the algorithm. Thus an algorithm with $\sqrt { T }$ regret in the adversarial setting implies $1 / \sqrt { T }$ convergence rate in games. However, better convergence rate is possible in the game setting where every player employs the same algorithm. For two-player zero-sum games, the pioneering work of [Daskalakis, Deckelbaum, and Kim, 2011] show $O ( \log T / T )$ convergence rate to NE, while optimistic online learning algorithms achieve the optimal $O ( 1 / T )$ convergence rate [Rakhlin and Sridharan, 2013], which can even be extended to last-iterate convergence [Cai and Zheng, 2023]. For general-sum games, Syrgkanis, Agarwal, Luo, and Schapire [2015] show the first fast convergence rate of $O ( 1 / T ^ { 1 / 4 } )$ , and a long line of work improve the convergence rate to O(log T/T) [Chen and Peng, 2020, Daskalakis, Deckelbaum, and Kim, 2011, Anagnostides, Daskalakis, Farina, Fishelson, Golowich, and Sandholm, 2022a, Anagnostides, Farina, Kroer, Lee, Luo, and Sandholm, 2022b, Farina, Anagnostides, Luo, Lee, Kroer, and Sandholm, 2022, Cai, Luo, Wei, and Zheng, 2024, Soleymani, Piliouras, and Farina, 2025b,a], some of which even hold for the stronger notion of correlated equilibrium and general convex games. A very recent work reports a $O ( \sqrt { \log T } / T )$ convergence rate [Tsuchiya, 2026]. However, a major problem for learning in games remains open: can we achieve $O ( 1 / T )$ convergence in general-sum games? This problem was open even for two-player games.

Alternating regret and learning dynamics. In this paper, we focus on alternation, a simple trick for two-player games where the players take turns to update their strategies (see Section 1.1.2 for a formal definition). Compared to simultaneous learning, alternating learning dynamics have strong empirical performance and are used for training superhuman AI models for poker [Tammelin, 2014, Bowling, Burch, Johanson, and Tammelin, 2015, Brown and Sandholm, 2018, 2019]. A theoretical explanation of the power of alternation, however, remained elusive until recent work. Wibisono, Tao, and Piliouras [2022] show that alternation with the classic Hedge algorithm [Freund and Schapire, 1997] gives a fast $O ( 1 / T ^ { 2 / 3 } )$ convergence rate in two-player zero-sum games. In their analysis, they propose a new regret notion capturing the nature of alternation, later termed as alternating regret by Cevher, Cutkosky, Kavis, Piliouras, Skoulakis, and Viano [2023]. In the most general setting of OCO, the alternating regret is defined as follows:

$$
\mathrm { R e g } _ { \mathrm { A l t } } ^ { T } : = \operatorname* { m a x } _ { x \in \mathcal { X } } \left\{ \mathrm { R e g } _ { \mathrm { A l t } } ^ { T } ( x ) = \sum _ { t = 1 } ^ { T } ( f _ { t - 1 } ( x _ { t } ) + f _ { t } ( x _ { t } ) ) - \sum _ { t = 1 } ^ { T } ( f _ { t - 1 } ( x ) + f _ { t } ( x ) ) \right\} .
$$

(alternating regret)

So the alternating regret can be seen as the standard regret with respect to the loss function $f _ { t - 1 } + f _ { t }$ in each iteration t (define $f _ { 0 } = 0 )$ . In alternating regret, the learner knows $f _ { t - 1 }$ when they choose the strategy $\mathbf { \mathcal { x } } _ { t }$ . This indicates that we may get $o ( \sqrt { T } )$ alternating regret even in the adversarial setting where standard regret sufers $\Omega ( { \sqrt { T } } )$ bound.

The work by Cevher, Cutkosky, Kavis, Piliouras, Skoulakis, and Viano [2023] show the first $o ( \sqrt { T } )$ alternating regret in the adversarial setting. They provide algorithms with $O ( \log ^ { 4 / 3 } ( d T ) \cdot \dot { T } ^ { 1 / 3 } )$ regret for OLO over the probability simplex $\mathcal { X } = \Delta _ { d } ~ \mathrm { ( i . e . }$ , the d-expert problem), and even O(log T) when X is the $\ell _ { 2 }$ unit ball. Their results were improved by Hait, Li, Luo, and Zhang [2025], who show that the classic Hedge algorithm already achieves an improved $O ( \log ^ { 2 / 3 } d \cdot T ^ { 1 / \bar { 3 } } )$ alternating regret for the d-expert problem. Moreover, Hait, Li, Luo, and Zhang [2025] show that the continuous Hedge algorithm [Narayanan and Rakhlin, 2010] achieves $\widetilde { \cal O } ( d ^ { 2 / 3 } \dot { T } ^ { 1 / 3 } )$ alternating regret for the general OCO problem over any convex set X.

However, it is unknown whether $T ^ { 1 / 3 }$ is a fundamental limit for alternating regret. In this paper, we attack the main open question on alternating regret:

## what is the minimax-optimal alternating regret for OLO/OCO?

As remarked by Hait, Li, Luo, and Zhang [2025], “This appears to be highly non-trivial even for very special cases—for example, even for 1-dimensional OLO over [−1, 1] where O(log T) regret is achieved by Cevher et al. [2023], figuring out whether the optimal bound is $\Theta ( \log T )$ or Θ(1) appears to require new techniques.”<sup>1</sup>

## 1.1 Our results

In this paper, we settle the minimax-optimal alternating regret for OLO and OCO. We provide algorithms with optimal alternating regret with matching lower bounds, and fast alternating learning dynamics in two-player games.

## 1.1.1 Optimal alternating regret

Online linear optimization over the probability simplex. We propose an online algorithm with $O ( \log d )$ alternating regret for OLO over the probability simplex $\Delta _ { d } .$ . The algorithm is anytime and does not need to know the time horizon T. It is remarkable that the alternating regret does not grow with T.

Theorem 1 (Constant alternating regret). For any $T \geq 1$ and adaptive chosen linear loss sequence $\begin{array} { r } { \pmb { c } _ { 1 } , \ldots , \pmb { c } _ { T } \in [ - 1 , 1 ] ^ { d } ; } \end{array}$ , the alternation-aware Hedge (AA-Hedge) algorithm (Algorithm 1) over $\Delta _ { d }$ guarantees that $\operatorname { R e g } _ { \mathrm { A l t } } ^ { T } \leq 8 \log d .$

We also show that for any $T \geq d - 1$ , an adaptive adversary could force the alternating regret to be at least $\mathrm { R e g } _ { \mathrm { A l t } } ^ { T } \geq 2 \log d - O ( 1 )$ (Theorem 6). Thus the minimax-optimal alternating regret for the d-expert problem is $\Theta ( \log d )$

The AA-Hedge algorithm (Algorithm 1) is a variant of the Hedge algorithm with non-trivial modifications. It has been shown that the vanilla Hedge algorithm can not achieve $o ( T ^ { 1 / 3 } )$ alternating regret [Hait, Li, Luo, and Zhang, 2025]. In round t, Algorithm 1 fully exploits the fact that the known loss $c _ { t - 1 }$ is part of that round’s loss. The played strategy $\mathbf { \mathcal { x } } _ { t }$ guarantees for any unknown $^ { c _ { t } , }$ the instant loss $( \pmb { c } _ { t - 1 } + \pmb { c } _ { t } ) ^ { \top } \pmb { x } _ { t }$ can be charged on a potential function without any additional terms. A potential analysis similar to that of the Hedge algorithm then concludes the constant alternating regret bound in general.

Online convex optimization. Motivated by the constant regret results for OLO over the simplex, we then study the general setting of online convex optimization over d-dimensional compact convex sets. We recall that the previous best is the $\widetilde { \cal O } ( d ^ { 2 / 3 } \bar { \cal T } ^ { 1 / 3 } )$ regret by Hait, Li, Luo, and Zhang [2025], achieved by the continuous Hedge algorithm. In Section 3, we extend Algorithm 1 and our analysis to the general online convex optimization setting with convex action set X, using ideas similar to the continuous Hedge algorithm. The resulting continuous AA-Hedge (Algorithm 2) enjoys $O ( d \log ( 1 + T / d ) )$ alternating regret.

Theorem 2 (Informal, alternating regret upper bound for OCO). For online convex optimization over any compact convex set $\boldsymbol { \mathcal { X } } \subseteq \mathbb { R } ^ { d }$ with any bounded convex losses, the continuous AA-Hedge (Algorithm 2) has $O ( d \log ( 1 + T / d ) )$ alternating regret.

For the special case of OLO over a convex polytope with N vertices, the continuous AA-Hedge algorithm enjoys a better alternating regret of 8 log N, which also recovers Theorem 1.

Somewhat surprisingly, we show that the $\Omega ( \log T )$ dependence is necessary even when the set $\mathcal { X }$ is the 2-dimensional unit ball (Theorem 9). Using this lower bound, we further establish a matching lower bound for OCO over d-dimensional convex sets (Theorem 10).

Theorem 3 (Informal, alternating regret lower bound for OCO). There exists a compact convex set X that lies in the d-dimensional unit $\ell _ { 2 }$ ball such that for any OCO algorithm, there exists an oblivious adversary such that the algorithm sufers $\Omega ( d \log ( 1 + T / d ) )$ alternating regret

Our upper and lower bounds together establish the minimax-optimal $\Theta ( d \log ( 1 { + } T / d ) )$ ) alternating regret for OCO (Corollary 3).

## 1.1.2 Fast alternating learning dynamics in games

As an important application, our result implies in alternating learning dynamics with fast convergence in two-player games. Consider a two-player convex game with loss functions $u _ { 1 } , u _ { 2 } : \mathcal { X } \times \mathcal { Y }  [ - 1$ , 1] for the x-player and the y-player respectively. The loss $u _ { 1 } ( { \pmb x } , { \pmb y } )$ is convex in x for any $\pmb { y } \in \mathcal { V }$ and the loss $u _ { 2 } ( { \pmb x } , { \pmb y } )$ is convex in y for any $\mathbf { \boldsymbol { x } } \in \mathcal { X }$ . The game is zero-sum if $u _ { 1 } = - u _ { 2 }$ and otherwise is general-sum. An important subclass is normal-form games.

Example 1 (Normal-form games). In a two-player normal-form game, X and Y are probability simplices and the loss functions are linear: $u _ { 1 } ( { \pmb x } , { \pmb y } ) = { \pmb x } ^ { \top } A { \pmb y } , u _ { 2 } ( { \pmb x } , { \pmb y } ) = { \pmb x } ^ { \top } B { \pmb y }$ . The game is zero-sum if $A = - B$ , otherwise it is general-sum.

We recall the definition of approximate Nash equilibrium and coarse correlated equilibrium.

Definition 1 (Nash equilibrium and coarse correlated equilibrium). A joint strategy profile $( { \pmb x } , { \pmb y } )$ is an ϵ-Nash equilibrium $\begin{array} { r } { ( \epsilon \mathrm { N E } ) \ i f u _ { 1 } ( { \pmb x } , { \pmb y } ) \leq \operatorname* { m i n } _ { { \pmb x } ^ { \prime } \in { \mathcal X } } u _ { 1 } ( { \pmb x } ^ { \prime } , { \pmb y } ) + \epsilon } \end{array}$ and $\begin{array} { r } { u _ { 2 } ( \pmb { x } , \pmb { y } ) \le \operatorname* { m i n } _ { \pmb { y } ^ { \prime } \in \mathcal { X } } u _ { 2 } ( \pmb { x } , \pmb { y } ^ { \prime } ) + \epsilon } \end{array}$ A distribution σ over joint strategy profiles $\mathcal { X } \times \mathcal { V }$ is an ϵ-coarse correlated equilibrium (ϵ-CCE) if $\begin{array} { r } { \mathbb { E } _ { ( \alpha , y ) \sim \sigma } [ u _ { 1 } ( x , y ) ] \leq \operatorname* { m i n } _ { x ^ { \prime } \in \mathcal { X } } \mathbb { E } _ { ( \alpha , y ) \sim \sigma } [ u _ { 1 } ( x ^ { \prime } , y ) ] + \epsilon \ a n d \mathbb { E } _ { ( \alpha , y ) \sim \sigma } [ u _ { 2 } ( x , y ) ] \leq \operatorname* { m i n } _ { y ^ { \prime } \in \mathcal { Y } } \mathbb { E } _ { ( \alpha , y ) \sim \sigma } [ u _ { 2 } ( x , y ^ { \prime } ) ] + \epsilon } \end{array}$ ϵ.

In a simultaneous learning dynamics, for every iteration $t \geq 1$ , both players use an OLO/OCO algorithm to choose their actions $\mathbf { \boldsymbol { x } } _ { t } \in \mathcal { X }$ and $\pmb { y } _ { t } \in \mathcal { V }$ simultaneously and then they observe their loss functions $u _ { 1 } ( \cdot , y _ { t } )$ and $u _ { 2 } ( x _ { t } , \cdot )$ . The alternating learning dynamics is slightly diferent: in iteration $t ,$ the x-player chooses $\mathbf { \mathcal { x } } _ { t }$ first; then after seeing $\mathbf { \Delta } \mathbf { x } _ { t } .$ , the y-player chooses $\mathbf { \mathscr { y } } _ { t }$ . More specifically, in each round $t \geq 1$ ，

1. the x-player chooses $\mathbf { \mathcal { x } } _ { t }$ based on $u _ { 1 } ( \cdot , \pmb { y } _ { 1 } ) , \dots , u _ { 1 } ( \cdot , \pmb { y } _ { t - 1 } )$ ;

2. the y-player chooses $\pmb { y } _ { t }$ based on $u _ { 2 } ( { \pmb x } _ { 1 } , \cdot ) , \ldots , u _ { 2 } ( { \pmb x } _ { t - 1 } , \cdot ) , u _ { 2 } ( { \pmb x } _ { t } , \cdot ) ;$

Algorithms for $\mathrm { O L O / O C O }$ with $o ( T )$ alternating regret imply convergence to approximate equilibria in two-player games, as observed by [Wibisono, Tao, and Piliouras, 2022, Cevher, Cutkosky, Kavis, Piliouras, Skoulakis, and Viano, 2023] for NE in zero-sum games and by [Hait, Li, Luo, and Zhang, 2025] for CCE in general-sum games. Formally, we have

Theorem 4 (Theorem 1 and 2 in [Hait, Li, Luo, and Zhang, 2025]). Suppose that in the above alternating learning dynamics, the x-player and the y-player use OCO algorithms with alternating regret bound $\mathrm { R e g } _ { \mathrm { A l t } } ^ { T } ( x )$ and $\mathrm { R e g } _ { \mathrm { A l t } } ^ { T } ( y )$ , respectively. Then

1. for a zero-sum game, the averaged strategy $( 1 / T \textstyle \sum _ { t = 1 } ^ { T } \pmb { x } _ { t } , 1 / T \textstyle \sum _ { t = 1 } ^ { T } \pmb { y } _ { t } )$ is an $\epsilon { - } N E$ with $\epsilon = O ( ( \mathrm { R e g } _ { \mathrm { A l t } } ^ { T } ( x ) + \mathrm { R e g } _ { \mathrm { A l t } } ^ { T } ( y ) ) / T )$ ;

2. for a general-sum game, the uniform distribution over $\{ ( \pmb { x } _ { t } , \pmb { y } _ { t } ) , ( \pmb { x } _ { t + 1 } , \pmb { y } _ { t } ) \} _ { t \in [ T ] }$ is an ϵ-CCE with $\epsilon = O ( \operatorname* { m a x } \{ \mathrm { R e g } _ { \mathrm { A l t } } ^ { T } ( x ) , \mathrm { R e g } _ { \mathrm { A l t } } ^ { T } ( y ) \} / T )$

As a corollary, our results imply alternating learning dynamics with fast convergence in games. Notably, we give the first learning dynamics with $O ( 1 / T )$ convergence to CCE in two-player general-sum normal-form games, while existing results sufer additional poly log T factors.

Corollary 1 (Fast convergence in two-player games). For two-player normal-form games where each player has at most d actions, there is an alternating learning dynamics with O(log d/T) convergence to NE in zero-sum games and $O ( \log d / T )$ convergence to CCE in general-sum games.

## 1.2 Further related works

Alternation has been studied in two-player zero-sum games and the related min-max optimization problem, mostly in the unconstrained setting where the strategy set is the whole Euclidean space. There is a line of work on the convergence of alternating gradient-descent-ascent in the unconstrained setting [Bailey, Gidel, and Piliouras, 2020, Zhang, Wang, Lessard, and Grosse, 2022, Lee, Cho, and Yun, 2024, Feng, Fujii, Skoulakis, Wang, and Cevher, 2025, Shugart and Altschuler, 2025], the simplex-constrained setting [Nan, Das Gupta, Iyengar, and Kroer, 2026], and manifold setting [Xu, Jiang, Liu, and ${ \mathrm { S o } } ,$ 2026]. The diference is that our work considers alternating regret in the more challenging setting of adversarial online learning, which implies convergence in the game setting.

## 2 Constant alternating regret for OLO over the simplex

In this section, we focus on the important case of online linear optimization over the simplex, i.e., the adversarial expert problem and present an algorithm with constant alternating regret.

We first introduce some notations. The action set is the d-dimensional probability simplex $\Delta _ { d } = \left\{ { \pmb x } \in \mathbb { R } ^ { d } : x _ { i } \geq 0 \right.$ and $\begin{array} { r } { \sum _ { i = 1 } ^ { d } x _ { i } = 1 \Big \} } \end{array}$ We define $c _ { 0 } = \mathbf { 0 }$ to be the all-0 vector. In each iteration $t \geq 1$ , the learner chooses $\pmb { x } _ { t } \in \Delta _ { d }$ based on the history $\{ c _ { 1 } , \ldots , c _ { t - 1 } \}$ . After observing $\mathbf { \Delta } \mathbf { x } _ { t } .$ , the adversary adaptively chooses $\pmb { c } _ { t } \in [ - 1 , 1 ] ^ { d }$ . The learner then observes $c _ { t }$ and sufers loss $( \pmb { c } _ { t - 1 } + \pmb { c } _ { t } ) ^ { \top } \pmb { x } _ { t }$ . In this case, the alternating regret can be written as

$$
{ \mathrm { R e g } } _ { \mathrm { A l t } } ^ { T } = \sum _ { t = 1 } ^ { T } ( { \pmb { c } } _ { t - 1 } + { \pmb { c } } _ { t } ) ^ { \top } { \pmb { x } } _ { t } - \operatorname* { m i n } _ { { \pmb { x } } \in \Delta _ { d } } \sum _ { t = 1 } ^ { T } ( { \pmb { c } } _ { t - 1 } + { \pmb { c } } _ { t } ) ^ { \top } { \pmb { x } } .\tag{1}
$$

Because the objective is linear, the comparator in (1) is one of the d pure actions.

## 2.1 The alternation-aware Hedge algorithm

We propose the alternation-aware Hedge algorithm, which introduce a bias toward the most recent known loss vector $c _ { t - 1 }$ when choosing $\scriptstyle { \mathbf { { x } } } _ { t } .$ . The algorithm is in Algorithm 1, where we define $\mathbf { { L } } _ { 0 } = \mathbf { { 0 } }$ and $\begin{array} { r } { \pmb { L } _ { t } = \sum _ { s = 1 } ^ { t } ( \pmb { c } _ { s - 1 } + \pmb { c } _ { s } ) } \end{array}$ as the efective cumulative loss for $t \geq 1$ . We first show that each step in Algorithm 1 is well-defined and feasible: (1) $\theta _ { t }$ is unique; (2) the played strategy $\mathbf { \mathcal { x } } _ { t }$ lies in the simplex $\Delta _ { d } .$

```tcl
Algorithm 1 alternation-aware Hedge
1: Set $\pmb { L } _ { 0 } = \mathbf { 0 } , \eta = 1 / 8 ,$ and $\beta = 1 / 3 2 .$
2: for $t = 1 , 2 , \dots$ do
3: Compute $\theta _ { t } = \arg$ ma $\mathbf { \boldsymbol { x } } _ { \theta \in \mathbb { R } }$ log $\begin{array} { r } { \sum _ { i = 1 } ^ { d } \exp \left[ - \eta L _ { t - 1 , i } - \beta ( c _ { t - 1 , i } - \theta ) ^ { 2 } \right] } \end{array}$
4: Set $\mathbf { } _ { \pmb { p } _ { t } }$ such that $p _ { t , i } \propto \exp \left[ - \eta L _ { t - 1 , i } - \beta ( c _ { t - 1 , i } - \theta _ { t } ) ^ { 2 } \right]$
5: Play $\mathbf { \mathcal { x } } _ { t }$ such that $\begin{array} { r } { x _ { t , i } = p _ { t , i } \left( 1 - \frac { c _ { t - 1 , i } - \theta _ { t } } { 2 } \right) } \end{array}$
6: Observe $c _ { t }$ and update $\pmb { L } _ { t } \gets \pmb { L } _ { t - 1 } + \pmb { c } _ { t - 1 } + \pmb { c } _ { t }$
```

Lemma 1. Assume $\beta < 1 / 2$ . In Algorithm 1, $\theta _ { t }$ is unique and $\begin{array} { r } { \theta _ { t } = \sum _ { i = 1 } ^ { d } p _ { t , i } c _ { t - 1 , i } } \end{array}$ . Moreover, $\pmb { x } _ { t } \in \Delta _ { d }$

Proof. Define $\begin{array} { r } { F ( \theta ) = \log { \sum _ { i = 1 } ^ { d } \exp \left[ - \eta L _ { t - 1 , i } - \beta ( c _ { t - 1 , i } - \theta ) ^ { 2 } \right] } } \end{array}$ and $\pmb { p } ( \theta ) \in \Delta _ { d }$ such that $p _ { i } ( \theta ) \propto$ $\exp [ - \eta L _ { t - 1 , i } - \beta ( c _ { t - 1 , i } - \theta ) ^ { 2 } ]$ . We can calculate both the first-order and the second-order derivative

of $F ( \theta )$

$$
\begin{array} { r l } & { F ^ { \prime } ( \theta ) = 2 \beta ( \mathbb { E } _ { p ( \theta ) } ( \pmb { c } _ { t - 1 } ) - \theta ) } \\ & { F ^ { \prime \prime } ( \theta ) = 4 \beta ^ { 2 } \cdot \operatorname { V a r } _ { p ( \theta ) } ( \pmb { c } _ { t - 1 } ) - 2 \beta } \end{array}
$$

where we define the expectation $\begin{array} { r } { \mathbb E _ { p ( \theta ) } ( \pmb { c } _ { t - 1 } ) = \sum _ { i } p _ { i } ( \theta ) c _ { t - 1 , i } } \end{array}$ and $\begin{array} { r } { \operatorname { V a r } _ { \pmb { p } ( \theta ) } ( \pmb { c } _ { t - 1 } ) = \sum _ { i } p _ { i } ( \theta ) ( c _ { t - 1 , i } - } \end{array}$ $\mathbb { E } _ { p ( \theta ) } ( \pmb { c } _ { t - 1 } ) ) ^ { 2 }$ . Since $c _ { t - 1 , i } \in [ - 1 , 1 ]$ , the variance term $\operatorname { V a r } _ { p ( \theta ) } ( c _ { t - 1 } ) \leq 1$ . Since $\beta < \frac { 1 } { 2 }$ , we have $F ^ { \prime \prime } ( \theta ) < 0$ and F is thus strictly concave. Moreover, $F ( \theta ) \to - \infty { \mathrm { ~ a s ~ } } | \theta | \to + \infty$ . Thus F has a unique maximizer $\theta _ { t }$ and ${ \mathbf { } } p _ { t } = p ( \theta _ { t } )$ . The first-order optimality condition $F ^ { \prime } ( \theta _ { t } ) = 0$ implies $\begin{array} { r } { \theta _ { t } = \sum _ { i = 1 } ^ { d } p _ { t , i } c _ { t - 1 , i } \in [ \operatorname* { m i n } _ { i } c _ { t - 1 , i } , \operatorname* { m a x } _ { i } c _ { t - 1 , i } ] } \end{array}$

Both $c _ { t - 1 , i }$ and $\theta _ { t }$ lie in $[ - 1 , 1 ]$ , so $x _ { t , i } = p _ { t , i } ( 1 - ( c _ { t - 1 , i } - \theta _ { t } ) / 2 ) \geq 0$ for all $i \in [ d ]$ . Moreover, $\begin{array} { r } { \theta _ { t } = \sum _ { i = 1 } ^ { d } p _ { t , i } c _ { t - 1 , i } } \end{array}$ implies $\textstyle \sum _ { i = 1 } ^ { d } x _ { t , i } = 1$ . Hence $\pmb { x } _ { t } \in \Delta _ { d }$ □

## 2.2 A potential proof of O(log d) alternating regret

We use a potential function argument to bound the alternating regret. Define the alternating regret vector $\scriptstyle { R _ { t } }$ such that $\begin{array} { r } { R _ { t , i } = \sum _ { s = 1 } ^ { t } ( { c _ { s - 1 } + c _ { s } } ) \top { \pmb x } _ { s } - \sum _ { s = 1 } ^ { t } ( { c _ { s - 1 , i } + c _ { s , i } } ) } \end{array}$ for all $i \in [ d ]$

Potential function. The potential function is

$$
\Phi _ { t } : = \operatorname* { m a x } _ { \theta \in \mathbb { R } } \log \left( \sum _ { i = 1 } ^ { d } \exp \left[ \eta R _ { t - 1 , i } - \beta ( c _ { t - 1 , i } - \theta ) ^ { 2 } \right] \right) = \log \left( \sum _ { i = 1 } ^ { d } \exp \left[ \eta R _ { t - 1 , i } - \beta ( c _ { t - 1 , i } - \theta _ { t } ) ^ { 2 } \right] \right) .\tag{2}
$$

The equality holds by definition of $\theta _ { t }$ in Algorithm 1 and the fact that constant terms do not afect the maximizer. Initially, $\Phi _ { 1 } = \log d .$

Theorem 5 (Monotonicity of potential). Assume $\begin{array} { r } { \eta \le \frac { 1 } { 8 } } \end{array}$ and $\beta = \eta / 4$ . For every round $t \geq 1$ , we have $\Phi _ { t + 1 } \leq \Phi _ { t } \leq$ log d.

Proof. Let us define $\pmb { \xi } _ { t } = \pmb { c } _ { t - 1 } - \theta _ { t } \mathbf { 1 }$ and $\pmb { \xi } _ { t + 1 } = \pmb { c } _ { t } - \theta _ { t + 1 } \mathbf { 1 }$ . Applying Lemma 1 at rounds t and t + 1 gives $\pmb { \xi } _ { t } , \pmb { \xi } _ { t + 1 } \in [ - 2 , 2 ] ^ { d } , \pmb { p } _ { t } ^ { \top } \pmb { \xi } _ { t } = 0$ , and the update gives $x _ { t , i } = p _ { t , i } ( 1 - \xi _ { t , i } / 2 )$ . We have the following identity:

$$
\begin{array} { r } { R _ { t , i } - R _ { t - 1 , i } = ( \pmb { c } _ { t - 1 } + \pmb { c } _ { t } ) ^ { \top } \pmb { x } _ { t } - c _ { t - 1 , i } - c _ { t , i } = ( \pmb { \xi } _ { t } + \pmb { \xi } _ { t + 1 } ) ^ { \top } \pmb { x } _ { t } - \xi _ { t , i } - \xi _ { t + 1 , i } . } \end{array}
$$

By definition of $\Phi _ { t }$ and $p _ { t , i } \propto \exp ( \eta R _ { t - 1 , i } - \beta ( c _ { t - 1 , i } - \theta _ { t } ) ^ { 2 } )$ , we have

$$
\begin{array} { r l } & { \Phi _ { t + 1 } - \Phi _ { t } = \log \left( \frac { \sum _ { i = 1 } ^ { d } \exp \big ( \eta R _ { t , i } - \beta ( c _ { t , i } - \theta _ { t + 1 } ) ^ { 2 } \big ) } { \sum _ { i = 1 } ^ { d } \exp \big ( \eta R _ { t - 1 , i } - \beta ( c _ { t - 1 , i } - \theta _ { t } ) ^ { 2 } \big ) } \right) } \\ & { \quad \quad = \log \left( \displaystyle \sum _ { i = 1 } ^ { d } p _ { t , i } \exp \big ( \eta ( R _ { t , i } - R _ { t - 1 , i } ) + \beta ( \xi _ { t , i } ^ { 2 } - \xi _ { t + 1 , i } ^ { 2 } ) \big ) \right) . } \\ & { \quad \quad = \log \left( \displaystyle \sum _ { i = 1 } ^ { d } p _ { t , i } \exp \Big ( \eta ( ( \xi _ { t } + \xi _ { t + 1 } ) ^ { \top } x _ { t } - \xi _ { t , i } - \xi _ { t + 1 , i } ) + \beta ( \xi _ { t , i } ^ { 2 } - \xi _ { t + 1 , i } ^ { 2 } ) \Big ) \right) . } \end{array}
$$

We then use the following lemma to show that $\Phi _ { t + 1 } \leq \Phi _ { t }$

Lemma 2. Assume $\eta \leq \frac { 1 } { 8 }$ and $\beta = \eta / 4$ . Then for any $t \geq 1$ , we have

$$
\log \left( \sum _ { i = 1 } ^ { d } p _ { t , i } \exp \Big ( \eta \big ( ( \pmb { \xi } _ { t } + \pmb { \xi } _ { t + 1 } ) ^ { \top } \pmb { x } _ { t } - \xi _ { t , i } - \xi _ { t + 1 , i } \big ) + \beta ( \xi _ { t , i } ^ { 2 } - \xi _ { t + 1 , i } ^ { 2 } ) \Big ) \right) \leq 0 .
$$

Proof. Let us define the function $H _ { t } : [ - 2 , 2 ] ^ { d } \to \mathbb { R }$ as follows:

$$
H _ { t } ( \boldsymbol { z } ) = \log \sum _ { i = 1 } ^ { d } p _ { t , i } \exp \Bigl ( \eta \bigl ( ( \boldsymbol { \xi } _ { t } + \boldsymbol { z } ) ^ { \top } \boldsymbol { x } _ { t } - \boldsymbol { \xi } _ { t , i } - \boldsymbol { z } _ { i } \bigr ) + \beta ( \boldsymbol { \xi } _ { t , i } ^ { 2 } - \boldsymbol { z } _ { i } ^ { 2 } ) \Bigr ) .
$$

It sufices to show $H _ { t } ( \pmb { \xi } _ { t + 1 } ) \leq 0$

Define $\pmb q ( z )$ be the distribution obtained by normalizing the summands:

$$
q _ { j } ( z ) = { \frac { p _ { t , j } \exp { \Big ( } \eta { \big ( } ( \xi _ { t } + z ) ^ { \top } \mathbf { x } _ { t } - \xi _ { t , j } - z _ { j } { \big ) } + \beta ( \xi _ { t , j } ^ { 2 } - z _ { j } ^ { 2 } ) { \Big ) } } { \sum _ { i = 1 } ^ { d } p _ { t , i } \exp { \Big ( } \eta { \big ( } ( \xi _ { t } + z ) ^ { \top } \mathbf { x } _ { t } - \xi _ { t , i } - z _ { i } { \big ) } + \beta ( \xi _ { t , i } ^ { 2 } - z _ { i } ^ { 2 } ) { \Big ) } } } .
$$

We can calculate the gradient $\nabla H _ { t } ( z ) = \eta \mathbf { x } _ { t } + \pmb { q } ( z ) \odot ( - \eta \mathbf { 1 } - 2 \beta z )$ , where we use the point-wise product notation $( { \pmb a } \odot { \pmb b } ) _ { i } = a _ { i } b _ { i }$ . We then calculate the Hessian

$$
\begin{array} { r l } & { \nabla ^ { 2 } H _ { t } ( z ) = \operatorname { D i a g } \big [ q _ { i } ( z ) \left( ( - \eta - 2 \beta z _ { i } ) ^ { 2 } - 2 \beta \right) \big ] } \\ & { \qquad - \left( \pmb { q } ( z ) \odot ( - \eta \mathbf { 1 } - 2 \beta z ) \right) \left( \pmb { q } ( z ) \odot ( - \eta \mathbf { 1 } - 2 \beta z ) \right) ^ { \top } , } \end{array}
$$

where we use notation $\mathrm { D i a g } [ a _ { i } ]$ to denote the square $d \times d$ diagonal matrix whose i-th diagonal is $a _ { i }$ for $i \in [ d ]$ . For any $z \in [ - 2 , 2 ] ^ { d }$ , since $2 \beta \ge \operatorname* { m a x } \{ ( \eta - 4 \beta ) ^ { 2 } , ( \eta + 4 \beta ) ^ { 2 } \}$ , we have $\nabla ^ { 2 } H _ { t } ( z )$ is negative semidefinite. Therefore $H _ { t }$ is concave over $[ - 2 , 2 ] ^ { d }$

Let us analyze $z = - \pmb { \xi } _ { t }$ . By definition, we have $H _ { t } ( - \pmb { \xi } _ { t } ) = 0$ and $\pmb q ( - \pmb \xi _ { t } ) = \pmb p _ { t }$ . Then the gradient $\nabla H _ { t } ( - \pmb { \xi } _ { t } )$ satisfies

$$
\begin{array} { r l } & { \nabla H _ { t } ( - \pmb { \xi } _ { t } ) = \eta \pmb { x } _ { t } + \pmb { p } _ { t } \odot ( - \eta + 2 \beta \pmb { \xi } _ { t } ) } \\ & { \qquad = \eta \pmb { p } _ { t } \odot ( 1 - \pmb { \xi } _ { t } / 2 ) + \pmb { p } _ { t } \odot ( - \eta + 2 \beta \pmb { \xi } _ { t } ) } \\ & { \qquad = ( 2 \beta - \eta / 2 ) \pmb { p } _ { t } \odot \pmb { \xi } _ { t } } \\ & { \qquad = \pmb { 0 } . } \end{array}
$$

Thus $- \pmb { \xi } _ { t }$ is the maximizer of $H _ { t }$ over $[ - 2 , 2 ] ^ { d }$ since $H _ { t }$ is concave. Then we have $H _ { t } ( \pmb { \xi } _ { t + 1 } ) \ \leq$ $H _ { t } ( - \pmb { \xi } _ { t } ) = 0$ . This completes the proof. □

By Lemma 2, we conclude that $\Phi _ { t + 1 } \leq \Phi _ { t }$ for every $t \geq 1$

Proof of Theorem 1. The choices $\eta = 1 / 8$ and $\beta = 1 / 3 2$ in Algorithm 1 satisfy the conditions in Lemma 1 and Theorem 5. Fix any $T \geq 1$ . By Theorem $5 ,$ we have $\Phi _ { T + 1 } \leq \Phi _ { 1 } = \log d .$ Fix any pure action $i \in [ d ]$ . By setting $\theta = c _ { T , i }$ in (2), we have $\Phi _ { T + 1 } \geq \eta R _ { T , i }$ . Therefore $R _ { T , i } \leq \log d / \eta = 8$ log d for every i. Thus the alternating regret is at most 8 log d for any $T \geq 1$ □

## 2.3 An Ω(log d) lower bound

We construct an adaptive adversary such that any algorithm sufers $\Omega ( \log d )$ alternating regret. Together with Theorem 1, we show that the minimax-optimal alternating regret for OLO over the simplex is $\Theta ( \log d )$ . In this lower bound construction, the adversary adaptively chooses the loss function so that the alternating regret in each round $t \in [ d - 1 ]$ is at least $2 / ( d + 1 - t )$ . This then implies the $\Omega ( \log d )$ bound.

Theorem 6 (Logarithmic dependence is necessary). For every $d \geq 2$ , every horizon $T \geq d - 1$ , and every online algorithm, an adaptive adversary can choose linear costs in $[ - 1 , 1 ] ^ { d }$ such that

$$
\mathrm { R e g } _ { \mathrm { A l t } } ^ { T } \geq 2 \left( \sum _ { m = 1 } ^ { d } { \frac { 1 } { m } } - 1 \right) = 2 \log d - O ( 1 ) .
$$

Proof. Define $S _ { 0 } = [ d ]$ . In round $1 \leq t \leq d - 1$ , after observing $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ , the adversary computes $j _ { t } \in \arg \operatorname* { m a x } _ { j \in S _ { t - 1 } } x _ { t , j }$ , update $S _ { t } : = S _ { t - 1 } \setminus \{ j _ { t } \}$ , and sets the loss $c _ { t }$

$$
c _ { t , i } = \left\{ { \begin{array} { l l } { - 1 , } & { i \in S _ { t } , } \\ { + 1 , } & { i \notin S _ { t } . } \end{array} } \right.
$$

The set $S _ { d - 1 }$ contains one action, denoted by $i ^ { \star }$ . For $t \geq d ,$ the adversary always chooses $c _ { t } = c _ { d - 1 }$ $\mathrm { A t } \ t = 1$ , we have ${ \mathfrak { c } } _ { 1 } ^ { \top } { \pmb { x } } _ { 1 } - { \mathfrak { c } } _ { 1 , i ^ { \star } } = 2 { \mathfrak { x } } _ { 1 , j _ { 1 } } \geq 2 / d$ . For $2 \leq t \leq d - 1$ , the efective cost $\pmb { g } _ { t } : = \pmb { c } _ { t - 1 } + \pmb { c } _ { t }$ is

$$
g _ { t , i } = \left\{ { \begin{array} { l l } { - 2 , } & { i \in S _ { t } , } \\ { 0 , } & { i = j _ { t } , } \\ { + 2 , } & { i \not \in S _ { t - 1 } . } \end{array} } \right.
$$

Thus the learner’s one-round alternating regret relative to $i ^ { \star }$ is

$$
g _ { t } ^ { \top } x _ { t } - g _ { t , i ^ { \star } } = 2 x _ { t , j _ { t } } + 4 \sum _ { i \notin S _ { t - 1 } } x _ { t , i } \geq 2 ( 1 - \sum _ { i \notin S _ { t - 1 } } x _ { t , i } ) / ( d + 1 - t ) + 4 \sum _ { i \notin S _ { t - 1 } } x _ { t , i } \geq \frac { 2 } { d + 1 - t } ,
$$

where the first inequality is because $\begin{array} { r } { x _ { t , j _ { t } } = \operatorname* { m a x } _ { j \in S _ { t - 1 } } x _ { t , j } \geq ( 1 - \sum _ { i \notin S _ { t - 1 } } x _ { t , i } ) / | S _ { t - 1 } | = ( 1 - \sum _ { i \notin S _ { t - 1 } } x _ { t , i } ) / | S _ { t - 1 } | = ( 1 - \sum _ { i \notin S _ { t - 1 } } x _ { t , i } ) / ( 1 - \sum _ { i \notin S _ { t - 1 } } \phi _ { i } ) . } \end{array}$ $\textstyle \sum _ { i \not \in S _ { t - 1 } } x _ { t , i } \big ) / ( d + 1 - t )$ . Therefore, we have

$$
\mathrm { R e g } _ { \mathrm { A l t } } ^ { d - 1 } \geq \sum _ { t = 1 } ^ { d - 1 } ( g _ { t } ^ { \top } x _ { t } - g _ { t , i ^ { \star } } ) \geq \sum _ { t = 1 } ^ { d - 1 } \frac 2 { d + 1 - t } = 2 \left( \sum _ { m = 1 } ^ { d } \frac 1 m - 1 \right) = 2 \log d - O ( 1 ) .
$$

Moreover, since $c _ { t } = c _ { d - 1 }$ for all $t \geq d ,$ , we have $\mathrm { R e g } _ { \mathrm { A l t } } ^ { t } \geq \mathrm { R e g } _ { \mathrm { A l t } } ^ { d - 1 } \geq 2 \log d - O ( 1 )$

## 3 Alternating regret for online convex optimization

In this section, we extend our result to the more general setting of online convex optimization.

Online convex optimization. We first introduce some notations. Let $\mathcal { X } \subseteq \mathbb { R } ^ { d }$ be a nonempty compact convex set. We denote by $\Delta x$ the set of Borel probability measures over X . For $\rho , \mu \in \Delta _ { \mathcal { X } }$ we say that $\mu$ dominates $\rho$ if $\rho$ is absolutely continuous with respect to $\mu , \mathrm { i . e . , } \rho ( A ) = 0$ for every measurable set A such that $\mu ( A ) = 0$ . We write $\rho \ll \mu$ if $\mu$ dominates $\rho .$

At round $t \geq 1$ , the learner chooses $\mathbf { \boldsymbol { x } } _ { t } \in \mathcal { X }$ using $f _ { 1 } , \ldots , f _ { t - 1 }$ , and then the adversary adaptively chooses a continuous convex loss function $f _ { t } : \mathcal { X } \to [ - 1 , 1 ]$ . The alternating regret to $\mathbf { \boldsymbol { u } } \in \mathcal { X }$ is

$$
\mathrm { R e g } _ { \mathrm { A l t } } ^ { T } ( u ) : = \sum _ { t = 1 } ^ { T } \bigl ( f _ { t - 1 } ( x _ { t } ) + f _ { t } ( x _ { t } ) \bigr ) - \sum _ { t = 1 } ^ { T } \bigl ( f _ { t - 1 } ( u ) + f _ { t } ( u ) \bigr ) = \sum _ { t = 1 } ^ { T } \bigl ( g _ { t } ( x _ { t } ) - g _ { t } ( u ) \bigr ) ,\tag{3}
$$

where we define $f _ { 0 } \equiv 0$ and the efective loss function $g _ { t } : = f _ { t - 1 } + f _ { t }$ for $t \geq 1$ . For a distribution $\rho \in \Delta \chi$ , we write $\mathrm { R e g } _ { \mathrm { A l t } } ^ { T } ( \rho ) : = \mathbb { E } _ { \boldsymbol { u } \sim \rho } [ \mathrm { R e g } _ { \mathrm { A l t } } ^ { T } ( \boldsymbol { u } ) ]$ . We also use the shorthand $\mathbb { E } _ { \mu } [ f ] : = \mathbb { E } _ { x \sim \mu } [ f ( x ) ]$

## 3.1 The continuous alternation-aware Hedge algorithm

We propose the continuous alternation-aware Hedge algorithm (Algorithm 2), a continuous generalization of Algorithm 1. Fix a reference distribution $\mu \in \Delta _ { \mathcal { X } }$ , and define $L _ { 0 } \equiv 0$ and $\begin{array} { r } { L _ { t } ( { \boldsymbol { \mathbf { \mathit { u } } } } ) : = \sum _ { s = 1 } ^ { t } g _ { s } ( { \boldsymbol { \mathbf { \mathit { u } } } } ) } \end{array}$ The algorithm replaces the finite weight vector $\mathbf { \nabla } _ { \mathbf { \mathcal { P } } _ { t } }$ by a probability measure $P _ { t }$ , corrects it to a

probability measure $Q _ { t } ,$ and plays the barycenter of $Q _ { t }$ . When $\mathcal { X } = \Delta _ { d } , \mu$ is uniform over the pure actions, and the losses are linear, the algorithm reduces to Algorithm 1.

As in Algorithm 1, we first show that the algorithm is well defined and feasible. The proof is deferred to Section A.

Lemma 3. Assume $\beta < 1 / 2$ . In Algorithm ${ \mathit { 2 } } , \theta _ { t }$ is unique and satisfies $\theta _ { t } = \mathbb { E } _ { P _ { t } } [ f _ { t - 1 } ]$ . Moreover, $Q _ { t }$ is a probability measure and $\mathbf { \boldsymbol { x } } _ { t } \in \mathcal { X }$

Algorithm 2 Continuous alternation-aware Hedge (continuous AA-Hedge)   
1: Given action set X and the reference distribution $\mu \in \Delta _ { \mathcal { X } }$   
2: Set $f _ { 0 } \equiv 0 , L _ { 0 } \equiv 0 , \eta = 1 / 8 ,$ , and $\beta = 1 / 3 2 .$   
3: for $t = 1 , 2 , \dots$ do   
4: Compute $\theta _ { t }$ as the unique maximizer of   
lo $\begin{array} { r } { \mathrm { ~ \small ~ \displaystyle ~  ~ \int _ { \chi } \exp [ - \eta L _ { t - 1 } ( \pmb { u } ) - \beta ( f _ { t - 1 } ( \pmb { u } ) - \theta ) ^ { 2 } ] \mu ( \mathrm { d } \pmb { u } ) . } } \end{array}$   
5: Set $\xi _ { t } ( { \pmb u } ) = f _ { t - 1 } ( { \pmb u } ) - \theta _ { t }$ and form the probability measure   
$P _ { t } ( \mathrm { d } \pmb { u } ) \propto \mathrm { e x p } \left[ - \eta L _ { t - 1 } ( \pmb { u } ) - \beta \xi _ { t } ( \pmb { u } ) ^ { 2 } \right] \mu ( \mathrm { d } \pmb { u } )$   
6: Set $Q _ { t } ( \mathrm { d } \pmb { u } ) = \left( 1 - \xi _ { t } ( \pmb { u } ) / 2 \right) P _ { t } ( \mathrm { d } \pmb { u } )$   
7: Play the barycenter $\begin{array} { r } { \pmb { x } _ { t } = \int _ { \mathcal { X } } \pmb { u } Q _ { t } ( \mathrm { d } \pmb { u } ) } \end{array}$   
8: Observe $f _ { t }$ and update $L _ { t } \gets L _ { t - 1 } + f _ { t - 1 } + f _ { t } .$

## 3.2 Alternating regret bounds

We first prove an upper bound on the distributional alternating regret ${ \mathrm { R e g } } _ { \mathrm { A l t } } ^ { T } ( \rho )$ for any $\rho \ll \mu$ The alternating regret is at most $O ( \mathrm { K L } ( \rho \| \mu ) )$ where the KL divergence over measures is defined as $\begin{array} { r } { \mathrm { K L } ( \rho \| \mu ) : = \int _ { \mathcal { X } } \log \frac { \mathrm { d } \rho } { \mathrm { d } \mu } \mathrm { d } \rho . } \end{array}$ This is the continuous analogue of comparing with a pure action in Theorem 1. The proof is similar to Theorem 1 and is deferred to Section $\mathrm { A } .$

Theorem 7 (Distributional alternating-regret bound). For every horizon $T \geq 1$ and every $\rho \ll \mu$ Algorithm 2 guarantees

$$
\mathrm { R e g } _ { \mathrm { A l t } } ^ { T } ( \rho ) \leq 8 \mathrm { K L } ( \rho \| \mu ) + \frac { \mathrm { V a r } _ { \rho } ( f _ { T } ) } { 4 } \leq 8 \mathrm { K L } ( \rho \| \mu ) + \frac { 1 } { 4 } .\tag{4}
$$

Then we instantiate the results with $\mu$ being the uniform probability measure over $\mathcal { X } ^ { 2 }$ , to get alternating regret bound ma $\mathfrak { c } _ { \mathbf { \boldsymbol { u } } \in \mathcal { X } } \mathrm { R e g } _ { \mathrm { A l t } } ^ { T } ( \mathbf { \boldsymbol { u } } ) = O ( d \log T )$

Theorem 8 $( O ( d \log ( 1 + T / d ) )$ alternating regret for OCO). For every horizon $T \geq 1$ , Algorithm 2 with the uniform probability measure $\mu \in \Delta _ { \mathcal { X } }$ guarantees

$$
\operatorname* { m a x } _ { \boldsymbol { u } \in \mathcal { X } } \mathrm { R e g } _ { \mathrm { A l t } } ^ { T } ( \boldsymbol { u } ) \leq 8 d \left( 1 + \log _ { + } \frac { T } { 2 d } \right) + \frac { 1 } { 4 } ,\tag{5}
$$

where $\log _ { + } z : = \operatorname* { m a x } \{ 0 , \log z \}$ . Since max $\mathrm { \Sigma } _ { \mathsf { L } \in \mathcal { X } } \mathrm { R e g } _ { \mathrm { A l t } } ^ { T } ( \pmb { u } ) \leq 4 T$ trivially, we have

$$
\operatorname* { m a x } _ { u \in \mathcal { X } } \mathrm { R e g } _ { \mathrm { A l t } } ^ { T } ( u ) = O \left( \operatorname* { m i n } \left\{ T , d \left( 1 + \log _ { + } \frac { T } { 2 d } \right) \right\} \right) = O ( d \log ( 1 + T / d ) )
$$

Proof. Fix $\mathbf { \boldsymbol { u } } \in \mathcal { X }$ and $\delta \in ( 0 , 1 )$ . Let $\rho _ { \pmb { u } , \delta }$ be the uniform probability measure on the set $( 1 - \delta ) { \pmb u } + \delta { \pmb \chi }$ The afine ma $) z \mapsto ( 1 - \delta ) \pmb { u } + \delta z$ scales d-dimensional volume by $\delta ^ { d }$ , so $\mathrm { K L } ( \rho _ { \mathbf { \boldsymbol { u } } , \delta } | | \mu ) = d \log ( 1 / \delta )$ For any ${ \pmb v } = ( 1 - \delta ) { \pmb u } + \delta { z }$ with $z \in \mathcal { X }$ , the convexity of $f _ { t } \in [ - 1 , 1 ]$ gives

$$
f _ { t } ( \pmb { v } ) \leq ( 1 - \delta ) f _ { t } ( \pmb { u } ) + \delta f _ { t } ( \pmb { z } ) \leq f _ { t } ( \pmb { u } ) + 2 \delta .
$$

Thus, we have $\mathbb { E } _ { \rho _ { u , \delta } } [ f _ { t } ] - f _ { t } ( u ) \le 2 \delta$ for any $t \geq 1$

Combining the above with Theorem $^ { 7 , }$ we have

$$
\begin{array} { l } { \displaystyle \mathrm { R e g } _ { \mathrm { A l t } } ^ { T } ( u ) = \mathrm { R e g } _ { \mathrm { A l t } } ^ { T } ( \rho _ { u , \delta } ) + 2 \sum _ { t = 1 } ^ { T - 1 } \left( \mathbb { E } _ { \rho _ { u , \delta } } [ f _ { t } ] - f _ { t } ( u ) \right) + \left( \mathbb { E } _ { \rho _ { u , \delta } } [ f _ { T } ] - f _ { T } ( u ) \right) . } \\ { \displaystyle \qquad \leq \mathrm { 8 K L } ( \rho _ { u , \delta } \| \mu ) + \frac { 1 } { 4 } + 4 T \delta } \\ { \displaystyle \qquad \leq 8 d \log ( 1 / \delta ) + \frac { 1 } { 4 } + 4 T \delta . } \end{array}
$$

Choosing $\delta = \operatorname* { m i n } \{ 1 , 2 d / T \}$ proves the claim.

Moreover, for OLO over a polytope X with N vertices, Algorithm 2 with µ being the uniform distribution over these vertices has alternating regret ${ \cal O } ( \log N )$

Corollary 2 (Constant alternating regret for polyhedral OLO). Let $\mathcal { X } = \operatorname { c o n v } \{ \pmb { v } _ { 1 } , \hdots , \pmb { v } _ { N } \}$ be a polytope. Consider linear losses $f _ { t } ( \pmb { x } ) = \langle \pmb { c } _ { t } , \pmb { x } \rangle$ , with $f _ { t } ( \pmb { x } ) \in [ - 1 , 1 ]$ for every $\mathbf { \boldsymbol { x } } \in \mathcal { X }$ . Running Algorithm 2 with µ being the uniform distribution on $\{ \pmb { v } _ { 1 } , \hdots , \pmb { v } _ { N } \}$ gives

$$
\operatorname* { m a x } _ { \pmb { u } \in \mathcal { X } } \mathrm { R e g } _ { \mathrm { A l t } } ^ { T } ( \pmb { u } ) \leq 8 \log N .
$$

In particular, for $\mathcal { X } = \Delta _ { d }$ and $\pmb { c } _ { t } \in [ - 1 , 1 ] ^ { d } .$ , Algorithm 2 coincides with Algorithm 1 and recovers the bound 8 log d.

Proof. The cumulative comparator loss is linear, so it attains its minimum over the polytope at one of its vertices. Applying Theorem 7 with $\rho = \delta _ { \pmb { v } _ { j } }$ and note that $\mathrm { K L } ( \delta _ { \boldsymbol { v } _ { j } } \| \mu ) = \log N$ and $\mathrm { V a r } _ { \delta _ { v _ { j } } } ( f _ { T } ) = 0$ . Thus we have ma $\begin{array} { r } { \mathbf { x } _ { \pmb { u } \in \mathcal { X } } \operatorname { R e g } _ { \mathrm { A l t } } ^ { T } ( \pmb { u } ) = \operatorname* { m a x } _ { j \in [ N ] } \operatorname { R e g } _ { \mathrm { A l t } } ^ { T } ( \pmb { v } _ { j } ) \leq 8 \log \bar { N } } \end{array}$ □

## 4 Matching lower bounds for OCO

In this section, we prove a alternating regret lower bound of $\Omega ( d \log ( 1 + T / d ) )$ for OCO with bounded convex losses over a d-dimensional convex set $\mathcal { X } \subseteq B _ { d } ( 1 ) : = \{ \pmb { x } \in \mathbb { R } ^ { d } : \| \pmb { x } \| _ { 2 } \leq 1 \}$ . In Section 4.1, we first prove a $\Omega ( \log T )$ lower bound when the action set is the 2-dimensional unit ball $B _ { 2 } ( 1 )$ (Theorem 9). Then, in Section 4.2, we extend the 2-dimensional construction to d-dimensional convex sets and prove the $\Omega ( d \log ( 1 + T / d ) )$ lower bound.

## 4.1 An Ω(log T) lower bound for OCO over 2-dimensional ball

In this section, we show that the log T dependence in Theorem 8 is unavoidable. The lower bound already holds on the two-dimensional Euclidean unit ball $B _ { 2 } ( 1 ) : = \{ { \pmb x } \in \mathbb { R } ^ { 2 } : \| { \pmb x } \| _ { 2 } \leq 1 \}$ . Throughout this section, let $\mathrm { R e g } _ { \mathrm { A l t } } ^ { T } : = \operatorname* { m a x } _ { \boldsymbol { u } \in B _ { 2 } ( 1 ) } \mathrm { R e g } _ { \mathrm { A l t } } ^ { T } ( \boldsymbol { u } )$

Theorem 9 (Logarithmic lower bound for OCO). For every integer $T \geq 1$ and every possibly randomized online algorithm A, there is an sequence of continuous convex losses $f _ { 1 } , \dots , f _ { T } : B _ { 2 } ( 1 ) \to$ [0, 1] such that

$$
\mathbb { E } _ { A } [ \mathrm { R e g } _ { \mathrm { A l t } } ^ { T } ] \geq \frac { 1 } { 1 2 5 } \left\lfloor \log _ { 1 6 } T \right\rfloor .
$$

The expectation is only over the randomness of A.

Proof overview. We in fact prove a stronger lower bound:

$$
\mathbb { E } _ { A } [ \mathrm { R e g } _ { \mathrm { A l t } } ^ { T } ] \ge \mathbb { E } _ { A } \left[ \sum _ { t = 1 } ^ { T } f _ { t } ( x _ { t } ) - 2 \operatorname* { m i n } _ { u \in B _ { 2 } ( 1 ) } \sum _ { t = 1 } ^ { T } f _ { t } ( u ) \right] \ge \Omega ( \log T ) .
$$

The first inequality holds since the losses are non-negative. The factor 2 in the cumulative loss of the best fixed action is thus not essential. Our lower bound can be easily extended to the regret notion where we compete with $c \cdot \mathrm { m i n } _ { { \boldsymbol u } \in B _ { 2 } ( 1 ) } \sum _ { t = 1 } ^ { T } f _ { t } ( { \boldsymbol u } )$ for any constant $c > 0$

The construction of the loss sequence proceeds in $K = O ( \log T )$ blocks. Each block $k \in [ K ]$ has a center $\mathbf { \Delta } \mathbf { u } _ { k }$ on the boundary of $B _ { 2 } ( 1 )$ and an angular scale $\alpha _ { k }$ . Within block $k ,$ each round randomizes between a common loss, which is minimized at $\mathbf { \Delta } \mathbf { u } _ { k }$ , and a rare loss, which is largest at $\mathbf { \Delta } \mathbf { u } _ { k }$ but vanishes after a rotation by $\alpha _ { k }$ . Fixing a small constant $\nu ,$ we choose the rare-loss probability $O ( \alpha _ { k } ^ { 2 } )$ and the block length proportional to $O ( \nu / \alpha _ { k } ^ { 2 } )$ . This choice ensures that any algorithm incurs expected loss at least ν in every block (Lemma 4) and thus cumulative loss of at least $\Omega ( \nu K )$ (Lemma 5).

After a block, we rotate the center by $\alpha _ { k }$ if the block contains a rare loss, and otherwise leave it unchanged. We maintain geometrically decreasing $\alpha _ { k } = \sqrt { \nu } / 4 ^ { k - 1 }$ . We use the endpoint ${ \pmb u } _ { K + 1 }$ after all K rotations in these blocks as the hindsight comparator. We prove that ${ \pmb u } _ { K + 1 }$ has 0 loss on every realized rare, while incurring only $O ( \nu ^ { 2 } )$ expected common loss per block (Lemma 6). Thus the (expected) alternating regret of any algorithm is at least $\Omega ( \nu K - \nu ^ { 2 } K ) = \Omega ( K ) = \Omega ( \log T )$

One-round loss pair. Fix a unit vector $\pmb { u } \in \partial B _ { 2 } ( 1 )$ and an angle $0 < \alpha \leq 1$ . Define

$$
\begin{array} { c } { \displaystyle \ell _ { u } ( \pmb { x } ) : = \frac { 1 - \langle \pmb { u } , \pmb { x } \rangle } { 2 } , } \\ { h _ { u , \alpha } ( \pmb { x } ) : = \operatorname* { m a x } \left\{ 0 , \frac { \langle \pmb { u } , \pmb { x } \rangle - \cos \alpha } { 1 - \cos \alpha } \right\} , } \\ { p _ { \alpha } : = \displaystyle \frac { 1 - \cos \alpha } { 3 - \cos \alpha } . } \end{array}
$$

We refer to $\ell _ { u }$ as the common loss and $h _ { u , \alpha }$ as the rare loss. We will randomly choose the common loss $\ell _ { u }$ with probability $1 - p _ { \alpha }$ and $h _ { u , \alpha }$ with probability $p _ { \alpha }$ . We first present some useful properties of these loss functions.

Lemma 4 (Properties of one-round loss pair). The common loss $\ell _ { u }$ and the rare loss $h _ { \boldsymbol { u } , \alpha }$ are continuous and convex and take values in $[ 0 , 1 ]$ . Moreover, $h _ { \mathbf { } u , \alpha } ( \pmb { u } ) = 1$ , and $h _ { { \bf u } , \alpha } ( { \pmb v } ) = 0$ whenever $\langle { \pmb u } , { \pmb v } \rangle \leq \cos \alpha$ . For every $\pmb { x } \in B _ { 2 } ( 1 )$ , we have

$$
( 1 - p _ { \alpha } ) \ell _ { \pmb { u } } ( \pmb { x } ) + p _ { \alpha } h _ { \pmb { u } , \alpha } ( \pmb { x } ) \geq p _ { \alpha } .
$$

Finally, $\alpha ^ { 2 } / 9 \le p _ { \alpha } \le \alpha ^ { 2 } / 4$

Proof. The common loss $\ell _ { u }$ is afine, and the rare loss $h _ { u , \alpha }$ is the maximum of two afine functions. Both are therefore continuous and convex. Since $\langle { \pmb u } , { \pmb x } \rangle \in [ - 1 , 1 ]$ , both take values in [0, 1]. By definition, we can verify $h _ { \mathbf { } u , \alpha } ( \pmb { u } ) = 1$ , and $h _ { { \bf { u } } , \alpha } ( { \pmb v } ) = 0$ whenever $\langle { \pmb u } , { \pmb v } \rangle \leq \cos \alpha$

It remains to prove the two inequalities. The definition of $p _ { \alpha }$ gives

$$
{ \frac { 1 - p _ { \alpha } } { 2 } } = { \frac { p _ { \alpha } } { 1 - \cos \alpha } } = { \frac { 1 } { 3 - \cos \alpha } } .
$$

If $\langle { \pmb u } , { \pmb x } \rangle \leq \cos \alpha .$ , then the rare loss is zero and the common-loss term is at least $p _ { \alpha }$ . Otherwise, the two terms sum to

$$
\frac { 1 - \langle { \pmb u } , { \pmb x } \rangle } { 3 - \cos \alpha } + \frac { \langle { \pmb u } , { \pmb x } \rangle - \cos \alpha } { 3 - \cos \alpha } = p _ { \alpha } .
$$

Finally, for $0 < \alpha \leq 1$ , we have $\alpha ^ { 2 } / 3 \le 1 - \cos \alpha \le \alpha ^ { 2 } / 2$ and $2 \leq 3 - \cos \alpha \leq 3$ , which proves the bounds on $p _ { \alpha }$ □

Block construction. Fix a positive integer K and set $\nu : = 1 0 ^ { - 2 }$ . Start from ${ \pmb u } _ { 1 } = ( 1 , 0 )$ and, for $k = 1 , \ldots , K$ , set

$$
\alpha _ { k } : = \frac { \sqrt \nu } { 4 ^ { k - 1 } } , \quad p _ { k } : = p _ { \alpha _ { k } } , \quad n _ { k } : = \left\lceil \frac { \nu } { p _ { k } } \right\rceil .
$$

We construct the loss sequence as follows: for each block k with $n _ { k }$ rounds

1. We independently choose the rare loss $h _ { { \pmb u } _ { k } , \alpha _ { k } }$ with probability $p _ { k }$ on each of the $n _ { k }$ rounds, and choose the common loss $\ell _ { { \pmb u } _ { k } }$ otherwise.

2. If no rare loss was sampled in block $k ,$ set $\mathbf { \Delta } u _ { k + 1 } \gets \mathbf { \Delta } u _ { k }$ . Otherwise at least one rare loss was sampled in block k, we update ${ \pmb u } _ { k + 1 }$ by rotating $\mathbf { \Delta } \mathbf { u } _ { k }$ counterclockwise through $\alpha _ { k }$ . That is, if ${ \pmb u } _ { k } = ( \cos ( \phi ) , \sin ( \phi ) )$ , then update ${ \pmb u } _ { k + 1 }  ( \cos ( \phi + \alpha _ { k } ) , \sin ( \phi + \alpha _ { k } ) )$

Let $\begin{array} { r } { N _ { K } = \sum _ { k = 1 } ^ { K } n _ { k } } \end{array}$ . We note that this is an oblivious randomized adversary. We use $\mathbb { E } _ { \mathrm { l o s s } }$ for expectation over the construction and $\mathbb { E } _ { \mathrm { l o s s } , \mathcal { A } }$ for expectation over both the construction and the randomness of ${ \mathcal { A } } .$

Lemma 5 (Cost of the algorithm). For every online algorithm $\mathcal { A } _ { : }$

$$
\mathbb { E } _ { \mathrm { l o s s } , A } \left[ \sum _ { t = 1 } ^ { N _ { K } } f _ { t } ( \pmb { x } _ { t } ) \right] \geq K \nu .
$$

Proof. Consider a round in block k and condition on the past and on the private randomness used to choose $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ . This fixes $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ and $\mathbf { \Delta } \mathbf { u } _ { k }$ , while the current common-or-rare choice remains independent. By Lemma 4, the conditional expected loss is at least $p _ { k }$ . Summing over the blocks gives

$$
\mathbb { E } _ { \mathrm { l o s s } , A } \left[ \sum _ { t = 1 } ^ { N _ { K } } f _ { t } ( \pmb { x } _ { t } ) \right] \geq \sum _ { k = 1 } ^ { K } n _ { k } p _ { k } \geq K \nu .
$$

Lemma 6 (Cost of the endpoint). The endpoint of the block construction $\pmb { u } _ { \star } : = \pmb { u } _ { K + 1 }$ satisfies

$$
\mathbb { E } _ { \mathrm { l o s s } } \left[ \sum _ { t = 1 } ^ { N _ { K } } f _ { t } ( \pmb { u } _ { \star } ) \right] \leq 1 0 K \nu ^ { 2 } .
$$

Proof. Let $I _ { j }$ indicate whether block $j$ contains a rare loss. By the union bound and the definition of $n _ { j }$ ,

$$
\operatorname* { P r } ( I _ { j } = 1 ) = 1 - ( 1 - p _ { j } ) ^ { n _ { j } } \leq n _ { j } p _ { j } \leq \nu + p _ { j } < 2 \nu ,
$$

where the last inequality uses $p _ { j } \leq \alpha _ { j } ^ { 2 } / 4 \leq \nu / 4$ by Lemma 4 and the definition $\alpha _ { j } = \sqrt { \nu } / 4 ^ { j - 1 }$

Define counterclockwise angle from $\mathbf { \Delta } \mathbf { u } _ { k }$ to $\pmb { u } _ { \star }$ as $\begin{array} { r } { \Delta _ { k } : = \sum _ { j = k } ^ { K } I _ { j } \alpha _ { j } } \end{array}$ . By definition of $\alpha _ { k }$ , we have

$$
0 \leq \Delta _ { k } \leq \sum _ { j = k } ^ { \infty } \alpha _ { j } = \frac { 4 \alpha _ { k } } { 3 } < \pi .
$$

If block k contains a rare loss, then $\Delta _ { k } \ge \alpha _ { k }$ . Thus $\langle { \pmb u } _ { k } , { \pmb u } _ { \star } \rangle = \cos \Delta _ { k } \leq$ cos $\alpha _ { k }$ and the rare loss $h _ { { \pmb u } _ { k } , \alpha _ { k } } ( { \pmb u } _ { \star } ) = 0$ is zero (Lemma 4). Each common loss in the block is at most $\begin{array} { r } { \ell _ { { \boldsymbol u } _ { k } } ( { \boldsymbol u } _ { \star } ) = \frac { 1 - \cos \Delta _ { k } } { 2 } \le \frac { \Delta _ { k } ^ { 2 } } { 4 } } \end{array}$ Since $\Delta _ { k } \leq 4 \alpha _ { k } / 3$ , we have

$$
\mathbb { E } _ { \mathrm { l o s s } } [ \Delta _ { k } ^ { 2 } ] \le \frac { 4 \alpha _ { k } } { 3 } \mathbb { E } _ { \mathrm { l o s s } } [ \Delta _ { k } ] = \frac { 4 \alpha _ { k } } { 3 } \sum _ { j = k } ^ { K } \alpha _ { j } \operatorname* { P r } ( I _ { j } = 1 ) < \frac { 4 \alpha _ { k } } { 3 } \cdot 2 \nu \cdot \frac { 4 \alpha _ { k } } { 3 } < 4 \nu \alpha _ { k } ^ { 2 } .
$$

Then by $p _ { k } \ge \alpha _ { k } ^ { 2 } / 9$ (Lemma 4) and $\alpha _ { k } ^ { 2 } \le \nu ,$ the expected loss of $\pmb { u } _ { \star }$ in block k is at most

$$
n _ { k } \mathbb { E } _ { \mathrm { l o s s } } [ \ell _ { u _ { k } } ( u _ { \star } ) ] \leq \frac { n _ { k } } { 4 } \mathbb { E } _ { \mathrm { l o s s } } [ \Delta _ { k } ^ { 2 } ] \leq n _ { k } \nu \alpha _ { k } ^ { 2 } \leq \nu \left( \frac { \nu } { p _ { k } } + 1 \right) \alpha _ { k } ^ { 2 } \leq \nu ( 9 \nu + \alpha _ { k } ^ { 2 } ) \leq 1 0 \nu ^ { 2 } .
$$

Summing over the blocks proves the claim.

Proof of Theorem 9. We assume $K : = \lfloor \log _ { 1 6 } T \rfloor \geq 1$ since otherwise the claim is trivial. We use the loss construction described above. Since $\alpha _ { k } ^ { 2 } = \nu / 1 6 ^ { k - 1 }$ , Lemma 4 gives $n _ { k } \le \nu / p _ { k } + 1 \le 9 \nu / \alpha _ { k } ^ { 2 } + 1 \le$ $9 \cdot 1 6 ^ { k - 1 } + 1$ . Consequently,

$$
N _ { K } = \sum _ { k = 1 } ^ { K } n _ { k } \leq { \frac { 9 ( 1 6 ^ { K } - 1 ) } { 1 5 } } + K \leq 1 6 ^ { K } \leq T .
$$

We pad zero losses for rounds $N _ { K } < t \leq T$

For every realization of the loss construction and of the algorithm’s randomness, we evaluate alternating regret at $\pmb { u } _ { \star }$ . Since all losses are nonnegative and each comparator loss appears at most twice, we have

$$
\mathrm { R e g } _ { \mathrm { A l t } } ^ { T } \geq \sum _ { t = 1 } ^ { N _ { K } } f _ { t } ( { \pmb x } _ { t } ) - 2 \sum _ { t = 1 } ^ { N _ { K } } f _ { t } ( { \pmb u } _ { \star } ) .
$$

Applying Lemmas 5 and 6 and using $\nu = 1 0 ^ { - 2 }$ gives

$$
\mathbb { E } _ { \mathrm { l o s s } } \left[ \mathbb { E } _ { \boldsymbol { A } } \left[ \mathrm { R e g } _ { \mathrm { A l t } } ^ { T } \mid f _ { 1 } , \dots , f _ { T } \right] \right] = \mathbb { E } _ { \mathrm { l o s s } , \boldsymbol { A } } [ \mathrm { R e g } _ { \mathrm { A l t } } ^ { T } ] \geq K \nu - 2 0 K \nu ^ { 2 } = \frac { K } { 1 2 5 } .
$$

Thus some fixed realization of the loss sequence satisfies the desired bound.

## 4.2 An $\Omega ( d \log ( 1 + T / d ) )$ lower bound for OCO

Theorem 9 shows that the log T factor in Theorem 8 is unavoidable even for 2-dimensional unit ball. We now show that the linear dependence on d is also necessary in the worst case over a convex set $\mathcal { X } \subseteq B _ { d } ( 1 )$ and prove the optimal lower bound of $\Omega ( d \log ( 1 + T / d ) )$ .

High-level idea: Roughly speaking, we construct the losses in phases, where in each phase we simulate the lower bound construction in Theorem 9 on 2 coordinates. There are $d / 2$ phases and each phase contains $m = \Theta ( T / d )$ rounds. By Theorem 9, the alternating regret in each phase is at least $\Omega ( \log ( 1 + T / d ) )$ . Thus the total alternating regret is at least $\Omega ( d \log ( 1 + T / d ) )$ .

Let $r : = 1 / \sqrt { d }$ . Define

$$
\mathcal { X } : = \underbrace { r B _ { 2 } ( 1 ) \times \cdot \cdot \cdot \times r B _ { 2 } ( 1 ) } _ { \lfloor d / 2 \rfloor \mathrm { ~ t i m e s ~ } } \times [ - r , r ] ^ { d - 2 \lfloor d / 2 \rfloor } \ \subseteq \ \mathbb { R } ^ { d } ,
$$

where $d - 2 \lfloor d / 2 \rfloor \in \{ 0 , 1 \}$ . The set $\mathcal { X }$ is compact, convex, and full-dimensional. Moreover, every $\mathbf { \boldsymbol { x } } \in \mathcal { X }$ satisfies

$$
\| { \pmb x } \| _ { 2 } ^ { 2 } \leq \lfloor d / 2 \rfloor r ^ { 2 } + ( d - 2 \lfloor d / 2 \rfloor ) r ^ { 2 } = ( d - \lfloor d / 2 \rfloor ) r ^ { 2 } \leq 1 ,
$$

so $\mathcal { X } \subseteq B _ { d } ( 1 )$ . For $j \in [ \vert d / 2 \vert ]$ , let $\pi _ { j } : \mathcal { X } \to B _ { 2 } ( 1 )$ be the projection onto the j-th two-dimensional block, rescaled by $1 / r$ . Thus $\pi _ { j }$ is linear and $\pi _ { j } ( \chi ) = B _ { 2 } ( 1 )$ . Throughout this subsection, let $\mathrm { R e g } _ { \mathrm { A l t } } ^ { T } : = \operatorname* { m a x } _ { \boldsymbol { u } \in \mathcal { X } } \mathrm { R e g } _ { \mathrm { A l t } } ^ { T } ( \boldsymbol { u } )$

Theorem 10 (Linear dimension dependence is necessary). For every d $\geq 2$ , every horizon $T \geq 1 6$ and every possibly randomized online algorithm ${ \mathcal { A } } ,$ there is a sequence $o f$ continuous convex losses $f _ { 1 } , \dots , f _ { T } : \mathcal { X } \to [ 0 , 1 ]$ such that

$$
\mathbb { E } _ { A } \left[ \mathrm { R e g } _ { \mathrm { A l t } } ^ { T } \right] \ \geq \ \frac { m K } { 1 2 5 } \quad w h e r e \quad m : = \operatorname* { m i n } \left\{ \left\lfloor \frac { d } { 2 } \right\rfloor , \left\lfloor \frac { T } { 1 6 } \right\rfloor \right\} , \quad K : = \left\lfloor \log _ { 1 6 } \left\lfloor \frac { T } { m } \right\rfloor \right\rfloor .
$$

In particular $\mathbb { E } _ { A } [ \mathrm { R e g } _ { \mathrm { A l t } } ^ { T } ] = \Omega ( d \log ( 1 + T / d ) )$

Proof. Set $T ^ { \prime } : = \lfloor T / m \rfloor$ . Since $m \le \lfloor T / 1 6 \rfloor$ , we have $T ^ { \prime } \geq 1 6$ and $K = \lfloor \log _ { 1 6 } T ^ { \prime } \rfloor \ge 1$ . Since $m \leq \lfloor d / 2 \rfloor$ , we use the first $m T ^ { \prime }$ rounds as m consecutive phases of $T ^ { \prime }$ rounds, one for each of the first m two-dimensional blocks.

We construct the losses in these phases successively. For $j \in [ m ]$ , suppose the losses in the first $j - 1$ phases have been fixed. This fixed prefix induces a distribution over $A \mathrm { { } s }$ internal state. Starting from this state, simulate $\mathcal { A }$ for $T ^ { \prime }$ more rounds: return its decisions projected by $\pi _ { j }$ , and feed each two-dimensional loss back to $\mathcal { A }$ after lifting it through $\pi _ { j }$ . This defines a possibly randomized online algorithm on $B _ { 2 } ( 1 )$ . Inspecting the proof of Theorem 9, a fixed sequence of continuous convex losses $\tilde { f } _ { 1 } ^ { ( j ) } , \ldots , \tilde { f } _ { T ^ { \prime } } ^ { ( j ) } : B _ { 2 } ( 1 )  [ 0 , 1 ]$ can be chosen for this projected algorithm. Let $u _ { \star } ^ { ( j ) } \in B _ { 2 } ( 1 )$ minimize $\textstyle \sum _ { i = 1 } ^ { T ^ { \prime } } { \tilde { f } } _ { i } ^ { ( j ) } ( { \boldsymbol { \mathbf { u } } } )$ . Then

$$
\mathbb { E } _ { \boldsymbol { A } } \left[ \sum _ { i = 1 } ^ { T ^ { \prime } } \tilde { f } _ { i } ^ { ( j ) } \big ( \pi _ { j } \big ( \pmb { x } _ { ( j - 1 ) T ^ { \prime } + i } \big ) \big ) - 2 \sum _ { i = 1 } ^ { T ^ { \prime } } \tilde { f } _ { i } ^ { ( j ) } ( \pmb { u } _ { \star } ^ { ( j ) } ) \right] \geq \frac { K } { 1 2 5 } .
$$

For $t = ( j - 1 ) T ^ { \prime } + i$ , set $f _ { t } : = \tilde { f } _ { i } ^ { ( j ) } \circ \pi _ { j }$ . After all m phases have been fixed, set $f _ { t } \equiv 0$ for m $T ^ { \prime } < t \leq T$ . Each phase is fixed for the randomized algorithm induced by the preceding phases, not for a realization of $\mathbfcal { A }$ s random choices. Thus the resulting loss sequence is deterministic and oblivious. Since $\pi _ { j }$ is linear, every $f _ { t }$ is continuous, convex, and takes values in $[ 0 , 1 ]$

Let $\ b u _ { \star } \in { \mathcal { X } }$ have j-th two-dimensional block $r { \pmb u } _ { \star } ^ { ( j ) }$ for $j \in [ m ]$ and all remaining coordinates equal to zero. Then $\pi _ { j } ( \pmb { u } _ { \star } ) = \pmb { u } _ { \star } ^ { ( j ) }$ for every $j \in [ m ]$ . Since the losses are nonnegative, we may discard the terms $f _ { t - 1 } ( { \pmb x } _ { t } )$ from the learner’s loss, while every comparator loss appears at most twice. Therefore,

$$
\mathrm { R e g } _ { \mathrm { A l t } } ^ { T } \geq \sum _ { t = 1 } ^ { T } f _ { t } ( x _ { t } ) - 2 \sum _ { t = 1 } ^ { T } f _ { t } ( u _ { \star } ) = \sum _ { j = 1 } ^ { m } \left[ \sum _ { i = 1 } ^ { T ^ { \prime } } \tilde { f } _ { i } ^ { ( j ) } \big ( \pi _ { j } \big ( x _ { ( j - 1 ) T ^ { \prime } + i } \big ) \big ) - 2 \sum _ { i = 1 } ^ { T ^ { \prime } } \tilde { f } _ { i } ^ { ( j ) } ( u _ { \star } ^ { ( j ) } ) \right] .
$$

Taking expectation and summing the phase guarantees gives

$$
\mathbb { E } _ { A } [ \mathrm { R e g } _ { \mathrm { A l t } } ^ { T } ] \geq \frac { m K } { 1 2 5 } .
$$

It remains to verify the asymptotic bound. If $T \geq 8 d .$ , then $m = \lfloor d / 2 \rfloor \geq d / 3$ and $T ^ { \prime } =$ $\lfloor T / m \rfloor \ge T / ( 2 m ) \ge T / d .$ . Since $T ^ { \prime } \geq 1 6$ , we have $K = \Omega ( \log ( 1 + T ^ { \prime } ) ) = \Omega ( \log ( 1 + T / d ) )$ . Hence m $K = \Omega ( d \log ( 1 + T / d ) )$ . If $T < 8 d .$ , then $m = \lfloor T / 1 6 \rfloor \geq T / 3 2$ and $K \geq 1$ . Since log $( 1 + x ) \leq x$ we have m $K \geq T / 3 2 \geq d \log \left( 1 + T / d \right) / 3 2$ . Thus we conclude $\mathbb { E } _ { A } [ \mathrm { R e g } _ { \mathrm { A l t } } ^ { T } ] = \Omega ( d \log ( 1 + T / d ) )$ .

Combining Theorem 10 with Theorem 8 pins down the worst-case alternating regret over d-dimensional bodies in every regime of T and d.

Corollary 3 (Minimax-optimal alternating rerget for OCO). Let $\begin{array} { r } { \mathcal { R } ^ { \star } ( d , T ) : = \operatorname* { s u p } _ { \chi } \operatorname* { i n f } _ { \boldsymbol { \mathcal { A } } } \operatorname* { s u p } _ { f _ { 1 } , \dots , f _ { T } } \operatorname { R e g } _ { \mathrm { A l t } } ^ { T } } \end{array}$ where the outer supremum is over compact convex $\mathcal { X } \subseteq \mathbb { R } ^ { d }$ and the losses are convex with values in $[ - 1 , 1 ]$ . Then, for all $d \geq 2$ and $T \geq 1 6$ ，

$$
\mathcal { R } ^ { \star } ( d , T ) = \Theta \left( d \log \left( 1 + \frac { T } { d } \right) \right) .
$$

## 5 Conclusion

In this paper, we settle the minimax-optimal alternating regret for both OLO and OCO. For OLO over the simplex $\Delta _ { d } .$ we give the alternation-aware Hedge (AA-Hedge) algorithm with optimal $O ( \log d )$ regret. For OCO, we give the continuous AA-Hedge algorithm with $O ( d \log ( 1 + T / d ) )$ regret and proves a $\Omega ( d \log ( 1 + T / d ) )$ lower bound. Our results imply the first alternating learning dynamics with $O ( 1 / T )$ convergence to CCE in two-player general-sum games.

## Acknowledgement

The proofs were first obtained with the assistance of ChatGPT 5.6 Sol and Claude Opus 5. The authors subsequently verified and significantly rewrote the proofs for better exposition.

## References

I. Anagnostides, C. Daskalakis, G. Farina, M. Fishelson, N. Golowich, and T. Sandholm. Near-optimal no-regret learning for correlated equilibria in multi-player general-sum games. In Proceedings of the 54th Annual ACM SIGACT Symposium on Theory of Computing (STOC), 2022a.

I. Anagnostides, G. Farina, C. Kroer, C.-W. Lee, H. Luo, and T. Sandholm. Uncoupled learning dynamics with o(log t) swap regret in multiplayer games. In Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS), 2022b.

J. P. Bailey, G. Gidel, and G. Piliouras. Finite regret and cycles with fixed step-size via alternating gradient descent-ascent. In Conference on Learning Theory, pages 391–407, 2020.

M. Bowling, N. Burch, M. Johanson, and O. Tammelin. Heads-up limit hold’em poker is solved. Science, 347(6218):145–149, 2015.

N. Brown and T. Sandholm. Superhuman AI for heads-up no-limit poker: Libratus beats top professionals. Science, 359(6374):418–424, 2018.

N. Brown and T. Sandholm. Superhuman AI for multiplayer poker. Science, 365(6456):885–890, 2019.

Y. Cai and W. Zheng. Doubly optimal no-regret learning in monotone games. In International Conference on Ma- chine Learning (ICML), pages 3507–3524, 2023.

Y. Cai, H. Luo, C.-Y. Wei, and W. Zheng. Near-optimal policy optimization for correlated equilibrium in general-sum markov games. In International Conference on Artificial Intelligence and Statistics, pages 3889–3897, 2024.

N. Cesa-Bianchi and G. Lugosi. Prediction, Learning, and Games. Cambridge university press, 2006.

V. Cevher, A. Cutkosky, A. Kavis, G. Piliouras, S. Skoulakis, and L. Viano. Alternation makes the adversary weaker in two-player games. In Advances in Neural Information Processing Systems, volume 36, 2023. doi: 10.52202/075280-0804. URL https://proceedings.neurips.cc/paper\_ files/paper/2023/hash/3acb49252187efa352a1ae0e4b066ced-Abstract-Conference.html.

X. Chen and B. Peng. Hedging in games: Faster convergence of external and swap regrets. In Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS), 2020.

C. Daskalakis, A. Deckelbaum, and A. Kim. Near-optimal no-regret algorithms for zero-sum games. In Proceedings of the twenty-second annual ACM-SIAM symposium on Discrete Algorithms, pages 235–254. SIAM, 2011.

G. Farina, I. Anagnostides, H. Luo, C.-W. Lee, C. Kroer, and T. Sandholm. Near-optimal no-regret learning dynamics for general convex games. In Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS), 2022.

Y. Feng, K. Fujii, S. Skoulakis, X. Wang, and V. Cevher. Continuous-time analysis of heavy ball momentum in min-max games. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=7vdg7VICsX.

D. P. Foster and R. V. Vohra. Calibrated learning and correlated equilibrium. Games and Economic Behavior, 21(589):40–55, 1997.

Y. Freund and R. E. Schapire. A decision-theoretic generalization of on-line learning and an application to boosting. Journal of computer and system sciences, 55(1):119–139, 1997.

Y. Freund and R. E. Schapire. Adaptive game playing using multiplicative weights. Games and Economic Behavior, 29(1-2):79–103, 1999.

S. Hait, P. Li, H. Luo, and M. Zhang. Alternating regret for online convex optimization. In Proceedings of the Thirty-Eighth Conference on Learning Theory, volume 291 of Proceedings of Machine Learning Research, pages 2632–2633. PMLR, 2025. URL https://proceedings.mlr. press/v291/hait25a.html.

S. Hart and A. Mas-Colell. A simple adaptive procedure leading to correlated equilibrium. Econometrica, 68(5):1127–1150, 2000.

E. Hazan. Introduction to online convex optimization. Foundations and Trends® in Optimization, 2(3-4):157–325, 2016.

J. Lee, H. Cho, and C. Yun. Fundamental benefit of alternating updates in minimax optimization. In Forty-first International Conference on Machine Learning, 2024. URL https://openreview. net/forum?id=s6ZAT8MLKU.

T. Nan, S. Das Gupta, G. Iyengar, and C. Kroer. On the o(1/t) convergence of alternating gradient descent–ascent in bilinear games. In International Conference on Learning Representations, volume 2026, pages 79281–79320, 2026.

H. Narayanan and A. Rakhlin. Random walk approach to regret minimization. Advances in Neural Information Processing Systems, 23, 2010.

F. Orabona. A modern introduction to online learning. arXiv preprint arXiv:1912.13213, 2019.

A. Rakhlin and K. Sridharan. Online learning with predictable sequences. In Proceedings of the 26th Annual Conference on Learning Theory, volume 30 of Proceedings of Machine Learning Research, pages 993–1019. PMLR, 2013. URL https://proceedings.mlr.press/v30/Rakhlin13.html.

H. Shugart and J. M. Altschuler. Negative stepsizes make gradient-descent-ascent converge. arXiv preprint arXiv:2505.01423, 2025.

A. Soleymani, G. Piliouras, and G. Farina. Cautious optimism: A meta-algorithm for nearconstant regret in general games. In Proceedings of the 26th ACM Conference on Economics and Computation, pages 870–870, 2025a.

A. Soleymani, G. Piliouras, and G. Farina. Faster rates for no-regret learning in general games via cautious optimism. In Proceedings of the 57th Annual ACM Symposium on Theory of Computing, pages 518–529, 2025b.

V. Syrgkanis, A. Agarwal, H. Luo, and R. E. Schapire. Fast convergence of regularized learning in games. In Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS), 2015.

O. Tammelin. Solving large imperfect information games using cfr+. arXiv preprint arXiv:1407.5042, 2014.

T. Tsuchiya. Sublogarithmic swap regret in multiplayer general-sum games via hybrid regularization. arXiv preprint arXiv:2608.04149, 2026.

A. Wibisono, M. Tao, and G. Piliouras. Alternating mirror descent for constrained min– max games. In Advances in Neural Information Processing Systems, volume 35, 2022. doi: 10.52202/068431-2551. URL https://proceedings.neurips.cc/paper\_files/paper/2022/ hash/e496e0ce207ba9cdcc7d79bd499db67e-Abstract-Conference.html.

M. Xu, B. Jiang, Y.-F. Liu, and A. M.-C. So. A riemannian alternating descent ascent algorithmic framework for nonconvex-linear minimax problems on riemannian manifolds. Mathematics of Operations Research, 2026.

G. Zhang, Y. Wang, L. Lessard, and R. B. Grosse. Near-optimal local convergence of alternating gradient descent-ascent for minimax optimization. In International Conference on Artificial Intelligence and Statistics, pages 7659–7679. PMLR, 2022.

M. Zinkevich. Online convex programming and generalized infinitesimal gradient ascent. In Proceedings of the 20th international conference on machine learning (ICML), 2003.

## A Proofs for Online Convex Optimization

The proof follows the same strategy as in Section 2: we first show that the update is well defined and feasible, and then use a one-step inequality to prove that a potential function is nonincreasing.

## A.1 Existence and feasibility of the update

Proof of Lemma 3. Define

$$
F _ { t } ( \theta ) : = \log \int _ { \mathcal { X } } \exp \left[ - \eta L _ { t - 1 } ( \boldsymbol { u } ) - \beta ( f _ { t - 1 } ( \boldsymbol { u } ) - \theta ) ^ { 2 } \right] \mu ( \mathrm { d } \boldsymbol { u } ) ,
$$

and let $P _ { t , \theta }$ be the probability measure obtained by normalizing the integrand. $F _ { t }$ is diferentiable since it is logarithm of a bounded integrand over a compact set. We calculate its first-order and second-order derivative:

$$
\begin{array} { r l } & { F _ { t } ^ { \prime } ( \theta ) = 2 \beta \left( \mathbb { E } _ { P _ { t , \theta } } [ f _ { t - 1 } ] - \theta \right) , } \\ & { F _ { t } ^ { \prime \prime } ( \theta ) = 4 \beta ^ { 2 } \operatorname { V a r } _ { P _ { t , \theta } } ( f _ { t - 1 } ) - 2 \beta . } \end{array}
$$

Since $f _ { t - 1 } \in [ - 1 , 1 ]$ , the variance is at most 1. Since $\beta < 1 / 2$ , we have $F _ { t } ^ { \prime \prime } ( \theta ) \leq 4 \beta ^ { 2 } - 2 \beta < 0$ , so $F _ { t }$ is strictly concave. Moreover, $F _ { t } ( \theta ) \to - \infty { \mathrm { ~ a s ~ } } | \theta | \to \infty$ , so the maximizer exists and is unique. The first-order condition gives

$$
\theta _ { t } = \mathbb { E } _ { P _ { t } } [ f _ { t - 1 } ] \in \left[ \operatorname* { m i n } _ { \boldsymbol { u } \in \mathrm { s u p p } ( \mu ) } f _ { t - 1 } ( \boldsymbol { u } ) , \operatorname* { m a x } _ { \boldsymbol { u } \in \mathrm { s u p p } ( \mu ) } f _ { t - 1 } ( \boldsymbol { u } ) \right] .
$$

Thus $\mathbb { E } _ { P _ { t } } [ \xi _ { t } ] = 0$ and $\xi _ { t } ( \pmb { u } ) \in [ - 2 , 2 ]$ on $\operatorname { s u p p } ( P _ { t } )$ . It follows that $1 - \xi _ { t } ( { \pmb u } ) / 2 \geq 0$ and

$$
Q _ { t } ( \mathcal { X } ) = 1 - \frac { 1 } { 2 } \mathbb { E } _ { P _ { t } } [ \xi _ { t } ] = 1 .
$$

Hence $Q _ { t }$ is a probability measure. Since X is compact and convex, the barycenter of $Q _ { t } , \boldsymbol { x } _ { t }$ , lies in $\mathcal { X }$ □

## A.2 A potential proof of logarithmic alternating regret

Define the learner’s cumulative efective loss through round $t - 1$ by $\begin{array} { r } { A _ { t - 1 } : = \sum _ { s = 1 } ^ { t - 1 } g _ { s } ( { \pmb x } _ { s } ) } \end{array}$ , and define its pointwise regret by $R _ { t - 1 } ( { \pmb u } ) : = A _ { t - 1 } - L _ { t - 1 } ( { \pmb u } )$ . Consider the potential

$$
\Phi _ { t } : = \operatorname* { m a x } _ { \theta \in \mathbb { R } } \log \int _ { \mathcal { X } } \exp \bigl ( \eta R _ { t - 1 } ( \boldsymbol { u } ) - \beta ( f _ { t - 1 } ( \boldsymbol { u } ) - \theta ) ^ { 2 } \bigr ) \mu ( \mathrm { d } \boldsymbol { u } ) .\tag{6}
$$

The factor $e ^ { \eta A _ { t - 1 } }$ is common to all comparator points, so the maximizing centering parameter and normalized distribution in (6) are exactly $\theta _ { t }$ and $P _ { t }$ in Algorithm 2. Initially, $\Phi _ { 1 } = 0$

Theorem 11 (Monotonicity of potential). Assume $\eta \leq 1 / 8$ and $\beta = \eta / 4$ . For every round $t \geq 1$ we have $\Phi _ { t + 1 } \leq \Phi _ { t } \leq 0$

Proof. Define $\xi _ { t } ( { \pmb u } ) : = f _ { t - 1 } ( { \pmb u } ) - \theta _ { t }$ and $\xi _ { t + 1 } ( \boldsymbol { u } ) : = f _ { t } ( \boldsymbol { u } ) - \theta _ { t + 1 }$ . By Lemma 3, both functions take values in $[ - 2 , 2 ] , \mathbb { E } _ { P _ { t } } [ \xi _ { t } ] = 0 , Q _ { t } = ( 1 - \xi _ { t } / 2 ) P _ { t }$ , and $\pmb { x } _ { t } = \mathbb { E } _ { Q _ { t } } [ \pmb { u } ]$

Since $g _ { t } = f _ { t - 1 } + f _ { t }$ is convex, Jensen’s inequality gives $g _ { t } ( \pmb { x } _ { t } ) \leq \mathbb { E } _ { Q _ { t } } [ g _ { t } ]$ . The two scalar centers cancel, so for every $\mathbf { \boldsymbol { u } } \in \mathcal { X }$

$$
R _ { t } ( \boldsymbol { u } ) - R _ { t - 1 } ( \boldsymbol { u } ) \leq \mathbb { E } _ { Q _ { t } } [ \xi _ { t } + \xi _ { t + 1 } ] - \xi _ { t } ( \boldsymbol { u } ) - \xi _ { t + 1 } ( \boldsymbol { u } ) .
$$

By the definition of the potential and the fact that $P _ { t }$ is its normalized exponential-weight distribution, we have

$$
\begin{array} { r l } & { \Phi _ { t + 1 } - \Phi _ { t } = \log \mathbb { E } _ { P _ { t } } \exp \bigl \{ \eta ( R _ { t } - R _ { t - 1 } ) + \beta ( \xi _ { t } ^ { 2 } - \xi _ { t + 1 } ^ { 2 } ) \bigr \} } \\ & { \qquad \leq \log \mathbb { E } _ { P _ { t } } \exp \bigl \{ \eta \bigl ( \mathbb { E } _ { Q _ { t } } [ \xi _ { t } + \xi _ { t + 1 } ] - \xi _ { t } - \xi _ { t + 1 } \bigr ) + \beta ( \xi _ { t } ^ { 2 } - \xi _ { t + 1 } ^ { 2 } ) \bigr \} . } \end{array}
$$

We use the following continuous counterpart of Lemma 2.

Lemma 7 (Functional one-step inequality). Let P be a probability measure on an arbitrary measurable space. Let ξ, ζ be measurable functions with values in $[ - 2 , 2 ]$ such that $\mathbb { E } _ { P } [ \xi ] = 0$ , and define $Q ( \mathrm { d } u ) : = ( 1 - \xi ( u ) / 2 ) P ( \mathrm { d } u ) . \ I f \eta \leq 1 / 8$ and $\beta = \eta / 4$ , then

$$
\log \mathbb { E } _ { P } \exp \big \{ \eta \big ( \mathbb { E } _ { Q } [ \xi + \zeta ] - \xi - \zeta \big ) + \beta ( \xi ^ { 2 } - \zeta ^ { 2 } ) \big \} \le 0 .
$$

Proof. For any measurable function z with values in $[ - 2 , 2 ]$ , define

$$
H ( z ) : = \log \mathbb { E } _ { P } \exp \bigl \{ \eta \bigl ( \mathbb { E } _ { Q } [ \xi + z ] - \xi - z \bigr ) + \beta ( \xi ^ { 2 } - z ^ { 2 } ) \bigr \} .
$$

It sufices to show that $H ( \zeta ) \leq 0$ . Let Π<sub>z</sub> be the probability measure obtained by normalizing the exponential in the definition of $H ( z )$ . To show that H is concave, fix functions $z , w$ such that $z _ { s } : = z + s w$ remains in $[ - 2 , 2 ]$ , and write $v _ { s } : = - \eta - 2 \beta z _ { s } . \ H ( z _ { s } )$ is diferentiable since it is logarithm of a bounded integrand over a compact set. We note that $\mathbb { E } _ { Q } [ \xi + z _ { s } ]$ is afine and contributes noting to the second-order derivative.

$$
\frac { \mathrm { d } ^ { 2 } } { \mathrm { d } s ^ { 2 } } H ( z _ { s } ) = \mathbb { E } _ { \Pi _ { z _ { s } } } \big [ ( v _ { s } ^ { 2 } - 2 \beta ) w ^ { 2 } \big ] - \big ( \mathbb { E } _ { \Pi _ { z _ { s } } } [ v _ { s } w ] \big ) ^ { 2 } .
$$

Since $z _ { s } \in [ - 2 , 2 ] , \beta = \eta / 4$ , and $\eta \leq 1 / 8$ , we have $v _ { s } \in [ - 2 \eta , 0 ]$ and $v _ { s } ^ { 2 } \le 4 \eta ^ { 2 } \le 2 \beta$ . Thus H is concave.

At $z = - \xi ,$ every exponent vanishes, so $H ( - \xi ) = 0 \ \mathrm { a n d } \ \Pi _ { - \xi } = P$ . For every bounded direction w, the directional derivative is

$$
\begin{array} { r l } & { D H ( - \xi ) [ w ] = \eta \mathbb { E } _ { Q } [ w ] + \mathbb { E } _ { P } [ ( - \eta + 2 \beta \xi ) w ] } \\ & { \qquad = ( 2 \beta - \eta / 2 ) \mathbb { E } _ { P } [ \xi w ] = 0 , } \end{array}
$$

where the second equality uses $Q = ( 1 - \xi / 2 ) P$ and the last uses $\beta = \eta / 4$ . Taking $w = \zeta + \xi$ , the path $- \xi + s w$ remains in $[ - 2 , 2 ]$ for $s \in [ 0 , 1 ]$ . Concavity therefore gives $H ( \zeta ) \leq H ( - \xi ) = 0$ □

Applying Lemma 7 with $P = P _ { t } , \xi = \xi _ { t }$ , and $\zeta = \xi _ { t + 1 }$ gives $\Phi _ { t + 1 } \leq \Phi _ { t }$ . Since $\Phi _ { 1 } = 0$ , the claim follows. □

## A.3 Proof of Theorem 7

Proof of Theorem 7. The choices $\eta = 1 / 8$ and $\beta = 1 / 3 2$ in Algorithm 2 satisfy the conditions in Theorem 11. Thus $\Phi _ { T + 1 } \leq \Phi _ { 1 } = 0$ . By the definition of the potential, for every $\theta \in \mathbb { R }$ ,

$$
\log \int _ { \chi } \exp \bigl ( \eta R _ { T } ( \boldsymbol { u } ) - \beta ( f _ { T } ( \boldsymbol { u } ) - \boldsymbol { \theta } ) ^ { 2 } \bigr ) \boldsymbol { \mu } ( \mathrm { d } \boldsymbol { u } ) \leq 0 .
$$

By (3), $R _ { T } ( { \mathbf { \boldsymbol { u } } } ) = \mathrm { R e g } _ { \mathrm { A l t } } ^ { T } ( { \mathbf { \boldsymbol { u } } } )$

If KL $_ { \iota } ( \rho \| \mu ) = \infty$ , the claim is immediate. Otherwise, write $r : = \mathrm { d } \rho / \mathrm { d } \mu$ . Since $r > 0$ ρ-almost surely, for every $\theta \in \mathbb { R }$

$$
\begin{array} { r l } {  { \int _ { \mathscr { X } } \exp \bigl ( \eta R _ { T } - \beta ( f _ { T } - \theta ) ^ { 2 } \bigr ) ~ \mathrm { d } \mu } } \\ & { \qquad \geq \int _ { \{ x \in \mathscr { X } : r > 0 \} } \exp \bigl ( \eta R _ { T } - \beta ( f _ { T } - \theta ) ^ { 2 } \bigr ) ~ \mathrm { d } \mu } \\ & { \qquad = \int _ { \mathscr { X } } \exp \bigl ( \eta R _ { T } - \beta ( f _ { T } - \theta ) ^ { 2 } - \log r \bigr ) ~ \mathrm { d } \rho . } \end{array}
$$

Here the inequality discards the possible µ-mass on $\{ x \in \mathcal { X } : r = 0 \}$ , and the equality uses ${ \mathrm { d } } \rho = r { \mathrm { ~ d } } \mu$ Taking logarithms and applying Jensen’s inequality to the probability measure ρ gives

$$
\begin{array} { r l } {  { 0 \geq \log \int _ { \chi } \exp \bigl ( \eta R _ { T } - \beta ( f _ { T } - \theta ) ^ { 2 } \bigr ) \ \mathrm { d } \mu } } \\ & { \geq \log \int _ { \chi } \exp \bigl ( \eta R _ { T } - \beta ( f _ { T } - \theta ) ^ { 2 } - \log r \bigr ) \ \mathrm { d } \rho } \\ & { \geq \mathbb { E } _ { \rho } \bigl [ \eta R _ { T } - \beta ( f _ { T } - \theta ) ^ { 2 } - \log r \bigr ] } \\ & { = \eta \operatorname { R e g } _ { \mathrm { A l t } } ^ { T } ( \rho ) - \beta \mathbb { E } _ { \rho } [ ( f _ { T } - \theta ) ^ { 2 } ] - \mathrm { K L } ( \rho \| \mu ) . } \end{array}
$$

Choose $\theta = \mathbb { E } _ { \rho } [ f _ { T } ]$ . Since $1 / \eta = 8$ and $\beta / \eta = 1 / 4$ , rearranging proves the first inequality in (4). The second follows from $\mathrm { V a r } _ { \rho } ( f _ { T } ) \leq 1$ □