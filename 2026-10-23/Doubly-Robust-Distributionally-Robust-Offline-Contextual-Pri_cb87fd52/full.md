# Doubly Robust Distributionally Robust Offline Contextual Pricing

Min Xu <sup>1</sup> Xinyi Yin <sup>2</sup> Yunfan Zhang <sup>3</sup> Yuxuan Han <sup>3</sup> Houcai Shen <sup>1</sup> Caihua Chen <sup>1</sup>

## Abstract

Offline contextual pricing often relies on logged observational data, but faces challenges from distributional shifts between training and deployment environments. Distributionally robust optimization (DRO) provides a principled approach to offpolicy evaluation and learning (OPE/L). However, existing methods are mostly limited to discrete actions. Recent work has explored DRO for continuous treatments using inverse propensity weighting (IPW), while such IPW-based estimators can be sensitive to the convergence rate of propensity score estimates, particularly when estimated nonparametrically, which may lead to larger estimation errors and regret. In this work, we develop a doubly robust (DR) framework for distributionally robust OPE/L in continuous pricing settings. For evaluation, we propose a localized DR estimator that addresses the computational challenges of worst-case expectations by fitting only a small number of regressions, comparable to standard non-robust DR, while achieving semiparametric efficiency under mild product rate conditions. For learning, we leverage the inherent smoothness of demand noise to handle pricing-specific discontinuities in revenue outcomes (e.g., threshold-based purchase decisions), establishing a finite-sample regret bound of $\tilde { \mathcal { O } } _ { p } ( T ^ { - s / ( 2 s + 1 ) } )$ for smoothness orders $s = 1 , 2$ . This bound improves upon existing regret rates in existing DRO-based off-policy learning (OPL) for continuous treatments. Extensive experiments under various levels of distribution shift validate our proposed framework.

## 1. Introduction

Personalized pricing has gained popularity on online platforms, as firms can account for consumer heterogeneity by setting prices based on individual contextual information (Cohen et al., 2020; Chen & Gallego, 2021; Tang et al., 2025; Zhang et al., 2025). While randomized controlled trials (RCTs) provide a reliable way to estimate treatment effects, see Colnet et al. (2024) for a review, the large number of possible price options (Ye et al., 2025) and the risks associated with real-time exploration often make off-policy evaluation and learning (OPE/L) from historical observational data a more practical and safer approach (Kallus & Zhou, 2018; Kitagawa & Tetenov, 2018; Athey & Wager, 2021; Zhou et al., 2023; Carranza & Athey, 2025).

Standard inverse probability weighting (IPW) (Horvitz & Thompson, 1952; Swaminathan & Joachims, 2015) and doubly robust (DR) estimators (Dud´ık et al., 2011; Zhao et al., 2015), have been well-developed for discrete treatment settings (Jiang & Li, 2016; Athey & Wager, 2021; Mbakop & Tabord-Meehan, 2021; Zhan et al., 2021; Fang et al., 2025b). Generally, direct application of these methods to continuous treatment settings is challenging, as it would reject most observed data due to the zero-probability of exact treatment matches, leading to high variance. To address these limitations, recent work has introduced kernel-based smoothing techniques that enable more stable estimation in continuous domains (Kallus & Zhou, 2018; Lee et al., 2022; Colangelo & Lee, 2026; Ai et al., 2026). However, even with kernel smoothing, OPE/L remains vulnerable to distributional shifts between training and deployment environments, which are common in dynamic markets due to concept drift or covariate shifts (Si et al., 2023; Wang et al., 2024; 2025). Distributionally robust optimization (DRO) provides a principled framework to mitigate such risks by optimizing for worst-case performance within an uncertainty set (Delage & Ye, 2010; Ben-Tal et al., 2013; Hu & Hong, 2013). DRO has been successfully applied to OPE/L in discrete treatment settings (Kallus et al., 2022; Si et al., 2023; Wang et al., 2025). For continuous treatments, recent works have extended DRO using kernel-smoothed IPW estimators (Leung et al., 2025), which provide finite-sample convergence guarantees. We summarize the comparison between our result and prior works in Table 1. Nevertheless, these IPW-based approaches remain dependent on the convergence rate of the propensity density estimates, where nonparametric estimation can lead to deteriorated estimation error and regret.

In this paper, we propose doubly robust algorithms for distributionally robust OPE/L, incorporating a scaled kernel function to handle continuous pricing settings. Unlike general continuous-treatment literature, which typically assumes a smooth outcome function (Ai et al., 2026; Colangelo & Lee, 2026), pricing inherently involves a discontinuous revenue function due to threshold purchase behaviors (Javanmard & Nazerzadeh, 2019; Cohen et al., 2020). Recently, Kallus & Zhou (2018) requires the smoothness of the outcome functions with respect to the treatment, which is also imposed on the conditional outcome density functions in Leung et al. (2025); Colangelo & Lee (2026). Nevertheless, we handle this rigorously through the s-smoothness of the cumulative distribution function (CDF) of noise in the demand function, for $s = 1 , 2$ . Our main contributions are as follows:

Distributionally Robust Off-policy Evaluation (OPE). While Neyman orthogonality and $\sqrt { T h }$ -consistency (h is the kernel bandwidth) have been established for continuous treatment in non-robust settings (Kennedy et al., 2017; Colangelo & Lee, 2026), extending it to a DRO framework poses a non-trivial challenge, as the DRO and DR components become highly coupled by a supremum over the log of moment generating functions. We propose the Localized Doubly Robust DRO OPE for Continuous Pricing (LDR<sup>2</sup>O<sup>2</sup>PE-CP) estimator and show that it achieves $\sqrt { T h } -$ consistency and enjoys semiparametric efficiency even under the DRO objective (see Section 3). Our approach ensures robustness to both environment shifts and estimation errors, remaining computationally tractable by fitting nuisances only three times.

Distributionally Robust Off-policy Learning (OPL). For policy learning, we establish a finite-sample regret bound of $\tilde { \mathcal { O } } _ { p } \big ( T ^ { - s / ( 2 \bar { s } + 1 ) } \big ) , s = 1 , 2$ , for a policy class with finite pseudo-dimension, even when nuisance functions are nonparametrically estimated at slow rates (see Section 4). This bound represents a theoretical improvement over the $\mathcal { O } _ { p } ( T ^ { - 1 / 4 } )$ rate established in existing distributionally robust (Leung et al., 2025) and non-robust (Kallus, 2018) approaches for continuous treatments, and is comparable to rates reported in the non-robust continuous treatment literature (Ai et al., 2026).

See https://github.com/OfflinePricing/ DR-DRO-PRICING for simulation and Expedia study.

## 1.1. Related Work

Off-Policy Evaluation/Learning. A large body of work has developed OPE/L methods for discrete treatment settings (Dud´ık et al., 2011; Kitagawa & Tetenov, 2018; Athey & Wager, 2021; Mbakop & Tabord-Meehan, 2021; Zhou et al.,

2023; Fang et al., 2025b), continuous treatment (Kallus & Zhou, 2018; Chernozhukov et al., 2019; Colangelo & Lee, 2026; Ai et al., 2026), and general treatment (Fang et al., 2025a). Although some allow the data to be adaptively collected (Zhan et al., 2021; 2024; Jin et al., 2021; 2025b), most remain non-robust to distributional shifts.

Recent efforts have incorporated distributional robustness to handle distribution shifts, primarily in discrete treatment settings (Kallus et al., 2022; Mu et al., 2022; Blanchet et al., 2023; Si et al., 2023; Guo et al., 2024; Jin et al., 2025a; Wang et al., 2025), and some also address multi-source shifts (Carranza & Athey, 2025; Imbens et al., 2025). The closest prior work is Leung et al. (2025), which develops a distributionally robust OPL framework for continuous treatments using kernel-smoothed IPW, with performance tied to propensity density convergence rates under nonparametric estimation. Our doubly robust approach achieves semiparametric efficiency under milder product rate conditions, yields a sharper finite-sample regret bound than Leung et al. (2025).

Contextual Pricing. Recently, many papers have been focusing on contextual pricing, see Den Boer (2015); Chen & Gallego (2021) for a review. Offline contextual pricing is also practically relevant in real marketplaces and platform operations (e.g., Zhou et al. (2024); Zhang et al. (2026)). Existing works can be categorized by the assumptions on the two unknown functions, namely the mean utility and the CDF of the noise. Javanmard & Nazerzadeh (2019); Cohen et al. (2020); Ban & Keskin (2021); Miao et al. (2022) assume the noise CDF is known in advance, and impose a parametric model on the product’s market value. Alternatively, Shah et al. (2019); Fan et al. (2024); Luo et al. (2024) allow the noise distribution also to be unknown, and use a semiparametric model where mean utility is parametrically organized. Furthermore, Nambiar et al. (2019); Chen & Gallego (2021); Hu et al. (2022); Bu et al. (2025) adopt a fully nonparametric approach to model both the mean utility function and the noise distribution. Chen et al. (2024) propose a doubly nonparametric random utility model for demand characterization, which is most relevant to our work.

## 2. Problem Formulation

## 2.1. Offline Contextual Pricing

We consider a contextual pricing setting where a context vector $X \in \mathcal { X } \subseteq \mathbb { R } ^ { d }$ and a random noise term $Z \in \mathbb { R }$ are drawn from a joint distribution $\mathbf { P } _ { 0 }$ . X and Z are independent and the marginal distributions are denoted by $F _ { X }$ and $F _ { Z } ,$ respectively. For a given context $X ,$ the buyer’s valuation is modeled as $v ( X , Z ) = f ( X ) + Z .$ , where $f : \mathcal { X } $ R is the unknown deterministic component of the valuation.

In the offline learning setting, we are given a training dataset $\mathcal { D } = \{ ( x _ { t } , p _ { t } , y _ { t } ) \} _ { t = 1 } ^ { T }$ consisting of $T$ independent and i.i.d.

Table 1. Comparison of convergence rates in the OPL literature. We report regret upper bounds with respect to sample size T, treating policy class complexity measures (e.g., entropy integrals) as constant factors. For our work and Ai et al. $( \hat { 2 } 0 2 6 ) , s = \hat { 1 } , 2$ corresponds to the smoothness order of the outcome functions, while Colangelo & Lee (2026) and Leung et al. (2025) require three-times differentiability $( s = 3 )$ of conditional outcome densities w.r.t. treatment. Here $\tilde { \mathcal { O } }$ hides logarithmic factors in T.
<table><tr><td>Paper</td><td>Action Space</td><td>Distributional Shifts?</td><td>Regret Bound</td></tr><tr><td>Kitagawa &amp; Tetenov (2018); Athey &amp; Wager (2021); Mbakop &amp; Tabord-Meehan (2021)</td><td>Binary</td><td>X</td><td> $\mathcal { O } _ { p } ( T ^ { - 1 / 2 } )$ </td></tr><tr><td>Dudík et al. (2011); Swaminathan &amp; Joachims (2015); Kallus (2018); Zhou et al. (2023)</td><td>Discrete</td><td>X</td><td> $\tilde { \mathcal { O } _ { p } } ( T ^ { - 1 / 2 } )$ </td></tr><tr><td>Kallus &amp; Zhou (2018); Colangelo &amp; Lee (2026); Ai et al. (2026)</td><td>Continuous</td><td>X</td><td> $\mathcal { O } _ { p } ( \dot { T } ^ { - s / ( 2 s + 1 ) } )$ </td></tr><tr><td>Kallus et al. (2022); Si et al. (2023); Wang et al. (2025)</td><td>Discrete</td><td>√</td><td> $\dot { \mathcal { O } } _ { p } ( T ^ { - 1 / 2 } )$ </td></tr><tr><td>Leung et al. (2025)</td><td>Continuous</td><td>√</td><td> $\mathcal { O } _ { p } ^ { \star } ( T ^ { - 1 / 4 } )$ </td></tr><tr><td>Ours</td><td>Continuous</td><td>√</td><td> $\tilde { \mathcal { O } } _ { p } \big ( T ^ { - s / ( 2 s + 1 ) } \big )$ </td></tr></table>

triples. Specifically, for each t, a context vector $x _ { t }$ and a noise term $z _ { t }$ are drawn from the underlying distribution $\mathbf { P } _ { 0 }$ . Subsequently, a price $p _ { t } \in \mathcal P$ is selected by a known logging policy $\pi _ { 0 }$ with probability $\pi _ { 0 } ( p _ { t } | x _ { t } )$ , where ${ \mathcal { P } } \triangleq$ $[ p _ { \mathrm { m i n } } , p _ { \mathrm { m a x } } ]$ denotes the set of feasible prices. Without loss of generality, we assume the prices are normalized such that $p _ { \operatorname* { m i n } } > 0 \mathrm { a n d } p _ { \operatorname* { m a x } } = 1$ . Then, the binary purchase decision $y _ { t } = \mathbf { 1 } \left\{ z _ { t } \geq p _ { t } - f ( x _ { t } ) \right\}$ is observed. Consequently, the observed revenue at time t is given by ${ \boldsymbol { r } } _ { t } = { \boldsymbol { p } } _ { t } { \boldsymbol { y } } _ { t }$ . We make the following assumptions on the data-generating process.

Assumption 2.1 (Unconfoundedness). The purchase decision $\bar { Y ( p ) } : = \mathbf { 1 } \{ v ( X , Z ) \geq p \}$ is independent of the price P conditional on the context $\bar { X } , \mathrm { i . e . , } Y ( p ) \bot \bot P \mid X$

Assumption 2.2 (Overlap). There exists a constant $\eta > 0$ such that $\pi _ { 0 } ( p | x ) \geq \eta$ for all $( x , p ) \in \mathcal { X } \times \mathcal { P }$

Assumption 2.3. There exists constants $\underline { { \rho } } , \overline { { \rho } } \in ( 0 , 1 )$ such that the joint distribution $\mathbf { P } _ { 0 }$ satisfies

$$
\underline { { \rho } } \leq \mathbf { P } _ { 0 } \left( v ( X , Z ) < P \mid X \right) \leq \overline { { \rho } } , \forall P \in \mathcal { P } , X \in \mathcal { X } .
$$

Assumption 2.4 (Smoothness of Noise Distribution). The noise distribution $F _ { Z }$ is s-differentiable with bounded sorder derivative, i.e. $, \operatorname* { s u p } _ { z } | F _ { Z } ^ { ( s ) } ( z ) | \le L _ { Z } < \infty$ for $s = 1$

Assumption 2.2 guarantees that every feasible price has a non-zero probability of being sampled for any given context (Kallus & Zhou, 2018; Si et al., 2023; Leung et al., 2025). This ensures sufficient exploration within the training data, making it possible to assess the performance of any new pricing strategy. Assumption 2.3 implies that there always exists the non-negligible probabilities that the customer will reject/accept the offer, regardless of the context. This boundedness is made to prevent our estimator from diverging to infinity. Assumption 2.4 is standard in the literature (Javanmard & Nazerzadeh, 2019; Chen & Gallego, 2021; Chen et al., 2024; Fan et al., 2024; Gong et al., 2025). And we discuss a stronger assumption that $F _ { Z }$ has a bounded second-order derivative, $s = 2$ (Kallus & Zhou, $2 0 1 8 ;$ Leung et al., 2025), which leads to an improved result in Section 4. These different smoothness assumptions align with the differentiability of the outcome regression in Ai et al. (2026).

## 2.2. Distributionally Robust Performance

A seller’s policy $\pi \in \Pi$ is defined as a deterministic function $\pi : \mathcal { X }  \mathcal { P }$ mapping contexts to prices, and the revenue is

$$
r ( X , \pi ( X ) , Z ) \triangleq \pi ( X ) \mathbf { 1 } \left\{ Z \geq \pi ( X ) - f ( X ) \right\} .
$$

We evaluate the performance of a policy π based on its expected revenue, formalized by the policy value function: Definition 2.5. The policy value function $R : \Pi  \mathbb { R }$ is defined as: $R ( \pi ) \triangleq \mathbb { E } _ { \mathbf { P } _ { 0 } } [ r ( X , \pi ( X ) , Z ) ]$

Given the distributional shift between training and testing environments, we adopt a distributionally robust framework that explicitly accounts for potential deviations from the nominal training distribution $\mathbf { P } _ { 0 }$ . We define distributional uncertainty set $\mathcal { U } _ { \mathbf { P } _ { 0 } } ( \delta )$ centered at $\mathbf { P } _ { 0 }$ with radius δ as:

$$
\mathcal { U } _ { \mathbf { P } _ { 0 } } ( \delta ) \triangleq \{ \mathbf { P } : \mathbf { P } \ll \mathbf { P } _ { 0 } \mathrm { ~ a n d ~ } D ( \mathbf { P } \| \mathbf { P } _ { 0 } ) \leq \delta \} ,
$$

where $\mathbf { P } \ll \mathbf { P } _ { 0 }$ denotes absolute continuity with respect to $\mathbf { P } _ { 0 }$ and $\begin{array} { r } { D ( \mathbf { P } \| \mathbf { P } _ { 0 } ) \triangleq \int \log ( \frac { \mathrm { d } \mathbf { P } } { \mathrm { d } \mathbf { P } _ { 0 } } ) \mathrm { d } \mathbf { P } } \end{array}$ is the Kullback-Leibler (KL) divergence between two probability measures P and $\mathbf { P } _ { 0 }$

Definition 2.6. For a given $\delta > 0$ , the distributionally robust value function $R _ { \delta } : \Pi $ R is defined as:

$$
R _ { \delta } ( \pi ) : = \operatorname* { i n f } _ { \mathbf { P } \in \mathcal { U } _ { \mathbf { P } _ { 0 } } ( \delta ) } \mathbb { E } _ { \mathbf { P } } [ r ( X , \pi ( X ) , Z ) ]
$$

The optimal policy $\pi ^ { * }$ is a policy that maximizes the distributionally robust value function: $\pi ^ { * } \in$ arg $\operatorname* { m a x } _ { \pi \in \Pi } R _ { \delta } ( \pi )$ While such an optimal policy does not always exist, one can construct a sequence of policies converging to the supremum $\operatorname { s u p } _ { \pi \in \Pi } \{ R _ { \delta } ( \pi ) \}$ ; for simplicity, we assume the maximum is attainable throughout our analysis. We formalize the learning metric via the discrepancy between the performance of the optimal policy and the performance of the learned policy π, defined as the distributionally robust regret:

Definition 2.7. The distributionally robust regret $\operatorname { R e g } ( \pi )$ of a policy $\pi \in \Pi$ is defined as $\mathrm { R e g } ( \pi ) \triangleq R _ { \delta } ( \pi ^ { * } ) - R _ { \delta } ( \pi )$ .

Recalling the strong duality results from Kallus et al. (2022) and Si et al. (2023), we transform the original infinitedimensional optimization over $\mathbf { P } \in \mathcal { U } _ { \mathbf { P } _ { 0 } } ( \delta )$ to a dual problem involving a single scalar α. For any policy $\pi \in \Pi$ , we assume that the distributionally robust value $R _ { \delta } ( \pi ) > 0$ which prevents degeneration into the trivial no-purchase outcome. It is natural in our setting and simply requires that the perturbation budget δ is not overly large, see Appendix A for a detailed discussion.

Lemma 2.8. Suppose Assumption 2.1 and 2.3. The distributionally robust value $R _ { \delta } ( \pi )$ is equivalent to

$$
R _ { \delta } ( \pi ) = \operatorname* { s u p } _ { \alpha \geq 0 } \varphi ( \pi , \alpha ) \triangleq - \alpha \log W ( \pi , \alpha ) - \alpha \delta ,\tag{1}
$$

where $W ( \pi , \alpha ) \triangleq \mathbb { E } _ { \mathbf { P } _ { 0 } } \left[ \exp \left( - r ( X , \pi ( X ) , Z ) / \alpha \right) \right]$ . Furthermore, $\varphi ( \pi , \cdot )$ is strictly concave with respect to α, and it admits a unique maximizer $\alpha ^ { \ast } ( \pi ) \in ( 0 , \bar { \alpha } ]$ , where $\bar { \alpha } \triangleq 1 / \delta .$

Unlike the discrete setting, using the indicator function (Dud´ık et al., 2011; Si et al., 2023; Zhan et al., 2024) to estimate the robust policy value $R _ { \delta } ( \pi )$ is not valid in the continuous setting. Kallus & Zhou (2018) and Leung et al. (2025) propose a kernel smoothing approach, where the indicator function is replaced by a kernel function $K ( \cdot )$ with bandwidth h. They also assume that the logging policy π<sub>0</sub> is known and propose to estimate $W ( \pi , \alpha )$ by IPW, based on normalizing the propensity ratios $\frac { K \left( { \frac { p _ { t } - \pi ( x _ { t } ) } { h } } \right) } { h \pi _ { 0 } ( p _ { t } | x _ { t } ) }$

$$
\hat { W } _ { T } ^ { I P W } ( \pi , \alpha ) = \sum _ { t = 1 } ^ { T } \frac { \frac { K \left( \frac { p _ { t } - \pi ( x _ { t } ) } { h } \right) } { h \pi _ { 0 } ( p _ { t } | x _ { t } ) } } { \sum _ { t = 1 } ^ { T } \frac { K \left( \frac { p _ { t } - \pi ( x _ { t } ) } { h } \right) } { h \pi _ { 0 } ( p _ { t } | x _ { t } ) } } \exp ( - p _ { t } y _ { t } / \alpha ) ,\tag{2}
$$

$$
\mathrm { t h e n ~ } \hat { \pi } ^ { I P W } \in \arg \operatorname* { m a x } _ { \pi \in \Pi } \hat { R } _ { \delta } ^ { I P W } ( \pi ) ,\tag{3}
$$

$$
\mathrm { w h e r e } \quad \hat { R } _ { \delta } ^ { I P W } ( \pi ) \overset { \triangle } { = } \operatorname* { s u p } _ { \alpha \in ( 0 , \bar { \alpha } ] } \Big \{ - \alpha \hat { W } _ { T } ^ { I P W } ( \pi , \alpha ) - \alpha \delta \Big \} .
$$

However, in practice, the logging policy $\pi _ { 0 }$ is unknown and needs to be learned from the data. Furthermore, neither the standard IPW estimator nor its self-normalized variant achieves semiparametric efficiency. To overcome these challenges, we adopt doubly robust methods.

## 3. Distributionally Robust Policy Evaluation

In this section, we develop a doubly robust framework for evaluating the distributionally robust value of a target policy. However, it is challenging to estimate the distributionally robust value directly, as it involves a non-linear dual optimization problem and, potentially, a continuum of nuisance regressions. Building upon the localized debiased machine learning framework proposed by Kallus et al. (2022; 2024), we extend the localization technique from discrete to continuous treatment settings. Unlike the discrete action case, our estimator incorporates kernel smoothing to handle the continuous policy space.

## 3.1. Localized Doubly Robust Estimation

As established in Lemma 2.8, the distributionally robust value $R _ { \delta } ( \pi )$ can be characterized via its dual form, and there exists a unique optimal dual parameter $\alpha ^ { * } : = \alpha ^ { * } ( \pi )$ that satisfies the following first-order condition

$$
- \log W _ { 0 } ( \pi , \alpha ^ { * } ) - { \frac { W _ { 1 } ( \pi , \alpha ^ { * } ) } { \alpha ^ { * } W _ { 0 } ( \pi , \alpha ^ { * } ) } } - \delta = 0 .\tag{4}
$$

where $\begin{array} { r l r } { W _ { j } ( \pi , \alpha ) } & { { } \triangleq } & { \mathbb { E } [ ( \pi ( X ) Y ( \pi ( X ) ) ) ^ { j } \exp ( - \pi ( X ) } \end{array}$ ${ \cal Y } ( \pi ( X ) ) \bar { / } \alpha ) ] , ~ j ~ = ~ 0 , 1$ . Using the shorthand $W _ { i } ^ { * } ~ =$ $W _ { j } ( \pi , \alpha ^ { * } )$ , the robust policy value is then given by

$$
R _ { \delta } ( \pi ) = - \alpha ^ { * } \log W _ { 0 } ^ { * } - \alpha ^ { * } \delta ,\tag{5}
$$

Therefore, estimating $\alpha ^ { * } ( \pi )$ and $R _ { \delta } ( \pi )$ in (4) and (5) is equivalent to estimating the root of the following moment equation with parameter $\theta = [ \alpha , W _ { 0 } , W _ { 1 } , R _ { \delta } ] ^ { \top }$

$$
\mathbb { E } \left[ U ( \pi ( X ) Y ( \pi ( X ) ) ; \alpha ) + G ( \theta ) \right] = \mathbf { 0 } ,\tag{6}
$$

$$
U ( r ; \alpha ) = \left[ \begin{array} { c } { \exp ( - r / \alpha ) } \\ { r \exp ( - r / \alpha ) } \\ { 0 } \\ { 0 } \end{array} \right] , G ( \theta ) = \left[ \begin{array} { c } { - W _ { 0 } } \\ { - W _ { 1 } } \\ { - \delta - \log W _ { 0 } - \frac { W _ { 1 } } { \alpha W _ { 0 } } } \\ { - R _ { \delta } - \alpha \log W _ { 0 } - \alpha \delta } \end{array} \right] .
$$

In practice, direct estimation of (4) and (5) is infeasible because the counterfactual outcomes $Y ( \pi ( X ) )$ are not observed. Instead, we consider the following doubly robust moment equation in terms of the observed variables $V = ( X , P , Y )$ , with nuisances $\lambda _ { 1 } , \lambda _ { 2 }$ to be estimated:

$$
\mathbb { E } \left[ \psi ( V ; \theta , \lambda _ { 1 } ^ { * } ( V ; \alpha ) , \lambda _ { 2 } ^ { * } ( V ) ) \right] = { \bf 0 } ,
$$

$$
\begin{array} { l } { { \psi ( v ; \theta , \lambda _ { 1 } , \lambda _ { 2 } ) = \displaystyle \frac { K ( \frac { p - \pi ( x ) } { h } ) } { h \lambda _ { 2 } ( x , p ) } \left( U ( p y ; \alpha ) - \lambda _ { 1 } ( x , p ; \alpha ) \right) } } \\ { { + \lambda _ { 1 } ( x , \pi ( x ) ; \alpha ) + G ( \theta ) , } } \end{array}
$$

where we incorporate kernel function $K ( \cdot )$ with bandwidth h to handle the continuous policy space. And $\lambda _ { 2 } ^ { * } ( x , p ) =$ $\pi _ { 0 } ( p | x )$ is the logging policy,

$$
\begin{array} { r l } & { \lambda _ { 1 } ^ { * } ( x , p ; \alpha ) = \mathbb { E } [ U ( P Y ; \alpha ) | X = x , P = p ] } \\ & { \qquad = [ m _ { 0 } ( x , p ; \alpha ) , m _ { 1 } ( x , p ; \alpha ) , 0 , 0 ] ^ { \top } , } \\ & { m _ { j } ( x , p ; \alpha ) = \mathbb { E } [ ( P Y ) ^ { j } \exp ( - P Y / \alpha ) | X = x , P = p ] . } \end{array}
$$

A significant technical hurdle in distributionally robust policy evaluation is that the outcome regressions, $m _ { j } ( \cdot ; \alpha )$ are coupled with the dual parameter α. Recalling that Neyman orthogonality for continuous treatment (Colangelo & Lee, 2026) implies that the Gateaux derivativesˆ of $\mathbb { E } \left[ \psi ( V ; \theta , \lambda _ { 1 } ( V ; \alpha ) , \lambda _ { 2 } ( V ) ) \right]$ with respect to the nuisances $( \lambda _ { 1 } , \lambda _ { 2 } )$ are $\mathcal { O } ( h ^ { s } )$ when evaluated at $\theta ^ { * } =$ $( \alpha ^ { * } , W _ { 0 } ^ { * } , W _ { 1 } ^ { * } , R _ { \delta } ( \pi ) ) , \lambda _ { 1 } ( \cdot ; \alpha ) = \lambda _ { 1 } ^ { * } ( \cdot ; \alpha ^ { * } )$ and $\lambda _ { 2 } = \lambda _ { 2 } ^ { * }$ This orthogonality ensures that our estimator remains firstorder insensitive to the estimation errors of the nuisances as the bandwidth $h  0 .$ . Thus, we leverage the localization technique (Kallus et al., 2022; 2024) to utilize an initial guess ${ \hat { \alpha } } _ { i n i t }$ and fit the nuisances only at this localized point, instead of a continuum of regressions for whole $\alpha \in ( 0 , \bar { \alpha } ]$ as provided the initial guess is sufficiently close to $\alpha ^ { * }$

the root of (7). The gradients are provided in Lemma C.1.   
3.2. Asymptotic Analysis   
Set $h = \mathcal { O } ( T ^ { - \gamma _ { h } } )$ . Define the estimation rates as follows,   
$\operatorname* { m a x } _ { j = 0 , 1 } \left\| \hat { m } _ { j } ^ { ( \ell ) } ( \cdot ; \hat { \alpha } _ { i n i t } ^ { ( \ell ) } ) - m _ { j } ( \cdot ; \hat { \alpha } _ { i n i t } ^ { ( \ell ) } ) \right\| _ { \infty } \leq o _ { p } ( T ^ { - \rho _ { m } } ) ,$   
$\begin{array} { r } { \left\| \hat { \pi } _ { 0 } ^ { ( \ell ) } - \pi _ { 0 } \right\| _ { \infty } \leq o _ { p } ( T ^ { - \rho _ { \pi } } ) , \ : \left| \hat { \alpha } _ { i n i t } - \alpha ^ { * } \right| \leq o _ { p } ( T ^ { - \rho _ { \alpha } } ) . } \end{array}$   
Assumption 3.1. We assume that $\begin{array} { r } { \rho _ { \pi } + \rho _ { \alpha } \wedge \rho _ { m } \geq \frac { 1 - \gamma _ { h } } { 2 } } \end{array}$   
Assumption 3.2. Kernel function $K ( u ) : \mathbb { R }  [ 0 , \infty )$ has   
Lipschitz constant $L _ { K }$ and satisfies:   
(i) $\begin{array} { r } { \int K ( u ) \mathrm { d } u = 1 , \int u K ( u ) \mathrm { d } u = 0 , } \end{array}$   
(ii) $\begin{array} { r } { \int | u | ^ { k } | K ( u ) | d \dot { u } < \infty } \end{array}$ and $\textstyle \int | u | ^ { k } K ^ { 2 } ( u ) d u < \infty$ for   
$k = { \mathrm { 0 } } , 1 , 2 .$   
Theorem 3.3. Let $\theta ^ { * } \ = \ [ \alpha ^ { * } , W _ { 0 } ^ { * } , W _ { 1 } ^ { * } , R _ { \delta } ( \pi ) ] ^ { \top }$ be the   
solution $o f \ ( 6 )$ . Let $\hat { \theta }$ be the estimator defined in Algo  
rithm 1. Then, under Assumptions 2.1-2.3, 3.1, 3.2, and   
$\sqrt { T h ^ { 1 + 2 s } }  0$ when Assumption 2.4 holds for s-order   
differentiability, s = 1, 2, we have   
$\begin{array} { r } { \sqrt { T h } ( \hat { \theta } - \theta ^ { * } ) \stackrel { d } {  } \mathcal { N } ( \mathbf { 0 } , \Sigma ^ { * } ) , \ \Sigma ^ { * } = J ^ { * - 1 } \Omega ^ { * } J ^ { * - \top } } \end{array}$ , where   
$J ^ { * } = \left| \begin{array} { c c c c } { { W _ { 1 } ^ { * } / \alpha ^ { * } { } ^ { 2 } } } & { { - 1 } } & { { 0 } } & { { 0 } } \\ { { W _ { 2 } ^ { * } / \alpha ^ { * } { } ^ { 2 } } } & { { 0 } } & { { - 1 } } & { { 0 } } \\ { { \frac { W _ { 1 } ^ { * } } { \alpha ^ { * } { } ^ { 2 } W _ { 0 } ^ { * } } } } & { { - \frac { 1 } { W _ { 0 } ^ { * } } + \frac { W _ { 1 } ^ { * } } { \alpha ^ { * } W _ { 0 } ^ { 2 } } } } & { { - \frac { 1 } { \alpha ^ { * } W _ { 0 } ^ { * } } } } & { { 0 } } \\ { { - \log W _ { 0 } ^ { * } - \delta } } & { { - \frac { \alpha ^ { * } } { W _ { 0 } ^ { * } } } } & { { 0 } } & { { - 1 } } \end{array} \right| ,$ $\lceil \quad W _ { 1 } ^ { * } / \alpha ^ { * 2 }$   
$\Omega ^ { * } = \int K ^ { 2 } ( u ) d u \cdot \mathrm { d i a g } ( \omega _ { 0 } , \omega _ { 1 } , 0 , 0 ) .$   
$\omega _ { j } = \mathbb { E } \left[ \frac { \mathrm { V a r } ( ( P Y ) ^ { j } \exp ( - P Y / \alpha ^ { * } ) | X , P = \pi ( X ) ) } { \pi _ { 0 } ( \pi ( X ) | X ) } \right]$   
and $\Sigma ^ { * }$ is the optimal covariance, then $\hat { \theta }$ achieves the semi  
parametric efficiency lower bound for θ<sup>∗</sup>.

Algorithm 1 Localized Doubly Robust DRO OPE for Con  
tinuous Pricing $\mathrm { ( L D R ^ { 2 } O ^ { 2 } P E { \cdot } C P ) }$   
1: Input: Dataset $\mathcal { D } ,$ uncertainty radius $\delta ,$ policy $\pi ,$ kernel   
function K with bandwidth h.   
2: Randomly evenly partition D into $\{ \mathcal { T } _ { 1 } , \ldots , \mathcal { T } _ { L } \}$   
3: for $\ell = 1$ to L do   
4: Train $\hat { \pi } _ { 0 } ^ { ( \ell ) } ( p | x )$ to fit $\pi _ { 0 } ( p | x )$ using $\mathcal { T } _ { \ell } ^ { c } = \mathcal { D } \setminus \mathcal { T } _ { \ell } .$   
5: Randomly split $\mathcal { T } _ { \ell } ^ { c }$ into two halves: $\mathcal { I } _ { 1 } , \mathcal { I } _ { 2 } .$   
6: $\alpha _ { i n i t } ^ { ( \ell ) } $ InitialEstimate $( { \mathcal { I } } _ { 1 } , \delta , \pi ) .$   
7: Train $\hat { m } _ { j } ^ { ( \ell ) } ( x , p ; \alpha _ { i n i t } ^ { ( \ell ) } )$ to fit $m _ { j } ( x , p ; \alpha _ { i n i t } ^ { ( \ell ) } )$ using   
data $\begin{array} { r } { \mathcal { I } _ { 2 } , \stackrel { \bf { \Delta } } { \jmath } = 0 , 1 . } \end{array}$   
8: end for   
9: Initialize $\begin{array} { r } { \alpha  \frac { 1 } { L } \sum _ { \ell = 1 } ^ { L } \alpha _ { i n i t } ^ { ( \ell ) } . } \end{array}$   
10: Find αˆ that solves the estimated moment equation:   
$- \log \hat { W } _ { 0 } ( \alpha ) - \frac { \hat { W } _ { 1 } ( \alpha ) } { \alpha \hat { W } _ { 0 } ( \alpha ) } - \delta = 0 ,$ , where (7)   
$\hat { W } _ { j } ( \alpha ) \triangleq \frac { 1 } { T } \sum _ { \ell = 1 } ^ { L } \sum _ { t \in \mathbb { Z } _ { \ell } } \hat { W } _ { j } ^ { ( t , \ell ) } ( \alpha ) , \quad j = 0 , 1 ,$   
$\hat { W } _ { j } ^ { ( t , \ell ) } ( \alpha ) \triangleq \hat { m } _ { j } ^ { \ell } ( x _ { t } , \pi ( x _ { t } ) ; \hat { \alpha } _ { i n i t } ^ { ( \ell ) } ) + \frac { K ( \frac { p _ { t } - \pi ( x _ { t } ) } { h } ) } { h \hat { \pi } _ { 0 } ^ { ( \ell ) } ( p _ { t } | x _ { t } ) }$   
$\cdot \left( ( p _ { t } y _ { t } ) ^ { j } \exp ( - p _ { t } y _ { t } / \alpha ) - \hat { m } _ { j } ^ { ( \ell ) } ( x _ { t } , p _ { t } ; \hat { \alpha } _ { i n i t } ^ { ( \ell ) } ) \right) .$   
11: Calculate $\hat { R } _ { { \delta } } \gets - \hat { \alpha } \log \hat { W } _ { 0 } ( \hat { \alpha } ) - \hat { \alpha } \delta .$   
12: Return: $\hat { \theta } = [ \hat { \alpha } , \hat { W } _ { 0 } ( \hat { \alpha } ) , \hat { W } _ { 1 } ( \hat { \alpha } ) , \hat { R } _ { \delta } ] ^ { \top }$

We propose Algorithm 1, Localized Doubly Robust DRO OPE for Continuous Pricing $\mathrm { ( L D R ^ { 2 } O ^ { 2 } P E { - } C P ) }$ , to summarize the localized doubly robust evaluation procedure for continuous treatment. Following the two-level cross-fitting framework from Kallus et al. (2022), our implementation is tailored to handle the continuous action space. First, we randomly evenly partition $\mathcal { D }$ into L disjoint folds $\{ \mathcal { T } _ { \ell } \} _ { \ell \in [ L ] }$ We estimate the propensity estimator $\hat { \pi } _ { 0 } ^ { ( \ell ) }$ using the out-offold data $\mathcal { T } _ { \ell } ^ { c } = \mathcal { D } \setminus \mathcal { T } _ { \ell } .$ And we address the dependence of outcome regressions on α by fitting $\hat { m } _ { j } ^ { ( \ell ) }$ only at a localized point $\hat { \alpha } _ { i n i t } ^ { ( \ell ) }$ . Here, $\hat { \alpha } _ { i n i t } ^ { ( \ell ) }$ is a preliminary consistent estimate obtained by InitialEstimate, for example, via a continuoustreatment IPW estimator (Leung et al., 2025). By leveraging the localization technique, we circumvent these continuum regression problems, requiring the training of only three machine learning models. Finally, we solve the empirical moment equation for the dual parameter α by the Newton-Raphson method (Si et al., 2023; Leung et al., 2025) to find

Theorem 3.3 holds under some additional regularity conditions on the nuisance smoothness, which are formally stated in Appendix B. Algorithm 1 achieves $\sqrt { T h }$ -consistency, aligned with the $\sqrt { T h }$ -consistency under non-robust setting (Colangelo & Lee, 2026). Furthermore, our $\mathrm { L D R ^ { 2 } O ^ { 2 } P E { - } C \bar { P } }$ estimator attains semiparametric efficiency, which enables a flexible trade-off in the convergence rates of the nuisance functions to satisfy the product rate condition in Assumption 3.1. Notably, this requirement is relaxed to $( 1 - \gamma _ { h } ) / 2$ which is strictly weaker than the $1 / 2$ threshold required for $\sqrt { T }$ -consistent semiparametric estimators in discrete action spaces (Kallus et al., 2022). Moreover, it yields a more stable covariance structure Ω<sup>∗</sup> that depends on the conditional variance rather than the second moment as in the self-normalized IPW estimator of (2) (Leung et al., 2025), even when the logging policy is known.

Finally, we note that the asymptotic normality in Theorem 3.3 requires the bandwidth h to satisfy an undersmoothing condition relative to the smoothness of the noise distribution in Assumption 2.4. Specifically, the approximation bias introduced by kernel-smoothing is $\mathcal { O } _ { p } ( h ^ { s } )$ for $s = 1 , 2$ . To ensure this bias vanishes in the limiting distribution, one requires $\sqrt { T h ^ { 1 + 2 s } }  0 .$ , leading to a regret rate slightly slower than $\tilde { \mathcal { O } } _ { p } ( T ^ { - s / ( 2 s + 1 ) } )$ (see Theorem 4.4). This undersmoothing strategy is standard in the double machine learning literature for continuous treatment (Colangelo & Lee, 2026) as it enables the construction of valid frequentist confidence intervals without explicit bias correction.

## 4. Distributionally Robust Policy Learning

In this section, we now focus on learning a robust policy πˆ that maximizes the empirical distributionally robust value defined in Definition 4.1, $\hat { \pi } \in \arg \operatorname* { m a x } _ { \pi \in \Pi } \hat { R } _ { \delta } ( \pi )$ , and analyze how the empirically optimal policy performs out of sample by a finite-sample theoretical guarantee.

Definition 4.1 (Doubly Robust Estimators). Let $K ( \cdot )$ be a kernel function with bandwidth $h > 0 .$ , and $\{ ( x _ { t } , p _ { t } , y _ { t } ) \} _ { t = 1 } ^ { T }$ be a given dataset. Define the DR estimator for a continuous pricing problem as

$$
\begin{array} { l } { \displaystyle \hat { W } _ { T } ( \pi , \alpha ) = \frac { 1 } { L } \sum _ { \ell = 1 } ^ { L } \hat { W } ^ { ( \ell ) } ( \pi , \alpha ) , \ ~ ( 8 ) } \\ { \displaystyle \hat { W } ^ { ( \ell ) } ( \pi , \alpha ) = \frac { 1 } { \left| \mathcal { Z } _ { \ell } \right| } \sum _ { \ell \in \mathcal { Z } _ { \ell } } \hat { m } _ { 0 } ^ { ( \ell ) } ( x _ { t } , \pi ( x _ { t } ) ; \alpha ) + \frac { 1 } { \left| \mathcal { Z } _ { \ell } \right| \hat { S } _ { \ell } ^ { \eta } } } \\ { \displaystyle ~ \cdot \sum _ { \ell \in \mathcal { Z } _ { \ell } } \frac { K \left( \frac { \rho _ { \ell } - z ( x _ { \ell } ) } { h } \right) } { h \hat { \pi } _ { 0 } ^ { ( \ell ) } \left( p _ { \ell } \left| x _ { t } \right| \right) } \left( \exp ( - p _ { \ell } \gamma _ { \ell } / \alpha ) - \hat { m } _ { 0 } ^ { ( \ell ) } ( x _ { t } , p _ { t } ; \alpha ) \right) , \ ~ } \\ { \displaystyle \mathrm { w h e r e } \ \hat { S } _ { \ell } ^ { \pi } = \frac { 1 } { \left| \mathcal { Z } _ { \ell } \right| } \sum _ { \ell \in \mathcal { Z } _ { \ell } } \frac { K \left( \frac { p _ { \ell } - \pi ( x _ { t } ) } { h } \right) } { h \hat { \pi } _ { 0 } ^ { ( \ell ) } \left( p _ { \ell } \left| x _ { t } \right| \right) } . } \end{array}
$$

Then, we define the distributionally robust value estimator $\hat { R } _ { \delta } : \Pi  \mathbb { R }$ as

$$
\begin{array} { r l } & { \hat { R } _ { \delta } ( \pi ) \triangleq \underset { \alpha \geq 0 } { \operatorname* { s u p } } \{ \hat { \varphi } _ { T } ( \pi , \alpha ) \} , } \\ & { \hat { \varphi } _ { T } ( \pi , \alpha ) \triangleq - \alpha \log \hat { W } _ { T } ( \pi , \alpha ) - \alpha \delta . } \end{array}
$$

Different from distributionally robust policy evaluation at an initial guess $\alpha _ { i n i t } ^ { ( \ell ) } ( \pi )$ , distributionally robust policy learning needs to compute the distributionally robust estimator defined in Definition 4.1, and find the optimal ${ \hat { \alpha } } ^ { * } ( \pi )$ for maximizing $\hat { \varphi } _ { T } ( \pi , \alpha )$ . Thus, policy learning inevitably requires $\hat { m } _ { 0 } ^ { ( \ell ) } ( x , p ; \alpha )$ as a continuum estimate of the outcome regression functions $\{ m _ { 0 } ( \cdot , \cdot ; \alpha ) : \mathcal { X } \times \mathcal { P }  \mathbb { R } \mid \alpha \in ( 0 , \bar { \alpha } ] \}$

## 4.1. Learning Algorithm

A significant challenge arises because $m _ { 0 }$ depends on the dual variable α. In the policy learning phase, α varies continuously, and retraining a regression model for every possible value of α would be computationally prohibitive. Following the approach proposed by Kallus et al. (2022), we utilize a local weighting approach to estimate $m _ { 0 } ( x , p ; \alpha )$ . Specifically, given a dataset, we first learn a set of data-driven weight functions $\{ \hat { \omega } _ { t } ^ { ( \ell ) } ( x , p ) \} _ { t \in \mathcal { T } _ { \ell } ^ { c } }$ . These weights represent the proximity or similarity of the observed covariate-action pair $( x _ { i } , p _ { i } )$ to a query point $( x , p )$ . Common methods for constructing these weights include k-nearest neighbors, kernel regressions, and generalized random forests (Athey et al., 2019; Oprescu et al., 2019; Bertsimas & Kallus, 2020; Cevid et al., 2022; Khosravi et al., 2022). With these weights, the estimator for the regression continuum for any $\alpha > 0$ is

$$
\hat { m } _ { 0 } ^ { ( \ell ) } ( x , p ; \alpha ) = \sum _ { t \in \mathbb { Z } _ { \ell } ^ { c } } \hat { \omega } _ { t } ^ { ( \ell ) } ( x , p ) \exp ( - p _ { t } y _ { t } / \alpha ) .\tag{9}
$$

In our experiments, we constructed the weights using generalized random forests (Athey et al., 2019). Specifically, we first train a regression forest targeting the reward r. For a given query point $( x , p )$ , the forest-induced weights $\hat { \omega } _ { t } ( x , p )$ are computed by averaging the normalized leaf memberships across all B trees,

$$
\hat { \omega } _ { t } ^ { ( \ell ) } ( x , p ) = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \frac { \mathbf { 1 } ( x _ { t } , p _ { t } ) \in L _ { b } ( x , p ) } { | L _ { b } ( x , p ) | } ,\tag{10}
$$

where $L _ { b } ( x , p )$ denotes the set of training examples falling into the same leaf as the query point in the b-th tree. This method has been successfully applied in statistical estimation and decision making (Bertsimas & Kallus, 2020; Kallus et al., 2022; Kallus & Mao, 2023).

Since joint optimization over $( \pi , \alpha )$ is generally non-convex and computationally intractable, we adopt an alternating fashion (Kallus et al., 2022; Si et al., 2023). While this method guarantees convergence only to a local maximum, it is computationally efficient and empirically effective. The complete procedure, termed Continuum Doubly Robust DRO OPL for Continuous Pricing $\mathrm { ( C D R ^ { 2 } O ^ { 2 } P L \mathrm { - } C P ) }$ , is detailed in Algorithm 2.

## 4.2. Statistical Performance Guarantee

We establish the finite-sample statistical performance guarantee for the distributionally robust optimal policy πˆ. Under Assumption 3.2, the bias introduced by kernel density estimation is $\mathcal { O } ( h ^ { s } )$ when Assumption 2.4 holds for s-order bounded derivative (see Lemma C.2), and depends on convergence rate of $\hat { m } _ { 0 } ^ { ( \ell ) } ( x , p ; \alpha )$ (see Lemma C.5).

Assumption 4.2. Suppose $\left\{ \hat { m } _ { 0 } ^ { ( \ell ) } ( \cdot ; \alpha ) : \alpha \in ( 0 , \bar { \alpha } ] \right\} _ { \ell = 1 } ^ { L }$ and $\{ \hat { \pi } _ { 0 } ^ { ( \ell ) } ( \cdot ) \} _ { \ell = 1 } ^ { L }$ are learned from a dataset of $\textstyle { \frac { T ( L - 1 ) } { L } }$ points, satisfying the following conditions:

(i). There exists $\gamma _ { \pi } , \gamma _ { m } > 0$ such that   
E $\left[ \left. \hat { \pi } _ { 0 } ^ { ( \ell ) } - \pi _ { 0 } \right. _ { \infty } ^ { 2 } \right] = o ( T ^ { - 2 \gamma _ { \pi } } )$ , and   
$\begin{array} { r } { \mathbb { E } \left[ \left( \operatorname* { s u p } _ { \alpha \in ( 0 , \bar { \alpha } ] } \left. \hat { m } _ { 0 } ^ { ( \ell ) } ( \cdot , \cdot ; \alpha ) - m _ { 0 } ( \cdot , \cdot ; \alpha ) \right. _ { \infty } \right) ^ { 2 } \right] = o ( T ^ { - 2 \gamma _ { m } } ) . } \end{array}$   
(ii). $\hat { \pi } _ { 0 } ^ { ( \ell ) } \in [ \eta , 1 ] , \hat { m } _ { 0 } ^ { ( \ell ) } ( \cdot ; \alpha ) \in [ 0 , 1 ] , \forall \ell \in [ L ] , \alpha \in ( 0 , \bar { \alpha } ] .$   
(iii). $\{ \hat { m } _ { 0 } ^ { ( \ell ) } \} _ { \ell = 1 } ^ { L }$ are $L _ { \hat { m } } \mathrm { - L i p s c h i t z }$ with respect to $( x , p ) \in$   
$\mathcal { X } \times \mathcal { P }$ and monotonically non-decreasing in α.

Algorithm 2 Continuum Doubly Robust DRO OPL for   
Continuous Pricing $( \mathrm { C D R ^ { 2 } O ^ { 2 } P L \mathrm { \bar { - } C P } ) }$   
1: Input: Dataset D, logging policy $\pi _ { 0 } ,$ uncertainty radius   
$\delta ,$ kernel function K with bandwidth h, policy class Π.   
2: Output: Distributionally robust optimal policy πˆ.   
Phase 1: Continuum Regression   
3: Randomly evenly partition D into L disjoint folds   
$\{ \mathcal { T } _ { 1 } , \ldots , \mathcal { T } _ { L } \}$   
4: for $\ell = 1$ to L do   
5: Let $\mathcal { T } _ { \ell } ^ { c } = \mathcal { D } \setminus \mathcal { T } _ { \ell }$ be the training data.   
6: Train $\hat { \pi } _ { 0 } ^ { ( \ell ) } ( p | x )$ to fit $\pi _ { 0 } ( p | x )$ using $\mathcal { T } _ { \ell } ^ { c } .$   
7: Train $\hat { m } _ { 0 } ^ { ( \ell ) } ( x , p ; \alpha )$ to fit $m _ { 0 } ( x , p ; \alpha )$ for all $\alpha \in$   
$( 0 , \bar { \alpha } ]$ using (9)–(10) with $\mathcal { T } _ { \ell } ^ { c }$   
8: end for   
Phase 2: Alternating Optimization   
9: Initialize πˆ.   
10: repeat   
11: Compute $\hat { W } _ { T } ( \hat { \pi } , \alpha )$ using (8).   
12: Update $\hat { \alpha } \gets \arg \operatorname* { m a x } _ { \alpha \geq 0 } - \alpha$ log $\hat { W } _ { T } ( \hat { \pi } , \alpha ) - \alpha \delta .$   
13: Update $\hat { \pi }  \arg \operatorname* { m i n } _ { \pi \in \Pi } \hat { W } _ { T } ( \pi , \hat { \alpha } )$   
14: until πˆ converges.   
15: Return: $\hat { \pi } .$

Assumption 4.2(i) imposes a uniform convergence rate $( L _ { \infty }$ norm) for the nuisance estimators. Kallus et al. (2022) and Zhou et al. (2023) assume $L _ { 2 }$ convergence rate for discrete treatment. In our continuous treatment setting, however, the $L _ { \infty }$ requirement is a necessary theoretical cost to ensure that estimation remains robust across the entire continuous treatment manifold, which is consistent with Ai et al. (2026). We can construct a continuum estimator of $m _ { 0 }$ by applying the GRF framework to (9)–(10). According to Theorem 5 in Athey et al. (2019), this yields the desired convergence rate of $o _ { p } ( T ^ { - 2 \gamma _ { m } } )$ under their regularity conditions, by setting the subsample size s used to grow each tree to be $s = T ^ { \beta }$ with $\beta < 1 - 2 \gamma _ { m }$ . To estimate the conditional probability $\pi _ { 0 } ,$ one can use the methods in Colangelo & Lee (2026) and Cattaneo et al. (2024). Assumption 4.2 (ii) and (iii) share the same properties as true functions $\pi _ { 0 }$ and $m _ { 0 }$ . Specifically, we can employ a smoothed version of the forest (e.g., using a smooth kernel for leaf membership) to ensure it. We also discuss a corresponding second-order bounded derivative assumption when $s = 2$ in Assumption 2.4.

Before giving the theorem, we measure the complexity of the policy class Π using pseudo-dimension. Pseudo dimension generalizes the classical VC-dimension from binaryvalued functions to real-valued functions. Assumption 4.3 is mild and is satisfied by a vast range of parametric policy classes used in practice, such as linear policies, deep neural networks (Bartlett et al., 2019). This assumption is used to derive the Rademacher complexity bound, see Lemma C.6.

Assumption 4.3 (Policy Class Complexity). The policy class Π has a finite pseudo-dimension, denoted as Pdim(Π).

Theorem 4.4. Suppose Assumption 2.1-2.4, 3.2, 4.2, 4.3 hold, Given $\epsilon > 0 , h > 0 ,$ and a finite pseudo-dimension policy class $d _ { \Pi } : = P d i m ( \Pi )$ , we have with probability at least $1 - 7 \epsilon ,$

$$
\begin{array} { r l } & { \mathrm { R e g } ( \hat { \pi } ) \lesssim \overline { { \alpha } } \Big ( \log \Big ( \displaystyle \frac 1 h \Big ) \cdot \Big ( \frac { d _ { \Pi } } { T h / L } + \sqrt { \frac { d _ { \Pi } } { T h / L } } } \\ & { + \sqrt { \displaystyle \frac { \log ( 1 / \epsilon ) } { T h / L } } + \frac { \log ( 1 / \epsilon ) } { T h / L } \Big ) + T ^ { - \gamma _ { \pi } - \gamma _ { m } } + h ^ { s } \Big ) . } \end{array}
$$

As shown in Theorem 4.4, with high probability, the distributionally robust regret of the learned policy is upper bounded by $\tilde { \mathcal { O } } _ { p } ( T ^ { - s / ( 2 s + 1 ) } )$ with the optimal bandwidth $h = \Theta ( T ^ { - 1 / ( \bar { 2 s } + 1 ) } )$ , where $\tilde { \mathcal { O } }$ hides logarithmic factors in T. This rate only requires a not strong condition $\gamma _ { \pi } + \gamma _ { m } \geq$ $s / ( 2 s + 1 )$ in Assumption 4.2. A very recent study by Leung et al. (2025) also investigates the distributionally robust policy learning problem in continuous action space. They employ the IPW estimator (2) under $s \ : = \ : 2$ assumption, and the suboptimal regret bound they derived is $\mathcal { O } _ { p } ( T ^ { - 1 / 4 } )$ with their optimal bandwidth $h = \Theta ( T ^ { - 1 / 8 } )$ While our DR estimator can naturally degenerate to their IPW estimator by setting the outcome regression $\hat { m } _ { 0 }$ to zero. Hence, we establish a sharper regret bound not only for our DR but also for their IPW. This also aligns with rates achieved in non-robust continuous treatment settings (Ai et al., 2026).

## 5. Experiments

We evaluate our proposed OPE/L methods in both simulated and real-world contextual pricing environments.

## 5.1. Simulation Experiment

The simulated setup is as follows. Context vectors $X \in \mathbb { R } ^ { 3 }$ are sampled from $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { 3 } )$ . Historical prices are generated according to a linear logging policy $\pi _ { 0 } ,$ where $P ~ \mid ~ X ~ \sim ~ \mathcal { N } ( \mu _ { 0 } ( X ) , \sigma _ { 0 } ^ { 2 } )$ with $\sigma _ { 0 } ~ = ~ 1 . 0 , ~ \mu _ { 0 } ( X ) ~ =$ $\beta _ { 0 } + \beta _ { 1 } ^ { \top } X , \beta _ { 0 } = 1 . 0 , \beta _ { 1 } = [ 0 . 8 , 1 . 0 , 0 . 0 ] ^ { \top }$ , clipped to $\mathcal { P } = [ 0 . 5 , 3 . 0 ]$ . The true customer valuation is $f ( X ) =$ $\theta _ { 0 } + \theta _ { 1 } ^ { \top } X + \theta _ { 2 } X _ { 1 } X _ { 2 } , \ \theta _ { 0 } = 1 . 5 , \ \theta _ { 1 } = [ 1 . 2 , 1 . 8 ] ^ { \top } , \ \theta _ { 2 } =$ $0 . 3 ,$ where $( X _ { 1 } , X _ { 2 } )$ are informative features and $X _ { 3 }$ acts as a nuisance feature. The binary purchase outcomes are $Y = \mathbf { 1 } \{ f ( X ) + Z \geq P \}$ , where $Z \sim \mathrm { L o g i s t i c } ( 0 , 0 . 2 )$ follows an infinitely differentiable logistic distribution with bounded derivatives. The reward is $R = P \cdot Y$

First, we compare our proposed $\mathrm { L D R ^ { 2 } O ^ { 2 } }$ PE-CP (Algorithm 1) against the DRO-IPW baseline (2) from Leung et al. (2025). The target policy we seek to evaluate is a linear policy $\pi _ { \mathrm { t a r g e t } } ( X ) = w _ { 1 } ^ { \top } X + w _ { 0 }$ with $w _ { \mathrm { t a r g e t } } = ( w _ { 0 } , w _ { 1 } )$ where w captures the intercept. Policies are trained with uncertainty radius $\delta _ { \mathrm { t r a i n } } = 0 . 2$ and evaluated under test radii $\delta \in \{ 0 . 1 , 0 . 2 , 0 . 3 \}$ , and T is the training sample size.. We use a Gaussian kernel $\begin{array} { r } { K ( u ) = \frac { 1 } { \sqrt { 2 \pi } } \exp ( - u ^ { \bar { 2 } } / 2 ) } \end{array}$ with different bandwidth. We compare different bandwidths, (i) the OPE-optimal bandwidth $h = \Theta ( T ^ { - 1 / 5 . 0 } )$ under $s = 2$ in Kallus & Zhou (2018); Leung et al. (2025); (ii) our undersmoothing bandwidth $h = \bar { \Theta ( } T ^ { - 1 / ( 2 s + 1 - 0 . 1 ) } )$ under $s =$ 1, 2. Furthermore, we apply these choices with a Silvermantype scaling constant, $\mathrm { i . e . , } h = 1 . 0 6 \cdot \mathrm { s t d _ { o } } \cdot \mathrm { T } ^ { - 1 / 5 . 0 }$ and $h = 1 . 0 6 \cdot \mathrm { s t d } _ { p } \cdot T ^ { - 1 / ( 2 s + 1 - 0 . 1 ) }$ , where ${ \mathrm { s t d } } _ { p }$ is the sample standard deviation of observed prices. Both methods use Ridge regression to estimate the logging policy by fitting $\mu _ { 0 } ( X )$ and using the residual standard deviation as $\sigma _ { 0 }$ . Outcome regressions are estimated by forest-induced weights (10) (via the econml package, 20 trees, max depth 10) to ensure consistency with the OPL setup. All models are fitted with $L = 5$ fold cross-fitting, and averaged over 20 random seeds. Shaded regions show 90% confidence intervals.

Figure 1 shows the mean squared error (MSE) of the estimated distributionally robust policy value $R _ { \delta } .$ . We observe that $\mathrm { L D R ^ { 2 } O ^ { 2 } P E { \mathrm { - } } C P }$ outperforms the DRO-IPW baseline, as long as the sample size T is sufficiently large. However, in the non-asymptotic regime $( T < 1 0 0 0 ) , \mathrm { L D R ^ { 2 } O ^ { 2 } P E \mathrm { - } C P }$ may exhibit less stability compared to DRO-IPW. Nevertheless, $\mathrm { L D R ^ { 2 } O ^ { 2 } P E { \mathrm { - } } C P }$ exhibits a faster convergence rate of MSE decrease and offers a significant improvement over the baseline. Overall, $\mathrm { L D R ^ { 2 } O ^ { 2 } P E { \mathrm { - } } C P }$ provides the efficacy of the localized technique even under varying adversarial environments and different bandwidth.

Next, we compare our proposed $\mathrm { C D R ^ { 2 } O ^ { 2 } P I }$ -CP (Algorithm 2) against the non-robust DR method from Ai et al. (2026) and the DRO-IPW learning method from Leung et al. (2025). The target policy class is $\{ \pi ( X ) = w _ { 1 } ^ { \top } X + w _ { 0 } : w =$ $( w _ { 0 } , w _ { 1 } ) , \| w \| _ { \infty } \leq 2 \}$ . The continuum of outcome regressions $\{ \hat { m } _ { 0 } ( x , p ; \alpha ) ; \alpha \}$ is estimated according to $( 9 ) \substack { - } ( 1 0 )$ using the same econml setup as in OPE. Optimization uses Adam (learning rate $= 3 \times 1 0 ^ { - 3 } )$ in Line 13 and bounded optimization for α in Line 12.

Figure 2 reports the distributionally robust policy value $R _ { \delta }$ of policies learned in the distributionally robust OPL task. For fair comparison, all methods, including the non-robust DR baseline from Ai et al. (2026), are evaluated using the worst-case robust metric $R _ { \delta } ( \pi )$ , regardless of whether robustness was optimized during training. The non-robust DR baseline exhibits substantially wider confidence intervals, indicating high sensitivity to random seeds and brittleness under distributional shifts. In contrast, both robust methods produce much more stable estimates. Moreover, our $\mathrm { C D R ^ { 2 } O ^ { 2 } P L { - } C P }$ not only achieves the highest robust values but also yields slightly tighter confidence intervals than DRO-IPW for large sample sizes. This reflects the statistical efficiency gain of doubly robust estimation over IPW-based methods, even when propensity scores are estimated reliably (e.g., via Ridge regression). Overall, $\mathrm { C D R ^ { 2 } O ^ { 2 } P L { - } C P }$ delivers more stable learning and higher worst-case rewards across different levels of distributional robustness.

![](images/87023093ef2ac68569820b0d5f0002d7ec7cd2004db2eb5ecd40a968ddc632ac.jpg)

(a) h = Θ(T<sup>−1/2.9</sup>)  
![](images/997183e6aac2aab8b7d0c03e791908b7e8acf6d157b52f0acd9d38db7b669df9.jpg)

(b) h = Θ(T<sup>−1/4.9</sup>)  
![](images/1aab60cea962b9556eb48c25a46215c8252cd0abacc4283f8888e86bef2732fb.jpg)  
(c) $h = \Theta ( T ^ { - 1 / 5 . 0 } )$  
Figure 1. Comparison of $\mathrm { L D R ^ { 2 } O ^ { 2 } P E { \cdot } C P }$ against DRO-IPW in the distributionally robust OPE task. Results are shown for different bandwidth choices and test environments $\delta \in \{ 0 . 1 , 0 . 2 , 0 . 3 \}$ with $\delta _ { \mathrm { t r a i n } } = 0 . 2$ and 2500 test samples. The x-axis is the training sample size T, and the y-axis is the mean squared error (MSE) of the estimated distributionally robust value $\dot { R } _ { \delta }$

## 5.2. Empirical Experiment - The Expedia Study

We study offline contextual pricing on the dataset of the Expedia Personalized Sort Competition (Adam et al., 2013) hosted by Kaggle in conjunction with Expedia.. Each record is a search impression for a hotel listing, containing visitor and search context, hotel attributes, competitor prices, the logged list price, and whether the listing was booked. Following the offline pricing formulation in Section 2, we map each impression to a triplet $( x _ { t } , p _ { t } , y _ { t } ) \colon x _ { t } \in \mathbb { R } ^ { 4 5 }$ collects numeric covariates (visitor history, search features, hotel characteristics, and competitor rates); $p _ { t }$ is the logged price in USD; and $y _ { t }$ is the booking indicator. To create distribution shifts, we split the training and test data based on hotel star rating prop starrating $\in \{ 1 , 2 , 3 , 4 , 5 \}$ . We train using only prop starrating ∈ {1, 2, 3} and test only prop starrating $\in \{ 4 , 5 \}$

We learn the non-robust linear policy $\hat { \pi } _ { \mathrm { L i n } }$ (Ai et al., 2026), distributionally robust linear policy πˆ<sub>DRO−IPW</sub> (Leung et al., 2025) and our proposed $\hat { \pi } _ { \mathrm { C D R ^ { 2 } O ^ { 2 } P L - C P } }$ in the linear policy class $\Pi _ { \mathrm { i n } }$ . Each policy $\pi _ { w } ( x ) \in \Pi _ { \mathrm { l i n } }$ is parameterized by a vector $w = ( w _ { 0 } , w _ { 1 } ) \in \mathbb { R } ^ { 4 6 }$ , and the mapping $\pi : \mathcal { X }  [ p _ { \mathrm { m i n } } , p _ { \mathrm { m a x } } ]$ is defined as

$$
\begin{array} { r } { \Pi _ { \operatorname* { l i n } } \triangleq \big \{ \pi _ { w } ( x ) = p _ { \operatorname* { m i n } } + \big ( p _ { \operatorname* { m a x } } - p _ { \operatorname* { m i n } } \big ) \sigma \big ( w _ { 1 } ^ { \top } x + w _ { 0 } \big ) \big \} , } \end{array}
$$

where $\sigma ( \cdot )$ is the logistic link. We fix the uncertainty radius $\delta _ { \mathrm { t r a i n } } = 0 . 1$ used in the training procedure and the training size $T = 5 0 0$ . To characterize adversarial performance on Expedia, we adopt an empirical KL-adversarial stress test following Si et al. (2023). We fix a deployment test pool of $T ^ { \prime } ~ = ~ 2 { , } 5 0 0$ triplets from the training dataset and generate $M = 1 0$ replicates. For each $j \in [ M ]$ , we draw a bootstrap batch $\{ ( \overline { { x _ { t } ^ { ( j ) } } } , p _ { t } ^ { ( j ) } , y _ { t } ^ { ( j ) } ) \} _ { t = 1 } ^ { T ^ { \prime } }$ with replacement from this pool, compute $\alpha _ { j } ^ { * } ( \pi )$ at test radius $\delta = 0 . 2$ by Lemma 2.8, and form weights $w _ { t } ^ { ( j ) }$ ∝ $K ( ( p _ { t } ^ { ( j ) } - \pi ( x _ { t } ^ { ( j ) } ) / h ) )$ hπˆ<sub>0</sub> $( p _ { t } ^ { ( j ) } | x _ { t } ^ { ( j ) } )$ exp(−p<sup>(j)</sup><sub>t</sub> y<sup>(j)</sup><sub>t</sub> /α<sup>∗</sup><sub>j</sub> (π)). We then obtain a new dataset $\{ \tilde { x } _ { t } ^ { ( j ) } , \tilde { p } _ { t } ^ { ( j ) } , \tilde { y } _ { t } ^ { ( j ) } \} _ { t = 1 } ^ { T ^ { \prime } }$ by resampling $T ^ { \prime }$ triplets according to $\{ w _ { t } ^ { ( j ) } \}$ and evaluate

$$
\hat { R } _ { \mathrm { a d v } } ( \pi ) = \operatorname* { m i n } _ { 1 \le j \le M } \frac { 1 } { T ^ { \prime } } \sum _ { t = 1 } ^ { T ^ { \prime } } \hat { r } \big ( \tilde { x } _ { t } ^ { ( j ) } , \pi ( \tilde { x } _ { t } ^ { ( j ) } ) \big ) ,
$$

where $\scriptstyle { { \hat { r } } } ( x , p )$ is a linear outcome regression for $\mathbb { E } [ r | X =$ $x , P = p ]$ , fitted on the training set. Table 2 reports the mean and standard error over 20 random repeats. We find that our learned linear policy outperforms $\hat { \pi } _ { \mathrm { L i n } }$ and πˆ<sub>DRO−IPW</sub>.

## 6. Conclusion

In this paper, we present a doubly robust framework for distributionally robust OPE/L in continuous action settings. Our framework protects against distributional shifts across environments. For evaluation, $\mathrm { L D R ^ { 2 } O ^ { 2 } P E { \mathrm { - } } C P }$ uses localization to avoid the intractability of optimizing over a continuum of outcome functions. It fits only three regressions, as tractable as standard non-robust DR, while achieving semiparametric efficiency. For learning, $\mathrm { C D R ^ { 2 } O ^ { 2 } P I }$ -CP fits a continuum of regressions with data-driven weights, smoothly adapting to continuous pricing. By leveraging the latent smoothness of demand noise, we overcome the discontinuity in purchase outcomes and derive a finite-sample regret bound of $\tilde { \mathcal { O } } _ { p } ( T ^ { - s / ( 2 s + 1 ) } )$ for $s = 1 , 2$ under mild product rate assumptions. This bound represents an improvement over the $\bar { \mathcal { O } } _ { p } ( T ^ { - 1 / 4 } )$ rate in prior work on DRO methods and aligns with rates in non-robust approaches.

Comprehensive simulations demonstrate that our methods exhibit strong robustness to distributional shifts. Although we adopt the KL-divergence, future research could explore other measures, such as Wasserstein distance, to offer better geometric protections in specific scenarios. Additionally, developing more scalable optimization routines to handle high-dimensional state spaces would further broaden the practical utility of our framework in large-scale applications.

![](images/0241d2b38e714c1e4b897a3e352d73af0d56fab3fcb4672e0418e0920441b99b.jpg)

(a) h = Θ(T<sup>−1/2.9</sup>)  
![](images/c61001b8134b7632a9bb1c8c4ab2e92eb99d307f76ff2e87f1e7b4827ff7aa16.jpg)

(b) h = Θ(T<sup>−1/4.9</sup>)  
![](images/0224d5e14b97dccbe3c76d0d599c4739f6d438d0881657fab473348f0ab55a39.jpg)  
(c) h = Θ(T<sup>−1/5.0</sup>)  
Figure 2. Comparison of $\mathrm { C D R ^ { 2 } O ^ { 2 } P I }$ -CP against the non-robust DR and DRO-IPW in the distributionally robust OPL task. Results are shown for different bandwidth choices and test environments $\delta \in \{ 0 . 1 , 0 . 2 , 0 . 3 \}$ with $\delta _ { \mathrm { t r a i n } } ~ = ~ 0 . 2$ and 2500 test samples. The x-axis is the training sample size $T ,$ , and the y-axis is the distributionally robust policy value $R _ { \delta }$

Table 2. Comparison of robust performance in Expedia study.
<table><tr><td>Method</td><td> $\hat { R } _ { \mathrm { a d v } }$ </td></tr><tr><td> $\hat { \pi } _ { \mathrm { L i n } }$ </td><td> $6 . 2 0 1 \pm 0 . 6 8 3$ </td></tr><tr><td> $\hat { \pi } _ { \mathrm { D R O - I P W } }$ </td><td> $6 . 2 5 4 \pm 0 . 8 7 3$ </td></tr><tr><td> $\hat { \pi } _ { \mathrm { C D R ^ { 2 } O ^ { 2 } P L - C P } }$ </td><td> ${ \bf 6 . 3 5 0 \pm 0 . 5 9 8 }$ </td></tr></table>

## Acknowledgements

This work was supported in part by the National Natural Science Foundation of China [Grants 72394363/72394360, and 72571132], and the Postgraduate Research & Practice Innovation Program of Jiangsu Province [Grant KYCX25 0112].

## Impact Statement

Our research introduces a doubly robust framework for distributionally robust policy evaluation/learning in continuous action spaces. This work primarily aims to contribute to the methodological progress in machine learning and causal inference. It enhances the reliability of data-driven decisionmaking under environmental shifts, which is crucial for building safer and more dependable systems in domains like resource allocation and economic policy analysis. As with any policy learning algorithm, when applied to realworld scenarios such as dynamic pricing, the objective function should ensure that it does not inadvertently lead to discriminatory outcomes or unfair treatment of specific subpopulations. We encourage practitioners to conduct fairness audits when deploying such algorithms in socially sensitive contexts.

## References

Adam, Hamner, B., Friedman, D., and SSA Expedia. Personalize expedia hotel searches - icdm 2013. https://kaggle.com/competitions/ expedia-personalized-sort, 2013. Kaggle.

Ai, C., Fang, Y., and Xie, H. Data-driven policy learning for continuous treatments. Journal ofEconometrics, 253: 106170, 2026.

Anthony, M. and Bartlett, P. L. Neural Network Learning: Theoretical Foundations. Cambridge University Press, 2009.

Athey, S. and Wager, S. Policy learning with observational data. Econometrica, 89(1):133–161, 2021.

Athey, S., Tibshirani, J., and Wager, S. Generalized random forests. The Annals ofStatistics, 47(2):1148–1178, 2019.

Ban, G.-Y. and Keskin, N. B. Personalized dynamic pricing with machine learning: High-dimensional features and heterogeneous elasticity. Management Science, 67(9): 5549–5568, 2021.

Bartlett, P. L., Harvey, N., Liaw, C., and Mehrabian, A. Nearly-tight vc-dimension and pseudodimension bounds for piecewise linear neural networks. Journal ofMachine Learning Research, 20(63):1–17, 2019.

Ben-Tal, A., Den Hertog, D., De Waegenaere, A., Melenberg, B., and Rennen, G. Robust solutions of optimization problems affected by uncertain probabilities. Management Science, 59(2):341–357, 2013.

Bertsimas, D. and Kallus, N. From predictive to prescriptive analytics. Management Science, 66(3):1025–1044, 2020.

Blanchet, J., Lu, M., Zhang, T., and Zhong, H. Double pessimism is provably efficient for distributionally robust offline reinforcement learning: Generic algorithm and robust partial coverage. In Advances in Neural Information Processing Systems, volume 36, pp. 66845–66859, 2023.

Bu, J., Simchi-Levi, D., and Wang, C. Context-based dynamic pricing with separable demand models. Management Science, 2025. doi: 10.1287/mnsc.2022.02260.

Carranza, A. G. and Athey, S. Robust offline policy learning with observational data from multiple sources. In International Conference on Artificial Intelligence and Statistics, volume 258, pp. 4897–4905. PMLR, 2025.

Cattaneo, M. D., Chandak, R., Jansson, M., and Ma, X. Boundary adaptive local polynomial conditional density estimators. Bernoulli, 30(4):3193 – 3223, 2024.

Cevid, D., Michel, L., Naf, J., B¨ uhlmann, P., and Mein-¨ shausen, N. Distributional random forests: Heterogeneity adjustment and multivariate distributional regression. Journal of Machine Learning Research, 23(333):1–79, 2022.

Chen, E., Chen, X., Gao, L., and Li, J. Dynamic contextual pricing with doubly non-parametric random utility models. arXiv preprint arXiv:2405.06866, 2024.

Chen, N. and Gallego, G. Nonparametric pricing analytics with customer covariates. Operations Research, 69(3):974–984, 2021. doi: 10.1287/opre.2020.2016. URL https://doi.org/10.1287/opre.2020. 2016.

Chernozhukov, V., Demirer, M., Lewis, G., and Syrgkanis, V. Semi-parametric efficient policy learning with continuous actions. In Advances in Neural Information Processing Systems, volume 32, pp. 15065–15075, 2019.

Cohen, M. C., Lobel, I., and Paes Leme, R. Feature-based dynamic pricing. Management Science, 66(11):4921– 4943, 2020. doi: 10.1287/mnsc.2019.3485. URL https: //doi.org/10.1287/mnsc.2019.3485.

Colangelo, K. and Lee, Y.-Y. Double debiased machine learning nonparametric inference with continuous treatments. Journal ofBusiness & Economic Statistics, 44(1): 67–79, 2026.

Colnet, B., Mayer, I., Chen, G., Dieng, A., Li, R., Varoquaux, G., Vert, J.-P., Josse, J., and Yang, S. Causal inference methods for combining randomized trials and observational studies: A review. Statistical science, 39 (1):165–191, 2024.

Delage, E. and Ye, Y. Distributionally robust optimization under moment uncertainty with application to data-driven problems. Operations research, 58(3):595–612, 2010.

Den Boer, A. V. Dynamic pricing and learning: Historical origins, current research, and new directions. Surveys in operations research and management science, 20(1): 1–18, 2015.

Dud´ık, M., Langford, J., and Li, L. Doubly robust policy evaluation and learning. In International Conference on Machine Learning, pp. 1097–1104, 2011. URL https://icml.cc/2011/papers/ 554\_icmlpaper.pdf.

Fan, J., Guo, Y., and Yu, M. Policy optimization using semiparametric models for dynamic pricing. Journal of the American Statistical Association, 119(545):552–564, 2024.

Fang, Y., Ridder, G., and Xie, H. Semiparametric efficiency in policy learning with general treatments. arXiv preprint arXiv:2512.19230, 2025a.

Fang, Y., Xi, J., and Xie, H. Model selection for multivaluedtreatment policy learning in observational studies. Journal of Business & Economic Statistics, 43(4):897–909, 2025b.

Gong, X., You, W., and Zhang, J. Minimax optimality in contextual dynamic pricing with general valuation models. Operations Research, 2025. doi: 10.1287/opre.2025. 1779.

Guo, Y., Liu, H., Yue, Y., and Liu, A. Distributionally robust policy evaluation under general covariate shift in contextual bandits. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https:// openreview.net/forum?id=R7PReNELww.

Haussler, D. Sphere packing numbers for subsets of the Boolean n-cube with bounded Vapnik-Chervonenkis dimension. Journal ofCombinatorial Theory, Series A, 69 (2):217–232, 1995.

Horvitz, D. G. and Thompson, D. J. A generalization of sampling without replacement from a finite universe. Journal ofthe American Statistical Association, 47(260):663–685, 1952.

Hu, Y., Kallus, N., and Mao, X. Smooth contextual bandits: Bridging the parametric and nondifferentiable regret regimes. Operations Research, 70(6):3261–3281, 2022.

Hu, Z. and Hong, L. J. Kullback-Leibler divergence constrained distributionally robust optimization. Available at Optimization Online, 2013.

Imbens, G., Kallus, N., Mao, X., and Wang, Y. Long-term causal inference under persistent confounding via data combination. Journal of the Royal Statistical Society Series B: Statistical Methodology, 87(2):362–388, 2025.

Javanmard, A. and Nazerzadeh, H. Dynamic pricing in high-dimensions. Journal ofMachine Learning Research, 20(9):1–49, 2019.

Jiang, N. and Li, L. Doubly robust off-policy value evaluation for reinforcement learning. In International conference on machine learning, pp. 652–661. PMLR, 2016.

Jin, Y., Yang, Z., and Wang, Z. Is pessimism provably efficient for offline RL? In International Conference on Machine Learning, volume 139, pp. 5084–5096. PMLR, 2021.

Jin, Y., Egami, N., and Rothenhausler, D. Beyond reweight-¨ ing: On the predictive role of covariate shift in effect generalization. Proceedings ofthe National Academy of Sciences, 122(45):e2427181122, 2025a.

Jin, Y., Ren, Z., Yang, Z., and Wang, Z. Policy learning “without” overlap: Pessimism and generalized empirical Bernstein’s inequality. The Annals of Statistics, 53(4): 1483–1512, 2025b.

Kallus, N. Balanced policy evaluation and learning. In Advances in Neural Information Processing Systems, volume 31, pp. 8895–8906, 2018.

Kallus, N. and Mao, X. Stochastic optimization forests. Management Science, 69(4):1975–1994, 2023.

Kallus, N. and Zhou, A. Policy evaluation and optimization with continuous treatments. In International Conference on Artificial Intelligence and Statistics, pp. 1243–1251. PMLR, 2018.

Kallus, N., Mao, X., Wang, K., and Zhou, Z. Doubly robust distributionally robust off-policy evaluation and learning. In International Conference on Machine Learning, pp. 10598–10632. PMLR, 2022.

Kallus, N., Mao, X., and Uehara, M. Localized debiased machine learning: Efficient inference on quantile treatment effects and beyond. Journal of Machine Learning Research, 25(16):1–59, 2024.

Kennedy, E. H., Ma, Z., McHugh, M. D., and Small, D. S. Non-parametric methods for doubly robust estimation of continuous treatment effects. Journal of the Royal Statistical Society Series B: Statistical Methodology, 79 (4):1229–1245, 2017.

Khosravi, K., Lewis, G., and Syrgkanis, V. Non-parametric inference adaptive to intrinsic dimension. In Proceedings of the First Conference on Causal Learning and Reasoning, pp. 373–389. PMLR, 2022.

Kitagawa, T. and Tetenov, A. Who should be treated? Empirical welfare maximization methods for treatment choice. Econometrica, 86(2):591–616, 2018.

Lee, H., Lee, J., Choi, Y., Jeon, W., Lee, B.-J., Noh, Y.- K., and Kim, K.-E. Local metric learning for off-policy evaluation in contextual bandits with continuous actions. In Advances in Neural Information Processing Systems, volume 35, pp. 3913–3925, 2022.

Leung, C. H., Huang, Y., Li, Y., and Wu, Q. Distributionally robust policy evaluation and learning for continuous treatment with observational data. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 18209–18217, 2025.

Luo, Y., Sun, W. W., and Liu, Y. Distribution-free contextual dynamic pricing. Mathematics of Operations Research, 49(1):599–618, 2024.

Mbakop, E. and Tabord-Meehan, M. Model selection for treatment choice: Penalized welfare maximization. Econometrica, 89(2):825–848, 2021.

Miao, S., Chen, X., Chao, X., Liu, J., and Zhang, Y. Contextbased dynamic pricing with online clustering. Production and Operations Management, 31(9):3559–3575, 2022.

Mu, T., Chandak, Y., Hashimoto, T. B., and Brunskill, E. Factored DRO: Factored distributionally robust policies for contextual bandits. In Advances in Neural Information Processing Systems, volume 35, pp. 8318–8331, 2022.

Nambiar, M., Simchi-Levi, D., and Wang, H. Dynamic learning and pricing with model misspecification. Management Science, 65(11):4980–5000, 2019.

Oprescu, M., Syrgkanis, V., and Wu, Z. S. Orthogonal random forest for causal inference. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings ofMachine Learning Research, pp. 4932–4941. PMLR, 2019.

Shah, V., Johari, R., and Blanchet, J. Semi-parametric dynamic contextual pricing. In Advances in Neural Information Processing Systems, volume 32, 2019.

Si, N., Zhang, F., Zhou, Z., and Blanchet, J. Distributionally robust batch contextual bandits. Management Science, 69 (10):5772–5793, 2023.

Swaminathan, A. and Joachims, T. Batch learning from logged bandit feedback through counterfactual risk minimization. Journal of Machine Learning Research, 16(1): 1731–1755, 2015.

Tang, J., Qi, Z., Fang, E., and Shi, C. Offline feature-based pricing under censored demand: A causal inference approach. Manufacturing & Service Operations Management, 27(2):535–553, 2025.

van der Vaart, A. W. Asymptotic Statistics. Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, 1998.

van der Vaart, A. W. and Wellner, J. A. Weak convergence and empirical processes: With applications to statistics. Springer, 1st edition, 1996.

Wainwright, M. J. High-Dimensional Statistics: A Non-Asymptotic Viewpoint. Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, 2019.

Wang, J., Ren, Z., Zhan, R., and Zhou, Z. Distributionally robust policy learning under concept drifts. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 64244–64281. PMLR, 2025.

Wang, T., Chen, N., and Wang, C. Contextual optimization under covariate shift: A robust approach by intersecting Wasserstein balls. arXiv preprint arXiv:2406.02426, 2024.

Ye, Z., Zhang, Z., Zhang, D. J., Zhang, H., and Zhang, R. Deep learning-based causal inference for large-scale combinatorial experiments: Theory and empirical evidence. Management Science, 2025. doi: 10.1287/mnsc.2024. 04625.

Zhan, R., Hadad, V., Hirshberg, D. A., and Athey, S. Offpolicy evaluation via adaptive weighting with data from contextual bandits. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, pp. 2125–2135. ACM, 2021.

Zhan, R., Ren, Z., Athey, S., and Zhou, Z. Policy learning with adaptively collected data. Management Science, 70 (8):5270–5297, 2024.

Zhang, Z., Zeng, Z., Zhan, R., and Zhang, D. Personalized policy learning through discrete experimentation: Theory and empirical evidence. Available at SSRN, 2025. URL http://dx.doi.org/10. 2139/ssrn.5277082.

Zhang, Z., Zeng, Z., Zhan, R., and Zhang, D. Personalized policy learning through discrete experimentation: Theory and empirical evidence. arXiv preprint arXiv:2602.05099, 2026.

Zhao, Y.-Q., Zeng, D., Laber, E. B., Song, R., Yuan, M., and Kosorok, M. R. Doubly robust learning for estimating

individualized treatment with censored data. Biometrika, 102(1):151–168, 2015.

Zhou, H., Huang, R., Li, S., Jiang, G., Zheng, J., Cheng, B., and Lin, W. Decision focused causal learning for direct counterfactual marketing optimization. In Proceedings ofthe 30th ACM SIGKDD conference on knowledge discovery and data mining, pp. 6368–6379, 2024.

Zhou, Z., Athey, S., and Wager, S. Offline multi-action policy learning: Generalization and optimization. Operations Research, 71(1):148–183, 2023.

## A. Proof of Results in Section 2

Proof of Lemma 2.8. By Theorem 1 of Hu & Hong (2013), the strong duality holds:

$$
\operatorname* { i n f } _ { \mathbf { P } \in \mathcal { U } _ { \mathbf { P } _ { 0 } } ( \delta ) } \mathbb { E } _ { \mathbf { P } } [ r ( X , \pi ( X ) , Z ) ] = \operatorname* { s u p } _ { \alpha \geq 0 } \left\{ - \alpha \log \mathbb { E } _ { \mathbf { P } _ { 0 } } \left[ \exp \left( - \frac { r ( X , \pi ( X ) , Z ) } { \alpha } \right) \right] - \alpha \delta \right\} .
$$

For brevity, let $r \triangleq r ( X , \pi ( X ) , Z )$ . To prove that $\varphi ( \pi , \alpha )$ is strictly concave for $\alpha > 0 .$ , we analyze its second derivative.

$$
\frac { \partial ^ { 2 } } { \partial \alpha ^ { 2 } } \varphi ( \pi , \alpha ) = \frac 1 { \alpha ^ { 3 } \mathbb { E } _ { \mathbf { P } _ { 0 } } \left[ \exp \left( - r / \alpha \right) \right] } \left( \frac { ( \mathbb { E } _ { \mathbf { P } _ { 0 } } \left[ r \exp \left( - r / \alpha \right) \right] ) ^ { 2 } } { \mathbb { E } _ { \mathbf { P } _ { 0 } } \left[ \exp \left( - r / \alpha \right) \right] } - \mathbb { E } _ { \mathbf { P } _ { 0 } } \left[ r ^ { 2 } \exp \left( - r / \alpha \right) \right] \right) .
$$

By the Cauchy-Schwarz inequality,

$$
\begin{array} { r } { \mathbb { E } _ { \mathbf { P } _ { 0 } } [ r \exp ( - r / \alpha ) ] ^ { 2 } \le \mathbb { E } _ { \mathbf { P } _ { 0 } } [ r ^ { 2 } \exp ( - r / \alpha ) ] \mathbb { E } _ { \mathbf { P } _ { 0 } } [ \exp ( - r / \alpha ) ] . } \end{array}
$$

Since $\mathbb { E } _ { \mathbf { P } _ { 0 } } [ \exp ( - r / \alpha ) ] > 0$ , we have $\begin{array} { r } { \frac { \partial ^ { 2 } } { \partial \alpha ^ { 2 } } \varphi ( \pi , \alpha ) \leq 0 } \end{array}$ . The Cauchy-Schwarz inequality holds strictly as an equality if and only if r is almost surely constant. Under Assumption $2 . 3 , \mathbf { P } _ { 0 } ( Y = 0 ) \in [ \underline { { \rho } } , \overline { { \rho } } ]$ , ensuring that r is non-constant almost surely. Thus, $\textstyle \frac { \partial ^ { 2 } } { \partial \alpha ^ { 2 } } \varphi ( \pi , \alpha ) < 0$ , which guarantees strict concavity

Because $r \in [ 0 , 1 ]$ , we obtain $R _ { \delta } ( \pi ) \geq 0$ . Furthermore, $r \leq 1$ implies $\mathbb { E } _ { \mathbf { P } _ { 0 } } [ \exp ( - r / \alpha ) ] \ge \exp ( - 1 / \alpha )$ , yielding the upper bound $\varphi ( \pi , \alpha ) \leq 1 - \alpha \delta$ . By strong duality, the optimal dual solution $\alpha ^ { * } ( \pi )$ must satisfy:

$$
0 \leq R _ { \delta } ( \pi ) = \varphi ( \pi , \alpha ^ { * } ( \pi ) ) \leq 1 - \alpha ^ { * } ( \pi ) \delta \implies \alpha ^ { * } ( \pi ) \leq \frac { 1 } { \delta } \triangleq \bar { \alpha } .
$$

Note that $\operatorname* { l i m } _ { \alpha \to 0 } \varphi ( \pi , \alpha ) = 0$ . If the optimal solution were $\alpha ^ { * } ( \pi ) = 0$ , the maximum robust value would be exactly 0, which contradicts the assumption $R _ { \delta } ( \pi ) > 0$ . Therefore, $\alpha ^ { * } ( \pi ) \in ( 0 , \bar { \alpha } ]$

We now prove that the condition $\delta < -$ log ρ is sufficient to guarantee $R _ { \delta } ( \pi ) > 0$ . Since li $\begin{array} { r } { \Omega _ { \alpha \to 0 ^ { + } } \varphi ( \pi , \alpha ) = 0 } \end{array}$ , it suffices to show that lim $\operatorname* { i n f } _ { \alpha \to 0 ^ { + } } \frac { \partial } { \partial \alpha } \varphi ( \pi , \alpha ) > 0$ . The first derivative of $\varphi ( \pi , \alpha )$ for $\alpha > 0$ is given by:

$$
\frac { \partial } { \partial \alpha } \varphi ( \pi , \alpha ) = - \log \mathbb { E } _ { \mathbf { P } _ { 0 } } \left[ \exp { ( - r / \alpha ) } \right] - \frac { \mathbb { E } _ { \mathbf { P } _ { 0 } } \left[ r / \alpha \exp { ( - r / \alpha ) } \right] } { \mathbb { E } _ { \mathbf { P } _ { 0 } } \left[ \exp { ( - r / \alpha ) } \right] } - \delta .
$$

We evaluate the limits of the expectation terms as $\alpha  0 ^ { + }$ . For the first term, since $r \geq 0 .$ , we have $\exp ( - r / \alpha ) \le 1$ . By the Dominated Convergence Theorem, we can interchange the limit and the expectation:

$$
\operatorname* { l i m } _ { \alpha \to 0 ^ { + } } \mathbb { E } _ { \mathbf { P } _ { 0 } } \left[ \exp \left( - r / \alpha \right) \right] = \mathbb { E } _ { \mathbf { P } _ { 0 } } \left[ \operatorname* { l i m } _ { \alpha \to 0 ^ { + } } \left( \mathbf { 1 } \{ Y = 0 \} + \mathbf { 1 } \{ Y = 1 \} \exp \left( - r / \alpha \right) \right) \right] = \mathbf { P } _ { 0 } ( Y = 0 ) .
$$

For the second term, since $\begin{array} { r l r l } { \operatorname* { l i m } _ { x \to \infty } x e ^ { - x } } & { { } = } & { 0 } \end{array}$ , applying the Dominated Convergence Theorem yields $\begin{array} { r } { \operatorname* { l i m } _ { \alpha  0 ^ { + } } \mathbb { E } _ { \mathbf { P } _ { 0 } } [ r / \alpha \exp ( - r / \alpha ) ] = 0 } \end{array}$ . Substituting these limits back into the derivative gives

$$
\operatorname* { l i m } _ { \alpha \to 0 ^ { + } } { \frac { \partial } { \partial \alpha } } \varphi ( \pi , \alpha ) = - \log ( \mathbf { P } _ { 0 } ( Y = 0 ) ) - \delta .
$$

By Assumption 2.3, we know $- \log ( \mathbf { P } _ { 0 } ( r = 0 ) ) \geq - \log \overline { { \rho } } .$ Therefore, if $\delta < - \log \overline { { \rho } } ,$ we obtain

$$
\operatorname* { l i m } _ { \alpha \to 0 ^ { + } } \frac { \partial } { \partial \alpha } \varphi ( \pi , \alpha ) \ge - \log \overline { { \rho } } - \delta > 0 ,
$$

which completes the proof.

## B. Proof of Results in Section 3

We first introduce the assumptions for the asymptotic analysis.

Assumption B.1. The oracle logging policy $\pi _ { 0 } ( p | x )$ is three-times differentiable with respect to $p \in \mathcal P$ for any $x \in \mathcal { X }$ Furthermore, there exists a compact neighborhood A of the true dual parameter $\alpha ^ { * }$ , such that for all $\alpha \in { \mathcal { A } }$ , the nuisance function function $\hat { \lambda } _ { 1 } ^ { ( \ell ) } ( x , p ; \alpha )$ has bounded first-order and second-order derivative with respect to $p \in \mathcal P$ for any $x \in \mathcal { X }$ and $\ell \in [ L ]$ , in probability, i.e.,

$$
\operatorname* { s u p } _ { x , p , \alpha \in \mathcal { A } } \left\| \frac { \partial ^ { s } } { \partial p ^ { s } } \hat { \lambda } _ { 1 } ^ { ( \ell ) } ( x , p ; \alpha ) \right\| _ { \infty } = \mathcal { O } _ { p } ( 1 ) , \quad s = 1 , 2 .
$$

And $\hat { \lambda } _ { 2 } ^ { ( \ell ) } ( x , p ) \ge \eta$ for all $x , p \in \mathcal { X } \times \mathcal { P } , \ell \in [ L ] .$

Assumption B.2. There exist a compact neighborhood $B _ { \theta }$ of $\theta ^ { * }$ and a neighborhood $B _ { \zeta }$ of $\zeta ^ { * }$ such that for all $\theta \in B _ { \theta }$ and $\zeta \in B _ { \zeta }$

(i) $\nabla _ { \boldsymbol { \theta } } \psi ( V ; \boldsymbol { \theta } , \zeta )$ is Lipschitz continuous in θ such that for any $\begin{array} { r } { \theta _ { 1 } , \theta _ { 2 } \ \in \ B _ { \theta } , \ \lVert \nabla _ { \theta } \psi ( V ; \theta _ { 1 } , \zeta ) \ - \nabla _ { \theta } \psi ( V ; \theta _ { 2 } , \zeta ) \rVert \ \leq } \end{array}$ $L ( V , h ) \| \theta _ { 1 } - \theta _ { 2 } \|$ , where the Lipschitz constant satisfies $\mathbb { E } [ L ( V , h ) ] = \mathcal { O } ( 1 )$

(ii) The conditional second moment is uniformly bounded, $\begin{array} { r } { \operatorname* { s u p } _ { x , p } \operatorname* { s u p } _ { ( \theta , \zeta ) \in \mathcal { B } _ { \theta } \times \mathcal { B } _ { \zeta } } \mathbb { E } [ \| \nabla _ { \theta } \psi ( V ; \theta , \zeta ) \| ^ { 2 } | X = x , P = p ] \le } \end{array}$ $\mathcal { O } ( 1 )$

(iii) For any fixed α in a compact neighborhood A of $\alpha ^ { * }$ , the map $\zeta \mapsto \mathbb { E } [ \nabla _ { \theta } \psi ( V ; \theta ^ { * } , \zeta ) ]$ is Lipschitz continuous in a neighborhood $B _ { \zeta }$ of $\begin{array} { r } { \zeta ^ { * } , \mathrm { i } . \mathrm { e } . , \| \mathbb { E } [ \nabla _ { \theta } \psi ( V ; \theta ^ { * } , \zeta _ { 1 } ( x , p ; \alpha ) ) ] - \mathbb { E } [ \nabla _ { \theta } \psi ( V ; \theta ^ { * } , \zeta _ { 2 } ( x , p ; \alpha ) ) ] \| \le C _ { \psi } \| \zeta _ { 1 } - \zeta _ { 2 } \| } \end{array}$ for some Lipschitz constant $C _ { \psi } < \infty$

Assumption B.2 is tailored to adapt Assumption 2 in Kallus et al. (2024) to the continuous pricing setting. Specifically, Assumption ${ \bf B } . 2 ( \mathrm { i } ) – ( \mathrm { i i } )$ directly corresponds to Assumption 2 (ii) and (iv-v) in Kallus et al. (2024), ensuring the existence and stability of the Jacobian and covariance matrices for the joint moment equations. Assumption B.2 (iii) is a localized version of the Frechet differentiability requirement discussed in Proposition 5 of´ Kallus et al. (2024), which is the fundamental pillar for the localization technique to achieve semiparametric efficiency using only a preliminary estimate ${ \hat { \alpha } } _ { i n i t }$

With the regressions $m _ { 0 }$ and $m _ { 1 }$ , we can show that the Gateaux derivatives of $\mathbb { E } [ \psi ( V ; \theta , \zeta ) ]$ are zero as $h  0$ , when evaluated at $\theta ^ { * } = ( \alpha ^ { * } , W _ { 0 } ^ { * } , W _ { 1 } ^ { * } , R _ { \delta } ( \pi ) ) , \lambda _ { 1 } ( \cdot ; \alpha ) = \lambda _ { 1 } ^ { * } ( \cdot ; \alpha ^ { * } )$ and $\lambda _ { 2 } ( \cdot ) = \lambda _ { 2 } ^ { * } ( \cdot )$ respectively.

Lemma B.3 (Neyman Orthogonality in S2 of Online supplement (Colangelo & Lee, 2026)). Under Assumption B.1, B.2, the Gateaux derivative ofthe moment mapping at the oracle parameter $\theta ^ { * }$ and nuisance $\zeta ^ { * }$ in the direction $\Delta \zeta = \zeta - \zeta ^ { * }$ is given by

$$
| d M ( \theta ^ { * } , \zeta ^ { * } ; \Delta _ { \zeta } ) | = \bigg | \frac { \partial } { \partial \tau } \mathbb { E } [ \psi ( V ; \theta ^ { * } , \zeta ^ { * } + \tau \Delta _ { \zeta } ) ] \Big | _ { \tau = 0 } \bigg | = \mathcal { O } _ { p } ( h ^ { s } ) ,\tag{11}
$$

where $s = 1$ under Lipschitz continuity and $s = 2$ under second-order bounded derivative in Assumption B.1. Consequently, as $h  0$ , the moment condition isfirst-order insensitive to the estimation errors ofthe nuisances.

ProofofTheorem 3.3. We decompose the proof into three main components: (i) control of errors introduced by dual variable localization and nuisance estimation; (ii) sensitivity and stability analysis of the joint moment equations (Jacobian consistency); and (iii) the linear representation via empirical process theory.

Step 1: Nuisance Estimation and Localization Error. We first examine the sensitivity of the outcome mode $\lambda _ { 1 } ^ { * } ( x , p ; \alpha )$ with respect to α. The derivative of $\lambda _ { 1 } ^ { * } ( x , p ; \alpha )$ is

$$
\frac { \partial } { \partial \alpha } \lambda _ { 1 } ^ { * } ( x , p ; \alpha ) = \left( \begin{array} { c } { \frac { 1 } { \alpha ^ { 2 } } \exp ( - p / \alpha ) \mathbf { P } ( \boldsymbol { Y } = 1 | \boldsymbol { X } = x , P = p ) } \\ { \frac { p } { \alpha ^ { 2 } } \exp ( - p / \alpha ) \mathbf { P } ( \boldsymbol { Y } = 1 | \boldsymbol { X } = x , P = p ) } \\ { 0 } \\ { 0 } \end{array} \right) ,
$$

is uniformly bounded by $\frac { 4 } { p _ { m i n } ^ { 2 } e ^ { 2 } }$ for $p \in [ p _ { \operatorname* { m i n } } , 1 ] , x \in \mathcal { X }$ and $\alpha > 0 .$ , which implies that

$$
| \lambda _ { 1 } ^ { * } ( x , p ; \hat { \alpha } _ { i n i t } ) - \lambda _ { 1 } ^ { * } ( x , p ; \alpha ^ { * } ) | \leq \frac { 4 } { p _ { m i n } ^ { 2 } e ^ { 2 } } | \hat { \alpha } _ { i n i t } - \alpha ^ { * } | .\tag{12}
$$

Combining this with Assumption 3.1, the total error in the outcome nuisance function under the localization strategy is bounded by

$$
| \lambda _ { 1 } ( x , p ; { \hat { \alpha } } _ { i n i t } ) - \lambda _ { 1 } ^ { * } ( x , p ; \alpha ^ { * } ) | \leq | \lambda _ { 1 } ( x , p ; { \hat { \alpha } } _ { i n i t } ) - \lambda _ { 1 } ^ { * } ( x , p ; { \hat { \alpha } } _ { i n i t } ) | + | \lambda _ { 1 } ^ { * } ( x , p ; { \hat { \alpha } } _ { i n i t } ) - \lambda _ { 1 } ^ { * } ( x , p ; \alpha ^ { * } ) |
$$

$$
\begin{array} { r l r l } & { } & & { \leq o _ { p } ( T ^ { - \rho _ { m } } ) + \displaystyle \frac { 4 } { p _ { m i n } ^ { 2 } e ^ { 2 } } | \hat { \alpha } _ { i n i t } - \alpha ^ { * } | } \\ & { } & & { \leq o _ { p } ( T ^ { - \rho _ { m } } + T ^ { - \rho _ { \alpha } } ) , } \\ & { } & & { = o _ { p } ( T ^ { - \rho _ { m } \wedge \rho _ { \alpha } } ) , } \\ & { } & & { | \lambda _ { 2 } ( x , p ) - \lambda _ { 2 } ^ { * } ( x , p ) | \leq o _ { p } ( T ^ { - \rho _ { \pi } } ) . } \end{array}\tag{13}
$$

and

(14)

Therefore, we obtain $\| \zeta ( \cdot ; \hat { \alpha } _ { i n i t } ) - \zeta ^ { * } ( \cdot ; \alpha ^ { * } ) \| _ { \infty } \leq o _ { p } ( T ^ { - \rho _ { m } \wedge \rho _ { \alpha } } + T ^ { - \rho _ { \pi } } )$

Step 2: Score Decomposition and Neyman Orthogonality. For each fold $\ell \in \mathsf { \Gamma } [ L ]$ , denote $\hat { \zeta } ^ { ( \ell ) } ( x , p ) =$ $( \hat { \lambda } _ { 1 } ^ { ( \ell ) } ( x , p ; \hat { \alpha } _ { i n i t } ^ { ( \ell ) } ) , \hat { \lambda } _ { 2 } ^ { ( \ell ) } ( x , p ) ) , \Delta _ { \lambda _ { 1 } } ^ { ( \ell ) } ( x , p ) = \hat { \lambda } _ { 1 } ^ { ( \ell ) } ( x , p ; \hat { \alpha } _ { i n i t } ^ { ( \ell ) } ) - \lambda _ { 1 } ^ { * } ( x , p ; \alpha ^ { * } )$ and $\Delta _ { \lambda _ { 2 } } ^ { ( \ell ) } ( x , p ) = \hat { \lambda } _ { 2 } ^ { ( \ell ) } ( x , p ) { - } \lambda _ { 2 } ^ { * } ( x , p )$ . Utilizing the cross-fitting structure, we decompose

$$
\begin{array} { r l } & { \sqrt { n _ { \ell } \hbar } \displaystyle \frac { 1 } { n _ { \ell } } \sum _ { t \in \mathcal { Z } _ { \ell } } \bigg ( \psi ( v _ { t } ; \theta ^ { * } , \hat { \zeta } ^ { ( \ell ) } ( v _ { t } ) ) - \psi ( v _ { t } ; \theta ^ { * } , \zeta ^ { * } ) \bigg ) } \\ & { = \sqrt { \displaystyle \frac { \hbar } { n _ { \ell } } } \sum _ { t \in \mathcal { Z } _ { \ell } } \bigg \{ \left[ \hat { \lambda } _ { 1 } ^ { ( \ell ) } ( x _ { t } , \pi ( x _ { t } ) ; \hat { \alpha } _ { \iota \wedge i } ^ { ( \ell ) } ) - \lambda _ { 1 } ^ { * } ( x _ { t } , \pi ( x _ { t } ) ; \alpha ^ { * } ) \right] } \\ & { \qquad + \left[ \frac { K ( \frac { \rho _ { t } - \pi ( x _ { t } ) } { \hbar } ) } { \hbar \hat { \lambda } _ { 2 } ^ { ( \ell ) } ( x _ { t } , p _ { t } ) } ( U ( p _ { t } y _ { t } ; \alpha ^ { * } ) - \hat { \lambda } _ { 1 } ^ { ( \ell ) } ( x _ { t } , p _ { t } ; \hat { \alpha } _ { i \wedge i } ^ { ( \ell ) } ) ) - \frac { K ( \frac { \rho _ { t } - \pi ( x _ { t } ) } { \hbar } ) } { \hbar \lambda _ { 2 } ^ { * } ( x _ { t } , p _ { t } ) } ( U ( p _ { t } y _ { t } ; \alpha ^ { * } ) - \lambda _ { 1 } ^ { * } ( x _ { t } , p _ { t } ; \alpha ^ { * } ) ) \right] \bigg \} } \\ & { = \sqrt { \displaystyle \frac { \hbar } { n _ { \ell } } } \sum _ { t \in \mathcal { Z } _ { \ell } } \left\{ \mathcal { T } _ { 1 , t } ^ { ( \ell ) } + \mathcal { T } _ { 2 , t } ^ { ( \ell ) } + \mathcal { T } _ { D R , t } ^ { ( \ell ) } + \mathcal { T } _ { p r o d , t } ^ { ( \ell ) } \right\} , } \end \end{array}
$$

where

$$
\begin{array} { r l } & { T _ { 1 , \epsilon } ^ { ( \delta ) } = \Bigg [ \Delta _ { x _ { 1 } } ^ { ( \beta ) } ( \kappa _ { \epsilon } , \pi ( x _ { t } ) ) - \frac { K ( \frac { p _ { \epsilon } - \pi ( \kappa ) } { h } ) } { h \lambda _ { 2 } ^ { ( \delta ) } ( \kappa _ { t } , p _ { t } ) } \Delta _ { \lambda _ { 1 } } ^ { ( \theta ) } ( x _ { t } , p _ { t } ) \Bigg ] - \mathbb { E } \Bigg [ \Delta _ { \lambda _ { 1 } } ^ { ( \epsilon ) } ( X , \pi ( X ) ) - \frac { K ( \frac { p - \pi ( X ) } { h } ) } { h \lambda _ { 2 } ^ { ( \delta ) } ( X , p ) } \Delta _ { \lambda _ { 1 } } ^ { ( \theta ) } ( X , P ) \Bigg ] T _ { \epsilon } ^ { \epsilon } \Bigg ] } \\ & { T _ { 2 , \epsilon } ^ { ( \delta ) } = \frac { K ( \frac { p - \pi ( \kappa ) - \pi ( \kappa ) } { h } ) } { h } ( U ( p _ { \epsilon } ) g _ { \epsilon } , \alpha ^ { * } ) - \lambda _ { 1 } ^ { * } ( \kappa _ { \epsilon } , p _ { \epsilon } ; \alpha ^ { * } ) ) \Bigg ( \frac { 1 } { \tilde { \lambda } _ { 2 } ^ { ( \epsilon ) } ( \kappa _ { t } , p _ { t } ) } - \frac { 1 } { \tilde { \lambda } _ { 2 } ^ { ( \epsilon ) } ( \kappa _ { t } , p _ { t } ) } \Bigg ) } \\ & { \qquad - \mathbb { E } \Bigg [ \frac { K ( \frac { p - \kappa ( X ) } { h } ) } { h } ( U ( P \gamma ; \alpha ^ { * } ) - \lambda _ { 1 } ^ { * } ( X , P ; \alpha ^ { * } ) ) \Bigg ( \frac { 1 } { \tilde { \lambda } _ { 2 } ^ { ( \epsilon ) } ( X , P ) } - \frac { 1 } { \tilde { \lambda } _ { 2 } ^ { * } ( X , P ) } \Bigg ) \Bigg | T _ { \epsilon } ^ { \epsilon } \Bigg ] } \\ &  T _ { D R } ^ { ( \delta ) } = \mathbb { E } \Bigg [ \Delta _  \end{array}
$$

By cross-fitting, $\hat { \zeta } ^ { ( \ell ) } = ( \hat { \lambda } _ { 1 } ^ { ( \ell ) } , \hat { \lambda } _ { 2 } ^ { ( \ell ) } )$ are trained on the out-of-fold data $\mathcal { T } _ { \ell } ^ { c }$ and are thus fixed functions. For the first term $\mathcal { T } _ { 1 , t } ^ { ( \ell ) }$ , since the summands are i.i.d. and mean-zero conditional on $\mathcal { T } _ { \ell } ^ { c }$ , the variance of the sum is

$$
\begin{array} { r l } {  { \operatorname { V a r } ( \sqrt { \frac { h } { n _ { \xi } } } \sum _ { t \in \mathbb { Z } _ { \xi } } \mathcal { T } _ { 1 , t } ^ { ( \ell ) } \Big | \mathcal { T } _ { \ell } ^ { c } ) = h \cdot \operatorname { V a r } ( \mathcal { T } _ { 1 , t } ^ { ( \ell ) } \big | \mathcal { T } _ { \ell } ^ { c } ) } } \\ & { \leq 2 h \mathbb { E } [ ( \Delta _ { \lambda _ { 1 } } ^ { ( \ell ) } ( X , \pi ( X ) ) ) ^ { 2 } \bigg | \mathcal { T } _ { \ell } ^ { c } ] + 2 h \mathbb { E } [ ( \frac { K ( \frac { P - \pi ( X ) } { h } ) } { h \lambda _ { 2 } ^ { * } ( X , P ) } \Delta _ { \lambda _ { 1 } } ^ { ( \ell ) } ( X , P ) ) ^ { 2 } \bigg | \mathcal { T } _ { \ell } ^ { c } ] } \\ & { \leq 2 h \| \Delta _ { \lambda _ { 1 } } ^ { ( \ell ) } \| _ { \infty } ^ { 2 } + 2 \| \Delta _ { \lambda _ { 1 } } ^ { ( \ell ) } \| _ { \infty } ^ { 2 } \cdot \mathbb { E } [ \int _ { P } \frac { K ^ { 2 } ( \frac { P - \pi ( X ) } { h } ) } { h } d p ] } \\ & { = 2 h \| \Delta _ { \lambda _ { 1 } } ^ { ( \ell ) } \| _ { \infty } ^ { 2 } + 2 \| \Delta _ { \lambda _ { 1 } } ^ { ( \ell ) } \| _ { \infty } ^ { 2 } \cdot \mathbb { E } [ \int _ { P } K ^ { 2 } ( u ) d u ] } \end{array}
$$

$$
\begin{array} { r l } & { \leq 2 h \| \Delta _ { \lambda _ { 1 } } ^ { ( \ell ) } \| _ { \infty } ^ { 2 } + 2 C _ { 3 } \| \Delta _ { \lambda _ { 1 } } ^ { ( \ell ) } \| _ { \infty } ^ { 2 } } \\ & { = o _ { p } ( T ^ { - 2 \rho _ { m } } + T ^ { - 2 \rho _ { \alpha } } ) = o _ { p } ( 1 ) , } \end{array}
$$

where the last equality follows from (13). By Chebyshev’s inequality, it follows that

$$
\sqrt { \frac { h } { n _ { \ell } } } \sum _ { t \in \mathcal { T } _ { \ell } } \mathcal { T } _ { 1 , t } ^ { ( \ell ) } = o _ { p } ( 1 ) .\tag{15}
$$

For the second term $\mathcal { T } _ { 2 , t } ^ { ( \ell ) }$ , since $\mathbb { E } [ U ( P Y ; \alpha ^ { * } ) | X , P ] = \lambda _ { 1 } ^ { * } ( X , P ; \alpha ^ { * } )$ , we get a similar variance bound as

$$
\begin{array} { r l } & { \mathrm { V a r } \left( \sqrt { \frac { h } { n _ { \ell } } } \displaystyle \sum _ { t \in \mathbb { Z } _ { \varepsilon } } T _ { 2 , t } ^ { ( \ell ) } \Big | T _ { \ell } ^ { c } \right) = h \cdot \mathbb { E } \left[ \frac { K ^ { 2 } \big ( \frac { P - \pi ( X ) } { h } \big ) } { h ^ { 2 } } ( U ( P Y ; \alpha ^ { * } ) - \lambda _ { 1 } ^ { * } ( X , P ; \alpha ^ { * } ) \big ) ^ { 2 } \left( \frac { 1 } { \hat { \lambda } _ { 2 } ^ { ( \ell ) } ( x _ { t } , p _ { t } ) } - \frac { 1 } { \lambda _ { 2 } ^ { * } ( x _ { t } , p _ { t } ) } \right) ^ { 2 } \Bigg | T _ { \ell } ^ { c } \right] } \\ & { \qquad \le \frac { C _ { 3 } } { \eta ^ { 3 } } \| \Delta _ { \lambda _ { 2 } } ^ { ( \ell ) } \| _ { \infty } ^ { 2 } = o _ { p } ( T ^ { - 2 \rho _ { \tau } } ) , } \end{array}
$$

where the last equality follows from (14). Then by Chebyshev’s inequality, it follows that

$$
\sqrt { \frac { h } { n _ { \ell } } } \sum _ { t \in \mathcal { T } _ { \ell } } \mathcal { T } _ { 2 , t } ^ { ( \ell ) } = o _ { p } ( 1 ) .\tag{16}
$$

For the third term $\mathcal { T } _ { D R } ^ { ( \ell ) }$ , we discuss in two cases. Case 1. First-order bounded derivative in Assumption B.1.

$$
\begin{array} { r l } & { \sqrt { n _ { \ell } h } \left| \mathcal { T } _ { D R } ^ { ( \ell ) } \right| = \sqrt { n _ { \ell } h } \left| \mathbb { E } \left[ \Delta _ { \lambda _ { 1 } } ^ { ( \ell ) } ( X , \pi ( X ) ) - \frac { K ( \frac { P - \pi ( X ) } { h } ) } { h \lambda _ { 2 } ^ { * } ( X , P ) } \Delta _ { \lambda _ { 1 } } ^ { ( \ell ) } ( X , P ) \middle | \mathcal { T } _ { \ell } ^ { \ell } \right] \right| } \\ & { \quad \quad \quad \quad = \sqrt { n _ { \ell } h } \left| \mathbb { E } _ { X } \left[ \Delta _ { \lambda _ { 1 } } ^ { ( \ell ) } ( X , \pi ( X ) ) - \int _ { \mathcal { P } } \frac { 1 } { h } K \left( \frac { p - \pi ( X ) } { h } \right) \Delta _ { \lambda _ { 1 } } ^ { ( \ell ) } ( X , p ) d p \right] \right| } \\ & { \quad \quad \quad \quad \quad \leq \sqrt { n _ { \ell } h } \left| \mathbb { E } _ { X } \left[ \int K ( u ) \Delta _ { \lambda _ { 1 } } ^ { ( \ell ) } ( X , \pi ( X ) ) - \Delta _ { \lambda _ { 1 } } ^ { ( \ell ) } ( X , \pi ( X ) + u h ) \right] d u \right] } \\ & { \quad \quad \quad \quad = \sqrt { n _ { \ell } h } \left| \mathbb { E } _ { X } \left[ \int K ( u ) u h \left\| \frac { \partial } { \partial p } \Delta _ { \lambda _ { 1 } } ^ { ( \ell ) } ( X , \tilde { p } ) \right\| _ { \infty } d u \right] \right| \leq \mathcal { O } _ { p } \left( \sqrt { n _ { \ell } h ^ { 3 } } \right) , } \end{array}\tag{17}
$$

where we use Assumption 3.2, Lipschitz continuity of $\Delta _ { \lambda _ { 1 } } ^ { ( \ell ) }$ in Assumption B.1 and 2.4, and the initial estimate $\hat { \alpha } _ { i n i t } =$ $\alpha ^ { * } + o _ { p } ( 1 )$ . Case 2. Second-order bounded derivative in Assumption B.1.

$$
\begin{array} { r l } & { \sqrt { n _ { \ell } h } \left| \mathcal { T } _ { D R } ^ { ( \ell ) } \right| = \sqrt { n _ { \ell } h } \left| \mathbb { E } \left[ \Delta _ { \lambda _ { 1 } } ^ { ( \ell ) } ( X , \pi ( X ) ) - \frac { K ( \frac { P - \pi ( X ) } { h } ) } { h \lambda _ { 2 } ^ { * } ( X , P ) } \Delta _ { \lambda _ { 1 } } ^ { ( \ell ) } ( X , P ) \middle | \mathcal { T } _ { \ell } ^ { \ell } \right] \right| } \\ & { \quad \quad \quad \quad = \sqrt { n _ { \ell } h } \left| \mathbb { E } _ { X } \left[ \Delta _ { \lambda _ { 1 } } ^ { ( \ell ) } ( X , \pi ( X ) ) - \int _ { \mathcal { P } } \frac { 1 } { h } K \left( \frac { p - \pi ( X ) } { h } \right) \Delta _ { \lambda _ { 1 } } ^ { ( \ell ) } ( X , p ) d p \right] \right| } \\ & { \quad \quad \quad \quad \quad \leq \sqrt { n _ { \ell } h } \left| \mathbb { E } _ { X } \left[ \int K ( u ) \Delta _ { \lambda _ { 1 } } ^ { ( \ell ) } ( X , \pi ( X ) ) - \Delta _ { \lambda _ { 1 } } ^ { ( \ell ) } ( X , \pi ( X ) + u h ) \right] d u \right] } \\ & { \quad \quad \quad \quad = \sqrt { n _ { \ell } h } \left| \mathbb { E } _ { X } \left[ \int K ( u ) \frac { ( u h ) ^ { 2 } } { 2 } \frac { \partial ^ { 2 } } { \partial p ^ { 2 } } \Delta _ { \lambda _ { 1 } } ^ { ( \ell ) } ( X , \tilde { p } ) d u \right] \right| \leq \mathcal { O } _ { p } \left( \sqrt { n _ { \ell } h ^ { 5 } } \right) . } \end{array}\tag{18}
$$

where we use Assumption 3.2 and a second-order Taylor expansion of the nuisance error $\Delta _ { \lambda _ { 1 } } ^ { ( \ell ) }$ around $\pi ( x )$ ,

$$
\Delta _ { \lambda _ { 1 } } ^ { ( \ell ) } ( x , \pi ( x ) + u h ) = \Delta _ { \lambda _ { 1 } } ^ { ( \ell ) } ( x , \pi ( x ) ) + ( u h ) \frac { \partial } { \partial p } \Delta _ { \lambda _ { 1 } } ^ { ( \ell ) } ( x , \pi ( x ) ) + \frac { ( u h ) ^ { 2 } } { 2 } \frac { \partial ^ { 2 } } { \partial p ^ { 2 } } \Delta _ { \lambda _ { 1 } } ^ { ( \ell ) } ( x , \tilde { p } ) ,
$$

where $\tilde { p }$ lies between $\pi ( x )$ and $\pi ( x ) + u h$ . And the last inequality directly follows from Assumption B.1 and the initia estimate $\hat { \alpha } _ { i n i t } = \alpha ^ { * } + o _ { p } ( 1 )$

For the fourth term $\mathcal { T } _ { p r o d , t } ^ { ( \ell ) } ,$ by using the change of variables $u = ( p - \pi ( x ) ) / h$ and (13)-(14), we reduce its expectation to

$$
\begin{array} { r } { \sqrt { T h } \left| \mathbb { E } \left[ \mathcal { T } _ { p r o d , t } ^ { ( \ell ) } \middle | \mathcal { Z } _ { \ell } ^ { c } \right] \right| \leq \mathcal { O } ( \sqrt { T h } \cdot \| \Delta _ { \lambda _ { 1 } } ^ { ( \ell ) } \| _ { \infty } \cdot \| \Delta _ { \lambda _ { 2 } } ^ { ( \ell ) } \| _ { \infty } ) \leq o _ { p } ( T ^ { \frac { 1 } { 2 } - \rho _ { \pi } - \rho _ { \alpha } \wedge \rho _ { m } } \cdot h ^ { 1 / 2 } ) = o _ { p } ( 1 ) , } \end{array}
$$

and the variance is bounded by

$$
\begin{array} { r l } & { \mathrm { V a r } \left( \sqrt { \frac { h } { n _ { \ell } } } \displaystyle \sum _ { t \in \mathcal { Z } _ { \ell } } \mathcal { T } _ { p r o d , t } ^ { ( \ell ) } \Big \vert \mathcal { Z } _ { \ell } ^ { c } \right) } \\ & { = h \cdot \mathrm { V a r } \left( \mathcal { T } _ { p r o d , t } ^ { ( \ell ) } \Big \vert \mathcal { Z } _ { \ell } ^ { c } \right) \leq h \cdot \mathbb { E } \left[ \left( \mathcal { T } _ { p r o d , t } ^ { ( \ell ) } \right) ^ { 2 } \Big \vert \mathcal { Z } _ { \ell } ^ { c } \right] \leq h \int \frac { 1 } { h ^ { 2 } } K ^ { 2 } \left( \frac { p - \pi ( x ) } { h } \right) \frac { \| \Delta _ { \lambda _ { 1 } } ^ { ( \ell ) } \| _ { \infty } ^ { 2 } \cdot \| \Delta _ { \lambda _ { 2 } } ^ { ( \ell ) } \| _ { \infty } ^ { 2 } } { \eta ^ { 3 } } d p d F _ { X } } \\ & { = \displaystyle \frac { 1 } { \eta ^ { 3 } } \int K ^ { 2 } ( u ) \| \Delta _ { \lambda _ { 1 } } ^ { ( \ell ) } \| _ { \infty } ^ { 2 } \cdot \| \Delta _ { \lambda _ { 2 } } ^ { ( \ell ) } \| _ { \infty } ^ { 2 } d u d F _ { X } \leq \frac { C _ { 3 } } { \eta ^ { 3 } } \| \Delta _ { \lambda _ { 1 } } ^ { ( \ell ) } \| _ { \infty } ^ { 2 } \cdot \| \Delta _ { \lambda _ { 2 } } ^ { ( \ell ) } \| _ { \infty } ^ { 2 } } \\ & { \leq o _ { p } ( T ^ { - 2 \rho _ { \tau } - 2 \rho _ { \alpha } } + T ^ { - 2 \rho _ { \tau } - 2 \rho _ { m } } ) = o _ { p } ( 1 ) . } \end{array}
$$

By applying Chebyshev’s inequality conditionally on $\mathcal { T } _ { \ell } ^ { c }$ , we obtain $\begin{array} { r } { \sqrt { n _ { \ell } h } \left| \frac { 1 } { n _ { \ell } } \sum _ { t \in \mathcal { Z } _ { \ell } } \mathcal { T } _ { p r o d , t } ^ { ( \ell ) } - \mathbb { E } \left[ \mathcal { T } _ { p r o d , t } ^ { ( \ell ) } \big | \mathcal { T } _ { \ell } ^ { c } \right] \right| = o _ { p } ( 1 ) . } \end{array}$ Furthermore, we get

$$
\left| \sqrt { \frac { h } { n _ { \ell } } } \sum _ { t \in \mathbb { Z } _ { \ell } } T _ { p r o d , t } ^ { ( \ell ) } \right| \leq \left| \sqrt { n _ { \ell } h } \left( \frac { 1 } { n _ { \ell } } \sum _ { t \in \mathbb { Z } _ { \ell } } T _ { p r o d , t } ^ { ( \ell ) } - \mathbb { E } \left[ T _ { p r o d , t } ^ { ( \ell ) } | \mathbb { Z } _ { \ell } ^ { c } \right] \right) \right| + \sqrt { n _ { \ell } h } \left| \mathbb { E } \left[ T _ { p r o d , t } ^ { ( \ell ) } | \mathbb { Z } _ { \ell } ^ { c } \right] \right| \leq o _ { p } ( 1 ) .\tag{19}
$$

To proceed, picking $\sqrt { T h ^ { 1 + 2 s } }  0$ and combining (13)-(19), we conclude that

$$
\sqrt { n _ { \ell } h } \frac { 1 } { n _ { \ell } } \sum _ { t \in \mathbb { Z } _ { \ell } } \Big ( \psi ( v _ { t } ; \theta ^ { * } , \hat { \zeta } ^ { ( \ell ) } ( v _ { t } ) ) - \psi ( v _ { t } ; \theta ^ { * } , \zeta ^ { * } ) \Big ) = o _ { p } ( 1 ) .\tag{20}
$$

Step 3: Jacobian Stability and Asymptotic Distribution. Denote $\theta ^ { * } = [ \alpha ^ { * } , W _ { 0 } ^ { * } , W _ { 1 } ^ { * } , R _ { \delta } ( \pi ) ] ^ { \top }$ . Since

$$
\begin{array} { l } { \displaystyle \mathbb { E } [ \psi ( V ; \theta , \zeta ^ { * } ) ] = \mathbb { E } \left[ \lambda _ { 1 } ^ { * } ( X , \pi ( X ) ; \alpha ) + \frac { K ( \frac { P - \pi ( X ) } { h } ) } { h \pi _ { 0 } ( P | X ) } \left( U ( P Y ; \alpha ) - \lambda _ { 1 } ^ { * } ( X , P ; \alpha ) \right) + G ( \theta ) \right] } \\ { \displaystyle \qquad = \mathbb { E } [ \lambda _ { 1 } ^ { * } ( X , \pi ( X ) ; \alpha ) ] + G ( \theta ) } \\ { \displaystyle \qquad \mathbb { E } [ \exp ( - \pi ( X ) Y ( \pi ( X ) ) / \alpha ) ] - W _ { 0 } } \\ { \displaystyle \qquad = \left( \mathbb { E } [ \pi ( X ) Y ( \pi ( X ) ) \exp ( - \pi ( X ) Y ( \pi ( X ) ) / \alpha ) ] - W _ { 1 } \right) } \\ { \displaystyle \qquad - \delta - \log W _ { 0 } - \frac { W _ { 1 } } { \alpha W _ { 0 } } } \\ { \displaystyle \qquad - R _ { \delta } ( \pi ) - \alpha \log W _ { 0 } - \alpha \delta } \end{array}
$$

following the proof of Theorem 3.2 in Kallus et al. (2022), by replacing r with $\pi ( X ) Y ( \pi ( X ) )$ ), the Jacobian is

$$
\begin{array} { r } { J ( \theta ) = \left[ \begin{array} { c c c c } { \frac { 1 } { \alpha ^ { 2 } } \mathbb { E } \left[ \pi ( X ) Y ( \pi ( X ) ) \exp ( - \pi ( X ) Y ( \pi ( X ) ) / \alpha ) \right] } & { - 1 } & { 0 } & { 0 } \\ { \frac { 1 } { \alpha ^ { 2 } } \mathbb { E } \left[ \pi ^ { 2 } ( X ) Y ^ { 2 } ( \pi ( X ) ) \exp ( - \pi ( X ) Y ( \pi ( X ) ) / \alpha ) \right] } & { 0 } & { - 1 } & { 0 } \\ { \frac { W _ { 1 } } { \alpha ^ { 2 } W _ { 0 } } } & { - \frac { 1 } { W _ { 0 } } + \frac { W _ { 1 } } { \alpha W _ { 0 } ^ { 2 } } } & { - \frac { 1 } { \alpha W _ { 0 } } } & { 0 } \\ { - \log W _ { 0 } - \delta } & { - \frac { \alpha } { W _ { 0 } } } & { 0 } & { - 1 } \end{array} \right] , } \end{array}\tag{21}
$$

and substituting in the optimal value, we have $J _ { \theta ^ { * } } = \mathbb { E } [ \nabla _ { \theta } \psi ( V ; \theta , \zeta ^ { * } ) ] | _ { \theta = \theta ^ { * } }$ is given by

$$
J ^ { * } = J _ { \theta ^ { * } } = \left( \begin{array} { c c c c } { { W _ { 1 } ^ { * } / \alpha ^ { * 2 } } } & { { - 1 } } & { { 0 } } & { { 0 } } \\ { { W _ { 2 } ^ { * } / \alpha ^ { * 2 } } } & { { 0 } } & { { - 1 } } & { { 0 } } \\ { { \frac { W _ { 1 } ^ { * } } { \alpha ^ { * 2 } W _ { 0 } ^ { * } } } } & { { - \frac { 1 } { W _ { 0 } ^ { * } } + \frac { W _ { 1 } ^ { * } } { \alpha ^ { * } W _ { 0 } ^ { 2 } } } } & { { - \frac { 1 } { \alpha ^ { * } W _ { 0 } ^ { * } } } } & { { 0 } } \\ { { - \log W _ { 0 } ^ { * } - \delta } } & { { - \frac { \alpha ^ { * } } { W _ { 0 } ^ { * } } } } & { { 0 } } & { { - 1 } } \end{array} \right) ,
$$

and $J ^ { * }$ is invertible with det $J ^ { * } = - \varphi ^ { ^ { \prime \prime } } ( \pi , \alpha ^ { * } ) > 0$ from Lemma 2.8, and the inverse of $J ^ { * }$ is given by

$$
J ^ { * - 1 } = - \frac { 1 } { \varphi ^ { \prime \prime } ( \pi , \alpha ^ { * } ) } \left( \begin{array} { c c c c } { - \frac { 1 } { W _ { 0 } ^ { * } } + \frac { W _ { 1 } ^ { * } } { \alpha ^ { * } W _ { 0 } ^ { * 2 } } } & { - \frac { 1 } { \alpha ^ { * } W _ { 0 } ^ { * } } } & { 1 } & { 0 } \\ { \frac { W _ { 2 } ^ { * } / \alpha ^ { * } - W _ { 1 } ^ { * } } { \alpha ^ { * 2 } W _ { 0 } ^ { * } } } & { - \frac { W _ { 1 } ^ { * } } { \alpha ^ { * 3 } W _ { 0 } ^ { * } } } & { \frac { W _ { 1 } ^ { * } } { \alpha ^ { * 2 } } } & { 0 } \\ { \frac { W _ { 2 } ^ { * } } { \alpha ^ { * 3 } W _ { 0 } ^ { * } } \left( \frac { W _ { 1 } ^ { * } } { W _ { 0 } ^ { * } } - \alpha ^ { * } \right) } & { - \frac { W _ { 1 } ^ { * 2 } } { \alpha ^ { * 3 } W _ { 0 } ^ { * 2 } } } & { \frac { W _ { 2 } ^ { * } } { \alpha ^ { * 2 } } } & { 0 } \\ { X _ { 1 } } & { \frac { X _ { 2 } } { \alpha ^ { * } W _ { 0 } ^ { * } } } & { - X _ { 2 } } & { - \varphi ^ { \prime \prime } ( \pi , \alpha ^ { * } ) } \end{array} \right) ,
$$

where X1 =  <sup>W∗</sup>1<sub>α∗W</sub>∗2 − <sup>1</sup><sub>W</sub>∗  (− log W<sup>∗</sup><sub>0</sub> − δ) − <sup>1</sup><sub>α∗W</sub>∗2 (W<sup>∗</sup><sub>1</sub> − W<sup>∗</sup><sub>2</sub> /α<sup>∗</sup>) and X2 = W<sup>∗</sup><sub>1</sub> /(α<sup>∗</sup>W<sup>∗</sup><sub>0</sub> ) + log W<sup>∗</sup><sub>0</sub> + δ .

Let $\hat { \theta } = [ \hat { \alpha } , \hat { W } _ { 0 } , \hat { W } _ { 1 } , \hat { R } _ { \mathrm { D R O } } ] ^ { \top }$ be the solution to the empirical equation

$$
\mathbb { E } _ { T } [ \psi ( V ; \theta , \hat { \zeta } ) ] \triangleq \frac { 1 } { T } \sum _ { \ell = 1 } ^ { L } \sum _ { t \in \mathcal { T } _ { \ell } } \psi ( v _ { t } ; \theta , \hat { \zeta } ^ { ( \ell ) } ) = \mathbf { 0 } .\tag{22}
$$

By second order Taylor expansion around $( \theta ^ { * } , \zeta ^ { * } )$

$$
\mathbf { 0 } = \mathbb { E } _ { T } [ \psi ( V ; \hat { \theta } , \hat { \zeta } ) ] = \mathbb { E } _ { T } [ \psi ( V ; \theta ^ { * } , \zeta ^ { * } ) ] + J ^ { * } ( \hat { \theta } - \theta ^ { * } ) + \mathbb { E } _ { T } [ \psi ( V ; \theta ^ { * } , \hat { \zeta } ) - \psi ( V ; \theta ^ { * } , \zeta ^ { * } ) ] + \mathcal { R } _ { T } ,\tag{23}
$$

where $\mathcal { R } _ { T }$ represents high-order Taylor remainders. Specifically,

$$
\mathcal { R } _ { T } = ( \hat { J } _ { T } ( \hat { \zeta } ) - J ^ { * } ) ( \hat { \theta } - \theta ^ { * } ) + \frac { 1 } { 2 } \nabla _ { \theta } ^ { 2 } \mathbb { E } _ { T } [ \psi ( V ; \hat { \theta } , \hat { \zeta } ) ] ( \hat { \theta } - \theta ^ { * } ) ^ { 2 } ,
$$

where $\tilde { \theta }$ lies between $\theta ^ { * }$ and $\hat { \theta } , \hat { J } _ { T } ( \hat { \zeta } ) : = \nabla _ { \theta } \mathbb { E } _ { T } [ \psi ( V ; \theta , \hat { \zeta } ) ] | _ { \theta } ,$ ∗ denotes the empirical Jacobian matrix evaluated with localized nuisance estimators.

Here we show the $\sqrt { T h } \mathcal { R } _ { T } = o _ { p } ( 1 )$ . First, we show the $\hat { J } _ { T } ( \hat { \zeta } ) = J ^ { * } + o _ { p } ( 1 )$

$$
\begin{array} { r l } & { | \hat { J } _ { T } ( \boldsymbol { \hat { \zeta } } ) - J ^ { * } | = \left| \frac { 1 } { T } \displaystyle \sum _ { \ell = 1 } ^ { L } \sum _ { \ell \in \mathcal { T } _ { \ell } } \Big ( \nabla _ { \theta } \psi ( v _ { \ell } ; \theta , \hat { \zeta } ^ { ( \ell ) } ) - \mathbb { E } [ \nabla _ { \theta } \psi ( V ; \theta , \zeta ^ { * } ) ] \Big ) _ { \theta = \theta ^ { * } } \right| } \\ & { \qquad \leq \displaystyle \frac { 1 } { L } \left| \frac { 1 } { | { \mathcal Z } _ { \ell } | } \displaystyle \sum _ { \ell = 1 } ^ { L } \sum _ { \ell \in \mathcal { T } _ { \ell } } \nabla _ { \theta } \psi ( v _ { \ell } ; \theta ^ { * } , \hat { \zeta } ^ { ( \ell ) } ) - \mathbb { E } [ \nabla _ { \theta } \psi ( V ; \theta ^ { * } , \hat { \zeta } ^ { ( \ell ) } ) ] \right| + \left| \mathbb { E } [ \nabla _ { \theta } \psi ( V ; \theta ^ { * } , \hat { \zeta } ^ { ( \ell ) } ) - \mathbb { E } [ \nabla _ { \theta } \psi ( V ; \theta ^ { * } , \zeta ^ { * } ) ] \right| } \end{array}
$$

By cross-fitting, $\hat { \zeta } ^ { ( \ell ) }$ is fixed given the training data $\mathcal { T } _ { \ell } ^ { c }$ . Due to $( 1 3 ) – ( 1 4 ) , \hat { \zeta } ^ { ( \ell ) } = \zeta ^ { * } + o _ { p } ( 1 )$ and using Assumption B.2(ii), the variance of average over $\mathcal { T } _ { \ell }$ samples scales as

$$
\mathrm { V a r } \left( \frac { 1 } { \vert \mathcal { Z } _ { \ell } \vert } \sum _ { t \in \mathcal { Z } _ { \ell } } \nabla _ { \theta } \psi ( v _ { t } ; \theta ^ { * } , \hat { \zeta } ^ { ( \ell ) } ) \bigg \vert \mathcal { Z } _ { \ell } ^ { c } \right) = \frac { 1 } { \vert \mathcal { Z } _ { \ell } \vert } \mathrm { V a r } \left( \nabla _ { \theta } \psi ( v _ { t } ; \theta ^ { * } , \hat { \zeta } ^ { ( \ell ) } ) \vert \mathcal { Z } _ { \ell } ^ { c } \right) \le \mathcal { O } _ { p } \left( \frac { 1 } { T h } \right)
$$

where we use the fact that $\begin{array} { r } { \mathbb { E } \left[ \frac { K ^ { 2 } ( \frac { P - \pi ( X ) } { h } ) } { h ^ { 2 } } \right] = \mathcal { O } ( \frac { 1 } { h } ) } \end{array}$ . Then by Chebyshev’s inequality, as $T h  \infty$ , we have

$$
\left| \frac { 1 } { | \mathscr { T } _ { \ell } | } \sum _ { \ell = 1 } ^ { L } \sum _ { t \in \mathscr { T } _ { \ell } } \nabla _ { \theta } \psi ( v _ { t } ; \theta ^ { * } , \hat { \zeta } ^ { ( \ell ) } ) - \mathbb { E } [ \nabla _ { \theta } \psi ( V ; \theta ^ { * } , \hat { \zeta } ^ { ( \ell ) } ) ] \right| = o _ { p } ( 1 ) ,\tag{24}
$$

Consider the mapping $\alpha \mapsto \mathbb { E } [ \nabla _ { \theta } \psi ( V ; \theta ^ { * } , \zeta ^ { * } ( x , p ; \alpha ) ) ]$ . Note that $\nabla _ { \boldsymbol { \theta } } \psi$ is composed of nuisance $\lambda _ { 1 } ^ { * }$ and $\nabla _ { \boldsymbol { \theta } } G ( \boldsymbol { \theta } )$ $\zeta ^ { * } ( x , p ; \alpha )$ is Lipschitz in α from (12). And $W _ { 0 } \ge \rho / 2$ by picking $\begin{array} { r } { h ^ { s } \ \le \ \frac { \rho } { 2 C _ { 1 } L _ { m } } \quad } \end{array}$ in (36) and (43), then $\nabla _ { \boldsymbol { \theta } } G ( \boldsymbol { \theta } )$ is uniformly bounded in the compact neighborhood A of $\alpha ^ { * }$ . Thus, the Jacobian of $\mathbb { E } [ \overleftarrow { \nabla } _ { \theta } ^ { m } \psi ( V ; \theta ^ { * } , \zeta ^ { * } ( x , p ; \alpha ) ) ]$ is Lipschitz in $\alpha$ with a bounded constant $C _ { l o c }$ . And using Assumption B.2 (iii), we have

$$
\begin{array} { r l } & { \left| \mathbb { E } [ \nabla _ { \theta } \psi ( V ; \theta ^ { * } , \hat { \zeta } ^ { ( \ell ) } ) - \mathbb { E } [ \nabla _ { \theta } \psi ( V ; \theta ^ { * } , \zeta ^ { * } ) ] \right| } \\ & { \leq \left| \mathbb { E } [ \nabla _ { \theta } \psi ( V ; \theta ^ { * } , \hat { \zeta } ^ { ( \ell ) } ( x , p ; \hat { \alpha } _ { i n i t } ^ { ( \ell ) } ) ) ] - \mathbb { E } [ \nabla _ { \theta } \psi ( V ; \theta ^ { * } , \zeta ^ { * } ( x , p ; \hat { \alpha } _ { i n i t } ^ { ( \ell ) } ) ) ] \right| } \\ & { \quad + \left| \mathbb { E } [ \nabla _ { \theta } \psi ( V ; \theta ^ { * } , \zeta ^ { * } ( x , p ; \hat { \alpha } _ { i n i t } ^ { ( \ell ) } ) ) ] - \mathbb { E } [ \nabla _ { \theta } \psi ( V ; \theta ^ { * } , \zeta ^ { * } ( x , p ; \alpha ^ { * } ) ) ] \right| } \\ & { \leq C _ { \Psi } \| \hat { \zeta } ^ { ( \ell ) } ( x , p ; \hat { \alpha } _ { i n i t } ^ { ( \ell ) } ) - \zeta ^ { * } ( x , p ; \hat { \alpha } _ { i n i t } ^ { ( \ell ) } ) \| _ { \infty } + C _ { l o c } \| \hat { \alpha } _ { i n i t } ^ { ( \ell ) } - \alpha ^ { * } \| _ { \infty } \leq o _ { p } ( 1 ) } \end{array}\tag{25}
$$

Combining (24) and (25), we have

$$
| \hat { J } _ { T } ( \boldsymbol { \hat { \zeta } } ) - J ^ { * } | = o _ { p } ( 1 ) .\tag{26}
$$

Second, by weak law of large numbers and Assumption B.2 (i), we have $\begin{array} { r } { \frac { 1 } { | \mathcal { T } _ { \ell } | } \sum _ { t \in \mathcal { T } _ { \ell } } L ( v _ { t } , h ) = \mathcal { O } _ { p } ( 1 ) } \end{array}$ . Thus,

$$
\begin{array} { r l } & { \displaystyle \left\| \nabla _ { \theta } ^ { 2 } \mathbb { E } _ { T } [ \psi ( V ; \tilde { \theta } , \hat { \zeta } ) ] \right\| = \left\| \frac { 1 } { T } \sum _ { \ell = 1 } ^ { L } \sum _ { t \in \mathcal { T } _ { \ell } } \nabla _ { \theta } ^ { 2 } \psi ( v _ { t } ; \tilde { \theta } , \hat { \zeta } ^ { ( \ell ) } ) \right\| } \\ & { \qquad \leq \displaystyle \frac { 1 } { T } \sum _ { \ell = 1 } ^ { L } \sum _ { t \in \mathcal { T } _ { \ell } } \left\| \nabla _ { \theta } ^ { 2 } \psi ( v _ { t } ; \tilde { \theta } , \hat { \zeta } ^ { ( \ell ) } ) \right\| \leq \displaystyle \frac { 1 } { T } \sum _ { \ell = 1 } ^ { L } \sum _ { t \in \mathcal { T } _ { \ell } } L ( v _ { t } , h ) = \mathcal { O } _ { p } ( 1 ) . } \end{array}\tag{27}
$$

Third, we show that $\lVert \hat { { \boldsymbol { \theta } } } - { \boldsymbol { \theta } } ^ { * } \rVert = \mathcal { O } _ { p } ( ( T h ) ^ { - 1 / 2 } )$ . By a first order Taylor expansion in (22) around $\theta ^ { * }$ , we have

$$
{ \bf 0 } = \mathbb { E } _ { T } [ \psi ( V ; \theta ^ { * } , \hat { \zeta } ) ] + \bar { J } _ { T } ( \hat { \theta } - \theta ^ { * } ) ,
$$

where $\begin{array} { r } { \bar { J } _ { T } = \int _ { 0 } ^ { 1 } \nabla _ { \theta } \mathbb { E } _ { T } [ \psi ( V ; \theta ^ { * } + u ( \hat { \theta } - \theta ^ { * } ) , \hat { \zeta } ) ] d u } \end{array}$ . Due to (26) and (27), we have $\bar { J } _ { T } = J ^ { * } + o _ { p } ( 1 )$ . Recalling that $J ^ { * }$ is invertible, then

$$
( \hat { \theta } - \theta ^ { * } ) = - ( J ^ { * } + o _ { p } ( 1 ) ) ^ { - 1 } \mathbb { E } _ { T } [ \psi ( V ; \theta ^ { * } , \hat { \zeta } ) ] .\tag{28}
$$

We can decompose the empirical expectation as follows

$$
\mathbb { E } _ { T } [ \psi ( V ; \theta ^ { * } , \hat { \zeta } ) ] = \mathbb { E } _ { T } [ \psi ( V ; \theta ^ { * } , \zeta ^ { * } ) ] + ( \mathbb { R } _ { T } - \mathbb { E } ) [ \psi ( V ; \theta ^ { * } , \hat { \zeta } ) - \psi ( V ; \theta ^ { * } , \zeta ^ { * } ) ] + \mathbb { E } [ \psi ( V ; \theta ^ { * } , \hat { \zeta } ) - \psi ( V ; \theta ^ { * } , \zeta ^ { * } ) ] .
$$

For the first term, using the fact tha $\operatorname { V a r } ( \psi ( V ; \theta ^ { * } , \zeta ^ { * } ) ) = \mathcal { O } ( 1 / h )$ , then $\operatorname { V a r } ( \mathbb { E } _ { T } [ \psi ] ) = O ( 1 / T h )$ . By the Central Limit Theorem, we have

$$
\mathbb { E } _ { T } [ \psi ( V ; \theta ^ { * } , \zeta ^ { * } ) ] = \mathcal { O } _ { p } ( ( T h ) ^ { - 1 / 2 } ) .\tag{29}
$$

For the second term, we evaluate the conditional variance Var $\Big ( ( \mathbb { E } _ { T } - \mathbb { E } ) [ \psi ( V ; \theta ^ { * } , \hat { \zeta } ^ { ( \ell ) } ) - \psi ( V ; \theta ^ { * } , \zeta ^ { * } ) ] \mid \mathcal { T } _ { \ell } ^ { c } \Big )$ . Following the variance scaling argument in Equation (A.5) in Colangelo & Lee (2026), we have

$$
\mathrm { V a r } \left( ( \mathbb { E } _ { T } - \mathbb { E } ) [ \psi ( V ; \theta ^ { \star } , \hat { \zeta } ^ { ( \ell ) } ) - \psi ( V ; \theta ^ { * } , \zeta ^ { * } ) ] \mid { \cal T } _ { \ell } ^ { c } \right) = { \cal O } _ { p } \left( \frac { \| \hat { \zeta } ^ { ( \ell ) } - \zeta ^ { * } \| _ { \infty } ^ { 2 } } { T h } \right) .
$$

Under the assumption that the nuisance estimators are consistent, $\| \hat { \zeta } ^ { ( \ell ) } - \zeta ^ { * } \| _ { \infty } = o _ { p } ( 1 )$ , and $T h  \infty$ , we have

$$
\begin{array} { r } { ( \mathbb { E } _ { T } - \mathbb { E } ) [ \psi ( V ; \theta ^ { * } , \hat { \zeta } ^ { ( \ell ) } ) - \psi ( V ; \theta ^ { * } , \zeta ^ { * } ) ] = o _ { p } ( ( T h ) ^ { - 1 / 2 } ) . } \end{array}\tag{30}
$$

For the third term, by the Neyman orthogonality B.3, we decompose the functional bias into

$$
\begin{array} { r l } & { \mathbb { E } [ \psi ( V ; \theta ^ { * } , \hat { \zeta } ) - \psi ( V ; \theta ^ { * } , \zeta ^ { * } ) ] = d M ( \zeta ^ { * } ; \hat { \zeta } - \zeta ^ { * } ) + \frac { 1 } { 2 } d ^ { 2 } M ( \bar { \zeta } ; \hat { \zeta } - \zeta ^ { * } ) \qquad ( \bar { \zeta } \mathrm { ~ l i e s ~ b e t w e e n ~ } \zeta ^ { * } \mathrm { ~ a n d ~ } \hat { \zeta } ) } \\ & { \qquad = \mathcal { O } _ { p } ( h ^ { s } \| \hat { \zeta } - \zeta ^ { * } \| _ { \infty } ) + \mathcal { O } _ { p } ( \| \hat { \lambda } _ { 1 } - \lambda _ { 1 } ^ { * } \| _ { \infty } \| \hat { \lambda } _ { 2 } - \lambda _ { 2 } ^ { * } \| _ { \infty } ) } \\ & { \qquad = \mathcal { O } _ { p } ( h ^ { s } ( T ^ { - \rho _ { m } \wedge \rho _ { \alpha } } + T ^ { - \rho _ { \pi } } ) ) + T ^ { - ( \rho _ { m } \wedge \rho \alpha ) - \rho _ { \pi } } ) . \qquad \mathrm { ( b y ~ ( 1 3 ) , ( 1 4 ) , ( 3 1 ) ) } } \end{array}
$$

where $\bar { \lambda } _ { 1 }$ lies between $\lambda _ { 1 } ^ { * }$ and $\hat { \lambda } _ { 1 } ^ { ( \ell ) }$ , satisfying $\bar { \lambda } _ { 1 } \geq \eta$ and $\| \lambda _ { 1 } ^ { * } - \bar { \lambda } _ { 1 } \| _ { \infty } \leq \| \hat { \lambda } _ { 1 } - \lambda _ { 1 } ^ { * } \| _ { \infty }$ ,

$$
\begin{array} { r l } & { d ^ { 2 } M ( \bar { \zeta } ; \Delta _ { \zeta } ) = 2 \mathbb { E } \left[ \frac { K \left( \frac { P - \pi ( X ) } { h } \right) } { h \tilde { \lambda } _ { 2 } ^ { 2 } \left( X , P \right) } \Delta _ { \lambda _ { 2 } } ( X , P ) \Delta _ { \lambda _ { 1 } } ( X , P ) \right] + 2 \mathbb { E } \left[ \frac { K \left( \frac { P - \pi ( X ) } { h } \right) \Delta _ { \lambda _ { 2 } } ^ { 2 } ( X , P ) } { h \tilde { \lambda } _ { 2 } ^ { 3 } \left( X , P \right) } \Big ( U \left( P Y ; \alpha ^ { * } \right) - \bar { \lambda } _ { 1 } ( X , P ; \alpha ^ { * } ) \Big ) \right] } \\ & { \qquad \leq \mathcal { O } \left( \| \Delta _ { \lambda _ { 1 } } \| _ { \infty } \cdot \| \Delta \lambda _ { 2 } \| _ { \infty } \right) + \mathcal { O } _ { p } ( \| \Delta _ { \lambda _ { 2 } } \| _ { \infty } ^ { 2 } \cdot \| \Delta _ { \lambda _ { 1 } } \| _ { \infty } ) . } \end{array}
$$

Under the bandwidth condition $\sqrt { T h ^ { 1 + 2 s } }  0 , s = 1 , 2$ , and the product rate condition in Assumption 3.1, we have

$$
\begin{array} { r } { \mathbb { E } [ \psi ( V ; \theta ^ { * } , \hat { \zeta } ) - \psi ( V ; \theta ^ { * } , \zeta ^ { * } ) ] = o _ { p } ( ( T h ) ^ { - 1 / 2 } ) . } \end{array}\tag{32}
$$

Combining above (29)-(32), we have $\mathbb { E } _ { T } [ \psi ( V ; \theta ^ { * } , \hat { \zeta } ) ] = O _ { p } ( ( T h ) ^ { - 1 / 2 } )$ , then substituting (28), we have

$$
\lVert \hat { { \boldsymbol { \theta } } } - { \boldsymbol { \theta } } ^ { * } \rVert = O _ { p } ( ( T h ) ^ { - 1 / 2 } ) .\tag{33}
$$

Hence, combining (26), (27), and (33), we have

$$
\sqrt { T h } \mathcal { R } _ { T } = o _ { p } ( 1 ) .\tag{34}
$$

Based on (29) and (34), multiplying (23) by $\sqrt { T h }$ yields

$$
\sqrt { T h } ( \hat { \theta } - \theta ^ { * } ) = - J ^ { * - 1 } \left( \sqrt { T h } \mathbb { E } _ { T } [ \psi ( V ; \theta ^ { * } , \zeta ^ { * } ) ] \right) + o _ { p } ( 1 ) ,
$$

which implies that $\hat { \theta }$ is asymptotically linear. Furthermore, by the central limit theorem, we have

$$
\begin{array} { r } { \sqrt { T h } ( \hat { \theta } - \theta ^ { * } ) \stackrel { d } {  } \mathcal { N } ( \mathbf { 0 } , \Sigma ^ { * } ) , \quad \Sigma ^ { * } = J ^ { * - 1 } \Omega ^ { * } J ^ { * - \top } , } \end{array}
$$

where the asymptotic covariance matrix $\Omega ^ { * }$ is defined as li $\operatorname { n } _ { h \to 0 } \operatorname { V a r } ( { \sqrt { h } } \psi ( V ; \theta ^ { * } , \zeta ^ { * } ) )$ . Since $\mathbb { E } [ \psi ( V ; \theta ^ { * } , \zeta ^ { * } ) ] = 0$ , the variance is given by

$$
\begin{array} { r l } & { \Omega ^ { * } = \underset { h  0 } { \operatorname* { l i m } } \mathbb { E } [ ( \sqrt { h } \psi ( V ; \theta ^ { * } , \zeta ^ { * } ) ) ^ { 2 } ] = \int K ^ { 2 } ( u ) d u \cdot \mathrm { d i a g } ( \omega _ { 0 } , \omega _ { 1 } , 0 , 0 ) , } \\ & { \omega _ { j } = \mathbb { E } [ \frac { \mathrm { V a r } ( ( P Y ) ^ { j } \exp ( - P Y / \alpha ^ { * } ) \vert X , P = \pi ( X ) ) } { \pi _ { 0 } ( \pi ( X ) \vert X ) } ] , j = 0 , 1 , } \end{array}
$$

where we use the change of variables formula $\begin{array} { r } { u = \frac { p - \pi ( x ) } { h } } \end{array}$ in integration, and only $K ^ { 2 } ( \cdot ) / h ^ { 2 }$ is non-zero when $h  0$

Now, we establish that our proposed estimator $\hat { \theta }$ achieves the semiparametric efficiency lower bound. First, we show that we achieve semiparametric efficiency for the individual functionals $W _ { j } ( \pi , \alpha ^ { * } ) , j = 0 , 1$ . Following the theoretical framework in Colangelo $\&$ Lee (2026) (see Supplement S2), the target parameters $\theta ^ { * }$ are defined via the functionals $W _ { j } ^ { * } = \mathbb { E } [ ( P Y ) ^ { j } \exp ( - P Y / \alpha ^ { * } ) | P = \pi ( X ) ] , j = 0 ,$ 1. Note that at the fixed optimal dual parameter $\alpha ^ { * }$ (determined by the first-order condition $( 4 ) )$ , by treating the term $( P Y ) ^ { j } \exp ( - P Y / \alpha ^ { * } )$ as a square-integrable transformation on the outcome variable, each $W _ { j } ^ { * }$ can be viewed as an average structural function. As established in Colangelo & Lee (2026) (see Remark S5), the kernel-smoothed structure $\frac { K ( \frac { p - \pi ( x ) } { h } ) } { h \lambda _ { 2 } }$ is the unique local Riesz representer for the evaluation of the conditional expectation at the point $p = \pi ( x )$ . This implies that the diagonal elements ω<sub>j</sub> of $\Omega ^ { * }$ defined in the expression above achieve the semiparametric efficiency lower bound for estimating $W _ { j } ^ { * }$

Second, we demonstrate that this efficiency propagates to the parameter vector $\theta ^ { * }$ through Hadamard differentiability. The moment function ψ is constructed as the doubly robust score. Specifically, the first two components of ψ (corresponding to $W _ { 0 }$ and $W _ { 1 } )$ use the local Riesz representer structure established in Colangelo & Lee (2026), which ensures that these com ponents achieve semiparametric efficiency. The resulting influence function of the system is given by $- J ^ { * - 1 } \psi ( V ; \theta ^ { * } , \zeta ^ { * } )$ From the first-order condition (4), we can explicitly express $\alpha ^ { * }$ as $\alpha ^ { * } = \frac { W _ { 1 } ^ { * } } { W _ { 0 } ^ { * } ( - \log W _ { 0 } ^ { * } - \delta ) }$ , and $R _ { \delta } ^ { * } = - \alpha ^ { * } \log W _ { 0 } ^ { * } - \alpha ^ { * } \delta$ from (5). The mapping from $( W _ { 0 } ^ { * } , W _ { 1 } ^ { * } )$ to $( \alpha ^ { * } , R _ { \delta } ^ { * } )$ is Hadamard differentiable. Specifically, since $W _ { 1 } ^ { * }$ is bounded and the functions involved (logarithm and division) are smooth in the region where $W _ { 0 } ^ { * } \ge \underline { { \rho } } / 2$ (by choosing $h$ such that $\begin{array} { r } { h ^ { 2 } \leq \frac { \underline { { \rho } } } { 2 C _ { 1 } L _ { m } } } \end{array}$ in (43)), the mapping is Hadamard differentiable. By Theorems 25.20, 25.21, and 25.47 of van der Vaar (1998), the efficiency of $W _ { 0 } ^ { * }$ and $W _ { 1 } ^ { * }$ propagates to the efficiency of each component of $\theta ^ { * }$ through the delta method. Since the influence function $- J ^ { * - 1 } \psi$ has the structure of the efficient influence function for the parameter vector $\theta ^ { * }$ the asymptotic covariance matrix $\Sigma ^ { * } = J ^ { * - 1 } \Omega ^ { * } J ^ { * - \top }$ achieves the semiparametric efficiency lower bound for the entire parameter vector $\theta ^ { * }$ , and in particular for each component $\alpha ^ { * } , W _ { 0 } ^ { * } , W _ { 1 } ^ { * }$ , and $R _ { \delta } ^ { * }$ □

## C. Proof of Results in Section 4

To ease the presentation, we use notations $W ,$ , m and $\hat { m } ^ { ( \ell ) }$ to represent the true and estimated functions, $W _ { 0 } , m _ { 0 }$ and $\hat { m } _ { 0 } ^ { ( \ell ) }$ respectively.

Lemma C.1. Let $\hat { W } _ { T } ( \alpha )$ be defined as in (8). The first and second derivatives ofthe objectivefunction $\hat { \varphi } _ { T } ( \alpha )$ required for the Newton update are:

$$
\frac { \partial } { \partial \alpha } \hat { \varphi } _ { T } ( \alpha ) = - \log \hat { W } _ { T } ( \alpha ) - \alpha \frac { \hat { W } _ { T } ^ { \prime } ( \alpha ) } { \hat { W } _ { T } ( \alpha ) } - \delta ,
$$

$$
\frac { \partial ^ { 2 } } { \partial \alpha ^ { 2 } } \hat { \varphi } _ { T } ( \alpha ) = \frac { \alpha ( \hat { W } _ { T } ^ { \prime } ( \alpha ) ) ^ { 2 } - 2 \hat { W } _ { T } ^ { \prime } ( \alpha ) \hat { W } _ { T } ( \alpha ) - \alpha \hat { W } _ { T } ^ { \prime \prime } ( \alpha ) \hat { W } _ { T } ( \alpha ) } { ( \hat { W } _ { T } ( \alpha ) ) ^ { 2 } } ,
$$

where the derivatives ofthe weighted generatingfunction are given by:

$$
\hat { W } _ { T } ^ { \prime } ( \alpha ) = \frac { 1 } { T } \sum _ { \ell = 1 } ^ { L } \sum _ { t \in \mathcal { Z } _ { \ell } } \frac { K \left( \frac { p _ { t } - \pi ( x _ { t } ) } { h } \right) } { \hat { S } _ { \ell } ^ { \pi } h \hat { \pi } _ { 0 } ( p _ { t } | x _ { t } ) } \exp \left( - \frac { p _ { t } y _ { t } } { \alpha } \right) \frac { p _ { t } y _ { t } } { \alpha ^ { 2 } } ,
$$

$$
\hat { W } _ { T } ^ { \prime \prime } ( \alpha ) = \frac { 1 } { T } \sum _ { \ell = 1 } ^ { L } \sum _ { t \in \mathcal { Z } _ { \ell } } \frac { K \left( \frac { p _ { t } - \pi ( x _ { t } ) } { h } \right) } { \hat { S } _ { \ell } ^ { \pi } h \hat { \pi } _ { 0 } ( p _ { t } | x _ { t } ) } \exp \left( - \frac { p _ { t } y _ { t } } { \alpha } \right) \left( \frac { ( p _ { t } y _ { t } ) ^ { 2 } } { \alpha ^ { 4 } } - \frac { 2 p _ { t } y _ { t } } { \alpha ^ { 3 } } \right) .
$$

Proof of Lemma C.1. This proof is straightforward by taking the derivatives of the function $\hat { \varphi } _ { T } ( \alpha )$ with respect to α.

ProofofTheorem 4.4. We can rewrite the DR estimator as follows,

$$
\hat { W } ^ { ( t ) } ( \pi , \alpha ) = \frac { 1 } { | \mathcal { Z } _ { t } | \hat { S } _ { \ell } ^ { \pi } } \sum _ { t \in \mathcal { Z } _ { \ell } } \frac { K \left( \frac { p _ { \ell } - \pi ( x _ { \ell } ) } { h } \right) } { h \hat { \pi } _ { 0 } ( p _ { t } | x _ { t } ) } \exp ( - p _ { t } y _ { t } / \alpha ) + \frac { 1 } { | \mathcal { Z } _ { \ell } | } \sum _ { t \in \mathcal { Z } _ { \ell } } \left( \hat { m } ^ { ( \ell ) } ( x _ { t } , \pi ( x _ { t } ) ; \alpha ) - \frac { 1 } { \hat { S } _ { \ell } ^ { \pi } } \hat { m } ^ { ( \ell ) } ( x _ { t } , p _ { t } ; \alpha ) \right) .
$$

We also define an infeasible optimal dual objective function $\begin{array} { r } { \tilde { R } _ { \delta } ^ { h } ( \pi ) \triangleq \operatorname* { s u p } _ { \alpha > 0 } \{ - \alpha \log \tilde { W } _ { h } ( \pi , \alpha ) - \alpha \delta \} } \end{array}$ , where

$$
\tilde { W } _ { h } ( \pi , \alpha ) \triangleq \mathbb { E } _ { \mathbb { P } _ { 0 } \ast \pi _ { 0 } } \left[ m ( X , \pi ( X ) ; \alpha ) + \frac { K \left( \frac { P - \pi ( X ) } { h } \right) } { h \pi _ { 0 } ( P | X ) } \left( \exp ( - r ( X , P , Z ) / \alpha ) - m ( X , P ; \alpha ) \right) \right] ,
$$

the empirical dual objective function $\begin{array} { r } { \tilde { R } _ { \delta } ( \pi ) \triangleq \operatorname* { s u p } _ { \alpha \geq 0 } \{ - \alpha \log \tilde { W } _ { T } ( \pi , \alpha ) - \alpha \delta \} } \end{array}$ , and the infeasible estimator is

$$
\tilde { W } ^ { ( 6 ) } ( \pi , \alpha ) \triangleq \frac { 1 } { | \mathbb { Z } _ { \ell } | S _ { \ell } ^ { \pi } } \sum _ { \ell \in \mathbb { Z } _ { \ell } } \frac { K \left( \frac { \rho _ { \ell } - \alpha ( x _ { \ell } ) } { h } \right) } { h \pi _ { 0 } ( p _ { \ell } | x _ { \ell } ) } \exp ( - p _ { \ell } y _ { \ell } / \alpha ) + \frac { 1 } { | \mathbb { Z } _ { \ell } | } \sum _ { \ell \in \mathbb { Z } _ { \ell } } \left( m ( x _ { t } , \pi ( x _ { t } ) ; \alpha ) - \frac { 1 } { S _ { \ell } ^ { \pi } } \frac { K \left( \frac { p _ { \ell } - \alpha ( x _ { \ell } ) } { h } \right) } { h \pi _ { 0 } ( p _ { \ell } | x _ { \ell } ) } m ( x _ { t } , p _ { \ell } ; \alpha ) \right) ,
$$

$$
S _ { \ell } ^ { \pi } \triangleq \frac { 1 } { | { \mathcal T } _ { \ell } | } \sum _ { t \in { \mathcal T } _ { \ell } } \frac { K \left( \frac { p _ { t } - \pi ( x _ { t } ) } { h } \right) } { h \pi _ { 0 } ( p _ { t } | x _ { t } ) } , \quad \mathrm { ~ a n d ~ } \quad \tilde { W } _ { T } ( \pi , \alpha ) \triangleq \frac { 1 } { L } \sum _ { \ell = 1 } ^ { L } \tilde { W } ^ { ( \ell ) } ( \pi , \alpha ) .
$$

Then we can decompose regret as

$$
\begin{array} { r l } {  { \mathrm { R e g } ( \hat { \pi } ) = R _ { \delta } ( \pi ^ { * } ) - \hat { R } _ { \delta } ( \hat { \pi } ) + \hat { R } _ { \delta } ( \hat { \pi } ) - R _ { \delta } ( \hat { \pi } ) } } \\ & { \leq \Big ( R _ { \delta } ( \pi ^ { * } ) - \hat { R } _ { \delta } ( \pi ^ { * } ) \Big ) + \Big ( \hat { R } _ { \delta } ( \hat { \pi } ) - R _ { \delta } ( \hat { \pi } ) \Big ) \leq 2 \operatorname* { s u p } _ { \pi \in \Pi } | R _ { \delta } ( \pi ) - \hat { R } _ { \delta } ( \pi ) | } \\ & { \leq 2 \operatorname* { s u p } _ { \pi \in \Pi } \vert R _ { \delta } ( \pi ) - \tilde { R } _ { \delta } ^ { h } ( \pi ) \vert + 2 \operatorname* { s u p } _ { \pi \in \Pi } \vert \tilde { R } _ { \delta } ^ { h } ( \pi ) - \tilde { R } _ { \delta } ( \pi ) \vert + 2 \operatorname* { s u p } _ { \pi \in \Pi } \vert \tilde { R } _ { \delta } ( \pi ) - \hat { R } _ { \delta } ( \pi ) \vert . } \end{array}
$$

By utilizing Lemmas C.2, C.4 and C.5, we obtain with probability at least $1 - 7 \epsilon$

$$
\mathrm { R e g } ( \hat { \pi } ) \lesssim \overline { { \alpha } } \left( \log \left( \frac { 1 } { \bar { h } } \right) \cdot \left( \frac { d _ { \Pi } } { T h / L } + \sqrt { \frac { d _ { \Pi } } { T h / L } } + \sqrt { \frac { \log ( 1 / \epsilon ) } { T h / L } } + \frac { \log ( 1 / \epsilon ) } { T h / L } \right) + T ^ { - \gamma _ { \pi } - \gamma _ { m } } + h ^ { s } \right) + h ^ { s } ,
$$

for $s = 1 , 2$ in Assumption 2.4, which completes the proof.

Lemma C.2. Under Assumptions 2.1, 2.3, 2.4, 3.2, we have $\begin{array} { r } { \operatorname* { s u p } _ { \pi \in \Pi } | R _ { \delta } ( \pi ) - \tilde { R } _ { \delta } ^ { h } ( \pi ) | \lesssim h ^ { s } ~ f o r ~ s = 1 } \end{array}$ , 2 in Assumption 2.4.

ProofofLemma C.2. To ease the notation, let $W ( \pi , \alpha ) : = \mathbb { E } _ { \mathbf { P } _ { 0 } } \left[ \exp \left( - r ( X , \pi ( X ) , Z ) / \alpha \right) \right]$ . Define an alternative approach using $m ( X , \pi ( X ) ; \alpha )$ instead of $m ( X , P ; \alpha )$

$$
\tilde { W } _ { h } ^ { \prime } ( \pi , \alpha ) \triangleq \mathbb { E } _ { \mathbb { P } _ { 0 } \ast \pi _ { 0 } } \left[ m ( X , \pi ( X ) ; \alpha ) + \frac { K \left( \frac { P - \pi ( X ) } { h } \right) } { h \pi _ { 0 } ( P | X ) } \left( \exp ( - r ( X , P , Z ) / \alpha ) - m ( X , \pi ( X ) ; \alpha ) \right) \right] .
$$

First, we show that su $ \mathrm { p } _ { \alpha > 0 } \{ \alpha | \tilde { W } _ { h } ^ { \prime } ( \pi , \alpha ) - W ( \pi , \alpha ) | \} \le \mathcal { O } ( h )$ . Recalling that the conditional expectation of the transformed reward is $m ( X , P ; \alpha ) : = \mathbb { E } _ { \mathbf { P } _ { 0 } } \left[ \exp ( - r ( X , P , \bar { Z } ) / \alpha ) \mid X , P \right]$ . Then we can rewrite as

$$
\begin{array} { r l } { W ( \pi , \alpha ) = \mathbb { E } _ { \mathbf { P } _ { 0 } } \left[ \exp { ( - r ( X , \pi ( X ) , Z ) / \alpha ) } \right] } \\ & { = \mathbb { E } _ { X } \left[ \mathbb { E } _ { Z } [ \exp { ( - r ( X , \pi ( X ) , Z ) / \alpha ) } | X ] \right] } \\ & { = \mathbb { E } _ { X } \left[ \mathbb { E } _ { Z } [ \exp { ( - r ( X , P , Z ) / \alpha ) } | X , P = \pi ( X ) ] \right] } \\ & { = \mathbb { E } _ { X } [ m ( X , \pi ( X ) ; \alpha ) ] } \\ & { = \mathbb { E } _ { X } \left[ \mathbb { E } _ { Z } [ \exp { ( - r ( P , Z ) / \alpha ) } | X , P = p ] \cdot \mathbf { 1 } \{ \pi ( X ) = p \} \right] } \\ & { = \mathbb { E } _ { X } \left[ \int _ { p _ { \operatorname* { m i n } } } ^ { 1 } \mathbb { E } _ { Z } [ \exp { ( - r ( P , Z ) / \alpha ) } | X , P = p ] \cdot \mathbf { 1 } \{ \pi ( X ) = p \} \mathrm { d } p \right] , } \end{array}
$$

$$
\begin{array} { r l } & { \tilde { W } _ { h } ^ { \prime } ( \pi , \alpha ) = \mathbb { E } _ { \mathbb { P } _ { 0 } } \left[ m ( X , \pi ( X ) ; \alpha ) + \displaystyle \frac { 1 } { h } \int _ { p _ { \operatorname* { m i n } } } ^ { 1 } K \left( \frac { p - \pi ( X ) } { h } \right) \left( e ^ { - r ( X , p , Z ) / \alpha } - m ( X , \pi ( X ) ; \alpha ) \right) \mathrm { d } p \right] } \\ & { \qquad = \displaystyle \frac { 1 } { h } \int _ { p _ { \operatorname* { m i n } } } ^ { 1 } \mathbb { E } _ { \mathbb { P } _ { 0 } } \left[ K \left( \frac { p - \pi ( X ) } { h } \right) e ^ { - r ( X , p , Z ) / \alpha } \right] \mathrm { d } p \quad \mathrm { ~ A s s u m p t i o n ~ } 3 . 2 \mathrm { ~ a n d ~ F u b i n i - T o n e l l i ~ } } \\ & { \qquad = \displaystyle \frac { 1 } { h } \int _ { p _ { \operatorname* { m i n } } } ^ { 1 } \mathbb { E } _ { X } \left[ K \left( \frac { p - \pi ( X ) } { h } \right) \mathbb { E } _ { \mathbb { P } _ { 0 } | X } \left[ e ^ { - r ( X , p , Z ) / \alpha } | X \right] \right] \mathrm { d } p } \\ & { \qquad = \mathbb { E } _ { X } \left[ \displaystyle \int _ { p _ { \operatorname* { m i n } } } ^ { 1 } \frac { K \left( \frac { p - \pi ( X ) } { h } \right) m ( X , p ; \alpha ) } { h } \mathrm { d } p \right] = \mathbb { E } _ { X } \left[ \displaystyle \int _ { - \infty } ^ { \infty } m ( X , t h + \pi ( X ) ; \alpha ) K ( t ) d t \right] . } \end{array}\tag{35}
$$

Case 1: Second-order bounded derivative holds in Assumption 2.4. Suppose that the second-order derivative of m $( X , P ; \alpha )$ is bounded by $L _ { m } .$ , and we delay the justification of $L _ { m }$ later. By the second-order Taylor expansion of $m ( X , P ; \alpha )$ around $P = \pi ( X )$ , we have

$$
| \tilde { W } _ { h } ^ { \prime } ( \pi , \alpha ) - W ( \pi , \alpha ) | = \left| \mathbb { E } _ { \mathbf { P } _ { 0 } } \left[ \int m ( X , t h + \pi ( X ) ; \alpha ) - m ( X , \pi ( X ) ; \alpha ) K ( t ) d t \right] \right| \le C _ { 1 } L _ { m } h ^ { 2 } \left| \int \_ { \alpha } \int \_ { \alpha } \int _ { \alpha } \mathbf { 1 } _ { \beta } \left[ \int \_ \pi \tilde { L } ( X , t h + \pi ( X ) ; \alpha ) \right] h ( t ) d t \right| .\tag{36}
$$

where we use $\begin{array} { r } { \int u K ( u ) d u = 0 } \end{array}$ and $\begin{array} { r } { \int u ^ { 2 } K ( u ) d u \leq C _ { 1 } < \infty } \end{array}$ from Assumption 3.2. Now, we provide the specific $L _ { m }$ . We derive the partial derivative as

$$
- 1 \leq \frac { \partial m ( X , P ; \alpha ) } { \partial P } = - \frac { 1 } { \alpha } e ^ { - P / \alpha } \cdot \mathbb { P } ( Z \geq P - f ( X ) | X ) + \left( 1 - e ^ { - P / \alpha } \right) f _ { Z } ( P - f ( X ) ) \leq L _ { Z } ,\tag{37}
$$

$$
0 \leq \frac { \partial ^ { 2 } m } { \partial P ^ { 2 } } = f _ { Z } ^ { \prime } ( P - f ( X ) ) ( 1 - e ^ { - P / \alpha } ) + \frac { 2 } \alpha e ^ { - P / \alpha } f _ { Z } ( P - f ( X ) ) + \frac { 1 } { \alpha ^ { 2 } } e ^ { - P / \alpha } \mathbb { P } ( Z \geq P - f ( X ) | X )
$$

$$
\leq L _ { Z } + \frac { 2 } { e p _ { \mathrm { { m i n } } } } + \frac { 4 } { e ^ { 2 } p _ { \mathrm { { m i n } } } ^ { 2 } }\tag{38}
$$

then $\begin{array} { r } { L _ { m } \equiv \operatorname* { m a x } \{ 1 , L _ { Z } + \frac { 2 } { e p _ { \mathrm { m i n } } } + \frac { 4 } { e ^ { 2 } p _ { \mathrm { m i n } } ^ { 2 } } \} } \end{array}$ , where $f _ { Z } , f _ { Z } ^ { \prime }$ are pdf and pdf derivative of $Z ,$ , and $L _ { Z }$ -bounded from Assumption 2.4. Multiplying by α, we get

$$
\operatorname* { s u p } _ { \alpha > 0 } \left\{ \alpha | \tilde { W } _ { h } ^ { \prime } ( \pi , \alpha ) - W ( \pi , \alpha ) | \right\} \leq \operatorname* { s u p } _ { \alpha > 0 } \alpha \left| \mathbb { E } \left[ \int m ( X , \iota h + \pi ( X ) ; \alpha ) - m ( X , \pi ( X ) ; \alpha ) K ( \iota ) d \iota \right] \right| \leq C _ { 1 } \hbar ^ { 2 } \operatorname* { s u p } _ { \alpha \geqslant 0 } \left\{ \left| \alpha \frac { \partial ^ { 2 } m } { \partial P ^ { 2 } } \right| \right\} .
$$

The following step is to show that $\textstyle \operatorname* { s u p } _ { \alpha > 0 } \{ \alpha \frac { \partial ^ { 2 } m } { \partial P ^ { 2 } } \}$ is a finite constant. Note that $p \in [ p _ { \operatorname* { m i n } } , 1 ]$ , then

$$
\begin{array} { r l r } {  { 0 \leq \alpha \frac { \partial ^ { 2 } m } { \partial P ^ { 2 } } = \alpha f _ { Z } ^ { \prime } ( P - f ( X ) ) ( 1 - e ^ { - P / \alpha } ) + 2 e ^ { - P / \alpha } f _ { Z } ( P - f ( X ) ) + \frac { 1 } { \alpha } e ^ { - P / \alpha } \mathbb { P } ( Z \geq P - f ( X ) | X ) } } \\ & { } & { \leq \alpha ( 1 - e ^ { - 1 / \alpha } ) L _ { Z } + 2 L _ { Z } + \frac { 1 } { \alpha } L _ { Z } + \frac { 1 } { e p _ { \operatorname* { m i n } } } \leq 3 L _ { Z } + \frac { 1 } { e p _ { \operatorname* { m i n } } } , \quad \forall \alpha > 0 . \qquad } \end{array}\tag{39}
$$

Hence, we conclude that $\begin{array} { r } { \operatorname* { s u p } _ { \alpha > 0 } \lbrace \alpha \frac { \partial ^ { 2 } m } { \partial P ^ { 2 } } \rbrace = 3 L _ { Z } + \frac { 1 } { e p _ { \mathrm { m i n } } } } \end{array}$ . With some abuse of notation, let $\begin{array} { r } { L _ { m } : = 1 + 3 L _ { Z } + \frac { 2 } { e p _ { \operatorname* { m i n } } } + \frac { 4 } { e ^ { 2 } p _ { \operatorname* { m i n } } ^ { 2 } } } \end{array}$ Then, we get

$$
\operatorname* { s u p } _ { \alpha > 0 } \Big \{ \alpha | \tilde { W } _ { h } ^ { \prime } ( \pi , \alpha ) - W ( \pi , \alpha ) | \Big \} \le C _ { 1 } L _ { m } h ^ { 2 } .\tag{40}
$$

By a similar process, we obtain

$$
\begin{array} { r l } & { \displaystyle \left| { { { \tilde { W } } _ { h } } ( \pi , \alpha ) - { { \tilde { W } } _ { h } ^ { \prime } } ( \pi , \alpha ) } \right| = \left| { { \mathbb E } _ { { { \bf { P } } _ { 0 } } \circ \pi _ { 0 } } \left[ \frac { K \left( \frac { P - \pi ( X ) } { h } \right) } { h \pi _ { 0 } ( P | X ) } \left( m ( X , \pi ( X ) ; \alpha ) - m ( X , P ; \alpha ) \right) \right] } \right| } \\ & { \displaystyle \leq \left| { \frac { 1 } { h } { \mathbb E } _ { X } \left[ \int _ { p _ { m i n } } ^ { 1 } K \left( \frac { p - \pi ( X ) } { h } \right) m ( X , \pi ( X ) ; \alpha ) - m ( X , p ; \alpha ) d p \middle | X \right] } \right| } \\ & { \displaystyle = \left| { { \mathbb E } _ { X } \left[ \int K ( u ) m ( X , \pi ( X ) ; \alpha ) - m ( X , \pi ( X ) + u h ; \alpha ) d u \middle | X \right] } \right| \leq C _ { 1 } { { L } _ { m } } h ^ { 2 } , \quad \mathrm { a n d } } \\ & { \displaystyle \underset { \alpha \geq 0 } { \operatorname* { s u p } } \left. \alpha | { { \tilde { W } } _ { h } ^ { \prime } ( \pi , \alpha ) - { { \tilde { W } } _ { h } } ( \pi , \alpha ) } | \right. \leq C _ { 1 } { { L } _ { m } } h ^ { 2 } . } \end{array}\tag{41}
$$

(42)

To proceed, we can pick $\begin{array} { r } { h ^ { 2 } \leq \frac { \underline { { \rho } } } { 2 C _ { 1 } L _ { m } } } \end{array}$ , the (36) and (41) imply that

$$
\left| \tilde { W } _ { h } ^ { \prime } ( \pi , \alpha ) - W ( \pi , \alpha ) \right| \leq \frac { \rho } { 2 } , \quad \left| \tilde { W } _ { h } ( \pi , \alpha ) - \tilde { W } _ { h } ^ { \prime } ( \pi , \alpha ) \right| \leq \frac { \rho } { 2 } .
$$

Therefore, we have

$$
\begin{array} { r l } & { | R _ { \delta } ( \pi ) - \tilde { R } _ { \delta } ^ { h } ( \pi ) | \leq \displaystyle \operatorname* { s u p } _ { \alpha > 0 } \left| ( - \alpha \log \tilde { W } _ { h } ( \pi , \alpha ) - \alpha \delta ) - ( - \alpha \log W ( \pi , \alpha ) - \alpha \delta ) \right| } \\ & { \qquad = \displaystyle \operatorname* { s u p } _ { \alpha > 0 } \left| \alpha \log \left( \frac { W ( \pi , \alpha ) } { \tilde { W } _ { h } ( \pi , \alpha ) } \right) \right| \leq \displaystyle \operatorname* { s u p } _ { \alpha > 0 } \left| \alpha \log \left( \frac { W ( \pi , \alpha ) } { \tilde { W } _ { h } ^ { \prime } ( \pi , \alpha ) } \right) \right| + \displaystyle \operatorname* { s u p } _ { \alpha > 0 } \left| \alpha \log \left( \frac { \tilde { W } _ { h } ( \pi , \alpha ) } { \tilde { W } _ { h } ^ { \prime } ( \pi , \alpha ) } \right) \right| . } \end{array}
$$

For the first term,

$$
\begin{array} { r l } & { \underset { \alpha > 0 } { \operatorname* { s u p } } \left| \alpha \log \left( \frac { W ( \pi , \alpha ) } { \bar { W } _ { h } ^ { \prime } ( \pi , \alpha ) } \right) \right| = \underset { \alpha > 0 } { \operatorname* { s u p } } \left| \alpha \log \left( 1 + \frac { W ( \pi , \alpha ) - \bar { W } _ { h } ^ { \prime } ( \pi , \alpha ) } { \bar { W } _ { h } ^ { \prime } ( \pi , \alpha ) } \right) \right| } \\ & { \overset { ( a ) } { \leq } 2 \underset { \alpha > 0 } { \operatorname* { s u p } } \left\{ \alpha \left| \frac { W ( \pi , \alpha ) - \bar { W } _ { h } ^ { \prime } ( \pi , \alpha ) } { \bar { W } _ { h } ^ { \prime } ( \pi , \alpha ) } \right| \right\} \overset { ( b ) } { \leq } \frac { 2 } { \underline { { \rho } } } \underset { \alpha > 0 } { \operatorname* { s u p } } \left\{ \alpha \left| W ( \pi , \alpha ) - \tilde { W } _ { h } ^ { \prime } ( \pi , \alpha ) \right| \right\} \overset { ( c ) } { \leq } \frac { 2 } { \underline { { \rho } } } \mathcal { C } _ { 1 } L _ { m } h ^ { 2 } , } \end{array}
$$

where step (a)-(b) follow from the inequality $| \log ( 1 + u ) | \leq 2 | u |$ for any $| u | \leq 1 / 2$ and $\tilde { W } _ { h } ^ { \prime } ( \pi , \alpha ) \geq \underline { { \rho } } ,$ which is continued from (35),

$$
\begin{array} { l l } { \displaystyle \tilde { W } _ { h } ^ { \prime } ( \pi , \alpha ) = \frac { 1 } { h } \int _ { p _ { \operatorname* { m i n } } } ^ { 1 } \mathbb { E } _ { X } \left[ K \left( \frac { p - \pi ( X ) } { h } \right) \mathbb { E } _ { \mathbf { P } _ { 0 } \mid X } \left[ e ^ { - r ( p , Z ) / \alpha } \big \vert X \right] \right] \mathrm { d } p } \\ { \displaystyle \qquad = \frac { 1 } { h } \int _ { p _ { \operatorname* { m i n } } } ^ { 1 } \mathbb { E } _ { X } \left[ K \left( \frac { p - \pi ( X ) } { h } \right) \Big ( e ^ { - p / \alpha } \cdot \mathbf { P } _ { 0 } ( v ( X , Z ) \ge p | X ) + \mathbf { P } _ { 0 } ( v ( X , Z ) < p | X ) \Big ) \right] \mathrm { d } p } \\ { \displaystyle \qquad \ge \frac { \rho } { h } \int _ { p _ { \operatorname* { m i n } } } ^ { 1 } \mathbb { E } _ { X } \left[ K \left( \frac { p - \pi ( X ) } { h } \right) \right] \mathrm { d } p } & { \mathrm { ~ b y ~ A s s u m p t i o n ~ } 2 . 3 } \end{array}
$$

$$
= \frac { \rho } { h } \mathbb { E } _ { X } \left[ \int _ { p _ { \mathrm { m i n } } } ^ { 1 } K \left( \frac { p - \pi ( X ) } { h } \right) \mathrm { d } p \right] \geq \underline { { \rho } } .\tag{43}
$$

And step (c) directly follows from (40). So does the second item. Combining above two terms (40) and (42), we get $| R _ { \delta } ( \pi ) \stackrel { \cdot } { - } \tilde { R } _ { \delta } ^ { h } ( \pi ) | \leq \stackrel { \cdot } { \frac { } { \rho } } C _ { 1 } L _ { m } h ^ { 2 }$

Case 2: First-order bounded derivative holds in Assumption 2.4. By a similar process, we have $| R _ { \delta } ( \pi ) ~ - ~ \tilde { R } _ { \delta } ^ { h } ( \pi ) | ~ \le$ $\begin{array} { r } { \frac { 4 } { \underline { { \rho } } } C _ { 1 } L _ { m } h , } \end{array}$ where we let $L _ { m } \equiv \operatorname* { m a x } \{ 1 , L _ { Z } \}$ from (37). □

## Lemma C.3. Under Assumption 2.3, we have

$$
| \tilde { R } _ { \mathrm { D R O } } ^ { h } ( \pi ) - \tilde { R } _ { \mathrm { D R O } } ( \pi ) | \leq \frac { 4 } { \underline { { \rho } } } \frac { 1 } { L } \sum _ { \ell = 1 } ^ { L } \Bigl ( 2 d _ { K S } ( F _ { \mathbb { Q } ^ { \pi } } , F _ { \hat { \mathbb { Q } } _ { \ell } ^ { \pi } } ) + d _ { K S } ( F _ { \mathbf { P } _ { 0 } } , F _ { \hat { \mathbb { P } } _ { \ell } } ) \Bigr ) ,
$$

where $d _ { \mathrm { K S } } ( \cdot , \cdot )$ denotes the KS distance, defined as $\begin{array} { r } { d _ { \mathrm { K S } } ( F _ { 1 } , F _ { 2 } ) \triangleq \operatorname* { s u p } _ { s \in [ 0 , 1 ] } | F _ { 1 } ( s ) - F _ { 2 } ( s ) | . } \end{array}$ , $\hat { \mathbb { P } } _ { \ell }$ is a uniform empirical measure on $\hat { \Omega } _ { \ell } : = \{ ( z _ { t } , x _ { t } , p _ { t } ) \} _ { t \in \mathcal { T } _ { \ell } } , \ell = 1 , . . . , L ,$ and the kernel-weighted measures are defined as

$$
\frac { \mathrm { d } \mathbb { Q } ^ { \pi } } { \mathrm { d } ( \mathbf { P } _ { 0 } * \pi _ { 0 } ) } \triangleq \frac { K \left( \frac { P - \pi ( X ) } { h } \right) } { h \pi _ { 0 } ( P | X ) } , \quad a n d \quad \frac { \mathrm { d } \hat { \mathbb { Q } } _ { \ell } ^ { \pi } } { \mathrm { d } \hat { \mathbb { P } } _ { \ell } } ( s ) \triangleq \frac { 1 } { S _ { \ell } ^ { \pi } } \cdot \frac { K \left( \frac { p _ { t } - \pi ( x _ { t } ) } { h } \right) } { h \pi _ { 0 } ( p _ { t } | x _ { t } ) } .
$$

Proof of Lemma C.3. For simplicity, we denote the first term in $\tilde { W } _ { T } ( \pi , \alpha )$ as

$$
\tilde { V } _ { 1 } ( \pi , \alpha ) \triangleq \frac { 1 } { L } \sum _ { \ell = 1 } ^ { L } \frac { 1 } { | \mathscr { T } _ { \ell } | S _ { \ell } ^ { \pi } } \sum _ { t \in \mathcal { Z } _ { \ell } } \frac { K \left( \frac { p _ { t } - \pi ( x _ { t } ) } { h } \right) } { h \pi _ { 0 } ( p _ { t } | x _ { t } ) } \exp ( - p _ { t } y _ { t } / \alpha )
$$

and the second term as

$$
\tilde { V } _ { 2 } ( \pi , \alpha ) \triangleq \frac { 1 } { L } \sum _ { \ell = 1 } ^ { L } \frac { 1 } { | \mathscr { T } _ { \ell } | } \sum _ { t \in \mathcal { T } _ { \ell } } \left( m ( x _ { t } , \pi ( x _ { t } ) ; \alpha ) - \frac { 1 } { S _ { \ell } ^ { \pi } } \frac { K ( \frac { p _ { t } - \pi ( x _ { t } ) } { h } ) } { h \pi _ { 0 } ( p _ { t } | x _ { t } ) } m ( x _ { t } , p _ { t } ; \alpha ) \right) ,
$$

so that $\tilde { W } _ { T } ( \pi , \alpha ) \equiv \tilde { V } _ { 1 } ( \pi , \alpha ) + \tilde { V } _ { 2 } ( \pi , \alpha )$ . Similarly, we write

$$
\tilde { V } _ { 1 } ^ { h } ( \pi , \alpha ) \triangleq \mathbb { E } _ { \mathbf { P } _ { 0 } \sim \pi _ { 0 } } \left[ \frac { K ( \frac { P - \pi ( X ) } { h } ) } { h \pi _ { 0 } ( P | X ) } \exp ( - P \cdot Y / \alpha ) \right] , \quad \tilde { V } _ { 2 } ^ { h } ( \pi , \alpha ) \triangleq \mathbb { E } _ { \mathbf { P } _ { 0 } \sim \pi _ { 0 } } \left[ m ( X , \pi ( X ) ; \alpha ) - \frac { K ( \frac { P - \pi ( X ) } { h } ) } { h \pi _ { 0 } ( P | X ) } m ( X , P ; \alpha ) \right] ,
$$

so that $\tilde { W } _ { h } ( \pi , \alpha ) \equiv \tilde { V } _ { 1 } ^ { h } ( \pi , \alpha ) + \tilde { V } _ { 2 } ^ { h } ( \pi , \alpha )$

We decompose $\tilde { W } _ { T } ( \pi , \alpha )$ and $\tilde { W } _ { h } ( \pi , \alpha )$ into three distinct components based on the following measures. Denote the CDF of $r ( X , P , Z )$ under the measure $\mathbb { Q } ^ { \pi }$ as $F _ { \mathbb { Q } ^ { \pi } } ( s )$

$$
F _ { \mathbb { Q } ^ { \pi } } ( s ) \triangleq \mathbb { Q } ^ { \pi } ( r ( X , P , Z ) \leq s ) = \mathbb { E } _ { \mathbb { Q } ^ { \pi } } [ \mathbf { 1 } \{ r ( X , P , Z ) \leq s \} ] = \mathbb { E } _ { \mathbb { P } _ { 0 } * \pi _ { 0 } } \left[ \mathbf { 1 } \{ r ( X , P , Z ) \leq s \} \cdot \frac { K \left( \frac { P - \pi ( X ) } { h } \right) } { h \pi _ { 0 } ( P | X ) } \right] .
$$

Hence, we can calculate the expectation of the function $g ( X , Z , P ) = \exp ( - r ( X , P , Z ) / \alpha )$ with respect to $\mathbb { Q } ^ { \pi }$ as follows:

$$
\begin{array} { r } { \tilde { V } _ { 1 } ^ { h } ( \pi , \alpha ) = \mathbb { E } _ { \mathbb { Q } ^ { \pi } } [ \exp ( - r ( X , P , Z ) / \alpha ) ] . } \end{array}
$$

Similarly, denote the CDF of $r ( X , P , Z )$ under the measure $\hat { \mathbb { Q } } _ { \ell } ^ { \pi }$ as $F _ { \hat { \mathbb { Q } } _ { \rho } ^ { \pi } } ( s )$

$$
F _ { \widehat { \mathbb { Q } } _ { \ell } ^ { \pi } } ( s ) \triangleq \widehat { \mathbb { Q } } _ { \ell } ^ { \pi } ( r \leq s ) = \mathbb { E } _ { \widehat { \mathbb { Q } } _ { \ell } ^ { \pi } } [ \mathbf { 1 } \{ r \leq s \} ] = \mathbb { E } _ { \widehat { \mathbb { P } } _ { \ell } } \left[ \mathbf { 1 } \{ r \leq s \} \cdot \frac { \mathrm { d } \widehat { \mathbb { Q } } _ { \ell } ^ { \pi } } { \mathrm { d } \widehat { \mathbb { P } } _ { \ell } } ( t ) \right] = \frac { 1 } { | \mathcal { Z } _ { \ell } | S _ { \ell } ^ { \pi } } \sum _ { t \in \mathcal { Z } _ { \ell } } \frac { K \left( \frac { p _ { t } - \pi ( x _ { t } ) } { h } \right) } { h \pi _ { 0 } ( p _ { t } | x _ { t } ) } \mathbf { 1 } \{ r _ { t } \leq s \} ,
$$

which implies that

$$
\mathbb { E } _ { \hat { \mathcal { G } } _ { \varepsilon } } \left[ \exp ( - r / \alpha ) \right] = \mathbb { E } _ { \hat { \mathcal { F } } _ { \varepsilon } } \left[ \exp ( - r / \alpha ) \frac { \mathrm { d } \hat { \mathbb { Q } } _ { \varepsilon } ^ { r } } { \mathrm { d } \hat { \mathbb { P } } _ { \varepsilon } } ( t ) \right] = \frac { 1 } { | \mathcal { Z } _ { \varepsilon } | S _ { \varepsilon } ^ { \pi } } \sum _ { t \in \mathcal { Z } _ { \varepsilon } } \frac { K \left( \frac { n _ { 1 } - \pi ( \alpha ) } { k } \right) } { h \pi _ { 0 } ( p _ { t } | x _ { t } ) } \exp ( - r _ { t } / \alpha ) , ~ \bar { V } _ { 1 } ( \pi , \alpha ) = \frac { 1 } { L } \sum _ { \varepsilon = 1 } ^ { L } \mathbb { E } _ { \hat { \mathcal { G } } _ { \varepsilon } } \left[ \exp ( - r / \alpha ) \right] .
$$

Similarly, using the empirical measures $\hat { \mathbb { P } } _ { \ell }$ and $\hat { \mathbb { Q } } _ { \ell } ^ { \pi }$ defined above, we can rewrite

$$
\tilde { V } _ { 2 } ( \pi , \alpha ) = \frac { 1 } { L } \sum _ { \ell = 1 } ^ { L } \Big ( \mathbb { E } _ { \mathbb { P } _ { \ell } } [ m ( X , \pi ( X ) ; \alpha ) ] - \mathbb { E } _ { \mathbb { Q } _ { \ell } ^ { \star } } [ m ( X , P ; \alpha ) ] \Big ) , \quad \tilde { V } _ { 2 } ^ { h } ( \pi , \alpha ) = \mathbb { E } _ { \mathbb { P } _ { 0 } } [ m ( X , \pi ( X ) ; \alpha ) ] - \mathbb { E } _ { \mathbb { Q } _ { \ell } ^ { \star } } [ m ( X , P ; \alpha ) ] .
$$

Picking $\begin{array} { r } { h ^ { s } \le \frac { \underline { { \rho } } } { 2 C _ { 1 } L _ { m } } } \end{array}$ from (41), we have $\left| \tilde { W } _ { h } ( \pi , \alpha ) - \tilde { W } _ { h } ^ { \prime } ( \pi , \alpha ) \right| \le \underline { { \rho } } / 2$ , and from (43), it follows from $\tilde { W } _ { h } ( \pi , \alpha ) \geq$ $\tilde { W } _ { h } ^ { \prime } ( \pi , \alpha ) - \underline { { \rho } } / 2 \geq \underline { { \rho } } / 2$ . Furthermore, by a similar process in (65), we can also get

$$
\left| \tilde { W } _ { T } ( \pi , \alpha ) - \tilde { W } _ { h } ( \pi , \alpha ) \right| = \mathcal { O } \left( \frac { d _ { \Pi } \log ( 1 / \epsilon ) } { T h } \right) ,
$$

with the probability at least $1 \ - \ \epsilon .$ To proceed, we can pick $\begin{array} { r } { \Omega \left( \frac { d _ { \Pi } \log \left( 1 / \epsilon \right) } { T \underline { { \rho } } } \right) ~ \leq ~ h ~ \leq ~ \left( \frac { \underline { { \rho } } } { 2 C _ { 1 } L _ { m } } \right) ^ { 1 / s } } \end{array}$ such that $\left| \tilde { W } _ { T } ( \pi , \alpha ) - \tilde { W } _ { h } ( \pi , \alpha ) \right| \le \underline { { \rho } } / 4$ . By a similar argument in proof of Lemma C.2, we can partition α $\left| \tilde { W } _ { T } ( \pi , \alpha ) - \tilde { W } _ { h } ( \pi , \alpha ) \right|$ into three terms.

$$
\begin{array} { r l } & { \displaystyle | \tilde { R } _ { \mathrm { D R O } } ^ { h } ( \pi ) - \tilde { R } _ { \mathrm { D R O } } ( \pi ) | = | \operatorname* { s u p } _ { \alpha > 0 } \{ - \alpha \log \tilde { W } _ { h } ( \pi , \alpha ) - \alpha \delta \} - \operatorname* { s u p } _ { \alpha > 0 } \{ - \alpha \log \tilde { W } _ { T } ( \pi , \alpha ) - \alpha \delta \} | } \\ & { \leq \operatorname* { s u p } _ { \alpha > 0 } | \alpha \log ( \frac { \tilde { W } _ { T } ( \pi , \alpha ) } { \tilde { W } _ { h } ( \pi , \alpha ) } ) | \leq \frac { 4 } { \underline { { \rho } } } \operatorname* { s u p } _ { \alpha > 0 } \{ \alpha | \tilde { W } _ { T } ( \pi , \alpha ) - \tilde { W } _ { h } ( \pi , \alpha ) | \} } \\ & { \leq \frac { 4 } { \underline { { \rho } } } \operatorname* { s u p } _ { \alpha > 0 } \{ \alpha | \frac { 1 } { L } \sum _ { \ell = 1 } ^ { L } ( \mathbb { E } _ { \tilde { Q } _ { T } ^ { \cdot } } - \mathbb { E } _ { \mathbb { Q } ^ { \cdot } } ) [ \exp ( - r ( X , P , Z ) / \alpha ) ] | \} + \frac { 4 } { \underline { { \rho } } } \operatorname* { s u p } _ { \alpha > 0 } \{ \alpha | \frac { 1 } { L } \sum _ { \ell = 1 } ^ { L } ( \mathbb { E } _ { \tilde { Q } _ { t } ^ { \cdot } } - \mathbb { E } _ { \mathbb { Q } ^ { \cdot } } ) [ m ( X , P ; \alpha ) ] | \} } \\ &  \quad + \frac { 4 } { \underline { { \rho } } } \operatorname* { s u p } _ { \alpha > 0 } \{ \alpha | \frac { 1 } { L } \sum _ { \ell = 1 } ^ { L } ( \mathbb { E } _ { \tilde { \tau } _ { \ell } } - \mathbb { E } _ { \mathbb { P } _ { 0 } } ) [ m ( X , \pi ( X ) \end{array}\tag{44}
$$

Note that for any non-negative random variable $S \in [ 0 , 1 ]$ with CDF $F _ { S }$ , and a differentiable function $g ( s )$ , we have $\begin{array} { r } { { \mathbb E } [ g ( S ) ] = g ( 0 ) + \int _ { 0 } ^ { 1 } g ^ { \prime } ( s ) ( 1 - F _ { S } ( s ) ) d s } \end{array}$ . In our setting, $g ( s ) = \exp ( - s / \alpha )$ . Using the definition of the Kolmogorov-Smirnov distance $\begin{array} { r } { \tilde { d _ { K S } } ( F _ { 1 } , F _ { 2 } ) = \operatorname* { s u p } _ { s } | F _ { 1 } ( s ) - F _ { 2 } ( s ) | . } \end{array}$ , we bound the first term $x ( \mathbb { E } _ { \hat { \mathbb { Q } } _ { o } ^ { \pi } } - \mathbb { E } _ { \mathbb { Q } ^ { \pi } } ) \left[ \exp ( - r ( P , Z ) / \alpha ) \right]$

$$
\begin{array} { r l } & { \left| \alpha ( \mathbb { E } _ { \hat { \mathbb { Q } } _ { \varepsilon } ^ { \pi } } - \mathbb { E } _ { \mathbb { Q } ^ { \pi } } ) [ e ^ { - r / \alpha } ] \right| = \left| \alpha \int _ { 0 } ^ { 1 } \left( - \frac { 1 } { \alpha } e ^ { - s / \alpha } \right) ( F _ { \mathbb { Q } ^ { \pi } } ( s ) - F _ { \hat { \mathbb { Q } } _ { \varepsilon } ^ { \pi } } ( s ) ) d s \right| } \\ & { \qquad \le d _ { K S } ( F _ { \mathbb { Q } ^ { \pi } } , F _ { \hat { \mathbb { Q } } _ { \varepsilon } ^ { \pi } } ) \cdot \displaystyle \int _ { 0 } ^ { 1 } e ^ { - s / \alpha } d s = d _ { K S } ( F _ { \mathbb { Q } ^ { \pi } } , F _ { \hat { \mathbb { Q } } _ { \varepsilon } ^ { \pi } } ) \cdot \alpha ( 1 - e ^ { - 1 / \alpha } ) . } \end{array}
$$

Taking the limit by L’Hopital,ˆ lim $\begin{array} { r } { \mathbf { \Lambda } _ { \alpha \to \infty } \frac { 1 - e ^ { - 1 / \alpha } } { 1 / \alpha } = 1 } \end{array}$ . Thus, we have $\begin{array} { r } { \operatorname* { s u p } _ { \alpha > 0 } \left\{ \alpha \left( 1 - e ^ { - 1 / \alpha } \right) \right\} = 1 } \end{array}$ from the monotone increasing. Therefore,

$$
\operatorname* { s u p } _ { \alpha > 0 } \left. \alpha ( \mathbb { E } _ { \hat { \mathbb { Q } } _ { \ell } ^ { \pi } } - \mathbb { E } _ { \mathbb { Q } ^ { \pi } } ) [ e ^ { - r / \alpha } ] \right. \leq d _ { K S } ( F _ { \mathbb { Q } ^ { \pi } } , F _ { \hat { \mathbb { Q } } _ { \ell } ^ { \pi } } ) .
$$

Applying the same logic to $\alpha ( \mathbb { E } _ { \hat { \mathbb { Q } } _ { \sigma } ^ { \pi } } - \mathbb { E } _ { \mathbb { Q } ^ { \pi } } ) \left[ m ( X , P ; \alpha ) \right]$

$$
\begin{array} { r l } & { \left| \alpha ( \mathbb { E } _ { \hat { \mathbb { Q } } _ { \varepsilon } ^ { \pi } } - \mathbb { E } _ { \mathbb { Q } ^ { \pi } } ) \left[ m ( X , P ; \alpha ) \right] \right| = \left| \alpha \displaystyle \int _ { 0 } ^ { 1 } \left( - \frac { 1 } { \alpha } e ^ { - s / \alpha } \right) ( F _ { \mathbb { Q } ^ { \pi } } ( s ) - F _ { \hat { \mathbb { Q } } _ { \varepsilon } ^ { \pi } } ( s ) ) d s \right| } \\ & { \qquad \le d _ { K S } ( F _ { \mathbb { Q } ^ { \pi } } , F _ { \hat { \mathbb { Q } } _ { \varepsilon } ^ { \pi } } ) \cdot \displaystyle \int _ { 0 } ^ { 1 } e ^ { - s / \alpha } d s = d _ { K S } ( F _ { \mathbb { Q } ^ { \pi } } , F _ { \hat { \mathbb { Q } } _ { \varepsilon } ^ { \pi } } ) \cdot \alpha ( 1 - e ^ { - 1 / \alpha } ) . } \end{array}
$$

Taking the average over $\ell = 1 , \ldots , L$ , we have

$$
| \tilde { R } _ { \mathrm { D R O } } ^ { h } ( \pi ) - \tilde { R } _ { \mathrm { D R O } } ( \pi ) | \leq \frac { 4 } { \underline { { \rho } } } \frac { 1 } { L } \sum _ { \ell = 1 } ^ { L } \left( 2 d _ { K S } ( F _ { \mathbb { Q } ^ { \pi } } , F _ { \hat { \mathbb { Q } } _ { \ell } ^ { \pi } } ) + d _ { K S } ( F _ { \mathbf { P } _ { 0 } } , F _ { \hat { \mathbb { P } } _ { \ell } } ) \right) .
$$

Lemma C.4. Under Assumptions 2.2 and 3.2, we have with probability at least $1 - 3 \epsilon$

$$
\operatorname* { s u p } _ { \pi \in \Pi } | \tilde { R } _ { \mathrm { D R O } } ^ { h } ( \pi ) - \tilde { R } _ { \mathrm { D R O } } ( \pi ) | \lesssim \frac { d _ { \Pi } L } { T h } + \sqrt { \frac { d _ { \Pi } L } { T h } } + \sqrt { \frac { \log ( 1 / \epsilon ) L } { T h } } + \frac { \log ( 1 / \epsilon ) L } { T h } ,
$$

Proof of Lemma C.4. We decompose the KS distance in Lemma C.3 using the triangle inequality,

$$
\begin{array} { r l } & { d _ { K S } ( F _ { \mathbb { Q } ^ { n } } , F _ { \hat { \mathbb { Q } } _ { \varepsilon } ^ { n } } ) = \underset { s \in [ 0 , 1 ] } { \operatorname* { s u p } } \left| \frac { 1 } { T S _ { \varepsilon } ^ { n } } \sum _ { t \in \mathcal { Z } _ { \varepsilon } } \frac { K \left( \frac { p _ { t } - \pi ( x _ { t } ) } { h } \right) } { h \pi _ { 0 } ( p _ { t } | x _ { t } ) } { \mathbf 1 } \{ r _ { t } \leq s \} - \mathbb { E } \left[ \frac { K \left( \frac { P - \pi ( X ) } { h } \right) } { h \pi _ { 0 } ( P | X ) } { \mathbf 1 } \mathbf { \{ \{ } r ( X , P ; \alpha ) \leq s \}  \right\right|}]   \\ & { \leq \underset { s \in [ 0 , 1 ] } { \operatorname* { s u p } } \left| \frac { 1 } { | \mathcal { Z } _ { \varepsilon } | } \sum _ { t \in \mathcal { Z } _ { \varepsilon } } \frac { K \left( \frac { p _ { t } - \pi ( x _ { t } ) } { h } \right) } { h \pi _ { 0 } ( p _ { t } | x _ { t } ) } { \mathbf 1 } \{ r _ { t } \leq s \} - \mathbb { E } \left[ \frac { K \left( \frac { P - \pi ( X ) } { h } \right) } { h \pi _ { 0 } ( P | X ) } { \mathbf 1 } \{ r ( X , P ; \alpha ) \leq s \} \right] \right| } \\ & { \quad \quad + \underset { s \in [ 0 , 1 ] } { \operatorname* { s u p } } \left| \frac { S _ { \varepsilon } ^ { n } - 1 } { | \mathcal { Z } _ { \varepsilon } | S _ { \varepsilon } ^ { n } } \sum _ { t \in \mathcal { Z } _ { \varepsilon } } \frac { K \left( \frac { p _ { t } - \pi ( x _ { t } ) } { h } \right) } { h \pi _ { 0 } ( p _ { t } | x _ { t } ) } { \mathbf 1 } \{ r _ { t } \leq s \} \right| , } \end{array}\tag{45}
$$

$$
d _ { K S } \big ( F _ { \mathbf { P } _ { 0 } } , F _ { \widehat { \mathbb { P } } _ { \ell } } \big ) = \operatorname* { s u p } _ { s \in [ 0 , 1 ] } \left. \frac { 1 } { \vert { \mathcal { Z } } _ { \ell } \vert } \sum _ { t \in { \mathcal { I } } _ { \ell } } \mathbf { 1 } \big \{ r ( x _ { t } , \pi ( x _ { t } ) ; \alpha ) \leq s \big \} - { \mathbb { E } } \left[ \mathbf { 1 } \big \{ r ( X , \pi ( X ) ; \alpha ) \leq s \big \} \right] \right. .\tag{46}
$$

Note that

$$
\left| \frac { S _ { \ell } ^ { \pi } - 1 } { | \mathcal { Z } _ { \ell } | S _ { \ell } ^ { \pi } } \sum _ { t \in \mathcal { Z } _ { \ell } } \frac { K \left( \frac { p _ { t } - \pi ( x _ { t } ) } { h } \right) } { h \pi _ { 0 } ( p _ { t } | x _ { t } ) } { \bf 1 } \{ r _ { t } \leq s \} \right| \leq | S _ { \ell } ^ { \pi } - 1 | \mathrm { a n d } \mathbb { E } [ S _ { \ell } ^ { \pi } ] = 1 .\tag{47}
$$

Define

$$
\begin{array} { l l l } { \displaystyle f _ { \pi } ( x , p ) = \frac { 1 } { h \pi _ { 0 } ( p | x ) } K \left( \frac { p - \pi ( x ) } { h } \right) , } & { \displaystyle f _ { \pi , s } ^ { 1 } ( x , p , z ) \triangleq \frac { 1 } { h \pi _ { 0 } ( p | x ) } K \left( \frac { p - \pi ( x ) } { h } \right) \mathbf { 1 } \big \{ r ( x , p , z ) \leq s \big \} } \\ { \displaystyle f _ { \pi , s } ^ { 2 } ( x , z ) \triangleq \mathbf { 1 } \big \{ r ( x , \pi ( x ) , z ) \leq s \big \} , } \end{array}
$$

and the corresponding function classes

$$
\begin{array} { r l } & { \mathcal { F } _ { \Pi } : = \left\{ f _ { \pi } | \pi \in \Pi \right\} , \quad \mathcal { G } _ { \Pi } : = \left\{ g _ { \pi } = f _ { \pi } - \mathbb { E } [ f _ { \pi } ] \Big | \pi \in \Pi \right\} , } \\ & { \mathcal { F } _ { \Pi , s } ^ { 1 } \triangleq \left\{ f _ { \pi , s } ^ { 1 } \Big | \pi \in \Pi , s \in [ 0 , 1 ] \right\} , \quad \mathcal { G } _ { \Pi , s } ^ { 1 } \triangleq \left\{ g _ { \pi , s } ^ { 1 } = f _ { \pi , s } ^ { 1 } - \mathbb { E } [ f _ { \pi , s } ^ { 1 } ] \Big | \pi \in \Pi , s \in [ 0 , 1 ] \right\} , } \\ & { \mathcal { F } _ { \Pi , s } ^ { 2 } \triangleq \left\{ f _ { \pi , s } ^ { 2 } ( x , z ) = f _ { \pi , s } ^ { 2 } - \mathbb { E } [ f _ { \pi , s } ^ { 2 } ] \Big | \pi \in \Pi , s \in [ 0 , 1 ] \right\} , \quad \mathcal { G } _ { \Pi , s } ^ { 2 } \triangleq \left\{ g _ { \pi , s } ^ { 2 } = f _ { \pi , s } ^ { 2 } - \mathbb { E } [ f _ { \pi , s } ^ { 2 } ] \Big | \pi \in \Pi , s \in [ 0 , 1 ] \right\} . } \end{array}
$$

Thus, we next provide the upper bounds for

$$
\operatorname* { s u p } _ { \tau \in \Pi } \left| \frac { 1 } { | \mathcal { Z } _ { t } | } \sum _ { t \in \mathcal { Z } _ { t } } f _ { \pi } ( x _ { t } , p _ { t } ) - \mathbb { E } [ f _ { \pi } ] \right| , \operatorname* { s u p } _ { \tau \in \Pi , s \in [ 0 , 1 ] } \left| \frac { 1 } { | \mathcal { Z } _ { t } | } \sum _ { t \in \mathcal { Z } _ { t } } f _ { \pi , s } ^ { 1 } ( x _ { t } , p _ { t } , z _ { t } ) - \mathbb { E } [ f _ { \pi , s } ^ { 1 } ] \right| , \operatorname* { s u p } _ { \tau \in \Pi , s \in [ 0 , 1 ] } \left| \frac { 1 } { | \mathcal { Z } _ { t } | } \sum _ { t \in \mathcal { Z } _ { t } } f _ { \pi , s } ^ { 2 } ( x _ { t } ) - \mathbb { E } [ f _ { \pi , s } ^ { 2 } ] \right| .
$$

Step 1. Bound for $\begin{array} { r } { \operatorname* { s u p } _ { \pi \in \Pi } \left| \frac { 1 } { | { \mathcal { Z } } _ { \ell } | } \sum _ { t \in { \mathcal { Z } } _ { \ell } } f _ { \pi } ( x _ { t } , p _ { t } ) - { \mathbb { E } } [ f _ { \pi } ] \right| } \end{array}$ . Note that $f _ { \pi }$ is uniformly bounded by a constant $\begin{array} { r } { B _ { \Pi } : = \frac { C _ { 2 } } { h \eta } } \end{array}$ due to Assumption 3.2 and 2.2. Let $\{ ( x _ { t } , p _ { t } ) \} _ { t \in \mathbb { Z } _ { \ell } }$ be i.i.d. random variables. Our goal is to derive a high-probability upper bound for

$$
\operatorname* { s u p } _ { \pi \in \Pi } \left. \frac { 1 } { \vert \mathcal { Z } _ { \ell } \vert } \sum _ { t \in \mathcal { T } _ { \ell } } f _ { \pi } ( x _ { t } , p _ { t } ) - \mathbb { E } [ f _ { \pi } ] \right. = \frac { 1 } { \vert \mathcal { Z } _ { \ell } \vert } \operatorname* { s u p } _ { g \in \mathcal { G } _ { \Pi } } \left. \sum _ { t \in \mathcal { Z } _ { \ell } } g ( x _ { t } , p _ { t } ) \right. \equiv \Delta .\tag{48}
$$

First, we provide the second-moment bound.

$$
\begin{array} { l } { \displaystyle \operatorname* { s u p } _ { \pi \in \Pi } \mathbb { E } [ f _ { \pi } ^ { 2 } ] = \displaystyle \operatorname* { s u p } _ { \pi \in \Pi } \int _ { \mathcal { X } } \int _ { \mathcal { P } } \frac { 1 } { h ^ { 2 } \pi _ { 0 } ( p | x ) ^ { 2 } } K ^ { 2 } \left( \frac { p - \pi ( x ) } { h } \right) \pi _ { 0 } ( p | x ) d p d Q ( x ) } \\ { = \displaystyle \operatorname* { s u p } _ { \pi \in \Pi } \int _ { \mathcal { X } } \int _ { \mathcal { P } } \frac { 1 } { h ^ { 2 } \pi _ { 0 } ( p | x ) } K ^ { 2 } \left( \frac { p - \pi ( x ) } { h } \right) d p d Q ( x ) \le \displaystyle \operatorname* { s u p } _ { \pi \in \Pi } \frac { 1 } { \eta h } \int _ { \mathcal { X } } \int _ { - \infty } ^ { \infty } K ^ { 2 } ( u ) d u \le \frac { C _ { 2 } } { h \eta } : = \sigma ^ { 2 } } \end{array}\tag{49}
$$

where the last inequality follows from $\textstyle \int K ^ { 2 } ( u ) d u < C _ { 2 }$ in Assumption 3.2 and $\begin{array} { r } { \sigma ^ { 2 } : = \frac { C _ { 2 } } { h \eta } } \end{array}$

From the definition of covering number and uniform covering number (see Definition D.2), and Lemma D.4, we have

$$
\mathcal { N } ( \epsilon , \mathcal { F } _ { \Pi } , L _ { 2 } ( P _ { T ^ { \prime } } ) ) \leq \mathcal { N } _ { 1 } \left( \frac { \epsilon ^ { 2 } } { B _ { \Pi } } , \mathcal { F } _ { \Pi } , T ^ { \prime } \right) \leq \mathcal { N } _ { 1 } \left( \frac { \epsilon ^ { 2 } } { B _ { \Pi } L _ { K } } , \Pi , T ^ { \prime } \right) ,
$$

which is directly by letting

$$
\mathcal { F } _ { 0 } = \Pi , \mathcal { F } _ { 1 } = \left\{ ( x , p ) \mapsto \frac { p - \pi ( x ) } { h } | \pi \in \Pi \right\} , \mathcal { F } _ { \Pi } = \Big \{ f _ { \pi } : ( x , p ) \mapsto \frac { K \left( \frac { p - \pi ( x ) } { h } \right) } { h \pi _ { 0 } ( p | x ) } \Big | \pi \in \Pi \Big \}\tag{50}
$$

and $\mathrm { P d i m } ( \mathcal { F } _ { 0 } ) = \mathrm { P d i m } ( \Pi ) : = d _ { \Pi }$ . Combined with Lemma D.3(ii), we conclude that

$$
\log \mathcal { N } ( \epsilon , \mathcal { F } _ { \Pi } , L _ { 1 } ( P _ { T ^ { \prime } } ) ) \leq \log \mathcal { N } ( \epsilon , \mathcal { F } _ { \Pi } , L _ { 2 } ( P _ { T ^ { \prime } } ) ) \lesssim d _ { \Pi } \log \left( \frac { B _ { \Pi } } { \epsilon } \right) .
$$

On the other hand, from Lemma D.3(iii), we have

$$
\log \mathcal { M } ( \epsilon , \mathcal { F } _ { \Pi } , L _ { 1 } ( P _ { T ^ { \prime } } ) ) \gtrsim \mathrm { P d i m } ( \mathcal { F } _ { \Pi } ) \log ( 1 / \epsilon ) ,
$$

for $T ^ { \prime } \geq \mathrm { P d i m } ( \mathcal { F } _ { \Pi } )$ . Combining the above two, which both hold for all sufficiently small $\epsilon > 0$ , we can get

$$
\mathrm { P d i m } ( \mathcal { F } _ { \Pi } ) \lesssim d _ { \Pi } .\tag{51}
$$

By setting $T ^ { \prime } = | \mathcal { T } _ { \ell } |$ in Lemma C.6, we obtain

$$
\mathbb { E } [ \Delta ] \equiv \frac { 1 } { | { \mathcal { Z } _ { \ell } } | } \mathbb { E } \left[ \operatorname* { s u p } _ { g \in { \mathcal { G } _ { \Pi } } } \left| \sum _ { { t } \in { \mathcal { Z } _ { \ell } } } g ( x _ { t } , p _ { t } ) \right| \right] \lesssim \frac { B _ { \Pi } d _ { \Pi } } { | { \mathcal { Z } _ { \ell } } | } + \sqrt { \frac { \sigma ^ { 2 } d _ { \Pi } } { | { \mathcal { Z } _ { \ell } } | } } .
$$

Now, we use Talagrand Inequality (see Lemma D.1) with deviation $\beta ( \epsilon ) > 0$

$$
\mathbb { P } \left( \Delta - \mathbb { E } [ \Delta ] \ge \beta ( \epsilon ) \right) \le 2 \exp \left\{ - \frac { \beta ( \epsilon ) ^ { 2 } | { \mathcal { Z } } _ { \ell } | } { 8 e \sigma ^ { 2 } + 1 6 e B _ { \Pi } \mathbb { E } [ \Delta ] + 4 B _ { \Pi } \beta ( \epsilon ) } \right\} .
$$

To get a bound holding with probability $1 - \epsilon .$ we can pick $\begin{array} { r } { \beta ( \epsilon ) \lesssim \sqrt { \frac { \sigma ^ { 2 } \log ( 1 / \epsilon ) } { | \mathcal { Z } _ { \ell } | } } + \frac { B _ { \Pi } \log ( 1 / \epsilon ) } { | \mathcal { Z } _ { \ell } | } } \end{array}$ . Hence, we have

$$
\operatorname* { s u p } _ { \pi \in \Pi } \left| \frac { 1 } { | \mathbb { Z } _ { \ell } | } \sum _ { t \in \mathbb { Z } _ { \ell } } f _ { \pi } ( x _ { t } , p _ { t } ) - \mathbb { E } [ f _ { \pi } ] \right| \lesssim \frac { d _ { \Pi } } { | \mathbb { Z } _ { \ell } | h } + \sqrt { \frac { d _ { \Pi } } { | \mathbb { Z } _ { \ell } | h } } + \sqrt { \frac { \log ( 1 / \epsilon ) } { | \mathbb { Z } _ { \ell } | h } } + \frac { \log ( 1 / \epsilon ) } { | \mathbb { Z } _ { \ell } | h } ,
$$

with probability at least $1 - \epsilon$

Step 2. Bound for $\begin{array} { r } { \operatorname* { s u p } _ { \pi \in \Pi , s \in [ 0 , 1 ] } \Big | \frac { 1 } { | \mathcal { T } _ { \ell } | } \sum _ { t \in \mathcal { T } _ { \ell } } f _ { \pi , s } ^ { 1 } ( x _ { t } , p _ { t } , z _ { t } ) - \mathbb { E } [ f _ { \pi , s } ^ { 1 } ] \Big | . } \end{array}$ . It is same as Step 1 that $\left| f _ { \pi , s } ^ { 1 } \right| \le B _ { \Pi }$ , and the second-moment bound is bounded by $\begin{array} { r } { \sigma ^ { 2 } : = \frac { C _ { 2 } } { h \eta } } \end{array}$ . We first claim that

$$
\mathcal { N } ( \epsilon , \mathcal { F } _ { \Pi , s } ^ { 1 } , L _ { 2 } ( P _ { | \mathcal { Z } _ { \ell } | } ) ) \le \mathcal { N } _ { 2 } ( \epsilon / 2 , \mathcal { F } _ { \Pi } , | \mathcal { Z } _ { \ell } | ) \operatorname* { s u p } _ { \mathbf { P } } \mathcal { N } ( \eta \epsilon / ( 2 C _ { 2 } ) , \mathcal { E } , \| \cdot \| _ { \mathbf { P } } ) ,
$$

where $\mathcal { E } = \{ i _ { s } ( \epsilon ) \triangleq \mathbf { 1 } \{ \epsilon \leq s \} | s \in [ 0 , 1 ] \}$ . For ease of notation, let

$$
N _ { 2 , \mathscr { F } } ( \epsilon ) : = \mathscr { N } _ { 2 } ( \epsilon / 2 , \mathscr { F } _ { \Pi } , | \mathscr { Z } _ { \ell } | ) , \quad N _ { \mathscr { E } } ( \epsilon ) : = \operatorname* { s u p } _ { \mathbf { P } } \mathscr { N } ( \eta \epsilon / ( 2 C _ { 2 } ) , \mathscr { E } , \| \cdot \| _ { \mathbf { P } } ) .
$$

Suppose $\left\{ f _ { \pi _ { 1 } } , . . . , f _ { \pi _ { N _ { 2 , \mathcal { F } } ( \epsilon ) } } \right\}$ is a $\epsilon / 2 \AA$ -cover for $\mathcal { F } _ { \Pi }$ , and $\{ { \bf 1 } \{ \epsilon \le s _ { 1 } \} , . . . , { \bf 1 } \{ \epsilon \le s _ { N _ { \mathcal { E } } ( \epsilon ) } \} \}$ is a $\epsilon / 2 \AA$ -cover for $\mathcal { E } ^ { \mathcal { C } }$ under the distance $\begin{array} { r } { \hat { \mathbb P } _ { \ell } , } \end{array}$ defined as

$$
\hat { \mathbb P } _ { \ell } \triangleq \frac { 1 } { | \mathcal T _ { \ell } | } \sum _ { t \in \mathcal T _ { \ell } } \Delta ( r ( x _ { t } , p _ { t } , z _ { t } ) ) .
$$

Then, we can prove that $\mathcal { F } _ { \Pi , s } ^ { 1 , \epsilon }$ is a ϵ-cover set for $\mathcal { F } _ { \Pi , s } ^ { 1 } ,$ where $\mathcal { F } _ { \Pi , s } ^ { 1 , \epsilon }$ is defined as

$$
\mathcal { F } _ { \Pi , s } ^ { 1 , \epsilon } \triangleq \left\{ f _ { \pi _ { i } , s _ { j } } ^ { 1 } = \frac { K \left( \frac { p - \pi _ { i } ( x ) } { h } \right) } { h \pi _ { 0 } ( p | x ) } \mathbf { 1 } \{ r ( x , p , z ) \leq s _ { j } \} \left| i \leq N _ { 2 , \mathcal { F } } ( \epsilon ) , j \leq N _ { \mathcal { E } } ( \epsilon ) \right. \right\} .
$$

For $f _ { \pi , s } ^ { 1 } ( x , p , z ) \in \mathcal { F } _ { \Pi , s } ^ { 1 } \mathrm { . }$ , we can pick $f _ { \tilde { \pi } , \tilde { s } } ^ { 1 } ( x , p , z ) \in \mathcal { F } _ { \Pi , s } ^ { 1 , \epsilon }$

$$
\begin{array} { r } { \| f _ { \pi } ^ { 1 } - f _ { \widetilde { \pi } } ^ { 1 } \| _ { L _ { 2 } ( P _ { | \mathcal { Z } _ { \ell } | } ) } \leq \epsilon / 2 \quad \mathrm { ~ a n d ~ } \quad \| \mathbf { 1 } \{ r \leq s \} - \mathbf { 1 } \{ r \leq \widetilde { s } \} \| _ { \hat { \mathbf { P } } _ { \ell } } \leq \eta \epsilon / ( 2 C _ { 2 } ) . } \end{array}
$$

Then, we have

$$
\| f _ { \pi , s } ^ { 1 } - f _ { \tilde { \pi } , \tilde { s } } ^ { 1 } \| _ { L _ { 2 } ( P _ { | \mathcal { Z } _ { \ell } | } ) } = \sqrt { \frac { 1 } { | \mathcal { Z } _ { \ell } | } \sum _ { t \in \mathcal { Z } _ { \ell } } \Big ( f _ { \pi , s } ^ { 1 } ( x _ { t } , p _ { t } , z _ { t } ) - f _ { \tilde { \pi } , \tilde { s } } ^ { 1 } ( x _ { t } , p _ { t } , z _ { t } ) \Big ) ^ { 2 } }
$$

$$
= \sqrt { \frac { 1 } { | \mathcal { Z } _ { \ell } | } \sum _ { t \in \mathcal { Z } _ { \ell } } \left( f _ { \pi } ^ { 1 } ( x _ { t } , p _ { t } , z _ { t } ) \mathbf { 1 } \{ r ( x _ { t } , p _ { t } , z _ { t } ) \leq s \} - f _ { \pi } ^ { 1 } ( x _ { t } , p _ { t } , z _ { t } ) \mathbf { 1 } \{ r ( x _ { t } , p _ { t } , z _ { t } ) \leq \tilde { s } \} \right) ^ { 2 } }
$$

$$
= \sqrt { \frac { 1 } { \left| \mathbb { Z } _ { t } \right| } \sum _ { t \in \mathbb { Z } _ { \varepsilon } } ( f _ { \pi , s } ^ { 1 } ( x _ { t } , p _ { t } , z _ { t } ) ) ^ { 2 } \left( \mathbf { 1 } \{ r ( x _ { t } , p _ { t } , z _ { t } ) \leq s \} - \mathbf { 1 } \{ r ( x _ { t } , p _ { t } , z _ { t } ) \leq \bar { s } \} \right) ^ { 2 } + \mathbf { 1 } \{ r ( x _ { t } , p _ { t } , z _ { t } ) \leq \bar { s } \} \left( f _ { \pi } ^ { 1 } ( x _ { t } , p _ { t } , z _ { t } ) - f _ { \pi } ^ { 1 } ( x _ { t } , p _ { t } , z _ { t } ) \right) ^ { 2 } }
$$

$$
\leq \frac { C _ { 2 } } { \eta } \sqrt { \frac { 1 } { \vert \mathcal { Z } _ { \ell } \vert } \sum _ { t \in \mathcal { Z } _ { \ell } } \left( \mathbf { 1 } \{ r ( x _ { t } , p _ { t } , z _ { t } ) \leq s \} - \mathbf { 1 } \{ r ( x _ { t } , p _ { t } , z _ { t } ) \leq \tilde { s } \} \right) ^ { 2 } } + \sqrt { \frac { 1 } { \vert \mathcal { Z } _ { \ell } \vert } \sum _ { t \in \mathcal { Z } _ { \ell } } \left( f _ { \overline { { \pi } } } ^ { 1 } ( x _ { t } , p _ { t } , z _ { t } ) - f _ { \overline { { \pi } } } ^ { 1 } ( x _ { t } , p _ { t } , z _ { t } ) \right) ^ { 2 } }
$$

$$
\leq \epsilon
$$

From Lemma 19.15 and Example 19.16 in van der Vaart (1998), we have sup<sub>P</sub> $\mathcal { N } ( \epsilon , \mathcal { E } , \| \cdot \| _ { \mathbf { P } } ) \lesssim \left( \frac { 1 } { \epsilon } \right) ^ { 2 }$ . Hence, we obtain

$$
\mathcal { N } ( \epsilon , \mathcal { F } _ { \Pi , s } ^ { 1 } , L _ { 2 } ( P _ { | \mathcal { Z } _ { \ell } | } ) ) \lesssim \mathcal { N } _ { 1 } \left( \frac { \epsilon ^ { 2 } } { 4 B ( \mathcal { F } _ { \Pi } ) } , \mathcal { F } _ { \Pi } , | \mathcal { Z } _ { \ell } | \right) \cdot \left( \frac { 1 } { \epsilon } \right) ^ { 2 } ,\tag{52}
$$

which is directly from Lemma D.3(v), and $B ( \mathcal { F } _ { \Pi } )$ is the uniform maximum of $\mathcal { F } _ { \Pi }$ . Combined with Lemma D.3(ii), we have

$$
\log \mathcal { N } ( \epsilon , \mathcal { F } _ { \Pi , s } ^ { 1 } , L _ { 2 } ( P _ { | \mathcal { Z } _ { \ell } | } ) ) \lesssim \mathrm { P d i m } ( \mathcal { F } _ { \Pi } ) \log \left( \frac { 1 } { \epsilon } \right) .
$$

On the other hand, from Lemma D.3(iii), we have

$$
\log \mathcal { M } _ { 1 } ( \epsilon , \mathcal { F } _ { \Pi , s } ^ { 1 } , | \mathcal { T } _ { \ell } | ) \gtrsim \mathrm { P d i m } ( \mathcal { F } _ { \Pi , s } ^ { 1 } ) \log ( 1 / \epsilon ) ,
$$

for $\mathcal { T } _ { \ell } \geq \mathrm { P d i m } ( \mathcal { F } _ { \Pi , s } ^ { 1 } )$ . Combining the above two inequalities, which both hold for all sufficiently small $\epsilon > 0 .$ , we can get

$$
\mathrm { P d i m } ( \mathcal { F } _ { \Pi , s } ^ { 1 } ) \le K \cdot \mathrm { P d i m } ( \mathcal { F } _ { \Pi } )\tag{53}
$$

for some positive constant K. Note that,

$$
\begin{array} { r } { \mathcal { N } ( \epsilon , \mathcal { G } _ { \Pi , s } ^ { 1 } , L _ { 2 } ( P _ { | \mathcal { Z } _ { \ell } | } ) ) \leq \mathcal { N } ( \epsilon / 2 , \mathcal { F } _ { \Pi , s } ^ { 1 } , L _ { 2 } ( P _ { | \mathcal { Z } _ { \ell } | } ) ) + N ( \epsilon / 2 , \overline { { \mathcal { F } } } _ { \Pi , s } ^ { 1 } , L _ { 2 } ( P _ { | \mathcal { Z } _ { \ell } | } ) ) , } \end{array}
$$

where

$$
\overline { { \mathcal { F } } } _ { \Pi , s } ^ { 1 } \triangleq \left\{ \overline { { f } } _ { \pi , s } ^ { 1 } = \mathbb { E } _ { \mathbf { P } _ { 0 } \ast \pi _ { 0 } } [ f _ { \pi , s } ^ { 1 } ( X , P , Z ) ] \ \big | \pi \in \Pi , s \in [ 0 , 1 ] \right\} ,
$$

is a function class of constant functions, satisfying

$$
\begin{array} { r } { \| \overline { { f } } _ { \pi _ { 1 } , s _ { 1 } } ^ { 1 } - \overline { { f } } _ { \pi _ { 2 } , s _ { 2 } } ^ { 1 } \| _ { L _ { 2 } ( P _ { | \mathcal { Z } _ { \ell } | } ) } = | \overline { { f } } _ { \pi _ { 1 } , s _ { 1 } } ^ { 1 } - \overline { { f } } _ { \pi _ { 2 } , s _ { 2 } } ^ { 1 } | \leq \mathbb { E } _ { \mathbf { P } _ { 0 } \ast \pi _ { 0 } } \left| f _ { \pi _ { 1 } , s _ { 1 } } ^ { 1 } ( X , P , Z ) - f _ { \pi _ { 2 } , s _ { 2 } } ^ { 1 } ( X , P , Z ) \right| . } \end{array}
$$

Consequently, we have

$$
\mathcal { N } ( \epsilon / 2 , \overline { { \mathcal { F } } } _ { \Pi , s } ^ { 1 } , L _ { 2 } ( P _ { | \mathcal { Z } _ { \ell } | } ) ) \leq \mathcal { N } ( \epsilon / 2 , \mathcal { F } _ { \Pi , s } ^ { 1 } , L _ { 1 } ( \mathbf { P } _ { 0 } * \pi _ { 0 } ) )
$$

Using Definition D.2 and Lemma D.3(iv), we have

$$
\begin{array} { r l r } { \log \mathcal { N } ( \epsilon / 2 , \mathcal { F } _ { \Pi , s } ^ { 1 } , L _ { 1 } ( { \mathbf { P } _ { 0 } * \pi _ { 0 } } ) ) } & { \le \log \mathcal { M } ( \epsilon / 2 , \mathcal { F } _ { \Pi , s } ^ { 1 } , L _ { 1 } ( { \mathbf { P } _ { 0 } * \pi _ { 0 } } ) ) } & \\ & { } & { \lesssim \mathrm { P d i m } ( \mathcal { F } _ { \Pi , s } ^ { 1 } ) \log \left( \frac { 1 } { \epsilon } \log \left( \frac { 1 } { \epsilon } \right) \right) \lesssim \mathrm { P d i m } ( \mathcal { F } _ { \Pi } ) \log \left( \frac { 1 } { \epsilon } \log \left( \frac { 1 } { \epsilon } \right) \right) } \end{array}\tag{by (53). (54}
$$

Therefore, we conclude that

$$
\begin{array} { r l } & { \log \mathcal { N } ( \epsilon , \mathcal { G } _ { \Pi , s } ^ { 1 } , L _ { 2 } ( P _ { | \mathcal { Z } _ { \epsilon } | } ) ) \leq 2 \log \mathcal { N } ( \epsilon / 2 , \mathcal { F } _ { \Pi , s } ^ { 1 } , L _ { 2 } ( P _ { | \mathcal { Z } _ { \epsilon } | } ) ) + 2 \log \mathcal { N } ( \epsilon / 2 , \overline { { \mathcal { F } } } _ { \Pi , s } ^ { 1 } , L _ { 2 } ( P _ { | \mathcal { Z } _ { \epsilon } | } ) ) } \\ & { \qquad \lesssim \mathrm { P d i m } ( \mathcal { F } _ { \Pi } ) \log \Bigg ( \frac { 1 } { \epsilon } \Bigg ) \lesssim \mathrm { P d i m } ( \Pi ) \log \Bigg ( \frac { 1 } { \epsilon } \Bigg ) \qquad \mathrm { b y ~ } ( 5 1 ) . } \end{array}
$$

Hence, we have an upper bound like (64), implying a Dudley’s integral and a Rademacher complexity bound like the proo of Lemma C.6. That is,

$$
\operatorname* { s u p } _ { \pi \in \Pi , s \in [ 0 , 1 ] } \left| \frac { 1 } { | { \cal Z } _ { \ell } | } \sum _ { t \in { \cal Z } _ { \ell } } f _ { \pi , s } ^ { 1 } ( x _ { t } , p _ { t } , z _ { t } ) - \mathbb { E } [ f _ { \pi , s } ^ { 1 } ] \right| \lesssim \frac { d _ { \Pi } } { | { \cal Z } _ { \ell } | h } + \sqrt { \frac { d _ { \Pi } } { | { \cal Z } _ { \ell } | h } } + \sqrt { \frac { \log ( 1 / \epsilon ) } { | { \cal Z } _ { \ell } | h } } + \frac { \log ( 1 / \epsilon ) } { | { \cal Z } _ { \ell } | h } ,
$$

with probability at least $1 - \epsilon$

Step 3. Bound for $\begin{array} { r } { \operatorname* { s u p } _ { \pi \in \Pi , s \in [ 0 , 1 ] } \Big | \frac { 1 } { T } \sum _ { t = 1 } ^ { T } f _ { \pi , s } ^ { 2 } ( x _ { t } , z _ { t } ) - \mathbb { E } [ f _ { \pi , s } ^ { 2 } ] \Big | } \end{array}$ . Using an analogous argument before, we can get

$$
\operatorname* { s u p } _ { \pi \in \Pi , s \in [ 0 , 1 ] } \left. \frac { 1 } { | \mathcal { Z } _ { \ell } | } \sum _ { t \in \mathcal { Z } _ { \ell } } f _ { \pi , s } ^ { 2 } ( x _ { t } , p _ { t } ) - \mathbb { E } [ f _ { \pi , s } ^ { 2 } ] \right. \lesssim \frac { d _ { \Pi } } { | \mathcal { Z } _ { \ell } | } + \sqrt { \frac { d _ { \Pi } } { | \mathcal { Z } _ { \ell } | } } + \sqrt { \frac { \log ( 1 / \epsilon ) } { | \mathcal { Z } _ { \ell } | } } + \frac { \log ( 1 / \epsilon ) } { | \mathcal { Z } _ { \ell } | } ,
$$

with probability at least $1 - \epsilon$ . Specifically, $\big | f _ { \pi , s } ^ { 2 } \big | \leq 1$ , and the second-moment bound is bounded by 1. For the indicator class $\mathcal { F } _ { \Pi , s } ^ { 2 }$ , we first note that for any binary-valued function class, its pseudo-dimension is identical to its VC-dimension (see Anthony & Bartlett, 2009). Note that

$$
\begin{array} { r l } & { \mathcal { F } _ { \Pi , s } ^ { 2 } = \{ ( x , z ) \mapsto \mathbf { 1 } \{ \pi ( x ) \cdot \mathbf { 1 } \{ z \geq \pi ( x ) - f ( x ) \} \leq s \} | \pi \in \Pi , s \in [ 0 , 1 ] \} } \\ & { \quad \quad = \mathbf { 1 } \{ ( \pi ( x ) - f ( x ) > z ) \mathrm { o r } ( \pi ( x ) - f ( x ) \leq z \mathrm { a n d } \pi ( x ) \leq s ) \} } \end{array}
$$

can be expressed as a finite Boolean combination of atomic level-set classes derived from Π. Since the VC-dimension of each atomic class is equivalent to $\mathrm { P d i m } ( \Pi ) = d _ { \Pi }$ , it follows from the closure properties of VC-classes (Lemma 2.6.17 van der Vaart & Wellner, 1996) that

$$
\mathrm { P d i m } ( \mathcal { F } _ { \Pi , s } ^ { 2 } ) = \mathrm { V C d i m } ( \mathcal { F } _ { \Pi , s } ^ { 2 } ) \lesssim d _ { \Pi } .
$$

Consequently, we can now directly apply Lemma 19.15 in van der Vaart (1998) to obtain the uniform covering number bound (52), consistent with the methodology used in Step 1 and Step 2. Note that $\vert \mathcal { I } _ { \ell } \vert = T / L$ , we get

$$
\begin{array} { r l } { \displaystyle \operatorname* { s u p } _ { \tau \in \Pi } | \tilde { R } _ { \mathrm { D R O } } ^ { h } ( \pi ) - \tilde { R } _ { \mathrm { D R O } } ( \pi ) | \lesssim \frac { 4 } { \underline { { \rho } } } \frac { 1 } { L } \sum _ { \varepsilon = 1 } ^ { L } \frac { d _ { \Pi } } { | \mathcal { Z } _ { \varepsilon } | h } + \sqrt { \frac { d _ { \Pi } } { T h } } + \sqrt { \frac { \log ( 1 / \epsilon ) } { | \mathcal { Z } _ { \varepsilon } | h } } + \frac { \log ( 1 / \epsilon ) } { | \mathcal { Z } _ { \varepsilon } | h } + \frac { d _ { \Pi } } { | \mathcal { Z } _ { \ell } | } + \sqrt { \frac { d _ { \Pi } } { | \mathcal { Z } _ { \ell } | } } } & { } \\ { \lesssim \frac { d _ { \Pi } L } { T h } + \sqrt { \frac { d _ { \Pi } L } { T h } } + \sqrt { \frac { \log ( 1 / \epsilon ) L } { T h } } + \frac { \log ( 1 / \epsilon ) L } { T h } } & { } \end{array}
$$

which completes the proof.

Lemma C.5. Let Assumptions 3.2, 2.2 and 4.2 hold. Picking $\begin{array} { r } { h \ge \Omega \left( \operatorname* { m a x } \left( \frac { d _ { \Pi } \log \left( 1 / \epsilon \right) } { T \underline { { \rho } } ^ { 2 } } , \sqrt { \frac { d _ { \Pi } + \log \left( 1 / \epsilon \right) } { 2 T / L } } \right) \right) } \end{array}$ , we have

$$
\left| \tilde { R } _ { \delta } ( \pi ) - \hat { R } _ { \delta } ( \pi ) \right| \lesssim \overline { { \alpha } } \left( \log \left( \frac { 1 } { h } \right) \cdot \left( \frac { d _ { \Pi } } { T h / L } + \sqrt { \frac { d _ { \Pi } } { T h / L } } + \sqrt { \frac { \log ( 1 / \epsilon ) } { T h / L } } + \frac { \log ( 1 / \epsilon ) } { T h / L } \right) + T ^ { - \gamma _ { \alpha } - \gamma _ { m } } + h ^ { s } \right) ,
$$

for $s = 1 , 2$ in Assumption 2.4, which holds with probability at least $1 - 4 \epsilon$

ProofofLemma C.5. For each fold ℓ, denote $\begin{array} { r } { \Delta _ { m } ^ { ( \ell ) } ( x , p ; \alpha ) = \hat { m } ^ { ( \ell ) } ( x , p ; \alpha ) - m ( x , p ; \alpha ) , v ( x , p ; \pi ) = \frac { K ( \frac { p - \pi ( x ) } { h } ) } { h \pi _ { 0 } ( p | x ) } / S _ { \ell } ^ { \pi } } \end{array}$ , and $\Delta _ { v } ^ { ( \ell ) } ( x , p ; \pi ) = \hat { v } ^ { ( \ell ) } - v ,$ , where $\hat { v } ^ { ( \ell ) }$ uses the estimated propensity score $\hat { \pi } _ { 0 } ^ { ( \ell ) } ( p | x )$ . We decompose $\hat { W } ^ { ( \ell ) } ( \pi , \alpha ) - \tilde { W } ^ { ( \ell ) } ( \pi , \alpha )$ into three components

$$
\hat { W } ^ { ( \ell ) } ( \pi , \alpha ) - \tilde { W } ^ { ( \ell ) } ( \pi , \alpha ) = \mathcal { E } _ { 1 } ^ { ( \ell ) } + \mathcal { E } _ { 2 } ^ { ( \ell ) } - \mathcal { E } _ { 3 } ^ { ( \ell ) } ,
$$

where

$$
\mathcal { E } _ { 1 } ^ { ( \ell ) } = \frac { 1 } { | \mathcal { Z } _ { \ell } | } \sum _ { t \in \mathcal { I } _ { \ell } } \left( \Delta _ { m } ^ { ( \ell ) } ( x _ { t } , \pi ( x _ { t } ) ; \alpha ) - \frac { 1 } { S _ { \ell } ^ { \pi } } \frac { K ( \frac { p _ { t } - \pi ( x _ { t } ) } { h } ) } { h \pi _ { 0 } ( p _ { t } | x _ { t } ) } \Delta _ { m } ^ { ( \ell ) } ( x _ { t } , p _ { t } ; \alpha ) \right) ,
$$

$$
\mathcal { E } _ { 2 } ^ { ( \ell ) } = \frac { 1 } { \left| \mathcal { T } _ { \ell } \right| } \sum _ { t \in \mathcal { T } _ { \ell } } \left( \Delta _ { v } ^ { ( \ell ) } ( x _ { t } , p _ { t } ; \pi ) \left( \exp ( - p _ { t } y _ { t } / \alpha ) - m ( x _ { t } , p _ { t } ; \alpha ) \right) \right) ,
$$

$$
\mathcal { E } _ { 3 } ^ { ( \ell ) } = \frac { 1 } { \left| \mathcal { T } _ { \ell } \right| } \sum _ { t \in \mathcal { T } _ { \ell } } \Big ( \Delta _ { v } ^ { ( \ell ) } ( x _ { t } , p _ { t } ; \pi ) \cdot \Delta _ { m } ^ { ( \ell ) } ( x _ { t } , p _ { t } ; \alpha ) \Big ) .
$$

Conditioning on dataset $\mathit { \Pi } _ { \mathcal { l } _ { \ell } ^ { c } } ^ { c } ,$ the nuisance estimates $\hat { m } ^ { ( \ell ) }$ and $\hat { \pi } _ { 0 } ^ { ( \ell ) }$ are fixed.

Step 1. Bias of $\begin{array} { r } { \operatorname* { s u p } _ { \pi \in \Pi , \alpha \in ( 0 , \overline { { \alpha } } ] } \left| \mathcal { E } _ { 1 } ^ { ( \ell ) } \right| , \forall \ell \in [ L ] } \end{array}$ . Let $\ell \in [ L ]$ be fixed for now. We decompose the error $\mathcal { E } _ { 1 } ^ { ( \ell ) }$ into two parts using the triangle inequality,

$$
\begin{array} { l } { \displaystyle \left. \mathcal { E } _ { 1 } ^ { ( t ) } \right. = \left. \frac { 1 } { \vert \mathbb { Z } _ { t } \vert } \sum _ { t \in \mathbb { Z } _ { \ell } } \left( \Delta _ { m } ^ { ( \ell ) } ( x _ { t } , \pi ( x _ { t } ) ; \alpha ) - \frac { 1 } { S _ { \ell } ^ { m } } \frac { K ( \frac { p _ { t } - \pi ( x _ { t } ) } { h } ) } { h \pi _ { 0 } ( p _ { t } \vert x _ { t } ) } \Delta _ { m } ^ { ( \ell ) } ( x _ { t } , p _ { t } ; \alpha ) \right) \right. } \\ { \displaystyle \qquad \leq \left. \frac { 1 } { \vert \mathbb { Z } _ { \ell } \vert } \sum _ { t \in \mathbb { Z } _ { \ell } } \left( \Delta _ { m } ^ { ( \ell ) } ( x _ { t } , \pi ( x _ { t } ) ; \alpha ) - \frac { K ( \frac { p _ { t } - \pi ( x _ { t } ) } { h } ) } { h \pi _ { 0 } ( p _ { t } \vert x _ { t } ) } \Delta _ { m } ^ { ( \ell ) } ( x _ { t } , p _ { t } ; \alpha ) \right) \right. + \left. \frac { S _ { \ell } ^ { \tau } } { S _ { \ell } ^ { \tau } \vert \mathbb { Z } _ { \ell } \vert } \sum _ { t \in \mathbb { Z } _ { \ell } } \frac { K ( \frac { p _ { t } - \pi ( x _ { t } ) } { h } ) } { h \pi _ { 0 } ( p _ { t } \vert x _ { t } ) } \Delta _ { m } ^ { ( \ell ) } ( x _ { t } , p _ { t } ; \alpha ) \right. } \end{array}
$$

For the second term, using the fact that m and $\hat { m } ^ { ( \ell ) }$ are bounded by 1, it is bounded by $| S _ { \ell } ^ { \pi } - 1 |$ . From the concentration results in Lemma C.4, we have

$$
\left| \frac { S _ { \ell } ^ { \pi } } { S _ { \ell } ^ { \pi } } - 1 \sum _ { t \in \mathbb { Z } _ { \ell } } \frac { K ( \frac { p _ { t } - \pi ( x _ { t } ) } { h } ) } { h \pi _ { 0 } ( p _ { t } | x _ { t } ) } \Delta _ { m } ^ { ( \ell ) } ( x _ { t } , p _ { t } ; \alpha ) \right| \le | S _ { \ell } ^ { \pi } - 1 | \lesssim \frac { d _ { \Pi } } { | { \cal Z } _ { \ell } | h } + \sqrt { \frac { d _ { \Pi } } { | { \cal Z } _ { \ell } | h } } + \sqrt { \frac { \log ( 1 / \epsilon ) } { | { \cal Z } _ { \ell } | h } } + \frac { \log ( 1 / \epsilon ) } { | { \cal Z } _ { \ell } | h } ,
$$

with probability at least $1 - \epsilon$

For the first term, we further split it into a zero-mean empirical process $\mathcal { E } _ { 1 , 1 } ^ { ( \ell ) }$ and a conditional bias $\mathcal { E } _ { 1 , 2 } ^ { ( \ell ) }$

$$
\phi _ { \pi , \alpha } ^ { ( \ell ) } ( x , p ) = \Delta _ { m } ^ { ( \ell ) } ( x , \pi ( x ) ; \alpha ) - \frac { K \left( \frac { p - \pi ( x ) } { h } \right) } { h \pi _ { 0 } ( p | x ) } \Delta _ { m } ^ { ( \ell ) } ( x , p ; \alpha ) ,
$$

$$
\mathcal { E } _ { 1 , 1 } ^ { ( \ell ) } = \frac { 1 } { \left| \mathcal { Z } _ { \ell } \right| } \sum _ { t \in \mathcal { Z } _ { \ell } } \left( \phi _ { \pi , \alpha } ^ { ( \ell ) } ( \pi , \alpha ) - \mathbb { E } \left[ \phi _ { \pi , \alpha } ^ { ( \ell ) } | \mathcal { Z } _ { \ell } ^ { c } \right] \right) , \quad \mathcal { E } _ { 1 , 2 } ^ { ( \ell ) } = \mathbb { E } \left[ \phi _ { \pi , \alpha } ^ { ( \ell ) } | \mathcal { Z } _ { \ell } ^ { c } \right] .
$$

For the conditional bias $\mathcal { E } _ { 1 , 2 } ^ { ( \ell ) }$ , we have

$$
\begin{array} { r } { \left| \mathcal { E } _ { 1 , 2 } ^ { ( \ell ) } \right| = \left| \mathbb { E } \left[ \Delta _ { m } ^ { ( \ell ) } ( X , \pi ( X ) ; \alpha ) \middle | \mathcal { T } _ { \ell } ^ { c } \right] - \mathbb { E } \left[ \mathbb { E } \left[ \frac { K \left( \frac { P - \pi ( X ) } { h } \right) } { h \pi _ { 0 } ( P | X ) } \Delta _ { m } ^ { ( \ell ) } ( X , P ; \alpha ) \middle | X , \mathcal { T } _ { \ell } ^ { c } \right] \right] \right| } \end{array}
$$

$$
\begin{array} { r l } & { = | \mathbb { E } [  \Delta _ { m } ^ { ( \ell ) } ( X , \pi ( X ) ; \alpha ) | \mathcal { L } _ { \ell } ^ { c } ] - \mathbb { E } [ \int _ { p _ { \operatorname* { m i n } } } ^ { 1 } \frac { 1 } { h } K ( \frac { p - \pi ( X ) } { h } ) \Delta _ { m } ^ { ( \ell ) } ( X , p ; \alpha ) \mathrm { d } p | \mathcal { L } _ { \ell } ^ { c } ] | } \\ & { = | \mathbb { E } [  \Delta _ { m } ^ { ( \ell ) } ( X , \pi ( X ) ; \alpha ) | \mathcal { L } _ { \ell } ^ { c } ] - \mathbb { E } [ \int \Delta _ { m } ^ { ( \ell ) } ( X , \pi ( X ) + t h ; \alpha ) K ( t ) \mathrm { d } t | \mathcal { L } _ { \ell } ^ { c } ] | } \\ & { = | \mathbb { E } [ \int  ( \Delta _ { m } ^ { ( \ell ) } ( X , \pi ( X ) ; \alpha ) - \Delta _ { m } ^ { ( \ell ) } ( X , \pi ( X ) + t h ; \alpha ) ) K ( t ) \mathrm { d } t ] | . } \end{array}
$$

Under s-order bounded derivative in Assumptions 2.4 and 4.2, Following from the s-order Taylor expansion of $\Delta _ { m } ^ { ( \ell ) }$ and Assumptions 3.2, 2.4,4.2, we obtain $\left| \mathcal { E } _ { 1 , 2 } ^ { ( \ell ) } \right| \lesssim h ^ { s }$ . To bound $\mathcal { E } _ { 1 , 1 } ^ { ( \ell ) }$ , we examine the Pseudo-dimension of the function class

$$
\begin{array} { r } { \Phi _ { 1 } : = \left\{ ( x , p ) \mapsto \phi _ { \pi , \alpha } ^ { ( \ell ) } - \mathbb { E } \left[ \phi _ { \pi , \alpha } ^ { ( \ell ) } \Big | \mathbb { Z } _ { \ell } ^ { c } \right] \Big | \pi \in \Pi , \alpha \in ( 0 , \overline { { \alpha } } ] \right\} . } \end{array}
$$

Let $\mathbf { M } : = \{ m _ { \pi , \alpha } : x \mapsto m ( x , \pi ( x ) ; \alpha ) | \pi \in \Pi , \alpha \in ( 0 , \overline { { \alpha } } ] \}$ . Note that for any fixed $( x , p ) , m ( x , p ; \alpha )$ is a monotonic function of α on $( 0 , \overline { { \alpha } } ]$ , then the function class $\mathcal { F } _ { \mathcal { A } } : = \{ f _ { \alpha } : ( x , p ) \mapsto m ( x , p ; \alpha ) \mid \alpha \in ( 0 , \overline { { \alpha } } ] \}$ has a Pseudo-dimension of 1 (Theorem 11.3 in Anthony & Bartlett, 2009). By Lemma D.3(ii), the covering number of a one-parameter monotonic family satisfies log $\mathcal { N } _ { 2 } ( \epsilon / 2 , \mathcal { F } _ { A } , | \mathcal { T } _ { \ell } | ) \lesssim \log ( 1 / \epsilon )$ . Crucially, because the monotonicity is uniform, we can select a grid of α values that forms an ϵ/2-cover for all x. We first claim that

$$
\mathcal { N } ( \epsilon , \mathbf { M } , L _ { 2 } ( P _ { | \mathcal { Z } _ { \ell } | } ) ) \leq \mathcal { N } _ { 2 } \left( \frac { \epsilon } { 2 L _ { m } } , \Pi , | \mathcal { Z } _ { \ell } | \right) \times \mathcal { N } _ { 2 } \left( \frac { \epsilon } { 2 } , \mathcal { F } _ { A } , | \mathcal { Z } _ { \ell } | \right) .
$$

For ease of notation, let

$$
N _ { 2 , \Pi } \left( \frac { \epsilon } { 2 L _ { m } } \right) : = \mathcal { N } _ { 2 } \left( \frac { \epsilon } { 2 L _ { m } } , \Pi , | \mathcal { Z } _ { \ell } | \right) , \quad N _ { 2 , A } \left( \frac { \epsilon } { 2 } \right) : = \mathcal { N } _ { 2 } \left( \frac { \epsilon } { 2 } , \mathcal { F } _ { A } , | \mathcal { Z } _ { \ell } | \right) .
$$

Suppose $\left\{ \pi _ { 1 } , . . . , \pi _ { N _ { 2 , \Pi } \left( \frac { \epsilon } { 2 L _ { m } } \right) } \right\}$ is a $\frac { \epsilon } { 2 L _ { m } }$ -cover for Π, and $f _ { \alpha _ { 1 } } , . . . , f _ { \alpha _ { N _ { 2 , A } ( \frac { \epsilon } { 2 } ) } }$ is a $\frac { \epsilon } { 2 }$ -cover for $\mathcal { F } _ { A }$ . Then, we can prove that M<sup>ϵ</sup> is a ϵ-cover set for M, where M<sup>ϵ</sup> is defined as

$$
\mathbf { M } ^ { \epsilon } = \left\{ m _ { \pi _ { i } , \alpha _ { j } } : x \mapsto m ( x , \pi _ { i } ( x ) ; \alpha _ { j } ) : i \leq N _ { 2 , \Pi } \left( \frac { \epsilon } { 2 L _ { m } } \right) , j \leq N _ { 2 , A } \left( \frac { \epsilon } { 2 } \right) \right\} .
$$

For $m ( x , \pi ( x ) ; \alpha ) \in \mathbf { M }$ , we can pick $m ( x , \pi ^ { \prime } ( x ) ; \alpha ^ { \prime } ) \in \mathbf { M } ^ { \epsilon }$ such that

$$
\| \pi - \pi ^ { \prime } \| _ { L _ { 2 } ( P _ { | \mathcal { Z } _ { \ell } | } ) } \leq \frac { \epsilon } { 2 L _ { m } } , \quad \| f _ { \alpha } - f _ { \alpha ^ { \prime } } \| _ { L _ { 2 } ( P _ { | \mathcal { Z } _ { \ell } | } ) } \leq \frac { \epsilon } { 2 } .
$$

Then, we have

$$
\begin{array} { r l } & { \displaystyle \| m _ { \pi , \alpha } - m _ { \pi ^ { \prime } , \alpha ^ { \prime } } \| _ { L _ { 2 } ( P | \tau _ { \ell } ) } = \sqrt { \frac { 1 } { | { \cal Z } _ { \ell } | } \displaystyle \sum _ { t \in { \cal Z } _ { \ell } } \left( m _ { \pi , \alpha } ( x _ { t } ) - m _ { \pi ^ { \prime } , \alpha ^ { \prime } } ( x _ { t } ) \right) ^ { 2 } } } \\ & { \le \sqrt { \frac { 1 } { | { \cal Z } _ { \ell } | } \displaystyle \sum _ { t \in { \cal Z } _ { \ell } } \left( m ( x _ { t } , \pi ( x _ { t } ) ; \alpha ) - m ( x _ { t } , \pi ^ { \prime } ( x _ { t } ) ; \alpha ) \right) ^ { 2 } + \left( m ( x _ { t } , \pi ^ { \prime } ( x _ { t } ) ; \alpha ) - m ( x _ { t } , \pi ^ { \prime } ( x _ { t } ) ; \alpha ^ { \prime } ) \right) ^ { 2 } } } \\ & { \le \sqrt { \displaystyle \frac { 1 } { | { \cal Z } _ { \ell } | } \displaystyle \sum _ { t \in { \cal Z } _ { \ell } } L _ { m } ^ { 2 } \left( \pi ( x _ { t } ) - \pi ^ { \prime } ( x _ { t } ) \right) ^ { 2 } + \left( f _ { \alpha } ( x _ { t } , \pi ^ { \prime } ( x _ { t } ) ) - f _ { \alpha ^ { \prime } } ( x _ { t } , \pi ^ { \prime } ( x _ { t } ) ) \right) ^ { 2 } } \le \epsilon . } \end{array}
$$

Thus, by Lemma D.3, we obtain

$$
\log \mathcal { N } \left( \epsilon , \mathbf { M } , L _ { 2 } ( P _ { | \mathcal { T } _ { \ell } | } ) \right) \lesssim d _ { \Pi } \log \left( \frac { 1 } { \epsilon } \right) .\tag{55}
$$

And let $\overline { { \mathbf { M } } } : = \{ \mathbb { E } [ m _ { \pi , \alpha } ( X ) ] | \pi \in \Pi , \alpha \in ( 0 , \overline { { \alpha } } ] \}$ . Then

$$
\begin{array} { r } { \left\| \mathbb { E } [ m _ { \pi , \alpha } ] - \mathbb { E } [ m _ { \pi ^ { \prime } , \alpha ^ { \prime } } ] \right\| _ { L _ { 2 } ( P _ { | \mathbb { Z } _ { \varepsilon | } } ) } = \left| \mathbb { E } [ m _ { \pi , \alpha } ] - \mathbb { E } [ m _ { \pi ^ { \prime } , \alpha ^ { \prime } } ] \right| \leq \mathbb { E } | m _ { \pi , \alpha } - m _ { \pi ^ { \prime } , \alpha ^ { \prime } } | } \end{array}
$$

implies that $\mathcal { N } ( \epsilon , \overline { { \mathbf { M } } } , L _ { 2 } ( P _ { | \mathcal { T } _ { \ell } | } ) ) \leq \mathcal { N } ( \epsilon , \mathbf { M } , L _ { 1 } ( \mathbf { P } _ { 0 } ) )$ ). Consequently, by Lemma D.3(iv), we have

$$
\log \mathcal { N } \left( \epsilon , \overline { { \mathbf { M } } } , L _ { 2 } ( P _ { | \mathcal { Z } _ { \ell } | } ) \right) \lesssim d _ { \Pi } \log \left( \frac { 1 } { \epsilon } \log \left( \frac { 1 } { \epsilon } \right) \right) .\tag{56}
$$

Conditional on $\mathcal { T } _ { \ell } ^ { c }$ , the estimated regression $\hat { m } ^ { ( \ell ) } ( x , p ; \alpha )$ is a fixed continuous function. Similarly, we obtain

$$
\log \mathcal { N } \left( \epsilon , \hat { \mathbf { M } } , L _ { 2 } ( P _ { | \mathcal { Z } _ { \epsilon } | } ) \right) \lesssim d _ { \Pi } \log \left( \frac { 1 } { \epsilon } \right) , \quad \log \mathcal { N } \left( \epsilon , \overline { { \widehat { \mathbf { M } } } } , L _ { 2 } ( P _ { | \mathcal { Z } _ { \epsilon } | } ) \right) \lesssim d _ { \Pi } \log \left( \frac { 1 } { \epsilon } \log \left( \frac { 1 } { \epsilon } \right) \right) .\tag{57}
$$

for the classes

$$
\hat { \mathbf { M } } : = \left\{ \hat { m } _ { \pi , \alpha } ^ { ( \ell ) } : x \mapsto \hat { m } ^ { ( \ell ) } ( x , \pi ( x ) ; \alpha ) \big | \pi \in \Pi , \alpha \in \left( 0 , \overline { { \alpha } } \right] \right\} \mathrm { a n d } \hat { \mathbf { M } } : = \left\{ \mathbb { E } \left[ \hat { m } _ { \pi , \alpha } ^ { ( \ell ) } ( X ) \right] \big | \pi \in \Pi , \alpha \in \left( 0 , \overline { { \alpha } } \right] \right\} .
$$

$$
\begin{array} { r } { \mathrm { L e t } K : = \left\{ k _ { \pi , \alpha } : ( x , p ) \mapsto \frac { K \left( \frac { p - \pi ( \alpha ) } { h } \right) } { h \pi _ { 0 } ( p | x ) } \Delta _ { m } ^ { ( \ell ) } ( x , p ; \alpha ) \big | \pi \in \mathrm { I I } , \alpha \in ( 0 , \overline { { \alpha } } ] \right\} \mathrm { . T h e n ~ w e ~ c a n ~ c l a i m ~ t h a t } } \end{array}
$$

$$
\mathcal { N } \left( \epsilon , K , L _ { 2 } ( P _ { | \mathcal { Z } _ { \ell } | } ) \right) \leq \mathcal { N } _ { 2 } \left( \epsilon / 2 , \mathcal { F } _ { \Pi } , | \mathcal { Z } _ { \ell } | \right) \times \mathcal { N } _ { 2 } \left( \frac { h \eta \epsilon } { 4 C _ { 2 } } , \mathcal { F } _ { A } , | \mathcal { Z } _ { \ell } | \right) \times \mathcal { N } _ { 2 } \left( \frac { h \eta \epsilon } { 4 C _ { 2 } } , \hat { \mathcal { F } } _ { A } , | \mathcal { Z } _ { \ell } | \right) ,
$$

where $\mathcal { F } _ { \Pi }$ is defined in (50) and $\hat { \mathcal { F } } _ { \mathcal { A } } ^ { ( \ell ) } : = \left\{ \hat { f } _ { \alpha } ^ { ( \ell ) } : ( x , p ) \mapsto \hat { m } ^ { ( \ell ) } ( x , p ; \alpha ) | \alpha \in ( 0 , \overline { { \alpha } } ] \right\}$ . It follows from

$$
\begin{array} { r l } & { \| k _ { \pi , \alpha } - k _ { \pi ^ { \prime } , \alpha ^ { \prime } } \| _ { L _ { 2 } ( P | z _ { \ell } | ) } } \\ & { \leq \sqrt { \displaystyle \frac { 1 } { | \mathcal { Z } _ { \ell } | } \sum _ { t \in \mathcal { Z } _ { \ell } } ( \frac { K ( \frac { p _ { t } - \pi ( x _ { t } ) } { h } ) } { h \pi _ { 0 } ( p _ { t } | x _ { t } ) } - \frac { K ( \frac { p _ { t } - \pi ^ { \prime } ( x _ { t } ) } { h } ) } { h \pi _ { 0 } ( p _ { t } | x _ { t } ) } ) ^ { 2 } } + \frac { C _ { 2 } } { h \eta } \sqrt { \frac { 1 } { | \mathcal { Z } _ { \ell } | } \sum _ { t \in \mathcal { Z } _ { \ell } } ( m ( x _ { t } , p _ { t } ; \alpha ) - m ( x _ { t } , p _ { t } ; \alpha ^ { \prime } ) ) ^ { 2 } } } \\ & { + \frac { C _ { 2 } } { h \eta } \sqrt { \frac { 1 } { | \mathcal { Z } _ { \ell } | } \sum _ { t \in \mathcal { Z } _ { \ell } } ( \hat { m } ^ { ( \ell ) } ( x _ { t } , p _ { t } ; \alpha ) - \hat { m } ^ { ( \ell ) } ( x _ { t } , p _ { t } ; \alpha ^ { \prime } ) ) ^ { 2 } } . } \end{array}
$$

Hence, we can bound the covering number as

$$
\log \mathcal { N } \left( \boldsymbol { \epsilon } , \boldsymbol { \mathcal { K } } , L _ { 2 } ( P _ { | \mathcal { Z } _ { \boldsymbol { \ell } } | } ) \right) \lesssim d _ { \Pi } \log \left( \frac { 1 } { h \epsilon } \right) .\tag{58}
$$

Also for $\overline { { \mathcal { K } } } : = \{ \mathbb { E } \left[ k _ { \pi , \alpha } \right] | \pi \in \Pi , \alpha \in ( 0 , \overline { { \alpha } } ] \}$ , we have

$$
\log \mathcal { N } \left( \epsilon , \overline { { \mathcal { K } } } , L _ { 2 } ( P _ { | \mathcal { Z } _ { \ell } | } ) \right) \lesssim d _ { \Pi } \log \left( \frac { 1 } { h \epsilon } \log \left( \frac { 1 } { h \epsilon } \right) \right) .\tag{59}
$$

Finally, we note that $\Phi _ { 1 }$ is a subset of the vector space spanned by these classes M, M<sup>ˆ</sup> , M, $\begin{array} { r } { \hat { \bf M } , \kappa , \overline { { \kappa } } . } \end{array}$ Combining (55)-(59), we obtain

$$
\log \mathcal { N } \left( \epsilon , \Phi _ { 1 } , L _ { 2 } ( P _ { | \mathcal { T } _ { \ell } | } ) \right) \lesssim d _ { \Pi } \log \left( \frac { 1 } { h \epsilon } \log \left( \frac { 1 } { h \epsilon } \right) \right) ,
$$

which is a similar upper bound like (64). We further have the bound on the functions in $\Phi _ { 1 }$

$$
\left| \phi _ { \pi , \alpha } ^ { ( \ell ) } ( x , p ) \right| \leq 1 + \frac { C _ { 2 } } { h \eta } , \quad \mathrm { a n d } \quad \mathbb { E } \left[ \left( \phi _ { \pi , \alpha } ^ { ( \ell ) } ( x , p ) \right) ^ { 2 } \right] \leq 1 + \frac { C _ { 2 } } { h \eta } .
$$

It implies a similar Dudley’s integral in Lemma C.6,

$$
\mathbb { E } \left[ \operatorname* { s u p } _ { \pi \in \Pi , \alpha \in ( 0 , \overline { { \alpha } } ] } \left. \mathcal { E } _ { 1 , 1 } ^ { ( \ell ) } \right. \right] \lesssim \log \left( \frac { 1 } { h } \right) \cdot \left( \frac { d _ { \Pi } } { \vert \mathcal { Z } _ { \ell } \vert h } + \sqrt { \frac { d _ { \Pi } } { \vert \mathcal { Z } _ { \ell } \vert h } } + \frac { d _ { \Pi } } { \vert \mathcal { Z } _ { \ell } \vert } + \sqrt { \frac { d _ { \Pi } } { \vert \mathcal { Z } _ { \ell } \vert } } \right) .
$$

Combining the above Rademacher complexity bound with Talagrand’s Inequality (Lemma D.1), we obtain with probability at least $1 - \epsilon$

$$
\operatorname* { s u p } _ { \pi \in \Pi , \alpha \in \left( 0 , \overline { { \alpha } } \right] } \Big | \mathcal { E } _ { 1 , 1 } ^ { ( \ell ) } \Big | \lesssim \mathbb { E } \left[ \operatorname* { s u p } _ { \pi \in \Pi } \Big | \mathcal { E } _ { 1 , 1 } ^ { ( \ell ) } \Big | \right] + \sqrt { \frac { \log ( 1 / \epsilon ) } { | \mathcal { Z } _ { \ell } | h } } + \frac { \log ( 1 / \epsilon ) } { | \mathcal { Z } _ { \ell } | h } + \sqrt { \frac { \log ( 1 / \epsilon ) } { | \mathcal { Z } _ { \ell } | } } + \frac { \log ( 1 / \epsilon ) } { | \mathcal { Z } _ { \ell } | } ,
$$

Hence, for $s = 1 , 2$ in Assumption 2.4, we have

$$
\operatorname* { s u p } _ { \pi \in \Pi , \alpha \in ( 0 , \pi ] } \left. \mathcal { E } _ { 1 } ^ { ( \delta ) } \right. \lesssim \log \left( \frac { 1 } { h } \right) \left( \frac { d _ { \Pi } } { | \mathcal { Z } _ { \ell } | | h } + \sqrt { \frac { d _ { \Pi } } { | \mathcal { Z } _ { \ell } | | h } } + \sqrt { \frac { \log ( 1 / \epsilon ) } { | \mathcal { Z } _ { \ell } | | h } } + \frac { \log ( 1 / \epsilon ) } { | \mathcal { Z } _ { \ell } | | h } + \frac { d _ { \Pi } } { | \mathcal { Z } _ { \ell } | } + \sqrt { \frac { d _ { \Pi } } { | \mathcal { Z } _ { \ell } | } } + \sqrt { \frac { \log ( 1 / \epsilon ) } { | \mathcal { Z } _ { \ell } | } } + \frac { \log ( 1 / \epsilon ) } { | \mathcal { Z } _ { \ell } | } \right) + h ^ { s } .
$$

Step 2. Bound for $\begin{array} { r } { \operatorname* { s u p } _ { \pi \in \Pi , \alpha \in ( 0 , \overline { { \alpha } } ] } \Big | \mathscr { E } _ { 2 } ^ { ( \ell ) } \Big | , \forall \ell \in [ L ] } \end{array}$ . Let $\ell \in [ L ]$ be fixed for now. For any $\pi \in \Pi$ and $\alpha \in ( 0 , \overline { { \alpha } } ]$ , we define

$$
\begin{array} { r } { \psi _ { \pi , \alpha } ^ { ( \ell ) } ( x , p , y ) = \Delta _ { v } ^ { ( \ell ) } ( x , p ; \pi ) \cdot ( \exp ( - p y / \alpha ) - m ( x , p ; \alpha ) ) , } \\ { \Delta _ { v } ^ { ( \ell ) } ( x , p ; \pi ) = \cfrac { 1 } { h } K \left( \cfrac { p - \pi ( x ) } { h } \right) \left( \cfrac { 1 } { \hat { S } _ { \ell } ^ { \pi } \hat { \pi } _ { 0 } ^ { ( \ell ) } ( p | x ) } - \cfrac { 1 } { S _ { \ell } ^ { \pi } \pi _ { 0 } ( p | x ) } \right) , } \end{array}
$$

and the nuisance estimator $\hat { \pi } _ { 0 } ^ { ( \ell ) }$ is fixed conditioned on the auxiliary dataset $\mathcal { T } _ { \ell } ^ { c }$ , We observe that each summand of $\mathcal { E } _ { 2 } ^ { ( \ell ) }$ is zero-mean, by the definition of m. Now we analyze the Pseudo-dimension of the function class

$$
\Psi _ { 2 } : = \left\{ ( x , p , y ) \mapsto \psi _ { \pi , \alpha } ^ { ( \ell ) } ( x , p , y ) | \pi \in \Pi , \alpha \in ( 0 , \overline { { \alpha } } ] \right\} .
$$

We decompose $\psi _ { \pi , \alpha } ^ { ( \ell ) }$ into three components involving π and α,

$$
\begin{array} { l } { \displaystyle \left\| \psi _ { \pi , \alpha } ^ { ( \ell ) } - \psi _ { \pi ^ { \prime } , \alpha ^ { \prime } } ^ { ( \ell ) } \right\| _ { L _ { 2 } ( P _ { \ell } \geq t ) } = \sqrt { \frac { 1 } { | \mathcal { Z } _ { \ell } | } \displaystyle \sum _ { t \in \mathcal { Z } _ { \ell } } \left( \psi _ { \pi , \alpha } ^ { ( \ell ) } ( x _ { t } , p _ { t } , y _ { t } ) - \psi _ { \pi ^ { \prime } , \alpha ^ { \prime } } ^ { ( \ell ) } ( x _ { t } , p _ { t } , y _ { t } ) \right) ^ { 2 } } } \\ { \leq \sqrt { \frac { 1 } { | \mathcal { Z } _ { \ell } | } \displaystyle \sum _ { t \in \mathcal { Z } _ { \ell } } \left( \Delta _ { v } ^ { ( \ell ) } ( x _ { t } , p _ { t } ; \pi ) - \Delta _ { v } ^ { ( \ell ) } ( x _ { t } , p _ { t } ; \pi ^ { \prime } ) \right) ^ { 2 } ( \exp ( - p y / \alpha ) - m ( x , p ; \alpha ) ) ^ { 2 } } } \\ { + \sqrt { \frac { 1 } { | \mathcal { Z } _ { \ell } | } \displaystyle \sum _ { t \in \mathcal { Z } _ { \ell } } \left( \Delta _ { v } ^ { ( \ell ) } ( x _ { t } , p _ { t } ; \pi ^ { \prime } ) \right) ^ { 2 } ( \exp ( - p _ { t } y _ { t } / \alpha ) - \exp ( - p _ { t } y _ { t } / \alpha ^ { \prime } ) ) ^ { 2 } } } \\ { + \sqrt { \frac { 1 } { | \mathcal { Z } _ { \ell } | } \displaystyle \sum _ { t \in \mathcal { Z } _ { \ell } } \left( \Delta _ { v } ^ { ( \ell ) } ( x _ { t } , p _ { t } ; \pi ^ { \prime } ) \right) ^ { 2 } ( m ( x _ { t } , p _ { t } ; \alpha ) - m ( x _ { t } , p _ { t } ; \alpha ^ { \prime } ) ) ^ { 2 } } . } \end{array}
$$

For the first term, we use the fact that $( a + b ) ^ { 2 } \leq 2 ( a ^ { 2 } + b ^ { 2 } )$ and $\begin{array} { r } { k _ { \pi } ( x , p ) : = \frac { 1 } { h } K \left( \frac { p - \pi ( x ) } { h } \right) } \end{array}$

$$
\begin{array} { r l } & { \sqrt { \displaystyle \frac { 1 } { | \mathcal { Z } _ { \ell } | } \sum _ { t \in \mathcal { Z } _ { \ell } } \left( \Delta _ { v } ^ { ( \ell ) } ( x _ { t } , p _ { t } ; \pi ) - \Delta _ { v } ^ { ( \ell ) } ( x _ { t } , p _ { t } ; \pi ^ { \prime } ) \right) ^ { 2 } ( \exp ( - p _ { t } y _ { t } / \alpha ) - m ( x _ { t } , p _ { t } ; \alpha ) ) ^ { 2 } } } \\ & { \le \sqrt { \displaystyle \frac { 2 } { | \mathcal { Z } _ { \ell } | } \sum _ { t \in \mathcal { Z } _ { \ell } } \left( \frac { k _ { \pi } ( x _ { t } , p _ { t } ) } { S _ { \ell } ^ { \pi } \pi _ { 0 } ( p _ { t } | x _ { t } ) } - \frac { k _ { \pi ^ { \prime } } ( x _ { t } , p _ { t } ) } { S _ { \ell } ^ { \pi ^ { \prime } } \pi _ { 0 } ( p _ { t } | x _ { t } ) } \right) ^ { 2 } } + \sqrt { \displaystyle \frac { 2 } { | \mathcal { Z } _ { \ell } | } \sum _ { t \in \mathcal { Z } _ { \ell } } \left( \frac { k _ { \pi } ( x _ { t } , p _ { t } ) } { \hat { S } _ { \ell } ^ { \pi } \hat { \pi } _ { 0 } ^ { ( \ell ) } ( p _ { t } | x _ { t } ) } - \frac { k _ { \pi ^ { \prime } } ( x _ { t } , p _ { t } ) } { \hat { S } _ { \ell } ^ { \pi ^ { \prime } } \hat { \pi } _ { 0 } ^ { ( \ell ) } ( p _ { t } | x _ { t } ) } \right) ^ { 2 } } } \end{array}
$$

By Step 1 in Lemma $\mathrm { C . 4 } ,$ , we can pick $\begin{array} { r } { h \ge \sqrt { \frac { d _ { \Pi } + \log ( 1 / \epsilon ) } { 2 \left| \mathcal { T } _ { \ell } \right| } } } \end{array}$ such that $\begin{array} { r } { \operatorname* { s u p } _ { \pi \in \Pi } | S _ { \ell } ^ { \pi } | \le \frac { 3 } { 2 } } \end{array}$ and $\mathrm { i n f } _ { \pi \in \Pi } | S _ { \ell } ^ { \pi } | \ge 1 / 2$ with probability at least $1 - \epsilon$ . Then, for any $\pi , \pi ^ { \prime } \in \dot { \Pi }$ , we have

$$
\sqrt { \frac { 1 } { \left| \mathcal { Z } _ { \ell } \right| } \sum _ { t \in \mathcal { Z } _ { \ell } } \left( \frac { k _ { \pi } ( x _ { t } , p _ { t } ) } { \mathcal { S } _ { \ell } ^ { \pi } \pi _ { 0 } ( p _ { t } | x _ { t } ) } - \frac { k _ { \pi ^ { \prime } } ( x _ { t } , p _ { t } ) } { \mathcal { S } _ { \ell } ^ { \pi ^ { \prime } } \pi _ { 0 } ( p _ { t } | x _ { t } ) } \right) ^ { 2 } } = \sqrt { \frac { 1 } { \left| \mathcal { Z } _ { \ell } \right| } \sum _ { t \in \mathcal { Z } _ { \ell } } \left( \frac { k _ { \pi } ( x _ { t } , p _ { t } ) S _ { \ell } ^ { \pi ^ { \prime } } - k _ { \pi ^ { \prime } } ( x _ { t } , p _ { t } ) S _ { \ell } ^ { \pi } } { S _ { \ell } ^ { \pi } S _ { \ell } ^ { \pi ^ { \prime } } \pi _ { 0 } ( p _ { t } | x _ { t } ) } \right) ^ { 2 } }
$$

$$
\begin{array} { r l } & { = \sqrt { \displaystyle \frac { 1 } { | \mathcal { T } _ { \ell } | } \sum _ { t \in \mathcal { T } _ { \ell } } \left( \frac { ( k _ { \pi } ( x _ { t } , p _ { t } ) - k _ { \pi ^ { \prime } } ( x _ { t } , p _ { t } ) ) S _ { \ell } ^ { \pi ^ { \prime } } + k _ { \pi ^ { \prime } } ( x _ { t } , p _ { t } ) ( S _ { \ell } ^ { \pi ^ { \prime } } - S _ { \ell } ^ { \pi } ) } { S _ { \ell } ^ { \pi } S _ { \ell } ^ { \pi ^ { \prime } } \pi _ { 0 } ( p _ { t } | x _ { t } ) } \right) ^ { 2 } } } \\ & { \le \displaystyle \frac { 2 } { \eta } \| k _ { \pi } - k _ { \pi ^ { \prime } } \| _ { L _ { 2 } ( P | \mathcal { Z } _ { \ell } | ) } + 2 | S _ { \ell } ^ { \pi } - S _ { \ell } ^ { \pi ^ { \prime } } | \le \displaystyle \frac { 4 } { \eta } \| k _ { \pi } - k _ { \pi ^ { \prime } } \| _ { L _ { 2 } ( P | \mathcal { Z } _ { \ell } | ) } , } \end{array}
$$

where we use $\begin{array} { r } { \sum _ { t } k _ { \pi ^ { \prime } } ^ { 2 } / ( S _ { \ell } ^ { \pi ^ { \prime } } \pi _ { 0 } ) ^ { 2 } \le 1 } \end{array}$ , and

$$
| S _ { \ell } ^ { \pi } - S _ { \ell } ^ { \pi ^ { \prime } } | = \left| \frac { 1 } { | \mathcal { T } _ { \ell } | } \sum _ { \ell \in \mathcal { T } _ { \ell } } \frac { 1 } { \pi _ { 0 } ( p _ { \ell } | x _ { \ell } ) } \big ( k _ { \pi } ( x _ { \ell } , p _ { \ell } ) - k _ { \pi ^ { \prime } } ( x _ { \ell } , p _ { \ell } ) \big ) \right| \leq \frac { 1 } { \eta } \sqrt { \frac { 1 } { | \mathcal { T } _ { \ell } | } \sum _ { \ell \in \mathcal { T } _ { \ell } } \big ( k _ { \pi } ( x _ { \ell } , p _ { \ell } ) - k _ { \pi ^ { \prime } } ( x _ { \ell } , p _ { \ell } ) \big ) ^ { 2 } } = \frac { 1 } { \eta } \| k _ { \pi } - k _ { \pi ^ { \prime } } \| _ { L _ { 2 } ( P | _ { \mathcal { T } _ { \ell } \ell } ) } .
$$

Furthermore, we note that

$$
\begin{array} { r l } { \underset { x \in \Pi } { \operatorname* { s u p } } | S _ { \varepsilon } ^ { \pi } - \hat { S } _ { \varepsilon } ^ { \pi } | = \underset { x \in \Pi } { \operatorname* { s u p } } \left| \frac { 1 } { | \mathcal { Z } _ { \varepsilon } | } \sum _ { t \in \mathcal { Z } _ { \varepsilon } } \frac { \hat { \pi } _ { 0 } ^ { ( \ell ) } ( p _ { t } | x _ { t } ) - \pi _ { 0 } ( p _ { t } | x _ { t } ) } { \hat { \pi } _ { 0 } ^ { ( \ell ) } ( p _ { t } | x _ { t } ) \pi _ { 0 } ( p _ { t } | x _ { t } ) } k _ { \pi } ( x _ { t } , p _ { t } ) \right| } & { \leq \frac { 1 } { \eta } \| \hat { \pi } _ { 0 } ^ { ( \ell ) } - \pi _ { 0 } \| _ { \infty } \cdot \underset { x \in \Pi } { \operatorname* { s u p } } \left( \frac { 1 } { | \mathcal { Z } _ { \varepsilon } | } \sum _ { t \in \mathcal { Z } _ { \varepsilon } } \frac { k _ { \pi } ( x _ { t } , p _ { t } ) } { \pi _ { 0 } ( p _ { t } | x _ { t } ) } \right) } \\ { = \frac { 1 } { \eta } \| \hat { \pi } _ { 0 } ^ { ( \ell ) } - \pi _ { 0 } \| _ { \infty } \cdot \underset { x \in \Pi } { \operatorname* { s u p } } S _ { \varepsilon } ^ { \pi } \leq \frac { 3 } { 2 \eta } \| \hat { \pi } _ { 0 } ^ { ( \ell ) } - \pi _ { 0 } \| _ { \infty } , ~ } & { ( 6 0 ) } \end{array}
$$

by picking $\begin{array} { r } { h \ge \sqrt { \frac { d _ { \Pi } + \log ( 1 / \epsilon ) } { 2 | \mathcal { T } _ { \ell } | } } } \end{array}$ , we have with probability $1 - \epsilon$

$$
\operatorname* { s u p } _ { \pi \in \Pi } \Big \lvert \mathbb { E } \left[ S _ { \ell } ^ { \pi } \right] - \mathbb { E } \left[ \hat { S } _ { \ell } ^ { \pi } \right] \Big \rvert \leq \mathbb { E } \left[ \operatorname* { s u p } _ { \pi \in \Pi } \Big \lvert S _ { \ell } ^ { \pi } - \hat { S } _ { \ell } ^ { \pi } \Big \rvert \right] \leq \frac { 3 } { 2 \eta } \cdot \bigg ( \mathbb { E } \left[ \Big \lVert \hat { \pi } _ { 0 } ^ { ( \ell ) } ( p | x ) - \pi _ { 0 } ( p | x ) \Big \rVert _ { \infty } ^ { 2 } \right] \bigg ) ^ { 1 / 2 } \lesssim \frac { 3 T ^ { - \gamma _ { \pi } } } { 2 \eta } .
$$

Then, there exists a sufficiently large T such that $\frac { 3 T ^ { - \gamma _ { \pi } } } { 2 \eta } \leq 1 / 4$ such that $\begin{array} { r } { \operatorname* { i n f } _ { \pi \in \Pi } \mathbb { E } \left[ \hat { S } _ { \ell } ^ { \pi } \right] \geq \operatorname* { i n f } _ { \pi \in \Pi } \mathbb { E } \left[ S _ { \ell } ^ { \pi } \right] - 1 / 4 \geq 1 / 4 } \end{array}$ with probability at least $1 - \epsilon .$ . Furthermore, we define

$$
\hat { f } _ { \pi } ^ { ( \ell ) } ( x , p ) = \frac { 1 } { h \hat { \pi } _ { 0 } ^ { ( \ell ) } ( p | x ) } K \left( \frac { p - \pi ( x ) } { h } \right) , \hat { \mathcal { F } } _ { \Pi } ^ { ( \ell ) } : = \{ f _ { \pi } ^ { ( \ell ) } | \pi \in \Pi \} , \hat { \mathcal { G } } _ { \Pi } ^ { ( \ell ) } : = \{ \hat { g } _ { \pi } ^ { ( \ell ) } = \hat { f } _ { \pi } ^ { ( \ell ) } - \mathbb { E } [ \hat { f } _ { \pi } ^ { ( \ell ) } ] | \pi \in \Pi \} .
$$

Using Lemma C.6, we have

$$
\mathbb { E } \left[ \operatorname* { s u p } _ { \pi \in \Pi } \left| \frac { 1 } { | { \mathcal { T } } _ { \ell } | } \sum _ { t \in { \mathcal { T } } _ { \ell } } \hat { f } _ { \pi } ^ { ( \ell ) } ( x _ { t } , p _ { t } ) - \mathbb { E } [ \hat { f } _ { \pi } ^ { ( \ell ) } ( x , p ) ] \right| \right] \lesssim \frac { d _ { \Pi } } { | { \mathcal { T } } _ { \ell } | h } + \sqrt { \frac { d _ { \Pi } } { | { \mathcal { T } } _ { \ell } | h } } .
$$

By a similar argument in Step 1 of the proof of Lemma $\mathrm { C . 4 } .$ , we have

$$
\operatorname* { s u p } _ { \pi \in \Pi } | \hat { f } _ { \pi } ^ { ( \ell ) } ( x , p ) | \leq \frac { C _ { 2 } } { h \eta } , \mathrm { ~ a n d ~ } \operatorname* { s u p } _ { \pi \in \Pi } \mathbb { E } \left[ \left( \hat { f } _ { \pi } ^ { ( \ell ) } ( x , p ) \right) ^ { 2 } \right] \leq \frac { C _ { 2 } } { h \eta ^ { 2 } } .\tag{61}
$$

Recalling that $\hat { \pi } _ { 0 } ^ { ( \ell ) }$ is trained on $T - | \mathcal { T } _ { \ell } |$ data points which is fixed, and $\{ \hat { f } _ { \pi } ^ { ( \ell ) } ( x _ { t } , p _ { t } ) \} _ { t \in \mathbb { Z } _ { t } }$ is i.i.d., we can get

$$
\operatorname* { s u p } _ { \pi \in \Pi } \left. \frac { 1 } { | \mathcal { Z } _ { \ell } | } \sum _ { t \in \mathcal { T } _ { \ell } } \hat { f } _ { \pi } ^ { ( \ell ) } ( x _ { t } , p _ { t } ) - \mathbb { E } \big [ \hat { f } _ { \pi } ^ { ( \ell ) } ( x , p ) \big ] \right. \lesssim \frac { d _ { \Pi } } { | \mathcal { Z } _ { \ell } | h } + \sqrt { \frac { d _ { \Pi } } { | \mathcal { Z } _ { \ell } | h } } + \sqrt { \frac { \log ( 1 / \epsilon ) } { | \mathcal { Z } _ { \ell } | h } } + \frac { \log ( 1 / \epsilon ) } { | \mathcal { Z } _ { \ell } | h } .
$$

To proceed, picking $\begin{array} { r } { h \ge \sqrt { \frac { d _ { \Pi } + \log ( 1 / \epsilon ) } { 8 | \mathcal { T } _ { \ell } | } } } \end{array}$ , we conclude inf $\begin{array} { r } { { \bf \tilde { \Sigma } } _ { \pi \in \Pi } \left| \hat { S } _ { \ell } ^ { \pi } \right| \geq \operatorname* { i n f } _ { \pi \in \Pi } \mathbb { E } [ \hat { S } _ { \ell } ^ { \pi } ] - 1 / 8 \geq 1 / 8 } \end{array}$ with probability at least $1 - 2 \epsilon$ . Thus, we reduce the second term to

$$
\sqrt { \frac { 1 } { | \mathcal { Z } _ { \ell } | } \sum _ { t \in \mathcal { Z } _ { \ell } } \left( \frac { k _ { \pi } ( x _ { t } , p _ { \ell } ) } { \tilde { S } _ { \ell } ^ { \pi } \tilde { \pi } _ { 0 } ^ { ( \ell ) } ( p _ { \ell } | x _ { t } ) } - \frac { k _ { \pi ^ { \prime } } ( x _ { t } , p _ { t } ) } { \tilde { S } _ { \ell } ^ { \pi ^ { \prime } } \tilde { \pi } _ { 0 } ^ { ( \ell ) } ( p _ { t } | x _ { t } ) } \right) ^ { 2 } } \le \frac { 8 } { \eta } \| k _ { \pi } - k _ { \pi ^ { \prime } } \| _ { L _ { 2 } ( P | \tau _ { \varepsilon } | ) } + 8 | \hat { S } _ { \ell } ^ { \pi } - \hat { S } _ { \ell } ^ { \pi ^ { \prime } } | \le \frac { 1 6 } { \eta } \| k _ { \pi } - k _ { \pi ^ { \prime } } \| _ { L _ { 2 } ( P | \tau _ { \varepsilon } | ) } .
$$

For the second and third terms, using Assumption 3.2, there exists a constant $C > 0$ such that $| K ( u ) | \le C$ , above results $\begin{array} { r } { \operatorname* { i n f } _ { \pi \in \Pi } | \hat { S } _ { \ell } ^ { \pi } | \ge \frac { 1 } { 8 } } \end{array}$ and $\begin{array} { r } { \operatorname* { i n f } _ { \pi \in \Pi } | S _ { \ell } ^ { \pi } | \geq \frac { 1 } { 2 } } \end{array}$ satisfied with probability at least $1 - 2 \epsilon$ , and the assumption that $\scriptstyle { \hat { \pi } } _ { 0 }$ and $\pi _ { 0 }$ are lower bounded by η,

$$
\operatorname* { s u p } _ { \pi \in \Pi } \big | \Delta _ { v } ^ { ( \ell ) } ( x _ { t } , p _ { t } ; \pi ) \big | \le \frac { 1 6 C } { h \eta } \operatorname* { s u p } _ { \pi \in \Pi } \big | S _ { \ell } ^ { \pi } - \hat { S } _ { \ell } ^ { \pi } \big | + \frac { 2 C } { h \eta ^ { 2 } } \| \pi _ { 0 } - \hat { \pi } _ { 0 } \big | \big | _ { \infty } \overset { \mathrm { { b y } } ( 6 0 ) } { \le } \frac { 2 6 C } { h \eta ^ { 2 } } \| \hat { \pi } _ { 0 } ^ { ( \ell ) } - \pi _ { 0 } \| _ { \infty } \le \frac { 2 6 C } { h \eta ^ { 2 } } .
$$

Thus, we reduce the second and third terms to

$$
\begin{array} { r l } & { \sqrt { \displaystyle \frac { 1 } { | \mathcal { Z } _ { \ell } | } \sum _ { t \in \mathcal { Z } _ { \ell } } \Big ( \Delta _ { v } ^ { ( \ell ) } ( x _ { t } , p _ { t } ; \pi ^ { \prime } ) \Big ) ^ { 2 } \big ( \exp \big ( - p _ { t } \mathcal { H } / \alpha \big ) - \exp \big ( - p _ { t } \mathcal { H } / \alpha ^ { \prime } \big ) \big ) ^ { 2 } } \leq \displaystyle \frac { 2 6 C } { h \eta ^ { 2 } } \sqrt { \displaystyle \frac { 1 } { | \mathcal { Z } _ { \ell } | } \sum _ { t \in \mathcal { Z } _ { \ell } } \big ( \exp \big ( - p _ { t } \mathcal { H } / \alpha \big ) - \exp \big ( - p _ { t } \mathcal { H } / \alpha ^ { \prime } \big ) \big ) ^ { 2 } } , } \\ & { \sqrt { \displaystyle \frac { 1 } { | \mathcal { Z } _ { \ell } | } \sum _ { t \in \mathcal { Z } _ { \ell } } \Big ( \Delta _ { v } ^ { ( \ell ) } \big ( x _ { t } , p _ { t } ; \pi ^ { \prime } \big ) \Big ) ^ { 2 } \big ( m ( x _ { t } , p _ { t } ; \alpha ) - m \big ( x _ { t } , p _ { t } ; \alpha ^ { \prime } \big ) \big ) ^ { 2 } } \leq \displaystyle \frac { 2 6 C } { h \eta ^ { 2 } } \sqrt { \displaystyle \frac { 1 } { | \mathcal { Z } _ { \ell } | } \sum _ { t \in \mathcal { Z } _ { \ell } } \big ( m ( x _ { t } , p _ { t } ; \alpha ) - m \big ( x _ { t } , p _ { t } ; \alpha ^ { \prime } \big ) \big ) ^ { 2 } } . } \end{array}
$$

Combined with the finite pseudo-dimension of $\begin{array} { r } { \mathcal { K } _ { \Pi } = \left\{ f _ { \pi } : ( x , p ) \mapsto \frac { 1 } { h } K \left( \frac { p - \pi ( x ) } { h } \right) | \pi \in \Pi \right\} } \end{array}$ and the covering number of a one-parameter monotonic family, this inequality implies that

$$
\log \mathcal { N } \left( \epsilon , \Psi _ { 2 } , L _ { 2 } ( P _ { | \mathcal { Z } _ { \ell } | } ) \right) \lesssim d _ { \Pi } \log \left( \frac { 1 } { h \epsilon } \right) .
$$

By the similar argument in (61), we get with probability at least $1 - 3 \epsilon$

$$
\begin{array} { r l } & { \quad \underset { \pi \in \mathbb { T } , \alpha \in ( 0 , \overline { { \alpha } } ] } { \operatorname* { s u p } } \left| \mathcal { E } _ { 2 } ^ { ( \ell ) } \right| \lesssim \log \big ( \frac { 1 } { \overline { { h } } } \big ) \left( \frac { d _ { \Pi } } { | \mathcal { Z } _ { \ell } | h } + \sqrt { \frac { d _ { \Pi } } { | \mathcal { Z } _ { \ell } | h } } + \sqrt { \frac { \log ( 1 / \epsilon ) } { | \mathcal { Z } _ { \ell } | h } } + \frac { \log ( 1 / \epsilon ) } { | \mathcal { Z } _ { \ell } | h } + \frac { d _ { \Pi } } { | \mathcal { Z } _ { \ell } | } + \sqrt { \frac { d _ { \Pi } } { | \mathcal { Z } _ { \ell } | } } + \sqrt { \frac { \log ( 1 / \epsilon ) } { | \mathcal { Z } _ { \ell } | } } + \frac { \log ( 1 / \epsilon ) } { | \mathcal { Z } _ { \ell } | } \right) } \\ & { \quad \lesssim \log \big ( \frac { 1 } { \overline { { h } } } \big ) \left( \frac { d _ { \Pi } } { | \mathcal { Z } _ { \ell } | h } + \sqrt { \frac { d _ { \Pi } } { | \mathcal { Z } _ { \ell } | h } } + \sqrt { \frac { \log ( 1 / \epsilon ) } { | \mathcal { Z } _ { \ell } | h } } + \frac { \log ( 1 / \epsilon ) } { | \mathcal { Z } _ { \ell } | h } \right) . } \end{array}
$$

Step 3. Bound for $\begin{array} { r } { \operatorname* { s u p } _ { \pi \in \Pi , \alpha \in ( 0 , \overline { { \alpha } } ] } \Big | { \mathcal E } _ { 3 } ^ { ( \ell ) } \Big | , \forall \ell \in [ L ] } \end{array}$ . Let $\ell \in \ [ L ]$ be fixed for now. By previous step 2, picking $\begin{array} { r } { h \ge \sqrt { \frac { d _ { \Pi } + \log ( 1 / \epsilon ) } { 2 | \mathcal { T } _ { \ell } | } } } \end{array}$ we have $\begin{array} { r } { \operatorname* { s u p } _ { \pi \in \Pi } | S _ { \ell } ^ { \pi } | \le \frac { 3 } { 2 } } \end{array}$ and $\operatorname* { i n f } _ { \pi \in \Pi } | S _ { \ell } ^ { \pi } | \geq 1 / 2$ with probability at least $1 - \epsilon .$ Then, utilizing the normalization of $S _ { \ell } ^ { \pi }$ and $\hat { S } _ { \ell } ^ { \pi }$ , we have

$$
\begin{array} { l } { \displaystyle \frac { 1 } { | { \cal Z } _ { \ell } | } \displaystyle \sum _ { { t \in { \cal Z } _ { \ell } } } \displaystyle \frac { K ( \displaystyle \frac { p _ { \ell } - \pi ( { \boldsymbol { \chi } } _ { \ell } ) } { \hbar } ) } { \hbar } | \displaystyle \frac { 1 } { S _ { \ell } ^ { \pi } \pi _ { 0 } ( p _ { \ell } | x ) } - \displaystyle \frac { 1 } { S _ { \ell } ^ { \pi } \hat { \pi } _ { 0 } ^ { ( \ell ) } ( p _ { \ell } | x ) } | = \displaystyle \frac { 1 } { | { \cal Z } _ { \ell } | } \displaystyle \sum _ { { t \in { \cal Z } _ { \ell } } } \displaystyle \frac { K ( \displaystyle \frac { p _ { \ell } - \pi ( { \boldsymbol { \chi } } _ { \ell } ) } { \hbar } ) } { \hbar } | \displaystyle \frac { \hat { S } _ { \ell } ^ { \pi } ( \hat { \pi } _ { 0 } ^ { ( \ell ) } - \pi _ { 0 } ) + \pi _ { 0 } ( \hat { S } _ { \ell } ^ { \pi } - S _ { \ell } ^ { \pi } ) } { S _ { \ell } ^ { \pi } \hat { S } _ { \ell } ^ { \pi } \pi _ { 0 } \hat { \pi } _ { 0 } ^ { ( \hat { \boldsymbol { \chi } } ) } } | } { S _ { \ell } ^ { \pi } \hat { S } _ { \ell } ^ { \pi } \pi _ { 0 } ^ { ( \hat { \boldsymbol { \chi } } ) } } |  \\  \displaystyle \leq \displaystyle \frac { 1 } { \eta } \displaystyle \frac { 1 } { | { \cal Z } _ { \ell } | } \displaystyle \sum _ { { t \in { \cal Z } _ { \ell } } } \displaystyle \frac { K ( \displaystyle \frac { p _ { \ell } - \pi ( { \boldsymbol { \chi } } _ { \ell } ) } { \hbar } ) } { \hbar \pi _ { 0 } ( p _ { \ell } | x _ { \ell } ) } \displaystyle \frac { 1 } { S _ { \ell } ^ { \pi } } \| \hat { \pi } _ { 0 } ^ { ( \ell ) } - \pi _ { 0 } \end{array}
$$

Furthermore, we reduce the expectation to

$$
\begin{array} { r l } & { \mathbb { \mathbb { E } } [ \underset { \pi \in \Pi , \alpha \in ( 0 , \overline { { \alpha } } ] } { \operatorname* { s u p } } \Big | \mathcal { E } _ { 3 } ^ { ( \ell ) } \Big | ] } \\ & { = \mathbb { E } [ \underset { \pi \in \Pi , \alpha \in ( 0 , \overline { { \alpha } } ] } { \operatorname* { s u p } } | \frac { 1 } { \big | \overline { { Z } } _ { \ell } \big | } \sum _ { t \in \overline { { Z } } _ { \ell } } \frac { K \Big ( \frac { p _ { \ell } - \pi ( x _ { * } ) } { \hbar } \Big ) } { \hbar } ( \frac { 1 } { S _ { \ell } ^ { \pi } \pi _ { 0 } ( p _ { t } | x _ { t } ) } - \frac { 1 } { S _ { \ell } ^ { \pi } \widehat { \pi } _ { 0 } ^ { ( \ell ) } ( p _ { t } | x _ { t } ) } ) ( \widehat { m } ^ { ( \ell ) } ( x _ { t } , p _ { t } ; \alpha ) - m ( x _ { t } , p _ { t } ; \alpha ) ) | ] } \\ & { \leq E [ ( \frac { 1 } { \eta } \| \widehat { \pi } _ { 0 } ^ { ( \ell ) } - \pi _ { 0 } \| _ { \infty } + 2 \underset { \pi \in \Pi } { \operatorname* { s u p } } | S _ { \ell } ^ { \pi } - \widehat { S } _ { \ell } ^ { \pi } | ) \cdot \underset { \alpha \in ( 0 , \overline { { \alpha } } ] } { \operatorname* { s u p } } \Big | \big | \widehat { m } ^ { ( \ell ) } - m \Big \| _ { \infty } ] } \end{array}
$$

$$
\leq \left( \frac { 1 } { \eta } \left( \mathbb { E } [ \Vert \hat { \pi } _ { 0 } ^ { ( \ell ) } - \pi _ { 0 } \Vert _ { \infty } ^ { 2 } ] \right) ^ { 1 / 2 } + 2 \mathbb { E } \left[ \left( \underset { \pi \in \Pi } { \operatorname* { s u p } } | S _ { \ell } ^ { \pi } - \hat { S } _ { \ell } ^ { \pi } | \right) ^ { 2 } \right] ^ { 1 / 2 } \right) \left( \mathbb { E } \left[ \left( \underset { \alpha \in ( 0 , \overline { { \alpha } } ] } { \operatorname* { s u p } } \Big \Vert \hat { m } ^ { ( \ell ) } - m \Big \Vert _ { \infty } \right) ^ { 2 } \right] \right) ^ { 1 / 2 } .
$$

Using definition of estimation rate, and the fact that $\hat { \pi } _ { 0 } ^ { ( \ell ) } ( p | x ) , \hat { m } ^ { ( \ell ) } ( x , p ; \alpha )$ is trained on $T - | \mathcal { T } _ { \ell } |$ data points (due to cross-fitting), we continue to bound (60)

$$
\mathbb { E } \left[ \left( \operatorname* { s u p } _ { \pi \in \Pi } | S _ { \ell } ^ { \pi } - \hat { S } _ { \ell } ^ { \pi } | \right) ^ { 2 } \right] \leq \frac { 1 } { \eta ^ { 2 } } \mathbb { E } \left[ \left. \hat { \pi } _ { 0 } ^ { ( \ell ) } - \pi _ { 0 } \right. _ { \infty } ^ { 2 } \right] \cdot \mathbb { E } \left[ \left( \operatorname* { s u p } _ { \pi \in \Pi } S _ { \ell } ^ { \pi } \right) ^ { 2 } \right] \leq \frac { 9 } { 4 \eta ^ { 2 } } \mathbb { E } \left[ \left. \hat { \pi } _ { 0 } ^ { ( \ell ) } - \pi _ { 0 } \right. _ { \infty } ^ { 2 } \right] .
$$

Thus, we have

$$
\mathbb { E } \left[ \operatorname* { s u p } _ { \pi \in \Pi , \alpha \in ( 0 , \overline { { \alpha } } ) } \Big | \mathcal { E } _ { 3 } ^ { ( \ell ) } \Big | \right] \leq \frac { 4 } { \eta } \left( \mathbb { E } [ \| \hat { \pi } _ { 0 } ^ { ( \ell ) } - \pi _ { 0 } \| _ { \infty } ^ { 2 } ] \right) ^ { 1 / 2 } \left( \mathbb { E } \left[ \left( \operatorname* { s u p } _ { \alpha \in ( 0 , \overline { { \alpha } } ] } \Big \| \hat { m } ^ { ( \ell ) } - m \Big \| _ { \infty } \right) ^ { 2 } \right] \right) ^ { 1 / 2 } \lesssim \frac { 4 } { \eta } T ^ { - \gamma _ { \infty } - \gamma _ { m } } .
$$

Following step (a)-(b) in the proof of Lemma C.2,

$$
\Big | \tilde { R } _ { \delta } ( \pi ) - \hat { R } _ { \delta } ( \pi ) \Big | \leq \operatorname* { s u p } _ { \alpha > 0 } \alpha \left| \log \frac { \frac { 1 } { L } \sum _ { \ell = 1 } ^ { L } \hat { W } ^ { ( \ell ) } ( \pi , \alpha ) } { \frac { 1 } { L } \sum _ { \ell = 1 } ^ { L } \hat { W } ^ { ( \ell ) } ( \pi , \alpha ) } \right| \leq \frac { 4 } { \underline { { \rho } } L } \sum _ { \ell = 1 } ^ { L } \operatorname* { s u p } _ { \alpha > 0 } \alpha \left| \hat { W } ^ { ( \ell ) } ( \pi , \alpha ) - \tilde { W } ^ { ( \ell ) } ( \pi , \alpha ) \right| ,
$$

and $\tilde { W } ^ { ( \ell ) } ( \pi , \alpha ) \ge \underline { { \rho } } / 2$ with probability at least $1 - \epsilon$ and $\begin{array} { r } { \Omega \left( \sqrt { \frac { d _ { \Pi } + \log ( 1 / \epsilon ) } { T } } \right) \leq h \leq \mathcal { O } \left( \frac { \rho } { 8 L _ { m } C _ { 1 } } \right) } \end{array}$ by Lemma C.7, we obtain with probability at least $1 - 4 \epsilon ,$ , by combining Steps 1-3,

$$
\Big | \tilde { R } _ { \delta } ( \pi ) - \hat { R } _ { \delta } ( \pi ) \Big | \lesssim \overline { { \alpha } } \Big ( \log \Big ( \frac { 1 } { h } \Big ) \Big ( \frac { d _ { \Pi } } { T h / L } + \sqrt { \frac { d _ { \Pi } } { T h / L } } + \sqrt { \frac { \log ( 1 / \epsilon ) } { T h / L } } + \frac { \log ( 1 / \epsilon ) } { T h / L } \Big ) + T ^ { - \gamma _ { \infty } - \gamma _ { m } } + h ^ { s } \Big ) ,
$$

for $s = 1 , 2$ in Assumption 2.4, which completes the proof.

Lemma C.6 (Rademacher Complexity Bound via Pseudo-dimension). Define afunction class ${ \mathcal { F } } _ { \Pi } = \{ f _ { \pi } | \pi \in \Pi \}$ , which has finite pseudo dimension $P d i m ( \mathcal { F } _ { \Pi } ) \leq C \cdot P d i m ( \Pi )$ for some constant C, and is uniformly bounded by $B _ { \Pi }$ . Let its second-moment be bounded by $\sigma ^ { 2 }$ . Under Assumption $4 . 3 ,$ , the expectation ofthe Rademacher complexity is bounded as:

$$
\mathbb { E } \Big [ \operatorname* { s u p } _ { \pi \in \Pi } \Big | \frac { 1 } { T ^ { \prime } } \sum _ { t = 1 } ^ { T ^ { \prime } } f _ { \pi } ( x _ { t } , p _ { t } ) - \mathbb { E } [ f _ { \pi } ] \Big | \Big ] \lesssim P d i m ( \Pi ) \frac { B _ { \Pi } } { T ^ { \prime } } + \sqrt { \frac { \sigma ^ { 2 } P d i m ( \Pi ) } { T ^ { \prime } } } , \mathrm { ~ } f o r \ a n y \ T ^ { \prime } \geq P d i m ( \mathcal { F } _ { \Pi } ) .
$$

ProofofLemma C.6. The proof structure closely follows that of Lemma A.5 in Kitagawa & Tetenov (2018) and Lemma B.8 in Ai et al. (2026). Kitagawa & Tetenov (2018) assumes that the function class $\mathcal { G } _ { \Pi }$ has the VC dimension $V ,$ , and Ai et al. (2026) relaxes this condition to the exponential bound on the covering number. In our case, we instead replace the VC-dimension-based covering number bound with the pseudo-dimension-based one. Here we provide the detailed proof.

We can bound this term by the standard symmetrization lemma (Proposition 4.11 in Wainwright, 2019). Define $\mathcal { G } _ { \Pi } : =$ $\{ g _ { \pi } = f _ { \pi } - \mathbb { E } [ f _ { \pi } ] \big \vert \pi \in \Pi \}$ as a centered function class. Introduce i.i.d. Rademacher variables $\epsilon _ { t } \in \{ - 1 , + 1 \}$ to get

$$
\mathbb { E } \Big [ \operatorname* { s u p } _ { g \in \mathcal { G } _ { \Pi } } \big | \sum _ { t = 1 } ^ { T ^ { \prime } } g ( x _ { t } , p _ { t } ) \big | \Big ] \leq 2 \mathbb { E } \Big [ \operatorname* { s u p } _ { g \in \mathcal { G } _ { \Pi } } \big | \sum _ { t = 1 } ^ { T ^ { \prime } } \epsilon _ { t } g ( x _ { t } , p _ { t } ) \big | \Big ]\tag{62}
$$

The expectation of the supremum of the Rademacher process is bounded by Dudley Theorem (Equation (5.48) in Wainwright, 2019):

$$
\mathbb { E } \Big [ \operatorname* { s u p } _ { g \in \mathcal { G } _ { \Pi } } \big | \sum _ { t = 1 } ^ { T ^ { \prime } } \epsilon _ { t } g ( x _ { t } , p _ { t } ) \big | \Big ] \lesssim \mathbb { E } \Big [ \sqrt { T ^ { \prime } } \int _ { 0 } ^ { 2 B _ { \Pi } } \sqrt { \log N ( \epsilon , \mathcal { G } _ { \Pi } , L _ { 2 } ( P _ { T ^ { \prime } } ) ) } d \epsilon \Big ]\tag{63}
$$

where $L _ { 2 } ( P _ { T ^ { \prime } } )$ is the empirical norm, $\mathcal { N } ( \epsilon , \mathcal { G } _ { \Pi } , L _ { 2 } ( P _ { T ^ { \prime } } ) )$ is the $L _ { 2 } ( P _ { T ^ { \prime } } )$ -covering number. Note that

$$
\begin{array} { r } { \mathcal { N } ( \epsilon , \mathcal { G } _ { \Pi } , L _ { 2 } ( P _ { T ^ { \prime } } ) ) \leq \mathcal { N } ( \epsilon / 2 , \mathcal { F } _ { \Pi } , L _ { 2 } ( P _ { T ^ { \prime } } ) ) + N ( \epsilon / 2 , \overline { { \mathcal { F } } } _ { \Pi } , L _ { 2 } ( P _ { T ^ { \prime } } ) ) , } \end{array}
$$

where

$$
\overline { { \mathcal { F } } } _ { \Pi } \triangleq \left\{ \overline { { f } } _ { \pi } = \mathbb { E } [ f _ { \pi } ( X , P ) ] \ : | \pi \in \Pi \right\}
$$

is a function class of constant functions, implying

$$
\begin{array} { r } { \Vert \overline { { f } } _ { \pi _ { 1 } } - \overline { { f } } _ { \pi _ { 2 } } \Vert _ { L _ { 2 } ( P _ { T ^ { \prime } } ) } = | \overline { { f } } _ { \pi _ { 1 } , } - \overline { { f } } _ { \pi _ { 2 } } | \leq \mathbb { E } _ { \mathbf { P } _ { 0 } \ast \pi _ { 0 } } | f _ { \pi _ { 1 } , } ( X , P ) - f _ { \pi _ { 2 } } ( X , P ) | . } \end{array}
$$

Consequently, we have

$$
\mathcal { N } ( \epsilon / 2 , \overline { { \mathcal { F } } } _ { \Pi } , L _ { 2 } ( P _ { T ^ { \prime } } ) ) \le \mathcal { N } ( \epsilon / 2 , \mathcal { F } _ { \Pi } , L _ { 1 } ( \mathbf { P } _ { 0 } * \pi _ { 0 } ) )
$$

To proceed, we use Definition D.2 and Lemma D.3,

$$
\log N ( \epsilon / 2 , \mathcal { F } _ { \Pi } , L _ { 1 } ( { \mathbf { P } _ { 0 } } * \pi _ { 0 } ) ) \le \log M ( \epsilon / 2 , \mathcal { F } _ { \Pi } , L _ { 1 } ( { \mathbf { P } _ { 0 } } * \pi _ { 0 } ) ) \lesssim \mathrm { P d i m } ( \mathcal { F } _ { \Pi } ) \log \left( \frac { 1 } { \epsilon } \log \left( \frac { 1 } { \epsilon } \right) \right) \lesssim d _ { \Pi } \log \left( \frac { 1 } { \epsilon } \log \left( \frac { 1 } { \epsilon } \right) \right)
$$

Hence, we conclude that

$$
\log \mathcal { N } ( \epsilon , \mathcal { G } _ { \Pi } , L _ { 2 } ( P _ { T ^ { \prime } } ) ) \leq 2 \log \mathcal { N } ( \epsilon / 2 , \mathcal { F } _ { \Pi } , L _ { 2 } ( P _ { T ^ { \prime } } ) ) + 2 \log \mathcal { N } ( \epsilon / 2 , \overline { { \mathcal { F } } } _ { \Pi } , L _ { 2 } ( P _ { T ^ { \prime } } ) ) \lesssim d _ { \Pi } \log \left( \frac { B _ { \Pi } } { \epsilon } \right) ,\tag{64}
$$

where we use log a + log $b \leq 2 \log ( a + b )$ and $\log ( x \log x ) \leq 2$ log x.

The following is slightly different from the last inequality in the proof of Lemma A.5 in Kitagawa & Tetenov (2018). In particular, replacing (A.5) by (64) gives

$$
\mathbb { E } \Big [ \operatorname* { s u p } _ { g \in \mathcal { G } _ { \Pi } } \sum _ { t = 1 } ^ { T ^ { \prime } } g ( x _ { t } , p _ { t } ) \Big ] \lesssim B _ { \Pi } \sqrt { T ^ { \prime } } \sqrt { d _ { \mathcal { G } _ { \Pi } } } \Big ( \sqrt { \frac { d _ { \Pi } } { T ^ { \prime } } } + \sqrt { \frac { d _ { \Pi } } { T ^ { \prime } } + \frac { \sigma ^ { 2 } } { B _ { \mathcal { G } _ { \Pi } } ^ { 2 } } } \Big ) \lesssim B _ { \Pi } d _ { \Pi } + \sqrt { T ^ { \prime } } \sqrt { \sigma ^ { 2 } d _ { \Pi } } ,
$$

which completes the proof.

Lemma C.7. Under Assumptions 3.2 and 2.2, picking $\begin{array} { r } { \Omega \left( \sqrt { \frac { d _ { \Pi } + \log ( 1 / \epsilon ) } { T } } \right) \leq h \leq \mathcal { O } \left( \frac { \rho } { 8 L _ { m } C _ { 1 } } \right) } \end{array}$ , then we have $\tilde { W } ^ { ( \ell ) } ( \pi , \alpha ) \geq$ $\rho / 2$ with probability at least $1 - \epsilon$

Proof of Lemma C.7. Define

$$
\tilde { W } ^ { ( \ell ) ^ { \prime } } ( \pi , \alpha ) \triangleq \frac { 1 } { | \mathcal { Z } _ { \ell } | S _ { \ell } ^ { \pi } } \sum _ { t \in \mathbb Z _ { \ell } } \frac { K ( \frac { p _ { t } - \pi ( x _ { t } ) } { \hbar } ) } { \hbar \pi _ { 0 } ( p _ { t } | x _ { t } ) } \exp ( - p _ { t } y _ { \ell } / \alpha ) + \frac { 1 } { | \mathcal { Z } _ { \ell } | } \sum _ { t \in \mathbb Z _ { \ell } } \Big ( m ( x _ { t } , \pi ( x _ { t } ) ; \alpha ) - \frac { 1 } { S _ { T } ^ { \pi } } \frac { K ( \frac { p _ { t } - \pi ( x _ { t } ) } { \hbar } ) } { \hbar \pi _ { 0 } ( p _ { t } | x _ { t } ) } m ( x _ { t } , \pi ( x _ { t } ) ; \alpha ) \Big ) ,
$$

Then, we can decompose as

$$
| \tilde { W } ^ { ( \ell ) } ( \pi , \alpha ) - \tilde { W } _ { h } ^ { \prime } ( \pi , \alpha ) | \le | \tilde { W } ^ { ( \ell ) } ( \pi , \alpha ) - \tilde { W } ^ { ( \ell ) ^ { \prime } } ( \pi , \alpha ) | + | \tilde { W } ^ { ( \ell ) ^ { \prime } } ( \pi , \alpha ) - \tilde { W } _ { h } ^ { \prime } ( \pi , \alpha ) | .
$$

Define

$$
A _ { \ell } ( \pi ) \triangleq \frac { 1 } { | { \mathcal Z } _ { \ell } | } \sum _ { t \in { \mathcal Z } _ { \ell } } \frac { K ( \frac { p \cdot - \pi ( x _ { t } ) } { \hbar } ) } { \hbar \pi _ { 0 } ( p _ { \ell } | x _ { t } ) } ( e ^ { - p _ { \ell } y _ { t } / \alpha } - m ( x _ { t } , \pi ( x _ { t } ) ; \alpha ) ) \quad \mathrm { ~ a n d ~ } \quad B _ { \ell } ( \pi ) \triangleq \frac { 1 } { | { \mathcal Z } _ { \ell } | } \sum _ { t \in { \mathcal Z } _ { \ell } } m ( x _ { t } , \pi ( x _ { t } ) ; \alpha ) .
$$

Then, we rewrite as $\begin{array} { r } { \tilde { W } ^ { ( \ell ) ^ { \prime } } ( \pi , \alpha ) = \frac { A _ { \ell } } { S _ { \ell } ^ { \pi } } + B _ { \ell } } \end{array}$ . First, we provide the high probability upper bound for $\mathrm { s u p } _ { \pi \in \Pi , \alpha \in ( 0 , \bar { \alpha } ] } | A _ { \ell } ( \pi ) -$ $\mathbb { E } [ A _ { \ell } ( \pi ) ]$ |. Define the function class $\mathcal { M } _ { A } = \{ m _ { \pi , \alpha } ^ { A } ( x , p , y ) : \pi \in \Pi , \alpha \in ( 0 , \bar { \alpha } ] \}$ , where

$$
m _ { \pi , \alpha } ^ { A } ( x , p , y ) = \frac { 1 } { h \pi _ { 0 } ( p | x ) } K \left( \frac { p - \pi ( x ) } { h } \right) \left[ e ^ { - p y / \alpha } - m ( x , \pi ( x ) ; \alpha ) \right] .
$$

Recalling that $| e ^ { - p y / \alpha } - m ( x , \pi ( x ) ; \alpha ) | \le 1$ , we have $\begin{array} { r } { \operatorname* { s u p } _ { m _ { \pi , \alpha } ^ { A } \in \mathcal { M } _ { A } } \| m _ { \pi } ^ { A } \| _ { \infty } \leq \frac { C _ { 2 } } { h \eta } : = B _ { \Pi } } \end{array}$ by Assumption 3.2 and 2.2. Also, for any $m _ { \pi , \alpha } ^ { A } \in { \mathcal { M } } _ { A }$ , we have the second-moment bound $\begin{array} { r } { \mathbb { E } [ ( m _ { \pi , \alpha } ^ { A } ) ^ { 2 } ] \ \leq \ \frac { C _ { 2 } } { h \eta } \ : = \ \sigma ^ { 2 } } \end{array}$ by (49). Following the similar argument of (58), we have $\mathrm { P d i m } ( \mathcal { M } _ { A } ) \ \lesssim \ d _ { \Pi }$ . Utilizing Talagrand Inequality (Lemma D.1) and setting $\begin{array} { r } { \beta ( \epsilon ) \lesssim \sqrt { \frac { \sigma ^ { 2 } \log ( 1 / \epsilon ) } { | { \mathcal T } _ { \ell } | } } + \frac { B _ { \Pi } \log ( 1 / \epsilon ) } { | { \mathcal T } _ { \ell } | } } \end{array}$ , we obtain

$$
\operatorname* { s u p } _ { \pi \in \Pi , \alpha \in ( 0 , \bar { \alpha } ] } \vert A _ { \ell } ( \pi ) - \mathbb { E } [ A _ { \ell } ( \pi ) ] \vert \leq \mathbb { E } \Big [ \operatorname* { s u p } _ { \pi \in \Pi , \alpha \in ( 0 , \bar { \alpha } ] } \vert A _ { \ell } ( \pi ) - \mathbb { E } [ A _ { \ell } ( \pi ) ] \vert \Big ] + \beta ( \epsilon ) ,
$$

with probability at least $1 - \epsilon / 3$ . Substituting $\begin{array} { r } { \mathbb { E } \left[ \operatorname* { s u p } _ { \pi \in \Pi , \alpha \in ( 0 , \bar { \alpha } ] } | A _ { \ell } ( \pi ) - \mathbb { E } [ A _ { \ell } ( \pi ) ] | \right] \lesssim \frac { B _ { \Pi } d _ { \Pi } } { | { \mathcal Z } _ { \ell } | } + \sqrt { \frac { \sigma ^ { 2 } d _ { \Pi } } { | { \mathcal Z } _ { \ell } | } } } \end{array}$ by Lemma C.6, we get the upper bound for

$$
\operatorname* { s u p } _ { \pi \in \Pi , \alpha \in ( 0 , \bar { \alpha } ] } | A _ { \ell } ( \pi ) - \mathbb { E } [ A _ { \ell } ( \pi ) ] | \lesssim \frac { d _ { \Pi } } { | { \mathcal { Z } } _ { \ell } | h } + \sqrt { \frac { d _ { \Pi } } { | { \mathcal { Z } } _ { \ell } | h } } + \sqrt { \frac { \log ( 1 / \epsilon ) } { | { \mathcal { Z } } _ { \ell } | h } } + \frac { \log ( 1 / \epsilon ) } { | { \mathcal { Z } } _ { \ell } | h } .
$$

Second, we provide the high probability upper bound for $\begin{array} { r } { \operatorname* { s u p } _ { \pi \in \Pi , \alpha \in ( 0 , \bar { \alpha } ] } | B _ { \ell } ( \pi ) - \mathbb { E } [ B _ { \ell } ( \pi ) ] } \end{array}$ |. Recalling $\mathbf { M } = \{ m _ { \pi , \alpha }$ $x \mapsto m ( x , \pi ( x ) ; \alpha ) : \pi \in \Pi , \alpha \in ( 0 , { \bar { \alpha } } ] \}$ in the proof of Lemma C.5, we also have

$$
\operatorname* { s u p } _ { \pi \in \Pi , \alpha \in ( 0 , \bar { \alpha } ] } \vert B _ { \ell } ( \pi ) - \mathbb { E } [ B _ { \ell } ( \pi ) ] \vert \lesssim \frac { d _ { \Pi } } { \vert { \mathcal { Z } } _ { \ell } \vert } + \sqrt { \frac { d _ { \Pi } } { \vert { \mathcal { Z } } _ { \ell } \vert } } + \sqrt { \frac { \log ( 1 / \epsilon ) } { \vert { \mathcal { Z } } _ { \ell } \vert } } + \frac { \log ( 1 / \epsilon ) } { \vert { \mathcal { Z } } _ { \ell } \vert } ,
$$

with probability at least $1 - \epsilon / 3$ . Thus, combined with Step 1 in Lemma C.4, with the probability at least $1 - 2 \epsilon / 3$ , we have

$$
| \tilde { W } ^ { ( \ell ) ^ { \prime } } ( \pi , \alpha ) - \tilde { W } _ { h } ^ { \prime } ( \pi , \alpha ) | = \bigg | \Big ( \frac { A _ { \ell } } { S _ { \ell } } + B _ { \ell } \Big ) - ( \mathbb { E } [ A _ { \ell } ] + \mathbb { E } [ B _ { \ell } ] ) \bigg | = \mathcal { O } \Big ( \sqrt { \frac { d _ { \Pi } \log ( 1 / \epsilon ) } { | \mathcal { Z } _ { \ell } | h } } \Big ) .\tag{65}
$$

To proceed, we pick $\begin{array} { r } { h \geq \Omega \left( \frac { d _ { \Pi } \log \left( 1 / \epsilon \right) } { \left. \mathcal { Z } _ { \ell } \right. \underline { { \rho } } ^ { 2 } } \right) } \end{array}$ such that $| \tilde { W } ^ { ( \ell ) ^ { \prime } } ( \pi , \alpha ) - \tilde { W } _ { h } ^ { \prime } ( \pi , \alpha ) | \le \underline { { \rho } } / 4 .$

On the other hand, , from the definitions of $\tilde { W } ^ { ( \ell ) } ( \pi , \alpha ) , \tilde { W } ^ { ( \ell ) ^ { \prime } } ( \pi , \alpha )$ , we have

$$
| \tilde { W } ^ { ( \ell ) } ( \pi , \alpha ) - \tilde { W } ^ { ( \ell ) ^ { \prime } } ( \pi , \alpha ) | = \left| \frac { 1 } { | \mathcal { Z } _ { \ell } | S _ { \ell } ^ { \pi } } \sum _ { t \in \mathcal { Z } _ { \ell } } \frac { K ( \frac { p _ { \ell } - \pi ( x _ { t } ) } { h } ) } { h \pi _ { 0 } ( p _ { t } | x _ { t } ) } \Big ( m ( x _ { t } , \pi ( x _ { t } ) ; \alpha ) - m ( x _ { t } , p _ { t } ; \alpha ) \Big ) \right| .
$$

$$
\leq \frac { 1 } { \vert { \mathcal { Z } } _ { \ell } \vert } \sum _ { t \in { \mathcal { Z } } _ { \ell } } \frac { K \left( \frac { p _ { t } - \pi \left( x _ { t } \right) } { h } \right) } { h \pi _ { 0 } ( p _ { t } \vert x _ { t } ) } \cdot L _ { m } \vert p _ { t } - \pi ( x _ { t } ) \vert + \vert S _ { \ell } ^ { \pi } - 1 \vert \quad \mathrm { ( b y ~ } L _ { m } \mathrm { - L i p s c h i t z ~ o f ~ } m ( x , \cdot ; \alpha ) \mathrm { ~ i n ~ } ( 3 7 ) \mathrm { ~ a n d ~ } ( 4 7 ) )
$$

Consider $\begin{array} { r } { \tilde { S } _ { \ell } ^ { \pi } \triangleq \frac { 1 } { | \mathcal { T } _ { \ell } | } \sum _ { t \in \mathcal { Z } _ { \ell } } \frac { K \left( \frac { p _ { t } - \pi ( x _ { t } ) } { h } \right) } { h \pi _ { 0 } ( p _ { t } | x _ { t } ) } \frac { | p _ { t } - \pi ( x _ { t } ) | } { h } } \end{array}$

Define a function class $\begin{array} { r } { \tilde { K } = \left\{ k _ { \pi } ( x , p ) = \frac { 1 } { h \pi _ { 0 } ( p | x ) } K \left( \frac { p - \pi ( x ) } { h } \right) \Big | \frac { p - \pi ( x ) } { h } \Big | : \pi \in \Pi \right\} } \end{array}$ . Under Assumption 3.2, it follows that $\begin{array} { r } { \operatorname* { s u p } _ { u } | u K ( u ) | \leq C _ { 1 } , \int u ^ { 2 } K ^ { 2 } \mathrm { \hat { ( } } u ) d u < C _ { 2 } } \end{array}$ and $\begin{array} { r } { \int | u K ( u ) | d u < \dot { C } _ { 3 } } \end{array}$ for some constants $C _ { 1 } , C _ { 2 } , C _ { 3 } < \infty$ . This ensures the uniform boundedness $\begin{array} { r } { \operatorname* { s u p } _ { k _ { \pi } \in \tilde { \mathcal { K } } } | k _ { \pi } | \leq \frac { C _ { 1 } } { h \eta } } \end{array}$ and $\begin{array} { r } { \mathbb { E } [ k _ { \pi } ^ { 2 } ] \leq \frac { C _ { 2 } } { h \eta } } \end{array}$ . Thus, by a similar process, we have with probability at least $1 - \epsilon / 6$

$$
\operatorname* { s u p } _ { \alpha \in \Pi } \big | \tilde { S } _ { T } ^ { \pi } \big | \lesssim | \mathbb { E } [ k _ { \pi } ] | + \frac { d _ { \Pi } } { | Z _ { \ell } | h } + \sqrt { \frac { d _ { \Pi } } { | Z _ { \ell } | h } } + \sqrt { \frac { \log ( 1 / \epsilon ) } { | Z _ { \ell } | h } } + \frac { \log ( 1 / \epsilon ) } { | Z _ { \ell } | h } \leq \frac { C _ { 3 } } { \eta } + \frac { d _ { \Pi } } { | Z _ { \ell } | h } + \sqrt { \frac { d _ { \Pi } } { | Z _ { \ell } | h } } + \sqrt { \frac { \log ( 1 / \epsilon ) } { | Z _ { \ell } | h } } + \frac { \log ( 1 / \epsilon ) } { | Z _ { \ell } | h }
$$

Picking $\begin{array} { r } { h \ge \Omega \left( \operatorname* { m a x } \{ \sqrt { \frac { d _ { \Pi } + \log ( 1 / \epsilon ) } { 2 C _ { 3 } ^ { 2 } | \mathcal { Z } _ { \ell } | / \eta ^ { 2 } } } , \sqrt { \frac { d _ { \Pi } + \log ( 1 / \epsilon ) } { 2 | \mathcal { Z } _ { \ell } | } } \} \right) } \end{array}$ , we have with probability at least $\begin{array} { r } { 1 - \epsilon / 3 , \mathrm { s u p } _ { \pi \in \Pi } | \tilde { S } _ { T } ^ { \pi } | \le 2 C _ { 3 } / \eta } \end{array}$ $\operatorname* { i n f } _ { \pi \in \Pi } | S _ { T } ^ { \pi } | \geq 1 / 2$ by Step 1 in Lemma C.4, then su $\begin{array} { r } { \operatorname { l p } _ { \pi \in \Pi } | \tilde { S } _ { T } ^ { \pi } / S _ { T } ^ { \pi } | \le 4 C _ { 3 } / \eta } \end{array}$ . To proceed, we pick $\begin{array} { r } { \Omega \left( \sqrt { \frac { d \mathrm { { n } } + \log \left( 1 / \epsilon \right) } { 2 | \mathcal { T } _ { \ell } | \underline { { \rho } } ^ { 2 } } } \right) \leq } \end{array}$ $\begin{array} { r } { h \leq \frac { \underline { { \rho } } \eta } { 8 L _ { m } C _ { 3 } } } \end{array}$ such that $| \tilde { W } _ { T } ^ { \prime } ( \pi , \alpha ) - \tilde { W } _ { h } ^ { \prime } ( \pi , \alpha ) | \le \rho / 4$

To conclude, we obtain $| \tilde { W } ^ { ( \ell ) } ( \pi , \alpha ) - \tilde { W } _ { h } ^ { \prime } ( \pi , \alpha ) | \le \underline { { \rho } } / 2$ , with probability at least $1 - \epsilon$ , which implies that $\tilde { W } _ { T } ( \pi , \alpha ) \geq$ $\underline { { \rho } } - \frac { \underline { { \rho } } } { 2 } \geq \frac { \underline { { \rho } } } { 2 }$ , where we use (43). □

## D. Proof of Auxiliary Results

Lemma D.1 (Talagrand’s Inequality, Lemma B.9 in Ai et al. (2026)). Let $Z _ { 1 } , . . . , Z _ { T }$ be i.i.d. Z-random variables and F be a class offunctions where each $f : { \mathcal { Z } } \to$ R in F satisfies su $) _ { f \in { \mathcal { F } } } \operatorname { s u p } _ { z \in { \mathcal { Z } } } | f ( z ) | \leq B ( { \mathcal { F } } )$ and $\begin{array} { r } { \operatorname* { s u p } _ { f \in \mathcal { F } } \mathbb { E } [ f ^ { 2 } ( Z ) ] \leq \sigma ^ { 2 } ( \mathcal { F } ) } \end{array}$ Denote $\begin{array} { r } { \Delta \equiv \operatorname* { s u p } _ { f \in \mathcal { F } } \Big | \frac { 1 } { T } \sum _ { t = 1 } ^ { T } ( f ( Z _ { t } ) - \mathbb { E } [ f ( Z _ { t } ) ] ) \Big | . } \end{array}$ . Then, thefollowing right and left tail bounds holdfor all $\epsilon > 0$

$$
\begin{array} { r } { { \mathbb { P } } \left( \Delta - { \mathbb { E } } [ \Delta ] > \epsilon \right) \leq 2 \exp \left( - \frac { T \epsilon ^ { 2 } } { 8 e ^ { 2 } \sigma ^ { \mathcal { ( F ) } } + 1 6 e B \left( \mathcal { F } \right) { \mathbb { E } } [ \Delta ] + 4 B \left( \mathcal { F } \right) \epsilon } \right) , } \\ { { \mathbb { P } } \left( - ( \Delta - { \mathbb { E } } [ \Delta ] ) > \epsilon \right) \leq 2 \exp \left( - \frac { T \epsilon ^ { 2 } } { 8 e ^ { 2 } \sigma ^ { \mathcal { ( F ) } } + 1 6 e B \left( \mathcal { F } \right) { \mathbb { E } } [ \Delta ] + 4 B \left( \mathcal { F } \right) \epsilon } \right) . } \end{array}
$$

Next, we introduce some related results for the pseudo-dimension.

Definition D.2. For a function class H and domain $x ,$

(i) (Theorem 12.1 in Anthony & Bartlett, 2009). For a set A, the covering number $\mathcal { N } ( \epsilon , A , d )$ is the size of the smallest ϵ-cover of A under metric d. Packing number $\mathcal { M } ( \epsilon , A , d )$ is the maximum cardinality of an ϵ-separated subset of A. For any subset $W \subset A$ , it satisfies $\mathcal { M } ( 2 \epsilon , W , d ) \leq \mathcal { N } ( \epsilon , W , d ) \leq \mathcal { M } ( \epsilon , W , d )$

$$
\begin{array} { r } { \mathcal { N } _ { p } ( \epsilon , \mathcal { H } , m ) : = \operatorname* { m a x } _ { | C | = m } \mathcal { N } ( \epsilon , \mathcal { H } | _ { C } , d _ { p } ) } \end{array}
$$

$$
\begin{array} { r } { \mathcal { M } _ { p } ( \epsilon , \mathcal { H } , m ) : = \operatorname* { m a x } _ { | C | = m } \mathcal { M } ( \epsilon , \mathcal { H } | _ { C } , d _ { p } ) } \end{array}
$$

$$
d _ { p }
$$

$$
\begin{array} { r } { d _ { 2 } ( x , y ) \ = \ \sqrt { \frac { 1 } { m } \sum _ { i = 1 } ^ { m } ( x _ { i } - y _ { i } ) ^ { 2 } } } \end{array}
$$

$$
\begin{array} { r } { d _ { 1 } ( x , y ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } | x _ { i } - y _ { i } | , } \end{array}
$$

$$
d _ { \infty } ( x , y ) = \operatorname* { m a x } _ { i \in [ m ] } | x _ { i } - y _ { i } |
$$

(iv) Pseudo-dimension Pdim(H) is the cardinality of the largest set $( w _ { 1 } , \ldots , w _ { m } )$ that can be pseudo-shattered, i.e., there exists a witness $u \in \mathbb { R } ^ { m }$ such that for all $b \in \{ \pm 1 \} ^ { m }$ , there is $h _ { b } \in \mathcal { H }$ satisfying $\operatorname { s g n } ( h _ { b } ( w _ { i } ) - u _ { i } ) = b _ { i }$ .

Lemma D.3 (Theorem 1,2 in Haussler (1995), Theorem 11.4, 18.5 in Anthony & Bartlett (2009)). Let $\mathcal { H } : \mathcal { X }  [ 0 , B ]$ with $P d i m ( \mathcal { H } ) = d .$ . Thefollowing hold:

(i) Define $B _ { \mathcal { H } } = \{ B _ { h } ( x , r ) = \operatorname { s g n } ( h ( x ) - r ) : h \in \mathcal { H } \}$ , then VC dimension satisfies $V C ( B _ { \mathcal { H } } ) = P d i m ( \mathcal { H } )$

(ii) Covering Number satisfies $\begin{array} { r } { \mathcal { N } _ { 1 } ( \epsilon , \mathcal { H } , m ) \leq e ( d + 1 ) ( 2 e / \epsilon ) ^ { d } . } \end{array}$

(iii) For all natural numbers $d \ge 1$ and $s \geq 1$ , let $n = s d .$ . There exists a subset $V \subset \{ 0 , 1 \} ^ { n }$ with VC dimension d such that for each integer $k \in \{ 1 , \ldots , n \}$ , the k/n-packing number of V with respect to the $d _ { 1 }$ metric satisfies $\begin{array} { r } { \mathcal { M } ( k / n , V , d _ { 1 } ) \geq \Big ( \frac { n } { 2 e ( k + d ) } \Big ) ^ { d } . } \end{array}$

(iv) Suppose $\mathcal { H } : \mathcal { X }  [ 0 , 1 ]$ , and has finite pseudo-dimension d. Then $\begin{array} { r } { \mathcal { M } ( \epsilon , \mathcal { H } , L _ { 1 } ( P ) ) \leq 2 \left( \frac { 2 e } { \epsilon } \log \left( \frac { 8 e } { \epsilon } \right) \right) ^ { d } } \end{array}$ for any probability distribution $P o n \mathcal { X } ,$ , and for all $0 < \epsilon \leq 1$

$\begin{array} { r l r } { ( \nu ) \ N ( \epsilon , \mathcal { H } , L _ { 2 } ( P _ { n } ) ) } & { { } \le } & { \ N ( \epsilon ^ { 2 } / ( 2 B ) , \mathcal { H } , L _ { 1 } ( P _ { n } ) ) } \end{array}$ , where $\begin{array} { r l r } { \| { \mathrm { ~  ~ \cdot ~ } } \| _ { L _ { 2 } ( P _ { n } ) } } & { { } : = } & { \sqrt { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( f ( z _ { i } ) - g ( z _ { i } ) ) ^ { 2 } } } \end{array}$ , and $\begin{array} { r } { \| \cdot \| _ { L _ { 1 } ( P _ { n } ) } : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } | f ( z _ { i } ) - g ( z _ { i } ) | . } \end{array}$

Lemma D.4 (Stability Properties of Pseudo-dimension). Let H be a class of real functions. Then

(i) $P d i m ( a \mathcal { H } + z ) = P d i m ( \mathcal { H } )$ for any constants $a , z \in \mathbb { R }$ with $a \neq 0 ;$

$( i i ) \mathcal { N } _ { p } ( L _ { \sigma } \epsilon , \sigma \circ \mathcal { H } , n ) \leq \mathcal { N } _ { p } ( \epsilon , \mathcal { H } , n )$ and Pdim( $\sigma \circ \mathcal { H } ) \lesssim P d i m ( \mathcal { H } )$ for any L<sub>σ</sub>-Lipschitzfunction $\sigma ;$

(iii) $P d i m ( g _ { 0 } \cdot \mathcal { H } ) \leq 2 P d i m ( \mathcal { H } )$ for fixed function g<sub>0</sub>.

ProofofLemma D.4. (i) Affine Transformation: If H shatters C with witness $r _ { H }$ , then $a { \mathcal { H } } + z$ shatters C with witness $a r _ { H } + z . \mathrm { I f } ~ a < 0$ , the required labels are simply flipped, not affecting shatterability.

(ii) Composition: Let C be an ϵ-cover for H. For any $h \in { \mathcal { H } } , \exists c \in { \mathcal { C } }$ s.t. $d _ { p } ( h , c ) \leq \epsilon .$ . By Lipschitzness, $| \sigma ( h ( x ) ) - $ $\overline { { \sigma ( c ( x ) ) | \le L _ { \sigma } | h ( x ) - c ( x ) | } }$ , thus $\sigma \circ \mathcal { C }$ is an $( L _ { \sigma } \epsilon )$ -cover for $\sigma \circ \mathcal { H } ,$ , which implies $\mathcal { N } _ { p } ( L _ { \sigma } \epsilon , \sigma \circ \mathcal { H } , n ) \leq \mathcal { N } _ { p } ( \epsilon , \mathcal { H } , n )$ Applying Definition D.2(i), Lemma D.3(ii,iii), we obtain Pdim $( \sigma \circ { \mathcal { H } } )$ l $\mathrm { { o g } } ( 1 / \epsilon ) \lesssim \log \bar { \mathcal { N } _ { 1 } } ( \epsilon , \mathcal { H } , n ) \lesssim \mathrm { { P d i m } } ( \mathcal { H } ) \log ( L _ { \sigma } / \epsilon )$ yielding the result for small ϵ. Finally, we get Pdim $( \sigma \circ { \mathcal { H } } ) \lesssim \operatorname { P d i m } ( { \mathcal { H } } )$

(iii) Multiplication: Let $\mathcal { G } _ { \Pi } = \boldsymbol { g } _ { 0 } \cdot \mathcal { H }$ and assume G<sub>Π</sub> pseudo-shatters a set C of size m with witness r. Partition C into $\overline { { C ^ { + } = \{ x \in C | g _ { 0 } ( x ) > 0 \} } }$ and $C ^ { - } = \{ x \in C | g _ { 0 } ( x ) < 0 \}$ . On $C ^ { + } , \operatorname { s g n } ( g _ { 0 } h - r ) = \operatorname { s g n } ( h - r / g _ { 0 } ) ;$ on $C ^ { - }$ , it is sgn $\left( h - r / g _ { 0 } \right) \cdot \left( - 1 \right)$ . It follows that H can pseudo-shatter the subsets $C ^ { + }$ and $C ^ { - }$ separately. Thus $| C | = | C ^ { + } | + | C ^ { - } | \leq$ 2Pdim(H). □