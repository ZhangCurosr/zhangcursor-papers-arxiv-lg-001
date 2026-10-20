# A Simple and Unified Method for Gradient-Variation Regret

Hang Yu Peng Zhao

YUHANG@LAMDA.NJU.EDU.CN

ZHAOP@LAMDA.NJU.EDU.CN

State Key Laboratoryfor Novel Software Technology, Nanjing University, China School of Artificial Intelligence, Nanjing University, China

## Abstract

Gradient-variation online learning has attracted increasing attention for its connections to game theory and optimization. It has been studied in both non-stationary online learning, which aims to minimize dynamic regret against time-varying comparator sequences, and universal online learning, which seeks to adapt to unknown loss curvature. Prior work obtained gradient-variation bounds using different techniques tailored to each setting. We propose a simple and unified method based on surrogate losses with a quadratic penalty on decision movement, yielding gradient-variation bounds in both settings. For non-stationary online learning, the method induces an optimisticto-gradient-variation reduction, enabling a broad class of algorithms to match the rate established by Zhao et al. (2024). For universal online learning, it yields a simple algorithm that matches the optimal universal regret bounds of Yan et al. (2024) while relaxing the extended smoothness assumption. Finally, we apply our method to online games and improve on existing guarantees.

Keywords: Gradient-variation regret, Universal online learning, Dynamic regret, Online game.

## 1. Introduction

Online convex optimization (OCO) provides a versatile framework for studying sequential interactions between a learner and an environment (Hazan, 2016; Orabona, 2019). At each round $t \in [ T ]$ the learner selects a decision $\mathbf { x } _ { t }$ from a compact convex set $\boldsymbol { \mathcal { X } } \subseteq \mathbb { R } ^ { d }$ , while the environment simultaneously chooses a convex loss function $f _ { t } : \mathcal { X }  \mathbb { R }$ The learner then incurs loss $f _ { t } ( \mathbf { x } _ { t } )$ and observes the gradient $\nabla f _ { t } ( \mathbf { x } _ { t } )$ , aiming to optimize the game-theoretical performance measure known as static regret (Cesa-Bianchi and Lugosi, 2006), which is defined as

$$
\mathrm { R E G } _ { T } = \sum _ { t = 1 } ^ { T } f _ { t } ( \mathbf { x } _ { t } ) - \underset { \mathbf { x } \in \mathcal { X } } { \operatorname* { m i n } } \sum _ { t = 1 } ^ { T } f _ { t } ( \mathbf { x } ) .\tag{1.1}
$$

For OCO, the minimax optimal regret bounds are $\mathcal { O } ( \sqrt { T } )$ for convex functions (Zinkevich, 2003), $\mathcal { O } ( \textstyle { \frac { 1 } { \lambda } } \log T )$ for λ-strongly convex functions (Hazan, 2016), and $\begin{array} { r } { \mathcal { O } ( { \frac { d } { \alpha } } \log T ) } \end{array}$ for α-exp-concave functions (Yang et al., 2014). Beyond worst-case minimax optimality, prior work studied regret bounds that adapt to problem-dependent hardness. Among these problem-dependent quantities, gradient variation (Chiang et al., 2012; Yang et al., 2014) is of particular interest. It quantifies the cumulative variation in gradients between consecutive loss functions:

$$
V _ { T } \triangleq \sum _ { t = 2 } ^ { T } \operatorname* { s u p } _ { \mathbf { x } \in \mathcal { X } } \Vert \nabla f _ { t } ( \mathbf { x } ) - \nabla f _ { t - 1 } ( \mathbf { x } ) \Vert _ { 2 } ^ { 2 } .\tag{1.2}
$$

Adapting to gradient variation improves regret bounds to $\mathcal { O } ( \sqrt { V _ { T } } ) , \mathcal { O } ( \frac { 1 } { \lambda } \log V _ { T } )$ , and $\begin{array} { r } { \mathcal { O } \big ( \frac { d } { \alpha } \log V _ { T } \big ) } \end{array}$ for convex, λ-strongly convex, and α-exp-concave loss functions, respectively. Gradient-variation adaptivity also helps bridge adversarial and stochastic optimization (Sachs et al., 2022; Chen et al., 2024), enables fast rates in games (Rakhlin and Sridharan, 2013b; Syrgkanis et al., 2015), and facilitates acceleration in smooth offline optimization (Cutkosky, 2019; Kavis et al., 2019; Zhao et al., 2025b), among other applications. Recent work further explored gradient variation in more challenging settings, including non-stationary online learning and universal online learning.

Non-stationary Online Learning. Static regret (1.1) measures performance against a fixed comparator $\mathbf { x } ^ { \star } \in \arg \operatorname* { m i n } _ { \mathbf { x } \in \mathcal { X } } \sum _ { t = 1 } ^ { T } f _ { t } ( \mathbf { x } )$ and becomes an inadequate measure in non-stationary environments where the optimal decision may change over time. To address this limitation, prior work studied dynamic regret (Zinkevich, 2003), which compares the learner’s performance with a sequence of time-varying comparators $\{ { \mathbf { u } } _ { t } \} _ { t = 1 } ^ { T }$ and is defined as follows:

$$
\mathrm { D - R E G } _ { T } ( \mathbf { u } _ { 1 : T } ) \triangleq \sum _ { t = 1 } ^ { T } f _ { t } ( \mathbf { x } _ { t } ) - \sum _ { t = 1 } ^ { T } f _ { t } ( \mathbf { u } _ { t } ) .\tag{1.3}
$$

A desirable dynamic regret bound adapts to the comparator path length $\begin{array} { r } { P _ { T } \triangleq \sum _ { t = 2 } ^ { T } \Vert \mathbf { u } _ { t } - \mathbf { u } _ { t - 1 } \Vert _ { 2 } , } \end{array}$ which measures cumulative changes in the comparator sequence and is unknown to the learner. Zhang et al. (2018) obtained the minimax-optimal dynamic regret bound $\mathcal { O } ( \sqrt { T ( 1 + P _ { T } ) } )$ , which was later strengthened for smooth functions to $\mathcal { O } ( \sqrt { ( 1 + V _ { T } + P _ { T } ) ( 1 + P _ { T } ) } )$ (Zhao et al., 2024).

Universal Online Learning. Classical approaches require prior knowledge of the curvature of loss functions to select suitable algorithms and obtain the corresponding regret guarantees. Such information, however, may be unavailable in practice. Universal online learning aims to develop a single algorithm that adapts to unknown curvature and matches the regret guarantees achievable when the curvature is known (van Erven and Koolen, 2016; Wang et al., 2020; Zhang et al., 2022; Yan et al., 2023, 2024). van Erven and Koolen (2016) initiated the study of universal online learning by developing a single algorithm with regret bounds of $\begin{array} { r } { \mathcal { O } \big ( \frac { d } { \alpha } \log T \big ) } \end{array}$ for α-exp-concave functions and $\mathcal { O } ( \sqrt { T } )$ for convex functions. Wang et al. (2020) subsequently obtained $\mathcal { O } ( \textstyle { \frac { 1 } { \lambda } } \log T )$ for λ-strongly convex functions while retaining optimal regret guarantees for convex and exp-concave functions. Exploiting smoothness, Zhang et al. (2022) refined minimax guarantees to optimal gradient-variation bounds for strongly convex and exp-concave functions. Yan et al. (2023) further obtained a regret of $\mathcal { O } ( \sqrt { V _ { T } \log V _ { T } } )$ bound for convex functions. Under an extended smoothness assumption, Yan et al. (2024) achieved optimal universal gradient-variation bounds for all three classes of functions.

Our Contributions. In this work, we propose a simple and unified method for gradient-variation regret bounds in these challenging settings. For non-stationary online learning, our method yields a black-box reduction that converts optimistic regret bounds into gradient-variation bounds. This reduction enables a broad class of algorithms to achieve $\mathcal { O } ( \sqrt { ( 1 + V _ { T } + P _ { T } ) ( 1 + P _ { T } ) } )$ , matching the rate established by Zhao et al. (2024). For universal online learning, our method simplifies algorithm design and yields an algorithm that matches the optimal gradient-variation bounds of Yan et al. (2024) while relaxing their extended smoothness assumption. These bounds are $\mathcal { O } ( \sqrt { V _ { T } } )$ $\begin{array} { r } { \mathcal { O } ( \frac { d } { \alpha } \log V _ { T } ) } \end{array}$ , and $\begin{array} { r } { \mathcal { O } ( \frac { 1 } { \lambda } \log V _ { T } ) } \end{array}$ for convex, α-exp-concave, and λ-strongly convex functions, respectively. Table 1 compares these guarantees with prior work. Notably, our algorithm preserves the “Regret bounded by Variation in Utilities” (RVU) property, which is crucial for fast convergence in games, while requiring minimal smoothness assumptions. We further apply our universal regret guarantee to facilitate fast convergence in online games. Under honest play, the sum of the two players’ regrets is $\mathcal { O } ( 1 )$ in both bilinear and strongly convex–strongly concave games, matching

Table 1: Comparison of universal regret guarantees. The regret columns show bounds for strongly convex, exp-concave, and convex loss functions using O(·) notation. We list universal guarantees in terms of the gradient variation $V _ { T }$ or the time horizon T. The “Regret bounded by Variation in Utilities” (RVU) column indicates whether the method supports this property for fast convergence in online games. In the Smoothness column, “No” means no smoothness assumption, “Yes” means smoothness on $x ,$ and “Extended” means smoothness on a larger domain containing X .
<table><tr><td rowspan=2 colspan=1>Works</td><td rowspan=1 colspan=1>Regret Bounds</td><td rowspan=2 colspan=1>RVU</td><td rowspan=2 colspan=1>Smoothness</td></tr><tr><td rowspan=1 colspan=1>Strongly Convex Exp-concave      Convex</td></tr><tr><td rowspan=1 colspan=1>Zhang et al. (2022)</td><td rowspan=1 colspan=1> $\mathcal { O } ( \log V _ { T } )$        $\mathcal { O } ( d \log V _ { T } )$         $\mathcal { O } ( \sqrt { T } )$ </td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1>Yes</td></tr><tr><td rowspan=1 colspan=1>Yan et al. (2023)</td><td rowspan=1 colspan=1> $\mathcal { O } ( \log V _ { T } )$        $\mathcal { O } ( d \log V _ { T } )$    $\mathcal { O } ( \sqrt { V _ { T } \log V _ { T } } )$ </td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>Yes</td></tr><tr><td rowspan=1 colspan=1>Yan et al. (2024)</td><td rowspan=1 colspan=1> $\mathcal { O } ( \log V _ { T } )$        $\mathcal { O } ( d \log V _ { T } )$        $\mathcal { O } ( \sqrt { V _ { T } } )$ </td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1>Extended</td></tr><tr><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=1> $\mathcal { O } ( \log V _ { T } )$        $\mathcal { O } ( d \log V _ { T } )$        $\mathcal { O } ( \sqrt { V _ { T } } )$ </td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>Yes</td></tr></table>

Yan et al. (2023). Under dishonest play, the individual regret guarantee in bilinear games improves from $\mathcal { O } ( \sqrt { T \log T } )$ in Yan et al. (2023) to $\mathcal { O } ( \sqrt { T } )$

Organization. The remainder of the paper is organized as follows. Section 2 introduces the preliminaries and reviews existing approaches to gradient-variation bounds. Section 3 presents our unified method for deriving gradient-variation regret bounds. Section 4 establishes a black-box reduction for non-stationary online learning. Section 5 develops a simple universal algorithm with optimal gradient-variation regret bounds. Section 6 applies these universal guarantees to online games. Finally, Section 7 concludes the paper. All proofs are deferred to the appendices.

## 2. Preliminaries

This section introduces the notation, assumptions, and definitions used throughout the paper and reviews two existing approaches to deriving gradient-variation regret bounds.

## 2.1. Notation, Assumptions, and Definitions

We use $\| \cdot \|$ for $\| \cdot \| _ { 2 }$ by default. For $\textbf { v } \in \mathbb { R } ^ { d }$ , let $v _ { i }$ denote its i-th coordinate. For a strictly convex and differentiable function $\psi : \mathcal { X }  \mathbb { R }$ , the induced Bregman divergence is defined as ${ \mathcal { D } } _ { \psi } ( \mathbf { x } , \mathbf { y } ) \triangleq \psi ( \mathbf { x } ) - \psi ( \mathbf { y } ) - \langle \nabla \psi ( \mathbf { y } ) , \mathbf { x } - \mathbf { y } \rangle$ . We write $a \lesssim b \ \mathrm { o r } a = \mathcal { O } ( b )$ when there exists a constant $C < \infty$ such that $a \leq C b$ . We use $\mathcal { O } ( \cdot )$ notation to highlight the dependence on T and $V _ { T }$ while treating iterated logarithmic factors as constants (Luo and Schapire, 2015). Assumption 1 (Boundedness). The domain $\mathcal { X } \subseteq \mathbb { R } ^ { d }$ has diameter at most D, so $\| \mathbf x - \mathbf y \| \leq D$ for all $\mathbf { x } , \mathbf { y } \in { \mathcal { X } }$ . For every $t \in [ T ]$ and $\mathbf { x } \in \mathcal { X }$ , the loss gradient satisfies $\| \nabla f _ { t } ( \mathbf { x } ) \| \leq G$ Assumption 2 (Smoothness). Each loss function $f _ { t }$ is L-smooth on X, meaning that $\parallel \nabla f _ { t } ( \mathbf { x } ) -$ $\nabla f _ { t } ( \mathbf { y } ) \| \leq L \| \mathbf { x } - \mathbf { y } \|$ for all $\mathbf { x } , \mathbf { y } \in { \mathcal { X } }$

The domain boundedness and gradient boundedness are standard assumptions for regret minimization in OCO (Shalev-Shwartz, 2007; Hazan, 2016). The smoothness assumption is necessary for first-order algorithms to achieve the gradient-variation regret (Chiang et al., 2012).

In the following, we provide formal definitions of strong convexity and exp-concavity.

Definition 1 (Strong Convexity). A function $f ( \cdot )$ is λ-strongly convex if $f ( \mathbf { x } ) - f ( \mathbf { y } ) \leq \langle \nabla f ( \mathbf { x } ) , \mathbf { x } -$ $\begin{array} { r } { \mathbf { y } \rangle - \frac { \lambda } { 2 } \cdot \Vert \mathbf { x } - \mathbf { y } \Vert ^ { 2 } } \end{array}$ holds for any $\mathbf { x } , \mathbf { y } \in { \mathcal { X } }$

Definition 2 (Exp-concavity). A function $f ( \cdot )$ is α-exponentially concave (abbreviated as expconcave) if $\begin{array} { r } { f ( \mathbf { x } ) - f ( \mathbf { y } ) \leq \langle \nabla f ( \mathbf { x } ) , \mathbf { x } - \mathbf { y } \rangle - \frac { \alpha } { 2 } \cdot \langle \nabla f ( \mathbf { x } ) , \mathbf { x } - \mathbf { y } \rangle ^ { 2 } } \end{array}$ holds for any $\mathbf { x } , \mathbf { y } \in { \mathcal { X } }$

The formal definition of β-exp-concavity is that $\exp ( - \beta f ( \cdot ) )$ is concave. Under Assumption 1, β-exp-concavity implies Definition 2 with $\alpha = { \textstyle { \frac { 1 } { 2 } } } \cdot \operatorname* { m i n } \{ 1 / ( 4 G D ) , \beta \}$ (Hazan, 2016, Lemma 4.3). For clarity and simplicity, we adopt Definition 2 as an alternative definition of exp-concavity.

## 2.2. Review of Existing Approaches

A standard approach to deriving gradient-variation regret bounds first controls regret in terms of the empirical gradient variation

$$
\bar { V } _ { T } \triangleq \sum _ { t = 2 } ^ { T } \Vert \nabla f _ { t } ( \mathbf x _ { t } ) - \nabla f _ { t - 1 } ( { \mathbf x _ { t - 1 } } ) \Vert ^ { 2 } .
$$

To express the resulting bound in terms of $V _ { T }$ , existing analyses decompose $\hat { V } _ { T }$ into a gradientvariation term and a term involving the learner’s decision trajectory. We summarize two types of analysis adopted in prior work for this decomposition: Correction-type and Bregman-type (Chiang et al., 2012; Rakhlin and Sridharan, 2013b; Syrgkanis et al., 2015; Zhao et al., 2020; Yan et al., 2023, 2024; Xie et al., 2024; Zhao et al., 2024, 2025a, 2026).

Correction-type Analysis Inserting $\nabla f _ { t - 1 } ( \mathbf { x } _ { t } )$ gives

$$
\begin{array} { l } { \displaystyle \bar { V } _ { T } \leq 2 \sum _ { t = 2 } ^ { T } \| \nabla f _ { t } ( { \mathbf x } _ { t } ) - \nabla f _ { t - 1 } ( { \mathbf x } _ { t } ) \| ^ { 2 } + 2 \displaystyle \sum _ { t = 2 } ^ { T } \| \nabla f _ { t - 1 } ( { \mathbf x } _ { t } ) - \nabla f _ { t - 1 } ( { \mathbf x } _ { t - 1 } ) \| ^ { 2 } } \\ { \displaystyle \quad \leq 2 V _ { T } + 2 L ^ { 2 } \displaystyle \sum _ { t = 2 } ^ { T } \| { \mathbf x } _ { t } - { \mathbf x } _ { t - 1 } \| ^ { 2 } , } \end{array}\tag{2.1}
$$

where the first inequality follows from the AM–GM inequality, and the second uses L-smoothness and the definition of $V _ { T } \ ( 1 . 2 )$ . To obtain a regret bound in terms of $V _ { T }$ alone, the key is to cancel the positive movement term $\begin{array} { r } { \sum _ { t = 2 } ^ { T } \| \mathbf x _ { t } - \mathbf x _ { t - 1 } \| ^ { 2 } } \end{array}$ in Eq. (2.1) with a matching negative term.

For static regret, optimistic online mirror descent (OOMD) provides the required negative movement terms when the loss function class and curvature parameters are known (Chiang et al., 2012). With a suitable configuration for each class, it achieves optimal gradient-variation regret bounds for convex, strongly convex, and exp-concave loss functions. These negative movement terms are also a key component of the RVU property, which enables fast convergence in online games (Rakhlin and Sridharan, 2013a; Syrgkanis et al., 2015). The cancellation required by Correction-type analysis is more difficult in universal online learning. Yan et al. (2023) made progress in this direction using a three-layer ensemble with cascaded corrections, but their universal regret bounds remain suboptimal. For dynamic regret, Zhao et al. (2024) designed surrogate losses for the base learners in a meta-base ensemble to cancel the movement term and establish a gradient-variation dynamic regret bound. Across the different settings discussed above, these works shared the Correction-type decomposition but used different mechanisms to obtain the negative terms required for cancellation.

Bregman-type Analysis For a fixed comparator $\mathbf { u } \in \mathcal { X }$ , Bregman-type decomposes $\hat { V } _ { T }$ by inserting the gradients $\nabla f _ { t } ( \mathbf { u } )$ and $\nabla f _ { t - 1 } ( \mathbf { u } )$

$$
\begin{array} { l } { \displaystyle \bar { V } _ { T } \leq 3 \sum _ { t = 2 } ^ { T } \Big ( \| \nabla f _ { t } ( { \mathbf x } _ { t } ) - \nabla f _ { t } ( { \mathbf u } ) \| ^ { 2 } + \| \nabla f _ { t } ( { \mathbf u } ) - \nabla f _ { t - 1 } ( { \mathbf u } ) \| ^ { 2 } + \| \nabla f _ { t - 1 } ( { \mathbf u } ) - \nabla f _ { t - 1 } ( { \mathbf x } _ { t - 1 } ) \| ^ { 2 } \Big ) } \\ { \displaystyle \quad \leq 6 L \sum _ { t = 2 } ^ { T } \mathcal { D } _ { f _ { t } } ( { \mathbf u } , { \mathbf x } _ { t } ) + 3 V _ { T } + 6 L \sum _ { t = 2 } ^ { T } \mathcal { D } _ { f _ { t - 1 } } ( { \mathbf u } , { \mathbf x } _ { t - 1 } ) \leq 3 V _ { T } + 1 2 L \sum _ { t = 1 } ^ { T } \mathcal { D } _ { f _ { t } } ( { \mathbf u } , { \mathbf x } _ { t } ) , \qquad ( 2 . 2 ) } \end{array}
$$

where the first inequality follows from the AM–GM inequality and the second from Theorem 2.1.5 of Nesterov (2018). Convexity yields matching negative Bregman-divergence terms in the regret decomposition that can cancel the positive terms in Eq. (2.2). Using this analysis, Yan et al. (2024) achieved optimal universal regret bounds for all three function classes. This analysis has two limitations. First, applying Theorem 2.1.5 of Nesterov (2018) requires L-smoothness on an extended domain $\mathcal { X } ^ { + }$ containing $x ,$ a stronger condition than Assumption 2. Second, the negative Bregman divergence terms do not directly support fast convergence in online games (Yan et al., 2024).

## 3. A General Method for Negative Movement

Our goal is to control the positive movement term in Eq. (2.1) while minimizing regret. This motivates a Lagrangian viewpoint: we treat cumulative movement $\begin{array} { r } { \sum _ { t = 2 } ^ { T } \| \mathbf x _ { t } - \mathbf x _ { t - 1 } \| ^ { 2 } } \end{array}$ as an auxiliary budget constraint and incorporate a quadratic penalty into the original loss function, yielding the surrogate loss functions

$$
\bar { f } _ { t } ( \mathbf { x } ) \triangleq f _ { t } ( \mathbf { x } ) + \frac { \gamma } { 2 } \left. \mathbf { x } - \mathbf { x } _ { t - 1 } \right. ^ { 2 } ,\tag{3.1}
$$

where $\gamma > 0$ is a parameter to be specified and $\mathbf { x } _ { 0 } ~ = ~ \mathbf { x } _ { 1 }$ . The quadratic penalty discourages movement from the preceding decision and induces a negative movement term in the analysis. Taking convex functions as an example, we have the following property:

Proposition 1. Under Assumption 1, let $f _ { t }$ be convex for every $t \in [ T ]$ . Let $\mathbf { v } _ { t } \triangleq \nabla f _ { t } ( \mathbf { x } _ { t } ) +$ $\gamma ( \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } )$ be the corrected gradient. For any $\gamma > 0$ and every decision sequence $\mathbf { x } _ { 1 : T } \in \mathcal { X } ^ { T }$ and comparator sequence $\mathbf { u } _ { 1 : T } \in \mathcal { X } ^ { T }$ , we always have

$$
\sum _ { t = 1 } ^ { T } \langle \nabla f _ { t } ( \mathbf { x } _ { t } ) , \mathbf { x } _ { t } - \mathbf { u } _ { t } \rangle = \underbrace { \sum _ { t = 1 } ^ { T } \langle \mathbf { v } _ { t } , \mathbf { x } _ { t } - \mathbf { u } _ { t } \rangle } _ { \mathrm { L i n e a r i z e d r e g r e t } } - \underbrace { \frac { \gamma } { 2 } \sum _ { t = 2 } ^ { T } \left. \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } \right. ^ { 2 } } _ { \mathrm { M o v e m e n t } } + \mathcal { O } ( P _ { T } ) ,\tag{3.2}
$$

where $\begin{array} { r } { P _ { T } \triangleq \sum _ { t = 2 } ^ { T } \Vert \mathbf { u } _ { t } - \mathbf { u } _ { t - 1 } \Vert } \end{array}$ is the comparator path length.

Proof Sketch. By the definition of $\mathbf { v } _ { t }$ , we have

$$
\begin{array} { r l } & { \displaystyle \sum _ { t = 1 } ^ { T } \langle \nabla f _ { t } ( { \mathbf x } _ { t } ) , { \mathbf x } _ { t } - { \mathbf u } _ { t } \rangle = \displaystyle \sum _ { t = 1 } ^ { T } \langle { \mathbf v } _ { t } , { \mathbf x } _ { t } - { \mathbf u } _ { t } \rangle - \gamma \displaystyle \sum _ { t = 1 } ^ { T } \langle { \mathbf x } _ { t } - { \mathbf x } _ { t - 1 } , { \mathbf x } _ { t } - { \mathbf u } _ { t } \rangle } \\ & { = \displaystyle \sum _ { t = 1 } ^ { T } \langle { \mathbf v } _ { t } , { \mathbf x } _ { t } - { \mathbf u } _ { t } \rangle - \displaystyle \frac { \gamma } { 2 } \sum _ { t = 2 } ^ { T } \| { \mathbf x } _ { t } - { \mathbf x } _ { t - 1 } \| ^ { 2 } - \displaystyle \frac { \gamma } { 2 } \sum _ { t = 1 } ^ { T } \left( \| { \mathbf x } _ { t } - { \mathbf u } _ { t } \| ^ { 2 } - \| { \mathbf x } _ { t - 1 } - { \mathbf u } _ { t } \| ^ { 2 } \right) , } \end{array}
$$

where the second equality follows from the three-point identity $2 \langle \mathbf { a } - \mathbf { b } , \mathbf { a } - \mathbf { c } \rangle = \| \mathbf { a } - \mathbf { c } \| ^ { 2 } -$ $\| \mathbf { b } - \mathbf { c } \| ^ { 2 } + \| \mathbf { a } - \mathbf { b } \| ^ { 2 }$ , for $\mathbf { a } , \mathbf { b } , \mathbf { c } \in \mathbb { R } ^ { d }$ and the last term of Eq. (3.2) is bounded by $\mathcal { O } ( P _ { T } )$

Note that we can generalize Proposition 1 to general Bregman divergence in Appendix A.2.

The negative movement in Eq. (3.2) plays an important role in optimistic online learning. In addition to a gradient sequence $\{ \mathbf { g } _ { t } \} _ { t = 1 } ^ { T }$ , an optimistic online learning algorithm has access to a prediction $\mathbf { \nabla } m _ { t + 1 }$ of the next gradient $\mathbf { g } _ { t + 1 }$ , called an optimism. By Proposition 1, any algorithm with a standard optimistic regret bound scaling with $\sqrt { \sum _ { t } \| \mathbf { g } _ { t } - \pmb { m } _ { t } \| ^ { 2 } }$ satisfies the “Regret bounded by Variation in Utilities” (RVU) property when applied to the surrogate losses in Eq. (3.1) with optimism $\mathbf { } m _ { t } =$ $\mathbf { g } _ { t - 1 }$ . This property is defined as follows:

Definition 3 (RVU Property). An algorithm with a parameter $\eta > 0$ satisfies the RVU property if there exist constants $\alpha , \beta , \omega > 0$ such that the regret $\mathrm { R E G } _ { T }$ on decision sequence $\{ \mathbf { x } _ { t } \} _ { t = 1 } ^ { T }$ and gradient sequence $\{ \mathbf { g } _ { t } \} _ { t = 1 } ^ { T }$ is bounded by

$$
\mathbf { R } \mathbf { \mathrm { E G } } _ { T } \leq \frac { \alpha } { \eta } + \beta \eta \sum _ { t = 2 } ^ { T } \| \mathbf { g } _ { t } - \mathbf { g } _ { t - 1 } \| _ { \infty } ^ { 2 } - \frac { \omega } { \eta } \sum _ { t = 2 } ^ { T } \| \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } \| _ { 1 } ^ { 2 } .\tag{3.3}
$$

A Toy Example. We illustrate our method with Optimistic-Adapt-ML-Prod (Wei et al., 2016). Its original optimistic regret bound lacks the negative movement term required by the RVU property and therefore does not directly establish this property. Our gradient correction introduces the required term through Proposition 1. Specifically, with the previous gradient as the optimism, Theorem 3.4 of Wei et al. (2016) gives the regret bound R $\begin{array} { r } { \mathtt { E G } _ { T } \le \mathcal { O } \big ( \sqrt { \sum _ { t } \| } \mathbf { g } _ { t } - \mathbf { g } _ { t - 1 } \| _ { \infty } ^ { 2 } \big ) } \end{array}$ . At each round t, we update the algorithm using the corrected gradient $\begin{array} { r } { \mathbf v _ { t } = \mathbf g _ { t } + { \frac { 1 } { 3 2 \eta } } \big ( \mathbf x _ { t } - \mathbf x _ { t - 1 } \big ) } \end{array}$ and set the optimism to $m _ { t } = \mathbf { v } _ { t - 1 }$ . Combining the original optimistic regret bound with Proposition 1 yields $\begin{array} { r } { \mathbf { R E G } _ { T } \leq \mathcal { O } \big ( \sqrt { \sum _ { t } \| \mathbf { g } _ { t } - \mathbf { g } _ { t - 1 } \| _ { \infty } ^ { 2 } } - \frac { 1 } { \eta } \sum _ { t } \| \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } \| _ { 1 } ^ { 2 } \big ) } \end{array}$ , establishing the RVU property for the modified algorithm. Thus, our method extends the applicability of the RVU property to optimistic algorithms for which a negative movement term was not previously established.

In the next two sections, we further illustrate how the negative movement term in Proposition 1 plays a key role in two settings: non-stationary online learning and universal online learning.

## 4. Gradient-Variation Non-Stationary Online Learning

In the non-stationary setting, Proposition 1 reduces the task of obtaining a gradient-variation dynamic regret bound to designing an algorithm with an optimistic dynamic regret bound.

Theorem 1 (Optimistic-to-Gradient-Variation Reduction). Under Assumption 1 and 2, let $\mathbf { u } _ { 1 : T }$ be a comparator sequence in X and let $\begin{array} { r } { P _ { T } \triangleq \sum _ { t = 2 } ^ { T } \Vert \mathbf { u } _ { t } - \mathbf { u } _ { t - 1 } \Vert } \end{array}$ ∥ be the path length. Suppose that the functions $\{ f _ { t } \} _ { t = 1 } ^ { T }$ are convex and that an online algorithm A satisfies the following bound for any gradient sequence $\{ \mathbf { g } _ { t } \} _ { t = 1 } ^ { T }$ and optimism sequence $\{ m _ { t } \} _ { t = 1 } ^ { T } .$

$$
\sum _ { t = 1 } ^ { T } \langle \mathbf { g } _ { t } , \mathbf { x } _ { t } - \mathbf { u } _ { t } \rangle \lesssim \sqrt { \left( 1 + P _ { T } \right) \left( 1 + P _ { T } + \sum _ { t = 1 } ^ { T } \left. \mathbf { g } _ { t } - \pmb { m } _ { t } \right. ^ { 2 } \right) } .\tag{4.1}
$$

Then, when run with $\mathbf { g } _ { t } = \nabla f _ { t } ( \mathbf { x } _ { t } ) + L ( \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } )$ and either of the following optimism,

$$
\begin{array} { r } { ( i ) \ m _ { t } = \nabla f _ { t - 1 } ( { \mathbf x } _ { t - 1 } ) + L ( { \mathbf x } _ { t - 1 } - { \mathbf x } _ { t - 2 } ) , \qquad ( i i ) \ m _ { t } = \nabla f _ { t - 1 } ( { \mathbf x } _ { t - 1 } ) , } \end{array}\tag{4.2}
$$

$$
\overline { { G , D , L , T > 0 } } .
$$

$$
\mathcal { A } _ { s }
$$

$$
t = 1
$$

$$
T
$$

$$
\mathbf { x } _ { 1 } \in { \mathcal { X } } .
$$

$$
\mathbf { x } _ { t } ,
$$

$$
f _ { t } ( \mathbf { x } _ { t } )
$$

$$
\nabla f _ { t } ( \mathbf { x } _ { t } )
$$

$$
\mathbf { g } _ { t } = \nabla f _ { t } ( \mathbf { x } _ { t } ) + L ( \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } )
$$

$$
\mathbf { g } _ { t }
$$

$$
\mathbf { \nabla } m _ { t + 1 }
$$

$$
\mathbf { \nabla } m _ { t + 1 }
$$

$$
\mathbf { x } _ { t + 1 }
$$

$$
\mathcal { A } _ { s }
$$

$$
\mathcal { A } _ { s } .
$$

$$
\begin{array} { r } { \mathrm { D - R E G } _ { T } ( \mathbf { u } _ { 1 : T } ) \leq \mathcal { O } \left( \sqrt { ( 1 + P _ { T } ) ( 1 + P _ { T } + V _ { T } ) } \right) . } \end{array}
$$

The proof is given in Appendix B.1, where we apply the optimistic dynamic regret bound in Eq. (4.1) to the linearized regret in Eq. (3.2) and then use the negative movement term in Eq. (3.2) to cancel the resulting positive terms, yielding the desired bound. Theorem 1 offers two advantages over the constrained version of the optimistic-to-gradient-variation reduction in Zhao et al. (2026, Theorem 1). First, our reduction requires smoothness only on $x ,$ whereas theirs requires smoothness on an extended domain $\mathcal { X } ^ { + }$ containing $\mathcal { X }$ or an unbounded domain. Second, our reduction preserves the RVU property, whereas theirs does not.

Conveniently, the Sword.optimism algorithm in Zhao et al. (2024) already satisfies the condition in Theorem 1. Sword.optimism uses an online ensemble framework with a two-layer meta-base architecture to adapt to an unknown comparator sequence. A diverse set of base learners is deployed to explore the environment, while a meta-algorithm adaptively tracks the best-performing base learner. Theorem 9 of Zhao et al. (2024) guarantees that Sword.optimism achieves the opti mistic dynamic regret bound in Eq. (4.1). We present Sword.optimism with the corrected gradient and optimism in Algorithm 1 and provide the proof in Appendix B.1.

Corollary 1. Under Assumption 1 and 2, suppose that $f _ { t }$ is convex for every $t \in [ T ]$ . Then, for every comparator sequence $\mathbf { u } _ { 1 : T } \in \mathcal { X } ^ { T }$ , Algorithm 1 guarantees

$$
\begin{array} { r } { \mathrm { D - R E G } _ { T } ( \mathbf { u } _ { 1 : T } ) \leq \mathcal { O } \left( \sqrt { ( 1 + P _ { T } + V _ { T } ) ( 1 + P _ { T } ) } \right) . } \end{array}
$$

Corollary 1 matches the gradient-variation dynamic regret established by Sword++ in Zhao et al. (2024) with a simpler analysis. Sword++ obtained the negative movement by combining negative terms from meta and base regret bounds, whereas we obtain it directly using the corrected gradient.

## 5. Gradient-Variation Universal Online Learning

For static regret, obtaining negative movement terms remains the main technical challenge in achieving gradient-variation bounds. When the function class is known, negative movement terms are relatively easy to obtain. Prior analyses (Chiang et al., 2012; Yang et al., 2014) can therefore establish the optimal gradient-variation bounds directly, without relying on Proposition 1.

In the universal setting, obtaining the required negative movement term is more challenging. To adapt to unknown function classes and curvature, universal online learning algorithms typically use a two-layer architecture: N base learners are maintained for strongly convex, exp-concave, and convex losses, while a meta learner tracks the best-performing base learner (Zhang et al., 2022; Yan et al., 2023, 2024). Specifically, for λ-strongly convex and α-exp-concave losses, these methods constructed candidate pools $\mathcal { H } ^ { \mathrm { s c } }$ and $\mathcal { H } ^ { \mathrm { e x p } }$ for the unknown parameters λ and $\alpha ,$ , respectively, using a geometric grid:

$$
{ \mathcal { H } } ^ { \mathrm { e x p } } = { \mathcal { H } } ^ { \mathrm { s c } } \triangleq \left\{ { \frac { 1 } { T } } , { \frac { 2 } { T } } , { \frac { 2 ^ { 2 } } { T } } , \cdots , { \frac { 2 ^ { n - 1 } } { T } } \right\} ,
$$

where $n = \lceil \log _ { 2 } T \rceil + 1 = \mathcal { O } ( \log T )$ is the number of candidates in each pool. One base learner is instantiated for each candidate $\lambda _ { i }$ or $\alpha _ { i }$ , with the final base learner, indexed by $N$ , assigned to convex losses. Let $\mathbf { x } _ { t , i }$ denote the decision of base learner i at round $t ,$ and let $p _ { t , i }$ denote its weight assigned by the meta-algorithm. The meta decision is $\begin{array} { r } { \mathbf { x } _ { t } = \sum _ { i = 1 } ^ { N } p _ { t , i } \mathbf { x } _ { t , i } } \end{array}$

For the meta algorithm, Zhao et al. (2025a) observed that a second-order optimistic regret bound with negative stability terms is crucial for achieving gradient-variation universality. Specifically, for each base learner $i ,$ the meta regret should satisfy

$$
\sum _ { t = 1 } ^ { T } r _ { t , i } \leq \mathcal { O } \left( \sqrt { \sum _ { t = 1 } ^ { T } ( r _ { t , i } - m _ { t , i } ) ^ { 2 } } - \sum _ { t = 2 } ^ { T } \left. \pmb { p } _ { t } - \pmb { p } _ { t - 1 } \right. _ { 1 } ^ { 2 } \right) ,\tag{5.1}
$$

where $r _ { t , i } \triangleq \langle \mathbf { g } _ { t } , \mathbf { x } _ { t } - \mathbf { x } _ { t , i } \rangle$ is the instantaneous meta regret relative to the i-th base learner, and $m _ { t , i }$ is the corresponding optimism. For strongly convex and exp-concave base learners, setting $m _ { t , i } = 0$ allows the negative curvature terms to offset the meta regret. For example, for α-expconcave functions, the negative curvature term $- \frac { \alpha } { 4 } \sum _ { t } r _ { t , i } ^ { 2 }$ yields $\begin{array} { r } { \sqrt { \sum _ { t } r _ { t , i } ^ { 2 } - \frac { \alpha } { 4 } \sum _ { t } r _ { t , i } ^ { 2 } } \le \mathcal { O } ( \bar { 1 } ) } \end{array}$ For convex functions, no negative curvature terms are available, so the negative stability term $\begin{array} { r } { - \sum _ { t = 2 } ^ { T } \| \pmb { p } _ { t } - \pmb { p } _ { t - 1 } \| _ { 1 } ^ { 2 } } \end{array}$ plays a key role in controlling the meta regret. To obtain the required guarantee, Yan et al. (2023) designed a complex two-layer meta algorithm based on MSMWC-MASTER (Chen et al., 2021). However, their regret bound for convex losses still contained an extra $\sqrt { \log V _ { T } }$ factor relative to the optimal rate.

In Proposition 1, the negative movement term in Eq. (3.2) can directly cancel the meta regret, without requiring additional negative terms in the meta regret bound. This simplifies the design of the meta algorithm: any algorithm with a second-order optimistic regret bound can serve as the meta algorithm. In what follows, we use Optimistic-Adapt-ML-Prod (Wei et al., 2016) as the meta algorithm, for which deriving negative stability terms in the regret bound was previously difficult.

Meta Learner. Optimistic-Adapt-ML-Prod updates the weight $p _ { t + 1 }$ as follows:

$$
\begin{array} { r l } & { W _ { t , i } = \left( W _ { t - 1 , i } \cdot \exp \left( \varepsilon _ { t - 1 , i } r _ { t , i } - \varepsilon _ { t - 1 , i } ^ { 2 } ( r _ { t , i } - m _ { t , i } ) ^ { 2 } \right) \right) ^ { \frac { \varepsilon _ { t , i } } { \varepsilon _ { t - 1 , i } } } , } \\ & { \qquad p _ { t + 1 , i } \propto \varepsilon _ { t , i } \cdot \exp ( \varepsilon _ { t , i } m _ { t + 1 , i } ) \cdot W _ { t , i } , } \end{array}\tag{5.2}
$$

where $r _ { t , i } = \langle \mathbf { g } _ { t } , \mathbf { x } _ { t } - \mathbf { x } _ { t , i } \rangle$ is the instantaneous meta regret relative to the i-th base learner. The optimistic vector ${ \mathbf m } _ { t } \in \mathbb { R } ^ { N }$ is designed as

$$
m _ { t , i } = \langle \mathbf { g } _ { t - 1 } , \mathbf { x } _ { t - 1 } - \mathbf { x } _ { t , i } \rangle { \mathrm { ~ f o r ~ } } i = N , { \mathrm { ~ a n d ~ } } m _ { t , i } = 0 { \mathrm { ~ f o r ~ } } i \neq N .\tag{5.3}
$$

Algorithm 2 Universal Gradient-Variation Regret with Corrected Gradient   
Input: Base learner configurations $\{ \mathcal { B } _ { i } \} _ { i \in [ N ] } \triangleq \{ \mathcal { B } _ { i } ^ { \mathrm { s c } } \} _ { i \in I ^ { \mathrm { s c } } } \cup \{ \mathcal { B } _ { i } ^ { \mathrm { e x p } } \} _ { i \in I ^ { \mathrm { e x p } } } \cup \{ \mathcal { B } ^ { \mathrm { c } } \}$   
1: Initialize: M — meta learner Optimistic-Adapt-ML-Prod with $\begin{array} { r } { W _ { 0 , i } = \frac { 1 } { N } } \end{array}$ for all $i \in [ N ]$   
2: $\{ B _ { i } \} _ { i \in [ N ] }$ — base learners as specified in Appendix C.1   
3: for $t = 1$ to $T$ do   
4: Play $\begin{array} { r } { \mathbf { x } _ { t } = \sum _ { i = 1 } ^ { N } p _ { t , i } \mathbf { x } _ { t , i } , } \end{array}$ , incur loss $f _ { t } ( \mathbf { x } _ { t } )$ , and observe $\nabla f _ { t } ( \mathbf { x } _ { t } )$   
5: Compute the corrected gradient $\begin{array} { r } { \mathbf { g } _ { t } = \nabla f _ { t } \big ( \mathbf { x } _ { t } \big ) + \frac { G } { D } \big ( \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } \big ) } \end{array}$   
6: $\{ B _ { i } \} _ { i = 1 } ^ { \bar { N } }$ update their own decisions to $\{ \mathbf { x } _ { t + 1 , i } \} _ { i = 1 } ^ { N }$ using g<sub>t</sub>   
7: For each $i \in [ N ] .$ , compute $r _ { t , i } = \langle \mathbf { g } _ { t } , \mathbf { x } _ { t } - \mathbf { x } _ { t , i } \rangle$ and set the optimism $\mathbf { \nabla } m _ { t + 1 }$ as in Eq. (5.3)   
8: Send $\mathbf { r } _ { t }$ and $m _ { t + 1 }$ to $\mathcal { M } ,$ , and obtain $\pmb { p } _ { t + 1 } \in \Delta _ { N }$ via Eq. (5.2) and Eq. (5.4)   
9: end for

Assuming that $\| \mathbf { g } _ { t } \| \leq H$ for all $t \in [ T ]$ , we set the learning rates as

$$
\varepsilon _ { t , i } = \operatorname* { m i n } \left\{ \frac { 1 } { 8 H D } , \sqrt { \frac { \ln N } { 4 H ^ { 2 } D ^ { 2 } + \sum _ { s = 1 } ^ { t } ( r _ { s , i } - m _ { s , i } ) ^ { 2 } } } \right\} , \qquad \varepsilon _ { 0 , i } = \frac { 1 } { 8 H D } .\tag{5.4}
$$

Following Yan et al. (2024), we construct the base learners as follows:

Base Learner. The base learners update their decisions $\{ \mathbf { x } _ { t , i } \} _ { i = 1 } ^ { N }$ using OOMD with their respective surrogate losses and optimisms. We construct the surrogate loss for each function class as follows:

$$
h _ { t , N } ^ { \mathrm { c } } ( \mathbf { x } ) \triangleq \langle \mathbf { g } _ { t } , \mathbf { x } \rangle , \quad h _ { t , i } ^ { \mathrm { s c } } ( \mathbf { x } ) \triangleq \langle \mathbf { g } _ { t } , \mathbf { x } \rangle + \frac { \lambda _ { i } } { 4 } \left. \mathbf { x } - \mathbf { x } _ { t } \right. ^ { 2 } , \quad h _ { t , i } ^ { \mathrm { c x p } } ( \mathbf { x } ) \triangleq \langle \mathbf { g } _ { t } , \mathbf { x } \rangle + \frac { \alpha _ { i } } { 4 } \langle \mathbf { g } _ { t } , \mathbf { x } - \mathbf { x } _ { t } \rangle ^ { 2 } .
$$

Here, $\lambda _ { i }$ and $\alpha _ { i }$ are selected from the candidate pools $\mathcal { H } ^ { \mathrm { s c } }$ and $\mathcal { H } ^ { \mathrm { e x p } }$ , respectively. For convex loss functions, we run optimistic online gradient descent (OOGD) on the surrogate loss function $h _ { t , N } ^ { \mathrm { c } } ( \cdot )$ with optimism $\nabla h _ { t - 1 , N } ^ { \mathrm { c } } \big ( \mathbf { x } _ { t - 1 } \big )$ . For strongly convex and exp-concave loss functions, we use OOGD and optimistic online Newton step (OONS), respectively, with surrogate loss functions $h _ { t , i } ^ { \mathrm { s c } } ( \cdot )$ and $h _ { t , i } ^ { \mathrm { e x p } } ( \cdot )$ and corresponding optimisms $\nabla h _ { t - 1 , i } ^ { \mathrm { s c } } ( \mathbf { x } _ { t - 1 , i } )$ and $\nabla h _ { t - 1 , i } ^ { \mathrm { e x p } } ( \mathbf { x } _ { t - 1 , i } )$ . For detailed update rules of differently configured OOMD, we refer readers to Appendix C.1.

Following Section 3, we use the corrected gradient $\mathbf { g } _ { t } = \nabla f _ { t } ( \mathbf { x } _ { t } ) + \gamma ( \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } )$ with $\gamma = G / D$ We denote the strongly convex and exp-concave base learners by $\boldsymbol { B } _ { i } ^ { \mathrm { s c } }$ and $\boldsymbol { B } _ { i } ^ { \mathrm { e x p } }$ , respectively, with corresponding index sets $I ^ { \mathrm { s c } }$ and $I ^ { \mathrm { e x p } }$ . The convex base learner is denoted by $B ^ { \mathrm { c } }$ . We summarize our algorithm in Algorithm 2 and present its universal guarantee in Theorem 2. The corresponding proof is provided in Appendix C.2.

Theorem 2. Under Assumption 1 and 2, Algorithm 2 achieves the following universal guarantee:

$$
\mathbf { R E G } _ { T } \leq \left\{ \begin{array} { l l } { \mathcal { O } \left( \frac { 1 } { \lambda } \log V _ { T } \right) , } & { f o r \lambda \ – s t r o n g l y c o n v e x l o s s e s , } \\ { \mathcal { O } \left( \frac { d } { \alpha } \log V _ { T } \right) , } & { f o r \alpha \ – e x p \ – c o n c a v e l o s s e s , } \\ { \mathcal { O } \left( \sqrt { V _ { T } } \right) , } & { f o r c o n v e x l o s s e s . } \end{array} \right.
$$

We highlight the advantages of Theorem 2 in the following three aspects:

① Regret Bounds. Theorem 2 improves the regret bound for convex losses in Yan et al. (2023) from $\mathcal { O } ( \sqrt { V _ { T } \log V _ { T } } )$ to the optimal rate $\mathcal { O } ( \sqrt { V _ { T } } )$ , while retaining optimal logarithmic regret bounds for strongly convex and exp-concave losses under the same assumptions. We also match the optimal rates of Yan et al. (2024) while removing their assumption of smoothness on an extended domain.

② Efficiency. Algorithm 2 uses only $\mathcal { O } ( \log T )$ base learners and one gradient query per round. Compared with Yan et al. (2023), it reduces the number of base learners from ${ \mathcal { O } } ( \log ^ { 2 } T )$ to ${ \mathcal { O } } ( \log T )$ It matches Yan et al. (2024) in both the number of base learners and gradient queries, while reducing computational overhead: our choice of optimism avoids their one-dimensional fixed-point computation, which requires $\mathcal { O } ( \log T )$ iterations per round.

③ Analysis. Our analysis uses a simple corrected gradient to obtain the negative movement term in Eq. (3.2) directly. This term allows us to cancel the meta regret without requiring additional negative stability terms in the meta regret bound, avoiding the analysis of cascaded correction terms in Yan et al. (2023). It also supports simpler optimism, removing the need to analyze the one-dimensional fixed-point problem in Yan et al. (2024).

## 6. Application to Fast Convergence in Online Games

In this section, we apply Theorem 2 to online games and provide a detailed case showing how our negative movement term in Proposition 1 can be used to facilitate fast convergence in games.

We study a two-player zero-sum game of the form min $\mathbf { \ x } { \in } x \operatorname* { m a x } _ { \mathbf { y } \in \mathcal { y } } f ( \mathbf { x } , \mathbf { y } )$ . The x-player minimizes the payoff, while the y-player maximizes it. To demonstrate universality, we consider two classes of games: bilinear games with $f ( \mathbf { x } , \mathbf { y } ) = \mathbf { x } ^ { \top } A \mathbf { y }$ , and strongly convex–strongly concave games in which f is λ-strongly convex in x and λ-strongly concave in y. At each round t, the players choose their strategies simultaneously and then observe their respective loss gradients $\nabla _ { \mathbf x } f ( \mathbf x _ { t } , \mathbf y _ { t } )$ and $- \nabla _ { \mathbf { y } } f ( \mathbf { x } _ { t } , \mathbf { y } _ { t } )$ . We call the setting honest when both players run the same agreed algorithm in a distributed manner, and dishonest when at least one player deviates from the agreed protocol. In this setting, we use the following standard boundedness and smoothness conditions.

Assumption 3 (Strategy sets and payoff gradients). The strategy sets X and Y are nonempty compact convex sets with diameter at most $D _ { \colon }$ , and both partial gradients $\nabla _ { \mathbf x } f ( \mathbf x , \mathbf y )$ and $\nabla _ { \mathbf { y } } { f } ( \mathbf { x } , \mathbf { y } )$ have norm at most G. For all feasible pairs $\displaystyle ( \mathbf { x } , \mathbf { y } )$ and $( \mathbf { x } ^ { \prime } , \mathbf { y } ^ { \prime } )$ , the payoff satisfies

$$
\begin{array} { r l } & { \| \nabla _ { \mathbf x } f ( \mathbf x , \mathbf y ) - \nabla _ { \mathbf x } f ( \mathbf x ^ { \prime } , \mathbf y ^ { \prime } ) \| \leq L ( \| \mathbf x - \mathbf x ^ { \prime } \| + \| \mathbf y - \mathbf y ^ { \prime } \| ) , } \\ & { \| \nabla _ { \mathbf y } f ( \mathbf x , \mathbf y ) - \nabla _ { \mathbf y } f ( \mathbf x ^ { \prime } , \mathbf y ^ { \prime } ) \| \leq L ( \| \mathbf x - \mathbf x ^ { \prime } \| + \| \mathbf y - \mathbf y ^ { \prime } \| ) . } \end{array}
$$

The pioneering work of Syrgkanis et al. (2015) showed that, in the honest setting, the sum of players regrets is bounded by $\mathcal { O } ( 1 )$ when each player runs an algorithm satisfying the RVU property defined in Definition 3. This bound enables efficient computation of Nash equilibria. Specifically, for bilinear games, setting the learning rates to $\begin{array} { r } { \eta ^ { \mathbf { x } } = \eta ^ { \mathbf { y } } = \sqrt { \frac { \omega } { L ^ { 2 } \beta } } } \end{array}$ in the RVU bounds yields

$$
\begin{array} { r l } {  { \mathrm { R E G } _ { T } ^ { \mathbf { x } } + \mathrm { R E G } _ { T } ^ { \mathbf { y } } \lesssim \frac { \alpha } { \eta ^ { \mathbf { x } } } + \frac { \alpha } { \eta ^ { \mathbf { y } } } + ( L ^ { 2 } \beta \eta ^ { \mathbf { y } } - \frac { \omega } { \eta ^ { \mathbf { x } } } ) \sum _ { t = 2 } ^ { T } \| \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } \| _ { 1 } ^ { 2 } } \quad } & { } \\ { + ( L ^ { 2 } \beta \eta ^ { \mathbf { x } } - \frac { \omega } { \eta ^ { \mathbf { y } } } ) \sum _ { t = 2 } ^ { T } \| \mathbf { y } _ { t } - \mathbf { y } _ { t - 1 } \| _ { 1 } ^ { 2 } \leq \mathcal { O } ( 1 ) , } \end{array}
$$

Table 2: Comparison of our regret guarantees with existing results. Honest play is measured by the sum of the players’ regrets, and dishonest play by each player’s individual regret. Both bilinear and strongly convex–strongly concave games are considered.
<table><tr><td></td><td>Games</td><td>Syrgkanis et al. (2015)</td><td>Zhang et al. (2022)</td><td>Yan et al. (2023)</td><td>Ours</td></tr><tr><td rowspan="2">Honest</td><td>Bilinear</td><td>0(1)</td><td> $\mathcal { O } ( \sqrt { T } )$ </td><td>0(1)</td><td>0(1)</td></tr><tr><td>Strongly convex- strongly concave</td><td>O(1)</td><td> $\mathcal { O } ( \lambda ^ { - 1 } \log T )$ </td><td>O(1)</td><td>O(1)</td></tr><tr><td rowspan="2">Dishonest</td><td>Bilinear</td><td> $\mathcal { O } ( \sqrt { T } )$ </td><td> $\mathcal { O } ( \sqrt { T } )$ </td><td> $\mathcal { O } ( \sqrt { T \log T } )$ </td><td> $\mathcal { O } ( \sqrt { T } )$ </td></tr><tr><td>Strongly convex- strongly concave</td><td> $\mathcal { O } ( \sqrt { T } )$ </td><td> $\mathcal { O } ( \lambda ^ { - 1 } \log T )$ </td><td> $\mathcal { O } ( \lambda ^ { - 1 } \log T )$ </td><td> $\mathcal { O } ( \lambda ^ { - 1 } \log T )$ </td></tr></table>

where the first step uses the smoothness of the payoff function. In the second step, the negative movement term in the RVU property plays a crucial role in cancellation.

For Algorithm 2, the negative movement term in Proposition 1 ensures the RVU property, allowing the algorithm to be applied directly to this setting. In both the honest and dishonest settings, Algorithm 2 achieves state-of-the-art performance. We provide the proof in Appendix D.

Theorem 3. Under Assumption 3, for both bilinear and strongly convex–strongly concave games, Algorithm 2 enjoys O(1) regret summation in the honest case:

$$
\begin{array} { r } { \mathbf { R E G } _ { T } ^ { \mathbf { x } } + \mathbf { R E G } _ { T } ^ { \mathbf { y } } \leq \mathcal { O } ( 1 ) . } \end{array}
$$

In the dishonest case, the algorithm achieves individual regret bounds of $\cdot \mathcal { O } ( \sqrt { T } )$ for bilinear games and $\mathcal { O } ( \lambda ^ { - 1 } \log T )$ for strongly convex–strongly concave games.

Table 2 compares our guarantees with prior work. Under honest play, Theorem 3 matches the $\mathcal { O } ( 1 )$ regret sums of Yan et al. (2023) for both game classes, improving upon the corresponding bounds of Zhang et al. (2022). Under dishonest play, it matches the individual regret bounds of Zhang et al. (2022) for both game classes and improves the dishonest-play bound of Yan et al. (2023) from $\mathcal { O } ( \sqrt { T \log T } )$ to $\mathcal { O } ( \sqrt { T } )$ for bilinear games.

## 7. Conclusion

We propose a simple and unified method for obtaining negative terms in online regret bounds. These terms are key to establishing gradient-variation guarantees and fast convergence in games. In the non-stationary setting, we establish a black-box reduction from optimistic regret bounds to gradient-variation bounds. This reduction enables a broad class of algorithms to recover the gradient-variation dynamic regret guarantee $\mathcal { O } ( \sqrt { ( 1 + V _ { T } + P _ { T } ) ( 1 + P _ { T } ) } )$ established by Zhao et al. (2024). Our method yields a simple and efficient universal algorithm with regret bounds of $\mathcal { O } ( \sqrt { V _ { T } } ) , \mathcal { O } ( \textstyle { \frac { d } { \alpha } } \log V _ { T } )$ , and $\begin{array} { r } { \mathcal { O } ( \frac { 1 } { \lambda } \log V _ { T } ) } \end{array}$ for convex, α-exp-concave, and λ-strongly convex losses, respectively. It matches the optimal rates of Yan et al. (2024) without their extended smoothness assumption, uses only $\mathcal { O } ( \log T )$ base learners and one gradient query per round, and incurs lower computational cost per round. We further apply the resulting universal guarantee to online games and improve on existing guarantees (Zhang et al., 2022; Yan et al., 2023).

## AI-use Statement

The authors developed the idea of introducing a negative movement term with assistance from GPT-6 Astra. Building on this idea, they developed new algorithms for gradient-variation non-stationary online learning and obtained improved results for gradient-variation universal online learning. GPT-6 Astra was also used to assist with reviewing and polishing the manuscript. The authors take full responsibility for the content and correctness of the paper.

## References

Nicolò Cesa-Bianchi and Gábor Lugosi. Prediction, Learning, and Games. Cambridge University Press, 2006.

Liyu Chen, Haipeng Luo, and Chen-Yu Wei. Impossible tuning made possible: A new expert algorithm and its applications. In Proceedings of the 34th Conference on Learning Theory (COLT), pages 1216–1259, 2021.

Sijia Chen, Yu-Jie Zhang, Wei-Wei Tu, Peng Zhao, and Lijun Zhang. Optimistic online mirror descent for bridging stochastic and adversarial online convex optimization. Journal of Machine Learning Research, 25(178):1–62, 2024.

Chao-Kai Chiang, Tianbao Yang, Chia-Jung Lee, Mehrdad Mahdavi, Chi-Jen Lu, Rong Jin, and Shenghuo Zhu. Online optimization with gradual variations. In Proceedings of the 25th Annual Conference on Learning Theory (COLT), pages 6.1–6.20, 2012.

Ashok Cutkosky. Anytime online-to-batch, optimism and acceleration. In Proceedings of the 36th International Conference on Machine Learning (ICML), pages 1446–1454, 2019.

Elad Hazan. Introduction to Online Convex Optimization. Foundations and Trends in Optimization, 2(3-4):157–325, 2016.

Ali Kavis, Kfir Y. Levy, Francis R. Bach, and Volkan Cevher. UniXGrad: A universal, adaptive algorithm with optimal guarantees for constrained optimization. In Advances in Neural Information Processing Systems 32 (NeurIPS), pages 6257–6266, 2019.

Haipeng Luo and Robert E. Schapire. Achieving all with no parameters: AdaNormalHedge. In Proceedings ofthe 28th Annual Conference on Learning Theory (COLT), pages 1286–1304, 2015.

Yurii Nesterov. Lectures on Convex Optimization, volume 137. Springer, 2018.

Francesco Orabona. A modern introduction to online learning. ArXiv Preprint, arXiv:1912.13213, 2019.

Alexander Rakhlin and Karthik Sridharan. Online learning with predictable sequences. In Proceedings ofthe 26th Conference on Learning Theory (COLT), pages 993–1019, 2013a.

Sasha Rakhlin and Karthik Sridharan. Optimization, learning, and games with predictable sequences. In Advances in Neural Information Processing Systems 26 (NIPS), pages 3066–3074, 2013b.

Sarah Sachs, Hedi Hadiji, Tim van Erven, and Cristóbal Guzmán. Between stochastic and adversarial online convex optimization: Improved regret bounds via smoothness. In Advances in Neural Information Processing Systems 35 (NeurIPS), pages 691–702, 2022.

Shai Shalev-Shwartz. Online Learning: Theory, Algorithms and Applications. PhD thesis, The Hebrew University of Jerusalem, 2007.

Vasilis Syrgkanis, Alekh Agarwal, Haipeng Luo, and Robert E Schapire. Fast convergence of regularized learning in games. In Advances in Neural Information Processing Systems 28 (NIPS), pages 2989–2997, 2015.

Tim van Erven and Wouter M. Koolen. MetaGrad: Multiple learning rates in online learning. In Advances in Neural Information Processing Systems 29 (NIPS), pages 3666–3674, 2016.

Guanghui Wang, Shiyin Lu, and Lijun Zhang. Adaptivity and optimality: A universal algorithm for online convex optimization. In Proceedings of the 35th Conference on Uncertainty in Artificial Intelligence (UAI), pages 659–668, 2020.

Chen-Yu Wei, Yi-Te Hong, and Chi-Jen Lu. Tracking the best expert in non-stationary stochastic environments. In Advances in Neural Information Processing Systems 29 (NIPS), pages 3972– 3980, 2016.

Yan-Feng Xie, Peng Zhao, and Zhi-Hua Zhou. Gradient-variation online learning under generalized smoothness. In Advances in Neural Information Processing Systems 37 (NeurIPS), pages 37865– 37899, 2024.

Yu-Hu Yan, Peng Zhao, and Zhi-Hua Zhou. Universal online learning with gradient variations: A multi-layer online ensemble approach. In Advances in Neural Information Processing Systems 36 (NeurIPS), pages 37682–37715, 2023.

Yu-Hu Yan, Peng Zhao, and Zhi-Hua Zhou. A simple and optimal approach for universal online learning with gradient variations. In Advances in Neural Information Processing Systems 37 (NeurIPS), pages 11132–11163, 2024.

Tianbao Yang, Mehrdad Mahdavi, Rong Jin, and Shenghuo Zhu. Regret bounded by gradual variation for online convex optimization. Machine Learning, 95(2):183–223, 2014.

Lijun Zhang, Shiyin Lu, and Zhi-Hua Zhou. Adaptive online learning in dynamic environments. In Advances in Neural Information Processing Systems 31 (NeurIPS), pages 1330–1340, 2018.

Lijun Zhang, Guanghui Wang, Jinfeng Yi, and Tianbao Yang. A simple yet universal strategy for online convex optimization. In Proceedings of the 39th International Conference on Machine Learning (ICML), pages 26605–26623, 2022.

Peng Zhao, Yu-Jie Zhang, Lijun Zhang, and Zhi-Hua Zhou. Dynamic regret of convex and smooth functions. In Advances in Neural Information Processing Systems 33 (NeurIPS), pages 12510– 12520, 2020.

Peng Zhao, Yu-Jie Zhang, Lijun Zhang, and Zhi-Hua Zhou. Adaptivity and non-stationarity: Problem-dependent dynamic regret for online convex optimization. Journal ofMachine Learning Research, 25(98):1–52, 2024.

Peng Zhao, Yu-Hu Yan, Hang Yu, and Zhi-Hua Zhou. Adaptivity and universality: Problemdependent universal regret for online convex optimization. ArXiv Preprint, arXiv:2511.19937, 2025a.

Yuheng Zhao, Yu-Hu Yan, Kfir Y Levy, and Peng Zhao. Gradient-variation online adaptivity for accelerated optimization with Hölder smoothness. In Advances in Neural Information Processing Systems 38 (NeurIPS), pages 3749–3779, 2025b.

Yuheng Zhao, Andrew Jacobsen, Nicolò Cesa-Bianchi, and Peng Zhao. Gradient-variation regret bounds for unconstrained online learning. In Proceedings of the 39th Annual Conference on Learning Theory (COLT), pages 7062–7104, 2026.

Martin Zinkevich. Online convex programming and generalized infinitesimal gradient ascent. In Proceedings of the 20th International Conference on Machine Learning (ICML), pages 928–936, 2003.

## Appendix A. Omitted Details in Section 3

In this section, we provide the proof of Proposition 1 in Appendix A.1 and its extension to general Bregman divergences in Appendix A.2.

## A.1. Proof of Proposition 1

Proof. Recall that $\mathbf { v } _ { t } \triangleq \nabla f _ { t } ( \mathbf { x } _ { t } ) + \gamma ( \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } )$ . We have

$$
\begin{array} { r l r } {  { \sum _ { t = 1 } ^ { T } \langle \nabla f _ { t } ( \mathbf x _ { t } ) , \mathbf x _ { t } - \mathbf u _ { t } \rangle = \sum _ { t = 1 } ^ { T } \langle \mathbf v _ { t } , \mathbf x _ { t } - \mathbf u _ { t } \rangle - \gamma \sum _ { t = 1 } ^ { T } \langle \mathbf x _ { t } - \mathbf x _ { t - 1 } , \mathbf x _ { t } - \mathbf u _ { t } \rangle } } \\ & { = \displaystyle \sum _ { t = 1 } ^ { T } \langle \mathbf v _ { t } , \mathbf x _ { t } - \mathbf u _ { t } \rangle - \frac { \gamma } { 2 } \sum _ { t = 2 } ^ { T } \| \mathbf x _ { t } - \mathbf x _ { t - 1 } \| ^ { 2 } - \frac { \gamma } { 2 } \displaystyle \sum _ { t = 1 } ^ { T } ( \| \mathbf x _ { t } - \mathbf u _ { t } \| ^ { 2 } - \| \mathbf x _ { t - 1 } - \mathbf u _ { t } \| ^ { 2 } ) } \\ & { \leq \displaystyle \sum _ { t = 1 } ^ { T } \langle \mathbf v _ { t } , \mathbf x _ { t } - \mathbf u _ { t } \rangle - \frac { \gamma } { 2 } \displaystyle \sum _ { t = 2 } ^ { T } \| \mathbf x _ { t } - \mathbf x _ { t - 1 } \| ^ { 2 } + \frac { \gamma D ^ { 2 } } { 2 } - \frac { \gamma } { 2 } \sum _ { t = 2 } ^ { T } ( \| \mathbf x _ { t - 1 } - \mathbf u _ { t - 1 } \| ^ { 2 } - \| \mathbf x _ { t - 1 } - \mathbf u _ { t } \| ^ { 2 } ) } \\ & { \leq \displaystyle \sum _ { t = 1 } ^ { T } \langle \mathbf v _ { t } , \mathbf x _ { t } - \mathbf u _ { t } \rangle - \frac { \gamma } { 2 } \sum _ { t = 2 } ^ { T } \| \mathbf x _ { t } - \mathbf x _ { t - 1 } \| ^ { 2 } + \frac { \gamma D ^ { 2 } } { 2 } + \gamma D P _ { T } , } & { ( \mathbf A . 1 } \end{array}
$$

where the second step follows from the three-point identity $2 \langle \mathbf { a - b } , \mathbf { a - c } \rangle = \| \mathbf { a - c } \| ^ { 2 } - \| \mathbf { b - c } \| ^ { 2 } +$ $\| \mathbf { a } - \mathbf { b } \| ^ { 2 }$ , for $\mathbf { a } , \mathbf { b } , \mathbf { c } \in \mathbb { R } ^ { d }$ , the third step uses the boundedness assumption and the last inequality follows from $\| \mathbf { a } \| ^ { 2 } - \| \mathbf { b } \| ^ { 2 } \leq \| \mathbf { a } + \mathbf { b } \| \cdot \| \mathbf { a } - \mathbf { b } \|$ for any $\mathbf { a } , \mathbf { b } \in \mathbb { R } ^ { d }$

## A.2. Extension to Bregman Movement

Proposition 2. Under Assumption 1, let $f _ { t }$ be convex for every $t \in [ T ]$ and let ψ be differentiable, convex, and $L ^ { \prime }$ -smooth on $\mathcal { X }$ . Set $\mathbf { x } _ { 0 } = \mathbf { x } _ { 1 }$ and let $\mathbf { v } _ { t } \triangleq \nabla f _ { t } ( \mathbf { x } _ { t } ) + \gamma \left( \nabla \psi ( \mathbf { x } _ { t } ) - \nabla \psi ( \mathbf { x } _ { t - 1 } ) \right)$ be the corrected gradient. For any $\gamma > 0$ and every decision sequence $\mathbf { x } _ { 1 : T } \in \mathcal { X } ^ { T }$ and comparator sequence $\mathbf { u } _ { 1 : T } \in \mathcal { X } ^ { T }$ , we have

$$
\sum _ { t = 1 } ^ { T } \langle \nabla f _ { t } ( \mathbf { x } _ { t } ) , \mathbf { x } _ { t } - \mathbf { u } _ { t } \rangle \leq \sum _ { t = 1 } ^ { T } \langle \mathbf { v } _ { t } , \mathbf { x } _ { t } - \mathbf { u } _ { t } \rangle - \gamma \sum _ { t = 2 } ^ { T } { D _ { \psi } ( \mathbf { x } _ { t } , \mathbf { x } _ { t - 1 } ) } + \frac { \gamma L ^ { \prime } D ^ { 2 } } { 2 } + \gamma L ^ { \prime } D P _ { T } ,
$$

where $\begin{array} { r } { P _ { T } \triangleq \sum _ { t = 2 } ^ { T } \Vert \mathbf { u } _ { t } - \mathbf { u } _ { t - 1 } \Vert } \end{array}$ is the comparator path length.

Proof. Recall that $\mathbf { v } _ { t } \triangleq \nabla f _ { t } ( \mathbf { x } _ { t } ) + \gamma \left( \nabla \psi ( \mathbf { x } _ { t } ) - \nabla \psi ( \mathbf { x } _ { t - 1 } ) \right)$ and $\mathbf { x } _ { 0 } = \mathbf { x } _ { 1 }$ . We have

$$
\begin{array} { r l } & { \displaystyle \sum _ { t = 1 } ^ { T } \langle \nabla f _ { t } ( { \mathbf x } _ { t } ) , { \mathbf x } _ { t } - { \mathbf u } _ { t } \rangle = \displaystyle \sum _ { t = 1 } ^ { T } \langle { \mathbf v } _ { t } , { \mathbf x } _ { t } - { \mathbf u } _ { t } \rangle - \gamma \displaystyle \sum _ { t = 1 } ^ { T } \langle \nabla \psi ( { \mathbf x } _ { t } ) - \nabla \psi ( { \mathbf x } _ { t - 1 } ) , { \mathbf x } _ { t } - { \mathbf u } _ { t } \rangle } \\ & { = \displaystyle \sum _ { t = 1 } ^ { T } \langle { \mathbf v } _ { t } , { \mathbf x } _ { t } - { \mathbf u } _ { t } \rangle - \gamma \displaystyle \sum _ { t = 2 } ^ { T } { \mathcal D } _ { \psi } ( { \mathbf x } _ { t } , { \mathbf x } _ { t - 1 } ) - \gamma \displaystyle \sum _ { t = 1 } ^ { T } ( { \mathcal D } _ { \psi } ( { \mathbf u } _ { t } , { \mathbf x } _ { t } ) - { \mathcal D } _ { \psi } ( { \mathbf u } _ { t } , { \mathbf x } _ { t - 1 } ) ) } \\ & { \le \displaystyle \sum _ { t = 1 } ^ { T } \langle { \mathbf v } _ { t } , { \mathbf x } _ { t } - { \mathbf u } _ { t } \rangle - \gamma \displaystyle \sum _ { t = 2 } ^ { T } { \mathcal D } _ { \psi } ( { \mathbf x } _ { t } , { \mathbf x } _ { t - 1 } ) + \frac { \gamma L ^ { \prime } D ^ { 2 } } { 2 } - \gamma \displaystyle \sum _ { t = 2 } ^ { T } ( { \mathcal D } _ { \psi } ( { \mathbf u } _ { t - 1 } , { \mathbf x } _ { t - 1 } ) - { \mathcal D } _ { \psi } ( { \mathbf u } _ { t } , { \mathbf x } _ { t - 1 } ) ) } \end{array}
$$

$$
\leq \sum _ { t = 1 } ^ { T } \langle \mathbf { v } _ { t } , \mathbf { x } _ { t } - \mathbf { u } _ { t } \rangle - \gamma \sum _ { t = 2 } ^ { T } \mathcal { D } _ { \psi } ( \mathbf { x } _ { t } , \mathbf { x } _ { t - 1 } ) + \frac { \gamma L ^ { \prime } D ^ { 2 } } { 2 } + \gamma L ^ { \prime } D P _ { T } ,
$$

where the second equality follows from the Bregman three-point identity $\langle \nabla \psi ( { \mathbf { a } } ) - \nabla \psi ( { \mathbf { b } } ) , { \mathbf { a - c } } \rangle =$ $\mathcal { D } _ { \psi } ( \mathbf { c } , \mathbf { a } ) - \mathcal { D } _ { \psi } ( \mathbf { c } , \mathbf { b } ) + \mathcal { D } _ { \psi } ( \mathbf { a } , \mathbf { b } )$ for $\mathbf { a } , \mathbf { b } , \mathbf { c } \in \mathbb { R } ^ { d }$ , the third inequality uses smoothness, and the last inequality uses $| \mathcal { D } _ { \psi } ( \mathbf { u } , \mathbf { x } ) - \mathcal { D } _ { \psi } ( \mathbf { u } ^ { \prime } , \mathbf { x } ) | \leq L ^ { \prime } D \left\| \mathbf { u } - \mathbf { u } ^ { \prime } \right\|$ for all $\mathbf { u } , \mathbf { u } ^ { \prime } , \mathbf { x } \in \mathcal { X }$

## Appendix B. Omitted Proofs in Section 4

In this section, we provide the omitted proofs for Theorem 1 and Corollary 1.

## B.1. Proof for Theorem 1

Proof. For the first choice of ${ \mathbf { } } m _ { t }$ in Eq. (4.2), we have

$$
\begin{array} { r l } {  { \sum _ { t = 1 } ^ { T } \| \mathbf { g } _ { t } - m _ { t } \| ^ { 2 } = \sum _ { t = 1 } ^ { T } \| \nabla f _ { t } ( \mathbf { x } _ { t } ) - \nabla f _ { t - 1 } ( \mathbf { x } _ { t - 1 } ) + L ( \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } ) - L ( \mathbf { x } _ { t - 1 } - \mathbf { x } _ { t - 2 } ) \| ^ { 2 } } } \\ & { \le G ^ { 2 } + 3 \sum _ { t = 2 } ^ { T } \| \nabla f _ { t } ( \mathbf { x } _ { t } ) - \nabla f _ { t - 1 } ( \mathbf { x } _ { t - 1 } ) \| ^ { 2 } + 1 2 L ^ { 2 } \sum _ { t = 2 } ^ { T } \| \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } \| ^ { 2 } } \\ & { \le G ^ { 2 } + 6 V _ { T } + 1 8 L ^ { 2 } \sum _ { t = 2 } ^ { T } \| \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } \| ^ { 2 } . } \end{array}
$$

For the second choice of $\mathbf { \nabla } m _ { t }$ in Eq. (4.2), we have

$$
\begin{array} { r l } {  { \sum _ { t = 1 } ^ { T } \| \mathbf { g } _ { t } - \boldsymbol { m } _ { t } \| ^ { 2 } = \sum _ { t = 1 } ^ { T } \| \nabla f _ { t } ( \mathbf { x } _ { t } ) - \nabla f _ { t - 1 } ( \mathbf { x } _ { t - 1 } ) + L ( \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } ) \| ^ { 2 } } } \\ & { \le G ^ { 2 } + 2 \sum _ { t = 2 } ^ { T } \| \nabla f _ { t } ( \mathbf { x } _ { t } ) - \nabla f _ { t - 1 } ( \mathbf { x } _ { t - 1 } ) \| ^ { 2 } + 2 L ^ { 2 } \sum _ { t = 2 } ^ { T } \| \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } \| ^ { 2 } } \\ & { \le G ^ { 2 } + 4 V _ { T } + 6 L ^ { 2 } \sum _ { t = 2 } ^ { T } \| \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } \| ^ { 2 } . } \end{array}
$$

For both choices, plugging the bound of $\begin{array} { r } { \sum _ { t = 1 } ^ { T } \| \mathbf { g } _ { t } - m _ { t } \| ^ { 2 } } \end{array}$ into Eq. (4.1), we have

$$
\sum _ { t = 1 } ^ { T } \langle \mathbf { g } _ { t } , \mathbf { x } _ { t } - \mathbf { u } _ { t } \rangle \lesssim \sqrt { \left( 1 + P _ { T } \right) \left( 1 + P _ { T } + V _ { T } + L ^ { 2 } \sum _ { t = 2 } ^ { T } \left. \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } \right. ^ { 2 } \right) } .
$$

Plugging it back to Eq. (A.1) with the choice of $\gamma = L$ , we have

$$
\mathrm { D - R E G } _ { T } ( \mathbf { u } _ { 1 : T } ) \leq \sum _ { t = 1 } ^ { T } \langle \nabla f _ { t } ( \mathbf { x } _ { t } ) , \mathbf { x } _ { t } - \mathbf { u } _ { t } \rangle
$$

$$
\begin{array} { r l } & { \leq \displaystyle \sum _ { t = 1 } ^ { T } \langle \mathbf { g } _ { t } , \mathbf { x } _ { t } - \mathbf { u } _ { t } \rangle - \displaystyle \frac { L } { 2 } \sum _ { t = 2 } ^ { T } \| \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } \| ^ { 2 } + \frac { L D ^ { 2 } } { 2 } + L D P _ { T } , } \\ & { \leq \mathcal { O } ( \sqrt { \left( 1 + P _ { T } \right) \left( 1 + P _ { T } + V _ { T } \right) } ) + \mathcal { O } \left( \sqrt { \left( 1 + P _ { T } \right) \displaystyle \sum _ { t = 2 } ^ { T } \left\| \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } \right\| ^ { 2 } } \right) - \frac { L } { 2 } \displaystyle \sum _ { t = 2 } ^ { T } \| \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } \| ^ { 2 } } \\ & { \leq \mathcal { O } ( \sqrt { \left( 1 + P _ { T } \right) \left( 1 + P _ { T } + V _ { T } \right) } ) , } \end{array}
$$

where the last inequality follows from $2 { \sqrt { a b } } - a \leq b$ for $a , b \geq 0 .$

## B.2. Proof of Corollary 1

Proof. Setting $\lambda = 0$ in Theorem 9 of Zhao et al. (2024) shows that Sword.optimism satisfies Eq. (4.1). Applying Theorem 1 then completes the proof. ■

## Appendix C. Omitted Details in Section 5

In this section, we provide the omitted details of the base learners in Algorithm 2 in Appendix C.1 and prove Theorem 2 in Appendix C.2.

## C.1. Base Learners of Algorithm 2

Following the conversion stated after Definition 2, we use the effective exp-concavity coefficient $\alpha = \textstyle { \frac { 1 } { 2 } } \operatorname* { m i n } \{ 1 / ( 4 G D ) , \beta \}$ , where $\exp ( - \beta f _ { t } ( \cdot ) )$ is concave for every $t \in [ T ]$ . In particular, $\alpha G D \leq$ $1 / 8$ . For the analysis, it suffices to consider $\lambda \in [ 1 / T , 1 ]$ and $\alpha \in [ 1 / T$ , min $\{ 1 , 1 / ( 8 G D ) \}$ ]. If either parameter is below $1 / T$ , the corresponding logarithmic regret bound does not improve on the trivial $\mathcal { O } ( T )$ bound, while the general convex regret bound remains applicable. Conversely, if $\lambda , \alpha > 1$ , they can be treated as $\lambda = 1$ and $\alpha = 1$ , which only worsens the regret by an ignorable constant factor.

Then, we construct the candidate pools $\mathcal { H } ^ { \mathrm { s c } }$ and $\mathcal { H } ^ { \mathrm { e x p } }$ for the unknown parameters λ and α respectively, using a geometric grid:

$$
{ \mathcal { H } } ^ { \mathrm { e x p } } = { \mathcal { H } } ^ { \mathrm { s c } } \triangleq \left\{ { \frac { 1 } { T } } , { \frac { 2 } { T } } , { \frac { 2 ^ { 2 } } { T } } , \cdots , { \frac { 2 ^ { n - 1 } } { T } } \right\} ,\tag{C.1}
$$

where $n = \lceil \log _ { 2 } T \rceil + 1 = \mathcal { O } ( \log T )$ is the number of candidates in each pool. One base learner is instantiated for each candidate $\lambda _ { i }$ or $\alpha _ { i }$ , with the final base learner, indexed by N, assigned to convex losses. Thus, there are $N = 2 n + 1 = \mathcal { O } ( \log T )$ base learners in total. We define the best base learner as the one with the right guess of the curvature type and the closest guess of the curvature coefficient. For example, suppose the online functions are α-exp-concave (while this is unknown to the online learner). The best base learner, indexed by $i ^ { \star }$ , uses a candidate parameter $\alpha _ { i ^ { \star } }$ satisfying $\alpha _ { i ^ { \star } } \leq \alpha \leq 2 \alpha _ { i ^ { \star } }$

The base learners below use the corrected gradient $\mathbf { g } _ { t }$ from Algorithm 2, with $\| \mathbf { g } _ { t } \| \leq H$ as in the main text. Under Assumption 1, $\| \mathbf { g } _ { t } \| \leq G + \gamma D$ , which equals 2G when $\gamma = G / D$ . Let $I ^ { \exp } \triangleq \{ 1 , \dots , n \}$ and $I ^ { \mathrm { s c } } \triangleq \{ n + 1 , \dots , 2 n \}$ be the two curved index sets. Initialize $\mathbf { x } _ { 0 } = \mathbf { x } _ { 1 }$ $\mathbf { g } _ { 0 } = \mathbf { 0 }$ , and $\widehat { \mathbf { x } } _ { 1 , i } = \mathbf { x } _ { 0 , i } = \mathbf { x } _ { 1 }$ for every $i \in [ N ]$ , with $h _ { 0 , i } \equiv 0$ . We write $h _ { t , i }$ for the surrogate assigned to base $i ;$ in particular, $h _ { t , N } = h _ { t , N } ^ { \mathrm { c } }$ for the convex base.

For α-exp-concave functions, we define a group of base learners $\{ B _ { i } ^ { \exp } \} _ { i \in I ^ { \exp } }$ , where the i-th base learner runs the algorithm below:

$$
\begin{array} { r l } & { \quad \mathbf { x } _ { t , i } = \underset { \mathbf { x } \in \mathcal { X } } { \arg \operatorname* { m i n } } \left\{ \langle \nabla h _ { t - 1 , i } ^ { \exp } ( \mathbf { x } _ { t - 1 , i } ) , \mathbf { x } \rangle + \mathcal { D } _ { \psi _ { t , i } } ( \mathbf { x } , \widehat { \mathbf { x } } _ { t , i } ) \right\} , } \\ & { \widehat { \mathbf { x } } _ { t + 1 , i } = \underset { \mathbf { x } \in \mathcal { X } } { \arg \operatorname* { m i n } } \left\{ \langle \nabla h _ { t , i } ^ { \exp } ( \mathbf { x } _ { t , i } ) , \mathbf { x } \rangle + \mathcal { D } _ { \psi _ { t , i } } ( \mathbf { x } , \widehat { \mathbf { x } } _ { t , i } ) \right\} , } \end{array}\tag{C.2}
$$

where $\begin{array} { r } { \psi _ { t , i } ( { \bf x } ) \triangleq \frac { 1 } { 2 } { \bf x } ^ { \top } U _ { t , i } { \bf x } , U _ { t , i } = \left( 1 + \frac { \alpha _ { i } H ^ { 2 } } { 2 } \right) I _ { d } + \frac { \alpha _ { i } } { 8 } \sum _ { s = 1 } ^ { t - 1 } \nabla h _ { s , i } ^ { \mathrm { e x p } } ( { \bf x } _ { s , i } ) \nabla h _ { s , i } ^ { \mathrm { e x p } } ( { \bf x } _ { s , i } ) ^ { \top } , \alpha _ { i } = \frac { \alpha _ { i } - 1 } { 2 } \nabla h _ { s , i } ^ { \mathrm { e x p } } ( { \bf x } _ { s , i } ) . } \end{array}$ is the corresponding element in $\mathcal { H } ^ { \mathrm { e x p } }$ , and ${ { \dot { h } } _ { t , i } ^ { \mathrm { { e x p } } } } ( \cdot )$ is a surrogate loss function for $\boldsymbol { B } _ { i } ^ { \mathrm { e x p } }$ , defined as

$$
h _ { t , i } ^ { \mathrm { e x p } } ( \mathbf { x } ) \triangleq \langle \mathbf { g } _ { t } , \mathbf { x } \rangle + \frac { \alpha _ { i } } { 4 } \langle \mathbf { g } _ { t } , \mathbf { x } - \mathbf { x } _ { t } \rangle ^ { 2 } .
$$

Similarly, for λ-strongly convex functions, we define a group of base learners $\{ B _ { i } ^ { \mathrm { s c } } \} _ { i \in I ^ { \mathrm { s c } } }$ , where the i-th base learner runs the algorithm below:

$$
\begin{array} { r } { \mathbf { x } _ { t , i } = \Pi _ { \mathcal { X } } \left[ \widehat { \mathbf { x } } _ { t , i } - \eta _ { t , i } \nabla h _ { t - 1 , i } ^ { \mathrm { s c } } ( \mathbf { x } _ { t - 1 , i } ) \right] , \quad \widehat { \mathbf { x } } _ { t + 1 , i } = \Pi _ { \mathcal { X } } \left[ \widehat { \mathbf { x } } _ { t , i } - \eta _ { t , i } \nabla h _ { t , i } ^ { \mathrm { s c } } ( \mathbf { x } _ { t , i } ) \right] , } \end{array}\tag{C.3}
$$

where $\eta _ { t , i } = 4 / ( 2 + \lambda _ { i } t ) , \lambda _ { i }$ is the corresponding element in $\mathcal { H } ^ { \mathrm { s c } }$ , and $h _ { t , i } ^ { \mathrm { s c } } ( \cdot )$ is a surrogate loss function for $\boldsymbol { B } _ { i } ^ { \mathrm { s c } }$ , defined as

$$
h _ { t , i } ^ { \mathrm { s c } } ( \mathbf { x } ) \triangleq \langle \mathbf { g } _ { t } , \mathbf { x } \rangle + \frac { \lambda _ { i } } { 4 } \left. \mathbf { x } - \mathbf { x } _ { t } \right. ^ { 2 } .
$$

For convex functions, we only have to define one base learner $B ^ { \mathrm { c } }$ , which updates as

$$
\begin{array} { r } { { \bf x } _ { t , N } = \Pi _ { \mathcal { X } } \left[ \widehat { \bf x } _ { t , N } - \eta _ { t , N } { \bf g } _ { t - 1 } \right] , \quad \widehat { \bf x } _ { t + 1 , N } = \Pi _ { \mathcal { X } } \left[ \widehat { \bf x } _ { t , N } - \eta _ { t , N } { \bf g } _ { t } \right] , } \end{array}\tag{C.4}
$$

where $\eta _ { t , N } = \operatorname* { m i n } \left\{ D / \sqrt { 1 + \textstyle \sum _ { s = 2 } ^ { t - 1 } \| \mathbf { g } _ { s } - \mathbf { g } _ { s - 1 } \| ^ { 2 } } , 1 \right\}$ . Finally, we summarize the configurations of base learners. Specifically, we deploy

$$
\{ B _ { i } \} _ { i \in [ N ] } \triangleq \{ \mathcal { B } _ { i } ^ { \mathrm { e x p } } \} _ { i \in I ^ { \mathrm { e x p } } } \cup \{ \mathcal { B } _ { i } ^ { \mathrm { s c } } \} _ { i \in I ^ { \mathrm { s c } } } \cup \{ \mathcal { B } ^ { \mathrm { c } } \}
$$

as the total set of base learners.

## C.2. Proof of Theorem 2

Proof. Here, we consider any comparator $\mathbf { u _ { \lambda } } \in \mathcal { X }$ . We first decompose the regret for the three curvature types, then analyze the meta and base regret separately, and finally combine their bounds. As in Algorithm 2, we use the corrected gradient $\mathbf { g } _ { t } = \nabla f _ { t } ( \mathbf { x } _ { t } ) + \gamma ( \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } )$ , with $\gamma = G / D$ and $\| \mathbf { g } _ { t } \| \leq H = 2 G$ . For simplicity, we write

$$
S _ { T } \triangleq \sum _ { t = 2 } ^ { T } \Vert \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } \Vert ^ { 2 } , \qquad \widetilde { V } _ { T } \triangleq \sum _ { t = 2 } ^ { T } \Vert \mathbf { g } _ { t } - \mathbf { g } _ { t - 1 } \Vert ^ { 2 } , \qquad K _ { \gamma } \triangleq 3 L ^ { 2 } + 1 2 \gamma ^ { 2 } .
$$

Here $\boldsymbol { \widetilde { V } } _ { T }$ is the variation of the corrected gradients; $\hat { V } _ { T }$ in Section 2 denotes the variation of the original loss gradients. Under Assumption 2, the same decomposition as in Eq. (2.1) gives

$$
\begin{array} { l } { \displaystyle \widetilde { V } _ { T } \leq 3 \sum _ { t = 2 } ^ { T } \| \nabla f _ { t } ( \mathbf { x } _ { t } ) - \nabla f _ { t - 1 } ( \mathbf { x } _ { t } ) \| ^ { 2 } + 3 \displaystyle \sum _ { t = 2 } ^ { T } \| \nabla f _ { t - 1 } ( \mathbf { x } _ { t } ) - \nabla f _ { t - 1 } ( \mathbf { x } _ { t - 1 } ) \| ^ { 2 } } \\ { \displaystyle \qquad + 3 \gamma ^ { 2 } \displaystyle \sum _ { t = 2 } ^ { T } \| \mathbf { x } _ { t } - 2 \mathbf { x } _ { t - 1 } + \mathbf { x } _ { t - 2 } \| ^ { 2 } } \\ { \leq 3 V _ { T } + 3 L ^ { 2 } \displaystyle \sum _ { t = 2 } ^ { T } \| \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } \| ^ { 2 } + 6 \gamma ^ { 2 } \displaystyle \sum _ { t = 2 } ^ { T } \Big ( \| \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } \| ^ { 2 } + \| \mathbf { x } _ { t - 1 } - \mathbf { x } _ { t - 2 } \| ^ { 2 } \Big ) } \\ { \leq 3 V _ { T } + ( 3 L ^ { 2 } + 1 2 \gamma ^ { 2 } ) S _ { T } . } \end{array}\tag{C.5}
$$

Regret Decomposition. For each base index $i \in [ N ]$ , put $r _ { t , i } = \langle \mathbf { g } _ { t } , \mathbf { x } _ { t } - \mathbf { x } _ { t , i } \rangle$ . The surrogates $h _ { t , i }$ are those explicitly defined in Appendix C.1.

For λ-strongly convex functions, choose an index $i ^ { \star } \in I ^ { \mathrm { s c } }$ whose guess $\lambda _ { i ^ { \star } }$ satisfies $\lambda _ { i ^ { \star } } \leq \lambda \leq 2 \lambda _ { i ^ { \star } }$ Strong convexity and Eq. (A.1) imply

$$
\begin{array} { r l } & { \displaystyle \mathrm { R E G } _ { T } \leq \sum _ { t = 1 } ^ { T } \left( \left. \mathbf { g } _ { t } , \mathbf { x } _ { t } - \mathbf { u } \right. - \frac { \lambda } { 2 } \left. \mathbf { x } _ { t } - \mathbf { u } \right. ^ { 2 } \right) - \frac { \gamma } { 2 } S _ { T } + \frac { \gamma D ^ { 2 } } { 2 } } \\ & { \leq \underbrace { \displaystyle \sum _ { t = 1 } ^ { T } \left( r _ { t , i ^ { \star } } - \frac { \lambda _ { i ^ { \star } } } { 4 } \left. \mathbf { x } _ { t } - \mathbf { x } _ { t , i ^ { \star } } \right. ^ { 2 } \right) } _ { \mathrm { M E F A - R E G } ^ { \mathrm { s c } } } + \underbrace { \sum _ { t = 1 } ^ { T } \bigl ( h _ { t , i ^ { \star } } ^ { \mathrm { s c } } ( \mathbf { x } _ { t , i ^ { \star } } ) - h _ { t , i ^ { \star } } ^ { \mathrm { s c } } ( \mathbf { u } ) \bigr ) } _ { \mathrm { B A S E - R E G } ^ { \mathrm { s c } } } - \frac { \gamma } { 2 } S _ { T } + \frac { \gamma D ^ { 2 } } { 2 } . } \end{array}\tag{C.6}
$$

Here $h _ { t , i ^ { \star } } ^ { \mathrm { s c } } ( \mathbf { x } ) = \langle \mathbf { g } _ { t } , \mathbf { x } \rangle + ( \lambda _ { i ^ { \star } } / 4 ) \| \mathbf { x } - \mathbf { x } _ { t } \| ^ { 2 } .$

For α-exp-concave functions, choose $i ^ { \star } \in I ^ { \mathsf { e x p } }$ with $\alpha _ { i ^ { \star } } \leq \alpha \leq 2 \alpha _ { i ^ { \star } }$ and transfer the directional curvature from the original gradient $\nabla f _ { t } ( \mathbf { x } _ { t } )$ to the corrected gradient $\mathbf { g } _ { t }$ . The inequality $( q - b ) ^ { 2 } \geq$ $q ^ { 2 } / 2 - b ^ { 2 }$ gives

$$
- \frac { \alpha } { 2 } \sum _ { t = 1 } ^ { T } \langle \nabla f _ { t } ( \mathbf { x } _ { t } ) , \mathbf { x } _ { t } - \mathbf { u } \rangle ^ { 2 } \leq - \frac { \alpha } { 4 } \sum _ { t = 1 } ^ { T } \langle \mathbf { g } _ { t } , \mathbf { x } _ { t } - \mathbf { u } \rangle ^ { 2 } + \frac { \alpha \gamma ^ { 2 } D ^ { 2 } } { 2 } S _ { T } .\tag{C.7}
$$

Since $\alpha \gamma D ^ { 2 } = \alpha G D \leq 1 / 8$ , combining this inequality with Eq. (A.1) gives

$$
\begin{array} { r l } & { \displaystyle \mathrm { R E G } _ { T } \leq \sum _ { t = 1 } ^ { T } \Big ( \langle \mathbf { g } _ { t } , \mathbf { x } _ { t } - \mathbf { u } \rangle - \frac { \alpha } { 4 } \langle \mathbf { g } _ { t } , \mathbf { x } _ { t } - \mathbf { u } \rangle ^ { 2 } \Big ) - \frac { \gamma } { 4 } S _ { T } + \frac { \gamma D ^ { 2 } } { 2 } } \\ & { \quad \leq \underbrace { \sum _ { t = 1 } ^ { T } \Big ( r _ { t , i ^ { \star } } - \frac { \alpha _ { i ^ { \star } } } { 4 } r _ { t , i ^ { \star } } ^ { 2 } \Big ) } _ { \mathrm { M E F A - R E G } ^ { \mathrm { e x p } } } + \underbrace { \sum _ { t = 1 } ^ { T } \big ( h _ { t , i ^ { \star } } ^ { \mathrm { e x p } } ( \mathbf { x } _ { t , i ^ { \star } } ) - h _ { t , i ^ { \star } } ^ { \mathrm { e x p } } ( \mathbf { u } ) \big ) } _ { \mathrm { B A S E - R E G } ^ { \mathrm { e x p } } } - \frac { \gamma } { 4 } S _ { T } + \frac { \gamma D ^ { 2 } } { 2 } , } \end{array}\tag{C.8}
$$

where $h _ { t , i ^ { \star } } ^ { \mathrm { e x p } } ( \mathbf { x } ) \triangleq \langle \mathbf { g } _ { t } , \mathbf { x } \rangle + ( \alpha _ { i ^ { \star } } / 4 ) \langle \mathbf { g } _ { t } , \mathbf { x } - \mathbf { x } _ { t } \rangle ^ { 2 }$

For convex functions, using the convex base indexed by N, convexity and Eq. (A.1) give

$$
\begin{array} { l } { \displaystyle \mathsf { R E G } _ { T } \leq \sum _ { t = 1 } ^ { T } \langle \mathbf { g } _ { t } , \mathbf { x } _ { t } - \mathbf { u } \rangle - \frac { \gamma } { 2 } S _ { T } + \frac { \gamma D ^ { 2 } } { 2 } } \\ { \displaystyle = \underbrace { \sum _ { t = 1 } ^ { T } r _ { t , N } } _ { \mathrm { M E T A - R E G } ^ { \mathrm { c } } } + \underbrace { \sum _ { t = 1 } ^ { T } \bigl ( h _ { t , N } ^ { \mathrm { c } } ( \mathbf { x } _ { t , N } ) - h _ { t , N } ^ { \mathrm { c } } ( \mathbf { u } ) \bigr ) } _ { \mathrm { B A S E - R E G ^ { c } } } - \frac { \gamma } { 2 } S _ { T } + \frac { \gamma D ^ { 2 } } { 2 } . } \end{array}\tag{C.9}
$$

Meta Regret Analysis. We adopt Optimistic-Adapt-ML-Prod (Wei et al., 2016) as the meta learner and recall its optimistic second-order regret guarantee below.

Lemma 1 (Theorem 3.4 of Wei et al. (2016)). Denote by ${ \pmb p } _ { t } \in \Delta _ { N }$ the algorithm’s weights, $\ell _ { t } \in$ $[ 0 , 1 ] ^ { N }$ the loss vector, and $\tilde { m } _ { t , i } \in [ - 1 , 1 ]$ the optimism available before observing $\ell _ { t } .$ . The regret of Optimistic-Adapt-ML-Prod with respect to any base learner $i \in [ N ]$ satisfies

$$
\sum _ { t = 1 } ^ { T } \widetilde { r } _ { t , i } \leq C _ { 0 } \sqrt { 1 + \sum _ { t = 1 } ^ { T } ( \widetilde { r } _ { t , i } - \widetilde { m } _ { t , i } ) ^ { 2 } } + C _ { 2 } ,
$$

$\begin{array} { r l } & { w h e r e ~ \widetilde { r } _ { t , i } = \langle p _ { t } , \ell _ { t } \rangle - \ell _ { t , i } , \ C _ { 0 } = \sqrt { \log { N } } + \log ( 1 + \frac { N } { \epsilon } ( 1 + \log ( T + 1 ) ) ) / \sqrt { \log { N } } , } \\ & { \frac { 1 } { 4 } ( \log { N } + \log ( 1 + \frac { N } { \epsilon } ( 1 + \log ( T + 1 ) ) ) ) + 2 \sqrt { \log { N } } + 1 6 \log { N } . } \end{array}$ , and $C _ { 2 } =$

To apply Lemma 1, we define $\begin{array} { r } { \ell _ { t , i } \triangleq \frac { 1 } { 2 } + \frac { \left. \mathbf { g } _ { t } , \mathbf { x } _ { t , i } - \mathbf { x } _ { 1 } \right. } { 2 H D } } \end{array}$ and rewrite Eq. (5.2) and Eq. (5.4) in the following normalized form:

$$
\left. \begin{array} { l } { \displaystyle \widetilde { r } _ { t , i } \triangleq \frac { r _ { t , i } } { 2 H D } = \langle \pmb { p } _ { t } , \pmb { \ell } _ { t } \rangle - \ell _ { t , i } , \quad \widetilde { m } _ { t , i } \triangleq \frac { m _ { t , i } } { 2 H D } , } \\ { \displaystyle \widetilde { \varepsilon } _ { t , i } = 2 H D \varepsilon _ { t , i } = \operatorname* { m i n } \left\{ \frac { 1 } { 4 } , \sqrt { \frac { \ln N } { 1 + \sum _ { s = 1 } ^ { t } ( \widetilde { r } _ { s , i } - \widetilde { m } _ { s , i } ) ^ { 2 } } } \right\} . } \end{array} \right.
$$

This normalization leaves the meta learner’s weight updates and predictions unchanged. Since $\| \mathbf { g } _ { t } \| \leq H$ and $\| \mathbf { x } _ { t , i } - \mathbf { x } _ { 1 } \| \leq D$ , we have $\ell _ { t , i } \in [ 0 , 1 ]$ and $\widetilde { m } _ { t , i } = m _ { t , i } / ( 2 H D ) \in [ - 1 / 2 , 1 / 2 ]$ Thus Lemma 1 yields

$$
\sum _ { t = 1 } ^ { T } r _ { t , i } \leq C _ { 0 } \sqrt { \sum _ { t = 1 } ^ { T } ( r _ { t , i } - m _ { t , i } ) ^ { 2 } } + C _ { 1 } H D ,\tag{C.10}
$$

where $C _ { 1 } = 2 C _ { 0 } + 2 C _ { 2 } = \mathcal { O } ( \log N )$

For λ-strongly convex functions, according to Eq. (5.3), we have $m _ { t , i ^ { \star } } = 0$ for the selected strongly convex base learner $i ^ { \star }$ . By Lemma 1, the meta regret in Eq. (C.6) can be bounded as

$$
\begin{array} { r l r } {  { \mathbf { M } \mathrm { E T A - R E G } ^ { \mathrm { s c } } \le C _ { 0 } H \sqrt { \displaystyle \sum _ { t = 1 } ^ { T } \| \mathbf { x } _ { t } - \mathbf { x } _ { t , i ^ { \star } } \| ^ { 2 } } + C _ { 1 } H D - \frac { \lambda _ { i ^ { \star } } } { 4 } \displaystyle \sum _ { t = 1 } ^ { T } \| \mathbf { x } _ { t } - \mathbf { x } _ { t , i ^ { \star } } \| ^ { 2 } } } \\ & { } & { \leq C _ { 1 } H D + \frac { 2 C _ { 0 } ^ { 2 } H ^ { 2 } } { \lambda _ { i ^ { \star } } } - \frac { \lambda _ { i ^ { \star } } } { 8 } \displaystyle \sum _ { t = 1 } ^ { T } \| \mathbf { x } _ { t } - \mathbf { x } _ { t , i ^ { \star } } \| ^ { 2 } . } \end{array}\tag{C.11}
$$

Here the first inequality uses $r _ { t , i ^ { \star } } ^ { 2 } \leq H ^ { 2 } \left\| \mathbf { x } _ { t } - \mathbf { x } _ { t , i ^ { \star } } \right\| ^ { 2 }$ , and the second uses the AM–GM inequality. For exp-concave functions, we similarly have $m _ { t , i ^ { \star } } = 0$ for the selected exp-concave base learner $i ^ { \star }$ . By Lemma 1, the meta regret in Eq. (C.8) satisfies

$$
\begin{array} { r l } & { \displaystyle \mathbf { M } \mathrm { E T A - R E G } ^ { \mathrm { e x p } } \leq C _ { 0 } \sqrt { \displaystyle \sum _ { t = 1 } ^ { T } r _ { t , i ^ { \star } } ^ { 2 } } + C _ { 1 } H D - \frac { \alpha _ { i ^ { \star } } } { 4 } \displaystyle \sum _ { t = 1 } ^ { T } r _ { t , i ^ { \star } } ^ { 2 } } \\ & { \qquad \leq C _ { 1 } H D + \frac { 2 C _ { 0 } ^ { 2 } } { \alpha _ { i ^ { \star } } } - \frac { \alpha _ { i ^ { \star } } } { 8 } \displaystyle \sum _ { t = 1 } ^ { T } r _ { t , i ^ { \star } } ^ { 2 } . } \end{array}\tag{C.12}
$$

The last step again follows from the AM–GM inequality. The remaining negative terms in these two bounds will cancel the surrogate-dependent terms in the base regret.

For convex functions, Eq. (5.3) gives $m _ { t , N } = \langle \mathbf { g } _ { t - 1 } , \mathbf { x } _ { t - 1 } - \mathbf { x } _ { t , N } \rangle$ . Consequently, the prediction error decomposes as

$$
r _ { t , N } - m _ { t , N } = \langle \mathbf { g } _ { t } - \mathbf { g } _ { t - 1 } , \mathbf { x } _ { t } - \mathbf { x } _ { t , N } \rangle + \langle \mathbf { g } _ { t - 1 } , \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } \rangle .
$$

Using $\| \mathbf { x } _ { t } - \mathbf { x } _ { t , N } \| \leq D , \| \mathbf { g } _ { t - 1 } \| \leq H$ , and $\mathbf { g } _ { 0 } = \mathbf { 0 }$ , we obtain

$$
\sum _ { t = 1 } ^ { T } ( r _ { t , N } - m _ { t , N } ) ^ { 2 } \leq H ^ { 2 } D ^ { 2 } + 2 D ^ { 2 } \widetilde { V } _ { T } + 2 H ^ { 2 } S _ { T } .\tag{C.13}
$$

By Lemma 1, the meta regret in Eq. (C.9) is therefore bounded by

$$
\begin{array} { r l } & { \mathbf { M } \mathrm { E T A - R E G } ^ { \mathrm { c } } \leq C _ { 0 } \sqrt { 2 D ^ { 2 } \widetilde { V } _ { T } + 2 H ^ { 2 } S _ { T } + H ^ { 2 } D ^ { 2 } } + C _ { 1 } H D } \\ & { \qquad \leq \sqrt { 2 } C _ { 0 } D \sqrt { \widetilde { V } _ { T } } + \sqrt { 2 } C _ { 0 } H \sqrt { S _ { T } } + ( C _ { 0 } + C _ { 1 } ) H D . } \end{array}\tag{C.14}
$$

The additional $\sqrt { S _ { T } }$ term is absorbed by the negative movement term in Proposition 1.

Base Regret Analysis. For λ-strongly convex functions, when using the update rule Eq. (C.3), following the OOMD analysis used for Eq. (C.9) of Yan et al. (2024), the base regret can be bounded as

$$
\mathbb { B } \mathrm { A s E - R E G } ^ { \mathrm { s c } } \leq \mathcal { O } \left( \frac { 1 } { \lambda _ { i ^ { \star } } } \log \left( 1 + \lambda _ { i ^ { \star } } \sum _ { t = 2 } ^ { T } \| \nabla h _ { t , i ^ { \star } } ( \mathbf { x } _ { t , i ^ { \star } } ) - \nabla h _ { t - 1 , i ^ { \star } } ( \mathbf { x } _ { t - 1 , i ^ { \star } } ) \| ^ { 2 } \right) \right) .\tag{C.15}
$$

Next, we analyze the empirical gradient variation of the surrogate function $h _ { t , i ^ { \star } } ^ { \mathrm { s c } } ( \cdot )$

$$
\sum _ { t = 2 } ^ { T } \left. \nabla h _ { t , i ^ { * } } ( \mathbf { x } _ { t , i ^ { * } } ) - \nabla h _ { t - 1 , i ^ { * } } ( \mathbf { x } _ { t - 1 , i ^ { * } } ) \right. ^ { 2 } = \sum _ { t = 2 } ^ { T } \left. \mathbf { g } _ { t } - \mathbf { g } _ { t - 1 } + \frac { \lambda _ { i ^ { * } } } { 2 } \left( \mathbf { x } _ { t , i ^ { * } } - \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 , i ^ { * } } + \mathbf { x } _ { t - 1 } \right) \right. ^ { 2 }
$$

$$
\leq 3 \widetilde { V } _ { T } + \frac { 3 \lambda _ { i ^ { * } } ^ { 2 } } { 2 } \sum _ { t = 1 } ^ { T } \| { \mathbf x } _ { t } - { \mathbf x } _ { t , i ^ { * } } \| ^ { 2 } \leq 9 V _ { T } + ( 9 L ^ { 2 } + 3 6 \gamma ^ { 2 } ) S _ { T } + \frac { 3 \lambda _ { i ^ { * } } ^ { 2 } } { 2 } \sum _ { t = 1 } ^ { T } \| { \mathbf x } _ { t } - { \mathbf x } _ { t , i ^ { * } } \| ^ { 2 } ,\tag{C.16}
$$

where the last inequality uses Eq. (C.5). Plugging the surrogate’s empirical gradient variation into Eq. (C.15), we obtain

$$
\mathbf { B A S E - R E G ^ { s c } } \leq \mathcal { O } \left( \frac { 1 } { \lambda _ { i ^ { \star } } } \log \left( 1 + \lambda _ { i ^ { \star } } \left( V _ { T } + S _ { T } + \lambda _ { i ^ { \star } } ^ { 2 } \sum _ { t = 1 } ^ { T } \| \mathbf { x } _ { t } - \mathbf { x } _ { t , i ^ { \star } } \| ^ { 2 } \right) \right) \right) .\tag{C.17}
$$

For α-exp-concave functions, when using the update rule Eq. (C.2), following the OOMD analysis used for Eq. (C.7) of Yan et al. (2024), the base regret can be bounded as

$$
\mathrm { B A S E - R E G } ^ { \mathrm { e x p } } \leq \mathcal { O } \left( \frac { d } { \alpha _ { i ^ { * } } } \log \left( 1 + \frac { \alpha _ { i ^ { * } } } { d } \sum _ { t = 2 } ^ { T } \| \nabla h _ { t , i ^ { * } } ( \mathbf { x } _ { t , i ^ { * } } ) - \nabla h _ { t - 1 , i ^ { * } } ( \mathbf { x } _ { t - 1 , i ^ { * } } ) \| ^ { 2 } \right) \right) + \mathcal { O } ( 1 ) .\tag{C.18}
$$

Next, we analyze the empirical gradient variation of the surrogate function $h _ { t , i ^ { \star } } ^ { \mathrm { e x p } } ( \cdot )$

$$
\sum _ { t = 2 } ^ { T } \left. \nabla h _ { t , i ^ { \star } } ( \mathbf { x } _ { t , i ^ { \star } } ) - \nabla h _ { t - 1 , i ^ { \star } } ( \mathbf { x } _ { t - 1 , i ^ { \star } } ) \right. ^ { 2 } = \sum _ { t = 2 } ^ { T } \left. \mathbf { g } _ { t } - \mathbf { g } _ { t - 1 } - \frac { \alpha _ { i ^ { \star } } } { 2 } \left( \mathbf { g } _ { t } r _ { t , i ^ { \star } } - \mathbf { g } _ { t - 1 } r _ { t - 1 , i ^ { \star } } \right) \right. ^ { 2 }
$$

$$
\leq 3 \widetilde { V } _ { T } + \frac { 3 \alpha _ { i ^ { \star } } ^ { 2 } H ^ { 2 } } { 2 } \sum _ { t = 1 } ^ { T } r _ { t , i ^ { \star } } ^ { 2 } \leq 9 V _ { T } + ( 9 L ^ { 2 } + 3 6 \gamma ^ { 2 } ) S _ { T } + \frac { 3 \alpha _ { i ^ { \star } } ^ { 2 } H ^ { 2 } } { 2 } \sum _ { t = 1 } ^ { T } r _ { t , i ^ { \star } } ^ { 2 } ,\tag{C.19}
$$

where the equality uses $\nabla h _ { t , i ^ { \star } } ( \mathbf { x } _ { t , i ^ { \star } } ) = \mathbf { g } _ { t } ( 1 - \alpha _ { i ^ { \star } } r _ { t , i ^ { \star } } / 2 )$ , the first inequality uses the squared triangle inequality, a shift of the summation index, and $\| \mathbf { g } _ { t } \| \leq H$ , and the last uses Eq. (C.5). Plugging the surrogate’s empirical gradient variation into Eq. (C.18), we obtain

$$
{ \mathrm { B A S E - R E G } } ^ { \mathrm { e x p } } \leq \mathcal { O } \left( \frac { d } { \alpha _ { i ^ { \star } } } \log \left( 1 + \frac { \alpha _ { i ^ { \star } } } { d } \left( V _ { T } + S _ { T } + \alpha _ { i ^ { \star } } ^ { 2 } \sum _ { t = 1 } ^ { T } r _ { t , i ^ { \star } } ^ { 2 } \right) \right) \right) .\tag{C.20}
$$

For convex functions, when using the update rule Eq. (C.4), following the OOMD analysis in the proof of Lemma 10 of Yan et al. (2023), the base regret can be bounded as

$$
\mathrm { B A S E - R E G } ^ { \mathrm { c } } \leq \mathcal { O } \left( \sqrt { 1 + \sum _ { t = 2 } ^ { T } \| \nabla h _ { t , N } ( \mathbf x _ { t , N } ) - \nabla h _ { t - 1 , N } ( \mathbf x _ { t - 1 , N } ) \| ^ { 2 } } \right) .\tag{C.21}
$$

Next, we analyze the empirical gradient variation of the surrogate function $h _ { t , N } ^ { \mathrm { c } } ( \cdot )$

$$
\sum _ { t = 2 } ^ { T } \| \nabla h _ { t , N } ( \mathbf { x } _ { t , N } ) - \nabla h _ { t - 1 , N } ( \mathbf { x } _ { t - 1 , N } ) \| ^ { 2 } = \widetilde { V } _ { T } \leq 3 V _ { T } + ( 3 L ^ { 2 } + 1 2 \gamma ^ { 2 } ) S _ { T } ,\tag{C.22}
$$

where the equality uses $\nabla h _ { t , N } ( \mathbf { x } ) = \mathbf { g } _ { t }$ and the expression for $\boldsymbol { \widetilde { V } } _ { T }$ , and the inequality uses Eq. (C.5). Plugging the surrogate’s empirical gradient variation into Eq. (C.21), we obtain

$$
\mathbf { B A S E - R E G } ^ { \mathbf { c } } \leq \mathcal { O } \left( \sqrt { 1 + V _ { T } + S _ { T } } \right) \leq \mathcal { O } \left( \sqrt { 1 + V _ { T } } + \sqrt { S _ { T } } \right) .\tag{C.23}
$$

Overall Regret Analysis. For λ-strongly convex functions, combining Eq. (C.6), Eq. (C.11), and Eq. (C.17), we obtain

$$
\begin{array} { r l } & { \mathbf { R E G } _ { T } \leq \mathcal { O } \left( \displaystyle \frac { 1 } { \lambda _ { i ^ { \star } } } \log \left( 2 + V _ { T } + S _ { T } + \lambda _ { i ^ { \star } } ^ { 2 } \displaystyle \sum _ { t = 1 } ^ { T } \| \mathbf { x } _ { t } - \mathbf { x } _ { t , i ^ { \star } } \| ^ { 2 } \right) \right) - \frac { \lambda _ { i ^ { \star } } } { 8 } \displaystyle \sum _ { t = 1 } ^ { T } \| \mathbf { x } _ { t } - \mathbf { x } _ { t , i ^ { \star } } \| ^ { 2 } - \frac { \gamma } { 2 } S _ { T } } \\ & { \qquad \leq \mathcal { O } \left( \displaystyle \frac { 1 } { \lambda } \log V _ { T } \right) , } \end{array}
$$

where the last inequality uses log $( 1 + x ) \leq x$ and $\lambda _ { i ^ { \star } } \leq \lambda \leq 2 \lambda _ { i ^ { \star } }$

For α-exp-concave functions, combining Eq. (C.8), Eq. (C.12), and Eq. (C.20), we obtain

$$
\begin{array} { r l } & { { \small \mathrm { R E G } _ { T } \leq \mathcal O \left( \frac { d } { \alpha _ { i ^ { * } } } \log \left( 2 + \frac { \alpha _ { i ^ { * } } } { d } \left( V _ { T } + S _ { T } + \alpha _ { i ^ { * } } ^ { 2 } \sum _ { t = 1 } ^ { T } r _ { t , i ^ { * } } ^ { 2 } \right) \right) \right) - \frac { \alpha _ { i ^ { * } } } { 8 } \sum _ { t = 1 } ^ { T } r _ { t , i ^ { * } } ^ { 2 } - \frac { \gamma } { 4 } S _ { T } } } \\ & { \quad \quad \quad \leq \mathcal O \left( \frac { d } { \alpha } \log V _ { T } \right) , } \end{array}
$$

where the last inequality uses lo $\begin{array} { r } { \mathfrak { g } ( 1 + x ) \le x } \end{array}$ and $\alpha _ { i ^ { \star } } \leq \alpha \leq 2 \alpha _ { i ^ { \star } }$

For convex functions, Eq. (C.14) and Eq. (C.21) give the empirical guarantee

$$
\sum _ { t = 1 } ^ { T } \langle \mathbf { g } _ { t } , \mathbf { x } _ { t } - \mathbf { u } \rangle \leq \mathcal { O } \left( \sqrt { \widetilde { V } _ { T } } + \sqrt { S _ { T } } \right) .\tag{C.24}
$$

Combining this with Eq. (C.9) and Eq. (C.5), and absorbing $\sqrt { S _ { T } }$ by Young’s inequality, gives

$$
\mathrm { R E G } _ { T } \leq \mathcal { O } \left( \sqrt { V _ { T } } + \sqrt { S _ { T } } \right) - \frac { \gamma } { 2 } S _ { T } \leq \mathcal { O } \left( \sqrt { V _ { T } } \right) .\tag{C.25}
$$

## Appendix D. Proofs for Section 6

Proof. Let $\mathbf { g } _ { t } ^ { \mathbf { x } } = \nabla _ { \mathbf { x } } f ( \mathbf { x } _ { t } , \mathbf { y } _ { t } ) + \gamma ( \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } )$ and $\mathbf { g } _ { t } ^ { \mathbf { y } } = - \nabla _ { \mathbf { y } } f ( \mathbf { x } _ { t } , \mathbf { y } _ { t } ) + \gamma ( \mathbf { y } _ { t } - \mathbf { y } _ { t - 1 } )$ be the corrected gradients of the two players, with $\gamma = G / D , { \bf x } _ { 0 } = { \bf x } _ { 1 }$ , and $\mathbf { y } _ { 0 } = \mathbf { y } _ { 1 }$ . Write the empirical gradient variations and cumulative squared movements as

$$
\bar { V } _ { T } ^ { \mathbf { x } } = \sum _ { t = 2 } ^ { T } \Vert \nabla _ { \mathbf x } f ( \mathbf x _ { t } , \mathbf y _ { t } ) - \nabla _ { \mathbf x } f ( \mathbf x _ { t - 1 } , \mathbf y _ { t - 1 } ) \Vert ^ { 2 } ,
$$

$$
\widetilde { V } _ { T } ^ { { \mathbf { x } } } = \sum _ { t = 2 } ^ { T } \left. { \mathbf { g } } _ { t } ^ { { \mathbf { x } } } - { \mathbf { g } } _ { t - 1 } ^ { { \mathbf { x } } } \right. ^ { 2 } , \qquad S _ { T } ^ { { \mathbf { x } } } = \sum _ { t = 2 } ^ { T } \left. { \mathbf { x } } _ { t } - { \mathbf { x } } _ { t - 1 } \right. ^ { 2 } ,
$$

$$
\bar { V } _ { T } ^ { \mathbf { y } } = \sum _ { t = 2 } ^ { T } \| \nabla _ { \mathbf { y } } f ( \mathbf x _ { t } , \mathbf y _ { t } ) - \nabla _ { \mathbf y } f ( \mathbf x _ { t - 1 } , \mathbf y _ { t - 1 } ) \| ^ { 2 } ,
$$

$$
\widetilde { V } _ { T } ^ { { \mathbf { y } } } = \sum _ { t = 2 } ^ { T } \left. { \mathbf { g } } _ { t } ^ { { \mathbf { y } } } - { \mathbf { g } } _ { t - 1 } ^ { { \mathbf { y } } } \right. ^ { 2 } , \qquad S _ { T } ^ { { \mathbf { y } } } = \sum _ { t = 2 } ^ { T } \left. { \mathbf { y } } _ { t } - { \mathbf { y } } _ { t - 1 } \right. ^ { 2 } .
$$

For the dishonest case, the two-player game degenerates to two separate online convex optimization problems. Therefore, the results for both bilinear and strongly-convex-strongly-concave games follow directly from Theorem 2 since $V _ { T } = \mathcal { O } ( T )$

For the honest case, we focus on the convex-concave game, where $f ( \mathbf { x } , \mathbf { y } )$ is convex in x and concave in $\mathbf { y } ,$ which subsumes both bilinear and strongly-convex-strongly-concave games. The proof begins by analyzing the regret of player-x, following the structure in Appendix C.2.

For the meta regret, Eq. (C.14) gives

$$
\begin{array} { r l } & { \mathbf { M E T A - R E G } ^ { \mathbf { x } } \leq \sqrt { 2 } C _ { 0 } D \sqrt { \widetilde { V } _ { T } ^ { \mathbf { x } } } + \sqrt { 2 } C _ { 0 } H \sqrt { S _ { T } ^ { \mathbf { x } } } + ( C _ { 0 } + C _ { 1 } ) H D , } \\ & { \mathbf { M E T A - R E G ^ { y } } \leq \sqrt { 2 } C _ { 0 } D \sqrt { \widetilde { V } _ { T } ^ { \mathbf { y } } } + \sqrt { 2 } C _ { 0 } H \sqrt { S _ { T } ^ { \mathbf { y } } } + ( C _ { 0 } + C _ { 1 } ) H D . } \end{array}\tag{D.1}
$$

For the base regret, Eq. (C.21) and Eq. (C.22) give

$$
\begin{array} { r } { \mathbf { B A S E - R E G } ^ { \mathbf { x } } \leq \mathcal { O } \left( \sqrt { \widetilde { V } _ { T } ^ { \mathbf { x } } } \right) , \quad \mathbf { B A S E - R E G } ^ { \mathbf { y } } \leq \mathcal { O } \left( \sqrt { \widetilde { V } _ { T } ^ { \mathbf { y } } } \right) . } \end{array}\tag{D.2}
$$

Combining Eq. (D.1) and Eq. (D.2), we have

$$
\begin{array} { r l } & { \quad \mathrm { R E G } _ { T } ^ { \mathbf { x } } + \mathrm { R E G } _ { T } ^ { \mathbf { y } } } \\ & { \leq \mathrm { M E T A } { - \mathrm { R E G } ^ { \mathbf { x } } } + \mathrm { M E T A } { - \mathrm { R E G } ^ { \mathbf { y } } } + \mathrm { B A S E } { - \mathrm { R E G } ^ { \mathbf { x } } } + \mathrm { B A S E } { - \mathrm { R E G } ^ { \mathbf { y } } } - \frac { \gamma } { 2 } S _ { T } ^ { \mathbf { x } } - \frac { \gamma } { 2 } S _ { T } ^ { \mathbf { y } } + { \mathcal O } ( 1 ) } \\ & { \leq { \mathcal O } \left( \sqrt { \widetilde { V } _ { T } ^ { \mathbf { x } } } + \sqrt { \widetilde { V } _ { T } ^ { \mathbf { y } } } + \sqrt { S _ { T } ^ { \mathbf { x } } } + \sqrt { S _ { T } ^ { \mathbf { y } } } \right) - \frac { \gamma } { 2 } S _ { T } ^ { \mathbf { x } } - \frac { \gamma } { 2 } S _ { T } ^ { \mathbf { y } } } \\ & { \leq { \mathcal O } \left( \sqrt { \widetilde { V } _ { T } ^ { \mathbf { x } } } + \sqrt { \widetilde { V } _ { T } ^ { \mathbf { y } } } + \sqrt { S _ { T } ^ { \mathbf { x } } } + \sqrt { S _ { T } ^ { \mathbf { y } } } \right) - \frac { \gamma } { 2 } S _ { T } ^ { \mathbf { x } } - \frac { \gamma } { 2 } S _ { T } ^ { \mathbf { y } } \leq { \mathcal O } ( 1 ) , } \end{array}
$$

where the third inequality follows from $\tilde { V } _ { T } ^ { \mathbf { x } } \lesssim \bar { V } _ { T } ^ { \mathbf { x } } + S _ { T } ^ { \mathbf { x } }$ and $\widetilde { V } _ { T } ^ { \mathbf { y } } \lesssim \bar { V } _ { T } ^ { \mathbf { y } } + S _ { T } ^ { \mathbf { y } }$ . The last inequality uses $\bar { V } _ { T } ^ { \mathbf { x } } \lesssim S _ { T } ^ { \mathbf { x } } + S _ { T } ^ { \mathbf { y } }$ and $\bar { V } _ { T } ^ { \mathbf { y } } \lesssim S _ { T } ^ { \mathbf { y } } + S _ { T } ^ { \mathbf { x } }$ ■