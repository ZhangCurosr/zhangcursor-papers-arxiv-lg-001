# Last-Iterate Convergence of Stochastic Gradient Methods for Unbounded Problems

Kevin Kurian Thomas Vaidyan Department of Computer Science University of British Columbia

Ahmet Alacaoglu<sup>∗</sup> Department of Mathematics University of British Columbia

Michael P. Friedlander<sup>∗</sup>   
Department of Computer Science and Mathematics   
University of British Columbia

kevinktv@cs.ubc.ca

ahmet.alacaoglu@ubc.ca

michael.friedlander@ubc.ca

## Abstract

Last-iterate convergence guarantees for stochastic gradient methods have attracted signif icant attention as practical implementations typically return the last iterate instead of an average of the iterates. In this paper, we study two approaches for analyzing the last-iterate convergence of proximal and projected stochastic gradient methods without bounded domain or bounded variance assumptions, where black-box variational inference serves as a running example. Under two related assumptions relaxing the bounded variance, we show near-optimal anytime rate $\widetilde { \mathcal { O } } ( 1 / \sqrt { T } )$ for the last iterate. Moreover, when the horizon T is known, we remove the logarithmic factor and obtain the optimal order $\mathcal { O } ( 1 / \sqrt { T } )$ , matching the order known under bounded variance or bounded domain assumptions.

Keywords: Stochastic optimization, proximal methods, last-iterate convergence, unbounded problems

## 1 Introduction

Stochastic gradient descent (SGD) and its variants are widely used in training large-scale machine learning models. In this paper, we study two standard stochastic first-order methods: proximal stochastic gradient descent (proximal SGD) for regularized problems of the form

$$
\operatorname* { m i n } _ { \mathbf { x } \in \mathbb { R } ^ { d } } \big \{ h ( \mathbf { x } ) : = f ( \mathbf { x } ) + g ( \mathbf { x } ) \big \} ,\tag{1.1}
$$

and projected stochastic gradient descent (projected SGD) for convex constrained optimization problems of the form

$$
\operatorname* { m i n } _ { \mathbf { x } \in \mathcal { C } } f ( \mathbf { x } ) .\tag{1.2}
$$

Most classical analyses of SGD provide guarantees on the suboptimality of averaged iterates or their weighted combinations (Nemirovski et al., 2009; Lan, 2020). That is, it is common to obtain rates for $f ( { \bar { \mathbf { x } } } _ { T } ) - f ( \mathbf { x } ^ { * } )$ where $\begin{array} { r } { \bar { \mathbf { x } } _ { T } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } } \end{array}$ x<sub>t</sub>. However, practitioners typically use the last iterate as the output, rather than the averaged iterate (Shamir and Zhang,

2013; Jain et al., 2019). For example, in the $\ell _ { 1 }$ -regularized least-squares problem

$$
\operatorname* { m i n } _ { \mathbf { x } \in \mathbb { R } ^ { d } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( \mathbf { a } _ { i } ^ { \top } \mathbf { x } - b _ { i } ) ^ { 2 } + \lambda \| \mathbf { x } \| _ { 1 } ,\tag{1.3}
$$

the $\ell _ { 1 }$ regularizer promotes sparsity in the solution. However, averaging can destroy this sparsity; hence, guarantees for the last iterate x are of interest (Chen et al., 2012, Sec. 1).

A second challenge concerns the assumptions made on the stochastic oracle. Many analyses require strong assumptions such as bounded second moments or bounded variance of the stochastic gradients $\tilde { \nabla } f ( \mathbf { x } )$ (Robbins and Monro, 1951; Nemirovski et al., 2009; Lan, 2020). That is, these works assume

$$
\begin{array} { r } { \mathbb { E } \| \widetilde { \nabla } f ( \mathbf { x } ) \| ^ { 2 } \leq \sigma ^ { 2 } \quad \mathrm { o r } \quad \mathbb { E } \| \widetilde { \nabla } f ( \mathbf { x } ) - \nabla f ( \mathbf { x } ) \| ^ { 2 } \leq \sigma ^ { 2 } . } \end{array}\tag{1.4}
$$

Both of these assumptions are widely accepted to be rather restrictive and may not hold over unbounded domains even for simple problems such as (1.3). In the context of (1.3), the stochastic gradient is given by $\widetilde { \nabla } f ( \mathbf { x } ) = 2 \mathbf { a } _ { i } ( \langle \mathbf { a } _ { i } , \mathbf { x } \rangle - b _ { i } )$ , which grows with $\| \mathbf { x } \|$ and is unbounded.

These observations motivate the central question of this paper: Can SGD (and its proximal and projected variants) attain last-iterate convergence guarantees without assuming uniformly bounded variance of the stochastic gradient oracle?

We answer this question under two complementary conditions: component-wise convexity and smoothness (CW) (Section 5) and the Blum-Gladyshev (B-G) growth condition (Section 3). Under both conditions, we establish an anytime last-iterate convergence rate of $\mathcal { O } ( \ln T / \sqrt { T } )$ and a log-free rate of $\mathcal { O } ( 1 / \sqrt { T } )$ when the horizon is known. Our analysis under (B-G) accommodates convex objectives that need not be smooth or Lipschitz continuous and yields applications to black-box variational inference. Under (CW), we extend the existing proximal SGD guarantees of Vaidyan et al. (2026) to variable step sizes and improve the convergence rate for horizon-dependent step sizes.

A Brief Discussion on Prior Work. Classical analyses established convergence rates for averaged iterates under bounded domain and bounded second-moment assumptions (Nemirovski et al., 2009). Subsequent works relaxed these assumptions, allowing unbounded domains and growing stochastic gradients (Grimmer, 2019; Khaled et al., 2023). A parallel line of research studied last-iterate convergence, under bounded second-moment or bounded variance assumptions (Shamir and Zhang, 2013; Harvey et al., 2019; Jain et al., 2019; Orabona, 2020; Liu and Zhou, 2024). More recently, several works have obtained lastiterate convergence guarantees under weaker assumptions on the variance (Garrigos et al., 2025; Kim et al., 2025; Attia et al., 2025; Alacaoglu et al., 2025; Vaidyan et al., 2026). More detailed comparisons appear in the respective sections in the sequel.

Component-Wise Convexity and Smoothness. The first approach to weaken (1.4) is to assume component-wise convexity and smoothness of the objective (Garrigos et al., 2025; Attia et al., 2025). In this setting, the objective decomposes as

$$
f ( \mathbf { x } ) = \mathbb { E } _ { i \sim P } [ f _ { i } ( \mathbf { x } ) ] ,\tag{1.5}
$$

for some unknown distribution $P ,$ from which we get i.i.d. samples. In particular, we assume that

$$
{ \mathrm { e a c h ~ c o m p o n e n t ~ f u n c t i o n ~ } } f _ { i } { \mathrm { ~ i s ~ c o n v e x ~ a n d ~ } } L { \mathrm { - s m o o t h } } .\tag{CW}
$$

The most standard setting where this assumption holds is the least squares problem with a regularizer; see (1.3), where each component $f _ { i } ( { \bf x } ) = ( { \bf a } _ { i } ^ { \top } { \bf x } - b _ { i } ) ^ { 2 }$ is convex and $\left( \operatorname* { m a x } _ { i } 2 \lVert \mathbf { a } _ { i } \rVert ^ { 2 } \right)$ smooth. Garrigos et al. (2025) and Attia et al. (2025) used this assumption along with the powerful technique introduced by Zamani and Glineur (2025) to obtain last-iterate guarantees for (1.1) with $g \equiv 0$ . A classical consequence of the component-wise convexity and smoothness assumption is the bound

$$
\begin{array} { r } { \mathbb { E } \| \nabla f _ { i } ( \mathbf { x } ) \| ^ { 2 } \leq 2 \mathbb { E } \| \nabla f _ { i } ( \mathbf { x } ^ { * } ) \| ^ { 2 } + 4 L ( f ( \mathbf { x } ) - f ( \mathbf { x } ^ { * } ) ) , } \end{array}\tag{1.6}
$$

where $\mathbf { x } ^ { * }$ is a minimizer of $f .$ This bound is a special case of the so-called ABC assumption (Polyak and Tsypkin, 1973; Khaled and Richt´arik, 2023; Khaled et al., 2023).

Using this property, Garrigos et al. (2025) and Attia et al. (2025) show that the last iterate of SGD converges at a rate of $\mathcal { O } ( \ln T / \sqrt { T } )$ . The analysis developed in these works is restricted to horizon-dependent step sizes and does not handle regularized problems such as (1.1) where the proximal method is needed. The work by Vaidyan et al. (2026) handles the proximal setting but is still restricted to horizon-dependent step sizes and sufers from a logarithmic factor in the rate, as do earlier works in this line.

Blum-Gladyshev Variance. The second approach to weaken (1.4) is the Blum-Gladyshev (BG) condition (Blum, 1954; Gladyshev, 1965) given as

$$
\begin{array} { r } { \mathbb { E } \| \widetilde { \nabla } f ( \mathbf { x } ) \| ^ { 2 } \leq B ^ { 2 } \| \mathbf { x } - \mathbf { x } _ { 0 } \| ^ { 2 } + G ^ { 2 } , } \end{array}\tag{B-G}
$$

where $\mathbf { x } _ { \mathrm { 0 } }$ is the initial iterate of the algorithm. Note that this assumption is also sometimes presented with a solution $\mathbf { x } ^ { * }$ instead of $\mathbf { x } _ { \mathrm { 0 } }$ on the right-hand side; the two forms are interchangeable (see Fact 1).

Unlike (CW), the Blum-Gladyshev condition (B-G) does not require the individual stochastic functions $f _ { i }$ to be smooth, Lipschitz continuous, or convex. We provide more concrete examples to illustrate this point in Section 4.

Moreover, the BG condition is strictly weaker than the ABC assumption. It was shown, for example, by Alacaoglu et al. (2025), that (1.6) (which itself is implied by CW) implies (B-G), when f is L-smooth, by straightforward estimations.

However, the converse need not be true. Alacaoglu et al. (2025, Example 1.1) show that, for a simple quadratic example over an unbounded domain with a standard stochastic oracle, (B-G) is satisfied, but (1.6) (and, consequently, CW) fails to hold. This establishes that (B-G) is a weaker condition than (CW) even for smooth problems.

Additionally, without the expectation, the (B-G) assumption also describes a class of functions f that are neither smooth nor Lipschitz. For this class, we obtain last-iterate rates for the classical subgradient method (see Remark 3.7), which are new to the best of our knowledge. Existing results for this class either give rates on the ergodic average (Lu, 2019; Grimmer, 2019; Li et al., 2025) or rely on transformations of the original problem to design specialized algorithms that depart from the simple subgradient method (Renegar, 2016; Grimmer, 2018).

Application: Black-Box Variational Inference. Interestingly, the (B-G) assumption has recently been shown to be relevant for black-box variational inference (BBVI) (Kim et al., 2025; Labarri\`ere et al., 2026; Domke et al., 2023). Variational inference is a classical technique in statistics and machine learning, widely used in applications including healthcare (Ranganath et al., 2016; Lopez et al., 2018). Its goal is to approximate a potentially intractable posterior distribution by a distribution from a simpler family, such as Gaussians, by minimizing the Kullback–Leibler divergence between the two.

When the target distribution is complex, closed-form updates for variational inference are often unavailable. However, accessing stochastic gradients is still feasible, making stochastic optimization-based approaches, such as BBVI (Ranganath et al., 2014), popular for this task. In BBVI, the second moment of the standard gradient estimator is not uniformly bounded, so (1.4) does not hold. However, under smoothness assumptions on the log-posterior, it satisfies the (B-G) condition (Domke et al., 2023). See Section 4.2 for technical details relevant to our exposition.

Comparison of the Two Approaches. The assumptions (B-G) and (CW) are complementary. As discussed earlier, the (B-G) assumption is weaker and covers a broader class of problems not handled by (CW). However, under (CW), the standard step size $\tau _ { t } = \tau / \sqrt { t + 1 }$ attains the optimal rate up to a logarithmic factor, which our analysis under (B-G) does not support; see Remark 5.6. Hence, there is a trade-of between the two assumptions: (B-G) is weaker but yields suboptimal order in the rate with the standard step size of $\tau _ { t } = \tau / \sqrt { t + 1 }$ , while (CW) is stronger but yields the optimal order in a wider variety of settings. See also Table 1.

## Summary of Contributions.

• Last-iterate rates under (B-G). We establish last-iterate convergence rates for proximal SGD when f is convex but not necessarily smooth or Lipschitz. Specifically, we obtain an anytime rate of $\mathcal { O } ( \ln T / \sqrt { T } )$ and, when the horizon is known, a log-free rate of $\mathcal { O } ( 1 / \sqrt { T } )$ using the step-size scheme of Liu and Zhou (2024) (see Theorem 3.5 for more details). Our results also apply to projected SGD. To our knowledge, these are the first anytime and the first log-free last-iterate rates for SGD without assuming bounded variance. As a special case, we obtain last-iterate rates for the classical subgradient method on convex functions that are not Lipschitz.

• Black-box variational inference. Using the gradient bounds of Domke et al. (2023), our results give anytime last-iterate guarantees for proximal and projected SGD for BBVI (Section 4.2). Existing guarantees hold for weighted averages of the iterates with horizon-dependent step sizes. Our results answer an open question by Kim et al. (2025, Section 2.4) regarding anytime guarantees of SGD for BBVI without strong convexity.

• Variable step sizes under (CW). We extend the proximal SGD analysis of Vaidyan et al. (2026), which is restricted to constant horizon-dependent step sizes, to handle non-increasing variable step sizes. We establish an anytime convergence rate of $\mathcal { O } ( \ln T / \sqrt { T } )$ and a log-free rate of $\mathcal { O } ( 1 / \sqrt { T } )$ when the horizon is known (see Theorem 5.4 for more details). Projected SGD follows as a special case. These results

answer an open question posed by (Garrigos et al., 2025, Remark 3.6, Section 5), in the unconstrained case.
<table><tr><td>Property</td><td>(CW) Assumption</td><td>(B-G) Assumption</td></tr><tr><td>Allows nonconvex  $\overline { { f _ { i } } }$ </td><td>X</td><td>√</td></tr><tr><td>Allows nonsmooth  $f _ { i }$ </td><td>X</td><td>V</td></tr><tr><td>Control mechanism</td><td>ABC-type inequality (1.6)</td><td>Iterates bounded in expectation</td></tr><tr><td>Step size regimes</td><td> $\begin{array} { r } { \tau _ { t } \leq \frac { 1 } { 4 L \sqrt { t + 1 } } } \end{array}$ </td><td> $\textstyle \sum _ { t = 1 } ^ { T } \tau _ { t } ^ { 2 } = \mathcal { O } ( 1 )$ </td></tr></table>

Table 1: Comparison of the assumptions considered in this paper.

Analysis Roadmap. We analyze SGD-based methods under both (B-G) and (CW). In both cases, the analysis unfolds in two steps:

1. We first establish a control on the second moment of the stochastic gradients along the trajectory of the algorithm, that is, on the quantity $\mathbb { E } \Vert \widetilde { \nabla } f ( \mathbf { x } _ { t } ) \Vert ^ { 2 }$ . Under (CW), this control is obtained through the ABC condition in (1.6). Under (B-G), we show that as long as the sum of the squared step sizes does not grow too quickly, the iterates of SGD remain bounded in expectation, which implies that the second moment of the stochastic gradients is bounded along the trajectory of the algorithm.

2. With this control in place, we adapt the reference point argument of Zamani and Glineur (2025) and its generalization in Liu and Zhou (2024): we prove a one-step progress bound against an arbitrary reference point, then choose these points as convex combinations of the minimizer and past iterates so that summing the bounds gets us the last-iterate guarantee.

Relation to Vaidyan et al. (2026). This paper extends our conference paper (Vaidyan et al., 2026), which analyzed proximal SGD under (CW) with a constant, horizon-dependent step size and obtained an $\mathcal { O } ( \ln T / \sqrt { T } )$ last-iterate rate. The present paper difers from it in three ways (see also Table 2).

• The analysis under (B-G) in Section 3 is new. It controls the second moment of the stochastic gradients through boundedness of the iterates in expectation, rather than through the ABC-type inequality (1.6) used in Vaidyan et al. (2026). As a result, it applies to proximal SGD on convex objectives that need not be smooth or Lipschitz.

• The applications in Section 4 are new. We show that (B-G) covers problems outside the scope of (CW), such as minimizing the maximum of convex quadratics (Section 4.1), and we use it to obtain last-iterate guarantees for SGD in black-box variational inference (Section 4.2). This application was not covered in Vaidyan et al. (2026), because the (CW) assumption does not hold for BBVI.

<table><tr><td>Reference</td><td>Constrained/ Regularized</td><td>Assumptions</td><td>Anytime rate</td><td></td><td>Horizon- dependent rate</td></tr><tr><td>Shamir and Zhang (2013)</td><td>√</td><td>Bdd gradients/domain</td><td></td><td> $\overline { { \mathcal { O } \Big ( \frac { \ln T } { \sqrt { T } } \Big ) } }$ </td><td></td></tr><tr><td>Jain et al. (2019)</td><td>√</td><td>Bdd gradients/domain</td><td></td><td></td><td> $\scriptstyle { \mathcal { O } } \left( { \frac { 1 } { \sqrt { T } } } \right)$ </td></tr><tr><td>Liu and Zhou (2024)</td><td>√</td><td>Bdd variance</td><td></td><td> $\begin{array} { r l } { \mathcal { O } \left( \frac { \ln T } { \sqrt { T } } \right) } & { { } } \end{array}$ </td><td> $\scriptstyle { \mathcal { O } } \left( { \frac { 1 } { \sqrt { T } } } \right)$ </td></tr><tr><td>Garrigos et al. (2025); Attia et al. (2025)</td><td>X</td><td>(CW)</td><td></td><td></td><td> $\begin{array} { r l } { \mathcal { O } \left( \frac { \ln T } { \sqrt { T } } \right) } & { { } } \end{array}$ </td></tr><tr><td>Vaidyan et al. (2026)</td><td>√</td><td>(CW)</td><td></td><td></td><td> $\begin{array} { r l } { \mathcal { O } \left( \frac { \ln T } { \sqrt { T } } \right) } & { { } } \end{array}$ </td></tr><tr><td>This work</td><td>√</td><td>(B-G)</td><td></td><td> $\overline { { \mathcal { O } \Big ( \frac { \ln T } { \sqrt { T } } \Big ) } }$ </td><td> $\overline { { \mathcal { O } \Big ( \frac { 1 } { \sqrt { T } } \Big ) } }$ </td></tr><tr><td>This work</td><td>√</td><td>(CW)</td><td></td><td> $\begin{array} { r l } { \mathcal { O } \left( \frac { \ln T } { \sqrt { T } } \right) } & { { } } \end{array}$ </td><td> $\begin{array} { r } { \underline { { \mathcal { O } \left( \frac { 1 } { \sqrt { T } } \right) } } } \end{array}$ </td></tr></table>

Table 2: Comparison with existing last-iterate convergence results for SGD variants.

• In Section 5 we return to the (CW) setting of Vaidyan et al. (2026) and extend its analysis from constant to non-increasing variable step sizes. The key technical step, Lemma 3.4, generalizes the reference-point construction of Vaidyan et al. (2026, Lemma A.6); this gives an anytime $\mathcal { O } ( \ln T / \sqrt { T } )$ rate with $\tau _ { t } = \tau / \sqrt { t + 1 }$ , and a logfree $\mathcal { O } ( 1 / \sqrt { T } )$ rate when the horizon is known (Theorem 5.4). This improves on the guarantee in Vaidyan et al. (2026, Theorem 3.1), which required constant step sizes and included a log term.

## 2 Problem Setup and Assumptions

In this section, we introduce the two stochastic first-order methods studied in the paper and fix the notation used throughout. Given iterates $\left( \mathbf { x } _ { t } \right)$ , let $\mathcal { F } _ { t } : = \sigma ( \mathbf { x } _ { 0 } , \ldots , \mathbf { x } _ { t } )$ denote the σ-algebra generated by $\mathbf { x } _ { 0 } , \ldots , \mathbf { x } _ { t } .$ , and write $\mathbb { E } _ { t } [ \cdot ] : = \mathbb { E } [ \cdot \mid \mathcal { F } _ { t } ]$ for the conditional expectation with respect to $\mathcal { F } _ { t }$ . We use $\langle \cdot , \cdot \rangle$ for the Euclidean inner product and ∥ · ∥ for the Euclidean norm. We use $\partial f ( \mathbf { x } )$ to denote the subdiferential of a convex function f at x.

Proximal Stochastic Gradient Descent. For the additively composite problem given in (1.1), we assume that $f \colon  { \mathbb { R } ^ { d } } \to  { \mathbb { R } }$ is convex, but not necessarily smooth, and $g \colon  { \mathbb { R } ^ { d } } \to$ $\mathbb { R } \cup \{ + \infty \}$ is proper, convex and closed. We denote $\mathbf { x } ^ { * } \in$ arg min $\mathfrak { l } _ { \mathbf { X } \in \mathbb { R } ^ { d } } h ( \mathbf { x } )$ as an optimal solution and $h ^ { * } = h ( \mathbf { x } ^ { * } )$ . The proximal operator of g is defined by

$$
\mathrm { p r o x } _ { g } ( \mathbf { y } ) = \underset { \mathbf { x } \in \mathbb { R } ^ { d } } { \arg \operatorname* { m i n } } \left. g ( \mathbf { x } ) + \frac { 1 } { 2 } \| \mathbf { x } - \mathbf { y } \| ^ { 2 } \right. .
$$

Given $\mathbf { x } _ { 0 } \in$ dom g, the proximal SGD update is given for $t \geq 0$ as

$$
\begin{array} { r } { \mathbf { x } _ { t + 1 } = \operatorname { p r o x } _ { \tau _ { t } g } \left( \mathbf { x } _ { t } - \tau _ { t } \widetilde { \nabla } f ( \mathbf { x } _ { t } ) \right) , } \end{array}\tag{proxSGD}
$$

where $\widetilde { \nabla } f ( \mathbf { x } _ { t } )$ denotes an unbiased stochastic subgradient of $f$ satisfying

$$
\mathbb { E } _ { t } [ \widetilde { \nabla } f ( \mathbf { x } _ { t } ) ] \in \partial f ( \mathbf { x } _ { t } ) .
$$

Projected Stochastic Gradient Descent. The constrained problem (1.2) is the special case of (1.1) with $g = \delta c$ , where $\delta _ { \mathcal { C } }$ is the indicator function of a nonempty, closed and convex set ${ \mathcal { C } } ,$ defined as $\delta c ( \mathbf { x } ) = 0 { \mathrm { ~ i f ~ } } \mathbf { x } \in { \mathcal { C } }$ and $\delta c ( \mathbf { x } ) = + \infty$ otherwise. In this case, $\mathrm { p r o x } _ { \tau _ { t } g }$ is the Euclidean projection onto C, so (proxSGD) reduces to projected SGD, and all of our results for proximal SGD apply to it.

Assumptions. We now state the standard assumptions under which we analyze proximal SGD.

Assumption 1 The problem (1.1) satisfies the following:

1. The solution set of (1.1) is nonempty.

2. The function $g \colon  { \mathbb { R } } ^ { d } \to  { \mathbb { R } } \cup \{ + \infty \}$ is proper, closed and convex.

3. The function $f \colon  { \mathbb { R } ^ { d } } \to  { \mathbb { R } }$ is convex.

$\it 4 .$ The oracle is unbiased: $\mathbb { E } _ { t } [ \widetilde { \nabla } f ( \mathbf { x } _ { t } ) ] \in \partial f ( \mathbf { x } _ { t } )$ for all $t \geq 0$

Note that Assumption 1 does not require f to be diferentiable. For projected SGD, that ${ \mathrm { i s } } ,$ with $g = \delta _ { \mathcal { C } }$ , the second item amounts to C being nonempty, closed and convex. Assumption 1 is standard in the literature on projected SGD (Nemirovski et al., 2009; Shamir and Zhang, 2013; Grimmer, 2019) and proximal SGD (Khaled et al., 2023; Rosasco et al., 2020; Vaidyan et al., 2026). For the stochastic oracle, we consider the two models discussed in Section 1.

Assumption 2 The stochastic oracle satisfies one of the following:

(a) There exist $B , G \geq 0$ such that (B-G) holds for all ${ \textbf { x } } \in$ dom g. That $i s ,$ , along the iterates, it holds that

$$
\begin{array} { r } { \mathbb { E } _ { t } \| \widetilde { \nabla } f ( \mathbf { x } _ { t } ) \| ^ { 2 } \leq B ^ { 2 } \| \mathbf { x } _ { t } - \mathbf { x } _ { 0 } \| ^ { 2 } + G ^ { 2 } . } \end{array}
$$

(b) The following hold:

(i) The function f decomposes as (1.5), where every $f _ { i }$ satisfies (CW).

(ii) The oracle is $\widetilde { \nabla } f ( \mathbf { x } _ { t } ) = \nabla f _ { i _ { t } } ( \mathbf { x } _ { t } )$ , where $i _ { t } \sim P$ is sampled independently of $\mathcal { F } _ { t }$

(iii) $\mathbb { E } _ { i \sim P } \| \nabla f _ { i } ( \mathbf { x } ^ { * } ) \| ^ { 2 } \leq \sigma _ { * } ^ { 2 } < \infty$ for some solution $\mathbf { x } ^ { * }$

In the subsequent sections, we will be combining Assumption 1 with an assumption on the oracle (Assumption $2 ( \mathrm { a } )$ or Assumption $2 ( \mathrm { b } ) )$ to analyze the convergence of proximal SGD. Under Assumption 2(b), f is L-smooth since every $f _ { i }$ is L-smooth, and $\mathbb { E } _ { t } [ \widetilde { \nabla } f ( \mathbf { x } _ { t } ) ] =$ $\nabla f ( \mathbf { x } _ { t } )$

## 3 Convergence Analysis under B-G

We now proceed to analyze the last-iterate convergence of proximal SGD under the BG condition (B-G). Because of its applicability to BBVI, SGD under this assumption has been studied in several recent works (Domke et al., 2023; Labarri\`ere et al., 2026; Kim et al., 2025). The analyses in these works set up the recursion

$$
\begin{array} { r } { 2 \tau _ { t } \mathbb { E } \left[ h ( \mathbf { x } _ { t + 1 } ) - h ^ { * } \right] \leq ( 1 + 2 \tau _ { t } ^ { 2 } B ^ { 2 } ) \mathbb { E } \| \mathbf { x } _ { t } - \mathbf { x } ^ { * } \| ^ { 2 } - \mathbb { E } \| \mathbf { x } _ { t + 1 } - \mathbf { x } ^ { * } \| ^ { 2 } + 2 \tau _ { t } ^ { 2 } G ^ { 2 } , } \end{array}\tag{3.1}
$$

where $\tau _ { t } = \tau$ is fixed. In this recursion, the terms involving $\mathbb { E } \lVert \mathbf { x } _ { t } - \mathbf { x } ^ { * } \rVert ^ { 2 }$ do not telescope due to the additional $1 + 2 \tau ^ { 2 } B ^ { 2 }$ factor. These works then handle this issue by weighting the inequalities with $\alpha _ { t + 1 } = \alpha _ { t } / ( 1 + 2 \tau ^ { 2 } B ^ { 2 } )$ which then leads to a telescoping sum on the right-hand side and yields a guarantee for the weighted average of the iterates. However, this approach does not accommodate variable step sizes, and it gives neither anytime nor last-iterate guarantees, both of which our analysis provides.

Li et al. (2025) studied the subgradient method without assuming Lipschitz continuity, using a strategy similar to our analysis roadmap. This work used normalized step sizes $\tau _ { t } = \beta _ { t } / \| \mathbf { s } _ { t } \|$ with $\mathbf { s } _ { t } \in \partial f ( \mathbf { x } _ { t } )$ and $\textstyle \sum _ { k = 0 } ^ { \infty } \beta _ { k } ^ { 2 } < \infty$ , which allows bounding the iterates and, consequently, the subgradient norms along the trajectory. With this bound in place, the standard analysis for bounded subgradients applies. Unlike our analysis, this approach requires the step size to depend on the norm of the sampled subgradient, which does not extend directly to stochastic oracles. As a result, this work analyzed the stochastic subgradient method only under the interpolation condition. In addition, this work derived rates for the average of the iterates, whereas our focus is on last-iterate rates.

Throughout this section, we require Assumptions 1 and $2 ( \mathrm { a } )$ . We follow the two-step strategy outlined in the analysis roadmap in Section 1. For the analysis, it is convenient to recenter the BG condition (B-G) at a solution $\mathbf { x } ^ { * }$ . We apply Fact 1 with $\mathbf { a } = \mathbf { x } _ { 0 }$ and b $= \mathbf { x } ^ { * }$ to obtain

$$
\begin{array} { r } { \mathbb { E } _ { t } \| \widetilde { \nabla } f ( \mathbf { x } _ { t } ) \| ^ { 2 } \leq 2 B ^ { 2 } \| \mathbf { x } _ { t } - \mathbf { x } ^ { * } \| ^ { 2 } + \widehat { G } ^ { 2 } , \qquad \widehat { G } ^ { 2 } : = 2 B ^ { 2 } \| \mathbf { x } _ { 0 } - \mathbf { x } ^ { * } \| ^ { 2 } + G ^ { 2 } . } \end{array}\tag{3.2}
$$

In order to establish boundedness of the iterates and the second moment of the stochastic subgradients, we first need the following lemma which analyzes one step of the algorithm.

Lemma 3.1 Let Assumption 1 hold. For all $t \geq 0$ , any $\mathcal { F } _ { t } – m e a s u r a b l e \ \mathbf { z } _ { t }$ , and any $\mathbf { q } _ { t + 1 } \in$ $\partial f ( \mathbf { x } _ { t + 1 } )$ , we have

$$
\begin{array} { r } { 2 \tau _ { t } \mathbb { E } [ h ( \mathbf { x } _ { t + 1 } ) - h ( \mathbf { z } _ { t } ) ] \leq \mathbb { E } \| \mathbf { x } _ { t } - \mathbf { z } _ { t } \| ^ { 2 } - \mathbb { E } \| \mathbf { x } _ { t + 1 } - \mathbf { z } _ { t } \| ^ { 2 } + \tau _ { t } ^ { 2 } \mathbb { E } \| \mathbf { q } _ { t + 1 } - \widetilde { \nabla } f ( \mathbf { x } _ { t } ) \| ^ { 2 } . } \end{array}
$$

The proof of Lemma 3.1 is given in Appendix C.1. We now use the above lemma to establish the expected boundedness of the iterates and the second moment of the stochastic gradients along the trajectory of the algorithm.

Proposition 3.2 (Expected boundedness) Let Assumptions 1 and $\mathcal { Z } ( a )$ hold, and set $\begin{array} { r } { \tau _ { t } \le \frac { 1 } { 2 \sqrt { 2 } B } } \end{array}$ . If for all $t \geq 0$ , the iterates satisfy

$$
\begin{array} { r } { \mathbb { E } \| { \mathbf { x } } _ { t + 1 } - { \mathbf { x } } ^ { * } \| ^ { 2 } \leq \mathbb { E } \| { \mathbf { x } } _ { t } - { \mathbf { x } } ^ { * } \| ^ { 2 } + \tau _ { t } ^ { 2 } \mathbb { E } \| { \mathbf { q } } _ { t + 1 } - \widetilde { \nabla } f ( { \mathbf { x } } _ { t } ) \| ^ { 2 } , } \end{array}\tag{3.3}
$$

where $\mathbf { q } _ { t + 1 } : = \mathbb { E } _ { t + 1 } [ \widetilde { \nabla } f ( \mathbf { x } _ { t + 1 } ) ] \in \partial f ( \mathbf { x } _ { t + 1 } )$ , then, we have

$$
\begin{array} { r } { \mathbb { E } \| { \mathbf x } _ { t + 1 } - { \mathbf x } ^ { * } \| ^ { 2 } \leq \exp \big ( 1 6 B ^ { 2 } P _ { t } \big ) \left( \| { \mathbf x } _ { 0 } - { \mathbf x } ^ { * } \| ^ { 2 } + 8 P _ { t } \widehat { G } ^ { 2 } \right) \quad a n d \mathbb { E } \| \widetilde { \nabla } f ( { \mathbf x } _ { t } ) \| ^ { 2 } \leq K _ { t - 1 } , } \end{array}
$$

where $\textstyle P _ { t } : = \sum _ { i = 0 } ^ { t } \tau _ { i } ^ { 2 }$ with $\begin{array} { r } { P _ { - 1 } : = 0 \ a n d \ K _ { t } : = 2 B ^ { 2 } \exp ( 1 6 B ^ { 2 } P _ { t } ) \left( \| \mathbf { x } _ { 0 } - \mathbf { x } ^ { * } \| ^ { 2 } + 8 P _ { t } \widehat { G } ^ { 2 } \right) + \widehat { G } ^ { 2 } . } \end{array}$

See Appendix A.1 for the proof of Proposition 3.2. We emphasize that, with an appropriate choice of step size, the factor $\exp ( 1 6 B ^ { 2 } P _ { t } )$ is bounded by an absolute constant that grows with neither $T$ nor $B .$ . To see this, consider the step sizes in Theorem 3.5(i)-(ii), where the quantity $P _ { t }$ stays bounded by a constant multiple of $\tau ^ { 2 }$ (Lemmas C.2 and $\mathrm { C . 3 ( i ) } )$ . Since $\tau$ is chosen proportional to $1 / B _ { ; }$ , the $B ^ { 2 }$ in the exponent cancels, so $B ^ { 2 } P _ { t }$ is bounded by an absolute constant.

For the third step size choice, that is, $\tau _ { t } = \tau / \sqrt { t + 1 }$ , the quantity $P _ { t }$ grows logarithmically (Lemma C.1(i)), so this exponential factor grows polynomially in $T _ { i }$ , which leads to the suboptimal order in Theorem 3.5(iii).

Applying Lemma 3.1 with $\mathbf { z } _ { t } = \mathbf { x } ^ { * }$ and $\mathbf { q } _ { t + 1 } = \mathbb { E } _ { t + 1 } [ \widetilde { \nabla } f ( \mathbf { x } _ { t + 1 } ) ]$ , which lies in $\partial f ( \mathbf { x } _ { t + 1 } )$ by Assumption 1, and dropping the nonnegative left-hand side shows that (proxSGD) satisfies (3.3). As a result, if $\begin{array} { r } { \tau _ { t } ~ \le ~ \frac { 1 } { 2 \sqrt { 2 } B } } \end{array}$ , Proposition 3.2 applies, giving us the estimate $\mathbb { E } \| \widetilde { \nabla } f ( \mathbf { x } _ { t } ) \| ^ { 2 } \leq K _ { t - 1 }$ for all $0 \leq t$

Thus, we have established control over the second moment, which is the first ingredient in the analysis roadmap. We now proceed to the second step in our roadmap. We obtain a one-step progress bound for proximal SGD under the BG noise model.

Lemma 3.3 (One-step of proximal SGD under (B-G)) Under Assumptions 1 and $\mathcal { Z } ( a )$ set $\begin{array} { r } { \tau _ { t } \le \frac { 1 } { 2 \sqrt { 2 } B } } \end{array}$ . Then for all $t \geq 0$ and every $\mathcal { F } _ { t }$ -measurable $\mathbf { z } _ { t }$ , (proxSGD) satisfies

$$
\mathbb { E } [ h ( \mathbf { x } _ { t + 1 } ) - h ( \mathbf { z } _ { t } ) ] \leq \frac { 1 } { 2 \tau _ { t } } \mathbb { E } [ \| \mathbf { x } _ { t } - \mathbf { z } _ { t } \| ^ { 2 } - \| \mathbf { x } _ { t + 1 } - \mathbf { z } _ { t } \| ^ { 2 } ] + 2 \tau _ { t } K _ { t } ,
$$

where $K _ { t }$ is defined in Proposition 3.2.

The proof of Lemma 3.3 is given in Appendix A.2.

To convert this one-step bound into a last-iterate convergence rate, we use the referencepoint technique of Zamani and Glineur (2025). Specifically, we take each $\mathbf { z } _ { t }$ to be a convex combination of $\mathbf { x } ^ { * }$ and the past iterates, with weights chosen so that, after summing the one-step bounds, every objective gap except the last one cancels. We can apply this due to our two-step analysis: in the first step (see Proposition 3.2), we established the bound $K _ { t }$ on the second moment of the stochastic subgradients observed by the algorithm. Hence, we have reduced our setting to one similar to the bounded subgradient settings of earlier work, by restricting to the trajectory of the algorithm. Lemma 3.4 extends this technique to the proximal setting, building on the analysis of Liu and Zhou (2024), which assumes bounded variance.

We state Lemma 3.4 in a more general form than we need in this section. By Lemma 3.3, its hypothesis holds with $\rho _ { t } \equiv 0$ under (B-G). Under (CW) in Section 5, however, the onestep inequality contains the extra term $\rho _ { t } [ h ( { \bf x } _ { t } ) - h ^ { * } ]$ , which we handle by choosing the weights of the convex combinations more carefully. This generalizes Vaidyan et al. (2026, Lemma $\mathrm { A . 6 } )$ from constant step sizes and constant $\rho$ to non-increasing step sizes and timevarying $\rho _ { t }$

Lemma 3.4 (Proximal last-iterate reduction) Let Assumption 1 hold with positive nonincreasing $\left( \tau _ { t } \right)$ , and define $\begin{array} { r } { S _ { t } : = \sum _ { s = t } ^ { T } \tau _ { s } } \end{array}$ . Suppose there exist $0 \leq \rho _ { t } \leq 1$ and $v _ { t } \geq 0$ such that for all $0 \leq t \leq T$ and every F<sub>t</sub>-measurable $\mathbf { z } _ { t }$ , we have

$$
\mathbb { E } [ h ( \mathbf { x } _ { t + 1 } ) - h ( \mathbf { z } _ { t } ) - \rho _ { t } h ( \mathbf { x } _ { t } ) + \rho _ { t } h ^ { * } ] \leq \frac { 1 } { 2 \tau _ { t } } \mathbb { E } [ \| \mathbf { x } _ { t } - \mathbf { z } _ { t } \| ^ { 2 } - \| \mathbf { x } _ { t + 1 } - \mathbf { z } _ { t } \| ^ { 2 } ] + v _ { t } .
$$

Then, it holds that

$$
\mathbb { E } [ h ( \mathbf { x } _ { T + 1 } ) - h ^ { * } ] \leq \frac { 1 } { 2 } \exp \left( \sum _ { t = 1 } ^ { T } \frac { \tau _ { t } \rho _ { t } } { S _ { t } } \right) \left( \frac { \| \mathbf { x } _ { 0 } - \mathbf { x } ^ { * } \| ^ { 2 } } { S _ { 0 } } + 2 \sum _ { t = 0 } ^ { T } v _ { t } \frac { \tau _ { t } } { S _ { t } } + \frac { 2 \rho _ { 0 } \tau _ { 0 } [ h ( \mathbf { x } _ { 0 } ) - h ^ { * } ] } { S _ { 0 } } \right) .
$$

The proof of Lemma 3.4 is given in Appendix C.3.

As just discussed, the hypothesis of Lemma 3.4 holds with $\rho _ { t } = 0$ due to Lemma $3 . 3 ,$ so the result of Lemma 3.4 applies. We next instantiate this result with three standard step size choices.

Theorem 3.5 (Proximal SGD under (B-G)) Let Assumptions 1 and $\mathcal { Z } ( a )$ hold. Then the following guarantees hold for the last iterate of (proxSGD):

(i) (Step size for log-free rate) $\begin{array} { r } { I f \tau _ { t } = \tau \frac { T - t + 1 } { ( T + 1 ) ^ { 3 / 2 } } } \end{array}$ with $\begin{array} { r } { \tau = \frac { 1 } { 4 B } } \end{array}$ , then for any $T \geq 1$ , we have

$$
\mathbb { E } [ h ( { \bf x } _ { T + 1 } ) - h ^ { * } ] = \mathcal { O } \left( \frac { B \| { \bf x } _ { 0 } - { \bf x } ^ { * } \| ^ { 2 } + G ^ { 2 } / B } { \sqrt { T + 1 } } \right) .
$$

(ii) (Anytime step size I) $\begin{array} { r } { I f \tau _ { t } = \frac { \tau } { \ln ( t + 2 ) \sqrt { t + 2 } } \ w i t h \ \tau = \frac { 1 } { 4 \sqrt { 3 } B } } \end{array}$ , then for any $T \geq 2 ,$ we have

$$
\mathbb { E } \big [ h ( \mathbf { x } _ { T + 1 } ) - h ^ { * } \big ] = { \cal O } \left( \frac { B \| \mathbf { x } _ { 0 } - \mathbf { x } ^ { * } \| ^ { 2 } + G ^ { 2 } / B } { \sqrt { T + 1 } } \ln ( T + 2 ) \right) .
$$

(iii) (Anytime step size II) $\begin{array} { r } { I f \tau _ { t } = \frac { \tau } { \sqrt { t + 1 } } } \end{array}$ with $\begin{array} { r } { \tau < \frac { 1 } { 4 \sqrt { 2 } B } } \end{array}$ , then for any $T \geq 1$ , we have

$$
\mathbb { E } [ h ( { \mathbf x } _ { T + 1 } ) - h ^ { * } ] = \mathcal { O } \left( \frac { \| { \mathbf x } _ { 0 } - { \mathbf x } ^ { * } \| ^ { 2 } } { \tau \sqrt { T } } + \frac { \ln ^ { 2 } ( T + 1 ) } { T ^ { 1 / 2 - 1 6 \tau ^ { 2 } B ^ { 2 } } } \right) .
$$

The proof of Theorem 3.5 is given in Appendix A.3. The bounds with explicit constants for (i) and (ii) are given in (A.10) and (A.11), respectively.

Remark 3.6 The result of Theorem 3.5 extends trivially to projected SGD for solving a constrained problem, under (B-G), by setting $g = \delta c$

Remark 3.7 Specializing Remark 3.6 to the deterministic oracle $\widetilde { \nabla } f ( \mathbf { x } _ { t } ) : = \mathbf { s } _ { t } \in \partial f ( \mathbf { x } _ { t } )$ gives the (projected) subgradient method. In this case, (B-G) reduces to

$$
\| \mathbf { s } \| ^ { 2 } \leq B ^ { 2 } \| \mathbf { x } - \mathbf { x } _ { 0 } \| ^ { 2 } + G ^ { 2 } \qquad { \mathrm { f o r ~ a l l ~ } } \mathbf { x } \in { \mathcal { C } } { \mathrm { ~ a n d ~ a l l ~ } } \mathbf { s } \in \partial f ( \mathbf { x } ) ,\tag{3.4}
$$

and Theorem 3.5 gives the same rates without expectations. This yields last-iterate guarantees for convex functions that are not necessarily smooth or Lipschitz; an example is given in Section 4.1.

Rates. We now discuss the rates in Theorem 3.5. The horizon-dependent step size in (i) gives the rate $\mathcal { O } ( 1 / \sqrt { T } )$ with no logarithmic factors, which is the optimal order in $T .$ but it requires knowing $T$ in advance. The step size in (ii) does not depend on $T ,$ but this comes at the cost of an additional ln $( T + 2 )$ factor. The standard step size in (iii) is also independent of $T ,$ , but it is not square summable. As discussed after Proposition 3.2, the exponential factor therein grows as $T ^ { 1 6 \tau ^ { 2 } B ^ { 2 } }$ with this step size, which gives the slower rate. The condition $\tau < 1 / ( 4 \sqrt { 2 } B )$ ensures $1 6 \tau ^ { 2 } B ^ { 2 } < 1 / 2$ , so this upper bound still goes to zero. Taking $\tau$ smaller brings the exponent closer to $1 / 2$ , but increases the first term in the bound.

Comparison with prior work. Our results extend the works of Garrigos et al. (2025) and Vaidyan et al. (2026), who required (CW), which is stronger than our assumption in this section, and were limited to horizon-dependent step sizes. The other relevant work by Liu and Zhou (2024) handles variable step sizes but requires bounded variance and smoothness. Finally, Theorem 3.5 (i), which uses the step size scheme of Liu and Zhou (2024), removes the extra log factors that appear in the rates of Garrigos et al. (2025) and Vaidyan et al. (2026). This answers an open question posed by (Garrigos et al., 2025, Remark 3.6, Section 5).

## 4 Applications of the B-G Assumption

In this section, we present examples to argue that the BG condition (B-G) applies to a broader class of problems than the component-wise convexity and smoothness (CW).

## 4.1 Minimizing the Maximum of Quadratics

Consider the following nonsmooth problem given as

$$
\operatorname* { m i n } _ { \mathbf { x } \in \mathbb { R } ^ { d } } \phi ( \mathbf { x } ) : = \operatorname* { m i n } _ { \mathbf { x } \in \mathbb { R } ^ { d } } \Big \{ \operatorname* { m a x } _ { i = 1 , \ldots , n } \phi _ { i } ( \mathbf { x } ) \Big \} ,
$$

where each $\phi _ { i }$ is convex, diferentiable and satisfies the BG condition (B-G). That is, for all $\mathbf { x } \in \mathbb { R } ^ { d }$ , we assume

$$
\| \nabla \phi _ { i } ( \mathbf { x } ) \| ^ { 2 } \leq B ^ { 2 } \| \mathbf { x } - \mathbf { x } _ { 0 } \| ^ { 2 } + G ^ { 2 } .\tag{4.1}
$$

A concrete instance of this problem is the maximum of finitely many convex quadratics (Lu, 2019, Section 5.3):

$$
\phi _ { i } ( \mathbf { x } ) = \frac { 1 } { 2 } \mathbf { x } ^ { \top } A _ { i } \mathbf { x } + \mathbf { b } _ { i } ^ { \top } \mathbf { x } + c _ { i } ,
$$

where $A _ { i }$ is positive semi-definite. In this case, $\nabla \phi _ { i } ( \mathbf { x } ) = A _ { i } \mathbf { x } + \mathbf { b } _ { i } = A _ { i } ( \mathbf { x } - \mathbf { x } _ { 0 } ) + ( A _ { i } \mathbf { x } _ { 0 } + \mathbf { b } _ { i } )$ ， so each $\phi _ { i }$ satisfies the BG condition since

$$
\| \nabla \phi _ { i } ( \mathbf { x } ) \| ^ { 2 } \leq 2 \operatorname* { m a x } _ { i = 1 , \ldots , n } \| A _ { i } \| ^ { 2 } \| \mathbf { x } - \mathbf { x } _ { 0 } \| ^ { 2 } + 2 \operatorname* { m a x } _ { i = 1 , \ldots , n } \| A _ { i } \mathbf { x } _ { 0 } + \mathbf { b } _ { i } \| ^ { 2 } .
$$

Notice that this problem is generally nonsmooth and not globally Lipschitz. Hence, most existing results for subgradient-based algorithms in the literature, such as the works of Harvey et al. (2019); Jain et al. (2019); Zamani and Glineur (2025), do not apply. Lu (2019) solves this problem in the deterministic setting using mirror descent with a problemspecific generating function so that the resulting Bregman divergence controls the growth of the subgradients. This approach yields an $\mathcal { O } ( 1 / \sqrt { T } )$ convergence rate for the averaged iterate. In contrast, our analysis under (B-G) applies to the classical subgradient method, with no modification across diferent problems. Additionally, our analysis yields last-iterate guarantees, which Lu (2019) does not provide.

The subdiferential of $\phi$ is the convex hull of active gradients $\nabla \phi _ { i } ( \mathbf { x } )$ where i is in the set of active indices $I ( \mathbf { x } ) = \arg \operatorname* { m a x } _ { j = 1 , \dots , n } \phi _ { j } ( \mathbf { x } )$ . An unbiased stochastic subgradient in this case can be obtained by sampling i uniformly from the set of active indices and taking $\widetilde { \nabla } \phi ( \mathbf { x } ) = \nabla \phi _ { i } ( \mathbf { x } )$ as the stochastic subgradient since

$$
\mathbb { E } [ \widetilde { \nabla } \phi ( \mathbf { x } ) ] = \frac { 1 } { | I ( \mathbf { x } ) | } \sum _ { i \in I ( \mathbf { x } ) } \nabla \phi _ { i } ( \mathbf { x } ) \in \partial \phi ( \mathbf { x } ) .
$$

Additionally, notice that $\phi$ satisfies the BG condition since

$$
\mathbb { E } \| \widetilde { \nabla } \phi ( \mathbf { x } ) \| ^ { 2 } = \frac { 1 } { | I ( \mathbf { x } ) | } \sum _ { i \in I ( \mathbf { x } ) } \| \nabla \phi _ { i } ( \mathbf { x } ) \| ^ { 2 } \leq B ^ { 2 } \| \mathbf { x } - \mathbf { x } _ { 0 } \| ^ { 2 } + G ^ { 2 } ,
$$

where we used (4.1). This example shows that the BG condition can handle nonsmooth and non-Lipschitz objectives arising from pointwise maxima. In particular, it provides a setting in which last-iterate guarantees for the classical stochastic subgradient method can be obtained through our BG framework, while (CW) does not apply. In fact, we are not aware of a last-iterate rate even for the deterministic subgradient method for this problem.

## 4.2 Black-Box Variational Inference

In this section, we highlight another application of our results for black-box variational inference. Variational inference is a powerful technique for approximating an intractable posterior distribution, $p ,$ by using a tractable distribution from a given family. Throughout this section, we follow the setting of Domke et al. (2023) and refer to this work for a more detailed discussion of the variational formulation and Gaussian parameterizations.

Let x denote the observed data and $\mathbf { z } \in \mathbb { R } ^ { d }$ the latent variables, and let $p ( \mathbf { x } , \mathbf { z } )$ be their joint density. The goal is to approximate the posterior $p ( \mathbf { z } \mid \mathbf { x } )$ by a member of the Gaussian variational family

$$
\begin{array} { r } { q _ { \mathbf { w } } ( \mathbf { z } ) = \mathcal { N } ( \mathbf { z } \mid \mathbf { m } , C C ^ { \top } ) , \qquad \mathbf { w } = ( \mathbf { m } , C ) , } \end{array}
$$

where m $\in \mathbb { R } ^ { d }$ is the mean and $C C ^ { \top }$ is the covariance. This is done by minimizing the negative evidence lower bound (ELBO)

$$
h ( \mathbf { w } ) = f ( \mathbf { w } ) + g ( \mathbf { w } ) ,\tag{4.2}
$$

where we introduce the functions $f$ and $g$ next.

By using the reparameterization of $q _ { \mathbf { w } } ( \mathbf { z } )$ as $T _ { \mathbf { w } } ( \mathbf { u } ) = \mathbf { m } + C \mathbf { u }$ with $\mathbf { u } \sim \mathcal { N } ( 0 , I )$ , the energy term $f$ is given by

$$
f ( \mathbf { w } ) = \mathbb { E } _ { \mathbf { u } \sim \mathcal { N } ( 0 , I ) } \left[ - \ln p ( \mathbf { x } , T _ { \mathbf { w } } ( \mathbf { u } ) ) \right] ,\tag{4.3}
$$

and $g$ is the negative entropy term given by

$$
g ( { \bf w } ) = - \ln \operatorname* { d e t } C - { \frac { d } { 2 } } \ln ( 2 \pi e ) .
$$

Our goal is to minimize h over the parameters w of the variational family. We assume that

$$
\psi ( \mathbf { z } ) : = - \ln p ( \mathbf { x } , \mathbf { z } )
$$

is convex and L-smooth and that the variational problem admits a solution $\mathbf { w } ^ { * }$ . Under these assumptions, f is convex and L-smooth (see Domke et al. (2023, Lemmas 18 and 19)).

As in Domke et al. (2023, Section 3.1), we will apply proximal SGD to minimize (4.2) over a domain we introduce next. In particular, we restrict C defining the family to be lower triangular with a positive diagonal, which gives us the optimization domain

$$
\begin{array} { r } { \mathcal { W } _ { \mathbb { T } } : = \left\{ ( \mathbf { m } , C ) \in \mathbb { R } ^ { d } \times \mathbb { T } ^ { d } : C _ { i i } > 0 , \ i = 1 , \dots , d \right\} , } \end{array}
$$

where $\mathbb { T } ^ { d }$ is the set of $d \times d$ lower-triangular matrices. Then, the optimization problem becomes

$$
\operatorname* { m i n } _ { \mathbf { w } \in \mathcal { W } _ { \mathbb { T } } } \Big \{ f ( \mathbf { w } ) + g ( \mathbf { w } ) : = \mathbb { E } _ { \mathbf { u } \sim \mathcal { N } ( 0 , I ) } \left[ - \ln p ( \mathbf { x } , T _ { \mathbf { w } } ( \mathbf { u } ) ) \right] - \ln \operatorname* { d e t } C - \frac { d } { 2 } \ln ( 2 \pi e ) \Big \} .
$$

In this case, the unbiased estimator of the energy term is given by

$$
\widetilde { \nabla } f ( \mathbf { w } ; \mathbf { u } ) : = - \nabla _ { \mathbf { w } } \ln p ( \mathbf { x } , T _ { \mathbf { w } } ( \mathbf { u } ) ) , \qquad \mathbf { u } \sim { \mathcal { N } } ( 0 , I ) ,
$$

and hence $\mathbb { E } _ { \mathbf { u } \sim \mathcal { N } ( 0 , I ) } [ \widetilde { \nabla } f ( \mathbf { w } ; \mathbf { u } ) ] = \nabla f ( \mathbf { w } )$ . By Domke et al. (2023, Theorem 2), we have that

$$
\begin{array} { r } { \mathbb { E } \| \widetilde \nabla f ( \mathbf { w } ; \mathbf { u } ) \| ^ { 2 } \leq \beta _ { \mathrm { e n e r g y } } ^ { 2 } \| \mathbf { w } - \mathbf { w } ^ { * } \| ^ { 2 } + \gamma _ { \mathrm { e n e r g y } } ^ { 2 } , } \end{array}\tag{4.4}
$$

for uniform constants β<sub>energy</sub>, γ<sub>energy</sub>, and any $\mathbf { w } ^ { * }$ in the solution set.

For $C \in \mathbb { T } ^ { d }$ , Domke (2020, Theorem 13) (see also Domke et al., 2023, Lemma 35) shows that the proximal operator for $g + \delta _ { \mathcal { W _ { \mathbb { T } } } }$ (needed to fit the problem to our composite template) is computed as

$$
\mathrm { p r o x } _ { \tau _ { t } ( g + \delta \nu _ { \mathbb { T } } ) } ( \mathbf { w } ) = ( \mathbf { m } , C + \Delta C ) ; \qquad \Delta C _ { i i } = \frac { 1 } { 2 } \left( \sqrt { C _ { i i } ^ { 2 } + 4 \tau _ { t } } - C _ { i i } \right) .
$$

Remark 4.1 Applying Fact 1 to (4.4) with $\mathbf { a } = \mathbf { w } ^ { * }$ and $\mathbf b = \mathbf w _ { 0 }$ shows that $\widetilde { \nabla } f$ satisfies (B-G). Hence, Theorem 3.5 applies, giving an anytime last-iterate rate of $\mathcal { O } ( \ln T / \sqrt { T } )$ for proximal SGD. Unlike Domke et al. (2023); Kim et al. (2025), we do not require strong convexity to get an anytime rate. When the horizon $T$ is known, the rate $\mathcal { O } ( 1 / \sqrt { T } )$ follows from Theorem 3.5.

Remark 4.2 Our results also cover the constrained formulation of the BBVI problem, which was described in Domke et al. (2023, Section 3.2). In this setting, the stochastic gradient satisfies (B-G) (Domke et al., 2023, Theorem 3), so Remark 3.6 gives the same last-iterate rates for projected SGD.

The key diference between the guarantees provided by Remark 4.1 and Remark 4.2, and those of Domke et al. (2023) is that our results provide last-iterate convergence guarantees in the convex setting, whereas Domke et al. (2023) establish convergence guarantees for weighted averages of the iterates. In addition, our results provide anytime last-iterate guarantees using the step size scheme $\tau _ { t } = \tau / ( \ln ( t + 2 ) \sqrt { t + 2 } )$ , whereas the parameters in Domke et al. (2023) depend on a fixed horizon T. As stated in Kim et al. (2025, Section 2.4), it was an open question to derive anytime bounds for this setting, which our results in Theorem 3.5 and Remark 3.6 address.

## 5 Convergence Analysis under CW

In this section, we analyze the last-iterate convergence of projected and proximal SGD under the assumptions of component-wise convexity and smoothness in (CW). Last-iterate guarantees in this setting were first obtained for SGD by Garrigos et al. (2025) and Attia et al. (2025). This analysis was then extended to proximal SGD by Vaidyan et al. (2026). All of these analyses use the variance transfer bound (1.6) with a reference point $\mathbf { z } _ { t }$ in the style of Zamani and Glineur (2025). This gives the one-step inequality

$$
\mathbb { E } \left[ h ( \mathbf { x } _ { t + 1 } ) - h ( \mathbf { z } _ { t } ) \right] \leq \frac { 1 } { 2 \tau _ { t } } \mathbb { E } \left[ \| \mathbf { x } _ { t } - \mathbf { z } _ { t } \| ^ { 2 } - \| \mathbf { x } _ { t + 1 } - \mathbf { z } _ { t } \| ^ { 2 } \right] + 4 \tau _ { t } L \mathbb { E } \left[ h ( \mathbf { x } _ { t } ) - h ^ { * } \right] + 2 \tau _ { t } \sigma _ { * } ^ { 2 } .
$$

In these works, $\tau _ { t } = \tau$ is fixed. Compared with the bounded second moment case (Liu and Zhou, 2024, Eq. 6), there is an extra term $\rho \mathbb { E } [ h ( \mathbf { x } _ { t } ) - h ^ { * } ]$ with $\rho = 4 \tau L$ on the right-hand side, and the weighted sum used by Zamani and Glineur (2025) no longer cancels exactly. Vaidyan et al. (2026) handle the extra term with a constant step size $\tau _ { t } = \tau$ . In this case, $\rho$ does not depend on t, and the weights of the convex combinations defining $\mathbf { z } _ { t }$ can be adjusted to absorb $\rho \mathbb { E } [ h ( \mathbf { x } _ { t } ) - h ^ { * } ]$ (see Vaidyan et al., 2026, Lemma A.6). However, the argument relies on both the step size and $\rho$ being constant. It therefore does not cover horizon-free decreasing step sizes such as $\tau _ { t } = \tau / \sqrt { t + 1 }$ , which give anytime guarantees. It also does not cover the time-varying schedule of Liu and Zhou (2024), which removes the logarithmic factor in the bounded variance setting.

Throughout this section, we require Assumptions 1 and 2(b). As in Section 3, we follow the same roadmap, beginning with an inequality that controls the second moment of the stochastic gradients; see Khaled et al. (2023, Lemma A.3).

Lemma 5.1 (Variance transfer) Let Assumptions 1 and $\mathcal { Q } ( b )$ hold. Then for every $\gamma > 0$ and $\mathbf { x } \in \mathbb { R } ^ { d }$ , we have

$$
\mathbb { E } \| \nabla f _ { i } ( { \mathbf x } ) \| ^ { 2 } \leq 2 L ( 1 + \gamma ) [ h ( { \mathbf x } ) - h ^ { * } ] + \left( 1 + \frac { 1 } { \gamma } \right) \sigma _ { * } ^ { 2 } .\tag{5.1}
$$

Since every $f _ { i }$ is L-smooth under Assumption 2(b), it follows that f is L-smooth. This allows us to use the following classical one-step inequality for proximal SGD.

Lemma 5.2 Let Assumptions 1 and $\boldsymbol { \mathcal { Z } } ( \boldsymbol { b } )$ hold, and set $\tau _ { t } \leq \frac { 1 } { 2 L }$ . For all $t \geq 0$ and any F<sub>t</sub>-measurable $\mathbf { z } _ { t }$ , we have

$$
2 \tau _ { t } \mathbb { E } [ h ( \mathbf { x } _ { t + 1 } ) - h ( \mathbf { z } _ { t } ) ] \leq \mathbb { E } \| \mathbf { x } _ { t } - \mathbf { z } _ { t } \| ^ { 2 } - \mathbb { E } \| \mathbf { x } _ { t + 1 } - \mathbf { z } _ { t } \| ^ { 2 } + 2 \tau _ { t } ^ { 2 } \mathbb { E } \| \widetilde { \nabla } f ( \mathbf { x } _ { t } ) \| ^ { 2 } .
$$

The proof of Lemma 5.2 is given in Appendix C.2. We now analyze one step of proximal SGD under (CW).

Lemma 5.3 (One-step of proximal SGD under (CW)) Let Assumptions 1 and $\boldsymbol { \mathcal { Z } } ( \boldsymbol { b } )$ hold, and set $\begin{array} { r } { \tau _ { t } \le \frac { 1 } { 4 L } } \end{array}$ for all t. Then for all $t \geq 0$ and every $\mathcal { F } _ { t }$ -measurable $\mathbf { z } _ { t } ,$ , (proxSGD) satisfies

$$
\mathbb { E } \left[ h ( \mathbf { x } _ { t + 1 } ) - h ( \mathbf { z } _ { t } ) - \rho _ { t } h ( \mathbf { x } _ { t } ) + \rho _ { t } h ^ { * } \right] \leq \frac { 1 } { 2 \tau _ { t } } \mathbb { E } \left[ \Vert \mathbf { x } _ { t } - \mathbf { z } _ { t } \Vert ^ { 2 } - \Vert \mathbf { x } _ { t + 1 } - \mathbf { z } _ { t } \Vert ^ { 2 } \right] + v _ { t } ,
$$

where $\rho _ { t } = 4 \tau _ { t } L$ and $v _ { t } = 2 \sigma _ { * } ^ { 2 } \tau _ { t }$

The proof of Lemma 5.3 is given in Appendix B.1. Using the above lemma with the proximal reduction in Lemma 3.4, we can obtain last-iterate guarantees for proximal SGD with decreasing step sizes.

Theorem 5.4 (Proximal SGD under (CW)) Let Assumptions 1 and $\boldsymbol { \mathcal { Z } } ( \boldsymbol { b } )$ hold, and set $\begin{array} { r } { \tau = \frac { 1 } { 4 L } } \end{array}$ . Then the following guarantees hold for the last iterate of (proxSGD):

(i) (Step size for log-free rate) $\begin{array} { r } { I f \tau _ { t } = \tau \frac { T - t + 1 } { ( T + 1 ) ^ { 3 / 2 } } } \end{array}$ , then for any $T \geq 1$ , we have

$$
\mathbb { E } [ h ( { \mathbf { x } } _ { T + 1 } ) - h ^ { * } ] = { \mathcal { O } } \left( \frac { 1 } { \sqrt { T + 1 } } \left[ L \| { \mathbf { x } } _ { 0 } - { \mathbf { x } } ^ { * } \| ^ { 2 } + \frac { \sigma _ { * } ^ { 2 } } { L } + \frac { h ( { \mathbf { x } } _ { 0 } ) - h ^ { * } } { T + 1 } \right] \right) .
$$

(ii) (Anytime step size) $\begin{array} { r } { I f \tau _ { t } = \frac { \tau } { \sqrt { t + 1 } } } \end{array}$ , then for any $T \geq 1$ , we have

$$
\mathbb { E } [ h ( { \mathbf x } _ { T + 1 } ) - h ^ { * } ] = \mathcal { O } \left( \frac { 1 } { \sqrt { T + 1 } } \left[ L \| { \mathbf x } _ { 0 } - { \mathbf x } ^ { * } \| ^ { 2 } + \frac { \sigma _ { * } ^ { 2 } \ln ( T + 1 ) } { L } + h ( { \mathbf x } _ { 0 } ) - h ^ { * } \right] \right) .
$$

The proof of Theorem 5.4 is given in Appendix B.2. The bounds with explicit constants for (i) and (ii) are given in (B.2) and (B.3), respectively.

Remark 5.5 The result of Theorem 5.4 extends trivially to projected SGD for solving a constrained problem, under (CW), by setting $g = \delta c$

Rates. The horizon-dependent step size in Theorem 5.4(i) gives the rate $\mathcal { O } ( 1 / \sqrt { T } )$ with no logarithmic factors, which is the optimal order in T, but it requires knowing T in advance. The standard step size in (ii) does not depend on T, but this incurs an additional cost of a ln(T + 1) factor, which multiplies the noise term $\sigma _ { * } ^ { 2 } / L$

Comparison with prior work. We extended the work of Vaidyan et al. (2026) by generalizing its analysis to a decreasing step size scheme that is not horizon-dependent. Additionally, the step size in Theorem 5.4(i), which follows Liu and Zhou (2024), removes the logarithmic factor, answering a question posed by Garrigos et al. (2025). This is a strict improvement over the results in Vaidyan et al. (2026), which require a horizon-dependent step size scheme and obtain a rate of $\mathcal { O } ( \ln ( T ) / \sqrt { T } )$

Remark 5.6 While the BG condition (B-G) is the weaker assumption, this generality comes at the cost of step size flexibility. Under the BG condition, the standard anytime step size $\tau _ { t } = \tau / \sqrt { t + 1 }$ only yields a suboptimal rate (see Theorem 3.5(iii)), while under component-wise convexity and smoothness (CW) the same step size attains the optimal order up to a logarithmic factor (see Theorem 5.4(ii)).

## 6 Conclusion and Perspectives

In this paper, we studied last-iterate convergence rates of projected and proximal SGD under assumptions weaker than the classical uniformly bounded second moment condition (1.4). We considered two diferent mechanisms for controlling the stochastic oracle. The first is based on the Blum-Gladyshev condition (B-G), which controls the stochastic gradients through the distance to the initial iterate. The second is based on component-wise convexity and smoothness (CW), which leads to ABC-type bounds (1.6) on the second moment of the stochastic gradients. Our work opens up several directions for future research. For example, under the BG condition, our guarantee for the standard anytime step size $\tau _ { t } = \tau / \sqrt { t + 1 }$ is suboptimal. It would be interesting to investigate whether this can be improved. Another direction is expanding our analysis to handle Bregman distances and establishing high probability guarantees on the last iterate.

## AI Disclosure Statement

After completing the proofs and the writing of the paper, GPT 6.0 (OpenAI) and Claude Opus 5.5 (Anthropic) were used for various copyediting tasks and for checking the proofs in this manuscript. For proof-checking, these tools were used to review derivations for errors such as incorrect indices, missing steps, or inconsistent constants. All suggested changes were reviewed and verified by the authors.

Our analysis of proximal SGD under (B-G) initially required f to be smooth, and we had a separate section for projected SGD under (B-G) where f was not necessarily smooth. When investigating whether smoothness could be removed for proximal SGD, we used these tools to explore whether such an extension was feasible, and their output suggested that it was. We then derived and wrote the proof of the nonsmooth result for proximal SGD, which allowed us to merge the separate sections for proximal and projected SGD. These tools were also used to assist in deriving the step size bounds in Lemma C.3. The resulting bounds and their proofs were verified by us.

## Acknowledgments

We acknowledge the support of the Natural Sciences and Engineering Research Council of Canada (NSERC), funding reference numbers RGPIN-2023-05250 (Friedlander) and RGPIN-2025-06634 (Alacaoglu).

## Appendix A. Proofs for Section 3

In this section, we provide the proofs of the various results in Section 3. We start by establishing the boundedness of the iterates under (B-G), which is a key step in our analysis.

## A.1 Proof of Proposition 3.2

Proof We start from the hypothesis (3.3) which states that

$$
\begin{array} { r } { \mathbb { E } \| { \mathbf { x } } _ { t + 1 } - { \mathbf { x } } ^ { * } \| ^ { 2 } \leq \mathbb { E } \| { \mathbf { x } } _ { t } - { \mathbf { x } } ^ { * } \| ^ { 2 } + \tau _ { t } ^ { 2 } \mathbb { E } \| { \mathbf { q } } _ { t + 1 } - \widetilde { \nabla } f ( { \mathbf { x } } _ { t } ) \| ^ { 2 } . } \end{array}\tag{A.1}
$$

Using Young’s inequality on the last term, we have

$$
\begin{array} { r } { \mathbb { E } \| \mathbf { q } _ { t + 1 } - \widetilde { \nabla } f ( \mathbf { x } _ { t } ) \| ^ { 2 } \leq 2 \mathbb { E } \| \mathbf { q } _ { t + 1 } \| ^ { 2 } + 2 \mathbb { E } \| \widetilde { \nabla } f ( \mathbf { x } _ { t } ) \| ^ { 2 } . } \end{array}\tag{A.2}
$$

We bound the second term on the right-hand side of (A.2) using the BG condition after recentering (3.2), which gives $\begin{array} { r } { \mathbb { E } \| \widetilde { \nabla } f ( \mathbf { x } _ { t } ) \| ^ { 2 } \leq 2 B ^ { 2 } \mathbb { E } \| \mathbf { x } _ { t } - \mathbf { x } ^ { * } \| ^ { 2 } + \widehat { G } ^ { 2 } } \end{array}$ . For the first term on the right-hand side of (A.2), we use Jensen’s inequality and (3.2) at $\mathbf { x } _ { t + 1 }$ to obtain

$$
\begin{array} { r } { \| \mathbf { q } _ { t + 1 } \| ^ { 2 } = \| \mathbb { E } _ { t + 1 } [ \widetilde { \nabla } f ( \mathbf { x } _ { t + 1 } ) ] \| ^ { 2 } \leq \mathbb { E } _ { t + 1 } \| \widetilde { \nabla } f ( \mathbf { x } _ { t + 1 } ) \| ^ { 2 } \leq 2 B ^ { 2 } \| \mathbf { x } _ { t + 1 } - \mathbf { x } ^ { * } \| ^ { 2 } + \widehat { G } ^ { 2 } . } \end{array}
$$

Plugging the above bounds into (A.2), then back into (A.1) and taking total expectation, we get

$$
\begin{array} { r } { \mathbb { E } \| { \bf x } _ { t + 1 } - { \bf x } ^ { * } \| ^ { 2 } \leq ( 1 + 4 \tau _ { t } ^ { 2 } B ^ { 2 } ) \mathbb { E } \| { \bf x } _ { t } - { \bf x } ^ { * } \| ^ { 2 } + 4 \tau _ { t } ^ { 2 } B ^ { 2 } \mathbb { E } \| { \bf x } _ { t + 1 } - { \bf x } ^ { * } \| ^ { 2 } + 4 \tau _ { t } ^ { 2 } \widehat G ^ { 2 } . } \end{array}
$$

Rearranging and simplifying yields

$$
\begin{array} { r l r } & { } & { \mathbb { E } \| { \bf x } _ { t + 1 } - { \bf x } ^ { * } \| ^ { 2 } \leq \displaystyle \frac { 1 + 4 \tau _ { t } ^ { 2 } B ^ { 2 } } { 1 - 4 \tau _ { t } ^ { 2 } B ^ { 2 } } \mathbb { E } \| { \bf x } _ { t } - { \bf x } ^ { * } \| ^ { 2 } + \frac { 4 \tau _ { t } ^ { 2 } \widehat G ^ { 2 } } { 1 - 4 \tau _ { t } ^ { 2 } B ^ { 2 } } } \\ & { } & { \leq ( 1 + 1 6 \tau _ { t } ^ { 2 } B ^ { 2 } ) \mathbb { E } \| { \bf x } _ { t } - { \bf x } ^ { * } \| ^ { 2 } + 8 \tau _ { t } ^ { 2 } \widehat G ^ { 2 } , } \end{array}
$$

where the second line used $\textstyle { \frac { 1 + a } { 1 - a } } = 1 + { \frac { 2 a } { 1 - a } }$ and $\textstyle { \frac { 1 } { 1 - a } } \leq 2$ where $a = 4 \tau _ { t } ^ { 2 } B ^ { 2 }$ because $4 \tau _ { t } ^ { 2 } B ^ { 2 } \leq$ $1 / 2 . ^ { 1 }$ Unrolling the recursion above, we get the bound

$$
\mathbb { E } \Vert { \mathbf { x } } _ { t + 1 } - { \mathbf { x } } ^ { * } \Vert ^ { 2 } \leq \prod _ { k = 0 } ^ { t } ( 1 + 1 6 \tau _ { k } ^ { 2 } B ^ { 2 } ) \Vert { \mathbf { x } } _ { 0 } - { \mathbf { x } } ^ { * } \Vert ^ { 2 } + 8 \sum _ { k = 0 } ^ { t } \left( \prod _ { j = k + 1 } ^ { t } ( 1 + 1 6 \tau _ { j } ^ { 2 } B ^ { 2 } ) \right) \tau _ { k } ^ { 2 } { \widehat G } ^ { 2 } .\tag{A.3}
$$

Now we bound the product on the right-hand side of this bound as

$$
\prod _ { k = 0 } ^ { t } ( 1 + 1 6 \tau _ { k } ^ { 2 } B ^ { 2 } ) \leq \prod _ { k = 0 } ^ { t } \exp ( 1 6 \tau _ { k } ^ { 2 } B ^ { 2 } ) = \exp \left( 1 6 \sum _ { k = 0 } ^ { t } \tau _ { k } ^ { 2 } B ^ { 2 } \right) \leq \exp \left( 1 6 P _ { t } B ^ { 2 } \right) ,
$$

where we used $1 + x \leq e ^ { x }$ and $\textstyle \sum _ { k = 0 } ^ { t } \tau _ { k } ^ { 2 } = P _ { t }$ . Substituting this back into (A.3), we get

$$
\begin{array} { r } { \mathbb { E } \| { \bf x } _ { t + 1 } - { \bf x } ^ { * } \| ^ { 2 } \leq \exp \left( 1 6 P _ { t } B ^ { 2 } \right) \left[ \| { \bf x } _ { 0 } - { \bf x } ^ { * } \| ^ { 2 } + 8 P _ { t } \widehat { G } ^ { 2 } \right] . } \end{array}\tag{A.4}
$$

This is the first assertion. The second assertion follows immediately by plugging the bound on the iterates from (A.4) into the BG condition (3.2). 7

We next prove a one-step progress bound for proximal SGD under (B-G).

## A.2 Proof of Lemma 3.3

Proof We apply Lemma 3.1 with $\mathbf { q } _ { t + 1 } : = \mathbb { E } _ { t + 1 } [ \widetilde { \nabla } f ( \mathbf { x } _ { t + 1 } ) ] \in \partial f ( \mathbf { x } _ { t + 1 } )$ to obtain

$$
\begin{array} { r } { 2 \tau _ { t } \mathbb { E } [ h ( \mathbf { x } _ { t + 1 } ) - h ( \mathbf { z } _ { t } ) ] \leq \mathbb { E } \| \mathbf { x } _ { t } - \mathbf { z } _ { t } \| ^ { 2 } - \mathbb { E } \| \mathbf { x } _ { t + 1 } - \mathbf { z } _ { t } \| ^ { 2 } + \tau _ { t } ^ { 2 } \mathbb { E } \| \mathbf { q } _ { t + 1 } - \widetilde { \nabla } f ( \mathbf { x } _ { t } ) \| ^ { 2 } . } \end{array}\tag{A.5}
$$

As in the proof of Proposition 3.2, we use Young’s inequality and Jensen’s inequality to bound the last term as

$$
\begin{array} { r } { \mathbb { E } \| \mathbf { q } _ { t + 1 } - \widetilde { \nabla } f ( \mathbf { x } _ { t } ) \| ^ { 2 } \leq 2 \mathbb { E } \| \mathbf { q } _ { t + 1 } \| ^ { 2 } + 2 \mathbb { E } \| \widetilde { \nabla } f ( \mathbf { x } _ { t } ) \| ^ { 2 } \leq 2 \mathbb { E } \| \widetilde { \nabla } f ( \mathbf { x } _ { t + 1 } ) \| ^ { 2 } + 2 \mathbb { E } \| \widetilde { \nabla } f ( \mathbf { x } _ { t } ) \| ^ { 2 } . } \end{array}
$$

By Proposition 3.2, we have that $\mathbb { E } \| \widetilde { \nabla } f ( \mathbf { x } _ { t } ) \| ^ { 2 } \leq K _ { t - 1 } \leq K _ { t } .$ , since $P _ { t }$ is non-decreasing, and also that $\mathbb { E } \| \widetilde { \nabla } f ( \mathbf { x } _ { t + 1 } ) \| ^ { 2 } \leq K _ { t }$ . Hence, the right-hand side is bounded by $4 K _ { t }$ . Substituting this bound to (A.5) and dividing by $2 \tau _ { t }$ gives us the desired result.

We proceed to obtain the last-iterate convergence rate of proximal SGD under (B-G) with various step size schemes.

## A.3 Proof of Theorem 3.5

Proof In all three step size schemes, $\left( \tau _ { t } \right)$ is positive and non-increasing, and $\begin{array} { r } { \tau _ { t } \le \tau _ { 0 } \le \frac { 1 } { 2 \sqrt { 2 } B } } \end{array}$ In particular, for (i), we have $\begin{array} { r } { \tau _ { 0 } = \frac { \tau } { \sqrt { T + 1 } } \le \frac { 1 } { 4 B } } \end{array}$ ; for (ii), we have $\begin{array} { r } { \tau _ { 0 } = \frac { \tau } { \sqrt { 2 } \ln { 2 } } \le \frac { 1 } { 2 \sqrt { 2 } B } } \end{array}$ since $4 { \sqrt { 3 } } \ln 2 \geq 2$ ; and for (iii), we have $\begin{array} { r } { \tau _ { 0 } = \tau < \frac { 1 } { 4 \sqrt { 2 } B } } \end{array}$ . Hence, we can apply Lemma 3.3 to get

$$
\mathbb { E } [ h ( \mathbf { x } _ { t + 1 } ) - h ( \mathbf { z } _ { t } ) ] \leq \frac { 1 } { 2 \tau _ { t } } \mathbb { E } [ \| \mathbf { x } _ { t } - \mathbf { z } _ { t } \| ^ { 2 } - \| \mathbf { x } _ { t + 1 } - \mathbf { z } _ { t } \| ^ { 2 } ] + 2 \tau _ { t } K _ { t } ,\tag{A.6}
$$

where $K _ { t }$ and $P _ { t }$ are defined in Proposition 3.2. Since $P _ { t }$ is non-decreasing, we have that $P _ { t } \leq P _ { T }$ , and hence, $K _ { t } \le K _ { T }$ . With this, we can upper bound the last term in (A.6) by $2 \tau _ { t } K _ { T }$ . We can now apply Lemma 3.4 with $\rho _ { t } \equiv 0$ and $v _ { t } = 2 \tau _ { t } K _ { T }$ to obtain

$$
\mathbb { E } [ h ( { \mathbf x } _ { T + 1 } ) - h ^ { * } ] \leq \frac { \| { \mathbf x } _ { 0 } - { \mathbf x } ^ { * } \| ^ { 2 } } { 2 S _ { 0 } } + 2 K _ { T } \sum _ { t = 0 } ^ { T } \frac { \tau _ { t } ^ { 2 } } { S _ { t } } .\tag{A.7}
$$

For the first two step size schemes, the corresponding bounds from Lemmas C.2 and C.3, respectively, give

<table><tr><td>Step size</td><td> $S _ { 0 }$ </td><td> $\sum _ { t = 0 } ^ { T } \frac { \tau _ { t } ^ { 2 } } { S _ { t } }$ </td><td> $P _ { T }$ </td></tr><tr><td>(i)  $\begin{array} { r } { \tau _ { t } = \tau \frac { T - t + 1 } { ( T + 1 ) ^ { 3 / 2 } } } \end{array}$ </td><td> $\geq { \frac { \tau { \sqrt { T + 1 } } } { 2 } }$ </td><td> $\leq \frac { 2 \tau } { \sqrt { T + 1 } }$ </td><td> $\leq \tau ^ { 2 }$ </td></tr><tr><td>(ii) τt = T ln(t+2)√t+2</td><td>τ√T +1 ≥ √2ln(T + 2)</td><td>6√2τ ln(T + 2) + 33τ ≤ √T+1</td><td>≤ 3τ2.</td></tr></table>

(A.8)

Since $\tau = 1 / ( 4 B )$ in (i) and $\tau = 1 / ( 4 \sqrt { 3 } B )$ in (ii), the bounds on $P _ { T }$ in (A.8) give $1 6 P _ { T } B ^ { 2 } \leq$ 1 for both cases. Using this bound in the definition of $K _ { T }$ (in Prop. 3.2) and then expanding the definition of $\widehat { G } ^ { 2 }$ from (3.2), we get

$$
K _ { T } \le 2 ( 2 e + 1 ) B ^ { 2 } \| { \bf x } _ { 0 } - { \bf x } ^ { * } \| ^ { 2 } + ( 1 + e ) G ^ { 2 } = : K .\tag{A.9}
$$

Substituting the bounds from (A.8) and (A.9) into (A.7), we obtain

(i)

$$
\mathbb { E } \big [ h ( { \mathbf x } _ { T + 1 } ) - h ^ { * } \big ] \leq \frac { 1 } { \sqrt { T + 1 } } \left( \frac { \| { \mathbf x } _ { 0 } - { \mathbf x } ^ { * } \| ^ { 2 } } { \tau } + 4 \tau K \right) ,\tag{A.10}
$$

$$
\mathrm { ( i i ) } \quad \mathbb { E } [ h ( \mathbf { x } _ { T + 1 } ) - h ^ { * } ] \leq \frac { 1 } { \sqrt { T + 1 } } \left[ \frac { \ln ( T + 2 ) } { \sqrt { 2 } \tau } \| \mathbf { x } _ { 0 } - \mathbf { x } ^ { * } \| ^ { 2 } + 2 \tau K \left( 6 \sqrt { 2 } \ln ( T + 2 ) + 3 3 \right) \right] .\tag{A.11}
$$

In both cases, $\tau = \Theta ( 1 / B )$ . Substituting this and the definition of K in (A.9) into (A.10) and (A.11), and suppressing constants, yields the first two claims of the theorem.

For the final step size scheme, Lemma C.1 gives us the bounds

$$
S _ { 0 } \geq 2 \tau ( \sqrt { T + 2 } - 1 ) , \qquad \sum _ { t = 0 } ^ { T } \frac { \tau _ { t } ^ { 2 } } { S _ { t } } \leq \frac { 2 \tau ( 1 + \ln ( T + 1 ) ) } { \sqrt { T + 2 } } , \qquad P _ { T } \leq \tau ^ { 2 } ( 1 + \ln ( T + 1 ) ) .
$$

Substituting the above bounds into (A.7), we get

$$
\mathbb { E } [ h ( \mathbf { x } _ { T + 1 } ) - h ^ { * } ] \leq \frac { \| \mathbf { x } _ { 0 } - \mathbf { x } ^ { * } \| ^ { 2 } } { 4 \tau ( \sqrt { T + 2 } - 1 ) } + K _ { T } \frac { 4 \tau ( 1 + \ln ( T + 1 ) ) } { \sqrt { T + 2 } } .\tag{A.12}
$$

Using the bound on $P _ { T }$ , we estimate $K _ { T }$ (see Prop. 3.2 for its definition) as follows:

$$
\begin{array} { r l } & { K _ { T } \leq 2 B ^ { 2 } e ^ { 1 6 \tau ^ { 2 } B ^ { 2 } } ( T + 1 ) ^ { 1 6 \tau ^ { 2 } B ^ { 2 } } \left( \| \mathbf { x } _ { 0 } - \mathbf { x } ^ { * } \| ^ { 2 } + 8 \tau ^ { 2 } ( 1 + \ln ( T + 1 ) ) \widehat { G } ^ { 2 } \right) + \widehat { G } ^ { 2 } } \\ & { \qquad = \mathcal { O } \left( T ^ { 1 6 \tau ^ { 2 } B ^ { 2 } } \ln ( T + 1 ) \right) . } \end{array}
$$

Substituting the bound on $K _ { T }$ into (A.12) and suppressing constants give

$$
\mathbb { E } [ h ( { \bf x } _ { T + 1 } ) - h ^ { * } ] = { \cal O } \bigg ( \frac { \| { \bf x } _ { 0 } - { \bf x } ^ { * } \| ^ { 2 } } { { \tau \sqrt { T } } } + T ^ { - 1 / 2 + 1 6 \tau ^ { 2 } B ^ { 2 } } ( \ln ( T + 1 ) ) ^ { 2 } \bigg ) .
$$

In particular, for $\tau < 1 / ( 4 \sqrt { 2 } B )$ , we have $- 1 / 2 + 1 6 \tau ^ { 2 } B ^ { 2 } < 0$ , and hence $\mathbb { E } [ h ( { \mathbf { x } } _ { T + 1 } ) - h ^ { * } ]  0$ as $T \to \infty$ . This concludes the proof of the theorem.

## Appendix B. Proofs for Section 5

In this section, we provide the proofs of the various results in Section 5. We start by establishing a one-step progress bound for proximal SGD under (CW).

## B.1 Proof of Lemma 5.3

Proof We start from the result of Lemma 5.2 using $\mathbf { z } = \mathbf { z } _ { t }$ to obtain,

$$
2 \tau _ { t } \mathbb { E } [ h ( \mathbf { x } _ { t + 1 } ) - h ( \mathbf { z } _ { t } ) ] \leq \mathbb { E } \| \mathbf { x } _ { t } - \mathbf { z } _ { t } \| ^ { 2 } - \mathbb { E } \| \mathbf { x } _ { t + 1 } - \mathbf { z } _ { t } \| ^ { 2 } + 2 \tau _ { t } ^ { 2 } \mathbb { E } \| \widetilde { \nabla } f ( \mathbf { x } _ { t } ) \| ^ { 2 } .
$$

Dividing throughout by $2 \tau _ { t }$ and using (5.1) with $\gamma = 1$ , we get

$$
\mathbb { E } [ h ( { \bf x } _ { t + 1 } ) - h ( { \bf z } _ { t } ) ] \leq \frac { 1 } { 2 \tau _ { t } } \mathbb { E } [ \| { \bf x } _ { t } - { \bf z } _ { t } \| ^ { 2 } - \| { \bf x } _ { t + 1 } - { \bf z } _ { t } \| ^ { 2 } ] + 4 \tau _ { t } L \mathbb { E } [ h ( { \bf x } _ { t } ) - h ^ { * } ] + 2 \tau _ { t } \sigma _ { * } ^ { 2 } .
$$

Rearranging terms, we obtain the desired result.

We now proceed to show the last-iterate convergence rate of proximal SGD under (CW) with various step size schemes.

## B.2 Proof of Theorem 5.4

Proof From Lemma 5.3, we have for all $t = 0 , \ldots , T$ and $\mathbf { z } _ { t } \in \mathcal { F } _ { t }$

$$
\mathbb { E } \left[ h ( \mathbf { x } _ { t + 1 } ) - h ( \mathbf { z } _ { t } ) - \rho _ { t } h ( \mathbf { x } _ { t } ) + \rho _ { t } h ^ { * } \right] \leq \frac { 1 } { 2 \tau _ { t } } \mathbb { E } \left[ \Vert \mathbf { x } _ { t } - \mathbf { z } _ { t } \Vert ^ { 2 } - \Vert \mathbf { x } _ { t + 1 } - \mathbf { z } _ { t } \Vert ^ { 2 } \right] + v _ { t } ,
$$

where $\rho _ { t } = 4 \tau _ { t } L$ and $v _ { t } = 2 \sigma _ { * } ^ { 2 } \tau _ { t }$ . Since $\tau _ { t } \leq \tau$ for both step size schemes and $\tau L \leq 1 / 4$ , we have $\rho _ { t } \le 1$ . Hence, Lemma 3.4 gives

$$
\mathbb { E } [ h ( { \mathbf x } _ { T + 1 } ) - h ^ { * } ] \le \frac { 1 } { 2 } \exp \left( \sum _ { t = 1 } ^ { T } \frac { \tau _ { t } \rho _ { t } } { S _ { t } } \right) \left( \frac { \| { \mathbf x } _ { 0 } - { \mathbf x } ^ { * } \| ^ { 2 } } { S _ { 0 } } + 4 \sigma _ { * } ^ { 2 } \sum _ { t = 0 } ^ { T } \frac { \tau _ { t } ^ { 2 } } { S _ { t } } + \frac { 2 \rho _ { 0 } \tau _ { 0 } [ h ( { \mathbf x } _ { 0 } ) - h ^ { * } ] } { S _ { 0 } } \right) .\tag{B.1}
$$

We first bound the exponential term, by the definition $\rho _ { t } = 4 \tau _ { t } L$ , as

$$
\exp \left( \sum _ { t = 1 } ^ { T } \frac { \tau _ { t } \rho _ { t } } { S _ { t } } \right) = \exp \left( 4 L \sum _ { t = 1 } ^ { T } \frac { \tau _ { t } ^ { 2 } } { S _ { t } } \right) .
$$

From Lemmas C.1 and C.2, we obtain the following bounds for the two step size schemes:

<table><tr><td rowspan=1 colspan=1>Step size</td><td rowspan=1 colspan=1> $S _ { 0 }$ </td><td rowspan=1 colspan=1> $\sum _ { t = 0 } ^ { T } \frac { \tau _ { t } ^ { 2 } } { S _ { t } }$ </td><td rowspan=1 colspan=1> $\exp \left( 4 L \sum _ { t = 1 } ^ { T } \frac { \tau _ { t } ^ { 2 } } { S _ { t } } \right)$ </td></tr><tr><td rowspan=1 colspan=1>(i) $\begin{array} { r } { \tau _ { t } = \tau \frac { T - t + 1 } { ( T + 1 ) ^ { 3 / 2 } } } \end{array}$ (ii) $\begin{array} { r } { \tau _ { t } = \frac { \tau } { \sqrt { t + 1 } } } \end{array}$ </td><td rowspan=1 colspan=1> $\geq { \frac { \tau { \sqrt { T + 1 } } } { 2 } }$  $\geq 2 \tau ( \sqrt { T + 2 } - 1 )$ </td><td rowspan=1 colspan=1> $\leq \frac { 2 \tau } { \sqrt { T + 1 } }$  $\leq \frac { 2 \tau ( 1 + \ln ( T + 1 ) ) } { \sqrt { T + 2 } }$ </td><td rowspan=1 colspan=1> $\leq e ^ { 2 }$ ≤ e5/2</td></tr></table>

where we bounded the last column using $\tau L \leq 1 / 4$ and the corresponding estimates for $\scriptstyle \sum _ { t = 0 } ^ { T } { \frac { \tau _ { t } ^ { 2 } } { S _ { t } } }$ from the third column. Substituting these bounds into (B.1), and additionally using

$$
{ \frac { 1 } { \sqrt { T + 2 } - 1 } } \leq { \frac { 2 } { \sqrt { T + 1 } } } , \qquad { \frac { 1 } { \sqrt { T + 2 } } } \leq { \frac { 1 } { \sqrt { T + 1 } } }
$$

for (ii), we obtain

$$
\mathbb { E } [ h ( { \mathbf x } _ { T + 1 } ) - h ^ { * } ] \leq \frac { e ^ { 2 } } { \sqrt { T + 1 } } \left[ \frac { \| { \mathbf x } _ { 0 } - { \mathbf x } ^ { * } \| ^ { 2 } } { \tau } + 4 \tau \sigma _ { * } ^ { 2 } + \frac { 8 L \tau \left[ h ( { \mathbf x } _ { 0 } ) - h ^ { * } \right] } { T + 1 } \right] ,\tag{B.2}
$$

$$
( \mathrm { i i } ) \quad \mathbb { E } \big [ h ( \mathbf { x } _ { T + 1 } ) - h ^ { * } \big ] \leq \frac { \epsilon ^ { 5 / 2 } } { \sqrt { T + 1 } } \left[ \frac { \| \mathbf { x _ { 0 } } - \mathbf { x } ^ { * } \| ^ { 2 } } { 2 \tau } + 4 \tau \sigma _ { * } ^ { 2 } ( 1 + \ln ( T + 1 ) ) + 4 L \tau \left[ h ( \mathbf { x _ { 0 } } ) - h ^ { * } \right] \right] .\tag{B.3}
$$

Substituting $\tau = 1 / ( 4 L )$ into (B.2) and (B.3), and suppressing constants, yield the claims of the theorem.

## Appendix C. Technical Lemmas for Proximal SGD

We provide the proofs of the technical lemmas used in our analysis. We first restate a classical consequence of the definition of the proximal operator that we use often. Its proof follows from the definition of $\operatorname { p r o x } _ { \tau g }$ and the convexity of g (see, e.g., Beck, 2017, Theorem 6.39). For $\mathbf { u } = \mathrm { p r o x } _ { \tau _ { t } g } ( \mathbf { y } )$ , it holds that

$$
\begin{array} { r } { \langle \mathbf { u } - \mathbf { y } , \mathbf { x } - \mathbf { u } \rangle \geq \tau _ { t } \big ( g ( \mathbf { u } ) - g ( \mathbf { x } ) \big ) , \qquad \forall \mathbf { x } \in \mathbb { R } ^ { d } . } \end{array}\tag{C.1}
$$

To bound the iterates and the second moment of the stochastic subgradients, we start from the following one-step inequality for (proxSGD).

## C.1 Proof of Lemma 3.1

Proof Using the identity $\begin{array} { r } { \langle \mathbf { a } , \mathbf { b } \rangle = - \frac { 1 } { 2 } \| \mathbf { a } \| ^ { 2 } - \frac { 1 } { 2 } \| \mathbf { b } \| ^ { 2 } + \frac { 1 } { 2 } \| \mathbf { a } + \mathbf { b } \| ^ { 2 } } \end{array}$ , we have

$$
\| { \bf x } _ { t + 1 } - { \bf z } _ { t } \| ^ { 2 } - \| { \bf x } _ { t } - { \bf z } _ { t } \| ^ { 2 } = \| { \bf x } _ { t + 1 } - { \bf x } _ { t } \| ^ { 2 } + 2 \langle { \bf x } _ { t + 1 } - { \bf x } _ { t } , { \bf x } _ { t } - { \bf z } _ { t } \rangle .\tag{C.2}
$$

Consider the last term in the above identity. We use the prox-inequality (C.1) with ${ \bf y } =$ $\mathbf { x } _ { t } - \tau _ { t } \widetilde { \nabla } f ( \mathbf { x } _ { t } )$ and $\mathbf { x } = \mathbf { z } _ { t }$ to obtain

$$
\begin{array} { r l } & { \langle \mathbf { x } _ { t + 1 } - \mathbf { x } _ { t } , \mathbf { x } _ { t } - \mathbf { z } _ { t } \rangle = \langle \mathbf { x } _ { t + 1 } - \mathbf { x } _ { t } , \mathbf { x } _ { t } - \mathbf { x } _ { t + 1 } \rangle + \langle \mathbf { x } _ { t + 1 } - \mathbf { x } _ { t } , \mathbf { x } _ { t + 1 } - \mathbf { z } _ { t } \rangle } \\ & { \qquad \leq - \| \mathbf { x } _ { t + 1 } - \mathbf { x } _ { t } \| ^ { 2 } + \tau _ { t } ( g ( \mathbf { z } _ { t } ) - g ( \mathbf { x } _ { t + 1 } ) ) + \tau _ { t } \langle \widetilde { \nabla } f ( \mathbf { x } _ { t } ) , \mathbf { z } _ { t } - \mathbf { x } _ { t + 1 } \rangle . } \end{array}
$$

Substituting this into (C.2) gives

$$
\begin{array} { r } { \| \mathbf { x } _ { t + 1 } - \mathbf { z } _ { t } \| ^ { 2 } - \| \mathbf { x } _ { t } - \mathbf { z } _ { t } \| ^ { 2 } \leq - \| \mathbf { x } _ { t + 1 } - \mathbf { x } _ { t } \| ^ { 2 } + 2 \tau _ { t } ( g ( \mathbf { z } _ { t } ) - g ( \mathbf { x } _ { t + 1 } ) ) + 2 \tau _ { t } \langle \widetilde { \nabla } f ( \mathbf { x } _ { t } ) , \mathbf { z } _ { t } - \mathbf { x } _ { t + 1 } \rangle . } \end{array}\tag{C.3}
$$

We can rewrite the last term as

$$
\begin{array} { r } { 2 \tau _ { t } \langle \widetilde { \nabla } f ( \mathbf { x } _ { t } ) , \mathbf { z } _ { t } - \mathbf { x } _ { t + 1 } \rangle = 2 \tau _ { t } \langle \widetilde { \nabla } f ( \mathbf { x } _ { t } ) , \mathbf { z } _ { t } - \mathbf { x } _ { t } \rangle + 2 \tau _ { t } \langle \widetilde { \nabla } f ( \mathbf { x } _ { t } ) , \mathbf { x } _ { t } - \mathbf { x } _ { t + 1 } \rangle . } \end{array}\tag{C.4}
$$

Plugging this back to (C.3) and taking conditional expectation with respect to $\mathcal { F } _ { t }$ , we have

$$
\begin{array} { r l } & { \mathbb { E } _ { t } \| { \mathbf { x } } _ { t + 1 } - { \mathbf { z } } _ { t } \| ^ { 2 } - \| { \mathbf { x } } _ { t } - { \mathbf { z } } _ { t } \| ^ { 2 } \leq - \mathbb { E } _ { t } \| { \mathbf { x } } _ { t + 1 } - { \mathbf { x } } _ { t } \| ^ { 2 } + 2 \tau _ { t } \mathbb { E } _ { t } [ g ( { \mathbf { z } } _ { t } ) - g ( { \mathbf { x } } _ { t + 1 } ) ] } \\ & { \qquad + \ 2 \tau _ { t } \langle { \mathbf { s } } _ { t } , { \mathbf { z } } _ { t } - { \mathbf { x } } _ { t } \rangle + 2 \tau _ { t } \mathbb { E } _ { t } \langle \widetilde { \nabla } f ( { \mathbf { x } } _ { t } ) , { \mathbf { x } } _ { t } - { \mathbf { x } } _ { t + 1 } \rangle , } \end{array}
$$

where $\mathbf { s } _ { t } : = \mathbb { E } _ { t } [ \widetilde { \nabla } f ( \mathbf { x } _ { t } ) ] \in \partial f ( \mathbf { x } _ { t } )$ by Assumption 1, and we used that $\mathbf { x } _ { t }$ and $\mathbf { z } _ { t }$ are $\mathcal { F } _ { t ^ { - } }$ measurable. We estimate $\langle \mathbf { s } _ { t } , \mathbf { z } _ { t } - \mathbf { x } _ { t } \rangle \leq [ f ( \mathbf { z } _ { t } ) - f ( \mathbf { x } _ { t } ) ]$ using the convexity of $f .$ . Adding and subtracting $2 \tau _ { t } f ( \mathbf { x } _ { t + 1 } )$ and rearranging terms give us

$$
\begin{array} { r l } & { \mathbb { E } _ { t } \| { \mathbf { x } } _ { t + 1 } - { \mathbf { z } } _ { t } \| ^ { 2 } - \| { \mathbf { x } } _ { t } - { \mathbf { z } } _ { t } \| ^ { 2 } \leq 2 \tau _ { t } \mathbb { E } _ { t } [ h ( { \mathbf { z } } _ { t } ) - h ( { \mathbf { x } } _ { t + 1 } ) ] - \mathbb { E } _ { t } \| { \mathbf { x } } _ { t + 1 } - { \mathbf { x } } _ { t } \| ^ { 2 } } \\ & { \qquad + \ 2 \tau _ { t } \mathbb { E } _ { t } \big [ f ( { \mathbf { x } } _ { t + 1 } ) - f ( { \mathbf { x } } _ { t } ) \big ] + 2 \tau _ { t } \mathbb { E } _ { t } \langle \widetilde { \nabla } f ( { \mathbf { x } } _ { t } ) , { \mathbf { x } } _ { t } - { \mathbf { x } } _ { t + 1 } \rangle . } \end{array}\tag{C.5}
$$

We now estimate the last two terms in the above inequality. Let $\mathbf { q } _ { t + 1 } \in \partial f ( \mathbf { x } _ { t + 1 } )$ . By convexity, we have $f ( \mathbf { x } _ { t + 1 } ) - f ( \mathbf { x } _ { t } ) \leq \langle \mathbf { q } _ { t + 1 } , \mathbf { x } _ { t + 1 } - \mathbf { x } _ { t } \rangle$ . Using this to bound the second line in (C.5) gives

$$
2 \tau _ { t } [ f ( \mathbf { x } _ { t + 1 } ) - f ( \mathbf { x } _ { t } ) ] + 2 \tau _ { t } \langle \widetilde { \nabla } f ( \mathbf { x } _ { t } ) , \mathbf { x } _ { t } - \mathbf { x } _ { t + 1 } \rangle \leq 2 \tau _ { t } \langle \mathbf { q } _ { t + 1 } - \widetilde { \nabla } f ( \mathbf { x } _ { t } ) , \mathbf { x } _ { t + 1 } - \mathbf { x } _ { t } \rangle .\tag{C.6}
$$

We estimate the right-hand side using Young’s inequality as

$$
\begin{array} { r } { 2 \tau _ { t } \langle \mathbf { q } _ { t + 1 } - \widetilde { \nabla } f ( \mathbf { x } _ { t } ) , \mathbf { x } _ { t + 1 } - \mathbf { x } _ { t } \rangle \leq \tau _ { t } ^ { 2 } \| \mathbf { q } _ { t + 1 } - \widetilde { \nabla } f ( \mathbf { x } _ { t } ) \| ^ { 2 } + \| \mathbf { x } _ { t + 1 } - \mathbf { x } _ { t } \| ^ { 2 } . } \end{array}\tag{C.7}
$$

We combine (C.6) and (C.7) (after taking conditional expectation) in (C.5) and group the terms involving $\| \mathbf { x } _ { t + 1 } - \mathbf { x } _ { t } \| ^ { 2 }$ to obtain

$$
\begin{array} { r } { \mathbb { E } _ { t } \| { \mathbf { x } } _ { t + 1 } - { \mathbf { z } } _ { t } \| ^ { 2 } - \| { \mathbf { x } } _ { t } - { \mathbf { z } } _ { t } \| ^ { 2 } \leq 2 \tau _ { t } \mathbb { E } _ { t } \big [ h ( { \mathbf { z } } _ { t } ) - h ( { \mathbf { x } } _ { t + 1 } ) \big ] + \tau _ { t } ^ { 2 } \mathbb { E } _ { t } \| { \mathbf { q } } _ { t + 1 } - \widetilde { \nabla } f ( { \mathbf { x } } _ { t } ) \| ^ { 2 } . } \end{array}
$$

Taking total expectation gives us the assertion.

We next prove the analogue of Lemma 3.1 under smoothness.

## C.2 Proof of Lemma 5.2

Proof Using the identity $\begin{array} { r } { \langle \mathbf { a } , \mathbf { b } \rangle = - \frac { 1 } { 2 } \| \mathbf { a } \| ^ { 2 } - \frac { 1 } { 2 } \| \mathbf { b } \| ^ { 2 } + \frac { 1 } { 2 } \| \mathbf { a } + \mathbf { b } \| ^ { 2 } } \end{array}$ , we have

$$
\| \mathbf { x } _ { t + 1 } - \mathbf { z } _ { t } \| ^ { 2 } - \| \mathbf { x } _ { t } - \mathbf { z } _ { t } \| ^ { 2 } = \| \mathbf { x } _ { t + 1 } - \mathbf { x } _ { t } \| ^ { 2 } + 2 \langle \mathbf { x } _ { t + 1 } - \mathbf { x } _ { t } , \mathbf { x } _ { t } - \mathbf { z } _ { t } \rangle .\tag{C.8}
$$

Consider the last term in the above identity. We use the prox-inequality (C.1) with ${ \bf y } =$ $\mathbf { x } _ { t } - \tau _ { t } \widetilde { \nabla } f ( \mathbf { x } _ { t } )$ and $\mathbf { x } = \mathbf { z } _ { t }$ to obtain

$$
\begin{array} { r l } & { \langle \mathbf { x } _ { t + 1 } - \mathbf { x } _ { t } , \mathbf { x } _ { t } - \mathbf { z } _ { t } \rangle = \langle \mathbf { x } _ { t + 1 } - \mathbf { x } _ { t } , \mathbf { x } _ { t } - \mathbf { x } _ { t + 1 } \rangle + \langle \mathbf { x } _ { t + 1 } - \mathbf { x } _ { t } , \mathbf { x } _ { t + 1 } - \mathbf { z } _ { t } \rangle } \\ & { \qquad \leq - \| \mathbf { x } _ { t + 1 } - \mathbf { x } _ { t } \| ^ { 2 } + \tau _ { t } ( g ( \mathbf { z } _ { t } ) - g ( \mathbf { x } _ { t + 1 } ) ) + \tau _ { t } \langle \widetilde { \nabla } f ( \mathbf { x } _ { t } ) , \mathbf { z } _ { t } - \mathbf { x } _ { t + 1 } \rangle . } \end{array}
$$

Substituting this into (C.8) gives

$$
\begin{array} { r } { \| \mathbf { x } _ { t + 1 } - \mathbf { z } _ { t } \| ^ { 2 } - \| \mathbf { x } _ { t } - \mathbf { z } _ { t } \| ^ { 2 } \leq - \| \mathbf { x } _ { t + 1 } - \mathbf { x } _ { t } \| ^ { 2 } + 2 \tau _ { t } ( g ( \mathbf { z } _ { t } ) - g ( \mathbf { x } _ { t + 1 } ) ) + 2 \tau _ { t } \langle \widetilde { \nabla } f ( \mathbf { x } _ { t } ) , \mathbf { z } _ { t } - \mathbf { x } _ { t + 1 } \rangle . } \end{array}\tag{C.9}
$$

We can rewrite the last term as

$$
\begin{array} { r l } & { 2 \tau _ { t } \langle \tilde { \nabla } f ( { \mathbf x } _ { t } ) , { \mathbf z } _ { t } - { \mathbf x } _ { t + 1 } \rangle = 2 \tau _ { t } \langle \tilde { \nabla } f ( { \mathbf x } _ { t } ) , { \mathbf z } _ { t } - { \mathbf x } _ { t } \rangle + 2 \tau _ { t } \langle \tilde { \nabla } f ( { \mathbf x } _ { t } ) - \nabla f ( { \mathbf x } _ { t } ) , { \mathbf x } _ { t } - { \mathbf x } _ { t + 1 } \rangle } \\ & { ~ + ~ 2 \tau _ { t } \langle \nabla f ( { \mathbf x } _ { t } ) , { \mathbf x } _ { t } - { \mathbf x } _ { t + 1 } \rangle . } \end{array}\tag{C.10}
$$

We estimate the second term in (C.10) using Young’s inequality as

$$
2 \tau _ { t } \langle \widetilde { \nabla } f ( \mathbf { x } _ { t } ) - \nabla f ( \mathbf { x } _ { t } ) , \mathbf { x } _ { t } - \mathbf { x } _ { t + 1 } \rangle \leq \frac { \tau _ { t } ^ { 2 } } { 1 - \tau _ { t } L } \Vert \widetilde { \nabla } f ( \mathbf { x } _ { t } ) - \nabla f ( \mathbf { x } _ { t } ) \Vert ^ { 2 } + ( 1 - \tau _ { t } L ) \Vert \mathbf { x } _ { t } - \mathbf { x } _ { t + 1 } \Vert ^ { 2 } .
$$

For the last term in (C.10), we can use the smoothness of $f$ to derive

$$
2 \tau _ { t } \langle \nabla f ( \mathbf { x } _ { t } ) , \mathbf { x } _ { t } - \mathbf { x } _ { t + 1 } \rangle \leq 2 \tau _ { t } \Big ( f ( \mathbf { x } _ { t } ) - f ( \mathbf { x } _ { t + 1 } ) + \frac { L } { 2 } \| \mathbf { x } _ { t } - \mathbf { x } _ { t + 1 } \| ^ { 2 } \Big ) .
$$

We plug the last two estimates into (C.10), which we then insert into (C.9) and rearrange to obtain

$$
\begin{array} { r l } & { \| { \mathbf x } _ { t + 1 } - { \mathbf z } _ { t } \| ^ { 2 } - \| { \mathbf x } _ { t } - { \mathbf z } _ { t } \| ^ { 2 } \leq 2 \tau _ { t } \langle \widetilde \nabla f ( { \mathbf x } _ { t } ) , { \mathbf z } _ { t } - { \mathbf x } _ { t } \rangle + 2 \tau _ { t } ( g ( { \mathbf z } _ { t } ) - g ( { \mathbf x } _ { t + 1 } ) ) } \\ & { \qquad + \frac { \tau _ { t } ^ { 2 } } { 1 - \tau _ { t } L } \| \widetilde \nabla f ( { \mathbf x } _ { t } ) - \nabla f ( { \mathbf x } _ { t } ) \| ^ { 2 } + 2 \tau _ { t } [ f ( { \mathbf x } _ { t } ) - f ( { \mathbf x } _ { t + 1 } ) ] . } \end{array}
$$

Since $\tau _ { t } \leq \frac { 1 } { 2 L }$ , we have $\frac { \tau _ { t } ^ { 2 } } { 1 - \tau _ { t } L } \leq 2 \tau _ { t } ^ { 2 }$ . Plugging in this estimate, and taking conditional expectation with respect to $\mathcal { F } _ { t }$ , we obtain,

$$
\begin{array} { r l } & { \mathbb { E } _ { t } \| { \mathbf { x } } _ { t + 1 } - { \mathbf { z } } _ { t } \| ^ { 2 } \leq \| { \mathbf { x } } _ { t } - { \mathbf { z } } _ { t } \| ^ { 2 } + 2 \tau _ { t } \langle \nabla f ( { \mathbf { x } } _ { t } ) , { \mathbf { z } } _ { t } - { \mathbf { x } } _ { t } \rangle + 2 \tau _ { t } \mathbb { E } _ { t } [ g ( { \mathbf { z } } _ { t } ) - g ( { \mathbf { x } } _ { t + 1 } ) ] } \\ & { \qquad + \left. 2 \tau _ { t } ^ { 2 } \mathbb { E } _ { t } [ \| \widetilde \nabla f ( { \mathbf { x } } _ { t } ) - \nabla f ( { \mathbf { x } } _ { t } ) \| ^ { 2 } ] + 2 \tau _ { t } \mathbb { E } _ { t } [ f ( { \mathbf { x } } _ { t } ) - f ( { \mathbf { x } } _ { t + 1 } ) ] . \right. } \end{array}
$$

Using $\mathbb { E } _ { t } [ \| \widetilde { \nabla } f ( \mathbf { x } _ { t } ) - \nabla f ( \mathbf { x } _ { t } ) \| ^ { 2 } ] \le \mathbb { E } _ { t } \| \widetilde { \nabla } f ( \mathbf { x } _ { t } ) \| ^ { 2 }$ , the convexity of $f ,$ , and $h = f + g$ , we obtain

$$
\begin{array} { r } { \mathbb { E } _ { t } \| { \mathbf { x } } _ { t + 1 } - { \mathbf { z } } _ { t } \| ^ { 2 } \leq \| { \mathbf { x } } _ { t } - { \mathbf { z } } _ { t } \| ^ { 2 } + 2 \tau _ { t } \mathbb { E } _ { t } \big [ h ( { \mathbf { z } } _ { t } ) - h ( { \mathbf { x } } _ { t + 1 } ) \big ] + 2 \tau _ { t } ^ { 2 } \mathbb { E } _ { t } [ \| \widetilde { \nabla } f ( { \mathbf { x } } _ { t } ) \| ^ { 2 } ] . } \end{array}
$$

Taking total expectation and rearranging gives the desired result.

We next prove the last-iterate reduction for proximal SGD, which converts the one-step bound into a bound on the last iterate.

## C.3 Proof of Lemma 3.4

Proof Let us define $\alpha _ { - 1 } = \alpha _ { 0 } = 1$ and $\begin{array} { r } { \alpha _ { t } = \frac { S _ { t - 1 } } { S _ { t } + \tau _ { t } \rho _ { t } } \alpha _ { t - 1 } } \end{array}$ for $t = 1 , \dots , T$ . Since $\rho _ { t } \le 1$ and $\tau _ { t - 1 } \geq \tau _ { t }$ , we have $S _ { t - 1 } = S _ { t } + \tau _ { t - 1 } \geq S _ { t } + \tau _ { t } \rho _ { t }$ . As a result, we have that the sequence $\left( \alpha _ { t } \right)$ is non-decreasing. Let $\mathbf { z } _ { - 1 } = \mathbf { x } ^ { * }$ and

$$
\mathbf { z } _ { t } = \left( 1 - { \frac { \alpha _ { t - 1 } } { \alpha _ { t } } } \right) \mathbf { x } _ { t } + { \frac { \alpha _ { t - 1 } } { \alpha _ { t } } } \mathbf { z } _ { t - 1 } , \qquad t = 0 , \ldots , T .
$$

Unrolling the definition of $\mathbf { z } _ { t }$ shows that it is a convex combination of $\mathbf { x } ^ { * } , \mathbf { x } _ { 0 } , \ldots , \mathbf { x } _ { t } ,$ since the coeficients below are nonnegative and sum to 1:

$$
\mathbf { z } _ { t } = \frac { 1 } { \alpha _ { t } } \mathbf { x } ^ { * } + \frac { 1 } { \alpha _ { t } } \sum _ { s = 0 } ^ { t } ( \alpha _ { s } - \alpha _ { s - 1 } ) \mathbf { x } _ { s } .\tag{C.11}
$$

Hence $\mathbf { z } _ { t }$ is $\mathcal { F } _ { t }$ -measurable, so the hypothesis of this lemma applies with this $\mathbf { z } _ { t }$ . Multiplying both sides of the hypothesis by $2 \alpha _ { t } \tau _ { t }$ gives

$$
2 \alpha _ { t } \tau _ { t } \mathbb { E } [ h ( \mathbf { x } _ { t + 1 } ) - h ( \mathbf { z } _ { t } ) - \rho _ { t } h ( \mathbf { x } _ { t } ) + \rho _ { t } h ^ { * } ] \leq \alpha _ { t } \mathbb { E } [ \| \mathbf { x } _ { t } - \mathbf { z } _ { t } \| ^ { 2 } - \| \mathbf { x } _ { t + 1 } - \mathbf { z } _ { t } \| ^ { 2 } ] + 2 \alpha _ { t } v _ { t } \tau _ { t } .\tag{C.12}
$$

The definition of $\mathbf { z } _ { t }$ also gives

$$
\mathbf { x } _ { t } - \mathbf { z } _ { t } = \mathbf { x } _ { t } - \left( 1 - \frac { \alpha _ { t - 1 } } { \alpha _ { t } } \right) \mathbf { x } _ { t } - \frac { \alpha _ { t - 1 } } { \alpha _ { t } } \mathbf { z } _ { t - 1 } = \frac { \alpha _ { t - 1 } } { \alpha _ { t } } \left( \mathbf { x } _ { t } - \mathbf { z } _ { t - 1 } \right) .
$$

Using this and $\alpha _ { t - 1 } \leq \alpha _ { t }$ , we estimate the first term on the right-hand side of (C.12) as

$$
\alpha _ { t } \mathbb { E } \Vert \mathbf { x } _ { t } - \mathbf { z } _ { t } \Vert ^ { 2 } = \frac { \alpha _ { t - 1 } ^ { 2 } } { \alpha _ { t } } \mathbb { E } \Vert \mathbf { x } _ { t } - \mathbf { z } _ { t - 1 } \Vert ^ { 2 } \leq \alpha _ { t - 1 } \mathbb { E } \Vert \mathbf { x } _ { t } - \mathbf { z } _ { t - 1 } \Vert ^ { 2 } .
$$

Substituting this estimate into (C.12) and dividing by 2, we get

$$
\begin{array} { r l r } {  { \alpha _ { t } \tau _ { t } \mathbb { E } \big [ h \big ( \mathbf { x } _ { t + 1 } \big ) - h \big ( \mathbf { z } _ { t } \big ) - \rho _ { t } h \big ( \mathbf { x } _ { t } \big ) + \rho _ { t } h ^ { * } \big ] } } \\ & { } & { \leq \frac { \alpha _ { t - 1 } } { 2 } \mathbb { E } \| \mathbf { x } _ { t } - \mathbf { z } _ { t - 1 } \| ^ { 2 } - \frac { \alpha _ { t } } { 2 } \mathbb { E } \| \mathbf { x } _ { t + 1 } - \mathbf { z } _ { t } \| ^ { 2 } + \alpha _ { t } v _ { t } \tau _ { t } . } \end{array}
$$

Summing from $t = 0$ to $T$ gives

$$
\Gamma : = \sum _ { t = 0 } ^ { T } \alpha _ { t } \tau _ { t } \mathbb { E } [ h ( \mathbf { x } _ { t + 1 } ) - h ( \mathbf { z } _ { t } ) - \rho _ { t } h ( \mathbf { x } _ { t } ) + \rho _ { t } h ^ { \ast } ] \leq \frac { 1 } { 2 } \| \mathbf { x } _ { 0 } - \mathbf { x } ^ { \ast } \| ^ { 2 } + \sum _ { t = 0 } ^ { T } \alpha _ { t } v _ { t } \tau _ { t } ,\tag{C.13}
$$

where we dropped the negative term on the right-hand side and used the fact $\mathbf { z } _ { - 1 } = \mathbf { x } ^ { * }$ and $\alpha _ { - 1 } = 1$ . We now lower bound Γ. By the convexity of h applied to (C.11), we have

$$
h ( { \bf z } _ { t } ) \le \frac { 1 } { \alpha _ { t } } h ( { \bf x } ^ { * } ) + \frac { 1 } { \alpha _ { t } } \sum _ { s = 0 } ^ { t } ( \alpha _ { s } - \alpha _ { s - 1 } ) h ( { \bf x } _ { s } ) ,
$$

and consequently

$$
\Gamma \geq \sum _ { t = 0 } ^ { T } \tau _ { t } \mathbb { E } \left[ \alpha _ { t } h ( \mathbf { x } _ { t + 1 } ) - \sum _ { s = 0 } ^ { t } ( \alpha _ { s } - \alpha _ { s - 1 } ) h ( \mathbf { x } _ { s } ) - \rho _ { t } \alpha _ { t } h ( \mathbf { x } _ { t } ) + \rho _ { t } \alpha _ { t } h ^ { * } - h ^ { * } \right] .\tag{C.14}
$$

Now consider the second term inside the expectation. We rewrite it as follows:

$$
\sum _ { s = 0 } ^ { t } ( \alpha _ { s } - \alpha _ { s - 1 } ) h ( { \bf x } _ { s } ) = \sum _ { s = 0 } ^ { t } ( \alpha _ { s } - \alpha _ { s - 1 } ) \left[ h ( { \bf x } _ { s } ) - h ^ { * } \right] + ( \alpha _ { t } - 1 ) h ^ { * } ,
$$

where we used the fact $\begin{array} { r } { \sum _ { s = 0 } ^ { t } ( \alpha _ { s } - \alpha _ { s - 1 } ) = \alpha _ { t } - \alpha _ { - 1 } = \alpha _ { t } - 1 } \end{array}$ . Substituting this back into (C.14) and simplifying gives

$$
\Gamma \geq \sum _ { t = 0 } ^ { T } \tau _ { t } \mathbb { E } \left[ \alpha _ { t } ( h ( \mathbf { x } _ { t + 1 } ) - h ^ { * } ) - \rho _ { t } \alpha _ { t } ( h ( \mathbf { x } _ { t } ) - h ^ { * } ) - \sum _ { s = 0 } ^ { t } ( \alpha _ { s } - \alpha _ { s - 1 } ) \left[ h ( \mathbf { x } _ { s } ) - h ^ { * } \right] \right] .\tag{C.15}
$$

We next proceed to show that this lower bound gives us an expression for the last iterate. We define

$$
r _ { t } = h ( \mathbf x _ { t } ) - h ^ { * } .
$$

Then (C.15) becomes

$$
\Gamma \geq \mathbb { E } \left[ \sum _ { t = 0 } ^ { T } \tau _ { t } \alpha _ { t } r _ { t + 1 } - \sum _ { t = 0 } ^ { T } \rho _ { t } \tau _ { t } \alpha _ { t } r _ { t } - \sum _ { t = 0 } ^ { T } \sum _ { s = 0 } ^ { t } \tau _ { t } ( \alpha _ { s } - \alpha _ { s - 1 } ) r _ { s } \right] .\tag{C.16}
$$

Consider the first term on the right-hand side of (C.16). We can rewrite it as

$$
\sum _ { t = 0 } ^ { T } \tau _ { t } \alpha _ { t } r _ { t + 1 } = \tau _ { T } \alpha _ { T } r _ { T + 1 } + \sum _ { t = 1 } ^ { T } \tau _ { t - 1 } \alpha _ { t - 1 } r _ { t } .
$$

The second term on the right-hand side of (C.16) can be rewritten as

$$
\sum _ { t = 0 } ^ { T } \rho _ { t } \tau _ { t } \alpha _ { t } r _ { t } = \rho _ { 0 } \tau _ { 0 } r _ { 0 } + \sum _ { t = 1 } ^ { T } \rho _ { t } \tau _ { t } \alpha _ { t } r _ { t } ,
$$

where we used the fact that $\alpha _ { 0 } = 1$ . And finally, for the last term on the right hand side of (C.16), we swap the order of summation and use the definition of $S _ { t }$ to get

$$
\sum _ { t = 0 } ^ { T } \sum _ { s = 0 } ^ { t } \tau _ { t } ( \alpha _ { s } - \alpha _ { s - 1 } ) r _ { s } = \sum _ { t = 0 } ^ { T } ( \alpha _ { t } - \alpha _ { t - 1 } ) r _ { t } S _ { t } = \sum _ { t = 1 } ^ { T } ( \alpha _ { t } - \alpha _ { t - 1 } ) r _ { t } S _ { t } ,
$$

where we used the fact that $\alpha _ { - 1 } = \alpha _ { 0 } = 1$ so the first term $( t = 0 )$ in the sum is zero.

Combining the last three display equations in (C.16) and grouping terms involving $r _ { t }$ gives

$$
\Gamma \geq \mathbb { E } \left[ \tau _ { T } \alpha _ { T } r _ { T + 1 } + \sum _ { t = 1 } ^ { T } r _ { t } \left( \tau _ { t - 1 } \alpha _ { t - 1 } - ( \alpha _ { t } - \alpha _ { t - 1 } ) S _ { t } - \tau _ { t } \rho _ { t } \alpha _ { t } \right) - \rho _ { 0 } \tau _ { 0 } r _ { 0 } \right] .\tag{C.17}
$$

From the definition of $\alpha _ { t }$ and $S _ { t - 1 } = S _ { t } + \tau _ { t - 1 }$ , we have

$$
\tau _ { t - 1 } \alpha _ { t - 1 } - ( \alpha _ { t } - \alpha _ { t - 1 } ) S _ { t } - \tau _ { t } \rho _ { t } \alpha _ { t } = 0 .
$$

Using this in (C.17), and plugging the result to (C.13), we obtain the following inequality after moving ρ<sub>0</sub>τ<sub>0</sub>r<sub>0</sub> to the right-hand side and dividing by τ<sub>T</sub>α<sub>T</sub>:

$$
\mathbb { E } [ r _ { T + 1 } ] \le \frac { 1 } { 2 } \left( \frac { \| { \bf x } _ { 0 } - { \bf x } ^ { * } \| ^ { 2 } } { \tau _ { T } \alpha _ { T } } + 2 \sum _ { t = 0 } ^ { T } \frac { \alpha _ { t } v _ { t } \tau _ { t } } { \tau _ { T } \alpha _ { T } } + \frac { 2 \rho _ { 0 } \tau _ { 0 } r _ { 0 } } { \tau _ { T } \alpha _ { T } } \right) .\tag{C.18}
$$

To estimate the factor, $\frac { 1 } { \tau _ { T } \alpha _ { T } }$ , let

$$
\Lambda _ { T } : = \prod _ { t = 1 } ^ { T } \left( 1 + \frac { \tau _ { t } \rho _ { t } } { S _ { t } } \right) .
$$

By the definition of $S _ { t }$ in the statement of the lemma, we have that $S _ { T } = \tau _ { T }$ . Using this and the definition of $\alpha _ { t }$ (along with $\alpha _ { 0 } = 1 )$ give

$$
{ \frac { 1 } { \tau _ { T } \alpha _ { T } } } = { \frac { 1 } { S _ { T } } } \prod _ { t = 1 } ^ { T } { \frac { \alpha _ { t - 1 } } { \alpha _ { t } } } = { \frac { 1 } { S _ { T } } } \prod _ { t = 1 } ^ { T } { \frac { S _ { t } + \tau _ { t } \rho _ { t } } { S _ { t - 1 } } } = { \frac { 1 } { S _ { T } } } \prod _ { t = 1 } ^ { T } { \frac { S _ { t } } { S _ { t - 1 } } } \left( 1 + { \frac { \tau _ { t } \rho _ { t } } { S _ { t } } } \right) = { \frac { 1 } { S _ { 0 } } } \Lambda _ { T } .
$$

We next estimate the second term inside the parenthesis in (C.18) similarly:

$$
\frac { \alpha _ { t } v _ { t } \tau _ { t } } { \tau _ { T } \alpha _ { T } } = \frac { v _ { t } \tau _ { t } } { S _ { T } } \prod _ { i = t + 1 } ^ { T } \frac { S _ { i } + \tau _ { i } \rho _ { i } } { S _ { i - 1 } } = \frac { v _ { t } \tau _ { t } } { S _ { T } } \prod _ { i = t + 1 } ^ { T } \frac { S _ { i } } { S _ { i - 1 } } \left( 1 + \frac { \tau _ { i } \rho _ { i } } { S _ { i } } \right) \leq \frac { v _ { t } \tau _ { t } } { S _ { t } } \Lambda _ { T } .\tag{C.19}
$$

Plugging these estimates into (C.18), we get

$$
\mathbb { E } [ r _ { T + 1 } ] \le \frac { 1 } { 2 } \Lambda _ { T } \left( \frac { \| \mathbf { x } _ { 0 } - \mathbf { x } ^ { * } \| ^ { 2 } } { S _ { 0 } } + 2 \sum _ { t = 0 } ^ { T } v _ { t } \frac { \tau _ { t } } { S _ { t } } + \frac { 2 \rho _ { 0 } \tau _ { 0 } r _ { 0 } } { S _ { 0 } } \right) .\tag{C.20}
$$

Now using the fact that $1 + x \leq e ^ { x }$ for all $x ,$ we can estimate $\Lambda _ { T }$ as follows:

$$
\Lambda _ { T } = \prod _ { t = 1 } ^ { T } \left( 1 + \frac { \tau _ { t } \rho _ { t } } { S _ { t } } \right) \leq \prod _ { t = 1 } ^ { T } \exp \left( \frac { \tau _ { t } \rho _ { t } } { S _ { t } } \right) = \exp \left( \sum _ { t = 1 } ^ { T } \frac { \tau _ { t } \rho _ { t } } { S _ { t } } \right) .
$$

Substituting this back into (C.20) yields the assertion.

## C.4 Technical Lemmas for Bounding the Step Size Terms

In this section, we prove various bounds related to the step sizes. Throughout this section, we define $\begin{array} { r } { S _ { t } : = \sum _ { s = t } ^ { T } \tau _ { s } } \end{array}$

Lemma C.1 Let us set $\begin{array} { r } { \tau _ { t } = \frac { \tau } { \sqrt { t + 1 } } } \end{array}$ for some $\tau > 0$ . Then, for any $T \geq 1$ , we have

$$
\sum _ { t = 0 } ^ { T } \tau _ { t } ^ { 2 } \leq \tau ^ { 2 } ( 1 + \ln ( T + 1 ) ) ,\tag{i}
$$

$$
( i i ) \ S _ { 0 } \geq 2 \tau ( \sqrt { T + 2 } - 1 ) ,\tag{iii}
$$

$$
\sum _ { t = 0 } ^ { T } \frac { \tau _ { t } ^ { 2 } } { S _ { t } } \leq \frac { 2 \tau ( 1 + \ln ( T + 1 ) ) } { \sqrt { T + 2 } } ,
$$

$$
\sum _ { t = 0 } ^ { T - 1 } \frac { \tau _ { t } ^ { 2 } } { S _ { t + 1 } } \leq \frac { 2 \tau ( 1 + \ln T ) } { \sqrt { T } } .
$$

Proof Assertion (i) follows trivially by summing the harmonic series. Next, by the definition of $S _ { t }$ in the statement of Lemma 3.4, we have

$$
S _ { t } = \sum _ { s = t } ^ { T } \frac { \tau } { \sqrt { s + 1 } } \geq \int _ { t } ^ { T + 1 } \frac { \tau } { \sqrt { x + 1 } } d x = 2 \tau ( \sqrt { T + 2 } - \sqrt { t + 1 } ) ,
$$

which gives (ii) after setting $t = 0$

We now estimate (iii) by using the lower bound for $S _ { t }$ , just derived, and the definition of $\tau _ { t }$ as

$$
\sum _ { t = 0 } ^ { T } \frac { \tau _ { t } ^ { 2 } } { S _ { t } } \leq \tau \sum _ { t = 0 } ^ { T } \frac { 1 } { 2 ( t + 1 ) ( \sqrt { T + 2 } - \sqrt { t + 1 } ) } \leq \tau \sqrt { T + 2 } \sum _ { t = 0 } ^ { T } \frac { 1 } { ( t + 1 ) ( T + 1 - t ) } ,\tag{C.21}
$$

since

$$
{ \frac { 1 } { { \sqrt { T + 2 } } - { \sqrt { t + 1 } } } } = { \frac { { \sqrt { T + 2 } } + { \sqrt { t + 1 } } } { T + 1 - t } } \leq { \frac { 2 { \sqrt { T + 2 } } } { T + 1 - t } } .
$$

Now, on (C.21), we use

$$
{ \frac { 1 } { ( t + 1 ) ( T + 1 - t ) } } = { \frac { 1 } { T + 2 } } \left( { \frac { 1 } { t + 1 } } + { \frac { 1 } { T + 1 - t } } \right) ,
$$

to obtain

$$
\sum _ { t = 0 } ^ { T } { \frac { \tau _ { t } ^ { 2 } } { S _ { t } } } \leq { \frac { \tau } { { \sqrt { T + 2 } } } } \sum _ { t = 0 } ^ { T } \left( { \frac { 1 } { t + 1 } } + { \frac { 1 } { T + 1 - t } } \right) = { \frac { 2 \tau } { { \sqrt { T + 2 } } } } \sum _ { t = 0 } ^ { T } { \frac { 1 } { t + 1 } } \leq { \frac { 2 \tau ( 1 + \ln ( T + 1 ) ) } { { \sqrt { T + 2 } } } } .
$$

Finally, assertion (iv) follows from a similar estimation. Since

$$
\sum _ { t = 0 } ^ { T - 1 } \frac { \tau _ { t } ^ { 2 } } { S _ { t + 1 } } \leq \frac { \tau } { 2 } \sum _ { t = 0 } ^ { T - 1 } \frac { 1 } { ( t + 1 ) ( \sqrt { T + 2 } - \sqrt { t + 2 } ) } \leq \tau \sqrt { T + 2 } \sum _ { t = 0 } ^ { T - 1 } \frac { 1 } { ( t + 1 ) ( T - t ) } ,
$$

we repeat the same argument as in the proof of the third assertion by replacing $T + 1 - t$ with $T - t$ to get

$$
\sum _ { t = 0 } ^ { T - 1 } \frac { \tau _ { t } ^ { 2 } } { S _ { t + 1 } } \leq \frac { 2 \tau \sqrt { T + 2 } ( 1 + \ln ( T ) ) } { T + 1 } .
$$

(iii)

The claim then follows by $\textstyle { \frac { { \sqrt { T + 2 } } } { T + 1 } } \leq { \frac { 1 } { \sqrt { T } } }$

Lemma C.2 Let us set $\begin{array} { r } { \tau _ { t } = \tau \frac { T - t + 1 } { ( T + 1 ) ^ { 3 / 2 } } } \end{array}$ for some $\tau > 0$ . Then, for any $T \geq 1$ , we have

$$
\sum _ { t = 0 } ^ { T } \tau _ { t } ^ { 2 } \leq \tau ^ { 2 } ,
$$

$$
S _ { 0 } \geq \frac { \tau \sqrt { T + 1 } } { 2 } ,
$$

$$
\sum _ { t = 0 } ^ { T } \frac { \tau _ { t } ^ { 2 } } { S _ { t } } \leq \frac { 2 \tau } { \sqrt { T + 1 } } ,
$$

$$
\sum _ { t = 0 } ^ { T - 1 } \frac { \tau _ { t } ^ { 2 } } { S _ { t + 1 } } \leq \frac { 4 \tau } { \sqrt { T + 1 } } .
$$

Proof By the definition of $\tau _ { t }$ , we have

$$
\sum _ { s = 0 } ^ { T } \tau _ { s } ^ { 2 } = \frac { \tau ^ { 2 } } { ( T + 1 ) ^ { 3 } } \sum _ { s = 0 } ^ { T } ( T - s + 1 ) ^ { 2 } .\tag{C.22}
$$

We estimate the sum as

$$
\sum _ { s = 0 } ^ { T } ( T - s + 1 ) ^ { 2 } = \sum _ { k = 1 } ^ { T + 1 } k ^ { 2 } = { \frac { ( T + 1 ) ( T + 2 ) ( 2 T + 3 ) } { 6 } } .
$$

Plugging this into (C.22) and using $( T + 2 ) ( 2 T + 3 ) \leq 6 ( T + 1 ) ^ { 2 }$ gives assertion (i).

For assertion (ii), we estimate $S _ { t }$ (defined in the statement of Lemma 3.4) as

$$
S _ { t } = \sum _ { s = t } ^ { T } \tau _ { s } = \frac { \tau } { ( T + 1 ) ^ { 3 / 2 } } \sum _ { s = t } ^ { T } ( T - s + 1 ) = \frac { \tau ( T - t + 1 ) ( T - t + 2 ) } { 2 ( T + 1 ) ^ { 3 / 2 } } ,\tag{C.23}
$$

where we used $\begin{array} { r } { \sum _ { s = t } ^ { T } T - s + 1 = \sum _ { j = 1 } ^ { T - t + 1 } j } \end{array}$ . The lower bound for $S _ { 0 }$ follows by plugging in $t = 0$ in the above bound and using a basic estimate.

For assertion (iii), the definition of $\tau _ { t }$ and (C.23) give

$$
\sum _ { t = 0 } ^ { T } \frac { \tau _ { t } ^ { 2 } } { S _ { t } } = \frac { 2 \tau } { ( T + 1 ) ^ { 3 / 2 } } \sum _ { t = 0 } ^ { T } \frac { T - t + 1 } { T - t + 2 } \leq \frac { 2 \tau } { \sqrt { T + 1 } } ,
$$

where the last step is because the summands are upper bounded by 1. This is the third estimate.

For assertion (iv), the definition of $\tau _ { t }$ and (C.23) give

$$
\sum _ { t = 0 } ^ { T - 1 } \frac { \tau _ { t } ^ { 2 } } { S _ { t + 1 } } = \frac { 2 \tau } { ( T + 1 ) ^ { 3 / 2 } } \sum _ { t = 0 } ^ { T - 1 } \frac { T - t + 1 } { T - t } \leq \frac { 4 \tau T } { ( T + 1 ) ^ { 3 / 2 } } \leq \frac { 4 \tau } { \sqrt { T + 1 } } ,
$$

since $\frac { T - t + 1 } { T - t } \leq 2$ for $t \leq T - 1$

Lemma C.3 Let $\begin{array} { r } { \tau _ { t } = \frac { \tau } { \ln ( t + 2 ) \sqrt { t + 2 } } } \end{array}$ for some $\tau > 0$ . Then, for any $T \geq 2$ , we have

$$
\begin{array} { r l r } { ( i ) } & { \displaystyle \sum _ { t = 0 } ^ { T } \tau _ { t } ^ { 2 } \le 3 \tau ^ { 2 } , } & { ( i i ) \ S _ { 0 } \ge \frac { \tau \sqrt { T + 1 } } { \sqrt { 2 } \ln ( T + 2 ) } , } \\ { ( i i i ) \ \displaystyle \sum _ { t = 0 } ^ { T } \frac { \tau _ { t } ^ { 2 } } { S _ { t } } \le \frac { 6 \sqrt { 2 } \tau \ln ( T + 2 ) + 3 3 \tau } { \sqrt { T + 1 } } , } & { ( i v ) \displaystyle \sum _ { t = 0 } ^ { T - 1 } \frac { \tau _ { t } ^ { 2 } } { S _ { t + 1 } } \le \frac { 6 \sqrt { 2 } \tau \ln ( T + 2 ) + 3 2 \tau } { \sqrt { T + 1 } } . } \end{array}
$$

Proof Assertion (i) follows from a standard estimation. Now, we estimate $S _ { t }$ (defined in the statement of Lemma 3.4) as

$$
S _ { t } = \sum _ { s = t } ^ { T } \tau _ { s } = \sum _ { s = t } ^ { T } \frac { \tau } { \ln ( s + 2 ) \sqrt { s + 2 } } \geq \frac { \tau ( T - t + 1 ) } { \ln ( T + 2 ) \sqrt { T + 2 } } .
$$

Assertion (ii) follows by plugging in $t = 0$ and using the fact that ${ \sqrt { T + 2 } } \leq { \sqrt { 2 } } { \sqrt { T + 1 } }$ We next prove assertion (iv). In order to control $\scriptstyle \sum _ { t = 0 } ^ { T - 1 } { \frac { \tau _ { t } ^ { 2 } } { S _ { t + 1 } } }$ , we will split the sum into two parts: $t = 0 , \ldots , \lfloor T / 2 \rfloor - 1$ and $t = \lfloor T / 2 \rfloor , \dots , T - 1$ . Let $\dot { m } = \lfloor T / 2 \rfloor$ . For the first half, the denominator $S _ { t + 1 }$ can be bounded from below by $S _ { m }$ . In the second half, the numerator $\tau _ { t } ^ { 2 }$ can be bounded by $\tau _ { m } ^ { 2 }$ . We will use these facts to bound the two parts separately.

For the first part, we have

$$
\sum _ { t = 0 } ^ { m - 1 } \frac { \tau _ { t } ^ { 2 } } { S _ { t + 1 } } \leq \frac { 1 } { S _ { m } } \sum _ { t = 0 } ^ { m - 1 } \tau _ { t } ^ { 2 } ,\tag{C.24}
$$

where we used the fact that $S _ { t + 1 }$ is decreasing in t. Now, we estimate $S _ { m }$ by lower bounding $\tau _ { s }$ as

$$
S _ { m } = \sum _ { s = m } ^ { T } \tau _ { s } \geq \frac { \tau ( T - m + 1 ) } { \ln ( T + 2 ) \sqrt { T + 2 } } \geq \frac { \tau ( T + 1 ) } { 2 \ln ( T + 2 ) \sqrt { T + 2 } } \geq \frac { \tau \sqrt { T + 1 } } { 2 \sqrt { 2 } \ln ( T + 2 ) } ,
$$

where we used $T - m + 1 \geq ( T + 1 ) / 2$ and ${ \sqrt { T + 2 } } \leq { \sqrt { 2 } } { \sqrt { T + 1 } }$ . Plugging this into (C.24) yields

$$
\sum _ { t = 0 } ^ { m - 1 } \frac { \tau _ { t } ^ { 2 } } { S _ { t + 1 } } \leq \frac { 2 \sqrt { 2 } \ln ( T + 2 ) } { \tau \sqrt { T + 1 } } \sum _ { t = 0 } ^ { m - 1 } \tau _ { t } ^ { 2 } \leq \frac { 6 \sqrt { 2 } \tau \ln ( T + 2 ) } { \sqrt { T + 1 } } ,\tag{C.25}
$$

where we use $m - 1 \leq T$ and assertion (i).

For the second half of the sum, we have $t \geq m$ . First, note that since $\tau _ { t }$ is non-increasing, we have that

$$
S _ { t + 1 } = \sum _ { i = t + 1 } ^ { T } \tau _ { i } \geq ( T - t ) \tau _ { T } .
$$

With this, and $\tau _ { t }$ being non-increasing, we have

$$
\sum _ { t = m } ^ { T - 1 } \frac { \tau _ { t } ^ { 2 } } { S _ { t + 1 } } \leq \tau _ { m } ^ { 2 } \sum _ { t = m } ^ { T - 1 } \frac { 1 } { S _ { t + 1 } } \leq \tau _ { m } ^ { 2 } \sum _ { t = m } ^ { T - 1 } \frac { 1 } { ( T - t ) \tau _ { T } } .
$$

Using now the definitions of $\tau _ { m } , \tau _ { T }$ , and $\begin{array} { r } { \sum _ { t = m } ^ { T - 1 } \frac { 1 } { T - t } \leq 1 + \ln ( T + 1 ) } \end{array}$ , we obtain

$$
\sum _ { t = m } ^ { T - 1 } \frac { \tau _ { t } ^ { 2 } } { S _ { t + 1 } } \leq \frac { \tau \ln ( T + 2 ) \sqrt { T + 2 } } { ( m + 2 ) \ln ^ { 2 } ( m + 2 ) } ( 1 + \ln ( T + 1 ) ) .\tag{C.26}
$$

First, it is trivial to obtain from $m = \lfloor T / 2 \rfloor \geq T / 2 - 1$ the estimate

$$
m + 2 \geq { \frac { T + 2 } { 2 } } .\tag{C.27}
$$

Second, it also follows, for $T \geq 2$ , that

$$
\ln ( m + 2 ) \geq { \frac { 1 } { 2 } } \ln ( T + 2 ) .\tag{C.28}
$$

Plugging (C.27) and (C.28) back into (C.26) gives us

$$
\sum _ { t = m } ^ { T - 1 } \frac { \tau _ { t } ^ { 2 } } { S _ { t + 1 } } \leq \frac { 8 \tau ( 1 + \ln ( T + 1 ) ) } { \sqrt { T + 2 } \ln ( T + 2 ) } \leq \frac { 3 2 \tau } { \sqrt { T + 2 } } ,\tag{C.29}
$$

where in the second inequality, we used $\textstyle { \frac { 1 + \ln ( T + 1 ) } { \ln ( T + 2 ) } } \leq 4$ for $T \geq 0$ . Combining (C.25) and (C.29) gives us

$$
\sum _ { t = 0 } ^ { T - 1 } \frac { \tau _ { t } ^ { 2 } } { S _ { t + 1 } } \leq \frac { 6 \sqrt { 2 } \tau \ln ( T + 2 ) + 3 2 \tau } { \sqrt { T + 1 } } .
$$

Finally, we prove assertion (iii). Since $S _ { t } \ \geq \ S _ { t + 1 }$ for $t = 0 , \ldots , T - 1$ and $S _ { T } = \tau _ { T }$ splitting the sum and using assertion (iv), we have

$$
\sum _ { t = 0 } ^ { T } \frac { \tau _ { t } ^ { 2 } } { S _ { t } } = \sum _ { t = 0 } ^ { T - 1 } \frac { \tau _ { t } ^ { 2 } } { S _ { t } } + \frac { \tau _ { T } ^ { 2 } } { S _ { T } } \leq \sum _ { t = 0 } ^ { T - 1 } \frac { \tau _ { t } ^ { 2 } } { S _ { t + 1 } } + \tau _ { T } \leq \frac { 6 \sqrt { 2 } \tau \ln ( T + 2 ) + 3 2 \tau } { \sqrt { T + 1 } } + \frac { \tau } { \ln ( T + 2 ) \sqrt { T + 2 } } .
$$

Using $1 / \ln ( T + 2 ) < 1$ for $T \geq 1$ and $\begin{array} { r } { \frac { 1 } { \sqrt { T + 2 } } < \frac { 1 } { \sqrt { T + 1 } } } \end{array}$ for $T \geq 0$ gives the inequality.

We have the following fact for recentering the (B-G) assumption. Its proof is trivial by using $\| \mathbf { x } - \mathbf { a } \| ^ { 2 } \leq 2 \| \mathbf { x } - \mathbf { b } \| ^ { 2 } + 2 \| \mathbf { b } - \mathbf { a } \| ^ { 2 }$

Fact 1 $I f \mathbb { E } \| \widetilde { \nabla } f ( \mathbf { x } ) \| ^ { 2 } \leq B ^ { 2 } \| \mathbf { x } - \mathbf { a } \| ^ { 2 } + G ^ { 2 }$ for all x and some fixed a, then for any other point b, we have

$$
\begin{array} { r } { \mathbb { E } \| \widetilde { \nabla } f ( \mathbf { x } ) \| ^ { 2 } \leq 2 B ^ { 2 } \| \mathbf { x } - \mathbf { b } \| ^ { 2 } + 2 B ^ { 2 } \| \mathbf { b } - \mathbf { a } \| ^ { 2 } + G ^ { 2 } . } \end{array}
$$

## References

A. Alacaoglu, Y. Malitsky, and S. J. Wright. Towards weaker variance assumptions for stochastic optimization. arXiv:2504.09951, 2025.

A. Attia, M. Schliserman, U. Sherman, and T. Koren. Fast last-iterate convergence of SGD in the smooth interpolation regime. In Advances in Neural Information Processing Systems, 2025.

A. Beck. First-order methods in optimization. SIAM, 2017.

J. R. Blum. Approximation methods which converge with probability one. The Annals of Mathematical Statistics, 25(2):382–386, 1954.

X. Chen, Q. Lin, and J. Pe˜na. Optimal regularized dual averaging methods for stochastic optimization. In Advances in Neural Information Processing Systems, 2012.

J. Domke. Provable smoothness guarantees for black-box variational inference. In International Conference on Machine Learning, 2020.

J. Domke, R. M. Gower, and G. Garrigos. Provable convergence guarantees for black-box variational inference. In Advances in Neural Information Processing Systems, 2023.

G. Garrigos, D. Cortild, L. Ketels, and J. Peypouquet. Last-iterate complexity of SGD for convex and smooth stochastic problems. arXiv:2507.14122, 2025.

E. G. Gladyshev. On stochastic approximation. Theory of Probability & Its Applications, 10(2):275–278, 1965.

B. Grimmer. Radial subgradient method. SIAM Journal on Optimization, 28(1):459–469, 2018.

B. Grimmer. Convergence rates for deterministic and stochastic subgradient methods without Lipschitz continuity. SIAM Journal on Optimization, 29(2):1350–1365, 2019.

N. J. A. Harvey, C. Liaw, Y. Plan, and S. Randhawa. Tight analyses for non-smooth stochastic gradient descent. In Conference on Learning Theory, 2019.

P. Jain, D. Nagaraj, and P. Netrapalli. Making the last iterate of SGD information theoretically optimal. In Conference on Learning Theory, 2019.

A. Khaled and P. Richt´arik. Better theory for SGD in the nonconvex world. Transactions on Machine Learning Research, 2023.

A. Khaled, O. Sebbouh, N. Loizou, R. M. Gower, and P. Richt´arik. Unified analysis of stochastic gradient methods for composite convex and smooth optimization. Journal of Optimization Theory and Applications, 199(2):499–540, 2023.

K. Kim, Y. Ma, T. Campbell, and J. Gardner. Nearly dimension-independent convergence of mean-field black-box variational inference. In Advances in Neural Information Processing Systems, 2025.

H. Labarri\`ere, C. Molinari, S. Villa, and L. Rosasco. SGD for variational inference: Tackling unbounded variance via preconditioning and dynamic batching. arXiv:2605.07531, 2026.

G. Lan. First-order and stochastic optimization methods for machine learning. Springer, 2020.

X. Li, L. Zhao, D. Zhu, and A. M.-C. So. Revisiting subgradient method: Complexity and convergence beyond Lipschitz continuity. Vietnam Journal of Mathematics, 53(4): 735–755, 2025.

Z. Liu and Z. Zhou. Revisiting the last-iterate convergence of stochastic gradient methods. In International Conference on Learning Representations, 2024.

R. Lopez, J. Regier, M. B. Cole, M. I. Jordan, and N. Yosef. Deep generative modeling for single-cell transcriptomics. Nature Methods, 15(12):1053–1058, 2018.

H. Lu. “Relative continuity” for non-Lipschitz nonsmooth convex optimization using stochastic (or deterministic) mirror descent. INFORMS Journal on Optimization, 1(4): 288–303, 2019.

A. Nemirovski, A. Juditsky, G. Lan, and A. Shapiro. Robust stochastic approximation approach to stochastic programming. SIAM Journal on Optimization, 19(4):1574–1609, 2009.

F. Orabona. Last iterate of SGD converges (even in unbounded domains), 2020. URL https://parameterfree.com/2020/08/07/ last-iterate-of-sgd-converges-even-in-unbounded-domains/.

B. T. Polyak and Y. Z. Tsypkin. Pseudogradient adaptation and training algorithms. Automation and Remote Control, 34(3):45–67, 1973.

R. Ranganath, S. Gerrish, and D. Blei. Black box variational inference. In Artificial Intelligence and Statistics, 2014.

R. Ranganath, A. Perotte, N. Elhadad, and D. Blei. Deep survival analysis. In Machine Learning for Healthcare Conference, 2016.

J. Renegar. “Eficient” subgradient methods for general convex optimization. SIAM Journal on Optimization, 26(4):2649–2676, 2016.

H. Robbins and S. Monro. A stochastic approximation method. The Annals of Mathematical Statistics, 22(3):400–407, 1951.

L. Rosasco, S. Villa, and B. C. V˜u. Convergence of stochastic proximal gradient algorithm. Applied Mathematics & Optimization, 82(3):891–917, 2020.

O. Shamir and T. Zhang. Stochastic gradient descent for non-smooth optimization: Convergence results and optimal averaging schemes. In International Conference on Machine Learning, 2013.

K. K. T. Vaidyan, M. P. Friedlander, and A. Alacaoglu. Convergence rate of the last iterate of stochastic proximal algorithms. In International Conference on Machine Learning, 2026.

M. Zamani and F. Glineur. Exact convergence rate of the last iterate in subgradient methods. SIAM Journal on Optimization, 35(3):2182–2201, 2025.