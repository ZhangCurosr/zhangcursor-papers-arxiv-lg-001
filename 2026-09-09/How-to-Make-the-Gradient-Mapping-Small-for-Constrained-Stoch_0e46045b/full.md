# How to Make the Gradient Mapping Small for Constrained Stochastic Min-Max Problems and Beyond

Ahmet Alacaoglu<sup>∗</sup>

## Abstract

We study the stochastic first-order oracle complexity for constrained or regularized convex-concave min-max optimization and stochastic monotone variational inequalities. We focus on the case when suboptimality is measured in terms of the gradient mapping, also known as, forward-backward or natural residual, an optimality notion that generalizes the gradient norm for unconstrained problems. In this setting, under standard unbiased oracle access with now-standard variance assumptions, the best-known complexity for making the norm of the gradient mapping less than $\varepsilon \ \mathrm { i s } \ \widetilde O ( \varepsilon ^ { - 4 } )$ , compared to the nearoptimal $\widetilde { O } ( \varepsilon ^ { - 2 } )$ that is established in the unconstrained case. We bridge this gap to improve the gradient mapping complexity for constrained convex-concave min-max problems to $\widetilde { O } ( \varepsilon ^ { - 2 } )$ . We then extend to prove the same complexity for problems without the bounded variance, by using the Blum-Gladyshev assumption.

## 1 Introduction

We study algorithms to solve the standard template of constrained convex-concave min-max optimization,

$$
\operatorname* { m i n } _ { \mathbf { u } \in U } \operatorname* { m a x } _ { \mathbf { v } \in V } f ( \mathbf { u } , \mathbf { v } ) ,\tag{1.1}
$$

where $f ( \cdot , \mathbf { v } )$ is convex, $f ( \mathbf { u } , \cdot )$ is concave, the gradients $\nabla _ { \mathbf { u } } f ( \mathbf { u } , \mathbf { v } ) , \nabla _ { \mathbf { v } } f ( \mathbf { u } , \mathbf { v } )$ are Lipschitz continuous, and $U \subseteq \mathbb { R } ^ { n } , V \subseteq \mathbb { R } ^ { m }$ are closed convex sets admitting eficient projection operators.

Two canonical examples of this template come from cases where the coupling between u and v is bilinear, such as matrix games

$$
\operatorname* { m i n } _ { \mathbf { u } \in \Delta _ { n } } \operatorname* { m a x } _ { \mathbf { v } \in \Delta _ { m } } \langle A \mathbf { u } , \mathbf { v } \rangle ,\tag{1.2}
$$

where $\Delta _ { n }$ denotes the probability simplex, that is, $\begin{array} { r } { \Delta _ { n } = \{ \mathbf { u } \in \mathbb { R } _ { + } ^ { n } \colon \textstyle \sum _ { i = 1 } ^ { n } u _ { i } = 1 \} } \end{array}$

The second common example is linearly constrained optimization

$$
\operatorname* { m i n } _ { \mathbf { u } \in \mathbb { R } ^ { n } } f ( \mathbf { u } ) : A \mathbf { u } \leq \mathbf { b } ,
$$

for a proper, convex and closed function $f \colon \mathbb { R } ^ { n }  ( - \infty , \infty ]$ . Solving this problem eficiently is generally done by utilizing convex duality to convert to a min-max problem given as

$$
\operatorname* { m i n } _ { \mathbf { u } \in \mathbb { R } ^ { n } } \operatorname* { m a x } _ { \mathbf { v } \geq 0 } f ( \mathbf { u } ) + \langle A \mathbf { u } - \mathbf { b } , \mathbf { v } \rangle .\tag{1.3}
$$

Strictly speaking, this problem does not fit the template (1.1) due to potential nonsmoothness in $f .$ However, our focus on (1.1) in this section is purely for a simple presentation. In the sequel, we develop our results for the more general problem of variational inequalities to cover this example, see (VI) for a precise problem statement.

We will focus on a setting where we do not have access to gradients $\nabla _ { \mathbf { u } } { f } ( \mathbf { u } , \mathbf { v } )$ and $\nabla _ { \mathbf { v } } { f } ( \mathbf { u } , \mathbf { v } )$ , but to their unbiased estimators such that

$$
\mathbb { E } [ \nabla _ { \mathbf { u } } f _ { \xi } ( \mathbf { u } , \mathbf { v } ) ] = \nabla _ { \mathbf { u } } f ( \mathbf { u } , \mathbf { v } ) ,
$$

where $\xi$ is sampled from an unknown distribution $P ,$ , where we can have $f ( \mathbf { u } , \mathbf { v } ) = \mathbb { E } _ { \boldsymbol { \xi } \sim P } [ f _ { \boldsymbol { \xi } } ( \mathbf { u } , \mathbf { v } ) ]$ . At every iteration of our algorithm, we will assume that we receive i.i.d. samples $\xi \sim P .$

More concretely, for (1.2) or (1.3), these stochastic oracles may correspond to randomly sampling rows or columns from the matrix A, whereas the full-gradients require accessing the matrix A at every iteration. A concrete example is given at the end of Section 1.1.

Two optimality notions. Arguably, the most widely used optimality notion for min-max problems is the primal-dual gap

$$
\mathrm { G a p } ( \bar { \mathbf { u } } , \bar { \mathbf { v } } ) = \operatorname* { m a x } _ { \mathbf { u } \in U , \mathbf { v } \in V } f ( \bar { \mathbf { u } } , \mathbf { v } ) - f ( \mathbf { u } , \bar { \mathbf { v } } ) .
$$

This is well-suited to problems such as matrix games in (1.2), however, when the sets $U , V$ are unbounded, such as the case of linearly constrained optimization in (1.3), this notion runs into problems. In particular, this quantity may be infinite at test points u¯, v¯ that are not optimal.

This well-known drawback gave rise to the restricted primal-dual gap function [Nes09] that takes the maximum over a compact set that needs to satisfy certain requirements. That is, this set needs to contain a solution and the iterates of the algorithm, see [Nes09, Lemma $4 ]$ . This approach can be compatible in the deterministic case when the boundedness of the iterates can be shown for standard methods.

For stochastic algorithms, boundedness of the iterates does not generally hold, causing dificulty in identifying a compact set to define the restricted gap function ´a la [Nes09]. Moreover, the restricted gap is generally taken over sets that generally depend on the solution, making it dificult to compute in general.

Despite its incompatibility with unbounded domains and stochastic problems, this notion is widely used because the standard convergence analyses for algorithms, such as extragradient [Kor76] or forward-backwardforward (FBF) methods [Tse00], give a rate result on the gap as a byproduct. As a result, optimal methods for reducing the gap is well-established for both deterministic and stochastic problems [Nem04, NJLS09].

Perhaps even a more natural way of measuring suboptimality for unbounded problems is the gradient mapping norm [Nes13, Section 2], which generalizes the gradient norm for unconstrained problems

$$
\| \mathcal { G } _ { \eta , f } ( \mathbf { x } , \mathbf { y } ) \| ^ { 2 } = \frac { 1 } { \eta ^ { 2 } } \left( \| \mathbf { x } - \mathrm { p r o j } _ { U } ( \mathbf { x } - \eta \nabla _ { \mathbf { x } } f ( \mathbf { x } , \mathbf { y } ) ) \| ^ { 2 } + \| \mathbf { y } - \mathrm { p r o j } _ { V } ( \mathbf { y } + \eta \nabla _ { \mathbf { y } } f ( \mathbf { x } , \mathbf { y } ) ) \| ^ { 2 } \right) ,\tag{1.4}
$$

where $\mathrm { p r o j } _ { C }$ denotes the Euclidean projection on $C .$ While this notion is widely used for nonconvex problems, optimal complexity results for making this quantity small, even for convex minimization problems is relatively recent, see [Nes12] for deterministic and [AZ18] for stochastic optimization.

Clearly, this reduces to the gradient norm when $( U , V ) = ( \mathbb { R } ^ { n } , \mathbb { R } ^ { m } )$ since

$$
\mathcal { G } _ { \eta , f } ( \mathbf { x } , \mathbf { y } ) = \binom { \nabla _ { \mathbf { x } } f ( \mathbf { x } , \mathbf { y } ) } { - \nabla _ { \mathbf { y } } f ( \mathbf { x } , \mathbf { y } ) } .
$$

Optimal methods for reducing the gradient – or gradient mapping – norm for deterministic convex-concave min-max problems is established relatively recently as well, see [Dia20, Kim21, YR21, KG22]. An optimal method for unconstrained and stochastic min-max problems is proposed in [CL24] who showed the complexity $\widetilde { O } ( \varepsilon ^ { - 2 } )$ for making the gradient norm less than ε with matching lower bounds.

Yet, the best-known complexity for constrained min-max problems, which are of interest in terms of many applications – such as matrix games (1.2), linearly constrained optimization (1.3), and distributionally robust optimization [ND16, Eq. (1)] – remained at $\widetilde { O } ( \varepsilon ^ { - 4 } )$ – with the exception of [TDNT26] that improved this to $\widetilde { O } ( \varepsilon ^ { - 1 0 / 3 } )$ under stronger assumptions on the stochastic oracle, see Table 1. The goal of this work is to improve this complexity to one that is optimal in terms of ε dependence. This discussion is summarized in Table 2 where we highlighted the main setting we focus on.

Generalized problems. For a simpler notation and a slight generalization, we will consider the problem of variational inequalities where the goal is to

$$
\mathrm { f i n d } \mathbf { x } ^ { \star } \mathrm { ~ s u c h ~ t h a t ~ } \langle F ( \mathbf { x } ^ { \star } ) , \mathbf { x } - \mathbf { x } ^ { \star } \rangle \geq r ( \mathbf { x } ^ { \star } ) - r ( \mathbf { x } ) \mathrm { ~ f o r ~ a n y ~ } \mathbf { x } \in \mathbb { R } ^ { d } ,\tag{VI}
$$

with a proper, convex and closed r : $\mathbb { R } ^ { d } \to ( - \infty , \infty ]$ and a monotone operator <sup>1</sup> $F \colon  { \mathbb { R } ^ { d } } \to  { \mathbb { R } ^ { d } }$ . Mapping (1.1) to this problem is done by setting

$$
\mathbf { x } = { \binom { \mathbf { u } } { \mathbf { v } } } , \quad F ( \mathbf { x } ) = { \binom { \nabla _ { \mathbf { u } } f ( \mathbf { u } , \mathbf { v } ) } { - \nabla _ { \mathbf { v } } f ( \mathbf { u } , \mathbf { v } ) } } , \quad r ( \mathbf { x } ) = \delta _ { U } ( \mathbf { u } ) + \delta _ { V } ( \mathbf { v } ) ,\tag{1.5}
$$

where $\delta _ { C } ( { \bf x } )$ denotes the indicator function, that is 0 when $\mathbf { x } \in C$ and ∞ otherwise.

We will often use the equivalent form written as a monotone inclusion problem where the goal is to

$$
\mathrm { f i n d } \mathbf { x } ^ { \star } \mathrm { s u c h \ t h a t } 0 \in H ( \mathbf { x } ^ { \star } ) : = ( F + \partial r ) ( \mathbf { x } ^ { \star } ) .\tag{MI}
$$

We focus on this specific monotone inclusion for presentation purposes, but by changing Algorithm 1 to a method based on FBF [Tse00] or forward-reflected-backward method [MT20], our analysis can be extended to solve general monotone inclusion where ∂r is replaced by a maximally monotone operator. We skip this extension for keeping our notation simpler.

By overloading the terminology for simplicity, we shall now write the gradient mapping as

$$
\mathcal G _ { \eta , H } ( \mathbf x ) = \frac { 1 } { \eta } \left( \mathbf x - \mathrm { p r o x } _ { \eta r } ( \mathbf x - \eta F ( \mathbf x ) ) \right) ,
$$

which is equivalent to (1.4) with the setting (1.5). We emphasize that F does not need to be the gradient of a function, and we merely use this notation to keep the link with min-max optimization (1.1) transparent. This notion is also referred to as forward-backward residual or natural residual (in the specific case when r is an indicator function), see [FP03, Section 10.3].

Our goal in the sequel is to find an approximate solution to (MI) by making the residual small. This result directly covers making the gradient mapping small for convex-concave and constrained min-max problems by the mapping (1.5).

Complexity. Given the problem $( \mathrm { V I } )$ introduced above, we measure complexity in terms of the number of calls to the unbiased stochastic oracles $F _ { \xi }$ such that

$$
\mathbb { E } [ F _ { \xi } ( { \bf x } ) ] = F ( { \bf x } )
$$

and to the proximal operator of $r \mathrm { ~ - ~ }$ which reduces to the number of calls to stochastic gradients and projections on $U , V$ in the case of min-max problems (1.1)— to get a certain optimality notion less than ε.

There exists now a large body of works that analyze methods such as stochastic extragradient with optimal complexity for reducing the gap. That ${ \mathrm { i s } } ,$ it is well-known how to get

$$
\mathbb { E } [ \operatorname { G a p } ( { \bar { \mathbf { x } } } , { \bar { \mathbf { y } } } ) ] \leq \varepsilon , { \mathrm { ~ w i t h ~ c o m p l e x i t y ~ } } O ( \varepsilon ^ { - 2 } ) .
$$

We refer to [NJLS09, JNT11, KLL22, Zha22, LL26] for representative examples.

Moreover, when $r \equiv 0$ , that is, we have a root finding problem, [CL24] established the oracle complexity

$$
\widetilde { O } ( \varepsilon ^ { - 2 } ) \mathrm { ~ f o r ~ m a k i n g ~ } \mathbb { E } \| F ( \mathbf { x } ) \| \leq \varepsilon ,
$$

which is optimal. The authors assumed that the unbiased oracle has bounded variance upper bounded by $\sigma ^ { 2 }$ and designed an algorithm that sets its parameters by using $\sigma ^ { 2 }$ and $\| \mathbf { x } _ { 0 } - \mathbf { x } ^ { \star } \| ^ { 2 }$ , the initial distance of their initial point to a solution. For unconstrained problems, with stronger assumptions such as sharpness, [CSGD22] also obtained a similar complexity, yet for unconstrained and merely monotone problems, the complexity in this work deteriorated to $\widetilde { \cal O } ( \varepsilon ^ { - 3 } )$

For constrained problems, the best-known complexity for the gradient mapping under our assumptions is

$$
O (  { \varepsilon } ^ { - 4 } ) \mathrm { f o r } \mathrm { m a k i n g } \mathbb { E } \| \mathcal { G } _ { \eta , H } ( \mathbf { x } ) \| \leq \varepsilon ,
$$

which is suboptimal. This complexity remained as the best-known on a large body of literature for constrained or regularized min-max problems [IJOT17, BMSV21, AMW25, LK21, PFL<sup>+</sup>23, B¨oh23, DDJ21, TDNT25].

<table><tr><td></td><td>Constraints</td><td>Need for setting alg. parameters</td><td>Stoc. oracle complexity</td><td>Variance assumption</td></tr><tr><td>[CL24]</td><td>X</td><td> $\sigma ^ { 2 } , \lVert \mathbf { x } _ { 0 } - \mathbf { x } ^ { \star } \rVert ^ { 2 } , L _ { F } , G$ </td><td> $\widetilde { O } ( \varepsilon ^ { - 2 } )$ </td><td>Bounded</td></tr><tr><td>[TDNT26]</td><td>√</td><td> $L _ { F }$ </td><td> $\widetilde { O } ( \varepsilon ^ { - 1 0 / 3 } )$ </td><td>Bounded variance Multi-point oracle mean-squared Lipschitz</td></tr><tr><td>[IJOT17] and other works</td><td></td><td> $L _ { F } , B$ </td><td> $\widetilde { O } ( \varepsilon ^ { - 4 } )$ </td><td>Blum-Gladyshev</td></tr><tr><td>This work</td><td></td><td> $L _ { F } , B$ </td><td> $\widetilde { O } ( \varepsilon ^ { - 2 } )$ </td><td>Blum-Gladyshev</td></tr></table>

Table 1: Complexity results for stochastic convex-concave optimization under Assumption 1 and variance assumption specified in the last column. For the first row, the complexity is for $\mathbb { E } \| F ( \mathbf { x } ) \| \leq \varepsilon$ . For the last three rows, the complexity is for $\begin{array} { r } { \mathbb { E } \| \mathcal { G } _ { \eta , H } ( \mathbf { x } ) \| \le \varepsilon . } \end{array}$ . See the notation in (1.5) and (MI).
<table><tr><td></td><td></td><td>Constrained Unconstrained</td></tr><tr><td>Gap</td><td> $O ( \varepsilon ^ { - 2 } )$  [JNT1i]</td><td> $O ( \varepsilon ^ { - 2 } )$  [JNT1i]</td></tr><tr><td>Gradient mapping</td><td> $O ( \varepsilon ^ { - 4 } )$ </td><td> $\widetilde { O } ( \varepsilon ^ { - 2 } )$ </td></tr></table>

Table 2: Best-known complexity results for stochastic convex-concave optimization under Assumption 1 and bounded variance. For the first row, the complexity is for making the (restricted) gap less than ε. For the second row, it is for making the gradient mapping (which reduces to gradient norm for unconstrained problem) less than ε.

Even though some of these works relied on some relaxations of convex-concavity, none of the analyses in those works gave rise to a better complexity for convex-concave problems. The aim of this work is to improve this complexity for problems satisfying the Blum-Gladyshev variance and without requiring parameters depending on initial distance to solution or variance upper bounds.

A recent work by [TDNT26] obtained a complexity of $\widetilde { O } ( \varepsilon ^ { - 1 0 / 3 } )$ for constrained problems by using variance reduction. Diferent from our work, they required a multi-point oracle that requires querying the stochastic oracle at the same seed for diferent points, the mean-square Lipschitzness assumption (which is stronger than mere Lipschitzness of F) and bounded variance.

## 1.1 Notation and Preliminaries

Given the history of random vectors $\xi _ { 0 } , \xi _ { 0 . 5 } , \ldots , \xi _ { k }$ , we denote the expectation conditioned on the generated σ-algebra as $\mathbb { E } _ { k } [ \cdot ] = \mathbb { E } [ \cdot \mid \sigma ( \xi _ { 0 } , \xi _ { 0 . 5 } , \ldots , \xi _ { k } ) ]$ . For a vector x, the notation $x _ { i }$ refers to its i-th coordinate.

In this paper, we always consider a regularizer $r \colon  { \mathbb { R } } ^ { d } \to ( - \infty , \infty ]$ that is convex, proper and closed. Its subdiferential set is denoted as ∂r. We define the proximal operator of r as

$$
\mathrm { p r o x } _ { r } ( \mathbf { x } ) = \arg \operatorname* { m i n } _ { \mathbf { y } } r ( \mathbf { y } ) + \frac { 1 } { 2 } \| \mathbf { x } - \mathbf { y } \| ^ { 2 } ,\tag{1.6}
$$

from which we can derive the prox-inequality, by using the optimality condition of (1.6),

$$
\mathbf { y } = \operatorname { p r o x } _ { r } ( \mathbf { x } ) \iff \langle \mathbf { y } - \mathbf { x } , \mathbf { u } - \mathbf { y } \rangle \geq r ( \mathbf { y } ) - r ( \mathbf { u } ) \quad \forall \mathbf { u } \in \mathbb { R } ^ { d } .\tag{1.7}
$$

When $r ( \mathbf { x } ) = \delta _ { C } ( \mathbf { x } ) = { \left\{ \begin{array} { l l } { 0 , { \mathrm { ~ i f ~ } } \mathbf { x } \in C , } \\ { \infty , { \mathrm { ~ i f ~ } } \mathbf { x } \not \in C , } \end{array} \right. }$ this reduces to the projection operator

$$
\operatorname { p r o j } _ { C } ( \mathbf { x } ) = \arg \operatorname* { m i n } _ { \mathbf { y } \in C } { \frac { 1 } { 2 } } \| \mathbf { x } - \mathbf { y } \| ^ { 2 } .
$$

We say that an operator $F \colon  { \mathbb { R } ^ { d } } \to  { \mathbb { R } ^ { d } }$ is µ-strongly monotone, when

$$
\begin{array} { r } { \langle F ( \mathbf { x } ) - F ( \mathbf { y } ) , \mathbf { x } - \mathbf { y } \rangle \geq \mu \Vert \mathbf { x } - \mathbf { y } \Vert ^ { 2 } , } \end{array}
$$

and monotone when this inequality holds with $\mu = 0$

The operator F is $L _ { F } .$ -Lipschitz when

$$
\| F ( { \mathbf x } ) - F ( { \mathbf y } ) \| \le L _ { F } \| { \mathbf x } - { \mathbf y } \| .
$$

Recalling our mapping $\boldsymbol { F } = \big ( \mathrm { \frac { ~ \nabla _ { \mathbf { u } } f ( \mathbf { u } , \mathbf { v } ) ~ } { ~ - ~ \nabla _ { \mathbf { v } } f ( \mathbf { u } , \mathbf { v } ) } } \big )$ for a convex-concave $f ,$ this corresponds to Lipschitzness of the gradients of $\nabla _ { \mathbf { u } } f ( \mathbf { u } , \mathbf { v } ) , \nabla _ { \mathbf { v } } f ( \mathbf { u } , \mathbf { v } )$

Assumption 1. For problem (MI), let F be monotone and $L _ { F } – L i p s c h i t z .$ Assume that r is proper, convex, closed. There exists $\mathbf { x } ^ { \star }$ such that $0 \in ( F + \partial r ) ( \mathbf { x } ^ { \star } )$ . We have access to an unbiased estimator of F such that $\mathbb { E } [ F _ { \xi } ( { \bf x } ) ] = F ( { \bf x } )$

In view of the mapping (1.5), Assumption 1 is satisfied when $f ( { \bf u } , { \bf v } )$ is convex-concave (implies that F is monotone), the gradients $\nabla _ { \mathbf { u } } { f } ( \mathbf { u } , \mathbf { v } )$ and $\nabla _ { \mathbf { v } } { f } ( \mathbf { u } , \mathbf { v } )$ are Lipschitz continuous (implies that $F$ is Lipschitz) and $U , V$ are convex closed sets (implies the requirements on r).

We now present the Blum-Gladyshev (BG) assumption from [Gla65, Blu54] that is studied for minimization and min-max problems recently.

Assumption 2 (Blum-Gladyshev variance). Let the unbiased estimator of F satisfy

$$
\begin{array} { r } { \mathbb { E } \| F _ { \xi } ( \mathbf { x } ) - F ( \mathbf { x } ) \| ^ { 2 } \leq B ^ { 2 } \| \mathbf { x } - \mathring { \mathbf { z } } \| ^ { 2 } + G ^ { 2 } , } \end{array}
$$

where $\mathring \mathbf { z }$ is a fixed initial point. In view of Assumption 1, denote $\widehat { L } _ { F } = L _ { F } + B$

The reason for selecting ˚z as the center in the assumption is for convenience. One can change it to any other fixed center point, by absorbing the diference in the constant G and slightly changing $B .$

We also define the initial distance to a solution as (see (MI))

$$
D _ { \star } = \| \mathring { \mathbf z } - \mathbf { x } ^ { \star } \| .\tag{1.8}
$$

Let us first remark that this is a generalization of the commonly used bounded variance assumption that is used in $\left[ { \mathrm { A Z 1 8 , C L 2 4 } } \right]$ . If $B = 0$ in Assumption $2 ,$ this would correspond to assuming a bounded variance.

In the sequel, we require the knowledge of B, but not $G$ for setting algorithmic parameters. As we explain now, B is often connected to the Lipschitz constant and hence its knowledge should not be too restrictive in many cases.

Let us observe that a suficient condition for the BG assumption is when we have $F ( \mathbf { x } ) = \mathbb { E } [ F _ { \xi \sim P } ( \mathbf { x } ) ]$ and $F _ { \xi } ( \mathbf { x } )$ is mean-square Lipschitz, that is,

$$
\begin{array} { r } { \mathbb { E } \| F _ { \xi } ( \mathbf { x } ) - F _ { \xi } ( \mathbf { y } ) \| ^ { 2 } \leq L _ { \exp } ^ { 2 } \| \mathbf { x } - \mathbf { y } \| ^ { 2 } , } \end{array}
$$

and $\mathbb { E } \| F _ { \xi } ( \mathbf { x } ^ { \star } ) - F ( \mathbf { x } ^ { \star } ) \| ^ { 2 } \leq \sigma _ { \star } ^ { 2 }$ . This is suficient because of the chain of inequalities

$$
\begin{array} { r l } & { \mathbb { E } \| F _ { \xi } ( \mathbf { x } ) - F ( \mathbf { x } ) \| ^ { 2 } \leq 2 \mathbb { E } \| F _ { \xi } ( \mathbf { x } ) - F ( \mathbf { x } ) - F _ { \xi } ( \mathbf { x } ^ { \star } ) + F ( \mathbf { x } ^ { \star } ) \| ^ { 2 } + 2 \mathbb { E } \| F _ { \xi } ( \mathbf { x } ^ { \star } ) - F ( \mathbf { x } ^ { \star } ) \| ^ { 2 } } \\ & { \qquad \leq 2 \mathbb { E } \| F _ { \xi } ( \mathbf { x } ) - F _ { \xi } ( \mathbf { x } ^ { \star } ) \| ^ { 2 } + 2 \mathbb { E } \| F _ { \xi } ( \mathbf { x } ^ { \star } ) - F ( \mathbf { x } ^ { \star } ) \| ^ { 2 } } \\ & { \qquad \leq 2 L _ { \exp } ^ { 2 } \| \mathbf { x } - \mathbf { x } ^ { \star } \| ^ { 2 } + 2 \sigma _ { \star } ^ { 2 } } \\ & { \qquad \leq \underbrace { 4 L _ { \exp } ^ { 2 } \| \mathbf { x } - \hat { \mathbf { z } } \| ^ { 2 } } _ { B ^ { 2 } } + \underbrace { 4 L _ { \exp } ^ { 2 } \| \hat { \mathbf { z } } - \mathbf { x } ^ { \star } \| ^ { 2 } + 2 \sigma _ { \star } ^ { 2 } } _ { G ^ { 2 } } , } \end{array}
$$

where the first and last steps are by Young’s inequality, the second step is using $\begin{array} { r } { \mathbb { E } [ F _ { \xi } ( { \bf x } ) - F _ { \xi } ( { \bf x } ^ { \star } ) ] = } \end{array}$ $F ( \mathbf { x } ) - F ( \mathbf { x } ^ { \star } )$ – along with the inequality $\mathbb { E } \Vert \ b { X } - \mathbb { E } \ b { X } \Vert ^ { 2 } \leq \mathbb { E } \Vert \ b { X } \Vert ^ { 2 }$ for random vectors $X$ . This proves that, for example, $[ \mathrm { M K S ^ { + } 2 0 }$ , Assumptions 2, 4] implies Assumption 2.

Algorithm 1 Anchored stochastic extragradient $- \operatorname { A S E G } ( A + \partial r , \mathbf { x } _ { 0 } , \beta , \alpha , K )$   
Initial iterate $\mathbf { x } _ { \mathrm { 0 } } .$ , global initial point ${ \overset { \circ } { \mathbf { z } } } ,$ anchoring parameter $\beta ,$ step size $\alpha > 0$ , unbiased evaluations of A   
denoted as $A _ { \xi }$   
Denote $\mathbf { x } _ { i } \equiv \mathbf { \overset { \circ } { x } } _ { i } ^ { s , n }$ for $i = 0 , \ldots , K$   
for $k = 0 , 1 , 2 , \ldots , K - 1$ do   
$\bar { \mathbf { x } } _ { k } = \beta \bar { \mathbf { z } } + ( 1 - \beta ) \mathbf { x } _ { k }$   
$\mathbf { x } _ { k + 1 / 2 } = \mathrm { p r o x } _ { \alpha r } \big ( \bar { \mathbf { x } } _ { k } - \alpha A _ { \xi _ { k } } ( \mathbf { x } _ { k } ) \big )$   
$\mathbf { x } _ { k + 1 } = \mathrm { p r o x } _ { \alpha r } ( \bar { \mathbf { x } } _ { k } - \alpha A _ { \xi _ { k + 1 / 2 } } ( \mathbf { x } _ { k + 1 / 2 } ) )$   
end for   
Output: $\begin{array} { r } { \mathbf { y } _ { n } ^ { s } : = \hat { \mathbf { x } } _ { K } ^ { s , n } = \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathbf { x } _ { k + 1 / 2 } . } \end{array}$

```latex
Algorithm 2 $\mathtt { S E G ^ { S C } } ( A + \partial r , \mathbf { y } _ { 0 } , \mu _ { A } , \widehat { L } _ { A } , B _ { A } , T )$
Initial iterate $\mathbf { y } _ { 0 }$ , oracle budget $T ,$ first period $\begin{array} { r } { N = \lfloor \frac { T } { 8 \widehat { L } _ { A } / \mu _ { A } } \rfloor } \end{array}$ , second period $\begin{array} { r } { M = \lfloor \log _ { 2 } \frac { T } { 1 6 \widehat { L } _ { A } / \mu _ { A } } \rfloor } \end{array}$
Denote $\mathbf { y } _ { i } : = \mathbf { y } _ { i } ^ { s }$ for $i = 0 , 1 , \ldots , N + M$
for $n = 1 , 2 , \ldots , N + M$ do
if $n \leq N$ then
$\begin{array} { r } { \mathbf { y } _ { n } ^ { - } = { \tt A S E G } ( A + \partial r , \mathbf { y } _ { n - 1 } , 3 \alpha ^ { 2 } B _ { A } ^ { 2 } , \alpha \equiv \frac { 1 } { 2 \widehat { L } _ { A } } , \frac { 2 \widehat { L } _ { A } } { \mu _ { A } } ) } \end{array}$
else
$\begin{array} { r } { \tilde { \mathbf { y } } _ { n } = \mathtt { A S E G } ( A + \partial r , \mathbf { y } _ { n - 1 } , 3 \alpha ^ { 2 } B _ { A } ^ { 2 } , \alpha \equiv \frac { 1 } { 2 ^ { n - N } \widehat { L } _ { A } } , \frac { 2 ^ { n - N + 1 } \widehat { L } _ { A } } { \mu _ { A } } ) } \end{array}$
end if
end for
Output: $\mathbf z _ { s } = \mathbf y _ { N + M } ^ { s }$
```

A more concrete example is the bilinearly coupled min-max problem where a common oracle is by sampling row-column pairs from matrix $A \in \mathbb { R } ^ { m \times n }$ :

$$
F _ { \xi } ( \mathbf { x } ) = { \binom { m A _ { j : } v _ { j } } { - n A _ { : i } u _ { i } } } ,
$$

where $\xi = ( i , j )$ are selected uniformly at random, $A _ { : i }$ is i-th column of A and $A _ { j } { \mathrm { : } }$ is j-th row of A.

Then, the variance $\begin{array} { r } { \mathbb { E } _ { i , j } \| ( { \binom { m A _ { j : } v _ { j } } { - n A _ { : i } u _ { i } } } - { \binom { A ^ { \top } v } { - A \mathbf { u } } } \| ^ { 2 } } \end{array}$ will not be bounded in general, unless $\mathbf { u } , \mathbf { v }$ live on compact sets – an unrealistic assumption in general, preventing applying the results of the form [CL24] even for unconstrained problems. Yet, the variance will scale quadratically in the norm of $\mathbf { x } = { \binom { \mathbf { u } } { \mathbf { v } } }$ , satisfying the BG assumption. This example can be generalized to solve problems with other afine operators and stochastic oracles.

Byproducts of our analysis also include optimal complexity guarantees for the gradient mapping norm for strongly convex-strongly concave problems. We also design algorithms not requiring dificult-to-compute constants such as the initial distance to solution, that is, $\| \mathring { \mathbf z } - \mathbf x ^ { \star } \| ^ { 2 }$ , required in [CL24], or a global variance upper bound, required in the work [CL24].

## 2 Algorithm & Statement of Main Results

The main algorithmic construction will be based on a key idea from [AZ18] who had focused on the same goal for composite convex optimization: recursive regularization. Indeed, [CL24] had also considered extension of [AZ18] to the min-max case. However, their extension came with three main drawbacks that we will address: (1) their analysis was limited to the unconstrained problem, (2) their analysis introduced a requirement for knowing the global variance upper bound and initial distance to optimum $\| \mathring { \mathbf z } - \mathbf x ^ { \star } \| ^ { 2 }$ , (3) they inherited the drawback from [AZ18] and required a globally bounded variance. We will go around these limitations by using the BG variance assumption and anchoring, proposed for a similar purpose in [NO24] for proving a complexity result for the primal-dual gap.

```latex
Algorithm 3 Recursive regularization $( F ^ { \mu } + \partial r , \mathring { \mathbf { z } } , \mu , L _ { F } , B , T )$
Initial iterate ${ \bf z } _ { 0 } = \mathsf { \bar { z } } , \widehat { L } _ { F } = L _ { F } + B _ { \mathrm { \ell } }$ oracle budget $\begin{array} { r } { T \geq 4 8 \frac { \widehat { L } _ { F } } { \mu } \left\lfloor \log _ { 2 } \frac { \widehat { L } _ { F } } { \mu } \right\rfloor , \widehat { L } _ { s - 1 } = L _ { F } + B + ( 2 ^ { s } - 1 ) \mu } \end{array}$
${ \cal F } ^ { 0 } = { \cal F } ^ { \mu } , \mu _ { 0 } = \mu$
for $\begin{array} { r } { s = { 1 , 2 , . . . , S } = \lfloor \log _ { 2 } { \frac { \widehat { L } _ { F } } { \mu } } \rfloor } \end{array}$ do
$\begin{array} { r } { \mathbf { z } _ { s } = \mathtt { S E G } ^ { \mathtt { S C } } ( H ^ { s - 1 } : = F ^ { s - 1 } + \partial r , \mathbf { z } _ { s - 1 } , ( 2 ^ { s } - 1 ) \mu , \widehat { L } _ { s - 1 } , B , T / S ) } \end{array}$
$\mu _ { s } = 2 \mu _ { s - 1 }$
$F ^ { s } ( { \bf z } ) = F ^ { s - 1 } ( { \bf z } ) + \mu _ { s } ( { \bf z } - { \bf z } _ { s } )$
end for
```

Recursive regularization – Algorithm 3. In our setting, this corresponds to solving strongly monotone inclusion problems with an increasing amount of added strong monotonicity. In particular, this algorithm will give us an approximate solution for the problem

$$
\mathrm { { f i n d } } \ \mathbf { z } _ { 0 } ^ { \star } \ \mathrm { s u c h \ t h a t } \ 0 \in H ^ { \mu } ( \mathbf { z } _ { 0 } ^ { \star } ) : = F ^ { \mu } ( \mathbf { z } _ { 0 } ^ { \star } ) + \partial r ( \mathbf { z } _ { 0 } ^ { \star } ) ,\tag{2.1}
$$

where

$$
F ^ { 0 } ( \mathbf { z } ) : = F ^ { \mu } ( \mathbf { z } ) = { \left\{ \begin{array} { l l } { F ( \mathbf { z } ) { \mathrm { ~ i f ~ } } F { \mathrm { ~ i s ~ } } \mu { \mathrm { - s t r o n g l y ~ m o n o t o n e , } } } \\ { F ( \mathbf { z } ) + \mu ( \mathbf { z } - { \hat { \mathbf { z } } } ) { \mathrm { ~ f o r ~ s o m e ~ } } \mu > 0 { \mathrm { ~ i f ~ } } F { \mathrm { ~ i s ~ m o n o t o n e . } } } \end{array} \right. }\tag{2.2}
$$

This setting ensures that $F ^ { \mu }$ is µ-strongly monotone for some $\mu > 0 .$ , even if F is only monotone.

Then, each step $s \geq 1$ of Algorithm 3 solves approximately the subproblem:

find $\mathbf { z } _ { s - 1 } ^ { \star }$ such that $0 \in ( F ^ { s - 1 } + \partial r ) ( \mathbf { z } _ { s - 1 } ^ { \star } )$ 2

$$
\mathrm { w h e r e } \ F ^ { s - 1 } ( { \bf z } ) = F ^ { s - 2 } ( { \bf z } ) + \mu _ { s - 1 } ( { \bf z } - { \bf z } _ { s - 1 } ) = F ^ { \mu } ( { \bf z } ) + \sum _ { i = 1 } ^ { s - 1 } \mu _ { i } ( { \bf z } - { \bf z } _ { i } ) , \ \mathrm { i f } \ s \ge 2 ,\tag{2.3}
$$

and for $s = 1$ , we have that $F ^ { 0 }$ is as defined in (2.2) and $\mathbf { z } _ { 0 } ^ { \star }$ is defined in (2.1).

That is, at iteration s, we form an auxiliary problem which is centered at current iterate $\mathbf { z } _ { s - 1 }$ and then use a strongly monotone subsolver in Algorithm 2 to solve this problem. The amount of strong monotonicity added is increasing exponentially fast, that is $\mu _ { s } = 2 ^ { s } \mu$ where $\mu$ is the initial strong monotonicity of $F ^ { 0 } = F ^ { \mu }$ defined in (2.2).

Our main aim is to solve a monotone problem, but similar to [AZ18], the approach for the best complexity requires designing the strongly monotone solver with the gradient mapping guarantee (see Algorithm 3 and Theorem 2.1). After this, we will invoke this result on a perturbed problem where the initial strong monotonicity level will depend on the desired accuracy, see (2.2), case 2.

Strongly convex subsolver – Algorithm 2. The first layer was described in the previous section where we solve a sequence of strongly monotone subproblems. To solve these problems, we use a restarting-based second layer (Algorithm 2) similar to [AZ18]. This method restarts Algorithm 1 with two diferent step size and inner iteration limit pairs.

Main workhorse: Anchored stochastic extragradient – Algorithm 1. Anchoring is mainly used to handle the BG variance assumption (Assumption 2). Indeed, if the variance is bounded, then $B = 0$ and the algorithm reduces to regular stochastic extragradient method. However, the anchoring allows us to handle nonzero B, to cover problems without bounded variance.

We now continue with the summary of two main results that will be developed in the sequel.

## 2.1 Complexity for Strongly Monotone Problems

We now state the main complexity results for solving strongly monotone problems, which correspond to strongly convex-strongly concave instances of (1.1). We state this result under bounded variance, since we

don’t prove it under Blum-Gladyshev variance due to space constraints. The following result is proven in Section 4.1. A similar result under BG variance assumption can be proven by using the tools in Section 5. We omit this extension for brevity.

Corollary 2.1. For problem (MI), let Assumption 1 hold and suppose that Assumption 2 holds with $B = 0$ Additionally assume that F is $\mu _ { F }$ -strongly monotone. Let $S = \lfloor \log _ { 2 } ( L _ { F } / \mu _ { F } ) \rfloor \ge 1 , T \ge 4 8 S L _ { F } / \mu _ { F }$ and $\begin{array} { r } { \eta = \frac { 1 } { 3 L _ { F } } } \end{array}$ . Then, we have that

$$
\mathbb { E } \| \mathcal { G } _ { \eta , H } ( \mathbf { z } _ { S } ) \| \le \varepsilon ,\tag{2.4}
$$

where the number of stochastic first-order oracles T is upper bounded by

$$
O \left( \frac { L _ { F } } { \mu _ { F } } \log \left( \frac { L _ { F } } { \mu _ { F } } \right) \log \left( \frac { \mu _ { F } D _ { \star } } { \varepsilon } + e \right) + \frac { G ^ { 2 } \log ^ { 3 } ( L _ { F } / \mu _ { F } ) } { \varepsilon ^ { 2 } } \right) .
$$

where O only suppresses the absolute constants.

Even for unconstrained problems, our result extends [CL24, Theorem 4.1] since our algorithmic parameters do not depend on $\| \mathbf { z } _ { 0 } - \mathbf { x } ^ { \star } \|$ or a global variance upper bound $\sigma ^ { 2 }$ , which were required for running the algorithm in [CL24, Theorem 4.1]. The parameters in our algorithm only depend on the oracle budget $T , L _ { F }$ and $\mu _ { F }$

## 2.2 Complexity for Monotone Problems

By using the result for the strongly monotone case with a perturbed version of the monotone problem, we can obtain the claimed complexity guarantee for solving the original monotone problem. For this we need to connect the solutions in terms of the gradient mapping norm and use the previous result. In particular, we will solve

$$
\mathrm { f i n d } \ \mathbf { z } _ { 0 } ^ { \star } \mathrm { ~ s u c h ~ t h a t ~ } 0 \in H ^ { \mu } ( \mathbf { z } _ { 0 } ^ { \star } ) : = F ( \mathbf { z } _ { 0 } ^ { \star } ) + \partial r ( \mathbf { z } _ { 0 } ^ { \star } ) + \mu ( \mathbf { z } _ { 0 } ^ { \star } - \bar { \mathbf { z } } ) ,\tag{2.5}
$$

where $\textstyle \mu = { \widetilde { O } } \left( { \frac { 1 } { \sqrt { T } } } \right)$ . Since $\mu$ is small depending on the oracle budget T, the perturbed problem is suficiently close to the original problem to yield the claimed complexity result on the gradient mapping. The proof of the following result is the focus of Section 5.

Corollary 2.2. For problem (MI), let Assumptions 1 and 2 hold. We apply Algorithm 3 to solve (2.1) with $\begin{array} { r } { \mu = \frac { 3 2 \widehat { L } _ { F } } { \sqrt { T } } \log _ { 2 } ^ { 3 / 2 } \Big ( \frac { \sqrt { T } } { 4 8 } \Big ) , T \geq 1 4 1 7 5 , \eta = \frac { 1 } { 3 \widehat { L } _ { F } } } \end{array}$ . Then, we have that

$$
\mathbb { E } \| \mathcal { G } _ { \eta , H } ( \mathbf { z } _ { S } ) \| = O \left( \frac { \widehat { L } _ { F } D _ { \star } ( \ln \sqrt { T } ) ^ { 3 / 2 } } { \sqrt { T } } + \frac { ( \ln \sqrt { T } ) ^ { 5 / 2 } ( B D _ { \star } + G ) } { \sqrt { T } } \right) .\tag{2.6}
$$

Consequently, to obtain

$$
\mathbb { E } \| \mathcal { G } _ { \eta , H } ( \mathbf { z } _ { S } ) \| \le \varepsilon ,\tag{2.7}
$$

the number of required stochastic first-order oracles $T$ is upper bounded by

$$
O \left( 1 + \frac { ( L _ { F } ^ { 2 } + B ^ { 2 } ) D _ { \star } ^ { 2 } + G ^ { 2 } } { \varepsilon ^ { 2 } } \ln ^ { 5 } \left( \frac { D _ { \star } ( L _ { F } + B ) + G } { \varepsilon } + 1 \right) \right) ,
$$

where we only suppress the absolute constants.

A similar discussion as the previous subsection apply here to compare our result with the corresponding result of [CL24]. In particular, we not only improve over this result to handle constrained or regularized problems, but also require a weaker variance assumption, and use algorithmic parameters only depending on $L _ { F } , T , B$ . An additional point of discussion is the parameter $\mu ,$ which is independent of $\| \mathring { \mathbf { z } } - \mathbf { x } ^ { \star } \| ^ { 2 }$ , whereas this quantity was used to set $\mu$ in the work of [CL24]. To transfer our guarantees to a guarantee on the stronger tangent residual (see its definition, e.g., in [COZ22]), one can apply a postprocessing step similar to [CAD24, Appendix C.3].

## 3 Analysis for Subsolvers

## 3.1 Layer 1: Anchored Stochastic Extragradient

We now analyze Alg. 1 which is the main workhorse of the construction. It is an extragradient-based method for solving a strongly monotone inclusion problem. The addition on top of the standard extragradient is the Halpern-type anchoring step [Hal67].

Even though we are not aware of this algorithm being proposed or analyzed before; in spirit, it can be considered as a stochastic version of regular anchored extragradient (see [YR21] and [KG22, Algorithm 3]) or a single-sample version of [LK21], or non-variance reduced version of [AMW25, Section 4.3]. Our choice for building on extragradient is purely for presentation purposes, as similar bounds can be proven for anchored versions of other algorithms.

All these works above focused on the non-strongly monotone case. However, our subproblems are strongly monotone, so we analyze for this case. We will call this method and its complexity result for a series of subproblems, so we introduce the problem:

$$
\mathrm { f i n d } \ \mathbf { x } _ { A } ^ { \star } \ \mathrm { s u c h \ t h a t } \ 0 \in ( A + \partial r ) ( \mathbf { x } _ { A } ^ { \star } ) ,\tag{3.1}
$$

when A is strongly monotone. At each step s of Algorithm 3, we will change A depending on $F ^ { s - 1 }$ , see (2.3). For convenience, let us introduce the assumptions to be used in this and next subsection.

Assumption 3. For problem (3.1), let A be monotone, L<sub>A</sub>-Lipschitz and $\mu _ { A }$ -strongly monotone. Assume that r is proper, convex, closed. There exists $\mathbf { x } _ { A } ^ { \star }$ such that $0 \in ( A + \partial r ) ( \mathbf { x } _ { A } ^ { \star } )$ We have access to an unbiased estimator of A such that $\mathbb { E } [ A _ { \xi } ( { \bf x } ) ] = A ( { \bf x } )$ and

$$
\begin{array} { r } { \mathbb { E } \| A _ { \xi } ( \mathbf { x } ) - A ( \mathbf { x } ) \| ^ { 2 } \leq B _ { A } ^ { 2 } \| \mathbf { x } - \mathring { \mathbf { z } } \| ^ { 2 } + G _ { A } ^ { 2 } , } \end{array}
$$

where ˚z is a globally fixed iterate.

To see the connection between the parameters $B _ { A } , G _ { A }$ of the variance assumption in this section and the variance assumption made for the global problem (Assumption 2), see Fact 6.5.

The following lemma, when $\beta = 0$ is completely classical, see [JNT11] for an early reference. We provide a generalization here with the anchoring step. We build on [NO24, AMW25] who analyzed similar algorithms in diferent contexts, without strong monotonicity. The main idea is to use the anchoring with weight $\beta$ to absorb the contribution coming from the iterate-dependent $B _ { A }$ term in the BG assumption (Assumption 3), which allows relaxing the bounded variance assumption.

Lemma 3.1. For problem (3.1), let Assumption 3 hold. When we run Algorithm 1 for K iterations, with parameters $\begin{array} { r } { \alpha \le \frac { 1 } { 2 ( L _ { A } + B _ { A } ) } , \beta = 3 \alpha ^ { 2 } B _ { A } ^ { 2 } } \end{array}$ , we obtain

$$
\mathbb { E } \| \hat { { \mathbf { x } } } _ { K } ^ { s , n } - { \mathbf { x } } _ { A } ^ { \star } \| ^ { 2 } \leq \frac { 1 } { 2 \alpha \mu _ { A } K } \| { \mathbf { x } } _ { 0 } ^ { s , n } - { \mathbf { x } } _ { A } ^ { \star } \| ^ { 2 } + \frac { 3 \alpha } { \mu _ { A } } \left( B _ { A } ^ { 2 } \| \hat { { \mathbf { z } } } - { \mathbf { x } } _ { A } ^ { \star } \| ^ { 2 } + \frac { 7 G _ { A } ^ { 2 } } { 1 8 } \right) .
$$

Remark 3.2. In this bound, the additional term $B _ { A } ^ { 2 } \| \mathring { \mathbf { z } } - \mathbf { x } _ { A } ^ { \star } \| ^ { 2 }$ coming from the relaxed variance requirement seems not problematic since $\mathbf { x } _ { A } ^ { \star }$ is fixed. However, we will later invoke this result with a dynamic set of problems where the solution $\mathbf { x } _ { A } ^ { \star }$ will be changing at each step. As a result, we cannot treat this term as a constant. We will have to craft a specialized analysis to handle this additional term since we do not have a global upper bound for the solutions of inner problems. This will be handled in Section 5.

Proof of Theorem 3.1. For a lighter notation, we suppress the superscript s, n from the iterates, as in the algorithm.

By the update rules of the iterates $\mathbf { x } _ { k + 1 / 2 } , \mathbf { x } _ { k + 1 }$ in Algorithm 1, and (1.7), we have

$$
\begin{array} { r } { \langle \mathbf { x } _ { k + 1 / 2 } - \bar { \mathbf { x } } _ { k } + \alpha A _ { \xi _ { k } } ( \mathbf { x } _ { k } ) , \mathbf { x } _ { k + 1 } - \mathbf { x } _ { k + 1 / 2 } \rangle \geq \alpha \big ( r ( \mathbf { x } _ { k + 1 / 2 } ) - r ( \mathbf { x } _ { k + 1 } ) \big ) , } \end{array}
$$

$$
\begin{array} { r } { \langle \mathbf { x } _ { k + 1 } - \bar { \mathbf { x } } _ { k } + \alpha A _ { \xi _ { k + 1 / 2 } } ( \mathbf { x } _ { k + 1 / 2 } ) , \mathbf { x } _ { A } ^ { \star } - \mathbf { x } _ { k + 1 } \rangle \geq \alpha ( r ( \mathbf { x } _ { k + 1 } ) - r ( \mathbf { x } _ { A } ^ { \star } ) ) . } \end{array}
$$

We sum up these inequalities and rearrange to obtain

$$
\begin{array} { r l } & { \alpha \big ( r ( \mathbf { x } _ { k + 1 / 2 } ) - r ( \mathbf { x } _ { A } ^ { \star } ) + \langle A _ { \xi _ { k + 1 / 2 } } ( \mathbf { x } _ { k + 1 / 2 } ) , \mathbf { x } _ { k + 1 / 2 } - \mathbf { x } _ { A } ^ { \star } \rangle \big ) } \\ & { \quad \leq \langle \mathbf { x } _ { k + 1 / 2 } - \bar { \mathbf { x } } _ { k } , \mathbf { x } _ { k + 1 } - \mathbf { x } _ { k + 1 / 2 } \rangle + \langle \mathbf { x } _ { k + 1 } - \bar { \mathbf { x } } _ { k } , \mathbf { x } _ { A } ^ { \star } - \mathbf { x } _ { k + 1 } \rangle } \\ & { \quad ~ + \alpha \langle A _ { \xi _ { k } } ( \mathbf { x } _ { k } ) - A _ { \xi _ { k + 1 / 2 } } ( \mathbf { x } _ { k + 1 / 2 } ) , \mathbf { x } _ { k + 1 } - \mathbf { x } _ { k + 1 / 2 } \rangle . } \end{array}\tag{3.2}
$$

We estimate the first two terms on the right-hand side. Applying $\| \mathbf { a } + \mathbf { b } \| ^ { 2 } = \| \mathbf { a } \| ^ { 2 } + \| \mathbf { b } \| ^ { 2 } + 2 \langle \mathbf { a } , \mathbf { b } \rangle$ gives

$$
\begin{array} { r l } & { 2 \langle \mathbf { x } _ { k + 1 / 2 } - \bar { \mathbf { x } } _ { k } , \mathbf { x } _ { k + 1 } - \mathbf { x } _ { k + 1 / 2 } \rangle = \| \mathbf { x } _ { k + 1 } - \bar { \mathbf { x } } _ { k } \| ^ { 2 } - \| \mathbf { x } _ { k + 1 / 2 } - \bar { \mathbf { x } } _ { k } \| ^ { 2 } - \| \mathbf { x } _ { k + 1 } - \mathbf { x } _ { k + 1 / 2 } \| ^ { 2 } , } \\ & { \qquad 2 \langle \mathbf { x } _ { k + 1 } - \bar { \mathbf { x } } _ { k } , \mathbf { x } _ { A } ^ { \star } - \mathbf { x } _ { k + 1 } \rangle = \| \mathbf { x } _ { A } ^ { \star } - \bar { \mathbf { x } } _ { k } \| ^ { 2 } - \| \mathbf { x } _ { k + 1 } - \bar { \mathbf { x } } _ { k } \| ^ { 2 } - \| \mathbf { x } _ { A } ^ { \star } - \mathbf { x } _ { k + 1 } \| ^ { 2 } . } \end{array}
$$

Plugging in these identities to (3.2), after multiplying both sides by $2 ,$ and taking expectation give

$$
\begin{array} { r l } & { 2 \alpha \mathbb { E } [ r ( \mathbf { x } _ { k + 1 / 2 } ) - r ( \mathbf { x } _ { A } ^ { \star } ) + \langle A _ { \xi _ { k + 1 / 2 } } ( \mathbf { x } _ { k + 1 / 2 } ) , \mathbf { x } _ { k + 1 / 2 } - \mathbf { x } _ { A } ^ { \star } \rangle ] } \\ & { \leq \mathbb { E } \left[ \| \mathbf { x } _ { A } ^ { \star } - \bar { \mathbf { x } } _ { k } \| ^ { 2 } - \| \mathbf { x } _ { A } ^ { \star } - \mathbf { x } _ { k + 1 } \| ^ { 2 } - \| \mathbf { x } _ { k + 1 } - \mathbf { x } _ { k + 1 / 2 } \| ^ { 2 } - \| \mathbf { x } _ { k + 1 / 2 } - \bar { \mathbf { x } } _ { k } \| ^ { 2 } \right] } \\ & { \quad + 2 \alpha \mathbb { E } \langle A _ { \xi _ { k } } ( \mathbf { x } _ { k } ) - A _ { \xi _ { k + 1 / 2 } } ( \mathbf { x } _ { k + 1 / 2 } ) , \mathbf { x } _ { k + 1 } - \mathbf { x } _ { k + 1 / 2 } \rangle . } \end{array}\tag{3.3}
$$

We now estimate the inner product on the right-hand side by

$$
\begin{array} { r l } & { 2 \alpha \mathbb { E } \langle A _ { \xi _ { k } } ( \mathbf { x } _ { k } ) - A _ { \xi _ { k + 1 / 2 } } ( \mathbf { x } _ { k + 1 / 2 } ) , \mathbf { x } _ { k + 1 } - \mathbf { x } _ { k + 1 / 2 } \rangle } \\ & { \leq \mathbb { E } \left[ \alpha ^ { 2 } \| A _ { \xi _ { k } } ( \mathbf { x } _ { k } ) - A _ { \xi _ { k + 1 / 2 } } ( \mathbf { x } _ { k + 1 / 2 } ) \| ^ { 2 } + \| \mathbf { x } _ { k + 1 } - \mathbf { x } _ { k + 1 / 2 } \| ^ { 2 } \right] } \\ & { \leq \alpha ^ { 2 } \mathbb { E } \left[ \frac { 4 } { 3 } \| A _ { \xi _ { k } } ( \mathbf { x } _ { k } ) - A ( \mathbf { x } _ { k } ) \| ^ { 2 } + \| A ( \mathbf { x } _ { k + 1 / 2 } ) - A _ { \xi _ { k + 1 / 2 } } ( \mathbf { x } _ { k + 1 / 2 } ) \| ^ { 2 } + 4 L _ { A } ^ { 2 } \| \mathbf { x } _ { k + 1 / 2 } - \mathbf { x } _ { k } \| ^ { 2 } \right] } \\ & { \quad + \mathbb { E } \| \mathbf { x } _ { k + 1 } - \mathbf { x } _ { k + 1 / 2 } \| ^ { 2 } . } \end{array}\tag{3.4}
$$

Here, the first step is by Young’s inequality and the second step used $L _ { A } .$ -Lipschitzness of A after applying the estimation

$$
\begin{array} { r l } & { \mathbb { E } \left[ \| A _ { \xi _ { k } } ( \mathbf { x } _ { k } ) - A _ { \xi _ { k + 1 / 2 } } ( \mathbf { x } _ { k + 1 / 2 } ) \| ^ { 2 } \right] } \\ & { = \mathbb { E } \left[ \| A _ { \xi _ { k } } ( \mathbf { x } _ { k } ) - A ( \mathbf { x } _ { k + 1 / 2 } ) \| ^ { 2 } + \| A ( \mathbf { x } _ { k + 1 / 2 } ) - A _ { \xi _ { k + 1 / 2 } } ( \mathbf { x } _ { k + 1 / 2 } ) \| ^ { 2 } \right] } \\ & { \leq \mathbb { E } \left[ \frac { 4 } { 3 } \| A _ { \xi _ { k } } ( \mathbf { x } _ { k } ) - A ( \mathbf { x } _ { k } ) \| ^ { 2 } + 4 \| A ( \mathbf { x } _ { k } ) - A ( \mathbf { x } _ { k + 1 / 2 } ) \| ^ { 2 } + \| A ( \mathbf { x } _ { k + 1 / 2 } ) - A _ { \xi _ { k + 1 / 2 } } ( \mathbf { x } _ { k + 1 / 2 } ) \| ^ { 2 } \right] , } \end{array}
$$

where the first line is by tower property, since $\mathbb { E } _ { k } [ A ( \mathbf { x } _ { k + 1 / 2 } ) - A _ { \xi _ { k + 1 / 2 } } ( \mathbf { x } _ { k + 1 / 2 } ) ] = 0$ and $A _ { \xi _ { k } } ( \mathbf x _ { k } ) - A ( \mathbf x _ { k + 1 / 2 } )$ is measurable under the conditioning of $\mathbb { E } _ { k }$

For the inner product on the left-hand side of (3.3), we use strong monotonicity of A to derive

$$
\begin{array} { r l } & { \mathbb { E } \langle A _ { \xi _ { k + 1 / 2 } } ( \mathbf { x } _ { k + 1 / 2 } ) , \mathbf { x } _ { k + 1 / 2 } - \mathbf { x } _ { A } ^ { \star } \rangle = \mathbb { E } \langle A ( \mathbf { x } _ { k + 1 / 2 } ) , \mathbf { x } _ { k + 1 / 2 } - \mathbf { x } _ { A } ^ { \star } \rangle } \\ & { \qquad \geq \mathbb { E } [ \langle A ( \mathbf { x } _ { A } ^ { \star } ) , \mathbf { x } _ { k + 1 / 2 } - \mathbf { x } _ { A } ^ { \star } \rangle + \mu _ { A } \| \mathbf { x } _ { A } ^ { \star } - \mathbf { x } _ { k + 1 / 2 } \| ^ { 2 } ] } \\ & { \qquad \geq \mathbb { E } [ r ( \mathbf { x } _ { A } ^ { \star } ) - r ( \mathbf { x } _ { k + 1 / 2 } ) + \mu _ { A } \| \mathbf { x } _ { A } ^ { \star } - \mathbf { x } _ { k + 1 / 2 } \| ^ { 2 } ] , } \end{array}\tag{3.5}
$$

where the first identity used the tower property, $\mathbb { E } _ { k } [ A _ { \xi _ { k + 1 / 2 } } ( \mathbf { x } _ { k + 1 / 2 } ) ] = A ( \mathbf { x } _ { k + 1 / 2 } )$ , and that $\mathbf { x } _ { k + 1 / 2 } - \mathbf { x } _ { A } ^ { \star }$ is measurable under the conditioning of $\mathbb { E } _ { k }$ . The third step used convexity of r as well as $- A ( \mathbf { x } _ { A } ^ { \star } ) \in \partial r ( \mathbf { x } _ { A } ^ { \star } )$ by the definition of the solution $\mathbf { x } _ { A } ^ { \star }$ in (3.1). Using (3.4) and (3.5) in (3.3) gives

$$
\begin{array} { r l } & { 2 \alpha \mu _ { A } \mathbb { E } \| { \mathbf { x } } _ { A } ^ { \star } - { \mathbf { x } } _ { k + 1 / 2 } \| ^ { 2 } \leq \mathbb { E } \left[ \| { \mathbf { x } } _ { A } ^ { \star } - \bar { \mathbf { x } } _ { k } \| ^ { 2 } - \| { \mathbf { x } } _ { A } ^ { \star } - { \mathbf { x } } _ { k + 1 } \| ^ { 2 } - \| \bar { \mathbf { x } } _ { k } - { \mathbf { x } } _ { k + 1 / 2 } \| ^ { 2 } \right] } \\ & { \qquad + \alpha ^ { 2 } \mathbb { E } \left[ \frac { 4 } { 3 } \| A ( \mathbf { x } _ { k } ) - A _ { \xi _ { k } } ( \mathbf { x } _ { k } ) \| ^ { 2 } + \| A ( \mathbf { x } _ { k + 1 / 2 } ) - A _ { \xi _ { k + 1 / 2 } } ( \mathbf { x } _ { k + 1 / 2 } ) \| ^ { 2 } \right] } \\ & { \qquad + 4 \alpha ^ { 2 } L _ { A } ^ { 2 } \mathbb { E } \| \mathbf { x } _ { k } - \mathbf { x } _ { k + 1 / 2 } \| ^ { 2 } . } \end{array}\tag{3.6}
$$

On the one hand, by the definition of $\bar { \mathbf { x } } _ { k }$ in Algorithm 1, we have

$$
\begin{array} { r } { \| \mathbf { x } _ { A } ^ { \star } - \bar { \mathbf { x } } _ { k } \| ^ { 2 } - \| \bar { \mathbf { x } } _ { k } - \mathbf { x } _ { k + 1 / 2 } \| ^ { 2 } = \beta ( \| \mathbf { x } _ { A } ^ { \star } - \hat { \mathbf { z } } \| ^ { 2 } - \| \hat { \mathbf { z } } - \mathbf { x } _ { k + 1 / 2 } \| ^ { 2 } ) + ( 1 - \beta ) ( \| \mathbf { x } _ { A } ^ { \star } - \mathbf { x } _ { k } \| ^ { 2 } - \| \mathbf { x } _ { k } - \mathbf { x } _ { k + 1 / 2 } \| ^ { 2 } ) } \end{array}
$$

and, by Young’s inequality,

$$
- \beta \| { \bf x } _ { A } ^ { \star } - { \bf x } _ { k } \| ^ { 2 } \leq - \frac { \beta } { 2 } \| \mathring { \bf z } - { \bf x } _ { k } \| ^ { 2 } + \beta \| \mathring { \bf z } - { \bf x } _ { A } ^ { \star } \| ^ { 2 } .
$$

On the other hand, to handle the terms in the second line of (3.6), we have by Assumption 3 that

$$
\begin{array} { r l } & { \alpha ^ { 2 } \mathbb { E } \left[ \frac { 4 } { 3 } \| A ( \mathbf { x } _ { k } ) - A _ { \xi _ { k } } ( \mathbf { x } _ { k } ) \| ^ { 2 } + \| A ( \mathbf { x } _ { k + 1 / 2 } ) - A _ { \xi _ { k + 1 / 2 } } ( \mathbf { x } _ { k + 1 / 2 } ) \| ^ { 2 } \right] } \\ & { \quad \leq \alpha ^ { 2 } B _ { A } ^ { 2 } \mathbb { E } \left[ \frac { 4 } { 3 } \| \mathbf { x } _ { k } - \mathring { \mathbf { z } } \| ^ { 2 } + \| \mathbf { x } _ { k + 1 / 2 } - \mathring { \mathbf { z } } \| ^ { 2 } \right] + \frac { 7 } { 3 } \alpha ^ { 2 } G _ { A } ^ { 2 } . } \end{array}
$$

Combining the last three estimates in (3.6) gives

$$
\begin{array} { r l } & { 2 \alpha \mu _ { A } \mathbb { E } \| \mathbf { x } _ { A } ^ { \star } - \mathbf { x } _ { k + 1 / 2 } \| ^ { 2 } \leq \mathbb { E } \left[ \| \mathbf { x } _ { A } ^ { \star } - \mathbf { x } _ { k } \| ^ { 2 } - \| \mathbf { x } _ { A } ^ { \star } - \mathbf { x } _ { k + 1 } \| ^ { 2 } + 2 \beta \| \hat { \mathbf { z } } - \mathbf { x } _ { A } ^ { \star } \| ^ { 2 } \right] + \frac { 7 } { 3 } \alpha ^ { 2 } G _ { A } ^ { 2 } } \\ & { \qquad + \left( \frac { 4 \alpha ^ { 2 } B _ { A } ^ { 2 } } { 3 } - \frac { \beta } { 2 } \right) \mathbb { E } \| \hat { \mathbf { z } } - \mathbf { x } _ { k } \| ^ { 2 } + ( \alpha ^ { 2 } B _ { A } ^ { 2 } - \beta ) \mathbb { E } \| \tilde { \mathbf { z } } - \mathbf { x } _ { k + 1 / 2 } \| ^ { 2 } } \\ & { \qquad + \left( 4 \alpha ^ { 2 } L _ { A } ^ { 2 } - ( 1 - \beta ) \right) \mathbb { E } \| \mathbf { x } _ { k } - \mathbf { x } _ { k + 1 / 2 } \| ^ { 2 } . } \end{array}\tag{3.7}
$$

By the requirements on α and $\beta ,$ we have that the last three terms on the right-hand side are nonpositive. Dividing both sides by $2 \alpha \mu _ { A }$ , summing for $k = 0 , \ldots , K - 1$ , dividing by K and using the definition of $\hat { \mathbf { x } } _ { K } ^ { s , n }$ from Algorithm 1 gives the result. ■

## 3.2 Layer 2: Restarting for Better Complexity

The following result analyzes the restart-based instantiations of the inner solver and the argument is the same as [AZ18] (see also [HK14]), invoked with slightly diferent parameters and constants.

Lemma 3.3. For problem (3.1), let Assumption 3 hold. When we run Algorithm 2 with budget $T \geq 3 2 \widehat { L } _ { A } / \mu _ { A }$ for the number of oracles and initial point $\mathbf { y } _ { 0 }$ , we have

$$
\mathbb { E } \Vert \mathbf { y } _ { N + M } ^ { s } - \mathbf { x } _ { A } ^ { \star } \Vert ^ { 2 } \leq \left( \frac { 1 } { 2 } \right) ^ { \frac { T } { 8 { \tilde { L } } _ { A } / \mu _ { A } } } \Vert \mathbf { y } _ { 0 } ^ { s } - \mathbf { x } _ { A } ^ { \star } \Vert ^ { 2 } + \frac { 1 9 2 B _ { A } ^ { 2 } \Vert \bar { \mathbf { z } } - \mathbf { x } _ { A } ^ { \star } \Vert ^ { 2 } + 7 5 G _ { A } ^ { 2 } } { T \mu _ { A } ^ { 2 } } ,
$$

where $\widehat { L } _ { A } = L _ { A } + B _ { A }$ . The total number of oracle calls is upper bounded by $T ,$

Proof. For a lighter notation, we omit the superscript s as in the algorithm and also suppress the subscript for $\mu _ { A } , \widehat { L } _ { A } , B _ { A } , G _ { A }$ throughout the proof. Let us use the notations

$$
\widehat { G } ^ { 2 } = B ^ { 2 } \| \mathring { \mathbf { z } } - \mathbf { x } _ { A } ^ { \star } \| ^ { 2 } + \frac { 7 G ^ { 2 } } { 1 8 } , \quad \mathrm { a n d } \quad \widehat { L } = L + B .
$$

We invoke Theorem 3.1 with $\begin{array} { r } { \alpha = \frac { 1 } { 2 \widehat { L } } } \end{array}$ and $\begin{array} { r } { K = \frac { 2 \widehat { L } } { \mu } } \end{array}$ . Since $\alpha \mu K = 1$ and $\textstyle { \frac { 3 \alpha } { \mu } } \ = \ { \frac { 3 } { 2 { \widehat { L } } \mu } }$ , we have for each $n = 1 , \ldots , N$ (where the initial point to Algorithm 1 is $\mathbf { y } _ { n - 1 } )$

$$
\begin{array} { r l r } {  { \mathbb { E } \| { \mathbf { y } } _ { n } - { \mathbf { x } } _ { A } ^ { \star } \| ^ { 2 } \leq \frac { 1 } { 2 } \mathbb { E } \| { \mathbf { y } } _ { n - 1 } - { \mathbf { x } } _ { A } ^ { \star } \| ^ { 2 } + \frac { 3 } { 2 \widehat { L } \mu } \widehat { G } ^ { 2 } } } \\ & { } & \\ & { } & { \leq \frac { 1 } { 2 ^ { n } } \| { \mathbf { y } } _ { 0 } - { \mathbf { x } } _ { A } ^ { \star } \| ^ { 2 } + \frac { 3 \widehat { G } ^ { 2 } } { 2 \widehat { L } \mu } \sum _ { i = 0 } ^ { n - 1 } \frac { 1 } { 2 ^ { i } } . } \end{array}
$$

At $n = N .$ , this gives us

$$
\mathbb { E } \| { \bf y } _ { N } - { \bf x } _ { A } ^ { \star } \| ^ { 2 } \leq \frac { 1 } { 2 ^ { N } } \| { \bf y } _ { 0 } - { \bf x } _ { A } ^ { \star } \| ^ { 2 } + \frac { 3 \widehat { G } ^ { 2 } } { \widehat { L } \mu } ( 1 - 2 ^ { - N } ) ,\tag{3.8}
$$

where the last term is by summing the geometric series.

We continue to analyze the iterations of Algorithm 2 when $n = N + 1$ to $N + M$ . By the setting of α and K in this case, we have $\alpha \mu K = 2$ and $\begin{array} { r } { \frac { 3 \alpha } { \mu } = \frac { - 3 } { 2 ^ { n - N } \widehat { L } \mu } } \end{array}$ . We invoke Theorem 3.1 at $n = N + M$ and then unroll until $n = N + 1$ to obtain

$$
\begin{array} { r l } { \displaystyle \mathbb { E } \| { \mathbf { y } } _ { N + M } - { \mathbf { x } } _ { A } ^ { \star } \| ^ { 2 } \leq \frac { 1 } { 4 } \mathbb { E } \| { \mathbf { y } } _ { N + M - 1 } - { \mathbf { x } } _ { A } ^ { \star } \| ^ { 2 } + \frac { 3 { \widehat { G } } ^ { 2 } } { 2 ^ { M } { \widehat { L } } \mu } ~ } & { } \\ { \leq \frac { 1 } { 4 ^ { M } } \mathbb { E } \| { \mathbf { y } } _ { N } - { \mathbf { x } } _ { A } ^ { \star } \| ^ { 2 } + \frac { 3 { \widehat { G } } ^ { 2 } } { 2 ^ { M } { \widehat { L } } \mu } \displaystyle \sum _ { i = 0 } ^ { M - 1 } \frac { 1 } { 2 ^ { i } } ~ } & { } \\ { \leq \frac { 1 } { 4 ^ { M } } \mathbb { E } \| { \mathbf { y } } _ { N } - { \mathbf { x } } _ { A } ^ { \star } \| ^ { 2 } + \frac { 3 { \widehat { G } } ^ { 2 } } { 2 ^ { M - 1 } { \widehat { L } } \mu } ( 1 - 2 ^ { - M } ) , } \end{array}
$$

where the last line is by summing the geometric series. On the last inequality, we plug in the bound of $\mathbb { E } \| \mathbf { y } _ { N } - \mathbf { x } _ { A } ^ { \star } \| ^ { 2 }$ from (3.8) to deduce

$$
\mathbb { E } \| { \bf y } _ { N + M } - { \bf x } _ { A } ^ { \star } \| ^ { 2 } \leq \frac { 1 } { 2 ^ { N + 2 M } } \| { \bf y } _ { 0 } - { \bf x } _ { A } ^ { \star } \| ^ { 2 } + \frac { 6 \widehat { G } ^ { 2 } } { 2 ^ { M } \widehat { L } \mu } .
$$

Since $\begin{array} { r } { N \ge \frac { T } { 8 \widehat { L } / \mu } - 1 , 2 ^ { M } \ge \frac { T } { 3 2 \widehat { L } / \mu } } \end{array}$ , and $M \geq 1$ , we have

$$
\mathbb { E } \| { \bf y } _ { N + M } - { \bf x } _ { A } ^ { \star } \| ^ { 2 } \leq \left( \frac { 1 } { 2 } \right) ^ { \frac { T } { 8 \hat { L } / \mu } } \| { \bf y } _ { 0 } - { \bf x } _ { A } ^ { \star } \| ^ { 2 } + \frac { 1 9 2 \widehat { G } ^ { 2 } } { T \mu ^ { 2 } } .
$$

Moreover, the total number of calls to $A _ { \xi }$ is less than

$$
2 \times \frac { 2 \widehat { L } } { \mu } \times N + \sum _ { m = 1 } ^ { M } 2 \times 2 ^ { m + 1 } \times \frac { \widehat { L } } { \mu } \leq \frac { T } { 2 } + 2 ^ { M } \times \frac { 8 \widehat { L } } { \mu } \leq T ,
$$

since $\begin{array} { r } { N \leq \frac { T } { 8 \widehat { L } / \mu } } \end{array}$ and $\begin{array} { r } { 2 ^ { M } \leq \frac { T } { 1 6 \widehat { L } / \mu } } \end{array}$

## 4 Complexity Analysis with Bounded Variance

We now analyze Algorithm 3. This method calls Algorithm 2 repeatedly for S times where each step approximates the problem in (2.3).

For analyzing this scheme, we will invoke Theorem 3.3, with $\mathbf { x } _ { A } ^ { \star } \gets \mathbf { z } _ { s - 1 } ^ { \star }$ and $\mathbf { y } _ { 0 } \gets \mathbf { z } _ { s - 1 }$ . Since the last term on the right-hand side of the result of Theorem 3.3 has a term depending on $B ^ { 2 } \| \mathring { \mathbf { z } } - \mathbf { z } _ { s - 1 } ^ { \star } \| ^ { 2 }$ , we need a dedicated analysis to control this term since such a term did not exist in prior analyses that required a uniformly bounded variance [CL24, AZ18]. For simplicity, we will first analyze the case $B = 0$ which corresponds to bounded variance and then we will show how to handle the $B \neq 0$ case in the next section.

The two results below will be used for both monotone and strongly monotone cases. As a result, they only use that $F ^ { \mu }$ is µ-strongly monotone which is true in both cases, due to the definition (2.2).

Lemma 4.1. For problem (MI), let Assumption 1 hold and suppose that Assumption 2 holds with $B = 0$ . We apply Algorithm 3 to solve (2.1) with budget $T \geq 4 8 \kappa S$ for number of oracles where $\begin{array} { r } { \kappa = \frac { L _ { F } } { \mu } , S = \lfloor \log _ { 2 } \kappa \rfloor \ge 1 } \end{array}$ and initial point ˚z. We have for all $s = 1 , \ldots , S$ that

$$
\mu _ { s } ^ { 2 } \mathbb { E } \| \mathbf { z } _ { s } - \mathbf { z } _ { s - 1 } ^ { \star } \| ^ { 2 } \leq \bigg ( \frac { 1 } { 2 } \bigg ) ^ { \frac { T s } { 2 4 \kappa S } } \mu ^ { 2 } \| \mathring { \mathbf { z } } - \mathbf { z } _ { 0 } ^ { \star } \| ^ { 2 } + \frac { 4 0 0 S G ^ { 2 } } { T } ,
$$

and the points $\mathbf { z } _ { s } ^ { \star }$ are as defined in (2.3).

Proof. At iteration $s \geq 1$ , we will invoke Theorem 3.3 with

$$
\mathbf { y } _ { 0 } ^ { s }  \mathbf { z } _ { s - 1 } , \quad \mathbf { x } _ { A } ^ { \star }  \mathbf { z } _ { s - 1 } ^ { \star } , \quad A  F ^ { s - 1 } , \quad T  T / S ,\tag{4.1}
$$

where $F ^ { 0 } = F ^ { \mu }$ as defined in (2.2).

With these settings, we can verify that the requirements of Assumption 3 hold. Then, the structural constants in the setting of Theorem 3.3 become

$$
\mu _ { A }  ( 2 ^ { s } - 1 ) \mu , \quad \widehat { L } _ { A } \equiv L _ { A }  L _ { F } + ( 2 ^ { s } - 1 ) \mu , \quad \frac { \widehat { L } _ { A } } { \mu _ { A } } = \frac { L _ { F } + ( 2 ^ { s } - 1 ) \mu } { ( 2 ^ { s } - 1 ) \mu } \leq \frac { 3 \kappa } { 2 } , \quad B _ { A }  0 , \quad G _ { A }  G ,\tag{4.2}
$$

where $\begin{array} { r } { \kappa = \frac { L _ { F } } { \iota \iota } \geq 2 } \end{array}$ , see Theorem 6.5 and $\mu$ is the strong monotonicity constant of $F ^ { \mu }$ . The last two conclusions also follow from Theorem 6.5.

As a result, from Theorem 3.3, we have for $s \geq 1$

$$
\mathbb { E } \| \mathbf { z } _ { s } - \mathbf { z } _ { s - 1 } ^ { \star } \| ^ { 2 } \leq \biggl ( \frac { 1 } { 2 } \biggr ) ^ { \frac { T } { 1 2 S \kappa } } \mathbb { E } \| \mathbf { z } _ { s - 1 } - \mathbf { z } _ { s - 1 } ^ { \star } \| ^ { 2 } + \frac { 7 5 S G ^ { 2 } } { T \mu _ { s - 1 } ^ { 2 } } ,\tag{4.3}
$$

since $\mu _ { s - 1 } \leq \mu _ { A }$

Particularly, for $s = 1$ , we have (recalling $\mu _ { 0 } = \mu )$

$$
\mu _ { 1 } ^ { 2 } \mathbb { E } \| \mathbf { z } _ { 1 } - \mathbf { z } _ { 0 } ^ { \star } \| ^ { 2 } \leq \left( \frac { 1 } { 2 } \right) ^ { \frac { T } { 2 4 S \kappa } } \mu ^ { 2 } \| \mathring { \mathbf { z } } - \mathbf { z } _ { 0 } ^ { \star } \| ^ { 2 } + \frac { 3 0 0 S G ^ { 2 } } { T } ,\tag{4.4}
$$

where we used $4 \left( { \textstyle { \frac { 1 } { 2 } } } \right) ^ { \frac { T } { 1 2 \kappa S } } \leq \left( { \textstyle { \frac { 1 } { 2 } } } \right) ^ { \frac { T } { 2 4 \kappa S } }$ due to $T \geq 4 8 \kappa S$

Using Theorem 6.1 and $\mu _ { s } ^ { 2 } = 4 \mu _ { s - 1 } ^ { 2 }$ , this implies that for $s \geq 2 \mathrm { : }$

$$
\begin{array} { r l r } {  { \mu _ { s } ^ { 2 } \mathbb { E } \| \mathbf { z } _ { s } - \mathbf { z } _ { s - 1 } ^ { \star } \| ^ { 2 } \leq ( \frac { 1 } { 2 } ) ^ { \frac { T } { 1 2 S \kappa } } 4 \mu _ { s - 1 } ^ { 2 } \mathbb { E } \| \mathbf { z } _ { s - 1 } - \mathbf { z } _ { s - 2 } ^ { \star } \| ^ { 2 } + \frac { 3 0 0 S G ^ { 2 } } { T } } } \\ & { } & { \leq ( \frac { 1 } { 2 } ) ^ { \frac { T } { 2 4 S \kappa } } \mu _ { s - 1 } ^ { 2 } \mathbb { E } \| \mathbf { z } _ { s - 1 } - \mathbf { z } _ { s - 2 } ^ { \star } \| ^ { 2 } + \frac { 3 0 0 S G ^ { 2 } } { T } , } \end{array}
$$

where the last step used $T \geq 4 8 S \kappa$ to get $\begin{array} { r } { \frac { T } { 1 2 S \kappa } - 2 \geq \frac { T } { 2 4 S \kappa } } \end{array}$ . By unrolling this inequality and summing the geometric series for the last term, we get for $s \geq 2 \colon$

$$
\begin{array} { r } { \mu _ { s } ^ { 2 } \mathbb { E } \| \mathbf { z } _ { s } - \mathbf { z } _ { s - 1 } ^ { \star } \| ^ { 2 } \leq \left( \displaystyle \frac { 1 } { 2 } \right) ^ { \frac { ( s - 1 ) T } { 2 4 s \kappa } } \mu _ { 1 } ^ { 2 } \mathbb { E } \| \mathbf { z } _ { 1 } - \mathbf { z } _ { 0 } ^ { \star } \| ^ { 2 } + \displaystyle \frac { 3 0 0 S G ^ { 2 } } { T } \sum _ { i = 0 } ^ { s - 2 } \left( \displaystyle \frac { 1 } { 2 } \right) ^ { \frac { i T } { 2 4 \kappa S } } } \\ { \leq \left( \displaystyle \frac { 1 } { 2 } \right) ^ { \frac { s T } { 2 4 S \kappa } } \mu ^ { 2 } \| \hat { \mathbf { z } } - \mathbf { z } _ { 0 } ^ { \star } \| ^ { 2 } + \displaystyle \frac { 3 0 0 S G ^ { 2 } } { T } \sum _ { i = 0 } ^ { s - 1 } \left( \displaystyle \frac { 1 } { 2 } \right) ^ { \frac { i T } { 2 4 \kappa S } } , } \end{array}
$$

where the second inequality used (4.4). Calculating the sum of geometric series and using $T \geq 4 8 \kappa S$ gives the result. ■

By using Theorem $6 . 4 \cdot$ we can now convert this result to a guarantee on the gradient mapping norm for the operator $H ^ { \mu }$ given in (2.1). In view of its definition, this will directly give us the required guarantee for the strongly monotone setting. In the monotone setting, we will select $\mu$ accordingly to get a guarantee for the original problem in (MI).

Lemma 4.2. For problem (MI), let Assumption 1 hold and suppose that Assumption 2 holds with $B = 0$ We apply Algorithm 3 to solve (2.1) with $\begin{array} { r } { \eta = \frac { 1 } { 3 L _ { F } } } \end{array}$ , oracle budget $T \geq 4 8 S L _ { F } / \mu$ where $\begin{array} { r } { S = \left\lfloor \log _ { 2 } \frac { L _ { F } } { \mu } \right\rfloor \ge 1 } \end{array}$ for the number of oracles and initial point ˚z. Then, we have that

$$
\mathbb { E } \| \mathcal { G } _ { \eta , H ^ { \mu } } ( \mathbf { z } _ { S } ) \| = O \left( \left( \frac { 1 } { 2 } \right) ^ { \frac { T } { 4 8 S L _ { F } / \mu } } \mu \| \mathring { \mathbf { z } } - \mathbf { z } _ { 0 } ^ { \star } \| + \frac { S ^ { 3 / 2 } G } { \sqrt { T } } \right) ,
$$

where $O$ only suppresses the absolute constants.

Proof. In view of the result of Theorem 4.1, we invoke Theorem 6.4 with $C \gets 4 0 0$ and ${ \widehat { G } }  G$ to obtain the claim. ■

## 4.1 Strongly Monotone Case

In the strongly monotone case, we will use the first case in the definition of $F ^ { \mu }$ in (2.2). This reduces $H ^ { \mu }$ to the original problem and we have the following theorem as the direct corollary of Theorem 4.2.

Theorem 4.3. For problem (MI), let Assumption 1 hold and suppose that Assumption 2 holds with $B = 0$ Additionally assume that F is µ<sub>F</sub> strongly monotone for $\mu _ { F } > 0$ . We apply Algorithm 3 to solve (2.1) with $\begin{array} { r } { \eta = \frac { 1 } { 3 L _ { F } } , S = \lfloor \log _ { 2 } ( L _ { F } / \mu _ { F } ) \rfloor \ge 1 } \end{array}$ and $T \geq 4 8 S L _ { F } / \mu _ { F }$ , and initial point ˚z. Then, we have that

$$
\mathbb { E } \| \mathcal { G } _ { \eta , H } ( \mathbf { z } _ { S } ) \| = O \left( \left( \frac { 1 } { 2 } \right) ^ { \frac { T } { 4 8 S L _ { F } / \mu _ { F } } } \mu _ { F } D _ { \star } + \frac { S ^ { 3 / 2 } G } { \sqrt { T } } \right) ,
$$

where O only suppresses the absolute constants.

Proof. Since F is strongly monotone, in view of (2.2), we have $F ^ { \mu } = F , \mu = \mu _ { F }$ , and $\mathbf { z } _ { 0 } ^ { \star } = \mathbf { x } ^ { \star }$ . Then, we invoke Theorem 4.2 with these settings and use the definition of $D _ { \star }$ from (1.8) to derive the bound. ■

The next corollary then immediately follows by finding the value of $T$ to make the right-hand side of Theorem 4.3 less than ε.

Corollary 4.4. Under the same setup as Theorem $4 . 9 ,$ , we have

$$
\mathbb { E } \| \mathcal { G } _ { \eta , H } ( \mathbf { z } _ { S } ) \| \le \varepsilon ,\tag{4.5}
$$

where the number of stochastic first-order oracles T is upper bounded $b y$

$$
O \left( \frac { L _ { F } } { \mu _ { F } } \log \left( \frac { L _ { F } } { \mu _ { F } } \right) \log \left( \frac { \mu _ { F } D _ { \star } } { \varepsilon } + e \right) + \frac { G ^ { 2 } \log ^ { 3 } ( L _ { F } / \mu _ { F } ) } { \varepsilon ^ { 2 } } \right) .
$$

Our algorithmic parameters do not depend on $\| \mathbf { z } _ { 0 } - \mathbf { x } ^ { \star } \|$ or a global variance upper bound $\sigma ^ { 2 }$ , which were required for running the algorithm in [CL24, Theorem 4.1]. The parameters in our algorithm only depend on the oracle budget $T , L _ { F }$ and $\mu _ { F } ,$ similar to the case of [AZ18] in the minimization case and the bound fo the gradient mapping in Theorem 4.3 follows. Converting this to a complexity bound gives Corollary 4.4.

## 4.2 Monotone Case

In view of the definitions in (2.1) and (2.2), we are in the second case of (2.2). Let us recall that Theorem 4.2 gives a bound for solving the perturbed problem, controlled by $\mu .$ By controlling $\mu$ and the distance between gradient mapping at perturbed problem and the original problem, we will derive a guarantee for the gradient mapping for solving the original problem in (MI).

Theorem 4.5 (Monotone F). For problem (MI), let Assumption 1 hold and suppose that Assumption 2 holds with $B = 0$ . We apply Algorithm 3 to solve (2.1) with any $\begin{array} { r } { L _ { F } / 2 \geq \mu > 0 , \eta = \frac { 1 } { 3 L _ { F } } , S = \lfloor \log _ { 2 } ( L _ { F } / \mu ) \rfloor \geq 1 } \end{array}$ and $T \geq 4 8 S L _ { F } / \mu$ , and initial point ˚z. Then, we have that

$$
\mathbb { E } \| \mathcal { G } _ { \eta , H } ( \mathbf { z } _ { S } ) \| = O \left( \mu \| \mathbf { x } ^ { \star } - \hat { \mathbf { z } } \| + \left( \frac { 1 } { 2 } \right) ^ { \frac { T } { 4 8 S L _ { F } / \mu } } \mu \| \hat { \mathbf { z } } - \mathbf { x } ^ { \star } \| + \frac { S ^ { 3 / 2 } G } { \sqrt { T } } \right) .
$$

Proof. Applying Theorem 6.3 yields the inequality

$$
\| \mathcal { G } _ { \eta , H } ( \mathbf { z } _ { S } ) \| \leq 2 . 5 \sum _ { j = 1 } ^ { S } \mu _ { j } \| \mathbf { z } _ { j } - \mathbf { z } _ { j - 1 } ^ { \star } \| + 9 L _ { F } \| \mathbf { z } _ { S } - \mathbf { z } _ { S - 1 } ^ { \star } \| + \mu \| \mathring { \mathbf { z } } - \mathbf { x } ^ { \star } \| .\tag{4.6}
$$

Since Theorem 4.1 holds in this case, on (4.6), we use (6.11) to bound the first term and (6.12) the second term on the right-hand side and use (6.8) to obtain the claimed bound.

Corollary 4.6. Under the same setup of Theorem $4 . 5 ,$ let $T \geq 2 2 0 0$ and $\begin{array} { r } { \mu = \frac { 9 6 L _ { F } \log _ { 2 } T } { T } } \end{array}$ . We apply Algorithm 3 to solve (2.1) with $\begin{array} { r } { \eta = \frac { 1 } { 3 L _ { F } } , S = \lfloor \log _ { 2 } ( L _ { F } / \mu ) \rfloor } \end{array}$ , and initial point ˚z. Then, we have that

$$
\mathbb { E } \| \mathcal { G } _ { \eta , H } ( \mathbf { z } _ { S } ) \| = O \left( \frac { L _ { F } D _ { \star } \log _ { 2 } T } { T } + \frac { G ( \log _ { 2 } T ) ^ { 3 / 2 } } { \sqrt { T } } \right) .\tag{4.7}
$$

Consequently, to obtain

$$
\mathbb { E } \| \mathcal { G } _ { \eta , H } ( \mathbf { z } _ { S } ) \| \le \varepsilon ,\tag{4.8}
$$

the required number of stochastic first-order oracles T is upper bounded by

$$
O \left( 1 + \frac { L _ { F } D _ { \star } } { \varepsilon } \ln \left( \frac { L _ { F } D _ { \star } } { \varepsilon } + 1 \right) + \frac { G ^ { 2 } } { \varepsilon ^ { 2 } } \ln ^ { 3 } \left( \frac { G } { \varepsilon } + 1 \right) \right) ,
$$

where we only suppress the absolute constants.

Proof. In this case, we start from the result of Theorem 6.3, after plugging in from (6.11) and (6.12) to derive

$$
\mathbb { E } \| \mathcal { G } _ { \eta , H } ( \mathbf { z } _ { S } ) \| \leq \mu D _ { \star } + 5 \left( \frac { 1 } { 2 } \right) ^ { \frac { T } { 4 8 \kappa \mathcal { S } } } \mu D _ { \star } + \frac { 5 0 S ^ { 3 / 2 } G } { \sqrt { T } } + 1 8 \bigg ( \left( \frac { 1 } { 2 } \right) ^ { \frac { T } { 4 8 \kappa } } \mu D _ { \star } + \frac { 2 0 \sqrt { S } G } { \sqrt { T } } \bigg ) ,\tag{4.9}
$$

where we also used Theorem 6.1 which gives $\left\| \bar { \mathbf z } - \mathbf z _ { 0 } ^ { \star } \right\| \leq \left\| \bar { \mathbf z } - \mathbf x ^ { \star } \right\| = D ,$ <sub>⋆</sub>. Plugging in the definition of $\mu$ gives (4.7). From this bound, it should be clear that this choice of $\mu$ will give a complexity result scaling as $\bar { \varepsilon } ^ { - \bar { 2 } }$ up to logarithmic terms. We now flesh out the explicit dependencies.

First, we verify that the requirement on T in Theorem 4.5 is satisfied. That is, we show that $T \geq 4 8 \kappa S$ By the definitions of $\begin{array} { r } { \kappa = \frac { L _ { F } } { \mu } = \frac { T } { 9 6 \log _ { 2 } T } \leq T } \end{array}$ and $\begin{array} { r } { S = \left| \log _ { 2 } \frac { L _ { F } } { \mu } \right| \leq \log _ { 2 } \frac { L _ { F } } { \mu } \leq \log _ { 2 } T } \end{array}$ , we have

$$
4 8 \kappa S = \frac { T } { 2 \log _ { 2 } T } S \leq T .
$$

Moreover, by the definition of $\kappa ,$ we also have that $\kappa \geq 2$ since $T \geq 2 2 0 0$

Because of this requirement, for making the first, second and fourth terms in (4.9) less than $\varepsilon ,$ we need $\mu D _ { \star } \leq \varepsilon / 9$ . This is true when

$$
\frac { 9 6 L _ { F } D _ { \star } \log _ { 2 } T } { T } \leq \frac { \varepsilon } { 9 } ,
$$

which is true for $\begin{array} { r } { T = \Theta \left( \frac { L _ { F } D _ { \star } } { \varepsilon } \ln \left( \frac { L _ { F } D _ { \star } } { \varepsilon } + 1 \right) \right) } \end{array}$  because of Theorem $6 . 7 ( i i )$ with $a = 9 6 L _ { F } D _ { \star } , b = 1 , c = 1$ $\varepsilon ^ { \prime } = \varepsilon / 9$

We finally calculate the order of $T$ to make the third and fifth terms on the right-hand side of (4.9) less than ε. Since the third term dominates the fifth up to absolute constants, we calculate $T$ to make

$$
{ \frac { 5 0 S ^ { 3 / 2 } G } { \sqrt { T } } } \leq \varepsilon ,
$$

where $\begin{array} { r } { S = \left\lfloor \log _ { 2 } \frac { L _ { F } } { \mu } \right\rfloor \le \log _ { 2 } \frac { L _ { F } } { \mu } \le \log _ { 2 } T } \end{array}$ . As a result, a suficient condition is

$$
\frac { 2 5 0 0 G ^ { 2 } ( \log _ { 2 } T ) ^ { 3 } } { T } \leq \varepsilon ^ { 2 } ,
$$

which is true for $\begin{array} { r } { T = \Theta \left( \frac { G ^ { 2 } } { \varepsilon ^ { 2 } } \ln ^ { 3 } \left( \frac { G } { \varepsilon } + 1 \right) \right) } \end{array}$ because of Theorem 6.7(ii) with $a = 2 5 0 0 G ^ { 2 } , b = 1 , c = 3 . \ \varepsilon ^ { \prime } = \varepsilon ^ { 2 }$ Combining the two bounds for T gives the assertion. ■

## 5 Complexity Analysis without Bounded Variance

For brevity, we will only analyze the monotone case, the strongly monotone case can be analyzed similarly by replacing the $\mu$ value by the strong monotonicity of the operator.

We now derive the result corresponding to Theorem 4.1 when the bounded variance assumption is lifted. Even for unconstrained problems, our result extends [CL24, Theorem 4.1] in that we do not require a globally upper bounded variance, but only Assumption 2.

Lemma 5.1. For problem (MI), let Assumption 1 and Assumption 2 hold. We apply Algorithm 3 to solve (2.1) with budget $T \geq 1 4 1 7 5$ for number of oracles, $\begin{array} { r } { \mu = \frac { 3 2 \widehat { L } _ { F } } { \sqrt { T } } \log _ { 2 } ^ { 3 / 2 } \frac { \sqrt { T } } { 4 8 } } \end{array}$ and initial point ˚z. We have for all $s = 1 , \ldots , S$ that

$$
\mu _ { s } ^ { 2 } \mathbb { E } \| \mathbf { z } _ { s } - \mathbf { z } _ { s - 1 } ^ { \star } \| ^ { 2 } \leq \left( \frac { 1 } { 2 } \right) ^ { \frac { T s } { 2 4 \kappa S } } \mu ^ { 2 } \| \mathring { \mathbf { z } } - \mathbf { z } _ { 0 } ^ { \star } \| ^ { 2 } + \frac { 2 0 4 8 S ^ { 3 } \widehat { G } ^ { 2 } } { T } ,
$$

where $\begin{array} { r } { \widehat { G } ^ { 2 } = B ^ { 2 } \| \mathring { \mathbf { z } } - \mathbf { z } _ { 0 } ^ { \star } \| ^ { 2 } + G ^ { 2 } , \kappa = \frac { \widehat { L } _ { F } } { \mu } = \frac { L _ { F } + B } { \mu } \ a n d S = \lfloor \log _ { 2 } \kappa \rfloor } \end{array}$

Proof. At iteration $s \geq 1$ , we will invoke Theorem 3.3 with

$$
\mathbf { y } _ { 0 }  \mathbf { z } _ { s - 1 } , \quad \mathbf { x } _ { A } ^ { \star }  \mathbf { z } _ { s - 1 } ^ { \star } , \quad A  F ^ { s - 1 } , \quad T  T / S ,\tag{5.1}
$$

With these settings, we can verify that the requirements of Assumption 3 hold. Then, we will get the structural constants in the setting of Theorem 3.3 becoming

$$
\mu _ { A }  ( 2 ^ { s } - 1 ) \mu , ~ L _ { A }  L _ { s - 1 } = L _ { F } + ( 2 ^ { s } - 1 ) \mu , ~ B _ { A }  B , ~ G _ { A }  G ,\tag{5.2}
$$

where we use Theorem 6.5. This also tells us that $\begin{array} { r } { \frac { \widehat { L } _ { A } } { \mu _ { A } } = \frac { L _ { A } + B _ { A } } { \mu _ { A } } \leq \frac { L _ { F } + ( 2 ^ { s } - 1 ) \mu + B } { ( 2 ^ { s } - 1 ) \mu } \leq 1 + \frac { L _ { F } + B } { \mu } \leq \frac { 3 \kappa } { 2 } } \end{array}$ where $\begin{array} { r } { \kappa = \frac { L _ { F } + B } { u } \geq 2 } \end{array}$ by Lemma 6.7(iii).

As a result, from Theorem 3.3, we have for $s \geq 1$ 1:

$$
\mathbb { E } \| \mathbf { z } _ { s } - \mathbf { z } _ { s - 1 } ^ { \star } \| ^ { 2 } \leq \bigg ( \frac { 1 } { 2 } \bigg ) ^ { \frac { T } { 1 2 S \kappa } } \mathbb { E } \| \mathbf { z } _ { s - 1 } - \mathbf { z } _ { s - 1 } ^ { \star } \| ^ { 2 } + \frac { 1 9 2 S \mathbb { E } ( B ^ { 2 } \| \hat { \mathbf { z } } - \mathbf { z } _ { s - 1 } ^ { \star } \| ^ { 2 } + G ^ { 2 } ) } { T \mu _ { s - 1 } ^ { 2 } } ,\tag{5.3}
$$

since $\mu _ { s - 1 } \leq \mu _ { A }$ . Here, we also majorized the constant in the last term for simplicity.

We next need to show that the term involving $\mathbb { E } \| \mathring { \mathbf { z } } - \mathbf { z } _ { s - 1 } ^ { \star } \| ^ { 2 }$ is not too large to deteriorate the bound. In particular, we use triangle inequality and Theorem 6.1 to get

$$
\| \bar { \mathbf { z } } - \mathbf { z } _ { s - 1 } ^ { \star } \| \leq \| \bar { \mathbf { z } } - \mathbf { z } _ { 0 } ^ { \star } \| + \sum _ { j = 1 } ^ { s - 1 } \| \mathbf { z } _ { j } ^ { \star } - \mathbf { z } _ { j - 1 } ^ { \star } \| \leq \| \bar { \mathbf { z } } - \mathbf { z } _ { 0 } ^ { \star } \| + \sum _ { j = 1 } ^ { s - 1 } \| \mathbf { z } _ { j } - \mathbf { z } _ { j - 1 } ^ { \star } \| .
$$

After taking the square of this estimate, we consequently have

$$
\| \mathring { \mathbf { z } } - \mathbf { z } _ { s - 1 } ^ { \star } \| ^ { 2 } \leq S \| \mathring { \mathbf { z } } - \mathbf { z } _ { 0 } ^ { \star } \| ^ { 2 } + S \sum _ { j = 1 } ^ { s - 1 } \| \mathbf { z } _ { j } - \mathbf { z } _ { j - 1 } ^ { \star } \| ^ { 2 } ,\tag{5.4}
$$

since $s \leq S .$

Using this bound and Theorem 6.1 on (5.3) yields for $s \geq 2$

$$
\mathbb { E } \Vert \mathbf { z } _ { s } - \mathbf { z } _ { s - 1 } ^ { \star } \Vert ^ { 2 } \leq \left( \frac { 1 } { 2 } \right) ^ { \frac { T } { 1 2 S \kappa } } \mathbb { E } \Vert \mathbf { z } _ { s - 1 } - \mathbf { z } _ { s - 2 } ^ { \star } \Vert ^ { 2 } + \frac { 1 9 2 S ^ { 2 } B ^ { 2 } ( \Vert \tilde { \mathbf { z } } - \mathbf { z } _ { 0 } ^ { \star } \Vert ^ { 2 } + \sum _ { j = 1 } ^ { s - 1 } \mathbb { E } \Vert \mathbf { z } _ { j } - \mathbf { z } _ { j - 1 } ^ { \star } \Vert ^ { 2 } ) } { T \mu _ { s - 1 } ^ { 2 } } + \frac { 1 9 2 S G ^ { 2 } } { T \mu _ { s - 1 } ^ { 2 } } .\tag{5.5}
$$

Due to Theorem $6 . 7 ( i i i )$ , the choice of $T$ and $\mu$ implies that

$$
T \geq 4 8 \kappa S , \qquad \mu ^ { 2 } \geq \frac { 1 0 2 4 S ^ { 3 } \widehat { L } _ { F } ^ { 2 } } { T } .
$$

As a result, we notice for the coeficients in (5.5) that

$$
\left( \frac { 1 } { 2 } \right) ^ { \frac { T } { 1 2 S _ { \mathrm { r } } } } \leq \frac { 1 } { 4 } , \quad \frac { 1 9 2 S ^ { 2 } B ^ { 2 } } { T \mu _ { s - 1 } ^ { 2 } } \leq \frac { 2 5 6 S ^ { 2 } B ^ { 2 } } { T \mu ^ { 2 } } \leq \frac { 1 } { 4 S } , \quad \frac { 1 9 2 S G ^ { 2 } } { T \mu _ { s - 1 } ^ { 2 } } \leq \frac { G ^ { 2 } } { 4 ( L _ { F } + B ) ^ { 2 } S ^ { 2 } } \leq \frac { G ^ { 2 } } { 4 ( L _ { F } + B ) ^ { 2 } } ,\tag{5.6}
$$

where we also used $\mu _ { s - 1 } \geq \mu$ and $S \geq 1$

We will next use induction. With these estimates, the inequality (5.5) becomes for $s \geq 2$

$$
\mathbb { E } \| \mathbf { z } _ { s } - \mathbf { z } _ { s - 1 } ^ { \star } \| ^ { 2 } \leq \frac { 1 } { 4 } \mathbb { E } \| \mathbf { z } _ { s - 1 } - \mathbf { z } _ { s - 2 } ^ { \star } \| ^ { 2 } + \frac { \| \tilde { \mathbf { z } } - \mathbf { z } _ { 0 } ^ { \star } \| ^ { 2 } + \sum _ { j = 1 } ^ { s - 1 } \mathbb { E } \| \mathbf { z } _ { j } - \mathbf { z } _ { j - 1 } ^ { \star } \| ^ { 2 } } { 4 S } + \frac { G ^ { 2 } } { 4 ( L _ { F } + B ) ^ { 2 } } .\tag{5.7}
$$

For $s = 1$ , from (5.3), with $\mathbf { z } _ { 0 } = \mathring { \mathbf { z } }$ and $\mathbf { z } _ { s - 1 } ^ { \star } = \mathbf { z } _ { 0 } ^ { \star }$ , we have

$$
\mathbb { E } \| \mathbf { z } _ { 1 } - \mathbf { z } _ { 0 } ^ { \star } \| ^ { 2 } \leq \bigg ( \frac { 1 } { 2 } \bigg ) ^ { \frac { T } { 1 2 S \kappa } } \| \mathring { \mathbf { z } } - \mathbf { z } _ { 0 } ^ { \star } \| ^ { 2 } + \frac { 1 9 2 S ( B ^ { 2 } \| \mathring { \mathbf { z } } - \mathbf { z } _ { 0 } ^ { \star } \| ^ { 2 } + G ^ { 2 } ) } { T \mu _ { 0 } ^ { 2 } } .
$$

Applying (5.4) and (5.6) gives

$$
\mathbb { E } \| \mathbf { z } _ { 1 } - \mathbf { z } _ { 0 } ^ { \star } \| ^ { 2 } \leq \frac { 1 } { 4 } \| \hat { \mathbf { z } } - \mathbf { z } _ { 0 } ^ { \star } \| ^ { 2 } + \frac { \| \hat { \mathbf { z } } - \mathbf { z } _ { 0 } ^ { \star } \| ^ { 2 } } { 4 S } + \frac { G ^ { 2 } } { 4 ( L _ { F } + B ) ^ { 2 } } .
$$

We will prove $\begin{array} { r } { \mathbb { E } \| \mathbf { z } _ { j } - \mathbf { z } _ { j - 1 } ^ { \star } \| ^ { 2 } \leq \| \mathring { \mathbf { z } } - \mathbf { z } _ { 0 } ^ { \star } \| ^ { 2 } + \frac { G ^ { 2 } } { ( L _ { F } + B ) ^ { 2 } } } \end{array}$ for $j \geq 1$ . By the last display equation, the claim holds at $j = 1$ . Assume that it holds for $j \le s - 1$ with $s \geq 2$ . Then, (5.7) gives

$$
\begin{array} { r l r } {  { \mathbb { E } \| \mathbf { z } _ { s } - \mathbf { z } _ { s - 1 } ^ { \star } \| ^ { 2 } \leq \displaystyle \frac { 1 } { 2 } ( \| \hat { \mathbf { z } } - \mathbf { z } _ { 0 } ^ { \star } \| ^ { 2 } + \frac { G ^ { 2 } } { ( L _ { F } + B ) ^ { 2 } } ) + \frac { G ^ { 2 } } { 4 ( L _ { F } + B ) ^ { 2 } } } } \\ & { } & { \leq \| \hat { \mathbf { z } } - \mathbf { z } _ { 0 } ^ { \star } \| ^ { 2 } + \frac { G ^ { 2 } } { ( L _ { F } + B ) ^ { 2 } } . \quad \quad } \end{array}\tag{5.8}
$$

This completes the induction to derive the global upper bound on $\mathbb { E } \| { \bf z } _ { s } - { \bf z } _ { s - 1 } ^ { \star } \| ^ { 2 }$

Then, using (5.4) and taking expectation, we have

$$
\begin{array} { r l } { \displaystyle \mathbb { E } \| \hat { \mathbf { z } } - \mathbf { z } _ { s - 1 } ^ { \star } \| ^ { 2 } \leq S \| \hat { \mathbf { z } } - \mathbf { z } _ { 0 } ^ { \star } \| ^ { 2 } + S \displaystyle \sum _ { j = 1 } ^ { s - 1 } \bigg ( \| \hat { \mathbf { z } } - \mathbf { z } _ { 0 } ^ { \star } \| ^ { 2 } + \frac { G ^ { 2 } } { ( L _ { F } + B ) ^ { 2 } } \bigg ) } & { } \\ { \leq 2 S ^ { 2 } \| \hat { \mathbf { z } } - \mathbf { z } _ { 0 } ^ { \star } \| ^ { 2 } + \displaystyle \frac { S ^ { 2 } G ^ { 2 } } { ( L _ { F } + B ) ^ { 2 } } . } & { } \end{array}
$$

We now plug this bound to (5.3) which gives us the bound

$$
\begin{array} { r l } & { \displaystyle \mathbb { E } \| \mathbf { z } _ { s } - \mathbf { z } _ { s - 1 } ^ { \star } \| ^ { 2 } \leq \bigg ( \frac { 1 } { 2 } \bigg ) ^ { \frac { T } { 1 2 S \kappa } } \mathbb { E } \| \mathbf { z } _ { s - 1 } - \mathbf { z } _ { s - 1 } ^ { \star } \| ^ { 2 } + \frac { 1 9 2 S ( B ^ { 2 } ( 2 S ^ { 2 } \| \hat { \mathbf { z } } - \mathbf { z } _ { 0 } ^ { \star } \| ^ { 2 } + \frac { S ^ { 2 } G ^ { 2 } } { ( L _ { F } + B ) ^ { 2 } } ) + G ^ { 2 } ) } { T \mu _ { s - 1 } ^ { 2 } } } \\ & { \qquad \leq \bigg ( \frac { 1 } { 2 } \bigg ) ^ { \frac { T } { 1 2 S \kappa } } \mathbb { E } \| \mathbf { z } _ { s - 1 } - \mathbf { z } _ { s - 1 } ^ { \star } \| ^ { 2 } + \frac { 3 8 4 S ^ { 3 } ( B ^ { 2 } \| \hat { \mathbf { z } } - \mathbf { z } _ { 0 } ^ { \star } \| ^ { 2 } + G ^ { 2 } ) } { T \mu _ { s - 1 } ^ { 2 } } } \end{array}\tag{5.9}
$$

(5.10)

We follow the exact same steps that followed (4.3) (where the diference is having $3 8 4 S ^ { 3 }$ instead of 75S and $B ^ { 2 } \| \mathring { \mathbf z } - \mathbf z _ { 0 } ^ { \star } \| ^ { 2 } + G ^ { 2 }$ instead of $G ^ { 2 }$ on the last term) to get the assertion. ■

Let us remark that the induction argument in this proof to bound $\mathbb { E } \| \mathring { \mathbf { z } } - \mathbf { z } _ { s - 1 } ^ { \star } \| ^ { 2 }$ can be strengthened with a more involved analysis to improve the order of the $S$ term in the final bound. However we chose the current argument for simplicity, since this order only afects the order of the logarithmic terms in the final bound in the sequel.

The next result corresponds to Theorem 4.2 and Theorem 4.5 without the bounded variance assumption.

Theorem 5.2. For problem (MI), let Assumptions 1 and 2 hold. We apply Algorithm 3 to solve (2.1) with

$$
\mu = \frac { 3 2 \widehat { L } _ { F } } { \sqrt { T } } \log _ { 2 } ^ { 3 / 2 } \left( \frac { \sqrt { T } } { 4 8 } \right) , \quad \eta = \frac { 1 } { 3 \widehat { L } _ { F } } , \quad S = \big \lfloor \log _ { 2 } ( \widehat { L } _ { F } / \mu ) \big \rfloor , \quad T \geq 1 4 1 7 5 ,
$$

and initial point ˚z. Then, we have that

$$
\mathbb { E } \| \mathcal { G } _ { \eta , H } ( \mathbf { z } _ { S } ) \| = O \left( \mu D _ { \star } + \left( \frac { 1 } { 2 } \right) ^ { \frac { T } { 4 8 S \tilde { L } _ { F } / \mu } } \mu D _ { \star } + \frac { S ^ { 5 / 2 } \left( B D _ { \star } + G \right) } { \sqrt { T } } \right) .
$$

Remark 5.3. By the value of $\textstyle \mu = { \widetilde { O } } \left( { \frac { 1 } { \sqrt { T } } } \right)$ and $T \geq 4 8 S \widehat { L } _ { F } / \mu$ as shown in Lemma 6.7, one can estimate that the complexity will be of the order $\varepsilon ^ { - 2 } ~ \mathrm { u p }$ to logarithmic terms. We follow by a corollary for the order of the terms and then we give the precise constants.

Proof. The result of Theorem 5.1 shows that the hypothesis of Theorem 6.4 is satisfied with

$$
C = 2 0 4 8 S ^ { 2 } \quad \mathrm { a n d } \quad \widehat { G } ^ { 2 } = B ^ { 2 } \| \mathring { \mathbf { z } } - \mathbf { z } _ { 0 } ^ { \star } \| ^ { 2 } + G ^ { 2 } .\tag{5.11}
$$

We also know that applying Theorem 6.3 yields the inequality

$$
\| \mathcal { G } _ { \eta , H } ( \mathbf { z } _ { S } ) \| \leq 2 . 5 \sum _ { j = 1 } ^ { S } \mu _ { j } \| \mathbf { z } _ { j } - \mathbf { z } _ { j - 1 } ^ { \star } \| + 9 \widehat { L } _ { F } \| \mathbf { z } _ { S } - \mathbf { z } _ { S - 1 } ^ { \star } \| + \mu \| \mathring { \mathbf { z } } - \mathbf { x } ^ { \star } \| .\tag{5.12}
$$

Using (6.11) and (6.12) and Theorem 6.1 for $\left\| \bar { \mathbf z } - \mathbf z _ { 0 } ^ { \star } \right\| \leq \left\| \bar { \mathbf z } - \mathbf x ^ { \star } \right\| = D ,$ to estimate the right-hand side of (5.12) gives the claimed bound. ■

Corollary 5.4. Under the same setup of Theorem 5.2, with $\begin{array} { r } { \mu = \frac { 3 2 \widehat { L } _ { F } } { \sqrt { T } } \log _ { 2 } ^ { 3 / 2 } \left( \frac { \sqrt { T } } { 4 8 } \right) , T \geq 1 4 1 7 5 } \end{array}$ , we have that

$$
\mathbb { E } \| \mathcal { G } _ { \eta , H } ( \mathbf { z } _ { S } ) \| = O \left( \frac { \widehat { L } _ { F } D _ { \star } ( \ln \sqrt { T } ) ^ { 3 / 2 } } { \sqrt { T } } + \frac { ( \ln \sqrt { T } ) ^ { 5 / 2 } ( B D _ { \star } + G ) } { \sqrt { T } } \right) .\tag{5.13}
$$

Consequently, to obtain

$$
\mathbb { E } \| \mathcal { G } _ { \eta , H } ( \mathbf { z } _ { S } ) \| \le \varepsilon ,\tag{5.14}
$$

the number of required stochastic first-order oracles $T$ is upper bounded by

$$
O \left( 1 + \frac { ( L _ { F } ^ { 2 } + B ^ { 2 } ) D _ { \star } ^ { 2 } + G ^ { 2 } } { \varepsilon ^ { 2 } } \ln ^ { 5 } \left( \frac { D _ { \star } ( L _ { F } + B ) + G } { \varepsilon } + 1 \right) \right) ,
$$

where we only suppress the absolute constants.

Proof. We next write down the precise bound for the gradient mapping norm, unpacking the bound in Theorem 5.2.

$$
\mathbb { E } \Vert \mathcal { G } _ { \eta , H } ( \mathbf { z } _ { S } ) \Vert \leq \mu D _ { \star } + 5 \left( \frac { 1 } { 2 } \right) ^ { \frac { T } { 4 8 \kappa 5 } } \mu D _ { \star } + \frac { 1 1 4 S ^ { 5 / 2 } \widehat { G } } { \sqrt { T } } + 1 8 \left( \left( \frac { 1 } { 2 } \right) ^ { \frac { T } { 4 8 \kappa } } \mu D _ { \star } + \frac { 4 6 S ^ { 3 / 2 } \widehat { G } } { \sqrt { T } } \right) ,\tag{5.15}
$$

where $\widehat { G }$ is as defined in (5.11), that is, $\widehat { G } ^ { 2 } = B ^ { 2 } \| \mathring { \mathbf z } - \mathbf z _ { 0 } ^ { \star } \| ^ { 2 } + G ^ { 2 }$ . Plugging in the definition of $\mu$ gives (5.13).

We first estimate the first term on the right-hand side. In particular, we have

$$
\mu D _ { \star } = { \frac { 3 2 \widehat { L } _ { F } D _ { \star } } { \sqrt { T } } } \left( \log _ { 2 } { \frac { \sqrt { T } } { 4 8 } } \right) ^ { 3 / 2 } .
$$

Then, by using Theorem 6.7(i) with $\begin{array} { r } { a = 3 2 \widehat { L } _ { F } D _ { \star } , b = \frac { 1 } { 4 8 } } \end{array}$ , and $\begin{array} { r } { c = \frac { 3 } { 2 } } \end{array}$ , we have that $\mu D _ { \star } \leq \varepsilon$ whenever

$$
T = \Theta \left( \frac { \widehat { L } _ { F } ^ { 2 } D _ { \star } ^ { 2 } } { \varepsilon ^ { 2 } } \ln ^ { 3 } \left( \frac { \widehat { L } _ { F } D _ { \star } } { \varepsilon } + 1 \right) \right) .\tag{5.16}
$$

As long as we have $\mu D _ { \star } \leq \varepsilon ,$ we also have for the second-term on the right-hand side of (5.15)

$$
5 \left( { \frac { 1 } { 2 } } \right) ^ { \frac { T } { 4 8 \kappa S } } \mu \| \mathring { \mathbf { z } } - \mathbf { z } _ { 0 } ^ { \star } \| \leq 2 . 5 \varepsilon ,
$$

because $T \geq 4 8 \kappa S$ , see Lemma $6 . 7 ( \mathrm { i i i } )$

Similarly, the fourth term on the right-hand side of (5.15) is upper bounded by these terms up to absolute constants, as a result, it will be also of the order ε scaled by an absolute constant.

As shown in Lemma 6.7(iii), we have

$$
S ^ { 5 / 2 } \leq \left( \log _ { 2 } { \frac { \sqrt { T } } { 4 8 } } \right) ^ { 5 / 2 } .
$$

Then, we estimate the third term on the right-hand side of (5.15) as

$$
\frac { 1 1 4 S ^ { 5 / 2 } \widehat { G } } { \sqrt { T } } \leq \varepsilon , \mathrm { ~ w h e n ~ } T = \Theta \left( \frac { B ^ { 2 } D _ { \star } ^ { 2 } + G ^ { 2 } } { \varepsilon ^ { 2 } } \ln ^ { 5 } \left( \frac { B D _ { \star } + G } { \varepsilon } + 1 \right) \right) ,\tag{5.17}
$$

where the calculation of T uses Theorem $6 . 7 ( \mathrm { i } )$ with $a = 1 1 4 \widehat { G } , b = \textstyle { \frac { 1 } { 4 8 } }$ , and $c = 5 / 2$ . We also used $\widehat { G } ^ { 2 } \leq B ^ { 2 } D _ { \star } ^ { 2 } + G ^ { 2 }$ and $\widehat { G } \leq B D _ { \star } + G$ , due to Lemma 6.1. The last term on the right-hand side of (5.15) is dominated up to constants by the third term so the given $T$ is suficient (up to constants) to make the last term on the right-hand side of (5.15) less than ε.

Let us recall that $T \geq 4 8 \kappa S$ was proved in Lemma 6.7(iii). Combining the estimates for T in (5.16) and (5.17) gives the assertion. ■

## 6 Auxiliary Lemmas

In this section, we will prove auxiliary results that were used in the main proofs. The first result connects the solutions of subproblems for diferent regularization levels.

Given $F ^ { 0 } = F ^ { \mu }$ , let us recall that at iteration $s \geq 1$ , we estimate $\mathbf { z } _ { s - 1 } ^ { \star }$ such that $0 \in ( F ^ { s - 1 } + \partial r ) ( \mathbf { z } _ { s - 1 } ^ { \star } )$ where

$$
F ^ { s - 1 } ( \mathbf { z } ) = F ^ { \mu } ( \mathbf { z } ) + \sum _ { i = 1 } ^ { s - 1 } \mu _ { i } ( \mathbf { z } - \mathbf { z } _ { i } ) , { \mathrm { ~ f o r ~ } } s = 1 , 2 , \ldots , S\tag{6.1}
$$

with $\mathbf { z } _ { 0 } ^ { \star }$ is as defined in (2.1).

Lemma 6.1. For $s = 1 , \ldots , S , \ i f$ we are given that $F ^ { 0 } = F ^ { \mu }$ (see (2.2)), $\mathbf { z } _ { s } ^ { \star } = ( F ^ { s } + \partial r ) ^ { - 1 } ( 0 )$ and $\mathbf { z } _ { s - 1 } ^ { \star } = ( F ^ { s - 1 } + \partial r ) ^ { - 1 } ( 0 )$ , where

$$
\begin{array} { r } { F ^ { s } ( \mathbf { z } ) = F ^ { s - 1 } ( \mathbf { z } ) + \mu _ { s } ( \mathbf { z } - \mathbf { z } _ { s } ) , } \end{array}\tag{6.2}
$$

it follows that

$$
\begin{array} { r } { \left\| \mathbf { z } _ { s } - \mathbf { z } _ { s } ^ { \star } \right\| \leq \left\| \mathbf { z } _ { s } - \mathbf { z } _ { s - 1 } ^ { \star } \right\| \ a n d \ \| \mathbf { z } _ { s } ^ { \star } - \mathbf { z } _ { s - 1 } ^ { \star } \| \leq \left\| \mathbf { z } _ { s } - \mathbf { z } _ { s - 1 } ^ { \star } \right\| . } \end{array}
$$

Moreover, we have that $\| \mathring { \mathbf { z } } - \mathbf { z } _ { 0 } ^ { \star } \| \leq \| \mathring { \mathbf { z } } - \mathbf { x } ^ { \star } \|$

Proof. By definition of the solutions, we have that

$$
0 \in ( F ^ { s } + \partial r ) ( \pmb { z } _ { s } ^ { \star } ) \ \mathrm { a n d } \ 0 \in ( F ^ { s - 1 } + \partial r ) ( \pmb { z } _ { s - 1 } ^ { \star } ) \ \mathrm { w h e r e } \ F ^ { s } ( \pmb { z } _ { s - 1 } ^ { \star } ) = F ^ { s - 1 } ( \pmb { z } _ { s - 1 } ^ { \star } ) + \mu _ { s } ( \pmb { z } _ { s - 1 } ^ { \star } - \pmb { z } _ { s } ) ,
$$

because of (6.2). Consequently, we have that $\mu _ { s } ( \mathbf { z } _ { s - 1 } ^ { \star } - \mathbf { z } _ { s } ) \in ( F ^ { s } + \partial r ) ( \mathbf { z } _ { s - 1 } ^ { \star } )$

Let us now use that $( F ^ { s } + \partial r )$ is $\mu _ { s }$ -strongly monotone to write

$$
\begin{array} { r l r } & { } & { \mu _ { s } \bigl \langle \mathbf { z } _ { s } - \mathbf { z } _ { s - 1 } ^ { \star } , \mathbf { z } _ { s } ^ { \star } - \mathbf { z } _ { s - 1 } ^ { \star } \bigr \rangle \geq \mu _ { s } \| \mathbf { z } _ { s } ^ { \star } - \mathbf { z } _ { s - 1 } ^ { \star } \| ^ { 2 } } \\ & { } & { \iff \displaystyle \frac { 1 } { 2 } \left( \| \mathbf { z } _ { s } - \mathbf { z } _ { s - 1 } ^ { \star } \| ^ { 2 } + \| \mathbf { z } _ { s } ^ { \star } - \mathbf { z } _ { s - 1 } ^ { \star } \| ^ { 2 } - \| \mathbf { z } _ { s } - \mathbf { z } _ { s } ^ { \star } \| ^ { 2 } \right) \geq \| \mathbf { z } _ { s } ^ { \star } - \mathbf { z } _ { s - 1 } ^ { \star } \| ^ { 2 } . } \end{array}
$$

Rearranging the final inequality gives both results.

In view of (2.2), if F is µ-strongly monotone, the last claim holds with equality. Otherwise, we use µ-strong monotonicity of $F ^ { \mu } + \partial r$ as in the first part of the proof, along with

$$
0 \in ( F ^ { \mu } + \partial r ) ( \mathbf { z } _ { 0 } ^ { \star } ) , \quad 0 \in ( F + \partial r ) ( \mathbf { x } ^ { \star } ) , \quad F ^ { \mu } ( \mathbf { x } ^ { \star } ) = F ( \mathbf { x } ^ { \star } ) + \mu ( \mathbf { x } ^ { \star } - \hat { \mathbf { z } } ) .
$$

This establishes the last inequality in the statement.

This lemma generalizes [CL24, Lemma A.1] which was specialized to unconstrained min-max problems. The extension is to handle the multi-valued mapping $\partial r$ in the definition of subproblems. For this, we keep all the terms on the right-hand side to depend on the quality of approximate subproblem solutions.

Lemma 6.2. Under Assumptions ${ 1 , \ 2 }$ and the setup of $( 2 . 1 ) \ - ( 2 . 3 )$ , set $\begin{array} { r } { \eta = \frac { 1 } { 3 \widehat { L } _ { F } } } \end{array}$ where $\widehat { L } _ { F } = L _ { F } + B$ . Then, we have

$$
\| \mathcal { G } _ { \eta , H ^ { \mu } } ( \mathbf { z } _ { S } ) \| \leq 2 \sum _ { j = 1 } ^ { S } \mu _ { j } \| \mathbf { z } _ { j } - \mathbf { z } _ { j - 1 } ^ { \star } \| + 9 \widehat { L } _ { F } \| \mathbf { z } _ { S } - \mathbf { z } _ { S - 1 } ^ { \star } \| ,\tag{6.3}
$$

where $\mu _ { s } = 2 ^ { s } \mu$ and $\mathbf { z } _ { s } , \mathbf { z } _ { s } ^ { \star }$ are as given in (6.1) (see also (2.3) and (2.1)).

Proof. We first recall the definition of the residual for operators $H ^ { \mu } = F ^ { \mu } + \partial r$ and $H ^ { s } = F ^ { s } + \partial r$ , respectively:

$$
\mathcal { G } _ { \eta , H ^ { \mu } } ( \mathbf { z } ) = \eta ^ { - 1 } \left( \mathbf { z } - \mathrm { p r o x } _ { \eta r } ( \mathbf { z } - \eta F ^ { \mu } ( \mathbf { z } ) ) \right) \quad \mathrm { a n d } \quad \mathcal { G } _ { \eta , H ^ { s - 1 } } ( \mathbf { z } ) = \eta ^ { - 1 } \left( \mathbf { z } - \mathrm { p r o x } _ { \eta r } ( \mathbf { z } - \eta F ^ { S - 1 } ( \mathbf { z } ) ) \right) .
$$

By these definitions, we have

$$
\begin{array} { r l } { \| \mathcal { G } _ { \eta , H ^ { \mu } } ( \mathbf { z } _ { S } ) - \mathcal { G } _ { \eta , H ^ { S - 1 } } ( \mathbf { z } _ { S } ) \| = \eta ^ { - 1 } \| \operatorname { p r o x } _ { \eta r } ( \mathbf { z } _ { S } - \eta F ^ { \mu } ( \mathbf { z } _ { S } ) ) - \operatorname { p r o x } _ { \eta r } ( \mathbf { z } _ { S } - \eta F ^ { S - 1 } ( \mathbf { z } _ { S } ) ) \| } & { } \\ & { \leq \| F ^ { \mu } ( \mathbf { z } _ { S } ) - F ^ { S - 1 } ( \mathbf { z } _ { S } ) \| } \\ & { \leq \displaystyle \sum _ { i = 1 } ^ { S - 1 } \mu _ { i } \| \mathbf { z } _ { S } - \mathbf { z } _ { i } \| , } \end{array}\tag{6.4}
$$

where the first inequality used nonexpansiveness of the proximal operator, see for example [Bec17, Theorem 6.42]. The last inequality is by (6.1) and triangle inequality.

Now we estimate the term on the right-hand side by using triangle inequality and Lemma 6.1

$$
\| \mathbf { z } _ { S } - \mathbf { z } _ { i } \| \leq \| \mathbf { z } _ { S } - \mathbf { z } _ { S - 1 } ^ { \star } \| + \sum _ { j = i } ^ { S - 2 } \| \mathbf { z } _ { j + 1 } ^ { \star } - \mathbf { z } _ { j } ^ { \star } \| + \| \mathbf { z } _ { i } ^ { \star } - \mathbf { z } _ { i } \| \leq \sum _ { j = i } ^ { S } \| \mathbf { z } _ { j } - \mathbf { z } _ { j - 1 } ^ { \star } \| .
$$

Substituting this and changing the order of summations give

$$
\begin{array} { r l r } {  { \sum _ { i = 1 } ^ { S - 1 } \mu _ { i } \| { \bf z } _ { S } - { \bf z } _ { i } \| \leq \sum _ { i = 1 } ^ { S - 1 } \mu _ { i } \sum _ { j = i } ^ { S } \| { \bf z } _ { j } - { \bf z } _ { j - 1 } ^ { \star } \| } } \\ & { } & { = \sum _ { j = 1 } ^ { S - 1 } \sum _ { i = 1 } ^ { j } \mu _ { i } \| { \bf z } _ { j } - { \bf z } _ { j - 1 } ^ { \star } \| + \sum _ { i = 1 } ^ { S - 1 } \mu _ { i } \| { \bf z } _ { S } - { \bf z } _ { S - 1 } ^ { \star } \| . } \end{array}
$$

By using the setting $\mu _ { i } = 2 ^ { i } \mu$ , we have $\begin{array} { r } { \sum _ { i = 1 } ^ { j } \mu _ { i } = \sum _ { i = 1 } ^ { j } 2 ^ { i } \mu \leq 2 ^ { j + 1 } \mu = \mu _ { j + 1 } = 2 \mu _ { j } } \end{array}$ and $\textstyle \sum _ { i = 1 } ^ { S - 1 } \mu _ { i } \leq \mu _ { S }$ , we get

$$
\sum _ { i = 1 } ^ { S - 1 } \mu _ { i } \| \mathbf { z } _ { S } - \mathbf { z } _ { i } \| \leq 2 \sum _ { j = 1 } ^ { S } \mu _ { j } \| \mathbf { z } _ { j } - \mathbf { z } _ { j - 1 } ^ { \star } \| .
$$

Using this in (6.4) gives

$$
\| \mathcal { G } _ { \eta , H ^ { \mu } } ( \mathbf { z } _ { S } ) \| \leq \| \mathcal { G } _ { \eta , H ^ { S - 1 } } ( \mathbf { z } _ { S } ) \| + 2 \sum _ { j = 0 } ^ { S - 1 } \mu _ { j + 1 } \| \mathbf { z } _ { j + 1 } - \mathbf { z } _ { j } ^ { \star } \| ,\tag{6.5}
$$

where we used the triangle inequality on the left-hand side of (6.4).

Next, we estimate $\| \mathcal { G } _ { \eta , H ^ { S - 1 } } ( \mathbf { z } _ { S } ) \|$ . First, by the definition of $\mathbf { z } _ { S - 1 } ^ { \star } \ ( \mathrm { s e e \ ( 6 . 1 ) } )$ , we have

$$
\begin{array} { r l } & { \mathcal { G } _ { \eta , H ^ { S - 1 } } ( \mathbf { z } _ { S - 1 } ^ { \star } ) = 0 \iff \mathbf { z } _ { S - 1 } ^ { \star } = \operatorname { p r o x } _ { \eta r } ( \mathbf { z } _ { S - 1 } ^ { \star } - \eta F ^ { S - 1 } ( \mathbf { z } _ { S - 1 } ^ { \star } ) ) } \\ & { \qquad \iff \mathbf { z } _ { S - 1 } ^ { \star } + \eta \partial r ( \mathbf { z } _ { S - 1 } ^ { \star } ) \ni \mathbf { z } _ { S - 1 } ^ { \star } - \eta F ^ { S - 1 } ( \mathbf { z } _ { S - 1 } ^ { \star } ) } \\ & { \qquad \iff 0 \in ( \partial r + F ^ { S - 1 } ) ( \mathbf { z } _ { S - 1 } ^ { \star } ) . } \end{array}
$$

Using this, we estimate as

$$
\begin{array} { r l } { \| \mathcal { G } _ { \eta , H ^ { s - 1 } } ( \mathbf { z } _ { S } ) \| = \| \mathcal { G } _ { \eta , H ^ { s - 1 } } ( \mathbf { z } _ { S } ) - \mathcal { G } _ { \eta , H ^ { s - 1 } } ( \mathbf { z } _ { S - 1 } ^ { \star } ) \| } & { } \\ & { \leq \eta ^ { - 1 } \left( \| \mathbf { z } _ { S } - \mathbf { z } _ { S - 1 } ^ { \star } \| + \| \operatorname { p r o x } _ { \eta r } ( \mathbf { z } _ { S } - \eta F ^ { S - 1 } ( \mathbf { z } _ { S } ) ) - \operatorname { p r o x } _ { \eta r } ( \mathbf { z } _ { S - 1 } ^ { \star } - \eta F ^ { S - 1 } ( \mathbf { z } _ { S - 1 } ^ { \star } ) ) \| \right) } \\ & { \leq \eta ^ { - 1 } \left( 2 \| \mathbf { z } _ { S } - \mathbf { z } _ { S - 1 } ^ { \star } \| + \eta \| F ^ { S - 1 } ( \mathbf { z } _ { S } ) - F ^ { S - 1 } ( \mathbf { z } _ { S - 1 } ^ { \star } ) \| \right) } \\ & { \leq \eta ^ { - 1 } \left( 2 \| \mathbf { z } _ { S } - \mathbf { z } _ { S - 1 } ^ { \star } \| + \eta L _ { S - 1 } \| \mathbf { z } _ { S } - \mathbf { z } _ { S - 1 } ^ { \star } \| \right) , } \end{array}
$$

where the first step is by triangle inequality, the second by triangle inequality and nonexpansiveness of the proximal operator, the third by the $L _ { S - 1 } \mathrm { - L i p s c h i t z n e s s }$ of $\overrightharpoon { F } ^ { S - 1 }$ . Since $L _ { S - 1 } \leq 2 \widehat { L } _ { F }$ , we insert this into (6.5) to obtain the bound.

This lemma is extending [CL24, Lemma 3.1] to using gradient mapping instead of the gradient and the existence of the regularizer r in the subproblems. In contrast to [AZ18, Lemma 5.1], we cannot rely on inequalities specific to minimization such as the descent lemma.

Lemma 6.3. Let $F$ be monotone and $F ^ { \mu } , ( { \bf z } _ { s } ^ { \star } ) , ( { \bf z } _ { s } )$ defined as (2.2), (2.3), and Algorithm ${ \mathcal { B } } ,$ respectively. Set $\begin{array} { r } { \eta = \frac { 1 } { 3 \widehat { L } _ { F } } } \end{array}$ where $\widehat { L } _ { F } = L _ { F } + B$ . Then, we have

$$
\| \mathcal { G } _ { \eta , H } ( \mathbf { z } _ { S } ) \| \leq 2 . 5 \sum _ { j = 1 } ^ { S } \mu _ { j } \| \mathbf { z } _ { j } - \mathbf { z } _ { j - 1 } ^ { \star } \| + 9 \widehat { L } _ { F } \| \mathbf { z } _ { S } - \mathbf { z } _ { S - 1 } ^ { \star } \| + \mu \| \mathring { \mathbf { z } } - \mathbf { x } ^ { \star } \| .\tag{6.6}
$$

Proof. Since $F$ is only monotone, we are in the second case of (2.2). First, we characterize the diference between the residual for solving the original problem (MI) and the perturbed problem (2.1).

By the definition of the residual, we have

$$
{ \mathcal { G } } _ { \eta , H } ( \mathbf { z } ) = \eta ^ { - 1 } \left( \mathbf { z } - \operatorname { p r o x } _ { \eta r } ( \mathbf { z } - \eta F ( \mathbf { z } ) ) \right) \quad { \mathrm { a n d } } \quad { \mathcal { G } } _ { \eta , H ^ { \mu } } ( \mathbf { z } ) = \eta ^ { - 1 } \left( \mathbf { z } - \operatorname { p r o x } _ { \eta r } ( \mathbf { z } - \eta F ^ { \mu } ( \mathbf { z } ) ) \right) .
$$

With this, nonexpansiveness of the proximal operator and the triangle inequality, we obtain

$$
\begin{array} { r } { \| \mathcal { G } _ { \eta , H } ( \mathbf { z } _ { S } ) - \mathcal { G } _ { \eta , H ^ { \mu } } ( \mathbf { z } _ { S } ) \| \leq \| F ( \mathbf { z } _ { S } ) - F ^ { \mu } ( \mathbf { z } _ { S } ) \| \leq \mu \| \mathbf { z } _ { S } - \hat { \mathbf { z } } \| \leq \mu ( \| \mathbf { z } _ { S } - \mathbf { z } _ { 0 } ^ { \star } \| + \| \mathbf { z } _ { 0 } ^ { \star } - \hat { \mathbf { z } } \| ) , } \end{array}\tag{6.7}
$$

where $\mathbf { z } _ { \mathrm { 0 } } ^ { \star }$ is the solution of the perturbed problem as in (2.1) and the second inequality uses the definition in the second case of (2.2). For the second term on the right-hand side of (6.7), we can use Theorem 6.1 to get

$$
\| \mathring { \mathbf { z } } - \mathbf { z } _ { 0 } ^ { \star } \| \leq \| \mathring { \mathbf { z } } - \mathbf { x } ^ { \star } \| .\tag{6.8}
$$

For the first term on the right-hand side of (6.7), we estimate similar to the proof of the Theorem 6.2:

$$
\| \mathbf { z } _ { S } - \mathbf { z } _ { 0 } ^ { \star } \| \leq \| \mathbf { z } _ { S } - \mathbf { z } _ { S - 1 } ^ { \star } \| + \sum _ { j = 1 } ^ { S - 1 } \| \mathbf { z } _ { j } ^ { \star } - \mathbf { z } _ { j - 1 } ^ { \star } \| \leq \| \mathbf { z } _ { S } - \mathbf { z } _ { S - 1 } ^ { \star } \| + \sum _ { j = 1 } ^ { S - 1 } \| \mathbf { z } _ { j } - \mathbf { z } _ { j - 1 } ^ { \star } \| = \sum _ { j = 1 } ^ { S } \| \mathbf { z } _ { j } - \mathbf { z } _ { j - 1 } ^ { \star } \| ,
$$

where the second step used Theorem 6.1.

On (6.7), we substitute the last two estimates to the right-hand side and use triangle inequality on the left-hand side to deduce

$$
\begin{array} { r l r } {  { \| \mathcal { G } _ { \eta , H } ( \mathbf { z } _ { S } ) \| \leq \| \mathcal { G } _ { \eta , H ^ { \mu } } ( \mathbf { z } _ { S } ) \| + \mu \displaystyle \sum _ { j = 1 } ^ { S } \| \mathbf { z } _ { j } - \mathbf { z } _ { j - 1 } ^ { \star } \| + \mu \| \bar { \mathbf { z } } - \mathbf { x } ^ { \star } \| } } \\ & { } & { \leq 2 . 5 \displaystyle \sum _ { j = 1 } ^ { S } \mu _ { j } \| \mathbf { z } _ { j } - \mathbf { z } _ { j - 1 } ^ { \star } \| + 9 \widehat { L } _ { F } \| \mathbf { z } _ { S } - \mathbf { z } _ { S - 1 } ^ { \star } \| + \mu \| \bar { \mathbf { z } } - \mathbf { x } ^ { \star } \| , } \end{array}\tag{6.9}
$$

where the last step is because of $\mu _ { j } = 2 ^ { j } \mu \Rightarrow \mu \leq \mu _ { j } / 2$ and Theorem 6.2.

Lemma 6.4. For problem (MI), let Assumptions 1 and 2 hold. We apply Algorithm 3 to solve (2.1) with budget $T \geq 4 8 S \kappa = 4 8 S \widehat { L } _ { F } / \mu$ for the number of oracles where $\begin{array} { r } { S = \left\lfloor \log _ { 2 } \frac { \widehat { L } _ { F } } { \mu } \right\rfloor \ge 1 } \end{array}$ and initial point ˚z. Set $\begin{array} { r } { \eta = \frac { 1 } { 3 \widehat { L } _ { F } } } \end{array}$ where $\widehat { L } _ { F } = L _ { F } + B$ . If we have an estimate of the form

$$
\mu _ { s } ^ { 2 } \mathbb { E } \| \mathbf { z } _ { s } - \mathbf { z } _ { s - 1 } ^ { \star } \| ^ { 2 } \leq \left( \frac { 1 } { 2 } \right) ^ { \frac { T s } { 2 4 \kappa S } } \mu ^ { 2 } \| \widehat { \mathbf { z } } - \mathbf { z } _ { 0 } ^ { \star } \| ^ { 2 } + \frac { C S \widehat { G } ^ { 2 } } { T } ,
$$

for some $C , { \widehat { G } } _ { \mathrm { { i } } }$ , then it follows that

$$
\mathbb { E } \| \mathcal { G } _ { \eta , H ^ { \mu } } ( \mathbf { z } _ { S } ) \| \leq \left( 4 \left( \frac { 1 } { 2 } \right) ^ { \frac { T } { 4 8 \kappa \mathrm { { s } } \mathrm { { s } } } } + 1 8 \left( \frac { 1 } { 2 } \right) ^ { \frac { T } { 4 8 \kappa } } \right) \mu \| \widehat { \mathbf { z } } - \mathbf { z } _ { 0 } ^ { \star } \| + \frac { \sqrt { C } \widehat { G } } { \sqrt { T } } \left( 2 S ^ { 3 / 2 } + 1 8 \sqrt { S } \right) .
$$

Proof. Let us recall that the hypothesis of the lemma implies for all $s = 1 , \ldots , S$ that

$$
\mu _ { s } \mathbb { E } \| { \mathbf z } _ { s } - { \mathbf z } _ { s - 1 } ^ { \star } \| \leq \left( \frac { 1 } { 2 } \right) ^ { \frac { T s } { 4 8 \kappa S } } \mu \| \hat { \mathbf z } - { \mathbf z } _ { 0 } ^ { \star } \| + \frac { \sqrt { C S } \widehat { G } } { \sqrt { T } } ,\tag{6.10}
$$

by Jensen’s inequality and concavity of the square root. Summing up both sides of the inequality gives

$$
\sum _ { s = 1 } ^ { S } \mu _ { s } \mathbb { E } \| \mathbf { z } _ { s } - \mathbf { z } _ { s - 1 } ^ { \star } \| \leq \mu \| \hat { \mathbf { z } } - \mathbf { z } _ { 0 } ^ { \star } \| \sum _ { s = 1 } ^ { S } \left( \frac { 1 } { 2 } \right) ^ { \frac { T _ { S } } { 4 8 \kappa 5 } } + \frac { \sqrt { C } S ^ { 3 / 2 } \widehat { G } } { \sqrt { T } } \leq 2 \left( \frac { 1 } { 2 } \right) ^ { \frac { T } { 4 8 \kappa 5 } } \mu \| \hat { \mathbf { z } } - \mathbf { z } _ { 0 } ^ { \star } \| + \frac { \sqrt { C } S ^ { 3 / 2 } \widehat { G } } { \sqrt { T } } ,\tag{6.11}
$$

where the last estimate is summing the geometric series: since $T \geq 4 8 \kappa S$ , we have $\begin{array} { r } { \sum _ { j = 1 } ^ { s } \left( \frac { 1 } { 2 } \right) ^ { T j / ( 4 8 \kappa S ) } \le } \end{array}$ $2 \left( { \textstyle { \frac { 1 } { 2 } } } \right) ^ { \frac { T } { 4 8 \kappa S } }$ . This is because $\begin{array} { r } { \sum _ { j = 1 } ^ { s } \alpha ^ { j } = \frac { \alpha ( 1 - \alpha ^ { S } ) } { 1 - \alpha } \leq \frac { \alpha } { 1 - \alpha } } \end{array}$ where $\alpha = \left( \frac { 1 } { 2 } \right) ^ { T / ( 4 8 \kappa S ) } \leq 1 / 2$

In view of the second term on the right-hand side of the result of Theorem 6.2, we have

$$
9 \widehat { L } _ { F } \mathbb { E } \lVert \mathbf { z } _ { S } - \mathbf { z } _ { S - 1 } ^ { \star } \rVert \leq 1 8 \mu _ { S } \mathbb { E } \lVert \mathbf { z } _ { S } - \mathbf { z } _ { S - 1 } ^ { \star } \rVert \leq 1 8 \left( \left( \frac { 1 } { 2 } \right) ^ { \frac { T } { 4 8 \kappa } } \mu \lVert \bar { \mathbf { z } } - \mathbf { z } _ { 0 } ^ { \star } \rVert + \frac { \sqrt { S C } \widehat { G } } { \sqrt { T } } \right) ,\tag{6.12}
$$

where the first inequality is because of $\begin{array} { r } { 2 ^ { S + 1 } \geq \frac { \widehat { L } _ { F } } { \mu } } \end{array}$ , which is by the definition of $S$ and $\mu _ { S } = 2 ^ { S } \mu$ . The second inequality is substituting (6.10) with $s = S$

Using (6.11) and (6.12) to estimate the right-hand side of Theorem 6.2 gives the claimed bound. ■

Fact 6.5. Let F satisfy Assumption 2 with constants B, G. Then, $F ^ { s }$ defined in (2.3), satisfies the BG assumption with the same constants. Moreover, $F ^ { s - 1 } ~ i s ~ ( L _ { F } + ( 2 ^ { s } - 1 ) \mu )  – L i p s c h i t z$ $( ( 2 ^ { s } - 1 ) \mu )$ -strongly monotone.

Proof. Since $\begin{array} { r } { F ^ { s } ( { \bf z } ) = F ^ { \mu } ( { \bf z } ) + \sum _ { i = 1 } ^ { s } \mu _ { i } ( { \bf z } - { \bf z } _ { i } ) \mathrm { ~ a n d ~ } F _ { \xi } ^ { s } ( { \bf z } ) = F _ { \xi } ^ { \mu } ( { \bf z } ) + \sum _ { i = 1 } ^ { s } \mu _ { i } ( { \bf z } - { \bf z } _ { i } ) } \end{array}$ , we have that

$$
\begin{array} { r } { \mathbb { E } \| F _ { \xi } ^ { s } ( \mathbf { z } ) - F ^ { s } ( \mathbf { z } ) \| ^ { 2 } = \mathbb { E } \| F _ { \xi } ( \mathbf { z } ) - F ( \mathbf { z } ) \| ^ { 2 } \leq B ^ { 2 } \| \mathbf { z } - \bar { \mathbf { z } } \| ^ { 2 } + G ^ { 2 } . } \end{array}
$$

This gives the first assertion.

For the second, by definition, we have $\begin{array} { r } { F ^ { s - 1 } ( { \bf z } ) = F ^ { \mu } ( { \bf z } ) + \sum _ { i = 1 } ^ { s - 1 } \mu _ { i } ( { \bf z } - { \bf z } _ { i } ) } \end{array}$ which is $L _ { F } + ( 2 ^ { s } - 1 ) \mu \mathrm { - L i p s c h i t z }$ since $\textstyle \sum _ { i = 1 } ^ { s - 1 } \mu _ { i } = \sum _ { i = 1 } ^ { s - 1 } 2 ^ { i } \mu = ( 2 ^ { s } - 2 ) \mu$ . The strong monotonicity constant is calculated in the same way. ■

For getting precise constants for the complexity results and the parameters (see for example Section 4.2), we often solve inequalities of the form $e ^ { x } \geq C x$ . For simplicity, we use the following simple suficient condition to satisfy these inequalities without resorting to the Lambert W function to find exact roots.

Lemma 6.6. [SSBD14, Lemma $A . { \mathcal { Z } } ]$ For some $C > 1 , i f x \geq 4 \ln 2 + 2 \ln C$ , then we have $x \geq \ln x + \ln C \iff$ $e ^ { x } \geq C x$

We now continue to derive numerical suficient conditions for our complexity results.

Lemma 6.7. i. Let $a , b , c , \varepsilon$ be positive real numbers and $\begin{array} { r } { T \geq \frac { 1 } { b ^ { 2 } } } \end{array}$ . Then, the inequality $\begin{array} { r } { \frac { a } { \sqrt { T } } \left( \log _ { 2 } b \sqrt { T } \right) ^ { c } \leq \varepsilon } \end{array}$ holds $\begin{array} { r } { i f T \geq \frac { a ^ { 2 } c ^ { 2 c } } { \varepsilon ^ { 2 } ( \ln 2 ) ^ { 2 c } } \left( 4 \ln 2 + 2 \ln \left( \frac { c ( a b ) ^ { 1 / c } } { \varepsilon ^ { 1 / c } \ln 2 } + 1 \right) \right) ^ { 2 c } } \end{array}$

ii. Let $a , b , c , \varepsilon ^ { \prime }$ be positive real numbers and $T \ \geq \ { \frac { 1 } { b } }$ . Then, the inequality $\begin{array} { r } { \frac { a } { T } ( \log _ { 2 } b T ) ^ { c } \leq \varepsilon ^ { \prime } } \end{array}$ holds if $\begin{array} { r } { T \geq \frac { a c ^ { c } } { \varepsilon ^ { \prime } ( \ln 2 ) ^ { c } } \left( 4 \ln 2 + 2 \ln \left( \frac { c ( a b ) ^ { 1 / c } } { ( \varepsilon ^ { \prime } ) ^ { 1 / c } \ln 2 } + 1 \right) \right) ^ { c } } \end{array}$

iii. $I f T \geq 1 4 1 7 5$ and $\begin{array} { r } { \mu = \frac { 3 2 \widehat { L } _ { F } } { \sqrt { T } } \log _ { 2 } ^ { 3 / 2 } \frac { \sqrt { T } } { 4 8 } } \end{array}$ , then the following inequalities hold: $\begin{array} { r } { T \geq 4 8 \kappa S , \mu ^ { 2 } \geq \frac { 1 0 2 4 S ^ { 3 } \widehat { L } _ { F } ^ { 2 } } { T } } \end{array}$ where $\begin{array} { r } { \kappa = \frac { \widehat { L } _ { F } } { \mu } } \end{array}$ and $\begin{array} { r } { S = \lfloor \log _ { 2 } \frac { \widehat { L } _ { F } } { \mu } \rfloor } \end{array}$ . Moreover, we have $\begin{array} { r } { S \le \log _ { 2 } \frac { \sqrt { T } } { 4 8 } , \kappa \ge 2 } \end{array}$ , and $S \geq 1$

Proof. 1. By the required conditions on $a , b , c , T$ , we have that the required inequality is equivalent to

$$
\log _ { 2 } b \sqrt { T } \leq \left( \frac { \varepsilon \sqrt { T } } { a } \right) ^ { 1 / c } .\tag{6.13}
$$

Let us define x such that

$$
{ \Bigg ( } { \frac { \varepsilon { \sqrt { T } } } { a } } { \Bigg ) } ^ { 1 / c } = { \frac { c x } { \ln 2 } } \iff T ^ { \frac { 1 } { 2 c } } = { \frac { c x a ^ { 1 / c } } { \varepsilon ^ { 1 / c } \ln 2 } } .\tag{6.14}
$$

We also note that

$$
\log _ { 2 } b \sqrt { T } = \frac { c } { \ln 2 } \ln ( b ^ { \frac { 1 } { c } } T ^ { \frac { 1 } { 2 c } } ) .
$$

Plugging in the last two estimates into (6.13) gives

$$
\ln ( b ^ { 1 / c } T ^ { 1 / ( 2 c ) } ) \leq x \iff b ^ { 1 / c } T ^ { 1 / ( 2 c ) } \leq e ^ { x } \iff { \frac { c ( a b ) ^ { 1 / c } } { \varepsilon ^ { 1 / c } \ln 2 } } x \leq e ^ { x } \Leftarrow \left( { \frac { c ( a b ) ^ { 1 / c } } { \varepsilon ^ { 1 / c } \ln 2 } } + 1 \right) x \leq e ^ { x } ,
$$

since $x \geq 0$

From Theorem 6.6, we know that this is satisfied as long as

$$
x \geq 4 \ln 2 + 2 \ln \left( { \frac { c ( a b ) ^ { 1 / c } } { \varepsilon ^ { 1 / c } \ln 2 } } + 1 \right) ,
$$

which is translated to a lower bound on $T$ given in the assertion by using the last identity in (6.14)

2. We use the proof of the first assertion by replacing $T \gets T ^ { 2 }$ and $\varepsilon \gets \varepsilon ^ { \prime }$

3. We first show that $S \geq 1$ for which it is enough to show $\kappa \geq 2$ . Let $x = \log _ { 2 } { \frac { \sqrt { T } } { 4 8 } } \iff { \sqrt { T } } = 4 8 \times 2 ^ { x }$ and hence $\begin{array} { r } { \kappa = \frac { \sqrt { T } } { 3 2 \log _ { 2 } ^ { 3 / 2 } \frac { \sqrt { T } } { 4 8 } } = \frac { \sqrt { T } } { 3 2 x ^ { 3 / 2 } } } \end{array}$ . The inequality $\kappa \geq 2$ is equivalent to

$$
2 ^ { x } \geq { \frac { 4 } { 3 } } x ^ { 3 / 2 } \iff { \frac { ( 2 ^ { 2 / 3 } ) ^ { x } } { x } } \geq \left( { \frac { 4 } { 3 } } \right) ^ { 2 / 3 } .
$$

Denoting $\beta = 2 ^ { 2 / 3 } > 1$ , we have by a standard calculation that min $. _ { x > 0 } \beta ^ { x } / x = e \ln \beta > 1 . 2 5 > 1 . 2 2 >$ $\left( { \frac { 4 } { 3 } } \right) ^ { 2 / 3 }$

We then prove the assertion $\mu ^ { 2 } T \geq 1 0 2 4 S ^ { 3 } \widehat { L } _ { F } ^ { 2 }$ . By a straightforward calculation, we have

$$
\begin{array} { l } { { \mu ^ { 2 } T = 1 0 2 4 \widehat { L } _ { F } ^ { 2 } \log _ { 2 } ^ { 3 } \displaystyle \frac { \sqrt { T } } { 4 8 } \geq 1 0 2 4 S ^ { 3 } \widehat { L } _ { F } ^ { 2 } = 1 0 2 4 \widehat { L } _ { F } ^ { 2 } \left\lfloor \log _ { 2 } \displaystyle \frac { \widehat { L } _ { F } } { \mu } \right\rfloor ^ { 3 } \asymp \log _ { 2 } \displaystyle \frac { \sqrt { T } } { 4 8 } \geq \log _ { 2 } \displaystyle \frac { \sqrt { T } } { 3 2 \log _ { 2 } ^ { 3 / 2 } \frac { \sqrt { T } } { 4 8 } } } } \\ { { \longleftrightarrow \log _ { 2 } ^ { 3 / 2 } \displaystyle \frac { \sqrt { T } } { 4 8 } \geq \frac { 3 } { 2 } \Leftarrow T \geq 1 4 1 7 5 . } } \end{array}
$$

Recalling $\begin{array} { r } { x = \log _ { 2 } { \frac { \sqrt { T } } { 4 8 } } } \end{array}$ , we have just established that $x \ge S$ . Then, since $\begin{array} { r } { \kappa = \frac { \sqrt { T } } { 3 2 \log _ { 2 } ^ { 3 / 2 } \frac { \sqrt { T } } { 4 8 } } = \frac { \sqrt { T } } { 3 2 x ^ { 3 / 2 } } } \end{array}$ , we have

$$
4 8 \kappa S \leq 4 8 \kappa x \leq { \frac { 3 { \sqrt { T } } } { 2 { \sqrt { x } } } } \leq { \frac { 3 { \sqrt { T } } } { 2 } } \leq T ,
$$

since $x \geq 1$ and $\sqrt { T } \geq 3 / 2$

## 7 Conclusions

Our goal in this work was to obtain a complexity result for the gradient mapping that is optimal up to logarithmic terms for solving constrained convex-concave min-max problems or stochastic VIs.

Our construction relied on the idea of recursive regularization of [AZ18] to get the best complexity, resulting in a three-loop construction. A clear direction for future research is to obtain the same complexity with a simpler approach, ideally with a single-loop algorithm. We believe that one may be able to go for a two-loop construction by combining [KLL22] and [AZ18], although it is not clear to us if this construction would still be able to handle the BG assumption. Obtaining the same complexity with a single-loop algorithm is open to our knowledge, even for minimization problems.

The second direction is analyzing the monotone case directly, rather than using perturbation by a small parameter $\mu .$ This is necessary even for minimization problems to obtain the best gradient-norm guarantee to our knowledge, so we believe it is a question with a wider scope than min-max problems. Third, we believe that extension of our results to general monotone inclusions is possible by changing the inner solver. Finally, by using similar ideas to [CL24], we suspect it would be possible to allow some form of nonmonotonicity, to solve problems with cohypomonotone operators and constraints.

## References

[AMW25] Ahmet Alacaoglu, Yura Malitsky, and Stephen J Wright. Towards weaker variance assumptions for stochastic optimization. arXiv:2504.09951, 2025.

[AZ18] Zeyuan Allen-Zhu. How to make the gradients small stochastically: Even faster convex and nonconvex sgd. Advances in Neural Information Processing Systems, 31, 2018.

[Bec17] Amir Beck. First-order methods in optimization. SIAM, 2017.

[Blu54] Julius R Blum. Approximation methods which converge with probability one. The Annals of Mathematical Statistics, pages 382–386, 1954.

[BMSV21] Radu Ioan Bot¸, Panayotis Mertikopoulos, Mathias Staudigl, and Phan Tu Vuong. Minibatch forward-backward-forward methods for solving stochastic variational inequalities. Stochastic Systems, 11(2):112–139, 2021.

[B¨oh23] Axel B¨ohm. Solving nonconvex-nonconcave min-max problems exhibiting weak minty solutions. Transactions on Machine Learning Research, 2023.

[CAD24] Xufeng Cai, Ahmet Alacaoglu, and Jelena Diakonikolas. Variance reduced halpern iteration for finite-sum monotone inclusions. In International Conference on Learning Representations, volume 2024, pages 42693–42725, 2024.

[CL24] Lesi Chen and Luo Luo. Near-optimal algorithms for making the gradient small in stochastic minimax optimization. Journal of Machine Learning Research, 25(387):1–44, 2024.

[COZ22] Yang Cai, Argyris Oikonomou, and Weiqiang Zheng. Tight last-iterate convergence of the extragradient and the optimistic gradient descent-ascent algorithm for constrained monotone variational inequalities. arXiv preprint arXiv:2204.09228, 2022.

[CSGD22] Xufeng Cai, Chaobing Song, Crist´obal Guzm´an, and Jelena Diakonikolas. Stochastic halpern iteration with variance reduction for stochastic monotone inclusions. Advances in Neural Information Processing Systems, 35:24766–24779, 2022.

[DDJ21] Jelena Diakonikolas, Constantinos Daskalakis, and Michael I Jordan. Eficient methods for structured nonconvex-nonconcave min-max optimization. In International Conference on Artificial Intelligence and Statistics, pages 2746–2754. PMLR, 2021.

[Dia20] Jelena Diakonikolas. Halpern iteration for near-optimal and parameter-free monotone inclusion and strong solutions to variational inequalities. In Conference on Learning Theory, pages 1428–1451. PMLR, 2020.

[FP03] Francisco Facchinei and Jong-Shi Pang. Finite-dimensional variational inequalities and complementarity problems. Springer, 2003.

[Gla65] EG Gladyshev. On stochastic approximation. Theory of Probability & Its Applications, 10(2):275– 278, 1965.

[Hal67] Benjamin Halpern. Fixed points of nonexpanding maps. Bulletin of the American Mathematical Society, 73(6):957–961, 1967.

[HK14] Elad Hazan and Satyen Kale. Beyond the regret minimization barrier: optimal algorithms for stochastic strongly-convex optimization. Journal of Machine Learning Research, 15(1):2489–2512, 2014.

[IJOT17] Alfredo N Iusem, Alejandro Jofr´e, Roberto Imbuzeiro Oliveira, and Philip Thompson. Extragradient method with variance reduction for stochastic variational inequalities. SIAM Journal on Optimization, 27(2):686–724, 2017.

[JNT11] Anatoli Juditsky, Arkadi Nemirovski, and Claire Tauvel. Solving variational inequalities with stochastic mirror-prox algorithm. Stochastic Systems, 1(1):17–58, 2011.

[KG22] Dmitry Kovalev and Alexander Gasnikov. The first optimal algorithm for smooth and stronglyconvex-strongly-concave minimax optimization. Advances in Neural Information Processing Systems, 35:14691–14703, 2022.

[Kim21] Donghwan Kim. Accelerated proximal point method for maximally monotone operators. Mathematical Programming, 190(1):57–87, 2021.

[KLL22] Georgios Kotsalis, Guanghui Lan, and Tianjiao Li. Simple and optimal methods for stochastic variational inequalities, i: operator extrapolation. SIAM Journal on Optimization, 32(3):2041–2073, 2022.

[Kor76] Galina M Korpelevich. The extragradient method for finding saddle points and other problems. Matecon, 12:747–756, 1976.

[LK21] Sucheol Lee and Donghwan Kim. Fast extra gradient methods for smooth structured nonconvexnonconcave minimax problems. Advances in Neural Information Processing Systems, 34:22588– 22600, 2021.

[LL26] Guanghui Lan and Yan Li. A novel catalyst scheme for stochastic minimax optimization. Mathematical Programming, pages 1–49, 2026.

[MKS<sup>+</sup>20] Konstantin Mishchenko, Dmitry Kovalev, Egor Shulgin, Peter Richt´arik, and Yura Malitsky. Revisiting stochastic extragradient. In International Conference on Artificial Intelligence and Statistics, pages 4573–4582. PMLR, 2020.

[MT20] Yura Malitsky and Matthew K Tam. A forward-backward splitting method for monotone inclusions without cocoercivity. SIAM Journal on Optimization, 30(2):1451–1472, 2020.

[ND16] Hongseok Namkoong and John C Duchi. Stochastic gradient methods for distributionally robust optimization with f-divergences. Advances in Neural Information Processing Systems, 29, 2016.

[Nem04] Arkadi Nemirovski. Prox-method with rate of convergence o (1/t) for variational inequalities with lipschitz continuous monotone operators and smooth convex-concave saddle point problems. SIAM Journal on Optimization, 15(1):229–251, 2004.

[Nes09] Yurii Nesterov. Primal-dual subgradient methods for convex problems. Mathematical Programming, 120(1):221–259, 2009.

[Nes12] Yurii Nesterov. How to make the gradients small. Optima. Mathematical Optimization Society Newsletter, (88):10–11, 2012.

[Nes13] Yurii Nesterov. Gradient methods for minimizing composite functions. Mathematical Programming, 140(1):125–161, 2013.

[NJLS09] Arkadi Nemirovski, Anatoli Juditsky, Guanghui Lan, and Alexander Shapiro. Robust stochastic approximation approach to stochastic programming. SIAM Journal on Optimization, 19(4):1574– 1609, 2009.

[NO24] Gergely Neu and Nneka Okolo. Dealing with unbounded gradients in stochastic saddle-point optimization. In International Conference on Machine Learning, pages 37508–37530, 2024.

[PFL<sup>+</sup>23] Thomas Pethick, Olivier Fercoq, Puya Latafat, Panagiotis Patrinos, and Volkan Cevher. Solving stochastic weak minty variational inequalities without increasing batch size. In International Conference on Learning Representations, 2023.

[SSBD14] Shai Shalev-Shwartz and Shai Ben-David. Understanding machine learning: From theory to algorithms. Cambridge university press, 2014.

[TDNT25] Quoc Tran-Dinh and Nghia Nguyen-Trung. Vfog: Variance-reduced fast optimistic gradient methods for a class of nonmonotone generalized equations. arXiv:2508.16791, 2025.

[TDNT26] Quoc Tran-Dinh and Nghia Nguyen-Trung. Unbiased and biased variance-reduced forwardreflected-backward splitting methods for stochastic composite inclusions. arXiv preprint arXiv:2603.15576, 2026.

[Tse00] Paul Tseng. A modified forward-backward splitting method for maximal monotone mappings. SIAM Journal on Control and Optimization, 38(2):431–446, 2000.

[YR21] TaeHo Yoon and Ernest K Ryu. Accelerated algorithms for smooth convex-concave minimax problems with O(1/k<sup>2</sup>) rate on squared gradient norm. In International Conference on Machine Learning, pages 12098–12109. PMLR, 2021.

[Zha22] Renbo Zhao. Accelerated stochastic algorithms for convex-concave saddle-point problems. Mathematics of Operations Research, 47(2):1443–1473, 2022.