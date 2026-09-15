# Improving the Last-Iterate Guarantees of Anytime Algorithms for Stochastic Monotone Variational Inequalities

Jun-Hyun Kim<sup>∗</sup>

Ahmet Alacaoglu<sup>†</sup>

## Abstract

We analyze a stochastic algorithm with Halpern anchoring for constrained convex-concave problems and monotone variational inequalities. This algorithm is single-loop and single-call since it uses one unbiased sample of the gradient operator at every iteration to be applicable to monotone games with noisy feedback. With t denoting the iteration counter, we prove the anytime last-iterate convergence rate of $\dot { O } ( t ^ { - 1 / 4 } )$ for both gradient-mapping norm and restricted gap, improving the best-known rate $O ( t ^ { - 1 / 5 } )$ that was obtained for the restricted gap function. Our rates cover constrained problems with a potentially unbounded feasible set as well as a structured class of oracles without a uniformly bounded variance.

## 1 Introduction

We focus on the unifying framework of monotone variational inequalities (VIs) that cover such problems as monotone games and convex-concave min-max problems. In particular, we focus on the specific inclusion describing this problem, given as

$$
0 \in G ( { \mathbf { z } } ^ { \star } ) + \partial r ( { \mathbf { z } } ^ { \star } ) ,\tag{1.1}
$$

where $G \colon  { \mathbb { R } } ^ { d } \to  { \mathbb { R } } ^ { d }$ is monotone, L-Lipschitz; and $r \colon \mathbb { R } ^ { d } \to \mathbb { R } \cup \{ + \infty \}$ is proper, convex, closed. This is equivalent to the following form of the monotone VI, where the goal is to find $\mathbf { z } ^ { \star }$ such that

$$
\langle G ( \mathbf { z } ^ { \star } ) , \mathbf { z } - \mathbf { z } ^ { \star } \rangle + r ( \mathbf { z } ) - r ( \mathbf { z } ^ { \star } ) \geq 0 \mathrm { ~ f o r ~ e v e r y ~ } \mathbf { z } \in \mathbb { R } ^ { d } .\tag{1.2}
$$

An important application of (1.2) is learning in monotone games [CBL06], where Nash equilibria can be characterized as solutions of a monotone VI, see, e.g., [MZ19]. In this setting, one action profile is played and one noisy payof feedback is observed at each round. Hence, single-call methods fit this feedback model, whereas two-call methods require an additional gradient evaluation at a diferent action profile. Since the current action profile is the one actually played and the horizon is generally unknown, last-iterate and anytime guarantees are particularly relevant [CZ23]. As a result, standard results for stochastic VI algorithms that show convergence rates on the averaged iterate or those that use increasing mini-batches or multi-loop algorithms are not suitable for this setting.

Optimality measures. Two standard optimality measures we will consider are gradient mapping norm, also known as the natural residual, and the restricted gap function. In particular, for a fixed $\rho > 0$ , define the gradient mapping by

$$
\mathcal { G } _ { \rho } ( \mathbf { z } ) : = \frac { 1 } { \rho } \big ( \mathbf { z } - \mathrm { p r o x } _ { \rho r } ( \mathbf { z } - \rho G ( \mathbf { z } ) ) \big ) .\tag{1.3}
$$

The norm $\| \mathcal { G } _ { \rho } ( \mathbf { z } ) \|$ vanishes if and only if (1.1) holds at z. For simplicity we will set $\textstyle \rho = { \frac { 1 } { L } }$ throughout. This is an optimality measure generalizing the gradient norm in the unconstrained case and is suitable for problems with unbounded feasible sets, in contrast to the optimality measure we introduce next.

<table><tr><td>Method</td><td>Constraint</td><td></td><td>Anytime Single proj.</td><td>Last-iterate rate and measure</td><td>Variance assumption</td></tr><tr><td>[ASAI25]</td><td>Bounded</td><td>X</td><td>V</td><td> $\widetilde { O } ( T ^ { - 1 / 7 } )$  , gap</td><td>Bounded</td></tr><tr><td rowspan="2">[ZLS26] [ZLS26]</td><td>Possibly unbounded</td><td>×</td><td>X</td><td> ${ \cal O } ( T ^ { - 1 / 4 } ) , \mathrm { g a p }$ </td><td>Bounded‡</td></tr><tr><td>Possibly unbounded</td><td>√</td><td>X</td><td> $O ( t ^ { - 1 / 5 } ) , { \mathrm { g a p } }$ </td><td>Bounded</td></tr><tr><td>[ITAA26]</td><td>Bounded</td><td>×</td><td>√</td><td> ${ \cal O } ( T ^ { - 1 / 4 } ) , \mathrm { g a p }$ </td><td>Bounded</td></tr><tr><td>[ITAA26]</td><td>Bounded</td><td>V</td><td>V</td><td> $O ( t ^ { - 1 / 5 } ) , { \mathrm { g a p } }$ </td><td>Bounded</td></tr><tr><td> $[ \mathrm { S Y L J ^ { + } 2 6 } ]$ </td><td>Unconstrained</td><td>√</td><td>√</td><td> $O ( t ^ { - 1 / 4 } ) , { \mathrm { g r a d } } .$  norm</td><td>(1.6)</td></tr><tr><td>This work Possibly unbounded</td><td></td><td>√</td><td></td><td> $O ( t ^ { - 1 / 4 } )$  , grad. mapping  $O ( t ^ { - 1 / 4 } )$  , gap</td><td>Blum-Gladyshev</td></tr></table>

Table 1: Comparison of single-loop, single-call stochastic methods with last-iterate guarantees. The symbol ✓ indicates an anytime method that does not require prior knowledge of the horizon T. The rates are stated with T for fixed horizon and t for anytime results. The Blum-Gladyshev condition is given in Assumption 1.2. <sup>‡</sup>[ZLS26, Section 2.1] remarked that their analysis extends to a stronger variant of our Assumption 1.2, but did not provide a proof.

We present the gap function for the special case of $r ( \mathbf { z } ) = \delta _ { Z } ( \mathbf { z } )$ where $\delta _ { Z }$ is the indicator function for a closed and convex set Z

$$
\mathrm { g a p } ( \mathbf { z } ) = \operatorname* { m a x } _ { \mathbf { u } \in Z } \left\{ \langle G ( \mathbf { z } ) , \mathbf { z } - \mathbf { u } \rangle \right\} .\tag{1.4}
$$

Since this quantity is not suitable when set $Z$ is not bounded, restricted versions of gap functions are often used [Nes07]. For restricted gap functions, they generally take the maximum over a compact set U and this set needs to satisfy certain requirements for restricted gap functions to be valid optimality measures [Nes07, Lemma 1]. For example, the set needs to contain the iterates of the algorithm and a solution. Since it is generally not possible to show the iterates stay bounded for stochastic algorithms, this optimality measure is less meaningful without a bounded domain or for stochastic algorithms. On the other hand, when the domain is bounded, such as in matrix games, then the gap functions are well-defined optimality measures.

To compare with some relevant works, we will prove rates for the gap function, which will be a direct consequence of the rates we prove on the natural residual/gradient mapping norm, given in (1.3).

Related works. In the deterministic case, optimal last iterate guarantees are obtained recently by anytime algorithms [YR21, CZ24, CZ23] by using anchoring; and suboptimal guarantees were shown in [GPDO20, GTG22] for standard VI algorithms. In the game setting, last iterate guarantees are proven in [GPD20] in the deterministic case. In the stochastic case, last iterate guarantees without using mini-batches or multi-loop algorithms has been of interest for learning in games, and many works focused on this direction, including [CM24, AIMM21, ASAI25, $\mathrm { S Y L J ^ { + } 2 6 }$ , ZLS26, ITAA26, AASI24, HACM22].

The literature on the last iterate guarantees for anytime, single-loop algorithms using a single sample (or 2 samples) at every iteration had recent interesting developments. In particular, for unconstrained problems $[ \mathrm { S Y L J ^ { + } 2 6 } ]$ showed the $O ( t ^ { - 1 / 4 } )$ rate for an anchored algorithm. For constrained problems, two parallel works showed the anytime rate $O ( t ^ { - 1 / 5 } )$ for the restricted gap function [ZLS26, ITAA26]. Surprisingly, both of these works concluded the following separation: when we let go of the anytime requirement, that is, when the horizon T is known in advance, it was possible to get the rate $O ( T ^ { - 1 / 4 } )$ for the last iterate where T is the horizon; yet for the anytime case, both works got the worse $O ( t ^ { - 1 / 5 } )$ rate.

Other single-loop algorithms in the literature are not compatible with the requirements of our setting: for example the developments in [CSGD22] focused on unconstrained problems (for monotone and Lipschitz operators) and used variance reduction along with increasing batches; the works $[ \mathrm { P F L ^ { + } 2 3 }$ , AK26, AMW25] relied on variance reduction and multi-point oracles and did not prove last iterate guarantees. Classical works obtain optimal rates on the gap when it is evaluated at the averaged iterate, see for example [NJLS09]

A large body of literature used increasing mini-batch sizes to obtain $O ( \varepsilon ^ { - 4 } )$ complexity [IJOT17, PXC23, KLL22, LK21]. Our complexity matches this by avoiding mini-batches. When we allow increasing minibatches or multi-loop algorithms, the complexity for residual-type optimality measures was improved to $O ( \varepsilon ^ { - 1 0 / 3 } )$ by using mini-batches and variance reduction in the work [TDNT26], where a stronger stochastic oracle was used due to variance reduction. Only very recently, methods based on multiple loops obtained the near-optimal $\widetilde { O } ( \varepsilon ^ { - 2 } )$ complexity for gradient norm for unconstrained problems [CL24] and natural residual and tangent residual in the constrained case [Ala26, JLZ26]. These methods are not suitable for learning in games setting since they rely on multiple loop algorithms, and are not anytime.

Context and Contributions. The best anytime rate for the restricted gap stayed at $O \left( t ^ { - 1 / 5 } \right)$ [ZLS26, ITAA26], whereas both works could obtain the horizon-dependent $O ( T ^ { - 1 / 4 } )$ rate. At the same time, an anytime $O ( t ^ { - 1 / 4 } )$ was known for unconstrained problems $[ \mathrm { S Y L J ^ { + } 2 6 , Z L S 2 6 } ]$ . The dificulty for extending to the constrained case was discussed in [ZLS26, Remark 4.2]; open questions to derive an anytime $O ( t ^ { - 1 / 4 } )$ rate for constrained problems were mentioned in $[ \mathrm { S Y L J ^ { + } 2 6 } ]$ , Section 7] and [ZLS26, Section 6].

In this work, we prove that the separation from [ZLS26, ITAA26] can be avoided, that is, we prove the anytime rate $O \left( t ^ { - 1 \bar { / } 4 } \right)$ for constrained problems. To allow unbounded constraint sets, we show rates on gradient mapping norm (in addition to the gap function), without the bounded variance assumption, by allowing variance to grow as fast as the displacement of the iterates from an initial point. Our algorithm is a stochastic gradient-type algorithm with Halpern anchoring, using a single-call of the unbiased operator at every iteration.

More technical details are provided in Section 3.2.

## 1.1 Assumptions

Monotonicity and Lipschitz continuity of G mean that, for all $\mathbf { x } , \mathbf { y } .$ , we have

$$
\left. G ( \mathbf { x } ) - G ( \mathbf { y } ) \mathbf { x } - \mathbf { y } \right. \geq 0 , \qquad \left\| G ( \mathbf { x } ) - G ( \mathbf { y } ) \right\| \leq L \| \mathbf { x } - \mathbf { y } \| .
$$

We now collect the assumptions made on our main problem.

Assumption 1.1. The function $r \colon \mathbb { R } ^ { d }  \mathbb { R } \cup \{ + \infty \}$ is proper, closed, and convex. The operator $G : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ is monotone and L-Lipschitz, and the set of solutions to (1.1) is nonempty.

We fix a deterministic initial point ${ \bf z } _ { 0 } \in  { }$ dom r. Let $\mathcal { F } _ { t }$ denote the history before the oracle call at iteration $t ,$ where $\mathbf { z } _ { t }$ is $\mathcal { F } _ { t } .$ -measurable, and we write $\mathbb { E } _ { t } [ \cdot ] = \mathbb { E } [ \cdot \mid \mathcal { F } _ { t } ]$ . The oracle returns an unbiased sample whose variance may grow with the distance from the initial point.

Assumption 1.2. At each iteration $t \geq 0$ , the sample $\widetilde { G } ( \mathbf { z } _ { t } , \xi _ { t } )$ satisfies

$$
\begin{array} { c } { \mathbb { E } _ { t } [ \widetilde { G } ( \mathbf { z } _ { t } , \boldsymbol { \xi } _ { t } ) ] = G ( \mathbf { z } _ { t } ) , } \\ { \mathbb { E } _ { t } \| \widetilde { G } ( \mathbf { z } _ { t } , \boldsymbol { \xi } _ { t } ) - G ( \mathbf { z } _ { t } ) \| ^ { 2 } \leq B ^ { 2 } \| \mathbf { z } _ { t } - \mathbf { z } _ { 0 } \| ^ { 2 } + \sigma ^ { 2 } . } \end{array}\tag{1.5}
$$

The case $B = 0$ recovers the bounded-variance assumption. Let us compare this to the assumption used in $[ \mathrm { S Y L J ^ { + } 2 6 } ]$ who required in the unconstrained case

$$
\begin{array} { r } { \mathbb { E } \| \widetilde { G } ( \mathbf { z } _ { t } , \xi _ { t } ) - G ( \mathbf { z } _ { t } ) \| ^ { 2 } \leq \sigma ^ { 2 } + c ^ { 2 } \| G ( \mathbf { z } _ { t } ) \| ^ { 2 } . } \end{array}\tag{1.6}
$$

To see that our assumption is weaker, we focus on the unconstrained case where $G ( { \mathbf { z } ^ { \star } } ) = 0$ . Then, this assumption implies by Lipschitzness of $G \mathrm { : }$

$$
\| G ( \mathbf { z } _ { t } ) \| ^ { 2 } = \| G ( \mathbf { z } _ { t } ) - G ( \mathbf { z } ^ { \star } ) \| ^ { 2 } \leq L ^ { 2 } \| \mathbf { z } _ { t } - \mathbf { z } ^ { \star } \| ^ { 2 } \leq 2 L ^ { 2 } \left( \| \mathbf { z } _ { t } - \mathbf { z } _ { 0 } \| ^ { 2 } + \| \mathbf { z } _ { 0 } - \mathbf { z } ^ { \star } \| ^ { 2 } \right) .
$$

That is, the variance assumption of $[ \mathrm { S Y L J ^ { + } 2 6 } ]$ implies Assumption 1.2.

Notation. When clear from the context, we shorten $\widetilde { G } ( \mathbf { z } _ { t } , \xi _ { t } )$ as $\widetilde { G } ( \mathbf { z } _ { t } )$ . For $\eta > 0$ , we define the proximal operator as

$$
\mathrm { p r o x } _ { \eta r } ( \mathbf { x } ) : = \underset { \mathbf { z } \in \mathbb { R } ^ { d } } { \mathrm { a r g m i n } } \left. r ( \mathbf { z } ) + \frac { 1 } { 2 \eta } \| \mathbf { z } - \mathbf { x } \| ^ { 2 } \right. ,
$$

which is well-defined, single-valued, and firmly nonexpansive under our assumptions on r. The optimality condition gives the well known prox-inequality:

$$
\begin{array} { r l } { \mathbf { z } = \mathrm { p r o x } _ { \eta r } ( \mathbf { x } ) \quad \Longleftrightarrow \quad \mathbf { x } - \mathbf { z } \in \eta \partial r ( \mathbf { z } ) \iff \langle \mathbf { z } - \mathbf { x } , \mathbf { u } - \mathbf { z } \rangle \geq \eta ( r ( \mathbf { z } ) - r ( \mathbf { u } ) ) \quad \forall \mathbf { u } \in \mathbb { R } ^ { d } . } \end{array}\tag{1.7}
$$

## 2 Algorithm and the Main Rate Results

Algorithm 1 is simple extension of stochastic gradient descent (or stochastic forward-backward method), when Halpern anchoring is introduced [Hal67]. This algorithm appeared many times in the literature. It was referred to as the regularized gradient method in [ITAA26] and Composite Objective Gradient Descent-Ascent in [NO24]. It was also analyzed by [LK21] with increasing mini-batch sizes. This algorithm is a single-call algorithm in view of [HIMM19], uses no mini-batch, variance reduction or inner loops.

Algorithm 1 A single-call stochastic Halpern method   
Require: Initial point ${ \bf z } _ { 0 } \in  { }$ dom $r , a \geq 2 ,$ , and $H ^ { 2 } = L ^ { 2 } + 2 B ^ { 2 }$ . Set $\begin{array} { r } { \beta _ { t } = \frac { a } { t + a } } \end{array}$ and $\begin{array} { r } { \eta _ { t } = \frac { 1 } { H ( t + a ) ^ { 3 / 4 } } . } \end{array}$   
1: for $t = 0 , 1 , \ldots$ do   
2: Obtain an unbiased estimate $\widetilde { G } ( \mathbf { z } _ { t } )$ of $G ( \mathbf { z } _ { t } )$   
3: $\mathbf { z } _ { t + 1 } = \mathrm { p r o x } _ { \eta _ { t } r } \left( ( 1 - \beta _ { t } ) \mathbf { z } _ { t } + \beta _ { t } \mathbf { z } _ { 0 } - \eta _ { t } \widetilde { G } ( \mathbf { z } _ { t } ) \right)$   
4: end for

We now present our main result which is an anytime convergence rate on the gradient mapping, or natural residual, norm. Then, we continue with its implications to anytime rates on gap functions.

Theorem 2.1. Let Assumptions 1.1 and 1.2 hold. Then, for every $t \geq 1$ , Algorithm 1 satisfies

$$
\mathbb { E } \| \mathcal { G } _ { 1 / L } ( \mathbf { z } _ { t } ) \| \leq \sqrt { \mathbb { E } \| \mathcal { G } _ { 1 / L } ( \mathbf { z } _ { t } ) \| ^ { 2 } } = O \left( \frac { ( L + B ) \| \mathbf { z } _ { 0 } - \mathbf { z } ^ { \star } \| + \sigma } { t ^ { 1 / 4 } } \right) .
$$

Corollary 2.2. Let $r = \delta _ { Z }$ for a convex, closed set Z. We then have

$$
\mathbb { E } [ \mathrm { g a p } _ { \mathcal { U } } ( \mathbf { z } _ { t } ) ] = O \left( \frac { 1 } { t ^ { 1 / 4 } } \right) ,
$$

for any compact set U, where $\mathrm { g a p } _ { \mathcal { U } }$ extends (1.4) by taking the maximum over a compact set U.

Corollary 2.3. Under the setup of Corollary 2.2, suppose that $\mathbf { z } = ( \mathbf { x } , \mathbf { y } ) , \mathscr { U } = \mathscr { U } _ { \mathbf { x } } \times \mathscr { U } _ { \mathbf { y } }$ , and

$$
G ( \mathbf { z } ) = \left( { \begin{array} { c } { \nabla _ { \mathbf { x } } f ( \mathbf { x } , \mathbf { y } ) } \\ { - \nabla _ { \mathbf { y } } f ( \mathbf { x } , \mathbf { y } ) } \end{array} } \right) ,
$$

where f is convex in x and concave in y. Then, we have

$$
\mathbb { E } \left[ \operatorname* { m a x } _ { \mathbf { y } ^ { \prime } \in \mathcal { U } _ { \mathbf { y } } } f ( \mathbf { x } _ { t } , \mathbf { y } ^ { \prime } ) - \operatorname* { m i n } _ { \mathbf { x } ^ { \prime } \in \mathcal { U } _ { \mathbf { x } } } f ( \mathbf { x } ^ { \prime } , \mathbf { y } _ { t } ) \right] = O \left( \frac { 1 } { t ^ { 1 / 4 } } \right) .
$$

Finally, we convert the convergence rate guarantees to iteration and sample complexity results.

Corollary 2.4. Under the same setup as Theorem 2.1, to obtain $\mathbb { E } \| \mathcal { G } _ { \rho } ( \mathbf { z } _ { t } ) \| \leq \varepsilon \ o r \mathbb { E } \operatorname { g a p } _ { \mathcal { U } } ( \mathbf { z } _ { t } ) \leq \varepsilon$ on the last iterate $\mathbf { z } _ { t } .$ , the required total number of stochastic oracles is of the order $O ( \varepsilon ^ { - 4 } )$ .

The proof of Corollaries 2.2 and 2.3 are given in Section 5, and the proof of Corollary 2.4 is omitted since it is immediate.

We continue with the proof of Theorem 2.1. The structure of our proof is inspired by the simple and powerful idea that first appeared in the work $[ \mathrm { S Y L J ^ { + } 2 6 } ]$ and then used also for the unconstrained result of [ZLS26]. The idea is to first analyze the deterministic shadow sequence given in (3.1) and then provide a pointwise bound between the deterministic sequence and the original stochastic sequence.

Instead of the gradient norm used in these works, we use the gradient mapping, which satisfies the necessary Lipschitz properties and then we show that the gradient mapping norm majorizes the gap function, to go around the dificulty of working directly with the gap function.

Proof of Theorem 2.1. Recall that $\rho = 1 / L$ . By adding and subtracting $\mathcal { G } _ { \rho } ( \bar { \mathbf { z } } _ { t } )$ and applying Young’s inequality, we obtain

$$
\begin{array} { r l } & { \mathbb { E } \| \mathcal { G } _ { \rho } ( \mathbf { z } _ { t } ) \| ^ { 2 } \leq 2 \| \mathcal { G } _ { \rho } ( \bar { \mathbf { z } } _ { t } ) \| ^ { 2 } + 2 \mathbb { E } \| \mathcal { G } _ { \rho } ( \mathbf { z } _ { T } ) - \mathcal { G } _ { \rho } ( \bar { \mathbf { z } } _ { t } ) \| ^ { 2 } } \\ & { \qquad \leq 2 \| \mathcal { G } _ { \rho } ( \bar { \mathbf { z } } _ { t } ) \| ^ { 2 } + 2 ( \rho ^ { - 2 } + L ^ { 2 } ) \mathbb { E } \| \mathbf { z } _ { t } - \bar { \mathbf { z } } _ { t } \| ^ { 2 } , } \end{array}\tag{2.1}
$$

where the last line is by (4.6). We bound the first term on the right-hand side. By Lemmas 3.2 and 4.3, we get

$$
\| \mathcal { G } _ { \rho } ( \bar { \mathbf { z } } _ { t } ) \| \leq \frac { 5 a H \| \mathbf { z } _ { 0 } - \mathbf { z } ^ { \star } \| } { 2 ( t - 1 + a ) ^ { 1 / 4 } } \left( \frac { 3 } { 2 } + \frac { L } { 2 H ( t - 1 + a ) ^ { 3 / 4 } } \right) .
$$

Taking the square of both sides and multiplying by 2 give us

$$
\begin{array} { r l r } {  { 2 \| \mathcal { G } _ { \rho } ( \bar { \mathbf { z } } _ { t } ) \| ^ { 2 } \leq \frac { 2 5 a ^ { 2 } H ^ { 2 } \| \mathbf { z } _ { 0 } - \mathbf { z } ^ { \star } \| ^ { 2 } } { 2 \sqrt { t - 1 + a } } ( \frac { 3 } { 2 } + \frac { L } { 2 H ( t - 1 + a ) ^ { 3 / 4 } } ) ^ { 2 } } } \\ & { } & { \leq \frac { 2 5 a ^ { 2 } H ^ { 2 } \| \mathbf { z } _ { 0 } - \mathbf { z } ^ { \star } \| ^ { 2 } } { 2 \sqrt { t - 1 + a } } ( \frac { 3 } { 2 } + \frac { L } { 2 H a ^ { 3 / 4 } } ) ^ { 2 } . } \end{array}
$$

The last inequality uses $t - 1 + a \geq a .$

For the second term on the right-hand side of (2.1), by Lemma 3.3, we have

$$
2 ( \rho ^ { - 2 } + L ^ { 2 } ) \mathbb { E } \| \mathbf { z } _ { t } - \bar { \mathbf { z } } _ { t } \| ^ { 2 } \leq \frac { 2 ( \rho ^ { - 2 } + L ^ { 2 } ) } { H ^ { 2 } ( a - \frac 1 2 ) \sqrt { t - 1 + a } } \left( \sigma ^ { 2 } + \frac { 2 5 } { 2 } B ^ { 2 } \| \mathbf { z } _ { 0 } - \mathbf { z } ^ { \star } \| ^ { 2 } \right) ,
$$

where we used $t + a \geq t - 1 + a$

Combining the last two estimates in (2.1) yields

$$
\begin{array} { r } { \displaystyle \mathbb { E } \| \mathcal { G } _ { \rho } ( \mathbf { z } _ { T } ) \| ^ { 2 } \leq \frac { 1 } { \sqrt { T - 1 + a } } \left[ \frac { 2 5 a ^ { 2 } H ^ { 2 } \| \mathbf { z } _ { 0 } - \mathbf { z } ^ { \star } \| ^ { 2 } } { 8 } \left( 3 + \frac { L } { H a ^ { 3 / 4 } } \right) ^ { 2 } \right. } \\ { \displaystyle \left. + \frac { 2 ( \rho ^ { - 2 } + L ^ { 2 } ) } { H ^ { 2 } ( a - \frac { 1 } { 2 } ) } \left( \sigma ^ { 2 } + \frac { 2 5 } { 2 } B ^ { 2 } \| \mathbf { z } _ { 0 } - \mathbf { z } ^ { \star } \| ^ { 2 } \right) \right] . } \end{array}
$$

Using $H \geq L$ and $a \geq 2 .$ , absorbing numerical constants and Jensen’s inequality, that is,

$$
\begin{array} { r } { \mathbb { E } \| \mathcal { G } _ { \rho } ( \mathbf { z } _ { T } ) \| \leq \left( \mathbb { E } \| \mathcal { G } _ { \rho } ( \mathbf { z } _ { T } ) \| ^ { 2 } \right) ^ { 1 / 2 } , } \end{array}
$$

provides the assertion.

## 2.1 Application for Learning in Games

We now apply our anytime guarantee for monotone games [CBL06]. Following the setup in [CZ23, ITAA26], we have N players where i-th player is denoted as $z _ { i } ,$ where $\mathbf { z } \overset { \cdot } { = } ( z ^ { i } ; z ^ { - i } ) = ( z ^ { 1 } , \ldots , z ^ { \bar { N } } )$ . Each player

selects actions from sets $Z ^ { i }$ which are assumed to be compact. Each player minimizes a convex loss function $\ell _ { i } ( z ^ { i } ; x ^ { - i } )$ . The operator G will be defined as

$$
G ( { \bf z } ) = \left[ \begin{array} { c } { { \nabla _ { x _ { 1 } } \ell _ { 1 } ( { \bf z } ) } } \\ { { \vdots } } \\ { { \nabla _ { x _ { N } } \ell _ { N } ( { \bf z } ) } } \end{array} \right] .
$$

We assume access to noisy estimates of $G ( \mathbf { z } )$ such that $\widetilde { G } ( \mathbf { z } _ { t } ) = G ( \mathbf { z } _ { t } ) + \zeta _ { t }$ where $\mathbb { E } [ \zeta _ { t } ] = 0$ and $\mathbb { E } \| \zeta _ { t } \| ^ { 2 } \leq$ $B ^ { 2 } \| \mathbf { z } _ { t } - \mathbf { z } _ { 0 } \| ^ { 2 } + \sigma ^ { 2 }$ . Even though this allows handling problems without bounded feasible sets, for this section, we assume that Z is compact.

For direct comparison with [ITAA26], we define the constants

$$
D = \operatorname* { m a x } _ { \mathbf { x } , \mathbf { y } \in C } \| \mathbf { x } - \mathbf { y } \| , \qquad U = \operatorname* { m a x } _ { \mathbf { z } \in C } \| G ( \mathbf { z } ) \| .
$$

Set $a = 2$ and define, for $\mathbf { u \in }$ dom $r _ { \ast }$

$$
\mathrm { R e g } _ { T } ( \mathbf { u } ) = \sum _ { t = 0 } ^ { T - 1 } \langle G ( \mathbf { z } _ { t } ) , \mathbf { z } _ { t } - \mathbf { u } \rangle .
$$

A direct consequence of Theorem 2.1, Corollary 2.2, and [ITAA26, Section 7] (by selecting parameters as this section) is that we obtain the convergence and regret guarantees

$$
\mathbb { E } \operatorname { g a p } ( \mathbf { z } _ { t } ) = O \left( \frac { D \big ( L D + U + \sigma \big ) } { t ^ { 1 / 4 } } \right) \quad \mathrm { a n d } \quad \mathbb { E } \operatorname { R e g } _ { T } ( \mathbf { u } ) = O \left( D \big ( L D + U + \sigma \big ) T ^ { 3 / 4 } \right) .\tag{2.2}
$$

Remark 2.5. Under the same bounded-domain and bounded-variance setting, [ITAA26, Theorem $5 . 1 ]$ obtained the gap bound $O ( D ( L D + U + \sigma ) t ^ { - 1 / 5 } )$ . Thus our result improves the anytime rate from $O ( t ^ { - 1 / 5 } )$ to $O ( t ^ { - 1 / 4 } )$ Our analysis also improves the regret guarantee obtained for the anytime algorithm in [ITAA26] which was $O ( T ^ { 4 / 5 } )$ to ${ \cal O } ( T ^ { 3 { \bar { / } } 4 } )$ . Of course, the regret guarantee is suboptimal, however, the aim is to prove that Algorithm 1 has sublinear regret, while improving the last iterate anytime guarantee for approaching a Nash equilibrium.

## 3 Convergence Analysis

We now continue with the details of the analysis, following the high-level description given in Section 2.

## 3.1 Deterministic Sequence

Starting from $\bar { \bf z } _ { 0 } : = { \bf z } _ { 0 }$ , define the deterministic reference iterates by

$$
\bar { \mathbf { z } } _ { t + 1 } = \mathrm { p r o x } _ { \eta _ { t } r } \left( ( 1 - \beta _ { t } ) \bar { \mathbf { z } } _ { t } + \beta _ { t } \mathbf { z } _ { 0 } - \eta _ { t } G ( \bar { \mathbf { z } } _ { t } ) \right) .\tag{3.1}
$$

These iterates are used only in the analysis. This sequence is indeed the algorithm analyzed by [CZ26], referred to as the anchored gradient descent. We follow the arguments of [CZ26], by adapting the parameter choices. Their $t ^ { - 1 / 2 }$ stepsize is replaced here by $t ^ { - 3 / 4 }$ ; the estimates below require only $H \geq L$

The first result proves the uniform boundedness of this deterministic sequence.

Lemma 3.1. Under Assumption $1 . 1 ,$ for the iterates generated by (3.1), we have for any $t \geq 0$ that

$$
\| \bar { \mathbf { z } } _ { t } - \mathbf { z } ^ { \star } \| \leq \frac { 3 } { 2 } \| \mathbf { z } _ { 0 } - \mathbf { z } ^ { \star } \| , \qquad \| \bar { \mathbf { z } } _ { t } - \mathbf { z } _ { 0 } \| \leq \frac { 5 } { 2 } \| \mathbf { z } _ { 0 } - \mathbf { z } ^ { \star } \| .\tag{3.2}
$$

Proof. By the definition of the solution, we have $- G ( \mathbf { z } ^ { \star } ) \in \partial r ( \mathbf { z } ^ { \star } )$ and $\begin{array} { r } { \mathbf { z } ^ { \star } = \mathrm { p r o x } _ { \eta _ { t } r } \big ( \mathbf { z } ^ { \star } - \eta _ { t } G ( \mathbf { z } ^ { \star } ) \big ) } \end{array}$ ). With this, nonexpansiveness of the proximal operator, and triangle inequality, we obtain

$$
\begin{array} { r l } & { \| \bar { \mathbf z } _ { t + 1 } - \mathbf z ^ { \star } \| \leq \| ( 1 - \beta _ { t } ) \bar { \mathbf z } _ { t } + \beta _ { t } \mathbf z _ { 0 } - \mathbf z ^ { \star } - \eta _ { t } G ( \bar { \mathbf z } _ { t } ) + \eta _ { t } G ( \mathbf z ^ { \star } ) \| } \\ & { \qquad \leq \| ( 1 - \beta _ { t } ) ( \bar { \mathbf z } _ { t } - \mathbf z ^ { \star } ) - \eta _ { t } ( G ( \bar { \mathbf z } _ { t } ) - G ( \mathbf z ^ { \star } ) ) \| + \beta _ { t } \| \mathbf z _ { 0 } - \mathbf z ^ { \star } \| . } \end{array}\tag{3.3}
$$

For the first norm on the right-hand side, we use Lemma 4.1 with ${ \bf u } = \bar { \bf z } _ { t } , { \bf v } = { \bf z } ^ { \star } , \alpha = ( 1 - \beta _ { t } ) , \gamma = \eta _ { t }$ to get

$$
\left\| \bar { \mathbf z } _ { t + 1 } - \mathbf z ^ { \star } \right\| \leq \sqrt { ( 1 - \beta _ { t } ) ^ { 2 } + \eta _ { t } ^ { 2 } L ^ { 2 } \left\| \bar { \mathbf z } _ { t } - \mathbf z ^ { \star } \right\| + \beta _ { t } \| \mathbf z _ { 0 } - \mathbf z ^ { \star } \| } .\tag{3.4}
$$

We first verify the induction base case $t = 1$ . At the first step, $\bar { \bf z } _ { 0 } = { \bf z } _ { 0 } , \beta _ { 0 } = 1$ and $\eta _ { 0 } = 1 / ( H a ^ { 3 / 4 } )$ . By definition of $\bar { \bf z } _ { 1 }$ and Lemma 4.1 with ${ \bf u } = { \bf z } _ { 0 } , { \bf v } = { \bf z } ^ { \star } , \alpha = 1 , \gamma = \eta _ { 0 }$ , we have

$$
\| \overline { { \mathbf { z } } } _ { 1 } - \mathbf { z } ^ { \star } \| ^ { 2 } \leq \| \mathbf { z } _ { 0 } - \mathbf { z } ^ { \star } - \eta _ { 0 } ( G ( \mathbf { z } _ { 0 } ) - G ( \mathbf { z } ^ { \star } ) ) \leq ( 1 + \eta _ { 0 } ^ { 2 } L ^ { 2 } ) \| \mathbf { z } _ { 0 } - \mathbf { z } ^ { \star } \| ^ { 2 } \leq \frac { 9 } { 4 } \| \mathbf { z } _ { 0 } - \mathbf { z } ^ { \star } \| ^ { 2 } .
$$

Indeed, $\eta _ { 0 } ^ { 2 } L ^ { 2 } = L ^ { 2 } / ( H ^ { 2 } a ^ { 3 / 2 } ) \leq a ^ { - 3 / 2 } \leq 5 / 4$ since $H \geq L$ , and $a \ge 2$

We now prove the first bound in (3.2) by induction. The case $t = 0$ follows from ${ \bar { \bf z } } _ { 0 } = { \bf z } _ { 0 } ,$ , and the calculation above proves the case $t = 1$ . Suppose that, for some $\begin{array} { r } { t \geq 1 , \| \bar { \mathbf z } _ { t } - \mathbf z ^ { \star } \| \leq \frac { 3 } { 2 } \| \mathbf z _ { 0 } - \mathbf z ^ { \star } \| } \end{array}$ . For $t \geq 1$ Fact 4.4 gives $\begin{array} { r } { \sqrt { ( 1 - \beta _ { t } ) ^ { 2 } + \eta _ { t } ^ { 2 } L ^ { 2 } } \le 1 - \frac { 2 } { 3 } \beta _ { t } } \end{array}$ . Using this and the inductive assumption in (3.4), we obtain

$$
\begin{array} { r l } { \displaystyle \| \bar { \mathbf z } _ { t + 1 } - { \mathbf z } ^ { \star } \| \leq \left( 1 - \frac 2 3 \beta _ { t } \right) \| \bar { \mathbf z } _ { t } - { \mathbf z } ^ { \star } \| + \beta _ { t } \| { \mathbf z } _ { 0 } - { \mathbf z } ^ { \star } \| } & { } \\ { \displaystyle \leq \left( 1 - \frac 2 3 \beta _ { t } \right) \frac 3 2 \| { \mathbf z } _ { 0 } - { \mathbf z } ^ { \star } \| + \beta _ { t } \| { \mathbf z } _ { 0 } - { \mathbf z } ^ { \star } \| } & { } \\ { \displaystyle } & { = \frac 3 2 \| { \mathbf z } _ { 0 } - { \mathbf z } ^ { \star } \| . } \end{array}
$$

This completes the induction and proves the first bound in (3.2). Finally, using the triangle inequality and the first bound in (3.2) gives the second bound in (3.2). ■

Let us remark that using the ideas from [CZ26], we get a guarantee on the optimality measure $\| G ( \mathbf { z } ) + \mathbf { v } \|$ where $\mathbf { v } \in \partial r ( \mathbf { z } )$ . However, we later use this bound to transfer the guarantee to the gradient mapping norm since the measure above does not admit the Lipschitzness properties required for the analysis.

Let us remark that this is definitely an unsurprising result since it is now classical. Indeed, starting from the work of [YR21] which is extended to the constrained case by [KG22, CZ24, COZ24], it is known how to prove that the residual has the last iterate rate of $O ( 1 / k )$ (with more sophisticated algorithms). In our case, since we are limited by the pointwise bounds in Section 3.2, we seek a simple bound for this sequence that will be just suficient for our purposes with the parameter choices for $\beta _ { t } , \eta _ { t }$

Lemma 3.2. Let Assumption 1.1 hold. For every $t \geq 0$ , we have

$$
\| \bar { \mathbf { z } } _ { t + 1 } - \bar { \mathbf { z } } _ { t } \| \leq \frac { 5 a \| \mathbf { z } _ { 0 } - \mathbf { z } ^ { \star } \| } { 4 ( t + a ) } .\tag{3.5}
$$

Moreover, for every $t \geq 1$ , we find $\mathbf { v } _ { t } \in \partial r ( \bar { \mathbf { z } } _ { t } )$ such that

$$
\| G ( \bar { \mathbf { z } } _ { t } ) + \mathbf { v } _ { t } \| \leq \frac { 5 a H \| \mathbf { z } _ { 0 } - \mathbf { z } ^ { \star } \| } { 2 ( t - 1 + a ) ^ { 1 / 4 } } \left( \frac { 3 } { 2 } + \frac { L } { 2 H ( t - 1 + a ) ^ { 3 / 4 } } \right) .\tag{3.6}
$$

Proof. By the definition in (3.1) and the definition of the proximal operator gives

$$
\mathbf { v } _ { t + 1 } : = \frac { ( 1 - \beta _ { t } ) \bar { \mathbf { z } } _ { t } + \beta _ { t } \mathbf { z } _ { 0 } - \eta _ { t } G ( \bar { \mathbf { z } } _ { t } ) - \bar { \mathbf { z } } _ { t + 1 } } { \eta _ { t } } \in \partial r ( \bar { \mathbf { z } } _ { t + 1 } ) .\tag{3.7}
$$

In particular, $\begin{array} { r } { \bar { \mathbf { z } } _ { t + 1 } = \mathrm { p r o x } _ { \eta _ { t + 1 } r } ( \bar { \mathbf { z } } _ { t + 1 } + \eta _ { t + 1 } \mathbf { v } _ { t + 1 } ) \iff \bar { \mathbf { z } } _ { t + 1 } + \eta _ { t + 1 } \partial r ( \bar { \mathbf { z } } _ { t + 1 } ) \ni \bar { \mathbf { z } } _ { t + 1 } + \eta _ { t + 1 } \mathbf { v } _ { t + 1 } } \end{array}$ . We compare this expression with the update $\bar { \mathbf { z } } _ { t + 2 } = \mathrm { p r o x } _ { \eta _ { t + 1 } r } \big ( ( 1 - \beta _ { t + 1 } ) \bar { \mathbf { z } } _ { t + 1 } + \beta _ { t + 1 } \mathbf { z } _ { 0 } - \eta _ { t + 1 } G ( \bar { \mathbf { z } } _ { t + 1 } ) \big )$ . By Fact 4.4, we have $\eta _ { t + 1 } / \eta _ { t } - \beta _ { t + 1 } \geq 0$ . Nonexpansiveness of the proximal operator and substitution of (3.7) give the first inequality below. Collecting the coeficients of $\bar { \mathbf { z } } _ { t + 1 } , \bar { \mathbf { z } } _ { t }$ , and $\mathbf { z } _ { 0 }$ gives the equality, while the last inequality

follows from the triangle inequality.

$$
\begin{array} { r l } & { \left\| \bar { \mathbf z } _ { t + 2 } - \bar { \mathbf z } _ { t + 1 } \right\| \leq \left\| - \beta _ { t + 1 } \bar { \mathbf z } _ { t + 1 } + \beta _ { t + 1 } \mathbf z _ { 0 } - \eta _ { t + 1 } G ( \bar { \mathbf z } _ { t + 1 } ) - \frac { \eta _ { t + 1 } } { \eta _ { t } } \big ( ( 1 - \beta _ { t } ) \bar { \mathbf z } _ { t } + \beta _ { t } \bar { \mathbf z } _ { 0 } - \eta _ { t } G ( \bar { \mathbf z } _ { t } ) - \bar { \mathbf z } _ { t + 1 } \big ) \right\| } \\ & { \qquad = \left\| \left( \frac { \eta _ { t + 1 } } { \eta _ { t } } - \beta _ { t + 1 } \right) ( \bar { \mathbf z } _ { t + 1 } - \bar { \mathbf z } _ { t } ) - \eta _ { t + 1 } \big ( G ( \bar { \mathbf z } _ { t + 1 } ) - G ( \bar { \mathbf z } _ { t } ) \big ) + \left( \beta _ { t + 1 } - \frac { \eta _ { t + 1 } } { \eta _ { t } } \beta _ { t } \right) ( \bar { \mathbf z } _ { 0 } - \bar { \mathbf z } _ { t } ) \right\| } \\ & { \qquad \leq \left\| \left( \frac { \eta _ { t + 1 } } { \eta _ { t } } - \beta _ { t + 1 } \right) ( \bar { \mathbf z } _ { t + 1 } - \bar { \mathbf z } _ { t } ) - \eta _ { t + 1 } \big ( G ( \bar { \mathbf z } _ { t + 1 } ) - G ( \bar { \mathbf z } _ { t } ) \big ) \right\| + \left| \beta _ { t + 1 } - \frac { \eta _ { t + 1 } } { \eta _ { t } } \beta _ { t } \right| \left\| \bar { \mathbf z } _ { 0 } - \bar { \mathbf z } _ { t } \right\| . } \end{array}\tag{3.8}
$$

We next estimate the first norm on the right-hand side. Using Lemma 4.1 with $\mathbf { u } \ = \ \bar { \mathbf { z } } _ { t + 1 } , \ \mathbf { v } \ = \ \bar { \mathbf { z } } _ { t }$ $\begin{array} { r } { \alpha = \frac { \eta _ { t + 1 } } { \eta _ { t } } - \beta _ { t + 1 } , \gamma = \eta _ { t + 1 } } \end{array}$ yields

$$
\begin{array} { r } { \bigg \| \left( \frac { \eta _ { t + 1 } } { \eta _ { t } } - \beta _ { t + 1 } \right) ( \bar { \mathbf z } _ { t + 1 } - \bar { \mathbf z } _ { t } ) - \eta _ { t + 1 } \big ( G ( \bar { \mathbf z } _ { t + 1 } ) - G ( \bar { \mathbf z } _ { t } ) \big ) \bigg \| ^ { 2 } \leq \left[ \left( \frac { \eta _ { t + 1 } } { \eta _ { t } } - \beta _ { t + 1 } \right) ^ { 2 } + \eta _ { t + 1 } ^ { 2 } L ^ { 2 } \right] \| \bar { \mathbf z } _ { t + 1 } - \bar { \mathbf z } _ { t } \| ^ { 2 } . } \end{array}
$$

Taking square roots, substituting this estimate into (3.8), and using Lemma 3.1 gives

$$
\| \bar { \mathbf { z } } _ { t + 2 } - \bar { \mathbf { z } } _ { t + 1 } \| \leq \sqrt { \left( \frac { \eta _ { t + 1 } } { \eta _ { t } } - \beta _ { t + 1 } \right) ^ { 2 } + \eta _ { t + 1 } ^ { 2 } L ^ { 2 } \left\| \bar { \mathbf { z } } _ { t + 1 } - \bar { \mathbf { z } } _ { t } \right\| + \frac { 5 } { 2 } \left| \beta _ { t + 1 } - \frac { \eta _ { t + 1 } } { \eta _ { t } } \beta _ { t } \right| \left\| \mathbf { z } _ { 0 } - \mathbf { z } ^ { \star } \right\| } .\tag{3.9}
$$

By Fact 4.4, we have

$$
\left( \frac { \eta _ { t + 1 } } { \eta _ { t } } - \beta _ { t + 1 } \right) ^ { 2 } + \eta _ { t + 1 } ^ { 2 } L ^ { 2 } \leq \left( 1 - \frac { 3 } { 2 s } \right) ^ { 2 } , \quad \mathrm { a n d } \quad \left| \beta _ { t + 1 } - \frac { \eta _ { t + 1 } } { \eta _ { t } } \beta _ { t } \right| \leq \frac { a } { 4 ( t + a ) ( t + a + 1 ) } .
$$

Substituting these bounds into (3.9) gives

$$
\| \bar { \mathbf { z } } _ { t + 2 } - \bar { \mathbf { z } } _ { t + 1 } \| \leq \left( 1 - \frac { 3 } { 2 ( t + a + 1 ) } \right) \| \bar { \mathbf { z } } _ { t + 1 } - \bar { \mathbf { z } } _ { t } \| + \frac { 5 a \| \mathbf { z } _ { 0 } - \mathbf { z } ^ { \star } \| } { 8 ( t + a ) ( t + a + 1 ) } .\tag{3.10}
$$

For $t = 0$ , since $\bar { \mathbf { z } } _ { 1 } = \mathrm { p r o x } _ { m r } \big ( \mathbf { z } _ { 0 } - \eta _ { 0 } G ( \mathbf { z } _ { 0 } ) \big )$ , the definition of the proximal operator gives

$$
\langle \bar { \mathbf z } _ { 1 } - \mathbf z _ { 0 } + \eta _ { 0 } G ( \mathbf { z } _ { 0 } ) , \mathbf z ^ { \star } - \bar { \mathbf z } _ { 1 } \rangle \geq \eta _ { 0 } \left( r ( \bar { \mathbf z } _ { 1 } ) - r ( \mathbf { z } ^ { \star } ) \right)
$$

By $2 \langle \mathbf { u } , \mathbf { v } \rangle = \| \mathbf { u } + \mathbf { v } \| ^ { 2 } - \| \mathbf { u } \| ^ { 2 } - \| \mathbf { v } \| ^ { 2 }$ and adding and subtracting $\eta _ { 0 } \langle G ( \mathbf { z } ^ { \star } ) , \mathbf { z } ^ { \star } - \bar { \mathbf { z } } _ { 1 } \rangle$ , we get

$$
\| z _ { 0 } - z ^ { \star } \| ^ { 2 } - \| \bar { z } _ { 1 } - z _ { 0 } \| ^ { 2 } - \| \bar { z } _ { 1 } - z ^ { \star } \| ^ { 2 } + 2 \eta _ { 0 } \langle G ( z _ { 0 } ) - G ( z ^ { \star } ) , \mathbf { z } ^ { \star } - \bar { z } _ { 1 } \rangle \geq 2 \eta _ { 0 } \left( r ( \bar { z } _ { 1 } ) - r ( \mathbf { z } ^ { \star } ) + \langle G ( \mathbf { z } ^ { \star } ) , \bar { \mathbf { z } } _ { 1 } - \mathbf { z } ^ { \star } \rangle \right)
$$

The right-hand side is nonnegative by (1.2) and rearranging gives

$$
\begin{array} { r l } & { \| \bar { \mathbf z } _ { 1 } - \mathbf z _ { 0 } \| ^ { 2 } \leq \| { \mathbf z } _ { 0 } - { \mathbf z } ^ { \star } \| ^ { 2 } - \| \bar { \mathbf z } _ { 1 } - { \mathbf z } ^ { \star } \| ^ { 2 } - 2 \eta _ { 0 } \langle \bar { \mathbf z } _ { 1 } - { \mathbf z } ^ { \star } , G ( { \mathbf z } _ { 0 } ) - G ( { \mathbf z } ^ { \star } ) \rangle } \\ & { \qquad \leq \| { \mathbf z } _ { 0 } - { \mathbf z } ^ { \star } \| ^ { 2 } + \eta _ { 0 } ^ { 2 } \| G ( { \mathbf z } _ { 0 } ) - G ( { \mathbf z } ^ { \star } ) \| ^ { 2 } \leq ( 1 + a ^ { - 3 / 2 } ) \| { \mathbf z } _ { 0 } - { \mathbf z } ^ { \star } \| ^ { 2 } \leq \displaystyle \frac { 2 5 } { 1 6 } \| { \mathbf z } _ { 0 } - { \mathbf z } ^ { \star } \| ^ { 2 } . } \end{array}
$$

Where the last two inequalities use $H \geq L$ and $a \geq 2$ , respectively. Taking square roots in the preceding estimate gives $\begin{array} { r } { \| \bar { \mathbf z } _ { 1 } - \bar { \mathbf z } _ { 0 } \| \le \frac { 5 } { 4 } \| { \mathbf z } _ { 0 } - { \mathbf z } ^ { \star } \| } \end{array}$ , which proves (3.5) for $t = 0$ . Suppose that, for some $t \geq 0$

$$
\| \bar { \mathbf { z } } _ { t + 1 } - \bar { \mathbf { z } } _ { t } \| \leq \frac { 5 a \| \mathbf { z } _ { 0 } - \mathbf { z } ^ { \star } \| } { 4 ( t + a ) } .
$$

Substituting this bound into (3.10) gives

$$
\begin{array} { r l } & { \left\| \bar { \mathbf z } _ { t + 2 } - \bar { \mathbf z } _ { t + 1 } \right\| \leq \left( 1 - \displaystyle \frac { 3 } { 2 ( t + a + 1 ) } \right) \displaystyle \frac { 5 a \| \mathbf z _ { 0 } - \mathbf z ^ { \star } \| } { 4 ( t + a ) } + \displaystyle \frac { 5 a \| \mathbf z _ { 0 } - \mathbf z ^ { \star } \| } { 8 ( t + a ) ( t + a + 1 ) } } \\ & { \qquad = \left( 1 - \displaystyle \frac { 3 } { 2 ( t + a + 1 ) } + \displaystyle \frac { 1 } { 2 ( t + a + 1 ) } \right) \displaystyle \frac { 5 a \| \mathbf z _ { 0 } - \mathbf z ^ { \star } \| } { 4 ( t + a ) } } \\ & { \qquad = \displaystyle \frac { 5 a \| \mathbf z _ { 0 } - \mathbf z ^ { \star } \| } { 4 ( t + a + 1 ) } . } \end{array}
$$

This completes the induction and proves (3.5).

Finally, setting $t \gets t - 1$ in (3.7) and rearranging $\mathrm { g i }$ ves

$$
G ( \bar { \bf z } _ { T } ) + { \bf v } _ { T } = G ( \bar { \bf z } _ { T } ) - G ( \bar { \bf z } _ { T - 1 } ) + \frac { \beta _ { T - 1 } ( { \bf z } _ { 0 } - \bar { \bf z } _ { T - 1 } ) + \bar { \bf z } _ { T - 1 } - \bar { \bf z } _ { T } } { \eta _ { T - 1 } } .
$$

Therefore, the triangle inequality and Lipschitzness $\mathrm { g i }$ ves

$$
\begin{array} { l } { \displaystyle \| G ( \bar { \bf z } _ { t } ) + { \bf v } _ { t } \| \leq L \| \bar { \bf z } _ { t } - \bar { \bf z } _ { t - 1 } \| + \frac { \beta _ { t - 1 } \| { \bf z } _ { 0 } - \bar { \bf z } _ { t - 1 } \| + \| \bar { \bf z } _ { t } - \bar { \bf z } _ { t - 1 } \| } { \eta _ { t - 1 } } } \\ { \displaystyle \leq \frac { 5 a L \| { \bf z } _ { 0 } - { \bf z } ^ { \star } \| } { 4 ( t - 1 + a ) } + H ( t - 1 + a ) ^ { 3 / 4 } \left( \frac { a } { t - 1 + a } \cdot \frac { 5 } { 2 } \| { \bf z } _ { 0 } - { \bf z } ^ { \star } \| + \frac { 5 a \| { \bf z } _ { 0 } - { \bf z } ^ { \star } \| } { 4 ( t - 1 + a ) } \right) } \\ { \displaystyle = \frac { 5 a L \| { \bf z } _ { 0 } - { \bf z } ^ { \star } \| } { 4 ( t - 1 + a ) } + \frac { 1 5 a H \| { \bf z } _ { 0 } - { \bf z } ^ { \star } \| } { 4 ( t - 1 + a ) ^ { 1 / 4 } } } \\ { \displaystyle = \frac { 5 a H \| { \bf z } _ { 0 } - { \bf z } ^ { \star } \| } { 2 ( t - 1 + a ) ^ { 1 / 4 } } \left( \frac { 3 } { 2 } + \frac { L } { 2 H ( t - 1 + a ) ^ { 3 / 4 } } \right) . } \end{array}
$$

The second inequality uses Lemma 3.1, (3.5), $\beta _ { t - 1 } = a / ( t - 1 + a )$ , and $\eta _ { t - 1 } ^ { - 1 } = H ( t - 1 + a ) ^ { 3 / 4 }$ . Since $\mathbf { v } _ { t } \in \partial r ( \bar { \mathbf { z } } _ { t } )$ , this proves (3.6). ■

## 3.2 Pointwise Bounds between Stochastic and Deterministic Sequences

We follow the stochastic comparison argument of $[ \mathrm { S Y L J ^ { + } 2 6 } ]$ , bounding the distance between the stochastic and deterministic iterates. For the constrained problem, we adapt the argument using proximal nonexpansiveness under Assumption 1.2. Since this variance assumption is weaker than the assumption in $[ \mathrm { S Y L J ^ { + } 2 6 } ]$ (see Section 1.1), our result extends this work even in the unconstrained case. Interestingly, a similar analysis was used in [ZLS26, Section 5.4] in the unconstrained case, but not in the constrained case where the strategy was characterizing the solution of a perturbed problem, similar to [ITAA26]. As a result, both works got the $O ( t ^ { - 1 / 5 } )$ anytime rate for the gap.

Our insight for the constrained extension departs from $[ \mathrm { S Y L J ^ { + } 2 6 }$ , Section 7] and [ZLS26, Remark 4.2] in that we do not directly analyze the gap function in the constrained setting, but go though the gradient mapping. The advantages are three-fold: (1) gradient mapping norm is Lipschitz; (2) gradient mapping norm is a classical optimality measure for problems without bounded constraint sets; (3) since gradient mapping norm majorizes the (restricted) gap, the anytime rate we prove for the gradient mapping norm directly transfers to the same rate on the gap.

Lemma 3.3. Under Assumptions 1.1 and 1.2, for every $t \geq 0$ , we have

$$
\mathbb { E } \Vert \mathbf { z } _ { t } - \bar { \mathbf { z } } _ { t } \Vert ^ { 2 } \leq \frac { \sigma ^ { 2 } + \frac { 2 5 } { 2 } B ^ { 2 } \Vert \mathbf { z } _ { 0 } - \mathbf { z } ^ { \star } \Vert ^ { 2 } } { H ^ { 2 } ( a - \frac { 1 } { 2 } ) \sqrt { t + a } } .\tag{3.11}
$$

Moreover, we have

$$
\mathbb { E } \| { \mathbf { z } } _ { t } - { \mathbf { z } } _ { 0 } \| ^ { 2 } \leq \frac { 2 5 } { 2 } \| { \mathbf { z } } _ { 0 } - { \mathbf { z } } ^ { \star } \| ^ { 2 } + \frac { 2 \sigma ^ { 2 } + 2 5 B ^ { 2 } \| { \mathbf { z } } _ { 0 } - { \mathbf { z } } ^ { \star } \| ^ { 2 } } { H ^ { 2 } ( a - \frac { 1 } { 2 } ) \sqrt { t + a } } .\tag{3.12}
$$

Proof. By the Young’s inequality and Lemma 3.1, we have

$$
\| \mathbf { z } _ { t } - \mathbf { z } _ { 0 } \| ^ { 2 } \leq 2 \| \mathbf { z } _ { t } - { \bar { \mathbf { z } } } _ { t } \| ^ { 2 } + 2 \| \mathbf { z } _ { 0 } - { \bar { \mathbf { z } } } _ { t } \| ^ { 2 } \leq 2 \| \mathbf { z } _ { t } - { \bar { \mathbf { z } } } _ { t } \| ^ { 2 } + \frac { 2 5 } { 2 } \| \mathbf { z } _ { 0 } - \mathbf { z } ^ { \star } \| ^ { 2 } .\tag{3.13}
$$

Using the definitions of $\mathbf { z } _ { t + 1 } , \bar { \mathbf { z } } _ { t + 1 }$ , and the nonexpansiveness of the proximal operator, we obtain

$$
\begin{array} { r l } & { \| \mathbf { z } _ { t + 1 } - \bar { \mathbf { z } } _ { t + 1 } \| ^ { 2 } = \| \operatorname* { p r o x } _ { \eta \neq r } \left( ( 1 - \beta _ { t } ) \mathbf { z } _ { t } + \beta _ { t } \mathbf { z } _ { 0 } - \eta _ { t } \tilde { G } ( \mathbf { z } _ { t } ) \right) - \operatorname* { p r o x } _ { \eta \neq r } \left( ( 1 - \beta _ { t } ) \bar { \mathbf { z } } _ { t } + \beta _ { t } \mathbf { z } _ { 0 } - \eta _ { t } G ( \bar { \mathbf { z } } _ { t } ) \right) \| ^ { 2 } } \\ & { \qquad \leq \left\| ( 1 - \beta _ { t } ) ( \mathbf { z } _ { t } - \bar { \mathbf { z } } _ { t } ) - \eta _ { t } \big ( \tilde { G } ( \mathbf { z } _ { t } ) - G ( \bar { \mathbf { z } } _ { t } ) \big ) \right\| ^ { 2 } . } \end{array}
$$

We add and subtract $G ( \mathbf { z } _ { t } )$ inside the norm and take conditional expectation. Since the stochastic oracle is unbiased, the noise cross term vanishes, yielding

$$
\begin{array} { r } { \mathbb { E } _ { t } \| { \mathbf { z } } _ { t + 1 } - \bar { { \mathbf { z } } } _ { t + 1 } \| ^ { 2 } \leq \left\| ( 1 - \beta _ { t } ) ( { \mathbf { z } } _ { t } - \bar { { \mathbf { z } } } _ { t } ) - \eta _ { t } \big ( G ( { \mathbf { z } } _ { t } ) - G ( \bar { { \mathbf { z } } } _ { t } ) \big ) \right\| ^ { 2 } + \eta _ { t } ^ { 2 } \mathbb { E } _ { t } \| \widetilde G ( { \mathbf { z } } _ { t } ) - G ( { \mathbf { z } } _ { t } ) \| ^ { 2 } . } \end{array}\tag{3.14}
$$

We use Lemma 4.1 with ${ \bf u } = { \bf z } _ { t } , { \bf v } = \bar { \bf z } _ { t } , \alpha = 1 - \beta _ { t } , \gamma = \eta _ { t }$ to get

$$
\left\| ( 1 - \beta _ { t } ) ( \mathbf { z } _ { t } - { \bar { \mathbf { z } } } _ { t } ) - \eta _ { t } ( G ( \mathbf { z } _ { t } ) - G ( { \bar { \mathbf { z } } } _ { t } ) ) \right\| \leq { \sqrt { ( 1 - \beta _ { t } ) ^ { 2 } + \eta _ { t } ^ { 2 } L ^ { 2 } \| \mathbf { z } _ { t } - { \bar { \mathbf { z } } } _ { t } \| } } .
$$

Next, Assumption 1.2 and (3.13) bound the last term on the right-hand side as

$$
\mathbb { E } _ { t } \Vert \widetilde { G } ( \mathbf { z } _ { t } ) - G ( \mathbf { z } _ { t } ) \Vert ^ { 2 } \leq B ^ { 2 } \Vert \mathbf { z } _ { t } - \mathbf { z } _ { 0 } \Vert ^ { 2 } + \sigma ^ { 2 } \leq 2 B ^ { 2 } \Vert \mathbf { z } _ { t } - \bar { \mathbf { z } } _ { t } \Vert ^ { 2 } + \frac { 2 5 } { 2 } B ^ { 2 } \Vert \mathbf { z } _ { 0 } - \mathbf { z } ^ { \star } \Vert ^ { 2 } + \sigma ^ { 2 } .\tag{3.15}
$$

Substituting these two bounds into (3.14) gives

$$
\mathbb { E } _ { t } \Vert \mathbf { z } _ { t + 1 } - \bar { \mathbf { z } } _ { t + 1 } \Vert ^ { 2 } \leq \left( ( 1 - \beta _ { t } ) ^ { 2 } + \eta _ { t } ^ { 2 } ( L ^ { 2 } + 2 B ^ { 2 } ) \right) \Vert \mathbf { z } _ { t } - \bar { \mathbf { z } } _ { t } \Vert ^ { 2 } + \eta _ { t } ^ { 2 } \left( \frac { 2 5 } { 2 } B ^ { 2 } \Vert \mathbf { z } _ { 0 } - \mathbf { z } ^ { \star } \Vert ^ { 2 } + \sigma ^ { 2 } \right) .\tag{3.16}
$$

For the rest of the proof let us set

$$
V _ { \star } = \frac { 2 5 } { 2 } B ^ { 2 } \| { \bf z } _ { 0 } - { \bf z } ^ { \star } \| ^ { 2 } + \sigma ^ { 2 } .\tag{3.17}
$$

For $t \geq 1$ , Fact 4.4 gives

$$
( 1 - \beta _ { t } ) ^ { 2 } + \eta _ { t } ^ { 2 } ( L ^ { 2 } + 2 B ^ { 2 } ) \leq 1 - \frac { a } { t + a } .
$$

Taking expectation in (3.16) and using the bound together with (3.17) yields

$$
\mathbb { E } \| \mathbf { z } _ { t + 1 } - \bar { \mathbf { z } } _ { t + 1 } \| ^ { 2 } \leq \left( 1 - \frac { a } { t + a } \right) \mathbb { E } \| \mathbf { z } _ { t } - \bar { \mathbf { z } } _ { t } \| ^ { 2 } + \frac { V _ { \star } } { H ^ { 2 } ( t + a ) ^ { 3 / 2 } } , \qquad t \geq 1 .\tag{3.18}
$$

At $t = 0 , \mathbf { z } _ { 0 } = \bar { \mathbf { z } } _ { 0 }$ . Substituting this into (3.14) and using $\eta _ { 0 } ^ { 2 } = 1 / ( H ^ { 2 } a ^ { 3 / 2 } )$ gives

$$
\mathbb { E } \Vert \mathbf { z } _ { 1 } - \bar { \mathbf { z } } _ { 1 } \Vert ^ { 2 } \leq \eta _ { 0 } ^ { 2 } \mathbb { E } \Vert \widetilde { G } ( \mathbf { z } _ { 0 } ) - G ( \mathbf { z } _ { 0 } ) \Vert ^ { 2 } \leq \eta _ { 0 } ^ { 2 } \left( V _ { \star } \right) = \frac { V _ { \star } } { H ^ { 2 } a ^ { 3 / 2 } } ,
$$

where the second inequality used (3.15) and (3.17). Thus, (3.18) also holds at $t = 0$

We now prove (3.11) by induction. The case $t = 0$ follows from ${ \bf z } _ { 0 } = \bar { \bf z } _ { 0 }$ . Suppose that, for some $t \geq 0$

$$
\mathbb { E } \Vert \mathbf { z } _ { t } - \bar { \mathbf { z } } _ { t } \Vert ^ { 2 } \leq \frac { V _ { \star } } { H ^ { 2 } ( a - \frac { 1 } { 2 } ) \sqrt { t + a } } .
$$

Since $\textstyle 1 - { \frac { a } { t + a } } \geq 0$ , substituting the inductive assumption into (3.18) gives

$$
\begin{array} { r l } & { \mathbb { E } \| { \mathbf z } _ { t + 1 } - \bar { { \mathbf z } } _ { t + 1 } \| ^ { 2 } \leq \left( 1 - \displaystyle \frac { a } { t + a } \right) \displaystyle \frac { V _ { \star } } { H ^ { 2 } ( a - \frac { 1 } { 2 } ) \sqrt { t + a } } + \displaystyle \frac { V _ { \star } } { H ^ { 2 } ( t + a ) ^ { 3 / 2 } } } \\ & { \quad \quad \quad \quad = \displaystyle \frac { V _ { \star } } { H ^ { 2 } ( a - \frac { 1 } { 2 } ) } \left( \displaystyle \frac { 1 } { \sqrt { t + a } } - \displaystyle \frac { 1 } { 2 ( t + a ) ^ { 3 / 2 } } \right) } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \end{array}
$$

The last inequality follows from convexity of $s \mapsto s ^ { - 1 / 2 }$ for $s = t + a ,$ , which gives $( s + 1 ) ^ { - 1 / 2 } \geq s ^ { - 1 / 2 } - \frac { 1 } { 2 } s ^ { - 3 / 2 }$ This completes the induction and proves (3.11) after using (3.17).

Finally, taking expectations in (3.13) and using (3.11) gives

$$
\mathbb { E } \Vert \mathbf { z } _ { t } - \mathbf { z } _ { 0 } \Vert ^ { 2 } \leq 2 \mathbb { E } \Vert \mathbf { z } _ { t } - \bar { \mathbf { z } } _ { t } \Vert ^ { 2 } + \frac { 2 5 } { 2 } \Vert \mathbf { z } _ { 0 } - \mathbf { z } ^ { \star } \Vert ^ { 2 } \leq \frac { 2 5 } { 2 } \Vert \mathbf { z } _ { 0 } - \mathbf { z } ^ { \star } \Vert ^ { 2 } + \frac { 2 V _ { \star } } { H ^ { 2 } ( a - \frac { 1 } { 2 } ) \sqrt { t + a } } ,
$$

which proves (3.12) after using (3.17).

## 4 Technical Results

The following lemma is generalizing a useful idea from [CZ26] and is used many times in the proofs.

Lemma 4.1. Let $G \colon  { \mathbb { R } ^ { d } } \to  { \mathbb { R } ^ { d } }$ be monotone and L-Lipschitz, and $\alpha , \gamma \in \mathbb { R }$ . Then, for any $\mathbf { u } , \mathbf { v } \in \mathbb { R } ^ { d }$ we have

$$
\| \alpha ( \mathbf { u } - \mathbf { v } ) - \gamma ( G ( \mathbf { u } ) - G ( \mathbf { v } ) ) \| \leq { \sqrt { \alpha ^ { 2 } + \gamma ^ { 2 } L ^ { 2 } } } \| \mathbf { u } - \mathbf { v } \| .
$$

Proof. By expanding the square of the left-hand side, we obtain

$$
\begin{array} { r l } & { \| \alpha ( \mathbf { u } - \mathbf { v } ) - \gamma ( G ( \mathbf { u } ) - G ( \mathbf { v } ) ) \| ^ { 2 } = \alpha ^ { 2 } \| \mathbf { u } - \mathbf { v } \| ^ { 2 } + \gamma ^ { 2 } \| G ( \mathbf { u } ) - G ( \mathbf { v } ) \| ^ { 2 } } \\ & { \qquad - 2 \alpha \gamma \langle \mathbf { u } - \mathbf { v } , G ( \mathbf { u } ) - G ( \mathbf { v } ) \rangle . } \end{array}\tag{4.1}
$$

Due to Lipschitzness of $G ,$ we have

$$
\gamma ^ { 2 } \| G ( \mathbf { u } ) - G ( \mathbf { v } ) \| ^ { 2 } \leq \gamma ^ { 2 } L ^ { 2 } \| \mathbf { u } - \mathbf { v } \| ^ { 2 } .
$$

By monotonicity of $G ,$ we have

$$
- 2 \alpha \gamma \langle \mathbf { u } - \mathbf { v } , G ( \mathbf { u } ) - G ( \mathbf { v } ) \rangle \leq 0 .
$$

Combining the last two estimates in (4.1) and taking the square root of both sides gives the assertion. ■

## 4.1 Results on Optimality Measures

The following standard result proves that the rate of convergence that we prove for the gradient mapping norm directly translates to the same rate on the restricted gap, up to an additional multiplicative constant.

Corollary 4.2. Under Assumption 1.1, let U be a compact set. Let $r = \delta _ { Z }$ , where Z is nonempty, closed, and convex, but not necessarily bounded. Then, for every $\mathbf { z } \in Z$ and any $\rho > 0$ , we have

$$
\mathrm { g a p } _ { \mathcal { U } } ( \mathbf { z } ) \leq \left( \operatorname* { m a x } _ { \mathbf { u } \in \mathcal { B } } \| \mathbf { z } _ { 0 } - \mathbf { u } \| + \rho \| G ( \mathbf { z } _ { 0 } ) \| + ( 1 + \rho L ) \| \mathbf { z } - \mathbf { z } _ { 0 } \| \right) \| \mathcal { G } _ { \rho } ( \mathbf { z } ) \| + \rho \| \mathcal { G } _ { \rho } ( \mathbf { z } ) \| ^ { 2 } .\tag{4.2}
$$

Proof. Fix $\mathbf { z } \in C$ and let $\mathbf { v } = \mathrm { p r o j } _ { Z } ( \mathbf { z } - \rho G ( \mathbf { z } ) )$ ). By (1.7), we have $\begin{array} { r } { \frac { \mathbf { z } - \mathbf { v } - \rho G ( \mathbf { z } ) } { \rho } \in \partial \delta _ { Z } ( \mathbf { v } ) } \end{array}$ , and since $\begin{array} { r } { \mathcal { G } _ { \rho } ( \mathbf { z } ) = \frac { \mathbf { z } - \mathbf { v } } { \rho } } \end{array}$ ， we have $\mathcal { G } _ { \rho } ( \mathbf { z } ) - G ( \mathbf { z } ) \in \partial \delta _ { Z } ( \mathbf { v } )$ . As a result, convexity of $\delta _ { Z } \mathrm { \ g i v e s } .$ , for every $\mathbf { u } \in B$ , that

$$
0 \leq \langle { \mathcal { G } } _ { \rho } ( \mathbf { z } ) - G ( \mathbf { z } ) , \mathbf { v } - \mathbf { u } \rangle \iff \langle G ( \mathbf { z } ) , \mathbf { v } - \mathbf { u } \rangle \leq \langle { \mathcal { G } } _ { \rho } ( \mathbf { z } ) , \mathbf { v } - \mathbf { u } \rangle .\tag{4.3}
$$

Using $\mathbf { z } - \mathbf { v } = \rho \mathcal { G } _ { \rho } ( \mathbf { z } )$ , we obtain

$$
\begin{array} { r l r } & { } & { \langle G ( \mathbf { z } ) , \mathbf { z } - \mathbf { u } \rangle = \rho \langle G ( \mathbf { z } ) , \mathcal { G } _ { \rho } ( \mathbf { z } ) \rangle + \langle G ( \mathbf { z } ) , \mathbf { v } - \mathbf { u } \rangle } \\ & { } & { \qquad \leq \rho \langle G ( \mathbf { z } ) , \mathcal { G } _ { \rho } ( \mathbf { z } ) \rangle + \langle \mathcal { G } _ { \rho } ( \mathbf { z } ) , \mathbf { v } - \mathbf { u } \rangle } \\ & { } & { \qquad \leq \left( \| \mathbf { v } - \mathbf { u } \| + \rho \| G ( \mathbf { z } ) \| \right) \| \mathcal { G } _ { \rho } ( \mathbf { z } ) \| , } \end{array}\tag{4.4}
$$

where the first inequality is by (4.3) and the second by Cauchy-Schwarz.

Since $\mathbf { v } = \mathrm { p r o j } _ { C } ( \mathbf { z } - \rho G ( \mathbf { z } ) ) \in C$ , for every $\mathbf { u } \in { \mathcal { U } } .$ , we have

$$
\begin{array} { c } { \| \mathbf { v } - \mathbf { u } \| \leq \| \mathbf { v } - \mathbf { z } \| + \| \mathbf { z } - \mathbf { z } _ { 0 } \| + \| \mathbf { z } _ { 0 } - \mathbf { u } \| } \\ { \quad \quad = \rho \| \mathcal { G } _ { \rho } ( \mathbf { z } ) \| + \| \mathbf { z } - \mathbf { z } _ { 0 } \| + \| \mathbf { z } _ { 0 } - \mathbf { u } \| , } \end{array}
$$

where the first identity is because of $\mathbf { z } - \mathbf { v } = \rho \mathcal { G } _ { \rho } ( \mathbf { z } )$ . Lipschitzness of G gives $\| G ( \mathbf { z } ) \| \leq \| G ( \mathbf { z } _ { 0 } ) \| + L \| \mathbf { z } - \mathbf { z } _ { 0 } \|$ Substituting these inequalities into (4.4) proves (4.2).

The following result shows that the natural residual (gradient mapping norm) is upper bounded by a residual-type quantity. Moreover, we prove that the natural residual is Lipschitz, which is another standard property.

Lemma 4.3. Under Assumption 1.1, for ${ \textbf { z } } \in$ dom ∂r and any $\mathbf { v } \in \partial r ( \mathbf { z } )$ , any $\rho > 0$ and for all $\mathbf { x } , \mathbf { y } \in$ dom r, we have

$$
\begin{array} { r } { \| \mathcal { G } _ { \rho } ( \mathbf { z } ) \| \leq \| G ( \mathbf { z } ) + \mathbf { v } \| , } \end{array}\tag{4.5}
$$

$$
\| \mathcal { G } _ { \rho } ( \mathbf { x } ) - \mathcal { G } _ { \rho } ( \mathbf { y } ) \| \leq \sqrt { \rho ^ { - 2 } + L ^ { 2 } } \| \mathbf { x } - \mathbf { y } \| .\tag{4.6}
$$

Proof. If $\mathbf { v } \in \partial r ( \mathbf { z } )$ , then $\mathbf { z } = \operatorname { p r o x } _ { \rho r } ( \mathbf { z } + \rho \mathbf { v } )$ since the latter is equivalent to $\mathbf { z } + \rho \partial r ( \mathbf { z } ) \ni \mathbf { z } + \rho \mathbf { v }$ . Using this identity with nonexpansiveness gives

$$
\begin{array} { r } { \rho \| \mathcal { G } _ { \rho } ( \mathbf { z } ) \| = \| \mathbf { z } - \mathrm { p r o x } _ { \rho r } ( \mathbf { z } - \rho G ( \mathbf { z } ) ) \| = \| \mathrm { p r o x } _ { \rho r } ( \mathbf { z } + \rho \mathbf { v } ) - \mathrm { p r o x } _ { \rho r } ( \mathbf { z } - \rho G ( \mathbf { z } ) ) \| \leq \rho \| G ( \mathbf { z } ) + \mathbf { v } \| . } \end{array}
$$

We divide by $\rho$ to obtain (4.5).

For the second assertion, the definition of the proximal operator gives $\mathcal { G } _ { \rho } ( \mathbf { x } ) - G ( \mathbf { x } ) \in \partial r ( \mathbf { x } - \rho \mathcal { G } _ { \rho } ( \mathbf { x } ) )$ since $\rho \mathcal { G } _ { \rho } ( \mathbf { x } ) = \mathbf { x } - \mathrm { p r o x } _ { \rho r } ( \mathbf { x } - \rho G ( \mathbf { x } )$ , and the analogous inclusion for y. Monotonicity of $\partial r$ implies

$$
\begin{array} { r } { \langle \mathcal { G } _ { \rho } ( \mathbf { x } ) - G ( \mathbf { x } ) - [ \mathcal { G } _ { \rho } ( \mathbf { y } ) - G ( \mathbf { y } ) ] , \mathbf { x } - \rho \mathcal { G } _ { \rho } ( \mathbf { x } ) - [ \mathbf { y } - \rho \mathcal { G } _ { \rho } ( \mathbf { y } ) ] \rangle \geq 0 . } \end{array}
$$

Expanding the inner product gives

$$
\begin{array} { r } { \rho \| \mathcal { G } _ { \rho } ( \mathbf { x } ) - \mathcal { G } _ { \rho } ( \mathbf { y } ) \| ^ { 2 } \leq \langle \mathcal { G } _ { \rho } ( \mathbf { x } ) - \mathcal { G } _ { \rho } ( \mathbf { y } ) , \mathbf { x } - \mathbf { y } + \rho ( G ( \mathbf { x } ) - G ( \mathbf { y } ) ) \rangle - \langle G ( \mathbf { x } ) - G ( \mathbf { y } ) , \mathbf { x } - \mathbf { y } \rangle . } \end{array}\tag{4.7}
$$

Young’s inequality gives

$$
\begin{array} { r l } & { \langle \mathcal { G } _ { \rho } ( \mathbf { x } ) - \mathcal { G } _ { \rho } ( \mathbf { y } ) , \mathbf { x } - \mathbf { y } + \rho ( G ( \mathbf { x } ) - G ( \mathbf { y } ) ) \rangle \leq \frac { \rho } { 2 } \| \mathcal { G } _ { \rho } ( \mathbf { x } ) - \mathcal { G } _ { \rho } ( \mathbf { y } ) \| ^ { 2 } } \\ & { \qquad + \frac { 1 } { 2 \rho } \left( \| \mathbf { x } - \mathbf { y } \| ^ { 2 } + 2 \rho \langle \mathbf { x } - \mathbf { y } , G ( \mathbf { x } ) - G ( \mathbf { y } ) \rangle + \rho ^ { 2 } \| G ( \mathbf { x } ) - G ( \mathbf { y } ) \| ^ { 2 } \right) . } \end{array}
$$

Plugging to (4.7) yields

$$
\rho \| { \mathcal { G } } _ { \rho } ( \mathbf { x } ) - { \mathcal { G } } _ { \rho } ( \mathbf { y } ) \| ^ { 2 } \leq { \frac { \rho } { 2 } } \| { \mathcal { G } } _ { \rho } ( \mathbf { x } ) - { \mathcal { G } } _ { \rho } ( \mathbf { y } ) \| ^ { 2 } + { \frac { 1 } { 2 \rho } } \| \mathbf { x } - \mathbf { y } \| ^ { 2 } + { \frac { \rho } { 2 } } \| G ( \mathbf { x } ) - G ( \mathbf { y } ) \| ^ { 2 } .
$$

After rearranging and using Lipschitz continuity of $G ,$ we deduce

$$
\| \mathcal { G } _ { \rho } ( \mathbf { x } ) - \mathcal { G } _ { \rho } ( \mathbf { y } ) \| ^ { 2 } \leq \rho ^ { - 2 } \| \mathbf { x } - \mathbf { y } \| ^ { 2 } + \| G ( \mathbf { x } ) - G ( \mathbf { y } ) \| ^ { 2 } \leq ( \rho ^ { - 2 } + L ^ { 2 } ) \| \mathbf { x } - \mathbf { y } \| ^ { 2 } .
$$

Taking square root of both sides proves (4.6).

## 4.2 Numerical Consequences of Parameters

The following fact collects some tedious, yet important estimations we used, which follow from the choice of our parameters.

Fact 4.4. Suppose that $a \ge 2 , H ^ { 2 } = L ^ { 2 } + 2 B ^ { 2 }$ , and

$$
\beta _ { t } = \frac a { t + a } , \qquad \eta _ { t } = \frac 1 { H ( t + a ) ^ { 3 / 4 } } .
$$

Then, for every $t \geq 1$

$$
\sqrt { ( 1 - \beta _ { t } ) ^ { 2 } + \eta _ { t } ^ { 2 } L ^ { 2 } } \leq 1 - \frac 2 3 \beta _ { t } .\tag{4.8}
$$

Moreover, for every $t \geq 0$ , writing $s = t + a + 1$ , we have,

$$
\left( \frac { \eta _ { t + 1 } } { \eta _ { t } } - \beta _ { t + 1 } \right) ^ { 2 } + \eta _ { t + 1 } ^ { 2 } L ^ { 2 } \leq \left( 1 - \frac { 1 1 } { 4 s } \right) ^ { 2 } + s ^ { - 3 / 2 } \leq \left( 1 - \frac { 3 } { 2 s } \right) ^ { 2 } ,\tag{4.9}
$$

and

$$
\left| \beta _ { t + 1 } - \frac { \eta _ { t + 1 } } { \eta _ { t } } \beta _ { t } \right| \leq \frac { a } { 4 ( t + a ) ( t + a + 1 ) } .\tag{4.10}
$$

Finally, for every $t \geq 1$

$$
( 1 - \beta _ { t } ) ^ { 2 } + \eta _ { t } ^ { 2 } ( L ^ { 2 } + 2 B ^ { 2 } ) \leq 1 - \frac { a } { t + a } .\tag{4.11}
$$

Proof. For the first estimate, we have

$$
\begin{array} { r l } & { \left( 1 - \frac { 2 } { 3 } \beta _ { t } \right) ^ { 2 } - ( 1 - \beta _ { t } ) ^ { 2 } - \eta _ { t } ^ { 2 } L ^ { 2 } = \frac { 2 } { 3 } \beta _ { t } - \frac { 5 } { 9 } \beta _ { t } ^ { 2 } - \eta _ { t } ^ { 2 } L ^ { 2 } } \\ & { \qquad = \displaystyle \frac { 1 } { t + a } \left( \frac { 2 a } { 3 } - \frac { 5 a ^ { 2 } } { 9 ( t + a ) } - \frac { L ^ { 2 } } { H ^ { 2 } \sqrt { t + a } } \right) } \\ & { \qquad \geq \displaystyle \frac { 1 } { t + a } \left( \frac { 2 a } { 3 } - \frac { 5 a ^ { 2 } } { 9 ( t + a ) } - \frac { 1 } { \sqrt { t + a } } \right) } \\ & { \qquad \geq \displaystyle \frac { 1 } { t + a } \left( \frac { a ( a + 6 ) } { 9 ( a + 1 ) } - \frac { 1 } { \sqrt { a + 1 } } \right) \geq 0 . } \end{array}
$$

Here, the first inequality uses $H \ \geq \ L$ , and the second uses $t \geq 1$ . The last inequality follows from $a ( a + 6 ) \geq 9 { \sqrt { a + 1 } }$ for $a \geq 2$ . This proves (4.8).

We next prove the estimates for $t \geq 0$ . Since

$$
\frac { \eta _ { t + 1 } } { \eta _ { t } } = \frac { ( t + a ) ^ { 3 / 4 } } { ( t + a + 1 ) ^ { 3 / 4 } } \geq \frac { t + a } { t + a + 1 } \geq \beta _ { t + 1 } = \frac { a } { t + a + 1 } ,
$$

we have $\eta _ { t + 1 } / \eta _ { t } - \beta _ { t + 1 } \geq 0$ . Write $s = t + a + 1 \geq 3$ . By concavity of $x ^ { 3 / 4 }$ , we have

$$
\left( 1 - { \frac { 1 } { s } } \right) ^ { 3 / 4 } \leq 1 + { \frac { 3 } { 4 } } \left[ \left( 1 - { \frac { 1 } { s } } \right) - 1 \right] = 1 - { \frac { 3 } { 4 s } } .
$$

Therefore, by definition of $\eta _ { t }$ and $\beta _ { t + 1 }$ , we have

$$
0 \leq \frac { \eta _ { t + 1 } } { \eta _ { t } } - \beta _ { t + 1 } = \left( 1 - \frac { 1 } { s } \right) ^ { 3 / 4 } - \frac { a } { s } \leq 1 - \frac { 3 } { 4 s } - \frac { a } { s } \leq 1 - \frac { 1 1 } { 4 s } ,
$$

where the last inequality uses $a \geq 2$ . Moreover, $\begin{array} { r } { \eta _ { t + 1 } ^ { 2 } L ^ { 2 } = \frac { L ^ { 2 } } { H ^ { 2 } } s ^ { - 3 / 2 } \leq s ^ { - 3 / 2 } } \end{array}$ because $H \geq L$ . Therefore,

$$
\left( \frac { \eta _ { t + 1 } } { \eta _ { t } } \beta _ { t + 1 } \right) ^ { 2 } + \eta _ { t + 1 } ^ { 2 } L ^ { 2 } \leq \left( 1 - \frac { 1 1 } { 4 s } \right) ^ { 2 } + s ^ { - 3 / 2 } \leq \left( 1 - \frac { 3 } { 2 s } \right) ^ { 2 } .
$$

The last inequality follows from $\begin{array} { r } { \frac { 5 } { 2 } - \frac { 8 5 } { 1 6 s } - s ^ { - 1 / 2 } \geq \frac { 3 5 } { 4 8 } - \frac { 1 } { \sqrt { 3 } } > 0 } \end{array}$ because $s \geq 3$ . This proves (4.9).

Similarly, using the definitions of $\beta _ { t }$ and $\eta _ { t } .$ we have

$$
\begin{array} { r l } & { \left| \beta _ { t + 1 } - \frac { \eta _ { t + 1 } } { \eta _ { t } } \beta _ { t } \right| = \left| \frac { a } { t + a + 1 } - \left( \frac { t + a } { t + a + 1 } \right) ^ { 3 / 4 } \frac { a } { t + a } \right| } \\ & { \qquad = \frac { a } { t + a + 1 } \left| 1 - \left( \frac { t + a + 1 } { t + a } \right) ^ { 1 / 4 } \right| } \\ & { \qquad = \frac { a } { t + a + 1 } \left[ \left( 1 + \frac { 1 } { t + a } \right) ^ { 1 / 4 } - 1 \right] . } \end{array}
$$

The last equality follows because $\left( 1 + \frac { 1 } { t + a } \right) ^ { 1 / 4 } \geq 1$ . Then, concavity of $x ^ { 1 / 4 }$ yields $\begin{array} { r } { \left( 1 + \frac { 1 } { t + a } \right) ^ { 1 / 4 } \leq 1 + \frac { 1 } { 4 ( t + a ) } } \end{array}$ Therefore, we have

$$
\left| \beta _ { t + 1 } - \frac { \eta _ { t + 1 } } { \eta _ { t } } \beta _ { t } \right| \leq \frac { a } { 4 ( t + a ) ( t + a + 1 ) } ,
$$

which proves (4.10).

Finally, for $t \geq 1$

$$
\left( t + a \right) \left[ 1 - \left( 1 - \beta _ { t } \right) ^ { 2 } - \eta _ { t } ^ { 2 } \bigl ( L ^ { 2 } + 2 B ^ { 2 } \bigr ) \right] = 2 a - \frac { a ^ { 2 } } { t + a } - \frac { L ^ { 2 } + 2 B ^ { 2 } } { H ^ { 2 } \sqrt { t + a } } \geq 2 a - \frac { a ^ { 2 } } { t + a } - \frac { 1 } { \sqrt { t + a } } \geq a .\tag{4.12}
$$

Here, the first inequality uses $H ^ { 2 } = L ^ { 2 } + 2 B ^ { 2 }$ , while the last inequality follows from

$$
a - { \frac { a ^ { 2 } } { t + a } } - { \frac { 1 } { \sqrt { t + a } } } \geq { \frac { a } { a + 1 } } - { \frac { 1 } { \sqrt { a + 1 } } } \geq 0
$$

for $t \geq 1$ and $a \geq 2$ . Dividing (4.12) by $t + a$ proves (4.11).

## 5 Deferred Proofs

Proof of Corollary 2.2. Setting $\mathbf { z } = \mathbf { z } _ { t }$ in (4.2), taking expectations, and applying Cauchy–Schwarz $\mathrm { g i }$ ve

$$
\begin{array} { r l } & { \mathbb { E } \operatorname { g a p } ( \mathbf { z } _ { t } ) \leq \left[ \underset { \mathbf { u } \in B } { \operatorname* { m a x } } \left. \mathbf { z } _ { 0 } - \mathbf { u } \right. + \rho \Vert G ( \mathbf { z } _ { 0 } ) \Vert \right] \mathbb { E } \Vert \mathcal { G } _ { \rho } ( \mathbf { z } _ { t } ) \Vert + ( 1 + \rho L ) \sqrt { \mathbb { E } \Vert \mathbf { z } _ { t } - \mathbf { z } _ { 0 } \Vert ^ { 2 } } \sqrt { \mathbb { E } \Vert \mathcal { G } _ { \rho } ( \mathbf { z } _ { t } ) \Vert ^ { 2 } } } \\ & { \qquad + \rho \mathbb { E } \Vert \mathcal { G } _ { \rho } ( \mathbf { z } _ { t } ) \Vert ^ { 2 } . } \end{array}\tag{5.1}
$$

By (3.12), because $t \geq 1$ , we have $1 / ( \sqrt { t + a } ) \leq 1 / ( \sqrt { a + 1 } )$ ，

$$
\mathbb { E } \| \mathbf { z } _ { t } - \mathbf { z } _ { 0 } \| ^ { 2 } \leq \frac { 2 5 } { 2 } \| \mathbf { z } _ { 0 } - \mathbf { z } ^ { \star } \| ^ { 2 } + \frac { 2 \sigma ^ { 2 } + 2 5 B ^ { 2 } \| \mathbf { z } _ { 0 } - \mathbf { z } ^ { \star } \| ^ { 2 } } { H ^ { 2 } ( a - \frac { 1 } { 2 } ) \sqrt { a } } < \infty .
$$

Combining this bound with Theorem 2.1 gives the assertion.

Proof of Corollary 2.3. For every $( \mathbf { x } ^ { \prime } , \mathbf { y } ^ { \prime } ) \in \mathcal { U } _ { \mathbf { x } } \times \mathcal { U } _ { \mathbf { y } }$ , convexity in x and concavity in y give

$$
\begin{array} { r l } & { f ( \mathbf { x } , \mathbf { y } ^ { \prime } ) - f ( \mathbf { x } ^ { \prime } , \mathbf { y } ) = f ( \mathbf { x } , \mathbf { y } ^ { \prime } ) - f ( \mathbf { x } , \mathbf { y } ) + f ( \mathbf { x } , \mathbf { y } ) - f ( \mathbf { x } ^ { \prime } , \mathbf { y } ) } \\ & { \qquad \leq \left. \nabla _ { \mathbf { y } } f ( \mathbf { x } , \mathbf { y } ) , \mathbf { y } ^ { \prime } - \mathbf { y } \right. + \left. \nabla _ { \mathbf { x } } f ( \mathbf { x } , \mathbf { y } ) , \mathbf { x } - \mathbf { x } ^ { \prime } \right. } \\ & { \qquad = \left. G ( \mathbf { z } ) , \mathbf { z } - \left( \mathbf { x } ^ { \prime } \right) \right. . } \end{array}
$$

Taking the maximum over $( \mathbf { x } ^ { \prime } , \mathbf { y } ^ { \prime } ) \in \mathcal { U } _ { \mathbf { x } } \times \mathcal { U } _ { \mathbf { y } }$ gives

$$
\operatorname* { m a x } _ { \mathbf { y } ^ { \prime } \in \mathcal { U } _ { \mathbf { y } } } f ( \mathbf { x } , \mathbf { y } ^ { \prime } ) - \operatorname* { m i n } _ { \mathbf { x } ^ { \prime } \in \mathcal { U } _ { \mathbf { x } } } f ( \mathbf { x } ^ { \prime } , \mathbf { y } ) \leq \mathrm { g a p } ( \mathbf { z } ) .
$$

Taking expectations at $\mathbf { z } = \mathbf { z } _ { t }$ and applying Corollary 2.2 proves the result.

## Acknowledgments

This research was funded by the Natural Sciences and Engineering Research Council of Canada (NSERC), [funding reference number RGPIN-2025-06634].

## References

[AASI24] Kenshi Abe, Kaito Ariu, Mitsuki Sakamoto, and Atsushi Iwasaki. Adaptively perturbed mirror descent for learning in games. In International Conference on Machine Learning, 2024.

[AIMM21] Wa¨ıss Azizian, Franck Iutzeler, J´erˆome Malick, and Panayotis Mertikopoulos. The last-iterate convergence rate of optimistic mirror descent in stochastic variational inequalities. In Conference on Learning Theory, pages 326–358. PMLR, 2021.

[AK26] Ahmet Alacaoglu and Jun-Hyun Kim. Solving stochastic variational inequalities without the bounded variance assumption. In International Conference on Machine Learning, 2026.

[Ala26] Ahmet Alacaoglu. How to make the gradient mapping small for constrained stochastic min-max problems and beyond. arXiv preprint arXiv:2609.08380, 2026.

[AMW25] Ahmet Alacaoglu, Yura Malitsky, and Stephen J Wright. Towards weaker variance assumptions for stochastic optimization. arXiv preprint arXiv:2504.09951, 2025.

[ASAI25] Kenshi Abe, Mitsuki Sakamoto, Kaito Ariu, and Atsushi Iwasaki. Boosting perturbed gradient ascent for last-iterate convergence in games. In International Conference on Learning Representations, volume 2025, pages 93223–93253, 2025.

[CBL06] Nicolo Cesa-Bianchi and G´abor Lugosi. Prediction, learning, and games, volume 1. Cambridge university press Cambridge, 2006.

[CL24] Lesi Chen and Luo Luo. Near-optimal algorithms for making the gradient small in stochastic minimax optimization. Journal of Machine Learning Research, 25(387):1–44, 2024.

[CM24] Zaiwei Chen and Eric Mazumdar. Last-iterate convergence for generalized frank-wolfe in monotone variational inequalities. Advances in Neural Information Processing Systems, 37:115440–115467, 2024.

[COZ24] Yang Cai, Argyris Oikonomou, and Weiqiang Zheng. Accelerated algorithms for constrained nonconvex-nonconcave min-max optimization and comonotone inclusion. In International Conference on Machine Learning, 2024.

[CSGD22] Xufeng Cai, Chaobing Song, Crist´obal Guzm´an, and Jelena Diakonikolas. Stochastic halpern iteration with variance reduction for stochastic monotone inclusions. Advances in Neural Information Processing Systems, 35:24766–24779, 2022.

[CZ23] Yang Cai and Weiqiang Zheng. Doubly optimal no-regret learning in monotone games. In International Conference on Machine Learning, pages 3507–3524. PMLR, 2023.

[CZ24] Yang Cai and Weiqiang Zheng. Accelerated single-call methods for constrained min-max optimization. In International Conference on Learning Representations, 2024.

[CZ26] Yang Cai and Weiqiang Zheng. Last-iterate convergence of anchored gradient descent. arXiv preprint arXiv:2604.12235, 2026.

[GPD20] Noah Golowich, Sarath Pattathil, and Constantinos Daskalakis. Tight last-iterate convergence rates for no-regret learning in multi-player games. Advances in neural information processing systems, 33:20766–20778, 2020.

[GPDO20] Noah Golowich, Sarath Pattathil, Constantinos Daskalakis, and Asuman Ozdaglar. Last iterate is slower than averaged iterate in smooth convex-concave saddle point problems. In Conference on Learning Theory, pages 1758–1784. PMLR, 2020.

[GTG22] Eduard Gorbunov, Adrien Taylor, and Gauthier Gidel. Last-iterate convergence of optimistic gradient method for monotone variational inequalities. Advances in neural information processing systems, 35:21858–21870, 2022.

[HACM22] Yu-Guan Hsieh, Kimon Antonakopoulos, Volkan Cevher, and Panayotis Mertikopoulos. No-regret learning in games with noisy feedback: Faster rates and adaptivity via learning rate separation. Advances in Neural Information Processing Systems, 35:6544–6556, 2022.

[Hal67] Benjamin Halpern. Fixed points of nonexpanding maps. Bulletin of the American Mathematical Society, 73(6):957–961, 1967.

[HIMM19] Yu-Guan Hsieh, Franck Iutzeler, J´erˆome Malick, and Panayotis Mertikopoulos. On the convergence of single-call stochastic extra-gradient methods. Advances in Neural Information Processing Systems, 32, 2019.

[IJOT17] Alfredo N Iusem, Alejandro Jofr´e, Roberto Imbuzeiro Oliveira, and Philip Thompson. Extragradient method with variance reduction for stochastic variational inequalities. SIAM Journal on Optimization, 27(2):686–724, 2017.

[ITAA26] Shinji Ito, Taira Tsuchiya, Kaito Ariu, and Kenshi Abe. Last-iterate convergence of regularized gradient methods for stochastic monotone variational inequalities. In International Conference on Machine Learning, 2026.

[JLZ26] Yao Ji, Guanghui Lan, and Jason Zhu. Computation of strong solutions to stochastic variational inequalities. arXiv preprint arXiv:2609.04188, 2026.

[KG22] Dmitry Kovalev and Alexander Gasnikov. The first optimal algorithm for smooth and stronglyconvex-strongly-concave minimax optimization. Advances in Neural Information Processing Systems, 35:14691–14703, 2022.

[KLL22] Georgios Kotsalis, Guanghui Lan, and Tianjiao Li. Simple and optimal methods for stochastic variational inequalities, i: operator extrapolation. SIAM Journal on Optimization, 32(3):2041– 2073, 2022.

[LK21] Sucheol Lee and Donghwan Kim. Fast extra gradient methods for smooth structured nonconvexnonconcave minimax problems. Advances in Neural Information Processing Systems, 34:22588– 22600, 2021.

[MZ19] Panayotis Mertikopoulos and Zhengyuan Zhou. Learning in games with continuous action sets and unknown payof functions. Mathematical Programming, 173(1):465–507, 2019.

[Nes07] Yurii Nesterov. Dual extrapolation and its applications to solving variational inequalities and related problems. Mathematical Programming, 109(2):319–344, 2007.

[NJLS09] Arkadi Nemirovski, Anatoli Juditsky, Guanghui Lan, and Alexander Shapiro. Robust stochastic approximation approach to stochastic programming. SIAM Journal on optimization, 19(4):1574– 1609, 2009.

[NO24] Gergely Neu and Nneka Okolo. Dealing with unbounded gradients in stochastic saddle-point optimization. In International Conference on Machine Learning, 2024.

[PFL<sup>+</sup>23] Thomas Pethick, Olivier Fercoq, Puya Latafat, Panagiotis Patrinos, and Volkan Cevher. Solving stochastic weak minty variational inequalities without increasing batch size. In International Conference on Learning Representations, 2023.

[PXC23] Thomas Pethick, Wanyun Xie, and Volkan Cevher. Stable nonconvex-nonconcave training via linear interpolation. Advances in Neural Information Processing Systems, 36:49830–49841, 2023.

[SYLJ<sup>+</sup>26] Motahareh Sohrabi, Jianxin You, Simon Lacoste-Julien, Eduard Gorbunov, and Gauthier Gidel. Accelerated and stable convergence with anchored generalized optimistic method. In International Conference on Machine Learning, 2026.

[TDNT26] Quoc Tran-Dinh and Nghia Nguyen-Trung. Unbiased and biased variance-reduced forwardreflected-backward splitting methods for stochastic composite inclusions. arXiv preprint arXiv:2603.15576, 2026.

[YR21] TaeHo Yoon and Ernest K Ryu. Accelerated algorithms for smooth convex-concave minimax problems with O(1/k<sup>2</sup>) rate on squared gradient norm. In International conference on machine learning, pages 12098–12109. PMLR, 2021.

[ZLS26] Taoli Zheng, Jiajin Li, and Anthony Man-Cho So. Last-iterate convergence of single-loop stochastic methods for constrained convex-concave minimax problems. arXiv preprint arXiv:2607.11056, 2026.