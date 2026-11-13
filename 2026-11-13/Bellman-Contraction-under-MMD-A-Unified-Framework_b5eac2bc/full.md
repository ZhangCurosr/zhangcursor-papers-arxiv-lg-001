# Bellman Contraction under MMD: A Unified Framework

Baptiste Debes ESAT-PSI KU Leuven baptiste.debes@kuleuven.be

Tinne Tuytelaars ESAT-PSI KU Leuven tinne.tuytelaars@kuleuven.be

## Abstract

We introduce a mathematical framework for studying contraction of the distributional Bellman operator under Maximum Mean Discrepancy (MMD). We show that contraction reduces to conditional positive definiteness of an associated kernel constructed from the base kernel, which we call the Bellman difference kernel. Focusing on radial kernels, we use Schoenberg’s theorem to connect this condition to Bernstein functions. This gives contraction criteria that hold in any dimension and can be verified analytically or tested numerically. Our framework unifies known results for the Gaussian RBF, power-distance, and multiquadric kernels. A numerical search identifies 11 new kernels that induce contraction. We also propose chaining, a general construction that produces kernels inducing contraction with any prescribed factor. To reconcile theory with practice, we show that contraction on a set only requires conditional positive definiteness on that set. This explains why the RBF kernel can remain a valid choice when rewards are bounded, despite its lack of global contraction. Finally, we identify a fundamental tradeoff between contraction rate and discriminative power, where kernels that contract faster tend to discriminate less between distributions. Code is available at https://github.com/BaptisteDebes/BellmanContractionMMD

## 1 Introduction

Distributional reinforcement learning (DRL) models the distribution over returns instead of their expectations [3, 4]. This in turn makes it possible to account for irreducible uncertainty in transitions and rewards exhibited by some environments [23]. It also enables risk-aware policy improvement [10, 32]. Distributional RL has shown substantial performance gains on a wide variety of tasks [2, 10, 18] and has been motivated theoretically [21, 27, 28]. A practical instantiation of DRL depends on both how the return distribution is parametrized and how discrepancies between probability distributions are measured [28]. A natural and flexible choice is to represent the return distribution as a set of equally weighted particles [10, 25, 33]. This representation enables the use of discrepancies computed from samples, such as Maximum Mean Discrepancy (MMD) [16, 19, 25, 38], Sinkhorn divergence [14, 34], or sliced divergences [13].

Maximum Mean Discrepancy depends on a kernel that controls which aspects of the distributions are compared and how sensitively. A central question is then whether the distributional Bellman operator [3] is contractive under the MMD induced by a given kernel, which governs whether the resulting learning objective is principled [3, 11]. Despite the flexibility afforded by kernel choice, existing understanding of which kernels induce contraction remains limited. In particular, it provides little guidance for identifying or constructing new ones. This in turn leaves practitioners with few theoretically motivated options.

The first contribution of this work is to connect contraction under MMD with a definiteness property of an associated kernel, namely conditional positive definiteness. For the important class of radial kernels, we leverage Schoenberg’s theorem [30], which links radial kernels to Bernstein functions [17, 29]. This perspective allows us to show that contraction can be verified analytically. This unifies known results from [12, 19, 25] under a single framework. More broadly, this framework provides practical recipes for identifying kernels that induce contraction and for constructing new ones, which we demonstrate in this work. Notably, all results hold in arbitrary dimension, directly covering the multivariate setting [13, 37, 38].

Secondly, a gap remains between theory and practice, as some kernels such as the RBF are known not to induce contraction in general, yet still perform among the best in practice [25]. We reconcile theory and practice by showing numerically that such kernels can in fact induce contraction when the return laws are supported on a bounded set. In reinforcement learning, this assumption arises naturally when rewards are bounded, since discounted returns are then bounded as well.

Finally, we identify a tradeoff between the contraction induced by a kernel and its ability to discriminate between distributions. In general, kernels with faster contraction tend to have weaker discriminative power, so these two aspects cannot be optimized independently. We study this tradeoff empirically by measuring the power of kernel two-sample tests.

## 2 Background and related work

We consider a discounted Markov decision process (MDP) $( S , { \mathcal { A } } , P , R , \gamma )$ , where $s$ is the state space, $\mathcal { A }$ the action space, $\textstyle P ( \cdot \mid s , a )$ the transition kernel, $R ( s , a )$ the random reward, and $\gamma \in ( 0 , 1 )$ the discount factor. Rewards may be multivariate, in which case returns take values in $\mathbb { R } ^ { d }$ . For a fixed policy $\pi ,$ , we define the random return as follows, with its expectation yielding the usual state-action value function:

$$
Z ^ { \pi } ( \boldsymbol { s } , \boldsymbol { a } ) : = \sum _ { t \geq 0 } \gamma ^ { t } R _ { t } , \qquad Q ^ { \pi } ( \boldsymbol { s } , \boldsymbol { a } ) : = \mathbb { E } [ Z ^ { \pi } ( \boldsymbol { s } , \boldsymbol { a } ) ] ,
$$

where $S _ { 0 } = s , A _ { 0 } = a , R _ { t } = R ( S _ { t } , A _ { t } ) , S _ { t + 1 } \sim P ( \cdot \mid S _ { t } , A _ { t } )$ and $A _ { t + 1 } \sim \pi ( \cdot \mid S _ { t + 1 } )$ . We represent return laws by maps $\eta : \mathcal { S } \times \mathcal { A }  \mathcal { P } ( \mathbb { R } ^ { d } )$ . For a fixed policy π, the distributional Bellman operator is defined, for $S ^ { \prime } \stackrel { . } { \sim } \dot { P ( } \cdot \mid s , a ) , A ^ { \prime } \sim \dot { \pi } ( \cdot \mid S ^ { \prime } )$ , and $X ^ { \prime } \sim \eta ( S ^ { \prime } , A ^ { \prime } )$ , by

$$
( T ^ { \pi } \eta ) ( s , a ) : = \operatorname { L a w } \big ( R ( s , a ) + \gamma X ^ { \prime } \big ) .\tag{1}
$$

Policy evaluation. This work focuses on distributional policy evaluation for a fixed policy $\pi ,$ where one seeks to solve or approximate the fixed point equation $\dot { \eta } ^ { \pi } = T ^ { \pi } \eta ^ { \pi }$ . Our goal is to study the contraction properties of the corresponding distributional Bellman operator $\tau ^ { \pi }$ , rather than policy improvement or control.

Maximum Mean Discrepancy. A kernel is a symmetric function $k : \mathbb { R } ^ { d } \times \mathbb { R } ^ { d }  \mathbb { R }$ . For laws $\mu , \nu \in \mathcal P ( \mathbb { R } ^ { d } )$ , the squared Maximum Mean Discrepancy (MMD) induced by $k$ is defined by [16]

$$
\mathbf { M M D } _ { k } ^ { 2 } ( \mu , \nu ) = \mathbb { E } _ { x , x ^ { \prime } \sim \mu } [ k ( x , x ^ { \prime } ) ] + \mathbb { E } _ { y , y ^ { \prime } \sim \nu } [ k ( y , y ^ { \prime } ) ] - 2 \mathbb { E } _ { x \sim \mu , y \sim \nu } [ k ( x , y ) ] .\tag{2}
$$

Some background on kernels and MMD is provided in Appendix A, including the biased estimator of $ { \mathbf { M M D } } _ { k } ^ { 2 }$ [16], which is the one commonly used in distributional RL [19, 25].

Contraction. Return laws are conditional distributions, so to compare two of them we use the supremum MMD over state-action pairs [3, 25], defined by

$$
\overline { { \mathbf { M M D } } } _ { k } ^ { 2 } ( \eta _ { 1 } , \eta _ { 2 } ) : = \operatorname* { s u p } _ { ( s , a ) \in \mathcal { S } \times \mathcal { A } } \mathbf { M M D } _ { k } ^ { 2 } \big ( \eta _ { 1 } ( s , a ) , \eta _ { 2 } ( s , a ) \big ) .\tag{3}
$$

For $\tau \in [ 0 , 1 )$ , we say that $\mathcal { T } ^ { \pi }$ is a τ-contraction in $\overline { { \mathbf { M M D } } } _ { k } ^ { 2 }$ if

$$
\overline { { \mathbf { M M D } } } _ { k } ^ { 2 } ( \mathcal T ^ { \pi } \eta _ { 1 } , \mathcal T ^ { \pi } \eta _ { 2 } ) \leq \tau \overline { { \mathbf { M M D } } } _ { k } ^ { 2 } ( \eta _ { 1 } , \eta _ { 2 } ) \qquad \forall \eta _ { 1 } , \eta _ { 2 } .
$$

We call any such $\tau { \mathrm { ~ a ~ } }$ contraction factor of $\mathcal { T } ^ { \pi }$ and denote by $\tau _ { k } ^ { \star } ( \mathcal { T } ^ { \pi } )$ the smallest one valid for every MDP with discount $\gamma$ and every fixed policy. Throughout this work, all contraction factors refer to $\overline { { \mathbf { M M D } } } _ { k } ^ { 2 }$ . The corresponding smallest contraction factor in $\overline { { \mathbf { M M D } } } _ { k }$ is the square root $\sqrt { \tau _ { k } ^ { \star } ( \mathcal { T } ^ { \pi } ) }$

In policy evaluation, $\tau _ { k } ^ { \star } ( \mathcal { T } ^ { \pi } ) < 1$ guarantees that the distributional Bellman equation has a unique solution $\eta ^ { \pi }$ . Repeated Bellman updates then converge to that solution in the supremum MMD at a geometric rate, $\overline { { \mathbf { M M D } } } _ { k } ^ { 2 } \big ( ( T ^ { \pi } ) ^ { n } \eta , \eta ^ { \pi } \big ) \leq \tau _ { k } ^ { \star } ( T ^ { \pi } ) ^ { n } \overline { { \mathbf { M M D } } } _ { k } ^ { 2 } ( \eta , \eta ^ { \pi } )$ for every η and every $n \geq 1$ More broadly, contraction is the standard way to show that the distributional Bellman equation is well posed under a given discrepancy [3, 4, 28].

<table><tr><td>Kernel</td><td>Formula</td><td>Contraction in  $\overline { { \mathbf { M M D } } } _ { k } ^ { 2 }$ </td></tr><tr><td>Power-distance [25]</td><td> $k _ { \alpha } ( x , y ) = - \| x - y \| ^ { \alpha } , \quad 0 < \alpha \leq 2$ </td><td> $\mathrm { Y e s , f a c t o r } \gamma ^ { \alpha }$ </td></tr><tr><td>Gaussian RBF [25]</td><td> $k _ { \mathrm { r b f } } ( x , y ) = \exp ( - h ^ { 2 } \| x - y \| ^ { 2 } )$ </td><td>No in general</td></tr><tr><td>Multiquadric [12, 19]</td><td> $k _ { \mathrm { m q } , h } ( x , y ) = - \sqrt { 1 + h ^ { 2 } \| x - y \| ^ { 2 } }$ </td><td>Yes, factor γ</td></tr></table>

Table 1: Known contraction results for specific kernels in distributional RL with MMD prior to this work. The multiquadric corresponds to the case $\beta = 1 / 2$ of the generalized multiquadric $k _ { \beta , h } ( x , y ) = - ( 1 + h ^ { \bar { 2 } } \| x - y \| ^ { 2 } ) ^ { \beta }$

Related work and known results. Existing work on distributional RL with MMD has established contraction results for a small set of kernels [19, 25]. In the univariate setting, sufficient conditions for Bellman contraction were established for kernels that are both shift invariant and scale sensitive of order $\alpha > 0$ , meaning that for all $x , y ,$ c in the underlying space [25],

$$
k ( x + c , y + c ) = k ( x , y ) , \qquad k ( c x , c y ) = | c | ^ { \alpha } k ( x , y ) .\tag{4}
$$

Table 1 summarizes the main contraction results known for specific kernels prior to this work. Distributional RL with MMD was subsequently extended to multivariate rewards [37, 38]. The earlier of the two minimized MMD but proved contraction of the multivariate distributional Bellman operator only in the Wasserstein metric [38], leaving contraction in MMD open. A later work laying the foundations of multivariate distributional RL proved contraction in MMD, but only for kernels satisfying the conditions of Equation 4 [37]. Taken together, these works identify a small set of kernels for which contraction can be proved or disproved, but do not provide a general criterion that unifies these results. Providing such a framework is among the contributions of this work.

## 3 Contraction in Maximum Mean Discrepancy

In this section, we establish our first main result, showing that contraction of the distributional Bellman operator in $\overline { { \mathbf { M M D } } } _ { k } ^ { 2 }$ reduces to a conditional positive definiteness condition on a kernel derived from k. We begin by recalling the notion of conditional positive definiteness.

Definition 3.1. Let $k : X \times X \to \mathbb { R }$ be symmetric. We say that k is conditionally positive definite (CPD), or conditionally negative definite (CND), if for every $n \geq 1$ , every $x _ { 1 } , \ldots , x _ { n } \in X$ , and every $w _ { 1 } , \ldots , w _ { n } \in \mathbb { R }$ with $\begin{array} { r } { \sum _ { i = 1 } ^ { n } { w _ { i } } = 0 } \end{array}$ , one has

$$
\sum _ { i , j = 1 } ^ { n } w _ { i } w _ { j } k ( x _ { i } , x _ { j } ) \geq 0 \qquad \mathrm { ( r e s p . ~ \leq 0 ) } .
$$

Theorem 3.1 below makes precise the reduction of Bellman contraction to a CPD condition. Conditional positive definiteness is well studied, especially for radial kernels, as we exploit in Section 4. The proof is deferred to Appendix C.

Theorem 3.1. Let $k : \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } $ R be a CPD kernel. For a reward value $q \in \mathbb { R } ^ { d }$ and a factor $\tau \in [ 0 , 1 )$ , define the Bellman difference kernel

$$
k _ { \mathrm { d i f f } } ^ { q , \gamma } ( x , y ) : = \tau k ( x , y ) - k ( q + \gamma x , q + \gamma y ) .\tag{5}
$$

Then the following two statements are equivalent. $( i ) \ T ^ { \pi }$ is a τ-contraction in $\overline { { \mathbf { M M D } } } _ { k } ^ { 2 } f o r$ every MDP with discount γ and every fixed policy. (ii) The kernel $k _ { \mathrm { d i f f } } ^ { q , \gamma }$ is CPD for every reward value $q \in \mathbb { R } ^ { d }$ . In particular, the smallest contractionfactor $\tau _ { k } ^ { \star } ( \mathcal { T } ^ { \pi } )$ is the smallest τ for which (ii) holds.

The CPD condition is therefore sufficient and necessary for $\mathcal { T } ^ { \pi }$ to be a τ-contraction for every MDP with discount $\gamma$ and every fixed policy. If it fails, there exist an MDP with discount $\gamma$ and a fixed policy for which $\mathcal { T } ^ { \pi }$ is not a τ -contraction.

## 4 Contraction with radial kernels

A kernel $k : \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } $ R is called radial if $k ( x , y ) = g ( \| x - y \| )$ ) for some function $g : [ 0 , \infty )  \mathbb { R }$ We focus on this class for two reasons. First, it covers the main kernels studied in distributional RL with MMD, see Table 1. Second, radial kernels admit powerful contraction results through Schoenberg’s theorem [30] and the theory of Bernstein functions [29], which we now develop.

Definition 4.1. A function $\phi : [ 0 , \infty )  [ 0 , \infty )$ is a Bernstein function if it admits the Lévy–Khintchine representation

$$
\phi ( t ) = a + b t + \int _ { ( 0 , \infty ) } \bigl ( 1 - e ^ { - s t } \bigr ) \Lambda ( d s ) , t \geq 0
$$

where $a , b \geq 0$ and Λ is a positive Borel measure on $( 0 , \infty )$ , called the Lévy measure of $\phi ,$ , with $\scriptstyle \int _ { ( 0 , \infty ) }$ min $\{ 1 , \stackrel { . } { s } \} \Lambda ( d s ) < \infty$ . It is a complete Bernstein function if, moreover, $\Lambda ( d s ) = m ( s )$ ds for a completely monotone density m $: ( 0 , \infty ) \to [ 0 , \infty )$ . In our setting $\phi ( 0 ) = a = 0$

![](images/0bd719ffaadb8ab2a00f44fc822abaf904022710ee11b268ee77ba4d27c816b7.jpg)  
Figure 1: Examples of Bernstein functions.

We now make the bridge from the CPD condition in Theorem 3.1 to the Bernstein reduction that follows. To this end, we state Schoenberg’s theorem on radial kernels [30], and use the elementary fact that CPD of a kernel k is equivalent to CND of −k.

Theorem 4.1 (Schoenberg [30]). A kernel of the form $( x , y ) \mapsto h ( \| x - y \| )$ , with h continuous and $h ( 0 ) = 0 ,$ is CND on $\mathbb { R } ^ { d }$ for every $d \geq 1$ if and only $i f h ( r ) = \phi ( r ^ { 2 } )$ for some Bernstein function $\phi .$

Bernstein reduction. The next theorem applies Schoenberg’s theorem to the Bellman difference kernel. Its proof is given in Appendix E.5.

Theorem 4.2. Let $k ( u , v ) = g ( \lVert u - v \rVert )$ be a radial kernel, and assume that $h ( r ) : = g ( 0 ) - g ( r )$ can be written as $h ( r ) = \phi ( r ^ { 2 } ) f o r$ a Bernsteinfunction ϕ, which we call the Bernstein profile ofk. For $\tau \in [ 0 , 1 )$ , define the Bellman difference profile

$$
\psi _ { \tau } ( t ) : = \tau \phi ( t ) - \phi ( \gamma ^ { 2 } t ) , \qquad t \geq 0 .\tag{6}
$$

$H \psi _ { \tau }$ is a Bernstein function, then $\mathcal { T } ^ { \pi }$ is a τ-contraction in $\overline { { \mathbf { M M D } } } _ { k } ^ { 2 }$ . Furthermore, this condition is necessary for τ -contraction to hold for every MDP with discount $\gamma ,$ every fixed policy, and every ambient dimension $d \geq 1$

The necessity part of Theorem 4.2 concerns every dimension at once. If $\psi _ { \tau }$ is not a Bernstein function, the Bellman difference kernel fails to be CPD in at least one dimension, but it may still be CPD in others. Theorem 3.1 then remains the right tool to check contraction in a given dimension.

Analytic reduction. Theorem 4.2 reduces Bellman contraction for radial kernels to checking whether the Bellman difference profile $\psi _ { \tau }$ is a Bernstein function. Theorem 4.3 makes this condition explicit through the Lévy–Khintchine representation of the Bernstein profile $\phi .$ In the complete Bernstein case, it also yields an explicit upper bound on the smallest contraction factor. The argument is developed in Appendix E.3–E.6.

Theorem 4.3. Adopt the setting of Theorem 4.2, and let b and Λ be the components of the $L e \nu y -$ Khintchine representation of ϕ from Definition 4.1. Then the Bellman difference profile $\psi _ { \tau }$ is a Bernsteinfunction ifand only $i f$

$$
b ( \tau - \gamma ^ { 2 } ) \geq 0 \qquad a n d \qquad \tau \Lambda ( B ) \geq \Lambda ( B / \gamma ^ { 2 } ) \quad f o r e \nu e r y B o r e l B \subset ( 0 , \infty ) ,
$$

where $B / \gamma ^ { 2 } : = \{ s > 0 : \gamma ^ { 2 } s \in B \}$ . In particular, if these conditions hold for some $\tau \in [ 0 , 1 )$ then $\tau ^ { \pi }$ is a τ -contraction in $\overline { { \mathbf { M M D } } } _ { k } ^ { 2 }$ . Furthermore, ifϕ is a complete Bernsteinfunction, so that $\Lambda ( d s ) = m ( s )$ dsfor a density $m ,$ then the smallest contractionfactor satisfies

$$
\tau _ { k } ^ { \star } ( T ^ { \pi } ) \leq \overline { { \tau } } _ { \mathrm { c b f } } ( \phi , \gamma ) , \qquad \overline { { \tau } } _ { \mathrm { c b f } } ( \phi , \gamma ) : = \operatorname* { m a x } \left\{ \gamma ^ { 2 } { \bf 1 } _ { \{ b > 0 \} } , \operatorname* { s u p } _ { s > 0 } \frac { m ( s / \gamma ^ { 2 } ) } { \gamma ^ { 2 } m ( s ) } \right\} .
$$

Theorems 4.2 and 4.3 form the backbone of the remainder of this work, unifying known contraction results under a single criterion and providing the tools that the next sections use to identify and construct new kernels that induce contraction.

Explaining known results. Applying Theorem 4.3 to the power-distance and multiquadric kernels of [12, 19, 25] recovers their contraction results from a single analytic criterion and extends the multiquadric result from $\beta = 1 / 2$ to the full generalized range $0 < \bar { \beta } \leq 1$ . In both cases the radial profile is a complete Bernstein function, so contraction reduces to evaluating $\overline { { \tau } } _ { \mathrm { c b f } } ( \phi , \gamma )$ directly. Proofs are deferred to Propositions F.1 and F.2 in the appendix.

Proposition 4.1. Consider the power-distance kernels, with parameter $0 < \alpha \leq 2 $ , and the generalized multiquadric kernels, with parameters $0 < \beta \leq 1$ and $h > 0 ;$

$$
k _ { \alpha } ( x , y ) = - \| x - y \| ^ { \alpha } , \qquad a n d \qquad k _ { \beta , h } ( x , y ) = - \big ( 1 + h ^ { 2 } \| x - y \| ^ { 2 } \big ) ^ { \beta } .
$$

Their radial profiles are generated by the Bernstein functions

$$
\phi _ { \alpha } ( t ) = t ^ { \alpha / 2 } , \qquad a n d \qquad \phi _ { \beta , h } ( t ) = ( 1 + h ^ { 2 } t ) ^ { \beta } - 1 .
$$

Moreover, both $\phi _ { \alpha }$ and $\phi _ { \beta , h }$ are complete Bernsteinfunctions, and evaluating the bound $\overline { { \tau } } _ { \mathrm { c b f } } ( \phi , \gamma )$ from Theorem 4.3 yields

$$
\begin{array} { r } { \overline { { \tau } } _ { \mathrm { c b f } } ( \phi _ { \alpha } , \gamma ) = \gamma ^ { \alpha } , \qquad a n d \qquad \overline { { \tau } } _ { \mathrm { c b f } } ( \phi _ { \beta , h } , \gamma ) = \gamma ^ { 2 \beta } . } \end{array}
$$

Hence $\mathcal { T } ^ { \pi } \ i s \ a \ \gamma ^ { \alpha }$ -contraction in $\overline { { \mathbf { M M D } } } _ { k } ^ { 2 }$ for the power-distance kernels, and $a \gamma ^ { 2 \beta }$ -contraction for the generalized multiquadric kernels.

A first obstruction: saturating profiles. Proposition 4.2 gives a first explanation for the noncontraction of the RBF kernel established in [25]. Saturating Bernstein profiles rule out contraction, and since the RBF profile shown in Figure 1 is saturating, the obstruction applies directly. The proof is given in Proposition E.3 in the appendix.

Proposition 4.2. Adopt the setting of Theorem 4.2, and assume that the Bernstein profile ϕ is saturating, meaning that lim $_ { 1  \infty } \phi ( t ) < \infty$ $I f \phi \not \equiv 0 ,$ , then there is no $\tau \in \mathsf { \Gamma } ( 0 , 1 )$ such that τ -contraction holds for every MDP with discount $\gamma$ and every fixed policy.

A second obstruction: finitely many atoms. The analytic criterion from Theorem 4.3 also rules out kernels whose Lévy measure has finitely many atoms, that is, finitely many Dirac masses, as stated in Proposition 4.3. This gives a second explanation for the failure of the RBF kernel. Its Bernstein profile is $\phi _ { \mathrm { r b f } , h } ( t ) = \bar { 1 } - \exp ( - h ^ { 2 } t )$ , whose Lévy measure $\Lambda = \delta _ { h ^ { 2 } }$ has a single atom at $h ^ { 2 }$

Proposition 4.3. Adopt the setting of Theorem 4.3, and let Λ be the Lévy measure of ϕ. If Λ has at least one but finitely many atoms, then there is no $\tau \in ( 0 , 1 )$ such that τ -contraction holds for every MDP with discount $\gamma ,$ every fixed policy, and every ambient dimension d $\geq 1$

The same argument also shows that no finite sum of RBF kernels, such as those used in [25, 38], induces contraction. Later, we will show that this obstruction disappears for suitable infinite sums of RBF kernels, which can in fact induce contraction. We refer to Lemma E.3 and Theorem E.4 in the appendix for the proof of Proposition 4.3, and to Proposition F.3 for the case of the RBF kernel.

## 5 Contraction on a bounded support

The contraction theory developed in Sections 3 and 4 requires the Bellman difference kernel to be CPD on the whole ambient space $\mathbb { R } ^ { d }$ , which can be unnecessarily restrictive in practice. The RBF kernel, which fails this global condition yet performs well empirically [25, 38], raises the question of whether contraction can hold under weaker conditions. In this section we show that under bounded rewards, CPD need only hold on the bounded region where return laws actually live.

In practical reinforcement learning settings, rewards can often be assumed bounded. In that case, the following proposition gives a bounded set that contains all discounted returns, and shows that return laws supported on this set remain supported on it after a Bellman update. The proof is given in Appendix C.2.

Proposition 5.1. Assume the rewards are bounded by $r _ { \mathrm { m a x } } ,$ , and let E be the ball of radius M centered at the origin, $E : = \{ x \in \mathbb { R } ^ { d } : \| x \| \leq M \}$ , with $M \geq r _ { \operatorname* { m a x } } / ( 1 - \gamma )$ . Then every return law $\eta ^ { \pi } ( s , a )$ is supported on $E .$ . Moreover, $i f \eta ( s , a )$ is supported on E for every $( s , a )$ , then $( \mathcal { T } ^ { \pi } \eta ) ( s , a )$ is also supported on E for every $( s , a )$

This motivates a notion of contraction restricted to return laws supported on $E ,$ , for which the difference kernel only needs to be CPD on E. The next theorem gives the corresponding criterion for Bellman contraction.

Theorem 5.1. Adopt the setting ofProposition $5 . I ,$ so that the return laws are supported on $E : =$ $\{ x \in \mathbb { R } ^ { d } : \| x \| \leq \grave { M } \}$ with $M \bar { \geq } \dot { r } _ { \mathrm { m a x } } \bar { / } ( 1 - \gamma )$ . Let k be a CPD kernel. Fix $\tau \in [ 0 , 1 )$ . For a reward value $q ,$ the Bellman difference kernel ${ \dot { k } } _ { \mathrm { d i f f } } ^ { q , \gamma }$ of Equation 5 is CPD on E if for every $n \geq 1$ , every $x _ { 1 } , \ldots , x _ { n } \in E $

$$
\sum _ { i , j = 1 } ^ { n } w _ { i } w _ { j } k _ { \mathrm { d i f f } } ^ { q , \gamma } ( x _ { i } , x _ { j } ) \geq 0 \qquad \forall w _ { 1 } , \ldots , w _ { n } \in \mathbb { R } w i t h \sum _ { i = 1 } ^ { n } w _ { i } = 0 .
$$

Then the following two statements are equivalent. $( i ) \ \mathcal { T } ^ { \pi }$ is a τ-contraction in $\overline { { \mathbf { M M D } } } _ { k } ^ { 2 }$ when restricted to return laws supported on $E ,$ for every fixed policy and every MDP with discount $\gamma$ whose rewards are bounded by $r _ { \mathrm { m a x } } . ( i i )$ The kernel $\dot { k _ { \mathrm { d i f f } } ^ { q , \gamma } }$ is CPD on $E$ for every reward value q with $\| q \| \leq r _ { \operatorname* { m a x } } .$

Under this weaker notion, kernels that fail the global condition may still be contractive on bounded sets. The proof of Theorem 5.1 is given in Appendix C.2. We now turn to testing the CPD condition on the ball E numerically, using the RBF kernel as a practical example.

Reduction to a generalized eigenvalue problem. Assuming bounded rewards, return laws are supported on a fixed bounded set $E ,$ so contraction reduces to checking that the associated Bellman difference kernel is CPD on $E$ rather than on all of $\mathbb { R } ^ { d }$ . While this condition cannot be verified numerically on all of $E ,$ , we can test it on finite subsets of $E .$ . As shown in Lemma C.5, CPD on E implies CPD on every finite subset of $E ,$ so if the CPD condition fails on some finite set $S \subseteq E$ contraction on $E$ cannot hold. Moreover, a larger finite subset gives a stricter test, because the condition on a set implies the condition on each of its subsets. On a finite set of points, computing the smallest $\tau$ for which the CPD condition holds reduces to a generalized eigenvalue problem [26], which can be solved efficiently. The next proposition makes this reduction explicit. Further background and derivation are given in Appendix D.

Proposition 5.2. Adopt the setting of Theorem 5.1, and assume that the kernel k is radial. Let $S = \bar { \{ z _ { 1 } , . . . , z _ { N } \} } \subseteq \bar { E } ,$ , let $Q \in \mathbb { R } ^ { \breve { N } \times ^ { \prime } ( N - 1 ) }$ be a matrix whose columnsform an orthonormal basis o ${ } ^ { c } \mathbf { 1 } _ { N } ^ { \perp } : = \{ w \in \mathbb { R } ^ { N } : \mathbf { 1 } _ { N } ^ { \top } w = 0 \}$ , and define

$$
K _ { i j } : = k ( z _ { i } , z _ { j } ) , \qquad ( K _ { \gamma } ) _ { i j } : = k ( \gamma z _ { i } , \gamma z _ { j } ) , \qquad A : = Q ^ { \top } K Q , \qquad B : = Q ^ { \top } K _ { \gamma } Q .
$$

Then thefollowing two statements are equivalent. (i) The Bellman difference kernel $k _ { \mathrm { d i f f } } ^ { q , \gamma }$ is CPD on $S .$ (ii) The matrix $\tau A - B$ is positive semidefinite. IfA is strictly positive definite, the smallest value of τ for which (ii) holds is the largest generalized eigenvalue $\lambda _ { \operatorname* { m a x } } ( B , A )$ ofthe pair $( B , A )$ . We call it the contraction threshold of $S \colon$

$$
\tau _ { S } ^ { \star } ( \gamma ) = \lambda _ { \operatorname* { m a x } } ( B , A ) = \operatorname* { s u p } _ { v \neq 0 } \frac { v ^ { \top } B v } { v ^ { \top } A v } .
$$

In particular, $i f \tau < \tau _ { S } ^ { \star } ( \gamma )$ for some $S \subseteq E ,$ , then, by Theorem $5 . I ,$ Bellman τ-contraction on return laws supported on E cannot hold.

Numerical experiments and contraction thresholds. We now use Proposition 5.2 to study contraction on a bounded set numerically. For each trial, we draw a set $S \overset { \triangledown } { = } \{ z _ { 1 } , \ldots , z _ { N } \}$ of $N$ points uniformly at random in the ball $B { \dot { ( } } 0 , R )$ , in dimension 1, and compute $\tau _ { S } ^ { \star } ( \gamma )$ by solving the generalized eigenvalue problem via Algorithm 1. Figure 2 reports the observed contraction threshold as a function of the discount $\gamma$ and the ratio $R / \ell .$ For the RBF kernel, the threshold stays below 1 when $R / \ell$ is small enough, which suggests that the kernel induces contraction on a ball of radius R even though it does not in general. This is numerical evidence on finite sets of points, not a proof on the whole ball. Finding a condition on $R / \ell$ that guarantees contraction remains an open question. To show that the same method applies beyond the RBF case, we also run it on the logarithmic kernel $k _ { \ell } ( x , y ) = - \log \big ( 1 + \| x - y \| ^ { \hat { 2 } } / \ell ^ { 2 } \big )$ , which is non-expansive by Proposition G.1(iii) and can become contractive on bounded supports. Further details are given in Appendix D.

![](images/fd697e7a6b6f1f97ca4aaddbaf74abfd1a60f8a53956dc97616856c491e0ad39.jpg)

![](images/4e5fc09154009b71e5371d110c23095cf97d6293cee7ac78d56589d03cb0a1bc.jpg)  
Figure 2: Contraction thresholds on a ball of radius $R ,$ for the RBF kernel $\exp ( - \| x - y \| ^ { 2 } / \ell ^ { 2 } )$ (left) and the logarithmic kernel $- \log ( 1 + \Vert x - y \Vert ^ { 2 } / \ell ^ { 2 } )$ (right). For each discount $\gamma$ and each ratio $R / \ell ,$ the color shows the largest contraction threshold $\tau _ { S } ^ { \star } ( \gamma )$ of Proposition 5.2 over 10 sets $S$ of points drawn at random in $B ( { \bar { 0 } } , R )$ , in dimension 1. In the hatched region, above the black line, this value exceeds 1, so contraction on the ball cannot hold. Where it stays below 1, the test does not rule out contraction. The RBF kernel passes the test when $R / \ell$ is small, even though it does not induce contraction on all of $\mathbb { R } ^ { d }$

## 6 New kernels inducing contraction

The criterion of Theorem 4.3 opens the door to a systematic search for kernels that induce contraction, beyond the two families previously known. Since the kernel determines which aspects of the distributions the MMD objective is sensitive to [31], a broader catalogue allows practitioners to make that choice deliberately rather than by default. In this section we develop two complementary approaches.

Searching over complete Bernstein functions. In the complete Bernstein case, Theorem 4.3 reduces the contraction test to a one-dimensional supremum that depends only on the Lévy density m(s). Standard references such as Schilling et al. [29] compile many complete Bernstein functions together with their Lévy densities. We extracted 43 entries from this reference, selected as described in Appendix G.1 and listed in Tables 3–6. Because the resulting list is too large to handle analytically case by case, we estimate the supremum numerically with the one-dimensional search summarized in Algorithm 2. Appendix G.1 validates this search against the exact contraction factors of Proposition 4.1. This search identifies 11 additional kernel families that induce contraction, collected in Table 8. Figure 8 shows their estimated contraction factors. Some of these families are parametric, and contraction holds only on part of the admissible parameter range, as Figure 9 in Appendix G.1.2 shows.

Chaining: a construction with prescribed contraction factor. In Section 4, we showed that the RBF kernel does not induce contraction. Surprisingly, a carefully constructed infinite sum of RBF kernels can. We call this construction chaining and show that given a discount factor $\gamma _ { : }$ , it can selectively induce contraction with any prescribed factor in $( \gamma ^ { 2 } , 1 )$

Theorem 6.1. Let $\tau \in ( \gamma ^ { 2 } , 1 )$ be a target contraction factor, andfix a scale $a > 0$ and a weight $c > 0 .$ . For a lengthscale $\ell > 0 ,$ , let $k _ { \mathrm { r b f } , \ell } ^ { - } ( x , y ) = \mathrm { e x p } ( - \| x - y \| ^ { 2 } / \ell ^ { 2 } )$ be the RBF kernel with lengthscale $\ell ,$ and define the chained RBF kernel

$$
k _ { \mathrm { c h a i n } } ( x , y ) : = \sum _ { n = 0 } ^ { \infty } c \tau ^ { - n } \big ( k _ { \mathrm { r b f } , \ell _ { n } } ( x , y ) - 1 \big ) , \qquad \ell _ { n } : = a ^ { - 1 / 2 } \gamma ^ { - n } .
$$

Then $\tau ^ { \pi }$ is a τ-contraction in $\overline { { \mathbf { M M D } } } _ { k _ { \mathrm { c h a i n } } } ^ { 2 }$ for every MDP with discount $\gamma$ and every fixed policy. Moreover, no smaller factor gives a contraction for every MDP with discount $\gamma ,$ every fixed policy and every ambient dimension d $\geq 1$

Its proof combines Theorem 3.1 and Proposition G.3 in the appendix. Proposition G.2 applies the same construction to kernels other than the RBF. Figures 12–13 plot the Bernstein function and the Lévy measure of the chained RBF kernel for three values of $\tau .$ . The construction combines RBF kernels whose lengthscales and weights both grow. Since $\gamma < 1$ , the lengthscales grow, $\ell _ { n + 1 } = \gamma ^ { - 1 } \ell _ { n } > \ell _ { n }$ , so each new term is sensitive to larger distances between returns. Since $\tau < 1$ the weights grow as well, $c \tau ^ { - ( n + 1 ) } = \tau ^ { - 1 } c \tau ^ { - n } > c \tau ^ { - n }$ , so each new term counts more. In practice, the sum must be truncated, and a few hundred terms are enough to make the truncation error small, as shown numerically in Appendix G.2.

The chaining construction produces a kernel that induces contraction with any prescribed factor, making it a principled new option for distributional RL with MMD. However, a faster contraction rate does not automatically make a kernel more desirable, as we discuss in Section 7.

## 7 A tradeoff perspective

The preceding sections provided tools for characterizing the smallest contraction factor of a given kernel across dimensions. One might therefore be tempted to compare kernels through this quantity alone and to conclude that the kernel with the smallest contraction factor is the most desirable. This conclusion would be misleading. Changing the kernel also changes the discrepancy itself, so a smaller contraction factor need not translate into better practical behavior when the induced MMD is less sensitive to distributional differences that matter in practice. This section makes that tension precise and quantifies it empirically through the lens of statistical testing. Within the family of kernels induced by Bernstein functions, we show that the kernels attaining the smallest contraction factors are precisely those whose induced MMD is least informative. We begin by recalling what it means for $\bar { \bf M M D _ { \boldsymbol { k } } }$ to be a metric.

Definition 7.1. A kernel k is said to induce a metric between probability distributions if $\mathbf { M M D } _ { k } ( P , Q ) = 0$ implies $P = Q$ for all $P , Q \in { \mathcal { P } } ( X )$ .

For the radial kernels considered in this work, which are built from Bernstein functions, the next two propositions show that these kernels induce a metric except in one case, where the resulting discrepancy reduces to mean comparison. The proofs are given in Appendix H and Appendix I.

Proposition 7.1. Let $k ( x , y ) = C - \phi ( \| x - y \| ^ { 2 } )$ , where ϕ is a Bernstein function with Lévy measure Λ. Then MMD<sub>k</sub> is a metric if and only $i f \Lambda \not \equiv 0$ . Equivalently, $\mathbf { M M D } _ { k }$ fails to be a metric exactly when $\phi ( t ) = b t ,$ , that is, when $k ( x , y ) = C - b \| x - y \| ^ { 2 }$

Proposition 7.2. Let $k ( x , y ) = C - b \| x - y \| ^ { 2 }$ with $b \geq 0$ . Then $\mathbf { M M D } _ { k } ^ { 2 } ( P , Q ) = 2 b \| \mathbb { E } _ { P } [ X ] -$ $\mathbb { E } _ { Q } [ X ] \lVert ^ { 2 }$ . In particular, $\mathbf { M M D } _ { k } ^ { 2 } ( P , Q ) = 0$ whenever P and Q have the same mean.

In this degenerate case, distributional policy evaluation reduces to comparing expected returns, as in classical Q-learning [36]. The next proposition is the first illustration of the tension discussed in this section, as it shows that the smallest possible contraction factor $\gamma ^ { 2 }$ is attained only in this degenerate case. Its proof is given in the appendix as Proposition H.2.

Proposition 7.3. Let $k ( x , y ) = C - \phi ( \| x - y \| ^ { 2 } )$ , where ϕ is a Bernsteinfunction with Lévy measure $\Lambda . \bar { I f } \Lambda \not \equiv 0 ,$ , that is, $i f \mathbf { M M D } _ { k }$ is a metric (Proposition $7 . l ) ,$ , then $\tau _ { k } ^ { \star } ( \mathcal { T } ^ { \pi } ) > \gamma ^ { 2 }$ , so the kernel cannot reach the smallest possible contractionfactor $\bar { \gamma ^ { 2 } }$ . Equivalently, this value can only be reached by the kernel $k ( x , y ) = \dot { C } - b \| x - y \| ^ { 2 }$

Empirical detectability. Beyond its role as a discrepancy, MMD can be used as a test statistic to decide whether two distributions differ [7, 15, 16]. Run with each kernel, such a test offers a principled way to compare kernels in practice, through the probability that it detects a difference. In distributional policy evaluation, this means detecting a statistically significant difference between predicted and target return laws from a limited number of samples.

Definition $7 . 2 . \mathrm { ~ A ~ }$ test $\Delta _ { k } ( X _ { n } , Y _ { m } ) \in \{ 0 , 1 \}$ uses an empirical MMD statistic to test $H _ { 0 } \colon P = Q$ against $H _ { 1 } \colon P \neq Q$ , with $\Delta _ { k } = 1$ meaning rejection. It has level α if $\begin{array} { r } { \mathrm { s u p } _ { P } \mathbb { P } _ { P \times P } ( \Delta _ { k } = 1 ) \le \alpha } \end{array}$ For $P \neq Q ,$ , its power is $\mathbb { P } _ { P \times Q } ( \Delta _ { k } = 1 )$ .

Motivated by Proposition 7.3, we make the following empirical claim: For a fixed discount γ and a parametric kernel family k<sub>θ</sub>, a smaller contraction factor on a ball E containing the data, $\tau _ { k _ { \theta _ { 1 } } , E } ^ { \star } ( \gamma ) < \tau _ { k _ { \theta _ { 2 } } , E } ^ { \star } ( \gamma )$ , empirically comes with reduced power to distinguish return laws.

![](images/eed88d2e1bebb417038fffbd2d4f641c0bd72b0c5c91d118e9fb6784dfe6f6c6.jpg)  
Figure 3: One-dimensional tradeoff experiments. The left column shows the clipped task distributions, with dashed lines marking the clipping radius $R .$ . The remaining columns show, for each kernel family, the best empirical test power achievable under the constraint $\tau _ { k _ { \theta } , E } ^ { \star } ( \gamma ) \leq \bar { \tau }$ . The dotted curve is the smallest possible contraction factor $\gamma ^ { 2 }$

To test this claim, we fix two distributions $P$ and $Q$ and measure how reliably a test of level α built from the empirical MMD can detect their difference. We use a permutation test [16] and estimate its power for each kernel configuration. The full protocol is described in Appendix H.1.

We consider several kernel families $k _ { \theta }$ , with parameter $\theta \in \Theta$ . We vary $\theta$ within Θ and, for each resulting kernel and each discount $\gamma _ { : }$ , use the procedure of Section 5 to estimate $\tau _ { k _ { \theta } , E } ^ { \star } ( \gamma )$ , the smallest contraction factor of $k _ { \theta }$ on $E .$ . We also estimate the power of each kernel with the permutation test. For each discount $\gamma$ and each bound $\bar { \tau }$ on the contraction factor, we then select, within each family, the kernel with the highest power among those with $\tau _ { k _ { \theta } , E } ^ { \star } ( \gamma ) \leq \bar { \tau }$ . This shows how each kernel family trades contraction against empirical detectability.

Results are shown in Figure 3 for the RBF, logarithmic, generalized multiquadric, and power-distance kernels. The tradeoff depends both on the distributions being compared and on the kernel family. When the two distributions differ only in their mean, the tradeoff essentially disappears across all kernels. Comparing means is then enough, and each family has parameter regimes close to comparing means only. When they have the same mean but different spreads, or when one is bimodal and the other unimodal, the tradeoff becomes much stronger, especially for some kernel families, showing that a contraction factor close to $\gamma ^ { 2 }$ can come at a substantial loss of power. Details about the experiments and additional two-dimensional examples, shown in Figure ${ \bar { 1 7 } } ,$ are provided in Appendix H. In Appendix H.3, we show that these observations carry over to dimensions up to 64, where the tradeoff stays negligible for a mean shift and grows for a checkerboard, until the dimension is too large for 100 samples per distribution to reveal the difference with any kernel.

## 8 Conclusion and Further Directions

This work develops a mathematical framework for studying contraction of the distributional Bellman operator under MMD. The central result is a general reduction showing that Bellman contraction is equivalent to the conditional positive definiteness of a kernel derived from the base kernel. For radial kernels, this criterion reduces further to a one-dimensional condition on the Lévy measure of the Bernstein function that generates the kernel, which can be verified analytically or tested numerically. The framework simultaneously explains and unifies existing results [12, 19, 25, 38] and yields new ones. We recover contraction results for the power-distance and multiquadric kernels, identify numerically 11 additional kernel families that induce contraction, and introduce the chaining construction, which produces a kernel that induces contraction with any prescribed factor by aggregating RBF kernels.

We also study the RBF kernel, which does not induce contraction in general, yet performs well in practice [19, 25, 38]. When rewards are bounded, return laws remain supported on a ball, and contraction only requires the Bellman difference kernel to be conditionally positive definite on that ball. We test this condition on random finite sets of points in the ball, where it reduces to a generalized eigenvalue problem. The results suggest that the RBF kernel induces contraction on balls whose radius is small relative to the lengthscale, which may explain part of its success.

We further show that contraction rate and discriminative power are in tension. Among the radial kernels of this work, those with the smallest contraction factors are those whose MMD is least sensitive to differences beyond the mean. This tradeoff suggests that the contraction rate alone is not a sufficient criterion for kernel selection in distributional RL, and that practitioners should also consider discriminative power.

Finally, this work leaves several further directions open. We focused on contraction in distributional policy evaluation. The natural next step is to run experiments on RL environments and compare kernels on control tasks. The tradeoff between contraction rate and discriminative power depends on the distributions being compared. In distributional RL, these are the predicted and target return laws, which depend on the environment and on the policy. Therefore, measuring the tradeoff on the return laws that arise during training could guide kernel selection.

## Acknowledgments and Disclosure of Funding

This project has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement n° 101021347).

## References

[1] Nachman Aronszajn. Theory of reproducing kernels. Transactions ofthe American mathematical society, 68(3):337–404, 1950.

[2] Gabriel Barth-Maron, Matthew W Hoffman, David Budden, Will Dabney, Dan Horgan, Dhruva Tb, Alistair Muldal, Nicolas Heess, and Timothy Lillicrap. Distributed distributional deterministic policy gradients. arXiv preprint arXiv:1804.08617, 2018.

[3] Marc G Bellemare, Will Dabney, and Rémi Munos. A distributional perspective on reinforcement learning. In International conference on machine learning, pages 449–458. Pmlr, 2017.

[4] Marc G Bellemare, Will Dabney, and Mark Rowland. Distributional reinforcement learning. MIT Press, 2023.

[5] Christian Berg, Jens Peter Reus Christensen, and Paul Ressel. Harmonic analysis on semigroups: theory ofpositive definite and relatedfunctions. Springer, 1984.

[6] Alain Berlinet and Christine Thomas-Agnan. Reproducing kernel Hilbert spaces in probability and statistics. Springer Science & Business Media, 2011.

[7] Karsten M Borgwardt, Arthur Gretton, Malte J Rasch, Hans-Peter Kriegel, Bernhard Schölkopf, and Alex J Smola. Integrating structured biological data by kernel maximum mean discrepancy. Bioinformatics, 22 (14):e49–e57, 2006.

[8] James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Yash Katariya, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. JAX: composable transformations of Python+NumPy programs, 2018. URL http://github.com/jax-ml/ jax.

[9] John B Conway. A course infunctional analysis. Springer, 1990.

[10] Will Dabney, Georg Ostrovski, David Silver, and Rémi Munos. Implicit quantile networks for distributional reinforcement learning. In International conference on machine learning, pages 1096–1105. PMLR, 2018.

[11] Will Dabney, Mark Rowland, Marc Bellemare, and Rémi Munos. Distributional reinforcement learning with quantile regression. In Proceedings ofthe AAAI conference on artificial intelligence, volume 32, 2018.

[12] Baptiste Debes and Tinne Tuytelaars. Distributional value gradients for stochastic environments. In International Conference on Learning Representations, 2026.

[13] Baptiste Debes and Tinne Tuytelaars. Multivariate distributional reinforcement learning using sliced divergences. In International Conference on Machine Learning, volume 306, pages 23456–23526. PMLR, 2026.

[14] Jean Feydy, Thibault Séjourné, François-Xavier Vialard, Shun-ichi Amari, Alain Trouvé, and Gabriel Peyré. Interpolating between optimal transport and MMD using Sinkhorn divergences. In The 22nd international conference on artificial intelligence and statistics, pages 2681–2690. PMLR, 2019.

[15] Arthur Gretton, Karsten Borgwardt, Malte Rasch, Bernhard Schölkopf, and Alex Smola. A kernel method for the two-sample-problem. Advances in neural information processing systems, 19, 2006.

[16] Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard Schölkopf, and Alexander Smola. A kernel two-sample test. Journal of Machine Learning Research, 13(25):723–773, 2012. URL http: //jmlr.org/papers/v13/gretton12a.html.

[17] Jean Carlo Guella. Generalization of the energy distance by Bernstein functions, 2021. URL https: //arxiv.org/abs/2102.00633.

[18] Matteo Hessel, Joseph Modayil, Hado Van Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney, Dan Horgan, Bilal Piot, Mohammad Azar, and David Silver. Rainbow: Combining improvements in deep reinforcement learning. In Proceedings ofthe AAAI conference on artificial intelligence, volume 32, 2018.

[19] Ludvig Killingberg and Helge Langseth. The multiquadric kernel for moment-matching distributional reinforcement learning. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https://openreview.net/forum?id=z49eaB8kiH.

[20] Erwin Kreyszig. Introductoryfunctional analysis with applications. John Wiley & Sons, 1989.

[21] Clare Lyle, Marc G Bellemare, and Pablo Samuel Castro. A comparative analysis of expected and distributional reinforcement learning. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 33, pages 4504–4511, 2019.

[22] Charles A Micchelli. Interpolation of scattered data: distance matrices and conditionally positive definite functions. Constructive Approximation, 2:11–22, 1986.

[23] Tetsuro Morimura, Masashi Sugiyama, Hisashi Kashima, Hirotaka Hachiya, and Toshiyuki Tanaka. Nonparametric return distribution approximation for reinforcement learning. In Proceedings ofthe 27th International Conference on Machine Learning (ICML-10), pages 799–806, 2010.

[24] The mpmath development team. mpmath: a Python library for arbitrary-precision floating-point arithmetic (version 1.4.0), 2026. http://mpmath.org/.

[25] Thanh Nguyen-Tang, Sunil Gupta, and Svetha Venkatesh. Distributional reinforcement learning via moment matching. In Proceedings of the AAAI conference on artificial intelligence, volume 35, pages 9144–9152, 2021.

[26] Beresford N Parlett. The symmetric eigenvalue problem. SIAM, 1998.

[27] Mark Rowland, Marc Bellemare, Will Dabney, Rémi Munos, and Yee Whye Teh. An analysis of categorical distributional reinforcement learning. In International Conference on Artificial Intelligence and Statistics, pages 29–37. PMLR, 2018.

[28] Mark Rowland, Robert Dadashi, Saurabh Kumar, Rémi Munos, Marc G Bellemare, and Will Dabney. Statistics and samples in distributional reinforcement learning. In International Conference on Machine Learning, pages 5528–5536. PMLR, 2019.

[29] René L Schilling, Renming Song, and Zoran Vondracek. ˇ Bernstein functions: theory and applications. Walter de Gruyter, 2009.

[30] Isaac J Schoenberg. Metric spaces and positive definite functions. Transactions ofthe American Mathematical Society, 44(3):522–536, 1938.

[31] Dino Sejdinovic, Bharath Sriperumbudur, Arthur Gretton, and Kenji Fukumizu. Equivalence of distancebased and RKHS-based statistics in hypothesis testing. The Annals ofStatistics, pages 2263–2291, 2013.

[32] Rahul Singh, Qinsheng Zhang, and Yongxin Chen. Improving robustness via risk averse distributional reinforcement learning. In Learningfor Dynamics and Control, pages 958–968. PMLR, 2020.

[33] Rahul Singh, Keuntaek Lee, and Yongxin Chen. Sample-based distributional policy gradient. In Learning for Dynamics and Control Conference, pages 676–688. PMLR, 2022.

[34] Ke Sun, Yingnan Zhao, Wulong Liu, Bei Jiang, and Linglong Kong. Distributional reinforcement learning with regularized Wasserstein loss. Advances in Neural Information Processing Systems, 37:63184–63221, 2024.

[35] Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, <sup>˙</sup>Ilhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17:261–272, 2020. doi: 10.1038/s41592-019-0686-2.

[36] Christopher JCH Watkins and Peter Dayan. Q-learning. Machine learning, 8(3):279–292, 1992.

[37] Harley Wiltzer, Jesse Farebrother, Arthur Gretton, and Mark Rowland. Foundations of multivariate distributional reinforcement learning. Advances in Neural Information Processing Systems, 37:101297– 101336, 2024.

[38] Pushi Zhang, Xiaoyu Chen, Li Zhao, Wei Xiong, Tao Qin, and Tie-Yan Liu. Distributional reinforcement learning for multi-dimensional reward functions. Advances in Neural Information Processing Systems, 34: 1519–1529, 2021.

## Appendix

A Background on Maximum Mean Discrepancy 14   
A.1 Reproducing kernel Hilbert spaces 14   
A.2 Maximum mean discrepancy 15   
A.3 Gram matrices and empirical MMD 16   
B Basic identities for MMD<sup>2</sup> 18   
C Bellman contraction in Maximum Mean Discrepancy 20   
C.1 Contraction via difference kernel 20   
C.2 Contraction on a bounded support 25   
D Contraction on a bounded support: numerical experiments 28   
D.1 From a bounded to a discrete support condition 28   
D.2 From a support condition to a matrix test . 28   
D.3 Experimental setup 31   
D.4 Results . 31   
E Global contraction under radial kernels via Bernstein functions 34   
E.1 Motivations for radial kernels 34   
E.2 Definitions and Schoenberg’s theorem 35   
E.3 Bernstein reduction of difference kernel 36   
E.4 Results on measures and the measure domination condition 39   
E.5 From Bernstein reduction to kernels inducing contraction 47   
E.6 Contraction from complete Bernstein functions 49   
E.7 Chaining as a general mechanism . 50   
F Explaining known results 57   
G Some new kernels inducing contraction 61   
G.1 From complete Bernstein functions 61   
G.2 From geometric chaining 73   
H Contraction rate and discriminability 78   
H.1 Permutation test and power estimation 80   
H.2 Estimating the tradeoff between power and contraction 81   
H.3 Higher dimensions 85   
I Properties of radial kernels built from Bernstein functions 86

## A Background on Maximum Mean Discrepancy

## A.1 Reproducing kernel Hilbert spaces

Definition A.1. A Hilbert space H is a vector space of functions equipped with an inner product $\langle u , v \rangle _ { \mathcal { H } }$ . The inner product measures angles between functions, and the norm it induces, $\| \bar { \boldsymbol { v } } \| _ { \mathcal { H } } : =$ $\sqrt { \langle v , v \rangle _ { \mathcal { H } } }$ , measures their lengths.

For the formal definition, which also requires completeness, see [9, Ch. 1] or [20, Ch. 3].

Definition A.2 (Kernel). Let X be a nonempty set. A kernel on X is any function k : $X \times X \to \mathbb { R }$ We say k is symmetric if $k ( x , y ) = k ( y , x )$ for all $x , y \in X$

Definition A.3 (Positive Semidefinite Kernel [6, Ch. 1]). Let X be a nonempty set and k : $X \times X $ R be symmetric. We say k is positive semidefinite (PSD) if for every integer $n \geq 1$ , any points $x _ { 1 } , \ldots , x _ { n } \in X$ and real coefficients $c _ { 1 } , \ldots , c _ { n }$

$$
\sum _ { i , j = 1 } ^ { n } c _ { i } c _ { j } k ( x _ { i } , x _ { j } ) \geq 0 .
$$

Equivalently,

$$
\iiint _ { X \times X } k ( x , y ) d \sigma ( x ) d \sigma ( y ) \geq 0
$$

for every finite signed measure σ on X for which this integral is well defined.

Definition A.4 (Reproducing Kernel and RKHS [1, p. 343]). A function k : $X \times X \to$ R is called a reproducing kernel if there exists a Hilbert space H of functions $f \colon X \to$ R such that $k ( x , \cdot ) \in \mathcal { H }$ for each $x \in X$ , and such that the reproducing property holds,

$$
f ( x ) = \left. f , k ( x , \cdot ) \right. _ { \mathcal { H } } \qquad { \mathrm { f o r ~ a l l ~ } } f \in { \mathcal { H } } { \mathrm { ~ a n d ~ } } x \in X .
$$

The Hilbert space H is uniquely determined by k and is called the reproducing kernel Hilbert space (RKHS) of $k .$

Proposition A.1 (Reproducing kernels are the symmetric PSD kernels [1, p. 344]). A function $k \colon \hat { X } \times X  \mathbb { R }$ is a reproducing kernel if and only if it is symmetric and positive semidefinite.

Feature map and mean embeddings. Let H be the RKHS of k. Then there is a canonicalfeature map

$$
\Phi : X \to { \mathcal { H } } , \qquad \Phi ( x ) = k ( x , \cdot ) , \qquad k ( x , x ^ { \prime } ) = \langle \Phi ( x ) , \Phi ( x ^ { \prime } ) \rangle _ { { \mathcal { H } } } \quad \forall x , x ^ { \prime } \in X .
$$

For any $P \in { \mathcal { P } } ( X )$ , the set of probability measures on $X ,$ the mean embedding of $P$ is

$$
\mu _ { P } : = \int _ { X } \Phi ( x ) d P ( x ) \in \mathcal { H } .
$$

Since $\mu _ { P } \in \mathcal { H } .$ , we may apply the reproducing property. For any $x ^ { \prime } \in X$

$$
\begin{array} { r l } { \displaystyle \mu _ { P } ( \boldsymbol { x } ^ { \prime } ) = \langle \mu _ { P } , k ( \boldsymbol { x } ^ { \prime } , \cdot ) \rangle _ { \mathcal { H } } } & { \mathrm { ( r e p r o d u c i n g ~ p r o p e r t y ) } } \\ { \displaystyle } & { = \left. \int _ { X } \Phi ( \boldsymbol { x } ) d P ( \boldsymbol { x } ) , k ( \boldsymbol { x } ^ { \prime } , \cdot ) \right. _ { \mathcal { H } } \quad \mathrm { ( d e f i n i t i o n ~ o f ~ } \mu _ { P } \mathrm { ) } } \\ { \displaystyle } & { = \int _ { X } \langle \Phi ( \boldsymbol { x } ) , k ( \boldsymbol { x } ^ { \prime } , \cdot ) \rangle _ { \mathcal { H } } d P ( \boldsymbol { x } ) \quad \mathrm { ( l i n e a r i t y ~ o f ~ i n n e r ~ p r o d u c t ) } } \\ { \displaystyle } & { = \int _ { X } k ( \boldsymbol { x } ^ { \prime } , \boldsymbol { x } ) d P ( \boldsymbol { x } ) \quad \mathrm { ( r e p r o d u c i n g ~ p r o p e r t y ) } } \\ { \displaystyle } & { = \mathbb { E } _ { \boldsymbol { x } \sim P } \big [ k ( \boldsymbol { x } ^ { \prime } , \boldsymbol { x } ) \big ] \quad \mathrm { ( d e f i n i t i o n ~ o f ~ e x p e c t a t i o n ) } . } \end{array}
$$

Hence $\mu _ { P } ( x ^ { \prime } )$ can be evaluated without computing the feature map, using kernel values alone, by averaging $k ( x ^ { \prime } , x )$ over points x drawn from P.

Expectation as an inner product. The RKHS turns integration under $P$ into a single inner product against the mean embedding $\mu _ { P }$

$$
\begin{array} { r l } { \mathbb { E } _ { x \sim P } [ f ( x ) ] = \displaystyle \int _ { X } f ( x ) d P ( x ) } & { \mathrm { ( d e f i n i t i o n ~ o f ~ e x p e c t a t i o n ) } } \\ { = \displaystyle \int _ { X } \langle f , k ( x , \cdot ) \rangle _ { \mathcal { H } } d P ( x ) } & { \mathrm { ( r e p r o d u c i n g ~ p r o p e r t y ) } } \\ { = \Big \langle f , \displaystyle \int _ { X } k ( x , \cdot ) d P ( x ) \Big \rangle _ { \mathcal { H } } } & { \mathrm { ( l i n e a r i t y ~ o f ~ i n n e r ~ p r o d u c t ) } } \\ { = \langle f , \mu _ { P } \rangle _ { \mathcal { H } } } & { \mathrm { ( d e f i n i t i o n ~ o f ~ } \mu _ { P } ) . } \end{array}
$$

Distance between feature maps. Since $k ( x , y ) = \langle \Phi ( x ) , \Phi ( y ) \rangle _ { \mathcal { H } }$ is an inner product, the squared distance in H between feature maps is

$$
\begin{array} { r l } {  { \big \| \Phi ( x ) - \Phi ( y ) \big \| _ { \mathcal { H } } ^ { 2 } = \big \langle \Phi ( x ) - \Phi ( y ) , \Phi ( x ) - \Phi ( y ) \big \rangle _ { \mathcal { H } } } } \\ & { \quad = \big \langle \Phi ( x ) , \Phi ( x ) \big \rangle _ { \mathcal { H } } - 2 \big \langle \Phi ( x ) , \Phi ( y ) \big \rangle _ { \mathcal { H } } + \big \langle \Phi ( y ) , \Phi ( y ) \big \rangle _ { \mathcal { H } } } \\ & { \quad = k ( x , x ) - 2 k ( x , y ) + k ( y , y ) . } \end{array}
$$

## A.2 Maximum mean discrepancy

Definition A.5 (CPD and CSPD [5, Def. 3.1.1], [19, Def. 3.2]). Let X be a measurable space and let $k : X \times X \to$ R be symmetric. We say that k is conditionally positive definite (CPD) if

$$
\iiint _ { X \times X } k ( x , y ) d \sigma ( x ) d \sigma ( y ) \geq 0
$$

for every finite signed measure σ on $X$ with $\sigma ( \boldsymbol { X } ) = 0$ for which this integral is well defined. If the inequality is strict for every nonzero such $\sigma ,$ then k is conditionally strictly positive definite (CSPD). For continuous kernels, CPD is equivalent to the condition with finite sums of Definition 3.1 [17, Lemma 3.4].

Definition A.6 (Maximum mean discrepancy). Let X be a measurable space and let k : $X \times X \to$ R be symmetric.

• If k is a reproducing (positive semidefinite) kernel with RKHS $\mathcal { H } _ { k }$ , then for $P , Q \in { \mathcal { P } } ( X )$ the squared maximum mean discrepancy [16, Def. 2 and Lemma 4] is defined by

$$
\mathbf { M M D } _ { k } ^ { 2 } ( P , Q ) : = \left\| \mu _ { P } - \mu _ { Q } \right\| _ { \mathcal { H } _ { k } } ^ { 2 } , \qquad \mu _ { P } : = \int _ { X } k ( x , \cdot ) d P ( x ) .
$$

• More generally, if k is CPD (Definition A.5), we define the squared maximum mean discrepancy by the quadratic form [31, Eq. (2.6) and Thm. 22]

$$
\mathbf { M M D } _ { k } ^ { 2 } ( P , Q ) : = \iint _ { X \times X } k ( x , y ) d ( P - Q ) ( x ) d ( P - Q ) ( y ) ,
$$

whenever the integral is well defined. Equivalently, writing $x , x ^ { \prime } \sim P$ and $y , y ^ { \prime } \sim Q$ , all independent, one has

$$
\begin{array} { r } { \mathbf { M } \mathbf { M } \mathbf { D } _ { k } ^ { 2 } ( P , Q ) = \mathbb { E } k ( x , x ^ { \prime } ) + \mathbb { E } k ( y , y ^ { \prime } ) - 2 \mathbb { E } k ( x , y ) , } \end{array}
$$

whenever the expectations are finite.

When $k$ is a reproducing kernel, the expressions as a quadratic form and as expectations coincide with the RKHS definition [16, Lemma 6].

Proposition A.2 (Centering CPD kernels [31]). Let X be a measurable space and let k : $X \times X \to \mathbb { R }$ be symmetric and CPD. Fix $x _ { 0 } \in X$ and define the kernel centered at $x _ { 0 }$

$$
k ^ { \circ } ( x , y ) : = k ( x , y ) - k ( x , x _ { 0 } ) - k ( x _ { 0 } , y ) + k ( x _ { 0 } , x _ { 0 } ) .
$$

Then $k ^ { \circ }$ is positive semidefinite and hence admits an RKHS. Moreover, for any $P , Q \in { \mathcal { P } } ( X )$ such that the expressions below are well defined,

$$
\boxed { \mathbf { M M D } _ { k } ^ { 2 } ( P , Q ) = \iint _ { X \times X } k ^ { \circ } ( x , y ) d ( P - Q ) ( x ) d ( P - Q ) ( y ) = \mathbf { M M D } _ { k ^ { \circ } } ^ { 2 } ( P , Q ) . }
$$

Definition A.7 (Characteristic kernel). A reproducing kernel k is characteristic if

$$
\mathbf { M M D } _ { k } ( P , Q ) = 0 \quad \Longrightarrow \quad P = Q \qquad { \mathrm { f o r ~ a l l ~ } } P , Q \in { \mathcal { P } } ( X ) .
$$

Proposition A.3 (MMD is a metric under CSPD). Let X be a measurable space and let $k : X \times$ $X \to \mathbb { R }$ be symmetric and CPD. Assume $\mathbf { M M D } _ { k } ^ { 2 } ( P , Q )$ is defined by Definition A.6 for the pairs considered, and set $ { \mathbf { M } }  { \mathbf { M } }  { \mathbf { D } } _ { k } ( P , Q ) : = \sqrt {  { \mathbf { M } }  { \mathbf { M } }  { \mathbf { D } } _ { k } ^ { 2 } ( P , Q ) }$ . Then $\mathbf { M M D } _ { k }$ satisfies the metric axioms, except possibly identity of indiscernibles. More precisely, for any $P , Q , R \in { \mathcal { P } } ( X )$ such that all the quantities below are finite, the following hold.

1. Nonnegativity: $\mathbf { M M D } _ { k } ( P , Q ) \geq 0 .$

2. Symmetry: $\mathbf { M M D } _ { k } ( P , Q ) = \mathbf { M M D } _ { k } ( Q , P )$

3. Triangle inequality: $\mathbf { M M D } _ { k } ( P , Q ) \leq \mathbf { M M D } _ { k } ( P , R ) + \mathbf { M M D } _ { k } ( R , Q )$

4. Identity of indiscernibles:

$$
\mathrm { \bf M M D } _ { k } ( P , Q ) = 0 \Rightarrow P = Q \quad \mathrm { h o l d s ~ i f ~ a n d ~ o n l y ~ i f } \quad k \mathrm { ~ i s ~ C S P D } .
$$

Proof. When k is a reproducing kernel, MMD is a distance in the RKHS norm [16, Lemma 4], hence it satisfies nonnegativity, symmetry, and the triangle inequality. For a CPD kernel k, Proposition $\mathbf { A } . 2 { \mathrm { ~ y i e l d s ~ a ~ } }$ centered positive semidefinite kernel $k ^ { \circ }$ such that $\mathbf { M } \mathbf { M } \mathbf { D } _ { k } ^ { 2 } = \mathbf { M } \mathbf { M } \mathbf { D } _ { k ^ { \circ } } ^ { 2 }$ , so $\mathbf { M M D } _ { k } = \mathbf { M M D } _ { k ^ { \circ } }$ is again an RKHS norm distance. Moreover, identity of indiscernibles holds exactly when k is CSPD [31]. □

## A.3 Gram matrices and empirical MMD

General formulation for discrete signed measures. Fix points $z _ { 1 } , \dots , z _ { N } \in X$ and coefficients $w \in \mathbb { R } ^ { N }$ , and define the finite signed measure

$$
\sigma : = \sum _ { i = 1 } ^ { N } w _ { i } \delta _ { z _ { i } } .
$$

Let $K \in \mathbb { R } ^ { N \times N }$ be the Gram matrix $K _ { i j } : = k ( z _ { i } , z _ { j } )$ . Then

$$
\iint _ { X \times X } k ( x , y ) d \sigma ( x ) d \sigma ( y ) = \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { N } w _ { i } w _ { j } k ( z _ { i } , z _ { j } ) = w ^ { \top } K w .
$$

In particular, $\begin{array} { r } { \mathrm { i f } \sum _ { i = 1 } ^ { N } w _ { i } = 0 } \end{array}$ and k is CPD in the sense of Definition A.5, then $w ^ { \top } K w \ge 0$

Two-sample empirical estimator. Given samples $x _ { 1 : n } \ = \ ( x _ { 1 } , . . . , x _ { n } )$ from P and $y _ { 1 : m } =$ $\left( y _ { 1 } , \ldots , y _ { m } \right)$ from $Q ,$ define the Gram matrices

$$
\begin{array} { r l r l } & { K _ { x x } \in \mathbb { R } ^ { n \times n } , } & & { ( K _ { x x } ) _ { i j } : = k ( x _ { i } , x _ { j } ) , } \\ & { K _ { y y } \in \mathbb { R } ^ { m \times m } , } & & { ( K _ { y y } ) _ { i j } : = k ( y _ { i } , y _ { j } ) , } \\ & { K _ { x y } \in \mathbb { R } ^ { n \times m } , } & & { ( K _ { x y } ) _ { i j } : = k ( x _ { i } , y _ { j } ) . } \end{array}
$$

Let $\mathbf { 1 } _ { n } \in \mathbb { R } ^ { n }$ denote the vector of ones, and similarly $\mathbf { 1 } _ { m } \in \mathbb { R } ^ { m }$

The standard biased empirical estimator of $\mathbf { M M D } _ { k } ^ { 2 } ( P , Q )$ is

$$
\widehat { \mathbf { M M D } } _ { k , \mathrm { b } } ^ { 2 } ( x _ { 1 : n } , y _ { 1 : m } ) : = \frac { 1 } { n ^ { 2 } } \sum _ { i , j = 1 } ^ { n } k ( x _ { i } , x _ { j } ) + \frac { 1 } { m ^ { 2 } } \sum _ { i , j = 1 } ^ { m } k ( y _ { i } , y _ { j } ) - \frac { 2 } { n m } \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { m } k ( x _ { i } , y _ { j } ) ,
$$

which can be written with Gram matrices as

$$
\widehat { \mathbf { M M D } } _ { k , \mathrm { b } } ^ { 2 } ( x _ { 1 : n } , y _ { 1 : m } ) = \frac { 1 } { n ^ { 2 } } \mathbf { 1 } _ { n } ^ { \top } K _ { x x } \mathbf { 1 } _ { n } + \frac { 1 } { m ^ { 2 } } \mathbf { 1 } _ { m } ^ { \top } K _ { y y } \mathbf { 1 } _ { m } - \frac { 2 } { n m } \mathbf { 1 } _ { n } ^ { \top } K _ { x y } \mathbf { 1 } _ { m } .
$$

The estimator can also be written with a single Gram matrix, built from all $n + m$ points. Let $N : = n + m$ and define the aggregate sample $z _ { 1 : N } = ( z _ { 1 } , \dots , z _ { N } )$ by

$$
z _ { i } : = x _ { i } \quad ( 1 \leq i \leq n ) , \qquad z _ { n + j } : = y _ { j } \quad ( 1 \leq j \leq m ) .
$$

Define the two-sample weight vector $\boldsymbol { \ell } \in \mathbb { R } ^ { N }$ by

$$
\ell : = \left( { \begin{array} { c } { { \frac { 1 } { n } } \mathbf { 1 } _ { n } } \\ { - { \frac { 1 } { m } } \mathbf { 1 } _ { m } } \end{array} } \right) , \qquad \mathrm { s o ~ t h a t ~ } \mathbf { 1 } _ { N } ^ { \top } \ell = 0 ,
$$

and let $K \in \mathbb { R } ^ { N \times N }$ be the Gram matrix of the aggregate sample, $K _ { i j } : = k ( z _ { i } , z _ { j } )$ , which has the blocks

$$
K = \left( \begin{array} { l l } { K _ { x x } } & { K _ { x y } } \\ { K _ { x y } ^ { \top } } & { K _ { y y } } \end{array} \right) .
$$

Multiplying out $\ell ^ { \top } K \ell$ block by block gives the expression with Gram matrices above, so

$$
\widehat { \mathbf { M M D } } _ { k , \mathrm { b } } ^ { 2 } ( x _ { 1 : n } , y _ { 1 : m } ) = \ell ^ { \top } K \ell .
$$

Discrete measures on a common support. Let $S = \{ z _ { 1 } , \dots , z _ { N } \} \subset X$ and consider two probability measures supported on S,

$$
\mu = \sum _ { i = 1 } ^ { N } a _ { i } \delta _ { z _ { i } } , \qquad \nu = \sum _ { i = 1 } ^ { N } b _ { i } \delta _ { z _ { i } } , \qquad \sum _ { i = 1 } ^ { N } a _ { i } = \sum _ { i = 1 } ^ { N } b _ { i } = 1 .
$$

Define $w \in \mathbb { R } ^ { N }$ by $w _ { i } : = a _ { i } - b _ { i }$ , so that $\mathbf { 1 } _ { N } ^ { \top } w = 0$ , and set $\begin{array} { r } { \sigma : = \mu - \nu = \sum _ { i = 1 } ^ { N } w _ { i } \delta _ { z _ { i } } } \end{array}$ . Then the same identity for discrete measures yields

$$
\mathbf { M M D } _ { k } ^ { 2 } ( \mu , \nu ) = \iint _ { X \times X } k ( x , y ) d ( \mu - \nu ) ( x ) d ( \mu - \nu ) ( y ) = w ^ { \top } K w , \qquad K _ { i j } : = k ( z _ { i } , z _ { j } ) .
$$

The constraint $\mathbf { 1 } _ { N } ^ { \top } w = 0$ is the discrete counterpart of the condition $( \mu - \nu ) ( X ) = 0$

Centering and restriction to vectors whose entries sum to zero. In several arguments in this paper, we must control quadratic forms $w ^ { \top } K w$ for all vectors w whose entries sum to zero. It is convenient to encode the constraint $\mathbf { 1 } _ { N } ^ { \top } w = 0$ using the centering projector

$$
H _ { N } : = I _ { N } - \frac 1 N \mathbf { 1 } _ { N } \mathbf { 1 } _ { N } ^ { \top } ,
$$

which satisfies $H _ { N } w = w$ whenever $\mathbf { 1 } _ { N } ^ { \top } w = 0$ . Consequently, for any such w,

$$
w ^ { \top } K w = ( H _ { N } w ) ^ { \top } K ( H _ { N } w ) = w ^ { \top } ( H _ { N } K H _ { N } ) w .
$$

Moreover, the condition

$$
w ^ { \top } K w \geq 0 \qquad { \mathrm { f o r ~ a l l ~ } } w \in \mathbb { R } ^ { N } { \mathrm { ~ w i t h ~ } } \mathbf { 1 } _ { N } ^ { \top } w = 0
$$

is equivalent to the matrix inequality

$$
H _ { N } K H _ { N } \succeq 0 .
$$

Indeed, assume first that $H _ { N } K H _ { N } \succeq 0$ and let $w \in \mathbb { R } ^ { N }$ satisfy $\mathbf { 1 } _ { N } ^ { \top } w = 0$ . Since $H _ { N } w = w$ , we have

$$
w ^ { \top } K w = ( H _ { N } w ) ^ { \top } K ( H _ { N } w ) = w ^ { \top } ( H _ { N } K H _ { N } ) w \geq 0 .
$$

Conversely, assume that $w ^ { \top } K w \ge 0$ for every $w \in \mathbb { R } ^ { N }$ with $\mathbf { 1 } _ { N } ^ { \top } w = 0$ . Fix $\boldsymbol { v } \in \mathbb { R } ^ { N }$ and set $w : = H _ { N } v$ , so that $\mathbf { 1 } _ { N } ^ { \top } w = 0$ . Then

$$
\begin{array} { r } { v ^ { \top } ( H _ { N } K H _ { N } ) v = ( H _ { N } v ) ^ { \top } K ( H _ { N } v ) = w ^ { \top } K w \ge 0 . } \end{array}
$$

Since this holds for all $v \in \mathbb { R } ^ { N }$ , it follows that $H _ { N } K H _ { N } \succeq 0 .$

## B Basic identities for MMD<sup>2</sup>

Lemma B.1 (Equivalence of nonnegativity of $\mathbf { M M D ^ { 2 } }$ on a set and CPD on that set). Let k : $X \times$ $X \to \mathbb { R }$ be a symmetric kernel on a measurable space $X$ , and let $E \subseteq X$ be measurable. Then

$$
\begin{array} { r l } { \mathbf { M M D } _ { k } ^ { 2 } ( P , Q ) \geq 0 } & { \forall P , Q \in \mathcal { P } ( X ) \ \mathrm { w i t h } \ \mathrm { s u p p } \ P , \mathrm { s u p p } Q \subseteq E } \\ & { \quad \ \mathrm { i f } \ \mathrm { a n d } \ \mathrm { o n l y } \ \mathrm { i f } } \\ & { \ k \ \mathrm { i s } \mathbf { C P D } \ \mathrm { o n } E } \end{array}
$$

Here, $\mathrm { \Omega ^ { 6 6 } C P D }$ on $E ^ { \ast }$ means that

$$
\int \int _ { E \times E } k ( x , y ) d \mu ( x ) d \mu ( y ) \geq 0
$$

for every finite signed measure $\mu$ supported on $E$ with $\mu ( E ) = 0$ for which this integral is well defined.

Proof.

CPD on E implies nonnegativity. Take any two probability measures $P , Q$ supported on $E ,$ and set

$$
\sigma : = P - Q .
$$

Then σ is supported on $E$ and satisfies $\sigma ( E ) = 1 - 1 = 0$ . Hence, by the assumption that k is CPD on $E _ { \mathrm { { i } } }$

$$
\mathbf { M M D } _ { k } ^ { 2 } ( P , Q ) = \int \int k ( x , y ) d \sigma ( x ) d \sigma ( y ) \geq 0 .
$$

Nonnegativity implies CPD on E. Assume that $\mathbf { M M D } _ { k } ^ { 2 } ( P , Q ) \geq 0$ for all probability measures $P , Q$ supported on E for which $\mathbf { M M D } _ { k } ^ { 2 } ( P , Q )$ is well defined. Let $\mu$ be any finite signed measure supported on $E$ with $\mu ( E ) = 0$ for which $\int \int k d \mu d \mu$ is well defined. By the Hahn–Jordan decomposition,

$$
\mu = \mu ^ { + } - \mu ^ { - } , \qquad \mu ^ { + } ( E ) = \mu ^ { - } ( E ) = : A ,
$$

where the equality follows from $\mu ( E ) = 0 . { \mathrm { ~ I f ~ } } A = 0 .$ , then $\mu ^ { + } ( E ) = \mu ^ { - } ( E ) = 0$ , hence $\mu ^ { + } =$ $\mu ^ { - } = 0$ , so that $\mu = 0$ . In that case,

$$
\iint k ( x , y ) d \mu ( x ) d \mu ( y ) = 0 ,
$$

and the conclusion is immediate. Assume therefore that $A > 0$ . Define

$$
P = \frac { 1 } { A } \mu ^ { + } , \qquad Q = \frac { 1 } { A } \mu ^ { - } .
$$

These are probability measures supported on $E ,$ and they satisfy

$$
P - Q = { \frac { 1 } { A } } \mu .
$$

By bilinearity of the quadratic form and the identity $\mu = A ( P - Q )$

$$
\begin{array} { r l } & { \displaystyle \iint k ( x , y ) d \mu ( x ) d \mu ( y ) = A ^ { 2 } \iint k ( x , y ) d ( P - Q ) ( x ) d ( P - Q ) ( y ) } \\ & { \quad \quad \quad = A ^ { 2 } \mathbf { M } \mathbf { M } \mathbf { D } _ { k } ^ { 2 } ( P , Q ) } \\ & { \quad \quad \quad \geq 0 . } \end{array}
$$

Hence k is CPD on $E .$

Lemma B.2 $\mathbf { ( M M D ^ { 2 } }$ of pushforward measures). Let $f \colon X \to Y$ be a measurable map, let $\nu , \mu$ be probability measures on $X$ , let k : $Y \times Y  \mathbb { R }$ be any kernel, and define $k _ { f } ( x , x ^ { \prime } ) : = k \big ( \bar { f } ( x ) , f ( x ^ { \prime } ) \big )$ Then

$$
\mathbf { M M D } _ { k } ^ { 2 } \big ( f \# \nu , ~ f \# \mu \big ) = \mathbf { M M D } _ { k _ { f } } ^ { 2 } \big ( \nu , ~ \mu \big ) . ~ 
$$

Proof. Define the pushforward measures

$$
( f \# \nu ) ( B ) \ : = \ : \nu \big ( f ^ { - 1 } ( B ) \big ) , \quad ( f \# \mu ) ( B ) \ : = \ : \mu \big ( f ^ { - 1 } ( B ) \big ) .
$$

By the definition of $\mathbf { M M D ^ { 2 } }$

$$
\begin{array} { r l } { \mathbf { M } \mathbf { M } \mathbf { M } _ { k } ^ { 2 } \big ( f \# \nu , f \# \mu \big ) = \underset { u , u ^ { \prime } \times f \# \nu } { \mathbb { E } } \big [ k ( u , u ^ { \prime } ) \big ] + \underset { v , v ^ { \prime } \times f \# \mu } { \mathbb { E } } \big [ k ( v , v ^ { \prime } ) \big ] } & { } \\ & { \quad - \underset { u \sim f \# \nu , v \sim f \# \mu } { \mathbb { E } } \big [ k ( u , v ) \big ] } \\ & { = \underset { x , x ^ { \prime \prime } \times v } { \mathbb { E } } \big [ k \big ( f ( x ) , f ( x ^ { \prime } ) \big ) \big ] + \underset { y , y ^ { \prime \prime } \sim \mu } { \mathbb { E } } \big [ k \big ( f ( y ) , f ( y ^ { \prime } ) \big ) \big ] } \\ & { \quad - 2 \underset { x \sim y , v \sim \mu } { \mathbb { E } } \big [ k \big ( f ( x ) , f ( y ) \big ) \big ] } \\ & { = \underset { x , x ^ { \prime } \sim \nu } { \mathbb { E } } \big [ k _ { f } ( x , x ^ { \prime } ) \big ] + \underset { y , y ^ { \prime \prime } \sim \mu } { \mathbb { E } } \big [ k _ { f } ( y , y ^ { \prime } ) \big ] - 2 \underset { x \sim y , y \sim \mu } { \mathbb { E } } \big [ k _ { f } ( x , y ) \big ] } \\ & { = \mathbf { M } \mathbf { M } \mathbf { M } \mathbf { \ } _ { k \mu } ^ { 2 } \big ( \nu , \mu \big ) . } \end{array}
$$

Lemma B.3 $( { \bf M M D } ^ { 2 }$ is linear in the kernel). Let $\nu , \mu$ be probability measures on a measurable space X, let $k _ { 1 } , k _ { 2 } : X \times X \to$ R be kernels, and let $\alpha _ { 1 } , \alpha _ { 2 } \in \mathbb { R }$ . Then

$$
\mathbf { M M D } _ { \alpha _ { 1 } k _ { 1 } + \alpha _ { 2 } k _ { 2 } } ^ { 2 } \left( \nu , \mu \right) = \alpha _ { 1 } \mathbf { M M D } _ { k _ { 1 } } ^ { 2 } \left( \nu , \mu \right) + \alpha _ { 2 } \mathbf { M M D } _ { k _ { 2 } } ^ { 2 } \left( \nu , \mu \right) . \mathbf { \Lambda } \Bigg \vert
$$

In particular, setting $k _ { 2 } \equiv 0$ gives

$$
\mathbf { M M D } _ { \alpha _ { 1 } } ^ { 2 } \left( \nu , \mu \right) = \alpha _ { 1 } \mathbf { M M D } _ { k _ { 1 } } ^ { 2 } \left( \nu , \mu \right) .
$$

Proof. This follows from

$$
\mathbf { M M D } _ { k } ^ { 2 } ( \nu , \mu ) = \iint _ { X \times X } k ( x , y ) \left( \nu - \mu \right) ( d x ) \left( \nu - \mu \right) ( d y )
$$

by linearity of the integral.

## C Bellman contraction in Maximum Mean Discrepancy

Preamble. This section provides the contraction theory underlying all subsequent arguments. By Theorem C.1, for a fixed map f, an MMD inequality of the form

$$
\mathbf { M M D } _ { k } ^ { 2 } \big ( f \# \nu , f \# \mu \big ) \leq \tau \mathbf { M M D } _ { k } ^ { 2 } ( \nu , \mu )
$$

holds for all $\nu , \mu$ if and only if the difference kernel $\tau k - k _ { f }$ is CPD, where $k _ { f } ( x , y ) = k ( f ( x ) , f ( y ) )$ This criterion is then extended to the full distributional Bellman operator by representing $\tau ^ { \pi }$ as a mixture of affine pushforwards and exploiting convexity of the CPD quadratic form under averaging, yielding a condition that is both sufficient and necessary for τ-contraction in $\overline { { \mathbf { M M D } } } _ { k } ^ { 2 }$ across all MDPs with discount $\gamma$ (Theorem C.2). Section C.2 then extends the analysis to return laws supported on a bounded set, where the CPD condition is required only on that set. It is therefore weaker, and holds for some kernels that fail it on all of $\mathbb { R } ^ { \dot { d } }$ . For radial kernels on $\mathbb { R } ^ { d }$ with d fixed, as in Proposition E.2, the same criterion can be equivalently expressed as a conditionally negative definite (CND) property for an associated radial kernel. This connects Bellman contraction in MMD to CND kernels and to the characterizations of Schoenberg and Bernstein used in the next section.

## C.1 Contraction via difference kernel

Theorem C.1 (Contraction of MMD on a set under CPD kernels). Let $f \colon X \to X$ be measurable, let $k \colon X \times X $ R be a symmetric, conditionally positive definite (CPD) kernel, fix a measurable set $E \subseteq X$ , and define

$$
k _ { f } ( x , y ) = k { \big ( } f ( x ) , f ( y ) { \big ) } , \qquad x , y \in X .
$$

Fix $\tau \in \mathbb { R }$ . Then

$$
\begin{array} { r l } { \boxed { \mathbf { M M D } _ { k } ^ { 2 } \big ( f \# \nu , f \# \mu \big ) \ \leq \ \tau \ \mathbf { M M D } _ { k } ^ { 2 } ( \nu , \mu ) \qquad \forall \mu , \nu \in \mathcal { P } ( X ) \ \mathrm { w i t h } \ \mathrm { s u p p } \ \mu , \mathrm { s u p p } \nu \subseteq E } } \\ & { \qquad \quad \mathrm { i f ~ a n d ~ o n l y ~ i f } } \\ & { \qquad \quad \tau k \ - \ k _ { f } \ \mathrm { i s } \mathrm { C P D ~ o n } \ E } \end{array}
$$

Here, “CPD on $E ^ { \ast }$ means that

$$
\iint _ { E \times E } \bigl ( \tau k - k _ { f } \bigr ) ( x , y ) d \sigma ( x ) d \sigma ( y ) \geq 0
$$

for every finite signed measure σ supported on $E$ with $\sigma ( E ) = 0$ for which this integral is well defined.

Accordingly, for laws supported on $E ,$ the smallest such τ is

$$
\tau _ { E } ^ { \star } : = \operatorname* { i n f } \{ \tau > 0 : \tau k - k _ { f } \mathrm { ~ i s ~ C P D ~ o n ~ } E \} .
$$

Contraction on laws supported on E holds exactly when $\tau _ { E } ^ { \star } < 1$ . In that case, $\tau _ { E } ^ { \star }$ is the smallest contraction factor on E in $\mathbf { M M D ^ { 2 } }$ , and the corresponding smallest contraction factor in MMD is $\sqrt { \tau _ { E } ^ { \star } }$ . The choice $E = X$ yields contraction on all probability laws on $X$ , while proper subsets E restrict the comparison to laws supported on a prescribed region, such as the bounded sets used later.

Proof. Fix $\mu , \nu \in { \mathcal { P } } ( X )$ with

$$
\operatorname { s u p p } \mu , \operatorname { s u p p } \nu \subseteq E .
$$

By Lemma B.2, we have

$$
\mathbf { M M D } _ { k } ^ { 2 } \big ( f \# \nu , f \# \mu \big ) = \mathbf { M M D } _ { k _ { f } } ^ { 2 } ( \nu , \mu ) .
$$

By Lemma B.3,

$$
\tau \mathbf { M } \mathbf { M } \mathbf { D } _ { k } ^ { 2 } ( \nu , \mu ) = \mathbf { M } \mathbf { M } \mathbf { D } _ { \tau k } ^ { 2 } ( \nu , \mu ) .
$$

Hence

$$
\tau \mathbf { M M D } _ { k } ^ { 2 } ( \nu , \mu ) - \mathbf { M M D } _ { k } ^ { 2 } \big ( f \# \nu , f \# \mu \big ) = \mathbf { M M D } _ { \tau k } ^ { 2 } ( \nu , \mu ) - \mathbf { M M D } _ { k _ { f } } ^ { 2 } ( \nu , \mu ) .
$$

By Lemma B.3, this difference is

$$
\mathbf { M M D } _ { \tau k - k _ { f } } ^ { 2 } ( \nu , \mu ) .
$$

Therefore,

$$
\mathbf { M M D } _ { k } ^ { 2 } \big ( f \# \nu , f \# \mu \big ) \leq \tau \mathbf { M M D } _ { k } ^ { 2 } ( \nu , \mu ) \quad \Longleftrightarrow \quad \mathbf { M M D } _ { \tau k - k _ { f } } ^ { 2 } ( \nu , \mu ) \geq 0 .
$$

Thus

$$
\mathbf { M M D } _ { k } ^ { 2 } \big ( f \# \nu , f \# \mu \big ) \leq \tau \mathbf { M M D } _ { k } ^ { 2 } ( \nu , \mu ) \qquad \forall \mu , \nu \in \mathcal { P } ( X ) \mathrm { ~ w i t h ~ s u p p ~ } \mu , \operatorname { s u p p } \nu \subseteq E
$$

if and only if

$$
\mathbf { M M D } _ { \tau k - k _ { f } } ^ { 2 } ( \nu , \mu ) \geq 0 \qquad \forall \mu , \nu \in \mathcal { P } ( X ) \ \mathrm { w i t h \ s u p p } \mu , \mathrm { s u p p } \nu \subseteq E .
$$

Finally, by Lemma $\mathrm { { B } } . 1$ , this is equivalent to $\tau k - k _ { f }$ being CPD on $E .$

Lemma C.1 (The Bellman operator averages pushforwards under affine maps). Fix $( s , a ) \in S \times A ,$ and let

$$
C : = \left( R ( s , a ) , S ^ { \prime } , A ^ { \prime } \right)
$$

with law $\rho _ { s , a }$ . For each realization $C = ( q , s ^ { \prime } , a ^ { \prime } )$ , define

$$
f _ { q } : \mathbb { R } ^ { d }  \mathbb { R } ^ { d } , \qquad f _ { q } ( x ) : = q + \gamma x .
$$

Let $\eta : \mathcal { S } \times \mathcal { A }  \mathcal { P } ( \mathbb { R } ^ { d } )$ , and let $X ^ { \prime } \mid C \sim \eta ( S ^ { \prime } , A ^ { \prime } )$ . Given $C$ (the reward, next state and next action), the Bellman update maps the next return law to its pushforward under the affine map $f _ { q }$

$$
\operatorname { L a w } ( q + \gamma X ^ { \prime } \mid C ) = f _ { q } \# \eta ( s ^ { \prime } , a ^ { \prime } ) .
$$

Consequently,

$$
\boxed { ( T ^ { \pi } \eta ) ( s , a ) = \int f _ { q } \# \eta ( s ^ { \prime } , a ^ { \prime } ) \rho _ { s , a } ( d C ) . }
$$

Proof. Fix a realization $C = ( q , s ^ { \prime } , a ^ { \prime } )$ . Conditionally on $C ,$ the random variable $X ^ { \prime }$ has law $\eta ( s ^ { \prime } , a ^ { \prime } )$ Hence, for every Borel set $B \subseteq \mathbb { R } ^ { d }$

$$
\operatorname* { P r } ( q + \gamma X ^ { \prime } \in B \mid C ) = \operatorname* { P r } ( X ^ { \prime } \in f _ { q } ^ { - 1 } ( B ) \mid C ) = \eta ( s ^ { \prime } , a ^ { \prime } ) { \big ( } f _ { q } ^ { - 1 } ( B ) { \big ) } = ( f _ { q } \# \eta ( s ^ { \prime } , a ^ { \prime } ) ) ( B ) .
$$

Therefore,

$$
\operatorname { L a w } ( q + \gamma X ^ { \prime } \mid C ) = f _ { q } \# \eta ( s ^ { \prime } , a ^ { \prime } ) .
$$

Integrating this conditional law with respect to $\rho _ { s , a }$ yields

$$
( T ^ { \pi } \eta ) ( s , a ) = \int f _ { q } \# \eta ( s ^ { \prime } , a ^ { \prime } ) \rho _ { s , a } ( d C ) ,
$$

which is the claimed representation.

Lemma C.2 (A Bellman operator can reduce to an affine map). Fix $q \in \mathbb { R } ^ { d }$ and $\gamma \in ( 0 , 1 )$ . Then there exists a discounted MDP with a single state s, a single action a whose transition always returns to $s ,$ and deterministic reward $q ,$ together with its unique fixed policy $\pi ,$ such that

$$
\boxed { ( \mathcal T ^ { \pi } \eta ) ( s , a ) = f _ { q } \# \eta ( s , a ) , \qquad f _ { q } ( x ) = q + \gamma x , }
$$

for every law $\eta ( s , a ) \in \mathcal { P } ( \mathbb { R } ^ { d } )$ . Moreover, since $\mathcal { S } \times \mathcal { A } = \{ ( s , a ) \}$

$$
\overline { { \mathbf { M M D } } } _ { k } ^ { 2 } ( \eta _ { 1 } , \eta _ { 2 } ) = \mathbf { M M D } _ { k } ^ { 2 } \big ( \eta _ { 1 } ( s , a ) , \eta _ { 2 } ( s , a ) \big )
$$

for all laws $\eta _ { 1 } ( s , a ) , \eta _ { 2 } ( s , a ) \in \mathcal { P } ( \mathbb { R } ^ { d } )$

Proof. Consider the MDP with one state $s ,$ one action $^ { a , }$ transition kernel

$$
P ( \{ s \} \mid s , a ) = 1 ,
$$

and deterministic reward equal to $q .$ Let $\pi$ be the unique policy on this MDP. For any law $\eta ( s , a ) \in$ ${ \mathcal { P } } ( \mathbb { R } ^ { d } )$ , let $X ^ { \prime } \sim \eta ( s , a )$ . By definition of the distributional Bellman operator,

$$
( T ^ { \pi } \eta ) ( s , a ) = \operatorname { L a w } ( q + \gamma X ^ { \prime } ) = f _ { q } \# \eta ( s , a ) .
$$

The identity for $\overline { { \mathbf { M M D } } } _ { k } ^ { 2 }$ is immediate because there is only one state-action pair.

Lemma C.3 (Averaging does not increase the CPD quadratic form). Let $k \colon  { \mathbb { R } } ^ { d } \times  { \mathbb { R } } ^ { d } \to  { \mathbb { R } }$ be a symmetric CPD kernel, let $( \Omega , { \mathcal { F } } , \rho )$ be a probability space, and let $( \sigma _ { c } ) _ { c \in \Omega }$ be a measurable family of finite signed measures on $\mathbb { R } ^ { d }$ , each with total mass 0. Define

$$
\bar { \sigma } : = \int _ { \Omega } \sigma _ { c } \rho ( d c ) .
$$

Then

$$
\boxed { \iint k ( x , y ) d \bar { \sigma } ( x ) d \bar { \sigma } ( y ) \leq \int _ { \Omega } \left( \iiint k ( x , y ) d \sigma _ { c } ( x ) d \sigma _ { c } ( y ) \right) \rho ( d c ) }
$$

whenever the quadratic forms are well defined.

Proof. Define

$$
Q ( \sigma ) : = \int \int k ( x , y ) d \sigma ( x ) d \sigma ( y ) .
$$

Since k is CPD and each $\sigma _ { c } - \bar { \sigma }$ has total mass $0 ,$ we have

$$
Q ( \sigma _ { c } - \bar { \sigma } ) \geq 0 \qquad \mathrm { f o r ~ e v e r y ~ } c \in \Omega .
$$

Integrating with respect to $\rho$ yields

$$
0 \leq \int _ { \Omega } Q ( \sigma _ { c } - \bar { \sigma } ) \rho ( d c ) .
$$

Expanding the quadratic form and using bilinearity,

$$
\begin{array} { l l l } { \displaystyle \int _ { \Omega } Q ( \sigma _ { c } - \bar { \sigma } ) \rho ( d c ) = \int _ { \Omega } Q ( \sigma _ { c } ) \rho ( d c ) - 2 \int _ { \Omega } \iint k ( x , y ) d \sigma _ { c } ( x ) d \bar { \sigma } ( y ) \rho ( d c ) } \\ { \displaystyle + Q ( \bar { \sigma } ) . } \end{array}
$$

To simplify the mixed term, write

$$
\bar { \sigma } = \int _ { \Omega } \sigma _ { c ^ { \prime } } \rho ( d c ^ { \prime } ) ,
$$

where $c ^ { \prime }$ is a dummy variable. Then, using bilinearity to expand σ¯ and Fubini to interchange the c and $c ^ { \prime }$ integrals, we obtain

$$
\begin{array} { l l l } { \displaystyle \int _ { \Omega } \iint k ( x , y ) d \sigma _ { c } ( x ) d \bar { \sigma } ( y ) \rho ( d c ) = \displaystyle \int _ { \Omega } \int _ { \Omega } \left( \iint k ( x , y ) d \sigma _ { c } ( x ) d \sigma _ { c ^ { \prime } } ( y ) \right) \rho ( d c ^ { \prime } ) \rho ( d c ) } \\ { \displaystyle \qquad = \displaystyle \iint k ( x , y ) d \left( \int _ { \Omega } \sigma _ { c } \rho ( d c ) \right) ( x ) d \left( \int _ { \Omega } \sigma _ { c ^ { \prime } } \rho ( d c ^ { \prime } ) \right) ( y ) } \\ { \displaystyle \qquad = \displaystyle \iint k ( x , y ) d \bar { \sigma } ( x ) d \bar { \sigma } ( y ) } \\ { \displaystyle \qquad = Q ( \bar { \sigma } ) . } \end{array}
$$

Therefore

$$
0 \leq \int _ { \Omega } Q ( \sigma _ { c } ) \rho ( d c ) - Q ( \bar { \sigma } ) ,
$$

which is exactly the claimed inequality.

Lemma ${ \bf C . 4 ( M M D ^ { 2 } }$ contraction from affine maps to the Bellman operator). Let $k \colon  { \mathbb { R } } ^ { d } \times  { \mathbb { R } } ^ { d } \to$ R be a symmetric CPD kernel as in Definition A.5, let $E \subseteq \mathbb { R } ^ { d }$ , and let $\eta _ { 1 } , \eta _ { 2 } : S \times \mathcal { A } \to \mathcal { P } ( \mathbb { R } ^ { d } )$ satisfy

$$
\operatorname { s u p p } \eta _ { i } ( s , a ) \subseteq E \qquad \forall ( s , a ) \in S \times { \mathcal { A } } , \quad i = 1 , 2 .
$$

Assume that there exists $\tau \geq 0$ such that, for every $q \in \mathbb { R } ^ { d }$ , the affine map

$$
f _ { q } : \mathbb { R } ^ { d }  \mathbb { R } ^ { d } , \qquad f _ { q } ( x ) = q + \gamma x ,
$$

satisfies

$$
\mathbf { M M D } _ { k } ^ { 2 } \big ( f _ { q } \# \nu , f _ { q } \# \mu \big ) \leq \tau \mathbf { M M D } _ { k } ^ { 2 } ( \nu , \mu ) \qquad \forall \nu , \mu \in \mathcal { P } ( \mathbb { R } ^ { d } ) \mathrm { ~ w i t h ~ s u p p } \nu , \mathrm { s u p p } \mu \subseteq E ,
$$

for which the quantities are well defined. Then, for every $( s , a ) \in S \times A .$

$$
\boxed { \mathbf { M M D } _ { k } ^ { 2 } \bigl ( ( \mathscr { T } ^ { \pi } \eta _ { 1 } ) ( s , a ) , ( \mathscr { T } ^ { \pi } \eta _ { 2 } ) ( s , a ) \bigr ) \le \tau \overline { { \mathbf { M M D } } } _ { k } ^ { 2 } ( \eta _ { 1 } , \eta _ { 2 } ) . }
$$

Proof. Fix $( s , a ) \in S \times A .$ and let

$$
C : = \left( R ( s , a ) , S ^ { \prime } , A ^ { \prime } \right)
$$

denote the reward, next state and next action drawn from $( s , a )$ , with law

$$
\rho _ { s , a } : = \operatorname { L a w } ( C ) .
$$

By Lemma C.1, for each realization $C = ( q , s ^ { \prime } , a ^ { \prime } )$ , define

$$
\mu _ { C } : = f _ { q } \# \eta _ { 1 } ( s ^ { \prime } , a ^ { \prime } ) , \qquad \nu _ { C } : = f _ { q } \# \eta _ { 2 } ( s ^ { \prime } , a ^ { \prime } ) .
$$

Then

$$
( { \cal T } ^ { \pi } \eta _ { 1 } ) ( s , a ) = \int \mu _ { C } \rho _ { s , a } ( d C ) , \qquad ( { \cal T } ^ { \pi } \eta _ { 2 } ) ( s , a ) = \int \nu _ { C } \rho _ { s , a } ( d C ) .
$$

Comparing quadratic forms for mixtures.

Write

$$
\sigma _ { C } : = \mu _ { C } - \nu _ { C } , \quad \quad \bar { \sigma } : = \int \sigma _ { C } \rho _ { s , a } ( d C ) .
$$

Since

$$
\bar { \sigma } = ( T ^ { \pi } \eta _ { 1 } ) ( s , a ) - ( T ^ { \pi } \eta _ { 2 } ) ( s , a ) ,
$$

Lemma C.3 gives

$$
\iint k ( x , y ) d \bar { \sigma } ( x ) d \bar { \sigma } ( y ) \leq \int \left( \iint k ( x , y ) d \sigma _ { C } ( x ) d \sigma _ { C } ( y ) \right) \rho _ { s , a } ( d C ) .
$$

By the definition of $ { \mathbf { M M D } } _ { k } ^ { 2 }$ as a quadratic form for CPD kernels in Definition A.6, this yields

$$
\mathbf { M M D } _ { k } ^ { 2 } \big ( ( \mathcal T ^ { \pi } \eta _ { 1 } ) ( s , a ) , ( \mathcal T ^ { \pi } \eta _ { 2 } ) ( s , a ) \big ) \leq \int \mathbf { M M D } _ { k } ^ { 2 } ( \mu _ { C } , \nu _ { C } ) \rho _ { s , a } ( d C ) .
$$

Applying the assumed contraction on $E .$

Since $\eta _ { 1 } ( s ^ { \prime } , a ^ { \prime } )$ and $\eta _ { 2 } ( s ^ { \prime } , a ^ { \prime } )$ are supported on $E ,$ the inequality assumed for $f _ { q }$ applies and gives

$$
\mathbf { M M D } _ { k } ^ { 2 } ( \mu _ { C } , \nu _ { C } ) \leq \tau \mathbf { M M D } _ { k } ^ { 2 } \big ( \eta _ { 1 } ( s ^ { \prime } , a ^ { \prime } ) , \eta _ { 2 } ( s ^ { \prime } , a ^ { \prime } ) \big ) .
$$

Therefore,

$$
\begin{array} { r l } & { \mathbf { M M D } _ { k } ^ { 2 } \big ( ( \mathcal T ^ { \pi } \eta _ { 1 } ) ( s , a ) , ( \mathcal T ^ { \pi } \eta _ { 2 } ) ( s , a ) \big ) \leq \int \mathbf { M M D } _ { k } ^ { 2 } ( \mu _ { C } , \nu _ { C } ) \rho _ { s , a } ( d C ) } \\ & { \qquad \leq \tau \int \mathbf { M M D } _ { k } ^ { 2 } \big ( \eta _ { 1 } ( s ^ { \prime } , a ^ { \prime } ) , \eta _ { 2 } ( s ^ { \prime } , a ^ { \prime } ) \big ) \rho _ { s , a } ( d C ) } \\ & { \qquad \leq \tau \overline { { \mathbf { M M D } } } _ { k } ^ { 2 } ( \eta _ { 1 } , \eta _ { 2 } ) , } \end{array}
$$

which proves the claim.

Theorem C.2 (CPD criterion for Bellman contraction in $\overline { { \mathbf { M M D } } } _ { k } ^ { 2 } )$ . Let $k \colon  { \mathbb { R } } ^ { d } \times  { \mathbb { R } } ^ { d } \to  { \mathbb { R } }$ be a symmetric CPD kernel as in Definition $\mathrm { A . } 5 .$ , and fix $\tau \in [ 0 , 1 )$ . For each $q \in \mathbb { R } ^ { d }$ , define the Bellman difference kerne

$$
k _ { \mathrm { d i f f } } ^ { q , \gamma } ( x , y ) : = \tau k ( x , y ) - k ( q + \gamma x , q + \gamma y ) .
$$

If $k _ { \mathrm { d i f f } } ^ { q , \gamma }$ is CPD for every $q \in \mathbb { R } ^ { d }$ , then $\tau ^ { \pi }$ is a τ-contraction in $\overline { { \mathbf { M M D } } } _ { k } ^ { 2 }$ , that ${ \mathrm { i s } } ,$

$$
\begin{array} { r l r } { \Big \vert \overline { { \mathbf { M M D } } } _ { k } ^ { 2 } ( T ^ { \pi } \eta _ { 1 } , \mathcal { T } ^ { \pi } \eta _ { 2 } ) \leq \tau \overline { { \mathbf { M M D } } } _ { k } ^ { 2 } ( \eta _ { 1 } , \eta _ { 2 } ) } & { } & { \forall \eta _ { 1 } , \eta _ { 2 } : \mathcal { S } \times \mathcal { A } \to \mathcal { P } ( \mathbb { R } ^ { d } ) . } \end{array}
$$

Furthermore, this condition is necessary for τ -contraction to hold for the Bellman operator of every MDP with discount γ under every fixed policy.

Proof.

Sufficiency. Let $\eta _ { 1 } , \eta _ { 2 } : S \times \mathcal { A } \to \mathcal { P } ( \mathbb { R } ^ { d } )$ . By Theorem C.1, applied with $X = \mathbb { R } ^ { d } , E = \mathbb { R } ^ { d }$ , and

$$
f _ { q } ( x ) = q + \gamma x ,
$$

the assumption that $k _ { \mathrm { d i f f } } ^ { q , \gamma }$ is CPD for every $q \in \mathbb { R } ^ { d }$ implies that

$$
\mathbf { M M D } _ { k } ^ { 2 } \big ( f _ { q } \# \nu , f _ { q } \# \mu \big ) \leq \tau \mathbf { M M D } _ { k } ^ { 2 } ( \nu , \mu ) \qquad \forall \nu , \mu \in \mathcal { P } ( \mathbb { R } ^ { d } ) ,
$$

for every $q \in \mathbb { R } ^ { d }$ and for all quantities that are well defined. Applying Lemma C.4 with $E = \mathbb { R } ^ { d }$ , we obtain, for every $( s , a ) \in S \times A$

$$
\mathbf { M M D } _ { k } ^ { 2 } \bigl ( ( \mathscr { T } ^ { \pi } \eta _ { 1 } ) ( s , a ) , ( \mathscr { T } ^ { \pi } \eta _ { 2 } ) ( s , a ) \bigr ) \leq \tau \overline { { \mathbf { M M D } } } _ { k } ^ { 2 } ( \eta _ { 1 } , \eta _ { 2 } ) .
$$

Taking the supremum over $( s , a )$ yields

$$
\overline { { \mathbf { M M D } } } _ { k } ^ { 2 } ( T ^ { \pi } \eta _ { 1 } , T ^ { \pi } \eta _ { 2 } ) \le \tau \overline { { \mathbf { M M D } } } _ { k } ^ { 2 } ( \eta _ { 1 } , \eta _ { 2 } ) ,
$$

so $\mathcal { T } ^ { \pi }$ is a τ-contraction in $\overline { { \mathbf { M M D } } } _ { k } ^ { 2 }$

Necessity. Assume now that τ-contraction holds for the Bellman operator of every MDP with discount γ under every fixed policy. Fix $q \in \mathbb { R } ^ { d }$ . By Lemma C.2, there exists a discounted MDP and a fixed policy for which

$$
( T ^ { \pi } \eta ) ( s , a ) = f _ { q } \# \eta ( s , a )
$$

at the unique state-action pair $( s , a )$ . Since in this problem

$$
\overline { { \mathbf { M M D } } } _ { k } ^ { 2 } ( \eta _ { 1 } , \eta _ { 2 } ) = \mathbf { M M D } _ { k } ^ { 2 } \big ( \eta _ { 1 } ( s , a ) , \eta _ { 2 } ( s , a ) \big ) ,
$$

the assumed Bellman τ -contraction implies

$$
\mathbf { M M D } _ { k } ^ { 2 } \big ( f _ { q } \# \nu , f _ { q } \# \mu \big ) \leq \tau \mathbf { M M D } _ { k } ^ { 2 } ( \nu , \mu ) \qquad \forall \nu , \mu \in \mathcal { P } ( \mathbb { R } ^ { d } ) .
$$

Applying Theorem C.1 with $X = \mathbb { R } ^ { d }$ and $E = \mathbb { R } ^ { d }$ , we conclude that $k _ { \mathrm { d i f f } } ^ { q , \gamma }$ is CPD. Since q was arbitrary, this holds for every $q \in \mathbb { R } ^ { d }$ □

## C.2 Contraction on a bounded support

Proposition C.1 (The ball as a natural set under bounded rewards). Assume that there exists $r _ { \operatorname* { m a x } } \ge 0$ such that

$$
\| R ( s , a ) \| \leq r _ { \operatorname* { m a x } } \qquad \mathrm { f o r } \mathrm { e v e r y } ( s , a ) \in \mathcal { S } \times \mathcal { A } .
$$

Let

$$
E : = B _ { M } = \{ x \in \mathbb { R } ^ { d } : \| x \| \leq M \} , \ M \geq \frac { r _ { \operatorname* { m a x } } } { 1 - \gamma } .
$$

Then the following hold.

1. For every $( s , a ) \in \mathcal S \times \mathcal A$ , the return law $\eta ^ { \pi } ( s , a )$ is supported on $E .$

2. If $\eta : \mathcal { S } \times \mathcal { A }  \mathcal { P } ( \mathbb { R } ^ { d } )$ satisfies

$$
\operatorname { s u p p } \eta ( s , a ) \subseteq E \qquad \forall ( s , a ) \in S \times \mathcal { A } ,
$$

then also

$$
\operatorname { s u p p } ( T ^ { \pi } \eta ) ( s , a ) \subseteq E \qquad \forall ( s , a ) \in \mathcal { S } \times \mathcal { A } .
$$

In other words, the class of laws supported on E is invariant under $\mathcal { T } ^ { \pi }$

## Proof.

## 1. Bounded rewards imply bounded returns.

Fix $( s , a ) \in \mathcal S \times \mathcal A$ , and consider the random return

$$
Z ^ { \pi } ( s , a ) = \sum _ { t \geq 0 } \gamma ^ { t } R _ { t } .
$$

For almost every realization of this random variable, the triangle inequality and the reward bound give

$$
\Vert Z ^ { \pi } ( s , a ) \Vert \leq \sum _ { t \geq 0 } \gamma ^ { t } \Vert R _ { t } \Vert \leq \sum _ { t \geq 0 } \gamma ^ { t } r _ { \operatorname* { m a x } } = \frac { r _ { \operatorname* { m a x } } } { 1 - \gamma } \leq M .
$$

Thus almost every realization of $Z ^ { \pi } ( s , a )$ belongs to $E = B _ { M }$ . Therefore,

$$
\operatorname { s u p p } \eta ^ { \pi } ( s , a ) \subseteq E .
$$

## 2. Bellman invariance of the ball.

Assume now that

$$
\operatorname { s u p p } \eta ( s , a ) \subseteq E \qquad \forall ( s , a ) \in S \times \mathcal { A } .
$$

Fix $( s , a )$ , and let $X ^ { \prime } \sim \eta ( S ^ { \prime } , A ^ { \prime } )$ in the Bellman update. Since $\eta ( S ^ { \prime } , A ^ { \prime } )$ is supported on $E ,$ almost every realization of $X ^ { \prime }$ satisfies

$$
\| X ^ { \prime } \| \leq M .
$$

Hence

$$
\| R ( s , a ) + \gamma X ^ { \prime } \| \leq \| R ( s , a ) \| + \gamma \| X ^ { \prime } \| \leq r _ { \operatorname* { m a x } } + \gamma M \leq M ,
$$

where the last inequality follows from $M \ge r _ { \operatorname* { m a x } } / ( 1 - \gamma )$ . Thus almost every realization of $R ( s , a ) + \gamma X ^ { \prime }$ lies in $E ,$ so

$$
\operatorname { s u p p } ( { \mathcal { T } } ^ { \pi } \eta ) ( s , a ) \subseteq E .
$$

Since $( s , a )$ was arbitrary, the conclusion holds for all state-action pairs.

Theorem C.3 (CPD criterion for Bellman contraction on a set). Let $k \colon  { \mathbb { R } } ^ { d } \times  { \mathbb { R } } ^ { d } \to$ R be a symmetric CPD kernel as in Definition $\mathrm { A . 5 , }$ let $E \subseteq \mathbb { R } ^ { d }$ be measurable, and fix $\tau \in [ 0 , 1 )$ . For each $q \in \mathbb { R } ^ { d }$ define the Bellman difference kernel

$$
k _ { \mathrm { d i f f } } ^ { q , \gamma } ( x , y ) : = \tau k ( x , y ) - k ( q + \gamma x , q + \gamma y ) .
$$

Assume that $k _ { \mathrm { d i f f } } ^ { q , \gamma }$ is CPD on E for every $q \in \mathbb { R } ^ { d }$ . Let

$$
\mathcal { C } _ { E } : = \big \{ \eta : \mathcal { S } \times \mathcal { A } \to \mathcal { P } ( \mathbb { R } ^ { d } ) \ : \ \mathrm { s u p p } \eta ( s , a ) \subseteq E \ \forall ( s , a ) \in \mathcal { S } \times \mathcal { A } \big \} .
$$

Then, for all $\eta _ { 1 } , \eta _ { 2 } \in { \mathcal { C } } _ { E }$

$$
\begin{array} { r } { \boxed { \overline { { \mathbf { M M D } } } _ { k } ^ { 2 } ( T ^ { \pi } \eta _ { 1 } , T ^ { \pi } \eta _ { 2 } ) \le \tau \overline { { \mathbf { M M D } } } _ { k } ^ { 2 } ( \eta _ { 1 } , \eta _ { 2 } ) . } } \end{array}
$$

If, in addition, ${ \mathcal { T } } ^ { \pi } ( { \mathcal { C } } _ { E } ) \subseteq { \mathcal { C } } _ { E }$ , then $\mathcal { T } ^ { \pi }$ is a τ-contraction on $\mathcal { C } _ { E }$ . Conversely, assume that for every MDP with discount γ under every fixed policy, whenever ${ \mathcal { T } } ^ { \pi } ( { \mathcal { C } } _ { E } ) \subseteq { \mathcal { C } } _ { E }$ , the Bellman operator is a τ-contraction on $\mathcal { C } _ { E }$ . Then, for every $q \in \mathbb { R } ^ { d }$ such that $q + \gamma E \subseteq E$ , the kernel $k _ { \mathrm { d i f f } } ^ { q , \gamma }$ is CPD on $E .$ Proposition C.1 gives one natural situation in which such an invariant set E exists.

Proof.

Sufficiency. Fix $\eta _ { 1 } , \eta _ { 2 } \in { \mathcal { C } } _ { E }$ . By Theorem C.1, applied with $X = \mathbb { R } ^ { d } .$ , the set $E ,$ , and

$$
f _ { q } ( x ) = q + \gamma x ,
$$

the assumption that $k _ { \mathrm { d i f f } } ^ { q , \gamma }$ is CPD on E implies that, for every $q \in \mathbb { R } ^ { d }$

$$
\mathbf { M M D } _ { k } ^ { 2 } \big ( f _ { q } \# \nu , f _ { q } \# \mu \big ) \leq \tau \mathbf { M M D } _ { k } ^ { 2 } ( \nu , \mu ) \qquad \forall \mu , \nu \in \mathcal { P } ( \mathbb { R } ^ { d } ) \mathrm { ~ w i t h ~ s u p p } \mu , \operatorname { s u p p } \nu \subseteq E .
$$

Since $\eta _ { 1 } , \eta _ { 2 } \in \mathcal { C } _ { E } .$ , the assumptions of Lemma C.4 are satisfied with the same set E. Applying that lemma, we obtain, for every $( s , a ) \in S \times A$

$$
\mathbf { M M D } _ { k } ^ { 2 } \bigl ( ( \mathscr { T } ^ { \pi } \eta _ { 1 } ) ( s , a ) , ( \mathscr { T } ^ { \pi } \eta _ { 2 } ) ( s , a ) \bigr ) \leq \tau \overline { { \mathbf { M M D } } } _ { k } ^ { 2 } ( \eta _ { 1 } , \eta _ { 2 } ) .
$$

Taking the supremum over $( s , a )$ yields

$$
\overline { { \mathbf { M M D } } } _ { k } ^ { 2 } ( T ^ { \pi } \eta _ { 1 } , T ^ { \pi } \eta _ { 2 } ) \le \tau \overline { { \mathbf { M M D } } } _ { k } ^ { 2 } ( \eta _ { 1 } , \eta _ { 2 } ) ,
$$

which proves the boxed inequality.

$\operatorname { I f } ,$ in addition, ${ \mathcal { T } } ^ { \pi } ( { \mathcal { C } } _ { E } ) \subseteq { \mathcal { C } } _ { E }$ , then $\mathcal { T } ^ { \pi }$ maps $\mathcal { C } _ { E }$ into itself and satisfies the above inequality on $\mathcal { C } _ { E }$ Hence $\mathcal { T } ^ { \pi }$ is a τ-contraction on $\mathcal { C } _ { E }$

Necessity. Suppose that, for every MDP with discount $\gamma$ under every fixed policy, the Bellman operator is a τ-contraction on $\mathcal { C } _ { E }$ whenever it maps $\mathcal { C } _ { E }$ into itself. We show that this forces $k _ { \mathrm { d i f f } } ^ { q , \gamma }$ to be CPD on E for every $q \in \mathbb { R } ^ { d }$ satisfying

$$
q + \gamma E \subseteq E .
$$

Fix such a $q .$ By Lemma C.2, there exists a discounted MDP with a single state $s ,$ a single action a whose transition always returns to s, deterministic reward q, and its unique fixed policy π, such that

$$
( T ^ { \pi } \eta ) ( s , a ) = f _ { q } \# \eta ( s , a ) , \qquad f _ { q } ( x ) = q + \gamma x .
$$

Moreover, this Bellman operator preserves laws supported on $E .$ . Indeed, if $\eta \in { \mathcal { C } } _ { E }$ , then $\eta ( s , a )$ is supported on $E .$ , and since $q + \gamma E \subseteq E$ , the pushforward $f _ { q } \# \eta ( s , a )$ is also supported on $E .$ . Hence

$$
{ \mathcal { T } } ^ { \pi } ( { \mathcal { C } } _ { E } ) \subseteq { \mathcal { C } } _ { E } .
$$

Therefore, the assumption applies to this MDP. Since $\pmb { \cal S } \times \mathcal { A } = \{ ( s , a ) \}$ , one has

$$
\overline { { \mathbf { M M D } } } _ { k } ^ { 2 } ( \eta _ { 1 } , \eta _ { 2 } ) = \mathbf { M M D } _ { k } ^ { 2 } \big ( \eta _ { 1 } ( s , a ) , \eta _ { 2 } ( s , a ) \big )
$$

for all $\eta _ { 1 } , \eta _ { 2 } \in { \mathcal { C } } _ { E }$ . The assumed τ -contraction on $\mathcal { C } _ { E }$ therefore implies

$$
\mathbf { M M D } _ { k } ^ { 2 } \big ( f _ { q } \# \nu , f _ { q } \# \mu \big ) \leq \tau \mathbf { M M D } _ { k } ^ { 2 } ( \nu , \mu ) \qquad \forall \mu , \nu \in \mathcal { P } ( \mathbb { R } ^ { d } ) \mathrm { ~ w i t h ~ s u p p } \mu , \operatorname { s u p p } \nu \subseteq E .
$$

Applying Theorem C.1 with the set E, we conclude that $k _ { \mathrm { d i f f } } ^ { q , \gamma } = \tau k - k _ { f _ { q } }$ is CPD on $E .$ . Since $q$ was arbitrary among those satisfying $q + \gamma E \subseteq E$ , this proves the necessity statement. □

Lemma C.5 (CPD on a set implies CPD on all subsets). Let X be a measurable space, let $k :$ $X \times X \to \mathbb { R }$ be symmetric, and let $F \subseteq E \subseteq X$ be measurable. If k is CPD on $E ,$ , then k is CPD on $F .$

Proof. Assume that k is CPD on $E .$ . Let $\mu$ be any finite signed measure supported on $F$ with $\mu ( F ) = 0$ . Since $F \subseteq E$ , the measure $\mu$ is also supported on $\bar { E } .$ Moreover,

$$
\mu ( { \cal E } ) = \mu ( { \cal F } ) = 0 ,
$$

because $\mu$ vanishes outside $F .$ . Since k is CPD on $E .$ , it follows that

$$
\int \int _ { E \times E } k ( x , y ) d \mu ( x ) d \mu ( y ) \geq 0 .
$$

As $\mu$ is supported on $F .$ , this is the same as

$$
\iiint _ { F \times F } k ( x , y ) d \mu ( x ) d \mu ( y ) \geq 0 .
$$

Hence k is CPD on $F .$

□

## D Contraction on a bounded support: numerical experiments

Preamble. This section turns the criterion of Theorem C.3, CPD on the bounded set $E ,$ into a numerical procedure for estimating $\tau _ { E } ^ { \star } ( \lambda )$ . Section D.1 replaces $E = B ( 0 , R )$ by random finite sets of points in $E ,$ which by Lemma C.5 give lower bounds on $\tau _ { E } ^ { \star } ( \lambda )$ that can only improve as more sets are drawn. Section D.2 reduces the CPD condition on a finite support to a generalized eigenvalue problem, solved by the symmetric reduction of Algorithm 1. Section D.4 validates the procedure against the power-distance and generalized multiquadric kernels of [19, 25], whose contraction factors are known. Section D.3 gives the settings behind Figure 2, which applies the procedure to the RBF and logarithmic kernels.

## D.1 From a bounded to a discrete support condition

By Theorem C.1, for a fixed map, $\mathbf { M M D ^ { 2 } }$ contraction on a set $E \subseteq \mathbb { R } ^ { d }$ is equivalent to the corresponding difference kernel being CPD on $E .$ . Theorem C.3 gives Bellman contraction on $E$ when this holds for the map $x \mapsto q +$ γx of every reward value $q .$

For a translation-invariant kernel, $k ( q + \lambda x , q + \lambda y ) = k ( \lambda x , \lambda y )$ , so the difference kernel is the same for every reward value $q .$ . We therefore take $q = 0$ and $f ( x ) = \lambda x .$ , and the relevant difference kernel is

$$
k _ { \mathrm { d i f f } , \tau } ^ { 0 , \lambda } ( x , y ) : = \tau k ( x , y ) - k ( \lambda x , \lambda y ) , \qquad \lambda \in ( 0 , 1 ) .
$$

For each measurable set $E \subseteq \mathbb { R } ^ { d }$ , define

$$
\tau _ { E } ^ { \star } ( \lambda ) : = \operatorname* { i n f } \Big \{ \tau > 0 : k _ { \mathrm { d i f f } , \tau } ^ { 0 , \lambda } \mathrm { i s } \mathrm { C P D o n } E \Big \} .
$$

By Lemma C.5, CPD on a set implies CPD on every subset, so $\tau _ { E } ^ { \star } ( \lambda )$ can only increase when E grows. In particular,

$$
\tau _ { \mathbb { R } ^ { d } } ^ { \star } ( \lambda ) \geq \tau _ { E } ^ { \star } ( \lambda ) \qquad \mathrm { f o r e v e r y m e a s u r a b l e } E \subseteq \mathbb { R } ^ { d } .
$$

By Theorem 3.1, for a translation-invariant kernel and $\lambda = \gamma , \tau _ { \parallel \parallel } ^ { \star } ( \gamma )$ is the smallest contraction factor $\tau _ { k } ^ { \star } ( \mathcal { T } ^ { \pi } )$ of the main text. Thus, requiring CPD only on a bounded set $E$ is less restrictive than requiring it on all of $\mathbb { R } ^ { d }$ , so the condition can hold on E for a kernel that fails it on all of $\mathbb { R } ^ { d }$ . In reinforcement learning, Proposition C.1 shows that under bounded rewards, return laws are supported in a ball $E = B _ { M }$ with $M \overset { \cdot } { \geq } r _ { \operatorname* { m a x } } / ( 1 - \gamma )$ ).

The CPD condition on $E$ concerns all signed measures supported on $E ,$ so it cannot be checked numerically. We test it instead on J finite sets of points drawn at random in $E _ { \mathrm { { : } } }$

$$
S _ { j } = \{ z _ { 1 } ^ { ( j ) } , \dots , z _ { N } ^ { ( j ) } \} \subseteq E , \qquad j = 1 , \dots , J ,
$$

and define the contraction threshold of each set,

$$
\tau _ { S _ { j } } ^ { \star } ( \lambda ) : = \operatorname* { i n f } \Big \{ \tau > 0 : k _ { \mathrm { d i f f } , \tau } ^ { 0 , \lambda } \mathrm { i s } \mathrm { C P D } \mathrm { o n } S _ { j } \Big \} .
$$

Since each $S _ { j }$ is a subset of $E ,$ Lemma C.5 gives

$$
\tau _ { \mathbb { R } ^ { d } } ^ { \star } ( \lambda ) \geq \tau _ { E } ^ { \star } ( \lambda ) \geq \operatorname* { m a x } _ { 1 \leq j \leq J } \tau _ { S _ { j } } ^ { \star } ( \lambda ) .
$$

So the largest of these thresholds is a lower bound on $\tau _ { E } ^ { \star } ( \lambda )$ . It can only increase with J, because the maximum is taken over more sets.

## D.2 From a support condition to a matrix test

Fix a finite set of points $S = \{ z _ { 1 } , . . . , z _ { N } \} \subseteq E$ . We call it a finite support, because the CPD condition on S is the contraction inequality for laws whose mass sits on these points. The threshold $\tau _ { S } ^ { \star } ( \lambda )$ of S is determined by whether the difference kernel

$$
k _ { \mathrm { d i f f } , \tau } ^ { 0 , \lambda } ( x , y ) = \tau k ( x , y ) - k ( \lambda x , \lambda y )
$$

is CPD on S. We now express this condition in matrix form.

Checking the CPD condition on $S .$ Define the Gram matrices

$$
K \in \mathbb R ^ { N \times N } , \qquad K _ { i j } : = k ( z _ { i } , z _ { j } ) , \qquad K _ { \lambda } \in \mathbb R ^ { N \times N } , \qquad ( K _ { \lambda } ) _ { i j } : = k ( \lambda z _ { i } , \lambda z _ { j } ) ,
$$

and the corresponding difference matrix

$$
D _ { \tau } ^ { ( \lambda ) } : = \tau K - K _ { \lambda } .
$$

A signed measure on $S$ is $\begin{array} { r } { \sigma = \sum _ { i = 1 } ^ { N } w _ { i } \delta _ { z _ { i } } } \end{array}$ , with $\sigma ( S ) = \mathbf { 1 } _ { N } ^ { \top }$ w and $\begin{array} { r } { \iint _ { \mathbf { \Omega } } k _ { \mathrm { d i f f } , \tau } ^ { 0 , \lambda } d \sigma d \sigma = w ^ { \top } D _ { \tau } ^ { ( \lambda ) } w , } \end{array}$ The CPD condition on $S$ is therefore

$$
\begin{array} { r } { w ^ { \top } D _ { \tau } ^ { ( \lambda ) } w \geq 0 \qquad \mathrm { f o r ~ a l l ~ } w \in \mathbf { 1 } _ { N } ^ { \bot } , \qquad \mathbf { 1 } _ { N } ^ { \bot } : = \{ w \in \mathbb { R } ^ { N } : \mathbf { 1 } _ { N } ^ { \top } w = 0 \} . } \end{array}
$$

Using the centering projector $\begin{array} { r } { H _ { N } = I _ { N } - \frac { 1 } { N } \mathbf { 1 } _ { N } \mathbf { 1 } _ { N } ^ { \top } } \end{array}$ introduced in Section ${ \mathrm { A } } . 3$ , this is equivalent to

$$
H _ { N } D _ { \tau } ^ { ( \lambda ) } H _ { N } \succeq 0 .
$$

A symmetric matrix is positive semidefinite if and only if its smallest eigenvalue is nonnegative. Since $H _ { N } D _ { \tau } ^ { ( \lambda ) } H _ { N }$ is symmetric, the CPD condition on S therefore holds if and only if

$$
\lambda _ { \operatorname* { m i n } } \bigl ( H _ { N } D _ { \tau } ^ { ( \lambda ) } H _ { N } \bigr ) \geq 0 .
$$

Monotonicity in $\tau .$ . Fix $\tau ^ { \prime } \geq \tau$ and any $w \in { \bf 1 } _ { N } ^ { \perp }$ . Since $D _ { \tau } ^ { ( \lambda ) } = \tau K - K _ { \lambda }$ , we have

$$
\boldsymbol { w } ^ { \top } \boldsymbol { D } _ { \tau ^ { \prime } } ^ { ( \lambda ) } \boldsymbol { w } = \boldsymbol { w } ^ { \top } \boldsymbol { D } _ { \tau } ^ { ( \lambda ) } \boldsymbol { w } + \left( \tau ^ { \prime } - \tau \right) \boldsymbol { w } ^ { \top } \boldsymbol { K } \boldsymbol { w } .
$$

Thus, $\mathrm { i f } \tau ^ { \prime } > \tau$ and $w ^ { \top } K w > 0$ , increasing τ strictly increases $w ^ { \top } D _ { \tau } ^ { ( \lambda ) } w .$ . More generally, if the CPD condition on $S$ holds for $\tau ,$ , it also holds for every $\tau ^ { \prime } \geq \tau$ whenever

$$
\begin{array} { r } { w ^ { \top } K w \geq 0 \qquad \forall w \in \mathbf { 1 } _ { N } ^ { \bot } , } \end{array}
$$

that is, whenever k is CPD on $S .$ . This is the case for every sampled support when k is CPD on $\mathbb { R } ^ { d }$

Working directly with vectors whose entries sum to zero. The CPD condition on $S$ concerns the quadratic form w ${ } ^ { \top } D _ { \tau } ^ { ( \lambda ) }$ w only for weights w satisfying $\begin{array} { r } { \mathbf { 1 } _ { N } ^ { \top } w = 0 . } \end{array}$ , that is, $w \in \mathbf { 1 } _ { N } ^ { \perp }$ . Although the centering projector $H _ { N }$ encodes this constraint, we do not form it explicitly. Instead, we choose an orthonormal basis $Q \in \mathbb { R } ^ { N \times ( N - 1 ) } \ \mathrm { o f } \ \mathbf { 1 } _ { N } ^ { \perp }$ , so that

$$
Q ^ { \top } Q = I _ { N - 1 } , \qquad \mathrm { R a n g e } ( Q ) = { \bf 1 } _ { N } ^ { \perp } .
$$

Since every $w \in { \bf 1 } _ { N } ^ { \perp }$ is $w = Q v$ for some $v \in \mathbb { R } ^ { N - 1 }$ , and $( Q v ) ^ { \top } D _ { \tau } ^ { ( \lambda ) } ( Q v ) = v ^ { \top } \big ( Q ^ { \top } D _ { \tau } ^ { ( \lambda ) } Q \big ) \colon$ v, the CPD condition on S holds if and only if $Q ^ { \top } D _ { \tau } ^ { ( \lambda ) } Q \succeq 0$

Generalized eigenvalues for vectors whose entries sum to zero. Write the two Gram matrices restricted to $\mathbf { 1 } _ { N } ^ { \perp }$ in the basis Q,

$$
\begin{array} { r } { A : = Q ^ { \top } K Q , \qquad B : = Q ^ { \top } K _ { \lambda } Q , } \end{array}
$$

so that

$$
Q ^ { \top } D _ { \tau } ^ { ( \lambda ) } Q = \tau A - B .
$$

Hence the CPD condition on $S$ is equivalent to

$$
\tau A - B \succeq 0 .
$$

If A is strictly positive definite, meaning

$$
v ^ { \top } A v > 0 \qquad \mathrm { f o r e v e r y } v \neq 0 ,
$$

or equivalently, as shown in the next paragraph,

$$
w ^ { \top } K w > 0 \qquad \mathrm { f o r ~ e v e r y ~ n o n z e r o ~ } w \in { \bf 1 } _ { N } ^ { \bot } ,
$$

then, for every nonzero v,

$$
\begin{array} { r l r l } { v ^ { \top } ( \tau A - B ) v \geq 0 } & { { } } & { \iff } & { } \tau \geq \frac { v ^ { \top } B v } { v ^ { \top } A v } .  \end{array}
$$

Therefore,

$$
\tau _ { S } ^ { \star } ( \lambda ) = \operatorname* { s u p } _ { v \neq 0 } \frac { v ^ { \top } B v } { v ^ { \top } A v } = \lambda _ { \operatorname* { m a x } } ( B , A ) ,
$$

that ${ \mathrm { i s } } ,$ the smallest $\tau$ for which the CPD condition holds on $S$ is exactly the largest generalized eigenvalue of $( B , A )$

CSPD and strict positivity of A. The assumption $A \succ 0$ is equivalent to requiring that k be CSPD on the finite support S in the sense of Definition A.5. Indeed, since the columns of Q form an orthonormal basis of $\mathbf { 1 } _ { N } ^ { \perp }$ , every w $\in \mathbf { 1 } _ { N } ^ { \perp }$ can be written uniquely as $w = Q v$ with $v \in \dot { \mathbb { R } ^ { N - 1 } }$ , and w $\neq 0$ if and only if $v \neq 0$ . Therefore,

$$
\boldsymbol { v } ^ { \top } \boldsymbol { A } \boldsymbol { v } = \boldsymbol { v } ^ { \top } \boldsymbol { Q } ^ { \top } \boldsymbol { K } \boldsymbol { Q } \boldsymbol { v } = ( \boldsymbol { Q } \boldsymbol { v } ) ^ { \top } \boldsymbol { K } ( \boldsymbol { Q } \boldsymbol { v } ) = \boldsymbol { w } ^ { \top } \boldsymbol { K } \boldsymbol { w } .
$$

Hence A is strictly positive definite if and only if

$$
w ^ { \top } K w > 0 \qquad \mathrm { f o r ~ e v e r y ~ n o n z e r o ~ } w \in { \bf 1 } _ { N } ^ { \bot } ,
$$

that ${ \mathrm { i s } } ,$ if and only if k is CSPD on S. In particular, this condition is automatically satisfied if k is CSPD globally. By Proposition A.3, this is equivalent to $\mathbf { M M D } _ { k }$ being a metric, and we assume it in the rest of this section.

Reduction to an ordinary symmetric eigenvalue problem. Under the assumption that k is CSPD, A is strictly positive definite. We can therefore reduce the generalized eigenvalue problem to an ordinary symmetric one, using the eigendecomposition of A [26, Theorem 15.3.2, Chapter 15.4, method 2],

$$
A = P \Delta ^ { 2 } P ^ { \top } , \qquad \Delta : = \mathrm { d i a g } ( \mu _ { 1 } ^ { 1 / 2 } , \ldots , \mu _ { N - 1 } ^ { 1 / 2 } ) , \qquad \mu _ { i } > 0 .
$$

We then introduce the change of coordinates $u : = \Delta P ^ { \top } v$ , or equivalently $v = P \Delta ^ { - 1 } u$ . With $\begin{array} { r } { C : = \Delta ^ { - 1 } P ^ { \top } B P \Delta ^ { - 1 } } \end{array}$ , the generalized eigenvalue problem $B v = \tau A v$ is transformed into the ordinary symmetric eigenvalue problem $C u = \tau u$ . Indeed,

$$
\begin{array} { l } { B v = B P \Delta ^ { - 1 } u } \\ { \ = \tau A v } \\ { \ = \tau A P \Delta ^ { - 1 } u } \\ { \ = \tau P \Delta ^ { 2 } P ^ { \top } P \Delta ^ { - 1 } u } \\ { \ = \tau P \Delta u . } \end{array}
$$

Applying $\Delta ^ { - 1 } P ^ { \top }$ to both sides of the identity

$$
B P \Delta ^ { - 1 } u = \tau P \Delta u
$$

yields

$$
\Delta ^ { - 1 } P ^ { \top } B P \Delta ^ { - 1 } u = \tau u ,
$$

that is,

$$
C u = \tau u .
$$

So the threshold of S is the largest eigenvalue of $C ,$

$$
\tau _ { S } ^ { \star } ( \lambda ) = \lambda _ { \operatorname* { m a x } } ( C ) .
$$

Truncation of very small eigenvalues. Although A is strictly positive definite, some of its eigenvalues can be extremely small in practice. The factors $\mu _ { i } ^ { - 1 / 2 }$ in the reduction above then become very large and make the reduced problem numerically unstable, a difficulty also discussed in [26, Chapter 15.4.1, Remark 5]. We therefore discard very small eigenvalues before forming the reduced matrix, following the idea of [26, Chapter 15.5].

Concretely, we compute

$$
A = P \mathrm { d i a g } ( \mu _ { 1 } , \dots , \mu _ { N - 1 } ) P ^ { \top } ,
$$

choose a relative tolerance $\varepsilon _ { \mathrm { { e i g } } } > 0$ , and retain only the eigenvalues satisfying

$$
\mu _ { i } > \varepsilon _ { \mathrm { e i g } } \mu _ { \mathrm { m a x } } , \qquad \mu _ { \mathrm { m a x } } : = \operatorname* { m a x } _ { i } \mu _ { i } .
$$

Setting $\mu _ { i } ^ { - 1 / 2 } = 0$ for discarded eigenvalues, we define

$$
\Delta _ { \varepsilon } ^ { - 1 } : = \mathrm { d i a g } ( \nu _ { 1 } , \ldots , \nu _ { N - 1 } ) , \qquad \nu _ { i } = \left\{ \begin{array} { l l } { \mu _ { i } ^ { - 1 / 2 } , } & { \mu _ { i } > \varepsilon _ { \mathrm { e i g } } \mu _ { \mathrm { m a x } } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

Table 2: Numerical settings for the heatmaps of Figure 2.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Radius</td><td> $R = 1$ </td></tr><tr><td>Dimension</td><td> $d = 1$ </td></tr><tr><td>Support sampler</td><td>Uniform on the ball of radius R</td></tr><tr><td>Support size</td><td> $N = 5 1 2$ </td></tr><tr><td>Number of sampled supports</td><td> $J = 1 0$ </td></tr><tr><td>Parameter grid</td><td> $R / \ell \in [ 0 . 0 5 , 2 . 0 0 ] ,$  81 linearly spaced values</td></tr><tr><td>Scaling grid</td><td> $\lambda \in [ 0 , \bar { 1 } ]$  , 81 linearly spaced values</td></tr><tr><td>Eigenvalue tolerance  $\varepsilon _ { \mathrm { e i g } }$ </td><td> $1 0 ^ { - 9 }$ </td></tr><tr><td>Hatched cells (contraction ruled out)</td><td> $\widehat { \tau } _ { E } ^ { \star } ( \lambda ) > 1 + 1 0 ^ { - 6 }$ </td></tr></table>

We then form

$$
\begin{array} { r } { C _ { \varepsilon } : = \Delta _ { \varepsilon } ^ { - 1 } P ^ { \top } B P \Delta _ { \varepsilon } ^ { - 1 } . } \end{array}
$$

This gives the truncated reduced problem, and we define the corresponding numerical estimate by

$$
\widehat { \tau } _ { S } ^ { \star } ( \lambda ) : = \lambda _ { \operatorname* { m a x } } ( C _ { \varepsilon } ) .
$$

If no eigenvalues are discarded, then $\widehat { \tau } _ { S } ^ { \star } ( \lambda ) = \tau _ { S } ^ { \star } ( \lambda )$ . Otherwise, let V be the space spanned by the eigenvectors of A that are kept. Then

$$
\widehat { \tau } _ { S } ^ { \star } ( \lambda ) = \operatorname* { m a x } _ { v \in V , v \neq 0 } \frac { v ^ { \top } B v } { v ^ { \top } A v } \leq \operatorname* { m a x } _ { v \neq 0 } \frac { v ^ { \top } B v } { v ^ { \top } A v } = \tau _ { S } ^ { \star } ( \lambda ) ,
$$

since the first maximum is taken over fewer vectors. So the estimate can only underestimate the threshold.

## D.3 Experimental setup

Motivated by Proposition C.1, we take

$$
E = B ( 0 , R ) : = \{ x \in \mathbb { R } ^ { d } : \| x \| \leq R \} .
$$

Fix a support size N and a trial count J. For each $j \in \{ 1 , \dotsc , J \}$ , we draw an independent random support

$$
S _ { j } = \{ z _ { 1 } ^ { ( j ) } , \ldots , z _ { N } ^ { ( j ) } \} \subseteq B ( 0 , R ) , \qquad z _ { i } ^ { ( j ) } \overset { \mathrm { i . i . d . } } { \sim } \mathrm { U n i f } \big ( B ( 0 , R ) \big ) .
$$

Each sampled support $S _ { j }$ induces a threshold $\tau _ { S _ { j } } ^ { \star } ( \lambda )$ . Our implementation computes instead the numerical approximation

$$
\widehat { \tau } _ { S _ { j } } ^ { \star } ( \lambda ) \approx \tau _ { S _ { j } } ^ { \star } ( \lambda )
$$

obtained from the truncated symmetric problem described above. We report the largest of these values,

$$
\widehat { \tau } _ { E } ^ { \star } ( \lambda ) : = \operatorname* { m a x } _ { 1 \leq j \leq J } \widehat { \tau } _ { S _ { j } } ^ { \star } ( \lambda ) ,
$$

which estimates the lower bound ma $\mathrm { x } _ { j } \tau _ { S _ { j } } ^ { \star } ( \lambda )$ on $\tau _ { E } ^ { \star } ( \lambda )$ of Section D.1. Algorithm 1 summarizes this process, and Table 2 gives the numerical settings used for the heatmaps of Figure 2.

Compute resources. The experiments of this section were implemented in JAX [8] and run on CPU using float64 precision. For a fixed kernel configuration, computing one contraction curve took at most one minute.

## D.4 Results

Verification on known kernels. The method introduced in this section was designed to estimate a lower bound on $\tau _ { E } ^ { \star } ( \lambda )$ , where $E = B ( 0 , R )$ . As R increases, the set E approaches the ambient space, so the estimate on $B ( 0 , R )$ should approach the smallest contraction factor on all of $\mathbb { R } ^ { d }$

$$
\operatorname* { l i m } _ { R \to \infty } \tau _ { B ( 0 , R ) } ^ { \star } ( \lambda ) = \tau _ { \mathbb { R } ^ { d } } ^ { \star } ( \lambda ) .
$$

Algorithm 1 Monte Carlo sampling of $\widehat { \tau } _ { S _ { j } } ^ { \star } ( \lambda )$ from CPD tests on finite supports   
1: Require: Dimension $d \geq 1 ,$ , support size $N \geq 2 ,$ , trial count $J \geq 1$   
2: Require: Radius $R > 0$ defining $E = B ( 0 , \overline { { R } } ) \subset \mathbb { R } ^ { d }$   
3: Require: Scaling parameter $\lambda \in ( 0 , 1 )$   
4: Require: Symmetric kernel k : $\mathbf { \nabla } \cdot \mathbf { \dot { \mathbb { R } ^ { d } } } \times \mathbf { \dot { \mathbb { R } ^ { d } } } \to \mathbb { R }$   
5: Require: Relative tolerance $\varepsilon _ { \mathrm { { e i g } } } > 0$   
6: Output: Samples $\{ \widehat { \tau } _ { S _ { j } } ^ { \star } ( \lambda ) \} _ { j = 1 } ^ { J }$   
7: Compute an orthonormal basis $Q \in \mathbb { R } ^ { N \times ( N - 1 ) }$ of $\mathbf { 1 } _ { N } ^ { \perp }$   
8: for $j ^ { \bar { } } = 1 , \dots , J$ do   
9: Sample a support and build the corresponding restricted matrices.   
10: Draw $S _ { j } = \{ z _ { 1 } ^ { ( j ) } , \dots , z _ { N } ^ { ( j ) } \}$ with $z _ { i } ^ { ( j ) } \overset { \mathrm { i . i . d . } } { \sim } \operatorname { U n i f } ( B ( 0 , R ) )$   
11: Form the Gram matrices   
$K _ { a b } \gets k \Big ( z _ { a } ^ { ( j ) } , z _ { b } ^ { ( j ) } \Big ) , \qquad ( K _ { \lambda } ) _ { a b } \gets k \Big ( \lambda z _ { a } ^ { ( j ) } , \lambda z _ { b } ^ { ( j ) } \Big )$   
12: Restrict to $\mathbf { 1 } _ { N } ^ { \perp }$   
$A  Q ^ { \top } K Q , \qquad B  Q ^ { \top } K _ { \lambda } Q$   
13: Reduce the generalized eigenvalue problem to a truncated ordinary symmetric one.   
14: Compute an eigendecomposition   
$A = P \operatorname { d i a g } ( \mu _ { 1 } , \dots , \mu _ { N - 1 } ) P ^ { \top }$   
15: Set   
µ<sub>max</sub> $ \operatorname* { m a x } _ { 1 \leq i \leq N - 1 } \mu _ { i }$   
16: Set the truncated diagonal matrix   
$\Delta _ { \varepsilon } ^ { - 1 } \gets \mathrm { d i a g } ( \nu _ { 1 } , \dots , \nu _ { N - 1 } ) , \qquad \nu _ { i } = \left\{ \begin{array} { l l } { \mu _ { i } ^ { - 1 / 2 } , } & { \mu _ { i } > \varepsilon _ { \mathrm { e i g } } \mu _ { \mathrm { m a x } } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } } \end{array} \right.$   
17: Form the reduced symmetric matrix   
$\begin{array} { r } { C _ { \varepsilon } \gets \Delta _ { \varepsilon } ^ { - 1 } P ^ { \top } B P \Delta _ { \varepsilon } ^ { - 1 } } \end{array}$   
18: Extract the threshold estimatefrom the largest eigenvalue.   
19: Compute   
$\widehat { \tau } _ { S _ { j } } ^ { \star } ( \lambda ) \gets \lambda _ { \operatorname* { m a x } } ( C _ { \varepsilon } )$   
20: end for   
21: Return: $\{ \widehat { \tau } _ { S _ { j } } ^ { \star } ( \lambda ) \} _ { j = 1 } ^ { J }$   
To validate Algorithm 1, we apply it to the power-distance and generalized multiquadric kernels,   
whose smallest contraction factor across dimensions $\tau ^ { \star } ( \lambda )$ is known (Propositions F.1 and F.2)   
and bounds $\tau _ { \mathbb { R } ^ { d } } ^ { \star } ( \lambda )$ from above. Figure 4 shows the estimates obtained on $\bar { B } ( 0 , R )$ for increasing   
values of $R .$ For the power-distance kernels, the estimates already match $\tau ^ { \star } ( \lambda )$ . For the generalized   
multiquadric kernels, the estimates are smaller on small balls and approach $\tau ^ { \star } ( \lambda )$ as $\breve { R }$ increases.   
This is consistent with the fact that CPD on $B ( 0 , R )$ is weaker than CPD on the whole ambient space.

![](images/34991d000e5064640616d4b9f1743b29cd2876a552e8ed3c6e9bb4b57fc77e56.jpg)  
Figure 4: Estimates on balls B(0, R) for kernels with known smallest contraction factors across dimensions. As R increases, the estimates approach the theoretical factors.

## E Global contraction under radial kernels via Bernstein functions

Preamble. This section studies contraction under radial kernels. It turns the CPD criterion of Appendix C into conditions on the Bernstein function that generates the kernel.

Section E.1 explains why we focus on radial kernels and shows that contraction under an affine map then reduces to a CND condition on the difference profile (Proposition E.2). Section E.2 collects the necessary background, namely the definitions of Bernstein and complete Bernstein functions and Schoenberg’s theorem (Theorem E.1), by which a radial kernel is CND exactly when its profile is a Bernstein function. In Section E.3, Theorem E.2 shows that contraction under affine maps, in every dimension, holds exactly when the difference profile $\psi _ { \tau } = \tau \phi - \phi ( \lambda ^ { 2 } \cdot )$ is a Bernstein function, and Proposition E.3 shows that saturating profiles cannot yield strict contraction.

Section E.4 proves the results on Lévy measures needed to make the Bernstein criterion explicit. Section E.5 uses them to express the criterion as pointwise inequalities on the drift and the Lévy measure (Theorem E.4). Section E.6 treats complete Bernstein functions, for which the smallest contraction factor across dimensions reduces to a single supremum of the $\operatorname { L e v y }$ density ratio (Theorem $\operatorname { E } . 5 )$ Section E.7 introduces geometric chaining, which builds Bernstein profiles that induce contraction with a chosen contraction factor (Theorem E.6).

## E.1 Motivations for radial kernels

For scalar returns $( d = 1 )$ , the following proposition shows that a symmetric kernel that is invariant under translations is radial. Hence, in one dimension, focusing on radial kernels covers every translation-invariant kernel.

Proposition E.1 (A symmetric, translation-invariant univariate kernel is radial). Let $k : \mathbb { R } \times \mathbb { R } \to \mathbb { R }$

(i) Translation invariance: $k ( x + t , y + t ) = k ( x , y )$ for all $x , y , t \in \mathbb { R }$

(ii) Symmetry: $k ( x , y ) = k ( y , x )$ for all $x , y \in \mathbb { R } .$

Then (i) and (ii) hold if and only if there exists a function $g : [ 0 , \infty )  \mathbb { R }$ such that

$$
k ( x , y ) = g ( | x - y | ) \qquad \forall x , y \in \mathbb { R } .
$$

Proof.

(⇒) Necessity. Assume translation invariance. Define $\kappa : \mathbb { R }  \mathbb { R } \mathfrak { b } \mathfrak { y } \kappa ( r ) : = k ( r , 0 )$ . Then for any $x , y \in \mathbb { R }$ , applying translation invariance with $t = - y$ gives

$$
k ( x , y ) = k ( x - y , 0 ) = \kappa ( x - y ) .
$$

Assume also that k is symmetric. Then for all $r \in \mathbb { R }$

$$
\kappa ( r ) = k ( r , 0 ) = k ( 0 , r ) = k ( - r , 0 ) = \kappa ( - r ) ,
$$

where the step $k ( 0 , r ) = k ( - r , 0 )$ uses translation invariance with $t = - r .$ . Thus κ is even. Define $g : [ 0 , \infty ) $ R by $g ( u ) : = \kappa ( u )$ for $u \geq 0$ . Since κ is even, $\kappa ( r ) = g ( | r | )$ for all $r \in \mathbb { R }$ , and therefore

$$
k ( x , y ) = \kappa ( x - y ) = g ( | x - y | ) .
$$

(⇐) Sufficiency. Conversely, if $k ( x , y ) = g ( | x - y | )$ , then for any $t \in \mathbb { R }$

$$
k ( x + t , y + t ) = g ( | ( x + t ) - ( y + t ) | ) = g ( | x - y | ) = k ( x , y ) ,
$$

so k is translation invariant. Also,

$$
k ( y , x ) = g ( | y - x | ) = g ( | x - y | ) = k ( x , y ) ,
$$

so k is symmetric.

Proposition E.2 (For radial kernels, contraction under an affine map reduces to a CND condition). Fix $\bar { d } \geq 1$ . Assume $f \colon  { \mathbb { R } ^ { d } } \to  { \mathbb { R } ^ { d } }$ is an affine map of the form

$$
f ( x ) = q + \lambda x , \qquad q \in \mathbb { R } ^ { d } , \lambda \in \mathbb { R } .
$$

Let $k ( u , v ) = g ( \lVert u - v \rVert )$ be a radial kernel on $\mathbb { R } ^ { d }$ , and define

$$
h ( r ) = g ( 0 ) - g ( r ) , \quad r \geq 0 .
$$

As in Theorem C.1, assume k is CPD on $\mathbb { R } ^ { d }$ , so that $( u , v ) \ \mapsto \ h ( \lVert u - v \rVert )$ is CND on $\mathbb { R } ^ { d }$

Then, for any $\tau \in \mathbb { R }$ , the contraction inequality

$$
\mathbf { M M D } _ { k } ^ { 2 } \big ( f \# \nu , f \# \mu \big ) \ \leq \ \tau \mathbf { M M D } _ { k } ^ { 2 } ( \nu , \mu )
$$

holds for every $\mu , \nu \in \mathcal P ( \mathbb { R } ^ { d } )$ if and only if

$$
\Gamma _ { \tau } ( x , y ) = \tau h \big ( \| x - y \| \big ) - h \big ( | \lambda | \| x - y \| \big ) \mathrm { i s } \mathrm { C N D o n } \mathbb { R } ^ { d } .
$$

Moreover, $\Gamma _ { \tau } ( x , x ) = 0$

Proof. Since $f ( x ) = q + \lambda x .$ , we have

$$
\| f ( x ) - f ( y ) \| = \left| \lambda \right| \left\| x - y \right\| \quad { \mathrm { f o r ~ a l l ~ } } x , y \in \mathbb { R } ^ { d } .
$$

By Theorem C.1, applied with $E = \mathbb { R } ^ { d }$ , the contraction statement

$$
\mathbf { M M D } _ { k } ^ { 2 } \big ( f \# \nu , f \# \mu \big ) \ \leq \ \tau \mathbf { M M D } _ { k } ^ { 2 } ( \nu , \mu ) \qquad \forall \mu , \nu \in \mathcal { P } ( \mathbb { R } ^ { d } )
$$

is equivalent to $\Gamma _ { \tau }$ being CND on $\mathbb { R } ^ { d } .$ , where

$$
\Gamma _ { \tau } ( x , y ) = \tau h \big ( \| x - y \| \big ) - h \big ( \| f ( x ) - f ( y ) \| \big ) .
$$

Substituting $\| f ( x ) - f ( y ) \| = | \lambda | \| x - y \| ~ ;$ yields

$$
\Gamma _ { \tau } ( x , y ) = \tau h \big ( \| x - y \| \big ) - h \big ( | \lambda | \| x - y \| \big ) .
$$

Finally, $h ( 0 ) = g ( 0 ) - g ( 0 ) = 0 ,$ , hence

$$
\Gamma _ { \tau } ( x , x ) = \tau h ( 0 ) - h ( 0 ) = 0 .
$$

## E.2 Definitions and Schoenberg’s theorem

Definition E.1 (Bernstein function). A function $\phi \colon [ 0 , \infty )  [ 0 , \infty )$ is called a Bernsteinfunction [29, Def. 3.1] if it is continuous at 0, infinitely differentiable on $( 0 , \infty )$ , and its derivative is completely monotone,

$$
\phi \in C ^ { \infty } ( ( 0 , \infty ) ) , \quad ( - 1 ) ^ { n - 1 } { \frac { d ^ { n } } { d t ^ { n } } } \phi ( t ) \geq 0 \quad \forall n \geq 1 , t > 0 .
$$

Equivalently, it admits the Lévy–Khintchine representation [29, Thm. 3.2]

$$
\phi ( t ) = a + b t + \int _ { ( 0 , \infty ) } \bigl ( 1 - e ^ { - t s } \bigr ) \Lambda ( d s ) , \qquad a \geq 0 , b \geq 0 ,
$$

together with the integrability condition

$$
\int _ { ( 0 , \infty ) } \operatorname* { m i n } \{ 1 , s \} \Lambda ( d s ) < \infty ,
$$

and moreover, by [29, Rem. 3.3(iv)], one has

$$
a = \phi ( 0 + ) , \qquad b = \operatorname* { l i m } _ { t \to \infty } \frac { \phi ( t ) } { t } .
$$

Thus $b ,$ called the drift of $\phi ,$ , determines whether ϕ has linear or sublinear growth.

Definition E.2 (Complete Bernstein function). A Bernstein function ϕ (Definition E.1) is called a complete Bernstein function [29, Def. 6.1] if its Lévy measure Λ is absolutely continuous with a completely monotone density m,

$$
\Lambda ( d s ) = m ( s ) d s ,
$$

where

$$
m \in C ^ { \infty } ( ( 0 , \infty ) ) , \quad ( - 1 ) ^ { n } m ^ { ( n ) } ( s ) \geq 0 \quad \forall n \geq 0 , s > 0 .
$$

Theorem E.1 (Schoenberg’s characterization of radial CND kernels, restated via Bernstein functions [30]). Let $h \colon [ 0 , \infty ) \to [ \bar { 0 } , \infty )$ be continuous with $h ( 0 ) = 0$ , and define the radial kernel

$$
k ( x , y ) = h \big ( \| x - y \| \big ) , \quad x , y \in \mathbb R ^ { d } .
$$

Then k is conditionally negative definite on $\mathbb { R } ^ { d }$ for every $d \geq 1$ if and only if there exists a Bernstein function ϕ such that

$$
h ( r ) = \phi ( r ^ { 2 } ) , \quad r \geq 0 .
$$

In particular, ϕ admits a Lévy–Khintchine representation

$$
\phi ( t ) = a + b t + \int _ { ( 0 , \infty ) } \left( 1 - e ^ { - s t } \right) \Lambda ( d s ) , \quad a , b \geq 0 .
$$

Moreover, since $h ( 0 ) = 0$ , evaluating at $r = 0 { \mathrm { g i v e s } } 0 = h ( 0 ) = \phi ( 0 ) = a$ , which in turn yields

$$
\boxed { \phi ( t ) = b t + \int _ { ( 0 , \infty ) } \bigl ( 1 - e ^ { - s t } \bigr ) \Lambda ( d s ) , \quad h ( r ) = b r ^ { 2 } + \int _ { ( 0 , \infty ) } \bigl ( 1 - e ^ { - s r ^ { 2 } } \bigr ) \Lambda ( d s ) . }
$$

## E.3 Bernstein reduction of difference kernel

Preamble. In this subsection, the contraction condition on the difference kernel is translated into a condition on a Bernstein function. The key point is that, in order for Schoenberg’s theorem to apply, the relevant radial kernel must be conditionally negative definite on $\mathbb { R } ^ { d }$ for every $d \geq 1$ , not only in one fixed dimension. Theorem E.2 shows that, when this holds in every dimension, the contraction condition is equivalent to the difference profile being a Bernstein function. Theorem E.3 then rewrites this criterion directly in terms of the drift term and the Lévy measure in the $\mathrm { L e v y - }$ Khintchine representation. Finally, Proposition E.3 shows that saturating profiles cannot yield strict MMD contraction under affine maps.

Theorem E.2 (For radial kernels, contraction in every dimension reduces to a Bernstein condition). Fix $\lambda \in \mathbb { R }$ . Let $g : [ 0 , \infty )  \mathbb { R }$ be continuous, and assume that, for every $d \geq 1$ , the radial kernel

$$
k ( u , v ) = g ( \| u - v \| ) , \qquad u , v \in \mathbb R ^ { d } ,
$$

is CPD on $\mathbb { R } ^ { d } .$ . Define

$$
h ( r ) = g ( 0 ) - g ( r ) , \qquad r \geq 0 .
$$

Then, for every $d \geq 1$ , the kernel $( u , v ) \mapsto h ( \| u - v \| )$ is CND on $\mathbb { R } ^ { d }$ . Hence, by Theorem E.1, there exists a Bernstein function ϕ such that

$$
h ( r ) = \phi ( r ^ { 2 } ) , \qquad r \geq 0 .
$$

For each $d \geq 1$ and $q \in \mathbb { R } ^ { d }$ , consider the affine map

$$
f ( x ) = q + \lambda x , \qquad x \in \mathbb { R } ^ { d } .
$$

In the first statement below, the kernel k is understood as the above radial kernel on the ambient space $\mathbb { R } ^ { d }$ . Then, for any $\tau \in \mathbb { R }$ , the following are equivalent.

$$
\begin{array} { r l } & { \mathbf { M M D } _ { k } ^ { 2 } \big ( f \# \nu , f \# \mu \big ) \ \leq \ \tau \mathbf { M M D } _ { k } ^ { 2 } ( \nu , \mu ) } \\ & { \quad \mathrm { ~ f o r ~ e v e r y ~ } d \geq 1 \ \mathrm { a n d ~ e v e r y } \ \mu , \nu \in \mathcal { P } ( \mathbb { R } ^ { d } ) } \\ & { \qquad \mathrm { i f ~ a n d ~ o n l y ~ i f ~ } } \\ & { \Gamma _ { \tau } ( x , y ) = \tau h ( \| x - y \| ) - h ( \| x \| \| x - y \| ) \ \mathrm { i s ~ C N D } } \\ & { \qquad \mathrm { o n ~ } \mathbb { R } ^ { d } \mathrm { ~ f o r ~ e v e r y ~ } d \geq 1 } \\ & { \qquad \mathrm { i f ~ a n d ~ o n l y ~ i f ~ } } \\ & { \psi _ { \tau } ( t ) = \tau \phi ( t ) - \phi ( \lambda ^ { 2 } t ) \ \mathrm { i s ~ B e r n s t e i n ~ f u n c t i o n } . } \end{array}
$$

Proof. Since, for every $d \geq 1$ , the kernel

$$
k ( u , v ) = g ( \lVert u - v \rVert )
$$

is CPD on $\mathbb { R } ^ { d }$ , the kernel

$$
( u , v ) \mapsto h ( \| u - v \| ) = g ( 0 ) - g ( \| u - v \| )
$$

is CND on $\mathbb { R } ^ { d }$ for every $d \geq 1$ . Theorem E.1 therefore yields a Bernstein function $\phi$ such that

$$
h ( r ) = \phi ( r ^ { 2 } ) , \qquad r \geq 0 .
$$

Fix $\tau \in \mathbb { R }$ , and let $d \geq 1$ and $q \in \mathbb { R } ^ { d }$ . By Proposition E.2, applied in the fixed ambient dimension d to the affine map $f ( x ) = q + \lambda x$ , the first statement is equivalent to

$$
\Gamma _ { \tau } ( x , y ) = \tau h ( \| x - y \| ) - \ h ( | \lambda | \| x - y \| )
$$

being CND on $\mathbb { R } ^ { d } . \operatorname { A s } d \geq 1$ was arbitrary, the first two statements are equivalent.

Finally, using $h ( r ) = \phi ( r ^ { 2 } )$ , we may rewrite

$$
\Gamma _ { \tau } ( x , y ) = \tau \phi ( \| x - y \| ^ { 2 } ) - \phi ( \lambda ^ { 2 } \| x - y \| ^ { 2 } ) .
$$

Thus $ { \Gamma } _ { \tau } ( x , y )$ depends on x and y only through $\| x - y \| ^ { 2 }$ . We therefore define the difference profile $\psi _ { \tau } : [ 0 , \infty ) \to \mathbb { R }$ by

$$
\psi _ { \tau } ( t ) = \tau \phi ( t ) - \phi ( \lambda ^ { 2 } t ) .
$$

With this notation,

$$
\Gamma _ { \tau } ( x , y ) = \psi _ { \tau } ( \| x - y \| ^ { 2 } ) .
$$

Applying Theorem E.1 once more, $\Gamma _ { \tau }$ is CND on $\mathbb { R } ^ { d }$ for every $d \geq 1$ if and only if $\psi _ { \tau }$ is a Bernstein function. □

Theorem E.3 (For the difference kernel, CND in every dimension reduces to a Lévy–Khintchine criterion). Let $\phi : [ 0 , \infty )  [ 0 , \infty )$ be a Bernstein function with $\phi ( 0 ) = 0$ , admitting the Lévy– Khintchine representation

$$
\phi ( t ) = \ b t \ + \ \int _ { 0 } ^ { \infty } \bigl ( 1 - e ^ { - s t } \bigr ) \Lambda ( d s ) ,
$$

where $b \geq 0$ and $\Lambda$ is a nonnegative Borel measure on $( 0 , \infty )$ . Fix $\lambda \in \mathbb { R } \setminus \{ 0 \}$ and $\tau \in \mathbb { R }$ , and define the difference profile $\psi _ { \tau }$ and the kernel Γ<sub>τ</sub> built from it by

$$
\psi _ { \tau } ( t ) : = \tau \phi ( t ) - \phi ( \lambda ^ { 2 } t ) , \qquad \Gamma _ { \tau } ( x , y ) = \psi _ { \tau } \big ( \| x - y \| ^ { 2 } \big ) , \qquad x , y \in \mathbb { R } ^ { d } .
$$

Then

$$
\begin{array} { r } { \boxed { \begin{array} { c c l l } { ( i ) } & { \Gamma _ { \tau } \mathrm { ~ i s ~ C N D ~ o n ~ } \mathbb { R } ^ { d } \mathrm { ~ f o r ~ e v e r y ~ } d \ge 1 , } \\ & { \qquad \mathrm { ~ i f ~ a n d ~ o n l y ~ i f ~ } } \\ & { \qquad \quad } \\ { ( i i ) } & { b ( \tau - \lambda ^ { 2 } ) \ge 0 } & { \mathrm { a n d } } & { \tau \Lambda ( B ) \ge \Lambda \big ( B / \lambda ^ { 2 } \big ) } & { \mathrm { f o r ~ e v e r y ~ B o r e l ~ s e t ~ } B \subset ( 0 , \infty ) . } \end{array} } } \end{array}
$$

Proof.

(i)⇒(ii). Assume $\Gamma _ { \tau }$ is CND on $\mathbb { R } ^ { d }$ for every d $\geq 1$ . Using the Lévy–Khintchine representation of $\phi ,$

$$
\psi _ { \tau } ( t ) = \underbrace { b ( \tau - \lambda ^ { 2 } ) } _ { \mathrm { d r i f t } } t + \int _ { ( 0 , \infty ) } \left( 1 - e ^ { - s t } \right) \underbrace { [ \tau \Lambda - \Lambda _ { \lambda } ] } _ { \mathrm { L \ ' { s v y m e a s u r e } } } ( d s ) , \qquad \Lambda _ { \lambda } ( B ) : = \Lambda ( B / \lambda ^ { 2 } ) .
$$

Since $\Gamma _ { \tau }$ is CND on $\mathbb { R } ^ { d }$ for every $d \geq 1$ , Theorem E.1 implies that $\psi _ { \tau }$ is Bernstein. Its drift and its Lévy measure are therefore nonnegative, hence

$$
b ( \tau - \lambda ^ { 2 } ) \geq 0 \quad \mathrm { a n d } \quad \tau \Lambda ( B ) \geq \Lambda ( B / \lambda ^ { 2 } ) \mathrm { ~ f o r ~ e v e r y ~ B o r e l ~ } B \subset ( 0 , \infty ) .
$$

(ii)⇒(i). Assume

$b ( \tau - \lambda ^ { 2 } ) \geq 0 \quad { \mathrm { a n d } } \quad \tau \Lambda ( B ) \geq \Lambda ( B / \lambda ^ { 2 } )$ for every Borel $B \subset ( 0 , \infty )$

Then the same Lévy–Khintchine representation shows that

$$
\psi _ { \tau } ( t ) = \tau \phi ( t ) - \phi ( \lambda ^ { 2 } t )
$$

is a Bernstein function. Applying Schoenberg’s theorem again shows that

$$
\Gamma _ { \tau } ( x , y ) = \psi _ { \tau } \big ( \| x - y \| ^ { 2 } \big )
$$

is CND on $\mathbb { R } ^ { d }$ for every $d \geq 1$

Proposition $\mathbf { E } . 3 \left( \mathrm { A } \right.$ saturating radial profile cannot induce contraction). Adopt the setting of Theorem E.2. Fix $d \geq 1$ and $\lambda \in ( 0 , 1 )$ , and define the affine map

$$
f ( x ) = q + \lambda x , \qquad x \in \mathbb { R } ^ { d } .
$$

Let $\phi : [ 0 , \infty )  [ 0 , \infty )$ satisfy $\phi ( 0 ) = 0$ and define the radial kernel

$$
k ( x , y ) = \phi ( 0 ) - \phi ( \| x - y \| ^ { 2 } ) = - \phi ( \| x - y \| ^ { 2 } ) .
$$

Assume that $\phi$ is saturating, meaning that lim $_ { 1 _ { t } \to \infty } \phi ( t ) < \infty$ . Then, for every $\tau \in ( 0 , 1 )$ ), no strict $\mathbf { M M D ^ { 2 } }$ contraction under $f$ can hold unless $\phi \equiv 0$ . More precisely, for any $\tau \in ( 0 , 1 )$ , define the difference profile $\psi _ { \tau }$ and the kernel $\Gamma _ { \tau }$ built from it by

$$
\psi _ { \tau } ( t ) = \tau \phi ( t ) - \phi ( \lambda ^ { 2 } t ) , \qquad \Gamma _ { \tau } ( x , y ) = \psi _ { \tau } ( \| x - y \| ^ { 2 } ) .
$$

$\mathrm { I f } \phi \neq 0$ and ϕ is saturating, then $\Gamma _ { \tau }$ is not conditionally negative definite on $\mathbb { R } ^ { d }$ . As a consequence, there is no $\tau \in ( 0 , 1 )$ such that

$$
\mathbf { M } \mathbf { M } \mathbf { D } _ { k } ^ { 2 } ( f _ { \# } \nu , f _ { \# } \mu ) \ \leq \ \tau \mathbf { M } \mathbf { M } \mathbf { D } _ { k } ^ { 2 } ( \nu , \mu )
$$

holds for all $\mu , \nu \in \mathcal P ( \mathbb { R } ^ { d } )$

Proof. Fix $\tau \in ( 0 , 1 )$ . If Γ were conditionally negative definite on $\mathbb { R } ^ { d } .$ , then for every choice of two points $x _ { 1 } , x _ { 2 } \in \mathbb { R } ^ { d }$ and weights $( w _ { 1 } , w _ { 2 } ) = ( 1 , - 1 )$ with $w _ { 1 } + w _ { 2 } = 0$ , we would have

$$
\sum _ { i , j = 1 } ^ { 2 } w _ { i } w _ { j } \Gamma _ { \tau } ( x _ { i } , x _ { j } ) \leq 0 .
$$

Let $r = \| x _ { 1 } - x _ { 2 } \|$ . Since $\Gamma _ { \tau } ( x _ { 1 } , x _ { 2 } ) = \psi _ { \tau } ( r ^ { 2 } )$ and $\Gamma _ { \tau } ( x _ { 1 } , x _ { 1 } ) = \Gamma _ { \tau } ( x _ { 2 } , x _ { 2 } ) = \psi _ { \tau } ( 0 )$ , a direct expansion gives

$$
\sum _ { i , j = 1 } ^ { 2 } w _ { i } w _ { j } \Gamma _ { \tau } ( x _ { i } , x _ { j } ) = 2 \psi _ { \tau } ( 0 ) - 2 \psi _ { \tau } ( r ^ { 2 } ) .
$$

Thus CND would imply

$$
\psi _ { \tau } ( r ^ { 2 } ) \geq \psi _ { \tau } ( 0 ) \qquad { \mathrm { f o r ~ a l l ~ } } r \geq 0 .
$$

Since $\phi ( 0 ) = 0$ , we have $\psi _ { \tau } ( 0 ) = \tau \phi ( 0 ) - \phi ( 0 ) = 0 \mathrm { { \Omega } }$ , hence

$$
\psi _ { \tau } ( t ) \geq 0 \qquad { \mathrm { f o r ~ a l l ~ } } t \geq 0 ,
$$

that is,

$$
\tau \phi ( t ) \geq \phi ( \lambda ^ { 2 } t ) \qquad \mathrm { f o r ~ a l l } t \geq 0 .
$$

Now assume $\phi$ is saturating and $\phi \not \equiv 0$ . Then the limit

$$
L : = \operatorname* { l i m } _ { t \to \infty } \phi ( t )
$$

exists and satisfies $L \in ( 0 , \infty )$ . Letting $t \to \infty$ in $\tau \phi ( t ) \geq \phi ( \lambda ^ { 2 } t )$ yields

$$
\tau L \geq L .
$$

Since $L > 0$ , this forces $\tau \geq 1$ , contradicting $\tau \in ( 0 , 1 )$ . Therefore, i ${ \mathrm { : } } \tau \in ( 0 , 1 )$ and $\phi$ is saturating with $\phi \not \equiv 0 ,$ , the kernel $\Gamma _ { \tau }$ cannot be CND on $\mathbb { R } ^ { d }$

Finally, by Proposition E.2, strict $\mathbf { M M D } ^ { 2 }$ contraction under the map $f ( x ) = q + \lambda x$ with factor $\tau \in ( 0 , 1 )$ would require $\Gamma _ { \tau }$ to be CND on $\mathbb { R } ^ { d }$ . This is impossible when ϕ is saturating and $\phi \not \equiv 0$ , so no such strict contraction can hold unless $\phi \equiv 0$ □

## E.4 Results on measures and the measure domination condition

Preamble. This subsection studies the condition

$$
\tau \Lambda ( B ) \geq \Lambda ( B / \rho ) \qquad \mathrm { f o r ~ a l l ~ B o r e l ~ s e t s ~ } B \subset ( 0 , \infty ) ,
$$

with $\rho = \lambda ^ { 2 }$ . Lemma E.2 reduces it to two pointwise inequalities, one on the density $m ( s )$ and one on the mass $\Lambda ( \{ x \} )$ ) at every point $x > 0$ . Lemma E.1 shows that a Lévy measure has at most countably many atoms. The remaining results describe the atoms of a measure that satisfies this condition. Lemma E.3 shows that a measure with at least one but finitely many atoms cannot satisfy the condition, so a measure that satisfies it has either no atoms or infinitely many. Lemma E.4 shows that a single geometric chain can occur, and Lemma E.5 partitions $( 0 , \infty )$ into disjoint geometric chains $\{ \bar { \rho ^ { n } } r : n \in \mathbb { Z } \}$ , indexed by their representative $r \in ( \rho , 1 ]$ . Proposition E.4 combines these results. When $\tau \leq \rho ,$ Λ has no atoms, and when $\tau > \rho ,$ , atoms can exist but the set of atoms must be a union of disjoint tails of geometric chains.

Lemma E.1 (Atoms of a Lévy measure are countable). Let Λ be a Lévy measure on $( 0 , \infty )$ , that is,

$$
\int _ { ( 0 , \infty ) } \operatorname* { m i n } \{ 1 , s \} \Lambda ( d s ) < \infty .
$$

Then the set of atoms $\{ x > 0 : \Lambda ( \{ x \} ) > 0 \}$ is at most countable.

Proof. For each $n \in \mathbb { N }$ let $E _ { n } = ( 1 / n , \infty )$ . By the Lévy condition,

$$
\int _ { ( 1 , \infty ) } 1 \Lambda ( d s ) = \int _ { ( 1 , \infty ) } \mathrm { m i n } \{ 1 , s \} \Lambda ( d s ) < \infty ,
$$

so $\Lambda ( ( 1 , \infty ) ) < \infty$ . Moreover,

$$
\int _ { ( 1 / n , 1 ] } 1 \Lambda ( d s ) \ \le \ n \int _ { ( 1 / n , 1 ] } s \Lambda ( d s ) \ \le \ n \int _ { ( 0 , 1 ] } s \Lambda ( d s ) \ \le \ n \int _ { ( 0 , \infty ) } \operatorname* { m i n } \{ 1 , s \} \Lambda ( d s ) \ < \ \infty ,
$$

hence $\Lambda ( ( 1 / n , 1 ] ) < \infty$ . Therefore $\Lambda ( E _ { n } ) < \infty$ for every n.

Fix n. We show that the set of atoms in $E _ { n }$ is countable by grouping atoms according to a lower bound on their mass. For each $\ell \in \mathbb { N } ,$ , define

$$
A _ { n , \ell } : = \{ x \in E _ { n } : \Lambda ( \{ x \} ) \geq 1 / \ell \} .
$$

Each $A _ { n , \ell }$ contains the atoms in $E _ { n }$ whose mass is at least $1 / \ell .$

We claim that $A _ { n , \ell }$ is finite for every ℓ. Indeed, if $A _ { n , \ell }$ had infinitely many points, then for any $M \in \mathbb { N }$ we could choose distinct $x _ { 1 } , \ldots , x _ { M } \in A _ { n , \ell } .$ . Since the singletons $\{ { \bar { x } } _ { i } \}$ are disjoint and contained in $E _ { n }$

$$
\Lambda ( E _ { n } ) \ \geq \ \sum _ { i = 1 } ^ { M } \Lambda ( \{ x _ { i } \} ) \ \geq \ M \cdot \frac { \textstyle 1 } { \ell } .
$$

Since M is arbitrary, letting $M \to \infty$ forces $\Lambda ( E _ { n } ) = \infty$ , contradicting $\Lambda ( E _ { n } ) < \infty$ . Hence each $A _ { n , \ell }$ is finite.

Now let $x \in E _ { n }$ be an atom with $\Lambda ( \{ x \} ) = \alpha > 0$ . Choose $\textstyle \ell > { \frac { 1 } { \alpha } }$ . Then $\Lambda ( \{ x \} ) \geq 1 / \ell ,$ , so $x \in A _ { n , \ell } .$ Therefore every atom in $E _ { n }$ belongs to some $A _ { n , \ell } ,$ , and

$$
\{ x \in E _ { n } : \Lambda ( \{ x \} ) > 0 \} \ = \ \bigcup _ { \ell = 1 } ^ { \infty } A _ { n , \ell } .
$$

Since this is a countable union of finite sets, the set of atoms in $E _ { n }$ is countable. Finally,

$$
\{ x > 0 : \Lambda ( \{ x \} ) > 0 \} \ = \ \bigcup _ { n = 1 } ^ { \infty } \{ x \in E _ { n } : \Lambda ( \{ x \} ) > 0 \}
$$

is a countable union of countable sets, hence at most countable.

Lemma E.2 (The measure domination condition reduces to an inequality on the density and one on the atoms). Let Λ be a nonnegative Borel measure on $( 0 , \infty )$ , locally finite in the sense that $\Lambda ( [ a , b ] ) < \infty$ for all $0 < a < b$ , admitting the decomposition

$$
\Lambda = \underbrace { m ( s ) d s } _ { \mathrm { \tiny { a b s o l u t e l y ~ c o n t i n u o u s ~ p a r t } } } + \sum _ { j \ge 1 } c _ { j } \delta _ { s _ { j } } \quad \mathrm { ( n o ~ s i n g u l a r ~ c o n t i n u o u s ~ p a r t ) } ,
$$

Here $m ( s ) \geq 0$ for almost every $s > 0$ , and each $c _ { j } > 0 .$ . Since Λ is locally finite, m is locally integrable on $( 0 , \infty )$ , that is, $\begin{array} { r } { \int _ { a } ^ { b } m ( s ) d s \leq \Lambda ( [ a , b ] ) < \infty } \end{array}$ for all $0 < a < b$

Fix $\lambda \in ( 0 , 1 ]$ , set $\rho = \lambda ^ { 2 }$ , and define, for every Borel set $B \subset ( 0 , \infty )$

$$
\Lambda _ { \lambda } ( B ) : = \Lambda \bigl ( B / \rho \bigr ) , \qquad B / \rho = \{ s > 0 : \rho s \in B \} .
$$

Let $\tau \in ( 0 , 1 )$ . Then

$$
\begin{array} { r l } { ( i ) } & { \tau \Lambda ( B ) \geq \Lambda _ { \lambda } ( B ) \mathrm { f o r e v e r y ~ B o r e l ~ s e t } B \subset ( 0 , \infty ) , } \\ & { \mathrm { i f ~ a n d ~ o n l y i f } } \\ { ( i i ) } & { \left\{ ( 1 ) \begin{array} { l l } { ( 1 ) } & { \tau \rho m ( s ) \geq m \big ( s / \rho ) \qquad \mathrm { f o r ~ a l m o s t ~ e v e r y ~ } s > 0 , } \\ { ( 2 ) } & { \tau \Lambda ( \{ x \} ) \geq \Lambda \big ( \{ x / \rho \} \big ) \quad \mathrm { f o r ~ e v e r y ~ } x > 0 . } \end{array} \right. } \end{array}
$$

Proof.

Necessity $( \mathbf { i } ) \Rightarrow ( \mathbf { i } \mathbf { i } )$

1. The inequality on the atoms. For any $x > 0 .$ , take $B = \{ x \}$ . Since $\{ x \} / \rho = \{ x / \rho \}$

$$
\Lambda _ { \lambda } ( \{ x \} ) = \Lambda ( \{ x / \rho \} ) .
$$

Thus the hypothesis $\tau \Lambda ( B ) \ge \Lambda _ { \lambda } ( B )$ gives

$$
\tau \Lambda ( \{ x \} ) \geq \Lambda ( \{ x / \rho \} ) \qquad { \mathrm { f o r ~ a l l ~ } } x > 0 .
$$

2. The inequality on the density. Let $S = \{ x > 0 : \Lambda ( \{ x \} ) > 0 \} = \{ s _ { j } : j \geq 1 \}$ be the set of atoms, which is countable. Take a Borel set $B \subset ( 0 , \infty )$ that contains no point of S and no point of $\rho S = \{ \rho x : x \in S \}$ . Then neither B nor $B / \rho$ contains an atom, so only the absolutely continuous part contributes, and we have

$$
\Lambda ( B ) = \int _ { B } m ( s ) d s , \qquad \Lambda _ { \lambda } ( B ) = \Lambda ( B / \rho ) = \int _ { B / \rho } m ( s ) d s .
$$

Hence the assumption gives

$$
\tau \int _ { B } m ( s ) d s \geq \int _ { B / \rho } m ( s ) d s .
$$

Perform the change of variables $u = \rho s , \mathrm { s o } s = u / \rho$ and $\begin{array} { r } { d s = \frac { 1 } { \rho } d u } \end{array}$ . Then

$$
\int _ { B / \rho } m ( s ) d s = \int _ { \{ s : \rho s \in B \} } m ( s ) d s = { \frac { 1 } { \rho } } \int _ { B } m ( u / \rho ) d u .
$$

Thus for every Borel set B that contains no point of $S \cup \rho S$

$$
\tau \int _ { B } m ( s ) d s \geq \frac { 1 } { \rho } \int _ { B } m ( s / \rho ) d s .
$$

Finally, choose B to be a small interval around a point $s > 0 ,$ , minus the countably many points of $S \cup \rho \mathbf { \bar { { S } } } . \mathbf { A }$ countable set has length zero, so this changes neither |B| nor the integrals of m. Divide

both sides by |B|, and let the interval shrink to $\{ s \}$ . By the Lebesgue differentiation theorem, in the limit one obtains

$$
\tau m ( s ) \geq { \frac { 1 } { \rho } } m ( s / \rho ) \quad { \mathrm { f o r ~ a l m o s t ~ e v e r y ~ } } s > 0 ,
$$

or equivalently

$$
\tau \rho m ( s ) \geq m ( s / \rho ) .
$$

Sufficiency $( \mathbf { i } \mathbf { i } ) \Rightarrow ( \mathbf { i } )$ . Assume

$$
\tau \rho m ( s ) \geq m ( s / \rho ) { \mathrm { ~ a l m o s t ~ e v e r y w h e r e } } , \qquad \tau \Lambda ( \{ x \} ) \geq \Lambda ( \{ x / \rho \} ) \forall x > 0 .
$$

Set

$$
\Delta \Lambda = \tau \Lambda - \Lambda _ { \lambda } = \left[ \tau m ( s ) - \frac { 1 } { \rho } m ( s / \rho ) \right] d s + \sum _ { x > 0 \atop \Lambda ( \{ x \} ) + \Lambda ( \{ x / \rho \} ) > 0 } \left[ \tau \Lambda ( \{ x \} ) - \Lambda ( \{ x / \rho \} ) \right] \delta _ { x } .
$$

Here the sum runs over the countably many points x such that x or $x / \rho$ is an atom of $\Lambda ,$ , and each such x carries the mass $\tau \Lambda ( \{ x \} ) - \dot { \Lambda } ( \{ x / \dot { \rho } \} )$ .

1. Absolutely continuous part. By assumption ( $\begin{array} { r } { \mathrm { i . 1 ) , } \tau m ( s ) - \frac { 1 } { \rho } m ( s / \rho ) \geq 0 } \end{array}$ almost everywhere, so the absolutely continuous part of $\Delta \Lambda$ is nonnegative.

2. Atomic part. By assumption (ii.2), each mass $\tau \Lambda ( \{ x \} ) - \Lambda ( \{ x / \rho \} )$ is nonnegative, so all atoms of $\Delta \Lambda$ are nonnegative.

Hence $\Delta \Lambda$ is a nonnegative measure and $\tau \Lambda = \Lambda _ { \lambda } + \Delta \Lambda \geq \Lambda _ { \lambda }$

Lemma E.3 (A measure with at least one but finitely many atoms cannot satisfy the measure domination condition). Fix $\lambda \in ( 0 , 1 )$ and $\tau \in ( 0 , 1 )$ ), and let $\dot { \rho } = \lambda ^ { 2 }$ . Assume that a Borel measure Λ on $( 0 , \infty )$ satisfies the measure domination condition

$$
\tau \Lambda ( B ) \ \geq \ \Lambda _ { \lambda } ( B ) { \mathrm { ~ f o r ~ e v e r y ~ B o r e l ~ s e t ~ } } B \subset ( 0 , \infty ) , \qquad \Lambda _ { \lambda } ( B ) : = \Lambda ( B / \rho ) .
$$

Assume in addition that Λ has finitely many atoms, that is, $\begin{array} { r } { \Lambda = m ( s ) d s + \sum _ { j = 1 } ^ { N } c _ { j } \delta _ { s _ { \underline { { s } } } } } \end{array}$ with $m ( s ) \geq 0$ and $c _ { j } \geq 0$ . Then Λ has no atoms.

Proof. Suppose for contradiction that Λ has an atom at some $s ^ { \star } > 0 ,$ , and let $c ^ { \star } : = \Lambda ( \{ s ^ { \star } \} ) > 0$ . To use the measure domination condition, we choose the singleton $B = \{ \rho s ^ { \star } \}$ , since $\vec { B / } \rho \stackrel { \cdot \cdot } { = } \left\{ s ^ { \star } \right\}$ , so the right-hand side is exactly the mass of the atom at $s ^ { \star }$ . Indeed, by definition of $\Lambda _ { \lambda }$

$$
\Lambda _ { \lambda } ( \{ \rho s ^ { \star } \} ) = \Lambda ( \{ s ^ { \star } \} ) = c ^ { \star } .
$$

Applying the measure domination condition with $B = \{ \rho s ^ { \star } \}$ gives

$$
\tau \Lambda ( \{ \rho s ^ { \star } \} ) \ \geq \ \Lambda _ { \lambda } ( \{ \rho s ^ { \star } \} ) \ = \ c ^ { \star } ,
$$

so $\Lambda ( \{ \rho s ^ { \star } \} ) > 0$ . Thus, for the measure domination condition to hold, $\rho s ^ { \star }$ must also be an atom of Λ. Iterating, we obtain

$$
\Lambda ( \{ \rho ^ { n } s ^ { \star } \} ) > 0 \qquad \mathrm { f o r e v e r y } n \geq 1 .
$$

The locations $\rho ^ { n } s ^ { \star }$ are all distinct because $\rho \in ( 0 , 1 )$ , so Λ has infinitely many atoms. This contradicts the assumption that Λ has finitely many atoms. Hence Λ has no atoms. □

Lemma E.4 (Atoms on a single geometric chain satisfy the measure domination condition). Fix $\lambda \in ( 0 , 1 )$ , set $\rho = \lambda ^ { 2 }$ , and let $\tau \in ( 0 , 1 )$ . Fix $s _ { 0 } > 0$ and $c > 0$ , and define the measure

$$
\Lambda = \sum _ { n = 0 } ^ { \infty } a _ { n } \delta _ { s _ { n } } , \qquad s _ { n } = \rho ^ { n } s _ { 0 } , \qquad a _ { n } = c \tau ^ { - n } .
$$

Define, for every Borel set $B \subset ( 0 , \infty )$

$$
\Lambda _ { \lambda } ( B ) : = \Lambda \bigl ( B / \rho \bigr ) , \qquad B / \rho = \{ s > 0 : \rho s \in B \} .
$$

Then the measure domination condition

$$
\tau \Lambda ( B ) \geq \Lambda _ { \lambda } ( B ) \qquad \mathrm { f o r e v e r y B o r e l s e t } B \subset ( 0 , \infty )
$$

holds.

Moreover, Λ is a Lévy measure if and only i $: \tau > \rho .$ In particular, when $\tau > \rho ,$ , a single geometric chain gives an example satisfying the measure domination condition.

Proof.

Step 1: Verification of the measure domination condition. We first compute $\Lambda _ { \lambda }$ on the singletons that support the chain. For $n \geq 1$ , the previous point of the chain is $s _ { n - 1 } = s _ { n } / \rho ,$ so

$$
\Lambda _ { \lambda } ( \{ s _ { n } \} ) = \Lambda ( \{ s _ { n } \} / \rho ) = \Lambda ( \{ s _ { n } / \rho \} ) = \Lambda ( \{ s _ { n - 1 } \} ) = a _ { n - 1 } = \tau a _ { n } = \tau \Lambda ( \{ s _ { n } \} ) .
$$

For $n = 0 ,$ , we instead have

$$
\Lambda _ { \lambda } \big ( \{ s _ { 0 } \} \big ) = \Lambda \big ( \{ s _ { 0 } \} / \rho \big ) = \Lambda \big ( \{ s _ { 0 } / \rho \} \big ) = 0 ,
$$

since $s _ { 0 } / \rho$ does not belong to the support of Λ, whereas $\tau \Lambda ( \{ s _ { 0 } \} ) = \tau a _ { 0 } > 0 .$

We now pass from these singleton identities to an arbitrary Borel set $B \subset ( 0 , \infty )$ . Since Λ is a sum of atoms,

$$
\begin{array} { r l } { \mathbb { A } _ { 5 } ( { \boldsymbol B } ) - \lambda ( \delta ) + l ( \delta ) \dot { \boldsymbol \theta } ( \dot { \boldsymbol \theta } ) } & { \quad \mathrm ( 4 \mathrm { R e m ~ } { \boldsymbol \theta } ) \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } } \\ &  \quad = \displaystyle \sum _ { m = 0 } ^ { \infty } \alpha _ { m } \lambda _ { m } \alpha _ { m } ( \delta \alpha \dot { \boldsymbol \theta } ) , \quad \mathrm ~ ( \Delta \dot { \Omega } \dot { \dot { \boldsymbol \theta } } \dot { \dot \boldsymbol \theta } \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } \dot \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } \dot \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } \dot \boldsymbol \theta \dot { \theta } \dot \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } \dot \boldsymbol \theta \dot { \theta } \dot \boldsymbol \theta \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } \dot \boldsymbol \theta \dot { \theta } \dot \boldsymbol \theta \dot { \boldsymbol \theta } \dot { \boldsymbol \theta } \dot \boldsymbol \theta \dot { \theta } \dot \boldsymbol \theta \dot { \boldsymbol \theta } \dot \boldsymbol \theta \dot { \theta } \dot \boldsymbol \theta \dot { \boldsymbol \theta } \dot \boldsymbol \theta \dot { \theta } \dot \boldsymbol \theta \dot { \boldsymbol \theta } \dot \boldsymbol \theta \dot { \theta } \dot \boldsymbol \theta \dot { \boldsymbol \theta } \dot \boldsymbol \theta \dot \theta \dot { \boldsymbol \theta } \dot  \end{array}
$$

Thus the measure domination condition holds for all Borel $B \subset ( 0 , \infty )$ . Moreover, equality holds whenever $s _ { 0 } \notin B$ , since the only extra term added in the inequality above is the $j = 0$ contribution.

Step 2: Verification of the Lévy condition. By definition, Λ is a Lévy measure when it satisfies the Lévy integrability condition

$$
\int _ { ( 0 , \infty ) } \operatorname* { m i n } \{ 1 , s \} \Lambda ( d s ) < \infty .
$$

Since Λ puts the mass $a _ { n }$ at $s _ { n }$ , this integral equals

$$
\sum _ { n \geq 0 } a _ { n } \operatorname* { m i n } \{ 1 , s _ { n } \} .
$$

Since $\rho < 1$ , the atoms $s _ { n } = \rho ^ { n } s _ { 0 }$ decrease to 0, so there is an index K after which they are all at most 1, with $K = 0$ when $s _ { 0 } \leq 1$ . For $n \geq K$ , min $\{ 1 , s _ { n } \} = s _ { n }$ . The sum above therefore differs from

$$
\sum _ { n \geq 0 } a _ { n } s _ { n } = s _ { 0 } c \sum _ { n \geq 0 } \left( { \frac { \rho } { \tau } } \right) ^ { n } , \qquad { \mathrm { u s i n g ~ } } a _ { n } s _ { n } = c \tau ^ { - n } \rho ^ { n } s _ { 0 } ,
$$

only in its first K terms, each of them finite, so one is finite exactly when the other is. This geometric series converges if and only if $\rho / \tau < 1$ , that is, if and only if $\tau > \rho .$ Therefore Λ is a Lévy measure if and only $\operatorname { i f } \tau > \rho .$ □

Lemma E.5 (Every $x > 0$ admits a unique decomposition $x = \rho ^ { n } r$ with $r \in ( \rho , 1 ] ,$ . Fix $\rho \in ( 0 , 1 )$ . Then every $x > 0$ admits a unique decomposition

$$
x = \rho ^ { n } r , \qquad n \in \mathbb { Z } , \quad r \in ( \rho , 1 ] ,
$$

given by

$$
n = \left\lfloor { \frac { \log x } { \log \rho } } \right\rfloor , \qquad r = \rho ^ { - n } x .
$$

Equivalently, each geometric chain $\{ \rho ^ { n } x : n \in \mathbb { Z } \}$ meets the interval $( \rho , 1 ]$ in exactly one point. We write $n ( x )$ and $r ( x )$ for these n and $r ,$ and call $r ( x )$ the chain representative of x and $n ( x )$ its chain index.

Proof. Fix $x > 0$ . For an integer $n ,$ set $r = \rho ^ { - n } x ,$ so that $x = \rho ^ { n } r$ . Then $r \in ( \rho , 1 ]$ means

$$
\rho < \rho ^ { - n } x \leq 1 ,
$$

and multiplying by $\rho ^ { n } > 0$ , this is equivalent to

$$
\rho ^ { n + 1 } < x \leq \rho ^ { n } ,
$$

that is, to x lying in the interval $( \rho ^ { n + 1 } , \rho ^ { n } ]$ . For $n \in \mathbb { Z }$ , these intervals cover $( 0 , \infty )$ without overlapping,

$$
\therefore , \quad ( \rho ^ { 2 } , \rho ] , \quad \quad ( \rho , 1 ] , \quad \quad ( 1 , \rho ^ { - 1 } ] , \quad \quad ( \rho ^ { - 1 } , \rho ^ { - 2 } ] , \quad \quad \cdot \cdot \cdot
$$

To find the one that contains $x ,$ we take logarithms and divide by log $\rho < 0$ , which reverses the inequalities. The condition becomes

$$
n \leq { \frac { \log x } { \log \rho } } < n + 1 .
$$

Exactly one integer satisfies it, namely $n = \lfloor \log x / \log \rho \rfloor$ . Hence x has exactly one decomposition $x = \rho ^ { n } r$ with $n \in \mathbb { Z }$ and $r \in ( \rho , 1 ]$ , given by $n ( x ) = \lfloor \log x / \log \rho \rfloor$ and $r ( x ) = \rho ^ { - n ( x ) } x$

Finally, multiplying by $\rho$ moves a point from one interval to the next one on its left, so the chain $\{ \rho ^ { n } x \colon n \in { \bar { \mathbb { Z } } } \}$ has exactly one point in each interval, and in particular exactly one in $( \rho , 1 ]$ , namely $r ( x )$ □

Proposition E.4 (The measure domination condition rules out atoms of the Lévy measure or forces them onto infinite geometric chains). Let Λ be a Lévy measure on $( 0 , \infty )$ , that is,

$$
\int _ { ( 0 , \infty ) } \operatorname* { m i n } \{ 1 , s \} \Lambda ( d s ) < \infty .
$$

Assume Λ has no singular continuous part and admits the decomposition

$$
\Lambda ( d s ) = m ( s ) d s + \sum _ { j \geq 1 } c _ { j } \delta _ { s _ { j } } , \qquad m ( s ) \geq 0 \mathrm { ~ f o r ~ a l m o s t \ : e v e r y ~ } s > 0 , \quad c _ { j } \geq 0 .
$$

Fix $\lambda \in ( 0 , 1 )$ , set $\rho = \lambda ^ { 2 }$ , and let $\tau \in ( 0 , 1 )$ . Define, for every Borel set $B \subset ( 0 , \infty )$

$$
\Lambda _ { \lambda } ( B ) : = \Lambda \bigl ( B / \rho \bigr ) , \qquad B / \rho = \{ s > 0 : \rho s \in B \} .
$$

Then the measure domination condition

$$
\tau \Lambda ( B ) \geq \Lambda _ { \lambda } ( B ) \quad \mathrm { f o r e v e r y ~ B o r e l ~ s e t } B \subset ( 0 , \infty )
$$

holds if and only if

$$
\tau \rho m ( s ) \geq m ( s / \rho ) { \mathrm { ~ f o r ~ a l m o s t ~ e v e r y ~ } } s > 0 , \qquad \tau \Lambda ( \{ x \} ) \geq \Lambda ( \{ x / \rho \} ) { \mathrm { ~ f o r ~ e v e r y ~ } } x > 0 .
$$

Moreover, two cases occur, depending on whether $\tau \leq \rho \mathrm { o r } \tau > \rho .$

1. If $\mathit { \Pi } \ : \tau \leq \rho$ and the measure domination condition holds, then $c _ { j } = 0$ for all $j ,$ so Λ has no atoms.

2. $\operatorname { I f } \tau > \rho ,$ , then Λ can have atoms and still satisfy the measure domination condition, but its atoms must then follow the chains of Lemma E.5. Write every $x > 0$ as

$$
\begin{array} { r } { x = \rho ^ { n } r , \qquad n \in \mathbb { Z } , \quad r \in ( \rho , 1 ] , } \end{array}
$$

as in Lemma $\mathrm { E . 5 , }$ so that x lies on the chain $\{ \rho ^ { n } r : \ n \in \mathbb { Z } \}$ of its representative $r _ { \cdot }$ If the condition holds and $\rho ^ { n } r$ is an atom, then every point of that chain closer to $0 ,$ namely $\rho ^ { n + 1 } r , \rho ^ { n + 2 } r , . . . ,$ , is also an atom. Hence, on a chain that carries an atom, the atoms are exactly the points $\rho ^ { n } r$ with $n \geq n _ { r }$ , where $n _ { r }$ is the index of the first atom, or $n _ { r } = - \infty$ if every point of the chain is an atom. The set of atoms of $\Lambda$ is therefore a union of disjoint tails of geometric chains, of the form

$$
\{ \rho ^ { n } r : n \geq n _ { r } \} , \qquad r \in ( \rho , 1 ] ,
$$

one for each chain that carries an atom. In particular, Λ has either no atoms or infinitely many.

Such atoms can occur on the tail of a single geometric chain, by Lemma E.4, and also on the tails of finitely or countably many geometric chains at once, provided $\begin{array} { r } { \int _ { ( 0 , \infty ) } \operatorname* { m i n } \{ 1 , s \} \Lambda ( d s ) < \dot { \infty } . } \end{array}$

Proof.

From the measure domination condition to the inequalities on the density and on the atoms. By assumption, Λ has no singular continuous part. Hence Lemma E.2 gives the equivalence

$$
\tau \Lambda ( B ) \geq \Lambda _ { \lambda } ( B ) \quad \mathrm { f o r e v e r y ~ B o r e l } \ : B \subset ( 0 , \infty )
$$

$$
\begin{array} { r } { \left\{ \tau \rho m ( s ) \geq m ( s / \rho ) \mathrm { ~ f o r ~ a l m o s t ~ e v e r y ~ } s > 0 , \right. } \\ { \left. \tau \Lambda ( \{ x \} ) \geq \Lambda ( \{ x / \rho \} ) \forall x > 0 . \right. } \end{array}
$$

In addition, since min $\{ 1 , s \} = s$ for $s \leq 1$ and min $\{ 1 , s \} = 1$ for $s > 1$ , the decomposition of $\Lambda$ splits the Lévy condition into two parts,

$$
\int _ { ( 0 , \infty ) } \operatorname* { m i n } \{ 1 , s \} \Lambda ( d s ) = \Bigl ( \int _ { 0 } ^ { 1 } s m ( s ) d s + \sum _ { j : s _ { j } \leq 1 } s _ { j } c _ { j } \Bigr ) + \Bigl ( \int _ { 1 } ^ { \infty } m ( s ) d s + \sum _ { j : s _ { j } > 1 } c _ { j } \Bigr ) < \infty .
$$

Both parts are nonnegative, so each of them is finite,

$$
\int _ { 0 } ^ { 1 } s m ( s ) d s + \sum _ { j : s _ { j } \leq 1 } s _ { j } c _ { j } < \infty , \qquad \int _ { 1 } ^ { \infty } m ( s ) d s + \sum _ { j : s _ { j } > 1 } c _ { j } < \infty .
$$

We now turn to the two regimes for $\tau .$

$( 1 ) \mathbf { I f } \tau \leq \rho ,$ then Λ has no atoms. Assume for contradiction that there is an atom at $s _ { 0 } > 0$ with $\Lambda ( \{ s _ { 0 } \} ) = : c _ { 0 } > 0$ . To use the measure domination condition, as in Lemma E.3, we evaluate it on the singleton $B = \{ \rho s _ { 0 } \}$

$$
\tau \Lambda ( \{ \rho s _ { 0 } \} ) \geq \Lambda _ { \lambda } ( \{ \rho s _ { 0 } \} ) = \Lambda ( \{ s _ { 0 } \} ) = c _ { 0 } .
$$

Since the absolutely continuous part gives no mass to singletons, only a possible atom at $\rho s _ { 0 }$ can contribute there. Let

$$
c _ { 1 } : = \Lambda ( \{ \rho s _ { 0 } \} ) ,
$$

which is the mass carried by Λ at $\rho s _ { 0 }$ and is 0 if there is no atom at that point. The inequality above therefore gives

$$
c _ { 1 } \geq c _ { 0 } / \tau > 0 .
$$

There are two cases.

Case 1 (no atom at $\rho s _ { 0 } )$ . Then $\Lambda ( \{ \rho s _ { 0 } \} ) = 0$ , contradicting $c _ { 1 } \geq c _ { 0 } / \tau > 0$ . Hence this case cannot occur.

Case 2 (an atom at $\rho s _ { 0 } )$ . Then $c _ { 1 } = \Lambda ( \{ \rho s _ { 0 } \} ) > 0$ . Testing again on $B = \{ \rho ^ { 2 } s _ { 0 } \}$ yields

$$
\tau \Lambda ( \{ \rho ^ { 2 } s _ { 0 } \} ) \geq \Lambda ( \{ \rho s _ { 0 } \} ) = c _ { 1 } \quad \Rightarrow \quad \Lambda ( \{ \rho ^ { 2 } s _ { 0 } \} ) = : c _ { 2 } \geq \tau ^ { - 1 } c _ { 1 } \geq \tau ^ { - 2 } c _ { 0 } > 0 .
$$

Iterating, for all $n \geq 1 .$

$$
c _ { n } : = \Lambda ( \{ \rho ^ { n } s _ { 0 } \} ) \geq \tau ^ { - n } c _ { 0 } > 0 .
$$

Since $\rho \in ( 0 , 1 )$ , we have $\rho ^ { n } s _ { 0 } \leq 1$ for all sufficiently large n. Therefore the contribution of these atoms to $\int _ { ( 0 , 1 ] } \dot { s } \Lambda ( d s )$ satisfies

$$
\int _ { ( 0 , 1 ] } s \Lambda ( d s ) \geq \sum _ { n : \rho ^ { n } s _ { 0 } \leq 1 } ( \rho ^ { n } s _ { 0 } ) c _ { n } \geq s _ { 0 } c _ { 0 } \sum _ { n : \rho ^ { n } s _ { 0 } \leq 1 } \Bigl ( \frac { \rho } { \tau } \Bigr ) ^ { n } .
$$

$\operatorname { I f } \tau \leq \rho ,$ , then $\rho / \tau \geq 1$ and the series on the right diverges, contradicting the Lévy integrability $\textstyle \int _ { ( 0 , 1 ] } s \Lambda ( d s ) < \infty$ . Thus Λ has no atoms.

(2) If $\tau > \rho ,$ then the set of atoms is a union of disjoint tails of geometric chains. By Lemma E.5, every $x > 0$ admits a unique decomposition

$$
x = \rho ^ { n } r , \qquad n \in \mathbb { Z } , \quad r \in ( \rho , 1 ] .
$$

Fix $r \in ( \rho , 1 ]$ , and let

$$
A _ { r } : = \left\{ n \in \mathbb { Z } : \Lambda ( \{ \rho ^ { n } r \} ) > 0 \right\}
$$

denote the set of indices along that chain at which Λ has an atom. Then, once a level n on that chain carries an atom, the next level $n + 1$ must also carry an atom. Indeed, if $n \in A _ { r } .$ , then $\Lambda ( \{ \rho ^ { n } r \} ) > 0$ and the inequality on the atoms, (ii.2) in Lemma E.2, applied with $x = \rho ^ { n + 1 } r$ , gives

$$
\tau \Lambda ( \{ \rho ^ { n + 1 } r \} ) \geq \Lambda ( \{ \rho ^ { n } r \} ) > 0 ,
$$

hence $\Lambda ( \{ \rho ^ { n + 1 } r \} ) > 0$ , that is,

$$
n \in A _ { r } \quad \Longrightarrow \quad n + 1 \in A _ { r } .
$$

Equivalently, for each chain representative $r \in ( \rho , 1 ]$ , there is a threshold index $n _ { r } \in \mathbb { Z } \cup \{ - \infty , + \infty \}$ that describes where the atoms begin on that chain, in the sense that

$$
A _ { r } = \{ n \in \mathbb { Z } \colon n \geq n _ { r } \} .
$$

Thus, if $n _ { r } \in \mathbb { Z } ,$ then the first atom on the chain represented by r occurs at level $n _ { r } .$ , and all later levels n $\geq n _ { r }$ also carry atoms. The conventions $n _ { r } = + \infty$ and $n _ { r } = - \infty$ correspond respectively to $A _ { r } = \varnothing$ and $A _ { r } = \dot { \mathbb { Z } }$

Therefore, for each chain representative $r \in ( \rho , 1 ]$ , the atoms on that chain are exactly

$$
\{ \rho ^ { n } r : \ n \in A _ { r } \} = \{ \rho ^ { n } r : \ n \geq n _ { r } \} .
$$

Since every atom belongs to a unique chain represented by some $r \in ( \rho , 1 ]$ , the set of all atoms is obtained by taking the union over all such chains. Hence the set of atoms is a union of disjoint tails of geometric chains, of the form

$$
\{ \rho ^ { n } r : n \geq n _ { r } \} , \qquad r \in ( \rho , 1 ] .
$$

In particular, Λ has either no atoms or infinitely many, in agreement with Lemma E.3.

Conversely, when $\tau > \rho ,$ , a single geometric chain can occur by Lemma E.4. More generally, let I be finite or countably infinite, let $\bar { ( } r _ { i } ) _ { i \in I } \subset ( \rho , 1 ]$ be distinct chain representatives, and for each $i \in I$ let

$$
\Lambda ^ { ( i ) } = \sum _ { n = 0 } ^ { \infty } a _ { i , n } \delta _ { \rho ^ { n } r _ { i } }
$$

be a chain measure satisfying the measure domination condition. Assume that

$$
\sum _ { i \in I } \int _ { ( 0 , \infty ) } \operatorname* { m i n } \{ 1 , s \} \Lambda ^ { ( i ) } ( d s ) < \infty .
$$

In particular, every term of this sum is finite, so each $\Lambda ^ { ( i ) }$ is a Lévy measure. Then

$$
\Lambda : = \sum _ { i \in I } \Lambda ^ { ( i ) }
$$

is a Lévy measure whose set of atoms is a union of disjoint tails of these chains, and it still satisfies the measure domination condition. Indeed,

$$
\int _ { ( 0 , \infty ) } \operatorname* { m i n } \{ 1 , s \} \Lambda ( d s ) = \sum _ { i \in I } \int _ { ( 0 , \infty ) } \operatorname* { m i n } \{ 1 , s \} \Lambda ^ { ( i ) } ( d s ) < \infty ,
$$

so $\Lambda$ is a Lévy measure. Moreover, for every Borel set $B \subset ( 0 , \infty )$

$$
\tau \Lambda ( B ) = \sum _ { i \in I } \tau \Lambda ^ { ( i ) } ( B ) \geq \sum _ { i \in I } \Lambda _ { \lambda } ^ { ( i ) } ( B ) = \Lambda _ { \lambda } ( B ) ,
$$

where the last identity follows from

$$
\Lambda _ { \lambda } ( B ) = \Lambda ( B / \rho ) = \sum _ { i \in I } \Lambda ^ { ( i ) } ( B / \rho ) = \sum _ { i \in I } \Lambda _ { \lambda } ^ { ( i ) } ( B ) .
$$

Thus, when $\tau > \rho ,$ finitely or countably many disjoint geometric chains can indeed occur, provided the resulting measure is still a Lévy measure (that is, $\textstyle \int _ { ( 0 , \infty ) } \operatorname* { m i n } \{ 1 , s \} \Lambda ( d s ) < \infty )$ □

## E.5 From Bernstein reduction to kernels inducing contraction

Theorem $\mathbf { E . 4 }$ (Contraction and smallest contraction factor across dimensions reduce to a $\mathrm { L e v y - K h i n t - }$ chine criterion). Adopt the setting of Theorem E.2. For each $d \geq 1$ and $q \in \mathbb { R } ^ { d }$ , consider the affine map

$$
f ( x ) = q + \lambda x , \qquad x \in \mathbb { R } ^ { d } ,
$$

together with the radial kernel

$$
k ( u , v ) = g ( \| u - v \| ) , \qquad u , v \in \mathbb R ^ { d } .
$$

Let $h ( r ) = g ( 0 ) - g ( r )$ , and let $\phi$ be the Bernstein function with $h ( r ) = \phi ( r ^ { 2 } )$ . Fix $\lambda \in ( 0 , 1 )$ $\tau \in ( 0 , 1 )$ , and define the difference profile ψ<sub>τ</sub> and the kernel $\Gamma _ { \tau }$ built from it by

$$
\psi _ { \tau } ( t ) = \tau \phi ( t ) - \phi ( \lambda ^ { 2 } t ) , \qquad \Gamma _ { \tau } ( x , y ) = \psi _ { \tau } ( \| x - y \| ^ { 2 } ) .
$$

The contraction inequality   
MMD<sup>2</sup>(f#ν, f#µ) ≤ τ MMD<sup>2</sup>(ν, µ)   
holds for every $d \geq 1$ and every $\mu , \nu \in \mathcal P ( \mathbb { R } ^ { d } )$ if and only if   
Γ<sub>τ</sub> is CND on $\mathbb { R } ^ { d }$ for every d $\geq 1 .$

Write the Lévy–Khintchine representation of $\phi$ as

$$
\phi ( t ) = b t + \int _ { ( 0 , \infty ) } ( 1 - e ^ { - s t } ) \Lambda ( d s ) , \qquad b \geq 0 .
$$

Assume Λ has no singular continuous part and decompose

$$
\Lambda ( d s ) = m ( s ) d s + \sum _ { j \geq 1 } c _ { j } \delta _ { s _ { j } } .
$$

Then Γ<sub>τ</sub> is CND on $\mathbb { R } ^ { d }$ for every $d \geq 1$ if and only if   
1. $b ( \tau - \lambda ^ { 2 } ) \geq 0 ,$   
2. $\tau \lambda ^ { 2 } m ( s ) \geq m ( s / \lambda ^ { 2 } )$ for almost every $s > 0 ,$ , and   
3. $\tau \Lambda ( \{ x \} ) \geq \Lambda ( \{ x / \lambda ^ { 2 } \} )$ for every $x > 0 .$   
Assume these conditions hold. If $\Lambda ( \{ s _ { 0 } \} ) > 0$ , then $\Lambda ( \{ \lambda ^ { 2 n } s _ { 0 } \} ) \ge \tau ^ { - n } \Lambda ( \{ s _ { 0 } \} ) > 0$   
for every $n \ge 0 . \mathrm { I f } \tau \le \lambda ^ { 2 }$ , then Λ has no atoms. $\mathrm { I f } \tau > \dot { \lambda } ^ { 2 }$ , the set of atoms of Λ is a   
union of disjoint tails of geometric chains, so Λ has either no atoms or infinitely many.

To summarize the contraction side, define

τ<sup>⋆</sup>(λ) := inf{τ > 0 : Γ<sub>τ</sub> is CND on $\mathbb { R } ^ { d }$ for every $d \geq 1 \}$   
= inf{τ > 0 : ψ<sub>τ</sub> is Bernstein}.   
Contraction across dimensions holds if and only $\mathrm { i f } \tau ^ { \star } ( \lambda ) < 1$   
Smallest contraction factor across dimensions: $\tau ^ { \star } ( \lambda )$ for $\mathbf { M } \mathbf { M } \mathbf { D } _ { k } ^ { 2 } , \mathbf { \Lambda } \sqrt { \tau ^ { \star } ( \lambda ) }$ for $\mathbf { M M D } _ { k } .$

Drift cases. The drift is $b = \mathrm { l i m } _ { t  \infty } \phi ( t ) / t$ . If $b > 0$ , then necessarily $\tau \geq \lambda ^ { 2 }$ . If $b = 0 ,$ , no inequality on τ follows from the drift term.

The factor $\tau ^ { \star } ( \lambda )$ asks for contraction in every dimension at once. For $\lambda = \gamma$ , a factor that gives contraction in every dimension gives it in particular for returns in $\mathbb { R } ^ { d }$ , so by Theorem 3.1 the smallest contraction factor of the main text satisfies $\tau _ { k } ^ { \star } ( \mathcal { T } ^ { \pi } ) \leq \tau ^ { \star } ( \gamma )$

Proof.

Step 1: Reduction to a CND test. The first boxed equivalence is exactly Theorem E.2.

Step 2: Lévy–Khintchine drift and the measure domination condition. Write the Lévy– Khintchine representation

$$
\phi ( t ) = b t + \int _ { ( 0 , \infty ) } ( 1 - e ^ { - s t } ) \Lambda ( d s ) , \qquad b \geq 0 ,
$$

with Λ a nonnegative Borel measure. By Theorem E.3,

$$
\begin{array} { c } { \Gamma _ { \tau } \mathrm { i s } \mathrm { C N D ~ o n ~ } \mathbb { R } ^ { d } \mathrm { ~ f o r ~ e v e r y ~ } d \geq 1 } \\ { \iff } \\ { b ( \tau - \lambda ^ { 2 } ) \geq 0 \mathrm { ~ a n d ~ } \tau \Lambda ( B ) \geq \Lambda ( B / \lambda ^ { 2 } ) \ \forall B \subset ( 0 , \infty ) \mathrm { ~ B o r e l } . } \end{array}
$$

Step 3: The inequalities on the density and on the atoms. By assumption, Λ has no singular continuous part and decomposes as

$$
\Lambda ( d s ) = m ( s ) d s + \sum _ { j \geq 1 } c _ { j } \delta _ { s _ { j } } .
$$

Applying Lemma E.2 with $\rho = \lambda ^ { 2 }$ rewrites the measure domination condition as

$$
\tau \Lambda ( B ) \geq \Lambda ( B / \lambda ^ { 2 } ) \ \forall \mathrm { B o r e l } \ B \quad \Longleftrightarrow \quad \left\{ \tau \lambda ^ { 2 } m ( s ) \geq m ( s / \lambda ^ { 2 } ) \qquad \mathrm { f o r ~ a l m o s t ~ e v e r y ~ } s > 0 , \right.
$$

which, together with $b ( \tau - \lambda ^ { 2 } ) \geq 0 .$ , gives the three conditions of the statement.

Step 4: Consequences for the atoms. Under condition 3 of the statement, if $\Lambda ( \{ s _ { 0 } \} ) > 0$ , then

$$
\Lambda ( \{ \lambda ^ { 2 n } s _ { 0 } \} ) \geq \tau ^ { - n } \Lambda ( \{ s _ { 0 } \} ) \qquad \mathrm { f o r ~ a l l } \ n \geq 0 .
$$

Thus, once an atom appears at a point, every smaller point of the same geometric chain must also carry an atom.

Moreover, Proposition E.4 with $\rho = \lambda ^ { 2 }$ shows that if $\tau \leq \lambda ^ { 2 }$ , then

$$
\Lambda ( \{ x \} ) = 0 \qquad \forall x > 0 .
$$

If instead $\tau > \lambda ^ { 2 }$ , then the set of atoms of Λ must be a union of disjoint tails of geometric chains, of the form

$$
\{ \lambda ^ { 2 n } r : \ n \geq n _ { r } \} , \qquad r \in ( \lambda ^ { 2 } , 1 ] ,
$$

and in particular Λ has either no atoms or infinitely many. Conversely, such tails can indeed occur, by Proposition E.4.

Step 5: Smallest contraction factor. By Theorem E.2,

$$
\Gamma _ { \tau } { \mathrm { ~ i s ~ C N D ~ o n ~ } } \mathbb { R } ^ { d } { \mathrm { ~ f o r ~ e v e r y ~ } } d \geq 1 \quad \Longleftrightarrow \quad \psi _ { \tau } { \mathrm { ~ i s ~ B e r n s t e i n } } .
$$

Therefore

$$
\tau ^ { \star } ( \lambda ) : = \operatorname* { i n f } \{ \tau > 0 : \Gamma _ { \tau } { \mathrm { ~ i s ~ C N D ~ o n ~ } } \mathbb { E } ^ { d } { \mathrm { ~ f o r ~ e v e r y ~ } } d \geq 1 \} = \operatorname* { i n f } \{ \tau > 0 : \psi _ { \tau } { \mathrm { ~ i s ~ B e r n s t e i n } } \} .
$$

Combining this with Step 1, contraction across dimensions holds if and only $\mathrm { i f } \tau ^ { \star } ( \lambda ) < 1$ , and then $\tau ^ { \star } ( \lambda )$ is the smallest contraction factor across dimensions in $ { \mathbf { M M D } } _ { k } ^ { 2 }$ , and $\sqrt { \tau ^ { \star } ( \lambda ) }$ in $\mathbf { M M D } _ { k }$

Step 6: Drift cases. The drift condition $b ( \tau - \lambda ^ { 2 } ) \geq 0$ of Step 2 gives $\tau \geq \lambda ^ { 2 } \operatorname { i f } b > 0$ , and no restriction on $\tau { \mathrm { i f } } b = 0$

## E.6 Contraction from complete Bernstein functions

Theorem E.5 (For complete Bernstein functions, the smallest contraction factor across dimensions reduces to a single supremum). Adopt the setting of Theorem E.4. For each $d \geq 1$ and $q \in \mathbb { R } ^ { d }$ consider the affine map

$$
f ( x ) = q + \lambda x , \qquad x \in \mathbb { R } ^ { d } .
$$

Assume moreover that $\phi$ is a complete Bernstein function with $\phi ( 0 ) = 0$ . Then $\phi$ admits a $\mathrm { L e v y - }$ Khintchine representation

$$
\phi ( t ) = b t + \int _ { 0 } ^ { \infty } \left( 1 - e ^ { - s t } \right) m ( s ) d s , \qquad t \geq 0 ,
$$

with $b \geq 0$ and a Lévy density $m : ( 0 , \infty ) \to [ 0 , \infty )$ such that, by Definition E.2,

$m \in C ^ { \infty } ( ( 0 , \infty ) )$ and m is completely monotone.

Fix $\lambda \in ( 0 , 1 )$ and $\tau > 0 .$ , and define the difference profile

$$
\psi _ { \tau } ( t ) = \tau \phi ( t ) - \phi ( \lambda ^ { 2 } t ) , \qquad t \geq 0 .
$$

Then $\psi _ { \tau }$ is a Bernstein function if and only if

$$
b ( \tau - \lambda ^ { 2 } ) \geq 0 \qquad \mathrm { a n d } \qquad \tau \lambda ^ { 2 } m ( s ) \geq m \Big ( \frac { s } { \lambda ^ { 2 } } \Big ) \mathrm { f o r e v e r y } s > 0 .
$$

If $m \equiv 0 ,$ , then

$$
\tau ^ { \star } ( \lambda ) = \lambda ^ { 2 } { \bf 1 } _ { \{ b > 0 \} } .
$$

If $m \not \equiv 0$ , then the smallest τ for which $\psi _ { \tau }$ is a Bernstein function is

$$
\boxed  \tau ^ { \star } ( \lambda ) = \operatorname* { m a x } \left\{ \lambda ^ { 2 } { \bf 1 } _ { \{ b > 0 \} } , \ \operatorname* { s u p } _ { s > 0 } \frac { m \left( \frac { s } { \lambda ^ { 2 } } \right) } { \lambda ^ { 2 } m ( s ) } \right\} .
$$

In particular, for every $d \geq 1$ and all $\mu , \nu \in \mathcal P ( \mathbb { R } ^ { d } )$

$$
\mathbf { M M D } _ { k } ^ { 2 } ( f \# \nu , f \# \mu ) \leq \tau \mathbf { M M D } _ { k } ^ { 2 } ( \nu , \mu )
$$

holds if and only $\mathrm { i f } \tau \geq \tau ^ { \star } ( \lambda )$ . Consequently, contraction across dimensions holds if and only if

$$
\tau ^ { \star } ( \lambda ) < 1 ,
$$

in which case $\tau ^ { \star } ( \lambda )$ is the smallest contraction factor across dimensions in $ { \mathbf { M M D } } _ { k } ^ { 2 }$

Proof. By Theorem $\mathrm { E } . 4 , \psi _ { \tau }$ is Bernstein if and only if

$$
b ( \tau - \lambda ^ { 2 } ) \geq 0 \qquad \mathrm { a n d } \qquad \tau \lambda ^ { 2 } m ( s ) \geq m ( s / \lambda ^ { 2 } ) \quad \mathrm { f o r ~ a l m o s t ~ e v e r y ~ } s > 0 .
$$

By Definition E.2, one has $m \in C ^ { \infty } ( ( 0 , \infty ) )$ . Hence both sides of the inequality are continuous in $s ,$ so the inequality, which holds almost everywhere, extends to every $s > 0$ . This proves the first claim.

$\mathrm { I f } m \equiv 0 .$ , only the drift condition remains, so

$$
\tau ^ { \star } ( \lambda ) = \lambda ^ { 2 } { \bf 1 } _ { \{ b > 0 \} } .
$$

Assume now that $m \not \equiv 0 .$ . Since m is completely monotone and nonzero, it is strictly positive on $( 0 , \infty )$ . Therefore $\lambda ^ { 2 } { \dot { m } } ( s ) > 0$ for every $s > 0 ,$ , so we may divide both sides by $\lambda ^ { 2 } m ( \bar { s } )$ . Thus

$$
\tau \lambda ^ { 2 } m ( s ) \geq m ( s / \lambda ^ { 2 } ) \qquad \forall s > 0
$$

is equivalent to

$$
\tau \geq \frac { m ( s / \lambda ^ { 2 } ) } { \lambda ^ { 2 } m ( s ) } \qquad \forall s > 0 .
$$

Combining this with $b ( \tau - \lambda ^ { 2 } ) \geq 0 .$ we obtain that $\psi _ { \tau }$ is Bernstein if and only if

$$
\tau \geq \lambda ^ { 2 } \mathbf { 1 } _ { \{ b > 0 \} } \qquad \mathrm { a n d } \qquad \tau \geq \frac { m ( s / \lambda ^ { 2 } ) } { \lambda ^ { 2 } m ( s ) } \quad \forall s > 0 .
$$

Therefore

$$
\tau ^ { \star } ( \lambda ) = \operatorname* { m a x } \\biggl \{ \lambda ^ { 2 } \mathbf { 1 } _ { \{ b > 0 \} } , \ : \ : \operatorname* { s u p } _ { s > 0 } \frac { m ( s / \lambda ^ { 2 } ) } { \lambda ^ { 2 } m ( s ) } \biggr \} .
$$

The final equivalence with the $ { \mathbf { M M D } } _ { k } ^ { 2 }$ contraction statement follows directly from Theorem E.4.

Corollary E.5.1 (Limit at $0 ^ { + }$ and consequences for the metric property and contraction). Adopt the setting of Theorem E.5. Assume there exists $\tau < 1$ such that

$$
\psi _ { \tau } ( t ) = \tau \phi ( t ) - \phi ( \lambda ^ { 2 } t )
$$

is a Bernstein function. Since m is completely monotone, it is nonincreasing, so its limit at $0 ^ { + }$

$$
m ( 0 ^ { + } ) : = \operatorname* { l i m } _ { s  0 ^ { + } } m ( s ) \in [ 0 , \infty ] ,
$$

exists. Then $m ( 0 ^ { + } )$ is either 0 or ∞.

1. If $m ( 0 ^ { + } ) = 0$ , then $m \equiv 0 ,$ , so $\phi ( t ) = b t$ and $k ( x , y ) = g ( 0 ) - b \| x - y \| ^ { 2 }$ . In this case, $\mathbf { M M D } _ { k }$ is not a metric by Theorem I.1. By Theorem E.5, the smallest contraction factor across dimensions is $\tau ^ { \star } ( \lambda ) = \lambda ^ { 2 }$ if $b > 0$ and $\tau ^ { \star } ( \lambda ) = 0 \mathrm { i f } b = 0$

2. If $\mathbf { M M D } _ { k }$ is a metric, then $m ( 0 ^ { + } ) = \infty$

Proof.

Step 1: Existence of $m ( 0 ^ { + } )$ . Since m is completely monotone, it is nonincreasing on $( 0 , \infty )$ Hence its boundary value at $0 ^ { + }$ is given by the supremum, and we may write

$$
m ( 0 ^ { + } ) = \operatorname* { s u p } _ { s > 0 } m ( s ) \ \in \ [ 0 , \infty ] .
$$

Step 2: A strict contraction excludes $m ( 0 ^ { + } ) \in ( 0 , \infty )$ . Assume for contradiction that $m ( 0 ^ { + } ) \in$ $( 0 , \infty )$ . By Theorem E.5, the assumption that $\psi _ { \tau }$ is a Bernstein function implies that

$$
\tau \lambda ^ { 2 } m ( s ) \geq m \Big ( \frac { s } { \lambda ^ { 2 } } \Big ) \qquad \mathrm { f o r ~ e v e r y ~ } s > 0 .
$$

Letting $s \to 0 ^ { + }$ and using $m ( s ) \to m ( 0 ^ { + } )$ and $m ( s / \lambda ^ { 2 } )  m ( 0 ^ { + } )$ gives

$$
\tau \lambda ^ { 2 } m ( 0 ^ { + } ) \geq m ( 0 ^ { + } ) .
$$

Since $m ( 0 ^ { + } ) > 0$ , we may divide by $m ( 0 ^ { + } )$ to obtain $\tau \lambda ^ { 2 } \geq 1$ . This contradicts $\tau < 1$ and $\lambda \in ( 0 , 1 )$ , which together imply $\tau \lambda ^ { 2 } < 1$ . Hence $m ( 0 ^ { + } ) \not \in ( 0 , \infty )$ , and therefore $m ( 0 ^ { + } ) \in \{ 0 , \infty \}$

Step 3: If $m ( 0 ^ { + } ) = 0$ then $m \equiv 0 ,$ , MMD is not a metric, and contraction depends only on b. If $m ( \bar { 0 } ^ { + } ) = 0$ and m is nonincreasing and nonnegative, then for all $s > 0$

$$
0 \ \leq \ m ( s ) \ \leq \ m ( 0 ^ { + } ) = 0 ,
$$

hence $m \equiv 0$ and $\phi ( t ) = b t . \ \mathbf { B } \mathbf { y }$ Theorem $\mathrm { I . l } , \mathbf { M M D } _ { k }$ is a metric if and only if $\Lambda \not \equiv 0 .$ , equivalently $m \not \equiv 0$ . Thus $\mathbf { M M D } _ { k }$ is not a metric in this case. Finally, when $m \equiv 0$ , the supremum term in Theorem E.5 vanishes and $\tau ^ { \star } ( \lambda ) = \lambda ^ { 2 } { \bf 1 } _ { \{ b > 0 \} }$

Step 4: If $\mathbf { M M D } _ { k }$ is a metric then $m ( 0 ^ { + } ) = \infty .$ . If $\mathbf { M M D } _ { k }$ is a metric, then Theorem I.1 implies $\Lambda \neq 0 ,$ , hence $m \not \equiv 0$ . Step 3 shows that $m ( 0 ^ { + } ) = 0$ would force $m \equiv 0$ , a contradiction. Therefore $\dot { m } ( 0 ^ { + } ) = \infty .$ □

## E.7 Chaining as a general mechanism

Preamble. Section E.4, and in particular Lemma E.4 and Proposition E.4, introduced a geometric chaining mechanism. When Λ has atoms and satisfies the measure domination condition, its atoms cannot lie anywhere. By Proposition E.4, the set of atoms is a union of disjoint tails of geometric chains.

In this subsection, we show that this mechanism is more general. The geometric chain of atoms is the special case of a broader construction in which the base Bernstein function is $t \mapsto 1 - e ^ { - s t }$

As a first step, we isolate the basic building block already present in Section E.4, namely a single geometric chain of atoms. This is the simplest instance of the general construction that follows.

Proposition E.5 (Single geometric chain of atoms). Fix $\lambda \in ( 0 , 1 )$ and set $\rho : = \lambda ^ { 2 } \in ( 0 , 1 )$ . Let $\tau \in ( 0 , 1 )$ , let $s _ { 0 } > 0$ , and let $c > 0$ . Define the measure

$$
\Lambda ( d s ) : = \sum _ { n = 0 } ^ { \infty } c \tau ^ { - n } \delta _ { \rho ^ { n } s _ { 0 } } ( d s ) .
$$

If $\tau > \rho ,$ , then

$$
\int _ { ( 0 , \infty ) } \operatorname* { m i n } \{ 1 , s \} \Lambda ( d s ) \leq s _ { 0 } c \sum _ { n = 0 } ^ { \infty } \Bigl ( \frac { \rho } { \tau } \Bigr ) ^ { n } < \infty ,
$$

so $\Lambda$ is a Lévy measure. In addition, Lemma E.4 shows that this single geometric chain satisfies the measure domination condition. Its Bernstein profile is therefore

$$
\phi _ { \mathrm { c h a i n } } ( t ) = b t + \sum _ { n = 0 } ^ { \infty } c \tau ^ { - n } \big ( 1 - e ^ { - \rho ^ { n } s _ { 0 } t } \big ) , \qquad t \ge 0 , \quad b \ge 0 .
$$

Since $\tau > \rho = \lambda ^ { 2 }$ , the drift condition in Theorem E.4 is also satisfied. Hence, under the radial kernel generated by $\phi _ { \mathrm { c h a i n } }$ , the affine map $x \mapsto q + \lambda x$ is a τ-contraction in $\mathbf { M M D } ^ { 2 }$

See Figure 5 for a visualization of the atoms and the resulting ϕ<sub>chain</sub>.

![](images/88f144d076d4b152173f7b7af8776adf3f7e14809696596a2a34aa6059449e24.jpg)

![](images/1d441ec5214514d83c43583d5bdaf69e1c3fa10b3dfa51483cc1bd9bef241cb8.jpg)  
Figure 5: Single geometric chain of atoms. Left, atoms at $s _ { n } = \lambda ^ { 2 n } s _ { 0 }$ with weights $a _ { n } = c \tau ^ { - n }$ . The chain is geometric in both location and weight, since $s _ { n + 1 } / s _ { n } = \lambda ^ { 2 }$ and $a _ { n + 1 } / a _ { n } = \tau ^ { - 1 }$ are constant. The left panel is truncated after a finite number of atoms for readability. Right, the corresponding Bernstein function $\begin{array} { r } { \phi _ { \mathrm { c h a i n } } ( t ) = b t + \sum _ { n > 0 } c \tau ^ { - n } \big ( 1 - e ^ { - s _ { n } t } \big ) } \end{array}$ , evaluated with truncation after 250 terms.

Proposition E.6 (Geometric chaining gives identities for the $\operatorname { L e v y }$ measure and the difference profile). Fix $\rho \in ( 0 , 1 )$ and $\tau \in ( 0 , 1 )$ . Let $\phi _ { 0 }$ be a base Bernstein function with $\phi _ { 0 } ( 0 ) = 0$ , and write its Lévy–Khintchine representation as

$$
\phi _ { 0 } ( t ) = b _ { 0 } t + \int _ { ( 0 , \infty ) } ( 1 - e ^ { - s t } ) \Lambda _ { 0 } ( d s ) , \qquad t \geq 0 ,
$$

where $b _ { 0 } \geq 0$ and $\Lambda _ { 0 }$ is a Lévy measure on $( 0 , \infty )$

Consider the chained profile

$$
\phi ( t ) : = \sum _ { n = 0 } ^ { \infty } \tau ^ { - n } \phi _ { 0 } ( \rho ^ { n } t ) , \qquad t \geq 0 .
$$

For each $n \geq 0 ,$ , define the rescaled measure

$$
\Lambda _ { 0 } ^ { ( n ) } ( B ) : = \Lambda _ { 0 } ( B / \rho ^ { n } ) , \qquad B / \rho ^ { n } : = \{ s > 0 : \rho ^ { n } s \in B \} .
$$

Define

$$
b : = \sum _ { n = 0 } ^ { \infty } \tau ^ { - n } \rho ^ { n } b _ { 0 } , \qquad \Lambda ( B ) : = \sum _ { n = 0 } ^ { \infty } \tau ^ { - n } \Lambda _ { 0 } ( B / \rho ^ { n } ) ,
$$

and assume that

$$
b < \infty \qquad \mathrm { a n d } \qquad \int _ { ( 0 , \infty ) } \operatorname * { m i n } \{ 1 , s \} \Lambda ( d s ) < \infty .
$$

Then $\phi$ is a Bernstein function and admits the Lévy–Khintchine representation

$$
\phi ( t ) = b t + \int _ { ( 0 , \infty ) } ( 1 - e ^ { - s t } ) \Lambda ( d s ) , \qquad t \geq 0 .
$$

Moreover, if we define

$$
\Lambda _ { \rho } ( B ) : = \Lambda ( B / \rho ) ,
$$

then

$$
\boxed { \tau \Lambda = \Lambda _ { \rho } + \tau \Lambda _ { 0 } }
$$

as nonnegative Borel measures on $( 0 , \infty )$ , and

$$
\boxed { \tau \phi ( t ) - \phi ( \rho t ) = \tau \phi _ { 0 } ( t ) }
$$

for all $t \geq 0$

Proof.

Lévy–Khintchine representation of $\phi .$ By definition,

$$
\phi ( t ) = \sum _ { n = 0 } ^ { \infty } \tau ^ { - n } \phi _ { 0 } ( \rho ^ { n } t ) .
$$

Using the Lévy–Khintchine representation of $\phi _ { 0 } ,$ this becomes

$$
\phi ( t ) = \sum _ { n = 0 } ^ { \infty } \tau ^ { - n } \left[ \rho ^ { n } b _ { 0 } t + \int _ { ( 0 , \infty ) } ( 1 - e ^ { - u \rho ^ { n } t } ) \Lambda _ { 0 } ( d u ) \right] .
$$

Since all terms are nonnegative, Tonelli’s theorem yields

$$
\phi ( t ) = \left( \sum _ { n = 0 } ^ { \infty } \tau ^ { - n } \rho ^ { n } b _ { 0 } \right) t + \int _ { ( 0 , \infty ) } \sum _ { n = 0 } ^ { \infty } \tau ^ { - n } ( 1 - e ^ { - u \rho ^ { n } t } ) \Lambda _ { 0 } ( d u ) .
$$

By the definition of the rescaled measures $\Lambda _ { 0 } ^ { ( n ) } ( B ) = \Lambda _ { 0 } ( B / \rho ^ { n } )$ , this is exactly

$$
\phi ( t ) = b t + \int _ { ( 0 , \infty ) } ( 1 - e ^ { - s t } ) \Lambda ( d s ) .
$$

Exact identities for Λ and $\phi .$ By definition,

$$
\Lambda _ { \rho } ( B ) = \Lambda ( B / \rho ) = \sum _ { n = 0 } ^ { \infty } \tau ^ { - n } \Lambda _ { 0 } ( B / \rho ^ { n + 1 } ) .
$$

Reindexing with $j = n + 1$ gives

$$
\Lambda _ { \rho } ( B ) = \sum _ { j = 1 } ^ { \infty } \tau ^ { - ( j - 1 ) } \Lambda _ { 0 } ( B / \rho ^ { j } ) = \tau \sum _ { j = 1 } ^ { \infty } \tau ^ { - j } \Lambda _ { 0 } ( B / \rho ^ { j } ) .
$$

On the other hand,

$$
\tau \Lambda ( B ) = \tau \sum _ { j = 0 } ^ { \infty } \tau ^ { - j } \Lambda _ { 0 } ( B / \rho ^ { j } ) = \tau \Lambda _ { 0 } ( B ) + \tau \sum _ { j = 1 } ^ { \infty } \tau ^ { - j } \Lambda _ { 0 } ( B / \rho ^ { j } ) .
$$

Comparing the two expressions, we obtain

$$
\tau \Lambda ( B ) = \Lambda _ { \rho } ( B ) + \tau \Lambda _ { 0 } ( B ) \qquad \mathrm { f o r e v e r y ~ B o r e l ~ s e t } \ : B \subset ( 0 , \infty ) .
$$

That is,

$$
\tau \Lambda = \Lambda _ { \rho } + \tau \Lambda _ { 0 } .
$$

Recall that

$$
\phi ( t ) = \sum _ { n = 0 } ^ { \infty } \tau ^ { - n } \phi _ { 0 } ( \rho ^ { n } t ) .
$$

Therefore

$$
\tau \phi ( t ) = \tau \sum _ { n = 0 } ^ { \infty } \tau ^ { - n } \phi _ { 0 } ( \rho ^ { n } t ) = \tau \phi _ { 0 } ( t ) + \tau \sum _ { j = 1 } ^ { \infty } \tau ^ { - j } \phi _ { 0 } ( \rho ^ { j } t ) .
$$

On the other hand,

$$
\phi ( \rho t ) = \sum _ { n = 0 } ^ { \infty } \tau ^ { - n } \phi _ { 0 } ( \rho ^ { n + 1 } t ) .
$$

Reindexing with $j = n + 1$ gives

$$
\phi ( \rho t ) = \sum _ { j = 1 } ^ { \infty } \tau ^ { - ( j - 1 ) } \phi _ { 0 } ( \rho ^ { j } t ) = \tau \sum _ { j = 1 } ^ { \infty } \tau ^ { - j } \phi _ { 0 } ( \rho ^ { j } t ) .
$$

Comparing the two expressions, we obtain

$$
\tau \phi ( t ) - \phi ( \rho t ) = \tau \phi _ { 0 } ( t ) ,
$$

which is exactly the claimed identity for the difference profile.

Proposition E.7 (When geometric chaining gives a Bernstein function). Fix $\rho \in ( 0 , 1 )$ and $\tau \in ( 0 , 1 )$ Let $\phi _ { 0 }$ be a base Bernstein function with $\phi _ { 0 } ( 0 ) = 0$ , and write its Lévy–Khintchine representation as

$$
\phi _ { 0 } ( t ) = b _ { 0 } t + \int _ { ( 0 , \infty ) } ( 1 - e ^ { - s t } ) \Lambda _ { 0 } ( d s ) , \qquad t \geq 0 ,
$$

where $b _ { 0 } \geq 0$ and $\Lambda _ { 0 }$ is a Lévy measure on $( 0 , \infty )$ . Consider the chained profile

$$
\phi ( t ) : = \sum _ { n = 0 } ^ { \infty } \tau ^ { - n } \phi _ { 0 } ( \rho ^ { n } t ) , \qquad t \geq 0 .
$$

For each $n \geq 0$ , define the rescaled measure

$$
\Lambda _ { 0 } ^ { ( n ) } ( A ) : = \Lambda _ { 0 } ( A / \rho ^ { n } ) , \qquad A / \rho ^ { n } : = \{ s > 0 : \rho ^ { n } s \in A \} ,
$$

and define, as in Proposition E.6,

$$
b : = \sum _ { n = 0 } ^ { \infty } \tau ^ { - n } \rho ^ { n } b _ { 0 } , \qquad \Lambda ( A ) : = \sum _ { n = 0 } ^ { \infty } \tau ^ { - n } \Lambda _ { 0 } ( A / \rho ^ { n } ) .
$$

Introduce also

$$
W _ { \rho , \tau } ( s ) : = \sum _ { n = 0 } ^ { \infty } \tau ^ { - n } \operatorname * { m i n } \{ 1 , \rho ^ { n } s \} , \qquad s > 0 .
$$

Then the conditions

$$
b < \infty \quad { \mathrm { a n d } } \quad \int _ { ( 0 , \infty ) } \operatorname* { m i n } \{ 1 , s \} \Lambda ( d s ) < \infty
$$

hold if and only if

$$
\sum _ { n = 0 } ^ { \infty } \tau ^ { - n } \rho ^ { n } b _ { 0 } < \infty \quad \mathrm { a n d } \quad \int _ { ( 0 , \infty ) } W _ { \rho , \tau } ( s ) \Lambda _ { 0 } ( d s ) < \infty .
$$

Whenever these equivalent conditions hold, the chained profile is a Bernstein function and admits the representation

$$
\phi ( t ) = b t + \int _ { ( 0 , \infty ) } ( 1 - e ^ { - s t } ) \Lambda ( d s ) , \qquad t \geq 0 .
$$

In particular, if $\phi _ { 0 } \not \equiv 0$ , these conditions can hold only if $\tau > \rho .$

Proof.

Equivalence of the integrability conditions. By definition,

$$
b = \sum _ { n = 0 } ^ { \infty } \tau ^ { - n } \rho ^ { n } b _ { 0 } ,
$$

so the drift part is well defined if and only if

$$
\sum _ { n = 0 } ^ { \infty } \tau ^ { - n } \rho ^ { n } b _ { 0 } < \infty .
$$

For the measure part, Tonelli’s theorem applies because the measures $\tau ^ { - n } \Lambda _ { 0 } ^ { ( n ) }$ are nonnegative and the integrand min{1, u} is nonnegative. Therefore

$$
\begin{array} { r l r } & { } & { \displaystyle \int _ { ( 0 , \infty ) } \operatorname* { m i n } \{ 1 , u \} \Lambda ( d u ) = \int _ { ( 0 , \infty ) } \operatorname* { m i n } \{ 1 , u \} \sum _ { n = 0 } ^ { \infty } \tau ^ { - n } \Lambda _ { 0 } ^ { ( n ) } ( d u ) } \\ & { } & { \displaystyle \quad \quad = \sum _ { n = 0 } ^ { \infty } \tau ^ { - n } \int _ { ( 0 , \infty ) } \operatorname* { m i n } \{ 1 , u \} \Lambda _ { 0 } ^ { ( n ) } ( d u ) . } \end{array}
$$

By the definition of $\Lambda _ { 0 } ^ { ( n ) }$

$$
\int _ { ( 0 , \infty ) } \operatorname* { m i n } \{ 1 , u \} \Lambda _ { 0 } ^ { ( n ) } ( d u ) = \int _ { ( 0 , \infty ) } \operatorname* { m i n } \{ 1 , \rho ^ { n } s \} \Lambda _ { 0 } ( d s ) .
$$

Hence

$$
\begin{array} { l } { \displaystyle \int _ { ( 0 , \infty ) } \operatorname* { m i n } \{ 1 , u \} \Lambda ( d u ) = \displaystyle \sum _ { n = 0 } ^ { \infty } \tau ^ { - n } \int _ { ( 0 , \infty ) } \operatorname* { m i n } \{ 1 , \rho ^ { n } s \} \Lambda _ { 0 } ( d s ) } \\ { \displaystyle \qquad = \int _ { ( 0 , \infty ) } \sum _ { n = 0 } ^ { \infty } \tau ^ { - n } \operatorname* { m i n } \{ 1 , \rho ^ { n } s \} \Lambda _ { 0 } ( d s ) } \\ { \displaystyle \qquad = \int _ { ( 0 , \infty ) } W _ { \rho , \tau } ( s ) \Lambda _ { 0 } ( d s ) . } \end{array}
$$

Thus Λ satisfies the Lévy integrability condition if and only if

$$
\int _ { ( 0 , \infty ) } W _ { \rho , \tau } ( s ) \Lambda _ { 0 } ( d s ) < \infty .
$$

This proves the equivalence.

Under these equivalent conditions, Proposition E.6 applies to

$$
\phi ( t ) = \sum _ { n = 0 } ^ { \infty } \tau ^ { - n } \phi _ { 0 } ( \rho ^ { n } t )
$$

and yields

$$
\phi ( t ) = b t + \int _ { ( 0 , \infty ) } ( 1 - e ^ { - s t } ) \Lambda ( d s ) .
$$

In particular, ϕ is then a Bernstein function.

The regime $\tau > \rho$ is unavoidable. Assume $\phi _ { 0 } \not \equiv 0$ . If $b _ { 0 } > 0$ , then

$$
b = b _ { 0 } \sum _ { n = 0 } ^ { \infty } \left( { \frac { \rho } { \tau } } \right) ^ { n } ,
$$

and therefore $b < \infty$ implies $\rho / \tau < 1$ , that is, $\tau > \rho .$

Assume now $b _ { 0 } ~ = ~ 0$ . Since $\phi _ { 0 } ~ \not \equiv ~ 0$ , the measure $\Lambda _ { 0 }$ is nonzero, so there exists an interval $[ u , v ] \subset ( 0 , \infty )$ such that

$$
\int _ { [ u , v ] } s \Lambda _ { 0 } ( d s ) > 0 .
$$

Choose $K$ such that $\rho ^ { n } v \leq 1$ for every $n \geq K$ . Then, for every $n \geq K$ and every $s \in [ u , v ]$ , one has $\rho ^ { n } s \leq \rho ^ { n } v \leq 1$ , hence mi $\quad _ { 1 } \{ 1 , \rho ^ { n } s \} \stackrel { \cdot } { = } \rho ^ { n } s$ . Therefore,

$$
\begin{array} { l l } { \displaystyle \int _ { ( 0 , \infty ) } \operatorname* { m i n } \{ 1 , \rho ^ { n } s \} \Lambda _ { 0 } ( d s ) \geq \int _ { [ u , v ] } \operatorname* { m i n } \{ 1 , \rho ^ { n } s \} \Lambda _ { 0 } ( d s ) } \\ { \displaystyle \qquad = \int _ { [ u , v ] } \rho ^ { n } s \Lambda _ { 0 } ( d s ) } \\ { \displaystyle \qquad = \rho ^ { n } \int _ { [ u , v ] } s \Lambda _ { 0 } ( d s ) . } \end{array}
$$

It follows, by Tonelli’s theorem as in the preceding part, that

$$
\begin{array} { r l } { \displaystyle \int _ { ( 0 , \infty ) } W _ { \rho , \tau } ( s ) \mathrm { A } _ { 0 } ( d s ) = \int _ { ( 0 , \infty ) } \displaystyle \sum _ { n = 0 } ^ { \infty } \tau ^ { - n } \operatorname* { m i n } \{ 1 , \rho ^ { n } s \} \mathrm { A } _ { 0 } ( d s ) } & { } \\ & { = \displaystyle \sum _ { n = 0 } ^ { \infty } \tau ^ { - n } \int _ { ( 0 , \infty ) } \operatorname* { m i n } \{ 1 , \rho ^ { n } s \} \mathrm { A } _ { 0 } ( d s ) } \\ & { \geq \displaystyle \sum _ { n = K } ^ { \infty } \tau ^ { - n } \int _ { ( 0 , \infty ) } \operatorname* { m i n } \{ 1 , \rho ^ { n } s \} \mathrm { A } _ { 0 } ( d s ) } \\ & { \geq \displaystyle \sum _ { n = K } ^ { \infty } \tau ^ { - n } \rho ^ { n } \int _ { [ n , \infty ] } s \mathrm { A } _ { 0 } ( d s ) } \\ & { = \displaystyle \left( \int _ { [ n , \infty ] } s _ { 0 } ( d s ) \right) \sum _ { n = K } ^ { \infty } \left( \frac { \rho } { \tau } \right) ^ { n } . } \end{array}
$$

If $\tau \leq \rho ,$ , then $\rho / \tau \geq 1$ , so the series on the right diverges. Therefore

$$
\int _ { ( 0 , \infty ) } W _ { \rho , \tau } ( s ) \Lambda _ { 0 } ( d s ) = \infty ,
$$

which contradicts the integrability criterion proved above. Thus any such geometric chaining construction must satisfy $\tau > \rho .$ □

Theorem E.6 (Geometric chaining gives contraction with factor τ). Fix $\lambda \in ( 0 , 1 )$ and set $\rho : = \lambda ^ { 2 }$ Let $\tau \in \mathsf { \Gamma } ( 0 , 1 )$ , and let $\phi _ { 0 }$ be a base Bernstein function with $\phi _ { 0 } ( 0 ) = 0$ . Adopt the setting of Proposition E.6, and assume that $b < \infty$ and that Λ is a Lévy measure, as in Proposition E.7. Let

$$
\phi ( t ) : = b t + \int _ { ( 0 , \infty ) } ( 1 - e ^ { - s t } ) \Lambda ( d s ) , \qquad t \geq 0 .
$$

For each $d \geq 1$ and $q \in \mathbb { R } ^ { d }$ , consider the affine map

$$
f ( x ) = q + \lambda x , \qquad x \in \mathbb { R } ^ { d } ,
$$

and define the radial profile and kernel by

$$
h ( r ) : = \phi ( r ^ { 2 } ) , \qquad k ( x , y ) : = C - \phi ( \| x - y \| ^ { 2 } ) ,
$$

where $C \in \mathbb { R }$ is arbitrary. Define moreover the difference profile

$$
\psi _ { \tau } ( t ) : = \tau \phi ( t ) - \phi ( \rho t ) , \qquad t \geq 0 .
$$

Then

$$
\begin{array} { r l } & { \psi _ { \tau } ( t ) = \tau \phi _ { 0 } ( t ) \qquad \mathrm { f o r } \mathrm { e v e r y } t \geq 0 , } \\ & { \qquad \mathrm { h e n c e } \psi _ { \tau } \mathrm { i s } \mathrm { a B e r n s t e i n f u n c t i o n } , } \end{array}
$$

$$
\begin{array} { r l } & { \mathrm { a n d ~ t h e r e f o r e ~ } \mathbf { M } \mathbf { M } \mathbf { D } _ { k } ^ { 2 } ( f \# \nu , f \# \mu ) \leq \tau \mathbf { M } \mathbf { M } \mathbf { D } _ { k } ^ { 2 } ( \nu , \mu ) } \\ & { \qquad \quad \mathrm { f o r ~ e v e r y ~ } d \geq 1 \mathrm { ~ a n d ~ e v e r y ~ } \mu , \nu \in \mathcal { P } ( \mathbb { R } ^ { d } ) . } \end{array}
$$

In particular, under the kernel generated by the geometrically chained profile ϕ, every such affine map f is a τ -contraction in $\mathbf { M M D ^ { 2 } }$

Proof.

Step 1: Exact difference identity. By Proposition E.6,

$$
\tau \phi ( t ) - \phi ( \rho t ) = \tau \phi _ { 0 } ( t ) \qquad \mathrm { f o r e v e r y } t \geq 0 .
$$

That is,

$$
\psi _ { \tau } ( t ) = \tau \phi _ { 0 } ( t ) .
$$

Step 2: Bernstein property and contraction. Since $\phi _ { 0 }$ is a Bernstein function and the class of Bernstein functions is closed under multiplication by a nonnegative scalar [29, Corollary 3.7(i)], it follows from Step 1 that $\psi _ { \tau } = \tau \phi _ { 0 }$ is Bernstein. Applying Theorem E.2 with the present notation and $\rho = \lambda ^ { 2 }$ , we obtain

$$
\mathbf { M M D } _ { k } ^ { 2 } ( f \# \nu , f \# \mu ) \leq \tau \mathbf { M M D } _ { k } ^ { 2 } ( \nu , \mu )
$$

for every $d \geq 1$ and every $\mu , \nu \in \mathcal P ( \mathbb { R } ^ { d } )$ . This is exactly the claimed τ -contraction in $\mathbf { M M D ^ { 2 } . } \quad \varTheta$

## F Explaining known results

Preamble. In this section, we show that our framework explains three known results on MMD contraction for radial kernels. As in Appendix E, we study contraction for a fixed affine map $f ( x ) = q + \lambda x .$ , where λ plays the role of the discount γ. These results can then be extended to the setting of distributional RL and its Bellman operators using the results from Appendix C.

The first positive result was established in [25] for power-distance kernels. Proposition F.1 shows that these kernels are induced by a complete Bernstein function and satisfy the criterion of Theorem E.5. Hence they induce contraction under a fixed map. The corresponding negative result was also established in [25], where it was shown that the RBF kernel does not induce contraction in general. Proposition F.3 shows that the RBF kernel is likewise induced by a Bernstein function, and that its non-contraction can be explained through its Lévy–Khintchine representation together with the obstruction for saturating profiles from Proposition E.3. The second positive result was established in [19] and [12, Lemma 10] for the multiquadric kernel. Proposition F.2 extends this result to the family of generalized multiquadric kernels and shows that these kernels are also induced by a complete Bernstein function.

Proposition F.1 (Contraction under power-distance kernels [25]). Let

$$
k _ { \alpha } ( x , y ) \ = \ - y \| x - y \| ^ { \alpha } , \qquad 0 < \alpha \leq 2 ,
$$

and let

$$
f ( x ) = q + \lambda x , \qquad \lambda \in ( 0 , 1 ) .
$$

Then

$$
h ( r ) = r ^ { \alpha } = \phi ( r ^ { 2 } ) , \phi ( t ) = t ^ { \alpha / 2 } ,
$$

and the smallest contraction factor across dimensions in $\mathbf { M M D } _ { k _ { \alpha } } ^ { 2 }$ is $\tau ^ { \star } ( \lambda ) = \lambda ^ { \alpha }$ . For $\alpha = 2$ $\mathbf { M M D } _ { k _ { 2 } }$ is not a metric on probability measures.

Proof.

Lévy–Khintchine representation. By [29, Ch. 15.2, Identity 1],

• for 0 < α < 2 (so α/2 ∈ (0, 1)),

$$
\phi ( t ) = \frac { \alpha / 2 } { \Gamma ( 1 - \alpha / 2 ) } \int _ { 0 } ^ { \infty } ( 1 - e ^ { - s t } ) s ^ { - 1 - \alpha / 2 } d s ,
$$

so the drift is b = 0 and the Lévy measure is absolutely continuous,

$$
\Lambda ( d s ) = c _ { \alpha } s ^ { - 1 - \alpha / 2 } d s , \qquad c _ { \alpha } = \frac { \alpha / 2 } { \Gamma ( 1 - \alpha / 2 ) } .
$$

• for $\alpha = 2 , \phi ( t ) = t ,$ so the drift is b = 1 and $\Lambda = 0$

Bernstein test for the difference profile. By Theorem E.2, in the present radial setting, contraction in $\mathbf { M M D ^ { 2 } }$ reduces to checking whether the difference profile

$$
\psi _ { \tau } ( t ) : = \tau \phi ( t ) - \phi ( \lambda ^ { 2 } t )
$$

is Bernstein. Here

$$
\psi _ { \tau } ( t ) = ( \tau - \lambda ^ { \alpha } ) t ^ { \alpha / 2 } .
$$

Since $\phi ( t ) = t ^ { \alpha / 2 }$ is Bernstein and the class of Bernstein functions is closed under multiplication by a nonnegative scalar [29, Corollary 3.7(i)], it follows that $\psi _ { \tau }$ is Bernstein if and only $\mathrm { i f } \stackrel { - } { \tau } - \lambda ^ { \alpha } \geq \stackrel { - } { 0 }$ Therefore

$$
\begin{array} { r l } & { \mathbf { M M D } _ { k _ { \alpha } } ^ { 2 } ( f \# \nu , f \# \mu ) \leq \tau \mathbf { M M D } _ { k _ { \alpha } } ^ { 2 } ( \nu , \mu ) } \\ & { \quad \mathrm { ~ f o r ~ e v e r y ~ } d \geq 1 \mathrm { ~ a n d ~ e v e r y ~ } \mu , \nu \in \mathcal { P } ( \mathbb { R } ^ { d } ) } \\ & { \qquad \Longleftrightarrow } \\ & { \qquad \tau \geq \lambda ^ { \alpha } . } \end{array}
$$

Via Theorem E.5. For $0 < \alpha < 2$ , the function $\phi ( t ) = t ^ { \alpha / 2 }$ is a complete Bernstein function by [29, Ch. 15.2, Identity 1], with $b = 0$ and

$$
m ( s ) = c _ { \alpha } s ^ { - 1 - \alpha / 2 } .
$$

Hence Theorem E.5 applies directly and gives

$$
\tau ^ { \star } ( \lambda ) = \operatorname* { s u p } _ { s > 0 } \frac { m ( s / \lambda ^ { 2 } ) } { \lambda ^ { 2 } m ( s ) } = \operatorname* { s u p } _ { s > 0 } \frac { c _ { \alpha } ( s / \lambda ^ { 2 } ) ^ { - 1 - \alpha / 2 } } { \lambda ^ { 2 } c _ { \alpha } s ^ { - 1 - \alpha / 2 } } = \lambda ^ { \alpha } .
$$

For $\alpha = 2 .$ , one has $\phi ( t ) = t .$ , hence $b = 1$ and $m \equiv 0 ,$ , so Theorem E.5 gives $\tau ^ { \star } ( \lambda ) = \lambda ^ { 2 }$ . Thus in all cases $\tau ^ { \star } ( \lambda ) = \lambda ^ { \alpha }$ , which agrees with the Bernstein test above.

Contraction factor. The smallest contraction factor across dimensions in $\mathbf { M M D } _ { k _ { c } } ^ { 2 }$ is

$$
\tau ^ { \star } ( \lambda ) = \lambda ^ { \alpha } ( < 1 ) ,
$$

so

$$
 { \mathbf { M M D } } _ { k _ { \alpha } } ^ { 2 } ( f \# \nu , f \# \mu ) \le \lambda ^ { \alpha }  { \mathbf { M M D } } _ { k _ { \alpha } } ^ { 2 } ( \nu , \mu ) .
$$

Equivalently,

$$
\mathbf { M M D } _ { k _ { \alpha } } ( f \# \nu , f \# \mu ) \leq \lambda ^ { \alpha / 2 } \mathbf { M M D } _ { k _ { \alpha } } ( \nu , \mu ) .
$$

Special case $\alpha = 2 .$ . Since $k _ { 2 } ( x , y ) = - \| x - y \| ^ { 2 }$ can be written in the Bernstein form

$$
k _ { 2 } ( x , y ) = C - \phi ( \| x - y \| ^ { 2 } ) \quad { \mathrm { w i t h } } \quad C = 0 , \phi ( t ) = t ,
$$

Theorem I.1 applies and gives the criterion

$$
\mathbf { M } \mathbf { M } \mathbf { D } _ { k _ { 2 } } { \mathrm { ~ i s ~ a ~ m e t r i c } } \quad \iff \quad \Lambda \neq 0 .
$$

Here $\phi ( t ) = t$ is linear, so its Lévy measure satisfies $\Lambda \equiv 0$ . Hence $\mathbf { M M D } _ { k _ { 2 } }$ is not a metric on probability measures. □

Proposition F.2 (Contraction under generalized multiquadric kernels, generalizing [19] and [12, Lemma 10]). Fix $h > 0$ and $\beta \in ( 0 , \bar { 1 } ]$ . Consider

$$
k _ { \beta , h } ( x , y ) ~ = ~ - \bigl ( 1 + h ^ { 2 } \| x - y \| ^ { 2 } \bigr ) ^ { \beta } , \qquad \phi _ { \beta } ( t ) ~ = ~ \bigl ( 1 + h ^ { 2 } t \bigr ) ^ { \beta } - 1 .
$$

Assume $f ( x ) = q + \rangle$ x with $\lambda \in ( 0 , 1 )$ . Then the smallest contraction factor across dimensions in $\mathbf { M M D } _ { k _ { \beta , h } } ^ { 2 } \mathrm { ~ i s ~ } \tau ^ { \star } ( \lambda ) = \lambda ^ { 2 \beta }$

Proof. Define

$$
\psi _ { \tau } ( t ) = \tau \phi _ { \beta } ( t ) - \phi _ { \beta } ( \lambda ^ { 2 } t ) , \qquad \Gamma _ { \tau } ( x , y ) = \psi _ { \tau } ( \| x - y \| ^ { 2 } ) .
$$

Lévy–Khintchine representation. By [29, Ch. 15.2, Identity 2], for $0 < \beta < 1$

$$
\phi _ { \beta } ( t ) = \frac { \beta h ^ { 2 \beta } } { \Gamma ( 1 - \beta ) } \int _ { 0 } ^ { \infty } \left( 1 - e ^ { - s t } \right) e ^ { - s / h ^ { 2 } } s ^ { - \beta - 1 } d s ,
$$

so $b = 0$ and the Lévy measure is absolutely continuous with density

$$
m _ { \beta , h } ( s ) = \frac { \beta h ^ { 2 \beta } } { \Gamma ( 1 - \beta ) } e ^ { - s / h ^ { 2 } } s ^ { - \beta - 1 } , \qquad s > 0 .
$$

For $\beta = 1 , \phi _ { 1 } ( t ) = h ^ { 2 } t$ is linear, hence $b = h ^ { 2 }$ and $\Lambda \equiv 0$ . In particular, $\phi _ { \beta }$ is a complete Bernstein function.

Bernstein test for the difference profile. By Theorem E.2, in the present radial setting, contraction in $\mathbf { M M D ^ { 2 } }$ reduces to checking whether the difference profile

$$
\psi _ { \tau } ( t ) = \tau \phi _ { \beta } ( t ) - \phi _ { \beta } ( \lambda ^ { 2 } t )
$$

is Bernstein. Thus

$$
\mathbf { M M D } _ { k _ { \beta , h } } ^ { 2 } ( f \# \nu , f \# \mu ) \leq \tau \mathbf { M M D } _ { k _ { \beta , h } } ^ { 2 } ( \nu , \mu ) \qquad { \mathrm { f o r ~ e v e r y ~ } } d \geq 1 { \mathrm { ~ a n d ~ e v e r y ~ } } \mu , \nu \in \mathcal { P } ( \mathbb { R } ^ { d } )
$$

holds if and only $\mathrm { i f } \psi _ { \tau }$ is Bernstein.

Via Theorem E.5. For $0 < \beta < 1$ , the representation above shows that $\phi _ { \beta }$ is a complete Bernstein function with $b = 0$ and

$$
m _ { \beta , h } ( s ) = \frac { \beta h ^ { 2 \beta } } { \Gamma ( 1 - \beta ) } e ^ { - s / h ^ { 2 } } s ^ { - \beta - 1 } .
$$

Hence Theorem E.5 applies directly, so the smallest contraction factor across dimensions is

$$
\tau ^ { \star } ( \lambda ) = \operatorname* { s u p } _ { s > 0 } \frac { m _ { \beta , h } ( s / \lambda ^ { 2 } ) } { \lambda ^ { 2 } m _ { \beta , h } ( s ) } .
$$

Using the formula for $m _ { \beta , h }$ , we obtain for all $s > 0 ,$

$$
\begin{array} { c } { { \displaystyle \frac { m _ { \beta , h } ( s / \lambda ^ { 2 } ) } { \lambda ^ { 2 } m _ { \beta , h } ( s ) } = \displaystyle \frac { 1 } { \lambda ^ { 2 } } \frac { \frac { \beta h ^ { 2 \beta } } { ( 1 - \beta ) } e ^ { - \frac { s } { \lambda ^ { 2 } h ^ { 2 } } } \left( \frac { s } { \lambda ^ { 2 } } \right) ^ { - \beta - 1 } } { \frac { \beta h ^ { 2 \beta } } { \Gamma ( 1 - \beta ) } e ^ { - \frac { s } { \kappa ^ { 2 } } } s ^ { - \beta - 1 } } } } \\ { { = \displaystyle \frac { 1 } { \lambda ^ { 2 } } \exp \Big ( - \frac { s } { \lambda ^ { 2 } h ^ { 2 } } + \frac { s } { h ^ { 2 } } \Big ) \left( \frac { s } { \lambda ^ { 2 } } \right) ^ { - \beta - 1 } s ^ { \beta + 1 } } } \\ { { = \displaystyle \frac { 1 } { \lambda ^ { 2 } } \exp \Big ( - \frac { 1 - \lambda ^ { 2 } } { h ^ { 2 } \lambda ^ { 2 } } s \Big ) \lambda ^ { 2 ( \beta + 1 ) } } } \\ { { = \displaystyle \lambda ^ { 2 \beta } \exp \Big ( - \frac { 1 - \lambda ^ { 2 } } { h ^ { 2 } \lambda ^ { 2 } } s \Big ) \leq \lambda ^ { 2 \beta } . } } \end{array}
$$

Since $\lambda \in ( 0 , 1 )$ , the coefficient $\frac { 1 - \lambda ^ { 2 } } { h ^ { 2 } \lambda ^ { 2 } }$ is positive, so the exponential factor converges to 1 as $s \to 0 ^ { + }$ Therefore the ratio converges to $\lambda ^ { 2 \beta }$ as $s \to 0 ^ { + }$ , and the supremum is exactly $\lambda ^ { 2 \beta }$ . Thus $\tau ^ { \star } ( \lambda ) = \lambda ^ { 2 \beta }$ for $0 < \beta < 1$ . For $\beta = 1$ , one has $\phi _ { 1 } ( t ) = h ^ { 2 } t$ , hence $\dot { b } = h ^ { 2 }$ and $m \equiv 0 ,$ , so Theorem E.5 gives $\tau ^ { \star } ( \lambda ) = \lambda ^ { 2 }$ . Therefore $\tau ^ { \star } ( \lambda ) = \lambda ^ { 2 \beta }$ for all $\beta \in ( 0 , 1 ]$ . Equivalently,

$$
\Gamma _ { \tau } \mathrm { i s } \mathrm { C N D } \mathrm { o n } \mathbb { R } ^ { d } \mathrm { f o r } \mathrm { e v e r y } d \geq 1 \quad \Longleftrightarrow \quad \tau \geq \lambda ^ { 2 \beta } ,
$$

or, by Theorem E.2,

$$
\mathbf { M M D } _ { k _ { \beta , h } } ^ { 2 } ( f \# \nu , f \# \mu ) \leq \tau \mathbf { M M D } _ { k _ { \beta , h } } ^ { 2 } ( \nu , \mu ) \quad \Longleftrightarrow \quad \tau \geq \lambda ^ { 2 \beta } .
$$

Contraction factor. The smallest contraction factor across dimensions in $\mathbf { M M D ^ { 2 } }$ is $\tau ^ { \star } ( \lambda ) = \lambda ^ { 2 \beta }$ and the corresponding smallest contraction factor in MMD is $\sqrt { \tau ^ { \star } ( \lambda ) } = \lambda ^ { \beta }$ , so

$$
\mathbf { M M D } _ { k _ { \beta , h } } ( f \# \nu , f \# \mu ) \leq \lambda ^ { \beta } \mathbf { M M D } _ { k _ { \beta , h } } ( \nu , \mu ) .
$$

In particular, for $\lambda \in \mathsf { \Gamma } ( 0 , 1 )$ one has $\tau ^ { \star } ( \lambda ) < 1$ , so $f$ is a contraction in $\mathbf { M M D ^ { 2 } }$ with factor $\tau ^ { \star } ( \lambda )$ □

Proposition F.3 (No contraction under RBF kernels [25]). Fix $h > 0$ and consider the RBF kernel

$$
k _ { \mathrm { r b f } } ( x , y ) = \exp \bigl ( - h ^ { 2 } \| x - y \| ^ { 2 } \bigr ) , \qquad g ( r ) = e ^ { - h ^ { 2 } r ^ { 2 } } .
$$

Then

$$
g ( 0 ) - g ( r ) = 1 - e ^ { - h ^ { 2 } r ^ { 2 } } = \phi ( r ^ { 2 } ) , \qquad \phi ( t ) = 1 - e ^ { - h ^ { 2 } t } .
$$

Assume $f ( x ) = q + \lambda$ x with $\lambda \in ( 0 , 1 )$ . Then there is no contraction factor in $\mathbf { M M D } _ { k _ { \mathrm { r b f } } } ^ { 2 }$ , that is, $\tau ^ { \star } ( \lambda ) \geq 1$

Proof. Define

$$
\psi _ { \tau } ( t ) = \tau \phi ( t ) - \phi ( \lambda ^ { 2 } t ) , \qquad \Gamma _ { \tau } ( x , y ) = \psi _ { \tau } ( \| x - y \| ^ { 2 } ) .
$$

Lévy–Khintchine representation. The function $\phi$ is a Bernstein function with

$$
\phi ( t ) = \int _ { ( 0 , \infty ) } ( 1 - e ^ { - s t } ) \Lambda ( d s ) , \qquad \Lambda = \delta _ { h ^ { 2 } } , \qquad b = 0 .
$$

Thus the Lévy measure is a single atom at $s = h ^ { 2 }$

Bernstein test for the difference profile. By Theorem E.2, in the present radial setting, contraction in $\mathbf { M M D ^ { 2 } }$ reduces to checking whether the difference profile $\psi _ { \tau }$ is Bernstein. Equivalently,

$$
\mathbf { M M D } _ { k _ { \mathrm { r b f } } } ^ { 2 } ( f \# \nu , f \# \mu ) \leq \tau \mathbf { M M D } _ { k _ { \mathrm { r b f } } } ^ { 2 } ( \nu , \mu ) \qquad { \mathrm { f o r ~ e v e r y ~ } } d \geq 1 { \mathrm { ~ a n d ~ e v e r y ~ } } \mu , \nu \in \mathcal { P } ( \mathbb { R } ^ { d } )
$$

holds if and only if $\psi _ { \tau }$ is Bernstein.

Via finitely many atoms. Since $\Lambda = \delta _ { h ^ { 2 } }$ has a single atom, Proposition E.4 rules out the required measure domination condition. Indeed, the case $\tau \leq \lambda ^ { 2 }$ is impossible because atoms must vanish, while for $\tau > \lambda ^ { 2 }$ the atoms of Λ must lie on disjoint tails of geometric chains, so there must be infinitely many of them, not a single one. Hence $\bar { \Gamma _ { \tau } }$ cannot be CND in every ambient dimension for any $\tau \in ( 0 , 1 )$

Via saturating profiles. Proposition E.3 gives the same conclusion, since lim $_ { 1 \to \infty } \phi ( t ) = 1$ , so $\phi$ is saturating.

Contraction factor. Under the fixed affine map and the RBF kernel, there is no contraction factor in $\mathbf { M M D ^ { 2 } }$ . Equivalently,

$$
\tau ^ { \star } ( \lambda ) = \operatorname* { i n f } \{ \tau > 0 : \Gamma _ { \tau } { \mathrm { ~ i s ~ C N D ~ o n ~ } } \mathbb { R } ^ { d } { \mathrm { ~ f o r ~ e v e r y ~ } } d \geq 1 \} \geq 1 .
$$

Hence no contraction holds in $\mathbf { M M D } _ { k _ { \mathrm { r b f } } } ^ { 2 }$ , and therefore no contraction holds in $\mathbf { M M D } _ { k _ { \mathrm { r b f } } }$ either. □

## G Some new kernels inducing contraction

Preamble. This section presents kernels that induce contraction and, to the best of our knowledge, are new. First, Section G.1 goes through the complete Bernstein functions listed in [29]. We use Theorem E.5, which gives the smallest contraction factor across dimensions as a supremum over one variable, to turn the contraction test into a numerical optimization problem. Second, Section G.2 applies the geometric chaining of Section E.7 to build kernels that induce contraction with a prescribed contraction factor. This gives the proof of Theorem 6.1 of the main text, where the kernel is built from RBF kernels.

## G.1 From complete Bernstein functions

Methodology. We started from the tables of complete Bernstein functions in [29] and retained only the entries for which an explicit Lévy density $m ( s )$ is available, since this is the regime needed to apply Theorem E.5. We then discarded the entries for which the profile ϕ or the density m involves hypergeometric functions or other special functions not listed below. The retained entries are reported in Tables 3–6.

Notation for special functions. In Tables 3–6, we use the notation for special functions of [29, Section 15.1]. The Euler gammafunction is

$$
\Gamma ( z ) = \int _ { 0 } ^ { \infty } e ^ { - t } t ^ { z - 1 } d t .
$$

The lower and upper incomplete gamma functions are

$$
\gamma ( a , x ) = \int _ { 0 } ^ { x } t ^ { a - 1 } e ^ { - t } d t , \qquad \Gamma ( a , x ) = \int _ { x } ^ { \infty } t ^ { a - 1 } e ^ { - t } d t
$$

(Entries 14, 63, 64, 85, 86). The modified Besselfunctions are

$$
I _ { \nu } ( x ) = \sum _ { n = 0 } ^ { \infty } \frac { ( x / 2 ) ^ { \nu + 2 n } } { n ! \Gamma ( \nu + n + 1 ) } , \qquad K _ { \nu } ( x ) = \frac { \pi } { 2 } \frac { I _ { - \nu } ( x ) - I _ { \nu } ( x ) } { \sin ( \nu \pi ) } ,
$$

so that ${ \cal I } _ { 0 } = { \cal I } _ { \nu } | _ { \nu = 0 } , { \cal I } _ { 1 } = { \cal I } _ { \nu } | _ { \nu = 1 }$ , and $K _ { 0 } = K _ { \nu } | _ { \nu = 0 }$ (Entries 7, 57, 77, 78). The exponential integral is

$$
\operatorname { E i } ( x ) = - \int _ { - x } ^ { \infty } { \frac { e ^ { - t } } { t } } d t ,
$$

interpreted as a Cauchy principal value when $x > 0$ (Entries 29, 30, 33, 100–104). The error and complementary error functions are

$$
\mathrm { E r f } ( x ) = { \frac { 2 } { \sqrt { \pi } } } \int _ { 0 } ^ { x } e ^ { - t ^ { 2 } } d t , \qquad \mathrm { E r f c } ( x ) = 1 - \mathrm { E r f } ( x )
$$

(Entries 38, 108, 109). The generalized exponential integral is

$$
E _ { n } ( x ) = \int _ { 1 } ^ { \infty } { \frac { e ^ { - x t } } { t ^ { n } } } d t
$$

(Entry 99). The inverse hyperbolic cosine is

$$
\operatorname { a r c o s h } ( x ) = \log ( x + { \sqrt { x ^ { 2 } - 1 } } )
$$

(Entries 77, 78). Finally, log Γ denotes the logarithm of the Euler gamma function, namely x 7→ log Γ(x) (Entry 88).

<table><tr><td>Entry</td><td>Profile φ, Lévy density m, and parameter range</td><td>Ana. Num.</td><td></td></tr><tr><td>1</td><td> $\phi ( t ) = t ^ { \alpha }$ </td><td>V</td><td>V</td></tr><tr><td rowspan="4"></td><td> $m ( s ) = { \frac { \alpha } { \Gamma ( 1 - \alpha ) } } s ^ { - 1 - \alpha }$ </td><td></td><td></td></tr><tr><td> $\mathrm { V a l i d ~ f o r ~ } 0 \ : \stackrel { \cdot } { < } \alpha < 1$ </td><td></td><td></td></tr><tr><td> $\overline { { \phi ( t ) = ( t + 1 ) ^ { \alpha } - 1 } }$ </td><td>V</td><td>V</td></tr><tr><td></td><td></td><td></td></tr><tr><td rowspan="4">3</td><td> $m ( s ) = { \frac { \alpha } { \Gamma ( 1 - \alpha ) } } e ^ { - s } s ^ { - 1 - \alpha }$ </td><td></td><td></td></tr><tr><td> $\mathrm { V a l i d ~ f o r ~ } 0 \ : \stackrel { \cdot } { < } \alpha < 1$ </td><td></td><td></td></tr><tr><td> $\overline { { \phi ( t ) = 1 - ( 1 + t ) ^ { \alpha - 1 } } }$ </td><td>X</td><td>X</td></tr><tr><td> $m ( s ) = { \frac { 1 } { \Gamma ( 1 - \alpha ) } } e ^ { - s } s ^ { - \alpha }$ </td><td></td><td></td></tr><tr><td rowspan="4">4</td><td></td><td></td><td></td></tr><tr><td> $\mathrm { V a l i d ~ f o r ~ } 0 < \alpha < 1$ </td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td> $\begin{array} { c c c } { { \displaystyle { \overline { { \phi ( t ) = \frac { t } { t + a } } } } } } \\ { { \displaystyle { m ( s ) = a e ^ { - a s } } } } \end{array}$ </td><td>X</td><td>X</td></tr><tr><td rowspan="4">5</td><td></td><td></td><td></td></tr><tr><td> $\mathrm { V a l i d ~ f o r } \ : a > 0$ </td><td></td><td></td></tr><tr><td> $\overline { { \phi ( t ) = \frac { t } { \sqrt { t + a } } } }$ </td><td></td><td>V</td></tr><tr><td> $m ( s ) = { \frac { \dot { e } ^ { - a s } ( 2 a s + 1 ) } { 2 { \sqrt { \pi } } s ^ { 3 / 2 } } }$ </td><td></td><td></td></tr><tr><td rowspan="4">6</td><td></td><td></td><td></td></tr><tr><td> $\mathrm { V a l i d ~ f o r } \ a > \mathrm { ~ 0 ~ }$ </td><td></td><td></td></tr><tr><td> $\overline { { \phi ( t ) = \frac { t } { ( t + a ) ^ { \alpha } } } }$ </td><td></td><td>V</td></tr><tr><td> $m ( s ) = { \frac { \sin ( \alpha \pi ) \Gamma ( 1 - \alpha ) } { - } } e ^ { - a s } s ^ { \alpha - 2 } ( a s + 1 - \alpha )$ </td><td></td><td></td></tr><tr><td rowspan="4">7</td><td></td><td></td><td></td></tr><tr><td> $\mathrm { V a l i d ~ f o r } a > 0 , \ 0 < \alpha < 1$ </td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td> $\phi ( t ) = \sqrt { \frac { t } { t + a } }$ </td><td></td><td>X</td></tr><tr><td rowspan="4"></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td> $m ( s ) = \frac { \ " { a } } { 2 } e ^ { - a s / 2 } \left( I _ { 0 } \left( \frac { a s } { 2 } \right) - I _ { 1 } \left( \frac { a s } { 2 } \right) \right)$   $\mathrm { V a l i d ~ f o r } \overline { { a } } > 0$ </td><td></td><td></td></tr><tr><td> $\boxed { \phi ( t ) = \frac { \pi } { \sin ( \alpha \pi ) } } \frac { t ( c ^ { \alpha } - t ^ { \alpha } ) } { c - t }$ </td><td></td><td></td></tr><tr><td rowspan="4">14</td><td></td><td></td><td>V</td></tr><tr><td></td><td></td><td></td></tr><tr><td> $m ( s ) = \Gamma ( \alpha + 2 ) c ^ { \alpha + 1 } e ^ { c s } \Gamma ( - \alpha - 1 , c s )$ </td><td></td><td></td></tr><tr><td> $\mathrm { V a l i d ~ f o r } c > 0 , - 1 < \alpha < 1$   $\overline { { \phi ( t ) = \sqrt { t } \left( 1 - e ^ { - 2 a \sqrt { t } } \right) } }$ </td><td></td><td></td></tr><tr><td rowspan="4">18</td><td></td><td></td><td>V</td></tr><tr><td></td><td></td><td></td></tr><tr><td> $m ( s ) = \frac { ( 2 a ^ { 2 } - s ) e ^ { - a ^ { 2 } / s } + s } { 2 \sqrt { \pi } s ^ { 5 / 2 } }$ </td><td></td><td></td></tr><tr><td>Valid for a &gt; 0  $\overline { { \phi ( t ) = \sqrt { t } \left( 1 + e ^ { - 2 a \sqrt { t } } \right) } }$ </td><td></td><td></td></tr><tr><td rowspan="4"></td><td></td><td></td><td>X</td></tr><tr><td></td><td></td><td></td></tr><tr><td> $m ( s ) = \frac { ( s - 2 a ^ { 2 } ) e ^ { - a ^ { 2 } / s } + s } { 2 \sqrt { \pi } s ^ { 5 / 2 } }$ </td><td></td><td></td></tr><tr><td>Valid for a &gt; 0</td><td></td><td></td></tr><tr><td rowspan="4">20</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td>V</td></tr><tr><td> $\begin{array} { l c l } { \phi ( t ) = \displaystyle \frac { t \left( 1 - e ^ { - 2 \sqrt { t + a } } \right) } { \sqrt { t + a } } } \\ { m ( s ) = \displaystyle \frac { e ^ { - 1 / s - a s } \left( 2 + s ( e ^ { 1 / s } - 1 ) ( 1 + 2 a s ) \right) } { 2 \sqrt { \pi } s ^ { 5 / 2 } } } \\ { \mathrm { V a l i d ~ f o r ~ } a > 0 } \end{array}$ </td><td></td><td></td></tr><tr><td rowspan="4"></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr></table>

Table 3: Complete Bernstein functions of [29] with an explicit Lévy density. Entry numbers refer to [29]. Our table departs from [29] in nine entries. For Entries 30, 104 and 108, we correct ϕ, and for Entries 82 and 84, we correct the constant factor in m. For Entry 85, we restrict the range $\nu \in \mathbb { R }$ to $\nu < 1$ , where m is positive. For Entries 64, 131 and 132, we write m or ϕ in another form. The last two columns tell whether the entry induces contraction (V) or not (X), according to an analytic result of this paper (Ana.) and to the numerical search (Num.). An empty Ana. cell means that the paper has no analytic result for that entry.

<table><tr><td>Entry</td><td>Profile φ, Lévy density m, and parameter range</td><td></td><td>Ana. Num.</td></tr><tr><td>26</td><td> $\phi ( t ) = \log \left( 1 + \frac { t } { a } \right)$   $m ( s ) = { \frac { e ^ { - a s } } { \alpha } }$ </td><td>X</td><td>X</td></tr><tr><td>27</td><td> $\mathrm { V a l i d ~ f o r } a > 0$   $\overline { { \phi ( t ) = t \log \left( 1 + \frac { a } { t } \right) } }$   $m ( s ) = { \frac { 1 - { \dot { e } } ^ { - a s } ( 1 + a s ) } { s ^ { 2 } } }$ </td><td></td><td>X</td></tr><tr><td>28</td><td> $\mathrm { V a l i d ~ f o r } \ : a > 0$   $\phi ( t ) = \log \left( \frac { b ( t + a ) } { a ( t + b ) } \right)$   $m ( s ) = { \frac { e ^ { - { \dot { a } } s } - e ^ { - { \dot { b } } s } } { \circ } }$ </td><td></td><td>X</td></tr><tr><td>29</td><td> $\mathrm { V a l i d ~ f o r ~ } 0 < a < b$   $\phi ( t ) = \frac { 1 } { a } - \frac { 1 } { t } \log \left( 1 + \frac { t } { a } \right)$   $m ( s ) = - \operatorname { E i } ( - a s )$   $\mathrm { V a l i d ~ f o r } \ : a > 0$ </td><td></td><td>X</td></tr><tr><td>30</td><td> $\phi ( t ) = \frac { a + t } { a t } \log \left( 1 + \frac { t } { a } \right) - \frac { 1 } { a }$   $m ( s ) = { \frac { e ^ { - a s } } { a s } } + \operatorname { E i } ( - a s )$   $\mathrm { V a l i d ~ f o r } a > 0$ </td><td></td><td>X</td></tr><tr><td>31</td><td> $\phi ( t ) = ( t + a ) \log ( t + a ) - t \log t - a \log a$   $m ( s ) = { \frac { 1 - e ^ { - a s } } { s ^ { 2 } } }$  Valid for a &gt; 0</td><td></td><td>X</td></tr><tr><td>32</td><td> $\begin{array} { r } { \phi ( t ) = ( t + b ) \log ( t + b ) - b \log b - ( t + a ) \log ( t + a ) + a \log a } \end{array}$   $m ( s ) = { \frac { e ^ { - a s } - e ^ { - b s } } { s ^ { 2 } } }$   $\mathrm { V a l i d ~ f o r ~ } 0 < \stackrel { \smile } { a } < b$ </td><td></td><td>X</td></tr><tr><td>33 38</td><td> $\overline { { \phi ( t ) = \frac { t } { a - t } \log \left( \frac { a } { t } \right) } }$   $m ( s ) = a e ^ { \stackrel { \smile } { a } s } \operatorname { E i } ( - a s ) + s ^ { - 1 }$   $\mathrm { V a l i d ~ f o r } \ : a > 0$   $\phi ( t ) = { \frac { 1 } { 2 } } \left( a ^ { 2 } + a \sqrt { a ^ { 2 } + 2 t } \right) - a ^ { 2 } + t \log \left( \sqrt { 1 + { \frac { a ^ { 2 } } { 2 t } } } + { \frac { a } { \sqrt { 2 t } } } \right)$ </td><td></td><td>X V</td></tr><tr><td></td><td> $m ( s ) = \frac { 1 } { 2 s ^ { 2 } } \operatorname { E r f } \left( a \sqrt { \frac { s } { 2 } } \right)$   $\mathrm { V a l i d ~ f o r } \ : a > 0$ </td><td></td><td></td></tr><tr><td>45</td><td> $\phi ( t ) = t ( 1 + t ) \log { \left( 1 + \frac { 1 } { t } \right) } - t$   $m ( s ) = \frac { e ^ { - s } ( 2 + s ) + s - 2 } { s ^ { 3 } }$  No free parameter.</td><td></td><td>X</td></tr><tr><td>46</td><td> $\phi ( t ) = t \left( 1 - t \log \left( 1 + { \frac { 1 } { t } } \right) \right)$ </td><td></td><td>X</td></tr><tr><td></td><td> $m ( s ) = { \frac { \dot { 2 } - e ^ { - s } ( 2 + 2 s + \dot { s } ^ { 2 } ) } { s ^ { 3 } } }$  No free parameter. C</td><td></td><td></td></tr><tr><td></td><td>Entry Profile φ, Lévy density m, and parameter range</td><td>Ana. Num.</td><td></td></tr><tr><td rowspan="3">57</td><td> $\phi ( t ) = \log \Big ( 1 + 2 t + 2 \sqrt { t ( 1 + t ) } \Big )$ </td><td></td><td>X</td></tr><tr><td> $m ( s ) = s ^ { - \mathrm { \ i } } e ^ { - s / 2 } I _ { 0 } \left( \frac { s } { 2 } \right)$ </td><td></td><td></td></tr><tr><td>No free parameter.</td><td></td><td></td></tr><tr><td rowspan="3">63</td><td> $\phi ( t ) = \sqrt { t } \arctan \left( \sqrt { \frac { a } { t } } \right)$ </td><td></td><td>X</td></tr><tr><td> $m ( s ) = { \frac { \gamma \left( { \frac { 3 } { 2 } } , a s \right) } { 2 s ^ { 3 / 2 } } }$ </td><td></td><td></td></tr><tr><td> $\mathrm { V a l i d ~ f o r } \ : a \ : \stackrel {  } { > } 0$   $\begin{array} { r } { \overline { { \phi ( t ) = \sqrt { t } \arctan \left( \sqrt { \frac { t } { a } } \right) } } } \end{array}$ </td><td></td><td></td></tr><tr><td rowspan="3">64 77</td><td></td><td></td><td>V</td></tr><tr><td> $m ( s ) = { \frac { \Gamma \left( { \frac { 3 } { 2 } } , a s \right) } { 2 s ^ { 3 / 2 } } }$ </td><td></td><td></td></tr><tr><td> $\mathrm { V a l i d ~ f o r } \ : a \ : \stackrel {  } { > } 0$   $\phi ( t ) = \operatorname { a r c o s h } ( 1 + t ) = \log \Big ( 1 + t + \sqrt { t ( t + 2 ) } \Big )$ </td><td></td><td></td></tr><tr><td></td><td> $m ( s ) = { \frac { e ^ { - s } I _ { 0 } ( s ) } { s } }$  No free parameter.</td><td></td><td>X</td></tr><tr><td>78</td><td> $\phi ( t ) = { \frac { 1 } { 2 } } \operatorname { a r c o s h } ^ { 2 } ( 1 + t )$   $m ( s ) = { \frac { e ^ { - s } K _ { 0 } ( s ) } { s } }$  No free parameter.</td><td></td><td>X</td></tr><tr><td>82</td><td> $\boxed { \phi ( t ) = \frac { \Gamma \left( \frac { t + a } { 2 a } \right) } { \Gamma \left( \frac { t } { 2 a } \right) } }$  2as  $m ( s ) = \frac { a e ^ { - \omega s } } { \sqrt { \pi } \left( e ^ { 2 a s } - 1 \right) ^ { 3 / 2 } }$  Valid for a &gt; 0</td><td></td><td>V</td></tr><tr><td>83</td><td> $\overline { { \phi ( t ) = \frac { \Gamma ( \alpha t + 1 ) } { \Gamma ( \alpha t + 1 - \alpha ) } } }$  e−s/α  $m ( s ) = \frac { \epsilon } { \Gamma ( 1 - \alpha ) \left( 1 - e ^ { - s / \alpha } \right) ^ { 1 + \alpha } }$   $\mathrm { V a l i d ~ f o r ~ } 0 < \alpha < \mathrm { 1 }$ </td><td></td><td>V</td></tr><tr><td>84</td><td> $\phi ( t ) = { \frac { t \Gamma ( \alpha t + 1 - \alpha ) } { \Gamma ( \alpha t + 1 ) } }$   $m ( s ) = \frac { ( 1 - \alpha ) e ^ { s / \alpha } } { \alpha \Gamma ( 1 + \alpha ) \left( e ^ { s / \alpha } - 1 \right) ^ { 2 - \alpha } }$ </td><td></td><td>V</td></tr><tr><td>85</td><td> $\mathrm { V a l i d ~ f o r } \ 0 < \alpha < \mathrm { i }$   $\overline { { \phi ( t ) = t ^ { 1 - \nu } e ^ { t } \Gamma ( \nu , t ) } }$ </td><td></td><td>X</td></tr><tr><td>86</td><td> $m ( s ) = \frac { \Gamma ( 2 - \nu ) } { \Gamma ( 1 - \nu ) } ( 1 + s ) ^ { \nu - 2 }$  Valid for ν &lt; 1  $\phi ( t ) = t ^ { 1 - \nu } e ^ { a t } \Gamma ( \nu , a t )$ </td><td></td><td></td></tr><tr><td>88</td><td> $m ( s ) = \frac { \Gamma ( 2 - \nu ) } { \Gamma ( 1 - \nu ) } ( a + s ) ^ { \nu - 2 }$   $\mathrm { V a l i d ~ f o r } \ : a > 0 , \ : \dot { 0 } < \nu < 1$   $\phi ( t ) = \log \Gamma ( t + a + b ) - \log \Gamma ( t + a ) + \log \Gamma ( a ) - \log \Gamma ( a + b )$   $m ( s ) = { \frac { e ^ { - a s } \left( 1 - e ^ { - b s } \right) } { s \left( 1 - e ^ { - s } \right) } }$ </td><td></td><td>X</td></tr><tr><td>Entry</td><td>Profile φ, Lévy density m, and parameter range</td><td></td><td>Ana. Num.</td></tr><tr><td>99</td><td> $\phi ( t ) = t e ^ { t } E _ { n } ( t )$ </td><td>X</td><td>X</td></tr><tr><td></td><td> $m ( s ) = { \frac { n } { ( 1 + s ) ^ { n + 1 } } }$ </td><td></td><td></td></tr><tr><td>100</td><td> $\mathrm { V a l i d ~ f o r } \dot { n } \in \mathbb { N }$   $\begin{array} { r } { \phi ( t ) = - t e ^ { a t } \mathrm { E i } ( - a t ) } \end{array}$ </td><td>X</td><td>X</td></tr><tr><td></td><td> $m ( s ) = \frac { 1 } { ( a + s ) ^ { 2 } }$ </td><td></td><td></td></tr><tr><td></td><td> $\mathrm { V a l i d ~ f o r } \ : a > 0$ </td><td></td><td></td></tr><tr><td>101</td><td> $\begin{array} { r } { \overline { { \phi ( t ) = t e ^ { a t } \left( \mathrm { E i } ( - a ( b + t ) ) - \mathrm { E i } ( - a t ) \right) } } } \end{array}$ </td><td></td><td>X</td></tr><tr><td></td><td> $m ( s ) = { \frac { 1 - ( b ( s + a ) + 1 ) e ^ { - ( s + a ) b } } { ( s + a ) ^ { 2 } } }$ </td><td></td><td></td></tr><tr><td></td><td> $\mathrm { V a l i d ~ f o r } a , b > 0$ </td><td></td><td></td></tr><tr><td>102</td><td> $\begin{array} { r } { \phi ( t ) = - t e ^ { a t } \operatorname { E i } ( - a ( b + t ) ) } \end{array}$ </td><td>X</td><td>X</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td> $m ( s ) = { \frac { ( b ( s + a ) + 1 ) e ^ { - ( s + a ) b } } { ( s + a ) ^ { 2 } } }$ </td><td></td><td></td></tr><tr><td></td><td>Valid for  $a , b > 0$ </td><td></td><td></td></tr><tr><td>103</td><td></td><td>X</td><td>X</td></tr><tr><td></td><td> $\begin{array} { l } { \phi ( t ) = ( - 1 ) ^ { n + 1 } t ^ { n + 1 } e ^ { a t } \operatorname { E i } ( - a t ) + \sum _ { k = 1 } ^ { n } ( - 1 ) ^ { n - k } ( k - 1 ) ! a ^ { - k } t ^ { n - k + 1 } } \\ { m ( s ) = \displaystyle \frac { ( n + 1 ) ! } { ( a + s ) ^ { n + 2 } } } \end{array}$ </td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td> $\mathrm { V a l i d f o r } a > 0 , n \in \mathbb { N }$ </td><td></td><td></td></tr><tr><td>104</td><td> $\begin{array} { r } { \overline { { \phi ( t ) = \gamma _ { E } + \log ( a t ) - e ^ { a t } \operatorname { E i } ( - a t ) } } } \end{array}$ </td><td></td><td>X</td></tr><tr><td></td><td> $m ( s ) = { \frac { u } { s ( a + s ) } }$  C</td><td></td><td></td></tr><tr><td>108</td><td>Valid for a &gt; 0</td><td></td><td></td></tr><tr><td></td><td> $\begin{array} { r } { \overline { { \phi ( t ) = \sqrt { \pi t } } } e ^ { a t } \operatorname { E r f c } \left( \sqrt { a t } \right) } \end{array}$ </td><td>X</td><td>X</td></tr><tr><td></td><td> $m ( s ) = \frac { 1 } { 2 ( a + s ) ^ { 3 / 2 } }$  Valid for a &gt; 0</td><td></td><td></td></tr><tr><td>109</td><td> $\phi ( t ) = \sqrt { \frac { \pi } { a } } t - \pi t ^ { 3 / 2 } e ^ { a t } \operatorname { E r f c } \left( \sqrt { a t } \right)$ </td><td>X</td><td>X</td></tr><tr><td></td><td> $m ( s ) = \Gamma \left( { \frac { 5 } { 2 } } \right) ( a + s ) ^ { - 5 / 2 }$ </td><td></td><td></td></tr><tr><td>131</td><td>Valid for a &gt; ó</td><td></td><td></td></tr><tr><td></td><td> $\overline { { \phi ( t ) = - ( 2 a ) ^ { \nu } \Gamma ( - \nu ) \frac { t \Gamma \left( 1 + \nu + \frac { t } { 2 a } \right) } { ( 2 a \nu + t ) \Gamma \left( 1 + \frac { t } { 2 a } \right) } } }$ </td><td></td><td>V</td></tr><tr><td></td><td> $m ( s ) = \left( \frac { a } { \sinh ( a s ) } \right) ^ { 1 + \nu } e ^ { ( 1 - \nu ) a s }$ </td><td></td><td></td></tr><tr><td></td><td> $\mathrm { V a l i d ~ f o r ~ } \dot { a } > 0 , \ 0 \stackrel { . } { < } \nu < 1$ </td><td></td><td></td></tr><tr><td>132</td><td></td><td></td><td></td></tr><tr><td></td><td> $\phi ( t ) = ( 2 a ) ^ { \nu } \Gamma ( - \nu ) \left( \frac { \Gamma ( \mu ) } { \Gamma ( 1 - \mu ) } - \frac { \Gamma \left( \mu + \frac { t } { 2 a } \right) } { \Gamma \left( 1 - \mu + \frac { t } { 2 a } \right) } \right)$ </td><td></td><td>V</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td>1+ν</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td> $m ( s ) = \left( \frac { a } { \sinh ( a s ) } \right)$ </td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td> $\mu = \frac { 1 + \nu } { 2 } , a > 0 , 0 < \nu < 1$ </td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>Valid for</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

Table 4: Continuation of Table 3.

Table 5: Continuation of Table 3.

Table 6: Continuation of Table 3.

Reduction to a supremum. For $\lambda \in ( 0 , 1 )$ , Theorem E.5 gives the smallest contraction factor across dimensions of a complete Bernstein function as

$$
\tau ^ { \star } ( \lambda ) = \operatorname* { m a x } \left\{ \lambda ^ { 2 } \mathbf { 1 } _ { \{ b > 0 \} } , \operatorname* { s u p } _ { s > 0 } R _ { \lambda } ( s ) \right\} , \qquad R _ { \lambda } ( s ) : = \frac { m ( s / \lambda ^ { 2 } ) } { \lambda ^ { 2 } m ( s ) } .
$$

Every entry of Tables 3–6 has drift $b = 0 , { \bf s o } \tau ^ { \star } ( \lambda )$ is the supremum of $R _ { \lambda }$ over $s > 0$ . Deciding whether an entry induces contraction thus amounts to computing, for each λ, the supremum of a function of one variable.

Two difficulties. Computing this supremum numerically raises two difficulties. First, s ranges over (0, ∞), but a numerical search can only explore a bounded range. The supremum may even be approached only as $s \to 0 ^ { + }$ or as $s \to \infty$ . The search mitigates this with a wide window, and reports when its best point lies at an end of the window, since the supremum may then lie further out. Second, the shape of $R _ { \lambda }$ is not known in advance, and a single local search may stop at a local maximum below the supremum. The search mitigates this by combining a fine grid over the whole window with local searches around its best points.

Algorithm 2 Search for the supremum of the ratio $R _ { \lambda }$   
1: Require: Discount $\lambda \in ( 0 , 1 )$ , Lévy density $m : ( 0 , \infty ) \to ( 0 , \infty )$   
2: Require: Parameters of Table $7 U , n _ { \mathrm { g r i d } } , K , c _ { \mathrm { l o c } }$   
3: Output: Estimate $\widehat { \tau } ^ { \star } ( \lambda )$ of $\operatorname* { s u p } _ { s > 0 } R _ { \lambda } ( s )$ , best point $u ^ { \star } , s ^ { \star } = e ^ { u ^ { \star } }$ , and whether $u ^ { \star }$ is an end of   
the window   
4: Set   
$\ell _ { \lambda } ( u ) \gets \log m ( e ^ { u } / \lambda ^ { 2 } ) - \log m ( e ^ { u } ) - \log ( \lambda ^ { 2 } )$   
5: Evaluate the objective $\ell _ { \lambda }$ on a $g r i d$ over the window.   
6: Form a uniform grid $u _ { 1 } , \dotsc , u _ { n _ { \mathrm { g r i d } } }$ on $[ - U , U ]$   
7: Evaluate $\ell _ { \lambda } ( u _ { j } )$ on the grid and select the K grid points with the largest values   
8: Set   
$\Delta u  { \frac { 2 U } { n _ { \mathrm { g r i d } } - 1 } }$   
9: for each of the K selected grid points $u _ { j }$ do   
10: Define the local interval   
$[ a _ { j } , b _ { j } ] \gets [ \mathrm { m a x } ( - U , u _ { j } - c _ { \mathrm { l o c } } \Delta u ) , \mathrm { m i n } ( U , u _ { j } + c _ { \mathrm { l o c } } \Delta u ) ]$   
11: Maximize $\ell _ { \lambda }$ on $[ a _ { j } , b _ { j } ]$ with minimize\_scalar, giving the point $\tilde { u } _ { j }$ and its value ${ \tilde { L } } _ { j }$   
12: end for   
13: Let $( u ^ { \star } , L ^ { \star } )$ be the pair $( \tilde { u } _ { j } , \tilde { L } _ { j } )$ with the largest value ${ \tilde { L } } _ { j }$   
14: return $\exp ( L ^ { \star } ) , u ^ { \star } , e ^ { u ^ { \star } }$ , and whether $| u ^ { \star } | = U$

Logarithmic scale. To treat small and large values of s alike, we write $s = e ^ { u }$ and search over $u \in \mathbb { R } . \mathrm { ~ A ~ }$ window $[ - U , U ]$ in u then covers s from $e ^ { - U }$ to $e ^ { U }$ , and a uniform grid in u puts about the same number of points in each interval $[ 1 0 ^ { k } , 1 0 ^ { k + 1 } ]$ of s. We also maximize the logarithm of the ratio,

$$
\ell _ { \lambda } ( u ) : = \log R _ { \lambda } ( e ^ { u } ) = \log m \left( \frac { e ^ { u } } { \lambda ^ { 2 } } \right) - \log m ( e ^ { u } ) - \log ( \lambda ^ { 2 } ) ,
$$

which avoids overflow and underflow when m is very large or very small. Since the logarithm is increasing,

$$
\tau ^ { \star } ( \lambda ) = \operatorname* { s u p } _ { s > 0 } R _ { \lambda } ( s ) = \exp \Bigl ( \operatorname* { s u p } _ { u \in \mathbb { R } } \ell _ { \lambda } ( u ) \Bigr ) .
$$

The search. The search maximizes $\ell _ { \lambda }$ on a window $[ - U , U ]$ in two steps.

1. Grid. It evaluates $\ell _ { \lambda }$ at $n _ { \mathrm { g r i d } }$ equally spaced points of $[ - U , U ]$ , spaced $\Delta u = 2 U / ( n _ { \mathrm { g r i d } } - 1 )$ apart, and selects the $K$ points with the largest values.

2. Local searches. It then solves K local maximization problems, one around each selected point $u _ { 0 } ,$ on the interval $[ u _ { 0 } \mathrm { ~ - ~ } c _ { \mathrm { l o c } } \Delta u$ $u _ { 0 } + c _ { \mathrm { l o c } } \Delta u ]$ cut to $[ - U , U ]$ . We use minimize\_scalar, the routine from SciPy [35] for optimizing a function of one variable on an interval. The best of the K results is denoted $u ^ { \star }$

The search returns exp $( \ell _ { \lambda } ( u ^ { \star } ) )$ as its estimate of $\tau ^ { \star } ( \lambda )$ , together with whether $u ^ { \star }$ is an end of the window. Algorithm 2 gives the details, and Table 7 lists the values of $U , n _ { \mathrm { g r i d } } , K$ and $c _ { \mathrm { l o c } }$ . The search does not guarantee that the supremum is found.

<table><tr><td>Setting</td><td>Value</td><td>Description</td></tr><tr><td>Window U</td><td>80</td><td>The window is  $[ - U , U ] .$ </td></tr><tr><td>Grid size  $n _ { \mathrm { g r i d } }$ </td><td>8500</td><td>Number of grid points on the window.</td></tr><tr><td>Selected grid points K</td><td>12</td><td>Number of best grid points around which  $\ell _ { \lambda }$  is maximized locally.</td></tr><tr><td>Local interval  $c _ { \mathrm { l o c } }$ </td><td>120</td><td>The local interval around a selected grid point uo is  $[ u _ { 0 } -$   $c _ { \mathrm { l o c } } \Delta u , u _ { 0 } + c _ { \mathrm { l o c } } \Delta u ]$ </td></tr><tr><td>Local tolerance</td><td> $1 0 ^ { - 1 0 }$ </td><td>Precision on u at which minimize_scalar stops.</td></tr></table>

Table 7: Values of the constants of Algorithm 2.

Experimental setup. We run the search on CPUs, at 100 equally spaced values of λ from 0.01 to 0.99, with the constants of Table 7. Entries with parameters use α = 0.6 in Entries 1–3, $\alpha = \nu = 0 . 5$ in the others, $b = 3$ in Entries 28 and 32, c = 2 in Entry 14, and $a = b = n = 1$ otherwise. Figure 9 varies one of these parameters.

Numerical precision. The values of the density m are computed with mpmath [24] at 80 significant digits, so that they stay accurate where they are very large or very small. Their logarithms are then rounded to double precision, in which the grid and the local searches work.

Checks against known results. Entries 1 and 2 give the power-distance and generalized multiquadric kernels, whose smallest contraction factors across dimensions are known in closed form (Propositions F.1 and F.2). For both entries, the search recovers these values up to rounding errors. Entry 2 shows that the search can find the supremum even when it is approached only as $s \to 0 ^ { + }$ Proposition G.1 proves that Entries 3, 4 and 26 do not induce contraction, in agreement with the search. By Corollary E.5.1, an entry does not induce contraction if its Lévy density m(s) has a finite limit as $s \to 0 ^ { + }$ . Entries 99, 100, 102, 103, 108 and 109 meet this condition. More entries can be confirmed with this corollary, but for brevity we only list those for which the limit is immediate.

## G.1.1 Entries that do not induce contraction

The search finds two groups of entries that do not induce contraction. For the entries of Figure $^ { 6 , }$ the estimate of $\tau ^ { \star } ( \lambda )$ exceeds 1 for some λ. Since the estimate never exceeds $\tau ^ { \star } ( \lambda )$ , these entries do not induce contraction. For the entries of Figure 7, the estimate equals 1 up to numerical precision for every λ. These entries do not induce contraction either, but they may be non-expansive.

Entry 78 also belongs to the second group, although its estimate stays below 1. At every $\lambda ,$ its best point is the left end of the window, $u ^ { \star } = - 8 0$ , so its estimate in Figure 7 may miss a larger value further left. Further left, the ratio keeps increasing towards 1. We conclude that, for Entry 78, $\tau ^ { \star } ( \lambda ) = 1$ for every $\lambda \in ( 0 , 1 )$ ), although its estimate in Figure 7, limited by the window, stays below 1.

![](images/41b7265181ced6163e52fa8b4118e51a827088ba00c0e6b91881bb22800ba543.jpg)  
Figure 6: Entries whose estimate of $\tau ^ { \star } ( \lambda )$ exceeds 1 for some λ. Entry numbers refer to [29].

## G.1.2 Entries that induce contraction

We accept an entry when its estimate of $\tau ^ { \star } ( \lambda )$ stays below 1 for every $\lambda ,$ and its best point is never at an end of the window. Table 8 lists the accepted entries, and Figures 8 and 11 show their estimates of $\tau ^ { \star } ( \lambda )$ and their profiles $\phi .$ For an entry with a parameter, these results hold at the parameter values given in the experimental setup. Contraction may then hold only on part of the range allowed in [29]. Figure 9 shows, for seven such entries, how the estimate approaches or crosses 1 as the parameter varies.

![](images/15f7700774dbb7aeb49a012aa160cb624ab4b252b75758045e559c41599f6c0b.jpg)  
Figure 7: Entries whose estimate of $\tau ^ { \star } ( \lambda )$ equals 1 up to numerical precision at every $\lambda ,$ and Entry 78. Entry numbers refer to [29].

For seven entries of Table 8, the parameter values giving contraction need no computation. Propositions F.1 and F.2 cover every value of the parameter of Entries 1 and 2. In Entries 5, 18, 38, 64 and 82, the parameter only changes the scale of t, which leaves $\tau ^ { \star } ( \lambda )$ unchanged. For the other seven, Entries 6, 14, 20, 83, 84, 131 and 132, whether the entry induces contraction may depend on the value of its parameter. We therefore ran the search for many values of the parameter across its range. Figure 10 shows the largest estimate over λ for each value. When this curve crosses 1, a root search on the parameter locates where it does, which gives the range in Table 8.

<table><tr><td>Entry</td><td>Parameter values giving contraction</td></tr><tr><td>1</td><td>every value</td></tr><tr><td>2</td><td>every value</td></tr><tr><td>5</td><td>every value</td></tr><tr><td>6</td><td> $\alpha \leq 3 / 4$ </td></tr><tr><td>14</td><td> $\alpha > 0$ </td></tr><tr><td>18</td><td>every value</td></tr><tr><td>20</td><td>every value</td></tr><tr><td>38</td><td>every value</td></tr><tr><td>64</td><td>every value</td></tr><tr><td>82</td><td>every value</td></tr><tr><td>83</td><td>every value</td></tr><tr><td>84</td><td>α up to about 0.70</td></tr><tr><td>131</td><td>ν from about 0.30</td></tr><tr><td>132</td><td>every value</td></tr></table>

Table 8: Entries that induce contraction. Entry numbers refer to [29]. The second column gives the parameter values for which the entry induces contraction, among those allowed in [29]. Entry 1 gives the power-distance kernel studied in [25], and Entry 2 with $\alpha \stackrel { - } { = } 1 / 2$ gives the multiquadric kernel studied in [19]. Entry 5 is Entry 6 with $\alpha = 1 / 2$

![](images/67281f0c7275d7cec470caef7866801667d575c9f4394707566f68b4f22299c0.jpg)  
Figure 8: Estimates of $\tau ^ { \star } ( \lambda )$ for the entries of Table 8. Entry numbers refer to [29].

![](images/9dbf01d50daafaeee5fe94354330831e54e4402568ea8dceab8e7eb45d99537a.jpg)  
Figure 9: Estimates of $\tau ^ { \star } ( \lambda )$ for seven entries with a parameter, at several values of the parameter. An entry induces contraction at a value of its parameter when its curve stays below the dashed line $\tau ^ { \star } ( \lambda ) = 1$ . Entry numbers refer to [29].

![](images/d0fef6bf8b60b2819c36226babf11507f1ba3d243572787b182dc11f1a65fd3f.jpg)

![](images/84a33f41fdc41fe16a9dd5f8c645eb30f0ab10cdaf36f361807cdef8aadbc77f.jpg)

![](images/3761883db732bcc0949e786af5ebf68ab8b513f20f888e814615e430ce97c323.jpg)

![](images/386057a7f40cc6291dd569f68e4cda69564c4bcf748332f39c994eb82e1b2c8f.jpg)

![](images/af591f19c50e452b1f267936cd4094c089c0682238212bb7069f904f0c38299a.jpg)

![](images/b7d3153886a7bbb9c3e84fd92407b8a551d6acc0cb05c4f141fcef0652f154e3.jpg)

![](images/4370cd9f9e45382a78d93ed2cbb98f99a394735226c0f1f706f42096b4974921.jpg)  
Figure 10: Each panel shows, for one entry of Table 8, the largest estimate of $\tau ^ { \star } ( \lambda )$ over λ as a function of the entry’s parameter. The entry induces contraction where the curve is below the dashed line. The dotted lines mark where the curve crosses 1, found by a root search. Entry numbers refer to [29].

![](images/a7093f2acc864343934c805cba2209e1ecee70a368bfb4ed88f5a26d254d875a.jpg)  
Figure 11: Profiles ϕ of the entries of Table $^ { 8 , }$ on logarithmic axes. Entry numbers refer to [29].

## G.1.3 Entries 3, 4 and 26 in closed form

Proposition G.1 (No contraction under inverse multiquadric, rational and logarithmic kernels). Assume $f ( x ) = q + \lambda x$ with $\lambda \in \mathsf { \Gamma } ( 0 , 1 )$ . For each kernel k below, with its profile $\phi ,$ , the stated $\tau ^ { \star } ( \lambda )$ is the smallest τ such that $\mathbf { M M D } _ { k } ^ { 2 } ( f \# \nu , f \# \mu ) \leq \tau \mathbf { M M D } _ { k } ^ { 2 } ( \nu , \mu )$ for every $d \geq 1$ and every $\mu , \nu \in \mathcal P ( \mathbb { R } ^ { d } )$ .

(i) Inverse multiquadric kernels, $\alpha \in ( 0 , 1 )$

$$
k _ { \alpha } ( x , y ) = \left( 1 + \| x - y \| ^ { 2 } \right) ^ { \alpha - 1 } , \qquad \phi _ { \alpha } ( t ) = 1 - ( 1 + t ) ^ { \alpha - 1 } , \qquad \tau ^ { \star } ( \lambda ) = \lambda ^ { 2 \alpha - 2 } .
$$

(ii) Rational kernels, $a > 0 \colon$

$$
k _ { a } ( x , y ) = \frac { a } { \| x - y \| ^ { 2 } + a } , \qquad \phi _ { a } ( t ) = \frac { t } { t + a } , \qquad \tau ^ { \star } ( \lambda ) = \lambda ^ { - 2 } .
$$

(iii) Logarithmic kernels, $a > 0 \colon$

$$
k _ { a } ( x , y ) = - \log \left( 1 + \frac { \| x - y \| ^ { 2 } } { a } \right) , \qquad \phi _ { a } ( t ) = \log \left( 1 + \frac { t } { a } \right) , \qquad \tau ^ { \star } ( \lambda ) = 1 .
$$

For (i) and $( \mathrm { i i } ) , \tau ^ { \star } ( \lambda ) > 1$ , so for some $d , \mu$ and ν the map f increases $ { \mathbf { M M D } } _ { k } ^ { 2 }$ . For (iii), $\tau ^ { \star } ( \lambda ) = 1$ so f is non-expansive in $\mathbf { M M D } _ { k _ { a } } ^ { 2 }$ , that is,

MMD<sup>2</sup><sub>k</sub> (f#ν, f#µ) ≤ MMD<sup>2</sup><sub>k</sub> (ν, µ) for every $d \geq 1$ and every µ, ν ∈ P(R<sup>d</sup>), but f is not a contraction in $\mathbf { M M D } _ { k _ { a } } ^ { 2 }$

Proof. Lévy–Khintchine representations. For (i), (ii) and (iii), Entries 3, 4 and 26 of Tables 3–4 give $b = 0$ and the Lévy densities

$$
\mathrm { ( i ) } m ( s ) = \frac { e ^ { - s } s ^ { - \alpha } } { \Gamma ( 1 - \alpha ) } , \qquad \mathrm { ( i i ) } m ( s ) = a e ^ { - a s } , \qquad \mathrm { ( i i i ) } m ( s ) = \frac { e ^ { - a s } } { s } , \qquad s > 0 .
$$

Application of Theorem E.5. Set $\rho = \lambda ^ { 2 } \in ( 0 , 1 )$ . Since $b = 0$ , Theorem E.5 gives, in each case,

$$
\tau ^ { \star } ( \lambda ) = \operatorname* { s u p } _ { s > 0 } R _ { \lambda } ( s ) , \qquad R _ { \lambda } ( s ) : = \frac { m ( s / \rho ) } { \rho m ( s ) } .
$$

Using the formula for m in each case, we obtain

$$
\mathrm { ( i ) } \quad R _ { \lambda } ( s ) = { \frac { 1 } { \rho } } { \frac { \Gamma ( 1 - \alpha ) ^ { - 1 } e ^ { - s / \rho } ( s / \rho ) ^ { - \alpha } } { \Gamma ( 1 - \alpha ) ^ { - 1 } e ^ { - s } s ^ { - \alpha } } } = \rho ^ { \alpha - 1 } \exp \left( - s \left( { \frac { 1 } { \rho } } - 1 \right) \right) ,
$$

$$
\mathrm { ( i i ) } R _ { \lambda } ( s ) = \frac { 1 } { \rho } \frac { a e ^ { - a s / \rho } } { a e ^ { - a s } } = \rho ^ { - 1 } \exp \left( - a s \left( \frac { 1 } { \rho } - 1 \right) \right) ,
$$

$$
\mathrm { ( i i i ) } R _ { \lambda } ( s ) = \frac 1 \rho \frac { e ^ { - a s / \rho } } { s / \rho } \frac s { e ^ { - a s } } = \exp \left( - a s \left( \frac 1 \rho - 1 \right) \right) .
$$

Suprema of the ratios. For (i), (ii) and (iii), we show that the ratios $R _ { \lambda }$ stay below $\rho ^ { \alpha - 1 } , \rho ^ { - 1 }$ and 1, respectively, and approach these values as $s \to 0 ^ { + }$ , which makes these values their suprema.

Since $0 < \rho < 1$ , the factor $1 / \rho - 1$ in each exponent above is positive. Each exponent is therefore a negative constant times $s ,$ so it is negative for every $s > 0$ and tends to 0 as $s \to 0 ^ { + }$ . Each exponential thus lies strictly between 0 and 1 for every $s > 0 ,$ , and tends to 1 as $s \to 0 ^ { + }$

In the expressions above, the ratios $R _ { \lambda } ( s )$ are these exponentials multiplied by $\rho ^ { \alpha - 1 }$ for (i), by $\rho ^ { - 1 }$ for (ii) and by 1 for (iii). Hence, for every $s > 0 ,$

$$
\mathrm { ( i ) } \ 0 < R _ { \lambda } ( s ) < \rho ^ { \alpha - 1 } , \qquad \mathrm { ( i i ) } \ 0 < R _ { \lambda } ( s ) < \rho ^ { - 1 } , \qquad \mathrm { ( i i i ) } \ 0 < R _ { \lambda } ( s ) < 1 .
$$

The ratios also tend to their upper bounds as $s \to 0 ^ { + }$ . Therefore, these upper bounds are their suprema, and Theorem E.5 yields

$$
\mathrm { ( i ) } \tau ^ { \star } ( \lambda ) = \rho ^ { \alpha - 1 } = \lambda ^ { 2 \alpha - 2 } , \qquad \mathrm { ( i i ) } \tau ^ { \star } ( \lambda ) = \rho ^ { - 1 } = \lambda ^ { - 2 } , \qquad \mathrm { ( i i i ) } \tau ^ { \star } ( \lambda ) = 1 .
$$

Conclusions. For (i), since $\alpha \in ( 0 , 1 )$ , we have $2 \alpha - 2 < 0 .$ , hence $\tau ^ { \star } ( \lambda ) = \lambda ^ { 2 \alpha - 2 } > 1$ for every $0 < \lambda < 1$ . For $( \mathrm { i i } ) , \tau ^ { \star } ( \lambda ) = \lambda ^ { - 2 } > 1$ for every $0 < \lambda < 1$ □

## G.2 From geometric chaining

Proposition G.2 (Geometric chaining builds kernels with prescribed contraction factor). Fix $\lambda \in$ $( 0 , 1 )$ and set $\rho : = \lambda ^ { 2 } \in ( 0 , 1 )$ . Let $\tau \in \mathsf { \Gamma } ( 0 , 1 )$ , let $\phi _ { 0 }$ be a Bernstein function on $[ 0 , \infty )$ with $\phi _ { 0 } ( 0 ) = 0 \quad$ , and define

$$
h _ { 0 } ( r ) : = \phi _ { 0 } ( r ^ { 2 } ) , \qquad r \geq 0 .
$$

Assume that the chained profile

$$
\phi ( t ) : = \sum _ { n = 0 } ^ { \infty } \tau ^ { - n } \phi _ { 0 } ( \rho ^ { n } t ) , \qquad t \geq 0 ,
$$

is well defined in the sense of Proposition E.7. Define its radial profile

$$
h _ { \mathrm { c h a i n } } ( r ) : = \phi ( r ^ { 2 } ) , \qquad r \geq 0 ,
$$

and, for an arbitrary constant $C \in \mathbb { R }$

$$
k _ { \mathrm { c h a i n } } ( x , y ) : = C - h _ { \mathrm { c h a i n } } ( \| x - y \| ) , \qquad x , y \in \mathbb { R } ^ { d } .
$$

Then, for every $d \geq 1$ , every $q \in \mathbb { R } ^ { d }$ and every affine map $f ( x ) = q + \lambda x .$ , one has

$$
\boxed { \mathbf { M M D } _ { k _ { \mathrm { c h a i n } } } ^ { 2 } ( f \# \nu , f \# \mu ) \leq \tau \mathbf { M M D } _ { k _ { \mathrm { c h a i n } } } ^ { 2 } ( \nu , \mu ) \qquad \forall \mu , \nu \in \mathcal { P } ( \mathbb { R } ^ { d } ) . }
$$

Moreover, the chained radial profile is given by

$$
\boxed { h _ { \mathrm { c h a i n } } ( r ) = \sum _ { n = 0 } ^ { \infty } \tau ^ { - n } h _ { 0 } ( \lambda ^ { n } r ) , \qquad r \geq 0 , }
$$

and satisfies the exact identity

$$
\begin{array} { r } { \boxed { \tau h _ { \mathrm { c h a i n } } ( r ) - h _ { \mathrm { c h a i n } } ( \lambda r ) = \tau h _ { 0 } ( r ) , \qquad r \geq 0 . } } \end{array}
$$

Thus, whenever the chained profile is well defined, every such affine map f is a τ-contraction in $\mathbf { M M D ^ { 2 } }$ under the radial kernel it produces.

Proof. By Proposition E.7, the chained profile ϕ defines a Bernstein function. Therefore Theorem E.6 applies and yields

$$
\mathbf { M M D } _ { k _ { \mathrm { c h a i n } } } ^ { 2 } ( f \# \nu , f \# \mu ) \leq \tau \mathbf { M M D } _ { k _ { \mathrm { c h a i n } } } ^ { 2 } ( \nu , \mu ) \qquad \forall d \geq 1 , \forall q \in \mathbb { R } ^ { d } , \forall \mu , \nu \in \mathcal { P } ( \mathbb { R } ^ { d } ) ,
$$

for $f ( x ) = q + \lambda x .$

It remains to identify the chained radial profile. Since $\rho = \lambda ^ { 2 }$

$$
h _ { \mathrm { c h a i n } } ( r ) = \phi ( r ^ { 2 } ) = \sum _ { n = 0 } ^ { \infty } \tau ^ { - n } \phi _ { 0 } ( \rho ^ { n } r ^ { 2 } ) = \sum _ { n = 0 } ^ { \infty } \tau ^ { - n } \phi _ { 0 } ( ( \lambda ^ { n } r ) ^ { 2 } ) = \sum _ { n = 0 } ^ { \infty } \tau ^ { - n } h _ { 0 } ( \lambda ^ { n } r ) .
$$

Finally, Proposition E.6 gives

$$
\tau \phi ( t ) - \phi ( \rho t ) = \tau \phi _ { 0 } ( t ) \qquad \forall t \geq 0 .
$$

Substituting $t = r ^ { 2 }$ and using $\rho r ^ { 2 } = ( \lambda r ) ^ { 2 }$ yields

$$
\tau h _ { \mathrm { c h a i n } } ( r ) - h _ { \mathrm { c h a i n } } ( \lambda r ) = \tau h _ { 0 } ( r ) , \qquad r \geq 0 .
$$

Proposition G.3 (The chained RBF kernel realizes any prescribed smallest contraction factor across dimensions). Fix $\lambda \in ( 0 , 1 )$ and set $\rho : = \lambda ^ { 2 } \in ( 0 , 1 )$ . Let $\tau \in ( \rho , 1 ) , a > 0$ , and $c > 0 .$ , and define

$$
\phi _ { \mathrm { c h a i n } } ( t ) : = \sum _ { n = 0 } ^ { \infty } c \tau ^ { - n } \big ( 1 - e ^ { - a \rho ^ { n } t } \big ) , \qquad h _ { \mathrm { c h a i n } } ( r ) : = \phi _ { \mathrm { c h a i n } } ( r ^ { 2 } ) .
$$

For an arbitrary constant $C \in \mathbb { R }$ , let

$$
k _ { \mathrm { c h a i n } } ( x , y ) : = C - h _ { \mathrm { c h a i n } } ( \| x - y \| ) , \qquad x , y \in \mathbb { R } ^ { d } .
$$

Then, for every $d \geq 1$ , every $q \in \mathbb { R } ^ { d }$ , and every affine map

$$
f ( x ) = q + \lambda x ,
$$

one has

$$
\mathbf { M M D } _ { k _ { \mathrm { c h a i n } } } ^ { 2 } ( f \# \nu , f \# \mu ) \leq \tau \mathbf { M M D } _ { k _ { \mathrm { c h a i n } } } ^ { 2 } ( \nu , \mu ) \qquad \forall \mu , \nu \in \mathcal { P } ( \mathbb { R } ^ { d } ) .
$$

Moreover, τ is the smallest contraction factor for $k _ { \mathrm { c h a i n } } .$ namely

$$
\tau ^ { \star } ( \lambda ) = \tau .
$$

Thus, for every prescribed factor $\tau \in ( \lambda ^ { 2 } , 1 )$ , this construction yields a kernel whose smallest contraction factor across dimensions is exactly τ. For the chained RBF kernel, smaller choices of τ lead to faster contraction, and

$$
\operatorname * { i n f } _ { \tau > \lambda ^ { 2 } } \tau ^ { \star } ( \lambda ) = \lambda ^ { 2 } ,
$$

although this value is not attained since the construction requires $\tau > \lambda ^ { 2 }$

If we introduce the lengthscales

$$
\ell _ { n } : = \frac { 1 } { \sqrt { a \rho ^ { n } } } = \frac { 1 } { \sqrt { a } } \lambda ^ { - n } , \qquad n \geq 0 ,
$$

then $a \rho ^ { n } = 1 / \ell _ { n } ^ { 2 }$ , and

$$
h _ { \mathrm { c h a i n } } ( r ) = \sum _ { n = 0 } ^ { \infty } c \tau ^ { - n } \Bigl ( 1 - \exp \bigl ( - r ^ { 2 } / \ell _ { n } ^ { 2 } \bigr ) \Bigr ) , \qquad r \geq 0 .
$$

Thus the geometric rescaling $r \mapsto \lambda ^ { n } r$ can be viewed as a series of RBF kernels with geometrically increasing lengthscales $( \ell _ { n } ) _ { n \geq 0 }$

Proof.

Smallest contraction factor. The base Bernstein function

$$
\phi _ { 0 } ( t ) : = c \big ( 1 - e ^ { - a t } \big )
$$

has zero drift and a Lévy measure with a single atom, $c \delta _ { a }$ . Thus the present construction is exactly the single geometric chain of atoms from Proposition E.5, with $s _ { 0 } = a$ . Since $\tau > \rho = \lambda ^ { 2 }$ , that proposition shows that $\phi _ { \mathrm { c h a i n } }$ is a Bernstein function and that, under the induced radial kernel, the affine map $x \mapsto q + .$ λx is a τ-contraction in $\mathbf { M M D ^ { 2 } }$ . In particular, for the kernel designed with parameter τ, this yields

$$
\tau ^ { \star } ( \lambda ) \leq \tau .
$$

This is also the RBF case of the general chaining results, namely Propositions E.6 and E.7 together with Proposition G.2.

It remains to prove

$$
\tau ^ { \star } ( \lambda ) \geq \tau ,
$$

since together with the already established $\tau ^ { \star } ( \lambda ) \leq \tau$ , this yields

$$
\tau ^ { \star } ( \lambda ) = \tau .
$$

Let $\bar { \tau } > 0$ be any contraction factor in $\mathbf { M M D } _ { k _ { \mathrm { c h a i n } } } ^ { 2 }$ . Write

$$
s _ { n } : = \rho ^ { n } a , \qquad \Lambda ( d s ) = \sum _ { n = 0 } ^ { \infty } c \tau ^ { - n } \delta _ { s _ { n } } ( d s ) .
$$

By condition 3 of Theorem $\operatorname { E . 4 , }$ one must have

$$
\bar { \tau } \Lambda ( \{ x \} ) \geq \Lambda ( \{ x / \rho \} ) \qquad \mathrm { f o r ~ e v e r y ~ } x > 0 .
$$

Applying this with $x = s _ { n }$ for $n \geq 1$ gives

$$
\bar { \tau } \Lambda ( \{ s _ { n } \} ) \geq \Lambda ( \{ s _ { n - 1 } \} ) .
$$

Since

$$
\Lambda ( \{ s _ { n } \} ) = c \tau ^ { - n } , \qquad \Lambda ( \{ s _ { n - 1 } \} ) = c \tau ^ { - ( n - 1 ) } ,
$$

it follows that

$$
\bar { \tau } c \tau ^ { - n } \geq c \tau ^ { - ( n - 1 ) } ,
$$

hence

$$
\bar { \tau } \geq \tau .
$$

Therefore $\tau ^ { \star } ( \lambda ) \geq \tau$ , and so $\tau ^ { \star } ( \lambda ) = \tau$ . Taking the infimum over $\tau > \lambda ^ { 2 }$ gives

$$
\operatorname* { i n f } _ { \tau > \lambda ^ { 2 } } \tau ^ { \star } ( \lambda ) = \lambda ^ { 2 } .
$$

Explicit form as a series of RBF kernels. By definition,

$$
\phi _ { \mathrm { c h a i n } } ( t ) = \sum _ { n = 0 } ^ { \infty } c \tau ^ { - n } \big ( 1 - e ^ { - a \rho ^ { n } t } \big ) ,
$$

and therefore

$$
h _ { \mathrm { c h a i n } } ( r ) = \phi _ { \mathrm { c h a i n } } ( r ^ { 2 } ) = \sum _ { n = 0 } ^ { \infty } c \tau ^ { - n } \Bigl ( 1 - e ^ { - a \rho ^ { n } r ^ { 2 } } \Bigr ) .
$$

Introducing

$$
\ell _ { n } = \frac { 1 } { \sqrt { a \rho ^ { n } } } = \frac { 1 } { \sqrt { a } } \lambda ^ { - n }
$$

gives $a \rho ^ { n } = 1 / \ell _ { n } ^ { 2 } .$ , so

$$
h _ { \mathrm { c h a i n } } ( r ) = \sum _ { n = 0 } ^ { \infty } c \tau ^ { - n } \Bigl ( 1 - \exp \bigl ( - r ^ { 2 } / \ell _ { n } ^ { 2 } \bigr ) \Bigr ) .
$$

Finally,

$$
k _ { \mathrm { c h a i n } } ( x , y ) = C - \sum _ { n = 0 } ^ { \infty } c \tau ^ { - n } \Bigl ( 1 - \exp \bigl ( - \| x - y \| ^ { 2 } / \ell _ { n } ^ { 2 } \bigr ) \Bigr ) ,
$$

which is the form used in Theorem 6.1.

![](images/7586c260db43a5f68d82b39e7f9db34cbf9bd0e5f9a5e2d6aab9f21805a2c6d8.jpg)

![](images/d3401acbb9aaadad231d456b42bb5ae5d99432f0b99832a688b870c3e0d0d209.jpg)  
Figure 12: Bernstein functions $\begin{array} { r l r } { \phi _ { \tau } ( t ) } & { { } = } & { \sum _ { n = 0 } ^ { K } c \tau ^ { - n } ( 1 ~ - ~ e ^ { - a \gamma ^ { 2 n } t } ) } \end{array}$ for $\tau \in \{ 0 . 8 4 , 0 . 9 0 , 0 . 9 6 \}$ , truncated at $K \stackrel { \cdot } { = } 7 0 0$ with $\gamma = 0 . 9 0 , a = 0 . 2 0$  
Figure 13: Lévy measure atoms $( s _ { n } , c \tau ^ { - n } )$ with $s _ { n } = a \gamma ^ { 2 n }$ , first 35 atoms shown on a log-log scale.

![](images/413ff1930ad043fb96861a4125cc0d2adcc6b5dcba74d500533ee9c47dc54f4a.jpg)  
Figure 14: Share of the omitted terms $R _ { K } ^ { ( N ) } ( t )$ for the chained RBF kernel. The white curve shows the truncation threshold $K _ { 1 0 ^ { - 3 } } ( t )$ , beyond which the omitted terms account for at most $\mathrm { \dot { 1 } 0 ^ { - 3 } }$ of the approximated radial profile.

![](images/a791152cbf4209f7d5ca67877702e321c1bea6a6136da1db6a50566cf0c80bf3.jpg)  
Figure 15: The same share of the omitted terms when $\rho / \tau$ is closer to one. The white curve shifts upward, showing that more terms are needed before the omitted terms account for less than $1 0 ^ { - 3 }$ of the approximated radial profile.

Table 9: Contraction of truncated chains on a ball, for the two chains of Figures 14 and 15, with $\rho = 0 . 9 0 2 5$ , that is $\gamma = 0 . 9 5$ , and $s _ { 0 } = c = 1$ . The test of Section 5 uses 10 sets of 512 points drawn at random in the ball of radius 1, in dimension 1, the same for every number of terms. The estimate is the largest contraction threshold over these sets. The full chain is computed with 2500 terms. We compute $R _ { K } ^ { ( N ) } ( t )$ with $N = 5 0 0 0$ , for every number of terms and squared distances up to 4. The test is run with every number of terms up to 40, then with 50, 60, 80, 100, 150, 200, 300, 400, 500, 700, 1000 and 1500 terms.
<table><tr><td></td><td> $\tau = 0 . 9 9 2 7 5 \quad \tau = 0 . 9 1 1 5 2 5$ </td><td></td></tr><tr><td>Estimated smallest contraction factor of the full chain</td><td>0.964</td><td>0.910</td></tr><tr><td>Terms needed to pass the test</td><td>13</td><td>12</td></tr><tr><td>Terms needed to be within  $1 0 ^ { - 3 }$  of the full chain</td><td>60</td><td>300</td></tr><tr><td>Terms needed for  $R _ { K } ^ { ( N ) } ( t ) \leq 1 0 ^ { - 3 }$  on the ball</td><td>80</td><td>706</td></tr></table>

Numerical estimation. To use the geometric chaining construction as an MMD kernel, the infinite series has to be approximated numerically and therefore truncated. Consider the Bernstein function of the chained RBF kernel

$$
\phi ( t ) = \sum _ { n \geq 0 } c \tau ^ { - n } \bigl ( 1 - \exp ( - s _ { 0 } \rho ^ { n } t ) \bigr ) , \qquad 0 < \rho < \tau < 1 ,
$$

and write its n-th term as

$$
\phi _ { n } ( t ) : = c \tau ^ { - n } \bigl ( 1 - \exp ( - s _ { 0 } \rho ^ { n } t ) \bigr ) .
$$

For a truncation index K, we measure the relative contribution of the omitted terms by

$$
R _ { K } ( t ) : = \frac { \sum _ { n \geq K } \phi _ { n } ( t ) } { \sum _ { n \geq 0 } \phi _ { n } ( t ) } .
$$

In practice, we approximate this quantity by replacing the infinite denominator and numerator with a sufficiently large finite cutoff N,

$$
R _ { K } ^ { ( N ) } ( t ) : = \frac { \sum _ { K \leq n \leq N } c \tau ^ { - n } \big ( 1 - \exp ( - s _ { 0 } \rho ^ { n } t ) \big ) } { \sum _ { 0 \leq n \leq N } c \tau ^ { - n } \big ( 1 - \exp ( - s _ { 0 } \rho ^ { n } t ) \big ) } , \qquad N \gg 1 .
$$

Given a tolerance level $\alpha > 0$ , we define the truncation threshold

$$
K _ { \alpha } ( t ) : = \operatorname* { m i n } \left\{ K \in \mathbb { N } \mid R _ { K } ^ { ( N ) } ( t ) \leq \alpha \right\} .
$$

In the numerical results below, we use $\alpha = 1 0 ^ { - 3 }$ and report the corresponding threshold $K _ { 1 0 ^ { - 3 } } ( t )$

As shown in Figure 14, the terms of the chain decay rapidly. For each fixed t, there is a finite index $K _ { 1 0 ^ { - 3 } } ( t )$ beyond which the omitted terms account for less than 0.1% of the approximated radial profile. Figure 15 shows the same diagnostic when $\rho / \tau$ is closer to one, where the share of the omitted terms decreases more slowly as $K$ grows and a larger truncation index is needed to reach the same 0.1% threshold.

Contraction of truncated chains. A truncated chain is a finite sum of RBF kernels, so by Proposition 4.3 it does not induce contraction on all of $\mathbb { R } ^ { d }$ . As discussed in Section 5, it can still induce contraction on a bounded set, which we check with the test detailed in Appendix D for the two chains of Figures 14 and 15 (Table 9). About a dozen terms are enough for the truncated chain to pass the test. With more terms, its estimated smallest contraction factor on the ball decreases towards that of the full chain. This limit is below τ , since contraction on a ball is a weaker requirement than on all of $\mathbb { R } ^ { d }$ . Truncating with the rule above, which makes the profile accurate to $1 0 ^ { - 3 }$ on the ball, already brings the estimated contraction factor of the truncated chain within $1 0 ^ { - 3 }$ of that of the full chain. Both take many more terms when $\rho / \tau$ is closer to one.

## H Contraction rate and discriminability

Preamble. A kernel with a smaller contraction factor is not always the better choice, because changing the kernel also changes which differences between two distributions the MMD can detect. This section studies this tension, which Section 7 introduces. It starts with two results on the kernels $k ( x , y ) = C - b \| x - y \| ^ { 2 }$ , whose Bernstein function $\phi ( t ) = b t$ has a zero Lévy measure. First, Proposition H.1 shows that such a kernel only compares the means of two distributions. Second, Proposition H.2 shows that only these kernels can reach the smallest possible contraction factor $\gamma ^ { 2 }$ The rest of the section describes the experiments. Section H.1 presents the permutation test we use to measure the power of a kernel, that ${ \mathrm { i s } } ,$ how often it detects that two distributions differ. Section H.2 then explains how we estimate the tradeoff between this power and the contraction factor. Finally, Section H.3 repeats these experiments in higher dimensions.

Proposition H.1 (The kernel $C - b \| x - y \| ^ { 2 }$ compares only means). Let $b \geq 0$ and $C \in \mathbb { R }$ , and consider the kernel

$$
k ( x , y ) : = C - b \| x - y \| ^ { 2 } , \qquad x , y \in \mathbb { R } ^ { d } .
$$

Empirical level. Given samples $x _ { 1 : n } = ( x _ { 1 } , \ldots , x _ { n } )$ and $y _ { 1 : m } = ( y _ { 1 } , \dots , y _ { m } )$ , define the empirical means

$$
\bar { x } _ { n } : = \frac 1 n \sum _ { i = 1 } ^ { n } x _ { i } , \qquad \bar { y } _ { m } : = \frac 1 m \sum _ { j = 1 } ^ { m } y _ { j } .
$$

Then the biased empirical estimator satisfies

$$
\begin{array} { r } { \biggr \vert \widehat { \mathbf { M M D } } _ { k , \mathrm { b } } ^ { 2 } ( x _ { 1 : n } , y _ { 1 : m } ) = 2 b \Vert \bar { x } _ { n } - \bar { y } _ { m } \Vert ^ { 2 } \biggr \vert . } \end{array}
$$

In particular, the biased empirical $\mathbf { M M D } ^ { 2 }$ depends only on the empirical means of the two samples. Population level. Similarly, at the population level, let $P , Q \in { \mathcal { P } } ( \mathbb { R } ^ { d } )$ have finite second moments. Then

$$
\boxed { \mathbf { M M D } _ { k } ^ { 2 } ( P , Q ) = 2 b \| \mathbb { E } _ { P } [ X ] - \mathbb { E } _ { Q } [ X ] \| ^ { 2 } } .
$$

In particular, $\mathbf { M } \mathbf { M } \mathbf { D } _ { k } ^ { 2 } ( P , Q ) = 0$ whenever P and Q have the same first moment.

Proof. Empirical identity. Let $N : = n + m$ , define the aggregate sample

$$
z _ { i } : = x _ { i } \quad ( 1 \leq i \leq n ) , \qquad z _ { n + j } : = y _ { j } \quad ( 1 \leq j \leq m ) ,
$$

and let

$$
\ell : = \left( \begin{array} { l } { \frac { 1 } { n } \mathbf { 1 } _ { n } } \\ { - \frac { 1 } { m } \mathbf { 1 } _ { m } } \end{array} \right) \in \mathbb { R } ^ { N } , \qquad \mathrm { s o ~ t h a t ~ } \mathbf { 1 } _ { N } ^ { \top } \ell = 0 .
$$

Let $K \in \mathbb { R } ^ { N \times N }$ be the Gram matrix of the aggregate sample, $K _ { i j } : = k ( z _ { i } , z _ { j } )$ . By Section $\mathrm { A . } 3 .$

$$
\widehat { \mathbf { M M D } } _ { k , \mathrm { b } } ^ { 2 } ( x _ { 1 : n } , y _ { 1 : m } ) = \ell ^ { \top } K \ell .
$$

Using $k ( z _ { i } , z _ { j } ) = C - b \| z _ { i } - z _ { j } \| ^ { 2 }$ , we obtain

$$
\ell ^ { \top } K \ell = C ( \mathbf { 1 } _ { N } ^ { \top } \ell ) ^ { 2 } - b \sum _ { i , j = 1 } ^ { N } \ell _ { i } \ell _ { j } \| z _ { i } - z _ { j } \| ^ { 2 } = - b \sum _ { i , j = 1 } ^ { N } \ell _ { i } \ell _ { j } \| z _ { i } - z _ { j } \| ^ { 2 } ,
$$

since $\mathbf { 1 } _ { N } ^ { \top } \boldsymbol { \ell } = 0$ . Applying Lemma I.1 then yields

$$
\widehat { \mathbf { M M D } } _ { k , \mathrm { b } } ^ { 2 } ( x _ { 1 : n } , y _ { 1 : m } ) = 2 b \Big \| \sum _ { i = 1 } ^ { N } \ell _ { i } z _ { i } \Big \| ^ { 2 } .
$$

Finally,

$$
\sum _ { i = 1 } ^ { N } \ell _ { i } z _ { i } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } x _ { i } - \frac { 1 } { m } \sum _ { j = 1 } ^ { m } y _ { j } = \bar { x } _ { n } - \bar { y } _ { m } ,
$$

which gives the result.

Population identity. Let $\sigma : = P - Q$ , so that $\sigma ( \mathbb { R } ^ { d } ) = 0$ . By the definition of $ { \mathbf { M M D } } _ { k } ^ { 2 }$ as a quadratic form,

$$
\mathbf { M M D } _ { k } ^ { 2 } ( P , Q ) = \int \int _ { \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } } k ( x , y ) d \sigma ( x ) d \sigma ( y ) .
$$

Expanding the kernel gives

$$
k ( x , y ) = C - b \| x \| ^ { 2 } - b \| y \| ^ { 2 } + 2 b x ^ { \top } y .
$$

Since $\sigma ( \mathbb { R } ^ { d } ) = 0$ , the constant term and the terms that depend on only one of x and y vanish in the quadratic form,

$$
\iint C d \sigma ( \boldsymbol { x } ) d \sigma ( \boldsymbol { y } ) = C \sigma ( \mathbb { R } ^ { d } ) ^ { 2 } = 0 ,
$$

and

$$
\iint \| x \| ^ { 2 } d \sigma ( x ) d \sigma ( y ) = \Big ( \int \| x \| ^ { 2 } d \sigma ( x ) \Big ) \sigma ( \mathbb { R } ^ { d } ) = 0 ,
$$

with the same identity for the $\| y \| ^ { 2 }$ term. Therefore,

$$
\mathbf { M M D } _ { k } ^ { 2 } ( P , Q ) = 2 b \int \int x ^ { \top } y d \sigma ( x ) d \sigma ( y ) = 2 b \Big \| \int x d \sigma ( x ) \Big \| ^ { 2 } .
$$

Since $\begin{array} { r } { \int x d \sigma ( x ) = \mathbb { E } _ { P } [ X ] - \mathbb { E } _ { Q } [ X ] } \end{array}$ , the claim follows.

Proposition H.2 (Only the kernels $C - b \| x - y \| ^ { 2 }$ reach the factor $\gamma ^ { 2 } )$ . Let $k ( x , y ) = C - \phi ( \| x - y \| ^ { 2 } )$ for $x , y \in \mathbb { R } ^ { d }$ , where $\phi$ is a Bernstein function with Lévy measure Λ. If $\Lambda \not \equiv 0 ,$ then $\tau _ { k } ^ { \star } ( \mathcal { T } ^ { \pi } ) > \gamma ^ { 2 } .$ Equivalently, the value $\gamma ^ { 2 }$ can only be reached by a kernel of the form $k ( x , y ) = C - \ddot { b } \| x - y \| ^ { 2 }$

Proof.

A lower bound from two point masses. We first derive a lower bound on every contraction factor. The second part then shows that this bound exceeds $\gamma ^ { 2 }$ when $\Lambda \neq 0$

A constant in $\phi$ only shifts $C ,$ so we take $\phi ( 0 ) ~ = ~ 0$ . For two point masses, this gives $\begin{array} { r } { \mathbf { M } \mathbf { M } \mathbf { D } _ { k } ^ { 2 } ( \delta _ { a } , \delta _ { b } ) = 2 \phi ( \| a - b \| ^ { 2 } ) } \end{array}$ . Take the one-state MDP of Lemma C.2 with reward 0, whose Bellman operator multiplies every return by γ. It has a single state-action pair, so the supremum in $\overline { { \mathbf { M M D } } } _ { k } ^ { 2 }$ is over that pair alone.

Fix $x \in \mathbb { R } ^ { d }$ with $x \neq 0$ , let $t = \| x \| ^ { 2 }$ , and let $\eta _ { 1 }$ and $\eta _ { 2 }$ put the point masses $\delta _ { x }$ and $\delta _ { 0 }$ on that pair. Before a Bellman step,

$$
\overline { { \mathbf { M M D } } } _ { k } ^ { 2 } ( \eta _ { 1 } , \eta _ { 2 } ) = \mathbf { M M D } _ { k } ^ { 2 } ( \delta _ { x } , \delta _ { 0 } ) = 2 \phi ( t ) .
$$

After the step, $\tau ^ { \pi } \eta _ { 1 }$ and $\mathcal { T } ^ { \pi } \eta _ { 2 }$ put $\delta _ { \gamma x }$ and $\delta _ { 0 }$ on the pair, so

$$
\overline { { \mathbf { M M D } } } _ { k } ^ { 2 } ( \mathcal T ^ { \pi } \eta _ { 1 } , \mathcal T ^ { \pi } \eta _ { 2 } ) = \mathbf { M M D } _ { k } ^ { 2 } ( \delta _ { \gamma x } , \delta _ { 0 } ) = 2 \phi ( \gamma ^ { 2 } t ) .
$$

If $\tau ^ { \pi }$ is a τ -contraction for this MDP, the definition of contraction in Section 2 applied to $\eta _ { 1 }$ and $\eta _ { 2 }$ gives

$$
2 \phi ( \gamma ^ { 2 } t ) = \overline { { \mathbf { M } \mathbf { M } \mathbf { D } } } _ { k } ^ { 2 } ( \mathcal { T } ^ { \pi } \eta _ { 1 } , \mathcal { T } ^ { \pi } \eta _ { 2 } ) \leq \tau \overline { { \mathbf { M } \mathbf { M } \mathbf { D } } } _ { k } ^ { 2 } ( \eta _ { 1 } , \eta _ { 2 } ) = 2 \tau \phi ( t ) ,
$$

so $\phi ( \gamma ^ { 2 } t ) \leq \tau \phi ( t )$ . The factor $\tau _ { k } ^ { \star } ( \mathcal { T } ^ { \pi } )$ is the smallest one valid for every MDP, so it is valid for this one as well. Hence, whenever $\phi ( t ) > 0$

$$
\tau _ { k } ^ { \star } ( T ^ { \pi } ) \geq \frac { \phi ( \gamma ^ { 2 } t ) } { \phi ( t ) } .
$$

The bound exceeds $\gamma ^ { 2 }$ when $\Lambda \neq 0 .$ In the Lévy–Khintchine representation of $\phi ,$ the terms in b cancel,

$$
\begin{array} { l } { { \phi ( \gamma ^ { 2 } t ) - \gamma ^ { 2 } \phi ( t ) = \left( b \gamma ^ { 2 } t - \gamma ^ { 2 } b t \right) + \displaystyle \int _ { ( 0 , \infty ) } \left( 1 - e ^ { - \gamma ^ { 2 } s t } \right) \Lambda ( d s ) - \gamma ^ { 2 } \int _ { ( 0 , \infty ) } \left( 1 - e ^ { - s t } \right) \Lambda ( d s ) } } \\ { { \displaystyle \qquad = \int _ { ( 0 , \infty ) } \left[ \left( 1 - e ^ { - \gamma ^ { 2 } s t } \right) - \gamma ^ { 2 } \left( 1 - e ^ { - s t } \right) \right] \Lambda ( d s ) . } } \end{array}
$$

The bracket has the form $g ( \gamma ^ { 2 } u ) - \gamma ^ { 2 } g ( u )$ with $g ( u ) : = 1 - e ^ { - u }$ and $u = s t .$ , and we show that it is positive for every $u > 0$ . First, g is strictly concave on $[ 0 , \infty )$ , since $g ^ { \prime \prime } ( u ) = - e ^ { - u } < 0$ , and $g ( 0 ) = 0$ . Second, since $\gamma ^ { 2 } \in ( 0 , 1 )$ , the point $\gamma ^ { 2 } u$ lies between 0 and u, as the convex combination $\bar { \gamma } ^ { 2 } \dot { u } = \gamma ^ { 2 } \cdot u + ( 1 - \gamma ^ { 2 } ) \cdot 0$ . Strict concavity at this point gives

$$
g ( \gamma ^ { 2 } u ) = g \big ( \gamma ^ { 2 } u + ( 1 - \gamma ^ { 2 } ) \cdot 0 \big ) > \gamma ^ { 2 } g ( u ) + ( 1 - \gamma ^ { 2 } ) g ( 0 ) = \gamma ^ { 2 } g ( u ) .
$$

So the bracket is positive for every $s > 0 ,$ . When $\Lambda \neq 0$ , the integral of this positive function against Λ is positive, and $\begin{array} { r } { \phi ( t ) \geq \int _ { ( 0 , \infty ) } \dot { g ( s t ) } \Lambda ( d s ) > 0 } \end{array}$ . Hence

$$
\tau _ { k } ^ { \star } ( \mathcal { T } ^ { \pi } ) \geq \frac { \phi ( \gamma ^ { 2 } t ) } { \phi ( t ) } > \gamma ^ { 2 } .
$$

Since $\tau ^ { \star } ( \gamma ) \geq \tau _ { k } ^ { \star } ( T ^ { \pi } )$ , as noted after Theorem E.4, the smallest contraction factor across dimensions also exceeds $\gamma ^ { 2 }$

When $\Lambda \equiv 0 .$ Then $\phi ( t ) = b t$ , and k is the kernel

$$
k ( x , y ) = C - b \| x - y \| ^ { 2 } .
$$

For this kernel, $\mathbf { M M D } _ { k }$ only compares means (Proposition H.1).

With $\Lambda \equiv 0 .$ , the criterion of Theorem E.4 reduces to $b ( \tau - \gamma ^ { 2 } ) \geq 0$ , which holds for every $\tau \in [ \gamma ^ { 2 } , 1 )$ Contraction then holds across dimensions, so

$$
\tau _ { k } ^ { \star } ( \mathcal { T } ^ { \pi } ) \leq \tau ^ { \star } ( \gamma ) \leq \gamma ^ { 2 } .
$$

Conversely, when $b > 0$ , the lower bound of the first part applies with $\phi ( t ) = b t > 0$ , and its ratio is exactly

$$
\frac { \phi ( \gamma ^ { 2 } t ) } { \phi ( t ) } = \frac { b \gamma ^ { 2 } t } { b t } = \gamma ^ { 2 } .
$$

Hence $\tau _ { k } ^ { \star } ( \mathcal { T } ^ { \pi } ) = \gamma ^ { 2 }$ , and this kernel reaches the value $\gamma ^ { 2 }$ . When $b = 0$ as well, the kernel is constant and $\mathbf { M M D } _ { k }$ is identically zero. □

## H.1 Permutation test and power estimation

$\mathbf { M M D ^ { 2 } }$ values are not comparable across kernels. For example, if $\boldsymbol { k } ^ { \prime } = c k$ with $c > 0$ , then $\mathbf { M } \mathbf { M } \mathbf { D } _ { k ^ { \prime } } ^ { 2 } = c \mathbf { M } \mathbf { M } \mathbf { D } _ { k } ^ { 2 }$ , so a larger value can reflect only a change of units. Following Borgwardt et al. [7], Gretton et al. $[ 1 5 , 1 6 ]$ , we therefore compare kernels through the power of a level-α permutation test built from the biased empirical statistic.

Level and power of a test. A hypothesis test of $H _ { 0 } \colon P = Q$ against $H _ { 1 } \colon P \neq Q$ is a binary rule $\Delta _ { k } \in \{ 0 , 1 \}$ , with $\Delta _ { k } = 1$ meaning rejection. It has level α if sup $) _ { P } \mathbb { P } _ { P \times P } ( \Delta _ { k } = 1 ) \le \alpha$ controlling the false-positive rate. Its power against $( P , Q )$ with $P \neq \bar { Q } \ \mathrm { i s } \ \mathbb { P } _ { P \times \bar { Q } } ( \Delta _ { k } = 1 )$ , the probability of correctly detecting that $\bar { P } \neq Q$

Test statistic. Given samples $X _ { 1 : n } \sim P ^ { n }$ and $Y _ { 1 : m } \sim Q ^ { m }$ , let $N : = n + m$ , form the aggregate sample $Z _ { 1 : N } = ( X _ { 1 : n } , Y _ { 1 : m } )$ , and let $K \in \mathbb { R } ^ { N \times N }$ be its Gram matrix, $K _ { i j } : = k ( Z _ { i } , Z _ { j } )$ ). Define the two-sample weight vector $\boldsymbol { \ell } \in \mathbb { R } ^ { N }$ as in Section A.3,

$$
\ell _ { i } : = \left\{ { \begin{array} { l l } { \frac { 1 } { n } , } & { 1 \leq i \leq n , } \\ { - \frac { 1 } { m } , } & { n + 1 \leq i \leq N , } \end{array} } \right.
$$

so that the test statistic is

$$
\begin{array} { r } { T _ { k } : = \ell ^ { \top } K \ell = \widehat { \mathbf { M M D } } _ { k , \mathrm { b } } ^ { 2 } ( X _ { 1 : n } , Y _ { 1 : m } ) . } \end{array}
$$

Null hypothesis and permutation test. Under $H _ { 0 } ,$ , the labels are exchangeable, so permuting $\{ 1 , \ldots , \bar { N } \}$ leaves the joint distribution of the aggregate sample unchanged. Given $B _ { \mathrm { p e r m } }$ uniformly drawn permutations $\pi _ { 1 } , \ldots , \pi _ { B _ { \mathrm { p e r m } } }$ of $\{ 1 , \ldots , \bar { N } \}$ , each induces a permuted statistic $T _ { k , \pi _ { b } } : =$ $\ell ^ { \top } K _ { \pi _ { b } } \ell ,$ , where $( K _ { \pi _ { b } } ) _ { i j } : = k ( Z _ { \pi _ { b } ( i ) } , Z _ { \pi _ { b } ( j ) } )$ . The permutation p-value is

$$
\widehat { p } : = \frac { 1 + \# \big \{ b \in \{ 1 , \dots , B _ { \mathrm { p e r m } } \} : T _ { k , \pi _ { b } } \geq T _ { k } \big \} } { B _ { \mathrm { p e r m } } + 1 } ,
$$

Checkerboard: one realization $\widehat { { \sf M M D } _ { \sf R B F } ^ { 2 } } \bullet$ null by label permutation   
3   
40   
2   
1 30   
P sample Density   
x<sup>2</sup> 0 Q sample   
20   
−1   
10   
−2 permutation cutoff   
observed statistic   
0   
−3 −2 −1 0 1 2 3 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7   
x<sub>1</sub> Biased MMD2̂ statistic  
Figure 16: One example of the permutation test used for power estimation, using the RBF kernel with lengthscale $\ell = \bar { 1 } . 0 5$ at level $\alpha = 0 . 0 5$ . Left, one realization of the checkerboard two-sample problem $( n = m = 7 0 )$ . Right, the permutation null for the biased $\widehat { \mathbf { M M D } } _ { \mathrm { R B F } } ^ { 2 }$ statistic, obtained from 2,000 label permutations on the aggregate sample. The dashed line is the level-α cutoff and the red line is the observed statistic.

```latex
Algorithm 3 Estimating the power of the permutation test for one kernel configuration
1: Fix $( P , Q )$ , sample sizes $( n , m )$ , a level α, an outer repetition count $B ,$ and a permutation budget
$B _ { \mathrm { p e r m } } .$
2: for $b = 1 , \dots , B$ do
3: Draw $X _ { 1 : n } ^ { ( b ) } \sim P ^ { n }$ and $Y _ { 1 : m } ^ { ( b ) } \sim Q ^ { m } .$
4: Build the Gram matrix $K ^ { ( b ) }$ of the aggregate sample and compute $T _ { k } ^ { ( b ) } = \widehat { \mathbf { M M D } } _ { k , \mathrm { b } } ^ { 2 } .$
5: Permute the labels $B _ { \mathrm { p e r m } }$ times on the aggregate sample and recompute the corresponding
statistics $T _ { k , \pi } ^ { ( b ) }$
6: Compute the permutation p-value
$\hat { p } ^ { ( b ) } = \frac { 1 + \# \{ T _ { k , \pi } ^ { ( b ) } \geq T _ { k } ^ { ( b ) } \} } { B _ { \mathrm { p e r m } } + 1 } .$
7: end for
8: Return $\begin{array} { r } { \widehat { D } _ { k , n , m , \alpha } ( P , Q ) : = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \mathbf { 1 } \big [ \widehat { p } ^ { ( b ) } \leq \alpha \big ] } \end{array}$
```

and $H _ { 0 }$ is rejected if and only $\mathrm { i f } \hat { p } \leq \alpha .$

Figure 16 illustrates one instance of this test. The aggregate sample is fixed, the labels are permuted to form the null distribution, and the observed statistic is compared with the resulting level-α cutoff.

Power estimation. The power of the level-α test for a given kernel k is $\mathbb { P } _ { H _ { 1 } } ( \boldsymbol { \hat { p } } \leq \alpha )$ . It is estimated by repeating the following procedure B times independently. Each run draws a fresh dataset pair $( \bar { X } _ { 1 : n } ^ { ( b ) } , Y _ { 1 : m } ^ { ( b ) } )$ , computes the observed statistic $T _ { k } ^ { ( b ) }$ , runs $B _ { \mathrm { p e r m } }$ permutations to obtain the p-value $\hat { p } ^ { ( b ) }$ , and records whether $\hat { p } ^ { ( b ) } \leq \alpha ,$ . The empirical power $\widehat { D } _ { k , n , m , \alpha } ( P , Q )$ is the fraction of rejections across the B repetitions. Algorithm 3 summarizes this procedure.

## H.2 Estimating the tradeoff between power and contraction

For a given distribution pair $( P , Q )$ and kernel family $\{ k _ { \theta } \}$ , lowering the smallest contraction factor on a bounded set and maximizing discriminative power pull in opposite directions. To make this tradeoff visible, we fix a contraction budget τ¯ and report, over all θ with $\widehat { \tau } _ { k _ { \theta } , E } ^ { \star } ( \gamma ) \leq \bar { \tau }$ , the best

```latex
Algorithm 4 Measuring the tradeoff between power and contraction
1: Require: Distribution pair $( P , Q )$ , sample sizes $( n , m )$ , level α, discount grid $\{ \gamma _ { l } \}$ , grid of
contraction budgets τ¯, parameter grid $\{ \bar { \theta _ { i } } \}$ , ball radius R, support size N, trial count J, outer
repetition count B, permutation budget $B _ { \mathrm { p e r m } } .$
2: Step 1: estimate powerfor each kernel parameter.
3: for each $\theta _ { i }$ do
4: Run Algorithm 3 with kernel $k _ { \theta _ { i } }$ . Set $\widehat { D } _ { i } : = \widehat { D } _ { k _ { \theta _ { i } } , n , m , \alpha } ( P , Q )$
5: end for
6: Step 2: estimate the smallest contractionfactor on E for each kernel parameter and discount.
7: for each $\theta _ { i }$ and each $\gamma _ { l }$ do
8: Run Algorithm 1 with kernel $k _ { \theta _ { i } }$ , scaling parameter $\lambda = \gamma _ { l } .$ , ball $B ( 0 , R )$ , support size $N ,$ and
J trials. Set $\widehat { \tau } _ { i , l } : = \operatorname* { m a x } _ { 1 \leq j \leq J } \widehat { \tau } _ { S _ { j } } ^ { \star } ( \gamma _ { l } )$
9: end for
10: Step 3: build the tradeoffheatmap.
11: for each $\gamma _ { l }$ and each budget $\bar { \tau }$ do
12: Record ma $\mathbf { x } _ { i } : \widehat { \tau } _ { i , l } \leq \bar { \tau } \widehat { D } _ { i } .$ , or mark the cell as empty if no $\theta _ { i }$ satisfies the constraint.
13: end for
```

<table><tr><td>Setting</td><td>Value</td><td>Description</td></tr><tr><td>Support size N</td><td>512</td><td>Number of points drawn uniformly in  $B ( 0 , R )$  for each trial support.</td></tr><tr><td>Trial count J</td><td>1</td><td>Number of independent trial supports used to estimate  $\boldsymbol { \tau } _ { k _ { \theta } , E } ^ { \star } ( \gamma )$ </td></tr><tr><td>Eigenvalue tolerance  $\varepsilon _ { \mathrm { e i g } }$ </td><td> $1 0 ^ { - 9 }$ </td><td>Relative tolerance below which eigenvalues of A are dis- carded before forming the reduced problem.</td></tr><tr><td>Outer repetitions B</td><td>1000</td><td>Number of independent dataset draws used to estimate power.</td></tr><tr><td>Permutation budget  $B _ { \mathrm { p e r m } }$ </td><td>1000</td><td>Number of label permutations used to compute the p-value on each dataset draw.</td></tr><tr><td>Level α</td><td>0.05</td><td>Significance level of the permutation test.</td></tr></table>

Table 10: Numerical settings used in Algorithm ${ \overline { { 4 . } } }$

achievable power

$$
\widehat { D } ^ { \star } ( \gamma , \bar { \tau } ) : = \operatorname * { m a x } \bigl \{ \widehat { D } _ { k _ { \theta } , n , m , \alpha } ( P , Q ) : \widehat { \tau } _ { k _ { \theta } , E } ^ { \star } ( \gamma ) \leq \bar { \tau } \bigr \} ,
$$

leaving cells with no feasible θ undefined. Sweeping over $( \gamma , { \bar { \tau } } )$ produces the tradeoff heatmap of Algorithm 4.

Estimation procedure. For each kernel family and parameter grid in Table 11, and for each $\gamma _ { : }$ the smallest contraction factor $\tau _ { k _ { \theta } , E } ^ { \star } ( \gamma )$ on the ball $E \doteq B ( 0 , R )$ is estimated by Algorithm 1. For each dataset in Table 12, R is chosen to contain most of the mass and is used to clip the sampled data. The numerical settings are given in Table 10. In particular, we use support size N and J independent random supports, retaining the maximum over the J estimates. For each $\theta ,$ the power $\widehat { D } _ { k _ { \theta } , n , m , \alpha } ( P , Q )$ is estimated by Algorithm 3. The full procedure is summarized in Algorithm 4.

<table><tr><td>Dataset</td><td>Distribution</td><td>Radius R</td></tr><tr><td>1D Mean Shift</td><td> $P = \mathcal { N } ( - 0 . 4 0 , 0 . 1 8 ^ { 2 } ) ,$   $Q = \mathcal { N } ( 0 . 4 0 , 0 . 1 8 ^ { 2 } )$ </td><td>1.40</td></tr><tr><td>1D Same Mean, Different Std</td><td> $P = \mathcal { N } ( 0 , 0 . 2 5 ^ { 2 } ) ,$   $Q = \mathcal { N } ( 0 , 0 . 5 5 ^ { 2 } )$ </td><td>2.00</td></tr><tr><td>1D Bimodal vs Unimodal</td><td> $\begin{array} { r } { P = \frac { 1 } { 2 } \mathcal { N } ( - 0 . 6 5 , 0 . 1 5 ^ { 2 } ) + \frac { 1 } { 2 } \mathcal { N } ( 0 . 6 5 , 0 . 1 5 ^ { 2 } ) , } \end{array}$   $Q = \mathcal { N } ( 0 , 0 . 6 5 ^ { 2 } + 0 . 1 5 ^ { 2 } )$ </td><td>2.00</td></tr><tr><td>Two-Gaussian Location</td><td> $P = \mathcal { N } \big ( ( 0 . 5 5 , 0 ) , 0 . 1 8 ^ { 2 } I _ { 2 } \big )$  2  $Q = \mathcal { N } \big ( ( - 0 . 5 5 , 0 ) , 0 . 1 8 ^ { 2 } I _ { 2 } \big )$ </td><td>1.35</td></tr><tr><td>Covariance Swap</td><td> $P = \mathcal { N } \bigg ( 0 , \bigg [ 0 . 2 8 \quad 0 . 1 8 \bigg ] \bigg )$   $\begin{array} { r l } { Q = \mathcal { N } \bigg ( 0 , \bigg [ 0 . 2 8 ~ } & { { } - 0 . 1 8 \bigg ] \bigg ) } \\ { - 0 . 1 8 ~ } & { { } 0 . 2 8 } \end{array}$ </td><td>2.40</td></tr><tr><td>Bimodal vs Unimodal</td><td> $\begin{array} { r } { P = \frac { 1 } { 2 } \mathcal { N } \big ( ( - 0 . 7 5 , 0 ) , 0 . 1 8 ^ { 2 } I _ { 2 } \big ) + \frac { 1 } { 2 } \mathcal { N } \big ( ( 0 . 7 5 , 0 ) , 0 . 1 8 ^ { 2 } I _ { 2 } \big ) } \end{array}$   $Q = { \mathcal { N } } { \left( 0 , \left[ \begin{array} { c c } { 0 . 7 5 ^ { 2 } + 0 . 1 8 ^ { 2 } } & { 0 } \\ { 0 } & { 0 . 1 8 ^ { 2 } } \end{array} \right] \right) }$ </td><td>， 2.40</td></tr><tr><td>Checkerboard</td><td> $\begin{array} { r } { P = \frac { 1 } { 2 } \mathcal { N } \big ( ( - 2 . 2 , - 1 . 7 ) , 0 . 3 5 ^ { 2 } I _ { 2 } \big ) + \frac { 1 } { 2 } \mathcal { N } \big ( ( 2 . 2 , 1 . 7 ) , 0 . 3 5 ^ { 2 } I _ { 2 } \big ) } \end{array}$   $\begin{array} { r } { Q = \frac { 1 } { 2 } \mathcal { N } \big ( ( - 2 . 0 , 1 . 9 ) , 0 . 3 5 ^ { 2 } I _ { 2 } \big ) + \frac { 1 } { 2 } \mathcal { N } \big ( ( 2 . 0 , - 1 . 9 ) , 0 . 3 5 ^ { 2 } I _ { 2 } \big ) } \end{array}$ </td><td>4.40</td></tr></table>

Table 12: Datasets used to study the tradeoff between power and contraction. The displayed Gaussian laws are sampled and then clipped to the ball $B ( 0 , R )$

<table><tr><td>Kernel family and formula</td><td>Parameter θ</td><td>Grid over Θ</td></tr><tr><td>RBF  $k _ { h } ( x , y ) = \exp { \left( - h ^ { 2 } \| x - y \| ^ { 2 } \right) }$ </td><td>Scale h</td><td> $\mathrm { g e o m } ( 1 0 ^ { - 3 } , 1 0 ^ { 3 } , 1 0 0 )$ </td></tr><tr><td>Generalized multiquadric  $k _ { \beta , h } ( x , y ) = - \big ( \dot { 1 + h ^ { 2 } } \| x - y \| ^ { 2 } \big ) ^ { \beta }$ </td><td>Power β</td><td>lin(0.05, 1.00, 50)</td></tr><tr><td>Power distance  $k _ { \alpha } ( x , y ) = - \| x - y \| ^ { \alpha }$ </td><td>Power α</td><td>lin(0.20, 2.00, 50)</td></tr><tr><td>Logarithmic kernel  $\bar { k _ { a } ( x , y ) } = - \log \bigl ( 1 + \| x - y \| ^ { 2 } / a \bigr )$ </td><td>Scale a</td><td> $\mathrm { g e o m } ( 1 0 ^ { - 3 } , 1 0 ^ { 3 } , 1 0 0 )$ </td></tr></table>

Table 11: Kernel families and the values of their parameter used to study the tradeoff between power and contraction. Here geom(a, b, n) denotes n geometrically spaced values from a to b, and lin(a, b, n) denotes n linearly spaced values from a to b.

![](images/6eb5b84d8075f531f65ebc219ad9756ccc8b203a21760cba7993557228fc33a1.jpg)  
Figure 17: Two-dimensional tradeoff experiments. The left column shows one sampled dataset for each task, with dashed circles marking the clipping radius R. The remaining columns show, for each kernel family, the best empirical test power achievable under the constraint $\bar { \tau } _ { k _ { \theta } , E } ^ { \star } ( \gamma ) \leq \bar { \tau }$ . The dotted curve is the smallest possible contraction factor $\gamma ^ { 2 }$

## H.3 Higher dimensions

Return distributions are multivariate when rewards are vector valued [13, 37, 38], so the two samples compared by the test can live in higher dimensions. We therefore check whether the tradeoff persists as the dimension grows, on two pairs of distributions defined in any dimension, a mean shift and a checkerboard on the corners of a cube (Table 13). We use the power-distance kernels $k ( x , y ) = - \| x - y \| ^ { \alpha }$ , whose smallest contraction factor across dimensions is $\gamma ^ { \alpha }$ (Proposition 4.1). The constraint of Appendix H.2 is then $\gamma ^ { \alpha } \leq \bar { \tau }$ , with no estimation on a ball.

Figure 18 shows the results. For the mean shift, the power does not depend on α, so faster contraction costs no power in any dimension. For the checkerboard, the two distributions have the same mean. The kernels with larger α contract faster and are closer to comparing means, and they lose their power first. As the dimension grows, they lose it at smaller α, so the tradeoff becomes stronger. As the dimension grows, each distribution also spreads over $2 ^ { d - 1 }$ corners, many more than the 100 points of a sample. By dimension 16, no kernel detects the difference, so the tradeoff can no longer be observed.

<table><tr><td>Dataset</td><td>Distribution</td></tr><tr><td>Cube Checkerboard</td><td>P and Q uniform on the corners  $s / \sqrt { d } , s \in \{ - 1 , 1 \} ^ { d } .$  , of a cube, colored like a checkerboard, with the corners having an even number of —1 for  $P$  and those having an odd number for Q</td></tr><tr><td>Mean Shift</td><td> $P = \mathcal { N } ( - 0 . 4 u , I _ { d } )$  and  $Q = { \mathcal { N } } ( 0 . 4 u , I _ { d } ) .$  with  $u = ( 1 , \ldots , 1 ) / \sqrt { d }$ </td></tr><tr><td>Setting</td><td>Value Description</td></tr><tr><td>Dimension d</td><td>2,3, 4, 8, 16, 32, 64 Dimension of the samples.</td></tr><tr><td>Power α</td><td>lin(0.20, 2.00, 50) Parameter of  $k _ { \alpha } ( x , y ) = - \| x - y \| ^ { \alpha } .$ </td></tr><tr><td>Sample sizes  $( n , m )$ </td><td>(100,100) Number of points in each sample.</td></tr></table>

Table 13: Distributions and settings of the experiments in higher dimensions. The outer repetitions, the permutation budget and the level are those of Table 10.

![](images/ad7667568307f576b2a1db6db4847046a1e0dab320a34fc4a8f16a78a45d0648.jpg)  
Figure 18: Tradeoff experiments in higher dimensions for the power-distance kernels $k ( x , y ) =$ $- \| x - y \| ^ { \alpha }$ , with the settings of Table 13. Each column is a dimension $d .$ Each panel shows the best empirical test power achievable under the constraint $\gamma ^ { \alpha } \leq \bar { \tau }$ . The dotted curve is the smallest possible contraction factor $\gamma ^ { 2 }$

## I Properties of radial kernels built from Bernstein functions

Preamble. This section studies when the radial kernels built from Bernstein functions induce a metric through MMD, rather than only a pseudometric. For kernels of the form

$$
k ( x , y ) = C - \phi ( \| x - y \| ^ { 2 } ) ,
$$

the key issue is whether the Bernstein part $\phi ( \| x - y \| ^ { 2 } )$ is strictly conditionally negative definite. Lemma I.2 shows that this happens exactly when the Lévy measure Λ is nonzero. Theorem I.1 then combines this with the general criterion for MMD to be a metric and shows that ${ \bf M M D } _ { k }$ is a metric if and only if $\Lambda \neq 0$

Lemma I.1 (Quadratic forms of squared Euclidean distances). Let $x _ { 1 } , \ldots , x _ { n } \in \mathbb { R } ^ { d }$ and $w _ { 1 } , \ldots , w _ { n } \in \mathbb { R }$ satisfy $\textstyle \sum _ { i = 1 } ^ { n } w _ { i } = 0$ . Then

$$
\sum _ { i , j = 1 } ^ { n } w _ { i } w _ { j } \left\| x _ { i } - x _ { j } \right\| ^ { 2 } = - 2 \left\| \sum _ { i = 1 } ^ { n } w _ { i } x _ { i } \right\| ^ { 2 } \ \leq \ 0 .
$$

Moreover, the left-hand side equals 0 if and only $\textstyle { \mathrm { i f } } \sum _ { i = 1 } ^ { n } w _ { i } x _ { i } = 0$

Proof. Expand

$$
\| x _ { i } - x _ { j } \| ^ { 2 } = \| x _ { i } \| ^ { 2 } + \| x _ { j } \| ^ { 2 } - 2 \langle x _ { i } , x _ { j } \rangle .
$$

Hence

$$
\sum _ { i , j = 1 } ^ { n } w _ { i } w _ { j } \| x _ { i } - x _ { j } \| ^ { 2 } = \sum _ { i , j = 1 } ^ { n } w _ { i } w _ { j } \| x _ { i } \| ^ { 2 } + \sum _ { i , j = 1 } ^ { n } w _ { i } w _ { j } \| x _ { j } \| ^ { 2 } - 2 \sum _ { i , j = 1 } ^ { n } w _ { i } w _ { j } \langle x _ { i } , x _ { j } \rangle .
$$

Using $\begin{array} { r } { \sum _ { j = 1 } ^ { n } w _ { j } = 0 } \end{array}$

$$
\sum _ { i , j = 1 } ^ { n } w _ { i } w _ { j } \| x _ { i } \| ^ { 2 } = \sum _ { i = 1 } ^ { n } \biggl ( w _ { i } \| x _ { i } \| ^ { 2 } \sum _ { j = 1 } ^ { n } w _ { j } \biggr ) = 0 , \quad \sum _ { i , j = 1 } ^ { n } w _ { i } w _ { j } \| x _ { j } \| ^ { 2 } = \sum _ { j = 1 } ^ { n } \biggl ( w _ { j } \| x _ { j } \| ^ { 2 } \sum _ { i = 1 } ^ { n } w _ { i } \biggr ) = 0 .
$$

Therefore

$$
\sum _ { i , j = 1 } ^ { n } w _ { i } w _ { j } \| x _ { i } - x _ { j } \| ^ { 2 } = - 2 \sum _ { i , j = 1 } ^ { n } w _ { i } w _ { j } \big \langle x _ { i } , x _ { j } \big \rangle .
$$

Finally,

$$
\sum _ { i , j = 1 } ^ { n } w _ { i } w _ { j } \left. x _ { i } , x _ { j } \right. = \Big \langle \sum _ { i = 1 } ^ { n } w _ { i } x _ { i } , \sum _ { j = 1 } ^ { n } w _ { j } x _ { j } \Big \rangle = \Big \| \sum _ { i = 1 } ^ { n } w _ { i } x _ { i } \Big \| ^ { 2 } ,
$$

which proves the identity and the inequality. The last claim follows since $\begin{array} { r } { \left\| \sum _ { i = 1 } ^ { n } w _ { i } x _ { i } \right\| ^ { 2 } = 0 } \end{array}$ only when $\textstyle \sum _ { i = 1 } ^ { n } w _ { i } x _ { i } = 0$ □

Lemma I.2 (When a radial kernel built from a Bernstein function is CSND). Let ϕ : $[ 0 , \infty )  [ 0 , \infty )$ be a Bernstein function with $\phi ( 0 ) = 0$ , admitting the Lévy–Khintchine representation

$$
\phi ( t ) = b t + \int _ { ( 0 , \infty ) } \bigl ( 1 - e ^ { - t s } \bigr ) \Lambda ( d s ) ,
$$

where $b \geq 0$ and Λ is a nonnegative measure satisfying $\int _ { ( 0 , \infty ) } \operatorname* { m i n } \{ 1 , s \} \Lambda ( d s ) < \infty$ . Define the radial kernel

$$
k ( x , y ) \ = \ \phi { \bigl ( } \| x - y \| ^ { 2 } { \bigr ) }
$$

on $\mathbb { R } ^ { d }$ . Then k is conditionally strictly negative definite (CSND) on $\mathbb { R } ^ { d }$ if and only if Λ is not identically zero (equivalently, ϕ is not of the form $\phi ( t ) = b t )$ . Moreover, if $\Lambda \equiv 0 .$ , then k is CND but not CSND.

Proof. We split into two cases.

Case 1: Zero Lévy measure, $\Lambda \equiv 0 .$ Then $\phi ( t ) = b t$ and

$$
k ( x , y ) = b \| x - y \| ^ { 2 } .
$$

For any finite choice $x _ { 1 } , \ldots , x _ { n } \in \mathbb { R } ^ { d }$ and weights $w _ { 1 } , \ldots , w _ { n }$ with $\begin{array} { r } { \sum _ { i } w _ { i } = 0 } \end{array}$

$$
\sum _ { i , j = 1 } ^ { n } w _ { i } w _ { j } k ( x _ { i } , x _ { j } ) = b \sum _ { i , j = 1 } ^ { n } w _ { i } w _ { j } \| x _ { i } - x _ { j } \| ^ { 2 } = - 2 b \left\| \sum _ { i = 1 } ^ { n } w _ { i } x _ { i } \right\| ^ { 2 } \leq 0
$$

by Lemma I.1. Hence k is CND. Moreover, i $\textstyle \sum _ { i } w _ { i } x _ { i } = 0$ with $\textstyle \sum _ { i } w _ { i } = 0$ and w nonzero, then the quadratic form vanishes, so k is not CSND.

Case 2: Nonzero Lévy measure, $\Lambda \neq 0 .$ . From the Lévy–Khintchine form,

$$
k ( x , y ) = b \| x - y \| ^ { 2 } + \int _ { ( 0 , \infty ) } \bigl ( 1 - e ^ { - s \| x - y \| ^ { 2 } } \bigr ) \Lambda ( d s ) .
$$

Fix pairwise distinct $x _ { 1 } , \ldots , x _ { n } \in \mathbb { R } ^ { d }$ and weights $w _ { 1 } , \ldots , w _ { n }$ with $\textstyle \sum _ { i } w _ { i } = 0$ , not all zero, and set

$$
S : = \sum _ { i , j = 1 } ^ { n } w _ { i } w _ { j } k ( x _ { i } , x _ { j } ) .
$$

Drift part. By Lemma I.1,

$$
b \sum _ { i , j } w _ { i } w _ { j } \| x _ { i } - x _ { j } \| ^ { 2 } \ = \ - 2 b \Big \| \sum _ { i = 1 } ^ { n } w _ { i } x _ { i } \Big \| ^ { 2 } \ \leq \ 0 .
$$

Interchanging thefinite sum and the integral. For each $( i , j )$ and all $s > 0$ , set $u : = s \| x _ { i } - x _ { j } \| ^ { 2 } \geq 0$ Since

$$
1 - e ^ { - u } \leq 1 \qquad { \mathrm { a n d } } \qquad 1 - e ^ { - u } \leq u ,
$$

we have

$$
0 \leq 1 - e ^ { - s \| x _ { i } - x _ { j } \| ^ { 2 } } \leq \operatorname* { m i n } \{ 1 , ~ s \| x _ { i } - x _ { j } \| ^ { 2 } \} .
$$

Moreover,

$$
\operatorname* { m i n } \{ 1 , s \| x _ { i } - x _ { j } \| ^ { 2 } \} \leq \operatorname* { m a x } \{ 1 , \| x _ { i } - x _ { j } \| ^ { 2 } \} \operatorname* { m i n } \{ 1 , s \} ,
$$

so

$$
\int _ { ( 0 , \infty ) } \bigl | 1 - e ^ { - s \| x _ { i } - x _ { j } \| ^ { 2 } } \bigr | \Lambda ( d s ) \le \operatorname* { m a x } \{ 1 , \| x _ { i } - x _ { j } \| ^ { 2 } \} \int _ { ( 0 , \infty ) } \operatorname* { m i n } \{ 1 , s \} \Lambda ( d s ) < \infty .
$$

Hence each integral is finite, and since there are only finitely many pairs $( i , j )$ , we may exchange the finite sum and the integral,

$$
\sum _ { i , j } w _ { i } w _ { j } \int _ { ( 0 , \infty ) } \bigl ( 1 - e ^ { - s \| x _ { i } - x _ { j } \| ^ { 2 } } \bigr ) \Lambda ( d s ) = \int _ { ( 0 , \infty ) } \Bigl [ \sum _ { i , j } w _ { i } w _ { j } \bigl ( 1 - e ^ { - s \| x _ { i } - x _ { j } \| ^ { 2 } } \bigr ) \Bigr ] \Lambda ( d s ) .
$$

Inner sum decomposition. For any fixed $s > 0$

$$
\sum _ { i , j } w _ { i } w _ { j } \big ( 1 - e ^ { - s \| x _ { i } - x _ { j } \| ^ { 2 } } \big ) = \underbrace { \sum _ { i , j } w _ { i } w _ { j } } _ { = 0 } - \sum _ { i , j } w _ { i } w _ { j } e ^ { - s \| x _ { i } - x _ { j } \| ^ { 2 } } = - Q ( s ) ,
$$

where

$$
Q ( s ) : = \sum _ { i , j = 1 } ^ { n } w _ { i } w _ { j } e ^ { - s \| x _ { i } - x _ { j } \| ^ { 2 } } .
$$

Strict positivity of $Q ( s )$ . Fix $s > 0$ . By [22, Cor. 2.1(c)], for pairwise distinct $x _ { 1 } , \ldots , x _ { n } \in \mathbb { R } ^ { d }$

$$
\sum _ { i , j = 1 } ^ { n } w _ { i } w _ { j } e ^ { - s \| x _ { i } - x _ { j } \| ^ { 2 } } > 0 \qquad \mathrm { f o r e v e r y ~ n o n z e r o } ( w _ { 1 } , \ldots , w _ { n } ) \in \mathbb { R } ^ { n } .
$$

Therefore $Q ( s ) > 0$ for all $s > 0$ whenever the weights are not all zero, which is the case for the weights fixed above.

Conclusion. Combining the pieces,

$$
S \ = \ - 2 b \Big \| \sum _ { i } w _ { i } x _ { i } \Big \| ^ { 2 } \ - \ \int _ { ( 0 , \infty ) } Q ( s ) \Lambda ( d s ) .
$$

The first term is $\leq 0$ . Moreover, since $\Lambda \neq 0$ and $Q ( s ) > 0$ for all $s > 0$ , we have

$$
\int _ { ( 0 , \infty ) } Q ( s ) \Lambda ( d s ) > 0 .
$$

Hence $S < 0 ,$ , and therefore k is conditionally strictly negative definite.

Theorem I.1 (When radial kernels built from Bernstein functions make MMD a metric). Let $d \geq 1$ and let $k : \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } $ R be symmetric. Assume that there exist a constant $C \in \mathbb R$ and a Bernstein function $\phi : [ 0 , \infty )  [ 0 , \infty )$ with $\phi ( 0 ) = 0$ such that

$$
k ( x , y ) = C - \phi { \big ( } \| x - y \| ^ { 2 } { \big ) } .
$$

By Theorem E.1, k is CPD, so $\mathbf { M M D } _ { k }$ is defined as a quadratic form, as in Definition A.6. Write the Lévy–Khintchine representation of ϕ as

$$
\phi ( t ) = b t + \int _ { ( 0 , \infty ) } \bigl ( 1 - e ^ { - t s } \bigr ) \Lambda ( d s ) , \qquad b \geq 0 ,
$$

where Λ is a nonnegative measure with $\begin{array} { r } { \int _ { ( 0 , \infty ) } \operatorname* { m i n } \lbrace 1 , s \rbrace \Lambda ( d s ) < \infty . } \end{array}$

Then $\mathbf { M M D } _ { k }$ satisfies the metric axioms except possibly identity of indiscernibles (as in Proposition A.3). Moreover,

$$
\begin{array} { r } { \boxed { \mathbf { M M D } _ { k } \mathrm { ~ i s ~ a ~ m e t r i c ~ i f ~ a n d ~ o n l y ~ i f ~ } \Lambda \not = 0 . } } \end{array}
$$

Proof.

Step 1: Reducing identity of indiscernibles to strictness. Since k is CPD, Proposition A.3 implies that $\mathbf { M M D } _ { k }$ satisfies nonnegativity, symmetry, and the triangle inequality. It is a metric if and only if identity of indiscernibles holds, which by the same proposition is equivalent to k being CSPD.

Step 2: From CSPD of k to CSND of the Bernstein part. Adding a constant does not change whether a kernel is CPD or CSPD, hence

$$
k \mathrm { i s } \mathrm { C S P D } \quad \Longleftrightarrow \quad - \phi ( \| x - y \| ^ { 2 } ) \mathrm { i s } \mathrm { C S P D } \quad \Longleftrightarrow \quad \phi ( \| x - y \| ^ { 2 } ) \mathrm { i s } \mathrm { C S N D } .
$$

Step 3: When the Bernstein part is CSND. Lemma I.2 proves this criterion on finite sets of points. For the general case required by Definition $\mathrm { A . } 5 ,$ , Theorem 4.1 of [17] shows that $\phi ( \| x - y \| ^ { 2 } )$ is CSND when $\Lambda \neq 0$ , and Lemma I.2 already shows that it is not when $\Lambda \equiv 0$ . Hence

$$
\phi ( \| x - y \| ^ { 2 } ) { \mathrm { ~ i s ~ C S N D ~ o n ~ } } \mathbb { R } ^ { d } \quad \iff \quad \Lambda \neq 0 .
$$

Combining Steps 1–3 yields

$$
\mathbf { M M D } _ { k } { \mathrm { ~ i s ~ a ~ m e t r i c } } \quad \iff \quad \Lambda \not \equiv 0 ,
$$

which proves the boxed statement.