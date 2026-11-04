# Logarithmic Regret for Alternating Mirror Descent

Tianlong Nan<sup>†</sup>

Christian Kroer<sup>†</sup>

## Abstract

Alternating online mirror descent (AltOMD) is a simple learning rule for zero-sum games, which improves upon simultaneous OMD numerically and theoretically. Previous work obtained $O ( T ^ { 1 / 3 } )$ and ${ \cal O } ( T ^ { 1 / 5 } )$ regret bounds for AltOMD, by leveraging modified-energy-function-based analyses. However, settling whether AltOMD achieves near-constant regret has remained an open question. In this paper we resolve the question of whether AltOMD achieves near-constant regret in the afirmative. In particular, we show an O(log T) regret bound on the sum of the players’ external regrets for any two-player zero-sum bilinear game with compact convex strategy sets, under mild regularity assumptions on the regularizers. We verify that our assumptions hold for the negative entropy regularizer, thereby obtaining logarithmic regret for the alternating multiplicative weights updates algorithm. The regret bound also yields an O(log T/T) duality gap bound for the averaged strategies. A Lean formalization of the results is available at https://github.com/CoffeeAndConvexity/AltOMD-Lean.

## 1 Introduction

Learning dynamics in games are studied both as models of adaptive strategic behavior and as decentralized algorithms for computing equilibria. External regret measures each player’s performance relative to its best fixed strategy in hindsight. In two-player zero-sum games, it also connects learning to equilibrium computation: the sum of the players’ regrets controls the equilibrium gap of their average strategies. Sharp regret bounds for simple learning rules therefore help characterize strategic behavior and establish eficient algorithms for solving games.

Alternation is a simple modification of learning dynamics that can substantially improve numerical performance. Alternating updates exhibit improved stability over simultaneous updates (Gidel et al., 2019) and are an important component of successful practical game solvers such as CFR+ (Tammelin et al., 2015). We study alternating online mirror descent (AltOMD), in which the players take turns applying the standard mirror-descent update (Katona et al., 2026; Wibisono et al., 2022). The first player updates against the second player’s current strategy, and the second then updates against the first player’s new strategy. Each player uses a regularizer to account for the geometry of its strategy space; negative Shannon entropy on the simplex, for example, yields alternating multiplicative weights update.

A central question is whether AltOMD can achieve near-constant regret in bilinear games with compact convex strategy sets under mild assumptions on the regularizers. In particular, an O(log T) bound on the sum of the players’ regrets over T rounds would yield an O(log T /T) equilibrium gap for their average strategies.

Wibisono, Tao, and Piliouras (2022) connected AltOMD to Hamiltonian dynamics in the dual space. By controlling the first correction to the energy under third-order smoothness, they obtained

$O ( T ^ { 1 / 3 } )$ total regret and an $O ( T ^ { - 2 / 3 } )$ equilibrium gap for the average strategies. Katona, Wang, and Wibisono (2026) extended this approach using a modified Hamiltonian constructed from the Baker– Campbell–Hausdorf (BCH) series, improving these rates to $O ( T ^ { 1 / 5 } )$ and $O ( T ^ { - 4 / 5 } )$ , respectively. Katona, Wang, and Wibisono (2026) further conjectured estimates at arbitrary order that would imply $O ( T ^ { \varepsilon } )$ regret for every fixed $\varepsilon > 0$ . They also established bounded regret under additional convergence and tail conditions on the modified Hamiltonian, or boundedness of the dual trajectories. A logarithmic regret guarantee under mild regularizer assumptions has nevertheless remained open, including for alternating multiplicative weights on the simplex.

The key obstacle is controlling the error when the modified Hamiltonian is truncated at an order that grows with the horizon T. Estimates at each fixed order do not sufice: their hidden constants may grow too quickly with the order. Moreover, the infinite modified-Hamiltonian series need not converge, and compact strategy sets do not ensure bounded dual trajectories. Extending this approach to logarithmic regret therefore requires explicit control of higher-order corrections throughout the dual space, as discussed by Katona et al. (2026, Section 9.1).

Our result. We prove an ${ \cal O } ( \log T )$ bound on the sum of the players’ external regrets for AltOMD in two-player zero-sum bilinear games with compact convex strategy sets (Theorem 3.7). The result applies to Legendre regularizers whose conjugates satisfy a uniform geometric bound on their Taylor coeficients, assuming a finite initial Bregman radius. For a prescribed horizon T, the algorithm uses a stepsize $\eta _ { T } = \Theta ( 1 / \log T )$ that remains constant throughout the run. The average strategies then have duality gap $O ( \log T / T )$

These assumptions hold for negative Shannon entropy, yielding the same guarantees for alternating multiplicative weights on the simplex (Corollary 4.2). Our result improves the conjectured ${ \cal O } ( T ^ { \varepsilon } )$ target to logarithmic regret for this class of regularizers and applies even when equilibria lie on the boundary and dual trajectories are unbounded.

## 1.1 Related work

Alternation in game dynamics. Alternation can substantially change the stability and convergence properties of learning dynamics. Early results concern unconstrained bilinear games, where simultaneous GDA diverges. Gidel et al. (2019) showed that alternating gradient descent-ascent (AltGDA) instead keeps the iterates bounded, though it does not converge, and that adding negative momentum yields linear convergence. Bailey et al. (2020) show that a player using alternating gradient descent has bounded regret against every fixed comparator, whatever the opponent does, under a regret notion that evaluates each opponent strategy against the player’s iterates both before and after its update.

For constrained games, Wibisono et al. (2022) connected AltOMD to Hamiltonian dynamics in the dual space and obtained $O ( T ^ { 1 / 3 } )$ total regret through a modified-energy argument. They also highlighted that the discretized AltOMD algorithm inherits some dynamic properties from continuous-time dynamics (Mertikopoulos et al., 2018). Katona et al. (2026) developed higher-order modified-Hamiltonian expansions and improved this bound to $O ( T ^ { 1 / 5 } )$ under stronger smoothness assumptions. For projected AltGDA, Nan et al. (2026a) established $O ( 1 / T )$ ergodic convergence under an interior-equilibrium assumption and a local $O ( 1 / T )$ ergodic guarantee in general matrix games. Nan et al. (2026b) subsequently proved global $O ( 1 / T )$ ergodic convergence for all finite two-player zero-sum matrix games. The efects of alternation have also been explored in self-play settings (Burch et al., 2019; Cai et al., 2025; Li and Huang, 2026; Sullins and Kash, 2026; Tammelin et al., 2015).

A complementary line studies alternating regret against adversarial losses, developing algorithms tailored to this criterion for online linear and convex optimization (Cevher et al., 2023; Hait et al., 2025). Recent independent work by Tao and Zheng (2026) and Zhang (2026) obtains optimal alternating-regret bounds using modified exponential-weights algorithms. In particular, their modified algorithms guarantee an $O ( \log d / T )$ averaged duality gap in finite two-player zero-sum games. Our work, in contrast, uses the standard unmodified AltOMD dynamics, showing that no modification is necessary to achieve such a rate.

Learning dynamics in matrix games. No-regret learning provides a general route to equilibrium computation. Standard no-regret learning dynamics can yield $O ( 1 / \sqrt { T } )$ average-iterate convergence rate in two-player zero-sum games, while faster convergence rates are available through predictive or corrective updates. Mirror-Prox achieves an $O ( 1 / T )$ averaged duality gap for smooth convex-concave saddle-point problems (Nemirovski, 2004). Rakhlin and Sridharan (2013) develop optimistic mirror descent using predictable sequences and obtain $O ( \log T / T )$ averaged-iterate convergence for strongly uncoupled learning in matrix games. Syrgkanis et al. (2015) further shows a constant bound on the sum of regrets for suitable optimistic updates with constant stepsizes, implying an $O ( 1 / T )$ average-iterate convergence rate in zero-sum games.

Recent work also sharpens individual-regret guarantees in finite general-sum games. Ha (2026b) proves $O ( { \sqrt { n } } \log d _ { i } \log T )$ individual external regret for Optimistic Hedge (Rakhlin and Sridharan, 2013), where $n$ is the number of players and $d _ { i }$ is player $i \mathrm { \ ' } _ { \mathrm { S } }$ number of actions, with a common stepsize of order $1 / ( { \sqrt { n } } \log T )$ held constant during play. Other recent algorithms obtain horizonindependent individual external regret through exponential-moving-average predictors and lifted regularization (Liu et al., 2026), discounted higher-order optimism (Abbadi et al., 2026), and one-step multiplicatively optimistic regret matching (Soleymani and Piliouras, 2026). For the stronger swap-regret criterion, Tsuchiya (2026) obtains $O ( { \sqrt { \log T } } )$ regret for fixed game dimensions by combining the Blum-Mansour reduction with hybrid-regularized optimistic FTRL.

## 2 Preliminaries

We study the bilinear two-player zero-sum game defined by the payof matrix $A \in \mathbb { R } ^ { m \times n }$ and a pair of nonempty compact and convex strategy sets $\mathcal { X } \subset \mathbb { R } ^ { n }$ and $\mathcal { V } \subset \mathbb { R } ^ { m }$ . In this game, the first player chooses her strategy x to minimize the bilinear payof, while the second player chooses her strategy $y$ to maximize it. The game can then be encoded as the following minimax problem:

$$
\operatorname* { m i n } _ { x \in { \mathcal { X } } } \operatorname* { m a x } _ { y \in { \mathcal { Y } } } { y ^ { \top } A x } .
$$

(Bilinear game)

Legendre regularizer and dual space. Let $\psi _ { \mathcal { X } }$ and ψ<sub>Y</sub> be proper closed convex functions whose domains have closures $\mathcal { X }$ and $\mathcal { V } ,$ respectively. We takes their values to be $+ \infty$ outside their domains. In this paper, we call them regularizers or mirror maps. We consider the Legendre regularizers defined in the following definition. We use $\mathrm { r i } ( \cdot )$ to denote the relative interior of a set.

Definition (Legendre functions). A proper closed convex function $\psi : \mathbb { R } ^ { d }  \mathbb { R } \cup \{ + \infty \}$ is Legendre if it is strictly convex and continuously diferentiable on $\operatorname { r i } ( \operatorname { d o m } \psi )$ , and $\| \nabla \psi ( x _ { k } ) \| _ { 2 } \to \infty$ whenever a sequence $x _ { k } \in \mathrm { r i } ( \operatorname { d o m } \psi )$ converges to a relative boundary point of dom $\psi$ . Here, gradients and boundaries are taken relative to the afine hull of the domain of $\psi .$

Then, we define the convex conjugates of the regularizer functions by

$$
\begin{array} { r } { F ( u ) : = \psi _ { \mathcal { X } } ^ { \ast } ( u ) = \operatorname* { s u p } _ { x \in \mathcal { X } } \left\{ \langle u , x \rangle - \psi _ { \mathcal { X } } ( x ) \right\} , } \\ { G ( w ) : = \psi _ { \mathcal { Y } } ^ { \ast } ( w ) = \operatorname* { s u p } _ { y \in \mathcal { Y } } \left\{ \langle w , y \rangle - \psi _ { \mathcal { Y } } ( y ) \right\} . } \end{array}\tag{1}
$$

Throughout this paper, we call the joint feasible set $\mathcal { X } \times \mathcal { V }$ the (product) primal space. To facilitate the analysis, we consider an associated (product) dual space or mirror space $\mathbb { R } ^ { n } \times \mathbb { R } ^ { m }$ with coordinates $z : = ( u , w )$ . Given Legendre regularizers $\psi _ { \mathcal { X } }$ and $\psi _ { \mathcal { Y } }$ , Each pair of dual variables u and w can be mapped to a corresponding pair of primal strategies x and y by

$$
\begin{array} { r } { x = \nabla F ( u ) \in \mathrm { r i } ( \mathcal { X } ) , \qquad y = \nabla G ( w ) \in \mathrm { r i } ( \mathcal { Y } ) . } \end{array}\tag{2}
$$

Alternating mirror descent (AltOMD). Consider a Legendre regularizer $\psi$ on $\mathbb { R } ^ { d }$ . For $x ^ { \prime } \in$ dom ψ and $x \in \mathrm { r i } ( \operatorname { d o m } \psi )$ , the Bregman divergence generated by $\psi$ is defined as

$$
D _ { \psi } ( x ^ { \prime } , x ) : = \psi ( x ^ { \prime } ) - \psi ( x ) - \langle \nabla \psi ( x ) , x ^ { \prime } - x \rangle \geq 0 ,
$$

where the nonnegativity follows from the convexity of $\psi .$ . For $x ^ { \prime } \notin$ dom $\psi _ { ; }$ , we set $D _ { \psi } ( x ^ { \prime } , x ) : = + \infty$ The most common example of a Bregman divergence is the KL divergence, which arises from the negative entropy regularizer.

The alternating online mirror descent (AltOMD) updates in the primal space are given by

$$
\begin{array} { r l } & { x _ { k + 1 } = \arg \operatorname* { m i n } _ { x \in \mathcal { X } } \Big \{ \eta \langle A ^ { \top } y _ { k } , x \rangle + D _ { \psi _ { \mathcal { X } } } ( x , x _ { k } ) \Big \} , } \\ & { y _ { k + 1 } = \arg \operatorname* { m i n } _ { y \in \mathcal { Y } } \Big \{ - \eta \langle A x _ { k + 1 } , y \rangle + D _ { \psi _ { \mathcal { Y } } } ( y , y _ { k } ) \Big \} , } \end{array}\tag{AltOMD-Primal}
$$

where $\eta > 0$ is the step size. Equivalently, by the first-order optimality conditions, the updates can be expressed in the dual space as follows:

$$
\begin{array} { r } { u _ { k + 1 } = u _ { k } - \eta A ^ { \top } y _ { k } , x _ { k + 1 } = \nabla F ( u _ { k + 1 } ) , } \\ { w _ { k + 1 } = w _ { k } + \eta A x _ { k + 1 } , y _ { k + 1 } = \nabla G ( w _ { k + 1 } ) . } \end{array}\tag{AltOMD-Dual}
$$

Alternation means that the players update their strategies one after another, rather than simultaneously: the first player updates $x _ { k + 1 }$ based on the previous strategy of the second player $y _ { k }$ , and then the second player updates $y _ { k + 1 }$ using the newly updated $x _ { k + 1 }$

Since the AltOMD updates in the dual space are unconstrained and additive, it is convenient to analyze the algorithm in the dual space.

Continuous-time Hamiltonian dynamics. Similar to the previous work (Katona et al., 2026; Wibisono et al., 2022), we leverage Hamiltonian dynamics to analyze AltOMD. Here, we present a brief overview of continuous-time Hamiltonian dynamics.

Hamiltonian dynamics describe continuous-time dynamics whose evolution is driven by a scalar energy function. In the AltOMD case, the continuous-time Hamiltonian dynamics corresponding to (AltOMD-Dual) is

$$
\dot { u } ( t ) = - A ^ { \top } \nabla G ( w ( t ) ) , \qquad \dot { w } ( t ) = A \nabla F ( u ( t ) ) .\tag{3}
$$

Set $z : = ( u , w )$ . Then Eq. (3) takes the compact form $\dot { z } = J _ { A } \nabla H _ { 0 } ( z )$ , where

$$
H _ { 0 } ( u , w ) : = F ( u ) + G ( w ) \mathrm { a n d } J _ { A } : = \left( \begin{array} { c c } { { 0 } } & { { - A ^ { \top } } } \\ { { A } } & { { 0 } } \end{array} \right) .\tag{4}
$$

Since $J _ { A } ^ { \top } = - J _ { A }$ , the energy is conserved along the continuous trajectory:

$$
\frac { \mathrm { d } } { \mathrm { d } t } H _ { 0 } ( z ( t ) ) = \nabla H _ { 0 } ( z ( t ) ) ^ { \top } \dot { z } ( t ) = \nabla H _ { 0 } ( z ( t ) ) ^ { \top } J _ { A } \nabla H _ { 0 } ( z ( t ) ) = 0 .
$$

AltOMD approximates the continuous-time Hamiltonian dynamics by sequentially updating the two variables.

Regret and duality gap. We define the regret up to time $T$ as the sum of the minimizing and maximizing players’ external regrets:

$$
\begin{array} { r l } & { \mathrm { R e g } _ { T } : = \displaystyle \operatorname* { m a x } _ { \tilde { x } \in \mathcal { X } } \sum _ { k = 0 } ^ { T - 1 } y _ { k } ^ { \top } A ( x _ { k + 1 } - \tilde { x } ) + \displaystyle \operatorname* { m a x } _ { \tilde { y } \in \mathcal { Y } } \sum _ { k = 0 } ^ { T - 1 } ( \tilde { y } - y _ { k } ) ^ { \top } A x _ { k + 1 } } \\ & { \quad \quad \quad = \displaystyle \operatorname* { m a x } _ { ( \tilde { x } , \tilde { y } ) \in \mathcal { X } \times \mathcal { Y } } \sum _ { k = 0 } ^ { T - 1 } \big ( \tilde { y } ^ { \top } A x _ { k + 1 } - y _ { k } ^ { \top } A \tilde { x } \big ) . } \end{array}\tag{5}
$$

Let $\begin{array} { r } { \bar { x } _ { T } : = \frac { 1 } { T } \sum _ { k = 0 } ^ { T - 1 } x _ { k + 1 } } \end{array}$ and $\begin{array} { r } { \bar { y } _ { T } : = \frac { 1 } { T } \sum _ { k = 0 } ^ { T - 1 } y _ { k } } \end{array}$ . For any strategy pair $( x , y ) \in \mathcal { X } \times \mathcal { Y }$ , the duality gap is

$$
\operatorname { G a p } ( x , y ) : = \operatorname* { m a x } _ { \tilde { y } \in \mathcal { V } } \tilde { y } ^ { \top } A x - \operatorname* { m i n } _ { \tilde { x } \in \mathcal { X } } y ^ { \top } A \tilde { x } .\tag{6}
$$

The duality gap is nonnegative and vanishes exactly when $( x , y )$ is a saddle point of (Bilinear game). By bilinearity, we have

$$
\begin{array} { r } { \mathrm { R e g } _ { T } = T \left( \operatorname* { m a x } _ { \tilde { y } \in \mathcal { V } } \tilde { y } ^ { \top } A \bar { x } _ { T } - \operatorname* { m i n } _ { \tilde { x } \in \mathcal { X } } \bar { y } _ { T } ^ { \top } A \tilde { x } \right) = T \mathrm { G a p } ( \bar { x } _ { T } , \bar { y } _ { T } ) . } \end{array}\tag{7}
$$

Hence, a sublinear regret implies that the duality gap of the averaged strategy converges to zero.

## 3 Logarithmic Regret Bounds

In this section, we establish logarithmic regret bounds for the AltOMD algorithm.

## 3.1 Notation and Assumptions

We first introduce some additional notation and assumptions. For a smooth scalar function $f :  { \mathbb { R } ^ { d } } \to  { \mathbb { R } }$ and an integer $s \geq 1$ , let $D ^ { s } f ( z )$ denote its s-th derivative at $z ,$ which is an order-s tensor. For an order-s tensor $T$ , we use $T [ v _ { 1 } , \dots , v _ { s } ]$ to denote its contraction with the vectors $v _ { 1 } , \ldots , v _ { s }$ For example, for $s = 1 , D ^ { 1 } f ( z ) [ v _ { 1 } ] = \nabla f ( z ) ^ { \top } v _ { 1 }$ , and for $s = 2 , D ^ { 2 } f ( z ) [ v _ { 1 } , v _ { 2 } ] = v _ { 1 } ^ { \top } \nabla ^ { 2 } f ( z ) v _ { 2 }$ . For any integer $s \geq 1$ , we define the operator norm induced by Euclidean norms as

$$
\| D ^ { s } f ( z ) \| _ { \mathrm { o p } } : = \operatorname* { s u p } _ { \| v _ { 1 } \| _ { 2 } , \ldots , \| v _ { s } \| _ { 2 } \leq 1 } \left| D ^ { s } f ( z ) [ v _ { 1 } , \ldots , v _ { s } ] \right| .\tag{8}
$$

For $s \ : = \ : 1$ and $s \ : = \ : 2$ , the operator norms reduce to the Euclidean norm and spectral norm, respectively.

The first assumption imposes a derivative growth condition on the conjugate functions $F = \psi _ { \mathcal { X } } ^ { * }$ and $G = \psi _ { \chi } ^ { * }$ . This condition ensures that the growth of derivatives is controlled at every order, which is crucial for the subsequent analysis.

Assumption 3.1 (Derivative growth condition). There is a constant $K \geq 1$ such that, for every integer $s \geq 1$ ，

$$
\frac { 1 } { s ! } \left( \operatorname* { s u p } _ { u } \lVert D ^ { s } F ( u ) \rVert _ { \mathrm { o p } } + \operatorname* { s u p } _ { w } \lVert D ^ { s } G ( w ) \rVert _ { \mathrm { o p } } \right) \leq K ^ { s } .\tag{9}
$$

We comment that this assumption is mild in the sense that it holds for common regularizers such as the negative entropy regularizer, which is shown in Section 4.

The second assumption is standard and ensures that the initial Bregman divergence is finite. For example, when the feasible sets are simplices and the regularizers are negative entropy, both players can choose the initial strategies to be the uniform distributions over the strategy sets.

Assumption 3.2 (Finite initial comparator radius). The initial strategy $( x _ { 0 } , y _ { 0 } )$ satisfies that

$$
\begin{array} { r } { \Lambda _ { 0 } : = \operatorname* { s u p } _ { \tilde { x } \in \mathcal { X } } D _ { \psi _ { \mathcal { X } } } ( \tilde { x } , x _ { 0 } ) + \operatorname* { s u p } _ { \tilde { y } \in \mathcal { Y } } D _ { \psi _ { \mathcal { Y } } } ( \tilde { y } , y _ { 0 } ) < \infty . } \end{array}\tag{10}
$$

## 3.2 Analysis

We begin with the following regret decomposition. All the proofs in this section are deferred to Appendix A.

Lemma 3.3 (Regret decomposition). Fix a time horizon $T \geq 1$ and a step size $\eta > 0$ . For any comparators $( \tilde { x } , \tilde { y } ) \in \mathrm { d o m } \psi _ { \mathcal { X } } \times \mathrm { d o m } \psi _ { \mathcal { Y } } ,$ define the Fenchel coupling:

$$
\Gamma _ { \tilde { x } , \tilde { y } } ( u , w ) : = F ( u ) - ( \langle u , \tilde { x } \rangle - \psi _ { \mathcal { X } } ( \tilde { x } ) ) + G ( w ) - ( \langle w , \tilde { y } \rangle - \psi _ { \mathcal { Y } } ( \tilde { y } ) ) \geq 0 ,
$$

where $( u , w )$ are the dual variables in (AltOMD-Dual). Then, we have $\Gamma _ { \tilde { x } , \tilde { y } } ( u _ { k } , w _ { k } ) = D _ { \psi _ { \mathcal { X } } } ( \tilde { x } , x _ { k } ) +$ $D _ { \psi _ { \mathcal { Y } } } ( \tilde { y } , y _ { k } )$ for all $k \geq 0$ . Moreover, all the iterates generated by the AltOMD updates satisfy

$$
\eta \big ( \tilde { y } ^ { \top } A x _ { k + 1 } - y _ { k } ^ { \top } A \tilde { x } \big ) = H _ { 0 } ( z _ { k + 1 } ) - H _ { 0 } ( z _ { k } ) + \Gamma _ { \tilde { x } , \tilde { y } } ( z _ { k } ) - \Gamma _ { \tilde { x } , \tilde { y } } ( z _ { k + 1 } ) ,\tag{11}
$$

where $H _ { 0 } ( z ) = F ( u ) + G ( w )$ is defined in Eq. (4).

We sum the right-hand side of Eq. (11) over $k = 0 , \ldots , T - 1$ . The intermediate terms cancel, so only the boundary terms remain: $H _ { 0 } ( z _ { T } ) - H _ { 0 } ( z _ { 0 } )$ and $\Gamma _ { \tilde { x } , \tilde { y } } ( z _ { 0 } ) - \Gamma _ { \tilde { x } , \tilde { y } } ( z _ { T } )$ . The second term $\Gamma _ { \tilde { x } , \tilde { y } } ( z _ { 0 } ) - \Gamma _ { \tilde { x } , \tilde { y } } ( z _ { T } )$ can be bounded by the nonnegativity of Bregman divergences and the initial strategy condition Assumption 3.2.

The dificulty lies in controlling the first term $H _ { 0 } ( z _ { T } ) - H _ { 0 } ( z _ { 0 } )$ . As mentioned earlier, this $H _ { 0 }$ corresponds to the original energy in the Hamiltonian dynamics, and is preserved under the continuous-time counterpart of the AltOMD. For this reason, the corresponding continuous-time dynamics achieves a constant regret because $H _ { 0 } ( z _ { T } ) = H _ { 0 } ( z _ { 0 } )$ for all $T \geq 0$ . However, discretized updates of the continuous-time dynamics generally do not preserve this energy, and there is no guarantee that $H _ { 0 } ( z _ { T } )$ will be bounded above.

To handle this issue, following Katona et al. (2026), we leverage a modified Hamiltonian energy function (Field and Nijhof, 2003). This modified energy need not converge, and because of this we treat it as a formal power series (Niven, 1969): its coeficients are manipulated algebraically without assuming convergence.

We write the modified energy in the following formal power series form:

$$
\widehat H _ { \eta } ( z ) : = H _ { 0 } ( z ) + \eta H _ { 1 } ( z ) + \eta ^ { 2 } H _ { 2 } ( z ) + \cdots ,\tag{12}
$$

where each correction term $H _ { r } , r \geq 1$ is a linear combination of nested/iterated Poisson brackets $\{ F , G \}$ of F and G, which is defined as

$$
\{ F , G \} : = \nabla F ^ { \top } J _ { A } \nabla G .
$$

(Poisson bracket)

For example, the first and second correction terms are

$$
\begin{array} { l } { \displaystyle { H _ { 1 } ( z ) = \frac { 1 } { 2 } \{ F , G \} ( u , w ) = - \frac { 1 } { 2 } \nabla G ( w ) ^ { \top } A \nabla F ( u ) , } } \\ { \displaystyle { H _ { 2 } ( z ) = \frac { 1 } { 1 2 } \{ G , \{ G , F \} \} ( u , w ) + \frac { 1 } { 1 2 } \{ F , \{ F , G \} \} ( u , w ) . } } \end{array}
$$

We defer the complete construction of the correction terms to Appendix A.1.

We further define $\phi _ { G } ^ { \eta } ( u , w ) : = \bigl ( u - \eta A ^ { \top } \nabla G ( w ) , w \bigr ) , \phi _ { F } ^ { \eta } ( u , w ) : = \bigl ( u , w + \eta A \nabla F ( u ) \bigr )$ and $\mathcal { M } _ { \eta } =$ $\phi _ { F } ^ { \eta } \circ \phi _ { G } ^ { \eta }$ , where ◦ denotes function composition. Hence, by (AltOMD-Dual), the AltOMD updates correspond to applying the map $\mathcal { M } _ { \eta }$ to the current iterate, $\mathrm { i . e . , ~ } z _ { k + 1 } = \mathcal { M } _ { \eta } ( z _ { k } )$ . With this notation, we present the following classical formal preservation identity, which is proved in, e.g., Hairer et al. (2006, Chapters III and IX). For completeness, we provide a self-contained proof in Appendix A.1. Here, the formal preservation refers to a coeficient-by-coeficient equality in the formal power series.

Lemma 3.4 (Formal preservation of $ { \widehat { H } } _ { \eta } )$ . Let F and G be smooth, let $H _ { r }$ be the BCH correction terms constructed in Appendix A.1, and let $\widehat { H } _ { \eta }$ be the formal modified energy defined by $E q .$ (12). Then, $\widehat { H } _ { \eta }$ satisfies

$$
\widehat { H } _ { \eta } \big ( \mathcal { M } _ { \eta } ( z ) \big ) = \widehat { H } _ { \eta } ( z ) \qquad f o r \ e v e r y \ z \in \mathbb { R } ^ { n } \times \mathbb { R } ^ { m } ,\tag{13}
$$

as an identity of formal power series in $\eta .$

For any integer $N \geq 0$ , we consider an arbitrary-order truncation for the modified energy. Given any integer $N \geq 0$ , we define a truncated modified energy of order $\eta ^ { N }$ :

$$
\widehat { H } _ { \eta } ^ { [ N ] } ( z ) : = \sum _ { r = 0 } ^ { N } \eta ^ { r } H _ { r } ( z ) = H _ { 0 } ( z ) + \sum _ { r = 1 } ^ { N } \eta ^ { r } H _ { r } ( z ) .\tag{14}
$$

Remark 3.5. Katona et al. (2026) considered the same modified energy, then used a finite truncation of the modified energy $\widehat { H } _ { \eta }$ and proved its approximate preservation. In particular, they verified approximate preservation of the truncated modified energy up to order three in arbitrary dimension. However, this finite truncation still leaves an upper bound on $H _ { 0 } ( z _ { T } ) - H _ { 0 } ( z _ { 0 } )$ with a polynomial dependence in $T ,$ which leads to an $O ( T ^ { 1 / 5 } )$ regret bound. In contrast, we consider the truncated modified energy for arbitrary order N. We show that, for any integer $N \geq 0$ , we can truncate the modified energy $\widehat { H } _ { \eta }$ at order $N .$ , and this truncated energy is always approximately preserved up to a suficiently small error. We then take $N = \Theta ( \log T )$ and use this T-dependent truncation to obtain an O(log T) regret bound.

By utilizing the coeficient-by-coeficient equality up to N order, and the upper bounds on the higher-order terms specified in Assumption 3.1, we can establish precise control over the truncated modified energy. In particular, we establish the following bounds for the truncated modified energy, which is the central component of our analysis.

Lemma 3.6. Under Assumption 3.1, set $C _ { 0 } = 2 4 \operatorname* { m a x } \lbrace \| J _ { A } \| _ { \mathrm { o p } } , 1 \rbrace K ^ { 2 }$ . For every integer $N \geq 0$ and stepsize $\begin{array} { r } { 0 < \eta \le \frac { 1 } { 1 6 C _ { 0 } ( N + 2 ) } } \end{array}$ , the truncated modified energy $\widehat { H } _ { \eta } ^ { [ N ] }$ defined in Eq. (14) satisfies, uniformly over all dual coordinates z,

$$
\begin{array} { c } { { \displaystyle \left| \widehat { H } _ { \eta } ^ { [ N ] } ( z ) - H _ { 0 } ( z ) \right| \leq \frac { 6 4 } { 3 } C _ { 0 } ^ { 2 } \eta , } } \\ { { \displaystyle \left| \widehat { H } _ { \eta } ^ { [ N ] } ( \mathcal { M } _ { \eta } ( z ) ) - \widehat { H } _ { \eta } ^ { [ N ] } ( z ) \right| \leq 1 6 e ^ { 2 } C _ { 0 } ^ { 2 } \eta 4 ^ { - ( N + 1 ) } . } } \end{array}\tag{15}
$$

The first bound in Lemma 3.6 controls the deviation of the truncated modified energy $\widehat { H } _ { n } ^ { [ N ] }$ from the original energy $H _ { 0 }$ . The second bound quantifies the one-step near-preservation of $\widehat { H } _ { \eta } ^ { [ N ] }$ It is crucial to guarantee the exponential decay of the one-step gap in the truncated modified energy as N increases, when we can take $\eta = \Theta ( 1 / N )$

With these tools from Eq. (15) in hand, we can control the change in the original energy $H _ { 0 }$ along the trajectory of the AltOMD updates. In particular, we can then bound $H _ { 0 } ( z _ { T } ) - H _ { 0 } ( z _ { 0 } )$ by

$$
\begin{array} { r l } & { \quad \bigl ( H _ { 0 } ( z _ { T } ) - \widehat { H } _ { \eta } ^ { [ N ] } ( z _ { T } ) \bigr ) + \widehat { H } _ { \eta } ^ { [ N ] } ( z _ { T } ) - \widehat { H } _ { \eta } ^ { [ N ] } ( z _ { 0 } ) - \bigl ( H _ { 0 } ( z _ { 0 } ) - \widehat { H } _ { \eta } ^ { [ N ] } ( z _ { 0 } ) \bigr ) } \\ & { \leq \Bigl | H _ { 0 } ( z _ { T } ) - \widehat { H } _ { \eta } ^ { [ N ] } ( z _ { T } ) \Bigr | + \displaystyle \sum _ { t = 0 } ^ { T - 1 } \left| \widehat { H } _ { \eta } ^ { [ N ] } ( z _ { t + 1 } ) - \widehat { H } _ { \eta } ^ { [ N ] } ( z _ { t } ) \right| + \Bigl | H _ { 0 } ( z _ { 0 } ) - \widehat { H } _ { \eta } ^ { [ N ] } ( z _ { 0 } ) \Bigr | } \\ & { \leq 1 6 e ^ { 2 } C _ { 0 } ^ { 2 } \eta T 4 ^ { - ( N + 1 ) } + \frac { 1 2 8 } { 3 } C _ { 0 } ^ { 2 } \eta . } \end{array}
$$

By Lemma 3.3, when $\eta = \Theta ( 1 / N )$ , the regret can be bounded by

$$
\begin{array} { l } { { \displaystyle { \mathrm { R e g } _ { T } \leq \frac { 1 } { \eta } \left( 1 6 e ^ { 2 } C _ { 0 } ^ { 2 } \eta T 4 ^ { - ( N + 1 ) } + \frac { 1 2 8 } { 3 } C _ { 0 } ^ { 2 } \eta + \Lambda _ { 0 } \right) } } } \\ { { \displaystyle ~ = 1 6 e ^ { 2 } C _ { 0 } ^ { 2 } T 4 ^ { - ( N + 1 ) } + \frac { 1 2 8 } { 3 } C _ { 0 } ^ { 2 } + \frac { \Lambda _ { 0 } } { \eta } . } } \end{array}
$$

Therefore, we can set $N = \Theta ( \log T )$ and obtain an ${ \cal O } ( \log T )$ regret bound. For integer $T \geq 1$ and $N = \lfloor \log _ { 2 } T \rfloor$ , we have $\begin{array} { r } { T 4 ^ { - ( N + 1 ) } \le \frac { 1 } { 4 } } \end{array}$ : this is equality for $T = 1$ , while for $T \geq 2 , T < 2 ^ { N + 1 } \leq 4 ^ { N }$

Theorem 3.7 (Logarithmic regret of AltOMD). Under Assumptions $\ 3 . 1 - 3 . 2 ,$ , for every integer time horizon $T \geq 1$ , set $N = \lfloor \log _ { 2 } T \rfloor$ and $\begin{array} { r } { \eta = \frac { 1 } { 1 6 C _ { 0 } ( N + 2 ) } } \end{array}$ . Let $( x _ { t } , y _ { t } ) _ { t = 0 } ^ { T }$ be the sequence of iterates generated by the AltOMD updates with stepsize η. Then, the regret defined in Eq. (5) satisfies

$$
\begin{array} { l } { { \displaystyle { \mathrm { R e g } _ { T } \leq 1 6 C _ { 0 } \Lambda _ { 0 } ( N + 2 ) + \left( \frac { 1 2 8 } { 3 } + 4 e ^ { 2 } \right) C _ { 0 } ^ { 2 } } } } \\ { { \displaystyle \qquad \leq 1 6 C _ { 0 } \Lambda _ { 0 } ( \log _ { 2 } T + 2 ) + \left( \frac { 1 2 8 } { 3 } + 4 e ^ { 2 } \right) C _ { 0 } ^ { 2 } . } } \end{array}\tag{16}
$$

Theorem 3.7 applies to arbitrary bilinear games with any compact convex strategy sets satisfying the stated assumptions, including games with non-interior or non-unique equilibria. By Eq. (7), the regret bound also implies $\mathrm { G a p } ( \bar { x } _ { T } , \bar { y } _ { T } ) = O ( \log T / T )$ . To the best of our knowledge, this is the first ${ \cal O } ( \log T )$ regret bound for alternating online mirror descent algorithm with constant stepsize in arbitrary bilinear games. This proves that, surprisingly, with only the alternation, the OMD algorithm achieves logarithmic regret in arbitrary bilinear games.

## 4 AltOMD with the Negative Entropy Regularizers

In this section, we apply the logarithmic regret bound in Theorem 3.7 to negative entropy regularizers on the probability simplex strategy sets, which is the well-known multiplicative weights update algorithm. We first describe the regularizers and the resulting AltOMD updates, then verify the assumptions of the theorem.

Let $\Delta _ { d } : = \{ p \in \mathbb { R } ^ { d } : p _ { i } \geq 0 , \sum _ { i = 1 } ^ { d } p _ { i } = 1 \}$ , and set $\mathcal { X } = \Delta _ { n }$ and $\mathcal { V } = \Delta _ { m }$ . For each player, use the negative entropy regularizer

$$
\psi _ { d } ( p ) : = \sum _ { i = 1 } ^ { d } p _ { i } \log p _ { i } , \qquad p \in \Delta _ { d } ,\tag{17}
$$

with the convention 0 log $0 = 0$ and $\psi _ { d } ( p ) = + \infty$ outside $\Delta _ { d }$ . That is, we set $\psi _ { \mathcal { X } } = \psi _ { n }$ and $\psi _ { \mathcal { Y } } = \psi _ { m }$ The convex conjugates of the negative entropy regularizers are log-sum-exp functions, and their gradients are the softmax maps:

$$
\begin{array} { l l } { F ( u ) = \log \sum _ { i = 1 } ^ { n } e ^ { u _ { i } } , } & { [ \nabla F ( u ) ] _ { i } = \frac { e ^ { u _ { i } } } { \sum _ { \ell = 1 } ^ { n } e ^ { u _ { \ell } } } , } \\ { G ( w ) = \log \sum _ { j = 1 } ^ { m } e ^ { w _ { j } } , } & { [ \nabla G ( w ) ] _ { j } = \frac { e ^ { w _ { j } } } { \sum _ { \ell = 1 } ^ { m } e ^ { w _ { \ell } } } . } \end{array}\tag{18}
$$

We substitute the softmax maps into (AltOMD-Dual) and obtain the multiplicative weights update algorithm:

$$
\begin{array} { r l } & { x _ { k + 1 , i } \propto x _ { k , i } \exp ( - \eta ( A ^ { \top } y _ { k } ) _ { i } ) \quad \forall i = 1 , \ldots , n , } \\ & { y _ { k + 1 , j } \propto y _ { k , j } \exp ( \eta ( A x _ { k + 1 } ) _ { j } ) \quad \forall j = 1 , \ldots , m , } \end{array}\tag{19}
$$

where $x _ { k , i }$ and $y _ { k , j }$ are the i-th and j-th components of the strategies at iteration $k ,$ respectively. The following proposition verifies the assumptions of Theorem 3.7 for the negative entropy regularizers.

Proposition 4.1 (Validity of negative entropy regularizers). If both players use negative entropy regularizers, i.e, $\psi _ { \mathcal { X } } = \psi _ { n }$ and $\psi _ { \mathcal { Y } } = \psi _ { m }$ , then the regularizers are Legendre relative to the afine hull of their respective simplices. Moreover, their convex conjugates F and G are smooth on $\mathbb { R } ^ { n }$ and $\mathbb { R } ^ { m }$ , respectively, and for every integer $s \geq 1$ 2

$$
\frac { 1 } { s ! } \left( \operatorname* { s u p } _ { u } \lVert D ^ { s } F ( u ) \rVert _ { \mathrm { o p } } + \operatorname* { s u p } _ { w } \lVert D ^ { s } G ( w ) \rVert _ { \mathrm { o p } } \right) \leq 4 ^ { s } .
$$

For any strictly positive initial strategies $x _ { 0 } \in \Delta _ { n }$ and $y _ { 0 } \in \Delta _ { m }$ , the initial Bregman divergence satisfies

$$
\Lambda _ { 0 } \leq \log \frac { 1 } { \operatorname* { m i n } _ { 1 \leq i \leq n } x _ { 0 , i } } + \log \frac { 1 } { \operatorname* { m i n } _ { 1 \leq j \leq m } y _ { 0 , j } } < \infty .
$$

Thus, Assumptions 3.1–3.2 hold with $K = 4$

The proofs in this section are provided in Appendix B.1. We apply Theorem 3.7 to obtain the following corollary.

Corollary 4.2 (Alternating multiplicative weights update). Let $A \in \mathbb { R } ^ { m \times n }$ and let $x _ { 0 } \in \Delta _ { n }$ and $y _ { 0 } \in$ $\Delta _ { m }$ be strictly positive initial strategies. Set $C = 3 8 4 \operatorname* { m a x } \{ \| A \| , 1 \}$ and $\Lambda _ { 0 } = \log ( 1 / \operatorname* { m i n } _ { 1 \leq i \leq n } x _ { 0 , i } ) +$ $\log ( 1 / \operatorname* { m i n } _ { 1 \leq j \leq m } y _ { 0 , j } )$ . For every integer $T \geq 1$ , choose the constant step size $\begin{array} { r } { \eta = \frac { 1 } { 1 6 C ( \lfloor \log _ { 2 } T \rfloor + 2 ) } } \end{array}$ and generate a sequence of iterates $f o r \ k = 0 , \ldots , T - 1$ by the updates in Eq. (19). Then, the total regret satisfies $\begin{array} { r } { \mathrm { R e g } _ { T } \leq 1 6 C \Lambda _ { 0 } ( \log _ { 2 } T + 2 ) + \left( \frac { 1 2 8 } { 3 } + 4 e ^ { 2 } \right) C ^ { 2 } } \end{array}$ . In particular, the regret bound for uniform initial strategies is

$$
\mathrm { R e g } _ { T } \leq 1 6 C \log ( n m ) ( \log _ { 2 } T + 2 ) + \left( { \frac { 1 2 8 } { 3 } } + 4 e ^ { 2 } \right) C ^ { 2 } .\tag{20}
$$

This corollary shows that the alternating multiplicative weights update algorithm achieves an $O ( \| A \| \log ( n m ) \log ( T ) + \| A \| ^ { 2 } )$ regret bound. By Eq. (7), the averaged strategies satisfy $\mathrm { G a p } ( \bar { x } _ { T } , \bar { y } _ { T } ) = \bar { O } ( \log T / T )$

## Acknowledgments

The research of Christian Kroer was supported by the Ofice of Naval Research awards N00014-23-1- 2374 and N00014-26-1-2490, and the National Science Foundation award IIS-2238960.

## LLM usage

The mathematical proof in this paper was developed using GPT-5.6 Sol. The Lean proof was formalized using GPT-6 Astra. The authors verified the proofs and revised the manuscript throughout to improve readability and exposition.

## References

Omar Abbadi, Rida Laraki, and Panayotis Mertikopoulos. Constant regret in general games via higher-order optimism. arXiv preprint arXiv:2609.04113, 2026.

James P Bailey, Gauthier Gidel, and Georgios Piliouras. Finite regret and cycles with fixed step-size via alternating gradient descent-ascent. In Conference on Learning Theory, pages 391–407. PMLR, 2020.

Henry Frederick Baker. Alternants and continuous groups. Proceedings of the London Mathematical Society, 2(1):24–47, 1905.

Stefan Banach. Uber homogene Polynome in<sup>¨</sup> $( L ^ { 2 } )$ . Studia Mathematica, 7(1):36–44, 1938.

Neil Burch, Matej Moravcik, and Martin Schmid. Revisiting CFR+ and alternating updates. Journal of Artificial Intelligence Research, 64:429–443, 2019.

Yang Cai, Gabriele Farina, Julien Grand-Cl´ement, Christian Kroer, Chung-Wei Lee, Haipeng Luo, and Weiqiang Zheng. Last-iterate convergence properties of regret-matching algorithms in games. In International Conference on Learning Representations, 2025.

John E Campbell. On a law of combination of operators bearing on the theory of continuous transformation groups. Proceedings of the London Mathematical Society, 1(1):381–390, 1896.

John Edward Campbell. On a law of combination of operators (second paper). Proceedings of the London Mathematical Society, 1(1):14–32, 1897.

Volkan Cevher, Ashok Cutkosky, Ali Kavis, Georgios Piliouras, Stratis Skoulakis, and Luca Viano. Alternation makes the adversary weaker in two-player games. Advances in Neural Information Processing Systems, 36:18263–18290, 2023.

Eugene Borisovich Dynkin. Calculation of the coeficients in the campbell-hausdorf formula. Dokl. Akad. Nauk. SSSR (NS), 57:323–326, 1947.

Chris M Field and Frank W Nijhof. A note on modified hamiltonians for numerical integrations admitting an exact invariant. Nonlinearity, 16(5):1673–1683, 2003.

Gauthier Gidel, Reyhane Askari Hemmat, Mohammad Pezeshki, R´emi Le Priol, Gabriel Huang, Simon Lacoste-Julien, and Ioannis Mitliagkas. Negative momentum for improved game dynamics. In The 22nd international conference on artificial intelligence and statistics, pages 1802–1811. PMLR, 2019.

Eugene Ha. A variation of eichler’s proof of the baker-campbell-hausdorf formula. arXiv preprint arXiv:2608.05053, 2026a.

Junsoo Ha. A logarithmic regret bound for Optimistic Hedge in general-sum games. arXiv preprint arXiv:2609.19677, 2026b.

Ernst Hairer, Christian Lubich, and Gerhard Wanner. Geometric numerical integration: structurepreserving algorithms for ordinary diferential equations, volume 31 of Springer Series in Computational Mathematics. Springer, 2 edition, 2006.

Soumita Hait, Ping Li, Haipeng Luo, and Mengxiao Zhang. Alternating regret for online convex optimization. In Proceedings of Thirty Eighth Conference on Learning Theory, volume 291 of Proceedings of Machine Learning Research, pages 2632–2633. PMLR, 2025.

Felix Hausdorf. Die symbolische exponentialformel in der gruppentheorie. Ber. Verh. Kgl. S¨achs. Ges. Wiss. Leipzig., Math.-phys. Kl., 58:19–48, 1906.

Harald Hofst¨atter. A relatively short self-contained proof of the baker–campbell–hausdorf theorem. Expositiones Mathematicae, 39(1):143–148, 2021.

Nathan Jacobson. Lie algebras. Courier Corporation, 2013.

Jonas E Katona, Xiuyuan Wang, and Andre Wibisono. A symplectic analysis of alternating mirror descent. Journal of Machine Learning Research, 27(44):1–61, 2026.

Boning Li and Longbo Huang. Why alternation speeds up counterfactual regret minimization? arXiv preprint arXiv:2610.21970, 2026.

Mingyang Liu, Gabriele Farina, and Asuman Ozdaglar. Constant individual regret in general games. arXiv preprint arXiv:2608.31166, 2026.

Panayotis Mertikopoulos, Christos Papadimitriou, and Georgios Piliouras. Cycles in adversarial regularized learning. In Proceedings of the twenty-ninth annual ACM-SIAM symposium on discrete algorithms, pages 2703–2717. SIAM, 2018.

Tianlong Nan, Shuvomoy Das Gupta, Garud Iyengar, and Christian Kroer. On the O(1/T) convergence of alternating gradient descent–ascent in bilinear games. In International Conference on Learning Representations, volume 2026, pages 79281–79320, 2026a.

Tianlong Nan, Garud Iyengar, Christian Kroer, and Shuvomoy Das Gupta. AltGDA achieves global O(1/T) ergodic convergence in matrix games. arXiv preprint arXiv:2609.32995, 2026b.

Arkadi Nemirovski. Prox-method with rate of convergence O(1/t) for variational inequalities with Lipschitz continuous monotone operators and smooth convex-concave saddle point problems. SIAM Journal on Optimization, 15(1):229–251, 2004.

Ivan Niven. Formal power series. The American Mathematical Monthly, 76(8):871–889, 1969.

Sasha Rakhlin and Karthik Sridharan. Optimization, learning, and games with predictable sequences. Advances in Neural Information Processing Systems, 26, 2013.

Ashkan Soleymani and Georgios Piliouras. Multiplicative optimism for constant regret in games. arXiv preprint arXiv:2609.21976, 2026.

Michael Sullins and Ian A. Kash. IR-PRM+ with previous-utility prediction: Constant regret and best-iterate convergence. arXiv preprint arXiv:2610.19016, 2026.

Vasilis Syrgkanis, Alekh Agarwal, Haipeng Luo, and Robert E Schapire. Fast convergence of regularized learning in games. Advances in Neural Information Processing Systems, 28, 2015.

Oskari Tammelin, Neil Burch, Michael Johanson, and Michael Bowling. Solving heads-up limit texas hold’em. In IJCAI, volume 15, pages 645–652, 2015.

Yixin Tao and Weiqiang Zheng. Optimal alternating regret for online learning and games. arXiv preprint arXiv:2608.24731, 2026.

Taira Tsuchiya. Sublogarithmic swap regret in multiplayer general-sum games via hybrid regularization. arXiv preprint arXiv:2608.04149, 2026.

Andre Wibisono, Molei Tao, and Georgios Piliouras. Alternating mirror descent for constrained min-max games. Advances in Neural Information Processing Systems, 35:35201–35212, 2022.

Mengxiao Zhang. Minimax alternating regret for the experts problem and online convex optimization. arXiv preprint arXiv:2608.25182, 2026.

## Appendices

## A Full proof of the main theorem

Theorem 3.7 (Logarithmic regret of AltOMD). Under Assumptions 3.1–3.2, for every integer time horizon $T \geq 1$ , set $N = \lfloor \log _ { 2 } T \rfloor$ and $\begin{array} { r } { \eta = \frac { 1 } { 1 6 C _ { 0 } ( N + 2 ) } } \end{array}$ . Let $( x _ { t } , y _ { t } ) _ { t = 0 } ^ { T }$ be the sequence of iterates generated by the AltOMD updates with stepsize η. Then, the regret defined in Eq. (5) satisfies

$$
\begin{array} { r l r } {  { \mathrm { R e g } _ { T } \le 1 6 C _ { 0 } \Lambda _ { 0 } ( N + 2 ) + ( \frac { 1 2 8 } { 3 } + 4 e ^ { 2 } ) C _ { 0 } ^ { 2 } } } \\ & { } & \\ & { } & { \le 1 6 C _ { 0 } \Lambda _ { 0 } ( \log _ { 2 } T + 2 ) + ( \frac { 1 2 8 } { 3 } + 4 e ^ { 2 } ) C _ { 0 } ^ { 2 } . } \end{array}
$$

Before starting the proofs, we first explain some notations here. We extend the conjugates $F$ and $G$ to the product dual space by $F ( z ) = F ( u )$ and $G ( z ) = G ( w )$ , where $\boldsymbol { z } = ( u , w ) \in \mathbb { R } ^ { n } \times \mathbb { R } ^ { m }$ For a smooth Hamiltonian (energy) Q, its Hamiltonian flow $\phi _ { Q } ^ { t }$ is defined, whenever the solution exists, by

$$
\frac { d } { d t } \phi _ { Q } ^ { t } ( z ) = J _ { A } \nabla Q ( \phi _ { Q } ^ { t } ( z ) ) , \qquad \phi _ { Q } ^ { 0 } ( z ) = z ,\tag{21}
$$

where $J _ { A }$ is defined in $\mathrm { E q . ~ ( 4 ) }$ . Recall the one-step AltOMD map from the main text:

$$
\mathcal { M } _ { \eta } = \phi _ { F } ^ { \eta } \circ \phi _ { G } ^ { \eta } .
$$

## A.1 Formal preservation and BCH correction bounds

In this subsection, we construct the BCH correction terms $H _ { r }$ and prove formal preservation. We bound their coeficients for the derivatives in the next subsection. Throughout this subsection, we consider formal power series: series identities hold coeficient by coeficient, without assuming convergence. Throughout the appendix, we write set $( a , b )$ for the set whose elements are a and $b ,$ to distinguish set notation from the Poisson bracket.

Poisson bracket. For two smooth functions $P , Q$ on the product dual space, their Poisson bracket is defined by

$$
\{ P , Q \} : = \nabla P ^ { \top } J _ { A } \nabla Q .\tag{22}
$$

This operation is linear in each argument. Because $J _ { A } ^ { \top } = - J _ { A }$ , we have $\{ P , Q \} = - \{ Q , P \}$

Jacobi identity. For smooth functions $P , Q , S$ , the Poisson bracket satisfies the Jacobi identity:

$$
\{ P , \{ Q , S \} \} + \{ Q , \{ S , P \} \} + \{ S , \{ P , Q \} \} = 0 .\tag{23}
$$

Proof. Since $J _ { A }$ is constant and $J _ { A } ^ { \top } = - J _ { A }$ , the product rule yields

$$
\{ P , \{ Q , S \} \} = - D \{ Q , S \} [ J _ { A } \nabla P ] = D ^ { 2 } S [ J _ { A } \nabla P , J _ { A } \nabla Q ] - D ^ { 2 } Q [ J _ { A } \nabla P , J _ { A } \nabla S ] ,
$$

with all derivatives evaluated at the same point. In the cyclic sum of this identity over $P , Q , S$ , all the terms cancel because $D ^ { 2 } P , D ^ { 2 } Q$ , and $D ^ { 2 } S$ are symmetric tensors. This proves Eq. (23).

Lie derivatives and update flows. For a Hamiltonian $Q$ and a smooth function f, define

$$
L _ { Q } f : = \{ f , Q \} .\tag{24}
$$

For any point $z$ in the product dual space, the chain rule yields

$$
\frac { d } { d t } f ( \phi _ { Q } ^ { t } ( z ) ) \stackrel { \mathrm { b y ~ E q . ~ ( 2 1 ) } } { = } \nabla f ( \phi _ { Q } ^ { t } ( z ) ) ^ { \top } J _ { A } \nabla Q ( \phi _ { Q } ^ { t } ( z ) ) \stackrel { \mathrm { b y ~ E q . ~ ( 2 4 ) } } { = } ( L _ { Q } f ) ( \phi _ { Q } ^ { t } ( z ) ) ,
$$

so $L _ { Q }$ diferentiates $f$ along the flow generated by Q. In particular, $L _ { G }$ and $L _ { F }$ diferentiate along the first and second AltOMD update flows, respectively. For $Q \in \mathrm { s e t } ( F , G )$ , repeated diferentiation gives

$$
e ^ { \eta L _ { Q } } f : = \sum _ { k \geq 0 } \frac { \eta ^ { k } } { k ! } L _ { Q } ^ { k } f = f \circ \phi _ { Q } ^ { \eta } ,
$$

where the right-hand side is interpreted as the formal Taylor series of the composition at $\eta = 0$ without assuming that this series converges. Since $\mathcal { M } _ { \eta } = \phi _ { F } ^ { \eta } \circ \phi _ { G } ^ { \eta }$ , it follows that

$$
f \circ \mathcal { M } _ { \eta } = f \circ \phi _ { F } ^ { \eta } \circ \phi _ { G } ^ { \eta } = e ^ { \eta L _ { G } } e ^ { \eta L _ { F } } f .\tag{25}
$$

More explicitly, Taylor expansion in independent parameters s, t gives the formal identity

$$
f \circ \phi _ { F } ^ { s } \circ \phi _ { G } ^ { t } = \sum _ { \alpha , \beta \geq 0 } \frac { t ^ { \alpha } s ^ { \beta } } { \alpha ! \beta ! } L _ { G } ^ { \alpha } L _ { F } ^ { \beta } f .
$$

Setting $s = t = \eta$ groups terms by total degree, with finitely many terms in each coeficient. This explains both the reversed operator order and the common parameter in Eq. (25); Lemma A.6 proves the corresponding finite Taylor formula with a remainder. This identity applies to formal series of smooth functions by applying each operator to every coeficient.

BCH theorem. We first state the classical BCH theorem in universal form, then apply it to the Hamiltonian Lie derivatives. The accompanying Lean proof of the following theorem follows Ha (2026a, Sections 1–2) and Hofst¨atter (2021, Section 3).

Proposition A.1 (BCH Theorem). Let U, V be free noncommuting symbols over $\mathbb { Q } , i . e . ,$ we form polynomials in $U , V$ with rational coeficients, and the order of the symbols matters. Define the commutator $[ P , Q ] : = P Q - Q P$ . Then, there is a unique formal series Z with zero constant term such that

$$
e ^ { Z } = e ^ { U } e ^ { V } ,\tag{26}
$$

where Z is given by

$$
Z = \log ( e ^ { U } e ^ { V } ) = \sum _ { q \geq 1 } \Phi _ { q } ( U , V ) ,
$$

where each $\Phi _ { q } ( U , V )$ is a homogeneous Lie polynomial of degree q: a rational linear combination of nested commutators containing q factors from U, V (Baker, 1905; Campbell, 1896, 1897; Hausdorf, 1906; Jacobson, 2013). The explicit formula is given by Dynkin’s formula (Dynkin, $ { 1 9 4 7 } )$ : for every $q \geq 1$

$$
\Phi _ { q } ( U , V ) = \frac { 1 } { q } \sum _ { m = 1 } ^ { q } \frac { ( - 1 ) ^ { m - 1 } } { m } \sum _ { \substack { r _ { i } , s _ { i } \geq 0 , r _ { i } + s _ { i } > 0 \atop \sum _ { i = 1 } ^ { m } ( r _ { i } + s _ { i } ) = q } } \frac { [ U ^ { r _ { 1 } } V ^ { s _ { 1 } } \cdot \cdot \cdot U ^ { r _ { m } } V ^ { s _ { m } } ] } { r _ { 1 } ! s _ { 1 } ! \cdot \cdot \cdot r _ { m } ! s _ { m } ! } ,\tag{27}
$$

where $[ W _ { 1 } ] : = W _ { 1 }$ and for $q \geq 2$

$$
[ W _ { 1 } \cdot \cdot \cdot W _ { q } ] : = [ W _ { 1 } , [ W _ { 2 } , \ldots , [ W _ { q - 1 } , W _ { q } ] \ldots ] ] , \qquad W _ { i } \in \mathrm { s e t } ( U , V ) .
$$

Application to the AltOMD update. By Eq. (25), one AltOMD step acts on a smooth function f through $f \circ \mathcal { M } _ { \ u { \eta } } = e ^ { \eta L _ { G } } e ^ { \eta L _ { F } } f$ . Here, products of Lie derivatives denote composition: ${ \cal L } _ { G } { \cal L } _ { F } f = { \cal L } _ { G } ( { \cal L } _ { F } f )$ . We apply Proposition A.1 by substituting

$$
U  \eta L _ { G } , \qquad V  \eta L _ { F } ,
$$

where $\eta$ is a formal scalar parameter that commutes with the operators. By homogeneity of $\Phi _ { q } .$ , it follows that

$$
e ^ { Z ( \eta ) } = e ^ { \eta L _ { G } } e ^ { \eta L _ { F } } .\tag{28}
$$

where

$$
Z ( \eta ) = \sum _ { q \geq 1 } \eta ^ { q } Z _ { q } , \qquad Z _ { q } : = \Phi _ { q } ( L _ { G } , L _ { F } ) ,\tag{29}
$$

where the explicit form of $\Phi _ { q }$ is given by Eq. (27):

$$
Z _ { q } = \Phi _ { q } ( L _ { G } , L _ { F } ) = \frac { 1 } { q } \sum _ { m = 1 } ^ { q } \frac { ( - 1 ) ^ { m - 1 } } { m } \sum _ { \substack { r _ { i , s _ { i } \geq 0 , r _ { i } + s _ { i } \leq 0 } ( 1 \leq i \leq m ) } } \frac { [ ( L _ { G } ) ^ { r _ { 1 } } ( L _ { F } ) ^ { s _ { 1 } } \cdot \cdot ( L _ { G } ) ^ { r _ { m } } ( L _ { F } ) ^ { s _ { m } } ] } { r _ { 1 } ! s _ { 1 } ! \cdot \cdot r _ { m } ! s _ { m } ! } ,\tag{30}
$$

For example,

$$
\begin{array} { r l } & { Z _ { 1 } = [ L _ { G } ] + [ L _ { F } ] = L _ { G } + L _ { F } , } \\ & { Z _ { 2 } = \underbrace { \frac { 1 } { 2 } \left( \frac { \left[ L _ { G } L _ { G } \right] } { 2 ! } + \left[ L _ { G } L _ { F } \right] + \frac { \left[ L _ { F } L _ { F } \right] } { 2 ! } \right) } _ { \begin{array} { l } { m = 1 } \\ { 1 } \end{array} } - \underbrace { \frac { 1 } { 4 } \left( \left[ L _ { G } L _ { G } \right] + \left[ L _ { G } L _ { F } \right] + \left[ L _ { F } L _ { G } \right] + \left[ L _ { F } L _ { F } \right] \right) } _ { \begin{array} { l } { m = 2 } \\ { 1 } \end{array} } } \\ & { \quad = \frac { 1 } { 4 } [ L _ { G } L _ { F } ] - \frac { 1 } { 4 } [ L _ { F } L _ { G } ] = \frac { 1 } { 2 } [ L _ { G } , L _ { F } ] . } \end{array}
$$

Here, we use $[ L _ { G } L _ { G } ] = [ L _ { G } , L _ { G } ] = 0 \mathrm { a n d } \left[ L _ { F } L _ { F } \right] = [ L _ { F } , L _ { F } ] = 0 .$

Equivalently,

$$
Z _ { q } = \frac { 1 } { q } \sum _ { \substack { \omega = ( Q _ { 1 } , \ldots , Q _ { q } ) \in \mathrm { s e t } ( G , F ) ^ { q } } } c _ { \omega } [ L _ { Q _ { 1 } } , [ L _ { Q _ { 2 } } , \ldots , [ L _ { Q _ { q - 1 } } , L _ { Q _ { q } } ] \ldots ] ] ,\tag{31}
$$

where

$$
c _ { \omega } = \sum _ { m = 1 } ^ { q } \frac { ( - 1 ) ^ { m - 1 } } { m } \sum _ { \substack { r _ { i } , s _ { i } \geq 0 , r _ { i } + s _ { i } > 0 ( 1 \leq i \leq m ) } } \frac { 1 } { r _ { 1 } ! s _ { 1 } ! \cdot \cdot \cdot r _ { m } ! s _ { m } ! } .\tag{32}
$$

Complete construction of the modified Hamiltonian (energy). To construct the modified Hamiltonian energy in Eq. (12), we construct each correction term $H _ { q - 1 }$ corresponding to the operator $Z _ { q }$ in Eq. (31).

Recall that, for two operators U, V, the commutator is defined by $[ U , V ] = U V - V U$ . By the Jacobi identity, we have

$$
[ { \cal L } _ { P } , { \cal L } _ { Q } ] f = \{ \{ f , Q \} , P \} - \{ \{ f , P \} , Q \} = \{ f , \{ Q , P \} \} = \{ f , - \{ P , Q \} \} = - { \cal L } _ { \{ P , Q \} } f .\tag{33}
$$

For any $Q _ { 1 } , \ldots , Q _ { q } \in \mathrm { s e t } ( G , F )$ , this relation implies by induction that

$$
[ L _ { Q _ { 1 } } , [ L _ { Q _ { 2 } } , \dots , [ L _ { Q _ { q - 1 } } , L _ { Q _ { q } } ] \dots ] ] = ( - 1 ) ^ { q - 1 } L _ { \{ Q _ { 1 } , \{ Q _ { 2 } , \dots , \{ Q _ { q - 1 } , Q _ { q } \} \dots \} }  .\tag{34}
$$

For every integer $q \geq 1$ , we define the correction terms by

$$
H _ { q - 1 } : = \frac { ( - 1 ) ^ { q - 1 } } { q } \sum _ { \substack { \omega = ( Q _ { 1 } , \ldots , Q _ { q } ) \in \mathrm { s e t } ( G , F ) ^ { q } } } c _ { \omega } \{ Q _ { 1 } , \{ Q _ { 2 } , \ldots , \{ Q _ { q - 1 } , Q _ { q } \} \ldots \} \} ,\tag{35}
$$

where $c _ { \omega }$ are defined by Eq. (32). For $q = 1$ , the nested bracket is interpreted as $Q _ { 1 }$ thus $H _ { 0 } = F + G$ By Eqs. (31) and (34) and the fact that the $L _ { Q }$ is linear in $Q { \mathrm { : } }$

$$
{ \cal L } _ { \sum _ { i } a _ { i } Q _ { i } } f = \left\{ f , \sum _ { i } a _ { i } Q _ { i } \right\} = \sum _ { i } a _ { i } \{ f , Q _ { i } \} = \left( \sum _ { i } a _ { i } L _ { Q _ { i } } \right) f ,
$$

we obtain

$$
Z _ { q } = L _ { H _ { q - 1 } } , \quad \forall ~ q \geq 1 .\tag{36}
$$

The formal modified Hamiltonian energy is therefore

$$
\widehat H _ { \eta } : = \sum _ { r \ge 0 } \eta ^ { r } H _ { r } .\tag{37}
$$

Again, because $L _ { Q }$ is linear in $Q ,$ we have

$$
Z ( \eta ) = \eta L _ { \widehat { H } _ { \eta } } .\tag{38}
$$

For example,

$$
H _ { 0 } = F + G , \quad Z _ { 1 } = L _ { H _ { 0 } }
$$

$$
H _ { 1 } = \frac { 1 } { 2 } \{ F , G \} , \quad Z _ { 2 } = L _ { H _ { 1 } } .\tag{39}
$$

Formal preservation of the modified energy.

Lemma 3.4 (Formal preservation of $\widehat { H } _ { \eta } )$ . Let F and G be smooth, let $H _ { r }$ be the BCH correction terms constructed in Appendix A.1, and let $\widehat { H } _ { \eta }$ be the formal modified energy defined by $E q .$ (12). Then, $\widehat { H } _ { \eta }$ satisfies

$$
\widehat { H } _ { \eta } \big ( \mathcal { M } _ { \eta } ( z ) \big ) = \widehat { H } _ { \eta } ( z ) \qquad f o r \ e v e r y \ z \in \mathbb { R } ^ { n } \times \mathbb { R } ^ { m } ,
$$

as an identity of formal power series in η.

Proof of Lemma $\it 3 . 4$ . By Eq. (38), we have $Z ( \eta ) = \eta L _ { \widehat { H } _ { n } }$ . By BCH Theorem, Eq. (28) yields

$$
e ^ { \eta L _ { G } } e ^ { \eta L _ { F } } = e ^ { \eta L _ { \widehat { H } _ { \eta } } } .
$$

By antisymmetry of the Poisson bracket,

$$
\widehat { L } _ { \widehat { H } _ { \eta } } \widehat { H } _ { \eta } = \{ \widehat { H } _ { \eta } , \widehat { H } _ { \eta } \} = ( \nabla \widehat { H } _ { \eta } ) ^ { \top } J _ { A } ( \nabla \widehat { H } _ { \eta } ) = \frac { 1 } { 2 } ( \nabla \widehat { H } _ { \eta } ) ^ { \top } J _ { A } ( \nabla \widehat { H } _ { \eta } ) + \frac { 1 } { 2 } ( \nabla \widehat { H } _ { \eta } ) ^ { \top } J _ { A } ^ { \top } ( \nabla \widehat { H } _ { \eta } ) = 0 .
$$

Therefore,

$$
e ^ { \eta L _ { G } } e ^ { \eta L _ { F } } \widehat { H } _ { \eta } = e ^ { \eta L _ { \widehat { H } _ { \eta } } } \widehat { H } _ { \eta } = \widehat { H } _ { \eta } + \sum _ { k \geq 1 } \frac { \eta ^ { k } } { k ! } L _ { \widehat { H } _ { \eta } } ^ { k } \widehat { H } _ { \eta } = \widehat { H } _ { \eta } ,\tag{40}
$$

where the second equality uses the formal exponential series $\begin{array} { r } { e ^ { \eta L _ { \widehat { H } _ { \eta } } } = \sum _ { k \geq 0 } ( \eta L _ { \widehat { H } _ { \eta } } ) ^ { k } / k ! } \end{array}$ . By Eqs. (25) and (40), we have $\widehat { H } _ { \eta } \circ \mathcal { M } _ { \eta } = \widehat { H } _ { \eta }$ , which proves the lemma. □

Coeficient bounds for the correction terms. The derivative estimates in the next subsection require a bound on the coeficients in a representation of each $H _ { r }$ by nested Poisson brackets. The formula Eq. (35) reduces this task to a bound on $\begin{array} { r } { \frac { 1 } { q } \sum _ { \omega } | c _ { \omega } | } \end{array}$ , where $q = r + 1$ . We estimate this sum directly from the formal logarithm.

For a finite linear combination P of distinct labelled operator words, let $a _ { \omega }$ denote the coeficient of the word $\omega .$ . Define the norm $\begin{array} { r } { \| P \| _ { 1 } : = \sum _ { \omega } | a _ { \omega } | } \end{array}$ , the sum of the absolute values of the coeficients $a _ { \omega }$ . In particular, $\begin{array} { r } { \| Z _ { q } \| _ { 1 } = \sum _ { \omega \in \mathrm { s e t } ( G , F ) ^ { q } } | c _ { \omega } | } \end{array}$ . This is a norm on formal word expressions, not a norm on the evaluated diferential operators; the empty word represents the identity operator I. Let $[ \eta ^ { q } ] f ( \eta )$ denote the coeficient of $\eta ^ { q }$ in a formal series $f ( \eta )$ . We use the same coeficient notation for series in other variables, including those with function or operator coeficients. Set

$$
E ( \eta ) : = e ^ { \eta L _ { G } } e ^ { \eta L _ { F } } - I = \sum _ { \ell \ge 1 } E _ { \ell } \eta ^ { \ell } , \qquad E _ { \ell } : = \sum _ { j = 0 } ^ { \ell } \frac { L _ { G } ^ { j } L _ { F } ^ { \ell - j } } { j ! ( \ell - j ) ! } .
$$

All coeficients of $E _ { \ell }$ are nonnegative, and the binomial theorem yields

$$
\| E _ { \ell } \| _ { 1 } = \sum _ { j = 0 } ^ { \ell } { \frac { 1 } { j ! ( \ell - j ) ! } } = { \frac { 2 ^ { \ell } } { \ell ! } } \leq 2 ^ { \ell } .\tag{41}
$$

For any two such polynomials P and $Q ,$ this norm satisfies

$$
\| P Q \| _ { 1 } \leq \| P \| _ { 1 } \| Q \| _ { 1 } .
$$

To see why, index the coeficients of P and $Q$ by labelled words $\sigma$ and $\tau ,$ respectively. The product of two words is the concatenation of their lists of Lie derivatives. The coeficient contributed by this product is $a _ { \sigma } b _ { \tau }$ , and the sum of the absolute values of their coeficients is $\begin{array} { r } { \sum _ { \sigma , \tau } | a _ { \sigma } b _ { \tau } | = } \end{array}$ $\begin{array} { r } { ( \sum _ { \sigma } | a _ { \sigma } | ) ( \sum _ { \tau } | b _ { \tau } | ) } \end{array}$ . By the triangle inequality, this sum cannot increase when we collect terms with the same word. The formal logarithm has the expansion

$$
Z ( \eta ) = \log ( I + E ( \eta ) ) = \sum _ { k \geq 1 } \frac { ( - 1 ) ^ { k + 1 } } { k } E ( \eta ) ^ { k } .
$$

The powers $E ( \eta ) ^ { k }$ contribute to the coeficient of $\eta ^ { q }$ as follows:

$$
Z _ { q } = \sum _ { k = 1 } ^ { q } \frac { ( - 1 ) ^ { k + 1 } } { k } \sum _ { \substack { \ell _ { 1 } + \dots + \ell _ { k } = q } } E _ { \ell _ { 1 } } \cdot \cdot \cdot E _ { \ell _ { k } } .
$$

The triangle inequality and the bound $\lVert P Q \rVert _ { 1 } \leq \lVert P \rVert _ { 1 } \lVert Q \rVert _ { 1 }$ imply

$$
\begin{array} { r l r } { \displaystyle | | Z _ { q } | | _ { 1 } \le \sum _ { k = 1 } ^ { q } \displaystyle \frac { 1 } { k } \sum _ { \ell _ { 1 } + \cdots + \ell _ { k } = q } | | E _ { \ell _ { 1 } } | | _ { 1 } \cdots | | E _ { \ell _ { k } } | | _ { 1 } = [ \eta ^ { q } ] \sum _ { k \ge 1 } \displaystyle \frac { 1 } { k } \left( \sum _ { \ell \ge 1 } | | E _ { \ell } | | _ { 1 } \eta ^ { \ell } \right) ^ { k } } & & { } & \\ & { } & { \le [ \eta ^ { q } ] \displaystyle \sum _ { k \ge 1 } \displaystyle \frac { 1 } { k } \left( \sum _ { \ell \ge 1 } 2 ^ { \ell } \eta ^ { \ell } \right) ^ { k } \qquad \mathrm { ( b y ~ E q . ~ } } \\ & { } & { = [ \eta ^ { q } ] \log \left( \displaystyle \frac { 1 - 2 \eta } { 1 - 4 \eta } \right) = \displaystyle \frac { 4 ^ { q } - 2 ^ { q } } { q } \le 4 ^ { q } , } \end{array}\tag{41)}
$$

(42)

where the last line follows from $\begin{array} { r } { \sum _ { \ell > 1 } 2 ^ { \ell } \eta ^ { \ell } = 2 \eta / ( 1 - 2 \eta ) \ \mathrm { a n d } \ - \log ( 1 - \eta ) = \sum _ { k > 1 } \eta ^ { k } / k . } \end{array}$

Together, Eqs. (31) and (42) bound the coeficients in the right-nested commutator representation of $Z _ { q }$ . Before like terms are collected, the sum of the absolute values of the coeficients in this representation of $Z _ { q }$ is at most

$$
\frac { 1 } { q } \sum _ { \omega \in \mathrm { s e t } ( G , F ) ^ { q } } | c _ { \omega } | = \frac { 1 } { q } \| Z _ { q } \| _ { 1 } \leq 4 ^ { q } .\tag{43}
$$

Lemma A.2 (Coeficient bound for the correction terms). For every integer $r \geq 0 , H _ { r }$ can be written as a linear combination of right-nested Poisson brackets $o f$ length $r + 1$ in $F , G$ , with the sum of the absolute values of the coeficients at most $4 ^ { r + 1 }$ . For $r = 0$ , length-one brackets mean F and G themselves.

Proof. With $q = r + 1$ , Eq. (35) is already a representation of $H _ { r }$ by right-nested Poisson brackets of length $r + 1$ . The common sign $( - 1 ) ^ { q - 1 }$ does not afect the absolute coeficient sum, which is $\begin{array} { r } { \frac { 1 } { q } \sum _ { \omega \in \mathrm { s e t } ( G , F ) ^ { q } } | c _ { \omega } | } \end{array}$ and is at most $4 ^ { q } = 4 ^ { r + 1 }$ by Eq. (43). For $r = 0$ , the length-one convention gives ${ \dot { H } } _ { 0 } = F { \dot { + } } { \dot { G } } .$ □

Truncation and formal energy error. For every integer $N \geq 0$ , recall from the main text that the truncated modified energy is $\begin{array} { r } { \widehat { H } _ { \eta } ^ { [ N ] } = \sum _ { r = 0 } ^ { N } \eta ^ { r } H _ { r } } \end{array}$ . In the subsequent analysis, for every integer $k \geq 0 , O ( \eta ^ { k } )$ denotes a formal series of functions or operators whose coeficients below degree k all vanish. The omitted terms satisfy

$$
\widehat H _ { \eta } - \widehat H _ { \eta } ^ { [ N ] } = \sum _ { r \ge N + 1 } \eta ^ { r } H _ { r } = O ( \eta ^ { N + 1 } ) .
$$

Since $e ^ { \eta L _ { G } } e ^ { \eta L _ { F } } - I = ( I + \eta L _ { G } + O ( \eta ^ { 2 } ) ) ( I + \eta L _ { F } + O ( \eta ^ { 2 } ) ) - I = O ( \eta )$ , we have

$$
\begin{array} { r l } & { \widehat { H } _ { \eta } ^ { [ N ] } \circ \mathcal { M } _ { \eta } - \widehat { H } _ { \eta } ^ { [ N ] } = ( e ^ { \eta L _ { G } } e ^ { \eta L _ { F } } - I ) \widehat { H } _ { \eta } ^ { [ N ] } } \\ & { \qquad = ( e ^ { \eta L _ { G } } e ^ { \eta L _ { F } } - I ) \left( \widehat { H } _ { \eta } - \sum _ { r \geq N + 1 } \eta ^ { r } H _ { r } \right) } \\ & { \qquad = - ( e ^ { \eta L _ { G } } e ^ { \eta L _ { F } } - I ) \sum _ { r \geq N + 1 } \eta ^ { r } H _ { r } } \\ & { \qquad = O ( \eta ^ { N + 2 } ) . } \end{array}\tag{by Lemma 3.4}
$$

(44)

Therefore, every Taylor coeficient of the one-step energy change through degree $N + 1$ is zero. In the next subsection, we use finite Taylor argument by combining this cancellation with the derivative bounds to control the actual change in $\bigcap _ { \eta } ^ { \bigcup } [ N ]$ over one iteration.

## A.2 Derivative bounds and finite modified-energy estimates

We first bound the derivatives of nested Poisson brackets and the correction terms $H _ { r } ,$ then use finite Taylor expansions to control the truncated modified energy. For the estimates in this subsection, write $J = J _ { A }$ and $\| B \| = \| B \| _ { \mathrm { o p } }$ for the spectral norm of a matrix B. For a smooth scalar- or vector-valued map f on the product dual space, set

$$
\| D ^ { s } f \| _ { \infty } : = \operatorname* { s u p } _ { z \in \mathbb { R } ^ { n } \times \mathbb { R } ^ { m } } \| D ^ { s } f ( z ) \| _ { \mathrm { o p } } , \qquad s \geq 1 ,
$$

using Euclidean norms on the inputs and output. For $s = 0 , \| f \| _ { \infty }$ is the supremum of the output norm, with absolute value for scalar-valued f. In this notation, Assumption 3.1 states that $\| D ^ { s } F \| _ { \infty } + \| D ^ { s } G \| _ { \infty } \leq K ^ { s } s !$ for every integer $s \geq 1$ . The Taylor formulas below use a variable stepsize h and are applied with $h = \eta$ . We use the formal identities from Appendix A.1 to identify Taylor coeficients.

Lemma 3.6. Under Assumption 3.1, set $C _ { 0 } = 2 4 \operatorname* { m a x } \lbrace \| J _ { A } \| _ { \mathrm { o p } } , 1 \rbrace K ^ { 2 }$ . For every integer $N \geq 0$ and stepsize $\begin{array} { r } { 0 < \eta \le \frac { 1 } { 1 6 C _ { 0 } ( N + 2 ) } } \end{array}$ , the truncated modified energy $\widehat { H } _ { \eta } ^ { [ N ] }$ defined in Eq. (14) satisfies, uniformly over all dual coordinates z,

$$
\begin{array} { c } { { \displaystyle \left| \widehat { H } _ { \eta } ^ { [ N ] } ( z ) - H _ { 0 } ( z ) \right| \leq \frac { 6 4 } { 3 } C _ { 0 } ^ { 2 } \eta , } } \\ { { \displaystyle \left| \widehat { H } _ { \eta } ^ { [ N ] } ( { \mathcal { M } } _ { \eta } ( z ) ) - \widehat { H } _ { \eta } ^ { [ N ] } ( z ) \right| \leq 1 6 e ^ { 2 } C _ { 0 } ^ { 2 } \eta 4 ^ { - ( N + 1 ) } . } } \end{array}
$$

The functions $H _ { r }$ and the constant $C _ { 0 }$ are independent of $\eta , N$ , and the horizon.

We first bound the derivatives of a single Lie derivative, then apply this bound repeatedly to nested Poisson brackets. Recall that we use $[ t ^ { s } ] b ( t )$ to denote the coeficient of $t ^ { s }$ in a formal scalar power series $b ( t )$ , and $\mathrm { u s e \preceq }$ to denote coeficient-wise inequality between such series.

Lemma A.3 (Coeficient bound for a Lie derivative). Under Assumption 3.1, let $Q \in \operatorname { s e t } ( F , G )$ , let $f : \mathbb { R } ^ { n } \times \mathbb { R } ^ { m } \to$ R be smooth, and set $v ( t ) = \| J \| K ( 1 - K t ) ^ { - 2 }$ . If b(t) is a formal power series with nonnegative coeficients satisfying

$$
\frac { \| D ^ { s } f \| _ { \infty } } { s ! } \leq [ t ^ { s } ] b ( t ) , \qquad \forall s \geq 1 ,
$$

then

$$
\frac { \| D ^ { s } ( L _ { Q } f ) \| _ { \infty } } { s ! } \leq [ t ^ { s } ] \bigl ( v ( t ) b ^ { \prime } ( t ) \bigr ) , \qquad \forall s \geq 0 .
$$

Proof. Because J is a constant matrix, the derivatives of the Hamiltonian vector field satisfy $\| D ^ { \ell } ( J \nabla Q ) \| _ { \infty } \leq \| J \| \| D ^ { \ell } ( \nabla Q ) \| _ { \infty } = \| J \| \| D ^ { \ell + 1 } Q \| _ { \infty }$ . For every integer $\ell \geq 0$ , Assumption 3.1 at order $\ell + 1$ implies

$$
\frac { \| D ^ { \ell } ( J \nabla Q ) \| _ { \infty } } { \ell ! } \leq \frac { \| J \| \| D ^ { \ell + 1 } Q \| _ { \infty } } { \ell ! } \leq \| J \| \frac { K ^ { \ell + 1 } ( \ell + 1 ) ! } { \ell ! } = \| J \| K ( \ell + 1 ) K ^ { \ell } .\tag{45}
$$

The coeficients of $v ( t )$ bound the normalized derivative norms of J∇Q:

$$
\begin{array} { r } { \sum _ { \ell \geq 0 } \frac { \| D ^ { \ell } ( J \nabla Q ) \| _ { \infty } } { \ell ! } t ^ { \ell } \preceq \| J \| K \sum _ { \ell \geq 0 } ( \ell + 1 ) ( K t ) ^ { \ell } = \frac { \| J \| K } { ( 1 - K t ) ^ { 2 } } = v ( t ) , } \end{array}\tag{46}
$$

where the first equality follows by diferentiating the formal geometric-series identity $\begin{array} { r } { \sum _ { \ell \geq 0 } K ^ { \ell } t ^ { \ell + 1 } = } \end{array}$ $t / ( 1 - K t )$ :

$$
\sum _ { \ell \geq 0 } ( \ell + 1 ) ( K t ) ^ { \ell } = { \frac { d } { d t } } \left( { \frac { t } { 1 - K t } } \right) = { \frac { 1 } { ( 1 - K t ) ^ { 2 } } } .
$$

Hence, Eq. (46) tells us that

$$
\frac { \| D ^ { \ell } ( J \nabla Q ) \| _ { \infty } } { \ell ! } \leq [ t ^ { \ell } ] v ( t ) , \qquad \forall \ell \geq 0 .\tag{47}
$$

Because $L _ { Q } f = \{ f , Q \} = ( J \nabla Q ) \cdot \nabla f$ and J is a constant matrix, by the higher-order product rule, we have

$$
\| D ^ { s } ( L _ { Q } f ) \| _ { \infty } \leq \sum _ { \ell = 0 } ^ { s } { \binom { s } { \ell } } \| D ^ { \ell } ( J \nabla Q ) \| _ { \infty } \| D ^ { s - \ell + 1 } f \| _ { \infty } .\tag{48}
$$

We divide both sides of Eq. (48) by s! and use $\textstyle { \binom { s } { \ell } } / s ! = 1 / ( \ell ! ( s - \ell ) ! )$ . For every integer $s \geq 0$ , we obtain

$$
\begin{array} { r l } { \frac { \displaystyle \| D ^ { s } ( L _ { Q } f ) \| _ { \infty } } { s ! } \le \sum _ { \ell = 0 } ^ { s } \frac { \displaystyle \| D ^ { \ell } ( J \nabla Q ) \| _ { \infty } } { \ell ! } \frac { \displaystyle \| D ^ { s - \ell + 1 } f \| _ { \infty } } { ( s - \ell ) ! } } & { } \\ { \le \sum _ { \ell = 0 } ^ { s } \big ( [ \ell ^ { \ell } ] v ( t ) \big ) ( s - \ell + 1 ) \big ( [ \ell ^ { s - \ell + 1 } ] b ( t ) \big ) } & { } \\ { \displaystyle } & { = \sum _ { \ell = 0 } ^ { s } \big ( [ \ell ^ { \ell } ] v ( t ) \big ) \big ( [ \ell ^ { s - \ell } ] b ^ { \prime } ( t ) \big ) } \\ { } & { = [ \ell ^ { \ell } ] \big ( v ( t ) b ^ { \prime } ( t ) \big ) , } \end{array}\tag{49}
$$

where the second inequality uses Eq. (47) and the assumed coeficient bounds on the derivatives of $f ,$ , the first equality follows from $[ t ^ { k } ] b ^ { \prime } ( t ) = ( k + 1 ) [ t ^ { k + 1 } ] b ( t )$ for every integer $k \geq 0$ , and the last equality is because the coeficient of $t ^ { s }$ in the product $v ( t ) b ^ { \prime } ( t )$ is contributed by $[ t ^ { \ell } ] v ( t )$ and $[ t ^ { s - \ell } ] b ^ { \prime } ( t )$ for some $\ell \in \{ 0 , 1 , \ldots , s \}$ □

We now iterate Lemma A.3 to bound the derivatives of nested Poisson brackets.

Lemma A.4 (Derivative bound for nested brackets). Assume Assumption 3.1 holds. Let $q \geq 2$ be an integer and $Q _ { 1 } , \dots , Q _ { q } \in \mathrm { s e t } ( F , G )$ . For every integer $s \geq 0$ , we have

$$
\frac { 1 } { ( q + s - 1 ) ! } \| D ^ { s } \{ Q _ { 1 } , \{ Q _ { 2 } , \dots , \{ Q _ { q - 1 } , Q _ { q } \} \dots \} \} \| _ { \infty } \le C _ { 0 } ^ { q + s - 1 } .\tag{50}
$$

For $q = 1$ and $s \geq 1$ , the derivative bounds are reduced to Assumption 3.1.

Proof. Let $v ( t )$ be the series defined in Lemma A.3. Define

$$
b _ { 1 } ( t ) = ( 1 - K t ) ^ { - 1 } = \sum _ { \ell \geq 0 } K ^ { \ell } t ^ { \ell } .\tag{51}
$$

By Assumption 3.1, for every $Q \in \operatorname { s e t } ( F , G )$ and integer $s \geq 1$

$$
\frac { \| D ^ { s } Q \| _ { \infty } } { s ! } \leq K ^ { s } = [ t ^ { s } ] b _ { 1 } ( t ) .
$$

The derivative of $b _ { 1 } ( t )$ satisfies

$$
b _ { 1 } ^ { \prime } ( t ) = \frac { d } { d t } \left( 1 + \sum _ { \ell \geq 1 } K ^ { \ell } t ^ { \ell } \right) = \sum _ { \ell \geq 1 } \ell K ^ { \ell } t ^ { \ell - 1 } = \sum _ { \ell \geq 0 } ( \ell + 1 ) K ^ { \ell + 1 } t ^ { \ell } .
$$

In particular, for every integer $s \geq 0$

$$
\frac { \| D ^ { s + 1 } Q \| _ { \infty } } { s ! } \le ( s + 1 ) K ^ { s + 1 } = [ t ^ { s } ] b _ { 1 } ^ { \prime } ( t ) ,\tag{52}
$$

where the first inequality uses Assumption 3.1. Since $\{ Q , f \} = - L _ { Q } f .$ , the nested bracket satisfies

$$
\left\{ Q _ { 1 } , \left\{ Q _ { 2 } , \ldots , \left\{ Q _ { q - 1 } , Q _ { q } \right\} \ldots \right\} \right\} = ( - 1 ) ^ { q - 1 } L _ { Q _ { 1 } } \cdot \cdot \cdot L _ { Q _ { q - 1 } } Q _ { q } .
$$

We start with $f = Q _ { q }$ and apply Lemma $\mathrm { A . 3 }$ successively with $Q = Q _ { q - 1 } , \ldots , Q _ { 1 }$ . Define the corresponding scalar series recursively by

$$
b _ { j + 1 } ( t ) : = v ( t ) b _ { j } ^ { \prime } ( t ) , \qquad j = 1 , \ldots , q - 1 ,
$$

which has the explicit form

$$
b _ { q } ( t ) = \left( v ( t ) \frac { d } { d t } \right) ^ { q - 1 } b _ { 1 } ( t ) = \| J \| ^ { q - 1 } K ^ { 2 q - 2 } \left[ \prod _ { \ell = 0 } ^ { q - 2 } ( 3 \ell + 1 ) \right] ( 1 - K t ) ^ { - ( 3 q - 2 ) } .\tag{53}
$$

Indeed, if $b _ { j } ( t ) = c _ { j } ( 1 - K t ) ^ { - ( 3 j - 2 ) }$ , then

$$
v ( t ) b _ { j } ^ { \prime } ( t ) = \| J \| K ^ { 2 } ( 3 j - 2 ) c _ { j } ( 1 - K t ) ^ { - ( 3 j + 1 ) } .
$$

Starting from $c _ { 1 } = 1$ , induction gives the prefactor in Eq. (53), with the empty product interpreted as 1. We show that $b _ { q }$ is the correct scalar series corresponding to the nested bracket by induction on $q .$ For $q = 2$ , we can apply Lemma A.3 with $f = Q _ { 2 } , Q = Q _ { 1 }$ , and $b = b _ { 1 }$ . Since $\{ Q _ { 1 } , Q _ { 2 } \} = - L _ { Q _ { 1 } } Q _ { 2 }$ for every integer $s \geq 0$

$$
\frac { \| D ^ { s } \{ Q _ { 1 } , Q _ { 2 } \} \| _ { \infty } } { s ! } = \frac { \| D ^ { s } ( L _ { Q _ { 1 } } Q _ { 2 } ) \| _ { \infty } } { s ! } \le [ t ^ { s } ] \big ( v ( t ) b _ { 1 } ^ { \prime } ( t ) \big ) = [ t ^ { s } ] b _ { 2 } ( t ) .\tag{54}
$$

Now, let $q \ \geq \ 3$ and assume the bound holds for all brackets with $q \mathrm { ~ - ~ } 1$ factors. Set $f \ =$ $\left\{ Q _ { 2 } , \left\{ Q _ { 3 } , \ldots , \left\{ Q _ { q - 1 } , Q _ { q } \right\} \ldots \right\} \right\}$ . The induction hypothesis states that

$$
\frac { \| D ^ { s } f \| _ { \infty } } { s ! } \leq [ t ^ { s } ] b _ { q - 1 } ( t ) , \qquad \forall s \geq 0 .
$$

For this $f ,$ Lemma A.3 with $Q = Q _ { 1 }$ and $b = b _ { q } .$ <sub>−1</sub> implies, for every integer $s \geq 0$

$$
\frac { \| D ^ { s } \{ Q _ { 1 } , \{ Q _ { 2 } , \dots , \{ Q _ { q - 1 } , Q _ { q } \} \dots \} \} \| _ { \infty } } { s ! } = \frac { \| D ^ { s } ( L _ { Q _ { 1 } } f ) \| _ { \infty } } { s ! } \le [ \iota ^ { s } ] \big ( v ( t ) b _ { q - 1 } ^ { \prime } ( t ) \big ) = [ \iota ^ { s } ] b _ { q } ( t ) .\tag{55}
$$

It remains to compute $[ t ^ { s } ] b _ { q } ( t )$ . To do so, first note that

$$
[ t ^ { s } ] ( 1 - K t ) ^ { - ( 3 q - 2 ) } = \frac { 1 } { s ! } \left. \frac { d ^ { s } } { d t ^ { s } } ( 1 - K t ) ^ { - ( 3 q - 2 ) } \right| _ { t = 0 } = K ^ { s } \frac { ( 3 q + s - 3 ) ! } { s ! ( 3 q - 3 ) ! } = K ^ { s } \binom { 3 q + s - 3 } { s } .
$$

Also, since $3 \ell + 1 \leq 3 ( \ell + 1 )$ for every $\ell \geq 0$ , we have $\begin{array} { r } { \prod _ { \ell = 0 } ^ { q - 2 } ( 3 \ell + 1 ) \le \prod _ { \ell = 0 } ^ { q - 2 } 3 ( \ell + 1 ) = 3 ^ { q - 1 } ( q - 1 ) ! } \end{array}$ By $\operatorname { E q }$ . (53) and the bounds above, we have

$$
[ t ^ { s } ] b _ { q } ( t ) \le \| J \| ^ { q - 1 } K ^ { 2 q - 2 } 3 ^ { q - 1 } ( q - 1 ) ! K ^ { s } \binom { 3 q + s - 3 } s .\tag{56}
$$

We use $\operatorname { E q } .$ (56) in $\operatorname { E q } .$ . (55) to obtain

$$
\| D ^ { s } \{ Q _ { 1 } , \{ Q _ { 2 } , \ldots , \{ Q _ { q - 1 } , Q _ { q } \} \ldots \} \} \| _ { \infty } \leq \| J \| ^ { q - 1 } K ^ { 2 q - 2 + s } 3 ^ { q - 1 } ( q - 1 ) ! s ! \binom { 3 q + s - 3 } s .\tag{57}
$$

The binomial theorem yields

$$
\binom { 3 q + s - 3 } { s } \leq \sum _ { j = 0 } ^ { 3 q + s - 3 } \binom { 3 q + s - 3 } { j } = 2 ^ { 3 q + s - 3 } .\tag{58}
$$

Since $\binom { q + s - 1 } { s } \geq 1$ , we also have

$$
( q - 1 ) ! s ! = { \frac { ( q + s - 1 ) ! } { \binom { q + s - 1 } { s } } } \leq ( q + s - 1 ) ! .\tag{59}
$$

The bounds in Eqs. (58) and (59) reduce Eq. (57) to

$$
\begin{array} { r l } & { \| D ^ { s } \{ Q _ { 1 } , \{ Q _ { 2 } , \dots , \{ Q _ { q - 1 } , Q _ { q } \} \dots \} \} \| _ { \infty } \leq \| J \| ^ { q - 1 } K ^ { 2 q - 2 + s } 3 ^ { q - 1 } 2 ^ { 3 q + s - 3 } ( q + s - 1 ) ! } \\ & { \qquad = ( 2 4 \| J \| K ^ { 2 } ) ^ { q - 1 } ( 2 K ) ^ { s } ( q + s - 1 ) ! } \\ & { \qquad \leq C _ { 0 } ^ { q + s - 1 } ( q + s - 1 ) ! , } \end{array}\tag{60}
$$

where the last inequality follows from $C _ { 0 } = 2 4 \operatorname* { m a x } \{ \| J \| , 1 \} K ^ { 2 }$ and $K \geq 1$ , which give $2 4 \| J \| K ^ { 2 } \le C _ { 0 }$ and $2 K \le 2 4 K ^ { 2 } \le C _ { 0 }$ □

We use Lemma A.4 to control the derivatives of the coeficients $H _ { r }$ of the modified energy along the two mirror descent update flows. The operators $L _ { G }$ and $L _ { F }$ diferentiate along the first and second update flows, respectively.

Lemma A.5 (Lie derivative bound for the correction terms). Assume Lemma $A . \mathit { 4 }$ holds. For all nonnegative integers $r , \alpha , \beta$ with $r + \alpha + \beta \geq 1$ , we have

$$
\| L _ { G } ^ { \alpha } L _ { F } ^ { \beta } H _ { r } \| _ { \infty } \leq ( 4 C _ { 0 } ) ^ { r + \alpha + \beta + 1 } ( r + \alpha + \beta ) ! .\tag{61}
$$

Proof. By Lemma $\mathrm { { A . 2 } } .$ , for each $r \geq 0$ , there is a finite representation

$$
H _ { r } = \sum _ { i \in I _ { r } } c _ { i } B _ { i } , \sum _ { i \in I _ { r } } | c _ { i } | \leq 4 ^ { r + 1 } ,\tag{62}
$$

where $I _ { r }$ is a finite index set and each $B _ { i }$ is a right-nested Poisson bracket of length $r + 1$ in $F , G$ We first describe the efect of the Lie derivatives on a single term $B _ { i }$ . Since $L _ { Q } f = \{ f , Q \} =$ $- \{ Q , f \}$ , the operator $L _ { Q }$ adds $Q$ as an outer bracket and changes the sign. To keep track of the order, define $B _ { i } ^ { ( 0 ) } = B _ { i }$ and, for $1 \leq j \leq \alpha + \beta$ 2

$$
B _ { i } ^ { ( j ) } = \left\{ \begin{array} { l l } { \{ F , B _ { i } ^ { ( j - 1 ) } \} , } & { 1 \le j \le \beta , } \\ { \{ G , B _ { i } ^ { ( j - 1 ) } \} , } & { \beta < j \le \alpha + \beta . } \end{array} \right.\tag{63}
$$

Hence, the first $\beta$ brackets correspond to $L _ { F } ^ { \beta }$ , and the next α correspond to $L _ { G } ^ { \alpha }$ . By Eq. (63), we have

$$
L _ { G } ^ { \alpha } L _ { F } ^ { \beta } B _ { i } = ( - 1 ) ^ { \alpha + \beta } B _ { i } ^ { ( \alpha + \beta ) }\tag{64}
$$

Moreover, by $\operatorname { E q } .$ (62), we have

$$
L _ { G } ^ { \alpha } L _ { F } ^ { \beta } H _ { r } = \sum _ { i \in I _ { r } } ( - 1 ) ^ { \alpha + \beta } c _ { i } B _ { i } ^ { ( \alpha + \beta ) } .\tag{65}
$$

Because $B _ { i }$ is the right-nested Poisson bracket of length $r { + 1 }$ , each $B _ { i } ^ { ( \alpha + \beta ) }$ has length $r + \alpha + \beta + 1 \ge 2$ By Lemma $\mathrm { A . 4 }$ with $q = r + \alpha + \beta + 1$ and $s = 0$ , each such bracket satisfies

$$
\| B _ { i } ^ { ( \alpha + \beta ) } \| _ { \infty } \le C _ { 0 } ^ { r + \alpha + \beta } ( r + \alpha + \beta ) ! , \qquad i \in I _ { r } .\tag{66}
$$

By the triangle inequality and Lemma A.2, we obtain

$$
\| L _ { G } ^ { \alpha } L _ { F } ^ { \beta } H _ { r } \| _ { \infty } \leq \sum _ { i \in I _ { r } } | c _ { i } | \| B _ { i } ^ { ( \alpha + \beta ) } \| _ { \infty } \leq 4 ^ { r + 1 } C _ { 0 } ^ { r + \alpha + \beta } ( r + \alpha + \beta ) ! \leq ( 4 C _ { 0 } ) ^ { r + \alpha + \beta + 1 } ( r + \alpha + \beta ) ! ,\tag{67}
$$

where the last inequality is by $\alpha + \beta \ge 0$ and $C _ { 0 } \geq 1$

We next bound the Taylor remainder of $f \circ { \mathcal { M } } _ { h }$ , then apply this bound to the truncated modified energy $\widehat { H } _ { h } ^ { [ N ] }$

Lemma A.6. Fix an integer $m \geq 0$ and a smooth scalar function f. If

$$
\| L _ { G } ^ { \alpha } L _ { F } ^ { \beta } f \| _ { \infty } \leq M \quad w h e n e v e r \alpha + \beta = m + 1 ,\tag{68}
$$

then

$$
\left| f ( \mathcal M _ { h } z ) - \sum _ { \alpha + \beta \leq m } \frac { h ^ { \alpha + \beta } } { \alpha ! \beta ! } L _ { G } ^ { \alpha } L _ { F } ^ { \beta } f ( z ) \right| \leq \frac { 2 ^ { m + 1 } } { ( m + 1 ) ! } M h ^ { m + 1 } , \qquad h \geq 0 .\tag{69}
$$

Proof. Fix a point z in the dual space and a stepsize $h \geq 0$ . For $Q \in \operatorname { s e t } ( F , G )$ , the flow satisfies $\begin{array} { r } { \frac { d } { d t } \phi _ { Q } ^ { t } ( z ) = J \nabla Q ( \phi _ { Q } ^ { t } ( z ) ) } \end{array}$ . For any smooth scalar function $^ { g , }$ the chain rule therefore yields

$$
\frac { d } { d t } g ( \phi _ { Q } ^ { t } ( z ) ) = \nabla g ( \phi _ { Q } ^ { t } ( z ) ) ^ { \top } \frac { d } { d t } \phi _ { Q } ^ { t } ( z ) = \nabla g ( \phi _ { Q } ^ { t } ( z ) ) ^ { \top } J \nabla Q ( \phi _ { Q } ^ { t } ( z ) ) = ( L _ { Q } g ) ( \phi _ { Q } ^ { t } ( z ) ) ,\tag{70}
$$

where the last equality uses $L _ { Q } g = \{ g , Q \} = \nabla g ^ { \mathsf { T } } J \nabla Q$ . The same identity with g replaced by $L _ { Q } ^ { j } g$ states that

$$
\frac { d } { d t } ( L _ { Q } ^ { j } g ) ( \phi _ { Q } ^ { t } ( z ) ) = \big ( L _ { Q } ( L _ { Q } ^ { j } g ) \big ) ( \phi _ { Q } ^ { t } ( z ) ) = ( L _ { Q } ^ { j + 1 } g ) ( \phi _ { Q } ^ { t } ( z ) ) .
$$

The base case is $L _ { Q } ^ { 0 } g = g$ . Induction on $j$ now yields

$$
\frac { d ^ { j } } { d t ^ { j } } g ( \phi _ { Q } ^ { t } ( z ) ) = ( L _ { Q } ^ { j } g ) ( \phi _ { Q } ^ { t } ( z ) ) , \qquad j \ge 0 .\tag{71}
$$

Since $\mathcal { M } _ { h } z = \phi _ { F } ^ { h } ( \phi _ { G } ^ { h } ( z ) )$ , Taylor’s formula for $t \mapsto f ( \phi _ { F } ^ { t } ( \phi _ { G } ^ { h } ( z ) ) )$ at $t = 0$ through order m is

$$
f ( \mathcal { M } _ { h } z ) = \sum _ { \beta = 0 } ^ { m } \frac { h ^ { \beta } } { \beta ! } ( L _ { F } ^ { \beta } f ) ( \phi _ { G } ^ { h } ( z ) ) + R _ { F } ( h , z ) ,\tag{72}
$$

where, by Eq. (71), the integral remainder is

$$
R _ { F } ( h , z ) : = \frac { 1 } { m ! } \int _ { 0 } ^ { h } ( h - t ) ^ { m } ( L _ { F } ^ { m + 1 } f ) ( \phi _ { F } ^ { t } ( \phi _ { G } ^ { h } ( z ) ) ) d t .
$$

By the assumed bound with $( \alpha , \beta ) = ( 0 , m + 1 )$ , we have

$$
| R _ { F } ( h , z ) | \leq \frac { M } { m ! } \int _ { 0 } ^ { h } ( h - t ) ^ { m } d t = \frac { M h ^ { m + 1 } } { ( m + 1 ) ! } .\tag{73}
$$

Next, for each $\beta \in \{ 0 , \ldots , m \}$ , we apply Taylor’s formula to $t \mapsto ( L _ { F } ^ { \beta } f ) ( \phi _ { G } ^ { t } ( z ) )$ through order $m - \beta$ . By Eq. (71), we obtain

$$
( L _ { F } ^ { \beta } f ) ( \phi _ { G } ^ { h } ( z ) ) = \sum _ { \alpha = 0 } ^ { m - \beta } \frac { h ^ { \alpha } } { \alpha ! } L _ { G } ^ { \alpha } L _ { F } ^ { \beta } f ( z ) + R _ { G , \beta } ( h , z ) ,\tag{74}
$$

where

$$
R _ { G , \beta } ( h , z ) : = \frac { 1 } { ( m - \beta ) ! } \int _ { 0 } ^ { h } ( h - t ) ^ { m - \beta } ( L _ { G } ^ { m + 1 - \beta } L _ { F } ^ { \beta } f ) ( \phi _ { G } ^ { t } ( z ) ) d t .
$$

Since $( m + 1 - \beta ) + \beta = m + 1$ , the assumed bound yields

$$
\frac { h ^ { \beta } } { \beta ! } | R _ { G , \beta } ( h , z ) | \leq \frac { h ^ { \beta } } { \beta ! } \frac { M } { ( m - \beta ) ! } \int _ { 0 } ^ { h } ( h - t ) ^ { m - \beta } d t = \frac { h ^ { \beta } } { \beta ! } \frac { M h ^ { m - \beta + 1 } } { ( m - \beta + 1 ) ! } = \frac { M h ^ { m + 1 } } { \beta ! ( m + 1 - \beta ) ! } .\tag{75}
$$

We substitute Eq. (74) into Eq. (72). The triangle inequality then gives

$$
\begin{array} { r l r } {  {  f ( \mathcal { M } _ { h } z ) - \sum _ { \alpha + \beta \le m } \frac { h ^ { \alpha + \beta } } { \alpha ! \beta ! } L _ { G } ^ { \alpha } L _ { F } ^ { \beta } f ( z )  \le \vert R _ { F } ( h , z ) \vert + \displaystyle \sum _ { \beta = 0 } ^ { m } \frac { h ^ { \beta } } { \beta ! } \vert R _ { G , \beta } ( h , z ) \vert } } \\ & { } & { \le M h ^ { m + 1 } ( \frac { 1 } { ( m + 1 ) ! } + \displaystyle \sum _ { \beta = 0 } ^ { m } \frac { 1 } { \beta ! ( m + 1 - \beta ) ! } ) } \\ & { } & { = \frac { M h ^ { m + 1 } } { ( m + 1 ) ! } \displaystyle \sum _ { \beta = 0 } ^ { m + 1 } \binom { m + 1 } { \beta } = \frac { 2 ^ { m + 1 } } { ( m + 1 ) ! } M h ^ { m + 1 } , } \end{array}\tag{76}
$$

where the second inequality uses Eqs. (73) and (75). Since all bounds are uniform in z, we prove Eq. (69). □

Finally, we now combine the derivative bounds and Taylor formula to prove Lemma 3.6.

Proof of Lemma 3.6. Fix an integer $N \geq 0$ and $0 < \eta \leq 1 / [ 1 6 C _ { 0 } ( N + 2 ) ]$ . We first bound the diference between the truncated modified energy $\widehat { H } _ { \eta } ^ { [ N ] }$ and $H _ { 0 }$ . For $N = 0$ , this diference is zero because $\widehat { H } _ { \eta } ^ { [ 0 ] } = H _ { 0 }$ , so the first bound holds immediately. For $N \geq 1$ and $1 \leq r \leq N$ , Lemma A.5 with $\alpha = \beta = 0$ implies

$$
\| \eta ^ { r } H _ { r } \| _ { \infty } \leq \eta ^ { r } ( 4 C _ { 0 } ) ^ { r + 1 } r ! = : a _ { r } .\tag{77}
$$

For $1 \leq r < N$ , we have

$$
\frac { a _ { r + 1 } } { a _ { r } } = 4 \eta C _ { 0 } ( r + 1 ) \le \frac { r + 1 } { 4 ( N + 2 ) } \le \frac { 1 } { 4 } .\tag{78}
$$

Since $a _ { 1 } = 1 6 C _ { 0 } ^ { 2 } \eta$ , by Eq. (78), we have $a _ { r } \leq 1 6 C _ { 0 } ^ { 2 } \eta ( 1 / 4 ) ^ { r - 1 }$ . By the triangle inequality and Eq. (77), we obtain

$$
\| \widehat { H } _ { \eta } ^ { [ N ] } - H _ { 0 } \| _ { \infty } \leq \sum _ { r = 1 } ^ { N } a _ { r } \leq 1 6 C _ { 0 } ^ { 2 } \eta \sum _ { r = 1 } ^ { N } \left( \frac { 1 } { 4 } \right) ^ { r - 1 } \leq \frac { 6 4 } { 3 } C _ { 0 } ^ { 2 } \eta .\tag{79}
$$

This proves the first bound in Eq. (15) for every $N \geq 0$

It remains to bound the change in $\widehat { H } _ { \eta } ^ { [ N ] }$ over one iteration. For each $r \in \{ 0 , \ldots , N \}$ and $\alpha + \beta = N + 2 - r$ , Lemma A.5 implies

$$
\| L _ { G } ^ { \alpha } L _ { F } ^ { \beta } H _ { r } \| _ { \infty } \leq ( 4 C _ { 0 } ) ^ { N + 3 } ( N + 2 ) ! ,\tag{80}
$$

since $r + \alpha + \beta = N + 2 \geq 1$ . We apply Lemma A.6 with $f = H _ { r } , m = N + 1 - r$ , and $M = ( 4 C _ { 0 } ) ^ { N + 3 } ( N + 2 ) !$ to obtain

$$
H _ { r } ( \mathcal { M } _ { \eta } z ) = \sum _ { \alpha + \beta \leq N + 1 - r } \frac { \eta ^ { \alpha + \beta } } { \alpha ! \beta ! } ( L _ { G } ^ { \alpha } L _ { F } ^ { \beta } H _ { r } ) ( z ) + R _ { r } ( \eta , z ) ,\tag{81}
$$

where the remainder satisfies, for any z,

$$
\eta ^ { r } \| R _ { r } ( \eta , z ) \| _ { \infty } \leq \eta ^ { N + 2 } ( 4 C _ { 0 } ) ^ { N + 3 } ( N + 2 ) ! \frac { 2 ^ { N + 2 - r } } { ( N + 2 - r ) ! } .\tag{82}
$$

Note that the term with $\alpha = \beta = 0$ in $\operatorname { E q . }$ (81) is just $H _ { r } ( z )$ . We subtract this term from both sides of Eq. (81), multiply by $\eta ^ { r }$ for each r, and sum over $r = 0 , \ldots , N$ . Thus

$$
\begin{array} { l } { \widehat { H } _ { \eta } ^ { [ N ] } ( \mathcal { M } _ { \eta } z ) - \widehat { H } _ { \eta } ^ { [ N ] } ( z ) = \displaystyle \sum _ { r = 0 } ^ { N } \eta ^ { r } \big ( H _ { r } ( \mathcal { M } _ { \eta } z ) - H _ { r } ( z ) \big ) } \\ { = \displaystyle \sum _ { r = 0 } ^ { N } \displaystyle \sum _ { 1 \leq \alpha + \beta \leq N + 1 - r } \frac { \eta ^ { r + \alpha + \beta } } { \alpha ! \beta ! } ( L _ { G } ^ { \alpha } L _ { F } ^ { \beta } H _ { r } ) ( z ) + \displaystyle \sum _ { r = 0 } ^ { N } \eta ^ { r } R _ { r } ( \eta , z ) . } \end{array}
$$

Equivalently, we have

$$
\widehat { H } _ { \eta } ^ { [ N ] } ( \mathcal { M } _ { \eta } z ) - \widehat { H } _ { \eta } ^ { [ N ] } ( z ) = \sum _ { j = 1 } ^ { N + 1 } \eta ^ { j } \sum _ { r = 0 } ^ { j - 1 } \sum _ { \alpha + \beta = j - r } \frac { ( L _ { G } ^ { \alpha } L _ { F } ^ { \beta } H _ { r } ) ( z ) } { \alpha ! \beta ! } + \sum _ { r = 0 } ^ { N } \eta ^ { r } R _ { r } ( \eta , z ) .\tag{83}
$$

We next show that the first term in Eq. (83) vanishes. For a formal variable $h ,$ the exponential operators in Eq. (25) satisfy

$$
e ^ { h L _ { G } } e ^ { h L _ { F } } - I = \sum _ { \alpha + \beta \geq 1 } \frac { h ^ { \alpha + \beta } } { \alpha ! \beta ! } L _ { G } ^ { \alpha } L _ { F } ^ { \beta } ,
$$

where we subtract I to remove the term $\alpha = \beta = 0$ . This operator acts on $\begin{array} { r } { \widehat { H } _ { h } ^ { [ N ] } = \sum _ { r = 0 } ^ { N } h ^ { r } H _ { \ i } } \end{array}$ as follows:

$$
\left( e ^ { h L _ { G } } e ^ { h L _ { F } } - I \right) \widehat { H } _ { h } ^ { [ N ] } = \sum _ { j \geq 1 } h ^ { j } \sum _ { \alpha + \beta + r = j , \alpha + \beta \geq 1 , r \leq N } \frac { L _ { G } ^ { \alpha } L _ { F } ^ { \beta } H _ { r } } { \alpha ! \beta ! } .
$$

Consequently, for every $1 \leq j \leq N + 1$ and every $z ,$

$$
\begin{array} { l } { { \displaystyle { \sum _ { r = 0 } ^ { j - 1 } \sum _ { \alpha + \beta = j - r } \frac { ( L _ { G } ^ { \alpha } L _ { F } ^ { \beta } H _ { r } ) ( z ) } { \alpha ! \beta ! } = [ h ^ { j } ] \Big ( \big ( e ^ { h L _ { G } } e ^ { h L _ { F } } - I \big ) \widehat { H } _ { h } ^ { [ N ] } \Big ) ( z ) } } } \\ { { \mathrm { } } } \\ { { \displaystyle { \qquad = [ h ^ { j } ] \Big ( \widehat { H } _ { h } ^ { [ N ] } ( \mathcal { M } _ { h } z ) - \widehat { H } _ { h } ^ { [ N ] } ( z ) \Big ) = 0 , } } } \end{array}
$$

where the second equality uses Eq. (25), and the last equality follows from Eq. (44): the formal energy change is $O ( h ^ { N + 2 } )$ , so all coeficients through degree $N + 1$ vanish. Therefore, Eq. (83) can be simplified to

$$
\widehat { H } _ { \eta } ^ { [ N ] } ( \mathcal { M } _ { \eta } z ) - \widehat { H } _ { \eta } ^ { [ N ] } ( z ) = \sum _ { r = 0 } ^ { N } \eta ^ { r } R _ { r } ( \eta , z ) .\tag{84}
$$

With the index $\ell = N + 2 - r .$ the sum becomes

$$
\sum _ { r = 0 } ^ { N } \frac { 2 ^ { N + 2 - r } } { ( N + 2 - r ) ! } = \sum _ { \ell = 2 } ^ { N + 2 } \frac { 2 ^ { \ell } } { \ell ! } \leq \sum _ { \ell = 0 } ^ { \infty } \frac { 2 ^ { \ell } } { \ell ! } = e ^ { 2 } .\tag{85}
$$

By the triangle inequality and Eqs. (82) and (85), the right-hand side of Eq. (84) satisfies

$$
\begin{array} { r l } & { \| \widehat { H } _ { \eta } ^ { [ N ] } \circ \mathcal { M } _ { \eta } - \widehat { H } _ { \eta } ^ { [ N ] } \| _ { \infty } \leq \displaystyle \sum _ { r = 0 } ^ { N } \eta ^ { r } \| R _ { r } ( \eta , \cdot ) \| _ { \infty } } \\ & { \qquad \leq \displaystyle \sum _ { r = 0 } ^ { N } \frac { 2 ^ { N + 2 - r } } { ( N + 2 - r ) ! } \eta ^ { N + 2 } ( 4 C _ { 0 } ) ^ { N + 3 } ( N + 2 ) ! } \\ & { \qquad \leq e ^ { 2 } \eta ^ { N + 2 } ( 4 C _ { 0 } ) ^ { N + 3 } ( N + 2 ) ! . } \end{array}\tag{86}
$$

Since $4 \eta C _ { 0 } \leq 1 / [ 4 ( N + 2 ) ]$ and $( N + 2 ) ! \leq ( N + 2 ) ^ { N + 1 }$ , we can further simplify Eq. (86) using these bounds:

$$
\begin{array} { r l } & { \| \widehat { H } _ { \eta } ^ { [ N ] } \circ \mathcal { M } _ { \eta } - \widehat { H } _ { \eta } ^ { [ N ] } \| _ { \infty } \leq 1 6 e ^ { 2 } C _ { 0 } ^ { 2 } \eta ( 4 \eta C _ { 0 } ) ^ { N + 1 } ( N + 2 ) ! } \\ & { \qquad \leq 1 6 e ^ { 2 } C _ { 0 } ^ { 2 } \eta \frac { ( N + 2 ) ! } { 4 ^ { N + 1 } ( N + 2 ) ^ { N + 1 } } } \\ & { \qquad \leq 1 6 e ^ { 2 } C _ { 0 } ^ { 2 } \eta 4 ^ { - ( N + 1 ) } . } \end{array}\tag{87}
$$

Together with Eq. (79), we prove Lemma 3.6.

## A.3 Proof of the regret bound

In this section, we first establish the comparator identity in Lemma 3.3, then use the modified-energy estimate to complete the proof of Theorem 3.7.

Lemma 3.3 (Regret decomposition). Fix a time horizon $T \geq 1$ and a step size $\eta > 0$ . For any comparators $( \tilde { x } , \tilde { y } ) \in$ dom $\psi _ { \mathcal { X } } \times$ dom $\psi _ { \mathcal { Y } }$ , define the Fenchel coupling:

$$
\Gamma _ { \tilde { x } , \tilde { y } } ( u , w ) : = F ( u ) - ( \langle u , \tilde { x } \rangle - \psi _ { \mathcal { X } } ( \tilde { x } ) ) + G ( w ) - ( \langle w , \tilde { y } \rangle - \psi _ { \mathcal { Y } } ( \tilde { y } ) ) \geq 0 ,
$$

where $( u , w )$ are the dual variables in (AltOMD-Dual). Then, we have $\Gamma _ { \tilde { x } , \tilde { y } } ( u _ { k } , w _ { k } ) = D _ { \psi _ { \mathcal { X } } } ( \tilde { x } , x _ { k } ) \ : + \ :$ $D _ { \psi _ { 3 } } ( \tilde { y } , y _ { k } )$ for all $k \geq 0$ . Moreover, all the iterates generated by the AltOMD updates satisfy

$$
\eta \big ( \tilde { y } ^ { \top } A x _ { k + 1 } - y _ { k } ^ { \top } A \tilde { x } \big ) = H _ { 0 } ( z _ { k + 1 } ) - H _ { 0 } ( z _ { k } ) + \Gamma _ { \tilde { x } , \tilde { y } } ( z _ { k } ) - \Gamma _ { \tilde { x } , \tilde { y } } ( z _ { k + 1 } ) ,
$$

where $H _ { 0 } ( z ) = F ( u ) + G ( w )$ is defined in $E q . \ ( 4 )$

Convex duality identities. We prove the two identities needed to express the Fenchel coupling as a Bregman divergence.

Proposition A.7 (Fenchel–Legendre identities). Let $\psi : \mathbb { R } ^ { d }  \mathbb { R } \cup \{ + \infty \}$ be a proper convex function that is diferentiable on ri(dom ψ) relative to the afine hull of its domain. Suppose $\psi ^ { * }$ is finite on a neighborhood of $u \in \mathbb { R } ^ { d }$ and diferentiable at u, and let $x = \nabla \psi ^ { * } ( u ) \in \mathrm { r i } ( \mathrm { d o m } \psi )$ . Then, for every $c \in$ dom $\psi _ { . }$

$$
\begin{array} { c } { { \psi ( x ) + \psi ^ { * } ( u ) = \langle u , x \rangle , } } \\ { { \langle u - \nabla \psi ( x ) , c - x \rangle = 0 . } } \end{array}\tag{88}
$$

Here $\nabla \psi ( x )$ is the gradient relative to the afine hull of dom ψ.

Proof. For $c \in \operatorname { d o m } \psi$ , we diferentiate the convexity inequality along $x + t ( c - x )$ at $t = 0$ from the right. We obtain

$$
\psi ( c ) \geq \psi ( x ) + \langle \nabla \psi ( x ) , c - x \rangle .
$$

Thus the supremum defining $\psi ^ { * } ( \nabla \psi ( x ) )$ is attained at x. The same argument for the convex function $\psi ^ { * }$ at u, with $\nabla \psi ^ { * } ( u ) = x$ , implies

$$
\begin{array} { r l } & { \langle \nabla \psi ( x ) , x \rangle - \psi ( x ) = \psi ^ { * } ( \nabla \psi ( x ) ) } \\ & { ~ \geq \psi ^ { * } ( u ) + \langle x , \nabla \psi ( x ) - u \rangle . } \end{array}
$$

Equivalently, $\psi ^ { * } ( u ) \leq \langle u , x \rangle - \psi ( x )$ . The reverse inequality follows from the definition of $\psi ^ { * } ( u )$ , so $\psi ( x ) + \psi ^ { * } ( u ) = \langle u , x \rangle$

Consequently, x maximizes $\langle u , y \rangle - \psi ( y )$ over $y \in$ dom ψ. Since $x \in \mathrm { r i } ( \operatorname { d o m } \psi )$ , for every $c \in$ dom ψ the points $x + t ( c - x )$ remain in the domain for all suficiently small positive and negative t. The derivative of $t \mapsto \langle u , x + t ( c - x ) \rangle - \psi ( x + t ( c - x ) )$ therefore vanishes at $t = 0$ , so $\langle \boldsymbol { u } - \nabla \psi ( \boldsymbol { x } ) , \boldsymbol { c } - \boldsymbol { x } \rangle = 0$ □

Fix an integer $T \geq 1$ and a constant stepsize $h > 0 .$ , and write ${ z } _ { k } = ( u _ { k } , w _ { k } )$ for the dual iterates of Eq. (AltOMD-Dual) with $\eta = h$ . For comparators x˜ ∈ dom ψ<sub>X</sub> and $\tilde { y } \in$ dom $\psi _ { \mathcal { V } }$ , use the Fenchel coupling $\Gamma _ { \tilde { x } , \tilde { y } }$ defined in Lemma 3.3. We apply Proposition A.7 to $\psi _ { \mathcal { X } }$ and $\psi _ { \mathcal { V } }$ . After we expand the Bregman divergences, we have

$$
\Gamma _ { \tilde { x } , \tilde { y } } ( u , w ) = D _ { \psi _ { \mathcal { X } } } ( \tilde { x } , \nabla F ( u ) ) + D _ { \psi _ { \mathcal { Y } } } ( \tilde { y } , \nabla G ( w ) ) \geq 0 .\tag{89}
$$

We next compute the change in the coupling over one iteration. Because $F ( u ) + G ( w ) = H _ { 0 } ( z )$

$$
\begin{array} { r l } & { \Gamma _ { \tilde { x } , \tilde { y } } ( z _ { k + 1 } ) - \Gamma _ { \tilde { x } , \tilde { y } } ( z _ { k } ) = H _ { 0 } ( z _ { k + 1 } ) - H _ { 0 } ( z _ { k } ) - \langle u _ { k + 1 } - u _ { k } , \tilde { x } \rangle - \langle w _ { k + 1 } - w _ { k } , \tilde { y } \rangle } \\ & { \qquad = H _ { 0 } ( z _ { k + 1 } ) - H _ { 0 } ( z _ { k } ) - h \big ( \tilde { y } ^ { \top } A x _ { k + 1 } - y _ { k } ^ { \top } A \tilde { x } \big ) . } \end{array}\tag{90}
$$

Equation (90) is equivalent to Eq. (11) with $\eta \ : = \ : h$ . By Assumption 3.2, dom $\psi _ { \mathcal { X } } = \mathcal { X }$ and dom $\psi _ { \mathcal { Y } } = \mathcal { Y } ,$ , since any comparator outside an efective domain would have infinite initial Bregman divergence. Thus the identity applies to every comparator in $\mathcal { X } \times \mathcal { V }$ in the regret argument below. We sum over $k = 0 , \ldots , T - 1$ and obtain

$$
\begin{array} { r } { h \displaystyle \sum _ { k = 0 } ^ { T - 1 } \left( \tilde { y } ^ { \top } A x _ { k + 1 } - y _ { k } ^ { \top } A \tilde { x } \right) = H _ { 0 } ( z _ { T } ) - H _ { 0 } ( z _ { 0 } ) + \Gamma _ { \tilde { x } , \tilde { y } } ( z _ { 0 } ) - \Gamma _ { \tilde { x } , \tilde { y } } ( z _ { T } ) } \\ { \leq H _ { 0 } ( z _ { T } ) - H _ { 0 } ( z _ { 0 } ) + \Lambda _ { 0 } , } \end{array}
$$

where the inequality uses $\Gamma _ { \tilde { x } , \tilde { y } } ( z _ { T } ) \ge 0$ and $\Gamma _ { \tilde { x } , \tilde { y } } ( z _ { 0 } ) \le \Lambda _ { 0 }$

By Eq. (5), the maximum over $( \tilde { x } , \tilde { y } ) \in \mathcal { X } \times \mathcal { Y }$ satisfies

$$
\begin{array} { r l } & { h \mathrm { R e g } _ { T } = \displaystyle \operatorname* { m a x } _ { ( \tilde { x } , \tilde { y } ) \in \mathcal { X } \times \mathcal { V } } h \sum _ { k = 0 } ^ { T - 1 } \big ( \tilde { y } ^ { \top } A x _ { k + 1 } - y _ { k } ^ { \top } A \tilde { x } \big ) } \\ & { \qquad \leq \Lambda _ { 0 } + H _ { 0 } ( z _ { T } ) - H _ { 0 } ( z _ { 0 } ) . } \end{array}\tag{91}
$$

It remains to bound the change in $H _ { 0 } = F + G$ over the run. Fix an integer $N \geq 0$ and suppose $0 < h \leq 1 / [ 1 6 C _ { 0 } ( N + 2 ) ]$ . We first decompose the change in the Hamiltonian as follows.

$$
H _ { 0 } ( z _ { T } ) - H _ { 0 } ( z _ { 0 } ) = \big [ H _ { 0 } ( z _ { T } ) - \widehat { H } _ { h } ^ { [ N ] } ( z _ { T } ) \big ] + \big [ \widehat { H } _ { h } ^ { [ N ] } ( z _ { T } ) - \widehat { H } _ { h } ^ { [ N ] } ( z _ { 0 } ) \big ] + \big [ \widehat { H } _ { h } ^ { [ N ] } ( z _ { 0 } ) - H _ { 0 } ( z _ { 0 } ) \big ] .\tag{92}
$$

By Lemma 3.6, we have

$$
\begin{array} { l } { \displaystyle { H _ { 0 } ( z _ { T } ) - H _ { 0 } ( z _ { 0 } ) \leq \frac { 1 2 8 } { 3 } C _ { 0 } ^ { 2 } h + \sum _ { k = 0 } ^ { T - 1 } \bigl | \widehat { H } _ { h } ^ { [ N ] } ( z _ { k + 1 } ) - \widehat { H } _ { h } ^ { [ N ] } ( z _ { k } ) \bigr | } } \\ { \displaystyle { \leq C _ { 0 } ^ { 2 } h \left( \frac { 1 2 8 } { 3 } + 1 6 e ^ { 2 } T 4 ^ { - ( N + 1 ) } \right) . } } \end{array}\tag{93}
$$

From Eqs. (91) and (93), we obtain, after division by h,

$$
\mathrm { R e g } _ { T } \leq \frac { \Lambda _ { 0 } } { h } + C _ { 0 } ^ { 2 } \left( \frac { 1 2 8 } { 3 } + 1 6 e ^ { 2 } T 4 ^ { - ( N + 1 ) } \right) .\tag{94}
$$

Set $N \ = \ \lfloor \log _ { 2 } T \rfloor$ and $h = \eta \ : = \ : 1 / [ 1 6 C _ { 0 } ( N + 2 ) ]$ as in Theorem 3.7. Since $T$ is an integer, $T \le 2 ^ { N + 1 } - 1 \le 4 ^ { N }$ , where the second inequality follows from $( 2 ^ { N } - 1 ) ^ { 2 } \geq 0$ . Thus $T 4 ^ { - ( N + 1 ) } \le 1 / 4$ so

$$
\begin{array} { r l r } {  { \mathrm { R e g } _ { T } \le 1 6 C _ { 0 } \Lambda _ { 0 } ( N + 2 ) + ( \frac { 1 2 8 } { 3 } + 4 e ^ { 2 } ) C _ { 0 } ^ { 2 } } } \\ & { } & \\ & { } & { \le 1 6 C _ { 0 } \Lambda _ { 0 } ( \log _ { 2 } T + 2 ) + ( \frac { 1 2 8 } { 3 } + 4 e ^ { 2 } ) C _ { 0 } ^ { 2 } . } \end{array}
$$

This proves Theorem 3.7.

## B Verification of the entropy regularizer

In this section, we verify the assumptions for negative entropy on the simplex and then prove the logarithmic regret bound in Corollary 4.2 for the alternating multiplicative weights update algorithm.

First, we record standard power-series estimates and Banach’s theorem for symmetric multilinear forms.

Proposition B.1. For $r > 0$ , let $\mathcal { A } _ { r }$ be the space of real power series $\begin{array} { r } { P ( t ) = \sum _ { j \geq 0 } P _ { j } t ^ { j } } \end{array}$ with finite weighted coeficient norm

$$
\| P \| _ { r } : = \sum _ { j \ge 0 } | P _ { j } | r ^ { j } < \infty .
$$

Then, the space $\mathcal { A } _ { r }$ is complete, and

$$
\| P Q \| _ { r } \leq \| P \| _ { r } \| Q \| _ { r } , \quad \forall P , Q \in \mathcal { A } _ { r } .\tag{95}
$$

For every $P \in { \mathcal { A } } _ { r }$ and integer $s \geq 0$ , we have

$$
| P ^ { ( s ) } ( 0 ) | = s ! | P _ { s } | \leq s ! r ^ { - s } \| P \| _ { r } .\tag{96}
$$

Furthermore, for every integer $s \geq 1$ and symmetric tensor M of order s on $\mathbb { R } ^ { d }$ , we have

$$
\| M \| _ { \mathrm { o p } } = \operatorname* { s u p } _ { \| h \| _ { 2 } \leq 1 } | M [ h , \ldots , h ] | .\tag{97}
$$

Proof of Proposition B.1. First, let $\begin{array} { r } { P ^ { [ n ] } ( t ) = \sum _ { i > 0 } a _ { n , j } t ^ { j } } \end{array}$ be a Cauchy sequence in $\mathcal { A } _ { r }$ . Since $| a _ { n , j } - a _ { m , j } | r ^ { j } \leq \| P ^ { [ n ] } - P ^ { [ m ] } \| _ { r }$ , for each fixed $j ,$ , the coeficient sequence converges to a real number $a _ { j } : = \dim _ { n \to \infty } a _ { n , j }$ . Define the candidate limit series $\textstyle P _ { \infty } ( t ) : = \sum _ { i > 0 } a _ { j } t ^ { j }$ . For every $\varepsilon > 0$ , choose $n _ { 0 }$ such that $\| P ^ { [ n ] } - P ^ { [ m ] } \| _ { r } \leq \varepsilon$ whenever $n , m \geq n _ { 0 }$ . For each $n \geq n _ { 0 }$ and integer $L \geq 0$ , coeficientwise convergence implies

$$
\sum _ { j = 0 } ^ { L } | a _ { n , j } - a _ { j } | r ^ { j } = \operatorname* { l i m } _ { m \to \infty } \sum _ { j = 0 } ^ { L } | a _ { n , j } - a _ { m , j } | r ^ { j } \leq \varepsilon .
$$

We let $L \to \infty$ and conclude that $\| P ^ { [ n ] } - P _ { \infty } \| _ { r } \leq \varepsilon$ . In particular, $\| P _ { \infty } \| _ { r } \leq \| P ^ { [ n _ { 0 } ] } \| _ { r } + \varepsilon < \infty .$ , so $P _ { \infty } \in { \mathcal { A } } _ { r }$ and $P ^ { [ n ] } \to P _ { \infty }$ in this space. This proves completeness.

For $P , Q \in { \mathcal { A } } _ { r }$ , the coeficient of $t ^ { j }$ in P Q is $\textstyle \sum _ { \ell = 0 } ^ { j } P _ { \ell } Q _ { j - \ell }$ . Hence, we have

$$
\begin{array} { r } { \| P Q \| _ { r } = \sum _ { j \geq 0 } \left| \sum _ { \ell = 0 } ^ { j } P _ { \ell } Q _ { j - \ell } \right| r ^ { j } \leq \sum _ { \ell \geq 0 } \sum _ { k \geq 0 } | P _ { \ell } | r ^ { \ell } | Q _ { k } | r ^ { k } = \| P \| _ { r } \| Q \| _ { r } , } \end{array}
$$

where the rearrangement is justified by Tonelli’s theorem for nonnegative series. This proves Eq. (95).

Next, for $P \in { \mathcal { A } } _ { r }$ and $| t | \leq r ,$ we have $| P _ { j } t ^ { j } | \leq | P _ { j } | r ^ { j }$ and $\begin{array} { r } { \sum _ { j > 0 } | P _ { j } | r ^ { j } = \| P \| _ { r } < \infty } \end{array}$ . Hence, the Weierstrass M-test implies that the series for P converges absolutely and uniformly on $[ - r , r ]$ . For every $0 < \rho < r$ and integer $s \geq 1$ , we have

$$
\sum _ { j \geq s } \frac { j ! } { ( j - s ) ! } | P _ { j } | \rho ^ { j - s } \leq \rho ^ { - s } \| P \| _ { r } \operatorname* { s u p } _ { j \geq s } j ^ { s } \left( \frac { \rho } { r } \right) ^ { j } < \infty ,
$$

where the last inequality holds because geometric decay dominates the fixed power $j ^ { s }$ . Thus, every formally diferentiated series converges uniformly on $[ - \rho , \rho ]$ . We integrate these series term by term and identify them successively as derivatives of P. At zero, these derivatives satisfy $P ^ { ( s ) } ( 0 ) = s ! P _ { s }$ Together with $| P _ { s } | r ^ { s } \leq \| P \| _ { r }$ , this proves Eq. (96) for $s \geq 1$ . The case $s = 0$ follows directly from $P ( 0 ) = P _ { 0 }$

Finally, Eq. (97) is by Banach (1938, Satz I) for symmetric multilinear forms on the real Hilbert space $( \mathbb { R } ^ { d } , \lVert \cdot \rVert _ { 2 } )$ □

## B.1 Shannon entropy regularizer

Proposition B.2. On $\Delta _ { d } ,$ , the function $\begin{array} { r } { \psi _ { d } ( x ) = \sum _ { i = 1 } ^ { d } x _ { i } \log x _ { i } } \end{array}$ , with 0 log $0 = 0$ , is Legendre relative to the afine hull and finite on the closed simplex. As in Eq. (17), extend $\psi _ { d }$ by $+ \infty$ outside $\Delta _ { d }$ The conjugate of $\psi _ { d }$ is

$$
F _ { d } ( u ) = \log \sum _ { i = 1 } ^ { d } e ^ { u _ { i } } , \qquad [ \nabla F _ { d } ( u ) ] _ { i } = \frac { e ^ { u _ { i } } } { \sum _ { j = 1 } ^ { d } e ^ { u _ { j } } } ,\tag{98}
$$

and, for every integer $s \geq 1$ 2

$$
\operatorname* { s u p } _ { u \in \mathbb { R } ^ { d } } \| D ^ { s } F _ { d } ( u ) \| _ { \mathrm { o p } } \leq \frac { 1 } { 2 } \ : 4 ^ { s } s ! .\tag{99}
$$

For each strictly positive $\begin{array} { r } { x \in \Delta _ { d } , \operatorname* { s u p } _ { \tilde { x } \in \Delta _ { d } } D _ { \psi _ { d } } ( \tilde { x } , x ) \leq \log \frac { 1 } { \operatorname* { m i n } _ { i } x _ { i } } < \infty . } \end{array}$

Proof. Since $( t \log t ) ^ { \prime \prime } = 1 / t > 0$ for $t > 0$ , ψ<sub>d</sub> is strictly convex and smooth on the relative interior of $\Delta _ { d } .$ , and it is finite and continuous on $\Delta _ { d }$ . Convexity extends to the closed simplex by continuity. Since $\Delta _ { d }$ is closed and $\psi _ { d }$ is continuous there, the epigraph $\left\{ ( x , r ) : x \in \Delta _ { d } , \ \psi _ { d } ( x ) \leq r \right\}$ is closed and convex. The extension by $+ \infty$ is therefore proper, closed, and convex, with efective domain $\Delta _ { d }$ . The gradient of the entropy expression has coordinates $1 + \log x _ { i }$ . As x approaches a relative boundary point, some $x _ { i } \to 0$ while another $x _ { j }$ tends to a positive value, so log $x _ { i } - \log x _ { j }  - \infty$ After projection, the gradient relative to the afine hull preserves the coordinate diferences, so the projected gradient also diverges. Thus, the norm of the gradient relative to the afine hull tends to infinity, so $\psi _ { d }$ has the required Legendre property.

Next, fix $u \in \mathbb { R } ^ { d }$ and set $\begin{array} { r } { p _ { i } = e ^ { u _ { i } } / \sum _ { j = 1 } ^ { d } e ^ { u _ { j } } } \end{array}$ . For every $x \in \Delta _ { d } ,$ the identity $\begin{array} { r } { \sum _ { i = 1 } ^ { d } x _ { i } = 1 } \end{array}$ implies

$$
\begin{array} { c l l } { \displaystyle { \langle u , x \rangle - \psi _ { d } ( x ) = \sum _ { i = 1 } ^ { d } x _ { i } \left( \log p _ { i } + \log \sum _ { j = 1 } ^ { d } e ^ { u _ { j } } - \log x _ { i } \right) } } \\ { \displaystyle { = \log \sum _ { j = 1 } ^ { d } e ^ { u _ { j } } - \sum _ { i = 1 } ^ { d } x _ { i } \log \frac { x _ { i } } { p _ { i } } , } } \end{array}\tag{100}
$$

where terms with $x _ { i } = 0$ are interpreted by continuity. We apply − log $a \ge 1 - a$ with $a = p _ { i } / x _ { i }$ on the positive coordinates of x and obtain

$$
\sum _ { i = 1 } ^ { d } x _ { i } \log { \frac { x _ { i } } { p _ { i } } } \geq \sum _ { \stackrel { 1 \leq i \leq d } { x _ { i } > 0 } } x _ { i } \left( 1 - { \frac { p _ { i } } { x _ { i } } } \right) = 1 - \sum _ { \stackrel { 1 \leq i \leq d } { x _ { i } > 0 } } p _ { i } \geq 0 .
$$

The sum $\textstyle \sum _ { i = 1 } ^ { d } x _ { i } \log { \frac { x _ { i } } { p _ { i } } }$ vanishes at $x = p ,$ . We take the supremum over $x \in \Delta _ { d }$ in Eq. (100) to obtain the conjugate in Eq. (98). We diferentiate $\begin{array} { r } { F _ { d } ( u ) = \log \sum _ { j = 1 } ^ { d } e ^ { u _ { j } } } \end{array}$ to obtain the gradient in Eq. (98) and verify that $F _ { d }$ is smooth on R<sup>d</sup>.

To prove Eq. (99), fix $h \in \mathbb { R } ^ { d }$ with $\| h \| _ { 2 } \leq 1$ . In terms of the probabilities $p _ { i }$ defined above, we can write

$$
\begin{array} { r l } & { F _ { d } ( u + t h ) - F _ { d } ( u ) = \log \left( \frac { \sum _ { i = 1 } ^ { d } e ^ { u _ { i } } e ^ { t h _ { i } } } { \sum _ { j = 1 } ^ { d } e ^ { u _ { j } } } \right) } \\ & { \qquad = \log \left( \sum _ { i = 1 } ^ { d } p _ { i } e ^ { t h _ { i } } \right) = \log ( 1 + P ( t ) ) , } \end{array}\tag{101}
$$

where $\begin{array} { r } { P ( t ) : = \sum _ { i = 1 } ^ { d } p _ { i } ( e ^ { t h _ { i } } - 1 ) } \end{array}$ . We expand the exponentials as follows:

$$
\begin{array} { r } { P ( t ) = \sum _ { i = 1 } ^ { d } p _ { i } \left( \sum _ { j \geq 1 } \frac { ( t h _ { i } ) ^ { j } } { j ! } \right) = \sum _ { j \geq 1 } \left( \frac { 1 } { j ! } \sum _ { i = 1 } ^ { d } p _ { i } h _ { i } ^ { j } \right) t ^ { j } . } \end{array}\tag{102}
$$

Here, the exchange of sums is valid because the sum over i is finite and each exponential series converges absolutely for every real t. Let $\begin{array} { r } { \| P \| _ { q } : = \sum _ { j \geq 0 } | P _ { j } | q ^ { j } } \end{array}$ for $q > 0$ . Since $| h _ { i } | \leq \| h \| _ { 2 } \leq 1$ and $\begin{array} { r } { \sum _ { i = 1 } ^ { d } p _ { i } = 1 } \end{array}$ 2

$$
\begin{array} { r } { \| P \| _ { 1 / 4 } = \sum _ { j \geq 1 } \left| \sum _ { i = 1 } ^ { d } p _ { i } h _ { i } ^ { j } \right| \frac { 1 } { 4 j \bar { j } ! } \leq \sum _ { i = 1 } ^ { d } p _ { i } \sum _ { j \geq 1 } \frac { | h _ { i } | ^ { j } } { 4 j \bar { j } ! } \leq \sum _ { j \geq 1 } \frac { 1 } { 4 ^ { j } j ! } \leq \sum _ { j \geq 1 } \frac { 1 } { 4 ^ { j } } = \frac { 1 } { 3 } . } \end{array}\tag{103}
$$

By Eq. (95), the logarithm series

$$
\log ( 1 + P ) : = \sum _ { k \geq 1 } { \frac { ( - 1 ) ^ { k + 1 } } { k } } P ^ { k }\tag{104}
$$

converges in $\mathcal { A } _ { 1 / 4 }$ , with

$$
\| \log ( 1 + P ) \| _ { 1 / 4 } \le \sum _ { k \ge 1 } \frac { \| P ^ { k } \| _ { 1 / 4 } } { k } \le \sum _ { k \ge 1 } \frac { \| P \| _ { 1 / 4 } ^ { k } } { k } \le \sum _ { k \ge 1 } \frac { 3 ^ { - k } } { k } = \log \frac { 3 } { 2 } < \frac { 1 } { 2 } .\tag{105}
$$

For real $| t | \leq 1 / 4$ , we have $| P ( t ) | \leq \| P \| _ { 1 / 4 } < 1$ , so the logarithm series in Eq. (104) represents the real logarithm in Eq. (101). For every integer $s \geq 1$ , Eq. (96) implies

$$
\begin{array} { r l r } {  { | D ^ { s } F _ { d } ( u ) [ h , \dots , h ] | = | \begin{array} { l } { \frac { d ^ { s } } { d t ^ { s } } \big ( F _ { d } ( u + t h ) - F _ { d } ( u ) \big ) \bigg | _ { t = 0 } } \\ { = s ! | [ t ^ { s } ] \log ( 1 + P ( t ) ) | \le 4 ^ { s } s ! \| \log ( 1 + P ) \| _ { 1 / 4 } \le \frac { 1 } { 2 } 4 ^ { s } s ! . } \end{array}  } } \end{array}\tag{106}
$$

Since $D ^ { s } F _ { d } ( u )$ is a symmetric tensor of order s, Eq. (97) now implies Eq. (99):

$$
\| D ^ { s } F _ { d } ( u ) \| _ { \mathrm { o p } } = \operatorname* { s u p } _ { \| h \| _ { 2 } \leq 1 } | D ^ { s } F _ { d } ( u ) [ h , \ldots , h ] | \leq { \frac { 1 } { 2 } } 4 ^ { s } s ! .
$$

Finally, fix a strictly positive $x \in \Delta _ { d }$ . The relative gradient is the orthogonal projection of $( 1 + \log x _ { i } ) _ { i = 1 } ^ { d }$ onto $\textstyle { \{ v : \sum _ { i } v _ { i } = 0 \} }$ . Since $\begin{array} { r } { \sum _ { i } ( \tilde { x } _ { i } - x _ { i } ) = 0 } \end{array}$ for every $\tilde { x } \in \Delta _ { d }$ , this projection does

not change the inner product with ${ \tilde { x } } - x ,$ so we may use the vector $( 1 + \log x _ { i } ) _ { i = 1 } ^ { d }$ in the Bregman formula. For every $\tilde { x } \in \Delta _ { d } .$ the Bregman divergence has the expansion

$$
\begin{array} { c l l } { { D _ { \psi _ { i } } ( \tilde { x } , x ) = \sum _ { i = 1 } ^ { d } { \tilde { x } _ { i } \log { \tilde { x } _ { i } } - \sum _ { i = 1 } ^ { d } { x _ { i } \log { x _ { i } } } - \sum _ { i = 1 } ^ { d } ( 1 + \log { x _ { i } } ) ( { \tilde { x } _ { i } } - x _ { i } ) } } } \\  { { = \sum _ { i = 1 } ^ { d } { \tilde { x } _ { i } \log { \frac { \tilde { x } _ { i } } { x _ { i } } } } , } } \end{array}\tag{107}
$$

where the second equality uses $\begin{array} { r } { \sum _ { i = 1 } ^ { d } \widetilde { x } _ { i } = \sum _ { i = 1 } ^ { d } x _ { i } = 1 } \end{array}$ . Since $0 \leq \tilde { x } _ { i } \leq 1$ , we have $\Sigma _ { i = 1 } ^ { d }$ x˜<sub>i</sub> log $\tilde { x } _ { i } \leq 0$ and hence

$$
D _ { \psi _ { d } } ( \tilde { x } , x ) \leq \sum _ { i = 1 } ^ { d } \tilde { x } _ { i } \log \frac { 1 } { x _ { i } } \leq \left( \sum _ { i = 1 } ^ { d } \tilde { x } _ { i } \right) \log \frac { 1 } { \operatorname* { m i n } _ { i } x _ { i } } = \log \frac { 1 } { \operatorname* { m i n } _ { i } x _ { i } } .
$$

This completes Proposition B.2.

Corollary 4.2 (Alternating multiplicative weights update). Let $A \in \mathbb { R } ^ { m \times n }$ and let $x _ { 0 } \in \Delta _ { n }$ and $y _ { 0 } \in$ $\Delta _ { m }$ be strictly positive initial strategies. Set C = 384 max $\{ \| A \| , 1 \}$ and $\Lambda _ { 0 } = \log ( 1 / \operatorname* { m i n } _ { 1 \le i \le n } x _ { 0 , i } )$ + $\log ( 1 / \operatorname* { m i n } _ { 1 \leq j \leq m } y _ { 0 , j } )$ . For every integer $T \geq 1$ , choose the constant step size $\begin{array} { r } { \eta = \frac { 1 } { 1 6 C ( \lfloor \log _ { 2 } T \rfloor + 2 ) } ; } \end{array}$ and generate a sequence of iterates for $k = 0 , \ldots , T - 1$ by the updates in $E q .$ (19). Then, the total regret satisfies $\begin{array} { r } { \mathrm { R e g } _ { T } \leq 1 6 C \Lambda _ { 0 } ( \log _ { 2 } T + 2 ) + \left( \frac { 1 2 8 } { 3 } + 4 e ^ { 2 } \right) C ^ { 2 } } \end{array}$ . In particular, the regret bound for uniform initial strategies is

$$
\mathrm { R e g } _ { T } \leq 1 6 C \log ( n m ) ( \log _ { 2 } T + 2 ) + \left( { \frac { 1 2 8 } { 3 } } + 4 e ^ { 2 } \right) C ^ { 2 } .
$$

Proof of Corollary $4 . 2 .$ . By Proposition B.2, the two entropy generators satisfy the relative Legendre assumptions. We apply $\operatorname { E q . }$ (99) to $F = F _ { n }$ and $G = F _ { m }$ . For every integer $s \geq 1$ , we obtain

$$
\frac { 1 } { s ! } \left( \operatorname* { s u p } _ { u } \lVert D ^ { s } F ( u ) \rVert _ { \mathrm { o p } } + \operatorname* { s u p } _ { w } \lVert D ^ { s } G ( w ) \rVert _ { \mathrm { o p } } \right) \leq 2 \cdot \frac { 1 } { 2 } 4 ^ { s } = 4 ^ { s } .
$$

Thus Assumption 3.1 holds with $K = 4$ . Here $\| A \|$ denotes the spectral norm. Since $J _ { A } ^ { \top } J _ { A } =$ di $\arg ( A ^ { \top } A , A A ^ { \top } )$ , we have $\| J _ { A } \| _ { \mathrm { o p } } = \| A \|$ . The constant in Lemma 3.6 therefore becomes

$$
C _ { 0 } = 2 4 \cdot 4 ^ { 2 } \operatorname* { m a x } \{ \| A \| , 1 \} = 3 8 4 \operatorname* { m a x } \{ \| A \| , 1 \} = C ,
$$

as stated in the corollary. For strictly positive initial strategies, Proposition B.2 yields

$$
\operatorname* { s u p } _ { \tilde { x } \in \Delta _ { n } } D _ { \psi _ { n } } ( \tilde { x } , x _ { 0 } ) + \operatorname* { s u p } _ { \tilde { y } \in \Delta _ { m } } D _ { \psi _ { m } } ( \tilde { y } , y _ { 0 } ) \leq \log \frac { 1 } { \operatorname* { m i n } _ { 1 \leq i \leq n } x _ { 0 , i } } + \log \frac { 1 } { \operatorname* { m i n } _ { 1 \leq j \leq m } y _ { 0 , j } } = \Lambda _ { 0 } < \infty ,
$$

which verifies Assumption 3.2, with initial comparator radius at most the stated $\Lambda _ { 0 }$

It remains to identify the updates with Eq. (19). Since $\begin{array} { r } { \boldsymbol { x } _ { k , i } = e ^ { \boldsymbol { u } _ { k , i } } / \sum _ { \ell = 1 } ^ { n } e ^ { \boldsymbol { u } _ { k , \ell } } } \end{array}$ , the first update in Eq. (AltOMD-Dual) satisfies

$$
x _ { k + 1 , i } = { \frac { e ^ { u _ { k , i } - \eta \left( A ^ { \top } y _ { k } \right) _ { i } } } { \sum _ { \ell = 1 } ^ { n } e ^ { u _ { k , \ell } - \eta \left( A ^ { \top } y _ { k } \right) _ { \ell } } } } = { \frac { x _ { k , i } e ^ { - \eta \left( A ^ { \top } y _ { k } \right) _ { i } } } { \sum _ { \ell = 1 } ^ { n } x _ { k , \ell } e ^ { - \eta \left( A ^ { \top } y _ { k } \right) _ { \ell } } } } ,
$$

where we divided the numerator and denominator by $\scriptstyle \sum _ { \ell = 1 } ^ { n } e ^ { u _ { k , \ell } }$ . The same calculation for $w _ { k + 1 } =$ $w _ { k } + \eta A x _ { k + 1 }$ establishes the second update in $\operatorname { E q . } \ ( 1 9 )$ . Set $N = \lfloor \log _ { 2 } T \rfloor$ . The stated step size is $\eta = 1 / [ 1 6 C ( N + 2 ) ] = 1 / [ 1 6 C _ { 0 } ( N + 2 ) ]$ , as required in Theorem 3.7. With the initial comparator radius bounded by $\Lambda _ { 0 }$ , Theorem 3.7 implies

$$
\mathrm { R e g } _ { T } \leq 1 6 C \Lambda _ { 0 } ( N + 2 ) + \left( \frac { 1 2 8 } { 3 } + 4 e ^ { 2 } \right) C ^ { 2 } \leq 1 6 C \Lambda _ { 0 } ( \log _ { 2 } T + 2 ) + \left( \frac { 1 2 8 } { 3 } + 4 e ^ { 2 } \right) C ^ { 2 } .
$$

For uniform initialization, we have $x _ { 0 , i } = 1 / n$ and $y _ { 0 , j } = 1 / m .$ so $\Lambda _ { 0 } = \log n + \log m = \log ( n m )$ This proves Eq. (20). □