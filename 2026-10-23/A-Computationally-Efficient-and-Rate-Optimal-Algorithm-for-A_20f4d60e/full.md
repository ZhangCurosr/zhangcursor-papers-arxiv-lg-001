# A Computationally Efficient and Rate-Optimal Algorithm for Adversarial Linear MDPs with (Aggregate) Bandit Feedback

Haolin Liu<sup>\*</sup> University of Virginia

Chen-Yu Wei<sup>†</sup> University of Virginia

Hanbin Zhou<sup>‡</sup> Hong Kong University of Science and Technology

## Abstract

We study episodic linear Markov decision processes with unknown transitions, adversarial losses, and bandit feedback. The state-of-the-art result for this setting [Liu et al., 2024a] either achieves $\sqrt { K }$ regret with a computationally inefficient algorithm or achieves suboptimal $K ^ { 3 / 4 }$ regret, where K is the number of episodes. Our work gives the first $\sqrt { K }$ regret with a polynomial-time algorithm.

Our method reduces adversarial linear MDPs to adversarial linear bandits whose action set K consists of policies’ expected feature vectors. The central challenge is that K is unknown and can be estimated only through exploration. We address this challenge by constructing an approximate separation oracle for K that either generates executable policies from queried features, or refines an outer approximation of K. This approach draws inspiration from the reduction of van Erven et al. [2025] for linear contextual bandits and the oracle-efficient linear bandit algorithm of Ito et al. [2019], but additionally deals with the imprecision of the oracle. The resulting algorithm requires only aggregate loss feedback, and to our knowledge. This is the first polynomial-time algorithm for adversarial MDPs with aggregate feedback beyond the tabular case.

## 1 Introduction

Achieving rate-optimal regret with polynomial computation is a central challenge in adversarial linear Markov decision processes (MDPs) with bandit feedback and unknown transitions. Starting with the $K ^ { 1 4 / 1 5 }$ regret bound of Luo et al. [2021], subsequent works improved the rate to $K ^ { 8 / 9 } , K ^ { 6 / 7 }$ , and $K ^ { 3 / 4 }$ [Dai et al., 2023, Sherman et al., 2023, Liu et al., 2024a], respectively. Meanwhile, Liu et al. [2024a] obtained the optimal $\sqrt { K }$ rate using a computationally inefficient method. Whether $\sqrt { K }$ regret can be achieved with an efficient (polynomial-time) algorithm has remained open.

Our work closes this gap by giving the first computationally efficient algorithm with poly $( d , H ) \sqrt { K }$ regret, where d is the feature dimension and H is the horizon length. In fact, our work handles the more challenging aggregate feedback model: the learner, despite observing the state sequence on the trajectory, only observes the total episode loss rather than the individual step losses. Prior work for the aggregate feedback model focuses either on the tabular case [Cohen et al., 2021, Lancewicki and Mansour, 2025, Ito et al., 2025] or the stochastic-loss setting [Cassel et al., 2024, Chen et al., 2025]. To our knowledge, our work gives the first polynomial-time algorithm for aggregate feedback model with adversarial losses beyond the tabular case.

Our approach builds on a reduction from linear MDPs to linear bandits. For each policy $\pi ,$ we stack its expected state-action features across layers as a policy-feature vector $\phi ^ { \pi }$ . Because the expected episode loss is linear in $\phi ^ { \pi }$ , the MDP can be viewed as a linear bandit whose action set consists of policy-feature vectors. This action set is unknown because $\phi ^ { \pi }$ depends on the unknown transition kernel. Existing methods based on this reduction [Kong et al., 2024, Liu et al., 2024a] discretize the policy class, estimate the policy-feature of every policy, and apply a linear-bandit algorithm to the resulting finite action set. While this achieves the optimal $\sqrt { K }$ regret rate [Liu et al., 2024a], their computation scales with an exponentially large policy class.

Under the same reduction idea, van Erven et al. [2025] develop an efficient reduction from linear contextual bandits to linear bandits. They construct a convex approximation $\widehat { \kappa }$ of $\kappa$ from past i.i.d. contexts and construct a linear optimization oracle for $\widehat { \kappa }$ . This oracle allows them to run a linear bandit algorithm (continuous exponential weights) over $\widehat { \kappa }$ without explicitly representing the set. Their linear contextual bandit setting is a special case of linear MDP when $H = 1$

Although the approach of van Erven et al. [2025] suggests a promising route for linear MDPs, a direct extension faces a major obstacle: the straightforward estimate $\widehat { \kappa }$ for linear MDP may not be convex due to the recursive structure of MDPs. Therefore, the log-concave sampling underlying continuous exponential weights no longer applies, and we also lose an exact optimization oracle over $\widehat { \kappa }$ . To address this, we leverage the idea of Ito et al. [2019]: we run continuous exponential weights over an outer convex set $\mathcal { O } \supseteq \mathcal { K }$ represented as an intersection of halfspaces (cuts). In Ito et al. [2019], whenever a point queried from O lies outside $\kappa .$ an exact oracle generates a separating cut to refine ${ \mathcal { O } } .$ Unlike in their setting, however, we lack an exact oracle for $\kappa$ . Thus, we construct cuts that exclude the queried point only when doing so preserves $\mathcal { O } \supseteq \mathcal { K }$ with high probability. We remark that while Ito et al. [2019] propose this mechanism to reduce oracle complexity, we use it for a completely different purpose of handling unknown action sets.

When the algorithm does not return a cut, it decomposes the queried point into a distribution over policies whose average policy feature approximates the query point. This will be the policy distribution played by the learner in that episode. Together, the two cases form a approximate separation procedure: it either constructs a finite policy mixture approximating the query, or returns a separating cut that refines O. We implement it using a Frank–Wolfe-style procedure. Besides a distribution over policies, our procedure simultaneously generates uncertainty measures, which will be crucial in constructing a bias-correction bonus fed to the learning procedure.

The paper is organized as follows. In Section 2 we describe the problem setting. In Section 3, we give a high-level overview of the main algorithm. In Section 4, we provide a sketch of regret analysis. In Section 5, we focus on the design of the approximate separation procedure. In Section $^ { 6 , }$ we analyze the computational complexity of the algorithm.

## 2 Preliminaries

We consider a finite-horizon Markov decision process with horizon H, layer-wise state sets $s _ { 1 } , \ldots , s _ { H + 1 }$ , a finite nonempty action set $\mathcal { A }$ with $A = | { \mathcal { A } } |$ , and a fixed initial state $s _ { 1 } \in S _ { 1 }$ . For each layer $h ,$ the transition kernel $P _ { h } ( \cdot \mid s , a )$ is a probability distribution on $S _ { h + 1 }$ . The kernels are unknown to the learner and fixed across episodes. A Markov policy $\pi = ( \pi _ { h } ) _ { h = 1 } ^ { H }$ consists of maps $\pi _ { h } : S _ { h }  \Delta ( { \mathcal { A } } )$ , where $\Delta ( \mathcal { A } )$ denotes the probability simplex over actions.

The learner interacts with the MDP for K episodes. Before interaction, an oblivious adversary fixes loss functions $\ell _ { k , h } : S _ { h } \times \mathcal { A } \to [ 0 , 1 ]$ for all episodes and layers. At the start of episode $k _ { : }$ , the learner selects a possibly random Markov policy $\pi _ { k }$ using past observations. Starting from $S _ { k , 1 } = s _ { 1 }$ , the trajectory is generated by $A _ { k , h } \sim \pi _ { k , h } ( \cdot \mid S _ { k , h } ) , S _ { k , h + 1 } \sim P _ { h } ( \cdot \mid S _ { k , h } , A _ { k , h } )$ , for every h. We assume the loss feedback observed by the

learner consists only of the total episode loss $\begin{array} { r } { L _ { k } = \sum _ { h = 1 } ^ { H } \ell _ { k , h } ( S _ { k , h } , A _ { k , h } ) } \end{array}$ , revealed at the end of the episode.   
We refer to this observation model as aggregate bandit feedback.

For a Markov policy π, define its expected loss in episode k by $\begin{array} { r } { V _ { k } ^ { \pi } = \mathbb { E } _ { P , \pi } \big [ \sum _ { h = 1 } ^ { H } \ell _ { k , h } ( S _ { h } , A _ { h } ) \big ] } \end{array}$ . Here $\mathbb { E } _ { P , \pi }$ denotes expectation over the trajectory generated by following policy π under transition kernels P. The objective we consider is to minimize regret relative to the best fixed Markov policy,

$$
{ \mathrm { R e g } _ { K } } = \mathbb { E } \left[ \sum _ { k = 1 } ^ { K } { V _ { k } ^ { \pi _ { k } } } \right] - \operatorname* { m i n } _ { \pi \mathrm { { M a r k o v } } } \sum _ { k = 1 } ^ { K } V _ { k } ^ { \pi } .
$$

Linear MDPs. We assume that transitions and losses are linear in known feature maps $\phi _ { h } : S _ { h } \times \mathcal { A }  \mathbb { R } ^ { d }$ Specifically, there exist unknown vectors $\psi _ { h } ( s ^ { \prime } ) \in \mathbb { R } ^ { d }$ and $\theta _ { k , h } \in \mathbb { R } ^ { d }$ such that, for every $k , h , s , a , s ^ { \prime }$ with $s \in S _ { h }$ and $s ^ { \prime } \in S _ { h + 1 }$

$$
\begin{array} { r } { P _ { h } ( s ^ { \prime } \mid s , a ) = \phi _ { h } ( s , a ) ^ { \top } \psi _ { h } ( s ^ { \prime } ) , \qquad \ell _ { k , h } ( s , a ) = \phi _ { h } ( s , a ) ^ { \top } \theta _ { k , h } . } \end{array}
$$

The transition coefficients $\psi _ { h }$ are fixed, while the loss coefficients $\theta _ { k , h }$ may vary across episodes. We assume $\lVert \phi _ { h } ( s , a ) \rVert _ { 2 } \leq 1 , \lVert \theta _ { k , h } \rVert _ { 2 } \leq \sqrt { d } .$ , and $\begin{array} { r } { \big \| \sum _ { s ^ { \prime } \in S _ { h + 1 } } \psi _ { h } ( s ^ { \prime } ) f ( s ^ { \prime } ) \big \| _ { 2 } \leq \sqrt { d } \| f \| _ { \infty } } \end{array}$ for any $f : S _ { h + 1 } \to \mathbb { R }$ without loss of generality [Jin et al., 2020, Wei et al., 2021]. For any $f : \mathcal { S } _ { h + 1 } \to \mathbb { R }$ , define $w _ { h } ( f ) =$ $\sum _ { s ^ { \prime } \in S _ { h + 1 } } \psi _ { h } ( s ^ { \prime } ) f ( s ^ { \prime } )$ and $P _ { h } f ( s , a ) = \mathbb { E } _ { s ^ { \prime } \sim P _ { h } ( \cdot \vert s , a ) } [ f ( s ^ { \prime } ) ] = \phi _ { h } ( s , a ) ^ { \top } w _ { h } ( f )$

Let $p \ = \ d H$ be the expected-feature dimension, obtained by stacking the d-dimensional features across all H layers. Set $\theta _ { k } = ( \theta _ { k , 1 } , \dots , \theta _ { k , H } )$ , and let $\phi ^ { \pi } = ( \phi _ { 1 } ^ { \pi } , \ldots , \phi _ { H } ^ { \pi } )$ with $\phi _ { h } ^ { \pi } = \mathbb { E } _ { P , \pi } [ \phi _ { h } ( S _ { h } , A _ { h } ) ]$ . Then $\| \phi ^ { \pi } \| _ { 2 } \leq \sqrt { H } , \| \theta _ { k } \| _ { 2 } \leq \sqrt { d H }$ , and $\mathbb { E } [ L _ { k } \mid \pi _ { k } = \pi ] = \langle \phi ^ { \pi } , \theta _ { k } \rangle = V _ { k } ^ { \pi }$ . Let ${ \mathcal { K } } = \{ \phi ^ { \pi } :$ π is a Markov policy} be the set of feasible policy-feature vectors. The policy-feature reduction of Liu et al. [2024a] relates regret to linear losses over this set through

$$
\mathrm { R e g } _ { K } = \mathbb { E } \left[ \sum _ { k = 1 } ^ { K } \langle \phi ^ { \pi _ { k } } , \theta _ { k } \rangle - \operatorname* { m i n } _ { u \in K } \sum _ { k = 1 } ^ { K } \langle u , \theta _ { k } \rangle \right]\tag{1}
$$

The set K depends on the unknown transitions. The reduction therefore yields an adversarial linear bandit problem over an unknown feasible set. The algorithm developed in Sections 4 and 5 achieves the following guarantee.

Theorem 2.1 (Main result). For adversarial linear MDPs with aggregate banditfeedback, Algorithm 1, with the complete implementation in Algorithm 4, achieves $\mathrm { R e g } _ { K } = \widetilde { O } ( d ^ { 5 } H ^ { 9 / 2 } \sqrt { K } )$ . Total computation is polynomial in $K , d , H , A$

Notation. For a positive integer m, write $[ m ] = \{ 1 , \dots , m \}$ , let ${ \mathbf I } _ { m }$ be the $m \times m$ identity matrix, and denote the Euclidean unit ball by $\mathbb { B } _ { 2 } ^ { m } = \left\{ x \in \mathbb { R } ^ { m } : \left\| x \right\| _ { 2 } \leq 1 \right\}$ . The indicator $\mathbf { 1 } \{ E \}$ equals one when E holds and zero otherwise. For $\Sigma \succeq 0 , \| x \| _ { \Sigma } \triangleq \sqrt { x ^ { \top } \Sigma x }$ . For a nonempty closed convex set $C \subseteq \mathbb { R } ^ { m }$ , write ${ \mathrm { P r o j } } _ { C } ( x ) =$ arg min $_ { z \in C } \| z - x \| _ { 2 } ^ { 2 }$ . For scalars $a \leq b .$ , define the clipping operator by $\mathrm { C L I P } _ { [ a , b ] } ( t ) = \operatorname* { m i n } \{ b , \operatorname* { m a x } \{ a , t \} \}$ For vectors, $\mathrm { C L I P } _ { [ a , b ] ^ { m } }$ applies this operator to each coordinate. For functions, clipping is applied pointwise. Throughout, $\widetilde O$ hides logarithmic factors in $K , d , H , A$

## 3 Algorithm Overview

The main algorithm is Algorithm 1. The algorithm maintains an outer approximation $\mathcal { O } \supset \kappa$ and a cumulative loss estimate $Z _ { k - 1 } \in \mathbb { R } ^ { p }$ . Since K may have empty interior, we actually require O to contain the enlarged set $\begin{array} { r } { \mathcal { K } _ { \zeta } = \mathcal { K } + \zeta \mathbb { B } _ { 2 } ^ { p } } \end{array}$ , for $\zeta = 1 / \sqrt { K }$ to ensure that the linear bandit algorithm is run over a set with non-zero volume. Each episode consists of the following steps:

Step 1. Construct a finite-support exploration distribution. The continuous exponential-weight distribution over O has density $q _ { k } ( x ) \propto \exp ( - \eta Z _ { k - 1 } ^ { \top } x ) \mathbf { 1 } \{ x \in \mathcal { O } \}$ . Let $\widehat { \mu } _ { k } \in \mathbb { R } ^ { p }$ and $\widehat { \Sigma } _ { k } \in \mathbb { R } ^ { \bar { p } \times p }$ be estimates of the mean and covariance of $q _ { k }$ . Instead of sampling a point directly from $q _ { k }$ , we construct a uniform distribution over the following 2p points:

$$
\left\{ \widehat { \mu } _ { k } + c _ { 0 } \widehat { \Sigma } _ { k } ^ { 1 / 2 } v : v \in \{ \pm e _ { 1 } , \pm e _ { 2 } , . . . , \pm e _ { p } \} \right\}\tag{2}
$$

where $e _ { 1 } , \ldots , e _ { p }$ are the standard basis vectors of $\mathbb { R } ^ { p } , c _ { 0 } > 0$ is a universal constant. This uniform distribution has mean $\widehat { \mu } _ { k }$ and covariance $( c _ { 0 } ^ { 2 } / p ) \widehat { \Sigma } _ { k }$ . This sampling scheme is similar to that used in SCRiBLe [Abernethy et al., 2009].

We aim for finite support because Step 2 explicitly enumerates and processes every possible support point before the learner samples a policy.

Step 2. Compile each support point or generate a cut. Denote the $2 p$ points in (2) as $\{ Y _ { k , i } \} _ { i = 1 } ^ { 2 p }$ . For each $i \in [ 2 p ]$ , APPROXSEP returns exactly one of the following:

Case 1. A set of policies $\{ \pi _ { k , i , j } \} _ { j = 1 } ^ { M }$ and an error certificate $\varepsilon _ { k , i }$ such that the average policy feature $\begin{array} { r } { \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \phi ^ { \pi _ { k , i , j } } } \end{array}$ approximates $Y _ { k , i }$ , with approximation error quantified by $\varepsilon _ { k , i }$

Case 2. A cut $( c , b )$ such that $c ^ { \top } Y _ { k , i } < b$ while, with high probability, $c ^ { \top } x \geq b$ for all $x \in \kappa _ { \zeta }$

If the return is Case 1 for all 2p points, the algorithm will proceed to Step 3; otherwise, the cut will be used to refine O, and the algorithm will go back to Step 1 to generate new points $\{ Y _ { k , i } \} _ { i = 1 } ^ { 2 p }$ with the refined O. At a high level, APPROXSEP implements an approximate separation oracle for $\kappa _ { \zeta }$ . The approximation error $\varepsilon _ { k , i }$ is constructed with confidence bounds from past transition data. APPROXSEP returns a cut only when the confidence bounds certify that the queried point is outside $\kappa _ { \zeta }$ . We will also show that: with high probability, the number of cuts produced by APPROXSEP is upper bounded by poly $( p , K )$ during the learning procedure, so the algorithm will finally proceed to the next step.

Step 3. Sample a policy. If APPROXSEP returns Case 1 for every $i \in [ 2 p ]$ , the learner draws $\left( I _ { k } , J _ { k } \right)$ uniformly from $[ 2 p ] \times [ M ]$ and executes $\pi _ { k , I _ { k } , J _ { k } }$ . Conditional on the completed preparation and $I _ { k }$ , averaging the policy feature over $J _ { k }$ gives $\begin{array} { r } { \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \phi ^ { \pi _ { k , I _ { k } , j } } } \end{array}$ , which approximates $Y _ { k , I _ { k } }$ within $\varepsilon _ { k , I _ { k } }$ by the guarantee in Case 1. Therefore, the overall algorithm approximates a standard linear bandit algorithm with the sampling scheme (2). After execution, the constructed loss estimator $\widehat { \theta _ { k } }$ and a bias-correction bonus $B _ { k }$ are used to update $Z _ { k }$

## 4 Sketch of Regret Analysis and Bonus Design

Let $\mu _ { k } = \mathbb { E } _ { X \sim q _ { k } } [ X ]$ and $\Sigma _ { k } = \operatorname { C o v } _ { X \sim q _ { k } } ( X )$ denote the exact mean and covariance of $q _ { k }$ after preparation of episode k; $\widehat { \mu } _ { k }$ and $\widehat { \Sigma } _ { k }$ are their estimates. Write $\mathbb { E } _ { k } [ \cdot ] = \mathbb { E } [ \cdot \mid \mathcal { F } _ { k } ]$ , where $\mathcal { F } _ { k }$ is the information available after preparation and before drawing $\left( I _ { k } , J _ { k } \right)$

The key step is the APPROXSEP procedure in Step 2 in Section 3. In this section, we provide a skeleton of regret analysis. Then in Section 5, we provide detailed explanation for APPROXSEP.

Algorithm 1 Continuous exponential weights with policy compilation and adaptive cuts   
Require: $\begin{array} { r } { c _ { 0 } = \frac { 1 } { 1 6 e } , \zeta = K ^ { - 1 / 2 } , R = \sqrt { H } + \zeta , M = K , \mathrm { a n d } \eta \in ( 0 , \frac { c _ { 0 } } { 1 2 8 \sqrt { d ^ { 3 } H ^ { 4 } } } ] . } \end{array}$   
1: $Z _ { 0 } \gets 0 ; \mathcal { O } \gets R \mathbb { B } _ { 2 } ^ { p } ; \mathcal { D } _ { 1 , h } \gets \emptyset$ for all $h \in [ H ]$   
2: for $k = 1 , \ldots , K$ do   
3: repeat   
4: restart← FALSE   
5: Define $q _ { k } ( x ) \propto \exp ( - \eta Z _ { k - 1 } ^ { \top } x ) \mathbf { 1 } \{ x \in \mathcal { O } \}$   
6: Estimate mean and covariance: $( \widehat { \mu } _ { k } , \widehat { \Sigma } _ { k } ) \gets$ MOMENTESTIMATE $( \mathcal { O } , Z _ { k - 1 } )$ (Section E.1)   
7: Let $\{ Y _ { k , i } \} _ { i = 1 } ^ { 2 p }$ be the set $\left\{ \widehat { \mu } _ { k } + c _ { 0 } \widehat { \Sigma } _ { k } ^ { 1 / 2 } v : v \in \{ \pm e _ { 1 } , \pm e _ { 2 } , \ldots , \pm e _ { p } \} \right\}$   
8: for $i = 1 , \ldots , 2 p$ do   
9: reply $ \mathrm { A P P R O X S E P } ( Y _ { k , i } , ( \mathcal { D } _ { k , h } ) _ { h = 1 } ^ { H } , M , \zeta )$ (Algorithm 2)   
10: if reply is a cut $( c , b )$ then   
11: $\mathcal { O }  \mathcal { O } \cap \{ x : c ^ { \top } x \geq b \}$   
12: restart← TRUE; break   
13: else   
14: $( ( \pi _ { k , i , j } ) _ { j = 1 } ^ { M } , \varepsilon _ { k , i } ) \gets \mathsf { r e p l y }$   
15: until restart= FALSE   
16: Draw $( I _ { k } , J _ { k } ) \sim \mathrm { U n i f } ( [ 2 p ] \times [ M ] )$ and execute $\pi _ { k } = \pi _ { k , I _ { k } , J _ { k } }$   
17: Observe feature vectors along the trajectory $( \phi _ { k , h } ) _ { h = 1 } ^ { H }$ and the total loss $L _ { k }$   
18: $\begin{array} { r } { \widehat { \theta } _ { k } \gets \left( \zeta ^ { 2 } \mathbf { I } _ { p } + \frac { c _ { 0 } ^ { 2 } } { p } \widehat { \boldsymbol { \Sigma } } _ { k } \right) ^ { - 1 } ( Y _ { k , I _ { k } } - \widehat { \mu } _ { k } ) L _ { k } } \end{array}$   
19: $Z _ { k } \gets Z _ { k - 1 } + \widehat { \theta } _ { k } - B _ { k }$ where $\begin{array} { r } { B _ { k } = \frac { 1 0 \eta p \varepsilon _ { k } } { c _ { 0 } } Z _ { k - 1 } } \end{array}$ with $\begin{array} { r } { \varepsilon _ { k } = \sqrt { \frac { 1 } { 2 p } \sum _ { i = 1 } ^ { 2 p } \varepsilon _ { k , i } ^ { 2 } } . } \end{array}$   
20: $\mathcal { D } _ { k + 1 , h } \gets \mathcal { D } _ { k , h } \cup \{ ( \phi _ { k , h } , S _ { k , h + 1 } ) \}$ for all $h \in [ H ]$

Recall that in episode k, Algorithm 1 forms $2 p$ target points (Line 7 of Algorithm 1). For each target $Y _ { k , i } \in \mathbb { R } ^ { p }$ the APPROXSEP procedure (Line 9 of Algorithm 1) either returns a cut or a list of M policies and a certificate $\varepsilon _ { k , i }$ Let $\begin{array} { r } { x _ { k , i } = \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \phi ^ { \pi _ { k , i , j } } } \end{array}$ . Then Lemma 5.2 shows $\| x _ { k , i } - Y _ { k , i } \| _ { 2 } \leq \varepsilon _ { k , i }$ . Further define $\begin{array} { r } { \overline { { x } } _ { k } = \frac { 1 } { 2 p } \sum _ { i = 1 } ^ { 2 p } x _ { k , i } } \end{array}$ Since $\begin{array} { r } { \widehat { \mu } _ { k } = \frac { 1 } { 2 p } \sum _ { i = 1 } ^ { 2 p } Y _ { k , i } } \end{array}$ , Jensen’s inequality gives

$$
\begin{array} { r } { \| \overline { { \boldsymbol { x } } } _ { k } - \widehat { \mu } _ { k } \| _ { 2 } \leq \varepsilon _ { k } : = \sqrt { \frac { 1 } { 2 p } \sum _ { i = 1 } ^ { 2 p } \varepsilon _ { k , i } ^ { 2 } } . } \end{array}\tag{3}
$$

Essentially, $\widehat { \mu } _ { k }$ is the mean feature suggested by the underlying linear bandit algorithm, while $\overline { { x } } _ { k }$ is the true expected feature of the policy executed by the learner. Their mismatch indicates that the underlying linear bandi algorithm is facing misspecified or corrupted feedback.

Quantifying bias and injecting bias-correction bonus. To handle misspecification, we inject bias-correction bonus $B _ { k }$ into the continuous exponential weight update [Liu et al., 2024c, van Erven et al., 2025]. To understand the design of $B _ { k }$ , we perform the following analysis. When the regret comparator has policy feature $u ,$ the true regret in episode k is $\langle \overline { { x } } _ { k } - u , \theta _ { k } \rangle$ , while the regret seen by the exponential weight algorithm is $\langle \mu _ { k } - u , \widehat { \theta } _ { k } - B _ { k } \rangle$ Their difference is

$$
\underbrace { \langle \overline { { x } } _ { k } - u , \theta _ { k } \rangle } _ { \mathrm { t r u e ~ r e g e t } } - \underbrace { \langle \mu _ { k } - u , \mathbb { E } _ { k } [ \widehat { \theta } _ { k } ] - B _ { k } \rangle } _ { \mathrm { p e r c e i v e d ~ r e g e t } } = \underbrace { \langle \overline { { x } } _ { k } - \mu _ { k } , \theta _ { k } \rangle } _ { \mathrm { f e a t u r e ~ e r r o r } } + \underbrace { \langle \mu _ { k } - u , \theta _ { k } - \mathbb { E } _ { k } [ \widehat { \theta } _ { k } ] \rangle } _ { \mathrm { l o s ~ s e s i m a t o r ~ b i a s } } + \underbrace { \langle \mu _ { k } - u , B _ { k } \rangle } _ { \mathrm { c o r r e c t i o n ~ t e r m } } .\tag{4}
$$

The first term (i.e., feature error) comes from the mismatch between the estimated and true policy feature. It is upper bounded by ${ \sqrt { p } } { \big ( } \varepsilon _ { k } + \| { \widehat { \mu } } _ { k } - \mu _ { k } \| _ { 2 } { \big ) }$ by (3) with the assumption $\| \theta _ { k } \| _ { 2 } \leq \sqrt { p }$ . The term $\| \widehat { \mu } _ { k } - \mu _ { k } \| _ { 2 }$ can be

made negligible by running MOMENTESTIMATE with high enough accuracy. Therefore, this part is dominated by $\sqrt { p } \varepsilon _ { k }$

The second term (i.e., loss estimator bias) comes from the bias of $\widehat { \theta } _ { k }$ . It can be upper bounded as $\langle \mu _ { k } - u , \theta _ { k } -$ $\begin{array} { r } { \mathbb { E } _ { k } [ \widehat { \theta } _ { k } ] \rangle \leq \| \mu _ { k } - u \| _ { \Sigma _ { k } ^ { - 1 } } \| \theta _ { k } - \mathbb { E } _ { k } [ \widehat { \theta } _ { k } ] \| _ { \Sigma _ { k } } } \end{array}$ . Lemma C.2 gives $\begin{array} { r } { \| \theta _ { k } - \mathbb { E } _ { k } [ \widehat { \theta } _ { k } ] \| _ { \Sigma _ { k } } \leq \frac { 5 p \varepsilon _ { k } } { 2 c _ { 0 } } } \end{array}$ , and hence

$$
\begin{array} { r } { \mathrm { l o s s \ e s t i m a t o r \ b i a s } = \langle \mu _ { k } - u , \theta _ { k } - \mathbb { E } _ { k } [ \widehat \theta _ { k } ] \rangle \leq \frac { 5 p \varepsilon _ { k } } { 2 c _ { 0 } } \| \mu _ { k } - u \| _ { \Sigma _ { k } ^ { - 1 } } . } \end{array}\tag{5}
$$

This bias depends on the regret comparator u and can be large if distribution $q _ { k }$ puts low weight on u. The solution to this term has become standard [Lee et al., 2020, Zimmert and Lattimore, 2022, Liu et al., 2024c]: The continuous exponential weight $q _ { k } ( x ) \propto \exp ( - \eta Z _ { k - 1 } ^ { \top } x ) \mathbf { 1 } \{ x \in \mathcal { O } \}$ over x $\in \mathbb { R } ^ { p }$ can be viewed as Followthe-Regularized-Leader with a p-self-concordant barrier [Bubeck and Eldan, 2015]. Lemma C.1 therefore gives $\| \mu _ { k } - u \| _ { \Sigma _ { \star } ^ { - 1 } } \leq - 4 \eta \langle \mu _ { k } - u , Z _ { k - 1 } \rangle + 2 4 p$ . Combining with (5) and recalling $\begin{array} { r } { B _ { k } = \frac { 1 0 \eta p \varepsilon _ { k } } { c _ { 0 } } Z _ { k - 1 } } \end{array}$ , we can bound the sum of the last two terms in (4) as

$$
\begin{array} { r } { \mathrm { l o s s \ e s t i m a t o r \ b i a s + c o r r e c t i o n \ t e r m } = \langle \mu _ { k } - u , \theta _ { k } - \mathbb { E } _ { k } [ \widehat \theta _ { k } ] + B _ { k } \rangle \le \frac { 6 0 p ^ { 2 } \varepsilon _ { k } } { c _ { 0 } } . } \end{array}
$$

This explains why we include the bonus term $B _ { k }$ in Line 19 of Algorithm 1. The bonus removes the comparatordependent term $O ( p ^ { 2 } \varepsilon _ { k } \| \mu _ { k } - u \| _ { \Sigma _ { k } ^ { - 1 } } )$ from the bias upper bound and introduces a comparator-independent overhead $O ( p ^ { 2 } \varepsilon _ { k } )$ . Overall, the discrepancy between the true regret and the perceived regret is of order $O \big ( p ^ { 2 } \textstyle \sum _ { k = 1 } ^ { K } \bar { \varepsilon } _ { k } \big )$

Bounding the perceived regret. The perceived regret in (4) follows the standard analysis of exponential weights. Even though the decision sets change over time, since they are always shrinking and the true regret comparator always lies in $\kappa \subseteq \mathcal { O }$ with high probability, the standard FTRL analysis applies (Lemma C.4).

The true regret is then bounded by the sum of the perceived regret and the discrepancy:

Lemma 4.1 (Regret bound). Algorithm 1 with the optimal choice ofη ensures

$$
\mathrm { R e g } _ { K } \leq \tilde { \mathcal { O } } \biggl ( \underbrace { d ^ { 3 / 2 } H ^ { 5 / 2 } \sqrt { K } } _ { \mathrm { p e r c e i v e d r e g r e t } } + \underbrace { d ^ { 2 } H ^ { 2 } \sqrt { K \mathbb { E } \left[ \sum _ { k = 1 } ^ { K } \varepsilon _ { k } ^ { 2 } \right] } } _ { \mathrm { b i a s + c o r r e c t i o n } } \biggr ) .\tag{6}
$$

To obtain $\widetilde { O } ( \sqrt { K } )$ regret from Eq. (6), it remains to bound $\textstyle \sum _ { k } \varepsilon _ { k } ^ { 2 }$ by polynomial factors in d, H and logarithmic factors in K. Section 5 establishes this guarantee, and we give the final bound for the “bias + correction” term above in Section 5.3.

## 5 From a target feature vector to a policy list

This section builds APPROXSEP, the procedure in Algorithm 1 that performs approximate separation:

Input: $y \in \mathbb { R } ^ { p }$   
Output: exactly one of the following:   
$( \{ \pi _ { j } \} _ { j = 1 } ^ { M } , \varepsilon )$ such that $\begin{array} { r } { \| y - \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \phi ^ { \pi _ { j } } \| _ { 2 } \leq \varepsilon . } \end{array}$   
$( c , b )$ such that $c ^ { \top } y < b$ and $c ^ { \top } x \geq b$ for all $\boldsymbol { x } \in \kappa _ { \zeta }$  
Figure 1: Approximate separation — realized by APPROXSEP (Algorithm 2)

To build approximate separation, we leverage the ability to perform approximate linear optimization:

```latex
Input: $c \in \mathbb { R } ^ { p }$
Output: $( \pi , { \widehat { \phi } } , \Delta )$ such that $\begin{array} { r } { \| \widehat { \phi } - \phi ^ { \pi } \| _ { 2 } \leq \Delta } \end{array}$ and $\begin{array} { r } { c ^ { \top } \widehat { \phi } \leq \operatorname* { m i n } _ { x \in { \mathcal { K } } } c ^ { \top } x + \| c \| _ { 2 } \Delta . } \end{array}$
```  
Figure 2: Approximate linear optimization — realized by APPROXLO (Algorithm 3)

In the analysis below, we will see how ε in Fig. 1 depends on $\Delta$ in $\mathrm { F i g . } 2$ (Line 9 of Algorithm 2). In implementing approximate linear optimization, the learner has access to previous transition data $( \mathcal { D } _ { k , h } ) _ { h = 1 } ^ { H }$ as the source of information for the unknown set $\kappa .$ The error $\Delta$ will be related to confidence bound of the transition (see (10)).

Since $\pi$ is selected based on the same data used to estimate its feature, we establish the confidence bounds underlying $\Delta$ uniformly over all possible output policies. We obtain these bounds by restricting the policy class to have controlled complexity and applying a covering argument, allowing the same data to be reused across adaptive linear-optimization queries.

In Section 5.1, we focus on building approximate separation from approximate linear optimization. Then in Section 5.2, we investigate the implementation of approximate linear optimization.

## 5.1 Constructing the policy list or excluding the target

This subsection constructs APPROXSEP (Algorithm 2), assuming access to APPROXLO. The algorithm can be viewed as an inexact Frank–Wolfe procedure for the objective $\begin{array} { r } { f ( x ) = \frac { 1 } { 2 } \| x - y \| _ { 2 } ^ { 2 } } \end{array}$ over $\kappa .$ . As in standard Frank–Wolfe, the iterate is an average over extreme points. Thus, the procedure also naturally produces a policy list whose average feature approximates the target $y .$

Algorithm 2 APPROXSEP $( y , ( \mathcal D _ { k , h } ) _ { h = 1 } ^ { H } , M , \zeta )$   
Require: Target $y \in \mathbb { R } ^ { p }$ , transition data $( \mathcal { D } _ { k , h } ) _ { h = 1 } ^ { H }$ , policy count M, enlargement radius $\zeta .$   
1: $\widehat { x } _ { 0 } \gets 0 .$   
2: for $j = 1 , \dots , M$ do   
3: $c _ { j } \gets { \widehat { x } } _ { j - 1 } - y$   
4: $( \pi _ { j } , \widehat { \phi } _ { j } , \Delta _ { j } )  \mathbf { A }$ PPROXL $O ( c _ { j } ; ( D _ { k , h } ) _ { h = 1 } ^ { H } )$ (Algorithm 3)   
5: $b _ { j }  c _ { j } ^ { \top } \widehat { \phi } _ { j } - \| c _ { j } \| _ { 2 } ( \Delta _ { j } + \zeta )$   
6: if $c _ { j } ^ { \top } y < b _ { j }$ then   
7: return cut $( c _ { j } , b _ { j } )$   
8: $\begin{array} { r } { \widehat { x } _ { j } \gets \frac { j - 1 } { j } \widehat { x } _ { j - 1 } + \frac { 1 } { j } \widehat { \phi } _ { j } } \end{array}$   
9: return $( ( \pi _ { j } ) _ { j = 1 } ^ { M } , \varepsilon )$ , where $\begin{array} { r } { \varepsilon = \sqrt { 4 \zeta ^ { 2 } + \frac { 6 } { M } \sum _ { j = 1 } ^ { M } \Delta _ { j } ^ { 2 } + \frac { 8 R ^ { 2 } \left( 1 + \log { M } \right) } { M } } } \end{array}$

First, we ensure that if APPROXLO performs as desired, then APPROXSEP returns a cut only when $y \notin \kappa _ { \zeta }$ This is shown in the following lemma:

Lemma 5.1. Suppose APPROXLO (called in Line 4 of Algorithm 2) satisfies the property in Fig. 2 and assume $y \in { \mathcal { K } } _ { \zeta } .$ . Then APPROXSEP will NOT return a cut.

Proof. By the guarantee of APPROXLO, we have

$$
c _ { j } ^ { \top } \widehat { \phi } _ { j } \leq \operatorname* { m i n } _ { x \in \mathcal K } c _ { j } ^ { \top } x + \| c _ { j } \| _ { 2 } \Delta _ { j } \leq \big ( \operatorname* { m i n } _ { x \in \mathcal K _ { \zeta } } c _ { j } ^ { \top } x + \zeta \| c _ { j } \| _ { 2 } \big ) + \| c _ { j } \| _ { 2 } \Delta _ { j } \leq c _ { j } ^ { \top } y + \| c _ { j } \| _ { 2 } ( \Delta _ { j } + \zeta )
$$

where we use the assumption $y \in \kappa _ { \zeta }$ in the last inequality. Thus, the $\mathbf { \hat { \Omega } } ^ { 6 6 } \mathbf { \dot { 1 } } \mathbf { \dot { f } } ^ { 3 }$ condition in Line 7 of Algorithm 2 does not hold, and the algorithm will not return a cut. □

Next, we ensure that if no cut is returned by APPROXSEP, the approximation error for y is controlled.

Lemma 5.2. Suppose APPROXLO (called in Line 4 of Algorithm 2) satisfies the property in $F i g . 2$ and suppose APPROXSEP does NOT return a cut. Then $\begin{array} { r } { \left\| \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \bar { \phi } ^ { \pi _ { j } } - y \right\| _ { 2 } \leq \varepsilon } \end{array}$ (defined in Line 9).

Proof. This proof is related to the analysis for Frank-Wolfe algorithms. For any $j = 1 , \dots , M$

$$
\begin{array} { r l r } {  { \| \widehat { x } _ { j } - y \| ^ { 2 } = \| ( 1 - \frac { 1 } { j } ) \widehat { x } _ { j } - 1 + \frac { 1 } { j } \widehat { \phi } _ { j } - y \| ^ { 2 } } } \quad & { \mathrm { ( L i n e ~ 8 ) } } \\ & { } & { = \| \widehat { x } _ { j - 1 } - y \| ^ { 2 } - \frac { 2 } { j } \langle \widehat { x } _ { j - 1 } - y , \widehat { x } _ { j - 1 } - \widehat { \phi } _ { j } \rangle + \frac { 1 } { j ^ { 2 } } \| \widehat { x } _ { j - 1 } - \widehat { \phi } _ { j } \| ^ { 2 } } \\ & { } & { \leq \| \widehat { x } _ { j - 1 } - y \| ^ { 2 } - \frac { 2 } { j } \| \widehat { x } _ { j - 1 } - y \| ^ { 2 } - \frac { 2 } { j } \langle c _ { j } , y - \widehat { \phi } _ { j } \rangle + \frac { 4 R ^ { 2 } } { j ^ { 2 } } \qquad \mathrm { ( } c _ { j } = \widehat { x } _ { j - 1 } - y \mathrm { ) } } \\ & { } & { \leq \| \widehat { x } _ { j - 1 } - y \| ^ { 2 } - \frac { 2 } { j } \| \widehat { x } _ { j - 1 } - y \| ^ { 2 } + \frac { 2 } { j } \| c _ { j } \| \mathrm { ( \Delta } _ { j } + \zeta \mathrm { ) } + \frac { 4 R ^ { 2 } } { j ^ { 2 } } } \\ & { } & { \mathrm { ( t h e ~ } ^ { \ast 1 \dag \mathrm { r } } \mathrm { c o n d i t i o n ~ i n e ~ l o r ~ a f ~ o f ~ A l g o r i t t m ~ 2 ~ d o e s ~ n o t ~ h o l d ) } } \\ & { } &  \leq ( 1 - \frac { 1 } { j } ) \| \widehat { x } _ { j - 1 } - y \| ^ { 2 } + \frac { 2 \Delta _ { j } ^ { 2 } + 2 \zeta ^ { 2 } } { j } + \frac { 4 R ^ { 2 } } { j ^ { 2 } } . \qquad ( c _ { j } = \widehat { x } _  j - \end{array}
$$

Expanding the recursion using Lemma D.1 yields

$$
\| \widehat { x } _ { M } - y \| _ { 2 } ^ { 2 } \leq 2 \zeta ^ { 2 } + \frac { 2 } { M } \sum _ { j = 1 } ^ { M } \Delta _ { j } ^ { 2 } + \frac { 4 R ^ { 2 } ( 1 + \log M ) } { M } .\tag{7}
$$

Let $\begin{array} { r } { x = \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \phi ^ { \pi _ { j } } } \end{array}$ . Since $\begin{array} { r } { \widehat { x } _ { M } = \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \widehat { \phi } _ { j } } \end{array}$ , by the assumed property of APPROXLO,

$$
\| x - \widehat { x } _ { M } \| _ { 2 } ^ { 2 } \leq \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \Big \| \phi ^ { \pi _ { j } } - \widehat { \phi } _ { j } \Big \| _ { 2 } ^ { 2 } \leq \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \Delta _ { j } ^ { 2 } .\tag{8}
$$

Combining (8) and (7) and using $\| x - y \| _ { 2 } ^ { 2 } \leq 2 \| x - \widehat { x } _ { M } \| _ { 2 } ^ { 2 } + 2 \| \widehat { x } _ { M } - y \| _ { 2 } ^ { 2 }$ finishes the proof.

Together, the two lemmas establish the desired dichotomy: feasible targets are never excluded, while any target not excluded is approximated by a policy list with controlled error. The next subsection relates this error to confidence bounds for the unknown transitions.

## 5.2 Planning and evaluating from past transitions

In this subsection, we develop APPROXLO that achieves the property in Fig. 2, with the help of the transition data $( \mathcal { D } _ { k , h } ) _ { h = 1 } ^ { H }$ collected before. The algorithm is adapted from the standard Least-Square Value Iteration (LSVI) procedure [Jin et al., 2020, Wang et al., 2020] and is displayed in Algorithm 3. Its guarantee is summarized in the following theorem:

Theorem 5.3 (Certified approximate linear optimization). Let $\beta = C _ { \beta } d \sqrt { \log ( K d H A / \delta ) }$ , where $C _ { \beta }$ is a sufficiently large universal constant. Then, with probability at least $1 - \delta ,$ , simultaneouslyfor every episode k and every call to Algorithm 3, the returned tuple $( \pi , { \widehat { \phi } } , \Delta )$ satisfies $\widehat { \phi } \in \sqrt { H } \mathbb { B } _ { 2 } ^ { p } , \Delta \in [ 0 , 2 \sqrt { H } ]$ , and

$$
\begin{array} { r } { \| \widehat \phi - \phi ^ { \pi } \| _ { 2 } \leq \Delta , \qquad c ^ { \top } \widehat \phi - \| c \| _ { 2 } \Delta \leq \operatorname* { m i n } _ { u \in \mathcal K } c ^ { \top } u . } \end{array}\tag{9}
$$

Moreover, the returned certificate satisfies

$$
\Delta \leq 1 0 \sqrt { p } \beta ^ { 2 } H \mathbb { E } _ { P , \pi } \left[ \sum _ { h = 1 } ^ { H } \| \phi _ { h } ( S _ { h } , A _ { h } ) \| _ { { \Lambda _ { k , h } ^ { - 1 } } } \right] + H \tau \log A .\tag{10}
$$

Below, we overview the elements in Algorithm 3.

Task 1: Policy optimization. We use optimistic LSVI [Jin et al., 2020] to find an approximately optimal policy under the cost function $\phi _ { h } ( s , a ) ^ { \top } c _ { h }$ . Different from Jin et al. [2020], we output a Gibbs (softmin) policy (Line 6 of Algorithm 3) instead of a greedy one, and use soft state value (Line 5). The reason of doing this is to make the mapping from c to the output policy feature $\widehat { \phi }$ be Lipschitz, leading to a bounded log covering number for the class of mappings (Appendix D). The clipping for $Q _ { h } ^ { \mathrm { { p l } } }$ is at $\sqrt { H }$ because $| \langle \phi ( s , a ) , \bar { c } \rangle | \leq \| \phi ( s , a ) \| _ { 2 } \| \bar { c } \| _ { 2 } \leq \sqrt { H }$ (Line 4). This implies $| V _ { h } ^ { \mathrm { p l } } ( s ) | \leq \sqrt { H } + \tau \log A \leq \sqrt { H } + 1$

The soft value preserves optimism because $V _ { h } ^ { \mathrm { p l } } ( s ) \leq \operatorname* { m i n } _ { a } Q _ { h } ^ { \mathrm { p l } } ( s , a )$ , giving $\begin{array} { r } { V _ { 1 } ^ { \mathrm { p l } } ( s _ { 1 } ) \le \operatorname* { m i n } _ { u \in \mathcal { K } } \bar { c } ^ { \top } u } \end{array}$ . Thus, to obtain the second guarantee in (9), it suffices to bound $\overline { { c } } ^ { \top } \widehat { \phi } - V _ { 1 } ^ { \mathrm { p l } } ( s _ { 1 } )$ . This difference can be decomposed into

$$
\begin{array} { r } { \bar { c } ^ { \top } \widehat { \phi } - V _ { 1 } ^ { \mathrm { p l } } ( s _ { 1 } ) = \underbrace { \bar { c } ^ { \top } ( \widehat { \phi } - \phi ^ { \pi } ) } _ { \mathrm { f e a t u r e ~ e s t i m a t i o n ~ e r r o r } } + \underbrace { \bar { c } ^ { \top } \phi ^ { \pi } - V _ { 1 } ^ { \mathrm { p l } } ( s _ { 1 } ) } _ { \mathrm { p l a n n i n g ~ e r r o r } } } \end{array}\tag{11}
$$

Algorithm 3 APPROXLO $( c , ( \mathcal D _ { k , h } ) _ { h = 1 } ^ { H } )$   
Require: $c \in \mathbb { R } ^ { p } .$ dataset $( \underline { { \mathcal { D } } } _ { k , h } ) _ { h = 1 } ^ { H } .$   
Parameters: $\tau = 1 / ( H \sqrt { K } \log ( \breve { e } A ) )$ and $\beta$ as in Theorem $5 . 3 .$   
Define: $\begin{array} { r } { \Lambda _ { k , h } = \mathbf { I } _ { d } + \sum _ { ( u , s ^ { \prime } ) \in \mathcal { D } _ { k , h } } u u ^ { \top } , \nu _ { k , h } ( s , a ) = \| \phi ( s , a ) \| _ { \Lambda _ { k , h } ^ { - 1 } } } \end{array}$ , and ${ \bar { c } } = c / \| c \| _ { 2 } .$   
Define: $\begin{array} { r } { \widehat { P } _ { k , h } f ( \boldsymbol { s } , \boldsymbol { a } ) = \phi _ { h } ( \boldsymbol { s } , \boldsymbol { a } ) ^ { \top } \Lambda _ { k , h } ^ { - 1 } \sum _ { ( \boldsymbol { u } , \boldsymbol { s } ^ { \prime } ) \in \mathcal { D } _ { k , h } } \boldsymbol { u } f ( \boldsymbol { s } ^ { \prime } ) } \end{array}$ for $f : S  \mathbb { R } .$   
1: Let $V _ { H + 1 } ^ { \mathrm { p l } } ( s ) = 0 \in \mathbb { R }$ and $V _ { H + 1 } ^ { \mathrm { f t } } ( s ) = 0 \in \mathbb { R } ^ { p }$ and $\widehat { U } _ { H + 1 } ( s ) = 0 \in \mathbb { R }$ for all $s \in S _ { H + 1 }$   
2: for $h = H , H - 1 , \ldots , 1$ do   
3: // Task 1: policy optimization: choose the policy $\pi$   
4: $Q _ { h } ^ { \mathrm { p l } } ( s , a ) \gets \mathrm { C L I P } _ { [ - \sqrt { H } , \sqrt { H } ] } \left( \phi _ { h } ( s , a ) ^ { \top } \bar { c } _ { h } + \widehat { P } _ { k , h } V _ { h + 1 } ^ { \mathrm { p l } } ( s , a ) - ( \sqrt { H } + 1 ) \beta \nu _ { k , h } ( s , a ) \right)$   
5: $\begin{array} { r } { V _ { h } ^ { \mathrm { p l } } ( s ) \gets - \tau \log \sum _ { a } \exp ( - Q _ { h } ^ { \mathrm { p l } } ( s , a ) / \tau ) . } \end{array}$   
6: $\pi _ { h } ( a \mid s ) \propto \exp ( - Q _ { h } ^ { \mathrm { p l } } ( s , a ) / \tau ) .$   
7: // Task 2: feature evaluation: estimate $\phi ^ { \pi }$   
8: $\boldsymbol { r } _ { h } ^ { \mathrm { f t } } ( \boldsymbol { s } , \boldsymbol { a } ) \gets ( 0 , \dots , 0 , \phi _ { h } ( \boldsymbol { s } , \boldsymbol { a } ) , 0 , \dots , 0 ) \in \mathbb { R } ^ { p } .$   
9: $Q _ { h } ^ { \mathrm { f t } } ( s , a ) \gets \mathrm { C L I P } _ { [ - 1 , 1 ] ^ { p } } ( r _ { h } ^ { \mathrm { f t } } ( s , a ) + \widehat { P } _ { k , h } V _ { h + 1 } ^ { \mathrm { f t } } ( s , a ) )$   
10: $\begin{array} { r } { V _ { h } ^ { \mathrm { f t } } ( s ) \gets \sum _ { a } \pi _ { h } ( a \mid s ) Q _ { h } ^ { \mathrm { f t } } ( s , a ) } \end{array}$   
11: // Task 3: uncertainty evaluation: upper bound $U _ { k } ^ { \pi }$   
12: $Q _ { h } ^ { \mathrm { u n c } } \gets \mathrm { C L I P } _ { [ 0 , H - h + 1 ] } ( ( 1 + H \beta ) \nu _ { k , h } + \widehat { P } _ { k , h } \widehat { U } _ { h + 1 } )$   
13: $\begin{array} { r } { \widehat { U } _ { h } ( s ) \gets \sum _ { a } \pi _ { h } ( a \mid s ) Q _ { h } ^ { \mathrm { u n c } } ( s , a ) } \end{array}$   
14: $\widehat { \phi }  \mathrm { P r o j } _ { \sqrt { H } \mathbb { B } _ { 2 } ^ { p } } ( V _ { 1 } ^ { \mathrm { f t } } ( s _ { 1 } ) )$   
15: $\Delta $ min $\{ 5 \sqrt { p } \beta \widehat { U } _ { 1 } ( s _ { 1 } ) + $ Hτ log $A , 2 { \sqrt { H } } \}$   
16: return $( \pi , { \widehat { \phi } } , \Delta )$

The feature-evaluation error is handled in the feature-evaluation step below. The planning error comes from Gibbs smoothing and the transition-prediction errors, which sum to $\bar { c } ^ { \top } \hat { \phi ^ { \pi } } - V _ { 1 } ^ { \mathrm { p l } } ( s _ { 1 } ) \hat { \leq } 2 ( \sqrt { \bar { H } } + 1 ) \beta U _ { k } ^ { \pi } + H \tau \log A$ where $\begin{array} { r } { U _ { k } ^ { \pi } = \mathbb { E } _ { P , \pi } [ \sum _ { h } \nu _ { k , h } ( S _ { h } , A _ { h } ) ] } \end{array}$

Task 2: Feature evaluation. Now fix the Gibbs policy π constructed in the policy-optimization step, and estimate its policy feature $\phi ^ { \pi }$ . Define vector $r _ { h } ^ { \mathrm { f t } } ( s , a ) \in \mathbb { R } ^ { p }$ as in Line 8. Then we have

$$
\mathbb { E } _ { P , \pi } \left[ \sum _ { h = 1 } ^ { H } r _ { h } ^ { \mathrm { f i } } ( S _ { h } , A _ { h } ) \right] = \left( \mathbb { E } _ { P , \pi } [ \phi _ { 1 } ( S _ { 1 } , A _ { 1 } ) ] , \ldots , \mathbb { E } _ { P , \pi } [ \phi _ { H } ( S _ { H } , A _ { H } ) ] \right) = \phi ^ { \pi } .
$$

Thus estimating $\phi ^ { \pi }$ is a policy-evaluation problem with p-dimensional vector rewards. We estimate $\phi ^ { \pi }$ by backward policy evaluation, obtaining $\widehat { \phi } \in \sqrt { H } \mathbb { B } _ { 2 } ^ { p }$ . A backward induction shows that the estimation error is controlled by $U _ { k } ^ { \pi }$ , giving $\| \widehat \phi - \phi ^ { \pi } \| _ { 2 } \le \sqrt { p } \beta U _ { k } ^ { \pi }$ . Since $\begin{array} { r } { \| \bar { c } \| _ { 2 } \leq 1 , \left| \bar { c } ^ { \top } \widehat { \phi } - \bar { c } ^ { \top } \phi ^ { \pi } \right| \leq \sqrt { p } \beta U _ { k } ^ { \pi } } \end{array}$ . Adding this estimation error to the policy optimization bound (11) gives

$$
\bar { c } ^ { \top } \widehat { \phi } - \left[ ( \sqrt { p } + 2 \sqrt { H } + 2 ) \beta U _ { k } ^ { \pi } + H \tau \log A \right] \leq V _ { 1 } ^ { \mathrm { p l } } ( s _ { 1 } ) \leq \operatorname* { m i n } _ { u \in { \cal K } } \bar { c } ^ { \top } u .\tag{12}
$$

Thus both inequalities in (9) hold if we choose $\Delta \ge 5 \sqrt { p } \beta U _ { k } ^ { \pi } + H \tau$ log A. However, the learner cannot compute this threshold because $U _ { k } ^ { \pi }$ is an expectation under the unknown transitions. We next obtain an estimated upper bound of $U _ { k } ^ { \pi }$ , from which we can choose $\Delta$

Task 3: Uncertainty evaluation. We use the same backward policy evaluation to estimate $U _ { k } ^ { \pi }$ . This can be achieved by viewing $\nu _ { k , h } ( s , a ) \in [ 0 , 1 ]$ as the reward. Since $\| \widehat { U } _ { h + 1 } \| _ { \infty } \leq H$ , we add the bonus $( 1 + \beta H ) \nu _ { k , h }$ to guard against underestimating future uncertainty. A backward induction gives $U _ { k } ^ { \pi } \leq \widehat { U } _ { 1 } ( s _ { 1 } ) \leq 2 \beta H U _ { k } ^ { \pi }$ . Using the computable upper bound $\widehat { U } _ { 1 } ( s _ { 1 } )$ in place of $U _ { k } ^ { \pi }$ in the feature evaluation step, APPROXLO $( c ; ( \mathcal { D } _ { k , h } ) _ { h = 1 } ^ { H } )$ (Algorithm 3) returns $( \pi , { \widehat { \phi } } , \Delta )$ with $\Delta = \operatorname* { m i n } \{ 5 \sqrt { p } \beta \widehat { U } _ { 1 } ( s _ { 1 } ) + H \tau \log A , 2 \sqrt { H } \}$ , where the clipping to $2 \sqrt { H }$ is because $\| \phi - \phi ^ { \prime } \| _ { 2 } \leq 2 \sqrt { H }$ for any policy features $\phi , \phi ^ { \prime }$

The above definition of $\Delta$ together with $\widehat { U } _ { 1 } ( s _ { 1 } ) \leq 2 \beta H U _ { k } ^ { \pi }$ gives (10).

## 5.3 Final Regret Bound

Proof of Theorem 2.1. Now we combine elements to establish the final regret bound. The sum of misspecification in the “bias + correction” term in Lemma 4.1 can be bounded as

$$
\begin{array} { r l r } {  { \mathbb E [ \sum _ { k = 1 } ^ { K } \frac { 1 } { 2 \mu _ { \widehat { \xi } } } \sum _ { i = 1 } ^ { 2 \rho } \frac { z ^ { 2 } } { \lambda } z _ { k , i } ^ { 2 } ] } } \\ & { = } & { \mathbb E [ \sum _ { k = 1 } ^ { K } ( 4 \zeta ^ { 2 } + \frac { 8 R ^ { 2 } ( 1 + \log \mathcal { M } ) } { \mathcal M } + \frac { 6 } { \mathcal M } \frac { 1 } { 2 p } \sum _ { \mathfrak { r } = 1 } ^ { 2 \rho } \sum _ { \mathfrak { r } = 1 } ^ { \mathcal M } \Delta _ { \mathfrak { r } , i \widehat { \xi } } ^ { 2 } ) ] \qquad \mathrm { ( L i n c ~ 9 ~ o f ~ A | \operatorname { g o r i t h m } 2 ) } } \\ & { \le \tilde { O } ( H + \mathbb E [ \frac { K } { \mathrm { L i - 1 } } \frac { \Delta _ { \mathfrak { r } , I _ { k - 1 } , I _ { k } } ^ { 2 } } { \Delta _ { \mathfrak { r } , I _ { k } , I _ { k } } } ] ) \qquad \mathrm { ( ( \it ( I _ { k } , J _ { k } ) ~ u n i f o r m y ~ d r a v e n ~ f r o m ~ | 2 \eta | \times | \mathcal { M } | ~ i n ~ A | \operatorname { g o r i t h m } ~ | 3 ) } } \\ & { \le \tilde { O } ( H + \mathbb E [ \sum _ { k = 1 } ^ { K } p \beta ^ { 4 } H ^ { 3 } \sum _ { \mathfrak { r } = 1 } ^ { \mathcal M } | \phi _ { \delta } ( S _ { k , k } , A _ { \mathfrak { r } , k } ) | ] _ { k , k } ^ { 2 } ) ) } \\ & { \le \tilde { O } ( \theta ^ { d } H ^ { 5 } ) , } &  \mathrm { ( E l l p i c a l ~ p a t e n i t ~ l e r m m z ~ \cdot ~ } \mathbb E [ \sum _ { k , k }  \phi _ { k } ( S _ { k , k } , A _ { k } )  _ { \mathcal { X } _ { k , k } ^ { 2 } } ^ { 2 } ] \le \tilde { O } ( d H  \end{array}
$$

Plugging this in Lemma 4.1 gives the desired bound.

## 6 Computation

We bound the computation time of Algorithm 1 by analyzing the costs of MOMENTESTIMATE and APPROXSEP, together with the number of cuts.

The MOMENTESTIMATE procedure and the desired guarantees are same as those in [Ito et al., 2019]. We directly use the results in their Section 4.2 to bound the complexity of MOMENTESTIMATE by $\mathrm { p o l y } ( d , H , K , \log ( 1 / \delta ) )$ calls to the membership oracle of O.

The total number of cuts added in each episode is upper bounded by $\widetilde { O } ( p + \sqrt { H } K ^ { 3 / 2 } )$ with high probability (see Lemma E.3). To prevent the low probability event to lead to non-termination of the algorithm, we present a version of Algorithm 1 that explicitly counts the number of restarts and executes the fallback plan if the count exceeds a certain threshold (Algorithm 4).

The APPROXLO procedure (Algorithm 3) can be run in poly $( d , H , K , A )$ time, even though the displayed Algorithm 3 enumerates the updates on all state-action pairs. This is common in prior work for linear MDP (Jin et al. [2020]). For completeness, we explicitly describe the implementation in Section E.3

## References

Jacob D Abernethy, Elad Hazan, and Alexander Rakhlin. Competing in the dark: An efficient algorithm for bandit linear optimization. In Conference on Learning Theory, number 110, 2009.

Sébastien Bubeck and Ronen Eldan. The entropic barrier: a simple and optimal universal self-concordant barrier. Conference on Learning Theory, 2015.

Qi Cai, Zhuoran Yang, Chi Jin, and Zhaoran Wang. Provably efficient exploration in policy optimization. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 1283–1294, 2020. URL https://proceedings.mlr.press/ v119/cai20d.html.

A. Cassel and A. Rosenberg. Warm-up free policy optimization: Improved regret in linear markov decision processes. In Advances in Neural Information Processing Systems, volume 37, 2024.

Asaf Cassel, Haipeng Luo, Aviv Rosenberg, and Dmitry Sotnikov. Near-optimal regret in linear mdps with aggregate bandit feedback. International Conference on Machine Learning, 2024.

Fan Chen, Zeyu Jia, Alexander Rakhlin, and Tengyang Xie. Outcome-based online reinforcement learning: Algorithms and fundamental limits. Advances in Neural Information Processing Systems, 38, 2025.

S. Chewi. The entropic barrier is n-self-concordant. In Geometric Aspects of Functional Analysis, pages 209–222. Springer, 2023.

A. Cohen, H. Kaplan, T. Koren, and Y. Mansour. Online Markov decision processes with aggregate bandit feedback. In Proceedings of the 34th Conference on Learning Theory, pages 1301–1329, 2021.

Yan Dai, Haipeng Luo, Chen-Yu Wei, and Julian Zimmert. Refined regret for adversarial mdps with linear function approximation. In International Conference on Machine Learning, pages 6726–6759. PMLR, 2023.

Simon Du, Sham Kakade, Jason Lee, Shachar Lovett, Gaurav Mahajan, Wen Sun, and Ruosong Wang. Bilinear classes: A structural framework for provable generalization in rl. In International Conference on Machine Learning, pages 2826–2836. PMLR, 2021.

Dylan J. Foster, Alexander Rakhlin, Ayush Sekhari, and Karthik Sridharan. On the complexity of adversarial decision making. In Advances in Neural Information Processing Systems, volume 35, pages 35404–35417, 2022. URL https://arxiv.org/abs/2206.13063.

Jiafan He, Dongruo Zhou, and Quanquan Gu. Near-optimal policy optimization algorithms for learning adversarial linear mixture MDPs. In Proceedings ofthe 25th International Conference on Artificial Intelligence and Statistics, volume 151 of Proceedings ofMachine Learning Research, pages 4259–4280, 2022. URL https://proceedings.mlr.press/v151/he22a.html.

S. Ito, D. Hatano, H. Sumita, K. Takemura, T. Fukunaga, N. Kakimura, and K.-I. Kawarabayashi. Oracle-efficient algorithms for online linear optimization with bandit feedback. In Advances in Neural Information Processing Systems, volume 32, 2019.

S. Ito, K. G. Jamieson, H. Luo, A. Maiti, and T. Tsuchiya. Adapting to stochastic and adversarial losses in episodic MDPs with aggregate bandit feedback. In Advances in Neural Information Processing Systems, 2025.

C. Jin, Z. Yang, Z. Wang, and M. I. Jordan. Provably efficient reinforcement learning with linear function approximation. In Proceedings ofthe 33rd Conference on Learning Theory, volume 125 of Proceedings of Machine Learning Research, pages 2137–2143, 2020. Full version: arXiv:1907.05388v2.

Fang Kong, Xiangcheng Zhang, Baoxiang Wang, and Shuai Li. Improved regret bounds for linear adversarial mdps via linear optimization. Transactions on Machine Learning Research, 2024.

T. Lancewicki and Y. Mansour. Near-optimal regret using policy optimization in online MDPs with aggregate bandit feedback. In Proceedings ofthe 42nd International Conference on Machine Learning, 2025.

C.-W. Lee, H. Luo, C.-Y. Wei, and M. Zhang. Bias no more: High-probability data-dependent regret bounds for adversarial bandits and MDPs. In Advances in Neural Information Processing Systems, volume 33, 2020.

Long-Fei Li, Peng Zhao, and Zhi-Hua Zhou. Improved algorithm for adversarial linear mixture MDPs with bandit feedback and unknown transition. In Proceedings of the 27th International Conference on Artificial Intelligence and Statistics, volume 238 of Proceedings ofMachine Learning Research, pages 3061–3069, 2024. URL https://proceedings.mlr.press/v238/li24n.html.

H. Liu, C.-Y. Wei, and J. Zimmert. Towards optimal regret in adversarial linear MDPs with bandit feedback. In International Conference on Learning Representations, 2024a.

Haolin Liu, Chen-Yu Wei, and Julian Zimmert. Bypassing the simulator: Near-optimal adversarial linear contextual bandits. In Advances in Neural Information Processing Systems, volume 36, 2023. URL https: //arxiv.org/abs/2309.00814.

Haolin Liu, Zakaria Mhammedi, Chen-Yu Wei, and Julian Zimmert. Beating adversarial low-rank MDPs with unknown transition and bandit feedback. In Advances in Neural Information Processing Systems, volume 37, 2024b. URL https://arxiv.org/abs/2411.06739.

Haolin Liu, Artin Tajdini, Andrew Wagenmaker, and Chen-Yu Wei. Corruption-robust linear bandits: Minimax optimality and gap-dependent misspecification. Advances in Neural Information Processing Systems, 37: 24277–24325, 2024c.

Haolin Liu, Chen-Yu Wei, and Julian Zimmert. Decision making in hybrid environments: A model aggregation approach. In Proceedings of the 38th Conference on Learning Theory, volume 291 of Proceedings of Machine Learning Research, pages 3732–3765, 2025. URL https://proceedings.mlr.press/ v291/liu25a.html.

Haolin Liu, Chen-Yu Wei, and Julian Zimmert. An improved model-free decision-estimation coefficient with applications in adversarial MDPs. In International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=lbLAgGF8OO.

L. Lovász and S. Vempala. The geometry of logconcave functions and sampling algorithms. Random Structures & Algorithms, 30(3):307–358, 2007.

Haipeng Luo, Chen-Yu Wei, and Chung-Wei Lee. Policy optimization in adversarial mdps: Improved exploration via dilated bonuses. Advances in Neural Information Processing Systems, 34:22931–22942, 2021.

A. S. Nemirovski and M. J. Todd. Interior-point methods for optimization. Acta Numerica, pages 191–234, 2008. doi: 10.1017/S0962492906370018.

Gergely Neu and Julia Olkhovskaya. Efficient and robust algorithms for adversarial linear contextual bandits. In Proceedings ofthe 33rd Conference on Learning Theory, volume 125 of Proceedings ofMachine Learning Research, pages 3049–3068, 2020. URL https://proceedings.mlr.press/v125/neu20b.html.

Gergely Neu and Julia Olkhovskaya. Online learning in MDPs with linear function approximation and bandit feedback. In Advances in Neural Information Processing Systems, volume 34, pages 10407–10417, 2021. URL https://papers.neurips.cc/paper\_files/paper/2021/hash/ 5631e6ee59a4175cd06c305840562ff3-Abstract.html.

Uri Sherman, Tomer Koren, and Yishay Mansour. Improved regret for efficient online reinforcement learning with linear function approximation. In International Conference on Machine Learning, pages 31117–31150. PMLR, 2023.

Uri Sherman, Alon Cohen, Tomer Koren, and Yishay Mansour. Rate-optimal policy optimization for linear Markov decision processes. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 44815–44837, 2024. URL https: //proceedings.mlr.press/v235/sherman24a.html.

Tim van Erven, Jack Mayo, Julia Olkhovskaya, and Chen-Yu Wei. An improved algorithm for adversarial linear contextual bandits via reduction. In Advances in Neural Information Processing Systems, volume 38, 2025.

Tim van Erven, Jack Mayo, Julia Olkhovskaya, and Chen-Yu Wei. An improved algorithm for adversarial linear contextual bandits via reduction. arXiv preprint arXiv:2508.11931v3, 2026. Revised version, June 1, 2026.

Ruosong Wang, Simon S Du, Lin Yang, and Russ R Salakhutdinov. On reward-free reinforcement learning with linear function approximation. Advances in neural information processing systems, 33:17816–17826, 2020.

Chen-Yu Wei, Mehdi Jafarnia Jahromi, Haipeng Luo, and Rahul Jain. Learning infinite-horizon average-reward mdps with linear function approximation. In International Conference on Artificial Intelligence and Statistics, pages 3007–3015. PMLR, 2021.

Tengyang Xie, Dylan J Foster, Yu Bai, Nan Jiang, and Sham M Kakade. The role of coverage in online reinforcement learning. arXiv preprint arXiv:2210.04157, 2022.

Yunbei Xu and Assaf Zeevi. Bayesian design principles for frequentist sequential learning. Journal of the ACM, 72(5):1–65, 2025.

Canzhe Zhao, Ruofeng Yang, Baoxiang Wang, and Shuai Li. Learning adversarial linear mixture Markov decision processes with bandit feedback and unknown transition. In International Conference on Learning Representations, 2023a. URL https://openreview.net/forum?id=sVU54nyaA9K.

Canzhe Zhao, Ruofeng Yang, Baoxiang Wang, Xuezhou Zhang, and Shuai Li. Learning adversarial low-rank Markov decision processes with unknown transition and full-information feedback. In Advances in Neural Information Processing Systems, volume 36, 2023b. URL https://papers.nips.cc/paper\_files/ paper/2023/hash/b93fda2862db7a7ac4a5c412adfb1ac2-Abstract-Conference. html.

H. Zhong and T. Zhang. A theoretical analysis of optimistic proximal policy optimization in linear Markov decision processes. In Advances in Neural Information Processing Systems, volume 36, 2023.

J. Zimmert and T. Lattimore. Return of the bias: Almost minimax optimal high probability bounds for adversarial linear bandits. In Proceedings ofthe 35th Conference on Learning Theory, 2022.

## Appendices

## Appendix contents

A Related Work 16   
A.1 Learning Adversarial MDPs with Linear Structure . 16   
A.2 Adversarial RL beyond Linear MDPs 16   
B Complete Algorithm 18   
C Omitted Proofs in Section 4 19   
C.1 Properties of continuous exponential weights 19   
C.2 Bounding bias and bonus 19   
C.3 Overall regret analysis 22   
D Omitted Proofs in Section 5 25   
E Omitted Proofs in Section 6 29   
E.1 Moment Estimates 29   
E.2 Number of cuts 29   
E.3 Implementation of Algorithm 3 . 30

## A Related Work

## A.1 Learning Adversarial MDPs with Linear Structure

We review results for learning adversarial rewards with fixed transition dynamics.

A closely related problem is adversarial linear contextual bandits with stochastic contexts, where contexts are drawn independently from a fixed distribution, while reward vectors may change adversarially. Neu and Olkhovskaya [2020] obtain efficient square-root regret when the context distribution is known. With access to a context simulator, Dai et al. [2023] also achieve square-root regret without knowing this distribution. Liu et al. [2023] remove the simulator requirement and obtain $\sqrt { K }$ regret with computation polynomial in the number of available actions. van Erven et al. [2026] obtains a reduction to linear bandits whose computation avoids enumerating the actions, and only requires linear optimization oracles for the action sets.

Linear mixture MDPs parameterize the transition kernel using known features of state-action-next-state triples, rather than the state-action features in linear MDPs. With unknown transitions and full-information reward feedback, Cai et al. [2020] establish $\sqrt { K }$ regret via optimistic policy optimization. He et al. [2022] improve the horizon dependence and obtain near-minimax guarantees. Under per-step bandit feedback, Zhao et al. [2023a] give the first efficient algorithm with $\sqrt { K }$ regret, and Li et al. [2024] subsequently improve its dependence on the number of states. Both bandit-feedback bounds retain polynomial dependence on state cardinality.

For adversarial linear MDPs, Neu and Olkhovskaya [2021] establish square-root regret with known transitions under an exploratory condition. With unknown transitions and full-information feedback, Zhong and Zhang [2023] obtain $K ^ { 3 / 4 }$ and Sherman et al. [2024] improved the guarantee to $\sqrt { K }$ . Cassel and Rosenberg [2024] retain the square-root rate, improve the dimension and horizon dependencies, and remove the initial reward-free exploration phase in Sherman et al. [2024]. For bandit-feedback under a simulator access, Luo et al. [2021] obtain $K ^ { 2 / 3 }$ regret, and Dai et al. [2023] improve this to $\sqrt { K }$ . Without a simulator, learning unknown transitions from the learner’s own trajectories is more difficult. Under bandit feedback, the efficient regret bounds progress from $K ^ { 1 4 / 1 5 }$ [Luo et al., 2021] to $K ^ { 8 / 9 }$ [Dai et al., 2023], $K ^ { 6 / 7 }$ [Sherman et al., 2023], and $K ^ { 3 / 4 }$ [Liu et al., 2024a]. Kong et al. [2024] obtain $K ^ { 4 / 5 }$ regret through a reduction to linear optimization under an exploratory assumption. Liu et al. [2024a] additionally achieve the optimal $\sqrt { K }$ rate without that assumption, but their algorithm is computationally inefficient. Our work obtains poly $( d , H ) \sqrt { K }$ regret with polynomial computation, and requires only the cummulative loss rather than the per-step losses along the trajectory.

## A.2 Adversarial RL beyond Linear MDPs

Low-rank MDPs extend linear MDPs by allowing the feature representation to be unknown. With unknown transitions and full-information feedback, Zhao et al. [2023b] and Liu et al. [2024b] give model-based, oracleefficient algorithms with $K ^ { 5 / 6 }$ and $K ^ { 2 / 3 }$ regret, respectively, allowing arbitrary rewards. Under bandit feedback, Liu et al. [2024b] show that additional reward structure is necessary to avoid polynomial dependence on the number of states. Assuming linear rewards, they obtain $K ^ { 2 / 3 }$ regret through computationally inefficient model-based and model-free algorithms, and $K ^ { 4 / 5 }$ regret through oracle-efficient model-free algorithms.

The decision-estimation coefficient (DEC) approach provides a general approach to adversarial RL. Foster et al. [2022], Xu and Zeevi [2025] characterize adversarial decision making using the DEC of the convexified model class, but they can only handle MDPs with known transitions. Liu et al. [2025] extend the DEC framework to MDPs with fixed unknown transitions and adversarial rewards, providing general model-based regret guarantees. As an application, their framework yields $\sqrt { K }$ regret for low-rank MDPs with adversarial linear rewards under both full-information and bandit feedback. For model-free learning with full-information reward feedback, they establish $K ^ { 3 / 4 }$ and $K ^ { 5 / 6 }$ regret for on-policy and off-policy bilinear classes [Du et al., 2021], respectively. Liu et al. [2026] extends model-free learning to adversarial bandit feedback under linear rewards. For on-policy and off-policy bilinear classes, they obtain $K ^ { 5 / 6 }$ and $K ^ { 1 3 / 1 5 }$ regret, respectively. With Bellman completeness, these rates improve to $K ^ { 3 / 4 }$ and $K ^ { 4 / 5 }$ ; they also obtain $K ^ { 3 / 4 }$ regret for coverable MDPs [Xie et al., 2022] under Bellman completeness. These results cover transition structures beyond linear MDPs, but do not provide the polynomial-time square-root guarantee studied here.

## B Complete Algorithm

Algorithm 1 involves a repeat-until loop that may not stop, though this happens only with negligible probability (Lemma E.3). In order to ensure the algorithm will terminate with probability 1, we present below in Algorithm 4 a more complete version that caps the number of repeats by a finite amount. As shown in Lemma E.3, with high probability, the number of repeats is smaller than the threshold. In the regret analysis (Lemma 4.1), the bad event only contributes to negligible expected regret. In Algorithm 4, we highlight the parts that are different from Algorithm 1.

```latex
Algorithm 4 Algorithm 1 with a restart cap and a fallback policy
Require: $\begin{array} { r } { c _ { 0 } = \frac { 1 } { 1 6 e } , \zeta = K ^ { - 1 / 2 } , R = \sqrt { H } + \zeta , M = K , N _ { \operatorname* { m a x } } = \widetilde \Theta ( p + \sqrt { H K ^ { 3 } } ) , \eta \leq \frac { c _ { 0 } } { 1 2 8 \sqrt { d ^ { 3 } H ^ { 4 } } } . } \end{array}$
1: $Z _ { 0 } \gets 0 ; \mathcal { O } \gets R \mathbb { B } _ { 2 } ^ { p } ; \mathcal { D } _ { 1 , h } \gets \emptyset$ for all $h \in [ H ]$
2: for $k = 1 , \ldots , K$ do
3: $n \gets 0$
4: repeat
5: n $ n + 1$
6: restart← FALSE
7: Define $q _ { k } ( x ) \propto \exp ( - \eta Z _ { k - 1 } ^ { \top } x ) \mathbf { 1 } \{ x \in \mathcal { O } \}$
8: (µb , Σb ) ← MOMENTESTIMATE $( \mathcal { O } , Z _ { k - 1 } )$
9: Let $\{ Y _ { k , i } \} _ { i = 1 } ^ { 2 p }$ be the set $\left\{ \widehat { \mu } _ { k } + c _ { 0 } \widehat { \Sigma } _ { k } ^ { 1 / 2 } v : v \in \{ \pm e _ { 1 } , \pm e _ { 2 } , \ldots , \pm e _ { p } \} \right\}$
10: for $i = 1 , \ldots , 2 p$ do
11: reply $ \mathrm { A P P R O X S E P } ( Y _ { k , i } , ( \mathcal { D } _ { k , h } ) _ { h = 1 } ^ { H } , M , \zeta )$
12: if reply is a cut $( c , b )$ then
13: $\mathcal { O }  \mathcal { O } \cap \{ x : c ^ { \top } x \geq b \}$
14: restart← TRUE; break
15: else
16: $( ( \pi _ { k , i , j } ) _ { j = 1 } ^ { M } , \varepsilon _ { k , i } ) \gets \mathsf { r e p l y }$
17: until restart= FALSE or $n = N _ { \operatorname* { m a x } }$
18: if $n = N _ { \mathrm { m a x } }$ then
19: Execute the policy $\pi _ { k }$ with $\pi _ { k , h } ( \cdot \mid s ) = \operatorname { U n i f } ( A )$ for all $h , s .$
20: $Z _ { k } \gets Z _ { k - 1 }$ and $\mathcal { D } _ { k + 1 , h } \gets \mathcal { D } _ { k , h }$ for all $h \in [ H ]$
21: else
22: Draw $( I _ { k } , J _ { k } ) \sim \mathrm { U n i f } ( [ 2 p ] \times [ M ] )$ and execute $\pi _ { k } = \pi _ { k , I _ { k } , J _ { k } }$
23: Observe feature vectors along the trajectory $( \phi _ { k , h } ) _ { h = 1 } ^ { H }$ and the total loss $L _ { k } .$
24: $\begin{array} { r } { \widehat { \theta } _ { k } \gets \left( \zeta ^ { 2 } \mathbf { I } _ { p } + \frac { c _ { 0 } ^ { 2 } } { p } \widehat { \boldsymbol { \Sigma } } _ { k } \right) ^ { - 1 } ( Y _ { k , I _ { k } } - \widehat { \mu } _ { k } ) L _ { k } } \end{array}$
25: $Z _ { k } \gets Z _ { k - 1 } + \widehat { \theta } _ { k } - B _ { k }$ , where $\begin{array} { r } { B _ { k } = \frac { 1 0 \eta p \varepsilon _ { k } } { c _ { 0 } } Z _ { k - 1 } } \end{array}$ and $\begin{array} { r } { \varepsilon _ { k } = \sqrt { \frac { 1 } { 2 p } \sum _ { i = 1 } ^ { 2 p } \varepsilon _ { k , i } ^ { 2 } } } \end{array}$
26: $\mathcal { D } _ { k + 1 , h } \gets \mathcal { D } _ { k , h } \cup \{ ( \phi _ { k , h } , S _ { k , h + 1 } ) \}$ for all $h \in [ H ]$
```

## C Omitted Proofs in Section 4

## C.1 Properties of continuous exponential weights

For a full-dimensional compact convex body $\mathcal { O } \subset \mathbb { R } ^ { p }$ , define

$$
\Phi _ { \mathcal { O } } ( v ) = \log \int _ { \mathcal { O } } e ^ { v ^ { \top } x } d x , \qquad F _ { \mathcal { O } } = \Phi _ { \mathcal { O } } ^ { * } .\tag{13}
$$

Here $F _ { \mathcal { O } } ( x ) = \operatorname* { s u p } _ { v } \{ v ^ { \top } x - \Phi _ { \mathcal { O } } ( v ) \}$ is the entropic barrier.

Lemma C.1 (Entropic-barrier toolkit). Let $X _ { v }$ have density proportional to $e ^ { v ^ { \top } x } \mathbf { 1 } \{ x \in \mathcal { O } \}$ . Then

$$
\nabla \Phi \mathcal { O } \left( v \right) = \mathbb { E } \left[ X _ { v } \right] , \qquad \nabla ^ { 2 } \Phi \mathcal { O } \left( v \right) = \operatorname { C o v } ( X _ { v } ) .\tag{14}
$$

At every conjugate pair $x = \nabla \Phi _ { \mathcal { O } } ( v )$ (v),

$$
\nabla F _ { \cal O } ( x ) = v , \qquad \nabla ^ { 2 } F _ { \cal O } ( x ) = \bigl ( \nabla ^ { 2 } \Phi _ { \cal O } ( v ) \bigr ) ^ { - 1 } .\tag{15}
$$

Moreover, for every x, u ∈ int(O),

$$
\| x - u \| _ { \nabla ^ { 2 } F \mathcal { O } ( x ) } \leq 4 \left. x - u , \nabla F \mathcal { O } ( x ) \right. + 2 4 p ,\tag{16}
$$

$$
\| \nabla F _ { \mathcal { O } } ( x ) \| _ { ( \nabla ^ { 2 } F _ { \mathcal { O } } ( x ) ) ^ { - 1 } } \leq \sqrt { p } .\tag{17}
$$

Finally, $i f r = \lVert \delta v \rVert _ { \nabla ^ { 2 } \Phi \mathcal { O } ( v ) } < 1$ , then

$$
\Phi _ { \mathcal { O } } ( v + \delta v ) - \Phi _ { \mathcal { O } } ( v ) - \langle \nabla \Phi _ { \mathcal { O } } ( v ) , \delta v \rangle \leq - r - \log ( 1 - r ) .\tag{18}
$$

Proof. Equation (14) follows by differentiation. Inverse-gradient duality gives (15) [Chewi, 2023, Section 2]. By Chewi [2023, Theorem 2], $F _ { \mathcal { O } }$ is a p-self-concordant barrier, giving (17). By Zimmert and Lattimore [2022, Lemma 14], some $\gamma ^ { \prime } \in [ 1 , 4 ]$ satisfies

$$
\| x - u \| _ { \nabla ^ { 2 } F _ { \mathcal { O } } ( x ) } \le \gamma ^ { \prime } \left. x - u , \nabla F _ { \mathcal { O } } ( x ) \right. + 4 \gamma ^ { \prime } p + 2 \sqrt { p } .
$$

For $p , \gamma ^ { \prime } \geq 1 , 4 \gamma ^ { \prime } p + 2 \sqrt { p } \leq 6 \gamma ^ { \prime } p$ . Thus

$$
\begin{array} { r } { 0 \leq \| x - u \| _ { \nabla ^ { 2 } F _ { \mathcal { O } } ( x ) } \leq \gamma ^ { \prime } \big ( \langle x - u , \nabla F _ { \mathcal { O } } ( x ) \rangle + 6 p \big ) \leq 4 \big ( \langle x - u , \nabla F _ { \mathcal { O } } ( x ) \rangle + 6 p \big ) , } \end{array}
$$

proving (16).

Since $\Phi _ { \mathcal { O } } = F _ { \mathcal { O } } ^ { * }$ and self-concordance is preserved under convex conjugation [Nemirovski and Todd, 2008, Section 2.1.3], $\Phi _ { \mathcal { O } }$ is self-concordant. Its local Taylor bound [Nemirovski and Todd, 2008, Section 2.1.1, Eq. (2.5)] gives (18). □

## C.2 Bounding bias and bonus

Lemma C.2 (Loss-estimator bias and local norm). Assume $\begin{array} { r } { \frac 1 2 \Sigma _ { k } \preceq \widehat { \Sigma } _ { k } \preceq 2 \Sigma _ { k } } \end{array}$ , and write $ { \widetilde { \theta } } _ { k } =  { \widehat { \theta } } _ { k } - B _ { k }$ . Then the loss estimator satisfies

$$
\| \theta _ { k } - \mathbb { E } _ { k } [ \widehat { \theta } _ { k } ] \| _ { \Sigma _ { k } } \leq \frac { 5 p \varepsilon _ { k } } { 2 c _ { 0 } } , \qquad \| \widehat { \theta } _ { k } \| _ { \Sigma _ { k } } ^ { 2 } \leq \frac { 2 H ^ { 2 } p ^ { 2 } } { c _ { 0 } ^ { 2 } } , \qquad \| \widetilde { \theta } _ { k } \| _ { \Sigma _ { k } } \leq \frac { \sqrt { 2 } H p } { c _ { 0 } } + \frac { 1 0 p ^ { 3 / 2 } \varepsilon _ { k } } { c _ { 0 } } .\tag{19}
$$

Proof of Lemma C.2. Define $\alpha = c _ { 0 } ^ { 2 } / p$ . From the estimator definition,

$$
\theta _ { k } - \widehat { \theta } _ { k } = ( \zeta ^ { 2 } \mathbf { I } _ { p } + \alpha \widehat { \Sigma } _ { k } ) ^ { - 1 } \Big ( ( \zeta ^ { 2 } \mathbf { I } _ { p } + \alpha \widehat { \Sigma } _ { k } ) \theta _ { k } - ( Y _ { k , I _ { k } } - \widehat { \mu } _ { k } ) L _ { k } \Big ) .
$$

Our construction gives

$$
\frac { 1 } { 2 p } \sum _ { i = 1 } ^ { 2 p } ( Y _ { k , i } - \widehat { \mu } _ { k } ) = 0 , \qquad \frac { 1 } { 2 p } \sum _ { i = 1 } ^ { 2 p } ( Y _ { k , i } - \widehat { \mu } _ { k } ) ( Y _ { k , i } - \widehat { \mu } _ { k } ) ^ { \top } = \alpha \widehat { \Sigma } _ { k } .
$$

Define

$$
m _ { k } : = \frac { 1 } { 2 p } \sum _ { i = 1 } ^ { 2 p } ( Y _ { k , i } - \widehat { \mu } _ { k } ) ( x _ { k , i } - Y _ { k , i } ) ^ { \top } \theta _ { k } .
$$

Using $\mathbb { E } _ { k } [ L _ { k } \mid I _ { k } = i ] = \langle x _ { k , i } , \theta _ { k } \rangle$ , we obtain

$$
\begin{array} { r l } & { \mathbb { E } _ { k } [ ( Y _ { k , I _ { k } } - \widehat { \mu } _ { k } ) L _ { k } ] = \displaystyle \frac { 1 } { 2 p } \sum _ { i = 1 } ^ { 2 p } ( Y _ { k , i } - \widehat { \mu } _ { k } ) x _ { k , i } ^ { \top } \theta _ { k } } \\ & { \qquad = \alpha \widehat { \Sigma } _ { k } \theta _ { k } + m _ { k } . } \end{array}
$$

Taking $\mathbb { E } _ { k }$ in the first display therefore gives

$$
\theta _ { k } - \mathbb { E } _ { k } [ \widehat \theta _ { k } ] = ( \zeta ^ { 2 } \mathbf { I } _ { p } + \alpha \widehat { \Sigma } _ { k } ) ^ { - 1 } ( \zeta ^ { 2 } \theta _ { k } - m _ { k } ) .\tag{20}
$$

For any $v \in \mathbb { R } ^ { p } .$ , Cauchy–Schwarz for the uniform average over $i \in [ 2 p ]$ gives

$$
\begin{array} { r l r } {  {  v , m _ { k }  \big | = | \frac { 1 } { 2 p } \displaystyle \sum _ { i = 1 } ^ { 2 p } v ^ { \top } ( Y _ { k , i } - \widehat { \mu } _ { k } ) ( x _ { k , i } - Y _ { k , i } ) ^ { \top } \theta _ { k } | } } \\ & { } & { \leq \sqrt { \frac { 1 } { 2 p } \displaystyle \sum _ { i = 1 } ^ { 2 p } v ^ { \top } ( Y _ { k , i } - \widehat { \mu } _ { k } ) ( Y _ { k , i } - \widehat { \mu } _ { k } ) ^ { \top } } v \sqrt { \frac { 1 } { 2 p } \displaystyle \sum _ { i = 1 } ^ { 2 p } \theta _ { k } ^ { \top } ( x _ { k , i } - Y _ { k , i } ) ( x _ { k , i } - Y _ { k , i } ) ^ { \top } \theta _ { k } } } \\ & { } & { \leq \sqrt { \alpha } \| v \| _ { \widehat { \Sigma } _ { k } } \sqrt { \displaystyle \frac { p } { 2 p } \displaystyle \sum _ { i = 1 } ^ { 2 p } \| x _ { k , i } - Y _ { k , i } \| _ { 2 } ^ { 2 } } \mathrm { ~ ~ } \qquad \qquad ( \| \theta _ { k } \| _ { 2 } ^ { 2 } \leq d H = p ) } \\ & { } & { = c _ { 0 } \| v \| _ { \widehat { \Sigma } _ { k } } \varepsilon _ { k } . } \end{array}
$$

Taking $v = \widehat { \Sigma } _ { k } ^ { - 1 } m _ { k }$ , we get $\| m _ { k } \| _ { \widehat { \Sigma } _ { k } ^ { - 1 } } ^ { 2 } \leq c _ { 0 } \| m _ { k } \| _ { \widehat { \Sigma } _ { k } ^ { - 1 } } \varepsilon _ { k }$ and hence

$$
\| m _ { k } \| _ { \widehat { \Sigma } _ { k } ^ { - 1 } } \leq c _ { 0 } \varepsilon _ { k } .
$$

Since $\Sigma _ { k } \preceq 2 \widehat { \Sigma } _ { k }$ , we have

$$
\begin{array} { r l r } & { } & { \left\| ( \zeta ^ { 2 } \mathbf { I } _ { p } + \alpha \widehat { \boldsymbol { \Sigma } } _ { k } ) ^ { - 1 } \boldsymbol { m } _ { k } \right\| _ { \Sigma _ { k } } ^ { 2 } \leq 2 \left\| ( \zeta ^ { 2 } \mathbf { I } _ { p } + \alpha \widehat { \boldsymbol { \Sigma } } _ { k } ) ^ { - 1 } \boldsymbol { m } _ { k } \right\| _ { \widehat { \boldsymbol { \Sigma } } _ { k } } ^ { 2 } } \\ & { } & { \leq \displaystyle \frac { 2 } { \alpha ^ { 2 } } \| \boldsymbol { m } _ { k } \| _ { \widehat { \boldsymbol { \Sigma } } _ { k } ^ { - 1 } } ^ { 2 } \leq \displaystyle \frac { 2 c _ { 0 } ^ { 2 } \varepsilon _ { k } ^ { 2 } } { \alpha ^ { 2 } } = \frac { 2 p ^ { 2 } \varepsilon _ { k } ^ { 2 } } { c _ { 0 } ^ { 2 } } . } \end{array}
$$

We also have

$$
\begin{array} { r l } & { \left\| \zeta ^ { 2 } ( \zeta ^ { 2 } \mathbf { I } _ { p } + \alpha \widehat { \Sigma } _ { k } ) ^ { - 1 } \theta _ { k } \right\| _ { \Sigma _ { k } } ^ { 2 } \leq 2 \zeta ^ { 4 } \theta _ { k } ^ { \top } ( \zeta ^ { 2 } \mathbf { I } _ { p } + \alpha \widehat { \Sigma } _ { k } ) ^ { - 1 } \widehat { \Sigma } _ { k } ( \zeta ^ { 2 } \mathbf { I } _ { p } + \alpha \widehat { \Sigma } _ { k } ) ^ { - 1 } \theta _ { k } } \\ & { \qquad \leq \displaystyle \frac { 2 } { \alpha } \zeta ^ { 4 } \theta _ { k } ^ { \top } ( \zeta ^ { 2 } \mathbf { I } _ { p } + \alpha \widehat { \Sigma } _ { k } ) ^ { - 1 / 2 } ( \zeta ^ { 2 } \mathbf { I } _ { p } + \alpha \widehat { \Sigma } _ { k } ) ^ { - 1 / 2 } \theta _ { k } } \\ & { \qquad \leq \displaystyle \frac { 2 \zeta ^ { 2 } } { \alpha } \| \theta _ { k } \| _ { 2 } ^ { 2 } = \displaystyle \frac { 2 \zeta ^ { 2 } p ^ { 2 } } { c _ { 0 } ^ { 2 } } . } \end{array}
$$

Combining these two bounds in (20) and using $\varepsilon _ { k } \geq 2 \zeta$ yields

$$
\| \theta _ { k } - \mathbb { E } _ { k } [ \widehat \theta _ { k } ] \| _ { \Sigma _ { k } } \leq \frac { \sqrt { 2 } p \varepsilon _ { k } } { c _ { 0 } } + \frac { \sqrt { 2 } p \zeta } { c _ { 0 } } \leq \frac { 5 p \varepsilon _ { k } } { 2 c _ { 0 } } .
$$

Moreover, $0 \leq L _ { k } \leq H$ and the leverage identity above imply

$$
\begin{array} { r l } & { \| \widehat \theta _ { k } \| _ { \Sigma _ { k } } ^ { 2 } \leq 2 H ^ { 2 } ( Y _ { k , I _ { k } } - \widehat \mu _ { k } ) ^ { \top } ( \zeta ^ { 2 } \mathbf { I } _ { p } + \alpha \widehat \Sigma _ { k } ) ^ { - 1 } \widehat \Sigma _ { k } ( \zeta ^ { 2 } \mathbf { I } _ { p } + \alpha \widehat \Sigma _ { k } ) ^ { - 1 } ( Y _ { k , I _ { k } } - \widehat \mu _ { k } ) } \\ & { \qquad \leq \displaystyle \frac { 2 H ^ { 2 } } { \alpha } ( Y _ { k , I _ { k } } - \widehat \mu _ { k } ) ^ { \top } ( \alpha \widehat \Sigma _ { k } ) ^ { - 1 } ( Y _ { k , I _ { k } } - \widehat \mu _ { k } ) \leq \displaystyle \frac { 2 H ^ { 2 } p ^ { 2 } } { c _ { 0 } ^ { 2 } } . } \end{array}
$$

This proves the first two bounds in (19).

By (14) and (15),

$$
\nabla F \mathcal { O } _ { k } ( \mu _ { k } ) = - \eta Z _ { k - 1 } , \qquad \nabla ^ { 2 } F \mathcal { O } _ { k } ( \mu _ { k } ) = \Sigma _ { k } ^ { - 1 } ,
$$

and hence $B _ { k } = - 1 0 p \varepsilon _ { k } \nabla F \mathcal { O } _ { k } ( \mu _ { k } ) / c _ { 0 }$ . Equation (17) gives

$$
\| B _ { k } \| _ { \Sigma _ { k } } \leq \frac { 1 0 p ^ { 3 / 2 } \varepsilon _ { k } } { c _ { 0 } } .
$$

The triangle inequality and $( a + b ) ^ { 2 } \leq 2 a ^ { 2 } + 2 b ^ { 2 }$ , together with the bound on $\| \widehat { \theta } _ { k } \| _ { \Sigma _ { k } }$ above, prove the last bound in (19). □

Lemma C.3 (Bias–bonus compensation). Under the assumptions of Lemma C.2, for every $u \in \mathcal { O } _ { k } ,$ , it holds that

$$
\langle \mu _ { k } - u , \theta _ { k } \rangle \leq \mathbb { E } _ { k } [ \langle \mu _ { k } - u , \widetilde { \theta } _ { k } \rangle ] + \frac { 6 0 p ^ { 2 } \varepsilon _ { k } } { c _ { 0 } } .\tag{21}
$$

Proof of Lemma C.3. By (14) and (15),

$$
\nabla F \mathcal { O } _ { k } ( \mu _ { k } ) = - \eta Z _ { k - 1 } , \qquad \nabla ^ { 2 } F \mathcal { O } _ { k } ( \mu _ { k } ) = \Sigma _ { k } ^ { - 1 } ,
$$

and hence $B _ { k } = - 1 0 p \varepsilon _ { k } \nabla F \mathcal { O } _ { k } ( \mu _ { k } ) / c _ { 0 }$ . Local Cauchy–Schwarz, Eq. (19), and Eq. (16) give

$$
\begin{array} { r l } { \langle \mu _ { k } - u , \theta _ { k } - \mathbb { E } _ { k } [ \widehat { \theta } _ { k } ] \rangle \leq \displaystyle \frac { 5 p \varepsilon _ { k } } { 2 c _ { 0 } } \| \mu _ { k } - u \| _ { \nabla ^ { 2 } F _ { \mathcal O _ { k } } ( \mu _ { k } ) } } & { } \\ { \leq \displaystyle \frac { 1 0 p \varepsilon _ { k } } { c _ { 0 } } \langle \mu _ { k } - u , \nabla F _ { \mathcal O _ { k } } ( \mu _ { k } ) \rangle + \displaystyle \frac { 6 0 p ^ { 2 } \varepsilon _ { k } } { c _ { 0 } } } & { } \\ { = - \langle \mu _ { k } - u , B _ { k } \rangle + \displaystyle \frac { 6 0 p ^ { 2 } \varepsilon _ { k } } { c _ { 0 } } . } \end{array}
$$

Since $B _ { k }$ is $\mathcal { F } _ { k }$ -measurable,

$$
\langle \mu _ { k } - u , \theta _ { k } \rangle \leq \mathbb { E } _ { k } [ \langle \mu _ { k } - u , \widehat { \theta } _ { k } - B _ { k } \rangle ] + \frac { 6 0 p ^ { 2 } \varepsilon _ { k } } { c _ { 0 } } ,
$$

which proves (21).

## C.3 Overall regret analysis

Let $\mathcal { F } _ { k }$ be the information available after preparation of episode k and before the policy draw, and write $\mathbb { E } _ { k } [ \cdot ] = \mathbb { E } [ \cdot \mid \mathcal { F } _ { k } ]$ . Define $\chi _ { k }$ to equal one if all Bellman-confidence and moment-estimation events have held through this preparation. Once it becomes zero, it remains zero. Thus $\chi _ { k }$ is $\mathcal { F } _ { k }$ -measurable and, by Theorem D.2, the moment-estimation guarantee, and a union bound over all calls,

$$
\mathbb { P } ( \chi _ { k } = 1 { \mathrm { ~ f o r ~ e v e r y ~ } } k \in [ K ] ) \geq 1 - \delta .\tag{22}
$$

where $\delta = 2 K ^ { - 2 }$ . Whenever $\chi _ { k } = 1 , \mathcal { K } _ { \zeta } \subseteq \mathcal { O } _ { k } \subseteq R \mathbb { B } _ { 2 } ^ { p }$ and the moment and compiler guarantees hold. Write $\overline { { x } } _ { k } = \mathbb { E } _ { k } [ \phi ^ { \pi _ { k } } ]$ and set the auxiliary quantities to zero when $\chi _ { k } = 0$

Lemma C.4 (Cumulative estimated loss on nested regions). Let $\mathcal { O } _ { 1 } \supseteq \cdots \supseteq \mathcal { O } _ { K }$ satisfy $\begin{array} { r } { K _ { \zeta } \subseteq \mathcal { O } _ { k } \subseteq R \mathbb { B } _ { 2 } ^ { p } . } \end{array}$ . Let $\left( \mu _ { k } , \Sigma _ { k } \right)$ be the exact mean and covariance of the density proportional to $e ^ { - \eta Z _ { k - 1 } ^ { \top } x }$ on ${ \mathcal { O } } _ { k } ,$ , where $Z _ { 0 } = 0$ and $Z _ { k } = Z _ { k - 1 } + \widetilde { \theta } _ { k } . \ I f \eta \| \widetilde { \theta } _ { k } \| _ { \Sigma _ { k } } \leq 1 / 2$ for every k, thenfor each $u \in \kappa$

$$
\sum _ { k = 1 } ^ { K } \langle \mu _ { k } - u , \widetilde \theta _ { k } \rangle \leq \frac { p \log ( R / \zeta ) } { \eta } + \eta \sum _ { k = 1 } ^ { K } \| \widetilde \theta _ { k } \| _ { \Sigma _ { k } } ^ { 2 } .
$$

Proof. Apply Eq. (18) at $v = - \eta Z _ { k - 1 }$ with increment $- \eta \widetilde { \theta } _ { k }$ . Since $r = \eta \lVert \widetilde { \theta } _ { k } \rVert _ { \Sigma _ { k } } \leq 1 / 2$ and $- r - \log ( 1 - r ) \leq$ $r ^ { 2 } ,$

$$
\begin{array} { r } { \Phi _ { \mathcal { O } _ { k } } ( - \eta Z _ { k } ) - \Phi _ { \mathcal { O } _ { k } } ( - \eta Z _ { k - 1 } ) \le - \eta \langle \mu _ { k } , \widetilde \theta _ { k } \rangle + \eta ^ { 2 } \| \widetilde \theta _ { k } \| _ { \Sigma _ { k } } ^ { 2 } . } \end{array}
$$

Since $\mathcal { O } _ { k + 1 } \subseteq \mathcal { O } _ { k }$ , by Eq. (13) we have $\Phi _ { { \mathcal { O } } _ { k + 1 } } \le \Phi _ { { \mathcal { O } } _ { k } }$ , and thus

$$
\begin{array} { r l } { \displaystyle \Phi _ { \mathcal { O } _ { K } } \bigl ( - \eta Z _ { K } \bigr ) - \Phi _ { \mathcal { O } _ { 1 } } \bigl ( 0 \bigr ) = \displaystyle \sum _ { k = 1 } ^ { K } \bigl [ \Phi _ { \mathcal { O } _ { k } } \bigl ( - \eta Z _ { k } \bigr ) - \Phi _ { \mathcal { O } _ { k } } \bigl ( - \eta Z _ { k - 1 } \bigr ) \bigr ] } & { } \\ { + \displaystyle \sum _ { k = 1 } ^ { K - 1 } \bigl [ \Phi _ { \mathcal { O } _ { k + 1 } } \bigl ( - \eta Z _ { k } \bigr ) - \Phi _ { \mathcal { O } _ { k } } \bigl ( - \eta Z _ { k } \bigr ) \bigr ] } & { } \\ { \leq - \eta \displaystyle \sum _ { k = 1 } ^ { K } \bigl \langle \mu _ { k } , \widetilde { \theta } _ { k } \bigr \rangle + \eta ^ { 2 } \displaystyle \sum _ { k = 1 } ^ { K } \| \widetilde { \theta } _ { k } \| _ { \Sigma _ { k } } ^ { 2 } . } \end{array}
$$

Finally, $\Phi _ { \mathcal { O } _ { 1 } } ( 0 ) \le \log \mathrm { V o l } ( R \mathbb { B } _ { 2 } ^ { p } )$ and Jensen on $u + \zeta \mathbb { B } _ { 2 } ^ { p } \subseteq \mathcal { O } _ { K }$ give

$$
\int _ { \mathcal { O } _ { K } } e ^ { - \eta Z _ { K } ^ { \top } x } d x \geq e ^ { - \eta Z _ { K } ^ { \top } u } \mathrm { V o l } ( \zeta \mathbb { B } _ { 2 } ^ { p } ) .
$$

Using $Z _ { K } = \textstyle \sum _ { k } \widetilde { \theta } _ { k }$ yields

$$
\sum _ { k = 1 } ^ { K } \langle \mu _ { k } - u , \widetilde \theta _ { k } \rangle \leq \frac { p \log ( R / \zeta ) } { \eta } + \eta \sum _ { k = 1 } ^ { K } \| \widetilde \theta _ { k } \| _ { \Sigma _ { k } } ^ { 2 } .\tag{23}
$$

□

Proof of Lemma 4.1. Define $\begin{array} { r } { \mathcal { E } _ { K } : = \mathbb { E } \left[ \sum _ { k = 1 } ^ { K } \chi _ { k } \varepsilon _ { k } ^ { 2 } \right] } \end{array}$ . On the valid prefix, Lemma C.3 gives

$$
\langle \mu _ { k } - u , \theta _ { k } \rangle \le \mathbb { E } _ { k } \left[ \langle \mu _ { k } - u , \widetilde { \theta } _ { k } \rangle \right] + \frac { 6 0 p ^ { 2 } \varepsilon _ { k } } { c _ { 0 } } .\tag{24}
$$

Furthermore,

$$
{ \overline { { x } } } _ { k } - \mu _ { k } = ( { \widehat { \mu } } _ { k } - \mu _ { k } ) + { \frac { 1 } { 2 p } } \sum _ { i } ( x _ { k , i } - Y _ { k , i } ) .\tag{25}
$$

Since $p = d H$ , the RMS approximation and moment bounds imply

$$
\langle \overline { { x } } _ { k } - u , \theta _ { k } \rangle \leq \mathbb { E } _ { k } \left[ \langle \mu _ { k } - u , \widetilde { \theta } _ { k } \rangle \right] + e _ { k } , \qquad e _ { k } : = \left( \frac { 6 0 p ^ { 2 } } { c _ { 0 } } + \sqrt { p } \right) \varepsilon _ { k } + 2 R \sqrt { p } \delta _ { \mu } .\tag{26}
$$

Write $r _ { k } = \langle \phi ^ { \pi _ { k } } - u , \theta _ { k } \rangle$ . Since $\chi _ { k }$ is $\mathcal { F } _ { k }$ -measurable,

$$
\mathbb { E } \left[ \sum _ { k = 1 } ^ { K } \chi _ { k } r _ { k } \right] = \mathbb { E } \left[ \sum _ { k = 1 } ^ { K } \chi _ { k } \langle \overline { { x } } _ { k } - u , \theta _ { k } \rangle \right] .\tag{27}
$$

Multiplying Eq. (26) by $\chi _ { k }$ and taking expectations gives

$$
\mathbb { E } \left[ \sum _ { k = 1 } ^ { K } \chi _ { k } r _ { k } \right] \leq \mathbb { E } \left[ \sum _ { k = 1 } ^ { K } \chi _ { k } \langle \mu _ { k } - u , \widetilde { \theta } _ { k } \rangle \right] + \mathbb { E } \left[ \sum _ { k = 1 } ^ { K } \chi _ { k } e _ { k } \right] .\tag{28}
$$

Here we used the tower property, since $\chi _ { k } ( \mu _ { k } - u )$ is $\mathcal { F } _ { k }$ -measurable.

By Lemma C.2,

$$
\left\| \widetilde { \theta } _ { k } \right\| _ { \Sigma _ { k } } ^ { 2 } \leq \frac { 4 H ^ { 2 } p ^ { 2 } } { c _ { 0 } ^ { 2 } } + \frac { 2 0 0 p ^ { 3 } \varepsilon _ { k } ^ { 2 } } { c _ { 0 } ^ { 2 } } .
$$

On the valid prefix, $\varepsilon _ { k } \leq 6 \sqrt { H }$ . Provided $\eta \leq c _ { 0 } / ( 1 2 8 p ^ { 3 / 2 } \sqrt { H } )$ , Eq. (19) gives

$$
\eta \left\| \widetilde { \theta } _ { k } \right\| _ { \Sigma _ { k } } \leq \eta \left( \frac { \sqrt 2 H p } { c _ { 0 } } + \frac { 1 0 p ^ { 3 / 2 } \varepsilon _ { k } } { c _ { 0 } } \right) \leq \frac { 6 0 + \sqrt { 2 / d } } { 1 2 8 } \leq \frac { 6 0 + \sqrt 2 } { 1 2 8 } < \frac { 1 } { 2 } ,
$$

which verifies the condition of the nested-region potential bound. On every sample path, the indices with $\chi _ { k } = 1$ form a prefix. Applying Lemma C.4 to this realized prefix and then taking expectations yields

$$
\mathbb { E } \left[ \sum _ { k = 1 } ^ { K } \chi _ { k } \langle \mu _ { k } - u , \widetilde { \theta } _ { k } \rangle \right] \leq \frac { p \log ( R / \zeta ) } { \eta } + \frac { 4 \eta H ^ { 2 } p ^ { 2 } K } { c _ { 0 } ^ { 2 } } + \frac { 2 0 0 \eta p ^ { 3 } } { c _ { 0 } ^ { 2 } } \mathcal { E } _ { K } .
$$

Furthermore,

$$
\mathbb { E } \left[ \sum _ { k } \chi _ { k } \varepsilon _ { k } \right] \leq \sqrt { K \mathcal { E } _ { K } } .
$$

Substitution into Eq. (28), followed by $p = d H$ , gives

$$
\mathbb { E } \left[ \sum _ { k = 1 } ^ { K } \chi _ { k } r _ { k } \right] \leq \widetilde { O } \bigg ( \frac { d H } { \eta } + \eta d ^ { 2 } H ^ { 4 } K + \eta d ^ { 3 } H ^ { 3 } \mathcal { E } _ { K } + d ^ { 2 } H ^ { 2 } \sqrt { K \mathcal { E } _ { K } } + R \sqrt { p } \delta _ { \mu } K \bigg ) .
$$

Finally, $r _ { k } \le H$ , and by Eq. (22),

$$
\mathbb { E } \left[ \sum _ { k = 1 } ^ { K } r _ { k } \right] = \mathbb { E } \left[ \sum _ { k = 1 } ^ { K } \chi _ { k } r _ { k } \right] + \mathbb { E } \left[ \sum _ { k = 1 } ^ { K } ( 1 - \chi _ { k } ) r _ { k } \right]
$$

$$
\leq \mathbb { E } \left[ \sum _ { k = 1 } ^ { K } \chi _ { k } r _ { k } \right] + H K \delta .
$$

Using $\delta _ { \mu } = 2 c _ { 0 } / \sqrt { K } , \delta = 2 / ( K + 1 ) ^ { 2 } , R = \sqrt { H } + K ^ { - 1 / 2 }$ , and $p = d H$ , we have

$$
R \sqrt { p } \delta _ { \mu } K + H K \delta = O \left( R \sqrt { p K } + \frac { H } { K } \right) = O \left( H \sqrt { d K } \right) .
$$

The unconstrained minimizer of the two $\mathcal { E } _ { K }$ -independent η-terms is $( d H ^ { 3 } K ) ^ { - 1 / 2 }$ . To also ensure the local-norm condition above, choose

$$
\eta = { \frac { c _ { 0 } } { 1 2 8 } } \operatorname* { m i n } \Bigl \{ ( d H ^ { 3 } K ) ^ { - 1 / 2 } , d ^ { - 3 / 2 } H ^ { - 2 } \Bigr \} .
$$

The second branch is exactly the required stability cap. Moreover, the pathwise bound $\mathcal { E } _ { K } \leq 3 6 H K$ gives

$$
\eta d ^ { 3 } H ^ { 3 } \mathcal { E } _ { K } \leq O \Big ( d ^ { 3 / 2 } H \mathcal { E } _ { K } \Big ) \leq O \Big ( d ^ { 2 } H ^ { 2 } \sqrt { K \mathcal { E } _ { K } } \Big ) .
$$

Thus the $\eta \mathcal { E } _ { K }$ term is absorbed by the $d ^ { 2 } H ^ { 2 } \sqrt { K { \mathcal { E } } _ { K } }$ term. In either branch, $H \sqrt { d K } = { \cal O } ( \eta d ^ { 2 } H ^ { 4 } K )$ . If $K \geq d ^ { 2 } H$ , the first branch applies and

$$
\frac { d { \cal H } } { \eta } + \eta d ^ { 2 } { \cal H } ^ { 4 } { \cal K } = { \cal O } \Big ( d ^ { 3 / 2 } { \cal H } ^ { 5 / 2 } \sqrt { { \cal K } } \Big ) .
$$

If $K < d ^ { 2 } H$ , the trivial bound $\mathrm { R e g } _ { K } \leq H K$ is no larger than the same expression. Therefore

$$
\mathrm { R e g } _ { K } \leq \widetilde { O } \left( d ^ { 3 / 2 } H ^ { 5 / 2 } \sqrt { K } + d ^ { 2 } H ^ { 2 } \sqrt { K \mathcal { E } _ { K } } \right) .
$$

This proves Lemma 4.1.

□

## D Omitted Proofs in Section 5

Lemma D.1 (Averaging recursion). Let $a _ { 0 } , a _ { 1 } , \dotsc , a _ { M }$ and $b _ { 1 } , \dots , b _ { M }$ be real sequences, and let $c \geq 0 .$ Suppose that, for every $j \in [ M ]$

$$
a _ { j } \leq \left( 1 - { \frac { 1 } { j } } \right) a _ { j - 1 } + { \frac { b _ { j } } { j } } + { \frac { c } { j ^ { 2 } } } .
$$

Then, for every $m \in [ M ]$

$$
a _ { m } \leq \frac { 1 } { m } \sum _ { j = 1 } ^ { m } b _ { j } + \frac { c ( 1 + \log m ) } { m } .\tag{29}
$$

Proof. Multiplying the recursion by j gives

$$
j a _ { j } \leq ( j - 1 ) a _ { j - 1 } + b _ { j } + \frac { c } { j } .
$$

Summing from $j = 1$ to m telescopes the terms involving $a _ { j }$ and yields

$$
m a _ { m } \leq \sum _ { j = 1 } ^ { m } b _ { j } + c \sum _ { j = 1 } ^ { m } { \frac { 1 } { j } } ,
$$

which proves Eq. (29) because $\begin{array} { r } { \sum _ { j = 1 } ^ { m } j ^ { - 1 } \leq 1 + } \end{array}$ log m.

Theorem D.2 (Uniform Bellman confidence (Lemma D.4 of Jin et al. [2020])). Fix $\delta \in ( 0 , 1 )$ and $\epsilon \in ( 0 , 1 )$ For each $h \in [ H ]$ , let $C \geq 1$ and let $\mathcal { F }$ be a deterministic class ofscalarfunctionsfrom $S _ { h + 1 } t o \left[ - C , C \right]$ . Define the relative covering number

$$
\begin{array} { r } { N _ { \mathcal { F } } ( \epsilon ) : = \mathcal { N } _ { \infty } ( C \epsilon , \mathcal { F } ) , } \end{array}
$$

where $\mathcal { N } _ { \infty }$ denotes the sup-norm covering number. Assume $N _ { \mathcal { F } } ( \epsilon ) < \infty .$ . There is a universal constant $c _ { 1 } > 0$ such that, if

$$
\beta \ge c _ { 1 } \left( \sqrt { d \log K + \log \left( \frac { H N _ { \mathcal { F } } ( \epsilon ) } { \delta } \right) } + K \epsilon + \sqrt { d } \right) ,\tag{30}
$$

then, with probability at least $1 - \delta ,$ , simultaneously for every $k \in [ K ] , h \in [ H ] , f \in \mathcal { F } _ { h + 1 }$ , and $( s , a ) \in S _ { h } \times \mathcal { A } ,$

$$
\begin{array} { r } { | ( \widehat { P } _ { k , h } - P _ { h } ) f ( s , a ) | \leq \beta C \| \phi _ { h } ( s , a ) \| _ { \Lambda _ { k , h } ^ { - 1 } } . } \end{array}\tag{31}
$$

ProofofTheorem 5.3. All three tasks in Algorithm 3 are based on least-squares value iteration. The key is to show (31) for all three tasks, each corresponding to a different function class ${ \mathcal F } .$ . Fix a layer h, write $\alpha = ( \alpha _ { 1 } , \alpha _ { 2 } , \alpha _ { 3 } )$ , and define

$$
\nu _ { h } ( s , a ; \Lambda ) = \sqrt { \phi _ { h } ( s , a ) ^ { \top } \Lambda ^ { - 1 } \phi _ { h } ( s , a ) }
$$

$$
\begin{array} { r } { q _ { h } ( s , a ; \Lambda , \vartheta , \alpha ) = \mathrm { C L I P } _ { [ \alpha _ { 1 } , \alpha _ { 2 } ] } \Big ( \phi _ { h } ( s , a ) ^ { \top } \vartheta + \alpha _ { 3 } \nu _ { h } ( s , a ; \Lambda ) \Big ) , } \end{array}
$$

$$
\pi _ { h } ( a \mid s ; \Lambda , \widetilde { \vartheta } ) \propto \exp \left( - q _ { h } ( s , a ; \Lambda , \widetilde { \vartheta } , \alpha ^ { \mathrm { p l } } ) / \tau \right) ,
$$

$$
v _ { h } ( s ; \Lambda , \vartheta , \widetilde { \vartheta } , \alpha ) = \sum _ { a } \pi _ { h } ( a \mid s ; \Lambda , \widetilde { \vartheta } ) q _ { h } ( s , a ; \Lambda , \vartheta , \alpha ) ,
$$

where

$$
\alpha ^ { \mathrm { p l } } = ( - \sqrt { H } , \sqrt { H } , - ( \sqrt { H } + 1 ) \beta ) , \qquad \alpha ^ { \mathrm { f t } } = ( - 1 , 1 , 0 ) , \qquad \alpha ^ { \mathrm { w c } } = ( 0 , H - h + 1 , 1 + H \beta ) .
$$

The function classes for the three tasks are respectively

$$
\begin{array} { l l } { { s \mapsto - \tau \log \displaystyle \sum _ { a } \exp \left( - q _ { h } ( s , a ; \Lambda , \vartheta , \alpha ^ { \mathrm { p l } } ) / \tau \right) } } & { { \qquad \mathrm { ( p l a n n i n g ) } } } \\ { { \displaystyle s \mapsto v _ { h } ( s ; \Lambda , \vartheta , \widetilde { \vartheta } , \alpha ^ { \mathrm { f t } } ) } } & { { \qquad \mathrm { ( f e a t u r e ~ e v a l u a t i o n ) } } } \\ { { \displaystyle s \mapsto v _ { h } ( s ; \Lambda , \vartheta , \widetilde { \vartheta } , \alpha ^ { \mathrm { u n c } } ) . } } & { { \qquad \mathrm { ( u n c e r t a i n t y ~ e v a l u a t i o n ) } } } \end{array}
$$

Denote these classes by $\mathcal { F } _ { h } ^ { \mathrm { p l } } , \mathcal { F } _ { h } ^ { \mathrm { f t } }$ , and ${ \mathcal { F } } _ { h } ^ { \mathrm { u n c } }$ , respectively, and set their terminal classes to $\{ 0 \}$ , with terminal scale one.

Bounding the covering number of each classes. Clipping and τ log $A \leq 1$ give the uniform bounds

$$
\begin{array} { r l } { f \in \mathscr { F } _ { h } ^ { \mathrm { p l } } : } & { \| f \| _ { \infty } \leq \sqrt { H } + 1 , } \\ { f \in \mathscr { F } _ { h } ^ { \mathrm { f t } } : } & { \| f \| _ { \infty } \leq 1 , } \\ { f \in \mathscr { F } _ { h } ^ { \mathrm { u n c } } : } & { \| f \| _ { \infty } \leq H - h + 1 . } \end{array}\tag{32}
$$

Write $C _ { h } ^ { \mathrm { p l } } = \sqrt { H } + 1 , C _ { h } ^ { \mathrm { f t } } = 1$ , and $C _ { h } ^ { \mathrm { u n c } } = H - h + 1$ for these scales. Therefore,

$$
\left\| \Lambda _ { k , h } ^ { - 1 } \sum _ { ( u , s ^ { \prime } ) \in \mathcal { D } _ { k , h } } u f ( s ^ { \prime } ) \right\| _ { 2 } \leq \sum _ { ( u , s ^ { \prime } ) \in \mathcal { D } _ { k , h } } \| u \| _ { 2 } | f ( s ^ { \prime } ) | \leq 3 K H .
$$

Including the immediate reward/cost gives $\lVert \vartheta \rVert _ { 2 } , \lVert \widetilde { \vartheta } \rVert _ { 2 } \leq 3 K H + 1$ . Also, $\mathbf { I } _ { d } \preceq \Lambda _ { k , h } \preceq ( K + 1 ) \mathbf { I } _ { d }$ . Thus, each class has at most

$$
n = { \frac { d ( d + 1 ) } { 2 } } + 2 d \leq 3 d ^ { 2 }
$$

scalar parameters, and each of them is bounded in absolute value by $\overline { { R } } = \operatorname* { m a x } \{ K + 1 , 3 K H + 1 \}$ . Moreover,

$$
\left| \nu _ { h } ( s , a ; \Lambda ) - \nu _ { h } ( s , a ; \Lambda ^ { \prime } ) \right| \leq \frac { 1 } { 2 } \left\| \Lambda - \Lambda ^ { \prime } \right\| _ { \mathrm { F } } ,
$$

so clipping gives, for each fixed α,

$$
\left\| q _ { h } ( \cdot ; \Lambda , \vartheta , \alpha ) - q _ { h } ( \cdot ; \Lambda ^ { \prime } , \vartheta ^ { \prime } , \alpha ) \right\| _ { \infty } \leq \left\| \vartheta - \vartheta ^ { \prime } \right\| _ { 2 } + \frac { | \alpha _ { 3 } | } { 2 } \left\| \Lambda - \Lambda ^ { \prime } \right\| _ { \mathrm { F } } .
$$

The softmax map satisfies

$$
\operatorname* { s u p } _ { s } \left\| \pi _ { h } ( \cdot \mid s ; q ) - \pi _ { h } ( \cdot \mid s ; q ^ { \prime } ) \right\| _ { 1 } \leq \frac { 2 } { \tau } \left\| q - q ^ { \prime } \right\| _ { \infty } .
$$

Hence, if C denotes the corresponding scale in (32),

$$
\left\| v _ { h } - v _ { h } ^ { \prime } \right\| _ { \infty } \leq \left\| q _ { h } - q _ { h } ^ { \prime } \right\| _ { \infty } + C \operatorname* { s u p } _ { s } \left\| \pi _ { h } ( \cdot  { | } s ) - \pi _ { h } ^ { \prime } ( \cdot  { | } s ) \right\| _ { 1 } .
$$

The planning soft minimum is nonexpansive in sup norm. Since $C \geq 1 , | \alpha _ { 3 } | / C \leq 1 + H \beta$ , and $( \sqrt { H } + 1 ) \beta \leq$ $2 H \beta$ , the evaluation is L-Lipschitz with respective to its parameters, with

$$
L = \frac { 8 ( 1 + H \beta ) } { \tau } .
$$

A parameter grid of mesh $\epsilon / ( n L )$ gives a Cϵ-cover. Therefore,

$$
\log \mathcal { N } _ { \infty } \big ( C \epsilon , \mathcal { F } \big ) = O \left( n \log \left( 1 + \frac { \overline { { R } } n L } { \epsilon } \right) \right) .\tag{33}
$$

Bounding error. Take

$$
\delta = K ^ { - 2 } , \qquad \epsilon _ { \mathrm { n e t } } = K ^ { - 2 } , \qquad \beta = C _ { \beta } d \sqrt { \log ( K d H A ) } .
$$

Since $\tau ^ { - 1 } = H \sqrt { K } \log ( e A )$ , (33) gives

$$
\log N _ { \mathcal { F } ^ { t } } ( \epsilon _ { \mathrm { n e t } } ) = O \bigl ( d ^ { 2 } \log ( K d H A ) \bigr ) .
$$

Thus a sufficiently large universal $C _ { \beta }$ makes (30) hold, and $\beta = { \widetilde { O } } ( d )$ . Applying Theorem D.2 directly to the three classes, with failure probability $\delta / 3$ each, and taking a union bound yields

$$
| ( \widehat { P } _ { k , h } - P _ { h } ) V _ { h + 1 } ^ { \mathrm { p l } } ( s , a ) | \leq ( \sqrt { H } + 1 ) \beta \nu _ { k , h } ( s , a ) ,\tag{34}
$$

$$
| ( \widehat { P } _ { k , h } - P _ { h } ) V _ { h + 1 } ^ { \mathrm { f t } , m } ( s , a ) | \le \beta \nu _ { k , h } ( s , a ) , \qquad m \in [ p ] ,\tag{35}
$$

$$
| ( \widehat { P } _ { k , h } - P _ { h } ) \widehat { U } _ { h + 1 } ( s , a ) | \leq H \beta \nu _ { k , h } ( s , a ) .\tag{36}
$$

Fix one call and let π be its returned policy. Write

$$
U _ { k } ^ { \pi } = \mathbb { E } _ { P , \pi } \Bigg [ \sum _ { h = 1 } ^ { H } \nu _ { k , h } ( S _ { h } , A _ { h } ) \Bigg ] \mathrm { ~ . ~ }
$$

By backward induction [Jin et al., 2020] on the update in Line 4 with confidence bound (34), and that $V _ { h } ^ { \mathrm { p l } } ( s ) \leq$ min<sub>a</sub> $Q _ { h } ^ { \mathrm { p l } } ( s , a )$ , we have $V _ { h } ^ { \mathrm { p l } } \le \operatorname* { m i n } _ { \pi } V _ { h } ^ { \pi }$ . Therefore,

$$
V _ { 1 } ^ { \mathrm { p l } } ( s _ { 1 } ) \leq \operatorname* { m i n } _ { u \in \mathcal { K } } \bar { c } ^ { \top } u .\tag{37}
$$

Likewise, backward induction on the update in Line 12 with confidence bound (36) gives

$$
U _ { k } ^ { \pi } \leq { \widehat { U } } _ { 1 } ( s _ { 1 } ) .\tag{38}
$$

Similarly, backward induction on the update in Line 10 with coordinatewise confidence bounds (35) gives

$$
\| \widehat { \phi } - \phi ^ { \pi } \| _ { 2 } \leq \sqrt { p } \beta U _ { k } ^ { \pi } \leq \sqrt { p } \beta \widehat { U } _ { 1 } ( s _ { 1 } ) .\tag{39}
$$

The feature-estimation error $\| \widehat { \phi } - \phi ^ { \pi } \| _ { 2 }$ is at most $2 \sqrt { H }$ because $\widehat { \phi } , \phi ^ { \pi } \in \sqrt { H } \mathbb { B } _ { 2 } ^ { p }$ . Thus,

$$
\| \widehat { \phi } - \phi ^ { \pi } \| _ { 2 } \leq \operatorname* { m i n } \Big \{ 5 \sqrt { p } \beta \widehat { U } _ { 1 } ( s _ { 1 } ) + H \tau \log A , ~ 2 \sqrt { H } \Big \} = \Delta
$$

proving the first bound in (9).

Comparing the planning recursion with the true value of the returned Gibbs policy, each layer contributes at most $2 ( \sqrt { H } + 1 ) \beta \nu _ { k , h }$ from transition error and at most τ log A from the soft minimum. Hence

$$
\bar { c } ^ { \top } \phi ^ { \pi } - V _ { 1 } ^ { \mathrm { p l } } ( s _ { 1 } ) \leq 2 ( \sqrt { H } + 1 ) \beta U _ { k } ^ { \pi } + H \tau \log A .\tag{40}
$$

The same uncertainty recursion also gives

$$
\widehat { U } _ { 1 } ( s _ { 1 } ) \leq 2 \beta H U _ { k } ^ { \pi } .\tag{41}
$$

Combining (39), (37), (40) and using $p = d H$ yields

$$
\bar { c } ^ { \top } \widehat \phi - \operatorname* { m i n } _ { u \in \mathcal K } \bar { c } ^ { \top } u \leq 5 \sqrt { p } \beta U _ { k } ^ { \pi } + H \tau \log A \leq 5 \sqrt { p } \beta \widehat U _ { 1 } ( s _ { 1 } ) + H \tau \log A .\tag{42}
$$

The normalized optimization gap $\bar { c } ^ { \top } \widehat { \phi } - \operatorname* { m i n } _ { u \in \mathcal { K } } \bar { c } ^ { \top }$ u is at most $2 \sqrt { H }$ because $\widehat { \phi } \in \sqrt { H } \mathbb { B } _ { 2 } ^ { p } , { \cal K } \subseteq \sqrt { H } \mathbb { B } _ { 2 } ^ { p }$ , and $\| \bar { c } \| _ { 2 } \leq 1$ . Thus,

$$
\bar { c } ^ { \top } \widehat \phi - \operatorname* { m i n } _ { u \in \mathcal K } \bar { c } ^ { \top } u \leq \operatorname* { m i n } \left\{ 5 \sqrt { p } \beta \widehat U _ { 1 } ( s _ { 1 } ) + H \tau \log A , \ 2 \sqrt { H } \right\} = \Delta
$$

proving the second bound in (9).

Finally, (41) gives

$$
\begin{array} { r } { \Delta \leq 1 0 \sqrt { p } \beta ^ { 2 } H U _ { k } ^ { \pi } + H \tau \log A , } \end{array}
$$

proving the final bound.

## E Omitted Proofs in Section 6

## E.1 Moment Estimates

We let MOMENTESTIMATE to be a procedure described in Ito et al. [2019], which takes a convex function $f : \mathbb { R } ^ { p } $ R and a convex body $\mathcal { P } \subset \mathbb { R } ^ { p }$ as input, and output estimators of the mean and covariance of the distribution $q ( x ) \propto \exp ( - f ( x ) ) \mathbf { 1 } \{ x \in \mathcal { P } \}$ within polynomial time. In our case $f ( x ) = \eta x ^ { \top } Z _ { t - 1 }$ and $\mathcal { P } = \mathcal { O }$ Its guarantee is the following:

Lemma E.1 (Corollary 1 of Ito et al. [2019]). Let $\mathcal { P } \subset \mathbb { R } ^ { p }$ be a convex body with non-zero Lubesgue measure, and let $q ( x ) \propto \exp ( - f ( x ) ) \mathbf { 1 } \{ x \in \mathcal { P } \}$ be a log-concave distribution with some convex function $f : \mathcal P \to$ R. Let $\mu$ and Σ denote the mean and covariance $o f q .$ . Then there is an algorithm that outputs a vector ${ \widehat { \mu } } \in \mathbb { R } ^ { p }$ and a symmetric matrix $\widehat { \Sigma } \in \mathbb { R } ^ { p \times p }$ satisfying $ { \frac { 1 } { 2 } }  { \Sigma } \preceq \widehat { \Sigma } \preceq 2 \Sigma$ and $\| { \widehat { \mu } } - \mu \| _ { \mathrm { C o v } ( q ) ^ { - 1 } } \leq \varepsilon$ with probability at least $1 - \delta .$ . The computational time, the number of calls for the membershpi oracle of P, and the value oracle of f are bounded by poly $\begin{array} { r } { ( p , \frac { 1 } { \varepsilon } , \log \frac { 1 } { \delta } ) } \end{array}$

We let $\varepsilon = \delta _ { \mu } = 2 c _ { 0 } / \sqrt { K }$ , which implies the estimation accuracy

$$
\| \widehat { \mu } _ { k } - \mu _ { k } \| _ { { \Sigma } _ { k } ^ { - 1 } } \leq \delta _ { \mu } , \qquad \frac { 1 } { 2 } { \Sigma } _ { k } \preceq \widehat { { \Sigma } } _ { k } \preceq 2 { \Sigma } _ { k }
$$

after calling MOMENTESTIMATE in Line 6 of Algorithm 1.

## E.2 Number of cuts

Lemma E.2 (Containment of the estimated covariance ellipsoid). Let q be a full-dimensional log-concave distribution supported on a convex body $\mathcal { O } \subseteq \mathbb { R } ^ { p }$ , with mean $\mu$ and covariance $\Sigma .$ Suppose $\widehat { \mu }$ and $\widehat { \Sigma }$ satisfy $ { \frac { 1 } { 2 } }  { \Sigma } \preceq \widehat { \Sigma } \preceq 2 \Sigma$ and $\| { \widehat { \boldsymbol { \mu } } } - { \boldsymbol { \mu } } \| _ { \Sigma ^ { - 1 } } \leq { \frac { 1 } { 1 2 e } }$ . Then

$$
\widehat { \mu } + \frac 1 { 1 6 e } \widehat { \Sigma } ^ { 1 / 2 } \mathbb { B } _ { 2 } ^ { p } \subseteq \mu + \frac 1 { 2 e } \Sigma ^ { 1 / 2 } \mathbb { B } _ { 2 } ^ { p } \subseteq \mathcal O .\tag{43}
$$

Proof. We first show that

$$
\mu + \frac { 1 } { e } \Sigma ^ { 1 / 2 } \mathbb { B } _ { 2 } ^ { p } \subseteq \mathcal { O } .\tag{44}
$$

Otherwise, separation gives a nonzero vector a and a point $y$ with $\| y - \mu \| _ { \Sigma ^ { - 1 } } \leq 1 / e$ such that $a ^ { \top } y \textless$ in $\boldsymbol { \mathrm { f } } _ { x \in \mathcal { O } } \boldsymbol { a } ^ { \top } \boldsymbol { x }$ . For $X \sim q .$ , denote the random scalar

$$
S = \frac { a ^ { \top } ( X - \mu ) } { \sqrt { a ^ { \top } \Sigma a } }
$$

and denote the density function of $S$ as $q _ { S }$ . Then $S$ is centered, variance-one, and log-concave. Moreover, its support is contained in $[ u _ { \mathrm { m i n } } , \infty )$ : Indeed, strict separation and compactness of $\mathcal { O }$ give

$$
u _ { \mathrm { m i n } } : = \frac { \operatorname* { i n f } _ { x \in \mathcal { O } } a ^ { \top } ( x - \mu ) } { \sqrt { a ^ { \top } \Sigma a } } > \frac { a ^ { \top } ( y - \mu ) } { \sqrt { a ^ { \top } \Sigma a } } \geq - \frac { 1 } { e } ,
$$

where the last inequality follows from Cauchy–Schwarz in the Σ-norm. Lemmas 5.4 and 5.5(a) of Lovász and Vempala [2007] give $^ { \mathtt { p } } ( S < 0 ) \geq 1 / e$ and $q _ { S } ( s ) \leq 1$ for any s. However, the support condition would instead imply

$$
\mathbb { P } ( S < 0 ) = \int _ { u _ { \mathrm { m i n } } } ^ { 0 } q _ { S } ( t ) d t \le | u _ { \mathrm { m i n } } | < \frac { 1 } { e } ,
$$

a contradiction. This proves Eq. (44).

Now fix $v \in \mathbb { B } _ { 2 } ^ { p }$ . The covariance assumption gives $\| \Sigma ^ { - 1 / 2 } \widehat \Sigma ^ { 1 / 2 } \| _ { \mathrm { o p } } \leq \sqrt { 2 }$ . Therefore

$$
\begin{array} { r l r } {  { \| \widehat { \mu } + \frac { 1 } { 1 6 e } \widehat { \Sigma } ^ { 1 / 2 } v - \mu \| _ { \Sigma ^ { - 1 } } \leq \| \widehat { \mu } - \mu \| _ { \Sigma ^ { - 1 } } + \frac { 1 } { 1 6 e } \| \Sigma ^ { - 1 / 2 } \widehat { \Sigma } ^ { 1 / 2 } v \| _ { 2 } } } \\ & { } & { \leq \frac { 1 } { 1 2 e } + \frac { \sqrt { 2 } } { 1 6 e } < \frac { 1 } { 2 e } . } \end{array}
$$

The first inclusion in Eq. (43) follows, and the second follows from Eq. (44).

Lemma E.3 (Number of cuts). Suppose every invoked moment call, with exact mean µ and covariance $\Sigma ,$ returns estimates satisfying $\textstyle { \frac { 1 } { 2 } } \sum \preceq \widehat { \Sigma } \preceq 2 \Sigma$ and $\| \widehat { \mu } - \mu \| _ { \Sigma ^ { - 1 } } \leq 1 / ( 1 2 e )$ . Then the number of cuts is upper bounded by $\widetilde { O } ( p K + \sqrt { H } K ^ { 5 / 2 } )$

Proof. By the definition of $\widehat { \theta } _ { k }$ , we have $\begin{array} { r } { \| \widehat { \theta } _ { k } \| _ { 2 } \leq \frac { 2 R H } { \zeta ^ { 2 } } } \end{array}$ . Since the update is $\begin{array} { r } { Z _ { k } = ( 1 - \frac { 1 0 \eta p \varepsilon _ { k } } { c _ { 0 } } ) Z _ { k - 1 } + \widehat { \theta } _ { k } } \end{array}$ , we have $\lVert Z _ { k } \rVert \leq \lVert Z _ { k - 1 } \rVert + \lVert \widehat { \theta } _ { k } \rVert$ . By induction, it holds that $\begin{array} { r } { \| Z _ { k } \| \le \frac { 2 R H K } { \zeta ^ { 2 } } } \end{array}$ for all k.

By Lemma E.2, every target used during preparation belongs to the current region O and satisfies $\| Y _ { i } - \mu \| _ { \Sigma ^ { - 1 } } <$ $1 / ( 2 e )$ . The local-cut argument of Ito et al. [2019, Lemma 5] shows that each cut removes at least a $\textstyle { \frac { 1 } { 2 e } }$ fraction of the current mass. Let $N _ { k }$ be the number of new cuts added in episode $k ,$ and let $\mathcal { O } _ { i } , \mathcal { O } _ { f }$ be the set $\mathcal { O }$ at the start and the end of episode k. Then we have

$$
\begin{array} { r l } { \mathrm { V o l } ( \zeta \mathbb { B } _ { 2 } ^ { p } ) e ^ { - \eta R \| Z _ { k - 1 } \| } \le \displaystyle \int _ { { \mathcal { O } } _ { f } } e ^ { - \eta x ^ { \top } Z _ { k - 1 } } d x } & { } \\ { \le \displaystyle \left( 1 - \frac { 1 } { 2 e } \right) ^ { N _ { k } } \displaystyle \int _ { { \mathcal { O } } _ { i } } e ^ { - \eta x ^ { \top } Z _ { k - 1 } } d x } & { } \\ { \le \displaystyle \left( 1 - \frac { 1 } { 2 e } \right) ^ { N _ { k } } \mathrm { V o l } ( R \mathbb { B } _ { 2 } ^ { p } ) e ^ { \eta R \| Z _ { k - 1 } \| } . } \end{array}
$$

Taking logarithms gives

$$
N _ { k } \leq O \left( p \log ( R / \zeta ) + \eta R \| Z _ { k - 1 } \| \right) \leq O \left( p \log ( R / \zeta ) + \frac { \eta R ^ { 2 } K H } { \zeta ^ { 2 } } \right) \leq \widetilde O \big ( p + \sqrt { H } K ^ { 3 / 2 } \big ) .
$$

Summing over episodes gives the desired bound.

## E.3 Implementation of Algorithm 3

Although Algorithm 3 defines value functions on the entire state space, it can be executed without enumerating states or state–action pairs. Below is an efficient way to implement Algorithm 3. We just use the computation of $V _ { h } ^ { \mathrm { p l } }$ as example; the other two tasks follow the same manner.

For any function $f : S _ { h + 1 } \to \mathbb { R }$ , define:

$$
\widehat { w } _ { h } ( f ) : = \Lambda _ { k , h } ^ { - 1 } \sum _ { j } u _ { j } f ( s _ { j } ^ { \prime } ) ^ { \top } \in \mathbb { R } ^ { d \times m } .
$$

Then, for every queried pair $( s , a )$

$$
\widehat { P } _ { k , h } f ( s , a ) = \phi _ { h } ( s , a ) ^ { \top } \widehat { w } _ { h } ( f ) .
$$

Below, we show that we only need to keep track of the vector $\widehat w _ { h } ( V _ { h + 1 } ^ { \mathrm { p l } } )$ , or equivalently $\vartheta _ { h } = \overline { { c } } _ { h } + \widehat { w } _ { h } ( V _ { h + 1 } ^ { \mathrm { p l } } )$ during the execution of Algorithm 3. They are computed by backward induction.

Suppose $\vartheta _ { h }$ has already been computed. For every state $s _ { j } ^ { \prime }$ appearing in $\mathcal { D } _ { k , h - 1 }$ , query $\phi _ { h } ( s _ { j } ^ { \prime } , a )$ for each $a \in { \mathcal { A } }$ and compute

$$
\begin{array} { r } { Q _ { h } ^ { \mathrm { p l } } ( s _ { j } ^ { \prime } , a ) = \mathrm { C L I P } _ { [ - \sqrt { H } , \sqrt { H } ] } \bigg ( \phi _ { h } ( s _ { j } ^ { \prime } , a ) ^ { \top } \vartheta _ { h } - ( \sqrt { H } + 1 ) \beta \left. \phi _ { h } ( s _ { j } ^ { \prime } , a ) \right. _ { \Lambda _ { k , h } ^ { - 1 } } \bigg ) , } \end{array}
$$

$$
V _ { h } ^ { \mathrm { p l } } ( s _ { j } ^ { \prime } ) = - \tau \log \sum _ { a \in \mathcal { A } } \exp \Bigl ( - Q _ { h } ^ { \mathrm { p l } } ( s _ { j } ^ { \prime } , a ) / \tau \Bigr ) .
$$

These are the values needed to form $\widehat { w } _ { h - 1 } ( V _ { h } ^ { \mathrm { p l } } )$ and hence

$$
\begin{array} { r } { \vartheta _ { h - 1 } = \bar { c } _ { h - 1 } + \widehat { w } _ { h - 1 } ( V _ { h } ^ { \mathrm { p l } } ) . } \end{array}
$$

Thus, $\vartheta _ { h }$ can be computed in a backward induction manner. This then allows us to evaluate $Q _ { h } ^ { \mathrm { p l } } ( s , a )$ and $V _ { h } ^ { \mathrm { p l } } ( s )$ efficiently for any queried $( s , a )$