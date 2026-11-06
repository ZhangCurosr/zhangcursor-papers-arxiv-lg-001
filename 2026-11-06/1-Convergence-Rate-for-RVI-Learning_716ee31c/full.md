# �(1/�) Convergence Rate for RVI �-Learning

Avik Kar<sup>1</sup>, Siddharth Chandak<sup>2</sup>, Rahul Singh<sup>3</sup>, and Shalabh Bhatnagar<sup>1</sup>

<sup>1</sup>Department of Computer Science and Automation, Indian Institute of Science, Bengaluru, India. avikkar@iisc.ac.in, shalabh@iisc.ac.in

<sup>2</sup>Department of Electrical Engineering, Stanford University, Stanford, California, USA. chandaks@stanford.edu <sup>3</sup>EPITA, Laboratoire Recherche de l’EPITA, Paris, France. rahulsingh0188@gmail.com

## Abstract

We establish finite-time convergence guarantee for synchronous relative value iteration (RVI) �-learning for finite weakly communicating Markov decision processes (MDPs). For stochastic Krasnosel’ski˘ı–Mann iterations driven by a non-expansive, linearly regular map on a finite-dimensional normed space, we establish last-iterate mean-squared residual bounds under martingale-diference noise whose conditional second moment grows at most linearly with the squared residual. Suitable polynomial stepsizes of order $n ^ { - a }$ yield an $O ( n ^ { - a } )$ rate for 0 < � ≤ 1. We verify linear regularity and the noise condition for the centered Bellman operator for finite weakly communicating MDPs. Applying the general result gives an $O ( n ^ { - 1 } )$ bound on the mean-squared Bellman span residual of RVI �-learning iterates when a suitable stepsize sequence is used. The positive gap between optimal and suboptimal stationary deterministic policies then yields an $O ( n ^ { - 1 } )$ bound on the expected suboptimality of the greedy policy. Consequently, $O ( \epsilon ^ { - 1 } )$ synchronous iterations sufice for an �-optimal policy in expectation. These guarantees are instance-dependent, with constants that depend on the hitting-time parameters and the suboptimality gap of the second-best policy.

Keywords: average-reward Markov decision processes; relative value iteration �-learning; stochastic approximation; finite-time bounds.

## 1 Introduction

Average-reward reinforcement learning (RL), in which the learning agent aims to maximize the long-run average reward, provides a natural model for sequential decision-making problems in systems where steadystate performance is crucial, and there is no prescribed terminal time, such as queueing networks, inventory control, robotics, power-grid management, and communication networks [16]. In contrast, discounted RL assigns geometrically decaying weights to future rewards and therefore does not directly capture the long-run steady-state performance of a policy. Discounting, however, ofers a key analytical advantage. The value function is the unique fixed point of the Bellman operator, which is a contraction [16, Chapter 6]. Similar properties hold in undiscounted problems when the state process reaches a terminal state almost surely under every policy [4]. This contractive structure underlies much of the finite-time theory for value-based RL algorithms such as �-learning and TD-learning [7, 9, 15, 18, 19]. Average-reward RL is inherently more challenging from this perspective. In the absence of discounting, the Bellman operator and related maps, such as the RVI map, are generally only non-expansive. Moreover, solutions to the average-reward optimality equation (1), i.e., the fixed points of the centered Bellman operator (33), are never unique. For unichain MDPs, the solution set has one degree of freedom [16, Chapter 8], whereas for the weakly communicating MDPs studied here, the solution set is, in general, higher dimensional [17, 21].

RVI �-learning, introduced by Abounadi et al. [1], is one of the earliest provably convergent model-free value-based algorithms for average-reward RL. It updates the estimate of the optimal �-function with a sampled Bellman step after subtracting an iterate-dependent normalization term from it (5). The normalization term removes the undetermined additive component of the �-function and stabilizes the iterates. Under certain regularity conditions on the normalization functional and standard choices of the stepsize sequence, Abounadi et al. [1] established almost-sure convergence of both synchronous and asynchronous RVI �-iterates for unichain MDPs with a common state recurrent under every stationary policy. Their result is asymptotic and does not quantify the number of iterations required for obtaining a $Q \cdot$ -iterate that nearly solves the average reward optimality equation (1), or for obtaining a nearly optimal policy. The goal of the current work is to provide such a finite-time characterization for synchronous RVI �-learning by studying the fixed-point residual, which upper bounds the suboptimality of the greedy policy induced by the current �-iterate.

Due to the non-expansive nature of the relevant operators, the finite-time analysis available for contractive settings cannot be applied here. Although the average-reward Bellman operator is non-expansive under the sup-norm, the RVI map is non-expansive only under the span seminorm. A standard approach is to analyze an auxiliary iteration driven by the centered Bellman operator, in which the constant optimal average reward replaces the iterate-dependent normalization term in RVI �-learning. The operator is non-expansive under the sup-norm, and the iterates, when run with the same samples, have the same Bellman span residuals as the RVI iterates. Hence, obtaining finite-time guarantees for RVI �-learning reduces to analyzing stochastic Krasnosel’ski˘ı–Mann (KM) iterations with non-expansive operators under an arbitrary norm.

The following two lines of work address finite-time guarantees for average-reward �-learning. Analyses via the theory of stochastic KM iterations assume nothing beyond non-expansiveness. Bravo and Cominetti [6] obtain an expected residual of order $\widetilde { O } ( n ^ { - 1 / 6 } )$ after � iterations for uniformly bounded noise variance, where $\widetilde { O }$ hides logarithmic factors. This rate degrades to $O ( n ^ { - 1 / 1 0 + \epsilon } )$ , for any $\epsilon \in ( 0 , 1 / 1 0 )$ , once the iterate-dependence of the RVI sampling noise is accounted for. On the other hand, a body of work assumes a contraction property and obtains the faster rate $O ( n ^ { - 1 / 2 } )$ . Chen et al. [10] assume a seminorm contraction and obtain $O ( n ^ { \bar { - } 1 / 2 } )$ ; Chen [8] obtain last-iterate guarantees under a span-contraction assumption with adaptive stepsizes; and Chen et al. [11] construct an instance-dependent seminorm after a lazy transformation of the dynamics and obtain an $\widetilde { O } ( \epsilon ^ { - 2 } )$ sample complexity under a reachability condition.

Instead of exploiting only the non-expansive property of the RVI �-learning operator, or imposing any contractive structure, the current work shows that a third property, called linear regularity [2], holds for the centered Bellman operator if the MDP is weakly communicating.

Definition 1 (Linearly regular operator). Let $( X , \| \cdot \| )$ be a normed space, and let $T : X  X$ be an operator with a non-empty set of fixed points $\operatorname { F i x } ( T ) : = \{ x \in X : T x = x \}$ . � is linearly regular if there exists a constant $\kappa > 0$ such that

$$
\operatorname* { i n f } _ { x ^ { \star } \in \mathrm { F i x } ( T ) } \left\| x - x ^ { \star } \right\| \leq \kappa \left\| x - T x \right\| , \forall x \in \mathcal { X } .
$$

Further, we show that the conditional second moment of the RVI sampling noise is at most a constant times one plus the squared residual of the current iterate. Using these two properties of the operator and the noise, the current work shows that the mean-squared residuals of the stochastic KM iterates decay at a rate of $O ( n ^ { - 1 } )$ . The same rate holds for the expected suboptimality of the policy greedy with respect to the RVI �-iterates.

Contributions. Our main results are summarized in the following three points.

(i) For the stochastic KM iteration $x _ { n } = ( 1 - \alpha _ { n } ) x _ { n - 1 } + \alpha _ { n } ( T x _ { n - 1 } + M _ { n } )$ driven by a non-expansive, linearly regular map $T$ with linear regularity constant � on an arbitrary finite-dimensional normed space, with stepsizes $\alpha _ { n }$ and martingale-diference noise $M _ { n }$ whose conditional second moment is bounded by $\sigma ^ { 2 } ( 1 + \| x _ { n - 1 } - T x _ { n - 1 } \| ^ { 2 } )$ , we prove a last-iterate bound on the mean-squared residual (Theorem 2). Polynomial stepsizes $c ( n + K _ { 0 } ) ^ { - a }$ with a fixed $a \in ( 0 , 1 ]$ attain the rate of $O ( n ^ { - a } )$

(ii) We prove that the centered Bellman operator $\bar { H } ( 3 3 )$ for every finite weakly communicating MDP is linearly regular with the explicit constant �, defined in (8) (Proposition 1). We also prove that the conditional second moment of its sampling noise is bounded by a constant times one plus the squared residual (Proposition 2).

(iii) Combining (i) and (ii), we obtain an $O ( n ^ { - 1 } )$ bound on the mean-squared Bellman span residual of the RVI �-iterates, with a prefactor of order $K ^ { 6 } ( 1 + D ) ^ { 2 } \log ( 2 | S | | \mathcal { A } | )$ (Theorem 1). The residual-to-policy relation (7) gives the same rate for expected suboptimality of the greedy policy, with the additional factor $1 / \Delta _ { \mathrm { p o l } }$ in its prefactor. Thus, $O ( \epsilon ^ { - 1 } )$ synchronous iterations sufice for the greedy policy to be �-optimal in expectation (Corollary 1).

Overview of the argument. To prove (i), we first consider the iteration without sampling noise. Because of linear regularity, over a suficiently large number of iterations, called a block, the distance to the nearest fixed point decreases by a fraction. We then compare the noisy and noiseless iterations starting from the same point. The comparison shows a similar decrease in the mean-squared distance, with an additional error proportional to the step size at the start of the block. In the next block, the accumulated error is reduced again, while new noise contributes another error term. The stepsize conditions ensure that this repeated reduction suppresses the errors introduced in earlier blocks. Consequently, the mean-squared distance, and hence the mean-squared residual, is bounded by a constant times the current stepsize plus a decaying contribution from the initial distance.

For (ii), we first obtain a preliminary bound on the distance to the solution set in terms of the residual, along with an additive constant. Whenever the residual is above a strictly positive threshold, the additive constant can be absorbed into the residual term at the expense of a larger constant. We set the threshold to be the policy gap, which is the smallest non-zero loss in average reward among stationary deterministic policies. Wherever the residual is smaller than the policy gap, the corresponding greedy policy must be optimal. We use this property to construct a nearby solution. Given a �-function, let the value of a state be the largest �-value over actions at that state. We show that these values can be raised by an amount proportional to the residual so that subsequent Bellman updates, with the optimal average reward subtracted, only decrease them. The updated values remain bounded below, so they converge to an exact state-value solution. To bound its distance from the original values, we use the closed groups of states in which the greedy policy eventually remains, called its recurrent classes. On these classes, optimality forces the Bellman updates to leave the raised values unchanged. For other states, summing Bellman errors until one of these classes is reached bounds how far the limiting values lie below the original values. The expected hitting times make this bound proportional to the residual. Together with the bound on the initial upward adjustment, this shows that the limiting state values difer from the original values by an amount proportional to the residual. Converting this state-value solution back to a �-function gives the nearby exact solution needed for linear regularity. Separately, we bound the spread of state values across states in terms of the residual. This controls the sampling noise, which comes from the bounded reward and the value of the sampled next state.

For (iii), we run the auxiliary and RVI iterations using the same sampled transitions. Their iterates difer only by a constant added to every entry, which leaves the Bellman span residual unchanged. The bound from (i) therefore gives a mean-squared residual bound for RVI. For each sample outcome, the greedy policy’s loss is at most the Bellman span residual and is either zero or at least the policy gap. These two facts bound the loss by the squared residual divided by the gap. Taking expectations gives the claimed $O ( n ^ { - 1 } )$ rate for the stated stepsizes of order $1 / n$ , with constants depending on the fixed MDP.

Related work. Abounadi et al. [1] establish convergence of RVI �-learning for unichain MDPs with a common state recurrent under every stationary policy, and Wan et al. [21] extend this result to weakly communicating MDPs. Wan et al. [20] introduce diferential $Q { \mathrm { - l e a r n i n g } } ,$ which avoids a normalization functional, and prove its asymptotic convergence. For finite-time guarantees, Zhang et al. [22] analyze tabular average-reward �-learning under span-seminorm contraction. Chen et al. [10] develop a seminorm Lyapunov framework yielding an $O ( n ^ { - 1 / 2 } )$ ) expected span-seminorm error bound. Chen [8] obtain asynchronous last-iterate guarantees with adaptive stepsizes under span contraction. Chen et al. [11] construct an instancedependent seminorm after a lazy transformation and obtain $\widetilde { O } ( \epsilon ^ { - 2 } )$ sample complexity under a reachability condition. For weakly communicating MDPs with a generative model, Zurek and Chen [23] use a discounted reduction to obtain the minimax-optimal sample complexity $\widetilde { \cal O } ( | S | | \mathcal { A } | \mathsf { H } / \epsilon ^ { 2 } )$ , where H is the span of the optimal value function. Our $O ( \epsilon ^ { - 1 } )$ iteration bound has instance-dependent constants, so it does not imply an improvement over this minimax benchmark.

Bravo and Cominetti [6] analyze stochastic KM iterations on arbitrary normed spaces, obtaining an expected residual rate of $\widetilde { O } ( n ^ { - 1 / 6 } )$ under uniformly bounded noise variance. For RVI �-learning, iteratedependent noise yields the slower rate $O ( n ^ { - 1 / 1 0 + \eta } )$ for any $\eta \in ( 0 , 1 / 1 0 )$ . Blaser and Zhang [5] study nonexpansive stochastic approximation with Markovian noise and apply their results to average-reward TD learning. Our analysis instead exploits linear regularity and residual-dependent martingale noise to obtain an $O ( n ^ { - 1 } )$ bound on the mean-squared residual.

Alternative algorithms include the adaptive-discount �-learning method of Jin et al. [13], which provides finite-sample guarantees without prior knowledge of problem parameters. Lee et al. [14] combine Halpern’s anchored iteration with recursive sampling to obtain an �-optimal policy using $\widetilde { \cal O } ( | S | | \mathcal { A } | \mathsf { H } ^ { 2 } / \epsilon ^ { 2 } )$ samples without requiring prior knowledge of H.

Organization. Section 2 introduces the problem and algorithm and states the main results. Section 3 develops a general convergence bound for stochastic KM iterations. Section 4 applies this bound to RVI �-learning, establishes the required linear regularity and noise growth condition, and proves the convergence and sample complexity guarantees. Section 5 concludes. Appendices $\mathrm { A } { - } \mathrm { C }$ contain supporting lemmas, a martingale inequality, and stepsize estimates, respectively.

## 2 Setup and Main Results

## 2.1 Average-Reward MDPs

Notation. For a finite dimensional vector �, $\begin{array} { r } { \| { \boldsymbol x } \| _ { \mathrm { s p } } : = \operatorname* { m a x } _ { i } { \boldsymbol x } ( i ) - \operatorname* { m i n } _ { i } { \boldsymbol x } ( i ) } \end{array}$ denotes the span seminorm. For a finite set $\chi , \Delta \chi$ is the probability simplex on $\chi$

Let $( S , { \mathcal { A } } , p , r )$ be an MDP, where S and A are finite state and action spaces, respectively, $p ( s , a , \cdot ) \in \Delta _ { S }$ governs the state transitions and $r : S \times \mathcal { A } \times S  [ 0 , 1 ]$ is the reward function. For a stationary policy $\pi : S \to \Delta _ { \mathcal { A } }$ , we write $\pi ( a | s )$ for the probability that � chooses action � in state �. Its average reward starting from state � is

$$
J ( \pi , s ) : = \operatorname* { l i m i n f } _ { n \to \infty } { \frac { 1 } { n } } \mathbb { E } _ { s } ^ { \pi } \left[ \sum _ { k = 0 } ^ { n - 1 } r ( S _ { k } , A _ { k } , S _ { k + 1 } ) \right] ,
$$

where $S _ { k }$ and $A _ { k }$ denote the state and the action at time $k ,$ and $\mathbb { E } _ { s } ^ { \pi }$ denotes expectation under the conditions that $S _ { 0 } = s$ and $A _ { k } \sim \pi ( \cdot | S _ { k } )$ for every $k \geq 0$ . The MDP satisfies the following assumption.

Assumption 1 (Weakly communicating). The state space admits a partition $S = S _ { \mathrm { c } } \cup S _ { \mathrm { t } }$ , where $S _ { \mathrm { c } }$ is closed, $\begin{array} { r } { i . e . , \sum _ { s ^ { \prime } \in S _ { \mathrm { c } } } p ( s , a , s ^ { \prime } ) = 1 f o r e \nu e r y s \in S _ { \mathrm { c } } } \end{array}$ and $a \in { \mathcal { A } }$ , every state of $S _ { \mathrm { c } }$ is accessible from every other state of $m _ { \mathrm { { c } } }$ under some stationary deterministic policy, and every state of the possibly empty set $S _ { \mathrm { t } }$ is transient under every stationary policy.

The class of weakly communicating MDPs contains the families of ergodic, unichain, and communicating MDPs; see Puterman [16, Section 8.3]. Under the above assumption, the optimal average reward su ${ } _ { 2 \pi } J ( \pi , s )$ is independent of the initial state � and we denote the scalar value by $J ^ { \star }$ . The average-reward optimality equation (AROE) is

$$
J + { \cal Q } ( s , a ) = \sum _ { s ^ { \prime } \in S } p ( s , a , s ^ { \prime } ) \big ( r ( s , a , s ^ { \prime } ) + V _ { Q } ( s ^ { \prime } ) \big ) , \quad \forall ( s , a ) ,\tag{1}
$$

where $V _ { Q } ( s ) : = \mathrm { m a x } _ { a \in \mathcal { A } } Q ( s , a )$ . Assumption 1 is suficient for the existence of $Q ^ { \star } \in \mathbb { R } ^ { S \times \mathcal { A } }$ such that $( J ^ { \star } , Q ^ { \star } )$ satisfies the AROE. In that case, $( J ^ { \star } , Q ^ { \star } + c { \bf 1 } )$ also satisfies the AROE for every $c \in \mathbb { R }$ , where 1 denotes the $\boldsymbol { s } \times \mathcal { A }$ -dimensional all-ones vector. These shifts need not exhaust the solution set. The AROE of a weakly communicating MDP may admit solutions that do not difer by a constant vector [17, 21]. A policy that chooses an action from arg $\operatorname* { m a x } _ { a } { Q ^ { \star } ( s , a ) }$ in each state � is optimal. See Puterman [16] for more details on average reward MDPs.

Useful MDP parameters. Let $\Pi _ { \mathrm { S D } }$ be the set of stationary deterministic policies, and let $\tau _ { A } : = \operatorname* { i n f } \{ k \geq 0$ $S _ { k } \in A \}$ and $\tau _ { s ^ { \prime } } : = \tau _ { \{ s ^ { \prime } \} }$ . Define the following complexity measure of weakly communicating MDPs,

$$
D : = \operatorname* { m a x } \left\{ \operatorname* { m a x } _ { \substack { s \neq s ^ { \prime } \in S _ { \mathrm { c } } } } \operatorname* { m i n } _ { \pi \in \Pi _ { \mathrm { S D } } } \mathbb { E } _ { s } ^ { \pi } \big [ \tau _ { s ^ { \prime } } \big ] , \ \operatorname* { m a x } _ { \pi \in \Pi _ { \mathrm { S D } } } \operatorname* { m a x } _ { s \in S _ { \mathrm { t } } } \mathbb { E } _ { s } ^ { \pi } \big [ \tau _ { S _ { \mathrm { c } } } \big ] \right\} ,\tag{2}
$$

with the convention that the maximum of an empty set is 0. The first term is the diameter of the closed communicating set, and the second is the worst-case expected time to reach the closed communicating set from any transient state. The policy gap $\Delta _ { \mathrm { p o l } }$ is the smallest non-zero sub-optimality of stationary deterministic policies,

$$
\Delta _ { \mathrm { p o l } } : = \operatorname* { m i n } _ { \substack { \pi \in \Pi _ { \mathrm { S D } } : J ^ { \star } > J ( \pi , s ) \mathrm { ~ f o r ~ s o m e } s } } \operatorname* { m a x } _ { s ^ { \prime } \in S } \left( J ^ { \star } - J ( \pi , s ^ { \prime } ) \right) ,\tag{3}
$$

with $\Delta _ { \mathrm { p o l } } : = 1$ if every $\pi \in \Pi _ { \mathrm { S D } }$ is optimal from every state. Since the policy set is finite and rewards lie in $[ 0 , 1 ]$ , we have $0 < \Delta _ { \mathrm { p o l } } \leq 1$ . Thus, a stationary deterministic policy whose sub-optimality is less than $\Delta _ { \mathrm { p o l } }$ from every state is optimal from every state. Let $\mathbb { P } _ { s } ^ { \pi }$ denote the probability conditioned by $S _ { 0 } = s$ and $A _ { k } \sim \pi ( \cdot | S _ { k } )$ for every $k \geq 0$ . Define

$$
\mathcal T : = \underset { \boldsymbol \pi \in \Pi _ { \mathrm { S D } } , \boldsymbol s \in \boldsymbol S , \ \boldsymbol \varnothing \neq \boldsymbol C \boldsymbol \subseteq \boldsymbol S } { \mathrm { m a x } } \mathbb { E } _ { \boldsymbol s } ^ { \pi } [ \tau _ { \boldsymbol C } ] .\tag{4}
$$

Thus, $\mathcal { T }$ is the largest expected hitting time among stationary deterministic policies that are optimal from every state, initial states, and target sets for which the target is reached almost surely. This maximum is finite. Indeed, i $\mathrm { ~ f ~ } \mathbb { P } _ { s } ^ { \pi } ( \tau _ { C } < \infty ) = 1$ , then every state reachable from � before hitting � has a path to � of at most |S transitions with positive probability under policy �. Hence, conditionally on not yet hitting �, the probability of reaching � within the next $| S |$ steps is bounded below by a positive number, so $\tau _ { C }$ has a geometric tail and a finite expectation. $\mathcal { T }$ is finite because there are only finitely many triplets $( \pi , s , C )$ .

## 2.2 Synchronous RVI �-Learning

Abounadi et al. [1] introduced the synchronous and asynchronous RVI �-learning algorithms for learning a solution to the AROE. Synchronous RVI �-learning updates the estimate of $Q ^ { \star }$ as follows: for all

(�, �) ∈ S × A,

$$
Q _ { n + 1 } ( s , a ) = Q _ { n } ( s , a ) + \alpha _ { n + 1 } \left( r ( s , a , Y _ { s , a } ^ { n + 1 } ) + V _ { Q _ { n } } ( Y _ { s , a } ^ { n + 1 } ) - Q _ { n } ( s , a ) - f ( Q _ { n } ) \right) ,\tag{5}
$$

where, for each $n \geq 0$ , the random variables $\{ Y _ { s , a } ^ { n + 1 } \} _ { ( s , a ) }$ are independent, with $Y _ { s , a } ^ { n + 1 } \sim p ( s , a , \cdot )$ , and are independent of $Y _ { s , a } ^ { m } , m \leq n , ( s , a ) \in S \times \mathcal { A } . \mathcal { F } _ { n } : = \sigma ( Q _ { 0 } , Y _ { s , a } ^ { m } : 1 \leq m \leq n , ( s , a ) \in S \times \mathcal { A } )$ denotes the natural filtration associated with the iterates. $\{ \alpha _ { n } \} _ { n \geq 1 }$ denotes the stepsize sequence, and $f : \mathbb { R } ^ { S \times \mathcal { A } } $ R is a continuous normalization functional. A few candidate �’s are $\begin{array} { r } { f ( Q ) = \operatorname* { m i n } _ { s , a } Q ( s , a ) , f ( Q ) = \operatorname* { m a x } _ { s , a } Q ( s , a ) } \end{array}$ $f ( Q ) = Q ( s _ { 0 } , a _ { 0 } )$ for some fixed $( s _ { 0 } , a _ { 0 } ) \in S \times \mathcal { A }$ , and $\begin{array} { r } { f ( Q ) = \frac { 1 } { | S | | \mathcal { R } | } \sum _ { s , a } Q ( s , a ) } \end{array}$

Abounadi et al. [1] showed that for unichain MDPs, $( f ( Q _ { n } ) , \ddot { Q } _ { n } ) , n \ge 0$ , almost surely converges to a unique solution of (1), $( J ^ { \star } , Q ^ { \star } )$ such that $f ( Q ^ { \star } ) = J ^ { \star }$ if the functional � satisfies certain regularity conditions and if the stepsizes satisfy

$$
\sum _ { k \geq 1 } \alpha _ { k } = \infty , \qquad \sum _ { k \geq 1 } \alpha _ { k } ^ { 2 } < \infty .
$$

Define the Bellman operator $\boldsymbol { B } : \mathbb { R } ^ { S \times \mathcal { A } }  \mathbb { R } ^ { S \times \mathcal { A } }$ as

$$
( B Q ) ( s , a ) : = \sum _ { s ^ { \prime } \in S } p ( s , a , s ^ { \prime } ) \left( r ( s , a , s ^ { \prime } ) + V _ { Q } ( s ^ { \prime } ) \right) ,\tag{6}
$$

write $\begin{array} { r } { \overline { { r } } ( s , a ) : = \sum _ { s ^ { \prime } \in S } p ( s , a , s ^ { \prime } ) r ( s , a , s ^ { \prime } ) } \end{array}$ for the expected one-step reward, and define the RVI map � such that $H Q : = B Q - f ( Q ) \mathbf { 1 }$ . The synchronous RVI �-learning update (5) can alternatively be written as

$$
Q _ { n + 1 } = Q _ { n } + \alpha _ { n + 1 } \left( H Q _ { n } - Q _ { n } + M _ { n + 1 } ( Q _ { n } ) \right) ,
$$

where

$$
M _ { n + 1 } ( Q ) : = \widehat B _ { n + 1 } Q - B Q , \quad \mathrm { w i t h } \quad ( \widehat B _ { n + 1 } Q ) ( s , a ) : = r ( s , a , Y _ { s , a } ^ { n + 1 } ) + V _ { Q } ( Y _ { s , a } ^ { n + 1 } ) .
$$

Note that a solution $Q ^ { \star }$ of the AROE with $f ( Q ^ { \star } ) = J ^ { \star }$ is a fixed point of �.

The goal is to learn an optimal or near-optimal policy. Fix a deterministic tie-breaking rule, and let $\pi _ { Q }$ be the resulting policy greedy with respect to �. We measure performance by the suboptimality of the greedy policy,

$$
\Delta _ { n } : = \operatorname* { m a x } _ { s \in S } \left\{ J ^ { \star } - J ( \pi _ { Q _ { n } } , s ) \right\} .
$$

For every $\begin{array} { r } { Q \in \mathbb { R } ^ { S \times \mathcal { A } } , 0 \leq \operatorname* { m a x } _ { s \in \mathcal { S } } \left\{ J ^ { \star } - J ( \pi _ { Q } , s ) \right\} \leq \| Q - B Q \| _ { \mathrm { s p } } } \end{array}$ . See Lee et al. [14, Proposition 2.1] for a proof. Moreover, $H Q - B Q = - f ( Q ) \mathbf { 1 }$ is a constant vector, so $\left\| Q - B Q \right\| _ { \mathrm { s p } } = \left\| Q - H Q \right\| _ { \mathrm { s p } }$ . Thus, $\Delta _ { n } \leq \| Q _ { n } - H Q _ { n } \| _ { \mathrm { s p } }$ . However, since either $\Delta _ { n } \geq \Delta _ { \mathrm { p o l } }$ or $\Delta _ { n } = 0 , \Delta _ { n } \leq \Delta _ { n } ^ { 2 } / \Delta _ { \mathrm { p o l } }$ . Thus,

$$
\Delta _ { n } \leq \frac { 1 } { \Delta _ { \mathrm { p o l } } } \left. Q _ { n } - H Q _ { n } \right. _ { \mathrm { s p } } ^ { 2 } .\tag{7}
$$

We bound the latter among the above two upperbounds of $\Delta _ { n }$ in order to bound $\Delta _ { n }$

## 2.3 Convergence Rates and Sample Complexity

We now state the convergence rates and the sample complexity of synchronous RVI �-learning. Recall �, $\Delta _ { \mathrm { p o l } }$ and T from (2)-(4). Set

$$
K : = \operatorname* { m a x } \left\{ 1 + \mathcal { T } , \ : \left( 1 + 3 D \right) \left( 1 + \frac { 1 } { \Delta _ { \mathrm { p o l } } } \right) \right\} .\tag{8}
$$

Theorem 1. Let Assumption 1 hold, let $| S | | { \mathcal { A } } | \geq 2 ,$ , and let $\{ Q _ { n } \}$ be generated by (5)from afixed $Q _ { 0 }$ with the sampling model ofSection 2.2 and stepsizes $\alpha _ { n } = c ( n + K _ { 0 } ) ^ { - a }$ , where $0 < a \leq 1 , c > 0 ,$ , and the integer $K _ { 0 } \ge 1$ satisfies

$$
c K _ { 0 } ^ { 1 - a } \geq 2 ^ { 8 } K ^ { 2 } , \qquad \alpha _ { 1 } \leq \frac { 1 } { \operatorname* { m a x } \big \{ 2 , 2 ^ { 6 } C _ { 1 } \big \} } ,\tag{9}
$$

where $C _ { 1 } : = 2 ^ { 8 } e ( 1 + 3 D ) ^ { 2 } ( 2 ^ { 6 } K ^ { 2 } + 1 ) ^ { 2 } \log ( 2 | S | | \mathcal { A } | )$ . Then, for every $n \geq 0$

$$
\mathbb { E } \Vert Q _ { n } - H Q _ { n } \Vert _ { \mathrm { s p } } ^ { 2 } \leq 1 0 0 \mathcal { E } ( Q _ { 0 } ) ^ { 2 } \exp \left( - \frac { \log ( 4 ) } { 2 ^ { 7 } K ^ { 2 } + \frac { 1 } { 2 } } \sum _ { k = 1 } ^ { n } \alpha _ { k } \right) + \frac { 4 5 C _ { 1 } c } { ( n + K _ { 0 } + 1 ) ^ { \alpha } } ,
$$

where $\begin{array} { r } { \mathcal { E } ( Q _ { 0 } ) = \operatorname* { i n f } _ { q \in \mathrm { F i x } ( \bar { H } ) } \| Q _ { 0 } - q \| _ { \infty } . \ B y \left( 7 \right) . } \end{array}$ , for every $n \geq 0 _ { ; }$

$$
\mathbb { E } \left[ \Delta _ { n } \right] \leq \frac { 1 0 0 \mathscr { E } ( Q _ { 0 } ) ^ { 2 } } { \Delta _ { \mathrm { p o l } } } \exp \left( - \frac { \log ( 4 ) } { 2 ^ { 7 } K ^ { 2 } + \frac { 1 } { 2 } } \sum _ { k = 1 } ^ { n } \alpha _ { k } \right) + \frac { 4 5 C _ { 1 } c } { \Delta _ { \mathrm { p o l } } ( n + K _ { 0 } + 1 ) ^ { a } } .\tag{10}
$$

In particular:

(i) $H a \in ( 0 , 1 )$ , then (9) holds for every suficiently large $K _ { 0 } ,$ , and E $[ \Delta _ { n } ] \leq C n ^ { - a }$ for every $n \geq 1$ , where � depends only on the MDP, $Q _ { 0 } , a , c ,$ , and $K _ { 0 }$

(ii) $H a = 1$ and $c = 2 ^ { 8 } K ^ { 2 }$ , then, with $p : = 2 ^ { 8 } K ^ { 2 } \log ( 4 ) / ( 2 ^ { 7 } K ^ { 2 } + 1 / 2 ) > 1$

$$
\mathbb { E } \left[ \Delta _ { n } \right] \leq \frac { 1 0 0 \mathcal { E } ( Q _ { 0 } ) ^ { 2 } } { \Delta _ { \mathrm { p o l } } } \left( \frac { K _ { 0 } + 1 } { n + K _ { 0 } + 1 } \right) ^ { p } + \frac { 4 5 \cdot 2 ^ { 1 6 } e K ^ { 2 } ( 2 ^ { 6 } K ^ { 2 } + 1 ) ^ { 2 } ( 1 + 3 D ) ^ { 2 } \log ( 2 | S | | \mathcal { A } | ) } { \Delta _ { \mathrm { p o l } } ( n + K _ { 0 } + 1 ) } .\tag{11}
$$

Theorem 1 improves on the $O ( n ^ { - 1 / 1 0 + \eta } )$ expected residual rate obtained for this algorithm by applying the iterate-dependent noise bound of Bravo and Cominetti [6], for any $\eta \in ( 0 , 1 / 1 0 )$ . For $a = 1$ , we obtain a convergence rate of $O ( n ^ { - 1 } )$ for each fixed MDP, with a prefactor of order $K ^ { 6 } ( 1 + D ) ^ { 2 } \log ( 2 | S | | \mathcal { A } | ) / \Delta _ { \mathrm { p o l } }$ The constants � and $\Delta _ { \mathrm { p o l } }$ make the dependence on the MDP explicit. The proof is given in Section 4. The following corollary converts the rate into a sample complexity result.

Corollary 1 (Sample complexity). Under the hypotheses of Theorem 1(ii), let $K _ { 0 }$ be the smallest integer satisfying (9), so that $K _ { 0 } = O ( K ^ { 2 } + K ^ { 6 } ( 1 + D ) ^ { 2 } \log ( 2 | S | | \mathcal { A } | ) )$ . There is a universal constant � such that, for every $\epsilon > 0 ,$ the greedy policy $\pi _ { Q _ { n } }$ satisfies E $[ \Delta _ { n } ] \leq \epsilon$ whenever

$$
n \geq C \left( \frac { K ^ { 6 } ( 1 + D ) ^ { 2 } \log ( 2 | S | | \mathcal { A } | ) } { \Delta _ { \mathrm { p o l } } \epsilon } + K _ { 0 } \left( 1 + \frac { \mathcal { E } ( Q _ { 0 } ) ^ { 2 } } { \Delta _ { \mathrm { p o l } } \epsilon } \right) \right) ,
$$

where $\begin{array} { r } { \mathcal { E } ( Q _ { 0 } ) \leq \frac { 1 } { 2 } ( \| Q _ { 0 } \| _ { \mathrm { s p } } + 1 + 3 D ) } \end{array}$

## 3 Stochastic KM Iterations for Linearly Regular Maps

This section studies stochastic KM iterations under two hypotheses: linear regularity and a residual-dependent growth of the noise. Throughout, $( X , \| \cdot \| )$ is a finite-dimensional real normed vector space and � is a non-expansive map on it. We define its fixed-point set, residual, and fixed-point distance below. Section 4 verifies the hypotheses for the centered Bellman operator by taking $\chi = \mathbb { R } ^ { S \times \bar { \mathcal { A } } }$ with the sup-norm and $T = { \bar { H } }$

## 3.1 Setup and Assumptions

Let $( \chi , \lVert \cdot \rVert )$ be a �-dimensional real normed vector space. Throughout this section, we consider the operator $T : X  X$ that satisfies the following assumption.

Assumption 2 (Non-expansive and linearly regular). $( i ) T : X  X$ be non-expansive, that is, $\| T x - T y \| \leq$ $\| x - y \| f o r a l l y , y \in X$

(ii) The set of fixed points

$$
\operatorname { F i x } ( T ) = \{ x \in X : T x = x \}
$$

is non-empty and � is linearly regular (Definition 1), that is, there exists $\kappa \in ( 0 , \infty )$ such that

$$
\operatorname* { i n f } _ { x ^ { \star } \in \mathrm { F i x } ( T ) } \left\| x - x ^ { \star } \right\| \leq \kappa \left\| x - T x \right\| , \quad \forall x \in \pmb { \chi } .
$$

For convenience, we denote $\mathcal { E } ( x ) : = \mathrm { i n f } _ { x ^ { \star } \in \mathrm { F i x } ( T ) } \| x - x ^ { \star } \|$ . Fix(�) is closed, since � is continuous, but it need not be either convex or a singleton. The reverse inequality of linear regularity is always true by non-expansiveness. Fix $x ^ { \star } \in \operatorname { F i x } ( T )$ arbitrarily. For $x \in { \mathcal { X } }$ , non-expansiveness gives $\| x - T x \| \leq$ $\left\| T x - T x ^ { \star } \right\| + \left\| x ^ { \star } - x \right\| \leq 2 \left\| x - x ^ { \star } \right\|$ . Taking the infimum over $x ^ { \star }$

$$
\| x - T x \| \leq 2 \mathcal { E } ( x ) , \qquad x \in X .\tag{12}
$$

Linear regularity is a weaker property than contraction. To see this, suppose that � is a �-contraction, $\gamma \in ( 0 , 1 )$ , with unique fixed point $x ^ { \star }$ . Then, $\mathcal { E } ( x ) = \| x - x ^ { \star } \| \leq \| x - T x \| + \| T x - T x ^ { \star } \| \leq \| x - T x \| + \gamma \| x - x ^ { \star } \|$ Thus, every contractive operator is linearly regular with $\begin{array} { r } { \kappa = \frac { 1 } { 1 - \gamma } } \end{array}$

However, not all linearly regular operators are contractive. Consider the reflection operator $T ( x ) = - x$ on $\chi = \mathbb { R }$ . Since $| T x - T y | = | x - y |$ , it is non-expansive and not a contraction. Its fixed-point set is $\{ 0 \}$ , and $\mathcal { E } ( x ) = \vert x \vert$ while $\left\| x - T x \right\| = \left| x - \left( - x \right) \right| = 2 \left| x \right|$ , so $T$ is linearly regular with $\kappa = 1$ . The following example shows that linear regularity is not a consequence of non-expansiveness either.

Example 1. On $\chi = \mathbb { R }$ consider the piecewise afine map

$$
T ( x ) = \left\{ { \begin{array} { l l } { x , } & { x \leq 0 , } \\ { 0 , } & { 0 \leq x \leq 1 , } \\ { x - 1 , } & { x \geq 1 . } \end{array} } \right.
$$

Each piece has slope 0 or 1, and the pieces agree at the breakpoints, so � is continuous and non-expansive, and Fix $( T ) = ( - \infty , 0 ]$ . For $x \ge 1$ , the residual is $\| x - T x \| = 1$ while $\mathcal { E } ( x ) = x$ . So, there exists no finite � such that $\mathcal { E } ( x ) \leq \kappa \| x - T x \|$ for all �.

We study the stochastic KM iteration

$$
x _ { n } = \left( 1 - \alpha _ { n } \right) x _ { n - 1 } + \alpha _ { n } \left( T x _ { n - 1 } + M _ { n } \right) , \qquad n \geq 1 ,\tag{13}
$$

where $x _ { 0 } \in X$ and the stepsizes $\{ \alpha _ { n } \} _ { n \geq 1 }$ are deterministic and $\{ M _ { n } \} _ { n \geq 1 }$ is adapted to a filtration $\{ { \mathcal { F } } _ { n } \} _ { n \geq 0 }$ with $x _ { n }$ being $\mathcal { F } _ { n }$ -measurable. The noise is allowed to grow with the current residual $\| x _ { n - 1 } - T x _ { n - 1 } \|$

Assumption 3 (Martingale diference noise with residual-dependent variance). There is a constant $\sigma \geq 0$ such that, almost surely, for every $n \geq 1$

$$
\begin{array} { r } { \mathbb { E } \left[ M _ { n } \mid \mathcal { F } _ { n - 1 } \right] = 0 , \qquad \mathbb { E } \left[ \left. M _ { n } \right. ^ { 2 } \bigm \rvert \mathcal { F } _ { n - 1 } \right] \leq \sigma ^ { 2 } \left( 1 + \left. x _ { n - 1 } - T x _ { n - 1 } \right. ^ { 2 } \right) . } \end{array}
$$

We verify Assumption 3 for RVI �-learning in Section 4.3. Let � be any finite constant such that

$$
\mathbb { E } \left. \sum _ { i } U _ { i } \right. ^ { 2 } \leq \nu \sum _ { i } \mathbb { E } \left. U _ { i } \right. ^ { 2 }\tag{14}
$$

for every finite X-valued martingale diference sequence $\{ U _ { i } \}$ . Such a constant exists by equivalence with a Euclidean norm. Lemma 6 provides the admissible choice $\nu = 2 e \log ( 2 d )$ for the sup-norm on $\mathbb { R } ^ { d }$ when $d \ge 2$ (and $\nu = 1$ sufices when $d = 1 )$ .

We consider polynomial stepsize sequences, formalized in the following assumption.

Assumption 4 (Stepsizes). The stepsizes are given by

$$
\alpha _ { n } = c ( n + K _ { 0 } ) ^ { - a } , \qquad 0 < a \leq 1 , \quad c > 0 , \quad K _ { 0 } \in \{ 1 , 2 , . . . \} ,\tag{15}
$$

for every $n \geq 1$ . Also,

$$
c K _ { 0 } ^ { 1 - a } \geq 2 ^ { 8 } \kappa ^ { 2 } , \qquad \alpha _ { 1 } = \frac { c } { ( K _ { 0 } + 1 ) ^ { a } } \leq \frac { 1 } { \operatorname* { m a x } \left\{ 2 , 2 ^ { 8 } \nu \sigma ^ { 2 } ( 2 ^ { 7 } \kappa ^ { 2 } + 2 ) ^ { 2 } \right\} } .\tag{16}
$$

For convenience, we write $L : = 2 ^ { 7 } \kappa ^ { 2 }$

## 3.2 The Residual Bound

Theorem 2 (Mean-squared residual bound). Let the operator � satisfy Assumption 2, and consider (13) under Assumptions 3 and 4. Then, for every $n \geq 0$

$$
\mathbb { E } [ \| x _ { n } - T x _ { n } \| ^ { 2 } ] \leq 2 5 \mathcal { E } ( x _ { 0 } ) ^ { 2 } \exp \left( - \frac { \log ( 4 ) } { L + 1 / 2 } \sum _ { k = 1 } ^ { n } \alpha _ { k } \right) + 4 5 \nu \sigma ^ { 2 } ( L + 2 ) ^ { 2 } \frac { c } { ( n + K _ { 0 } + 1 ) ^ { a } } .\tag{17}
$$

The first term of (17) is the initial error, reduced by at least a factor of four once per block of iterations whose stepsizes sum to �, so that it is damped by a factor determined by the accumulated stepsize $\scriptstyle \sum _ { k = 1 } ^ { n } \alpha _ { k }$ The second term is the noise accumulated over the single most recent such block. The noise of every earlier block has a negligible efect due to exponential decay.

Remark 1 (Second term dominates). For $a \in ( 0 , 1 )$ , by (46) in Lemma 7,

$$
\sum _ { k = 1 } ^ { n } \alpha _ { k } \geq \frac { c } { 1 - a } [ ( n + K _ { 0 } + 1 ) ^ { 1 - a } - ( K _ { 0 } + 1 ) ^ { 1 - a } ] .
$$

Hence, for every $\begin{array} { r } { a \in ( 0 , 1 ) , \exp \left( - \frac { \log ( 4 ) } { L + 1 / 2 } \sum _ { k = 1 } ^ { n } \alpha _ { k } \right) } \end{array}$ decays faster than $( n + K _ { 0 } ) ^ { - a }$ . For $a = 1$ , by (47) in Lemma 7,

$$
\sum _ { k = 1 } ^ { n } \alpha _ { k } \geq c \log \left( \frac { n + K _ { 0 } + 1 } { K _ { 0 } + 1 } \right) .
$$

$\begin{array} { r } { \operatorname { I f } T = I . } \end{array}$ , the residual is zero at every point. Otherwise, (12) and linear regularity imply $\kappa \ge 1 / 2$ , and hence $L \ge 3 2$ . The preceding bound and $c \geq 2 L$ then give $c \log ( 4 ) / ( L + 1 / 2 ) \geq 6 4 \log ( 4 ) / ( 3 2 + 1 / 2 ) > 1$ , so the exponential term decays faster than $1 / ( n + K _ { 0 } )$ . Thus the mean-squared residual is $O ( n ^ { - a } )$ for $0 < a \le 1$

Remark 2 (Removing the ofset condition). For $a \in ( 0 , 1 )$ , the exponent � for the mean-squared residual also holds for every fixed $c > 0$ and integer $K _ { 0 } \ge 1$ with $\alpha _ { 1 } \leq 1$ , without imposing (16). For $a = 1$ , the same holds provided $c \geq 2 L$ . For the proof, choose a deterministic � for which $K _ { 0 } ^ { \prime } : = K _ { 0 } + n _ { 0 }$ satisfies (16), and apply Theorem 2 from time $n _ { 0 }$ onward. The proof uses the initial point only through the second moment of its distance to Fix(�), so it applies verbatim to the random point $x _ { n _ { 0 } }$ with $\mathcal { E } ( x _ { 0 } ) ^ { 2 }$ replaced by $\mathbb { E } [ \mathcal { E } ( x _ { n _ { 0 } } ) ^ { 2 } ]$ and $\scriptstyle \sum _ { k = 1 } ^ { n } \alpha _ { k }$ by $\scriptstyle \sum _ { k = n _ { 0 } + 1 } ^ { n } \alpha _ { k }$ . This second moment is finite. To see this, fix $x ^ { \star } \in \operatorname { F i x } ( T )$ with $\| x _ { 0 } - x ^ { \star } \| = \mathcal { E } ( x _ { 0 } )$ (13), (12) and Minkowski’s inequality give $\begin{array} { r } { \mathbb { E } [ \left\| x _ { k } - x ^ { \star } \right\| ^ { 2 } ] ^ { \frac { 1 } { 2 } } \leq ( 1 + 2 \sigma \alpha _ { k } ) \mathbb { E } [ \left\| x _ { k - 1 } - x ^ { \star } \right\| ^ { 2 } ] ^ { \frac { 1 } { 2 } } + \sigma \alpha _ { k } } \end{array}$ , from which we obtain

$$
\begin{array} { r } { \mathbb { E } [ \mathcal { E } ( x _ { n _ { 0 } } ) ^ { 2 } ] \leq \left[ \left( \mathcal { E } ( x _ { 0 } ) + \frac { 1 } { 2 } \right) e ^ { 2 \sigma \sum _ { k = 1 } ^ { n _ { 0 } } \alpha _ { k } } - \frac { 1 } { 2 } \right] ^ { 2 } . } \end{array}
$$

The rate $O ( n ^ { - a } )$ for the mean-squared residual therefore holds for any such fixed ofset, with a possibly large transient constant.

## 3.3 Proof of the Residual Bound

## 3.3.1 Contraction over a Block

Define deterministic block endpoints by $t _ { 0 } : = 0$ and

$$
t _ { j + 1 } : = \operatorname* { m i n } \left\{ b > t _ { j } : \sum _ { k = t _ { j } + 1 } ^ { b } \alpha _ { k } \geq L \right\} , \qquad j \geq 0 .\tag{18}
$$

Since $\textstyle \sum _ { k = 1 } ^ { \infty } \alpha _ { k } = \infty$ , every tail sum diverges, so the endpoints in (18) are finite. By (16), for each $j \geq 0$ $\textstyle \sum _ { k = t _ { i } + 1 } ^ { t _ { j + 1 } } \alpha _ { k } \in [ L , L + 1 / 2 ]$

Lemma 1. Let $T : X  X$ satisfy Assumption 2. For any integer $m \geq 0$ and any $y _ { m } \in X$ , consider the noiseless KM iteration

$$
y _ { k } = ( 1 - \alpha _ { k } ) y _ { k - 1 } + \alpha _ { k } T y _ { k - 1 } , \qquad k > m ,\tag{19}
$$

where the stepsizes satisfy the hypotheses of Theorem 2. Its distance to the fixed-point set is non-increasing, $i . e .$

$$
{ \mathcal { E } } ( y _ { k } ) \leq { \mathcal { E } } ( y _ { k - 1 } ) \leq { \mathcal { E } } ( y _ { m } ) , \qquad k > m .\tag{20}
$$

Moreover, $i f m = t _ { j }$ , then

$$
\mathcal { E } ( y _ { t _ { j + 1 } } ) \leq \frac { 1 } { 4 \sqrt { \pi } } \mathcal { E } ( y _ { t _ { j } } ) .\tag{21}
$$

Proof. Fix $m \ge 0 , y _ { m } \in X$ , and $x ^ { \star } \in \operatorname { F i x } ( T )$ , and set $R : = \| y _ { m } - x ^ { \star } \|$ . If $R = 0$ , then $y _ { k } = x ^ { \star }$ for every $k \geq m$ , so the residual bound is immediate. Suppose $R > 0$ and define $C : = \left\{ z \in X : \| z - x ^ { \star } \| \leq R \right\}$ . This is a non-empty, bounded, closed convex subset of the normed space $\chi _ { \cdot }$ , with $y _ { m } \in C$ and $\mathrm { d i a m } ( C ) \leq 2 R$ . For every $z \in C$ , non-expansiveness and $T x ^ { \star } = x ^ { \star }$ give

$$
\left\| T z - x ^ { \star } \right\| = \left\| T z - T x ^ { \star } \right\| \leq \left\| z - x ^ { \star } \right\| \leq R .
$$

Thus $T ( C ) \subseteq C$ , and the restriction $T | _ { C } : C \to C$ is non-expansive. Since $\alpha _ { k } \in [ 0 , 1 ]$ , convexity of � and induction in (19) show that $y _ { k } \in C$ for every $k \geq m$

To apply Cominetti et al. [12, Theorem 1], set $u _ { \ell } : = y _ { m + \ell }$ for $\ell \geq 0$ and $\beta _ { \ell } : = \alpha _ { m + \ell }$ for $\ell \geq 1$ . Then $u _ { 0 } = y _ { m } \in C$ and

$$
u _ { \ell } = ( 1 - \beta _ { \ell } ) u _ { \ell - 1 } + \beta _ { \ell } ( T | _ { C } ) ( u _ { \ell - 1 } ) , \qquad \ell \geq 1 .
$$

These are precisely the KM iterates of a non-expansive self-map of the bounded convex set �, with relaxation parameters $\beta _ { \ell } \in [ 0 , 1 ]$ , as required by that theorem. For any integer $b > m$ , applying it at $N = b - m \geq 1$ yields

$$
\| y _ { b } - T y _ { b } \| \leq \frac { \dim ( C ) } { \sqrt { \pi \sum _ { \ell = 1 } ^ { N } \beta _ { \ell } ( 1 - \beta _ { \ell } ) } } \leq \frac { 2 \| y _ { m } - x ^ { \star } \| } { \sqrt { \pi \sum _ { k = m + 1 } ^ { b } \alpha _ { k } ( 1 - \alpha _ { k } ) } } .
$$

The denominator is positive and independent of $x ^ { \star }$ . Taking the infimum over $x ^ { \star } \in \operatorname { F i x } ( T )$ , we have that

$$
\| y _ { b } - T y _ { b } \| \leq \frac { 2 \mathcal { E } ( y _ { m } ) } { \sqrt { \pi \sum _ { k = m + 1 } ^ { b } \alpha _ { k } ( 1 - \alpha _ { k } ) } } , \qquad b > m .\tag{22}
$$

For every $x ^ { \star } \in \operatorname { F i x } ( T )$ and $k > m$ , the triangle inequality and non-expansiveness give

$$
\begin{array} { r } { \left\| y _ { k } - x ^ { \star } \right\| \leq \left( 1 - \alpha _ { k } \right) \left\| y _ { k - 1 } - x ^ { \star } \right\| + \alpha _ { k } \left\| T y _ { k - 1 } - T x ^ { \star } \right\| \leq \left\| y _ { k - 1 } - x ^ { \star } \right\| . } \end{array}
$$

Taking the infimum over $x ^ { \star }$ , and iterating proves (20).

For $m = t _ { j }$ and $b = t _ { j + 1 }$ , minimality of � gives $\textstyle \sum _ { k = m + 1 } ^ { b - 1 } { \alpha _ { k } } < L$ . Together with $\alpha _ { b } \leq 1 / 2$ , this proves the asserted block-sum bounds. Also,

$$
\sum _ { k = t _ { j } + 1 } ^ { t _ { j + 1 } } \alpha _ { k } ( 1 - \alpha _ { k } ) \geq \frac 1 2 \sum _ { k = t _ { j } + 1 } ^ { t _ { j + 1 } } \alpha _ { k } \geq \frac { L } { 2 } .\tag{23}
$$

Finally, linear regularity, (22), and (23) imply

$$
\mathcal { E } ( y _ { b } ) \leq \kappa \| y _ { b } - T y _ { b } \| \leq \frac { 2 \kappa } { \sqrt { \pi L / 2 } } \mathcal { E } ( y _ { m } ) = \frac { 1 } { 4 \sqrt { \pi } } \mathcal { E } ( y _ { m } ) \leq \frac { 1 } { 4 } \mathcal { E } ( y _ { m } ) .
$$

This proves (21).

## 3.3.2 Tracking a Block with Martingale Noise

Lemma 2 (Block estimate). Let $m = t _ { j }$ and $b = t _ { j + 1 } .$ , and put $\eta _ { j } : = \sqrt { \nu } \sigma ( L + 2 ) \sqrt { \alpha _ { m + 1 } } . \ I f \eta _ { j } \leq 1 / 1 6 ,$ then

$$
\operatorname* { m a x } _ { m \leq k \leq b } \mathbb { E } [ \mathcal { E } ( x _ { k } ) ^ { 2 } ] \leq \frac { 4 8 } { 3 1 } \mathbb { E } [ \mathcal { E } ( x _ { m } ) ^ { 2 } ] + \frac { 6 4 } { 3 1 } \eta _ { j } ^ { 2 } ,\tag{24}
$$

$$
\mathbb { E } [ \mathcal { E } ( x _ { b } ) ^ { 2 } ] \le \frac { 1 } { 8 } \mathbb { E } [ \mathcal { E } ( x _ { m } ) ^ { 2 } ] + \frac { 8 0 } { 9 3 } \eta _ { j } ^ { 2 } .\tag{25}
$$

Bound (24) also holds with � replaced by any � with $m \leq n < t _ { j + 1 }$

Proof. First, note that $\| T x \| \leq \| x \| + \| T 0 \|$ . Fix $x ^ { \star } \in \mathrm { F i x } ( T )$ . By Assumption 3 and (12), $\mathbb { E } [ \| M _ { k } \| ^ { 2 } \ |$ $\mathcal { F } _ { k - 1 } \big ] \leq \sigma ^ { 2 } ( 1 + 4 \left\| x _ { k - 1 } - x ^ { \star } \right\| ^ { 2 } )$ . $\mathrm { S o } , \mathbb { E } \| x _ { k - 1 } \| ^ { 2 } <$ ∞ implies E $\| x _ { k } \| ^ { 2 } <$ ∞ by (13). Since $x _ { 0 }$ is deterministic, induction gives $\mathbb { E } \| x _ { k } \| ^ { 2 } <$ ∞ for every �. Thus, $\mathbb { E } [ \mathcal { E } ( x _ { k } ) ^ { 2 } ] < \infty$ , and $D _ { \operatorname* { m a x } } : = \operatorname* { m a x } _ { m \leq k \leq b } \mathbb { E } [ \mathcal { E } ( x _ { k } ) ^ { 2 } ] < \infty$

Run the noiseless comparison iteration (19) from the random point $y _ { m } : = x _ { m }$ , and reset, on this block only, the averaged noise process

$$
\xi _ { m } : = 0 , \qquad \xi _ { k } : = ( 1 - \alpha _ { k } ) \xi _ { k - 1 } + \alpha _ { k } M _ { k } , \qquad z _ { k } : = x _ { k } - \xi _ { k } , \quad k > m .
$$

Write $\Phi _ { k } : = ( 1 - \alpha _ { k } ) I + \alpha _ { k } T$ , where � is the identity map on $\chi .$ . The map $\Phi _ { k }$ is non-expansive as a convex combination of non-expansive maps, and the noise cancels in $z _ { k }$ :

$$
z _ { k } = ( 1 - \alpha _ { k } ) z _ { k - 1 } + \alpha _ { k } T x _ { k - 1 } = \Phi _ { k } z _ { k - 1 } + \alpha _ { k } \left( T x _ { k - 1 } - T z _ { k - 1 } \right) ,
$$

so $\left\{ \boldsymbol { z } _ { k } \right\}$ is an inexact KM iteration with per-step error at most $\alpha _ { k } \left\| x _ { k - 1 } - z _ { k - 1 } \right\| = \alpha _ { k } \left\| \xi _ { k - 1 } \right\|$ . Since $z _ { m } = y _ { m }$ and $\Phi _ { k }$ is non-expansive, $\left\| z _ { k } - y _ { k } \right\| \leq \left\| z _ { k - 1 } - y _ { k - 1 } \right\| + \alpha _ { k } \left\| \xi _ { k - 1 } \right\|$ , and therefore, pathwise,

$$
\begin{array} { r l } & { \quad \quad \| z _ { k } - y _ { k } \| \leq \displaystyle \sum _ { i = m + 1 } ^ { k } \alpha _ { i } \| \xi _ { i - 1 } \| } \\ & { \Longrightarrow \| x _ { k } - y _ { k } \| \leq \| x _ { k } - z _ { k } \| + \displaystyle \sum _ { i = m + 1 } ^ { k } \alpha _ { i } \| \xi _ { i - 1 } \| \leq \| \xi _ { k } \| + \displaystyle \sum _ { i = m + 1 } ^ { k } \alpha _ { i } \| \xi _ { i - 1 } \| , \quad m \leq k \leq b . } \end{array}\tag{26}
$$

Unrolling the recursion for $\xi _ { k }$ from $\xi _ { m } = 0$ gives $\begin{array} { r } { \xi _ { k } = \sum _ { i = m + 1 } ^ { k } w _ { i , k } M _ { i } } \end{array}$ with the deterministic weights $\begin{array} { r } { w _ { i , k } : = \alpha _ { i } \prod _ { \ell = i + 1 } ^ { k } \left( 1 - \alpha _ { \ell } \right) } \end{array}$ . Since the stepsizes are non-increasing and at most $1 / 2$ , induction on � gives

$$
\sum _ { i = m + 1 } ^ { k } w _ { i , k } ^ { 2 } = ( 1 - \alpha _ { k } ) ^ { 2 } \sum _ { i = m + 1 } ^ { k - 1 } w _ { i , k - 1 } ^ { 2 } + \alpha _ { k } ^ { 2 } \leq ( 1 - \alpha _ { k } ) ^ { 2 } \frac { \alpha _ { m + 1 } } { 2 - \alpha _ { m + 1 } } + \alpha _ { k } ^ { 2 } \leq \frac { \alpha _ { m + 1 } } { 2 - \alpha _ { m + 1 } } \leq \frac { 2 } { 3 } \alpha _ { m + 1 } .\tag{27}
$$

The induction starts with an empty sum at $k = m ;$ the second inequality follows from $\alpha _ { k } / ( 2 - \alpha _ { k } ) \ \leq$ $\alpha _ { m + 1 } / ( 2 - \alpha _ { m + 1 } )$ . The sequence $\{ w _ { i , k } M _ { i } \} _ { i }$ <sub>�</sub> is a martingale diference sequence, so (14) applies. Combining it with (27), Assumption 3, the tower property of conditional expectation and (12),

$$
\mathbb { E } \left\| \dot { \xi } _ { k } \right\| ^ { 2 } \leq \nu \sum _ { i = m + 1 } ^ { k } w _ { i , k } ^ { 2 } \mathbb { E } \left\| M _ { i } \right\| ^ { 2 } \leq \frac { 2 } { 3 } \nu \alpha _ { m + 1 } \sigma ^ { 2 } \left( 1 + 4 D _ { \operatorname* { m a x } } \right) , \qquad m \leq k \leq b ,
$$

since each index $i - 1$ lies in $[ m , b ]$ . Applying weighted Cauchy–Schwarz to (26) gives

$$
\mathbb { E } \left. x _ { k } - y _ { k } \right. ^ { 2 } \leq \left( 1 + \sum _ { i = m + 1 } ^ { k } \alpha _ { i } \right) \left( \mathbb { E } \left. \xi _ { k } \right. ^ { 2 } + \sum _ { i = m + 1 } ^ { k } \alpha _ { i } \mathbb { E } \left. \xi _ { i - 1 } \right. ^ { 2 } \right) .
$$

Since $\begin{array} { r } { 1 + \sum _ { i = m + 1 } ^ { b } \alpha _ { i } \leq L + 3 / 2 \leq L + 2 } \end{array}$ , we obtain

$$
\operatorname* { m a x } _ { m \leq k \leq b } \mathbb { E } \left\| x _ { k } - y _ { k } \right\| ^ { 2 } \leq \frac { 2 } { 3 } \eta _ { j } ^ { 2 } \left( 1 + 4 D _ { \operatorname* { m a x } } \right) .\tag{28}
$$

The distance to a non-empty set is 1-Lipschitz, so $\mathcal { E } ( x _ { k } ) \leq \mathcal { E } ( y _ { k } ) + \| x _ { k } - y _ { k } \|$ . By (20) and $y _ { m } = x _ { m }$ weighted Cauchy–Schwarz and (28) give

$$
\mathbb { E } [ \mathcal { E } ( x _ { k } ) ^ { 2 } ] \le \frac { 3 } { 2 } \mathbb { E } [ \mathcal { E } ( x _ { m } ) ^ { 2 } ] + 3 \mathbb { E } \| x _ { k } - y _ { k } \| ^ { 2 } \le \frac { 3 } { 2 } \mathbb { E } [ \mathcal { E } ( x _ { m } ) ^ { 2 } ] + 2 \eta _ { j } ^ { 2 } \left( 1 + 4 D _ { \operatorname* { m a x } } \right) , \qquad m \le k \le b .
$$

The right-hand side does not depend on �, so $\begin{array} { r } { ( 1 - 8 \eta _ { j } ^ { 2 } ) D _ { \mathrm { m a x } } \leq \frac { 3 } { 2 } \mathbb { E } [ \mathcal { E } ( x _ { m } ) ^ { 2 } ] + 2 \eta _ { j } ^ { 2 } , } \end{array}$ ; this rearrangement is legitimate because $D _ { \mathrm { m a x } }$ is finite. Since $\eta _ { j } \leq 1 / 1 6 ~ \mathrm { g i v e s } ~ 1 - 8 \eta _ { i } ^ { 2 } \geq 3 1 / 3 2$ , we obtain

$$
D _ { \operatorname* { m a x } } \leq \frac { \frac { 3 } { 2 } \mathbb { E } [ \mathcal { E } ( x _ { m } ) ^ { 2 } ] + 2 \eta _ { j } ^ { 2 } } { 1 - 8 \eta _ { j } ^ { 2 } } \leq \frac { 4 8 } { 3 1 } \mathbb { E } [ \mathcal { E } ( x _ { m } ) ^ { 2 } ] + \frac { 6 4 } { 3 1 } \eta _ { j } ^ { 2 } ,
$$

which is (24). At the block endpoint, (21) gives ${ \mathcal E } ( y _ { b } ) ^ { 2 } \le { \mathcal E } ( x _ { m } ) ^ { 2 } / ( 1 6 \pi )$ pathwise. Using weighted Cauchy–Schwarz with coeficients 5 and $5 / 4 ,$ , and substituting the first bound on $D _ { \mathrm { m a x } }$ above into (28), yields

$$
\begin{array} { r l r } {  { \mathbb { E } [ \mathcal { E } ( x _ { b } ) ^ { 2 } ] \leq \frac { 5 } { 1 6 \pi } \mathbb { E } [ \mathcal { E } ( x _ { m } ) ^ { 2 } ] + \frac { 5 } { 4 } \mathbb { E } \| x _ { b } - y _ { b } \| ^ { 2 } } } \\ & { } & { \leq \frac { 5 } { 1 6 \pi } \mathbb { E } [ \mathcal { E } ( x _ { m } ) ^ { 2 } ] + \frac { 5 } { 6 } \eta _ { j } ^ { 2 } ( 1 + 4 D _ { \operatorname* { m a x } } ) } \\ & { } & { \leq ( \frac { 5 } { 1 6 \pi } + \frac { 5 \eta _ { j } ^ { 2 } } { 1 - 8 \eta _ { j } ^ { 2 } } ) \mathbb { E } [ \mathcal { E } ( x _ { m } ) ^ { 2 } ] + \frac { 5 \eta _ { j } ^ { 2 } } { 6 ( 1 - 8 \eta _ { j } ^ { 2 } ) } } \\ & { } & { \leq ( \frac { 5 } { 4 8 } + \frac { 5 } { 2 4 8 } ) \mathbb { E } [ \mathcal { E } ( x _ { m } ) ^ { 2 } ] + \frac { 8 0 } { 9 3 } \eta _ { j } ^ { 2 } } \\ & { } & { \leq \frac { 1 } { 8 } \mathbb { E } [ \mathcal { E } ( x _ { m } ) ^ { 2 } ] + \frac { 8 0 } { 9 3 } \eta _ { j } ^ { 2 } . } \end{array}
$$

We used $\pi > 3$ and $5 / 4 8 + 5 / 2 4 8 = 1 8 5 / 1 4 8 8 < 1 / 8$ . This is (25), and the same argument for (24) applies to any partial block. This completes the proof. □

## 3.3.3 Completing the Proof

Proof of Theorem 2. We first verify the hypothesis of Lemma 2. By (16), for every $j \geq 0$

$$
\eta _ { j } ^ { 2 } = \nu \sigma ^ { 2 } ( L + 2 ) ^ { 2 } \alpha _ { t _ { j } + 1 } \leq \frac { \nu \sigma ^ { 2 } ( L + 2 ) ^ { 2 } } { \operatorname* { m a x } \left\{ 2 , 2 ^ { 8 } \nu \sigma ^ { 2 } ( L + 2 ) ^ { 2 } \right\} } \leq 2 ^ { - 8 } , \qquad \mathrm { t h a t ~ i s , } \qquad \eta _ { j } \leq 2 ^ { - 4 } .\tag{29}
$$

Next, we show that the blocks at most double on the shifted clock $n \mapsto n + K _ { 0 }$ . Applying (48) in Lemma 7 with $m = t _ { j }$ , the stepsizes indexed by $t _ { j } + 1 , \ldots , 2 t _ { j } + K _ { 0 }$ already sum to at least �, so by the minimality in (18),

$$
t _ { j + 1 } \leq 2 t _ { j } + K _ { 0 } , \qquad \mathrm { h e n c e } \qquad t _ { j + 1 } + K _ { 0 } \leq 2 \left( t _ { j } + K _ { 0 } \right) .\tag{30}
$$

This uses the first condition in (16), and it is a statement about the stepsize sequence alone, so it is valid for every �. By (49) in Lemma 7 together with (30),

$$
\alpha _ { t _ { j } + 1 } \leq 2 ^ { a } \alpha _ { n + 1 } \leq 2 \alpha _ { n + 1 } , \qquad t _ { j } \leq n \leq t _ { j + 1 } .\tag{31}
$$

Set $e _ { j } : = \mathbb { E } [ \mathcal { E } ( x _ { t _ { i } } ) ^ { 2 } ]$ and $q : = 1 / 8$ . Thus, $e _ { 0 } = \mathcal { E } ( x _ { 0 } ) ^ { 2 }$ . By (29), Lemma 2 applies to every block and gives $\begin{array} { r } { e _ { j + 1 } \le q e _ { j } + \frac { 8 \bar { 0 } } { 9 3 } \eta _ { j } ^ { 2 } } \end{array}$ , from which we obtain

$$
e _ { j } \leq q ^ { j } \mathcal { E } ( x _ { 0 } ) ^ { 2 } + \frac { 8 0 } { 9 3 } \sum _ { i = 0 } ^ { j - 1 } q ^ { j - 1 - i } \eta _ { i } ^ { 2 } .
$$

Iterating (30) gives $t _ { j } + 1 + K _ { 0 } \le 2 ^ { j - i } \big ( t _ { i } + 1 + K _ { 0 } \big ) \mathrm { ~ f o r ~ } i < j ,$ so

$$
\eta _ { i } ^ { 2 } = \eta _ { j } ^ { 2 } \left( \frac { t _ { j } + 1 + K _ { 0 } } { t _ { i } + 1 + K _ { 0 } } \right) ^ { a } \leq \eta _ { j } ^ { 2 } 2 ^ { j - i } ,
$$

and, substituting $\ell = j - i ,$

$$
\frac { 8 0 } { 9 3 } \sum _ { i = 0 } ^ { j - 1 } q ^ { j - 1 - i } \eta _ { i } ^ { 2 } \le \frac { 1 6 0 } { 9 3 } \eta _ { j } ^ { 2 } \sum _ { \ell = 1 } ^ { \infty } ( 2 q ) ^ { \ell - 1 } = \frac { 1 6 0 } { 9 3 ( 1 - 1 / 4 ) } \eta _ { j } ^ { 2 } = \frac { 6 4 0 } { 2 7 9 } \eta _ { j } ^ { 2 } .
$$

Here $2 q \ < \ 1$ . Thus, the contributions of earlier blocks form a convergent geometric sum, and $e _ { j } \ \leq$ $\begin{array} { r } { q ^ { j } \mathcal { E } ( x _ { 0 } ) ^ { 2 } + \frac { 6 4 0 } { 2 7 9 } \eta _ { j } ^ { 2 } } \end{array}$ . For $t _ { j } \leq n < t _ { j + 1 }$ , bound (24) together with (31) gives

$$
\mathbb { E } [ \mathcal { E } ( x _ { n } ) ^ { 2 } ] \leq \frac { 4 8 } { 3 1 } e _ { j } + \frac { 6 4 } { 3 1 } \eta _ { j } ^ { 2 } \leq \frac { 4 8 } { 3 1 } q ^ { j } \mathcal { E } ( x _ { 0 } ) ^ { 2 } + \frac { 1 6 1 9 2 } { 2 8 8 3 } \eta _ { j } ^ { 2 } \leq \frac { 4 8 } { 3 1 } 4 ^ { - j } \mathcal { E } ( x _ { 0 } ) ^ { 2 } + \frac { 3 2 3 8 4 } { 2 8 8 3 } \nu \sigma ^ { 2 } ( L + 2 ) ^ { 2 } \alpha _ { n + 1 } .\tag{32}
$$

We used $q \leq 1 / 4$ . It remains to express $j$ in terms of the accumulated stepsize. Every completed block contributes at most $L + 1 / 2$ and the current partial block contributes less than $L ,$ , so $\begin{array} { r } { \sum _ { k = 1 } ^ { n } \alpha _ { k } < j ( L + 1 / 2 ) + L \le } \end{array}$ $( j + 1 ) ( L + 1 / 2 )$ , that is, $\begin{array} { r } { j \geq \frac { 1 } { L + 1 / 2 } \sum _ { k = 1 } ^ { n } \alpha _ { k } - 1 } \end{array}$ , and therefore

$$
\frac { 1 9 2 } { 3 1 } { 4 ^ { - j } } \leq \frac { 7 6 8 } { 3 1 } \exp \left( - \frac { \log ( 4 ) } { L + 1 / 2 } \sum _ { k = 1 } ^ { n } \alpha _ { k } \right) \leq 2 5 \exp \left( - \frac { \log ( 4 ) } { L + 1 / 2 } \sum _ { k = 1 } ^ { n } \alpha _ { k } \right) .
$$

Multiplying (32) by 4 and using (12), $1 2 9 5 3 6 / 2 8 8 3 < 4 5$ , and $\alpha _ { n + 1 } = c ( n + K _ { 0 } + 1 ) ^ { - a }$ proves (17). For $n = 0$ the claim is immediate from $\| x _ { 0 } - T x _ { 0 } \| ^ { 2 } \leq 4 \mathcal { E } ( x _ { 0 } ) ^ { 2 }$ □

## 4 Proofs of the Main Results

This section proves Theorem 1 and Corollary 1. We begin by introducing an auxiliary iteration governed by the centered Bellman operator, which is non-expansive in the sup-norm. Then, we show that analyzing the auxiliary iteration is suficient to establish the convergence rate of RVI Q-learning. Subsequently, this section verifies the two hypotheses needed to apply Theorem 2 to the auxiliary iteration, i.e., linear regularity and residual-dependent growth of the sampling noise. We then complete the proofs.

## 4.1 The Centered Bellman Operator and Its Equivalence to the RVI Map

� is non-expansive in the span seminorm, which is not a norm, and in general, � is not non-expansive with respect to the sup-norm, because the normalization term $f ( Q ) \mathbf { 1 }$ depends on �. The obstruction disappears once that term is replaced by a constant. To this end, define the centered Bellman operator $\bar { H }$ as

$$
\bar { H } Q : = B Q - J ^ { \star } \mathbf { 1 } , \quad \forall Q \in \mathbb { R } ^ { S \times \mathcal { A } } ,\tag{33}
$$

where � is the Bellman operator (6). �<sup>¯</sup> is non-expansive with respect to the sup-norm, and the set of fixed points of ${ \bar { H } } ,$ denoted by Fix(�<sup>¯</sup> ), is exactly the set of solutions of the AROE (1). Define the following auxiliary iteration:

$$
\bar { Q } _ { n + 1 } = \bar { Q } _ { n } + \alpha _ { n + 1 } \left( \bar { H } \bar { Q } _ { n } - \bar { Q } _ { n } + M _ { n + 1 } ( \bar { Q } _ { n } ) \right) , \quad n \geq 0 ,\tag{34}
$$

where $\bar { Q } _ { 0 } = Q _ { 0 }$ . The auxiliary iterates $\{ \bar { Q } _ { n } \}$ are constructed on the same probability space and use the same samples $Y _ { s , a } ^ { n + 1 }$ as (5). The value $J ^ { \star }$ is not known to the agent. Hence, this iteration cannot be used to learn $Q ^ { \star }$ and will be used only for analysis. It turns out that $\left\| Q _ { n } - H Q _ { n } \right\| _ { \mathrm { s p } } \leq 2 \left\| \bar { Q } _ { n } - \bar { H } \bar { Q } _ { n } \right\| _ { \infty }$ , which is the next lemma.

## Lemma 3. For every $n \geq 0 , \left. Q _ { n } - H Q _ { n } \right. _ { \mathrm { s p } } \leq 2 \left. \bar { Q } _ { n } - \bar { H } \bar { Q } _ { n } \right. _ { \infty } .$

Proof. We first show that $\left. Q _ { n } - { \bar { Q } } _ { n } \right. _ { \mathrm { s p } } = 0$ using induction. The base case holds due to equality. Assume the induction hypothesis, $\bar { Q } _ { n } = Q _ { n } + c _ { n } \mathbf { 1 }$ , for some $c _ { n } \in \mathbb { R }$ . Then,

$$
\bar { Q } _ { n + 1 } ( s , a ) = \bar { Q } _ { n } ( s , a ) + \alpha _ { n + 1 } \left( r ( s , a , Y _ { s , a } ^ { n + 1 } ) + V _ { \bar { Q } _ { n } } ( Y _ { s , a } ^ { n + 1 } ) - \bar { Q } _ { n } ( s , a ) - J ^ { \star } \right)
$$

$$
\begin{array} { r l } & { = Q _ { n } ( s , a ) + c _ { n } + \alpha _ { n + 1 } \left( r ( s , a , Y _ { s , a } ^ { n + 1 } ) + V _ { Q _ { n } } ( Y _ { s , a } ^ { n + 1 } ) - Q _ { n } ( s , a ) - f ( Q _ { n } ) + ( f ( Q _ { n } ) - J ^ { \star } ) \right) } \\ & { = Q _ { n + 1 } ( s , a ) + c _ { n } + \alpha _ { n + 1 } ( f ( Q _ { n } ) - J ^ { \star } ) . } \end{array}
$$

Since $c _ { n + 1 } : = c _ { n } + \alpha _ { n + 1 } ( f ( Q _ { n } ) - J ^ { \star } )$ does not depend on $( s , a )$ , the claim follows.

Note that $\bar { Q } _ { n } - \bar { H } \bar { Q } _ { n } = Q _ { n } + c _ { n } { \bf 1 } - B ( Q _ { n } + c _ { n } { \bf 1 } ) + J ^ { \star } { \bf 1 } = Q _ { n } - B Q _ { n } + J ^ { \star } { \bf 1 }$ , which difers from $Q _ { n } - B Q _ { n }$ by a multiple of 1. Hence $\left\| { \bar { Q } } _ { n } - { \bar { H } } { \bar { Q } } _ { n } \right\| _ { \mathrm { s p } } = \left\| Q _ { n } - B Q _ { n } \right\| _ { \mathrm { s p } } = \left\| Q _ { n } - H Q _ { n } \right\| _ { \mathrm { s p } }$ . The proof is then complete due to the fact that $\| \cdot \| _ { \mathrm { s p } } \leq 2 \ : \| \cdot \| _ { \infty }$ □

Hence, it sufices to analyze (34) in order to study the convergence behavior of $\| Q _ { n } - H Q _ { n } \| _ { \mathrm { s p } }$ , and hence of the policy suboptimality $\Delta _ { n }$

## 4.2 Linear Regularity

Our goal is to show that �<sup>¯</sup> is linearly regular in the sup-norm (Definition 1), namely, to find a finite constant $\kappa \geq 1$ such that

$$
\mathcal { E } ( Q ) \leq \kappa \left. Q - \bar { H } Q \right. _ { \infty } , \qquad Q \in \mathbb { R } ^ { S \times \mathcal { A } } ,
$$

where $\begin{array} { r } { { \mathcal E } ( Q ) = \operatorname* { i n f } _ { q \in \mathrm { F i x } ( \bar { H } ) } \| Q - q \| _ { \infty } } \end{array}$ is the distance to $\operatorname { F i x } ( { \bar { H } } )$ . We first bound the span of a �-function by its residual. By Lemma 4 in Appendix A, every $Q \in \mathbb { R } ^ { S \times \mathcal { A } }$ satisfies

$$
\left\| Q \right\| _ { \mathrm { s p } } \leq \left( 1 + 3 D \right) \left( 1 + \left\| Q - \bar { H } Q \right\| _ { \mathrm { s p } } \right) ,
$$

and in particular $\| Q \| _ { \mathsf { s p } } \leq 1 + 3 D$ for every $Q \in \operatorname { F i x } ( { \bar { H } } )$ . Shift invariance of $\operatorname { F i x } ( { \bar { H } } )$ allows $\varepsilon ( Q )$ to be expressed using the span of �. For every $\boldsymbol { u } \in \mathbb { R } ^ { \boldsymbol { S \times \mathcal { A } } }$ , centering its range gives inf<sub>�∈R</sub> $\begin{array} { r } { \| u - c \mathbf { 1 } \| _ { \infty } = \frac { 1 } { 2 } \left\| u \right\| _ { \mathrm { s p } } . } \end{array}$ Hence,

$$
\begin{array} { r } { \mathcal { E } ( Q ) = \displaystyle \frac { 1 } { 2 } \operatorname* { i n f } _ { z \in \mathrm { F i x } ( \bar { H } ) } \| Q - z \| _ { \mathrm { s p } } \leq \frac { 1 } { 2 } \left( \| Q \| _ { \mathrm { s p } } + ( 1 + 3 D ) \right) \leq ( 1 + 3 D ) \left( 1 + \frac { 1 } { 2 } \left\| Q - \bar { H } Q \right\| _ { \mathrm { s p } } \right) } \\ { \leq ( 1 + 3 D ) \left( 1 + \left\| Q - \bar { H } Q \right\| _ { \infty } \right) . } \end{array}\tag{35}
$$

The next proposition removes the additive constant from the above upperbound, with the explicit regularity constant � of (8).

Proposition 1 (Linear regularity). Let Assumption 1 hold. Then �<sup>¯</sup> is linearly regular in the sup-norm with constant �, that is,

$$
\mathcal { E } ( Q ) \leq K \left. Q - \bar { H } Q \right. _ { \infty } , \qquad Q \in \mathbb { R } ^ { S \times \mathcal { A } } .
$$

Proof. Fix $Q \in \mathbb { R } ^ { S \times \mathcal { A } }$ . We consider two cases, corresponding to the two terms in (8).

Case $I \colon \left\| Q - \bar { H } Q \right\| _ { \infty } \geq \Delta _ { \mathrm { p o l } }$ . Then $1 \leq \left. Q - \bar { H } Q \right. _ { \infty } / \Delta _ { \mathrm { p o l } }$ , so (35) gives

$$
\mathcal { E } ( Q ) \leq \left( 1 + 3 D \right) \left( 1 + \left\| Q - \bar { H } Q \right\| _ { \infty } \right) \leq \left( 1 + 3 D \right) \left( 1 + \frac { 1 } { \Delta _ { \mathrm { p o l } } } \right) \left\| Q - \bar { H } Q \right\| _ { \infty } \leq K \left\| Q - \bar { H } Q \right\| _ { \infty } .
$$

Case $2 \colon \left\| Q - \bar { H } Q \right\| _ { \infty } < \Delta _ { \mathrm { p o l } }$ . For $\boldsymbol { x } \in \mathbb { R } ^ { S }$ and $\pi \in \Pi _ { \mathrm { S D } }$ , define

$$
( { \bar { G } } x ) ( s , a ) : = { \overline { { r } } } ( s , a ) - J ^ { \star } + \sum _ { s ^ { \prime } \in S } p ( s , a , s ^ { \prime } ) x ( s ^ { \prime } ) ,
$$

$$
F x : = V _ { \bar { G } x } , \qquad F _ { \pi } x : = \overline { { { r } } } _ { \pi } - J ^ { \star } \mathbf { 1 } + P ^ { \pi } x ,
$$

where $\overline { { r } } _ { \pi } ( s ) : = \overline { { r } } ( s , \pi ( s ) )$ and $P ^ { \pi } ( s , s ^ { \prime } ) : = p ( s , \pi ( s ) , s ^ { \prime } )$ . Here, 1 denotes the all-ones vector in $\mathbb { R } ^ { s }$ , and vector inequalities are coordinatewise. The maps ${ \bar { G } } , F .$ , and $F _ { \pi }$ are monotone and commute with constant shifts.

Since $\bar { G } V _ { Q } = \bar { H } Q , Q ( s , a ) \leq V _ { Q } ( s ) , Q ( s , \pi _ { Q } ( s ) ) = V _ { Q } ( s )$ , and every entry of ${ \bar { H } } Q - Q$ lies in $[ - \left. Q - \bar { H } Q \right. _ { \infty } , \left. Q - \bar { H } Q \right. _ { \infty } ]$ , we have

$$
F V _ { Q } \leq V _ { Q } + \left\| Q - \bar { H } Q \right\| _ { \infty } \mathbf { 1 } , \qquad F _ { \pi _ { Q } } V _ { Q } \geq V _ { Q } - \left\| Q - \bar { H } Q \right\| _ { \infty } \mathbf { 1 } .\tag{36}
$$

Taking expectations under $\pi _ { Q }$ and summing the second inequality in (36) over � steps gives

$$
\frac { 1 } { k } \mathbb { E } _ { s } ^ { \pi _ { Q } } \left[ \sum _ { t = 0 } ^ { k - 1 } \bar { r } _ { \pi _ { Q } } ( S _ { t } ) \right] \geq J ^ { \star } - \left\| Q - \bar { H } Q \right\| _ { \infty } + \frac { V _ { Q } ( s ) - \mathbb { E } _ { s } ^ { \pi _ { Q } } [ V _ { Q } ( S _ { k } ) ] } { k } .
$$

Since $V _ { Q }$ is bounded, letting $k \to \infty$ yields $0 \leq J ^ { \star } - J ( \pi _ { Q } , s ) \leq \left\| Q - { \bar { H } } Q \right\| _ { \infty } < \Delta _ { \mathrm { p o l } }$ for every �. Hence (3) shows that $\pi _ { Q }$ is optimal from every state.

Since $F _ { \pi _ { Q } } V _ { Q } \leq F V _ { Q }$ , (36) implies $\left\| F V _ { Q } - V _ { Q } \right\| _ { \infty } \leq \left\| Q - \bar { H } Q \right\| _ { \infty } < \Delta _ { \mathrm { p o l } }$ . Thus Lemma 5 in Appendix A applies with $\nu = V _ { Q }$ and $\rho = \left\| Q - { \bar { H } } Q \right\| _ { \infty }$ , and gives $w \in \mathbb { R } ^ { S }$ such that

$$
F w \leq w , \qquad V _ { Q } \leq w \leq V _ { Q } + \mathcal { T } \left\| Q - \bar { H } Q \right\| _ { \infty } \mathbf { 1 } .\tag{37}
$$

Let � be the union of all recurrent classes of $P ^ { \pi _ { Q } }$ . Fix one recurrent class, and let $\mu$ be its stationary distribution, extended by zero outside that class, so that $\mu ^ { \intercal } P ^ { \pi _ { Q } } = \mu ^ { \intercal }$ . For � in that class, the averages $\begin{array} { r } { \frac { 1 } { k } \sum _ { t = 0 } ^ { k - 1 } ( P ^ { \pi _ { Q } } ) ^ { t } ( s , \cdot ) } \end{array}$ converge to � as $k  \infty ,$ so $J ( \pi _ { Q } , s ) = \mu ^ { \top } \overline { { r } } _ { \pi _ { Q } }$ , which equals $J ^ { \star }$ because $\pi _ { Q }$ is optimal. Using $F _ { \pi _ { O } } w \leq F w \leq w$ and $\mu ^ { \top } P ^ { \pi _ { Q } } = \mu ^ { \top }$ , we get

$$
w - F _ { \pi _ { Q } } w \geq 0 , \qquad \mu ^ { \top } ( w - F _ { \pi _ { Q } } w ) = J ^ { \star } - \mu ^ { \top } \overline { { r } } _ { \pi _ { Q } } = 0 .
$$

The strict positivity of � on its class implies $F _ { \pi _ { O } } w = w$ on that class. As the class was arbitrary, $F _ { \pi _ { Q } } w = w$ on �, and then $F _ { \pi _ { O } } w \leq F w \leq u$ gives $F w = w$ on �.

Define $w _ { 0 } : = w$ and $w _ { n + 1 } : = F w _ { n } . \mathrm { ~ B y ~ }$ monotonicity and $F w \leq w$ , this sequence is non-increasing. It is bounded below by a suficiently low constant shift of a fixed point of $F \colon$ such a fixed point exists because $V _ { q } = F V _ { q }$ for any $q \in \operatorname { F i x } ( { \bar { H } } )$ . Thus $w _ { n }$ converges to a finite vector ℎ, and continuity gives $F h = h$ and $h \leq w$ . Moreover, $w _ { n } = w$ on � for every �. Indeed, if this holds at step $n ,$ closure of � under $\pi _ { Q }$ and $w _ { n } \leq w \mathrm { g i v e } , \mathrm { o n } C$

$$
w = F _ { \pi _ { Q } } w _ { n } \leq F w _ { n } \leq F w = w .
$$

Consequently, $h = w \geq V _ { Q }$ on $C .$

Set � $: = V _ { Q } - h$ . By (36) and $h = F h \geq F _ { \pi _ { O } } h .$

$$
d \leq P ^ { \pi _ { Q } } d + \left\| Q - { \bar { H } } Q \right\| _ { \infty } \mathbf { 1 } , \qquad d \leq 0 \mathrm { o n } C .
$$

The chain under $\pi _ { Q }$ reaches � almost surely from every state. Since $\pi _ { Q }$ is optimal from every state, (4) gives $\mathbb { E } _ { s } ^ { \pi _ { Q } } [ \tau _ { C } ] \leq \mathcal { T }$ . Iterating the preceding inequality up to $\tau _ { C } \wedge n$ gives

$$
\begin{array} { r } { d ( s ) \leq \mathbb { E } _ { s } ^ { \pi _ { Q } } [ d ( S _ { \tau _ { C } \wedge n } ) ] + \left\| Q - \bar { H } Q \right\| _ { \infty } \mathbb { E } _ { s } ^ { \pi _ { Q } } [ \tau _ { C } \wedge n ] . } \end{array}
$$

Since � is bounded and $\tau _ { C }$ has finite mean, letting $n \to \infty$ yields

$$
\begin{array} { r } { d ( s ) \leq \mathbb { E } _ { s } ^ { \pi _ { Q } } [ d ( S _ { \tau _ { C } } ) ] + \left\| Q - \bar { H } Q \right\| _ { \infty } \mathbb { E } _ { s } ^ { \pi _ { Q } } [ \tau _ { C } ] \leq \mathcal { T } \left\| Q - \bar { H } Q \right\| _ { \infty } . } \end{array}
$$

Together with (37) and $h \leq w$ , this proves

$$
V _ { Q } - \mathcal { T } \left. Q - \bar { H } Q \right. _ { \infty } \mathbf { 1 } \leq h \leq V _ { Q } + \mathcal { T } \left. Q - \bar { H } Q \right. _ { \infty } \mathbf { 1 } ,
$$

that is, $\left\| V _ { Q } - h \right\| _ { \infty } \leq \mathcal { T } \left\| Q - \bar { H } Q \right\| _ { \infty }$ Finally, set $q : = G h .$ . Then $V _ { q } = F h = h$ , so $\bar { H } q = \bar { G } V _ { q } = \bar { G } h = q$ . Since $( \bar { G } V _ { Q } - \bar { G } h ) ( s , a ) =$   
$\begin{array} { r l } { \sum _ { s ^ { \prime } } p ( s , a , s ^ { \prime } ) ( V _ { Q } ( s ^ { \prime } ) - h ( s ^ { \prime } ) ) } & { { } } \end{array}$

$$
\begin{array} { r l } & { \mathcal { E } ( Q ) \leq \| Q - q \| _ { \infty } \leq \left\| Q - \bar { H } Q \right\| _ { \infty } + \left\| \bar { G } V _ { Q } - \bar { G } h \right\| _ { \infty } } \\ & { \qquad \leq \left\| Q - \bar { H } Q \right\| _ { \infty } + \left\| V _ { Q } - h \right\| _ { \infty } } \\ & { \qquad \leq \left( 1 + \mathcal { T } \right) \left\| Q - \bar { H } Q \right\| _ { \infty } \leq K \left\| Q - \bar { H } Q \right\| _ { \infty } . } \end{array}
$$

This completes the proof in both cases.

## 4.3 Sampling Noise

We next verify Assumption 3.

Proposition 2. Let Assumption 1 hold. For every $n \geq 1$ and every $\mathcal { F } _ { n - 1 }$ -measurable $Q \in \mathbb { R } ^ { S \times \mathcal { A } }$ with $\mathbb { E } \| Q \| _ { \infty } ^ { 2 } < \infty ,$ almost surely,

$$
\begin{array} { r } { \mathbb { E } [ M _ { n } ( Q ) \ | \ { \mathcal F } _ { n - 1 } ] = 0 , \qquad \mathbb { E } [ \| M _ { n } ( Q ) \| _ { \infty } ^ { 2 } \ | \ { \mathcal F } _ { n - 1 } ] \le 8 ( 1 + 3 D ) ^ { 2 } \left( 1 + \| Q - \bar { H } Q \| _ { \infty } ^ { 2 } \right) . } \end{array}
$$

Proof. Conditional sampling gives $\mathbb { E } [ M _ { n } ( Q ) \mid { \mathcal { F } } _ { n - 1 } ] = 0$ for every $\mathcal { F } _ { n - 1 }$ -measurable �. For each coordinate,

$$
\begin{array} { l } { \displaystyle | M _ { n } ( Q ) ( s , a ) | \leq \left| r ( s , a , Y _ { s , a } ^ { n } ) - \sum _ { s ^ { \prime } \in S } p ( s , a , s ^ { \prime } ) r ( s , a , s ^ { \prime } ) \right| + \left| V _ { Q } ( Y _ { s , a } ^ { n } ) - \sum _ { s ^ { \prime } } p ( s , a , s ^ { \prime } ) V _ { Q } ( s ^ { \prime } ) \right| } \\ { \displaystyle \qquad \leq 1 + \left\| V _ { Q } \right\| _ { \mathrm { s p } } } \\ { \displaystyle \qquad \overset { ( a ) } { \leq } 1 + 3 D \left( 1 + \left\| Q - \bar { H } Q \right\| _ { \mathrm { s p } } \right) } \\ { \displaystyle \qquad \overset { ( b ) } { \leq } 2 ( 1 + 3 D ) \left( 1 + \left\| Q - \bar { H } Q \right\| _ { \infty } \right) } \\ { \displaystyle \qquad \overset { ( c ) } { \leq } 2 \sqrt { 2 } \left( 1 + 3 D \right) \sqrt { 1 + \left\| Q - \bar { H } Q \right\| _ { \infty } ^ { 2 } } , } \end{array}
$$

where (�) follows from Lemma 4, (�) follows from the fact that $\| \cdot \| _ { \mathrm { s p } } \leq 2 \| \cdot \| _ { \infty }$ , and (�) follows from $( 1 + t ) ^ { 2 } \leq 2 ( 1 + t ^ { 2 } )$ . This pathwise bound implies the conditional second-moment bound. □

## 4.4 Proofs of the Rates and Sample Complexity

We now apply Theorem 2 using Propositions 1 and 2. By the discussion after (7) and Lemma 3,

$$
\Delta _ { n } \leq \frac { \Delta _ { n } ^ { 2 } } { \Delta _ { \mathrm { p o l } } } \leq \frac { \left. Q _ { n } - B Q _ { n } \right. _ { \mathrm { s p } } ^ { 2 } } { \Delta _ { \mathrm { p o l } } } \leq \frac { 4 } { \Delta _ { \mathrm { p o l } } } \left. \bar { Q } _ { n } - \bar { H } \bar { Q } _ { n } \right. _ { \infty } ^ { 2 } .
$$

Thus, a mean-squared sup-norm residual bound for the auxiliary iteration gives an expected policy guarantee from every initial state.

Proof of Theorem 1. Apply Theorem 2 on $( \mathbb { R } ^ { S \times \mathcal { A } } , \| \cdot \| _ { \infty } )$ with $T = { \bar { H } }$ and $x _ { n } = { \bar { Q } } _ { n }$ . The map is nonexpansive and has a non-empty fixed-point set by Section 4.1, and (34) is the required KM iteration. By Proposition 1, the map is linearly regular with constant $K \geq 1$ . Proposition 2 verifies Assumption 3 with noise constant $2 \sqrt { 2 } \left( 1 + 3 D \right)$ , and the sup-norm martingale constant stated in Section 3.1 gives (14) with constant $2 e \log ( 2 | S | | \mathcal { A } | )$ . Substituting these three constants into (16) yields (9) and also gives

$$
\nu \sigma ^ { 2 } ( L + 2 ) ^ { 2 } = 2 ^ { 6 } e ( 2 ^ { 6 } K ^ { 2 } + 1 ) ^ { 2 } ( 1 + 3 D ) ^ { 2 } \log ( 2 | S | | \mathcal { A } | ) .
$$

Since $\bar { Q } _ { 0 } = Q _ { 0 }$ , the initial distance in Theorem 2 is $\mathcal { E } ( Q _ { 0 } )$ . Multiplying (17) by 4 and using Lemma 3 gives

$$
\begin{array} { r l r } {  { \mathbb { E } \| Q _ { n } - B Q _ { n } \| _ { \mathrm { s p } } ^ { 2 } \le 1 0 0 \mathcal { E } ( Q _ { 0 } ) ^ { 2 } \exp ( - \frac { \log ( 4 ) } { 2 ^ { 7 } K ^ { 2 } + \frac { 1 } { 2 } } \sum _ { k = 1 } ^ { n } \alpha _ { k } ) } } \\ & { } & { \qquad + 4 5 \cdot 2 ^ { 8 } e ( 2 ^ { 6 } K ^ { 2 } + 1 ) ^ { 2 } ( 1 + 3 D ) ^ { 2 } \log ( 2 | S | | \mathcal { R } | ) \alpha _ { n + 1 } . } \end{array}
$$

Dividing by $\Delta _ { \mathrm { p o l } }$ and using the preceding policy-loss bound and $\alpha _ { n + 1 } = c ( n + K _ { 0 } + 1 ) ^ { - a }$ proves (10).

For part (i), let $a \in ( 0 , 1 )$ . In $( 9 ) , c K _ { 0 } ^ { 1 - a }$ increases to infinity with $K _ { 0 }$ and $\alpha _ { 1 } = c / ( K _ { 0 } + 1 ) ^ { a }$ decreases to zero, so both inequalities hold for every suficiently large $K _ { 0 }$ . Since $t \mapsto t ^ { - a }$ is decreasing,

$$
\sum _ { k = 1 } ^ { n } \alpha _ { k } \geq \int _ { K _ { 0 } + 1 } ^ { n + K _ { 0 } + 1 } c t ^ { - a } d t = { \frac { c } { 1 - a } } \left[ ( n + K _ { 0 } + 1 ) ^ { 1 - a } - ( K _ { 0 } + 1 ) ^ { 1 - a } \right] .
$$

Thus, the transient in (10) decays faster than every negative power of �, while its last term is a constant multiple of $( n + K _ { 0 } + 1 ) ^ { - a } \leq n ^ { - a }$ . Hence $n ^ { a }$ times the right-hand side of (10) is bounded over $n \geq 1$ , and its supremum is an admissible �.

For part (ii), let $a = 1$ and $c = 2 ^ { 8 } K ^ { 2 }$ . The same integral comparison gives $\begin{array} { r } { \sum _ { k = 1 } ^ { n } \alpha _ { k } \geq c \log ( ( n + K _ { 0 } + } \end{array}$ $1 ) / ( K _ { 0 } + 1 ) )$ , so the first term of (10) is at most $1 0 0 \mathcal { E } ( Q _ { 0 } ) ^ { 2 } ( ( K _ { 0 } + 1 ) / ( n + K _ { 0 } + 1 ) ) ^ { p } / \Delta _ { \mathrm { p o l } }$ . Moreover, $\alpha _ { n + 1 } = 2 ^ { 8 } K ^ { 2 } / ( n + K _ { 0 } + 1 )$ . Substituting both into (10) gives (11). Since $p$ increases with $K \geq 1$ , we have $p \ge 2 ^ { 8 } \log ( 4 ) / ( 2 ^ { 7 } + 1 / 2 ) > 1$ □

Proof of Corollary 1. $\mathbf { B y } ( 1 1 ) , \mathbb { E } [ \Delta _ { n } ] \le \epsilon$ whenever both of its terms are at most $\epsilon / 2$ , that is, whenever

$$
n + K _ { 0 } + 1 \geq \operatorname* { m a x } \left\{ \frac { 4 5 \cdot 2 ^ { 1 7 } e K ^ { 2 } ( 2 ^ { 6 } K ^ { 2 } + 1 ) ^ { 2 } ( 1 + 3 D ) ^ { 2 } \log ( 2 | S | | \mathcal { A } | ) } { \Delta _ { \mathrm { p o l } } \epsilon } , \ : ( K _ { 0 } + 1 ) \left( \frac { 2 0 0 \mathcal { E } ( Q _ { 0 } ) ^ { 2 } } { \Delta _ { \mathrm { p o l } } \epsilon } \right) ^ { 1 / P } \right\} .
$$

The first term ofthe maximum is $O ( K ^ { 6 } ( 1 + D ) ^ { 2 } \log ( 2 | S | | \mathcal { A } | ) / ( \Delta _ { \mathrm { p o l } } \epsilon ) )$ . Since $p > 1 , ( 2 0 0 \mathcal { E } ( Q _ { 0 } ) ^ { 2 } / ( \Delta _ { \mathrm { p o l } } \epsilon ) ) ^ { 1 / p } \leq$ $1 + 2 0 0 \mathcal { E } ( Q _ { 0 } ) ^ { 2 } / ( \Delta _ { \mathrm { p o l } } \epsilon )$ , so the second term is $O ( K _ { 0 } ( 1 + \mathcal { E } ( Q _ { 0 } ) ^ { 2 } / ( \Delta _ { \mathrm { p o l } } \epsilon ) ) )$ . This proves the suficient condition on �.

With $a = 1$ and $c = 2 ^ { 8 } K ^ { 2 }$ , the first inequality in (9) holds for every $K _ { 0 } .$ , and the second, in which $\alpha _ { 1 } = c / ( K _ { 0 } + 1 )$ ), holds if

$$
K _ { 0 } + 1 \geq 2 ^ { 8 } K ^ { 2 } \operatorname* { m a x } \left\{ 2 , 2 ^ { 1 4 } e ( 1 + 3 D ) ^ { 2 } ( 2 ^ { 6 } K ^ { 2 } + 1 ) ^ { 2 } \log ( 2 | S | | \mathcal { A } | ) \right\} .
$$

Hence the smallest admissible $K _ { 0 }$ is $O ( K ^ { 2 } + K ^ { 6 } ( 1 + D ) ^ { 2 } \log ( 2 | S | | \mathcal { A } | ) )$ . The initial-distance estimate follows from (35), and each synchronous iteration uses $| S | | \mathcal { A } |$ transition samples. □

## 5 Conclusion

For each fixed finite weakly communicating MDP, we show that the expected suboptimality of the greedy policy produced by synchronous RVI �-learning decreases at a rate of $O ( n ^ { - 1 } )$ . The analysis exploits the linear

regularity property of the relevant operator to obtain the convergence rate of $O ( n ^ { - 1 } )$ for the mean-squared residual, and the positive policy gap of finite MDPs enables the same rate for the suboptimality of the greedy policy with respect to the current � iterate.

Sharpening the current parameter dependence or proving that the dependence is unimprovable remains unaddressed. Extending the analysis to high-probability guarantees would further broaden the scope of the results. The convergence rate of asynchronous RVI �-learning remains an open direction for future work.

## Disclosure of AI Use

The authors independently formulated the noise condition under which the conditional second moment grows with the squared residual and established that RVI Q-learning satisfies this condition (Proposition 2). Subsequently, GPT-6 Astra (OpenAI) suggested the linear regularity condition and provided preliminary proofs supporting the improved convergence rate. AI tools were also used to assist with writing and editing the manuscript. The authors have independently verified all results, proofs, and references and take full responsibility for the content of the paper.

## A Supporting Lemmas

This appendix proves the two lemmas used in Section 4.2. Lemma 4 bounds the span of a �-function by its residual, and Lemma 5 constructs the vector � used in the proof of Proposition 1. We use the maps ${ \bar { G } } , F$ and $F _ { \pi }$ and the transition matrices $P ^ { \pi }$ defined in that proof.

Lemma 4. Let Assumption 1 hold. For every $Q \in \mathbb { R } ^ { S \times \mathcal { A } }$

$$
\left. V _ { Q } \right. _ { \mathrm { s p } } \leq 3 D \left( 1 + \left. Q - \bar { H } Q \right. _ { \mathrm { s p } } \right) , \quad \left. Q \right. _ { \mathrm { s p } } \leq \left( 1 + 3 D \right) \left( 1 + \left. Q - \bar { H } Q \right. _ { \mathrm { s p } } \right) .
$$

In particular, $\| Q \| _ { \mathrm { s p } } \leq 1 + 3 D$ for every $Q \in \operatorname { F i x } ( { \bar { H } } )$

Proof. Decompose the residual into a scalar and a small part,

$$
Q - B Q = \lambda { \bf 1 } + \zeta , \qquad \| \zeta \| _ { \infty } \leq \frac { \left\| Q - \bar { H } Q \right\| _ { \mathrm { s p } } } { 2 } ,\tag{38}
$$

where � is the midpoint of the range of $Q - B Q$ . The greedy policy $\pi _ { Q }$ of Section 2.2 satisfies $Q ( s , \pi _ { Q } ( s ) ) =$ $V _ { Q } ( s )$ . Evaluating (38) at $( s , \pi _ { Q } ( s ) )$ and using (6) gives the Poisson equation with zero gain

$$
V _ { Q } = P ^ { \pi _ { Q } } V _ { Q } + h , \qquad h ( s ) : = \overline { { { r } } } \left( s , \pi _ { Q } ( s ) \right) + \lambda + \zeta \left( s , \pi _ { Q } ( s ) \right) ,\tag{39}
$$

where $P ^ { \pi }$ denotes the transition matrix induced by �. Let $\mu _ { \pi _ { Q } }$ be a stationary distribution corresponds to $P ^ { \pi _ { Q } }$ . Multiplying (39) by $\mu _ { \pi _ { O } } ^ { \top }$ gives $\mu _ { \pi _ { O } } ^ { \top } h = 0$ , that is, $\lambda = { ^ { - } \mu } _ { \pi _ { Q } } ^ { \top } \overline { { r } } ( \cdot , \pi _ { Q } ( \cdot ) ) - \mu  _ { \pi _ { Q } } ^ { \top } \zeta ( \cdot , \pi _ { Q } ( \cdot ) )$ . Since $\overline { { r } } \in [ 0 , 1 ]$ and $\left\| \zeta \right\| _ { \infty } \leq \left\| Q - \bar { H } Q \right\| _ { \mathrm { s p } } / 2$

$$
- 1 - \frac { \left\| \boldsymbol { Q } - \boldsymbol { \bar { H } Q } \right\| _ { \mathrm { s p } } } { 2 } \leq \lambda \leq \frac { \left\| \boldsymbol { Q } - \boldsymbol { \bar { H } Q } \right\| _ { \mathrm { s p } } } { 2 } , \quad \mathrm { a n d ~ c o n s e q u e n t l y } \quad \left\| \boldsymbol { h } \right\| _ { \infty } \leq 1 + \left\| \boldsymbol { Q } - \boldsymbol { \bar { H } Q } \right\| _ { \mathrm { s p } } .\tag{40}
$$

Furthermore, $V _ { Q } ( s ) \geq Q ( s , \pi ( s ) )$ for every $\pi \in \Pi _ { \mathrm { S D } }$ , so (38) also yields the family of inequalities

$$
V _ { Q } \geq P ^ { \pi } V _ { Q } + g _ { \pi } , \qquad g _ { \pi } ( s ) : = \overline { { r } } \left( s , \pi ( s ) \right) + \lambda - \frac { \left\| Q - \bar { H } Q \right\| _ { \mathrm { s p } } } { 2 } , \qquad \pi \in \Pi _ { \mathrm { S D } } ,\tag{41}
$$

and $- g _ { \pi } \leq 1 + \left. Q - { \bar { H } } Q \right. _ { \mathrm { s p } }$ by (40). Unlike (39), which holds only for the greedy policy, (41) holds for every policy, so the policy used to travel between two states of $S _ { \mathrm { c } }$ may be chosen freely.

Let �<sub>−</sub> and $s _ { + }$ attain the minimum and the maximum of $V _ { Q }$ over $S _ { \mathrm { c } } . \mathrm { I f } s _ { - } = s _ { + }$ <sub>+</sub> the range over $S _ { \mathrm { c } }$ is zero. Otherwise pick $\pi \in \Pi _ { \mathrm { S D } }$ attaining min $\mathbf { \sigma } _ { \pi ^ { \prime } } \mathbb { E } _ { s _ { - } } ^ { \pi ^ { \prime } } [ \tau _ { s _ { + } } ]$ , which is at most $D$ by (2), and apply optional stopping to (41) at $\tau _ { s _ { + } }$ :

$$
V _ { Q } ( s _ { - } ) \geq \mathbb { E } _ { s _ { - } } ^ { \pi } \left[ V _ { Q } ( S _ { \tau _ { s _ { + } } } ) \right] + \mathbb { E } _ { s _ { - } } ^ { \pi } \left[ \sum _ { t = 0 } ^ { \tau _ { s _ { + } } - 1 } g _ { \pi } ( S _ { t } ) \right] \geq V _ { Q } ( s _ { + } ) - \mathbb { E } _ { s _ { - } } ^ { \pi } [ \tau _ { s _ { + } } ] \left( 1 + \left. Q - \bar { H } Q \right. _ { \mathrm { s p } } \right) ,
$$

so that max<sub>S</sub> �<sub>�</sub> − min<sub>S</sub> $V _ { Q } \leq D ( 1 + \left. Q - { \bar { H } } Q \right. _ { \mathrm { s } \mathrm { { D } } } )$ . For $s \in S _ { \mathrm { t } }$ , apply optional stopping to the equality (39) at $\tau _ { S _ { \mathrm { c } } }$ under $\pi _ { Q }$ , whose expectation is at most � by (2):

$$
\left. V _ { Q } ( s ) - \mathbb { E } _ { s } ^ { \pi _ { Q } } \left[ V _ { Q } \left( S _ { \tau _ { S _ { c } } } \right) \right] \right. = \left. \mathbb { E } _ { s } ^ { \pi _ { Q } } \left[ \sum _ { t = 0 } ^ { \tau _ { S _ { c } } - 1 } h ( S _ { t } ) \right] \right. \leq D \left. \left. h \right. \right. _ { \infty } \leq D ( 1 + \left. \mathcal { Q } - \bar { H } Q \right. _ { \mathrm { s p } } ) .
$$

Since $S _ { \tau _ { S _ { \mathrm { ~ } } } } \in S _ { \mathrm { c } }$ , the expectation on the left lies between the minimum and the maximum of $V _ { Q }$ over $S _ { \mathrm { c } }$ whence $\left\| \tilde { V } _ { Q } \right\| _ { \mathrm { s p } } \leq D ( 1 + \left\| Q - \bar { H } Q \right\| _ { \mathrm { s p } } ) + 2 D ( 1 + \left\| Q - \bar { H } Q \right\| _ { \mathrm { s p } } ) = 3 D ( 1 + \left\| Q - \bar { H } Q \right\| _ { \mathrm { s p } } )$ . Both stopping arguments are justified by first stopping at the hitting time truncated at � and letting $m  \infty { : }$ S is finite, so $V _ { Q } , { \bar { r } } ,$ , and ℎ are bounded, and the hitting times used have finite means under Assumption 1.

Finally, by (6) the range of $B Q$ is at most the range of ${ \mathit { \bar { r } } } ,$ which is at most 1, plus the range of $V _ { Q }$ , so

$$
\begin{array} { r l } & { \left\| Q \right\| _ { \mathrm { s p } } \leq \left\| B Q \right\| _ { \mathrm { s p } } + \left\| Q - \bar { H } Q \right\| _ { \mathrm { s p } } } \\ & { \qquad \leq 1 + \left\| V _ { Q } \right\| _ { \mathrm { s p } } + \left\| Q - \bar { H } Q \right\| _ { \mathrm { s p } } } \\ & { \qquad \leq 1 + 3 D ( 1 + \left\| Q - \bar { H } Q \right\| _ { \mathrm { s p } } ) + \left\| Q - \bar { H } Q \right\| _ { \mathrm { s p } } } \\ & { \qquad = ( 1 + 3 D ) ( 1 + \left\| Q - \bar { H } Q \right\| _ { \mathrm { s p } } ) . } \end{array}
$$

For $z \in \operatorname { F i x } ( { \bar { H } } )$ we have $\left\| z - { \bar { H } } z \right\| _ { \mathrm { s p } } = \left\| z - { \bar { H } } z \right\| _ { \infty } = 0$ by the fixed-point identity $z = { \bar { H } } z ,$ so $\| z \| _ { \mathrm { s p } } ~ \leq$ $1 + 3 D$ □

For the next lemma and its proof, we call a vector $w \in \mathbb { R } ^ { S }$ a super-solution if $F w \leq w$ , equivalently, $V _ { \bar { G } w } \leq w$ , with inequalities understood coordinatewise.

Lemma 5 (Minimal super-solution). Let Assumption 1 hold, and let $\nu \in \mathbb { R } ^ { S }$ satisfy $\left\| V _ { \bar { G } \nu } - \nu \right\| _ { \infty } \leq \rho$ for some $0 \leq \rho < \Delta _ { \mathrm { p o l } }$ . Then there is a smallest super-solution $w \geq \nu ,$ , and

$$
0 \leq w - \nu \leq \mathcal { T } \rho \mathbf { 1 } .\tag{42}
$$

Proof. Step 1: $\operatorname { I f } x \leq x ^ { \prime }$ , then $\bar { G } x \leq \bar { G } x ^ { \prime }$ , hence $V _ { \bar { G } x } \leq V _ { \bar { G } x ^ { \prime } }$ . Set

$$
w _ { 0 } : = \nu , \qquad w _ { k + 1 } : = \operatorname* { m a x } \left\{ \nu , V _ { \bar { G } w _ { k } } \right\} , \quad k \geq 0 .
$$

(a) $\{ w _ { k } \}$ is non-decreasing: $w _ { 1 } = \operatorname* { m a x } \{ \nu , V _ { \bar { G } \nu } \} \geq \nu = w _ { 0 }$ , and if $w _ { k } \ge w _ { k - 1 }$ , then $w _ { k + 1 } = \operatorname* { m a x } \{ \nu , V _ { \bar { G } w _ { k } } \} \geq$ max $\{ \nu , V _ { \bar { G } w _ { k - 1 } } \} = w _ { k }$

(b) $\{ w _ { k } \}$ is bounded above: fix $Q ^ { \star } \in \operatorname { F i x } ( \bar { H } )$ , which is non-empty by Assumption 1, and set $u : = V _ { Q ^ { \star } }$ , $c : = \operatorname* { m a x } _ { s \in S } \{ \nu ( s ) - u ( s ) \}$ , and $z : = u + c { \bf 1 }$ . Then $z \geq \nu$ , and since $\bar { G } u = \bar { H } Q ^ { \star } = Q ^ { \star }$

$$
V _ { \bar { G } z } = V _ { \bar { G } u } + c \mathbf { 1 } = V _ { Q ^ { \star } } + c \mathbf { 1 } = z .
$$

Thus $w _ { 0 } \leq z ,$ , and $w _ { k } \leq z$ implies $w _ { k + 1 } \leq \operatorname* { m a x } \{ \nu , V _ { \bar { G } z } \} = z .$

By (a) and (b), each $w _ { k } ( s )$ is non-decreasing in � and bounded above by $z ( s )$ , so $w _ { k } $ � for some $w \in \mathbb { R } ^ { S }$ . Since � ↦→ max $\big \{ \nu , V _ { \bar { G } x } \big \}$ is continuous,

$$
w = \operatorname* { m a x } \left\{ \nu , V _ { \bar { G } w } \right\} .\tag{43}
$$

Hence $w ~ \geq ~ \nu$ and $w \ge V _ { \bar { G } w }$ . If $w ^ { \prime } \geq \nu$ is a super-solution, then $w _ { 0 } ~ \le ~ w ^ { \prime }$ , and $w _ { k } \ \leq \ w ^ { \prime }$ implies $w _ { k + 1 } \leq \operatorname* { m a x } \{ \nu , V _ { \bar { G } w ^ { \prime } } \} \leq w ^ { \prime }$ , so $w \leq w ^ { \prime }$ . Thus � is the smallest super-solution above �.

Step 2: Let $C _ { 0 } : = \{ s : w ( s ) = \nu ( s ) \}$ , and call � active at � if $w ( s ) = ( \bar { G } w ) ( s , a )$ . If $s \notin C _ { 0 }$ , then $w ( s ) > \nu ( s )$ , so (43) gives $w ( s ) = V _ { \bar { G } w } ( s ) = \mathrm { m a x } _ { a } ( \bar { G } w ) ( s , a )$ , and � has an active action. Let $\mathcal { G }$ be the directed graph on S with an edge $s \to s ^ { \prime } \operatorname { i f } p ( s , a , s ^ { \prime } ) > 0$ for some � active at �. We claim that every state has a path to $C _ { 0 }$ in $\mathcal { G }$ . Suppose not, and let $U \neq \emptyset$ be the set of states with no such path. Then $U \cap C _ { 0 } = \emptyset$ , and

$$
\sum _ { s ^ { \prime } \in U } p ( s , a , s ^ { \prime } ) = 1 , \qquad s \in U , \ a \mathrm { a c t i v e \ a t } \ s ,\tag{44}
$$

since an edge from � to a state outside � would give � a path to $C _ { 0 }$ . Choose $\delta > 0$ such that

$$
\delta \leq w ( s ) - \nu ( s ) , \quad s \in U , \quad \mathrm { a n d } \quad \delta \leq w ( s ) - ( \bar { G } w ) ( s , a ) , \quad s \in U , \ a \mathrm { n o t } \ a c t i v e \ a t \ s .
$$

The first inequality is possible because $U \cap C _ { 0 } = \emptyset$ , and the second inequality is possible because $w \geq V _ { \bar { G } w }$ and � is not active. Let $\tilde { w } : = w - \delta \mathbb { 1 } _ { U }$ . Then $\tilde { w } \geq \nu$ and $\begin{array} { r } { ( \bar { G } \tilde { w } ) ( s , a ) = ( \bar { G } w ) ( s , a ) - \delta \sum _ { s ^ { \prime } \in U } p ( s , a , s ^ { \prime } ) } \end{array}$ , so, using (44) in the second line,

$$
\begin{array} { r l } { s \notin U : } & { ( \bar { G } \tilde { w } ) ( s , a ) \le ( \bar { G } w ) ( s , a ) \le w ( s ) = \tilde { w } ( s ) , } \\ { \quad s \in U , a \mathrm { ~ a c t i v e : } } & { ( \bar { G } \tilde { w } ) ( s , a ) = ( \bar { G } w ) ( s , a ) - \delta = w ( s ) - \delta = \tilde { w } ( s ) , } \\ { s \in U , a \mathrm { ~ n o t ~ a c t i v e : } } & { ( \bar { G } \tilde { w } ) ( s , a ) \le ( \bar { G } w ) ( s , a ) \le w ( s ) - \delta = \tilde { w } ( s ) . } \end{array}
$$

Hence $\tilde { w } \geq V _ { \bar { G } \tilde { w } }$ is a super-solution above � with $\tilde { w } < w$ on $U _ { : }$ , which contradicts the minimality of $w .$

Step 3: Let $d ( s )$ be the number of edges of a shortest path from � to $C _ { 0 }$ in $\mathcal { G }$ , so that $d = 0$ on $C _ { 0 }$ and $1 \leq d ( s ) \leq | S | - 1$ otherwise. For $s \notin C _ { 0 } .$ , pick an edge $s  s ^ { \prime }$ with $d ( s ^ { \prime } ) = d ( s ) - 1$ , and let ${ \widehat { \pi } } ( s )$ be an action active at � with $p ( s , { \widehat { \pi } } ( s ) , s ^ { \prime } ) > 0$ . On $C _ { 0 } .$ , choose $\widehat { \pi } ( s ) \in \arg \operatorname* { m a x } _ { a } ( \bar { G } w ) ( s , a )$ . Then $\widehat { \pi } \in \Pi _ { \mathrm { S D } }$ and, with $p _ { \operatorname* { m i n } } : = \operatorname* { m i n } \{ p ( s , a , s ^ { \prime } ) : p ( s , a , s ^ { \prime } ) > 0 \}$ ,

$$
\begin{array} { r } { \mathbb { P } _ { s } ^ { \widehat { \pi } } \left( \tau _ { C _ { 0 } } \leq | S | \right) \geq p _ { \operatorname* { m i n } } ^ { | S | } = : \beta > 0 , \qquad s \in S . } \end{array}
$$

By the Markov property, $\mathbb { P } _ { s } ^ { \widehat { \pi } } ( \tau _ { C _ { 0 } } > m | S | ) \leq ( 1 - \beta ) ^ { m }$ for $m \geq 1$ . Hence $\mathbb { P } _ { s } ^ { \widehat { \pi } } ( \tau _ { C _ { 0 } } < \infty ) = 1$

To apply (4), we show that b� is optimal from every state. Outside $C _ { 0 }$ , its actions are active, so $F _ { \widehat { \pi } } w = w$ On $C _ { 0 } .$ , monotonicity and $w = \nu$ give

$$
w \ge F _ { \widehat { \pi } } w = V _ { \bar { G } w } \ge V _ { \bar { G } \nu } \ge \nu - \rho { \bf 1 } = w - \rho { \bf 1 } .
$$

Thus $0 \leq w - F _ { \widehat { \pi } } w \leq \rho \mathbf { 1 }$ on all of S. Telescoping this inequality along trajectories under ${ \widehat { \pi } } ,$ dividing by the horizon, and using boundedness of � yields

$$
0 \le J ^ { \star } - J ( \widehat \pi , s ) \le \rho < \Delta _ { \mathrm { p o l } } , \qquad s \in S .
$$

By (3), b� is optimal from every state. Consequently, (4) gives

$$
\mathbb { E } _ { s } ^ { \widehat { \pi } } [ \tau _ { C _ { 0 } } ] \leq \mathcal { T } , \qquad s \in S .\tag{45}
$$

Step 4: Write $\tau : = \tau _ { C _ { 0 } } , \mathbb { E } _ { s } : = \mathbb { E } _ { s } ^ { \widehat { \pi } }$ , and $g ( s ) : = \overline { { r } } ( s , \widehat { \pi } ( s ) ) - J ^ { \star }$ , so that $| g | \le 1$ because $\overline { { r } } , J ^ { \star } \in [ 0 , 1 ]$ Since ${ \widehat { \pi } } ( s )$ is active at � ∉ �<sub>0</sub>, and $( { \bar { G } } \nu ) ( s , a ) - \nu ( s ) \leq \rho$ for all $( s , a )$

$$
w ( s ) = g ( s ) + ( P ^ { \widehat { \pi } } w ) ( s ) , \quad s \notin C _ { 0 } , \qquad \nu ( s ) \geq g ( s ) - \rho + ( P ^ { \widehat { \pi } } \nu ) ( s ) , \quad s \in S .
$$

Fix $k \geq 1$ . For $t < k$ , the event $\{ t < \tau \}$ depends only on $S _ { 0 } , \ldots , S _ { t }$ , and $S _ { t } \notin C _ { 0 }$ on it. By the Markov property,

$$
\begin{array} { r l } & { \mathbb { E } _ { s } \left[ \left( w ( S _ { t } ) - w ( S _ { t + 1 } ) \right) \mathbb { 1 } _ { \{ t < \tau \} } \right] = \mathbb { E } _ { s } \left[ g ( S _ { t } ) \mathbb { 1 } _ { \{ t < \tau \} } \right] , } \\ & { \mathbb { E } _ { s } \left[ \left( \nu ( S _ { t } ) - \nu ( S _ { t + 1 } ) \right) \mathbb { 1 } _ { \{ t < \tau \} } \right] \geq \mathbb { E } _ { s } \left[ \left( g ( S _ { t } ) - \rho \right) \mathbb { 1 } _ { \{ t < \tau \} } \right] . } \end{array}
$$

Summing over $t = 0 , \ldots , k - 1$ , the left-hand sides telescope:

$$
w ( s ) = \mathbb { E } _ { s } \left[ \sum _ { t = 0 } ^ { \tau \wedge k - 1 } g ( S _ { t } ) + w ( S _ { \tau \wedge k } ) \right] , \qquad \nu ( s ) \geq \mathbb { E } _ { s } \left[ \sum _ { t = 0 } ^ { \tau \wedge k - 1 } ( g ( S _ { t } ) - \rho ) + \nu ( S _ { \tau \wedge k } ) \right] .
$$

Let $k \to \infty$ . The sums are bounded in absolute value by $( 1 + \rho ) \tau$ , which is integrable by (45), � and � are bounded, and $\tau <$ ∞ almost surely. By dominated convergence, and since $w ( S _ { \tau } ) = \nu ( S _ { \tau } )$ because $S _ { \tau } \in C _ { 0 }$

$$
w ( s ) = \mathbb { E } _ { s } \left[ \sum _ { t = 0 } ^ { \tau - 1 } g ( S _ { t } ) + \nu ( S _ { \tau } ) \right] , \qquad \nu ( s ) \ge \mathbb { E } _ { s } \left[ \sum _ { t = 0 } ^ { \tau - 1 } g ( S _ { t } ) + \nu ( S _ { \tau } ) \right] - \rho \mathbb { E } _ { s } [ \tau ] .
$$

Subtracting and using (45) gives $w ( s ) - \nu ( s ) \leq \rho \mathbb { E } _ { s } [ \tau ] \leq \mathcal { T } \rho$ . Together with $w \geq \nu$ from Step 1, this proves (42). □

## B A Martingale Inequality in the Sup-Norm

Condition (14) holds on any finite-dimensional normed space, with a constant obtained from equivalence with a Euclidean norm. For the sup-norm, that yields $\nu = d .$ The following lemma establishes that $\nu = O ( \log ( d ) )$ , which is used in Theorem 1. It is the sup-norm form of the smooth-ℓ<sup>�</sup> estimate used by Chen et al. [9].

Lemma 6 (Martingale constant in the sup-norm). Let $d \geq 2 ,$ , let $\chi = \mathbb { R } ^ { d }$ be equipped with $\| \cdot \| _ { \infty } ,$ , and let $\{ U _ { i } \}$ be a finite X-valued martingale diference sequence. Then,

$$
\mathbb { E } \left. \sum _ { i } U _ { i } \right. ^ { 2 } \leq 2 e \log ( 2 d ) \sum _ { i } \mathbb { E } \left. U _ { i } \right. ^ { 2 } .
$$

Proof. Let $p \ge 2$ . The function $\begin{array} { r } { x \mapsto \frac { 1 } { 2 } \left. x \right. _ { p } ^ { 2 } } \end{array}$ is convex and $( p - 1 )$ -smooth with respect to $\left\| \cdot \right\| _ { p } \left[ 3 \right]$ , so the descent lemma applied along the partial sums, followed by conditioning and telescoping, gives the standard estimate

$$
\mathbb { E } \left. \sum _ { i } U _ { i } \right. _ { p } ^ { 2 } \leq ( p - 1 ) \sum _ { i } \mathbb { E } \left. U _ { i } \right. _ { p } ^ { 2 } .
$$

Combining it with $\left\| x \right\| _ { \infty } \leq \left\| x \right\| _ { p } \leq d ^ { 1 / p } \left\| x \right\| _ { \infty }$ yields

$$
\mathbb { E } \left\| \sum _ { i } U _ { i } \right\| _ { \infty } ^ { 2 } \leq \left( p - 1 \right) d ^ { 2 / p } \sum _ { i } \mathbb { E } \left\| U _ { i } \right\| _ { \infty } ^ { 2 } .
$$

Choose $p = \operatorname* { m a x } \{ 2 , 2 \log ( d ) \}$ . If $d \ge e$ then $d ^ { 2 / p } = d ^ { 1 / \log ( d ) } = e$ and the prefactor is $( 2 \log ( d ) - 1 ) e ~ \leq$ $2 e \log ( 2 d ) ;$ if $\textit { d } < \textit { e }$ , that is $d \ = \ 2$ , the prefactor is $( 2 - 1 ) \cdot 2 = 2 \leq 2 e \log ( 2 d )$ . In either case $\begin{array} { r } { \mathbb E \left\| \sum _ { i } U _ { i } \right\| _ { \infty } ^ { 2 } \leq 2 e \log ( 2 d ) \sum _ { i } \mathbb E \left\| U _ { i } \right\| _ { \infty } ^ { 2 } . } \end{array}$ , which proves the claim. □

## C Stepsize Estimates

The following elementary estimates are the only properties of the stepsize sequence (15) used in Section 3. Bounds (46) and (47) convert the accumulated stepsize into an explicit rate; bounds (48) and (49) are what keep the blocks of (18) from growing by more than a factor two on the shifted clock $n \mapsto n + K _ { 0 }$

Lemma 7 (Stepsize sums and block growth). Let the stepsizes satisfy Assumption 4. $H a \in ( 0 , 1 )$ , then for every $n \geq 0$

$$
\sum _ { k = 1 } ^ { n } \alpha _ { k } \geq \frac { c } { 1 - a } \left[ ( n + K _ { 0 } + 1 ) ^ { 1 - a } - ( K _ { 0 } + 1 ) ^ { 1 - a } \right] ,\tag{46}
$$

while $i f a = 1$ , then for every $n \geq 0$

$$
\sum _ { k = 1 } ^ { n } \alpha _ { k } \geq c \log \left( \frac { n + K _ { 0 } + 1 } { K _ { 0 } + 1 } \right) .\tag{47}
$$

If in addition the first inequality in (16) holds, then for every integer $m \geq 0$

$$
\sum _ { k = m + 1 } ^ { 2 m + K _ { 0 } } \alpha _ { k } \geq \frac { c K _ { 0 } ^ { 1 - a } } { 2 } \geq L .\tag{48}
$$

Finally, for all integers $m \leq n \leq 2 m + K _ { 0 }$

$$
\alpha _ { m + 1 } \leq 2 ^ { a } \alpha _ { n + 1 } \leq 2 \alpha _ { n + 1 } .\tag{49}
$$

Proof. Since $t \mapsto t ^ { - a }$ is decreasing, $\begin{array} { r } { c ( k + K _ { 0 } ) ^ { - a } \geq \int _ { k + K _ { 0 } } ^ { k + K _ { 0 } + 1 } c t ^ { - a } d t } \end{array}$ , and summing over $k = 1 , \ldots , n$ gives

$$
\sum _ { k = 1 } ^ { n } \alpha _ { k } \geq \int _ { K _ { 0 } + 1 } ^ { n + K _ { 0 } + 1 } c t ^ { - a } d t ,
$$

which is (46) for $a < 1$ and (47) for $a = 1$

For (48), the indices $k \in \{ m + 1 , \ldots , 2 m + K _ { 0 } \}$ satisfy $k + K _ { 0 } \le 2 ( m + K _ { 0 } )$ , so each of these $m + K _ { 0 }$ stepsizes is at least $c \ ( 2 ( m + K _ { 0 } ) ) ^ { - a }$ and

$$
\sum _ { k = m + 1 } ^ { 2 m + K _ { 0 } } \alpha _ { k } \ge \frac { c ( m + K _ { 0 } ) } { ( 2 ( m + K _ { 0 } ) ) ^ { a } } = \frac { c ( m + K _ { 0 } ) ^ { 1 - a } } { 2 ^ { a } } \ge \frac { c K _ { 0 } ^ { 1 - a } } { 2 } \ge L ,
$$

using $m + K _ { 0 } \ge K _ { 0 }$ together with $1 - a \geq 0$ , and $2 ^ { a } \leq 2$

For (49), the hypothesis $n \leq 2 m + K _ { 0 }$ gives $n + 1 + K _ { 0 } \le 2 m + 2 K _ { 0 } + 1 < 2 ( m + 1 + K _ { 0 } )$ , hence

$$
\frac { \alpha _ { m + 1 } } { \alpha _ { n + 1 } } = \left( \frac { n + 1 + K _ { 0 } } { m + 1 + K _ { 0 } } \right) ^ { a } \leq 2 ^ { a } \leq 2 . ~ \boxed { 2 }
$$

## References

[1] Jinane Abounadi, Dimitri B. Bertsekas, and Vivek S Borkar. Learning algorithms for Markov decision processes with average cost. SIAM Journal on Control and Optimization, 40(3):681–698, 2001.

[2] Heinz H Bauschke, Dominikus Noll, and Hung M Phan. Linear and strong convergence of algorithms involving averaged nonexpansive operators. Journal ofMathematical Analysis and Applications, 421(1):1–20, 2015.

[3] Amir Beck. First-order methods in optimization. SIAM, 2017.

[4] Dimitri P. Bertsekas and John N. Tsitsiklis. An analysis of stochastic shortest path problems. Mathematics of Operations Research, 16(3):580–595, 1991.

[5] Ethan Blaser and Shangtong Zhang. Asymptotic and finite sample analysis of nonexpansive stochastic approximations with Markovian noise. Proceedings ofthe AAAI Conference on Artificial Intelligence, 2026.

[6] Mario Bravo and Roberto Cominetti. Stochastic fixed-point iterations for nonexpansive maps: Convergence and error bounds. SIAM Journal on Control and Optimization, 62(1):191–219, 2024.

[7] Siddharth Chandak, Vivek S. Borkar, and Parth Dodhia. Concentration of contractive stochastic approximation and reinforcement learning. Stochastic Systems, 12(4):411–430, 2022.

[8] Zaiwei Chen. Non-asymptotic guarantees for average-reward q-learning with adaptive stepsizes. Advances in Neural Information Processing Systems, 38:147194–147249, 2026.

[9] Zaiwei Chen, Siva Theja Maguluri, Sanjay Shakkottai, and Karthikeyan Shanmugam. Finite-sample analysis of contractive stochastic approximation using smooth convex envelopes. Advances in Neural Information Processing Systems, 33:8223–8234, 2020.

[10] Zaiwei Chen, Sheng Zhang, Jimmy Zhang, Shaan Ul Haque, and Siva Theja Maguluri. A nonasymptotic theory of seminorm lyapunov stability: From deterministic to stochastic iterative algorithms. Mathematics of Operations Research, 2026. doi: 10.1287/moor.2025.0920.

[11] Zijun Chen, Zaiwei Chen, Nian Si, and Shengbo Wang. Achieving $\epsilon ^ { - 2 }$ dependence for average-reward Q-learning with a new contraction principle. arXiv preprint arXiv:2601.21301, 2026.

[12] Roberto Cominetti, José A. Soto, and José Vaisman. On the rate of convergence of Krasnosel’ski˘ı–Mann iterations and their connection with sums of Bernoulli random variables. Israel Journal ofMathematics, 199(2):757–772, 2014.

[13] Ying Jin, Ramki Gummadi, Zhengyuan Zhou, and Jose Blanchet. Feasible Q-learning for average reward reinforcement learning. In Proceedings ofthe 27th International Conference on Artificial Intelligence and Statistics, volume 238 of Proceedings ofMachine Learning Research, pages 1630–1638. PMLR, 2024.

[14] Jongmin Lee, Mario Bravo, and Roberto Cominetti. Near-optimal sample complexity for MDPs via anchoring. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267, pages 32907–32929. PMLR, 2025.

[15] Gen Li, Changxiao Cai, Yuxin Chen, Yuting Wei, and Yuejie Chi. Is Q-learning minimax optimal? a tight sample complexity analysis. Operations Research, 72(1):222–236, 2024.

[16] Martin L Puterman. Markov decision processes: discrete stochastic dynamic programming. John Wiley & Sons, 2014.

[17] Paul J Schweitzer and Awi Federgruen. The functional equations of undiscounted Markov renewal programming. Mathematics ofOperations Research, 3(4):308–321, 1978.

[18] R. Srikant and Lei Ying. Finite-time error bounds for linear stochastic approximation and TD learning. In Proceedings of the Thirty-Second Conference on Learning Theory, volume 99 of Proceedings of Machine Learning Research, pages 2803–2830. PMLR, 2019.

[19] Martin J. Wainwright. Stochastic approximation with cone-contractive operators: Sharp ℓ<sub>∞</sub>-bounds for Q-learning. arXiv preprint arXiv:1905.06265, 2019.

[20] Yi Wan, Abhishek Naik, and Richard S. Sutton. Learning and planning in average-reward Markov decision processes. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, pages 10653–10662. PMLR, 2021.

[21] Yi Wan, Huizhen Yu, and Richard S. Sutton. On convergence of average-reward Q-learning in weakly communicating Markov decision processes. arXiv preprint arXiv:2408.16262, 2024.

[22] Sheng Zhang, Zhe Zhang, and Siva Theja Maguluri. Finite sample analysis of average-reward TD learning and Q-learning. In Advances in Neural Information Processing Systems, volume 34, pages 1230–1242, 2021.

[23] Matthew Zurek and Yudong Chen. Span-based optimal sample complexity for weakly communicating and general average reward MDPs. Advances in Neural Information Processing Systems, 37, 2024