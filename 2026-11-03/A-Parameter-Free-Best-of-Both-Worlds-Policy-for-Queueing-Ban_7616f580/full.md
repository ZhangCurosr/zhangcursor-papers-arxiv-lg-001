# A Parameter-Free Best-of-Both-Worlds Policy for Queueing Bandits

Jongyeong Lee<sup>1</sup> jongyeong@kist.re.kr

Chansoo Kim<sup>1,2</sup>

eau@kist.re.kr

<sup>1</sup> Korea Institute of Science and Technology <sup>2</sup> University of Science and Technology

## Abstract

We study online channel scheduling for a single queue served by K channels under bandit feedback. We propose a simple best-of-both-worlds policy that explores when the queue is empty. During busy periods, it selects the empirical best channel while the prescribed conditions hold and, if any condition is violated, switches to a robust bandit policy to clear the queue. For arbitrary arrivals and services determined by an oblivious adversary, our policy achieves $\tilde { \mathcal { O } } ( \sqrt { K L _ { T } } )$ worst-case queue length regret with high probability, where $L _ { T }$ is the longest busy-period length. This result improves the previous $T ^ { 3 / 4 }$ dependence and matches the worst-case lower bound up to logarithmic factors. Under bounded i.i.d. stochastic models with a unique best stabilizing channel, we show that the cumulative queue length regret converges to a finite problem-dependent constant, and further establish the logarithmic expected worst-case queue length regret bound.

## 1 Introduction

Bandit problems provide a fundamental framework for sequential decision-making with partial feedback (Lattimore and Szepesvári, 2019). In stochastic environments, the learner seeks to identify arms with high expected rewards, whereas adversarial environments require robustness without distributional assumptions. Best-of-both-worlds policies adapt to both regimes within a single policy (Bubeck and Slivkins, 2012). For classical bandits, such policies achieve logarithmic, problem-dependent regret in stochastic environments while retaining near-optimal guarantee in adversarial environments without knowing the regime in advance (Auer and Chiang, 2016; Zimmert and Seldin, 2021; Honda et al., 2023). A natural question is whether such adaptation is possible when decisions have lasting effects on a system and performance depends on the system states induced by past decisions rather than solely on the sum of immediate outcomes.

An important example of such state-dependent decision-making arises in queueing bandits, which model scheduling systems with unknown service outcomes (Krishnasamy et al., 2016, 2021). In this problem, jobs arrive sequentially, and in each round, the scheduler selects one of the channels to serve the queue and observes only the service provided by the selected one. A poor selection may leave more work unserved in the system, inducing additional backlog in subsequent rounds, whereas the accumulated effect of earlier mistakes disappears once the queue becomes empty. Therefore, standard bandit regret does not fully characterize queueing performance since the timing of service losses and the regenerative structure of the queue are both important.

A classical objective in queueing systems is stability (Dai, 1995). One commonly used notion is mean-rate stability, which requires lim $_ { T  \infty } \mathbb { E } [ Q _ { T } ^ { \pi } ] / T = 0$ , where Q<sup>π</sup> is the queue length under the policy π at round T (Neely, 2010). When service statistics are unknown, one should learn which channels are effective while controlling the queue lengths (Yang et al., 2023; Freund et al., 2024). Stability alone, however, does not quantify the additional backlog incurred by learning, since two policies may both stabilize the system while producing substantially different queue trajectories.

Table 1: Comparison of queue length regret guarantees. Here, (S) indicates that the unique best arm is stabilizing, while (E) and (P) indicate guarantees in expectation and with high-probability, respectively. The quantity $L _ { T }$ denotes the length of the longest busy period and satisfies $L _ { T } \leq T$ . The queue dynamics considered in Stahlbuhk et al. (2021), given in (8), differ from those considered in Krishnakumar and Sinha (2025) and ours, given in (1).
<table><tr><td>Metric</td><td>Reference</td><td>Assumptions</td><td>Regret</td></tr><tr><td rowspan="2"> $\operatorname* { m a x } _ { t \in [ T ] } \operatorname* { m a x } _ { i \in [ K ] } \left( Q _ { t } ^ { \pi } - Q _ { t } ^ { i } \right)$  (worst-case regret)</td><td>Krishnakumar and Sinha (2025)</td><td>oblivious  $A _ { t } \geq 0 , R _ { t } \in [ 0 , 1 ] ^ { K }$ </td><td> $\tilde { \mathcal { O } } ( \sqrt { K } T ^ { 3 / 4 } ) \left( \mathbf { P } \right)$ </td></tr><tr><td>Algorithm 1 (Theorem 3) Algorithm 1 (Corollary 5)</td><td>oblivious  $A _ { t } \geq 0 , R _ { t } \in [ 0 , 1 ] ^ { K }$  i.i.d. bounded</td><td> $\tilde { \mathcal { O } } ( \sqrt { K L _ { T } } ) \left( \mathbf { P } \right)$   $\mathcal { O } ( \log ( K T ) / \Delta _ { \operatorname* { m i n } } ) \left( \mathrm { E } \right)$ </td></tr><tr><td> $\textstyle \sum _ { t } \mathbb { E } [ Q _ { t } ^ { \pi } - Q _ { t } ^ { i ^ { * } } ]$ </td><td>Stahlbuhk et al. (2021)</td><td> $A _ { t } , R _ { t } , ( { \mathsf { S } } ) , ( 4 )$  i.i.d. Bernoulli  $A _ { t } , R _ { t } , ( \mathbf { S } )$ </td><td>O(1)</td></tr><tr><td>(cumulative regret)</td><td>Algorithm 1 (Theorem 4)</td><td>i.i.d. bounded  $A _ { t } , R _ { t } , ( { \mathsf { S } } ) , ( 4 )$ </td><td>O(1)</td></tr></table>

This limitation motivates finer performance metrics for queueing bandits that compare learning policies with an informed oracle. For the single queue model, Krishnasamy et al. (2016) introduce the queue length regret $\mathbb { E } [ Q _ { T } ^ { \pi } - Q _ { T } ^ { * } ]$ where the oracle knows the service statistics and always selects the best channel. They show that this quantity initially behaves similarly to classical bandit regret, but eventually decreases at rate $\tilde { \mathcal { O } } ( 1 / T )$ . Subsequent work extends this analysis to multiple queues (Krishnasamy et al., 2021; Kim and Oh, 2024) and contextual systems (Bae et al., 2026).

Another criterion is the cumulative queue length regret, $\textstyle \sum _ { t } \mathbb { E } [ Q _ { t } ^ { \pi } - Q _ { t } ^ { * } ]$ , which measures the backlog accumulated over the horizon. For a stochastic single queue system with at least one stabilizing channel, Stahlbuhk et al. (2021) construct queue-dependent policies with $\mathcal { O } ( 1 )$ cumulative regret. This is fundamentally stronger than summing a queue length regret of order $1 / t ,$ , which would yield logarithmic cumulative regret. Complementarily, Freund et al. (2023) study the maximum increase in expected time-averaged queue length.

These results rely mainly on stochastic assumptions on arrivals and services, and much less is known when both processes can vary arbitrarily. In the oblivious adversarial regime, Krishnakumar and Sinha (2025) study the largest queue length regret over the horizon relative to the best constant policy. For a single queue served by K channels, they obtain a high-probability bound of $\tilde { \mathcal { O } } ( \sqrt { K } T ^ { 3 / 4 } )$ . Since a reduction from adversarial bandits gives a lower bound $\Omega ( \sqrt { K T } )$ , this result leaves a gap in the dependence on $T ,$ Moreover, an adversarially robust policy is particularly desirable if it retains the stronger guarantees available in stochastic regimes. Therefore, it is desirable to design a single policy that achieves near-optimal worst-case queue length regret in adversarial regimes while retaining constant expected cumulative regret in stochastic regimes.

The main difficulty is that queue length regret is governed by service losses accumulated over the current busy period, whose endpoints depend on the policy itself and may therefore be random. Hence, a bandit regret measured over the entire horizon can be too coarse, and the analysis must instead control regret over data-dependent intervals. Exploration must also be adapted to the queue state, since repeated exploration in the presence of new arrivals can cause backlog to accumulate and persist into subsequent rounds.

We address these difficulties by a parameter-free, queue-dependent policy, which collects observations through uniform exploration when the queue is empty, exploits the resulting estimates when a new busy period begins. When a queue-dependent test indicates that continued greedy exploitation may be unsafe, it switches to a robust bandit policy, Exp3-IX (Neu, 2015).

Contributions. In the oblivious adversarial regime with non-negative arrivals and bounded services, our policy achieves $\tilde { \mathcal { O } } ( \sqrt { K L _ { T } } )$ worst-case queue length regret with high-probability for the length of the longest busy period $L _ { T }$ . Since $L _ { T } \leq T$ , this matches the $\Omega ( \sqrt { K T } )$ lower bound up to logarithmic factors, while providing a sharper guarantee when busy periods are short. Under i.i.d. bounded arrivals and services with a unique stabilizing best channel, the same policy obtains $\mathcal { O } ( 1 )$ expected cumulative regret and also $\mathcal { O } ( \log ( K T ) / \Delta _ { \operatorname* { m i n } } )$ expected worst-case regret. Hence, combined with the adversarial result, it establishes a best-of-both-worlds guarantee. Table 1 summarizes our results and those of prior work.

## 2 Preliminaries

We consider a single queue served by $K \geq 2$ channels over rounds $t \in [ T ] = \{ 1 , \dots , T \}$ . Let $Q _ { t } ^ { \pi } \geq 0$ denote the queue length at the beginning of round t under a policy $\pi ,$ with $Q _ { 1 } ^ { \pi } = 0$ . At each round $t ,$ an amount $A _ { t } \geq 0$ of data arrives, and channel $i \in [ K ]$ offers a service $R _ { t , i } \in [ 0 , 1 ]$ . In this paper, we adopt an arrivals-first convention, where $A _ { t }$ enters the queue before the service (Borodin et al., 2001; Kim and Oh, 2024). We additionally assume that $A _ { t }$ is revealed before the scheduling decision. At each round t, based on $Q _ { t } ^ { \pi } , A _ { t }$ and previous observations, the policy π selects a channel $i _ { t } \in [ K ]$ without knowing $R _ { t }$ , and observes only the service $R _ { t , i _ { t } }$ of the selected channel. Then, the queue length is updated according to

$$
Q _ { t + 1 } ^ { \pi } = ( Q _ { t } ^ { \pi } + A _ { t } - R _ { t , i _ { t } } ) ^ { + } ,\tag{1}
$$

where $( x ) ^ { + } : = \operatorname* { m a x } ( x , 0 )$ . Following the previous work (Stahlbuhk et al., 2021; Freund et al., 2024), we assume $R _ { t , i _ { t } }$ is observable even when $Q _ { t } ^ { \pi } + A _ { t } = 0$

## 2.1 Queue length regret

In this paper, we define an empty period as a maximal consecutive interval satisfying $Q _ { t } ^ { \pi } + A _ { t } = 0$ , which is more restrictive than previous approaches based only on $Q _ { t } ^ { \pi } = 0$ . A busy period begins at round $\beta$ with $Q _ { \beta } ^ { \pi } = 0$ and $A _ { \beta } > 0$ , and consists of the rounds from $\beta$ up to, but not including, the next round $t > \beta$ at which $Q _ { t } ^ { \pi } = 0$ . If no such round exists, the busy period continues until $T .$ . We denote the busy periods chronologically by $( B _ { p } ) _ { p \geq 1 }$ and the empty period immediately preceding $B _ { p }$ by $\mathcal { E } _ { p }$ . Note that consecutive busy periods $B _ { p } , B _ { p + 1 }$ may be adjacent if the queue is cleared in one round and a positive arrival occurs in the next. In this case, they are separated only by the empty boundary state $Q _ { t } ^ { \pi } = 0$ , and we set the intervening empty period $\mathcal { E } _ { p + 1 } = \emptyset$ for notational consistency. Thus, the sequence of periods is represented by $( \mathcal { E } _ { 1 } , B _ { 1 } , \mathcal { E } _ { 2 } , \ldots )$ . We define the longest length of busy period up to T by

$$
L _ { T } : = \operatorname* { m a x } \big \{ | \mathcal { B } _ { p } \cap [ T ] | : p \in \mathbb { N } \big \} .\tag{2}
$$

For $i \in [ K ]$ , let $Q _ { t } ^ { i }$ be the queue length under the constant policy that always selects $i ,$ so that $Q _ { t + 1 } ^ { i } =$ $( Q _ { t } ^ { i } + A _ { t } - R _ { t , i } ) ^ { + }$ . The worst-case queue length regret is defined by

$$
\operatorname { R e g } _ { \pi } ^ { \operatorname* { m a x } } ( T ) : = \operatorname* { m a x } _ { t \in [ T ] } \left( \operatorname* { m a x } _ { i \in [ K ] } Q _ { t } ^ { \pi } - Q _ { t } ^ { i } \right) .\tag{3}
$$

In the adversarial regime, we assume the sequence $( ( A _ { t } , R _ { t } ) ) _ { t }$ is determined by an oblivious adversary in advance satisfying $A _ { t } \geq 0$ and $R _ { t } \in [ 0 , 1 ] ^ { K }$ . The following lemma shows that the queue length regret is bounded by the cumulative service regret within the current busy period rather than over the entire horizon.

Lemma 1 (Lemma 1 of Krishnakumar and Sinha (2025)). For any $p \in \mathbb N ,$ , let $\beta _ { p } : =$ min $B _ { p }$ denote thefirst round of $\boldsymbol { B } _ { p }$ . Then, for any $t \in B _ { p }$

$$
Q _ { t } ^ { \pi } - \operatorname* { m i n } _ { i \in [ K ] } Q _ { t } ^ { i } \leq \operatorname* { m a x } _ { i \in [ K ] } \sum _ { \tau = \beta _ { p } } ^ { t - 1 } ( R _ { \tau , i } - R _ { \tau , i _ { \tau } } ) .
$$

This result also explains why no upper bound on $A _ { t }$ is imposed in the adversarial regret. Although large $A _ { t }$ can produce large queue lengths, these terms cancel when queue lengths are compared since the policy and every constant policy receive the same arrivals.

In the stochastic regime, we assume the sequence $( ( A _ { t } , R _ { t } ) )$ is i.i.d., where $A _ { t } \in [ 0 , 1 ]$ and $R _ { t } \in [ 0 , 1 ] ^ { K }$ and $A _ { t }$ is independent of $R _ { t }$ . We allow arbitrary bounded distributions and fractional arrivals and services, which encompasses the Bernoulli model in Stahlbuhk et al. (2021). Instead, we only require

$$
\operatorname* { P r } ( A _ { t } = 0 ) = \rho > 0 , \forall t \in \mathbb { N } ,\tag{4}
$$

where $\rho$ is a problem-dependent constant unknown to the policy. We denote their expectations by

$$
\lambda : = \mathbb { E } [ A _ { t } ] , \quad \mu _ { i } : = \mathbb { E } [ R _ { t , i } ] , \quad \forall i \in [ K ] , \forall t \in \mathbb { N } .
$$

We assume that the best channel $i ^ { * } : = \arg \operatorname* { m a x } _ { i \in [ K ] } \mu _ { i }$ is unique and stabilizing, i.e., $\epsilon : = \mu _ { i ^ { * } } - \lambda > 0$ We define suboptimality gaps by $\Delta _ { i } = \mu _ { i ^ { * } } - \mu _ { i }$ with $\Delta _ { \mathrm { m i n } } = \mathrm { m i n } _ { i \neq i ^ { \ast } } \Delta _ { i }$ . The expected cumulative queue length regret is defined as

$$
\operatorname { R e g } _ { \pi } ^ { \mathrm { c u m } } ( T ) : = \sum _ { t = 1 } ^ { T } \mathbb { E } \Big [ Q _ { t } ^ { \pi } - Q _ { t } ^ { i ^ { \ast } } \Big ] .\tag{5}
$$

The conditions $\rho > 0$ and $\epsilon > 0$ are needed to guarantee bounded cumulative queue length regret in general. If $\rho = 0$ , a multi-armed bandit instance gives an $\Omega ( \log T )$ lower bound. Without $\epsilon > 0$ , every policy can incur $\Omega ( T )$ cumulative regret (Stahlbuhk et al., 2021).

## 2.2 Previous approaches

Since each channel can be viewed as a bandit arm with loss $\ell _ { t , i } = 1 - R _ { t , i }$ , a natural baseline is to apply a bandit policy directly to the observed services. However, even in the stochastic regime, such queueindependent policies can incur Ω(log T) cumulative queue length regret (Stahlbuhk et al., 2021). To exploit the queueing structure, Stahlbuhk et al. (2021) construct a queue-dependent three-phase policy whose cumulative regret is bounded uniformly over T and thus converges to a finite, problem-dependent constant under Bernoulli models whenever at least one channel is stabilizing.

In particular, at the first round of each empty period, they sample an arm uniformly at random and update its empirical estimate, after which it remains idle until the queue becomes busy. During the $p \mathrm { - }$ -th busy period, they select the empirical best arm, based on observations during empty periods, at most $p$ times. If the queue remains busy even after these rounds, the policy enters a recovery phase that alternates between uniform-exploration rules and greedy selection based on the local estimates until the queue empties.

In contrast, Krishnakumar and Sinha (2025) study the worst-case queue length regret under arrivals and services selected by an oblivious adversary. They reduce the queueing problem to controlling adversarial bandit regret uniformly over all subintervals and introduce a queue-independent policy that combines online gradient descent with importance-weighted (IW) loss estimates and uniform exploration, using knowledge of

T to tune its exploration rates. It achieves a high-probability regret bound of $\tilde { \mathcal { O } } ( \sqrt { K } T ^ { 3 / 4 } )$ , leaving a gap to the $\Omega ( \sqrt { K T } )$ lower bound.

Combining these guarantees requires controlling the backlog accumulated due to greedy exploitation in adversarial regimes while ensuring that recovery occurs sufficiently rare in stochastic regimes. The time-out rule of Stahlbuhk et al. (2021) allows increasingly long exploitation rounds as the busy period index grows, which may lead to large regret in adversarial regimes. Lemma 1 suggests that the queue state can help address these difficulties by restricting the relevant intervals to individual busy periods.

## 3 Proposed Policy

We propose a parameter-free, queue-dependent policy with three phases, following the structure of Stahlbuhk et al. (2021). The policy requires only the number of arms K without knowledge of any environmental parameters or the horizon T. Its key components are an additional queue-dependent condition for exploitation and an adversarial bandit policy for recovery. During empty periods, our policy explores arms uniformly and updates its empirical loss estimates. Then, each busy period begins by exploiting the empirical best arm identified from these observations. If the prescribed conditions fail before the queue empties, it enters the recovery phase by initializing an Exp3-IX instance (Neu, 2015), which continues until the queue becomes empty.

The queue-dependent condition is introduced to control the excess backlog before recovery, while Exp3- IX controls the additional service regret during recovery. Initializing a fresh instance in each recovery phase allows this regret to be analyzed over its local rounds within the current busy period.

## 3.1 Exploration phase

Whenever $Q _ { t } ^ { \pi } + A _ { t } = 0$ , the policy enters the exploration phase, where we count all exploration rounds by $m \in \mathbb { N }$ . At the m-th exploration round, corresponding to global round t, the policy draws $i _ { t }$ uniformly from [K] and observes $R _ { t , i _ { t } }$ . It then constructs an IW loss estimator by $\hat { \ell } _ { m , i } ^ { \varnothing } = K \mathbb { 1 } [ i _ { t } = i ] ( 1 - R _ { t , i _ { t } } )$ and updates the cumulative loss estimate by $\begin{array} { r } { \hat { L } _ { m + 1 } ^ { \varnothing } = \sum _ { \tau = 1 } ^ { m } \hat { \ell } _ { \tau } ^ { \varnothing } } \end{array}$ with $\hat { L } _ { 1 } ^ { \varnothing } = \mathbf { 0 }$ . These estimates are updated only during exploration phase and subsequently used to identify the empirical best arm at the beginning of each busy period.

## 3.2 Exploitation phase

At the beginning of $\boldsymbol { B } _ { p }$ , where $Q _ { t } ^ { \pi } = 0$ and $A _ { t } > 0$ , the policy enters the exploitation phase and initializes the local exploitation round to $n = 1$ . In this phase, the policy always selects

$$
\hat { i } _ { p } ^ { * } = \mathop { \arg \operatorname* { m i n } } _ { i \in [ K ] } \hat { L } _ { m + 1 , i } ^ { \varnothing } , \mathrm { ~ w h e r e ~ } m = \sum _ { \tau = 1 } ^ { p } | { \mathcal E } _ { \tau } | .\tag{6}
$$

To determine whether this greedy selection is safe, we introduce an auxiliary optimistic queue length $O _ { t }$ with $O _ { 1 } = 0$ updated in every round according to

$$
O _ { t + 1 } = ( O _ { t } + A _ { t } - 1 ) ^ { + } .
$$

Since $R _ { t , i } \leq 1 , O _ { t } \leq Q _ { t } ^ { \pi }$ holds. Hence, $Q _ { t } ^ { \pi } - O _ { t }$ measures the excess backlog incurred by imperfect service.

At the n-th exploitation round of $B _ { p } .$ corresponding to global round t, the policy keeps selecting $\hat { i } _ { p } ^ { * }$ if

$$
\begin{array} { r } { \mathbb { 1 } \big [ n \leq p , Q _ { t } ^ { \pi } - O _ { t } \leq 4 \sqrt { n \log ( p + 1 ) } = : g _ { p } ( n ) \big ] = 1 . } \end{array}\tag{7}
$$

The first condition is the time-out rule of Stahlbuhk et al. (2021), and the second controls the excess backlog, which is essential for our adversarial guarantee. Note that both conditions hold automatically at $n = 1$ , since $p \geq 1$ and $Q _ { \beta _ { p } } = O _ { \beta _ { p } } = 0$ . If the queue becomes empty, the exploitation phase ends, and if either condition fails while the queue remains busy, the policy transitions to the recovery phase.

Remark 2 (Observation of arrivals). Under our arrivals-first convention in discrete-time scheduling, $Q _ { t } ^ { \pi }$ represents the backlog carried into round t before new work enters, while $A _ { t }$ represents the work accumulated since the beginning of round $t - 1$ . At the beginning of round $t ,$ this work is admitted before the scheduling decision and service. We assume that the scheduler can observe whether the resulting queue is empty. In other words, our policy requires only the indicator $\mathbb { 1 } [ Q _ { t } ^ { \pi } + A _ { t } = 0 ]$ and this is a queue-state observation at the beginning of round t and does not provide anyfuture information about arrivals or services.

This indicator distinguishes an exploration opportunity, where $Q _ { t } ^ { \pi } = A _ { t } = 0$ , from the beginning of a busy period, where $Q _ { t } ^ { \pi } = 0$ and $A _ { t } > 0$ . Hence, the policy can use genuinely empty rounds to improve its estimations in preparation for the next busy period and then select the empirical best arm $\hat { i } _ { p } ^ { * }$ from its first round $\beta _ { p }$ . In particular, the indicator allows the policy to respond to newly admitted work without a one-round delay. If the decision were based only on $Q _ { t } ^ { \pi }$ , the policy could not distinguish these two cases and might perform uniform exploration at round $\beta _ { p } ,$ , selecting the empirical best arm only in the following round. Such unnecessary exploration can create additional queue length regret even when the optimal arm has already been identified correctly. This immediate transition from exploration to exploitation is needed for our stochastic cumulative regret guarantee. Further details are given in Appendix C.1.

Indeed, the need for this assumption in our policy depends on the queue dynamics. Our definition in (1) follows the arrivals-first convention adopted in much of the queueing bandit literature most closely related to our setting (Krishnasamy et al., 2016, 2021; Kim and Oh, 2024; Freund et al., 2024; Krishnakumar and Sinha, 2025; Bae et al., 2026). In comparison, Stahlbuhk et al. (2021) study cumulative queue length regret under the service-first convention

$$
Q _ { t + 1 } ^ { \pi } = ( Q _ { t } ^ { \pi } - R _ { t , i _ { t } } ) ^ { + } + A _ { t } ,\tag{8}
$$

which is used in the stochastic network optimization literature (Neely, 2010). In their model, $Q _ { t } ^ { \pi } = 0$ implies $Q _ { t + 1 } ^ { \pi } = A _ { t }$ , regardless of the selected arm. Therefore, exploration during empty periods preserves the queue length difference without observing the current arrival before decision. Our arrival-observation assumption serves an analogous purpose under the dynamics in (1).

## 3.3 Recovery phase

Once the policy enters the recovery phase, then it remains in this phase for the rest of the current busy period. At the beginning of the recovery phase in $B _ { p }$ , it initializes an Exp3-IX instance with local round counter $s = 1$ and cumulative estimated losses $\hat { L } _ { 1 } ^ { \mathrm { i x } } = \mathbf { \dot { 0 } }$ . At the s-th recovery round, corresponding to global round t, we define for all $i \in [ K ]$

$$
\eta _ { s } = \sqrt { \frac { \log K } { K s } } , \ \mathrm { a n d } \ w _ { s , i } = \frac { \exp \left( - \eta _ { s } \hat { L } _ { s , i } ^ { \mathrm { i x } } \right) } { \sum _ { j = 1 } ^ { K } \exp \left( - \eta _ { s } \hat { L } _ { s , j } ^ { \mathrm { i x } } \right) } .\tag{9}
$$

Algorithm 1: Best-of-Both-Worlds Bandit Scheduling   
1 Input: $K \geq 2 ;$ Initialize $t = 1 , Q _ { 1 } = O _ { 1 } = 0 , m = p = 0 , \hat { L } _ { 1 } ^ { \varnothing } = { \bf 0 } .$   
2 while True do   
3 Observe the arrival $A _ { t } .$   
4 if $Q _ { t } + A _ { t } = 0$ then $/ /$ Empty period: Exploration phase   
5 Update exploration round m $ m + 1$   
6 Play i ∼ Uniform(K) and observe service $R _ { t , i _ { t } }$   
7 Set $\hat { \ell } _ { m , i } ^ { \varnothing } = \mathbb { 1 } [ i _ { t } = i ] K ( 1 - R _ { t , i _ { t } } )$ and update $\hat { L } _ { m + 1 } ^ { \varnothing } = \hat { L } _ { m } ^ { \varnothing } + \hat { \ell } _ { m } ^ { \varnothing } .$   
8 else $/ /$ Busy period   
9 if $Q _ { t } = 0$ then // First round of Exploitation phase   
10 Set phase ← exploitation.   
11 Increase the busy period count $p \gets p + 1$ , reset exploitation round n $ 1 .$   
12 Take the empirical best $\hat { i } _ { p } ^ { * } = \mathrm { a r g }$ min $\iota \in [ K ]  \stackrel { \widehat { L } ^ { \varnothing } } { m + 1 , i }$ in empty periods.   
13 Play $i _ { t } = \hat { i } _ { p } ^ { * }$ and observe service $R _ { t , i _ { t } }$   
14 Update exploitation round n $ 2 .$   
15 else $/ /$ When $Q _ { t } ^ { \pi } > 0$   
16 if phase = exploitation then $/ /$ Exploitation phase   
17 if $n \leq p , Q _ { t } - Q _ { t } \leq g _ { p } ( n )$ then   
18 Play $i _ { t } = \hat { i } _ { p } ^ { * } .$ observe service $R _ { t , i _ { t } }$ , and update exploitation round $n  n + 1 .$   
19 else $/ / \quad \mathtt { I n i t i a l i z e }$ the recovery phase   
20 Set phase ← recovery.   
21 Initialize a new Exp3-IX instance by $s \gets 1$ and $\hat { L } _ { 1 } ^ { \mathrm { i x } } = \mathbf { 0 } .$   
22 if phase = recovery then $/ /$ Recovery phase: Exp3-IX   
23 Set $\eta _ { s }$ and $w _ { s }$ as in (9), and $\zeta _ { s } = \eta _ { s } / 2 .$   
24 Play $i _ { t } \sim w _ { s }$ and observe service $R _ { t , i _ { t } } .$   
25 Set $\hat { \ell } _ { s , i } ^ { \mathrm { i x } }$ as in (10) and update $\hat { L } _ { s + 1 } ^ { \mathrm { i x } } = \hat { L } _ { s } ^ { \mathrm { i x } } + \hat { \ell } _ { s } ^ { \mathrm { i x } }$ and recovery round $s \gets s + 1$   
26 Update queue length $Q _ { t + 1 } = ( Q _ { t } + A _ { t } - R _ { t , i _ { t } } ) ^ { + } \mathrm { ~ a n d ~ } O _ { t + 1 } = ( O _ { t } + A _ { t } - 1 ) ^ { + }$   
27 Update global round $t  t + 1 .$

The policy samples $i _ { t } \sim w _ { s } ,$ observes $R _ { t , i _ { t } }$ , constructs the implicit-exploration loss estimator

$$
\hat { \ell } _ { s , i } ^ { \mathrm { i x } } = \frac { \mathbb { 1 } [ i _ { t } = i ] ( 1 - R _ { t , i _ { t } } ) } { w _ { s , i } + \zeta _ { s } } ,\tag{10}
$$

and then updates $\begin{array} { r } { \hat { L } _ { s + 1 } ^ { \mathrm { i x } } = \sum _ { \tau = 1 } ^ { s } \hat { \ell } _ { \tau } ^ { \mathrm { i x } } } \end{array}$ . Here, $\zeta _ { s } = \eta _ { s } / 2$ is the implicit-exploration parameter introduced by Neu (2015). Adding $\zeta _ { s }$ introduces a controlled bias and limits the magnitude of the loss estimator for small $w _ { s , i }$ , enabling high-probability adversarial regret guarantees without explicit uniform exploration.

The Exp3-IX instance runs until the queue becomes empty, after which its local estimates are discarded. Thus, recovery observations are not shared across busy periods and not incorporated into the exploration estimate $\hat { L } _ { m } ^ { \varnothing }$ . While Stahlbuhk et al. (2021) employed UCB-type policies in their heuristic policy, our use of Exp3-IX is designed to provide guarantees under both stochastic and adversarial regimes. Specifically, its high-probability regret bound controls the additional service regret during recovery in the adversarial regime. The complete pseudo-code is given in Algorithm 1.

## 4 Theoretical Guarantee

In this section, we provide the regret bounds of our policy in both adversarial and stochastic regimes, whose detailed proofs are given in Appendices A, C, and E.

Theorem 3. In the oblivious adversarial regime, for any $T \geq 1$ and $\delta \in ( 0 , 1 )$ , Algorithm 1 satisfies

$$
\mathrm { R e g } _ { \pi } ^ { \mathrm { m a x } } ( T ) \leq 4 \sqrt { L _ { T } \log ( T + 1 ) } + 4 \sqrt { K L _ { T } \log K } + \left( 2 \sqrt { \frac { K L _ { T } } { \log K } } + 1 \right) \log \left( \frac { 6 L _ { T } ^ { 2 } T ^ { 2 } } { \delta } \right) + 3 \sqrt { \frac { K L _ { T } } { \log K } } .
$$

with probability at least $1 - \delta$ whenever $L _ { T } \ge 1$

Note that $L _ { T } = 0$ means that the queue remains empty for all rounds, where the worst-case queue length regret is trivially zero. Moreover, since $\operatorname { R e g } _ { \pi } ^ { \operatorname* { m a x } } ( T ) \leq T$ holds by $R _ { t } \in [ 0 , 1 ] ^ { K }$ and Lemma 1, setting $\delta = 1 / T$ gives an expected regret bound of the same order, up to logarithmic factors.

Proofsketch. Since threshold $g _ { p } ( n )$ controls the excess backlog accumulated during exploitation and Exp3- IX controls the additional regret incurred during recovery, the analysis reduces to controlling the regret of each recovery instance at every local round.

Let ${ \mathrm { R e g } } _ { \mathrm { I X } } ( s ; p )$ be the regret of corresponding Exp3-IX instance accumulated over the first s recovery rounds during $p \cdot$ -th busy period. Then, we establish that

$$
\mathrm { R e g } _ { \pi } ^ { \operatorname* { m a x } } ( T ) \lesssim \sqrt { L _ { T } \log T } + \operatorname* { m a x } _ { p , s } ( \mathrm { R e g } _ { \mathrm { I X } } ( s ; p ) ) ^ { + } .
$$

The first term follows the exploitation threshold in (7), which controls the worst-case regret when the policy enters recovery. For the second term, Neu (2015) provides a high-probability regret bound of $\tilde { \mathcal { O } } ( \sqrt { K H } )$ for an Exp3-IX instance over H rounds. We apply this bound to each instance and take a union bound over all instance indices $p$ and local rounds $s .$ Since every recovery instance lasts at most $L _ { T }$ rounds, we obtain the desired high-probability regret bound. □

In the stochastic regime, the cumulative regret is finite if the unique best arm is stabilizing and both the arrivals and services are i.i.d. with support [0, 1].

Theorem 4. Suppose that $( ( A _ { t } , R _ { t } ) ) _ { t }$ <sub>t</sub> is i.i.d. with $A _ { t } \in [ 0 , 1 ] , \operatorname* { P r } ( A _ { t } = 0 ) = \rho > 0 ,$ , and $R _ { t } \in [ 0 , 1 ] ^ { K }$ , and that the unique best arm $i ^ { * }$ is stabilizing with $\epsilon = \mu _ { i ^ { * } } - \lambda > 0$ . Then, Algorithm 1 satisfies

$$
\sum _ { t = 1 } ^ { \infty } \mathbb { E } \bigg [ \Big ( Q _ { t } ^ { \pi } - Q _ { t } ^ { i ^ { * } } \Big ) ^ { + } \bigg ] \leq C _ { \mathrm { s t o } } < \infty
$$

where

$$
C _ { \mathrm { s t o } } = \mathcal { O } \bigg ( \frac { K ^ { 4 } } { \epsilon ^ { 2 } \Delta _ { \mathrm { m i n } } ^ { 6 } \rho ^ { 3 } } + \frac { K ^ { 4 } \log ^ { 2 } K } { \epsilon ^ { 4 } \Delta _ { \mathrm { m i n } } ^ { 2 } \rho } + \frac { 1 } { \epsilon ^ { 8 } } + \frac { K ^ { 2 } \log ^ { 2 } K } { \epsilon ^ { 6 } } \bigg ) .
$$

Proof sketch. For the cumulative queue length regret, we follow the busy period decomposition of Stahlbuhk et al. (2021). We first define the good period indicator

$$
\mathcal { I } _ { p } : = \mathbb { 1 } \Big [ \widehat { i } _ { p } ^ { * } = i ^ { * } , \mathcal { B } _ { p } ^ { \mathrm { i x } } = \emptyset \Big ] ,
$$

where $B _ { p } ^ { \mathrm { i x } }$ denotes the interval of recovery phase during the busy period $B _ { p }$ . When $\mathcal { I } _ { p } = 1$ , the policy selects $i ^ { * }$ throughout $B _ { p }$ , including its first round $\beta _ { p }$ . Since the empty period is determined by the queue length of the policy, $Q _ { \beta _ { p } } ^ { i ^ { * } }$ can be positive and the monotonicity of the queue recursion implies $Q _ { t } ^ { \pi } \leq Q _ { t } ^ { i ^ { * } }$ throughout this period. Empty periods also contribute no positive queue length difference. Hence, it suffices to control the contributions of busy periods satisfying $\mathcal { I } _ { p } = 0$

The argument combines a bound on the regret in $\mathcal { I } _ { p } = 0$ with a bound on its probability. Since bounded arrivals imply $O _ { t } = 0 , g _ { p } ( n )$ directly controls the queue length, while the time-out condition limits greedy exploitation to at most $p$ rounds. During the recovery phase, the Exp3-IX regret bound and the stability gap ϵ provide tail bounds on the rounds required to empty the queue. By combining these arguments, we obtain

$$
\mathbb { E } \left[ \sum _ { t \in \mathcal { B } _ { p } } ( Q _ { t } ^ { \pi } - Q _ { t } ^ { i ^ { * } } ) ^ { + } \right] \leq C _ { p } \operatorname* { P r } ( \mathcal { T } _ { p } = 0 ) ,
$$

where $C _ { p }$ is polynomial in $p , \epsilon ^ { - 1 }$ , and $K$ .

By definition, $\mathcal { I } _ { p } = 0$ occurs only if $\hat { i } _ { p } ^ { * }$ is incorrect or exploitation phase ends before the queue becomes empty. Since the assumption $\operatorname* { P r } ( A _ { t } = \hat { 0 } ) = \rho > 0$ provides certain degree of exploration opportunities, combining a concentration inequality on the number of exploration rounds, with a time-uniform version of Freedman’s inequality for the loss estimates, controls the probability of selecting an incorrect empirical best arm. When the empirical best arm is correct, Hoeffding’s inequality and $\epsilon > 0$ controls the probabilities of violating the queue length threshold or reaching the time-out before the queue empties. The obtained probability bound decays sufficiently fast so that

$$
\sum _ { p = 1 } ^ { \infty } C _ { p } \operatorname* { P r } ( \mathcal { T } _ { p } = 0 ) \le C _ { \mathrm { s t o } } < \infty ,
$$

which concludes the proof.

However, Theorem 4 does not directly bound the expected worst-case queue length regret, since the maximum over time is taken before the expectation. The following corollary provides such a guarantee.

Corollary 5. Under the assumptions of Theorem 4, for any $T \geq 1$ , Algorithm 1 satisfies

$$
\mathbb { E } [ \mathrm { R e g } _ { \pi } ^ { \operatorname* { m a x } } ( T ) ] \leq C _ { \mathrm { s t o } } + { \mathcal O } \bigg ( \frac { \log ( K T ) } { \Delta _ { \operatorname* { m i n } } } \bigg ) .
$$

Thus, our policy simultaneously achieves uniformly bounded expected cumulative regret and logarithmic expected worst-case regret in the stochastic regime.

Proof Sketch. For worst-case regret, it suffices to additionally compare $Q _ { t } ^ { i ^ { * } }$ with every constant policy as

$$
\mathrm { R e g } _ { \pi } ^ { \operatorname* { m a x } } ( T ) \leq \operatorname* { m a x } _ { t } ( Q _ { t } ^ { \pi } - Q _ { t } ^ { i ^ { * } } ) ^ { + } + \operatorname* { m a x } _ { t , i } ( Q _ { t } ^ { i ^ { * } } - Q _ { t } ^ { i } ) ^ { + } .
$$

The expected first term is bounded by $C _ { \mathrm { s t o } }$ . For each suboptimal arm $i ,$ the service difference $R _ { t , i } - R _ { t , i ^ { * } }$ has negative mean $- \Delta _ { i }$ . Applying concentration inequalities to these service differences over all relevant intervals provides the bound on the second term,

$$
\mathbb { E } [ \mathrm { R e g } _ { \pi } ^ { \operatorname* { m a x } } ( T ) ] \leq C _ { \mathrm { s t o } } + O \left( \frac { \log ( K T ) } { \Delta _ { \operatorname* { m i n } } } \right) ,
$$

which concludes the proof.

![](images/44c4f42eb9148e368bcd604504b430e477eee1e3a7a1f0f7e1a09ac7ede52056.jpg)  
(a) Worst-case regret.

![](images/bd6abb5d43fe8791da8f9a4cdf88f230b5e7b1cb6c432027573d3631b0730d53.jpg)  
(b) Cumulative regret.  
Figure 1: Results in the adversarial regime.

## 5 Numerical experiments

To verify our best-of-both-worlds guarantee, we evaluate our policy in an adversarially varying environment and an i.i.d. stochastic environment, conducting 1000 independent runs of T = 10000 rounds in each setting. In all figures, the lines denote the empirical means and shaded regions indicate pointwise 95% confidence intervals. Further experimental details are in Appendix G.

Baselines. We compare our policy, Algorithm 1, with (i) Q-ThS (Krishnasamy et al., 2016), which is designed to minimize the queue length regret, (ii) the three phase policy of Stahlbuhk et al. (2021), which is designed for cumulative queue length regret, and (iii) OGD-based policy of Krishnakumar and Sinha (2025), which is designed to minimize worst-case queue length regret. In this section, we refer to the latter two policies as SSM-3P and KS-OGD, respectively.

Adversarial environments. In this setting, we follow the instance considered in Krishnakumar and Sinha (2025), where there are $K = 5$ non-stationary arms and we divide the horizon into $h = 7$ consecutive blocks of equal lengths. At the beginning of each block, we sample a coefficient $\mu _ { i } ^ { ( h ) }$ from Uniform(0, 1) and fix this value. Then, the services of i at every $t + 1$ in block h is determined by

$$
R _ { t + 1 , i } = \operatorname* { m a x } ( 0 , \operatorname* { m i n } ( 1 , \mu _ { i } ^ { ( h ) } \cdot R _ { t , i } + \xi _ { t , i } ) ) ,
$$

where $\xi _ { t , i } \sim \mathrm { U n i f o r m } ( - 1 , 1 )$ is an i.i.d. additive noise. The arrivals are sampled from a uniform distribution with mean λ satisfying $\mu _ { i _ { T } ^ { * } } - \lambda = 0 . 0 5$

Figure 1 shows the empirical queue length regret in this adversarial instance. KS-OGD performs best on this instance, whereas SSM-3P performs worst, consistent with the observations in Krishnakumar and Sinha (2025). Our policy achieves performance comparable to KS-OGD and Q-ThS in both worst-case and cumulative queue length regret.

The queue length trajectories in Figure 3a explicitly show the increase after some block transitions, particularly between rounds 4000 and 6000, i.e., the fourth block. All four policies experience substantial regret increase during this interval, suggesting that these transitions introduce significant environmental changes. For our policy, such changes may make the arm selected for greedy exploitation suboptimal, resulting in additional backlog before the recovery phase.

Stochastic environments. For the stochastic regime, we follow the setting of Stahlbuhk et al. (2021), with i.i.d. Bernoulli arrivals and services of expectations $\lambda = 0 . 5$ and $\mu = ( 0 . 1 , 0 . 3 , 0 . 5 , 0 . 7 )$ , respectively. As shown in Figure 2, our policy achieves substantially lower regret than the baselines under both regrets.

![](images/b49f29c990ea3a35cce035169a41d49ed06d6cb097f37e778fe251ee1921c75f.jpg)  
(a) Worst-case regret.

![](images/38c57ed69ace9806cd8845780ddeac0adee9ae3477677c8f1b546ac0d9f62e8d.jpg)  
(b) Cumulative regret.

Figure 2: Results in the stochastic regime.  
![](images/bd5ab5e4bb6607a957980a0aae089a2775cec11f7113774922a783adcdcf43cb.jpg)  
(a) Adversarial regime.

![](images/bd7bdbd27ec6738f5ede1129b3b65c83dcb27bacc1fe7c7b9a589e8722d8a2fc.jpg)  
(b) Stochastic regime.  
Figure 3: The trajectories of queue lengths $Q _ { t } ^ { \pi }$

Moreover, our policy, KS-OGD, and Q-ThS appear to converge to finite values as the horizon increases. Combined with the adversarial experiments, these results complement our theoretical analysis by illustrating the competitive performance of our policy across both regimes.

The queue length trajectories in Figure 3b provide further insight into these differences. In particular, Q ThS and KS-OGD exhibit a clear transition from initial queue growth to subsequent stabilization, qualitatively resembling the two-stage behavior of queue length regret studied by Krishnasamy et al. (2016). During the early stage, exploration and inaccurate service estimates can lead to an increase the queue length. As the policy learns to select the best arm more often, it can stabilize the queue and thus queue length decreases.

In contrast, two phase-based policies, ours and SSM-3P, exhibit a less distinct separation between the two stages and a gradual decrease in later rounds. While our policy initially has the shortest queue and reduces it first, $Q _ { t } ^ { \pi }$ of KS-OGD and Q-ThS decrease rapidly and eventually fall below that of ours. The gradual decrease under our policy may result from additional explorations needed for robustness, which can select suboptimal arms and delay queue clearance in stochastic regimes.

## 6 Conclusion

We proposed a parameter-free policy that achieves best-of-both-worlds guarantees for single queueing bandits. Our policy achieves $\widetilde { \mathcal { O } } ( \sqrt { K L _ { T } } )$ ) worst-case queue length regret with high-probability in the oblivious adversarial regime, where $L _ { T }$ denotes the length of the longest busy period. Under i.i.d. bounded sequences with a unique stabilizing optimal arm, it simultaneously achieves constant expected cumulative queue length regret and logarithmic expected worst-case queue length regret. These results show that a simple policy can retain strong stochastic guarantees as well as adversarial guarantees, without knowledge of the regime. Numerical results support our analysis by showing competitive performance across both regimes.

A promising direction is to investigate whether ideas from decoupled bandits can improve the use of exploration opportunities (Rouyer and Seldin, 2020; Kim et al., 2026). Our policy already separates selections for exploration from those for exploitation, assigning these roles to empty and busy periods, respectively. This structure suggests replacing uniform exploration with adaptive sampling that accounts for uncertainty in the estimates and their importance to subsequent decisions. Such an approach may reduce the cost of initial learning and improve practical performance. Establishing whether such improvements can be achieved while preserving both the stochastic guarantee and adversarial robustness is an interesting direction for future work.

## AI use statement

In this work, we used generative AI tools, Chat-GPT 5.6 Sol xHigh, to help develop the conceptual framework through a survey of the queueing bandit literature, formulate mathematical claims, particularly notation and definitions, suggest ingredients for proofs, particularly for understanding and adapting the formulation of Howard et al. (2020) to our one-dimensional notation, assist in writing proofs through choices of terminology and notation, refine research hypotheses through a survey of different queue dynamics, provide feedback on the research methodology and experimental design, implement methods in Python. We did not use generative AI tools for interpretation of results, writing the manuscript from the beginning, translation, cleaning or reformatting datasets, generating synthetic datasets, and qualitative and thematic data analysis was not applicable to this work.

Additionally, we used the tool for brainstorming, searching for related literature, identifying relevant references, suggesting experimental settings, generating and editing software code, revising the manuscript, improving readability, and suggesting the title of the paper.

We reviewed all AI-assisted work. All mathematical claims were checked and written line by line by the authors. All AI-assisted code was reviewed and tested against the stated algorithms, and the reported experimental results were generated and verified by the authors. All references and literature summaries were checked against the original sources through a manual literature review. The authors revised and approved all AI-assisted text. We take responsibility for the final content of this work, including text, claims, and artifacts produced with the aid of generative AI.

## Acknowledgements

JL was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. RS-2024-00395303) and by the grant No. 2024-00460980 (IITP) funded by Korea Government (MSIT). CK was supported by the grant Nos. 2024-00460980; and 2025-02304717 (IITP) funded by the Korea government (the Ministry of Science and ICT).

## Bibliography

Peter Auer and Chao-Kai Chiang. An algorithm with nearly optimal pseudo-regret for both stochastic and adversarial bandits. In Conference on Learning Theory, pages 116–120. PMLR, 2016.

Seoungbin Bae, Garyeong Kang, and Dabeen Lee. Queue length regret bounds for contextual queueing bandits. In International Conference on Learning Representations, 2026.

Allan Borodin, Jon Kleinberg, Prabhakar Raghavan, Madhu Sudan, and David P Williamson. Adversarial queuing theory. Journal ofthe ACM (JACM), 48(1):13–38, 2001.

Sébastien Bubeck and Aleksandrs Slivkins. The best of both worlds: Stochastic and adversarial bandits. In Conference on Learning Theory, pages 42–1. JMLR Workshop and Conference Proceedings, 2012.

Jim G Dai. On positive harris recurrence of multiclass queueing networks: a unified approach via fluid limit models. The Annals ofApplied Probability, 5(1):49–77, 1995.

Daniel Freund, Thodoris Lykouris, and Wentao Weng. Quantifying the cost of learning in queueing systems. Advances in Neural Information Processing Systems, 36:6532–6544, 2023.

Daniel Freund, Thodoris Lykouris, and Wentao Weng. Efficient decentralized multi-agent learning in asymmetric bipartite queueing systems. Operations Research, 72(3):1049–1070, 2024.

Junya Honda, Shinji Ito, and Taira Tsuchiya. Follow-the-perturbed-leader achieves best-of-both-worlds for bandit problems. In International Conference on Algorithmic Learning Theory, pages 726–754. PMLR, 2023.

Steven R. Howard, Aaditya Ramdas, Jon McAuliffe, and Jasjeet Sekhon. Time-uniform chernoff bounds via nonnegative supermartingales. Probability Surveys, 17:257–317, 2020. doi: 10.1214/18-PS321.

Chaiwon Kim, Jongyeong Lee, and Min-hwan Oh. Follow-the-perturbed-leader for decoupled bandits: Best-of-both-worlds and practicality. International Conference on Machine Learning, 2026.

Jung-hun Kim and Min-hwan Oh. Queueing matching bandits with preference feedback. Advances in Neural Information Processing Systems, 37:64649–64702, 2024.

G Krishnakumar and Abhishek Sinha. Minimizing queue length regret for arbitrarily varying channels. In IEEE International Symposium on Information Theory, pages 1–6. IEEE, 2025.

Subhashini Krishnasamy, Rajat Sen, Ramesh Johari, and Sanjay Shakkottai. Regret of queueing bandits. Advances in Neural Information Processing Systems, 29, 2016.

Subhashini Krishnasamy, Rajat Sen, Ramesh Johari, and Sanjay Shakkottai. Learning unknown service rates in queues: A multiarmed bandit approach. Operations research, 69(1):315–330, 2021.

Tor Lattimore and Csaba Szepesvári. Bandit algorithms. Cambridge University, 355, 2019.

Michael Neely. Stochastic network optimization with application to communication and queueing systems. Morgan & Claypool Publishers, 2010.

Gergely Neu. Explore no more: Improved high-probability regret bounds for non-stochastic bandits. Advances in Neural Information Processing Systems, 28, 2015.

Chloé Rouyer and Yevgeny Seldin. Tsallis-inf for decoupled exploration and exploitation in multi-armed bandits. In Conference on Learning Theory, volume 125, pages 3227–3249. PMLR, 2020.

Thomas Stahlbuhk, Brooke Shrader, and Eytan Modiano. Learning algorithms for minimizing queue length regret. IEEE Transactions on Information Theory, 67(3):1759–1781, 2021.

Zixian Yang, R Srikant, and Lei Ying. Learning while scheduling in multi-server systems with unknown statistics: Maxweight with discounted ucb. In International Conference on Artificial Intelligence and Statistics, pages 4275–4312. PMLR, 2023.

Julian Zimmert and Yevgeny Seldin. Tsallis-inf: An optimal algorithm for stochastic and adversarial bandits. Journal ofMachine Learning Research, 22(28):1–49, 2021.

## Appendix Contents

A Proof of Theorem 3: adversarial regime 16   
B Proof of Lemmas for Theorem 3 17   
B.1 Proof of Lemma 7 . 17   
B.2 Proof of Lemma 8 . 18   
C Proof of Theorem 4: stochastic regime 18   
C.1 Observation of arrivals 21   
D Proofs of Lemmas for Theorem 4 22   
D.1 Proof of Lemma 11 22   
D.2 Proof of Lemma 12 24   
E Proof of Corollary 5 25   
F Proofs of auxiliary lemmas 27   
F.1 Proof of Lemma 9 . 27   
F.2 Proof of Lemma 10 27   
F.3 Proof of Lemma 13 27   
F.4 Proof of Lemma 14 30   
G Details on experiments 31

## A Proof of Theorem 3: adversarial regime

For notational simplicity, we partition each busy period into two consecutive subintervals according to the phases of Algorithm 1. For each busy period $B _ { p } ,$ let $\boldsymbol { B } _ { p } ^ { \mathrm { e x } }$ denote its exploitation interval and $B _ { p } ^ { \mathrm { i x } }$ be the subsequent recovery interval, during which an Exp3-IX instance is run. Thus, $B _ { p } = B _ { p } ^ { \mathrm { e x } } \sqcup B _ { p } ^ { \mathrm { i x } }$ holds, where $B _ { p } ^ { \mathrm { i x } }$ may be empty if the queue empties during the exploitation phase.

To quantify the contribution of the recovery phase, consider any $p$ such that $B _ { p } ^ { \mathrm { i x } } \neq \varnothing$ , and let $\tau ( p , s )$ denote the global round of the s-th recovery round in $B _ { p } ^ { \mathrm { i x } }$ . Specifically, it satisfies

$$
\tau ( p , s ) = \sum _ { p ^ { \prime } = 1 } ^ { p - 1 } | \mathcal { E } _ { p ^ { \prime } } \sqcup \mathcal { B } _ { p ^ { \prime } } | + | \mathcal { E } _ { p } | + | \mathcal { B } _ { p } ^ { \mathrm { e x } } | + s .
$$

For any $1 \leq s \leq | \beta _ { p } ^ { \mathrm { i x } } |$ , define the regret of the corresponding Exp3-IX instance over its first s recovery rounds by

$$
\mathrm { R e g } _ { \mathrm { I X } } ( s ; p ) : = \operatorname* { m a x } _ { i \in [ K ] } \sum _ { s ^ { \prime } = 1 } ^ { s } ( R _ { \tau ( p , s ^ { \prime } ) , i } - R _ { \tau ( p , s ^ { \prime } ) , i _ { \tau ( p , s ^ { \prime } ) } } ) = \sum _ { s ^ { \prime } = 1 } ^ { s } \ell _ { \tau ( p , s ^ { \prime } ) , i _ { \tau ( p , s ^ { \prime } ) } } - \operatorname* { m i n } _ { i \in [ K ] } \sum _ { s ^ { \prime } = 1 } ^ { s } \ell _ { \tau ( p , s ^ { \prime } ) , i } .\tag{11}
$$

The high-probability bound for this regret is given as follows.

Lemma 6 (Theorem 1 of Neu (2015)). Consider any oblivious loss sequence $( \ell _ { t } ) _ { t }$ <sub>t</sub> with $\ell _ { t } \in [ 0 , 1 ] ^ { K }$ . Then, for anyfixed $s \geq 1$ and any $\delta \in ( 0 , 1 )$ , Exp3-IX satisfies

$$
\operatorname* { P r } \left( \operatorname { R e g } _ { \mathrm { I X } } ( s ; \cdot ) > 4 { \sqrt { K s \log K } } + \left( 2 { \sqrt { \frac { K s } { \log K } } } + 1 \right) \log { \frac { 2 } { \delta } } \right) \leq \delta .
$$

Since the index $p$ only identifies the busy periods, it does not appear in this per-phase bound. We next express the worst-case queue length regret in terms of ${ \mathrm { R e g } } _ { \mathrm { I X } } ( s ; p )$ , whose proof is given in Appendix B.1.

Lemma 7. Algorithm 1 satisfies for any $T \geq 1$

$$
\mathrm { R e g } _ { \pi } ^ { \mathrm { m a x } } ( T ) \leq 3 + 4 \sqrt { L _ { T } \log ( T + 1 ) } + \operatorname* { m a x } _ { p : \mathcal { B } _ { p } ^ { \mathrm { i x } } \cap [ T ] \neq \emptyset } \operatorname* { m a x } _ { s \leq | \mathcal { B } _ { p } ^ { \mathrm { i x } } \cap [ T ] | } ( \mathrm { R e g } _ { \mathrm { I X } } ( s ; p ) ) ^ { + } .
$$

Note that Lemma 6 considers a single Exp3-IX instance at a fixed length s. Since each recovery phase initializes a fresh instance, Lemma 6 can be applied to ${ \mathrm { R e g } } _ { \mathrm { I X } } ( s ; p )$ for every fixed pair $( p , s )$ after conditioning on the history preceding the $p \mathrm { - }$ -th recovery phase. Hence, a union bound extends this guarantee to all Exp3-IX instances and their realized pairs as follows.

Lemma 8. For any $\delta \in ( 0 , 1 )$ and for any $T \geq 1$

$$
\operatorname* { m a x } _ { \substack { p : B _ { p } ^ { \mathrm { i s } } \cap [ T ] \neq \emptyset } } \operatorname* { m a x } _ { \substack { s \leq | B _ { p } ^ { \mathrm { i s } } \cap [ T ] | } } \mathrm { R e g } _ { \mathrm { I X } } ( s ; p ) \leq 4 \sqrt { K L _ { T } \log K } + \left( 2 \sqrt { \frac { K L _ { T } } { \log K } } + 1 \right) \log \left( \frac { 6 L _ { T } ^ { 2 } T ^ { 2 } } { \delta } \right) .
$$

holds with probability at least $1 - \delta .$

ProofofTheorem 3. Combining Lemmas 7 and 8 concludes the proof.

## B Proof of Lemmas for Theorem 3

## B.1 Proof of Lemma 7

To derive the upper bounds on the worst-case queue length regret, we first introduce the relationship between queue length $Q _ { t } ^ { \pi }$ and optimistic queue length $O _ { t }$ defined in Section 3. Since

$$
\begin{array} { r } { Q _ { t + 1 } ^ { \pi } = ( Q _ { t } ^ { \pi } + A _ { t } - R _ { t , i _ { t } } ) ^ { + } \quad \mathrm { a n d } \quad O _ { t + 1 } = ( O _ { t } + A _ { t } - 1 ) ^ { + } , } \end{array}
$$

for any $t \geq 1$ , it holds that

$$
O _ { t } \leq Q _ { t } ^ { \pi } , \quad \mathrm { a n d } \quad O _ { t } \leq Q _ { t } ^ { i } , \forall i \in [ K ] .\tag{12}
$$

Define the observable excess queue length $E _ { t }$ by

$$
E _ { t } : = Q _ { t } ^ { \pi } - O _ { t } .\tag{13}
$$

Then, (12) and its definition imply that

$$
Q _ { t } ^ { \pi } - \operatorname* { m i n } _ { i \in [ K ] } Q _ { t } ^ { i } \leq E _ { t } , \quad \mathrm { a n d } \quad E _ { t } \leq E _ { t - 1 } + 1 .\tag{14}
$$

Then, we need following auxiliary lemmas to prove Lemma 7, where the proofs are postponed to Appendix F.

Lemma 9. For any $t \in B _ { p } ^ { \mathrm { e x } }$ , whose corresponding local exploitation round is $n \geq 1$ , Algorithm 1 satisfies

$$
Q _ { t } ^ { \pi } - \operatorname* { m i n } _ { i \in [ K ] } Q _ { t } ^ { i } \leq g _ { p } ( n ) ,
$$

where $g _ { p } ( n ) = 4 { \sqrt { n \log ( p + 1 ) } }$ as defined in $( 7 )$ . Moreover, if $B _ { p } ^ { \mathrm { i x } } \neq \varnothing$ , it holds that

$$
E _ { \tau ( p , 1 ) } \leq g _ { p } ( | B _ { p } ^ { \mathrm { e x } } | ) + 1 ,
$$

where $\tau ( p , 1 )$ denotes the global round of the first local recovery rounds in the p-th busy period.

Lemma 10. For any $t \in B _ { p } ^ { \mathrm { i x } }$ , whose corresponding local recovery round is $s \geq 1$ , Algorithm 1 satisfies

$$
Q _ { t } ^ { \pi } - \operatorname* { m i n } _ { i \in [ K ] } Q _ { t } ^ { i } \leq E _ { \tau ( p , 1 ) } + \mathrm { R e g } _ { \mathrm { I X } } ( s ; p ) + 1 .
$$

Proof of Lemma 7. Let $\operatorname* { m a x } _ { x \in \varnothing } f ( x ) = 0$ for any function $f .$ Then, by Lemmas 9 and 10, we obtain

$$
\begin{array} { r l r } { \mathrm { R e } _ { 5 } ^ { \mathrm { g u a x } } ( T ) = \operatorname* { m a x } _ { t \in [ 1 ] \times \mathrm { e } } ( Q _ { t } ^ { \pi } - Q _ { t } ^ { i } ) } \\ & { = \frac { \operatorname* { m a x } _ { \rho } } { \rho ^ { 2 } \Sigma } \operatorname* { m a x } _ { t \in [ N ] } \operatorname* { m a x } _ { t } ( Q _ { t } ^ { \pi } - Q _ { t } ^ { i } ) } & { \qquad } & { \mathrm { ( : : : ~ } Q _ { t } ^ { \pi } = \mathrm { ~ f o r ~ } \iota \in \mathcal { E } _ { p } \mathrm { ~ , ~ } \forall p \geq \mathrm { ~ 1 ~ ) } } \\ & { = \frac { \operatorname* { m a x } _ { \rho } } { \rho ^ { 2 } \Sigma } \operatorname* { m a x } _ { t \in [ N ] } \operatorname* { m a x } _ { \rho } ( Q _ { t } ^ { \pi } - Q _ { t } ^ { i } ) , \quad \mathrm { ~ m a x ~ } \quad \mathrm { ~ m a x ~ } ( Q _ { \tau } ^ { \pi } - Q _ { t } ^ { i } ) \bigg ) } \\ & { = \frac { \operatorname* { m a x } _ { \rho } \operatorname* { m a x } _ { \rho } } { \rho \Sigma } \Bigg ( \frac { \operatorname* { m a x } _ { \rho } } { \tau \Sigma \operatorname* { m } _ { p } ^ { \rho } ( T ) \{ c \{ K \} \{ Z \{ K \} \{ Z \{ K \} \} } } \{ Q _ { \tau } ^ { \pi } - Q _ { t } ^ { i }  \} , \quad \mathrm { ~ m a x ~ } \quad \mathrm { ~ m a x ~ } ( Q _ { \tau } ^ { \pi } - Q _ { t } ^ { i } ) \Bigg )  & { \quad \mathrm { ( d i s j o i n f ~ p l a s e s ) } } \\ & { \leq \frac { \operatorname* { m a x } _ { \rho } \operatorname* { m a x } _ { \rho } } { \rho \Sigma } \Bigg ( \frac { \operatorname* { m a x } _ { \rho } } { \tau \Sigma \{ S _ { \rho } ^ { \pi } \{ P _ { p } ^ { \pi } \} } \{ \rho \} } \ \end{array}
$$

The last inequality follows since $| B _ { p } ^ { \mathrm { e x } } | \le L _ { T }$ by definition of $L _ { T }$ in (2) and $p \leq T$

## B.2 Proof of Lemma 8

Proof. For each pair $( p , s ) \in  { \mathbb { N } } ^ { 2 }$ , define

$$
\delta _ { p , s } : = \frac { \delta } { 3 p ^ { 2 } s ^ { 2 } } .
$$

Since

$$
\sum _ { p = 1 } ^ { \infty } \sum _ { s = 1 } ^ { \infty } \delta _ { p , s } = \frac { \pi ^ { 4 } } { 1 0 8 } \delta < \delta ,
$$

applying Lemma 6 with $\delta _ { p , s }$ and taking a union bound over all $( p , s )$ shows that

$$
\mathrm { R e g } _ { \mathrm { I X } } ( s ; p ) \leq 4 { \sqrt { K s \log K } } + \left( 2 { \sqrt { \frac { K s } { \log K } } } + 1 \right) \log \left( { \frac { 6 p ^ { 2 } s ^ { 2 } } { \delta } } \right)
$$

for any realized pair $( p , s )$ with probability at least $1 - \delta .$ . For every recovery phase, we have $p \leq T$ and $s \leq | B _ { p } ^ { \mathrm { i x } } \cap [ T ] | \leq L _ { T }$ , which concludes the proof. □

## C Proof of Theorem 4: stochastic regime

In this section, we consider primarily the loss representation $\ell _ { t , i } = 1 - R _ { t , i }$ to keep the notation consistent with our loss estimators. Due to the similar phase-dependent policy structure, our analysis follows the outline of Stahlbuhk et al. (2021), although the technical arguments are different.

Recall that, in the stochastic regime, $A _ { t } \in [ 0 , 1 ]$ and $R _ { t } \in [ 0 , 1 ] ^ { K }$ are i.i.d. across all rounds. First, for each busy period $\begin{array} { r } { B _ { p } . } \end{array}$ , we define the good-period indicator by

$$
\mathcal { I } _ { p } : = \mathbb { 1 } \left[ \hat { i } _ { p } ^ { * } = i ^ { * } , \mathcal { B } _ { p } ^ { \mathrm { i x } } = \varnothing \right] .\tag{15}
$$

In other words, $\mathcal { I } _ { p } = 1$ implies that the empirical best arm in exploration phases is correctly identified and the queue becomes empty without entering the recovery phase, and otherwise, $\mathcal { I } _ { p } = 0$ . Therefore, our policy always selects $i ^ { * }$ throughout the busy period $B _ { p }$ if $\mathcal { I } _ { p } = 1$ . Then, we have the following lemmas, whose proofs are given in Appendix D.

Lemma 11. For $K \geq 2$ and $\epsilon = \mu _ { i ^ { * } } - \lambda \in ( 0 , 1 ]$ , Algorithm 1 satisfies that

$$
\mathbb { E } \left[ \sum _ { t \in \mathcal { B } _ { p } } ( Q _ { t } ^ { \pi } - Q _ { t } ^ { i ^ { * } } ) ^ { + } \right] \leq C _ { p } \operatorname* { P r } ( \mathcal { T } _ { p } = 0 ) ,
$$

where

$$
C _ { p } \leq c \bigg ( p \sqrt { p \log p } + \frac { p \log p } { \epsilon ^ { 2 } } + \frac { K ^ { 2 } \log ^ { 2 } K } { \epsilon ^ { 4 } } \bigg )
$$

for a universal constant $c > 0$

Lemma 12. For $p \ge \lceil 1 + \frac { 2 } { \epsilon } \rceil$ , it holds that

$$
\operatorname* { P r } ( \mathcal { I } _ { p } = 0 ) \leq K \exp \left( - \frac { \Delta _ { \operatorname* { m i n } } ^ { 2 } p \rho } { 8 K } \right) + ( p + 1 ) ^ { - 7 } + \exp \left( - \frac { \epsilon ^ { 2 } p } { 8 } \right) .
$$

Proof of Theorem 4. Recall that busy and empty periods are defined by the policy’s queue, so $Q _ { t } ^ { i ^ { * } } > 0$ is possible even during an empty period. On the good busy period $B _ { p }$ where $\mathcal { I } _ { p } = 1$ , Algorithm 1 selects $i ^ { * }$ throughout the exploitation phase, equivalent to whole busy period, including its first round $\beta _ { p }$ . Since $Q _ { \beta _ { p } } ^ { \pi } = 0 \leq Q _ { \beta _ { p } } ^ { i ^ { * } }$ , monotonicity of the queue recursion under the same arrivals and services imply that

$$
Q _ { t } ^ { \pi } \leq Q _ { t } ^ { i ^ { * } } , \forall t \in \mathcal { B } _ { p } , \forall p : \mathcal { I } _ { p } = 1 \implies \sum _ { t \in \mathcal { B } _ { p } } ( Q _ { t } ^ { \pi } - Q _ { t } ^ { i ^ { * } } ) ^ { + } = 0 .\tag{16}
$$

Establishing this result for our policy requires the assumption that arrivals are observed before decision, as explained in Appendix C.1. Therefore, if $\mathcal { I } _ { p } = 1 , B _ { p }$ contribute no positive cumulative queue length regret so that considering the busy period with $\mathcal { I } _ { p } = 0$ is sufficient.

Since $Q _ { t } ^ { \pi } = 0$ in the empty period, we have

$$
\sum _ { t = 1 } ^ { \infty } \mathbb { E } \left[ \left( Q _ { t } ^ { \pi } - Q _ { t } ^ { i ^ { * } } \right) \right] \leq \sum _ { p = 1 } ^ { \infty } \mathbb { E } \left[ \sum _ { t \in \mathcal { B } _ { p } } \left( Q _ { t } ^ { \pi } - Q _ { t } ^ { i ^ { * } } \right) \right] .
$$

By Lemma 11, there exists a constant $c > 0$ satisfying for all $p \geq 1$

$$
\mathbb { E } \left[ \sum _ { { t } \in \mathcal { B } _ { p } } ( Q _ { t } ^ { \pi } - Q _ { t } ^ { i ^ { * } } ) \right] \leq c \Bigg ( p \sqrt { p \log p } + \frac { p \log p } { \epsilon ^ { 2 } } + \frac { K ^ { 2 } \log ^ { 2 } K } { \epsilon ^ { 4 } } \Bigg ) \operatorname* { P r } ( \mathcal { T } _ { p } = 0 ) .
$$

Let $\begin{array} { r } { p _ { \Delta } = \lceil 1 + \frac { 2 } { \epsilon } \rceil } \end{array}$ . Then, by Lemma 12 and log $p \leq p .$ it suffices to show that there exists a finite constant $C _ { \mathrm { s t o } }$ satisfying

$$
\begin{array} { r l r } {  { \sum _ { p = p _ { \Delta } } ^ { \infty } c \bigg ( ( 1 + \epsilon ^ { - 2 } ) p ^ { 2 } + \frac { K ^ { 2 } \log ^ { 2 } K } { \epsilon ^ { 4 } } \bigg ) \bigg ( K \exp \bigg ( - \frac { \Delta _ { \operatorname* { m i n } } ^ { 2 } p \rho } { 8 K } \bigg ) + ( p + 1 ) ^ { - 7 } + \exp \bigg ( - \frac { \epsilon ^ { 2 } p } { 8 } \bigg ) \bigg ) } } \\ & { } & { + \displaystyle \sum _ { p = 1 } ^ { p _ { \Delta } } c \bigg ( ( 1 + \epsilon ^ { - 2 } ) p ^ { 2 } + \frac { K ^ { 2 } \log ^ { 2 } K } { \epsilon ^ { 4 } } \bigg ) \leq C _ { \mathrm { s t o } } . } \end{array}
$$

For simplicity, let

$$
a = \frac { \Delta _ { \mathrm { m i n } } ^ { 2 } \rho } { 8 K } , b = \frac { \epsilon ^ { 2 } } { 8 } , C ^ { \prime } = \frac { K ^ { 2 } \log ^ { 2 } K } { \epsilon ^ { 4 } } .
$$

First term. Consider the following term,

$$
\sum _ { p = p _ { \Delta } } ^ { \infty } c K ( ( 1 + \epsilon ^ { - 2 } ) p ^ { 2 } + C ^ { \prime } ) \exp ( - a \sqrt { p } ) .
$$

Then, by simple arithmetic calculation, we have

$$
\begin{array} { r l } { \displaystyle \sum _ { p = p _ { \Delta } } ^ { \infty } c K ( ( 1 + \epsilon ^ { - 2 } ) p ^ { 2 } + C ^ { \prime } ) e ^ { - a p } \le c e ^ { a } \int _ { p _ { \Delta } } ^ { \infty } ( ( 1 + \epsilon ^ { - 2 } ) x ^ { 2 } + C ^ { \prime } ) e ^ { - a x } \mathrm { d } x } & { } \\ { \displaystyle \le c K e ^ { a } \biggl ( \frac { 2 ( 1 + \epsilon ^ { - 2 } ) } { a ^ { 3 } } + \frac { C ^ { \prime } } { a } \biggr ) } & { } \\ { \displaystyle \le \mathcal { O } \biggl ( K \exp \biggl ( \frac { \Delta _ { \operatorname* { m i n } } ^ { 2 } \rho } { K } \biggr ) \biggl ( \frac { K ^ { 3 } } { \epsilon ^ { 2 } \Delta _ { \operatorname* { m i n } } ^ { 6 } \rho ^ { 3 } } + \frac { K ^ { 3 } \log ^ { 2 } K } { \epsilon ^ { 4 } \Delta _ { \operatorname* { m i n } } ^ { 2 } \rho } \biggr ) \biggr ) } \\ { \displaystyle } & { = \mathcal { O } \biggl ( \frac { K ^ { 4 } } { \epsilon ^ { 2 } \Delta _ { \operatorname* { m i n } } ^ { 6 } \rho ^ { 3 } } + \frac { K ^ { 4 } \log ^ { 2 } K } { \epsilon ^ { 4 } \Delta _ { \operatorname* { m i n } } ^ { 2 } \rho } \biggr ) . } \end{array}
$$

Second term. The second part is indeed negligible since $p ^ { - 7 }$ is dominating, i.e.,

$$
\sum _ { p = p _ { \Delta } } ^ { \infty } c ( ( 1 + \epsilon ^ { - 2 } ) p ^ { 2 } + C ^ { \prime } ) ( p + 1 ) ^ { - 7 } \leq \mathcal { O } ( 1 + \epsilon ^ { - 2 } + C ^ { \prime } ) .
$$

Third term. Similarly to the first term, we obtain

$$
\begin{array} { r l } { \displaystyle \sum _ { p = p _ { \Delta } } ^ { \infty } c ( ( 1 + \epsilon ^ { - 2 } ) p ^ { 2 } + C ^ { \prime } ) e ^ { - b p } \le c e ^ { b } \displaystyle \int _ { p _ { \Delta } } ^ { \infty } ( ( 1 + \epsilon ^ { - 2 } ) x ^ { 2 } + C ^ { \prime } ) e ^ { - b x } \mathrm { d } x } & { } \\ { \le c e ^ { b } \displaystyle \left( \frac { 2 ( 1 + \epsilon ^ { - 2 } ) } { b ^ { 3 } } + \frac { C ^ { \prime } } { b } \right) } & { } \\ { \le \mathcal { O } ( \epsilon ^ { - 8 } + C ^ { \prime } \epsilon ^ { - 2 } ) } \end{array}
$$

Fourth term. Finally, we have

$$
\begin{array} { r } { \displaystyle \sum _ { p = 1 } ^ { p _ { \Delta } } c \bigg ( ( 1 + \epsilon ^ { - 2 } ) p ^ { 2 } + \frac { K ^ { 2 } \log ^ { 2 } K } { \epsilon ^ { 4 } } \bigg ) \leq \mathcal { O } \bigg ( \epsilon ^ { - 2 } p _ { \Delta } ^ { 3 } + \frac { K ^ { 2 } \log K p _ { \Delta } } { \epsilon ^ { 4 } } \bigg ) } \\ { \leq \mathcal { O } \bigg ( \epsilon ^ { - 2 } \Delta _ { \operatorname* { m i n } } ^ { - 6 } + \frac { K ^ { 2 } \log K } { \Delta _ { \operatorname* { m i n } } ^ { 2 } \epsilon ^ { 4 } } \bigg ) . } \end{array}
$$

Conclusion. In sum, by definition of $p _ { \Delta }$ , we obtain

$$
\sum _ { p = 1 } ^ { \infty } \mathbb { E } \left[ \sum _ { { t } \in { \mathcal { B } } _ { p } } \left( Q _ { { t } } ^ { \pi } - Q _ { { t } } ^ { { i } ^ { * } } \right) _ { + } \right] \leq C _ { \mathrm { s t o } } = \mathrm { p o l y } \big ( K , \Delta _ { \operatorname* { m i n } } ^ { - 1 } , \epsilon ^ { - 1 } , \log K \big ) ,
$$

which is finite since $\epsilon > 0 , \Delta _ { \mathrm { m i n } } > 0$ , and $K < \infty$

Therefore, for any $T \geq 1$

$$
\mathrm { R e g } _ { \pi } ^ { \mathrm { c u m } } ( T ) = \sum _ { t = 1 } ^ { T } \mathbb { E } \Big [ Q _ { t } ^ { \pi } - Q _ { t } ^ { i ^ { * } } \Big ] \leq \sum _ { t = 1 } ^ { \infty } \mathbb { E } \Big [ ( Q _ { t } ^ { \pi } - Q _ { t } ^ { i ^ { * } } ) ^ { + } \Big ] \leq C _ { \mathrm { s t o } } ,
$$

which concludes the proof.

## C.1 Observation of arrivals

Observing $A _ { t }$ before selecting $i _ { t }$ allows the policy to distinguish an opportunity for exploration from the first round of a new busy period. This distinction prevents the first decision of a good busy period from inducing excess backlog relative to the comparator and supports the queue comparison used in our stochastic analysis.

Let $\beta _ { p }$ be the first round of the busy period $\boldsymbol { B } _ { p } .$ . By definition,

$$
Q _ { \beta _ { p } } ^ { \pi } = 0 , \quad \mathrm { a n d } \quad A _ { \beta _ { p } } > 0 .
$$

Therefore, on the event $\mathcal { I } _ { p } = 1$ , Algorithm 1 selects the optimal arm $i ^ { * }$ throughout $B _ { p } ,$ , including the round $\beta _ { p } .$ . Since busy and empty periods are defined by the policy’s queue, $Q _ { t } ^ { i ^ { * } }$ need not be empty at $\beta _ { p } ,$ , which implies $Q _ { \beta _ { p } } ^ { \pi } \leq Q _ { \beta _ { p } } ^ { i ^ { * } }$ . Since both queues subsequently receive the same arrivals and service, monotonicity of the queue recursion preserves this ordering. Therefore,

$$
\begin{array} { r } { \mathcal { I } _ { p } = 1 \implies Q _ { t } ^ { \pi } \le Q _ { t } ^ { i ^ { * } } , \forall t \in \mathcal { B } _ { p } , } \end{array}
$$

which implies

$$
\sum _ { t \in { \mathcal { B } } _ { p } } ( Q _ { t } ^ { \pi } - Q _ { t } ^ { i ^ { * } } ) \leq 0 .
$$

Thus, these good busy periods contribute no positive cumulative queue length regret.

This argument relies on selecting $i ^ { * }$ from the first round of the busy period, where $Q _ { t } ^ { \pi } = 0$ and $A _ { t } > 0$ To see what can fail without arrival observation, consider the modification π¯ that explores uniformly whenever $Q _ { t } ^ { \bar { \pi } } = 0$ , before observing $A _ { t }$ . If $A _ { t } = 0$ , exploration does not change the queue status regardless of the selected arm. If $A _ { t } > 0$ , however, the decision before the newly arrived data, giving

$$
Q _ { t + 1 } ^ { \bar { \pi } } = ( A _ { t } - R _ { t , i _ { t } } ) ^ { + } \geq 0 .
$$

In particular, if $Q _ { t } ^ { i ^ { * } } = 0$ as well, then

$$
Q _ { t + 1 } ^ { \bar { \pi } } - Q _ { t + 1 } ^ { i ^ { * } } = ( A _ { t } - R _ { t , i _ { t } } ) ^ { + } - ( A _ { t } - R _ { t , i ^ { * } } ) ^ { + } ,
$$

which can be positive. The queue comparison can therefore incur linear regret even when the empirical best arm is already correct and the policy selects it in every subsequent round. Although this discrepancy may disappear after the queue becomes empty, similar costs can incur at the start of later busy periods.

Indeed, this modification can incur linear cumulative regret on an instance satisfying our stochastic assumptions. Consider an instance

$$
K = 2 , \quad A _ { t } \sim \mathrm { B e r n o u l l i } ( 1 / 2 ) , \quad R _ { t , 1 } = 1 , \quad R _ { t , 2 } = 0 , \forall t .
$$

Here, $\rho = 1 / 2$ , arrivals and services are independent, and $i ^ { * } = 1$ is uniquely optimal and stabilizing. The comparator queue satisfies $Q _ { t } ^ { i ^ { * } } = 0$ for every t.

When $Q _ { t } ^ { \bar { \pi } } = 0 .$ , uniform exploration incurs one unit of backlog if $A _ { t } = 1$ and arm 2 is selected, an event with conditional probability $1 / 4$ . When $Q _ { t } ^ { \bar { \pi } } > 0$ , the queue is integer-valued and therefore at least one. On the event $A _ { t } = 1$ , service of at most one unit cannot empty this queue, regardless of the selected arm. Thus,

$$
\mathbb { E } \big [ Q _ { t + 1 } ^ { \bar { \pi } } \big | Q _ { t } ^ { \bar { \pi } } = q \big ] \geq \left\{ 1 / 4 , \ q = 0 , \right.
$$

This implies that for every $T \geq 1$ , we have

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \left( Q _ { t } ^ { \bar { \pi } } - Q _ { t } ^ { i ^ { * } } \right) \right] \geq \frac { T - 1 } { 4 } .
$$

Therefore, uniform exploration based only on the pre-arrival condition $Q _ { t } ^ { \bar { \pi } } = 0$ cannot achieve a sublinear stochastic cumulative-regret guarantee in our queue recursion definition.

The corresponding issue does not arise under the queue dynamics of Stahlbuhk et al. (2021), where they use

$$
Q _ { t + 1 } = ( Q _ { t } - R _ { t , i _ { t } } ) ^ { + } + A _ { t } .
$$

In their model, arrivals in round t become available for service only in the following round. Therefore,

$$
Q _ { t } ^ { \pi } = 0 \Longrightarrow Q _ { t + 1 } ^ { \pi } = A _ { t } \leq ( Q _ { t } ^ { i ^ { * } } - R _ { t , i ^ { * } } ) ^ { + } + A _ { t } = Q _ { t + 1 } ^ { i ^ { * } }
$$

holds regardless of the decision in round t. Under their queue dynamics, exploration during empty periods preserves this queue comparison.

In other words, our arrivals-first assumption is introduced since we adopt the standard Lindley recursion

$$
Q _ { t + 1 } ^ { \pi } = ( Q _ { t } ^ { \pi } + A _ { t } - R _ { t , i _ { t } } ) ^ { + } ,
$$

which is also used in prior work on stochastic and adversarial queueing bandits (Krishnasamy et al., 2016; Freund et al., 2024; Kim and Oh, 2024; Krishnakumar and Sinha, 2025; Bae et al., 2026). Under this formulation, newly arriving data can receive service in the same round.

Our additional assumption that arrivals are observed before decision allows the policy to identify rounds in which exploration cannot create backlog. Specifically, the policy explores when $Q _ { t } ^ { \pi } + A _ { t } = 0$ and begins exploitation immediately when $Q _ { t } ^ { \pi } = 0$ and $A _ { t } > 0$ . Only the indicator $\mathbb { 1 } [ Q _ { t } ^ { \pi } + A _ { t } = 0 ]$ is needed before the decision and the exact arrival amount can be revealed afterward for the queue updates.

Remark 2 (Adversarial guarantee without arrival observation). Consider a variant of Algorithm 1 that explores whenever $Q _ { t } ^ { \pi } = 0$ and starts exploitation at the first round with $Q _ { t } ^ { \pi } > 0 , \mathrm { i . e . }$ , where its busy periods are maximal intervals on which $Q _ { t } ^ { \pi } > 0$ . Let $\tilde { L } _ { T }$ denote the longest such period up to $T$ . The bound of Theorem 3 holds for this variant with $L _ { T }$ replaced by $\tilde { L } _ { T }$ . The variant does not need to observe $A _ { t } ,$ since its auxiliary queue $O _ { t }$ can be updated from consecutive queue lengths and the observed service of the selected channel. Thus, arrival observation is required for our stochastic cumulative-regret guarantee, but not for the adversarial guarantee.

## D Proofs of Lemmas for Theorem 4

## D.1 Proof of Lemma 11

To prove this lemma, we first need the following result, whose proof is given in Appendix F.

Lemma 13. Let $\tau _ { q } : = \operatorname* { i n f } \{ s \geq 1 : Q _ { \tau ( p , s ) } ^ { \pi } = 0 , Q _ { \tau ( p , 1 ) } = q \}$ denote the total number of local recovery rounds with the first round queue length $q > 0 .$ . Then, for any $K \geq 2$ and $\epsilon = \mu _ { i ^ { * } } - \lambda \in ( 0 , 1 ]$ , it holds that

$$
\operatorname* { P r } ( \tau _ { q } > s ) \le \exp \left( - \frac { \epsilon ^ { 2 } s } { 1 8 } \right) + 2 \exp \left( - \frac { \epsilon } { 1 8 } \sqrt { \frac { s \log K } { K } } \right) .
$$

for $\begin{array} { r } { s \ge s _ { 0 } ( q ) = \operatorname* { m a x } ( \frac { 3 q } { \epsilon } , \frac { 5 7 6 K \log K } { \epsilon ^ { 2 } } ) } \end{array}$ . Therefore, $\tau _ { q }$ has a finite r-th moment satisfying

$$
\mathbb { E } [ \tau _ { q } ^ { r } ] \le ( s _ { 0 } ( q ) ) ^ { r } + e ^ { 1 / 1 8 } r \Gamma ( r ) \biggl ( \frac { 1 8 } { \epsilon ^ { 2 } } \biggr ) ^ { r } + 4 e ^ { 1 / 1 8 } r \Gamma ( 2 r ) \biggl ( \frac { 3 6 4 K } { \epsilon ^ { 2 } \log K } \biggr ) ^ { r } ,
$$

which implies that every fixed moment is polynomially bounded in terms of $\cdot _ { q , }$ K and ϵ.

Proof. Since $\begin{array} { r } { \sum _ { t \in \mathcal { B } _ { v } } ( Q _ { t } ^ { \pi } - Q _ { t } ^ { i ^ { * } } ) _ { + } = 0 \operatorname { i f } \mathcal { B } _ { p } } \end{array}$ satisfies $\mathcal { I } _ { p } = 1$ as in (16), it suffices to consider the case of $\mathcal { I } _ { p } = 0$ . Hence, we consider the exploitation phase and recovery phase, respectively.

Exploitation phase. Note that $O _ { t } = 0$ and thus $E _ { t } = Q _ { t } ^ { \pi }$ in the stochastic setting with $A _ { t } \in [ 0 , 1 ]$ by (13). Therefore, at the n-th round of exploitation phase in busy period $B _ { p }$ , threshold condition implies that

$$
Q _ { t } ^ { \pi } \leq g _ { p } ( n ) = 4 { \sqrt { n \log ( p + 1 ) } } .
$$

Since the exploitation phase occurs at most p rounds in $B _ { p }$ by time-out condition, it holds

$$
\begin{array} { r l r } {  { \sum _ { t \in \mathcal { B } _ { p } ^ { \mathrm { e x } } } ( Q _ { t } ^ { \pi } - Q _ { t } ^ { i ^ { * } } ) _ { + } \leq \sum _ { n = 1 } ^ { p } 4 \sqrt { n \log ( p + 1 ) } } } \\ & { } & { \leq 4 p ^ { 3 / 2 } \sqrt { \log ( p + 1 ) } . } \end{array}
$$

If it does not transition to the recovery phase, $\boldsymbol { B } _ { p } ^ { \mathrm { e x } } = \boldsymbol { B } _ { p }$ holds, which concludes the proof.

Recovery phase. Therefore, we consider the case where the recovery phase exists. Let $Q _ { \tau ( p , 1 ) } ^ { \pi } = q$ , which implies that the exploitation ends with queue length $q > 0$ and thus satisfies

$$
q \leq g _ { p } ( n ) + 1 \leq 4 { \sqrt { n \log ( p + 1 ) } } + 1 .\tag{17}
$$

For simplicity, we index the recovery phase from 1 so that $Q _ { 0 } ^ { \pi } = q .$ . Since $A _ { t } \in [ 0 , 1 ]$ , it holds that

$$
\begin{array} { r l r } {  { \sum _ { s = 0 } ^ { \tau _ { q } - 1 } ( Q _ { s } ^ { \pi } - Q _ { s } ^ { i ^ { * } } ) _ { + } \leq \sum _ { s = 0 } ^ { \tau _ { q } - 1 } ( q + s ) } } \\ & { } & { \leq q \tau _ { q } + \frac { \tau _ { q } ^ { 2 } } { 2 } . } \end{array}
$$

By applying Lemma 13 for $r = 1 , 2 .$ , we obtain

$$
\begin{array} { r l r } {  { \mathbb { E } [ \sum _ { s = 0 } ^ { \tau _ { - 1 } } ( Q _ { s } ^ { \pi } - Q _ { s } ^ { \pi ^ { * } } ) _ { + } ] \leq \mathcal { O } \bigg ( q s _ { 0 } ( q ) + ( s _ { 0 } ( q ) ) ^ { 2 } + \frac { 1 } { \epsilon ^ { 2 } } + \frac { K } { \epsilon ^ { 2 } \log K } + \frac { 1 } { \epsilon ^ { 4 } } + \frac { K ^ { 2 } } { \epsilon ^ { 4 } \log ^ { 2 } K } \bigg ) } } \\ & { } & { \leq \mathcal { O } \bigg ( \frac { q ^ { 2 } } { \epsilon } + \frac { q K \log K } { \epsilon ^ { 2 } } + \frac { q ^ { 2 } } { \epsilon ^ { 2 } } + \frac { K ^ { 2 } \log ^ { 2 } K } { \epsilon ^ { 4 } } + \frac { K } { \epsilon ^ { 2 } \log K } + \frac { 1 } { \epsilon ^ { 4 } } + \frac { K ^ { 2 } } { \epsilon ^ { 4 } \log ^ { 2 } K } \bigg ) } \\ & { } & { \leq \mathcal { O } \bigg ( \frac { q ^ { 2 } } { \epsilon ^ { 2 } } + \frac { q K \log K } { \epsilon ^ { 2 } } + \frac { K ^ { 2 } \log ^ { 2 } K } { \epsilon ^ { 4 } } \bigg ) } \\ & { } & { \leq \mathcal { O } \bigg ( \frac { q ^ { 2 } } { \epsilon ^ { 2 } } + \frac { K ^ { 2 } \log ^ { 2 } K } { \epsilon ^ { 4 } } \bigg ) \quad \quad \quad \quad \quad \quad \quad ( \cdot \ a b \leq ( a ^ { 2 } + b ^ { 2 } ) / 2 ) } \\ & { } & { \leq \mathcal { O } \bigg ( \frac { p \log p } { \epsilon ^ { 2 } } + \frac { K ^ { 2 } \log ^ { 2 } K } { \epsilon ^ { 4 } } \bigg ) , \quad \quad \quad \quad ( \mathrm { b y ~ ( \it 1 7 ) } ) } \end{array}
$$

which concludes the proof.

## D.2 Proof of Lemma 12

The following results are required to provide the upper bounds on the probability of $\mathcal { I } _ { p } = 0$ , whose proofs are given in Appendix F.

Lemma 14. For any $m _ { 0 } \geq 1$ , it holds that

$$
\operatorname* { P r } \left( \exists m \ge m _ { 0 } : \operatorname* { a r g m i n } _ { i \in [ K ] } \hat { L } _ { m + 1 , i } ^ { \varnothing } \neq i ^ { * } \right) \le ( K - 1 ) \exp \left( - \frac { m _ { 0 } \Delta _ { \operatorname* { m i n } } ^ { 2 } } { 4 K } \right) .
$$

Note that by our index notation, $\hat { L } _ { 1 , i } ^ { \varnothing } = 0$ , and thus the probability is $1 - 1 / K$ for $m = 0$ case.

ProofofLemma 12. By definition, $\mathcal { I } _ { p } = 0$ if its empirical best arm is incorrect or if it enters the recovery phase despite identifying the correct best arm. Let

$$
m _ { p } : = \sum _ { j = 1 } ^ { p } | { \mathcal { E } } _ { j } |
$$

be the number of exploration rounds up to the beginning of busy period $B _ { p }$ . Since the $\mathcal { E } _ { j }$ can be empty due to positive arrivals right after the queue becomes empty, let $\alpha _ { j }$ be the first round with $Q _ { t } ^ { \pi } = 0$ after the $( j - 1 ) { \cdot } \mathtt { t h }$ the busy period, with $\alpha _ { 1 } = 1$

$$
Z _ { j } : = \mathbb { 1 } [ A _ { \alpha _ { j } } = 0 ] .
$$

Therefore, $Z _ { j } = 1$ implies that Algorithm 1 is in exploration phase at the round $\alpha _ { j }$ , which implies

$$
m _ { p } \geq \sum _ { j = 1 } ^ { p } Z _ { j } .
$$

Since $\alpha _ { j }$ is determined by the history before its arrival, i.i.d. arrival assumption implies that $Z _ { j }$ are independent Bernoulli random variables with success probability $\rho .$ Hence, its summation follows the Binomial distribution with $p$ trials and $\rho$ success probability. By applying multiplicative Chernoff bound, we obtain

$$
\operatorname* { P r } \Big ( m _ { p } < \frac { \rho p } { 2 } \Big ) \leq \exp \Big ( - \frac { \rho p } { 8 } \Big ) .\tag{18}
$$

Incorrect empirical best arm. By the definition of $i _ { p } ^ { * }$ in (6), Lemma 14 implies that for $m ^ { * } = \rho p / 2$

$$
\begin{array} { r l r } { \operatorname* { P r } \big ( i _ { p } ^ { * } \neq i ^ { * } \big ) \leq \operatorname* { P r } ( m _ { p } < m ^ { * } ) + \operatorname* { P r } \bigg ( \underset { i \in [ K ] } { \arg \operatorname* { m i n } } \hat { L } _ { m + 1 , i } ^ { \varnothing } \neq i ^ { * } , m _ { p } \geq m ^ { * } \bigg ) } \\ & { } & \\ { \leq \exp \Big ( - \frac { \rho p } { 8 } \Big ) + ( K - 1 ) \exp \bigg ( - \frac { \Delta _ { \operatorname* { m i n } } ^ { 2 } \rho p } { 8 K } \bigg ) } \\ & { } & { \leq K \exp \bigg ( - \frac { \Delta _ { \operatorname* { m i n } } ^ { 2 } \rho p } { 8 K } \bigg ) . \quad \quad \quad \quad ( \cdot \Delta _ { \operatorname* { m i n } } \in ( 0 , 1 ] , K \geq 2 ) } \end{array}
$$

Recovery phase with correct best arm. Let us consider the case where $i _ { p } ^ { * } = i ^ { * }$ holds. Then, during the exploitation phase $\boldsymbol { B } _ { p } ^ { \mathrm { e x } }$ , the policy always selects $i ^ { * }$ . Define the residual queue length by $\bar { Q } _ { s } : = A _ { s } - R _ { s , i ^ { * } }$ which satisfies

$$
\bar { Q } _ { s } \in [ - 1 , 1 ] , \quad \mathbb { E } [ \bar { Q } _ { s } ] = \lambda - \mu _ { i ^ { * } } = - \epsilon .
$$

Therefore, $\bar { Q } _ { s } + \epsilon$ is of expectation zero.

Case 1: threshold violation. Consider the case the exploitation phase end because the queue length exceeds the threshold. Notice that the threshold violation cannot occur at local exploitation round $n = 1$ since $Q _ { \beta _ { v } } ^ { \pi } = 0$ and at $n = 2$ since $Q _ { \beta _ { p } + 1 } ^ { \pi } \leq 1 < g _ { p } ( 2 )$ by definitions.

If such threshold violation occurs at round $n \geq 3$ , this means that the preceding $n - 1$ times selecting $i ^ { * }$ did not empty the queue. Therefore,

$$
Q _ { \beta _ { p } + n - 1 } ^ { \pi } = A _ { \beta _ { p } } - R _ { \beta _ { p } , i ^ { * } } + \sum _ { j = 1 } ^ { n - 2 } \bar { Q } _ { j } \leq 1 + \sum _ { j = 1 } ^ { n - 2 } \bar { Q } _ { j } .
$$

By applying Hoeffding’s inequality, we obtain

$$
\begin{array} { r l } & { \operatorname* { P r } \Big ( Q _ { \beta _ { p } + n - 1 } ^ { \pi } > g _ { p } ( n ) \Big ) \leq \operatorname* { P r } \left( \displaystyle \sum _ { j = 1 } ^ { n - 2 } ( \bar { Q } _ { s } + \epsilon ) > g _ { p } ( n ) - 1 + \epsilon ( n - 2 ) \right) } \\ & { \qquad \leq \exp \Big ( - \displaystyle \frac { \left( g _ { p } ( n ) - 1 \right) ^ { 2 } } { n - 2 } \Big ) \leq \exp \Big ( - \displaystyle \frac { 9 n \log ( p + 1 ) } { n - 2 } \Big ) \leq ( p + 1 ) ^ { - 8 } . } \end{array}
$$

Since such threshold violation occur at most once for each busy period, a union bound gives

$$
\operatorname* { P r } \Bigl ( \mathrm { t h r e s h o l d ~ v i o l a t i o n ~ r e c o v e r y } , \hat { i } _ { p } ^ { * } = i ^ { * } \Bigr ) \leq p ( p + 1 ) ^ { - 8 } \leq ( p + 1 ) ^ { - 7 } .
$$

Case 2. time-out. If the recovery phase starts by the time-out after $p$ optimal arm selections, the queue remains positive after these $p$ rounds. Therefore,

$$
0 < Q _ { \beta _ { p } + p } ^ { \pi } \leq 1 + \sum _ { j = 1 } ^ { p - 1 } { \bar { Q } } _ { j } \implies \sum _ { j = 1 } ^ { p - 1 } ( { \bar { Q } } _ { j } + \epsilon ) > \epsilon ( p - 1 ) - 1 .
$$

Here, for $p \ge 1 + 2 / \epsilon , \epsilon ( p - 1 ) - 1 \ge \epsilon ( p - 1 ) / 2$ holds. By applying Hoeffding’s inequality, we obtain

$$
\operatorname* { P r } \Bigl ( \mathrm { t i m e - o u t ~ r e c o v e r y } , \hat { i } _ { p } ^ { * } = i ^ { * } \Bigr ) \leq \exp \Bigl ( - \frac { \epsilon ^ { 2 } ( p - 1 ) } { 4 } \Bigr ) \leq \exp \biggl ( - \frac { \epsilon ^ { 2 } p } { 8 } \Bigr ) .
$$

Finally, combining all the results with a union bound concludes the proof.

## E Proof of Corollary 5

Proof. By definition, we have

$$
Q _ { t } ^ { \pi } - \operatorname* { m i n } _ { i \in [ K ] } Q _ { t } ^ { i } \le Q _ { t } ^ { \pi } - Q _ { t } ^ { i ^ { * } } + Q _ { t } ^ { i ^ { * } } - \operatorname* { m i n } _ { i \in [ K ] } Q _ { t } ^ { i } ,
$$

which implies

$$
\mathrm { R e g } _ { \pi } ^ { \operatorname* { m a x } } ( T ) \leq \operatorname* { m a x } _ { t \in [ T ] } \Big ( Q _ { t } ^ { \pi } - Q _ { t } ^ { i ^ { * } } \Big ) ^ { + } + \operatorname* { m a x } _ { t \in [ T ] } \bigg ( Q _ { t } ^ { i ^ { * } } - \operatorname* { m i n } _ { i \in [ K ] } Q _ { t } ^ { i } \bigg ) ^ { + } .
$$

By Theorem 4, the first term satisfies

$$
\mathbb { E } \bigg [ \operatorname* { m a x } _ { t \in [ T ] } \Big ( Q _ { t } ^ { \pi } - Q _ { t } ^ { i ^ { * } } \Big ) _ { + } \bigg ] \leq \sum _ { t = 1 } ^ { \infty } \mathbb { E } \bigg [ \Big ( Q _ { t } ^ { \pi } - Q _ { t } ^ { i ^ { * } } \Big ) _ { + } \bigg ] \leq C _ { \mathrm { s t o } } .
$$

For the second term, let us consider the busy period $B _ { p } ^ { * }$ in terms of constant policy of selecting $i ^ { * }$ , with its initial round $\beta _ { p } ^ { * }$ . Then, for $t \in B _ { p } ^ { * }$ , we have

$$
{ Q _ { t } ^ { i ^ { * } } } = \sum _ { s = \beta _ { p } ^ { * } } ^ { t - 1 } ( A _ { s } - R _ { s , i ^ { * } } ) .
$$

During this period, for any $j \neq i ^ { * }$ , it holds that

$$
Q _ { t } ^ { j } \geq \sum _ { s = \beta _ { p } ^ { * } } ^ { t - 1 } ( A _ { s } - R _ { s , j } ) .
$$

Hence, we obtain

$$
Q _ { t } ^ { i ^ { * } } - Q _ { t } ^ { j } \leq \sum _ { s = \beta _ { p } ^ { * } } ^ { t - 1 } ( R _ { s , j } - R _ { s , i ^ { * } } ) \leq \operatorname* { m a x } _ { t ^ { \prime } \in [ t - 1 ] } \sum _ { s = t ^ { \prime } } ^ { t - 1 } ( R _ { s , j } - R _ { s , i ^ { * } } ) .
$$

Let $g _ { s } ^ { j } = R _ { s , j } - R _ { s , i ^ { * } }$ , which satisfies $g _ { s } ^ { j } \in [ - 1 , 1 ]$ and $\mathbb { E } [ g _ { s } ^ { j } ] = - \Delta _ { j }$ . For any $n \in \mathbb { N } ,$ , Hoeffding’s inequality gives that

$$
\begin{array}{c} \operatorname* { P r } \left( \sum _ { s = 1 } ^ { n } g _ { s } ^ { j } \geq x \right) = \operatorname* { P r } \left( \sum _ { s = 1 } ^ { n } ( g _ { s } ^ { j } + \Delta _ { j } ) \geq x + n \Delta _ { j } \right) \leq \exp \left( - { \frac { ( x + n \Delta _ { j } ) ^ { 2 } } { 2 n } } \right)  \\ { \leq e ^ { - 2 x \Delta _ { j } } . \qquad \ ( \cdot \ ( a + b ) ^ { 2 } \geq 4 a b ) } \end{array}
$$

By taking a union bound, we have for any $x > 0$ that

$$
\operatorname* { P r } \left( \operatorname* { m a x } _ { j \neq i ^ { * } } \operatorname* { m a x } _ { 1 \leq t ^ { \prime } \leq t \leq T } \sum _ { s = t ^ { \prime } } ^ { t } g _ { s } ^ { j } \geq x \right) \leq \operatorname* { m i n } \left( 1 , ( K - 1 ) T ^ { 2 } e ^ { - 2 x \Delta _ { \operatorname* { m i n } } } \right) .
$$

This is because the possible choices of interval pairs $[ t ^ { \prime } , t ] \subseteq [ T ]$ is $T ( T + 1 ) / 2 \leq T ^ { 2 }$ . Let $\begin{array} { r } { x _ { 0 } = \frac { \log \left( ( K - 1 ) T ^ { 2 } \right) } { 2 \Delta _ { \operatorname* { m i n } } } } \end{array}$ so that $( K - 1 ) T ^ { 2 } e ^ { - 2 x \Delta _ { \mathrm { m i n } } } = 1$ . Since $\begin{array} { r } { E [ X ] = \int _ { 0 } ^ { \infty } P r ( X \geq x ) \mathrm { d } x } \end{array}$ for any non-negative random variable $X ,$ , we obtain

$$
\begin{array} { r l r } {  { \mathbb { E } [ \operatorname* { m a x } _ { t \in [ T ] } ( Q _ { t } ^ { i ^ { * } } - \frac { \operatorname* { m i n } } { i \neq i ^ { * } } Q _ { t } ^ { i } ) ^ { + } ] \leq \int _ { 0 } ^ { \infty } \operatorname* { m i n } \big ( 1 , ( K - 1 ) T ^ { 2 } e ^ { - 2 x \Delta _ { \operatorname* { m i n } } } \big ) \mathrm { d } x } } \\ & { } & { \leq \int _ { 0 } ^ { x _ { 0 } } 1 \mathrm { d } x + \int _ { x _ { 0 } } ^ { \infty } ( K - 1 ) T ^ { 2 } e ^ { - 2 x \Delta _ { \operatorname* { m i n } } } \mathrm { d } x } \\ & { } & { = x _ { 0 } + \frac { ( K - 1 ) T ^ { 2 } e ^ { - 2 x _ { 0 } \Delta _ { \operatorname* { m i n } } } } { 2 \Delta _ { \operatorname* { m i n } } } } \\ & { } & { = x _ { 0 } + \frac { 1 } { 2 \Delta _ { \operatorname* { m i n } } } , \qquad \mathrm { ( b y ~ } } \end{array}
$$

which concludes the proof.

## F Proofs of auxiliary lemmas

## F.1 Proof of Lemma 9

Proof. Indeed, the lemma follows directly from the construction of the policy and the relations between $Q _ { t } ^ { \pi }$ $O _ { t } ,$ and $E _ { t }$ . Fix any $t \in B _ { p } ^ { \mathrm { e x } }$ , and let $n \geq 1$ be its local exploitation round. Since the policy remains in the exploitation phase at round t, the threshold condition in (7) ensures that

$$
Q _ { t } ^ { \pi } - O _ { t } = E _ { t } \leq g _ { p } ( n ) = 4 { \sqrt { n \log ( p + 1 ) } } .
$$

Combining this inequality with (14) concludes the first part. The second part follows from $E _ { t } \le E _ { t - 1 } + 1$ .

## F.2 Proof of Lemma 10

Proof. Fix any $t \in B _ { p } ^ { \mathrm { i x } }$ , and let $s \geq 1$ be its local recovery round, so that $t = \tau ( p , s )$ . Since the policy queue remains non-empty throughout the busy period, its recursion is linear between the first recovery round and round s. Therefore,

$$
Q _ { t } ^ { \pi } = Q _ { \tau ( p , 1 ) } ^ { \pi } + \sum _ { s ^ { \prime } = 1 } ^ { s - 1 } ( A _ { \tau ( p , s ^ { \prime } ) } - R _ { \tau ( p , s ^ { \prime } ) , i _ { \tau ( p , s ^ { \prime } ) } } ) ,
$$

where $i _ { s ^ { \prime } }$ are selected by the corresponding Exp3-IX instance. In contrast, the queue length under the constant policy can be empty during this interval. Nevertheless, since $( x ) ^ { + } \geq x$ , we have for any $i \in [ K ]$

$$
Q _ { t } ^ { i } \geq Q _ { \tau ( p , 1 ) } ^ { i } + \sum _ { s ^ { \prime } = 1 } ^ { s - 1 } ( A _ { \tau ( p , s ^ { \prime } ) } - R _ { \tau ( p , s ^ { \prime } ) , i } ) .
$$

Therefore, for any $i \in [ K ]$ and $t \in B _ { p } ^ { \mathrm { i x } }$ satisfying $t = \tau ( p , s )$ , we obtain

$$
\begin{array} { r l } & { Q _ { t } ^ { \pi } - Q _ { t } ^ { i } \leq Q _ { \tau ( p , 1 ) } ^ { \pi } - Q _ { \tau ( p , 1 ) } ^ { i } + \displaystyle \sum _ { s ^ { \prime } = 1 } ^ { s - 1 } ( R _ { \tau ( p , s ^ { \prime } ) , i } - R _ { \tau ( p , s ^ { \prime } ) , i _ { \tau ( p , s ^ { \prime } ) } } ) } \\ & { \qquad \leq Q _ { \tau ( p , 1 ) } ^ { \pi } - O _ { \tau ( p , 1 ) } + \displaystyle \sum _ { s ^ { \prime } = 1 } ^ { s - 1 } ( R _ { \tau ( p , s ^ { \prime } ) , i } - R _ { \tau ( p , s ^ { \prime } ) , i _ { \tau ( p , s ^ { \prime } ) } } ) } \\ & { \qquad \leq E _ { \tau ( p , 1 ) } + \displaystyle \sum _ { s ^ { \prime } = 1 } ^ { s - 1 } ( R _ { \tau ( p , s ^ { \prime } ) , i } - R _ { \tau ( p , s ^ { \prime } ) , i _ { \tau ( p , s ^ { \prime } ) } } ) } \\ & { \qquad \leq E _ { \tau ( p , 1 ) } + \displaystyle \sum _ { s ^ { \prime } = 1 } ^ { s } ( R _ { \tau ( p , s ^ { \prime } ) , i } - R _ { \tau ( p , s ^ { \prime } ) , i _ { \tau ( p , s ^ { \prime } ) } } ) + 1 } \end{array}\tag{by (12)}
$$

which concludes the proof. Note that the additive constant 1 is not essential and is included only to simplify the presentation by shifting the index. □

## F.3 Proof of Lemma 13

Proof. Let $\mathcal { H } _ { 0 }$ denote the complete history until the beginning of the recovery phase, i.e., $\mathcal { F } _ { s }$ for $s =$ $\tau ( p , 1 ) - 1$ . Under i.i.d. stochastic model, the future observations during the recovery phase are independent of $\mathcal { H } _ { 0 }$ and have the same distribution as the original process.

For simplicity, we use the round index from 0, so that $Q _ { 0 } ^ { \pi } = q$ holds at the beginning of the recovery phase. On the event $\{ \tau _ { q } > n \}$ , the queue remains strictly positive throughout the first n recovery rounds, i.e., $Q _ { s } ^ { \pi } > 0$ for all $s \in [ n ]$ , which implies

$$
Q _ { n } ^ { \pi } = q + \sum _ { s = 1 } ^ { n } A _ { s } - R _ { s , i _ { s } } > 0 .
$$

Note that $i _ { s }$ is selected by Exp3-IX policy and

$$
0 < q + \sum _ { s = 1 } ^ { n } A _ { s } - R _ { s , i ^ { * } } + \sum _ { s = 1 } ^ { n } R _ { s , i ^ { * } } - R _ { s , i _ { s } } .\tag{19}
$$

Define

$$
F _ { n } : = \sum _ { s = 1 } ^ { n } ( A _ { s } - R _ { s , i ^ { * } } + \epsilon ) \quad { \mathrm { a n d } } \quad G _ { n } : = \sum _ { s = 1 } ^ { n } ( R _ { s , i ^ { * } } - R _ { s , i _ { s } } ) ,
$$

so that (19) implies $q - \epsilon n + F _ { n } + G _ { n } > 0$

Suppose $\begin{array} { r } { n \geq \frac { 3 q } { \epsilon } } \end{array}$ , which implies that $\epsilon n - q \geq { \frac { 2 \epsilon n } { 3 } }$ . Then, (19) implies that $\begin{array} { r } { F _ { n } + G _ { n } > \frac { 2 \epsilon n } { 3 } } \end{array}$ holds if $n \geq 3 q / \epsilon$ . In other words,

$$
\{ \tau _ { q } > n \} \subseteq \Big \{ F _ { n } \geq \frac { \epsilon n } { 3 } \Big \} \cup \Big \{ G _ { n } \geq \frac { \epsilon n } { 3 } \Big \} .
$$

Case 1. We first control the first event related to $F _ { n }$ . Let $f _ { s } : = A _ { s } - R _ { s , i ^ { * } } + \epsilon _ { \ast }$ , so that $\textstyle F _ { n } = \sum _ { s = 1 } ^ { n } f _ { s }$ . By the definition of ϵ, we have $\mathbb { E } [ f _ { s } ] = 0$ and $f _ { s } \in [ - 1 + \epsilon , 1 + \epsilon ]$ . Therefore, Hoeffding’s inequality gives

$$
\mathrm { P r } \Bigl ( F _ { n } > \frac { \epsilon n } { 3 } \Bigr ) \leq \exp \biggl ( - \frac { \epsilon ^ { 2 } n } { 1 8 } \biggr ) .
$$

Case 2. Here, let us consider the case for $G _ { n }$ . By definition of ${ \mathrm { R e g } } _ { \mathrm { I X } } ( n ; p )$ in (11), it holds that

$$
G _ { n } \leq \operatorname* { m a x } _ { i \in [ K ] } \sum _ { r = 1 } ^ { n } R _ { r , i } - R _ { r , i _ { r } } = \mathrm { R e g } _ { \mathrm { I X } } ( n ; p )
$$

regardless of $p \geq 1$ . Therefore, by choosing $\begin{array} { r } { \delta : = 2 \exp \biggl ( - \frac { \epsilon n } { 6 ( 2 d _ { n } + 1 ) } \biggr ) } \end{array}$ in Lemma 6 for $d _ { n } = \sqrt { K n / \log K }$ we have

$$
\operatorname* { P r } \left( G _ { n } > 4 d _ { n } \log K + ( 2 d _ { n } + 1 ) \log { \frac { 2 } { \delta } } \right) \leq \delta .
$$

Here, by the choice of $d _ { n }$ and $\delta ,$ for $K \geq 2$ and $n \geq 1$ , we have $2 d _ { n } + 1 \leq 3 d _ { n }$ and thus

$$
{ \frac { \epsilon n } { 6 ( 2 d _ { n } + 1 ) } } \geq { \frac { \epsilon n } { 1 8 d _ { n } } } .
$$

Hence, if $n \geq { \frac { 5 7 6 K \log K } { \epsilon ^ { 2 } } }$

$$
\frac { \epsilon n } { 1 8 d _ { n } } = \frac { \epsilon \sqrt { n \log K } } { 1 8 \sqrt { K } } \geq \frac { 4 } { 3 } \log K \implies \delta \leq 2 K ^ { - 4 / 3 } < 1 ,
$$

which implies our choice of δ is valid. Moreover, for $n \geq { \frac { 5 7 6 K \log K } { \epsilon ^ { 2 } } }$ , we have

$$
4 d _ { n } \log K = 4 \sqrt { K n \log K } \leq \frac { \epsilon n } { 6 } .
$$

Since $\begin{array} { r } { \log ( 2 / \delta ) = \frac { \epsilon n } { 6 ( 2 d _ { n } + 1 ) } } \end{array}$ by definition, we obtain

$$
4 d _ { n } \log K + ( 2 d _ { n } + 1 ) \log { \frac { 2 } { \delta } } \leq { \frac { \epsilon n } { 3 } } .
$$

Therefore,

$$
\operatorname* { P r } \Bigl ( G _ { n } > \frac { \epsilon n } { 3 } \Bigr ) \leq \operatorname* { P r } \Bigl ( \operatorname { R e g } _ { \mathrm { I X } } ( n ) > \frac { \epsilon n } { 3 } \Bigr ) \leq 2 \exp \biggl ( - \frac { \epsilon n } { 6 ( 2 d _ { n } + 1 ) } \biggr ) \leq 2 \exp \biggl ( - \frac { \epsilon } { 1 8 } \sqrt { \frac { n \log K } { K } } \biggr ) ,
$$

which concludes the proof for the first part.

Set $\begin{array} { r } { a = \frac { \epsilon ^ { 2 } } { 1 8 } } \end{array}$ and $b = \textstyle { \frac { \epsilon } { 1 8 } } { \sqrt { \frac { \log K } { K } } }$ . Then, for any $n \geq n _ { 0 } ( q )$ , the first part shows that $\operatorname* { P r } ( \tau _ { q } > n ) \ \leq$ $e ^ { - a n } + 2 e ^ { - b { \sqrt { n } } }$ . For any integer $r \geq 1$ , it holds that

$$
\begin{array} { r l } & { \mathbb { E } [ \tau _ { q } ^ { r } ] = \displaystyle \sum _ { n = 0 } ^ { \infty } ( ( n + 1 ) ^ { r } - n ^ { r } ) \operatorname* { P r } ( \tau _ { q } > n ) } \\ & { \quad \quad = \displaystyle \sum _ { n = 0 } ^ { n _ { 0 } ( q ) - 1 } ( ( n + 1 ) ^ { r } - n ^ { r } ) \operatorname* { P r } ( \tau _ { q } > n ) + \displaystyle \sum _ { n = n _ { 0 } ( q ) } ^ { \infty } ( ( n + 1 ) ^ { r } - n ^ { r } ) \operatorname* { P r } ( \tau _ { q } > n ) } \\ & { \quad \quad \le n _ { 0 } ^ { r } ( q ) + \displaystyle \sum _ { n = n _ { 0 } ( q ) } ^ { \infty } ( ( n + 1 ) ^ { r } - n ^ { r } ) \operatorname* { P r } ( \tau _ { q } > n ) } \\ & { \quad \quad \le n _ { 0 } ^ { r } ( q ) + \displaystyle \sum _ { n = n _ { 0 } ( q ) } ^ { \infty } r ( n + 1 ) ^ { r - 1 } ( e ^ { - a n } + 2 e ^ { - b \sqrt { n } } ) \qquad \quad \scriptstyle { \mathrm { ( b y ~ m e a n . v a l u e ~ t h e ~ o r e m ) } } } \end{array}
$$

Since $n _ { 0 } ( q ) \geq 1$

$$
\begin{array} { l } { \displaystyle \sum _ { n = n _ { 0 } ( q ) } ^ { \infty } r ( n + 1 ) ^ { r - 1 } ( e ^ { - a n } + 2 e ^ { - b \sqrt { n } } ) \leq \displaystyle \sum _ { n = 1 } ^ { \infty } r ( n + 1 ) ^ { r - 1 } ( e ^ { - a n } + 2 e ^ { - b \sqrt { n } } ) } \\ { = \displaystyle \sum _ { n = 2 } ^ { \infty } r n ^ { r - 1 } ( e ^ { - a ( n - 1 ) } + 2 e ^ { - b \sqrt { n - 1 } } ) } \\ { \leq \displaystyle \sum _ { n = 2 } ^ { \infty } r n ^ { r - 1 } ( e ^ { a } e ^ { - a n } + 2 e ^ { b } e ^ { - b \sqrt { n } } ) } \end{array}
$$

By definition of Gamma function Γ, we have

$$
\begin{array} { l } { \displaystyle \sum _ { n = 2 } ^ { \infty } n ^ { r - 1 } e ^ { - a n } \leq \int _ { 1 } ^ { \infty } z ^ { r - 1 } e ^ { - a z } \mathrm { d } z } \\ { \displaystyle \qquad \leq \frac { \Gamma ( r ) } { a ^ { r } } , } \end{array}
$$

and

$$
\begin{array} { l } { \displaystyle \sum _ { n = 2 } ^ { \infty } n ^ { r - 1 } e ^ { - b \sqrt { n } } \leq \int _ { 1 } ^ { \infty } z ^ { r - 1 } e ^ { - b \sqrt { z } } \mathrm { d } z } \\ { \displaystyle \qquad \leq \frac { 2 \Gamma ( 2 r ) } { b ^ { 2 r } } , } \end{array}
$$

which concludes the proof.

## F.4 Proof of Lemma 14

We first introduce a variant of Freedman’s inequality.

Lemma 15 (Corollary 1(b) of Howard et al. (2020)). Suppose $( Z _ { t } ) _ { t \geq 0 }$ is a real-valued martingale adapted tofiltration $( \mathcal { F } _ { t } ) _ { t }$ with $Z _ { 0 } = 0$ satisfying $Z _ { t } - Z _ { t - 1 } \leq \mathfrak { c }$ c almost surelyfor all $t \geq 1$ . Define the predictable quadratic variation

$$
V _ { t } : = \sum _ { s = 1 } ^ { t } \mathbb { E } \big [ ( Z _ { s } - Z _ { s - 1 } ) ^ { 2 } | \mathcal { F } _ { s - 1 } \big ] .
$$

Then, for every $x , k \geq 0 .$

$$
\operatorname* { P r } \Big ( \exists t \geq 1 : Z _ { t } \geq x + \mathfrak { s } _ { p } \Big ( \frac x k \Big ) ( V _ { t } - k ) \Big ) \leq \exp \bigg ( - \frac { x ^ { 2 } } { 2 ( k + c x / 3 ) } \bigg ) ,\tag{20}
$$

where $\textstyle { \mathfrak { s } } _ { P } ( u ) = { \frac { c u - \log ( 1 + c u ) } { c \log ( 1 + c u ) } }$

Proof. Let $Y _ { m , i } = \hat { \ell } _ { m , i } ^ { \varnothing } - \hat { \ell } _ { m , i } ^ { \varnothing }$ <sub>∗</sub> and define a martingale difference sequence

$$
\begin{array} { r } { X _ { m , i } : = \hat { \ell } _ { m , i } ^ { \varnothing } - \hat { \ell } _ { m , i ^ { * } } ^ { \varnothing } - \mathbb { E } \Big [ \hat { \ell } _ { m , i } ^ { \varnothing } - \hat { \ell } _ { m , i ^ { * } } ^ { \varnothing } \Big | \mathcal { F } _ { m - 1 } \Big ] = Y _ { m , i } - \mathbb { E } [ Y _ { m , i } | \mathcal { F } _ { m - 1 } ] , } \end{array}\tag{21}
$$

for the global history $\mathcal { F } _ { m - 1 }$ until m-th exploration rounds. Since $\hat { \ell } _ { m , i } ^ { \varnothing }$ is IW loss estimator with the uniform exploration, we have

$$
\begin{array} { r } { \mathbb { E } [ Y _ { m , i } | \mathcal { F } _ { m - 1 } ] = \Delta _ { i } , \quad | Y _ { m , i } | \le K , \quad \mathrm { a n d } \quad \mathbb { E } [ Y _ { m , i } ^ { 2 } | \mathcal { F } _ { m - 1 } ] = K \mathbb { E } [ \ell _ { m , i } ^ { 2 } + \ell _ { m , i ^ { * } } ^ { 2 } ] \le K \mathbb { E } [ \ell _ { m , i } + \ell _ { m , i ^ { * } } ] \le K ( 2 - \Delta _ { i } ) , } \end{array}
$$

which implies

$$
| X _ { m , i } | \leq K , \quad \mathrm { a n d } \quad \mathbb { E } \big [ X _ { m , i } ^ { 2 } \big | \mathcal { F } _ { m - 1 } \big ] \leq K ( 2 - \Delta _ { i } ) , \forall m \geq 1 , i \neq i ^ { * } .\tag{22}
$$

The last inequality follows since the variance is at most the second moment, i.e., $\mathbb { E } [ X _ { m , i } ^ { 2 } | \cdot ] \leq \mathbb { E } [ Y _ { m , i } ^ { 2 } | \cdot ]$

By definition, it holds that

$$
\begin{array} { r } { \left\{ \underset { i \in [ K ] } { \arg \operatorname* { m i n } } \hat { L } _ { m + 1 , i } ^ { \varnothing } \neq i ^ { * } \right\} \subseteq \Big \{ \exists i \neq i ^ { * } : \hat { L } _ { m + 1 , i } ^ { \varnothing } - \hat { L } _ { m + 1 , i ^ { * } } ^ { \varnothing } \leq 0 \Big \} . } \end{array}
$$

Here,

$$
\begin{array} { c } { \displaystyle \hat { L } _ { m + 1 , i } ^ { \emptyset } - \hat { L } _ { m + 1 , i ^ { * } } ^ { \emptyset } = \displaystyle \sum _ { b = 1 } ^ { m } Y _ { b , i } = \displaystyle \sum _ { b = 1 } ^ { m } \mathbb { E } [ Y _ { b , i } | \mathcal { F } _ { b - 1 } ] + \displaystyle \sum _ { b = 1 } ^ { m } X _ { b , i } } \\ { \displaystyle = m \Delta _ { i } + \displaystyle \sum _ { b = 1 } ^ { m } X _ { b , i } , } \end{array}
$$

which implies

$$
\Big \{ \exists i \neq i ^ { * } : \hat { L } _ { m + 1 , i } ^ { \varnothing } - \hat { L } _ { m + 1 , i ^ { * } } ^ { \varnothing } \leq 0 \Big \} = \Bigg \{ \sum _ { b = 1 } ^ { m } X _ { b , i } \leq - m \Delta _ { i } \Bigg \} .
$$

Recall that $X _ { m , i }$ is a martingale difference sequence satisfying (22). Then, by applying the variation of Freedman’s inequality in Lemma $1 5 ^ { 1 }$ , we obtain for $m _ { 0 } \geq 1$ that

$$
\operatorname* { P r } \biggl ( \exists m \geq m _ { 0 } : \sum _ { b = 1 } ^ { m } X _ { b , i } \leq - m \Delta _ { i } \biggr ) \leq \exp \biggl ( - \frac { m _ { 0 } ^ { 2 } \Delta _ { i } ^ { 2 } } { 2 ( K m _ { 0 } ( 2 - \Delta _ { i } ) + K m _ { 0 } \Delta _ { i } / 3 ) } \biggr ) \leq \exp \biggl ( - \frac { m _ { 0 } \Delta _ { i } ^ { 2 } } { 4 K } \biggr ) .
$$

Therefore, taking a union bound over the suboptimal arms conclude the proof.

## G Details on experiments

As briefly discussed in Sections 2 and 3, the definitions of empty and busy periods and the queue dynamics differ across the policies considered in our experiments. Let $Q _ { t } ^ { j }$ be the queue length under the arrival-first convention in (1) and $\hat { Q } _ { t } ^ { j }$ be that under the service-first convention in (8) for the same selection sequence j.

We first show by induction that

$$
0 \leq \bar { Q } _ { t } ^ { j } - Q _ { t } ^ { j } \leq 1\tag{23}
$$

for every t. The claim holds for any empty period and at $t = \beta _ { p }$ since both queues are initialized at $t = 1$ Suppose that (23) holds at round $t \in B _ { p }$ for some $p \in \mathbb N$ and let $R _ { t }$ denote the service obtained by the common arm. Then, we have

$$
Q _ { t + 1 } ^ { j } = ( Q _ { t } ^ { j } + A _ { t } - R _ { t } ) ^ { + } \leq ( Q _ { t } ^ { j } - R _ { t } ) ^ { + } + A _ { t } \leq ( \bar { Q } _ { t } ^ { j } - R _ { t } ) ^ { + } + A _ { t } = \bar { Q } _ { t + 1 } ^ { j } .
$$

Using the induction hypothesis and $R _ { t } \leq 1$ , we also obtain

$$
\begin{array} { r } { \bar { Q } _ { t + 1 } ^ { j } \leq ( Q _ { t } ^ { j } + 1 - R _ { t } ) ^ { + } + A _ { t } \leq ( Q _ { t } ^ { j } + A _ { t } - R _ { t } ) ^ { + } + 1 = Q _ { t + 1 } ^ { j } + 1 , } \end{array}
$$

which proves (23). Therefore, the worst-case queue length regrets obtained by the same selection sequence under the two recursions differ by at most 1.

The above comparison holds when the action sequence is fixed across the two queue dynamics. Since queue-dependent policies may select different actions under different recursions, it does not guarantee that their worst-case regrets differ by at most 1 when they are run separately. Nevertheless, the direct discrepancy between the two queue recursions is bounded at each round, and worst-case regret takes a maximum rather than accumulating differences over time. We therefore expect the choice of queue dynamics to have a limited effect on worst-case regret. In contrast, cumulative regret aggregates roundwise differences and can be substantially more sensitive to this modeling choice.

To follow the original formulations, Section 5 reports results obtained by implementing each policy with the queue dynamics and period definitions used in the corresponding paper. For our policy, empty and busy periods are defined by $Q _ { t } ^ { \pi } + A _ { t } = 0$ and $Q _ { t } ^ { \pi } + A _ { t } > 0$ , respectively. For the baseline policies, we retain the

$$
\operatorname* { P r } ( \exists t \geq 1 : Z _ { t } \geq x , V _ { t } \leq m ) \leq \operatorname* { P r } \Big ( \exists t \geq 1 : Z _ { t } \geq x + \mathfrak { s } _ { p } \Big ( \frac { x } { m } \Big ) \big ( V _ { t } - m \big ) \Big ) .
$$

convention that empty and busy rounds satisfy $Q _ { t } ^ { \pi } = 0$ and $Q _ { t } ^ { \pi } > 0 ,$ , respectively. In particular, our policy, Q-ThS, and KS-OGD use the arrivals-first dynamics in (1), whereas SSM-3P uses the service-first recursion in (8). This choice reproduces each policy under its original model, although cumulative-regret comparisons involving SSM-3P may partly reflect the difference in queue dynamics.

To address this difference, Figures 4 and 5 provide additional results in which all policies are run under the arrivals-first dynamics in (1) and under the service-first in (8), respectively. These figures show that the overall qualitative trends remain largely unchanged.

All baseline policies and parameter settings remain unchanged, while our policy is modified only as described below under the service-first convention. As discussed in Remark 1, under the service-first convention, our policy does not need to observe $A _ { t }$ at the beginning of each round. Accordingly, we define an empty period by $Q _ { t } ^ { \pi } = 0$ , matching the definition used by the baseline policies.

We implement Q-ThS, SSM-3P, and KS-OGD by translating the MATLAB code released by Krishnakumar and Sinha (2025) into Python while preserving its algorithmic update rules and parameter settings. The code and instructions required to reproduce all experiments will be made publicly available on GitHub.

![](images/d8084674a8da761d4c6e7a361795ea0379fd61e49151fce5140fa11d1f221aa4.jpg)  
(a) Worst-case regret in adversarial regime.

![](images/326f034327f4458baad27f642d4bd45ab8631ab31d1bea5f6e07b4c49eb46976.jpg)  
(b) Worst-case regret in stochastic regime.

![](images/b87d3fe578c7f65245d77a8abb1c7f304807cd509a2bda361cb7843117abc575.jpg)  
(c) Cumulative regret in adversarial regime.

![](images/9e3c6d7808978e36cc812e1e7d8ca8de3cba40fad67bb57978e44712d4f8c7dc.jpg)  
(d) Cumulative regret in stochastic regime.

![](images/6f7c2ec6373d775b39087cf9bdb9ccee0f7e0de0113b7d01f2258ffb48f3a9a8.jpg)  
(e) Queue lengths $Q _ { t } ^ { \pi }$ in adversarial regime.

![](images/1efceb870931d5bd773699496989c41129cd11df0298e77f484ebf917806d955.jpg)  
(f) Queue lengths $Q _ { t } ^ { \pi }$ in stochastic regime.  
Figure 4: Results under arrival-first dynamics in (1)

![](images/894ca920d10f0d33343e9cfca3606d91a25b2b174741579c337ab828ad8a4d62.jpg)  
(a) Worst-case regret in adversarial regime.

![](images/4525e1d92f18dfb195221e00f8ec238ca6e927ef95a095cb873083545c4e2904.jpg)  
(b) Worst-case regret in stochastic regime.

![](images/4f0d49ef5f001ebb9d4ba9b14bcc1c52ba3d43a698354b958f81cc083199eb29.jpg)  
(c) Cumulative regret in adversarial regime.

![](images/aaabb5bd21dd4648b8e8c7971cc016210278dbe5e0efcb507f4894fa604df555.jpg)  
(d) Cumulative regret in stochastic regime.

![](images/eefe8c95628be876f6911675762ddc080fda9072ff2889c0a8410ab019290bb9.jpg)  
(e) Queue lengths $Q _ { t } ^ { \pi }$ in adversarial regime.

![](images/56157d90099a64dcef5c14b9e4115c6c801c616e5fba2ae29cf1a3520eb60028.jpg)  
(f) Queue lengths $Q _ { t } ^ { \pi }$ in stochastic regime.  
Figure 5: Results under service-first dynamics in (8)