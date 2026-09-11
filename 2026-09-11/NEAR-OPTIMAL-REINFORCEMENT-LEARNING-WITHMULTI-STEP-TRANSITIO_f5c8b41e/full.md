# NEAR-OPTIMAL REINFORCEMENT LEARNING WITHMULTI-STEP TRANSITION LOOKAHEAD

Corentin Pla   
CREST, ENSAE   
Criteo AI Lab   
FairPlay Joint Team   
c.pla@criteo.com   
Hugo Richard   
Criteo AI Lab   
FairPlay Joint Team   
h.richard@criteo.com   
Marc Abeille   
Criteo AI Lab   
FairPlay Joint Team   
m.abeille@criteo.com   
Vianney Perchet   
CREST, ENSAE   
Criteo AI Lab   
FairPlay Joint Team   
v.perchet@criteo.com

## ABSTRACT

We study reinforcement learning (RL) with transition look-ahead, where the agent may observe which states would be visited upon playing any sequence of ℓ actions before deciding its course of action. Although look-ahead can substantially improve achievable performance, it is known that optimal planning with multi-step transition look-ahead is NP-hard, but this hardness was established using discount factors arbitrarily close to one. It was therefore unknown whether the problem remains hard for any discount factor, and whether near-optimal planning can nevertheless be performed efficiently. We resolve both questions. First, we show that for every fixed rational discount factor (γ ∈ (0, 1)), exact planning remains NPhard. Second, we introduce a randomized polynomial-time approximation scheme for every fixed look-ahead depth. We then extend our approach to unknown transitions and stochastic rewards using optimism and variance-adaptive confidence bounds. The resulting algorithm achieves cumulative regret whose leading term matches classical tabular discounted RL up to logarithmic factors. Thus, although exact planning with transition look-ahead is NP-hard, efficient near-optimal planning and learning remain possible.

## 1 INTRODUCTION

Reinforcement Learning (RL) (Sutton & Barto, 2018) addresses the problem of learning how to act in a dynamic environment. This problem is modeled via a Markov Decision Process (MDP) which involves a transition kernel, describing how states of the environment evolve in response to the agent’s actions, and a reward function, providing feedback to the agent for taking a particular action in a given state. The agent’s goal is to select actions that maximize the cumulative collected reward called return, accounting not only for immediate gains but also for the long-term impact of its decisions on the state dynamics (Jaksch et al., 2010; Azar et al., 2017; Jin et al., 2018). In this work, we focus on stationary MDP, in which the reward function and transition kernel are independent of time.

In the standard RL framework, the reward and the next state are revealed only after an action has been taken. However, RL with transition look-ahead assumes that, in addition to this underlying dynamical model, extra predictive information is available at decision time: the agent observe before taking its action, which states would be visited upon playing any sequence of actions of length ℓ. This captures situations where one benefits from privileged information channels beyond standard interaction. A typical example is collaborative navigation systems that allow real-time traffic information (e.g., Waze, Coyote...) where information from nearby drivers can be used to estimate future position, speed, and traffic conditions given a sequence of routing decisions (Vasserman et al., 2015).

Other examples include access to high-fidelity but expensive simulators that can provide look-ahead on demand, or supply-chain systems where estimated delivery or arrival times are provided in advance. Standard RL algorithms do not come with off-the-shelf tools to incorporate look-ahead, and a naive policy would be to just discard this additional information which is sub-optimal.

Related work. The idea of augmenting reinforcement learning with look-ahead information has recently gained attention. Merlis (2024) introduced a pseudo-polynomial algorithm for one-step transition look-ahead in the finite-horizon setting, while Merlis et al. (2024) studied general look ahead horizons for reward look-ahead, focusing on the value of additional information. Closer to our setting, Pla et al. (2026a) studied the computational complexity of perfect transition look-ahead in stationary MDPs, showing that planning is polynomial-time solvable for $\ell = 1$ but NP-hard for every $\ell \geq 2 .$ . Lu et al. (2025) considered discounted MDPs with imperfect and partially available transition look-ahead and proposed BOLA, an algorithm that uses a ℓ-step transition prediction to select a sequence of ℓ actions before replanning. This procedure does not in general implement the optimal adaptive policy in our transition-look-ahead model, where the agent observes an updated look-ahead window and makes a new decision at every time step.

Our hardness result also connects to the broader literature on the computational complexity of MDP planning. Classical planning in stationary MDPs is polynomial-time solvable under the discounted criterion, with foundational complexity results due to Papadimitriou (1987); subsequent work has further investigated the complexity of MDP planning under different objectives and horizons (Mundhenk et al., 2000; Littman et al., 2013; Balaji et al., 2018; Chen & Wang, 2017). Beyond these classical formulations, changes in the information structure can fundamentally alter computationa complexity. For instance, Walsh et al. (2009) show that planning with delayed feedback can become NP-hard because of the exponential blow-up of the augmented state space, while partial observability leads to even stronger hardness results (Papadimitriou, 1987). Transition look-ahead exhibits a complementary phenomenon: providing additional information also induces a large augmented state space and makes exact planning intractable. Our positive result therefore belongs to a complementary algorithmic question: whether such exponentially large planning problems can nevertheless be approximated without explicitly solving the augmented MDP.

In this respect, our approach is related to sampling-based and empirical methods for approximate planning. Kearns et al. (2002) introduced sparse sampling, which computes near-optimal actions by exploring only a randomly sampled portion of the full look-ahead tree. This method has a complexity that is exponential in the effective horizon $( 1 - \gamma ) ^ { - 1 }$ , whereas our algorithm has polynomial complexity in $( \bar { 1 } - \gamma ) ^ { - 1 }$ for fixed look-ahead depth ℓ. Our method can be viewed as a dynamicprogramming analogue of sample-average approximation (Kleywegt et al., 2002): a fixed sample of complete transition tables replaces the Bellman expectations and induces a closed empirical planning problem. Unlike classical SAA, this problem is solved once during preprocessing, and its solution extends with a uniform guarantee to any look-ahead window observed at deployment.

The look-ahead information exploited by our policy can also be viewed as prediction available to the decision maker before acting, this connects our setting to the growing literature on algorithms with predictions (Mitzenmacher & Vassilvitskii, 2020; Benomar et al., 2025; Benomar & Perchet, 2025; Merlis et al., 2023). These works study how side information can be used to improve upon worst-case performance, often through trade-offs between consistency (when predictions are accurate) and robustness (when they are not). This perspective has recently been explored in several sequential decision-making settings. Li et al. (2024) design a learning-augmented controller for LQR with latent perturbations, where accurate predictions lead to near-optimal performance while robustness to prediction errors is retained. Lyu et al. (2026) study discounted MDPs equipped with predictions of the transition matrix and show that such predictions can reduce sample complexity, while Li et al. (2023) establish consistency–robustness trade-offs when the advice is provided in the form of predicted Q-values in non-stationary MDPs. Another line of work studies MDPs with exogenous information or dynamics. Pla et al. (2026b) consider discounted MDPs with i.i.d. exogenous contexts that are revealed before the agent acts and derive minimax PAC guarantees that exploit this structure, while Maran et al. (2026) study MDPs markovian exogenous contexts and show that this structure can substantially improve learning guarantees.

Contribution. We resolve two open questions concerning planning with multi-step transition look-ahead. First, we show that the known NP-hardness does not rely on a large effective horizon: for every fixed rational discount factor $\gamma \in ( 0 , 1 )$ , exact planning remains NP-hard for $\ell \geq 2$ (Theorem 1). Second, for every fixed look-ahead depth ℓ, we give a randomized polynomial-time approximation scheme that constructs, with high probability, a uniformly near-optimal policy (Theorem 2).We further extend our approach to unknown transitions and stochastic rewards with a regret learning algorithm (Theorem 3).

## 2 SETTING AND OBJECTIVES

## 2.1 MARKOV DECISION PROCESSES

We study finite tabular Markov decision processes (MDP) $\boldsymbol { \mathcal { M } } = ( \boldsymbol { \mathcal { S } } , \boldsymbol { \mathcal { A } } , \boldsymbol { P } , \boldsymbol { r } )$ , where $s$ is a finite state space $| S | = n ,$ , A is a finite action space $| { \mathcal { A } } | = m , P ( s ^ { \prime } \mid { \dot { s } } , a )$ denotes the probability of reaching state $s ^ { \prime } \in \mathcal { S }$ after taking action $a \in { \mathcal { A } }$ in state $s \in S .$ , and $r : \mathcal { S } \times \mathcal { A } \stackrel {  } { \to } [ 0 , R _ { \mathrm { m a x } } ]$ is the reward function. A (possibly randomized) stationary memoryless policy is a mapping $\pi : { \mathcal { S } } $ $\Delta ( \mathcal { A } )$ , where $\Delta ( \mathcal { A } )$ denotes the simplex over ${ \mathcal { A } } .$ Under such a policy, the interaction evolves as follows: at each time $t \in \mathbb { N }$ , the system is in state $s _ { t } \in S$ , the agent selects an action $a _ { t } \sim \pi ( \cdot \mid s _ { t } )$ receives reward $r ( s _ { t } , a _ { t } )$ , and the next state is sampled according to $\cdot s _ { t + 1 } \sim P ( \cdot \mid s _ { t } , a _ { t } )$

We consider the standard discounted-return objective (Puterman, 2014, Chapter $^ { 6 ) }$ . For a dis count factor $\gamma ~ \in ~ ( 0 , 1 )$ and an initial state $s \in \ S$ , the value of a policy π is $\begin{array} { r l } { V ^ { \pi } ( s ) } & { { } = } \end{array}$ $\mathbb { E } ^ { \pi } [ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r ( s _ { t } , a _ { t } ) { \mathrm { ~ } } | { \mathrm { ~ } } s _ { 0 } = s ]$ . The optimal discounted value function is defined by $V ^ { * } { \dot { ( } } s { \dot { ) } } ~ = { }$ sup<sub>π</sub> $V ^ { \pi } ( s ) , \forall s \in \mathcal { S }$ , where the supremum is taken over stationary memoryless policies. The optimal value function $V ^ { * }$ is the unique solution to the Bellman optimality equations: $V ^ { \ast } ( s ) =$ $\begin{array} { r l } & { \operatorname* { m a x } _ { a \in \mathcal { A } } \Bigl \{ r ( s , a ) + \gamma \sum _ { s ^ { \prime } \in \mathcal { S } } P ( s ^ { \prime } \mid s , a ) V ^ { * } ( s ^ { \prime } ) \Bigr \} , \forall s \in \mathcal { S } . } \end{array}$

## 2.2 TRANSITION LOOK-AHEAD

We now formalize the extra information provided by the look-ahead in terms of state observability and provide an augmented MDP construction that allows us to embed this problem into the standard evaluation framework introduced above.

## 2.2.1 LOOK-AHEAD AND STATE OBSERVABILITY

A convenient generative view of the transition kernel is the following. At each time $t ,$ for every state-action pair $( s , a )$ , the environment independently draws a potential successor $\Theta _ { t } ( s , a ) \sim P ( \cdot \vert$ $s , a )$ , collecting these draws defines a random transition table $\mathsf { \bar { \Theta } } _ { t } : \mathcal { S } \times \mathcal { A } \to \mathcal { S }$ , taking values in $\dot { \Omega } : = \mathcal { S } ^ { \mathcal { S } \times \mathcal { A } }$ , Its distribution $\mathsf { Q }$ is therefore

$$
{ \mathsf { Q } } ( \theta ) : = \prod _ { ( s , a ) \in S \times A } P \big ( \theta ( s , a ) \mid s , a \big )\tag{1}
$$

We let $( \Theta _ { t } ) _ { t \geq 0 }$ be an i.i.d. sequence of distribution $\mathsf { Q } .$ . In a standard MDP, $\Theta _ { t }$ is hidden from the agent and, after action $A _ { i }$ <sub>t</sub> is selected, the realized transition is simply $S _ { t + 1 } = \Theta _ { t } ( S _ { t } , A _ { t } )$ . Transition look-ahead changes the information available before acting. At time $t ,$ the environment reveals some of the tables $\Theta _ { t } , \Theta _ { t + 1 } , . . .$ . to the agent.

Definition 1 (ℓ-step transition look-ahead). Let $\ell \geq 1$ . At each decision time t, an agent with ℓ-step transition look-ahead observes, before choosing $A _ { t } ,$ the window

$$
C _ { t } ^ { \ell } : = ( \Theta _ { t } , \Theta _ { t + 1 } , \dots , \Theta _ { t + \ell - 1 } ) \in \Omega ^ { \ell } .\tag{2}
$$

This window encodes the complete depth-ℓ transition tree rooted at the current state. Indeed, starting from $s _ { 0 } = S _ { t }$ , any action sequence $a _ { 0 } , \ldots , a _ { k - 1 }$ , with $k \leq \ell ,$ , determines the trajectory

$$
s _ { j + 1 } = \Theta _ { t + j } ( s _ { j } , a _ { j } ) , \qquad j = 0 , \ldots , k - 1 .
$$

Thus, $C _ { t } ^ { \ell }$ reveals the outcome of every action sequence of length at most ℓ before the first action is chosen. In particular, for $\ell = 1$ , the agent knows the successor $\Theta _ { t } ( S _ { t } , a )$ of every action $a \in { \mathcal { A } }$ After one transition, the oldest table is discarded and a fresh independent table is revealed, so that $C _ { t + 1 } ^ { \ell } = ( \Theta _ { t + 1 } , \ldots , \Theta _ { t + \ell - 1 } , \Theta _ { t + \ell } )$ , with $\Theta _ { t + \ell } \sim \ Q$ independent of the previous tables. Hence, the pair $( S _ { t } , C _ { t } ^ { \ell } )$ evolves as a Markov process on $\boldsymbol { \mathcal { S } \times \Omega ^ { \ell } }$ , allowing the transition look-ahead problem to be treated as a standard discounted MDP on this augmented state space.

## 2.2.2 POLICIES AND VALUE FUNCTIONS

A stationary policy with ℓ-step transition look-ahead acts on the augmented state and is therefore a mapping $\pi : \mathcal { S } \times \mathrm { \bar { \Omega } } ^ { \ell } \to \Delta ( \bar { \mathcal { A } } )$ . Its value at an observed state-window pair $( s , c ) \in \mathcal { S } \times \Omega ^ { \ell }$ , with $\begin{array} { r } { c = ( \theta _ { 0 } , \dots , \theta _ { \ell - 1 } ) , \mathrm { i s } V _ { \ell } ^ { \pi } ( s , c ) : = \mathbb { E } ^ { \pi } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r ( S _ { t } , A _ { t } ) \Big | S _ { 0 } = s , C _ { 0 } ^ { \ell } = c \right] } \end{array}$

We denote the optimal value by $\begin{array} { r c l } { V _ { \ell } ^ { \star } ( s , c ) } & { : = } & { \operatorname* { s u p } _ { \pi } V _ { \ell } ^ { \pi } ( s , c ) } \end{array}$ . For any bounded function $V : \mathcal { S } \times \Omega ^ { \ell } \to \mathbb { R }$ , define the Bellman optimality operator $( \mathcal { T } _ { \ell } V ) ( s , \theta _ { 0 } , \ldots , \theta _ { \ell - 1 } ) : =$ ma $\mathfrak { c } _ { a \in \mathcal { A } } \left\{ r ( s , a ) + \gamma \mathbb { E } _ { \Theta \sim \mathbb { Q } } \left[ V \big ( \theta _ { 0 } ( s , a ) , \theta _ { 1 } , \ldots , \theta _ { \ell - 1 } , \Theta \big ) \right] \right\}$ . The operator $\mathcal { T } _ { \ell }$ is a γ-contraction, so $V _ { \ell } ^ { \star }$ is its unique fixed point. The corresponding optimal action-value function is $Q _ { \ell } ^ { \star } ( s , \theta _ { 0 } , \ldots , \theta _ { \ell - 1 } , a ) : = r ( s , a ) + \gamma \mathbb { E } _ { \Theta \sim \mathbb { Q } } \left[ V _ { \ell } ^ { \star } \big ( \theta _ { 0 } ( s , a ) , \theta _ { 1 } , \ldots , \theta _ { \ell - 1 } , \Theta \big ) \right]$ . An optimal policy is then obtained by choosing $\pi _ { \ell } ^ { \star } ( s , \theta _ { 0 } , \ldots , \theta _ { \ell - 1 } ) \in \arg \operatorname* { m a x } _ { a \in \mathcal { A } } Q _ { \ell } ^ { \star } ( s , \theta _ { 0 } , \ldots , \theta _ { \ell - 1 } , a )$

## 3 HARDNESS OF EXACT PLANNING

Our first result strengthens the known hardness of multi-step transition look-ahead planning of Pla et al. (2026a) by showing that it persists for every fixed discount factor. Analogously to the statevalue criterion in a standard MDP, we average over the look-ahead window revealed at the initial decision time and define $v _ { \ell } ^ { \pi } ( s ) : = \mathbb { E } _ { C \sim \mathsf { Q } ^ { \otimes \ell } } \left[ \bar { V } _ { \ell } ^ { \pi } ( s , C ) \right]$ , and $v _ { \ell } ^ { \star } ( s ) : = \mathbb { E } _ { C \sim \mathsf { Q } ^ { \otimes \ell } } \left[ V _ { \ell } ^ { \star } ( s , C ) \right]$

Theorem 1 (Fixed-discount hardness of transition look-ahead planning). Fix an integer $\ell \geq 2$ and a rational discountfactor $\gamma \in ( 0 , 1 )$ . Given afinite MDP M, an initial state $s _ { 0 } \in S ,$ , and a rational threshold θ, deciding whether there exists a policy with perfect ℓ-step transition look-ahead such that $v _ { \ell } ^ { \pi } ( s _ { 0 } ) \geq \theta \mathrm { ~ } i s \mathrm { ~ } \tilde { N P } \mathrm { - } h a r d .$

Proofsketch. We adapt the hardness reduction of Pla et al. (2026a), which relies on the expectedmaximum gap construction of Mehta et al. (2020). The construction provides independent random variables $X _ { 1 } , \ldots , X _ { n }$ and an integer k for which it is hard to distinguish between the following two cases: either every subset $Y \subseteq [ { \overline { { n } } } ]$ of size k satisfies $\mathbb { E } \left[ \operatorname* { m a x } _ { i \in Y } \breve { X } _ { i } \right] \leq U$ , or there exists a subset $Y ^ { \star }$ of size k such that E $\begin{array} { r } { [ \operatorname* { m a x } _ { i \in Y ^ { \star } } X _ { i } ] \geq U + \Delta } \end{array}$ , for some threshold $U$ and gap $\Delta > 0$

The MDP reduction turns this construction into a planning problem. At the root state, the look-ahead reveals a random candidate subset $Y$ of size at most $k .$ . The agent can either commit to this subset, in which case its value is $\gamma ^ { \ell + 1 } \mathbb { E } \left[ \operatorname* { m a x } _ { i \in Y } X _ { i } \right]$ , or wait one step and observe a fresh independent candidate subset. Therefore, if every subset has expected maximum at most $U$ , every commitment is worth at most $T : = \gamma ^ { \ell + 1 } \dot { U }$ . The difficulty is that a valuable subset may require many waiting steps before it appears, and its eventual gain is then strongly discounted unless $\gamma$ is sufficiently close to one. Our modification removes this issue by assigning the waiting action the reward $( 1 - \dot { \gamma } ) T$ . This choice makes T a fixed point of waiting, since $( 1 - \gamma ) \bar { T } + \gamma T = T$ . Hence, when all candidate subsets have expected maximum at most $U _ { : }$ , committing gives at most $T .$ , while waiting also preserves an upper bound of T. The optimal value at the root is therefore at most $T$

On the other hand, if a subset $Y ^ { \star }$ satisfies $\mathbb { E } \left[ \operatorname* { m a x } _ { i \in Y ^ { \star } } X _ { i } \right] \geq U + \Delta$ , then committing when $Y ^ { \star }$ appears gives strictly more than $T .$ The agent can repeatedly wait until this favorable subset is revealed and then commit. The waiting rewards preserve the baseline $T$ , while only the positive surplus above $T$ is discounted until the subset appears. Since the favorable subset occurs with positive probability at each fresh draw, this surplus remains strictly positive for every fixed $\gamma \in$ $( 0 , 1 )$ Thus the two cases of the expected-maximum gap lead respectively to an optimal value at most $T$ and strictly larger than $T .$ This yields the desired hardness result for every fixed $\gamma \in ( 0 , 1 )$ removing the requirement in Pla et al. (2026a) that $\gamma$ be chosen sufficiently close to one. The full reduction and encoding argument are given in Appendix A.1. □

## 4 NEAR OPTIMAL PLANNING

We now turn to our positive result: for every fixed look-ahead depth, near-optimal planning admits a randomized polynomial-time approximation scheme.

## 4.1 ALGORITHM

Our algorithm proceeds in two phases. Offline, it samples a finite collection of transition tables and uses them to build a finite state space, value iteration is then run once on this sampled state space. Online, the agent combines the precomputed values with the actual look-ahead window it observes in order to select an action.

```latex
Algorithm 1 Offline Value Iteration
Input: MDP $\boldsymbol { \mathcal { M } } = ( \boldsymbol { \mathcal { S } } , \boldsymbol { \mathcal { A } } , \boldsymbol { P } , \boldsymbol { r } )$ , look-ahead depth ℓ, dictionary size $N _ { \ast }$ , iterations K
Output: Policy representation D
1: Sample $\Theta ^ { 1 } , \dots , \Theta ^ { N } \overset { \mathrm { i . i . d . } } { \sim } \mathbf { Q }$
2: Set $\dot { \mathcal { C } } _ { N } \gets \dot { \mathcal { S } } \times [ N ] ^ { \ell }$
3: Initialize $v ^ { 0 } ( s , \dot { i _ { 0 } } , \dot { . } . . , i _ { \ell - 1 } ) \gets 0$ for all $( s , i _ { 0 } , \dotsc , i _ { \ell - 1 } ) \in \mathcal { C } _ { N }$
4: for $k = 0 , \ldots , K - 1$ do
for all $( s , i _ { 0 } , \dotsc , i _ { \ell - 1 } ) \in \mathcal { C } _ { N }$ do
6: $v ^ { k + 1 } ( s , i _ { 0 } , \ldots , i _ { \ell - 1 } ) \gets \operatorname* { m a x } _ { a \in \mathcal { A } } \{ r ( s , a ) + \frac { \gamma } { N } \sum _ { j = 1 } ^ { N } v ^ { k } \big ( \Theta ^ { i _ { 0 } } ( s , a ) , i _ { 1 } , \ldots , i _ { \ell - 1 } , j \big ) \}$ (3)
7: end for
8: end for
9: $D \gets ( \Theta ^ { 1 } , \dots , \Theta ^ { N } , v ^ { K } )$
10: return D
```

Algorithm 2 Online Action Selection   
Input: $D = ( \Theta ^ { 1 } , \dots , \Theta ^ { N } , V ^ { K } )$ , current state $s ,$ observed window $c = ( \theta _ { 0 } , \ldots , \theta _ { \ell - 1 } )$   
Output: Action $\pi _ { D } ( s , c )$   
1: $U _ { \ell , c } ^ { V ^ { K } } ( u , i _ { 0 } , \ldots , i _ { \ell - 1 } ) \gets V ^ { K } ( u , i _ { 0 } , \ldots , i _ { \ell - 1 } )$ for all $( u , i _ { 0 } , \ldots , i _ { \ell - 1 } ) \in \mathcal { C } _ { N }$   
2: for $k = \ell - 1 , \ldots , 1$ do   
for all $( u , i _ { 0 } , \ldots , i _ { k - 1 } ) \in \mathcal { S } \times [ N ] ^ { k }$ do   
4: $U _ { k , c } ^ { V ^ { K } } ( u , i _ { 0 } , \ldots , i _ { k - 1 } ) \gets \operatorname* { m a x } _ { a \in \mathcal { A } } \{ r ( u , a ) + \frac { \gamma } { N } \sum _ { i _ { k } = 1 } ^ { N } U _ { k + 1 , c } ^ { V ^ { K } } \big ( \theta _ { k } ( u , a ) , i _ { 0 } , \ldots , i _ { k } \big ) \}$   
5: end for   
6: end for   
7: for all $a \in { \mathcal { A } }$ do   
8: $\widetilde { Q } _ { V ^ { \kappa } } ( a )  r ( s , a ) + \frac { \gamma } { N } \sum _ { i _ { 0 } = 1 } ^ { N } U _ { 1 , c } ^ { V ^ { \kappa } } ( \theta _ { 0 } ( s , a ) , i _ { 0 } )$   
9: end for   
10: return $\pi _ { D } ( s , c )  \arg \operatorname* { m a x } _ { a \in \mathcal { A } } \widetilde { Q } _ { V ^ { K } } ( a )$

Offline phase. Given N and K, sample a dictionary $\Theta ^ { 1 } , \dots , \Theta ^ { N } \stackrel { \mathrm { i . i . d . } } { \sim } \ Q$ and define $\mathcal { C } _ { N } : = \mathcal { S } \times$ $[ N ] ^ { \ell }$ , where $( i _ { 0 } , \dots , i _ { \ell - 1 } ) \in [ N ] ^ { \ell }$ represents the window $( \Theta ^ { i _ { 0 } } , \ldots , \Theta ^ { i _ { \ell - 1 } } )$ . Algorithm 1 runs K value-iteration steps on $\mathcal { C } _ { N }$ (as defined equation 3), obtaining $V ^ { \acute { K } } : \mathcal { C } _ { N }  [ \overset { } { 0 } , V _ { \mathrm { m a x } } ]$ , where $V _ { \mathrm { m a x } } =$ $R _ { \operatorname* { m a x } } / ( 1 - \gamma )$ Hence, the offline output $D = ( \Theta ^ { 1 } , \dots , \Theta ^ { N } , V ^ { \bar { K } } )$ is computed once, without enumerating Ω<sup>ℓ</sup>.

Online phase. At decision time, the agent observes its current state s and a look-ahead window $c = ( \bar { \theta _ { 0 } } , \dots , \theta _ { \ell - 1 } ) \in \Omega ^ { \ell }$ . In general, the tables $\theta _ { 0 } , \ldots , \theta _ { \ell - 1 }$ are different from those sampled offline, so $V ^ { K }$ cannot be evaluated directly on the observed window. Instead, Algorithm 2 connects the observed window to the values computed offline through a backward recursion. Starting from $V ^ { K }$ , the recursion processes the observed tables in reverse order. Initialize

$$
\begin{array} { r } { { U _ { \ell , c } ^ { V ^ { K } } } ( u , i _ { 0 } , \ldots , i _ { \ell - 1 } ) : = { V ^ { K } } ( u , i _ { 0 } , \ldots , i _ { \ell - 1 } ) , \quad \forall ( u , i _ { 0 } , \ldots , i _ { \ell - 1 } ) \in \mathcal { C } _ { N } } \end{array}\tag{4}
$$

Then, for $m = \ell - 1 , \ldots , 1$ , define

$$
U _ { m , c } ^ { V ^ { K } } ( u , i _ { 0 } , \dots , i _ { m - 1 } ) : = \operatorname* { m a x } _ { a \in \mathcal { A } } \left\{ r ( u , a ) + \frac { \gamma } { N } \sum _ { i _ { m } = 1 } ^ { N } U _ { m + 1 , c } ^ { V ^ { K } } \big ( \theta _ { m } ( u , a ) , i _ { 0 } , \dots , i _ { m } \big ) \right\} .\tag{5}
$$

At level $m _ { : }$ , the observed table $\theta _ { m }$ determines the next state, while the transition table entering beyond the observed window is averaged over the $N$ samples drawn offline. Proceeding backward in this way incorporates the entire observed look-ahead window into the precomputed values. Once $\theta _ { 1 } , \ldots , \theta _ { \ell - 1 }$ have been processed, each action at the current state is assigned the score

$$
\widetilde { Q } _ { V ^ { \kappa } } ( s , c , a ) : = r ( s , a ) + \frac { \gamma } { N } \sum _ { i _ { 0 } = 1 } ^ { N } U _ { 1 , c } ^ { V ^ { \kappa } } \left( \theta _ { 0 } ( s , a ) , i _ { 0 } \right) .\tag{6}
$$

And the policy selects : $\pi _ { D } ( s , c ) = \arg \operatorname* { m a x } _ { a \in \mathcal { A } } \widetilde { Q } _ { V ^ { K } } ( s , c , a )$

## 4.2 THEORETICAL GUARANTEES

Theorem 2 (RPTAS for perfect transition look-ahead). Fix $\ell \geq 2$ and $\varepsilon , \delta \in ( 0 , 1 )$ . For suitable choices of the sample size N and the number of value-iteration steps K, algorithm 1–2 described above computes a policy π such that

$$
\mathbb { P } \Big ( V _ { \ell } ^ { \pi } ( s , c ) \geq V _ { \ell } ^ { \star } ( s , c ) - \varepsilon V _ { \mathrm { m a x } } , \qquad \forall ( s , c ) \in \mathcal { S } \times \Omega _ { \mathbf { \varepsilon } } ^ { \ell } \Big ) \geq 1 - \delta .\tag{7}
$$

For every fixed ℓ, computing and storing the policy, as well as selecting an action at each decision time, require time and memory polynomial in $n , \dot { m _ { \ell } } \varepsilon ^ { - 1 } , \log ( 1 / \delta )$ , and $( 1 - \gamma ) ^ { - 1 }$

Proofsketch. We first control the error introduced by replacing the true transition distribution with the $N$ sampled tables. Let $\begin{array} { r } { \widehat { \mathsf Q } _ { N } : = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \delta _ { \Theta ^ { j } } } \end{array}$ be the distribution supported on the sampled transition tables, and let $\widehat { V } ^ { \star }$ and $\widehat { Q } ^ { \star }$ denote the optimal value and action-value functions obtained by replacing Q with ${ \widehat { \mathsf { Q } } } _ { N }$ . Since $V _ { \ell } ^ { \star }$ is fixed independently of the samples, Hoeffding’s inequality and a union bound over all states and look-ahead suffixes give, with probability at least $\dot { 1 } - \dot { \delta } .$

$\begin{array} { r } { \underset { u \in S } { \operatorname* { s u p } } \left| \frac { 1 } { N } \sum _ { j = 1 } ^ { N } V _ { \ell } ^ { \star } ( u , \theta _ { 1 : \ell - 1 } , \Theta ^ { j } ) - \mathbb { E } _ { \Theta \sim \mathbb { Q } } \left[ V _ { \ell } ^ { \star } ( u , \theta _ { 1 : \ell - 1 } , \Theta ) \right] \right| \leq \eta . } \end{array}$ . Then, usual contraction argument yields $\begin{array} { r } { \| \widehat { V } ^ { \star } - V _ { \ell } ^ { \star } \| _ { \infty } \leq \frac { \gamma \eta } { 1 - \gamma } } \end{array}$ , and $\begin{array} { r } { \| \widehat { Q } ^ { \star } - Q _ { \ell } ^ { \star } \| _ { \infty } \leq \frac { \gamma \eta } { 1 - \gamma } } \end{array}$ . The important point is that this bound holds simultaneously for every possible look-ahead window.

The idea is to use the same sampled tables both to define the empirical distribution ${ \widehat { \mathsf { Q } } } _ { N }$ and to construct the finite state space $\mathcal { C } _ { N }$ . Consider the finite MDP induced by the dictionary, with state space $\mathcal { C } _ { N }$ . From a state $( s , i _ { 0 } , \ldots , i _ { \ell - 1 } )$ , taking action a yields reward $r ( s , a )$ and the next state $s ^ { \prime } =$ $\bar { \Theta } ^ { i _ { 0 } } ( s , a ) , ( i _ { 1 } , \ldots , i _ { \ell - 1 } , J ) , J \sim \mathrm { U n i f } ( [ N ] )$ . This is exactly the dynamics of the augmented MDP under $\hat { \mathsf { Q } } _ { N } { \boldsymbol { : } }$ when the current window is $( \Theta ^ { i _ { 0 } } , \ldots , \Theta ^ { i _ { \ell - 1 } } )$ , the next table is uniformly distributed over $\Theta ^ { 1 } , \dots , \Theta ^ { N }$ . Therefore, the finite MDP and the empirical augmented MDP have identical Bellman equations on dictionary windows. Their optimal values consequently satisfy $V _ { N } ^ { \star } ( s , i _ { 0 } , \ldots , i _ { \ell - 1 } ) =$ $\widehat { V } ^ { \star } \left( s , \Theta ^ { i _ { 0 } } , \ldots , \Theta ^ { i _ { \ell - } } \right)$ , where $V _ { N } ^ { * }$ is the optimal value function on $C _ { N }$ . Since Algorithm 1 performs only K value-iteration steps,

$$
\operatorname* { s u p } _ { \substack { s , i _ { 0 } , \ldots , i _ { \ell - 1 } } } \left| V ^ { K } ( s , i _ { 0 } , \ldots , i _ { \ell - 1 } ) - \widehat { V } ^ { \star } \big ( s , \Theta ^ { i _ { 0 } } , \ldots , \Theta ^ { i _ { \ell - 1 } } \big ) \right| \leq \gamma ^ { K } V _ { \operatorname* { m a x } } .
$$

Online, the observed window $c = ( \theta _ { 0 } , \ldots , \theta _ { \ell - 1 } )$ need not belong to the dictionary. The online recursion does not approximate this window: it inserts its tables exactly, from $\theta _ { \ell - 1 }$ back to $\theta _ { 0 }$ and uses the dictionary only to average over the new table entering beyond the observed horizon. Starting from the exact offline value $V _ { N } ^ { \star }$ , backward induction gives

$$
U _ { k , c } ^ { V _ { N } ^ { \star } } ( u , i _ { 0 } , \ldots , i _ { k - 1 } ) = \widehat { V } ^ { \star } \big ( u , \theta _ { k } , \ldots , \theta _ { \ell - 1 } , \Theta ^ { i _ { 0 } } , \ldots , \Theta ^ { i _ { k - 1 } } \big ) .
$$

At the root, this yields $\widetilde { Q } _ { V _ { N } ^ { \star } } ( s , c , a ) = \widehat { Q } ^ { \star } ( s , c , a )$ . Using $V ^ { K }$ instead of $V _ { N } ^ { \star }$ introduces at most $\gamma ^ { K + \ell } V _ { \mathrm { m a x } }$ error, since each of the ℓ online backups is γ-Lipschitz. Together with the sampling error, uniformly over all $( s , c , a )$ ,

$$
\Bigl | \widetilde { Q } _ { V ^ { \kappa } } ( s , c , a ) - Q _ { \ell } ^ { \star } ( s , c , a ) \Bigr | \le \frac { \gamma \eta } { 1 - \gamma } + \gamma ^ { K + \ell } V _ { \mathrm { m a x } } .
$$

Set $\begin{array} { r l r l r } { \eta } & { = } & { \frac { \varepsilon ( 1 - \gamma ) ^ { 2 } V _ { \mathrm { m a x } } } { 4 } , N } & { = } & { \left\lceil \frac { 8 } { \varepsilon ^ { 2 } ( 1 - \gamma ) ^ { 4 } } \left( \log \frac { 2 } { \delta } + \left[ 1 + ( \ell - 1 ) n m \right] \log n \right) \right\rceil } \end{array}$ , and $\begin{array} { r l } { K } & { { } = } \end{array}$ $\begin{array} { r } { \left\lceil \frac { 1 } { 1 - \gamma } \log \frac { 4 } { \varepsilon ( 1 - \gamma ) } \right\rceil } \end{array}$ . The score error is then at most $\varepsilon ( 1 - \gamma ) V _ { \mathrm { m a x } } / 2$ simultaneously for every window. The preprocessing time, query time, and representation size are respectively

$$
O \big ( N n m + K n ( m + 1 ) N ^ { \ell } \big ) , \qquad O \big ( \ell n ( m + 1 ) N ^ { \ell } \big ) , \qquad O \big ( N n m + n N ^ { \ell } + \ell n m \big ) ,
$$

and are polynomial for fixed ℓ.

## 5 REGRET MINIMIZATION

We now consider the online setting in which both the transition kernel $P$ and the mean reward function $r : \mathcal { S } \times \mathcal { A } \to [ 0 , \bar { R _ { \mathrm { m a x } } } ]$ are initially unknown. At time t, the learner observes $X _ { t } ~ = ~ ( S _ { t } , C _ { t } ^ { \ell } )$ , chooses $A _ { t }$ , and receives a stochastic reward $R _ { t } ~ \in ~ [ 0 , R _ { \operatorname* { m a x } } ]$ with conditional mean $r ( S _ { t } , A _ { t } )$ We measure performance through the cumulative Bellman gap $\mathrm { R e g } _ { \ell } ( T ) : =$ $\begin{array} { r } { \sum _ { t = 0 } ^ { T - 1 } \left[ V _ { \ell } ^ { \star } ( X _ { t } ) - Q _ { \ell } ^ { \star } ( X _ { t } , A _ { t } ) \right] } \end{array}$

We introduce DLA-UCB (Dictionary Look-Ahead with Upper Confidence Bounds), an online extension of the planner from Section 4. At each update, the learner estimates the transition law from past tables, constructs optimistic reward estimates, samples a fresh dictionary from the estimated model, and solves the resulting finite MDP using an optimistic Bellman backup. At decision time, it applies the same backward recursion as before, with the observed look-ahead window and the optimistic backup.

More precisely, from N past transition tables, the learner estimates each marginal $\textstyle P ( \cdot \mid s , a )$ and forms the corresponding product distribution ${ \widehat { \sf Q } } _ { N }$ . It then samples $Z ^ { 1 } , \dots , Z ^ { M _ { N } } \overset { \mathrm { i . i . d . } } { \sim } \widehat { \sf Q } _ { N } , M _ { N } =$ $\left\lceil \frac { N } { ( 1 - \gamma ) ^ { 3 } } \right\rceil$ , and runs optimistic value iteration on $\mathcal { S } \times [ M _ { N } ] ^ { \ell }$ . For any vector $y = ( y _ { 1 } , \dots , y _ { M _ { N } } ) \in$ $[ 0 , V _ { \operatorname* { m a x } } ] ^ { M _ { N } }$ , let y and $s ( y )$ denote its empirical mean and standard deviation, and define

$$
\begin{array} { r } { U _ { N } ( y ) : = \operatorname* { m i n } \left\{ V _ { \operatorname* { m a x } } , \overline { { y } } + 8 V _ { \operatorname* { m a x } } \alpha _ { N } \rho _ { N } + \operatorname* { m a x } \left\{ 7 \alpha _ { N } s ( y ) , 4 9 V _ { \operatorname* { m a x } } \alpha _ { N } ^ { 2 } \right\} \right\} . } \end{array}\tag{8}
$$

In a Bellman update, y<sub>j</sub> is the value of the successor obtained by appending the jth dictionary table: $y _ { j } = v \big ( Z ^ { i _ { 0 } } ( s , a ) , i _ { 1 } , . . . , i _ { \ell - 1 } , j \big )$ . Thus, $U _ { N } ( y )$ replaces the empirical average of the possible successor values by an optimistic, variance-adaptive estimate. The variance term provides a Bernstein-type transition bonus, the $\alpha _ { N } \rho _ { N }$ term controls data-dependent continuation values, and the quadratic term preserves monotonicity. The optimistic Bellman operator is then

$$
\begin{array} { r l } & { ( \mathcal { T } _ { N , r ^ { + } } ^ { D , + } v ) ( s , i _ { 0 } , \ldots , i _ { \ell - 1 } ) } \\ & { \quad : = \underset { a \in A } { \operatorname* { m a x } } \operatorname* { m i n } \Biggl \{ V _ { \operatorname* { m a x } } , r ^ { + } ( s , a ) + \gamma U _ { N } \left( \bigl ( v \bigl ( Z ^ { i _ { 0 } } ( s , a ) , i _ { 1 } , \ldots , i _ { \ell - 1 } , j \bigr ) \bigr ) _ { j = 1 } ^ { M _ { N } } \right) \Biggr \} . } \end{array}\tag{9}
$$

Starting from $v ^ { 0 } \equiv V _ { \mathrm { m a x } } , { \mathrm { D L A - U C B - U P D A T E } }$ applies this operator $\begin{array} { r } { K _ { N } = \left\lceil \frac { 1 } { 1 - \gamma } \log \frac { 2 N ^ { 2 } } { 1 - \gamma } \right\rceil } \end{array}$ times. During interaction, the resulting value array is extended to each observed window using the same optimistic backup. The planner is recomputed only when the transition sample size or a reward count doubles; between updates, the dictionary and value array remain fixed.

Algorithm 3 DLA-UCB-UPDATE   
Input: Sample size N, past transition tables $\Theta _ { 0 } , \dots , \Theta _ { N - 1 }$ , current reward bound $r ^ { + }$ , horizon $T ,$   
confidence level δ   
Output: $\mathcal { D } = ( Z ^ { 1 : M _ { N } } , v ^ { K _ { N } } , r ^ { + } , \alpha _ { N } , \rho _ { N } )$   
1: Compute $\widehat { \sf Q } _ { N } , \alpha _ { N } , \rho _ { N } , M _ { N } ,$ and $K _ { N }$   
2: Sample $Z ^ { 1 } , \dots , Z ^ { M _ { N } } \overset { \mathrm { i . i . d . } } { \sim } \widehat { \sf Q } _ { N }$   
3: Initialize $v ^ { 0 } \equiv \dot { V } _ { \mathrm { m a x } } \ : \mathrm { o n } \ : { \cal S } \times \lbrack M _ { N } \rbrack ^ { \ell }$   
4: for $k = 0 , \ldots , K _ { N } - 1$ do   
5: $v ^ { k + 1 } \gets T _ { N , r ^ { + } } ^ { D , + } v ^ { k }$   
6: end for   
7: return $\mathcal { D } = ( Z ^ { 1 : M _ { N } } , v ^ { K _ { N } } , r ^ { + } , \alpha _ { N } , \rho _ { N } )$

Algorithm 4 DLA-UCB-ONLINE   
Input: Horizon T, look-ahead depth ℓ, confidence level δ   
Output: Actions $A _ { 0 } , \ldots , A _ { T - 1 }$   
1: Set $r ^ { + } ( s , a ) \gets R _ { \mathrm { m a x } }$ for every $( s , a )$ and $\mathcal { D }  \emptyset$   
2: Observe $X _ { 0 } = ( S _ { 0 } , C _ { 0 } ^ { \ell } )$ and play an arbitrary action $A _ { 0 }$   
3: Observe $R _ { 0 } ,$ update the reward statistics, and store $\Theta _ { 0 }$   
4: for $t = 1 , \ldots , \mathbf { \bar { \boldsymbol { T } } } - 1$ do   
5: Observe $\dot { X } _ { t } = ( S _ { t } , C _ { t } ^ { \ell } )$   
6: if $\mathcal { D } = \mathcal { D } ,$ t is a power of two, or a reward count reached a new power of two after time $t - 1$   
then   
7: Set $N \gets 2 ^ { \lfloor \log _ { 2 } t \rfloor }$   
8: $\mathcal { D } \gets \mathrm { D L A - U C B - U P D A T E } ( N , \Theta _ { 0 : N - 1 } , r ^ { + } , T , \delta )$   
9: end if   
10: Compute $\widetilde { Q } _ { t } ( X _ { t } , \cdot )$ from $( Z ^ { 1 : M _ { N } } , v ^ { K _ { N } } , r ^ { + } , \alpha _ { N } , \rho _ { N } )$ using the online recursion   
11: Play $A _ { t } \in \arg \operatorname* { m a x } _ { a \in \mathcal { A } } \widetilde Q _ { t } ( X _ { t } , a )$   
12: Observe $R _ { t }$ , update the reward statistics, and store $\Theta _ { t }$   
13: end for

There are at most O(nm log T) recomputations. One call to DLA-UCB-UPDATE costs $O \left( K _ { N } n ( m + 1 ) M _ { N } ^ { \ell } \right)$ , while the online action-selection recursion costs $O ( \ell n ( m + 1 ) M _ { N } ^ { \ell } )$ per decision. Therefore, for fixed $\ell ,$ both costs are polynomial in $n , m , T$ , and $( 1 - \gamma ) ^ { - 1 }$

Theorem 3 (Regret of DLA-UCB). Fix the look-ahead depth ℓ. For every $T \geq 2$ and $\delta \in ( 0 , 1 )$ with probability at least $1 - \delta ,$ DLA-UCB satisfies

$$
\mathrm { R e g } _ { \ell } ( T ) = \widetilde { \cal O } \left( R _ { \mathrm { m a x } } \sqrt { \frac { n m T } { 1 - \gamma } } + \frac { R _ { \mathrm { m a x } } n ^ { 2 } m } { ( 1 - \gamma ) ^ { 2 } } \right) .\tag{10}
$$

The leading term matches, up to logarithmic factors, the optimal dependence on $n , m , T ,$ , and $( 1 - \gamma ) ^ { - 1 }$ obtained for classical tabular discounted MDPs (Liu & Su, 2021; He et al., 2022). Thus, despite the exponentially large augmented state space, transition look-ahead does not worsen the leading statistical dependence of the regret. The lower-order term comes from the uniform transition confidence bound required for the data-dependent optimistic value functions.

Proofsketch. Fix a planner update based on N previously observed transition tables. The learner first estimates each marginal $\textstyle P ( \cdot \mid s , a )$ and takes their product to obtain $\widehat { \sf Q } _ { N }$ . This may still have exponentially large support. To obtain a finite planning problem, DLA-UCB then draws an independent dictionary $\Theta ^ { 1 } , \ldots , \Theta ^ { M _ { N } } \stackrel { \mathrm { i . i . d . } } { \sim } \widehat { \mathsf { Q } } _ { N }$ . Thus, the transition approximation occurs in two steps: ${ \sf Q } \longrightarrow \widehat { \sf Q } _ { N } \longrightarrow \{ \Theta ^ { 1 } , \dots , \Theta ^ { M _ { N } } \}$ . The first step is a statistical estimation error, whereas the second is the Monte Carlo error used to make planning computationally tractable.

Statistical estimation error. First consider an ideal planner that uses the same bonuses as DLA UCB, but computes expectations and variances exactly under ${ \widehat { \sf Q } } _ { N }$ . For any bounded function V :

$$
\begin{array} { r l } & { \Omega \to [ 0 , V _ { \operatorname* { m a x } } ] , \mathrm { d e f i n e : } } \\ & { \mathrm { U C B } _ { N } ( V ) } \\ & { : = \operatorname* { m i n } \left\{ V _ { \operatorname* { m a x } } , \mathbb { E } _ { \Theta \sim \widehat { \Theta } _ { N } } [ V ( \Theta ) ] + 8 V _ { \operatorname* { m a x } } \alpha _ { N } \rho _ { N } + \operatorname* { m a x } \left\{ 7 \alpha _ { N } \sqrt { \mathrm { V a r } _ { \widehat { \Theta } _ { N } } ( V ( \Theta ) ) } , 4 9 V _ { \operatorname* { m a x } } \alpha _ { N } ^ { 2 } \right\} \right\} . } \end{array}
$$

Now, fix $c = ( \theta _ { 0 } , \ldots , \theta _ { \ell - 1 } )$ and $a \in { \mathcal { A } }$ , and let $\sigma _ { \mathsf { Q } }$ and ${ \widehat { \sigma } } _ { N }$ be the standard deviations, under Q and $\widehat { \sf Q } _ { N }$ , of $V _ { \ell } ^ { \star } ( \theta _ { 0 } ( s , a ) , \theta _ { 1 : \ell - 1 } , \Theta )$ . The Bernstein bound gives

$$
\mathbb { E } _ { \mathbb { Q } } \left[ V _ { \ell } ^ { \star } ( \theta _ { 0 } ( s , a ) , \theta _ { 1 : \ell - 1 } , \Theta ) \right] \le \mathbb { E } _ { \widehat { \mathbb { Q } } _ { N } } \left[ V _ { \ell } ^ { \star } ( \theta _ { 0 } ( s , a ) , \theta _ { 1 : \ell - 1 } , \Theta ) \right] + \sqrt { 2 } \alpha _ { N } \sigma _ { \mathbb { Q } } + \frac { V _ { \operatorname* { m a x } } \alpha _ { N } ^ { 2 } } { 3 } .\tag{11}
$$

And Hellinger bound gives $\sigma _ { \mathsf { Q } } \leq \widehat { \sigma } _ { N } + 2 V _ { \operatorname* { m a x } } \rho _ { N }$ . The bonus in $\mathrm { U C B } _ { N }$ therefore dominates the preceding estimation error, so, uniformly over $( s , c , a )$

$$
\mathrm { U C B } _ { N } \left( V _ { \ell } ^ { \star } ( \theta _ { 0 } ( s , a ) , \theta _ { 1 : \ell - 1 } , \cdot ) \right) \geq \mathbb { E } _ { \mathbf { Q } } \left[ V _ { \ell } ^ { \star } ( \theta _ { 0 } ( s , a ) , \theta _ { 1 : \ell - 1 } , \Theta ) \right] .
$$

Since $r ^ { + } \geq r ,$ the ideal Bellman operator satisfies $\overline { { \mathcal { T } } } _ { N , r ^ { + } } V _ { \ell } ^ { \star } \geq \mathcal { T } _ { \ell } V _ { \ell } ^ { \star } = V _ { \ell } ^ { \star }$ . Monotonicity and contraction then imply that its fixed point is optimistic: $\overline { { V } } _ { N , r ^ { + } } \geq V _ { \ell } ^ { \star }$

Finite-dictionary approximation. The ideal planner averages over all transition tables under $\widehat { \sf Q } _ { N }$ whose support may be exponentially large. DLA-UCB instead samples $\Theta ^ { 1 } , \ldots , \Theta ^ { M _ { N } } \stackrel { \mathrm { i . i . d . } } { \sim } \widehat { \mathsf { Q } } _ { N }$ and replaces each expectation $\mathbb { E } _ { \Theta \sim \widehat { \Omega } _ { N } } \left[ \overline { { V } } _ { N , r ^ { + } } \left( \theta _ { 0 } ( s , a ) , \theta _ { 1 : \ell - 1 } , \Theta \right) \right]$ by the dictionary average $\begin{array} { r } { \frac { 1 } { M _ { N } } \sum _ { j = 1 } ^ { M _ { N } } \overline { { V } } _ { N , r ^ { + } } \left( \theta _ { 0 } ( s , a ) , \theta _ { 1 : \ell - 1 } , \Theta ^ { j } \right) } \end{array}$ , and similarly replaces the corresponding standard deviation by its empirical counterpart. Conditionally on the past data, $\widehat { \sf Q } _ { N }$ and $\overline { { V } } _ { N , r ^ { + } }$ are fixed. Standard concentration therefore shows that replacing the exact mean and standard deviation by their dictionary estimates changes one Bellman backup by at most $\widetilde { O } \left( V _ { \mathrm { m a x } } \sqrt { \frac { n m } { M _ { N } } } \right)$ uniformly over all states and look-ahead suffixes. Since both Bellman operators are γ-contractions, their fixed points differ by at most this one-step error divided by $1 - \gamma$ . Stopping value iteration after $K _ { N }$ steps adds $\gamma ^ { K _ { N } } V _ { \mathrm { m a x } } ,$ so $\begin{array} { r } { \varepsilon _ { N } ^ { D } \leq \widetilde { O } _ { \ell } \left( \frac { V _ { \operatorname* { m a x } } } { 1 - \gamma } \sqrt { \frac { n m } { M _ { N } } } \right) + \gamma ^ { K _ { N } } V _ { \operatorname* { m a x } } } \end{array}$ . With $\begin{array} { r } { M _ { N } = \left\lceil \frac { N } { ( 1 - \gamma ) ^ { 3 } } \right\rceil } \end{array}$ and $\begin{array} { r } { K _ { N } = \left\lceil \frac { 1 } { 1 - \gamma } \log \frac { 2 N ^ { 2 } } { 1 - \gamma } \right\rceil } \end{array}$ hi b

$$
\varepsilon _ { N } ^ { D } = \widetilde { O } _ { \ell } \left( R _ { \mathrm { m a x } } \sqrt { \frac { n m } { N ( 1 - \gamma ) } } + \frac { R _ { \mathrm { m a x } } } { N ^ { 2 } } \right) .
$$

As in the planning analysis, the online recursion inserts the observed tables exactly and uses the same backup as the finite Bellman operator. It therefore introduces no new approximation, and

$$
\operatorname* { s u p } _ { s , c , a } \left| \widetilde { Q } _ { N , r ^ { + } } ( s , c , a ) - \overline { { Q } } _ { N , r ^ { + } } ( s , c , a ) \right| \leq \varepsilon _ { N } ^ { D } .
$$

Regret bound. Let $\Delta _ { t } ( x ) : = \overline { { V } } _ { N _ { t } , r _ { i } ^ { + } } ( x ) - V _ { \ell } ^ { \star } ( x ) \geq 0$ denote the optimism gap of the ideal planner active at time t. Approximate greediness and the optimistic Bellman equation bound the instantaneous Bellman gap by the reward-estimation error, the dictionary error, the transition-confidence width, and

$$
\gamma \mathbb { E } [ \Delta _ { t } ( X _ { t + 1 } ) \mid \mathcal { F } _ { t } ] - \Delta _ { t } ( X _ { t } ) .
$$

The last two terms do not accumulate over time. Indeed, if the planner remains unchanged between times p and q, replacing conditional expectations by realized values gives

$$
\sum _ { t = p } ^ { q } \left[ \gamma \Delta _ { t } ( X _ { t + 1 } ) - \Delta _ { t } ( X _ { t } ) \right] = - \Delta _ { p } ( X _ { p } ) + \gamma \Delta _ { p } ( X _ { q + 1 } ) - ( 1 - \gamma ) \sum _ { t = p + 1 } ^ { q } \Delta _ { p } ( X _ { t } ) .
$$

A predictable-to-realized concentration argument justifies this replacement. Since $0 \leq \Delta _ { p } \leq V _ { \operatorname* { m a x } } .$ the boundary terms cost at most $V _ { \mathrm { m a x } }$ per planner update, while the negative sum absorbs the part of the transition width depending on the optimism gap.

Summing the remaining reward, dictionary, and confidence terms with the doubling schedule gives

$$
\mathrm { R e g } _ { \ell } ( T ) \leq \widetilde { \cal O } _ { \ell } \left( R _ { \mathrm { m a x } } \sqrt { \frac { n m T } { 1 - \gamma } } + \frac { R _ { \mathrm { m a x } } n ^ { 2 } m } { ( 1 - \gamma ) ^ { 2 } } \right) + C \sum _ { t < T } \gamma \alpha _ { t } \sigma _ { t } ,
$$

where $\sigma _ { t } ^ { 2 } = \operatorname { V a r } ( V _ { \ell } ^ { \star } ( X _ { t + 1 } ) \mid { \mathcal { F } } _ { t } )$ . It remains to control this Bernstein term. The Bellman equation and Freedman’s inequality imply

$$
\sum _ { t < T } \gamma ^ { 2 } \sigma _ { t } ^ { 2 } \lesssim V _ { \operatorname* { m a x } } R _ { \operatorname* { m a x } } T + V _ { \operatorname* { m a x } } \mathrm { R e g } _ { \ell } ( T ) + V _ { \operatorname* { m a x } } ^ { 2 } \log ( 1 / \delta ) .
$$

Together with $\begin{array} { r } { \sum _ { t < T } \alpha _ { t } ^ { 2 } = \widetilde { O } _ { \ell } ( n m ) } \end{array}$ , Cauchy–Schwarz yields

$$
\sum _ { t < T } \gamma \alpha _ { t } \sigma _ { t } = \widetilde { O } _ { \ell } \left( R _ { \mathrm { m a x } } \sqrt { \frac { n m T } { 1 - \gamma } } + \sqrt { n m V _ { \mathrm { m a x } } \mathrm { R e g } _ { \ell } ( T ) } \right) .
$$

Young’s inequality absorbs the last term into the regret itself, giving

$$
\mathrm { R e g } _ { \ell } ( T ) = \widetilde { O } _ { \ell } \left( R _ { \mathrm { m a x } } \sqrt { \frac { n m T } { 1 - \gamma } } + \frac { R _ { \mathrm { m a x } } n ^ { 2 } m } { ( 1 - \gamma ) ^ { 2 } } \right) .
$$

## 6 CONCLUSION

We studied the computational complexity of planning with multi-step transition look-ahead in discounted MDPs. We first showed that exact planning remains NP-hard for every fixed rational discount factor and every fixed look-ahead horizon $\bar { \ell } \geq 2 .$ Despite this hardness, we proved that near-optimal planning is computationally tractable: for every fixed ℓ, our randomized approximation scheme constructs, with high probability, a uniformly near-optimal policy in polynomial time and memory. The key algorithmic idea is to replace the distribution of transition realizations by a finite dictionary of complete transition tables, solve the resulting empirical Bellman problem on a finite core, and extend the resulting value function to arbitrary observed look-ahead windows at deployment. These results show that the exponential augmented state space induced by transition look-ahead does not by itself preclude efficient approximate control. An interesting direction for future work is to determine whether similar approximation guarantees can be obtained when the look-ahead depth is part of the input, or under imperfect, dependent, or partially observed transition predictions.

## REFERENCES

Mohammad Gheshlaghi Azar, Ian Osband, and Remi Munos. Minimax regret bounds for reinforce-´ ment learning. In International conference on machine learning, pp. 263–272. PMLR, 2017.

Nikhil Balaji, Stefan Kiefer, Petr Novotny, Guillermo A P\` erez, and Mahsa Shirmohammadi. On the´ complexity of value iteration. arXiv preprint arXiv:1807.04920, 2018.

Ziyad Benomar and Vianney Perchet. On tradeoffs in learning-augmented algorithms. In Yingzhen Li, Stephan Mandt, Shipra Agrawal, and Emtiyaz Khan (eds.), Proceedings of The 28th International Conference on Artificial Intelligence and Statistics, volume 258 of Proceedings ofMachine Learning Research, pp. 802–810. PMLR, 03–05 May 2025. URL https://proceedings. mlr.press/v258/benomar25a.html.

Ziyad Benomar, Lorenzo Croissant, Vianney Perchet, and Spyros Angelopoulos. Pareto-optimality, smoothness, and stochasticity in learning-augmented one-max-search. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum? id=ZTOVlPC5Vf.

Yichen Chen and Mengdi Wang. Lower bound on the computational complexity of discounted markov decision problems, 2017. URL https://arxiv.org/abs/1705.07312.

Jiafan He, Dongruo Zhou, and Quanquan Gu. Nearly minimax optimal reinforcement learning for discounted mdps, 2022. URL https://arxiv.org/abs/2010.00587.

Thomas Jaksch, Rudolf Ortner, and Peter Auer. Near-optimal regret bounds for reinforcement learning. In Journal ofMachine Learning Research, volume 11, pp. 1563–1600, 2010.

Chi Jin, Zeyuan Allen-Zhu, Sebastien Bubeck, and Michael I. Jordan. Is q-learning provably effi-´ cient? In Advances in Neural Information Processing Systems (NeurIPS), volume 31, pp. 4863– 4873, 2018.

Michael Kearns, Yishay Mansour, and Andrew Y. Ng. A sparse sampling algorithm for near-optimal planning in large markov decision processes. Machine Learning, 49(2–3):193–208, 2002. doi: 10.1023/A:1017932429737.

Anton J. Kleywegt, Alexander Shapiro, and Tito Homem-de Mello. The sample average approximation method for stochastic discrete optimization. SIAM Journal on Optimization, 12(2):479–502, 2002. doi: 10.1137/S1052623499363220.

Tongxin Li, Yiheng Lin, Shaolei Ren, and Adam Wierman. Beyond black-box advice: Learningaugmented algorithms for MDPs with q-value predictions. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id= RACcp8Zbr9.

Tongxin Li, Hao Liu, and Yisong Yue. Disentangling linear quadratic control with untrusted ml predictions. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 86860–86898. Curran Associates, Inc., 2024. URL https://proceedings.neurips.cc/paper\_files/paper/2024/file/ 9dff3b83d463fab213941bfee23341ba-Paper-Conference.pdf.

Michael L. Littman, Thomas L. Dean, and Leslie Pack Kaelbling. On the complexity of solving markov decision problems, 2013. URL https://arxiv.org/abs/1302.4971.

Shuang Liu and Hao Su. Regret bounds for discounted mdps, 2021. URL https://arxiv. org/abs/2002.05138.

Chenbei Lu, Zaiwei Chen, Tongxin Li, Chenye Wu, and Adam Wierman. Reinforcement learning with imperfect transition predictions: A bellman-jensen approach. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview. net/forum?id=DYuPwwDy9n.

Lixing Lyu, Jiashuo Jiang, and Wang Chi Cheung. Efficiently solving discounted MDPs via predictions with unknown prediction errors. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/forum?id=0nrxgFZEEq.

Davide Maran, Davide Salaorni, and Marcello Restelli. Learning in markov decision processes with exogenous dynamics, 2026. URL https://arxiv.org/abs/2603.02862.

Andreas Maurer and Massimiliano Pontil. Empirical bernstein bounds and sample variance penalization, 2009. URL https://arxiv.org/abs/0907.3740.

Aranyak Mehta, Uri Nadav, Alexandros Psomas, and Aviad Rubinstein. Hitting the high notes: Subset selection for maximizing expected order statistics. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 15800–15810. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper\_files/paper/2020/ file/b6417f112bd27848533e54885b66c288-Paper.pdf.

Nadav Merlis. Reinforcement learning with lookahead information. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview. net/forum?id=wlqfOvlTQz.

Nadav Merlis, Hugo Richard, Flore Sentenac, Corentin Odic, Mathieu Molina, and Vianney Perchet. On preemption and learning in stochastic scheduling. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 24478–24516. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr.press/v202/merlis23a.html.

Nadav Merlis, Dorian Baudry, and Vianney Perchet. The value of reward lookahead in reinforcement learning. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://arxiv.org/abs/2403.11637.

Michael Mitzenmacher and Sergei Vassilvitskii. Algorithms with predictions, 2020. URL https: //arxiv.org/abs/2006.09123.

Martin Mundhenk, Judy Goldsmith, Christopher Lusena, and Eric Allender. Complexity of finitehorizon markov decision process problems. In Proceedings of the 17th National Conference on Artificial Intelligence (AAAI), pp. 494–499, 2000.

C. H. Papadimitriou. The complexity of markov decision processes. Mathematics of Operations Research, 12(3):441–450, 1987.

Corentin Pla, Hugo Richard, Marc Abeille, Nadav Merlis, and Vianney Perchet. On the hardness of reinforcement learning with transition lookahead. In The 29th International Conference on Artificial Intelligence and Statistics, 2026a. URL https://openreview.net/forum?id= clyOoEL3pS.

Corentin Pla, Hugo Richard, Marc Abeille, and Vianney Perchet. Minimax pac bounds for learning in exogenous contextual mdps, 2026b. URL https://arxiv.org/abs/2606.25170.

Martin L. Puterman. Markov Decision Processes: Discrete Stochastic Dynamic Programming. John Wiley & Sons, Hoboken, NJ, 2nd edition, 2014.

Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. MIT Press, Cambridge, MA, 2nd edition, 2018.

Shoshana Vasserman, Michal Feldman, and Avinatan Hassidim. Implementing the wisdom of waze. In IJCAI, volume 15, pp. 660–666, 2015.

Thomas J. Walsh, Ali Nouri, Lihong Li, and Michael L. Littman. Learning and planning in environments with delayed feedback. Autonomous Agents and Multi-Agent Systems, 18(1):83–105, 2009. doi: 10.1007/s10458-008-9056-7.

## A APPENDIX

## A.1 PROOF OF THEOREM 1

We use the notation of Section $2 \colon \Omega = { \mathcal { S } } ^ { S \times { \mathcal { A } } }$ is the space of one-step transition tables, $\mathsf { Q }$ is the product law equation 1 induced by the kernel $P ,$ , and $V _ { \ell } ^ { \star }$ denotes the optimal value function on the augmented state space $s \times \Omega ^ { \ell }$ . For a policy π and an initial state $s _ { 0 } \in S ,$ define the values averaged over the initial look-ahead window,

$$
v _ { \ell , \gamma } ^ { \pi } ( s _ { 0 } ) : = \mathbb { E } _ { C \sim \mathsf { Q } ^ { \otimes \ell } } \bigl [ V _ { \ell } ^ { \pi } ( s _ { 0 } , C ) \bigr ] , \qquad v _ { \ell , \gamma } ^ { \star } ( s _ { 0 } ) : = \mathbb { E } _ { C \sim \mathsf { Q } ^ { \otimes \ell } } \bigl [ V _ { \ell } ^ { \star } ( s _ { 0 } , C ) \bigr ] ,\tag{12}
$$

so that $v _ { \ell , \gamma } ^ { \pi } ( s _ { 0 } )$ is the quantity appearing in the statement of Theorem 1. For a window $c =$ $( \theta _ { 0 } , \ldots , \theta _ { \ell - 1 } ) \in \Omega ^ { \ell } ,$ a state $s \in S ,$ , and an action sequence $\sigma = ( a _ { 0 } , \ldots , a _ { k - 1 } )$ with $k \leq \ell ,$ we write $\phi _ { c } ( s , \sigma ) \in \mathcal { S }$ for the state reached after playing σ from s under the window $c ,$ that is, the state $s _ { k }$ obtained from the recursion

$$
s _ { 0 } = s , \qquad s _ { j + 1 } = \theta _ { j } ( s _ { j } , a _ { j } ) , \qquad j = 0 , \ldots , k - 1 .
$$

Definition 2 (ℓ-DVDP). Given a finite MDP $\mathcal { M } = ( \mathcal { S } , \mathcal { A } , P , r )$ , a rational discount factor $\gamma \in$ $( 0 , 1 )$ , an initial state $s _ { 0 } \in S _ { : }$ , and a rational threshold θ, decide whether there exists a policy π with ℓ-step transition look-ahead such that $v _ { \ell , \gamma } ^ { \pi } ( s _ { 0 } ) \geq \theta ,$ , equivalently whether $v _ { \ell , \gamma } ^ { \star } ( s _ { 0 } ) \geq \bar { \theta }$

Fix an integer $\ell \geq 2$ and a rational discount factor $\gamma \in ( 0 , 1 )$ . We build upon Pla et al. (2026a) and exhibit a polynomial-time reduction from INDEPENDENT SET on regular graphs to ℓ-DVDP that is valid for this fixed $\gamma ;$ NP-hardness is therefore preserved even for $\gamma$ far from 1. We use the following consequence of the reduction of Mehta et al. (2020); it is the same gap gadget as in the hardness proof of Pla et al. (2026a). The graph parameters are denoted $n _ { G }$ and $m _ { G }$

Lemma 1 (Expected-maximum gap). There is a polynomial-time reduction that maps an instance $( G , k )$ of INDEPENDENT SET on a regular graph $G = ( V , E )$ , with $n _ { G } = | V |$ and $m _ { G } = | E |$ , to mutually independent, nonnegative, finite-support random variables $( X _ { v } ) _ { v \in V }$ with rational values and probabilities ofpolynomial encoding length. All variables have the same expectation $\mu ,$ and the following hold:

1. IfG contains an independent set $S ^ { \star }$ ofsize $k ,$ then

$$
\mathbb { E } \left[ \operatorname* { m a x } _ { v \in S ^ { \star } } X _ { v } \right] \geq k \mu - \frac { 2 } { m _ { G } } .
$$

2. IfG contains no independent set ofsize $k ,$ then every $S \subseteq V$ with $| S | = k$ satisfies

$$
\mathbb { E } \left[ \operatorname* { m a x } _ { v \in S } X _ { v } \right] \leq k \mu - 1 .
$$

We may assume without loss of generality that $m _ { G } \geq 3 ;$ instances with at most two edges can be decided in polynomial time. Set

$$
\Delta : = 1 - \frac { 2 } { m _ { G } } > 0 .
$$

Define

$$
B : = k \mu - 1 , \qquad L : = 1 + \operatorname* { m a x } \{ 0 , - B \} , \qquad U : = L + B .
$$

Then $U \geq 1$ . Shift every random variable by the same deterministic amount:

$$
Y _ { v } : = X _ { v } + L .
$$

For every nonempty $S \subseteq V$

$$
\mathbb { E } \left[ \operatorname* { m a x } _ { v \in S } Y _ { v } \right] = L + \mathbb { E } \left[ \operatorname* { m a x } _ { v \in S } X _ { v } \right] .
$$

Consequently, Lemma 1 becomes

$$
\mathrm { N O ~ c a s e : \quad } \mathbb { E } \left[ \operatorname* { m a x } _ { v \in S } Y _ { v } \right] \leq U \qquad { \mathrm { ~ f o r ~ e v e r y ~ } } S \subseteq V , \ | S | = k ,\tag{13}
$$

$$
\mathrm { Y E S ~ c a s e : } \quad \mathbb { E } \left[ \operatorname* { m a x } _ { v \in S ^ { \star } } Y _ { v } \right] \geq U + \Delta \qquad \mathrm { ~ f o r ~ s o m e ~ i n d e p e n d e n t ~ s e t ~ } S ^ { \star } , \ | S ^ { \star } | = k .\tag{14}
$$

The shift is used only to ensure that all rewards introduced below are nonnegative.

MDP construction. For each $v \in V$ , write the finite support of $Y _ { v }$ as

$$
\operatorname { s u p p } ( Y _ { v } ) = \{ y _ { v , 1 } , \dots , y _ { v , N _ { v } } \} , \qquad \mathbb { P } ( Y _ { v } = y _ { v , h } ) = p _ { v , h } .
$$

The state space of the MDP $\mathcal { M } _ { G }$ is

$$
\begin{array} { c } { S = \left\{ s _ { 0 } , s _ { 1 } , s _ { T } \right\} \cup \left\{ d _ { 1 } , \ldots , d _ { \ell - 2 } \right\} \cup \left\{ s _ { v } : v \in V \right\} } \\ { \cup \left\{ x _ { v , h } : v \in V , h \in \left[ N _ { v } \right] \right\} . } \end{array}\tag{15}
$$

The delay states $d _ { 1 } , \ldots , d _ { \ell - 2 }$ are absent when $\ell = 2$ . The action set is

$$
\begin{array} { r } { \mathcal { A } = \{ \mathsf { w a i t } , \mathsf { g o } , \mathsf { a d v a n c e } , \mathsf { c l a i m } , \mathsf { c o l l e c t } \} \cup \{ \mathsf { p i c k } _ { 1 } , \ldots , \mathsf { p i c k } _ { k } \} . } \end{array}\tag{16}
$$

Set

$$
T : = \gamma ^ { \ell + 1 } U , \qquad b : = ( 1 - \gamma ) T .\tag{17}
$$

The only nonzero rewards are

$$
r ( s _ { 0 } , \mathsf { w a i t } ) = b , \qquad r ( x _ { v , h } , \mathsf { c o l l e c t } ) = y _ { v , h } .
$$

All rewards are nonnegative and bounded by $R _ { \operatorname* { m a x } } : = \operatorname* { m a x } _ { v , h } y _ { v , h }$

The transition kernel is as follows.

1. At the root $s _ { 0 } .$

$$
P ( s _ { 0 } \mid s _ { 0 } , { \mathsf { w a i t } } ) = 1 .
$$

If $\ell = 2 .$ , action $\mathtt { g o }$ moves deterministically to $s _ { 1 } ; \mathrm { i f } \ell \ge 3$ , it moves deterministically to $d _ { 1 }$

2. At a delay state $d _ { i }$ , action advance moves deterministically to $d _ { i + 1 }$ when $i < \ell - 2$ , and from $d _ { \ell - 2 } \mathrm { t o } \ s _ { 1 }$

3. At the selector state $s _ { 1 }$ , each action pic $\iota _ { j } , j \in [ k ]$ , has the uniform transition law

$$
P ( s _ { v } \mid s _ { 1 } , \mathsf { p i c k } _ { j } ) = { \frac { 1 } { n _ { G } } } , \qquad v \in V .\tag{18}
$$

4. At a vertex state $s _ { v }$ , action claim samples the corresponding payoff state:

$$
P ( x _ { v , h } \mid s _ { v } , \mathsf { c l a i m } ) = p _ { v , h } , \qquad h \in [ N _ { v } ] .
$$

5. At a payoff state $x _ { v , h }$ , action collect moves deterministically to $s _ { T }$

6. The terminal state $s _ { T }$ is absorbing. Every action not explicitly specified above moves deterministically to $s _ { T }$ and gives reward zero.

Candidate tuples and the value of committing. We call root augmented state any augmented state of the form $( s _ { 0 } , c )$ with $c = ( \theta _ { 0 } , \ldots , \theta _ { \ell - 1 } ) \in \Omega ^ { \ell }$ . For $j \in [ k ]$ , define the length-ℓ action sequence

$$
\boldsymbol { \sigma } _ { j } : = ( \mathsf { g o } , \underbrace { \mathsf { a d v a n c e } , \ldots , \mathsf { a d v a n c e } } _ { \ell - 2 \mathrm { ~ t i m e s } } , \mathsf { p i c k } _ { j } ) ,\tag{19}
$$

with the obvious interpretation $( \mathsf { g o } , \mathsf { p i c k } _ { j } )$ when $\ell = 2$ . The first $\ell - 1$ actions of $\sigma _ { j }$ deterministically drive the system from $s _ { 0 }$ to $s _ { 1 }$ , so there is a unique $q _ { j } ( c ) \in V$ such that

$$
\phi _ { c } \bigl ( s _ { 0 } , \sigma _ { j } \bigr ) = s _ { q _ { j } ( c ) } , \qquad \mathrm { n a m e l y } \ s _ { q _ { j } ( c ) } = \theta _ { \ell - 1 } \bigl ( s _ { 1 } , \mathrm { p i c k } _ { j } \bigr ) .
$$

Write

$$
q ( c ) : = ( q _ { 1 } ( c ) , \ldots , q _ { k } ( c ) ) \in V ^ { k } , \qquad S ( c ) : = \{ q _ { 1 } ( c ) , \ldots , q _ { k } ( c ) \} .\tag{20}
$$

When $C \sim \mathsf { Q } ^ { \otimes \ell }$ , the coordinates of $q ( C )$ are independent and uniform on $V ,$ , because they arise from the distinct state–action pairs $( s _ { 1 } , \mathsf { p i c k } _ { j } )$ of a single product-distributed table, via equation 18.

Lemma 2 (Commit value and root recursion). For every root augmented state $( s _ { 0 } , c )$ , the value obtained by choosing go at $( s _ { 0 } , c )$ and then acting optimally is

$$
C ( c ) = \gamma ^ { \ell + 1 } \mathbb { E } \left[ \operatorname* { m a x } _ { v \in S ( c ) } Y _ { v } \right] .\tag{21}
$$

Moreover,

$$
V _ { \ell } ^ { \star } ( s _ { 0 } , c ) = \operatorname* { m a x } \left\{ C ( c ) , \ : b + \gamma \mathbb { E } [ V _ { \ell } ^ { \star } ( s _ { 0 } , C ^ { \prime } ) \mid c , \mathrm { w a i t } ] \right\} ,\tag{22}
$$

where $C ^ { \prime } = ( \theta _ { 1 } , \dots , \theta _ { \ell - 1 } , \Theta )$ with $\Theta \sim \mathsf Q$ independent of the past. Finally, after playing wait, the next candidate tuple $q ( C ^ { \prime } )$ is an independentfresh drawfrom the uniform distribution on $\dot { V } ^ { k }$

Proof. After $\mathsf { g o , }$ the process traverses the deterministic delay chain and cannot return to $s _ { 0 }$ . There are no rewards on this part of the trajectory. The system reaches $s _ { 1 }$ exactly $\ell - 1$ steps after leaving $s _ { 0 }$ . At that time, the current ℓ-step look-ahead window contains, for every $j \in [ k ]$ , both

$$
s _ { q _ { j } ( c ) } \quad \mathrm { a n d } \quad x _ { q _ { j } ( c ) , h _ { j } }
$$

along the continuation $\big ( \mathsf { p i c k } _ { j } , \mathsf { c l a i m } \big )$ , since $\ell \geq 2 .$ . Hence the realization of $Y _ { q _ { j } ( c ) }$ is known before the action ${ \mathsf { p i c k } } _ { j }$ is selected. Note also that, by prefix consistency of the look-ahead windows, the vertex reached under pick at that time is exactly $q _ { j } ( c )$ : the transition $( s _ { 1 } , \mathsf { p i c k } _ { j } )$ is governed by the last table of the root window, which is also the table governing the system when it actually reaches $s _ { 1 }$ . For distinct vertices, these payoff samples are independent; if the same vertex appears several times in $q ( c )$ , prefix consistency makes the corresponding branches share the same sample from the pair $( s _ { v } ,$ , claim). Therefore the selector obtains the largest realized value among the distinct variables indexed by $S ( \dot { c } )$

Moreover, the payoff realizations are not observable at the root: the payoff states $x _ { v , h }$ lie at depth $\ell { + 1 }$ from $s _ { 0 }$ under $( \sigma _ { j } , \mathsf { c l a i m } )$ , one level beyond the depth-ℓ tree revealed by $c .$ . Hence, conditionally on the root window, the variables $( Y _ { v } ) _ { v \in S ( c ) }$ have their unconditional product law.

Starting from $s _ { 1 }$ , the reward is collected two steps later: one transition under $\mathsf { p i c k } _ { j }$ , one under claim, and then the immediate reward of collect at the payoff state. The conditional value at $s _ { 1 }$ is thus

$$
\gamma ^ { 2 } \operatorname* { m a x } _ { v \in S ( c ) } Y _ { v } .
$$

The selector is reached $\ell - 1$ steps after leaving $s _ { 0 }$ , so the value at the root is

$$
\gamma ^ { \ell - 1 } \mathbb { E } \bigg [ \gamma ^ { 2 } \operatorname* { m a x } _ { v \in S ( c ) } Y _ { v } \bigg ] = \gamma ^ { \ell + 1 } \mathbb { E } \bigg [ \operatorname* { m a x } _ { v \in S ( c ) } Y _ { v } \bigg ] ,
$$

which proves equation 21.

At $s _ { 0 } ,$ the only potentially optimal actions are go and wait, since every other action leads to the absorbing state $s _ { T }$ with zero reward. The former has value $C ( c )$ . The latter gives immediate reward $b ,$ returns to $s _ { 0 } ,$ , and then has continuation value $\gamma \mathbb { E } [ V _ { \ell } ^ { \star } ( s _ { 0 } , \dot { C } ^ { \prime } ) \mid ($ , wait]. This proves equation 22.

Finally, after wait, the candidate transitions lie one level beyond the old look-ahead tree: $q ( C ^ { \prime } )$ is determined by the entries $( s _ { 1 } , \mathsf { p i c k } _ { j } )$ of the newly revealed table Θ, which is independent of all previously observed tables. The augmented transition kernel therefore samples them independently from equation 18, independently of the previous tuple. Hence $q ( C ^ { \prime } )$ is a fresh uniform draw from $V ^ { k }$ . In particular, the current window never reveals the next candidate tuple before wait is played. □

Proof of Theorem 1. We show that the map $( G , k ) \mapsto ( \mathcal { M } _ { G } , s _ { 0 } , \gamma , \theta )$ , with θ defined in equation 30 below, is a valid many-one reduction from INDEPENDENT SET to $\ell { \ - } \operatorname { D V D P }$

Soundness. Assume that G is a NO instance of INDEPENDENT SET. Fix a root window $c \in \Omega ^ { \ell }$ The set $S ( c )$ may contain fewer than k vertices because the tuple can have repetitions. Extend it to any set $\widetilde { S } \supseteq S ( c )$ of cardinality k. Since the expected maximum is monotone under set inclusion (the $Y _ { v }$ are nonnegative), equation 13 gives

$$
\mathbb { E } \left[ \operatorname* { m a x } _ { v \in S ( c ) } Y _ { v } \right] \leq \mathbb { E } \left[ \operatorname* { m a x } _ { v \in \widetilde { S } } Y _ { v } \right] \leq U .\tag{23}
$$

By equation 21 and equation 17,

$$
C ( c ) \leq \gamma ^ { \ell + 1 } U = T .\tag{24}
$$

Let

$$
\begin{array} { r } { M : = \operatorname* { m a x } \{ V _ { \ell } ^ { \star } ( s _ { 0 } , c ) : c \in \Omega ^ { \ell } \} . } \end{array}
$$

The set $\Omega ^ { \ell }$ is finite, so the maximum exists. From Lemma 2 and equation 24,

$$
M \leq \operatorname* { m a x } \{ T , \ b + \gamma M \} .
$$

If $M > T$ , then necessarily

$$
M \leq b + \gamma M = ( 1 - \gamma ) T + \gamma M ,
$$

which implies $( 1 - \gamma ) M \leq ( 1 - \gamma ) T$ , contradicting $M > T$ . Therefore

$$
M \leq T .\tag{25}
$$

In particular, after averaging over the initial root window,

$$
v _ { \ell , \gamma } ^ { \star } ( s _ { 0 } ) \leq T .\tag{26}
$$

Completeness. Assume that G is a YES instance, and let

$$
S ^ { \star } = \{ v _ { 1 } , \ldots , v _ { k } \}
$$

be an independent set satisfying equation 14. Consider the following stationary policy on the augmented state space:

• at a root augmented state $( s _ { 0 } , c )$ , choose go if

$$
q ( c ) = ( v _ { 1 } , \ldots , v _ { k } ) ,
$$

and choose wait otherwise;

• traverse the deterministic delay chain using advance;

• at $s _ { 1 } .$ , choose an index whose revealed payoff is maximal;

• use claim and then collect.

This policy is well defined: the tuple $q ( c )$ is observable at the root, and the realized payoffs are observable at $s _ { 1 } , \mathbf { b y }$ Lemma 2.

At every visit to $s _ { 0 }$ , the target tuple appears with probability

$$
\rho : = n _ { G } ^ { - k } .\tag{27}
$$

By the last part of Lemma 2, successive tuples are independent. Let $\tau \in \{ 0 , 1 , 2 , \ldots \}$ be the number of waiting actions before the target tuple first appears. Then

$$
\mathbb { P } ( \tau = t ) = ( 1 - \rho ) ^ { t } \rho ,
$$

and therefore

$$
\alpha : = \mathbb { E } [ \gamma ^ { \tau } ] = \sum _ { t = 0 } ^ { \infty } \rho ( 1 - \rho ) ^ { t } \gamma ^ { t } = \frac { \rho } { 1 - \gamma ( 1 - \rho ) } .\tag{28}
$$

The discounted return of this policy is

$$
\sum _ { t = 0 } ^ { \tau - 1 } \gamma ^ { t } b + \gamma ^ { \tau + \ell + 1 } \operatorname* { m a x } _ { v \in S ^ { \star } } Y _ { v } .
$$

The payoff samples generated after commitment are independent of the waiting time (they are drawn from table entries never used to define the tuples) and have the laws $( Y _ { v } ) _ { v \in S ^ { \star } }$ , with all vertices of the target tuple distinct. Taking expectations and using $b / ( 1 - \gamma ) = T$ gives

$$
\begin{array} { r l } & { v _ { \ell , \gamma } ^ { \pi } ( s _ { 0 } ) = T \big ( 1 - \mathbb { E } [ \gamma ^ { \tau } ] \big ) + \mathbb { E } [ \gamma ^ { \tau } ] \gamma ^ { \ell + 1 } \mathbb { E } \bigg [ \underset { v \in S ^ { \star } } { \operatorname* { m a x } } Y _ { v } \bigg ] } \\ & { ~ = T + \alpha \bigg ( \gamma ^ { \ell + 1 } \mathbb { E } \bigg [ \underset { v \in S ^ { \star } } { \operatorname* { m a x } } Y _ { v } \bigg ] - T \bigg ) } \\ & { ~ \geq T + \alpha \gamma ^ { \ell + 1 } \Delta , } \end{array}\tag{29}
$$

where the last inequality follows from equation 14 and $T = \gamma ^ { \ell + 1 } U$

Decision threshold and polynomial encoding. Define

$$
\theta : = T + \frac { 1 } { 2 } \alpha \gamma ^ { \ell + 1 } \Delta .\tag{30}
$$

Because $\gamma > 0 , \rho > 0$ , and $\Delta > 0$ , the added term is strictly positive. Equations equation 26 and equation 29 imply

$$
G { \mathrm { ~ i s ~ a ~ N O ~ i n s t a n c e } } \quad \Longrightarrow \quad v _ { \ell , \gamma } ^ { \star } ( s _ { 0 } ) \leq T < \theta ,
$$

$$
G \mathrm { \ i s \ a \ Y E S \ i n s t a n c e } \quad \Longrightarrow \quad v _ { \ell , \gamma } ^ { \star } ( s _ { 0 } ) \geq T + \alpha \gamma ^ { \ell + 1 } \Delta > \theta .
$$

Thus $( G , k ) \mapsto ( \mathcal { M } _ { G } , s _ { 0 } , \gamma , \theta )$ is a valid many-one reduction to ℓ-DVDP.

It remains to verify polynomial size. The stochastic gadget in Lemma 1 has polynomially many support points and polynomial-bit rational values and probabilities. The MDP has

$$
O \left( n _ { G } + \sum _ { v \in V } N _ { v } + \ell \right)
$$

states and $k + 5$ actions. Since ℓ is fixed, this is polynomial. The shift $L ,$ the quantities $U , T , b , \Delta$ and the threshold $\theta$ are obtained by a constant number of rational arithmetic operations. Moreover, $\rho = n _ { G } ^ { - k }$ has encoding length $O ( k \log n _ { G } )$ , and equation 28 therefore has polynomial encoding length. For fixed rational $\gamma ,$ , its encoding length is constant; $\operatorname { i f } \gamma$ is supplied in binary, all constructed numbers still have encoding length polynomial in the combined input size. This completes the reduction and the proof. □

## A.2 PROOF OF THEOREM 2

$$
2 \colon \Omega = { \mathcal { S } } ^ { S \times { \mathcal { A } } } ,
$$

$$
\Omega , \mathcal { T } _ { \ell }
$$

$$
\boldsymbol { s } \times \Omega ^ { \ell }
$$

$$
V _ { \ell } ^ { \star } , Q _ { \ell } ^ { \star }
$$

$$
n = | S | , m = | A |
$$

$$
\begin{array} { r } { \dot { V } _ { \mathrm { m a x } } : = \frac { R _ { \mathrm { m a x } } } { 1 - \gamma } } \end{array}
$$

$$
[ 0 , V _ { \mathrm { m a x } } ]
$$

$$
\theta _ { k : k ^ { \prime } } = ( \theta _ { k } , \ldots , \theta _ { k ^ { \prime } } )
$$

We begin by defining the ideal empirical model. This model replaces the unknown expectation under Q with an average over a sampled dictionary, while retaining the same augmented state space as the true Bellman equation. Sample once and for all, independently,

$$
\Theta ^ { 1 } , \ldots , \Theta ^ { N } \stackrel { \mathrm { i . i . d . } } { \sim } \mathsf Q ,\tag{31}
$$

and set the empirical distribution

$$
\widehat { \mathsf Q } _ { N } : = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \delta _ { \Theta ^ { j } } .\tag{32}
$$

The associated empirical Bellman operator is

$$
( \widehat { \mathcal { T } } _ { \ell } V ) ( s , \theta _ { 0 } , \ldots , \theta _ { \ell - 1 } ) : = \operatorname* { m a x } _ { a \in \mathcal { A } } \Bigl \{ r ( s , a ) + \frac { \gamma } { N } \sum _ { j = 1 } ^ { N } V \bigl ( \theta _ { 0 } ( s , a ) , \theta _ { 1 } , \ldots , \theta _ { \ell - 1 } , \Theta ^ { j } \bigr ) \Bigr \} .\tag{33}
$$

It is a $\gamma \cdot$ -contraction for the sup norm on $s \times \Omega ^ { \ell } ;$ we denote by $\widehat { V } ^ { \star }$ its unique fixed point, and define the empirical action scores

$$
Q _ { \ell } ^ { \star } ( s , \theta _ { 0 : \ell - 1 } , a ) = r ( s , a ) + \gamma \mathbb { E } _ { \Theta \sim \mathbb { Q } } \bigl [ V _ { \ell } ^ { \star } \bigl ( \theta _ { 0 } ( s , a ) , \theta _ { 1 : \ell - 1 } , \Theta \bigr ) \bigr ] ,\tag{34}
$$

$$
\widehat { Q } ^ { \star } ( s , \theta _ { 0 : \ell - 1 } , a ) : = r ( s , a ) + \frac { \gamma } { N } \sum _ { j = 1 } ^ { N } \widehat { V } ^ { \star } \big ( \theta _ { 0 } ( s , a ) , \theta _ { 1 : \ell - 1 } , \Theta ^ { j } \big ) ,\tag{35}
$$

where equation 34 simply restates the definition of $Q _ { \ell } ^ { \star }$ from Section 2. We now compare the ideal empirical model with the true model. Consider the function class

$$
\mathcal { F } : = \big \{ \theta \mapsto V _ { \ell } ^ { \star } ( u , \theta _ { 1 : \ell - 1 } , \theta ) : u \in \mathcal { S } , \theta _ { 1 : \ell - 1 } \in \Omega ^ { \ell - 1 } \big \} ,\tag{36}
$$

whose cardinality is at most $| { \mathcal { F } } | \leq n | \Omega | ^ { \ell - 1 } = n ^ { 1 + ( \ell - 1 ) n m }$ , crucially, log $| \mathcal F |$ is therefore polynomial in $n _ { \mathrm { { ; } } }$ , m for fixed $\ell .$

Lemma 3 (Uniform concentration). For every $\eta > 0$

$$
\begin{array} { r l } & { \mathbb { P } \left( \underset { u , \theta _ { 1 : \ell - 1 } } { \operatorname* { s u p } } \left| \frac { 1 } { N } \sum _ { j = 1 } ^ { N } V _ { \ell } ^ { \star } ( u , \theta _ { 1 : \ell - 1 } , \Theta ^ { j } ) - \mathbb { E } _ { \Theta \sim \mathbb { Q } } \left[ V _ { \ell } ^ { \star } ( u , \theta _ { 1 : \ell - 1 } , \Theta ) \right] \right| > \eta \right) } \\ & { \leq 2 n | \Omega | ^ { \ell - 1 } \exp \left( - \frac { 2 N \eta ^ { 2 } } { V _ { \mathrm { m a x } } ^ { 2 } } \right) . } \end{array}\tag{37}
$$

Proof. The function $V _ { \ell } ^ { \star }$ is deterministic and does not depend on the sampled dictionary, so each function in $\mathcal { F }$ is fixed and takes values in $[ 0 , V _ { \mathrm { m a x } } ]$ . We apply Hoeffding’s inequality to each function of $\mathcal { F }$ and then take a union bound. □

Denote the event

$$
\mathcal { E } _ { \eta } : = \left\{ \operatorname* { s u p } _ { u , \theta _ { 1 : \ell - 1 } } \left| \frac { 1 } { N } \sum _ { j = 1 } ^ { N } V _ { \ell } ^ { \star } ( u , \theta _ { 1 : \ell - 1 } , \Theta ^ { j } ) - \mathbb { E } _ { \Theta \sim \Theta } \left[ V _ { \ell } ^ { \star } ( u , \theta _ { 1 : \ell - 1 } , \Theta ) \right] \right| \leq \eta \right\} .
$$

Lemma 4 (Fixed-point stability). Under the event $\mathcal { E } _ { \eta } ^ { \mathrm { ~ ~ } } ;$

$$
\| \widehat { V } ^ { \star } - V _ { \ell } ^ { \star } \| _ { \infty } \leq \frac { \gamma \eta } { 1 - \gamma } .\tag{38}
$$

Proof. Fix $\left( s , \theta _ { 0 : \ell - 1 } \right)$ . For every $a \in A .$ , the bracketed terms in $\widehat { \tau _ { \ell } } V _ { \ell } ^ { \star }$ and $\mathcal { T } _ { \ell } V _ { \ell } ^ { \star }$ share the reward $r ( s , a )$ and differ by γ times an empirical-versus-true average of $V _ { \ell } ^ { \star } ( \tilde { \theta } _ { 0 } ( s , a ) , \theta _ { 1 : \ell - 1 } ^ { \setminus } , \cdot )$ , which is at most $\gamma \eta$ in absolute value on $\mathcal { E } _ { \eta }$ . Since the maximum over a is 1-Lipschitz,

$$
\| \widehat { T } _ { \ell } V _ { \ell } ^ { \star } - \mathcal { T } _ { \ell } V _ { \ell } ^ { \star } \| _ { \infty } \leq \gamma \eta .\tag{39}
$$

As $V _ { \ell } ^ { \star } = \mathcal { T } _ { \ell } V _ { \ell } ^ { \star } , \widehat { V } ^ { \star } = \widehat { \mathcal { T } _ { \ell } } \widehat { V } ^ { \star }$ , and $\widehat { \mathcal { T } } _ { \ell }$ is a γ-contraction,

$$
\begin{array} { r l } & { \| \widehat { V } ^ { \star } - V _ { \ell } ^ { \star } \| _ { \infty } \leq \| \widehat { \mathcal { T } } _ { \ell } \widehat { V } ^ { \star } - \widehat { \mathcal { T } } _ { \ell } V _ { \ell } ^ { \star } \| _ { \infty } + \| \widehat { \mathcal { T } } _ { \ell } V _ { \ell } ^ { \star } - \mathcal { T } _ { \ell } V _ { \ell } ^ { \star } \| _ { \infty } } \\ & { \qquad \leq \gamma \| \widehat { V } ^ { \star } - V _ { \ell } ^ { \star } \| _ { \infty } + \gamma \eta , } \end{array}
$$

and rearranging proves the claim.

Lemma 5. On $\mathcal { E } _ { \eta } ,$

$$
\operatorname* { s u p } _ { s , \theta _ { 0 } , \varepsilon _ { - 1 } , a } | \widehat { Q } ^ { \star } ( s , \theta _ { 0 : \ell - 1 } , a ) - Q _ { \ell } ^ { \star } ( s , \theta _ { 0 : \ell - 1 } , a ) | \leq \frac { \gamma \eta } { 1 - \gamma } .\tag{40}
$$

Proof. By adding and subtracting the empirical average of $V _ { \ell } ^ { \star }$ ,

$$
\begin{array} { r l } & { | \widehat { Q } ^ { \star } ( s , \theta _ { 0 : \ell - 1 } , a ) - Q _ { \ell } ^ { \star } ( s , \theta _ { 0 : \ell - 1 } , a ) | } \\ & { \ \leq \displaystyle \frac { \gamma } { N } \sum _ { j = 1 } ^ { N } \left| \widehat { V } ^ { \star } \big ( \theta _ { 0 } ( s , a ) , \theta _ { 1 : \ell - 1 } , \Theta ^ { j } \big ) - V _ { \ell } ^ { \star } \big ( \theta _ { 0 } ( s , a ) , \theta _ { 1 : \ell - 1 } , \Theta ^ { j } \big ) \right| + \gamma \eta } \\ & { \ \leq \gamma \| \widehat { V } ^ { \star } - V _ { \ell } ^ { \star } \| _ { \infty } + \gamma \eta } \\ & { \ \leq \displaystyle \frac { \gamma \eta } { 1 - \gamma } , } \end{array}
$$

where the last step uses Lemma 4 and the identity $\begin{array} { r } { \gamma \frac { \gamma \eta } { 1 - \gamma } + \gamma \eta = \frac { \gamma \eta } { 1 - \gamma } } \end{array}$

The previous paragraph shows that $\widehat { V } ^ { \star }$ is close to $V _ { \ell } ^ { \star }$ , but $\widehat { V } ^ { \star }$ is still formally defined on $s \times \Omega ^ { \ell }$ We now show that windows composed only of tables from the dictionary equation 31 form a finite state space that is preserved by the empirical Bellman operator. Set

$$
[ N ] : = \{ 1 , \ldots , N \} , \qquad \mathcal { D } _ { N } : = \mathcal { S } \times [ N ] ^ { \ell } .\tag{41}
$$

For each $i \in [ N ]$ , the index i refers to the sampled transition table $\Theta ^ { i }$ of the dictionary. Consequently, an index tuple $\mathbf { i _ { \theta } } = ( i _ { 0 } , \dots , i _ { \ell - 1 } ) \in \mathsf { \Gamma } [ N ] ^ { \ell }$ encodes the window of transition tables $( \Theta ^ { \dot { i } _ { 0 } } , \dots , \Theta ^ { i _ { \ell - 1 } } ) \in \Omega ^ { \ell }$ ; the index $i _ { k }$ specifies the dictionary table used at temporal level k of the look-ahead window. Define, on functions $V : \mathcal { D } _ { N } \to \mathbb { R }$

$$
( \widehat { T } _ { \mathcal { D } _ { N } } V ) ( s , i _ { 0 } , \ldots , i _ { \ell - 1 } ) : = \operatorname* { m a x } _ { a \in \mathcal { A } } \left\{ r ( s , a ) + \frac { \gamma } { N } \sum _ { j = 1 } ^ { N } V \big ( \Theta ^ { i _ { 0 } } ( s , a ) , i _ { 1 } , \ldots , i _ { \ell - 1 } , j \big ) \right\} .\tag{42}
$$

This operator is a γ-contraction; let $\widehat { V } _ { \mathcal { D } _ { N } } ^ { \star }$ denote its unique fixed point.

Lemma 6 (Exact restriction to the dictionary). For every $( s , i _ { 0 } , \ldots , i _ { \ell - 1 } ) \in \mathcal { D } _ { N }$

$$
\widehat { V } _ { \mathcal { D } _ { N } } ^ { \star } ( s , \Theta ^ { i _ { 0 } } , \ldots , \Theta ^ { i _ { \ell - 1 } } ) = \widehat { V } ^ { \star } ( s , \Theta ^ { i _ { 0 } } , \ldots , \Theta ^ { i _ { \ell - 1 } } ) .\tag{43}
$$

Proof. For a function $V : \mathcal { S } \times \Omega ^ { \ell } \to \mathbb { R }$ , denote $R _ { { D } _ { N } } V$ its restriction to the dictionary $\mathcal { D } _ { N }$ Let $( s , i _ { 0 } , \ldots , i _ { \ell - 1 } ) \in \mathcal { D } _ { N }$

$$
\begin{array} { l } { { ( R _ { \mathcal { D } _ { N } } \widehat { T } _ { \ell } V ) ( s , \Theta ^ { i _ { 0 } } , \ldots , \Theta ^ { i _ { \ell - 1 } } ) = ( \widehat { T } _ { \ell } V ) ( s , \Theta ^ { i _ { 0 } } , \ldots , \Theta ^ { i _ { \ell - 1 } } ) } } \\ { { \displaystyle \qquad = \operatorname* { m a x } _ { a \in \mathcal { A } } \left\{ r ( s , a ) + \frac { \gamma } { N } \sum _ { j = 1 } ^ { N } V \big ( \Theta ^ { i _ { 0 } } ( s , a ) , \Theta ^ { i _ { 1 } } , \ldots , \Theta ^ { i _ { \ell - 1 } } , \Theta ^ { j } \big ) \right\} } } \\ { { \displaystyle \qquad = \operatorname* { m a x } _ { a \in \mathcal { A } } \left\{ r ( s , a ) + \frac { \gamma } { N } \sum _ { j = 1 } ^ { N } ( R _ { \mathcal { D } _ { N } } V ) \big ( \Theta ^ { i _ { 0 } } ( s , a ) , \Theta ^ { i _ { 1 } } , \ldots , \Theta ^ { i _ { \ell - 1 } } , \Theta ^ { j } \big ) \right\} } } \\ { { \displaystyle \qquad = ( \widehat { T } _ { \mathcal { D } _ { N } } R _ { \mathcal { D } _ { N } } V ) \big ( s , \Theta ^ { i _ { 0 } } ( s , a ) , \Theta ^ { i _ { 1 } } , \ldots , \Theta ^ { i _ { \ell - 1 } } \big ) , } } \end{array}
$$

Now, since $\widehat { \mathcal { T } } _ { \ell } \widehat { V } ^ { \star } = \widehat { V } ^ { \star }$ , we obtain

$$
\begin{array} { r } { \widehat { T } _ { \mathcal { D } _ { N } } ( R _ { N } \widehat { V } ^ { \star } ) = R _ { N } ( \widehat { T _ { \ell } } \widehat { V } ^ { \star } ) = R _ { N } \widehat { V } ^ { \star } . } \end{array}
$$

Thus $R _ { N } \widehat { V } ^ { \star }$ is a fixed point of $\widehat { T } _ { { D } _ { N } }$ . Since $\widehat { T } _ { D _ { N } }$ is a γ-contraction, this fixed point is unique and therefore

$$
R _ { N } \widehat { V } ^ { \star } = \widehat { V } _ { D _ { N } } ^ { \star } .
$$

Which gives equation 43.

Remark 1 (Compute and memory cost of one interation.). For a function $v : \mathcal { D } _ { N } \to \mathbb { I }$ R and every tuple $( u , i _ { 1 } , \ldots , \dot { i } _ { \ell - 1 } ) \in \mathcal { S } \times [ N ] ^ { \check { \ell } - 1 }$ we compute

$$
\mathrm { M e a n } _ { v } ( u , i _ { 1 } , \ldots , i _ { \ell - 1 } ) = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } v ( u , i _ { 1 } , \ldots , i _ { \ell - 1 } , j ) .\tag{44}
$$

Computing all the Mea $\mathrm { n } _ { v }$ costs $O ( n N ^ { \ell } )$ ; then computing all the maxima costs $O ( n m N ^ { \ell } )$ . All in all, one iteration ofequation 42 costs

$$
O \big ( n ( m + 1 ) N ^ { \ell } \big ) .\tag{45}
$$

The operator $\widehat { T } _ { D _ { N } }$ is a γ-contraction on the finite dictionary $\mathcal { D } _ { N }$ and, therefore, can be solved by exact value iteration:

$$
V ^ { 0 } = 0 , \qquad V ^ { k + 1 } = \widehat { T } _ { \mathcal { D } _ { N } } V ^ { k } .\tag{46}
$$

$\widehat { V } _ { \mathcal { D } _ { N } } ^ { \star }$ can only be evaluated on windows whose tables all belong to the sampled dictionary $\mathcal { D } _ { N }$ However, the look-ahead window observed during deployment contains arbitrary tables, and therefore need not belong to the dictionary. To evaluate an arbitrary observed window $c = ( \theta _ { 0 } , \ldots , \theta _ { \ell - 1 } )$ we work backward from $\widehat { V } _ { \mathcal { D } _ { N } } ^ { \star }$ defined on dictionary windows, then, the recursion successively reintroduces the observed tables $\theta _ { \ell - 1 } , \ldots , \theta _ { 0 }$ . We show below that, when initialized with the exact dictionary fixed point $v ^ { \star }$ , this procedure returns exactly $\widehat { V } ^ { \star } ( s , c )$

Formally, let $c = ( \theta _ { 0 } , \ldots , \theta _ { \ell - 1 } ) \in \Omega ^ { \ell }$ be an arbitrary window. Define the functions $U _ { k , c } ^ { v }$ by backward recursion. At the terminal level,

$$
U _ { \ell , c } ( s , i _ { 0 } , \ldots , i _ { \ell - 1 } ) : = \widehat { V } _ { \mathcal { D } _ { N } } ^ { \star } ( s , i _ { 0 } , \ldots , i _ { \ell - 1 } ) .\tag{47}
$$

For $k = \ell - 1 , \ldots , 1$

$$
U _ { k , c } ( \boldsymbol { s } , i _ { 0 } , \ldots , i _ { k - 1 } ) : = \operatorname* { m a x } _ { a \in \mathcal { A } } \left\{ r ( \boldsymbol { s } , a ) + \frac { \gamma } { N } \sum _ { i _ { k } = 1 } ^ { N } U _ { k + 1 , c } ( \theta _ { k } ( \boldsymbol { s } , a ) , i _ { 0 } , \ldots , i _ { k } ) \right\} .\tag{48}
$$

Then, for every root action, define

$$
\widetilde { Q } ( s , \theta _ { 0 : \ell - 1 } , a ) : = r ( s , a ) + \frac { \gamma } { N } \sum _ { i _ { 0 } = 1 } ^ { N } U _ { 1 , c } \big ( \theta _ { 0 } ( s , a ) , i _ { 0 } \big ) ,\tag{49}
$$

so that $\begin{array} { r } { U _ { 0 , c } ( s ) : = \operatorname* { m a x } _ { a \in \mathcal { A } } \left\{ \widetilde { Q } ( s , \theta _ { 0 : \ell - 1 } , a ) \right\} } \end{array}$

Lemma 7 (Exact extension identity). For every $( s , \theta _ { 0 : \ell - 1 } , a ) \in \mathcal { S } \times \Omega ^ { \ell } \times \mathcal { A } ,$

$$
\widetilde { Q } _ { ( } s , \theta _ { 0 : \ell - 1 } , a ) = \widehat { Q } ^ { \star } ( s , \theta _ { 0 : \ell - 1 } , a ) .\tag{50}
$$

Proof. Fix an arbitrary window $c = ( \theta _ { 0 } , \ldots , \theta _ { \ell - 1 } ) \in \Omega ^ { \ell }$ . We first establish the intermediate identity

$$
U _ { k , c } ^ { v ^ { \star } } ( u , i _ { 0 } , \ldots , i _ { k - 1 } ) = \widehat { V } ^ { \star } \big ( u , \theta _ { k } , \ldots , \theta _ { \ell - 1 } , \Theta ^ { i _ { 0 } } , \ldots , \Theta ^ { i _ { k - 1 } } \big )\tag{51}
$$

for every $k \in \{ 1 , \ldots , \ell \}$ , by backward induction on k.

For $k = \ell _ { \mathrm { : } }$ , the terminal condition equation 47 and Lemma 6 give

$$
\begin{array} { r l } & { U _ { \ell , c } ( u , i _ { 0 } , \ldots , i _ { \ell - 1 } ) = \widehat { V } _ { \mathcal { D } _ { N } } ^ { \star } ( u , i _ { 0 } , \ldots , i _ { \ell - 1 } ) } \\ & { \phantom { { = } } = \widehat { V } ^ { \star } \big ( u , \Theta ^ { i _ { 0 } } , \ldots , \Theta ^ { i _ { \ell - 1 } } \big ) , } \end{array}
$$

which is precisely equation 51 at level ℓ. Now suppose that equation 51 holds at level $k + 1$ , for some $k \in \{ 1 , \ldots , \ell - 1 \}$ }. By equation 48 and the induction hypothesis,

$$
\begin{array} { r l } & { \widehat { U } _ { k , c } ( u , i _ { 0 } , \ldots , i _ { k - 1 } ) } \\ & { = \displaystyle \operatorname* { m a x } _ { b \in \mathcal { A } } \Bigg \{ r ( u , b ) + \frac { \gamma } { N } \sum _ { i _ { k } = 1 } ^ { N } U _ { k + 1 , c } \big ( \theta _ { k } ( u , b ) , i _ { 0 } , \ldots , i _ { k } \big ) \Bigg \} } \\ & { = \displaystyle \operatorname* { m a x } _ { b \in \mathcal { A } } \Bigg \{ r ( u , b ) + \frac { \gamma } { N } \sum _ { i _ { k } = 1 } ^ { N } \widehat { V } ^ { \star } \big ( \theta _ { k } ( u , b ) , \theta _ { k + 1 } , \ldots , \theta _ { \ell - 1 } , \Theta ^ { i _ { 0 } } , \ldots , \Theta ^ { i _ { k } } \big ) \Bigg \} } \\ & { = ( \widehat { T } _ { \ell } \widehat { V } ^ { \star } ) \big ( u , \theta _ { k } , \ldots , \theta _ { \ell - 1 } , \Theta ^ { i _ { 0 } } , \ldots , \Theta ^ { i _ { k - 1 } } \big ) } \\ & { = \widehat { V } ^ { \star } \big ( u , b _ { k } , \ldots , \theta _ { \ell - 1 } , \Theta ^ { i _ { 0 } } , \ldots , \Theta ^ { i _ { k - 1 } } \big ) , } \end{array}
$$

where the last equality uses $\widehat { \mathcal { T } _ { \ell } } \widehat { V } ^ { \star } = \widehat { V } ^ { \star }$ . This proves equation 51 for every $k \in \{ 1 , \ldots , \ell \}$ . Finally, fix the root action a. Applying the identity at level $k = 1$ in the definition equation 49 yields

$$
\begin{array} { l } { \displaystyle \widetilde { Q } ( s , \theta _ { 0 : \ell - 1 } , a ) = r ( s , a ) + \frac { \gamma } { N } \sum _ { i _ { 0 } = 1 } ^ { N } U _ { 1 , c } \big ( \theta _ { 0 } ( s , a ) , i _ { 0 } \big ) } \\ { = r ( s , a ) + \frac { \gamma } { N } \sum _ { i _ { 0 } = 1 } ^ { N } \widehat { V } ^ { \star } \big ( \theta _ { 0 } ( s , a ) , \theta _ { 1 } , \dots , \theta _ { \ell - 1 } , \Theta ^ { i _ { 0 } } \big ) } \\ { = \widehat { Q } ^ { \star } ( s , \theta _ { 0 : \ell - 1 } , a ) , } \end{array}
$$

where the last equality is the definition equation 35.

Lemma 8 (Stability of the extension). For all arrays $v _ { 1 } , v _ { 2 } : \mathcal { C } _ { N } \to [ 0 , V _ { \mathrm { m a x } } ]$ and every $k \in$ $\{ 0 , \ldots , \ell \}$

$$
\operatorname* { s u p } _ { \substack { c , s , i _ { 0 } , \ldots , i _ { k - 1 } } } \left| U _ { k , c } ^ { v _ { 1 } } ( s , i _ { 0 } , \ldots , i _ { k - 1 } ) - U _ { k , c } ^ { v _ { 2 } } ( s , i _ { 0 } , \ldots , i _ { k - 1 } ) \right| \leq \gamma ^ { \ell - k } \| v _ { 1 } - v _ { 2 } \| _ { \infty } .\tag{52}
$$

Consequently,

$$
\operatorname* { s u p } _ { s , c } | U _ { 0 , c } ^ { v _ { 1 } } ( s ) - U _ { 0 , c } ^ { v _ { 2 } } ( s ) | \leq \gamma ^ { \ell } \Vert v _ { 1 } - v _ { 2 } \Vert _ { \infty } ,\tag{53}
$$

$$
\operatorname* { s u p } _ { s , c , a } | \widetilde { Q } _ { v _ { 1 } } ( s , c , a ) - \widetilde { Q } _ { v _ { 2 } } ( s , c , a ) | \leq \gamma ^ { \ell } \Vert v _ { 1 } - v _ { 2 } \Vert _ { \infty } .\tag{54}
$$

Proof. Backward induction on $k ,$ the window ${ \boldsymbol { c } } = ( \theta _ { 0 } , \dots , \theta _ { \ell - 1 } )$ being fixed throughout since equation 48 never modifies it.

For $k = \ell ,$ the terminal condition equation $^ { 4 7 }$ gives $U _ { \ell , c } ^ { v _ { 1 } } - U _ { \ell , c } ^ { v _ { 2 } } = v _ { 1 } - v _ { 2 }$ , whose arguments range over exactly $\mathcal { C } _ { N }$ , so equation 52 holds with equality.

Assume equation 52 at level $k + 1$ and fix $s , i _ { 0 } , \ldots , i _ { k - 1 }$ . In equation 48, the two braces associated with $v _ { 1 }$ and $v _ { 2 }$ share the term $r ( s , a )$ , so for every $a \in { \mathcal { A } }$ their difference is

$$
\frac { \gamma } { N } \sum _ { i _ { k } = 1 } ^ { N } \left[ U _ { k + 1 , c } ^ { v _ { 1 } } - U _ { k + 1 , c } ^ { v _ { 2 } } \right] \left( \theta _ { k } ( s , a ) , i _ { 0 } , \ldots , i _ { k } \right) ,
$$

whose absolute value is at most $\gamma ^ { \ell - k } \| v _ { 1 } - v _ { 2 } \| _ { \infty }$ by the induction hypothesis. Since maximizing over a is 1-Lipschitz, the same bound holds for $| \ddot { U } _ { k , c } ^ { v _ { 1 } } ( s , i _ { 0 } , \ldots , i _ { k - 1 } ) - \dot { U } _ { k , c } ^ { v _ { 2 } } ( s , i _ { 0 } , \ldots , i _ { k - 1 } ) |$ , uniformly in c.

The case $k = 0$ is equation 53. For equation 54, the term $r ( s , a )$ cancels likewise in equation 49, so $| \widetilde { Q } _ { v _ { 1 } } ( s , c , a ) - \widetilde { Q } _ { v _ { 2 } } ( s , c , a ) |$ equals $\gamma$ times an average of differences at level 1, hence is at most $\gamma \cdot \gamma ^ { \ell - \mathrm { \bar { 1 } } } \| v _ { 1 } - v _ { 2 } \| _ { \infty }$ □

Remark 2 (Deployment cost). Proceeding level by level as in equation 44, the cost of computing all action scoresfor one queried pair $( s , c )$ is

$$
O \left( \sum _ { k = 0 } ^ { \ell - 1 } ( n N ^ { k + 1 } + n m N ^ { k } ) \right) = O { \left( \ell n ( m + 1 ) N ^ { \ell } \right) } .\tag{55}
$$

Moreover, note that the terminal array $v ,$ of size $n N ^ { \ell } ,$ , is already stored. During an online query, the largest temporary array is $U _ { \ell - 1 , c } ^ { v } ,$ which contains $n N ^ { \ell - 1 }$ entries. Indeed, once $U _ { k , c } ^ { v }$ has been computedfrom $U _ { k + 1 , c } ^ { v } ,$ the array $U _ { k + 1 , c } ^ { v ^ { \prime } }$ is no longer needed and can be discarded. Thus, the online computation requires only $O ( n N ^ { \ell - 1 } )$ additional working memory.

Run Algorithm 1 with

$$
N = \left\lceil { \frac { 8 } { \varepsilon ^ { 2 } ( 1 - \gamma ) ^ { 4 } } } \left( \log { \frac { 2 } { \delta } } + \left( 1 + ( \ell - 1 ) n m \right) \log n \right) \right\rceil
$$

and

$$
K = \left\lceil { \frac { 1 } { 1 - \gamma } } \log { \frac { 4 } { \varepsilon ( 1 - \gamma ) } } \right\rceil .
$$

It returns $D = ( \Theta ^ { 1 } , \dots , \Theta ^ { N } , v ^ { K } )$ , and Algorithm 2 defines the policy $\pi _ { D }$

Apply Lemma 3 with

$$
\eta = \frac { \varepsilon ( 1 - \gamma ) ^ { 2 } V _ { \mathrm { m a x } } } { 4 } .
$$

The above choice of N ensures that the failure probability in equation 37 is at most $\delta .$ Therefore, with probability at least $1 - \delta$ , Lemma 5 gives

$$
\begin{array} { r l r } {  { \operatorname* { s u p } _ { s , c , a } \Big | \widehat { Q } ^ { \star } ( s , c , a ) - Q _ { \ell } ^ { \star } ( s , c , a ) \Big | \leq \frac { \gamma \eta } { 1 - \gamma } } } \\ & { } & { \leq \frac { \varepsilon ( 1 - \gamma ) V _ { \operatorname* { m a x } } } { 4 } . } \end{array}\tag{56}
$$

By the choice of $K ,$

$$
\begin{array} { r l } {  { \| v ^ { K } - v ^ { \star } \| _ { \infty } \leq \gamma ^ { K } V _ { \operatorname* { m a x } } } } \\ & { \leq e ^ { - ( 1 - \gamma ) K } V _ { \operatorname* { m a x } } } \\ & { \leq \frac { \varepsilon ( 1 - \gamma ) V _ { \operatorname* { m a x } } } { 4 } . } \end{array}
$$

Lemma 8 then implies

$$
\begin{array} { r } { \displaystyle \operatorname* { s u p } _ { s , c , a } \left| \widetilde { Q } _ { v ^ { \kappa } } ( s , c , a ) - \widehat { Q } ^ { \star } ( s , c , a ) \right| \leq \gamma ^ { \ell } \| v ^ { K } - v ^ { \star } \| _ { \infty } } \\ { \leq \frac { \varepsilon ( 1 - \gamma ) V _ { \operatorname* { m a x } } } { 4 } . } \end{array}\tag{57}
$$

Combining equation 56 and equation 57 yields

$$
\operatorname* { s u p } _ { s , c , a } \Big | \widetilde { Q } _ { v ^ { \kappa } } ( s , c , a ) - Q _ { \ell } ^ { \star } ( s , c , a ) \Big | \leq \frac { \varepsilon ( 1 - \gamma ) V _ { \operatorname* { m a x } } } { 2 } .\tag{58}
$$

Fix $( s , c )$ , let $a ^ { \star }$ maximize $Q _ { \ell } ^ { \star } ( s , c , \cdot )$ , and recall that $\pi _ { D } ( s , c )$ maximizes $\widetilde { Q } _ { v ^ { K } } ( s , c , \cdot )$ . It follows from equation 58 that

$$
\begin{array} { r l } { V _ { \ell } ^ { \star } ( s , c ) = Q _ { \ell } ^ { \star } ( s , c , a ^ { \star } ) } & { } \\ { \displaystyle } & { \leq \widetilde { Q } _ { v ^ { \star } } ( s , c , a ^ { \star } ) + \frac { \varepsilon ( 1 - \gamma ) V _ { \operatorname* { m a x } } } { 2 } } \\ { \displaystyle } & { \leq \widetilde { Q } _ { v ^ { \star } } ( s , c , \pi _ { D } ( s , c ) ) + \frac { \varepsilon ( 1 - \gamma ) V _ { \operatorname* { m a x } } } { 2 } } \\ { \displaystyle } & { \leq Q _ { \ell } ^ { \star } ( s , c , \pi _ { D } ( s , c ) ) + \varepsilon ( 1 - \gamma ) V _ { \operatorname* { m a x } } . } \end{array}
$$

Let $\mathcal { T } _ { \ell } ^ { \pi _ { D } }$ denote the Bellman operator of $\pi _ { D }$ in the true augmented MDP. The preceding inequality is equivalent to

$$
V _ { \ell } ^ { \star } \le \mathcal { T } _ { \ell } ^ { \pi _ { D } } V _ { \ell } ^ { \star } + \varepsilon ( 1 - \gamma ) V _ { \mathrm { m a x } } \mathbf { 1 } .
$$

Since $V _ { \ell } ^ { \pi _ { D } } = T _ { \ell } ^ { \pi _ { D } } V _ { \ell } ^ { \pi _ { D } }$ and $\mathcal { T } _ { \ell } ^ { \pi _ { D } }$ is a γ-contraction,

$$
\begin{array} { r } { \| V _ { \ell } ^ { \star } - V _ { \ell } ^ { \pi _ { D } } \| _ { \infty } \leq \gamma \| V _ { \ell } ^ { \star } - V _ { \ell } ^ { \pi _ { D } } \| _ { \infty } + \varepsilon ( 1 - \gamma ) V _ { \mathrm { m a x } } . } \end{array}
$$

Rearranging gives

$$
\begin{array} { r } { \| V _ { \ell } ^ { \star } - V _ { \ell } ^ { \pi _ { D } } \| _ { \infty } \leq \varepsilon V _ { \mathrm { m a x } } , } \end{array}
$$

which proves equation 7 simultaneously for every $( s , c ) \in \mathcal { S } \times \Omega ^ { \ell }$

It remains to verify the computational guarantees. Sampling the N dictionary tables costs $O ( N n m )$ and the $K$ value-iteration steps cost

$$
O \big ( K n ( m + 1 ) N ^ { \ell } \big ) .
$$

Hence the total preprocessing time is

$$
O \big ( N n m + K n ( m + 1 ) N ^ { \ell } \big ) .
$$

The policy representation $D = ( \Theta ^ { 1 } , \dots , \Theta ^ { N } , v ^ { K } )$ occupies

$$
O \big ( N n m + n N ^ { \ell } \big )
$$

memory locations.

For a given state s and observed window $c ,$ computing all the scores $\widetilde { Q } _ { v ^ { K } } ( s , c , a )$ costs

$$
O ( \ell n ( m + 1 ) N ^ { \ell } ) .
$$

The observed window occupies $O ( \ell n m )$ memory locations. Moreover, the arrays used during this computation can be discarded level by level, so the temporary memory does not exceed $O ( n N ^ { \bar { \ell } - 1 } )$ . The total memory used during online action selection is therefore

$$
O \big ( N n m + n N ^ { \ell } + \ell n m \big ) .
$$

Finally,

$$
N = O \left( { \frac { \log ( 1 / \delta ) + \left( 1 + ( \ell - 1 ) n m \right) \log n } { \varepsilon ^ { 2 } ( 1 - \gamma ) ^ { 4 } } } \right) , \qquad K = O \left( { \frac { \log ( 1 / [ \varepsilon ( 1 - \gamma ) ] ) } { 1 - \gamma } } \right) .
$$

Thus, for every fixed $\ell ,$ preprocessing time, representation size, and per-decision computation are polynomial in $n , m , \varepsilon ^ { - 1 } , \log ( 1 / \delta )$ , and $( 1 - \gamma ) ^ { - 1 }$ . This completes the proof. □

## A.3 PROOF OF THEOREM 3

We prove the regret bound for DLA-UCB. After N transition tables have expired, the learner has observed N independent samples from every row $\textstyle P ( \cdot \mid s , a )$ . It therefore uses the usual empirical transition kernel and reconstructs the corresponding product law:

$$
\begin{array} { c } { { \displaystyle \widehat { P } _ { N } ( s ^ { \prime } \mid s , a ) : = \displaystyle \frac { 1 } { N } \sum _ { j = 0 } ^ { N - 1 } \mathbf { 1 } \{ \Theta _ { j } ( s , a ) = s ^ { \prime } \} , } } \\ { { \displaystyle \widehat { \mathbb { Q } } _ { N } : = \bigoplus _ { ( s , a ) \in S \times \mathcal { A } } \widehat { P } _ { N } ( \cdot \mid s , a ) . } } \end{array}\tag{59}
$$

The following lemma shows that, for every fixed value function, the product estimator satisfies the same Bernstein inequality as an average of N independent complete transition tables.

Lemma 9 (Product-estimator Bernstein bound). Let $\Theta _ { 0 } , \dots , \Theta _ { N - 1 }$ be independent transition tables with distribution $\mathsf { Q } ,$ , and let $\widehat { \sf Q } _ { N }$ be defined by equation 59. Then, for every fixed function $V : \Omega $ $[ 0 , V _ { \mathrm { m a x } } ]$ and every $x > 0$

$$
\mathbb { P } \Bigg ( \Big | \mathbb { E } _ { \widehat { \mathbb { Q } } _ { N } } [ V ] - \mathbb { E } _ { \mathbb { Q } } [ V ] \Big | > \sqrt { \frac { 2 \operatorname { V a r } _ { \mathbb { Q } } ( V ) x } { N } } + \frac { V _ { \operatorname* { m a x } } x } { 3 N } \Bigg ) \leq 2 e ^ { - x } .\tag{60}
$$

Proof. For every $( s , a ) \in S \times A .$ , let $\pi _ { s , a }$ be an independent uniform permutation of $\{ 0 , \ldots , N - 1 \}$ , independent of the observed tables. For each $j ,$ define

$$
\Theta _ { j } ^ { \pi } \big ( s , a \big ) : = \Theta _ { \pi _ { s , a } ( j ) } ( s , a ) , \overline { { V } } ^ { \pi } : = \frac { 1 } { N } \sum _ { j = 0 } ^ { N - 1 } V ( \Theta _ { j } ^ { \pi } ) .
$$

Conditionally on the observations, the permutations select every coordinate independently and uniformly from its empirical marginal. Therefore,

$$
\mathbb { E } _ { \boldsymbol { \pi } } \left[ \boldsymbol { \overline { { V } } } ^ { \pi } \Big | \Theta _ { 0 , \cdot \cdot \cdot } , \Theta _ { N - 1 } \right] = \mathbb { E } _ { \widehat { \mathsf { Q } } _ { N } } [ V ] .\tag{61}
$$

For every fixed collection of permutations, the rematched tables $\Theta _ { 0 } ^ { \pi } , \ldots , \Theta _ { N - 1 } ^ { \pi }$ are independent and identically distributed according to $\mathsf { Q } .$ . Indeed, at each coordinate the permutation uses distinct observations, while all observations belonging to different coordinates are independent under the product model.

For every $\lambda \in \mathbb { R }$ , Jensen’s inequality and equation 61 give

$$
\begin{array} { r l } & { \mathbb { E } \exp \left( \lambda [ \mathbb { E } _ { \widehat { \mathsf { Q } } _ { N } } [ V ] - \mathbb { E } _ { \mathsf { Q } } [ V ] ] \right) } \\ & { \qquad \leq \mathbb { E } \mathbb { E } _ { \pi } \exp \left( \lambda [ \overline { { V } } ^ { \pi } - \mathbb { E } _ { \mathsf { Q } } [ V ] ] \right) . } \end{array}
$$

For fixed permutations, the right-hand side is the moment generating function of the average of $N$ independent copies of $V ( \Theta ) \stackrel { } { - } \mathbb { E } _ { \mathsf { Q } } [ V ]$ . The scalar Bernstein bound therefore applies with variance $\operatorname { V a r } _ { \mathsf { Q } } ( V )$ . Applying its upper-tail form to $V$ , its lower-tail form $\mathbf { t o } - V$ , and taking a union bound proves equation 60. □

We now apply Lemma 9 to the continuation values generated by $V _ { \ell } ^ { \star }$ . Define

$$
\iota _ { T } ( \delta ) : = \log \left( \frac { 2 n | \Omega | ^ { \ell - 1 } \big ( 1 + \lceil \log _ { 2 } T \rceil \big ) } { \delta } \right) .\tag{62}
$$

Lemma 10 (Uniform confidence for optimal continuation values). With probability at least $1 - \delta ,$ simultaneously for every transition sample size N used by the planner up to time $T ,$ every $u \in S$ and every $\theta _ { 1 : \ell - 1 } \in \Omega ^ { \ell - 1 }$ ,, and every $\theta _ { 1 : \ell - 1 } \in \Omega ^ { \ell - 1 }$

$$
\begin{array} { r l } & {  { \Big | \mathbb { E } _ { \Theta \sim \widehat { \mathbb { Q } } _ { N } } [ V _ { \ell } ^ { \star } ( u , \theta _ { 1 : \ell - 1 } , \Theta ) ] - \mathbb { E } _ { \Theta \sim \mathbb { Q } } [ V _ { \ell } ^ { \star } ( u , \theta _ { 1 : \ell - 1 } , \Theta ) ] \Big | } } \\ & { \qquad \leq \sqrt { \frac { 2 \operatorname { V a r } _ { \Theta \sim \mathbb { Q } } ( V _ { \ell } ^ { \star } ( u , \theta _ { 1 : \ell - 1 } , \Theta ) ) \iota _ { T } ( \delta ) } { N } } + \frac { V _ { \operatorname* { m a x } } \iota _ { T } ( \delta ) } { 3 N } . } \end{array}\tag{63}
$$

Proof. Fix N, u, and $\theta _ { 1 : \ell - 1 }$ , and consider the function

$$
V _ { \ell } ^ { \star } ( u , \theta _ { 1 : \ell - 1 } , \theta ) , \qquad \theta \in \Omega .
$$

This function takes values in $[ 0 , V _ { \mathrm { m a x } } ]$ and is fixed independently of the transition observations. Lemma 9, applied with $x = \iota _ { T } ( \delta )$ , therefore gives equation 63 with failure probability at most $2 e ^ { - \iota _ { T } ( \delta ) }$

The planner uses at most $1 + \lceil \log _ { 2 } T \rceil$ distinct transition sample sizes up to time T. A union bound thus gives a total failure probability of at most

$$
2 n | \Omega | ^ { \ell - 1 } \bigl ( 1 + \lceil \log _ { 2 } T \rceil \bigr ) e ^ { - \iota _ { T } ( \delta ) } = \delta .
$$

Since $| \Omega | = n ^ { n m }$

$$
\log \left( n | \Omega | ^ { \ell - 1 } \right) = \left[ 1 + ( \ell - 1 ) n m \right] \log n .
$$

Thus, although the number of possible look-ahead suffixes is exponential, its contribution to the confidence radius is only logarithmic.

Lemma 10 cannot be applied directly to the optimistic value function, since that function is constructed from the same transition observations as $\widehat { \sf Q } _ { N }$ . We therefore complement it with a weaker bound that holds simultaneously for every bounded function.

For every dyadic $N \leq T$ , define

$$
\rho _ { N } ( \delta ) : = \operatorname* { m i n } \left\{ 1 , \sqrt { \frac { n ^ { 2 } m \log ( N + 1 ) + \log \left( \frac { 1 + \lceil \log _ { 2 } T \rceil } { \delta } \right) } { 2 N } } \right\} .\tag{64}
$$

Lemma 11 (Uniform control of the value function). With probability at least $1 - \delta ,$ , simultaneously for every transition sample size N used by the planner up to time $\dot { T } ,$ , and every function $V : \Omega \stackrel { } {  }$ $[ 0 , V _ { \mathrm { m a x } } ]$

$$
\begin{array} { r l } & { \Big | \mathbb { E } _ { \Theta \sim \widehat { \mathbb { Q } } _ { N } } [ V ( \Theta ) ] - \mathbb { E } _ { \Theta \sim \mathbb { Q } } [ V ( \Theta ) ] \Big | } \\ & { \qquad \leq 2 \sqrt { 2 } \rho _ { N } ( \delta ) \sqrt { \mathrm { V a r } _ { \Theta \sim \mathbb { Q } } ( V ( \Theta ) ) } + 2 V _ { \operatorname* { m a x } } \rho _ { N } ^ { 2 } ( \delta ) , } \end{array}\tag{65}
$$

and

$$
\left| \sqrt { \mathrm { V a r } _ { \Theta \sim \widehat { \mathsf { Q } } _ { N } } ( V ( \Theta ) ) } - \sqrt { \mathrm { V a r } _ { \Theta \sim \mathsf { Q } } ( V ( \Theta ) ) } \right| \leq 2 V _ { \operatorname* { m a x } } \rho _ { N } ( \delta ) .\tag{66}
$$

In particular, these inequalities remain valid when V is selected after observing $\Theta _ { 0 } , \dots , \Theta _ { N - 1 }$

Proof. For distributions $p , q$ on Ω, let $\begin{array} { r } { H ^ { 2 } ( p , q ) = 1 - \sum _ { \theta } \sqrt { p ( \theta ) q ( \theta ) } } \end{array}$ . Writing $\mu = \mathbb { E } _ { q } [ V ]$ and

$$
p - q = ( { \sqrt { p } } - { \sqrt { q } } ) \bigl ( 2 { \sqrt { q } } + { \sqrt { p } } - { \sqrt { q } } \bigr ) ,
$$

Cauchy–Schwarz and $\| { \sqrt { p } } - { \sqrt { q } } \| _ { 2 } ^ { 2 } = 2 H ^ { 2 } ( p , q )$ give

$$
\begin{array} { r } { | \mathbb { E } _ { p } [ V ] - \mathbb { E } _ { q } [ V ] | \leq 2 \sqrt { 2 } H ( p , q ) \sqrt { \mathrm { V a r } _ { q } ( V ) + 2 V _ { \operatorname* { m a x } } H ^ { 2 } ( p , q ) } . } \end{array}
$$

Moreover,

$$
{ \sqrt { \operatorname { V a r } _ { p } ( V ) } } = \operatorname* { i n f } _ { c \in \mathbb { R } } \| ( V - c ) { \sqrt { p } } \| _ { 2 } .
$$

Evaluating the infimum at $c = \mathbb { E } _ { q } [ V ]$ , applying the triangle inequality, and then exchanging p and q yields

$$
\left| \sqrt { \mathrm { V a r } _ { p } ( V ) } - \sqrt { \mathrm { V a r } _ { q } ( V ) } \right| \leq 2 V _ { \operatorname* { m a x } } H ( p , q ) .
$$

Finally, the method of types, KL tensorization, and $2 H ^ { 2 } \leq \mathrm { K I }$ imply

$$
\mathbb { P } \Big ( H ( \widehat { \sf Q } _ { N } , \sf { Q } ) > \rho _ { N } ( \delta ) \Big ) \leq \frac { \delta } { 1 + \lceil \log _ { 2 } T \rceil } .
$$

A union bound over the planner update sample sizes completes the proof.

Unlike Lemma 10, this result is uniform over V. We will use its larger radius only to control the datadependent difference between the optimistic value function and $V _ { \ell } ^ { \star }$ . We now combine the Bernstein and Hellinger bounds into the optimistic expectation used by the planner. For every dyadic $N \leq T$ let

$$
\alpha _ { N } : = \operatorname* { m i n } \left\{ 1 , \sqrt { \frac { \iota _ { T } ( \delta ) } { N } } \right\} , \qquad \rho _ { N } : = \rho _ { N } ( \delta ) ,\tag{67}
$$

and, for $V : \Omega \to [ 0 , V _ { \mathrm { m a x } } ]$ , write

$$
{ \widehat { \sigma } } _ { N } ( V ) : = { \sqrt { \operatorname { V a r } _ { \Theta \sim { \widehat { \mathbb { Q } } } _ { N } } ( V ( \Theta ) ) } } .
$$

Define

$$
\begin{array} { r l } & { \mathrm { U C B } _ { N } ( V ) : = \operatorname* { m i n } \Bigl \{ V _ { \mathrm { m a x } } , \mathbb { E } _ { \Theta \sim \widehat { \mathbb { Q } } _ { N } } [ V ( \Theta ) ] + 8 V _ { \mathrm { m a x } } \alpha _ { N } \rho _ { N } } \\ & { \qquad + \operatorname* { m a x } \left\{ 7 \alpha _ { N } \widehat { \sigma } _ { N } ( V ) , 4 9 V _ { \mathrm { m a x } } \alpha _ { N } ^ { 2 } \right\} \Bigr \} . } \end{array}\tag{68}
$$

The second term inside the maximum ensures that the optimistic expectation remains monotone in the value function.

Lemma 12 (Properties of the optimistic expectation). On the intersection ofthe events in Lemmas 10 and 11, thefollowing properties hold simultaneouslyfor every dyadic $N \leq T$

1. The mapping

$$
V \longmapsto \operatorname { U C B } _ { N } ( V )
$$

is nondecreasing for the pointwise order and is 1-Lipschitz in the supremum norm.

2. For every $u \in S$ and $\theta _ { 1 : \ell - 1 } \in \Omega ^ { \ell - 1 }$ , the optimal continuation value

$$
V ( \theta ) : = V _ { \ell } ^ { \star } ( u , \theta _ { 1 : \ell - 1 } , \theta )
$$

satisfies

$$
\operatorname { U C B } _ { N } ( V ) \geq \mathbb { E } _ { \Theta \sim \mathbf { Q } } [ V ( \Theta ) ] .\tag{69}
$$

3. Let $W : \Omega \to [ 0 , V _ { \mathrm { m a x } } ]$ be any possibly data-dependent function such that $W \geq V$ pointwise. There exists a constant C s.t.

$$
\begin{array} { r l } { \mathrm { U C B } _ { N } ( W ) - \mathbb { E } _ { \Theta \sim \Theta } [ W ( \Theta ) ] \leq C \Bigg [ \alpha _ { N } \sqrt { \mathrm { V a r } _ { \Theta \sim \Theta } ( V ( \Theta ) ) } } & { } \\ { + \left( \alpha _ { N } + \rho _ { N } \right) \sqrt { V _ { \operatorname* { m a x } } \mathbb { E } _ { \Theta \sim \Theta } [ W ( \Theta ) - V ( \Theta ) ] } } & { } \\ { + V _ { \operatorname* { m a x } } ( \alpha _ { N } + \rho _ { N } ) ^ { 2 } \Bigg ] } & { } \end{array}\tag{70}
$$

Proof. (1.) Consider the expression inside the outer minimum in equation 68. Fix $\theta \in \Omega$ , hold all other values of V fixed, and increase $V ( \theta )$ . If the maximum in equation 68 equals $4 9 V _ { \operatorname* { m a x } } \alpha _ { N } ^ { 2 }$ , its derivative with respect to $V ( \theta )$ is simply

$$
{ \widehat { \mathsf { Q } } } _ { N } ( \theta ) \geq 0 .
$$

If the maximum equals $7 \alpha _ { N } \widehat { \sigma } _ { N } ( V )$ , then necessarily

$$
7 \alpha _ { N } \widehat { \sigma } _ { N } ( V ) \geq 4 9 V _ { \mathrm { m a x } } \alpha _ { N } ^ { 2 } ,
$$

and therefore

$$
\widehat { \sigma } _ { N } ( V ) \geq 7 V _ { \operatorname* { m a x } } \alpha _ { N } .
$$

Whenever $\widehat { \sigma } _ { N } ( V ) > 0$ , the derivative of the empirical mean plus the variance bonus is

$$
\widehat { \mathsf { Q } } _ { N } ( \theta ) \left( 1 + 7 \alpha _ { N } \frac { V ( \theta ) - \mathbb { E } _ { \widehat { \mathsf { Q } } _ { N } } [ V ] } { \widehat { \sigma } _ { N } ( V ) } \right) .
$$

Since

$$
\begin{array} { r } { V ( \theta ) - \mathbb { E } _ { \widehat { \mathsf { Q } } _ { N } } [ V ] \geq - V _ { \operatorname* { m a x } } , } \end{array}\tag{71}
$$

this derivative is at least

$$
\widehat { \mathsf { Q } } _ { N } ( \theta ) \left( 1 - \frac { 7 \alpha _ { N } V _ { \mathrm { m a x } } } { \widehat { \sigma } _ { N } ( V ) } \right) \geq 0 .\tag{72}
$$

When the two terms inside the maximum are equal, the expression remains continuous, so increasing $V ( \theta )$ cannot produce a downward jump. If $\widehat { \sigma } _ { N } ( V ) = 0$ , the constant term is selected unless $\alpha _ { N } = 0$ in which case the expression reduces to the empirical mean. Thus, increasing any value $V ( \theta )$ cannot decrease $\operatorname { U C B } _ { N } ( V )$ . Consequently, if $V ( \theta ) \leq W ( \theta )$ for every $\theta \in \Omega$ , then

$$
\operatorname { U C B } _ { N } ( V ) \leq \operatorname { U C B } _ { N } ( W ) .\tag{73}
$$

The additive term $8 V _ { \mathrm { m a x } } \alpha _ { N } \rho _ { N }$ and the clipping at $V _ { \mathrm { m a x } }$ clearly preserve this property.

It remains to prove the Lipschitz property. The preceding argument uses only that the difference between any value of the function and its mean is at least $- \bar { V } _ { \mathrm { m a x } } ;$ it therefore remains valid after adding a constant to the function. Since

$$
V \leq W + \| V - W \| _ { \infty }\tag{74}
$$

pointwise, monotonicity and the fact that adding a constant changes the empirical mean by that constant but leaves the empirical standard deviation unchanged give

$$
\operatorname { U C B } _ { N } ( V ) \leq \operatorname { U C B } _ { N } ( W ) + \| V - W \| _ { \infty } .\tag{75}
$$

Exchanging V and W yields

$$
\lvert \mathrm { U C B } _ { N } ( V ) - \mathrm { U C B } _ { N } ( W ) \rvert \le \lVert V - W \rVert _ { \infty } .\tag{76}
$$

This proves the first property.

(2.) First, note that

$$
\operatorname* { m a x } \{ 7 x , 4 9 y \} \geq 2 x + 5 y , \qquad x , y \geq 0 ,\tag{77}
$$

If $\alpha _ { N } = 1$ , then $\mathrm { U C B } _ { N } ( V ) = V _ { \mathrm { m a x } }$ , so equation 69 is immediate. Suppose therefore that $\alpha _ { N } < 1$ Lemma 10 gives

$$
\mathbb { E } _ { \mathsf { Q } } [ V ] - \mathbb { E } _ { \widehat { \mathsf { Q } } _ { N } } [ V ] \leq \sqrt { 2 } \alpha _ { N } \sqrt { \mathrm { V a r } _ { \mathsf { Q } } ( V ) } + \frac { V _ { \operatorname* { m a x } } \alpha _ { N } ^ { 2 } } { 3 } .
$$

By Lemma 11,

$$
\sqrt { \operatorname { V a r } _ { \mathbb { Q } } ( V ) } \leq \widehat { \sigma } _ { N } ( V ) + 2 V _ { \operatorname* { m a x } } \rho _ { N } .
$$

Consequently,

$$
\mathbb { E } _ { \mathbb { Q } } [ V ] - \mathbb { E } _ { \widehat { \mathbb { Q } } _ { N } } [ V ] \leq \sqrt { 2 } \alpha _ { N } \widehat { \sigma } _ { N } ( V ) + 2 \sqrt { 2 } V _ { \operatorname* { m a x } } \alpha _ { N } \rho _ { N } + \frac { V _ { \operatorname* { m a x } } \alpha _ { N } ^ { 2 } } { 3 } .
$$

The bonus in equation 68 dominates the right-hand side by equation 77, proving equation 69.

(3.) If $\alpha _ { N } = 1$ , the result follows immediately after enlarging C, since both expectations lie in $[ 0 , V _ { \mathrm { m a x } } ]$ . Assume henceforth that $\alpha _ { N } < 1$ and write

$$
D : = W - V \geq 0 .
$$

Since clipping can only decrease the optimistic estimate,

$$
\begin{array} { r l } { \mathrm { U C B } _ { N } ( W ) - \mathbb { E } _ { \mathsf { Q } } [ W ] \le \left( \mathbb { E } _ { \widehat { \mathsf { Q } } _ { N } } [ V ] - \mathbb { E } _ { \mathsf { Q } } [ V ] \right) } & { } \\ { + \left( \mathbb { E } _ { \widehat { \mathsf { Q } } _ { N } } [ D ] - \mathbb { E } _ { \mathsf { Q } } [ D ] \right) + 8 V _ { \mathrm { m a x } } \alpha _ { N } \rho _ { N } } & { } \\ { + 7 \alpha _ { N } \widehat { \sigma } _ { N } ( W ) + 4 9 V _ { \mathrm { m a x } } \alpha _ { N } ^ { 2 } . } \end{array}\tag{78}
$$

The first difference is controlled by Lemma 10. Since D may depend on the observations, we use instead Lemma 11:

$$
\mathbb { E } _ { \widehat { \mathbb { Q } } _ { N } } [ D ] - \mathbb { E } _ { \mathsf { Q } } [ D ] \leq 2 \sqrt { 2 } \rho _ { N } \sqrt { V _ { \mathrm { m a x } } \mathbb { E } _ { \mathsf { Q } } [ D ] } + 2 V _ { \mathrm { m a x } } \rho _ { N } ^ { 2 } ,\tag{79}
$$

where we used

$$
\begin{array} { r } { \operatorname { V a r } _ { \mathbb { Q } } ( D ) \leq \mathbb { E } _ { \mathbb { Q } } [ D ^ { 2 } ] \leq V _ { \operatorname* { m a x } } \mathbb { E } _ { \mathbb { Q } } [ D ] . } \end{array}
$$

Finally, the triangle inequality for standard deviations and Lemma 11 give

$$
\begin{array} { r l } & { \widehat { \sigma } _ { N } ( W ) \leq \widehat { \sigma } _ { N } ( V ) + \widehat { \sigma } _ { N } ( D ) } \\ & { \qquad \leq \sqrt { \mathrm { V a r } _ { \mathbb { Q } } ( V ) } + \sqrt { \mathrm { V a r } _ { \mathbb { Q } } ( D ) } + 4 V _ { \operatorname* { m a x } } \rho _ { N } } \\ & { \qquad \leq \sqrt { \mathrm { V a r } _ { \mathbb { Q } } ( V ) } + \sqrt { V _ { \operatorname* { m a x } } \mathbb { E } _ { \mathbb { Q } } [ D ] } + 4 V _ { \operatorname* { m a x } } \rho _ { N } . } \end{array}
$$

Substituting these bounds into equation 78 and collecting terms proves equation 70.

We now insert the optimistic expectation into the Bellman operator. Consider a planner update using N past transition tables and an optimistic reward function $r ^ { + } : S \times \mathcal { A }  [ 0 , R _ { \mathrm { m a x } } ]$ . For every bounded value function V and every window $c = ( \theta _ { 0 } , \ldots , \theta _ { \ell - 1 } )$ , define

$$
\begin{array} { r } { ( \overline { { \mathcal { T } } } _ { N , r ^ { + } } V ) ( s , c ) : = \underset { a \in A } { \operatorname* { m a x } } \operatorname* { m i n } \Bigl \{ V _ { \mathrm { m a x } } , r ^ { + } ( s , a ) + \gamma \mathrm { U C B } _ { N } \bigl ( \theta \mapsto V ( \theta _ { 0 } ( s , a ) , \theta _ { 1 } , \ldots , \theta _ { \ell - 1 } , \theta ) \bigr ) \Bigr \} . } \end{array}\tag{80}
$$

Let $\overline { { V } } _ { N , r ^ { + } }$ denote its fixed point. The corresponding action scores are

$$
\overline { { Q } } _ { N , r ^ { + } } ( s , c , a ) : = \operatorname* { m i n } \Big \{ V _ { \mathrm { m a x } } , r ^ { + } ( s , a ) + \gamma \operatorname { U C B } _ { N } \big ( \theta \mapsto \overline { { V } } _ { N , r ^ { + } } \big ( \theta _ { 0 } ( s , a ) , \theta _ { 1 } , \ldots , \theta _ { \ell - 1 } , \theta \big ) \big ) \Big \} .\tag{81}
$$

Lemma 13 (Contraction and optimism of the ideal planner). On the events ofLemmas 10 and 11, if $r ^ { + } \geq r$ pointwise, then $\overline { { \mathcal { T } } } _ { N , r ^ { + } }$ is a monotone γ-contraction and

$$
\overline { { V } } _ { N , r ^ { + } } \left( s , c \right) \geq V _ { \ell } ^ { \star } ( s , c ) \qquad f o r e \nu e r y \left( s , c \right) \in \mathcal { S } \times \Omega ^ { \ell } .\tag{82}
$$

Proof. Lemma 12 shows that $\mathrm { U C B } _ { N }$ is monotone and 1-Lipschitz. Maximization over actions and clipping preserve these properties. Therefore, for any two bounded value functions $V$ and $V ^ { \prime }$

$$
\begin{array} { r } { \mathopen { } \mathclose \bgroup \left\| \overline { { \mathcal { T } } } _ { N , r ^ { + } } V - \overline { { \mathcal { T } } } _ { N , r ^ { + } } V ^ { \prime } \aftergroup \egroup \right\| _ { \infty } \leq \gamma \| V - V ^ { \prime } \| _ { \infty } . } \end{array}
$$

Hence, $\overline { { \mathcal { T } } } _ { N , r ^ { + } }$ has a unique fixed point. Fix a window ${ \boldsymbol { c } } = ( \theta _ { 0 } , \dots , \theta _ { \ell - 1 } )$ , a state $s ,$ and an action a. Applying equation 69 to the function $\theta \mapsto V _ { \ell } ^ { \star } \bigl ( \theta _ { 0 } ( s , a ) , \theta _ { 1 } , \ldots , \theta _ { \ell - 1 } , \theta \bigr )$ and using $r ^ { + } \geq r$ gives

$$
\begin{array} { r } { \overline { { \mathcal { T } } } _ { N , r ^ { + } } V _ { \ell } ^ { \star } \geq \mathcal { T } _ { \ell } V _ { \ell } ^ { \star } = V _ { \ell } ^ { \star } . } \end{array}
$$

By monotonicity,

$$
\left( \overline { { T } } _ { N , r ^ { + } } \right) ^ { k } V _ { \ell } ^ { \star } \geq V _ { \ell } ^ { \star } \qquad \mathrm { f o r } \mathrm { e v e r y } k \geq 1 .
$$

The contraction property implies that the left-hand side converges to $\overline { { V } } _ { N , r ^ { + } }$ . Letting $k  + \infty$ proves equation 82. □

The operator in equation 80 is used only in the analysis. To approximate its expectations efficiently, conditionally on the tables observed before the current planner update, the planner draws

$$
Z ^ { 1 } , \dots , Z ^ { M _ { N } } \overset { \mathrm { i . i . d . } } { \sim } \widehat { \mathbf { Q } } _ { N } , \qquad M _ { N } = \left\lceil \frac { N } { ( 1 - \gamma ) ^ { 3 } } \right\rceil .\tag{83}
$$

For later use, denote the empirical distribution of the dictionary by

$$
\mathsf Q _ { N } ^ { D } : = \frac { 1 } { M _ { N } } \sum _ { j = 1 } ^ { M _ { N } } \delta _ { Z ^ { j } } , \qquad \sigma _ { N } ^ { D } ( V ) : = \sqrt { \mathrm { V a r } _ { \Theta \sim \mathsf Q _ { N } ^ { D } } ( V ( \Theta ) ) } .
$$

Let $v : \mathcal { S } \times [ M _ { N } ] ^ { \ell } \to [ 0 , V _ { \mathrm { m a x } } ]$ be the current value array. For a dictionary state $( s , i _ { 0 } , \dotsc , i _ { \ell - 1 } )$ and an action $^ { a , }$ define the empirical continuation mean by

$$
\overline { { v } } ( s , a , i _ { 0 } , \ldots , i _ { \ell - 1 } ) : = \frac { 1 } { M _ { N } } \sum _ { j = 1 } ^ { M _ { N } } v \bigl ( Z ^ { i _ { 0 } } ( s , a ) , i _ { 1 } , \ldots , i _ { \ell - 1 } , j \bigr ) ,
$$

and its empirical variance by

$$
s _ { v } ^ { 2 } ( s , a , i _ { 0 } , \ldots , i _ { \ell - 1 } ) : = \frac { 1 } { M _ { N } } \sum _ { j = 1 } ^ { M _ { N } } \left( v \bigl ( Z ^ { i _ { 0 } } ( s , a ) , i _ { 1 } , \ldots , i _ { \ell - 1 } , j \bigr ) - \overline { { v } } ( s , a , i _ { 0 } , \ldots , i _ { \ell - 1 } ) \right) ^ { 2 } .
$$

The optimistic continuation estimate associated with v is

$$
\begin{array} { r l } & { \mathrm { U C B } _ { N } ^ { D } ( v ; s , a , i _ { 0 } , \ldots , i _ { \ell - 1 } ) } \\ & { \quad : = \operatorname* { m i n } \Biggl \{ V _ { \mathrm { m a x } } , \overline { { v } } ( s , a , i _ { 0 } , \ldots , i _ { \ell - 1 } ) + 8 V _ { \mathrm { m a x } } \alpha _ { N } \rho _ { N } } \\ & { \qquad + \operatorname* { m a x } \Bigl \{ 7 \alpha _ { N } s _ { v } ( s , a , i _ { 0 } , \ldots , i _ { \ell - 1 } ) , 4 9 V _ { \mathrm { m a x } } \alpha _ { N } ^ { 2 } \Bigr \} \Biggl \} . } \end{array}
$$

The corresponding Bellman operator is

$$
\begin{array} { r l } & { ( { \mathcal { T } } _ { N , r ^ { + } } ^ { D , + } v ) ( s , i _ { 0 } , \ldots , i _ { \ell - 1 } ) } \\ & { \qquad : = \underset { a \in \mathcal { A } } { \operatorname* { m a x } } \operatorname* { m i n } \left\{ V _ { \operatorname* { m a x } } , r ^ { + } ( s , a ) + \gamma { \mathrm { U C B } } _ { N } ^ { D } ( v ; s , a , i _ { 0 } , \ldots , i _ { \ell - 1 } ) \right\} . } \end{array}\tag{84}
$$

Value iteration applies equation 84 to the dictionary states. For a window $c \in \Omega ^ { \ell }$ observed during interaction, the backward extension applies the same update successively to the tables in c. Thus, the finite planner replaces the expectation and standard deviation under ${ \widehat { \sf Q } } _ { N }$ by their empirical counterparts over the sampled dictionary. For a window ${ \boldsymbol { c } } = ( \theta _ { 0 } , \ldots , \theta _ { \ell - 1 } )$ ) observed during interaction, the backward extension of Algorithm 3 uses the same empirical continuation estimate at every level. We denote by $\widetilde { Q } _ { N , r ^ { + } } ( s , c , a )$ the resulting action score when action a is kept fixed at the first Bellman update.

Lemma 14 (Concentration of a sample mean and standard deviation). Let $Y _ { 1 } , \dots , Y _ { M } \in [ 0 , B ]$ be i.i.d. random variables, and let $\widehat { \mu } _ { M }$ and ${ \widehat { \sigma } } _ { M }$ be their empirical mean and empirical standard deviation, with denominator M. For every $x \geq 1$ , with probability at least $1 - 6 e ^ { - x }$

$$
| \widehat { \mu } _ { M } - \mathbb { E } ( Y ) | \leq B \sqrt { \frac { 2 x } { M } } , \qquad | \widehat { \sigma } _ { M } - \sqrt { \mathrm { V a r } ( Y ) } | \leq 5 B \sqrt { \frac { x } { M } } .\tag{85}
$$

Proof. The mean bound follows from Hoeffding’s inequality, and the standard-deviation bound follows from Theorem 10 of Maurer & Pontil (2009), after rescaling to [0, B] and accounting for the use of denominator M. A union bound completes the proof. □

Let $E _ { T }$ be the total number of planner updates up to time T. Under the update rule of DLA-UCB,

$$
E _ { T } \leq 2 + ( 1 + n m ) \big ( 1 + \lceil \log _ { 2 } T \rceil \big ) .\tag{86}
$$

Indeed, the global number of expired tables and each of the nm reward counts can double at most $1 + \lceil \log _ { 2 } T \rceil$ times. Define

$$
L _ { T } ^ { D } ( \delta ) : = \log \left( \frac { 1 2 E _ { T } n | \Omega | ^ { \ell - 1 } } { \delta } \right) .\tag{87}
$$

Recall that $\widetilde { Q } _ { t } ( s , c , a )$ denotes the action score computed by the backward extension of the planning object used at time t. If its most recent update used N transition tables and reward bound $r ^ { + }$ , we compare this computed score with the ideal score $\overline { { Q } } _ { N , r ^ { + } } ( s , c , a )$ defined in equation 81.

Lemma 15 (Accuracy of the dictionary planner). Suppose that, at each planner update, the dictionary is sampled as in equation 83, with $\begin{array} { r } { M _ { N } \ge \frac { N } { ( 1 - \gamma ) ^ { 3 } } . I j } \end{array}$ value iteration is run for $\begin{array} { r } { \left\lceil \frac { 1 } { 1 - \gamma } \log \frac { 2 N ^ { 2 } } { 1 - \gamma } \right\rceil } \end{array}$ iterations, then, with probability at least $1 - \delta ,$ at every planner update,

$$
\operatorname* { s u p } _ { s \in \mathcal { S } , c \in \Omega ^ { \ell } , a \in \mathcal { A } } \left| \widetilde { Q } _ { N , r ^ { + } } ( s , c , a ) - \overline { { Q } } _ { N , r ^ { + } } ( s , c , a ) \right| \leq C R _ { \operatorname* { m a x } } \sqrt { \frac { L _ { T } ^ { D } ( \delta ) } { N ( 1 - \gamma ) } + \frac { R _ { \operatorname* { m a x } } } { N ^ { 2 } } } .\tag{88}
$$

Proof. Consider one planner update. At this point, ${ \widehat { \mathsf { Q } } } _ { N }$ and $r ^ { + }$ have already been computed from the previously observed transition tables and rewards. We treat these quantities as fixed and consider only the randomness of the new dictionary $Z ^ { 1 } , \dots , Z ^ { M _ { N } }$ , which is sampled independently from ${ \widehat { \mathsf { Q } } } _ { N }$ . Then $\widehat { \mathsf Q } _ { N } , r ^ { + }$ , and the exact-expectation fixed point $\overline { { V } } _ { N , r ^ { + } }$ are fixed, whereas ${ Z } ^ { 1 } , \ldots , { Z } ^ { M _ { N } }$ are independent samples from ${ \widehat { \mathsf { Q } } } _ { N }$ . For every $u \in S$ and $\theta _ { 1 : \ell - 1 } \in \Omega ^ { \ell - 1 }$ , apply Lemma 14 to

$$
Z \longmapsto { \overline { { V } } } _ { N , r ^ { + } } ( u , \theta _ { 1 : \ell - 1 } , Z ) .
$$

A union bound over the states, suffixes, and planner updates gives, simultaneously,

$$
\left| \mathbb { E } _ { \mathbf { Q } _ { N } ^ { D } } [ \overline { { V } } _ { N , r ^ { + } } ] - \mathbb { E } _ { \widehat { \mathbf { Q } } _ { N } } [ \overline { { V } } _ { N , r ^ { + } } ] \right| \leq C V _ { \operatorname* { m a x } } \sqrt { \frac { L _ { T } ^ { D } ( \delta ) } { M _ { N } } } ,
$$

and

$$
\left| \sigma _ { N } ^ { D } ( \overline { { V } } _ { N , r ^ { + } } ) - \widehat { \sigma } _ { N } ( \overline { { V } } _ { N , r ^ { + } } ) \right| \leq C V _ { \operatorname* { m a x } } \sqrt { \frac { L _ { T } ^ { D } ( \delta ) } { M _ { N } } } .
$$

The state and suffix arguments are omitted. The optimistic continuation estimate is the sum of the mean and max $\left\{ 7 \omega _ { N } \sigma , 4 9 V _ { \mathrm { m a x } } \alpha _ { N } ^ { 2 } \right\}$ . Since $\alpha _ { N } \ \leq \ 1$ , changing the mean and standard deviation by at most the preceding amount changes the optimistic continuation estimate by at most $\begin{array} { r } { C V _ { \operatorname* { m a x } } \sqrt { \frac { L _ { T } ^ { D } ( \delta ) } { M _ { N } } } } \end{array}$ . Maximization over actions and clipping cannot increase this error. Therefore,

$$
\left\| \overline { { \mathcal { T } } } _ { N , r ^ { + } } ^ { D } \overline { { V } } _ { N , r ^ { + } } - \overline { { \mathcal { T } } } _ { N , r ^ { + } } \overline { { V } } _ { N , r ^ { + } } \right\| _ { \infty } \leq C \gamma V _ { \operatorname* { m a x } } \sqrt { \frac { L _ { T } ^ { D } ( \delta ) } { M _ { N } } } .
$$

Denote the fixed point of $\overline { { \mathcal { T } } } _ { N , r ^ { + } } ^ { D } \ : \mathsf { b y } \ : \overline { { V } } _ { N , r ^ { + } } ^ { D }$ . Since both operators are γ-contractions,

$$
\begin{array} { r l } { \left\| \overline { { V } } _ { N , r ^ { + } } ^ { D } - \overline { { V } } _ { N , r ^ { + } } \right\| _ { \infty } \leq \left\| \overline { { \mathcal { T } } } _ { N , r ^ { + } } ^ { D } \overline { { V } } _ { N , r ^ { + } } ^ { D } - \overline { { \mathcal { T } } } _ { N , r ^ { + } } ^ { D } \overline { { V } } _ { N , r ^ { + } } \right\| _ { \infty } } & { } \\ & { \qquad + \left\| \overline { { \mathcal { T } } } _ { N , r ^ { + } } ^ { D } \overline { { V } } _ { N , r ^ { + } } - \overline { { \mathcal { T } } } _ { N , r ^ { + } } \overline { { V } } _ { N , r ^ { + } } \right\| _ { \infty } } \\ & { \qquad \leq \gamma \left\| \overline { { V } } _ { N , r ^ { + } } ^ { D } - \overline { { V } } _ { N , r ^ { + } } \right\| _ { \infty } + C \gamma V _ { \operatorname* { m a x } } \sqrt { \frac { L _ { T } ^ { D } ( \delta ) } { M _ { N } } } . } \end{array}
$$

Rearranging gives

$$
\left. \overline { { V } } _ { N , r ^ { + } } ^ { D } - \overline { { V } } _ { N , r ^ { + } } \right. _ { \infty } \leq \frac { C \gamma V _ { \operatorname* { m a x } } } { 1 - \gamma } \sqrt { \frac { L _ { T } ^ { D } ( \delta ) } { M _ { N } } } .
$$

The restriction, extension, and stability arguments of Lemmas $6 , 7 ,$ , and 8 apply unchanged to the dictionary operator, since the same optimistic continuation estimate is used in the finite value iteration and in the backward extension. Hence the backward extension introduces no additional approximation, while stopping value iteration after $K _ { N }$ iterations contributes at most $\gamma ^ { K _ { N } } V _ { \mathrm { m a x } }$ . The difference between the scores returned after convergence and $\overline { { Q } } _ { N , r ^ { + } } ( s , c , a )$ is therefore at most

$$
\frac { C V _ { \mathrm { m a x } } } { 1 - \gamma } \sqrt { \frac { L _ { T } ^ { D } ( \delta ) } { M _ { N } } } .
$$

This proves equation 88.

Conditional on the expired tables, concentration is applied only to the ideal fixed point, which is independent of the fresh dictionary; contraction then transfers the estimate to the dictionary-dependent fixed point. For a state–action pair visited $k \geq 1$ times, let ${ \widehat { r } } _ { k } ( s , a )$ be the empirical reward mean. At each planner update triggered by a doubled reward count, define

$$
\begin{array} { r l } & { U _ { k } ^ { R } ( s , a ) : = \operatorname* { m i n } \left\{ R _ { \mathrm { m a x } } , \widehat { r } _ { k } ( s , a ) + R _ { \mathrm { m a x } } \sqrt { \frac { 2 L _ { T } ^ { R } ( \delta ) } { k } } \right\} , } \\ & { ~ L _ { T } ^ { R } ( \delta ) : = \log \frac { 8 n m ( 1 + \lceil \log _ { 2 } T \rceil ) } { \delta } . } \end{array}\tag{89}
$$

For an unvisited pair, set $U _ { 0 } ^ { R } ( s , a ) = R _ { \mathrm { m a x } }$ . As in the main algorithm, $r ^ { + }$ is the minimum of all successive upper bounds for that pair.

Lemma 16 (Reward confidence). With probability at least $1 - \delta ,$

$$
r ( s , a ) \leq r _ { t } ^ { + } ( s , a ) \quad f o r e \nu e r y t , s , a ,\tag{90}
$$

and

$$
\sum _ { t = 0 } ^ { T - 1 } \bigl ( r _ { t } ^ { + } ( S _ { t } , A _ { t } ) - r ( S _ { t } , A _ { t } ) \bigr ) \leq C R _ { \operatorname* { m a x } } \left[ \sqrt { n m T L _ { T } ^ { R } ( \delta ) } + n m L _ { T } ^ { R } ( \delta ) \right] .\tag{91}
$$

Proof. Hoeffding’s inequality and a union bound over the nm pairs and their dyadic counts prove equation 90. On this event, if the current visit count of a pair is $j ,$ the estimate used by the algorithm was computed from at least $j / 2$ observations. Its width is therefore at most

$$
C R _ { \operatorname* { m a x } } \sqrt { \frac { L _ { T } ^ { R } ( \delta ) } { j \vee 1 } } .
$$

For a pair visited ${ \cal N } _ { T } ( s , a )$ times, $\begin{array} { r } { \sum _ { j = 1 } ^ { N _ { T } ( s , a ) } j ^ { - 1 / 2 } \le 2 \sqrt { N _ { T } ( s , a ) } } \end{array}$ . Summing over pairs and applying Cauchy–Schwarz gives

$$
\sum _ { s , a } { \sqrt { N _ { T } ( s , a ) } } \leq { \sqrt { n m T } } .
$$

The first visits and the linear parts of the confidence radii contribute the second term in equation 91. □

At time $t \geq 1$ , let $N _ { t }$ denote the number of transition tables used by the current planner. Since the planner is updated whenever this number doubles, $\begin{array} { r } { \frac { t } { 2 } \leq N _ { t } \leq t } \end{array}$ . For every transition sample size N used by the planner, define

$$
\varepsilon _ { N } ^ { D } : = C R _ { \operatorname* { m a x } } \sqrt { \frac { L _ { T } ^ { D } ( \delta ) } { N ( 1 - \gamma ) } } + \frac { R _ { \operatorname* { m a x } } } { N ^ { 2 } } .\tag{92}
$$

Lemma 17 (Cumulative confidence bounds). For fixed ℓ,

$$
\sum _ { t = 1 } ^ { T - 1 } \alpha _ { N _ { t } } ^ { 2 } = \widetilde { O } ( n m ) ,\tag{93}
$$

$$
\sum _ { t = 1 } ^ { T - 1 } \bigl ( \alpha _ { N _ { t } } + \rho _ { N _ { t } } \bigr ) ^ { 2 } = \widetilde { O } ( n ^ { 2 } m ) ,\tag{94}
$$

$$
\sum _ { t = 1 } ^ { T - 1 } \varepsilon _ { N _ { t } } ^ { D } = \widetilde { O } \left( R _ { \operatorname* { m a x } } \sqrt { \frac { n m T } { 1 - \gamma } } \right) .\tag{95}
$$

Where $\alpha _ { N _ { 1 } }$ is defined $6 7 , \rho _ { N }$ is defined $6 4 , \varepsilon _ { N _ { t } }$ is defined

Proof. The definitions of the confidence radii give

$$
\alpha _ { N } ^ { 2 } \leq 1 \wedge \widetilde { O } \left( \frac { n m } { N } \right) , \qquad \rho _ { N } ^ { 2 } \leq 1 \wedge \widetilde { O } \left( \frac { n ^ { 2 } m } { N } \right) .
$$

Consequently,

$$
\left( \alpha _ { N } + \rho _ { N } \right) ^ { 2 } \leq 4 \land \widetilde { O } \left( \frac { n ^ { 2 } m } { N } \right) .
$$

Since $N _ { t } \geq t / 2$

$$
\sum _ { t = 1 } ^ { T - 1 } \frac { 1 } { N _ { t } } = O ( \log T ) , \qquad \sum _ { t = 1 } ^ { T - 1 } \frac { 1 } { \sqrt { N _ { t } } } = O ( \sqrt { T } ) , \qquad \sum _ { t = 1 } ^ { T - 1 } \frac { 1 } { N _ { t } ^ { 2 } } = O ( 1 ) .
$$

Summing the preceding bounds proves equation 93 and equation 94.

Moreover, equation 92 gives

$$
\varepsilon _ { N } ^ { D } = \widetilde { O } \left( R _ { \mathrm { m a x } } \sqrt { \frac { n m } { N ( 1 - \gamma ) } } + \frac { R _ { \mathrm { m a x } } } { N ^ { 2 } } \right) .
$$

Summing this inequality and using the last two bounds above proves equation 95.

Let $e ( t )$ denote the planner epoch active at time $t ,$ and abbreviate

$$
W _ { t } : = W _ { N _ { t } , r _ { t } ^ { + } } , \qquad D _ { t } : = W _ { t } - V _ { \ell } ^ { \star } .
$$

The function $D _ { t }$ is fixed within each planner epoch and, by Lemma 13,

$$
0 \leq D _ { t } ( x ) \leq V _ { \operatorname* { m a x } } , \qquad x \in \mathcal { X } .\tag{96}
$$

Let $\mathcal { F } _ { t }$ contain the history and the current look-ahead window before $A _ { t }$ is chosen. Define

$$
\mu _ { t } : = \mathbb { E } \left[ D _ { t } ( X _ { t + 1 } ) \mid { \mathcal { F } } _ { t } \right] ,\tag{97}
$$

$$
\sigma _ { t } ^ { 2 } : = \operatorname { V a r } \left( V _ { \ell } ^ { \star } ( X _ { t + 1 } ) \mid { \mathcal { F } } _ { t } \right) .\tag{98}
$$

Conditionally on $\mathcal { F } _ { t }$ , the only randomness in $X _ { t + 1 }$ is the fresh table $\Theta _ { t + \ell } .$ . Hence these are precisely the mean and variance under Q appearing in Lemma 12.

Lemma 18 (One-step optimistic recursion). On the joint transition, reward, and dictionary confidence event, the action selected by DLA-UCB satisfies

$$
\begin{array} { r l } & { D _ { t } ( X _ { t } ) + V _ { \ell } ^ { \star } ( X _ { t } ) - Q _ { \ell } ^ { \star } ( X _ { t } , A _ { t } ) } \\ & { \qquad \leq r _ { t } ^ { + } ( S _ { t } , A _ { t } ) - r ( S _ { t } , A _ { t } ) + 2 \varepsilon _ { N _ { t } } ^ { D } + \gamma \mu _ { t } } \\ & { \qquad + C \gamma \left[ \alpha _ { t } \sigma _ { t } + \beta _ { t } \sqrt { V _ { \operatorname* { m a x } } \mu _ { t } } + V _ { \operatorname* { m a x } } \beta _ { t } ^ { 2 } \right] . } \end{array}\tag{99}
$$

Proof. By equation 88, the action maximizing the computed dictionary score is $2 \varepsilon _ { N _ { t } } ^ { D }$ -greedy for the ideal scores:

$$
W _ { t } ( X _ { t } ) \leq \overline { { Q } } _ { N _ { t } , r _ { t } ^ { + } } ( X _ { t } , A _ { t } ) + 2 \varepsilon _ { N _ { t } } ^ { D } .
$$

Consequently,

$$
\begin{array} { r l } & { D _ { t } ( X _ { t } ) + V _ { \ell } ^ { \star } ( X _ { t } ) - Q _ { \ell } ^ { \star } ( X _ { t } , A _ { t } ) = W _ { t } ( X _ { t } ) - Q _ { \ell } ^ { \star } ( X _ { t } , A _ { t } ) } \\ & { \qquad \leq r _ { t } ^ { + } ( S _ { t } , A _ { t } ) - r ( S _ { t } , A _ { t } ) + 2 \varepsilon _ { N _ { t } } ^ { D } } \\ & { \qquad + \gamma \left[ \mathrm { U C B } _ { N _ { t } } ( ( W _ { t } ) _ { X _ { t } , A _ { t } } ) - \mathbb { E } \big [ V _ { \ell } ^ { \star } ( X _ { t + 1 } ) \mid \mathcal { F } _ { t } \big ] \right] . } \end{array}
$$

Clipping cannot invalidate this upper bound because $Q _ { \ell } ^ { \star } \leq V _ { \operatorname* { m a x } } .$ Add and subtract $\mathbb { E } [ W _ { t } ( X _ { t + 1 } ) \ |$ $\mathcal { F } _ { t } ]$ ]. The added difference is $\mu _ { t }$ , and equation 70, with $V = ( V _ { \ell } ^ { \star } ) _ { X _ { t } , A _ { t } }$ and $W = \dot { ( W _ { t } ) } _ { X _ { t } , A _ { t } } ,$ bounds the remaining term. This proves equation 99. □

The first lemma converts predictable nonnegative quantities into their realized counterparts without paying a $\sqrt { T }$ term.

Lemma 19 (Predictable-to-realized comparison). Let $Y _ { t } \in [ 0 , B ]$ be $\mathcal { F } _ { t + \cdot }$ -measurable and let $\mu _ { t } =$ $\mathbb { E } [ Y _ { t } \mid \mathcal { F } _ { t } ]$ ]. For every $\eta \in ( 0 , 1 ]$ , with probability at least $1 - \delta _ { \mathrm { { \scriptsize \cdot } } }$

$$
\sum _ { t = 0 } ^ { T - 1 } \mu _ { t } \leq ( 1 + \eta ) \sum _ { t = 0 } ^ { T - 1 } Y _ { t } \frac { 2 B \log ( 1 / \delta ) } { \eta } .\tag{100}
$$

Proof. For $\lambda \in ( 0 , 1 ]$ , conditional Jensen’s inequality for exp $\left( - \lambda Y _ { t } / B \right)$ gives

$$
\mathbb { E } \left[ e ^ { - \lambda Y _ { t } / B } \mid \mathcal { F } _ { t } \right] \leq \exp \left( - ( 1 - e ^ { - \lambda } ) \frac { \mu _ { t } } { B } \right) .
$$

The corresponding product is a nonnegative supermartingale. Ville’s inequality therefore gives, with probability at least $1 - \delta$

$$
( 1 - e ^ { - \lambda } ) \sum _ { t } \mu _ { t } \leq \lambda \sum _ { t } Y _ { t } + B \log ( 1 / \delta ) .
$$

For $\lambda \in ( 0 , 1 ] , \lambda / ( 1 - e ^ { - \lambda } ) \leq 1 + \lambda$ and $( 1 - e ^ { - \lambda } ) ^ { - 1 } \leq 2 / \lambda$ . Taking $\lambda = \eta$ proves the result.

The next result is the discounted law of total variance needed for the Bernstein term.

Lemma 20 (Discounted total variance). Let

$$
G _ { T } : = \sum _ { t = 0 } ^ { T - 1 } \left[ V _ { \ell } ^ { \star } ( X _ { t } ) - Q _ { \ell } ^ { \star } ( X _ { t } , A _ { t } ) \right] .
$$

With probability at least $1 - \delta ,$

$$
\sum _ { t = 0 } ^ { T - 1 } \gamma ^ { 2 } \sigma _ { t } ^ { 2 } \le C \left[ V _ { \operatorname* { m a x } } R _ { \operatorname* { m a x } } T + V _ { \operatorname* { m a x } } G _ { T } + V _ { \operatorname* { m a x } } ^ { 2 } \log \frac { 1 } { \delta } \right] .\tag{101}
$$

Proof. Write

$$
v _ { t } : = V _ { \ell } ^ { \star } ( X _ { t } ) , \qquad g _ { t } : = V _ { \ell } ^ { \star } ( X _ { t } ) - Q _ { \ell } ^ { \star } ( X _ { t } , A _ { t } ) , \qquad y _ { t } : = r ( S _ { t } , A _ { t } ) + g _ { t } .
$$

The Bellman equation gives

$$
v _ { t } = y _ { t } + \gamma \mathbb { E } [ v _ { t + 1 } \mid \mathcal { F } _ { t } ] .\tag{102}
$$

All terms are nonnegative, and $0 \le y _ { t } \le v _ { t } \le V _ { \mathrm { m a x } }$ . Using equation 102,

$$
\begin{array} { r l } & { \gamma ^ { 2 } \sigma _ { t } ^ { 2 } = \gamma ^ { 2 } \mathbb { E } [ v _ { t + 1 } ^ { 2 } \mid \mathcal { F } _ { t } ] - ( v _ { t } - y _ { t } ) ^ { 2 } } \\ & { \qquad \leq \gamma ^ { 2 } \mathbb { E } [ v _ { t + 1 } ^ { 2 } \mid \mathcal { F } _ { t } ] - v _ { t } ^ { 2 } + 2 V _ { \operatorname* { m a x } } y _ { t } . } \end{array}
$$

Summing and inserting the realized $v _ { t + 1 } ^ { 2 }$ yields

$$
\sum _ { t = 0 } ^ { T - 1 } \gamma ^ { 2 } \sigma _ { t } ^ { 2 } \le V _ { \operatorname* { m a x } } ^ { 2 } + 2 V _ { \operatorname* { m a x } } ( R _ { \operatorname* { m a x } } T + G _ { T } ) + M _ { T } ,\tag{103}
$$

where

$$
M _ { T } : = \sum _ { t = 0 } ^ { T - 1 } \gamma ^ { 2 } \left( \mathbb { E } [ v _ { t + 1 } ^ { 2 } \mid \mathcal { F } _ { t } ] - v _ { t + 1 } ^ { 2 } \right)
$$

is a martingale. If $\begin{array} { r } { S _ { T } : = \sum _ { t } \gamma ^ { 2 } \sigma _ { t } ^ { 2 } } \end{array}$ , then

$$
\sum _ { t } \operatorname { V a r } \left( \gamma ^ { 2 } v _ { t + 1 } ^ { 2 } \mid \mathcal { F } _ { t } \right) \leq 4 V _ { \operatorname* { m a x } } ^ { 2 } S _ { T } .
$$

Indeed, $z \mapsto z ^ { 2 }$ is $2 V _ { \mathrm { m a x } }$ -Lipschitz on $[ 0 , V _ { \mathrm { m a x } } ]$ , so its conditional variance is at most $4 V _ { \operatorname* { m a x } } ^ { 2 } \sigma _ { t } ^ { 2 }$ Freedman’s inequality therefore gives

$$
M _ { T } \leq \sqrt { 8 V _ { \operatorname* { m a x } } ^ { 2 } S _ { T } \log ( 1 / \delta ) } + \frac { V _ { \operatorname* { m a x } } ^ { 2 } } { 3 } \log \frac { 1 } { \delta } .
$$

Substitute this into equation 103 and use $\sqrt { a b } \le a / 2 + b / 2$ to absorb $S _ { T } / 2$ into the left-hand side. This proves equation 101. □

Lemma 21 (Summing the optimistic recursion). On an event of probability at least $1 - \delta ,$

$$
\begin{array} { r l } & { G _ { T } \leq C \Bigg [ \displaystyle \sum _ { t = 0 } ^ { T - 1 } \bigl ( r _ { t } ^ { + } ( S _ { t } , A _ { t } ) - r ( S _ { t } , A _ { t } ) \bigr ) + \displaystyle \sum _ { t = 1 } ^ { T - 1 } \varepsilon _ { N _ { t } } ^ { D } } \\ & { \qquad + \displaystyle \sum _ { t = 1 } ^ { T - 1 } \gamma \alpha _ { t } \sigma _ { t } + \frac { V _ { \operatorname* { m a x } } } { 1 - \gamma } \displaystyle \sum _ { t = 1 } ^ { T - 1 } \beta _ { t } ^ { 2 } } \\ & { \qquad + V _ { \operatorname* { m a x } } E _ { T } + \frac { V _ { \operatorname* { m a x } } \log ( 1 / \delta ) } { 1 - \gamma } \Bigg ] . } \end{array}\tag{104}
$$

Proof. The first decision contributes at most $V _ { \mathrm { m a x } }$ , which can be absorbed into the last two terms. Sum equation 99 from $t = 1$ to $T - 1$ . Set

$$
Y _ { t } : = D _ { t } ( X _ { t + 1 } ) , \qquad \mu _ { t } = \mathbb { E } [ Y _ { t } \mid { \mathcal F } _ { t } ] .
$$

Within a planner epoch $\{ p , \ldots , q \} , D _ { t }$ is a fixed function, and hence

$$
\begin{array} { c } { { \displaystyle \sum _ { t = p } ^ { q } \left[ \gamma D _ { t } ( X _ { t + 1 } ) - D _ { t } ( X _ { t } ) \right] = - D _ { t } ( X _ { p } ) + D _ { t } ( X _ { q + 1 } ) - ( 1 - \gamma ) \sum _ { t = p } ^ { q } Y _ { t } } } \\ { { \leq V _ { \mathrm { m a x } } - ( 1 - \gamma ) \displaystyle \sum _ { t = p } ^ { q } Y _ { t } . } } \end{array}
$$

Summing over epochs and replacing $Y _ { t }$ by $\mu _ { t } .$ , the potential terms are therefore bounded by

$$
V _ { \mathrm { m a x } } E _ { T } + \gamma \sum _ { t } \mu _ { t } - \sum _ { t } Y _ { t } .\tag{105}
$$

Use Young’s inequality in the adaptive square-root term:

$$
C \gamma \beta _ { t } \sqrt { V _ { \operatorname* { m a x } } \mu _ { t } } \le \frac { 1 - \gamma } { 8 } \mu _ { t } \frac { C V _ { \operatorname* { m a x } } \beta _ { t } ^ { 2 } } { 1 - \gamma } .
$$

Let $\bar { \gamma } : = \gamma + ( 1 - \gamma ) / 8$ and apply Lemma 19 with $\eta = ( 1 - \gamma ) / 4$ . Since

$$
\bar { \gamma } ( 1 + \eta ) \leq 1 - \frac { 1 - \gamma } { 2 } ,
$$

we obtain

$$
\bar { \gamma } \sum _ { t } \mu _ { t } - \sum _ { t } Y _ { t } \leq - \frac { 1 - \gamma } { 2 } \sum _ { t } Y _ { t } \frac { C V _ { \operatorname* { m a x } } \log ( 1 / \delta ) } { 1 - \gamma } .
$$

Discarding the negative term and collecting the remaining contributions proves equation 104.

Completion ofthe proofofTheorem 3. Invoke all preceding confidence events with failure probabilities that sum to at most δ. By Cauchy–Schwarz,

$$
\sum _ { t = 1 } ^ { T - 1 } \gamma \alpha _ { t } \sigma _ { t } \leq \sqrt { A _ { T } \sum _ { t = 1 } ^ { T - 1 } \gamma ^ { 2 } \sigma _ { t } ^ { 2 } } .\tag{106}
$$

Lemmas 17 and 20 imply

$$
\sum _ { t = 1 } ^ { T - 1 } \gamma \alpha _ { t } \sigma _ { t } \leq C \sqrt { A _ { T } \left[ V _ { \operatorname* { m a x } } R _ { \operatorname* { m a x } } T + V _ { \operatorname* { m a x } } G _ { T } + V _ { \operatorname* { m a x } } ^ { 2 } \log \frac { 1 } { \delta } \right] } .\tag{107}
$$

The term involving $G _ { T }$ is absorbed through

$$
C \sqrt { A _ { T } V _ { \operatorname* { m a x } } G _ { T } } \leq \frac { 1 } { 4 } G _ { T } + C ^ { \prime } A _ { T } V _ { \operatorname* { m a x } } .
$$

We now substitute the reward bound equation 91, the coefficient and dictionary bounds equation $9 3 -$ equation 95, and the epoch bound equation 86 into equation 104. Together with equation 107 and $\bar { V _ { \operatorname* { m a x } } } = R _ { \operatorname* { m a x } } / ( 1 - \gamma )$ , this gives

$$
\mathrm { R e g } _ { \ell } ( T ) = G _ { T } \leq \widetilde { \cal O } \left( { \cal R } _ { \mathrm { m a x } } \sqrt { \frac { n m T } { 1 - \gamma } } + \frac { { \cal R } _ { \mathrm { m a x } } n ^ { 2 } m } { ( 1 - \gamma ) ^ { 2 } } \right) .\tag{108}
$$

For fixed $\ell ,$ all suppressed factors are logarithmic in $n , m , T , \delta ^ { - 1 }$ , and $( 1 - \gamma ) ^ { - 1 }$ . This proves the claimed high-probability regret bound. □

## B EXPERIMENTS

We evaluate our planner on the wind-farm storage-control benchmark of Lu et al. (2025), reusing their CAISO wind-generation forecasts and observations, their electricity-price series, their preprocessed arrays, their state and action discretisation, and their evaluation interval.

## C EXTENSION TO NOISY TRANSITION LOOK-AHEAD

This section extends DLA-UCB to locally corrupted transition look-ahead. We first specify the observation model and the correct comparator, and then give the complete algorithm and regret analysis. The reward-confidence construction is unchanged from Appendix A.3.

## C.1 NOISY TRANSITION LOOK-AHEAD

Recall that $\Omega = { \mathcal { S } } ^ { S \times { \mathcal { A } } }$ and that the transition tables are i.i.d. with product law

$$
{ \sf Q } ( \theta ) = \prod _ { ( s , a ) \in \mathcal { S } \times \mathcal { A } } P \big ( \theta ( s , a ) \mid s , a \big ) .\tag{109}
$$

For every $( s , a , s ^ { \prime } )$ , let $I ( \cdot \mid s , a , s ^ { \prime } )$ be a distribution on ${ \mathcal { S } } .$ Conditional on $\Theta _ { t } = \theta .$ , the noisy table $\widetilde { \Theta } _ { t }$ is sampled according to

$$
\mathbb { P } \Big ( \widetilde { \Theta } _ { t } = \widetilde { \theta } \mid \Theta _ { t } = \theta \Big ) = \prod _ { ( s , a ) \in S \times A } I \big ( \widetilde { \theta } ( s , a ) \mid s , a , \theta ( s , a ) \big ) .\tag{110}
$$

The pairs $( \Theta _ { t } , \widetilde { \Theta } _ { t } )$ are independent across time. Before choosing $A _ { t }$ , the learner observes

$$
\widetilde { C } _ { t } ^ { \ell } : = ( \widetilde { \Theta } _ { t } , \dots , \widetilde { \Theta } _ { t + \ell - 1 } ) ,\tag{111}
$$

whereas the true transition remains $S _ { t + 1 } = \Theta _ { t } ( S _ { t } , A _ { t } )$ . After acting, the learner observes only $( R _ { t } , S _ { t + 1 } )$ ; in particular, it never observes the complete table $\Theta _ { t }$

Let $\widetilde { \mathsf Q }$ be the marginal law of a noisy table. It is again a product distribution. The true successor conditional on the noisy entry is governed by

$$
\widetilde { P } ( s ^ { \prime } \mid s , a , z ) : = \mathbb { P } \big ( \Theta _ { t } ( s , a ) = s ^ { \prime } \mid \widetilde { \Theta } _ { t } ( s , a ) = z \big ) .\tag{112}
$$

Whenever the conditioning event has positive probability, Bayes’ rule gives

$$
\widetilde P ( s ^ { \prime } \mid s , a , z ) = \frac { P ( s ^ { \prime } \mid s , a ) I ( z \mid s , a , s ^ { \prime } ) } { \sum _ { u \in { \cal S } } P ( u \mid s , a ) I ( z \mid s , a , u ) } .\tag{113}
$$

The value of the posterior can be chosen arbitrarily on null conditioning events. The product assumptions imply the following identity

$$
\mathbb { P } \big ( \Theta _ { t } ( s , a ) = s ^ { \prime } \mid \widetilde { \Theta } _ { t } = \widetilde { \theta } \big ) = \widetilde { P } \big ( s ^ { \prime } \mid s , a , \widetilde { \theta } ( s , a ) \big ) .\tag{114}
$$

For $V : \mathcal { S } \times \Omega ^ { \ell } \to [ 0 , V _ { \mathrm { m a x } } ]$ , the Bellman operator is consequently

$$
\begin{array} { r l } & { ( \widetilde { \mathcal { T } } _ { \ell } V ) ( s , \widetilde { \theta } _ { 0 : \ell - 1 } ) } \\ & { \quad : = \underset { a \in A } { \operatorname* { m a x } } \Biggl \{ r ( s , a ) + \gamma \sum _ { s ^ { \prime } \in S } \widetilde { P } \bigl ( s ^ { \prime } \mid s , a , \widetilde { \theta } _ { 0 } ( s , a ) \bigr ) \mathbb { E } _ { \widetilde { \Theta } \sim \widetilde { \Theta } } \left[ V ( s ^ { \prime } , \widetilde { \theta } _ { 1 : \ell - 1 } , \widetilde { \Theta } ) \right] \Biggr \} . } \end{array}\tag{115}
$$

It is a monotone γ-contraction. Its fixed point and associated action scores are denoted by $\widetilde { V } _ { \ell } ^ { \star }$ and $\widetilde { Q } _ { \ell } ^ { \star }$ . The oracle observes the same noisy windows as the learner; it does not observe the latent tables. We measure regret by

$$
\widetilde { \mathrm { R e g } } _ { \ell } ( T ) : = \sum _ { t = 0 } ^ { T - 1 } \left[ \widetilde { V } _ { \ell } ^ { \star } ( S _ { t } , \widetilde { C } _ { t } ^ { \ell } ) - \widetilde { Q } _ { \ell } ^ { \star } ( S _ { t } , \widetilde { C } _ { t } ^ { \ell } , A _ { t } ) \right] .\tag{116}
$$

## C.2 NEAR-OPTIMAL PLANNING

Suppose first that one can sample from $\widetilde { \mathsf Q }$ and evaluate ${ \widetilde { P } } .$ . Draw once and for all

$$
\widetilde { \Theta } ^ { 1 } , \ldots , \widetilde { \Theta } ^ { N } \stackrel { \mathrm { i . i . d . } } { \sim } \widetilde { \mathsf { Q } } .\tag{117}
$$

On $\mathcal { C } _ { N } = \mathcal { S } \times [ N ] ^ { \ell }$ , define

$$
\begin{array} { r l r } {  { ( \widetilde { T } _ { N } v ) ( s , i _ { 0 } , \ldots , i _ { \ell - 1 } ) } } \\ & { } & { \quad : = \operatorname* { m a x } _ { a \in \mathcal { A } } \Biggl \{ r ( s , a ) + \gamma \sum _ { s ^ { \prime } \in \mathcal { S } } \widetilde { P } \bigl ( s ^ { \prime } \mid s , a , \widetilde { \Theta } ^ { i _ { 0 } } ( s , a ) \bigr ) \frac { 1 } { N } \sum _ { j = 1 } ^ { N } v ( s ^ { \prime } , i _ { 1 } , \ldots , i _ { \ell - 1 } , j ) \Biggr \} . } \end{array}\tag{118}
$$

Starting from $\widetilde V ^ { 0 } = 0 .$ , run $\widetilde { V } ^ { k + 1 } = \widetilde { T } _ { N } \widetilde { V } ^ { k }$

For an arbitrary observed window $\widetilde { c } = ( \widetilde { \theta } _ { 0 } , \dots , \widetilde { \theta } _ { \ell - 1 } )$ , set

$$
\widetilde { U } _ { \ell , \tilde { c } } ^ { V } ( s , i _ { 0 } , \ldots , i _ { \ell - 1 } ) : = V ( s , i _ { 0 } , \ldots , i _ { \ell - 1 } ) ,\tag{119}
$$

and, for $k = \ell - 1 , \ldots , 0$ , define

$$
\begin{array} { r l r } {  { \widetilde { U } _ { k , \widetilde { c } } ^ { V } ( s , i _ { 0 } , \dots , i _ { k - 1 } ) } } \\ & { } & { \quad : = \operatorname* { m a x } _ { a \in A } \Biggl \{ r ( s , a ) + \gamma \sum _ { s ^ { \prime } \in S } \widetilde { P } ( s ^ { \prime } \mid s , a , \widetilde { \theta } _ { k } ( s , a ) ) \frac { 1 } { N } \sum _ { i _ { k } = 1 } ^ { N } \widetilde { U } _ { k + 1 , \widetilde { c } } ^ { V } ( s ^ { \prime } , i _ { 0 } , \dots , i _ { k } ) \Biggr \} . } \end{array}\tag{120}
$$

At $k = 0$ the index list is empty. Keeping the root action fixed defines $\widetilde { Q } _ { v } ( s , \widetilde { c } , a )$ , and the returned policy is greedy with respect to $\smash { \widetilde { Q } } _ { \widetilde { v } ^ { K } }$

Theorem 4 (RPTAS for noisy transition look-ahead). Fix $\ell \geq 2$ and $\varepsilon , \delta \in ( 0 , 1 )$ . For suitable choices of the sample size $N$ and the number of value-iteration steps $K$ , the algorithm described above computes a policy $\widetilde { \pi } _ { D }$ such that

$$
\mathbb { P } \Big ( \widetilde { V } _ { \ell } ^ { \widetilde { \pi } _ { D } } \big ( s , \widetilde { c } \big ) \geq \widetilde { V } _ { \ell } ^ { \star } \big ( s , \widetilde { c } \big ) - \varepsilon V _ { \mathrm { m a x } } , \qquad \forall ( s , \widetilde { c } ) \in \mathcal { S } \times \Omega _ { \mathbf { \varepsilon } } ^ { \ell } \Big ) \geq 1 - \delta .\tag{121}
$$

For every fixed ℓ, computing and storing the policy, as well as selecting an action at each decision time, require time and memory polynomial in n, $\dot { m , \varepsilon ^ { - 1 } }$ , log(1/δ), and $( 1 - \gamma ) ^ { - 1 }$

Proof. Let $\begin{array} { r } { \widehat { \tilde { \mathsf Q } } _ { N } = N ^ { - 1 } \sum _ { j = 1 } ^ { N } \delta _ { \widetilde \Theta ^ { j } } } \end{array}$ . For each $s ^ { \prime }$ and noisy suffix, apply Hoeffding’s inequality to $\widetilde { \Theta } \mapsto \widetilde { V } _ { \ell } ^ { \star } ( s ^ { \prime } , \widetilde { \theta } _ { 1 : \ell - 1 } , \widetilde { \Theta } )$ and take a union bound over the $n | \Omega | ^ { \ell - 1 } = n ^ { 1 + ( \ell - 1 ) n m }$ possible functions. Set $\eta = \varepsilon ( 1 - \gamma ) ^ { 2 } V _ { \mathrm { m a x } } / 1 6$ . A sufficient choice is

$$
N \geq \frac { 1 2 8 } { \varepsilon ^ { 2 } ( 1 - \gamma ) ^ { 4 } } \left( \log \frac 2 \delta [ 1 + ( \ell - 1 ) n m ] \log n \right) .\tag{122}
$$

Then, with probability at least $1 - \delta ,$ , all the corresponding empirical and true expectations differ by at most $\eta .$ Averaging with respect to the posterior $\widetilde { P } ( \cdot \mid s , a , z )$ is a convex combination and does not enlarge this error. The fixed-point perturbation argument used in the proof of Theorem 2 therefore gives

$$
\| \widehat { \widetilde { V } } ^ { \star } - \widetilde { V } _ { \ell } ^ { \star } \| _ { \infty } \leq \frac { \gamma \eta } { 1 - \gamma } , \qquad \| \widehat { \widetilde { Q } } ^ { \star } - \widetilde { Q } _ { \ell } ^ { \star } \| _ { \infty } \leq \frac { \gamma \eta } { 1 - \gamma } .
$$

The restriction of the empirical Bellman equation to dictionary windows is exactly equation 118. Thus Lemma 6 applies with the deterministic successor replaced by the posterior average. The same one-line replacement in the backward induction of Lemma 7 proves that equation 120 recovers the empirical action scores on every arbitrary window. Likewise, Lemma 8 is unchanged because posterior averaging is 1-Lipschitz. Consequently,

$$
\operatorname* { s u p } _ { s , \tilde { c } , a } \left| \widetilde Q _ { \widetilde V ^ { \kappa } } ( s , \widetilde c , a ) - \widetilde Q _ { \ell } ^ { \star } ( s , \widetilde c , a ) \right| \leq \frac { \gamma \eta } { 1 - \gamma } + \gamma ^ { K + \ell } V _ { \operatorname* { m a x } } .\tag{123}
$$

It is enough to take

$$
K \geq \frac { 1 } { 1 - \gamma } \log \frac { 3 2 } { \varepsilon ( 1 - \gamma ) } .\tag{124}
$$

The choices in equation 122–equation 124 make the right-hand side at most $\varepsilon ( 1 - \gamma ) V _ { \mathrm { { m a x } } } / 2$ . Applying the approximate-greedy argument used in the proof of Theorem 2 proves equation 121.

After the dictionary averages have been precomputed, one sweep of equation 118 costs $O ( n ( n m +$ $1 ) N ^ { \ell } )$ operations, and the backward extension costs $O ( \ell n ( n \dot { m } + 1 ) \dot { N } ^ { \ell } )$ . Relative to the perfect planner, the posterior sum introduces only one additional factor n. □

## C.3 THE NOISY VERSION OF DLA-UCB

Theorem 5 (Regret of NOISY-DLA-UCB). Assume equation 109– equation 110 and the stochastic reward model of Section 5. Fix the look-ahead depth ℓ. For every $\overset { \cdot } { T } \geq 2$ and $\delta \in \mathsf { \Gamma } ( 0 , 1 )$ , with probability at least $1 - \delta ,$ , the algorithm described below, NOISY-DLA-UCB, satisfies

$$
\widetilde { \mathrm { R e g } } _ { \ell } ( T ) = \widetilde { O } _ { \ell } \left( R _ { \mathrm { m a x } } n \sqrt { \frac { m T } { 1 - \gamma } } + \frac { R _ { \mathrm { m a x } } n ^ { 3 } m } { ( 1 - \gamma ) ^ { 2 } } \right) .\tag{125}
$$

Here $\widetilde { O } _ { \ell }$ hides factors logarithmic in $n , m , T , \delta ^ { - 1 }$ , and $( 1 - \gamma ) ^ { - 1 }$ , and constants depending only on thefixed depth ℓ. Forfixed ℓ, the algorithm has polynomial update time, representation size, and per-decision computation.

Algorithmic modifications. The observed noisy entry no longer determines the physical successor. Relative to DLA-UCB, the learner must therefore estimate the conditional law of that successor given the current state, action, and noisy entry. A synthetic dictionary element must consequently contain two objects: a noisy table, which is appended to the look-ahead window, and, for every possible local context, a physical successor sampled from the corresponding estimated conditional law. These two objects jointly simulate the next augmented state. The reward estimates and their doubling schedule are unchanged.

At an update based on the first N expired tables, define, for $h = ( s , a , z ) \in \mathcal S \times \mathcal A \times \mathcal S .$

$$
k _ { N } ( h ) : = \sum _ { t = 0 } ^ { N - 1 } \mathbf { 1 } \Bigl \{ S _ { t } = s , ~ A _ { t } = a , ~ \widetilde { \Theta } _ { t } ( s , a ) = z \Bigr \} ,\tag{126}
$$

$$
\widehat { \widetilde { P } } _ { N } ( s ^ { \prime } \mid h ) : = \frac { \sum _ { t = 0 } ^ { N - 1 } \mathbf { 1 } \Big \{ S _ { t } = s , ~ A _ { t } = a , ~ \widetilde { \Theta } _ { t } ( s , a ) = z , ~ S _ { t + 1 } = s ^ { \prime } \Big \} } { k _ { N } ( h ) } , \qquad k _ { N } ( h ) > 0 .\tag{127}
$$

When $k _ { N } ( h ) = 0$ , set $\widehat { \widetilde { P } } _ { N } ( \cdot \mid h )$ to the uniform distribution. This choice is immaterial because the continuation estimate used at an unvisited context is clipped at $V _ { \mathrm { m a x } }$

Every noisy table is observed in full. Its product law is therefore estimated from all expired tables as

$$
\widehat { \tilde { \mathsf { Q } } } _ { N } : = \bigotimes _ { ( s , a ) \in { \cal S } \times { \cal A } } \left( \frac { 1 } { N } \sum _ { t = 0 } ^ { N - 1 } \delta _ { \widetilde { \Theta } _ { t } ( s , a ) } \right) .\tag{128}
$$

This distribution can be sampled coordinatewise, without enumerating Ω.

As in the perfect-look-ahead case, every planner update constructs a fresh dictionary. Set

$$
M _ { N } : = \left\lceil \frac { N } { ( 1 - \gamma ) ^ { 3 } } \right\rceil , \qquad K _ { N } : = \left\lceil \frac { 1 } { 1 - \gamma } \log \frac { 2 N ^ { 2 } } { 1 - \gamma } \right\rceil .\tag{129}
$$

Conditionally on the observations available at the update, independently for $j = 1 , \dots , M _ { N }$ , draw

$$
\widetilde { Z } ^ { j } \sim \widehat { \widetilde { \Omega } } _ { N } , \qquad Y ^ { j } ( h ) \sim \widehat { \widetilde { P } } _ { N } ( \cdot \mid h ) \quad \mathrm { f o r } \mathrm { e v e r y } h \in { \mathcal { S } } \times A \times { \mathcal { S } } .\tag{130}
$$

All draws in equation 130 are mutually independent conditional on the data. Thus, when the current local context is h, the jth simulated next augmented state uses $Y ^ { j } ( h )$ as its physical state and $\widetilde { Z } ^ { j }$ as the new table appended to the window.

The rest of the implementation mirrors DLA-UCB. Value iteration is performed on $\mathcal { S } \times [ M _ { N } ] ^ { \ell }$ with the compound successors in equation 130 replacing the deterministic successors of the perfectlook-ahead dictionary. At decision time, the same backward extension is applied to the observed noisy window. The exact optimistic continuation rule used in these backups is specified in the proof; no additional online exploration step is performed.

Algorithm 5 NOISY-DLA-UCB-UPDATE   
Input: Number N of expired tables, noisy tables $\widetilde { \Theta } _ { 0 : N - 1 }$ , transition and reward observations, pre  
vious reward bound $\bar { r } _ { \mathrm { o l d } } ^ { + }$ , horizon T, confidence level δ   
Output: Planning object $\overrightharpoon { D }$   
1: Compute $\widehat { \widetilde { \mathsf { Q } } } _ { N }$ using equation 128   
2: for every $h \in \mathcal { S } \times \mathcal { A } \times \mathcal { S }$ do   
3: Compute $k _ { N } ( h )$ and $\widehat { \widetilde { P } } _ { N } ( \cdot \mid h )$ using equation 126–equation 127   
4: end for   
5: Update $r ^ { + }$ exactly as in Algorithm 3   
6: Compute $M _ { N }$ and $K _ { N }$ from equation 129   
7: Draw the compound dictionary according to equation 130   
8: Construct the optimistic dictionary backup specified in the proof below   
9: Initialize $v ^ { 0 } \equiv \dot { V } _ { \mathrm { m a x } }$ on $\mathcal { S } \times [ M _ { N } ] ^ { \ell }$   
10: for $k = 0 , \ldots , K _ { N } - 1$ do   
11: Apply one optimistic dictionary Bellman backup to obtain $v ^ { k + 1 }$ from $v ^ { k }$   
12: end for   
13: return $\mathcal { D } = ( N , \widetilde { Z } ^ { 1 : M _ { N } } , ( Y ^ { 1 : M _ { N } } ( h ) ) _ { h } , v ^ { K _ { N } } , r ^ { + } , k _ { N } )$

Algorithm 6 NOISY-DLA-UCB-ONLINE   
Input: Horizon T, look-ahead depth $\ell ,$ confidence level δ   
Output: Actions $A _ { 0 } , \ldots , A _ { T - 1 }$   
1: Set $\mathcal { D }  \emptyset$ and $r ^ { + } ( s , a ) \gets R _ { \mathrm { m a x } }$ for every $( s , a )$   
2: Observe $X _ { 0 } = ( S _ { 0 } , \widetilde { C } _ { 0 } ^ { \ell } )$ and play an arbitrary action $A _ { 0 }$   
3: Observe $( R _ { 0 } , S _ { 1 } )$ and update the reward statistics and the conditional-transition statistics of   
$h _ { 0 } = ( S _ { 0 } , A _ { 0 } , \widetilde { \Theta } _ { 0 } ( S _ { 0 } , A _ { 0 } ) )$   
4: for $t = 1 , \dots , T - 1$ do   
5: Observe $X _ { t } = ( S _ { t } , \widetilde { C } _ { t } ^ { \ell } )$   
6: if $\mathcal { D } = \mathcal { D } ,$ t is a power of two, or a reward or local conditional-transition count has just   
entered $\{ 1 , 2 , 4 , \ldots \}$ then   
7: Set $N \gets t$   
8: D ← NOISY-DOLAR-UPDATE $( N , \widetilde { \Theta } _ { 0 : N - 1 } ,$ past observations, $r ^ { + } , T , \delta )$   
9: Extract the updated reward bound $r ^ { + }$ from D   
10: end if   
11: Compute the action scores by backward extension from $\mathcal { D }$ to $\widetilde { C } _ { t } ^ { \ell }$   
12: Play an action maximizing these scores   
13: Observe $( R _ { t } , S _ { t + 1 } )$ and update the reward statistics and the conditional-transition statistics   
of $h _ { t } = ( S _ { t } , A _ { t } , \widetilde { \Theta } _ { t } ( S _ { t } , A _ { t } ) )$   
14: end for

The global table count, each of the nm reward counts, and each of the $n ^ { 2 } m$ local conditionaltransition counts can enter $\{ 1 , 2 , 4 , \ldots \}$ at most $1 + \lceil \log _ { 2 } T \rceil$ times. Hence the number of planner updates is $O ( n ^ { 2 } m \log T )$ . An update based on N observations costs

$$
O \left( N n m + M _ { N } n ^ { 2 } m + K _ { N } n ^ { 2 } m M _ { N } ^ { \ell } \right) ,\tag{131}
$$

and a backward extension costs $O ( \ell n ^ { 2 } m M _ { N } ^ { \ell } )$ per decision. The planning object occupies $O ( M _ { N } n ^ { 2 } m + n M _ { N } ^ { \ell } )$ memory. As in the perfect case, these bounds use precomputed means and standard deviations over the last dictionary index.

## C.3.1 PROOF OF THEOREM 5

We analyze the fixed point of the compound-dictionary operator used by the algorithm, rather than introducing an ideal optimistic fixed point on all of $\mathcal { S } \times \hat { \Omega } ^ { \ell }$ . The latter route would require a uniform concentration bound over all noisy suffixes and would lose the dimension dependence in equation 125. The proof instead establishes optimism only on the finite computation graph visited by the core and the backward extensions. Stability then controls the dictionary-dependent fixed point away from that graph, and a separate lemma bounds the expected negative part of its gap to $\widetilde { V } _ { \ell } ^ { \star }$

The reward argument is unchanged, so Lemma 16 will be used directly. The scalar monotonicity argument of Lemma 12 and the restriction and extension arguments of Lemmas 6–8 will also be reused below; replacing a deterministic successor by an average over compound successors does not affect those arguments.

Confidence bookkeeping. The number of planning epochs is deterministically bounded by

$$
{ \overline { { E } } } _ { T } : = 2 + { \bigl ( } 1 + n m + n ^ { 2 } m { \bigr ) } { \bigl ( } 1 + \bigl \lceil \log _ { 2 } T { \bigr \rceil } { \bigr ) } .\tag{132}
$$

Let

$$
M _ { \mathrm { m a x } } : = \left\lceil \frac { T } { ( 1 - \gamma ) ^ { 3 } } \right\rceil\tag{133}
$$

and fix

$$
L _ { T } : = \operatorname* { m a x } \left\{ 1 , \ell , \log \left( \frac { 1 2 8 ( \ell + 1 ) T ^ { 3 } \overline { { E } } _ { T } n ^ { 3 } m ( M _ { \mathrm { m a x } } + 1 ) ^ { \ell } } { \delta } \right) \right\} .\tag{134}
$$

For fixed $\ell , L _ { T }$ is logarithmic in the problem parameters. Its slack covers all union bounds below;   
the precise numerical constant is irrelevant.

Lemma 22 (Posterior sample streams). Conditionally on the complete noisy-table sequence $( \widetilde { \Theta } _ { t } ) _ { t \geq 0 } ,$ , the successors observed on successive visits to a fixed $h = ( s , a , z )$ are independent with common law ${ \widetilde { P } } ( \cdot \mid h )$ . The statement remains true after conditioning on synthetic noisy tables sampled independently of the latent successors.

Proof. By equation 114, conditional on the noisy tables and the past, the unrevealed coordinate $\Theta _ { t } { \left( s , a \right) }$ has law $\widetilde { P } ( \cdot \mid s , a , \widetilde { \Theta } _ { t } ( s , a ) )$ . Whether the algorithm visits h at time t is decided before this coordinate is revealed. Optional skipping of the conditionally independent latent coordinates gives the claimed i.i.d. stream. Independent synthetic noisy tables do not alter this conditional law.

The following comparison is deliberately coordinatewise. It will be applied to functions selected after seeing the data.

Lemma 23 (Uniform posterior comparison). With probability at least $1 - \delta / 8 ,$ , simultaneously for every update, every local context $h ,$ every d : ${ S }  [ 0 , { V } _ { \mathrm { m a x } } ] ,$ , and every $\eta \in ( 0 , 1 ]$

$$
\left| \left( \widehat { \widetilde { P } } _ { N } ( \cdot  { | } h ) - \widetilde { P } ( \cdot  { | } h ) \right) d \right| \leq \eta \widetilde { P } ( \cdot  { | } h ) d + \frac { C V _ { \operatorname* { m a x } } n L _ { T } } { \eta ( 1 \vee k _ { N } ( h ) ) } .\tag{135}
$$

The inequality remains valid when d is data-dependent.

Proof. By Lemma 22, scalar Bernstein and a union bound over $h , s ^ { \prime } .$ , and all possible local sample counts give

$$
\left| \widehat { \widetilde { P } } _ { N } ( s ^ { \prime } \mid h ) - \widetilde { P } ( s ^ { \prime } \mid h ) \right| \leq \sqrt { \frac { 2 \widetilde { P } ( s ^ { \prime } \mid h ) L _ { T } } { 1 \vee k _ { N } ( h ) } + \frac { 2 L _ { T } } { 3 ( 1 \vee k _ { N } ( h ) ) } } .
$$

This event is coordinatewise and hence uniform over d. Multiplication by $d ( s ^ { \prime } )$ , summation over $s ^ { \prime }$ Cauchy–Schwarz, and Young’s inequality give equation 135. □

For an update using N expired tables, set

$$
q _ { N } : = \operatorname* { m i n } \left\{ 1 , \sqrt { \frac { n ^ { 2 } m \log ( N + 1 ) + L _ { T } } { 2 N } } \right\} .\tag{136}
$$

Lemma 24 (Uniform control of the noisy-table law). With probability at least $1 - \delta / 8 ,$ , simultaneouslyfor every sample size used by the algorithm,

$$
H \biggl ( \widehat { \widetilde { \mathsf { Q } } } _ { N } , \widetilde { \mathsf { Q } } \biggr ) \leq q _ { N } .\tag{137}
$$

Consequently,for every possibly data-dependent $f : \Omega \to [ 0 , V _ { \mathrm { m a x } } ] ;$

$$
\left| \mathbb { E } _ { \widehat { \widetilde { \mathbb { Q } } } _ { N } } f - \mathbb { E } _ { \widetilde { \mathbb { Q } } } f \right| \leq 2 \sqrt { 2 } q _ { N } \sqrt { \mathrm { V a r } _ { \widetilde { \mathbb { Q } } } ( f ) } + 2 V _ { \operatorname* { m a x } } q _ { N } ^ { 2 } ,\tag{138}
$$

$$
\left| \sqrt { \mathrm { V a r } _ { \widehat { \widetilde { \mathbf { Q } } } _ { N } } ( f ) } - \sqrt { \mathrm { V a r } _ { \widetilde { \mathbf { Q } } } ( f ) } \right| \leq 2 V _ { \operatorname* { m a x } } q _ { N } .\tag{139}
$$

Proof. For one $N ,$ , the method of types and product tensorization give

$$
\begin{array} { r } { \mathbb { P } \bigg ( \mathrm { K L } \bigg ( \widehat { \tilde { \ Q } } _ { N } \bigg | \bigg | \widetilde { \mathbf Q } \bigg ) > x \bigg ) \leq ( N + 1 ) ^ { n ^ { 2 } m } e ^ { - N x } . } \end{array}
$$

Use $2 H ^ { 2 } \le \mathrm { K L }$ and take a union bound over $N \leq T$ . The definition of $q _ { N }$ and the slack in $L _ { T }$ prove equation 137. The deterministic Hellinger mean and standard-deviation inequalities proved in Lemma 11 then give equation 138– equation 139. □

We next identify the finite collection on which scalar Bernstein confidence is required. This construction is used only in the proof. Condition on the complete noisy-table sequence and, for every $N \in \{ 1 , \ldots , T \}$ , pre-sample the noisy-table part $\widetilde { Z } _ { N } ^ { 1 : M _ { N } }$ of a possible dictionary from $\widehat { \widetilde { \mathsf { Q } } } _ { N }$ . For every $t < T$ , every $k \in \{ 0 , \ldots , \ell \}$ , and every $( i _ { 0 } , \dots , i _ { k - 1 } ) \in [ M _ { N } ] ^ { k }$ , include the window

$$
( \widetilde { \Theta } _ { t + k } , \dots , \widetilde { \Theta } _ { t + \ell - 1 } , \widetilde { Z } _ { N } ^ { i _ { 0 } } , \dots , \widetilde { Z } _ { N } ^ { i _ { k - 1 } } )\tag{140}
$$

in ${ \mathfrak { G } } _ { T }$ , with the first block empty when $k = \ell .$ . Thus ${ \mathfrak { G } } _ { T }$ contains every observed window, every dictionary window, and every hybrid window used by any possible backward extension. Moreover,

$$
\begin{array} { r } { | \mathfrak { G } _ { T } | \leq ( \ell + 1 ) T ^ { 2 } ( M _ { \mathrm { m a x } } + 1 ) ^ { \ell } . } \end{array}\tag{141}
$$

Crucially, ${ \mathfrak { G } } _ { T }$ depends on the noisy tables and the synthetic noisy-table draws, but not on the observed posterior-successor streams or the synthetic successors $Y ^ { j } ( h )$ . Pre-sampling is only a coupling device; the algorithm draws and reveals a dictionary only when an update occurs.

For $h = ( s , a , z )$ , define, when first needed,

$$
\mathcal { P } _ { h } : = \widetilde { P } ( \cdot \mid h ) \otimes \widetilde { \mathsf { Q } } , \qquad \widehat { \mathcal { P } } _ { N , h } : = \widehat { \widetilde { P } } _ { N } ( \cdot \mid h ) \otimes \widehat { \widetilde { \mathsf { Q } } } _ { N } .\tag{142}
$$

These are the true and estimated conditional laws of the physical successor and the fresh noisy table. Also set

$$
a _ { N } ( h ) : = \operatorname* { m i n } \left\{ 1 , \sqrt { \frac { L _ { T } } { 1 \vee k _ { N } ( h ) } } \right\} .\tag{143}
$$

Lemma 25 (Confidence for optimal continuations on the relevant graph). With probability at least $1 - \delta / 8 ,$ simultaneously for every update, every $h ,$ every $\widetilde { c } \in \mathfrak { G } _ { T }$ , and the continuation

$$
V ( s ^ { \prime } , \widetilde { z } ) : = \widetilde { V } _ { \ell } ^ { \star } ( s ^ { \prime } , \widetilde { c } _ { 1 : \ell - 1 } , \widetilde { z } ) ,
$$

we have

$$
\begin{array} { r l } & { \Big | \mathbb { E } _ { \widehat { \mathcal { P } } _ { N , h } } V - \mathbb { E } _ { \mathcal { P } _ { h } } V \Big | \leq C \Big [ ( a _ { N } ( h ) + q _ { N } ) \sqrt { \mathrm { V a r } _ { \widehat { \mathcal { P } } _ { N , h } } ( V ) } } \\ & { \qquad + V _ { \operatorname* { m a x } } ( a _ { N } ( h ) + q _ { N } ) ^ { 2 } \Big ] , } \end{array}\tag{144}
$$

and

$$
\begin{array} { r } { \left| \sqrt { \operatorname { V a r } _ { \widehat { \mathcal { P } } _ { N , h } } ( V ) } - \sqrt { \operatorname { V a r } _ { \mathcal { P } _ { h } } ( V ) } \right| \leq C V _ { \operatorname* { m a x } } ( a _ { N } ( h ) + q _ { N } ) . } \end{array}\tag{145}
$$

Proof. Fix h and $\widetilde { c } \in \mathfrak { G } _ { T }$ . Conditional on the noisy tables and on the synthetic noisy tables that occur in c, the continuation is fixed with respect to the posterior sample stream. We use the following elementary conditional-mixture version of scalar empirical Bernstein. If $S _ { 1 } , \ldots , S _ { k }$ are i.i.d. from $p , \widehat { p }$ is their empirical law, K is a fixed Markov kernel, and $f \in [ 0 , V _ { \operatorname* { m a x } } ]$ , then

$$
\begin{array} { r } { | \widehat { p } K f - p K f | \leq C \left[ a \sqrt { \mathrm { V a r } _ { \widehat { p } K } ( f ) } + V _ { \operatorname* { m a x } } a ^ { 2 } \right] , } \end{array}
$$

$$
\begin{array} { r } { \left| \sqrt { \mathrm { V a r } _ { \widehat { p } K } ( f ) } - \sqrt { \mathrm { V a r } _ { p K } ( f ) } \right| \leq C V _ { \operatorname* { m a x } } a , } \end{array}
$$

with the usual empirical-Bernstein probability, where $a = 1 \land \sqrt { x / ( 1 \lor k ) }$ . The first inequality follows by applying scalar empirical Bernstein to $s ^ { \prime } \mapsto K f ( s ^ { \prime } )$ and observing that $\begin{array} { r } { \mathrm { V a r } _ { \widehat { p } } ( \bar { K } f ) \ \leq } \end{array}$ $\operatorname { V a r } _ { \widehat { P } \widehat { K } } ( f )$ . For the second, apply Bernstein to

$$
s ^ { \prime } \longmapsto K \left[ ( f - p K f ) ^ { 2 } \right] ( s ^ { \prime } ) .
$$

Its variance is at most $V _ { \operatorname* { m a x } } ^ { 2 } \operatorname { V a r } _ { p K } ( f )$ ; the identity $\begin{array} { r } { \mathrm { V a r } ( f ) \ : = \ : \operatorname* { i n f } _ { c } p K [ ( f - c ) ^ { 2 } ] } \end{array}$ , used in both directions, then gives the displayed standard-deviation comparison.

Apply this fact conditionally with $p = \widetilde { P } ( \cdot \mid h ) , \widehat { p } = \widehat { \widetilde { P } } _ { N } ( \cdot \mid h )$ , and $K = \widehat { \widetilde { \mathsf { Q } } } _ { N }$ . It controls the first step in

$$
\widehat { \widetilde { P } } _ { N } \otimes \widehat { \widetilde { \mathbf { Q } } } _ { N } \longrightarrow \widetilde { P } \otimes \widehat { \widetilde { \mathbf { Q } } } _ { N } \longrightarrow \widetilde { P } \otimes \widetilde { \mathbf { Q } }
$$

with coefficient $a _ { N } ( h )$ . For the second step, tensoring both noisy-table laws with the same posterior leaves their Hellinger distance unchanged. Hence Lemma 24 controls its mean and standard deviation with coefficient $q _ { N }$ . The triangle inequality and $( a _ { N } ( h ) + q _ { N } ) ^ { 2 } \geq 2 a _ { N } ( h ) q _ { N }$ give equation 144– equation 145.

The union bound is over local contexts, possible local sample counts, and the graph in equation 141; it is covered by $L _ { T }$ □

The compound-dictionary operator. We now give the continuation rule referred to in Algorithms $5 - 6$ . Write D for the collection of compound samples in equation 130. At an update based on N observations, define

$$
d _ { N } : = \operatorname* { m i n } \left\{ 1 , \sqrt { \frac { L _ { T } } { M _ { N } } } \right\} , \qquad \lambda _ { N } ( h ) : = \operatorname* { m i n } \left\{ 1 , c _ { 0 } \big ( a _ { N } ( h ) + q _ { N } + d _ { N } \big ) \right\} ,\tag{146}
$$

where $c _ { 0 }$ is a sufficiently large universal numerical constant. For $\begin{array} { r c l } { y } & { = } & { \left( y _ { 1 } , \dots , y _ { M _ { N } } \right) \in } \end{array}$ $[ 0 , V _ { \mathrm { m a x } } ] ^ { \mathrm { ' } M _ { N } }$ , let

$$
{ \overline { { y } } } : = \frac { 1 } { M _ { N } } \sum _ { j = 1 } ^ { M _ { N } } y _ { j } , \qquad s ( y ) : = \sqrt { \frac { 1 } { M _ { N } } \sum _ { j = 1 } ^ { M _ { N } } ( y _ { j } - { \overline { { y } } } ) ^ { 2 } } ,
$$

and set

$$
\begin{array} { r } { U _ { N } ^ { D } ( y ; h ) : = \operatorname* { m i n } \left\{ V _ { \operatorname* { m a x } } , \overline { { y } } + \operatorname* { m a x } \left\{ 7 \lambda _ { N } ( h ) s ( y ) , 4 9 V _ { \operatorname* { m a x } } \lambda _ { N } ^ { 2 } ( h ) \right\} \right\} . } \end{array}\tag{147}
$$

For a core array $v : \mathcal { S } \times [ M _ { N } ] ^ { \ell } \to [ 0 , V _ { \mathrm { m a x } } ]$ , a core state $x = ( s , i _ { 0 } , \dotsc , i _ { \ell - 1 } )$ , and $a \in { \mathcal { A } }$ , put

$$
h _ { x } ( a ) : = \big ( s , a , \widetilde { Z } ^ { i _ { 0 } } ( s , a ) \big ) , \qquad y _ { j } ( v ; x , a ) : = v \big ( Y ^ { j } ( h _ { x } ( a ) ) , i _ { 1 } , \ldots , i _ { \ell - 1 } , j \big ) .
$$

The Bellman update used by Algorithm 5 is

$$
\begin{array} { r l } {  { ( \widetilde { T } _ { N , r ^ { + } } ^ { D , + } v ) ( x ) } } \\ & { \mathrel { \phantom { = } } \operatorname* { m a x } \operatorname* { m i n } \Big \{ V _ { \mathrm { m a x } } , r ^ { + } ( s , a ) + \gamma U _ { N } ^ { D } \big ( ( y _ { j } ( v ; x , a ) ) _ { j = 1 } ^ { M _ { N } } ; h _ { x } ( a ) \big ) \Big \} . } \end{array}\tag{148}
$$

For completeness, its backward extension to an arbitrary observed window $\widetilde { c } = ( \widetilde { \theta } _ { 0 } , \dots , \widetilde { \theta } _ { \ell - 1 } )$ defined as follows. Set

$$
U _ { \ell , \tilde { c } } ^ { N , D , v } ( s , i _ { 0 } , \ldots , i _ { \ell - 1 } ) : = v ( s , i _ { 0 } , \ldots , i _ { \ell - 1 } ) .\tag{149}
$$

For $k = \ell - 1 , \ldots , 1$ , let

$$
\begin{array} { r l } & { U _ { k , \tilde { c } } ^ { N , D , v } ( s , i _ { 0 } , \ldots , i _ { k - 1 } ) } \\ & { \quad : = \underset { a \in A } { \operatorname* { m a x } } \operatorname* { m i n } \left\{ V _ { \operatorname* { m a x } } , r ^ { + } ( s , a ) + \gamma U _ { N } ^ { D } \left( \left( U _ { k + 1 , \tilde { c } } ^ { N , D , v } ( Y ^ { j } ( h ) , i _ { 0 } , \ldots , i _ { k - 1 } , j ) \right) _ { j = 1 } ^ { M _ { N } } ; h \right) \right\} , } \end{array}\tag{150}
$$

where $h = ( s , a , \widetilde { \theta } _ { k } ( s , a ) )$ ) in this display. Keeping the root action fixed gives the deployed score

$$
\begin{array} { r l } & { \widetilde { Q } _ { N , r ^ { + } } ^ { D , v } ( s , \widetilde { c } , a ) } \\ & { \qquad : = \operatorname* { m i n } \left\{ V _ { \operatorname* { m a x } } , r ^ { + } ( s , a ) + \gamma U _ { N } ^ { D } \left( \left( U _ { 1 , \widetilde { c } } ^ { N , D , v } \big ( Y ^ { j } ( h ) , j \big ) \right) _ { j = 1 } ^ { M _ { N } } ; h \right) \right\} , } \end{array}\tag{151}
$$

where now $h = ( s , a , \widetilde { \theta _ { 0 } } ( s , a ) )$ .

It is useful to give a semantic, index-free representation of the same operator. For $W : \boldsymbol { S } \times \Omega ^ { \ell } $ $[ 0 , V _ { \mathrm { m a x } } ]$ , define

$$
\begin{array} { r l } & { ( \widetilde { B } _ { N , r ^ { + } } ^ { D , + } W ) ( s , \widetilde { c } ) } \\ & { \quad : = \underset { a \in A } { \operatorname* { m a x } } \operatorname* { m i n } \left\{ V _ { \operatorname* { m a x } } , r ^ { + } ( s , a ) + \gamma U _ { N } ^ { D } \left( \left( W \big ( Y ^ { j } ( h ) , \widetilde { \theta } _ { 1 } , \dots , \widetilde { \theta } _ { \ell - 1 } , \widetilde { Z } ^ { j } \big ) \right) _ { j = 1 } ^ { M _ { N } } ; h \right) \right\} , } \end{array}\tag{152}
$$

where $h = ( s , a , \widetilde { \theta _ { 0 } } ( s , a ) )$ . The scalar argument in part 1 of Lemma 12 shows that $U _ { N } ^ { D } ( \cdot ; h )$ is nondecreasing in every coordinate and is 1-Lipschitz in supremum norm. Therefore both equation 148 and equation 152 are monotone γ-contractions.

Let $W _ { N } ^ { D }$ be the fixed point of equation 152. The proofs of Lemmas 6 and 7 apply verbatim to the compound samples: the restriction of $W _ { N } ^ { D }$ to dictionary windows is the fixed point of equation 148, and equation 149– equation 151, initialized with that core fixed point, recovers $W _ { N } ^ { D }$ and its action scores on every explicit window. By Lemma 8, using $v ^ { K _ { N } }$ instead changes every deployed action score by at most $\gamma ^ { \dot { K } _ { N } } V _ { \mathrm { m a x } }$ . Hence the action selected by the algorithm is $2 \gamma ^ { K _ { N } } V _ { \mathrm { m a x } } ^ { \overline { { } } }$ -greedy for the exact dictionary scores.

We next verify optimism only where it is needed. For a continuation V of $\widetilde { V } _ { \ell } ^ { \star }$ at a graph window and a local context h, write

$$
\overline { { V } } _ { N } ^ { D } : = \frac { 1 } { M _ { N } } \sum _ { j = 1 } ^ { M _ { N } } V ( Y ^ { j } ( h ) , \widetilde { Z } ^ { j } ) , \qquad \sigma _ { N } ^ { D } ( V ) : = s \big ( ( V ( Y ^ { j } ( h ) , \widetilde { Z } ^ { j } ) ) _ { j = 1 } ^ { M _ { N } } \big ) ,
$$

with the fixed suffix suppressed.

Lemma 26 (Dictionary concentration on the relevant graph). With probability at least $1 - \delta / 8 ,$ simultaneously for every update, every graph continuation of $\widetilde { V } _ { \ell } ^ { \star }$ , and every $h ,$

$$
\begin{array} { r } { \mathbb { E } _ { \widehat { \mathcal { P } } _ { N , h } } V \le \overline { { V } } _ { N } ^ { D } + 3 d _ { N } \sigma _ { N } ^ { D } ( V ) + C V _ { \operatorname* { m a x } } d _ { N } ^ { 2 } , } \end{array}\tag{153}
$$

$$
\left| \sigma _ { N } ^ { D } ( V ) - \sqrt { \mathrm { V a r } _ { \widehat { \mathcal { P } } _ { N , h } } ( V ) } \right| \leq C V _ { \operatorname* { m a x } } d _ { N } .\tag{154}
$$

Proof. A graph continuation contains at most ℓ noisy tables drawn from the same dictionary. Fix their indices, condition on the corresponding compound samples, and omit those indices from the empirical mean. The remaining pairs are i.i.d. with law $\widehat { \mathcal { P } } _ { N , h }$ , and the continuation is now fixed. The empirical Bernstein inequality and Lemma 14 give equation 153– equation 154 for the reduced sample. Restoring at most ℓ observations changes the mean by at most $\bar { \ell V _ { \mathrm { m a x } } } / M _ { N }$ and the standard deviation by at most $2 V _ { \operatorname* { m a x } } \sqrt { \ell / M _ { N } }$ . Since $L _ { T } \ge \ell .$ , these errors are absorbed by $C V _ { \operatorname* { m a x } } d _ { N } ^ { 2 }$ and $C V _ { \operatorname* { m a x } } d _ { N }$ , respectively. If $\dot { M } _ { N } \le 2 \ell$ , the result is immediate after increasing C. A union bound over the index tuples, local contexts, and possible updates is covered by equation 134. □

Lemma 27 (Optimism on the relevant computation graph). On the events of Lemmas 25 and 26, if $r ^ { + } \geq r$ pointwise, then

$$
W _ { N } ^ { D } ( s , \widetilde { c } ) \ge \widetilde { V } _ { \ell } ^ { \star } ( s , \widetilde { c } ) \qquad f o r e \nu e r y s \in \mathcal { S } a n d \widetilde { c } \in \mathfrak { G } _ { T } .\tag{155}
$$

Proof. If $\lambda _ { N } ( h ) = 1$ , then $U _ { N } ^ { D } ( \cdot ; h ) = V _ { \mathrm { m a x } }$ . Otherwise, equation 144 and equation 153– equation 154, together with the choice of the universal constant $c _ { 0 } .$ , imply

$$
{ U _ { N } ^ { D } } \big ( ( V ( Y ^ { j } ( h ) , \widetilde { Z } ^ { j } ) ) _ { j = 1 } ^ { M _ { N } } ; h \big ) \ge \mathbb { E } _ { \mathcal { P } _ { h } } V\tag{156}
$$

for every graph continuation V of $\widetilde { V } _ { \ell } ^ { \star }$ . Here one uses the same elementary domination max $\{ 7 x , { \bar { 4 9 y } } \} \geq 2 x + 5 y$ as in the proof of Lemma 12.

Dictionary windows are closed under the sampled dynamics. Applying equation 156 on this finite core and using monotonicity and contraction proves fixed-point optimism on the core, exactly as in Lemma 13. Starting from the core and proceeding backward through equation 150 proves the same claim successively for every hybrid window in equation 140, including the observed window at the root. This proves equation 155 without asserting optimism on any other window. □

Stability and the off-graph negative part. Graph optimism alone is insufficient in a one-step conditional expectation: the true continuation law may assign mass to windows outside ${ \mathfrak { G } } _ { T }$ . We next control this mass through stability of the deployed fixed point.

Lemma 28 (Stability on explicit windows). Let D and $D ^ { \prime }$ be two compound dictionaries that differ in one compound sample, while the estimated model, reward estimate, and coefficients are held fixed. Then

$$
\operatorname* { s u p } _ { ( s , \tilde { c } ) \in \mathcal { S } \times \Omega ^ { \ell } } \left| W _ { N } ^ { D } ( s , \widetilde { c } ) - W _ { N } ^ { D ^ { \prime } } ( s , \widetilde { c } ) \right| \le \beta _ { N } , \qquad \beta _ { N } : = \frac { 2 \gamma V _ { \mathrm { m a x } } } { ( 1 - \gamma ) M _ { N } } .\tag{157}
$$

This comparison concerns the same explicit window under the two dictionaries; it does not compare equal index tuples, whose semantic windows change when a dictionary table is replaced.

Proof. Fix h and consider the unclipped expression in equation 147. On the constant branch of the maximum, changing one coordinate changes it by at most $V _ { \mathrm { m a x } } / M _ { N }$ . On the variance branch, $s ( y ) \geq 7 V _ { \operatorname* { m a x } } \lambda _ { N } ( h )$ and, whenever $s ( y ) > 0$

$$
\left| \frac { \partial s ( y ) } { \partial y _ { j } } \right| = \frac { | y _ { j } - \overline { { y } } | } { M _ { N } s ( y ) } .
$$

The contribution of the variance term to the derivative is therefore at most $1 / M _ { N }$ . Continuity at the branch boundary and clipping show that changing one coordinate of y changes $U _ { N } ^ { D } ( y ; h )$ by at most $2 V _ { \mathrm { m a x } } / M _ { N }$

Let $\Delta$ denote the supremum on the left-hand side of equation 157. If the two dictionaries differ in their jth compound sample, then every continuation coordinate other than j changes by at most $\Delta .$ whereas the jth coordinate may change by at most $V _ { \mathrm { m a x } }$ . The 1-Lipschitz property and the preceding single-coordinate bound, applied to the two explicit Bellman equations equation 152, give

$$
\Delta \leq \gamma \Delta + \frac { 2 \gamma V _ { \mathrm { m a x } } } { M _ { N } } .
$$

Rearranging proves equation 157.

Define, at the point where they are needed,

$$
g _ { N } : = \frac { C V _ { \mathrm { m a x } } } { 1 - \gamma } \sqrt { \frac { L _ { T } } { M _ { N } } } , \qquad s _ { N } : = C V _ { \mathrm { m a x } } \left( \frac { L _ { T } } { ( 1 - \gamma ) ^ { 2 } M _ { N } } \right) ^ { 1 / 4 } .\tag{158}
$$

Lemma 29 (Stable generalization of deployed continuations). With probability at least $1 - \delta / 8$ simultaneously at every update, for every actually observed window $\widetilde { c } = ( \widetilde { \theta } _ { 0 } , \dots , \widetilde { \theta } _ { \ell - 1 } )$ and every $h ,$

$$
\begin{array} { r l r } {  { | \frac { 1 } { M _ { N } } \sum _ { j = 1 } ^ { M _ { N } } W _ { N } ^ { D } \bigl ( Y ^ { j } ( h ) , \widetilde { \theta } _ { 1 : \ell - 1 } , \widetilde { Z } ^ { j } \bigr )  } } \\ & { } & {  - \mathbb { E } _ { ( S ^ { \prime } , \widetilde { Z } ) \sim \widehat { \mathcal { P } } _ { N , h } } W _ { N } ^ { D } \bigl ( S ^ { \prime } , \widetilde { \theta } _ { 1 : \ell - 1 } , \widetilde { Z } \bigr ) | \le g _ { N } , } \end{array}\tag{159}
$$

and

$$
\begin{array} { r l } & { s \biggl ( \Bigl ( W _ { N } ^ { D } \bigl ( Y ^ { j } ( h ) , \widetilde { \theta } _ { 1 : \ell - 1 } , \widetilde { Z } ^ { j } \bigr ) \Bigr ) _ { j = 1 } ^ { M _ { N } } \biggr ) } \\ & { \qquad \le \sqrt { \operatorname { V a r } _ { \widehat { \mathcal { P } } _ { N , h } } \Bigl ( W _ { N } ^ { D } ( S ^ { \prime } , \widetilde { \theta } _ { 1 : \ell - 1 } , \widetilde { Z } ) \Bigr ) } + s _ { N } . } \end{array}\tag{160}
$$

The mean bound equation 159 also holds after replacing $W _ { N } ^ { D }$ by $[ \widetilde { V } _ { \ell } ^ { \star } - W _ { N } ^ { D } ] _ { + }$

Proof. Treat the jth compound draw $( \widetilde { Z } ^ { j } , ( Y ^ { j } ( h ) ) _ { h } )$ as one training sample. For a fixed observed suffix, let $F _ { D } ( \boldsymbol { \xi } )$ be the corresponding continuation of $W _ { N } ^ { D }$ evaluated at a fresh compound sample

ξ. By Lemma 28, replacing one training sample changes $F _ { D } ( \boldsymbol { \xi } )$ , uniformly in $\xi ,$ by at most $\beta _ { N }$ . The usual ghost-sample replacement argument therefore gives

$$
\left| \mathbb { E } _ { D } \left[ \mathbb { E } _ { \xi } F _ { D } ( \xi ) - \frac { 1 } { M _ { N } } \sum _ { j = 1 } ^ { M _ { N } } F _ { D } ( \xi _ { j } ) \right] \right| \leq \beta _ { N } .
$$

Replacing one training sample changes the quantity in brackets by at most $2 \beta _ { N } + V _ { \operatorname* { m a x } } / M _ { N }$ . Mc-Diarmid’s inequality and the union bound encoded by $L _ { T }$ hence give

$$
\left| \mathbb { E } _ { \xi } F _ { D } ( \xi ) - \frac { 1 } { M _ { N } } \sum _ { j = 1 } ^ { M _ { N } } F _ { D } ( \xi _ { j } ) \right| \leq C \left[ \frac { V _ { \operatorname* { m a x } } } { \sqrt { M _ { N } } } + \beta _ { N } \sqrt { M _ { N } } \right] \sqrt { L _ { T } } \leq g _ { N } .
$$

The positive-part map is 1-Lipschitz, so the same argument applies to the stated loss.

Apply the argument once more to $F _ { D } ^ { 2 }$ . Its range is $[ 0 , V _ { \mathrm { m a x } } ^ { 2 } ]$ and its stability is at most $2 V _ { \mathrm { m a x } } \beta _ { N }$ Thus its empirical and population second moments differ by at most

$$
\frac { C V _ { \operatorname* { m a x } } ^ { 2 } } { 1 - \gamma } \sqrt { \frac { L _ { T } } { M _ { N } } } .
$$

Combining this with the mean bound, the identity $\operatorname { V a r } ( F ) = \mathbb { E } F ^ { 2 } - ( \mathbb { E } F ) ^ { 2 }$ , and $| { \sqrt { x } } - { \sqrt { y } } | \leq$ $\sqrt { | x - y | }$ proves equation 160. □

Lemma 30 (Expected off-graph negative part). Fix a planning object, a state-window pair $x =$ $( s , \widetilde { c } )$ actually observed while it is active, and an action a. Let $h \ = \ ( s , a , \widetilde { \theta } _ { 0 } ( s , a ) )$ , set $G =$ $W _ { N } ^ { D } - \widetilde { V } _ { \ell } ^ { \star }$ , and write $G _ { - } = ( - G ) _ { + }$ . Then

$$
\mathbb { E } _ { \mathcal { P } _ { h } } G _ { - } \bigl ( S ^ { \prime } , \widetilde { c } _ { 1 : \ell - 1 } , \widetilde { Z } \bigr ) \le C \left[ g _ { N } + V _ { \mathrm { m a x } } q _ { N } ^ { 2 } + \frac { V _ { \mathrm { m a x } } n L _ { T } } { 1 \vee k _ { N } ( h ) } \right] .\tag{161}
$$

Proof. For every $j ,$ the compound continuation $( Y ^ { j } ( h ) , \widetilde { c } _ { 1 : \ell - 1 } , \widetilde { Z } ^ { j } )$ is a hybrid graph state. Lemma 27 therefore gives

$$
\frac { 1 } { M _ { N } } \sum _ { j = 1 } ^ { M _ { N } } G _ { - } \bigl ( Y ^ { j } ( h ) , \widetilde { c } _ { 1 : \ell - 1 } , \widetilde { Z } ^ { j } \bigr ) = 0 .
$$

The loss version of Lemma 29 bounds the expectation of this loss under $\widehat { \mathcal { P } } _ { N , h }$ by $g _ { N }$

Next replace the estimated noisy-table marginal by the true one. Applying equation 138 conditionally on the successor state, averaging, and using $q _ { N } \sqrt { V _ { \operatorname* { m a x } } x } \le \bar { x / 4 } + \overline { { C \bar { V _ { \operatorname* { m a x } } } q _ { N } ^ { 2 } } }$ bounds the resulting expectation by $C ( g _ { N } + V _ { \operatorname* { m a x } } q _ { N } ^ { \overleftarrow } )$ . Finally apply Lemma 23 with $\eta = 1 / 2$ to the nonnegative function obtained after averaging over the true noisy-table marginal. Absorbing the resulting half of the target expectation proves equation 161. □

One-step recursion and deterministic sums. At time $t \geq 1$ , let $N _ { t }$ be the sample size stored in the active planning object and let $k _ { t } ( h ) : = k _ { N _ { t } } ( h )$ . Let $W _ { t }$ be the exact extended fixed point of that planning object and set

$$
D _ { t } : = W _ { t } - \widetilde { V } _ { \ell } ^ { \star } .
$$

After $A _ { t }$ is selected, define

$$
h _ { t } : = \big ( S _ { t } , A _ { t } , \widetilde { \Theta } _ { t } ( S _ { t } , A _ { t } ) \big ) , \qquad \lambda _ { t } : = \lambda _ { N _ { t } } ( h _ { t } ) .
$$

Conditionally on $\mathcal { F } _ { t }$ , which contains the history and the current noisy window before $A _ { t }$ is selected, put

$$
\begin{array} { r l r l } & { p _ { t } : = \mathbb { E } [ ( D _ { t } ) _ { + } ( X _ { t + 1 } ) \mid \mathcal { F } _ { t } ] , } & { \qquad } & { v _ { t } : = \mathbb { E } [ ( D _ { t } ) _ { - } ( X _ { t + 1 } ) \mid \mathcal { F } _ { t } ] , } \\ & { \sigma _ { t } ^ { 2 } : = \mathrm { V a r } \Big ( \widetilde { V } _ { \ell } ^ { \star } ( X _ { t + 1 } ) \mid \mathcal { F } _ { t } \Big ) . } \end{array}\tag{162}
$$

Lemma 31 (Noisy one-step recursion). On the joint confidence event, the action selected by Algorithm 6 satisfies

$$
\begin{array} { r l } & { D _ { t } ( X _ { t } ) + \widetilde { V } _ { \ell } ^ { \star } ( X _ { t } ) - \widetilde { Q } _ { \ell } ^ { \star } ( X _ { t } , A _ { t } ) } \\ & { \quad \leq r _ { t } ^ { + } ( S _ { t } , A _ { t } ) - r ( S _ { t } , A _ { t } ) + 2 \gamma ^ { K _ { N _ { t } } } V _ { \operatorname* { m a x } } + \gamma ( p _ { t } - v _ { t } ) } \\ & { \qquad + C \gamma \Bigg [ \lambda _ { t } \sigma _ { t } + \lambda _ { t } \sqrt { V _ { \operatorname* { m a x } } p _ { t } } + \lambda _ { t } \sqrt { V _ { \operatorname* { m a x } } v _ { t } } + V _ { \operatorname* { m a x } } \lambda _ { t } ^ { 2 } } \\ & { \qquad + \frac { V _ { \operatorname* { m a x } } n L _ { T } } { ( 1 - \gamma ) ( 1 \vee k _ { t } ( h _ { t } ) ) } + g _ { N _ { t } } + \lambda _ { t } s _ { N _ { t } } \Bigg ] + \frac { \gamma ( 1 - \gamma ) } { 3 2 } ( p _ { t } + v _ { t } ) . } \end{array}\tag{163}
$$

Proof. The restriction and extension identities, contraction, and the choice of $K _ { N _ { t } }$ imply that the selected action is $2 \gamma ^ { K _ { N _ { t } } } V _ { \mathrm { m a x } }$ -greedy for the exact dictionary scores. Repeating the first Bellman comparison in the proof of Lemma 18 gives

$$
\begin{array} { r l } & { D _ { t } ( X _ { t } ) + \widetilde { V } _ { \ell } ^ { \star } ( X _ { t } ) - \widetilde { Q } _ { \ell } ^ { \star } ( X _ { t } , A _ { t } ) } \\ & { \quad \leq r _ { t } ^ { + } ( S _ { t } , A _ { t } ) - r ( S _ { t } , A _ { t } ) + 2 \gamma ^ { K _ { N _ { t } } } V _ { \operatorname* { m a x } } } \\ & { \quad ~ + \gamma \left[ U _ { N _ { t } } ^ { D } ( W _ { t } ; h _ { t } ) - \mathbb { E } _ { \mathcal { P } _ { h _ { t } } } \widetilde { V } _ { \ell } ^ { \star } ( X _ { t + 1 } ) \right] , } \end{array}
$$

where $U _ { N _ { t } } ^ { D } ( W _ { t } ; h _ { t } )$ denotes equation 147 applied to the continuations of $W _ { t }$ from the current observed suffix. Adding and subtracting $\mathbb { E } p _ { h _ { t } } W _ { t } ( X _ { t + 1 } )$ produces exactly $p _ { t } - v _ { t }$

It remains to control the width of the sampled continuation rule at the dictionary-dependent value $W _ { t }$ . Write $W _ { t } = \widetilde { V } _ { \ell } ^ { \star } + D _ { t }$ . For the first term, use Lemma 25; for the empirical mean and standard deviation of $W _ { t }$ , use Lemma 29. For the signed difference $D _ { t }$ , apply Lemma 23 separately to $( D _ { t } ) _ { + }$ and $( D _ { t } ) _ { - }$ , and apply equation 138 to the noisy-table marginal. Taking the parameter in equation 135 to be a sufficiently small universal multiple of $1 - \gamma$ gives

$$
\begin{array} { r l } & { \left. \mathbb { E } _ { \widehat { \mathcal { P } } _ { N _ { t } , h _ { t } } } D _ { t } - \mathbb { E } _ { \mathcal { P } _ { h _ { t } } } D _ { t } \right. \leq \frac { 1 - \gamma } { 3 2 } ( p _ { t } + v _ { t } ) } \\ & { \qquad + C \left[ q _ { N _ { t } } \sqrt { V _ { \operatorname* { m a x } } ( p _ { t } + v _ { t } ) } + V _ { \operatorname* { m a x } } q _ { N _ { t } } ^ { 2 } + \frac { V _ { \operatorname* { m a x } } n L _ { T } } { ( 1 - \gamma ) ( 1 \vee k _ { t } ( h _ { t } ) ) } \right] . } \end{array}\tag{164}
$$

The same comparisons, now applied to $| D _ { t } |$ with a constant parameter, give

$$
\begin{array} { r l } { \sqrt { \operatorname { V a r } _ { \widehat { \mathcal { P } } _ { N _ { t } , h _ { t } } } ( D _ { t } ) } \leq C \Bigg [ \sqrt { V _ { \operatorname* { m a x } } p _ { t } } + \sqrt { V _ { \operatorname* { m a x } } v _ { t } } + V _ { \operatorname* { m a x } } q _ { N _ { t } } } & { } \\ { + V _ { \operatorname* { m a x } } \sqrt { \frac { n L _ { T } } { 1 \vee k _ { t } ( h _ { t } ) } } \Bigg ] . } \end{array}
$$

Together with equation 145 and the triangle inequality for standard deviations, these bounds imply

$$
\begin{array} { r l } & { U _ { N _ { t } } ^ { D } ( W _ { t } ; h _ { t } ) - \mathbb { E } _ { \mathcal { P } _ { h _ { t } } } W _ { t } ( X _ { t + 1 } ) } \\ & { \quad \le C \Bigg [ \lambda _ { t } \sigma _ { t } + \lambda _ { t } \sqrt { V _ { \operatorname* { m a x } } p _ { t } } + \lambda _ { t } \sqrt { V _ { \operatorname* { m a x } } v _ { t } } + V _ { \operatorname* { m a x } } \lambda _ { t } ^ { 2 } } \\ & { \qquad + \displaystyle \frac { V _ { \operatorname* { m a x } } n L _ { T } } { ( 1 - \gamma ) ( 1 \vee k _ { t } ( h _ { t } ) ) } + g _ { N _ { t } } + \lambda _ { t } s _ { N _ { t } } \Bigg ] + \frac { 1 - \gamma } { 3 2 } ( p _ { t } + v _ { t } ) . } \end{array}
$$

Here $q _ { N _ { t } } \leq c _ { 0 } ^ { - 1 } \lambda _ { t }$ whenever $\lambda _ { t } < 1$ , while the case $\lambda _ { t } = 1$ is immediate from boundedness; also,

$$
\lambda _ { t } \sqrt { \frac { n L _ { T } } { 1 \vee k _ { t } ( h _ { t } ) } } \leq \lambda _ { t } ^ { 2 } + \frac { n L _ { T } } { 1 \vee k _ { t } ( h _ { t } ) } .
$$

Substitution proves equation 163.

Lemma 32 (Cumulative noisy confidence terms). Forfixed ℓ,

$$
\sum _ { t = 1 } ^ { T - 1 } \lambda _ { t } ^ { 2 } = \widetilde { \cal O } _ { \ell } ( n ^ { 2 } m ) ,\tag{165}
$$

$$
\sum _ { t = 1 } ^ { T - 1 } \frac { n L _ { T } } { 1 \vee k _ { t } ( h _ { t } ) } = \widetilde { O } _ { \ell } ( n ^ { 3 } m ) ,\tag{166}
$$

$$
\sum _ { t = 1 } ^ { T - 1 } g _ { N _ { t } } = \widetilde { O } _ { \ell } \left( R _ { \operatorname* { m a x } } \sqrt { \frac { T } { 1 - \gamma } } \right) ,\tag{167}
$$

$$
\sum _ { t = 1 } ^ { T - 1 } \lambda _ { t } s _ { N _ { t } } = \widetilde O _ { \ell } \bigg ( \frac { R _ { \operatorname* { m a x } } n \sqrt { m } T ^ { 1 / 4 } } { ( 1 - \gamma ) ^ { 3 / 4 } } \bigg ) ,\tag{168}
$$

$$
\sum _ { t = 1 } ^ { T - 1 } v _ { t } = \widetilde O _ { \ell } \Bigg ( R _ { \operatorname* { m a x } } \sqrt { \frac { T } { 1 - \gamma } } + { \frac { R _ { \operatorname* { m a x } } n ^ { 3 } m } { 1 - \gamma } } \Bigg ) .\tag{169}
$$

Proof. The global power-of-two updates imply $t / 2 \leq N _ { t } \leq t .$ . If a local context is visited J times, its doubling schedule gives

$$
\sum _ { j = 1 } ^ { J } \frac { 1 } { 1 \vee k _ { j } } \le C ( 1 + \log J ) ,
$$

where $k _ { j }$ is the frozen count used on its jth visit. Summing over the $n ^ { 2 }$ m contexts proves equation 166 and controls the contribution of $a _ { N _ { t } } ^ { 2 } ( h _ { t } )$ to equation 165. Moreover,

$$
\sum _ { t < T } q _ { N _ { t } } ^ { 2 } = \widetilde { O } ( n ^ { 2 } m ) , \qquad \sum _ { t < T } d _ { N _ { t } } ^ { 2 } = \widetilde { O } _ { \ell } ( 1 ) ,
$$

because $N _ { t } \geq t / 2$ and $M _ { N } \geq N / ( 1 - \gamma ) ^ { 3 }$ . This proves equation 165.

The definition of $g _ { N }$ and $\begin{array} { r } { \sum _ { t < T } N _ { t } ^ { - 1 / 2 } = O ( \sqrt { T } ) } \end{array}$ give equation 167. Furthermore,

$$
s _ { N } ^ { 2 } \leq C V _ { \operatorname* { m a x } } ^ { 2 } \sqrt { \frac { ( 1 - \gamma ) L _ { T } } { N } } ,
$$

so Cauchy–Schwarz and equation 165 give equation 168. Finally, Lemma 30, applied along the trajectory, gives

$$
v _ { t } \leq C \left[ g _ { N _ { t } } + V _ { \operatorname* { m a x } } q _ { N _ { t } } ^ { 2 } + \frac { V _ { \operatorname* { m a x } } n L _ { T } } { 1 \vee k _ { t } ( h _ { t } ) } \right] .
$$

Summing this inequality and using the preceding bounds proves equation 169.

Completion ofthe proofofTheorem 5. Invoke the preceding events, the reward event of Lemma 16, the predictable-to-realized event of Lemma 19, and the discounted total-variance event of Lemma 20, with confidence budgets whose sum is at most δ. The first decision contributes at most $V _ { \mathrm { m a x } }$

By Lemma 27, $D _ { t } ( X _ { t } ) \geq 0$ at every actually observed window. Apply Lemma 19 to the nonnegative random variable

$$
Y _ { t } : = ( D _ { t } ) _ { + } ( X _ { t + 1 } ) , \qquad \mathbb { E } [ Y _ { t } \mid { \mathcal F } _ { t } ] = p _ { t } .
$$

On the graph-optimism event, the realized successor window also belongs to ${ \mathfrak { G } } _ { T }$ , so $Y _ { t } = D _ { t } ( X _ { t + 1 } )$ along the trajectory. Within a planning epoch $\{ p , \ldots , q \} , D _ { t }$ is fixed and therefore

$$
\sum _ { t = p } ^ { q } \bigl ( Y _ { t } - D _ { t } ( X _ { t } ) \bigr ) = D _ { t } ( X _ { q + 1 } ) - D _ { t } ( X _ { p } ) \leq V _ { \operatorname* { m a x } } .\tag{170}
$$

Use Young’s inequality in the positive adaptive term:

$$
C \gamma \lambda _ { t } \sqrt { V _ { \operatorname* { m a x } } p _ { t } } \leq \frac { 1 - \gamma } { 1 6 } p _ { t } + \frac { C V _ { \operatorname* { m a x } } \lambda _ { t } ^ { 2 } } { 1 - \gamma } .
$$

Together with the explicit $( 1 - \gamma ) p _ { t } / 3 2$ term in equation 163, the coefficient of $p _ { t }$ remains at most $1 - c ( 1 - \gamma )$ for a universal $c > 0$ . Thus Lemma 19 and equation 170 yield the same epoch potential bound as in Lemma 21. The negative contribution

$$
- \gamma v _ { t } + \frac { \gamma ( 1 - \gamma ) } { 3 2 } v _ { t }
$$

is nonpositive and can be discarded. Summing equation 163 therefore gives

$$
\begin{array} { r l } & { \widetilde { \mathrm { R e g } } _ { \ell } ( T ) \leq C \Bigg [ \displaystyle \sum _ { t = 0 } ^ { T - 1 } \big ( r _ { t } ^ { + } ( S _ { t } , A _ { t } ) - r ( S _ { t } , A _ { t } ) \big ) + \displaystyle \sum _ { t = 1 } ^ { T - 1 } \gamma \lambda _ { t } \sigma _ { t } } \\ & { \qquad + \displaystyle \frac { V _ { \operatorname* { m a x } } } { 1 - \gamma } \displaystyle \sum _ { t = 1 } ^ { T - 1 } \lambda _ { t } ^ { 2 } + \displaystyle \frac { V _ { \operatorname* { m a x } } H L _ { T } } { 1 - \gamma } \displaystyle \sum _ { t = 1 } ^ { T - 1 } \frac { 1 } { 1 \vee k _ { t } \left( h _ { t } \right) } } \\ & { \qquad + \displaystyle \sum _ { t = 1 } ^ { T - 1 } \big ( g _ { N _ { t } } + \lambda _ { t } s _ { N _ { t } } \big ) + \displaystyle \sum _ { t = 1 } ^ { T - 1 } \lambda _ { t } \sqrt { V _ { \operatorname* { m a x } } } \sigma _ { t } } \\ & { \qquad + V _ { \operatorname* { m a x } } \overline { { E } } _ { T } + \displaystyle \frac { V _ { \operatorname* { m a x } } L _ { T } } { 1 - \gamma } + R _ { \operatorname* { m a x } } \Bigg ] . } \end{array}\tag{171}
$$

Indeed, the value-iteration errors are summable because

$$
\gamma ^ { K _ { N } } V _ { \mathrm { m a x } } \leq \frac { R _ { \mathrm { m a x } } } { 2 N ^ { 2 } } .
$$

Lemma 20 uses only the Bellman equation and the filtration of the augmented Markov process. It therefore applies verbatim to the noisy augmented process and gives

$$
\sum _ { t = 1 } ^ { T - 1 } \gamma ^ { 2 } \sigma _ { t } ^ { 2 } \le C \left[ V _ { \operatorname* { m a x } } R _ { \operatorname* { m a x } } T + V _ { \operatorname* { m a x } } \widetilde { \mathrm { R e g } } _ { \ell } ( T ) + V _ { \operatorname* { m a x } } ^ { 2 } L _ { T } \right] .\tag{172}
$$

Consequently, by Cauchy–Schwarz and equation 165,

$$
\sum _ { t = 1 } ^ { T - 1 } \gamma \lambda _ { t } \sigma _ { t } \leq \widetilde { O } _ { \ell } \left( \sqrt { n ^ { 2 } m \left[ V _ { \mathrm { m a x } } R _ { \mathrm { m a x } } T + V _ { \mathrm { m a x } } \widetilde { \mathrm { R e g } } _ { \ell } ( T ) \right] } \right) .
$$

The self-bounding term is absorbed through

$$
C \sqrt { n ^ { 2 } m V _ { \mathrm { m a x } } \widetilde { \mathrm { R e g } } _ { \ell } ( T ) } \le \frac { 1 } { 4 } \widetilde { \mathrm { R e g } } _ { \ell } ( T ) + C V _ { \mathrm { m a x } } n ^ { 2 } m .
$$

It remains to verify that the off-graph contribution is lower order. Lemma 32 and Cauchy–Schwarz give

$$
\begin{array} { r l r } {  { \sum _ { t = 1 } ^ { T - 1 } \lambda _ { t } \sqrt { V _ { \operatorname* { m a x } } v _ { t } } \le \sqrt { ( \sum _ { t } \lambda _ { t } ^ { 2 } ) V _ { \operatorname* { m a x } } \sum _ { t } v _ { t } } } } \\ & { } & { \le \widetilde O _ { \ell } \bigg ( \frac { R _ { \operatorname* { m a x } } n \sqrt m T ^ { 1 / 4 } } { ( 1 - \gamma ) ^ { 3 / 4 } } + \frac { R _ { \operatorname* { m a x } } n ^ { 5 / 2 } m } { 1 - \gamma } \bigg ) . } \end{array}\tag{173}
$$

If $T \geq ( 1 - \gamma ) ^ { - 1 }$ , the first term in equation 173 is bounded by the leading term in equation 125; i $\ : T < ( 1 - \gamma ) ^ { - 1 } \ :$ , it is bounded by the lower-order term. The second term is also bounded by the lower-order term because $n \geq 1$ and $1 - \gamma \leq 1$ . The same case split controls equation 168, while equation 167 is smaller still.

Finally substitute Lemma 16, Lemma 32, and equation 132 into equation 171, use $V _ { \mathrm { m a x } } ~ =$ $R _ { \operatorname* { m a x } } \dot { } / ( 1 - \gamma )$ , and absorb the preceding quarter of the regret. The leading term is

$$
\widetilde { O } _ { \ell } \left( \sqrt { n ^ { 2 } m V _ { \mathrm { m a x } } R _ { \mathrm { m a x } } T } \right) = \widetilde { O } _ { \ell } \left( R _ { \mathrm { m a x } } n \sqrt { \frac { m T } { 1 - \gamma } } \right) ,
$$

and the largest lower-order contribution is

$$
\widetilde O _ { \ell } \biggl ( \frac { V _ { \mathrm { m a x } } n ^ { 3 } m } { 1 - \gamma } \biggr ) = \widetilde O _ { \ell } \biggl ( \frac { R _ { \mathrm { m a x } } n ^ { 3 } m } { ( 1 - \gamma ) ^ { 2 } } \biggr ) .
$$

This proves equation 125.