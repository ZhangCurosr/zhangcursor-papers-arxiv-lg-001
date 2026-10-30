# Closing the Log Gap: Minimax-Optimal Regret for Episodic RL

Guy Zamir<sup>∗</sup>, Victor Boone<sup>†</sup>, Matthew Zurek<sup>∗</sup>, and Yudong Chen<sup>∗</sup>

<sup>∗</sup>Department of Computer Sciences, University of Wisconsin–Madison <sup>†</sup>IRIT, Université de Toulouse, CNRS, Toulouse INP, Toulouse, France

gzamir@wisc.edu victor.boone@irit.fr matthew.zurek@wisc.edu yudongchen@cs.wisc.edu

## Abstract

In this paper, we study the regret minimization problem in episodic tabular MDPs. Our algorithm achieves the first expected regret bound scaling as $\sqrt { H ^ { 3 } S A K }$ for suficiently large K, where H is the episode length, K is the number of episodes, and S and A are the numbers of states and actions, respectively. Our method is the first to match the minimax lower bound without extra polylog factors. We introduce a variant of UCBVI which does not maintain strictly optimistic estimates, and uses MOSS-style exploration bonuses whose logarithmic terms adapt to both visit counts and estimated variances. A key challenge is controlling the propagation of errors caused by under-optimism, which we accomplish in an occupancy-weighted way, avoiding logarithmic overhead that would be incurred by uniform-over-rows concentration. Our analysis develops time-uniform and variance-aware concentration bounds that may be of independent interest. As a corollary we also develop an algorithm for (ε, δ)-PAC best-policy identification, which is the first to match the $\Omega \left( H ^ { 3 } S A \log ( \textstyle { \frac { 1 } { \delta } } ) \varepsilon ^ { - 2 } \right)$ minimax lower bound up to constants in some regimes.

## 1 Introduction

In online reinforcement learning (RL), the balance of exploration and exploitation is one of the most central and investigated problems. A canonical setting for the study of the exploration-exploitation tradeof is that of finite-horizon Markov decision processes (MDPs), and in particular the problem of regret minimization in such environments. In regret minimization, the learning agent dynamically explores the unknown environment by playing actions, and its performance is measured by comparing its aggregate reward with that of an omniscient agent that knows the environment in advance. In this work, we are interested in the derivation of formal bounds of the worst-case regret (a.k.a. minimax regret) in expectation and in the time-inhomogeneous setting, where rewards and transitions are time-dependent.

This problem has been extensively studied. In the class of tabular MDPs with S states, A actions per state and fixed horizon H, it is known [Domingues et al., 2021] that the minimax regret of every learning algorithm is at least $\Omega ( \sqrt { H ^ { 3 } S A K } )$ ), where K is the total number of episodes. In the community, UCBVI [Azar et al., 2017] is recognized as the first algorithm to reach the minimax lower bound of Domingues et al. [2021]. It was succeeded by many others, such as Euler [Zanette and Brunskill, 2019] or MVP [Zhang et al., 2021], with better second order terms or better dependency with respect to the episode horizon H (see Table 1 for a more comprehensive list). Although these algorithms are claimed to be minimax optimal, all their regret guarantees are $\widetilde { \cal O } ( \sqrt { H ^ { 3 } S A K } )$ , where ${ \widetilde { O } } ( - )$ is a variant of $O ( - )$ that ignores logarithmic factors. Unraveling what the ${ \widetilde { O } } ( - )$ hides, the true regret bounds of these methods are $O ( \sqrt { H ^ { 3 } S A K \log ( H S A K ) ^ { \gamma } } )$ for $\gamma > 0$ (Table 1). As such, no algorithm reaches the true minimax optimal rate prior to this work.

<table><tr><td>Algorithm</td><td>Expected regret scaling as K → ∞</td><td>Original work is time-inhomogeneous</td><td>Optimistic with high probability</td></tr><tr><td>UCBVI Azar et al. [2017]</td><td> $\overline { { \sqrt { H ^ { 3 } S A K \log ( K ) ^ { 2 } } } }$ </td><td>No</td><td>Yes</td></tr><tr><td>Euler Zanette and Brunskill [2019]</td><td> $\sqrt { H ^ { 3 } S A K \log ( K ) ^ { 6 } }$ </td><td>No</td><td>Yes</td></tr><tr><td>UCB-Advantage Zhang et al. [2020]</td><td> $\sqrt { H ^ { 3 } S A K \log ( K ) } + \sqrt { H ^ { 3 } K \log ( K ) ^ { 3 } }$ </td><td>Yes</td><td>Yes</td></tr><tr><td>UCBMQ Ménard et al. [2021]</td><td> $\sqrt { H ^ { 3 } S A K \log ( K ) ^ { 3 } }$ </td><td>Yes</td><td>Yes</td></tr><tr><td>MVP Zhang et al. [2021]</td><td> $\sqrt { H ^ { 3 } S A K \log ( K ) ^ { 2 } }$ </td><td>No</td><td>Yes</td></tr><tr><td>Modified MVP Zhang et al. [2025]</td><td> $\sqrt { H ^ { 3 } S A K \log ( K ) ^ { 5 } }$ </td><td>Yes</td><td>Yes</td></tr><tr><td>SSR Xiong et al. [2022]</td><td> $\sqrt { H ^ { 3 } S A K \log ( K ) ^ { 3 } }$ </td><td>Yes</td><td>Yes</td></tr><tr><td>OPSRL Tiapkin et al. [2022]</td><td> $\sqrt { H ^ { 3 } S A K \log ( K ) ^ { 3 } }$ </td><td>Yes</td><td>Yes</td></tr><tr><td>EQ0 Lee and Oh [2025], EQ0+ Lee and Oh [2026]</td><td> $\sqrt { H ^ { 3 } S A K \log ( H S A ) \log ( K ) }$ </td><td>No</td><td>No</td></tr><tr><td>MORE-UCBVI (ours)</td><td>√H3SAK</td><td>Yes</td><td>No</td></tr><tr><td>Lower Bound Domingues et al. [2021]</td><td>√H3SAK</td><td></td><td></td></tr></table>

Table 1: A compendium of algorithms which are minimax optimal up to at most log factors.

In this paper, we close this longstanding gap and construct the first algorithm with minimax regret $O ( \sqrt { H ^ { 3 } S A K } )$ for suficiently large K.

Our contributions. We introduce MORE-UCBVI, a variant of UCBVI Azar et al. [2017] that uses MOSS-style bonuses Audibert and Bubeck [2009] in place of UCB-style bonuses Agrawal [1995], Auer et al. [2002].

• We derive a MOSS-style bonus for UCBVI whose logarithmic term adapts to both visit counts and estimated variances. The resulting algorithm (Section 5, Algorithm 1) deviates from the UCBVI framework by allowing for under-optimism; that is, its value estimates may underestimate the optimal value function.

• We obtain an expected regret bound of $O ( \sqrt { H ^ { 3 } S A K } )$ for suficiently large K without logarithmic factors, thereby matching the minimax rate up to a universal constant for the first time (Section 6, Theorem 6.1).

• Our main result requires a sharper analysis. We replace the standard monotonicity-based optimism induction with a stability argument that controls how under-optimism propagates through Bellman updates. This argument enables us to bound the cumulative cost of under-optimism without the logarithmic overhead induced by a union bound. We provide a more detailed proof sketch in Section 7.

• As a corollary, we construct an algorithm for (ε, δ)-PAC best-policy identification which is the first to match the minimax sample complexity lower bound of $\Omega ( H ^ { 3 } S A \log ( \textstyle { \frac { 1 } { \delta } } ) \varepsilon ^ { - 2 } )$ established by Domingues et al. [2021] in some regime, improving the logarithmic factor relative to prior works (Section 6, Corollary 6.2).

• Along the way, we prove a time-uniform Bernstein inequality with a linear drift, giving an analogue of [Lattimore and Szepesvári, 2020, Lemma 9.3] for sub-gamma random variables (Section 4, Theorem 4.1). This variance-aware concentration result may be of independent interest.

Related work. The literature of regret minimization in RL is vast. Focusing on episodic MDPs, Table 1 lists existing algorithms matching the minimax lower bound up to logarithmic factors. Some are originally for the homogeneous setting, in which all $P _ { 1 } , \ldots , P _ { H }$ are equal, in which case they can be applied to the inhomogeneous setting by augmenting all states with a time index in [H], thus multiplying the size of the state space by H. Many of these have been introduced to enrich regret minimization with parallel considerations. MVP [Zhang et al., 2021] is meant to improve on the dependency in the horizon H, for what is known as horizon-free RL. EQO+ [Lee and Oh, 2025, 2026] provides regret guarantees in probability simultaneously for every confidence level $\delta > 0$ . UCBMQ [Ménard et al., 2021] is a variant of Q-Learning [Watkins and Dayan,

1992] with near minimax optimal guarantees; it is model-free and therefore has a cheaper memory cost. As discussed in Section 3, the logarithmic factors incurred by these algorithms are inherent to their construction.

Beyond the nearly minimax optimal algorithms in Table 1, there exists a large range of methods with formal guarantees that are interesting in their own right, such as the line of works on model-free methods [Ménard et al., 2021] or that on Bayesian methods [Osband et al., 2013, Tiapkin et al., 2022].

Furthermore, parallel to minimax regret bounds, there is a growing literature on problem-dependent regret analysis with bounds scaling as log(K) rather than $\sqrt { K }$ , see for instance Xu et al. [2021], Tirinzoni et al. [2021], Dann et al. [2021], Yang et al. [2021]. We also note that many episodic RL algorithms have analogues in average-reward RL. For instance, the analogue of UCBVI [Azar et al., 2017] is FOCUS [Zamir et al., 2026], although the dominant template there is rather UCRL2 [Auer et al., 2008] and its variants [Filippi et al., 2010, Bourel et al., 2020, Boone and Zhang, 2024].

## 2 Preliminaries

A tabular episodic Markov decision process (MDP) is a tuple $( \mathcal S , \mathcal A , H , ( P _ { h } ) _ { h = 1 } ^ { H } , ( r _ { h } ) _ { h = 1 } ^ { H } , \mu )$ . S and A are the state and action spaces with cardinalities S and A, respectively. H is the length of each episode. $P _ { h } : S \times \mathcal { A }  \Delta ( S )$ is the transition kernel at step h, with $P _ { h } ( \cdot | s , a )$ denoting the distribution over possible next states upon taking action a at state s. $r _ { h } : S \times \mathcal { A }  [ 0 , 1 ]$ is the (deterministic) reward function at step $h . ~ \mu$ is the initial state distribution.

We study tabular online RL. For each episode $k = 1 , \ldots , K$ , the learner plays a policy $\pi ^ { k } = ( \pi _ { h } ^ { k } ) _ { h = 1 } ^ { H }$ 1 then observes the trajectory $s _ { 1 } ^ { k } , a _ { 1 } ^ { k } , r _ { 1 } ^ { k } , \ldots , s _ { H } ^ { k } , a _ { H } ^ { k } , r _ { H } ^ { k }$ , where $s _ { 1 } ^ { k } \sim \mu$ , then for each $h \in [ H ] , a _ { h } ^ { k } = \pi _ { h } ^ { k } ( s _ { h } ^ { k } )$ $r _ { h } ^ { k } = r _ { h } ( s _ { h } ^ { k } , a _ { h } ^ { k } )$ ), and $s _ { h + 1 } ^ { k } \sim P _ { h } ( \cdot | s _ { h } ^ { k } , a _ { h } ^ { k } )$ . Note that we assume the learner does not know the transition kernels $\left( P _ { h } \right)$ or initial state distribution µ but otherwise knows $s , \mathcal { A } , H , ( r _ { h } )$ , and $K$

For a policy $\pi = ( \pi _ { h } ) _ { h = 1 } ^ { H }$ we let $( V _ { h } ^ { \pi } ) _ { h = 1 } ^ { H + 1 }$ and $( Q _ { h } ^ { \pi } ) _ { h = 1 } ^ { H + 1 }$ be the value and action-value functions for the policy π. We also denote by $( V _ { h } ^ { \star } )$ and $( Q _ { h } ^ { \star } )$ the optimal value and action-value functions. The learner’s goal is to minimize the regret, defined as

$$
\operatorname { R e g } ( K ) : = \sum _ { k = 1 } ^ { K } { \Big ( } V _ { 1 } ^ { \star } ( s _ { 1 } ^ { k } ) - V _ { 1 } ^ { \pi ^ { k } } ( s _ { 1 } ^ { k } ) { \Big ) } ~ .\tag{1}
$$

We are interested in minimizing the regret in expectation and in the worst case, called minimax regret, that is, bounding the expectation of $\mathrm { R e g } ( K )$ uniformly for every MDP with S states, A actions and horizon H.

## 2.1 Additional Notation

Let P be a transition kernel. Given a step h and a state-action pair $( s , a )$ , we often write $P _ { h , s , a }$ to denote the row vector of length S whose entries are $P _ { h } ( s ^ { \prime } | s , a )$ for each $s ^ { \prime } .$ Hence, for $V \in \mathbb { R } ^ { s }$ we use $P _ { h , s , a } V$ as shorthand for $\begin{array} { r } { \sum _ { s ^ { \prime } } P _ { h } ( s ^ { \prime } | s , a ) V ( s ^ { \prime } ) } \end{array}$ . Given a distribution $p \in \Delta ( \mathcal { S } )$ and value function $V \in \mathbb { R } ^ { s }$ , we define

$$
\begin{array} { r } { \operatorname { V a r } ( p , V ) : = p V ^ { 2 } - ( p V ) ^ { 2 } = \operatorname { V a r } _ { s \sim p } ( V ( s ) ) . } \end{array}
$$

For an integer $n \geq 1$ , we let $[ n ] : = \{ 1 , \dots , n \}$ . 1 to denotes the all-ones vector and I denotes the indicator function. We write x $\lesssim y$ or $x \le O ( y )$ if $\textit { x } \leq \textit { c y }$ for some universal constant $c > 0$ , and $x \ \asymp \ y$ if both $x \ \lesssim \ y$ and $y \lesssim x .$ . Finally, we define $\log ^ { \geq 0 } ( x ) : = \log ( \operatorname* { m a x } \{ 1 , x \} ) , \ \log ^ { \geq 1 } ( x ) : = \operatorname* { m a x } \{ 1 , \log ^ { \geq 0 } ( x ) \}$ $\operatorname { V a r } ^ { \geq 1 } ( p , V ) : = \operatorname* { m a x } \{ 1 , \operatorname { V a r } ( p , V ) \}$ , and $x _ { + } : = \operatorname* { m a x } \{ 0 , x \}$

## 3 Insuficiency of State-of-the-Art Methods

There are multiple routes to obtain an algorithm with regret of order $\sqrt { H ^ { 3 } S A K \log ( K ) ^ { \alpha } }$ . These include variants of UCBVI [Azar et al., 2017], optimistic variants of PSRL [Osband et al., 2013], and optimistic variants of Q-Learning [Watkins and Dayan, 1992]. Note that most state-of-the-art approaches are optimistic, with only a few exceptions [Lee and Oh, 2025, 2026]. In this work, we take the first route and design our minimax optimal algorithm as a variant of UCBVI. Our approach difers significantly from the existing family of UCBVI-methods, which is perhaps unsurprising since every existing algorithm sufers from polylog(K) factors and is therefore not minimax optimal (Table 1). In this section, we identify the culprit responsible for this suboptimality: the optimistic bonus design.

## 3.1 The UCBVI Algorithm

The UCBVI algorithm of Azar et al. [2017] consists of performing dynamic programming (DP) over the MDP with overestimated reward function $( r _ { h } + b _ { h } ^ { k } ) _ { h = } ^ { H }$ and empirical transition kernel

$$
\begin{array} { r } { P _ { h , s , a } ^ { k } ( s ^ { \prime } ) : = \left\{ \begin{array} { l l } { \frac { N _ { h } ^ { k } ( s , a , s ^ { \prime } ) } { N _ { h } ^ { k } ( s , a ) } } & { \mathrm { i f ~ } N _ { h } ^ { k } ( s , a ) > 0 \ , } \\ { \frac { 1 } { S } } & { \mathrm { i f ~ } N _ { h } ^ { k } ( s , a ) = 0 \ , } \end{array} \right. } \end{array}\tag{2}
$$

where $\begin{array} { r } { N _ { h } ^ { k } ( s , a , s ^ { \prime } ) : = \sum _ { i = 1 } ^ { k - 1 } \mathbb { I } ( s _ { h } ^ { i } = s , a _ { h } ^ { i } = a , s _ { h + 1 } ^ { i } = s ^ { \prime } ) } \end{array}$ and $\begin{array} { r } { N _ { h } ^ { k } ( s , a ) : = \sum _ { s ^ { \prime } } N _ { h } ^ { k } ( s , a , s ^ { \prime } ) } \end{array}$ are the empirical visit counts before episode k. To be more precise, given the above notation, one computes

$$
\begin{array} { l } { { Q _ { h } ^ { k } ( s , a ) : = \operatorname* { m i n } \left\{ { r _ { h } ( s , a ) + P _ { h , s , a } ^ { k } V _ { h + 1 } ^ { k } + b _ { h } ^ { k } ( s , a ) } \right\} , \hfill } } \\ { { \quad H - h + 1 \hfill } } \\ { { V _ { h } ^ { k } ( s ) : = \operatorname* { m a x } _ { a \in \mathcal { A } } Q _ { h } ^ { k } ( s , a ) , \hfill } } \end{array}
$$

with $V _ { H + 1 } ^ { k } ( s ) : = 0$ for every state $s \in { \mathcal { S } }$ . Then, during episode $k ,$ the learner plays the greedy policy with respect to $Q _ { h } ^ { k }$ :

$$
\pi _ { h } ^ { k } ( s _ { h } ^ { k } ) \in \arg \operatorname* { m a x } _ { a \in \mathcal { A } } Q _ { h } ^ { k } ( s _ { h } ^ { k } , a ) \ ,
$$

breaking ties arbitrarily.

## 3.2 Why log(K) Appears in Variants of UCBVI

To derive minimax regret bounds for UCBVI, the bonus $( b _ { h } ^ { k } ) _ { h = 1 } ^ { H }$ is designed so that $V _ { h } ^ { k }$ upper bounds $V _ { h } ^ { \star }$ uniformly over steps $h \in [ H ]$ and episodes $k \in [ K ]$ with high probability. Typically, $b _ { h } ^ { k }$ is constructed in order to invoke an empirical Bernstein inequality [Maurer and Pontil, 2009]:

$$
b _ { h } ^ { k } ( s , a ) \asymp \sqrt { \frac { \mathrm { V a r } ( P _ { h , s , a } ^ { k } , V _ { h + 1 } ^ { k } ) } { N _ { h } ^ { k } ( s , a ) } \ell + \frac { H ^ { 2 } } { N _ { h } ^ { k } ( s , a ) } \ell } , \qquad \ell = \log \biggl ( \frac { K } { \delta } \biggr ) ,\tag{3}
$$

with the convention $b _ { h } ^ { k } ( s , a ) = + \infty$ for $N _ { h } ^ { k } ( s , a ) = 0$ . With this bonus, one usually obtains high-probability regret bounds of the form

$$
\mathbb { P } \bigg ( \mathrm { R e g } ( K ) \geq C \sqrt { H ^ { 3 } S A K \big ( \log ( K / \delta ) \big ) ^ { \gamma } } \bigg ) \leq \delta
$$

for some $\gamma \geq 1$ . To convert the above into a bound in expectation, one can choose the confidence parameter as $\delta \asymp 1 / \mathrm { p o l y } ( K )$ ). The reason for this choice of $\delta$ is that, when the good event in the above display fails to hold, not much can be done other than bounding $\mathrm { R e g } ( K )$ by its worst possible value HK. Consequently, the

expected regret is bounded by

$$
\begin{array} { r } { \mathbb { E } [ \mathrm { R e g } ( K ) ] \lesssim \sqrt { H ^ { 3 } S A K \big ( \log ( K / \delta ) \big ) ^ { \gamma } } + \delta \cdot H K . } \end{array}
$$

This expected regret bound of order $\sqrt { H ^ { 3 } S A K \left( \log ( K ) \right) ^ { \gamma } }$ is sub-optimal by a factor $\log ( K ) ^ { \gamma / 2 }$

Where the $\log ( K ) ^ { \gamma / 2 }$ comes from. The cost of optimism must at some point be paid. In standard regret analyses, this fact appears via the sum of bonuses $\begin{array} { r } { \sum _ { k , h } b _ { h } ^ { k } \big ( s _ { h } ^ { k } , a _ { h } ^ { k } \big ) } \end{array}$ . Intuitively speaking, the regret bound sufers when the bonus is larger than necessary. Keeping only the dominant terms of the analysis, the bound of this term goes as such:

$$
\begin{array} { l } { \displaystyle \sum _ { k , k } b _ { i } ^ { k } ( s _ { k } ^ { k } , o _ { k } ^ { k } ) \stackrel { \scriptscriptstyle * } { \approx } \sqrt { \ell } \cdot \displaystyle \sum _ { k , h } \sqrt { \frac { \mathrm { V a r } ( P _ { k , s , a _ { k } ^ { k } } ^ { k } , s _ { k } ^ { k } ) } { N _ { k } ^ { k } ( s _ { k } ^ { k } , a _ { k } ^ { k } ) } } } \\ { \displaystyle \stackrel { \scriptscriptstyle \lesssim } \sqrt { \ell } \cdot \displaystyle \sum _ { k , h } \sqrt { \frac { \mathrm { V a r } ( P _ { k , s , a _ { k } ^ { k } } ^ { k } , \theta _ { k } ^ { k } , V _ { k + 1 } ^ { k } ) } { N _ { k } ^ { k } ( s _ { k } ^ { k } , a _ { k } ^ { k } ) } } } \\ { \displaystyle \stackrel { \scriptscriptstyle \lesssim } \sqrt { \ell } \cdot \displaystyle \sum _ { s , a , h } \sqrt { N _ { k } ^ { k + 1 } ( s , a ) \mathrm { V a r } ( P _ { k , s , a _ { k } } , V _ { k + 1 } ^ { \star } ) } } \\ { \displaystyle \stackrel { \scriptscriptstyle \lesssim } \sqrt { \ell } \cdot \sqrt { S A H \sum _ { s , a , h } N _ { k } ^ { K + 1 } ( s , a ) \mathrm { V a r } ( P _ { k , s , a _ { k } } , V _ { k + 1 } ^ { \star } ) } } \\ { \displaystyle \stackrel { \scriptscriptstyle \lesssim } \sqrt { \ell } \cdot \sqrt { S A H \sum _ { s , a , h } N _ { k } ^ { K + 1 } ( s , a ) \mathrm { V a r } ( P _ { k , s , a _ { k } } , V _ { k + 1 } ^ { \star } ) } } \\ { \displaystyle \stackrel { \scriptscriptstyle \lesssim } \sqrt { \ell } \cdot \sqrt { H ^ { 3 } S A K } } \\ { \displaystyle \stackrel { \scriptscriptstyle \lesssim } \sqrt { \ell } \cdot \sqrt { H ^ { 3 } S A K } } \end{array}
$$

where (°1) unfolds the definition of the bonus $( b _ { h } ^ { k } ) ; ( ^ { \circ } 2 )$ approximates various estimators by their true values; (°3) rewrites the sum over triplets $( s , a , h )$ and integrates; (°4) follows from Cauchy–Schwarz; and (°5) follows by a well-known lemma of the field.

In the final bound after (°5), we recognize a term scaling as the lower bound, as well as an extra factor $\sqrt { \ell } \asymp \log ( K ) ^ { \gamma / 2 }$ . This factor $\sqrt { \ell }$ was pulled from the start at (°1) and is therefore incompressible; we pay it as soon as we have to bound $\begin{array} { r } { \sum _ { k , h } b _ { h } ^ { k } \big ( s _ { h } ^ { k } , a _ { h } ^ { k } \big ) } \end{array}$ . In other words, prior algorithms which use bonuses of the form (3) cannot attain the minimax lower bound. We emphasize that this suboptimality is not simply an artifact of the above analysis, and that the resulting log factor in the regret is unavoidable because the fixed-log bonus is too large. This claim is supported by lower bounds in the bandits setting for the UCB algorithm, which also has a fixed log factor in its bonus and is similarly doomed to incur an extra log factor in its regret [Kalvit and Zeevi, 2021].

## 4 Variance-Aware Time-Uniform Concentration

Section 3 identifies the fixed logarithmic factor in standard UCBVI bonuses as an obstacle to achieving minimax optimality. Now, we introduce the main concentration tool that we use to overcome this obstacle.

In stochastic bandits, UCB [Auer et al., 2002] was made minimax-optimal via MOSS [Audibert and Bubeck, 2009], which is essentially the same algorithm with optimistic bonuses tailored to guarantees in expectation. Because UCBVI takes its roots in UCB, it is only natural to generalize MOSS’s bonuses to our setting. The essence of MOSS’s bonuses is dividing by visit counts within the logarithmic factors, replacing the static $\log ( 1 / \delta )$ with a term of the form $\log ^ { \geq 0 } ( 1 / ( n \delta ) )$ , where n is an appropriate visit count. This modification allows the logarithmic terms to shrink as observations accumulate, rather than always paying the same factor. By taking $\delta \asymp 1 / K$ , it follows that the log factor vanishes when $n \asymp K$ is large. As such and with careful analysis, the factor $\sqrt { \ell }$ from the computation (°1–5) in Section 3 becomes removable.

We now face a new challenge: these smaller bonuses need not preserve optimism. Instead of ensuring that an estimate remains optimistic, we will control how far below the true value it can fall. The following variance-aware, time-uniform concentration inequality provides this control. We defer the proof to Appendix C.

Theorem 4.1 (Variance-aware MOSS concentration). Let $( X _ { n } ) _ { n = 1 } ^ { \infty }$ be i.i.d. random variables such that

$$
\mathbb { E } [ X _ { i } ] = \mu , \qquad \operatorname { V a r } ( X _ { i } ) = \sigma ^ { 2 } , \qquad | X _ { i } - \mu | \leq c \quad a . s .
$$

For $n \geq 1$ , write $\begin{array} { r } { \hat { \mu } _ { n } : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } X _ { i } } \end{array}$ . Set $\alpha _ { 0 } = 4$ and $\beta _ { 0 } = 2 / 3$ . For every $\delta \in ( 0 , 1 )$ , the random variable

$$
Z _ { n } : = \mu - \hat { \mu } _ { n } - \sqrt { \frac { \alpha _ { 0 } \sigma ^ { 2 } } { n } \log ^ { 2 0 } \left( \frac { 1 } { n \delta } \right) } - \frac { \beta _ { 0 } c } { n } \log ^ { \geq 0 } \left( \frac { 1 } { n \delta } \right)
$$

satisfies $\begin{array} { r } { \mathbb { P } ( \exists n \geq 1 : Z _ { n } \geq \Delta ) \leq 1 6 \delta \left( \frac { \sigma ^ { 2 } } { \Delta ^ { 2 } } + \frac { c } { \Delta } \right) } \end{array}$ for every error level $\Delta > 0$

Theorem 4.1 is a time-uniform analogue of Bernstein’s inequality with linear drift (the quantity $\Delta )$ . It bounds the probability that the empirical mean plus a confidence interval underestimates the true mean by at least $\Delta$ at any sample size. That is, the bound applies at every sample size without a union bound.

To see how this result guides the bonus design, fix a triplet $( h , s , a )$ , write $P = P _ { h , s , a } $ , and let $\hat { P } = P _ { h , s , a , n }$ be its corresponding empirical transition row after n observed transitions from $( h , s , a )$ . In previous works that require optimism, the bonus is designed to be an upper bound to $\left( P - \hat { P } \right) V _ { h + 1 } ^ { \star }$ . Applying the usual Bernstein’s inequality and then taking a union bound over n leads to the fixed log factor discussed in Section 3. Theorem 4.1 instead suggests a bonus of the form

$$
\sqrt { \frac { \alpha _ { 0 } \mathrm { V a r } \big ( \hat { P } , V _ { h + 1 } ^ { k } \big ) } { n } \log ^ { \ge 0 } \bigg ( \frac { 1 } { n \delta } \bigg ) + \frac { \beta _ { 0 } c } { n } \log ^ { \ge 0 } \bigg ( \frac { 1 } { n \delta } \bigg ) } \ .
$$

When $\hat { P } \approx P$ and $\mathbb { V } _ { h + 1 } ^ { k } \approx V _ { h + 1 } ^ { \star }$ , the variance estimate is approximately $\mathrm { V a r } \big ( P , V _ { h + 1 } ^ { \star } \big )$ , and Theorem 4.1 provides control on how $\left( P - \hat { P } \right) V _ { h + 1 } ^ { \star }$ relates to the bonus. The exact bonus we use in our algorithm, which we present in the next section, has roughly the same form but requires some additional care.

## 5 Our Algorithm: MORE-UCBVI

In this section we introduce MORE-UCBVI (Minimax Optimal Regret in Expectation with UCBVI), given in Algorithm 1. We build on the UCBVI framework described in Section 3. To recap, before each episode k we update an empirical transition kernel according to (2) then use it in value iteration to compute our value estimates $Q ^ { k }$ and $V ^ { k }$ . Then for the duration of the episode we act greedily with respect to $Q ^ { k }$ , with ties between best actions broken arbitrarily.

The diference between the original UCBVI of Azar et al. [2017] and our algorithm lies in the bonus. Write $n _ { h } ^ { k } ( s , a ) : = \operatorname* { m a x } \bigl \{ 1 , N _ { h } ^ { k } ( s , a ) \bigr \} , B : = H K / ( S A )$ . For a distribution $p \in \Delta ( \mathcal { S } )$ , visit count $n \geq 1$ , and value function $V \in \mathbb { R } ^ { s }$ , we define a bonus function

$$
b ( p , n , V ) : = \operatorname* { m a x } \Bigl \{ b _ { 1 } ( p , n , V ) , b _ { 2 } ( p , n , V ) \Bigr \}\tag{4}
$$

where

$$
b _ { 1 } ( p , n , V ) : = \sqrt { \frac { \widetilde { \mathrm { o V a r } } ( p , n , V ) } { n } \log ^ { 2 1 } \biggl ( \frac { B } { n \widetilde { \mathrm { V a r } } ( p , n , V ) } \biggr ) }
$$

$$
b _ { 2 } ( p , n , V ) : = \frac { \beta H ^ { 2 } S L } { n } \quad \mathrm { w i t h } \quad L : = \log ( 1 0 H S ^ { 2 } A K ^ { 4 } ) \ .
$$

```latex
Algorithm 1: MORE-UCBVI: Minimax Optimal Regret in Expectation for UCBVI
for $k = 1 , \ldots , K$ do
Compute empirical kernels $P ^ { k }$ , see $\operatorname { E q . } \ ( 2 )$
Set $\bar { V } _ { H + 1 } ^ { k } ( s ) \bar {  } 0$ for all states s
for $h \stackrel { \cdot } { = } H , \ldots , 1$ do $/ /$ optimistic DP
$Q _ { h } ^ { k } ( s , a ) \gets \operatorname* { m i n } \bigl \{ r _ { h } ( s , a ) + P _ { h , s , a } ^ { k } V _ { h + 1 } ^ { k } + b _ { h } ^ { k } ( s , a ) , H - h + 1 \bigr \}$ for all pairs (s, a)
$V _ { h } ^ { k } ( s ) \gets \operatorname* { m a x } _ { a } Q _ { h } ^ { k } ( s , a )$ for all states s
end
Observe the initial state $s _ { 1 } ^ { k }$
for $h = 1 , \ldots , H$ do // run episode
Take action $a _ { h } ^ { k } = \operatorname { a r g m a x } _ { a } Q _ { h } ^ { k } ( s _ { h } ^ { k } , a )$
Observe the new state $s _ { h + 1 } ^ { k }$
end
end
```

Here, $\alpha = 6 4 , \beta = 4 0 9 6 , B = H K / ( S A )$ , and $\widetilde { \mathrm { V a r } }$ is a high probability upper-bound of the variance:

$$
\widetilde { \mathrm { V a r } } ( p , n , V ) : = 2 \mathrm { V a r } ^ { \geq 1 } ( p , V ) + { \frac { 6 4 H ^ { 2 } S L } { n } } ~ .
$$

Then the bonus used by the algorithm to compute $Q _ { h } ^ { k } ( s , a )$ before episode k is simply the bonus function evaluated at the corresponding empirical transition kernel, count, and next-step value function:

$$
b _ { h } ^ { k } ( s , a ) : = b \big ( P _ { h , s , a } ^ { k } , n _ { h } ^ { k } ( s , a ) , V _ { h + 1 } ^ { k } \big ) \ .
$$

We remark that B can be interpreted as an exploration budget because the logarithmic factor decreases to 1 as the visit count increases to $B / \widetilde { \mathrm { V a r } }$ . Choosing larger B results in larger bonuses, while decreasing B yields worse under-optimism. The choice $B = H K / ( S A )$ optimally balances this tradeof. optimize α and $\beta ,$ which could likely be significantly improved.

The bonus $b : = \operatorname* { m a x } \{ b _ { 1 } , b _ { 2 } \}$ is typical of sub-gamma random variables. The part $b _ { 1 }$ is due to a sub-gaussian tail that dominates for large n, while $b _ { 2 }$ is the shadow of a sub-exponential tail that outweighs $b _ { 1 }$ for small n. Note that $b _ { 1 }$ is a MOSS-style bonus while $b _ { 2 }$ is reminiscent of a UCB-style bonus. Because $b _ { 1 }$ ends up being dominant for large $n ,$ taking it in MOSS-style is crucial to have tight guarantees in expectation. On the other hand, taking $b _ { 2 }$ as UCB-style instead will be more appropriate for the use of good events. In addition, note the presence of an optimistic estimator of variance in $b _ { 1 }$ . Its role cannot be explained by Theorem 4.1 alone, so we defer further clarifying its presence to Section 7.2.

## 6 Main Theorem

We now present the main result of our paper.

Theorem 6.1. The regret of Algorithm 1 satisfies

$$
\begin{array} { r } { \mathbb { E } \left[ \mathrm { R e g } ( K ) \right] \leq O \left( \sqrt { H ^ { 3 } S A K } + H ^ { 3 } S ^ { 2 } A ( \log K ) ^ { 2 } \right) . } \end{array}
$$

We provide a proof sketch in Section 7 and a complete proof in Appendix A. The leading term in Theorem 6.1 matches the $\Omega \big ( \sqrt { H ^ { 3 } S A K } \big )$ minimax lower bound of Domingues et al. [2021] for time-inhomogeneous episodic MDPs up to a universal constant. Unlike the results of previous works summarized in Table 1, the leading term of our regret bound contains no extra logarithmic factors. Theorem 6.1 is therefore the first result to completely close the gap between upper and lower bounds (for K greater than a polynomial of H, S, A).

An application to best policy identification. We next provide an application of our main result to the problem of best-policy identification [Domingues et al., 2021]. We say that an algorithm is $( \varepsilon , \delta ) – \mathrm { P A C }$ if, for every MDP, it terminates after a finite number of episodes and returns a policy $\hat { \pi } = ( \hat { \pi } ^ { h } ) _ { h = 1 } ^ { H }$ satisfying

$$
\mathbb { P } \left( V _ { 1 } ^ { \star } ( \mu ) - V _ { 1 } ^ { \hat { \pi } } ( \mu ) \leq \varepsilon \right) \geq 1 - \delta \ .
$$

Here $\begin{array} { r } { V _ { 1 } ^ { \hat { \pi } } ( \mu ) = \sum _ { s } \mu ( s ) V _ { 1 } ^ { \hat { \pi } } ( s ) } \end{array}$ is the expected value function with respect to the starting distribution $\mu ,$ and likewise $\begin{array} { r } { V _ { 1 } ^ { \star } ( \mu ) = \sum _ { s } \mu ( s ) V _ { 1 } ^ { \star } ( s ) } \end{array}$

Corollary 6.2. There exists an algorithm which, for any $\varepsilon \in ( 0 , H ] , \delta \in ( 0 , 1 ] , \ i s \ ( \varepsilon , \delta ) { \cdot } P A C$ for best-policy identification and uses at most

$$
O \left( { \frac { H ^ { 3 } S A L _ { \delta } } { \varepsilon ^ { 2 } } } + { \frac { H ^ { 2 } L _ { \delta } ^ { 2 } } { \varepsilon ^ { 2 } } } + H ^ { 3 } S ^ { 3 } A ( \log ( e H S A ) ) ^ { 4 } L _ { \delta } \right)
$$

episodes, where $L _ { \delta } = \log ( 2 / \delta )$

In the regime where ε is suficiently small and $\delta$ is not exponentially small in $S A H .$ , specifically $\varepsilon \ \leq$ $\frac { 1 } { S ( \log ( e H S A ) ) ^ { 2 } }$ and $\delta \ge \exp ( - S A H )$ , the total number of episodes used is $O \left( H ^ { 3 } S A \log ( \frac { 2 } { \delta } ) \varepsilon ^ { - 2 } \right)$ . To the best of our knowledge this is the first result to match the $\Omega \left( H ^ { 3 } S A \log ( \textstyle { \frac { 1 } { \delta } } ) \varepsilon ^ { - 2 } \right)$ minimax lower bound shown by Domingues et al. [2021] in some regimes, improving the logarithmic factors relative to prior work.

Corollary 6.2 follows easily from Theorem 6.1 by applying standard regret-to-PAC techniques [Kaufmann et al., 2021]: we can run Algorithm 1 for suficiently many episodes $K _ { 0 }$ so that $\mathbb { E } [ \mathrm { R e g } ( K _ { 0 } ) / K _ { 0 } ] \le O ( \varepsilon )$ , use Markov’s inequality to conclude that a uniformly randomly selected policy from $\pi _ { 1 } , . . . , \pi _ { K _ { 0 } }$ has suboptimality $O ( \varepsilon )$ with at least constant probability, and then boost the success probability by repeating ${ \cal O } ( \log ( 1 / \delta ) )$ times, evaluating all policies to error $O ( \varepsilon )$ and returning the best one. We refer to Appendix D for the formal algorithm statement, proof, and more discussion of related work on best-policy identification and the regimes where the lower bound of Domingues et al. [2021] holds.

## 7 Proof Sketch

In this section we sketch the proof of Theorem 6.1, focusing on the main technical contributions. We defer the full analysis to Appendix A.

Using standard arguments (see Azar et al. [2017], Zhang et al. [2021] and Lemma $\mathrm { A . 1 } )$ , we decompose the expected regret into two

$$
\mathbb { E } [ \mathrm { R e g } ( K ) ] \leq \mathbb { E } [ \mathcal { T } ^ { - } ] + \mathbb { E } [ \mathcal { T } ^ { + } ] ,\tag{5}
$$

where

$$
\mathcal { T } ^ { - } : = \sum _ { k } \big ( V _ { 1 } ^ { \star } ( s _ { 1 } ^ { k } ) - V _ { 1 } ^ { k } ( s _ { 1 } ^ { k } ) \big ) _ { + } , \qquad \mathcal { T } ^ { + } : = \sum _ { k , h } \Big ( \Big ( P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } ^ { k } - P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } \Big ) V _ { h + 1 } ^ { k } + b _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) \Big ) _ { + } .
$$

The term $\mathcal { T } ^ { - }$ quantifies the total cost due to the optimistic value $V _ { 1 } ^ { k } ( s _ { 1 } ^ { k } )$ underestimating the true value $V _ { 1 } ^ { \star } ( s _ { 1 } ^ { k } )$ , and is therefore a cost of under-optimism. That is, as the bonus decreases and the algorithm becomes less optimistic, $\mathcal { T } ^ { - }$ increases. Meanwhile, ${ \mathcal { T } } ^ { + }$ quantifies how much the optimistic value $V _ { 1 } ^ { k } ( s _ { 1 } ^ { k } )$ tends to overshoot the true value $V _ { 1 } ^ { \star } ( s _ { 1 } ^ { k } )$ , and is therefore a cost of over-optimism; the larger the bonus, the larger ${ \mathcal { T } } ^ { + }$ . To have the best regret possible, these two terms must be balanced. In particular, and in opposition to past analysis such as Azar et al. [2017] and Zhang et al. [2021], we will see that the MOSS-style bonuses no longer ensure that $\mathcal { T } ^ { - }$ vanishes.

## 7.1 Bounding $\mathcal { T } ^ { - } \colon$ Under-Optimism

As mentioned in Section 3, optimistic algorithms choose bonuses large enough to ensure that $V _ { h } ^ { k } \geq V _ { h } ^ { \star }$ simultaneously for all $k , h \mathrm { ~ - ~ }$ and therefore $\mathcal { T } ^ { - } \leq 0 -$ with high probability. We first recall the role of monotonicity in this argument before explaining what changes for our smaller MOSS-style bonus.

The argument of UCBVI and its successors. Consider a UCB-style bonus $b _ { \mathrm { u c b } }$ for which the Bellman update $V \mapsto p V + b _ { \mathrm { u c b } } ( p , n , V )$ is monotone, as in the analysis of MVP [Zhang et al., 2021]. Fix k, and let $V _ { h } ^ { k } , h \in [ H + 1 ]$ , denote the value estimates computed before the k-th episode using UCBVI with the bonus $b _ { \mathrm { u c b } }$ . The proof shows $V _ { h } ^ { k } \geq V _ { h } ^ { \star }$ ∀h by backwards induction over h. The base case is of course trivial because $V _ { H + 1 } ^ { k } = V _ { H + 1 } ^ { \star } = 0$ . Then, for the inductive case, suppose that $V _ { h + 1 } ^ { k } \geq V _ { h + 1 } ^ { \star }$ and let s be arbitrary.

For ease of notation, write $a ^ { \star } = \pi _ { h } ^ { \star } ( s ) , P = P _ { h , s , a ^ { \star } } , \hat { P } = P _ { h , s , a ^ { \star } } ^ { k }$ <sub>⋆</sub> and $n = n _ { h } ^ { k } ( s , a ^ { \star } )$ . If $V _ { h } ^ { k } ( s ) = H - h + 1$ 2 the claim follows immediately. Otherwise, with high probability,

$$
\begin{array} { r l } & { V _ { h } ^ { k } ( s ) \overset { \circ _ { 1 } } { \geq } r _ { h } ( s , a ^ { \star } ) + \hat { P } V _ { h + 1 } ^ { k } + b _ { \mathrm { u c b } } \big ( \hat { P } , n , V _ { h + 1 } ^ { k } \big ) } \\ & { \qquad \overset { \circ _ { 2 } } { \geq } r _ { h } ( s , a ^ { \star } ) + \hat { P } V _ { h + 1 } ^ { \star } + b _ { \mathrm { u c b } } \big ( \hat { P } , n , V _ { h + 1 } ^ { \star } \big ) } \\ & { \qquad \overset { \circ _ { 3 } } { \geq } r _ { h } ( s , a ^ { \star } ) + P V _ { h + 1 } ^ { \star } = V _ { h } ^ { \star } ( s ) . } \end{array}
$$

Here, the inequality (°1) is by definition of $V _ { h } ^ { k }$ , and the inequality (°3) is because $b _ { \mathrm { u c b } }$ is large enough so that

$$
\bigl ( P - \hat { P } \bigr ) V _ { h + 1 } ^ { \star } \leq b _ { \mathrm { u c b } } \bigl ( \hat { P } , n , V _ { h + 1 } ^ { \star } \bigr )\tag{6}
$$

with high probability (simultaneously for all $h , s , a , n )$ . The inequality (°2) crucially depends on the aforementioned monotonicity of the Bellman update. In particular, $V _ { h + 1 } ^ { k } \geq V _ { h + 1 } ^ { \star }$ implies $\hat { P } V _ { h + 1 } ^ { k } + b _ { \mathrm { u c b } } ( \hat { P } , n , V _ { h + 1 } ^ { k } ) \geq$ $\hat { P } V _ { h + 1 } ^ { \star } + b _ { \mathrm { u c b } } ( \hat { P } , n , V _ { h + 1 } ^ { \star } )$ . This monotonicity allows us to replace the data-dependent vector $V _ { h + 1 } ^ { k }$ by the fixed vector $V _ { h + 1 } ^ { \star }$ before applying concentration, thereby avoiding potentially loose covering arguments over value functions.

Controlling the cost of under-optimism. For our smaller MOSS-style bonus $b ,$ a concentration inequality of the form of Eq. (6) need not hold simultaneously for all $h , s , a ,$ n with high enough probability. We therefore cannot assume $V _ { h + 1 } ^ { k } \geq V _ { h + 1 } ^ { \star }$ at the next step of the induction, even though the function $V \mapsto \hat { P } V + b ( \hat { P } , n , V )$ is still monotone (see Corollary A.5). However, while we cannot order $V _ { h + 1 } ^ { k }$ and $V _ { h + 1 } ^ { \star }$ in a strict manner, we can instead control how a next-step shortfall propagates through the Bellman update. This is made possible by the following result, which is a slightly simplified version of Corollary A.6.

Lemma 7.1 (Stability of the Bellman Update). Let $p \in \Delta ( \mathcal { S } ) , n \geq 1$ , and $V , V ^ { \prime } \in [ 0 , H ] ^ { s }$ . Then

$$
p V + b ( p , n , V ) \geq p V ^ { \prime } + b ( p , n , V ^ { \prime } ) - \left( 1 + \frac { 1 } { 1 6 H } \right) p ( V ^ { \prime } - V ) _ { + } .
$$

Applying this lemma with $V = V _ { h + 1 } ^ { k }$ and $V ^ { \prime } = V _ { h + 1 } ^ { \star }$ gives nearly the same substitution as monotonicity, except that now $V _ { h + 1 } ^ { k }$ and $V _ { h + 1 } ^ { \star }$ may be unordered. Moreover, this substitution only incurs an extra cost of approximately $\hat { P } ( V _ { h + 1 } ^ { \star } - V _ { h + 1 } ^ { k } ) _ { + }$ . Roughly speaking, the previous argument with $b _ { \mathrm { u c b } }$ was that $^ { 6 \cdot } V _ { h + 1 } ^ { k }$ is optimistic so $V _ { h } ^ { k }$ is optimistic”. The new argument with b states that $^ { \mathfrak { a } } V _ { h + 1 } ^ { k }$ is nearly optimistic so $V _ { h } ^ { k }$ is nearly optimistic”.

Explicitly, under a certain high probability good event E, stability yields

$$
( V _ { h } ^ { \star } ( s ) - V _ { h } ^ { k } ( s ) ) _ { + } \leq \Big ( ( P - \hat { P } ) V _ { h + 1 } ^ { \star } - \bar { b } \big ( \hat { P } , n , V _ { h + 1 } ^ { \star } \big ) \Big ) _ { + } + \Big ( 1 + \frac { 1 } { 4 H } \Big ) P \big ( V _ { h + 1 } ^ { \star } - V _ { h + 1 } ^ { k } \big ) _ { + } .
$$

There are three notable adjustments which require explaining: instead of b we have ${ \bar { b } } ;$ instead of $( 1 + 1 / ( 1 6 H ) )$ we have $( 1 + 1 / ( 4 H ) )$ ; and instead of $\hat { P }$ we have $P .$ To obtain the desired bound on $( V _ { 1 } ^ { \star } ( s ) - V _ { 1 } ^ { k } ( s ) ) _ { + }$ we will unroll the above inequality backwards from $h = H + 1$ to 1, and for reasons that will shortly be apparent, it is important that the next-step shortfall is weighted by $P$ instead of ${ \hat { P } } .$ We accomplish this change by splitting of half the bonus and using it to approximately upper bound $\left| \hat { P } - P \right|$ under the good event E. $\bar { b } = b / 2$ is the remaining half of the bonus, and the “approximate” upper bounding is what necessitates changing $( 1 + 1 / ( 1 6 H ) )$ to $( 1 + 1 / ( 4 H ) )$ .

Since $V _ { H + 1 } ^ { k } = V _ { H + 1 } ^ { \star } = 0$ , as mentioned above, we can unroll the previous inequality backwards. Using $( 1 + 1 / ( 4 H ) ) ^ { H } \leq 2$ , what remains is an occupancy weighted sum of only the first terms. Specifically,

$$
\mathbb { E } \big [ \mathbb { I } ( \mathcal { E } ) ( V _ { 1 } ^ { \star } ( s _ { 1 } ^ { k } ) - V _ { 1 } ^ { k } ( s _ { 1 } ^ { k } ) ) _ { + } \big ] \lesssim \sum _ { h , s , a } \mathbb { P } _ { \mu } ^ { \pi ^ { \star } } \binom { s _ { h } = s } { a _ { h } = a } \mathbb { E } \Big [ \mathbb { I } ( \mathcal { E } ) \Big ( ( P _ { t } - P _ { t } ^ { k } ) V _ { h + 1 } ^ { \star } - \bar { b } ( P _ { t } ^ { k } , n _ { t } ^ { k } , V _ { h + 1 } ^ { \star } ) \Big ) \Big ] .
$$

Here, $\mathbb { P } _ { \mu } ^ { \pi ^ { \star } } ( s _ { h } = s , a _ { h } = a )$ is the probability of visiting $( s , a )$ at step h in the true MDP, starting from a state randomly drawn from $\mu$ then following $\pi ^ { \star }$ . These occupancies belong to the true MDP under $\pi ^ { \star }$ and are thus deterministic. Consequently, we only need an expected shortfall bound for each fixed $( h , s , a )$ , rather than a simultaneous guarantee for all $( h , s , a )$ . Note that if we kept $\hat { P }$ instead of $P ,$ we would encounter issues due to data-dependent occupancies.

We obtain the following lemma by applying the time-uniform concentration inequality Theorem 4.1 and integrating its tail bound. Time-uniformity is essential because $n _ { h } ^ { k } ( s , a )$ is chosen adaptively, and we canno aford to take a union bound over its possible values.

Lemma 7.2. For each episode k and triplet $t \equiv ( h , s , a )$ 2

$$
\mathbb { E } \Big [ \mathbb { I } ( \mathscr { E } ) \big ( ( P _ { t } - P _ { t } ^ { k } ) V _ { h + 1 } ^ { \star } - \bar { b } ( P _ { t } ^ { k } , n _ { t } ^ { k } , V _ { h + 1 } ^ { \star } ) \big ) _ { + } \Big ] \lesssim \frac { 1 } { \sqrt { B } } \mathrm { V a r } ^ { \geq 1 } ( P _ { t } , V _ { h + 1 } ^ { \star } ) .
$$

All that remains is accounting for $\mathcal { E } ^ { c }$ , a standard argument to bound a sum of variance terms, summing over $K$ , and plugging in $B = H K / ( S A )$ . These imply

$$
\mathbb { E } [ \mathcal { T } ^ { - } ] \lesssim \frac { K H ^ { 2 } } { \sqrt { B } } + \frac { H } { K ^ { 2 } } \lesssim \sqrt { H ^ { 3 } S A K } .
$$

## 7.2 Bounding ${ \mathcal { T } } ^ { + }$ : Over-Optimism

To bound the over-optimism error ${ \mathcal { T } } ^ { + }$ , we utilize the splitting $V _ { h + 1 } ^ { k } = V _ { h + 1 } ^ { \star } + ( V _ { h + 1 } ^ { k } - V _ { h + 1 } ^ { \star } )$ . Specifically, we upper bound $\mathcal { T } ^ { + }$ by the sum of three terms:

$$
\begin{array} { r l } & { \mathcal { T } _ { 1 } ^ { + } = \displaystyle \sum _ { k , h } \left( \left( P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } ^ { k } - P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } \right) V _ { h + 1 } ^ { \star } \right) _ { + } , } \\ & { \mathcal { T } _ { 2 } ^ { + } = \displaystyle \sum _ { k , h } \left. \left( P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } ^ { k } - P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } \right) \left( V _ { h + 1 } ^ { k } - V _ { h + 1 } ^ { \star } \right) \right. , } \\ & { \mathcal { T } _ { 3 } ^ { + } = \displaystyle \sum _ { k , h } b _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) . } \end{array}
$$

By splitting like so, we are allowed to apply concentration results to the fixed vector $V _ { h + 1 } ^ { \star }$ , thereby avoiding $\mathcal { T } _ { 1 } ^ { + }$ issues of dependence between the empirical transition kernel $P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } ^ { k }$ and the value estimate $V _ { h + 1 } ^ { k }$

Overall, the decomposition of $\mathcal { T } _ { 1 } ^ { + } , \mathcal { T } _ { 2 } ^ { + } , \mathcal { T } _ { 3 } ^ { + }$ is standard and can be found in Zhang et al. [2020] for instance. We can handle $\mathcal { T } _ { 2 } ^ { + }$ using a standard variance analysis along the learner’s trajectory. For the other sums $\mathcal { T } _ { 1 } ^ { + }$ and ${ \mathcal { T } } _ { 3 } ^ { + }$ , however, we must develop sharper arguments to avoid extra logarithms in the leading term. This is what we discuss in the remainder of the section. The detailed analyses of $\mathbb { E } [ \mathcal { T } _ { 1 } ^ { + } ] , \mathbb { E } [ \mathcal { T } _ { 2 } ^ { + } ]$ , and

$\mathbb { E } [ \mathcal { T } _ { 3 } ^ { + } ]$ are found in Appendix A.5.

Bounding $\mathcal { T } _ { 1 } ^ { + }$ : Concentration of cumulative deviations. For a fixed triplet $( h , s , a )$ , bounding $d _ { n } : = ( \hat { P } _ { h , s , a , n } - P _ { h , s , a } ) V _ { h + 1 } ^ { \star }$ simultaneously for all $n \in [ K ]$ would require a union bound over K events, incurring a factor of $\log ( K )$ , or at least log $\log ( K )$ by the law of iterated logarithms. Instead, we develop a general result — which may be of independent interest — to directly bound the sum of deviations $\textstyle \sum _ { n } d _ { n }$ (proved in Appendix A.5, Proposition A.12).

Proposition 7.3 (Concentration on sums with slackness). Let $( X _ { n } ) _ { n = 1 } ^ { \infty }$ be i.i.d. random variables with mean $\mu ,$ variance $\sigma ^ { 2 }$ , and $| X _ { i } - \mu | \leq c$ almost surely. Write $\begin{array} { r } { \hat { \mu } _ { n } = n ^ { - 1 } \sum _ { i = 1 } ^ { n } X _ { i } } \end{array}$ the empirical mean. There is a universal constant $C > 0$ such that, for every $\Delta > 0$

$$
\sum _ { n = 1 } ^ { \infty } \mathbb { E } [ ( \hat { \mu } _ { n } - \mu - \Delta ) _ { + } ] \leq C \left( \frac { \sigma ^ { 2 } } { \Delta } + c \right) .
$$

The following corollary helps bound $\mathcal { T } _ { 1 } ^ { + }$ . It follows by adding and subtracting a slackness parameter $\Delta$ then tuning to the optimal choice $\Delta = \sqrt { \sigma ^ { 2 } / \mathbb { E } [ N ] }$

Corollary 7.4 (Concentration on sums). Under the setting of Proposition 7.3, for any nonnegative integervalued random variable N with $\mathbb { E } [ N ] < \infty$ ,

$$
\mathbb { E } \left[ \sum _ { n = 1 } ^ { N } ( \hat { \mu } _ { n } - \mu ) _ { + } \right] \lesssim \sqrt { \sigma ^ { 2 } \mathbb { E } [ N ] } + c .
$$

Recall that $N _ { h } ( s , a )$ is the total number of visits to $( h , s , a )$ over the K episodes. After rearranging the sum over $k , h$ into a sum over $h , s , a , n$ , Corollary 7.4 gives

$$
\mathbb { E } \Big [ \mathbb { I } ( \mathcal { E } ) \big ( ( P _ { t } - P _ { t } ^ { k } ) V _ { h + 1 } ^ { \star } - \bar { b } ( P _ { t } ^ { k } , n _ { t } ^ { k } , V _ { h + 1 } ^ { \star } ) \big ) _ { + } \Big ] \lesssim \frac { 1 } { \sqrt { B } } \mathrm { V a r } ^ { \geq 1 } ( P _ { t } , V _ { h + 1 } ^ { \star } ) .
$$

where the second inequality follows from Cauchy-Schwarz and Lemma $\mathrm { A . 9 }$

Bounding $\mathbb { E } [ \mathcal { T } _ { 3 } ^ { + } ]$ : Summing the bonus. Up to lower order terms incurred by converting from empirical estimates to true values, we have

$$
b _ { h } ^ { k } ( s , a ) \approx \sqrt { \frac { \sigma _ { h } ( s , a ) ^ { 2 } } { n _ { h } ^ { k } ( s , a ) } \log ^ { \ge 1 } \biggl ( \frac { B } { n _ { h } ^ { k } ( s , a ) \sigma _ { h } ( s , a ) ^ { 2 } } \biggr ) }
$$

where $\sigma _ { h } ( s , a ) ^ { 2 } : = \mathrm { V a r } ^ { \geq 1 } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } )$ is the local transition variance. Hence, $\mathbb { E } [ \mathcal { T } _ { 3 } ^ { + } ]$ is approximately

$$
\sum _ { h , s , a } \mathbb { E } \left[ \sum _ { n = 1 } ^ { N _ { h } ( s , a ) } \sqrt { \frac { \sigma _ { h } ( s , a ) ^ { 2 } } { n } \log ^ { 2 1 } \left( \frac { B } { n \sigma _ { h } ( s , a ) ^ { 2 } } \right) } \right]\tag{7}
$$

Unlike the fixed logarithmic factor in the bonus in Section 3, here the logarithmic term $\log ^ { \ge 1 } ( B / ( n \sigma _ { h } ( s , a ) ^ { 2 } ) )$ decreases as the visit count n increases. Crucially, we compute the sum with the decreasing logarithmic terms directly, rather than upper bounding all logs by their largest value and pulling a $\sqrt { \log B }$ factor outside the sum. We formalize this in the following lemma.

Lemma 7.5 (Variance-normalized MOSS sum). For every $q > 0 , B > 0$ , and integer $N \geq 0$ ，

$$
\sum _ { n = 1 } ^ { N } \sqrt { \frac { q } { n } \log ^ { \ge 1 } \biggl ( \frac { B } { n q } \biggr ) } \le 6 \sqrt { B } + \frac { q N } { \sqrt { B } } .
$$

We remark that, as opposed to UCB-style bonuses, summing MOSS-style bonuses via Lemma 7.5 does not spawn logarithmic terms. The proof splits the sum according to $n \leq B / q$ and $n > B / q$ . For $n \leq B / q .$ , an integral comparison sums the bonuses to at most $6 { \sqrt { B } } .$ . For $n > B / q$ , the logarithmic factor equals 1, so each bonus is at most $q / { \sqrt { B } } ,$ giving the remaining $q N / \sqrt { B }$

With all the results so far, we can finally explain when the inclusion of the variance in the denominator of the logarithm of $b _ { 1 }$ in $\mathrm { E q . \ ( 4 ) }$ makes its play. Without it, the same integral comparison as before would yield a first term of $\begin{array} { r } { \sum _ { h , s , a } \sqrt { \sigma _ { h } ( s , a ) ^ { 2 } B } } \end{array}$ , whose worst-case bound is of order $H ^ { 2 } S A \sqrt { B } = \sqrt { H ^ { 5 } S A K }$ , a factor H larger than the desired $\sqrt { H ^ { 3 } S A K }$ rate. Including the variance in the denominator changes the range of the first sum from $n \leq B$ to $n \leq B / q$ . The integral comparison then gives $\sqrt { B }$ rather than $\sqrt { q B }$ , since the factor $\sqrt { q }$ in the bonus is exactly canceled by the factor $1 / { \sqrt { q } }$ from the shorter summation range.

Applying Lemma 7.5, we get

$$
\begin{array} { r l r } {  { ( \overline { { \boldsymbol { \imath } } } ) \lesssim H S A \sqrt { B } + \frac { 1 } { \sqrt { B } } \sum _ { h , s , a } \sigma _ { h } ( s , a ) ^ { 2 } \mathbb { E } [ N _ { h } ( s , a ) ] } } \\ & { } & { \lesssim H S A \sqrt { B } + \frac { H ^ { 2 } K } { \sqrt { B } } \lesssim \sqrt { H ^ { 3 } S A K } . } \end{array}
$$

The last line follows from a folklore result on the variance sum (Lemma A.9), then plugging in $\begin{array} { r } { B = \frac { H K } { S A } } \end{array}$

## 8 Conclusion

In this paper we match the minimax regret lower bound in episodic RL without extra polylogarithmic factors, using a MOSS-style bonus which incorporates visit counts and variance estimates. We believe that these algorithmic techniques, as well as our concentration results and analysis of the resulting under-optimism, may be more widely useful. Future directions include improving the additive “burn-in” regret term (similar to Zhang et al. [2025]) and handling the time-homogeneous setting.

## AI Acknowledgment

The problem formulation and most of the proof ideas were developed by the authors. The initial writing was completely done by the authors. ChatGPT 5.6 Sol and ChatGPT 6 Astra were used for the following: (i) they helped refine some of our arguments, including removing a log H factor by inserting the variance approximation Varg in the denominator of the log in the bonus, and it helped to remove a log S factor with the stability lemma; (ii) they helped with literature review, catching typos, improving the writing, and double-checking the correctness of proofs.

Any components which were assisted with LLMs were thoroughly reviewed and re-written with our own notation, exposition, and understanding. We take complete responsibility for the final content of this work including text, claims, or artifacts produced with the aid of LLMs.

## Acknowledgment

G. Zamir, M. Zurek, and Y. Chen acknowledge support by National Science Foundation grants CCF-2233152 and DMS-2023239, by a Cisco Systems Fellowship, and by a Vilas Associates Award.

## References

Rajeev Agrawal. Sample mean based index policies by o (log n) regret for the multi-armed bandit problem. Advances in applied probability, 27(4):1054–1078, 1995.

Jean-Yves Audibert and Sébastien Bubeck. Minimax policies for adversarial and stochastic bandits. In Colt, pages 217–226, 2009.

Peter Auer, Nicolo Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multiarmed bandit problem. Machine learning, 47(2):235–256, 2002.

Peter Auer, Thomas Jaksch, and Ronald Ortner. Near-optimal regret bounds for reinforcement learning. Advances in neural information processing systems, 21, 2008.

Mohammad Gheshlaghi Azar, Ian Osband, and Rémi Munos. Minimax regret bounds for reinforcement learning. In International conference on machine learning, pages 263–272. PMLR, 2017.

Victor Boone and Zihan Zhang. Achieving Tractable Minimax Optimal Regret in Average Reward MDPs. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 26728–26769. Curran Associates, Inc., 2024.

Stéphane Boucheron, Gábor Lugosi, and Pascal Massart. Concentration Inequalities: A Nonasymptotic Theory of Independence. Oxford University Press, 02 2013. ISBN 9780199535255. doi: 10.1093/acprof: oso/9780199535255.001.0001.

Hippolyte Bourel, Odalric Maillard, and Mohammad Sadegh Talebi. Tightening Exploration in Upper Confidence Reinforcement Learning. In Hal Daumé III and Aarti Singh, editors, Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 1056–1066. PMLR, July 2020.

Christoph Dann, Lihong Li, Wei Wei, and Emma Brunskill. Policy Certificates: Towards Accountable Reinforcement Learning. In Proceedings of the 36th International Conference on Machine Learning, pages 1507–1516. PMLR, May 2019. URL https://proceedings.mlr.press/v97/dann19a.html.

Christoph Dann, Teodor Vanislavov Marinov, Mehryar Mohri, and Julian Zimmert. Beyond value-function gaps: Improved instance-dependent regret bounds for episodic reinforcement learning. In Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan, editors, Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 1–12, 2021.

Omar Darwiche Domingues, Pierre Ménard, Emilie Kaufmann, and Michal Valko. Episodic reinforcement learning in finite mdps: Minimax lower bounds revisited. In Algorithmic Learning Theory, pages 578–598. PMLR, 2021.

Sarah Filippi, Olivier Cappé, and Aurélien Garivier. Optimism in Reinforcement Learning and Kullback-Leibler Divergence. 2010 48th Annual Allerton Conference on Communication, Control, and Computing (Allerton), pages 115–122, September 2010. arXiv: 1004.5229.

Chi Jin, Zeyuan Allen-Zhu, Sebastien Bubeck, and Michael Jordan. Is Q-Learning Provably Eficient? In Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018. URL https://proceedings.neurips.cc/paper\_files/paper/2018/hash/ d3b1fb02964aa64e257f9f26a31f72cf-Abstract.html.

Anand Kalvit and Assaf Zeevi. A closer look at the worst-case behavior of multi-armed bandit algorithms. In Advances in Neural Information Processing Systems, volume 34, pages 8807–8819, 2021. URL https: //arxiv.org/abs/2106.02126.

Emilie Kaufmann, Pierre Ménard, Omar Darwiche Domingues, Anders Jonsson, Edouard Leurent, and Michal Valko. Adaptive reward-free exploration. In Vitaly Feldman, Katrina Ligett, and Sivan Sabato,

editors, Proceedings of the 32nd International Conference on Algorithmic Learning Theory, volume 132 of Proceedings of Machine Learning Research, pages 865–891. PMLR, 16–19 Mar 2021. URL https: //proceedings.mlr.press/v132/kaufmann21a.html.

Cyrille Kone and Kevin Jamieson. Optimal Posterior Sampling for Policy Identification in Tabular Markov Decision Processes. In Proceedings of The 29th International Conference on Artificial Intelligence and Statistics, pages 766–774. PMLR, August 2026. URL https://proceedings.mlr.press/v300/kone26a. html.

Tor Lattimore and Csaba Szepesvári. Bandit algorithms. Cambridge University Press, 2020.

Harin Lee and Min-hwan Oh. Minimax optimal reinforcement learning with quasi-optimism. In International Conference on Learning Representations, volume 2025, pages 13184–13231, 2025.

Harin Lee and Min-hwan Oh. Unified framework of distributional regret in multi-armed bandits and reinforcement learning. arXiv preprint arXiv:2605.05102, 2026.

Andreas Maurer and Massimiliano Pontil. Empirical bernstein bounds and sample variance penalization, 2009. URL https://arxiv.org/abs/0907.3740.

Pierre Menard, Omar Darwiche Domingues, Anders Jonsson, Emilie Kaufmann, Edouard Leurent, and Michal Valko. Fast active learning for pure exploration in reinforcement learning. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 7599–7608. PMLR, 18–24 Jul 2021. URL https://proceedings.mlr.press/v139/menard21a.html.

Pierre Ménard, Omar Darwiche Domingues, Xuedong Shang, and Michal Valko. Ucb momentum q-learning: Correcting the bias without forgetting. In International Conference on Machine Learning, pages 7609–7618. PMLR, 2021.

Ian Osband, Daniel Russo, and Benjamin Van Roy. (more) eficient reinforcement learning via posterior sampling. Advances in Neural Information Processing Systems, 26, 2013.

Daniil Tiapkin, Denis Belomestny, Daniele Calandriello, Eric Moulines, Remi Munos, Alexey Naumov, Mark Rowland, Michal Valko, and Pierre Ménard. Optimistic posterior sampling for reinforcement learning with few samples and tight guarantees. Advances in Neural Information Processing Systems, 35:10737–10751, 2022.

Andrea Tirinzoni, Matteo Pirotta, and Alessandro Lazaric. A fully problem-dependent regret lower bound for finite-horizon mdps. CoRR, abs/2106.13013, 2021. URL https://arxiv.org/abs/2106.13013.

Andrea Tirinzoni, Aymen Al-Marjani, and Emilie Kaufmann. Optimistic PAC Reinforcement Learning: the Instance-Dependent View. In Proceedings of The 34th International Conference on Algorithmic Learning Theory, pages 1460–1480. PMLR, February 2023. URL https://proceedings.mlr.press/v201/ tirinzoni23a.html.

Andrew J Wagenmaker, Max Simchowitz, and Kevin Jamieson. Beyond no regret: Instance-dependent pac reinforcement learning. In Po-Ling Loh and Maxim Raginsky, editors, Proceedings of Thirty Fifth Conference on Learning Theory, volume 178 of Proceedings of Machine Learning Research, pages 358–418. PMLR, 02–05 Jul 2022. URL https://proceedings.mlr.press/v178/wagenmaker22a.html.

Christopher JCH Watkins and Peter Dayan. Q-learning. Machine learning, 8(3):279–292, 1992.

Zhihan Xiong, Ruoqi Shen, Qiwen Cui, Maryam Fazel, and Simon Du. Near-optimal randomized exploration for tabular markov decision processes. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 6358–6371. Curran Associates, Inc., 2022.

Haike Xu, Tengyu Ma, and Simon S. Du. Fine-grained gap-dependent bounds for tabular mdps via adaptive multi-step bootstrap. In Mikhail Belkin and Samory Kpotufe, editors, Conference on Learning Theory, COLT 2021, 15-19 August 2021, Boulder, Colorado, USA, Proceedings of Machine Learning Research, pages 4438–4472. PMLR, 2021.

Kunhe Yang, Lin F. Yang, and Simon S. Du. Q-learning with logarithmic regret. In Arindam Banerjee and Kenji Fukumizu, editors, The 24th International Conference on Artificial Intelligence and Statistics, AISTATS 2021, April 13-15, 2021, Virtual Event, Proceedings of Machine Learning Research, pages 1576–1584. PMLR, 2021.

Guy Zamir, Matthew Zurek, and Yudong Chen. Optimal variance-dependent regret bounds for infinite-horizon mdps. In Steve Hanneke and Tor Lattimore, editors, Proceedings of Thirty Ninth Conference on Learning Theory, volume 336 of Proceedings of Machine Learning Research, pages 7016–7061. PMLR, 29 Jun–03 Jul 2026.

Andrea Zanette and Emma Brunskill. Tighter problem-dependent regret bounds in reinforcement learning without domain knowledge using value function bounds. In International Conference on Machine Learning, pages 7304–7312. PMLR, 2019.

Zihan Zhang, Yuan Zhou, and Xiangyang Ji. Almost optimal model-free reinforcement learningvia referenceadvantage decomposition. Advances in Neural Information Processing Systems, 33:15198–15207, 2020.

Zihan Zhang, Xiangyang Ji, and Simon S Du. Is reinforcement learning more dificult than bandits? a near-optimal algorithm escaping the curse of horizon. In COLT, pages 4528–4531, 2021.

Zihan Zhang, Yuxin Chen, Jason Lee, and Simon S Du. Settling the sample complexity of online reinforcement learning. Journal of the ACM, 72(3):1–63, 2025.

## A Complete Proof of Theorem 6.1

In this section we prove our main result, Theorem 6.1.

## A.1 Overview of Notation and Conventions

We recall notation used throughout the proof. Unless stated otherwise, sums over $k$ and $h$ range over [K] and [H], respectively, and inequalities between vectors are understood componentwise. For each $( h , s , a ) \in [ H ] \times S \times A$ , let $( X _ { h , s , a , i } ) _ { i \geq 1 }$ be an i.i.d. sequence with law $P _ { h , s , a }$ . We take these sequences to be mutually independent and independent of the initial states $( s _ { 1 } ^ { k } ) _ { k = 1 } ^ { K }$ , which are drawn i.i.d. from $\mu$ Equivalently, the trajectory can be generated by revealing the next unused sample from the corresponding sequence whenever the learner visits $( h , s , a )$ . For every integer $n \geq 1$ , define the empirical row

$$
\hat { P } _ { h , s , a , n } ( s ^ { \prime } ) : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbb { I } \{ X _ { h , s , a , i } = s ^ { \prime } \} , \qquad s ^ { \prime } \in \mathcal { S } .\tag{A.1}
$$

This definition also applies to sample sizes that the learner does not reach.

The following table summarizes the transition and count notation:

Notation Meaning   
$P _ { h , s , a }$ True transition row at $( h , s , a )$   
$P _ { h , s , a } ^ { k }$ Transition row used by the algorithm in episode $k ,$ as defined in   
Eq. (2).   
$\hat { P } _ { h , s , a , n }$ Empirical row after the first n samples from $( h , s , a )$ , see $\mathrm { E q . \ ( A . 1 ) }$   
$N _ { h } ^ { k } ( s , a )$ Number of visits to $( h , s , a )$ before episode $k .$   
$n _ { h } ^ { k } ( s , a )$ Safeguarded count max $\{ 1 , N _ { h } ^ { k } ( s , a ) \}$   
$N _ { h } ( s , a )$ Total number of visits to $( h , s , a )$ after all K episodes.

In particular,

$$
N _ { h } ^ { k } ( s , a ) : = \sum _ { i = 1 } ^ { k - 1 } \mathbb { I } \{ s _ { h } ^ { i } = s , ~ a _ { h } ^ { i } = a \} , \qquad N _ { h } ( s , a ) : = \sum _ { k = 1 } ^ { K } \mathbb { I } \{ s _ { h } ^ { k } = s , ~ a _ { h } ^ { k } = a \} .
$$

When $N _ { h } ^ { k } ( s , a ) \ge 1$ , we have $P _ { h , s , a } ^ { k } = \hat { P } _ { h , s , a , N _ { h } ^ { k } ( s , a ) } = \hat { P } _ { h , s , a , n _ { h } ^ { k } ( s , a ) }$ . When $N _ { h } ^ { k } ( s , a ) = 0 $ , the algorithm instead uses the uniform row $P _ { h , s , a } ^ { k } ( s ^ { \prime } ) = 1 / S$ . We do not identify this initialization with $\hat { P } _ { h , s , a , 1 }$ , so we handle first visits separately when using concentration bounds for the empirical rows.

We also recall that $B = H K / ( S A )$ and $L = \log ( 1 0 H S ^ { 2 } A K ^ { 4 } )$ , and that

$$
\begin{array} { r } { \log ^ { \geq 0 } ( x ) : = \log ( \operatorname* { m a x } \{ 1 , x \} ) , \qquad \log ^ { \geq 1 } ( x ) : = \operatorname* { m a x } \{ 1 , \log ^ { \geq 0 } ( x ) \} , \qquad \mathrm { V a r } ^ { \geq 1 } ( p , V ) : = \operatorname* { m a x } \{ 1 , \mathrm { V a r } ( p , V ) \} . } \end{array}
$$

Powers and positive parts of vectors are taken componentwise.

The algorithm uses the bonus $b _ { h } ^ { k } ( s , a ) = b \big ( P _ { h , s , a } ^ { k } , n _ { h } ^ { k } ( s , a ) , V _ { h + 1 } ^ { k } \big )$ . The bonus function b is defined by $b = \operatorname* { m a x } \{ b _ { 1 } , b _ { 2 } \}$ , where $b _ { 1 }$ and $b _ { 2 }$ are defined as in Section 5. For the under-optimism analysis we define the half-bonus

$$
{ \bar { b } } ( p , n , V ) : = { \frac { 1 } { 2 } } b ( p , n , V ) .
$$

As briefly discussed in Section $^ { 7 , }$ we will use $\bar { b }$ to compare value functions and reserve the remaining half of b to replace empirical transitions by true transitions. Finally, we recall our choices of constants within the bonus:

$$
\alpha : = 6 4 , \qquad \beta : = 4 0 9 6 .
$$

## A.2 Regret Decomposition

Define

$$
\mathcal { T } ^ { + } : = \sum _ { k , h } \left( \big ( P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } ^ { k } - P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } \big ) V _ { h + 1 } ^ { k } + b _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) \right) _ { + } , \qquad \mathcal { T } ^ { - } : = \sum _ { k } \left( V _ { 1 } ^ { \star } ( s _ { 1 } ^ { k } ) - V _ { 1 } ^ { k } ( s _ { 1 } ^ { k } ) \right) _ { + } .
$$

With the following lemma, we split the regret into the sum of these two terms. We defer the proof to Section B.1.

Lemma A.1 (Regret Decomposition, Eq. (5)). We have

$$
\mathbb { E } \left[ \mathrm { R e g } ( K ) \right] \leq \mathbb { E } [ \mathcal { T } ^ { - } ] + \mathbb { E } [ \mathcal { T } ^ { + } ] .
$$

As discussed in Section 7, we view $\mathcal { T } ^ { - }$ as the cost of under-optimism and ${ \mathcal { T } } ^ { + }$ as the cost of over-optimism.

## A.3 High Probability Good Event

Recall that $L = \log \left( 1 0 H S ^ { 2 } A K ^ { 4 } \right)$ . We define the high probability good event

$$
\mathcal { E } : = \left\{ \left| \hat { P } _ { h , s , a , n } ( s ^ { \prime } ) - P _ { h , s , a } ( s ^ { \prime } ) \right| \leq \sqrt { \frac { 2 P _ { h , s , a } ( s ^ { \prime } ) L } { n } } + \frac { L } { n } \right\} .\tag{A.2}
$$

The following lemma states that E holds with high probability and provides three useful inequalities which hold under E. The proof follows from Bernstein’s inequality and a union bound over all $( h , s , a , s ^ { \prime } , n )$ . We defer the complete proof to Appendix B.2.

Lemma A.2. The event E defined in Eq. (A.2) holds with probability at least $1 - 1 / K ^ { 3 }$ . Moreover, under E, the following hold simultaneously for every $( h , s , a ) \in [ H ] \times S \times A$ , every $n \in [ K ]$ , and every (potentially data-dependent) $V \in [ 0 , H ] ^ { S }$ :

1. Va $\begin{array} { r } { \cdot ( P _ { h , s , a } , V ) \leq 2 \operatorname { V a r } \bigl ( \hat { P } _ { h , s , a , n } , V \bigr ) + \frac { 8 H ^ { 2 } S L } { n } , } \end{array}$

2. Var P<sup>ˆ</sup><sub>h,s,a,n</sub>, V  ≤ 2 Var(P<sub>h,s,a</sub>, V ) + <sup>8H2SL</sup><sub>n</sub> ,

$$
\begin{array} { r } { \hat { P } _ { h , s , a , n } V \le \left( 1 + \frac { 1 } { 1 6 H } \right) P _ { h , s , a } V + \frac { 9 H ^ { 2 } S L } { n } . } \end{array}
$$

## A.4 Bounding T <sup>−</sup>: Cost of Under-Optimism

In this section, we bound the expectation of the cost of under-optimism, that is $\mathbb { E } [ \mathcal { T } ^ { - } ]$

Lemma A.3 (Expected amount of under-optimism). The expected cost of under-optimism is bounded by

$$
\mathbb { E } [ \mathcal { T } ^ { - } ] \leq O \Big ( \sqrt { H ^ { 3 } S A K } \Big ) \Big ) \ .\tag{A.3}
$$

## A.4.1 Stability Properties of Our Bonus Function

We start by providing various stability results on the bonus function. The goal of the present section is to prove Lemma 7.1, restated in the form of Corollary A.6. The following lemma controls how much the bonus changes when the value function increases. We defer the proof to Section B.3.

Lemma A.4 (Stability of the Bonus Function). Let $p \in \Delta ( \mathcal { S } )$ and $n \geq 1$ be arbitrary. For any $0 \leq V \leq$ $V ^ { \prime } \leq H \mathbf { 1 }$

$$
\left| b ( p , n , V ^ { \prime } ) - b ( p , n , V ) \right| \leq \frac { 1 } { 1 6 H } p ( V ^ { \prime } - V ) .
$$

As an immediate consequence, the corresponding Bellman update has the following monotonicity property.

Corollary A.5 (Monotonicity of the Bellman Update). Let $p \in \Delta ( \mathcal { S } )$ be a probability distribution and $n \geq 1$ be arbitrary. For any vectors $0 \leq V \leq V ^ { \prime } \leq H \mathbf { 1 }$ , and any $c \in ( 0 , 1 6 )$ , we have

$$
p V ^ { \prime } + c \cdot b ( p , n , V ^ { \prime } ) \geq p V + c \cdot b ( p , n , V ) .
$$

Proof. For $0 \leq V \leq V ^ { \prime } \leq H \mathbf { 1 }$ and $c \in ( 0 , 1 6 )$ ,

$$
\begin{array} { r l } & { p V + c \cdot b ( p , n , V ) = p V ^ { \prime } + c \cdot b ( p , n , V ^ { \prime } ) + p ( V - V ^ { \prime } ) + c \cdot b ( p , n , V ) - c \cdot b ( p , n , V ^ { \prime } ) } \\ & { \overset { \circ _ { 1 } } { \leq } p V ^ { \prime } + c \cdot b ( p , n , V ^ { \prime } ) + \Bigl ( 1 - \displaystyle \frac { c } { 1 6 H } \Bigr ) p ( V - V ^ { \prime } ) } \\ & { \overset { \circ _ { 2 } } { \leq } p V ^ { \prime } + c \cdot b ( p , n , V ^ { \prime } ) } \end{array}
$$

where (°1) follows by Lemma $\mathrm { A . 4 } ;$ and (°2) uses that $V - V ^ { \prime } \leq 0$ and that $1 - c / ( 1 6 H ) \geq 0 .$

Of course, we need something that holds when V and $V ^ { \prime }$ are not necessarily ordered. It is not hard to generalize the previous lemma to this case by considering V and $V + ( V ^ { \prime } - V ) _ { + }$ , see below.

Corollary A.6 (Stability of the Bellman Update, Complete Version of Lemma 7.1). Let $p \in \Delta ( \mathcal { S } )$ and $n \geq 1$ be arbitrary. For any $V , V ^ { \prime } \in [ 0 , H ] ^ { s }$ , and any $c \in ( 0 , 1 6 )$ , we have

$$
p V + c \cdot b ( p , n , V ) \geq p V ^ { \prime } + c \cdot b ( p , n , V ^ { \prime } ) - \left( 1 + \frac { c } { 1 6 H } \right) p ( V ^ { \prime } - V ) _ { + } .
$$

Proof. The idea is to apply previous results with V and $V + ( V ^ { \prime } - V ) _ { + }$ . We have

$$
\begin{array} { l } { p V + c \cdot b ( p , n , V ) \overset { \circ 1 } { \geq } p V + c \cdot b ( p , n , V + ( V ^ { \prime } - V ) _ { + } ) - \displaystyle \frac { c } { 1 6 H } p ( V ^ { \prime } - V ) _ { + } } \\ { = p \left( V + ( V ^ { \prime } - V ) _ { + } \right) + c \cdot b ( p , n , V + ( V ^ { \prime } - V ) _ { + } ) - \left( 1 + \displaystyle \frac { c } { 1 6 H } \right) p ( V ^ { \prime } - V ) _ { + } } \\ { \overset { \circ _ { 3 } } { \geq } p V ^ { \prime } + c \cdot b ( p , n , V ^ { \prime } ) - \left( 1 + \displaystyle \frac { c } { 1 6 H } \right) p ( V ^ { \prime } - V ) _ { + } } \end{array}
$$

where (°1) follows by Lemma $\mathrm { A . 4 } ;$ and (°3) holds by $V + ( V ^ { \prime } - V ) _ { + } \geq V ^ { \prime }$ and Corollary A.5. □

## A.4.2 Proof of Lemma A.3

We now move to the proof of Lemma A.3. Let E be the good event of Lemma A.2.

Recall that $\bar { b } = b / 2$ . Writing $c _ { H } = 1 + 1 / ( 3 2 H )$ and using $b \geq b _ { 2 }$ , Corollary A.6 with $c = 1 / 2$ gives

$$
\begin{array} { r l } { V _ { h } ^ { \star } ( s ) - V _ { h } ^ { k } ( s ) \leq P _ { h , s , \pi _ { h } ^ { \star } ( s ) } V _ { h + 1 } ^ { \star } - P _ { h , s , \pi _ { h } ^ { \star } ( s ) } ^ { k } V _ { h + 1 } ^ { k } - b \left( P _ { h , s , \pi _ { h } ^ { \star } ( s ) } ^ { k } , n _ { h } ^ { k } ( s , \pi _ { h } ^ { \star } ( s ) ) , V _ { h + 1 } ^ { k } \right) } & { } \\ { \leq P _ { h , s , \pi _ { h } ^ { \star } ( s ) } V _ { h + 1 } ^ { \star } - P _ { h , s , \pi _ { h } ^ { \star } ( s ) } ^ { k } V _ { h + 1 } ^ { k } - \overline { { b } } \left( P _ { h , s , \pi _ { h } ^ { \star } ( s ) } ^ { k } , n _ { h } ^ { k } ( s , \pi _ { h } ^ { \star } ( s ) ) , V _ { h + 1 } ^ { k } \right) - \frac { \beta H ^ { 2 } S L } { 2 n _ { h } ^ { k } ( s , \pi _ { h } ^ { \star } ( s ) ) } } & { } \\ { \leq P _ { h , s , \pi _ { h } ^ { \star } ( s ) } V _ { h + 1 } ^ { \star } - P _ { h , s , \pi _ { h } ^ { \star } ( s ) } ^ { k } V _ { h + 1 } ^ { \star } - \overline { { b } } \left( P _ { h , s , \pi _ { h } ^ { \star } ( s ) } ^ { k } , n _ { h } ^ { k } ( s , \pi _ { h } ^ { \star } ( s ) ) , V _ { h + 1 } ^ { \star } \right) } & { } \\ { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } & { } \\  \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \end{array}
$$

At this point the next-step error is weighted by the empirical transition $P _ { h , s , \pi _ { h } ^ { \star } ( s ) } ^ { k }$ . Unrolling the induction on h now would produce data-dependent visitation probabilities. Instead, we use the correction term split

of from the bonus to replace $P _ { h , s , \pi _ { h } ^ { \star } ( s ) } ^ { k }$ by the true transition $P _ { h , s , \pi _ { h } ^ { \star } ( s ) }$ , and only then we proceed with the induction. Assume that $N _ { h } ^ { k } ( s , \pi _ { h } ^ { \star } ( s ) ) \ge 1$ . Then, on $\mathcal { E } .$ , we have

$$
\begin{array} { r l } & { \mathrm { T e r m ~ ( I ) } \overset { \circ _ { 1 } } { \leq } c _ { H } \left( 1 + \displaystyle \frac { 1 } { 1 6 H } \right) P _ { h , s , \pi _ { h } ^ { \star } ( s ) } ( V _ { h + 1 } ^ { \star } - V _ { h + 1 } ^ { k } ) _ { + } + \left( 9 c _ { H } - \displaystyle \frac { \beta } { 2 } \right) \displaystyle \frac { H ^ { 2 } S L } { n _ { h } ^ { k } ( s , \pi _ { h } ^ { \star } ( s ) ) } } \\ & { \overset { \circ _ { 2 } } { \leq } \left( 1 + \displaystyle \frac { 1 } { 4 H } \right) P _ { h , s , \pi _ { h } ^ { \star } ( s ) } ( V _ { h + 1 } ^ { \star } - V _ { h + 1 } ^ { k } ) _ { + } } \end{array}
$$

where $( ^ { \circ } 1 )$ follows by Part 3 of Lemma $\mathrm { A . 2 } ;$ and (°2) holds since $9 c _ { H } \leq { \textstyle { \frac { 1 } { 2 } } } \beta$ by choice of $\beta .$ . In the case that ${ \cal N } _ { h } ^ { k } ( s , \pi _ { h } ^ { \star } ( s ) ) = 0 .$ , then $( V _ { h + 1 } ^ { \star } - V _ { h + 1 } ^ { k } ) _ { + } = 0$ and the above inequality holds trivially. Combining with the previous display inequality and taking the positive parts of both sides, we obtain

$$
\begin{array} { l } { \displaystyle \left( V _ { h } ^ { \star } ( s ) - V _ { h } ^ { k } ( s ) \right) _ { + } \leq \left( ( P _ { h , s , \pi _ { h } ^ { \star } ( s ) } - P _ { h , s , \pi _ { h } ^ { \star } ( s ) } ^ { k } ) V _ { h + 1 } ^ { \star } - \bar { b } \left( P _ { h , s , \pi _ { h } ^ { \star } ( s ) } ^ { k } , n _ { h } ^ { k } ( s , \pi _ { h } ^ { \star } ( s ) ) , V _ { h + 1 } ^ { k } \right) \right) _ { + } } \\ { \displaystyle \qquad + \left( 1 + \frac { 1 } { 4 H } \right) P _ { h , s , \pi _ { h } ^ { \star } ( s ) } ( V _ { h + 1 } ^ { \star } - V _ { h + 1 } ^ { k } ) _ { + } . } \end{array}
$$

Observing the recursive nature of this inequality, we can iterate backwards and use the fact that $\begin{array} { r } { ( 1 + \frac { 1 } { 4 H } ) ^ { H } \leq 2 } \end{array}$ to obtain that, on $\mathcal { E } ,$ for every fixed initial state $s _ { 1 }$

$$
\big ( V _ { 1 } ^ { \star } ( s _ { 1 } ) - V _ { 1 } ^ { k } ( s _ { 1 } ) \big ) _ { + } \leq 2 \sum _ { h = 1 } ^ { H } \sum _ { s \in \mathcal { S } } \mathbb { F } _ { s _ { 1 } } ^ { \mathbf { r } ^ { \star } } ( s _ { h } = s ) \left( ( P _ { h , s , \pi _ { h } ^ { \star } ( s ) } - P _ { h , s , \pi _ { h } ^ { \star } ( s ) } ^ { k } ) V _ { h + 1 } ^ { \star } - \bar { b } \left( P _ { h , s , \pi _ { h } ^ { \star } ( s ) } ^ { k } , n _ { h } ^ { k } ( s , \pi _ { h } ^ { \star } ( s ) ) , V _ { h + 1 } ^ { \star } \right) \right) _ { + } ,\tag{A.4}
$$

where $\mathbb { P } _ { s _ { 1 } } ^ { \pi ^ { \star } } ( s _ { h } = s )$ denotes the probability of visiting s at step $h$ in the true MDP, starting from $s _ { 1 }$ and following the policy $\pi ^ { \star }$

Since the weights in the above display are deterministic for a fixed $s _ { 1 }$ , it sufices to bound the expected shortfall for each fixed $( k , h , s , a )$ instead of requiring all shortfalls to vanish simultaneously. That is, no union bound is necessary.

The following lemma gives us the desired bound on the expected shortfall for a fixed $( k , h , s , a )$ . The proof, which we give in Section B.4, follows from the time-uniform concentration inequality Theorem 4.1 by integrating its tail bound.

Lemma A.7 (Expected Shortfall). For every $k \in [ K ] , h \in [ H ] , s \in \mathcal { S }$ , and $a \in { \mathcal { A } }$

$$
\mathbb { E } \left[ \mathbb { I } ( \mathcal { E } ) \left( ( P _ { h , s , a } - P _ { h , s , a } ^ { k } ) V _ { h + 1 } ^ { \star } - \bar { b } ( P _ { h , s , a } ^ { k } , n _ { h } ^ { k } ( s , a ) , V _ { h + 1 } ^ { \star } ) \right) _ { + } \right] \lesssim \frac { \mathrm { V a r } ^ { \ge 1 } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } ) } { \sqrt { B } } .
$$

Now, fix k and a deterministic initial state $s _ { 1 }$ . Applying Lemma A.7 to (A.4) yields

$$
\mathbb { E } \big [ \mathbb { I } ( \mathcal { E } ) ( V _ { 1 } ^ { \star } ( s _ { 1 } ) - V _ { 1 } ^ { k } ( s _ { 1 } ) ) _ { + } \big ] \lesssim \frac { 1 } { \sqrt { B } } \sum _ { h = 1 } ^ { H } \sum _ { s \in \mathcal { S } } \mathbb { P } _ { s _ { 1 } } ^ { \pi ^ { \star } } ( s _ { h } = s ) \mathrm { V a r } ^ { \ge 1 } \big ( P _ { h , s , \pi _ { h } ^ { \star } ( s ) } , V _ { h + 1 } ^ { \star } \big ) .
$$

Then using Lemma $\mathrm { A . 2 }$ to bound $\mathbb { P } ( \mathcal { E } ^ { c } ) \le 1 / K ^ { 3 }$ ，

$$
\mathbb { E } \big [ ( V _ { 1 } ^ { \star } ( s _ { 1 } ) - V _ { 1 } ^ { k } ( s _ { 1 } ) ) _ { + } \big ] \lesssim \frac { 1 } { \sqrt { B } } \sum _ { h = 1 } ^ { H } \sum _ { s \in S } \mathbb { P } _ { s _ { 1 } } ^ { \pi ^ { \star } } ( s _ { h } = s ) \mathrm { V a r } ^ { \ge 1 } \big ( P _ { h , s , \pi _ { h } ^ { \star } ( s ) } , V _ { h + 1 } ^ { \star } \big ) + \frac { H } { K ^ { 3 } } .
$$

This is for a fixed state $s _ { 1 }$ instead of the observed $s _ { 1 } ^ { k }$ which is randomly drawn from $\mu .$ . Let $\mathbb { P } _ { \mu } ^ { \pi ^ { \star } } ( s _ { h } = s )$ denote the probability of visiting s at step h in the true MDP, starting from a state randomly drawn from $\mu$ then following $\pi ^ { \star }$ . Denote by $\mathbb { E } _ { \mu } ^ { \pi ^ { \star } } [ - ]$ the associated expectation operator. Because $V _ { 1 } ^ { k }$ is computed from the history

before $s _ { 1 } ^ { k } \sim \mu$ is drawn, it is independent of $s _ { 1 } ^ { k } .$ . Subsequently, since $\begin{array} { r } { \mathbb { P } _ { \mu } ^ { \pi ^ { \star } } ( s _ { h } = s ) = \sum _ { s _ { 1 } } \mu ( s _ { 1 } ) \mathbb { P } _ { s _ { 1 } } ^ { \pi ^ { \star } } ( s _ { h } = s ) } \end{array}$

$$
\begin{array} { r l } & { \mathbb { E } \big [ ( V _ { 1 } ^ { \star } ( s _ { 1 } ^ { k } ) - V _ { 1 } ^ { k } ( s _ { 1 } ^ { k } ) ) _ { + } \big ] = \displaystyle \sum _ { s _ { 1 } \in S } \mu ( s _ { 1 } ) \mathbb { E } \left[ ( V _ { 1 } ^ { \star } ( s _ { 1 } ) - V _ { 1 } ^ { k } ( s _ { 1 } ) ) _ { + } \right] } \\ & { \qquad \lesssim \displaystyle \frac { 1 } { \sqrt { B } } \displaystyle \sum _ { h = 1 } ^ { H } \displaystyle \sum _ { s \in S } \mathbb { P } _ { \mu } ^ { \pi ^ { \star } } ( s _ { h } = s ) \mathrm { V a r } ^ { \ge 1 } \big ( P _ { h , s , \pi _ { h } ^ { \star } ( s ) } , V _ { h + 1 } ^ { \star } \big ) + \displaystyle \frac { H } { K ^ { 3 } } . } \end{array}\tag{A.5}
$$

To bound the weighted sum of variances, observe that, along an episode generated by $\pi ^ { \star }$ , the optimal Bellman equation and telescoping give

$$
\begin{array} { r l } & { \displaystyle \sum _ { h = 1 } ^ { H } \sum _ { s \in \mathcal { S } } \mathbb { P } _ { \mu } ^ { \pi ^ { * } } ( s _ { h } = s ) \mathrm { Y a r } ( P _ { h , s , \pi _ { h } ^ { * } ( s _ { h } ) } , V _ { h + 1 } ^ { * } ) } \\ & { \displaystyle = { \mathbb E } _ { \mu } ^ { \pi ^ { * } } \sum _ { h = 1 } ^ { H } \Big [ P _ { h , s , \pi _ { h } ^ { * } ( s _ { h } ) } ( V _ { h + 1 } ^ { * } ) ^ { 2 } - \big ( V _ { h } ^ { * } ( s _ { h } ) - r _ { h } ( s _ { h } , \pi _ { h } ^ { * } ( s _ { h } ) ) \big ) ^ { 2 } \Big ] } \\ & { \displaystyle = - \mathbb { R } _ { s _ { 1 } \sim \mu } \left[ ( V _ { 1 } ^ { * } ( s _ { 1 } ) ) ^ { 2 } \right] + \mathbb { E } _ { \mu } ^ { \pi ^ { * } } \left[ \sum _ { h = 1 } ^ { H } \big [ 2 r _ { h } \big ( s _ { h } , \pi _ { h } ^ { * } ( s _ { h } ) \big ) V _ { h } ^ { * } ( s _ { h } ) - r _ { h } \big ( s _ { h } , \pi _ { h } ^ { * } ( s _ { h } ) \big ) ^ { 2 } \right] } \\ & { \displaystyle \leq 2 H \mathbb { E } _ { \mu } ^ { \pi ^ { * } } \left[ \sum _ { h = 1 } ^ { H } r _ { h } ( s _ { h } , \pi _ { h } ^ { * } ( s _ { h } ) ) \right] } \\ & { \displaystyle \leq 2 H ^ { 2 } . } \end{array}
$$

Since $\operatorname { V a r } ^ { \geq 1 } ( p , V ) \leq \operatorname { V a r } ( p , V ) + 1$ and $\begin{array} { r } { \sum _ { s } \mathbb { P } _ { \mu } ^ { \pi ^ { \star } } ( s _ { h } = s ) = 1 } \end{array}$ , it follows that

$$
\sum _ { h = 1 } ^ { H } \sum _ { s \in S } \mathbb { P } _ { \mu } ^ { \pi ^ { \star } } ( s _ { h } = s ) \mathrm { V a r } ^ { \geq 1 } ( P _ { h , s , \pi _ { h } ^ { \star } ( s ) } , V _ { h + 1 } ^ { \star } ) \leq 2 H ^ { 2 } + H \leq 3 H ^ { 2 } .
$$

Substituting back into (A.5) and plugging in $B = H K / ( S A )$ gives us that

$$
\mathbb { E } \left[ ( V _ { 1 } ^ { \star } ( s _ { 1 } ^ { k } ) - V _ { 1 } ^ { k } ( s _ { 1 } ^ { k } ) ) _ { + } \right] \lesssim \frac { H ^ { 2 } } { \sqrt { B } } + \frac { H } { K ^ { 3 } } \lesssim \sqrt { H ^ { 3 } S A / K } .
$$

By summing over $k ,$ we have completed the proof of Lemma A.3.

## A.5 Bounding $\mathcal { T } ^ { + }$ : Expected Amount of Over-optimism

In this section, we prove the following result.

Lemma A.8 (Expected amount of over-optimism). The amount of over-optimism is bounded $b y$

$$
\mathbb { E } [ \mathcal { T } ^ { + } ] = O \left( \sqrt { H ^ { 3 } S A K } + H ^ { 3 } S ^ { 2 } A L ^ { 2 } \right) .\tag{A.6}
$$

To prove Lemma A.8, we further decompose $\mathcal { T } ^ { + }$ into three terms:

$$
\mathcal { T } ^ { + } \leq \underbracket { \sum _ { k , h } \bigg ( ( P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } ^ { k } - P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } ) V _ { h + 1 } ^ { k } \bigg ) _ { + } } _ { = : \mathcal { I } _ { 1 } ^ { + } } + \underbracket { \sum _ { k , h } \bigg | ( P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } ^ { k } - P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } ) ( V _ { h + 1 } ^ { k } - V _ { h + 1 } ^ { \star } ) \bigg | } _ { = : \mathcal { I } _ { 2 } ^ { + } } + \underbracket { \sum _ { k , h } b _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) } _ { = : \mathcal { I } _ { 3 } ^ { + } } .
$$

The term $\mathcal { T } _ { 1 } ^ { + }$ measures transition estimation error applied to the optimal value function, while $\mathcal { T } _ { 2 } ^ { + }$ measures transition error applied to the diference $V _ { h + 1 } ^ { k } - V _ { h + 1 } ^ { \star }$ . The remaining term ${ \mathcal { T } } _ { 3 } ^ { + }$ is the sum of bonuses over

the learner’s trajectory. We will additionally make use of the following two variance sums:

$$
\mathrm { V a r } ^ { \mathrm { d i f f } } : = \sum _ { k , h } \mathrm { V a r } \left( P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } , V _ { h + 1 } ^ { k } - V _ { h + 1 } ^ { \star } \right) , \qquad \mathrm { V a r } ^ { \star } : = \sum _ { k , h } \mathrm { V a r } ^ { \geq 1 } \left( P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } , V _ { h + 1 } ^ { \star } \right) .
$$

## A.5.1 A Few Useful Technical Results

The following lemma, which we prove in Appendix B.5, gives a bound on $\mathbb { E } [ \mathrm { V a r } ^ { \star } ]$ . We deal with ${ \mathrm { V a r } } ^ { \mathrm { d i f f } }$ later. Lemma A.9 (Sum of variances). For the total visit counts $N _ { h } ( s , a )$ defined above, we have

$$
\mathbb { E } \left[ \sum _ { h , s , a } \mathrm { V a r } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } ) N _ { h } ( s , a ) \right] \leq 4 H ^ { 2 } K .
$$

Consequently, the sum of floored variances satisfies

$$
\mathbb { E } [ \mathrm { V a r } ^ { \star } ] \leq 5 H ^ { 2 } K .
$$

We also frequently use the following deterministic sum. The proof is given in Appendix B.6.

Lemma A.10. $\begin{array} { r } { \sum _ { k , h } \frac { 1 } { n _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) } \le H S A L . } \end{array}$

## A.5.2 Analysis of $\mathcal { T } _ { 1 } ^ { + }$

We can now proceed to analyzing the first term $\mathcal { T } _ { 1 } ^ { + }$ . We show the following bound.

Lemma A.11 (Bound on $\mathcal { T } _ { 1 } ^ { + } )$ . We have $\mathbb { E } [ \mathcal { T } _ { 1 } ^ { + } ] \leq O \left( \sqrt { H ^ { 3 } S A K } + H ^ { 2 } S A \right)$

To do so, we prove a general concentration inequality – which may be of independent interest – involving sums of deviations. It is related to, but diferent than the time-uniform Bernstein-type concentration result, which considers a maximum over deviations. The complete proof is in Appendix B.7.

Proposition A.12 (Proposition 7.3, Concentration on sums with slackness). Let $X _ { 1 } , X _ { 2 } , . . .$ . be $i . i . d .$ random variables such that

$$
\mathbb { E } [ X _ { i } ] = \mu , \qquad \operatorname { V a r } ( X _ { i } ) = \sigma ^ { 2 } , \qquad | X _ { i } - \mu | \leq c \quad a . s .
$$

For $n \geq 1$ , write $\begin{array} { r } { \hat { \mu } _ { n } : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } X _ { i } } \end{array}$ . There is a universal constant $C > 0$ such that, for every slackness $\Delta > 0$

$$
\sum _ { n = 1 } ^ { \infty } \mathbb { E } \left[ ( \hat { \mu } _ { n } - \mu - \Delta ) _ { + } \right] \leq C \left( \frac { \sigma ^ { 2 } } { \Delta } + c \right) .
$$

By tuning the slackness $\Delta .$ , we obtain the following useful corollary.

Corollary A.13 (Corollary 7.4, Concentration on sums). Under the setting of Proposition A.12, there is a universal constant $C ^ { \prime } > 0$ such that, for any nonnegative integer-valued random variable N with $\mathbb { E } [ N ] < \infty$

$$
\mathbb { E } \left[ \sum _ { n = 1 } ^ { N } \left( \hat { \mu } _ { n } - \mu \right) _ { + } \right] \leq C ^ { \prime } \left( \sqrt { \sigma ^ { 2 } \mathbb { E } [ N ] } + c \right) .
$$

Proof. The cases $\mathbb { E } [ N ] = 0$ and $\sigma ^ { 2 } = 0$ are trivial, so assume $\mathbb { E } [ N ] > 0$ and $\sigma ^ { 2 } > 0$ . By Proposition A.12, we have for any $\Delta > 0$

$$
\mathbb { E } \left[ \sum _ { n = 1 } ^ { N } \left( \hat { \mu } _ { n } - \mu \right) _ { + } \right] \leq \Delta \mathbb { E } [ N ] + C \left( \frac { \sigma ^ { 2 } } { \Delta } + c \right) .
$$

Then taking $\Delta = \sqrt { \sigma ^ { 2 } / \mathbb { E } [ N ] }$ and $C ^ { \prime } = C + 1$ yields the desired result.

Proof of Lemma A.11. For each fixed $( h , s , a )$ , apply Corollary A.13 to the i.i.d. random variables $V _ { h + 1 } ^ { \star } ( X _ { h , s , a , i } )$ whose mean is $P _ { h , s , a } V _ { h + 1 } ^ { \star }$ , variance is Var $( P _ { h , s , a } , V _ { h + 1 } ^ { \star } )$ , and centered absolute value is at most H. Since $N _ { h } ( s , a ) \leq K$ , the corollary gives

$$
\mathbb { E } \left[ \sum _ { n = 1 } ^ { N _ { h } ( s , a ) } \left( ( \hat { P } _ { h , s , a , n } - P _ { h , s , a } ) V _ { h + 1 } ^ { \star } \right) _ { + } \right] \lesssim \sqrt { \mathrm { V a r } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } ) \mathbb { E } [ N _ { h } ( s , a ) ] } + H .
$$

It follows that

$$
\begin{array} { l } { \displaystyle \mathbb { E } [ \mathcal { P } _ { 1 } ^ { + } ] = \mathbb { E } \left[ \sum _ { k , h } \left( ( P _ { h , s _ { k } ^ { k } , a _ { k } ^ { k } } ^ { k } - P _ { h , s _ { k } ^ { k } , a _ { k } ^ { k } } ) V _ { h + 1 } ^ { \star } \right) _ { + } \right] } \\ { \displaystyle \stackrel { \because } { \leq } H ^ { 2 } S A + \sum _ { h , s _ { 0 } = 4 } { \mathbb { E } \left[ \sum _ { n = 1 } ^ { N } \left( \widehat { P } _ { h , s , a , n } V _ { h + 1 } ^ { \star } - P _ { h , s , a } V _ { h + 1 } ^ { \star } \right) _ { + } \right] } } \\ { \displaystyle \lesssim H ^ { 2 } S A + \sum _ { h , s _ { 0 } = 4 } { \sqrt { \mathrm { V a r } ( P _ { h , s , a } V _ { h + 1 } ^ { \star } ) \mathbb { E } [ N _ { h } ( s , a ) ] } } } \\ { \displaystyle \stackrel { \because } { \leq } H ^ { 2 } S A + \sqrt { \mu S A \sum _ { h , s , a } \mathrm { V a r } ( P _ { h , s , a } V _ { h + 1 } ^ { \star } ) \mathbb { E } [ N _ { h } ( s , a ) ] } } \\ { \displaystyle \stackrel { \leq } { \leq } H ^ { 2 } S A + \sqrt { H ^ { 3 } S A \sum _ { h , s , a } \mathrm { V a r } ( P _ { h , s , a } V _ { h + 1 } ^ { \star } ) \mathbb { E } [ N _ { h } ( s , a ) ] } } \\ { \displaystyle \stackrel { \leq } { \leq } H ^ { 2 } S A + \sqrt { H ^ { 3 } S A K } . } \end{array}
$$

Here, (°1) accounts for the time steps where the count is zero; (°2) is by Cauchy-Schwarz; and (°3) is by Lemma A.9. □

## A.5.3 Analysis of $\mathcal { T } _ { 2 } ^ { + }$

Next, we bound the second term $\mathcal { T } _ { 2 } ^ { + }$ . The analysis of this term is relatively standard.

Lemma A.14 (Bound on $\mathcal { T } _ { 2 } ^ { + } )$ . We have $\mathbb { E } [ \mathcal { T } _ { 2 } ^ { + } ] \leq O \left( \sqrt { H S ^ { 2 } A L ^ { 2 } \mathbb { E } [ \mathrm { V a r } ^ { \mathrm { d i f f } } ] } + H ^ { 2 } S ^ { 2 } A L ^ { 2 } \right)$

Proof of Lemma $A . 1 4$ . Let E be the good event of Lemma A.2. Under E, for any $h \in [ H ] , s \in { \mathcal { S } } , a \in { \mathcal { A } } , n \in [ K ]$ and (possibly data-dependent) $V \in [ - H , H ] ^ { s }$ , we have

$$
\begin{array} { r l } { \left| \left( \hat { P } _ { h , s , a , n } - P _ { h , s , a } \right) V \right| \overset { \circ _ { 1 } } { = } \left| \left( \hat { P } _ { h , s , a , n } - P _ { h , s , a } \right) ( V - P _ { h , s , a } V ) \right| } & { } \\ { \overset { \circ _ { 2 } } { \leq } \displaystyle \sum _ { s ^ { \prime } } \left| \hat { P } _ { h , s , a , n } ( s ^ { \prime } ) - P _ { h , s , a } ( s ^ { \prime } ) \right| \left| V ( s ^ { \prime } ) - P _ { h , s , a } V \right| } & { } \\ { \overset { \circ _ { 3 } } { \leq } \displaystyle \sum _ { s ^ { \prime } } \sqrt { \frac { 2 P _ { h , s , a } ( s ^ { \prime } ) L } { n } } \displaystyle \vert V ( s ^ { \prime } ) - P _ { h , s , a } V \vert + \frac { L } { n } \displaystyle \sum _ { s ^ { \prime } } \left. V ( s ^ { \prime } ) - P _ { h , s , a } V \right. } & { } \\ { \overset { \circ _ { 4 } } { \leq } \sqrt { \frac { 2 S L V \mathrm { a r } ( P _ { h , s , a } , V ) } { n } } + \frac { 2 H S L } { n } . } & { } \end{array}
$$

(°1) holds because $( \hat { P } _ { h , s , a , n } - P _ { h , s , a } ) c = 0$ for any real number $c ; ( ^ { \circ } 2 )$ is by the triangle inequality; (°3) uses $\mathcal { E } ;$ and (°4) follows from Cauchy-Schwarz and $| V ( s ^ { \prime } ) - P _ { h , s , a } V | \le 2 H$ . Hence, we have on $\mathcal { E }$

$$
\begin{array} { l } { \displaystyle \mathcal { T } _ { 2 } ^ { + } = \sum _ { k , h } \Big | ( P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } ^ { k } - P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } ) ( V _ { h + 1 } ^ { k } - V _ { h + 1 } ^ { \star } ) \Big | } \\ { \displaystyle \qquad \leq 2 H ^ { 2 } S A + \sum _ { k , h \colon N _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) \geq 1 } \Big | ( P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } ^ { k } - P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } ) ( V _ { h + 1 } ^ { k } - V _ { h + 1 } ^ { \star } ) \Big | } \end{array}
$$

$$
\begin{array} { r l } & { = 2 H ^ { 2 } S A + \displaystyle \sum _ { k , h ^ { \prime } , k ^ { \prime } ( s _ { h } ^ { \prime } , a _ { k } ^ { \prime } ) \geq 1 } \left. \big ( \hat { P } _ { h , s _ { h } ^ { \prime } , a _ { k } ^ { \prime } , b _ { k } ^ { \prime } ( s _ { h } ^ { \prime } , a _ { k } ^ { \prime } ) } - P _ { h , s _ { h } ^ { \prime } , a _ { k } ^ { \prime } } \big ) \big ( V _ { h + 1 } ^ { k } - V _ { h + 1 } ^ { \ast } \big ) \right. } \\ & { \leq 2 H ^ { 2 } S A + \displaystyle \sum _ { k , h ^ { \prime } , \chi _ { h } ^ { \prime } ( s _ { h } ^ { \prime } , a _ { k } ^ { \prime } ) \geq 1 } \left( \sqrt { \frac { 2 S L V a r \big ( P _ { h , s _ { h } ^ { \prime } , a _ { k } ^ { \prime } } , V _ { h + 1 } ^ { k } - V _ { h + 1 } ^ { k } - V _ { h + 1 } ^ { k } \big ) } { N _ { k } ^ { \prime } ( s _ { h } ^ { \prime } , a _ { k } ^ { \prime } ) } + \frac { 2 H S L } { N _ { h } ^ { \prime } ( s _ { h } ^ { \prime } , a _ { k } ^ { \prime } ) } } \right) } \\ & { \leq 2 H ^ { 2 } S A + \displaystyle \sum _ { k , h ^ { \prime } } \sqrt { \frac { 2 S L V \operatorname* { m a x } \big ( P _ { h , s _ { h } ^ { \prime } , a _ { k } ^ { \prime } } , V _ { h + 1 } ^ { k } - V _ { h + 1 } ^ { k } \big ) } { \pi _ { h } ^ { 2 } ( s _ { h } ^ { \prime } , a _ { k } ^ { \prime } ) } + \sum _ { k , h } \frac { 2 H S L } { h _ { k } ^ { 2 } ( s _ { h } ^ { \prime } , a _ { k } ^ { \prime } ) } } } \\ &  \overset { \leq } \sum _ { k } 2 H ^ { 2 } S A + \displaystyle \sum _ { k , h ^ { \prime } } 2 S L V a r  \end{array}
$$

where (°5) is due to Cauchy-Schwarz, and (°6) is due to Lemma A.10 (which bounds $\begin{array} { r } { \sum _ { k , h } 1 / n _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) \le } \end{array}$ HSAL) and the definition of $\mathrm { V a r } ^ { \mathrm { d i f f } }$

When E does not hold, we have the trivial bound $\mathcal { T } _ { 2 } ^ { + } \leq 2 H ^ { 2 } K$ . Combining this bound with $\mathbb { P } ( \mathcal { E } ^ { c } ) \le \frac { 1 } { K ^ { 3 } }$ from Lemma $\mathrm { { A . 2 } } .$ , we conclude

$$
\begin{array} { r l } { \mathbb { E } [ \mathcal { T } _ { 2 } ^ { + } ] = \mathbb { E } [ \mathbb { I } ( \mathcal { E } ) \mathcal { T } _ { 2 } ^ { + } ] + \mathbb { E } [ \mathbb { I } ( \mathcal { E } ^ { c } ) \mathcal { T } _ { 2 } ^ { + } ] } \\ { \ } & { \leq \sqrt { 2 H S ^ { 2 } A L ^ { 2 } } \mathbb { E } \left[ \mathbb { I } ( \mathcal { E } ) \sqrt { \mathrm { V a r } ^ { \mathrm { d i f f } } } \right] + 4 H ^ { 2 } S ^ { 2 } A L ^ { 2 } + 2 H ^ { 2 } K \mathbb { P } ( \mathcal { E } ^ { c } ) } \\ { \ } & { \overset { \circ _ { 7 } } { \leq } \sqrt { 2 H S ^ { 2 } A L ^ { 2 } \mathbb { E } \left[ \mathrm { V a r } ^ { \mathrm { d i f f } } \right] } + 4 H ^ { 2 } S ^ { 2 } A L ^ { 2 } + \frac { 2 H ^ { 2 } } { K ^ { 2 } } } \\ { \ } & { \lesssim \sqrt { H S ^ { 2 } A L ^ { 2 } \mathbb { E } \left[ \mathrm { V a r } ^ { \mathrm { d i f f } } \right] } + H ^ { 2 } S ^ { 2 } A L ^ { 2 } . } \end{array}
$$

(°7) uses $\mathbb { E } [ \mathbb { I } ( \mathcal { E } ) \sqrt { \mathrm { V a r } ^ { \mathrm { d i f f } } } ] \le \mathbb { E } [ \sqrt { \mathrm { V a r } ^ { \mathrm { d i f f } } } ] \le \sqrt { \mathbb { E } [ \mathrm { V a r } ^ { \mathrm { d i f f } } ] }$ by Jensen’s inequality.

## A.5.4 Analysis of ${ \mathcal { T } } _ { 3 } ^ { + }$

The bound on the third term ${ \mathcal { T } } _ { 3 } ^ { + }$ requires the following key lemma. We will use it to show that our MOSS-style bonus can be summed without introducing log factors.

Lemma A.15 (Variance-Normalized MOSS Sum). For any fixed $q > 0 , B > 0$ , and integer $N \geq 0$ , we have

$$
\sum _ { n = 1 } ^ { N } { \sqrt { \frac { q \log ^ { \geq 1 } { \frac { B } { n q } } } { n } } } \leq 6 { \sqrt { B } } + { \frac { q N } { \sqrt { B } } } .
$$

Proof. Write $m = \lfloor B / q \rfloor$ . Interpret empty sums below as zero. We upper bound the sum from 1 to N by the sum from 1 to m plus the sum from $m + 1$ to N. Now, since $x \mapsto { \sqrt { \frac { q \log ^ { \geq 1 } { \frac { B } { x q } } } { x } } }$ is nonincreasing on $( 0 , \infty )$ and $m \leq B / q$ , we can bound

$$
\begin{array} { r l r } {  { \sum _ { n = 1 } ^ { m } \sqrt { \frac { q \log ^ { 2 1 } \frac { B } { n q } } { n } } \le \int _ { 0 } ^ { B / q } \sqrt { \frac { q \log ^ { 2 1 } \frac { B } { x q } } { x } } d x = \sqrt { B } \int _ { 0 } ^ { 1 } \sqrt { \frac { \log ^ { 2 1 } \frac { 1 } { u } } { u } } d u } } \\ & { } & { \le \sqrt { B } \int _ { 0 } ^ { 1 } \sqrt { \frac { 1 + \log \frac { 1 } { u } } { u } } d u } \\ & { } & { = \sqrt { B } \int _ { 0 } ^ { \infty } e ^ { - t / 2 } \sqrt { 1 + t } d t \le 6 \sqrt { B } . } \end{array}
$$

Above we used the substitutions $u = x q / B$ and $t = - \log u$ . For every remaining index $m < n \leq N$ , we have $n > B / q$ and hence $\begin{array} { r } { \log ^ { \ge 1 } { \frac { B } { n q } } = 1 } \end{array}$ , so

$$
\sum _ { n = m + 1 } ^ { N } { \sqrt { \frac { q \log ^ { \geq 1 } { n q } } { n } } } = \sum _ { n = m + 1 } ^ { N } { \sqrt { \frac { q } { n } } } \leq N { \sqrt { \frac { q } { B / q } } } = { \frac { q N } { \sqrt { B } } } .
$$

Combining the two sums yields the desired bound.

With this lemma we can obtain the following bound on ${ \mathcal { T } } _ { 3 } ^ { + }$ . The main ideas here are as follows. We use E so that the empirical variances are approximately the same as the true variances up to an additive lower order term. Then, we split terms involving $V _ { h + 1 } ^ { k }$ into terms involving $V ^ { \star }$ and terms involving $\it V _ { h + 1 } ^ { k } - \it V ^ { \star }$ The former we bound by $O ( \sqrt { H ^ { 3 } S A K } + H ^ { 3 } S ^ { 2 } A L ^ { 2 } )$ by direct computation on top of Lemma A.15, and the latter we bound by a term involving ${ \mathrm { V a r } } ^ { \mathrm { d i f f } }$ . For the complete proof, see Appendix B.8.

Lemma A.16. $\mathbb { E } [ \mathcal { T } _ { 3 } ^ { + } ] \leq O \left( \sqrt { H ^ { 3 } S A K } + \sqrt { H S A L ^ { 2 } \mathbb { E } [ \mathrm { V a r } ^ { \mathrm { d i f f } } ] } + H ^ { 3 } S ^ { 2 } A L ^ { 2 } \right) .$

## A.5.5 Conclusion: Bounding $\mathcal { T } ^ { + }$ and Proving Lemma A.8

Combining Lemma A.11 on $\mathcal { T } _ { 1 } ^ { + }$ , Lemma A.14 on $\mathcal { T } _ { 2 } ^ { + }$ , and Lemma A.16 on ${ \mathcal { T } } _ { 3 } ^ { + }$ , we obtain

$$
\begin{array} { r } { \mathbb { E } [ \mathcal { T } ^ { + } ] \leq O \left( \sqrt { H ^ { 3 } S A K } + H ^ { 3 } S ^ { 2 } A L ^ { 2 } + \sqrt { H S ^ { 2 } A L ^ { 2 } \mathbb { E } [ \mathrm { V a r } ^ { \mathrm { d i f f } } ] } \right) . } \end{array}\tag{A.7}
$$

The following lemma, which we prove in Appendix B.9, bounds ${ \mathrm { V a r } } ^ { \mathrm { d i f f } }$

Lemma A.17. We have $\mathbb { E } [ \mathrm { V a r } ^ { \mathrm { d i f f } } ] \leq 4 H \mathbb { E } [ \mathcal { T } ^ { + } ] + 2 H \mathbb { E } [ \mathcal { T } ^ { - } ]$

Combining Lemma A.17 with (A.7) and the bound on $\mathcal { T } ^ { - }$ (Lemma A.3) implies

$$
\mathbb { E } [ \mathrm { V a r } ^ { \mathrm { d i f f } } ] \leq O \left( H \sqrt { H ^ { 3 } S A K } + H ^ { 4 } S ^ { 2 } A L ^ { 2 } + \sqrt { H ^ { 3 } S ^ { 2 } A L ^ { 2 } \mathbb { E } [ \mathrm { V a r } ^ { \mathrm { d i f f } } ] } \right) .
$$

Thus, by $x \leq a + { \sqrt { b x } } \implies x \leq 2 a + b ,$

$$
\mathbb { E } [ \mathrm { V a r } ^ { \mathrm { d i f f } } ] \leq O \left( H \sqrt { H ^ { 3 } S A K } + H ^ { 4 } S ^ { 2 } A L ^ { 2 } \right) .
$$

Then, plugging back into (A.7) gives us

$$
\begin{array} { r l } & { \mathbb { E } [ \mathcal { T } ^ { + } ] \leq O \left( \sqrt { H ^ { 3 } S A K } + H ^ { 3 } S ^ { 2 } A L ^ { 2 } + \sqrt { H ^ { 2 } S ^ { 2 } A L ^ { 2 } \sqrt { H ^ { 3 } S A K } } \right) } \\ & { \qquad \leq O \left( \sqrt { H ^ { 3 } S A K } + H ^ { 3 } S ^ { 2 } A L ^ { 2 } \right) , } \end{array}\tag{A.8}
$$

where the second inequality holds by ${ \sqrt { x y } } \leq x + y$ with $x = H ^ { 2 } S ^ { 2 } A L ^ { 2 }$ and $y = \sqrt { H ^ { 3 } S A K }$

## A.6 Conclusion: Bounding the Expected Regret and Proving Theorem 6.1

We conclude by the regret decomposition (Lemma A.1), the bound on $\mathbb { E } [ \mathcal { T } ^ { - } ]$ (Lemma A.3), and the bound on $\mathbb { E } [ \mathcal { T } ^ { + } ] \ ( \mathrm { E q . \ ( A . 8 ) } )$ that

$$
\mathbb { E } [ \mathrm { R e g } ( K ) ] \le \mathbb { E } [ \mathcal { T } ^ { - } ] + \mathbb { E } [ \mathcal { T } ^ { + } ] \le O \left( \sqrt { H ^ { 3 } S A K } + H ^ { 3 } S ^ { 2 } A L ^ { 2 } \right) .
$$

If $K \le H S A$ , then $\mathbb { E } [ \mathrm { R e g } ( K ) ] \le H K \le \sqrt { H ^ { 3 } S A K }$ . Otherwise, $K > H S A$ implies $K \geq 2$ and $H S ^ { 2 } A \leq$ $( H S A ) ^ { 2 } < K ^ { 2 }$ , so $L = \log ( 1 0 H S ^ { 2 } A K ^ { 4 } ) \leq \log ( 1 0 K ^ { 6 } ) \lesssim$ log K. In either case,

$$
\begin{array} { r } { \mathbb { E } [ \mathrm { R e g } ( K ) ] \leq O \left( \sqrt { H ^ { 3 } S A K } + H ^ { 3 } S ^ { 2 } A ( \log K ) ^ { 2 } \right) . } \end{array}
$$

## B Missing Proofs from Section A

This section contains proofs which were omitted for ease of presentation. We also state Bernstein’s inequality here for completeness.

Lemma B.1 (Bernstein’s Inequality, Theorem 3 in Maurer and Pontil [2009]). Let $( X _ { k } ) _ { k = 1 } ^ { n }$ be a sequence $o f i . i . d .$ random variables. Let $\mu : = \mathbb { E } [ X _ { 1 } ]$ be their expected value and $v : = \mathrm { V a r } ( X _ { 1 } )$ be their variance. If $X _ { 1 } - \mu \leq c \ a . s .$ for some $c > 0$ , then for $a l l x \ge 0$

$$
\mathbb { P } \Bigg ( \frac { 1 } { n } \sum _ { k = 1 } ^ { n } ( X _ { k } - \mu ) \geq \sqrt { \frac { 2 v x } { n } } + \frac { c x } { 3 n } \Bigg ) \leq \mathrm { e } ^ { - x } .
$$

## B.1 Proof of Regret Decomposition, Lemma A.1

We decompose

$$
\begin{array} { r l } & { \log ( K ) = \displaystyle \sum _ { k = 1 } ^ { K } \Big ( V _ { 1 } ^ { \star } ( s _ { 1 } ^ { k } ) - V _ { 1 } ^ { \pi ^ { k } } ( s _ { 1 } ^ { k } ) \Big ) } \\ & { \qquad \le \displaystyle \sum _ { k = 1 } ^ { K } \big ( V _ { 1 } ^ { \star } ( s _ { 1 } ^ { k } ) - V _ { 1 } ^ { k } ( s _ { 1 } ^ { k } ) \big ) _ { + } + \displaystyle \sum _ { k = 1 } ^ { K } \Big ( V _ { 1 } ^ { k } ( s _ { 1 } ^ { k } ) - V _ { 1 } ^ { \pi ^ { k } } ( s _ { 1 } ^ { k } ) \Big ) . } \end{array}
$$

The first sum is precisely $\mathcal { T } ^ { - }$ , so it remains to bound the expectation of the second sum by $\mathbb { E } [ \mathcal { T } ^ { + } ]$ . Towards this end, fix arbitrary $k \in [ K ]$ . We have

$$
\begin{array} { r l } & { \mathbb { E } \left[ V _ { 1 } ^ { k } ( s _ { 1 } ^ { k } ) - V _ { 1 } ^ { \pi ^ { k } } ( s _ { 1 } ^ { k } ) \right] = \mathbb { E } \left[ V _ { 1 } ^ { k } ( s _ { 1 } ^ { k } ) - \displaystyle \sum _ { h = 1 } ^ { H } r _ { h } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) \right] + \mathbb { E } \left[ \displaystyle \sum _ { h = 1 } ^ { H } r _ { h } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) - V _ { 1 } ^ { \pi ^ { k } } ( s _ { 1 } ^ { k } ) \right] } \\ & { \qquad = \mathbb { E } \left[ V _ { 1 } ^ { k } ( s _ { 1 } ^ { k } ) - \displaystyle \sum _ { h = 1 } ^ { H } r _ { h } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) \right] . } \end{array}
$$

Moreover, using $V _ { H + 1 } ^ { k } = 0$ , telescoping gives us

$$
\begin{array} { l l } { { \displaystyle V _ { 1 } ^ { k } ( s _ { 1 } ^ { k } ) - \sum _ { h = 1 } ^ { H } r _ { h } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) } } & { { } } \\ { { } } & { { = \displaystyle \sum _ { h = 1 } ^ { H } \left( V _ { h } ^ { k } ( s _ { h } ^ { k } ) - r _ { h } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) - V _ { h + 1 } ^ { k } ( s _ { h + 1 } ^ { k } ) \right) } } \\ { { } } & { { = \displaystyle \sum _ { h = 1 } ^ { H } \left( P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } V _ { h + 1 } ^ { k } - V _ { h + 1 } ^ { k } ( s _ { h + 1 } ^ { k } ) \right) + \displaystyle \sum _ { h = 1 } ^ { H } \left( V _ { h } ^ { k } ( s _ { h } ^ { k } ) - r _ { h } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) - P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } V _ { h + 1 } ^ { k } \right) . } } \end{array}
$$

The first sum has expectation 0, and for the second sum we use

$$
V _ { h } ^ { k } ( s _ { h } ^ { k } ) = Q _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) \le r _ { h } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) + P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } ^ { k } V _ { h + 1 } ^ { k } + b _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) .
$$

Hence,

$$
\mathbb { E } \left[ V _ { 1 } ^ { k } ( s _ { 1 } ^ { k } ) - \sum _ { h = 1 } ^ { H } r _ { h } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) \right] \leq \mathbb { E } \left[ \sum _ { h = 1 } ^ { H } \left( \left( P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } ^ { k } - P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } \right) V _ { h + 1 } ^ { k } + b _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) \right) _ { + } \right] .
$$

Summing over k completes the argument.

## B.2 Proof of Probability of the Good Event $\mathcal { E } .$ , Lemma A.2

We begin by proving that $\mathbb { P } ( \mathcal { E } ) \geq 1 - 1 / K ^ { 3 }$ . We have

$$
\mathbb { P } \Big ( ( \mathscr { E } ) ^ { \complement } \Big ) \overset { \circ _ { 1 } } { \leq } \sum _ { h , s , a , s ^ { \prime } } \sum _ { n = 1 } ^ { K } \mathbb { P } \Bigg ( \Big | \hat { P } _ { h , s , a , n } ( s ^ { \prime } ) - P _ { h , s , a } ( s ^ { \prime } ) \Big | > \sqrt { \frac { 2 P _ { h , s , a } ( s ^ { \prime } ) L } { n } } + \frac { L } { n } \Bigg )
$$

where (°1) follows by a union bound; (°2) follows by bounding the summand via Bernstein’s inequality (Lemma B.1) and bounding the number of terms by $H S ^ { 2 } A K$ ; and (°3) holds by the definition of L.

We continue with the proof of Assertion (1.). By the AM–GM inequality, $\sqrt { 2 p L / n } \leq p / 2 + L / n$ . So, on $\mathcal { E } ,$ for every $( h , s , a , s ^ { \prime } , n ) \in [ H ] \times S \times A \times S \times [ K ]$ , we have

$$
\hat { P } _ { h , s , a , n } ( s ^ { \prime } ) \ d s \frac { 3 } { 2 } P _ { h , s , a } ( s ^ { \prime } ) + \frac { 2 L } { n } \quad \mathrm { a n d } \quad P _ { h , s , a } ( s ^ { \prime } ) \ d s \ d s 2 \hat { 2 } P _ { h , s , a , n } ( s ^ { \prime } ) + \frac { 4 L } { n } \ .\tag{B.1}
$$

From there, we prove Assertion (1.) as such:

$$
\begin{array} { r l } & { \mathrm { V a r } ( P _ { h , s , a } , V ) = \displaystyle \operatorname* { m i n } _ { x \in \mathbb R } \sum _ { s ^ { \prime } } P _ { h , s , a } ( s ^ { \prime } ) \big ( V ( s ^ { \prime } ) - x \big ) ^ { 2 } } \\ & { \qquad \le \displaystyle \sum _ { s ^ { \prime } } P _ { h , s , a } ( s ^ { \prime } ) \Big ( V ( s ^ { \prime } ) - \hat { P } _ { h , s , a , n } V \Big ) ^ { 2 } } \\ & { \stackrel { \circ _ { 4 } } \le \displaystyle \sum _ { s ^ { \prime } } \bigg ( 2 \hat { P } _ { h , s , a , n } ( s ^ { \prime } ) + \frac { 4 L } { n } \bigg ) \Big ( V ( s ^ { \prime } ) - \hat { P } _ { h , s , a , n } V \Big ) ^ { 2 } } \\ & { \stackrel { \circ _ { 5 } } \le 2 \mathrm { V a r } ( \hat { P } _ { h , s , a , n } , V ) + \frac { 4 S H ^ { 2 } L } { n } } \end{array}
$$

where in (°4) we bound $P _ { h , s , a } ( s ^ { \prime } )$ via Eq. (B.1); and in (°5) we use that $| V ( s ^ { \prime } ) - \hat { P } _ { h , s , a , n } V | \le H$ for all $s ^ { \prime } { \mathrm { . } }$ which holds true since $V \in [ 0 , H ] ^ { S }$

We prove Assertion (2.) similarly, except now we use the first bound in Eq. (B.1):

$$
\begin{array} { l } { \mathrm { V a r } ( \hat { P } _ { h , s , a , n } , V ) \leq \displaystyle \sum _ { s ^ { \prime } } \hat { P } _ { h , s , a , n } ( s ^ { \prime } ) \big ( V ( s ^ { \prime } ) - P _ { h , s , a } V \big ) ^ { 2 } } \\ { \leq \displaystyle \sum _ { s ^ { \prime } } \left( \frac { 3 } { 2 } P _ { h , s , a } ( s ^ { \prime } ) + \frac { 2 L } { n } \right) \big ( V ( s ^ { \prime } ) - P _ { h , s , a } V \big ) ^ { 2 } } \\ { \leq \frac { 3 } { 2 } \mathrm { V a r } ( P _ { h , s , a } , V ) + \displaystyle \frac { 2 H ^ { 2 } S L } { n } } \\ { \leq 2 \mathrm { V a r } ( P _ { h , s , a } , V ) + \frac { 8 H ^ { 2 } S L } { n } . } \end{array}
$$

We are left with the proof of Assertion (3.):

$$
\begin{array} { r l } & { \hat { P } _ { h , s , a , n } V = P _ { h , s , a } V + \displaystyle \sum _ { s ^ { \prime } } \Bigl ( \hat { P } _ { h , s , a , n } ( s ^ { \prime } ) - P _ { h , s , a } ( s ^ { \prime } ) \Bigr ) V ( s ^ { \prime } ) } \\ & { \stackrel { \circ _ { 6 } } { \leq } P _ { h , s , a } V + \displaystyle \sum _ { s ^ { \prime } } \Biggl ( \sqrt { \frac { 2 P _ { h , s , a } ( s ^ { \prime } ) L } { n } } + \frac { L } { n } \Biggr ) V ( s ^ { \prime } ) } \\ & { \stackrel { \circ _ { 7 } } { \leq } P _ { h , s , a } V + \displaystyle \sum _ { s ^ { \prime } } \Biggl ( \frac { P _ { h , s , a } ( s ^ { \prime } ) } { 1 6 H } + \frac { ( 8 H + 1 ) L } { n } \Biggr ) V ( s ^ { \prime } ) } \\ & { \stackrel { \circ _ { 8 } } { \leq } \left( 1 + \frac { 1 } { 1 6 H } \right) P _ { h , s , a } V + \frac { 9 H ^ { 2 } S L } { n } } \end{array}
$$

where (°6) follows by $V \geq 0$ and because we are on $\mathcal { E } ; ( ^ { \circ } 7 )$ follows from the AM–GM inequality $\sqrt { 2 p L / n } \leq$ $p / ( 1 6 H ) + 8 H L / n ;$ and (°8) uses that $V ( s ^ { \prime } ) \leq H$ and $H \geq 1$ □

## B.3 Proof of Ordered Stability, Lemma A.4

Fix arbitrary $p \in \Delta ( \mathcal { S } ) , n \geq 1$ , and $0 \leq V \leq V ^ { \prime } \leq H \mathbf { 1 }$ . We first observe that

$$
\begin{array} { r l } { \widetilde { \mathrm { V a r } } ( p , n , V ^ { \prime } ) - \widetilde { \mathrm { V a r } } ( p , n , V ) \Big | \overset { \circ _ { 1 } } { = } 2 \Big | \mathrm { V a r } ^ { \Sigma ^ { 1 } } ( p , V ^ { \prime } ) - \mathrm { V a r } ^ { \Sigma ^ { 1 } } ( p , V ) \Big | } & { } \\ & { \overset { \circ _ { 2 } } { \leq } 2 | \mathrm { V a r } ( p , V ^ { \prime } ) - \mathrm { V a r } ( p , V ) | } \\ & { \overset { \circ _ { 3 } } { = } 2 | p \big ( ( V ^ { \prime } ) ^ { 2 } - V ^ { 2 } \big ) - \big ( ( p V ^ { \prime } ) ^ { 2 } - ( p V ) ^ { 2 } \big ) \Big | } \\ & { \overset { \circ _ { 4 } } { = } 2 | p \big ( ( V ^ { \prime } - V ) ( V ^ { \prime } + V ) \big ) - p ( V ^ { \prime } - V ) p ( V ^ { \prime } + V ) \big ) } \\ & { \overset { \circ _ { 5 } } { = } 2 | p \big ( ( V ^ { \prime } - V ) ( V ^ { \prime } + V - p ( V ^ { \prime } + V ) \mathbf { 1 } ) \big ) | } \\ & { \overset { \circ _ { 6 } } { \leq } 4 H p ( V ^ { \prime } - V ) . } \end{array}
$$

Here, (°1) unfolds the definition of $\widetilde { \mathrm { V a r } } ;$ (°2) holds because $\lvert \mathrm { m a x } ( u , 1 ) - \mathrm { m a x } ( v , 1 ) \rvert \leq \lvert u - v \rvert$ for $u , v \in \mathbb { R } ;$ and (°3) unfolds the definition of Var. For (°4), the diference-of-squares identity and linearity of $p$ give

$$
( p V ^ { \prime } ) ^ { 2 } - ( p V ) ^ { 2 } = ( p V ^ { \prime } - p V ) ( p V ^ { \prime } + p V ) = p ( V ^ { \prime } - V ) p ( V ^ { \prime } + V ) .
$$

Then (°5) uses that $p ( V ^ { \prime } + V )$ is a scalar and can therefore be moved inside $p .$ Finally, (°6) follows from $V ^ { \prime } - V \geq 0$ and

$$
| V ^ { \prime } + V - p ( V ^ { \prime } + V ) \mathbf { 1 } | \leq 2 H \mathbf { 1 } ~ .
$$

Next, the function $\Phi ( x ) = x \log ^ { \geq 1 } ( B / ( n x ) )$ is continuous on $[ 1 , \infty )$ , and, wherever it is diferentiable,

$$
\frac { \mathrm { d } } { \mathrm { d } x } \bigg ( x \log ^ { \geq 1 } \bigg ( \frac { B } { n x } \bigg ) \bigg ) = \left\{ \begin{array} { l l } { \log \bigg ( \frac { B } { n x } \bigg ) - 1 , } & { x < \frac { B } { e n } , } \\ { 1 , } & { x > \frac { B } { e n } . } \end{array} \right.
$$

Both expressions lie in $\left[ 0 , \log ^ { \geq 1 } ( B / n ) \right]$ on their respective domains within $[ 1 , \infty )$

Now we argue that, for $x , x ^ { \prime } \geq 1$ we have

$$
| \Phi ( x ^ { \prime } ) - \Phi ( x ) | \leq \log ^ { \geq 1 } \biggl ( \frac { B } { n } \biggr ) | x ^ { \prime } - x | .\tag{B.2}
$$

If $x = x ^ { \prime }$ the bound is immediate, so otherwise assume WLOG that $x ^ { \prime } > x$ . Let $x _ { 0 } = B / ( e n )$ . If $x _ { 0 } \not \in ( x , x ^ { \prime } )$

then Φ is diferentiable on the interval $( x , x ^ { \prime } )$ and the mean value theorem immediately implies (B.2). If $x < x _ { 0 } < x ^ { \prime }$ then applying the mean value theorem on each of the intervals $[ x , x _ { 0 } ]$ and $[ x _ { 0 } , x ^ { \prime } ]$ (noting that Φ is continuous at $x _ { 0 } )$ yields that

$$
\begin{array} { r l } & { | \Phi ( x ^ { \prime } ) - \Phi ( x ) | \le | \Phi ( x ^ { \prime } ) - \Phi ( x _ { 0 } ) | + | \Phi ( x _ { 0 } ) - \Phi ( x ) | } \\ & { \qquad \le | x ^ { \prime } - x _ { 0 } | + \log ^ { \ge 1 } ( B / n ) | x _ { 0 } - x | } \\ & { \qquad \le \log ^ { \ge 1 } ( B / n ) | x ^ { \prime } - x | . } \end{array}
$$

Thus we have shown (B.2).

Now, noting $\begin{array} { r } { b _ { 1 } ( p , n , V ) ^ { 2 } = \frac { \alpha } { n } \Phi ( \widetilde { \mathrm { V a r } } ( p , n , V ) ) } \end{array}$ , we can apply (B.2) to obtain that

$$
\bigl | b _ { 1 } ( p , n , V ^ { \prime } ) ^ { 2 } - b _ { 1 } ( p , n , V ) ^ { 2 } \bigr | \leq \frac { \alpha } { n } \log ^ { \geq 1 } \biggl ( \frac { B } { n } \biggr ) \Bigl | \widetilde { \mathrm { V a r } } ( p , n , V ^ { \prime } ) - \widetilde { \mathrm { V a r } } ( p , n , V ) \Bigr | \ ,
$$

where we used that ${ \widetilde { \operatorname { V a r } } } ( p , n , V )$ and $\widetilde { \mathrm { V a r } } ( p , n , V ^ { \prime } )$ are both at least 2.

Moreover, from definitions we have $\log ^ { \ge 1 } ( B / n ) \le 2 L$ and $\begin{array} { r } { b _ { 2 } ( p , n , V ^ { \prime } ) = b _ { 2 } ( p , n , V ) = \frac { \beta H ^ { 2 } S L } { n } > 0 } \end{array}$ . Thus,

$$
\begin{array} { r l } {  k _ { ( F , \pm , n , V ^ { * } ) } - b ( j _ { F } , n , V )  = \frac { | \langle k _ { ( F , \pm , n , V ^ { * } ) } , F ^ { * } \rangle - \langle b ( j _ { F } , n , V )  ^ { 2 } } { b ( j _ { F } , n , V ^ { * } ) + b ( j _ { F } , n , V ) } } & { } \\ & { \lesssim \frac { \epsilon } { 2 } | b _ { \mathrm { t } } ( \cdot _ { \mathbb { P } ^ { * } } , N , V ^ { * } ) | ^ { 2 } - b ( j _ { \mathbb { P } ^ { * } } , N , V ) ^ { 2 } | } \\ & { \lesssim \frac { \epsilon } { 2 } \frac { \epsilon } { 2 \omega _ { \mathcal { F } } ( j _ { \mathbb { P } ^ { * } } , N , V ) } \frac { \epsilon } { | \tilde { \mathcal { F } } _ { \omega } ( j _ { \mathbb { P } ^ { * } } , N , V ) | ^ { 2 } } } \\ & { \lesssim \frac { \epsilon } { 2 \omega _ { \mathcal { F } } ( j _ { \mathbb { P } ^ { * } } , N , V ) } \frac { \epsilon } { | \tilde { \mathcal { F } } _ { \omega } ( j _ { \mathbb { P } ^ { * } } , N , V ^ { * } ) | ^ { 3 } } \frac { \epsilon } { | \tilde { \mathcal { F } } _ { \omega } ( j _ { \mathbb { P } ^ { * } } , N ^ { * } ) | ^ { 2 } } - \overbar { \mathcal { \mathrm { W } } } _ { ( \mu ^ { * } , N ^ { * } ) } \frac { \epsilon } { | \tilde { \mathcal { F } } _ { \omega } ( j _ { \mathbb { P } ^ { * } } , N ^ { * } ) | ^ { 2 } } } \\ &  \stackrel { \lesssim } \frac { 2 \omega _ { \mathcal { F } } | \langle B _ { \mu } \rangle | ^ { 2 } } { 2 \omega _ { \mathcal { F } } | \langle B _ { \mu } \rangle | ^ { 3 } } { p ( V ^ { * } , N ^ { * } ) }  p ( V ^ { * } , N  \end{array}
$$

Here, (°7) is the diference-of-squares identity. For (°8), observe that

$$
b ( p , n , V ) ^ { 2 } = \operatorname* { m a x } \big \{ b _ { 1 } ( p , n , V ) ^ { 2 } , b _ { 2 } ( p , n , V ) ^ { 2 } \big \} ,
$$

and the second branch (in $b _ { 2 } )$ is the same for $V$ and $V ^ { \prime }$ . Thus, taking the maximum with this common term cannot increase the diference in the numerator, while both terms in the denominator are at least $b _ { 2 } ( p , n , V )$ Inequality (°9) follows from the preceding bound on $b _ { 1 } ^ { 2 }$ , and $( ^ { \circ } 1 0 )$ follows from (°6). Equality (°11) unfolds the definition of $b _ { 2 } , ( ^ { \circ } 1 2 )$ uses $\log ^ { \ge 1 } ( B / n ) \le 2 L$ , and (°13) follows by choosing $\beta \geq$ 64α and using $S \geq 1$ □

## B.4 Proof of Expected Shortfall, Lemma A.7

Fix k, h, s, a. When $N _ { h } ^ { k } ( s , a ) = 0$ , we have $\bar { b } \big ( P _ { h , s , a } ^ { k } , n _ { h } ^ { k } ( s , a ) , V _ { h + 1 } ^ { \star } \big ) \geq H$ , and trivially the desired claim holds. The desired claim also trivially holds if $\mathrm { V a r } ^ { \geq 1 } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } ) / \sqrt { B } \geq H$ . It therefore sufices to consider the case $N _ { h } ^ { k } ( s , a ) \ge 1$ and $\mathrm { V a r } ^ { \geq 1 } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } ) / \sqrt { B } < H$

On $\mathcal { E } ,$ , Lemma A.2 gives, for every $n \in [ K ]$

$$
\widetilde { \mathrm { V a r } } ( \hat { P } _ { h , s , a , n } , n , V _ { h + 1 } ^ { \star } ) \geq \mathrm { V a r } ^ { \geq 1 } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } ) .
$$

Then, since $x \mapsto$ x $\log ^ { \geq 1 } ( B / ( n x ) )$ is nondecreasing on $( 0 , \infty )$ ),

$$
b _ { 1 } ( \hat { P } _ { h , s , a , n } , n , V _ { h + 1 } ^ { \star } ) \geq \sqrt { \frac { \alpha \mathrm { V a r } ^ { \geq 1 } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } ) } { n } \log ^ { \geq 1 } \frac { B } { n \mathrm { V a r } ^ { \geq 1 } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } ) } } .
$$

For

$$
\frac { \mathrm { V a r } ^ { \geq 1 } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } ) } { \sqrt { B } } < x \leq H ,
$$

define

$$
\delta _ { x } : = \frac { \left( \mathrm { V a r } ^ { \geq 1 } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } ) \right) ^ { 2 } } { B \left( \mathrm { V a r } ^ { \geq 1 } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } ) + H x \right) } .
$$

Since $x > \mathrm { V a r } ^ { \geq 1 } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } ) / \sqrt { B }$ and $x \leq H$ , we have

$$
\left( \mathrm { V a r } ^ { \geq 1 } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } ) \right) ^ { 2 } < B x ^ { 2 } \leq B H x ,
$$

and hence $\delta _ { x } \in ( 0 , 1 )$ . Moreover,

$$
\frac { 1 } { n \delta _ { x } } = \frac { B } { n \mathrm { V a r } ^ { \ge 1 } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } ) } \left( 1 + \frac { H x } { \mathrm { V a r } ^ { \ge 1 } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } ) } \right) .
$$

Thus, using $\log ^ { \geq 0 } ( a b ) \leq \log ^ { \geq 1 } ( a ) + \log b$ for $b \geq 1$

$$
\log ^ { \ge 0 } \frac { 1 } { n \delta _ { x } } \le \log ^ { \ge 1 } \frac { B } { n \mathrm { V a r } ^ { \ge 1 } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } ) } + \log \left( 1 + \frac { H x } { \mathrm { V a r } ^ { \ge 1 } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } ) } \right) .
$$

We will use the two bounds

$$
\log \left( 1 + \frac { H x } { \mathrm { V a r } ^ { \geq 1 } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } ) } \right) \leq \frac { H x } { \mathrm { V a r } ^ { \geq 1 } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } ) }
$$

and

$$
\log \left( 1 + \frac { H x } { \mathrm { V a r } ^ { \geq 1 } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } ) } \right) \leq \log ( 1 + H ^ { 2 } ) \leq 2 H .
$$

Subsequently, on $\mathcal { E } ,$ for every $n \in [ K ]$

$$
\begin{array} { r } { \sqrt { \frac { \alpha _ { 0 } \mathrm { V a r } ( P _ { h , s , a } , V _ { h + 1 } ^ { * } ) } { n } \log ^ { 2 0 } { \frac { 1 } { n \delta _ { x } } } } + \frac { \beta _ { 0 } H } { n } \log ^ { \ge 0 } { \frac { 1 } { n \delta _ { x } } } \overset { \le _ { 1 } } { \le } \frac { 1 } { 4 } b _ { 1 } ( \widehat { P } _ { h , s , a , n } , n , V _ { h + 1 } ^ { * } ) + \sqrt { \frac { \alpha _ { 0 } H x } { n } } + \frac { \beta _ { 0 } H } { n } \log ^ { \ge 1 } { \frac { B } { n } } + \frac { 2 \beta _ { 0 } H ^ { 2 } } { n } } \\ { \overset { \le _ { 2 } } { \le } \frac { 1 } { 4 } b _ { 1 } ( \widehat { P } _ { h , s , a , n , n } , n , V _ { h + 1 } ^ { * } ) + \frac { 1 } { 4 } b _ { 2 } ( \widehat { P } _ { h , s , a , n , n } , n , V _ { h + 1 } ^ { * } ) + \frac { x } { 2 } } \\ { \overset { \le _ { 3 } } { \le } \bar { b } ( \widehat { P } _ { h , s , a , n , n } , n , V _ { h + 1 } ^ { * } ) + \frac { x } { 2 } . \qquad } \end{array}
$$

For (°1), we use $\begin{array} { r } { \sqrt { u + v } \le \sqrt { u } + \sqrt { v } , \mathrm { V a r } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } ) \le \mathrm { V a r } ^ { \ge 1 } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } ) } \end{array}$ , the preceding lower bound on $b _ { 1 } .$ , and

$$
\log ^ { \geq 1 } \frac { B } { n \mathrm { V a r } ^ { \geq 1 } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } ) } \leq \log ^ { \geq 1 } \frac { B } { n } .
$$

For (°2), first note that

$$
{ \sqrt { \frac { \alpha _ { 0 } H x } { n } } } \leq { \frac { x } { 2 } } + { \frac { \alpha _ { 0 } H } { 2 n } } ,
$$

while $H \geq 1 , S \geq 1 , \log ^ { \geq 1 } ( B / n ) \geq 1 , \log ^ { \geq 1 } ( B / n ) \leq 2 L$ and $\beta \ge 4 ( \alpha _ { 0 } + 6 \beta _ { 0 } )$ ) imply

$$
\begin{array} { r } { \frac { \alpha _ { 0 } H } { 2 n } + \frac { \beta _ { 0 } H } { n } \log ^ { \ge 1 } \displaystyle \frac { B } { n } + \frac { 2 \beta _ { 0 } H ^ { 2 } } { n } \le \frac { ( \alpha _ { 0 } + 6 \beta _ { 0 } ) H ^ { 2 } } { 2 n } \log ^ { \ge 1 } \displaystyle \frac { B } { n } } \\ { \le \frac { 1 } { 4 } b _ { 2 } ( \widehat { P } _ { h , s , a , n } , n , V _ { h + 1 } ^ { \star } ) . } \end{array}
$$

Finally, (°3) follows directly from $\bar { b } = \operatorname* { m a x } \{ b _ { 1 } / 2 , b _ { 2 } / 2 \}$

For fixed $( h , s , a )$ , the successive transition observations can be generated in advance as an i.i.d. sequence with law $P _ { h , s , a } ,$ , and $\hat { P } _ { h , s , a , n }$ is the empirical transition row after its first n observations. Their $V _ { h + 1 } ^ { \star }$ values have mean $P _ { h , s , a } V _ { h + 1 } ^ { \star }$ , variance $\mathrm { V a r } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } )$ , and absolute centered values bounded by H. For positive visit counts, $P _ { h , s , a } ^ { k } = \hat { P } _ { h , s , a , n _ { h } ^ { k } ( s , a ) }$ . Therefore, on $\mathcal { E } .$

$$
( P _ { h , s , a } - P _ { h , s , a } ^ { k } ) V _ { h + 1 } ^ { \star } - \bar { b } ( P _ { h , s , a } ^ { k } , n _ { h } ^ { k } ( s , a ) , V _ { h + 1 } ^ { \star } ) > x
$$

implies the failure event in Theorem 4.1 with confidence $\delta _ { x }$ and slackness $x / 2$ . Since Theorem 4.1 is uniform over $n ,$ it applies at the random count $n _ { h } ^ { k } ( s , a )$ without conditioning on the count or on $\mathcal { E }$ . We can therefore bound

$$
\begin{array} { r l } & { \mathbb { P } \left( \mathcal { E } , ( P _ { h , s , a } - P _ { h , s , a } ^ { k } ) V _ { h + 1 } ^ { \star } - \bar { b } ( P _ { h , s , a } ^ { k } , n _ { h } ^ { k } ( s , a ) , V _ { h + 1 } ^ { \star } ) > x \right) } \\ & { \quad \quad \leq 1 6 \delta _ { x } \left( \frac { 4 \mathrm { V a r } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } ) } { x ^ { 2 } } + \frac { 2 H } { x } \right) } \\ & { \quad \quad \leq \frac { 6 4 \left( \mathrm { V a r } ^ { \geq 1 } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } ) \right) ^ { 2 } } { B x ^ { 2 } } . } \end{array}
$$

The final inequality follows by substituting the definition of $\delta _ { x }$ and using $\mathrm { V a r } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } ) \leq \mathrm { V a r } ^ { \geq 1 } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } )$

Since this holds for arbitrary $x \in ( \mathrm { V a r } ^ { \geq 1 } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } ) / \sqrt { B } , H ]$ , and the positive part in the statement is bounded by H, we compute

$$
\begin{array} { r l } & { \mathbb { E } \left[ \mathbb { I } ( \mathcal { E } ) \left( ( P _ { h , s , a } - P _ { h , s , a } ^ { k } ) V _ { h + 1 } ^ { \star } - \bar { b } ( P _ { h , s , a } ^ { k } , n _ { h } ^ { k } ( s , a ) , V _ { h + 1 } ^ { \star } ) \right) _ { + } \right] } \\ & { \quad \quad = \int _ { 0 } ^ { H } \mathbb { P } \left( \mathcal { E } , ( P _ { h , s , a } - P _ { h , s , a } ^ { k } ) V _ { h + 1 } ^ { \star } - \bar { b } ( P _ { h , s , a } ^ { k } , n _ { h } ^ { k } ( s , a ) , V _ { h + 1 } ^ { \star } ) > x \right) d x } \\ & { \quad \quad \le \frac { \mathrm { V a r } ^ { \ge 1 } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } ) } { \sqrt { B } } + \frac { 6 4 \left( \mathrm { V a r } ^ { \ge 1 } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } ) \right) ^ { 2 } } { B } \int _ { \mathrm { V a r } ^ { \ge 1 } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } ) / \sqrt { B } } ^ { H } \frac { 1 } { x ^ { 2 } } d x } \\ & { \quad \quad \le 6 5 \frac { \mathrm { V a r } ^ { \ge 1 } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } ) } { \sqrt { B } } . } \end{array}
$$

This completes the proof.

## B.5 Proof of the Variance Sum Bound, Lemma A.9

Denote by $\Delta _ { h , s , a } : = V _ { h } ^ { \star } ( s ) - r _ { h } ( s , a ) - P _ { h , s , a } V _ { h + 1 } ^ { \star }$ the Bellman gap of the triplet $( h , s , a )$ . Note that the Bellman optimality equation implies that $\Delta _ { h , s , a } \geq 0$ . For every triplet $( h , s , a )$ , we have

$$
\begin{array} { r l } & { \mathrm { V a r } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } ) = P _ { h , s , a } ( V _ { h + 1 } ^ { \star } ) ^ { 2 } - \big ( P _ { h , s , a } V _ { h + 1 } ^ { \star } \big ) ^ { 2 } } \\ & { \qquad = P _ { h , s , a } ( V _ { h + 1 } ^ { \star } ) ^ { 2 } - ( V _ { h } ^ { \star } ( s ) ) ^ { 2 } + 2 V _ { h } ^ { \star } ( s ) ( r _ { h } ( s , a ) + \Delta _ { h , s , a } ) - ( r _ { h } ( s , a ) + \Delta _ { h , s , a } ) ^ { 2 } } \end{array}
$$

$$
\begin{array} { r l } & { \overset { \circ _ { 1 } } { \leq } P _ { h , s , a } ( V _ { h + 1 } ^ { \star } ) ^ { 2 } - ( V _ { h } ^ { \star } ( s ) ) ^ { 2 } + 2 H ( r _ { h } ( s , a ) + \Delta _ { h , s , a } ) } \\ & { \overset { \circ _ { 2 } } { \leq } P _ { h , s , a } ( V _ { h + 1 } ^ { \star } ) ^ { 2 } - ( V _ { h } ^ { \star } ( s ) ) ^ { 2 } + 2 H + 2 H \Delta _ { h , s , a } } \end{array}
$$

where (°1) uses that $r _ { h } ( s , a ) + \Delta _ { h , s , a } \geq 0$ and the bound ${ V } _ { h } ^ { \star } \leq { H } ;$ and $( ^ { \circ } 2 )$ uses that $r _ { h } ( s , a ) \leq 1$ . Further summing over episodes $k \in [ K ]$ and time $h \in [ H ]$ , we conclude the proof as such:

$$
\begin{array} { r l } & { \mathbb { E } \left[ \displaystyle \sum _ { h , s , o } \mathrm { V a r } ( P _ { h , s , a } , V _ { h + 1 } ^ { * } ) N _ { h } ( s , a ) \right] } \\ & { \quad \stackrel { \mathrm { e } _ { 3 } } { \leq } \mathbb { E } \left[ \displaystyle \sum _ { k = 1 } ^ { K } \displaystyle \sum _ { h = 1 } ^ { H } \Bigl ( P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } ( V _ { h + 1 } ^ { * } ) ^ { 2 } - ( V _ { h } ^ { \star } ( s _ { h } ^ { k } ) ) ^ { 2 } \Bigr ) \right] + 2 H ^ { 2 } K + 2 H \mathbb { E } \left[ \displaystyle \sum _ { k = 1 } ^ { K } \displaystyle \sum _ { h = 1 } ^ { H } \Delta _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } \right] } \\ & { \quad \stackrel { \mathrm { e } _ { 4 } } { \leq } \mathbb { E } \left[ \displaystyle \sum _ { k = 1 } ^ { K } \Bigl ( P _ { H , s _ { H } ^ { k } , a _ { H } ^ { k } } ( V _ { H + 1 } ^ { * } ) ^ { 2 } - ( V _ { 1 } ^ { \star } ( s _ { 1 } ^ { k } ) ) ^ { 2 } \Bigr ) \right] + 2 H ^ { 2 } K + 2 H \underbrace { \mathbb { E } \left[ \displaystyle \sum _ { k = 1 } ^ { K } \displaystyle \sum _ { h = 1 } ^ { H } \Delta _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } \right] } _ { \leq \mathbb { E } \left[ \displaystyle \mathrm { R e g } ( K ) \right] \leq \mathbb { H } K } } \end{array}
$$

where $( ^ { \circ } 3 )$ follows by invoking $( ^ { \circ } 2 )$ ; and (°4) follows by simplifying the telescopic sum in the first expectation of (°3). In (°4), the first expectation is nonpositive since $P _ { H , s _ { H } ^ { k } , a _ { H } ^ { k } } ( V _ { H + 1 } ^ { \star } ) ^ { 2 } - ( V _ { 1 } ^ { \star } ( s _ { 1 } ^ { k } ) ) ^ { 2 } = - ( V _ { 1 } ^ { \star } ( s _ { 1 } ^ { k } ) ) ^ { 2 } \leq 0 ;$ and the second expectation is equal to the expected regret, so is trivially bounded by $H K$

For the second part, $\operatorname { V a r } ^ { \geq 1 } ( p , V ) \leq \operatorname { V a r } ( p , V ) + 1$ , so

$$
\begin{array} { r } { \mathbb { E } [ \mathrm { V a r } ^ { \star } ] \leq 4 H ^ { 2 } K + H K \leq 5 H ^ { 2 } K . } \end{array}
$$

## B.6 Proof of Lemma A.10

First, observe that

$$
\sum _ { k , h } \frac { 1 } { n _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) } \leq \sum _ { h , s , a } \left( 1 + \sum _ { n = 1 } ^ { N _ { h } ( s , a ) - 1 } \frac { 1 } { n } \right)
$$

where the 1 accounts for the first visit. Since

$$
\sum _ { n = 1 } ^ { N _ { h } ( s , a ) - 1 } \frac { 1 } { n } \le \sum _ { n = 1 } ^ { K } \frac { 1 } { n } \le 1 + \log K ,
$$

we conclude that $\begin{array} { r } { \sum _ { k , h } \frac { 1 } { n _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) } \le H S A ( 2 + \log K ) \le H S A L } \end{array}$

## B.7 Proof of the Concentration Result on Sums with Slackness, Proposition A.12

Let $( X _ { i } ) _ { i = 1 } ^ { \infty }$ be a sequence of i.i.d. random variables and let $\mathbb { E } [ X _ { i } ] = \mu$ and $\sigma ^ { 2 } = \mathbb { E } [ ( X _ { i } - \mu ) ^ { 2 } ]$ be their respective expectation and variance. Further assume that $| X _ { i } - \mu | \leq c \mathrm { a . s . }$ , for some $c \in \mathbb { R } _ { \geq 0 }$ . Recall that $\begin{array} { r } { \hat { \mu } _ { n } : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } X _ { i } } \end{array}$ denotes the empirical mean after n samples. We show that, for all $\Delta > 0$

$$
\sum _ { n = 1 } ^ { \infty } \mathbb { E } \big [ ( \hat { \mu } _ { n } - \mu - \Delta ) _ { + } \big ] \leq C \bigg ( \frac { \sigma ^ { 2 } } { \Delta } + c \bigg ) .\tag{B.3}
$$

where $C < \infty$ is a universal constant.

Note that for $\sigma = 0$ , we have $\mathbb { E } [ ( \hat { \mu } _ { n } - \mu - \Delta ) _ { + } ] = 0$ for all $\Delta > 0$ , so that Eq. B.3 is trivial. In the sequel, we assume that $\sigma > 0$

Fix $\Delta > 0$ . To bound the summand $\mathbb { E } [ ( \hat { \mu } _ { n } - \mu - \Delta ) _ { + } ]$ , we distinguish between two cases, depending on whether the visit count n is below or above the threshold:

$$
n ^ { \star } : = \left\lceil \frac { \sigma ^ { 2 } } { \Delta ^ { 2 } } + \frac { c ^ { 2 } } { \sigma ^ { 2 } } \right\rceil .
$$

Low visit counts are handled diferently because the tails spawned by Bernstein’s inequality are inadequate and the resulting bound is not tight enough. We therefore consider two cases.

• Partial sum for $n \leq n ^ { \star }$ . For low visit counts, a simple Jensen inequality is enough:

$$
\begin{array} { l } { { \displaystyle \sum _ { n = 1 } ^ { n ^ { * } } \mathbb { E } \big [ ( \hat { \mu } _ { n } - \mu - \Delta ) _ { + } \big ] \overset { ^ { \circ } } \leq \sum _ { n = 1 } ^ { n ^ { * } } \mathbb { E } \big [ | \hat { \mu } _ { n } - \mu | \big ] } } \\ { { \displaystyle \overset { ^ { \circ } } \leq \sum _ { n = 1 } ^ { n ^ { * } } \sqrt { \mathrm { V a r } ( \hat { \mu } _ { n } ) } } } \\ { { \displaystyle \overset { ^ { \circ } \geq } \sum _ { n = 1 } ^ { n ^ { * } } \sqrt { \frac { 1 } { n } \mathrm { V a r } ( X _ { 1 } ) } \overset { ^ { \circ } _ { 4 } } { \leq } 2 \sqrt { n ^ { \star } \sigma ^ { 2 } } \overset { ^ { \circ } \leq } O \bigg ( \frac { \sigma ^ { 2 } } { \Delta } + c \bigg ) } } \end{array}
$$

where (°1) follows by linearity of $\mathbb { E } [ - ]$ together with the inequality $( x - y ) _ { + } \leq | x |$ for $y \ge 0 ; ( ^ { \circ } 2 )$ uses $| x | = { \sqrt { x ^ { 2 } } }$ and invokes Jensen’s inequality; (°3) uses that $\hat { \mu } _ { n }$ is the average of n independent copies of $X _ { 1 } ; ( ^ { \circ } 4 )$ follows by a series–integral comparison; and $( ^ { \circ } 5 )$ unfolds the definition of $n ^ { \star }$ and uses the sub-additivity of $x \mapsto { \sqrt { x } }$

• Partial sum for $n > n ^ { \star }$ . The bound for high visit counts is slightly more involved and relies on Bernstein’s inequality. We have:

$$
\begin{array} { r l } { \displaystyle \sum _ { n = n ^ { * } } ^ { \infty } \mathbb { E } \big [ ( \widehat { \mu } _ { n } - \mu - \Delta ) _ { + } \big ] \leq \displaystyle \sum _ { n = n ^ { * } } ^ { \infty } \int _ { \Delta } ^ { \infty } \mathbb { F } ( \widehat { \mu } _ { n } - \mu \geq x ) \mathrm { d } x } \\ { \displaystyle \leq \displaystyle \sum _ { n = n ^ { * } } ^ { \infty } \int _ { \Delta } ^ { \infty } \exp \bigg ( - \frac { n x ^ { 2 } } { 2 ( \sigma ^ { 2 } + \frac { 3 } { s } c ^ { 2 } ) } \bigg ) \mathrm { d } x } \\ { \leq \displaystyle \sum _ { n = n ^ { * } } ^ { \infty } \int _ { \Delta } ^ { \infty } \exp \bigg ( - n \operatorname* { m i n } \bigg \{ \frac { x ^ { 2 } } { 4 \sigma ^ { 2 } } , \frac { 3 x } { 4 c } \bigg \} \bigg ) \mathrm { d } x } \\ { \displaystyle \overset { \lesssim } \sum _ { n = n ^ { * } } ^ { \infty } \bigg \{ \int _ { \Delta } ^ { \infty } \mathrm { e } ^ { - \frac { \kappa x ^ { 2 } } { 4 \sigma ^ { 2 } } } \mathrm { d } x + \displaystyle \int _ { 3 \sigma ^ { 2 } / c } ^ { \infty } \mathrm { e } ^ { - \frac { 3 a x } { 4 \sigma } } \mathrm { d } x \bigg \} } \\ { \displaystyle \overset { \lesssim } \sum _ { n = n ^ { * } } ^ { \infty } \bigg \{ 2 \Delta \mathrm { e } ^ { - \frac { \kappa a ^ { 2 } } { 4 \sigma ^ { 2 } } } + \frac { 4 C } { 3 n } \mathrm { e } ^ { - 2 \frac { \kappa a ^ { 2 } } { 4 \sigma ^ { 2 } } } \bigg \} = O \bigg ( \frac { \sigma ^ { 2 } } { \Delta } \bigg ) + O \bigg ( \frac { c ^ { 3 } } { \widehat { \sigma ^ { 2 } n ^ { * } } } \bigg ) \overset { \cong } 0 \bigg ( \frac { \sigma ^ { 2 } } { \Delta } + c \bigg ) } \end{array}
$$

where (°6) invokes Bernstein’s inequality (Lemma B.1); (°7) splits the integral depending on which of the quadratic and the linear terms dominate; and (°8) uses standard bounds on the error function together with $n \geq n ^ { \star }$ for the first integral; and (°9) uses that $n ^ { \star } \geq c ^ { 2 } / \sigma ^ { 2 }$

We conclude by combining the two parts (°5) and (°9).

## B.8 Proof of the $\mathcal { T } _ { 3 } ^ { + }$ Bound, Lemma A.16

From the definition of $b _ { h } ^ { k }$ and max $\{ a , b \} \leq a + b$ for $a , b \geq 0$ , we have

$$
\sum _ { k , h } b _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) = \sum _ { k , h } b \left( P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } ^ { k } , n _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) , V _ { h + 1 } ^ { k } \right)
$$

$$
\leq \sum _ { k , h } b _ { 1 } \left( P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } ^ { k } , n _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) , V _ { h + 1 } ^ { k } \right) + \sum _ { k , h } b _ { 2 } \left( P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } ^ { k } , n _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) , V _ { h + 1 } ^ { k } \right)
$$

Now, fix $k , h ,$ and write $\hat { P } = P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } ^ { k } , P = P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } , n = n _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) , \hat { V } = V _ { h + 1 } ^ { k } ;$ , and $V ^ { \star } = V _ { h + 1 } ^ { \star }$ for ease of notation. We again utilize the strategy of splitting $\hat { V }$ into $V ^ { \star }$ and $\hat { V } - V ^ { \star }$ . In particular, $\mathrm { V a r } ^ { \geq 1 } ( P , \hat { V } ) \leq$ $2 \mathrm { V a r } ^ { \geq 1 } ( P , V ^ { \star } ) + 2 \mathrm { V a r } ( P , \hat { V } - V ^ { \star } )$ . Subsequently, under $\mathcal { E } .$ , Lemma $\mathrm { A . 2 }$ gives us

$$
\begin{array} { r l } & { \widetilde { \mathrm { V a r } } \left( \hat { P } , n , \hat { V } \right) \lesssim \mathrm { V a r } ^ { \geq 1 } ( P , \hat { V } ) + \displaystyle \frac { H ^ { 2 } S L } { n } } \\ & { \qquad \lesssim \mathrm { V a r } ^ { \geq 1 } ( P , V ^ { \star } ) + \mathrm { V a r } ( P , \hat { V } - V ^ { \star } ) + \displaystyle \frac { H ^ { 2 } S L } { n } } \end{array}
$$

which due to the monotonicity of $x \mapsto x \log ^ { \geq 1 } { \frac { B } { n x } }$ implies

$$
\begin{array} { r l } & { b _ { 1 } \left( \hat { P } , n , \hat { V } \right) \leq \sqrt { \frac { \alpha \widetilde { \mathrm { V a r } } ( \hat { P } , n , \hat { V } ) \log ^ { \geq 1 } \frac { B } { n \widetilde { \mathrm { V a r } } ( \hat { P } , n , \hat { V } ) } } { n } } } \\ & { \qquad \lesssim \sqrt { \frac { \mathrm { V a r } ^ { \geq 1 } ( P , V ^ { \star } ) \log ^ { \geq 1 } \frac { B } { n \sqrt { \mathrm { V a r } ^ { \geq 1 } ( P , V ^ { \star } ) } } } { n } } + \sqrt { \frac { \mathrm { V a r } ( P , \hat { V } - V ^ { \star } ) L } { n } } + \frac { H S L } { n } . } \end{array}
$$

Thus, under $\mathcal { E } ,$ we have

$$
\begin{array} { r l } { \mathcal { T } _ { 3 } ^ { + } = \sum _ { k , h } b _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) } & { } \\ & { \lesssim \sum _ { k , h } b _ { 1 } \Big ( P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } ^ { k } , n _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) , V _ { h + 1 } ^ { k } \Big ) + H ^ { 3 } S ^ { 2 } A L ^ { 2 } } \\ & { \lesssim \sum _ { k , h } \sqrt { \frac { \mathrm { V a r } ^ { 2 1 } \big ( P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } , V _ { h + 1 } ^ { k } \big ) \log ^ { 2 1 } \frac { B } { n _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) \sqrt { \mathrm { V a r } ^ { 2 1 } ( P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } , V _ { h + 1 } ^ { k } ) } } } { n _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) } } } \\ & { \lesssim \sum _ { k , h } \sqrt { \frac { \mathrm { V a r } ( P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } , V _ { h + 1 } ^ { k } - V _ { h + 1 } ^ { k } ) L } { n _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) } + \sum _ { k , h } \frac { H S L } { n _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) } } } \\ &  \quad + \sum _ { k , h } \sqrt  \frac { \mathrm { V a r } ( P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } , V _ { h + 1 } ^ { k } - V _ { h + 1 } ^ { k } ) L } { n _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) } + \sum _ { k , h } \frac { H S L }  \end{array}
$$

We can bound the third sum by $H ^ { 2 } S ^ { 2 } A L ^ { 2 }$ via Lemma A.10. Further, we can handle the second sum exactly as we handled a similar looking term while bounding $\mathcal { T } _ { 2 } ^ { + }$ . Specifically, Cauchy-Schwarz gives us

$$
\begin{array} { r l }   { \sum _ { k , h } \sqrt { \frac { \mathrm { V a r } ( P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } , V _ { h + 1 } ^ { k } - V _ { h + 1 } ^ { \star } ) L } { n _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) } } \le \sqrt { \sum _ { k , h } \mathrm { V a r } ( P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } , V _ { h + 1 } ^ { k } - V _ { h + 1 } ^ { \star } ) L } \sqrt { \sum _ { k , h } \frac { 1 } { n _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) } } } \end{array}
$$

It remains to bound the first sum. We have

$$
\begin{array} { l l } { \displaystyle \sum _ { k , h } \sqrt { \frac { \mathrm { V a r } ^ { \geq 1 } ( P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } , V _ { h + 1 } ^ { \star } ) \log ^ { \geq 1 } \frac { B } { n _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) \mathrm { V a r } ^ { \geq 1 } ( P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } , V _ { h + 1 } ^ { \star } ) } } { n _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) } } } \\ { \displaystyle \leq H ^ { 2 } S A L ^ { 2 } + \sum _ { k , h \colon N _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) \geq 1 } \sqrt { \frac { \mathrm { V a r } ^ { \geq 1 } ( P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } , V _ { h + 1 } ^ { \star } ) \log ^ { \geq 1 } \frac { B } { n _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) \mathrm { V a r } ^ { \geq 1 } ( P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } , V _ { h + 1 } ^ { \star } ) } } { n _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) } } } \end{array}
$$

$$
\begin{array} { l } { { \displaystyle \le H ^ { 2 } S A L ^ { 2 } + \sum _ { h , s , a } \sum _ { n = 1 } ^ { N _ { h } ( s , a ) } \sqrt { \frac { \mathrm { V a r } ^ { \ge 1 } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } ) \log ^ { \ge 1 } \frac { B } { n \mathrm { V a r } ^ { \ge 1 } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } ) } } } } } \\ { { \displaystyle \stackrel { \circ } { \lesssim } H ^ { 2 } S A L ^ { 2 } + \sum _ { h , s , a } \left( \sqrt { B } + \frac { \mathrm { V a r } ^ { \ge 1 } ( P _ { h , s , a } , V _ { h + 1 } ^ { \star } ) N _ { h } ( s , a ) } { \sqrt { B } } \right) } } \\ { { \displaystyle \stackrel { \circ } { = } H ^ { 2 } S A L ^ { 2 } + H S A \sqrt { B } + \frac { \mathrm { V a r } ^ { \star } } { \sqrt { B } } , } } \end{array}
$$

with (°1) following from Lemma A.15 bounding the MOSS-style sum and (°2) holding by the definition of $\operatorname { V a r } ^ { \star }$ . By the choice of $\begin{array} { r } { B = \frac { H K } { S A } } \end{array}$ , the first sum is therefore bounded by $\begin{array} { r } { \sqrt { H ^ { 3 } S A K } + \sqrt { \frac { S A } { H K } } \mathrm { V a r } ^ { \star } + H ^ { 2 } S A L ^ { 2 } } \end{array}$ (up to a universal constant).

Since in the worst case $b _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) \lesssim H ^ { 2 } S L$ , we conclude that

$$
\begin{array} { l l l } { \displaystyle \mathbb { E } [ \mathcal { T } _ { 3 } ^ { + } ] = \mathbb { E } [ \mathbb { I } ( \mathcal { E } ) \mathcal { T } _ { 3 } ^ { + } ] + \mathbb { E } [ \mathbb { I } ( \mathcal { E } ^ { c } ) \mathcal { T } _ { 3 } ^ { + } ] } \\ { \displaystyle \qquad \lesssim \sqrt { H ^ { 3 } S A K } + H ^ { 3 } S ^ { 2 } A L ^ { 2 } + \sqrt { H S A L ^ { 2 } \mathbb { E } [ \mathrm { V a r } ^ { \mathrm { d i f f } } ] } + \sqrt { \frac { S A } { H K } } \mathbb { E } [ \mathrm { V a r } ^ { \star } ] + \frac { H ^ { 3 } S L K } { K ^ { 3 } } } \\ { \displaystyle \qquad \lesssim \sqrt { H ^ { 3 } S A K } + H ^ { 3 } S ^ { 2 } A L ^ { 2 } + \sqrt { H S A L ^ { 2 } \mathbb { E } [ \mathrm { V a r } ^ { \mathrm { d i f f } } ] } . } \end{array}
$$

Here, the first inequality is due to Jensen’s inequality and Lemma A.2 bounding $\mathbb { P } ( \mathcal { E } ^ { c } ) \le 1 / K ^ { 3 }$ , and the second inequality is by Lemma A.9 bounding $\mathbb { E } [ \mathrm { V a r } ^ { \star } ] \lesssim H ^ { 2 } K$ □

## B.9 Proof of the $\mathrm { V a r } ^ { \mathrm { d i f f } }$ Bound

Fix arbitrary $k \in [ K ]$ . For each $h \in [ H + 1 ]$ , write $D _ { h } : = V _ { h } ^ { k } - V _ { h } ^ { \star }$ . We directly compute for each h,

$$
\begin{array} { r l } & { \mathrm { V a r } ( P _ { k _ { \perp } , s _ { n } , n } ^ { \ast } , D _ { h + 1 } ) } \\ & { = P _ { h , s _ { n } , s _ { n } ^ { \ast } } ( B _ { h + 1 } ) ^ { 2 } - ( P _ { h , s _ { n } ^ { \ast } , s _ { n } ^ { \ast } } D _ { h + 1 } ) ^ { 2 } } \\ & { \leq P _ { h , s _ { n } , s _ { n } ^ { \ast } } ( B _ { h + 1 } ) ^ { 2 } - ( D _ { h } ( s _ { h } ^ { h } ) ) ^ { 2 } + ( D _ { h } ( s _ { h } ^ { h } ) ) ^ { 2 } - ( P _ { h , s _ { n } ^ { \ast } , s _ { n } ^ { \ast } } D _ { h + 1 } ) ^ { 2 } } \\ & { = P _ { h , s _ { n } , s _ { n } ^ { \ast } } ( B _ { h + 1 } ) ^ { 2 } - ( D _ { h } ( s _ { h } ^ { h } ) ) ^ { 2 } + ( D _ { h } ( s _ { h } ^ { h } ) + P _ { h , s _ { n } ^ { \ast } , s _ { n } ^ { \ast } } D _ { h + 1 } ) ( D _ { h } ( s _ { h } ^ { h } ) - P _ { h , s _ { n } ^ { \ast } , s _ { n } ^ { \ast } } D _ { h + 1 } ) } \\ & { \leq P _ { h , s _ { n } ^ { \ast } , s _ { n } ^ { \ast } } ( B _ { h + 1 } ) ^ { 2 } - ( D _ { h } ( s _ { h } ^ { h } ) ) ^ { 2 } + 2 H [ D _ { h } ( s _ { h } ^ { h } ) - P _ { h , s _ { n } ^ { \ast } , s _ { n } ^ { \ast } } D _ { h + 1 } ] } \\ &  = P _ { h , s _ { n } ^ { \ast } , s _ { n } ^ { \ast } } ( B _ { h + 1 } ) ^ { 2 } - ( D _ { h } ( s _ { h } ^ { h } ) ) ^ { 2 } + 2 H ( D _ { h } ( s _ { h } ^ { h } ) - P _  h , s _ { n } ^  \ast \end{array}
$$

Observe that

$$
\begin{array} { r l } & { ~ \displaystyle \sum _ { h } \mathbb { E } [ P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } ( D _ { h + 1 } ) ^ { 2 } - ( D _ { h } ( s _ { h } ^ { k } ) ) ^ { 2 } ] } \\ & { = - \mathbb { E } \left[ ( D _ { 1 } ( s _ { 1 } ^ { k } ) ) ^ { 2 } \right] + \displaystyle \sum _ { h } \mathbb { E } [ P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } ( D _ { h + 1 } ) ^ { 2 } - ( D _ { h + 1 } ( s _ { h + 1 } ^ { k } ) ) ^ { 2 } ] } \\ & { = - \mathbb { E } \left[ ( D _ { 1 } ( s _ { 1 } ^ { k } ) ) ^ { 2 } \right] \leq 0 } \end{array}
$$

and

$$
\begin{array} { r l } {  { \sum _ { h } \mathbb { E } [ D _ { h } ( s _ { h } ^ { k } ) - P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } D _ { h + 1 } ] } \quad } & { } \\ & { = \mathbb { E } [ D _ { 1 } ( s _ { 1 } ^ { k } ) ] + \sum _ { h } \mathbb { E } [ D _ { h + 1 } ( s _ { h + 1 } ^ { k } ) - P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } D _ { h + 1 } ] } \\ & { = \mathbb { E } [ D _ { 1 } ( s _ { 1 } ^ { k } ) ] = \mathbb { E } [ V _ { 1 } ^ { k } ( s _ { 1 } ^ { k } ) - V _ { 1 } ^ { \star } ( s _ { 1 } ^ { k } ) ] . } \end{array}
$$

Subsequently, we have

$$
\begin{array} { r l } & { \mathbb { E } \left[ \displaystyle \sum _ { h = 1 } ^ { H } \mathrm { V a r } ( P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } , D _ { h + 1 } ) \right] } \\ & { \leq 4 H \mathbb { E } \left[ \displaystyle \sum _ { h = 1 } ^ { H } \left( ( P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } ^ { k } - P _ { h , s _ { h } ^ { k } , a _ { h } ^ { k } } ) V _ { h + 1 } ^ { k } + b _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) \right) _ { + } \right] + 2 H \mathbb { E } \left[ ( V _ { 1 } ^ { \star } ( s _ { 1 } ^ { k } ) - V _ { 1 } ^ { k } ( s _ { 1 } ^ { k } ) ) _ { + } \right] . } \end{array}
$$

We obtain the desired result by summing over k and applying the definitions of $\mathcal { T } ^ { - }$ and ${ \mathcal { T } } ^ { + }$

## C Proof of Theorem 4.1

In this section we prove Theorem 4.1.

Theorem C.1 (Restatement of Theorem 4.1). Let $X _ { 1 } , X _ { 2 } , . . .$ . be i.i.d. random variables such that

$$
\mathbb { E } [ X _ { i } ] = \mu , \qquad \operatorname { V a r } ( X _ { i } ) = \sigma ^ { 2 } , \qquad | X _ { i } - \mu | \leq c \quad a . s .
$$

For $n \geq 1$ , write $\begin{array} { r } { \hat { \mu } _ { n } : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } X _ { i } } \end{array}$ . Set $\alpha _ { 0 } = 4$ and $\beta _ { 0 } = 2 / 3$ . Then, for every $\delta \in ( 0 , 1 )$ and $\Delta > 0$

$$
\mathbb { P } \left( \exists n \geq 1 : \hat { \mu } _ { n } + \sqrt { \frac { \alpha _ { 0 } \sigma ^ { 2 } } { n } } \log ^ { \geq 0 } \left( \frac { 1 } { n \delta } \right) + \frac { \beta _ { 0 } c } { n } \log ^ { \geq 0 } \left( \frac { 1 } { n \delta } \right) + \Delta \leq \mu \right) \leq 1 6 \delta \cdot \left( \frac { \sigma ^ { 2 } } { \Delta ^ { 2 } } + \frac { c } { \Delta } \right) .
$$

We will rely on the technical lemmas below.

Lemma C.2 (Maximal Bernstein inequality). Let $( X _ { n } )$ be a sequence $o f i . i . d .$ random variables with mean $\mu$ and variance $\sigma ^ { 2 }$ and such that $| X _ { n } - \mu | \leq c \ a . s$ . Let $S _ { n } : = X _ { 1 } + . . . + X _ { n }$ . Then, for every integer $N \geq 1$ and $x > 0$

$$
\mathbb { P } \Bigg ( \operatorname* { m a x } _ { 1 \leq n \leq N } ( S _ { n } - \mathbb { E } [ S _ { n } ] ) \geq x \Bigg ) \leq \exp \Bigg ( \frac { - x ^ { 2 } } { 2 ( N \sigma ^ { 2 } + c x / 3 ) } \Bigg ) ~ .
$$

Proof of Lemma C.2. This result is folklore.

The cases $c = 0$ and $\sigma = 0$ are trivial, because in either case $X _ { n } = \mu { \mathrm { ~ a . s . } }$ . and the probability bound is trivial. $\mathrm { S o }$ , we assume in the sequel that $c > 0$ and $\sigma > 0$ . Following typical computations for the proof of Bernstein’s inequality, the Laplace transform of $X _ { n }$ can be bounded as such (see [Boucheron et al., 2013, Corollary 2.11 and the discussion that follows $\mathrm { i t } ] )$ :

$$
\forall \lambda \in \left( 0 , \frac { 3 } { c } \right) : \qquad \log \mathbb { E } [ \exp ( X _ { k } - \mu ) ] \leq \frac { \lambda ^ { 2 } \sigma ^ { 2 } } { 2 \left( 1 - \frac { 1 } { 3 } \lambda c \right) } = : \psi ( \lambda ) \ .\tag{C.1}
$$

So for all $\lambda \in ( 0 , 3 / c ) , W _ { n } ^ { \lambda } : = \exp ( \lambda ( S _ { n } - \mathbb { E } [ S _ { n } ] ) - n \psi ( \lambda ) )$ is a non-negative super-martingale. So:

$$
\mathbb P \Bigg ( \operatorname* { m a x } _ { 1 \leq n \leq N } ( S _ { n } - \mathbb E [ S _ { n } ] ) \geq x \Bigg ) \leq \operatorname* { i n f } _ { \lambda > 0 } \mathbb P \Bigg ( \operatorname* { m a x } _ { 1 \leq n \leq N } W _ { n } ^ { \lambda } \geq \mathrm { e } ^ { - N \psi ( \lambda ) + \lambda x } \Bigg )
$$

$$
\begin{array} { l } { \stackrel { \circ _ { 1 } } { \leq } \displaystyle \operatorname* { i n f } _ { \lambda > 0 } \left. \mathbb { E } [ W _ { 0 } ^ { \lambda } ] \mathrm { e } ^ { N \psi ( \lambda ) - \lambda x } \right. \leq \exp \left( - \frac { x ^ { 2 } } { 2 ( N \sigma ^ { 2 } + c x / 3 ) } \right) } \end{array}
$$

where (°1) follows by Ville’s inequality and (°2) is obtained by picking $\lambda = x / ( N \sigma ^ { 2 } + c x / 3 )$

Lemma C.3. For all $\alpha , \beta > 0$ , the function $\phi \equiv \phi _ { \alpha , \beta } : \mathbb { R } _ { \ge 0 } \to \mathbb { R } _ { \ge 0 }$ given by

$$
\phi ( x ) \equiv \phi _ { \alpha , \beta } ( x ) : = \frac { x ^ { 2 } } { 2 ( \alpha + \beta x ) }
$$

is super-additive<sup>1</sup> and increasing on its domain.

Proof of Lemma C.3. The derivative of ϕ is $\begin{array} { r } { \phi ^ { \prime } ( x ) = \frac { x ( 2 \alpha + \beta x ) } { 2 ( \alpha + \beta x ) ^ { 2 } } > 0 } \end{array}$ for $x > 0$ , so $\phi$ is increasing on $\mathbb { R } _ { \geq 0 }$ . For the super-additivity, remark that

$$
\phi ( x + y ) - \phi ( x ) - \phi ( y ) = { \frac { \alpha x y ( 2 \alpha + \beta ( x + y ) ) } { 2 ( \alpha + \beta x ) ( \alpha + \beta y ) ( \alpha + \beta ( x + y ) ) } } \geq 0
$$

for $x , y \geq 0$

Proof of Theorem C.1. We write ${ \mathcal E } \equiv { \mathcal E } _ { \delta , \Delta }$ the event of interest. If $\sigma ^ { 2 } = 0$ then $X _ { k } = \mu \mathrm { ~ a . s ~ }$ . so $\mathbb { P } ( \mathcal { E } ) = 0$ trivially. In the sequel, we assume $\sigma ^ { 2 } > 0$ . We bound $\mathbb { P } ( \mathcal { E } )$ by doing a union bound with a geometric grid:

$$
\mathbb { P } ( \mathcal { E } ) \leq \sum _ { \ell = 0 } ^ { \infty } \mathbb { P } \Bigg ( \exists n \in \mathbb { I } 2 ^ { \ell } , 2 ^ { \ell + 1 } \mathbb { I } \colon \hat { \mu } _ { n } + \sqrt { \frac { 4 \sigma ^ { 2 } } { n } \log ^ { \geq 0 } \left( \frac { 1 } { n \delta } \right) } + \frac { 2 c } { 3 n } \log ^ { \geq 0 } \left( \frac { 1 } { n \delta } \right) + \Delta \leq \mu \Bigg )
$$

$$
\sum _ { \ell = 0 } ^ { \infty _ { 1 } } \mathbb { P } \Bigg ( \exists n \leq 2 ^ { \ell + 1 } : S _ { n } - n \mu + \sqrt { 4 \sigma ^ { 2 } 2 ^ { \ell } \log ^ { \geq 0 } \biggl ( \frac { 1 } { 2 ^ { \ell + 1 } \delta } \biggr ) } + \frac { 2 c } { 3 } \log ^ { \geq 0 } \biggl ( \frac { 1 } { 2 ^ { \ell + 1 } \delta } \biggr ) + 2 ^ { \ell } \Delta \leq 0 \Bigg )
$$

$$
\stackrel { \circ 2 } { \leq } \sum _ { \ell = 0 } ^ { \infty } \exp \left( - \phi _ { \ell } \left( \sqrt { 4 \sigma ^ { 2 } 2 ^ { \ell } \log ^ { \geq 0 } \left( \frac { 1 } { 2 ^ { \ell + 1 } \delta } \right) } + \frac { 2 c } { 3 } \log ^ { \geq 0 } \left( \frac { 1 } { 2 ^ { \ell + 1 } \delta } \right) + 2 ^ { \ell } \Delta \right) \right)
$$

$$
\underset { \ell = 0 } { \overset { \circ _ { 3 } } { \leq } } \sum _ { \ell = 0 } ^ { \infty } \underset { \ell = \ell \mathrm { m } } { \underbrace { \exp \left( - \phi _ { \ell } \left( \sqrt { 4 \sigma ^ { 2 } 2 ^ { \ell } \log ^ { \geq 0 } \left( \frac { 1 } { 2 ^ { \ell + 1 } \delta } \right) } + \frac { 2 c } { 3 } \log ^ { \geq 0 } \left( \frac { 1 } { 2 ^ { \ell + 1 } \delta } \right) \right) \right) } }  \underset { \mathrm { F a c t o r ~ ( I ) } _ { \ell } } { \underbrace { \exp \left( - \phi _ { \ell } \left( 2 ^ { \ell } \Delta \right) \right) } }
$$

where in (°1) we use that $\textstyle n \mapsto \log ^ { \geq 0 } ( { \frac { 1 } { n \delta } } )$ is non-increasing; in (°2) we invoke the maximal Bernstein inequality (Lemma C.2) and introduce the notation $\phi _ { \ell } \equiv \phi _ { \alpha _ { \ell } , \beta _ { \ell } }$ (see Lemma C.3) with $\alpha _ { \ell } : = 2 ^ { \ell + 1 } \sigma ^ { 2 }$ and $\beta _ { \ell } \equiv \beta : = c / 3 ;$ and in (°3) we use the super-additivity of $\phi _ { \ell }$ (see Lemma C.3).

We proceed by bounding the two factors (I)<sub>ℓ</sub> and (II)<sub>ℓ</sub>.

Fix $\ell \geq 0$ . For short, write $m : = 2 ^ { \ell }$ and $\begin{array} { r } { L : = \log ^ { \geq 0 } \bigl ( \frac { 1 } { 2 m \delta } \bigr ) } \end{array}$ . Recall that $\alpha \equiv \alpha _ { \ell } = 2 m \sigma ^ { 2 }$ and $\beta = c / 3$

• Factor $( \mathrm { I } ) _ { \ell } \colon$ For the first factor $( \mathrm { I } ) _ { \ell } \colon$ , by setting $q : = \sqrt { 2 \alpha L } + 2 \beta L$ , we show via straight-forward algebra that $q ^ { 2 } - 2 L ( \alpha + \beta q ) = 2 \beta L \sqrt { 2 \alpha L } \ge 0$ . This readily implies $\begin{array} { r } { \phi _ { \ell } ( q ) = \frac { q ^ { 2 } } { 2 ( \alpha + \beta q ) } \geq L } \end{array}$ . So:

$$
\begin{array} { r l r } {  { \mathrm { ( I ) } _ { \ell } \equiv \exp \Bigl ( - \phi _ { \ell } \Bigl ( - \sqrt { 2 \alpha L } + 2 \beta L \Bigr ) \Bigr ) \equiv \exp ( - \phi _ { \ell } ( q ) ) } } \\ & { } & { \leq \exp ( - \log ^ { \geq 0 } ( \frac { 1 } { 2 m \delta } ) ) \stackrel { \circ 4 } { \leq } 2 m \delta } \end{array}\tag{C.2}
$$

where (°4) follows from $\exp ( - \log ^ { \geq 0 } ( 1 / x ) ) \leq \operatorname* { m i n } \{ 1 , x \} \leq x .$

• Factor $( \operatorname { I I } ) _ { \ell } { \mathrm { : } }$ Continuing with the second factor $( \operatorname { I I } ) _ { \ell } ,$ we have

$$
\mathrm { ( I I ) } _ { \ell } \equiv \exp \left( - { \frac { m ^ { 2 } \Delta ^ { 2 } } { 2 \big ( 2 m \sigma ^ { 2 } + { \frac { c m \Delta } { 3 } } \big ) } } \right) = \exp \left( - { \frac { m \Delta ^ { 2 } } { 4 \sigma ^ { 2 } + { \frac { 2 c \Delta } { 3 } } } } \right) .\tag{C.3}
$$

Injecting Eq. (C.2) and Eq. (C.3) back in to (°3), we pursue the bound of $\mathbb { P } ( \mathcal { E } )$ as such:

$$
\begin{array} { r l r } {  { \mathbb { P } ( \mathscr { E } ) \overset { \circ _ { 5 } } { \leq } 2 \delta \sum _ { \ell = 0 } ^ { \infty } 2 ^ { \ell } \exp ( - 2 ^ { \ell } \cdot \frac { \Delta ^ { 2 } } { 4 \sigma ^ { 2 } + \frac { 2 c \Delta } { 3 } } ) } } \\ & { \overset { \circ _ { 6 } } { \leq } 2 \delta \cdot 2 \displaystyle \int _ { 0 } ^ { \infty } \exp ( - \frac { \Delta ^ { 2 } } { 4 \sigma ^ { 2 } + \frac { 2 c \Delta } { 3 } } \cdot x ) \mathrm { d } x = \frac { 4 \delta } { \Delta ^ { 2 } } ( 4 \sigma ^ { 2 } + \frac { 2 c \Delta } { 3 } ) \leq 1 6 \delta ( \frac { \sigma ^ { 2 } } { \Delta ^ { 2 } } + \frac { c } { \Delta } ) . } \end{array}
$$

where (°5) follows by injecting Eq. (C.2) and Eq. (C.3) into (°3); and (°6) follows by an integral-series comparison. □

## D Best Policy Identification

## D.1 Related Work

First we discuss additional related work on the problem of best-policy identification in episodic MDPs.

Jin et al. [2018] describe a standard regret-to-PAC conversion for episodic MDPs which selects a policy from a single uniformly random episode, and only achieves a constant success probability (once the average regret is suficiently small). A regret-to-PAC conversion from a high-probability regret bound can instead retain the same success probability if the conversion instead outputs a mixture policy, but it is generally more common to require the learner to output a single policy [Wagenmaker et al., 2022]. Menard et al. [2021] also discuss the poor dependence on the failure probability if a single uniformly random episode’s policy is chosen. However, Kaufmann et al. [2021] note that it is instead possible to amplify the success probability by estimating the values of multiple policies that have constant probability of being $O ( \varepsilon )$ -optimal and outputting the best; this is the same approach used within our Algorithm 2. The minimax sample complexity lower bound $\Omega ( H ^ { 3 } S A \log ( 1 / \delta ) / \varepsilon ^ { 2 } )$ (measured in episodes) is due to Domingues et al. [2021], and it requires the mild additional assumption that $H \geq \Omega ( \log _ { A } S )$ , as well as that $S , A ,$ and $H / \varepsilon$ are at least certain absolute constants. We refer to their Theorems $7$ and 10 for the precise requirements. This lower bound allows the learner to have a random stopping time and concerns its expected stopping time, so it applies to our Corollary 6.2 which has a deterministic stopping time. Some prior algorithms discussed below use random stopping times and only bound them with high probability, but such results can be easily converted to algorithms with deterministic stopping times by simply terminating the algorithm once the budget from its proven complexity guarantee is reached.

Next we compare with other sample complexity bounds for $( \varepsilon , \delta ) – \mathrm { P A C }$ best-policy identification, measuring episodes and assuming rewards are in [0, 1]. Throughout we abbreviate $L _ { 1 } = \log ( e S A H / \delta ) , L _ { 2 } =$ $\log ( e S A H ^ { 2 } L _ { 1 } / \varepsilon )$ . The BPI-UCRL algorithm of Kaufmann et al. [2021] has a suboptimal outer $H ^ { 4 }$ horizon dependence rather than $H ^ { 3 }$ . The BPI-UCBVI algorithm of Menard et al. [2021] give a sample complexity bound for suficiently small $\varepsilon$ of $\begin{array} { r l } { O \left( \frac { H ^ { 3 } S A } { \varepsilon ^ { 2 } } L _ { 1 } L _ { 2 } ^ { 2 } \right) } \end{array}$ . After expanding the hidden logarithmic factors, to the best of our knowledge the ORLC algorithm of Dann et al. [2019] yields a PAC sample complexity bound of $\begin{array} { r } { O \left( \left( \frac { H ^ { 3 } S A } { \varepsilon ^ { 2 } } + \frac { H ^ { 5 } S ^ { 2 } A } { \varepsilon } \right) L _ { 2 } ( L _ { 1 } + \log L _ { 2 } ) \right) } \end{array}$ . After augmenting the state space with a time index to apply the time-homogeneous results of Lee and Oh [2025] to our inhomogenous setting, their EQO-based method achieves a PAC complexity bound of $\begin{array} { r } { O \left( \frac { H ^ { 3 } S A } { \varepsilon ^ { 2 } } L _ { 1 } L _ { 2 } + \frac { H ^ { 3 } S ^ { 2 } A } { \varepsilon } ( L _ { 1 } + \log L _ { 2 } ) L _ { 2 } \right) } \end{array}$ . In summary, the best sample complexity guarantees for best-policy identification still sufer from a gap of multiplicative logarithmic factors relative to the minimax lower bound.

Additionally, the aforementioned regret-to-PAC conversions [Kaufmann et al., 2021] could be applied to any algorithms that enjoy suitable sublinear expected regret or high-probability regret bounds, including the nearly minimax-optimal algorithms from Table 1, but such reductions would still retain the suboptimal logarithmic factors. (We note that if there did exist a high-probability regret bound with suboptima dependence only on $1 / \delta ,$ , then, because the regret-to-PAC reduction of Kaufmann et al. [2021] only requires constant success probability, the algorithm could still be used to obtain an optimal sample complexity for best-policy identification in some regimes. However, to the best of our knowledge even for $\begin{array} { r } { \frac { 1 } { \delta } = { O } \left( 1 \right) } \end{array}$ all previous work still contains suboptimal logarithmic factors in $S , A ,$ , H, or $K$ , so this is not possible from existing guarantees.) Of particular interest is the result of Zhang et al. [2025], because despite the suboptimal multiplicative logarithmic factors in its leading order $\sqrt { K }$ regret term, it does not sufer from additional additive “burn-in” regret terms, and so combining with the regret-to-PAC conversion of Kaufmann et al. [2021] would yield a best-policy identification sample complexity of $\begin{array} { r } { O \left( \frac { H ^ { 3 } S A L _ { \delta } } { \varepsilon ^ { 2 } } \left( \log ( e S A H / \varepsilon ) \right) ^ { 5 } + \frac { H ^ { 2 } L _ { \delta } ^ { 2 } } { \varepsilon ^ { 2 } } \right) } \end{array}$ (recall $L _ { \delta } = \log ( 2 / \delta ) )$ , which can be superior to our Corollary 6.2 in regimes of suficiently large ε.

Wagenmaker et al. [2022] and Tirinzoni et al. [2023] study instance-dependent complexity of best-policy identification, with dependencies on suboptimality gaps and state visitation probabilities, rather than the minimax complexity for fixed S, A, H, ε, δ. Kone and Jamieson [2026] study the complexity of exact identification of an optimal policy in the regime of small δ.

## D.2 Algorithm and Proof of Corollary 6.2

Next we present the algorithm which is the subject of Corollary 6.2, as well as the proof. In Algorithm 2, $C , C ^ { \prime }$ are absolute constants which are defined in the proof below in terms of the constants for which the regret bound in Theorem 6.1 holds.

```latex
Algorithm 2: MORE-BPI
$\begin{array} { r } { m = \left\lceil \log _ { 2 } ( \frac { 2 } { \delta } ) \right\rceil , K _ { 0 } = \left\lceil \frac { C S A H ^ { 3 } } { \varepsilon ^ { 2 } } + C ^ { \prime } H ^ { 3 } S ^ { 3 } A ( \log ( e H S A ) ) ^ { 4 } \right\rceil , M = \left\lceil \frac { 9 H ^ { 2 } \log ( 4 m / \delta ) } { 2 \varepsilon ^ { 2 } } \right\rceil } \end{array}$
for $i = 1 , \ldots , m$ do
Independently run $K _ { 0 }$ episodes of Algorithm 1, generating policies $\pi ^ { ( i ) , 1 } , \ldots , \pi ^ { ( i ) , K _ { 0 } }$
Independently sample $J _ { i } \sim$ Uniform $\{ 1 , \ldots , K _ { 0 } \}$ and let $\bar { \pi } ^ { ( i ) } = \pi ^ { ( i ) , J _ { i } }$
end
for $i = 1 , \ldots , m$ do $/ /$ estimate the value of policy $\bar { \pi } ^ { \left( i \right) }$
Follow policy $\bar { \pi } ^ { \left( i \right) }$ for M episodes and record cumulative returns within each episode
$R _ { i , 1 } , \ldots , R _ { i } ,$ M
$\begin{array} { r } { \widehat { V } ^ { i } \gets \frac { 1 } { M } \sum _ { j = 1 } ^ { M } R _ { i , j } } \end{array}$
end
Let $I \in \mathrm { a r g m a x } _ { i } { \widehat { V } } ^ { i }$ and return $\bar { \pi } ^ { ( I ) }$
```

Algorithm 2 has two phases. First it generates m candidate policies $\bar { \pi } ^ { ( 1 ) } , \ldots , \bar { \pi } ^ { ( m ) }$ . For each $i \in [ m ]$ this is done by running $K _ { 0 }$ episodes of Algorithm 1, where $K _ { 0 }$ is chosen so that the expected average regret over $K _ { 0 }$ episodes is at most $\varepsilon / 6$ . As shown in the proof, this implies via Markov’s inequality that if we sample an episode $J _ { i }$ uniformly at random from $[ K _ { 0 } ]$ and use the policy $\pi ^ { ( i ) , J _ { i } }$ from that episode, then $\pi ^ { ( i ) , J _ { i } }$ (which is the definition of $\bar { \pi } ^ { ( i ) } )$ will be $\varepsilon / 3 \mathrm { - o p t i m a l }$ with constant probability.

The success probability is then amplified by estimating the value of all $m = O ( \log _ { 2 } ( 2 / \delta ) )$ policies and choosing the (estimated) best one. This amplification step is standard and is basically identical to [Kaufmann et al., 2021, Proposition 12], but we provide the short argument for completeness.

Lemma D.1. Let $\begin{array} { r } { X > 1 . \ I f K \geq e ^ { 4 } X ( \log ( e ^ { 4 } X ) ) ^ { 4 } , t h e n \ \frac { K } { ( \log K ) ^ { 4 } } \geq X . } \end{array}$

Proof. Define the function $\begin{array} { r } { g ( x ) = \frac { x } { ( \log x ) ^ { 4 } } } \end{array}$ . This function is non-decreasing on $[ e ^ { 4 } , \infty )$ because

$$
g ^ { \prime } ( x ) = { \frac { 1 } { ( \log x ) ^ { 4 } } } \left( 1 - { \frac { 4 } { \log x } } \right) \geq 0 .
$$

Letting $x _ { 0 } = e ^ { 4 } X ( \log ( e ^ { 4 } X ) ) ^ { 4 } \geq e ^ { 4 }$ , since $x _ { 0 } \geq e ^ { 4 }$ it sufices to show that $g ( x _ { 0 } ) \geq X$ , because then for all $K \geq x _ { 0 }$ we have

$$
{ \frac { K } { ( \log K ) ^ { 4 } } } = g ( K ) \geq g ( x _ { 0 } ) \geq X .
$$

To do $\mathrm { s o } ,$ we calculate

$$
g ( x _ { 0 } ) = { \frac { e ^ { 4 } X ( \log ( e ^ { 4 } X ) ) ^ { 4 } } { ( \log x _ { 0 } ) ^ { 4 } } } \geq { \frac { e ^ { 4 } X ( \log ( e ^ { 4 } X ) ) ^ { 4 } } { e ^ { 4 } ( \log ( e ^ { 4 } X ) ) ^ { 4 } } } = X
$$

where the inequality step comes from the bound

$$
\log x _ { 0 } = \log ( e ^ { 4 } X ) + 4 \log \log ( e ^ { 4 } X ) \leq \log ( e ^ { 4 } X ) + { \frac { 4 } { e } } \log ( e ^ { 4 } X ) \leq e \log ( e ^ { 4 } X )
$$

using the standard fact that log $\begin{array} { r } { t \leq \frac { t } { e } \ ( \mathrm { f o r } \ t > 0 ) } \end{array}$

Proof of Corollary 6.2. From Theorem 6.1, there exist $C _ { 1 } , C _ { 2 } \geq 1$ such that

$$
\begin{array} { r } { \mathbb { E } [ \mathrm { R e g } ( K ) ] \leq \sqrt { C _ { 1 } H ^ { 3 } S A K } + C _ { 2 } H ^ { 3 } S ^ { 2 } A ( \log ( K ) ) ^ { 2 } . } \end{array}
$$

For the first term to dominate, we have the implications

$$
\begin{array} { r l } & { \qquad \sqrt { C _ { 1 } H ^ { 3 } S A K } \ge C _ { 2 } H ^ { 3 } S ^ { 2 } A ( \log ( K ) ) ^ { 2 } } \\ & { \Longleftrightarrow K \ge \frac { C _ { 2 } ^ { 2 } H ^ { 6 } S ^ { 4 } A ^ { 2 } ( \log ( K ) ) ^ { 4 } } { C _ { 1 } H ^ { 3 } S A } = \frac { C _ { 2 } ^ { 2 } } { C _ { 1 } } H ^ { 3 } S ^ { 3 } A ( \log ( K ) ) ^ { 4 } } \\ & { \Longleftrightarrow K \ge e ^ { 4 } \frac { C _ { 2 } ^ { 2 } } { C _ { 1 } } H ^ { 3 } S ^ { 3 } A \left( \log \left( e ^ { 4 } \frac { C _ { 2 } ^ { 2 } } { C _ { 1 } } H ^ { 3 } S ^ { 3 } A \right) \right) ^ { 4 } } \\ & { \Longleftrightarrow K \ge C _ { 4 } H ^ { 3 } S ^ { 3 } A ( \log ( e H S A ) ) ^ { 4 } } \end{array}
$$

where the penultimate implication is by Lemma D.1, and the final implication is by choosing a suficiently large constant $C _ { 4 }$ . Hence as long as $K \geq C _ { 4 } H ^ { 3 } S ^ { 3 } A ( \log ( e H S A ) ) ^ { 4 }$ , we have

$$
\mathbb { E } [ \mathrm { R e g } ( K ) ] \le 2 \sqrt { C _ { 1 } H ^ { 3 } S A K } = \sqrt { C _ { 3 } H ^ { 3 } S A K }
$$

where $C _ { 3 } = 4 C _ { 1 }$

Next, for any $i \in [ m ]$ , the expected suboptimality of the candidate policy $\bar { \pi } ^ { ( i ) } = \pi ^ { ( i ) , J _ { i } }$ is bounded as

$$
\mathbb { E } \left[ V _ { 1 } ^ { \star } ( \mu ) - V _ { 1 } ^ { \pi ^ { ( s ) } } ( \mu ) \right] = \mathbb { E } \left[ \frac { 1 } { K _ { 0 } } \sum _ { k = 1 } ^ { K _ { 0 } } \left( V _ { 1 } ^ { \star } ( \mu ) - V _ { 1 } ^ { \pi ^ { ( s ) , k } } ( \mu ) \right) \right] = \frac { \mathbb { E } [ \mathrm { R e g } ( K _ { 0 } ) ] } { K _ { 0 } } \leq \frac { \sqrt { C _ { 3 } H ^ { 3 } S A K _ { 0 } } } { K _ { 0 } } = \sqrt { \frac { C _ { 3 } H ^ { 3 } S A } { K _ { 0 } } } .
$$

Here the first equality is because the conditional expectation (on the transitions used in forming the policies $\pi ^ { ( i ) , 1 } , \ldots , \pi ^ { ( i ) , K _ { 0 } } )$ of $\bar { V } _ { 1 } ^ { \bar { \pi } ^ { ( i ) } } ( \mu )$ is equal to $\begin{array} { r } { \frac { 1 } { K _ { 0 } } \sum _ { k = 1 } ^ { K _ { 0 } } \dot { V _ { 1 } ^ { \pi ^ { ( i ) , k } } } ( \mu ) } \end{array}$ . The second equality uses the fact that each $\pi ^ { ( i ) , k }$ is chosen before $s _ { 1 } ^ { k }$ is sampled from $\mu ,$ so $\mathbb { E } \left[ V _ { 1 } ^ { \star } ( \mu ) - V _ { 1 } ^ { \pi ^ { ( i ) , k } } ( \mu ) \right] = \mathbb { E } \left[ V _ { 1 } ^ { \star } ( s _ { 1 } ^ { k } ) - V _ { 1 } ^ { \pi ^ { ( i ) , k } } ( s _ { 1 } ^ { k } ) \right]$ . By Markov’s inequality (since $V _ { 1 } ^ { \star } ( \mu ) - V _ { 1 } ^ { \bar { \pi } ^ { ( i ) } } ( \mu ) \geq 0$ almost surely),

$$
\mathbb { P } \left( V _ { 1 } ^ { \star } ( \mu ) - V _ { 1 } ^ { \bar { \pi } ^ { ( i ) } } ( \mu ) \geq \frac { \varepsilon } { 3 } \right) \leq \frac { \mathbb { E } \left[ V _ { 1 } ^ { \star } ( \mu ) - V _ { 1 } ^ { \bar { \pi } ^ { ( i ) } } ( \mu ) \right] } { \varepsilon / 3 } \leq \sqrt { \frac { 9 C _ { 3 } H ^ { 3 } S A } { K _ { 0 } \varepsilon ^ { 2 } } } \leq \frac { 1 } { 2 }
$$

where the final inequality holds if $\begin{array} { r } { K _ { 0 } \ge \frac { 3 6 C _ { 3 } S A H ^ { 3 } } { \varepsilon ^ { 2 } } } \end{array}$ , which is clearly true for the choice

$$
K _ { 0 } = \left\lceil \frac { 3 6 C _ { 3 } S A H ^ { 3 } } { \varepsilon ^ { 2 } } + C _ { 4 } H ^ { 3 } S ^ { 3 } A ( \log ( e H S A ) ) ^ { 4 } \right\rceil ,
$$

and so we let $C = 3 6 C _ { 3 }$ and $C ^ { \prime } = C _ { 4 }$

Consequently, applying this fact to the $\bar { \pi } ^ { ( 1 ) } , \ldots , \bar { \pi } ^ { ( m ) }$ generated by Algorithm 2, and noting their independence, we can bound the probability of all these policies being $\varepsilon / 3$ -suboptimal as

$$
\mathbb { P } \left( \forall i , \ V _ { 1 } ^ { \star } ( \mu ) - V _ { 1 } ^ { \bar { \pi } ^ { ( i ) } } ( \mu ) \geq \frac { \varepsilon } { 3 } \right) \leq 2 ^ { - m } \leq \frac { \delta } { 2 } .
$$

Letting $I ^ { \star } \in \mathrm { a r g m a x } _ { i } V _ { 1 } ^ { \bar { \pi } ^ { ( i ) } } ( \mu )$ , we consequently have

$$
\mathbb { P } \left( V _ { 1 } ^ { \bar { \pi } ^ { ( I ^ { \star } ) } } ( \mu ) \geq V _ { 1 } ^ { \star } ( \mu ) - \frac { \varepsilon } { 3 } \right) \geq 1 - \frac { \delta } { 2 } .
$$

For each fixed $i \in [ m ] , j \in [ M ]$ , letting D be the sigma-algebra generated by the candidate policies $\bar { \pi } ^ { ( 1 ) } , \ldots , \bar { \pi } ^ { ( m ) }$ , since $\mathbb { E } [ R _ { i , j } \mid \mathcal { D } ] = V _ { 1 } ^ { \bar { \pi } ^ { ( i ) } } ( \mu )$ and $0 \le R _ { i , j } \le H$ almost surely, by (conditional) Hoefding’s inequality (using conditional independence from the data used to construct the $\bar { \pi } ^ { ( i ) } )$ and the union bound we have that

$$
\mathbb { P } \left( \forall i , \ \left| { \widehat { V } } ^ { i } - V _ { 1 } ^ { \pi ^ { ( i ) } } ( \mu ) \right| \leq { \frac { \varepsilon } { 3 } } \right) \geq 1 - { \frac { \delta } { 2 } }
$$

by the choice of $\begin{array} { r } { M = \frac { 9 H ^ { 2 } \log ( 4 m / \delta ) } { 2 \varepsilon ^ { 2 } } } \end{array}$

On the intersection of both of the above good events, which has probability at least $1 - \delta$ , we have that

$$
{ V _ { 1 } ^ { \bar { \pi } ^ { ( I ) } } ( \mu ) \geq \widehat { V } ^ { I } - \frac { \varepsilon } { 3 } \geq \widehat { V } ^ { I ^ { \star } } - \frac { \varepsilon } { 3 } \geq V _ { 1 } ^ { \bar { \pi } ^ { ( I ^ { \star } ) } } ( \mu ) - \frac { 2 \varepsilon } { 3 } \geq V _ { 1 } ^ { \star } ( \mu ) - \varepsilon } .
$$

Finally, we note that for all $\delta \in \mathsf { \Gamma } ( 0 , 1 )$ , from the standard bound $\log ( 2 / \delta ) \ \leq \ { \textstyle \frac { 2 } { \delta } } - 1$ we have that $\log ( 2 / \delta ) \leq { \frac { 2 } { \delta } }$ , so $\begin{array} { r } { M = O \left( \frac { H ^ { 2 } \log ( 2 / \delta ) } { \varepsilon ^ { 2 } } \right) } \end{array}$ The bound on the number of episodes is thus m $K _ { 0 } + m M \ \leq$ $\begin{array} { r } { O \left( \frac { H ^ { 3 } S A L _ { \delta } } { \varepsilon ^ { 2 } } + \frac { H ^ { 2 } L _ { \delta } ^ { 2 } } { \varepsilon ^ { 2 } } + H ^ { 3 } S ^ { 3 } A ( \log ( e H S A ) ) ^ { 4 } L _ { \delta } \right) } \end{array}$ where $L _ { \delta } = \log ( 2 / \delta )$ □